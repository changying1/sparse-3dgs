"""Default-off, budget-conserving I2 active reallocation interface.

This module accepts only a fail-closed D0 plan backed by a separately hashed
static-deletion counterfactual.  It never consumes legacy FSGS RGG UIDs or
checkpoint metadata; all identity changes go through the current A0 lifecycle.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import torch


ACTIVE_EVENT_FORMAT = "i2_active_event_v1"
ACTIVE_EVENT_RECEIVER_V2_FORMAT = "i2_active_event_receiver_v2_v1"
ACTIVE_EVENT_V2_SELECTION_ANCHOR_CLONE_FORMAT = (
    "i2_active_event_v2_selection_anchor_clone_v1"
)
RELEASE_ONLY_EVENT_FORMAT = "i2_release_only_event_v1"
BIRTH_ONLY_EVENT_FORMAT = "i2_birth_only_event_rd_d3_v1"
COUNTERFACTUAL_FORMAT = "i2_static_deletion_counterfactual_v1"
JOINT_COUNTERFACTUAL_FORMAT = "i2_joint_deletion_counterfactual_v1"
ACTIVE_PLAN_FORMAT = "i2_d0_read_only_plan_v2"
RECEIVER_V2_PLAN_FORMAT = "i2_d1_receiver_v2_plan_v1"
V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT = (
    "i2_d1_receiver_v2_selection_anchor_clone_plan_v1"
)
RECEIVER_V2_EVIDENCE_FORMAT = "i2_receiver_v2_geometry_evidence_v1"
RD_V1_PLAN_FORMAT = "i2_rd_v1_active_plan"
ACTIVE_EVENT_RD_V1_FORMAT = "i2_active_event_rd_v1"
V2_SELECTION_ANCHOR_CLONE_METHOD = "receiver_v2_selection_anchor_clone_v1"
RECEIVER_V2_MIN_DISPLACEMENT = 1e-8
RECEIVER_V2_DISPLACEMENT_REL_TOL = 1e-5
RECEIVER_V2_DISPLACEMENT_ABS_TOL = 1e-7


class ActivePilotError(ValueError):
    pass


def receiver_v2_displacement_contract(anchor_xyz, target_xyz, recorded_norm=None):
    """Check serialized Receiver-v2 coordinates against the loader contract."""
    try:
        anchor = [float(value) for value in anchor_xyz]
        target = [float(value) for value in target_xyz]
    except (TypeError, ValueError):
        return False, "invalid_effective_xyz", None, None
    if (
        len(anchor) != 3 or len(target) != 3
        or not all(math.isfinite(value) for value in anchor + target)
    ):
        return False, "invalid_effective_xyz", None, None
    displacement = [new - old for old, new in zip(anchor, target)]
    norm = math.sqrt(sum(value * value for value in displacement))
    if norm <= RECEIVER_V2_MIN_DISPLACEMENT:
        return False, "effective_xyz_overlap", displacement, norm
    if recorded_norm is not None:
        try:
            recorded = float(recorded_norm)
        except (TypeError, ValueError):
            return False, "invalid_displacement_norm", displacement, norm
        if not math.isfinite(recorded) or recorded < 0:
            return False, "invalid_displacement_norm", displacement, norm
        if not math.isclose(
            norm, recorded,
            rel_tol=RECEIVER_V2_DISPLACEMENT_REL_TOL,
            abs_tol=RECEIVER_V2_DISPLACEMENT_ABS_TOL,
        ):
            return False, "displacement_norm_mismatch", displacement, norm
    return True, None, displacement, norm


def select_receiver_v2_candidates(candidates, *, budget, spatial_dedup_radius):
    """Apply the exact XYZ contract before frozen ranking and spatial NMS."""
    valid = []
    rejected = []
    for row in candidates:
        ok, reason, _, _ = receiver_v2_displacement_contract(
            row.get("anchor_xyz", []), row.get("new_xyz", []),
            row.get("displacement_norm"),
        )
        if not ok:
            rejected.append({"candidate_id": row.get("candidate_id"), "reason": reason})
            continue
        valid.append(row)
    valid.sort(key=lambda row: (
        -row["photometric_alignment"], -row["demand_score"], row["candidate_id"]
    ))
    selected = []
    for evidence in valid:
        point = evidence["new_xyz"]
        if any(math.dist(point, old["new_xyz"]) < spatial_dedup_radius for old in selected):
            rejected.append({
                "candidate_id": evidence["candidate_id"],
                "reason": "v2_spatial_nms",
            })
            continue
        selected.append(evidence)
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise ActivePilotError(
            f"NO-GO: only {len(selected)} geometrically valid Receiver-v2 "
            f"candidates for requested K={budget}"
        )
    return selected, rejected


def add_i2_active_pilot_arguments(parser) -> None:
    parser.add_argument("--enable_i2_active_pilot", action="store_true", default=False)
    parser.add_argument("--enable_i2_release_only", action="store_true", default=False)
    parser.add_argument("--enable_i2_birth_only", action="store_true", default=False)
    parser.add_argument("--i2_active_plan", type=str, default="")
    parser.add_argument("--i2_active_event_log", type=str, default="")
    parser.add_argument("--i2_adaptive_decision", type=str, default="")
    parser.add_argument("--i2_active_budget", type=int, default=20)


def validate_i2_active_configuration(
    enabled, plan_path, event_log, budget, *, release_only=False, birth_only=False
) -> None:
    if sum(bool(value) for value in (enabled, release_only, birth_only)) > 1:
        raise ActivePilotError("Active reallocation, release-only and birth-only modes are mutually exclusive")
    if not (enabled or release_only or birth_only):
        return
    if isinstance(budget, bool) or int(budget) <= 0:
        raise ActivePilotError("Active pilot K must be a positive integer")
    if not str(plan_path).strip() or not str(event_log).strip():
        raise ActivePilotError("Active pilot requires a plan and independent event log")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(base: Path, source, label="Artifact") -> Path:
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise ActivePilotError(f"{label} path/SHA256 is missing")
    path = Path(source["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != str(source["sha256"]).lower():
        raise ActivePilotError(f"{label} is missing or has a SHA256 mismatch")
    return path


def _json_artifact(base: Path, source, label):
    path = _artifact(base, source, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivePilotError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ActivePilotError(f"{label} root must be an object")
    return path, value


def _finite_number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActivePilotError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ActivePilotError(f"{label} must be a finite number")
    return result


def _json_equivalent(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _load_rd_v1_plan(path: Path, plan: dict, *, requested_budget: int,
                     source_checkpoint=None) -> dict:
    """Strict adapter for RD-v1; it does not synthesize legacy safety reports."""
    budget = int(requested_budget)
    if (
        plan.get("format") != RD_V1_PLAN_FORMAT
        or plan.get("policy_version") != "RD-v1"
        or plan.get("uses_test_views") is not False
        or plan.get("selected_k") != budget
        or budget <= 0
        or plan.get("performance_safety_reports") is not None
    ):
        raise ActivePilotError("RD-v1 Plan identity or budget is invalid")
    source = plan.get("source")
    if not isinstance(source, dict) or type(source.get("iteration")) is not int:
        raise ActivePilotError("RD-v1 source binding is invalid")
    checkpoint_path = _artifact(path.parent, source.get("checkpoint"), "RD-v1 source checkpoint")
    if source_checkpoint is not None and checkpoint_path != Path(source_checkpoint).resolve():
        raise ActivePilotError("RD-v1 Plan does not match the runtime source checkpoint")
    manifest_path = _artifact(
        path.parent, plan.get("training_view_manifest"), "RD-v1 training view manifest"
    )
    candidate_table = _artifact(
        path.parent, plan.get("receiver_candidate_table"), "RD-v1 receiver candidate table"
    )
    receiver_path, receiver_audit = _json_artifact(
        path.parent, plan.get("receiver_audit"), "RD-v1 Receiver audit"
    )
    donor_path, donor_audit = _json_artifact(
        path.parent, plan.get("donor_audit"), "RD-v1 Donor audit"
    )
    if (
        receiver_audit.get("format") != "i2_receiver_v2_audit_all_rd_v1"
        or receiver_audit.get("test_views_used") is not False
        or receiver_audit.get("coverage", {}).get("complete") is not True
        or receiver_audit.get("receiver_budget", {}).get("complete_sequence_audited") is not True
        or donor_audit.get("format") != "i2_rd_v1_donor_capacity_v1"
        or donor_audit.get("test_views_used") is not False
        or donor_audit.get("coverage_complete") is not True
        or donor_audit.get("performance_gates_used") is not False
        or donor_audit.get("render_counterfactuals_used") is not False
        or receiver_audit.get("scene") != plan.get("scene")
        or donor_audit.get("scene") != plan.get("scene")
        or receiver_audit.get("source_checkpoint_sha256") != _sha256(checkpoint_path)
        or donor_audit.get("source_checkpoint_sha256") != _sha256(checkpoint_path)
    ):
        raise ActivePilotError("RD-v1 evidence coverage/scope is invalid")
    receiver_manifest_path, receiver_manifest = _json_artifact(
        receiver_path.parent, receiver_audit.get("receiver_manifest"),
        "RD-v1 Receiver manifest",
    )
    manifest_candidate = receiver_manifest.get("receiver_candidates")
    manifest_candidate_path = Path(
        manifest_candidate.get("path", "") if isinstance(manifest_candidate, dict) else ""
    )
    if not manifest_candidate_path.is_absolute():
        manifest_candidate_path = receiver_manifest_path.parent / manifest_candidate_path
    if (
        not isinstance(manifest_candidate, dict)
        or str(manifest_candidate.get("sha256", "")).lower() != _sha256(candidate_table)
        or manifest_candidate_path.resolve() != candidate_table
    ):
        raise ActivePilotError("RD-v1 Receiver candidate-table lineage is invalid")
    receivers, donors, matches = plan.get("receivers"), plan.get("donors"), plan.get("matches")
    if not all(isinstance(rows, list) and len(rows) == budget for rows in (receivers, donors, matches)):
        raise ActivePilotError("RD-v1 Plan must contain exactly K receivers, donors and matches")
    coverage = plan.get("coverage", {})
    r = receiver_audit.get("receiver_budget", {}).get("receiver_capacity")
    d = donor_audit.get("donor_capacity")
    if (
        type(r) is not int or type(d) is not int or budget != min(r, d)
        or coverage.get("receiver_total") != r
        or coverage.get("donor_total") != d
        or coverage.get("complete") is not True
    ):
        raise ActivePilotError("RD-v1 Plan violates K=min(R,D) or coverage totals")
    donor_uids = [int(row.get("uid", -1)) for row in donors]
    receiver_ids = [str(row.get("candidate_id", "")) for row in receivers]
    if len(set(donor_uids)) != budget or len(set(receiver_ids)) != budget or not all(receiver_ids):
        raise ActivePilotError("RD-v1 selected identities must be unique and stable")
    donor_prefix = [int(row["uid"]) for row in donor_audit.get("ranked_eligible", [])[:budget]]
    receiver_prefix = receiver_audit.get("accepted_order", [])[:budget]
    if (
        donor_uids != donor_prefix or receiver_ids != receiver_prefix
        or coverage.get("donor_selected_order") != donor_uids
        or coverage.get("receiver_selected_order") != receiver_ids
    ):
        raise ActivePilotError("RD-v1 selected rows are not the audited deterministic prefixes")
    match_uids = [int(row.get("donor_uid", -1)) for row in matches]
    match_receivers = [str(row.get("receiver_candidate_id", "")) for row in matches]
    if (
        match_uids != donor_uids or match_receivers != receiver_ids
        or any(row.get("released_units") != 1 or row.get("received_units") != 1 for row in matches)
        or plan.get("budget_audit") != {
            "released_units": budget, "received_units": budget, "conserved": True
        }
    ):
        raise ActivePilotError("RD-v1 Plan does not conserve K releases and births")
    targets, anchors, opacities, scales = [], [], [], []
    for row in receivers:
        if row.get("geometry_valid") is not True or row.get("fail_closed") is not False:
            raise ActivePilotError("RD-v1 Receiver geometry is invalid")
        checks = row.get("cross_view_checks")
        if not isinstance(checks, list) or not checks or not all(
            isinstance(check, dict) and check.get("safe") is True for check in checks
        ):
            raise ActivePilotError("RD-v1 Receiver training-view geometry checks are incomplete")
        anchor = [_finite_number(v, "RD-v1 anchor XYZ") for v in row.get("anchor_xyz", [])]
        target = [_finite_number(v, "RD-v1 target XYZ") for v in row.get("new_xyz", [])]
        ok, _, _, _ = receiver_v2_displacement_contract(
            anchor, target, row.get("displacement_norm")
        )
        if not ok:
            raise ActivePilotError("RD-v1 Receiver displacement contract failed")
        anchors.append(int(row["anchor_uid"]))
        targets.append(target)
        opacity = _finite_number(row.get("activated_opacity"), "RD-v1 opacity", nonnegative=True)
        scale = [_finite_number(v, "RD-v1 scale", nonnegative=True) for v in row.get("activated_scale", [])]
        if not (0 < opacity <= 0.15) or len(scale) != 3 or not all(v > 0 for v in scale):
            raise ActivePilotError("RD-v1 Receiver opacity/scale is invalid")
        opacities.append(opacity); scales.append(scale)
    if set(anchors) & set(donor_uids):
        raise ActivePilotError("RD-v1 donor set contains a Receiver anchor UID")
    plan.update({
        "_plan_path": str(path), "_plan_sha256": _sha256(path),
        "_counterfactual_artifact": None, "_joint_counterfactual_artifact": None,
        "_receiver_candidate_table": str(candidate_table),
        "_donor_uids": donor_uids, "_receiver_xyz": targets,
        "_anchor_uids": anchors, "_receiver_candidate_ids": receiver_ids,
        "_receiver_v2": True, "_receiver_v2_selection_anchor_clone": False,
        "_anchor_clone_original_v2_plan": None,
        "_anchor_clone_view_manifest": str(manifest_path),
        "_receiver_v2_artifact": str(receiver_path),
        "_receiver_v2_rows": receivers,
        "_receiver_opacities": opacities, "_receiver_scales": scales,
        "_rd_v1": True, "_rd_v1_donor_audit": str(donor_path),
        "input_audit": {"source_iteration": source["iteration"]},
    })
    return plan


def load_active_plan(path, *, requested_budget: int, source_checkpoint=None) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ActivePilotError(f"Active plan does not exist: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivePilotError("Active plan is not valid JSON") from exc
    budget = int(requested_budget)
    plan_format = plan.get("format")
    if plan_format == RD_V1_PLAN_FORMAT:
        return _load_rd_v1_plan(
            path, plan, requested_budget=budget, source_checkpoint=source_checkpoint
        )
    if plan_format not in {
        ACTIVE_PLAN_FORMAT,
        RECEIVER_V2_PLAN_FORMAT,
        V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT,
    }:
        raise ActivePilotError("Active pilot requires a supported, regenerated plan")
    anchor_clone = plan_format == V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT
    anchor_clone_block = None
    anchor_clone_original_plan = None
    anchor_clone_view_manifest = None
    if anchor_clone:
        anchor_clone_block = plan.get("receiver_v2_selection_anchor_clone")
        if (
            not isinstance(anchor_clone_block, dict)
            or anchor_clone_block.get("method_version")
            != V2_SELECTION_ANCHOR_CLONE_METHOD
            or anchor_clone_block.get("test_views_used") is not False
        ):
            raise ActivePilotError(
                "Receiver-v2 selection Anchor-Clone method binding is invalid"
            )
        original_path, original_payload = _json_artifact(
            path.parent,
            anchor_clone_block.get("original_receiver_v2_plan"),
            "Original Receiver-v2 Plan",
        )
        if original_payload.get("format") != RECEIVER_V2_PLAN_FORMAT:
            raise ActivePilotError("Anchor-Clone source is not a Receiver-v2 plan")
        expected = _json_equivalent(original_payload)
        expected["format"] = V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT
        expected["method_version"] = V2_SELECTION_ANCHOR_CLONE_METHOD
        expected["receiver_v2_selection_anchor_clone"] = _json_equivalent(
            anchor_clone_block
        )
        if _json_equivalent(plan) != expected:
            raise ActivePilotError(
                "Anchor-Clone plan differs from the frozen Receiver-v2 plan beyond "
                "its isolated format/method binding"
            )
        evidence_path = _artifact(
            path.parent,
            anchor_clone_block.get("receiver_v2_geometry_evidence"),
            "Frozen Receiver-v2 Geometry Evidence",
        )
        original_evidence = original_payload.get("receiver_v2", {}).get("artifact")
        if (
            not isinstance(original_evidence, dict)
            or str(original_evidence.get("sha256", "")).lower()
            != _sha256(evidence_path)
        ):
            raise ActivePilotError(
                "Anchor-Clone geometry evidence differs from the frozen Receiver-v2 plan"
            )
        anchor_clone_view_manifest = _artifact(
            path.parent,
            anchor_clone_block.get("training_view_manifest"),
            "Frozen Training View Manifest",
        )
        expected_manifest_sha = original_payload.get("input_audit", {}).get(
            "receiver_protocol", {}
        ).get("view_manifest_sha256")
        if _sha256(anchor_clone_view_manifest) != expected_manifest_sha:
            raise ActivePilotError(
                "Anchor-Clone training view manifest differs from Receiver-v2"
            )
        anchor_clone_original_plan = original_path
    if plan.get("mode") != "read_only_dry_run" or plan.get("strategy") != "b1_only_exploratory":
        raise ActivePilotError("First active pilot requires a B1-only read-only D0 plan")
    if int(plan.get("configuration", {}).get("budget_k", -1)) != budget:
        raise ActivePilotError("Active plan budget does not match requested K")
    readiness = plan.get("safety_readiness", {})
    if readiness.get("b1_only_exploratory") is not True:
        raise ActivePilotError("B1-only safety readiness is false")
    if plan.get("budget_audit", {}).get("conserved") is not True:
        raise ActivePilotError("D0 plan does not conserve the requested budget")
    checkpoint_sha = plan.get("input_audit", {}).get("source_checkpoint_sha256")
    if source_checkpoint is not None:
        checkpoint_path = Path(source_checkpoint).resolve()
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint_sha:
            raise ActivePilotError("Active plan does not match the actual source checkpoint")
    donors = plan.get("donors", {}).get("selected")
    receivers = plan.get("receivers", {}).get("rows")
    matches = plan.get("matches")
    if not all(isinstance(value, list) for value in (donors, receivers, matches)):
        raise ActivePilotError("D0 donor/receiver/match tables are malformed")
    if len(donors) != budget or len(matches) != budget:
        raise ActivePilotError("D0 plan does not contain exactly K donors and matches")
    donor_uids = []
    for donor in donors:
        if donor.get("birth_type") not in {"clone", "split"}:
            raise ActivePilotError("Root/unknown Gaussians are protected from active release")
        uid = int(donor["uid"])
        if uid in donor_uids:
            raise ActivePilotError("Active donor UIDs must be unique")
        donor_uids.append(uid)
    selected_uid_set = set(donor_uids)
    match_uids = []
    receiver_use = Counter()
    for match in matches:
        if int(match.get("released_units", -1)) != 1 or int(match.get("received_units", -1)) != 1:
            raise ActivePilotError("Each active match must conserve one Gaussian")
        try:
            match_uid = int(match["donor_uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivePilotError("Active match donor UID is invalid") from exc
        match_uids.append(match_uid)
        receiver_use[str(match.get("receiver_candidate_id"))] += 1
    if len(set(match_uids)) != budget or set(match_uids) != selected_uid_set:
        raise ActivePilotError("Match donor UIDs must equal selected donors exactly once")

    input_audit = plan.get("input_audit", {})
    source_iteration = input_audit.get("source_iteration")
    if type(source_iteration) is not int or source_iteration < 0:
        raise ActivePilotError("Active plan source iteration is invalid")
    receiver_ref = input_audit.get("receiver_candidate_table")
    receiver_path = _artifact(path.parent, receiver_ref, "Receiver Candidate Table")
    try:
        with receiver_path.open(encoding="utf-8", newline="") as handle:
            original_receiver_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ActivePilotError("Receiver Candidate Table is malformed") from exc
    protocol = input_audit.get("receiver_protocol")
    try:
        from utils.i2_d0_dry_run import validate_receivers
        receiver_manifest = {
            "_manifest_path": str(path),
            "source_checkpoint": {"sha256": checkpoint_sha},
            "receiver_protocol": protocol,
        }
        verified_receivers = validate_receivers(original_receiver_rows, receiver_manifest)["rows"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ActivePilotError("Receiver Candidate Table/evidence binding is invalid") from exc
    verified_by_id = {row["candidate_id"]: _json_equivalent(row) for row in verified_receivers}
    if len(verified_by_id) != len(verified_receivers):
        raise ActivePilotError("Receiver Candidate Table contains duplicate IDs")
    embedded_by_id = {}
    for row in receivers:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in embedded_by_id:
            raise ActivePilotError("Embedded receiver rows contain duplicate/invalid IDs")
        embedded_by_id[candidate_id] = row
    matched_ids = set(receiver_use)
    if not matched_ids.issubset(embedded_by_id) or not matched_ids.issubset(verified_by_id):
        raise ActivePilotError("Active match references an unknown receiver")
    targets, anchor_uids = [], []
    match_receiver_ids = []
    for candidate_id, used in receiver_use.items():
        original = verified_by_id[candidate_id]
        if _json_equivalent(embedded_by_id[candidate_id]) != original:
            raise ActivePilotError("Embedded receiver row differs from verified Candidate Table")
        if used > int(original["requested_units"]):
            raise ActivePilotError("Receiver usage exceeds requested_units")
    for match in matches:
        candidate_id = str(match["receiver_candidate_id"])
        receiver = verified_by_id[candidate_id]
        targets.append([_finite_number(v, "Receiver target XYZ") for v in receiver["target_xyz"]])
        anchor_uid = int(receiver["anchor_uid"])
        if anchor_uid in selected_uid_set:
            raise ActivePilotError("Receiver anchor UID is also a selected donor")
        anchor_uids.append(anchor_uid)
        match_receiver_ids.append(candidate_id)

    receiver_v2 = plan_format == RECEIVER_V2_PLAN_FORMAT
    receiver_v2_selection = receiver_v2 or anchor_clone
    receiver_opacities = None
    receiver_scales = None
    receiver_v2_artifact = None
    receiver_v2_rows = None
    if receiver_v2_selection:
        receiver_v2_block = plan.get("receiver_v2")
        if (
            not isinstance(receiver_v2_block, dict)
            or receiver_v2_block.get("method_version") != "receiver_v2_tangent_jacobian_v1"
        ):
            raise ActivePilotError("Receiver-v2 method version is missing or unsupported")
        receiver_v2_artifact, receiver_v2_payload = _json_artifact(
            path.parent, receiver_v2_block.get("artifact"), "Receiver-v2 Evidence"
        )
        if (
            receiver_v2_payload.get("format") != RECEIVER_V2_EVIDENCE_FORMAT
            or receiver_v2_payload.get("method_version") != "receiver_v2_tangent_jacobian_v1"
            or receiver_v2_payload.get("source_checkpoint_sha256") != checkpoint_sha
            or receiver_v2_payload.get("source_iteration") != source_iteration
            or receiver_v2_payload.get("test_views_used") is not False
            or receiver_v2_payload.get("receiver_candidate_table_sha256")
            != str(receiver_ref.get("sha256", "")).lower()
        ):
            raise ActivePilotError("Receiver-v2 evidence binding is invalid")
        receiver_v2_rows = receiver_v2_payload.get("rows")
        if not isinstance(receiver_v2_rows, list) or len(receiver_v2_rows) != budget:
            raise ActivePilotError("Receiver-v2 evidence must contain exactly K rows")
        v2_by_id = {}
        for row in receiver_v2_rows:
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in v2_by_id:
                raise ActivePilotError("Receiver-v2 evidence has duplicate/invalid candidates")
            if row.get("geometry_valid") is not True or row.get("fail_closed") is not False:
                raise ActivePilotError("Receiver-v2 contains a geometrically invalid candidate")
            checks = row.get("cross_view_checks")
            if not isinstance(checks, list) or len(checks) != len(protocol["training_camera_ids"]):
                raise ActivePilotError("Receiver-v2 cross-view evidence is incomplete")
            if not all(isinstance(check, dict) and check.get("safe") is True for check in checks):
                raise ActivePilotError("Receiver-v2 cross-view safety check failed")
            v2_by_id[candidate_id] = row
        if set(v2_by_id) != set(match_receiver_ids):
            raise ActivePilotError("Receiver-v2 evidence candidates differ from active matches")
        targets, receiver_opacities, receiver_scales = [], [], []
        for candidate_id, expected_anchor in zip(match_receiver_ids, anchor_uids):
            row = v2_by_id[candidate_id]
            if int(row.get("anchor_uid", -1)) != expected_anchor:
                raise ActivePilotError("Receiver-v2 anchor differs from verified candidate")
            old_xyz = [_finite_number(v, "Receiver-v2 anchor XYZ") for v in row.get("anchor_xyz", [])]
            new_xyz = [_finite_number(v, "Receiver-v2 target XYZ") for v in row.get("new_xyz", [])]
            if len(old_xyz) != 3 or len(new_xyz) != 3:
                raise ActivePilotError("Receiver-v2 XYZ must have three coordinates")
            recorded_displacement = _finite_number(
                row.get("displacement_norm"), "Receiver-v2 displacement", nonnegative=True
            )
            displacement_ok, _, _, _ = receiver_v2_displacement_contract(
                old_xyz, new_xyz, recorded_displacement
            )
            if not displacement_ok:
                raise ActivePilotError("Receiver-v2 must be non-overlapping with its anchor")
            opacity = _finite_number(
                row.get("activated_opacity"), "Receiver-v2 opacity", nonnegative=True
            )
            scale = [_finite_number(v, "Receiver-v2 scale", nonnegative=True)
                     for v in row.get("activated_scale", [])]
            if not (0.0 < opacity <= 0.15) or len(scale) != 3 or not all(v > 0 for v in scale):
                raise ActivePilotError("Receiver-v2 opacity/scale is outside conservative bounds")
            if row.get("rotation_source") != "anchor" or row.get("sh_source") != "anchor":
                raise ActivePilotError("Receiver-v2 v1 must preserve anchor rotation and SH")
            targets.append(old_xyz if anchor_clone else new_xyz)
            if not anchor_clone:
                receiver_opacities.append(opacity)
                receiver_scales.append(scale)
    counterfactual = plan.get("donor_counterfactual")
    if not isinstance(counterfactual, dict) or counterfactual.get("format") != COUNTERFACTUAL_FORMAT:
        raise ActivePilotError("A verified static-deletion counterfactual is required")
    artifact_path, original_counterfactual = _json_artifact(
        path.parent, counterfactual.get("artifact"), "Counterfactual artifact"
    )
    if counterfactual.get("source_checkpoint_sha256") != checkpoint_sha:
        raise ActivePilotError("Counterfactual/checkpoint binding is invalid")
    if (
        original_counterfactual.get("format") != COUNTERFACTUAL_FORMAT
        or original_counterfactual.get("source_checkpoint_sha256") != checkpoint_sha
        or original_counterfactual.get("source_iteration") != source_iteration
        or original_counterfactual.get("test_views_used") is not False
    ):
        raise ActivePilotError("Original counterfactual metadata binding is invalid")
    thresholds = original_counterfactual.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ActivePilotError("Counterfactual thresholds are missing")
    mean_limit = _finite_number(thresholds.get("max_mean_abs_rgb_delta"), "RGB threshold", nonnegative=True)
    psnr_limit = _finite_number(thresholds.get("max_psnr_drop_db"), "PSNR threshold", nonnegative=True)
    original_rows = original_counterfactual.get("rows")
    if not isinstance(original_rows, list):
        raise ActivePilotError("Original counterfactual rows are missing")
    training_view_count = original_counterfactual.get("training_view_count")
    if type(training_view_count) is not int or training_view_count <= 0:
        raise ActivePilotError("Counterfactual training-view count is invalid")
    if original_counterfactual.get("audited_count") != len(original_rows):
        raise ActivePilotError("Counterfactual audited_count is inconsistent")
    original_by_uid = {}
    for row in original_rows:
        if not isinstance(row, dict):
            raise ActivePilotError("Original counterfactual row is malformed")
        try:
            uid = int(row["uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivePilotError("Original counterfactual UID is invalid") from exc
        if uid in original_by_uid:
            raise ActivePilotError("Original counterfactual contains duplicate UIDs")
        mean_delta = _finite_number(row.get("max_mean_abs_rgb_delta"), "RGB counterfactual", nonnegative=True)
        max_delta = _finite_number(row.get("max_abs_rgb_delta"), "Maximum RGB counterfactual", nonnegative=True)
        psnr_drop = _finite_number(row.get("max_psnr_drop_db"), "PSNR counterfactual")
        per_view = row.get("per_view")
        if not isinstance(per_view, list) or len(per_view) != training_view_count:
            raise ActivePilotError("Counterfactual per-view rows are incomplete")
        view_ids, view_mean, view_max, view_psnr = set(), [], [], []
        for view in per_view:
            if not isinstance(view, dict):
                raise ActivePilotError("Counterfactual per-view row is malformed")
            view_id = (view.get("training_camera_uid"), str(view.get("training_camera_id", "")))
            if view_id in view_ids or not view_id[1]:
                raise ActivePilotError("Counterfactual training-view identity is invalid")
            view_ids.add(view_id)
            view_mean.append(_finite_number(
                view.get("mean_abs_rgb_delta"), "Per-view RGB counterfactual", nonnegative=True
            ))
            view_max.append(_finite_number(
                view.get("max_abs_rgb_delta"), "Per-view maximum RGB counterfactual", nonnegative=True
            ))
            view_psnr.append(_finite_number(
                view.get("psnr_drop_db"), "Per-view PSNR counterfactual"
            ))
        if mean_delta != max(view_mean) or max_delta != max(view_max) or psnr_drop != max(view_psnr):
            raise ActivePilotError("Counterfactual aggregate values do not match per-view evidence")
        recomputed = mean_delta <= mean_limit and psnr_drop <= psnr_limit
        if type(row.get("safe_to_release")) is not bool or row["safe_to_release"] != recomputed:
            raise ActivePilotError("Original counterfactual safety marker is invalid")
        original_by_uid[uid] = row
    if original_counterfactual.get("safe_count") != sum(
        row["safe_to_release"] is True for row in original_rows
    ):
        raise ActivePilotError("Counterfactual safe_count is inconsistent")
    rows = counterfactual.get("rows")
    if not isinstance(rows, list) or len(rows) != budget:
        raise ActivePilotError("Counterfactual donor rows are missing")
    embedded_counterfactual_uids = []
    for row in rows:
        try:
            uid = int(row["uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivePilotError("Embedded counterfactual UID is invalid") from exc
        embedded_counterfactual_uids.append(uid)
        if uid not in original_by_uid or _json_equivalent(row) != _json_equivalent(original_by_uid[uid]):
            raise ActivePilotError("Embedded counterfactual row differs from verified artifact")
        if original_by_uid[uid]["safe_to_release"] is not True:
            raise ActivePilotError("Selected donor is unsafe in the original counterfactual")
    if len(set(embedded_counterfactual_uids)) != budget or set(embedded_counterfactual_uids) != selected_uid_set:
        raise ActivePilotError("Counterfactual UIDs must equal selected donors exactly once")

    joint = plan.get("joint_deletion_counterfactual")
    if not isinstance(joint, dict) or joint.get("format") != JOINT_COUNTERFACTUAL_FORMAT:
        raise ActivePilotError("A verified joint-deletion counterfactual is required")
    joint_path, joint_payload = _json_artifact(path.parent, joint.get("artifact"), "Joint counterfactual")
    if (
        joint_payload.get("format") != JOINT_COUNTERFACTUAL_FORMAT
        or joint_payload.get("source_checkpoint_sha256") != checkpoint_sha
        or joint_payload.get("source_iteration") != source_iteration
        or joint_payload.get("test_views_used") is not False
        or joint_payload.get("donor_uids") != donor_uids
        or joint_payload.get("safe_to_release") is not True
    ):
        raise ActivePilotError("Joint counterfactual binding/safety is invalid")
    joint_thresholds = joint_payload.get("thresholds")
    if not isinstance(joint_thresholds, dict):
        raise ActivePilotError("Joint counterfactual thresholds are missing")
    joint_mean_limit = _finite_number(
        joint_thresholds.get("max_mean_abs_rgb_delta"), "Joint RGB threshold", nonnegative=True
    )
    joint_psnr_limit = _finite_number(
        joint_thresholds.get("max_psnr_drop_db"), "Joint PSNR threshold", nonnegative=True
    )
    joint_mean = _finite_number(
        joint_payload.get("max_mean_abs_rgb_delta"), "Joint RGB counterfactual", nonnegative=True
    )
    joint_psnr = _finite_number(
        joint_payload.get("max_psnr_drop_db"), "Joint PSNR counterfactual"
    )
    joint_views = joint_payload.get("per_view")
    joint_view_count = joint_payload.get("training_view_count")
    if (
        type(joint_view_count) is not int or joint_view_count <= 0
        or not isinstance(joint_views, list) or len(joint_views) != joint_view_count
    ):
        raise ActivePilotError("Joint counterfactual per-view rows are incomplete")
    joint_view_means, joint_view_psnr = [], []
    for view in joint_views:
        if not isinstance(view, dict):
            raise ActivePilotError("Joint counterfactual per-view row is malformed")
        joint_view_means.append(_finite_number(
            view.get("mean_abs_rgb_delta"), "Joint per-view RGB", nonnegative=True
        ))
        _finite_number(view.get("max_abs_rgb_delta"), "Joint per-view maximum RGB", nonnegative=True)
        joint_view_psnr.append(_finite_number(view.get("psnr_drop_db"), "Joint per-view PSNR"))
        _finite_number(
            view.get("single_point_sum_mean_abs_rgb_delta"),
            "Joint single-point comparison", nonnegative=True,
        )
        _finite_number(
            view.get("joint_minus_single_sum_mean_abs_rgb_delta"),
            "Joint-minus-single comparison",
        )
    recomputed_joint_safe = joint_mean <= joint_mean_limit and joint_psnr <= joint_psnr_limit
    if (
        joint_mean != max(joint_view_means) or joint_psnr != max(joint_view_psnr)
        or joint_payload.get("safe_to_release") is not recomputed_joint_safe
        or not recomputed_joint_safe
    ):
        raise ActivePilotError("Joint counterfactual exceeds its declared safety threshold")
    plan["_plan_path"] = str(path)
    plan["_plan_sha256"] = _sha256(path)
    plan["_counterfactual_artifact"] = str(artifact_path)
    plan["_joint_counterfactual_artifact"] = str(joint_path)
    plan["_receiver_candidate_table"] = str(receiver_path)
    plan["_donor_uids"] = donor_uids
    plan["_receiver_xyz"] = targets
    plan["_anchor_uids"] = anchor_uids
    plan["_receiver_candidate_ids"] = match_receiver_ids
    plan["_receiver_v2"] = receiver_v2
    plan["_receiver_v2_selection_anchor_clone"] = anchor_clone
    plan["_anchor_clone_original_v2_plan"] = (
        None if anchor_clone_original_plan is None else str(anchor_clone_original_plan)
    )
    plan["_anchor_clone_view_manifest"] = (
        None if anchor_clone_view_manifest is None else str(anchor_clone_view_manifest)
    )
    plan["_receiver_v2_artifact"] = (
        None if receiver_v2_artifact is None else str(receiver_v2_artifact)
    )
    plan["_receiver_v2_rows"] = receiver_v2_rows
    plan["_receiver_opacities"] = receiver_opacities
    plan["_receiver_scales"] = receiver_scales
    return plan


def _alignment(gaussians) -> dict:
    count = int(gaussians.get_xyz.shape[0])
    rows = {
        name: int(getattr(gaussians, name).shape[0])
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    }
    if gaussians.lifecycle is not None:
        rows["lifecycle.uid"] = int(gaussians.lifecycle.uid.numel())
    if gaussians.observation is not None:
        rows["observation"] = int(gaussians.observation.real_view_opportunities.numel())
    if gaussians.future_utility is not None:
        rows["future_utility"] = int(gaussians.future_utility.tracked_cohort.numel())
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            name = f"optimizer.{group.get('name', 'unnamed')}"
            if len(group.get("params", [])) != 1:
                rows[name] = -1
                continue
            parameter = group["params"][0]
            rows[name] = int(parameter.shape[0])
            state = optimizer.state.get(parameter)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state.get(key)
                    rows[f"{name}.state.{key}"] = (
                        int(value.shape[0]) if torch.is_tensor(value) and value.ndim > 0 else -1
                    )
    mismatched = {name: value for name, value in rows.items() if value != count}
    return {"gaussian_count": count, "rows": rows, "mismatched": mismatched, "pass": not mismatched}


def _audit_tracking_state(gaussians, *, label: str) -> None:
    try:
        gaussians.lifecycle.audit_strict(expected_count=gaussians.get_xyz.shape[0])
        gaussians.observation.audit_strict(
            expected_uid=gaussians.lifecycle.uid,
            expected_count=gaussians.get_xyz.shape[0],
        )
        if gaussians.future_utility is not None:
            gaussians.future_utility.audit_strict(expected_count=gaussians.get_xyz.shape[0])
    except (ValueError, RuntimeError) as exc:
        raise ActivePilotError(f"{label} A0/B0/C state audit failed") from exc


def _receiver_birth_parameters(gaussians, plan: dict, row_by_uid: dict[int, int]):
    """Build the frozen Receiver parameters shared by RD-v1 and Birth-only."""
    targets = plan.get("_receiver_xyz")
    anchor_uids = plan.get("_anchor_uids")
    receiver_ids = plan.get("_receiver_candidate_ids")
    if (
        not isinstance(targets, list) or not isinstance(anchor_uids, list)
        or not isinstance(receiver_ids, list)
        or not (len(targets) == len(anchor_uids) == len(receiver_ids) > 0)
    ):
        raise ActivePilotError("Use load_active_plan before applying an intervention")
    if any(uid not in row_by_uid for uid in anchor_uids):
        raise ActivePilotError("A verified receiver anchor UID is not live")
    anchor_rows = torch.tensor(
        [row_by_uid[uid] for uid in anchor_uids], dtype=torch.int64,
        device=gaussians.get_xyz.device,
    )
    anchor_raw = {
        "_xyz": gaussians._xyz[anchor_rows].detach().clone(),
        "_features_dc": gaussians._features_dc[anchor_rows].detach().clone(),
        "_features_rest": gaussians._features_rest[anchor_rows].detach().clone(),
        "_opacity": gaussians._opacity[anchor_rows].detach().clone(),
        "_scaling": gaussians._scaling[anchor_rows].detach().clone(),
        "_rotation": gaussians._rotation[anchor_rows].detach().clone(),
    }
    target_xyz = torch.tensor(
        targets, dtype=gaussians._xyz.dtype, device=gaussians.get_xyz.device
    )
    receiver_v2 = plan.get("_receiver_v2") is True
    anchor_clone = plan.get("_receiver_v2_selection_anchor_clone") is True
    if receiver_v2 or anchor_clone:
        evidence_rows = plan.get("_receiver_v2_rows")
        if not isinstance(evidence_rows, list) or len(evidence_rows) != len(anchor_uids):
            raise ActivePilotError("Receiver-v2 evidence rows are incomplete")
        recorded_anchor_xyz = torch.tensor(
            [row["anchor_xyz"] for row in evidence_rows],
            dtype=gaussians._xyz.dtype, device=gaussians.get_xyz.device,
        )
        if not torch.allclose(recorded_anchor_xyz, anchor_raw["_xyz"], rtol=0, atol=1e-6):
            raise ActivePilotError("Receiver-v2 anchor XYZ differs from the source checkpoint")
        if receiver_v2:
            if not torch.isfinite(target_xyz).all() or torch.any(
                torch.linalg.vector_norm(target_xyz - anchor_raw["_xyz"], dim=1) <= 1e-8
            ).item():
                raise ActivePilotError("Receiver-v2 runtime XYZ is non-finite or overlapping")
        elif not torch.equal(target_xyz, anchor_raw["_xyz"]):
            raise ActivePilotError("Receiver-v2 selection Anchor-Clone XYZ must exactly equal its Anchor")
    elif not torch.equal(target_xyz, anchor_raw["_xyz"]):
        raise ActivePilotError("Verified receiver XYZ does not equal its live anchor UID coordinate")
    if anchor_clone:
        target_xyz = anchor_raw["_xyz"]
    if receiver_v2:
        activated_opacity = torch.tensor(
            plan["_receiver_opacities"], dtype=gaussians._opacity.dtype,
            device=gaussians.get_xyz.device,
        ).reshape(-1, 1)
        activated_scale = torch.tensor(
            plan["_receiver_scales"], dtype=gaussians._scaling.dtype,
            device=gaussians.get_xyz.device,
        )
        new_opacity = gaussians.inverse_opacity_activation(activated_opacity)
        new_scaling = gaussians.scaling_inverse_activation(activated_scale)
        if not torch.isfinite(new_opacity).all() or not torch.isfinite(new_scaling).all():
            raise ActivePilotError("Receiver-v2 runtime opacity/scale is non-finite")
    else:
        new_opacity, new_scaling = anchor_raw["_opacity"], anchor_raw["_scaling"]
    return {
        "target_xyz": target_xyz, "features_dc": anchor_raw["_features_dc"],
        "features_rest": anchor_raw["_features_rest"], "opacity": new_opacity,
        "scaling": new_scaling, "rotation": anchor_raw["_rotation"],
        "anchor_raw": anchor_raw, "anchor_uids": anchor_uids,
        "receiver_ids": receiver_ids, "receiver_v2": receiver_v2,
        "anchor_clone": anchor_clone,
    }


def _append_receiver_births(gaussians, birth: dict, *, anchor_rows, iteration: int):
    """Append Receiver children without altering existing rows or statistics."""
    previous_count = int(gaussians.get_xyz.shape[0])
    gaussians.densification_postfix(
        birth["target_xyz"], birth["features_dc"], birth["features_rest"],
        birth["opacity"], birth["scaling"], birth["rotation"],
        preserve_existing_stats=True,
    )
    child_uid = gaussians.lifecycle.append_children_from_rows(
        anchor_rows.to(dtype=torch.int64), "clone", int(iteration)
    )
    count = len(birth["anchor_uids"])
    gaussians.observation.append_zeros(count)
    if gaussians.future_utility is not None:
        tail = slice(previous_count, previous_count + count)
        gaussians.future_utility.append_births(
            child_uid=child_uid,
            birth_iter=gaussians.lifecycle.birth_iter[tail],
            birth_type=gaussians.lifecycle.birth_type[tail],
            source_uid=gaussians.lifecycle.source_uid[tail],
            generation=gaussians.lifecycle.generation[tail],
            xyz=gaussians.get_xyz[tail], opacity=gaussians.get_opacity[tail],
            scaling=gaussians.get_scaling[tail], previous_count=previous_count,
        )
    return previous_count, child_uid


def apply_active_reallocation(gaussians, plan: dict, *, iteration: int, event_log) -> dict:
    """Release growth-only UIDs and add exactly K receiver-anchored clones."""
    if getattr(gaussians, "_i2_active_intervention_applied", False):
        raise ActivePilotError("Active intervention has already been applied")
    if gaussians.lifecycle is None or gaussians.observation is None:
        raise ActivePilotError("Active intervention requires current A0+B0 state")
    uids = plan.get("_donor_uids")
    targets = plan.get("_receiver_xyz")
    anchor_uids = plan.get("_anchor_uids")
    receiver_ids = plan.get("_receiver_candidate_ids")
    if (
        not isinstance(uids, list) or not isinstance(targets, list)
        or not isinstance(anchor_uids, list) or not isinstance(receiver_ids, list)
        or not (len(uids) == len(targets) == len(anchor_uids) == len(receiver_ids))
    ):
        raise ActivePilotError("Use load_active_plan before applying an intervention")
    if int(iteration) != int(plan.get("input_audit", {}).get("source_iteration", -1)):
        raise ActivePilotError("Runtime iteration does not match the verified plan")
    event_path = Path(event_log).resolve()
    if event_path.exists():
        raise ActivePilotError("Active event log already exists; refusing ambiguous success state")
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=event_path.parent,
            prefix=f".{event_path.name}.preflight-", delete=True,
        ) as handle:
            handle.write("preflight")
            handle.flush()
    except OSError as exc:
        raise ActivePilotError("Active event log destination is not writable") from exc
    live_uids = gaussians.lifecycle.uid.detach().cpu().tolist()
    row_by_uid = {int(uid): row for row, uid in enumerate(live_uids)}
    if len(row_by_uid) != len(live_uids) or any(uid not in row_by_uid for uid in uids):
        raise ActivePilotError("Selected donor UID is missing or live UID rows are not unique")
    donor_rows = torch.tensor([row_by_uid[uid] for uid in uids], dtype=torch.int64,
                              device=gaussians.get_xyz.device)
    if torch.any(gaussians.lifecycle.birth_type[donor_rows] == 0).item():
        raise ActivePilotError("Root Gaussian release is forbidden")
    if any(uid not in row_by_uid for uid in anchor_uids):
        raise ActivePilotError("A verified receiver anchor UID is not live")
    if set(anchor_uids) & set(uids):
        raise ActivePilotError("A verified receiver anchor is selected for release")
    alignment_before = _alignment(gaussians)
    if not alignment_before["pass"]:
        raise ActivePilotError("Pre-intervention Gaussian/optimizer state is misaligned")
    try:
        gaussians.lifecycle.audit_strict(expected_count=gaussians.get_xyz.shape[0])
        gaussians.observation.audit_strict(
            expected_uid=gaussians.lifecycle.uid, expected_count=gaussians.get_xyz.shape[0]
        )
        if gaussians.future_utility is not None:
            gaussians.future_utility.audit_strict(expected_count=gaussians.get_xyz.shape[0])
    except (ValueError, RuntimeError) as exc:
        raise ActivePilotError("Pre-intervention A0/B0/C state audit failed") from exc
    before = int(gaussians.get_xyz.shape[0])
    if before <= len(uids):
        raise ActivePilotError("Active budget would remove every Gaussian")
    mask = torch.zeros(before, dtype=torch.bool, device=gaussians.get_xyz.device)
    mask[donor_rows] = True
    survivor_mask = ~mask
    expected_xyz_gradient_accum = gaussians.xyz_gradient_accum[survivor_mask].detach().clone()
    expected_denom = gaussians.denom[survivor_mask].detach().clone()
    expected_max_radii2D = gaussians.max_radii2D[survivor_mask].detach().clone()
    birth = _receiver_birth_parameters(gaussians, plan, row_by_uid)
    receiver_v2 = birth["receiver_v2"]
    anchor_clone = birth["anchor_clone"]
    anchor_raw = birth["anchor_raw"]
    gaussians.prune_points(mask, lifecycle_reason="i2_active_release", iteration=int(iteration))
    after_release = int(gaussians.get_xyz.shape[0])
    surviving_rows = {int(uid): row for row, uid in enumerate(
        gaussians.lifecycle.uid.detach().cpu().tolist()
    )}
    anchor_rows = torch.tensor(
        [surviving_rows[uid] for uid in anchor_uids], dtype=torch.int64,
        device=gaussians.get_xyz.device,
    )
    previous_count, child_uid = _append_receiver_births(
        gaussians, birth, anchor_rows=anchor_rows, iteration=int(iteration)
    )
    after = int(gaussians.get_xyz.shape[0])
    alignment = _alignment(gaussians)
    conserved = before == after and before - after_release == len(uids) and after - after_release == len(uids)
    stats_preserved = (
        torch.equal(gaussians.xyz_gradient_accum[:after_release], expected_xyz_gradient_accum)
        and torch.equal(gaussians.denom[:after_release], expected_denom)
        and torch.equal(gaussians.max_radii2D[:after_release], expected_max_radii2D)
        and bool(torch.count_nonzero(gaussians.xyz_gradient_accum[after_release:]).item() == 0)
        and bool(torch.count_nonzero(gaussians.denom[after_release:]).item() == 0)
        and bool(torch.count_nonzero(gaussians.max_radii2D[after_release:]).item() == 0)
    )
    parameter_copy = {
        name: bool(torch.equal(getattr(gaussians, name)[previous_count:], expected))
        for name, expected in anchor_raw.items()
    } if anchor_clone else {}
    lifecycle_source_matches = bool(torch.equal(
        gaussians.lifecycle.source_uid[previous_count:],
        torch.tensor(anchor_uids, dtype=torch.int64, device=gaussians.lifecycle.uid.device),
    ))
    try:
        gaussians.lifecycle.audit_strict(expected_count=after)
        gaussians.observation.audit_strict(
            expected_uid=gaussians.lifecycle.uid, expected_count=after
        )
        if gaussians.future_utility is not None:
            gaussians.future_utility.audit_strict(expected_count=after)
    except (ValueError, RuntimeError) as exc:
        raise ActivePilotError("Post-intervention A0/B0/C state audit failed") from exc
    if (
        not conserved or not alignment["pass"] or not stats_preserved
        or not lifecycle_source_matches
        or (anchor_clone and not all(parameter_copy.values()))
    ):
        raise ActivePilotError("Active intervention failed budget/alignment audit")
    gaussians._i2_active_intervention_applied = True
    event = {
        "format": (
            ACTIVE_EVENT_RD_V1_FORMAT if plan.get("_rd_v1") is True else
            ACTIVE_EVENT_V2_SELECTION_ANCHOR_CLONE_FORMAT
            if anchor_clone else
            ACTIVE_EVENT_RECEIVER_V2_FORMAT if receiver_v2 else ACTIVE_EVENT_FORMAT
        ),
        "iteration": int(iteration),
        "strategy": (
            "rd_v1_receiver_v2_reallocation" if plan.get("_rd_v1") is True else
            "receiver_v2_selection_anchor_clone" if anchor_clone else
            "receiver_v2_geometry_constrained" if receiver_v2 else "b1_only_exploratory"
        ),
        "method_version": (
            "RD-v1" if plan.get("_rd_v1") is True else
            V2_SELECTION_ANCHOR_CLONE_METHOD if anchor_clone else
            "receiver_v2_tangent_jacobian_v1" if receiver_v2 else "receiver_v1_anchor_clone"
        ),
        "plan_path": plan["_plan_path"], "plan_sha256": plan["_plan_sha256"],
        "counterfactual_artifact": plan["_counterfactual_artifact"],
        "joint_counterfactual_artifact": plan["_joint_counterfactual_artifact"],
        "rd_v1_donor_audit": plan.get("_rd_v1_donor_audit"),
        "receiver_candidate_table": plan["_receiver_candidate_table"],
        "receiver_v2_evidence": plan.get("_receiver_v2_artifact"),
        "original_receiver_v2_plan": plan.get("_anchor_clone_original_v2_plan"),
        "training_view_manifest": plan.get("_anchor_clone_view_manifest"),
        "released_uids": uids, "born_uids": child_uid.detach().cpu().tolist(),
        "receiver_candidate_ids": receiver_ids, "anchor_uids": anchor_uids,
        "birth_path": (
            "receiver_v2_selection_anchor_raw_clone" if anchor_clone else
            "receiver_v2_geometry_constrained_clone" if receiver_v2 else "receiver_anchored_clone"
        ),
        "death_path": "i2_active_release",
        "gaussian_before": before, "gaussian_after_release": after_release,
        "gaussian_after": after, "budget_conserved": conserved, "alignment_before": alignment_before,
        "alignment": alignment, "survivor_densification_stats_preserved": stats_preserved,
        "anchor_raw_parameter_copy": parameter_copy,
        "lifecycle_source_uid_matches_anchor_order": lifecycle_source_matches,
        "rollback": "Reload the untouched source checkpoint and rerun without --enable_i2_active_pilot.",
    }
    temporary_event = event_path.with_name(f".{event_path.name}.tmp")
    temporary_event.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_event.replace(event_path)
    return event


def apply_birth_only(gaussians, plan: dict, *, iteration: int, event_log) -> dict:
    """Append the frozen RD-v1 Receiver sequence without releasing any Donor."""
    if getattr(gaussians, "_i2_active_intervention_applied", False):
        raise ActivePilotError("Active intervention has already been applied")
    if plan.get("_rd_v1") is not True:
        raise ActivePilotError("Birth-only accepts only a verified RD-v1 Plan")
    if gaussians.lifecycle is None or gaussians.observation is None:
        raise ActivePilotError("Birth-only intervention requires current A0+B0 state")
    if int(iteration) != int(plan.get("input_audit", {}).get("source_iteration", -1)):
        raise ActivePilotError("Runtime iteration does not match the verified plan")
    donor_uids = plan.get("_donor_uids")
    anchor_uids = plan.get("_anchor_uids")
    if not isinstance(donor_uids, list) or not donor_uids:
        raise ActivePilotError("Use load_active_plan before applying an intervention")
    event_path = Path(event_log).resolve()
    if event_path.exists():
        raise ActivePilotError("Active event log already exists; refusing ambiguous success state")
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=event_path.parent,
            prefix=f".{event_path.name}.preflight-", delete=True,
        ) as handle:
            handle.write("preflight"); handle.flush()
    except OSError as exc:
        raise ActivePilotError("Active event log destination is not writable") from exc

    live_uids = gaussians.lifecycle.uid.detach().cpu().tolist()
    row_by_uid = {int(uid): row for row, uid in enumerate(live_uids)}
    if len(row_by_uid) != len(live_uids):
        raise ActivePilotError("Live UID rows are not unique")
    if any(uid not in row_by_uid for uid in donor_uids):
        raise ActivePilotError("Selected donor UID is missing")
    donor_rows = torch.tensor(
        [row_by_uid[uid] for uid in donor_uids], dtype=torch.int64,
        device=gaussians.get_xyz.device,
    )
    if torch.any(gaussians.lifecycle.birth_type[donor_rows] == 0).item():
        raise ActivePilotError("Root Gaussian release candidate is forbidden")
    if set(anchor_uids) & set(donor_uids):
        raise ActivePilotError("A verified receiver anchor is selected as a donor")
    alignment_before = _alignment(gaussians)
    if not alignment_before["pass"]:
        raise ActivePilotError("Pre-intervention Gaussian/optimizer state is misaligned")
    _audit_tracking_state(gaussians, label="Pre-intervention")
    before = int(gaussians.get_xyz.shape[0])
    birth = _receiver_birth_parameters(gaussians, plan, row_by_uid)
    raw_before = {
        name: getattr(gaussians, name).detach().clone()
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    }
    stats_before = {
        name: getattr(gaussians, name).detach().clone()
        for name in ("xyz_gradient_accum", "denom", "max_radii2D")
    }
    optimizer_before = {}
    for group in gaussians.optimizer.param_groups:
        parameter = group["params"][0]
        state = gaussians.optimizer.state.get(parameter, {})
        optimizer_before[group.get("name", "unnamed")] = {
            key: value.detach().clone()
            for key, value in state.items()
            if key in ("exp_avg", "exp_avg_sq") and torch.is_tensor(value)
        }
    anchor_rows = torch.tensor(
        [row_by_uid[uid] for uid in anchor_uids], dtype=torch.int64,
        device=gaussians.get_xyz.device,
    )
    previous_count, child_uid = _append_receiver_births(
        gaussians, birth, anchor_rows=anchor_rows, iteration=int(iteration)
    )
    after = int(gaussians.get_xyz.shape[0])
    alignment = _alignment(gaussians)
    existing_parameters_preserved = all(
        torch.equal(getattr(gaussians, name)[:before], value)
        for name, value in raw_before.items()
    )
    survivor_stats_preserved = all(
        torch.equal(getattr(gaussians, name)[:before], value)
        and torch.count_nonzero(getattr(gaussians, name)[before:]).item() == 0
        for name, value in stats_before.items()
    )
    optimizer_state_preserved = True
    for group in gaussians.optimizer.param_groups:
        parameter = group["params"][0]
        state = gaussians.optimizer.state.get(parameter, {})
        for key, value in optimizer_before.get(group.get("name", "unnamed"), {}).items():
            current = state.get(key)
            optimizer_state_preserved = optimizer_state_preserved and bool(
                torch.is_tensor(current)
                and torch.equal(current[:before], value)
                and torch.count_nonzero(current[before:]).item() == 0
            )
    source_matches = torch.equal(
        gaussians.lifecycle.source_uid[before:],
        torch.tensor(anchor_uids, dtype=torch.int64, device=gaussians.lifecycle.uid.device),
    )
    _audit_tracking_state(gaussians, label="Post-intervention")
    received = len(anchor_uids)
    if (
        previous_count != before or after != before + received or not alignment["pass"]
        or not existing_parameters_preserved or not survivor_stats_preserved
        or not optimizer_state_preserved or not bool(source_matches)
    ):
        raise ActivePilotError("Birth-only intervention failed count/alignment audit")
    gaussians._i2_active_intervention_applied = True
    event = {
        "format": BIRTH_ONLY_EVENT_FORMAT,
        "iteration": int(iteration), "strategy": "birth_only",
        "policy_version": "RD-v1",
        "plan_path": plan["_plan_path"], "plan_sha256": plan["_plan_sha256"],
        "adaptive_decision_path": plan.get("_adaptive_decision_path"),
        "adaptive_decision_sha256": plan.get("_adaptive_decision_sha256"),
        "born_uids": child_uid.detach().cpu().tolist(),
        "anchor_uids": anchor_uids,
        "receiver_candidate_ids": birth["receiver_ids"],
        "released_uids": [], "gaussian_before": before,
        "gaussian_after_release": before, "gaussian_after": after,
        "released_units": 0, "received_units": received,
        "budget_conserved": False,
        "alignment_before": alignment_before, "alignment": alignment,
        "existing_parameters_preserved": existing_parameters_preserved,
        "survivor_densification_stats_preserved": survivor_stats_preserved,
        "optimizer_state_preserved": optimizer_state_preserved,
        "lifecycle_source_uid_matches_anchor_order": bool(source_matches),
        "rollback": "Reload the untouched source checkpoint and rerun without --enable_i2_birth_only.",
    }
    temporary_event = event_path.with_name(f".{event_path.name}.tmp")
    temporary_event.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_event.replace(event_path)
    return event


def apply_release_only(gaussians, plan: dict, *, iteration: int, event_log) -> dict:
    """Release the verified donor UIDs without creating receiver Gaussians."""
    if getattr(gaussians, "_i2_active_intervention_applied", False):
        raise ActivePilotError("Active intervention has already been applied")
    if gaussians.lifecycle is None or gaussians.observation is None:
        raise ActivePilotError("Release-only intervention requires current A0+B0 state")
    uids = plan.get("_donor_uids")
    if not isinstance(uids, list) or not uids:
        raise ActivePilotError("Use load_active_plan before applying an intervention")
    if int(iteration) != int(plan.get("input_audit", {}).get("source_iteration", -1)):
        raise ActivePilotError("Runtime iteration does not match the verified plan")

    event_path = Path(event_log).resolve()
    if event_path.exists():
        raise ActivePilotError("Active event log already exists; refusing ambiguous success state")
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=event_path.parent,
            prefix=f".{event_path.name}.preflight-", delete=True,
        ) as handle:
            handle.write("preflight")
            handle.flush()
    except OSError as exc:
        raise ActivePilotError("Active event log destination is not writable") from exc

    live_uids = gaussians.lifecycle.uid.detach().cpu().tolist()
    row_by_uid = {int(uid): row for row, uid in enumerate(live_uids)}
    if len(row_by_uid) != len(live_uids) or any(uid not in row_by_uid for uid in uids):
        raise ActivePilotError("Selected donor UID is missing or live UID rows are not unique")
    donor_rows = torch.tensor(
        [row_by_uid[uid] for uid in uids], dtype=torch.int64,
        device=gaussians.get_xyz.device,
    )
    if torch.any(gaussians.lifecycle.birth_type[donor_rows] == 0).item():
        raise ActivePilotError("Root Gaussian release is forbidden")

    alignment_before = _alignment(gaussians)
    if not alignment_before["pass"]:
        raise ActivePilotError("Pre-intervention Gaussian/optimizer state is misaligned")
    try:
        gaussians.lifecycle.audit_strict(expected_count=gaussians.get_xyz.shape[0])
        gaussians.observation.audit_strict(
            expected_uid=gaussians.lifecycle.uid,
            expected_count=gaussians.get_xyz.shape[0],
        )
        if gaussians.future_utility is not None:
            gaussians.future_utility.audit_strict(expected_count=gaussians.get_xyz.shape[0])
    except (ValueError, RuntimeError) as exc:
        raise ActivePilotError("Pre-intervention A0/B0/C state audit failed") from exc

    before = int(gaussians.get_xyz.shape[0])
    if before <= len(uids):
        raise ActivePilotError("Release-only budget would remove every Gaussian")
    mask = torch.zeros(before, dtype=torch.bool, device=gaussians.get_xyz.device)
    mask[donor_rows] = True
    survivor_mask = ~mask
    expected_xyz_gradient_accum = gaussians.xyz_gradient_accum[survivor_mask].detach().clone()
    expected_denom = gaussians.denom[survivor_mask].detach().clone()
    expected_max_radii2D = gaussians.max_radii2D[survivor_mask].detach().clone()

    gaussians.prune_points(
        mask, lifecycle_reason="i2_active_release", iteration=int(iteration)
    )
    after = int(gaussians.get_xyz.shape[0])
    alignment = _alignment(gaussians)
    stats_preserved = (
        torch.equal(gaussians.xyz_gradient_accum, expected_xyz_gradient_accum)
        and torch.equal(gaussians.denom, expected_denom)
        and torch.equal(gaussians.max_radii2D, expected_max_radii2D)
    )
    expected_after = before - len(uids)
    if after != expected_after or not alignment["pass"] or not stats_preserved:
        raise ActivePilotError("Release-only intervention failed count/alignment audit")
    try:
        gaussians.lifecycle.audit_strict(expected_count=after)
        gaussians.observation.audit_strict(
            expected_uid=gaussians.lifecycle.uid, expected_count=after
        )
        if gaussians.future_utility is not None:
            gaussians.future_utility.audit_strict(expected_count=after)
    except (ValueError, RuntimeError) as exc:
        raise ActivePilotError("Post-intervention A0/B0/C state audit failed") from exc

    gaussians._i2_active_intervention_applied = True
    event = {
        "format": RELEASE_ONLY_EVENT_FORMAT,
        "iteration": int(iteration),
        "strategy": "release_only",
        "policy_version": "RD-v1" if plan.get("_rd_v1") is True else None,
        "plan_path": plan["_plan_path"],
        "plan_sha256": plan["_plan_sha256"],
        "adaptive_decision_path": plan.get("_adaptive_decision_path"),
        "adaptive_decision_sha256": plan.get("_adaptive_decision_sha256"),
        "counterfactual_artifact": plan["_counterfactual_artifact"],
        "joint_counterfactual_artifact": plan["_joint_counterfactual_artifact"],
        "released_uids": uids,
        "death_path": "i2_active_release",
        "released_units": len(uids),
        "received_units": 0,
        "gaussian_before": before,
        "gaussian_after": after,
        "budget_conserved": False,
        "budget_conserving_reallocation": False,
        "alignment_before": alignment_before,
        "alignment": alignment,
        "survivor_densification_stats_preserved": stats_preserved,
        "dropout_randomness": {
            "initial_reference_count": before,
            "first_reference_extension": None,
            "cross_arm_alignment_before_extension": (
                "conditional_on_other_arm_draw_count_equal_to_initial_reference_count"
            ),
            "strict_rng_alignment_after_extension": False,
        },
        "rollback": "Reload the untouched source checkpoint and rerun without --enable_i2_release_only.",
    }
    temporary_event = event_path.with_name(f".{event_path.name}.tmp")
    temporary_event.write_text(
        json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_event.replace(event_path)
    return event


def record_release_only_reference_extension(
    event_log, *, iteration: int, gaussian_count: int, effective_reference_count: int
) -> None:
    """Atomically record the first long-run Dropout reference extension."""
    event_path = Path(event_log).resolve()
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivePilotError("Release-only event log cannot record RNG extension") from exc
    if (
        event.get("format") != RELEASE_ONLY_EVENT_FORMAT
        or event.get("strategy") != "release_only"
    ):
        raise ActivePilotError("RNG extension target is not a release-only event log")
    randomness = event.get("dropout_randomness")
    if not isinstance(randomness, dict):
        raise ActivePilotError("Release-only event log has no randomness audit block")
    if randomness.get("first_reference_extension") is not None:
        return
    randomness["first_reference_extension"] = {
        "iteration": int(iteration),
        "gaussian_count": int(gaussian_count),
        "effective_reference_count": int(effective_reference_count),
    }
    randomness["strict_rng_alignment_lost_at_iteration"] = int(iteration)
    temporary_event = event_path.with_name(f".{event_path.name}.tmp")
    temporary_event.write_text(
        json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_event.replace(event_path)
