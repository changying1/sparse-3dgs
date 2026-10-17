"""Read-only I2-D0 donor/receiver allocation planning.

The planner consumes exported tables only.  It never imports GaussianModel and
cannot prune, clone, split, or mutate optimizer/model state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


D0_REPORT_VERSION = "i2_d0_read_only_plan_v2"
SUPPORTED_RECEIVER_BASES = {"reconstruction_error", "structural_defect", "growth_demand"}
RECEIVER_PROTOCOL_VERSION = "i2_receiver_protocol_v1"


class D0AuditError(ValueError):
    """Input evidence cannot safely support a D0 plan."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(manifest_path: Path, artifact: dict) -> Path:
    if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
        raise D0AuditError("Manifest artifact must include path and sha256")
    path = Path(artifact["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    try:
        if not path.is_file():
            raise D0AuditError(f"Manifest artifact is missing: {path}")
        actual_hash = _sha256(path)
    except PermissionError as exc:
        raise D0AuditError(f"Manifest artifact cannot be read: {path}") from exc
    if actual_hash != artifact["sha256"]:
        raise D0AuditError(f"Manifest artifact hash mismatch: {path}")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_d0_inputs(manifest_path) -> tuple[dict, list[dict], list[dict], list[dict] | None]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "i2_d_input_manifest_v1":
        raise D0AuditError("Unsupported D input manifest format")
    if manifest.get("join_key") != "uid" or manifest.get("uid_alignment") != "exact":
        raise D0AuditError("D input manifest lacks exact UID alignment")
    if manifest.get("diagnostic_status", {}).get("active_intervention_performed") is not False:
        raise D0AuditError("D0 requires a read-only diagnostic input manifest")
    source_checkpoint = _resolve_artifact(manifest_path, manifest.get("source_checkpoint"))
    b1_path = _resolve_artifact(manifest_path, manifest.get("candidate_tables", {}).get("b1"))
    b1_rows = _read_csv(b1_path)
    if len(b1_rows) != int(manifest.get("gaussian_count", -1)):
        raise D0AuditError("B1 table row count does not match manifest Gaussian count")
    c_rows = []
    c_artifact = manifest.get("candidate_tables", {}).get("c")
    if c_artifact:
        c_rows = _read_csv(_resolve_artifact(manifest_path, c_artifact))
    receiver_rows = None
    receiver_artifact = manifest.get("receiver_candidates")
    if receiver_artifact:
        receiver_path = _resolve_artifact(manifest_path, receiver_artifact)
        receiver_rows = _read_csv(receiver_path)
        manifest["_verified_receiver_artifact"] = str(receiver_path)
    manifest["_verified_source_checkpoint"] = str(source_checkpoint)
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = _sha256(manifest_path)
    return manifest, b1_rows, c_rows, receiver_rows


def _as_int(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise D0AuditError(f"Invalid donor field: {field}") from exc


def _as_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise D0AuditError(f"Invalid donor field: {field}") from exc
    if not math.isfinite(value):
        raise D0AuditError(f"Non-finite donor field: {field}")
    return value


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise D0AuditError(f"Invalid boolean value: {value!r}")


def select_donors(b1_rows, *, budget: int, min_age: int = 100, min_opportunities: int = 1,
                  excluded_uids=()):
    if type(budget) is not int or budget < 0:
        raise D0AuditError("budget must be a non-negative integer")
    if min_age < 0 or min_opportunities < 1:
        raise D0AuditError("Donor eligibility thresholds are invalid")
    normalized = []
    seen = set()
    excluded = {int(uid) for uid in excluded_uids}
    protected = Counter()
    for source in b1_rows:
        uid = _as_int(source, "uid")
        if uid in seen:
            raise D0AuditError(f"Duplicate live donor UID: {uid}")
        seen.add(uid)
        if "alive_at_checkpoint" in source and not _as_bool(source["alive_at_checkpoint"]):
            raise D0AuditError(f"B1 candidate UID is not alive at checkpoint: {uid}")
        age = _as_int(source, "age")
        opportunities = _as_int(source, "real_view_opportunities")
        retained = _as_int(source, "dropout_retained_hits")
        supported = _as_int(source, "screen_supported_hits")
        if not (0 <= supported <= retained <= opportunities):
            raise D0AuditError(f"Observation counters are inconsistent for UID {uid}")
        low = _as_bool(source.get("low_screen_support_candidate"))
        reason = None
        birth_type = str(source.get("birth_type", "unknown")).strip()
        if uid in excluded:
            reason = "protected_counterfactual_high_contribution"
        elif birth_type not in {"clone", "split"}:
            reason = "protected_root_or_unknown_birth"
        elif age < min_age:
            reason = "protected_too_young"
        elif opportunities < min_opportunities:
            reason = "protected_zero_opportunity"
        elif retained == 0:
            reason = "protected_no_retained_opportunity"
        elif not low:
            reason = "not_b1_low_support_proxy"
        if reason:
            protected[reason] += 1
            continue
        normalized.append({
            "uid": uid,
            "birth_type": birth_type,
            "age": age,
            "real_view_opportunities": opportunities,
            "dropout_retained_hits": retained,
            "screen_supported_hits": supported,
            "screen_support_rate": _as_float(source, "screen_support_rate"),
            "real_view_coverage_rate": _as_float(source, "real_view_coverage_rate"),
            "observation_status": source.get("observation_status", "unknown"),
            "selection_reason": (
                "B1-only exploratory proxy: mature, observed, live UID in the stable "
                "low-screen-support cohort; not asserted to be low future value"
            ),
        })
    normalized.sort(key=lambda row: (
        row["screen_support_rate"], row["real_view_coverage_rate"],
        -row["real_view_opportunities"], row["uid"],
    ))
    capacity = len(normalized)
    selected = normalized[:budget] if capacity >= budget else []
    controls = normalized[budget:2 * budget] if capacity >= 2 * budget else []
    strata = Counter((row["birth_type"], row["observation_status"]) for row in normalized)
    return {
        "eligible": normalized,
        "selected": selected,
        "controls": controls,
        "capacity": capacity,
        "selection_complete": len(selected) == budget,
        "control_complete": len(controls) == budget,
        "protected_counts": dict(sorted(protected.items())),
        "strata": [
            {"birth_type": key[0], "observation_status": key[1], "count": count}
            for key, count in sorted(strata.items())
        ],
    }


def _receiver_evidence_path(source, manifest, protocol, camera_id):
    raw = str(source.get("evidence_path", "")).strip()
    expected_hash = str(source.get("evidence_sha256", "")).strip().lower()
    if not raw or len(expected_hash) != 64:
        raise D0AuditError("Receiver evidence must include a path and SHA256")
    path = Path(raw)
    base = Path(manifest.get("_manifest_path", ".")).resolve().parent
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != expected_hash:
        raise D0AuditError(f"Receiver evidence is missing or has a SHA256 mismatch: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D0AuditError(f"Receiver evidence metadata is malformed: {path}") from exc
    if (
        evidence.get("format") != "i2_receiver_view_evidence_v1"
        or str(evidence.get("training_camera_id")) != camera_id
        or str(evidence.get("source_checkpoint_sha256", "")).lower()
        != protocol["source_checkpoint_sha256"]
        or evidence.get("generation_rule_id") != protocol["generation_rule_id"]
    ):
        raise D0AuditError("Receiver evidence metadata binding is invalid")
    payload = evidence.get("payload")
    if not isinstance(payload, dict):
        raise D0AuditError("Receiver evidence payload reference is missing")
    payload_path = Path(str(payload.get("path", "")))
    if not payload_path.is_absolute():
        payload_path = path.parent / payload_path
    payload_hash = str(payload.get("sha256", "")).lower()
    if not payload_path.is_file() or len(payload_hash) != 64 or _sha256(payload_path) != payload_hash:
        raise D0AuditError("Receiver evidence payload is missing or has a SHA256 mismatch")
    return path


def _receiver_protocol(manifest):
    protocol = manifest.get("receiver_protocol")
    if not isinstance(protocol, dict) or protocol.get("format") != RECEIVER_PROTOCOL_VERSION:
        raise D0AuditError("Receiver protocol is missing or unsupported")
    cameras = protocol.get("training_camera_ids")
    if not isinstance(cameras, list) or not cameras or len(cameras) != len(set(map(str, cameras))):
        raise D0AuditError("Receiver protocol must declare unique frozen training cameras")
    checkpoint_sha = str(protocol.get("source_checkpoint_sha256", "")).lower()
    manifest_sha = str(manifest.get("source_checkpoint", {}).get("sha256", "")).lower()
    if len(checkpoint_sha) != 64 or checkpoint_sha != manifest_sha:
        raise D0AuditError("Receiver protocol is not bound to the source checkpoint")
    for field in ("generation_rule_id", "source_code_fingerprint", "score_normalization"):
        if not str(protocol.get(field, "")).strip():
            raise D0AuditError(f"Receiver protocol lacks {field}")
    radius = _as_float(protocol, "spatial_dedup_radius")
    max_pixel_distance = _as_float(protocol, "max_anchor_pixel_distance")
    if radius <= 0 or max_pixel_distance <= 0:
        raise D0AuditError("Receiver spatial dedup radius must be positive")
    return protocol, set(map(str, cameras)), radius, max_pixel_distance


def validate_receivers(receiver_rows, manifest):
    if receiver_rows is None:
        return {"status": "blocked_missing_receiver_evidence", "rows": [], "capacity": 0}
    protocol, allowed_cameras, dedup_radius, max_pixel_distance = _receiver_protocol(manifest)
    normalized = []
    seen = set()
    for source in receiver_rows:
        candidate_id = str(source.get("candidate_id", "")).strip()
        basis = str(source.get("demand_basis", "")).strip()
        if not candidate_id or candidate_id in seen:
            raise D0AuditError("Receiver candidate IDs must be present and unique")
        seen.add(candidate_id)
        if basis not in SUPPORTED_RECEIVER_BASES:
            raise D0AuditError(
                f"Receiver {candidate_id} lacks credible reconstruction/structure/growth evidence"
            )
        camera_id = str(source.get("training_camera_id", "")).strip()
        if camera_id not in allowed_cameras:
            raise D0AuditError(f"Receiver {candidate_id} is not from a frozen training camera")
        evidence_path = _receiver_evidence_path(source, manifest, protocol, camera_id)
        try:
            score = float(source["demand_score"])
            units = int(source.get("requested_units", 1))
            xyz = tuple(float(source[field]) for field in ("world_x", "world_y", "world_z"))
            region = tuple(int(source[field]) for field in ("pixel_x0", "pixel_y0", "pixel_x1", "pixel_y1"))
            pixel = (int(source["pixel_u"]), int(source["pixel_v"]))
            reprojection_error = float(source["reprojection_error_px"])
            local_count = int(source["local_gaussian_count"])
            local_radius = float(source["local_density_radius"])
            anchor_uid = int(source["anchor_uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise D0AuditError(f"Receiver {candidate_id} demand is malformed") from exc
        if (
            not math.isfinite(score) or not 0 < score <= 1 or units <= 0
            or not all(math.isfinite(value) for value in xyz)
            or not math.isfinite(local_radius) or local_radius <= 0 or local_count < 0
            or anchor_uid < 0
            or region[0] < 0 or region[1] < 0 or region[2] <= region[0] or region[3] <= region[1]
            or not (region[0] <= pixel[0] < region[2] and region[1] <= pixel[1] < region[3])
            or not math.isfinite(reprojection_error) or reprojection_error > max_pixel_distance
        ):
            raise D0AuditError(f"Receiver {candidate_id} demand must be positive and finite")
        if not _as_bool(source.get("depth_valid")) or not _as_bool(source.get("geometry_valid")):
            raise D0AuditError(f"Receiver {candidate_id} lacks valid depth/geometry")
        if str(source.get("source_checkpoint_sha256", "")).lower() != protocol["source_checkpoint_sha256"]:
            raise D0AuditError(f"Receiver {candidate_id} checkpoint binding is invalid")
        if str(source.get("generation_rule_id", "")) != protocol["generation_rule_id"]:
            raise D0AuditError(f"Receiver {candidate_id} generation rule is invalid")
        for prior in normalized:
            dx = sum((xyz[index] - prior["target_xyz"][index]) ** 2 for index in range(3)) ** 0.5
            same_region = camera_id == prior["training_camera_id"] and not (
                region[2] <= prior["pixel_region"][0] or prior["pixel_region"][2] <= region[0]
                or region[3] <= prior["pixel_region"][1] or prior["pixel_region"][3] <= region[1]
            )
            if same_region or dx < dedup_radius:
                raise D0AuditError(f"Receiver {candidate_id} duplicates an existing pixel/spatial request")
        normalized.append({
            "candidate_id": candidate_id, "demand_basis": basis,
            "demand_score": score, "evidence_path": str(evidence_path),
            "evidence_sha256": str(source["evidence_sha256"]).lower(),
            "training_camera_id": camera_id, "pixel_region": region,
            "pixel": pixel, "reprojection_error_px": reprojection_error,
            "target_xyz": xyz, "anchor_type": str(source.get("anchor_type", "depth_unprojection")),
            "anchor_uid": anchor_uid,
            "local_gaussian_count": local_count, "local_density_radius": local_radius,
            "requested_units": units,
            "selection_reason": f"verified {basis} demand from {evidence_path.name}",
        })
    normalized.sort(key=lambda row: (-row["demand_score"], row["candidate_id"]))
    return {"status": "available", "rows": normalized,
            "capacity": sum(row["requested_units"] for row in normalized)}


def _validate_c_prediction(manifest, b1_rows, c_rows):
    if manifest.get("diagnostic_status", {}).get("c") not in {"available", "available_empty_cohort"}:
        return False, "c_status_unavailable"
    prediction = manifest.get("c_prediction")
    if not isinstance(prediction, dict):
        return False, "missing_prediction_manifest"
    try:
        path = _resolve_artifact(Path(manifest.get("_manifest_path", ".")), prediction.get("artifact"))
        prediction_iteration = int(prediction["prediction_iteration"])
        feature_cutoff = int(prediction["feature_cutoff_iteration"])
    except (D0AuditError, KeyError, TypeError, ValueError):
        return False, "invalid_prediction_artifact"
    if feature_cutoff > prediction_iteration or not c_rows:
        return False, "prediction_time_leakage_or_empty"
    b1_uids = {_as_int(row, "uid") for row in b1_rows}
    seen = set()
    forbidden = {"death_reason", "lifecycle_transition", "raw_uid_survival", "true_prune_avoidance"}
    for row in c_rows:
        try:
            uid = _as_int(row, "uid")
            value = _as_float(row, "predicted_future_utility")
            row_time = _as_int(row, "prediction_iteration")
        except D0AuditError:
            return False, "invalid_prediction_value"
        if uid in seen or uid not in b1_uids or row_time != prediction_iteration:
            return False, "prediction_uid_or_time_mismatch"
        if any(str(row.get(field, "")).strip() for field in forbidden):
            return False, "prediction_uses_future_outcome"
        seen.add(uid)
    return bool(seen) and path.is_file(), "verified" if seen else "empty_prediction"


def build_d0_plan(manifest, b1_rows, c_rows, receiver_rows, *, budget=500,
                  min_age=100, min_opportunities=1, excluded_uids=()):
    donor = select_donors(
        b1_rows, budget=budget, min_age=min_age, min_opportunities=min_opportunities,
        excluded_uids=excluded_uids,
    )
    receiver = validate_receivers(receiver_rows, manifest)
    c_informed, c_reason = _validate_c_prediction(manifest, b1_rows, c_rows)
    strategy = "c_future_utility_informed" if c_informed else "b1_only_exploratory"
    feasible_max = min(donor["capacity"], receiver["capacity"])
    ready = donor["selection_complete"] and receiver["capacity"] >= budget
    matches = []
    if ready:
        slots = []
        for row in receiver["rows"]:
            slots.extend([row] * row["requested_units"])
        for donor_row, receiver_row in zip(donor["selected"], slots[:budget]):
            matches.append({
                "donor_uid": donor_row["uid"],
                "receiver_candidate_id": receiver_row["candidate_id"],
                "released_units": 1, "received_units": 1,
                "receiver_reason": receiver_row["selection_reason"],
            })
    released = len(matches)
    received = sum(row["received_units"] for row in matches)
    blockers = []
    if not donor["selection_complete"]:
        blockers.append("donor_capacity_below_requested_budget")
    if receiver["status"] != "available":
        blockers.append("missing_credible_receiver_evidence")
    elif receiver["capacity"] < budget:
        blockers.append("receiver_capacity_below_requested_budget")
    b1_ready = ready and donor["control_complete"]
    return {
        "format": D0_REPORT_VERSION,
        "mode": "read_only_dry_run",
        "strategy": strategy,
        "semantic_guards": {
            "low_screen_support_is_candidate_proxy_only": True,
            "low_screen_support_is_low_value_label": False,
            "model_or_optimizer_mutated": False,
            "gaussian_prune_clone_split_called": False,
        },
        "configuration": {"budget_k": budget, "min_age": min_age,
                          "min_real_view_opportunities": min_opportunities,
                          "excluded_uids": sorted(int(uid) for uid in excluded_uids)},
        "input_audit": {
            "source_checkpoint": manifest.get("_verified_source_checkpoint"),
            "source_checkpoint_sha256": manifest.get("source_checkpoint", {}).get("sha256"),
            "source_checkpoint_sha256_verified": bool(
                manifest.get("_verified_source_checkpoint")
            ),
            "b1_uid_count": len(b1_rows),
            "b1_live_rows_from_checkpoint": True,
            "uid_uniqueness_verified": True,
            "source_iteration": manifest.get("iteration"),
            "source_manifest": {
                "path": manifest.get("_manifest_path"),
                "sha256": manifest.get("_manifest_sha256"),
            },
            "receiver_candidate_table": {
                "path": manifest.get("_verified_receiver_artifact"),
                "sha256": manifest.get("receiver_candidates", {}).get("sha256"),
            },
            "receiver_protocol": manifest.get("receiver_protocol"),
        },
        "donors": {key: value for key, value in donor.items() if key != "eligible"},
        "receivers": receiver,
        "matches": matches,
        "budget_audit": {
            "requested_k": budget, "released_units": released,
            "received_units": received,
            "conserved": ready and released == budget and received == budget,
            "donor_supply_range": [0, donor["capacity"]],
            "matched_feasible_range": [0, feasible_max],
        },
        "rollback": {
            "status": "ready" if ready else "no_active_plan_to_rollback",
            "donor_uids": [row["uid"] for row in donor["selected"]] if ready else [],
            "receiver_candidate_ids": [row["receiver_candidate_id"] for row in matches],
            "instruction": "Restore the untouched source checkpoint; D0 itself made no mutation.",
        },
        "safety_readiness": {
            "b1_only_exploratory": b1_ready,
            "c_informed": b1_ready and c_informed,
            "c_validation": c_reason,
        },
        "pilot_minimum_safety_ready": b1_ready,
        "blockers": blockers,
    }


def write_d0_outputs(plan, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "d0_dry_run_report.json"
    report.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    tables = {
        "d0_selected_donors.csv": plan["donors"]["selected"],
        "d0_control_donors.csv": plan["donors"]["controls"],
        "d0_candidate_matches.csv": plan["matches"],
    }
    for name, rows in tables.items():
        path = output_dir / name
        fields = list(rows[0]) if rows else ["status"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            if rows:
                writer.writerows(rows)
    return report
