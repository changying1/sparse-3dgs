"""Pure helpers for auditable I2 adaptive-K capacity decisions.

R0-v1 remains available for the frozen bounded experiments.  RD-v1 is a
separate, uncapped policy whose only budget equation is ``K = min(R, D)``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from utils.i2_active_pilot import receiver_v2_displacement_contract


class AdaptiveKError(RuntimeError):
    """R0-v1 evidence is invalid or insufficient, so execution must stop."""


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_receiver_selection(candidates, *, spatial_dedup_radius: float):
    """Apply the frozen Receiver-v2 rank/NMS rules without a budget break."""
    try:
        radius = float(spatial_dedup_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial_dedup_radius must be positive and finite") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("spatial_dedup_radius must be positive and finite")
    valid, rejected = [], []
    for row in candidates:
        try:
            alignment_value = row["photometric_alignment"]
            demand_value = row["demand_score"]
            if (
                isinstance(alignment_value, bool)
                or not isinstance(alignment_value, (int, float))
                or isinstance(demand_value, bool)
                or not isinstance(demand_value, (int, float))
            ):
                raise TypeError("ranking evidence must be numeric JSON values")
            alignment = float(alignment_value)
            demand_score = float(demand_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptiveKError(
                f"candidate {row.get('candidate_id')!r} has indeterminate ranking evidence"
            ) from exc
        if not math.isfinite(alignment) or not math.isfinite(demand_score):
            raise AdaptiveKError(
                f"candidate {row.get('candidate_id')!r} has non-finite ranking evidence"
            )
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
    accepted = []
    for row in valid:
        conflict = next(
            (old for old in accepted
             if math.dist(row["new_xyz"], old["new_xyz"]) < radius),
            None,
        )
        if conflict is not None:
            rejected.append({
                "candidate_id": row["candidate_id"],
                "reason": "v2_spatial_nms",
                "conflict_candidate_id": conflict["candidate_id"],
            })
            continue
        accepted.append(row)
    return accepted, rejected


def receiver_evidence_budget(accepted, *, k_cap: int | None = 20) -> dict:
    """Compute R and enforce the frozen R0-v1 positive-prefix contract."""
    if k_cap is not None and (
        isinstance(k_cap, bool) or not isinstance(k_cap, int) or k_cap <= 0
    ):
        raise AdaptiveKError("K_cap must be a positive integer or None")
    accepted_ids = []
    alignments = []
    for row in accepted:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AdaptiveKError("accepted candidate has no stable candidate_id")
        try:
            value = row["photometric_alignment"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("alignment must be a numeric JSON value")
            alignment = float(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptiveKError(
                f"candidate {candidate_id!r} has indeterminate photometric_alignment"
            ) from exc
        if not math.isfinite(alignment):
            raise AdaptiveKError(
                f"candidate {candidate_id!r} has non-finite photometric_alignment"
            )
        accepted_ids.append(candidate_id)
        alignments.append(alignment)
    if len(set(accepted_ids)) != len(accepted_ids):
        raise AdaptiveKError("accepted sequence contains duplicate candidate_id values")

    geometry_nms_capacity = len(accepted_ids)
    capped_count = geometry_nms_capacity if k_cap is None else min(k_cap, geometry_nms_capacity)
    capped_ids = accepted_ids[:capped_count]
    positive_ids = [
        candidate_id for candidate_id, alignment
        in zip(capped_ids, alignments[:capped_count]) if alignment > 0.0
    ]
    receiver_budget = len(positive_ids)
    expected_prefix = accepted_ids[:receiver_budget]
    if positive_ids != expected_prefix:
        raise AdaptiveKError(
            "positive evidence set differs from the accepted-sequence prefix of length R"
        )
    return {
        "k_cap": k_cap,
        "complete_sequence_audited": k_cap is None,
        "geometry_nms_capacity_g": geometry_nms_capacity,
        "audited_prefix_count": capped_count,
        "receiver_evidence_budget_r": receiver_budget,
        "accepted_prefix_ids": capped_ids,
        "positive_evidence_ids": positive_ids,
        "positive_prefix_exact_match": True,
    }


def rd_receiver_capacity(accepted) -> dict:
    """Return uncapped R after proving that positives are one exact prefix."""
    result = receiver_evidence_budget(accepted, k_cap=None)
    result["format"] = "i2_rd_v1_receiver_capacity_v1"
    result["receiver_capacity"] = result["receiver_evidence_budget_r"]
    return result


def rd_donor_capacity(rows, *, receiver_anchor_uids, live_uids=None) -> dict:
    """Validate and count the complete structurally eligible donor ordering.

    The caller supplies the frozen D0/B1 ordering.  No rendering or quality
    threshold is accepted by this function, deliberately keeping performance
    signals out of D.
    """
    anchors = {int(value) for value in receiver_anchor_uids}
    live = None if live_uids is None else {int(value) for value in live_uids}
    selected, excluded = [], []
    seen = set()
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise AdaptiveKError("donor evidence row must be an object")
        try:
            uid = int(row["uid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptiveKError("donor evidence contains an invalid UID") from exc
        if uid in seen:
            raise AdaptiveKError(f"donor evidence contains duplicate UID {uid}")
        seen.add(uid)
        reason = None
        if uid in anchors:
            reason = "receiver_anchor_uid"
        elif live is not None and uid not in live:
            reason = "not_live_in_common_source"
        elif row.get("birth_type") not in {"clone", "split"}:
            reason = "root_or_invalid_birth_type"
        if reason:
            excluded.append({"uid": uid, "input_rank": rank, "reason": reason})
            continue
        selected.append({**row, "rd_rank": len(selected) + 1})
    return {
        "format": "i2_rd_v1_donor_capacity_v1",
        "complete_eligible_input_count": len(rows),
        "excluded_count": len(excluded),
        "donor_capacity": len(selected),
        "ranked_eligible": selected,
        "excluded": excluded,
        "coverage_complete": True,
        "performance_gates_used": False,
    }


def make_rd_decision(*, scene: str, source_checkpoint_sha256: str,
                     receiver_audit: dict, donor_audit: dict, evidence: dict,
                     input_bindings: dict | None = None,
                     determined: bool = True, blocker: str | None = None) -> dict:
    """Create an RD-v1 Decision; never substitutes a bounded fallback."""
    if not determined:
        selected = "UNKNOWN"
        route = "blocked_no_execution"
    else:
        try:
            r = int(receiver_audit["receiver_capacity"])
            d = int(donor_audit["donor_capacity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptiveKError("RD-v1 capacity is not completely determined") from exc
        if r < 0 or d < 0:
            raise AdaptiveKError("RD-v1 capacity cannot be negative")
        if receiver_audit.get("complete_sequence_audited") is not True:
            raise AdaptiveKError("RD-v1 Receiver enumeration is incomplete")
        if donor_audit.get("coverage_complete") is not True:
            raise AdaptiveKError("RD-v1 Donor enumeration is incomplete")
        selected = min(r, d)
        route = "control_no_intervention" if selected == 0 else "adaptive_plan_available_no_training_started"
    result = {
        "format": "i2_adaptive_k_rd_v1_decision",
        "policy_version": "RD-v1",
        "scene": scene,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "uses_test_views": False,
        "receiver_capacity": receiver_audit.get("receiver_capacity", "UNKNOWN"),
        "donor_capacity": donor_audit.get("donor_capacity", "UNKNOWN"),
        "selected_k": selected,
        "execution_route": route,
        "receiver_audit": receiver_audit,
        "evidence": evidence,
        "input_bindings": input_bindings or {},
        "failure_reasons": ([blocker or "capacity_not_determined"] if selected == "UNKNOWN" else []),
    }
    return result


def contiguous_pass_capacity(rows, *, pass_field="safe") -> tuple[int, str]:
    """Return the safe prefix length and its deterministic stopping reason."""
    capacity = 0
    for expected, row in enumerate(rows, start=1):
        if int(row.get("k", -1)) != expected:
            return capacity, f"missing_or_out_of_order_prefix_k{expected}"
        if row.get(pass_field) is not True:
            return capacity, f"first_failed_prefix_k{expected}"
        capacity = expected
    return capacity, "all_audited_prefixes_passed"


def require_artifact_hash(reference: dict, *, label: str = "artifact") -> Path:
    """Resolve and verify a {path, sha256} reference, failing closed."""
    if not isinstance(reference, dict):
        raise AdaptiveKError(f"{label} reference must be an object")
    path_value, expected = reference.get("path"), reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str) or len(expected) != 64:
        raise AdaptiveKError(f"{label} reference has no valid path/hash binding")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise AdaptiveKError(f"{label} is missing: {path}")
    if sha256(path).lower() != expected.lower():
        raise AdaptiveKError(f"{label} hash mismatch: {path}")
    return path


def make_decision(*, scene: str, source_checkpoint_sha256: str,
                  receiver_capacity: int, donor_capacity: int,
                  initialization_rows, evidence, determined: bool,
                  blocker: str | None = None, k_cap: int = 20,
                  receiver_audit: dict | None = None):
    init_capacity, init_stop = contiguous_pass_capacity(
        initialization_rows, pass_field="safe"
    )
    preinitialization_limit = min(k_cap, receiver_capacity, donor_capacity)
    incomplete_initialization = bool(
        determined and preinitialization_limit > 0 and (
            init_stop.startswith("missing_or_out_of_order")
            or (init_stop == "all_audited_prefixes_passed"
                and init_capacity < preinitialization_limit)
        )
    )
    if not determined or incomplete_initialization:
        selected = "UNKNOWN"
        route = "blocked_no_execution"
        stop_reason = blocker or (
            "initialization_capacity_not_determined"
            if incomplete_initialization else "capacity_not_determined"
        )
    else:
        selected = min(k_cap, receiver_capacity, donor_capacity, init_capacity)
        route = "control_no_intervention" if selected == 0 else "adaptive_plan_available_no_training_started"
        if selected < min(receiver_capacity, donor_capacity):
            stop_reason = init_stop
        elif donor_capacity < receiver_capacity:
            stop_reason = "donor_contiguous_safe_capacity"
        elif receiver_capacity < donor_capacity:
            stop_reason = "receiver_evidence_budget"
        elif k_cap < min(receiver_capacity, donor_capacity, init_capacity):
            stop_reason = "uniform_k_cap"
        else:
            stop_reason = "bounded_capacities_equal"
    return {
        "format": "i2_adaptive_k_r0_v1_decision",
        "policy_version": "R0-v1",
        "scene": scene,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "uses_test_views": False,
        "selected_k": selected,
        "execution_route": route,
        "receiver_capacity": receiver_capacity,
        "k_cap": k_cap,
        "donor_contiguous_safe_capacity": donor_capacity,
        "initialization_contiguous_safe_capacity": init_capacity,
        "stop_reason": stop_reason,
        "failure_reasons": [stop_reason] if selected == "UNKNOWN" else [],
        "evidence": evidence,
        "receiver_audit": receiver_audit,
    }


def write_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
