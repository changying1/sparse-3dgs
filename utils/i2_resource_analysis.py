"""Offline B1 observation diagnostics for A0+B0 checkpoints.

This module is deliberately read-only: candidate labels are diagnostic inputs
for a later intervention and never mutate or prune Gaussian state.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from utils.i2_lifecycle import CODE_TO_BIRTH_TYPE


B1_TABLE_VERSION = "i2_b1_observation_candidates_v1"


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """FSGS-compatible zero-safe elementwise ratio."""
    result = torch.zeros_like(numerator, dtype=torch.float64)
    valid = denominator > 0
    result[valid] = numerator[valid].to(torch.float64) / denominator[valid].to(
        torch.float64
    )
    return result


def compute_support_coverage(
    supported_distinct_views: torch.Tensor, total_real_views: int
) -> torch.Tensor:
    """Fraction of frozen real training views that ever supported each row."""
    if total_real_views <= 0:
        raise ValueError("total_real_views must be positive")
    return supported_distinct_views.to(torch.float64) / float(total_real_views)


def compute_visibility_persistence(
    screen_supported_hits: torch.Tensor, real_view_opportunities: torch.Tensor
) -> torch.Tensor:
    """FSGS screen-support persistence over real-view opportunities.

    This is intentionally separate from Dropout retention and is not the
    historical radius > 2 px signal.
    """
    return safe_ratio(screen_supported_hits, real_view_opportunities)


def _rank_low_cohort(primary, secondary, uid, eligible, fraction):
    """Port of FSGS rank-cohort selection with UID-stable tie breaking."""
    indices = eligible.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
    low = torch.zeros_like(eligible, dtype=torch.bool)
    if not indices:
        return low, False, 0.0, 0.0
    primary_cpu = primary.detach().cpu()
    secondary_cpu = secondary.detach().cpu()
    uid_cpu = uid.detach().cpu()
    primary_spread = float(primary_cpu[indices].max() - primary_cpu[indices].min())
    secondary_spread = float(secondary_cpu[indices].max() - secondary_cpu[indices].min())
    degenerate = primary_spread <= 1.0e-12 and secondary_spread <= 1.0e-12
    if degenerate:
        return low, True, primary_spread, secondary_spread
    cohort_size = min(int(len(indices) * float(fraction)), len(indices) // 2)
    decorated = sorted(
        (float(primary_cpu[i]), float(secondary_cpu[i]), int(uid_cpu[i]), i)
        for i in indices
    )
    if cohort_size:
        low[torch.tensor([entry[-1] for entry in decorated[:cohort_size]])] = True
    return low, False, primary_spread, secondary_spread


def unwrap_checkpoint(payload):
    """Return ``(model_state, iteration)`` from a saved training checkpoint."""
    iteration = None
    model_state = payload
    if (
        isinstance(payload, (tuple, list))
        and len(payload) == 2
        and isinstance(payload[0], (tuple, list))
    ):
        model_state, iteration = payload
    if not isinstance(model_state, (tuple, list)) or len(model_state) not in (12, 13):
        raise ValueError("Unsupported Gaussian checkpoint payload")
    return model_state, iteration


def _require_i2_states(model_state):
    if len(model_state) != 13 or not isinstance(model_state[12], dict):
        raise ValueError("B1 requires a 13-field A0+B0 checkpoint")
    extension = model_state[12]
    lifecycle = extension.get("i2_lifecycle")
    observation = extension.get("i2_observation")
    if not isinstance(lifecycle, dict) or not isinstance(observation, dict):
        raise ValueError("B1 requires both i2_lifecycle and i2_observation metadata")
    return lifecycle, observation, extension


def analyze_observation_checkpoint(payload, *, iteration=None, low_quantile=0.25):
    """Audit one checkpoint and build B1 rows plus cohort summaries."""
    model_state, saved_iteration = unwrap_checkpoint(payload)
    lifecycle, observation, extension = _require_i2_states(model_state)
    if iteration is None:
        iteration = saved_iteration
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Checkpoint iteration is required for B1 age calculation")
    if not 0.0 <= float(low_quantile) <= 1.0:
        raise ValueError("low_quantile must be in [0, 1]")

    uid = lifecycle.get("uid")
    birth_iter = lifecycle.get("birth_iter")
    birth_type = lifecycle.get("birth_type")
    source_uid = lifecycle.get("source_uid")
    generation = lifecycle.get("generation")
    obs_uid = observation.get("uid")
    opportunities = observation.get("real_view_opportunities")
    retained = observation.get("dropout_retained_hits")
    supported = observation.get("screen_supported_hits")
    history = observation.get("screen_supported_view_history")
    tensors_1d = (uid, birth_iter, birth_type, source_uid, generation, obs_uid,
                  opportunities, retained, supported)
    if any(not torch.is_tensor(value) or value.ndim != 1 for value in tensors_1d):
        raise ValueError("Malformed row-aligned A0/B0 checkpoint tensors")
    count = uid.numel()
    if any(value.numel() != count for value in tensors_1d):
        raise ValueError("A0/B0 checkpoint rows are not aligned")
    if uid.dtype != torch.int64 or obs_uid.dtype != torch.int64:
        raise ValueError("A0/B0 UID tensors must use int64")
    if not torch.equal(uid.cpu(), obs_uid.cpu()):
        raise ValueError("B0 observation UID order does not match A0 lifecycle UID order")
    if len(set(uid.detach().cpu().tolist())) != count:
        raise ValueError("Live lifecycle UIDs are not unique")
    if (
        not torch.is_tensor(history)
        or history.dtype != torch.bool
        or history.ndim != 2
        or history.shape[0] != count
    ):
        raise ValueError("Malformed B0 supported-view history")
    if bool(torch.any(opportunities < retained).item()) or bool(
        torch.any(retained < supported).item()
    ):
        raise ValueError("B0 observation counters violate opportunity/retention/support order")

    views = tuple(observation.get("view_identifiers") or ())
    if history.shape[1] != len(views) or not views:
        raise ValueError("B0 view identifiers do not match supported-view history")
    unique_views = history.sum(dim=1, dtype=torch.int64)
    dropout_rate = safe_ratio(retained, opportunities)
    screen_rate = compute_visibility_persistence(supported, opportunities)
    support_coverage = compute_support_coverage(unique_views, len(views))

    eligible = (opportunities > 0) & (supported > 0)
    threshold = None
    if bool(torch.any(eligible).item()):
        threshold = float(torch.quantile(screen_rate[eligible], float(low_quantile)).item())
    ranked_low, signal_degenerate, primary_spread, coverage_spread = _rank_low_cohort(
        screen_rate, support_coverage, uid, eligible, float(low_quantile)
    )
    no_real_opportunity = opportunities == 0
    no_retained_opportunity = (opportunities > 0) & (retained == 0)
    no_screen_support = (opportunities > 0) & (supported == 0)
    low_screen = ranked_low | no_screen_support
    age = iteration - birth_iter
    if bool(torch.any(age < 0).item()):
        raise ValueError("Checkpoint iteration predates a live Gaussian birth")

    cpu = [value.detach().cpu().tolist() for value in (
        uid, birth_iter, birth_type, source_uid, generation, age, opportunities,
        retained, supported, unique_views, dropout_rate, screen_rate,
        support_coverage, low_screen, no_real_opportunity, no_retained_opportunity,
        no_screen_support,
    )]
    rows = []
    for values in zip(*cpu):
        code = int(values[2])
        if code not in CODE_TO_BIRTH_TYPE:
            raise ValueError(f"Unknown lifecycle birth type code: {code}")
        rows.append({
            "uid": int(values[0]),
            "birth_iteration": int(values[1]),
            "birth_type": CODE_TO_BIRTH_TYPE[code],
            "source_uid": int(values[3]),
            "generation": int(values[4]),
            "age": int(values[5]),
            "real_view_opportunities": int(values[6]),
            "dropout_retained_hits": int(values[7]),
            "screen_supported_hits": int(values[8]),
            "unique_supported_real_views": int(values[9]),
            "dropout_retention_rate": float(values[10]),
            "screen_support_rate": float(values[11]),
            "real_view_coverage_rate": float(values[12]),
            "low_screen_support_candidate": bool(values[13]),
            "observation_status": (
                "no_real_view_opportunity" if values[14]
                else "no_dropout_retained_opportunity" if values[15]
                else "no_screen_support" if values[16]
                else "low_screen_support" if values[13]
                else "observed"
            ),
            "historical_radius_gt_2px_comparable": False,
        })

    strata = _stratify(rows)
    return {
        "format": B1_TABLE_VERSION,
        "iteration": iteration,
        "gaussian_count": count,
        "uid_alignment": "exact",
        "view_count": len(views),
        "low_support_definition": {
            "metric": "screen_supported_hits / real_view_opportunities",
            "quantile": float(low_quantile),
            "threshold": threshold,
            "selection": "FSGS rank cohort with real-view coverage then UID tie-break",
            "zero_support_always_candidate": True,
            "signal_degenerate": signal_degenerate,
            "primary_spread": primary_spread,
            "coverage_spread": coverage_spread,
        },
        "semantic_guards": {
            "low_screen_support_is_future_utility": False,
            "low_screen_support_is_image_quality_contribution": False,
            "historical_radius_gt_2px_available": False,
            "historical_radius_gt_2px_reason": (
                "B0 records radii > 0 support only; the old >2px signal cannot be reconstructed"
            ),
        },
        "candidate_count": sum(row["low_screen_support_candidate"] for row in rows),
        "rows": rows,
        "strata": strata,
        "c_utility_state_present": "i2_future_utility" in extension,
    }


def _age_bucket(age):
    if age < 100:
        return "age_lt_100"
    if age < 500:
        return "age_100_499"
    return "age_ge_500"


def _stratify(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["birth_type"], _age_bucket(row["age"]), row["observation_status"])
        groups[key].append(row)
    output = []
    for (birth_type, age_bucket, support_stratum), members in sorted(groups.items()):
        output.append({
            "birth_type": birth_type,
            "age_bucket": age_bucket,
            "screen_support_stratum": support_stratum,
            "count": len(members),
            "low_screen_support_count": sum(
                row["low_screen_support_candidate"] for row in members
            ),
            "mean_screen_support_rate": sum(
                row["screen_support_rate"] for row in members
            ) / len(members),
            "mean_unique_supported_real_views": sum(
                row["unique_supported_real_views"] for row in members
            ) / len(members),
        })
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_b1_outputs(analysis, output_dir, *, source_checkpoint=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "b1_observation_candidates.csv"
    fields = list(analysis["rows"][0]) if analysis["rows"] else [
        "uid", "birth_iteration", "birth_type", "source_uid", "generation", "age",
        "real_view_opportunities", "dropout_retained_hits", "screen_supported_hits",
        "unique_supported_real_views", "dropout_retention_rate", "screen_support_rate",
        "real_view_coverage_rate", "low_screen_support_candidate", "observation_status",
        "historical_radius_gt_2px_comparable",
    ]
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis["rows"])
    summary = {key: value for key, value in analysis.items() if key != "rows"}
    summary_path = output_dir / "b1_observation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return candidate_path, summary_path
