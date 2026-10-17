"""Export B1/C candidate tables and a UID-keyed D input manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.i2_resource_analysis import (
    analyze_observation_checkpoint,
    sha256_file,
    unwrap_checkpoint,
    write_b1_outputs,
)
from utils.i2_future_utility import FUTURE_UTILITY_CHECKPOINT_VERSION


FEATURES = (
    "visible_rate",
    "unique_real_views",
    "opacity_retention_ratio",
    "scale_ratio_from_birth",
    "xyz_shift_from_birth",
)
OUTCOME_DEFINITIONS = {
    "raw_uid_survival": {"alive": 1, "split_replaced": 0, "training_prune": 0},
    "true_prune_avoidance": {"alive": 1, "split_replaced": 1, "training_prune": 0},
    "lifecycle_transition": ["alive", "split_replaced", "training_prune"],
}


def _flatten_c_record(record):
    birth = int(record["birth_iteration"])
    death = record.get("death_iteration")
    terminal_age = None if death is None else int(death) - birth
    snapshots = record.get("age_snapshots") or {}
    transition = None
    outcome_status = "unresolved_age500"
    if "500" in snapshots and terminal_age is not None and terminal_age <= 500:
        raise ValueError(
            f"uid={record.get('uid', '?')}: age500 snapshot conflicts with "
            f"terminal_age={terminal_age}"
        )
    if "500" in snapshots:
        transition = "alive"
        outcome_status = "resolved"
    elif terminal_age is not None and terminal_age <= 500:
        reason = record.get("death_reason")
        if reason in ("split_replaced", "training_prune"):
            transition = reason
            outcome_status = "resolved"
        else:
            outcome_status = "unsupported_death_reason"
    row = {
        "uid": int(record["uid"]),
        "birth_iteration": birth,
        "birth_type": record["birth_type"],
        "source_uid": int(record["source_uid"]),
        "generation": int(record["generation"]),
        "birth_xyz": json.dumps(record["birth_xyz"], separators=(",", ":")),
        "birth_opacity": float(record["birth_opacity"]),
        "birth_scaling": json.dumps(record["birth_scaling"], separators=(",", ":")),
        "alive_at_checkpoint": bool(record["alive"]),
        "death_iteration": death,
        "death_reason": record.get("death_reason"),
        "terminal_age": terminal_age,
        "lifecycle_transition": transition,
        "raw_uid_survival": None if transition is None else int(transition == "alive"),
        "true_prune_avoidance": (
            None if transition is None else int(transition != "training_prune")
        ),
        "outcome_status": outcome_status,
    }
    for age in (100, 500):
        snapshot = snapshots.get(str(age))
        row[f"age{age}_status"] = "available" if snapshot else "missing"
        for field in FEATURES + (
            "iteration", "age", "real_view_opportunities", "dropout_retained_hits",
            "screen_supported_hits", "screen_support_rate", "opacity", "scale_mean",
        ):
            row[f"age{age}_{field}"] = None if snapshot is None else snapshot.get(field)
    terminal = record.get("terminal_state")
    row["terminal_state_status"] = "available" if terminal else "not_applicable_live"
    if terminal:
        for field in (
            "real_view_opportunities", "dropout_retained_hits", "screen_supported_hits",
            "unique_real_views", "visible_rate", "opacity", "opacity_retention_ratio",
            "xyz_shift_from_birth", "scale_mean", "scale_ratio_from_birth",
        ):
            row[f"terminal_{field}"] = terminal.get(field)
    return row


def build_c_candidates(model_state):
    if len(model_state) != 13 or not isinstance(model_state[12], dict):
        return [], "missing_c_checkpoint_extension"
    state = model_state[12].get("i2_future_utility")
    if not isinstance(state, dict):
        return [], "missing_c_checkpoint_extension"
    if (
        state.get("version") != FUTURE_UTILITY_CHECKPOINT_VERSION
        or not isinstance(state.get("records"), list)
    ):
        raise ValueError("Malformed i2_future_utility checkpoint extension")
    rows = [_flatten_c_record(record) for record in state["records"]]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("C candidate records contain duplicate UIDs")
    return rows, "available" if rows else "available_empty_cohort"


def _rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def _spearman(feature, outcome):
    if len(feature) < 2 or np.ptp(feature) == 0 or np.ptp(outcome) == 0:
        return None
    x = _rankdata(feature)
    y = _rankdata(outcome)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return None if denominator == 0 else float(np.sum(x * y) / denominator)


def _q4_minus_q1(feature, outcome):
    order = np.argsort(feature, kind="mergesort")
    quartile = np.minimum(3, np.arange(len(order)) * 4 // len(order))
    q1 = outcome[order][quartile == 0]
    q4 = outcome[order][quartile == 3]
    if not len(q1) or not len(q4):
        return None
    return float(q4.mean() - q1.mean())


def _grouped_bootstrap(feature, outcome, groups, metric, reps, seed):
    unique = np.unique(groups)
    if reps <= 0 or len(unique) < 2:
        return {"status": "insufficient_groups", "valid_replicates": 0, "ci95": None}
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        value = metric(feature[chosen], outcome[chosen])
        if value is not None and np.isfinite(value):
            values.append(value)
    return {
        "status": "ok" if values else "insufficient_variation",
        "valid_replicates": len(values),
        "ci95": None if not values else [
            float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
        ],
    }


def analyze_c_rows(rows, *, bootstrap_reps=500, seed=20261014):
    resolved = [row for row in rows if row["outcome_status"] == "resolved"]
    transitions = {name: sum(row["lifecycle_transition"] == name for row in resolved)
                   for name in OUTCOME_DEFINITIONS["lifecycle_transition"]}
    report = {
        "format": "i2_c_future_utility_analysis_v1",
        "outcome_definitions": OUTCOME_DEFINITIONS,
        "interpretation_guard": (
            "true_prune_avoidance is a lifecycle proxy, not a direct reconstruction-quality label"
        ),
        "age100_predictor_contract": {
            "allowed_features": [f"age100_{name}" for name in FEATURES],
            "future_death_fields_excluded": True,
        },
        "cohort_count": len(rows),
        "resolved_outcome_count": len(resolved),
        "transition_counts": transitions,
        "analyses": {},
    }
    for age in (100,):
        for feature_name in FEATURES:
            available = [row for row in resolved if row[f"age{age}_{feature_name}"] is not None]
            key = f"age{age}_{feature_name}"
            result = {"n": len(available), "outcomes": {}}
            for outcome_name in ("raw_uid_survival", "true_prune_avoidance"):
                if len(available) < 4:
                    result["outcomes"][outcome_name] = {
                        "status": "insufficient_samples", "spearman": None,
                        "q4_minus_q1": None, "grouped_bootstrap": None,
                    }
                    continue
                feature = np.asarray([row[f"age{age}_{feature_name}"] for row in available], dtype=np.float64)
                outcome = np.asarray([row[outcome_name] for row in available], dtype=np.float64)
                groups = np.asarray([row["birth_iteration"] for row in available], dtype=np.int64)
                if np.ptp(outcome) == 0:
                    result["outcomes"][outcome_name] = {
                        "status": "single_outcome", "spearman": None,
                        "q4_minus_q1": None, "grouped_bootstrap": None,
                    }
                    continue
                result["outcomes"][outcome_name] = {
                    "status": "ok",
                    "spearman": _spearman(feature, outcome),
                    "q4_minus_q1": _q4_minus_q1(feature, outcome),
                    "grouped_bootstrap": {
                        "grouping": "birth_iteration",
                        "spearman": _grouped_bootstrap(
                            feature, outcome, groups, _spearman, bootstrap_reps, seed
                        ),
                        "q4_minus_q1": _grouped_bootstrap(
                            feature, outcome, groups, _q4_minus_q1, bootstrap_reps, seed + 1
                        ),
                    },
                }
            report["analyses"][key] = result
    report["status"] = (
        "insufficient_data" if not resolved
        else "insufficient_outcome_variation" if len([v for v in transitions.values() if v]) < 2
        else "analyzed"
    )
    return report


def _write_c(rows, output_dir):
    path = output_dir / "c_utility_candidates.csv"
    fields = list(rows[0]) if rows else [
        "uid", "birth_iteration", "birth_type", "source_uid", "generation",
        "outcome_status", "lifecycle_transition",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fingerprint(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def run(checkpoint, output_dir, *, bootstrap_reps=500, seed=20261014):
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state, saved_iteration = unwrap_checkpoint(payload)
    b1 = analyze_observation_checkpoint(payload)
    b1_path, b1_summary_path = write_b1_outputs(
        b1, output_dir, source_checkpoint=str(checkpoint)
    )
    c_rows, c_state = build_c_candidates(model_state)
    c_path = _write_c(c_rows, output_dir)
    c_analysis = analyze_c_rows(c_rows, bootstrap_reps=bootstrap_reps, seed=seed)
    c_analysis["checkpoint_data_status"] = c_state
    c_report_path = output_dir / "c_utility_analysis.json"
    c_report_path.write_text(json.dumps(c_analysis, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "format": "i2_d_input_manifest_v1",
        "source_checkpoint": _fingerprint(checkpoint),
        "iteration": saved_iteration,
        "gaussian_count": b1["gaussian_count"],
        "uid_alignment": b1["uid_alignment"],
        "b0_data_version": model_state[12]["i2_observation"].get("version"),
        "c_data_version": (
            None if c_state.startswith("missing")
            else model_state[12]["i2_future_utility"].get("version")
        ),
        "diagnostic_status": {
            "b1": "available",
            "c": c_state,
            "c_analysis": c_analysis["status"],
            "active_intervention_performed": False,
        },
        "candidate_tables": {
            "b1": _fingerprint(b1_path),
            "c": _fingerprint(c_path),
        },
        "reports": {
            "b1": _fingerprint(b1_summary_path),
            "c": _fingerprint(c_report_path),
        },
        "join_key": "uid",
    }
    manifest_path = output_dir / "d_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    manifest = run(
        args.checkpoint, args.output_dir,
        bootstrap_reps=args.bootstrap_reps, seed=args.seed,
    )
    print(json.dumps(manifest["diagnostic_status"], sort_keys=True))


if __name__ == "__main__":
    main()
