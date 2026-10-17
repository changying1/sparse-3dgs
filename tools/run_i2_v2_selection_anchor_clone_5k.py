#!/usr/bin/env python3
"""Guarded v2-selection Anchor-Clone K20 5k launcher; preflight-only by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_i2_receiver_v2_1252 import (
    DATASET, DONORS, EVIDENCE, SOURCE, SOURCE_SHA256, VIEW_MANIFEST,
    run_logged, sha256,
)
from utils.i2_active_pilot import load_active_plan

PLAN = (
    ROOT / "reports/i2_v2_selection_anchor_clone_horns_1201_k20"
    / "i2_d1_v2_selection_anchor_clone_k20_plan.json"
)
ORIGINAL_V2_PLAN = (
    ROOT / "reports/i2_receiver_v2_horns_1201_k20/i2_d1_receiver_v2_k20_plan.json"
)
PLAN_AUDIT = PLAN.parent / "plan_correspondence_audit.json"
INITIALIZATION_AUDIT = PLAN.parent / "initialization_audit.json"
SHORT_OUTPUT = ROOT / "output/I2_D1_V2Selection_AnchorClone_K20_1252"
SHORT_META = SHORT_OUTPUT / "launch_meta.json"
OUTPUT = ROOT / "output/I2_D1_V2Selection_AnchorClone_K20_5K"


def preflight() -> dict:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT}")
    verified = load_active_plan(PLAN, requested_budget=20, source_checkpoint=SOURCE)
    plan_audit = json.loads(PLAN_AUDIT.read_text(encoding="utf-8"))
    initialization = json.loads(INITIALIZATION_AUDIT.read_text(encoding="utf-8"))
    short = json.loads(SHORT_META.read_text(encoding="utf-8"))
    short_metrics = json.loads(
        (SHORT_OUTPUT / "formal_metrics_1252.json").read_text(encoding="utf-8")
    )
    short_event = json.loads(
        (SHORT_OUTPUT / "active_event.json").read_text(encoding="utf-8")
    )
    source_code_unchanged = all(
        (ROOT / name).is_file() and sha256(ROOT / name) == expected
        for name, expected in short.get("source_code_sha256", {}).items()
    )
    expected_manifest_sha = verified["input_audit"]["receiver_protocol"]["view_manifest_sha256"]
    if (
        sha256(SOURCE) != SOURCE_SHA256
        or verified["_donor_uids"] != DONORS
        or verified.get("_receiver_v2_selection_anchor_clone") is not True
        or sha256(VIEW_MANIFEST) != expected_manifest_sha
        or plan_audit.get("status") != "PASS"
        or plan_audit.get("new_anchor_clone_plan", {}).get("sha256") != sha256(PLAN)
        or initialization.get("status") != "GO"
        or initialization.get("test_views_used") is not False
        or short.get("status") != "PASS"
        or not source_code_unchanged
        or short_metrics.get("checkpoint_audit", {}).get("pass") is not True
        or short_event.get("format")
        != "i2_active_event_v2_selection_anchor_clone_v1"
        or not all(short_event.get("anchor_raw_parameter_copy", {}).values())
        or not short_event.get("lifecycle_source_uid_matches_anchor_order")
        or short.get("frozen_inputs", {}).get("plan_sha256") != sha256(PLAN)
        or short.get("frozen_inputs", {}).get("view_manifest_sha256") != expected_manifest_sha
    ):
        raise RuntimeError("Anchor-Clone 5k frozen preflight failed")
    return {
        "source_checkpoint_sha256": sha256(SOURCE),
        "plan_sha256": sha256(PLAN),
        "original_receiver_v2_plan_sha256": sha256(ORIGINAL_V2_PLAN),
        "receiver_v2_evidence_sha256": sha256(EVIDENCE),
        "view_manifest_sha256": sha256(VIEW_MANIFEST),
        "plan_correspondence_audit_sha256": sha256(PLAN_AUDIT),
        "initialization_audit_sha256": sha256(INITIALIZATION_AUDIT),
        "short_launch_meta_sha256": sha256(SHORT_META),
        "short_formal_metrics_sha256": sha256(SHORT_OUTPUT / "formal_metrics_1252.json"),
        "short_active_event_sha256": sha256(SHORT_OUTPUT / "active_event.json"),
        "short_source_code_hashes_reverified": source_code_unchanged,
        "donor_uids": DONORS,
        "receiver_candidate_ids": verified["_receiver_candidate_ids"],
        "anchor_uids": verified["_anchor_uids"],
        "test_views_used_for_initialization": False,
    }


def command() -> list[str]:
    return [
        sys.executable, "-u", "train.py", "-s", str(DATASET), "-m", str(OUTPUT),
        "--dataset_name", "LLFF", "--eval", "--n_views", "3", "--resolution", "8",
        "--sh_degree", "1", "--view_manifest", str(VIEW_MANIFEST), "--iterations", "5000",
        "--dropout_factor", "0.2", "--sigma_noise", "0.0", "--densify_from_iter", "1000",
        "--densification_interval", "100", "--densify_grad_threshold", "0.0002",
        "--oe_structural_weight", "0.1", "--oe_structural_start_iter", "1",
        "--oe_structural_end_iter", "1201", "--oe_structure_gate_mode", "stable_evidence",
        "--oe_stable_quantile", "0.8", "--oe_preserve_baseline_densification_stats",
        "--oe_depth_cache_dir",
        r"D:\3DGS\Binocular3DGS + Dropout\output\I1_Horns_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "--checkpoint_iterations", "2000", "3500", "5000",
        "--test_iterations", "1500", "2000", "3000", "4000", "5000",
        "--save_iterations", "5000", "--enable_i2_lifecycle_tracking",
        "--enable_i2_observation_tracking", "--start_checkpoint", str(SOURCE),
        "--enable_i2_active_pilot", "--i2_active_plan", str(PLAN),
        "--i2_active_budget", "20", "--i2_active_event_log", str(OUTPUT / "active_event.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    frozen = preflight()
    train_command = command()
    if not args.execute:
        print(json.dumps({
            "status": "PASS", "mode": "preflight_only", "output": str(OUTPUT),
            "target_output_absent": not OUTPUT.exists(), "frozen_inputs": frozen,
            "command": train_command,
            "protocol": {
                "a0_lifecycle": True, "b0_observation": True,
                "c_future_utility": False, "i1_structural_loss": False,
                "checkpoint_iterations": [2000, 3500, 5000],
                "test_iterations": [1500, 2000, 3000, 4000, 5000],
                "final_ply_iteration": 5000, "formal_full_float_metrics": True,
            },
            "note": "No output directory was created and no GPU training was started.",
        }, indent=2))
        return

    OUTPUT.mkdir(parents=True)
    metadata = {
        "format": "i2_v2_selection_anchor_clone_5k_launch_v1",
        "status": "RUNNING", "source_iteration": 1201, "final_iteration": 5000,
        "effective_optimizer_updates": 3798, "frozen_inputs": frozen,
        "argv": train_command,
    }
    meta_path = OUTPUT / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(train_command, OUTPUT / "train.log")
        evaluation = [
            sys.executable, "-u", "tools/evaluate_i2_5k.py",
            "-m", str(OUTPUT), "--iteration", "5000",
        ]
        metadata["evaluation_argv"] = evaluation
        metadata["evaluation_seconds"] = run_logged(evaluation, OUTPUT / "evaluation.log")
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["error"] = repr(exc)
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
