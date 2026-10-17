#!/usr/bin/env python3
"""Audited v2-selection Anchor-Clone K20 short run from 1201 to 1252."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_i2_receiver_v2_1252 import (
    BASELINE,
    DATASET,
    DONORS,
    EVIDENCE,
    SOURCE,
    SOURCE_SHA256,
    VIEW_MANIFEST,
    run_logged,
    sha256,
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
OUTPUT = ROOT / "output/I2_D1_V2Selection_AnchorClone_K20_1252"


def preflight() -> dict:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT}")
    verified = load_active_plan(PLAN, requested_budget=20, source_checkpoint=SOURCE)
    plan_audit = json.loads(PLAN_AUDIT.read_text(encoding="utf-8"))
    initialization = json.loads(INITIALIZATION_AUDIT.read_text(encoding="utf-8"))
    if (
        sha256(SOURCE) != SOURCE_SHA256
        or verified["_donor_uids"] != DONORS
        or verified.get("_receiver_v2_selection_anchor_clone") is not True
        or plan_audit.get("status") != "PASS"
        or plan_audit.get("new_anchor_clone_plan", {}).get("sha256") != sha256(PLAN)
        or initialization.get("status") != "GO"
        or initialization.get("test_views_used") is not False
        or initialization.get("anchor_clone_plan_sha256") != sha256(PLAN)
        or initialization.get("receiver_v2_plan_sha256") != sha256(ORIGINAL_V2_PLAN)
        or sha256(VIEW_MANIFEST)
        != verified["input_audit"]["receiver_protocol"]["view_manifest_sha256"]
    ):
        raise RuntimeError("Anchor-Clone short-run frozen preflight failed")
    return {
        "source_checkpoint_sha256": sha256(SOURCE),
        "plan_sha256": sha256(PLAN),
        "original_receiver_v2_plan_sha256": sha256(ORIGINAL_V2_PLAN),
        "receiver_v2_evidence_sha256": sha256(EVIDENCE),
        "view_manifest_sha256": sha256(VIEW_MANIFEST),
        "plan_correspondence_audit_sha256": sha256(PLAN_AUDIT),
        "initialization_audit_sha256": sha256(INITIALIZATION_AUDIT),
        "donor_uids": DONORS,
        "receiver_candidate_ids": verified["_receiver_candidate_ids"],
        "anchor_uids": verified["_anchor_uids"],
        "test_views_used_for_initialization": False,
    }


def command() -> list[str]:
    return [
        sys.executable, "-u", "train.py", "-s", str(DATASET), "-m", str(OUTPUT),
        "--dataset_name", "LLFF", "--eval", "--n_views", "3", "--resolution", "8",
        "--sh_degree", "1", "--view_manifest", str(VIEW_MANIFEST),
        "--iterations", "1252", "--dropout_factor", "0.2", "--sigma_noise", "0.0",
        "--densify_from_iter", "1000", "--densification_interval", "100",
        "--densify_grad_threshold", "0.0002", "--oe_structural_weight", "0.1",
        "--oe_structural_start_iter", "1", "--oe_structural_end_iter", "1201",
        "--oe_structure_gate_mode", "stable_evidence", "--oe_stable_quantile", "0.8",
        "--oe_preserve_baseline_densification_stats", "--oe_depth_cache_dir",
        r"D:\3DGS\Binocular3DGS + Dropout\output\I1_Horns_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "--checkpoint_iterations", "1252", "--test_iterations", "1252",
        "--save_iterations", "1252", "--enable_i2_lifecycle_tracking",
        "--enable_i2_observation_tracking", "--start_checkpoint", str(SOURCE),
        "--enable_i2_active_pilot", "--i2_active_plan", str(PLAN),
        "--i2_active_budget", "20", "--i2_active_event_log", str(OUTPUT / "active_event.json"),
    ]


def main() -> None:
    frozen = preflight()
    train_command = command()
    OUTPUT.mkdir(parents=True)
    source_files = (
        "train.py", "scene/gaussian_model.py", "utils/i2_active_pilot.py",
        "tools/build_i2_v2_selection_anchor_clone.py",
        "tools/audit_i2_v2_selection_anchor_clone_initialization.py",
        "tools/run_i2_v2_selection_anchor_clone_1252.py",
    )
    metadata = {
        "format": "i2_v2_selection_anchor_clone_short_launch_v1",
        "status": "RUNNING", "source_iteration": 1201, "final_iteration": 1252,
        "effective_optimizer_updates": 50, "argv": train_command,
        "frozen_inputs": frozen,
        "source_code_sha256": {name: sha256(ROOT / name) for name in source_files},
    }
    meta_path = OUTPUT / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(train_command, OUTPUT / "train.log")
        evaluation = [
            sys.executable, "-u", "tools/evaluate_i2_5k.py",
            "-m", str(OUTPUT), "--iteration", "1252",
        ]
        metadata["evaluation_argv"] = evaluation
        metadata["evaluation_seconds"] = run_logged(evaluation, OUTPUT / "evaluation.log")
        current = json.loads((OUTPUT / "formal_metrics_1252.json").read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        existing_v2 = json.loads(
            (ROOT / "output/I2_D1_ReceiverV2_K20_1252/formal_metrics_1252.json").read_text(
                encoding="utf-8"
            )
        )
        anchor = {
            **current["test"]["aggregate"],
            "gaussians": current["checkpoint_audit"]["gaussian_count"],
        }
        release = baseline["test_metrics"]["ReleaseOnly_K20"]
        v2 = {
            **existing_v2["test"]["aggregate"],
            "gaussians": existing_v2["checkpoint_audit"]["gaussian_count"],
        }
        keys = ("PSNR", "SSIM", "LPIPS", "L1", "gaussians")
        comparison = {
            "format": "i2_v2_selection_anchor_clone_short_comparison_v1",
            "iteration": 1252, "effective_optimizer_updates": 50,
            "test_views_post_freeze_only": True,
            "arms": {"ReleaseOnly_K20": release, "ReceiverV2_K20": v2,
                     "V2Selection_AnchorClone_K20": anchor},
            "anchor_clone_minus_release": {key: anchor[key] - release[key] for key in keys},
            "anchor_clone_minus_receiver_v2": {key: anchor[key] - v2[key] for key in keys},
            "anchor_clone_train": current["train"],
            "randomness_caveat": (
                "Fresh process with seed 0; checkpoint has no RNG state and densification "
                "can cause later random-stream divergence, so pairing is not strict."
            ),
        }
        (OUTPUT / "short_comparison_1252.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        event = json.loads((OUTPUT / "active_event.json").read_text(encoding="utf-8"))
        if (
            event.get("format") != "i2_active_event_v2_selection_anchor_clone_v1"
            or not event.get("budget_conserved")
            or not event.get("alignment", {}).get("pass")
            or not all(event.get("anchor_raw_parameter_copy", {}).values())
            or not event.get("lifecycle_source_uid_matches_anchor_order")
            or not current.get("checkpoint_audit", {}).get("pass")
        ):
            raise RuntimeError("Anchor-Clone short-run output audit failed")
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["error"] = repr(exc)
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
