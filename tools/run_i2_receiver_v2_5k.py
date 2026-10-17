#!/usr/bin/env python3
"""Guarded Receiver-v2 K20 5k launcher; defaults to preflight-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_i2_receiver_v2_1252 import (
    AUDIT, DATASET, DONORS, EVIDENCE, PLAN, ROOT, SOURCE, SOURCE_SHA256,
    VIEW_MANIFEST, run_logged, sha256,
)
from utils.i2_active_pilot import load_active_plan

OUTPUT = ROOT / "output/I2_D1_ReceiverV2_K20_5K"


def preflight():
    verified = load_active_plan(PLAN, requested_budget=20, source_checkpoint=SOURCE)
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    short_meta = json.loads(
        (ROOT / "output/I2_D1_ReceiverV2_K20_1252/launch_meta.json").read_text(encoding="utf-8")
    )
    if (
        sha256(SOURCE) != SOURCE_SHA256
        or verified["_donor_uids"] != DONORS
        or evidence.get("test_views_used") is not False
        or audit.get("status") != "GO"
        or audit.get("test_views_used") is not False
        or audit.get("receiver_v2_plan_sha256") != sha256(PLAN)
        or audit.get("receiver_v2_evidence_sha256") != sha256(EVIDENCE)
        or short_meta.get("status") != "PASS"
        or short_meta["frozen_inputs"].get("plan_sha256") != sha256(PLAN)
    ):
        raise RuntimeError("Receiver-v2 5k preflight failed")
    return {
        "source_checkpoint_sha256": sha256(SOURCE), "plan_sha256": sha256(PLAN),
        "evidence_sha256": sha256(EVIDENCE), "initialization_audit_sha256": sha256(AUDIT),
        "short_run_meta_sha256": sha256(ROOT / "output/I2_D1_ReceiverV2_K20_1252/launch_meta.json"),
        "donor_uids": DONORS, "test_views_used_for_initialization": False,
    }


def command():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    frozen = preflight()
    train_command = command()
    if not args.execute:
        print(json.dumps({
            "status": "PASS", "mode": "preflight_only", "output": str(OUTPUT),
            "frozen_inputs": frozen, "command": train_command,
            "note": "No output directory was created and no GPU training was started.",
        }, indent=2))
        return
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    metadata = {
        "format": "i2_receiver_v2_5k_launch_v1", "status": "RUNNING",
        "source_iteration": 1201, "final_iteration": 5000,
        "effective_optimizer_updates": 3798, "frozen_inputs": frozen,
        "argv": train_command,
    }
    meta_path = OUTPUT / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(train_command, OUTPUT / "train.log")
        evaluation = [sys.executable, "-u", "tools/evaluate_i2_5k.py", "-m", str(OUTPUT), "--iteration", "5000"]
        metadata["evaluation_argv"] = evaluation
        metadata["evaluation_seconds"] = run_logged(evaluation, OUTPUT / "evaluation.log")
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"; metadata["error"] = repr(exc); raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
