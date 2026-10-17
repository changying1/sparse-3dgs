#!/usr/bin/env python3
"""Audited one-arm Receiver-v2 K20 short run from iteration 1201 to 1252."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "output/I2_B0_Horns_3v_1201_Growth_z98bpswi/chkpnt1201.pth"
PLAN = ROOT / "reports/i2_receiver_v2_horns_1201_k20/i2_d1_receiver_v2_k20_plan.json"
EVIDENCE = ROOT / "reports/i2_receiver_v2_horns_1201_k20/receiver_v2_geometry_evidence.json"
AUDIT = ROOT / "reports/i2_receiver_v2_horns_1201_k20/initialization_audit.json"
OUTPUT = ROOT / "output/I2_D1_ReceiverV2_K20_1252"
BASELINE = ROOT / "output/I2_D1_K20_50Updates_Paired_16d9or2i/three_way_summary.json"
VIEW_MANIFEST = Path(r"D:\3DGS\Binocular3DGS + Dropout\output\A3_A3_Horns_3v_Baseline_5k_FrozenPDC\view_manifest.json")
DATASET = Path(r"D:\dataset\nerf_llff_data_colmap\horns")
SOURCE_SHA256 = "1d5ee65d9bb1f7b50b2d8ea6bd277e5b52e513b212d0216e3a205e014b428c1f"
V1_PLAN_SHA256 = "2ca1e203848515a1c596368c03b46b04ae6dde85176c81e847ea732cc64ac9c2"
DONORS = [101118, 98638, 105196, 108646, 103207, 105020, 106164, 101472,
          105836, 106674, 100547, 99071, 99564, 99569, 100344, 100413,
          100596, 102469, 102703, 102916]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log_path):
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in process.stdout:
            encoding = sys.stdout.encoding or "utf-8"
            sys.stdout.write(line.encode(encoding, errors="replace").decode(encoding))
            log.write(line); log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}")
    return time.perf_counter() - start


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {OUTPUT}")
    from utils.i2_active_pilot import load_active_plan
    verified = load_active_plan(PLAN, requested_budget=20, source_checkpoint=SOURCE)
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (
        sha256(SOURCE) != SOURCE_SHA256
        or verified["_donor_uids"] != DONORS
        or evidence.get("receiver_v1_plan_sha256") != V1_PLAN_SHA256
        or evidence.get("test_views_used") is not False
        or audit.get("status") != "GO"
        or audit.get("test_views_used") is not False
        or audit.get("receiver_v2_plan_sha256") != sha256(PLAN)
        or audit.get("receiver_v2_evidence_sha256") != sha256(EVIDENCE)
    ):
        raise RuntimeError("Receiver-v2 frozen preflight failed")
    command = [
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
    OUTPUT.mkdir(parents=True)
    metadata = {
        "format": "i2_receiver_v2_short_launch_v1", "status": "RUNNING",
        "source_iteration": 1201, "final_iteration": 1252,
        "effective_optimizer_updates": 50, "argv": command,
        "frozen_inputs": {
            "source_checkpoint_sha256": sha256(SOURCE), "plan_sha256": sha256(PLAN),
            "evidence_sha256": sha256(EVIDENCE), "initialization_audit_sha256": sha256(AUDIT),
            "donor_uids": DONORS, "test_views_used_for_initialization": False,
        },
        "source_code_sha256": {
            name: sha256(ROOT / name) for name in (
                "train.py", "scene/gaussian_model.py", "utils/i2_active_pilot.py",
                "tools/build_i2_receiver_v2.py", "tools/audit_i2_receiver_v2_initialization.py",
                "tools/run_i2_receiver_v2_1252.py",
            )
        },
    }
    meta_path = OUTPUT / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(command, OUTPUT / "train.log")
        evaluation = [sys.executable, "-u", "tools/evaluate_i2_5k.py", "-m", str(OUTPUT), "--iteration", "1252"]
        metadata["evaluation_argv"] = evaluation
        metadata["evaluation_seconds"] = run_logged(evaluation, OUTPUT / "evaluation.log")
        current = json.loads((OUTPUT / "formal_metrics_1252.json").read_text())
        baseline = json.loads(BASELINE.read_text())
        v2 = {**current["test"]["aggregate"], "gaussians": current["checkpoint_audit"]["gaussian_count"]}
        arms = {**baseline["test_metrics"], "ReceiverV2_K20": v2}
        comparison = {
            "format": "i2_receiver_v2_short_comparison_v1", "iteration": 1252,
            "effective_optimizer_updates": 50, "test_views_post_freeze_only": True,
            "arms": arms,
            "receiver_v2_minus_release": {key: v2[key] - arms["ReleaseOnly_K20"][key]
                                           for key in ("PSNR", "SSIM", "LPIPS", "L1", "gaussians")},
            "receiver_v2_minus_v1": {key: v2[key] - arms["Reallocation_K20"][key]
                                      for key in ("PSNR", "SSIM", "LPIPS", "L1", "gaussians")},
            "receiver_v2_train": current["train"],
            "randomness_caveat": "Fresh process with seed 0; checkpoint has no RNG state and cross-arm identity pairing is not strict.",
        }
        (OUTPUT / "short_comparison_1252.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"; metadata["error"] = repr(exc); raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
