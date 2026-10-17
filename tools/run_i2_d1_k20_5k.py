"""Audited launcher for the three I2-D1-K20 1201 -> 5000 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.i2_active_pilot import load_active_plan


SOURCE = ROOT / "output/I2_B0_Horns_3v_1201_Growth_z98bpswi/chkpnt1201.pth"
PLAN = ROOT / "reports/i2_d1_horns_1201_counterfactual_v2/i2_d1_horns_1201_k20_v2_active_plan.json"
VIEW_MANIFEST = Path(r"D:\3DGS\Binocular3DGS + Dropout\output\A3_A3_Horns_3v_Baseline_5k_FrozenPDC\view_manifest.json")
DATASET = Path(r"D:\dataset\nerf_llff_data_colmap\horns")
SOURCE_SHA256 = "1d5ee65d9bb1f7b50b2d8ea6bd277e5b52e513b212d0216e3a205e014b428c1f"
PLAN_SHA256 = "2ca1e203848515a1c596368c03b46b04ae6dde85176c81e847ea732cc64ac9c2"
VIEW_MANIFEST_SHA256 = "a43d544c48f4767369ab37cf9ff82106f1d64365cb0e03d129559d297108586b"
FROZEN_GIT_HEAD = "2fda731ae7390a1bf2ad637aa7dcc7a329b1dba0"
SELECTED_TRAIN_IMAGES = [
    "DJI_20200223_163017_967.jpg",
    "DJI_20200223_163053_863.jpg",
    "DJI_20200223_163225_243.jpg",
]
ARMS = {
    "control": "I2_D1_K20_5K_Control",
    "release": "I2_D1_K20_5K_ReleaseOnly",
    "reallocation": "I2_D1_K20_5K_Reallocation",
}
HASHED_CODE = [
    "train.py", "gaussian_renderer/__init__.py", "scene/gaussian_model.py",
    "utils/i2_active_pilot.py", "utils/i2_lifecycle.py",
    "utils/i2_observation.py", "utils/i2_future_utility.py",
    "utils/i2_randomness.py",
    "tools/evaluate_i2_5k.py", "tools/run_i2_d1_k20_5k.py",
    "script/run_i2_d1_k20_5k.ps1",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def preflight() -> dict:
    observed = {
        "source_checkpoint_sha256": sha256(SOURCE),
        "active_plan_sha256": sha256(PLAN),
        "view_manifest_sha256": sha256(VIEW_MANIFEST),
        "git_head": git_head(),
    }
    expected = {
        "source_checkpoint_sha256": SOURCE_SHA256,
        "active_plan_sha256": PLAN_SHA256,
        "view_manifest_sha256": VIEW_MANIFEST_SHA256,
        "git_head": FROZEN_GIT_HEAD,
    }
    if observed != expected:
        raise RuntimeError(f"Frozen input mismatch: observed={observed}, expected={expected}")
    manifest = json.loads(VIEW_MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    verified_plan = load_active_plan(
        PLAN, requested_budget=20, source_checkpoint=SOURCE
    )
    receiver_views = plan["input_audit"]["receiver_protocol"]["training_camera_ids"]
    if manifest.get("selected_train_images") != SELECTED_TRAIN_IMAGES:
        raise RuntimeError("Frozen selected training views changed")
    if receiver_views != SELECTED_TRAIN_IMAGES:
        raise RuntimeError("Plan receiver views differ from frozen training views")
    if (
        plan.get("format") != "i2_d0_read_only_plan_v2"
        or plan["input_audit"].get("source_checkpoint_sha256") != SOURCE_SHA256
        or plan["input_audit"].get("source_iteration") != 1201
        or plan["configuration"].get("budget_k") != 20
        or len(plan["donors"]["selected"]) != 20
        or len(plan["matches"]) != 20
        or verified_plan["_donor_uids"]
        != [int(row["uid"]) for row in plan["donors"]["selected"]]
    ):
        raise RuntimeError("K20 v2 plan binding/count audit failed")
    return {
        **observed,
        "selected_train_images": SELECTED_TRAIN_IMAGES,
        "donor_uids": [int(row["uid"]) for row in plan["donors"]["selected"]],
        "receiver_candidate_ids": [row["receiver_candidate_id"] for row in plan["matches"]],
    }


def command_for(arm: str, output: Path) -> list[str]:
    command = [
        sys.executable, "-u", "train.py",
        "-s", str(DATASET), "-m", str(output),
        "--dataset_name", "LLFF", "--eval", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1",
        "--view_manifest", str(VIEW_MANIFEST),
        "--iterations", "5000", "--dropout_factor", "0.2",
        "--sigma_noise", "0.0", "--densify_from_iter", "1000",
        "--densification_interval", "100",
        "--densify_grad_threshold", "0.0002",
        "--oe_structural_weight", "0.1",
        "--oe_structural_start_iter", "1",
        "--oe_structural_end_iter", "1201",
        "--oe_structure_gate_mode", "stable_evidence",
        "--oe_stable_quantile", "0.8",
        "--oe_preserve_baseline_densification_stats",
        "--oe_depth_cache_dir", r"D:\3DGS\Binocular3DGS + Dropout\output\I1_Horns_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "--checkpoint_iterations", "2000", "3500", "5000",
        "--test_iterations", "1500", "2000", "3000", "4000", "5000",
        "--save_iterations", "5000",
        "--enable_i2_lifecycle_tracking", "--enable_i2_observation_tracking",
        "--start_checkpoint", str(SOURCE),
    ]
    if arm in {"release", "reallocation"}:
        command.extend([
            "--i2_active_plan", str(PLAN), "--i2_active_budget", "20",
            "--i2_active_event_log", str(output / "active_event.json"),
        ])
        command.append(
            "--enable_i2_release_only" if arm == "release"
            else "--enable_i2_active_pilot"
        )
    return command


def run_logged(command: list[str], log_path: Path) -> float:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Command failed with exit code {return_code}: {command}")
    return time.perf_counter() - start


def write_summary(outputs: dict[str, Path]):
    if set(outputs) != set(ARMS):
        return
    rows = {}
    for arm, output in outputs.items():
        metrics = json.loads((output / "formal_metrics_5000.json").read_text())
        meta = json.loads((output / "launch_meta.json").read_text())
        rows[arm] = {
            **metrics["test"]["aggregate"],
            "gaussian_count": metrics["checkpoint_audit"]["gaussian_count"],
            "training_seconds": meta["training_seconds"],
        }
    def subtract(left, right):
        return {
            key: rows[left][key] - rows[right][key]
            for key in ("PSNR", "SSIM", "LPIPS", "L1", "gaussian_count", "training_seconds")
        }
    payload = {
        "format": "i2_d1_k20_5k_comparison_v1",
        "source_iteration": 1201,
        "final_iteration": 5000,
        "effective_optimizer_updates": 3798,
        "arms": rows,
        "release_minus_control": subtract("release", "control"),
        "reallocation_minus_control": subtract("reallocation", "control"),
        "reallocation_minus_release": subtract("reallocation", "release"),
        "randomness_caveat": (
            "Same seed is not per-UID strict pairing. Release-only records the first "
            "Dropout reference extension; after it, strict cross-arm RNG-stream "
            "alignment must not be claimed."
        ),
    }
    (ROOT / "output/I2_D1_K20_5K_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("all", *ARMS), default="all",
        help="Run one arm or all three in control/release/reallocation order.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    audit = preflight()
    selected = list(ARMS) if args.arm == "all" else [args.arm]
    commands = {
        arm: command_for(arm, ROOT / "output" / ARMS[arm]) for arm in selected
    }
    if args.preflight_only:
        print(json.dumps({
            "status": "PASS", "frozen_inputs": audit,
            "commands": commands,
            "note": "No output directory was created and no GPU training was started.",
        }, indent=2))
        return

    outputs = {}
    for arm in selected:
        output = ROOT / "output" / ARMS[arm]
        if output.exists():
            raise RuntimeError(f"Refusing to reuse existing experiment directory: {output}")
        output.mkdir(parents=True)
        command = commands[arm]
        metadata = {
            "format": "i2_d1_k20_5k_launch_v1",
            "arm": arm,
            "source_iteration": 1201,
            "final_iteration": 5000,
            "effective_optimizer_updates": 3798,
            "frozen_inputs": audit,
            "source_code_sha256": {
                path: sha256(ROOT / path) for path in HASHED_CODE
            },
            "argv": command,
            "training_config": {
                "evaluation_iterations": [1500, 2000, 3000, 4000, 5000],
                "checkpoint_iterations": [2000, 3500, 5000],
                "final_ply_iteration": 5000,
                "i1_structural_loss_enabled": False,
                "i1_learned_state_from_checkpoint_preserved": True,
                "a0_lifecycle_tracking": True,
                "b0_observation_tracking": True,
                "c_future_utility_tracking": False,
            },
            "randomness": {
                "seed": 0,
                "checkpoint_contains_rng_state": False,
                "same_seed_is_not_per_uid_strict_pairing": True,
                "release_reference_rule": "max(initial_reference_count, current_gaussian_count)",
                "pre_extension_cross_arm_alignment": (
                    "conditional_on_other_arm_draw_count_equal_to_initial_reference_count"
                ),
                "post_release_reference_extension_strict_rng_alignment": False,
            },
            "status": "RUNNING",
        }
        meta_path = output / "launch_meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        try:
            metadata["training_seconds"] = run_logged(command, output / "train.log")
            evaluation_command = [
                sys.executable, "-u", "tools/evaluate_i2_5k.py",
                "-m", str(output), "--iteration", "5000",
            ]
            metadata["evaluation_argv"] = evaluation_command
            metadata["evaluation_seconds"] = run_logged(
                evaluation_command, output / "evaluation.log"
            )
            metadata["status"] = "PASS"
        except Exception as exc:
            metadata["status"] = "FAILED"
            metadata["error"] = repr(exc)
            raise
        finally:
            meta_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        outputs[arm] = output
    completed_outputs = {
        arm: ROOT / "output" / directory
        for arm, directory in ARMS.items()
        if (ROOT / "output" / directory / "formal_metrics_5000.json").is_file()
        and (ROOT / "output" / directory / "launch_meta.json").is_file()
    }
    write_summary(completed_outputs)


if __name__ == "__main__":
    main()
