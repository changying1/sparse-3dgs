#!/usr/bin/env python3
"""Build Fern's fresh I1+A0/B0 iteration-1201 common source.

The default mode is a read-only preflight.  GPU training and local PDC staging
require the explicit ``--execute`` switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(r"D:\dataset\nerf_llff_data_colmap\fern")
VIEW_MANIFEST = Path(
    r"D:\3DGS\Binocular3DGS + Dropout\output"
    r"\A3_A3_Fern_3v_Baseline_5k_FrozenPDC\view_manifest.json"
)
SOURCE_PDC = Path(
    r"D:\3DGS\Binocular3DGS + Dropout\keypoints_to_3d"
    r"\LLFF_Fern_3v_FrozenPDC_v2\fern_keypoints_to_3d.ply"
)
LOCAL_PDC = ROOT / "keypoints_to_3d/LLFF_Fern_3v_FrozenPDC_v2/fern_keypoints_to_3d.ply"
DEPTH_CACHE = Path(
    r"D:\3DGS\Binocular3DGS + Dropout\output"
    r"\I1_Fern_3v_MiDaS_DPT_Hybrid_Cache_v2"
)
OUTPUT = ROOT / "output/I2_Fern_CommonSource_I1_A0B0_1201"
DEFAULT_PYTHON = Path(r"D:\Users\huashuo\anaconda3\envs\fsgs\python.exe")

MANIFEST_SHA256 = "dfe1c8a769b4e3cea96c585014119c05757fe513fe9d24485f2f99e3cf15aa24"
PDC_SHA256 = "bf423f3c4cd0dd8db68a5f7593982c9a287c4ee0b26f30b540d883686d411f8d"
CACHE_MANIFEST_SHA256 = "5a34b188b81ec116fff3200e3dcb362573b557eb2537b4f3794264a2a0d9d991"
TRAIN_IMAGES = ["IMG_4027.JPG", "IMG_4036.JPG", "IMG_4045.JPG"]
TEST_IMAGES = ["IMG_4026.JPG", "IMG_4034.JPG", "IMG_4042.JPG"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty file: {path}")


def require_hash(path: Path, expected: str) -> str:
    require_file(path)
    found = sha256(path)
    if found.lower() != expected.lower():
        raise RuntimeError(
            f"SHA-256 mismatch: {path}\n  expected {expected}\n  found    {found}"
        )
    return found


def inspect_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise RuntimeError(f"PLY magic missing: {path}")
        for _ in range(100):
            line = handle.readline()
            if not line:
                break
            if line.startswith(b"element vertex "):
                count = int(line.split()[2])
                if count <= 0:
                    raise RuntimeError(f"PLY has no vertices: {path}")
                return count
    raise RuntimeError(f"PLY vertex declaration missing: {path}")


def preflight(python_executable: Path) -> dict:
    require_file(python_executable)
    require_hash(VIEW_MANIFEST, MANIFEST_SHA256)
    require_hash(SOURCE_PDC, PDC_SHA256)
    require_hash(DEPTH_CACHE / "oe_depth_cache.json", CACHE_MANIFEST_SHA256)

    manifest = json.loads(VIEW_MANIFEST.read_text(encoding="utf-8-sig"))
    if (
        manifest.get("dataset") != "LLFF"
        or manifest.get("scene") != "fern"
        or manifest.get("view_count") != 3
        or manifest.get("selected_train_images") != TRAIN_IMAGES
        or manifest.get("pdc_initialization_images") != TRAIN_IMAGES
        or manifest.get("test_images") != TEST_IMAGES
        or set(TRAIN_IMAGES) & set(TEST_IMAGES)
    ):
        raise RuntimeError("Fern frozen view manifest does not match the preregistered split")

    disk_images = sorted(path.name for path in (DATASET / "images").iterdir() if path.is_file())
    if disk_images != manifest.get("sorted_all_images"):
        raise RuntimeError("Fern image inventory differs from the frozen manifest")

    cache_meta_path = DEPTH_CACHE / "oe_depth_cache.json"
    cache = json.loads(cache_meta_path.read_text(encoding="utf-8-sig"))
    if (
        cache.get("cache_version") != "oe-depth-cache-v2"
        or cache.get("model_type") != "DPT_Hybrid"
        or cache.get("source_view_manifest_sha256", "").lower() != MANIFEST_SHA256
        or cache.get("selected_train_images") != TRAIN_IMAGES
        or [row.get("image_name") for row in cache.get("records", [])] != TRAIN_IMAGES
    ):
        raise RuntimeError("Fern MiDaS cache metadata does not match the frozen manifest")

    cache_hashes = {}
    source_hashes = {}
    for record in cache["records"]:
        image_path = DATASET / "images" / record["source_image"]
        depth_path = DEPTH_CACHE / record["depth_file"]
        source_hashes[record["image_name"]] = require_hash(
            image_path, record["source_sha256"]
        )
        cache_hashes[record["image_name"]] = require_hash(
            depth_path, record["depth_sha256"]
        )

    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to reuse common-source output: {OUTPUT}")
    if LOCAL_PDC.exists() and sha256(LOCAL_PDC) != PDC_SHA256:
        raise RuntimeError(f"Existing local Fern PDC has the wrong hash: {LOCAL_PDC}")

    return {
        "status": "PASS",
        "mode": "read_only_preflight",
        "scene": "fern",
        "source_iteration": 0,
        "target_iteration": 1201,
        "effective_optimizer_updates": 1200,
        "manifest_sha256": MANIFEST_SHA256,
        "pdc_sha256": PDC_SHA256,
        "pdc_vertices": inspect_ply_vertex_count(SOURCE_PDC),
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "source_image_sha256": source_hashes,
        "depth_sha256": cache_hashes,
        "train_images": TRAIN_IMAGES,
        "test_images": TEST_IMAGES,
        "fresh_uid_and_optimizer_state": True,
        "test_views_used_for_selection": False,
    }


def training_command(python_executable: Path) -> list[str]:
    return [
        str(python_executable), "-u", str(ROOT / "train.py"),
        "-s", str(DATASET), "-m", str(OUTPUT),
        "--dataset_name", "LLFF", "--eval", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1",
        "--view_manifest", str(VIEW_MANIFEST),
        "--suffix", "Fern_3v_FrozenPDC_v2",
        "--iterations", "1201", "--dropout_factor", "0.2", "--sigma_noise", "0.0",
        "--densify_from_iter", "1000", "--densification_interval", "100",
        "--densify_grad_threshold", "0.0002",
        "--enable_oe_structural_loss", "--oe_structural_weight", "0.1",
        "--oe_structural_start_iter", "1", "--oe_structural_end_iter", "1201",
        "--oe_structure_gate_mode", "stable_evidence", "--oe_stable_quantile", "0.8",
        "--oe_preserve_baseline_densification_stats",
        "--oe_depth_cache_dir", str(DEPTH_CACHE),
        "--enable_i2_lifecycle_tracking", "--enable_i2_observation_tracking",
        "--checkpoint_iterations", "1201", "--test_iterations", "1201",
        "--save_iterations", "1201",
    ]


def run_logged(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            log.write(block)
            log.flush()
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Fern common-source training failed with exit code {returncode}")
    return time.perf_counter() - started


def audit_checkpoint(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, (tuple, list)) or len(checkpoint) != 2:
        raise RuntimeError("Malformed Fern source checkpoint wrapper")
    model_state, iteration = checkpoint
    if iteration != 1201 or not isinstance(model_state, (tuple, list)) or len(model_state) != 13:
        raise RuntimeError("Fern source is not an iteration-1201 A0/B0 checkpoint")
    extension = model_state[12]
    if not isinstance(extension, dict) or set(extension) != {"i2_lifecycle", "i2_observation"}:
        raise RuntimeError("Fern source must contain A0+B0 only; C/legacy state is forbidden")
    gaussian_count = int(model_state[1].shape[0])
    lifecycle_count = int(extension["i2_lifecycle"]["uid"].numel())
    observation_count = int(
        extension["i2_observation"]["real_view_opportunities"].numel()
    )
    if gaussian_count != lifecycle_count or gaussian_count != observation_count:
        raise RuntimeError("Fern source Gaussian/A0/B0 rows are not aligned")
    return {
        "checkpoint_sha256": sha256(path),
        "iteration": iteration,
        "checkpoint_tuple_fields": len(model_state),
        "gaussian_count": gaussian_count,
        "a0_b0_row_aligned": True,
        "contains_future_utility_c": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="stage PDC and start GPU training")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()

    frozen = preflight(args.python)
    command = training_command(args.python)
    if not args.execute:
        print(json.dumps({**frozen, "command": command}, indent=2))
        return 0

    LOCAL_PDC.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_PDC.exists():
        shutil.copyfile(SOURCE_PDC, LOCAL_PDC)
    require_hash(LOCAL_PDC, PDC_SHA256)
    OUTPUT.mkdir(parents=True, exist_ok=False)

    metadata = {
        "format": "i2_fern_common_source_launch_v1",
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": frozen,
        "argv": command,
        "method_constraints": {
            "i1_active_iterations": [1, 1201],
            "a0_lifecycle": True,
            "b0_real_training_view_observation": True,
            "future_utility_c": False,
            "fresh_optimizer": True,
            "fresh_scene_local_uids": True,
        },
        "source_code_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "train.py",
                "scene/gaussian_model.py",
                "utils/i2_lifecycle.py",
                "utils/i2_observation.py",
                "utils/evidence_structural_loss.py",
                "utils/oe_depth_cache.py",
                "tools/run_i2_fern_source_1201.py",
            )
        },
    }
    meta_path = OUTPUT / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(command, OUTPUT / "train.log")
        metadata["checkpoint_audit"] = audit_checkpoint(OUTPUT / "chkpnt1201.pth")
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
