# -*- coding: utf-8 -*-
"""LLFF 3-view / 5000-iteration paired FSGS train launcher.

Download this file to: D:\\3DGS\\Binocular3DGS + Dropout\\llff_i1_5k_continue.py
Run (Windows PowerShell):
  & 'D:\\Users\\huashuo\\anaconda3\\envs\\fsgs\\python.exe' -u .\\llff_i1_5k_continue.py --scene flower --variant i1

No training starts if the target experiment output already exists. Frozen scene
manifests and PDC artifacts are verified against initialization_freeze.json.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\3DGS\Binocular3DGS + Dropout")
DATA_ROOT = Path(r"D:\dataset\nerf_llff_data_colmap")
OUT_ROOT = ROOT / "output"
SCENES = {
    "flower": "Flower",
    "fortress": "Fortress",
    "orchids": "Orchids",
    "room": "Room",
    "trex": "T-Rex",
    "leaves": "Leaves",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty file: {path}")


def require_hash(path: Path, expected: str) -> None:
    require_file(path)
    found = sha256(path)
    if found != str(expected).upper():
        raise RuntimeError(f"Frozen SHA256 mismatch: {path}\n  expected {expected}\n  found    {found}")


def read_cfg(path: Path) -> dict:
    require_file(path)
    node = ast.parse(path.read_text(encoding="utf-8-sig"), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
        raise RuntimeError(f"Unrecognized cfg_args: {path}")
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def ply_vertices(path: Path) -> int:
    require_file(path)
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise RuntimeError(f"PLY magic missing: {path}")
        for _ in range(1000):
            line = f.readline()
            if not line:
                break
            if line.startswith(b"element vertex "):
                count = int(line.split()[2])
                if count < 1:
                    raise RuntimeError(f"Empty PLY: {path}")
                return count
    raise RuntimeError(f"PLY vertex header missing: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(SCENES))
    parser.add_argument("--variant", required=True, choices=("baseline", "i1"))
    parser.add_argument("--dry-run", action="store_true", help="print resolved command without writing outputs")
    args = parser.parse_args()

    scene, label = args.scene, SCENES[args.scene]
    suffix = f"{label}_3v_FrozenPDC_v1"
    frozen_dir = OUT_ROOT / suffix
    manifest_path = frozen_dir / "view_manifest.json"
    freeze_path = frozen_dir / "initialization_freeze.json"
    pdc_path = ROOT / "keypoints_to_3d" / f"LLFF_{suffix}" / f"{scene}_keypoints_to_3d.ply"
    scene_dir = DATA_ROOT / scene
    cache_dir = OUT_ROOT / f"I1_{label}_3v_MiDaS_DPT_Hybrid_Cache_v2"
    baseline_dir = OUT_ROOT / f"A3_A3_{label}_3v_Baseline_5k_FrozenPDC"
    experiment_dir = (baseline_dir if args.variant == "baseline" else
                      OUT_ROOT / f"I1_{label}_3v_5k_StableEvidence_W01_Protected_01")
    final_ply = experiment_dir / "point_cloud" / "iteration_5000" / "point_cloud.ply"
    train_log = experiment_dir / "train.log"

    # One brief audit of immutable, per-scene frozen training inputs; no costly rebuild.
    if not ROOT.joinpath("train.py").is_file():
        raise RuntimeError(f"train.py not found: {ROOT}")
    if experiment_dir.exists():
        raise RuntimeError(f"REFUSING to overwrite existing experiment: {experiment_dir}")
    require_file(freeze_path)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    require_hash(manifest_path, frozen["view_manifest_sha256"])
    require_hash(pdc_path, frozen["initialization_ply_sha256"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    selected = manifest["selected_train_images"]
    if (manifest.get("scene") != scene or manifest.get("dataset") != "LLFF"
        or manifest.get("view_count") != 3 or len(selected) != 3
        or len(set(selected)) != 3
        or selected != manifest["pdc_initialization_images"]
        or selected != frozen["selected_train_images"]
        or set(selected) & set(manifest["test_images"])):
        raise RuntimeError(f"Invalid frozen three-view manifest: {manifest_path}")
    for name in selected:
        require_hash(scene_dir / "images" / name, frozen["selected_train_image_sha256"][name])

    cache_metadata_sha = None
    if args.variant == "i1":
        require_file(baseline_dir / "point_cloud" / "iteration_5000" / "point_cloud.ply")
        baseline_cfg = read_cfg(baseline_dir / "cfg_args")
        if (baseline_cfg.get("iterations") != 5000
            or baseline_cfg.get("n_views") != 3
            or baseline_cfg.get("resolution") != 8
            or baseline_cfg.get("suffix") != suffix
            or baseline_cfg.get("enable_oe_structural_loss") is not False
            or Path(baseline_cfg.get("view_manifest", "")).resolve() != manifest_path.resolve()):
            raise RuntimeError("Baseline cfg_args does not match frozen protocol")
        meta_path = cache_dir / "oe_depth_cache.json"
        require_file(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if (meta.get("cache_version") != "oe-depth-cache-v2"
            or meta.get("model_type") != "DPT_Hybrid"
            or meta.get("selected_train_images") != selected
            or str(meta.get("source_view_manifest_sha256", "")).upper() != sha256(manifest_path)
            or len(list(cache_dir.glob("*.npy"))) != 3):
            raise RuntimeError(f"I1 cache protocol mismatch: {cache_dir}")
        cache_metadata_sha = sha256(meta_path)

    command = [
        sys.executable, "-u", str(ROOT / "train.py"),
        "-s", str(scene_dir), "-m", str(experiment_dir), "--eval",
        "--dataset_name", "LLFF", "--n_views", "3",
        "--view_manifest", str(manifest_path), "--suffix", suffix,
        "--iterations", "5000", "--resolution", "8", "--sh_degree", "1",
        "--dropout_factor", "0.2", "--sigma_noise", "0.0",
        "--save_iterations", "5000",
    ]
    if args.variant == "i1":
        command.extend([
            "--enable_oe_structural_loss",
            "--oe_preserve_baseline_densification_stats",
            "--oe_structural_weight", "0.1",
            "--oe_structural_start_iter", "1",
            "--oe_structural_end_iter", "5000",
            "--oe_stable_quantile", "0.8",
            "--oe_structure_gate_mode", "stable_evidence",
            "--oe_depth_cache_dir", str(cache_dir),
        ])

    print(f"Scene: {scene} | variant: {args.variant} | train images: {selected}", flush=True)
    print(f"Frozen PDC SHA256: {sha256(pdc_path)}", flush=True)
    print(f"Frozen manifest SHA256: {sha256(manifest_path)}", flush=True)
    print(f"Output: {experiment_dir}", flush=True)
    print(f"Command: {subprocess.list2cmdline(command)}", flush=True)
    if args.dry_run:
        print("STATUS: DRY_RUN_ONLY")
        return 0

    env = os.environ.copy()
    env.update({
        "MKL_THREADING_LAYER": "SEQUENTIAL", "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    env.pop("KMP_DUPLICATE_LIB_OK", None)

    experiment_dir.mkdir(parents=True, exist_ok=False)
    with train_log.open("xb") as log:
        header = (
            f"EXPERIMENT: {scene} {args.variant} 3v 5k\n"
            f"FROZEN_MANIFEST_SHA256: {sha256(manifest_path)}\n"
            f"FROZEN_PDC_SHA256: {sha256(pdc_path)}\n"
            f"MIDAS_METADATA_SHA256: {cache_metadata_sha}\n"
            "MKL_THREADING_LAYER=SEQUENTIAL\nOMP_NUM_THREADS=1\n"
            "MKL_NUM_THREADS=1\nKMP_DUPLICATE_LIB_OK=UNSET\n"
            f"COMMAND: {subprocess.list2cmdline(command)}\n\n"
        )
        log.write(header.encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        assert proc.stdout is not None
        while True:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        returncode = proc.wait()

    print(f"Train exit code: {returncode}", flush=True)
    if returncode:
        print(f"FAILED; preserved output and log: {train_log}", flush=True)
        return returncode
    require_file(final_ply)
    cfg = read_cfg(experiment_dir / "cfg_args")
    expected_cfg = {
        "iterations": 5000, "n_views": 3, "resolution": 8,
        "sh_degree": 1, "suffix": suffix,
        "dropout_factor": 0.2, "sigma_noise": 0.0,
        "eval": True, "enable_oe_structural_loss": args.variant == "i1",
    }
    if args.variant == "i1":
        expected_cfg.update({
            "oe_preserve_baseline_densification_stats": True,
            "oe_structural_weight": 0.1,
            "oe_structural_start_iter": 1,
            "oe_structural_end_iter": 5000,
            "oe_stable_quantile": 0.8,
            "oe_structure_gate_mode": "stable_evidence",
        })
    for key, expected in expected_cfg.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"cfg_args mismatch for {key}: {cfg.get(key)!r} != {expected!r}")
    if (Path(cfg["view_manifest"]).resolve() != manifest_path.resolve()
        or Path(cfg["source_path"]).resolve() != scene_dir.resolve()
        or Path(cfg["model_path"]).resolve() != experiment_dir.resolve()):
        raise RuntimeError("Trained paths differ from frozen scene")
    if args.variant == "i1" and Path(cfg["oe_depth_cache_dir"]).resolve() != cache_dir.resolve():
        raise RuntimeError("I1 depth cache path mismatch")
    log_text = train_log.read_text(encoding="utf-8-sig", errors="replace")
    if "Training complete." not in log_text or "[ITER 5000] Saving Gaussians" not in log_text:
        raise RuntimeError("5000-iteration training completion markers missing")
    summary = None
    if args.variant == "i1":
        summaries = []
        for m in re.finditer(r"\[I1\]\s*(\{[^\r\n]+\})", log_text):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "summary":
                summaries.append(obj)
        if len(summaries) != 1 or summaries[0].get("actual_activation_count") != 5000 or summaries[0].get("nonfinite_loss_seen") is not False:
            raise RuntimeError("I1 activation/finite-loss summary audit failed")
        summary = summaries[0]
    require_hash(manifest_path, frozen["view_manifest_sha256"])
    require_hash(pdc_path, frozen["initialization_ply_sha256"])
    audit = {
        "scene": scene, "variant": args.variant, "iterations": 5000,
        "train_images": selected, "test_images": manifest["test_images"],
        "pdc_suffix": suffix, "frozen_pdc_sha256": sha256(pdc_path),
        "frozen_manifest_sha256": sha256(manifest_path),
        "depth_cache_metadata_sha256": cache_metadata_sha,
        "final_ply": str(final_ply), "final_ply_sha256": sha256(final_ply),
        "final_gaussians": ply_vertices(final_ply),
        "train_log": str(train_log), "train_log_sha256": sha256(train_log),
        "train_exitcode": 0, "i1_summary": summary, "command": command,
    }
    with (experiment_dir / "launch_audit.json").open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Final Gaussian count: {audit['final_gaussians']}", flush=True)
    print(f"Final PLY SHA256: {audit['final_ply_sha256']}", flush=True)
    print(f"Full train.log: {train_log}", flush=True)
    print(f"STATUS: {scene.upper()}_{args.variant.upper()}_5K_TRAIN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
