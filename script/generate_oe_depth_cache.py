"""Explicit MiDaS DPT-Hybrid cache generator for frozen LLFF training views.

This module performs no model loading at import time.  Running the script is
the only path that invokes torch.hub/MiDaS.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from llff_view_protocol import image_path, load_view_manifest
from utils.oe_depth_cache import (
    CACHE_MANIFEST_NAME,
    CACHE_VERSION,
    MIDAS_MODEL_TYPE,
    MIDAS_PREPROCESSING,
)


def _training_size(width, height, resolution):
    if resolution in (1, 2, 4, 8):
        return round(width / resolution), round(height / resolution)
    global_down = width / 1600 if resolution == -1 and width > 1600 else 1.0
    if resolution != -1:
        global_down = width / resolution
    return int(width / global_down), int(height / global_down)


def preflight_generation(view_manifest, images_dir, output_dir, resolution):
    """Perform every filesystem/image/shape check before MiDaS is initialized."""
    manifest_path = Path(view_manifest).resolve()
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = load_view_manifest(manifest_path)
    selected = list(manifest["selected_train_images"])
    tests = set(manifest["test_images"])
    if set(selected) & tests:
        raise RuntimeError("Refusing to generate depth for a test view.")
    if resolution != -1 and resolution <= 0:
        raise ValueError("--resolution must be -1 or a positive integer.")
    if not images_dir.is_dir():
        raise RuntimeError(f"Images directory does not exist: {images_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Cache output path is not a directory: {output_dir}")
        existing = sorted(item.name for item in output_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"Refusing to write into non-empty cache directory {output_dir}; existing={existing}."
            )

    # Validate every frozen-manifest image, not only the selected subset.  This
    # binds the manifest to the source scene before any model/network access.
    missing = [name for name in manifest["sorted_all_images"] if not image_path(images_dir, name).is_file()]
    if missing:
        raise RuntimeError(f"Frozen view_manifest source images are missing: {missing}")

    plan = []
    for index, name in enumerate(selected):
        source = image_path(images_dir, name)
        try:
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot validate source image {source}: {exc}") from exc
        target_wh = _training_size(width, height, resolution)
        if target_wh[0] <= 0 or target_wh[1] <= 0:
            raise RuntimeError(f"Invalid target resolution for {name}: {target_wh}")
        depth_file = f"{index:04d}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}.npy"
        plan.append({
            "image_name": name,
            "source": source,
            "source_sha256": _sha256(source),
            "target_wh": target_wh,
            "depth_file": depth_file,
        })
    return manifest_path, images_dir, output_dir, manifest, plan


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_depth_fsgs(midas, rgb_chw, target_hw):
    """Exact preprocessing/inference semantics of FSGS utils/depth_utils.py."""
    norm_img = (rgb_chw[None] - 0.5) / 0.5
    norm_img = F.interpolate(norm_img, size=(384, 512), mode="bicubic", align_corners=False)
    with torch.no_grad():
        prediction = midas(norm_img)
        prediction = F.interpolate(
            prediction.unsqueeze(1), size=target_hw, mode="bicubic", align_corners=False
        ).squeeze()
    return prediction


def generate(args):
    manifest_path, images_dir, output_dir, manifest, plan = preflight_generation(
        args.view_manifest, args.images_dir, args.output_dir, args.resolution
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Cache directory changed after preflight: {output_dir}")
    device = torch.device(args.device)
    # Deliberately lazy: default training/render/metrics imports never reach this call.
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE, trust_repo=True, skip_validation=True)
    # Preserve the reference code's transform load even though dpt_transform is not called.
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
    _unused_dpt_transform = midas_transforms.dpt_transform
    midas.to(device).eval()
    for parameter in midas.parameters():
        parameter.requires_grad = False

    records = []
    for item in plan:
        name = item["image_name"]
        source = item["source"]
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            target_wh = item["target_wh"]
            # Match utils.general_utils.PILtoTorch before applying FSGS MiDaS
            # normalization/interpolation.
            rgb = rgb.resize(target_wh)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_chw = torch.from_numpy(array).permute(2, 0, 1).to(device)
        target_hw = (target_wh[1], target_wh[0])
        depth = estimate_depth_fsgs(midas, rgb_chw, target_hw).detach().float().cpu().numpy()
        if depth.shape != target_hw or not np.isfinite(depth).all():
            raise RuntimeError(f"Invalid MiDaS output for {name}: shape={depth.shape}")
        depth_file = item["depth_file"]
        depth_path = output_dir / depth_file
        # Exclusive creation is the final race-safe overwrite guard.
        with depth_path.open("xb") as handle:
            np.save(handle, depth.astype(np.float32, copy=False), allow_pickle=False)
        records.append({
            "image_name": name,
            "height": target_hw[0],
            "width": target_hw[1],
            "depth_file": depth_file,
            "source_image": name,
            "source_sha256": item["source_sha256"],
            "depth_sha256": _sha256(depth_path),
        })

    cache_manifest = {
        "cache_version": CACHE_VERSION,
        "model_type": MIDAS_MODEL_TYPE,
        "preprocessing": MIDAS_PREPROCESSING,
        "source_view_manifest": str(manifest_path),
        "source_view_manifest_sha256": _sha256(manifest_path),
        "source_images_dir": str(images_dir),
        "selected_train_images": list(manifest["selected_train_images"]),
        "records": records,
    }
    # Serialize to an incomplete marker, then publish the success manifest with
    # an atomic, no-overwrite hard link.  A failed dump never creates the final
    # manifest name and therefore cannot be mistaken for a successful cache.
    incomplete_manifest = output_dir / f".{CACHE_MANIFEST_NAME}.incomplete"
    final_manifest = output_dir / CACHE_MANIFEST_NAME
    with incomplete_manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(cache_manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(incomplete_manifest, final_manifest)
    incomplete_manifest.unlink()


def main():
    parser = argparse.ArgumentParser(description="Generate an explicit I1 MiDaS cache for frozen training views.")
    parser.add_argument("--view_manifest", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
