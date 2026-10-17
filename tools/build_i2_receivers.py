#!/usr/bin/env python3
"""Build verified 3D Receiver evidence from frozen Horns training views."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import PipelineParams
from gaussian_renderer import render
from llff_view_protocol import load_view_manifest
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.evidence_support import compute_real_depth_evidence_maps
from utils.i2_observation import build_real_view_camera_mapping
from utils.oe_depth_cache import OEDepthCache


FORMAT = "i2_receiver_evidence_v1"
RULE = "train_rgb_i1_structure_blockmax_visible_gaussian_anchor_nms_v2"
FIELDS = (
    "candidate_id", "demand_basis", "demand_score", "requested_units",
    "training_camera_id", "training_camera_uid", "training_image_name",
    "pixel_x0", "pixel_y0", "pixel_x1", "pixel_y1", "pixel_u", "pixel_v",
    "world_x", "world_y", "world_z", "anchor_type", "anchor_uid", "depth", "depth_valid",
    "geometry_valid", "reprojection_error_px", "rgb_residual", "structural_residual",
    "local_gaussian_count", "local_density_radius", "nearest_gaussian_distance",
    "evidence_path", "evidence_sha256", "source_checkpoint_sha256",
    "generation_rule_id", "score_normalization",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_cfg(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    tree = ast.parse(text, mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "Namespace":
        raise ValueError("cfg_args must contain a single argparse Namespace expression")
    return argparse.Namespace(**{item.arg: ast.literal_eval(item.value) for item in call.keywords})


def safe_checkpoint(path, device):
    # The frozen checkpoint contains NumPy scalar optimizer metadata.  Keep
    # weights_only=True and allowlist only the concrete scalar/dtype classes.
    from numpy._core.multiarray import scalar
    torch.serialization.add_safe_globals([
        scalar, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
    ])
    model_state, iteration = torch.load(path, map_location="cpu", weights_only=True)

    def move_optimizer(value):
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move_optimizer(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move_optimizer(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_optimizer(item) for item in value)
        return value

    # Lifecycle event history is intentionally kept on CPU for strict audit.
    # Only trainable base tensors and optimizer state move to CUDA.
    state = list(model_state)
    for index in range(1, 10):
        if torch.is_tensor(state[index]):
            state[index] = state[index].to(device=device)
    state[10] = move_optimizer(state[10])
    return tuple(state), iteration


def load_runtime(cfg, checkpoint):
    info = sceneLoadTypeCallbacks["Colmap"](
        cfg.source_path, cfg.images, cfg.eval, cfg.n_views, cfg.dataset_name,
        suffix=getattr(cfg, "suffix", None), view_manifest=cfg.view_manifest,
    )
    cameras = cameraList_from_camInfos(info.train_cameras, 1.0, cfg)
    identifiers, _ = build_real_view_camera_mapping(cameras, cfg.view_manifest)
    model_state, iteration = safe_checkpoint(checkpoint, "cuda")
    model = GaussianModel(cfg.sh_degree)
    model.enable_lifecycle_tracking()
    model.enable_observation_tracking(identifiers)
    model.restore(model_state, cfg)
    return model, cameras, int(iteration), float(info.nerf_normalization["radius"])


def unproject(camera, u, v, depth):
    intrinsic, world_to_camera = camera.get_camera_matrix()
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera_xyz = torch.stack(((u - cx) * depth / fx, (v - cy) * depth / fy, depth))
    world = torch.linalg.inv(world_to_camera) @ torch.cat((camera_xyz, camera_xyz.new_ones(1)))
    projected = world_to_camera @ world
    pu = fx * projected[0] / projected[2] + cx
    pv = fy * projected[1] / projected[2] + cy
    error = torch.sqrt((pu - u) ** 2 + (pv - v) ** 2)
    return world[:3], error


def block_candidates(score, valid, block, limit):
    rows = []
    height, width = score.shape
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            y1, x1 = min(y0 + block, height), min(x0 + block, width)
            local = score[y0:y1, x0:x1].masked_fill(~valid[y0:y1, x0:x1], -1)
            flat = int(torch.argmax(local).item())
            value = float(local.reshape(-1)[flat].item())
            if value <= 0 or not math.isfinite(value):
                continue
            ly, lx = divmod(flat, x1 - x0)
            rows.append((value, x0 + lx, y0 + ly, x0, y0, x1, y1))
    rows.sort(key=lambda item: (-item[0], item[2], item[1]))
    return rows if limit == 0 else rows[:limit]


@torch.no_grad()
def build(args):
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_cfg(args.cfg)
    model, cameras, iteration, extent = load_runtime(cfg, checkpoint)
    checkpoint_sha = sha256(checkpoint)
    view_manifest = load_view_manifest(cfg.view_manifest)
    frozen_names = list(view_manifest["selected_train_images"])
    camera_by_stem = {camera.image_name: camera for camera in cameras}
    if {Path(name).stem for name in frozen_names} != set(camera_by_stem):
        raise ValueError("Runtime cameras differ from frozen selected_train_images")
    depth_cache = OEDepthCache(
        cfg.oe_depth_cache_dir, cfg.view_manifest, cameras,
        Path(cfg.source_path) / cfg.images,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    parser = argparse.ArgumentParser(add_help=False)
    pipe_group = PipelineParams(parser)
    pipe = pipe_group.extract(parser.parse_args([]))
    xyz = model.get_xyz.detach()
    density_radius = max(extent * float(args.density_radius_fraction), 1e-8)
    spatial_nms = max(extent * float(args.spatial_nms_fraction), 1e-8)
    raw = []
    evidence_files = []
    complete_enumeration = args.per_view_limit == 0 and args.max_candidates == 0
    candidate_prefix = (
        f"{Path(cfg.source_path).name}{iteration}" if complete_enumeration else "horns1201"
    )
    normalization = "per_view_rgb_p95_and_structure_p95_each_capped_3x_scaled_then_equal_mean"
    for camera_index, full_name in enumerate(frozen_names):
        camera = camera_by_stem[Path(full_name).stem]
        package = render(camera, model, pipe, background, train=False,
                         dropout_factor=float(cfg.dropout_factor))
        rendered = package["render"].clamp(0, 1)
        gt = camera.original_image[:3]
        rgb = (rendered - gt).abs().mean(dim=0)
        rendered_depth = package["rendered_depth"].squeeze()
        alpha = package["rendered_alpha"].squeeze()
        reference_depth = depth_cache.load_for_camera(camera, device=rendered_depth.device)
        maps = compute_real_depth_evidence_maps(
            rendered_depth, reference_depth, detach_rendered=True
        )
        structural = (maps["z_render"] - maps["z_reference"]).abs() * maps["evidence"]
        valid = (
            maps["valid_mask"] & torch.isfinite(rgb) & torch.isfinite(rendered_depth)
            & (rendered_depth > 0) & torch.isfinite(alpha) & (alpha >= args.min_alpha)
        )
        if int(valid.sum().item()) == 0:
            continue
        rgb_p95 = torch.quantile(rgb[valid], 0.95).clamp_min(1e-8)
        struct_p95 = torch.quantile(structural[valid], 0.95).clamp_min(1e-8)
        rgb_normalized = (rgb / rgb_p95).clamp(0, 3) / 3
        structural_normalized = (structural / struct_p95).clamp(0, 3) / 3
        score = 0.5 * rgb_normalized + 0.5 * structural_normalized
        score = score.masked_fill(~valid, 0)
        intrinsic, world_to_camera = camera.get_camera_matrix()
        xyz_h = torch.cat((xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)), dim=1)
        camera_xyz = (world_to_camera @ xyz_h.T).T[:, :3]
        projected_u = intrinsic[0, 0] * camera_xyz[:, 0] / camera_xyz[:, 2] + intrinsic[0, 2]
        projected_v = intrinsic[1, 1] * camera_xyz[:, 1] / camera_xyz[:, 2] + intrinsic[1, 2]
        projected_valid = (
            package["visibility_filter"] & torch.isfinite(projected_u) & torch.isfinite(projected_v)
            & torch.isfinite(camera_xyz[:, 2]) & (camera_xyz[:, 2] > camera.znear)
        )
        payload_path = output / f"receiver_evidence_view_{camera_index:02d}.npz"
        np.savez_compressed(
            payload_path,
            rgb_residual=rgb.detach().cpu().numpy().astype(np.float32),
            structural_residual=structural.detach().cpu().numpy().astype(np.float32),
            demand_score=score.detach().cpu().numpy().astype(np.float32),
            rendered_depth=rendered_depth.detach().cpu().numpy().astype(np.float32),
            rendered_alpha=alpha.detach().cpu().numpy().astype(np.float32),
            valid_mask=valid.detach().cpu().numpy(),
        )
        evidence_path = output / f"receiver_evidence_view_{camera_index:02d}.json"
        evidence_metadata = {
            "format": "i2_receiver_view_evidence_v1", "training_camera_id": full_name,
            "source_checkpoint_sha256": checkpoint_sha, "generation_rule_id": RULE,
            "payload": {"path": str(payload_path), "sha256": sha256(payload_path)},
        }
        evidence_path.write_text(
            json.dumps(evidence_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence_hash = sha256(evidence_path)
        evidence_files.append({"path": str(evidence_path), "sha256": evidence_hash,
                               "payload": evidence_metadata["payload"]})
        for value, u, v, x0, y0, x1, y1 in block_candidates(
            score, valid, args.block_size, args.per_view_limit
        ):
            in_region = (
                projected_valid & (projected_u >= x0) & (projected_u < x1)
                & (projected_v >= y0) & (projected_v < y1)
            )
            anchor_rows = torch.nonzero(in_region, as_tuple=False).reshape(-1)
            if anchor_rows.numel() == 0:
                continue
            screen_distance = (
                (projected_u[anchor_rows] - float(u)).square()
                + (projected_v[anchor_rows] - float(v)).square()
            )
            anchor_row = anchor_rows[torch.argmin(screen_distance)]
            world = xyz[anchor_row]
            depth = camera_xyz[anchor_row, 2]
            reprojection = torch.sqrt(screen_distance.min())
            if not torch.isfinite(world).all() or float(reprojection.item()) > math.sqrt(2) * args.block_size:
                continue
            distances = torch.linalg.vector_norm(xyz - world[None, :], dim=1)
            nearest = float(distances.min().item())
            local_count = int((distances <= density_radius).sum().item())
            rgb_component = float(rgb_normalized[v, u])
            structural_component = float(structural_normalized[v, u])
            basis = "structural_defect" if structural_component >= rgb_component else "reconstruction_error"
            raw.append({
                "candidate_id": f"{candidate_prefix}-v{camera_index}-x{u}-y{v}",
                "demand_basis": basis, "demand_score": value, "requested_units": 1,
                "training_camera_id": full_name, "training_camera_uid": camera.uid,
                "training_image_name": full_name,
                "pixel_x0": x0, "pixel_y0": y0, "pixel_x1": x1, "pixel_y1": y1,
                "pixel_u": u, "pixel_v": v,
                "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
                "anchor_type": "visible_existing_gaussian_projection",
                "anchor_uid": int(model.lifecycle.uid[anchor_row].item()), "depth": float(depth),
                "depth_valid": True, "geometry_valid": True,
                "reprojection_error_px": float(reprojection), "rgb_residual": float(rgb[v, u]),
                "structural_residual": float(structural[v, u]),
                "local_gaussian_count": local_count, "local_density_radius": density_radius,
                "nearest_gaussian_distance": nearest, "evidence_path": str(evidence_path),
                "evidence_sha256": evidence_hash, "source_checkpoint_sha256": checkpoint_sha,
                "generation_rule_id": RULE, "score_normalization": normalization,
            })
    raw.sort(key=lambda row: (-row["demand_score"], row["candidate_id"]))
    selected = []
    for row in raw:
        point = np.array([row["world_x"], row["world_y"], row["world_z"]])
        if any(np.linalg.norm(point - np.array([old["world_x"], old["world_y"], old["world_z"]])) < spatial_nms
               for old in selected):
            continue
        selected.append(row)
        if args.max_candidates > 0 and len(selected) >= args.max_candidates:
            break
    csv_path = output / "receiver_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    code_fingerprint = sha256(Path(__file__))
    report = {
        "format": FORMAT, "status": "available" if selected else "blocked",
        "candidate_count": len(selected), "source_iteration": iteration,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": checkpoint_sha,
        "view_manifest": str(Path(cfg.view_manifest).resolve()),
        "view_manifest_sha256": sha256(cfg.view_manifest), "training_camera_ids": frozen_names,
        "test_views_used": False, "generation_rule_id": RULE,
        "source_code_fingerprint": code_fingerprint, "score_normalization": normalization,
        "spatial_dedup_radius": spatial_nms, "pixel_block_size": args.block_size,
        "max_anchor_pixel_distance": math.sqrt(2) * args.block_size,
        "local_density_radius": density_radius, "receiver_csv": str(csv_path),
        "receiver_csv_sha256": sha256(csv_path), "evidence_files": evidence_files,
        "demand_source_distribution": {
            key: sum(row["demand_basis"] == key for row in selected)
            for key in sorted({row["demand_basis"] for row in selected})
        },
        "enumeration": {
            "complete": complete_enumeration,
            "per_view_limit": args.per_view_limit,
            "max_candidates": args.max_candidates,
            "raw_candidate_count": len(raw),
            "post_nms_candidate_count": len(selected),
        },
    }
    report_path = output / "receiver_evidence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    base_manifest = json.loads(Path(args.d_manifest).read_text(encoding="utf-8"))
    base_manifest["receiver_candidates"] = {"path": str(csv_path), "sha256": sha256(csv_path)}
    base_manifest["receiver_protocol"] = {
        "format": "i2_receiver_protocol_v1", "source_checkpoint_sha256": checkpoint_sha,
        "source_iteration": iteration, "training_camera_ids": frozen_names,
        "view_manifest_sha256": report["view_manifest_sha256"],
        "generation_rule_id": RULE, "source_code_fingerprint": code_fingerprint,
        "score_normalization": normalization, "spatial_dedup_radius": spatial_nms,
        "max_anchor_pixel_distance": math.sqrt(2) * args.block_size,
    }
    manifest_path = output / "d_input_manifest_with_receivers.json"
    manifest_path.write_text(json.dumps(base_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "manifest": str(manifest_path), **report}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--d_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block_size", type=int, default=12)
    parser.add_argument("--per_view_limit", type=int, default=250)
    parser.add_argument("--max_candidates", type=int, default=200)
    parser.add_argument("--min_alpha", type=float, default=0.5)
    parser.add_argument("--max_reprojection_error", type=float, default=1e-3)
    parser.add_argument("--spatial_nms_fraction", type=float, default=0.001)
    parser.add_argument("--density_radius_fraction", type=float, default=0.005)
    args = parser.parse_args()
    if args.per_view_limit < 0 or args.max_candidates < 0:
        parser.error("candidate limits must be non-negative; zero means complete enumeration")
    build(args)


if __name__ == "__main__":
    main()
