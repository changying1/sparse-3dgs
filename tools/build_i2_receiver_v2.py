#!/usr/bin/env python3
"""Build a fail-closed, geometry-constrained I2 Receiver-v2 plan.

Only the three frozen training cameras and the frozen iteration-1201 checkpoint
are used.  Rasterizer depth is used solely as an occlusion-risk check; Receiver
positions are obtained from the anchor Gaussian tangent plane and the analytic
camera projection Jacobian.
"""

from __future__ import annotations

import argparse
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

from tools.build_i2_receivers import load_runtime, parse_cfg
from arguments import PipelineParams
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.i2_active_pilot import (
    load_active_plan,
    receiver_v2_displacement_contract,
    select_receiver_v2_candidates,
)
from utils.sh_utils import eval_sh


METHOD_VERSION = "receiver_v2_tangent_jacobian_v1"
EVIDENCE_FORMAT = "i2_receiver_v2_geometry_evidence_v1"
PLAN_FORMAT = "i2_d1_receiver_v2_plan_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(camera, xyz: torch.Tensor):
    intrinsic, world_to_camera = camera.get_camera_matrix()
    homogeneous = torch.cat((xyz, xyz.new_ones(1)))
    camera_xyz = (world_to_camera @ homogeneous)[:3]
    x, y, z = camera_xyz
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    u = fx * x / z + intrinsic[0, 2]
    v = fy * y / z + intrinsic[1, 2]
    jac_camera = xyz.new_tensor([
        [fx / z, 0.0, -fx * x / (z * z)],
        [0.0, fy / z, -fy * y / (z * z)],
    ])
    jac_world = jac_camera @ world_to_camera[:3, :3]
    return torch.stack((u, v)), z, jac_world


def load_depth_payloads(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payloads = {}
    for index, item in enumerate(report["evidence_files"]):
        metadata_path = Path(item["path"])
        if sha256(metadata_path) != item["sha256"]:
            raise RuntimeError("Receiver-v1 evidence metadata SHA256 mismatch")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload_path = Path(metadata["payload"]["path"])
        if sha256(payload_path) != metadata["payload"]["sha256"]:
            raise RuntimeError("Receiver-v1 evidence payload SHA256 mismatch")
        with np.load(payload_path) as payload:
            payloads[index] = {
                "rendered_depth": payload["rendered_depth"].copy(),
                "valid_mask": payload["valid_mask"].copy(),
            }
    return report, payloads


def finite_list(tensor):
    values = [float(value) for value in tensor.detach().cpu().tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Receiver-v2 produced a non-finite value")
    return values


@torch.no_grad()
def candidate_geometry(row, *, model, row_by_uid, camera_by_name, camera_order,
                       payloads, max_condition, displacement_factor,
                       opacity, footprint_pixels, base_images):
    anchor_uid = int(row["anchor_uid"])
    anchor_row = row_by_uid.get(anchor_uid)
    if anchor_row is None:
        return None, "anchor_not_live"
    anchor = model.get_xyz[anchor_row]
    scales = model.get_scaling[anchor_row]
    rotation = build_rotation(model._rotation[anchor_row:anchor_row + 1])[0]
    if not torch.isfinite(anchor).all() or not torch.isfinite(scales).all() or torch.any(scales <= 0):
        return None, "invalid_anchor_parameters"

    source_name = Path(row["training_camera_id"]).stem
    source_camera = camera_by_name[source_name]
    projected, source_depth, jacobian = project(source_camera, anchor)
    desired_pixel = anchor.new_tensor(row["pixel"], dtype=anchor.dtype)
    pixel_delta = desired_pixel - projected
    before_error = float(torch.linalg.vector_norm(pixel_delta).item())
    if before_error <= 0.25:
        return None, "insufficient_nonoverlap_pixel_demand"

    order = torch.argsort(scales, descending=True)
    tangent_indices = order[:2]
    normal_index = int(order[2].item())
    tangent = rotation[:, tangent_indices]
    projected_tangent = jacobian @ tangent
    singular = torch.linalg.svdvals(projected_tangent)
    condition = float((singular.max() / singular.min().clamp_min(1e-12)).item())
    if not math.isfinite(condition) or condition > max_condition or float(singular.min()) <= 1e-6:
        return None, "ill_conditioned_tangent_projection"

    coefficients = torch.linalg.solve(projected_tangent, pixel_delta)
    displacement = tangent @ coefficients
    raw_norm = float(torch.linalg.vector_norm(displacement).item())
    max_displacement = float(
        displacement_factor * torch.linalg.vector_norm(scales[tangent_indices]).item()
    )
    if raw_norm > max_displacement:
        displacement = displacement * (max_displacement / raw_norm)
    target = anchor + displacement
    # Evidence and model initialization both use the final float32 XYZ.  The
    # effective displacement must therefore be derived after float32 addition.
    displacement = target - anchor
    displacement_norm = float(torch.linalg.vector_norm(displacement).item())
    minimum_displacement = float(0.10 * scales[tangent_indices].min().item())
    if displacement_norm <= max(1e-8, minimum_displacement):
        return None, "displacement_too_small_for_nonoverlap"
    achieved, _, _ = project(source_camera, target)
    after_error = float(torch.linalg.vector_norm(desired_pixel - achieved).item())
    if after_error >= min(before_error * 0.75, before_error - 0.25):
        return None, "insufficient_source_pixel_coverage"

    # First-order alpha compositing guard.  Receiver-v2 deliberately reuses
    # anchor SH, so accept it only when that copied color moves the frozen
    # source-view render toward the observed RGB target at the residual pixel.
    view_direction = anchor - source_camera.camera_center
    view_direction = view_direction / torch.linalg.vector_norm(view_direction).clamp_min(1e-12)
    sh = model.get_features[anchor_row].transpose(0, 1)
    anchor_rgb = (eval_sh(model.active_sh_degree, sh, view_direction) + 0.5).clamp(0, 1)
    x0, y0, x1, y1 = map(int, row["pixel_region"])
    rendered_rgb = base_images[source_name][:, y0:y1, x0:x1].mean(dim=(1, 2))
    target_rgb = source_camera.original_image[:3, y0:y1, x0:x1].mean(dim=(1, 2))
    residual_rgb = target_rgb - rendered_rgb
    blend_direction = anchor_rgb - rendered_rgb
    photometric_alignment = float(torch.dot(residual_rgb, blend_direction).item())
    if not math.isfinite(photometric_alignment):
        return None, "invalid_photometric_alignment"

    # New scale uses the requested residual footprint and the analytic pixel
    # sensitivity of each copied anchor axis, bounded to a conservative
    # fraction of the anchor scale.  The smallest (normal) axis is contracted.
    new_scale = scales.clone()
    for axis_index in tangent_indices.tolist():
        sensitivity = float(torch.linalg.vector_norm(jacobian @ rotation[:, axis_index]).item())
        if not math.isfinite(sensitivity) or sensitivity <= 1e-8:
            return None, "invalid_projected_scale_sensitivity"
        geometric = footprint_pixels / sensitivity
        lower = 0.20 * float(scales[axis_index].item())
        upper = 0.60 * float(scales[axis_index].item())
        new_scale[axis_index] = min(upper, max(lower, geometric))
    new_scale[normal_index] = 0.35 * scales[normal_index]
    if not torch.isfinite(new_scale).all() or torch.any(new_scale <= 0):
        return None, "invalid_new_scale"

    checks = []
    source_index = camera_order.index(source_name)
    depth_extent = float(3.0 * torch.linalg.vector_norm(new_scale).item())
    for view_index, camera_name in enumerate(camera_order):
        camera = camera_by_name[camera_name]
        old_pixel, old_depth, _ = project(camera, anchor)
        new_pixel, new_depth, _ = project(camera, target)
        width, height = int(camera.image_width), int(camera.image_height)
        u, v = [float(value) for value in new_pixel]
        in_front = bool(torch.isfinite(new_depth).item() and new_depth.item() > camera.znear)
        in_frame = in_front and 0 <= u < width and 0 <= v < height
        depth_shift = abs(float((new_depth - old_depth).item()))
        depth_consistent = depth_shift <= max(depth_extent, 1e-6)
        front_occlusion_safe = True
        reference_depth = None
        if in_frame:
            px = min(width - 1, max(0, int(round(u))))
            py = min(height - 1, max(0, int(round(v))))
            payload = payloads[view_index]
            if bool(payload["valid_mask"][py, px]):
                reference_depth = float(payload["rendered_depth"][py, px])
                # Depth is never used to construct XYZ.  It only vetoes a new
                # point that would sit conspicuously in front of the rendered surface.
                front_occlusion_safe = float(new_depth) >= reference_depth - max(depth_extent, 0.01 * reference_depth)
        safe = in_front and depth_consistent and front_occlusion_safe
        if view_index == source_index:
            safe = safe and in_frame and after_error < before_error
        checks.append({
            "training_camera_id": camera_name + ".jpg",
            "projected_pixel": finite_list(new_pixel),
            "anchor_projected_pixel": finite_list(old_pixel),
            "depth": float(new_depth),
            "anchor_depth": float(old_depth),
            "depth_shift": depth_shift,
            "in_front": in_front,
            "in_frame": in_frame,
            "rendered_depth_for_risk_check": reference_depth,
            "front_occlusion_safe": front_occlusion_safe,
            "depth_consistent": depth_consistent,
            "safe": bool(safe),
        })
    if not all(check["safe"] for check in checks):
        return None, "cross_view_check_failed"

    result = {
        "candidate_id": row["candidate_id"],
        "anchor_uid": anchor_uid,
        "training_camera_id": row["training_camera_id"],
        "demand_score": float(row["demand_score"]),
        "demand_basis": row["demand_basis"],
        "residual_pixel": list(map(int, row["pixel"])),
        "pixel_region": list(map(int, row["pixel_region"])),
        "anchor_xyz": finite_list(anchor),
        "new_xyz": finite_list(target),
        "displacement": finite_list(displacement),
        "displacement_norm": displacement_norm,
        "max_displacement": max_displacement,
        "tangent_axis_indices": [int(value) for value in tangent_indices.tolist()],
        "normal_axis_index": normal_index,
        "projection_condition": condition,
        "source_pixel_offset_before": before_error,
        "source_pixel_offset_after": after_error,
        "source_rgb_residual": finite_list(residual_rgb),
        "anchor_rgb_at_source_view": finite_list(anchor_rgb),
        "photometric_alignment": photometric_alignment,
        "activated_opacity": float(opacity),
        "anchor_activated_opacity": float(model.get_opacity[anchor_row].item()),
        "activated_scale": finite_list(new_scale),
        "anchor_activated_scale": finite_list(scales),
        "rotation_source": "anchor",
        "sh_source": "anchor",
        "cross_view_checks": checks,
        "geometry_valid": True,
        "fail_closed": False,
    }
    return result, None


@torch.no_grad()
def build(args):
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_cfg(args.cfg)
    old_plan_path = Path(args.v1_plan).resolve()
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    verified = load_active_plan(
        old_plan_path, requested_budget=args.budget, source_checkpoint=checkpoint
    )
    frozen_donors = verified["_donor_uids"]
    model, cameras, iteration, _ = load_runtime(cfg, checkpoint)
    if iteration != 1201:
        raise RuntimeError("Receiver-v2 requires the frozen iteration-1201 checkpoint")
    camera_by_name = {camera.image_name: camera for camera in cameras}
    camera_order = [Path(name).stem for name in old_plan["input_audit"]["receiver_protocol"]["training_camera_ids"]]
    if set(camera_by_name) != set(camera_order):
        raise RuntimeError("Runtime cameras differ from the frozen training views")
    report, payloads = load_depth_payloads(Path(args.receiver_report).resolve())
    if report["test_views_used"] is not False:
        raise RuntimeError("Receiver evidence unexpectedly used test views")
    row_by_uid = {int(uid): index for index, uid in enumerate(model.lifecycle.uid.tolist())}
    parser = argparse.ArgumentParser(add_help=False)
    pipeline = PipelineParams(parser).extract(parser.parse_args([]))
    background = torch.tensor(
        [1, 1, 1] if cfg.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    base_images = {
        camera.image_name: render(
            camera, model, pipeline, background, train=True,
            dropout_factor=0.0, sigma_noise=0.0,
        )["render"].clamp(0, 1)
        for camera in cameras
    }
    candidates = old_plan["receivers"]["rows"]
    valid, rejected = [], []
    for row in candidates:
        evidence, reason = candidate_geometry(
            row, model=model, row_by_uid=row_by_uid,
            camera_by_name=camera_by_name, camera_order=camera_order,
            payloads=payloads, max_condition=args.max_condition,
            displacement_factor=args.displacement_factor,
            opacity=args.opacity, footprint_pixels=args.footprint_pixels,
            base_images=base_images,
        )
        if evidence is None:
            rejected.append({"candidate_id": row["candidate_id"], "reason": reason})
            continue
        valid.append(evidence)
    selected, selection_rejected = select_receiver_v2_candidates(
        valid, budget=args.budget,
        spatial_dedup_radius=report["spatial_dedup_radius"],
    )
    rejected.extend(selection_rejected)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    table_path = output / "receiver_v2_initialization.csv"
    table_fields = [
        "candidate_id", "anchor_uid", "training_camera_id", "demand_score",
        "anchor_x", "anchor_y", "anchor_z", "new_x", "new_y", "new_z",
        "displacement_norm", "source_pixel_offset_before", "source_pixel_offset_after",
        "activated_opacity", "scale_x", "scale_y", "scale_z", "projection_condition",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "candidate_id": row["candidate_id"], "anchor_uid": row["anchor_uid"],
                "training_camera_id": row["training_camera_id"], "demand_score": row["demand_score"],
                **{f"anchor_{axis}": row["anchor_xyz"][i] for i, axis in enumerate("xyz")},
                **{f"new_{axis}": row["new_xyz"][i] for i, axis in enumerate("xyz")},
                "displacement_norm": row["displacement_norm"],
                "source_pixel_offset_before": row["source_pixel_offset_before"],
                "source_pixel_offset_after": row["source_pixel_offset_after"],
                "activated_opacity": row["activated_opacity"],
                **{f"scale_{axis}": row["activated_scale"][i] for i, axis in enumerate("xyz")},
                "projection_condition": row["projection_condition"],
            })
    candidate_table = Path(old_plan["input_audit"]["receiver_candidate_table"]["path"])
    evidence = {
        "format": EVIDENCE_FORMAT,
        "method_version": METHOD_VERSION,
        "source_iteration": iteration,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "receiver_candidate_table": str(candidate_table),
        "receiver_candidate_table_sha256": sha256(candidate_table),
        "receiver_v1_plan": str(old_plan_path),
        "receiver_v1_plan_sha256": sha256(old_plan_path),
        "training_camera_ids": old_plan["input_audit"]["receiver_protocol"]["training_camera_ids"],
        "test_views_used": False,
        "position_formula": "delta_X=B_tangent*(J_projection*B_tangent)^-1*(p_residual-p_anchor), norm-clamped",
        "photometric_ranking": "descending dot(mean_block(GT-render), anchor_SH_RGB-mean_block(render)) on the source 12x12 evidence block after geometry filtering",
        "depth_usage": "rasterizer depth is veto-only for front-occlusion risk; never used for XYZ backprojection",
        "opacity_formula": f"constant activated opacity={args.opacity:.8g}",
        "scale_formula": "tangent_i=clip(footprint_px/||J*r_i||,0.20*s_i,0.60*s_i); normal=0.35*s_normal",
        "parameters": {
            "max_condition": args.max_condition,
            "displacement_factor": args.displacement_factor,
            "opacity": args.opacity,
            "footprint_pixels": args.footprint_pixels,
        },
        "initialization_table": {"path": str(table_path), "sha256": sha256(table_path)},
        "candidate_count": len(selected),
        "requested_k": args.budget,
        "rejected_before_requested_k": rejected,
        "rows": selected,
    }
    evidence_path = output / "receiver_v2_geometry_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    new_plan = json.loads(json.dumps(old_plan))
    new_plan["format"] = PLAN_FORMAT
    new_plan["method_version"] = METHOD_VERSION
    new_plan["matches"] = [
        {
            "donor_uid": donor_uid,
            "receiver_candidate_id": receiver["candidate_id"],
            "released_units": 1,
            "received_units": 1,
        }
        for donor_uid, receiver in zip(frozen_donors, selected)
    ]
    new_plan["receiver_v2"] = {
        "method_version": METHOD_VERSION,
        "artifact": {"path": str(evidence_path), "sha256": sha256(evidence_path)},
        "test_views_used": False,
    }
    new_plan["receiver_selection_change"] = {
        "v1_candidate_ids": verified["_receiver_candidate_ids"],
        "v2_candidate_ids": [row["candidate_id"] for row in selected],
        "changed": verified["_receiver_candidate_ids"] != [row["candidate_id"] for row in selected],
        "donor_uids_unchanged": frozen_donors,
    }
    plan_path = output / f"i2_d1_receiver_v2_k{args.budget}_plan.json"
    plan_path.write_text(json.dumps(new_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_active_plan(
        plan_path, requested_budget=args.budget, source_checkpoint=checkpoint
    )
    summary = {
        "status": "GO",
        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "evidence": str(evidence_path), "evidence_sha256": sha256(evidence_path),
        "table": str(table_path), "table_sha256": sha256(table_path),
        "requested_k": args.budget, "selected_count": len(selected),
        "rejected_before_requested_k": len(rejected),
        "donor_uids_unchanged": frozen_donors,
        "test_views_used": False,
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--v1_plan", required=True)
    parser.add_argument("--receiver_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_condition", type=float, default=25.0)
    parser.add_argument("--displacement_factor", type=float, default=2.0)
    parser.add_argument("--opacity", type=float, default=0.05)
    parser.add_argument("--footprint_pixels", type=float, default=2.0)
    parser.add_argument("--budget", type=int, default=20)
    args = parser.parse_args()
    if not (0 < args.opacity <= 0.15 and args.max_condition > 1
            and args.displacement_factor > 0 and args.footprint_pixels > 0
            and args.budget > 0):
        parser.error("Receiver-v2 initialization parameters are invalid")
    build(args)


if __name__ == "__main__":
    main()
