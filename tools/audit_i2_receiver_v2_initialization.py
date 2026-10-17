#!/usr/bin/env python3
"""Immediate three-training-view audit for Release, Receiver-v1 and Receiver-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import PipelineParams
from gaussian_renderer import render
from lpipsPyTorch import lpips
from tools.build_i2_receivers import load_runtime, parse_cfg
from utils.i2_active_pilot import apply_active_reallocation, apply_release_only, load_active_plan
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def render_arm(args, arm, plan_path, temp: Path):
    cfg = parse_cfg(args.cfg)
    model, cameras, iteration, _ = load_runtime(cfg, args.checkpoint)
    plan = load_active_plan(
        plan_path, requested_budget=args.budget, source_checkpoint=args.checkpoint
    )
    event_path = temp / f"{arm}.json"
    if arm == "release":
        apply_release_only(model, plan, iteration=iteration, event_log=event_path)
    else:
        apply_active_reallocation(model, plan, iteration=iteration, event_log=event_path)
    parser = argparse.ArgumentParser(add_help=False)
    pipeline = PipelineParams(parser).extract(parser.parse_args([]))
    background = torch.tensor(
        [1, 1, 1] if cfg.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    images = {}
    for camera in cameras:
        images[camera.image_name] = render(
            camera, model, pipeline, background, train=True,
            dropout_factor=0.0, sigma_noise=0.0,
        )["render"].clamp(0, 1)
    return images, {camera.image_name: camera for camera in cameras}, json.loads(event_path.read_text())


def metric_values(image, target):
    return {
        "PSNR": float(psnr(image, target).mean().item()),
        "SSIM": float(ssim(image, target).mean().item()),
        "LPIPS": float(lpips(image, target, net_type="vgg").item()),
        "L1": float(l1_loss(image, target).mean().item()),
    }


@torch.no_grad()
def audit(args):
    checkpoint = Path(args.checkpoint).resolve()
    v1_plan = Path(args.v1_plan).resolve()
    v2_plan = Path(args.v2_plan).resolve()
    evidence_path = Path(args.v2_evidence).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="i2-receiver-v2-audit-") as directory:
        temp = Path(directory)
        release, cameras, release_event = render_arm(args, "release", v1_plan, temp)
        v1, _, v1_event = render_arm(args, "receiver_v1", v1_plan, temp)
        v2, _, v2_event = render_arm(args, "receiver_v2", v2_plan, temp)

    masks = {
        name: torch.zeros(
            (int(camera.image_height), int(camera.image_width)),
            dtype=torch.bool, device="cuda",
        )
        for name, camera in cameras.items()
    }
    for row in evidence["rows"]:
        name = Path(row["training_camera_id"]).stem
        x0, y0, x1, y1 = row["pixel_region"]
        masks[name][y0:y1, x0:x1] = True

    arms = {"release_only": release, "receiver_v1": v1, "receiver_v2": v2}
    results = {}
    for arm, images in arms.items():
        per_view = []
        totals = {key: 0.0 for key in ("PSNR", "SSIM", "LPIPS", "L1")}
        target_abs_sum = target_pixels = outside_abs_sum = outside_pixels = 0.0
        coverage_sum = coverage_pixels = 0.0
        for name, image in images.items():
            target = cameras[name].original_image[:3].clamp(0, 1)
            values = metric_values(image, target)
            for key, value in values.items():
                totals[key] += value
            mask = masks[name]
            residual = (image - target).abs().mean(dim=0)
            outside = ~mask
            if bool(mask.any()):
                target_abs_sum += float(residual[mask].sum().item())
                target_pixels += int(mask.sum().item())
                coverage_sum += float((image - release[name]).abs().mean(dim=0)[mask].sum().item())
                coverage_pixels += int(mask.sum().item())
            outside_abs_sum += float(residual[outside].sum().item())
            outside_pixels += int(outside.sum().item())
            per_view.append({"image_name": name, **values})
        results[arm] = {
            "aggregate": {key: value / len(images) for key, value in totals.items()},
            "per_view": per_view,
            "v2_target_region_L1": target_abs_sum / target_pixels,
            "outside_target_region_L1": outside_abs_sum / outside_pixels,
            "v2_target_region_render_change_vs_release": (
                coverage_sum / coverage_pixels if coverage_pixels else 0.0
            ),
        }

    release_metrics = results["release_only"]
    v2_metrics = results["receiver_v2"]
    target_delta = (
        v2_metrics["v2_target_region_L1"] - release_metrics["v2_target_region_L1"]
    )
    outside_delta = (
        v2_metrics["outside_target_region_L1"]
        - release_metrics["outside_target_region_L1"]
    )
    global_delta = v2_metrics["aggregate"]["L1"] - release_metrics["aggregate"]["L1"]
    gates = {
        "receiver_v2_changes_target_regions":
            v2_metrics["v2_target_region_render_change_vs_release"] > 0,
        "target_region_not_worse_than_release": target_delta <= args.max_target_l1_increase,
        "outside_region_no_obvious_degradation": outside_delta <= args.max_outside_l1_increase,
        "global_no_obvious_degradation": global_delta <= args.max_global_l1_increase,
        "budget_and_alignment_pass": bool(
            v2_event["budget_conserved"] and v2_event["alignment"]["pass"]
        ),
    }
    payload = {
        "format": "i2_receiver_v2_initialization_audit_v1",
        "source_iteration": 1201,
        "source_checkpoint_sha256": sha256(checkpoint),
        "receiver_v1_plan_sha256": sha256(v1_plan),
        "receiver_v2_plan_sha256": sha256(v2_plan),
        "receiver_v2_evidence_sha256": sha256(evidence_path),
        "training_views_only": True,
        "test_views_used": False,
        "protocol": "full-float render, dropout=0, sigma_noise=0",
        "gate_thresholds": {
            "max_target_l1_increase": args.max_target_l1_increase,
            "max_outside_l1_increase": args.max_outside_l1_increase,
            "max_global_l1_increase": args.max_global_l1_increase,
        },
        "arms": results,
        "receiver_v2_minus_release": {
            "target_region_L1": target_delta,
            "outside_target_region_L1": outside_delta,
            "global_L1": global_delta,
        },
        "gates": gates,
        "status": "GO" if all(gates.values()) else "NO-GO",
        "scope_caveat": "Initialization feasibility/risk audit only; not final reconstruction evidence.",
        "events": {
            "release": release_event["format"], "receiver_v1": v1_event["format"],
            "receiver_v2": v2_event["format"],
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "GO":
        raise RuntimeError("NO-GO: Receiver-v2 initialization audit failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--v1_plan", required=True)
    parser.add_argument("--v2_plan", required=True)
    parser.add_argument("--v2_evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_target_l1_increase", type=float, default=0.0)
    parser.add_argument("--max_outside_l1_increase", type=float, default=5e-5)
    parser.add_argument("--max_global_l1_increase", type=float, default=5e-5)
    parser.add_argument("--budget", type=int, default=20)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
