#!/usr/bin/env python3
"""Read-only static-deletion RGB contribution audit for active-pilot donors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import PipelineParams
from gaussian_renderer import render
from tools.build_i2_receivers import load_runtime, parse_cfg, sha256


@torch.no_grad()
def audit(args):
    cfg = parse_cfg(args.cfg)
    model, cameras, iteration, _ = load_runtime(cfg, Path(args.checkpoint).resolve())
    checkpoint_sha = sha256(args.checkpoint)
    plans = [(Path(path).resolve(), json.loads(Path(path).read_text(encoding="utf-8")))
             for path in args.plans]
    donor_uids = sorted({
        int(row["uid"]) for _, plan in plans for row in plan["donors"]["selected"]
    })
    uid_rows = {int(uid): row for row, uid in enumerate(model.lifecycle.uid.detach().cpu().tolist())}
    if any(uid not in uid_rows for uid in donor_uids):
        raise ValueError("A planned donor UID is not live in the source checkpoint")
    background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    parser = argparse.ArgumentParser(add_help=False)
    group = PipelineParams(parser)
    pipe = group.extract(parser.parse_args([]))
    baselines = []
    baseline_psnr = []
    for camera in cameras:
        image = render(camera, model, pipe, background, train=False,
                       dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
        gt = camera.original_image[:3]
        mse = (image - gt).square().mean().clamp_min(1e-12)
        baselines.append(image)
        baseline_psnr.append(float(-10 * torch.log10(mse)))
    rows = []
    for index, uid in enumerate(donor_uids):
        row = uid_rows[uid]
        original = model._opacity[row].detach().clone()
        model._opacity[row].fill_(-100.0)
        view_rows = []
        try:
            for view_index, (camera, baseline, base_psnr) in enumerate(
                zip(cameras, baselines, baseline_psnr)
            ):
                deleted = render(camera, model, pipe, background, train=False,
                                 dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
                delta = (deleted - baseline).abs()
                gt = camera.original_image[:3]
                mse = (deleted - gt).square().mean().clamp_min(1e-12)
                psnr = float(-10 * torch.log10(mse))
                view_rows.append({
                    "training_camera_uid": int(camera.uid), "training_camera_id": camera.image_name,
                    "mean_abs_rgb_delta": float(delta.mean()),
                    "max_abs_rgb_delta": float(delta.max()),
                    "psnr_drop_db": float(base_psnr - psnr),
                })
        finally:
            model._opacity[row].copy_(original)
        mean_delta = max(item["mean_abs_rgb_delta"] for item in view_rows)
        max_delta = max(item["max_abs_rgb_delta"] for item in view_rows)
        max_psnr_drop = max(item["psnr_drop_db"] for item in view_rows)
        finite = all(math.isfinite(value) for value in (mean_delta, max_delta, max_psnr_drop))
        rows.append({
            "uid": uid, "max_mean_abs_rgb_delta": mean_delta,
            "max_abs_rgb_delta": max_delta, "max_psnr_drop_db": max_psnr_drop,
            "safe_to_release": bool(
                finite and mean_delta <= args.max_mean_abs_rgb_delta
                and max_psnr_drop <= args.max_psnr_drop_db
            ),
            "per_view": view_rows,
        })
        print(f"counterfactual {index + 1}/{len(donor_uids)} uid={uid}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "donor_static_deletion_counterfactual.json"
    payload = {
        "format": "i2_static_deletion_counterfactual_v1", "source_iteration": iteration,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "training_view_count": len(cameras), "test_views_used": False,
        "thresholds": {"max_mean_abs_rgb_delta": args.max_mean_abs_rgb_delta,
                       "max_psnr_drop_db": args.max_psnr_drop_db},
        "audited_count": len(rows), "safe_count": sum(row["safe_to_release"] for row in rows),
        "rows": rows,
    }
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_ref = {"path": str(artifact), "sha256": sha256(artifact)}
    by_uid = {row["uid"]: row for row in rows}
    outputs = []
    for source_path, plan in plans:
        selected = [by_uid[int(row["uid"])] for row in plan["donors"]["selected"]]
        selected_uids = [int(row["uid"]) for row in plan["donors"]["selected"]]
        plan["donor_counterfactual"] = {
            "format": payload["format"], "artifact": artifact_ref,
            "source_checkpoint_sha256": checkpoint_sha, "rows": selected,
        }
        ready = all(row["safe_to_release"] for row in selected)
        originals = {
            uid: model._opacity[uid_rows[uid]].detach().clone() for uid in selected_uids
        }
        for uid in selected_uids:
            model._opacity[uid_rows[uid]].fill_(-100.0)
        joint_views = []
        try:
            for view_index, (camera, baseline, base_psnr) in enumerate(
                zip(cameras, baselines, baseline_psnr)
            ):
                deleted = render(camera, model, pipe, background, train=False,
                                 dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
                delta = (deleted - baseline).abs()
                gt = camera.original_image[:3]
                mse = (deleted - gt).square().mean().clamp_min(1e-12)
                psnr = float(-10 * torch.log10(mse))
                single_sum = sum(row["per_view"][view_index]["mean_abs_rgb_delta"]
                                 for row in selected)
                joint_mean = float(delta.mean())
                joint_views.append({
                    "training_camera_uid": int(camera.uid),
                    "training_camera_id": camera.image_name,
                    "mean_abs_rgb_delta": joint_mean,
                    "max_abs_rgb_delta": float(delta.max()),
                    "psnr_drop_db": float(base_psnr - psnr),
                    "single_point_sum_mean_abs_rgb_delta": single_sum,
                    "joint_minus_single_sum_mean_abs_rgb_delta": joint_mean - single_sum,
                    "single_point_max_mean_abs_rgb_delta": max(
                        row["per_view"][view_index]["mean_abs_rgb_delta"] for row in selected
                    ),
                })
        finally:
            for uid, original in originals.items():
                model._opacity[uid_rows[uid]].copy_(original)
        joint_mean = max(row["mean_abs_rgb_delta"] for row in joint_views)
        joint_psnr = max(row["psnr_drop_db"] for row in joint_views)
        joint_safe = bool(
            math.isfinite(joint_mean) and math.isfinite(joint_psnr)
            and joint_mean <= args.max_mean_abs_rgb_delta
            and joint_psnr <= args.max_psnr_drop_db
        )
        joint_payload = {
            "format": "i2_joint_deletion_counterfactual_v1",
            "source_iteration": iteration, "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_checkpoint_sha256": checkpoint_sha, "training_view_count": len(cameras),
            "test_views_used": False, "donor_uids": selected_uids,
            "thresholds": {"max_mean_abs_rgb_delta": args.max_mean_abs_rgb_delta,
                           "max_psnr_drop_db": args.max_psnr_drop_db},
            "max_mean_abs_rgb_delta": joint_mean, "max_psnr_drop_db": joint_psnr,
            "safe_to_release": joint_safe, "per_view": joint_views,
        }
        joint_path = output / f"donor_joint_deletion_counterfactual_k{len(selected_uids)}.json"
        joint_path.write_text(json.dumps(joint_payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        plan["joint_deletion_counterfactual"] = {
            "format": joint_payload["format"],
            "artifact": {"path": str(joint_path), "sha256": sha256(joint_path)},
        }
        plan["safety_readiness"]["static_deletion_counterfactual"] = ready
        plan["safety_readiness"]["joint_deletion_counterfactual"] = joint_safe
        plan["active_intervention_ready"] = (
            plan["safety_readiness"]["b1_only_exploratory"] is True and ready and joint_safe
        )
        target = output / f"{source_path.parent.name}_active_plan.json"
        target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(str(target))
    print(json.dumps({"artifact": str(artifact), "audited": len(rows),
                      "safe": payload["safe_count"], "plans": outputs}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--plans", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_mean_abs_rgb_delta", type=float, default=1e-4)
    parser.add_argument("--max_psnr_drop_db", type=float, default=0.01)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
