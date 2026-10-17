#!/usr/bin/env python3
"""Three-training-view initialization audit for the v2-selection Anchor-Clone arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_i2_receiver_v2_initialization import metric_values, render_arm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def audit(args) -> dict:
    checkpoint = Path(args.checkpoint).resolve()
    v2_plan = Path(args.v2_plan).resolve()
    anchor_plan = Path(args.anchor_clone_plan).resolve()
    evidence_path = Path(args.v2_evidence).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="i2-v2-selection-anchor-clone-audit-") as directory:
        temp = Path(directory)
        release, cameras, release_event = render_arm(args, "release", v2_plan, temp)
        v2, _, v2_event = render_arm(args, "receiver_v2", v2_plan, temp)
        anchor, _, anchor_event = render_arm(args, "anchor_clone", anchor_plan, temp)

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

    images_by_arm = {
        "release_only": release,
        "receiver_v2_geometry_constrained": v2,
        "receiver_v2_selection_anchor_clone": anchor,
    }
    results = {}
    for arm, images in images_by_arm.items():
        totals = {key: 0.0 for key in ("PSNR", "SSIM", "LPIPS", "L1")}
        per_view = []
        target_abs_sum = target_pixels = outside_abs_sum = outside_pixels = 0.0
        change_sum = change_pixels = 0.0
        for name, image in images.items():
            target = cameras[name].original_image[:3].clamp(0, 1)
            values = metric_values(image, target)
            for key, value in values.items():
                totals[key] += value
            mask = masks[name]
            residual = (image - target).abs().mean(dim=0)
            change = (image - release[name]).abs().mean(dim=0)
            outside = ~mask
            target_abs_sum += float(residual[mask].sum().item())
            target_pixels += int(mask.sum().item())
            outside_abs_sum += float(residual[outside].sum().item())
            outside_pixels += int(outside.sum().item())
            change_sum += float(change[mask].sum().item())
            change_pixels += int(mask.sum().item())
            per_view.append({"image_name": name, **values})
        results[arm] = {
            "aggregate": {key: value / len(images) for key, value in totals.items()},
            "per_view": per_view,
            "v2_target_region_L1": target_abs_sum / target_pixels,
            "outside_target_region_L1": outside_abs_sum / outside_pixels,
            "v2_target_region_render_change_vs_release": change_sum / change_pixels,
        }

    anchor_metrics = results["receiver_v2_selection_anchor_clone"]
    release_metrics = results["release_only"]
    v2_metrics = results["receiver_v2_geometry_constrained"]
    finite = all(
        math.isfinite(value)
        for arm in results.values()
        for value in arm["aggregate"].values()
    )
    copy_audit = anchor_event.get("anchor_raw_parameter_copy", {})
    gates = {
        "training_views_only": True,
        "budget_and_alignment_pass": bool(
            anchor_event.get("budget_conserved")
            and anchor_event.get("alignment", {}).get("pass")
            and anchor_event.get("gaussian_before") == 65876
            and anchor_event.get("gaussian_after_release") == 65856
            and anchor_event.get("gaussian_after") == 65876
        ),
        "six_raw_parameters_exact": bool(
            set(copy_audit) == {
                "_xyz", "_features_dc", "_features_rest",
                "_opacity", "_scaling", "_rotation",
            } and all(copy_audit.values())
        ),
        "lifecycle_source_uid_order_exact": bool(
            anchor_event.get("lifecycle_source_uid_matches_anchor_order")
        ),
        "finite_metrics": finite,
        "mode_isolation": bool(
            release_event.get("format") == "i2_release_only_event_v1"
            and v2_event.get("format") == "i2_active_event_receiver_v2_v1"
            and anchor_event.get("format")
            == "i2_active_event_v2_selection_anchor_clone_v1"
        ),
    }

    def deltas(left, right):
        return {
            key: left["aggregate"][key] - right["aggregate"][key]
            for key in ("PSNR", "SSIM", "LPIPS", "L1")
        } | {
            "target_region_L1": left["v2_target_region_L1"] - right["v2_target_region_L1"],
            "outside_target_region_L1": (
                left["outside_target_region_L1"] - right["outside_target_region_L1"]
            ),
        }

    payload = {
        "format": "i2_v2_selection_anchor_clone_initialization_audit_v1",
        "status": "GO" if all(gates.values()) else "NO-GO",
        "source_iteration": 1201,
        "source_checkpoint_sha256": sha256(checkpoint),
        "receiver_v2_plan_sha256": sha256(v2_plan),
        "anchor_clone_plan_sha256": sha256(anchor_plan),
        "receiver_v2_evidence_sha256": sha256(evidence_path),
        "protocol": "three real training views; full-float; dropout=0; sigma_noise=0",
        "training_views_only": True,
        "test_views_used": False,
        "arms": results,
        "anchor_clone_minus_release": deltas(anchor_metrics, release_metrics),
        "anchor_clone_minus_receiver_v2": deltas(anchor_metrics, v2_metrics),
        "gates": gates,
        "anchor_clone_event": anchor_event,
        "scope_caveat": (
            "Execution/initialization audit only. Metric deltas are descriptive and "
            "were not used to select or tune Receiver candidates or initialization."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "GO":
        raise RuntimeError("NO-GO: Anchor-Clone initialization audit failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--v2_plan", required=True)
    parser.add_argument("--anchor_clone_plan", required=True)
    parser.add_argument("--v2_evidence", required=True)
    parser.add_argument("--output", required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
