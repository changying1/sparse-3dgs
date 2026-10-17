"""Formal in-memory evaluation and checkpoint alignment audit for I2-D1 5k."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def audit_checkpoint(path: Path, expected_iteration: int) -> dict:
    model_state, iteration = torch.load(path, map_location="cpu", weights_only=False)
    if iteration != expected_iteration or len(model_state) != 13:
        raise ValueError("Final checkpoint iteration/extension is invalid")
    count = int(model_state[1].shape[0])
    extension = model_state[12]
    lifecycle = extension["i2_lifecycle"]
    observation = extension["i2_observation"]
    uid = lifecycle["uid"]
    if uid.numel() != count or torch.unique(uid).numel() != count:
        raise ValueError("Final checkpoint UID rows are not unique/aligned")
    lifecycle_rows = {
        key: int(lifecycle[key].numel())
        for key in ("uid", "birth_iter", "birth_type", "source_uid", "generation")
    }
    if set(lifecycle_rows.values()) != {count}:
        raise ValueError("Final checkpoint lifecycle rows are misaligned")
    observation_rows = {
        key: int(observation[key].shape[0])
        for key in (
            "real_view_opportunities", "dropout_retained_hits",
            "screen_supported_hits", "screen_supported_view_history",
        )
    }
    if set(observation_rows.values()) != {count}:
        raise ValueError("Final checkpoint observation rows are misaligned")
    optimizer = model_state[10]
    optimizer_rows = {
        int(value["exp_avg"].shape[0])
        for value in optimizer["state"].values()
    }
    if optimizer_rows != {count} or len(optimizer["state"]) != 6:
        raise ValueError("Final checkpoint optimizer rows are misaligned")
    return {
        "iteration": iteration,
        "gaussian_count": count,
        "uid_count": int(uid.numel()),
        "uid_unique_count": int(torch.unique(uid).numel()),
        "lifecycle_rows": lifecycle_rows,
        "observation_rows": observation_rows,
        "optimizer_state_groups": len(optimizer["state"]),
        "optimizer_state_rows": sorted(optimizer_rows),
        "pass": True,
    }


def evaluate_views(name, cameras, gaussians, pipeline, background):
    rows = []
    totals = {metric: 0.0 for metric in ("PSNR", "SSIM", "LPIPS", "L1")}
    for index, camera in enumerate(cameras):
        # This is the training_report image protocol: full float render, no
        # Dropout and no test-time opacity scaling.  train=True is intentional.
        image = torch.clamp(
            render(
                camera, gaussians, pipeline, background,
                train=True, dropout_factor=0.0, sigma_noise=0.0,
            )["render"],
            0.0, 1.0,
        )
        target = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        values = {
            "PSNR": float(psnr(image, target).mean().item()),
            "SSIM": float(ssim(image, target).mean().item()),
            "LPIPS": float(lpips(image, target, net_type="vgg").item()),
            "L1": float(l1_loss(image, target).mean().item()),
        }
        rows.append({
            "index": index,
            "camera_uid": int(camera.uid),
            "image_name": str(camera.image_name),
            **values,
        })
        for metric, value in values.items():
            totals[metric] += value
    aggregate = {
        metric: value / len(rows) for metric, value in totals.items()
    }
    return {"split": name, "view_count": len(rows), "aggregate": aggregate, "per_view": rows}


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="LLFF")
    parser.add_argument("--n_views", type=int, default=3)
    args = get_combined_args(parser)
    safe_state(args.quiet)
    iteration = int(args.iteration)
    model_path = Path(args.model_path).resolve()
    checkpoint_audit = audit_checkpoint(
        model_path / f"chkpnt{iteration}.pth", iteration
    )
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(
            model.extract(args), gaussians, load_iteration=iteration, shuffle=False
        )
        if int(gaussians.get_xyz.shape[0]) != checkpoint_audit["gaussian_count"]:
            raise ValueError("PLY Gaussian count differs from final checkpoint")
        background = torch.tensor(
            [1, 1, 1] if args.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda",
        )
        test_result = evaluate_views(
            "test", scene.getTestCameras(), gaussians,
            pipeline.extract(args), background,
        )
        train_result = evaluate_views(
            "train", scene.getTrainCameras(), gaussians,
            pipeline.extract(args), background,
        )
        test_view_count = test_result["view_count"]
        result = {
            "format": "i2_d1_formal_metrics_v1",
            "iteration": iteration,
            "protocol": {
                "source": "in_memory_float_render",
                "clamp": "[0,1]",
                "dropout_factor": 0.0,
                "sigma_noise": 0.0,
                "opacity_scaling": False,
                "png_quantization": False,
                "test_view_count": test_view_count,
                "test_training_report_consistency": (
                    "Exact same full-float render, clamp, metric functions and all "
                    f"{test_view_count} test views as train.py training_report; camera ordering "
                    "does not affect the aggregate mean."
                ),
                "train_training_report_difference": (
                    "Formal output evaluates each of the three unique training views "
                    "once. train.py training_report uses five indexed samples with "
                    "repeats, so only the formal test aggregate is directly identical "
                    "in evaluation population."
                ),
            },
            "checkpoint_audit": checkpoint_audit,
            "test": test_result,
            "train": train_result,
        }
    output = model_path / f"formal_metrics_{iteration}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["test"]["aggregate"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
