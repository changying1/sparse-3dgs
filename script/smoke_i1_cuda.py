"""Read-only real-rasterizer CUDA smokes for Innovation 1 Phases 2-A and 2-B.

This script is intentionally read-only with respect to the frozen experiment:
it loads iteration 5000, renders one frozen training camera, computes gradients
with ``torch.autograd.grad``, and writes diagnostics to a new external folder.
It never creates an optimizer, steps parameters, densifies, prunes, or runs MiDaS.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.evidence_structural_loss import compute_oe_structural_loss  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.oe_depth_cache import validate_rendered_depth  # noqa: E402
from utils.oe_densification import (  # noqa: E402
    add_densification_stats_with_baseline_gradient,
    extract_baseline_viewspace_gradient,
    map_dropout_visibility_to_full_mask,
)


REPORT_NAMES = {
    "phase2a": "i1_phase2a_cuda_smoke_report.json",
    "phase2b": "i1_phase2b_cuda_smoke_report.json",
}
LOG_NAMES = {
    "phase2a": "i1_phase2a_cuda_smoke.log",
    "phase2b": "i1_phase2b_cuda_smoke.log",
}
EXCEPTION_NAME = "exception.txt"
DEFAULT_ITERATION = 5000
DEFAULT_DROPOUT_FACTORS = (0.0, 0.2)


class SmokeCheckError(RuntimeError):
    pass


class RecordingDensificationProxy:
    """Observe the gradient passed to stats without touching the real model."""

    def __init__(self, *, error=None):
        self.error = error
        self.gradient_seen = None
        self.mask_seen = None

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gradient_seen = viewspace_point_tensor.grad.detach().clone()
        self.mask_seen = update_filter.detach().clone()
        if self.error is not None:
            raise self.error


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_directory(output_dir, frozen_model_dir):
    """Create one new report directory and reject every overwrite/frozen child."""
    output = Path(output_dir).resolve()
    frozen = Path(frozen_model_dir).resolve()
    if output == frozen or _is_relative_to(output, frozen):
        raise SmokeCheckError("Smoke output_dir must not be inside the frozen Baseline directory.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def synthetic_reference_depth(height, width, *, device, dtype):
    """Deterministic synthetic reference; never represents MiDaS or I1 quality."""
    y = torch.linspace(0.0, 1.0, int(height), device=device, dtype=dtype)[:, None]
    x = torch.linspace(0.0, 1.0, int(width), device=device, dtype=dtype)[None, :]
    return (210.0 - 9.0 * x - 17.0 * y + 2.0 * x * y).detach()


def gradient_stats(gradient):
    if gradient is None:
        return {
            "present": False,
            "finite": False,
            "element_count": 0,
            "nonzero_count": 0,
            "l1": None,
            "l2": None,
            "max_abs": None,
        }
    detached = gradient.detach()
    finite = torch.isfinite(detached)
    return {
        "present": True,
        "finite": bool(finite.all().item()),
        "element_count": int(detached.numel()),
        "nonzero_count": int(torch.count_nonzero(detached).item()),
        "l1": float(detached.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(detached.float()).item()),
        "max_abs": float(detached.abs().max().item()) if detached.numel() else 0.0,
    }


def gaussian_parameter_items(gaussians):
    return [
        ("xyz", gaussians._xyz),
        ("features_dc", gaussians._features_dc),
        ("features_rest", gaussians._features_rest),
        ("opacity", gaussians._opacity),
        ("scaling", gaussians._scaling),
        ("rotation", gaussians._rotation),
    ]


def clear_gaussian_parameter_gradients(gaussians):
    for _, parameter in gaussian_parameter_items(gaussians):
        parameter.grad = None


def _clone_parameter_gradients(parameter_items):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameter_items
    }


def _parameter_gradients_equal(parameter_items, expected):
    for name, parameter in parameter_items:
        current = parameter.grad
        saved = expected[name]
        if (current is None) != (saved is None):
            return False
        if current is not None and not torch.equal(current, saved):
            return False
    return True


def _grad_field_unchanged(current, previous, previous_copy):
    if previous is None:
        return current is None
    return current is previous and torch.equal(current, previous_copy)


def _timed_call(function, tensor):
    if tensor.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(tensor.device)
        start.record()
        result = function()
        end.record()
        torch.cuda.synchronize(tensor.device)
        return result, float(start.elapsed_time(end))
    started = time.perf_counter()
    result = function()
    return result, float((time.perf_counter() - started) * 1000.0)


def audit_dropout_indexing(
    dropout_mask,
    radii,
    visibility_filter,
    viewspace_point_tensor,
    *,
    dropout_factor,
    baseline_viewspace_grad=None,
    combined_viewspace_grad=None,
):
    """Validate the exact subset-to-full indexing used by training densification."""
    if dropout_mask.ndim != 1 or dropout_mask.dtype != torch.bool:
        raise SmokeCheckError("dropout_mask must be a 1D bool tensor.")
    if radii.ndim != 1 or visibility_filter.ndim != 1 or visibility_filter.dtype != torch.bool:
        raise SmokeCheckError("radii and visibility_filter must be aligned 1D tensors.")
    if radii.shape != visibility_filter.shape:
        raise SmokeCheckError("radii and visibility_filter shapes differ.")
    gaussian_count = int(dropout_mask.numel())
    kept_indices = torch.nonzero(dropout_mask, as_tuple=True)[0]
    kept_count = int(kept_indices.numel())
    if viewspace_point_tensor.ndim != 2 or viewspace_point_tensor.shape != (gaussian_count, 3):
        raise SmokeCheckError("viewspace_point_tensor is not full-Gaussian [N,3].")
    if int(radii.numel()) != kept_count:
        raise SmokeCheckError(
            f"Raster subset mismatch: radii={radii.numel()}, dropout-kept={kept_count}."
        )
    if float(dropout_factor) == 0.0 and kept_count != gaussian_count:
        raise SmokeCheckError("dropout_factor=0 unexpectedly removed Gaussians.")
    if float(dropout_factor) > 0.0 and not 0 < kept_count < gaussian_count:
        raise SmokeCheckError("Positive dropout did not produce both kept and dropped Gaussians.")

    visible_full_indices = kept_indices[visibility_filter]
    combined_mask = torch.zeros_like(dropout_mask, dtype=torch.bool)
    combined_mask[visible_full_indices] = True
    if int(combined_mask.sum().item()) != int(visibility_filter.sum().item()):
        raise SmokeCheckError("Visible subset did not map one-to-one to full Gaussian indices.")
    if bool((combined_mask & ~dropout_mask).any().item()):
        raise SmokeCheckError("combined_mask references a dropped Gaussian.")

    result = {
        "dropout_mask_shape": list(dropout_mask.shape),
        "radii_shape": list(radii.shape),
        "visibility_filter_shape": list(visibility_filter.shape),
        "viewspace_point_tensor_shape": list(viewspace_point_tensor.shape),
        "combined_mask_shape": list(combined_mask.shape),
        "gaussian_count": gaussian_count,
        "dropout_kept_count": kept_count,
        "dropout_dropped_count": gaussian_count - kept_count,
        "radii_count": int(radii.numel()),
        "visible_subset_count": int(visibility_filter.sum().item()),
        "combined_mask_count": int(combined_mask.sum().item()),
        "mapping_is_exact": True,
    }
    for label, gradient in (
        ("baseline", baseline_viewspace_grad),
        ("baseline_plus_i1", combined_viewspace_grad),
    ):
        if gradient is None:
            result[f"{label}_dropped_grad_nonzero_count"] = None
            continue
        if gradient.shape != viewspace_point_tensor.shape:
            raise SmokeCheckError(f"{label} viewspace gradient shape differs from [N,3].")
        dropped_nonzero = torch.count_nonzero(gradient[~dropout_mask]).item()
        result[f"{label}_dropped_grad_nonzero_count"] = int(dropped_nonzero)
        if dropped_nonzero:
            raise SmokeCheckError(f"{label} gradient leaked into dropped Gaussian indices.")
    return combined_mask, result


def baseline_photometric_loss(render_pkg, gt_image, lambda_dssim, gt_alpha_mask=None):
    image = render_pkg["render"]
    loss = (1.0 - float(lambda_dssim)) * l1_loss(image, gt_image)
    loss = loss + float(lambda_dssim) * (1.0 - ssim(image, gt_image))
    if gt_alpha_mask is not None:
        loss = loss + torch.mean(
            torch.abs(render_pkg["rendered_alpha"]) * (1 - gt_alpha_mask)
        )
    return loss


def evaluate_render_case(
    render_pkg,
    gaussians,
    camera,
    *,
    dropout_factor,
    lambda_dssim,
    structural_weight,
    structural_loss_fn=compute_oe_structural_loss,
):
    """Evaluate one differentiable render graph without mutating any parameter."""
    depth_raw = render_pkg["rendered_depth"]
    depth = validate_rendered_depth(depth_raw, camera.image_height, camera.image_width)
    reference = synthetic_reference_depth(
        camera.image_height,
        camera.image_width,
        device=depth.device,
        dtype=depth.dtype,
    )
    structural_loss, structural_stats, _ = structural_loss_fn(
        depth,
        reference,
        gt_rgb=camera.original_image,
        gate_mode="none",
    )
    baseline_loss = baseline_photometric_loss(
        render_pkg,
        camera.original_image,
        lambda_dssim,
        getattr(camera, "gt_alpha_mask", None),
    )
    total_loss = baseline_loss + float(structural_weight) * structural_loss

    viewspace = render_pkg["viewspace_points"]
    parameter_items = [
        ("xyz", gaussians._xyz),
        ("opacity", gaussians._opacity),
        ("scaling", gaussians._scaling),
        ("rotation", gaussians._rotation),
    ]
    baseline_viewspace_grad = torch.autograd.grad(
        baseline_loss, viewspace, retain_graph=True, allow_unused=True
    )[0]
    combined_viewspace_grad = torch.autograd.grad(
        total_loss, viewspace, retain_graph=True, allow_unused=True
    )[0]
    structural_gradients = torch.autograd.grad(
        structural_loss,
        [depth] + [parameter for _, parameter in parameter_items],
        retain_graph=False,
        allow_unused=True,
    )
    depth_gradient = structural_gradients[0]
    parameter_gradients = structural_gradients[1:]

    combined_mask, index_audit = audit_dropout_indexing(
        render_pkg["dropout_mask"],
        render_pkg["radii"],
        render_pkg["visibility_filter"],
        viewspace,
        dropout_factor=dropout_factor,
        baseline_viewspace_grad=baseline_viewspace_grad,
        combined_viewspace_grad=combined_viewspace_grad,
    )
    baseline_stats = gradient_stats(baseline_viewspace_grad)
    combined_stats = gradient_stats(combined_viewspace_grad)
    gradient_delta = (
        None
        if baseline_viewspace_grad is None or combined_viewspace_grad is None
        else combined_viewspace_grad - baseline_viewspace_grad
    )
    delta_stats = gradient_stats(gradient_delta)
    gaussian_stats = {
        name: gradient_stats(gradient)
        for (name, _), gradient in zip(parameter_items, parameter_gradients)
    }

    checks = {
        "depth_shape": tuple(depth.shape) == (int(camera.image_height), int(camera.image_width)),
        "depth_finite": bool(torch.isfinite(depth).all().item()),
        "depth_requires_grad": bool(depth.requires_grad),
        "depth_has_grad_fn": depth.grad_fn is not None,
        "structural_loss_finite": bool(torch.isfinite(structural_loss).all().item()),
        "structural_loss_nonzero": bool(structural_loss.detach().abs().item() > 0),
        "depth_gradient_finite_nonzero": bool(
            gradient_stats(depth_gradient)["finite"] and gradient_stats(depth_gradient)["nonzero_count"] > 0
        ),
        "xyz_gradient_finite_nonzero": bool(
            gaussian_stats["xyz"]["finite"] and gaussian_stats["xyz"]["nonzero_count"] > 0
        ),
        "baseline_densification_gradient_extractable": bool(
            baseline_stats["finite"] and baseline_stats["nonzero_count"] > 0
        ),
        "i1_changes_densification_gradient": bool(
            delta_stats["finite"] and delta_stats["nonzero_count"] > 0
        ),
        "dropout_index_mapping_exact": index_audit["mapping_is_exact"],
    }
    result = {
        "dropout_factor": float(dropout_factor),
        "synthetic_reference": True,
        "synthetic_reference_purpose": "gradient connectivity only; not an I1 quality result",
        "rendered_depth": {
            "raw_shape": list(depth_raw.shape),
            "validated_shape": list(depth.shape),
            "requires_grad": bool(depth.requires_grad),
            "grad_fn": type(depth.grad_fn).__name__ if depth.grad_fn is not None else None,
            "min": float(depth.detach().min().item()),
            "max": float(depth.detach().max().item()),
        },
        "losses": {
            "baseline_loss": float(baseline_loss.detach().item()),
            "oe_struct_loss": float(structural_loss.detach().item()),
            "weighted_oe_loss": float((float(structural_weight) * structural_loss).detach().item()),
            "baseline_plus_i1_loss": float(total_loss.detach().item()),
        },
        "structural_stats": _json_safe(structural_stats),
        "depth_structural_gradient": gradient_stats(depth_gradient),
        "gaussian_structural_gradients": gaussian_stats,
        "baseline_viewspace_gradient": baseline_stats,
        "baseline_plus_i1_viewspace_gradient": combined_stats,
        "viewspace_gradient_delta": delta_stats,
        "dropout_index_audit": index_audit,
        "densification_visible_gradient": {
            "baseline": gradient_stats(
                None if baseline_viewspace_grad is None else baseline_viewspace_grad[combined_mask, :2]
            ),
            "baseline_plus_i1": gradient_stats(
                None if combined_viewspace_grad is None else combined_viewspace_grad[combined_mask, :2]
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    return result


def evaluate_densification_protection_case(
    render_pkg,
    gaussians,
    camera,
    *,
    dropout_factor,
    lambda_dssim,
    structural_weight,
    structural_loss_fn=compute_oe_structural_loss,
):
    """Exercise the production Phase 2-B helpers on one render graph."""
    depth_raw = render_pkg["rendered_depth"]
    depth = validate_rendered_depth(depth_raw, camera.image_height, camera.image_width)
    reference = synthetic_reference_depth(
        camera.image_height,
        camera.image_width,
        device=depth.device,
        dtype=depth.dtype,
    )
    structural_loss, structural_stats, _ = structural_loss_fn(
        depth,
        reference,
        gt_rgb=camera.original_image,
        gate_mode="none",
    )
    baseline_loss = baseline_photometric_loss(
        render_pkg,
        camera.original_image,
        lambda_dssim,
        getattr(camera, "gt_alpha_mask", None),
    )
    total_loss = baseline_loss + float(structural_weight) * structural_loss

    viewspace = render_pkg["viewspace_points"]
    parameter_items = gaussian_parameter_items(gaussians)
    pre_extract_parameter_gradients = _clone_parameter_gradients(parameter_items)
    pre_extract_grad = viewspace.grad
    pre_extract_copy = None if pre_extract_grad is None else pre_extract_grad.detach().clone()
    baseline_viewspace_grad, extraction_ms = _timed_call(
        lambda: extract_baseline_viewspace_gradient(baseline_loss, viewspace),
        viewspace,
    )
    extraction_preserved_grad = _grad_field_unchanged(
        viewspace.grad,
        pre_extract_grad,
        pre_extract_copy,
    )
    extraction_preserved_parameter_grads = _parameter_gradients_equal(
        parameter_items, pre_extract_parameter_gradients
    )

    _, joint_backward_ms = _timed_call(total_loss.backward, viewspace)
    joint_viewspace_grad = viewspace.grad
    if joint_viewspace_grad is None:
        raise SmokeCheckError("total_loss.backward() did not populate the viewspace gradient.")
    joint_viewspace_copy = joint_viewspace_grad.detach().clone()
    parameter_gradients = _clone_parameter_gradients(parameter_items)

    production_mask = map_dropout_visibility_to_full_mask(
        render_pkg["dropout_mask"],
        render_pkg["visibility_filter"],
        viewspace,
    )
    audited_mask, index_audit = audit_dropout_indexing(
        render_pkg["dropout_mask"],
        render_pkg["radii"],
        render_pkg["visibility_filter"],
        viewspace,
        dropout_factor=dropout_factor,
        baseline_viewspace_grad=baseline_viewspace_grad,
        combined_viewspace_grad=joint_viewspace_copy,
    )
    mask_matches_audit = torch.equal(production_mask, audited_mask)

    proxy = RecordingDensificationProxy()
    add_densification_stats_with_baseline_gradient(
        proxy,
        viewspace,
        production_mask,
        baseline_viewspace_grad,
    )
    proxy_received_baseline = torch.equal(proxy.gradient_seen, baseline_viewspace_grad)
    proxy_received_joint = torch.equal(proxy.gradient_seen, joint_viewspace_copy)
    proxy_received_mask = torch.equal(proxy.mask_seen, production_mask)
    joint_restored = (
        viewspace.grad is joint_viewspace_grad
        and torch.equal(viewspace.grad, joint_viewspace_copy)
    )
    parameters_unchanged = _parameter_gradients_equal(parameter_items, parameter_gradients)

    raising_proxy = RecordingDensificationProxy(error=RuntimeError("synthetic proxy failure"))
    exception_observed = False
    try:
        add_densification_stats_with_baseline_gradient(
            raising_proxy,
            viewspace,
            production_mask,
            baseline_viewspace_grad,
        )
    except RuntimeError as exc:
        if str(exc) != "synthetic proxy failure":
            raise
        exception_observed = True
    exception_restored_joint = (
        viewspace.grad is joint_viewspace_grad
        and torch.equal(viewspace.grad, joint_viewspace_copy)
    )
    exception_preserved_parameters = _parameter_gradients_equal(
        parameter_items, parameter_gradients
    )
    exception_proxy_received_baseline = torch.equal(
        raising_proxy.gradient_seen, baseline_viewspace_grad
    )

    baseline_stats = gradient_stats(baseline_viewspace_grad)
    joint_stats = gradient_stats(joint_viewspace_copy)
    delta_stats = gradient_stats(joint_viewspace_copy - baseline_viewspace_grad)
    gaussian_stats = {
        name: gradient_stats(parameter_gradients[name]) for name, _ in parameter_items
    }
    all_parameter_gradients_finite = all(
        stats["present"] and stats["finite"] for stats in gaussian_stats.values()
    )
    any_parameter_gradient_nonzero = any(
        stats["nonzero_count"] > 0 for stats in gaussian_stats.values()
    )
    checks = {
        "baseline_gradient_extracted": baseline_stats["present"] and baseline_stats["finite"],
        "extraction_did_not_pollute_grad": extraction_preserved_grad,
        "extraction_did_not_pollute_gaussian_parameter_grads": (
            extraction_preserved_parameter_grads
        ),
        "joint_viewspace_gradient_finite_nonzero": (
            joint_stats["finite"] and joint_stats["nonzero_count"] > 0
        ),
        "i1_changes_viewspace_gradient": delta_stats["finite"] and delta_stats["nonzero_count"] > 0,
        "gaussian_parameter_gradients_present_finite": all_parameter_gradients_finite,
        "gaussian_parameter_gradient_nonzero": any_parameter_gradient_nonzero,
        "xyz_joint_gradient_nonzero": gaussian_stats["xyz"]["nonzero_count"] > 0,
        "dropout_index_mapping_exact": index_audit["mapping_is_exact"] and mask_matches_audit,
        "proxy_received_baseline_gradient": proxy_received_baseline,
        "proxy_did_not_receive_joint_gradient": not proxy_received_joint,
        "proxy_received_combined_mask": proxy_received_mask,
        "joint_viewspace_gradient_restored": joint_restored,
        "gaussian_parameter_gradients_unchanged": parameters_unchanged,
        "exception_path_observed": exception_observed,
        "exception_path_restored_joint_gradient": exception_restored_joint,
        "exception_path_preserved_parameter_gradients": exception_preserved_parameters,
        "exception_proxy_received_baseline_gradient": exception_proxy_received_baseline,
    }
    return {
        "dropout_factor": float(dropout_factor),
        "synthetic_reference": True,
        "synthetic_reference_purpose": "gradient protection connectivity only; not an I1 quality result",
        "losses": {
            "baseline_loss": float(baseline_loss.detach().item()),
            "oe_struct_loss": float(structural_loss.detach().item()),
            "weighted_oe_loss": float(
                (float(structural_weight) * structural_loss).detach().item()
            ),
            "baseline_plus_i1_loss": float(total_loss.detach().item()),
        },
        "structural_stats": _json_safe(structural_stats),
        "timing_ms": {
            "baseline_gradient_extraction": extraction_ms,
            "joint_backward": joint_backward_ms,
        },
        "baseline_viewspace_gradient": baseline_stats,
        "baseline_plus_i1_viewspace_gradient": joint_stats,
        "viewspace_gradient_delta": delta_stats,
        "densification_visible_gradient": {
            "baseline": gradient_stats(baseline_viewspace_grad[production_mask, :2]),
            "baseline_plus_i1": gradient_stats(joint_viewspace_copy[production_mask, :2]),
            "proxy_received": gradient_stats(proxy.gradient_seen[production_mask, :2]),
        },
        "gaussian_joint_gradients": gaussian_stats,
        "dropout_index_audit": index_audit,
        "proxy_audit": {
            "received_gradient_equals_baseline": proxy_received_baseline,
            "received_gradient_equals_joint": proxy_received_joint,
            "received_mask_equals_combined_mask": proxy_received_mask,
            "normal_path_restored_joint_gradient": joint_restored,
            "exception_path_observed": exception_observed,
            "exception_path_restored_joint_gradient": exception_restored_joint,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_cuda_smoke(args, dataset, pipeline, report):
    if not torch.cuda.is_available():
        raise SmokeCheckError("CUDA is not available.")
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    model_dir = Path(dataset.model_path).resolve()
    point_cloud = model_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise SmokeCheckError(f"Frozen Gaussian PLY does not exist: {point_cloud}")
    frozen_before = sha256_file(point_cloud)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.cuda.set_device(int(args.gpu))
    torch.cuda.reset_peak_memory_stats()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if gaussians.optimizer is not None:
        raise SmokeCheckError("Smoke unexpectedly created a Gaussian optimizer.")
    cameras = scene.getTrainCameras()
    if not 0 <= int(args.camera_index) < len(cameras):
        raise SmokeCheckError(f"camera_index {args.camera_index} is outside {len(cameras)} training views.")
    camera = cameras[int(args.camera_index)]
    gaussian_count = int(gaussians.get_xyz.shape[0])
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    report.update({
        "device": {
            "name": torch.cuda.get_device_name(int(args.gpu)),
            "index": int(args.gpu),
            "capability": list(torch.cuda.get_device_capability(int(args.gpu))),
        },
        "iteration": int(args.iteration),
        "camera_index": int(args.camera_index),
        "camera_image_name": str(camera.image_name),
        "gaussian_count_before": gaussian_count,
        "synthetic_reference_only": True,
        "cases": [],
    })
    case_peak_memory = []
    for dropout_factor in DEFAULT_DROPOUT_FACTORS:
        clear_gaussian_parameter_gradients(gaussians)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        render_pkg = render(
            camera,
            gaussians,
            pipeline,
            background,
            dropout_factor=dropout_factor,
            sigma_noise=0.0,
            train=True,
        )
        evaluator = (
            evaluate_densification_protection_case
            if args.mode == "phase2b"
            else evaluate_render_case
        )
        case = evaluator(
            render_pkg,
            gaussians,
            camera,
            dropout_factor=dropout_factor,
            lambda_dssim=args.lambda_dssim,
            structural_weight=args.structural_weight,
        )
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())
        case["peak_cuda_memory_bytes"] = peak_bytes
        case["peak_cuda_memory_mib"] = peak_bytes / (1024.0 * 1024.0)
        case_peak_memory.append(peak_bytes)
        report["cases"].append(case)
        timing = case.get("timing_ms", {})
        timing_text = (
            f" baseline_grad_ms={timing['baseline_gradient_extraction']:.3f}"
            f" joint_backward_ms={timing['joint_backward']:.3f}"
            if timing
            else ""
        )
        print(
            "[I1Smoke] "
            f"mode={args.mode} dropout={dropout_factor} passed={case['passed']} "
            f"gaussians={case['dropout_index_audit']['gaussian_count']} "
            f"kept={case['dropout_index_audit']['dropout_kept_count']} "
            f"visible={case['dropout_index_audit']['visible_subset_count']} "
            f"peak_mib={case['peak_cuda_memory_mib']:.3f}"
            f"{timing_text}"
        )
        if not case["passed"]:
            failed = [name for name, passed in case["checks"].items() if not passed]
            raise SmokeCheckError(
                f"CUDA smoke checks failed for dropout={dropout_factor}: {failed}"
            )
        del render_pkg

    torch.cuda.synchronize()
    report["peak_cuda_memory_bytes"] = max(case_peak_memory, default=0)
    report["peak_cuda_memory_mib"] = report["peak_cuda_memory_bytes"] / (1024.0 * 1024.0)
    report["gaussian_count_after"] = int(gaussians.get_xyz.shape[0])
    report["optimizer_created"] = gaussians.optimizer is not None
    report["frozen_point_cloud_sha256_before"] = frozen_before
    report["frozen_point_cloud_sha256_after"] = sha256_file(point_cloud)
    report["frozen_point_cloud_unchanged"] = (
        report["frozen_point_cloud_sha256_before"] == report["frozen_point_cloud_sha256_after"]
    )
    report["gaussian_count_unchanged"] = report["gaussian_count_before"] == report["gaussian_count_after"]
    report["smoke_mode"] = args.mode
    if report["optimizer_created"]:
        raise SmokeCheckError("An optimizer appeared during the read-only smoke.")
    if not report["frozen_point_cloud_unchanged"] or not report["gaussian_count_unchanged"]:
        raise SmokeCheckError("Frozen Gaussian state changed during smoke.")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _is_relative_to(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def _write_json_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def build_parser():
    from arguments import ModelParams, PipelineParams

    parser = argparse.ArgumentParser(description="Read-only I1 real CUDA smoke")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--structural_weight", type=float, default=0.1)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("phase2a", "phase2b"), default="phase2a")
    return parser, model, pipeline


def main():
    from arguments import get_combined_args

    parser, model, pipeline = build_parser()
    args = get_combined_args(parser)
    if not args.model_path:
        parser.error("--model_path is required")
    output_dir = prepare_output_directory(args.output_dir, args.model_path)
    report_name = REPORT_NAMES[args.mode]
    log_name = LOG_NAMES[args.mode]
    report = {
        "schema_version": f"i1-{args.mode}-cuda-smoke-v1",
        "status": "RUNNING",
        "mode": args.mode,
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(output_dir),
        "log_file": log_name,
        "constraints": {
            "optimizer_step": False,
            "densify_or_prune": False,
            "midas": False,
            "synthetic_reference": True,
        },
    }
    exit_code = 1
    with (output_dir / log_name).open("x", encoding="utf-8", newline="\n") as log_handle:
        with redirect_stdout(_Tee(sys.stdout, log_handle)), redirect_stderr(
            _Tee(sys.stderr, log_handle)
        ):
            try:
                dataset = model.extract(args)
                pipe = pipeline.extract(args)
                run_cuda_smoke(args, dataset, pipe, report)
                report["status"] = "PASS"
                exit_code = 0
            except BaseException as exc:
                full_exception = traceback.format_exc()
                report["status"] = "FAIL"
                report["exception_type"] = type(exc).__name__
                report["exception"] = str(exc)
                report["traceback_file"] = EXCEPTION_NAME
                with (output_dir / EXCEPTION_NAME).open(
                    "x", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(full_exception)
                print(full_exception, file=sys.stderr)
            finally:
                _write_json_exclusive(output_dir / report_name, report)
            print(
                f"I1 {args.mode} CUDA smoke: {report['status']} -> "
                f"{output_dir / report_name}"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
