"""Baseline densification-gradient protection for Innovation 1."""

from __future__ import annotations

import torch


def effective_densify_until_iter(
    iteration,
    densify_from_iter,
    densify_until_iter,
    total_iterations,
):
    """Predict the same-iteration cutoff used by the existing training branch."""
    if int(iteration) > int(densify_from_iter):
        return int(total_iterations)
    return int(densify_until_iter)


def should_preserve_baseline_densification_stats(
    preserve_flag,
    oe_loss_active,
    iteration,
    densify_until_iter,
):
    """Enable protection only while I1 and densification-stat collection are active."""
    return bool(preserve_flag) and bool(oe_loss_active) and int(iteration) < int(densify_until_iter)


def maybe_extract_baseline_viewspace_gradient(
    baseline_total_loss,
    viewspace_point_tensor,
    *,
    preserve_flag,
    oe_loss_active,
    iteration,
    densify_until_iter,
):
    """Return a baseline gradient only when every protection gate is open."""
    if not should_preserve_baseline_densification_stats(
        preserve_flag,
        oe_loss_active,
        iteration,
        densify_until_iter,
    ):
        return None
    return extract_baseline_viewspace_gradient(baseline_total_loss, viewspace_point_tensor)


def extract_baseline_viewspace_gradient(baseline_total_loss, viewspace_point_tensor):
    """Extract the baseline gradient from the current graph without changing ``.grad``."""
    previous_grad = viewspace_point_tensor.grad
    try:
        try:
            baseline_viewspace_grad = torch.autograd.grad(
                baseline_total_loss,
                viewspace_point_tensor,
                retain_graph=True,
                allow_unused=False,
            )[0]
        except RuntimeError as exc:
            raise RuntimeError(
                "baseline loss is not connected to the full viewspace_point_tensor"
            ) from exc
    finally:
        # Some custom renderer backward implementations may populate a retained
        # non-leaf as a side effect of autograd.grad().
        viewspace_point_tensor.grad = previous_grad

    if baseline_viewspace_grad is None:
        raise RuntimeError("baseline viewspace gradient extraction returned None")
    baseline_viewspace_grad = baseline_viewspace_grad.detach()
    _validate_viewspace_gradient(
        baseline_viewspace_grad,
        viewspace_point_tensor,
        role="baseline",
    )
    return baseline_viewspace_grad


def map_dropout_visibility_to_full_mask(
    dropout_mask,
    visibility_filter,
    viewspace_point_tensor,
):
    """Map renderer subset visibility back to the full ``[N, 3]`` Gaussian tensor."""
    if viewspace_point_tensor.ndim != 2 or viewspace_point_tensor.shape[1] != 3:
        raise ValueError("viewspace_point_tensor must be the full [N, 3] Gaussian tensor")
    gaussian_count = viewspace_point_tensor.shape[0]
    if dropout_mask.dtype != torch.bool or dropout_mask.ndim != 1:
        raise ValueError("dropout_mask must be a one-dimensional boolean tensor")
    if dropout_mask.shape[0] != gaussian_count:
        raise ValueError("dropout_mask must index the full viewspace_point_tensor")
    if dropout_mask.device != viewspace_point_tensor.device:
        raise ValueError("dropout_mask and viewspace_point_tensor must be on the same device")
    if visibility_filter.dtype != torch.bool or visibility_filter.ndim != 1:
        raise ValueError("visibility_filter must be a one-dimensional boolean tensor")
    if visibility_filter.device != viewspace_point_tensor.device:
        raise ValueError("visibility_filter and viewspace_point_tensor must be on the same device")

    retained_indices = torch.nonzero(dropout_mask, as_tuple=True)[0]
    if visibility_filter.shape[0] != retained_indices.numel():
        raise ValueError(
            "visibility_filter must index exactly the renderer's retained Dropout subset"
        )
    visible_full_indices = retained_indices[visibility_filter]
    combined_mask = torch.zeros_like(dropout_mask, dtype=torch.bool)
    combined_mask[visible_full_indices] = True
    return combined_mask


def add_densification_stats_with_baseline_gradient(
    gaussians,
    viewspace_point_tensor,
    combined_mask,
    baseline_viewspace_grad,
):
    """Temporarily expose the baseline gradient only to densification statistics."""
    _validate_full_index_mask(combined_mask, viewspace_point_tensor)
    _validate_viewspace_gradient(
        baseline_viewspace_grad,
        viewspace_point_tensor,
        role="baseline",
    )

    joint_grad = viewspace_point_tensor.grad
    _validate_viewspace_gradient(joint_grad, viewspace_point_tensor, role="baseline+I1")
    try:
        viewspace_point_tensor.grad = baseline_viewspace_grad
        gaussians.add_densification_stats(viewspace_point_tensor, combined_mask)
    finally:
        viewspace_point_tensor.grad = joint_grad


def _validate_full_index_mask(combined_mask, viewspace_point_tensor):
    if combined_mask.dtype != torch.bool or combined_mask.ndim != 1:
        raise ValueError("combined_mask must be a one-dimensional boolean tensor")
    if combined_mask.shape[0] != viewspace_point_tensor.shape[0]:
        raise ValueError("combined_mask must index the full viewspace_point_tensor")
    if combined_mask.device != viewspace_point_tensor.device:
        raise ValueError("combined_mask and viewspace_point_tensor must be on the same device")


def _validate_viewspace_gradient(gradient, viewspace_point_tensor, *, role):
    if gradient is None:
        raise ValueError(f"{role} viewspace gradient is required for densification stats")
    if gradient.shape != viewspace_point_tensor.shape:
        raise ValueError(
            f"{role} viewspace gradient shape must match viewspace_point_tensor shape"
        )
    if gradient.device != viewspace_point_tensor.device:
        raise ValueError(
            f"{role} viewspace gradient device must match viewspace_point_tensor device"
        )
    if not bool(torch.isfinite(gradient).all().item()):
        raise FloatingPointError(f"{role} viewspace gradient contains non-finite values")
