"""Budgeted densification selection helpers for Sparse-view 3DGS Phase 8.

These helpers are intentionally pure: they do not call GaussianModel,
renderer, pruning, clone/split generation, losses, or structural metrics.
"""

import torch

from utils.value_allocation import allocate_budget


def compute_gradient_score(xyz_gradient_accum, denom):
    """Compute the official screen-space gradient score used by 3DGS.

    Official densification uses ``xyz_gradient_accum / denom`` after
    ``add_densification_stats`` has accumulated screen-space xy gradient
    magnitudes.  Invalid, unobserved, NaN, or Inf entries are assigned 0 so
    they cannot enter Top-k selection.
    """
    with torch.no_grad():
        if not torch.is_tensor(xyz_gradient_accum) or not torch.is_tensor(denom):
            raise ValueError("xyz_gradient_accum and denom must be tensors.")
        if xyz_gradient_accum.shape != denom.shape:
            raise ValueError("xyz_gradient_accum and denom must have the same shape.")
        if xyz_gradient_accum.ndim == 2 and xyz_gradient_accum.shape[1] == 1:
            accum = xyz_gradient_accum.squeeze(1)
            denom_flat = denom.squeeze(1)
        elif xyz_gradient_accum.ndim == 1:
            accum = xyz_gradient_accum
            denom_flat = denom
        else:
            raise ValueError("xyz_gradient_accum and denom must have shape [N] or [N, 1].")

        score = torch.zeros_like(accum, dtype=torch.float32)
        valid = torch.isfinite(accum) & torch.isfinite(denom_flat) & (denom_flat != 0)
        if valid.any():
            score[valid] = accum[valid].to(dtype=torch.float32) / denom_flat[valid].to(dtype=torch.float32)
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def select_gradient_topk(gradient_score, budget, candidate_mask=None, return_mask=False):
    """Select Top-k positive finite gradient scores.

    Gradient and Value modes share ``allocate_budget`` so their budgeted Top-k
    behavior remains identical for fair comparison.
    """
    return allocate_budget(
        gradient_score,
        budget=budget,
        candidate_mask=candidate_mask,
        return_mask=return_mask,
    )


def build_official_clone_split_masks(selected_mask, scaling, percent_dense, scene_extent):
    """Split selected Gaussians into official clone/split action masks."""
    with torch.no_grad():
        if not torch.is_tensor(selected_mask) or selected_mask.ndim != 1:
            raise ValueError("selected_mask must be a BoolTensor[N].")
        if not torch.is_tensor(scaling) or scaling.ndim != 2:
            raise ValueError("scaling must be a Tensor[N, C].")
        if scaling.shape[0] != selected_mask.shape[0]:
            raise ValueError("selected_mask and scaling must have matching first dimensions.")

        selected = selected_mask.to(device=scaling.device, dtype=torch.bool)
        threshold = float(percent_dense) * float(scene_extent)
        max_scaling = torch.max(scaling, dim=1).values
        clone_mask = selected & (max_scaling <= threshold)
        split_mask = selected & (max_scaling > threshold)
        return clone_mask, split_mask
