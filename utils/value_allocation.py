"""Value allocation utilities for Sparse-view 3DGS Phase 7.

This module only computes diagnostic value scores and discrete selections.  It
does not call GaussianModel, renderer, clone/split, pruning, densification, or
loss code.
"""

import torch


def compute_observation_scarcity(visible_view_count, tau_e, tau_s):
    """Compute evidence-gated observation scarcity O_i.

    Args:
        visible_view_count: Tensor[N] with the number of distinct training views
            that observed each Gaussian.
        tau_e: Positive evidence time constant.
        tau_s: Positive saturation time constant.

    Returns:
        Tensor[N] on the input device:
            O_i = (1 - exp(-n_i / tau_e)) * exp(-n_i / tau_s)

    Completely unobserved Gaussians have O_i = 0 by construction; no extra
    heuristic gives them high value.
    """
    with torch.no_grad():
        if tau_e <= 0 or tau_s <= 0:
            raise ValueError("tau_e and tau_s must be positive.")
        if not torch.is_tensor(visible_view_count):
            raise ValueError("visible_view_count must be a Tensor[N].")
        if visible_view_count.ndim != 1:
            raise ValueError("visible_view_count must be a Tensor[N].")
        if (visible_view_count < 0).any():
            raise ValueError("visible_view_count must be non-negative.")

        counts = visible_view_count.to(dtype=torch.float32)
        evidence = 1.0 - torch.exp(-counts / float(tau_e))
        saturation = torch.exp(-counts / float(tau_s))
        scarcity = evidence * saturation
        return torch.nan_to_num(scarcity, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def robust_normalize(values, low_quantile=0.05, high_quantile=0.95, eps=1e-8):
    """Quantile-normalize a 1D tensor to [0, 1].

    Statistics are computed only from finite entries using torch.quantile:
        clip((X - Q_low) / (Q_high - Q_low + eps), 0, 1)

    Non-finite input positions are assigned 0 in the output.  If all inputs are
    invalid, or the quantile range is degenerate, the function returns zeros.
    """
    with torch.no_grad():
        if not (0.0 <= low_quantile < high_quantile <= 1.0):
            raise ValueError("Require 0 <= low_quantile < high_quantile <= 1.")
        if not torch.is_tensor(values):
            raise ValueError("values must be a Tensor[N].")
        if values.ndim != 1:
            raise ValueError("values must be a Tensor[N].")

        output = torch.zeros_like(values, dtype=torch.float32)
        finite_mask = torch.isfinite(values)
        if not finite_mask.any():
            return output

        finite_values = values[finite_mask].to(dtype=torch.float32)
        q_low = torch.quantile(finite_values, low_quantile)
        q_high = torch.quantile(finite_values, high_quantile)
        denom = q_high - q_low
        if not torch.isfinite(denom) or denom.abs() <= eps:
            return output

        normalized = ((values.to(dtype=torch.float32) - q_low) / (denom + eps)).clamp(0.0, 1.0)
        output[finite_mask] = normalized[finite_mask]
        return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def compute_structural_value(boundary_support, geometric_turning, continuity_defect,
                             omega_boundary, omega_turning, omega_defect):
    """Fuse B/K/D structural statistics with explicit weights.

    Recommended Phase 7 usage keeps normalization separate from fusion:
        B_norm = robust_normalize(B)
        K_norm = robust_normalize(K)
        D_norm = robust_normalize(D)
        S = compute_structural_value(B_norm, K_norm, D_norm, ...)

    This function does not normalize, softmax, sigmoid, or learn any weights.
    Formula:
        S_i = omega_boundary * B_i + omega_turning * K_i + omega_defect * D_i
    """
    with torch.no_grad():
        _validate_same_shape(boundary_support, geometric_turning, continuity_defect)
        device = boundary_support.device
        dtype = boundary_support.dtype if boundary_support.is_floating_point() else torch.float32
        b = boundary_support.to(dtype=dtype)
        k = geometric_turning.to(device=device, dtype=dtype)
        d = continuity_defect.to(device=device, dtype=dtype)
        value = (
            _scalar_to_tensor(omega_boundary, dtype, device) * b
            + _scalar_to_tensor(omega_turning, dtype, device) * k
            + _scalar_to_tensor(omega_defect, dtype, device) * d
        )
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def compute_refine_utility(observation_scarcity, structural_value, redundancy,
                           lambda_redundancy, eps=1e-8):
    """Compute refinement utility with redundancy diminishing return.

    Recommended Phase 7 usage normalizes redundancy before calling this function:
        R_norm = robust_normalize(R)
        U = compute_refine_utility(O, S, R_norm, lambda_redundancy)

    The function does not require R to be normalized.  Formula:
        U_i = O_i * S_i / (1 + lambda_redundancy * R_i)
    """
    with torch.no_grad():
        if lambda_redundancy < 0:
            raise ValueError("lambda_redundancy must be non-negative.")
        _validate_same_shape(observation_scarcity, structural_value, redundancy)
        device = observation_scarcity.device
        dtype = observation_scarcity.dtype if observation_scarcity.is_floating_point() else torch.float32
        o = observation_scarcity.to(dtype=dtype)
        s = structural_value.to(device=device, dtype=dtype)
        r = redundancy.to(device=device, dtype=dtype)
        lam = _scalar_to_tensor(lambda_redundancy, dtype, device)
        denom = 1.0 + lam * r
        utility = o * s / denom.clamp_min(eps)
        return torch.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)


def compute_demand_weighted_value_score(gradient_score, utility, candidate_mask=None):
    """Compute demand-weighted allocation score A_i = g_i * U_i.

    This reuses the same gradient demand score used by gradient gating and does
    not normalize, soften, or otherwise transform either input.
    """
    with torch.no_grad():
        _validate_same_shape(gradient_score, utility)
        device = gradient_score.device
        gradient = gradient_score.to(dtype=torch.float32)
        value = utility.to(device=device, dtype=torch.float32)
        valid = torch.isfinite(gradient) & torch.isfinite(value)
        if candidate_mask is not None:
            if not torch.is_tensor(candidate_mask) or candidate_mask.shape != gradient_score.shape:
                raise ValueError("candidate_mask must be a BoolTensor[N] matching gradient_score.")
            valid = valid & candidate_mask.to(device=device, dtype=torch.bool)

        allocation_score = torch.zeros_like(gradient, dtype=torch.float32)
        allocation_score[valid] = gradient[valid] * value[valid]
        return torch.nan_to_num(allocation_score, nan=0.0, posinf=0.0, neginf=0.0)


def select_gradient_priority_value_rerank_topk(
    gradient_score,
    utility,
    candidate_mask,
    budget,
    rerank_fraction=0.25,
    return_mask=False,
):
    """Select gradient-priority Top-k with limited Value reranking at the boundary.

    Gradient ranking first protects the highest-gradient core.  Structural
    utility may only rerank candidates inside the boundary pool:
        ranks protected_count + 1 through K + rerank_slots.
    """
    with torch.no_grad():
        _validate_same_shape(gradient_score, utility)
        if candidate_mask is None or not torch.is_tensor(candidate_mask) or candidate_mask.shape != gradient_score.shape:
            raise ValueError("candidate_mask must be a BoolTensor[N] matching gradient_score.")
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        rerank_fraction = float(rerank_fraction)
        if not (0.0 <= rerank_fraction <= 0.5):
            raise ValueError("rerank_fraction must satisfy 0.0 <= rerank_fraction <= 0.5.")

        k = int(budget)
        device = gradient_score.device
        if k <= 0:
            selected_indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            gradient = gradient_score.to(dtype=torch.float32)
            value = utility.to(device=device, dtype=torch.float32)
            valid = (
                candidate_mask.to(device=device, dtype=torch.bool)
                & torch.isfinite(gradient)
                & torch.isfinite(value)
            )
            eligible_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            eligible_count = int(eligible_indices.numel())
            if eligible_count <= k:
                selected_indices = eligible_indices
            else:
                eligible_gradient = gradient[eligible_indices]
                gradient_order = torch.argsort(eligible_gradient, descending=True)
                ranked_indices = eligible_indices[gradient_order]

                rerank_slots = int(round(k * rerank_fraction))
                rerank_slots = max(0, min(rerank_slots, k))
                protected_count = k - rerank_slots

                protected_indices = ranked_indices[:protected_count]
                if rerank_slots == 0:
                    selected_indices = protected_indices
                else:
                    boundary_end = min(k + rerank_slots, eligible_count)
                    boundary_indices = ranked_indices[protected_count:boundary_end]
                    boundary_slots = min(rerank_slots, int(boundary_indices.numel()))
                    if boundary_slots > 0:
                        boundary_value = value[boundary_indices]
                        value_order = torch.argsort(boundary_value, descending=True)
                        reranked_indices = boundary_indices[value_order[:boundary_slots]]
                    else:
                        reranked_indices = torch.empty((0,), dtype=torch.long, device=device)
                    selected_indices = torch.cat((protected_indices, reranked_indices), dim=0)

                    if selected_indices.numel() < k:
                        selected_mask = torch.zeros_like(gradient, dtype=torch.bool)
                        selected_mask[selected_indices] = True
                        fill_indices = ranked_indices[~selected_mask[ranked_indices]]
                        selected_indices = torch.cat(
                            (selected_indices, fill_indices[: k - selected_indices.numel()]),
                            dim=0,
                        )

        if return_mask:
            selected_mask = torch.zeros_like(gradient_score, dtype=torch.bool)
            selected_mask[selected_indices] = True
            return selected_mask
        return selected_indices.to(dtype=torch.long)


def build_candidate_pool(
    recent_visible_mask=None,
    high_gradient_mask=None,
    low_observation_mask=None,
    high_boundary_mask=None,
):
    """Union explicit candidate masks without inventing thresholds.

    Each provided mask is converted to bool and ORed:
        recent_visible OR high_gradient OR low_observation OR high_boundary

    At least one BoolTensor[N]-like mask must be provided.  This function does
    not decide what "high" or "low" means and does not call GaussianModel.
    """
    with torch.no_grad():
        masks = [m for m in (recent_visible_mask, high_gradient_mask, low_observation_mask, high_boundary_mask) if m is not None]
        if not masks:
            raise ValueError("At least one candidate mask must be provided.")
        if not all(torch.is_tensor(mask) for mask in masks):
            raise ValueError("All candidate masks must be tensors.")

        first_shape = masks[0].shape
        if len(first_shape) != 1:
            raise ValueError("Candidate masks must have shape [N].")
        candidate = torch.zeros(first_shape, dtype=torch.bool, device=masks[0].device)
        for mask in masks:
            if mask.shape != first_shape:
                raise ValueError("All candidate masks must have the same shape.")
            candidate = candidate | mask.to(device=candidate.device, dtype=torch.bool)
        return candidate


def allocate_budget(utility, budget, candidate_mask=None, return_mask=False):
    """Select top-k positive finite utility indices from optional candidates.

    Args:
        utility: Tensor[N] utility scores. NaN/Inf and utility <= 0 entries are
            never selected.
        budget: Non-negative integer selection budget.
        candidate_mask: Optional BoolTensor[N]. Selection is restricted to True
            entries when provided.
        return_mask: If True, return BoolTensor[N] selected_mask instead of
            LongTensor[k] selected_indices.

    Returns:
        LongTensor[k] sorted by utility descending, or BoolTensor[N] when
        return_mask=True.  k = min(budget, number_of_valid_candidates).
    """
    with torch.no_grad():
        if not torch.is_tensor(utility) or utility.ndim != 1:
            raise ValueError("utility must be a Tensor[N].")
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        budget = int(budget)
        device = utility.device
        valid = torch.isfinite(utility) & (utility > 0)
        if candidate_mask is not None:
            if not torch.is_tensor(candidate_mask) or candidate_mask.shape != utility.shape:
                raise ValueError("candidate_mask must be a BoolTensor[N] matching utility.")
            valid = valid & candidate_mask.to(device=device, dtype=torch.bool)

        candidate_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        k = min(budget, candidate_indices.numel())
        if k == 0:
            selected_indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            candidate_scores = utility[candidate_indices].to(dtype=torch.float32)
            topk = torch.topk(candidate_scores, k=k, largest=True, sorted=True).indices
            selected_indices = candidate_indices[topk].to(dtype=torch.long)

        if return_mask:
            selected_mask = torch.zeros_like(utility, dtype=torch.bool)
            selected_mask[selected_indices] = True
            return selected_mask
        return selected_indices


def _validate_same_shape(*tensors):
    if not tensors or not all(torch.is_tensor(tensor) for tensor in tensors):
        raise ValueError("All inputs must be tensors.")
    shape = tensors[0].shape
    if len(shape) != 1:
        raise ValueError("Inputs must have shape [N].")
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError("All inputs must have the same shape.")


def _scalar_to_tensor(value, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device)
