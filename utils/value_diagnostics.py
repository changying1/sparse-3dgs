"""Pure diagnostics for Value Top-k densification selections."""

import torch


@torch.no_grad()
def compute_value_selection_diagnostics(
    scaling,
    selected_mask,
    clone_mask,
    split_mask,
    percent_dense,
    scene_extent,
    gradient_score,
    gradient_selected_mask,
    official_gradient_threshold,
    source_type=None,
    boundary=None,
    turning=None,
    defect=None,
    redundancy=None,
    utility=None,
):
    """Summarize Value-selected Gaussians without affecting densification.

    The function is intentionally observational: it reads tensors, sanitizes
    non-finite values for reporting, and returns Python scalars only.
    """
    if not torch.is_tensor(scaling) or scaling.ndim != 2:
        raise ValueError("scaling must be a Tensor[N, C].")
    n = scaling.shape[0]
    selected = _as_bool_mask(selected_mask, n, scaling.device, "selected_mask")
    clone = _as_bool_mask(clone_mask, n, scaling.device, "clone_mask")
    split = _as_bool_mask(split_mask, n, scaling.device, "split_mask")
    gradient = _as_1d_tensor(gradient_score, n, scaling.device, "gradient_score")
    gradient_selected = _as_bool_mask(gradient_selected_mask, n, scaling.device, "gradient_selected_mask")

    max_scaling = torch.max(scaling, dim=1).values
    threshold = float(percent_dense) * float(scene_extent)
    clone_eligible = max_scaling <= threshold
    split_eligible = max_scaling > threshold
    selected_count = int(selected.sum().item())
    all_count = int(n)

    selected_gradient = _finite_values(gradient[selected])
    positive_selected_gradient = torch.isfinite(gradient[selected]) & (gradient[selected] > 0)
    above_threshold_selected_gradient = (
        torch.isfinite(gradient[selected])
        & (gradient[selected] >= float(official_gradient_threshold))
    )
    overlap_count = int((selected & gradient_selected).sum().item())

    stats = {
        "clone_threshold": _finite_float(threshold),
        "all_scale_mean": _masked_mean(max_scaling, None),
        "all_scale_median": _masked_quantile(max_scaling, None, 0.5),
        "all_scale_q10": _masked_quantile(max_scaling, None, 0.1),
        "all_scale_q90": _masked_quantile(max_scaling, None, 0.9),
        "all_clone_eligible_count": int(clone_eligible.sum().item()),
        "all_split_eligible_count": int(split_eligible.sum().item()),
        "all_clone_eligible_ratio": _ratio(int(clone_eligible.sum().item()), all_count),
        "all_split_eligible_ratio": _ratio(int(split_eligible.sum().item()), all_count),
        "selected_scale_mean": _masked_mean(max_scaling, selected),
        "selected_scale_median": _masked_quantile(max_scaling, selected, 0.5),
        "selected_scale_q10": _masked_quantile(max_scaling, selected, 0.1),
        "selected_scale_q90": _masked_quantile(max_scaling, selected, 0.9),
        "selected_clone_eligible_count": int((selected & clone_eligible).sum().item()),
        "selected_split_eligible_count": int((selected & split_eligible).sum().item()),
        "selected_clone_eligible_ratio": _ratio(int((selected & clone_eligible).sum().item()), selected_count),
        "selected_split_eligible_ratio": _ratio(int((selected & split_eligible).sum().item()), selected_count),
        "selected_gradient_mean": _mean_finite(selected_gradient),
        "selected_gradient_median": _quantile_finite(selected_gradient, 0.5),
        "selected_gradient_positive_ratio": _ratio(int(positive_selected_gradient.sum().item()), selected_count),
        "selected_gradient_above_official_threshold_ratio": _ratio(
            int(above_threshold_selected_gradient.sum().item()),
            selected_count,
        ),
        "value_gradient_overlap_count": overlap_count,
        "value_gradient_overlap_ratio": _ratio(overlap_count, selected_count),
        "value_selected_count": selected_count,
        "gradient_selected_count": int(gradient_selected.sum().item()),
    }

    stats.update(_source_stats(source_type, selected, n, scaling.device))
    stats.update(_utility_stats(clone, split, selected, boundary, turning, defect, redundancy, utility, n, scaling.device))
    return {key: _finite_float(value) for key, value in stats.items()}


def format_value_diagnostics_log(iteration, stats):
    """Build the complete four-line Value diagnostic log message."""
    return (
        f"[ValueDiag ITER {iteration}] "
        f"threshold={stats['clone_threshold']:.6f} "
        f"all_scale_mean={stats['all_scale_mean']:.6f} "
        f"all_scale_median={stats['all_scale_median']:.6f} "
        f"all_scale_q10={stats['all_scale_q10']:.6f} "
        f"all_scale_q90={stats['all_scale_q90']:.6f} "
        f"selected_scale_mean={stats['selected_scale_mean']:.6f} "
        f"selected_scale_median={stats['selected_scale_median']:.6f} "
        f"selected_scale_q10={stats['selected_scale_q10']:.6f} "
        f"selected_scale_q90={stats['selected_scale_q90']:.6f} "
        f"all_clone_ratio={stats['all_clone_eligible_ratio']:.6f} "
        f"all_split_ratio={stats['all_split_eligible_ratio']:.6f} "
        f"selected_clone_ratio={stats['selected_clone_eligible_ratio']:.6f} "
        f"selected_split_ratio={stats['selected_split_eligible_ratio']:.6f}\n"
        f"[ValueDiag ITER {iteration}] "
        f"selected_gradient_mean={stats['selected_gradient_mean']:.6f} "
        f"selected_gradient_median={stats['selected_gradient_median']:.6f} "
        f"grad_positive_ratio={stats['selected_gradient_positive_ratio']:.6f} "
        f"grad_threshold_ratio={stats['selected_gradient_above_official_threshold_ratio']:.6f} "
        f"gradient_overlap={int(stats['value_gradient_overlap_count'])}/{int(stats['value_selected_count'])}\n"
        f"[ValueDiag ITER {iteration}] "
        f"source_initial={int(stats['selected_source_initial'])} "
        f"source_clone={int(stats['selected_source_clone'])} "
        f"source_split={int(stats['selected_source_split'])} "
        f"source_completion={int(stats['selected_source_completion'])} "
        f"clone_source_ratio={stats['selected_clone_source_ratio']:.6f}\n"
        f"[ValueDiag ITER {iteration}] "
        f"selected_mean_B={stats['selected_mean_B']:.6f} "
        f"selected_mean_K={stats['selected_mean_K']:.6f} "
        f"selected_mean_D={stats['selected_mean_D']:.6f} "
        f"selected_mean_R={stats['selected_mean_R']:.6f} "
        f"selected_mean_U={stats['selected_mean_U']:.6f} "
        f"clone_mean_U={stats['clone_selected_mean_U']:.6f} "
        f"split_mean_U={stats['split_selected_mean_U']:.6f}"
    )


def _as_bool_mask(mask, n, device, name):
    if not torch.is_tensor(mask) or mask.shape != (n,):
        raise ValueError(f"{name} must be a BoolTensor[N].")
    return mask.to(device=device, dtype=torch.bool)


def _as_1d_tensor(values, n, device, name):
    if not torch.is_tensor(values) or values.shape != (n,):
        raise ValueError(f"{name} must be a Tensor[N].")
    return values.to(device=device)


def _optional_1d(values, n, device, name):
    if values is None:
        return None
    return _as_1d_tensor(values, n, device, name)


def _finite_values(values):
    values = values.to(dtype=torch.float32)
    return values[torch.isfinite(values)]


def _masked_values(values, mask):
    if mask is not None:
        values = values[mask]
    return _finite_values(values)


def _mean_finite(values):
    finite = _finite_values(values)
    if finite.numel() == 0:
        return 0.0
    return finite.mean().item()


def _quantile_finite(values, q):
    finite = _finite_values(values)
    if finite.numel() == 0:
        return 0.0
    return torch.quantile(finite, q).item()


def _masked_mean(values, mask):
    return _mean_finite(_masked_values(values, mask))


def _masked_quantile(values, mask, q):
    return _quantile_finite(_masked_values(values, mask), q)


def _ratio(count, total):
    return float(count) / float(max(int(total), 1))


def _source_stats(source_type, selected, n, device):
    selected_count = int(selected.sum().item())
    if source_type is None:
        source = torch.zeros((n,), dtype=torch.long, device=device)
    else:
        source = _as_1d_tensor(source_type, n, device, "source_type").to(dtype=torch.long)
    selected_source = source[selected]
    source_initial = int((selected_source == 0).sum().item())
    source_clone = int((selected_source == 1).sum().item())
    source_split = int((selected_source == 2).sum().item())
    source_completion = int((selected_source == 3).sum().item())
    return {
        "selected_source_initial": source_initial,
        "selected_source_clone": source_clone,
        "selected_source_split": source_split,
        "selected_source_completion": source_completion,
        "selected_clone_source_ratio": _ratio(source_clone, selected_count),
    }


def _utility_stats(clone, split, selected, boundary, turning, defect, redundancy, utility, n, device):
    b = _optional_1d(boundary, n, device, "boundary")
    k = _optional_1d(turning, n, device, "turning")
    d = _optional_1d(defect, n, device, "defect")
    r = _optional_1d(redundancy, n, device, "redundancy")
    u = _optional_1d(utility, n, device, "utility")
    return {
        "clone_selected_mean_U": _masked_mean(u, clone) if u is not None else 0.0,
        "split_selected_mean_U": _masked_mean(u, split) if u is not None else 0.0,
        "selected_mean_B": _masked_mean(b, selected) if b is not None else 0.0,
        "selected_mean_K": _masked_mean(k, selected) if k is not None else 0.0,
        "selected_mean_D": _masked_mean(d, selected) if d is not None else 0.0,
        "selected_mean_R": _masked_mean(r, selected) if r is not None else 0.0,
        "selected_mean_U": _masked_mean(u, selected) if u is not None else 0.0,
    }


def _finite_float(value):
    if isinstance(value, int):
        return value
    value = float(value)
    if value != value or value == float("inf") or value == -float("inf"):
        return 0.0
    return value
