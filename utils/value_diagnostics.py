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
    allocation_score=None,
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
    stats.update(_utility_stats(clone, split, selected, boundary, turning, defect, redundancy, utility, allocation_score, n, scaling.device))
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
        + (
            f" selected_mean_A={stats['selected_mean_A']:.6f} "
            f"clone_mean_A={stats['clone_selected_mean_A']:.6f} "
            f"split_mean_A={stats['split_selected_mean_A']:.6f}"
            if "selected_mean_A" in stats
            else ""
        )
    )


@torch.no_grad()
def build_multiview_value_diagnostics(
    scaling,
    selected_mask,
    effective_budget,
    gradient_score,
    official_gradient_threshold,
    visible_view_count,
    observation,
    boundary=None,
    turning=None,
    defect=None,
    redundancy=None,
    utility=None,
    allocation_score=None,
    structural_value=None,
    lambda_redundancy=0.0,
    source_type=None,
    value_reference_mask=None,
):
    """Compare eligible, Value-selected, and matched Gradient Top-k pools.

    This is a read-only diagnostic. It constructs comparison masks and returns
    Python summaries; no selection tensor used by training is modified.
    """
    if not torch.is_tensor(scaling) or scaling.ndim != 2:
        raise ValueError("scaling must be a Tensor[N, C].")
    n = scaling.shape[0]
    device = scaling.device
    selected = _as_bool_mask(selected_mask, n, device, "selected_mask")
    gradient = _as_1d_tensor(gradient_score, n, device, "gradient_score")
    visible = _as_1d_tensor(visible_view_count, n, device, "visible_view_count")
    obs = _as_1d_tensor(observation, n, device, "observation")
    b = _optional_1d(boundary, n, device, "boundary")
    k_val = _optional_1d(turning, n, device, "turning")
    d = _optional_1d(defect, n, device, "defect")
    r = _optional_1d(redundancy, n, device, "redundancy")
    u = _optional_1d(utility, n, device, "utility")
    a = _optional_1d(allocation_score, n, device, "allocation_score")
    s = _optional_1d(structural_value, n, device, "structural_value")
    value_reference = None
    if value_reference_mask is not None:
        value_reference = _as_bool_mask(value_reference_mask, n, device, "value_reference_mask")

    threshold = float(official_gradient_threshold)
    eligible = torch.isfinite(gradient) & (gradient >= threshold)
    eligible_count = int(eligible.sum().item())
    matched_k = min(max(int(effective_budget), 0), eligible_count)
    gradient_topk = _topk_mask(gradient, matched_k, eligible)

    value_selected = selected
    max_scaling = torch.max(scaling, dim=1).values
    component_tensors = {
        "visible_view_count": visible,
        "O": obs,
        "gradient": gradient,
        "B": b,
        "K": k_val,
        "D": d,
        "R": r,
        "S": s,
        "U": u,
        "scale": max_scaling,
    }
    if a is not None:
        component_tensors["A"] = a
    groups = {
        "eligible": eligible,
        "value_selected": value_selected,
        "gradient_topk": gradient_topk,
    }

    gradient_rank = _rank_percentiles(gradient, eligible)
    observation_rank = _rank_percentiles(obs, eligible)
    no_o_utility = _compute_no_o_utility(s, r, lambda_redundancy, u)
    no_o_topk = _topk_mask(no_o_utility, matched_k, eligible)

    result = {
        "eligible_count": eligible_count,
        "eligible_ratio": _ratio(eligible_count, n),
        "selected_count": int(value_selected.sum().item()),
        "gradient_topk_count": int(gradient_topk.sum().item()),
        "matched_budget": matched_k,
        "groups": {
            name: _multiview_group_stats(
                mask,
                component_tensors,
                gradient,
                threshold,
                source_type,
                n,
                device,
            )
            for name, mask in groups.items()
        },
        "visibility_bins": _visibility_bin_stats(
            visible,
            eligible,
            value_selected,
            gradient_topk,
            obs,
            gradient,
            u,
        ),
        "rank": {
            "value_vs_gradient_overlap_count": _overlap_count(value_selected, gradient_topk),
            "value_vs_gradient_overlap_ratio": _overlap_ratio(value_selected, gradient_topk, matched_k),
            "value_selected_gradient_rank_percentile_mean": _masked_mean(gradient_rank, value_selected),
            "value_selected_gradient_rank_percentile_median": _masked_quantile(gradient_rank, value_selected, 0.5),
            "selected_O_rank_percentile_mean": _masked_mean(observation_rank, value_selected),
            "selected_O_rank_percentile_median": _masked_quantile(observation_rank, value_selected, 0.5),
        },
        "counterfactual_no_o": {
            "value_vs_noO_overlap_count": _overlap_count(value_selected, no_o_topk),
            "value_vs_noO_overlap_ratio": _overlap_ratio(value_selected, no_o_topk, matched_k),
            "noO_vs_gradient_overlap_count": _overlap_count(no_o_topk, gradient_topk),
            "noO_vs_gradient_overlap_ratio": _overlap_ratio(no_o_topk, gradient_topk, matched_k),
        },
    }
    if a is not None:
        result["allocation_score"] = {
            "formula": "A = gradient * U",
            "selected_A_mean": _masked_mean(a, value_selected),
            "selected_A_median": _masked_quantile(a, value_selected, 0.5),
        }
        result["rank"]["demand_value_vs_gradient_overlap_count"] = _overlap_count(value_selected, gradient_topk)
        result["rank"]["demand_value_vs_gradient_overlap_ratio"] = _overlap_ratio(value_selected, gradient_topk, matched_k)
        if value_reference is not None:
            result["rank"]["demand_value_vs_value_overlap_count"] = _overlap_count(value_selected, value_reference)
            result["rank"]["demand_value_vs_value_overlap_ratio"] = _overlap_ratio(value_selected, value_reference, matched_k)
    return _sanitize_nested(result)


def format_multiview_value_diagnostics_log(iteration, stats):
    """Build compact ValueDiagMV log lines for Phase 12G."""
    selected = stats["groups"]["value_selected"]
    eligible = stats["groups"]["eligible"]
    gradient_topk = stats["groups"]["gradient_topk"]
    rank = stats["rank"]
    no_o = stats["counterfactual_no_o"]
    summary_line = (
        f"[ValueDiagMV ITER {iteration}] "
        f"eligible_count={stats['eligible_count']} "
        f"eligible_ratio={stats['eligible_ratio']:.6f} "
        f"selected_count={stats['selected_count']} "
        f"gradient_topk_count={stats['gradient_topk_count']} "
        f"matched_budget={stats['matched_budget']} "
        f"value_gradient_overlap={rank['value_vs_gradient_overlap_count']}/{stats['matched_budget']} "
        f"value_gradient_overlap_ratio={rank['value_vs_gradient_overlap_ratio']:.6f} "
        f"value_noO_overlap={no_o['value_vs_noO_overlap_count']}/{stats['matched_budget']} "
        f"value_noO_overlap_ratio={no_o['value_vs_noO_overlap_ratio']:.6f} "
        f"noO_gradient_overlap={no_o['noO_vs_gradient_overlap_count']}/{stats['matched_budget']} "
        f"noO_gradient_overlap_ratio={no_o['noO_vs_gradient_overlap_ratio']:.6f}"
    )
    if "demand_value_vs_gradient_overlap_count" in rank:
        summary_line += (
            f" demand_value_gradient_overlap={rank['demand_value_vs_gradient_overlap_count']}/{stats['matched_budget']} "
            f"demand_value_gradient_overlap_ratio={rank['demand_value_vs_gradient_overlap_ratio']:.6f}"
        )
    if "demand_value_vs_value_overlap_count" in rank:
        summary_line += (
            f" demand_value_value_overlap={rank['demand_value_vs_value_overlap_count']}/{stats['matched_budget']} "
            f"demand_value_value_overlap_ratio={rank['demand_value_vs_value_overlap_ratio']:.6f}"
        )

    lines = [
        summary_line,
        (
            f"[ValueDiagMV ITER {iteration}] "
            f"eligible_grad_median={eligible['gradient_median']:.6f} "
            f"selected_grad_median={selected['gradient_median']:.6f} "
            f"gradient_topk_grad_median={gradient_topk['gradient_median']:.6f} "
            f"selected_grad_rank_mean={rank['value_selected_gradient_rank_percentile_mean']:.6f} "
            f"selected_grad_rank_median={rank['value_selected_gradient_rank_percentile_median']:.6f} "
            f"selected_O_rank_mean={rank['selected_O_rank_percentile_mean']:.6f} "
            f"selected_O_rank_median={rank['selected_O_rank_percentile_median']:.6f}"
        ),
        (
            f"[ValueDiagMV ITER {iteration}] "
            f"selected_grad_margin_median={selected['grad_margin_median']:.6f} "
            f"selected_near_1.25x={selected['near_threshold_1_25x_ratio']:.6f} "
            f"selected_near_1.5x={selected['near_threshold_1_5x_ratio']:.6f} "
            f"selected_near_2x={selected['near_threshold_2x_ratio']:.6f}"
        ),
    ]
    for name in ("eligible", "value_selected", "gradient_topk"):
        lines.extend(_format_multiview_group_lines(iteration, name, stats["groups"][name]))
    if "allocation_score" in stats:
        allocation = stats["allocation_score"]
        lines.append(
            f"[ValueDiagMV ITER {iteration}] "
            f"A_formula='{allocation['formula']}' "
            f"selected_A_mean={allocation['selected_A_mean']:.6f} "
            f"selected_A_median={allocation['selected_A_median']:.6f}"
        )
    for bin_stats in stats["visibility_bins"]:
        lines.append(
            f"[ValueDiagMV ITER {iteration}][n={bin_stats['n']}] "
            f"eligible={bin_stats['eligible_count']} "
            f"value_selected={bin_stats['value_selected_count']} "
            f"gradient_topk={bin_stats['gradient_topk_count']} "
            f"value_rate={bin_stats['value_selection_rate']:.6f} "
            f"gradient_rate={bin_stats['gradient_selection_rate']:.6f} "
            f"mean_O={bin_stats['mean_O']:.6f} "
            f"eligible_grad_mean={bin_stats['eligible_mean_gradient']:.6f} "
            f"selected_grad_mean={bin_stats['value_selected_mean_gradient']:.6f} "
            f"gradient_topk_grad_mean={bin_stats['gradient_topk_mean_gradient']:.6f} "
            f"eligible_U_mean={bin_stats['eligible_mean_U']:.6f} "
            f"selected_U_mean={bin_stats['value_selected_mean_U']:.6f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def compute_value_rerank_diagnostics(
    gradient_score,
    utility,
    candidate_mask,
    selected_mask,
    budget,
    rerank_fraction,
):
    """Summarize the gradient-priority Value-rerank selection boundary."""
    if not torch.is_tensor(gradient_score) or gradient_score.ndim != 1:
        raise ValueError("gradient_score must be a Tensor[N].")
    n = gradient_score.shape[0]
    device = gradient_score.device
    gradient = gradient_score.to(dtype=torch.float32)
    value = _as_1d_tensor(utility, n, device, "utility").to(dtype=torch.float32)
    candidate = _as_bool_mask(candidate_mask, n, device, "candidate_mask")
    selected = _as_bool_mask(selected_mask, n, device, "selected_mask")

    effective_budget = max(int(budget), 0)
    rerank_slots = int(round(effective_budget * float(rerank_fraction)))
    rerank_slots = max(0, min(rerank_slots, effective_budget))
    protected_count = effective_budget - rerank_slots
    eligible = candidate & torch.isfinite(gradient) & torch.isfinite(value)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    eligible_count = int(eligible_indices.numel())
    matched_k = min(effective_budget, eligible_count)

    if eligible_count == 0 or effective_budget <= 0:
        gradient_topk = torch.zeros_like(selected, dtype=torch.bool)
        boundary_mask = torch.zeros_like(selected, dtype=torch.bool)
        protected_mask = torch.zeros_like(selected, dtype=torch.bool)
        selected_ranks = torch.empty((0,), dtype=torch.long, device=device)
    else:
        gradient_order = torch.argsort(gradient[eligible_indices], descending=True)
        ranked_indices = eligible_indices[gradient_order]
        gradient_topk_indices = ranked_indices[:matched_k]
        gradient_topk = torch.zeros_like(selected, dtype=torch.bool)
        gradient_topk[gradient_topk_indices] = True

        protected_end = min(protected_count, eligible_count)
        boundary_end = min(effective_budget + rerank_slots, eligible_count)
        protected_mask = torch.zeros_like(selected, dtype=torch.bool)
        boundary_mask = torch.zeros_like(selected, dtype=torch.bool)
        protected_mask[ranked_indices[:protected_end]] = True
        boundary_mask[ranked_indices[protected_end:boundary_end]] = True

        rank_lookup = torch.zeros((n,), dtype=torch.long, device=device)
        rank_lookup[ranked_indices] = torch.arange(1, eligible_count + 1, dtype=torch.long, device=device)
        selected_ranks = rank_lookup[selected & eligible]

    overlap = int((selected & gradient_topk).sum().item())
    final_selected_count = int(selected.sum().item())
    selected_below_original_topk = int(((selected & eligible) & ~gradient_topk).sum().item())
    selected_max_rank = int(selected_ranks.max().item()) if selected_ranks.numel() else 0
    stats = {
        "budget": effective_budget,
        "eligible_count": eligible_count,
        "protected_count": protected_count,
        "rerank_slots": rerank_slots,
        "boundary_pool_count": int(boundary_mask.sum().item()),
        "final_selected_count": final_selected_count,
        "gradient_topk_overlap": overlap,
        "gradient_topk_overlap_ratio": _ratio(overlap, matched_k),
        "selected_grad_mean": _masked_mean(gradient, selected),
        "selected_grad_median": _masked_quantile(gradient, selected, 0.5),
        "gradient_topk_grad_mean": _masked_mean(gradient, gradient_topk),
        "gradient_topk_grad_median": _masked_quantile(gradient, gradient_topk, 0.5),
        "selected_U_mean": _masked_mean(value, selected),
        "selected_U_median": _masked_quantile(value, selected, 0.5),
        "boundary_U_mean": _masked_mean(value, boundary_mask),
        "boundary_U_median": _masked_quantile(value, boundary_mask, 0.5),
        "selected_max_gradient_rank": selected_max_rank,
        "selected_below_original_topk_count": selected_below_original_topk,
        "protected_selected_count": int((selected & protected_mask).sum().item()),
    }
    return {key: _finite_float(value) for key, value in stats.items()}


def format_value_rerank_diagnostics_log(iteration, stats):
    """Build the compact ValueRerank diagnostic log message."""
    return (
        f"[ValueRerank ITER {iteration}] "
        f"budget={int(stats['budget'])} "
        f"eligible_count={int(stats['eligible_count'])} "
        f"protected_count={int(stats['protected_count'])} "
        f"protected_selected_count={int(stats['protected_selected_count'])} "
        f"rerank_slots={int(stats['rerank_slots'])} "
        f"boundary_pool_count={int(stats['boundary_pool_count'])} "
        f"final_selected_count={int(stats['final_selected_count'])} "
        f"gradient_topk_overlap={int(stats['gradient_topk_overlap'])} "
        f"gradient_topk_overlap_ratio={stats['gradient_topk_overlap_ratio']:.6f}\n"
        f"[ValueRerank ITER {iteration}] "
        f"selected_grad_mean={stats['selected_grad_mean']:.6f} "
        f"selected_grad_median={stats['selected_grad_median']:.6f} "
        f"gradient_topk_grad_mean={stats['gradient_topk_grad_mean']:.6f} "
        f"gradient_topk_grad_median={stats['gradient_topk_grad_median']:.6f} "
        f"selected_U_mean={stats['selected_U_mean']:.6f} "
        f"selected_U_median={stats['selected_U_median']:.6f} "
        f"boundary_U_mean={stats['boundary_U_mean']:.6f} "
        f"boundary_U_median={stats['boundary_U_median']:.6f} "
        f"selected_max_gradient_rank={int(stats['selected_max_gradient_rank'])} "
        f"selected_below_original_topk_count={int(stats['selected_below_original_topk_count'])}"
    )


def _format_multiview_group_lines(iteration, name, stats):
    prefix = f"[ValueDiagMV ITER {iteration}][group={name}]"
    return [
        (
            f"{prefix} "
            f"count={stats['count']} "
            f"visible_mean={stats['visible_view_count_mean']:.6f} "
            f"visible_median={stats['visible_view_count_median']:.6f} "
            f"visible_q10={stats['visible_view_count_q10']:.6f} "
            f"visible_q90={stats['visible_view_count_q90']:.6f} "
            f"visible_min={stats['visible_view_count_min']:.6f} "
            f"visible_max={stats['visible_view_count_max']:.6f}"
        ),
        (
            f"{prefix} "
            f"O_mean={stats['O_mean']:.6f} "
            f"O_median={stats['O_median']:.6f} "
            f"O_q10={stats['O_q10']:.6f} "
            f"O_q90={stats['O_q90']:.6f} "
            f"gradient_mean={stats['gradient_mean']:.6f} "
            f"gradient_median={stats['gradient_median']:.6f} "
            f"gradient_q10={stats['gradient_q10']:.6f} "
            f"gradient_q90={stats['gradient_q90']:.6f} "
            f"gradient_max={stats['gradient_max']:.6f}"
        ),
        (
            f"{prefix} "
            f"scale_mean={stats['scale_mean']:.6f} "
            f"scale_median={stats['scale_median']:.6f} "
            f"scale_q10={stats['scale_q10']:.6f} "
            f"scale_q90={stats['scale_q90']:.6f} "
            f"B_norm_mean={stats['B_mean']:.6f} "
            f"B_norm_median={stats['B_median']:.6f} "
            f"K_norm_mean={stats['K_mean']:.6f} "
            f"K_norm_median={stats['K_median']:.6f} "
            f"D_norm_mean={stats['D_mean']:.6f} "
            f"D_norm_median={stats['D_median']:.6f}"
        ),
        (
            f"{prefix} "
            f"R_norm_mean={stats['R_mean']:.6f} "
            f"R_norm_median={stats['R_median']:.6f} "
            f"S_mean={stats['S_mean']:.6f} "
            f"S_median={stats['S_median']:.6f} "
            f"U_mean={stats['U_mean']:.6f} "
            f"U_median={stats['U_median']:.6f} "
            + (
                f"A_mean={stats['A_mean']:.6f} "
                f"A_median={stats['A_median']:.6f} "
                if "A_mean" in stats
                else ""
            )
            + (
            f"source_initial_ratio={stats['initial_ratio']:.6f} "
            f"source_clone_ratio={stats['clone_ratio']:.6f} "
            f"source_split_ratio={stats['split_ratio']:.6f} "
            f"source_completion_ratio={stats['completion_ratio']:.6f}"
            )
        ),
        (
            f"{prefix} "
            f"grad_margin_mean={stats['grad_margin_mean']:.6f} "
            f"grad_margin_median={stats['grad_margin_median']:.6f} "
            f"grad_margin_q10={stats['grad_margin_q10']:.6f} "
            f"grad_margin_q90={stats['grad_margin_q90']:.6f} "
            f"near_1.25x={stats['near_threshold_1_25x_ratio']:.6f} "
            f"near_1.5x={stats['near_threshold_1_5x_ratio']:.6f} "
            f"near_2x={stats['near_threshold_2x_ratio']:.6f}"
        ),
    ]


def _as_bool_mask(mask, n, device, name):
    if not torch.is_tensor(mask) or mask.shape != (n,):
        raise ValueError(f"{name} must be a BoolTensor[N].")
    return mask.to(device=device, dtype=torch.bool)


def _as_1d_tensor(values, n, device, name):
    if not torch.is_tensor(values) or values.shape != (n,):
        raise ValueError(f"{name} must be a Tensor[N].")
    return values.to(device=device)


def _topk_mask(scores, budget, candidate_mask):
    if not torch.is_tensor(candidate_mask) or candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must be a BoolTensor[N] matching scores.")
    selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
    valid = candidate_mask.to(device=scores.device, dtype=torch.bool) & torch.isfinite(scores)
    candidate_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    k = min(max(int(budget), 0), int(candidate_indices.numel()))
    if k == 0:
        return selected
    candidate_scores = scores[candidate_indices].to(dtype=torch.float32)
    topk = torch.topk(candidate_scores, k=k, largest=True, sorted=True).indices
    selected[candidate_indices[topk]] = True
    return selected


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


def _source_ratio_stats(source_type, mask, n, device):
    count = int(mask.sum().item())
    if source_type is None:
        source = torch.zeros((n,), dtype=torch.long, device=device)
    else:
        source = _as_1d_tensor(source_type, n, device, "source_type").to(dtype=torch.long)
    values = source[mask]
    initial = int((values == 0).sum().item())
    clone = int((values == 1).sum().item())
    split = int((values == 2).sum().item())
    completion = int((values == 3).sum().item())
    return {
        "initial_ratio": _ratio(initial, count),
        "clone_ratio": _ratio(clone, count),
        "split_ratio": _ratio(split, count),
        "completion_ratio": _ratio(completion, count),
    }


def _multiview_group_stats(mask, components, gradient, threshold, source_type, n, device):
    stats = {"count": int(mask.sum().item())}
    for name, values in components.items():
        if values is None:
            stats[f"{name}_mean"] = 0.0
            stats[f"{name}_median"] = 0.0
            if name in ("visible_view_count", "O", "gradient", "scale"):
                stats[f"{name}_q10"] = 0.0
                stats[f"{name}_q90"] = 0.0
            if name == "visible_view_count":
                stats[f"{name}_min"] = 0.0
                stats[f"{name}_max"] = 0.0
            if name == "gradient":
                stats[f"{name}_max"] = 0.0
            continue
        stats[f"{name}_mean"] = _masked_mean(values, mask)
        stats[f"{name}_median"] = _masked_quantile(values, mask, 0.5)
        if name in ("visible_view_count", "O", "gradient", "scale"):
            stats[f"{name}_q10"] = _masked_quantile(values, mask, 0.1)
            stats[f"{name}_q90"] = _masked_quantile(values, mask, 0.9)
        if name == "gradient":
            stats[f"{name}_max"] = _masked_max(values, mask)
    visible = components.get("visible_view_count")
    if visible is not None:
        stats["visible_view_count_min"] = _masked_min(visible, mask)
        stats["visible_view_count_max"] = _masked_max(visible, mask)
    grad_values = gradient[mask]
    margin = grad_values.to(dtype=torch.float32) / max(float(threshold), 1e-12)
    finite_grad = torch.isfinite(grad_values)
    stats.update(
        {
            "grad_margin_mean": _mean_finite(margin),
            "grad_margin_median": _quantile_finite(margin, 0.5),
            "grad_margin_q10": _quantile_finite(margin, 0.1),
            "grad_margin_q90": _quantile_finite(margin, 0.9),
            "near_threshold_1_25x_ratio": _ratio(
                int((finite_grad & (grad_values >= threshold) & (grad_values < 1.25 * threshold)).sum().item()),
                int(finite_grad.sum().item()),
            ),
            "near_threshold_1_5x_ratio": _ratio(
                int((finite_grad & (grad_values >= threshold) & (grad_values < 1.5 * threshold)).sum().item()),
                int(finite_grad.sum().item()),
            ),
            "near_threshold_2x_ratio": _ratio(
                int((finite_grad & (grad_values >= threshold) & (grad_values < 2.0 * threshold)).sum().item()),
                int(finite_grad.sum().item()),
            ),
        }
    )
    stats.update(_source_ratio_stats(source_type, mask, n, device))
    return stats


def _visibility_bin_stats(visible, eligible, value_selected, gradient_topk, observation, gradient, utility):
    bins = []
    finite_visible = torch.isfinite(visible.to(dtype=torch.float32))
    if not finite_visible.any():
        return bins
    unique_counts = torch.unique(visible[finite_visible].to(dtype=torch.long), sorted=True)
    for count_value in unique_counts:
        bin_mask = finite_visible & (visible.to(dtype=torch.long) == count_value)
        eligible_bin = bin_mask & eligible
        value_bin = bin_mask & value_selected
        gradient_bin = bin_mask & gradient_topk
        eligible_count = int(eligible_bin.sum().item())
        bins.append(
            {
                "n": int(count_value.item()),
                "eligible_count": eligible_count,
                "value_selected_count": int(value_bin.sum().item()),
                "gradient_topk_count": int(gradient_bin.sum().item()),
                "value_selection_rate": _ratio(int(value_bin.sum().item()), eligible_count),
                "gradient_selection_rate": _ratio(int(gradient_bin.sum().item()), eligible_count),
                "mean_O": _masked_mean(observation, eligible_bin),
                "eligible_mean_gradient": _masked_mean(gradient, eligible_bin),
                "value_selected_mean_gradient": _masked_mean(gradient, value_bin),
                "gradient_topk_mean_gradient": _masked_mean(gradient, gradient_bin),
                "eligible_mean_U": _masked_mean(utility, eligible_bin) if utility is not None else 0.0,
                "value_selected_mean_U": _masked_mean(utility, value_bin) if utility is not None else 0.0,
            }
        )
    return bins


def _rank_percentiles(values, mask):
    ranks = torch.zeros_like(values, dtype=torch.float32)
    valid = mask.to(device=values.device, dtype=torch.bool) & torch.isfinite(values)
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return ranks
    valid_values = values[valid_indices].to(dtype=torch.float32)
    if valid_indices.numel() == 1:
        ranks[valid_indices] = 1.0
        return ranks
    sorted_values = torch.sort(valid_values).values
    lower = torch.searchsorted(sorted_values, valid_values, right=False).to(dtype=torch.float32)
    upper = torch.searchsorted(sorted_values, valid_values, right=True).to(dtype=torch.float32)
    average_rank = (lower + upper - 1.0) * 0.5
    ranks[valid_indices] = average_rank / float(valid_indices.numel() - 1)
    return ranks


def _compute_no_o_utility(structural_value, redundancy, lambda_redundancy, utility):
    if structural_value is None:
        if utility is None:
            raise ValueError("structural_value or utility must be provided for No-O diagnostics.")
        return torch.nan_to_num(utility.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if redundancy is None:
        redundancy = torch.zeros_like(structural_value)
    lam = float(lambda_redundancy)
    denom = 1.0 + lam * redundancy.to(dtype=torch.float32)
    no_o = structural_value.to(dtype=torch.float32) / denom.clamp_min(1e-8)
    return torch.nan_to_num(no_o, nan=0.0, posinf=0.0, neginf=0.0)


def _overlap_count(left, right):
    return int((left & right).sum().item())


def _overlap_ratio(left, right, denominator):
    return _ratio(_overlap_count(left, right), denominator)


def _masked_min(values, mask):
    finite = _masked_values(values, mask)
    if finite.numel() == 0:
        return 0.0
    return finite.min().item()


def _masked_max(values, mask):
    finite = _masked_values(values, mask)
    if finite.numel() == 0:
        return 0.0
    return finite.max().item()


def _utility_stats(clone, split, selected, boundary, turning, defect, redundancy, utility, allocation_score, n, device):
    b = _optional_1d(boundary, n, device, "boundary")
    k = _optional_1d(turning, n, device, "turning")
    d = _optional_1d(defect, n, device, "defect")
    r = _optional_1d(redundancy, n, device, "redundancy")
    u = _optional_1d(utility, n, device, "utility")
    a = _optional_1d(allocation_score, n, device, "allocation_score")
    stats = {
        "clone_selected_mean_U": _masked_mean(u, clone) if u is not None else 0.0,
        "split_selected_mean_U": _masked_mean(u, split) if u is not None else 0.0,
        "selected_mean_B": _masked_mean(b, selected) if b is not None else 0.0,
        "selected_mean_K": _masked_mean(k, selected) if k is not None else 0.0,
        "selected_mean_D": _masked_mean(d, selected) if d is not None else 0.0,
        "selected_mean_R": _masked_mean(r, selected) if r is not None else 0.0,
        "selected_mean_U": _masked_mean(u, selected) if u is not None else 0.0,
    }
    if a is not None:
        stats.update(
            {
                "clone_selected_mean_A": _masked_mean(a, clone),
                "split_selected_mean_A": _masked_mean(a, split),
                "selected_mean_A": _masked_mean(a, selected),
            }
        )
    return stats


def _finite_float(value):
    if isinstance(value, int):
        return value
    value = float(value)
    if value != value or value == float("inf") or value == -float("inf"):
        return 0.0
    return value


def _sanitize_nested(value):
    if isinstance(value, dict):
        return {key: _sanitize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested(item) for item in value]
    if isinstance(value, (int, float)):
        return _finite_float(value)
    return value
