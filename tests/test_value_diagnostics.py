import torch

from utils.budget_densification import (
    build_official_clone_split_masks,
    compute_gradient_score,
    select_gradient_topk,
)
from utils.value_allocation import allocate_budget
from utils.value_diagnostics import (
    build_multiview_value_diagnostics,
    compute_value_selection_diagnostics,
    compute_value_rerank_diagnostics,
    format_multiview_value_diagnostics_log,
    format_value_diagnostics_log,
    format_value_rerank_diagnostics_log,
)


def _diag(
    scaling,
    selected,
    clone,
    split,
    gradient=None,
    gradient_selected=None,
    source_type=None,
    utility=None,
    allocation_score=None,
):
    n = scaling.shape[0]
    if gradient is None:
        gradient = torch.ones((n,), dtype=torch.float32)
    if gradient_selected is None:
        gradient_selected = torch.zeros((n,), dtype=torch.bool)
    if utility is None:
        utility = torch.arange(1, n + 1, dtype=torch.float32)
    return compute_value_selection_diagnostics(
        scaling=scaling,
        selected_mask=selected,
        clone_mask=clone,
        split_mask=split,
        percent_dense=0.1,
        scene_extent=1.0,
        gradient_score=gradient,
        gradient_selected_mask=gradient_selected,
        official_gradient_threshold=0.5,
        source_type=source_type,
        boundary=utility + 10.0,
        turning=utility + 20.0,
        defect=utility + 30.0,
        redundancy=utility + 40.0,
        utility=utility,
        allocation_score=allocation_score,
    )


def test_all_selected_are_clone_eligible():
    scaling = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.20, 0.20, 0.20]])
    selected = torch.tensor([True, True, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)

    stats = _diag(scaling, selected, clone, split)

    assert stats["selected_clone_eligible_count"] == 2
    assert stats["selected_split_eligible_count"] == 0
    assert stats["selected_clone_eligible_ratio"] == 1.0
    assert stats["selected_split_eligible_ratio"] == 0.0


def test_clone_split_mixed_selection_counts_and_mean_u():
    scaling = torch.tensor([[0.02, 0.02, 0.02], [0.20, 0.20, 0.20], [0.05, 0.05, 0.05]])
    selected = torch.tensor([True, True, True])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    utility = torch.tensor([2.0, 10.0, 4.0])

    stats = _diag(scaling, selected, clone, split, utility=utility)

    assert stats["selected_clone_eligible_count"] == 2
    assert stats["selected_split_eligible_count"] == 1
    assert stats["clone_selected_mean_U"] == 3.0
    assert stats["split_selected_mean_U"] == 10.0


def test_empty_selected_mask_returns_zero_selected_stats():
    scaling = torch.tensor([[0.02, 0.02, 0.02], [0.20, 0.20, 0.20]])
    selected = torch.tensor([False, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)

    stats = _diag(scaling, selected, clone, split)

    assert stats["value_selected_count"] == 0
    assert stats["selected_scale_median"] == 0.0
    assert stats["selected_gradient_mean"] == 0.0
    assert stats["selected_clone_eligible_ratio"] == 0.0
    assert stats["clone_selected_mean_U"] == 0.0
    assert stats["split_selected_mean_U"] == 0.0


def test_source_type_distribution_and_clone_source_ratio():
    scaling = torch.full((5, 3), 0.02)
    selected = torch.tensor([True, True, True, True, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    source_type = torch.tensor([0, 1, 2, 3, 1])

    stats = _diag(scaling, selected, clone, split, source_type=source_type)

    assert stats["selected_source_initial"] == 1
    assert stats["selected_source_clone"] == 1
    assert stats["selected_source_split"] == 1
    assert stats["selected_source_completion"] == 1
    assert stats["selected_clone_source_ratio"] == 0.25


def test_gradient_overlap_is_value_selected_denominator():
    scaling = torch.full((5, 3), 0.02)
    selected = torch.tensor([True, True, True, False, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient_selected = torch.tensor([False, True, False, True, True])

    stats = _diag(scaling, selected, clone, split, gradient_selected=gradient_selected)

    assert stats["value_gradient_overlap_count"] == 1
    assert stats["value_gradient_overlap_ratio"] == 1.0 / 3.0


def test_gradient_threshold_ratio_uses_official_threshold():
    scaling = torch.full((4, 3), 0.02)
    selected = torch.tensor([True, True, True, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient = torch.tensor([0.1, 0.5, 1.2, 3.0])

    stats = _diag(scaling, selected, clone, split, gradient=gradient)

    assert stats["selected_gradient_positive_ratio"] == 1.0
    assert stats["selected_gradient_above_official_threshold_ratio"] == 2.0 / 3.0


def test_scale_quantiles_and_median_are_finite_with_invalid_inputs():
    scaling = torch.tensor(
        [
            [0.02, 0.03, 0.04],
            [float("nan"), 0.05, 0.06],
            [float("inf"), 0.07, 0.08],
            [0.20, 0.20, 0.20],
        ]
    )
    selected = torch.tensor([True, True, True, True])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient = torch.tensor([0.1, float("nan"), float("inf"), -float("inf")])
    utility = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])

    stats = _diag(scaling, selected, clone, split, gradient=gradient, utility=utility)

    assert all(torch.isfinite(torch.tensor(float(value))) for value in stats.values())
    assert stats["all_scale_median"] > 0.0
    assert stats["selected_scale_q10"] > 0.0
    assert stats["selected_scale_q90"] > 0.0


def test_empty_clone_or_split_mean_u_is_zero():
    scaling = torch.tensor([[0.20, 0.20, 0.20], [0.30, 0.30, 0.30]])
    selected = torch.tensor([True, True])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)

    stats = _diag(scaling, selected, clone, split, utility=torch.tensor([5.0, 7.0]))

    assert clone.sum().item() == 0
    assert stats["clone_selected_mean_U"] == 0.0
    assert stats["split_selected_mean_U"] == 6.0


def test_diagnostic_helper_does_not_mutate_inputs():
    scaling = torch.tensor([[0.02, 0.02, 0.02], [0.20, 0.20, 0.20], [0.05, 0.05, 0.05]])
    selected = torch.tensor([True, True, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient = torch.tensor([0.3, 0.8, 1.5])
    gradient_selected = torch.tensor([False, True, True])
    source_type = torch.tensor([0, 1, 2])
    utility = torch.tensor([1.0, 2.0, 3.0])
    inputs = [scaling, selected, clone, split, gradient, gradient_selected, source_type, utility]
    before = [tensor.clone() for tensor in inputs]

    _diag(scaling, selected, clone, split, gradient, gradient_selected, source_type, utility)

    for tensor, expected in zip(inputs, before):
        assert torch.equal(tensor, expected)


def test_training_mode_masks_are_unchanged_by_diagnostics():
    utility = torch.tensor([0.1, 0.9, 0.7, 0.2])
    scaling = torch.tensor(
        [
            [0.02, 0.02, 0.02],
            [0.20, 0.20, 0.20],
            [0.03, 0.03, 0.03],
            [0.30, 0.30, 0.30],
        ]
    )
    selected = allocate_budget(utility, 3, candidate_mask=None, return_mask=True)
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient = compute_gradient_score(
        torch.tensor([[1.0], [4.0], [3.0], [2.0]]),
        torch.ones((4, 1)),
    )
    gradient_selected = select_gradient_topk(gradient, 3, candidate_mask=None, return_mask=True)

    _diag(scaling, selected, clone, split, gradient, gradient_selected, utility=utility)
    selected_after = allocate_budget(utility, 3, candidate_mask=None, return_mask=True)
    clone_after, split_after = build_official_clone_split_masks(selected_after, scaling, 0.1, 1.0)

    assert torch.equal(selected, selected_after)
    assert torch.equal(clone, clone_after)
    assert torch.equal(split, split_after)


def test_format_value_diagnostics_log_contains_expected_fields():
    scaling = torch.full((2, 3), 0.02)
    selected = torch.tensor([True, False])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    stats = _diag(scaling, selected, clone, split)

    message = format_value_diagnostics_log(2100, stats)
    lines = message.splitlines()

    assert "[ValueDiag ITER 2100]" in message
    assert len(lines) == 4
    assert all(line.startswith("[ValueDiag ITER 2100]") for line in lines)
    for field in (
        "threshold=",
        "all_scale_mean=",
        "all_scale_median=",
        "all_scale_q10=",
        "all_scale_q90=",
        "selected_scale_mean=",
        "selected_scale_median=",
        "selected_scale_q10=",
        "selected_scale_q90=",
        "all_clone_ratio=",
        "all_split_ratio=",
        "selected_clone_ratio=",
        "selected_split_ratio=",
    ):
        assert field in lines[0]
    for field in (
        "selected_gradient_mean=",
        "selected_gradient_median=",
        "grad_positive_ratio=",
        "grad_threshold_ratio=",
        "gradient_overlap=",
    ):
        assert field in lines[1]
    for field in (
        "source_initial=",
        "source_clone=",
        "source_split=",
        "source_completion=",
        "clone_source_ratio=",
    ):
        assert field in lines[2]
    for field in (
        "selected_mean_B=",
        "selected_mean_K=",
        "selected_mean_D=",
        "selected_mean_R=",
        "selected_mean_U=",
        "clone_mean_U=",
        "split_mean_U=",
    ):
        assert field in lines[3]


def test_value_diagnostics_include_optional_allocation_score_stats():
    scaling = torch.full((3, 3), 0.02)
    selected = torch.tensor([True, False, True])
    clone, split = build_official_clone_split_masks(selected, scaling, 0.1, 1.0)
    gradient = torch.tensor([1.0, 2.0, 3.0])
    utility = torch.tensor([3.0, 2.0, 1.0])
    allocation_score = gradient * utility

    stats = _diag(
        scaling,
        selected,
        clone,
        split,
        gradient=gradient,
        utility=utility,
        allocation_score=allocation_score,
    )
    message = format_value_diagnostics_log(2100, stats)

    assert stats["selected_mean_A"] == 3.0
    assert "selected_mean_A=3.000000" in message


def _mv_diag(
    gradient=None,
    selected=None,
    utility=None,
    visible=None,
    observation=None,
    structural=None,
    redundancy=None,
    source_type=None,
    allocation_score=None,
    value_reference=None,
    budget=3,
    threshold=0.5,
):
    if gradient is None:
        gradient = torch.tensor([0.5, 0.7, 1.4, 0.2, 2.0, float("nan")])
    n = gradient.shape[0]
    if selected is None:
        if n == 6:
            selected = torch.tensor([True, True, False, False, True, False])
        else:
            selected = torch.zeros((n,), dtype=torch.bool)
            selected[: min(2, n)] = True
    if utility is None:
        if n == 6:
            utility = torch.tensor([9.0, 8.0, 1.0, 100.0, 7.0, 50.0])
        else:
            utility = torch.arange(n, 0, -1, dtype=torch.float32)
    if visible is None:
        if n == 6:
            visible = torch.tensor([4, 4, 5, 5, 6, 6])
        else:
            visible = torch.arange(n, dtype=torch.long) // 2 + 4
    if observation is None:
        if n == 6:
            observation = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.1, 0.0])
        else:
            observation = torch.linspace(0.9, 0.1, steps=n)
    if structural is None:
        if n == 6:
            structural = torch.tensor([2.0, 1.0, 8.0, 3.0, 6.0, 10.0])
        else:
            structural = torch.arange(1, n + 1, dtype=torch.float32)
    if redundancy is None:
        redundancy = torch.zeros((n,), dtype=torch.float32)
    scaling = torch.full((n, 3), 0.02)
    return build_multiview_value_diagnostics(
        scaling=scaling,
        selected_mask=selected,
        effective_budget=budget,
        gradient_score=gradient,
        official_gradient_threshold=threshold,
        visible_view_count=visible,
        observation=observation,
        boundary=structural + 10.0,
        turning=structural + 20.0,
        defect=structural + 30.0,
        redundancy=redundancy,
        utility=utility,
        allocation_score=allocation_score,
        structural_value=structural,
        lambda_redundancy=0.0,
        source_type=source_type,
        value_reference_mask=value_reference,
    )


def test_multiview_matched_gradient_topk_uses_eligible_pool_and_budget():
    stats = _mv_diag(budget=2)

    assert stats["eligible_count"] == 4
    assert stats["matched_budget"] == 2
    assert stats["gradient_topk_count"] == 2
    gradient_topk = stats["groups"]["gradient_topk"]
    assert abs(gradient_topk["gradient_median"] - 1.7) < 1e-6
    assert stats["rank"]["value_vs_gradient_overlap_count"] == 1


def test_multiview_near_threshold_ratios_use_finite_eligible_values():
    gradient = torch.tensor([0.5, 0.6249, 0.625, 0.7499, 0.75, 0.9999, 1.0, float("nan"), float("inf")])
    selected = torch.tensor([True, True, True, True, True, True, True, False, False])
    n = gradient.shape[0]
    scaling = torch.full((n, 3), 0.02)

    stats = build_multiview_value_diagnostics(
        scaling=scaling,
        selected_mask=selected,
        effective_budget=7,
        gradient_score=gradient,
        official_gradient_threshold=0.5,
        visible_view_count=torch.arange(n),
        observation=torch.ones((n,), dtype=torch.float32),
        utility=torch.ones((n,), dtype=torch.float32),
        structural_value=torch.ones((n,), dtype=torch.float32),
        redundancy=torch.zeros((n,), dtype=torch.float32),
    )

    selected_stats = stats["groups"]["value_selected"]
    assert selected_stats["near_threshold_1_25x_ratio"] == 2.0 / 7.0
    assert selected_stats["near_threshold_1_5x_ratio"] == 4.0 / 7.0
    assert selected_stats["near_threshold_2x_ratio"] == 6.0 / 7.0
    assert abs(selected_stats["grad_margin_median"] - 1.4998) < 1e-6


def test_multiview_visibility_bins_report_three_selection_counts_and_rates():
    stats = _mv_diag(budget=3)
    bins = {item["n"]: item for item in stats["visibility_bins"]}

    assert bins[4]["eligible_count"] == 2
    assert bins[4]["value_selected_count"] == 2
    assert bins[4]["gradient_topk_count"] == 1
    assert bins[4]["value_selection_rate"] == 1.0
    assert bins[4]["gradient_selection_rate"] == 0.5
    assert bins[5]["eligible_count"] == 1
    assert bins[5]["value_selected_count"] == 0
    assert bins[6]["eligible_count"] == 1
    assert bins[6]["gradient_topk_count"] == 1


def test_multiview_rank_percentiles_are_high_for_high_gradient_and_stable_for_ties():
    gradient = torch.tensor([0.5, 1.0, 1.0, 2.0])
    selected = torch.tensor([False, True, True, True])
    stats = build_multiview_value_diagnostics(
        scaling=torch.full((4, 3), 0.02),
        selected_mask=selected,
        effective_budget=3,
        gradient_score=gradient,
        official_gradient_threshold=0.5,
        visible_view_count=torch.tensor([1, 1, 1, 1]),
        observation=torch.tensor([0.1, 0.2, 0.2, 0.9]),
        utility=torch.ones((4,), dtype=torch.float32),
        structural_value=torch.ones((4,), dtype=torch.float32),
        redundancy=torch.zeros((4,), dtype=torch.float32),
    )

    assert stats["rank"]["value_selected_gradient_rank_percentile_median"] == 0.5
    assert stats["rank"]["value_selected_gradient_rank_percentile_mean"] > 0.6
    singleton = build_multiview_value_diagnostics(
        scaling=torch.full((1, 3), 0.02),
        selected_mask=torch.tensor([True]),
        effective_budget=1,
        gradient_score=torch.tensor([2.0]),
        official_gradient_threshold=0.5,
        visible_view_count=torch.tensor([1]),
        observation=torch.tensor([1.0]),
        utility=torch.tensor([1.0]),
        structural_value=torch.tensor([1.0]),
        redundancy=torch.tensor([0.0]),
    )
    assert singleton["rank"]["value_selected_gradient_rank_percentile_median"] == 1.0


def test_multiview_no_o_counterfactual_ranking_and_overlap_are_read_only():
    gradient = torch.tensor([0.6, 0.7, 0.8, 0.9])
    selected = torch.tensor([True, True, False, False])
    utility = torch.tensor([100.0, 90.0, 1.0, 2.0])
    structural = torch.tensor([1.0, 2.0, 100.0, 90.0])
    before = utility.clone()

    stats = build_multiview_value_diagnostics(
        scaling=torch.full((4, 3), 0.02),
        selected_mask=selected,
        effective_budget=2,
        gradient_score=gradient,
        official_gradient_threshold=0.5,
        visible_view_count=torch.tensor([1, 1, 2, 2]),
        observation=torch.tensor([0.9, 0.8, 0.1, 0.1]),
        utility=utility,
        structural_value=structural,
        redundancy=torch.zeros((4,), dtype=torch.float32),
    )

    assert torch.equal(utility, before)
    assert stats["counterfactual_no_o"]["value_vs_noO_overlap_count"] == 0
    assert stats["counterfactual_no_o"]["noO_vs_gradient_overlap_count"] == 2


def test_multiview_non_finite_inputs_do_not_propagate_to_stats():
    gradient = torch.tensor([0.5, float("nan"), float("inf"), -float("inf")])
    observation = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    utility = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])

    stats = build_multiview_value_diagnostics(
        scaling=torch.full((4, 3), 0.02),
        selected_mask=torch.tensor([True, False, False, False]),
        effective_budget=10,
        gradient_score=gradient,
        official_gradient_threshold=0.5,
        visible_view_count=torch.tensor([1, 2, 3, 4]),
        observation=observation,
        utility=utility,
        structural_value=utility,
        redundancy=torch.zeros((4,), dtype=torch.float32),
    )

    assert stats["eligible_count"] == 1
    assert stats["matched_budget"] == 1
    assert stats["groups"]["eligible"]["O_mean"] == 1.0
    assert stats["groups"]["value_selected"]["U_mean"] == 1.0


def test_format_multiview_value_diagnostics_log_contains_expected_lines():
    stats = _mv_diag(budget=2)
    message = format_multiview_value_diagnostics_log(7000, stats)
    lines = message.splitlines()

    assert lines[0].startswith("[ValueDiagMV ITER 7000]")
    assert "eligible_count=" in lines[0]
    assert "value_gradient_overlap=" in lines[0]
    assert "selected_grad_rank_median=" in lines[1]
    assert "selected_near_1.25x=" in lines[2]
    assert any(line.startswith("[ValueDiagMV ITER 7000][n=4]") for line in lines)


def test_multiview_log_contains_group_component_stats():
    stats = _mv_diag(
        budget=2,
        source_type=torch.tensor([0, 1, 2, 3, 1, 0]),
    )
    message = format_multiview_value_diagnostics_log(7000, stats)

    for group_name in ("eligible", "value_selected", "gradient_topk"):
        group_lines = [
            line
            for line in message.splitlines()
            if line.startswith(f"[ValueDiagMV ITER 7000][group={group_name}]")
        ]
        assert len(group_lines) == 5
        joined = " ".join(group_lines)
        for field in (
            "count=",
            "visible_mean=",
            "visible_median=",
            "visible_q10=",
            "visible_q90=",
            "visible_min=",
            "visible_max=",
            "O_mean=",
            "O_median=",
            "O_q10=",
            "O_q90=",
            "gradient_mean=",
            "gradient_median=",
            "gradient_q10=",
            "gradient_q90=",
            "gradient_max=",
            "scale_mean=",
            "scale_median=",
            "scale_q10=",
            "scale_q90=",
            "B_norm_mean=",
            "B_norm_median=",
            "K_norm_mean=",
            "K_norm_median=",
            "D_norm_mean=",
            "D_norm_median=",
            "R_norm_mean=",
            "R_norm_median=",
            "S_mean=",
            "S_median=",
            "U_mean=",
            "U_median=",
            "source_initial_ratio=",
            "source_clone_ratio=",
            "source_split_ratio=",
            "source_completion_ratio=",
            "grad_margin_mean=",
            "grad_margin_median=",
            "grad_margin_q10=",
            "grad_margin_q90=",
            "near_1.25x=",
            "near_1.5x=",
            "near_2x=",
        ):
            assert field in joined


def test_visibility_bin_log_contains_eligible_and_selected_utility():
    stats = _mv_diag(budget=2)
    message = format_multiview_value_diagnostics_log(7000, stats)
    n4_line = next(line for line in message.splitlines() if line.startswith("[ValueDiagMV ITER 7000][n=4]"))
    assert "eligible_grad_mean=" in n4_line
    assert "selected_grad_mean=" in n4_line
    assert "gradient_topk_grad_mean=" in n4_line
    assert "eligible_U_mean=" in n4_line
    assert "selected_U_mean=" in n4_line


def test_multiview_diagnostics_accept_normalized_components():
    boundary_norm = torch.tensor([0.0, 0.5, 1.0, 0.25])
    turning_norm = torch.tensor([1.0, 0.5, 0.0, 0.75])
    defect_norm = torch.tensor([0.2, 0.4, 0.6, 0.8])
    redundancy_norm = torch.tensor([0.1, 0.3, 0.5, 0.7])

    stats = build_multiview_value_diagnostics(
        scaling=torch.full((4, 3), 0.02),
        selected_mask=torch.tensor([True, True, False, False]),
        effective_budget=2,
        gradient_score=torch.tensor([0.5, 0.6, 0.7, 0.8]),
        official_gradient_threshold=0.5,
        visible_view_count=torch.tensor([1, 1, 2, 2]),
        observation=torch.ones((4,), dtype=torch.float32),
        boundary=boundary_norm,
        turning=turning_norm,
        defect=defect_norm,
        redundancy=redundancy_norm,
        utility=torch.ones((4,), dtype=torch.float32),
        structural_value=torch.tensor([0.4, 0.5, 0.6, 0.7]),
        lambda_redundancy=0.0,
    )

    eligible = stats["groups"]["eligible"]
    assert abs(eligible["B_mean"] - boundary_norm.mean().item()) < 1e-6
    assert abs(eligible["K_mean"] - turning_norm.mean().item()) < 1e-6
    assert abs(eligible["D_mean"] - defect_norm.mean().item()) < 1e-6
    assert abs(eligible["R_mean"] - redundancy_norm.mean().item()) < 1e-6


def test_multiview_diagnostics_report_demand_value_allocation_score_and_overlaps():
    gradient = torch.tensor([1.0, 10.0, 0.8])
    utility = torch.tensor([10.0, 2.0, 3.0])
    demand_selected = torch.tensor([False, True, True])
    value_reference = torch.tensor([True, False, True])

    stats = _mv_diag(
        gradient=gradient,
        utility=utility,
        selected=demand_selected,
        allocation_score=gradient * utility,
        value_reference=value_reference,
        budget=2,
        threshold=0.5,
    )
    message = format_multiview_value_diagnostics_log(7000, stats)

    assert stats["allocation_score"]["formula"] == "A = gradient * U"
    assert stats["rank"]["demand_value_vs_gradient_overlap_count"] == 1
    assert stats["rank"]["demand_value_vs_value_overlap_count"] == 1
    assert "A_formula='A = gradient * U'" in message
    assert "demand_value_value_overlap=1/2" in message


def test_value_rerank_diagnostics_report_boundary_and_rank_bound_fields():
    gradient = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    utility = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 0.8, 100.0, 200.0])
    candidate = torch.ones((8,), dtype=torch.bool)
    selected = torch.tensor([True, True, False, False, True, True, False, False])

    stats = compute_value_rerank_diagnostics(
        gradient_score=gradient,
        utility=utility,
        candidate_mask=candidate,
        selected_mask=selected,
        budget=4,
        rerank_fraction=0.5,
    )
    message = format_value_rerank_diagnostics_log(9000, stats)

    assert stats["budget"] == 4
    assert stats["protected_count"] == 2
    assert stats["protected_selected_count"] == 2
    assert stats["rerank_slots"] == 2
    assert stats["boundary_pool_count"] == 4
    assert stats["final_selected_count"] == 4
    assert stats["selected_max_gradient_rank"] == 6
    assert stats["selected_max_gradient_rank"] <= stats["budget"] + stats["rerank_slots"]
    assert stats["selected_below_original_topk_count"] == 2
    assert message.startswith("[ValueRerank ITER 9000]")
    assert "gradient_topk_overlap=" in message
    assert "boundary_U_median=" in message
    assert "selected_max_gradient_rank=6" in message
