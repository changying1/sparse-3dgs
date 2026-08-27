import torch

from utils.budget_densification import (
    build_official_clone_split_masks,
    compute_gradient_score,
    select_gradient_topk,
)
from utils.value_allocation import allocate_budget
from utils.value_diagnostics import compute_value_selection_diagnostics, format_value_diagnostics_log


def _diag(
    scaling,
    selected,
    clone,
    split,
    gradient=None,
    gradient_selected=None,
    source_type=None,
    utility=None,
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
