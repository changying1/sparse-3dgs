import pytest
import torch

from utils.value_allocation import (
    allocate_budget,
    build_candidate_pool,
    compute_demand_weighted_value_score,
    compute_refine_utility,
    compute_structural_value,
    robust_normalize,
    select_gradient_priority_value_rerank_topk,
)


def test_robust_normalize_range_and_outlier_clipping():
    values = torch.tensor([0.0, 1.0, 2.0, 1000.0])

    normalized = robust_normalize(values, low_quantile=0.25, high_quantile=0.75)

    assert normalized.shape == values.shape
    assert torch.isfinite(normalized).all()
    assert ((normalized >= 0.0) & (normalized <= 1.0)).all()
    assert normalized[-1] == 1.0


def test_constant_normalization_returns_stable_zeros():
    values = torch.ones(5)

    normalized = robust_normalize(values)

    assert torch.isfinite(normalized).all()
    assert torch.equal(normalized, torch.zeros_like(normalized))


def test_robust_normalize_non_finite_inputs_become_zero():
    values = torch.tensor([0.0, 1.0, float("nan"), float("inf"), -float("inf")])

    normalized = robust_normalize(values, low_quantile=0.0, high_quantile=1.0)

    assert torch.isfinite(normalized).all()
    assert normalized[2] == 0.0
    assert normalized[3] == 0.0
    assert normalized[4] == 0.0


def test_structural_value_weighted_sum():
    boundary = torch.tensor([0.1, 0.2])
    turning = torch.tensor([0.3, 0.4])
    defect = torch.tensor([0.5, 0.6])

    value = compute_structural_value(boundary, turning, defect, 2.0, 3.0, 4.0)

    expected = 2.0 * boundary + 3.0 * turning + 4.0 * defect
    assert torch.isfinite(value).all()
    assert torch.allclose(value, expected)


def test_redundancy_suppresses_utility():
    observation = torch.tensor([0.8, 0.8])
    structural = torch.tensor([0.5, 0.5])
    redundancy = torch.tensor([0.0, 2.0])

    utility = compute_refine_utility(observation, structural, redundancy, lambda_redundancy=1.0)

    assert torch.isfinite(utility).all()
    assert utility[0] > utility[1]


def test_zero_evidence_zeroes_utility():
    observation = torch.tensor([0.0])
    structural = torch.tensor([100.0])
    redundancy = torch.tensor([0.0])

    utility = compute_refine_utility(observation, structural, redundancy, lambda_redundancy=1.0)

    assert torch.isfinite(utility).all()
    assert utility.item() == 0.0


def test_zero_lambda_redundancy_equals_observation_times_structural_value():
    observation = torch.tensor([0.2, 0.5])
    structural = torch.tensor([3.0, 4.0])
    redundancy = torch.tensor([100.0, 200.0])

    utility = compute_refine_utility(observation, structural, redundancy, lambda_redundancy=0.0)

    assert torch.isfinite(utility).all()
    assert torch.allclose(utility, observation * structural)


def test_candidate_pool_union():
    a = torch.tensor([True, False, False, False])
    b = torch.tensor([False, True, False, False])
    c = torch.tensor([False, False, True, False])
    d = torch.tensor([False, False, False, True])

    candidate = build_candidate_pool(a, b, c, d)

    assert candidate.dtype == torch.bool
    assert torch.equal(candidate, torch.tensor([True, True, True, True]))


def test_budget_topk_selects_highest_utility_indices():
    utility = torch.tensor([0.1, 0.9, 0.4, 0.8])

    selected = allocate_budget(utility, budget=2)

    assert selected.dtype == torch.long
    assert selected.tolist() == [1, 3]


def test_candidate_mask_excludes_high_utility_non_candidate():
    utility = torch.tensor([0.1, 0.9, 0.4, 0.8])
    candidate = torch.tensor([True, False, True, True])

    selected = allocate_budget(utility, budget=2, candidate_mask=candidate)

    assert 1 not in selected.tolist()
    assert selected.tolist() == [3, 2]


def test_budget_larger_than_candidates_returns_all_valid_candidates():
    utility = torch.tensor([0.1, 0.9, 0.4])
    candidate = torch.tensor([True, False, True])

    selected = allocate_budget(utility, budget=10, candidate_mask=candidate)

    assert selected.tolist() == [2, 0]


def test_zero_utility_does_not_consume_budget():
    utility = torch.tensor([0.8, 0.4, 0.0, 0.0])

    selected = allocate_budget(utility, budget=4)

    assert selected.tolist() == [0, 1]


def test_all_zero_utility_returns_empty():
    utility = torch.zeros(4)

    selected = allocate_budget(utility, budget=3)

    assert selected.dtype == torch.long
    assert selected.shape == (0,)


def test_nan_and_inf_utility_are_not_selected():
    utility = torch.tensor([0.1, float("nan"), float("inf"), 0.4, -float("inf")])

    selected = allocate_budget(utility, budget=5)

    assert selected.tolist() == [3, 0]


def test_demand_weighted_value_score_is_raw_gradient_times_utility():
    gradient = torch.tensor([1.0, 2.0, 3.0])
    utility = torch.tensor([3.0, 2.0, 1.0])

    allocation_score = compute_demand_weighted_value_score(gradient, utility)

    assert torch.allclose(allocation_score, torch.tensor([3.0, 4.0, 3.0]))


def test_demand_weighted_value_score_does_not_mutate_inputs():
    gradient = torch.tensor([1.0, 2.0, 3.0])
    utility = torch.tensor([3.0, 2.0, 1.0])
    gradient_before = gradient.clone()
    utility_before = utility.clone()

    compute_demand_weighted_value_score(gradient, utility)

    assert torch.equal(gradient, gradient_before)
    assert torch.equal(utility, utility_before)


def test_demand_weighted_value_score_sanitizes_non_finite_inputs():
    gradient = torch.tensor([1.0, float("nan"), float("inf"), 4.0])
    utility = torch.tensor([2.0, 3.0, 4.0, float("inf")])

    allocation_score = compute_demand_weighted_value_score(gradient, utility)

    assert torch.isfinite(allocation_score).all()
    assert torch.equal(allocation_score, torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_return_mask_selects_highest_positive_utility_positions():
    utility = torch.tensor([0.2, 0.0, 0.9, 0.5])

    selected_mask = allocate_budget(utility, budget=2, return_mask=True)

    assert selected_mask.dtype == torch.bool
    assert selected_mask.shape == utility.shape
    assert selected_mask.sum().item() == 2
    assert torch.equal(selected_mask, torch.tensor([False, False, True, True]))


def test_zero_budget_returns_empty_long_tensor():
    utility = torch.tensor([0.1, 0.9])

    selected = allocate_budget(utility, budget=0)

    assert selected.dtype == torch.long
    assert selected.shape == (0,)


def test_value_rerank_fraction_zero_matches_gradient_topk():
    gradient = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
    utility = torch.tensor([0.1, 100.0, 50.0, 0.2, 0.3])
    candidate = torch.ones((5,), dtype=torch.bool)

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=3,
        rerank_fraction=0.0,
    )

    assert selected.tolist() == [0, 1, 2]


def test_value_rerank_reorders_only_boundary_pool():
    gradient = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    utility = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 0.8, 0.0, 0.0, 0.0, 1000.0])
    candidate = torch.ones((10,), dtype=torch.bool)

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=4,
        rerank_fraction=0.5,
    )

    assert selected.tolist() == [0, 1, 4, 5]
    assert 9 not in selected.tolist()


def test_value_rerank_selects_all_when_eligible_count_is_within_budget():
    gradient = torch.tensor([10.0, 9.0, 8.0, 7.0])
    utility = torch.tensor([0.1, 0.2, 0.3, 0.4])
    candidate = torch.tensor([True, False, True, False])

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=3,
        rerank_fraction=0.25,
    )

    assert selected.tolist() == [0, 2]


def test_value_rerank_candidate_mask_excludes_extreme_scores():
    gradient = torch.tensor([10.0, 9.0, 1000.0, 7.0])
    utility = torch.tensor([0.1, 0.2, 1000.0, 0.4])
    candidate = torch.tensor([True, True, False, True])

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=2,
        rerank_fraction=0.5,
    )

    assert 2 not in selected.tolist()


def test_value_rerank_rejects_non_finite_gradient_or_utility():
    gradient = torch.tensor([10.0, float("nan"), float("inf"), 7.0])
    utility = torch.tensor([0.1, 0.2, 0.3, float("inf")])
    candidate = torch.ones((4,), dtype=torch.bool)

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=3,
        rerank_fraction=0.5,
    )

    assert selected.tolist() == [0]


def test_value_rerank_is_deterministic_and_respects_budget():
    gradient = torch.tensor([5.0, 5.0, 4.0, 4.0, 3.0])
    utility = torch.tensor([1.0, 2.0, 5.0, 4.0, 100.0])
    candidate = torch.ones((5,), dtype=torch.bool)

    first = select_gradient_priority_value_rerank_topk(gradient, utility, candidate, 3, 0.5)
    second = select_gradient_priority_value_rerank_topk(gradient, utility, candidate, 3, 0.5)

    assert torch.equal(first, second)
    assert first.numel() == 3


def test_value_rerank_protects_core_and_bounds_selected_rank():
    gradient = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    utility = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.9, 0.8, 100.0, 200.0])
    candidate = torch.ones((8,), dtype=torch.bool)

    selected = select_gradient_priority_value_rerank_topk(
        gradient,
        utility,
        candidate,
        budget=4,
        rerank_fraction=0.5,
    )

    assert {0, 1}.issubset(set(selected.tolist()))
    assert max(selected.tolist()) <= 5


def test_invalid_quantile_and_masks_raise_value_error():
    with pytest.raises(ValueError):
        robust_normalize(torch.tensor([1.0, 2.0]), low_quantile=0.9, high_quantile=0.1)
    with pytest.raises(ValueError):
        build_candidate_pool()
