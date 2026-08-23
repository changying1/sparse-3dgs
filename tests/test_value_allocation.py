import pytest
import torch

from utils.value_allocation import (
    allocate_budget,
    build_candidate_pool,
    compute_refine_utility,
    compute_structural_value,
    robust_normalize,
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


def test_invalid_quantile_and_masks_raise_value_error():
    with pytest.raises(ValueError):
        robust_normalize(torch.tensor([1.0, 2.0]), low_quantile=0.9, high_quantile=0.1)
    with pytest.raises(ValueError):
        build_candidate_pool()
