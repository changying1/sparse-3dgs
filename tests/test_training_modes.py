from argparse import Namespace

import pytest

from utils.training_mode_utils import (
    compute_effective_densification_budget,
    should_run_densification,
    validate_densification_mode,
)


def _opt(mode, **overrides):
    values = {
        "densification_mode": mode,
        "tau_e": None,
        "tau_s": None,
        "omega_boundary": None,
        "omega_turning": None,
        "omega_defect": None,
        "lambda_redundancy": None,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
    }
    values.update(overrides)
    return Namespace(**values)


def test_official_does_not_require_value_parameters():
    assert validate_densification_mode(_opt("official")) == "official"


def test_budget_gradient_does_not_require_value_parameters():
    assert validate_densification_mode(_opt("budget_gradient")) == "budget_gradient"


def test_value_allocation_missing_tau_e_raises_value_error():
    opt = _opt(
        "value_allocation",
        tau_s=10.0,
        omega_boundary=1.0,
        omega_turning=1.0,
        omega_defect=1.0,
        lambda_redundancy=0.0,
    )

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_non_positive_tau_e():
    opt = _valid_value_opt(tau_e=0.0)

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_invalid_quantiles():
    opt = _valid_value_opt(normalization_low_quantile=0.95, normalization_high_quantile=0.05)

    with pytest.raises(ValueError, match="normalization_low_quantile"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_nan_tau_e():
    opt = _valid_value_opt(tau_e=float("nan"))

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_inf_omega_boundary():
    opt = _valid_value_opt(omega_boundary=float("inf"))

    with pytest.raises(ValueError, match="omega_boundary"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_negative_omega_turning():
    opt = _valid_value_opt(omega_turning=-1.0)

    with pytest.raises(ValueError, match="omega_turning"):
        validate_densification_mode(opt)


def test_value_allocation_rejects_nan_lambda_redundancy():
    opt = _valid_value_opt(lambda_redundancy=float("nan"))

    with pytest.raises(ValueError, match="lambda_redundancy"):
        validate_densification_mode(opt)


def test_value_gestalt_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="Phase 10"):
        validate_densification_mode(_opt("value_gestalt"))


def test_full_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="Phase 11"):
        validate_densification_mode(_opt("full"))


def test_effective_budget_is_shared_by_gradient_and_value_modes():
    gradient_budget = compute_effective_densification_budget(
        budget=100,
        max_gaussians=1000,
        current_gaussian_count=950,
    )
    value_budget = compute_effective_densification_budget(
        budget=100,
        max_gaussians=1000,
        current_gaussian_count=950,
    )

    assert gradient_budget == value_budget == 50
    assert compute_effective_densification_budget(100, 1000, 1000) == 0


def test_densification_schedule_matches_official_cadence():
    assert should_run_densification(600, 500, 15000, 100)
    assert not should_run_densification(500, 500, 15000, 100)
    assert not should_run_densification(601, 500, 15000, 100)
    assert not should_run_densification(15000, 500, 15000, 100)


def _valid_value_opt(**overrides):
    values = {
        "tau_e": 2.0,
        "tau_s": 10.0,
        "omega_boundary": 1.0,
        "omega_turning": 1.0,
        "omega_defect": 1.0,
        "lambda_redundancy": 0.0,
    }
    values.update(overrides)
    return _opt("value_allocation", **values)
