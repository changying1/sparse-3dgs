from argparse import Namespace

import pytest

from utils.training_mode_utils import (
    compute_effective_densification_budget,
    should_run_densification,
    validate_densification_mode,
    validate_gestalt_parameters,
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
        "value_rerank_fraction": 0.25,
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


def test_value_gestalt_is_supported_with_value_and_enabled_gestalt_parameters():
    opt = _valid_value_gestalt_opt()

    assert validate_densification_mode(opt) == "value_gestalt"


def test_value_gestalt_missing_value_parameters_raises_value_error():
    opt = _opt(
        "value_gestalt",
        enable_gestalt_loss=True,
        gestalt_warmup=5000,
        lambda_gestalt=0.01,
        lambda_normal=1.0,
        gestalt_edge_sample_num=20000,
        gestalt_graph_refresh_interval=100,
    )

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_value_gestalt_requires_enabled_gestalt_loss():
    opt = _valid_value_gestalt_opt(enable_gestalt_loss=False)

    with pytest.raises(ValueError, match="enable_gestalt_loss"):
        validate_densification_mode(opt)


def test_demand_value_is_supported_with_value_parameters():
    opt = _valid_value_opt(mode="demand_value")

    assert validate_densification_mode(opt) == "demand_value"


def test_demand_value_reuses_value_parameter_validation():
    opt = _valid_value_opt(mode="demand_value", tau_e=0.0)

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_demand_value_gestalt_requires_enabled_gestalt_loss():
    opt = _valid_value_gestalt_opt(mode="demand_value_gestalt", enable_gestalt_loss=False)

    with pytest.raises(ValueError, match="enable_gestalt_loss"):
        validate_densification_mode(opt)


def test_demand_value_gestalt_passes_with_gestalt_loss_enabled():
    opt = _valid_value_gestalt_opt(mode="demand_value_gestalt", enable_gestalt_loss=True)

    assert validate_densification_mode(opt) == "demand_value_gestalt"


def test_value_rerank_is_supported_with_value_parameters():
    opt = _valid_value_opt(mode="value_rerank")

    assert validate_densification_mode(opt) == "value_rerank"


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5])
def test_value_rerank_accepts_valid_fraction_range(fraction):
    opt = _valid_value_opt(mode="value_rerank", value_rerank_fraction=fraction)

    assert validate_densification_mode(opt) == "value_rerank"


@pytest.mark.parametrize("fraction", [-0.1, 0.51])
def test_value_rerank_rejects_invalid_fraction_range(fraction):
    opt = _valid_value_opt(mode="value_rerank", value_rerank_fraction=fraction)

    with pytest.raises(ValueError, match="value_rerank_fraction"):
        validate_densification_mode(opt)


def test_value_rerank_does_not_require_gestalt_flag():
    opt = _valid_value_opt(mode="value_rerank")
    if hasattr(opt, "enable_gestalt_loss"):
        delattr(opt, "enable_gestalt_loss")

    assert validate_densification_mode(opt) == "value_rerank"


def test_value_rerank_gestalt_is_supported_with_value_rerank_and_enabled_gestalt_parameters():
    opt = _valid_value_gestalt_opt(mode="value_rerank_gestalt")

    assert validate_densification_mode(opt) == "value_rerank_gestalt"
    assert opt.value_rerank_fraction == 0.25


def test_value_rerank_gestalt_requires_enabled_gestalt_loss():
    opt = _valid_value_gestalt_opt(mode="value_rerank_gestalt", enable_gestalt_loss=False)

    with pytest.raises(ValueError, match="enable_gestalt_loss"):
        validate_densification_mode(opt)


def test_value_rerank_gestalt_reuses_value_parameter_validation():
    opt = _valid_value_gestalt_opt(mode="value_rerank_gestalt", tau_e=0.0)

    with pytest.raises(ValueError, match="tau_e"):
        validate_densification_mode(opt)


def test_value_rerank_gestalt_reuses_fraction_validation():
    opt = _valid_value_gestalt_opt(mode="value_rerank_gestalt", value_rerank_fraction=0.51)

    with pytest.raises(ValueError, match="value_rerank_fraction"):
        validate_densification_mode(opt)


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


def test_valid_gestalt_parameters_pass_validation():
    opt = _valid_gestalt_opt()

    assert validate_gestalt_parameters(opt) is opt
    assert opt.lambda_gestalt == 0.01
    assert opt.lambda_normal == 1.0


def test_gestalt_warmup_rejects_negative_value():
    opt = _valid_gestalt_opt(gestalt_warmup=-1)

    with pytest.raises(ValueError, match="gestalt_warmup"):
        validate_gestalt_parameters(opt)


def test_lambda_gestalt_rejects_negative_value():
    opt = _valid_gestalt_opt(lambda_gestalt=-0.1)

    with pytest.raises(ValueError, match="lambda_gestalt"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_lambda_gestalt_rejects_non_finite_values(value):
    opt = _valid_gestalt_opt(lambda_gestalt=value)

    with pytest.raises(ValueError, match="lambda_gestalt"):
        validate_gestalt_parameters(opt)


def test_lambda_normal_rejects_negative_value():
    opt = _valid_gestalt_opt(lambda_normal=-0.1)

    with pytest.raises(ValueError, match="lambda_normal"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_lambda_normal_rejects_non_finite_values(value):
    opt = _valid_gestalt_opt(lambda_normal=value)

    with pytest.raises(ValueError, match="lambda_normal"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [0, -1])
def test_gestalt_edge_sample_num_rejects_non_positive_values(value):
    opt = _valid_gestalt_opt(gestalt_edge_sample_num=value)

    with pytest.raises(ValueError, match="gestalt_edge_sample_num"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [0, -1])
def test_gestalt_graph_refresh_interval_rejects_non_positive_values(value):
    opt = _valid_gestalt_opt(gestalt_graph_refresh_interval=value)

    with pytest.raises(ValueError, match="gestalt_graph_refresh_interval"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [1.5, "20000"])
def test_gestalt_edge_sample_num_rejects_non_integer_values(value):
    opt = _valid_gestalt_opt(gestalt_edge_sample_num=value)

    with pytest.raises(ValueError, match="gestalt_edge_sample_num"):
        validate_gestalt_parameters(opt)


@pytest.mark.parametrize("value", [1.5, "100"])
def test_gestalt_graph_refresh_interval_rejects_non_integer_values(value):
    opt = _valid_gestalt_opt(gestalt_graph_refresh_interval=value)

    with pytest.raises(ValueError, match="gestalt_graph_refresh_interval"):
        validate_gestalt_parameters(opt)


def test_enable_gestalt_loss_rejects_non_bool_value():
    opt = _valid_gestalt_opt(enable_gestalt_loss=1)

    with pytest.raises(ValueError, match="enable_gestalt_loss"):
        validate_gestalt_parameters(opt)


def _valid_value_opt(mode="value_allocation", **overrides):
    values = {
        "tau_e": 2.0,
        "tau_s": 10.0,
        "omega_boundary": 1.0,
        "omega_turning": 1.0,
        "omega_defect": 1.0,
        "lambda_redundancy": 0.0,
    }
    values.update(overrides)
    return _opt(mode, **values)


def _valid_gestalt_opt(**overrides):
    values = {
        "enable_gestalt_loss": False,
        "gestalt_warmup": 5000,
        "lambda_gestalt": 0.01,
        "lambda_normal": 1.0,
        "gestalt_edge_sample_num": 20000,
        "gestalt_graph_refresh_interval": 100,
    }
    values.update(overrides)
    return Namespace(**values)


def _valid_value_gestalt_opt(mode="value_gestalt", **overrides):
    values = {
        "tau_e": 2.0,
        "tau_s": 10.0,
        "omega_boundary": 1.0,
        "omega_turning": 1.0,
        "omega_defect": 1.0,
        "lambda_redundancy": 0.0,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
        "enable_gestalt_loss": True,
        "gestalt_warmup": 5000,
        "lambda_gestalt": 0.01,
        "lambda_normal": 1.0,
        "gestalt_edge_sample_num": 20000,
        "gestalt_graph_refresh_interval": 100,
    }
    values.update(overrides)
    return _opt(mode, **values)
