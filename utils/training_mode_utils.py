"""Training mode validation helpers for Sparse-view 3DGS Phase 9."""

import math


SUPPORTED_DENSIFICATION_MODES = ("official", "budget_gradient", "value_allocation")
UNIMPLEMENTED_DENSIFICATION_MODES = {
    "value_gestalt": "value_gestalt requires Phase 10.",
    "full": "full requires Phase 11.",
}
VALUE_REQUIRED_PARAMS = (
    "tau_e",
    "tau_s",
    "omega_boundary",
    "omega_turning",
    "omega_defect",
    "lambda_redundancy",
)


def validate_densification_mode(opt):
    mode = getattr(opt, "densification_mode", "official")
    if mode in UNIMPLEMENTED_DENSIFICATION_MODES:
        raise NotImplementedError(UNIMPLEMENTED_DENSIFICATION_MODES[mode])
    if mode not in SUPPORTED_DENSIFICATION_MODES:
        raise ValueError(f"Unknown densification_mode '{mode}'.")

    if mode != "value_allocation":
        return mode

    missing = [name for name in VALUE_REQUIRED_PARAMS if getattr(opt, name, None) is None]
    if missing:
        raise ValueError(
            "value_allocation requires explicit CLI values for: "
            + ", ".join(missing)
        )

    opt.tau_e = _to_float(opt.tau_e, "tau_e")
    opt.tau_s = _to_float(opt.tau_s, "tau_s")
    opt.omega_boundary = _to_float(opt.omega_boundary, "omega_boundary")
    opt.omega_turning = _to_float(opt.omega_turning, "omega_turning")
    opt.omega_defect = _to_float(opt.omega_defect, "omega_defect")
    opt.lambda_redundancy = _to_float(opt.lambda_redundancy, "lambda_redundancy")
    opt.normalization_low_quantile = _to_float(opt.normalization_low_quantile, "normalization_low_quantile")
    opt.normalization_high_quantile = _to_float(opt.normalization_high_quantile, "normalization_high_quantile")

    for name in (
        "tau_e",
        "tau_s",
        "omega_boundary",
        "omega_turning",
        "omega_defect",
        "lambda_redundancy",
        "normalization_low_quantile",
        "normalization_high_quantile",
    ):
        if not math.isfinite(getattr(opt, name)):
            raise ValueError(f"{name} must be finite.")
    if opt.tau_e <= 0:
        raise ValueError("tau_e must be positive.")
    if opt.tau_s <= 0:
        raise ValueError("tau_s must be positive.")
    if opt.omega_boundary < 0:
        raise ValueError("omega_boundary must be non-negative.")
    if opt.omega_turning < 0:
        raise ValueError("omega_turning must be non-negative.")
    if opt.omega_defect < 0:
        raise ValueError("omega_defect must be non-negative.")
    if opt.lambda_redundancy < 0:
        raise ValueError("lambda_redundancy must be non-negative.")
    if not (0.0 <= opt.normalization_low_quantile < opt.normalization_high_quantile <= 1.0):
        raise ValueError("Require 0 <= normalization_low_quantile < normalization_high_quantile <= 1.")
    return mode


def compute_effective_densification_budget(budget, max_gaussians, current_gaussian_count):
    if budget < 0:
        raise ValueError("densification budget must be non-negative.")
    effective_budget = int(budget)
    if max_gaussians is not None:
        effective_budget = min(effective_budget, int(max_gaussians) - int(current_gaussian_count))
    return max(0, effective_budget)


def should_run_densification(iteration, densify_from_iter, densify_until_iter, densification_interval):
    if densification_interval <= 0:
        raise ValueError("densification_interval must be positive.")
    return (
        iteration > densify_from_iter
        and iteration < densify_until_iter
        and iteration % densification_interval == 0
    )


def _to_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value.") from exc
