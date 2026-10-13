import argparse
import json
from unittest import mock

import pytest
import torch

from utils.evidence_structural_loss import add_oe_structural_arguments
from utils.oe_densification import (
    add_densification_stats_with_baseline_gradient,
    effective_densify_until_iter,
    extract_baseline_viewspace_gradient,
    map_dropout_visibility_to_full_mask,
    maybe_extract_baseline_viewspace_gradient,
    should_preserve_baseline_densification_stats,
)
from utils.oe_training import I1Diagnostics


class RecordingGaussians:
    def __init__(self, *, error=None):
        self.error = error
        self.gradient_seen = None
        self.mask_seen = None

    def add_densification_stats(self, viewspace, update_filter):
        self.gradient_seen = None if viewspace.grad is None else viewspace.grad.clone()
        self.mask_seen = update_filter.clone()
        if self.error is not None:
            raise self.error


def _toy_losses():
    gaussian_parameter = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    gaussian_parameter.requires_grad_(True)
    viewspace = gaussian_parameter * 2.0
    viewspace.retain_grad()
    baseline_loss = viewspace.sum()
    i1_loss = 3.0 * viewspace.sum()
    return gaussian_parameter, viewspace, baseline_loss, i1_loss


def test_cli_defaults_off_and_can_enable():
    parser = argparse.ArgumentParser()
    add_oe_structural_arguments(parser)
    assert parser.parse_args([]).oe_preserve_baseline_densification_stats is False
    assert parser.parse_args(
        ["--oe_preserve_baseline_densification_stats"]
    ).oe_preserve_baseline_densification_stats is True


@pytest.mark.parametrize(
    "iteration,densify_from,current_until,total_iterations,expected",
    [
        (1000, 1000, 15000, 30000, 15000),
        (1001, 1000, 15000, 30000, 30000),
        (16001, 1000, 15000, 30000, 30000),
    ],
)
def test_effective_cutoff_matches_same_iteration_training_mutation(
    iteration, densify_from, current_until, total_iterations, expected
):
    assert effective_densify_until_iter(
        iteration,
        densify_from,
        current_until,
        total_iterations,
    ) == expected


def test_resumed_iteration_extracts_when_same_iteration_will_collect_stats():
    _, viewspace, baseline_loss, _ = _toy_losses()
    effective_until = effective_densify_until_iter(16001, 1000, 15000, 30000)
    gradient = maybe_extract_baseline_viewspace_gradient(
        baseline_loss,
        viewspace,
        preserve_flag=True,
        oe_loss_active=True,
        iteration=16001,
        densify_until_iter=effective_until,
    )
    assert torch.equal(gradient, torch.ones_like(viewspace))


@pytest.mark.parametrize("iteration", [1000, 1001, 14999, 15000, 16001, 29999, 30000])
def test_protection_gate_matches_actual_same_iteration_stats_branch(iteration):
    densify_from = 1000
    current_until = 15000
    total_iterations = 30000
    effective_until = effective_densify_until_iter(
        iteration, densify_from, current_until, total_iterations
    )
    protection_active = should_preserve_baseline_densification_stats(
        True, True, iteration, effective_until
    )

    if iteration > densify_from:
        current_until = total_iterations
    stats_branch_active = iteration < current_until

    assert protection_active is stats_branch_active


@pytest.mark.parametrize(
    "preserve_flag,oe_active,iteration,densify_until,expected",
    [
        (False, False, 5, 10, False),
        (False, True, 5, 10, False),
        (True, False, 5, 10, False),
        (True, True, 5, 10, True),
        (True, True, 10, 10, False),
    ],
)
def test_protection_gate_requires_flag_i1_activation_and_densification_window(
    preserve_flag, oe_active, iteration, densify_until, expected
):
    assert should_preserve_baseline_densification_stats(
        preserve_flag, oe_active, iteration, densify_until
    ) is expected


@pytest.mark.parametrize(
    "preserve_flag,oe_active,iteration",
    [
        (False, False, 5),  # I1 disabled: baseline path is untouched.
        (False, True, 5),  # I1 active, protection disabled: joint-gradient path remains.
        (True, False, 5),  # I1 is outside its activation interval.
        (True, True, 10),  # No statistics are accumulated at densify_until_iter.
    ],
)
def test_closed_gate_does_not_extract_an_extra_gradient(preserve_flag, oe_active, iteration):
    _, viewspace, baseline_loss, _ = _toy_losses()
    with mock.patch("utils.oe_densification.torch.autograd.grad") as autograd_grad:
        result = maybe_extract_baseline_viewspace_gradient(
            baseline_loss,
            viewspace,
            preserve_flag=preserve_flag,
            oe_loss_active=oe_active,
            iteration=iteration,
            densify_until_iter=10,
        )
    assert result is None
    autograd_grad.assert_not_called()


def test_active_protection_uses_baseline_stats_but_joint_parameter_gradient():
    parameter, viewspace, baseline_loss, i1_loss = _toy_losses()
    baseline_gradient = maybe_extract_baseline_viewspace_gradient(
        baseline_loss,
        viewspace,
        preserve_flag=True,
        oe_loss_active=True,
        iteration=5,
        densify_until_iter=10,
    )
    (baseline_loss + i1_loss).backward()
    joint_gradient = viewspace.grad.clone()
    model = RecordingGaussians()
    visible = torch.tensor([True, False])

    add_densification_stats_with_baseline_gradient(
        model, viewspace, visible, baseline_gradient
    )

    assert torch.equal(model.gradient_seen, torch.ones_like(viewspace))
    assert torch.equal(viewspace.grad, joint_gradient)
    assert torch.equal(joint_gradient, torch.full_like(viewspace, 4.0))
    assert torch.equal(parameter.grad, torch.full_like(parameter, 8.0))


def test_unprotected_i1_stats_use_existing_joint_gradient():
    _, viewspace, baseline_loss, i1_loss = _toy_losses()
    (baseline_loss + i1_loss).backward()
    model = RecordingGaussians()
    visible = torch.tensor([True, True])

    model.add_densification_stats(viewspace, visible)

    assert torch.equal(model.gradient_seen, torch.full_like(viewspace, 4.0))


@pytest.mark.parametrize(
    "dropout_mask,visibility_filter,expected",
    [
        ([True, True, True, True, True], [True, False, True, False, True], [0, 2, 4]),
        ([True, False, True, True, True], [False, True, False, True], [2, 4]),
    ],
    ids=["dropout_0", "dropout_0_2"],
)
def test_dropout_subset_visibility_maps_to_full_gaussian_indices(
    dropout_mask, visibility_filter, expected
):
    viewspace = torch.zeros((5, 3), requires_grad=True)
    combined = map_dropout_visibility_to_full_mask(
        torch.tensor(dropout_mask),
        torch.tensor(visibility_filter),
        viewspace,
    )
    legacy_mask = torch.zeros(len(dropout_mask), dtype=torch.bool)
    retained = torch.nonzero(torch.tensor(dropout_mask), as_tuple=True)[0]
    legacy_mask[retained[torch.tensor(visibility_filter)]] = True
    assert torch.equal(combined, legacy_mask)
    assert torch.nonzero(combined, as_tuple=True)[0].tolist() == expected


def test_dropout_visibility_length_mismatch_fails_explicitly():
    with pytest.raises(ValueError, match="retained Dropout subset"):
        map_dropout_visibility_to_full_mask(
            torch.tensor([True, False, True]),
            torch.tensor([True, False, True]),
            torch.zeros((3, 3)),
        )


def test_disconnected_baseline_gradient_fails_explicitly():
    viewspace = torch.zeros((2, 3), requires_grad=True)
    unrelated = torch.ones((), requires_grad=True)
    with pytest.raises(RuntimeError, match="not connected"):
        extract_baseline_viewspace_gradient(unrelated, viewspace)


@pytest.mark.parametrize(
    "bad_gradient,error_type,message",
    [
        (None, ValueError, "required"),
        (torch.ones((2, 2)), ValueError, "shape"),
        (torch.tensor([[1.0, 1.0, 1.0], [1.0, float("nan"), 1.0]]), FloatingPointError, "non-finite"),
    ],
)
def test_invalid_baseline_gradient_fails_explicitly(bad_gradient, error_type, message):
    viewspace = torch.zeros((2, 3), requires_grad=True)
    viewspace.grad = torch.full_like(viewspace, 4.0)
    with pytest.raises(error_type, match=message):
        add_densification_stats_with_baseline_gradient(
            RecordingGaussians(),
            viewspace,
            torch.tensor([True, False]),
            bad_gradient,
        )


def test_original_joint_gradient_is_restored_when_stats_raise():
    viewspace = torch.zeros((2, 3), requires_grad=True)
    joint_gradient = torch.full_like(viewspace, 4.0)
    viewspace.grad = joint_gradient
    model = RecordingGaussians(error=RuntimeError("stats failed"))

    with pytest.raises(RuntimeError, match="stats failed"):
        add_densification_stats_with_baseline_gradient(
            model,
            viewspace,
            torch.tensor([True, False]),
            torch.ones_like(viewspace),
        )

    assert model.gradient_seen is not None
    assert torch.equal(model.gradient_seen, torch.ones_like(viewspace))
    assert viewspace.grad is joint_gradient


def test_gradient_diagnostics_are_low_frequency_and_visible_only():
    logs = []
    diagnostics = I1Diagnostics({}, (1, 20), interval=10, emit=logs.append)
    baseline = torch.tensor([[3.0, 4.0, 9.0], [0.0, 2.0, 9.0]])
    joint = torch.tensor([[6.0, 8.0, 9.0], [0.0, 4.0, 9.0]])
    visible = torch.tensor([False, True])

    with mock.patch("utils.oe_training.torch.norm", wraps=torch.norm) as norm:
        assert diagnostics.record_densification_gradients(
            1, baseline, joint, visible, protection_effective=True
        ) is None
        norm.assert_not_called()

    payload = diagnostics.record_densification_gradients(
        10, baseline, joint, visible, protection_effective=True
    )
    emitted = json.loads(logs[-1].removeprefix("[I1] "))
    assert emitted == payload
    assert payload["protection_effective"] is True
    assert payload["baseline_gradient_mean"] == 3.5
    assert payload["joint_gradient_mean"] == 7.0
    assert payload["visible_baseline_gradient_mean"] == 2.0
    assert payload["visible_joint_gradient_mean"] == 4.0
    assert payload["visible_gaussian_count"] == 1
