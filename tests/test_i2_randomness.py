import random

import torch

from tests.test_i2_active_pilot import _plan, _real_cpu_model
from utils.i2_active_pilot import (
    apply_active_reallocation,
    apply_release_only,
    load_active_plan,
)
from utils.i2_randomness import ReleaseOnlyDropoutReference, aligned_uniform_prefix


def test_release_only_dropout_prefix_and_rng_state_match_reallocation():
    torch.manual_seed(0)
    full_prefix, _ = aligned_uniform_prefix(40, None, device="cpu")
    full_next = torch.rand(8)

    torch.manual_seed(0)
    release_prefix, release_reference = aligned_uniform_prefix(20, 40, device="cpu")
    release_next = torch.rand(8)

    assert torch.equal(release_prefix, full_prefix[:20])
    assert release_reference.shape == (40,)
    assert torch.equal(release_next, full_next)


def test_intervention_free_python_camera_sequence_is_repeatable():
    cameras = list(range(3))
    random.seed(0)
    first = [random.choice(cameras) for _ in range(50)]
    random.seed(0)
    second = [random.choice(cameras) for _ in range(50)]
    assert first == second


def test_invalid_dropout_reference_count_fails_closed():
    try:
        aligned_uniform_prefix(21, 20, device="cpu")
    except ValueError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("undersized reference count was accepted")


def test_active_interventions_do_not_advance_python_or_torch_rng(tmp_path):
    path, checkpoint = _plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    for name, intervention in (
        ("reallocation", apply_active_reallocation),
        ("release", apply_release_only),
    ):
        random.seed(37)
        torch.manual_seed(41)
        python_before = random.getstate()
        torch_before = torch.get_rng_state().clone()
        intervention(
            _real_cpu_model(), plan, iteration=1201,
            event_log=tmp_path / f"{name}.json",
        )
        assert random.getstate() == python_before
        assert torch.equal(torch.get_rng_state(), torch_before)


def test_release_reference_below_initial_count_uses_frozen_length():
    reference = ReleaseOnlyDropoutReference(40)
    assert reference.resolve(20, 1202) == 40
    assert reference.first_extension_iteration is None


def test_release_reference_equal_to_initial_count_uses_frozen_length():
    reference = ReleaseOnlyDropoutReference(40)
    assert reference.resolve(40, 1300) == 40
    assert reference.first_extension_iteration is None


def test_release_reference_growth_above_initial_count_extends_safely():
    reference = ReleaseOnlyDropoutReference(40)
    assert reference.resolve(41, 1400) == 41
    assert reference.first_extension_iteration == 1400
    assert reference.first_extension_gaussian_count == 41


def test_release_reference_multiple_growth_keeps_first_extension_record():
    reference = ReleaseOnlyDropoutReference(40)
    assert reference.resolve(39, 1202) == 40
    assert reference.resolve(41, 1500) == 41
    assert reference.resolve(45, 1600) == 45
    assert reference.resolve(43, 1700) == 43
    assert reference.first_extension_iteration == 1500
    assert reference.first_extension_gaussian_count == 41
