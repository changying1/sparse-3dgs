import pytest
import torch

from utils.value_allocation import compute_observation_scarcity


def test_zero_visible_count_has_zero_observation_scarcity():
    counts = torch.tensor([0], dtype=torch.long)

    scarcity = compute_observation_scarcity(counts, tau_e=2.0, tau_s=10.0)

    assert scarcity.shape == (1,)
    assert torch.isfinite(scarcity).all()
    assert scarcity.item() == 0.0


def test_observation_scarcity_rises_then_falls():
    counts = torch.tensor([0, 1, 4, 20], dtype=torch.long)

    scarcity = compute_observation_scarcity(counts, tau_e=2.0, tau_s=10.0)

    assert torch.isfinite(scarcity).all()
    assert scarcity[0] == 0.0
    assert scarcity[2] > scarcity[1]
    assert scarcity[2] > scarcity[3]


def test_observation_scarcity_is_finite_and_non_negative():
    counts = torch.tensor([0, 1, 2, 8, 32], dtype=torch.long)

    scarcity = compute_observation_scarcity(counts, tau_e=2.0, tau_s=10.0)

    assert torch.isfinite(scarcity).all()
    assert (scarcity >= 0).all()


def test_observation_scarcity_rejects_non_positive_tau():
    counts = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError):
        compute_observation_scarcity(counts, tau_e=0.0, tau_s=10.0)
    with pytest.raises(ValueError):
        compute_observation_scarcity(counts, tau_e=2.0, tau_s=-1.0)
