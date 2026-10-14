import argparse
import io
import types
from unittest import mock

import pytest
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.i2_lifecycle import (
    BIRTH_TYPE_TO_CODE,
    GaussianLifecycle,
    ROOT_SOURCE_UID,
    add_i2_lifecycle_arguments,
)
from utils.oe_densification import map_dropout_visibility_to_full_mask


def _values(tensor):
    return tensor.tolist()


def _make_model(count=3, *, tracking=False):
    model = GaussianModel(1)
    model._xyz = nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 3))
    model._features_dc = nn.Parameter(torch.zeros(count, 1, 3))
    model._features_rest = nn.Parameter(torch.zeros(count, 3, 3))
    model._scaling = nn.Parameter(torch.zeros(count, 3))
    model._rotation = nn.Parameter(torch.zeros(count, 4))
    model._opacity = nn.Parameter(torch.zeros(count, 1))
    model.max_radii2D = torch.zeros(count)
    model.xyz_gradient_accum = torch.zeros(count, 1)
    model.denom = torch.ones(count, 1)
    model.spatial_lr_scale = 1.0
    groups = [
        {"params": [model._xyz], "name": "xyz"},
        {"params": [model._features_dc], "name": "f_dc"},
        {"params": [model._features_rest], "name": "f_rest"},
        {"params": [model._opacity], "name": "opacity"},
        {"params": [model._scaling], "name": "scaling"},
        {"params": [model._rotation], "name": "rotation"},
    ]
    model.optimizer = torch.optim.Adam(groups, lr=0.0)
    if tracking:
        model.enable_lifecycle_tracking()
    return model


def _cpu_training_setup(self, _training_args):
    groups = [
        {"params": [self._xyz], "name": "xyz"},
        {"params": [self._features_dc], "name": "f_dc"},
        {"params": [self._features_rest], "name": "f_rest"},
        {"params": [self._opacity], "name": "opacity"},
        {"params": [self._scaling], "name": "scaling"},
        {"params": [self._rotation], "name": "rotation"},
    ]
    self.optimizer = torch.optim.Adam(groups, lr=0.0)


def test_cli_is_disabled_by_default_and_can_be_enabled():
    parser = argparse.ArgumentParser()
    add_i2_lifecycle_arguments(parser)
    assert parser.parse_args([]).enable_i2_lifecycle_tracking is False
    assert parser.parse_args(
        ["--enable_i2_lifecycle_tracking"]
    ).enable_i2_lifecycle_tracking is True


def test_root_uids_are_unique_ordered_and_have_root_metadata():
    state = GaussianLifecycle.roots(4, torch.device("cpu"))
    assert _values(state.uid) == [0, 1, 2, 3]
    assert torch.unique(state.uid).numel() == 4
    assert _values(state.birth_iter) == [0, 0, 0, 0]
    assert _values(state.birth_type) == [BIRTH_TYPE_TO_CODE["root"]] * 4
    assert _values(state.source_uid) == [ROOT_SOURCE_UID] * 4
    assert _values(state.generation) == [0, 0, 0, 0]
    assert state.next_uid == 4


def test_clone_children_get_new_uid_parent_and_generation_without_parent_death():
    state = GaussianLifecycle.roots(3, torch.device("cpu"))
    children = state.append_children(
        torch.tensor([True, False, True]), "clone", iteration=100
    )
    assert _values(children) == [3, 4]
    assert _values(state.uid) == [0, 1, 2, 3, 4]
    assert _values(state.source_uid[-2:]) == [0, 2]
    assert _values(state.generation[-2:]) == [1, 1]
    assert state.events[-1]["birth_type"] == "clone"
    assert not any(event["event_type"] == "death" for event in state.events)


def test_split_children_follow_parameter_repeat_order_and_parents_are_replaced():
    state = GaussianLifecycle.roots(4, torch.device("cpu"))
    selected = torch.tensor([False, True, False, True])
    children = state.append_children(selected, "split", iteration=200, repeats=2)
    # torch rows use selected_rows.repeat(N, ...): [parent1, parent3, parent1, parent3].
    assert _values(children) == [4, 5, 6, 7]
    assert _values(state.source_uid[-4:]) == [1, 3, 1, 3]
    assert _values(state.generation[-4:]) == [1, 1, 1, 1]

    split_prune = torch.cat((selected, torch.zeros(4, dtype=torch.bool)))
    dead = state.prune(split_prune, "split_replaced", iteration=200)
    assert _values(dead) == [1, 3]
    assert _values(state.uid) == [0, 2, 4, 5, 6, 7]
    assert state.events[-1]["reason"] == "split_replaced"
    assert not any(
        event.get("reason") == "training_prune" for event in state.events
    )


def test_clone_split_prune_sequence_stays_aligned_and_auditable():
    state = GaussianLifecycle.roots(4, torch.device("cpu"))
    state.append_children(
        torch.tensor([True, False, False, False]), "clone", iteration=300
    )
    split_parents = torch.tensor([False, True, False, False, True])
    state.append_children(split_parents, "split", iteration=300, repeats=2)
    state.prune(
        torch.cat((split_parents, torch.zeros(4, dtype=torch.bool))),
        "split_replaced",
        iteration=300,
    )
    # Surviving rows are [uid0, uid2, uid3, uid5, uid6, uid7, uid8].
    training_prune = torch.tensor([False, True, False, False, True, False, False])
    dead = state.prune(training_prune, "training_prune", iteration=300)

    assert _values(dead) == [2, 6]
    assert _values(state.uid) == [0, 3, 5, 7, 8]
    assert len(state.uid) == len(state.birth_iter) == len(state.source_uid)
    assert len(state.uid) == len(state.birth_type) == len(state.generation)
    assert [event["sequence"] for event in state.events] == list(
        range(len(state.events))
    )
    assert [event.get("birth_type") or event.get("reason") for event in state.events] == [
        "root",
        "clone",
        "split",
        "split_replaced",
        "training_prune",
    ]


def test_prune_synchronously_crops_every_live_metadata_array():
    state = GaussianLifecycle.roots(5, torch.device("cpu"))
    state.birth_iter = torch.arange(5)
    state.prune(
        torch.tensor([True, False, True, False, False]),
        "training_prune",
        iteration=9,
    )
    assert _values(state.uid) == [1, 3, 4]
    assert _values(state.birth_iter) == [1, 3, 4]
    assert _values(state.generation) == [0, 0, 0]
    assert _values(state.source_uid) == [ROOT_SOURCE_UID] * 3


def test_model_prune_applies_the_identical_mask_to_parameters_and_lifecycle():
    model = _make_model(4, tracking=True)
    model.prune_points(
        torch.tensor([False, True, False, True]),
        lifecycle_reason="training_prune",
        iteration=7,
    )
    assert model.get_xyz.shape[0] == 2
    assert _values(model.lifecycle.uid) == [0, 2]
    assert _values(model.lifecycle.events[-1]["uids"]) == [1, 3]
    assert model.lifecycle.events[-1]["reason"] == "training_prune"


def test_deleted_uids_are_never_reused():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.prune(torch.tensor([False, True]), "training_prune", iteration=1)
    child = state.append_children(torch.tensor([True]), "clone", iteration=2)
    assert _values(child) == [2]
    assert state.next_uid == 3


def test_tracking_off_capture_keeps_exact_legacy_twelve_tuple():
    model = _make_model(tracking=False)
    checkpoint = model.capture()
    assert isinstance(checkpoint, tuple)
    assert len(checkpoint) == 12
    assert model.lifecycle is None


def test_tracking_checkpoint_round_trip_preserves_full_state_and_counter():
    source = _make_model(tracking=True)
    source.lifecycle.append_children(
        torch.tensor([True, False, True]), "clone", iteration=11
    )
    source.lifecycle.prune(
        torch.tensor([False, True, False, False, False]),
        "training_prune",
        iteration=12,
    )
    # Keep model parameter rows aligned with the lifecycle rows for capture.
    rows = torch.tensor([0, 2, 0, 2])
    source._xyz = nn.Parameter(source._xyz[rows])
    source._features_dc = nn.Parameter(source._features_dc[rows])
    source._features_rest = nn.Parameter(source._features_rest[rows])
    source._scaling = nn.Parameter(source._scaling[rows])
    source._rotation = nn.Parameter(source._rotation[rows])
    source._opacity = nn.Parameter(source._opacity[rows])
    source.max_radii2D = source.max_radii2D[rows]
    source.xyz_gradient_accum = source.xyz_gradient_accum[rows]
    source.denom = source.denom[rows]
    _cpu_training_setup(source, None)

    checkpoint = source.capture()
    assert len(checkpoint) == 13
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    target = _make_model(0, tracking=True)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(loaded, object())
    assert _values(target.lifecycle.uid) == _values(source.lifecycle.uid)
    assert _values(target.lifecycle.source_uid) == _values(source.lifecycle.source_uid)
    assert _values(target.lifecycle.generation) == _values(source.lifecycle.generation)
    assert target.lifecycle.next_uid == source.lifecycle.next_uid
    assert len(target.lifecycle.events) == len(source.lifecycle.events)


def test_tracking_enabled_rejects_legacy_checkpoint_explicitly():
    legacy_checkpoint = _make_model(tracking=False).capture()
    target = _make_model(0, tracking=True)
    with pytest.raises(ValueError, match="legacy checkpoint without lifecycle metadata"):
        target.restore(legacy_checkpoint, object())


def test_tracking_disabled_restores_legacy_checkpoint():
    legacy_checkpoint = _make_model(tracking=False).capture()
    target = _make_model(0, tracking=False)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(legacy_checkpoint, object())
    assert target.lifecycle is None
    assert target.get_xyz.shape[0] == 3


def test_tracking_disabled_can_restore_i2_checkpoint_without_allocating_metadata():
    checkpoint = _make_model(tracking=True).capture()
    target = _make_model(0, tracking=False)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(checkpoint, object())
    assert target.lifecycle is None
    assert target.get_xyz.shape[0] == 3


def test_empty_clone_split_and_prune_are_noops():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    event_count = len(state.events)
    next_uid = state.next_uid
    assert state.append_children(
        torch.tensor([False, False]), "clone", iteration=1
    ).numel() == 0
    assert state.append_children(
        torch.tensor([False, False]), "split", iteration=1, repeats=2
    ).numel() == 0
    assert state.prune(
        torch.tensor([False, False]), "training_prune", iteration=1
    ).numel() == 0
    assert _values(state.uid) == [0, 1]
    assert state.next_uid == next_uid
    assert len(state.events) == event_count


def test_observation_interface_rejects_shifted_views_and_requires_full_mask():
    state = GaussianLifecycle.roots(3, torch.device("cpu"))
    full_mask = torch.tensor([True, False, True])
    state.observation_update_interface(full_mask, 5, is_real_view=True)
    with pytest.raises(ValueError, match="real training view"):
        state.observation_update_interface(full_mask, 5, is_real_view=False)
    with pytest.raises(ValueError, match="length"):
        state.observation_update_interface(
            torch.tensor([True, False]), 5, is_real_view=True
        )


def test_dropout_excluded_gaussians_are_not_marked_observed_and_i1_mapping_is_unchanged():
    dropout_mask = torch.tensor([True, False, True, True, False])
    subset_visibility = torch.tensor([False, True, False])
    viewspace = torch.zeros((5, 3))
    full_mask = map_dropout_visibility_to_full_mask(
        dropout_mask, subset_visibility, viewspace
    )
    assert _values(full_mask) == [False, False, True, False, False]
    state = GaussianLifecycle.roots(5, torch.device("cpu"))
    state.observation_update_interface(full_mask, 8, is_real_view=True)


def test_checkpoint_validation_rejects_misalignment_and_unsafe_uid_counter():
    state = GaussianLifecycle.roots(3, torch.device("cpu")).state_dict()
    state["generation"] = state["generation"][:-1]
    with pytest.raises(ValueError, match="row-aligned"):
        GaussianLifecycle.from_state_dict(
            state, expected_count=3, device=torch.device("cpu")
        )

    state = GaussianLifecycle.roots(3, torch.device("cpu")).state_dict()
    state["next_uid"] = 2
    with pytest.raises(ValueError, match="allocated range|birth history"):
        GaussianLifecycle.from_state_dict(
            state, expected_count=3, device=torch.device("cpu")
        )


def test_checkpoint_restore_then_clone_keeps_uid_counter_monotonic():
    source = GaussianLifecycle.roots(3, torch.device("cpu"))
    source.append_children(torch.tensor([True, False, False]), "clone", iteration=4)
    restored = GaussianLifecycle.from_state_dict(
        source.state_dict(), expected_count=4, device=torch.device("cpu")
    )
    previous_next_uid = restored.next_uid
    child = restored.append_children(
        torch.tensor([False, True, False, False]), "clone", iteration=8
    )
    assert child.tolist() == [previous_next_uid]
    assert restored.next_uid == previous_next_uid + 1
    restored.audit_strict()


def test_checkpoint_restore_then_split_and_prune_preserves_lineage():
    source = GaussianLifecycle.roots(3, torch.device("cpu"))
    restored = GaussianLifecycle.from_state_dict(
        source.state_dict(), expected_count=3, device=torch.device("cpu")
    )
    selected = torch.tensor([False, True, False])
    children = restored.append_children(selected, "split", iteration=5, repeats=2)
    assert children.tolist() == [3, 4]
    assert restored.source_uid[-2:].tolist() == [1, 1]
    assert restored.generation[-2:].tolist() == [1, 1]
    restored.prune(
        torch.cat((selected, torch.zeros(2, dtype=torch.bool))),
        "split_replaced",
        iteration=5,
    )
    restored.prune(
        torch.tensor([False, False, True, False]),
        "training_prune",
        iteration=6,
    )
    assert restored.uid.tolist() == [0, 2, 4]
    assert restored.events[-2]["reason"] == "split_replaced"
    assert restored.events[-1]["reason"] == "training_prune"
    restored.audit_strict()


def test_strict_audit_rejects_live_uid_in_death_history():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.events.append(
        {
            "sequence": 1,
            "iteration": 1,
            "event_type": "death",
            "reason": "training_prune",
            "uids": torch.tensor([0], dtype=torch.int64),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        state.audit_strict()


@pytest.mark.parametrize("corruption", ["duplicate", "missing"])
def test_strict_audit_rejects_duplicate_or_missing_birth(corruption):
    state = GaussianLifecycle.roots(3, torch.device("cpu"))
    if corruption == "duplicate":
        state.events.append(
            {
                "sequence": 1,
                "iteration": 0,
                "event_type": "birth",
                "birth_type": "root",
                "uids": torch.tensor([0], dtype=torch.int64),
                "source_uids": torch.tensor([ROOT_SOURCE_UID], dtype=torch.int64),
            }
        )
    else:
        state.events[0]["uids"] = state.events[0]["uids"][1:]
        state.events[0]["source_uids"] = state.events[0]["source_uids"][1:]
    with pytest.raises(ValueError, match="duplicate|missing"):
        state.audit_strict()


def test_strict_audit_rejects_generation_inconsistent_with_parent_history():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.append_children(torch.tensor([True, False]), "clone", iteration=2)
    state.generation[-1] = 7
    with pytest.raises(ValueError, match="generation"):
        state.audit_strict()


def test_strict_audit_rejects_event_iteration_rollback():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.append_children(torch.tensor([True, False]), "clone", iteration=5)
    state.append_children(
        torch.tensor([False, True, False]), "clone", iteration=4
    )
    with pytest.raises(ValueError, match="backwards"):
        state.audit_strict()


def test_strict_audit_rejects_child_born_after_parent_death():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.prune(
        torch.tensor([True, False]), "training_prune", iteration=1
    )
    state.events.append(
        {
            "sequence": 2,
            "iteration": 2,
            "event_type": "birth",
            "birth_type": "clone",
            "uids": torch.tensor([2], dtype=torch.int64),
            "source_uids": torch.tensor([0], dtype=torch.int64),
        }
    )
    state.uid = torch.tensor([1, 2], dtype=torch.int64)
    state.birth_iter = torch.tensor([0, 2], dtype=torch.int64)
    state.birth_type = torch.tensor(
        [BIRTH_TYPE_TO_CODE["root"], BIRTH_TYPE_TO_CODE["clone"]],
        dtype=torch.uint8,
    )
    state.source_uid = torch.tensor([ROOT_SOURCE_UID, 0], dtype=torch.int64)
    state.generation = torch.tensor([0, 1], dtype=torch.int64)
    state.next_uid = 3

    with pytest.raises(ValueError, match="parent that was already dead"):
        state.audit_strict()


def test_strict_audit_accepts_split_children_before_parent_replacement():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    selected = torch.tensor([True, False])
    state.append_children(selected, "split", iteration=3, repeats=2)
    state.prune(
        torch.cat((selected, torch.zeros(2, dtype=torch.bool))),
        "split_replaced",
        iteration=3,
    )
    state.audit_strict()
    assert state.uid.tolist() == [1, 2, 3]


def test_strict_audit_rejects_split_replacement_without_split_child():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.prune(torch.tensor([True, False]), "split_replaced", iteration=3)
    with pytest.raises(ValueError, match="no preceding split child"):
        state.audit_strict()


def test_training_prune_does_not_require_split_children():
    state = GaussianLifecycle.roots(2, torch.device("cpu"))
    state.prune(torch.tensor([True, False]), "training_prune", iteration=3)
    state.audit_strict()
    assert state.uid.tolist() == [1]


def test_cpu_checkpoint_round_trip_preserves_real_adam_state():
    source = _make_model(3, tracking=True)
    for group in source.optimizer.param_groups:
        group["lr"] = 0.01
    loss = sum(group["params"][0].sum() for group in source.optimizer.param_groups)
    loss.backward()
    source.optimizer.step()
    checkpoint = source.capture()

    target = _make_model(0, tracking=True)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(checkpoint, object())

    for source_group, target_group in zip(
        source.optimizer.param_groups, target.optimizer.param_groups
    ):
        source_state = source.optimizer.state[source_group["params"][0]]
        target_state = target.optimizer.state[target_group["params"][0]]
        assert source_group["name"] == target_group["name"]
        assert source_group["lr"] == target_group["lr"]
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            torch.testing.assert_close(source_state[key], target_state[key])


def test_tracking_disabled_never_calls_lifecycle_validation():
    source = _make_model(2, tracking=False)
    target = _make_model(0, tracking=False)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    with mock.patch.object(
        GaussianLifecycle, "validate_lightweight", side_effect=AssertionError
    ), mock.patch.object(GaussianLifecycle, "audit_strict", side_effect=AssertionError):
        checkpoint = source.capture()
        target.restore(checkpoint, object())
    assert len(checkpoint) == 12
    assert target.lifecycle is None


def test_hot_path_uses_lightweight_validation_only():
    state = GaussianLifecycle.roots(3, torch.device("cpu"))
    with mock.patch.object(state, "audit_strict", side_effect=AssertionError):
        state.append_children(torch.tensor([True, False, False]), "clone", iteration=1)
        state.prune(
            torch.tensor([False, True, False, False]),
            "training_prune",
            iteration=2,
        )
    state.audit_strict()


def test_lifecycle_statistics_report_counts_and_event_memory():
    state = GaussianLifecycle.roots(3, torch.device("cpu"))
    state.append_children(torch.tensor([True, False, False]), "clone", iteration=1)
    state.prune(
        torch.tensor([False, True, False, False]),
        "training_prune",
        iteration=2,
    )
    stats = state.statistics()
    assert stats["live_gaussian_count"] == 3
    assert stats["allocated_uid_count"] == 4
    assert stats["event_count"] == 3
    assert stats["birth_event_count"] == 2
    assert stats["death_event_count"] == 1
    assert stats["history_uid_entry_count"] == 5
    assert stats["event_tensor_bytes"] > 0
    assert stats["estimated_event_bytes"] >= stats["event_tensor_bytes"]
