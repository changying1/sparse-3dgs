import argparse
import inspect
import io
import random
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

from llff_view_protocol import build_view_manifest, write_view_manifest
from scene.gaussian_model import GaussianModel
from utils.i2_lifecycle import GaussianLifecycle
from utils.i2_observation import (
    GaussianObservation,
    add_i2_observation_arguments,
    build_real_view_camera_mapping,
    validate_i2_observation_configuration,
)
from utils.oe_densification import map_dropout_visibility_to_full_mask


VIEW_IDS = ("image_001.png", "image_013.png", "image_024.png")


def _make_model(count=3, *, lifecycle=False, observation=False):
    model = GaussianModel(1)
    model._xyz = nn.Parameter(
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    )
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
    if lifecycle:
        model.enable_lifecycle_tracking()
    if observation:
        model.enable_observation_tracking(VIEW_IDS)
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


def _record(state, dropout, supported, view, iteration):
    state.record_real_view(
        torch.tensor(dropout, dtype=torch.bool),
        torch.tensor(supported, dtype=torch.bool),
        view,
        iteration,
        is_real_view=True,
    )


def test_cli_default_and_lifecycle_dependency():
    parser = argparse.ArgumentParser()
    add_i2_observation_arguments(parser)
    assert parser.parse_args([]).enable_i2_observation_tracking is False
    assert parser.parse_args(
        ["--enable_i2_observation_tracking"]
    ).enable_i2_observation_tracking is True
    validate_i2_observation_configuration(False, False)
    validate_i2_observation_configuration(True, True)
    with pytest.raises(ValueError, match="requires --enable_i2_lifecycle_tracking"):
        validate_i2_observation_configuration(True, False)
    with pytest.raises(ValueError, match="requires lifecycle tracking"):
        _make_model(lifecycle=False).enable_observation_tracking(VIEW_IDS)


def test_three_layer_counts_and_no_dropout_case():
    state = GaussianObservation.zeros(4, torch.device("cpu"), VIEW_IDS)
    _record(state, [True, False, True, True], [True, False, False, True], 0, 1)
    _record(state, [True, True, True, True], [False, True, True, False], 1, 2)
    assert state.real_view_opportunities.tolist() == [2, 2, 2, 2]
    assert state.dropout_retained_hits.tolist() == [2, 1, 2, 2]
    assert state.screen_supported_hits.tolist() == [1, 1, 1, 1]
    state.audit_strict(expected_uid=torch.arange(4, dtype=torch.int64))


def test_dropout_subset_mapping_updates_exact_full_rows():
    dropout = torch.tensor([True, False, True, True, False])
    subset_visibility = torch.tensor([False, True, False])
    full = map_dropout_visibility_to_full_mask(
        dropout, subset_visibility, torch.zeros(5, 3)
    )
    state = GaussianObservation.zeros(5, torch.device("cpu"), VIEW_IDS)
    state.record_real_view(dropout, full, 0, 1, is_real_view=True)
    assert full.tolist() == [False, False, True, False, False]
    assert state.dropout_retained_hits.tolist() == [1, 0, 1, 1, 0]
    assert state.screen_supported_hits.tolist() == [0, 0, 1, 0, 0]


def test_illegal_support_is_filtered_and_remains_strictly_auditable():
    state = GaussianObservation.zeros(1, torch.device("cpu"), VIEW_IDS)
    _record(state, [True], [False], 0, 1)
    _record(state, [False], [True], 1, 2)

    # The illegal second support must not enter either official statistic.
    assert state.dropout_retained_hits.tolist() == [1]
    assert state.screen_supported_hits.tolist() == [0]
    assert state.screen_supported_view_history.tolist() == [[False, False, False]]
    with pytest.raises(ValueError, match="outside the Dropout retained set"):
        state.audit_strict(expected_uid=torch.arange(1, dtype=torch.int64))


def test_valid_dropout_mapping_does_not_latch_an_illegal_support_event():
    dropout = torch.tensor([False, True, True, False, True])
    subset_visibility = torch.tensor([True, False, True])
    supported = map_dropout_visibility_to_full_mask(
        dropout, subset_visibility, torch.zeros(5, 3)
    )
    state = GaussianObservation.zeros(5, torch.device("cpu"), VIEW_IDS)
    state.record_real_view(dropout, supported, 2, 7, is_real_view=True)

    assert supported.tolist() == [False, True, False, False, True]
    assert not state.invalid_screen_support_seen.item()
    state.audit_strict(expected_uid=torch.arange(5, dtype=torch.int64))


def test_distinct_views_count_repeated_camera_once_and_different_cameras():
    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    _record(state, [True] * 3, [True, False, True], 0, 1)
    _record(state, [True] * 3, [True, True, False], 0, 2)
    assert state.unique_screen_supported_real_views.tolist() == [1, 1, 1]
    _record(state, [True] * 3, [True, False, True], 2, 3)
    assert state.unique_screen_supported_real_views.tolist() == [2, 1, 2]
    assert state.screen_supported_hits.tolist() == [3, 1, 2]


def test_non_real_views_are_rejected_and_train_hook_precedes_shifted_render():
    state = GaussianObservation.zeros(2, torch.device("cpu"), VIEW_IDS)
    with pytest.raises(ValueError, match="real training view"):
        state.record_real_view(
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            0,
            1,
            is_real_view=False,
        )
    source = Path("train.py").read_text(encoding="utf-8")
    hook = source.index("gaussians.record_real_view_observation(")
    shifted = source.index("shifted_cam = scene.getShiftedCamera")
    assert hook < shifted
    assert source.count("gaussians.record_real_view_observation(") == 1


def test_manifest_camera_mapping_uses_frozen_identifiers(tmp_path):
    names = [f"image_{index:03d}.png" for index in range(25)]
    manifest = build_view_manifest(
        dataset="LLFF",
        scene="horns",
        image_names=names,
        view_count=3,
        source_commit="test",
    )
    manifest_path = tmp_path / "view_manifest.json"
    write_view_manifest(manifest_path, manifest)
    selected = manifest["selected_train_images"]
    cameras = [
        types.SimpleNamespace(uid=index + 10, image_name=Path(name).stem)
        for index, name in enumerate(reversed(selected))
    ]
    identifiers, mapping = build_real_view_camera_mapping(cameras, manifest_path)
    assert identifiers == tuple(selected)
    for camera in cameras:
        assert identifiers[mapping[camera.uid]] == next(
            name for name in selected if Path(name).stem == camera.image_name
        )


def test_clone_and_split_newborns_start_with_zero_observation():
    lifecycle = GaussianLifecycle.roots(3, torch.device("cpu"))
    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    _record(state, [True] * 3, [True, True, False], 0, 1)

    clone_parents = torch.tensor([True, False, True])
    lifecycle.append_children(clone_parents, "clone", iteration=2)
    state.append_zeros(2)
    assert state.real_view_opportunities[-2:].tolist() == [0, 0]
    assert not state.screen_supported_view_history[-2:].any()

    split_parents = torch.tensor([False, True, False, False, True])
    lifecycle.append_children(split_parents, "split", iteration=3, repeats=2)
    state.append_zeros(4)
    assert state.screen_supported_hits[-4:].tolist() == [0, 0, 0, 0]
    assert not state.screen_supported_view_history[-4:].any()

    replacement = torch.cat(
        (split_parents, torch.zeros(4, dtype=torch.bool))
    )
    lifecycle.prune(replacement, "split_replaced", iteration=3)
    state.prune(replacement)
    assert state.real_view_opportunities.tolist() == [1, 1, 0, 0, 0, 0, 0]
    state.audit_strict(expected_uid=lifecycle.uid)


def test_model_prune_uses_identical_mask_for_a0_and_b0():
    model = _make_model(4, lifecycle=True, observation=True)
    _record(
        model.observation,
        [True, True, False, True],
        [True, False, False, True],
        0,
        1,
    )
    model.prune_points(
        torch.tensor([False, True, False, True]),
        lifecycle_reason="training_prune",
        iteration=2,
    )
    assert model.lifecycle.uid.tolist() == [0, 2]
    assert model.observation.dropout_retained_hits.tolist() == [1, 0]
    assert model.observation.screen_supported_hits.tolist() == [1, 0]
    model.observation.audit_strict(expected_uid=model.lifecycle.uid)


def test_strict_audit_rejects_shapes_values_and_checkpoint_uid_order():
    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    state.screen_supported_hits[0] = 1
    with pytest.raises(ValueError, match="cannot exceed dropout_retained_hits"):
        state.audit_strict(expected_uid=torch.arange(3, dtype=torch.int64))

    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    malformed = state.state_dict(lifecycle_uid=torch.arange(3, dtype=torch.int64))
    malformed["screen_supported_view_history"] = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="view history is malformed"):
        GaussianObservation.from_state_dict(
            malformed,
            lifecycle_uid=torch.arange(3, dtype=torch.int64),
            expected_view_identifiers=VIEW_IDS,
            expected_count=3,
            device=torch.device("cpu"),
        )

    malformed = state.state_dict(lifecycle_uid=torch.arange(3, dtype=torch.int64))
    malformed["uid"] = torch.tensor([1, 0, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="UID rows"):
        GaussianObservation.from_state_dict(
            malformed,
            lifecycle_uid=torch.arange(3, dtype=torch.int64),
            expected_view_identifiers=VIEW_IDS,
            expected_count=3,
            device=torch.device("cpu"),
        )


def test_checkpoint_round_trip_restores_and_continues_accumulating():
    source = _make_model(3, lifecycle=True, observation=True)
    _record(source.observation, [True, False, True], [True, False, False], 0, 1)
    checkpoint = source.capture()
    assert len(checkpoint) == 13
    assert set(checkpoint[12]) == {"i2_lifecycle", "i2_observation"}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    target = _make_model(0, lifecycle=True, observation=True)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(torch.load(buffer, weights_only=False), object())
    assert target.observation.view_identifiers == VIEW_IDS
    assert target.observation.screen_supported_view_history.tolist() == [
        [True, False, False],
        [False, False, False],
        [False, False, False],
    ]
    with pytest.raises(ValueError, match="once in increasing iteration order"):
        _record(target.observation, [True] * 3, [False] * 3, 1, 1)
    _record(target.observation, [True] * 3, [True, True, False], 1, 2)
    assert target.observation.real_view_opportunities.tolist() == [2, 2, 2]
    assert target.observation.unique_screen_supported_real_views.tolist() == [2, 1, 0]


def test_strict_audit_enforces_iteration_evidence_continuity():
    uid = torch.arange(2, dtype=torch.int64)

    before_first_view = GaussianObservation.zeros(2, torch.device("cpu"), VIEW_IDS)
    checkpoint = before_first_view.state_dict(lifecycle_uid=uid)
    checkpoint["real_view_opportunities"] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="before a real-view iteration"):
        GaussianObservation.from_state_dict(
            checkpoint,
            lifecycle_uid=uid,
            expected_view_identifiers=VIEW_IDS,
            expected_count=2,
            device=torch.device("cpu"),
        )

    impossible_count = GaussianObservation.zeros(2, torch.device("cpu"), VIEW_IDS)
    _record(impossible_count, [True, True], [False, False], 0, 2)
    checkpoint = impossible_count.state_dict(lifecycle_uid=uid)
    checkpoint["real_view_opportunities"] = torch.tensor([4, 1])
    with pytest.raises(ValueError, match="exceed the recorded iteration range"):
        GaussianObservation.from_state_dict(
            checkpoint,
            lifecycle_uid=uid,
            expected_view_identifiers=VIEW_IDS,
            expected_count=2,
            device=torch.device("cpu"),
        )


def test_checkpoint_boundary_rejects_historical_illegal_support():
    state = GaussianObservation.zeros(1, torch.device("cpu"), VIEW_IDS)
    _record(state, [True], [False], 0, 1)
    _record(state, [False], [True], 1, 2)
    with pytest.raises(ValueError, match="outside the Dropout retained set"):
        state.state_dict(lifecycle_uid=torch.arange(1, dtype=torch.int64))


def test_checkpoint_compatibility_and_fail_closed_restore_rules():
    legacy = _make_model(2).capture()
    assert len(legacy) == 12
    a0_only = _make_model(2, lifecycle=True).capture()
    assert len(a0_only) == 13
    assert set(a0_only[12]) == {"i2_lifecycle"}

    target = _make_model(0, lifecycle=True, observation=True)
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    with pytest.raises(ValueError, match="without i2_observation metadata"):
        target.restore(a0_only, object())

    b0 = _make_model(2, lifecycle=True, observation=True).capture()
    assert b0[12]["i2_observation"]["version"] == 1
    assert set(b0[12]["i2_observation"]) == {
        "version",
        "uid",
        "real_view_opportunities",
        "dropout_retained_hits",
        "screen_supported_hits",
        "screen_supported_view_history",
        "view_identifiers",
        "last_real_view_iteration",
    }
    a0_target = _make_model(0, lifecycle=True)
    a0_target.training_setup = types.MethodType(_cpu_training_setup, a0_target)
    a0_target.restore(b0, object())
    assert a0_target.lifecycle is not None
    assert a0_target.observation is None

    legacy_target = _make_model(0)
    legacy_target.training_setup = types.MethodType(_cpu_training_setup, legacy_target)
    legacy_target.restore(legacy, object())
    assert legacy_target.lifecycle is None
    assert legacy_target.observation is None


def test_checkpoint_rejects_changed_training_view_identity_or_order():
    source = _make_model(2, lifecycle=True, observation=True)
    checkpoint = source.capture()
    target = _make_model(0, lifecycle=True)
    target.enable_observation_tracking(tuple(reversed(VIEW_IDS)))
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    with pytest.raises(ValueError, match="training-view identifiers"):
        target.restore(checkpoint, object())


def test_hot_path_does_not_call_strict_audit():
    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    with mock.patch.object(state, "audit_strict", side_effect=AssertionError):
        _record(state, [True, False, True], [True, False, False], 0, 1)
        state.append_zeros(1)
        state.prune(torch.tensor([False, True, False, False]))
    state.audit_strict(expected_uid=torch.tensor([0, 2, 3], dtype=torch.int64))


def test_hot_path_has_no_host_value_synchronization_or_async_assertion():
    source = inspect.getsource(GaussianObservation.record_real_view)
    assert ".item(" not in source
    assert ".cpu(" not in source
    assert "bool(" not in source
    assert "torch._assert_async" not in source


def test_cpu_rng_state_is_unchanged_by_observation_update():
    random.seed(17)
    torch.manual_seed(23)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    state = GaussianObservation.zeros(3, torch.device("cpu"), VIEW_IDS)
    _record(state, [True, False, True], [True, False, True], 0, 1)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_rng_state_is_unchanged_by_observation_update():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(29)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    state = GaussianObservation.zeros(3, device, VIEW_IDS)
    state.record_real_view(
        torch.tensor([True, False, True], device=device),
        torch.tensor([True, False, False], device=device),
        0,
        1,
        is_real_view=True,
    )
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_three_view_history_memory_cost_is_one_byte_per_entry():
    state = GaussianObservation.zeros(100_000, torch.device("cpu"), VIEW_IDS)
    assert state.view_history_bytes == 300_000
