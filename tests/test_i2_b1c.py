import argparse
import csv
import io
import types

import pytest
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from tools.analyze_i2_b1c import analyze_c_rows, build_c_candidates
from utils.i2_future_utility import (
    FUTURE_UTILITY_CHECKPOINT_VERSION,
    GaussianFutureUtility,
    add_i2_future_utility_arguments,
    validate_i2_future_utility_configuration,
)
from utils.i2_lifecycle import BIRTH_TYPE_TO_CODE, GaussianLifecycle
from utils.i2_observation import GaussianObservation
from utils.i2_resource_analysis import analyze_observation_checkpoint, write_b1_outputs


VIEWS = ("a.png", "b.png", "c.png")


def _model(count=2):
    model = GaussianModel(1)
    model._xyz = nn.Parameter(torch.zeros(count, 3))
    model._features_dc = nn.Parameter(torch.zeros(count, 1, 3))
    model._features_rest = nn.Parameter(torch.zeros(count, 3, 3))
    model._scaling = nn.Parameter(torch.zeros(count, 3))
    model._rotation = nn.Parameter(torch.zeros(count, 4))
    model._opacity = nn.Parameter(torch.zeros(count, 1))
    model.max_radii2D = torch.zeros(count)
    model.xyz_gradient_accum = torch.zeros(count, 1)
    model.denom = torch.ones(count, 1)
    model.spatial_lr_scale = 1.0
    _cpu_training_setup(model, None)
    return model


def _cpu_training_setup(self, _args):
    groups = [
        {"params": [getattr(self, name)], "name": label}
        for name, label in (
            ("_xyz", "xyz"), ("_features_dc", "f_dc"),
            ("_features_rest", "f_rest"), ("_opacity", "opacity"),
            ("_scaling", "scaling"), ("_rotation", "rotation"),
        )
    ]
    self.optimizer = torch.optim.Adam(groups, lr=0.0)


def _b0_payload(iteration=1201):
    lifecycle = GaussianLifecycle.roots(3, torch.device("cpu"))
    lifecycle.birth_iter = torch.tensor([0, 1000, 1100], dtype=torch.int64)
    lifecycle.birth_type = torch.tensor(
        [BIRTH_TYPE_TO_CODE["root"], BIRTH_TYPE_TO_CODE["clone"], BIRTH_TYPE_TO_CODE["split"]],
        dtype=torch.uint8,
    )
    # This fixture targets the offline row audit, not strict lifecycle history.
    observation = GaussianObservation.zeros(3, torch.device("cpu"), VIEWS)
    observation.real_view_opportunities[:] = torch.tensor([10, 10, 0])
    observation.dropout_retained_hits[:] = torch.tensor([8, 5, 0])
    observation.screen_supported_hits[:] = torch.tensor([6, 1, 0])
    observation.screen_supported_view_history[:] = torch.tensor(
        [[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
    )
    observation.last_real_view_iteration = iteration
    extension = {
        "i2_lifecycle": {
            "version": 1,
            "uid": lifecycle.uid,
            "birth_iter": lifecycle.birth_iter,
            "birth_type": lifecycle.birth_type,
            "source_uid": lifecycle.source_uid,
            "generation": lifecycle.generation,
        },
        "i2_observation": observation.state_dict(lifecycle_uid=lifecycle.uid),
    }
    return ((None,) * 12 + (extension,), iteration)


def test_b1_checkpoint_export_preserves_uid_semantics_and_marks_radius_unavailable(tmp_path):
    analysis = analyze_observation_checkpoint(_b0_payload())
    assert analysis["uid_alignment"] == "exact"
    assert analysis["rows"][1]["birth_type"] == "clone"
    assert analysis["rows"][1]["screen_support_rate"] == pytest.approx(0.1)
    assert analysis["rows"][1]["real_view_coverage_rate"] == pytest.approx(1 / 3)
    assert analysis["rows"][2]["observation_status"] == "no_real_view_opportunity"
    assert analysis["semantic_guards"]["historical_radius_gt_2px_available"] is False
    candidate, summary = write_b1_outputs(analysis, tmp_path)
    assert candidate.exists() and summary.exists()
    with candidate.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_b1_rejects_uid_row_mismatch():
    payload = _b0_payload()
    payload[0][12]["i2_observation"]["uid"] = torch.tensor([1, 0, 2])
    with pytest.raises(ValueError, match="UID order"):
        analyze_observation_checkpoint(payload)


def _tracker_with_clone():
    lifecycle = GaussianLifecycle.roots(1, torch.device("cpu"))
    child_uid = lifecycle.append_children(torch.tensor([True]), "clone", 10)
    observation = GaussianObservation.zeros(1, torch.device("cpu"), VIEWS)
    observation.append_zeros(1)
    tracker = GaussianFutureUtility.empty(1, torch.device("cpu"))
    tracker.append_births(
        child_uid=child_uid,
        birth_iter=lifecycle.birth_iter[-1:],
        birth_type=lifecycle.birth_type[-1:],
        source_uid=lifecycle.source_uid[-1:],
        generation=lifecycle.generation[-1:],
        xyz=torch.tensor([[1.0, 2.0, 3.0]]),
        opacity=torch.tensor([[0.25]]),
        scaling=torch.tensor([[1.0, 1.0, 1.0]]),
        previous_count=1,
    )
    return lifecycle, observation, tracker


def test_c_vectorized_age_snapshot_terminal_state_and_checkpoint_round_trip():
    lifecycle, observation, tracker = _tracker_with_clone()
    observation.real_view_opportunities[1] = 5
    observation.dropout_retained_hits[1] = 4
    observation.screen_supported_hits[1] = 3
    observation.screen_supported_view_history[1, :2] = True
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 3.0]])
    opacity = torch.tensor([[0.1], [0.5]])
    scaling = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    tracker.capture_due(
        lifecycle=lifecycle, observation=observation, xyz=xyz, opacity=opacity,
        scaling=scaling, iteration=110,
    )
    snap = tracker.records[1]["age_snapshots"]["100"]
    assert snap["opacity_retention_ratio"] == pytest.approx(2.0)
    assert snap["xyz_shift_from_birth"] == pytest.approx(1.0)
    assert snap["unique_real_views"] == 2
    mask = torch.tensor([False, True])
    tracker.record_deaths(
        mask=mask, lifecycle=lifecycle, observation=observation, xyz=xyz,
        opacity=opacity, scaling=scaling, iteration=160, reason="split_replaced",
    )
    tracker.prune(mask)
    lifecycle.prune(mask, "split_replaced", 160)
    observation.prune(mask)
    state = tracker.state_dict(lifecycle_uid=lifecycle.uid)
    restored = GaussianFutureUtility.from_state_dict(
        state, lifecycle_uid=lifecycle.uid, device=torch.device("cpu")
    )
    assert restored.records[1]["death_reason"] == "split_replaced"
    assert restored.records[1]["terminal_state"]["age"] == 150


def test_c_outcomes_keep_split_replacement_distinct_and_refuse_single_outcome_claims():
    lifecycle, observation, tracker = _tracker_with_clone()
    record = tracker.records[1]
    record["alive"] = False
    record["death_iteration"] = 200
    record["death_reason"] = "split_replaced"
    state = {"version": FUTURE_UTILITY_CHECKPOINT_VERSION, "records": [record]}
    rows, status = build_c_candidates((None,) * 12 + ({"i2_future_utility": state},))
    assert status == "available"
    assert rows[0]["raw_uid_survival"] == 0
    assert rows[0]["true_prune_avoidance"] == 1
    report = analyze_c_rows(rows, bootstrap_reps=10)
    assert report["status"] == "insufficient_outcome_variation"
    assert all(
        outcome["status"] == "insufficient_samples"
        for result in report["analyses"].values()
        for outcome in result["outcomes"].values()
    )


def test_c_spearman_quartile_and_grouped_bootstrap_run_on_resolved_mixed_outcomes():
    rows = []
    for index in range(8):
        transition = ("alive", "training_prune", "split_replaced")[index % 3]
        row = {
            "uid": index,
            "birth_iteration": (index // 2) * 10,
            "outcome_status": "resolved",
            "lifecycle_transition": transition,
            "raw_uid_survival": int(transition == "alive"),
            "true_prune_avoidance": int(transition != "training_prune"),
        }
        for offset, feature in enumerate((
            "visible_rate", "unique_real_views", "opacity_retention_ratio",
            "scale_ratio_from_birth", "xyz_shift_from_birth",
        )):
            row[f"age100_{feature}"] = float(index + offset)
        rows.append(row)
    report = analyze_c_rows(rows, bootstrap_reps=20, seed=7)
    assert report["status"] == "analyzed"
    result = report["analyses"]["age100_visible_rate"]["outcomes"]["raw_uid_survival"]
    assert result["status"] == "ok"
    assert result["grouped_bootstrap"]["spearman"]["valid_replicates"] > 0


def test_c_cli_is_opt_in_and_requires_a0_b0():
    parser = argparse.ArgumentParser()
    add_i2_future_utility_arguments(parser)
    assert parser.parse_args([]).enable_i2_future_utility_tracking is False
    assert parser.parse_args(["--enable_i2_future_utility_tracking"]).enable_i2_future_utility_tracking
    validate_i2_future_utility_configuration(True, True, True)
    with pytest.raises(ValueError, match="requires both"):
        validate_i2_future_utility_configuration(True, True, False)


def test_c_uses_existing_13_field_extension_and_round_trips_without_breaking_a0_b0():
    source = _model()
    source.enable_lifecycle_tracking()
    source.enable_observation_tracking(VIEWS)
    source.enable_future_utility_tracking()
    checkpoint = source.capture()
    assert len(checkpoint) == 13
    assert set(checkpoint[12]) == {
        "i2_lifecycle", "i2_observation", "i2_future_utility"
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    target = _model()
    target.enable_lifecycle_tracking()
    target.enable_observation_tracking(VIEWS)
    target.enable_future_utility_tracking()
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(torch.load(buffer, weights_only=False), object())
    assert target.lifecycle.uid.tolist() == [0, 1]
    assert target.future_utility.records == {}


def test_c_can_bootstrap_prospectively_from_frozen_a0_b0_without_backfill():
    source = _model()
    source.enable_lifecycle_tracking()
    source.enable_observation_tracking(VIEWS)
    frozen_b0 = source.capture()
    assert "i2_future_utility" not in frozen_b0[12]
    target = _model()
    target.enable_lifecycle_tracking()
    target.enable_observation_tracking(VIEWS)
    target.enable_future_utility_tracking()
    target.training_setup = types.MethodType(_cpu_training_setup, target)
    target.restore(frozen_b0, object())
    assert target.future_utility.records == {}
    assert target.future_utility.snapshot_taken.shape == (2, 2)
    assert not target.future_utility.tracked_cohort.any()


def test_c_non_snapshot_iteration_avoids_nonzero_and_host_sync(monkeypatch):
    lifecycle, observation, tracker = _tracker_with_clone()
    original_nonzero = torch.Tensor.nonzero

    def forbidden_nonzero(*args, **kwargs):
        raise AssertionError("nonzero must not run before the next snapshot boundary")

    monkeypatch.setattr(torch.Tensor, "nonzero", forbidden_nonzero)
    tracker.capture_due(
        lifecycle=lifecycle, observation=observation, xyz=torch.zeros(2, 3),
        opacity=torch.full((2, 1), 0.5), scaling=torch.ones(2, 3), iteration=109,
    )
    monkeypatch.setattr(torch.Tensor, "nonzero", original_nonzero)
    assert tracker.records[1]["age_snapshots"] == {}


def test_c_prospective_resume_tracks_only_post_resume_births_through_prune_and_checkpoint():
    # The frozen B0 state already contains one clone and one split child.
    lifecycle = GaussianLifecycle.roots(1, torch.device("cpu"))
    lifecycle.append_children(torch.tensor([True]), "clone", 10)
    lifecycle.append_children(torch.tensor([True, False]), "split", 20)
    source = _model(3)
    source.lifecycle = lifecycle
    source.observation = GaussianObservation.zeros(3, torch.device("cpu"), VIEWS)
    frozen_b0 = source.capture()

    resumed = _model(3)
    resumed.enable_lifecycle_tracking()
    resumed.enable_observation_tracking(VIEWS)
    resumed.enable_future_utility_tracking()
    resumed.training_setup = types.MethodType(_cpu_training_setup, resumed)
    resumed.restore(frozen_b0, object())

    values3 = torch.zeros(3, 3)
    opacity3 = torch.full((3, 1), 0.5)
    resumed.future_utility.capture_due(
        lifecycle=resumed.lifecycle, observation=resumed.observation, xyz=values3,
        opacity=opacity3, scaling=torch.ones(3, 3), iteration=120,
    )
    assert resumed.future_utility.records == {}  # old clone reached age100, still excluded

    old_split_uid = int(resumed.lifecycle.uid[2])
    old_prune = torch.tensor([False, False, True])
    resumed.future_utility.record_deaths(
        mask=old_prune, lifecycle=resumed.lifecycle, observation=resumed.observation,
        xyz=values3, opacity=opacity3, scaling=torch.ones(3, 3),
        iteration=121, reason="training_prune",
    )
    resumed.future_utility.prune(old_prune)
    resumed.observation.prune(old_prune)
    resumed.lifecycle.prune(old_prune, "training_prune", 121)
    assert old_split_uid not in resumed.future_utility.records

    child_uid = resumed.lifecycle.append_children(
        torch.tensor([True, False]), "clone", 130
    )
    resumed.observation.append_zeros(1)
    resumed.future_utility.append_births(
        child_uid=child_uid,
        birth_iter=resumed.lifecycle.birth_iter[-1:],
        birth_type=resumed.lifecycle.birth_type[-1:],
        source_uid=resumed.lifecycle.source_uid[-1:],
        generation=resumed.lifecycle.generation[-1:],
        xyz=torch.zeros(1, 3), opacity=torch.full((1, 1), 0.5),
        scaling=torch.ones(1, 3), previous_count=2,
    )
    resumed.future_utility.capture_due(
        lifecycle=resumed.lifecycle, observation=resumed.observation,
        xyz=torch.zeros(3, 3), opacity=torch.full((3, 1), 0.5),
        scaling=torch.ones(3, 3), iteration=230,
    )
    assert set(resumed.future_utility.records) == {int(child_uid[0])}
    assert "100" in resumed.future_utility.records[int(child_uid[0])]["age_snapshots"]

    checkpoint = resumed.capture()
    assert checkpoint[12]["i2_future_utility"]["version"] == 2
    restored = _model(3)
    restored.enable_lifecycle_tracking()
    restored.enable_observation_tracking(VIEWS)
    restored.enable_future_utility_tracking()
    restored.training_setup = types.MethodType(_cpu_training_setup, restored)
    restored.restore(checkpoint, object())
    assert restored.future_utility.tracked_cohort.tolist() == [False, False, True]
    assert set(restored.future_utility.records) == {int(child_uid[0])}


def _c_record_with_outcome(*, terminal_age=None, reason=None, age500=False):
    _, _, tracker = _tracker_with_clone()
    record = tracker.records[1]
    if age500:
        record["age_snapshots"]["500"] = {"opacity_retention_ratio": 1.0}
    if terminal_age is not None:
        record["alive"] = False
        record["death_iteration"] = record["birth_iteration"] + terminal_age
        record["death_reason"] = reason
    return record


def _candidate_row(record):
    state = {
        "version": FUTURE_UTILITY_CHECKPOINT_VERSION,
        "records": [record],
    }
    return build_c_candidates((None,) * 12 + ({"i2_future_utility": state},))[0][0]


def test_c_age500_rejects_snapshot_with_terminal_at_or_before_boundary():
    for terminal_age in (499, 500):
        with pytest.raises(ValueError, match="age500 snapshot conflicts"):
            _candidate_row(_c_record_with_outcome(
                terminal_age=terminal_age, reason="training_prune", age500=True
            ))


def test_c_age500_legal_snapshot_and_early_death_semantics():
    alive = _candidate_row(_c_record_with_outcome(
        terminal_age=501, reason="training_prune", age500=True
    ))
    assert alive["lifecycle_transition"] == "alive"
    split = _candidate_row(_c_record_with_outcome(
        terminal_age=400, reason="split_replaced"
    ))
    assert split["raw_uid_survival"] == 0
    assert split["true_prune_avoidance"] == 1
    pruned = _candidate_row(_c_record_with_outcome(
        terminal_age=400, reason="training_prune"
    ))
    assert pruned["true_prune_avoidance"] == 0


def test_c_age500_immature_record_remains_unresolved():
    row = _candidate_row(_c_record_with_outcome())
    assert row["outcome_status"] == "unresolved_age500"
    assert row["lifecycle_transition"] is None
