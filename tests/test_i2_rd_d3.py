import json

import pytest
import torch

from test_i2_active_pilot import _real_cpu_model
from test_i2_rd_v1 import _rd_plan, _sha
from utils.i2_active_pilot import (
    ActivePilotError, apply_active_reallocation, apply_birth_only,
    apply_release_only, load_active_plan, validate_i2_active_configuration,
)
from utils.i2_lifecycle import GaussianLifecycle
from utils.i2_observation import GaussianObservation


def _rd_fixture(tmp_path):
    path, checkpoint = _rd_plan(tmp_path, k=2)
    donor_path = tmp_path / "donor_audit.json"
    donor = json.loads(donor_path.read_text(encoding="utf-8"))
    donors = [
        {"uid": 10002 + index, "birth_type": "clone", "rd_rank": index + 1}
        for index in range(2)
    ]
    donor["ranked_eligible"] = donors
    donor_path.write_text(json.dumps(donor), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["donor_audit"] = {"path": str(donor_path), "sha256": _sha(donor_path)}
    payload["donors"] = donors
    payload["matches"] = [
        {"donor_uid": donors[i]["uid"], "receiver_candidate_id": f"c{i:04d}",
         "released_units": 1, "received_units": 1}
        for i in range(2)
    ]
    payload["coverage"]["donor_selected_order"] = [row["uid"] for row in donors]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_active_plan(path, requested_budget=2, source_checkpoint=checkpoint)


def _model():
    model = _real_cpu_model(10004)
    with torch.no_grad():
        model._xyz[10000] = torch.tensor([0.0, 0.0, 0.0])
        model._xyz[10001] = torch.tensor([10.0, 0.0, 0.0])
    lifecycle = GaussianLifecycle.roots(10002, torch.device("cpu"))
    mask = torch.zeros(10002, dtype=torch.bool); mask[-2:] = True
    lifecycle.append_children(mask, "clone", 10)
    model.lifecycle = lifecycle
    model.observation = GaussianObservation.zeros(
        10004, torch.device("cpu"), ["train-a.jpg"]
    )
    return model


def test_birth_only_uses_exact_rd_v1_receiver_initialization(tmp_path):
    plan = _rd_fixture(tmp_path)
    rd_model, birth_model = _model(), _model()
    rd = apply_active_reallocation(
        rd_model, plan, iteration=1201, event_log=tmp_path / "rd.json"
    )
    birth = apply_birth_only(
        birth_model, plan, iteration=1201, event_log=tmp_path / "birth.json"
    )
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert torch.equal(getattr(rd_model, name)[-2:], getattr(birth_model, name)[-2:])
    assert rd["born_uids"] == birth["born_uids"]
    assert birth["anchor_uids"] == [10000, 10001]
    assert birth["released_units"] == 0 and birth["received_units"] == 2
    assert birth["gaussian_after"] == birth["gaussian_before"] + 2
    assert birth["budget_conserved"] is False


def test_release_and_birth_only_counts_alignment_and_uid_uniqueness(tmp_path):
    plan = _rd_fixture(tmp_path)
    release_model, birth_model = _model(), _model()
    release = apply_release_only(
        release_model, plan, iteration=1201, event_log=tmp_path / "release.json"
    )
    birth = apply_birth_only(
        birth_model, plan, iteration=1201, event_log=tmp_path / "birth.json"
    )
    assert release["released_uids"] == plan["_donor_uids"]
    assert release["received_units"] == 0
    assert release["gaussian_after"] == release["gaussian_before"] - 2
    assert birth["released_uids"] == [] and len(birth["born_uids"]) == 2
    for model in (release_model, birth_model):
        assert len(set(model.lifecycle.uid.tolist())) == model.get_xyz.shape[0]
        assert model.observation.real_view_opportunities.shape[0] == model.get_xyz.shape[0]


def test_birth_only_rejects_non_rd_plan_missing_donor_and_existing_event(tmp_path):
    plan = _rd_fixture(tmp_path)
    fake = dict(plan); fake["_rd_v1"] = False
    with pytest.raises(ActivePilotError, match="RD-v1"):
        apply_birth_only(_model(), fake, iteration=1201, event_log=tmp_path / "a.json")
    missing = dict(plan); missing["_donor_uids"] = [999999]
    with pytest.raises(ActivePilotError, match="missing"):
        apply_birth_only(_model(), missing, iteration=1201, event_log=tmp_path / "b.json")
    event = tmp_path / "exists.json"; event.write_text("{}", encoding="utf-8")
    with pytest.raises(ActivePilotError, match="already exists"):
        apply_birth_only(_model(), plan, iteration=1201, event_log=event)


def test_three_intervention_modes_are_mutually_exclusive_and_default_off():
    validate_i2_active_configuration(False, "", "", 0)
    with pytest.raises(ActivePilotError, match="mutually exclusive"):
        validate_i2_active_configuration(
            True, "plan.json", "event.json", 2, release_only=True
        )
    with pytest.raises(ActivePilotError, match="mutually exclusive"):
        validate_i2_active_configuration(
            False, "plan.json", "event.json", 2, release_only=True, birth_only=True
        )
