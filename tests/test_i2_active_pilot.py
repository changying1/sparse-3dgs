import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.i2_active_pilot import ActivePilotError, V2_SELECTION_ANCHOR_CLONE_METHOD, V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT, add_i2_active_pilot_arguments, apply_active_reallocation, apply_release_only, load_active_plan, receiver_v2_displacement_contract, record_release_only_reference_extension, select_receiver_v2_candidates, validate_i2_active_configuration
from utils.i2_lifecycle import GaussianLifecycle
from utils.i2_observation import GaussianObservation
from utils.i2_future_utility import GaussianFutureUtility


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path)}


def _plan(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha(checkpoint)
    payload = tmp_path / "receiver.npz"
    payload.write_bytes(b"real receiver arrays")
    evidence = tmp_path / "receiver.json"
    evidence_ref = _write_json(evidence, {
        "format": "i2_receiver_view_evidence_v1", "training_camera_id": "train-a.jpg",
        "source_checkpoint_sha256": checkpoint_sha, "generation_rule_id": "rule-v1",
        "payload": {"path": str(payload), "sha256": _sha(payload)},
    })
    fields = ["candidate_id", "demand_basis", "demand_score", "requested_units",
              "training_camera_id", "pixel_x0", "pixel_y0", "pixel_x1", "pixel_y1",
              "pixel_u", "pixel_v", "reprojection_error_px", "world_x", "world_y", "world_z",
              "anchor_type", "anchor_uid", "depth_valid", "geometry_valid", "local_gaussian_count",
              "local_density_radius", "evidence_path", "evidence_sha256",
              "source_checkpoint_sha256", "generation_rule_id"]
    receiver_csv = tmp_path / "receivers.csv"
    raw_receivers = []
    with receiver_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(20):
            row = {
                "candidate_id": f"r{i}", "demand_basis": "structural_defect",
                "demand_score": str(1 - i / 100), "requested_units": "1",
                "training_camera_id": "train-a.jpg", "pixel_x0": str(i * 10), "pixel_y0": "0",
                "pixel_x1": str(i * 10 + 8), "pixel_y1": "8", "pixel_u": str(i * 10 + 4),
                "pixel_v": "4", "reprojection_error_px": "1", "world_x": str(float(3 * i)),
                "world_y": str(float(3 * i + 1)), "world_z": str(float(3 * i + 2)),
                "anchor_type": "visible_existing_gaussian_projection",
                "anchor_uid": str(i), "depth_valid": "true", "geometry_valid": "true",
                "local_gaussian_count": "1", "local_density_radius": "0.1",
                "evidence_path": str(evidence), "evidence_sha256": evidence_ref["sha256"],
                "source_checkpoint_sha256": checkpoint_sha, "generation_rule_id": "rule-v1",
            }
            raw_receivers.append(row)
            writer.writerow(row)
    protocol = {
        "format": "i2_receiver_protocol_v1", "source_checkpoint_sha256": checkpoint_sha,
        "source_iteration": 1201, "training_camera_ids": ["train-a.jpg"],
        "generation_rule_id": "rule-v1", "source_code_fingerprint": "code-sha",
        "score_normalization": "test", "spatial_dedup_radius": 0.01,
        "max_anchor_pixel_distance": 12.0,
    }
    from utils.i2_d0_dry_run import validate_receivers
    receivers = validate_receivers(raw_receivers, {
        "_manifest_path": str(tmp_path / "manifest.json"),
        "source_checkpoint": {"sha256": checkpoint_sha}, "receiver_protocol": protocol,
    })["rows"]
    receivers = json.loads(json.dumps(receivers))
    donors = [{"uid": uid, "birth_type": "clone"} for uid in range(20, 40)]
    cf_rows = [{"uid": uid, "max_mean_abs_rgb_delta": 0.00001,
                "max_abs_rgb_delta": 0.001, "max_psnr_drop_db": 0.001,
                "safe_to_release": True, "per_view": [{
                    "training_camera_uid": 0, "training_camera_id": "train-a",
                    "mean_abs_rgb_delta": 0.00001, "max_abs_rgb_delta": 0.001,
                    "psnr_drop_db": 0.001,
                }]} for uid in range(20, 40)]
    cf_ref = _write_json(tmp_path / "counterfactual.json", {
        "format": "i2_static_deletion_counterfactual_v1", "source_iteration": 1201,
        "source_checkpoint_sha256": checkpoint_sha, "training_view_count": 1,
        "test_views_used": False,
        "thresholds": {"max_mean_abs_rgb_delta": 0.0001, "max_psnr_drop_db": 0.01},
        "audited_count": 20, "safe_count": 20, "rows": cf_rows,
    })
    donor_uids = list(range(20, 40))
    joint_ref = _write_json(tmp_path / "joint.json", {
        "format": "i2_joint_deletion_counterfactual_v1", "source_iteration": 1201,
        "source_checkpoint_sha256": checkpoint_sha, "test_views_used": False,
        "donor_uids": donor_uids,
        "training_view_count": 1,
        "thresholds": {"max_mean_abs_rgb_delta": 0.0001, "max_psnr_drop_db": 0.01},
        "max_mean_abs_rgb_delta": 0.00001, "max_psnr_drop_db": 0.001,
        "safe_to_release": True, "per_view": [{
            "mean_abs_rgb_delta": 0.00001, "max_abs_rgb_delta": 0.001,
            "psnr_drop_db": 0.001, "single_point_sum_mean_abs_rgb_delta": 0.00001,
            "joint_minus_single_sum_mean_abs_rgb_delta": 0.0,
        }],
    })
    plan = {
        "format": "i2_d0_read_only_plan_v2", "mode": "read_only_dry_run",
        "strategy": "b1_only_exploratory", "configuration": {"budget_k": 20},
        "safety_readiness": {"b1_only_exploratory": True}, "budget_audit": {"conserved": True},
        "input_audit": {"source_checkpoint_sha256": checkpoint_sha, "source_iteration": 1201,
                        "receiver_candidate_table": {"path": str(receiver_csv), "sha256": _sha(receiver_csv)},
                        "receiver_protocol": protocol},
        "donors": {"selected": donors}, "receivers": {"rows": receivers},
        "matches": [{"donor_uid": 20 + i, "receiver_candidate_id": f"r{i}",
                     "released_units": 1, "received_units": 1} for i in range(20)],
        "donor_counterfactual": {"format": "i2_static_deletion_counterfactual_v1",
                                 "artifact": cf_ref, "source_checkpoint_sha256": checkpoint_sha,
                                 "rows": cf_rows},
        "joint_deletion_counterfactual": {"format": "i2_joint_deletion_counterfactual_v1",
                                          "artifact": joint_ref},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, checkpoint


def _mutate_plan(path, mutation):
    plan = json.loads(path.read_text(encoding="utf-8"))
    mutation(plan)
    path.write_text(json.dumps(plan), encoding="utf-8")


def _receiver_v2_plan(tmp_path):
    path, checkpoint = _plan(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i in range(20):
        anchor = [float(3 * i), float(3 * i + 1), float(3 * i + 2)]
        target = [anchor[0] + 0.5, anchor[1], anchor[2]]
        rows.append({
            "candidate_id": f"r{i}", "anchor_uid": i,
            "anchor_xyz": anchor, "new_xyz": target,
            "displacement_norm": 0.5, "activated_opacity": 0.05,
            "activated_scale": [0.4, 0.3, 0.2],
            "rotation_source": "anchor", "sh_source": "anchor",
            "geometry_valid": True, "fail_closed": False,
            "cross_view_checks": [{"training_camera_id": "train-a.jpg", "safe": True}],
        })
    evidence_ref = _write_json(tmp_path / "receiver_v2_evidence.json", {
        "format": "i2_receiver_v2_geometry_evidence_v1",
        "method_version": "receiver_v2_tangent_jacobian_v1",
        "source_checkpoint_sha256": _sha(checkpoint), "source_iteration": 1201,
        "receiver_candidate_table_sha256": plan["input_audit"]["receiver_candidate_table"]["sha256"],
        "test_views_used": False, "rows": rows,
    })
    plan["format"] = "i2_d1_receiver_v2_plan_v1"
    plan["receiver_v2"] = {
        "method_version": "receiver_v2_tangent_jacobian_v1", "artifact": evidence_ref,
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, checkpoint


def _v2_selection_anchor_clone_plan(tmp_path):
    v2_path, checkpoint = _receiver_v2_plan(tmp_path)
    original_path = tmp_path / "original_receiver_v2_plan.json"
    original = json.loads(v2_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "view_manifest.json"
    manifest_path.write_text(json.dumps({"selected_train_images": ["train-a.jpg"]}), encoding="utf-8")
    original["input_audit"]["receiver_protocol"]["view_manifest_sha256"] = _sha(manifest_path)
    original_path.write_text(json.dumps(original), encoding="utf-8")
    plan = json.loads(json.dumps(original))
    plan["format"] = V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT
    plan["method_version"] = V2_SELECTION_ANCHOR_CLONE_METHOD
    plan["receiver_v2_selection_anchor_clone"] = {
        "method_version": V2_SELECTION_ANCHOR_CLONE_METHOD,
        "original_receiver_v2_plan": {
            "path": str(original_path), "sha256": _sha(original_path),
        },
        "receiver_v2_geometry_evidence": plan["receiver_v2"]["artifact"],
        "training_view_manifest": {
            "path": str(manifest_path), "sha256": _sha(manifest_path),
        },
        "initialization": {
            "xyz": "anchor_raw__xyz", "opacity": "anchor_raw__opacity",
            "scale": "anchor_raw__scaling", "rotation": "anchor_raw__rotation",
            "sh_dc": "anchor_raw__features_dc", "sh_rest": "anchor_raw__features_rest",
        },
        "test_views_used": False,
    }
    path = tmp_path / "anchor_clone_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, checkpoint


def _real_cpu_model(count=40):
    model = GaussianModel(1)
    model._xyz = nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 3))
    model._features_dc = nn.Parameter(torch.zeros(count, 1, 3))
    model._features_rest = nn.Parameter(torch.zeros(count, 3, 3))
    model._opacity = nn.Parameter(torch.zeros(count, 1))
    model._scaling = nn.Parameter(torch.zeros(count, 3))
    model._rotation = nn.Parameter(torch.zeros(count, 4)); model._rotation.data[:, 0] = 1
    groups = [{"params": [parameter], "name": name, "lr": 0.01} for name, parameter in
              (("xyz", model._xyz), ("f_dc", model._features_dc), ("f_rest", model._features_rest),
               ("opacity", model._opacity), ("scaling", model._scaling), ("rotation", model._rotation))]
    model.optimizer = torch.optim.Adam(groups, lr=0.0)
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        model.optimizer.state[parameter] = {"step": torch.tensor(1.0),
                                            "exp_avg": torch.ones_like(parameter),
                                            "exp_avg_sq": torch.full_like(parameter, 2)}
    model.xyz_gradient_accum = torch.arange(count, dtype=torch.float32)[:, None]
    model.denom = torch.arange(count, dtype=torch.float32)[:, None] + 10
    model.max_radii2D = torch.arange(count, dtype=torch.float32) + 20
    model.lifecycle = GaussianLifecycle.roots(20, torch.device("cpu"))
    model.lifecycle.append_children(torch.ones(20, dtype=torch.bool), "clone", 10)
    model.observation = GaussianObservation.zeros(count, torch.device("cpu"), ["train-a.jpg"])
    model.future_utility = None
    return model


def test_active_flags_are_default_off():
    parser = argparse.ArgumentParser(); add_i2_active_pilot_arguments(parser)
    args = parser.parse_args([])
    assert args.enable_i2_active_pilot is False
    assert args.enable_i2_release_only is False and args.i2_active_budget == 20


def test_release_only_and_reallocation_are_mutually_exclusive():
    with pytest.raises(ActivePilotError, match="mutually exclusive"):
        validate_i2_active_configuration(True, "plan", "event", 20, release_only=True)


def test_tampered_embedded_counterfactual_cannot_override_unsafe_artifact(tmp_path):
    path, _ = _plan(tmp_path)
    artifact = json.loads((tmp_path / "counterfactual.json").read_text())
    artifact["rows"][0]["max_mean_abs_rgb_delta"] = 1.0
    artifact["rows"][0]["per_view"][0]["mean_abs_rgb_delta"] = 1.0
    artifact["rows"][0]["safe_to_release"] = False
    artifact["safe_count"] = 19
    ref = _write_json(tmp_path / "counterfactual.json", artifact)
    def mutate(plan):
        plan["donor_counterfactual"]["artifact"] = ref
        plan["donor_counterfactual"]["rows"][0]["safe_to_release"] = True
    _mutate_plan(path, mutate)
    with pytest.raises(ActivePilotError, match="differs from verified artifact"):
        load_active_plan(path, requested_budget=20)


def test_match_donor_uids_must_equal_selected_donors(tmp_path):
    path, _ = _plan(tmp_path)
    _mutate_plan(path, lambda plan: [row.update(donor_uid=99999) for row in plan["matches"]])
    with pytest.raises(ActivePilotError, match="equal selected donors"):
        load_active_plan(path, requested_budget=20)


@pytest.mark.parametrize("c_enabled", [False, True])
def test_real_gaussian_model_cpu_optimizer_uid_stats_and_observation_alignment(tmp_path, c_enabled):
    path, checkpoint = _plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    model = _real_cpu_model()
    if c_enabled:
        model.future_utility = GaussianFutureUtility.empty(40, torch.device("cpu"))
    before_stats = (model.xyz_gradient_accum[:20].clone(), model.denom[:20].clone(), model.max_radii2D[:20].clone())
    event = apply_active_reallocation(model, plan, iteration=1201, event_log=tmp_path / "active_event.json")
    assert event["budget_conserved"] and event["survivor_densification_stats_preserved"]
    assert event["gaussian_before"] == event["gaussian_after"] == 40
    assert torch.equal(model.xyz_gradient_accum[:20], before_stats[0])
    assert torch.equal(model.denom[:20], before_stats[1])
    assert torch.equal(model.max_radii2D[:20], before_stats[2])
    assert model.lifecycle.uid.dtype == torch.int64
    assert model.observation.real_view_opportunities.shape[0] == 40
    assert torch.count_nonzero(model.observation.real_view_opportunities[-20:]) == 0
    if c_enabled:
        assert model.future_utility.tracked_cohort.shape[0] == 40
        assert bool(torch.all(model.future_utility.tracked_cohort[-20:]))
    death = model.lifecycle.events[-2]
    assert death["event_type"] == "death" and death["reason"] == "i2_active_release"
    assert death["uids"].tolist() == list(range(20, 40))
    model.lifecycle.audit_strict(expected_count=40)
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        assert parameter.shape[0] == model.optimizer.state[parameter]["exp_avg"].shape[0] == 40


def test_bad_plan_rejected_before_real_model_mutation(tmp_path):
    path, _ = _plan(tmp_path)
    _mutate_plan(path, lambda plan: plan["receivers"]["rows"][0].update(target_xyz=[9, 9, 9]))
    model = _real_cpu_model(); before = model._xyz.detach().clone()
    with pytest.raises(ActivePilotError):
        load_active_plan(path, requested_budget=20)
    assert torch.equal(model._xyz, before)


def test_receiver_v2_plan_isolated_nonoverlap_conservative_and_aligned(tmp_path):
    path, checkpoint = _receiver_v2_plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    assert plan["_receiver_v2"] is True
    model = _real_cpu_model()
    anchor_before = model.get_xyz[:20].detach().clone()
    event = apply_active_reallocation(
        model, plan, iteration=1201, event_log=tmp_path / "receiver_v2_event.json"
    )
    assert event["format"] == "i2_active_event_receiver_v2_v1"
    assert event["method_version"] == "receiver_v2_tangent_jacobian_v1"
    assert event["budget_conserved"] and event["alignment"]["pass"]
    assert torch.all(torch.linalg.vector_norm(model.get_xyz[-20:] - anchor_before, dim=1) > 0)
    assert torch.allclose(model.get_opacity[-20:], torch.full((20, 1), 0.05))
    assert torch.allclose(model.get_scaling[-20:], torch.tensor([0.4, 0.3, 0.2]).repeat(20, 1))
    assert torch.allclose(model.get_rotation[-20:], model.get_rotation[:20])
    model.lifecycle.audit_strict(expected_count=40)
    model.observation.audit_strict(expected_uid=model.lifecycle.uid, expected_count=40)
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        assert parameter.shape[0] == model.optimizer.state[parameter]["exp_avg"].shape[0] == 40


@pytest.mark.parametrize("mutation,match", [
    (lambda evidence: evidence["rows"][0].update(new_xyz=evidence["rows"][0]["anchor_xyz"]),
     "non-overlapping"),
    (lambda evidence: evidence["rows"][0].update(activated_opacity=0.5),
     "opacity/scale"),
    (lambda evidence: evidence["rows"][0]["cross_view_checks"][0].update(safe=False),
     "cross-view"),
])
def test_receiver_v2_fail_closed_evidence_validation(tmp_path, mutation, match):
    path, _ = _receiver_v2_plan(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    evidence_path = tmp_path / "receiver_v2_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    mutation(evidence)
    plan["receiver_v2"]["artifact"] = _write_json(evidence_path, evidence)
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ActivePilotError, match=match):
        load_active_plan(path, requested_budget=20)


def _selection_row(candidate_id, x, score, *, recorded_norm=1.0):
    return {
        "candidate_id": candidate_id,
        "anchor_xyz": [x, 0.0, 0.0],
        "new_xyz": [x + 1.0, 0.0, 0.0],
        "displacement_norm": recorded_norm,
        "photometric_alignment": score,
        "demand_score": score,
    }


def test_receiver_v2_float32_rounded_xyz_overlap_is_rejected():
    # At 2**24, adding one is lost by float32 storage.
    anchor = torch.tensor([float(2**24), 0.0, 0.0], dtype=torch.float32)
    target = anchor + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    ok, reason, _, norm = receiver_v2_displacement_contract(
        anchor.tolist(), target.tolist(), 1.0
    )
    assert not ok and reason == "effective_xyz_overlap" and norm == 0.0


def test_receiver_v2_recorded_norm_must_match_effective_xyz():
    ok, reason, _, actual = receiver_v2_displacement_contract(
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 0.6
    )
    assert not ok and reason == "displacement_norm_mismatch" and actual == 0.5


def test_receiver_v2_rejection_replenishes_in_frozen_order():
    rows = [
        _selection_row("invalid-top", 0.0, 10.0, recorded_norm=2.0),
        _selection_row("first", 10.0, 9.0),
        _selection_row("second", 20.0, 8.0),
    ]
    selected, rejected = select_receiver_v2_candidates(
        rows, budget=2, spatial_dedup_radius=0.1
    )
    assert [row["candidate_id"] for row in selected] == ["first", "second"]
    assert rejected == [{
        "candidate_id": "invalid-top", "reason": "displacement_norm_mismatch"
    }]


def test_receiver_v2_spatial_nms_records_current_candidate_and_replenishes():
    rows = [
        _selection_row("first", 0.0, 10.0),
        _selection_row("nms-current", 0.05, 9.0),
        _selection_row("replacement", 10.0, 8.0),
    ]
    selected, rejected = select_receiver_v2_candidates(
        rows, budget=2, spatial_dedup_radius=0.1
    )
    assert [row["candidate_id"] for row in selected] == ["first", "replacement"]
    assert {"candidate_id": "nms-current", "reason": "v2_spatial_nms"} in rejected


def test_receiver_v2_insufficient_valid_capacity_is_explicit_no_go():
    with pytest.raises(ActivePilotError, match="NO-GO: only 1.*K=2"):
        select_receiver_v2_candidates(
            [_selection_row("only", 0.0, 1.0)], budget=2,
            spatial_dedup_radius=0.1,
        )


def test_horns_frozen_receiver_v2_artifacts_are_unchanged():
    evidence = ROOT / "reports/i2_receiver_v2_horns_1201_k20/receiver_v2_geometry_evidence.json"
    plan = ROOT / "reports/i2_receiver_v2_horns_1201_k20/i2_d1_receiver_v2_k20_plan.json"
    assert _sha(evidence) == "e1c3472cf9fb74d54dfd5c74b45ef27c4c4e1ed8fcf6f41dac6f7d24b22d9c08"
    assert _sha(plan) == "0bcd58837112fcf06cbb9041cda09afb9a97a3b0034845deb24f463f94827f46"


def test_v2_selection_anchor_clone_exact_raw_copy_budget_and_isolation(tmp_path):
    path, checkpoint = _v2_selection_anchor_clone_plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    assert plan["_receiver_v2"] is False
    assert plan["_receiver_v2_selection_anchor_clone"] is True
    model = _real_cpu_model()
    expected = {
        name: getattr(model, name)[:20].detach().clone()
        for name in (
            "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"
        )
    }
    event = apply_active_reallocation(
        model, plan, iteration=1201, event_log=tmp_path / "anchor_clone_event.json"
    )
    assert event["format"] == "i2_active_event_v2_selection_anchor_clone_v1"
    assert event["method_version"] == V2_SELECTION_ANCHOR_CLONE_METHOD
    assert event["birth_path"] == "receiver_v2_selection_anchor_raw_clone"
    assert event["gaussian_before"] == 40
    assert event["gaussian_after_release"] == 20
    assert event["gaussian_after"] == 40
    assert event["budget_conserved"] and event["alignment"]["pass"]
    assert all(event["anchor_raw_parameter_copy"].values())
    for name, value in expected.items():
        assert torch.equal(getattr(model, name)[-20:], value)
    assert model.lifecycle.source_uid[-20:].tolist() == list(range(20))
    assert torch.count_nonzero(model.xyz_gradient_accum[-20:]) == 0
    assert torch.count_nonzero(model.denom[-20:]) == 0
    assert torch.count_nonzero(model.max_radii2D[-20:]) == 0
    model.lifecycle.audit_strict(expected_count=40)
    model.observation.audit_strict(expected_uid=model.lifecycle.uid, expected_count=40)
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        state = model.optimizer.state[parameter]
        assert parameter.shape[0] == state["exp_avg"].shape[0] == state["exp_avg_sq"].shape[0] == 40


@pytest.mark.parametrize("mutation,match", [
    (lambda plan: plan["matches"].reverse(), "differs from the frozen Receiver-v2 plan"),
    (lambda plan: plan["receiver_v2_selection_anchor_clone"].update(method_version="bad"),
     "method binding"),
    (lambda plan: plan["receiver_v2_selection_anchor_clone"]["original_receiver_v2_plan"].update(
        sha256="0" * 64), "SHA256 mismatch"),
])
def test_v2_selection_anchor_clone_plan_tampering_fails_closed(tmp_path, mutation, match):
    path, _ = _v2_selection_anchor_clone_plan(tmp_path)
    _mutate_plan(path, mutation)
    with pytest.raises(ActivePilotError, match=match):
        load_active_plan(path, requested_budget=20)


def test_v2_selection_anchor_clone_missing_anchor_uid_fails_before_mutation(tmp_path):
    path, checkpoint = _v2_selection_anchor_clone_plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    model = _real_cpu_model()
    before = model._xyz.detach().clone()
    model.lifecycle.uid[0] = 999
    with pytest.raises(ActivePilotError, match="anchor UID is not live"):
        apply_active_reallocation(
            model, plan, iteration=1201, event_log=tmp_path / "missing_anchor.json"
        )
    assert torch.equal(model._xyz, before)
    assert not (tmp_path / "missing_anchor.json").exists()


def test_v2_selection_anchor_clone_parameter_copy_mismatch_fails_closed(
    tmp_path, monkeypatch
):
    path, checkpoint = _v2_selection_anchor_clone_plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    model = _real_cpu_model()
    original = model.densification_postfix

    def corrupt(*args, **kwargs):
        original(*args, **kwargs)
        model._xyz.data[-1, 0] += 1

    monkeypatch.setattr(model, "densification_postfix", corrupt)
    event_path = tmp_path / "corrupt_copy.json"
    with pytest.raises(ActivePilotError, match="budget/alignment"):
        apply_active_reallocation(model, plan, iteration=1201, event_log=event_path)
    assert not event_path.exists()


def test_receiver_v1_plan_remains_receiver_v1(tmp_path):
    path, checkpoint = _plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    assert plan["_receiver_v2"] is False
    assert plan["_receiver_v2_selection_anchor_clone"] is False


@pytest.mark.parametrize("c_enabled", [False, True])
def test_release_only_uid_reason_optimizer_count_and_checkpoint_round_trip(
    tmp_path, c_enabled, monkeypatch
):
    path, checkpoint = _plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    model = _real_cpu_model()
    if c_enabled:
        model.future_utility = GaussianFutureUtility.empty(40, torch.device("cpu"))
    expected_uid = model.lifecycle.uid[:20].clone()
    expected_optimizer = {}
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        expected_optimizer[group["name"]] = (
            model.optimizer.state[parameter]["exp_avg"][:20].clone(),
            model.optimizer.state[parameter]["exp_avg_sq"][:20].clone(),
        )

    event = apply_release_only(
        model, plan, iteration=1201, event_log=tmp_path / "release_event.json"
    )
    assert event["gaussian_before"] == 40 and event["gaussian_after"] == 20
    assert event["budget_conserved"] is False
    assert event["budget_conserving_reallocation"] is False
    assert event["death_path"] == "i2_active_release"
    assert torch.equal(model.lifecycle.uid, expected_uid)
    assert model.lifecycle.events[-1]["reason"] == "i2_active_release"
    assert model.lifecycle.events[-1]["uids"].tolist() == list(range(20, 40))
    assert model.observation.real_view_opportunities.shape[0] == 20
    if c_enabled:
        assert model.future_utility.tracked_cohort.shape[0] == 20
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        exp_avg, exp_avg_sq = expected_optimizer[group["name"]]
        assert torch.equal(model.optimizer.state[parameter]["exp_avg"], exp_avg)
        assert torch.equal(model.optimizer.state[parameter]["exp_avg_sq"], exp_avg_sq)

    # Exercise the same capture/torch.save/restore boundary used by checkpoints.
    import io
    buffer = io.BytesIO()
    torch.save(model.capture(), buffer)
    buffer.seek(0)
    restored = _real_cpu_model(20)
    restored.lifecycle = GaussianLifecycle.roots(20, torch.device("cpu"))
    restored.observation = GaussianObservation.zeros(20, torch.device("cpu"), ["train-a.jpg"])
    if c_enabled:
        restored.future_utility = GaussianFutureUtility.empty(20, torch.device("cpu"))

    def cpu_training_setup(self, _args):
        groups = [{"params": [parameter], "name": name, "lr": 0.01} for name, parameter in
                  (("xyz", self._xyz), ("f_dc", self._features_dc),
                   ("f_rest", self._features_rest), ("opacity", self._opacity),
                   ("scaling", self._scaling), ("rotation", self._rotation))]
        self.optimizer = torch.optim.Adam(groups, lr=0.0)

    monkeypatch.setattr(GaussianModel, "training_setup", cpu_training_setup)
    restored.restore(torch.load(buffer, weights_only=False), object())
    assert restored.get_xyz.shape[0] == 20
    assert torch.equal(restored.lifecycle.uid, expected_uid)
    assert restored.lifecycle.events[-1]["reason"] == "i2_active_release"
    assert restored.observation.real_view_opportunities.shape[0] == 20
    for group in restored.optimizer.param_groups:
        parameter = group["params"][0]
        assert model.optimizer.state[next(
            g["params"][0] for g in model.optimizer.param_groups if g["name"] == group["name"]
        )]["exp_avg"].shape == restored.optimizer.state[parameter]["exp_avg"].shape


def test_release_only_event_records_only_first_dropout_reference_extension(tmp_path):
    path, checkpoint = _plan(tmp_path)
    plan = load_active_plan(path, requested_budget=20, source_checkpoint=checkpoint)
    event_path = tmp_path / "release_event.json"
    apply_release_only(_real_cpu_model(), plan, iteration=1201, event_log=event_path)
    initial = json.loads(event_path.read_text(encoding="utf-8"))
    assert initial["dropout_randomness"]["initial_reference_count"] == 40
    assert initial["dropout_randomness"]["first_reference_extension"] is None

    record_release_only_reference_extension(
        event_path, iteration=1500, gaussian_count=41, effective_reference_count=41
    )
    record_release_only_reference_extension(
        event_path, iteration=1600, gaussian_count=45, effective_reference_count=45
    )
    recorded = json.loads(event_path.read_text(encoding="utf-8"))["dropout_randomness"]
    assert recorded["first_reference_extension"] == {
        "iteration": 1500,
        "gaussian_count": 41,
        "effective_reference_count": 41,
    }
    assert recorded["strict_rng_alignment_lost_at_iteration"] == 1500
