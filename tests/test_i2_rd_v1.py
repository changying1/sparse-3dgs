import hashlib
import json
from pathlib import Path

import pytest

from tools import run_i2_scene
from utils.i2_active_pilot import ActivePilotError, load_active_plan
from utils.i2_adaptive_k import (
    AdaptiveKError,
    make_rd_decision,
    rd_donor_capacity,
    rd_receiver_capacity,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path):
    return {"path": str(path), "sha256": _sha(path)}


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receiver(index, alignment):
    x = float(index * 10)
    return {
        "candidate_id": f"c{index:04d}",
        "photometric_alignment": float(alignment),
        "demand_score": float(1000 - index),
        "anchor_uid": 10000 + index,
        "anchor_xyz": [x, 0.0, 0.0],
        "new_xyz": [x + 1.0, 0.0, 0.0],
        "displacement_norm": 1.0,
        "geometry_valid": True,
        "fail_closed": False,
        "cross_view_checks": [{"training_camera_id": "train-a", "safe": True}],
        "activated_opacity": 0.05,
        "activated_scale": [0.4, 0.3, 0.2],
    }


def test_rd_receiver_capacity_is_not_truncated_at_20_or_200():
    rows = [_receiver(index, 1.0) for index in range(251)]
    audit = rd_receiver_capacity(rows)
    assert audit["geometry_nms_capacity_g"] == 251
    assert audit["audited_prefix_count"] == 251
    assert audit["receiver_capacity"] == 251
    assert audit["k_cap"] is None and audit["complete_sequence_audited"] is True


def test_rd_receiver_positive_evidence_must_be_exact_prefix():
    with pytest.raises(AdaptiveKError, match="positive evidence set differs"):
        rd_receiver_capacity([_receiver(0, 1), _receiver(1, 0), _receiver(2, 1)])


def test_rd_donor_capacity_excludes_anchors_and_keeps_complete_order():
    rows = [{"uid": uid, "birth_type": "clone", "score": 100 - uid} for uid in range(35)]
    audit = rd_donor_capacity(rows, receiver_anchor_uids={3, 7}, live_uids=range(35))
    assert audit["donor_capacity"] == 33
    assert [row["uid"] for row in audit["ranked_eligible"][:4]] == [0, 1, 2, 4]
    assert {row["uid"] for row in audit["excluded"]} == {3, 7}
    assert audit["performance_gates_used"] is False


def test_rd_donor_duplicate_uid_fails_closed():
    with pytest.raises(AdaptiveKError, match="duplicate UID"):
        rd_donor_capacity(
            [{"uid": 1, "birth_type": "clone"}, {"uid": 1, "birth_type": "split"}],
            receiver_anchor_uids=set(), live_uids={1},
        )


@pytest.mark.parametrize("r,d,k", [(31, 45, 31), (45, 31, 31), (0, 50, 0)])
def test_rd_decision_uses_only_min_r_d(r, d, k):
    receiver = {
        "receiver_capacity": r, "complete_sequence_audited": True,
        "positive_prefix_exact_match": True,
    }
    donor = {"donor_capacity": d, "coverage_complete": True}
    decision = make_rd_decision(
        scene="flower", source_checkpoint_sha256="a" * 64,
        receiver_audit=receiver, donor_audit=donor, evidence={},
    )
    assert decision["selected_k"] == k
    assert decision["policy_version"] == "RD-v1"
    assert "initialization" not in decision and "k_cap" not in decision
    assert decision["execution_route"] == (
        "control_no_intervention" if k == 0 else "adaptive_plan_available_no_training_started"
    )


def test_rd_unknown_is_not_coerced_to_k0():
    decision = make_rd_decision(
        scene="fern", source_checkpoint_sha256="b" * 64,
        receiver_audit={}, donor_audit={}, evidence={}, determined=False,
        blocker="enumeration_resource_exhausted",
    )
    assert decision["selected_k"] == "UNKNOWN"
    assert decision["execution_route"] == "blocked_no_execution"


def _rd_plan(tmp_path, k=2):
    checkpoint = _write(tmp_path / "source.pth", {"source": True})
    manifest = _write(tmp_path / "views.json", {"selected_train_images": ["train-a"]})
    table = tmp_path / "receivers.csv"
    table.write_text("candidate_id\nc0000\nc0001\n", encoding="utf-8")
    receiver_manifest = _write(
        tmp_path / "receiver_manifest.json", {"receiver_candidates": _ref(table)}
    )
    receivers = [_receiver(index, 1.0) for index in range(k)]
    receiver_audit = {
        "format": "i2_receiver_v2_audit_all_rd_v1", "test_views_used": False,
        "scene": "flower", "source_checkpoint_sha256": _sha(checkpoint),
        "receiver_manifest": _ref(receiver_manifest),
        "coverage": {"complete": True},
        "accepted_order": [row["candidate_id"] for row in receivers],
        "receiver_budget": {
            "receiver_capacity": k, "complete_sequence_audited": True,
        },
    }
    receiver_path = _write(tmp_path / "receiver_audit.json", receiver_audit)
    donors = [{"uid": index + 50, "birth_type": "clone", "rd_rank": index + 1} for index in range(k)]
    donor_audit = {
        "format": "i2_rd_v1_donor_capacity_v1", "test_views_used": False,
        "scene": "flower", "source_checkpoint_sha256": _sha(checkpoint),
        "coverage_complete": True, "performance_gates_used": False,
        "render_counterfactuals_used": False, "donor_capacity": k,
        "ranked_eligible": donors,
    }
    donor_path = _write(tmp_path / "donor_audit.json", donor_audit)
    plan = {
        "format": "i2_rd_v1_active_plan", "policy_version": "RD-v1",
        "scene": "flower", "uses_test_views": False,
        "source": {"iteration": 1201, "checkpoint": _ref(checkpoint)},
        "training_view_manifest": _ref(manifest),
        "receiver_candidate_table": _ref(table),
        "receiver_audit": _ref(receiver_path), "donor_audit": _ref(donor_path),
        "selected_k": k, "donors": donors, "receivers": receivers,
        "matches": [
            {"donor_uid": donor["uid"], "receiver_candidate_id": receiver["candidate_id"],
             "released_units": 1, "received_units": 1}
            for donor, receiver in zip(donors, receivers)
        ],
        "budget_audit": {"released_units": k, "received_units": k, "conserved": True},
        "coverage": {
            "receiver_total": k, "donor_total": k, "complete": True,
            "receiver_selected_order": [row["candidate_id"] for row in receivers],
            "donor_selected_order": [row["uid"] for row in donors],
        },
        "performance_safety_reports": None,
    }
    return _write(tmp_path / "plan.json", plan), checkpoint


def test_rd_plan_loads_without_fake_legacy_safety_reports(tmp_path):
    path, checkpoint = _rd_plan(tmp_path)
    plan = load_active_plan(path, requested_budget=2, source_checkpoint=checkpoint)
    assert plan["_rd_v1"] is True
    assert plan["_counterfactual_artifact"] is None
    assert plan["_donor_uids"] == [50, 51]
    assert len(plan["_receiver_xyz"]) == 2


def test_rd_plan_hash_tampering_fails_closed(tmp_path):
    path, checkpoint = _rd_plan(tmp_path)
    (tmp_path / "donor_audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ActivePilotError, match="SHA256 mismatch"):
        load_active_plan(path, requested_budget=2, source_checkpoint=checkpoint)


def test_rd_policy_has_no_fixed_cap_and_is_strict():
    policy = json.loads((ROOT / "configs/i2_adaptive_k_rd_v1.json").read_text(encoding="utf-8"))
    assert set(policy) == {
        "schema_version", "policy_id", "version", "budget_rule", "receiver",
        "donor", "forbidden_budget_signals",
    }
    assert policy["budget_rule"] == "K=min(R,D)"
    assert "candidate_budget_cap" not in policy


def test_complete_generator_uses_zero_as_unbounded_sentinel():
    source = (ROOT / "tools/build_i2_receivers.py").read_text(encoding="utf-8")
    assert "return rows if limit == 0 else rows[:limit]" in source
    assert "if args.max_candidates > 0" in source


@pytest.mark.parametrize("k,route", [(0, "control"), (37, "receiver_v2")])
def test_adaptive_smoke_cli_chooses_arm_from_decision(tmp_path, monkeypatch, k, route):
    scene = {"scene": "flower"}
    method = {"budget": {"requested_k": 20}}
    fingerprints = {}
    decision = tmp_path / "decision.json"
    decision.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_i2_scene, "load_configs", lambda *_a, **_k: (scene, method, fingerprints))
    monkeypatch.setattr(
        run_i2_scene, "preflight", lambda *_a, **_k: {"status": "GO", "blockers": []}
    )
    monkeypatch.setattr(
        run_i2_scene, "validate_adaptive_training_binding",
        lambda *_a, **_k: {
            "decision": decision, "selected_k": k, "route": route,
            "plan": None if k == 0 else tmp_path / "plan.json", "audit": None,
            "policy_version": "RD-v1",
        },
    )
    monkeypatch.setattr(
        run_i2_scene, "training_command",
        lambda *_a, **_k: (["python", "train.py"], tmp_path / f"smoke-{k}", 1252),
    )
    assert run_i2_scene.main([
        "--scene", "flower", "--stage", "adaptive-smoke",
        "--adaptive-decision", str(decision),
    ]) == 0


def test_paired_control_uses_decision_k_but_no_intervention_plan(tmp_path, monkeypatch):
    scene = {"scene": "flower"}
    method = {"budget": {"requested_k": 20}}
    decision = tmp_path / "decision.json"
    decision.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        run_i2_scene, "preflight", lambda *_a, **_k: {"status": "GO", "blockers": []}
    )
    monkeypatch.setattr(
        run_i2_scene, "validate_adaptive_training_binding",
        lambda *_a, **_k: {
            "decision": decision, "selected_k": 31, "route": "receiver_v2",
            "plan": tmp_path / "plan.json", "audit": None,
            "policy_version": "RD-v1",
        },
    )
    observed = {}
    def command(*_args, **kwargs):
        observed.update(kwargs)
        return ["python", "train.py"], tmp_path / "control-formal", 5000
    monkeypatch.setattr(run_i2_scene, "training_command", command)
    payload = run_i2_scene.training_stage(
        scene, method, {}, stage="formal", arm="control", execute=False,
        adaptive_decision=decision, paired_control=True,
    )
    assert payload["arm"] == "control" and payload["requested_k"] == 31
    assert observed["active_plan"] is None and observed["requested_budget"] == 31

