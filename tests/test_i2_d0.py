import hashlib

import pytest

from utils.i2_d0_dry_run import D0AuditError, build_d0_plan, select_donors


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _donor(uid, *, age=200, opportunities=10, retained=8, supported=1,
           rate=0.1, coverage=0.3, low=True, birth_type="split"):
    return {
        "uid": str(uid), "birth_type": birth_type, "age": str(age),
        "real_view_opportunities": str(opportunities),
        "dropout_retained_hits": str(retained), "screen_supported_hits": str(supported),
        "screen_support_rate": str(rate), "real_view_coverage_rate": str(coverage),
        "low_screen_support_candidate": str(low), "observation_status": "low_screen_support",
    }


def _inputs(tmp_path, *, c_status="missing_c_checkpoint_extension"):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    payload = tmp_path / "view-evidence.npz"
    payload.write_bytes(b"real evidence arrays")
    evidence = tmp_path / "view-evidence.json"
    checkpoint_sha = _sha(checkpoint)
    evidence.write_text(__import__("json").dumps({
        "format": "i2_receiver_view_evidence_v1", "training_camera_id": "train-a.jpg",
        "source_checkpoint_sha256": checkpoint_sha, "generation_rule_id": "rule-v1",
        "payload": {"path": str(payload), "sha256": _sha(payload)},
    }), encoding="utf-8")
    manifest_path = tmp_path / "d.json"
    manifest = {
        "_manifest_path": str(manifest_path), "_verified_source_checkpoint": str(checkpoint),
        "source_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "diagnostic_status": {"c": c_status},
        "receiver_protocol": {
            "format": "i2_receiver_protocol_v1", "source_checkpoint_sha256": checkpoint_sha,
            "training_camera_ids": ["train-a.jpg"], "generation_rule_id": "rule-v1",
            "source_code_fingerprint": "code-sha", "score_normalization": "p95-clipped",
            "spatial_dedup_radius": 0.01,
            "max_anchor_pixel_distance": 12.0,
        },
    }
    receiver = {
        "candidate_id": "defect-a", "demand_basis": "structural_defect",
        "demand_score": "0.8", "requested_units": "1", "training_camera_id": "train-a.jpg",
        "pixel_x0": "0", "pixel_y0": "0", "pixel_x1": "8", "pixel_y1": "8",
        "pixel_u": "4", "pixel_v": "4", "reprojection_error_px": "1.0",
        "world_x": "1", "world_y": "2", "world_z": "3",
        "anchor_uid": "99",
        "anchor_type": "rendered_depth_unprojection", "depth_valid": "true",
        "geometry_valid": "true", "local_gaussian_count": "3", "local_density_radius": "0.1",
        "evidence_path": str(evidence), "evidence_sha256": _sha(evidence),
        "source_checkpoint_sha256": checkpoint_sha, "generation_rule_id": "rule-v1",
    }
    return manifest, receiver


def test_d0_fixed_budget_stable_tie_break_and_growth_only_protections():
    rows = [
        _donor(9, rate=0.1), _donor(3, rate=0.1), _donor(8, birth_type="root"),
        _donor(1, age=99), _donor(2, opportunities=0, retained=0, supported=0),
        _donor(4, retained=0, supported=0), _donor(5, low=False),
    ]
    result = select_donors(rows, budget=1)
    assert [row["uid"] for row in result["selected"]] == [3]
    assert [row["uid"] for row in result["controls"]] == [9]
    assert result["protected_counts"]["protected_root_or_unknown_birth"] == 1


def test_d0_missing_receiver_evidence_fails_closed_without_inventing_matches(tmp_path):
    manifest, _ = _inputs(tmp_path)
    plan = build_d0_plan(manifest, [_donor(1), _donor(2)], [], None, budget=1)
    assert plan["receivers"]["status"] == "blocked_missing_receiver_evidence"
    assert plan["matches"] == []
    assert plan["safety_readiness"]["b1_only_exploratory"] is False


def test_d0_verified_receiver_matches_exact_budget_and_separates_readiness(tmp_path):
    manifest, receiver = _inputs(tmp_path)
    plan = build_d0_plan(manifest, [_donor(1), _donor(2)], [], [receiver], budget=1)
    assert len(plan["matches"]) == 1
    assert plan["budget_audit"]["conserved"] is True
    assert plan["safety_readiness"] == {
        "b1_only_exploratory": True, "c_informed": False,
        "c_validation": "c_status_unavailable",
    }


def test_d0_fake_receiver_reference_and_hash_fail_closed(tmp_path):
    manifest, receiver = _inputs(tmp_path)
    receiver["evidence_path"] = "fake:does-not-exist"
    with pytest.raises(D0AuditError, match="missing or has a SHA256 mismatch"):
        build_d0_plan(manifest, [_donor(1), _donor(2)], [], [receiver], budget=1)
    manifest, receiver = _inputs(tmp_path)
    receiver["evidence_sha256"] = "0" * 64
    with pytest.raises(D0AuditError, match="SHA256 mismatch"):
        build_d0_plan(manifest, [_donor(1), _donor(2)], [], [receiver], budget=1)


@pytest.mark.parametrize("value", ["nan", "abc", "inf"])
def test_d0_invalid_c_prediction_never_triggers_c_informed(tmp_path, value):
    manifest, receiver = _inputs(tmp_path, c_status="available")
    prediction = tmp_path / "prediction.json"
    prediction.write_text("{}", encoding="utf-8")
    manifest["c_prediction"] = {
        "artifact": {"path": str(prediction), "sha256": _sha(prediction)},
        "prediction_iteration": 1201, "feature_cutoff_iteration": 1200,
    }
    c_rows = [{"uid": "1", "predicted_future_utility": value, "prediction_iteration": "1201"}]
    plan = build_d0_plan(manifest, [_donor(1), _donor(2)], c_rows, [receiver], budget=1)
    assert plan["strategy"] == "b1_only_exploratory"
    assert plan["safety_readiness"]["c_informed"] is False


def test_d0_c_prediction_requires_uid_time_artifact_and_no_future_outcome(tmp_path):
    manifest, receiver = _inputs(tmp_path, c_status="available")
    prediction = tmp_path / "prediction.json"
    prediction.write_text('{"model":"frozen"}', encoding="utf-8")
    manifest["c_prediction"] = {
        "artifact": {"path": str(prediction), "sha256": _sha(prediction)},
        "prediction_iteration": 1201, "feature_cutoff_iteration": 1200,
    }
    row = {"uid": "1", "predicted_future_utility": "0.25", "prediction_iteration": "1201"}
    plan = build_d0_plan(manifest, [_donor(1), _donor(2)], [row], [receiver], budget=1)
    assert plan["strategy"] == "c_future_utility_informed"
    assert plan["safety_readiness"]["c_informed"] is True
    row["death_reason"] = "training_prune"
    plan = build_d0_plan(manifest, [_donor(1), _donor(2)], [row], [receiver], budget=1)
    assert plan["strategy"] == "b1_only_exploratory"


def test_d0_receiver_rejects_bad_geometry_and_spatial_duplicates(tmp_path):
    manifest, receiver = _inputs(tmp_path)
    with pytest.raises(D0AuditError, match="depth/geometry"):
        build_d0_plan(manifest, [_donor(1), _donor(2)], [], [dict(receiver, depth_valid="false")], budget=1)
    duplicate = dict(
        receiver, candidate_id="defect-b", pixel_x0="20", pixel_x1="28", pixel_u="24"
    )
    with pytest.raises(D0AuditError, match="duplicates"):
        build_d0_plan(manifest, [_donor(1), _donor(2)], [], [receiver, duplicate], budget=1)
