import importlib.util
import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adaptive", ROOT / "utils" / "i2_adaptive_k.py")
ADAPTIVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ADAPTIVE)


def receiver(candidate_id, alignment, score, xyz):
    return {
        "candidate_id": candidate_id,
        "photometric_alignment": alignment,
        "demand_score": score,
        "anchor_xyz": [0.0, 0.0, 0.0],
        "new_xyz": xyz,
        "displacement_norm": sum(v * v for v in xyz) ** 0.5,
    }


def test_audit_all_preserves_rank_and_records_nms_conflict():
    rows = [
        receiver("b", 0.2, 0.9, [2.0, 0.0, 0.0]),
        receiver("a", 0.3, 0.5, [1.0, 0.0, 0.0]),
        receiver("c", 0.2, 0.8, [2.01, 0.0, 0.0]),
    ]
    accepted, rejected = ADAPTIVE.complete_receiver_selection(
        rows, spatial_dedup_radius=0.1
    )
    assert [row["candidate_id"] for row in accepted] == ["a", "b"]
    assert rejected == [{
        "candidate_id": "c", "reason": "v2_spatial_nms",
        "conflict_candidate_id": "b",
    }]


def test_contiguous_prefix_stops_at_first_failure_even_if_later_passes():
    capacity, reason = ADAPTIVE.contiguous_pass_capacity([
        {"k": 1, "safe": True}, {"k": 2, "safe": False},
        {"k": 3, "safe": True},
    ])
    assert capacity == 1
    assert reason == "first_failed_prefix_k2"


def test_k0_is_control_decision_without_plan():
    decision = ADAPTIVE.make_decision(
        scene="synthetic", source_checkpoint_sha256="a" * 64,
        receiver_capacity=3, donor_capacity=3,
        initialization_rows=[{"k": 1, "safe": False}],
        evidence={"receiver": {"path": "r", "sha256": "b" * 64}},
        determined=True,
    )
    assert decision["selected_k"] == 0
    assert decision["execution_route"] == "control_no_intervention"
    assert "plan" not in decision
    assert decision["uses_test_views"] is False


def test_unknown_is_not_coerced_to_zero():
    decision = ADAPTIVE.make_decision(
        scene="synthetic", source_checkpoint_sha256="a" * 64,
        receiver_capacity=3, donor_capacity=1, initialization_rows=[],
        evidence={}, determined=False, blocker="scan_limit",
    )
    assert decision["selected_k"] == "UNKNOWN"
    assert decision["execution_route"] == "blocked_no_execution"


def test_fixed_k20_failure_does_not_block_smaller_safe_adaptive_prefix():
    initialization = [
        {"k": k, "safe": k < 5} for k in range(1, 6)
    ]
    decision = ADAPTIVE.make_decision(
        scene="synthetic", source_checkpoint_sha256="a" * 64,
        receiver_capacity=20, donor_capacity=20,
        initialization_rows=initialization, evidence={}, determined=True,
    )
    assert decision["selected_k"] == 4
    assert decision["initialization_contiguous_safe_capacity"] == 4
    assert decision["execution_route"] == "adaptive_plan_available_no_training_started"


def test_runtime_binding_uses_scene_config_common_source_and_prepare_artifacts():
    source = (ROOT / "tools" / "audit_i2_adaptive_k.py").read_text(encoding="utf-8")
    assert "resolve_common_source(scene_cfg, method)" in source
    assert 'parser.add_argument("--receiver-manifest", required=True)' in source
    assert 'parser.add_argument("--receiver-report", required=True)' in source
    assert "build_d0_plan(" in source and "budget=0" in source
    assert 'paths["counterfactual"]' not in source
    assert "f05_audit.json" not in source
    assert 'choices=("horns", "fern")' not in source


def test_receiver_runtime_restore_is_no_grad():
    source = (ROOT / "tools" / "audit_i2_adaptive_k.py").read_text(encoding="utf-8")
    assert "@torch.no_grad()\ndef audit_receivers" in source


def test_plan_builder_uses_requested_prefix_k():
    source = (ROOT / "tools" / "audit_i2_adaptive_k.py").read_text(encoding="utf-8")
    assert "def build_plan(k, old_plan" in source
    assert "k = len(donors)" not in source


def test_audit_output_contract_never_silently_versions_existing_directory():
    source = (ROOT / "tools" / "audit_i2_adaptive_k.py").read_text(encoding="utf-8")
    function = source[source.index("def new_output"):source.index("def pipeline_params")]
    assert "exist_ok=False" in function
    assert "with_name" not in function
    assert 'output / "adaptive_k_failure.json"' in source


def test_individual_safe_count_is_not_joint_prefix_capacity():
    source = (ROOT / "tools" / "audit_i2_adaptive_k.py").read_text(encoding="utf-8")
    assert '"individual_safe_count": sum(row["safe_to_release"] is True for row in individual)' in source


def accepted_rows_from_historical_audit(scene):
    root = ROOT / "reports" / "i2_scene_runs" / "k01_adaptive_k_v20261015T100927Z" / scene
    audit = json.loads((root / "receiver_audit_all.json").read_text(encoding="utf-8"))
    evidence_by_id = {
        row["candidate_id"]: row["geometry_evidence"]
        for row in audit["rows"] if row.get("nms_status") == "ACCEPT"
    }
    return root, audit, [evidence_by_id[candidate_id] for candidate_id in audit["accepted_order"]]


def test_first_twenty_positive_evidence_count_and_exact_prefix():
    accepted = [receiver(str(i), 1.0 if i < 7 else -0.1, 1.0, [i + 1.0, 0, 0]) for i in range(25)]
    result = ADAPTIVE.receiver_evidence_budget(accepted, k_cap=20)
    assert result["geometry_nms_capacity_g"] == 25
    assert result["receiver_evidence_budget_r"] == 7
    assert result["positive_evidence_ids"] == result["accepted_prefix_ids"][:7]


def test_positive_members_must_equal_actual_receiver_prefix():
    accepted = [
        receiver("a", 0.2, 1.0, [1, 0, 0]),
        receiver("b", -0.1, 1.0, [2, 0, 0]),
        receiver("c", 0.1, 1.0, [3, 0, 0]),
    ]
    with pytest.raises(ADAPTIVE.AdaptiveKError, match="positive evidence set"):
        ADAPTIVE.receiver_evidence_budget(accepted)


def test_receiver_budget_when_g_is_less_than_cap():
    accepted = [receiver(str(i), 0.1, 1.0, [i + 1.0, 0, 0]) for i in range(4)]
    result = ADAPTIVE.receiver_evidence_budget(accepted)
    assert result["audited_prefix_count"] == result["receiver_evidence_budget_r"] == 4


def test_more_than_twenty_positive_candidates_obey_uniform_cap():
    accepted = [receiver(str(i), 0.1, 1.0, [i + 1.0, 0, 0]) for i in range(30)]
    assert ADAPTIVE.receiver_evidence_budget(accepted)["receiver_evidence_budget_r"] == 20


def test_receiver_budget_zero_is_valid_control_evidence():
    accepted = [receiver("a", 0.0, 1.0, [1, 0, 0]), receiver("b", -0.1, 1.0, [2, 0, 0])]
    assert ADAPTIVE.receiver_evidence_budget(accepted)["receiver_evidence_budget_r"] == 0


@pytest.mark.parametrize("label", ["donor", "initialization"])
def test_contiguous_safety_stops_at_intermediate_failure(label):
    rows = [{"k": 1, "safe": True}, {"k": 2, "safe": True}, {"k": 3, "safe": False}, {"k": 4, "safe": True}]
    capacity, reason = ADAPTIVE.contiguous_pass_capacity(rows)
    assert (capacity, reason) == (2, "first_failed_prefix_k3"), label


@pytest.mark.parametrize("alignment", [math.nan, math.inf, -math.inf, None])
def test_invalid_alignment_is_rejected(alignment):
    with pytest.raises(ADAPTIVE.AdaptiveKError, match="evidence"):
        ADAPTIVE.complete_receiver_selection(
            [receiver("bad", alignment, 1.0, [1, 0, 0])], spatial_dedup_radius=0.1
        )


def test_hash_mismatch_is_rejected(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ADAPTIVE.AdaptiveKError, match="hash mismatch"):
        ADAPTIVE.require_artifact_hash({"path": str(artifact), "sha256": "0" * 64})


@pytest.mark.parametrize("scene,expected_g,expected_r", [("horns", 32, 16), ("fern", 48, 13)])
def test_historical_r0_decisions_are_read_only_compatible(scene, expected_g, expected_r):
    root, _, accepted = accepted_rows_from_historical_audit(scene)
    result = ADAPTIVE.receiver_evidence_budget(accepted)
    plan = json.loads((root / "plans" / f"adaptive_k{expected_r}_plan.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (root / "plans" / f"receiver_v2_geometry_evidence_k{expected_r}.json").read_text(encoding="utf-8")
    )
    plan_ids = [row["receiver_candidate_id"] for row in plan["matches"]]
    evidence_ids = [row["candidate_id"] for row in evidence["rows"]]
    assert result["geometry_nms_capacity_g"] == expected_g
    assert result["receiver_evidence_budget_r"] == expected_r
    assert plan_ids == evidence_ids == result["positive_evidence_ids"]


def test_fixed_method_identities_are_unchanged():
    expected = {
        "i2_method_v2.json": "d7a48bb3c3ea3679b2e72ef186f9e972c4a52fd607670cfce38a2a02e64c2f2d",
        "i2_method_adaptive_k21_v1.json": "37b540c08b3e75d4de7bfadbf7a665cce96e2e2e6132e7d90c11b029b37d167d",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest() == digest
