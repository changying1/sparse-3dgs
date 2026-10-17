import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import utils.i2_scene_pipeline as pipeline
from utils.i2_scene_pipeline import (
    PipelineError,
    adaptive_prepare_command,
    build_adaptive_evidence_plan,
    build_prepare_plan,
    execute_commands,
    load_configs,
    preflight,
    recover_prepare_commands,
    register_source_binding,
    resolve_common_source,
    resolve_training_pdc,
    script_cli_options,
    training_command,
    validate_external_training_binding,
    validate_adaptive_training_binding,
    validate_adaptive_smoke_metadata,
    validate_initialization_gate,
    validate_method_config,
    validate_prepare_contracts,
    validate_scene_config,
)
from utils.i2_active_pilot import ActivePilotError, validate_i2_active_configuration
from tools import run_i2_scene


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, with_source=False, test_count=2,
             scene_name="demo", output_namespace="i2_demo"):
    dataset = tmp_path / "data" / scene_name
    images = dataset / "images"
    images.mkdir(parents=True)
    train = ["train0.jpg", "train1.jpg", "train2.jpg"]
    test = [f"test{i}.jpg" for i in range(test_count)]
    for index, name in enumerate(train + test):
        (images / name).write_bytes(f"image-{index}".encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "LLFF", "scene": scene_name, "view_count": 3,
        "sorted_all_images": sorted(train + test), "selected_train_images": train,
        "pdc_initialization_images": train, "test_images": test,
    }))
    pdc = tmp_path / f"{scene_name}.ply"; pdc.write_bytes(b"ply\nend_header\n")
    cache = tmp_path / "cache"; cache.mkdir()
    records = []
    for name in train:
        depth = cache / f"{name}.npy"; depth.write_bytes(name.encode())
        records.append({
            "image_name": name, "source_image": name,
            "source_sha256": _sha(images / name), "depth_file": depth.name,
            "depth_sha256": _sha(depth),
        })
    cache_meta = cache / "oe_depth_cache.json"
    cache_meta.write_text(json.dumps({
        "source_view_manifest_sha256": _sha(manifest),
        "selected_train_images": train, "records": records,
    }))
    checkpoint = tmp_path / "source.pth"
    if with_source:
        checkpoint.write_bytes(b"source")
    scene = {
        "schema_version": "i2-scene-config-v1", "scene": scene_name,
        "dataset_root": str(dataset),
        "view_manifest": {"path": str(manifest), "sha256": _sha(manifest)},
        "pdc": {"path": str(pdc), "sha256": _sha(pdc)},
        "depth_cache": {"path": str(cache), "metadata_sha256": _sha(cache_meta)},
        "split": {"train_images": train, "test_images": test},
        "common_source": {
            "checkpoint": str(checkpoint) if with_source else None,
            "sha256": _sha(checkpoint) if with_source else None,
        },
        "output_namespace": output_namespace,
    }
    method = json.loads((ROOT / "configs/i2_method_v2.json").read_text())
    fingerprints = {"scene_config_sha256": "a" * 64, "method_config_sha256": "b" * 64}
    return scene, method, fingerprints


def _install_training_pdc(tmp_path, scene, method, *, suffix="Demo_FrozenPDC_v1"):
    checkpoint = Path(scene["common_source"]["checkpoint"])
    cfg_path = checkpoint.parent / "cfg_args"
    cfg_path.write_text(
        "Namespace("
        f"source_path={scene['dataset_root']!r}, "
        f"dataset_name={method['training']['dataset_name']!r}, "
        f"suffix={suffix!r})",
        encoding="utf-8",
    )
    frozen_pdc = Path(scene["pdc"]["path"])
    (checkpoint.parent / "input.ply").write_bytes(frozen_pdc.read_bytes())
    runtime_suffix = (
        f"{scene['output_namespace']}_source" if suffix is None else suffix
    )
    actual = (
        tmp_path / "keypoints_to_3d"
        / f"{method['training']['dataset_name']}_{runtime_suffix}"
        / f"{scene['scene']}_keypoints_to_3d.ply"
    )
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_bytes(frozen_pdc.read_bytes())
    return actual, cfg_path


def test_repository_configs_separate_scene_and_method():
    fern, method, fingerprints = load_configs("fern")
    horns, same_method, horns_fingerprints = load_configs("horns")
    assert fern["scene"] == "fern" and horns["scene"] == "horns"
    assert method == same_method
    assert fingerprints["method_config_sha256"] == horns_fingerprints["method_config_sha256"]
    assert len(fern["split"]["test_images"]) == 3
    assert len(horns["split"]["test_images"]) == 8


def test_all_eight_scene_configs_share_one_method_and_have_no_budget_override():
    scenes = ("fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex")
    methods = []
    for name in scenes:
        scene, method, _ = load_configs(name)
        assert scene["scene"] == name
        assert "budget" not in scene
        methods.append(method)
    assert all(method == methods[0] for method in methods[1:])


def test_scene_config_rejects_method_override(tmp_path):
    scene, _, _ = _fixture(tmp_path)
    scene["max_projection_condition"] = 99
    with pytest.raises(PipelineError, match="keys mismatch"):
        validate_scene_config(scene)


def test_scene_config_rejects_cross_split_contamination(tmp_path):
    scene, _, _ = _fixture(tmp_path)
    scene["split"]["test_images"][0] = scene["split"]["train_images"][0]
    with pytest.raises(PipelineError, match="disjoint"):
        validate_scene_config(scene)


def test_method_config_rejects_receiver_threshold_change(tmp_path):
    _, method, _ = _fixture(tmp_path)
    method["receiver_v2"]["activated_opacity"] = 0.06
    with pytest.raises(PipelineError, match="frozen numerical contract"):
        validate_method_config(method)


def test_preflight_reports_source_required_and_actual_test_count(tmp_path):
    scene, method, fingerprints = _fixture(tmp_path, test_count=5)
    result = preflight(scene, method, fingerprints, audit_checkpoint=False)
    assert result["status"] == "SOURCE_REQUIRED"
    assert result["test_view_count"] == 5
    assert result["next_stage"]["stage"] == "source"


def test_preflight_reuses_hash_bound_source_without_mutation(tmp_path):
    scene, method, fingerprints = _fixture(tmp_path, with_source=True)
    result = preflight(scene, method, fingerprints, audit_checkpoint=False)
    assert result["status"] == "GO"
    assert result["source_status"] == "SOURCE_REUSABLE"
    assert result["next_stage"]["stage"] == "prepare"


def test_preflight_fails_closed_on_hash_mismatch(tmp_path):
    scene, method, fingerprints = _fixture(tmp_path)
    scene["pdc"]["sha256"] = "0" * 64
    result = preflight(scene, method, fingerprints, audit_checkpoint=False)
    assert result["status"] == "NO_GO"
    assert any(item.startswith("HASH_MISMATCH_PDC") for item in result["blockers"])


def test_prepare_plan_uses_shared_tools_and_no_test_views(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    plan = build_prepare_plan(scene, method)
    tools = [Path(entry["argv"][2]).name for entry in plan["commands"]]
    assert tools == [
        "analyze_i2_b1c.py", "build_i2_receivers.py", "build_i2_d0_dry_run.py",
        "audit_i2_donor_counterfactual.py", "build_i2_receiver_v2.py",
        "audit_i2_receiver_v2_initialization.py",
    ]
    flattened = json.dumps(plan)
    assert not any(name in flattened for name in scene["split"]["test_images"])
    assert plan["requested_k"] == method["budget"]["requested_k"]
    init = plan["commands"][-1]["argv"]
    output = Path(init[init.index("--output") + 1])
    assert output.name == "initialization_audit.json"
    assert output.parent.name == "initialization"


def test_all_six_prepare_cli_contracts_match_real_argparse_and_chain(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    plan = build_prepare_plan(scene, method)
    expected = {
        "b1": {"--checkpoint", "--output-dir", "--bootstrap-reps", "--seed"},
        "receiver_candidates": {
            "--checkpoint", "--cfg", "--d_manifest", "--output", "--block_size",
            "--per_view_limit", "--max_candidates", "--min_alpha",
            "--max_reprojection_error", "--spatial_nms_fraction",
            "--density_radius_fraction",
        },
        "d0": {
            "--manifest", "--output-dir", "--budget", "--min-age",
            "--min-opportunities", "--exclude-uid",
        },
        "donor_counterfactual": {
            "--checkpoint", "--cfg", "--plans", "--output",
            "--max_mean_abs_rgb_delta", "--max_psnr_drop_db",
        },
        "receiver_v2": {
            "--checkpoint", "--cfg", "--v1_plan", "--receiver_report", "--output",
            "--max_condition", "--displacement_factor", "--opacity",
            "--footprint_pixels", "--budget",
        },
        "initialization_audit": {
            "--checkpoint", "--cfg", "--v1_plan", "--v2_plan", "--v2_evidence",
            "--output", "--max_target_l1_increase", "--max_outside_l1_increase",
            "--max_global_l1_increase", "--budget",
        },
    }
    assert [row["name"] for row in plan["contracts"]] == list(expected)
    for contract in plan["contracts"]:
        assert set(contract["accepted_cli_options"]) == expected[contract["name"]]
        assert script_cli_options(contract["script"]) == expected[contract["name"]]
        assert all(item["type"] in {"file", "directory"} for item in contract["inputs"])
        assert all(item["type"] in {"file", "directory"} for item in contract["outputs"])
    validate_prepare_contracts(plan["commands"], plan["contracts"])
    producer_outputs = {
        row["name"]: {(item["path"], item["type"]) for item in row["outputs"]}
        for row in plan["contracts"]
    }
    for contract in plan["contracts"]:
        for item in contract["inputs"]:
            if item["produced_by"]:
                assert (item["path"], item["type"]) in producer_outputs[item["produced_by"]]


def test_execute_commands_rejects_wrong_initialization_output_path(tmp_path, monkeypatch):
    script = tmp_path / "audit.py"
    script.write_text("# placeholder\n", encoding="utf-8")
    wrong = tmp_path / "prepare" / "initialization_audit.json"
    expected = tmp_path / "prepare" / "initialization" / "initialization_audit.json"
    command = {"name": "initialization_audit", "argv": ["python", "-u", str(script)]}
    contract = {
        "name": "initialization_audit", "script": str(script),
        "accepted_cli_options": [], "inputs": [],
        "outputs": [{"path": str(expected), "type": "file"}],
        "framework_creates_output_parent": True,
    }
    monkeypatch.setattr(pipeline, "run_logged", lambda command, log: wrong.write_text("{}"))
    with pytest.raises(PipelineError, match="did not produce contracted file"):
        execute_commands([command], tmp_path / "prepare", [contract])


def test_failed_prepare_recovery_cannot_bypass_final_audit(tmp_path, monkeypatch):
    root = tmp_path / "prepare"
    root.mkdir()
    original_manifest = root / "stage_manifest.json"
    original_manifest.write_text(json.dumps({"status": "FAILED", "error": "old failure"}))
    reused_output = root / "b1" / "done.json"
    reused_output.parent.mkdir()
    reused_output.write_text("{}")
    failed_output = root / "receiver_v2" / "plan.json"
    failed_output.parent.mkdir()
    failed_output.write_text("old failed plan")
    scripts = []
    for name in ("b1", "receiver_v2", "initialization_audit"):
        script = tmp_path / f"{name}.py"
        script.write_text("# test tool\n")
        scripts.append(script)
    init_output = root / "initialization" / "audit.json"
    commands = [
        {"name": name, "argv": ["python", "-u", str(script)]}
        for name, script in zip(("b1", "receiver_v2", "initialization_audit"), scripts)
    ]
    contracts = [
        {"name": "b1", "script": str(scripts[0]), "accepted_cli_options": [],
         "inputs": [], "outputs": [{"path": str(reused_output), "type": "file"}],
         "framework_creates_output_parent": False},
        {"name": "receiver_v2", "script": str(scripts[1]), "accepted_cli_options": [],
         "inputs": [{"path": str(reused_output), "type": "file", "produced_by": "b1"}],
         "outputs": [{"path": str(failed_output), "type": "file"}],
         "framework_creates_output_parent": False},
        {"name": "initialization_audit", "script": str(scripts[2]),
         "accepted_cli_options": [],
         "inputs": [{"path": str(failed_output), "type": "file", "produced_by": "receiver_v2"}],
         "outputs": [{"path": str(init_output), "type": "file"}],
         "framework_creates_output_parent": True},
    ]

    def fake_run(command, log):
        if Path(command[2]).stem == "receiver_v2":
            failed_output.parent.mkdir(parents=True, exist_ok=True)
            failed_output.write_text("new plan")
        else:
            init_output.write_text(json.dumps({"status": "GO"}))

    monkeypatch.setattr(pipeline, "run_logged", fake_run)
    with pytest.raises(PipelineError, match="audit still failed"):
        recover_prepare_commands(
            commands, root, contracts, start_stage="receiver_v2",
            final_validator=lambda: (_ for _ in ()).throw(PipelineError("audit still failed")),
        )
    assert json.loads(original_manifest.read_text())["status"] == "FAILED"
    recovery_manifests = list((root / "recovery").glob("*/recovery_manifest.json"))
    assert len(recovery_manifests) == 1
    recovery = json.loads(recovery_manifests[0].read_text())
    assert recovery["status"] == "FAILED"
    assert [row["stage"] for row in recovery["reused_stages"]] == ["b1"]
    assert any("failed_artifacts" in row["path"] for row in recovery["backed_up_failed_artifacts"])


def _write_gate_artifacts(scene, method):
    paths = pipeline.prepare_paths(scene)
    k = method["budget"]["requested_k"]
    checkpoint = Path(scene["common_source"]["checkpoint"])
    checkpoint_sha = _sha(checkpoint)
    paths["counterfactual"].mkdir(parents=True)
    paths["receiver_v2"].mkdir(parents=True)
    paths["initialization"].mkdir(parents=True)
    joint_path = paths["counterfactual"] / f"donor_joint_deletion_counterfactual_k{k}.json"
    donor_uids = list(range(k))
    joint_path.write_text(json.dumps({
        "format": "i2_joint_deletion_counterfactual_v1", "source_iteration": 1201,
        "source_checkpoint_sha256": checkpoint_sha, "test_views_used": False,
        "donor_uids": donor_uids, "safe_to_release": True,
    }))
    joint_ref = {"format": "i2_joint_deletion_counterfactual_v1",
                 "artifact": {"path": str(joint_path), "sha256": _sha(joint_path)}}
    v1_path = paths["counterfactual"] / f"{paths['d0'].name}_active_plan.json"
    v1 = {
        "donors": {"selected": [{"uid": uid} for uid in donor_uids]},
        "joint_deletion_counterfactual": joint_ref,
        "safety_readiness": {"joint_deletion_counterfactual": True},
        "active_intervention_ready": True,
    }
    v1_path.write_text(json.dumps(v1))
    evidence = paths["receiver_v2"] / "receiver_v2_geometry_evidence.json"
    evidence.write_text("{}")
    v2_path = paths["receiver_v2"] / f"i2_d1_receiver_v2_k{k}_plan.json"
    v2_path.write_text(json.dumps({"joint_deletion_counterfactual": joint_ref}))
    audit_path = paths["initialization"] / "initialization_audit.json"
    audit = {
        "status": "GO", "test_views_used": False,
        "source_checkpoint_sha256": checkpoint_sha,
        "receiver_v1_plan_sha256": _sha(v1_path),
        "receiver_v2_plan_sha256": _sha(v2_path),
        "receiver_v2_evidence_sha256": _sha(evidence),
        "gates": {
            "receiver_v2_changes_target_regions": True,
            "target_region_not_worse_than_release": True,
            "outside_region_no_obvious_degradation": True,
            "global_no_obvious_degradation": True,
            "budget_and_alignment_pass": True,
        },
    }
    audit_path.write_text(json.dumps(audit))
    return audit_path, audit, v1_path, v2_path


def test_initialization_gate_checks_all_gates_and_source_plan_hashes(tmp_path, monkeypatch):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    audit_path, audit, _, _ = _write_gate_artifacts(scene, method)
    assert validate_initialization_gate(scene, method)[0] == audit_path
    audit["gates"]["budget_and_alignment_pass"] = False
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(PipelineError, match="failed or malformed safety gate"):
        validate_initialization_gate(scene, method)
    audit["gates"]["budget_and_alignment_pass"] = True
    audit["receiver_v2_plan_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(PipelineError, match="hash binding"):
        validate_initialization_gate(scene, method)


def test_final_plan_must_preserve_joint_not_only_single_point_safety(tmp_path, monkeypatch):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    audit_path, audit, _, v2_path = _write_gate_artifacts(scene, method)
    v2_path.write_text(json.dumps({"donor_counterfactual": {"rows": []}}))
    audit["receiver_v2_plan_sha256"] = _sha(v2_path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(PipelineError, match="did not preserve the joint"):
        validate_initialization_gate(scene, method)


def test_source_binding_records_provenance_and_hash_fails_closed(tmp_path, monkeypatch):
    scene, method, fingerprints = _fixture(tmp_path, with_source=False)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    checkpoint = tmp_path / "generated" / "chkpnt1201.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"audited source")
    audit = {"iteration": 1201, "uid_unique": True, "observation_rows": {"uid": 7}}
    binding_path, binding = register_source_binding(scene, method, checkpoint, audit)
    assert binding_path.is_file()
    assert binding["scene"] == scene["scene"]
    assert binding["iteration"] == 1201
    assert binding["a0_status"] == binding["b0_status"] == "PASS"
    assert binding["checkpoint_sha256"] == _sha(checkpoint)
    assert set(binding["input_sources"]) == {"dataset_root", "view_manifest", "pdc", "depth_cache"}
    resolved, digest, used_binding = resolve_common_source(scene, method)
    assert (resolved, digest, used_binding) == (checkpoint.resolve(), _sha(checkpoint), binding_path)
    assert preflight(scene, method, fingerprints, audit_checkpoint=False)["status"] == "GO"
    checkpoint.write_bytes(b"different source")
    failed = preflight(scene, method, fingerprints, audit_checkpoint=False)
    assert failed["status"] == "NO_GO"
    assert any(item.startswith("HASH_MISMATCH_COMMON_SOURCE") for item in failed["blockers"])


def test_frozen_config_source_cannot_be_rebound(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    with pytest.raises(PipelineError, match="must not be rebound"):
        register_source_binding(
            scene, method, Path(scene["common_source"]["checkpoint"]),
            {"iteration": 1201, "uid_unique": True, "observation_rows": {"uid": 1}},
        )


def test_evaluation_metadata_uses_runtime_test_view_count():
    source = (ROOT / "tools/evaluate_i2_5k.py").read_text(encoding="utf-8")
    assert "eight test views" not in source
    assert '"test_view_count": test_view_count' in source
    assert 'f"{test_view_count} test views' in source


@pytest.mark.parametrize("arm,flag", [
    ("control", None), ("release", "--enable_i2_release_only"),
    ("receiver_v2", "--enable_i2_active_pilot"),
])
def test_formal_commands_share_protocol_and_arm_only_changes_intervention(tmp_path, monkeypatch, arm, flag):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _install_training_pdc(tmp_path, scene, method)
    command, output, iteration = training_command(scene, method, stage="formal", arm=arm)
    assert iteration == 5000
    assert "--enable_oe_structural_loss" not in command
    assert "--enable_i2_future_utility_tracking" not in command
    assert command[command.index("--suffix") + 1] == "Demo_FrozenPDC_v1"
    assert (flag in command) if flag else not any("active_pilot" in x or "release_only" in x for x in command)
    assert scene["output_namespace"] in output.name


def test_scene_namespaces_prevent_uid_and_output_cross_contamination(tmp_path, monkeypatch):
    first, method, _ = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _install_training_pdc(tmp_path, first, method)
    second = copy.deepcopy(first)
    second["output_namespace"] = "i2_other"
    first_output = training_command(first, method, stage="formal", arm="control")[1]
    second_output = training_command(second, method, stage="formal", arm="control")[1]
    assert first_output != second_output
    assert first["common_source"]["checkpoint"] == second["common_source"]["checkpoint"]


def test_fern_smoke_and_all_formal_arms_use_real_source_suffix_and_pdc():
    scene, method, _ = load_configs("fern")
    audit = resolve_training_pdc(scene, method)
    assert audit["suffix"] == "Fern_3v_FrozenPDC_v2"
    assert audit["path"] == (
        ROOT / "keypoints_to_3d/LLFF_Fern_3v_FrozenPDC_v2/fern_keypoints_to_3d.ply"
    ).resolve()
    assert _sha(audit["path"]) == scene["pdc"]["sha256"]
    commands = [training_command(scene, method, stage="smoke", arm="receiver_v2")[0]]
    commands.extend(
        training_command(scene, method, stage="formal", arm=arm)[0]
        for arm in ("control", "release", "receiver_v2")
    )
    assert {
        command[command.index("--suffix") + 1] for command in commands
    } == {"Fern_3v_FrozenPDC_v2"}


@pytest.mark.parametrize("scene_name,namespace,suffix", [
    ("legal_one", "i2_legal_one", "LegalOne_FrozenPDC"),
    ("legal_two", "i2_legal_two", "LegalTwo-FrozenPDC-v2"),
])
def test_legal_source_suffix_is_preserved(
    tmp_path, monkeypatch, scene_name, namespace, suffix
):
    scene, method, _ = _fixture(
        tmp_path, with_source=True, scene_name=scene_name,
        output_namespace=namespace,
    )
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _install_training_pdc(tmp_path, scene, method, suffix=suffix)
    command = training_command(scene, method, stage="smoke", arm="control")[0]
    assert command[command.index("--suffix") + 1] == suffix
    assert "Fern_3v_FrozenPDC_v2" not in command


@pytest.mark.parametrize("scene_name,namespace", [
    ("legacy_one", "i2_legacy_one"),
    ("legacy_two", "i2_legacy_two"),
])
def test_legacy_none_suffix_uses_canonical_source_namespace(
    tmp_path, monkeypatch, scene_name, namespace
):
    scene, method, _ = _fixture(
        tmp_path, with_source=True, scene_name=scene_name,
        output_namespace=namespace,
    )
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    actual, _ = _install_training_pdc(tmp_path, scene, method, suffix=None)

    audit = resolve_training_pdc(scene, method)

    assert audit["suffix"] == f"{namespace}_source"
    assert audit["path"] == actual.resolve()
    assert _sha(audit["path"]) == _sha(audit["frozen_pdc"])
    assert _sha(audit["source_input_ply"]) == _sha(audit["frozen_pdc"])
    command = training_command(scene, method, stage="smoke", arm="control")[0]
    assert command[command.index("--suffix") + 1] == f"{namespace}_source"


@pytest.mark.parametrize("scene_name,namespace", [
    ("missing_one", "i2_missing_one"),
    ("missing_two", "i2_missing_two"),
])
def test_training_pdc_missing_is_no_go_before_output(
    tmp_path, monkeypatch, scene_name, namespace
):
    scene, method, fingerprints = _fixture(
        tmp_path, with_source=True, scene_name=scene_name,
        output_namespace=namespace,
    )
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    actual, _ = _install_training_pdc(tmp_path, scene, method)
    actual.unlink()
    result = preflight(scene, method, fingerprints)
    assert result["status"] == "NO_GO"
    assert any("Scene PDC is missing" in item for item in result["blockers"])
    assert any(str(actual.resolve()) in item for item in result["blockers"])


@pytest.mark.parametrize("scene_name,namespace", [
    ("mismatch_one", "i2_mismatch_one"),
    ("mismatch_two", "i2_mismatch_two"),
])
@pytest.mark.parametrize("corrupt_target,expected_label", [
    ("runtime", "Scene PDC SHA-256 differs"),
    ("source", "common-source input.ply SHA-256 differs"),
])
def test_training_pdc_hash_mismatch_is_no_go_before_output(
    tmp_path, monkeypatch, scene_name, namespace, corrupt_target, expected_label
):
    scene, method, fingerprints = _fixture(
        tmp_path, with_source=True, scene_name=scene_name,
        output_namespace=namespace,
    )
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    actual, _ = _install_training_pdc(tmp_path, scene, method)
    corrupt_path = (
        actual if corrupt_target == "runtime"
        else Path(scene["common_source"]["checkpoint"]).parent / "input.ply"
    )
    corrupt_path.write_bytes(b"different")
    result = preflight(scene, method, fingerprints)
    assert result["status"] == "NO_GO"
    assert any(expected_label in item for item in result["blockers"])


@pytest.mark.parametrize("scene_name,namespace", [
    ("invalid_one", "i2_invalid_one"),
    ("invalid_two", "i2_invalid_two"),
])
@pytest.mark.parametrize("suffix_field", ["", ", suffix='../escape'"])
def test_common_source_cfg_missing_or_invalid_suffix_is_no_go(
    tmp_path, monkeypatch, scene_name, namespace, suffix_field
):
    scene, method, fingerprints = _fixture(
        tmp_path, with_source=True, scene_name=scene_name,
        output_namespace=namespace,
    )
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _, cfg_path = _install_training_pdc(tmp_path, scene, method)
    cfg_path.write_text(
        "Namespace("
        f"source_path={scene['dataset_root']!r}, "
        f"dataset_name={method['training']['dataset_name']!r}{suffix_field})",
        encoding="utf-8",
    )
    result = preflight(scene, method, fingerprints)
    assert result["status"] == "NO_GO"
    assert any("missing or invalid suffix" in item for item in result["blockers"])


def test_training_pdc_audit_is_read_only_and_frozen_training_contract_stays_intact(
    tmp_path, monkeypatch
):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    actual, _ = _install_training_pdc(tmp_path, scene, method)
    before = (actual.stat().st_mtime_ns, actual.read_bytes())
    command = training_command(scene, method, stage="formal", arm="receiver_v2")[0]
    after = (actual.stat().st_mtime_ns, actual.read_bytes())
    assert after == before
    assert command[command.index("--iterations") + 1] == "5000"
    assert command[command.index("--dropout_factor") + 1] == "0.2"
    assert command[command.index("--densify_from_iter") + 1] == "1000"
    assert command[command.index("--densification_interval") + 1] == "100"
    assert command[command.index("--densify_grad_threshold") + 1] == "0.0002"
    assert "--enable_oe_structural_loss" not in command
    assert "--enable_i2_future_utility_tracking" not in command


def test_budget_is_global_method_parameter_not_scene_field(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    assert "budget" not in scene
    assert method["budget"]["requested_k"] == 20


def test_non_default_budget_flows_through_shared_prepare_plan(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    method["budget"]["requested_k"] = 7
    plan = build_prepare_plan(scene, method)
    assert plan["requested_k"] == 7
    flattened = [value for entry in plan["commands"] for value in entry["argv"]]
    assert "i2_d1_receiver_v2_k7_plan.json" in " ".join(flattened)
    assert flattened.count("7") >= 3


def test_different_scene_configs_use_the_same_adaptive_entry(tmp_path, monkeypatch):
    policy = ROOT / "configs" / "i2_adaptive_k_r0_v1.json"
    commands = []
    for scene_name in ("scene_a", "scene_b"):
        scene, method, _ = _fixture(
            tmp_path / scene_name, with_source=True, scene_name=scene_name,
            output_namespace=f"i2_{scene_name}",
        )
        prepare = tmp_path / scene_name / "prepare"
        receiver_report = prepare / "receivers" / "receiver_evidence_report.json"
        receiver_manifest = prepare / "receivers" / "d_input_manifest_with_receivers.json"
        receiver_report.parent.mkdir(parents=True)
        receiver_report.write_text("{}", encoding="utf-8")
        receiver_manifest.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pipeline, "prepare_paths", lambda _scene, root=prepare: {
            "root": root, "b1": root / "b1", "receivers": root / "receivers",
            "d0": root / "d0_k", "counterfactual": root / "counterfactual",
            "receiver_v2": root / "receiver_v2", "initialization": root / "initialization",
        })
        fingerprints = {
            "scene_config": str(tmp_path / scene_name / "scene.json"),
            "method_config": str(ROOT / "configs" / "i2_method_v2.json"),
        }
        command, _ = adaptive_prepare_command(
            scene, method, fingerprints, policy_config=policy,
            receiver_manifest=receiver_manifest, receiver_report=receiver_report,
            output_dir=tmp_path / scene_name / "adaptive",
        )
        commands.append(command)
    script_indices = [command.index("-u") + 1 for command in commands]
    assert commands[0][script_indices[0]] == commands[1][script_indices[1]]
    assert commands[0][script_indices[0]].endswith("tools\\audit_i2_adaptive_k.py")
    assert commands[0][commands[0].index("--scene") + 1] == "scene_a"
    assert commands[1][commands[1].index("--scene") + 1] == "scene_b"


def test_adaptive_evidence_plan_stops_before_all_k_dependent_safety_stages(tmp_path):
    scene, method, _ = _fixture(tmp_path, with_source=True)
    plan = build_adaptive_evidence_plan(scene, method, tmp_path / "adaptive-evidence")
    assert [row["name"] for row in plan["commands"]] == ["b1", "receiver_candidates"]
    assert plan["contains_k_dependent_safety_audit"] is False
    scripts = [Path(row["argv"][2]).name for row in plan["commands"]]
    assert scripts == ["analyze_i2_b1c.py", "build_i2_receivers.py"]
    flattened = " ".join(value for row in plan["commands"] for value in row["argv"])
    assert "audit_i2_donor_counterfactual.py" not in flattened
    assert "audit_i2_receiver_v2_initialization.py" not in flattened


def _adaptive_smoke_metadata(**overrides):
    value = {
        "status": "PASS", "scene": "demo", "stage": "smoke",
        "arm": "receiver_v2", "requested_k": 4,
        "plan_sha256": "a" * 64,
        "initialization_audit_sha256": "b" * 64,
        "initialization_plan_sha256": "a" * 64,
        "adaptive_decision_sha256": "c" * 64,
        "source_checkpoint_sha256": "d" * 64,
        "view_manifest_sha256": "e" * 64,
        "fingerprints": {"method_config_sha256": "f" * 64},
    }
    value.update(overrides)
    return value


def _validate_adaptive_smoke(
    smoke, *, arm="receiver_v2", k=4, plan="a" * 64,
    initialization_audit="b" * 64,
):
    validate_adaptive_smoke_metadata(
        smoke, scene="demo", arm=arm, requested_k=k, plan_sha256=plan,
        initialization_audit_sha256=initialization_audit,
        adaptive_decision_sha256="c" * 64,
        source_checkpoint_sha256="d" * 64,
        view_manifest_sha256="e" * 64,
        method_config_sha256="f" * 64,
    )


def test_adaptive_formal_rejects_same_k_plan_with_different_decision_sha():
    smoke = _adaptive_smoke_metadata(adaptive_decision_sha256="9" * 64)
    with pytest.raises(PipelineError, match="adaptive_decision_sha256"):
        _validate_adaptive_smoke(smoke)


def test_adaptive_smoke_launch_metadata_records_decision_sha():
    source = inspect.getsource(pipeline.execute_training)
    assert '"adaptive_decision_sha256"' in source
    assert 'sha256(adaptive_binding["decision"])' in source


@pytest.mark.parametrize("field,value", [
    ("scene", "other"), ("arm", "control"),
    ("source_checkpoint_sha256", "9" * 64),
])
def test_adaptive_formal_rejects_scene_arm_or_common_source_mismatch(field, value):
    smoke = _adaptive_smoke_metadata(**{field: value})
    with pytest.raises(PipelineError, match=field):
        _validate_adaptive_smoke(smoke)


@pytest.mark.parametrize("field,value", [
    ("requested_k", 5), ("plan_sha256", "9" * 64),
    ("initialization_audit_sha256", "9" * 64),
    ("initialization_plan_sha256", "9" * 64),
    ("view_manifest_sha256", "9" * 64),
])
def test_adaptive_formal_rejects_budget_plan_audit_or_manifest_mismatch(field, value):
    smoke = _adaptive_smoke_metadata(**{field: value})
    with pytest.raises(PipelineError, match=field):
        _validate_adaptive_smoke(smoke)


def test_adaptive_formal_rejects_method_config_mismatch():
    smoke = _adaptive_smoke_metadata(
        fingerprints={"method_config_sha256": "9" * 64},
    )
    with pytest.raises(PipelineError, match="method_config_sha256"):
        _validate_adaptive_smoke(smoke)


def test_adaptive_k0_smoke_identity_requires_control_and_no_plan():
    smoke = _adaptive_smoke_metadata(
        arm="control", requested_k=0, plan_sha256=None,
        initialization_audit_sha256=None, initialization_plan_sha256=None,
    )
    _validate_adaptive_smoke(
        smoke, arm="control", k=0, plan=None, initialization_audit=None,
    )
    smoke["arm"] = "receiver_v2"
    with pytest.raises(PipelineError, match="arm"):
        _validate_adaptive_smoke(
            smoke, arm="control", k=0, plan=None, initialization_audit=None,
        )


def test_adaptive_prepare_missing_base_evidence_writes_unknown_failure(tmp_path, monkeypatch):
    scene, method, fingerprints = _fixture(tmp_path, with_source=True)
    output = tmp_path / "adaptive-output"
    monkeypatch.setattr(
        run_i2_scene, "preflight",
        lambda *_args, **_kwargs: {"status": "GO", "blockers": []},
    )
    monkeypatch.setattr(
        run_i2_scene, "build_adaptive_evidence_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PipelineError("missing evidence")),
    )
    with pytest.raises(PipelineError, match="failure evidence"):
        run_i2_scene.adaptive_prepare_stage(
            scene, method, fingerprints, True,
            ROOT / "configs/i2_adaptive_k_r0_v1.json", output,
        )
    failure = json.loads((output / "adaptive_prepare_failure.json").read_text())
    assert failure["selected_k"] == "UNKNOWN"
    assert failure["status"] == "NO_GO"


def test_adaptive_prepare_output_collision_refuses_overwrite_or_rename(tmp_path, monkeypatch):
    scene, method, fingerprints = _fixture(tmp_path, with_source=True)
    output = tmp_path / "adaptive-output"
    output.mkdir()
    monkeypatch.setattr(
        run_i2_scene, "preflight",
        lambda *_args, **_kwargs: {"status": "GO", "blockers": []},
    )
    with pytest.raises(PipelineError, match="Refusing to overwrite"):
        run_i2_scene.adaptive_prepare_stage(
            scene, method, fingerprints, True,
            ROOT / "configs/i2_adaptive_k_r0_v1.json", output,
        )
    assert list(tmp_path.glob("adaptive-output_v*")) == []


def test_adaptive_decision_binds_k_plan_and_initialization_gate(tmp_path):
    scene, method, fingerprints = load_configs("fern")
    root = ROOT / "reports/i2_scene_runs/k01_adaptive_k_v20261015T100927Z/fern"
    plan = root / "plans/adaptive_k13_plan.json"
    audits = [root / f"initialization_audit_k{k}.json" for k in range(1, 14)]
    audit = audits[-1]
    policy = ROOT / "configs/i2_adaptive_k_r0_v1.json"
    prepare_input = tmp_path / "prepare.json"
    prepare_input.write_text("{}", encoding="utf-8")
    reference = lambda path: {"path": str(path.resolve()), "sha256": _sha(path)}
    receiver_ids = [f"r{i}" for i in range(20)]
    receiver_summary = {
        "k_cap": 20, "geometry_nms_capacity_g": 48, "audited_prefix_count": 20,
        "receiver_evidence_budget_r": 13, "accepted_prefix_ids": receiver_ids,
        "positive_evidence_ids": receiver_ids[:13], "positive_prefix_exact_match": True,
    }
    receiver_audit = tmp_path / "receiver-audit.json"
    receiver_audit.write_text(json.dumps({
        "scene": "fern", "source_checkpoint_sha256": scene["common_source"]["sha256"],
        "receiver_budget": receiver_summary,
        "accepted_order": receiver_ids + [f"tail{i}" for i in range(28)],
        "test_views_used": False,
    }), encoding="utf-8")
    donor_audit = tmp_path / "donor-audit.json"
    donor_audit.write_text(json.dumps({
        "format": "i2_bounded_donor_capacity_audit_v1", "scene": "fern",
        "contiguous_joint_safe_capacity": 13, "capacity_determined": True,
        "test_views_used": False,
    }), encoding="utf-8")
    decision = {
        "format": "i2_adaptive_k_r0_v1_decision", "policy_version": "R0-v1",
        "scene": "fern", "uses_test_views": False, "selected_k": 13,
        "execution_route": "adaptive_plan_available_no_training_started",
        "source_checkpoint_sha256": scene["common_source"]["sha256"],
        "k_cap": 20, "receiver_capacity": 13,
        "donor_contiguous_safe_capacity": 13,
        "initialization_contiguous_safe_capacity": 13,
        "receiver_audit": receiver_summary,
        "adaptive_plan": reference(plan),
        "evidence": {
            "receiver": reference(receiver_audit), "donor": reference(donor_audit),
            "initialization": [reference(path) for path in audits],
        },
        "input_bindings": {
            "scene_config": reference(Path(fingerprints["scene_config"])),
            "method_config": reference(Path(fingerprints["method_config"])),
            "policy_config": reference(policy),
            "common_source": reference(Path(scene["common_source"]["checkpoint"])),
            "source_binding": None,
            "prepare": {
                "receiver_report": reference(prepare_input),
                "receiver_manifest": reference(prepare_input),
            },
        },
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    bound = validate_adaptive_training_binding(
        scene, method, decision_path=decision_path
    )
    assert bound["selected_k"] == 13
    assert bound["plan"] == plan.resolve()
    assert bound["audit"] == audit.resolve()


def test_adaptive_training_rejects_manual_k_plan_or_gate(tmp_path, monkeypatch):
    scene, method, fingerprints = _fixture(tmp_path, with_source=True)
    monkeypatch.setattr(
        run_i2_scene, "preflight",
        lambda *_args, **_kwargs: {"status": "GO", "blockers": []},
    )
    with pytest.raises(PipelineError, match="cannot be combined"):
        run_i2_scene.training_stage(
            scene, method, fingerprints, stage="smoke", arm="receiver_v2",
            execute=False, adaptive_decision="decision.json", requested_budget=7,
        )


def test_active_budget_validation_is_positive_not_scene_whitelist():
    validate_i2_active_configuration(True, "plan.json", "event.json", 7)
    with pytest.raises(ActivePilotError, match="positive integer"):
        validate_i2_active_configuration(True, "plan.json", "event.json", 0)


def test_fern_k21_external_plan_budget_and_output_binding(tmp_path):
    scene, method, _ = load_configs(
        "fern", method_config="configs/i2_method_adaptive_k21_v1.json"
    )
    plan = ROOT / (
        "reports/i2_scene_runs/k01_adaptive_k_v20261015T100927Z/fern/"
        "plans/adaptive_k21_plan.json"
    )
    audit = ROOT / (
        "reports/i2_scene_runs/k01_adaptive_k_v20261015T100927Z/fern/"
        "initialization_audit_k21.json"
    )
    bound_plan, bound_audit, payload = validate_external_training_binding(
        scene, method, active_plan=plan, requested_budget=21,
        initialization_audit=audit,
    )
    output = tmp_path / "fern-adaptive-k21-smoke"
    command, actual_output, iteration = training_command(
        scene, method, stage="smoke", arm="receiver_v2",
        active_plan=bound_plan, requested_budget=21, output_dir=output,
    )
    assert bound_audit == audit.resolve()
    assert payload["safe"] is True and payload["k"] == 21
    assert actual_output == output.resolve() and iteration == 1252
    assert command[command.index("--i2_active_plan") + 1] == str(plan.resolve())
    assert command[command.index("--i2_active_budget") + 1] == "21"
    assert command[command.index("-m") + 1] == str(output.resolve())
    assert command[command.index("--start_checkpoint") + 1].endswith("chkpnt1201.pth")
    assert command[command.index("--view_manifest") + 1] == str(
        Path(scene["view_manifest"]["path"]).resolve()
    )
    assert "--enable_oe_structural_loss" not in command


def test_fern_k21_external_plan_rejects_budget_or_audit_mismatch(tmp_path):
    scene, method, _ = load_configs(
        "fern", method_config="configs/i2_method_adaptive_k21_v1.json"
    )
    root = ROOT / "reports/i2_scene_runs/k01_adaptive_k_v20261015T100927Z/fern"
    plan = root / "plans/adaptive_k21_plan.json"
    audit = root / "initialization_audit_k21.json"
    with pytest.raises(PipelineError, match="budget"):
        validate_external_training_binding(
            scene, method, active_plan=plan, requested_budget=20,
            initialization_audit=audit,
        )
    modified = json.loads(audit.read_text(encoding="utf-8"))
    modified["plan"]["sha256"] = "0" * 64
    bad_audit = tmp_path / "bad-audit.json"
    bad_audit.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(PipelineError, match="not safely bound"):
        validate_external_training_binding(
            scene, method, active_plan=plan, requested_budget=21,
            initialization_audit=bad_audit,
        )

