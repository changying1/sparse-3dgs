"""Configuration-driven orchestration for the cross-scene I2 protocol.

This module owns paths, gates, fingerprints, and command construction.  It
delegates all scientific operations to the already validated I2 tools.
"""

from __future__ import annotations

import ast
import hashlib
import json
import stat
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_KEYS = {
    "schema_version", "scene", "dataset_root", "view_manifest", "pdc",
    "depth_cache", "split", "common_source", "output_namespace",
}
METHOD_KEYS = {
    "schema_version", "method_id", "budget", "source", "donor",
    "receiver_candidates", "receiver_v2", "initialization_audit", "training",
    "smoke", "evaluation", "runtime", "randomness_caveat",
}
ARMS = ("control", "release", "receiver_v2")
D3_ARMS = ("release", "birth_only")
CODE_FINGERPRINT_FILES = (
    "train.py", "scene/gaussian_model.py", "utils/i2_active_pilot.py",
    "utils/i2_scene_pipeline.py", "tools/run_i2_scene.py",
    "tools/analyze_i2_b1c.py", "tools/build_i2_d0_dry_run.py",
    "tools/audit_i2_donor_counterfactual.py", "tools/build_i2_receivers.py",
    "tools/build_i2_receiver_v2.py",
    "tools/audit_i2_receiver_v2_initialization.py", "tools/evaluate_i2_5k.py",
    "utils/i2_adaptive_k.py", "tools/audit_i2_adaptive_k.py",
)
SOURCE_BINDING_FORMAT = "i2_source_binding_v1"
INITIALIZATION_GATE_NAMES = {
    "receiver_v2_changes_target_regions",
    "target_region_not_worse_than_release",
    "outside_region_no_obvious_degradation",
    "global_no_obvious_degradation",
    "budget_and_alignment_pass",
}
ADAPTIVE_INITIALIZATION_GATE_NAMES = {
    "receiver_changes_target_regions",
    "target_region_not_worse_than_release",
    "outside_region_no_obvious_degradation",
    "global_no_obvious_degradation",
    "budget_and_alignment_pass",
}


class PipelineError(RuntimeError):
    """A pipeline gate failed closed."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_keys(payload: dict, expected: set[str], label: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise PipelineError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot load JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON root must be an object: {path}")
    return value


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def source_binding_path(scene_cfg: dict) -> Path:
    """Return the single auditable binding location for an unbound scene."""
    return (
        ROOT / "reports" / "i2_scene_runs" / scene_cfg["output_namespace"]
        / "source_binding.json"
    )


def _source_inputs(scene_cfg: dict) -> dict:
    return {
        "dataset_root": str(resolve_path(scene_cfg["dataset_root"])),
        "view_manifest": {
            "path": str(resolve_path(scene_cfg["view_manifest"]["path"])),
            "sha256": scene_cfg["view_manifest"]["sha256"].lower(),
        },
        "pdc": {
            "path": str(resolve_path(scene_cfg["pdc"]["path"])),
            "sha256": scene_cfg["pdc"]["sha256"].lower(),
        },
        "depth_cache": {
            "path": str(resolve_path(scene_cfg["depth_cache"]["path"])),
            "metadata_sha256": scene_cfg["depth_cache"]["metadata_sha256"].lower(),
        },
    }


def register_source_binding(scene_cfg: dict, method_cfg: dict, checkpoint: Path,
                            checkpoint_audit: dict) -> tuple[Path, dict]:
    """Bind a newly audited source without mutating the scene configuration."""
    if scene_cfg["common_source"]["checkpoint"] is not None:
        raise PipelineError("Frozen/config-bound common sources must not be rebound")
    checkpoint = checkpoint.resolve()
    expected_iteration = int(method_cfg["source"]["iteration"])
    if checkpoint_audit.get("iteration") != expected_iteration:
        raise PipelineError("Source binding iteration differs from checkpoint audit")
    binding = {
        "format": SOURCE_BINDING_FORMAT,
        "scene": scene_cfg["scene"],
        "iteration": expected_iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "a0_status": "PASS" if checkpoint_audit.get("uid_unique") is True else "FAIL",
        "b0_status": "PASS" if checkpoint_audit.get("observation_rows") else "FAIL",
        "input_sources": _source_inputs(scene_cfg),
    }
    if binding["a0_status"] != "PASS" or binding["b0_status"] != "PASS":
        raise PipelineError("Source binding requires audited A0 and B0 PASS states")
    path = source_binding_path(scene_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PipelineError(f"Refusing to overwrite source binding: {path}")
    path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, binding


def resolve_common_source(scene_cfg: dict, method_cfg: dict) -> tuple[Path | None, str | None, Path | None]:
    """Resolve only an explicit config binding or the one deterministic binding file."""
    configured = scene_cfg["common_source"]
    if configured["checkpoint"] is not None:
        return resolve_path(configured["checkpoint"]), configured["sha256"].lower(), None
    binding_path = source_binding_path(scene_cfg)
    if not binding_path.is_file():
        return None, None, binding_path
    binding = _load_json(binding_path)
    expected_keys = {
        "format", "scene", "iteration", "checkpoint", "checkpoint_sha256",
        "a0_status", "b0_status", "input_sources",
    }
    _require_keys(binding, expected_keys, "source binding")
    if (
        binding["format"] != SOURCE_BINDING_FORMAT
        or binding["scene"] != scene_cfg["scene"]
        or binding["iteration"] != int(method_cfg["source"]["iteration"])
        or binding["a0_status"] != "PASS"
        or binding["b0_status"] != "PASS"
        or binding["input_sources"] != _source_inputs(scene_cfg)
        or not isinstance(binding["checkpoint_sha256"], str)
        or len(binding["checkpoint_sha256"]) != 64
    ):
        raise PipelineError("Source binding metadata does not match the scene contract")
    checkpoint = Path(binding["checkpoint"])
    if not checkpoint.is_absolute():
        raise PipelineError("Source binding checkpoint must be an absolute path")
    return checkpoint.resolve(), binding["checkpoint_sha256"].lower(), binding_path


def load_configs(scene: str, *, scene_config: str | Path | None = None,
                 method_config: str | Path | None = None) -> tuple[dict, dict, dict]:
    scene_path = resolve_path(str(scene_config)) if scene_config else (
        ROOT / "configs" / "i2_scenes" / f"{scene}.json"
    )
    method_path = resolve_path(str(method_config)) if method_config else (
        ROOT / "configs" / "i2_method_v2.json"
    )
    if scene_path is None or method_path is None:
        raise PipelineError("Configuration path is missing")
    scene_cfg = _load_json(scene_path)
    method_cfg = _load_json(method_path)
    validate_scene_config(scene_cfg, expected_scene=scene)
    validate_method_config(method_cfg)
    return scene_cfg, method_cfg, {
        "scene_config": str(scene_path),
        "scene_config_sha256": sha256(scene_path),
        "method_config": str(method_path),
        "method_config_sha256": sha256(method_path),
        "code_sha256": {
            name: sha256(ROOT / name) for name in CODE_FINGERPRINT_FILES
        },
    }


def validate_scene_config(config: dict, *, expected_scene: str | None = None) -> None:
    _require_keys(config, SCENE_KEYS, "scene config")
    if config.get("schema_version") != "i2-scene-config-v1":
        raise PipelineError("Unsupported scene configuration schema")
    scene = config.get("scene")
    if not isinstance(scene, str) or not scene or (expected_scene and scene != expected_scene):
        raise PipelineError("Scene configuration identity mismatch")
    if not isinstance(config.get("dataset_root"), str):
        raise PipelineError("dataset_root must be a path string")
    for field, hash_field in (
        ("view_manifest", "sha256"), ("pdc", "sha256"),
        ("depth_cache", "metadata_sha256"),
    ):
        value = config.get(field)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise PipelineError(f"{field} must contain a path")
        digest = value.get(hash_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise PipelineError(f"{field}.{hash_field} must be SHA-256")
    split = config.get("split")
    if not isinstance(split, dict) or set(split) != {"train_images", "test_images"}:
        raise PipelineError("split must contain only train_images and test_images")
    train, test = split["train_images"], split["test_images"]
    if not isinstance(train, list) or not isinstance(test, list) or len(train) != 3:
        raise PipelineError("Scene split must declare three training views and a test list")
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise PipelineError("Training/test views must be unique and disjoint")
    source = config.get("common_source")
    if not isinstance(source, dict) or set(source) != {"checkpoint", "sha256"}:
        raise PipelineError("common_source schema is invalid")
    if (source["checkpoint"] is None) != (source["sha256"] is None):
        raise PipelineError("common_source checkpoint and hash must be both set or both null")
    if source["sha256"] is not None and len(source["sha256"]) != 64:
        raise PipelineError("common_source.sha256 must be SHA-256")
    namespace = config.get("output_namespace")
    if not isinstance(namespace, str) or not namespace or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in namespace):
        raise PipelineError("output_namespace is invalid")


def validate_method_config(config: dict) -> None:
    _require_keys(config, METHOD_KEYS, "method config")
    if config.get("schema_version") != "i2-method-config-v1":
        raise PipelineError("Unsupported method configuration schema")
    if config["budget"].get("requested_k", 0) <= 0:
        raise PipelineError("requested_k must be positive")
    receiver = config["receiver_v2"]
    frozen = {
        "max_projection_condition": 25.0,
        "max_displacement_tangent_norm_factor": 2.0,
        "activated_opacity": 0.05,
        "footprint_pixels": 2.0,
        "tangent_scale_lower_factor": 0.2,
        "tangent_scale_upper_factor": 0.6,
        "normal_scale_factor": 0.35,
        "fail_closed": True,
    }
    if any(receiver.get(key) != value for key, value in frozen.items()):
        raise PipelineError("Receiver-v2 frozen numerical contract was changed")
    training = config["training"]
    if training.get("i1_after_source") is not False or training.get("future_utility_c") is not False:
        raise PipelineError("Post-source I1 and Future Utility C must remain disabled")
    if config["source"].get("enable_c") is not False:
        raise PipelineError("Source protocol must not enable Future Utility C")


def _require_hash(path: Path, expected: str, blockers: list[str], label: str) -> None:
    try:
        if not path.is_file():
            blockers.append(f"MISSING_{label}:{path}")
        elif sha256(path).lower() != expected.lower():
            blockers.append(f"HASH_MISMATCH_{label}:{path}")
    except (OSError, PermissionError) as exc:
        blockers.append(f"UNREADABLE_{label}:{path}:{exc}")


def _parse_cfg(path: Path) -> dict:
    try:
        node = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PipelineError(f"Cannot parse cfg_args: {path}: {exc}") from exc
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
        raise PipelineError(f"Unrecognized cfg_args: {path}")
    try:
        return {item.arg: ast.literal_eval(item.value) for item in node.keywords}
    except (ValueError, TypeError) as exc:
        raise PipelineError(f"cfg_args contains a non-literal value: {path}") from exc


def _source_suffix(scene_cfg: dict) -> str:
    """Return the canonical suffix used when this pipeline creates a source."""
    return f"{scene_cfg['output_namespace']}_source"


def _validated_source_suffix(cfg: dict, cfg_path: Path, scene_cfg: dict) -> str:
    if "suffix" not in cfg:
        raise PipelineError(f"Common-source cfg_args has missing or invalid suffix: {cfg_path}")
    suffix = cfg["suffix"]
    if suffix is None:
        return _source_suffix(scene_cfg)
    if (
        not isinstance(suffix, str)
        or not suffix
        or suffix in {".", ".."}
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in suffix)
    ):
        raise PipelineError(f"Common-source cfg_args has missing or invalid suffix: {cfg_path}")
    return suffix


def resolve_training_pdc(scene_cfg: dict, method_cfg: dict) -> dict:
    """Resolve and verify the exact PDC path Scene will read for resumed training."""
    checkpoint, _, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Training PDC audit requires a bound common source")
    cfg_path = checkpoint.parent / "cfg_args"
    input_ply = checkpoint.parent / "input.ply"
    if not cfg_path.is_file():
        raise PipelineError(f"Common-source cfg_args is missing: {cfg_path}")
    cfg = _parse_cfg(cfg_path)
    suffix = _validated_source_suffix(cfg, cfg_path, scene_cfg)
    training = method_cfg["training"]
    dataset_name = training["dataset_name"]
    if cfg.get("dataset_name") != dataset_name:
        raise PipelineError("Common-source dataset_name differs from the frozen training protocol")
    dataset_root = resolve_path(scene_cfg["dataset_root"])
    if dataset_root is None or Path(cfg.get("source_path", "")).resolve() != dataset_root:
        raise PipelineError("Common-source dataset path differs from scene config")
    scene_name = dataset_root.name
    if scene_name != scene_cfg["scene"]:
        raise PipelineError("Scene name differs from the dataset root basename used by Scene")

    # Mirrors scene/dataset_readers.py:191 with train.py running at ROOT.
    actual_pdc = (
        ROOT / "keypoints_to_3d" / f"{dataset_name}_{suffix}"
        / f"{scene_name}_keypoints_to_3d.ply"
    ).resolve()
    frozen_pdc = resolve_path(scene_cfg["pdc"]["path"])
    expected = scene_cfg["pdc"]["sha256"].lower()
    checks = (
        (actual_pdc, "Scene PDC"),
        (frozen_pdc, "frozen PDC"),
        (input_ply, "common-source input.ply"),
    )
    for path, label in checks:
        if path is None or not path.is_file():
            raise PipelineError(f"{label} is missing: {path}")
        if sha256(path).lower() != expected:
            raise PipelineError(f"{label} SHA-256 differs from the frozen PDC: {path}")
    return {
        "suffix": suffix,
        "path": actual_pdc,
        "sha256": expected,
        "source_cfg": cfg_path,
        "source_input_ply": input_ply,
        "frozen_pdc": frozen_pdc,
    }


def audit_source_checkpoint(checkpoint: Path, scene_cfg: dict, expected_iteration: int) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise PipelineError("Checkpoint audit requires the configured fsgs Python") from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise PipelineError("Common-source checkpoint wrapper is malformed")
    state, iteration = payload
    if iteration != expected_iteration or not isinstance(state, (tuple, list)) or len(state) != 13:
        raise PipelineError("Common source is not the required 13-field iteration-1201 checkpoint")
    count = int(state[1].shape[0])
    if any(not hasattr(state[index], "shape") or int(state[index].shape[0]) != count for index in range(1, 10)):
        raise PipelineError("Gaussian trainable/statistic rows are misaligned")
    extension = state[12]
    if not isinstance(extension, dict) or set(extension) != {"i2_lifecycle", "i2_observation"}:
        raise PipelineError("Common source must contain exactly A0+B0 and no C state")
    lifecycle, observation = extension["i2_lifecycle"], extension["i2_observation"]
    lifecycle_fields = ("uid", "birth_iter", "birth_type", "source_uid", "generation")
    observation_fields = (
        "uid", "real_view_opportunities", "dropout_retained_hits",
        "screen_supported_hits", "screen_supported_view_history",
    )
    lifecycle_rows = {key: int(lifecycle[key].shape[0]) for key in lifecycle_fields}
    observation_rows = {key: int(observation[key].shape[0]) for key in observation_fields}
    uid = lifecycle["uid"]
    if set(lifecycle_rows.values()) != {count} or set(observation_rows.values()) != {count}:
        raise PipelineError("A0/B0 rows are not aligned with Gaussian rows")
    if int(torch.unique(uid).numel()) != count or not torch.equal(uid, observation["uid"]):
        raise PipelineError("A0 UIDs are not unique or B0 UID order differs")
    train_identifiers = list(scene_cfg["split"]["train_images"])
    if list(observation.get("view_identifiers", [])) != train_identifiers:
        raise PipelineError("B0 view identifiers differ from the frozen training split")
    opportunities = observation["real_view_opportunities"]
    retained = observation["dropout_retained_hits"]
    supported = observation["screen_supported_hits"]
    if bool(torch.any(opportunities < retained)) or bool(torch.any(retained < supported)):
        raise PipelineError("B0 observation counters are invalid")
    optimizer = state[10]
    states = optimizer.get("state", {}) if isinstance(optimizer, dict) else {}
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    row_counts = {
        int(item["exp_avg"].shape[0])
        for item in states.values()
        if isinstance(item, dict) and "exp_avg" in item
    }
    if len(states) != 6 or len(groups) != 6 or row_counts != {count}:
        raise PipelineError("The six Adam states are not row-aligned")

    parent = checkpoint.parent
    cfg_path, input_ply, cameras_path = parent / "cfg_args", parent / "input.ply", parent / "cameras.json"
    for required in (cfg_path, input_ply, cameras_path):
        if not required.is_file():
            raise PipelineError(f"Common-source provenance file is missing: {required}")
    cfg = _parse_cfg(cfg_path)
    if Path(cfg.get("source_path", "")).resolve() != resolve_path(scene_cfg["dataset_root"]):
        raise PipelineError("Common-source dataset path differs from scene config")
    if Path(cfg.get("view_manifest", "")).resolve() != resolve_path(scene_cfg["view_manifest"]["path"]):
        raise PipelineError("Common-source manifest differs from scene config")
    if sha256(input_ply) != scene_cfg["pdc"]["sha256"].lower():
        raise PipelineError("Common-source input.ply differs from frozen PDC")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    camera_names = {row["img_name"] for row in cameras}
    expected_names = {Path(name).stem for name in scene_cfg["split"]["train_images"] + scene_cfg["split"]["test_images"]}
    if camera_names != expected_names:
        raise PipelineError("Common-source cameras differ from the frozen train/test split")
    return {
        "iteration": iteration,
        "gaussian_count": count,
        "checkpoint_fields": len(state),
        "uid_unique": True,
        "lifecycle_rows": lifecycle_rows,
        "observation_rows": observation_rows,
        "optimizer_state_groups": len(states),
        "optimizer_rows": sorted(row_counts),
        "future_utility_c_present": False,
        "training_view_identifiers": train_identifiers,
        "provenance_match": True,
    }


def preflight(scene_cfg: dict, method_cfg: dict, fingerprints: dict,
              *, audit_checkpoint: bool = True) -> dict:
    blockers: list[str] = []
    dataset = resolve_path(scene_cfg["dataset_root"])
    manifest_path = resolve_path(scene_cfg["view_manifest"]["path"])
    pdc_path = resolve_path(scene_cfg["pdc"]["path"])
    cache_dir = resolve_path(scene_cfg["depth_cache"]["path"])
    assert dataset and manifest_path and pdc_path and cache_dir
    if not dataset.is_dir():
        blockers.append(f"MISSING_DATASET:{dataset}")
    _require_hash(manifest_path, scene_cfg["view_manifest"]["sha256"], blockers, "MANIFEST")
    _require_hash(pdc_path, scene_cfg["pdc"]["sha256"], blockers, "PDC")
    cache_meta = cache_dir / "oe_depth_cache.json"
    _require_hash(cache_meta, scene_cfg["depth_cache"]["metadata_sha256"], blockers, "DEPTH_CACHE_METADATA")

    if not blockers:
        manifest = _load_json(manifest_path)
        train, test = scene_cfg["split"]["train_images"], scene_cfg["split"]["test_images"]
        if manifest.get("scene") != scene_cfg["scene"] or manifest.get("dataset") != "LLFF":
            blockers.append("MANIFEST_SCENE_OR_DATASET_MISMATCH")
        if manifest.get("selected_train_images") != train or manifest.get("pdc_initialization_images") != train:
            blockers.append("MANIFEST_TRAIN_OR_PDC_SPLIT_MISMATCH")
        if manifest.get("test_images") != test or set(train) & set(test):
            blockers.append("MANIFEST_TEST_SPLIT_MISMATCH")
        try:
            disk_images = sorted(path.name for path in (dataset / "images").iterdir() if path.is_file())
            if disk_images != manifest.get("sorted_all_images"):
                blockers.append("DATASET_IMAGE_INVENTORY_MISMATCH")
        except (OSError, PermissionError) as exc:
            blockers.append(f"UNREADABLE_DATASET_IMAGES:{exc}")
        cache = _load_json(cache_meta)
        if cache.get("source_view_manifest_sha256", "").lower() != scene_cfg["view_manifest"]["sha256"].lower():
            blockers.append("DEPTH_CACHE_MANIFEST_MISMATCH")
        if cache.get("selected_train_images") != train:
            blockers.append("DEPTH_CACHE_TRAIN_SPLIT_MISMATCH")
        for record in cache.get("records", []):
            _require_hash(dataset / "images" / record.get("source_image", ""), record.get("source_sha256", ""), blockers, "SOURCE_IMAGE")
            _require_hash(cache_dir / record.get("depth_file", ""), record.get("depth_sha256", ""), blockers, "DEPTH_ARRAY")

    source_status = "SOURCE_REQUIRED"
    source_audit = None
    binding_path = None
    try:
        checkpoint, checkpoint_sha, binding_path = resolve_common_source(scene_cfg, method_cfg)
    except PipelineError as exc:
        checkpoint, checkpoint_sha = None, None
        blockers.append(f"SOURCE_BINDING_INVALID:{exc}")
    if checkpoint is not None:
        source_status = "SOURCE_INVALID"
        before = len(blockers)
        _require_hash(checkpoint, checkpoint_sha, blockers, "COMMON_SOURCE")
        if len(blockers) == before and audit_checkpoint:
            try:
                resolve_training_pdc(scene_cfg, method_cfg)
            except PipelineError as exc:
                blockers.append(f"TRAINING_PDC_AUDIT_FAILED:{exc}")
        if len(blockers) == before and audit_checkpoint:
            try:
                source_audit = audit_source_checkpoint(
                    checkpoint, scene_cfg, int(method_cfg["source"]["iteration"])
                )
            except Exception as exc:
                blockers.append(f"COMMON_SOURCE_AUDIT_FAILED:{exc}")
        if len(blockers) == before:
            source_status = "SOURCE_REUSABLE"
    status = "GO" if not blockers and source_status == "SOURCE_REUSABLE" else (
        "SOURCE_REQUIRED" if not blockers else "NO_GO"
    )
    namespace = scene_cfg["output_namespace"]
    next_stage = "prepare" if status == "GO" else "source" if status == "SOURCE_REQUIRED" else None
    return {
        "format": "i2_scene_preflight_v1",
        "status": status,
        "scene": scene_cfg["scene"],
        "source_status": source_status,
        "blockers": blockers,
        "train_images": scene_cfg["split"]["train_images"],
        "test_images": scene_cfg["split"]["test_images"],
        "test_view_count": len(scene_cfg["split"]["test_images"]),
        "source_checkpoint": None if checkpoint is None else str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_binding": None if binding_path is None else str(binding_path),
        "source_audit": source_audit,
        "fingerprints": fingerprints,
        "next_stage": {
            "stage": next_stage,
            "command": None if next_stage is None else (
                f"python tools/run_i2_scene.py --scene {scene_cfg['scene']} --stage {next_stage}"
            ),
            "gpu_execution_requires_execute": True,
            "prepare_output": str(ROOT / "reports/i2_scene_runs" / namespace / "prepare"),
        },
    }


def prepare_paths(scene_cfg: dict) -> dict[str, Path]:
    root = ROOT / "reports/i2_scene_runs" / scene_cfg["output_namespace"] / "prepare"
    return {
        "root": root,
        "b1": root / "b1",
        "receivers": root / "receivers",
        "d0": root / "d0_k",
        "counterfactual": root / "counterfactual",
        "receiver_v2": root / "receiver_v2",
        "initialization": root / "initialization",
    }


def initialization_audit_path(scene_cfg: dict) -> Path:
    return prepare_paths(scene_cfg)["initialization"] / "initialization_audit.json"


def adaptive_prepare_output(scene_cfg: dict) -> Path:
    return (
        ROOT / "reports" / "i2_scene_runs" / scene_cfg["output_namespace"]
        / "adaptive_r0_v1"
    )


def adaptive_prepare_command(scene_cfg: dict, method_cfg: dict, fingerprints: dict,
                             *, policy_config: str | Path,
                             receiver_manifest: str | Path,
                             receiver_report: str | Path,
                             output_dir: str | Path | None = None) -> tuple[list[str], Path]:
    """Build the one public R0-v1 adaptive preparation command for every scene."""
    policy = resolve_path(str(policy_config))
    if policy is None or not policy.is_file():
        raise PipelineError(f"Adaptive policy configuration is missing: {policy_config}")
    output = _training_output_path(output_dir, adaptive_prepare_output(scene_cfg))
    command = [
        method_cfg["runtime"]["python"], "-u", str(ROOT / "tools/audit_i2_adaptive_k.py"),
        "--scene", scene_cfg["scene"], "--scene-config", fingerprints["scene_config"],
        "--method-config", fingerprints["method_config"], "--policy-config", str(policy),
        "--receiver-manifest", str(Path(receiver_manifest).resolve()),
        "--receiver-report", str(Path(receiver_report).resolve()),
        "--output", str(output),
    ]
    return command, output


def _verified_reference(reference: dict, label: str) -> Path:
    if not isinstance(reference, dict):
        raise PipelineError(f"Adaptive decision {label} reference is missing")
    path = resolve_path(reference.get("path"))
    expected = reference.get("sha256")
    if path is None or not isinstance(expected, str) or len(expected) != 64:
        raise PipelineError(f"Adaptive decision {label} path/hash binding is invalid")
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise PipelineError(f"Adaptive decision {label} hash binding failed")
    return path


def validate_adaptive_training_binding(
    scene_cfg: dict, method_cfg: dict, *, decision_path: str | Path,
) -> dict:
    """Resolve K and Plan only from a hash-bound R0-v1 or RD-v1 decision."""
    path = resolve_path(str(decision_path))
    assert path is not None
    decision = _load_required_json(path, "Adaptive decision")
    if decision.get("policy_version") == "RD-v1":
        return _validate_rd_v1_training_binding(
            scene_cfg, method_cfg, path=path, decision=decision
        )
    if (
        decision.get("format") != "i2_adaptive_k_r0_v1_decision"
        or decision.get("policy_version") != "R0-v1"
        or decision.get("scene") != scene_cfg["scene"]
        or decision.get("uses_test_views") is not False
    ):
        raise PipelineError("Adaptive decision identity or evidence scope is invalid")
    selected = decision.get("selected_k")
    if selected == "UNKNOWN":
        raise PipelineError("Adaptive decision is UNKNOWN and blocks execution")
    if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected <= 20:
        raise PipelineError("Adaptive decision selected_k is invalid")
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None or decision.get("source_checkpoint_sha256", "").lower() != checkpoint_sha:
        raise PipelineError("Adaptive decision common-source binding differs")
    bindings = decision.get("input_bindings")
    if not isinstance(bindings, dict):
        raise PipelineError("Adaptive decision input bindings are missing")
    scene_config_path = _verified_reference(bindings.get("scene_config"), "scene_config")
    method_config_path = _verified_reference(bindings.get("method_config"), "method_config")
    policy_config_path = _verified_reference(bindings.get("policy_config"), "policy_config")
    common_source_path = _verified_reference(bindings.get("common_source"), "common_source")
    if _load_json(scene_config_path) != scene_cfg or _load_json(method_config_path) != method_cfg:
        raise PipelineError("Adaptive decision configuration binding differs")
    policy = _load_json(policy_config_path)
    expected_policy = {
        "schema_version": "i2-adaptive-k-policy-v1",
        "policy_id": "i2_adaptive_k_r0", "version": "R0-v1",
        "candidate_budget_cap": 20,
        "receiver": {
            "geometry": "receiver_v2_tangent_jacobian_v1",
            "ranking": "photometric_alignment_desc_demand_score_desc_candidate_id_asc",
            "nms": "greedy_spatial_nms_v1",
            "positive_evidence": "photometric_alignment_gt_zero",
            "require_finite_alignment": True,
            "require_positive_set_equals_prefix": True,
            "audit_all_candidates": True,
        },
        "donor": {
            "ranking": "frozen_d0_order", "require_individual_safety": True,
            "require_contiguous_joint_prefix": True,
        },
        "initialization": {
            "audit_prefixes_from_k": 1, "stop_after_first_failure": True,
            "require_contiguous_safe_prefix": True,
        },
        "forbidden_budget_signals": ["PSNR", "SSIM", "LPIPS", "test_images"],
    }
    if policy != expected_policy:
        raise PipelineError("Adaptive decision policy binding is not frozen R0-v1")
    if common_source_path != checkpoint.resolve():
        raise PipelineError("Adaptive decision common-source path differs")
    if bindings.get("source_binding") is not None:
        _verified_reference(bindings["source_binding"], "source_binding")
    prepare = bindings.get("prepare")
    if not isinstance(prepare, dict) or set(prepare) != {"receiver_manifest", "receiver_report"}:
        raise PipelineError("Adaptive decision prepare bindings are missing")
    for label, reference in prepare.items():
        _verified_reference(reference, f"prepare.{label}")
    receiver_audit = decision.get("receiver_audit")
    if not isinstance(receiver_audit, dict):
        raise PipelineError("Adaptive decision Receiver evidence summary is missing")
    raw_capacities = (
        decision.get("receiver_capacity"),
        decision.get("donor_contiguous_safe_capacity"),
        decision.get("initialization_contiguous_safe_capacity"),
        receiver_audit.get("geometry_nms_capacity_g"),
        receiver_audit.get("audited_prefix_count"),
    )
    if any(isinstance(value, bool) for value in raw_capacities):
        raise PipelineError("Adaptive decision capacity fields are invalid")
    try:
        receiver_capacity = int(decision["receiver_capacity"])
        donor_capacity = int(decision["donor_contiguous_safe_capacity"])
        initialization_capacity = int(decision["initialization_contiguous_safe_capacity"])
        geometry_capacity = int(receiver_audit["geometry_nms_capacity_g"])
        audited_count = int(receiver_audit["audited_prefix_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError("Adaptive decision capacity fields are invalid") from exc
    if (
        decision.get("k_cap") != 20
        or receiver_audit.get("k_cap") != 20
        or receiver_audit.get("receiver_evidence_budget_r") != receiver_capacity
        or receiver_audit.get("positive_prefix_exact_match") is not True
        or receiver_audit.get("positive_evidence_ids")
        != receiver_audit.get("accepted_prefix_ids", [])[:receiver_capacity]
        or geometry_capacity < 0
        or audited_count != min(20, geometry_capacity)
        or len(receiver_audit.get("accepted_prefix_ids", [])) != audited_count
        or len(receiver_audit.get("positive_evidence_ids", [])) != receiver_capacity
        or len(set(receiver_audit.get("accepted_prefix_ids", []))) != audited_count
        or min(receiver_capacity, donor_capacity, initialization_capacity) < 0
        or donor_capacity > receiver_capacity
        or min(20, receiver_capacity, donor_capacity, initialization_capacity) != selected
    ):
        raise PipelineError("Adaptive decision capacities violate R0-v1")
    evidence = decision.get("evidence")
    if not isinstance(evidence, dict) or "receiver" not in evidence:
        raise PipelineError("Adaptive decision evidence bindings are missing")
    receiver_path = _verified_reference(evidence["receiver"], "Receiver audit")
    receiver_payload = _load_required_json(receiver_path, "Receiver audit")
    if (
        receiver_payload.get("scene") != scene_cfg["scene"]
        or receiver_payload.get("source_checkpoint_sha256") != checkpoint_sha
        or receiver_payload.get("receiver_budget") != receiver_audit
        or len(receiver_payload.get("accepted_order", [])) != geometry_capacity
        or receiver_payload.get("accepted_order", [])[:audited_count]
        != receiver_audit.get("accepted_prefix_ids")
        or receiver_payload.get("test_views_used") is not False
    ):
        raise PipelineError("Adaptive Receiver audit binding differs")
    if receiver_capacity > 0:
        donor_path = _verified_reference(evidence.get("donor"), "Donor audit")
        donor_payload = _load_required_json(donor_path, "Donor audit")
        if (
            donor_payload.get("format") != "i2_bounded_donor_capacity_audit_v1"
            or donor_payload.get("scene") != scene_cfg["scene"]
            or donor_payload.get("contiguous_joint_safe_capacity") != donor_capacity
            or donor_payload.get("capacity_determined") is not True
            or donor_payload.get("test_views_used") is not False
        ):
            raise PipelineError("Adaptive Donor audit binding differs")
    initialization = evidence.get("initialization", [])
    if not isinstance(initialization, list):
        raise PipelineError("Adaptive initialization evidence bindings are invalid")
    for index, reference in enumerate(initialization, 1):
        initialization_path = _verified_reference(reference, f"initialization K{index}")
        initialization_payload = _load_required_json(
            initialization_path, f"Initialization K{index}"
        )
        expected_safe = index <= initialization_capacity
        if (
            initialization_payload.get("format")
            != "i2_adaptive_joint_initialization_audit_v1"
            or initialization_payload.get("k") != index
            or initialization_payload.get("safe") is not expected_safe
            or initialization_payload.get("test_views_used") is not False
        ):
            raise PipelineError(f"Adaptive initialization prefix K{index} is invalid")
    expected_initialization_rows = (
        initialization_capacity
        if initialization_capacity == min(20, receiver_capacity, donor_capacity)
        else initialization_capacity + 1
    )
    if len(initialization) != expected_initialization_rows:
        raise PipelineError("Adaptive initialization prefix is incomplete or continued after failure")
    if selected == 0:
        if decision.get("execution_route") != "control_no_intervention" or "adaptive_plan" in decision:
            raise PipelineError("K=0 adaptive decision is not a pure Control route")
        return {"decision": path, "selected_k": 0, "route": "control", "plan": None, "audit": None}
    if decision.get("execution_route") != "adaptive_plan_available_no_training_started":
        raise PipelineError("Positive adaptive decision has no executable Plan route")
    plan = _verified_reference(decision.get("adaptive_plan"), "adaptive Plan")
    if not isinstance(initialization, list) or len(initialization) < selected:
        raise PipelineError("Adaptive decision lacks the selected initialization prefix")
    audit = _verified_reference(initialization[selected - 1], "selected initialization audit")
    validate_external_training_binding(
        scene_cfg, method_cfg, active_plan=plan, requested_budget=selected,
        initialization_audit=audit,
    )
    return {
        "decision": path, "selected_k": selected, "route": "receiver_v2",
        "plan": plan, "audit": audit,
    }


def _validate_rd_v1_training_binding(scene_cfg: dict, method_cfg: dict, *,
                                     path: Path, decision: dict) -> dict:
    if (
        decision.get("format") != "i2_adaptive_k_rd_v1_decision"
        or decision.get("scene") != scene_cfg["scene"]
        or decision.get("uses_test_views") is not False
    ):
        raise PipelineError("RD-v1 Decision identity or evidence scope is invalid")
    selected = decision.get("selected_k")
    if selected == "UNKNOWN":
        raise PipelineError("RD-v1 Decision is UNKNOWN and blocks execution")
    r, d = decision.get("receiver_capacity"), decision.get("donor_capacity")
    if any(type(value) is not int or value < 0 for value in (r, d, selected)) or selected != min(r, d):
        raise PipelineError("RD-v1 Decision violates K=min(R,D)")
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None or decision.get("source_checkpoint_sha256", "").lower() != checkpoint_sha:
        raise PipelineError("RD-v1 Decision common-source binding differs")
    bindings = decision.get("input_bindings")
    if not isinstance(bindings, dict):
        raise PipelineError("RD-v1 Decision input bindings are missing")
    scene_path = _verified_reference(bindings.get("scene_config"), "scene_config")
    method_path = _verified_reference(bindings.get("method_config"), "method_config")
    policy_path = _verified_reference(bindings.get("policy_config"), "policy_config")
    source_path = _verified_reference(bindings.get("common_source"), "common_source")
    if _load_json(scene_path) != scene_cfg or _load_json(method_path) != method_cfg:
        raise PipelineError("RD-v1 Decision configuration binding differs")
    policy = _load_json(policy_path)
    if (
        policy.get("schema_version") != "i2-adaptive-k-policy-v2"
        or policy.get("policy_id") != "i2_adaptive_k_rd"
        or policy.get("version") != "RD-v1"
        or policy.get("budget_rule") != "K=min(R,D)"
        or "candidate_budget_cap" in policy
        or policy.get("receiver", {}).get("require_complete_enumeration") is not True
        or policy.get("donor", {}).get("require_complete_enumeration") is not True
        or policy.get("donor", {}).get("performance_gates_used") is not False
    ):
        raise PipelineError("RD-v1 policy binding is invalid")
    if source_path != checkpoint.resolve():
        raise PipelineError("RD-v1 common-source path differs")
    if bindings.get("source_binding") is not None:
        _verified_reference(bindings["source_binding"], "source_binding")
    prepare = bindings.get("prepare")
    if not isinstance(prepare, dict) or set(prepare) != {"receiver_manifest", "receiver_report"}:
        raise PipelineError("RD-v1 prepare bindings are incomplete")
    for label, reference in prepare.items():
        _verified_reference(reference, f"prepare.{label}")
    evidence = decision.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"receiver", "donor"}:
        raise PipelineError("RD-v1 evidence bindings are incomplete")
    receiver_path = _verified_reference(evidence["receiver"], "RD-v1 Receiver audit")
    donor_path = _verified_reference(evidence["donor"], "RD-v1 Donor audit")
    receiver = _load_required_json(receiver_path, "RD-v1 Receiver audit")
    donor = _load_required_json(donor_path, "RD-v1 Donor audit")
    embedded = decision.get("receiver_audit")
    if (
        receiver.get("receiver_budget") != embedded
        or embedded.get("receiver_capacity") != r
        or embedded.get("complete_sequence_audited") is not True
        or embedded.get("positive_prefix_exact_match") is not True
        or len(embedded.get("accepted_prefix_ids", []))
        != embedded.get("geometry_nms_capacity_g")
        or receiver.get("coverage", {}).get("complete") is not True
        or receiver.get("test_views_used") is not False
        or donor.get("donor_capacity") != d
        or donor.get("coverage_complete") is not True
        or donor.get("performance_gates_used") is not False
        or donor.get("render_counterfactuals_used") is not False
        or donor.get("test_views_used") is not False
    ):
        raise PipelineError("RD-v1 capacity evidence is incomplete or inconsistent")
    if selected == 0:
        if decision.get("execution_route") != "control_no_intervention" or "adaptive_plan" in decision:
            raise PipelineError("RD-v1 K=0 must be a pure Control route")
        return {
            "decision": path, "selected_k": 0, "route": "control",
            "plan": None, "audit": None, "policy_version": "RD-v1",
        }
    if decision.get("execution_route") != "adaptive_plan_available_no_training_started":
        raise PipelineError("RD-v1 positive K has no executable route")
    plan_path = _verified_reference(decision.get("adaptive_plan"), "RD-v1 Plan")
    plan_payload = _load_required_json(plan_path, "RD-v1 Plan")
    plan_manifest = _verified_reference(
        plan_payload.get("training_view_manifest"), "RD-v1 training view manifest"
    )
    expected_manifest = resolve_path(scene_cfg["view_manifest"]["path"])
    if (
        expected_manifest is None or plan_manifest != expected_manifest.resolve()
        or sha256(plan_manifest).lower() != str(scene_cfg["view_manifest"]["sha256"]).lower()
        or plan_payload.get("scene") != scene_cfg["scene"]
    ):
        raise PipelineError("RD-v1 Plan training-view or scene binding differs")
    from utils.i2_active_pilot import ActivePilotError, load_active_plan
    try:
        load_active_plan(plan_path, requested_budget=selected, source_checkpoint=checkpoint)
    except ActivePilotError as exc:
        raise PipelineError(f"RD-v1 Plan validation failed: {exc}") from exc
    return {
        "decision": path, "selected_k": selected, "route": "receiver_v2",
        "plan": plan_path, "audit": None, "policy_version": "RD-v1",
    }


def validate_adaptive_smoke_metadata(
    smoke: dict, *, scene: str, arm: str, requested_k: int,
    plan_sha256: str | None, initialization_audit_sha256: str | None,
    adaptive_decision_sha256: str, source_checkpoint_sha256: str,
    view_manifest_sha256: str, method_config_sha256: str,
) -> None:
    """Require exact adaptive Decision identity before a Formal launch."""
    expected = {
        "status": "PASS", "scene": scene, "stage": "smoke", "arm": arm,
        "requested_k": requested_k, "plan_sha256": plan_sha256,
        "initialization_audit_sha256": initialization_audit_sha256,
        "initialization_plan_sha256": plan_sha256,
        "adaptive_decision_sha256": adaptive_decision_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "view_manifest_sha256": view_manifest_sha256.lower(),
        "method_config_sha256": method_config_sha256,
    }
    actual = {
        "status": smoke.get("status"), "scene": smoke.get("scene"),
        "stage": smoke.get("stage"), "arm": smoke.get("arm"),
        "requested_k": smoke.get("requested_k"),
        "plan_sha256": smoke.get("plan_sha256"),
        "initialization_audit_sha256": smoke.get("initialization_audit_sha256"),
        "initialization_plan_sha256": smoke.get("initialization_plan_sha256"),
        "adaptive_decision_sha256": smoke.get("adaptive_decision_sha256"),
        "source_checkpoint_sha256": smoke.get("source_checkpoint_sha256"),
        "view_manifest_sha256": str(smoke.get("view_manifest_sha256", "")).lower(),
        "method_config_sha256": smoke.get("fingerprints", {}).get(
            "method_config_sha256"
        ),
    }
    mismatches = [key for key, value in expected.items() if actual[key] != value]
    if mismatches:
        raise PipelineError(
            f"Adaptive Formal smoke identity mismatch: {', '.join(mismatches)}"
        )


def script_cli_options(script: str | Path) -> set[str]:
    """Statically extract long options accepted by a tool's argparse parsers."""
    tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                options.add(arg.value)
    return options


def _prepare_contracts(paths: dict[str, Path], checkpoint: Path, cfg: Path,
                       k: int) -> list[dict]:
    b1_manifest = paths["b1"] / "d_input_manifest.json"
    receiver_manifest = paths["receivers"] / "d_input_manifest_with_receivers.json"
    receiver_report = paths["receivers"] / "receiver_evidence_report.json"
    d0_report = paths["d0"] / "d0_dry_run_report.json"
    v1_plan = paths["counterfactual"] / f"{paths['d0'].name}_active_plan.json"
    joint = paths["counterfactual"] / f"donor_joint_deletion_counterfactual_k{k}.json"
    v2_plan = paths["receiver_v2"] / f"i2_d1_receiver_v2_k{k}_plan.json"
    v2_evidence = paths["receiver_v2"] / "receiver_v2_geometry_evidence.json"
    init_audit = paths["initialization"] / "initialization_audit.json"
    specs = [
        ("b1", "tools/analyze_i2_b1c.py",
         [(checkpoint, "file", None)], [(b1_manifest, "file")]),
        ("receiver_candidates", "tools/build_i2_receivers.py",
         [(checkpoint, "file", None), (cfg, "file", None), (b1_manifest, "file", "b1")],
         [(receiver_manifest, "file"), (receiver_report, "file")]),
        ("d0", "tools/build_i2_d0_dry_run.py",
         [(receiver_manifest, "file", "receiver_candidates")], [(d0_report, "file")]),
        ("donor_counterfactual", "tools/audit_i2_donor_counterfactual.py",
         [(checkpoint, "file", None), (cfg, "file", None), (d0_report, "file", "d0")],
         [(v1_plan, "file"), (joint, "file"),
          (paths["counterfactual"] / "donor_static_deletion_counterfactual.json", "file")]),
        ("receiver_v2", "tools/build_i2_receiver_v2.py",
         [(checkpoint, "file", None), (cfg, "file", None),
          (v1_plan, "file", "donor_counterfactual"),
          (receiver_report, "file", "receiver_candidates")],
         [(v2_plan, "file"), (v2_evidence, "file")]),
        ("initialization_audit", "tools/audit_i2_receiver_v2_initialization.py",
         [(checkpoint, "file", None), (cfg, "file", None),
          (v1_plan, "file", "donor_counterfactual"),
          (v2_plan, "file", "receiver_v2"), (v2_evidence, "file", "receiver_v2")],
         [(init_audit, "file")]),
    ]
    contracts = []
    for name, script, inputs, outputs in specs:
        script_path = ROOT / script
        contracts.append({
            "name": name,
            "script": str(script_path),
            "accepted_cli_options": sorted(script_cli_options(script_path)),
            "inputs": [
                {"path": str(path), "type": kind, "produced_by": producer}
                for path, kind, producer in inputs
            ],
            "outputs": [{"path": str(path), "type": kind} for path, kind in outputs],
            "framework_creates_output_parent": name == "initialization_audit",
        })
    return contracts


def validate_prepare_contracts(commands: list[dict], contracts: list[dict]) -> None:
    if [row["name"] for row in commands] != [row["name"] for row in contracts]:
        raise PipelineError("Prepare command/contract stage order differs")
    outputs_by_stage = {
        row["name"]: {(item["path"], item["type"]) for item in row["outputs"]}
        for row in contracts
    }
    for command, contract in zip(commands, contracts):
        if Path(command["argv"][2]).resolve() != Path(contract["script"]).resolve():
            raise PipelineError(f"Prepare script mismatch for {contract['name']}")
        used = {value for value in command["argv"] if value.startswith("--")}
        accepted = set(contract["accepted_cli_options"])
        if not used.issubset(accepted):
            raise PipelineError(
                f"Unsupported CLI option(s) for {contract['name']}: {sorted(used - accepted)}"
            )
        for item in contract["inputs"]:
            producer = item["produced_by"]
            if producer and (item["path"], item["type"]) not in outputs_by_stage.get(producer, set()):
                raise PipelineError(
                    f"{contract['name']} input is not an exact {producer} output: {item['path']}"
                )


def build_adaptive_evidence_plan(scene_cfg: dict, method_cfg: dict,
                                 output_root: str | Path, *, rd_v1: bool = False) -> dict:
    """Build only K-independent B1 and Receiver candidate evidence."""
    root = Path(output_root).resolve()
    paths = {
        "root": root,
        "b1": root / "b1",
        "receivers": root / "receivers",
        "d0": root / "unused_d0",
        "counterfactual": root / "unused_counterfactual",
        "receiver_v2": root / "unused_receiver_v2",
        "initialization": root / "unused_initialization",
    }
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Adaptive evidence preparation requires a common source")
    cfg = checkpoint.parent / "cfg_args"
    python = method_cfg["runtime"]["python"]
    candidates = method_cfg["receiver_candidates"]
    commands = [
        {
            "name": "b1",
            "argv": [str(value) for value in (
                python, "-u", ROOT / "tools/analyze_i2_b1c.py", "--checkpoint", checkpoint,
                "--output-dir", paths["b1"],
            )],
        },
        {
            "name": "receiver_candidates",
            "argv": [str(value) for value in (
                python, "-u", ROOT / "tools/build_i2_receivers.py", "--checkpoint", checkpoint,
                "--cfg", cfg, "--d_manifest", paths["b1"] / "d_input_manifest.json",
                "--output", paths["receivers"], "--block_size", candidates["block_size"],
                "--per_view_limit", 0 if rd_v1 else candidates["per_view_limit"],
                "--max_candidates", 0 if rd_v1 else candidates["max_candidates"],
                "--min_alpha", candidates["minimum_alpha"],
                "--max_reprojection_error", candidates["max_reprojection_error_px"],
                "--spatial_nms_fraction", candidates["spatial_nms_fraction"],
                "--density_radius_fraction", candidates["density_radius_fraction"],
            )],
        },
    ]
    contracts = _prepare_contracts(paths, checkpoint, cfg, 0)[:2]
    validate_prepare_contracts(commands, contracts)
    return {
        "format": "i2_adaptive_base_evidence_plan_v1",
        "output_root": str(root), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "receiver_manifest": str(paths["receivers"] / "d_input_manifest_with_receivers.json"),
        "receiver_report": str(paths["receivers"] / "receiver_evidence_report.json"),
        "commands": commands, "contracts": contracts,
        "contains_k_dependent_safety_audit": False,
        "receiver_enumeration": "complete" if rd_v1 else "legacy_bounded",
    }


def build_prepare_plan(scene_cfg: dict, method_cfg: dict) -> dict:
    paths = prepare_paths(scene_cfg)
    python = method_cfg["runtime"]["python"]
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Prepare requires a common-source checkpoint")
    cfg = checkpoint.parent / "cfg_args"
    k = int(method_cfg["budget"]["requested_k"])
    donor = method_cfg["donor"]
    candidates = method_cfg["receiver_candidates"]
    receiver = method_cfg["receiver_v2"]
    init = method_cfg["initialization_audit"]
    commands = []
    def add(name, argv):
        commands.append({"name": name, "argv": [str(item) for item in argv]})
    add("b1", [python, "-u", ROOT / "tools/analyze_i2_b1c.py", "--checkpoint", checkpoint,
               "--output-dir", paths["b1"]])
    add("receiver_candidates", [
        python, "-u", ROOT / "tools/build_i2_receivers.py", "--checkpoint", checkpoint,
        "--cfg", cfg, "--d_manifest", paths["b1"] / "d_input_manifest.json",
        "--output", paths["receivers"], "--block_size", candidates["block_size"],
        "--per_view_limit", candidates["per_view_limit"], "--max_candidates", candidates["max_candidates"],
        "--min_alpha", candidates["minimum_alpha"], "--max_reprojection_error", candidates["max_reprojection_error_px"],
        "--spatial_nms_fraction", candidates["spatial_nms_fraction"],
        "--density_radius_fraction", candidates["density_radius_fraction"],
    ])
    add("d0", [
        python, "-u", ROOT / "tools/build_i2_d0_dry_run.py", "--manifest",
        paths["receivers"] / "d_input_manifest_with_receivers.json", "--output-dir", paths["d0"],
        "--budget", k, "--min-age", donor["minimum_age"],
        "--min-opportunities", donor["minimum_real_view_opportunities"],
    ])
    add("donor_counterfactual", [
        python, "-u", ROOT / "tools/audit_i2_donor_counterfactual.py", "--checkpoint", checkpoint,
        "--cfg", cfg, "--plans", paths["d0"] / "d0_dry_run_report.json", "--output", paths["counterfactual"],
        "--max_mean_abs_rgb_delta", donor["max_single_or_joint_mean_abs_rgb_delta"],
        "--max_psnr_drop_db", donor["max_single_or_joint_psnr_drop_db"],
    ])
    v1_plan = paths["counterfactual"] / f"{paths['d0'].name}_active_plan.json"
    add("receiver_v2", [
        python, "-u", ROOT / "tools/build_i2_receiver_v2.py", "--checkpoint", checkpoint,
        "--cfg", cfg, "--v1_plan", v1_plan, "--receiver_report",
        paths["receivers"] / "receiver_evidence_report.json", "--output", paths["receiver_v2"],
        "--max_condition", receiver["max_projection_condition"],
        "--displacement_factor", receiver["max_displacement_tangent_norm_factor"],
        "--opacity", receiver["activated_opacity"], "--footprint_pixels", receiver["footprint_pixels"],
        "--budget", k,
    ])
    v2_plan = paths["receiver_v2"] / f"i2_d1_receiver_v2_k{k}_plan.json"
    add("initialization_audit", [
        python, "-u", ROOT / "tools/audit_i2_receiver_v2_initialization.py",
        "--checkpoint", checkpoint, "--cfg", cfg, "--v1_plan", v1_plan,
        "--v2_plan", v2_plan, "--v2_evidence", paths["receiver_v2"] / "receiver_v2_geometry_evidence.json",
        "--output", initialization_audit_path(scene_cfg),
        "--max_target_l1_increase", init["max_target_l1_increase"],
        "--max_outside_l1_increase", init["max_outside_l1_increase"],
        "--max_global_l1_increase", init["max_global_l1_increase"],
        "--budget", k,
    ])
    contracts = _prepare_contracts(paths, checkpoint, cfg, k)
    validate_prepare_contracts(commands, contracts)
    return {
        "format": "i2_scene_prepare_plan_v1", "scene": scene_cfg["scene"],
        "requested_k": k, "output_root": str(paths["root"]),
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": checkpoint_sha,
        "commands": commands, "contracts": contracts,
        "uses_training_views_only": True, "performs_training": False,
    }


def source_command(scene_cfg: dict, method_cfg: dict) -> tuple[list[str], Path, Path]:
    namespace = scene_cfg["output_namespace"]
    suffix = _source_suffix(scene_cfg)
    local_pdc = ROOT / "keypoints_to_3d" / f"LLFF_{suffix}" / f"{scene_cfg['scene']}_keypoints_to_3d.ply"
    output = ROOT / "output" / f"{namespace}_CommonSource_I1_A0B0_1201"
    source = method_cfg["source"]
    training = method_cfg["training"]
    command = [
        method_cfg["runtime"]["python"], "-u", str(ROOT / "train.py"),
        "-s", str(resolve_path(scene_cfg["dataset_root"])), "-m", str(output),
        "--dataset_name", training["dataset_name"], "--eval", "--n_views", training["n_views"],
        "--resolution", training["resolution"], "--sh_degree", training["sh_degree"],
        "--view_manifest", str(resolve_path(scene_cfg["view_manifest"]["path"])), "--suffix", suffix,
        "--iterations", source["iteration"], "--dropout_factor", training["dropout_factor"],
        "--sigma_noise", training["sigma_noise"], "--densify_from_iter", training["densify_from_iteration"],
        "--densification_interval", training["densification_interval"],
        "--densify_grad_threshold", training["densify_grad_threshold"], "--enable_oe_structural_loss",
        "--oe_structural_weight", source["i1_weight"], "--oe_structural_start_iter", source["i1_start_iteration"],
        "--oe_structural_end_iter", source["i1_end_iteration"], "--oe_structure_gate_mode", source["structure_gate_mode"],
        "--oe_stable_quantile", source["stable_quantile"], "--oe_preserve_baseline_densification_stats",
        "--oe_depth_cache_dir", str(resolve_path(scene_cfg["depth_cache"]["path"])),
        "--enable_i2_lifecycle_tracking", "--enable_i2_observation_tracking",
        "--checkpoint_iterations", source["iteration"], "--test_iterations", source["iteration"],
        "--save_iterations", source["iteration"],
    ]
    return [str(value) for value in command], local_pdc, output


def _plan_path(scene_cfg: dict, method_cfg: dict) -> Path:
    k = method_cfg["budget"]["requested_k"]
    return prepare_paths(scene_cfg)["receiver_v2"] / f"i2_d1_receiver_v2_k{k}_plan.json"


def _training_output_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_external_training_binding(
    scene_cfg: dict,
    method_cfg: dict,
    *,
    active_plan: str | Path,
    requested_budget: int,
    initialization_audit: str | Path,
) -> tuple[Path, Path, dict]:
    """Validate a pre-audited external plan without touching frozen prepare outputs."""
    checkpoint, _, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("External Plan validation requires a bound common source")
    plan_path = resolve_path(str(active_plan))
    audit_path = resolve_path(str(initialization_audit))
    assert plan_path is not None and audit_path is not None
    from utils.i2_active_pilot import ActivePilotError, load_active_plan

    try:
        plan = load_active_plan(
            plan_path,
            requested_budget=int(requested_budget),
            source_checkpoint=checkpoint,
        )
    except (ActivePilotError, OSError, ValueError) as exc:
        raise PipelineError(f"External active Plan validation failed: {exc}") from exc
    protocol = plan.get("input_audit", {}).get("receiver_protocol", {})
    if (
        plan.get("input_audit", {}).get("source_iteration")
        != int(method_cfg["source"]["iteration"])
        or protocol.get("training_camera_ids") != scene_cfg["split"]["train_images"]
        or protocol.get("view_manifest_sha256", "").lower()
        != scene_cfg["view_manifest"]["sha256"].lower()
    ):
        raise PipelineError(
            "External active Plan differs from the frozen source iteration or training views"
        )
    audit = _load_required_json(audit_path, "External initialization audit")
    gates = audit.get("gates")
    plan_ref = audit.get("plan")
    evidence_ref = audit.get("evidence")
    receiver_ref = plan.get("receiver_v2", {}).get("artifact")
    if (
        audit.get("format") != "i2_adaptive_joint_initialization_audit_v1"
        or audit.get("safe") is not True
        or audit.get("training_views_only") is not True
        or audit.get("test_views_used") is not False
        or int(audit.get("k", -1)) != int(requested_budget)
        or not isinstance(gates, dict)
        or set(gates) != ADAPTIVE_INITIALIZATION_GATE_NAMES
        or any(value is not True for value in gates.values())
        or not isinstance(plan_ref, dict)
        or str(plan_ref.get("sha256", "")).lower() != sha256(plan_path)
        or not isinstance(evidence_ref, dict)
        or not isinstance(receiver_ref, dict)
        or str(evidence_ref.get("sha256", "")).lower()
        != str(receiver_ref.get("sha256", "")).lower()
    ):
        raise PipelineError("External initialization audit is not safely bound to the Plan")
    return plan_path, audit_path, audit


def training_command(scene_cfg: dict, method_cfg: dict, *, stage: str,
                     arm: str = "receiver_v2", active_plan: str | Path | None = None,
                     requested_budget: int | None = None,
                     output_dir: str | Path | None = None,
                     adaptive_decision: str | Path | None = None) -> tuple[list[str], Path, int]:
    if arm not in ARMS + ("birth_only",):
        raise PipelineError(f"Unknown arm: {arm}")
    checkpoint, _, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Training requires a common source")
    pdc = resolve_training_pdc(scene_cfg, method_cfg)
    protocol = method_cfg["smoke"] if stage == "smoke" else method_cfg["training"]
    final_iteration = int(protocol["final_iteration"])
    k = int(
        method_cfg["budget"]["requested_k"]
        if requested_budget is None else requested_budget
    )
    if k < 0 or (k == 0 and arm != "control"):
        raise PipelineError("Training budget K must be positive for intervention arms")
    namespace = scene_cfg["output_namespace"]
    label = "Smoke" if stage == "smoke" else "Formal"
    arm_label = {
        "control": "Control", "release": "ReleaseOnly",
        "receiver_v2": "ReceiverV2", "birth_only": "BirthOnly",
    }[arm]
    default_output = ROOT / "output" / f"{namespace}_{label}_{arm_label}_K{k}_{final_iteration}"
    output = _training_output_path(output_dir, default_output)
    training = method_cfg["training"]
    command = [
        method_cfg["runtime"]["python"], "-u", str(ROOT / "train.py"),
        "-s", str(resolve_path(scene_cfg["dataset_root"])), "-m", str(output),
        "--dataset_name", training["dataset_name"], "--eval", "--n_views", training["n_views"],
        "--resolution", training["resolution"], "--sh_degree", training["sh_degree"],
        "--view_manifest", str(resolve_path(scene_cfg["view_manifest"]["path"])),
        "--suffix", pdc["suffix"],
        "--iterations", final_iteration, "--dropout_factor", training["dropout_factor"],
        "--sigma_noise", training["sigma_noise"], "--densify_from_iter", training["densify_from_iteration"],
        "--densification_interval", training["densification_interval"],
        "--densify_grad_threshold", training["densify_grad_threshold"],
        "--checkpoint_iterations", *protocol["checkpoint_iterations"],
        "--test_iterations", *protocol["test_iterations"], "--save_iterations", *protocol["save_iterations"],
        "--enable_i2_lifecycle_tracking", "--enable_i2_observation_tracking",
        "--start_checkpoint", str(checkpoint),
    ]
    if arm != "control":
        command.extend([
            "--enable_i2_release_only" if arm == "release" else
            "--enable_i2_birth_only" if arm == "birth_only" else
            "--enable_i2_active_pilot",
            "--i2_active_plan", str(
                resolve_path(str(active_plan)) if active_plan is not None
                else _plan_path(scene_cfg, method_cfg)
            ), "--i2_active_budget", k,
            "--i2_active_event_log", str(output / "active_event.json"),
        ])
        if adaptive_decision is not None:
            command.extend(["--i2_adaptive_decision", str(resolve_path(str(adaptive_decision)))])
    return [str(value) for value in command], output, final_iteration


def run_logged(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            log.write(block); log.flush(); sys.stdout.buffer.write(block); sys.stdout.buffer.flush()
        code = process.wait()
    if code:
        raise PipelineError(f"Command failed with exit code {code}: {command}")
    return time.perf_counter() - started


def _verify_contract_outputs(contract: dict) -> None:
    for item in contract["outputs"]:
        path = Path(item["path"])
        exists = path.is_file() if item["type"] == "file" else path.is_dir()
        if not exists:
            raise PipelineError(
                f"{contract['name']} did not produce contracted {item['type']}: {path}"
            )


def execute_commands(commands: list[dict], output_root: Path,
                     contracts: list[dict] | None = None) -> None:
    if output_root.exists():
        raise PipelineError(f"Refusing to reuse stage output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    if contracts is not None:
        validate_prepare_contracts(commands, contracts)
    for index, entry in enumerate(commands):
        contract = None if contracts is None else contracts[index]
        if contract and contract["framework_creates_output_parent"]:
            for item in contract["outputs"]:
                if item["type"] == "file":
                    Path(item["path"]).parent.mkdir(parents=True, exist_ok=True)
        run_logged(entry["argv"], output_root / f"{entry['name']}.log")
        if contract:
            _verify_contract_outputs(contract)


def recover_prepare_commands(commands: list[dict], output_root: Path,
                             contracts: list[dict], *, start_stage: str,
                             final_validator) -> tuple[Path, dict]:
    """Resume a failed prepare without rewriting its original failure record."""
    validate_prepare_contracts(commands, contracts)
    manifest = output_root / "stage_manifest.json"
    if not output_root.is_dir() or not manifest.is_file():
        raise PipelineError("Prepare recovery requires an existing failed stage manifest")
    original = _load_json(manifest)
    if original.get("status") != "FAILED":
        raise PipelineError("Prepare recovery is allowed only from an audited FAILED run")
    names = [entry["name"] for entry in commands]
    if start_stage not in names:
        raise PipelineError(f"Unknown prepare recovery stage: {start_stage}")
    start_index = names.index(start_stage)
    reused = []
    for contract in contracts[:start_index]:
        _verify_contract_outputs(contract)
        reused.append({
            "stage": contract["name"],
            "outputs": [
                {**item, "sha256": sha256(Path(item["path"]))}
                for item in contract["outputs"] if item["type"] == "file"
            ],
        })
    for contract in contracts[start_index:]:
        for item in contract["inputs"]:
            path = Path(item["path"])
            exists = path.is_file() if item["type"] == "file" else path.is_dir()
            producer_index = names.index(item["produced_by"]) if item["produced_by"] else -1
            if producer_index < start_index and not exists:
                raise PipelineError(
                    f"Recovery input from reused stage is missing: {path}"
                )

    recovery_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    recovery_root = output_root / "recovery" / recovery_id
    backup_root = recovery_root / "failed_artifacts"
    recovery_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, recovery_root / "original_stage_manifest.json")
    backed_up = []
    stage_roots = []
    for contract in contracts[start_index:]:
        parents = {Path(item["path"]).parent for item in contract["outputs"]}
        if len(parents) == 1:
            stage_root = next(iter(parents))
            if stage_root.parent == output_root and stage_root.exists():
                stage_roots.append(stage_root)
    for stage_root in stage_roots:
        destination = backup_root / stage_root.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage_root), str(destination))
        backed_up.append({"path": str(destination), "source": str(stage_root)})
    for entry in commands[start_index:]:
        old_log = output_root / f"{entry['name']}.log"
        if old_log.is_file():
            destination = backup_root / "logs" / old_log.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_log, destination)
            backed_up.append({
                "path": str(destination), "source": str(old_log),
                "sha256": sha256(destination),
            })

    audit = {
        "format": "i2_scene_prepare_recovery_v1",
        "status": "RUNNING",
        "start_stage": start_stage,
        "original_stage_manifest": str(manifest),
        "original_stage_manifest_sha256": sha256(manifest),
        "reused_stages": reused,
        "backed_up_failed_artifacts": backed_up,
        "commands": commands[start_index:],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path = recovery_root / "recovery_manifest.json"
    try:
        for index in range(start_index, len(commands)):
            entry = commands[index]
            contract = contracts[index]
            if contract["framework_creates_output_parent"]:
                for item in contract["outputs"]:
                    if item["type"] == "file":
                        Path(item["path"]).parent.mkdir(parents=True, exist_ok=True)
            run_logged(entry["argv"], recovery_root / f"{entry['name']}.log")
            _verify_contract_outputs(contract)
        final_validator()
        for backup in backup_root.rglob("*"):
            if backup.is_file():
                backup.chmod(stat.S_IREAD)
        audit["failed_artifact_backup_read_only"] = True
        audit["produced_stages"] = [
            {
                "stage": contract["name"],
                "outputs": [
                    {**item, "sha256": sha256(Path(item["path"]))}
                    for item in contract["outputs"] if item["type"] == "file"
                ],
            }
            for contract in contracts[start_index:]
        ]
        audit["status"] = "PASS"
    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error"] = repr(exc)
        raise
    finally:
        audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit_path, audit


def _load_required_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise PipelineError(f"{label} is missing: {path}")
    return _load_json(path)


def validate_donor_plan(scene_cfg: dict, method_cfg: dict) -> dict:
    paths = prepare_paths(scene_cfg)
    k = int(method_cfg["budget"]["requested_k"])
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Donor audit requires a bound source")
    plan_path = paths["counterfactual"] / f"{paths['d0'].name}_active_plan.json"
    plan = _load_required_json(plan_path, "Donor active plan")
    joint_ref = plan.get("joint_deletion_counterfactual")
    if not isinstance(joint_ref, dict) or joint_ref.get("format") != "i2_joint_deletion_counterfactual_v1":
        raise PipelineError("Final donor Plan does not contain joint counterfactual evidence")
    artifact_ref = joint_ref.get("artifact")
    if not isinstance(artifact_ref, dict) or not artifact_ref.get("path") or not artifact_ref.get("sha256"):
        raise PipelineError("Final donor Plan joint artifact binding is missing")
    joint_path = Path(artifact_ref["path"])
    if not joint_path.is_absolute():
        joint_path = plan_path.parent / joint_path
    joint_path = joint_path.resolve()
    if not joint_path.is_file() or sha256(joint_path) != str(artifact_ref["sha256"]).lower():
        raise PipelineError("Final donor Plan joint artifact hash is invalid")
    joint = _load_required_json(joint_path, "Joint donor counterfactual")
    donor_uids = [int(row["uid"]) for row in plan.get("donors", {}).get("selected", [])]
    if (
        len(donor_uids) != k
        or joint.get("source_checkpoint_sha256") != checkpoint_sha
        or joint.get("donor_uids") != donor_uids
        or joint.get("test_views_used") is not False
        or joint.get("safe_to_release") is not True
        or plan.get("safety_readiness", {}).get("joint_deletion_counterfactual") is not True
        or plan.get("active_intervention_ready") is not True
    ):
        raise PipelineError("Final donor Plan failed the joint counterfactual safety contract")
    return {"path": str(plan_path), "sha256": sha256(plan_path), "joint": str(joint_path)}


def validate_initialization_gate(scene_cfg: dict, method_cfg: dict) -> tuple[Path, dict]:
    path = initialization_audit_path(scene_cfg)
    audit = _load_required_json(path, "Initialization audit")
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Initialization gate requires a bound source")
    k = int(method_cfg["budget"]["requested_k"])
    paths = prepare_paths(scene_cfg)
    v1_plan = paths["counterfactual"] / f"{paths['d0'].name}_active_plan.json"
    v2_plan = paths["receiver_v2"] / f"i2_d1_receiver_v2_k{k}_plan.json"
    evidence = paths["receiver_v2"] / "receiver_v2_geometry_evidence.json"
    gates = audit.get("gates")
    if audit.get("status") != "GO" or audit.get("test_views_used") is not False:
        raise PipelineError("Initialization audit is not GO or used test views")
    if (
        not isinstance(gates, dict)
        or set(gates) != INITIALIZATION_GATE_NAMES
        or any(value is not True for value in gates.values())
    ):
        raise PipelineError("Initialization audit contains a failed or malformed safety gate")
    expected_hashes = {
        "source_checkpoint_sha256": checkpoint_sha,
        "receiver_v1_plan_sha256": sha256(v1_plan),
        "receiver_v2_plan_sha256": sha256(v2_plan),
        "receiver_v2_evidence_sha256": sha256(evidence),
    }
    if any(audit.get(key) != value for key, value in expected_hashes.items()):
        raise PipelineError("Initialization audit source/Plan/evidence hash binding is invalid")
    donor = validate_donor_plan(scene_cfg, method_cfg)
    v2_payload = _load_required_json(v2_plan, "Receiver-v2 Plan")
    if v2_payload.get("joint_deletion_counterfactual") != _load_required_json(
        Path(donor["path"]), "Donor active plan"
    ).get("joint_deletion_counterfactual"):
        raise PipelineError("Receiver-v2 Plan did not preserve the joint donor counterfactual")
    return path, audit


def execute_training(scene_cfg: dict, method_cfg: dict, fingerprints: dict,
                     *, stage: str, arm: str, active_plan: str | Path | None = None,
                     requested_budget: int | None = None,
                     output_dir: str | Path | None = None,
                     initialization_audit: str | Path | None = None,
                     skip_evaluation: bool = False,
                     adaptive_decision: str | Path | None = None,
                     paired_control: bool = False,
                     mechanism_arm: str | None = None) -> dict:
    adaptive_binding = None
    if adaptive_decision is not None:
        adaptive_binding = validate_adaptive_training_binding(
            scene_cfg, method_cfg, decision_path=adaptive_decision
        )
        expected_arm = "control" if paired_control else (mechanism_arm or adaptive_binding["route"])
        expected_plan = None if paired_control else adaptive_binding["plan"]
        expected_audit = None if paired_control else adaptive_binding["audit"]
        if (
            arm != expected_arm
            or requested_budget != adaptive_binding["selected_k"]
            or active_plan != expected_plan
            or initialization_audit != expected_audit
        ):
            raise PipelineError("Training arguments differ from the adaptive decision binding")
    command, output, iteration = training_command(
        scene_cfg, method_cfg, stage=stage, arm=arm, active_plan=active_plan,
        requested_budget=requested_budget, output_dir=output_dir,
        adaptive_decision=adaptive_decision,
    )
    checkpoint, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
    if checkpoint is None:
        raise PipelineError("Training requires a bound common source")
    k = int(method_cfg["budget"]["requested_k"] if requested_budget is None else requested_budget)
    if adaptive_binding is not None and (adaptive_binding["selected_k"] == 0 or paired_control):
        init_path = adaptive_binding["decision"]
        initialization_plan_sha = None
    elif adaptive_binding is not None and adaptive_binding.get("policy_version") == "RD-v1":
        # RD-v1 has no legacy initialization gate; the Decision validator has
        # already verified the independent Plan and all of its hashes.
        init_path = adaptive_binding["decision"]
        initialization_plan_sha = sha256(adaptive_binding["plan"])
    elif active_plan is None:
        if initialization_audit is not None:
            raise PipelineError("--initialization-audit requires --active-plan")
        init_path, init_audit = validate_initialization_gate(scene_cfg, method_cfg)
        initialization_plan_sha = init_audit["receiver_v2_plan_sha256"]
    else:
        if arm == "control":
            raise PipelineError("--active-plan is only valid for an intervention arm")
        if initialization_audit is None:
            raise PipelineError("External active Plan requires --initialization-audit")
        plan, init_path, init_audit = validate_external_training_binding(
            scene_cfg, method_cfg, active_plan=active_plan,
            requested_budget=k, initialization_audit=initialization_audit,
        )
        initialization_plan_sha = sha256(plan)
    if output.exists():
        raise PipelineError(f"Refusing to reuse training output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    plan = None if arm == "control" else (
        resolve_path(str(active_plan)) if active_plan is not None
        else _plan_path(scene_cfg, method_cfg)
    )
    metadata = {
        "format": "i2_scene_training_launch_v1", "status": "RUNNING",
        "scene": scene_cfg["scene"], "stage": stage, "arm": arm,
        "requested_k": k,
        "source_checkpoint_sha256": checkpoint_sha,
        "plan_sha256": None if plan is None else sha256(plan),
        "initialization_audit_sha256": (
            None
            if adaptive_binding is not None and adaptive_binding["audit"] is None
            else sha256(init_path)
        ),
        "initialization_plan_sha256": initialization_plan_sha,
        "adaptive_decision_sha256": (
            None if adaptive_binding is None else sha256(adaptive_binding["decision"])
        ),
        "adaptive_policy_version": (
            None if adaptive_binding is None else adaptive_binding.get("policy_version", "R0-v1")
        ),
        "mechanism_arm": mechanism_arm,
        "diagnostic_source_batch": (
            "RDv1_20261016_153009_622830" if mechanism_arm is not None else None
        ),
        "paired_control": paired_control,
        "view_manifest_sha256": scene_cfg["view_manifest"]["sha256"],
        "random_seed": method_cfg["training"]["random_seed"],
        "fingerprints": fingerprints, "argv": command,
        "i1_after_source": False, "future_utility_c": False,
        "randomness_caveat": method_cfg["randomness_caveat"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output / "launch_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        metadata["training_seconds"] = run_logged(command, output / "train.log")
        if not skip_evaluation:
            evaluation = [method_cfg["runtime"]["python"], "-u", str(ROOT / "tools/evaluate_i2_5k.py"),
                          "-m", str(output), "--iteration", str(iteration)]
            metadata["evaluation_argv"] = evaluation
            metadata["evaluation_seconds"] = run_logged(evaluation, output / "evaluation.log")
        metadata["status"] = "PASS"
    except Exception as exc:
        metadata["status"] = "FAILED"; metadata["error"] = repr(exc); raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

