"""Fail-fast, single-scene LLFF runner for Windows and local development.

No dependency is installed and no model is downloaded by this script. The
default stage is manifest generation only; GPU stages must be explicitly named.
"""

from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llff_view_protocol import (  # noqa: E402
    build_view_manifest,
    load_view_manifest,
    read_and_validate_colmap_scene,
    write_view_manifest,
)


SOURCE_COMMIT = "01677f59e9a23abcffc95a097ca59b3964fa03fd"
STAGE_ORDER = ("init", "train", "render", "metrics")
FSGS_PYTHON = r"D:\Users\huashuo\anaconda3\envs\fsgs\python.exe"
MAX_INITIALIZATION_VERTICES = 100_000_000
PLY_REQUIRED_VERTEX_PROPERTIES = frozenset(("x", "y", "z", "red", "green", "blue"))
PLY_SCALAR_TYPES = {
    "char": ("b", int), "int8": ("b", int),
    "uchar": ("B", int), "uint8": ("B", int),
    "short": ("h", int), "int16": ("h", int),
    "ushort": ("H", int), "uint16": ("H", int),
    "int": ("i", int), "int32": ("i", int),
    "uint": ("I", int), "uint32": ("I", int),
    "float": ("f", float), "float32": ("f", float),
    "double": ("d", float), "float64": ("d", float),
}


class StageFailure(RuntimeError):
    def __init__(self, stage: str, returncode: int):
        super().__init__(f"Stage {stage!r} failed with exit code {returncode}")
        self.stage = stage
        self.returncode = returncode


class InitializationArtifactError(ValueError):
    """Raised when a frozen initialization artifact cannot be trusted."""


@dataclass(frozen=True)
class DevelopmentConfig:
    scene: str = "horns"
    n_views: int = 3
    iterations: int = 5000
    fallback_iterations: int = 3000
    dropout_factor: float = 0.2
    sigma_noise: float = 0.0
    resolution: int = 8
    gpu: int = 0
    dataset_name: str = "LLFF"
    eval: bool = True
    densification_diagnostics: bool = False


@dataclass(frozen=True)
class RunPaths:
    repository_root: Path
    scene_path: Path
    experiment_dir: Path
    initialization_dir: Path
    view_manifest: Path
    run_manifest: Path


def initialization_ply_path(paths: RunPaths, config: DevelopmentConfig) -> Path:
    return paths.initialization_dir / f"{config.scene}_keypoints_to_3d.ply"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise InitializationArtifactError("expected-initialization-sha256 must be 64 hexadecimal characters")
    return normalized


def _ply_number(token: str, type_name: str):
    try:
        return PLY_SCALAR_TYPES[type_name][1](token)
    except (KeyError, ValueError, OverflowError) as exc:
        raise InitializationArtifactError(
            f"Invalid PLY value {token!r} for scalar type {type_name!r}"
        ) from exc


def _consume_exact(handle, size: int, context: str) -> None:
    remaining = size
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise InitializationArtifactError(f"{context} is truncated")
        remaining -= len(chunk)


def inspect_initialization_ply(path: str | Path) -> dict:
    """Parse a PLY without third-party modules and validate its complete payload."""
    ply_path = Path(path)
    if not ply_path.is_file():
        raise InitializationArtifactError(f"Initialization PLY does not exist: {ply_path}")
    size_bytes = ply_path.stat().st_size
    if size_bytes <= 0:
        raise InitializationArtifactError(f"Initialization PLY is empty: {ply_path}")

    elements = []
    current_element = None
    format_name = None
    header_size = 0
    with ply_path.open("rb") as handle:
        first = handle.readline()
        header_size += len(first)
        if first.rstrip(b"\r\n") != b"ply":
            raise InitializationArtifactError("PLY magic line is missing")
        while True:
            raw = handle.readline()
            if not raw:
                raise InitializationArtifactError("PLY header is truncated before end_header")
            header_size += len(raw)
            if header_size > 1024 * 1024:
                raise InitializationArtifactError("PLY header exceeds 1 MiB")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise InitializationArtifactError("PLY header is not ASCII") from exc
            if line == "end_header":
                break
            if not line or line.startswith("comment ") or line.startswith("obj_info "):
                continue
            fields = line.split()
            if fields[0] == "format":
                if len(fields) != 3 or fields[2] != "1.0" or format_name is not None:
                    raise InitializationArtifactError(f"Unsupported or duplicate PLY format line: {line!r}")
                format_name = fields[1]
                if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
                    raise InitializationArtifactError(f"Unsupported PLY format: {format_name!r}")
            elif fields[0] == "element":
                if len(fields) != 3:
                    raise InitializationArtifactError(f"Malformed PLY element line: {line!r}")
                try:
                    count = int(fields[2])
                except ValueError as exc:
                    raise InitializationArtifactError(f"Invalid PLY element count: {fields[2]!r}") from exc
                if count < 0:
                    raise InitializationArtifactError("PLY element count cannot be negative")
                current_element = {"name": fields[1], "count": count, "properties": []}
                elements.append(current_element)
            elif fields[0] == "property":
                if current_element is None:
                    raise InitializationArtifactError("PLY property appears before an element")
                if len(fields) == 3:
                    if fields[1] not in PLY_SCALAR_TYPES:
                        raise InitializationArtifactError(f"Unsupported PLY scalar type: {fields[1]!r}")
                    current_element["properties"].append(("scalar", fields[1], fields[2]))
                elif len(fields) == 5 and fields[1] == "list":
                    if fields[2] not in PLY_SCALAR_TYPES or fields[3] not in PLY_SCALAR_TYPES:
                        raise InitializationArtifactError(f"Unsupported PLY list type: {line!r}")
                    current_element["properties"].append(("list", fields[2], fields[3], fields[4]))
                else:
                    raise InitializationArtifactError(f"Malformed PLY property line: {line!r}")
            else:
                raise InitializationArtifactError(f"Unsupported PLY header line: {line!r}")

        if format_name is None:
            raise InitializationArtifactError("PLY format line is missing")
        vertex_elements = [element for element in elements if element["name"] == "vertex"]
        if len(vertex_elements) != 1:
            raise InitializationArtifactError("PLY must contain exactly one vertex element")
        vertex = vertex_elements[0]
        vertex_count = vertex["count"]
        if not 1 <= vertex_count <= MAX_INITIALIZATION_VERTICES:
            raise InitializationArtifactError(
                f"PLY vertex count {vertex_count} is outside [1, {MAX_INITIALIZATION_VERTICES}]"
            )
        vertex_names = [prop[-1] for prop in vertex["properties"]]
        if len(vertex_names) != len(set(vertex_names)):
            raise InitializationArtifactError("PLY vertex property names are not unique")
        missing = sorted(PLY_REQUIRED_VERTEX_PROPERTIES - set(vertex_names))
        if missing:
            raise InitializationArtifactError(f"PLY vertex is missing required properties: {missing}")
        for prop in vertex["properties"]:
            if prop[-1] in PLY_REQUIRED_VERTEX_PROPERTIES and prop[0] != "scalar":
                raise InitializationArtifactError(f"PLY key vertex property {prop[-1]!r} must be scalar")

        coordinate_indices = {name: vertex_names.index(name) for name in ("x", "y", "z")}
        color_indices = {name: vertex_names.index(name) for name in ("red", "green", "blue")}
        endian = "<" if format_name == "binary_little_endian" else ">"

        for element in elements:
            for record_index in range(element["count"]):
                values = []
                if format_name == "ascii":
                    raw_record = handle.readline()
                    if not raw_record:
                        raise InitializationArtifactError(
                            f"PLY payload ended in {element['name']} record {record_index}"
                        )
                    try:
                        tokens = raw_record.decode("ascii").split()
                    except UnicodeDecodeError as exc:
                        raise InitializationArtifactError("ASCII PLY payload is not ASCII") from exc
                    token_index = 0
                    for prop in element["properties"]:
                        if prop[0] == "scalar":
                            if token_index >= len(tokens):
                                raise InitializationArtifactError("ASCII PLY record has too few values")
                            values.append(_ply_number(tokens[token_index], prop[1]))
                            token_index += 1
                        else:
                            if token_index >= len(tokens):
                                raise InitializationArtifactError("ASCII PLY list count is missing")
                            list_count = _ply_number(tokens[token_index], prop[1])
                            token_index += 1
                            if not isinstance(list_count, int) or list_count < 0:
                                raise InitializationArtifactError("ASCII PLY list count is invalid")
                            if token_index + list_count > len(tokens):
                                raise InitializationArtifactError("ASCII PLY list payload is truncated")
                            for token in tokens[token_index:token_index + list_count]:
                                _ply_number(token, prop[2])
                            token_index += list_count
                            values.append(None)
                    if token_index != len(tokens):
                        raise InitializationArtifactError("ASCII PLY record has extra values")
                else:
                    for prop in element["properties"]:
                        if prop[0] == "scalar":
                            code = PLY_SCALAR_TYPES[prop[1]][0]
                            width = struct.calcsize(code)
                            raw_value = handle.read(width)
                            if len(raw_value) != width:
                                raise InitializationArtifactError("Binary PLY scalar payload is truncated")
                            values.append(struct.unpack(endian + code, raw_value)[0])
                        else:
                            count_code = PLY_SCALAR_TYPES[prop[1]][0]
                            count_width = struct.calcsize(count_code)
                            raw_count = handle.read(count_width)
                            if len(raw_count) != count_width:
                                raise InitializationArtifactError("Binary PLY list count is truncated")
                            list_count = struct.unpack(endian + count_code, raw_count)[0]
                            if not isinstance(list_count, int) or list_count < 0:
                                raise InitializationArtifactError("Binary PLY list count is invalid")
                            item_code = PLY_SCALAR_TYPES[prop[2]][0]
                            list_width = struct.calcsize(item_code) * list_count
                            _consume_exact(handle, list_width, "Binary PLY list payload")
                            values.append(None)

                if element is vertex:
                    coordinates = [values[coordinate_indices[name]] for name in ("x", "y", "z")]
                    if not all(math.isfinite(float(value)) for value in coordinates):
                        raise InitializationArtifactError("PLY contains non-finite vertex coordinates")
                    colors = [values[color_indices[name]] for name in ("red", "green", "blue")]
                    if not all(math.isfinite(float(value)) and 0 <= float(value) <= 255 for value in colors):
                        raise InitializationArtifactError("PLY contains color values outside [0, 255]")

        trailing = handle.read()
        if format_name == "ascii":
            if trailing.strip():
                raise InitializationArtifactError("ASCII PLY contains unexpected trailing data")
        elif trailing:
            raise InitializationArtifactError("Binary PLY contains unexpected trailing data")

    return {
        "path": str(ply_path.resolve()),
        "size_bytes": size_bytes,
        "sha256": sha256_file(ply_path),
        "format": format_name,
        "vertex_count": vertex_count,
        "vertex_properties": vertex_names,
    }


def stage_frozen_initialization(
    *,
    source_path: str | Path,
    expected_sha256: str,
    paths: RunPaths,
    config: DevelopmentConfig,
    view_manifest: Mapping,
) -> dict:
    on_disk_manifest = load_view_manifest(paths.view_manifest)
    if dict(view_manifest) != on_disk_manifest:
        raise InitializationArtifactError(
            "In-memory view manifest does not match the manifest written for this experiment"
        )
    source = Path(source_path).resolve()
    expected = normalize_sha256(expected_sha256)
    if not source.is_file():
        raise InitializationArtifactError(f"Initialization PLY does not exist: {source}")
    source_hash_before = sha256_file(source)
    if source_hash_before != expected:
        raise InitializationArtifactError(
            f"Initialization PLY SHA-256 mismatch: expected {expected}, got {source_hash_before}"
        )
    source_info = inspect_initialization_ply(source)
    target = initialization_ply_path(paths, config)
    target.parent.mkdir(parents=True, exist_ok=True)

    action = "reused_existing"
    if target.exists():
        if not target.is_file():
            raise InitializationArtifactError(f"Initialization target is not a file: {target}")
        target_hash = sha256_file(target)
        if target_hash != expected:
            raise InitializationArtifactError(
                f"Refusing to overwrite initialization PLY with different SHA-256: {target} "
                f"has {target_hash}, expected {expected}"
            )
    else:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".tmp",
                dir=target.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_hash = sha256_file(temporary_path)
            if temporary_hash != expected:
                raise InitializationArtifactError(
                    f"Temporary initialization copy SHA-256 mismatch: {temporary_hash}"
                )
            try:
                os.link(temporary_path, target)
                action = "copied"
            except FileExistsError:
                target_hash = sha256_file(target)
                if target_hash != expected:
                    raise InitializationArtifactError(
                        f"Refusing concurrent overwrite of different initialization PLY: {target}"
                    )
                action = "reused_concurrent_identical"
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    target_info = inspect_initialization_ply(target)
    source_hash_after = sha256_file(source)
    manifest_hash = sha256_file(paths.view_manifest)
    if source_hash_after != source_hash_before:
        raise InitializationArtifactError("Source initialization PLY changed while it was being staged")
    if target_info["sha256"] != source_hash_before:
        raise InitializationArtifactError("Copied initialization PLY differs from its source")
    if target_info["vertex_count"] != source_info["vertex_count"]:
        raise InitializationArtifactError("Copied initialization PLY vertex count changed")

    return {
        "mode": "external_frozen_ply",
        "status": "verified",
        "copy_action": action,
        "expected_sha256": expected,
        "source": source_info,
        "destination": target_info,
        "source_unchanged_after_copy": True,
        "byte_identical": True,
        "view_manifest_sha256": manifest_hash,
        "view_protocol": {
            "dataset": on_disk_manifest["dataset"],
            "scene": on_disk_manifest["scene"],
            "view_count": on_disk_manifest["view_count"],
            "selected_train_images": list(on_disk_manifest["selected_train_images"]),
            "pdc_initialization_images": list(on_disk_manifest["pdc_initialization_images"]),
        },
    }


def validate_frozen_initialization_for_training(paths: RunPaths, artifact: Mapping) -> None:
    if artifact.get("mode") != "external_frozen_ply" or artifact.get("status") != "verified":
        raise InitializationArtifactError("External initialization artifact is not verified")
    expected = normalize_sha256(str(artifact["expected_sha256"]))
    if sha256_file(paths.view_manifest) != artifact.get("view_manifest_sha256"):
        raise InitializationArtifactError(
            "View manifest changed after initialization staging; refusing to start training"
        )
    destination = Path(artifact["destination"]["path"])
    if not destination.is_file() or sha256_file(destination) != expected:
        raise InitializationArtifactError(
            "Frozen initialization destination is missing or its SHA-256 changed"
        )
    current_info = inspect_initialization_ply(destination)
    if current_info["vertex_count"] != artifact["destination"]["vertex_count"]:
        raise InitializationArtifactError("Frozen initialization vertex count changed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def profile_config(profile: str) -> DevelopmentConfig:
    if profile == "5k":
        return DevelopmentConfig(iterations=5000)
    if profile == "3k":
        return DevelopmentConfig(iterations=3000)
    raise ValueError(f"Unknown development profile: {profile}")


def validate_experiment_name(name: str) -> str:
    if not name or name in {".", ".."} or any(char in name for char in "/\\"):
        raise ValueError("experiment-name must be one safe directory name")
    return name


def make_paths(
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    experiment_name: str,
    config: DevelopmentConfig,
) -> RunPaths:
    name = validate_experiment_name(experiment_name)
    repo = repository_root.resolve()
    dataset = dataset_root if dataset_root.is_absolute() else repo / dataset_root
    output = output_root if output_root.is_absolute() else repo / output_root
    experiment = (output / name).resolve()
    return RunPaths(
        repository_root=repo,
        scene_path=(dataset / config.scene).resolve(),
        experiment_dir=experiment,
        initialization_dir=(repo / "keypoints_to_3d" / f"LLFF_{name}").resolve(),
        view_manifest=experiment / "view_manifest.json",
        run_manifest=experiment / "run_manifest.json",
    )


def get_git_sha(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot read Git SHA: {result.stderr.strip()}")
    return result.stdout.strip()


def generate_view_manifest(paths: RunPaths, config: DevelopmentConfig, git_sha: str) -> dict:
    records = read_and_validate_colmap_scene(paths.scene_path)
    manifest = build_view_manifest(
        dataset=config.dataset_name,
        scene=config.scene,
        image_names=[record.image_name for record in records],
        view_count=config.n_views,
        source_commit=git_sha,
    )
    write_view_manifest(paths.view_manifest, manifest)
    return manifest


def validate_existing_view_manifest(paths: RunPaths, config: DevelopmentConfig) -> dict:
    records = read_and_validate_colmap_scene(paths.scene_path)
    return load_view_manifest(
        paths.view_manifest,
        current_image_names=[record.image_name for record in records],
        expected_dataset=config.dataset_name,
        expected_scene=config.scene,
        expected_view_count=config.n_views,
    )


def build_stage_commands(
    paths: RunPaths, config: DevelopmentConfig, python_executable: str
) -> dict[str, list[str]]:
    repo = paths.repository_root
    suffix = paths.experiment_dir.name
    common_protocol = [
        "--n_views", str(config.n_views),
        "--resolution", str(config.resolution),
        "--dataset_name", config.dataset_name,
        "--view_manifest", str(paths.view_manifest),
    ]
    init = [
        python_executable,
        str(repo / "submodules" / "dense_matcher" / "triangulate.py"),
        "--data_path", str(paths.scene_path),
        "--output_path", str(paths.initialization_dir),
        *common_protocol,
    ]
    train = [
        python_executable,
        str(repo / "train.py"),
        "-s", str(paths.scene_path),
        "-m", str(paths.experiment_dir),
        "--iterations", str(config.iterations),
        "--dropout_factor", str(config.dropout_factor),
        "--sigma_noise", str(config.sigma_noise),
        "--suffix", suffix,
        *common_protocol,
    ]
    if config.eval:
        train.append("--eval")
    if config.densification_diagnostics:
        train.append("--densification_diagnostics")
    render = [
        python_executable,
        str(repo / "render.py"),
        "--model_path", str(paths.experiment_dir),
        "--n_views", str(config.n_views),
        "--resolution", str(config.resolution),
        "--dataset_name", config.dataset_name,
        "--dropout_factor", str(config.dropout_factor),
    ]
    if config.eval:
        render.append("--eval")
    metrics = [
        python_executable,
        str(repo / "metrics.py"),
        "--model_paths", str(paths.experiment_dir),
        "--dataset_name", config.dataset_name,
    ]
    return {"init": init, "train": train, "render": render, "metrics": metrics}


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_stage(
    *,
    stage: str,
    command: Sequence[str],
    log_path: Path,
    cwd: Path,
    environment: Mapping[str, str],
    output_stream: TextIO | None = None,
) -> int:
    """Run one stage while teeing its merged byte stream to the terminal and log.

    Reading fixed-size binary chunks is intentional: tqdm redraws with carriage
    returns that may not be followed by a newline, so line iteration can hide
    progress until much later.  An incremental decoder keeps split UTF-8 code
    points intact for the terminal while the log receives the original bytes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console = sys.stdout if output_stream is None else output_stream
    child_environment = dict(environment)
    child_environment["PYTHONUTF8"] = "1"
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("wb") as log:
        header = (
            f"stage: {stage}\n"
            f"started_utc: {utc_now()}\n"
            f"cwd: {cwd}\n"
            f"command: {command_text(command)}\n"
            "--- combined stdout/stderr ---\n"
        )
        log.write(header.encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
            raise RuntimeError("Subprocess stdout pipe was not created")
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                text = decoder.decode(chunk)
                if text:
                    console.write(text)
                    console.flush()
            trailing_text = decoder.decode(b"", final=True)
            if trailing_text:
                console.write(trailing_text)
                console.flush()
        finally:
            process.stdout.close()
        returncode = process.wait()
        footer = (
            f"\nfinished_utc: {utc_now()}\n"
            f"\n--- exit_code: {returncode} ---\n"
        )
        log.write(footer.encode("utf-8"))
        log.flush()
    if returncode != 0:
        raise StageFailure(stage, returncode)
    return returncode


def execute_stage_plan(
    *,
    selected_stages: Iterable[str],
    commands: Mapping[str, Sequence[str]],
    paths: RunPaths,
    environment: Mapping[str, str],
    run_record: dict,
) -> None:
    selected = set(selected_stages)
    for stage in STAGE_ORDER:
        if stage not in selected:
            continue
        entry = run_record["stages"][stage]
        entry["status"] = "running"
        entry["started_utc"] = utc_now()
        write_json(paths.run_manifest, run_record)
        try:
            if stage == "train" and run_record.get("initialization_artifact") is not None:
                validate_frozen_initialization_for_training(
                    paths, run_record["initialization_artifact"]
                )
            run_stage(
                stage=stage,
                command=commands[stage],
                log_path=paths.experiment_dir / f"{stage}.log",
                cwd=paths.repository_root,
                environment=environment,
            )
        except Exception as exc:
            entry["status"] = "failed"
            entry["returncode"] = exc.returncode if isinstance(exc, StageFailure) else None
            entry["error"] = str(exc)
            entry["finished_utc"] = utc_now()
            run_record["status"] = "failed"
            write_json(paths.run_manifest, run_record)
            raise
        entry["status"] = "succeeded"
        entry["returncode"] = 0
        entry["finished_utc"] = utc_now()
        write_json(paths.run_manifest, run_record)


def make_run_record(
    *,
    config: DevelopmentConfig,
    paths: RunPaths,
    git_sha: str,
    selected_stages: Sequence[str],
    commands: Mapping[str, Sequence[str]],
    external_initialization: bool = False,
) -> dict:
    record = {
        "manifest_version": (
            "llff-single-run-v2" if external_initialization else "llff-single-run-v1"
        ),
        "created_utc": utc_now(),
        "status": "prepared",
        "git_sha": git_sha,
        "expected_upstream_commit": SOURCE_COMMIT,
        "selected_stages": list(selected_stages),
        "config": asdict(config),
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "stages": {
            "manifest": {
                "status": "pending",
                "path": str(paths.view_manifest),
                "returncode": None,
            },
            **{
            stage: {
                "status": "pending" if stage in selected_stages else "not_selected",
                "command_argv": list(commands[stage]),
                "command_display": command_text(commands[stage]),
                "log": str(paths.experiment_dir / f"{stage}.log"),
                "returncode": None,
            }
            for stage in STAGE_ORDER
            },
        },
    }
    if external_initialization:
        record["initialization_artifact"] = None
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditable single-scene LLFF runner")
    parser.add_argument("--profile", choices=("5k", "3k"), default="5k")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--scene", default="horns")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--dropout-factor", type=float, default=0.2)
    parser.add_argument("--sigma-noise", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/nerf_llff_data_colmap"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--python", default=FSGS_PYTHON,
        help="absolute Python executable for GPU stages (defaults to the verified fsgs environment)",
    )
    parser.add_argument(
        "--initialization-ply", type=Path, default=None,
        help="external frozen PDC initialization PLY to verify and stage",
    )
    parser.add_argument(
        "--expected-initialization-sha256", default=None,
        help="required SHA-256 for --initialization-ply",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("manifest", *STAGE_ORDER),
        default=["manifest"],
        help="GPU stages are opt-in; the safe default only writes the view manifest",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--densification-diagnostics",
        action="store_true",
        default=False,
        help="pass --densification_diagnostics only to the train.py subprocess",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base = profile_config(args.profile)
    config = DevelopmentConfig(
        scene=args.scene,
        n_views=args.n_views,
        iterations=base.iterations if args.iterations is None else args.iterations,
        fallback_iterations=base.fallback_iterations,
        dropout_factor=args.dropout_factor,
        sigma_noise=args.sigma_noise,
        resolution=args.resolution,
        gpu=args.gpu,
        densification_diagnostics=args.densification_diagnostics,
    )
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    experiment_name = args.experiment_name or (
        f"llff_{config.scene}_{config.n_views}view_{config.iterations}iter_r{config.resolution}"
    )
    paths = make_paths(
        repository_root=REPOSITORY_ROOT,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        experiment_name=experiment_name,
        config=config,
    )
    paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    git_sha = get_git_sha(paths.repository_root)
    selected_stages = list(dict.fromkeys(args.stages))
    has_initialization = args.initialization_ply is not None
    has_initialization_hash = args.expected_initialization_sha256 is not None
    if has_initialization != has_initialization_hash:
        raise InitializationArtifactError(
            "--initialization-ply and --expected-initialization-sha256 must be supplied together"
        )
    if has_initialization and "init" in selected_stages:
        raise InitializationArtifactError(
            "External frozen initialization and the PDC init stage are mutually exclusive"
        )

    commands = build_stage_commands(paths, config, args.python)
    run_record = make_run_record(
        config=config,
        paths=paths,
        git_sha=git_sha,
        selected_stages=selected_stages,
        commands=commands,
        external_initialization=has_initialization,
    )
    write_json(paths.run_manifest, run_record)

    try:
        if "manifest" in selected_stages:
            view_manifest = generate_view_manifest(paths, config, git_sha)
            run_record["stages"]["manifest"]["status"] = "succeeded"
        else:
            view_manifest = validate_existing_view_manifest(paths, config)
            run_record["stages"]["manifest"]["status"] = "validated_existing"
        run_record["stages"]["manifest"]["returncode"] = 0
    except Exception as exc:
        run_record["stages"]["manifest"]["status"] = "failed"
        run_record["stages"]["manifest"]["error"] = str(exc)
        run_record["status"] = "failed"
        write_json(paths.run_manifest, run_record)
        raise
    write_json(paths.run_manifest, run_record)

    if has_initialization:
        try:
            run_record["initialization_artifact"] = stage_frozen_initialization(
                source_path=args.initialization_ply,
                expected_sha256=args.expected_initialization_sha256,
                paths=paths,
                config=config,
                view_manifest=view_manifest,
            )
        except Exception as exc:
            run_record["initialization_artifact"] = {
                "mode": "external_frozen_ply",
                "status": "failed",
                "source_path": str(args.initialization_ply.resolve()),
                "expected_sha256": args.expected_initialization_sha256,
                "error": str(exc),
            }
            run_record["status"] = "failed"
            write_json(paths.run_manifest, run_record)
            raise
        write_json(paths.run_manifest, run_record)

    print(f"View manifest: {paths.view_manifest}")
    for stage in STAGE_ORDER:
        if stage in selected_stages:
            print(f"[{stage}] {command_text(commands[stage])}")
    if args.dry_run:
        run_record["status"] = "dry_run"
        write_json(paths.run_manifest, run_record)
        return 0

    try:
        execute_stage_plan(
            selected_stages=selected_stages,
            commands=commands,
            paths=paths,
            environment={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(config.gpu),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            run_record=run_record,
        )
    except StageFailure as exc:
        print(str(exc), file=sys.stderr)
        return exc.returncode or 1
    run_record["status"] = "succeeded"
    run_record["finished_utc"] = utc_now()
    write_json(paths.run_manifest, run_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
