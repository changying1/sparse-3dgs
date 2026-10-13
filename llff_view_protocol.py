"""Auditable, CUDA-free LLFF view selection and manifest utilities.

This module intentionally depends only on the Python standard library so the
split protocol can be tested before PyTorch, the rasterizer, or PDCNet+ exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Iterable, Mapping, Sequence


MANIFEST_VERSION = "llff-view-manifest-v1"
LLFF_HOLD = 8


class ViewProtocolError(ValueError):
    """Raised when a view split or COLMAP/image mapping is unsafe."""


@dataclass(frozen=True)
class ColmapImageRecord:
    image_id: int
    camera_id: int
    image_name: str


def canonical_image_name(name: str) -> str:
    """Return a stable relative POSIX image name and reject unsafe paths."""
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ViewProtocolError(f"Unsafe COLMAP image name: {name!r}")
    return path.as_posix()


def sort_image_names(image_names: Iterable[str]) -> list[str]:
    names = [canonical_image_name(name) for name in image_names]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ViewProtocolError(f"Duplicate image names: {duplicates}")
    if not names:
        raise ViewProtocolError("No COLMAP images were found")
    return sorted(names)


def uniform_positions(item_count: int, selected_count: int) -> list[int]:
    """Match the historical round(linspace(0, M-1, n)) protocol without NumPy."""
    if selected_count <= 0:
        raise ViewProtocolError("view_count must be positive")
    if item_count < selected_count:
        raise ViewProtocolError(
            f"Cannot select {selected_count} views from {item_count} candidates"
        )
    if selected_count == 1:
        return [0]
    positions = [
        round(index * (item_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    ]
    if len(set(positions)) != selected_count:
        raise ViewProtocolError(f"Uniform selection produced duplicate positions: {positions}")
    return positions


def split_llff_images(
    image_names: Iterable[str], view_count: int = 3, hold: int = LLFF_HOLD
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return sorted-all, test, train-candidates, and selected-train filenames."""
    if hold <= 0:
        raise ViewProtocolError("LLFF hold interval must be positive")
    sorted_all = sort_image_names(image_names)
    test_images = [name for index, name in enumerate(sorted_all) if index % hold == 0]
    train_candidates = [
        name for index, name in enumerate(sorted_all) if index % hold != 0
    ]
    positions = uniform_positions(len(train_candidates), view_count)
    selected_train = [train_candidates[index] for index in positions]
    if set(test_images) & set(selected_train):
        raise ViewProtocolError("LLFF selected training images overlap the test split")
    return sorted_all, test_images, train_candidates, selected_train


def build_view_manifest(
    *,
    dataset: str,
    scene: str,
    image_names: Iterable[str],
    view_count: int,
    source_commit: str,
    pdc_initialization_images: Sequence[str] | None = None,
) -> dict:
    if dataset.upper() != "LLFF":
        raise ViewProtocolError(f"This manifest protocol only supports LLFF, got {dataset!r}")
    sorted_all, test_images, train_candidates, selected_train = split_llff_images(
        image_names, view_count=view_count
    )
    pdc_images = (
        list(selected_train)
        if pdc_initialization_images is None
        else [canonical_image_name(name) for name in pdc_initialization_images]
    )
    manifest = {
        "dataset": "LLFF",
        "scene": scene,
        "sorted_all_images": sorted_all,
        "test_images": test_images,
        "train_candidate_images": train_candidates,
        "selected_train_images": selected_train,
        "pdc_initialization_images": pdc_images,
        "view_count": view_count,
        "split_rule": "sort(image_name); test=index%8==0; uniformly select n views from remaining candidates",
        "source_commit": source_commit,
        "manifest_version": MANIFEST_VERSION,
    }
    validate_view_manifest(manifest, current_image_names=sorted_all)
    return manifest


def validate_view_manifest(
    manifest: Mapping,
    *,
    current_image_names: Iterable[str] | None = None,
    expected_dataset: str | None = None,
    expected_scene: str | None = None,
    expected_view_count: int | None = None,
) -> dict:
    required = {
        "dataset",
        "scene",
        "sorted_all_images",
        "test_images",
        "train_candidate_images",
        "selected_train_images",
        "pdc_initialization_images",
        "view_count",
        "split_rule",
        "source_commit",
        "manifest_version",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ViewProtocolError(f"Manifest is missing fields: {missing}")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ViewProtocolError(
            f"Unsupported manifest version: {manifest['manifest_version']!r}"
        )

    dataset = str(manifest["dataset"])
    scene = str(manifest["scene"])
    view_count = int(manifest["view_count"])
    if expected_dataset is not None and dataset != expected_dataset:
        raise ViewProtocolError(f"Manifest dataset {dataset!r} != {expected_dataset!r}")
    if expected_scene is not None and scene != expected_scene:
        raise ViewProtocolError(f"Manifest scene {scene!r} != {expected_scene!r}")
    if expected_view_count is not None and view_count != expected_view_count:
        raise ViewProtocolError(
            f"Manifest view_count {view_count} != requested {expected_view_count}"
        )

    sorted_all = sort_image_names(manifest["sorted_all_images"])
    if list(manifest["sorted_all_images"]) != sorted_all:
        raise ViewProtocolError("sorted_all_images is not strictly image-name sorted")
    rebuilt = build_view_manifest_fields(sorted_all, view_count)
    for field in ("test_images", "train_candidate_images", "selected_train_images"):
        actual = [canonical_image_name(name) for name in manifest[field]]
        if actual != rebuilt[field]:
            raise ViewProtocolError(f"Manifest field {field} does not match the LLFF split rule")

    selected = [canonical_image_name(name) for name in manifest["selected_train_images"]]
    pdc = [canonical_image_name(name) for name in manifest["pdc_initialization_images"]]
    tests = [canonical_image_name(name) for name in manifest["test_images"]]
    if selected != pdc:
        raise ViewProtocolError(
            "selected_train_images must exactly equal pdc_initialization_images"
        )
    overlap = sorted(set(tests) & set(pdc))
    if overlap:
        raise ViewProtocolError(f"Test images leaked into PDC initialization: {overlap}")
    if current_image_names is not None:
        current = sort_image_names(current_image_names)
        if current != sorted_all:
            raise ViewProtocolError(
                "Manifest image set/order does not match the current COLMAP model"
            )
    return dict(manifest)


def build_view_manifest_fields(sorted_all: Sequence[str], view_count: int) -> dict:
    all_names, tests, candidates, selected = split_llff_images(
        sorted_all, view_count=view_count
    )
    return {
        "sorted_all_images": all_names,
        "test_images": tests,
        "train_candidate_images": candidates,
        "selected_train_images": selected,
    }


def load_view_manifest(path: str | Path, **validation_kwargs) -> dict:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewProtocolError(f"Cannot read view manifest {manifest_path}: {exc}") from exc
    return validate_view_manifest(manifest, **validation_kwargs)


def write_view_manifest(path: str | Path, manifest: Mapping) -> None:
    validated = validate_view_manifest(manifest)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def image_path(images_dir: str | Path, name: str) -> Path:
    canonical = canonical_image_name(name)
    return Path(images_dir).joinpath(*PurePosixPath(canonical).parts)


def validate_colmap_records(
    records: Iterable[ColmapImageRecord],
    camera_ids: Iterable[int],
    images_dir: str | Path,
) -> list[ColmapImageRecord]:
    records_list = list(records)
    names = sort_image_names(record.image_name for record in records_list)
    if len({record.image_id for record in records_list}) != len(records_list):
        raise ViewProtocolError("COLMAP image IDs are not unique")
    known_cameras = set(camera_ids)
    missing_cameras = sorted(
        {record.camera_id for record in records_list if record.camera_id not in known_cameras}
    )
    if missing_cameras:
        raise ViewProtocolError(
            f"COLMAP images reference missing camera IDs: {missing_cameras}"
        )
    missing_images = [
        name for name in names if not image_path(images_dir, name).is_file()
    ]
    if missing_images:
        raise ViewProtocolError(f"COLMAP image files are missing: {missing_images}")
    by_name = {canonical_image_name(record.image_name): record for record in records_list}
    return [by_name[name] for name in names]


def select_pdc_initialization_records(
    ordered_records: Sequence[ColmapImageRecord], manifest: Mapping
) -> list[ColmapImageRecord]:
    """Return only manifest-authorized PDC views, in manifest order."""
    ordered_names = [canonical_image_name(record.image_name) for record in ordered_records]
    validated = validate_view_manifest(manifest, current_image_names=ordered_names)
    by_name = {
        canonical_image_name(record.image_name): record for record in ordered_records
    }
    selected = [
        by_name[name] for name in validated["pdc_initialization_images"]
    ]
    if [record.image_name for record in selected] != validated["selected_train_images"]:
        raise ViewProtocolError("PDC record order does not match selected training images")
    return selected


def pdc_initialization_image_paths(
    selected_records: Sequence[ColmapImageRecord], images_dir: str | Path
) -> list[Path]:
    """Build the exact ordered image-read set for PDC initialization."""
    paths = [image_path(images_dir, record.image_name) for record in selected_records]
    if len(paths) != len(set(paths)):
        raise ViewProtocolError("PDC initialization image paths are not unique")
    return paths


def validate_shared_camera_intrinsics(
    intrinsics: Sequence[Sequence[Sequence[float]]],
    *,
    image_sizes: Sequence[tuple[int, int]] | None = None,
    labels: Sequence[str] | None = None,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
) -> list[list[float]]:
    """Validate the shared-intrinsic assumption used by dense point sampling.

    The triangulation pipeline passes one K/focal/center to both views and to
    later patch projection. Supporting heterogeneous K therefore requires a
    larger mathematical change; A1-Fix1 deliberately fails closed instead.
    """
    if not intrinsics:
        raise ViewProtocolError("No camera intrinsics were supplied")
    matrices = []
    for index, matrix in enumerate(intrinsics):
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ViewProtocolError(f"Intrinsic matrix {index} is not 3x3")
        matrices.append([[float(value) for value in row] for row in matrix])
    if labels is not None and len(labels) != len(matrices):
        raise ViewProtocolError("Intrinsic labels do not align with matrices")
    if image_sizes is not None:
        if len(image_sizes) != len(matrices):
            raise ViewProtocolError("Image sizes do not align with intrinsic matrices")
        reference_size = tuple(image_sizes[0])
        for index, size in enumerate(image_sizes[1:], start=1):
            if tuple(size) != reference_size:
                label = labels[index] if labels is not None else str(index)
                raise ViewProtocolError(
                    f"PDC views must share image dimensions; {label!r} has {tuple(size)}, "
                    f"expected {reference_size}"
                )
    reference = matrices[0]
    for index, matrix in enumerate(matrices[1:], start=1):
        for row in range(3):
            for column in range(3):
                if not math.isclose(
                    matrix[row][column],
                    reference[row][column],
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                ):
                    label = labels[index] if labels is not None else str(index)
                    raise ViewProtocolError(
                        f"PDC shared-intrinsic assumption violated by {label!r} at "
                        f"K[{row},{column}]: {matrix[row][column]} != "
                        f"{reference[row][column]}"
                    )
    return reference


def _read_exact(handle, size: int, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ViewProtocolError(f"Truncated COLMAP binary while reading {context}")
    return data


def _read_c_string(handle, context: str) -> str:
    data = bytearray()
    while True:
        value = _read_exact(handle, 1, context)
        if value == b"\x00":
            break
        data.extend(value)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ViewProtocolError(f"Invalid UTF-8 COLMAP image name in {context}") from exc


def read_colmap_images_binary(path: str | Path) -> list[ColmapImageRecord]:
    records = []
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8, "image count"))[0]
        for _ in range(count):
            values = struct.unpack("<idddddddi", _read_exact(handle, 64, "image record"))
            image_id, camera_id = values[0], values[-1]
            name = canonical_image_name(_read_c_string(handle, f"image {image_id} name"))
            points_count = struct.unpack(
                "<Q", _read_exact(handle, 8, f"image {image_id} point count")
            )[0]
            _read_exact(handle, points_count * 24, f"image {image_id} points")
            records.append(ColmapImageRecord(image_id, camera_id, name))
        if handle.read(1):
            raise ViewProtocolError(f"Unexpected trailing data in {path}")
    return records


def read_colmap_camera_ids_binary(path: str | Path) -> set[int]:
    num_params_by_model = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                           7: 5, 8: 4, 9: 5, 10: 12}
    camera_ids = set()
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8, "camera count"))[0]
        for _ in range(count):
            camera_id, model_id, _width, _height = struct.unpack(
                "<iiQQ", _read_exact(handle, 24, "camera record")
            )
            if model_id not in num_params_by_model:
                raise ViewProtocolError(f"Unknown COLMAP camera model ID: {model_id}")
            _read_exact(handle, num_params_by_model[model_id] * 8, "camera parameters")
            if camera_id in camera_ids:
                raise ViewProtocolError(f"Duplicate COLMAP camera ID: {camera_id}")
            camera_ids.add(camera_id)
        if handle.read(1):
            raise ViewProtocolError(f"Unexpected trailing data in {path}")
    return camera_ids


def read_colmap_images_text(path: str | Path) -> list[ColmapImageRecord]:
    physical_lines = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                continue
            physical_lines.append(line.rstrip("\r\n"))
    records = []
    index = 0
    while index < len(physical_lines):
        image_line = physical_lines[index].strip()
        if not image_line:
            index += 1
            continue
        fields = image_line.split(maxsplit=9)
        if len(fields) != 10:
            raise ViewProtocolError(f"Malformed COLMAP image line: {image_line!r}")
        records.append(
            ColmapImageRecord(int(fields[0]), int(fields[8]), canonical_image_name(fields[9]))
        )
        index += 1
        if index >= len(physical_lines):
            raise ViewProtocolError(f"Missing COLMAP points line after image {fields[0]}")
        index += 1
    return records


def read_colmap_camera_ids_text(path: str | Path) -> set[int]:
    camera_ids = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                raise ViewProtocolError(f"Malformed COLMAP camera line: {stripped!r}")
            camera_id = int(fields[0])
            if camera_id in camera_ids:
                raise ViewProtocolError(f"Duplicate COLMAP camera ID: {camera_id}")
            camera_ids.add(camera_id)
    return camera_ids


def read_and_validate_colmap_scene(
    scene_path: str | Path, images_folder: str = "images"
) -> list[ColmapImageRecord]:
    scene = Path(scene_path)
    sparse = scene / "sparse" / "0"
    images_bin, cameras_bin = sparse / "images.bin", sparse / "cameras.bin"
    images_txt, cameras_txt = sparse / "images.txt", sparse / "cameras.txt"
    if images_bin.is_file() and cameras_bin.is_file():
        records = read_colmap_images_binary(images_bin)
        camera_ids = read_colmap_camera_ids_binary(cameras_bin)
    elif images_txt.is_file() and cameras_txt.is_file():
        records = read_colmap_images_text(images_txt)
        camera_ids = read_colmap_camera_ids_text(cameras_txt)
    else:
        raise ViewProtocolError(
            "COLMAP model must contain a complete images/cameras .bin pair or .txt pair"
        )
    return validate_colmap_records(records, camera_ids, scene / images_folder)
