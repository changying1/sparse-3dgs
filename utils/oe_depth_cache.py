"""Strict, inference-free loader for Innovation 1 monocular-depth caches."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from llff_view_protocol import canonical_image_name, load_view_manifest

CACHE_MANIFEST_NAME = "oe_depth_cache.json"
CACHE_VERSION = "oe-depth-cache-v2"
MIDAS_MODEL_TYPE = "DPT_Hybrid"
MIDAS_PREPROCESSING = {
    "input_rgb_range": "[0,1]",
    "normalization": "(img - 0.5) / 0.5",
    "network_size_hw": [384, 512],
    "resize_to_network": "bicubic_align_corners_false",
    "resize_to_training_image": "bicubic_align_corners_false",
    "official_dpt_transform_called": False,
    "source_image_resize": "PIL.Image.resize default (matches baseline PILtoTorch)",
}


class OEDepthCacheError(RuntimeError):
    pass


def create_oe_depth_cache_if_enabled(
    enabled,
    cache_dir,
    view_manifest_path,
    train_cameras,
    images_dir=None,
    cache_factory=None,
):
    """Keep the disabled baseline path free of all cache filesystem access."""
    if not bool(enabled):
        return None
    factory = OEDepthCache if cache_factory is None else cache_factory
    return factory(cache_dir, view_manifest_path, train_cameras, images_dir)


class OEDepthCache:
    """Validate the complete frozen training-view cache, then load by camera."""

    def __init__(self, cache_dir, view_manifest_path, train_cameras, images_dir):
        self.cache_dir = Path(cache_dir)
        self.view_manifest_path = Path(view_manifest_path)
        if not str(cache_dir):
            raise OEDepthCacheError("--oe_depth_cache_dir is required when I1 is enabled.")
        if not str(view_manifest_path):
            raise OEDepthCacheError("--view_manifest is required when I1 is enabled; views may not be resampled.")
        if images_dir is None or not str(images_dir):
            raise OEDepthCacheError("The current scene images directory is required when I1 is enabled.")
        self.images_dir = Path(images_dir).resolve()

        manifest = load_view_manifest(self.view_manifest_path)
        selected = [canonical_image_name(name) for name in manifest["selected_train_images"]]
        tests = {canonical_image_name(name) for name in manifest["test_images"]}
        if set(selected) & tests:
            raise OEDepthCacheError("View manifest leaks test views into selected training views.")

        cache_manifest_path = self.cache_dir / CACHE_MANIFEST_NAME
        try:
            with cache_manifest_path.open("r", encoding="utf-8") as handle:
                cache_manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise OEDepthCacheError(f"Cannot read I1 cache manifest {cache_manifest_path}: {exc}") from exc

        self._validate_header(cache_manifest, selected, self.view_manifest_path, self.images_dir)
        records = cache_manifest.get("records")
        if not isinstance(records, list):
            raise OEDepthCacheError("I1 cache manifest records must be a list.")
        by_name = {}
        for record in records:
            if not isinstance(record, dict):
                raise OEDepthCacheError(f"Malformed I1 cache record: {record!r}.")
            name = canonical_image_name(record.get("image_name", ""))
            if name in by_name:
                raise OEDepthCacheError(f"Duplicate I1 cache record for {name!r}.")
            by_name[name] = record
        if set(by_name) != set(selected):
            missing = sorted(set(selected) - set(by_name))
            extra = sorted(set(by_name) - set(selected))
            raise OEDepthCacheError(f"I1 cache view mismatch; missing={missing}, extra={extra}.")
        if set(by_name) & tests:
            raise OEDepthCacheError("I1 cache contains a test-view reference depth.")

        camera_by_stem = {}
        for camera in train_cameras:
            stem = str(camera.image_name)
            if stem in camera_by_stem:
                raise OEDepthCacheError(f"Training camera stem is ambiguous: {stem!r}.")
            camera_by_stem[stem] = camera
        selected_by_stem = {PurePosixPath(name).stem: name for name in selected}
        if len(selected_by_stem) != len(selected):
            raise OEDepthCacheError("Selected training image stems are not unique.")
        if set(camera_by_stem) != set(selected_by_stem):
            raise OEDepthCacheError(
                "Scene training cameras do not exactly match view_manifest selected_train_images."
            )

        self._records_by_stem = {}
        for stem, name in selected_by_stem.items():
            record = by_name[name]
            self._validate_record(record, camera_by_stem[stem], name)
            self._records_by_stem[stem] = record

    @staticmethod
    def _validate_header(cache_manifest, selected, view_manifest_path, images_dir):
        if cache_manifest.get("cache_version") != CACHE_VERSION:
            raise OEDepthCacheError(
                f"Unsupported I1 cache version: {cache_manifest.get('cache_version')!r}; "
                f"{CACHE_VERSION!r} is required. V1 is intentionally rejected because it lacks content hashes."
            )
        if cache_manifest.get("model_type") != MIDAS_MODEL_TYPE:
            raise OEDepthCacheError("I1 cache was not generated with MiDaS DPT_Hybrid.")
        if cache_manifest.get("preprocessing") != MIDAS_PREPROCESSING:
            raise OEDepthCacheError("I1 cache preprocessing does not match the frozen FSGS semantics.")
        cached_selected = [canonical_image_name(name) for name in cache_manifest.get("selected_train_images", [])]
        if cached_selected != selected:
            raise OEDepthCacheError("I1 cache selected_train_images/order differs from view_manifest.")
        expected_manifest_hash = _sha256_file(view_manifest_path)
        if cache_manifest.get("source_view_manifest_sha256") != expected_manifest_hash:
            raise OEDepthCacheError("I1 cache source view_manifest hash does not match the supplied manifest.")
        cached_images_dir = cache_manifest.get("source_images_dir")
        if not isinstance(cached_images_dir, str) or not cached_images_dir:
            raise OEDepthCacheError("I1 cache source_images_dir is missing.")
        if Path(cached_images_dir).resolve() != images_dir:
            raise OEDepthCacheError(
                "I1 cache belongs to a different scene images directory: "
                f"cache={Path(cached_images_dir).resolve()}, current={images_dir}."
            )

    def _validate_record(self, record, camera, expected_image_name):
        try:
            height = int(record["height"])
            width = int(record["width"])
            relative_file = record["depth_file"]
            source_image = record["source_image"]
            source_sha256 = record["source_sha256"]
            depth_sha256 = record["depth_sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise OEDepthCacheError(f"Malformed I1 cache record: {record!r}.") from exc
        try:
            source_image = canonical_image_name(source_image)
        except (TypeError, ValueError) as exc:
            raise OEDepthCacheError("I1 cache record source_image is invalid.") from exc
        if source_image != expected_image_name:
            raise OEDepthCacheError(
                f"I1 source image mismatch: cache={source_image!r}, expected={expected_image_name!r}."
            )
        if not _is_sha256(source_sha256):
            raise OEDepthCacheError("I1 cache record source_sha256 must be a SHA-256 hex digest.")
        if not _is_sha256(depth_sha256):
            raise OEDepthCacheError("I1 cache record depth_sha256 must be a SHA-256 hex digest.")
        source_path = self.images_dir.joinpath(*PurePosixPath(expected_image_name).parts)
        if not source_path.is_file():
            raise OEDepthCacheError(f"Missing I1 source image from the current scene: {source_path}.")
        if _sha256_file(source_path) != source_sha256:
            raise OEDepthCacheError(f"I1 source image hash mismatch: {source_path}.")
        if (height, width) != (int(camera.image_height), int(camera.image_width)):
            raise OEDepthCacheError(
                f"I1 depth size mismatch for {record.get('image_name')!r}: "
                f"cache={(height, width)}, training={(camera.image_height, camera.image_width)}."
            )
        if (
            not isinstance(relative_file, str)
            or not relative_file
            or "/" in relative_file
            or "\\" in relative_file
            or relative_file in (".", "..")
        ):
            raise OEDepthCacheError("I1 depth_file must be a plain filename inside the cache directory.")
        path = self.cache_dir / relative_file
        if not path.is_file():
            raise OEDepthCacheError(f"Missing I1 depth file: {path}.")
        if _sha256_file(path) != depth_sha256:
            raise OEDepthCacheError(f"I1 depth content hash mismatch: {path}.")
        try:
            depth = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise OEDepthCacheError(f"Cannot read I1 depth file {path}: {exc}") from exc
        if depth.shape != (height, width):
            raise OEDepthCacheError(f"I1 depth array shape mismatch in {path}: {depth.shape}.")
        if not np.issubdtype(depth.dtype, np.floating):
            raise OEDepthCacheError(f"I1 depth array must be floating point: {path}.")
        if not np.isfinite(depth).all():
            raise OEDepthCacheError(f"I1 depth array contains non-finite values: {path}.")

    def load_for_camera(self, camera, *, device, dtype=torch.float32):
        stem = str(camera.image_name)
        if stem not in self._records_by_stem:
            raise OEDepthCacheError(f"Camera {stem!r} is not an authorized frozen training view.")
        record = self._records_by_stem[stem]
        path = self.cache_dir / record["depth_file"]
        if _sha256_file(path) != record["depth_sha256"]:
            raise OEDepthCacheError(f"I1 depth content changed after cache validation: {path}.")
        depth = np.load(path, allow_pickle=False)
        if depth.shape != (int(camera.image_height), int(camera.image_width)):
            raise OEDepthCacheError(f"I1 depth size changed after cache validation: {path}.")
        if not np.isfinite(depth).all():
            raise OEDepthCacheError(f"I1 depth became non-finite after cache validation: {path}.")
        return torch.from_numpy(np.array(depth, dtype=np.float32, copy=True)).to(device=device, dtype=dtype).detach()


def validate_rendered_depth(rendered_depth, expected_height, expected_width):
    if not torch.is_tensor(rendered_depth):
        raise ValueError('render_pkg["rendered_depth"] must be a torch.Tensor.')
    depth = rendered_depth
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or tuple(depth.shape) != (int(expected_height), int(expected_width)):
        raise ValueError(
            f'render_pkg["rendered_depth"] has shape {tuple(rendered_depth.shape)}, '
            f"expected {(expected_height, expected_width)} (optionally singleton-channel)."
        )
    if not depth.requires_grad or depth.grad_fn is None:
        raise RuntimeError('render_pkg["rendered_depth"] is not differentiable.')
    if not torch.isfinite(depth).all():
        raise ValueError('render_pkg["rendered_depth"] contains non-finite values.')
    return depth


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()
