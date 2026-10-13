import importlib
import importlib.util
import hashlib
import argparse
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

try:
    import pytest
except ImportError:  # Keep the repository's standard-library unittest workflow usable.
    class _Raises:
        def __init__(self, exception, match=None):
            self.exception = exception
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, _tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exception.__name__}")
            if not issubclass(exc_type, self.exception):
                return False
            if self.match is not None and self.match not in str(exc):
                raise AssertionError(f"{self.match!r} not found in {str(exc)!r}")
            return True

    class _Mark:
        @staticmethod
        def parametrize(names, values):
            parameter_names = [name.strip() for name in names.split(",")]

            def decorate(function):
                def wrapped():
                    for value_set in values:
                        if len(parameter_names) == 1:
                            value_set = (value_set,)
                        function(*value_set)
                return wrapped
            return decorate

    class _PytestCompat:
        mark = _Mark()

        @staticmethod
        def skip(message):
            raise unittest.SkipTest(message)

        @staticmethod
        def raises(exception, match=None):
            return _Raises(exception, match=match)

    pytest = _PytestCompat()

from utils.evidence_structural_loss import (
    OE_STRUCTURE_GATE_MODES,
    add_oe_structural_arguments,
    compute_oe_structural_loss,
    compute_stable_mask_from_gt_rgb,
    should_apply_oe_structural_loss,
)
from utils.evidence_support import compute_real_depth_evidence_maps, select_fsgs_reference_depth
from utils.oe_depth_cache import (
    CACHE_MANIFEST_NAME,
    CACHE_VERSION,
    MIDAS_MODEL_TYPE,
    MIDAS_PREPROCESSING,
    OEDepthCache,
    OEDepthCacheError,
    create_oe_depth_cache_if_enabled,
    validate_rendered_depth,
)
from utils.oe_training import (
    I1Diagnostics,
    apply_oe_structural_training_loss,
    validate_oe_training_configuration,
)
from llff_view_protocol import build_view_manifest, write_view_manifest

ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "output" / "A3_A3_Horns_3v_Baseline_5k_FrozenPDC" / "view_manifest.json"
FSGS_STRUCTURAL = ROOT.parent / "FSGS" / "utils" / "evidence_structural_loss.py"
FSGS_SUPPORT = ROOT.parent / "FSGS" / "utils" / "evidence_support.py"


def _inputs():
    rendered = torch.tensor(
        [[0.4, 0.7, 1.0, 1.2], [0.6, 0.9, 1.4, 1.8], [0.8, 1.1, 1.7, 2.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference = torch.tensor(
        [[202.0, 201.0, 199.0, 197.0], [203.0, 200.0, 198.0, 196.0], [205.0, 202.0, 197.0, 194.0]],
        dtype=torch.float32,
    )
    rgb = torch.tensor(
        [
            [[0.1, 0.1, 0.2, 0.9], [0.1, 0.2, 0.3, 0.8], [0.2, 0.2, 0.4, 0.7]],
            [[0.2, 0.2, 0.3, 0.8], [0.2, 0.3, 0.4, 0.7], [0.3, 0.3, 0.5, 0.6]],
            [[0.3, 0.3, 0.4, 0.7], [0.3, 0.4, 0.5, 0.6], [0.4, 0.4, 0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    return rendered, reference, rgb


def _load_fsgs_reference_module():
    if not FSGS_STRUCTURAL.is_file():
        pytest.skip(f"FSGS reference checkout unavailable: {FSGS_STRUCTURAL}")
    spec = importlib.util.spec_from_file_location("_fsgs_reference_structural", FSGS_STRUCTURAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_ast(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    assert set(found) == set(names)
    return found


def test_migrated_math_function_asts_match_fsgs_source_exactly():
    if not FSGS_STRUCTURAL.is_file() or not FSGS_SUPPORT.is_file():
        pytest.skip("FSGS reference checkout unavailable")
    support_functions = (
        "_as_depth_map",
        "_safe_float",
        "_pearson_corr",
        "_reference_candidates",
        "select_fsgs_reference_depth",
        "_centered_rms_normalize",
        "compute_real_depth_evidence_maps",
    )
    structural_functions = (
        "should_apply_oe_structural_loss",
        "compute_stable_mask_from_gt_rgb",
        "compute_oe_structural_loss",
        "_weighted_loss",
        "_pair_terms",
        "_gate_weights",
        "_evidence_distribution_stats",
        "_empty_stats",
        "_fraction",
        "_safe_float",
    )
    assert _function_ast(ROOT / "utils" / "evidence_support.py", support_functions) == _function_ast(
        FSGS_SUPPORT, support_functions
    )
    assert _function_ast(ROOT / "utils" / "evidence_structural_loss.py", structural_functions) == _function_ast(
        FSGS_STRUCTURAL, structural_functions
    )


def test_disabled_path_does_not_construct_cache_or_load_midas():
    calls = []

    def forbidden_factory(*args):
        calls.append(args)
        raise AssertionError("disabled path touched the cache")

    assert create_oe_depth_cache_if_enabled(
        False, "missing", "missing", [], cache_factory=forbidden_factory
    ) is None
    assert calls == []
    with mock.patch.object(torch.hub, "load", side_effect=AssertionError("MiDaS initialized")):
        module = importlib.import_module("script.generate_oe_depth_cache")
        importlib.reload(module)


def test_cli_defaults_are_inert_and_gate_modes_are_complete():
    parser = argparse.ArgumentParser()
    add_oe_structural_arguments(parser)
    args = parser.parse_args([])
    assert args.enable_oe_structural_loss is False
    assert args.oe_structural_weight == 0.0
    assert args.oe_structural_start_iter == 0
    assert args.oe_structural_end_iter == 30_000
    assert args.oe_stable_quantile == 0.80
    assert args.oe_structure_gate_mode == "stable_evidence"
    assert args.oe_depth_cache_dir == ""
    assert OE_STRUCTURE_GATE_MODES == (
        "none", "stable_only", "evidence_only", "stable_evidence", "matched_stable"
    )


@pytest.mark.parametrize("gate_mode", OE_STRUCTURE_GATE_MODES)
def test_migrated_loss_matches_fsgs_reference_for_all_gate_modes(gate_mode):
    reference_module = _load_fsgs_reference_module()
    rendered, reference, rgb = _inputs()
    migrated, migrated_stats, _ = compute_oe_structural_loss(
        rendered, reference, gt_rgb=rgb, gate_mode=gate_mode, stable_quantile=0.65
    )
    original, original_stats, _ = reference_module.compute_oe_structural_loss(
        rendered.detach().clone().requires_grad_(True),
        reference,
        gt_rgb=rgb,
        gate_mode=gate_mode,
        stable_quantile=0.65,
    )
    torch.testing.assert_close(migrated.detach(), original.detach(), rtol=0, atol=0)
    for key in (
        "raw_struct_loss",
        "gated_struct_loss",
        "stable_only_gated_loss",
        "stable_evidence_gated_loss",
        "active_pair_fraction",
        "valid_pair_count",
        "active_pair_count",
        "reference_transform",
    ):
        assert migrated_stats[key] == original_stats[key]


def test_stable_region_matches_fsgs_and_manual_gradient_rule():
    reference_module = _load_fsgs_reference_module()
    _, _, rgb = _inputs()
    stable, threshold, magnitude = compute_stable_mask_from_gt_rgb(rgb, stable_quantile=0.65)
    old_stable, old_threshold, old_magnitude = reference_module.compute_stable_mask_from_gt_rgb(
        rgb, stable_quantile=0.65
    )
    assert torch.equal(stable, old_stable)
    assert threshold == old_threshold
    torch.testing.assert_close(magnitude, old_magnitude, equal_nan=True)
    gray = (rgb * torch.tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    dx, dy = torch.zeros_like(gray), torch.zeros_like(gray)
    dx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    dy[:-1, :] = gray[1:, :] - gray[:-1, :]
    manual_mag = torch.sqrt(dx.square() + dy.square())
    manual_threshold = torch.quantile(manual_mag, 0.65)
    assert torch.equal(stable, manual_mag < manual_threshold)


@pytest.mark.parametrize(
    "iteration, expected",
    [(9, False), (10, True), (20, True), (21, False)],
)
def test_activation_interval_is_inclusive(iteration, expected):
    assert should_apply_oe_structural_loss(True, iteration, 10, 20, 0.1) is expected
    assert not should_apply_oe_structural_loss(False, iteration, 10, 20, 0.1)
    assert not should_apply_oe_structural_loss(True, iteration, 10, 20, 0.0)


def test_reference_selection_and_evidence_detach_semantics():
    rendered, reference, _ = _inputs()
    _, transform, corr, candidates, valid = select_fsgs_reference_depth(rendered, reference)
    assert transform in ("negative_midas", "inverse_midas_shift_200")
    assert corr == candidates[transform]
    assert valid.all()
    maps = compute_real_depth_evidence_maps(rendered, reference, detach_rendered=False)
    assert maps["normalization"] == "centered_rms"
    assert maps["z_render"].requires_grad
    assert not maps["z_reference"].requires_grad
    assert not maps["evidence"].requires_grad
    assert not maps["valid_mask"].requires_grad


def test_loss_keeps_rendered_depth_gradient_and_detaches_gates():
    rendered, reference, rgb = _inputs()
    stable, _, _ = compute_stable_mask_from_gt_rgb(rgb)
    loss, _, maps = compute_oe_structural_loss(
        rendered, reference, stable_mask=stable, gate_mode="stable_evidence"
    )
    assert loss.requires_grad
    loss.backward()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad.abs().sum() > 0
    assert not stable.requires_grad
    assert not maps["evidence"].requires_grad
    assert not maps["z_reference"].requires_grad


def _make_cache(tmp_path, *, shape=(2, 3), mutation=None):
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["selected_train_images"]
    cameras = [
        SimpleNamespace(image_name=Path(name).stem, image_height=shape[0], image_width=shape[1])
        for name in selected
    ]
    images_dir = tmp_path / "scene" / "images"
    images_dir.mkdir(parents=True)
    records = []
    for index, name in enumerate(selected):
        source_path = images_dir / name
        source_path.write_bytes(f"source-{index}".encode("ascii"))
        filename = f"depth_{index}.npy"
        np.save(tmp_path / filename, np.full(shape, index + 1, dtype=np.float32), allow_pickle=False)
        records.append({
            "image_name": name,
            "height": shape[0],
            "width": shape[1],
            "depth_file": filename,
            "source_image": name,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "depth_sha256": hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest(),
        })
    manifest_hash = hashlib.sha256(FROZEN_MANIFEST.read_bytes()).hexdigest()
    cache_manifest = {
        "cache_version": CACHE_VERSION,
        "model_type": MIDAS_MODEL_TYPE,
        "preprocessing": MIDAS_PREPROCESSING,
        "selected_train_images": selected,
        "source_view_manifest_sha256": manifest_hash,
        "source_images_dir": str(images_dir.resolve()),
        "records": records,
    }
    if mutation:
        mutation(cache_manifest, cameras)
    (tmp_path / CACHE_MANIFEST_NAME).write_text(json.dumps(cache_manifest), encoding="utf-8")
    return cameras, images_dir


def test_cache_loads_only_manifest_selected_training_views(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    cache = OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)
    depth = cache.load_for_camera(cameras[0], device="cpu")
    assert depth.shape == (2, 3)
    assert not depth.requires_grad
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    test_camera = SimpleNamespace(image_name=Path(frozen["test_images"][0]).stem, image_height=2, image_width=3)
    with pytest.raises(OEDepthCacheError, match="not an authorized frozen training view"):
        cache.load_for_camera(test_camera, device="cpu")


def test_missing_cache_is_an_explicit_error(tmp_path):
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    cameras = [SimpleNamespace(image_name=Path(name).stem, image_height=2, image_width=3)
               for name in manifest["selected_train_images"]]
    with pytest.raises(OEDepthCacheError, match="Cannot read I1 cache manifest"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, tmp_path / "images")


def test_cache_size_mismatch_is_an_explicit_error(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    cameras[0].image_height = 4
    with pytest.raises(OEDepthCacheError, match="depth size mismatch"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_nonfinite_cache_is_an_explicit_error(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    np.save(tmp_path / "depth_1.npy", np.array([[1.0, np.nan, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32))
    cache_path = tmp_path / CACHE_MANIFEST_NAME
    payload = json.loads(cache_path.read_text())
    payload["records"][1]["depth_sha256"] = hashlib.sha256((tmp_path / "depth_1.npy").read_bytes()).hexdigest()
    cache_path.write_text(json.dumps(payload))
    with pytest.raises(OEDepthCacheError, match="non-finite"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_cache_rejects_test_view_record(tmp_path):
    def leak_test_view(cache_manifest, _cameras):
        manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        cache_manifest["records"][0]["image_name"] = manifest["test_images"][0]

    cameras, images_dir = _make_cache(tmp_path, mutation=leak_test_view)
    with pytest.raises(OEDepthCacheError, match="view mismatch"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_cache_rejects_wrong_source_image_hash(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    (images_dir / json.loads(FROZEN_MANIFEST.read_text())["selected_train_images"][0]).write_bytes(b"changed")
    with pytest.raises(OEDepthCacheError, match="source image hash mismatch"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_cache_rejects_wrong_depth_content_hash(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    with (tmp_path / "depth_0.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(OEDepthCacheError, match="depth content hash mismatch"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_cache_rejects_depth_modified_after_initial_validation(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    cache = OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)
    with (tmp_path / "depth_0.npy").open("ab") as handle:
        handle.write(b"changed-after-validation")
    with pytest.raises(OEDepthCacheError, match="changed after cache validation"):
        cache.load_for_camera(cameras[0], device="cpu")


def test_cache_rejects_wrong_scene_path_and_old_v1(tmp_path):
    cameras, images_dir = _make_cache(tmp_path)
    other = tmp_path / "other_scene" / "images"
    other.mkdir(parents=True)
    with pytest.raises(OEDepthCacheError, match="different scene images directory"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, other)
    cache_path = tmp_path / CACHE_MANIFEST_NAME
    payload = json.loads(cache_path.read_text())
    payload["cache_version"] = "oe-depth-cache-v1"
    cache_path.write_text(json.dumps(payload))
    with pytest.raises(OEDepthCacheError, match="Unsupported I1 cache version"):
        OEDepthCache(tmp_path, FROZEN_MANIFEST, cameras, images_dir)


def test_rendered_depth_validation_requires_shape_finite_and_gradient():
    base = torch.arange(6, dtype=torch.float32, requires_grad=True)
    depth = (base * 2).reshape(1, 2, 3)
    validated = validate_rendered_depth(depth, 2, 3)
    assert validated.shape == (2, 3) and validated.requires_grad
    with pytest.raises(ValueError, match="shape"):
        validate_rendered_depth(depth, 3, 2)
    with pytest.raises(RuntimeError, match="not differentiable"):
        validate_rendered_depth(torch.ones(2, 3), 2, 3)
    bad = (base * 2).reshape(2, 3).clone()
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_rendered_depth(bad, 2, 3)


def test_generator_uses_fsgs_normalization_and_resize_semantics():
    from script.generate_oe_depth_cache import estimate_depth_fsgs

    class DummyMidas:
        def __init__(self):
            self.seen = None

        def __call__(self, tensor):
            self.seen = tensor.detach().clone()
            return tensor.mean(dim=1)

    model = DummyMidas()
    rgb = torch.linspace(0, 1, 3 * 5 * 7).reshape(3, 5, 7)
    output = estimate_depth_fsgs(model, rgb, (5, 7))
    assert model.seen.shape == (1, 3, 384, 512)
    expected = torch.nn.functional.interpolate(
        ((rgb[None] - 0.5) / 0.5), size=(384, 512), mode="bicubic", align_corners=False
    )
    torch.testing.assert_close(model.seen, expected)
    assert output.shape == (5, 7)


def _make_generator_fixture(tmp_path):
    names = [f"image_{index:02d}.jpg" for index in range(10)]
    manifest = build_view_manifest(
        dataset="LLFF", scene="fixture", image_names=names, view_count=2, source_commit="fixture"
    )
    manifest_path = tmp_path / "view_manifest.json"
    write_view_manifest(manifest_path, manifest)
    images_dir = tmp_path / "fixture" / "images"
    images_dir.mkdir(parents=True)
    for index, name in enumerate(names):
        Image.new("RGB", (6, 4), color=(index, index, index)).save(images_dir / name)
    return manifest_path, images_dir, tmp_path / "cache", manifest


def test_generator_refuses_any_existing_output_before_midas(tmp_path):
    from script.generate_oe_depth_cache import generate

    for index, existing_name in enumerate(("existing.npy", CACHE_MANIFEST_NAME, "experiment.log")):
        case_root = tmp_path / f"case_{index}"
        manifest_path, images_dir, output_dir, _ = _make_generator_fixture(case_root)
        output_dir.mkdir()
        existing = output_dir / existing_name
        existing.write_text("preserve")
        args = SimpleNamespace(
            view_manifest=str(manifest_path), images_dir=str(images_dir), output_dir=str(output_dir),
            resolution=-1, device="cpu",
        )
        with mock.patch.object(torch.hub, "load", side_effect=AssertionError("MiDaS must not initialize")):
            with pytest.raises(FileExistsError, match="non-empty cache directory"):
                generate(args)
        assert existing.read_text() == "preserve"


def test_generator_success_publishes_complete_v2_cache(tmp_path):
    from script.generate_oe_depth_cache import generate

    manifest_path, images_dir, output_dir, manifest = _make_generator_fixture(tmp_path)

    class DummyMidas:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

        def __call__(self, tensor):
            return tensor.mean(dim=1)

    args = SimpleNamespace(
        view_manifest=str(manifest_path), images_dir=str(images_dir), output_dir=str(output_dir),
        resolution=-1, device="cpu",
    )
    with mock.patch.object(
        torch.hub, "load", side_effect=[DummyMidas(), SimpleNamespace(dpt_transform=object())]
    ):
        generate(args)
    assert (output_dir / CACHE_MANIFEST_NAME).is_file()
    assert not (output_dir / f".{CACHE_MANIFEST_NAME}.incomplete").exists()
    cameras = [
        SimpleNamespace(image_name=Path(name).stem, image_height=4, image_width=6)
        for name in manifest["selected_train_images"]
    ]
    cache = OEDepthCache(output_dir, manifest_path, cameras, images_dir)
    assert cache.load_for_camera(cameras[0], device="cpu").shape == (4, 6)


def test_generator_partial_failure_never_writes_success_manifest(tmp_path):
    from script.generate_oe_depth_cache import generate

    manifest_path, images_dir, output_dir, _ = _make_generator_fixture(tmp_path)

    class FailingMidas:
        def __init__(self):
            self.calls = 0

        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

        def __call__(self, tensor):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic inference failure")
            return tensor.mean(dim=1)

    model = FailingMidas()
    transforms = SimpleNamespace(dpt_transform=object())
    args = SimpleNamespace(
        view_manifest=str(manifest_path), images_dir=str(images_dir), output_dir=str(output_dir),
        resolution=-1, device="cpu",
    )
    with mock.patch.object(torch.hub, "load", side_effect=[model, transforms]):
        with pytest.raises(RuntimeError, match="synthetic inference failure"):
            generate(args)
    assert not (output_dir / CACHE_MANIFEST_NAME).exists()
    assert len(list(output_dir.glob("*.npy"))) == 1


def test_training_helper_disabled_and_outside_window_are_baseline_identity():
    baseline = torch.tensor(3.0, requires_grad=True)

    class ForbiddenCache:
        def load_for_camera(self, *_args, **_kwargs):
            raise AssertionError("inactive I1 accessed cache")

    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("computed I1"))
    disabled, loss, stats = apply_oe_structural_training_loss(
        baseline, enabled=False, iteration=5, start_iter=1, end_iter=10, weight=1.0,
        depth_cache=ForbiddenCache(), compute_fn=forbidden,
    )
    outside, _, _ = apply_oe_structural_training_loss(
        baseline, enabled=True, iteration=11, start_iter=1, end_iter=10, weight=1.0,
        depth_cache=ForbiddenCache(), compute_fn=forbidden,
    )
    assert disabled is baseline and outside is baseline and loss is None and stats is None


def test_training_helper_active_calls_cache_math_and_propagates_gradient():
    rendered_leaf, reference, rgb = _inputs()
    rendered = rendered_leaf * 1.0  # renderer-like non-leaf tensor with grad_fn
    camera = SimpleNamespace(image_name="train", image_height=3, image_width=4)

    class Cache:
        calls = 0

        def load_for_camera(self, seen_camera, *, device, dtype):
            self.calls += 1
            assert seen_camera is camera
            return reference.to(device=device, dtype=dtype)

    cache = Cache()
    baseline = rendered.sum() * 0.0 + 2.0
    total, structural, stats = apply_oe_structural_training_loss(
        baseline, enabled=True, iteration=5, start_iter=5, end_iter=5, weight=0.25,
        depth_cache=cache, camera=camera, rendered_depth=rendered, gt_rgb=rgb,
        gate_mode="none",
    )
    assert cache.calls == 1
    assert structural.item() > 0
    assert stats["valid_pair_count"] == 17
    torch.testing.assert_close(total, baseline + 0.25 * structural)
    total.backward()
    assert rendered_leaf.grad is not None and rendered_leaf.grad.abs().sum() > 0


def test_training_helper_cache_error_stops_immediately():
    class BadCache:
        def load_for_camera(self, *_args, **_kwargs):
            raise OEDepthCacheError("bad cache")

    base = torch.ones(2, 3, requires_grad=True)
    rendered = base * 2
    camera = SimpleNamespace(image_height=2, image_width=3)
    with pytest.raises(OEDepthCacheError, match="bad cache"):
        apply_oe_structural_training_loss(
            rendered.sum(), enabled=True, iteration=1, start_iter=1, end_iter=1, weight=1,
            depth_cache=BadCache(), camera=camera, rendered_depth=rendered,
        )


def test_i1_diagnostics_match_actual_calculation_and_summary():
    logs = []
    diagnostics = I1Diagnostics({"weight": 0.5}, (2, 3), interval=100, emit=logs.append)
    stats = {
        "valid_pair_count": 7, "active_pair_count": 4, "mean_pair_evidence": 0.25,
        "reference_transform": "negative_midas",
    }
    diagnostics.record(2, torch.tensor(1.5), torch.tensor(0.75), stats)
    summary = diagnostics.finish()
    activation = json.loads(logs[1].removeprefix("[I1] "))
    assert activation["oe_struct_loss"] == 1.5
    assert activation["weighted_oe_loss"] == 0.75
    assert activation["valid_pair_count"] == 7
    assert activation["active_pair_count"] == 4
    assert activation["mean_pair_evidence"] == 0.25
    assert activation["reference_transform"] == "negative_midas"
    assert summary == {
        "event": "summary", "actual_activation_count": 1, "total_valid_pair_count": 7,
        "total_active_pair_count": 4, "nonfinite_loss_seen": False,
    }


def test_nonfinite_i1_loss_is_logged_and_stops_training_path():
    logs = []
    diagnostics = I1Diagnostics({"enabled": True}, (1, 1), emit=logs.append)
    base_depth = torch.ones(2, 3, requires_grad=True)
    rendered = base_depth * 2
    camera = SimpleNamespace(image_height=2, image_width=3)

    class Cache:
        def load_for_camera(self, *_args, **_kwargs):
            return torch.ones(2, 3)

    stats = {
        "valid_pair_count": 7, "active_pair_count": 3, "mean_pair_evidence": 0.5,
        "reference_transform": "negative_midas",
    }

    def nonfinite(*_args, **_kwargs):
        return rendered.sum() * float("nan"), stats, {}

    with pytest.raises(FloatingPointError, match="non-finite"):
        apply_oe_structural_training_loss(
            rendered.sum(), enabled=True, iteration=1, start_iter=1, end_iter=1, weight=1,
            depth_cache=Cache(), camera=camera, rendered_depth=rendered,
            diagnostics=diagnostics, compute_fn=nonfinite,
        )
    event = json.loads(logs[-1].removeprefix("[I1] "))
    assert event["event"] == "nonfinite_loss"
    assert event["nonfinite_loss_seen"] is True
    assert event["actual_activation_count"] == 1
    assert event["total_valid_pair_count"] == 7


def test_invalid_i1_configuration_is_rejected_before_training():
    for values, message in (
        ((True, 0, 1, 2, 10), "weight"),
        ((True, 1, 5, 4, 10), "interval"),
        ((True, 1, 11, 20, 10), "cannot activate"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_oe_training_configuration(*values)
    assert validate_oe_training_configuration(False, 0, 99, 1, 0) is None


def test_train_integration_uses_primary_rendered_depth_and_helper():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert 'depth = render_pkg["rendered_depth"]' in source
    assert "baseline_total_loss = loss + disparity_loss + alpha_loss" in source
    assert "apply_oe_structural_training_loss(" in source
    assert "total_loss, oe_struct_loss, oe_stats" in source


class Innovation1Phase1Unittest(unittest.TestCase):
    """Adapters let ``python -m unittest`` run the same CPU checks without pytest."""

    def test_disabled_path(self):
        test_disabled_path_does_not_construct_cache_or_load_midas()

    def test_cli_defaults(self):
        test_cli_defaults_are_inert_and_gate_modes_are_complete()

    def test_all_gate_modes_match(self):
        for gate_mode in OE_STRUCTURE_GATE_MODES:
            with self.subTest(gate_mode=gate_mode):
                test_migrated_loss_matches_fsgs_reference_for_all_gate_modes(gate_mode)

    def test_exact_math_source_migration(self):
        test_migrated_math_function_asts_match_fsgs_source_exactly()

    def test_stable_region(self):
        test_stable_region_matches_fsgs_and_manual_gradient_rule()

    def test_activation_boundaries(self):
        for iteration, expected in ((9, False), (10, True), (20, True), (21, False)):
            with self.subTest(iteration=iteration):
                test_activation_interval_is_inclusive(iteration, expected)

    def test_evidence_detach(self):
        test_reference_selection_and_evidence_detach_semantics()

    def test_gradient(self):
        test_loss_keeps_rendered_depth_gradient_and_detaches_gates()

    def test_cache_success(self):
        with tempfile.TemporaryDirectory() as directory:
            test_cache_loads_only_manifest_selected_training_views(Path(directory))

    def test_cache_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            test_missing_cache_is_an_explicit_error(Path(directory))

    def test_cache_size(self):
        with tempfile.TemporaryDirectory() as directory:
            test_cache_size_mismatch_is_an_explicit_error(Path(directory))

    def test_cache_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            test_nonfinite_cache_is_an_explicit_error(Path(directory))

    def test_cache_test_view_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            test_cache_rejects_test_view_record(Path(directory))

    def test_rendered_depth_validation(self):
        test_rendered_depth_validation_requires_shape_finite_and_gradient()

    def test_generator_preprocessing(self):
        test_generator_uses_fsgs_normalization_and_resize_semantics()

    def test_train_integration(self):
        test_train_integration_uses_primary_rendered_depth_and_helper()


if __name__ == "__main__":
    unittest.main()
