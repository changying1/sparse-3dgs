import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from script.smoke_i1_cuda import (
    SmokeCheckError,
    audit_dropout_indexing,
    evaluate_densification_protection_case,
    evaluate_render_case,
    gradient_stats,
    prepare_output_directory,
    synthetic_reference_depth,
)


class MockGaussians:
    def __init__(self, count):
        generator = torch.Generator().manual_seed(4)
        self._xyz = torch.nn.Parameter(torch.randn(count, 3, generator=generator))
        self._features_dc = torch.nn.Parameter(torch.randn(count, 1, 3, generator=generator))
        self._features_rest = torch.nn.Parameter(torch.randn(count, 3, 3, generator=generator))
        self._opacity = torch.nn.Parameter(torch.randn(count, 1, generator=generator))
        self._scaling = torch.nn.Parameter(torch.randn(count, 3, generator=generator))
        self._rotation = torch.nn.Parameter(torch.randn(count, 4, generator=generator))


def make_mock_graph(dropout_factor=0.2):
    height, width, count = 3, 4, 5
    gaussians = MockGaussians(count)
    dropout_mask = torch.tensor([True, True, False, True, False])
    if dropout_factor == 0.0:
        dropout_mask = torch.ones(count, dtype=torch.bool)
    kept_count = int(dropout_mask.sum())
    visibility = torch.ones(kept_count, dtype=torch.bool)
    if kept_count > 1:
        visibility[1] = False

    viewspace_leaf = torch.zeros(count, 3, requires_grad=True)
    viewspace = viewspace_leaf + 0.0
    viewspace.retain_grad()
    basis = torch.arange(1, count * height * width + 1, dtype=torch.float32)
    basis = basis.reshape(count, height, width) / float(count * height * width)
    active = dropout_mask.float()[:, None, None]
    depth = (
        ((gaussians._xyz[:, 0, None, None] + viewspace[:, 0, None, None]) * basis * active).sum(0)
        + gaussians._features_dc.sum() * 0.0001
        + gaussians._features_rest.sum() * 0.0001
        + gaussians._opacity.sum() * 0.0001
        + gaussians._scaling.sum() * 0.0001
        + gaussians._rotation.sum() * 0.0001
        + 0.2 * torch.sin(basis[0] * 5.0)
    ).unsqueeze(0)
    image_signal = (
        gaussians._xyz[dropout_mask].sum() * 0.002
        + gaussians._features_dc[dropout_mask].sum() * 0.0001
        + gaussians._features_rest[dropout_mask].sum() * 0.0001
        + gaussians._opacity[dropout_mask].sum() * 0.0001
        + gaussians._scaling[dropout_mask].sum() * 0.0001
        + gaussians._rotation[dropout_mask].sum() * 0.0001
        + viewspace[dropout_mask].sum() * 0.03
    )
    pixels = torch.linspace(0.1, 0.9, 3 * height * width).reshape(3, height, width)
    image = torch.sigmoid(pixels + image_signal)
    render_pkg = {
        "render": image,
        "rendered_depth": depth,
        "rendered_alpha": torch.sigmoid(depth),
        "viewspace_points": viewspace,
        "dropout_mask": dropout_mask,
        "radii": torch.ones(kept_count),
        "visibility_filter": visibility,
    }
    camera = SimpleNamespace(
        image_height=height,
        image_width=width,
        original_image=torch.zeros(3, height, width),
        gt_alpha_mask=None,
    )
    return render_pkg, gaussians, camera, viewspace_leaf


class I1Phase2ACudaSmokeCpuTests(unittest.TestCase):
    def test_report_directory_is_new_and_outside_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "baseline"
            frozen.mkdir()
            with self.assertRaises(SmokeCheckError):
                prepare_output_directory(frozen / "smoke", frozen)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_output_directory(existing, frozen)
            created = prepare_output_directory(root / "new_smoke", frozen)
            self.assertTrue(created.is_dir())

    def test_synthetic_reference_is_deterministic_finite_and_detached(self):
        first = synthetic_reference_depth(3, 4, device="cpu", dtype=torch.float32)
        second = synthetic_reference_depth(3, 4, device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 4))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(first.requires_grad)

    def test_dropout_subset_maps_exactly_back_to_full_indices(self):
        mask = torch.tensor([True, False, True, True, False])
        radii = torch.ones(3)
        visible = torch.tensor([True, False, True])
        viewspace = torch.zeros(5, 3, requires_grad=True)
        baseline_grad = torch.zeros_like(viewspace)
        combined_grad = torch.zeros_like(viewspace)
        baseline_grad[[0, 3], 0] = 1
        combined_grad[[0, 3], 0] = 2
        combined_mask, report = audit_dropout_indexing(
            mask,
            radii,
            visible,
            viewspace,
            dropout_factor=0.2,
            baseline_viewspace_grad=baseline_grad,
            combined_viewspace_grad=combined_grad,
        )
        self.assertEqual(torch.nonzero(combined_mask, as_tuple=True)[0].tolist(), [0, 3])
        self.assertEqual(report["visible_subset_count"], 2)
        self.assertTrue(report["mapping_is_exact"])

    def test_dropout_audit_rejects_subset_shape_and_gradient_leak(self):
        mask = torch.tensor([True, False, True])
        viewspace = torch.zeros(3, 3)
        with self.assertRaisesRegex(SmokeCheckError, "Raster subset mismatch"):
            audit_dropout_indexing(
                mask, torch.ones(3), torch.ones(3, dtype=torch.bool), viewspace,
                dropout_factor=0.2,
            )
        leaking = torch.zeros_like(viewspace)
        leaking[1, 0] = 1
        with self.assertRaisesRegex(SmokeCheckError, "leaked"):
            audit_dropout_indexing(
                mask, torch.ones(2), torch.ones(2, dtype=torch.bool), viewspace,
                dropout_factor=0.2, baseline_viewspace_grad=leaking,
            )

    def test_mock_render_graph_proves_gradient_and_index_checks(self):
        render_pkg, gaussians, camera, viewspace_leaf = make_mock_graph(0.2)
        result = evaluate_render_case(
            render_pkg,
            gaussians,
            camera,
            dropout_factor=0.2,
            lambda_dssim=0.2,
            structural_weight=0.1,
        )
        self.assertTrue(result["passed"], result["checks"])
        self.assertGreater(result["depth_structural_gradient"]["nonzero_count"], 0)
        self.assertGreater(result["gaussian_structural_gradients"]["xyz"]["nonzero_count"], 0)
        self.assertGreater(result["viewspace_gradient_delta"]["nonzero_count"], 0)
        self.assertIsNone(viewspace_leaf.grad)  # autograd.grad does not mutate .grad fields
        for parameter in (
            gaussians._xyz, gaussians._opacity, gaussians._scaling, gaussians._rotation
        ):
            self.assertIsNone(parameter.grad)

    def test_no_dropout_mock_keeps_every_gaussian(self):
        render_pkg, gaussians, camera, _ = make_mock_graph(0.0)
        result = evaluate_render_case(
            render_pkg,
            gaussians,
            camera,
            dropout_factor=0.0,
            lambda_dssim=0.2,
            structural_weight=0.1,
        )
        audit = result["dropout_index_audit"]
        self.assertEqual(audit["dropout_kept_count"], audit["gaussian_count"])
        self.assertEqual(audit["dropout_dropped_count"], 0)

    def test_phase2b_mock_exercises_real_protection_helpers(self):
        for dropout_factor in (0.0, 0.2):
            with self.subTest(dropout_factor=dropout_factor):
                render_pkg, gaussians, camera, _ = make_mock_graph(dropout_factor)
                result = evaluate_densification_protection_case(
                    render_pkg,
                    gaussians,
                    camera,
                    dropout_factor=dropout_factor,
                    lambda_dssim=0.2,
                    structural_weight=0.1,
                )
                self.assertTrue(result["passed"], result["checks"])
                self.assertGreaterEqual(
                    result["timing_ms"]["baseline_gradient_extraction"], 0.0
                )
                self.assertGreaterEqual(result["timing_ms"]["joint_backward"], 0.0)
                self.assertTrue(
                    result["proxy_audit"]["received_gradient_equals_baseline"]
                )
                self.assertFalse(
                    result["proxy_audit"]["received_gradient_equals_joint"]
                )
                self.assertTrue(
                    result["proxy_audit"]["exception_path_restored_joint_gradient"]
                )
                for parameter in (
                    gaussians._xyz,
                    gaussians._features_dc,
                    gaussians._features_rest,
                    gaussians._opacity,
                    gaussians._scaling,
                    gaussians._rotation,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_renderer_backward_exception_is_not_swallowed(self):
        render_pkg, gaussians, camera, _ = make_mock_graph(0.2)

        class BackwardFailure(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, _gradient):
                raise RuntimeError("synthetic renderer backward failure")

        def failing_loss(depth, _reference, **_kwargs):
            loss = BackwardFailure.apply(depth).mean()
            return loss, {"valid_pair_count": 1, "active_pair_count": 1}, {}

        with self.assertRaisesRegex(RuntimeError, "synthetic renderer backward failure"):
            evaluate_render_case(
                render_pkg,
                gaussians,
                camera,
                dropout_factor=0.2,
                lambda_dssim=0.2,
                structural_weight=0.1,
                structural_loss_fn=failing_loss,
            )

    def test_gradient_stats_reject_nonfinite_as_not_finite(self):
        stats = gradient_stats(torch.tensor([1.0, float("nan")]))
        self.assertTrue(stats["present"])
        self.assertFalse(stats["finite"])

    def test_script_contains_no_optimizer_or_densification_calls(self):
        source = (Path(__file__).resolve().parents[1] / "script" / "smoke_i1_cuda.py").read_text()
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("densify_and_prune(", source)
        self.assertNotIn("gaussians.add_densification_stats(", source)
        self.assertNotIn("torch.hub.load(", source)


if __name__ == "__main__":
    unittest.main()
