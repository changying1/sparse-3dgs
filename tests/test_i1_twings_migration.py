import ast
import unittest
from pathlib import Path

import torch

from utils.i1_evidence_structural_loss import (
    _pair_terms,
    adapt_depthpro_reference,
    combine_i1_loss,
    compute_depthpro_evidence_maps,
    compute_i1_structural_loss,
    compute_stable_mask_from_gt_rgb,
    resolve_i1_gate_mode,
    should_preserve_i1_densification_stats,
)


ROOT = Path(__file__).resolve().parents[1]


class I1TwingsMigrationTests(unittest.TestCase):
    def test_frozen_schedule_boundaries_and_disabled_path(self):
        expected = {
            1599: None,
            1600: "stable_only",
            1999: "stable_only",
            2000: "stable_evidence",
            5000: "stable_evidence",
            5001: None,
        }
        for iteration, mode in expected.items():
            self.assertEqual(resolve_i1_gate_mode(iteration), mode)
        self.assertIsNone(resolve_i1_gate_mode(2000, enabled=False))

    def test_stable_mask_matches_gray_gradient_quantile(self):
        rgb = torch.zeros((3, 3, 3))
        rgb[:, :, 2] = 1.0
        stable, threshold, magnitude = compute_stable_mask_from_gt_rgb(rgb, 0.80)
        gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        dy[:-1, :] = gray[1:, :] - gray[:-1, :]
        expected_magnitude = torch.sqrt(dx.square() + dy.square())
        expected_threshold = torch.quantile(expected_magnitude, 0.80)
        self.assertTrue(torch.equal(stable, expected_magnitude < expected_threshold))
        self.assertAlmostEqual(threshold, expected_threshold.item())
        self.assertTrue(torch.allclose(magnitude, expected_magnitude))

    def test_perfect_match_and_affine_invariance(self):
        reference = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        stable = torch.ones_like(reference, dtype=torch.bool)
        for rendered in (reference.clone(), 3.0 * reference + 7.0):
            loss, _, _ = compute_i1_structural_loss(
                rendered, reference, stable, "stable_evidence"
            )
            self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_structural_mismatch_is_positive(self):
        reference = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        rendered = torch.tensor([[1.0, 4.0], [2.0, 3.0]])
        loss, _, _ = compute_i1_structural_loss(
            rendered, reference, torch.ones_like(reference, dtype=torch.bool), "stable_only"
        )
        self.assertGreater(loss.item(), 0.0)

    def test_stable_pair_requires_both_endpoints(self):
        rendered = torch.tensor([[0.0, 1.0]])
        terms = _pair_terms(
            rendered,
            torch.zeros_like(rendered),
            torch.ones_like(rendered),
            torch.tensor([[True, False]]),
            torch.ones_like(rendered, dtype=torch.bool),
            dim=1,
        )
        self.assertFalse(terms[2][0])

    def test_pair_evidence_uses_minimum_endpoint(self):
        rendered = torch.tensor([[0.0, 1.0]])
        terms = _pair_terms(
            rendered,
            torch.zeros_like(rendered),
            torch.tensor([[0.2, 0.8]]),
            torch.ones_like(rendered, dtype=torch.bool),
            torch.ones_like(rendered, dtype=torch.bool),
            dim=1,
        )
        self.assertAlmostEqual(terms[1][0].item(), 0.2, places=6)

    def test_gradients_only_flow_to_rendered_depth(self):
        rendered = torch.tensor([[1.0, 3.0], [2.0, 6.0]], requires_grad=True)
        reference = torch.tensor([[1.0, 2.0], [4.0, 5.0]], requires_grad=True)
        loss, _, maps = compute_i1_structural_loss(
            rendered,
            reference,
            torch.ones_like(rendered, dtype=torch.bool),
            "stable_evidence",
        )
        loss.backward()
        self.assertIsNotNone(rendered.grad)
        self.assertGreater(rendered.grad.abs().sum().item(), 0.0)
        self.assertIsNone(reference.grad)
        self.assertFalse(maps["evidence"].requires_grad)

    def test_invalid_and_zero_variance_are_safe_zero(self):
        for rendered, reference in (
            (torch.ones((2, 2), requires_grad=True), torch.ones((2, 2))),
            (
                torch.full((2, 2), float("nan"), requires_grad=True),
                torch.ones((2, 2)),
            ),
        ):
            loss, _, _ = compute_i1_structural_loss(
                rendered,
                reference,
                torch.ones((2, 2), dtype=torch.bool),
                "stable_evidence",
            )
            self.assertEqual(loss.item(), 0.0)
            self.assertTrue(torch.isfinite(loss))

    def test_depthpro_adapter_is_direct_and_detached(self):
        rendered = torch.zeros((2, 2))
        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        adapted = adapt_depthpro_reference(reference, rendered)
        self.assertTrue(torch.equal(adapted, reference.detach()))
        self.assertFalse(adapted.requires_grad)
        maps = compute_depthpro_evidence_maps(rendered + torch.tensor([[0.0, 1.0], [2.0, 4.0]]), reference)
        self.assertEqual(maps["reference_mode"], "depthpro_direct")
        source = (ROOT / "utils" / "i1_evidence_structural_loss.py").read_text(encoding="utf-8")
        self.assertNotIn("negative_midas", source)
        self.assertNotIn("inverse_midas_shift_200", source)

    def test_baseline_off_returns_identical_loss_object(self):
        baseline = torch.tensor(2.0, requires_grad=True)
        self.assertIs(combine_i1_loss(baseline, None, 0.05, None), baseline)

    def test_densification_override_default_and_counting(self):
        source = (ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        method = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "add_densification_stats"
        )
        function = ast.FunctionDef(
            name=method.name,
            args=method.args,
            body=method.body,
            decorator_list=[],
            returns=method.returns,
            type_comment=method.type_comment,
        )
        isolated = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {"torch": torch}
        exec(compile(isolated, str(ROOT / "scene" / "gaussian_model.py"), "exec"), namespace)

        class FakeModel:
            pass

        model = FakeModel()
        model.xyz_gradient_accum = torch.zeros((2, 1))
        model.denom = torch.zeros((2, 1))
        viewspace = torch.zeros((2, 3), requires_grad=True)
        viewspace.grad = torch.tensor([[3.0, 4.0, 9.0], [5.0, 12.0, 9.0]])
        visible = torch.tensor([True, False])
        namespace["add_densification_stats"](model, viewspace, visible)
        self.assertTrue(torch.equal(model.xyz_gradient_accum[:, 0], torch.tensor([5.0, 0.0])))
        override = torch.tensor([[6.0, 8.0, 0.0], [0.0, 2.0, 0.0]])
        namespace["add_densification_stats"](model, viewspace, visible, override)
        self.assertTrue(torch.equal(model.xyz_gradient_accum[:, 0], torch.tensor([15.0, 0.0])))
        self.assertTrue(torch.equal(model.denom[:, 0], torch.tensor([2.0, 0.0])))

    def test_i1_optimizer_and_densification_gradient_isolation(self):
        parameter = torch.tensor([2.0], requires_grad=True)
        viewspace = parameter * torch.tensor([1.0, 2.0, 3.0])
        viewspace.retain_grad()
        baseline = viewspace[0] + 2.0 * viewspace[1]
        structural = 3.0 * viewspace[2]
        baseline_grad = torch.autograd.grad(baseline, viewspace, retain_graph=True)[0].detach()
        # ``retain_grad`` also records the probe gradient on this non-leaf;
        # training clears it before the real total-loss backward for the same reason.
        viewspace.grad = None
        combine_i1_loss(baseline, structural, 1.0, "stable_evidence").backward()
        self.assertTrue(torch.equal(baseline_grad, torch.tensor([1.0, 2.0, 0.0])))
        self.assertTrue(torch.equal(viewspace.grad, torch.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(parameter.grad.item(), 14.0)
        self.assertTrue(should_preserve_i1_densification_stats(True, "stable_only", 999, 1000))
        self.assertFalse(should_preserve_i1_densification_stats(True, None, 999, 1000))

    def test_train_applies_i1_only_to_real_render_package(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn('compute_i1_structural_loss(\n                render_pkg["depth"][0]', source)
        self.assertNotIn("compute_i1_structural_loss(\n                render_pkg_pseudo", source)


if __name__ == "__main__":
    unittest.main()
