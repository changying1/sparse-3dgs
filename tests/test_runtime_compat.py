import unittest
import ast
from collections import namedtuple
from pathlib import Path

import torch

from utils.runtime_compat import (
    cuda_model_session,
    detached_depth_inference,
    detached_tensor_to,
    make_rasterization_settings,
    offload_cuda_model,
    should_sample_pseudo,
    unwrap_first_result,
)


class FakeModel:
    def __init__(self):
        self.events = []
        self.grad_enabled_during_infer = None

    def cuda(self):
        self.events.append("cuda")
        return self

    def cpu(self):
        self.events.append("cpu")
        return self

    def infer(self, image):
        self.events.append("infer")
        self.grad_enabled_during_infer = torch.is_grad_enabled()
        return {"depth": image * 2}


class RuntimeCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _load_depth_transform_adapter():
        source_path = Path(__file__).resolve().parents[1] / "utils" / "depth_utils.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_adapt_depth_pro_transform_for_tensor_input"
        )
        isolated = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {}
        exec(compile(isolated, str(source_path), "exec"), namespace)
        return namespace["_adapt_depth_pro_transform_for_tensor_input"]

    def test_depth_transform_adapter_removes_only_leading_to_tensor(self):
        class ToTensor:
            pass

        class DeviceLambda:
            pass

        class Normalize:
            pass

        class ConvertImageDtype:
            pass

        class FakeCompose:
            def __init__(self, transforms):
                self.transforms = list(transforms)

        adapter = self._load_depth_transform_adapter()
        tail = [DeviceLambda(), Normalize(), ConvertImageDtype()]
        original = FakeCompose([ToTensor(), *tail])
        adapted = adapter(original)

        self.assertIsInstance(adapted, FakeCompose)
        self.assertIsNot(adapted, original)
        self.assertEqual(len(adapted.transforms), 3)
        self.assertTrue(all(actual is expected for actual, expected in zip(adapted.transforms, tail)))

    def test_depth_transform_adapter_keeps_non_to_tensor_transform(self):
        class Normalize:
            pass

        class FakeCompose:
            def __init__(self, transforms):
                self.transforms = list(transforms)

        adapter = self._load_depth_transform_adapter()
        original = FakeCompose([Normalize()])
        self.assertIs(adapter(original), original)

    def test_depth_transform_adapter_keeps_empty_transform(self):
        class FakeCompose:
            def __init__(self, transforms):
                self.transforms = list(transforms)

        adapter = self._load_depth_transform_adapter()
        original = FakeCompose([])
        self.assertIs(adapter(original), original)

    def test_dist_result_accepts_tensor_tuple_and_list(self):
        tensor = torch.tensor([1.0])
        self.assertIs(unwrap_first_result(tensor), tensor)
        self.assertIs(unwrap_first_result((tensor, "extra")), tensor)
        self.assertIs(unwrap_first_result([tensor, "extra"]), tensor)

    def test_rasterizer_settings_with_and_without_confidence(self):
        OfficialSettings = namedtuple("OfficialSettings", "value")
        FsgsSettings = namedtuple("FsgsSettings", "value confidence")
        confidence = torch.ones(1)

        official = make_rasterization_settings(
            OfficialSettings, confidence=confidence, value=3
        )
        fsgs = make_rasterization_settings(
            FsgsSettings, confidence=confidence, value=3
        )

        self.assertEqual(official.value, 3)
        self.assertIs(fsgs.confidence, confidence)

    def test_offload_does_not_load_model(self):
        model = FakeModel()
        cache_events = []
        offload_cuda_model(model, lambda: cache_events.append("empty_cache"))
        self.assertEqual(model.events, ["cpu"])
        self.assertEqual(cache_events, ["empty_cache"])

    def test_non_pseudo_iteration_does_not_open_cuda_session(self):
        model = FakeModel()
        if should_sample_pseudo(iteration=1990, interval=10, start=2000, end=5000):
            with cuda_model_session(model, lambda: None):
                pass
        self.assertEqual(model.events, [])

        self.assertFalse(should_sample_pseudo(2000, 10, 2000, 5000))
        self.assertTrue(should_sample_pseudo(2010, 10, 2000, 5000))
        self.assertFalse(should_sample_pseudo(5000, 10, 2000, 5000))

    def test_cuda_session_offloads_after_success(self):
        model = FakeModel()
        cache_events = []
        with cuda_model_session(model, lambda: cache_events.append("empty_cache")):
            model.events.append("inference")
        self.assertEqual(model.events, ["cuda", "inference", "cpu"])
        self.assertEqual(cache_events, ["empty_cache"])

    def test_cuda_session_offloads_after_error(self):
        model = FakeModel()
        cache_events = []
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with cuda_model_session(model, lambda: cache_events.append("empty_cache")):
                raise RuntimeError("inference failed")
        self.assertEqual(model.events, ["cuda", "cpu"])
        self.assertEqual(cache_events, ["empty_cache"])

    def test_depth_inference_is_detached_and_has_no_grad(self):
        model = FakeModel()
        image = torch.tensor([2.0], requires_grad=True)
        depth = detached_depth_inference(model, lambda value: value, image)
        self.assertFalse(model.grad_enabled_during_infer)
        self.assertFalse(depth.requires_grad)
        self.assertIsNone(depth.grad_fn)

    def test_detached_target_does_not_break_render_gradient(self):
        rendered_depth = torch.tensor([1.0, 2.0], requires_grad=True)
        target_source = torch.tensor([2.0, 4.0], requires_grad=True)
        target = detached_tensor_to(target_source, rendered_depth.device)
        loss = (rendered_depth - target).square().mean()
        loss.backward()

        self.assertIsNotNone(rendered_depth.grad)
        self.assertIsNone(target_source.grad)
        self.assertFalse(target.requires_grad)


if __name__ == "__main__":
    unittest.main()
