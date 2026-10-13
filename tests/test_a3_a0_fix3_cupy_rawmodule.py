import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORRELATION_MODULES = (
    REPOSITORY_ROOT / 'submodules' / 'dense_matcher' / 'third_party' / 'GOCor' / 'GOCor'
    / 'local_correlation' / 'correlation.py',
    REPOSITORY_ROOT / 'submodules' / 'dense_matcher' / 'models' / 'modules'
    / 'local_correlation' / 'correlation.py',
)


def _identity_memoize(**_kwargs):
    return lambda function: function


def _load_correlation_module(path, index):
    fake_cupy = types.ModuleType('cupy')
    fake_cupy_util = types.ModuleType('cupy.util')
    fake_cupy_util.memoize = _identity_memoize
    fake_cupy.util = fake_cupy_util
    fake_cupy.cuda = types.SimpleNamespace(compile_with_cache=mock.Mock())
    fake_cupy.RawModule = mock.Mock()

    module_name = '_a3_fix3_correlation_{}'.format(index)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    fake_stream = types.SimpleNamespace(cuda_stream=0)
    with mock.patch.dict(sys.modules, {'cupy': fake_cupy, 'cupy.util': fake_cupy_util}), \
            mock.patch.object(torch.cuda, 'current_stream', return_value=fake_stream):
        spec.loader.exec_module(module)
    return module, fake_cupy


class CuPyRawModuleCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [
            _load_correlation_module(path, index)
            for index, path in enumerate(CORRELATION_MODULES)
        ]

    def test_rawmodule_compiles_with_legacy_equivalent_defaults_and_function_name(self):
        for correlation, fake_cupy in self.modules:
            with self.subTest(module=correlation.__file__):
                kernel = object()
                compiled_module = mock.Mock()
                compiled_module.get_function.return_value = kernel
                fake_cupy.RawModule.reset_mock()
                fake_cupy.RawModule.return_value = compiled_module
                fake_cupy.cuda.compile_with_cache.reset_mock()

                result = correlation._compile_kernel('kernel_Correlation_updateOutput', 'CUDA SOURCE')

                self.assertIs(result, kernel)
                fake_cupy.RawModule.assert_called_once_with(
                    code='CUDA SOURCE', options=(), backend='nvrtc'
                )
                compiled_module.get_function.assert_called_once_with(
                    'kernel_Correlation_updateOutput'
                )
                fake_cupy.cuda.compile_with_cache.assert_not_called()

    def test_cupy_launch_delegates_to_compile_wrapper(self):
        for correlation, _fake_cupy in self.modules:
            with self.subTest(module=correlation.__file__), \
                    mock.patch.object(correlation, '_compile_kernel', return_value='kernel') as compile_kernel:
                self.assertEqual(
                    correlation.cupy_launch('kernel_Correlation_rearrange', 'CUDA SOURCE'),
                    'kernel',
                )
                compile_kernel.assert_called_once_with(
                    'kernel_Correlation_rearrange', 'CUDA SOURCE'
                )

    def test_missing_rawmodule_is_rejected_without_legacy_fallback(self):
        for correlation, fake_cupy in self.modules:
            with self.subTest(module=correlation.__file__):
                saved_raw_module = fake_cupy.RawModule
                del fake_cupy.RawModule
                fake_cupy.cuda.compile_with_cache.reset_mock()
                try:
                    with self.assertRaisesRegex(RuntimeError, 'requires? cupy.RawModule'):
                        correlation._compile_kernel('kernel_name', 'CUDA SOURCE')
                finally:
                    fake_cupy.RawModule = saved_raw_module
                fake_cupy.cuda.compile_with_cache.assert_not_called()

    def test_compile_and_function_lookup_failures_are_explicit(self):
        for correlation, fake_cupy in self.modules:
            for failure_stage in ('compile', 'get_function'):
                with self.subTest(module=correlation.__file__, stage=failure_stage):
                    fake_cupy.cuda.compile_with_cache.reset_mock()
                    if failure_stage == 'compile':
                        fake_cupy.RawModule.side_effect = ValueError('compile failed')
                    else:
                        compiled_module = mock.Mock()
                        compiled_module.get_function.side_effect = LookupError('symbol missing')
                        fake_cupy.RawModule.side_effect = None
                        fake_cupy.RawModule.return_value = compiled_module

                    with self.assertRaisesRegex(
                            RuntimeError, "kernel 'kernel_Correlation_rearrange'"):
                        correlation._compile_kernel(
                            'kernel_Correlation_rearrange', 'CUDA SOURCE'
                        )
                    fake_cupy.cuda.compile_with_cache.assert_not_called()

                fake_cupy.RawModule.side_effect = None


if __name__ == '__main__':
    unittest.main()
