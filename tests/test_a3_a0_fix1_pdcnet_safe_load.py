import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DENSE_MATCHER_ROOT = REPOSITORY_ROOT / 'submodules' / 'dense_matcher'
if str(DENSE_MATCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(DENSE_MATCHER_ROOT))

from admin.local import EnvironmentSettings
from admin.model_constructor import NetConstructor
from admin.settings import Settings
from admin.stats import AverageMeter, StatValue


# Loading model_selection normally imports every supported architecture, including
# the optional CUDA/CuPy GOCor extension. These unit tests exercise only checkpoint
# loading, so use inert architecture modules and keep the test CPU-only.
def _stub_model_module(name, exported_names):
    module = types.ModuleType(name)
    for exported_name in exported_names:
        setattr(module, exported_name, object())
    sys.modules[name] = module


_stub_model_module('models.GLUNet.GLU_Net', ['GLUNetModel'])
_stub_model_module('models.PWCNet.pwc_net', ['PWCNetModel'])
_stub_model_module('models.PDCNet.PDCNet', ['PDCNet_vgg16'])
_stub_model_module('models.GLUNet.Semantic_GLUNet', ['SemanticGLUNetModel'])
_stub_model_module('models.semantic_matching_models.SFNet', ['SFNet', 'SFNetWithBin'])
_stub_model_module('models.semantic_matching_models.NCNet', ['NCNetWithBin', 'ImMatchNet'])
_stub_model_module('models.semantic_matching_models.cats', ['CATs'])

from model_selection import load_network


class UnexpectedCheckpointType:
    pass


def _legacy_metadata():
    return {
        'constructor': NetConstructor('unused', 'unused', (), {}),
        'environment': EnvironmentSettings(),
        'settings': Settings(),
        'average_meter': AverageMeter(),
        'stat_value': StatValue(),
    }


class PDCNetSafeLoadTests(unittest.TestCase):
    def _save_checkpoint(self, path, state_dict, **extra):
        checkpoint = {'state_dict': state_dict, **_legacy_metadata(), **extra}
        torch.save(checkpoint, path)

    def test_safe_load_accepts_expected_legacy_types_without_global_registration(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)

        with tempfile.TemporaryDirectory(prefix='pdcnet-safe-load-') as temp_dir:
            checkpoint_path = Path(temp_dir) / 'PDCNet_plus_megadepth.pth.tar'
            self._save_checkpoint(checkpoint_path, source.state_dict())
            safe_globals_before = list(torch.serialization.get_safe_globals())

            loaded = load_network(
                target,
                checkpoint_path=str(checkpoint_path),
                safe_pdcnet_plus_megadepth=True,
            )

            self.assertIs(loaded, target)
            self.assertEqual(safe_globals_before, torch.serialization.get_safe_globals())
            for name, parameter in source.state_dict().items():
                torch.testing.assert_close(parameter, target.state_dict()[name])

    def test_safe_load_rejects_unexpected_checkpoint_type(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory(prefix='pdcnet-unsafe-type-') as temp_dir:
            checkpoint_path = Path(temp_dir) / 'PDCNet_plus_megadepth.pth.tar'
            self._save_checkpoint(
                checkpoint_path,
                model.state_dict(),
                unexpected=UnexpectedCheckpointType(),
            )

            with self.assertRaisesRegex(RuntimeError, 'outside the local safe-global allowlist'):
                load_network(
                    model,
                    checkpoint_path=str(checkpoint_path),
                    safe_pdcnet_plus_megadepth=True,
                )

    def test_safe_load_rejects_parameter_mismatches(self):
        valid_state = torch.nn.Linear(3, 2).state_dict()
        mismatches = {
            'missing key': {'weight': valid_state['weight']},
            'unexpected key': {**valid_state, 'unexpected': torch.zeros(1)},
            'shape mismatch': {**valid_state, 'weight': torch.zeros(4, 3)},
        }

        for mismatch_name, state_dict in mismatches.items():
            with self.subTest(mismatch=mismatch_name):
                model = torch.nn.Linear(3, 2)
                with tempfile.TemporaryDirectory(prefix='pdcnet-mismatch-') as temp_dir:
                    checkpoint_path = Path(temp_dir) / 'PDCNet_plus_megadepth.pth.tar'
                    self._save_checkpoint(checkpoint_path, state_dict)

                    with self.assertRaisesRegex(RuntimeError, 'do not exactly match'):
                        load_network(
                            model,
                            checkpoint_path=str(checkpoint_path),
                            safe_pdcnet_plus_megadepth=True,
                        )


if __name__ == '__main__':
    unittest.main()
