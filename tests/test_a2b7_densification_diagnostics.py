import argparse
import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from utils.densification_diagnostics import (  # noqa: E402
    DensificationEventLogger,
    add_densification_diagnostics_argument,
    build_densification_event,
    optimizer_dimension_errors,
)


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


def make_optimizer(gaussian_count, bad_group=None, bad_state=None):
    groups = []
    state = {}
    for name in ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"):
        parameter_count = gaussian_count - 1 if name == bad_group else gaussian_count
        parameter = FakeTensor(parameter_count, 3)
        groups.append({"name": name, "params": [parameter]})
        exp_avg_count = gaussian_count - 2 if name == bad_state else gaussian_count
        state[parameter] = {
            "exp_avg": FakeTensor(exp_avg_count, 3),
            "exp_avg_sq": FakeTensor(gaussian_count, 3),
        }
    return SimpleNamespace(param_groups=groups, state=state)


class DensificationDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_default_is_disabled(self):
        parser = argparse.ArgumentParser()
        add_densification_diagnostics_argument(parser)
        self.assertFalse(parser.parse_args([]).densification_diagnostics)
        self.assertTrue(
            parser.parse_args(["--densification_diagnostics"]).densification_diagnostics
        )

    def test_count_conservation(self):
        row = build_densification_event(
            1100,
            {
                "gaussians_before": 19397,
                "clone_added": 40,
                "split_parents_selected": 15,
                "split_children_created": 30,
                "split_parents_deleted": 15,
                "opacity_prune_deleted": 640,
                "size_prune_deleted": 0,
                "prune_deleted_total": 640,
                "gaussians_after": 18812,
            },
        )
        self.assertEqual(row["expected_gaussians_after"], 18812)
        self.assertEqual(row["conservation_delta"], 0)
        self.assertTrue(row["conservation_ok"])

    def test_count_conservation_failure_is_detected(self):
        row = build_densification_event(
            1100,
            {
                "gaussians_before": 10,
                "clone_added": 2,
                "split_parents_selected": 1,
                "split_children_created": 2,
                "split_parents_deleted": 1,
                "opacity_prune_deleted": 1,
                "prune_deleted_total": 1,
                "gaussians_after": 13,
            },
        )
        self.assertFalse(row["conservation_ok"])
        self.assertEqual(row["conservation_delta"], 1)

    def test_every_optimizer_group_and_adam_moment_is_checked(self):
        self.assertEqual(optimizer_dimension_errors(make_optimizer(12), 12), [])

        parameter_errors = optimizer_dimension_errors(
            make_optimizer(12, bad_group="scaling"), 12
        )
        self.assertTrue(any("scaling[0] parameter" in item for item in parameter_errors))

        state_errors = optimizer_dimension_errors(
            make_optimizer(12, bad_state="opacity"), 12
        )
        self.assertTrue(any("opacity[0] exp_avg" in item for item in state_errors))

    def test_logger_writes_checked_csv_to_experiment_directory(self):
        counts = {
            "gaussians_before": 10,
            "clone_added": 2,
            "split_parents_selected": 1,
            "split_children_created": 2,
            "split_parents_deleted": 1,
            "opacity_prune_deleted": 1,
            "size_prune_deleted": 0,
            "prune_deleted_total": 1,
            "gaussians_after": 12,
        }
        with tempfile.TemporaryDirectory() as model_path:
            logger = DensificationEventLogger(model_path)
            row = logger.record(1100, counts, make_optimizer(12))
            self.assertTrue(row["conservation_ok"])
            self.assertTrue(row["optimizer_dimensions_ok"])
            self.assertEqual(
                logger.path,
                os.path.join(model_path, "densification_events.csv"),
            )
            with open(logger.path, newline="", encoding="utf-8") as csv_file:
                saved = list(csv.DictReader(csv_file))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["iteration"], "1100")
            self.assertEqual(saved[0]["optimizer_dimensions_ok"], "True")


if __name__ == "__main__":
    unittest.main()
