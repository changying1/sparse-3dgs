import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from script.run_llff_single import (  # noqa: E402
    DevelopmentConfig,
    FSGS_PYTHON,
    RunPaths,
    build_stage_commands,
    make_run_record,
    parse_args,
)


class RunnerDensificationDiagnosticsTests(unittest.TestCase):
    def _paths(self, root: Path) -> RunPaths:
        experiment = root / "output" / "a2b7_horns_densification_diag_1200"
        return RunPaths(
            repository_root=root,
            scene_path=root / "dataset" / "horns",
            experiment_dir=experiment,
            initialization_dir=(
                root
                / "keypoints_to_3d"
                / "LLFF_a2b7_horns_densification_diag_1200"
            ),
            view_manifest=experiment / "view_manifest.json",
            run_manifest=experiment / "run_manifest.json",
        )

    def test_flag_is_off_by_default_and_baseline_commands_are_unchanged(self):
        self.assertFalse(parse_args([]).densification_diagnostics)
        with tempfile.TemporaryDirectory(prefix="a2b7 runner off ") as temp:
            paths = self._paths(Path(temp))
            commands = build_stage_commands(
                paths,
                DevelopmentConfig(iterations=1200),
                FSGS_PYTHON,
            )
            for command in commands.values():
                self.assertNotIn("--densification_diagnostics", command)

    def test_enabled_flag_is_forwarded_only_to_train_and_never_invokes_pdc(self):
        args = parse_args(["--densification-diagnostics"])
        self.assertTrue(args.densification_diagnostics)
        self.assertEqual(args.stages, ["manifest"])

        with tempfile.TemporaryDirectory(prefix="a2b7 runner on ") as temp:
            paths = self._paths(Path(temp))
            config = DevelopmentConfig(
                iterations=1200,
                densification_diagnostics=True,
            )
            commands = build_stage_commands(paths, config, FSGS_PYTHON)

            self.assertEqual(commands["train"].count("--densification_diagnostics"), 1)
            for stage in ("init", "render", "metrics"):
                self.assertNotIn("--densification_diagnostics", commands[stage])
            self.assertNotIn("triangulate.py", " ".join(commands["train"]))

            run_record = make_run_record(
                config=config,
                paths=paths,
                git_sha="test-sha",
                selected_stages=["train"],
                commands=commands,
                external_initialization=True,
            )
            self.assertEqual(run_record["selected_stages"], ["train"])
            self.assertEqual(run_record["stages"]["init"]["status"], "not_selected")
            self.assertEqual(
                run_record["stages"]["train"]["log"],
                str(paths.experiment_dir / "train.log"),
            )
            model_path_index = commands["train"].index("-m") + 1
            self.assertEqual(commands["train"][model_path_index], str(paths.experiment_dir))
            self.assertEqual(
                paths.experiment_dir / "densification_events.csv",
                Path(commands["train"][model_path_index])
                / "densification_events.csv",
            )


if __name__ == "__main__":
    unittest.main()
