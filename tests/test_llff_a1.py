import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import OptimizationParams
from llff_view_protocol import (
    ColmapImageRecord,
    ViewProtocolError,
    build_view_manifest,
    pdc_initialization_image_paths,
    select_pdc_initialization_records,
    sort_image_names,
    validate_colmap_records,
    validate_shared_camera_intrinsics,
    validate_view_manifest,
)
from script.run_llff_single import (
    DevelopmentConfig,
    RunPaths,
    StageFailure,
    build_stage_commands,
    execute_stage_plan,
    make_paths,
    profile_config,
    run_stage,
)


def image_names(count=25):
    return [f"image_{index:03d}.png" for index in range(count)]


class LLFFViewProtocolTests(unittest.TestCase):
    def test_zero_index_test_view_never_enters_initialization(self):
        manifest = build_view_manifest(
            dataset="LLFF",
            scene="horns",
            image_names=image_names(),
            view_count=3,
            source_commit="test-sha",
        )
        self.assertEqual(manifest["test_images"][0], "image_000.png")
        self.assertNotIn("image_000.png", manifest["pdc_initialization_images"])

    def test_three_view_selection_is_the_shared_training_selection(self):
        manifest = build_view_manifest(
            dataset="LLFF",
            scene="horns",
            image_names=image_names(17),
            view_count=3,
            source_commit="test-sha",
        )
        candidates = manifest["train_candidate_images"]
        expected = [candidates[0], candidates[round((len(candidates) - 1) / 2)], candidates[-1]]
        self.assertEqual(manifest["selected_train_images"], expected)
        self.assertEqual(
            manifest["selected_train_images"],
            manifest["pdc_initialization_images"],
        )

    def test_non_contiguous_colmap_ids_map_by_image_name(self):
        with tempfile.TemporaryDirectory() as temp:
            images_dir = Path(temp)
            for name in ("a.png", "b.png", "c.png"):
                (images_dir / name).write_bytes(b"test")
            records = [
                ColmapImageRecord(91, 7, "c.png"),
                ColmapImageRecord(3, 7, "a.png"),
                ColmapImageRecord(44, 7, "b.png"),
            ]
            ordered = validate_colmap_records(records, {7}, images_dir)
            self.assertEqual([record.image_id for record in ordered], [3, 44, 91])

    def test_image_name_sorting_is_stable_and_id_independent(self):
        self.assertEqual(
            sort_image_names(["z/02.png", "a.png", "z/01.png"]),
            ["a.png", "z/01.png", "z/02.png"],
        )

    def test_manifest_rejects_train_initialization_mismatch(self):
        manifest = build_view_manifest(
            dataset="LLFF", scene="horns", image_names=image_names(),
            view_count=3, source_commit="test-sha"
        )
        broken = copy.deepcopy(manifest)
        broken["pdc_initialization_images"] = list(reversed(broken["selected_train_images"]))
        with self.assertRaises(ViewProtocolError):
            validate_view_manifest(broken)

    def test_manifest_rejects_test_initialization_overlap(self):
        manifest = build_view_manifest(
            dataset="LLFF", scene="horns", image_names=image_names(),
            view_count=3, source_commit="test-sha"
        )
        broken = copy.deepcopy(manifest)
        broken["selected_train_images"][0] = broken["test_images"][0]
        broken["pdc_initialization_images"] = list(broken["selected_train_images"])
        with self.assertRaises(ViewProtocolError):
            validate_view_manifest(broken)

    def test_missing_image_fails_immediately(self):
        with tempfile.TemporaryDirectory() as temp:
            images_dir = Path(temp)
            (images_dir / "present.png").write_bytes(b"test")
            records = [
                ColmapImageRecord(1, 5, "present.png"),
                ColmapImageRecord(2, 5, "missing.png"),
            ]
            with self.assertRaises(ViewProtocolError):
                validate_colmap_records(records, {5}, images_dir)

    def test_pdc_read_set_contains_only_selected_training_images(self):
        names = image_names(25)
        manifest = build_view_manifest(
            dataset="LLFF", scene="horns", image_names=names,
            view_count=3, source_commit="test-sha"
        )
        records = [
            ColmapImageRecord(1000 - index * 7, 5, name)
            for index, name in enumerate(reversed(names))
        ]
        ordered = sorted(records, key=lambda record: record.image_name)
        selected = select_pdc_initialization_records(ordered, manifest)
        self.assertEqual(
            [record.image_name for record in selected],
            manifest["selected_train_images"],
        )
        self.assertTrue(
            set(record.image_name for record in selected).isdisjoint(manifest["test_images"])
        )
        paths = pdc_initialization_image_paths(selected, Path("images root"))
        self.assertEqual(
            [path.name for path in paths],
            [Path(name).name for name in manifest["selected_train_images"]],
        )

    def test_selected_image_camera_alignment_uses_manifest_order(self):
        names = image_names(17)
        manifest = build_view_manifest(
            dataset="LLFF", scene="horns", image_names=names,
            view_count=3, source_commit="test-sha"
        )
        id_by_name = {name: 500 + index * 11 for index, name in enumerate(names)}
        records = [
            ColmapImageRecord(id_by_name[name], 9, name) for name in reversed(names)
        ]
        selected = select_pdc_initialization_records(
            sorted(records, key=lambda record: record.image_name), manifest
        )
        self.assertEqual(
            [record.image_id for record in selected],
            [id_by_name[name] for name in manifest["selected_train_images"]],
        )

    def test_shared_intrinsics_are_accepted(self):
        intrinsic = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        result = validate_shared_camera_intrinsics(
            [intrinsic, copy.deepcopy(intrinsic), copy.deepcopy(intrinsic)],
            image_sizes=[(640, 480)] * 3,
            labels=["a.png", "b.png", "c.png"],
        )
        self.assertEqual(result, intrinsic)

    def test_different_intrinsics_or_dimensions_fail_closed(self):
        intrinsic = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        different = copy.deepcopy(intrinsic)
        different[0][0] = 510.0
        with self.assertRaises(ViewProtocolError):
            validate_shared_camera_intrinsics(
                [intrinsic, different],
                image_sizes=[(640, 480), (640, 480)],
                labels=["a.png", "b.png"],
            )
        with self.assertRaises(ViewProtocolError):
            validate_shared_camera_intrinsics(
                [intrinsic, copy.deepcopy(intrinsic)],
                image_sizes=[(640, 480), (800, 600)],
                labels=["a.png", "b.png"],
            )


class WindowsRunnerTests(unittest.TestCase):
    def _paths(self, root: Path) -> RunPaths:
        experiment = root / "output with spaces" / "experiment"
        return RunPaths(
            repository_root=root,
            scene_path=root / "data with spaces" / "horns",
            experiment_dir=experiment,
            initialization_dir=root / "keypoints with spaces",
            view_manifest=experiment / "view_manifest.json",
            run_manifest=experiment / "run_manifest.json",
        )

    def test_paths_with_spaces_are_single_argv_items(self):
        with tempfile.TemporaryDirectory(prefix="a1 path with spaces ") as temp:
            paths = self._paths(Path(temp))
            commands = build_stage_commands(paths, DevelopmentConfig(), sys.executable)
            self.assertIn(str(paths.scene_path), commands["train"])
            self.assertIn(str(paths.experiment_dir), commands["train"])
            self.assertNotIn(f'"{paths.scene_path}"', commands["train"])

    def test_nonzero_subprocess_stops_following_stage_and_keeps_log(self):
        with tempfile.TemporaryDirectory(prefix="a1 stop test ") as temp:
            root = Path(temp)
            paths = self._paths(root)
            marker = root / "should_not_exist.txt"
            commands = {
                "init": [sys.executable, "-c", "import sys; print('failed'); sys.exit(7)"],
                "train": [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')"],
                "render": [sys.executable, "-c", "pass"],
                "metrics": [sys.executable, "-c", "pass"],
            }
            record = {
                "status": "prepared",
                "stages": {
                    stage: {"status": "pending", "returncode": None}
                    for stage in commands
                },
            }
            with self.assertRaises(StageFailure) as caught:
                execute_stage_plan(
                    selected_stages=["init", "train"],
                    commands=commands,
                    paths=paths,
                    environment=os.environ,
                    run_record=record,
                )
            self.assertEqual(caught.exception.returncode, 7)
            self.assertFalse(marker.exists())
            self.assertTrue((paths.experiment_dir / "init.log").is_file())
            self.assertIn("failed", (paths.experiment_dir / "init.log").read_text(encoding="utf-8"))

    def test_log_is_written_to_experiment_directory(self):
        with tempfile.TemporaryDirectory(prefix="a1 logging test ") as temp:
            root = Path(temp)
            paths = self._paths(root)
            log_path = paths.experiment_dir / "train.log"
            result = run_stage(
                stage="train",
                command=[sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
                log_path=log_path,
                cwd=root,
                environment=os.environ,
            )
            self.assertEqual(result, 0)
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("out", content)
            self.assertIn("err", content)
            self.assertIn("exit_code: 0", content)

    def test_5k_and_3k_profiles_generate_explicit_iteration_commands(self):
        with tempfile.TemporaryDirectory(prefix="a1 commands ") as temp:
            paths = self._paths(Path(temp))
            for profile, expected in (("5k", "5000"), ("3k", "3000")):
                config = profile_config(profile)
                command = build_stage_commands(paths, config, sys.executable)["train"]
                index = command.index("--iterations")
                self.assertEqual(command[index + 1], expected)

    def test_official_training_default_remains_30000(self):
        parser = argparse.ArgumentParser()
        OptimizationParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.iterations, 30_000)

    def test_make_paths_keeps_space_containing_roots(self):
        with tempfile.TemporaryDirectory(prefix="repository with spaces ") as temp:
            root = Path(temp)
            paths = make_paths(
                repository_root=root,
                dataset_root=Path("data root"),
                output_root=Path("output root"),
                experiment_name="safe-name",
                config=DevelopmentConfig(),
            )
            self.assertIn(" ", str(paths.scene_path))
            self.assertEqual(paths.experiment_dir.name, "safe-name")


if __name__ == "__main__":
    unittest.main()
