import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from llff_view_protocol import build_view_manifest, write_view_manifest
from script.run_llff_single import (
    DevelopmentConfig,
    FSGS_PYTHON,
    InitializationArtifactError,
    RunPaths,
    build_stage_commands,
    execute_stage_plan,
    initialization_ply_path,
    inspect_initialization_ply,
    make_run_record,
    parse_args,
    sha256_file,
    stage_frozen_initialization,
    write_json,
)


def write_valid_binary_ply(path: Path, offset: float = 0.0) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment synthetic A2-B3 fixture\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    payload = b"".join(
        (
            struct.pack("<fffBBBB", offset + 0.0, 0.0, 1.0, 255, 0, 10, 255),
            struct.pack("<fffBBBB", offset + 0.5, 0.25, 2.0, 0, 128, 255, 255),
        )
    )
    path.write_bytes(header + payload)


class FrozenInitializationTests(unittest.TestCase):
    def _paths(self, root: Path, experiment: str) -> RunPaths:
        experiment_dir = root / "output" / experiment
        return RunPaths(
            repository_root=root,
            scene_path=root / "dataset" / "horns",
            experiment_dir=experiment_dir,
            initialization_dir=root / "keypoints_to_3d" / f"LLFF_{experiment}",
            view_manifest=experiment_dir / "view_manifest.json",
            run_manifest=experiment_dir / "run_manifest.json",
        )

    def _manifest(self, paths: RunPaths, source_commit: str = "frozen-test-sha") -> dict:
        manifest = build_view_manifest(
            dataset="LLFF",
            scene="horns",
            image_names=[f"image_{index:03d}.png" for index in range(17)],
            view_count=3,
            source_commit=source_commit,
        )
        write_view_manifest(paths.view_manifest, manifest)
        return manifest

    def _stage(self, source: Path, paths: RunPaths, manifest: dict) -> dict:
        return stage_frozen_initialization(
            source_path=source,
            expected_sha256=sha256_file(source),
            paths=paths,
            config=DevelopmentConfig(),
            view_manifest=manifest,
        )

    def test_valid_ply_is_verified_and_copied_byte_identically(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 valid ") as temp:
            root = Path(temp)
            source = root / "frozen source.ply"
            write_valid_binary_ply(source)
            source_bytes = source.read_bytes()
            paths = self._paths(root, "baseline")
            artifact = self._stage(source, paths, self._manifest(paths))
            target = initialization_ply_path(paths, DevelopmentConfig())

            self.assertEqual(artifact["status"], "verified")
            self.assertEqual(artifact["copy_action"], "copied")
            self.assertEqual(artifact["source"]["vertex_count"], 2)
            self.assertEqual(artifact["destination"]["sha256"], sha256_file(source))
            self.assertEqual(target.read_bytes(), source_bytes)
            self.assertEqual(source.read_bytes(), source_bytes)

            commands = build_stage_commands(paths, DevelopmentConfig(), FSGS_PYTHON)
            record = make_run_record(
                config=DevelopmentConfig(), paths=paths, git_sha="test-sha",
                selected_stages=["train"], commands=commands,
                external_initialization=True,
            )
            record["initialization_artifact"] = artifact
            write_json(paths.run_manifest, record)
            serialized = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(serialized["initialization_artifact"]["source"]["sha256"], sha256_file(source))
            self.assertEqual(serialized["initialization_artifact"]["destination"]["sha256"], sha256_file(source))

    def test_wrong_hash_fails_immediately(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 hash ") as temp:
            root = Path(temp)
            source = root / "source.ply"
            write_valid_binary_ply(source)
            paths = self._paths(root, "baseline")
            manifest = self._manifest(paths)
            with self.assertRaisesRegex(InitializationArtifactError, "SHA-256 mismatch"):
                stage_frozen_initialization(
                    source_path=source,
                    expected_sha256="0" * 64,
                    paths=paths,
                    config=DevelopmentConfig(),
                    view_manifest=manifest,
                )
            self.assertFalse(initialization_ply_path(paths, DevelopmentConfig()).exists())

    def test_missing_file_fails_immediately(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 missing ") as temp:
            root = Path(temp)
            paths = self._paths(root, "baseline")
            manifest = self._manifest(paths)
            with self.assertRaisesRegex(InitializationArtifactError, "does not exist"):
                stage_frozen_initialization(
                    source_path=root / "missing.ply",
                    expected_sha256="0" * 64,
                    paths=paths,
                    config=DevelopmentConfig(),
                    view_manifest=manifest,
                )

    def test_invalid_ply_fails_before_copy(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 invalid ") as temp:
            root = Path(temp)
            source = root / "invalid.ply"
            source.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n")
            paths = self._paths(root, "baseline")
            manifest = self._manifest(paths)
            with self.assertRaisesRegex(InitializationArtifactError, "missing required properties"):
                self._stage(source, paths, manifest)
            self.assertFalse(initialization_ply_path(paths, DevelopmentConfig()).exists())

    def test_existing_different_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 no overwrite ") as temp:
            root = Path(temp)
            source = root / "source.ply"
            write_valid_binary_ply(source)
            paths = self._paths(root, "baseline")
            manifest = self._manifest(paths)
            target = initialization_ply_path(paths, DevelopmentConfig())
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing different bytes")
            original_target = target.read_bytes()

            with self.assertRaisesRegex(InitializationArtifactError, "Refusing to overwrite"):
                self._stage(source, paths, manifest)
            self.assertEqual(target.read_bytes(), original_target)

    def test_baseline_i1_i2_share_one_source_and_identical_manifest(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 shared ") as temp:
            root = Path(temp)
            source = root / "shared.ply"
            write_valid_binary_ply(source)
            source_before = source.read_bytes()
            paths_a = self._paths(root, "baseline")
            paths_b = self._paths(root, "i1")
            paths_c = self._paths(root, "i2")
            artifact_a = self._stage(source, paths_a, self._manifest(paths_a))
            artifact_b = self._stage(source, paths_b, self._manifest(paths_b))
            artifact_c = self._stage(source, paths_c, self._manifest(paths_c))

            for paths, artifact in (
                (paths_a, artifact_a), (paths_b, artifact_b), (paths_c, artifact_c)
            ):
                commands = build_stage_commands(paths, DevelopmentConfig(), FSGS_PYTHON)
                record = make_run_record(
                    config=DevelopmentConfig(), paths=paths, git_sha="test-sha",
                    selected_stages=["train"], commands=commands,
                    external_initialization=True,
                )
                record["initialization_artifact"] = artifact
                write_json(paths.run_manifest, record)

            self.assertEqual(paths_a.view_manifest.read_bytes(), paths_b.view_manifest.read_bytes())
            self.assertEqual(paths_a.view_manifest.read_bytes(), paths_c.view_manifest.read_bytes())
            self.assertEqual(artifact_a["view_manifest_sha256"], artifact_b["view_manifest_sha256"])
            self.assertEqual(artifact_a["view_manifest_sha256"], artifact_c["view_manifest_sha256"])
            self.assertEqual(artifact_a["source"]["sha256"], artifact_b["source"]["sha256"])
            self.assertEqual(artifact_a["source"]["sha256"], artifact_c["source"]["sha256"])
            self.assertEqual(artifact_a["destination"]["sha256"], artifact_b["destination"]["sha256"])
            self.assertEqual(artifact_a["destination"]["sha256"], artifact_c["destination"]["sha256"])
            self.assertEqual(initialization_ply_path(paths_a, DevelopmentConfig()).read_bytes(), source_before)
            self.assertEqual(initialization_ply_path(paths_b, DevelopmentConfig()).read_bytes(), source_before)
            self.assertEqual(initialization_ply_path(paths_c, DevelopmentConfig()).read_bytes(), source_before)
            self.assertEqual(source.read_bytes(), source_before)
            for paths in (paths_a, paths_b, paths_c):
                recorded = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
                self.assertEqual(
                    recorded["initialization_artifact"]["source"]["sha256"],
                    sha256_file(source),
                )
                self.assertEqual(
                    recorded["initialization_artifact"]["destination"]["sha256"],
                    sha256_file(source),
                )

    def test_manifest_change_refuses_to_start_training(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 manifest guard ") as temp:
            root = Path(temp)
            source = root / "source.ply"
            write_valid_binary_ply(source)
            paths = self._paths(root, "baseline")
            manifest = self._manifest(paths)
            artifact = self._stage(source, paths, manifest)

            changed = copy.deepcopy(manifest)
            changed["source_commit"] = "unexpected-different-manifest"
            write_view_manifest(paths.view_manifest, changed)
            marker = root / "train_started.txt"
            commands = {
                "init": [sys.executable, "-c", "pass"],
                "train": [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('started')"],
                "render": [sys.executable, "-c", "pass"],
                "metrics": [sys.executable, "-c", "pass"],
            }
            record = {
                "status": "prepared",
                "initialization_artifact": artifact,
                "stages": {
                    stage: {"status": "pending", "returncode": None}
                    for stage in commands
                },
            }
            with self.assertRaisesRegex(InitializationArtifactError, "View manifest changed"):
                execute_stage_plan(
                    selected_stages=["train"],
                    commands=commands,
                    paths=paths,
                    environment=os.environ,
                    run_record=record,
                )
            self.assertFalse(marker.exists())
            self.assertEqual(record["stages"]["train"]["status"], "failed")

    def test_train_only_command_uses_fsgs_and_never_calls_pdc(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 train only ") as temp:
            root = Path(temp)
            paths = self._paths(root, "baseline_5k")
            source = root / "source.ply"
            write_valid_binary_ply(source)
            self._stage(source, paths, self._manifest(paths))
            commands = build_stage_commands(paths, DevelopmentConfig(iterations=5000), FSGS_PYTHON)
            train = commands["train"]
            legacy_record = make_run_record(
                config=DevelopmentConfig(), paths=paths, git_sha="test-sha",
                selected_stages=["train"], commands=commands,
            )

            self.assertEqual(train[0], FSGS_PYTHON)
            self.assertEqual(parse_args([]).python, FSGS_PYTHON)
            self.assertIn("5000", train)
            self.assertNotIn("triangulate.py", " ".join(train))
            self.assertEqual(legacy_record["manifest_version"], "llff-single-run-v1")
            self.assertNotIn("initialization_artifact", legacy_record)

    def test_complete_ascii_payload_is_readable(self):
        with tempfile.TemporaryDirectory(prefix="a2b3 ascii ") as temp:
            path = Path(temp) / "valid_ascii.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n0.0 1.0 2.0 10 20 30\n",
                encoding="ascii",
            )
            info = inspect_initialization_ply(path)
            self.assertEqual(info["format"], "ascii")
            self.assertEqual(info["vertex_count"], 1)


if __name__ == "__main__":
    unittest.main()
