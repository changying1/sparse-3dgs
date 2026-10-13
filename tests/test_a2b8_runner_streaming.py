import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from script.run_llff_single import (  # noqa: E402
    RunPaths,
    StageFailure,
    execute_stage_plan,
    run_stage,
)


class NotifyingTextStream(io.StringIO):
    def __init__(self, marker: str):
        super().__init__()
        self.marker = marker
        self.seen = threading.Event()

    def write(self, text: str) -> int:
        written = super().write(text)
        if self.marker in self.getvalue():
            self.seen.set()
        return written


class RunnerStreamingOutputTests(unittest.TestCase):
    def _paths(self, root: Path) -> RunPaths:
        experiment = root / "output" / "streaming-test"
        return RunPaths(
            repository_root=root,
            scene_path=root / "dataset" / "horns",
            experiment_dir=experiment,
            initialization_dir=root / "initialization",
            view_manifest=experiment / "view_manifest.json",
            run_manifest=experiment / "run_manifest.json",
        )

    def test_output_is_visible_before_exit_and_log_preserves_utf8_and_carriage_returns(self):
        with tempfile.TemporaryDirectory(prefix="a2b8 streaming ") as temp:
            root = Path(temp)
            log_path = root / "output" / "experiment" / "train.log"
            console = NotifyingTextStream("Training progress: 10%")
            child = (
                "import sys,time; "
                "sys.stderr.write('Training progress: 10% | Loss=0.42 | Gaussian=123 | 5it/s | ETA 00:09\\r'); "
                "sys.stderr.flush(); time.sleep(1.0); "
                "sys.stdout.write('完成 100%\\n'); sys.stdout.flush()"
            )
            failures = []

            def invoke_runner():
                try:
                    run_stage(
                        stage="train",
                        command=[sys.executable, "-c", child],
                        log_path=log_path,
                        cwd=root,
                        environment=os.environ,
                        output_stream=console,
                    )
                except BaseException as exc:  # surfaced in the test thread below
                    failures.append(exc)

            worker = threading.Thread(target=invoke_runner)
            worker.start()
            self.assertTrue(console.seen.wait(5), "first progress update was not streamed")
            self.assertTrue(worker.is_alive(), "output only became visible after child exit")
            worker.join(5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertIn("完成 100%", console.getvalue())
            log_bytes = log_path.read_bytes()
            self.assertIn(
                "Training progress: 10% | Loss=0.42 | Gaussian=123 | 5it/s | ETA 00:09\r".encode("utf-8"),
                log_bytes,
            )
            self.assertIn("完成 100%".encode("utf-8"), log_bytes)
            self.assertIn(b"--- exit_code: 0 ---", log_bytes)

    def test_failure_output_and_real_exit_code_are_saved_in_log_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="a2b8 failure ") as temp:
            root = Path(temp)
            paths = self._paths(root)
            command = [
                sys.executable,
                "-c",
                (
                    "import sys; print('metric: PSNR=12.3'); "
                    "sys.stderr.write('Traceback: simulated failure\\n'); sys.stderr.flush(); "
                    "sys.exit(9)"
                ),
            ]
            commands = {stage: command for stage in ("init", "train", "render", "metrics")}
            record = {
                "status": "prepared",
                "stages": {
                    stage: {"status": "pending", "returncode": None}
                    for stage in commands
                },
            }

            with contextlib.redirect_stdout(io.StringIO()) as console:
                with self.assertRaises(StageFailure) as caught:
                    execute_stage_plan(
                        selected_stages=["train"],
                        commands=commands,
                        paths=paths,
                        environment=os.environ,
                        run_record=record,
                    )

            self.assertEqual(caught.exception.returncode, 9)
            self.assertIn("metric: PSNR=12.3", console.getvalue())
            self.assertIn("Traceback: simulated failure", console.getvalue())
            log = (paths.experiment_dir / "train.log").read_text(encoding="utf-8")
            self.assertIn("metric: PSNR=12.3", log)
            self.assertIn("Traceback: simulated failure", log)
            self.assertIn("--- exit_code: 9 ---", log)
            manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["stages"]["train"]["status"], "failed")
            self.assertEqual(manifest["stages"]["train"]["returncode"], 9)


if __name__ == "__main__":
    unittest.main()
