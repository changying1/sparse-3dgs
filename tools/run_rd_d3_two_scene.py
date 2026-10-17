#!/usr/bin/env python3
"""RD-D3 diagnostic batch: Fortress/T-Rex Release-only and Birth-only.

Requires the explicitly defined rd-d3-smoke/rd-d3-formal CLI in
Codex_RD_D3_一次性实现与验收任务.md.  No execution without --execute.
All sources and original RD-v1/Control outputs are READ ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_TAG = "RDv1_20261016_153009_622830"
BASE = ROOT / "reports" / "i2_audit" / BASE_TAG
AUDIT_BASE = ROOT / "reports" / "i2_audit"
SCENES = ("fortress", "trex")
ARMS = ("release", "birth_only")
CODE_PATHS = (
    "train.py", "utils/i2_active_pilot.py", "utils/i2_scene_pipeline.py",
    "utils/i2_adaptive_k.py", "tools/run_i2_scene.py",
    "tools/run_rd_d3_two_scene.py", "tools/evaluate_i2_5k.py",
    "configs/i2_method_v2.json", "configs/i2_adaptive_k_rd_v1.json",
    "configs/i2_scenes/fortress.json", "configs/i2_scenes/trex.json",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_original() -> tuple[dict, dict]:
    path = BASE / "batch_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Frozen RD-v1 batch manifest is missing: {path}")
    m = read(path)
    if m.get("batch_id") != BASE_TAG or m.get("status") != "PASS":
        raise RuntimeError("Original RD-v1 batch must be PASS")
    verified = {}
    for scene in SCENES:
        row = m["scenes"][scene]
        if any(row.get(stage) != "PASS" for stage in (
            "prepare", "adaptive_smoke", "adaptive_formal", "adaptive_control_formal"
        )):
            raise RuntimeError(f"{scene}: original RD-v1 stages are not all PASS")
        k = row.get("K")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or k != min(row["R"], row["D"]):
            raise RuntimeError(f"{scene}: invalid frozen K")
        decision_path = Path(row["decision"])
        if not decision_path.is_file() or sha(decision_path) != row["decision_sha256"]:
            raise RuntimeError(f"{scene}: Decision hash mismatch")
        decision = read(decision_path)
        plan_ref = decision.get("adaptive_plan", {})
        plan_path = Path(plan_ref.get("path", ""))
        if not plan_path.is_file() or sha(plan_path) != plan_ref.get("sha256"):
            raise RuntimeError(f"{scene}: original RD-v1 Plan hash mismatch")
        if decision.get("selected_k") != k or decision.get("policy_version") != "RD-v1":
            raise RuntimeError(f"{scene}: original Decision identity mismatch")
        for prefix in ("adaptive_formal", "adaptive_control_formal"):
            output = Path(row[prefix + "_output"])
            meta = output / "launch_meta.json"
            metric = Path(row[prefix + "_metrics"])
            if not meta.is_file() or read(meta).get("status") != "PASS":
                raise RuntimeError(f"{scene}: frozen {prefix} status is not PASS")
            if not metric.is_file() or read(metric).get("iteration") != 5000:
                raise RuntimeError(f"{scene}: frozen {prefix} metrics missing/invalid")
        verified[scene] = {
            "scene": scene, "K": k, "decision": str(decision_path),
            "decision_sha256": sha(decision_path),
            "plan_sha256": sha(plan_path),
            "source_checkpoint_sha256": decision["source_checkpoint_sha256"],
            "frozen_rd_metrics": row["adaptive_formal_metrics"],
            "frozen_control_metrics": row["adaptive_control_formal_metrics"],
        }
    return m, verified


def check_code(fp: dict) -> None:
    for name, digest in fp.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Code/config changed during RD-D3 batch: {name}")


def launch(scene: str, arm: str, stage: str, destination: Path, decision: str,
           audit: Path, *, execute: bool, smoke_output: Path | None = None) -> dict:
    command = [
        sys.executable, "-u", "-m", "tools.run_i2_scene",
        "--scene", scene, "--stage", stage, "--arm", arm,
        "--adaptive-decision", decision, "--output-dir", str(destination),
    ]
    if smoke_output is not None:
        command.extend(["--smoke-output-dir", str(smoke_output)])
    if execute:
        command.append("--execute")
    logfile = audit / f"{scene}_{arm}_{stage}{'_preflight' if not execute else ''}.log"
    if logfile.exists():
        raise RuntimeError(f"Cannot overwrite previous stage log: {logfile}")
    print(f"[{scene}] {arm}/{stage} {'START' if execute else 'PREFLIGHT'}", flush=True)
    start = time.monotonic()
    with logfile.open("xb") as handle:
        handle.write(("Command: " + subprocess.list2cmdline(command) + "\n\n").encode("utf-8"))
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(10)
            if execute:
                print(f"  [{scene}] {arm}/{stage}: running {int(time.monotonic()-start)}s", flush=True)
        code = process.returncode
    raw = logfile.read_text(encoding="utf-8", errors="replace")
    if code != 0:
        print("\n".join(raw.splitlines()[-45:]), flush=True)
        raise RuntimeError(f"{scene} {arm}/{stage} failed (code {code}); log={logfile}")
    if not execute:
        # Text may start with implementation stdout; pick last JSON object using decoder.
        offset = raw.find("\n{")
        if offset < 0:
            offset = raw.find("{") - 1
        if offset < -1:
            raise RuntimeError(f"{scene} {arm}: no dry-run JSON")
        payload = json.loads(raw[offset + 1:])
        if payload.get("status") != "PREFLIGHT_PASS" or payload.get("executed") is not False:
            raise RuntimeError(f"{scene} {arm}: CLI dry-run was not a clean PREFLIGHT_PASS")
        if payload.get("requested_k") is None:
            raise RuntimeError(f"{scene} {arm}: no K in dry-run")
        return payload
    meta_path = destination / "launch_meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"{scene} {arm}: launch_meta.json not found")
    meta = read(meta_path)
    if meta.get("status") != "PASS" or meta.get("scene") != scene:
        raise RuntimeError(f"{scene} {arm}: launch metadata not PASS")
    if meta.get("mechanism_arm") != arm or meta.get("diagnostic_source_batch") != BASE_TAG:
        raise RuntimeError(f"{scene} {arm}: missing RD-D3 immutable arm/batch identity")
    if not all((destination / fn).is_file() for fn in ("train.log", "evaluation.log", "active_event.json")):
        raise RuntimeError(f"{scene} {arm}: required train/evaluation/event logs missing")
    iteration = 1252 if stage == "rd-d3-smoke" else 5000
    metrics = destination / f"formal_metrics_{iteration}.json"
    if not metrics.is_file():
        raise RuntimeError(f"{scene} {arm}: metrics not found: {metrics}")
    result = read(metrics)
    if result.get("iteration") != iteration or result.get("checkpoint_audit", {}).get("pass") is not True:
        raise RuntimeError(f"{scene} {arm}: metrics/checkpoint audit failed")
    event = read(destination / "active_event.json")
    before, after = event.get("gaussian_before"), event.get("gaussian_after")
    k = meta.get("requested_k")
    expected_release = (arm == "release")
    if not all(type(x) is int and x > 0 for x in (before, after, k)):
        raise RuntimeError(f"{scene} {arm}: invalid event counts")
    if after != before + (-k if expected_release else k):
        raise RuntimeError(f"{scene} {arm}: intervention count mismatch")
    if event.get("released_units") != (k if expected_release else 0) or event.get("received_units") != (0 if expected_release else k):
        raise RuntimeError(f"{scene} {arm}: non-isolated mechanism intervention")
    if event.get("alignment", {}).get("pass") is not True:
        raise RuntimeError(f"{scene} {arm}: Gaussian/optimizer alignment failed")
    print(f"[{scene}] {arm}/{stage}: PASS ({time.monotonic()-start:.1f}s)", flush=True)
    return {"output": str(destination), "metrics": str(metrics), "meta": str(meta_path),
            "active_event": str(destination / "active_event.json"), "status": "PASS"}


def write_comparison(manifest: dict, audit: Path) -> None:
    lines = ["scene,arm,PSNR,SSIM,LPIPS,L1,delta_psnr_vs_control,delta_ssim_vs_control,delta_lpips_vs_control"]
    for scene in SCENES:
        frozen = manifest["scenes"][scene]
        control = read(Path(frozen["frozen_control_metrics"]))["test"]["aggregate"]
        for arm, path in (
            ("control", frozen["frozen_control_metrics"]),
            ("rd_v1", frozen["frozen_rd_metrics"]),
            ("release", frozen["arms"]["release"]["formal"]["metrics"]),
            ("birth_only", frozen["arms"]["birth_only"]["formal"]["metrics"]),
        ):
            metrics = read(Path(path))["test"]["aggregate"]
            lines.append(",".join([scene, arm] + [f"{metrics[n]:.9f}" for n in ("PSNR", "SSIM", "LPIPS", "L1")] +
                                  [f"{metrics[n] - control[n]:+.9f}" for n in ("PSNR", "SSIM", "LPIPS")]))
    (audit / "rd_d3_four_arm_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bundle(manifest: dict, audit: Path) -> Path:
    destination = audit / "RD_D3_Comparison_Evidence.zip"
    if destination.exists():
        raise RuntimeError(f"Diagnostic evidence archive already exists: {destination}")
    entries = [
        ("rd_d3_batch_manifest.json", audit / "rd_d3_batch_manifest.json"),
        ("rd_d3_four_arm_comparison.csv", audit / "rd_d3_four_arm_comparison.csv"),
    ]
    for scene in SCENES:
        row = manifest["scenes"][scene]
        entries += [
            (f"{scene}/control_metrics.json", Path(row["frozen_control_metrics"])),
            (f"{scene}/rd_v1_metrics.json", Path(row["frozen_rd_metrics"])),
            (f"{scene}/decision.json", Path(row["decision"])),
        ]
        for arm in ARMS:
            formal = row["arms"][arm]["formal"]
            entries.extend([
                (f"{scene}/{arm}/metrics.json", Path(formal["metrics"])),
                (f"{scene}/{arm}/launch_meta.json", Path(formal["meta"])),
                (f"{scene}/{arm}/active_event.json", Path(formal["active_event"])),
            ])
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as out:
        for name, path in entries:
            if not path.is_file():
                raise RuntimeError(f"Evidence artifact missing: {path}")
            out.write(path, name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually launch eight new GPU runs: four smoke and four formal")
    parser.add_argument("--resume", help="Resume one incomplete RD-D3 batch without overwriting PASS outputs")
    args = parser.parse_args()
    if not args.execute:
        print("RD-D3 DRY MODE: no GPU launched. Add --execute after Codex CPU tests.")
        print("Frozen baseline:", BASE)
        print("Diagnostic scenes:", SCENES, "arms:", ARMS)
        return 0

    frozen, verified = check_original()
    batch = args.resume or ("RD_D3_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    audit = AUDIT_BASE / batch
    mp = audit / "rd_d3_batch_manifest.json"
    if args.resume:
        if not mp.is_file():
            raise RuntimeError("Resume manifest missing: " + str(mp))
        manifest = read(mp)
        if manifest.get("batch_id") != batch:
            raise RuntimeError("RD-D3 resume identity mismatch")
    else:
        audit.mkdir(parents=True, exist_ok=False)
        fps = {x: sha(ROOT / x) for x in CODE_PATHS if (ROOT / x).is_file()}
        if len(fps) != len(CODE_PATHS):
            raise RuntimeError("RD-D3 code/config input missing; Codex integration incomplete")
        manifest = {
            "format": "rd_d3_mechanism_batch_v1", "batch_id": batch,
            "status": "RUNNING", "source_batch": BASE_TAG,
            "source_batch_manifest_sha256_at_start": sha(BASE / "batch_manifest.json"),
            "fingerprints": fps,
            "scenes": {scene: {**verified[scene], "arms": {arm: {} for arm in ARMS}} for scene in SCENES},
        }
        save(mp, manifest)
    if sha(BASE / "batch_manifest.json") != manifest["source_batch_manifest_sha256_at_start"]:
        raise RuntimeError("Original RD-v1 batch manifest mutated")
    check_code(manifest["fingerprints"])

    def persist() -> None:
        save(mp, manifest)

    try:
        for phase, it in (("smoke", 1252), ("formal", 5000)):
            print(f"\n=== RD-D3 {phase.upper()}: Fortress/T-Rex Release-only and Birth-only ===", flush=True)
            for scene in SCENES:
                row = manifest["scenes"][scene]
                for arm in ARMS:
                    stage = f"rd-d3-{phase}"
                    suffix = "ReleaseOnly" if arm == "release" else "BirthOnly"
                    dest = ROOT / "output" / f"i2_{scene}_{batch}_{suffix}_{phase.capitalize()}_{it}"
                    entry = row["arms"][arm]
                    if entry.get(phase, {}).get("status") == "PASS":
                        old = entry[phase]
                        if Path(old["output"]) != dest or not Path(old["metrics"]).is_file() or read(Path(old["meta"])).get("status") != "PASS":
                            raise RuntimeError(f"{scene}/{arm}/{phase}: invalid previously completed stage")
                        print(f"[{scene}] {arm}/{phase}: REUSED PASS", flush=True)
                        continue
                    if dest.exists():
                        raise RuntimeError(f"{scene}/{arm}/{phase}: output exists; do not overwrite: {dest}")
                    check_code(manifest["fingerprints"])
                    smoke = None
                    if phase == "formal":
                        smoke = Path(entry["smoke"]["output"])
                    dry = launch(scene, arm, stage, dest, row["decision"], audit,
                                 execute=False, smoke_output=smoke)
                    if dry.get("requested_k") != row["K"]:
                        raise RuntimeError(f"{scene}/{arm}: diagnostic K differs from frozen RD-v1 K")
                    record = launch(scene, arm, stage, dest, row["decision"], audit,
                                    execute=True, smoke_output=smoke)
                    meta = read(Path(record["meta"]))
                    if meta.get("adaptive_decision_sha256") != row["decision_sha256"] or meta.get("plan_sha256") != row["plan_sha256"]:
                        raise RuntimeError(f"{scene}/{arm}: Decision or Plan binding mismatch")
                    if meta.get("source_checkpoint_sha256") != row["source_checkpoint_sha256"]:
                        raise RuntimeError(f"{scene}/{arm}: CommonSource binding mismatch")
                    entry[phase] = record
                    persist()
        write_comparison(manifest, audit)
        manifest["status"] = "PASS"
        persist()
        bundle = write_bundle(manifest, audit)
    except BaseException as exc:
        manifest["status"] = "NO_GO"
        manifest["error"] = repr(exc)
        persist()
        print(f"STOPPED; no overwrite: {exc}", flush=True)
        print("Resume with:", sys.executable, "-u -m tools.run_rd_d3_two_scene --execute --resume", batch)
        return 2
    print("\nRD-D3 MECHANISM BATCH: PASS")
    print("Comparison:", audit / "rd_d3_four_arm_comparison.csv")
    print("Manifest:", mp)
    print("BUNDLE:", bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
