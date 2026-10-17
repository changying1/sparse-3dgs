#!/usr/bin/env python3
"""Run the post-patch I2 RD-v1 eight-scene protocol. Windows/local GPU required.

Requires Codex's unified RD-v1 CLI contract to be implemented first:
- --stage adaptive-prepare --policy-config configs/i2_adaptive_k_rd_v1.json
- --stage adaptive-smoke / adaptive-formal --adaptive-decision ...
- --stage adaptive-control-formal --adaptive-decision ... (paired Control)
- adaptive route chosen automatically from the decision
No fixed K, Plan, or legacy init gate passed by this runner.

This runner does not modify any source config and never overwrites an existing run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("horns", "fern", "flower", "fortress", "orchids", "room", "leaves", "trex")
POLICY_PATH = ROOT / "configs/i2_adaptive_k_rd_v1.json"
AUDIT_BASE = ROOT / "reports/i2_audit"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(stage: str, scene: str, args: list[str], audit: Path, *, quiet: bool = False) -> str:
    cmd = [sys.executable, "-u", str(ROOT / "tools/run_i2_scene.py"), "--scene", scene, "--stage", stage, *args]
    logfile = audit / f"{scene}_{stage}.log"
    print(f"\n[{scene}] {stage}: START", flush=True)
    start = time.monotonic()
    with logfile.open("wb") as log:
        log.write(("Command: " + subprocess.list2cmdline(cmd) + "\n\n").encode("utf-8"))
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while True:
            code = p.poll()
            if code is not None:
                break
            time.sleep(20)
            print(f"  [{scene}] {stage}: running {int(time.monotonic()-start)}s; log={logfile.name}", flush=True)
    output = logfile.read_text(encoding="utf-8", errors="replace")
    if code != 0:
        print("\n".join(output.splitlines()[-45:]), flush=True)
        raise RuntimeError(f"{scene} {stage}: exit={code}, log={logfile}")
    print(f"[{scene}] {stage}: PASS ({time.monotonic()-start:.1f}s)", flush=True)
    if not quiet:
        lines = [x for x in output.splitlines() if x.strip()]
        for line in lines[-4:]:
            print("  " + line[:320])
    return output


def assert_methods_frozen(fingerprints: dict[str, str]) -> None:
    for filename, expected in fingerprints.items():
        path = ROOT / filename
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"A frozen method/config file changed while running: {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run eight-scene GPU evidence, smoke, formal and paired Control")
    parser.add_argument("--resume", help="Resume same batch id, e.g. RDv1_20261016_160000; never overwrite PASS stages")
    parser.add_argument("--skip-controls", action="store_true", help="Run only RD-v1 arms (not recommended for final comparison)")
    ns = parser.parse_args()
    if not ns.execute:
        print("Dry safety check: add --execute to start the eight-scene protocol. No GPU run launched.")
        print("Required new policy:", POLICY_PATH)
        print("Scenes:", ", ".join(SCENES))
        return 0
    if not POLICY_PATH.is_file():
        raise RuntimeError("New RD-v1 policy not installed. Run Codex migration first; old R0-v1 is forbidden.")
    policy = read_json(POLICY_PATH)
    if policy.get("version") != "RD-v1" or "candidate_budget_cap" in policy:
        raise RuntimeError("Policy is not the new unbounded RD-v1; refusing to run GPU.")

    tag = ns.resume or ("RDv1_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    audit = AUDIT_BASE / tag
    if ns.resume:
        if not audit.is_dir():
            raise RuntimeError("Requested resume batch not found: " + str(audit))
    else:
        audit.mkdir(parents=True, exist_ok=False)
    manifest_path = audit / "batch_manifest.json"
    if ns.resume:
        manifest = read_json(manifest_path)
        if manifest.get("batch_id") != tag:
            raise RuntimeError("Batch identifier mismatch")
        fingerprints = manifest["fingerprints"]
    else:
        from utils.i2_scene_pipeline import CODE_FINGERPRINT_FILES
        paths = set(CODE_FINGERPRINT_FILES) | {
            "configs/i2_adaptive_k_rd_v1.json", "configs/i2_method_v2.json",
            "tools/run_i2_rd_8scenes.py", "tools/run_i2_scene.py",
            *(f"configs/i2_scenes/{s}.json" for s in SCENES),
        }
        fingerprints = {}
        for filename in sorted(paths):
            path = ROOT / filename
            if not path.is_file():
                raise RuntimeError("Code fingerprint input missing: " + filename)
            fingerprints[filename] = digest(path)
        manifest = {"batch_id": tag, "protocol": "RD-v1", "status": "RUNNING", "fingerprints": fingerprints, "scenes": {}}
        dump(manifest_path, manifest)
    assert_methods_frozen(fingerprints)

    def persist() -> None:
        dump(manifest_path, manifest)

    def row_for(scene: str) -> dict:
        cfg = read_json(ROOT / "configs/i2_scenes" / f"{scene}.json")
        namespace = cfg["output_namespace"]
        item = manifest["scenes"].setdefault(scene, {"scene": scene, "namespace": namespace})
        return item

    def ensure_preflight(scene: str, row: dict) -> None:
        if row.get("preflight") == "PASS":
            return
        raw = run("preflight", scene, [], audit, quiet=True)
        data = json.loads(raw.split("\n\n", 1)[1] if raw.startswith("Command:") else raw)
        if data.get("status") != "GO" or data.get("source_status") != "SOURCE_REUSABLE":
            raise RuntimeError(f"{scene} common source not ready: {data.get('status')} / {data.get('blockers')}")
        row["preflight"] = "PASS"
        persist()

    def ensure_decision(scene: str, row: dict) -> None:
        namespace = row["namespace"]
        output = ROOT / "reports/i2_scene_runs" / namespace / tag
        decision_path = output / "decision/adaptive_k_decision.json"
        if row.get("prepare") != "PASS":
            if output.exists():
                raise RuntimeError(f"{scene} incomplete evidence directory already exists: {output}; inspect before resume")
            assert_methods_frozen(fingerprints)
            run("adaptive-prepare", scene, ["--policy-config", str(POLICY_PATH), "--output-dir", str(output), "--execute"], audit)
        if not decision_path.is_file():
            raise RuntimeError(f"{scene}: no RD-v1 Decision at {decision_path}")
        decision = read_json(decision_path)
        if decision.get("scene") != scene or decision.get("policy_version") != "RD-v1" or decision.get("uses_test_views") is not False:
            raise RuntimeError(f"{scene}: wrong/unsafe adaptive Decision identity")
        r = decision.get("receiver_capacity")
        d = decision.get("donor_capacity")
        k = decision.get("selected_k")
        g = (decision.get("receiver_audit") or {}).get("geometry_nms_capacity_g")
        if any(type(x) is not int or x < 0 for x in (r, d, k)) or k != min(r, d):
            raise RuntimeError(f"{scene}: budget violates K=min(R,D): R={r} D={d} K={k}")
        if r > 0 and d > 0 and "adaptive_plan" not in decision:
            raise RuntimeError(f"{scene}: positive K has no Plan")
        if k == 0 and "adaptive_plan" in decision:
            raise RuntimeError(f"{scene}: K=0 must not have Plan")
        row.update(prepare="PASS", decision=str(decision_path), decision_sha256=digest(decision_path), G=g, R=r, D=d, K=k)
        persist()
        print(f"[{scene}] RD-v1 K=min({r},{d})={k}", flush=True)

    def ensure_run(scene: str, row: dict, stage: str, output: Path, extra: list[str], *, decision_bound: bool = True) -> None:
        label = stage.replace("-", "_")
        meta_path = output / "launch_meta.json"
        if row.get(label) == "PASS":
            if not meta_path.is_file() or read_json(meta_path).get("status") != "PASS":
                raise RuntimeError(f"{scene}: previously recorded {stage}=PASS but metadata missing/failed")
            return
        if output.exists():
            raise RuntimeError(f"{scene}: output collision/potential partial run, do not overwrite: {output}")
        assert_methods_frozen(fingerprints)
        args = ["--output-dir", str(output), *extra, "--execute"]
        if decision_bound:
            decision = Path(row["decision"])
            if digest(decision) != row["decision_sha256"]:
                raise RuntimeError(f"{scene}: Decision mutated")
            args = ["--adaptive-decision", str(decision), *args]
        run(stage, scene, args, audit, quiet=True)
        meta = read_json(meta_path)
        if meta.get("status") != "PASS" or meta.get("scene") != scene:
            raise RuntimeError(f"{scene}: {stage} launch metadata invalid")
        if decision_bound and meta.get("adaptive_decision_sha256") != row["decision_sha256"]:
            raise RuntimeError(f"{scene}: {stage} not bound to RD-v1 Decision")
        if not (output / "train.log").is_file() or not (output / "evaluation.log").is_file():
            raise RuntimeError(f"{scene}: train/evaluation logs missing: {output}")
        metrics_path = output / f"formal_metrics_{1252 if 'smoke' in stage else 5000}.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"{scene}: formal metrics missing: {metrics_path}")
        if decision_bound and stage not in ("adaptive-control-formal",) and row["K"] > 0:
            if not (output / "active_event.json").is_file():
                raise RuntimeError(f"{scene}: K>0 active_event.json missing: {output}")
        row[label] = "PASS"
        row[label + "_output"] = str(output)
        row[label + "_metrics"] = str(metrics_path)
        persist()

    try:
        print("\n=== Phase A: validate all eight frozen 1201 shared sources ===", flush=True)
        for scene in SCENES:
            ensure_preflight(scene, row_for(scene))
        print("\n=== Phase B: prepare eight new RD-v1 decisions ===", flush=True)
        for scene in SCENES:
            ensure_decision(scene, row_for(scene))
        print("\n=== Phase C: eight RD-v1 smokes (to 1252) ===", flush=True)
        for scene in SCENES:
            row = row_for(scene)
            smoke = ROOT / "output" / f"{row['namespace']}_{tag}_RD_Smoke_1252"
            ensure_run(scene, row, "adaptive-smoke", smoke, [])
        print("\n=== Phase D: eight RD-v1 formals (to 5000) ===", flush=True)
        for scene in SCENES:
            row = row_for(scene)
            formal = ROOT / "output" / f"{row['namespace']}_{tag}_RD_Formal_5000"
            ensure_run(scene, row, "adaptive-formal", formal, ["--smoke-output-dir", row["adaptive_smoke_output"]])
        if not ns.skip_controls:
            print("\n=== Phase E: eight same-source paired Controls (to 5000) ===", flush=True)
            for scene in SCENES:
                row = row_for(scene)
                if row["K"] == 0:
                    row["paired_control_formal"] = "SAME_AS_RD_K0"
                    persist()
                    continue
                control = ROOT / "output" / f"{row['namespace']}_{tag}_Control_Formal_5000"
                ensure_run(scene, row, "adaptive-control-formal", control, [])
        manifest["status"] = "PASS" if not ns.skip_controls else "RD_ARMS_PASS_CONTROLS_SKIPPED"
    except BaseException as exc:
        manifest["status"] = "NO_GO"
        manifest["error"] = repr(exc)
        persist()
        print(f"\nSTOPPED (no rerun/overwrite): {exc}", flush=True)
        print("Resume with:", sys.executable, str(__file__), "--execute --resume", tag)
        return 2
    finally:
        persist()
    print("\n=== BATCH SUMMARY ===")
    print(f"{'Scene':10s} {'G':>6s} {'R':>6s} {'D':>8s} {'K':>6s} {'Smoke':>10s} {'Formal':>10s} {'Control':>20s}")
    for scene in SCENES:
        r = row_for(scene)
        print(f"{scene:10s} {str(r.get('G', '-')):>6s} {r['R']:6d} {r['D']:8d} {r['K']:6d} {r.get('adaptive_smoke','-'):>10s} {r.get('adaptive_formal','-'):>10s} {r.get('adaptive_control_formal',r.get('paired_control_formal','-')):>20s}")
    print("Batch:", audit)
    print("Status:", manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
