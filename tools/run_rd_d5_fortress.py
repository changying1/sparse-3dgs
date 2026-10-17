#!/usr/bin/env python3
"""RD-D5 Fortress same-code four-arm validation (READ-ONLY by default).

Reuse D3 Release/Birth; retrain only original RD-v1 and paired Control with
exactly the D3 science/evaluation code fingerprint. Never edits Decision/Plan.
Requires the existing run_i2_scene adaptive-* public CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D3_TAG = "RD_D3_20261017_104858_958268"
RD_TAG = "RDv1_20261016_153009_622830"
D3_DIR = ROOT / "reports/i2_audit" / D3_TAG
RD_DIR = ROOT / "reports/i2_audit" / RD_TAG
SCENE = "fortress"
PHASES = (
    ("rd_smoke", "adaptive-smoke", 1252),
    ("rd_formal", "adaptive-formal", 5000),
    ("control_formal", "adaptive-control-formal", 5000),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required evidence missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def update_manifest(path: Path, data: dict) -> None:
    # Only this new D5 manifest may be updated; no legacy files are touched.
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def ref_artifacts() -> dict:
    d3_path = D3_DIR / "rd_d3_batch_manifest.json"
    original_path = RD_DIR / "batch_manifest.json"
    d3, original = read(d3_path), read(original_path)
    if d3.get("batch_id") != D3_TAG or d3.get("status") != "PASS":
        raise RuntimeError("D3 is not a PASS frozen reference")
    if d3.get("source_batch") != RD_TAG or original.get("batch_id") != RD_TAG or original.get("status") != "PASS":
        raise RuntimeError("RD-v1 original batch mismatch / not PASS")
    if digest(original_path) != d3.get("source_batch_manifest_sha256_at_start"):
        raise RuntimeError("Original RD batch manifest differs from its D3 frozen hash")
    row = d3["scenes"][SCENE]
    orig_row = original["scenes"][SCENE]
    k = row["K"]
    if type(k) is not int or k <= 0 or k != orig_row["K"] or k != min(orig_row["R"], orig_row["D"]):
        raise RuntimeError("Frozen K=min(R,D) inconsistent")
    decision = Path(row["decision"])
    if digest(decision) != row["decision_sha256"] or decision.resolve() != Path(orig_row["decision"]).resolve():
        raise RuntimeError("Original RD-v1 Decision altered")
    value = read(decision)
    if value.get("policy_version") != "RD-v1" or value.get("selected_k") != k or value.get("uses_test_views") is not False:
        raise RuntimeError("Invalid RD-v1 Decision identity")
    plan_ref = value.get("adaptive_plan", {})
    plan = Path(plan_ref.get("path", ""))
    if not plan.is_file() or digest(plan) != row["plan_sha256"] or digest(plan) != plan_ref.get("sha256"):
        raise RuntimeError("Frozen original Plan missing or hash changed")
    if value.get("source_checkpoint_sha256") != row["source_checkpoint_sha256"]:
        raise RuntimeError("Source Checkpoint binding inconsistent")

    formal_metas = {}
    for arm in ("release", "birth_only"):
        group = row["arms"][arm]["formal"]
        if group.get("status") != "PASS":
            raise RuntimeError(f"D3 {arm} Formal is not PASS")
        meta = read(Path(group["meta"]))
        metric = read(Path(group["metrics"]))
        if meta.get("status") != "PASS" or meta.get("mechanism_arm") != arm or meta.get("diagnostic_source_batch") != RD_TAG:
            raise RuntimeError(f"D3 {arm} provenance mismatch")
        if metric.get("iteration") != 5000 or metric.get("checkpoint_audit", {}).get("pass") is not True:
            raise RuntimeError(f"D3 {arm} checkpoint/evaluation failed")
        if meta.get("requested_k") != k or meta.get("plan_sha256") != row["plan_sha256"]:
            raise RuntimeError(f"D3 {arm} frozen K/Plan differs")
        if meta.get("adaptive_decision_sha256") != row["decision_sha256"] or meta.get("source_checkpoint_sha256") != row["source_checkpoint_sha256"]:
            raise RuntimeError(f"D3 {arm} Decision/Source differs")
        if meta.get("random_seed") != 0:
            raise RuntimeError(f"D3 {arm} unexpected random seed")
        formal_metas[arm] = meta
    a, b = formal_metas["release"], formal_metas["birth_only"]
    if a["fingerprints"] != b["fingerprints"] or a["view_manifest_sha256"] != b["view_manifest_sha256"]:
        raise RuntimeError("D3 Release/Birth did not use exactly the same frozen code/input hashes")
    if a.get("adaptive_policy_version") != "RD-v1" or b.get("adaptive_policy_version") != "RD-v1":
        raise RuntimeError("D3 policy differs from RD-v1")
    if a.get("fingerprints", {}).get("code_sha256", {}).get("train.py") != (
        "a4e6fea2565760867133f12de31c4757d65c3151ccc28983a320cd65d9b05cc7"
    ):
        raise RuntimeError("D3 scientific-code reference identity not recognized")
    return {
        "d3_manifest_path": d3_path, "d3_manifest_sha256": digest(d3_path),
        "original_manifest_path": original_path, "original_manifest_sha256": digest(original_path),
        "K": k, "decision": decision, "plan": plan,
        "decision_sha256": row["decision_sha256"], "plan_sha256": row["plan_sha256"],
        "source_checkpoint_sha256": row["source_checkpoint_sha256"],
        "view_manifest_sha256": a["view_manifest_sha256"], "seed": a["random_seed"],
        "fp": a["fingerprints"],
        "d3_code_fp": d3["fingerprints"],
        "release_metrics": Path(row["arms"]["release"]["formal"]["metrics"]),
        "birth_metrics": Path(row["arms"]["birth_only"]["formal"]["metrics"]),
        "release_meta": Path(row["arms"]["release"]["formal"]["meta"]),
        "birth_meta": Path(row["arms"]["birth_only"]["formal"]["meta"]),
        "historical_rd_metrics": Path(row["frozen_rd_metrics"]),
        "historical_control_metrics": Path(row["frozen_control_metrics"]),
    }


def verify_fingerprint(ref: dict) -> None:
    fp = ref["fp"]
    targets = dict(fp["code_sha256"])
    # D3 runner fingerprint inventory can also include the D3 runner; do not
    # require that it be deployed for new D5 science, but check every other file.
    for name, expected in ref["d3_code_fp"].items():
        if name != "tools/run_rd_d3_two_scene.py":
            if name in targets and targets[name] != expected:
                raise RuntimeError(f"D3 manifest/meta disagree on {name}")
            targets[name] = expected
    for name, expected in sorted(targets.items()):
        candidate = ROOT / name
        if not candidate.is_file():
            raise RuntimeError(f"CODE_VERSION_NO_GO: missing {name}")
        actual = digest(candidate)
        if actual.lower() != str(expected).lower():
            raise RuntimeError(f"CODE_VERSION_NO_GO: {name}: D3={expected}, current={actual}")
    for path_key, sha_key in (("scene_config", "scene_config_sha256"), ("method_config", "method_config_sha256")):
        path = Path(fp[path_key])
        if not path.is_file() or digest(path) != fp[sha_key]:
            raise RuntimeError(f"CONFIG_VERSION_NO_GO: {path_key}")
    for f in (ref["decision"], ref["plan"], ref["d3_manifest_path"], ref["original_manifest_path"]):
        key = (
            "decision_sha256" if f == ref["decision"] else
            "plan_sha256" if f == ref["plan"] else
            "d3_manifest_sha256" if f == ref["d3_manifest_path"] else
            "original_manifest_sha256"
        )
        if digest(f) != ref[key]:
            raise RuntimeError(f"FROZEN_EVIDENCE_CHANGED: {f}")


def arguments(stage: str, output: Path, ref: dict, *, smoke: Path | None = None, execute: bool = False) -> list[str]:
    result = [
        sys.executable, "-u", "-m", "tools.run_i2_scene",
        "--scene", SCENE, "--stage", stage,
        "--adaptive-decision", str(ref["decision"]), "--output-dir", str(output),
    ]
    if smoke is not None:
        result.extend(["--smoke-output-dir", str(smoke)])
    if execute:
        result.append("--execute")
    return result


def parse_json_output(output: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char == "{":
            try:
                value, end = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and output[index + end:].strip() == "":
                return value
    raise RuntimeError(f"Could not parse CLI dry-run JSON: {output[-2000:]}")


def dry_check(phase: str, stage: str, destination: Path, ref: dict, *, smoke=None) -> dict:
    command = arguments(stage, destination, ref, smoke=smoke)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{phase}: dry-run rejected: {result.stdout[-1200:]}\n{result.stderr[-1800:]}")
    value = parse_json_output(result.stdout)
    if value.get("status") != "PREFLIGHT_PASS" or value.get("executed") is not False:
        raise RuntimeError(f"{phase}: dry-run status is not PREFLIGHT_PASS")
    if value.get("scene") != SCENE or value.get("requested_k") != ref["K"]:
        raise RuntimeError(f"{phase}: dry-run scene/K differs")
    if Path(value.get("output", "")).resolve() != destination.resolve():
        raise RuntimeError(f"{phase}: unexpected output destination")
    expected_arm = "control" if phase == "control_formal" else "receiver_v2"
    if value.get("arm") != expected_arm:
        raise RuntimeError(f"{phase}: unexpected arm {value.get('arm')}")
    if destination.exists():
        raise RuntimeError(f"{phase}: dry-run unexpectedly created training output")
    return value


def validate_finished(phase: str, folder: Path, ref: dict, iteration: int) -> dict:
    meta_path = folder / "launch_meta.json"
    metrics_path = folder / f"formal_metrics_{iteration}.json"
    meta, metrics = read(meta_path), read(metrics_path)
    arm = "control" if phase == "control_formal" else "receiver_v2"
    if meta.get("status") != "PASS" or meta.get("scene") != SCENE or meta.get("arm") != arm:
        raise RuntimeError(f"{phase}: launch_meta is not PASS for expected arm")
    if meta.get("stage") != ("smoke" if phase == "rd_smoke" else "formal"):
        raise RuntimeError(f"{phase}: incorrect stage")
    for label, expected in (
        ("requested_k", ref["K"]),
        ("source_checkpoint_sha256", ref["source_checkpoint_sha256"]),
        ("adaptive_decision_sha256", ref["decision_sha256"]),
        ("view_manifest_sha256", ref["view_manifest_sha256"]),
        ("random_seed", ref["seed"]),
    ):
        if meta.get(label) != expected:
            raise RuntimeError(f"{phase}: {label} differs from D3 reference")
    if meta.get("fingerprints") != ref["fp"]:
        raise RuntimeError(f"{phase}: full science/config fingerprints differ from D3")
    if phase == "control_formal":
        if meta.get("paired_control") is not True or meta.get("plan_sha256") is not None:
            raise RuntimeError("Control is not a clean same-source paired control")
    elif meta.get("plan_sha256") != ref["plan_sha256"]:
        raise RuntimeError(f"{phase}: Plan hash differs")
    if metrics.get("iteration") != iteration or metrics.get("checkpoint_audit", {}).get("pass") is not True:
        raise RuntimeError(f"{phase}: metrics/checkpoint audit missing or invalid")
    if not all((folder / p).is_file() for p in ("train.log", "evaluation.log")):
        raise RuntimeError(f"{phase}: full output-directory logs missing")
    if phase != "control_formal":
        event = read(folder / "active_event.json")
        if event.get("budget_conserved") is not True or event.get("alignment", {}).get("pass") is not True:
            raise RuntimeError(f"{phase}: RD-v1 intervention did not conserve and align Gaussians")
    return {"status": "PASS", "output": str(folder), "meta": str(meta_path),
            "metrics": str(metrics_path), "iteration": iteration,
            "metrics_sha256": digest(metrics_path)}


def run_logged(phase: str, cmd: list[str], logfile: Path) -> None:
    if logfile.exists():
        raise RuntimeError(f"Refusing to overwrite batch log: {logfile}")
    t0 = time.monotonic()
    with logfile.open("xb") as stream:
        stream.write(("Command: " + subprocess.list2cmdline(cmd) + "\n\n").encode("utf-8"))
        stream.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(10)
            print(f"  [{phase}] running {int(time.monotonic() - t0)}s; log={logfile.name}", flush=True)
    if p.returncode != 0:
        tail = logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(f"{phase}: child exit {p.returncode}; {logfile}\n" + "\n".join(tail))


def write_comparison(ref: dict, manifest: dict, audit: Path) -> tuple[Path, Path]:
    paths = {
        "Control": Path(manifest["phases"]["control_formal"]["metrics"]),
        "Release-only": ref["release_metrics"],
        "Birth-only": ref["birth_metrics"],
        "RD-v1": Path(manifest["phases"]["rd_formal"]["metrics"]),
    }
    metrics = {arm: read(path) for arm, path in paths.items()}
    baseline = metrics["Control"]["test"]["aggregate"]
    first_views = [(v["image_name"], v["camera_uid"]) for v in metrics["Control"]["test"]["per_view"]]
    if not first_views or len(set(first_views)) != len(first_views):
        raise RuntimeError("Control test views empty/duplicated")
    for arm, value in metrics.items():
        if value.get("iteration") != 5000 or value.get("checkpoint_audit", {}).get("pass") is not True:
            raise RuntimeError(f"{arm}: invalid final checkpoint")
        viewids = [(v["image_name"], v["camera_uid"]) for v in value["test"]["per_view"]]
        if viewids != first_views:
            raise RuntimeError(f"{arm}: test camera identity/order differs")
    measures = ("PSNR", "SSIM", "LPIPS", "L1")
    csv_path = audit / "rd_d5_fortress_four_arm_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["scene", "arm", *measures, "delta_psnr", "delta_ssim", "delta_lpips"])
        for arm, value in metrics.items():
            record = value["test"]["aggregate"]
            writer.writerow([SCENE, arm, *(f"{record[x]:.9f}" for x in measures),
                             *(f"{record[x] - baseline[x]:+.9f}" for x in measures[:3])])
    perview = audit / "rd_d5_fortress_per_view.csv"
    with perview.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "arm", *measures, "delta_psnr", "delta_ssim", "delta_lpips"])
        refviews = metrics["Control"]["test"]["per_view"]
        for arm, value in metrics.items():
            for base, view in zip(refviews, value["test"]["per_view"]):
                writer.writerow([view["image_name"], arm, *(f"{view[x]:.9f}" for x in measures),
                                 *(f"{view[x] - base[x]:+.9f}" for x in measures[:3])])
    psnr = {arm: val["test"]["aggregate"]["PSNR"] for arm, val in metrics.items()}
    interaction = psnr["RD-v1"] - psnr["Release-only"] - psnr["Birth-only"] + psnr["Control"]
    print("\nRD-D5 FORTRESS FOUR ARM, same code, test mean", flush=True)
    for arm in ("Control", "Release-only", "Birth-only", "RD-v1"):
        print(f"  {arm:13} PSNR={psnr[arm]:.6f}, delta={psnr[arm]-psnr['Control']:+.6f}")
    print(f"  PSNR descriptive interaction I={interaction:+.6f} dB")
    return csv_path, perview


def bundle(ref: dict, manifest_file: Path, audit: Path, metrics_csv: Path, perview_csv: Path) -> Path:
    target = audit / "RD_D5_Fortress_Comparison_Evidence.zip"
    files = [
        ("rd_d5_manifest.json", manifest_file),
        ("rd_d5_four_arm_metrics.csv", metrics_csv),
        ("rd_d5_per_view_metrics.csv", perview_csv),
        ("d3_reference/release_metrics.json", ref["release_metrics"]),
        ("d3_reference/birth_metrics.json", ref["birth_metrics"]),
        ("d3_reference/release_launch_meta.json", ref["release_meta"]),
        ("d3_reference/birth_launch_meta.json", ref["birth_meta"]),
        ("frozen/decision.json", ref["decision"]),
        ("historical/control_metrics.json", ref["historical_control_metrics"]),
        ("historical/rd_metrics.json", ref["historical_rd_metrics"]),
    ]
    for name, rec in read(manifest_file)["phases"].items():
        if rec.get("status") == "PASS":
            files.append((f"rd_d5/{name}_metrics.json", Path(rec["metrics"])))
            files.append((f"rd_d5/{name}_launch_meta.json", Path(rec["meta"])))
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in files:
            if not path.is_file():
                raise RuntimeError(f"Missing archive artifact: {path}")
            z.write(path, name)
    return target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="Run exactly one Smoke and two 5000 Formal training jobs")
    p.add_argument("--resume", help="Resume existing incomplete D5 batch, no output overwrite")
    args = p.parse_args(argv)
    reference = ref_artifacts()
    verify_fingerprint(reference)
    print(f"RD-D5 VERSION GATE: PASS | fortress K={reference['K']}; matching D3 science fingerprints")
    t = "RD_D5_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") if not args.resume else args.resume
    audit = ROOT / "reports/i2_audit" / t
    manifest_path = audit / "rd_d5_batch_manifest.json"
    if args.resume:
        m = read(manifest_path)
        if m.get("batch_id") != t or m.get("reference_d3_manifest_sha256") != reference["d3_manifest_sha256"]:
            raise RuntimeError("Resume manifest/batch D3 binding mismatch")
    elif args.execute:
        audit.mkdir(parents=True, exist_ok=False)
        m = {
            "format": "rd_d5_fortress_same_code_v1", "status": "RUNNING", "batch_id": t,
            "scene": SCENE, "K": reference["K"], "d3_tag": D3_TAG, "rd_tag": RD_TAG,
            "reference_d3_manifest_sha256": reference["d3_manifest_sha256"],
            "reference_original_manifest_sha256": reference["original_manifest_sha256"],
            "plan_sha256": reference["plan_sha256"], "decision_sha256": reference["decision_sha256"],
            "frozen_science_and_evaluation_fingerprints": reference["fp"],
            "phases": {},
        }
        write_new(manifest_path, m)
    else:
        m = {"phases": {}}
    for name, stage, iteration in PHASES:
        dirname = ("RD_Smoke_1252" if name == "rd_smoke" else
                   "RD_Formal_5000" if name == "rd_formal" else "Control_Formal_5000")
        dest = ROOT / "output" / f"i2_{SCENE}_{t}_{dirname}"
        if name in m["phases"] and m["phases"][name].get("status") == "PASS":
            valid = validate_finished(name, dest, reference, iteration)
            print(f"[{name}] REUSED PASS | {valid['output']}")
            continue
        if dest.exists():
            raise RuntimeError(f"Non-PASS destination exists; refusing overwrite: {dest}")
        verify_fingerprint(reference)
        smoke = (ROOT / "output" / f"i2_{SCENE}_{t}_RD_Smoke_1252") if name == "rd_formal" else None
        if not args.execute and name == "rd_formal" and not smoke.is_dir():
            # Formal stage deliberately requires a PASS smoke; do not fake one.
            print("[rd_formal] DEFERRED: no RD-D5 PASS Smoke yet; will gate before actual Formal")
            continue
        _ = dry_check(name, stage, dest, reference, smoke=smoke)
        print(f"[{name}] DRY GATE: PASS; arm={'control' if name=='control_formal' else 'receiver_v2'}")
        if not args.execute:
            continue
        print(f"[{name}] EXECUTE: START", flush=True)
        logfile = audit / f"{name}.launcher.log"
        cmd = arguments(stage, dest, reference, smoke=smoke, execute=True)
        try:
            run_logged(name, cmd, logfile)
            v = validate_finished(name, dest, reference, iteration)
            m["phases"][name] = v
            update_manifest(manifest_path, m)
            print(f"[{name}] PASS", flush=True)
        except Exception as exc:
            m["status"] = "NO_GO"
            m["error"] = str(exc)
            update_manifest(manifest_path, m)
            print(f"[{name}] NO_GO: {exc}", file=sys.stderr)
            print(f"Resume only after resolving blocker: {sys.executable} -m tools.run_rd_d5_fortress --execute --resume {t}")
            return 2
    if not args.execute:
        print("RD-D5 DRY MODE PASS: no GPU, no output or batch directory created")
        print("Run with --execute after CPU regressions pass")
        return 0
    a, b = write_comparison(reference, m, audit)
    m["status"] = "PASS"
    update_manifest(manifest_path, m)
    package = bundle(reference, manifest_path, audit, a, b)
    print("RD-D5 FORTRESS SAME-CODE BATCH: PASS")
    print("BUNDLE:", package)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("RD-D5 NO_GO:", e, file=sys.stderr)
        raise SystemExit(2)
