#!/usr/bin/env python3
"""RD-D2: CPU-only post-training UID survival and Donor cohort selection audit.

Inputs are the *original* RD-v1 Decision, complete Donor audit, training event,
and RD/paired-Control iteration-5000 checkpoints.  It never starts a GPU, edits
models, reruns training, or changes any existing experiment evidence.

Run after placing in tools/:
    python -u -m tools.audit_rd_d2_checkpoint_utility \
        --batch RDv1_20261016_153009_622830
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

SCENES = ("horns", "fern", "flower", "fortress", "orchids", "room", "leaves", "trex")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Not a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_ref(ref: dict, label: str) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str):
        raise RuntimeError(f"Missing hash reference: {label}")
    path = Path(ref["path"])
    if not path.is_file() or sha256(path).lower() != ref["sha256"].lower():
        raise RuntimeError(f"Reference missing or SHA-256 mismatch: {label}: {path}")
    return path


def read_uid_state(path: Path):
    """Read a trusted experiment's 13-slot checkpoint, return UID sets + optional real-view coverage.

    This deserializes the user's own trusted training checkpoints, using CPU only.
    """
    import torch

    if not path.is_file():
        raise RuntimeError(f"Formal checkpoint missing: {path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise RuntimeError(f"Unexpected checkpoint envelope: {path}")
    state, iteration = payload
    if iteration != 5000 or not isinstance(state, (tuple, list)) or len(state) != 13:
        raise RuntimeError(f"Expected 13-slot iteration-5000 checkpoint: {path}")
    ext = state[12]
    if not isinstance(ext, dict) or "i2_lifecycle" not in ext:
        raise RuntimeError(f"Missing A0 UID lifecycle extension: {path}")
    lifecycle = ext["i2_lifecycle"]
    uid_tensor = lifecycle["uid"]
    if not torch.is_tensor(uid_tensor) or uid_tensor.ndim != 1:
        raise RuntimeError(f"Malformed UID tensor: {path}")
    uids = [int(x) for x in uid_tensor.tolist()]
    result = set(uids)
    if len(result) != len(uids):
        raise RuntimeError(f"Duplicate live UID in checkpoint: {path}")
    if len(result) != int(state[1].shape[0]):
        raise RuntimeError(f"Checkpoint UID / Gaussian row mismatch: {path}")
    obs_by_uid = {}
    obs = ext.get("i2_observation")
    if isinstance(obs, dict):
        obs_uids = obs.get("uid")
        opportunities = obs.get("real_view_opportunities")
        hits = obs.get("screen_supported_hits")
        if all(torch.is_tensor(v) and v.ndim == 1 and len(v) == len(uids)
               for v in (obs_uids, opportunities, hits)):
            if not torch.equal(obs_uids.to(dtype=uid_tensor.dtype), uid_tensor):
                raise RuntimeError(f"Observation UID ordering differs: {path}")
            for i, uid in enumerate(uids):
                opportunity = int(opportunities[i])
                hit = int(hits[i])
                if hit < 0 or opportunity < hit:
                    raise RuntimeError(f"Observation counts invalid: {path}")
                obs_by_uid[uid] = {"real_view_opportunities": opportunity,
                                   "screen_supported_hits": hit}
    del payload, state, ext, lifecycle, uid_tensor
    gc.collect()
    return result, obs_by_uid


def numeric(rows, key):
    values = [float(row[key]) for row in rows
              if type(row.get(key)) in (int, float)]
    if not values:
        return {"n": 0, "median": None, "mean": None}
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values)}


def percent(numerator, denominator):
    return round(100.0 * numerator / denominator, 2) if denominator else None


def inspect_scene(root: Path, batch: str, manifest: dict, scene: str) -> dict:
    row = manifest["scenes"][scene]
    if not all(row.get(key) == "PASS" for key in (
        "preflight", "prepare", "adaptive_smoke", "adaptive_formal",
        "adaptive_control_formal"
    )):
        raise RuntimeError(f"{scene}: original batch stages not all PASS")
    decision_path = Path(row["decision"])
    if not decision_path.is_file() or sha256(decision_path) != row["decision_sha256"]:
        raise RuntimeError(f"{scene}: Decision missing/hash mismatch")
    decision = load(decision_path)
    if decision.get("policy_version") != "RD-v1":
        raise RuntimeError(f"{scene}: not an RD-v1 decision")
    r, d, k = [int(decision[key]) for key in ("receiver_capacity", "donor_capacity", "selected_k")]
    if min(r, d) != k or k != int(row["K"]):
        raise RuntimeError(f"{scene}: K is inconsistent with min(R,D)")
    donor_path = require_ref(decision["evidence"]["donor"], f"{scene}: Donor audit")
    donor = load(donor_path)
    ranked = donor.get("ranked_eligible")
    if donor.get("coverage_complete") is not True or not isinstance(ranked, list) or len(ranked) != d:
        raise RuntimeError(f"{scene}: incomplete Donor evidence")
    selected, unselected = ranked[:k], ranked[k:]
    select_uids = [int(x["uid"]) for x in selected]
    remaining_uids = [int(x["uid"]) for x in unselected]
    if len(set(select_uids + remaining_uids)) != d:
        raise RuntimeError(f"{scene}: duplicate Donor UID")
    rd_out = Path(row["adaptive_formal_output"])
    control_out = Path(row["adaptive_control_formal_output"])
    for label, path in (("RD", rd_out), ("Control", control_out)):
        meta_path = path / "launch_meta.json"
        metrics_path = path / "formal_metrics_5000.json"
        if not meta_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"{scene}: {label} formal metadata/metrics missing")
        meta = load(meta_path)
        metrics = load(metrics_path)
        if meta.get("status") != "PASS" or metrics.get("iteration") != 5000 or metrics.get("checkpoint_audit", {}).get("pass") is not True:
            raise RuntimeError(f"{scene}: {label} final metadata/metrics not PASS")
    event = load(rd_out / "active_event.json")
    if event.get("format") != "i2_active_event_rd_v1" or event.get("budget_conserved") is not True:
        raise RuntimeError(f"{scene}: runtime intervention event invalid")
    born = [int(uid) for uid in event.get("born_uids", [])]
    released = [int(uid) for uid in event.get("released_uids", [])]
    if len(born) != k or len(set(born)) != k or released != select_uids:
        raise RuntimeError(f"{scene}: active_event does not match selected donors / newborn count")
    rd_ckpt = rd_out / "chkpnt5000.pth"
    ctrl_ckpt = control_out / "chkpnt5000.pth"
    print(f"[{scene}] reading RD-v1 checkpoint: {rd_ckpt.name}", flush=True)
    rd_alive, rd_observations = read_uid_state(rd_ckpt)
    print(f"[{scene}] reading paired Control checkpoint: {ctrl_ckpt.name}", flush=True)
    control_alive, _ = read_uid_state(ctrl_ckpt)
    newborn_alive = [uid for uid in born if uid in rd_alive]
    donor_alive_control = [uid for uid in select_uids if uid in control_alive]
    other_alive_control = [uid for uid in remaining_uids if uid in control_alive]
    newborn_obs = [rd_observations[uid] for uid in newborn_alive if uid in rd_observations]
    observed = [v for v in newborn_obs if v["real_view_opportunities"] > 0]
    newborn_screen_rate = [v["screen_supported_hits"] / v["real_view_opportunities"] for v in observed]
    result = {
        "scene": scene, "batch_id": batch,
        "K": k, "R": r, "D": d,
        "selected_donor_count": len(selected),
        "unselected_eligible_donor_count": len(unselected),
        "selected_donor_age": numeric(selected, "age"),
        "unselected_donor_age": numeric(unselected, "age"),
        "selected_donor_screen_support_rate": numeric(selected, "screen_support_rate"),
        "unselected_donor_screen_support_rate": numeric(unselected, "screen_support_rate"),
        "selected_donor_birth_type": dict(Counter(str(x.get("birth_type")) for x in selected)),
        "unselected_donor_birth_type": dict(Counter(str(x.get("birth_type")) for x in unselected)),
        "selected_donor_age_distribution": dict(Counter(str(x.get("age")) for x in selected)),
        "unselected_donor_age_distribution": dict(Counter(str(x.get("age")) for x in unselected)),
        "rd_newborn_alive_5000": len(newborn_alive),
        "rd_newborn_survival_pct": percent(len(newborn_alive), k),
        "control_selected_donor_alive_5000": len(donor_alive_control),
        "control_selected_donor_survival_pct": percent(len(donor_alive_control), k),
        "control_unselected_eligible_alive_5000": len(other_alive_control),
        "control_unselected_eligible_survival_pct": percent(len(other_alive_control), len(unselected)),
        "rd_newborn_observed_alive": len(newborn_obs),
        "rd_newborn_opportunities_gt_zero": len(observed),
        "rd_newborn_screen_support_rate_5000": numeric(
            [{"rate": x} for x in newborn_screen_rate], "rate"
        ),
        "rd_gaussian_count_5000": len(rd_alive),
        "control_gaussian_count_5000": len(control_alive),
        "rd_checkpoint": str(rd_ckpt), "control_checkpoint": str(ctrl_ckpt),
        "donor_audit_sha256": sha256(donor_path),
        "decision_sha256": sha256(decision_path),
        "limitations": [
            "UID survival is a lifecycle outcome, not a causal future-utility label.",
            "RD and Control have changed training trajectories; survival comparisons are descriptive.",
            "Only selected Donors have the exact intervention; unselected eligible Donors are a differently ranked cohort.",
        ],
    }
    print(
        f"[{scene}] K={k} selected-Donor Control alive={len(donor_alive_control)}/{k} "
        f"({result['control_selected_donor_survival_pct']}%) "
        f"RD newborn alive={len(newborn_alive)}/{k} "
        f"({result['rd_newborn_survival_pct']}%)",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", default="RDv1_20261016_153009_622830")
    args = parser.parse_args()
    root = args.root.resolve()
    batch_dir = root / "reports" / "i2_audit" / args.batch
    manifest_path = batch_dir / "batch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing batch manifest: {manifest_path}")
    manifest = load(manifest_path)
    if manifest.get("batch_id") != args.batch or manifest.get("status") != "PASS":
        raise RuntimeError("Batch identity or PASS status invalid")
    results = []
    for scene in SCENES:
        results.append(inspect_scene(root, args.batch, manifest, scene))
    # Write only after all eight have been inspected successfully.
    out_dir = batch_dir / ("RD_D2_Checkpoint_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out_dir.mkdir(parents=False, exist_ok=False)
    output_json = out_dir / "rd_d2_summary.json"
    output_json.write_text(json.dumps({
        "format": "rd_d2_checkpoint_utility_v1", "source_batch": args.batch,
        "batch_manifest_sha256": sha256(manifest_path),
        "scenes": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_csv = out_dir / "rd_d2_summary.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene", "K", "R", "D", "rd_newborn_alive_5000", "rd_newborn_survival_pct",
                  "control_selected_donor_alive_5000", "control_selected_donor_survival_pct",
                  "control_unselected_eligible_alive_5000", "control_unselected_eligible_survival_pct",
                  "rd_gaussian_count_5000", "control_gaussian_count_5000"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    bundle = out_dir / "RD_D2_Checkpoint_Utility.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_json, output_json.name)
        archive.write(output_csv, output_csv.name)
    print("\nRD-D2 CPU CHECKPOINT AUDIT: PASS")
    print("OUTPUT:", out_dir)
    print("BUNDLE:", bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
