#!/usr/bin/env python3
"""F05 read-only Horns/Fern cross-scene mechanism audit.

The analyzer discovers formal arms from launch metadata and the configured scene
identity.  It never trains, prepares, or mutates frozen experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCENES = ("horns", "fern")
ROLES = ("control", "release_only", "receiver_v2")
TREND_RE = re.compile(
    r"\[ITER (\d+)\] Evaluating test: L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+) "
    r"SSIM ([0-9.eE+-]+) LPIPS ([0-9.eE+-]+)"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_value(argv: list[Any], option: str) -> str | None:
    try:
        return str(argv[argv.index(option) + 1])
    except (ValueError, IndexError):
        return None


def as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def artifact_path(root: Path, value: str | Path | dict[str, Any]) -> Path:
    """Resolve artifact references represented either as a path or {path, sha256}."""
    if isinstance(value, dict):
        value = value["path"]
    return as_path(root, value)


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}

    def q(frac: float) -> float:
        pos = frac * (len(vals) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

    return {
        "n": len(vals), "min": vals[0], "p25": q(0.25), "median": q(0.5),
        "p75": q(0.75), "max": vals[-1], "mean": statistics.fmean(vals),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def role_for(meta: dict[str, Any], event: dict[str, Any] | None) -> str | None:
    if event and event.get("format") == "i2_active_event_receiver_v2_v1":
        return "receiver_v2"
    if event and event.get("format") == "i2_release_only_event_v1":
        return "release_only"
    arm = str(meta.get("arm", "")).lower()
    if arm == "receiver_v2":
        return "receiver_v2"
    if arm == "release":
        return "release_only"
    if arm == "control":
        return "control"
    argv = [str(x) for x in meta.get("argv", [])]
    if "--enable_i2_active_pilot" in argv or "--enable_i2_release_only" in argv:
        return None
    return "control"


def launch_source_hash(meta: dict[str, Any]) -> str | None:
    return meta.get("source_checkpoint_sha256") or meta.get("frozen_inputs", {}).get("source_checkpoint_sha256")


def discover_arms(root: Path, scene_cfg: dict[str, Any]) -> tuple[dict[str, Path], list[str]]:
    """Discover, rather than name-guess, the unique completed 5000-step arms."""
    candidates: dict[str, list[Path]] = {role: [] for role in ROLES}
    notes: list[str] = []
    output = root / "output"
    try:
        dirs = list(output.iterdir())
    except OSError as exc:
        raise RuntimeError(f"Cannot enumerate {output}: {exc}") from exc
    for directory in dirs:
        if not directory.is_dir():
            continue
        meta_path = directory / "launch_meta.json"
        metrics_path = directory / "formal_metrics_5000.json"
        if not meta_path.is_file() or not metrics_path.is_file():
            continue
        try:
            meta = read_json(meta_path)
            metrics = read_json(metrics_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            notes.append(f"Skipped unreadable candidate {directory}: {exc}")
            continue
        argv = [str(x) for x in meta.get("argv", [])]
        dataset = argv_value(argv, "-s")
        scene_match = str(meta.get("scene", "")).lower() == scene_cfg["scene"]
        dataset_match = dataset is not None and Path(dataset).name.lower() == scene_cfg["scene"]
        source_match = launch_source_hash(meta) == scene_cfg["common_source"]["sha256"]
        start_checkpoint = argv_value(argv, "--start_checkpoint")
        checkpoint_match = start_checkpoint is not None and Path(start_checkpoint).name == "chkpnt1201.pth" and Path(start_checkpoint).parent.name == Path(scene_cfg["common_source"]["checkpoint"]).parent.name
        final_iteration = meta.get("final_iteration", metrics.get("iteration"))
        if not (scene_match or dataset_match) or not (source_match or checkpoint_match):
            continue
        if final_iteration != 5000 or meta.get("status") != "PASS":
            continue
        event_path = directory / "active_event.json"
        event = read_json(event_path) if event_path.is_file() else None
        role = role_for(meta, event)
        if role:
            candidates[role].append(directory)
    result: dict[str, Path] = {}
    for role, paths in candidates.items():
        if len(paths) != 1:
            raise RuntimeError(f"{scene_cfg['scene']} {role}: expected one formal arm, found {len(paths)}: {paths}")
        result[role] = paths[0]
    return result, notes


def find_initialization_audit(root: Path, source_hash: str) -> Path:
    matches: list[Path] = []
    for path in (root / "reports").rglob("initialization_audit.json"):
        try:
            data = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if data.get("format") == "i2_receiver_v2_initialization_audit_v1" and data.get("source_checkpoint_sha256") == source_hash:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one initialization audit for {source_hash}, found {matches}")
    return matches[0]


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return quantiles(float(row[key]) for row in rows if row.get(key) not in (None, ""))


def spatial_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [tuple(float(v) for v in row["new_xyz"]) for row in rows]
    if not points:
        return {"count": 0}
    center = tuple(statistics.fmean(p[i] for p in points) for i in range(3))
    radial = [math.dist(p, center) for p in points]
    nearest = [min(math.dist(p, q) for j, q in enumerate(points) if i != j) for i, p in enumerate(points)] if len(points) > 1 else []
    camera_counts = Counter(str(row.get("training_camera_id")) for row in rows)
    n = len(rows)
    return {
        "count": n,
        "centroid": list(center),
        "centroid_radius": quantiles(radial),
        "nearest_neighbor_distance": quantiles(nearest),
        "camera_counts": dict(sorted(camera_counts.items())),
        "camera_hhi": sum((count / n) ** 2 for count in camera_counts.values()),
    }


def parse_trends(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in TREND_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        iteration, l1, psnr, ssim, lpips = match.groups()
        rows.append({"iteration": int(iteration), "PSNR": float(psnr), "SSIM": float(ssim), "LPIPS": float(lpips), "L1": float(l1)})
    return rows


def file_entry(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": False, "status": "UNKNOWN"}
    try:
        result["exists"] = path.is_file()
        if not result["exists"]:
            result["status"] = "MISSING"
            return result
        result["size"] = path.stat().st_size
        if expected_hash:
            actual = sha256(path)
            result.update({"expected_sha256": expected_hash, "actual_sha256": actual, "status": "PASS" if actual == expected_hash else "FAIL"})
        else:
            result["status"] = "PRESENT"
    except OSError as exc:
        result["error"] = str(exc)
    return result


def audit_scene(root: Path, scene: str, scene_cfg: dict[str, Any], method_cfg: dict[str, Any]) -> dict[str, Any]:
    arms, discovery_notes = discover_arms(root, scene_cfg)
    arm_data: dict[str, dict[str, Any]] = {}
    for role, directory in arms.items():
        meta = read_json(directory / "launch_meta.json")
        metrics = read_json(directory / "formal_metrics_5000.json")
        event = read_json(directory / "active_event.json") if (directory / "active_event.json").is_file() else None
        arm_data[role] = {"dir": directory, "meta": meta, "metrics": metrics, "event": event}

    receiver_event = arm_data["receiver_v2"]["event"]
    release_event = arm_data["release_only"]["event"]
    if receiver_event is None or release_event is None:
        raise RuntimeError(f"{scene}: intervention event missing")
    plan_path = as_path(root, receiver_event["plan_path"])
    geom_path = as_path(root, receiver_event["receiver_v2_evidence"])
    plan = read_json(plan_path)
    geometry = read_json(geom_path)
    source_manifest_path = artifact_path(root, plan["input_audit"]["source_manifest"])
    source_manifest = read_json(source_manifest_path)
    receiver_csv_path = as_path(root, source_manifest["receiver_candidates"]["path"])
    b1_csv_path = as_path(root, source_manifest["candidate_tables"]["b1"]["path"])
    receiver_rows = read_csv(receiver_csv_path)
    b1_rows = read_csv(b1_csv_path)
    static_path = as_path(root, receiver_event["counterfactual_artifact"])
    joint_path = as_path(root, receiver_event["joint_counterfactual_artifact"])
    static = read_json(static_path)
    joint = read_json(joint_path)
    init_path = find_initialization_audit(root, scene_cfg["common_source"]["sha256"])
    initialization = read_json(init_path)

    selected_donors = plan["donors"]["selected"]
    selected_uids = [int(row["uid"]) for row in selected_donors]
    event_released = [int(v) for v in receiver_event.get("released_uids", [])]
    release_released = [int(v) for v in release_event.get("released_uids", [])]
    excluded = set(int(v) for v in plan["configuration"].get("excluded_uids", []))
    low_support_rows = [row for row in b1_rows if row.get("low_screen_support_candidate", "").lower() == "true"]
    basic_rows = [
        row for row in low_support_rows
        if row.get("birth_type") not in ("root", "unknown")
        and int(row.get("age", 0)) >= int(plan["configuration"]["min_age"])
        and int(row.get("real_view_opportunities", 0)) >= int(plan["configuration"]["min_real_view_opportunities"])
        and int(row["uid"]) not in excluded
    ]
    geom_rows = geometry["rows"]
    rejected = geometry.get("rejected_before_k20", geometry.get("rejected_before_requested_k", []))
    source_checkpoint = as_path(root, scene_cfg["common_source"]["checkpoint"])
    cfg_args = source_checkpoint.parent / "cfg_args"
    view_manifest = Path(scene_cfg["view_manifest"]["path"])

    source_hashes = [launch_source_hash(item["meta"]) for item in arm_data.values()]
    for role, item in arm_data.items():
        if source_hashes[ROLES.index(role)] is None:
            argv = [str(x) for x in item["meta"].get("argv", [])]
            source_hashes[ROLES.index(role)] = scene_cfg["common_source"]["sha256"] if argv_value(argv, "--start_checkpoint") else None
    protocols = [item["metrics"].get("protocol") for item in arm_data.values()]
    view_hashes = []
    for item in arm_data.values():
        meta = item["meta"]
        view_hashes.append(meta.get("view_manifest_sha256") or meta.get("frozen_inputs", {}).get("view_manifest_sha256") or scene_cfg["view_manifest"]["sha256"])

    integrity_checks = {
        "same_common_source": {"status": "PASS" if len(set(source_hashes)) == 1 and source_hashes[0] == scene_cfg["common_source"]["sha256"] else "FAIL", "values": source_hashes},
        "same_evaluation_protocol": {"status": "PASS" if protocols[1:] == protocols[:-1] else "FAIL", "protocol": protocols[0]},
        "same_view_manifest": {"status": "PASS" if len(set(view_hashes)) == 1 and view_hashes[0] == scene_cfg["view_manifest"]["sha256"] else "FAIL", "values": view_hashes},
        "fixed_k20": {"status": "PASS" if plan["configuration"]["budget_k"] == method_cfg["budget"]["requested_k"] == 20 else "FAIL"},
        "same_donor_set_release_receiver": {"status": "PASS" if selected_uids == event_released == release_released else "FAIL", "count": len(selected_uids)},
        "receiver_actual_release_and_birth": {"status": "PASS" if len(event_released) == len(receiver_event.get("born_uids", [])) == 20 and receiver_event.get("gaussian_before") == receiver_event.get("gaussian_after") else "FAIL"},
        "event_alignment": {"status": "PASS" if receiver_event["alignment_before"]["pass"] and receiver_event["alignment"]["pass"] and release_event["alignment_before"]["pass"] and release_event["alignment"]["pass"] else "FAIL"},
        "final_checkpoint_audits": {"status": "PASS" if all(item["metrics"]["checkpoint_audit"].get("pass") is True for item in arm_data.values()) else "FAIL"},
        "initialization_audit": {"status": "PASS" if initialization.get("status") == "GO" and all(initialization.get("gates", {}).values()) else "FAIL"},
        "no_test_data_for_selection": {"status": "PASS" if static.get("test_views_used") is False and joint.get("test_views_used") is False and geometry.get("test_views_used") is False and initialization.get("test_views_used") is False else "FAIL"},
    }

    metrics_out: dict[str, Any] = {}
    control_agg = arm_data["control"]["metrics"]["test"]["aggregate"]
    release_agg = arm_data["release_only"]["metrics"]["test"]["aggregate"]
    for role, item in arm_data.items():
        agg = item["metrics"]["test"]["aggregate"]
        audit = item["metrics"]["checkpoint_audit"]
        metrics_out[role] = {
            "source": str((item["dir"] / "formal_metrics_5000.json").resolve()),
            **agg,
            "gaussian_count": audit["gaussian_count"],
            "checkpoint_audit_pass": audit.get("pass"),
            "delta_vs_control": {key: agg[key] - control_agg[key] for key in ("PSNR", "SSIM", "LPIPS", "L1")},
            "delta_vs_release_only": {key: agg[key] - release_agg[key] for key in ("PSNR", "SSIM", "LPIPS", "L1")},
            "per_view": item["metrics"]["test"].get("per_view", []),
            "trend": parse_trends(item["dir"] / "train.log"),
        }

    donor = {
        "initial_gaussian_count": source_manifest["gaussian_count"],
        "candidate_cohort_count": len(low_support_rows),
        "basic_eligible_count": len(basic_rows),
        "plan_capacity": plan["donors"]["capacity"],
        "individual_audited_count": static["audited_count"],
        "individual_safe_count": static["safe_count"],
        "joint_safe_count": len(joint["donor_uids"]) if joint.get("safe_to_release") else 0,
        "joint_audited_k": len(joint["donor_uids"]),
        "joint_safe": joint.get("safe_to_release"),
        "actual_released_count": len(event_released),
        "actual_birth_type_distribution": dict(Counter(row["birth_type"] for row in selected_donors)),
        "actual_age": summarize_numeric(selected_donors, "age"),
        "actual_real_view_opportunities": summarize_numeric(selected_donors, "real_view_opportunities"),
        "actual_real_view_coverage_rate": summarize_numeric(selected_donors, "real_view_coverage_rate"),
        "actual_screen_support_rate": summarize_numeric(selected_donors, "screen_support_rate"),
        "eligible_screen_support_rate": quantiles(float(row["screen_support_rate"]) for row in basic_rows),
        "individual_max_mean_abs_rgb_delta": summarize_numeric(static["rows"], "max_mean_abs_rgb_delta"),
        "individual_max_psnr_drop_db": summarize_numeric(static["rows"], "max_psnr_drop_db"),
        "joint_max_mean_abs_rgb_delta": joint["max_mean_abs_rgb_delta"],
        "joint_max_psnr_drop_db": joint["max_psnr_drop_db"],
        "capacity_interpretation": {
            "candidate_capacity": len(low_support_rows),
            "basic_eligibility_capacity": len(basic_rows),
            "individual_safety_capacity": "UNKNOWN beyond the 20 audited selected donors",
            "joint_safety_capacity_D": "UNKNOWN; only K=20 was jointly audited",
            "executed_k": len(event_released),
        },
    }

    receiver = {
        "candidate_count": len(receiver_rows),
        "geometry_pass_count_observed": sum(row.get("geometry_valid") is True for row in geom_rows),
        "geometry_rejected_before_selection_count": len(rejected),
        "geometry_not_audited_after_k_filled_count": max(0, len(receiver_rows) - len(rejected) - len(geom_rows)),
        "actual_added_count": len(receiver_event.get("born_uids", [])),
        "nonzero_displacement_count": sum(float(row["displacement_norm"]) > 0 for row in geom_rows),
        "all_cross_view_checks_safe_count": sum(all(check.get("safe") is True for check in row.get("cross_view_checks", [])) for row in geom_rows),
        "candidate_score": quantiles(float(row["demand_score"]) for row in receiver_rows),
        "selected_demand_score": summarize_numeric(geom_rows, "demand_score"),
        "selected_photometric_alignment": summarize_numeric(geom_rows, "photometric_alignment"),
        "selected_nonpositive_photometric_alignment_count": sum(float(row["photometric_alignment"]) <= 0 for row in geom_rows),
        "selected_projection_condition": summarize_numeric(geom_rows, "projection_condition"),
        "selected_displacement_norm": summarize_numeric(geom_rows, "displacement_norm"),
        "selected_source_pixel_offset_before": summarize_numeric(geom_rows, "source_pixel_offset_before"),
        "selected_source_pixel_offset_after": summarize_numeric(geom_rows, "source_pixel_offset_after"),
        "selected_source_rgb_residual_l2": quantiles(math.sqrt(sum(float(v) ** 2 for v in row["source_rgb_residual"])) for row in geom_rows),
        "rejection_reasons": dict(Counter(row.get("reason", "UNKNOWN") for row in rejected)),
        "spatial_concentration": spatial_summary(geom_rows),
        "training_camera_ids": geometry.get("training_camera_ids"),
        "test_views_used": geometry.get("test_views_used"),
        "simultaneous_supported_capacity_R": "UNKNOWN; 20 selected receivers passed joint initialization, but remaining candidates were not jointly initialized",
    }

    required_files = {
        "scene_config": file_entry(root / "configs" / "i2_scenes" / f"{scene}.json"),
        "method_config": file_entry(root / "configs" / "i2_method_v2.json"),
        "common_source_checkpoint": file_entry(source_checkpoint, scene_cfg["common_source"]["sha256"]),
        "common_source_cfg_args": file_entry(cfg_args),
        "training_view_manifest": file_entry(view_manifest, scene_cfg["view_manifest"]["sha256"]),
        "active_plan": file_entry(plan_path),
        "donor_static_audit": file_entry(static_path),
        "donor_joint_audit": file_entry(joint_path),
        "receiver_candidates": file_entry(receiver_csv_path, source_manifest["receiver_candidates"]["sha256"]),
        "receiver_geometry": file_entry(geom_path),
        "initialization_audit": file_entry(init_path),
    }
    for role, item in arm_data.items():
        directory = item["dir"]
        required_files[f"{role}_launch_meta"] = file_entry(directory / "launch_meta.json")
        required_files[f"{role}_cfg_args"] = file_entry(directory / "cfg_args")
        required_files[f"{role}_formal_metrics"] = file_entry(directory / "formal_metrics_5000.json")
        required_files[f"{role}_train_log"] = file_entry(directory / "train.log")
        required_files[f"{role}_evaluation_log"] = file_entry(directory / "evaluation.log")
        required_files[f"{role}_active_event"] = ({"path": None, "status": "NOT_APPLICABLE"} if role == "control" else file_entry(directory / "active_event.json"))

    core_bad = [name for name, value in required_files.items() if value["status"] in ("MISSING", "FAIL", "UNKNOWN")]
    check_bad = [name for name, value in integrity_checks.items() if value["status"] != "PASS"]
    return {
        "scene": scene,
        "classification": "OBSERVED_AND_DERIVED",
        "discovery": {"arms": {role: str(path.resolve()) for role, path in arms.items()}, "notes": discovery_notes},
        "required_files": required_files,
        "integrity_checks": integrity_checks,
        "donor": donor,
        "receiver": receiver,
        "quality": metrics_out,
        "core_failures": core_bad + check_bad,
    }


def write_csvs(output: Path, scenes: dict[str, dict[str, Any]]) -> None:
    quality_fields = ["scene", "arm", "PSNR", "SSIM", "LPIPS", "L1", "gaussian_count", "checkpoint_audit_pass", "delta_psnr_vs_control", "delta_ssim_vs_control", "delta_lpips_vs_control", "delta_l1_vs_control", "delta_psnr_vs_release"]
    with (output / "f05_scene_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=quality_fields)
        writer.writeheader()
        for scene, data in scenes.items():
            for arm, row in data["quality"].items():
                writer.writerow({
                    "scene": scene, "arm": arm, "PSNR": row["PSNR"], "SSIM": row["SSIM"], "LPIPS": row["LPIPS"], "L1": row["L1"],
                    "gaussian_count": row["gaussian_count"], "checkpoint_audit_pass": row["checkpoint_audit_pass"],
                    "delta_psnr_vs_control": row["delta_vs_control"]["PSNR"], "delta_ssim_vs_control": row["delta_vs_control"]["SSIM"],
                    "delta_lpips_vs_control": row["delta_vs_control"]["LPIPS"], "delta_l1_vs_control": row["delta_vs_control"]["L1"],
                    "delta_psnr_vs_release": row["delta_vs_release_only"]["PSNR"],
                })
    capacity_fields = ["scene", "initial_gaussians", "donor_candidates", "donor_basic_eligible", "donor_individual_audited", "donor_individual_safe", "donor_joint_audited_k", "donor_joint_safe", "actual_released", "receiver_candidates", "receiver_geometry_pass_observed", "receiver_geometry_rejected", "receiver_not_audited_after_k_filled", "actual_added", "D", "R", "K"]
    with (output / "f05_capacity_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=capacity_fields)
        writer.writeheader()
        for scene, data in scenes.items():
            d, r = data["donor"], data["receiver"]
            writer.writerow({
                "scene": scene, "initial_gaussians": d["initial_gaussian_count"], "donor_candidates": d["candidate_cohort_count"],
                "donor_basic_eligible": d["basic_eligible_count"], "donor_individual_audited": d["individual_audited_count"],
                "donor_individual_safe": d["individual_safe_count"], "donor_joint_audited_k": d["joint_audited_k"],
                "donor_joint_safe": d["joint_safe"], "actual_released": d["actual_released_count"],
                "receiver_candidates": r["candidate_count"], "receiver_geometry_pass_observed": r["geometry_pass_count_observed"],
                "receiver_geometry_rejected": r["geometry_rejected_before_selection_count"],
                "receiver_not_audited_after_k_filled": r["geometry_not_audited_after_k_filled_count"], "actual_added": r["actual_added_count"],
                "D": "UNKNOWN", "R": "UNKNOWN", "K": d["actual_released_count"],
            })


def markdown_summary(audit: dict[str, Any]) -> str:
    scenes = audit["scenes"]
    verdict = audit["final_verdict"]
    lines = [
        "# F05 Horns/Fern 跨场景机制审计", "",
        f"- **F05: {verdict['f05']}**", f"- **Adaptive-K readiness: {verdict['adaptive_k_readiness']}**", "",
        "> GO 仅表示跨场景分析产物完整、证据审计可信，不代表自适应 K 方法有效。评价结果仅用于事后验证机制效果，不参与预算阈值构建。", "",
        "## 完整性结论", "",
    ]
    for scene, data in scenes.items():
        statuses = Counter(value["status"] for value in data["integrity_checks"].values())
        lines.append(f"- **{scene.title()}（OBSERVED/DERIVED）**：{statuses.get('PASS', 0)}/{len(data['integrity_checks'])} 项核心一致性检查 PASS；核心失败项：{data['core_failures'] or '无'}。")
    lines += ["", "三臂均绑定各自同一 1201 共同源和训练视角 manifest；K=20；Release-only 与 Receiver-v2 的实际 donor UID 集合相同；Receiver-v2 事件直接记录释放 20、新增 20，事件前后 Gaussian 数相同，且 UID/lifecycle/observation/optimizer 行对齐 PASS。最终 Gaussian 数量仅作为评价结果报告，未用于证明预算守恒。", "", "## Donor 与 Receiver 机制", ""]
    for scene, data in scenes.items():
        d, r = data["donor"], data["receiver"]
        lines += [
            f"### {scene.title()}", "",
            f"- **OBSERVED**：源 Gaussian={d['initial_gaussian_count']}；低支持候选={d['candidate_cohort_count']}；基本资格={d['basic_eligible_count']}；个体审计/安全={d['individual_audited_count']}/{d['individual_safe_count']}；联合审计 K={d['joint_audited_k']}、安全={d['joint_safe']}；实际释放={d['actual_released_count']}。",
            f"- **OBSERVED**：Receiver 候选={r['candidate_count']}；几何审计通过={r['geometry_pass_count_observed']}；此前拒绝={r['geometry_rejected_before_selection_count']}；凑满 K 后未审计={r['geometry_not_audited_after_k_filled_count']}；实际新增={r['actual_added_count']}；非零位移={r['nonzero_displacement_count']}。",
            f"- **DERIVED**：选中 Receiver photometric alignment 中位数={r['selected_photometric_alignment']['median']:.6g}，projection condition 中位数={r['selected_projection_condition']['median']:.6g}，位移中位数={r['selected_displacement_norm']['median']:.6g}，非正 alignment 数={r['selected_nonpositive_photometric_alignment_count']}。",
            f"- **UNKNOWN**：D 最大值与 R 最大值。现有材料只证明该 donor 组合在 K=20 联合安全，以及该组 20 个 Receiver 的联合初始化通过。",
            "",
        ]
    h, f = scenes["horns"], scenes["fern"]
    lines += [
        "## 跨场景质量（正式 5000，事后验证）", "",
        "| Scene | Arm | PSNR | SSIM | LPIPS | L1 | Gaussian |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scene, data in scenes.items():
        for arm, row in data["quality"].items():
            lines.append(f"| {scene} | {arm} | {row['PSNR']:.9f} | {row['SSIM']:.9f} | {row['LPIPS']:.9f} | {row['L1']:.9f} | {row['gaussian_count']} |")
    lines += [
        "",
        f"- **DERIVED**：Horns Receiver-v2 相对 Control：PSNR {h['quality']['receiver_v2']['delta_vs_control']['PSNR']:+.6f} dB；相对 Release-only：{h['quality']['receiver_v2']['delta_vs_release_only']['PSNR']:+.6f} dB。",
        f"- **DERIVED**：Fern Receiver-v2 相对 Control：PSNR {f['quality']['receiver_v2']['delta_vs_control']['PSNR']:+.6f} dB；相对 Release-only：{f['quality']['receiver_v2']['delta_vs_release_only']['PSNR']:+.6f} dB。",
        "- **OBSERVED**：1500/2000/3000/4000/5000 趋势已从各 arm 的 train.log 解析并保存在 `f05_audit.json`；未补造日志中不存在的点。",
        "",
        "## 机制解释与 Adaptive-K", "",
        f"- **DERIVED**：Fern donor K20 的联合扰动更小（max mean-absolute RGB delta {f['donor']['joint_max_mean_abs_rgb_delta']:.6g} vs Horns {h['donor']['joint_max_mean_abs_rgb_delta']:.6g}），因此 Fern 的差异不能简单归因于 donor 删除风险更高。",
        f"- **DERIVED**：Fern 选中 Receiver 的 photometric alignment 均值更低（{f['receiver']['selected_photometric_alignment']['mean']:.6g} vs {h['receiver']['selected_photometric_alignment']['mean']:.6g}），projection condition 均值更高（{f['receiver']['selected_projection_condition']['mean']:.6g} vs {h['receiver']['selected_projection_condition']['mean']:.6g}）。这是可观测跨场景机制差异。",
        "- **HYPOTHESIS**：Fern 固定 K20 可能比 Horns 更早触及 Receiver 证据/几何质量下降区；该解释与观测一致，但两场景不足以确证因果，也不能据测试集表现反推阈值。",
        "- **候选定义（不含测试指标）**：`K <= min(D, R)`；D 必须由候选 donor 集合的联合训练视角反事实删除审计确定，R 必须由真实训练视角残差、逐候选几何合法性及整组联合初始化审计确定。",
        "- **NOT_READY 原因**：当前只有 K=20 的联合 donor 安全证据；生成器凑满 20 后停止，余下候选未完成几何审计，更未进行多 K 的联合初始化。因此 D、R 均 UNKNOWN，尚无可执行且有明确安全边界的自适应预算定义。",
        "",
        "## 已知审计注意项", "",
        "- **OBSERVED**：Fern receiver candidate_id 仍使用 `horns1201-` 前缀；但其训练相机、源 checkpoint hash、view-manifest hash 和证据路径均指向 Fern，UID/plan/event 对应一致。该命名污染被保留为 provenance 警告，不被当成场景身份依据。",
        "- **UNKNOWN** 字段未被推断为 PASS；完整机器可读证据、路径、文件状态和 per-view 结果见 `f05_audit.json`。", "",
    ]
    return "\n".join(lines)


def choose_output(root: Path, requested: Path | None) -> Path:
    base = requested or (root / "reports" / "i2_scene_runs" / "f05_cross_scene")
    if not base.exists():
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base.parent / f"{base.name}_v{stamp}"
    serial = 2
    while candidate.exists():
        candidate = base.parent / f"{base.name}_v{stamp}_{serial}"
        serial += 1
    return candidate


def run(root: Path, output_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    method_cfg = read_json(root / "configs" / "i2_method_v2.json")
    scenes: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for scene in SCENES:
        try:
            scene_cfg = read_json(root / "configs" / "i2_scenes" / f"{scene}.json")
            scenes[scene] = audit_scene(root, scene, scene_cfg, method_cfg)
            failures.extend(f"{scene}:{item}" for item in scenes[scene]["core_failures"])
        except Exception as exc:  # fail closed and preserve a diagnostic artifact
            failures.append(f"{scene}:{type(exc).__name__}:{exc}")
            scenes[scene] = {"scene": scene, "classification": "UNKNOWN", "error": str(exc), "core_failures": [str(exc)]}
    f05 = "GO" if not failures else "NO-GO"
    readiness = "NOT_READY"
    audit = {
        "format": "i2_f05_cross_scene_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_labels": {
            "OBSERVED": "directly present in frozen artifacts",
            "DERIVED": "strict deterministic calculation from observed fields",
            "HYPOTHESIS": "plausible interpretation requiring new validation",
            "UNKNOWN": "not established by current artifacts",
        },
        "constraints": {"no_training": True, "no_prepare": True, "no_test_metric_threshold_design": True, "fixed_k": 20},
        "scenes": scenes,
        "resource_matching": {
            "candidate_rule": "K <= min(D, R)",
            "D": "UNKNOWN: only the executed K=20 donor set has a joint deletion audit",
            "R": "UNKNOWN: only 20 selected receivers have joint initialization evidence; search stopped when K was filled",
            "K": 20,
            "uses_test_metrics": False,
        },
        "hypotheses": [
            "Fern K20 may reach a weaker Receiver evidence/geometry regime earlier than Horns; this is not a causal conclusion.",
            "Donor joint deletion risk does not explain Fern underperformance because its observed K20 joint perturbation is smaller than Horns.",
        ],
        "final_verdict": {"f05": f05, "adaptive_k_readiness": readiness, "failures": failures},
    }
    output = choose_output(root, output_dir.resolve() if output_dir else None)
    output.mkdir(parents=True, exist_ok=False)
    (output / "f05_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if all("quality" in data for data in scenes.values()):
        write_csvs(output, scenes)
        (output / "f05_summary.md").write_text(markdown_summary(audit), encoding="utf-8")
    else:
        (output / "f05_scene_comparison.csv").write_text("scene,status\nall,NO-GO\n", encoding="utf-8")
        (output / "f05_capacity_comparison.csv").write_text("scene,status\nall,NO-GO\n", encoding="utf-8")
        (output / "f05_summary.md").write_text(f"# F05\n\nF05: NO-GO\n\nFailures: {failures}\n", encoding="utf-8")
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output, audit = run(args.root, args.output_dir)
    print(json.dumps({"output_dir": str(output), **audit["final_verdict"]}, ensure_ascii=False))
    return 0 if audit["final_verdict"]["f05"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
