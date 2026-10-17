#!/usr/bin/env python3
"""R0-v1/RD-v1 training-evidence-only adaptive-K audit for configured I2 scenes.

This tool consumes the common-source binding and shared prepare artifacts.  It
never trains and writes only to a new, versioned report tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import PipelineParams
from gaussian_renderer import render
from tools.audit_i2_receiver_v2_initialization import render_arm
from tools.build_i2_receiver_v2 import candidate_geometry, load_depth_payloads
from tools.build_i2_receivers import load_runtime, parse_cfg
from utils.i2_active_pilot import load_active_plan
from utils.i2_adaptive_k import (
    AdaptiveKError,
    complete_receiver_selection,
    make_decision,
    receiver_evidence_budget,
    rd_receiver_capacity,
    rd_donor_capacity,
    make_rd_decision,
    sha256,
    write_json,
)
from utils.i2_d0_dry_run import build_d0_plan, load_d0_inputs, select_donors
from utils.i2_scene_pipeline import (
    PipelineError,
    load_configs,
    preflight,
    resolve_common_source,
    resolve_path,
)


class Log:
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def __call__(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self):
        self.handle.close()


def new_output(root: Path) -> Path:
    """Create exactly the requested directory; never reuse or silently rename it."""
    root.mkdir(parents=True, exist_ok=False)
    return root


def pipeline_params():
    parser = argparse.ArgumentParser(add_help=False)
    return PipelineParams(parser).extract(parser.parse_args([]))


def peak():
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


@torch.no_grad()
def audit_receivers(scene_cfg, method, policy, checkpoint, cfg_path,
                    receiver_manifest_path, report_path, output, log):
    started = time.perf_counter()
    scene = scene_cfg["scene"]
    manifest, b1_rows, c_rows, receiver_rows = load_d0_inputs(receiver_manifest_path)
    donor_cfg = method["donor"]
    old_plan = build_d0_plan(
        manifest, b1_rows, c_rows, receiver_rows, budget=0,
        min_age=int(donor_cfg["minimum_age"]),
        min_opportunities=int(donor_cfg["minimum_real_view_opportunities"]),
    )
    report, payloads = load_depth_payloads(report_path)
    if report.get("test_views_used") is not False:
        raise RuntimeError("receiver evidence used test views")
    rd_v1 = policy.get("version") == "RD-v1"
    if rd_v1 and report.get("enumeration", {}).get("complete") is not True:
        raise AdaptiveKError("RD-v1 requires complete Receiver candidate enumeration")

    cfg = parse_cfg(cfg_path)
    model, cameras, iteration, _ = load_runtime(cfg, checkpoint)
    expected_iteration = int(method["source"]["iteration"])
    if iteration != expected_iteration:
        raise RuntimeError(f"expected source iteration {expected_iteration}, got {iteration}")
    camera_by_name = {camera.image_name: camera for camera in cameras}
    camera_order = [Path(x).stem for x in old_plan["input_audit"]["receiver_protocol"]["training_camera_ids"]]
    configured_order = [Path(x).stem for x in scene_cfg["split"]["train_images"]]
    if camera_order != configured_order or set(camera_by_name) != set(camera_order):
        raise RuntimeError("runtime training cameras differ from frozen receiver protocol")
    row_by_uid = {int(uid): i for i, uid in enumerate(model.lifecycle.uid.tolist())}
    background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    pipe = pipeline_params()
    base_images = {camera.image_name: render(
        camera, model, pipe, background, train=True, dropout_factor=0.0, sigma_noise=0.0
    )["render"].clamp(0, 1) for camera in cameras}
    receiver_cfg = method["receiver_v2"]
    valid, geometry_rows = [], []
    for index, row in enumerate(old_plan["receivers"]["rows"], 1):
        evidence, reason = candidate_geometry(
            row, model=model, row_by_uid=row_by_uid, camera_by_name=camera_by_name,
            camera_order=camera_order, payloads=payloads,
            max_condition=float(receiver_cfg["max_projection_condition"]),
            displacement_factor=float(receiver_cfg["max_displacement_tangent_norm_factor"]),
            opacity=float(receiver_cfg["activated_opacity"]),
            footprint_pixels=float(receiver_cfg["footprint_pixels"]),
            base_images=base_images,
        )
        if evidence is None:
            geometry_rows.append({"candidate_id": row["candidate_id"], "geometry_status": "FAIL",
                                  "geometry_failure_reason": reason, "nms_status": "NOT_ELIGIBLE"})
        else:
            valid.append(evidence)
            geometry_rows.append({"candidate_id": row["candidate_id"], "geometry_status": "PASS",
                                  "geometry_failure_reason": None, "geometry_evidence": evidence})
        if index % 25 == 0:
            log(f"{scene}: receiver geometry {index}/{len(old_plan['receivers']['rows'])}")
    accepted, nms_rejected = complete_receiver_selection(
        valid, spatial_dedup_radius=float(report["spatial_dedup_radius"])
    )
    accepted_ids = [row["candidate_id"] for row in accepted]
    accepted_rank = {cid: i + 1 for i, cid in enumerate(accepted_ids)}
    rejected_by_id = {row["candidate_id"]: row for row in nms_rejected}
    for row in geometry_rows:
        cid = row["candidate_id"]
        if row["geometry_status"] != "PASS":
            continue
        if cid in accepted_rank:
            row.update(nms_status="ACCEPT", accepted_rank=accepted_rank[cid], nms_failure_reason=None)
        else:
            reject = rejected_by_id[cid]
            row.update(nms_status="REJECT", accepted_rank=None,
                       nms_failure_reason=reject["reason"],
                       conflict_candidate_id=reject.get("conflict_candidate_id"))
    receiver_budget = (
        rd_receiver_capacity(accepted)
        if rd_v1 else
        receiver_evidence_budget(accepted, k_cap=int(policy["candidate_budget_cap"]))
    )
    payload = {
        "format": "i2_receiver_v2_audit_all_rd_v1" if rd_v1 else "i2_receiver_v2_audit_all_v1", "scene": scene,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256(checkpoint),
        "policy": {"id": policy["policy_id"], "version": policy["version"]},
        "receiver_manifest": {
            "path": str(receiver_manifest_path), "sha256": sha256(receiver_manifest_path)
        },
        "receiver_report": {"path": str(report_path), "sha256": sha256(report_path)},
        "training_camera_ids": old_plan["input_audit"]["receiver_protocol"]["training_camera_ids"],
        "protocol": "frozen Receiver-v2 geometry/rank/greedy-NMS; full-float train renders; dropout=0; sigma_noise=0",
        "test_views_used": False, "candidate_count": len(geometry_rows),
        "geometry_pass_count": len(valid), "nms_accepted_count": len(accepted),
        "accepted_order": accepted_ids, "receiver_budget": receiver_budget,
        "rows": geometry_rows, "elapsed_seconds": time.perf_counter() - started,
        "render_count": len(cameras), "peak_cuda_memory": peak(),
        "coverage": {
            "complete": bool(rd_v1),
            "generator": report.get("enumeration"),
            "input_candidate_count": len(old_plan["receivers"]["rows"]),
            "geometry_audited_count": len(geometry_rows),
        },
    }
    path = write_json(output / "receiver_audit_all.json", payload)
    log(f"{scene}: receiver audit-all accepted {len(accepted)}/{len(valid)} geometry-pass candidates")
    return model, cameras, old_plan, accepted, receiver_budget, path, payload


@torch.no_grad()
def audit_donors(scene, model, cameras, old_plan, checkpoint, cfg_path, receiver_capacity,
                 output, method, source_iteration, log):
    started = time.perf_counter()
    manifest_path = Path(old_plan["input_audit"]["source_manifest"]["path"])
    manifest, b1_rows, _, _ = load_d0_inputs(manifest_path)
    config = old_plan["configuration"]
    donor_result = select_donors(
        b1_rows, budget=0, min_age=int(config["min_age"]),
        min_opportunities=int(config["min_real_view_opportunities"]),
        excluded_uids=config.get("excluded_uids", []),
    )
    accepted_anchor_uids = {int(row["anchor_uid"]) for row in old_plan["receivers"]["rows"]}
    ranked = [row for row in donor_result["eligible"] if int(row["uid"]) not in accepted_anchor_uids]
    # This is an operational upper bound, not a new safety/quality threshold.
    scan_limit = min(len(ranked), max(receiver_capacity * 4, receiver_capacity + 64))
    mean_limit = float(method["donor"]["max_single_or_joint_mean_abs_rgb_delta"])
    psnr_limit = float(method["donor"]["max_single_or_joint_psnr_drop_db"])
    cfg = parse_cfg(cfg_path)
    background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    pipe = pipeline_params()
    uid_rows = {int(uid): i for i, uid in enumerate(model.lifecycle.uid.detach().cpu().tolist())}
    baselines, base_psnr = [], []
    for camera in cameras:
        image = render(camera, model, pipe, background, train=False,
                       dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
        mse = (image - camera.original_image[:3]).square().mean().clamp_min(1e-12)
        baselines.append(image)
        base_psnr.append(float(-10 * torch.log10(mse)))
    render_count = len(cameras)
    individual, safe_rows, joint_rows = [], [], []
    joint_dir = output / "donor_joint_prefixes"
    joint_dir.mkdir()
    stop_reason = "receiver_capacity_reached"
    for rank, donor in enumerate(ranked[:scan_limit], 1):
        uid = int(donor["uid"])
        tensor_row = uid_rows.get(uid)
        if tensor_row is None:
            raise RuntimeError(f"ranked donor UID {uid} not live in checkpoint")
        original = model._opacity[tensor_row].detach().clone()
        model._opacity[tensor_row].fill_(-100.0)
        views = []
        try:
            for camera, baseline, baseline_value in zip(cameras, baselines, base_psnr):
                deleted = render(camera, model, pipe, background, train=False,
                                 dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
                render_count += 1
                delta = (deleted - baseline).abs()
                mse = (deleted - camera.original_image[:3]).square().mean().clamp_min(1e-12)
                views.append({"training_camera_uid": int(camera.uid),
                              "training_camera_id": camera.image_name,
                              "mean_abs_rgb_delta": float(delta.mean()),
                              "max_abs_rgb_delta": float(delta.max()),
                              "psnr_drop_db": float(baseline_value - float(-10 * torch.log10(mse)))})
        finally:
            model._opacity[tensor_row].copy_(original)
        max_mean = max(x["mean_abs_rgb_delta"] for x in views)
        max_rgb = max(x["max_abs_rgb_delta"] for x in views)
        max_drop = max(x["psnr_drop_db"] for x in views)
        safe = bool(all(math.isfinite(x) for x in (max_mean, max_rgb, max_drop))
                    and max_mean <= mean_limit and max_drop <= psnr_limit)
        item = {"uid": uid, "rank": rank, "max_mean_abs_rgb_delta": max_mean,
                "max_abs_rgb_delta": max_rgb, "max_psnr_drop_db": max_drop,
                "safe_to_release": safe, "per_view": views}
        individual.append(item)
        if not safe:
            log(f"{scene}: donor rank {rank} uid={uid} individual FAIL")
            continue
        safe_rows.append(donor)
        originals = {}
        for selected in safe_rows:
            selected_uid = int(selected["uid"])
            selected_row = uid_rows[selected_uid]
            originals[selected_uid] = model._opacity[selected_row].detach().clone()
            model._opacity[selected_row].fill_(-100.0)
        joint_views = []
        try:
            for view_index, (camera, baseline, baseline_value) in enumerate(zip(cameras, baselines, base_psnr)):
                deleted = render(camera, model, pipe, background, train=False,
                                 dropout_factor=float(cfg.dropout_factor))["render"].clamp(0, 1)
                render_count += 1
                delta = (deleted - baseline).abs()
                mse = (deleted - camera.original_image[:3]).square().mean().clamp_min(1e-12)
                selected_cf = [next(x for x in individual if x["uid"] == int(d["uid"])) for d in safe_rows]
                single_sum = sum(x["per_view"][view_index]["mean_abs_rgb_delta"] for x in selected_cf)
                value = float(delta.mean())
                joint_views.append({"training_camera_uid": int(camera.uid),
                    "training_camera_id": camera.image_name, "mean_abs_rgb_delta": value,
                    "max_abs_rgb_delta": float(delta.max()),
                    "psnr_drop_db": float(baseline_value - float(-10 * torch.log10(mse))),
                    "single_point_sum_mean_abs_rgb_delta": single_sum,
                    "joint_minus_single_sum_mean_abs_rgb_delta": value - single_sum})
        finally:
            for selected_uid, saved in originals.items():
                model._opacity[uid_rows[selected_uid]].copy_(saved)
        k = len(safe_rows)
        joint_mean = max(x["mean_abs_rgb_delta"] for x in joint_views)
        joint_drop = max(x["psnr_drop_db"] for x in joint_views)
        joint_safe = bool(math.isfinite(joint_mean) and math.isfinite(joint_drop)
                          and joint_mean <= mean_limit and joint_drop <= psnr_limit)
        joint_payload = {"format": "i2_joint_deletion_counterfactual_v1",
            "source_iteration": source_iteration, "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": sha256(checkpoint), "training_view_count": len(cameras),
            "test_views_used": False, "donor_uids": [int(x["uid"]) for x in safe_rows],
            "thresholds": {"max_mean_abs_rgb_delta": mean_limit, "max_psnr_drop_db": psnr_limit},
            "max_mean_abs_rgb_delta": joint_mean, "max_psnr_drop_db": joint_drop,
            "safe_to_release": joint_safe, "per_view": joint_views}
        joint_path = write_json(joint_dir / f"donor_joint_deletion_counterfactual_k{k}.json", joint_payload)
        joint_rows.append({"k": k, "safe": joint_safe, "artifact": str(joint_path),
                           "sha256": sha256(joint_path), "max_mean_abs_rgb_delta": joint_mean,
                           "max_psnr_drop_db": joint_drop})
        log(f"{scene}: donor joint prefix K{k} {'PASS' if joint_safe else 'FAIL'}")
        if not joint_safe:
            safe_rows.pop()
            stop_reason = f"first_joint_failure_k{k}"
            break
        if k == receiver_capacity:
            break
    else:
        stop_reason = (
            "eligible_donors_exhausted"
            if scan_limit == len(ranked)
            else "bounded_scan_exhausted_before_receiver_capacity"
        )
    static_payload = {"format": "i2_static_deletion_counterfactual_v1",
        "source_iteration": source_iteration, "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint), "training_view_count": len(cameras),
        "test_views_used": False,
        "thresholds": {"max_mean_abs_rgb_delta": mean_limit, "max_psnr_drop_db": psnr_limit},
        "audited_count": len(individual), "safe_count": sum(x["safe_to_release"] for x in individual),
        "rows": individual}
    static_path = write_json(output / "donor_static_deletion_counterfactual.json", static_payload)
    determined = stop_reason != "bounded_scan_exhausted_before_receiver_capacity"
    summary = {"format": "i2_bounded_donor_capacity_audit_v1", "scene": scene,
        "eligible_count": donor_result["capacity"], "anchor_excluded_count": donor_result["capacity"] - len(ranked),
        "receiver_capacity_target": receiver_capacity, "scan_limit": scan_limit,
        "individual_audited_count": len(individual),
        "individual_safe_count": sum(row["safe_to_release"] is True for row in individual),
        "contiguous_joint_safe_capacity": len(safe_rows), "capacity_determined": determined,
        "stop_reason": stop_reason, "joint_prefixes": joint_rows,
        "static_artifact": {"path": str(static_path), "sha256": sha256(static_path)},
        "elapsed_seconds": time.perf_counter() - started, "render_count": render_count,
        "peak_cuda_memory": peak(), "test_views_used": False}
    summary_path = write_json(output / "donor_capacity_audit.json", summary)
    return safe_rows, individual, joint_rows, static_path, summary_path, summary


def audit_rd_donors(scene, model, old_plan, checkpoint, output, source_iteration,
                    accepted_receivers):
    """Audit the complete structural D0/B1 donor set without render gates."""
    manifest_path = Path(old_plan["input_audit"]["source_manifest"]["path"])
    _, b1_rows, _, _ = load_d0_inputs(manifest_path)
    config = old_plan["configuration"]
    donor_result = select_donors(
        b1_rows, budget=0, min_age=int(config["min_age"]),
        min_opportunities=int(config["min_real_view_opportunities"]),
        excluded_uids=config.get("excluded_uids", []),
    )
    live_uids = [int(value) for value in model.lifecycle.uid.detach().cpu().tolist()]
    anchors = [int(row["anchor_uid"]) for row in accepted_receivers]
    capacity = rd_donor_capacity(
        donor_result["eligible"], receiver_anchor_uids=anchors, live_uids=live_uids
    )
    payload = {
        **capacity,
        "scene": scene,
        "source_iteration": source_iteration,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "test_views_used": False,
        "selection_inputs": "D0/B1 frozen observation evidence only",
        "render_counterfactuals_used": False,
    }
    path = write_json(output / "donor_capacity_rd_v1.json", payload)
    return capacity["ranked_eligible"], path, payload


def build_rd_plan(*, scene_cfg, k, old_plan, checkpoint, receivers, donors,
                  receiver_capacity, receiver_audit_path, donor_audit_path,
                  output, source_iteration):
    """Write the independent RD-v1 Plan schema consumed by the runtime adapter."""
    if not (k > 0 and k <= len(receivers) and k <= len(donors)):
        raise AdaptiveKError("RD-v1 Plan K is outside complete capacities")
    selected_receivers = receivers[:k]
    selected_donors = donors[:k]
    view_manifest = resolve_path(scene_cfg["view_manifest"]["path"])
    candidate_table = Path(old_plan["input_audit"]["receiver_candidate_table"]["path"])
    if view_manifest is None:
        raise AdaptiveKError("RD-v1 training view manifest is missing")
    plan = {
        "format": "i2_rd_v1_active_plan",
        "policy_version": "RD-v1",
        "scene": scene_cfg["scene"],
        "uses_test_views": False,
        "source": {
            "iteration": source_iteration,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        },
        "training_view_manifest": {"path": str(view_manifest), "sha256": sha256(view_manifest)},
        "receiver_candidate_table": {"path": str(candidate_table), "sha256": sha256(candidate_table)},
        "receiver_audit": {"path": str(receiver_audit_path), "sha256": sha256(receiver_audit_path)},
        "donor_audit": {"path": str(donor_audit_path), "sha256": sha256(donor_audit_path)},
        "selected_k": k,
        "donors": selected_donors,
        "receivers": selected_receivers,
        "matches": [
            {
                "donor_uid": int(donor["uid"]),
                "receiver_candidate_id": receiver["candidate_id"],
                "released_units": 1,
                "received_units": 1,
            }
            for donor, receiver in zip(selected_donors, selected_receivers)
        ],
        "budget_audit": {"released_units": k, "received_units": k, "conserved": True},
        "coverage": {
            "receiver_total": receiver_capacity,
            "donor_total": len(donors),
            "receiver_selected_order": [row["candidate_id"] for row in selected_receivers],
            "donor_selected_order": [int(row["uid"]) for row in selected_donors],
            "complete": True,
        },
        "performance_safety_reports": None,
    }
    path = write_json(output / f"rd_v1_k{k}_plan.json", plan)
    load_active_plan(path, requested_budget=k, source_checkpoint=checkpoint)
    return path


def build_plan(k, old_plan, checkpoint, receivers, donors, individual, static_path,
               joint_row, output, source_iteration):
    if not (1 <= k <= len(donors) and k <= len(receivers)):
        raise ValueError("plan K is outside the audited donor/receiver prefix")
    evidence = {"format": "i2_receiver_v2_geometry_evidence_v1",
        "method_version": "receiver_v2_tangent_jacobian_v1", "source_iteration": source_iteration,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256(checkpoint),
        "receiver_candidate_table": old_plan["input_audit"]["receiver_candidate_table"]["path"],
        "receiver_candidate_table_sha256": old_plan["input_audit"]["receiver_candidate_table"]["sha256"],
        "test_views_used": False, "candidate_count": k, "requested_k": k,
        "rows": receivers[:k]}
    evidence_path = write_json(output / f"receiver_v2_geometry_evidence_k{k}.json", evidence)
    plan = copy.deepcopy(old_plan)
    plan["format"] = "i2_d1_receiver_v2_plan_v1"
    plan["method_version"] = "receiver_v2_tangent_jacobian_v1"
    plan["configuration"]["budget_k"] = k
    plan["donors"]["selected"] = donors[:k]
    plan["matches"] = [{"donor_uid": int(donor["uid"]),
        "receiver_candidate_id": receiver["candidate_id"], "released_units": 1,
        "received_units": 1} for donor, receiver in zip(donors[:k], receivers[:k])]
    plan["budget_audit"].update(requested_k=k, released_units=k, received_units=k, conserved=True)
    plan["rollback"].update(donor_uids=[int(x["uid"]) for x in donors[:k]],
                            receiver_candidate_ids=[x["candidate_id"] for x in receivers[:k]])
    individual_by_uid = {int(x["uid"]): x for x in individual}
    plan["donor_counterfactual"] = {"format": "i2_static_deletion_counterfactual_v1",
        "artifact": {"path": str(static_path), "sha256": sha256(static_path)},
        "source_checkpoint_sha256": sha256(checkpoint),
        "rows": [individual_by_uid[int(x["uid"])] for x in donors[:k]]}
    plan["joint_deletion_counterfactual"] = {"format": "i2_joint_deletion_counterfactual_v1",
        "artifact": {"path": joint_row["artifact"], "sha256": joint_row["sha256"]}}
    plan["safety_readiness"]["static_deletion_counterfactual"] = True
    plan["safety_readiness"]["joint_deletion_counterfactual"] = True
    plan["active_intervention_ready"] = True
    plan["receiver_v2"] = {"method_version": "receiver_v2_tangent_jacobian_v1",
        "artifact": {"path": str(evidence_path), "sha256": sha256(evidence_path)},
        "test_views_used": False}
    plan["receiver_selection_change"] = {"v2_candidate_ids": [x["candidate_id"] for x in receivers[:k]],
                                         "donor_uids_unchanged": [int(x["uid"]) for x in donors[:k]]}
    plan_path = write_json(output / f"adaptive_k{k}_plan.json", plan)
    load_active_plan(plan_path, requested_budget=k, source_checkpoint=checkpoint)
    return plan_path, evidence_path


@torch.no_grad()
def audit_initialization(k, plan_path, evidence_path, checkpoint, cfg_path, output, method):
    started = time.perf_counter()
    args = argparse.Namespace(cfg=str(cfg_path), checkpoint=Path(checkpoint), budget=k)
    with tempfile.TemporaryDirectory(prefix=f"i2-adaptive-k{k}-") as directory:
        temp = Path(directory)
        release, cameras, release_event = render_arm(args, "release", plan_path, temp)
        receiver, _, receiver_event = render_arm(args, "receiver_v2", plan_path, temp)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    masks = {name: torch.zeros((int(camera.image_height), int(camera.image_width)),
             dtype=torch.bool, device="cuda") for name, camera in cameras.items()}
    for row in evidence["rows"]:
        name = Path(row["training_camera_id"]).stem
        x0, y0, x1, y1 = row["pixel_region"]
        masks[name][y0:y1, x0:x1] = True
    def values(images, reference=None):
        target_sum = outside_sum = changed_sum = 0.0
        target_n = outside_n = changed_n = 0
        global_l1 = 0.0
        for name, image in images.items():
            target = cameras[name].original_image[:3].clamp(0, 1)
            residual = (image - target).abs().mean(dim=0)
            mask = masks[name]
            outside = ~mask
            target_sum += float(residual[mask].sum()); target_n += int(mask.sum())
            outside_sum += float(residual[outside].sum()); outside_n += int(outside.sum())
            global_l1 += float((image - target).abs().mean())
            if reference is not None:
                changed_sum += float((image-reference[name]).abs().mean(dim=0)[mask].sum())
                changed_n += int(mask.sum())
        return {"target_region_L1": target_sum / target_n, "outside_region_L1": outside_sum / outside_n,
                "global_L1": global_l1 / len(images),
                "target_change_vs_release": changed_sum / changed_n if changed_n else 0.0}
    release_values = values(release)
    receiver_values = values(receiver, release)
    thresholds = method["initialization_audit"]
    deltas = {name: receiver_values[name] - release_values[name] for name in
              ("target_region_L1", "outside_region_L1", "global_L1")}
    gates = {"receiver_changes_target_regions": receiver_values["target_change_vs_release"] > 0,
        "target_region_not_worse_than_release": deltas["target_region_L1"] <= float(thresholds["max_target_l1_increase"]),
        "outside_region_no_obvious_degradation": deltas["outside_region_L1"] <= float(thresholds["max_outside_l1_increase"]),
        "global_no_obvious_degradation": deltas["global_L1"] <= float(thresholds["max_global_l1_increase"]),
        "budget_and_alignment_pass": bool(receiver_event["budget_conserved"] and receiver_event["alignment"]["pass"])}
    payload = {"format": "i2_adaptive_joint_initialization_audit_v1", "k": k,
        "training_views_only": True, "test_views_used": False,
        "protocol": "release versus Receiver-v2; full-float train render, dropout=0, sigma_noise=0",
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "evidence": {"path": str(evidence_path), "sha256": sha256(evidence_path)},
        "thresholds": thresholds, "release": release_values, "receiver": receiver_values,
        "receiver_minus_release": deltas, "gates": gates, "safe": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started, "model_load_count": 2,
        "render_count": 2 * len(cameras), "peak_cuda_memory": peak(),
        "events": {"release": release_event["format"], "receiver_v2": receiver_event["format"]}}
    path = write_json(output / f"initialization_audit_k{k}.json", payload)
    return payload, path


def audit_scene(scene_cfg, checkpoint, cfg_path, method, policy,
                receiver_manifest_path, receiver_report_path, output, log):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    scene = scene_cfg["scene"]
    source_iteration = int(method["source"]["iteration"])
    model, cameras, old_plan, receivers, receiver_budget, receiver_path, receiver_summary = audit_receivers(
        scene_cfg, method, policy, checkpoint, cfg_path,
        receiver_manifest_path, receiver_report_path, output, log)
    receiver_capacity = int(receiver_budget["receiver_evidence_budget_r"])
    if policy.get("version") == "RD-v1":
        donors, donor_path, donor_summary = audit_rd_donors(
            scene, model, old_plan, checkpoint, output, source_iteration,
            receivers[:receiver_capacity]
        )
        del model, cameras
        torch.cuda.empty_cache()
        evidence = {
            "receiver": {"path": str(receiver_path), "sha256": sha256(receiver_path)},
            "donor": {"path": str(donor_path), "sha256": sha256(donor_path)},
        }
        decision = make_rd_decision(
            scene=scene, source_checkpoint_sha256=sha256(checkpoint),
            receiver_audit=receiver_budget, donor_audit=donor_summary,
            evidence=evidence,
        )
        k = decision["selected_k"]
        if k > 0:
            plans_dir = output / "plans"
            plans_dir.mkdir()
            plan_path = build_rd_plan(
                scene_cfg=scene_cfg, k=k, old_plan=old_plan, checkpoint=checkpoint,
                receivers=receivers, donors=donors,
                receiver_capacity=receiver_capacity,
                receiver_audit_path=receiver_path, donor_audit_path=donor_path,
                output=plans_dir, source_iteration=source_iteration,
            )
            decision["adaptive_plan"] = {"path": str(plan_path), "sha256": sha256(plan_path)}
        decision["gpu_cost"] = {
            "elapsed_seconds": time.perf_counter() - began,
            "receiver_render_count": receiver_summary["render_count"],
            "donor_render_count": 0,
            "initialization_render_count": 0,
            "initialization_model_load_count": 0,
            "peak_cuda_memory": peak(),
        }
        decision_path = write_json(output / "adaptive_k_decision.json", decision)
        return decision, decision_path
    if receiver_capacity == 0:
        del model, cameras
        decision = make_decision(
            scene=scene, source_checkpoint_sha256=sha256(checkpoint),
            receiver_capacity=0, donor_capacity=0, initialization_rows=[],
            evidence={"receiver": {"path": str(receiver_path), "sha256": sha256(receiver_path)}},
            determined=True, k_cap=int(policy["candidate_budget_cap"]),
            receiver_audit=receiver_budget,
        )
        return decision, write_json(output / "adaptive_k_decision.json", decision)
    donors, individual, joint, static_path, donor_path, donor_summary = audit_donors(
        scene, model, cameras, old_plan, checkpoint, cfg_path, receiver_capacity,
        output, method, source_iteration, log)
    del model, cameras
    torch.cuda.empty_cache()
    if not donor_summary["capacity_determined"]:
        decision = make_decision(scene=scene, source_checkpoint_sha256=sha256(checkpoint),
            receiver_capacity=receiver_capacity, donor_capacity=len(donors), initialization_rows=[],
            evidence={"receiver": {"path": str(receiver_path), "sha256": sha256(receiver_path)},
                      "donor": {"path": str(donor_path), "sha256": sha256(donor_path)}},
            determined=False, blocker=donor_summary["stop_reason"],
            k_cap=int(policy["candidate_budget_cap"]), receiver_audit=receiver_budget)
        decision_path = write_json(output / "adaptive_k_decision.json", decision)
        return decision, decision_path
    max_k = min(receiver_capacity, len(donors), int(policy["candidate_budget_cap"]))
    init_rows = []
    plans_dir = output / "plans"; plans_dir.mkdir()
    joint_by_k = {int(x["k"]): x for x in joint if x["safe"]}
    for k in range(1, max_k + 1):
        plan_path, evidence_path = build_plan(k, old_plan, checkpoint, receivers, donors, individual,
                                               static_path, joint_by_k[k], plans_dir,
                                               source_iteration)
        payload, path = audit_initialization(k, plan_path, evidence_path, checkpoint, cfg_path, output, method)
        init_rows.append({"k": k, "safe": payload["safe"], "artifact": str(path), "sha256": sha256(path),
                          "elapsed_seconds": payload["elapsed_seconds"],
                          "render_count": payload["render_count"], "model_load_count": 2})
        log(f"{scene}: initialization prefix K{k} {'PASS' if payload['safe'] else 'FAIL'}")
        if k == 2:
            write_json(output / "preflight_k1_k2.json", {"format": "i2_adaptive_preflight_v1",
                "scene": scene, "rows": init_rows[:2], "status": "GO" if all(x["safe"] for x in init_rows[:2]) else "NO-GO"})
        if not payload["safe"]:
            break
    decision = make_decision(scene=scene, source_checkpoint_sha256=sha256(checkpoint),
        receiver_capacity=receiver_capacity, donor_capacity=len(donors), initialization_rows=init_rows,
        evidence={"receiver": {"path": str(receiver_path), "sha256": sha256(receiver_path)},
                  "donor": {"path": str(donor_path), "sha256": sha256(donor_path)},
                  "initialization": [{"path": x["artifact"], "sha256": x["sha256"]} for x in init_rows]},
        determined=True, k_cap=int(policy["candidate_budget_cap"]),
        receiver_audit=receiver_budget)
    if decision["selected_k"] != 0:
        final_plan = plans_dir / f"adaptive_k{decision['selected_k']}_plan.json"
        decision["adaptive_plan"] = {"path": str(final_plan), "sha256": sha256(final_plan)}
    decision["gpu_cost"] = {"elapsed_seconds": time.perf_counter() - began,
        "receiver_render_count": receiver_summary["render_count"],
        "donor_render_count": donor_summary["render_count"],
        "initialization_render_count": sum(x["render_count"] for x in init_rows),
        "initialization_model_load_count": sum(x["model_load_count"] for x in init_rows),
        "peak_cuda_memory": peak()}
    decision_path = write_json(output / "adaptive_k_decision.json", decision)
    return decision, decision_path


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("version") == "RD-v1":
        expected = {
            "schema_version": "i2-adaptive-k-policy-v2",
            "policy_id": "i2_adaptive_k_rd",
            "version": "RD-v1",
            "budget_rule": "K=min(R,D)",
            "receiver": {
                "geometry": "receiver_v2_tangent_jacobian_v1",
                "ranking": "photometric_alignment_desc_demand_score_desc_candidate_id_asc",
                "nms": "greedy_spatial_nms_v1",
                "positive_evidence": "photometric_alignment_gt_zero",
                "require_finite_alignment": True,
                "require_positive_set_equals_prefix": True,
                "require_complete_enumeration": True,
            },
            "donor": {
                "ranking": "frozen_d0_order",
                "eligibility": "frozen_d0_b1_structural_eligibility",
                "exclude_receiver_anchors": True,
                "require_complete_enumeration": True,
                "performance_gates_used": False,
            },
            "forbidden_budget_signals": [
                "PSNR", "SSIM", "LPIPS", "test_images", "target_region_L1",
                "global_L1", "deletion_render_delta",
            ],
        }
        if policy != expected or "candidate_budget_cap" in policy:
            raise AdaptiveKError("adaptive policy differs from frozen RD-v1")
        return policy
    expected = {
        "schema_version", "policy_id", "version", "candidate_budget_cap",
        "receiver", "donor", "initialization", "forbidden_budget_signals",
    }
    if set(policy) != expected:
        raise AdaptiveKError("adaptive policy keys do not match R0-v1 schema")
    if (
        policy["schema_version"] != "i2-adaptive-k-policy-v1"
        or policy["policy_id"] != "i2_adaptive_k_r0"
        or policy["version"] != "R0-v1"
        or policy["candidate_budget_cap"] != 20
        or policy["receiver"] != {
            "geometry": "receiver_v2_tangent_jacobian_v1",
            "ranking": "photometric_alignment_desc_demand_score_desc_candidate_id_asc",
            "nms": "greedy_spatial_nms_v1",
            "positive_evidence": "photometric_alignment_gt_zero",
            "require_finite_alignment": True,
            "require_positive_set_equals_prefix": True,
            "audit_all_candidates": True,
        }
        or policy["donor"] != {
            "ranking": "frozen_d0_order",
            "require_individual_safety": True,
            "require_contiguous_joint_prefix": True,
        }
        or policy["initialization"] != {
            "audit_prefixes_from_k": 1,
            "stop_after_first_failure": True,
            "require_contiguous_safe_prefix": True,
        }
        or policy["forbidden_budget_signals"] != ["PSNR", "SSIM", "LPIPS", "test_images"]
    ):
        raise AdaptiveKError("adaptive policy differs from frozen R0-v1")
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-config")
    parser.add_argument("--method-config", default="configs/i2_method_v2.json")
    parser.add_argument("--policy-config", default="configs/i2_adaptive_k_r0_v1.json")
    parser.add_argument("--receiver-manifest", required=True)
    parser.add_argument("--receiver-report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    scene_cfg, method, fingerprints = load_configs(
        args.scene, scene_config=args.scene_config, method_config=args.method_config
    )
    policy_path = resolve_path(args.policy_config)
    assert policy_path is not None
    policy = load_policy(policy_path)
    requested_output = (
        resolve_path(args.output) if args.output else
        ROOT / "reports" / "i2_scene_runs" / scene_cfg["output_namespace"] / (
            "adaptive_rd_v1" if policy["version"] == "RD-v1" else "adaptive_r0_v1"
        )
    )
    assert requested_output is not None
    output = new_output(requested_output)
    log = Log(output / "audit.log")
    checkpoint = checkpoint_sha = binding_path = None
    required_prepare = {}
    try:
        status = preflight(scene_cfg, method, fingerprints)
        if status["status"] != "GO":
            raise PipelineError(
                f"adaptive preparation requires GO preflight: {status['blockers'] or status['source_status']}"
            )
        checkpoint, checkpoint_sha, binding_path = resolve_common_source(scene_cfg, method)
        if checkpoint is None or checkpoint_sha is None:
            raise PipelineError("adaptive preparation has no bound common source")
        cfg_path = checkpoint.parent / "cfg_args"
        receiver_manifest_path = resolve_path(args.receiver_manifest)
        receiver_report_path = resolve_path(args.receiver_report)
        assert receiver_manifest_path is not None and receiver_report_path is not None
        required_prepare = {
            "receiver_manifest": receiver_manifest_path,
            "receiver_report": receiver_report_path,
        }
        missing = [str(path) for path in required_prepare.values() if not path.is_file()]
        if missing:
            raise PipelineError(f"adaptive preparation inputs are missing: {missing}")
        if not torch.cuda.is_available():
            raise PipelineError("CUDA is not available; no GPU audit was performed")
        log(f"{scene_cfg['scene']}: starting {policy['version']} adaptive preparation")
        decision, _ = audit_scene(
            scene_cfg, checkpoint, cfg_path, method, policy,
            receiver_manifest_path, receiver_report_path, output, log
        )
        decision["input_bindings"] = {
            "scene_config": {
                "path": fingerprints["scene_config"],
                "sha256": fingerprints["scene_config_sha256"],
            },
            "method_config": {
                "path": fingerprints["method_config"],
                "sha256": fingerprints["method_config_sha256"],
            },
            "policy_config": {"path": str(policy_path), "sha256": sha256(policy_path)},
            "common_source": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "source_binding": None if binding_path is None else {
                "path": str(binding_path), "sha256": sha256(binding_path)
            },
            "prepare": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in required_prepare.items()
            },
        }
        decision_path = write_json(output / "adaptive_k_decision.json", decision)
        summary = {
            "format": (
                "i2_adaptive_k_rd_v1_run_summary"
                if policy["version"] == "RD-v1" else "i2_adaptive_k_r0_v1_run_summary"
            ),
            "status": "GO" if decision["selected_k"] != "UNKNOWN" else "NO-GO",
            "scene": scene_cfg["scene"], "selected_k": decision["selected_k"],
            "output": str(output), "decision": str(decision_path),
            "decision_sha256": sha256(decision_path), "no_training_performed": True,
            "test_views_used": False,
        }
        write_json(output / "run_summary.json", summary)
        log(json.dumps(summary, indent=2))
        return summary
    except Exception as exc:
        log(f"{scene_cfg['scene']}: NO-GO: {exc}\n{traceback.format_exc()}")
        blocker = {
            "format": (
                "i2_adaptive_k_rd_v1_decision"
                if policy["version"] == "RD-v1" else "i2_adaptive_k_r0_v1_decision"
            ), "policy_version": policy["version"],
            "status": "NO_GO",
            "scene": scene_cfg["scene"], "selected_k": "UNKNOWN",
            "execution_route": "blocked_no_execution", "stop_reason": str(exc),
            "failure_reasons": [str(exc)], "uses_test_views": False,
            "input_bindings": {
                "scene_config": {
                    "path": fingerprints["scene_config"],
                    "sha256": fingerprints["scene_config_sha256"],
                },
                "method_config": {
                    "path": fingerprints["method_config"],
                    "sha256": fingerprints["method_config_sha256"],
                },
                "policy_config": {"path": str(policy_path), "sha256": sha256(policy_path)},
                "common_source": None if checkpoint is None else {
                    "path": str(checkpoint), "sha256": checkpoint_sha
                },
                "source_binding": None if binding_path is None else {
                    "path": str(binding_path), "sha256": sha256(binding_path)
                },
                "prepare": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in required_prepare.items() if path.is_file()
                },
            },
        }
        write_json(output / "adaptive_k_failure.json", blocker)
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
