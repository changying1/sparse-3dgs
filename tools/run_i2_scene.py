#!/usr/bin/env python3
"""Unified, fail-closed CLI for multi-scene Innovation 2 experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.i2_scene_pipeline import (
    ARMS,
    D3_ARMS,
    PipelineError,
    audit_source_checkpoint,
    adaptive_prepare_output,
    adaptive_prepare_command,
    build_adaptive_evidence_plan,
    build_prepare_plan,
    execute_commands,
    execute_training,
    load_configs,
    preflight,
    prepare_paths,
    recover_prepare_commands,
    register_source_binding,
    resolve_common_source,
    resolve_path,
    run_logged,
    sha256,
    source_command,
    training_command,
    _training_output_path,
    validate_external_training_binding,
    validate_adaptive_training_binding,
    validate_adaptive_smoke_metadata,
    validate_initialization_gate,
)


def dump(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def source_stage(scene_cfg, method_cfg, fingerprints, execute):
    status = preflight(scene_cfg, method_cfg, fingerprints)
    if status["status"] == "GO":
        return {"status": "SOURCE_REUSED", "preflight": status, "executed": False}
    if status["status"] != "SOURCE_REQUIRED":
        raise PipelineError(f"Source preflight is NO-GO: {status['blockers']}")
    command, local_pdc, output = source_command(scene_cfg, method_cfg)
    plan = {
        "status": "SOURCE_REQUIRED", "executed": False, "command": command,
        "local_pdc": str(local_pdc), "output": str(output), "fingerprints": fingerprints,
    }
    if not execute:
        return plan
    if output.exists():
        raise PipelineError(f"Refusing to overwrite source output: {output}")
    source_pdc = resolve_path(scene_cfg["pdc"]["path"])
    assert source_pdc is not None
    local_pdc.parent.mkdir(parents=True, exist_ok=True)
    if local_pdc.exists() and sha256(local_pdc) != scene_cfg["pdc"]["sha256"]:
        raise PipelineError(f"Local PDC collision: {local_pdc}")
    if not local_pdc.exists():
        shutil.copyfile(source_pdc, local_pdc)
    output.mkdir(parents=True, exist_ok=False)
    meta = {
        "format": "i2_scene_source_launch_v1", "status": "RUNNING",
        "scene": scene_cfg["scene"], "fingerprints": fingerprints,
        "argv": command, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output / "launch_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    try:
        meta["training_seconds"] = run_logged(command, output / "train.log")
        checkpoint = output / f"chkpnt{method_cfg['source']['iteration']}.pth"
        temporary_scene = json.loads(json.dumps(scene_cfg))
        temporary_scene["common_source"] = {
            "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
        }
        meta["checkpoint_audit"] = audit_source_checkpoint(
            checkpoint, temporary_scene, method_cfg["source"]["iteration"]
        )
        meta["checkpoint_sha256"] = sha256(checkpoint)
        binding_path, binding = register_source_binding(
            scene_cfg, method_cfg, checkpoint, meta["checkpoint_audit"]
        )
        meta["source_binding"] = str(binding_path)
        meta["source_binding_sha256"] = sha256(binding_path)
        meta["a0_status"] = binding["a0_status"]
        meta["b0_status"] = binding["b0_status"]
        meta["status"] = "PASS"
    except Exception as exc:
        meta["status"] = "FAILED"; meta["error"] = repr(exc); raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def prepare_stage(scene_cfg, method_cfg, fingerprints, execute, recover_from=None):
    status = preflight(scene_cfg, method_cfg, fingerprints)
    if status["status"] != "GO":
        raise PipelineError(f"Prepare requires GO preflight: {status['blockers'] or status['source_status']}")
    plan = build_prepare_plan(scene_cfg, method_cfg)
    plan["fingerprints"] = fingerprints
    plan["status"] = "PREFLIGHT_PASS"
    plan["executed"] = False
    if not execute:
        return plan
    root = Path(plan["output_root"])
    manifest = root / "stage_manifest.json"
    if recover_from:
        audit_path, audit = recover_prepare_commands(
            plan["commands"], root, plan["contracts"], start_stage=recover_from,
            final_validator=lambda: validate_initialization_gate(scene_cfg, method_cfg),
        )
        plan["status"] = "RECOVERY_PASS"
        plan["executed"] = True
        plan["recovery_manifest"] = str(audit_path)
        plan["recovery_manifest_sha256"] = sha256(audit_path)
        plan["recovery"] = audit
        return plan
    try:
        execute_commands(plan["commands"], root, plan["contracts"])
        validate_initialization_gate(scene_cfg, method_cfg)
        plan["status"] = "PASS"
        plan["executed"] = True
    except Exception as exc:
        plan["status"] = "FAILED"; plan["error"] = repr(exc); raise
    finally:
        if root.exists():
            manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def adaptive_prepare_stage(scene_cfg, method_cfg, fingerprints, execute,
                           policy_config, output_dir=None):
    policy_path = resolve_path(policy_config)
    if policy_path is None or not policy_path.is_file():
        raise PipelineError(f"Adaptive policy does not exist: {policy_config}")
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_version = policy_payload.get("version")
    if policy_version not in {"R0-v1", "RD-v1"}:
        raise PipelineError("Unsupported adaptive policy version")
    decision_format = (
        "i2_adaptive_k_rd_v1_decision"
        if policy_version == "RD-v1" else "i2_adaptive_k_r0_v1_decision"
    )
    status = preflight(scene_cfg, method_cfg, fingerprints)
    run_output = _training_output_path(output_dir, adaptive_prepare_output(scene_cfg))
    if execute and run_output.exists():
        raise PipelineError(f"Refusing to overwrite adaptive output: {run_output}")
    if status["status"] != "GO":
        failure_reasons = status.get("blockers") or [status.get("source_status", "NO_GO")]
        if not execute:
            return {
                "status": "NO_GO", "preflight": status, "executed": False,
                "scene": scene_cfg["scene"], "stage": "adaptive-prepare",
                "output": str(run_output), "failure_reasons": failure_reasons,
                "fingerprints": fingerprints,
            }
        run_output.mkdir(parents=True, exist_ok=False)
        failure = run_output / "adaptive_prepare_failure.json"
        failure.write_text(json.dumps({
            "format": decision_format, "policy_version": policy_version,
            "scene": scene_cfg["scene"], "selected_k": "UNKNOWN",
            "execution_route": "blocked_no_execution", "uses_test_views": False,
            "failure_reasons": failure_reasons, "status": "NO_GO",
            "preflight": status,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise PipelineError(
            f"Adaptive preflight blocked; failure evidence: {failure}"
        )
    evidence = command = None
    decision_root = run_output / "decision"
    try:
        evidence = build_adaptive_evidence_plan(
            scene_cfg, method_cfg, run_output / "evidence",
            rd_v1=policy_version == "RD-v1",
        )
        command, _ = adaptive_prepare_command(
            scene_cfg, method_cfg, fingerprints, policy_config=policy_config,
            receiver_manifest=evidence["receiver_manifest"],
            receiver_report=evidence["receiver_report"], output_dir=decision_root,
        )
    except Exception as exc:
        if not execute:
            return {
                "status": "NO_GO", "preflight": status, "executed": False,
                "scene": scene_cfg["scene"], "stage": "adaptive-prepare",
                "output": str(run_output), "failure_reasons": [str(exc)],
                "fingerprints": fingerprints,
            }
        run_output.mkdir(parents=True, exist_ok=False)
        failure = run_output / "adaptive_prepare_failure.json"
        failure.write_text(json.dumps({
            "format": decision_format, "policy_version": policy_version,
            "scene": scene_cfg["scene"], "selected_k": "UNKNOWN",
            "execution_route": "blocked_no_execution", "uses_test_views": False,
            "failure_reasons": [str(exc)], "status": "NO_GO",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise PipelineError(
            f"Adaptive evidence contract blocked; failure evidence: {failure}: {exc}"
        ) from exc
    payload = {
        "status": "PREFLIGHT_PASS" if status["status"] == "GO" else "NO_GO",
        "preflight": status, "executed": False,
        "scene": scene_cfg["scene"], "stage": "adaptive-prepare",
        "evidence_plan": evidence, "command": command,
        "output": str(run_output),
        "decision": str(decision_root / "adaptive_k_decision.json"),
        "fingerprints": fingerprints,
    }
    if not execute:
        return payload
    decision = decision_root / "adaptive_k_decision.json"
    try:
        execute_commands(
            evidence["commands"], Path(evidence["output_root"]), evidence["contracts"]
        )
        run_logged(command, run_output / "adaptive_audit.log")
    except Exception as exc:
        if decision.is_file():
            raise PipelineError(
                f"Adaptive preparation blocked; decision: {decision}: {exc}"
            ) from exc
        run_output.mkdir(parents=True, exist_ok=True)
        failure = run_output / "adaptive_prepare_failure.json"
        failure_payload = {
            "format": decision_format, "policy_version": policy_version,
            "scene": scene_cfg["scene"], "selected_k": "UNKNOWN",
            "execution_route": "blocked_no_execution", "uses_test_views": False,
            "failure_reasons": [str(exc)], "status": "NO_GO",
            "evidence_plan": evidence,
        }
        failure.write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise PipelineError(
            f"Adaptive evidence preparation blocked; failure evidence: {failure}: {exc}"
        ) from exc
    try:
        binding = validate_adaptive_training_binding(
            scene_cfg, method_cfg, decision_path=decision
        )
    except Exception as exc:
        failure = run_output / "adaptive_validation_failure.json"
        failure.write_text(json.dumps({
            "format": "i2_adaptive_prepare_validation_failure_v1",
            "status": "NO_GO", "scene": scene_cfg["scene"],
            "decision": str(decision),
            "decision_sha256": sha256(decision) if decision.is_file() else None,
            "failure_reasons": [str(exc)],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise PipelineError(
            f"Adaptive decision validation failed; evidence: {failure}: {exc}"
        ) from exc
    payload.update(
        status="PASS", executed=True, decision=str(decision), output=str(run_output),
        decision_sha256=sha256(decision), selected_k=binding["selected_k"],
    )
    return payload


def _read_initialization_gate(scene_cfg, method_cfg):
    return validate_initialization_gate(scene_cfg, method_cfg)


def training_stage(scene_cfg, method_cfg, fingerprints, *, stage, arm, execute,
                   active_plan=None, requested_budget=None, output_dir=None,
                   initialization_audit=None, smoke_output_dir=None,
                   skip_evaluation=False, adaptive_decision=None,
                   paired_control=False, mechanism_arm=None):
    status = preflight(scene_cfg, method_cfg, fingerprints)
    if status["status"] != "GO":
        raise PipelineError("Training stage requires GO scene preflight")
    if adaptive_decision is not None:
        if active_plan is not None or initialization_audit is not None or requested_budget is not None:
            raise PipelineError(
                "--adaptive-decision cannot be combined with manual Plan, audit or K"
            )
        binding = validate_adaptive_training_binding(
            scene_cfg, method_cfg, decision_path=adaptive_decision
        )
        required_arm = "control" if paired_control else (mechanism_arm or binding["route"])
        if mechanism_arm is not None and (
            mechanism_arm not in D3_ARMS or binding.get("policy_version") != "RD-v1"
        ):
            raise PipelineError("RD-D3 mechanism arms require a frozen RD-v1 Decision")
        if arm != required_arm:
            raise PipelineError(
                f"Adaptive decision requires the {required_arm} arm"
            )
        requested_budget = binding["selected_k"]
        active_plan = None if paired_control else binding["plan"]
        initialization_audit = None if paired_control else binding["audit"]
        init_path = binding["decision"] if binding["audit"] is None else binding["audit"]
    elif active_plan is None:
        if initialization_audit is not None:
            raise PipelineError("--initialization-audit requires --active-plan")
        init_path, _ = _read_initialization_gate(scene_cfg, method_cfg)
    else:
        if initialization_audit is None:
            raise PipelineError("External active Plan requires --initialization-audit")
        _, init_path, _ = validate_external_training_binding(
            scene_cfg, method_cfg, active_plan=active_plan,
            requested_budget=int(
                method_cfg["budget"]["requested_k"]
                if requested_budget is None else requested_budget
            ),
            initialization_audit=initialization_audit,
        )
    command, output, iteration = training_command(
        scene_cfg, method_cfg, stage=stage, arm=arm, active_plan=active_plan,
        requested_budget=requested_budget, output_dir=output_dir,
        adaptive_decision=adaptive_decision,
    )
    if stage == "formal" and not paired_control:
        smoke_arm = mechanism_arm or ("control" if requested_budget == 0 else "receiver_v2")
        smoke_output = _training_output_path(smoke_output_dir, training_command(
            scene_cfg, method_cfg, stage="smoke", arm=smoke_arm,
            active_plan=active_plan, requested_budget=requested_budget,
            adaptive_decision=adaptive_decision,
        )[1])
        smoke_meta = smoke_output / "launch_meta.json"
        smoke = json.loads(smoke_meta.read_text()) if smoke_meta.is_file() else {}
        expected_plan = None if active_plan is None else sha256(resolve_path(active_plan))
        expected_k = int(
            method_cfg["budget"]["requested_k"]
            if requested_budget is None else requested_budget
        )
        fixed_gate_failed = (
            smoke.get("status") != "PASS"
            or smoke.get("requested_k") != expected_k
            or (expected_plan is not None and smoke.get("plan_sha256") != expected_plan)
        )
        if adaptive_decision is not None:
            _, checkpoint_sha, _ = resolve_common_source(scene_cfg, method_cfg)
            expected_decision = sha256(resolve_path(str(adaptive_decision)))
            expected_initialization = (
                None if binding["audit"] is None else sha256(Path(init_path))
            )
            validate_adaptive_smoke_metadata(
                smoke, scene=scene_cfg["scene"], arm=smoke_arm,
                requested_k=expected_k, plan_sha256=expected_plan,
                initialization_audit_sha256=expected_initialization,
                adaptive_decision_sha256=expected_decision,
                source_checkpoint_sha256=checkpoint_sha,
                view_manifest_sha256=scene_cfg["view_manifest"]["sha256"],
                method_config_sha256=fingerprints["method_config_sha256"],
            )
        if fixed_gate_failed:
            raise PipelineError("Formal stage requires the matching PASS smoke run")
    payload = {
        "status": "PREFLIGHT_PASS", "executed": False, "scene": scene_cfg["scene"],
        "stage": stage, "arm": arm, "iteration": iteration, "output": str(output),
        "command": command, "initialization_audit": str(init_path),
        "requested_k": int(method_cfg["budget"]["requested_k"] if requested_budget is None else requested_budget),
        "fingerprints": fingerprints,
        "mechanism_arm": mechanism_arm,
    }
    if output.exists():
        raise PipelineError(f"Output collision: {output}")
    if execute:
        return execute_training(
            scene_cfg, method_cfg, fingerprints, stage=stage, arm=arm,
            active_plan=active_plan, requested_budget=requested_budget,
            output_dir=output_dir, initialization_audit=initialization_audit,
            skip_evaluation=skip_evaluation, adaptive_decision=adaptive_decision,
            paired_control=paired_control, mechanism_arm=mechanism_arm,
        )
    return payload


def report_stage(scene_cfg, method_cfg, fingerprints):
    namespace = scene_cfg["output_namespace"]
    iteration = method_cfg["training"]["final_iteration"]
    k = method_cfg["budget"]["requested_k"]
    rows = {}
    for arm in ARMS:
        _, output, _ = training_command(scene_cfg, method_cfg, stage="formal", arm=arm)
        metrics = output / f"formal_metrics_{iteration}.json"
        if metrics.is_file():
            rows[arm] = json.loads(metrics.read_text(encoding="utf-8"))
    result = {
        "format": "i2_scene_result_v1", "scene": scene_cfg["scene"],
        "requested_k": k, "test_view_count": len(scene_cfg["split"]["test_images"]),
        "fingerprints": fingerprints, "arms": rows,
        "status": "COMPLETE" if len(rows) == 3 else "INCOMPLETE",
        "randomness_caveat": method_cfg["randomness_caveat"],
    }
    report_dir = ROOT / "reports/i2_scene_runs" / namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--stage", required=True, choices=(
        "preflight", "source", "prepare", "adaptive-prepare",
        "smoke", "formal", "adaptive-smoke", "adaptive-formal",
        "adaptive-control-formal", "rd-d3-smoke", "rd-d3-formal", "report",
    ))
    parser.add_argument("--arm", choices=ARMS + ("birth_only",))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scene-config")
    parser.add_argument("--method-config")
    parser.add_argument("--policy-config", default="configs/i2_adaptive_k_r0_v1.json")
    parser.add_argument("--recover-from")
    parser.add_argument("--active-plan")
    parser.add_argument("--initialization-audit")
    parser.add_argument("--adaptive-decision")
    parser.add_argument("--budget-k", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-output-dir")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    scene_cfg, method_cfg, fingerprints = load_configs(
        args.scene, scene_config=args.scene_config, method_config=args.method_config
    )
    if args.stage == "preflight":
        result = preflight(scene_cfg, method_cfg, fingerprints)
    elif args.stage == "source":
        result = source_stage(scene_cfg, method_cfg, fingerprints, args.execute)
    elif args.stage == "prepare":
        if args.recover_from and not args.execute:
            raise PipelineError("--recover-from requires --execute")
        result = prepare_stage(
            scene_cfg, method_cfg, fingerprints, args.execute, args.recover_from
        )
    elif args.stage == "adaptive-prepare":
        if args.recover_from:
            raise PipelineError("--recover-from is not valid for adaptive preparation")
        result = adaptive_prepare_stage(
            scene_cfg, method_cfg, fingerprints, args.execute,
            args.policy_config, args.output_dir,
        )
    elif args.stage in ("smoke", "formal", "adaptive-smoke", "adaptive-formal", "adaptive-control-formal", "rd-d3-smoke", "rd-d3-formal"):
        d3_training = args.stage.startswith("rd-d3-")
        adaptive_training = args.stage.startswith("adaptive-") or d3_training
        if adaptive_training and args.adaptive_decision is None:
            raise PipelineError(
                f"{args.stage} requires --adaptive-decision; K/Plan/gate cannot be supplied manually"
            )
        paired_control = args.stage == "adaptive-control-formal"
        runtime_stage = (
            "formal" if paired_control else
            args.stage.removeprefix("adaptive-").removeprefix("rd-d3-")
        )
        if d3_training and args.arm not in D3_ARMS:
            raise PipelineError("rd-d3 stages require --arm release or birth_only")
        arm = (
            "control" if paired_control else
            (args.arm or ("receiver_v2" if not adaptive_training else
                          validate_adaptive_training_binding(
                              scene_cfg, method_cfg, decision_path=args.adaptive_decision
                          )["route"]))
        )
        result = training_stage(
            scene_cfg, method_cfg, fingerprints, stage=runtime_stage,
            arm=arm, execute=args.execute, active_plan=args.active_plan,
            requested_budget=args.budget_k, output_dir=args.output_dir,
            initialization_audit=args.initialization_audit,
            smoke_output_dir=args.smoke_output_dir,
            skip_evaluation=args.skip_evaluation,
            adaptive_decision=args.adaptive_decision,
            paired_control=paired_control,
            mechanism_arm=args.arm if d3_training else None,
        )
    else:
        result = report_stage(scene_cfg, method_cfg, fingerprints)
    dump(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(json.dumps({"status": "NO_GO", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(2)
