#!/usr/bin/env python3
"""Build the frozen Receiver-v2-selection/Anchor-Clone K20 plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.i2_active_pilot import (
    V2_SELECTION_ANCHOR_CLONE_METHOD,
    V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT,
    load_active_plan,
)

SOURCE = ROOT / "output/I2_B0_Horns_3v_1201_Growth_z98bpswi/chkpnt1201.pth"
V2_PLAN = ROOT / "reports/i2_receiver_v2_horns_1201_k20/i2_d1_receiver_v2_k20_plan.json"
V2_EVIDENCE = ROOT / "reports/i2_receiver_v2_horns_1201_k20/receiver_v2_geometry_evidence.json"
VIEW_MANIFEST = Path(
    r"D:\3DGS\Binocular3DGS + Dropout\output\A3_A3_Horns_3v_Baseline_5k_FrozenPDC\view_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/i2_v2_selection_anchor_clone_horns_1201_k20"
    / "i2_d1_v2_selection_anchor_clone_k20_plan.json"
)
SOURCE_SHA256 = "1d5ee65d9bb1f7b50b2d8ea6bd277e5b52e513b212d0216e3a205e014b428c1f"
V2_PLAN_SHA256 = "0bcd58837112fcf06cbb9041cda09afb9a97a3b0034845deb24f463f94827f46"
V2_EVIDENCE_SHA256 = "e1c3472cf9fb74d54dfd5c74b45ef27c4c4e1ed8fcf6f41dac6f7d24b22d9c08"
VIEW_MANIFEST_SHA256 = "a43d544c48f4767369ab37cf9ff82106f1d64365cb0e03d129559d297108586b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build(output: Path) -> dict:
    frozen = {
        SOURCE: SOURCE_SHA256,
        V2_PLAN: V2_PLAN_SHA256,
        V2_EVIDENCE: V2_EVIDENCE_SHA256,
        VIEW_MANIFEST: VIEW_MANIFEST_SHA256,
    }
    mismatched = {
        str(path): {"expected": expected, "actual": sha256(path) if path.is_file() else None}
        for path, expected in frozen.items()
        if not path.is_file() or sha256(path) != expected
    }
    if mismatched:
        raise RuntimeError(f"Frozen input mismatch: {mismatched}")

    original = json.loads(V2_PLAN.read_text(encoding="utf-8"))
    plan = copy.deepcopy(original)
    plan["format"] = V2_SELECTION_ANCHOR_CLONE_PLAN_FORMAT
    plan["method_version"] = V2_SELECTION_ANCHOR_CLONE_METHOD
    plan["receiver_v2_selection_anchor_clone"] = {
        "method_version": V2_SELECTION_ANCHOR_CLONE_METHOD,
        "original_receiver_v2_plan": artifact(V2_PLAN),
        "receiver_v2_geometry_evidence": artifact(V2_EVIDENCE),
        "training_view_manifest": artifact(VIEW_MANIFEST),
        "initialization": {
            "xyz": "anchor_raw__xyz",
            "opacity": "anchor_raw__opacity",
            "scale": "anchor_raw__scaling",
            "rotation": "anchor_raw__rotation",
            "sh_dc": "anchor_raw__features_dc",
            "sh_rest": "anchor_raw__features_rest",
        },
        "test_views_used": False,
    }
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Refusing to overwrite a different plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")

    verified = load_active_plan(output, requested_budget=20, source_checkpoint=SOURCE)
    evidence = json.loads(V2_EVIDENCE.read_text(encoding="utf-8"))
    evidence_by_id = {row["candidate_id"]: row for row in evidence["rows"]}
    receiver_ids = verified["_receiver_candidate_ids"]
    anchor_uids = verified["_anchor_uids"]
    expected_anchors = [int(evidence_by_id[candidate]["anchor_uid"]) for candidate in receiver_ids]
    audit = {
        "format": "i2_v2_selection_anchor_clone_plan_correspondence_v1",
        "status": "PASS",
        "source_checkpoint": artifact(SOURCE),
        "original_receiver_v2_plan": artifact(V2_PLAN),
        "new_anchor_clone_plan": artifact(output),
        "receiver_v2_geometry_evidence": artifact(V2_EVIDENCE),
        "training_view_manifest": artifact(VIEW_MANIFEST),
        "donor_uids_identical_and_ordered": (
            verified["_donor_uids"] == [int(row["donor_uid"]) for row in original["matches"]]
        ),
        "receiver_candidate_ids_identical_and_ordered": (
            receiver_ids == [str(row["receiver_candidate_id"]) for row in original["matches"]]
        ),
        "anchor_uids_identical_and_ordered": anchor_uids == expected_anchors,
        "donor_counterfactual_unchanged": plan["donor_counterfactual"] == original["donor_counterfactual"],
        "joint_counterfactual_unchanged": (
            plan["joint_deletion_counterfactual"] == original["joint_deletion_counterfactual"]
        ),
        "only_top_level_changes": [
            "format", "method_version", "receiver_v2_selection_anchor_clone"
        ],
        "test_views_used": False,
    }
    if not all(
        audit[key] is True for key in (
            "donor_uids_identical_and_ordered",
            "receiver_candidate_ids_identical_and_ordered",
            "anchor_uids_identical_and_ordered",
            "donor_counterfactual_unchanged",
            "joint_counterfactual_unchanged",
        )
    ):
        raise RuntimeError("Strict v2/Anchor-Clone plan correspondence audit failed")
    audit_path = output.parent / "plan_correspondence_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
