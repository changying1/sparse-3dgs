"""Build a read-only I2-D0 resource reallocation plan from a D manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.i2_d0_dry_run import build_d0_plan, load_d0_inputs, write_d0_outputs


def run(manifest_path, output_dir, *, budget=500, min_age=100, min_opportunities=1,
        excluded_uids=()):
    manifest, b1_rows, c_rows, receiver_rows = load_d0_inputs(manifest_path)
    plan = build_d0_plan(
        manifest, b1_rows, c_rows, receiver_rows, budget=budget,
        min_age=min_age, min_opportunities=min_opportunities,
        excluded_uids=excluded_uids,
    )
    write_d0_outputs(plan, output_dir)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--min-age", type=int, default=100)
    parser.add_argument("--min-opportunities", type=int, default=1)
    parser.add_argument("--exclude-uid", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    plan = run(
        args.manifest, args.output_dir, budget=args.budget,
        min_age=args.min_age, min_opportunities=args.min_opportunities,
        excluded_uids=args.exclude_uid,
    )
    print(json.dumps({
        "strategy": plan["strategy"],
        "donor_capacity": plan["donors"]["capacity"],
        "receiver_status": plan["receivers"]["status"],
        "matched_feasible_range": plan["budget_audit"]["matched_feasible_range"],
        "pilot_minimum_safety_ready": plan["pilot_minimum_safety_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
