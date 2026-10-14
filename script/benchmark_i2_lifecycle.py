"""Small CPU benchmark for I2 lifecycle validation layers.

This is a diagnostic benchmark, not a pass/fail performance test. It creates
only root metadata and compares the hot-path structural check with the explicit
full-history audit on the same state.
"""

import argparse
import json
import sys
import statistics
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.i2_lifecycle import GaussianLifecycle


def _median_milliseconds(operation, warmups, repeats):
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return statistics.median(samples)


def run_benchmark(sizes, warmups=2, repeats=5):
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    torch.set_num_threads(1)
    results = []
    for count in sizes:
        if count < 0:
            raise ValueError("Gaussian counts must be non-negative")
        lifecycle = GaussianLifecycle.roots(count, torch.device("cpu"))
        light_ms = _median_milliseconds(
            lambda: lifecycle.validate_lightweight(expected_count=count),
            warmups,
            repeats,
        )
        strict_ms = _median_milliseconds(
            lambda: lifecycle.audit_strict(expected_count=count),
            warmups,
            repeats,
        )
        results.append(
            {
                "gaussian_count": count,
                "warmups": warmups,
                "repeats": repeats,
                "lightweight_median_ms": light_ms,
                "strict_median_ms": strict_ms,
                "strict_to_lightweight_ratio": strict_ms / light_ms,
                "lifecycle_statistics": lifecycle.statistics(),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for result in run_benchmark(args.sizes, args.warmups, args.repeats):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
