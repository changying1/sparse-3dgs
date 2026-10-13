"""Optional, event-level diagnostics for Gaussian densification.

This module deliberately avoids CUDA operations.  Tensor shape inspection is used
for optimizer validation, while the caller supplies event counts collected by the
Gaussian model only when diagnostics are enabled.
"""

import csv
import os


EVENT_FIELDS = (
    "iteration",
    "gaussians_before",
    "clone_added",
    "split_parents_selected",
    "split_children_created",
    "split_parents_deleted",
    "opacity_prune_deleted",
    "size_prune_deleted",
    "prune_deleted_total",
    "gaussians_after",
    "expected_gaussians_after",
    "conservation_delta",
    "conservation_ok",
    "optimizer_dimensions_ok",
    "optimizer_dimension_errors",
)


def add_densification_diagnostics_argument(parser):
    """Add the opt-in CLI flag; the baseline default is intentionally false."""
    parser.add_argument(
        "--densification_diagnostics",
        action="store_true",
        default=False,
        help="write per-event Clone/Split/Prune counts and optimizer shape checks",
    )


def build_densification_event(iteration, counts):
    """Build an event row and compute its Gaussian-count conservation result."""
    row = {
        "iteration": int(iteration),
        "gaussians_before": int(counts["gaussians_before"]),
        "clone_added": int(counts["clone_added"]),
        "split_parents_selected": int(counts["split_parents_selected"]),
        "split_children_created": int(counts["split_children_created"]),
        "split_parents_deleted": int(counts["split_parents_deleted"]),
        "opacity_prune_deleted": int(counts["opacity_prune_deleted"]),
        "size_prune_deleted": int(counts.get("size_prune_deleted", 0)),
        "prune_deleted_total": int(counts["prune_deleted_total"]),
        "gaussians_after": int(counts["gaussians_after"]),
    }
    expected = (
        row["gaussians_before"]
        + row["clone_added"]
        + row["split_children_created"]
        - row["split_parents_deleted"]
        - row["prune_deleted_total"]
    )
    row["expected_gaussians_after"] = expected
    row["conservation_delta"] = row["gaussians_after"] - expected
    row["conservation_ok"] = row["conservation_delta"] == 0
    return row


def optimizer_dimension_errors(optimizer, gaussian_count):
    """Return all parameter/Adam-moment first-dimension mismatches."""
    errors = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", str(group_index))
        params = group.get("params", ())
        if not params:
            errors.append("{}: missing parameter".format(group_name))
            continue
        for param_index, parameter in enumerate(params):
            label = "{}[{}]".format(group_name, param_index)
            if len(parameter.shape) == 0 or parameter.shape[0] != gaussian_count:
                errors.append(
                    "{} parameter dim0={} expected={}".format(
                        label,
                        "scalar" if len(parameter.shape) == 0 else parameter.shape[0],
                        gaussian_count,
                    )
                )
            state = optimizer.state.get(parameter)
            if state is None:
                errors.append("{} optimizer state missing".format(label))
                continue
            for state_name in ("exp_avg", "exp_avg_sq"):
                state_tensor = state.get(state_name)
                if state_tensor is None:
                    errors.append("{} {} missing".format(label, state_name))
                elif len(state_tensor.shape) == 0 or state_tensor.shape[0] != gaussian_count:
                    errors.append(
                        "{} {} dim0={} expected={}".format(
                            label,
                            state_name,
                            "scalar" if len(state_tensor.shape) == 0 else state_tensor.shape[0],
                            gaussian_count,
                        )
                    )
    return errors


class DensificationEventLogger:
    """Append checked densification events to one CSV in the experiment folder."""

    def __init__(self, model_path, filename="densification_events.csv"):
        self.path = os.path.join(model_path, filename)

    def record(self, iteration, counts, optimizer):
        row = build_densification_event(iteration, counts)
        errors = optimizer_dimension_errors(optimizer, row["gaussians_after"])
        row["optimizer_dimensions_ok"] = not errors
        row["optimizer_dimension_errors"] = " | ".join(errors)

        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EVENT_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

        if not row["conservation_ok"]:
            raise RuntimeError(
                "Densification count conservation failed at iteration {} (delta={})".format(
                    iteration, row["conservation_delta"]
                )
            )
        if errors:
            raise RuntimeError(
                "Densification optimizer dimension check failed at iteration {}: {}".format(
                    iteration, "; ".join(errors)
                )
            )
        return row
