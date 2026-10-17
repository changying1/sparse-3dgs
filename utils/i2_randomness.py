"""Small helpers for common-random-number I2 ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReleaseOnlyDropoutReference:
    """Track the release-only draw length and its first growth extension."""

    initial_count: int
    first_extension_iteration: int | None = None
    first_extension_gaussian_count: int | None = None

    def __post_init__(self):
        self.initial_count = int(self.initial_count)
        if self.initial_count < 0:
            raise ValueError("Initial Dropout reference count must be non-negative")

    def resolve(self, current_count: int, iteration: int) -> int:
        current_count = int(current_count)
        iteration = int(iteration)
        if current_count < 0:
            raise ValueError("Current Gaussian count must be non-negative")
        if iteration < 0:
            raise ValueError("Iteration must be non-negative")
        if (
            current_count > self.initial_count
            and self.first_extension_iteration is None
        ):
            self.first_extension_iteration = iteration
            self.first_extension_gaussian_count = current_count
        return max(self.initial_count, current_count)


def aligned_uniform_prefix(count: int, reference_count: int | None, *, device):
    """Draw the reference stream and expose the prefix used by live rows.

    A release-only arm has fewer live rows.  Drawing the pre-release count keeps
    the CUDA generator aligned with a budget-conserving arm while assigning the
    same samples to their common, compacted survivor prefix.
    """
    count = int(count)
    if reference_count is None:
        reference_count = count
    reference_count = int(reference_count)
    if count < 0 or reference_count < count:
        raise ValueError("Dropout reference count must be at least the live Gaussian count")
    draws = torch.rand(reference_count, device=device)
    return draws[:count], draws
