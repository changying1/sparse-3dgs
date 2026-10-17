"""Opt-in C-stage evidence capture for growth-created Gaussians.

The tracker reuses A0 UIDs and B0 observation counters.  Training-time work is
vectorized; host transfer occurs only for births, fixed-age crossings, deaths,
and checkpoint/export boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.i2_lifecycle import CODE_TO_BIRTH_TYPE, BIRTH_TYPE_TO_CODE


FUTURE_UTILITY_CHECKPOINT_VERSION = 2
SNAPSHOT_AGES = (100, 500)


def add_i2_future_utility_arguments(parser) -> None:
    parser.add_argument(
        "--enable_i2_future_utility_tracking",
        action="store_true",
        default=False,
        help="Capture opt-in age100/age500 and terminal evidence for I2-C.",
    )


def validate_i2_future_utility_configuration(
    enabled: bool, lifecycle_enabled: bool, observation_enabled: bool
) -> None:
    if enabled and not (lifecycle_enabled and observation_enabled):
        raise ValueError(
            "--enable_i2_future_utility_tracking requires both A0 lifecycle "
            "and B0 observation tracking"
        )


@dataclass
class GaussianFutureUtility:
    """Persistent newborn registry plus row-aligned snapshot latches."""

    tracked_cohort: torch.Tensor
    snapshot_taken: torch.Tensor
    records: dict[int, dict]
    # Host-side scheduling metadata.  This deliberately avoids a CUDA
    # ``nonzero``/``item`` probe on ordinary iterations.
    next_due_iteration: int | None = None

    @classmethod
    def empty(cls, count: int, device: torch.device) -> "GaussianFutureUtility":
        return cls(
            tracked_cohort=torch.zeros(count, dtype=torch.bool, device=device),
            snapshot_taken=torch.zeros(
                (count, len(SNAPSHOT_AGES)), dtype=torch.bool, device=device
            ),
            records={},
        )

    def validate_lightweight(self, expected_count: int) -> None:
        if (
            not torch.is_tensor(self.tracked_cohort)
            or self.tracked_cohort.dtype != torch.bool
            or self.tracked_cohort.ndim != 1
            or self.tracked_cohort.numel() != expected_count
        ):
            raise ValueError("Future-utility cohort mask is not row-aligned")
        if (
            not torch.is_tensor(self.snapshot_taken)
            or self.snapshot_taken.dtype != torch.bool
            or self.snapshot_taken.ndim != 2
            or self.snapshot_taken.shape != (expected_count, len(SNAPSHOT_AGES))
        ):
            raise ValueError("Future-utility snapshot latches are not row-aligned")
        if not isinstance(self.records, dict):
            raise ValueError("Future-utility registry must be a dict")

    def audit_strict(self, expected_count: int) -> None:
        """Run value-scanning checks only at checkpoint/export boundaries."""
        self.validate_lightweight(expected_count)
        if bool(torch.any(self.snapshot_taken & ~self.tracked_cohort[:, None]).item()):
            raise ValueError("Untracked Gaussian rows cannot have C snapshots")

    def append_births(
        self,
        *,
        child_uid: torch.Tensor,
        birth_iter: torch.Tensor,
        birth_type: torch.Tensor,
        source_uid: torch.Tensor,
        generation: torch.Tensor,
        xyz: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        previous_count: int,
    ) -> None:
        child_count = int(child_uid.numel())
        self.validate_lightweight(previous_count)
        if child_count == 0:
            return
        packed = torch.cat(
            (
                child_uid[:, None].to(torch.float64),
                birth_iter[:, None].to(torch.float64),
                birth_type[:, None].to(torch.float64),
                source_uid[:, None].to(torch.float64),
                generation[:, None].to(torch.float64),
                xyz.to(torch.float64),
                opacity.reshape(-1, 1).to(torch.float64),
                scaling.to(torch.float64),
            ),
            dim=1,
        ).detach().cpu().tolist()
        for values in packed:
            uid = int(values[0])
            code = int(values[2])
            if code not in CODE_TO_BIRTH_TYPE or code == BIRTH_TYPE_TO_CODE["root"]:
                raise ValueError("C tracking accepts clone/split newborns only")
            if uid in self.records:
                raise ValueError(f"Duplicate future-utility newborn UID: {uid}")
            self.records[uid] = {
                "uid": uid,
                "birth_iteration": int(values[1]),
                "birth_type": CODE_TO_BIRTH_TYPE[code],
                "source_uid": int(values[3]),
                "generation": int(values[4]),
                "birth_xyz": [float(value) for value in values[5:8]],
                "birth_opacity": float(values[8]),
                "birth_scaling": [float(value) for value in values[9:12]],
                "age_snapshots": {},
                "alive": True,
                "death_iteration": None,
                "death_reason": None,
                "terminal_state": None,
            }
            first_due = int(values[1]) + SNAPSHOT_AGES[0]
            if self.next_due_iteration is None or first_due < self.next_due_iteration:
                self.next_due_iteration = first_due
        self.tracked_cohort = torch.cat(
            (
                self.tracked_cohort,
                torch.ones(child_count, dtype=torch.bool, device=self.tracked_cohort.device),
            )
        )
        self.snapshot_taken = torch.cat(
            (
                self.snapshot_taken,
                torch.zeros(
                    (child_count, len(SNAPSHOT_AGES)),
                    dtype=torch.bool,
                    device=self.snapshot_taken.device,
                ),
            ),
            dim=0,
        )

    def capture_due(
        self, *, lifecycle, observation, xyz, opacity, scaling, iteration: int
    ) -> None:
        count = int(lifecycle.uid.numel())
        self.validate_lightweight(count)
        if self.next_due_iteration is None or int(iteration) < self.next_due_iteration:
            return
        ages = int(iteration) - lifecycle.birth_iter
        thresholds = torch.tensor(SNAPSHOT_AGES, dtype=torch.int64, device=ages.device)
        due = self.tracked_cohort[:, None] & (ages[:, None] >= thresholds[None, :])
        due &= ~self.snapshot_taken
        pairs = due.nonzero(as_tuple=False)
        if pairs.numel() == 0:
            return
        rows = pairs[:, 0]
        slots = pairs[:, 1]
        unique_views = observation.unique_screen_supported_real_views
        packed = torch.cat(
            (
                lifecycle.uid[rows, None].to(torch.float64),
                ages[rows, None].to(torch.float64),
                observation.real_view_opportunities[rows, None].to(torch.float64),
                observation.dropout_retained_hits[rows, None].to(torch.float64),
                observation.screen_supported_hits[rows, None].to(torch.float64),
                unique_views[rows, None].to(torch.float64),
                xyz[rows].to(torch.float64),
                opacity[rows].reshape(-1, 1).to(torch.float64),
                scaling[rows].to(torch.float64),
                slots[:, None].to(torch.float64),
            ),
            dim=1,
        ).detach().cpu().tolist()
        for values in packed:
            uid = int(values[0])
            record = self.records.get(uid)
            if record is None or not record["alive"]:
                raise ValueError("Due C snapshot UID is missing from the live registry")
            slot = int(values[13])
            key = str(SNAPSHOT_AGES[slot])
            record["age_snapshots"][key] = self._state_record(
                record=record,
                iteration=iteration,
                age=int(values[1]),
                opportunities=int(values[2]),
                retained=int(values[3]),
                supported=int(values[4]),
                unique_views=int(values[5]),
                xyz=values[6:9],
                opacity=float(values[9]),
                scaling=values[10:13],
            )
        self.snapshot_taken[rows, slots] = True
        self._refresh_next_due_iteration()

    def _refresh_next_due_iteration(self) -> None:
        due = []
        for record in self.records.values():
            if not record.get("alive"):
                continue
            snapshots = record.get("age_snapshots") or {}
            birth = int(record["birth_iteration"])
            for age in SNAPSHOT_AGES:
                if str(age) not in snapshots:
                    due.append(birth + age)
                    break
        self.next_due_iteration = min(due) if due else None

    def record_deaths(
        self,
        *,
        mask,
        lifecycle,
        observation,
        xyz,
        opacity,
        scaling,
        iteration: int,
        reason: str,
    ) -> None:
        count = int(lifecycle.uid.numel())
        self.validate_lightweight(count)
        rows = (mask & self.tracked_cohort).nonzero(as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            return
        unique_views = observation.unique_screen_supported_real_views
        ages = int(iteration) - lifecycle.birth_iter[rows]
        packed = torch.cat(
            (
                lifecycle.uid[rows, None].to(torch.float64),
                ages[:, None].to(torch.float64),
                observation.real_view_opportunities[rows, None].to(torch.float64),
                observation.dropout_retained_hits[rows, None].to(torch.float64),
                observation.screen_supported_hits[rows, None].to(torch.float64),
                unique_views[rows, None].to(torch.float64),
                xyz[rows].to(torch.float64),
                opacity[rows].reshape(-1, 1).to(torch.float64),
                scaling[rows].to(torch.float64),
            ),
            dim=1,
        ).detach().cpu().tolist()
        for values in packed:
            uid = int(values[0])
            record = self.records.get(uid)
            if record is None or not record["alive"]:
                raise ValueError("Pruned C UID is missing from the live registry")
            record["terminal_state"] = self._state_record(
                record=record,
                iteration=iteration,
                age=int(values[1]),
                opportunities=int(values[2]),
                retained=int(values[3]),
                supported=int(values[4]),
                unique_views=int(values[5]),
                xyz=values[6:9],
                opacity=float(values[9]),
                scaling=values[10:13],
            )
            record["alive"] = False
            record["death_iteration"] = int(iteration)
            record["death_reason"] = str(reason)
        self._refresh_next_due_iteration()

    def prune(self, mask: torch.Tensor) -> None:
        self.tracked_cohort = self.tracked_cohort[~mask]
        self.snapshot_taken = self.snapshot_taken[~mask]

    @staticmethod
    def _state_record(
        *, record, iteration, age, opportunities, retained, supported,
        unique_views, xyz, opacity, scaling
    ):
        birth_xyz = torch.tensor(record["birth_xyz"], dtype=torch.float64)
        current_xyz = torch.tensor(xyz, dtype=torch.float64)
        birth_scale = torch.tensor(record["birth_scaling"], dtype=torch.float64)
        current_scale = torch.tensor(scaling, dtype=torch.float64)
        birth_scale_mean = float(birth_scale.mean().item())
        scale_mean = float(current_scale.mean().item())
        birth_opacity = float(record["birth_opacity"])
        return {
            "iteration": int(iteration),
            "age": int(age),
            "real_view_opportunities": int(opportunities),
            "dropout_retained_hits": int(retained),
            "screen_supported_hits": int(supported),
            "unique_real_views": int(unique_views),
            "visible_rate": None if opportunities == 0 else supported / opportunities,
            "screen_support_rate": None if retained == 0 else supported / retained,
            "opacity": float(opacity),
            "opacity_retention_ratio": (
                None if birth_opacity == 0 else float(opacity) / birth_opacity
            ),
            "xyz": [float(value) for value in xyz],
            "xyz_shift_from_birth": float(
                torch.linalg.vector_norm(current_xyz - birth_xyz).item()
            ),
            "scaling": [float(value) for value in scaling],
            "scale_mean": scale_mean,
            "scale_ratio_from_birth": (
                None if birth_scale_mean == 0 else scale_mean / birth_scale_mean
            ),
        }

    def state_dict(self, *, lifecycle_uid: torch.Tensor) -> dict:
        self.audit_strict(int(lifecycle_uid.numel()))
        live_registry = {uid for uid, record in self.records.items() if record["alive"]}
        tracked_uid = set(
            lifecycle_uid[self.tracked_cohort].detach().cpu().tolist()
        )
        if live_registry != tracked_uid:
            raise ValueError("Live C registry UIDs do not exactly match cohort rows")
        self._validate_registry_latches(lifecycle_uid)
        return {
            "version": FUTURE_UTILITY_CHECKPOINT_VERSION,
            "snapshot_ages": list(SNAPSHOT_AGES),
            "uid": lifecycle_uid,
            "tracked_cohort": self.tracked_cohort,
            "snapshot_taken": self.snapshot_taken,
            "records": [self.records[uid] for uid in sorted(self.records)],
        }

    def _validate_registry_latches(self, lifecycle_uid: torch.Tensor) -> None:
        uid_rows = lifecycle_uid.detach().cpu().tolist()
        latches = self.snapshot_taken.detach().cpu()
        tracked = self.tracked_cohort.detach().cpu().tolist()
        for row, uid in enumerate(uid_rows):
            if not tracked[row]:
                continue
            record = self.records.get(int(uid))
            if record is None or not record.get("alive"):
                raise ValueError("Tracked C row is missing a live registry record")
            snapshots = record.get("age_snapshots") or {}
            expected = [str(age) in snapshots for age in SNAPSHOT_AGES]
            if latches[row].tolist() != expected:
                raise ValueError("C snapshot latches disagree with registry snapshots")

    @classmethod
    def from_state_dict(cls, state, *, lifecycle_uid, device):
        if not isinstance(state, dict) or state.get("version") != FUTURE_UTILITY_CHECKPOINT_VERSION:
            raise ValueError("Unsupported or malformed future-utility checkpoint state")
        if tuple(state.get("snapshot_ages") or ()) != SNAPSHOT_AGES:
            raise ValueError("Future-utility checkpoint snapshot ages do not match")
        checkpoint_uid = state.get("uid")
        tracked = state.get("tracked_cohort")
        latches = state.get("snapshot_taken")
        records = state.get("records")
        if not torch.is_tensor(checkpoint_uid) or checkpoint_uid.dtype != torch.int64:
            raise ValueError("Future-utility checkpoint UID tensor is malformed")
        if not torch.equal(checkpoint_uid.to(device=device), lifecycle_uid):
            raise ValueError("Future-utility UID rows do not match lifecycle UID rows")
        if (
            not torch.is_tensor(tracked)
            or tracked.dtype != torch.bool
            or tracked.ndim != 1
            or tracked.numel() != lifecycle_uid.numel()
        ):
            raise ValueError("Future-utility cohort mask is malformed")
        if (
            not torch.is_tensor(latches)
            or latches.dtype != torch.bool
            or latches.shape != (lifecycle_uid.numel(), len(SNAPSHOT_AGES))
        ):
            raise ValueError("Future-utility snapshot latches are malformed")
        if not isinstance(records, list):
            raise ValueError("Future-utility checkpoint records are malformed")
        registry = {}
        for source in records:
            if not isinstance(source, dict) or "uid" not in source:
                raise ValueError("Future-utility record is malformed")
            uid = int(source["uid"])
            if uid in registry:
                raise ValueError("Future-utility checkpoint contains duplicate UIDs")
            record = dict(source)
            record["age_snapshots"] = {
                str(age): dict(snapshot)
                for age, snapshot in source.get("age_snapshots", {}).items()
            }
            record["terminal_state"] = (
                None if source.get("terminal_state") is None
                else dict(source["terminal_state"])
            )
            registry[uid] = record
        tracker = cls(
            tracked.to(device=device).clone(),
            latches.to(device=device).clone(),
            registry,
        )
        tracker.audit_strict(int(lifecycle_uid.numel()))
        live_registry = {uid for uid, record in registry.items() if record.get("alive")}
        tracked_uid = set(lifecycle_uid[tracker.tracked_cohort].detach().cpu().tolist())
        if live_registry != tracked_uid:
            raise ValueError("Restored live C registry does not match cohort rows")
        tracker._validate_registry_latches(lifecycle_uid)
        tracker._refresh_next_due_iteration()
        return tracker
