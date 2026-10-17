"""Opt-in Gaussian lifecycle metadata for Innovation 2.

Hot-path mutations use metadata-only structural checks. Full history auditing is
reserved for explicit diagnostics and checkpoint boundaries.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List

import torch


ROOT_SOURCE_UID = -1
LIFECYCLE_CHECKPOINT_VERSION = 1

BIRTH_TYPE_TO_CODE = {"root": 0, "clone": 1, "split": 2}
CODE_TO_BIRTH_TYPE = {value: key for key, value in BIRTH_TYPE_TO_CODE.items()}
DEATH_REASONS = {"split_replaced", "training_prune", "i2_active_release"}


def add_i2_lifecycle_arguments(parser) -> None:
    parser.add_argument(
        "--enable_i2_lifecycle_tracking",
        action="store_true",
        default=False,
        help="Track stable Gaussian identities and lineage (I2-A0 only).",
    )


@dataclass
class GaussianLifecycle:
    """Row-aligned lifecycle state for the currently alive Gaussians."""

    uid: torch.Tensor
    birth_iter: torch.Tensor
    birth_type: torch.Tensor
    source_uid: torch.Tensor
    generation: torch.Tensor
    next_uid: int
    events: List[Dict]

    @classmethod
    def roots(cls, count: int, device: torch.device) -> "GaussianLifecycle":
        if count < 0:
            raise ValueError("Gaussian count cannot be negative")
        uid = torch.arange(count, dtype=torch.int64, device=device)
        state = cls(
            uid=uid,
            birth_iter=torch.zeros(count, dtype=torch.int64, device=device),
            birth_type=torch.full(
                (count,), BIRTH_TYPE_TO_CODE["root"], dtype=torch.uint8, device=device
            ),
            source_uid=torch.full(
                (count,), ROOT_SOURCE_UID, dtype=torch.int64, device=device
            ),
            generation=torch.zeros(count, dtype=torch.int64, device=device),
            next_uid=count,
            events=[],
        )
        state._record_birth(0, "root", uid, state.source_uid)
        state.validate_lightweight(expected_count=count)
        return state

    def append_children(
        self, parent_mask: torch.Tensor, birth_type: str, iteration: int, repeats: int = 1
    ) -> torch.Tensor:
        """Append children in the same order as ``tensor[parent_mask].repeat``."""
        self.validate_lightweight()
        if birth_type not in ("clone", "split"):
            raise ValueError("Children must have birth_type 'clone' or 'split'")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Lifecycle iteration must be a non-negative integer")
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        self._validate_mask(parent_mask)

        parent_uid = self.uid[parent_mask].repeat(repeats)
        parent_generation = self.generation[parent_mask].repeat(repeats)
        child_count = parent_uid.numel()
        child_uid = torch.arange(
            self.next_uid,
            self.next_uid + child_count,
            dtype=torch.int64,
            device=self.uid.device,
        )
        self.next_uid += child_count

        self.uid = torch.cat((self.uid, child_uid))
        self.birth_iter = torch.cat(
            (
                self.birth_iter,
                torch.full(
                    (child_count,), iteration, dtype=torch.int64, device=self.uid.device
                ),
            )
        )
        self.birth_type = torch.cat(
            (
                self.birth_type,
                torch.full(
                    (child_count,),
                    BIRTH_TYPE_TO_CODE[birth_type],
                    dtype=torch.uint8,
                    device=self.uid.device,
                ),
            )
        )
        self.source_uid = torch.cat((self.source_uid, parent_uid))
        self.generation = torch.cat((self.generation, parent_generation + 1))
        self._record_birth(iteration, birth_type, child_uid, parent_uid)
        self.validate_lightweight()
        return child_uid

    def append_children_from_rows(
        self, parent_rows: torch.Tensor, birth_type: str, iteration: int
    ) -> torch.Tensor:
        """Append intervention children from explicit rows, allowing repeats.

        Native clone/split continues to use :meth:`append_children`; this
        separate entry point makes the active path auditable without changing
        frozen densification decisions.
        """
        self.validate_lightweight()
        if birth_type not in ("clone", "split"):
            raise ValueError("Children must have birth_type 'clone' or 'split'")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Lifecycle iteration must be a non-negative integer")
        if not torch.is_tensor(parent_rows) or parent_rows.dtype != torch.int64 or parent_rows.ndim != 1:
            raise ValueError("parent_rows must be a one-dimensional int64 tensor")
        parent_rows = parent_rows.to(device=self.uid.device)
        if parent_rows.numel() and bool(
            torch.any((parent_rows < 0) | (parent_rows >= self.uid.numel())).item()
        ):
            raise ValueError("parent_rows contains an out-of-range row")
        parent_uid = self.uid[parent_rows]
        parent_generation = self.generation[parent_rows]
        child_count = int(parent_rows.numel())
        child_uid = torch.arange(
            self.next_uid, self.next_uid + child_count,
            dtype=torch.int64, device=self.uid.device,
        )
        self.next_uid += child_count
        self.uid = torch.cat((self.uid, child_uid))
        self.birth_iter = torch.cat((self.birth_iter, torch.full(
            (child_count,), iteration, dtype=torch.int64, device=self.uid.device
        )))
        self.birth_type = torch.cat((self.birth_type, torch.full(
            (child_count,), BIRTH_TYPE_TO_CODE[birth_type],
            dtype=torch.uint8, device=self.uid.device
        )))
        self.source_uid = torch.cat((self.source_uid, parent_uid))
        self.generation = torch.cat((self.generation, parent_generation + 1))
        self._record_birth(iteration, birth_type, child_uid, parent_uid)
        self.validate_lightweight()
        return child_uid

    def prune(self, mask: torch.Tensor, reason: str, iteration: int) -> torch.Tensor:
        """Record deaths and apply exactly the same row mask as Gaussian pruning."""
        self.validate_lightweight()
        if reason not in DEATH_REASONS:
            raise ValueError(f"Unknown lifecycle death reason: {reason}")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Lifecycle iteration must be a non-negative integer")
        self._validate_mask(mask)
        dead_uid = self.uid[mask].clone()
        if dead_uid.numel():
            self.events.append(
                {
                    "sequence": len(self.events),
                    "iteration": iteration,
                    "event_type": "death",
                    "reason": reason,
                    "uids": dead_uid.detach().cpu(),
                }
            )
        keep = ~mask
        self.uid = self.uid[keep]
        self.birth_iter = self.birth_iter[keep]
        self.birth_type = self.birth_type[keep]
        self.source_uid = self.source_uid[keep]
        self.generation = self.generation[keep]
        self.validate_lightweight()
        return dead_uid

    def observation_update_interface(
        self, full_observed_mask: torch.Tensor, iteration: int, *, is_real_view: bool
    ) -> None:
        """Validate a future evidence update without accumulating any score."""
        self.validate_lightweight()
        if not is_real_view:
            raise ValueError("Lifecycle evidence updates require a real training view")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Lifecycle iteration must be a non-negative integer")
        self._validate_mask(full_observed_mask)

    def validate_lightweight(self, expected_count: int | None = None) -> None:
        """Check row structure without scanning values or event history."""
        tensor_specs = (
            ("uid", self.uid, torch.int64),
            ("birth_iter", self.birth_iter, torch.int64),
            ("birth_type", self.birth_type, torch.uint8),
            ("source_uid", self.source_uid, torch.int64),
            ("generation", self.generation, torch.int64),
        )
        device = self.uid.device
        lengths = set()
        for name, tensor, dtype in tensor_specs:
            if not torch.is_tensor(tensor) or tensor.ndim != 1:
                raise ValueError(f"Lifecycle {name} must be a one-dimensional tensor")
            if tensor.dtype != dtype:
                raise ValueError(f"Lifecycle {name} has an invalid dtype")
            if tensor.device != device:
                raise ValueError("Lifecycle live tensors must share one device")
            lengths.add(tensor.numel())
        if len(lengths) != 1:
            raise ValueError("Lifecycle arrays are not row-aligned")
        count = self.uid.numel()
        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"Lifecycle row count {count} does not match Gaussian count {expected_count}"
            )
        if type(self.next_uid) is not int or self.next_uid < 0:
            raise ValueError("next_uid must be a non-negative integer")
        if not isinstance(self.events, list):
            raise ValueError("Lifecycle event history must be a list")

    def audit_strict(self, expected_count: int | None = None) -> None:
        """Perform an explicit, exhaustive audit of live state and full history."""
        self.validate_lightweight(expected_count=expected_count)

        live_uid = self.uid.detach().cpu().tolist()
        live_birth_iter = self.birth_iter.detach().cpu().tolist()
        live_birth_type = self.birth_type.detach().cpu().tolist()
        live_source_uid = self.source_uid.detach().cpu().tolist()
        live_generation = self.generation.detach().cpu().tolist()

        if len(set(live_uid)) != len(live_uid):
            raise ValueError("Live Gaussian UIDs are not unique")
        if any(uid < 0 or uid >= self.next_uid for uid in live_uid):
            raise ValueError("Live Gaussian UID is outside the allocated range")

        births = {}
        deaths = {}
        split_births_by_parent_and_iteration = set()
        previous_iteration = -1
        for sequence, event in enumerate(self.events):
            if not isinstance(event, dict) or event.get("sequence") != sequence:
                raise ValueError("Lifecycle event sequence is malformed")
            iteration = event.get("iteration")
            if type(iteration) is not int or iteration < 0:
                raise ValueError("Lifecycle event iteration is malformed")
            if iteration < previous_iteration:
                raise ValueError("Lifecycle event iterations move backwards")
            previous_iteration = iteration
            event_type = event.get("event_type")
            if event_type not in {"birth", "death"}:
                raise ValueError("Lifecycle event type is malformed")
            event_uids = self._audit_event_tensor(event.get("uids"), "UIDs")

            if event_type == "birth":
                birth_type = event.get("birth_type")
                if birth_type not in BIRTH_TYPE_TO_CODE:
                    raise ValueError("Lifecycle birth event type is malformed")
                source_uids = self._audit_event_tensor(
                    event.get("source_uids"), "parent UIDs"
                )
                if len(source_uids) != len(event_uids):
                    raise ValueError("Lifecycle birth event parent UIDs are misaligned")
                for uid, source_uid in zip(event_uids, source_uids):
                    if uid in births:
                        raise ValueError("Lifecycle history contains a duplicate birth UID")
                    if birth_type == "root":
                        if source_uid != ROOT_SOURCE_UID or iteration != 0:
                            raise ValueError("Lifecycle root birth metadata is invalid")
                        generation = 0
                    else:
                        if source_uid not in births or source_uid >= uid:
                            raise ValueError("Lifecycle child references an invalid parent")
                        if source_uid in deaths:
                            raise ValueError(
                                "Lifecycle child references a parent that was already dead"
                            )
                        generation = births[source_uid]["generation"] + 1
                        if birth_type == "split":
                            split_births_by_parent_and_iteration.add(
                                (source_uid, iteration)
                            )
                    births[uid] = {
                        "iteration": iteration,
                        "birth_type": birth_type,
                        "source_uid": source_uid,
                        "generation": generation,
                    }
            else:
                reason = event.get("reason")
                if reason not in DEATH_REASONS:
                    raise ValueError("Lifecycle death event reason is malformed")
                for uid in event_uids:
                    if uid not in births:
                        raise ValueError("Lifecycle death references an unallocated UID")
                    if uid in deaths:
                        raise ValueError("A Gaussian UID appears in more than one death event")
                    if iteration < births[uid]["iteration"]:
                        raise ValueError("Lifecycle death predates Gaussian birth")
                    if (
                        reason == "split_replaced"
                        and (uid, iteration)
                        not in split_births_by_parent_and_iteration
                    ):
                        raise ValueError(
                            "Split-replaced parent has no preceding split child "
                            "birth in the same iteration"
                        )
                    deaths[uid] = event

        expected_uids = set(range(self.next_uid))
        if set(births) != expected_uids:
            raise ValueError("Lifecycle birth history has duplicate or missing UIDs")
        live_set = set(live_uid)
        death_set = set(deaths)
        if live_set & death_set:
            raise ValueError("Live Gaussian UIDs overlap the death history")
        if live_set | death_set != expected_uids:
            raise ValueError("Allocated UIDs are neither alive nor recorded dead")

        valid_codes = set(BIRTH_TYPE_TO_CODE.values())
        for row, uid in enumerate(live_uid):
            history = births[uid]
            if live_birth_type[row] not in valid_codes:
                raise ValueError("Lifecycle birth_type contains an unknown code")
            if live_birth_type[row] != BIRTH_TYPE_TO_CODE[history["birth_type"]]:
                raise ValueError("Live birth type disagrees with birth history")
            if live_birth_iter[row] != history["iteration"]:
                raise ValueError("Live birth iteration disagrees with birth history")
            if live_source_uid[row] != history["source_uid"]:
                raise ValueError("Live parent UID disagrees with birth history")
            if live_generation[row] != history["generation"]:
                raise ValueError("Live generation disagrees with parent history")

    # Backward-compatible explicit full-validation entry point.
    def validate(self, expected_count: int | None = None) -> None:
        self.audit_strict(expected_count=expected_count)

    def statistics(self) -> Dict[str, int]:
        """Return inexpensive diagnostics without validating or copying UID values."""
        birth_events = 0
        death_events = 0
        history_uid_entries = 0
        tensor_bytes = 0
        estimated_bytes = sys.getsizeof(self.events)
        for event in self.events:
            estimated_bytes += sys.getsizeof(event)
            if event.get("event_type") == "birth":
                birth_events += 1
            elif event.get("event_type") == "death":
                death_events += 1
            for value in event.values():
                estimated_bytes += sys.getsizeof(value)
                if torch.is_tensor(value):
                    payload_bytes = value.numel() * value.element_size()
                    tensor_bytes += payload_bytes
                    estimated_bytes += payload_bytes
            event_uids = event.get("uids")
            if torch.is_tensor(event_uids):
                history_uid_entries += event_uids.numel()
        return {
            "live_gaussian_count": self.uid.numel(),
            "allocated_uid_count": self.next_uid,
            "event_count": len(self.events),
            "birth_event_count": birth_events,
            "death_event_count": death_events,
            "history_uid_entry_count": history_uid_entries,
            "event_tensor_bytes": tensor_bytes,
            "estimated_event_bytes": estimated_bytes,
        }

    def state_dict(
        self, *, strict: bool = True, expected_count: int | None = None
    ) -> Dict:
        if strict:
            self.audit_strict(expected_count=expected_count)
        else:
            self.validate_lightweight(expected_count=expected_count)
        return {
            "version": LIFECYCLE_CHECKPOINT_VERSION,
            "birth_type_encoding": dict(BIRTH_TYPE_TO_CODE),
            "uid": self.uid,
            "birth_iter": self.birth_iter,
            "birth_type": self.birth_type,
            "source_uid": self.source_uid,
            "generation": self.generation,
            "next_uid": self.next_uid,
            "events": self.events,
        }

    @classmethod
    def from_state_dict(
        cls, state: Dict, *, expected_count: int, device: torch.device, strict: bool = True
    ) -> "GaussianLifecycle":
        if not isinstance(state, dict) or state.get("version") != LIFECYCLE_CHECKPOINT_VERSION:
            raise ValueError("Unsupported or malformed lifecycle checkpoint state")
        required = {
            "birth_type_encoding",
            "uid",
            "birth_iter",
            "birth_type",
            "source_uid",
            "generation",
            "next_uid",
            "events",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Lifecycle checkpoint is missing fields: {sorted(missing)}")
        if state["birth_type_encoding"] != BIRTH_TYPE_TO_CODE:
            raise ValueError("Lifecycle checkpoint uses an unknown birth-type encoding")
        if type(state["next_uid"]) is not int:
            raise ValueError("Lifecycle checkpoint next_uid must be an integer")
        if not isinstance(state["events"], list):
            raise ValueError("Lifecycle checkpoint events must be a list")
        checkpoint_tensor_specs = {
            "uid": torch.int64,
            "birth_iter": torch.int64,
            "birth_type": torch.uint8,
            "source_uid": torch.int64,
            "generation": torch.int64,
        }
        for name, dtype in checkpoint_tensor_specs.items():
            value = state[name]
            if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != dtype:
                raise ValueError(f"Lifecycle checkpoint {name} tensor is malformed")
        events = [dict(event) if isinstance(event, dict) else event for event in state["events"]]
        lifecycle = cls(
            uid=state["uid"].to(device=device),
            birth_iter=state["birth_iter"].to(device=device),
            birth_type=state["birth_type"].to(device=device),
            source_uid=state["source_uid"].to(device=device),
            generation=state["generation"].to(device=device),
            next_uid=state["next_uid"],
            events=events,
        )
        if strict:
            lifecycle.audit_strict(expected_count=expected_count)
        else:
            lifecycle.validate_lightweight(expected_count=expected_count)
        return lifecycle

    def _validate_mask(self, mask: torch.Tensor) -> None:
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("Lifecycle masks must be one-dimensional boolean tensors")
        if mask.device != self.uid.device:
            raise ValueError("Lifecycle masks must be on the lifecycle device")
        if mask.numel() != self.uid.numel():
            raise ValueError("Lifecycle mask length must match the live Gaussian count")

    @staticmethod
    def _audit_event_tensor(value, label: str) -> List[int]:
        if (
            not torch.is_tensor(value)
            or value.ndim != 1
            or value.dtype != torch.int64
            or value.device.type != "cpu"
        ):
            raise ValueError(f"Lifecycle event {label} are malformed")
        return value.tolist()

    def _record_birth(
        self, iteration: int, birth_type: str, uids: torch.Tensor, source_uids: torch.Tensor
    ) -> None:
        if uids.numel():
            self.events.append(
                {
                    "sequence": len(self.events),
                    "iteration": iteration,
                    "event_type": "birth",
                    "birth_type": birth_type,
                    "uids": uids.detach().cpu().clone(),
                    "source_uids": source_uids.detach().cpu().clone(),
                }
            )
