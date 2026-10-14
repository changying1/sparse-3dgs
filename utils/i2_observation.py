"""Opt-in real-training-view observation evidence for Innovation 2.

The counters and distinct-view history are row-aligned with the currently alive
Gaussians.  Hot-path updates perform only tensor operations; exhaustive value
and UID checks are reserved for checkpoint boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, Iterable, Sequence

import torch

from llff_view_protocol import canonical_image_name, load_view_manifest


OBSERVATION_CHECKPOINT_VERSION = 1


def add_i2_observation_arguments(parser) -> None:
    parser.add_argument(
        "--enable_i2_observation_tracking",
        action="store_true",
        default=False,
        help="Track real-view observation evidence for live Gaussians (I2-B0).",
    )


def validate_i2_observation_configuration(
    observation_enabled: bool, lifecycle_enabled: bool
) -> None:
    if observation_enabled and not lifecycle_enabled:
        raise ValueError(
            "--enable_i2_observation_tracking requires "
            "--enable_i2_lifecycle_tracking"
        )


def build_real_view_camera_mapping(
    train_cameras: Iterable, view_manifest_path: str = ""
) -> tuple[tuple[str, ...], dict[int, int]]:
    """Build a stable runtime-camera to frozen-view mapping.

    LLFF runs use the exact ``selected_train_images`` identifiers and order from
    the already validated frozen manifest.  The camera objects retain only the
    image stem, so matching is explicit and must remain one-to-one.  Legacy
    non-manifest entry points use sorted image names as stable identifiers.
    """
    cameras = list(train_cameras)
    if not cameras:
        raise ValueError("Observation tracking requires at least one training camera")

    camera_records = []
    seen_uids = set()
    seen_names = set()
    for camera in cameras:
        uid = getattr(camera, "uid", None)
        name = getattr(camera, "image_name", None)
        if type(uid) is not int or uid in seen_uids:
            raise ValueError("Training camera UIDs must be unique integers")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("Training camera image names must be unique non-empty strings")
        seen_uids.add(uid)
        seen_names.add(name)
        camera_records.append((uid, name))

    if view_manifest_path:
        manifest = load_view_manifest(view_manifest_path)
        identifiers = tuple(
            canonical_image_name(name) for name in manifest["selected_train_images"]
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Observation manifest view identifiers must be unique")
        index_by_stem = {}
        for index, identifier in enumerate(identifiers):
            stem = PurePosixPath(identifier).stem
            if stem in index_by_stem:
                raise ValueError(
                    "Observation manifest image stems must be unique for camera mapping"
                )
            index_by_stem[stem] = index
        if set(index_by_stem) != seen_names:
            raise ValueError(
                "Training cameras do not exactly match manifest selected_train_images"
            )
        return identifiers, {
            uid: index_by_stem[name] for uid, name in camera_records
        }

    identifiers = tuple(sorted(seen_names))
    index_by_name = {name: index for index, name in enumerate(identifiers)}
    return identifiers, {uid: index_by_name[name] for uid, name in camera_records}


@dataclass
class GaussianObservation:
    """Row-aligned counters for real training-view evidence."""

    real_view_opportunities: torch.Tensor
    dropout_retained_hits: torch.Tensor
    screen_supported_hits: torch.Tensor
    screen_supported_view_history: torch.Tensor
    invalid_screen_support_seen: torch.Tensor
    view_identifiers: tuple[str, ...]
    last_real_view_iteration: int = -1

    @classmethod
    def zeros(
        cls, count: int, device: torch.device, view_identifiers: Sequence[str]
    ) -> "GaussianObservation":
        if type(count) is not int or count < 0:
            raise ValueError("Gaussian count must be a non-negative integer")
        identifiers = cls._validate_view_identifiers(view_identifiers)
        zeros = torch.zeros(count, dtype=torch.int64, device=device)
        history = torch.zeros(
            (count, len(identifiers)), dtype=torch.bool, device=device
        )
        state = cls(
            zeros,
            zeros.clone(),
            zeros.clone(),
            history,
            torch.zeros((), dtype=torch.bool, device=device),
            identifiers,
        )
        state.validate_lightweight(expected_count=count)
        return state

    @property
    def unique_screen_supported_real_views(self) -> torch.Tensor:
        """Return the exact distinct supported-view count for each live row."""
        return self.screen_supported_view_history.sum(dim=1, dtype=torch.int64)

    @property
    def view_history_bytes(self) -> int:
        return (
            self.screen_supported_view_history.numel()
            * self.screen_supported_view_history.element_size()
        )

    def record_real_view(
        self,
        dropout_mask: torch.Tensor,
        screen_supported_mask: torch.Tensor,
        view_index: int,
        iteration: int,
        *,
        is_real_view: bool,
    ) -> None:
        """Accumulate one primary real-view render using full-row masks."""
        self.validate_lightweight()
        if is_real_view is not True:
            raise ValueError("Observation evidence requires a real training view")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Observation iteration must be a non-negative integer")
        if iteration <= self.last_real_view_iteration:
            raise ValueError(
                "Real-view observation updates must occur once in increasing iteration order"
            )
        self._validate_full_mask(dropout_mask)
        self._validate_full_mask(screen_supported_mask)
        if type(view_index) is not int or not 0 <= view_index < len(
            self.view_identifiers
        ):
            raise ValueError("Observation view index is outside the training-view mapping")
        # Keep malformed support out of the evidence counters.  The scalar error
        # latch remains on the state device and is inspected only at cold audit
        # boundaries, so this path does not synchronize CUDA with the host.
        valid_screen_supported = screen_supported_mask & dropout_mask
        self.invalid_screen_support_seen.logical_or_(
            torch.any(screen_supported_mask != valid_screen_supported)
        )
        # All per-Gaussian operations below are vectorized on the state device.
        self.real_view_opportunities.add_(1)
        self.dropout_retained_hits.add_(dropout_mask)
        self.screen_supported_hits.add_(valid_screen_supported)
        self.screen_supported_view_history[:, view_index].logical_or_(
            valid_screen_supported
        )
        self.last_real_view_iteration = iteration

    def append_zeros(self, child_count: int) -> None:
        """Append zero evidence for newly born clone/split children."""
        self.validate_lightweight()
        if type(child_count) is not int or child_count < 0:
            raise ValueError("Observation child count must be a non-negative integer")
        if child_count == 0:
            return
        zeros = torch.zeros(
            child_count,
            dtype=torch.int64,
            device=self.real_view_opportunities.device,
        )
        self.real_view_opportunities = torch.cat(
            (self.real_view_opportunities, zeros)
        )
        self.dropout_retained_hits = torch.cat(
            (self.dropout_retained_hits, zeros.clone())
        )
        self.screen_supported_hits = torch.cat(
            (self.screen_supported_hits, zeros.clone())
        )
        self.screen_supported_view_history = torch.cat(
            (
                self.screen_supported_view_history,
                torch.zeros(
                    (child_count, len(self.view_identifiers)),
                    dtype=torch.bool,
                    device=self.real_view_opportunities.device,
                ),
            ),
            dim=0,
        )
        self.validate_lightweight()

    def prune(self, mask: torch.Tensor) -> None:
        """Apply the exact Gaussian deletion mask to all observation counters."""
        self.validate_lightweight()
        self._validate_full_mask(mask)
        keep = ~mask
        self.real_view_opportunities = self.real_view_opportunities[keep]
        self.dropout_retained_hits = self.dropout_retained_hits[keep]
        self.screen_supported_hits = self.screen_supported_hits[keep]
        self.screen_supported_view_history = self.screen_supported_view_history[keep]
        self.validate_lightweight()

    def validate_lightweight(self, expected_count: int | None = None) -> None:
        """Validate row structure without scanning counter values."""
        tensors = (
            self.real_view_opportunities,
            self.dropout_retained_hits,
            self.screen_supported_hits,
        )
        device = self.real_view_opportunities.device
        lengths = set()
        for tensor in tensors:
            if not torch.is_tensor(tensor) or tensor.ndim != 1:
                raise ValueError(
                    "Observation counters must be one-dimensional tensors"
                )
            if tensor.dtype != torch.int64:
                raise ValueError("Observation counters must use torch.int64")
            if tensor.device != device:
                raise ValueError("Observation counters must share one device")
            lengths.add(tensor.numel())
        if len(lengths) != 1:
            raise ValueError("Observation counters are not row-aligned")
        count = self.real_view_opportunities.numel()
        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"Observation row count {count} does not match Gaussian count "
                f"{expected_count}"
            )
        self._validate_view_identifiers(self.view_identifiers)
        history = self.screen_supported_view_history
        if (
            not torch.is_tensor(history)
            or history.dtype != torch.bool
            or history.ndim != 2
            or history.shape != (count, len(self.view_identifiers))
        ):
            raise ValueError(
                "Observation view history must be a row-aligned boolean matrix"
            )
        if history.device != device:
            raise ValueError("Observation view history must share the counter device")
        invalid_seen = self.invalid_screen_support_seen
        if (
            not torch.is_tensor(invalid_seen)
            or invalid_seen.dtype != torch.bool
            or invalid_seen.ndim != 0
            or invalid_seen.device != device
        ):
            raise ValueError(
                "Observation invalid-support marker must be a boolean scalar "
                "on the counter device"
            )
        if (
            type(self.last_real_view_iteration) is not int
            or self.last_real_view_iteration < -1
        ):
            raise ValueError("last_real_view_iteration must be an integer >= -1")

    def audit_strict(
        self, *, expected_uid: torch.Tensor, expected_count: int | None = None
    ) -> None:
        """Validate values and exact A0 UID row alignment at a cold boundary."""
        self.validate_lightweight(expected_count=expected_count)
        self._validate_expected_uid(expected_uid)
        opportunities = self.real_view_opportunities
        retained = self.dropout_retained_hits
        supported = self.screen_supported_hits
        if bool(self.invalid_screen_support_seen.item()):
            raise ValueError(
                "Screen support was observed outside the Dropout retained set"
            )
        if bool(torch.any(opportunities < 0).item()):
            raise ValueError("real_view_opportunities must be non-negative")
        if bool(torch.any(retained < 0).item()):
            raise ValueError("dropout_retained_hits must be non-negative")
        if bool(torch.any(supported < 0).item()):
            raise ValueError("screen_supported_hits must be non-negative")
        if bool(torch.any(retained < supported).item()):
            raise ValueError(
                "screen_supported_hits cannot exceed dropout_retained_hits"
            )
        if bool(torch.any(opportunities < retained).item()):
            raise ValueError(
                "dropout_retained_hits cannot exceed real_view_opportunities"
            )
        distinct = self.unique_screen_supported_real_views
        if bool(torch.any(distinct > supported).item()):
            raise ValueError(
                "Distinct supported views cannot exceed screen-supported hits"
            )
        has_evidence = torch.any(opportunities != 0)
        has_evidence.logical_or_(torch.any(retained != 0))
        has_evidence.logical_or_(torch.any(supported != 0))
        has_evidence.logical_or_(torch.any(self.screen_supported_view_history))
        if self.last_real_view_iteration == -1 and bool(has_evidence.item()):
            raise ValueError(
                "Observation evidence cannot exist before a real-view iteration"
            )
        if self.last_real_view_iteration >= 0 and bool(
            torch.any(opportunities > self.last_real_view_iteration + 1).item()
        ):
            raise ValueError(
                "real_view_opportunities exceed the recorded iteration range"
            )

    def state_dict(self, *, lifecycle_uid: torch.Tensor) -> Dict:
        self.audit_strict(
            expected_uid=lifecycle_uid,
            expected_count=lifecycle_uid.numel(),
        )
        return {
            "version": OBSERVATION_CHECKPOINT_VERSION,
            "uid": lifecycle_uid,
            "real_view_opportunities": self.real_view_opportunities,
            "dropout_retained_hits": self.dropout_retained_hits,
            "screen_supported_hits": self.screen_supported_hits,
            "screen_supported_view_history": self.screen_supported_view_history,
            "view_identifiers": list(self.view_identifiers),
            "last_real_view_iteration": self.last_real_view_iteration,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict,
        *,
        lifecycle_uid: torch.Tensor,
        expected_view_identifiers: Sequence[str],
        expected_count: int,
        device: torch.device,
    ) -> "GaussianObservation":
        if (
            not isinstance(state, dict)
            or state.get("version") != OBSERVATION_CHECKPOINT_VERSION
        ):
            raise ValueError("Unsupported or malformed observation checkpoint state")
        required = {
            "uid",
            "real_view_opportunities",
            "dropout_retained_hits",
            "screen_supported_hits",
            "screen_supported_view_history",
            "view_identifiers",
            "last_real_view_iteration",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"Observation checkpoint is missing fields: {sorted(missing)}"
            )
        for name in (
            "uid",
            "real_view_opportunities",
            "dropout_retained_hits",
            "screen_supported_hits",
        ):
            value = state[name]
            if (
                not torch.is_tensor(value)
                or value.ndim != 1
                or value.dtype != torch.int64
            ):
                raise ValueError(
                    f"Observation checkpoint {name} tensor is malformed"
                )
        if type(state["last_real_view_iteration"]) is not int:
            raise ValueError(
                "Observation checkpoint last_real_view_iteration must be an integer"
            )
        identifiers = cls._validate_view_identifiers(state["view_identifiers"])
        expected_identifiers = cls._validate_view_identifiers(
            expected_view_identifiers
        )
        if identifiers != expected_identifiers:
            raise ValueError(
                "Observation checkpoint training-view identifiers do not match "
                "the current training views"
            )
        history = state["screen_supported_view_history"]
        if (
            not torch.is_tensor(history)
            or history.dtype != torch.bool
            or history.ndim != 2
            or history.shape != (expected_count, len(identifiers))
        ):
            raise ValueError("Observation checkpoint view history is malformed")
        checkpoint_uid = state["uid"].to(device=device)
        if checkpoint_uid.numel() != expected_count or not torch.equal(
            checkpoint_uid, lifecycle_uid
        ):
            raise ValueError(
                "Observation checkpoint UID rows do not match lifecycle UID rows"
            )
        observation = cls(
            real_view_opportunities=state["real_view_opportunities"]
            .to(device=device)
            .clone(),
            dropout_retained_hits=state["dropout_retained_hits"]
            .to(device=device)
            .clone(),
            screen_supported_hits=state["screen_supported_hits"]
            .to(device=device)
            .clone(),
            screen_supported_view_history=history.to(device=device).clone(),
            # state_dict() audits this runtime-only latch before serializing, so
            # every valid checkpoint necessarily restores with a clear marker.
            invalid_screen_support_seen=torch.zeros(
                (), dtype=torch.bool, device=device
            ),
            view_identifiers=identifiers,
            last_real_view_iteration=state["last_real_view_iteration"],
        )
        observation.audit_strict(
            expected_uid=lifecycle_uid,
            expected_count=expected_count,
        )
        return observation

    def _validate_full_mask(self, mask: torch.Tensor) -> None:
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("Observation masks must be one-dimensional boolean tensors")
        if mask.device != self.real_view_opportunities.device:
            raise ValueError("Observation masks must be on the state device")
        if mask.numel() != self.real_view_opportunities.numel():
            raise ValueError(
                "Observation mask length must match the live Gaussian count"
            )

    def _validate_expected_uid(self, expected_uid: torch.Tensor) -> None:
        if (
            not torch.is_tensor(expected_uid)
            or expected_uid.dtype != torch.int64
            or expected_uid.ndim != 1
        ):
            raise ValueError("Lifecycle UID rows must be a one-dimensional int64 tensor")
        if expected_uid.device != self.real_view_opportunities.device:
            raise ValueError("Lifecycle UID rows must be on the observation-state device")
        if expected_uid.numel() != self.real_view_opportunities.numel():
            raise ValueError("Observation rows do not match lifecycle UID rows")

    @staticmethod
    def _validate_view_identifiers(
        view_identifiers: Sequence[str],
    ) -> tuple[str, ...]:
        if not isinstance(view_identifiers, (list, tuple)):
            raise ValueError("Observation view identifiers must be a list or tuple")
        identifiers = tuple(view_identifiers)
        if not identifiers:
            raise ValueError("Observation tracking requires at least one real view")
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("Observation view identifiers must be non-empty strings")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Observation view identifiers must be unique")
        return identifiers
