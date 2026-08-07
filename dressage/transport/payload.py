"""Schemas and helpers for lazy TransferQueue training payloads."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any

TRAINING_PAYLOAD_SCHEMA_VERSION = "dressage.transport.training_payload/v2"
TRAINING_BATCH_SCHEMA_VERSION = "dressage.transport.training_batch/v2"
TRAINING_PAYLOAD_METADATA_KEY = "tq_training_payload_ref"
LAZY_TRAJECTORY_METADATA_KEY = "tq_lazy_trajectory_id"
TRAINING_PAYLOAD_RETENTION_SECONDS = 86400.0


@dataclass(frozen=True)
class TrainingPayloadRef:
    payload_key: str
    trajectory_id: str
    segment_id: str
    token_count: int
    response_start: int
    response_length: int
    routed_experts_shape: list[int]
    batch_id: int | None = None
    schema_version: str = TRAINING_PAYLOAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bind_batch(self, batch_id: int) -> TrainingPayloadRef:
        return replace(self, batch_id=int(batch_id))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingPayloadRef:
        if value.get("schema_version") != TRAINING_PAYLOAD_SCHEMA_VERSION:
            raise RuntimeError("TransferQueue training payload schema does not match")
        return cls(**value)


def transfer_queue_enabled() -> bool:
    return os.environ.get("DRESSAGE_ENABLE_TRANSFER_QUEUE", "0") == "1"


def validate_transfer_queue_training_args(args: Any) -> None:
    if not transfer_queue_enabled():
        return
    if getattr(args, "mopd_teacher_config", None) or os.environ.get(
        "DRESSAGE_MOPD_TEACHER_CONFIG"
    ):
        raise ValueError("MOPD does not support lazy TransferQueue payloads")


def bind_training_batch(samples: list[Any], batch_id: int) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        value = metadata.get(TRAINING_PAYLOAD_METADATA_KEY)
        if value is None:
            continue
        metadata[TRAINING_PAYLOAD_METADATA_KEY] = (
            TrainingPayloadRef.from_dict(value).bind_batch(batch_id).to_dict()
        )
