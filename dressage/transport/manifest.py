"""TransferQueue trajectory manifest schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

MANIFEST_SCHEMA_VERSION = "dressage.transport.manifest/v3"
ManifestState = Literal["SEALED", "PREPARED", "DRAINING"]


@dataclass
class TrajectoryManifest:
    session_id: str
    trajectory_id: str
    instance_id: str
    finalization_id: str
    step_refs: list[str]
    num_steps: int
    num_turns: int
    num_lineage_segments: int
    num_timeline_segments: int
    history_rewritten: bool
    label: Any
    build_config: dict[str, Any]
    config_fingerprint: str
    sealed_at: float
    retention_seconds: float
    state: ManifestState = "SEALED"
    schema_version: str = MANIFEST_SCHEMA_VERSION
    training_payload_keys: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrajectoryManifest:
        if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                "TransferQueue trajectory manifest schema does not match"
            )
        return cls(**value)
