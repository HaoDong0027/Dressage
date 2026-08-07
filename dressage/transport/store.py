"""TransferQueue-backed transport storage for Dressage trajectories."""

from __future__ import annotations

import threading
import uuid
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dressage.proxy.session_manager import StepRecord

from .manifest import TrajectoryManifest

STEP_SCHEMA_VERSION = "dressage.transfer_queue/v2"
_PAYLOAD_FIELD = "payload"


class TransferQueueTrajectoryStore:
    """Store lazy trajectory state in TransferQueue."""

    def __init__(
        self,
        tq_api: Any,
        *,
        store_id: str | None = None,
    ):
        self._tq = tq_api
        self.store_id = store_id or uuid.uuid4().hex
        self.step_partition = f"dressage:{self.store_id}:steps"
        self.manifest_partition = f"dressage:{self.store_id}:manifests"
        self.training_payload_partition = (
            f"dressage:{self.store_id}:training-payloads"
        )
        self.training_batch_partition = f"dressage:{self.store_id}:training-batches"
        self._index_lock = threading.Lock()
        self._step_refs: set[str] = set()

    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        store_id: str | None = None,
    ) -> TransferQueueTrajectoryStore:
        """Connect to the Ray cluster and initialize TransferQueue from YAML."""

        import ray
        try:
            import transfer_queue as tq
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "TransferQueue is not installed; install Dressage with "
                "the transfer-queue optional dependency"
            ) from exc

        if not ray.is_initialized():
            ray.init(address="auto")
        config = OmegaConf.load(Path(config_path))
        tq.init(config)
        return cls(
            tq,
            store_id=store_id,
        )

    @classmethod
    def from_existing(
        cls,
        *,
        store_id: str,
    ) -> TransferQueueTrajectoryStore:
        import ray

        try:
            import transfer_queue as tq
        except ImportError as exc:
            raise RuntimeError(
                "TransferQueue is not installed; install Dressage with "
                "the transfer-queue optional dependency"
            ) from exc

        if not ray.is_initialized():
            ray.init(address="auto")
        tq.init()
        return cls(
            tq,
            store_id=store_id,
        )

    @staticmethod
    def _step_to_dict(step: StepRecord) -> dict[str, Any]:
        return {item.name: getattr(step, item.name) for item in fields(step)}

    @staticmethod
    def _column_item(column: Any, index: int) -> Any:
        item = column[index]
        return getattr(item, "data", item)

    @classmethod
    def _payloads(cls, data: Any, expected_size: int) -> list[dict[str, Any]]:
        column = data[_PAYLOAD_FIELD]
        try:
            actual_size = int(data.batch_size[0])
        except (AttributeError, IndexError, TypeError):
            actual_size = len(column)
        if actual_size != expected_size:
            raise RuntimeError(
                "TransferQueue returned an unexpected item count: "
                f"expected={expected_size}, actual={actual_size}"
            )
        payloads = [cls._column_item(column, index) for index in range(actual_size)]
        if any(not isinstance(payload, dict) for payload in payloads):
            raise RuntimeError("TransferQueue payload is not a dictionary")
        return payloads

    async def write_step(
        self,
        *,
        session_id: str,
        step_index: int,
        step: StepRecord,
    ) -> str:
        key = f"{session_id}:{step_index}"
        payload = {
            "schema_version": STEP_SCHEMA_VERSION,
            "session_id": session_id,
            "step_index": step_index,
            "record": self._step_to_dict(step),
        }
        await self._tq.async_kv_put(
            key=key,
            partition_id=self.step_partition,
            fields={_PAYLOAD_FIELD: payload},
        )
        with self._index_lock:
            self._step_refs.add(key)
        return key

    async def read_steps(self, step_refs: list[str]) -> list[StepRecord]:
        if not step_refs:
            return []
        data = await self._tq.async_kv_batch_get(
            keys=step_refs,
            partition_id=self.step_partition,
            select_fields=_PAYLOAD_FIELD,
        )
        payloads = self._payloads(data, len(step_refs))
        from dressage.proxy.session_manager import StepRecord

        steps: list[StepRecord] = []
        for step_ref, payload in zip(step_refs, payloads, strict=True):
            expected_index = int(step_ref.rsplit(":", 1)[1])
            if (
                payload.get("schema_version") != STEP_SCHEMA_VERSION
                or payload.get("step_index") != expected_index
            ):
                raise RuntimeError(
                    "TransferQueue StepRecord order or schema does not match"
                )
            steps.append(StepRecord(**payload["record"]))
        return steps

    async def discard_step_routed_experts(self, step_ref: str) -> None:
        data = await self._tq.async_kv_batch_get(
            keys=step_ref,
            partition_id=self.step_partition,
            select_fields=_PAYLOAD_FIELD,
        )
        payload = self._payloads(data, 1)[0]
        if payload.get("schema_version") != STEP_SCHEMA_VERSION:
            raise RuntimeError("TransferQueue StepRecord schema does not match")
        record = dict(payload["record"])
        record["response_routed_experts_chunks"] = []
        payload = dict(payload)
        payload["record"] = record
        await self._tq.async_kv_put(
            key=step_ref,
            partition_id=self.step_partition,
            fields={_PAYLOAD_FIELD: payload},
        )

    async def clear_steps(self, step_refs: list[str]) -> None:
        if not step_refs:
            return
        await self._tq.async_kv_clear(
            keys=step_refs,
            partition_id=self.step_partition,
        )
        with self._index_lock:
            self._step_refs.difference_update(step_refs)

    def forget_steps(self, step_refs: list[str]) -> None:
        with self._index_lock:
            self._step_refs.difference_update(step_refs)

    async def write_manifest(self, manifest: TrajectoryManifest) -> None:
        await self._tq.async_kv_put(
            key=manifest.trajectory_id,
            partition_id=self.manifest_partition,
            fields={_PAYLOAD_FIELD: manifest.to_dict()},
        )

    async def read_manifest(
        self,
        trajectory_id: str,
    ) -> TrajectoryManifest | None:
        try:
            data = await self._tq.async_kv_batch_get(
                keys=trajectory_id,
                partition_id=self.manifest_partition,
                select_fields=_PAYLOAD_FIELD,
            )
        except ValueError as exc:
            if "not found" not in str(exc).lower():
                raise
            return None
        try:
            size = int(data.batch_size[0])
        except (AttributeError, IndexError, TypeError):
            size = len(data[_PAYLOAD_FIELD])
        if size == 0:
            return None
        payload = self._payloads(data, 1)[0]
        return TrajectoryManifest.from_dict(payload)

    async def list_manifests(self) -> list[TrajectoryManifest]:
        partitions = await self._tq.async_kv_list(
            partition_id=self.manifest_partition,
        )
        keys = list(partitions.get(self.manifest_partition, {}))
        if not keys:
            return []
        data = await self._tq.async_kv_batch_get(
            keys=keys,
            partition_id=self.manifest_partition,
            select_fields=_PAYLOAD_FIELD,
        )
        return [
            TrajectoryManifest.from_dict(payload)
            for payload in self._payloads(data, len(keys))
        ]

    async def clear_manifest(self, trajectory_id: str) -> None:
        await self._tq.async_kv_clear(
            keys=trajectory_id,
            partition_id=self.manifest_partition,
        )

    async def write_training_payload(
        self,
        *,
        payload_ref: dict[str, Any],
        routed_experts: Any,
        full_logprobs: Any,
    ) -> None:
        from .payload import TrainingPayloadRef

        ref = TrainingPayloadRef.from_dict(payload_ref)
        metadata = {
            "schema_version": ref.schema_version,
            "payload_key": ref.payload_key,
            "trajectory_id": ref.trajectory_id,
            "segment_id": ref.segment_id,
            "token_count": ref.token_count,
            "routed_experts_shape": list(ref.routed_experts_shape),
        }
        await self._tq.async_kv_put(
            key=ref.payload_key,
            partition_id=self.training_payload_partition,
            fields={
                _PAYLOAD_FIELD: metadata,
                "routed_experts": routed_experts,
                "full_logprobs": full_logprobs,
            },
        )

    async def read_training_payloads(
        self,
        payload_refs: list[dict[str, Any]],
        *,
        include_routed_experts: bool,
        include_logprobs: bool,
    ) -> list[dict[str, Any]]:
        from .payload import TrainingPayloadRef

        refs = [TrainingPayloadRef.from_dict(value) for value in payload_refs]
        fields = [_PAYLOAD_FIELD]
        if include_routed_experts:
            fields.append("routed_experts")
        if include_logprobs:
            fields.append("full_logprobs")
        data = await self._tq.async_kv_batch_get(
            keys=[ref.payload_key for ref in refs],
            partition_id=self.training_payload_partition,
            select_fields=fields,
        )
        metadata = self._payloads(data, len(refs))
        result: list[dict[str, Any]] = []
        for index, (ref, stored) in enumerate(zip(refs, metadata, strict=True)):
            if (
                stored.get("schema_version") != ref.schema_version
                or stored.get("payload_key") != ref.payload_key
                or stored.get("trajectory_id") != ref.trajectory_id
                or stored.get("segment_id") != ref.segment_id
                or int(stored.get("token_count", -1)) < ref.token_count
                or list(stored.get("routed_experts_shape") or [])
                != ref.routed_experts_shape
            ):
                raise RuntimeError(
                    "TransferQueue training payload reference does not match"
                )
            item: dict[str, Any] = {"payload_ref": ref.to_dict()}
            if include_routed_experts:
                item["routed_experts"] = (
                    self._column_item(data["routed_experts"], index)
                    .detach()
                    .clone()
                )
            if include_logprobs:
                item["full_logprobs"] = (
                    self._column_item(data["full_logprobs"], index)
                    .detach()
                    .clone()
                )
            result.append(item)
        return result

    async def clear_training_payloads(self, payload_keys: list[str]) -> None:
        if not payload_keys:
            return
        await self._tq.async_kv_clear(
            keys=payload_keys,
            partition_id=self.training_payload_partition,
        )

    async def write_training_batch(
        self,
        *,
        batch_id: int,
        trajectory_ids: list[str],
        payload_keys: list[str],
        created_at: float,
    ) -> None:
        from .payload import TRAINING_BATCH_SCHEMA_VERSION

        await self._tq.async_kv_put(
            key=str(batch_id),
            partition_id=self.training_batch_partition,
            fields={
                _PAYLOAD_FIELD: {
                    "schema_version": TRAINING_BATCH_SCHEMA_VERSION,
                    "batch_id": batch_id,
                    "trajectory_ids": list(trajectory_ids),
                    "payload_keys": list(payload_keys),
                    "created_at": created_at,
                }
            },
        )

    async def read_training_batch(self, batch_id: int) -> dict[str, Any] | None:
        from .payload import TRAINING_BATCH_SCHEMA_VERSION

        batches = await self._tq.async_kv_list(
            partition_id=self.training_batch_partition,
        )
        if str(batch_id) not in batches.get(self.training_batch_partition, {}):
            return None
        try:
            data = await self._tq.async_kv_batch_get(
                keys=str(batch_id),
                partition_id=self.training_batch_partition,
                select_fields=_PAYLOAD_FIELD,
            )
        except ValueError as exc:
            if "not found" not in str(exc).lower():
                raise
            return None
        payload = self._payloads(data, 1)[0]
        if (
            payload.get("schema_version") != TRAINING_BATCH_SCHEMA_VERSION
            or payload.get("batch_id") != batch_id
        ):
            raise RuntimeError("TransferQueue training batch schema does not match")
        return payload

    async def list_training_batches(self) -> list[dict[str, Any]]:
        batches = await self._tq.async_kv_list(
            partition_id=self.training_batch_partition,
        )
        keys = list(batches.get(self.training_batch_partition, {}))
        if not keys:
            return []
        data = await self._tq.async_kv_batch_get(
            keys=keys,
            partition_id=self.training_batch_partition,
            select_fields=_PAYLOAD_FIELD,
        )
        return self._payloads(data, len(keys))

    async def clear_training_batch_record(self, batch_id: int) -> None:
        await self._tq.async_kv_clear(
            keys=str(batch_id),
            partition_id=self.training_batch_partition,
        )

    @staticmethod
    def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from dressage.proxy.trajectory_store import TrajectoryStore

        items = [TrajectoryStore._item_from_dict(record) for record in records]
        TrajectoryStore._validate_finalized_batch(items)
        return [item.to_dict() for item in items]

    def stats(self) -> dict[str, Any]:
        with self._index_lock:
            return {
                "step_refs": len(self._step_refs),
                "transfer_queue_enabled": True,
                "transfer_queue_ready": True,
                "transfer_queue_store_id": self.store_id,
            }
