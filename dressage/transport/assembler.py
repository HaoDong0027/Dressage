"""Ray actors that assemble TransferQueue StepRecords outside the proxy."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .builder import TrajectoryBuildConfig, TrajectoryBuilder
from .manifest import MANIFEST_SCHEMA_VERSION
from .payload import (
    TRAINING_PAYLOAD_RETENTION_SECONDS,
    TRAINING_PAYLOAD_SCHEMA_VERSION,
    TrainingPayloadRef,
)
from .store import STEP_SCHEMA_VERSION, TransferQueueTrajectoryStore

TRANSPORT_NAMESPACE = "dressage_transport"
COORDINATOR_NAME = "DressageTransportCoordinator"
ASSEMBLER_NAME_PREFIX = "DressageTrajectoryAssembler#"
TRAINING_LOADER_NAME_PREFIX = "DressageTrainingPayloadLoader#"


def transport_config_fingerprint(
    config_path: str,
    build_config: TrajectoryBuildConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(Path(config_path).read_bytes())
    digest.update(MANIFEST_SCHEMA_VERSION.encode("utf-8"))
    digest.update(STEP_SCHEMA_VERSION.encode("utf-8"))
    digest.update(TRAINING_PAYLOAD_SCHEMA_VERSION.encode("utf-8"))
    digest.update(
        json.dumps(
            build_config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def assembler_name_for(
    trajectory_id: str,
    assembler_names: list[str],
) -> str:
    digest = hashlib.sha256(trajectory_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(assembler_names)
    return assembler_names[index]


class TrajectoryAssembler:
    def __init__(
        self,
        *,
        store_id: str,
        build_config: dict[str, Any],
        config_fingerprint: str,
        cleanup_owner: bool,
    ):
        self._store = TransferQueueTrajectoryStore.from_existing(store_id=store_id)
        self._builder = TrajectoryBuilder(TrajectoryBuildConfig.from_dict(build_config))
        self._config_fingerprint = config_fingerprint
        self._cleanup_owner = cleanup_owner
        self._last_cleanup = 0.0

    async def _cleanup_expired(self, now: float) -> None:
        if not self._cleanup_owner or now - self._last_cleanup < 60.0:
            return
        self._last_cleanup = now
        for manifest in await self._store.list_manifests():
            if now - manifest.sealed_at < manifest.retention_seconds:
                continue
            await self._store.clear_steps(manifest.step_refs)
            await self._store.clear_training_payloads(
                manifest.training_payload_keys or []
            )
            await self._store.clear_manifest(manifest.trajectory_id)

    @staticmethod
    def _decode_routed_experts(record: dict[str, Any]) -> Any:
        import numpy as np
        import torch

        try:
            import pybase64
        except ImportError:
            import base64 as pybase64

        arrays = []
        width: int | None = None
        for chunk in record.get("routed_experts_chunks") or []:
            row_count = int(chunk["row_count"])
            values = np.frombuffer(
                pybase64.b64decode(chunk["data"].encode("ascii")),
                dtype=np.int32,
            )
            if row_count <= 0 or values.size % row_count != 0:
                raise ValueError("routed_experts chunk shape is invalid")
            chunk_width = values.size // row_count
            if width is None:
                width = chunk_width
            elif width != chunk_width:
                raise ValueError("routed_experts chunks have inconsistent widths")
            arrays.append(values.reshape(row_count, chunk_width))

        if not arrays:
            return torch.empty((0, 0), dtype=torch.int32)
        combined = np.concatenate(arrays, axis=0).copy()
        expected_rows = len(record["tokens"]) - 1
        if combined.shape[0] != expected_rows:
            raise ValueError(
                "routed_experts length does not match trajectory tokens: "
                f"expected={expected_rows}, actual={combined.shape[0]}"
            )
        return torch.from_numpy(combined)

    async def _write_training_payloads(
        self,
        manifest: Any,
        records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        import torch

        light_records: list[dict[str, Any]] = []
        payload_keys: list[str] = []
        try:
            for record in records:
                routed_experts = self._decode_routed_experts(record)
                loss_mask = list(record["full_loss_mask"])
                response_start = next(
                    (
                        index
                        for index, value in enumerate(loss_mask)
                        if int(value) == 1
                    ),
                    len(record["tokens"]),
                )
                payload_key = (
                    f"{manifest.trajectory_id}:{manifest.finalization_id}:"
                    f"{record['extra_info']['segment_view']}:"
                    f"{int(record.get('segment_index', 0))}"
                )
                payload_ref = TrainingPayloadRef(
                    payload_key=payload_key,
                    trajectory_id=manifest.trajectory_id,
                    segment_id=str(record["uid"]),
                    token_count=len(record["tokens"]),
                    response_start=response_start,
                    response_length=len(record["tokens"]) - response_start,
                    routed_experts_shape=list(routed_experts.shape),
                ).to_dict()
                await self._store.write_training_payload(
                    payload_ref=payload_ref,
                    routed_experts=routed_experts,
                    full_logprobs=torch.tensor(
                        record["full_logprobs"],
                        dtype=torch.float32,
                    ),
                )
                payload_keys.append(payload_key)
                light_record = dict(record)
                light_record.pop("full_logprobs", None)
                light_record.pop("routed_experts_chunks", None)
                light_record["training_payload_ref"] = payload_ref
                light_records.append(light_record)
        except Exception:
            await self._store.clear_training_payloads(payload_keys)
            raise
        return light_records, payload_keys

    async def _drain(self, manifest: Any) -> dict[str, Any]:
        if manifest.state != "DRAINING":
            manifest = replace(manifest, state="DRAINING")
            await self._store.write_manifest(manifest)
        await self._store.clear_steps(manifest.step_refs)
        await self._store.clear_training_payloads(
            manifest.training_payload_keys or []
        )
        await self._store.clear_manifest(manifest.trajectory_id)
        return {"success": True, "state": "DRAINED"}

    async def prepare(
        self,
        *,
        trajectory_id: str,
        instance_id: str | None = None,
        segment_view: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        await self._cleanup_expired(now)
        manifest = await self._store.read_manifest(trajectory_id)
        if manifest is None or (
            instance_id is not None and manifest.instance_id != instance_id
        ):
            return {
                "success": False,
                "mode": "trajectory",
                "data": [],
                "drained": False,
            }
        if manifest.config_fingerprint != self._config_fingerprint:
            raise RuntimeError(
                "trajectory manifest configuration does not match assembler"
            )
        expected_view = (
            "lineage"
            if manifest.build_config["token_build_mode"] == "tito"
            else "timeline"
        )
        selected_view = segment_view or expected_view
        if selected_view not in {"lineage", "timeline"} or (
            manifest.build_config["token_build_mode"] == "snapshot"
            and selected_view != "timeline"
        ):
            return {
                "success": False,
                "mode": "trajectory",
                "data": [],
                "drained": False,
            }
        if manifest.state == "DRAINING":
            await self._drain(manifest)
            return {
                "success": False,
                "mode": "trajectory",
                "data": [],
                "drained": True,
            }
        if manifest.state == "PREPARED":
            return {
                "success": False,
                "mode": "trajectory",
                "data": [],
                "drained": True,
            }

        prepared = replace(manifest, state="PREPARED")
        payload_keys: list[str] = []
        await self._store.write_manifest(prepared)
        try:
            from dressage.proxy.session_manager import Lineage, Session

            steps = await self._store.read_steps(prepared.step_refs)
            lineages: dict[str, Lineage] = {}
            for step in steps:
                lineage = lineages.get(step.lineage_id)
                if lineage is None:
                    lineage = Lineage(
                        id=step.lineage_id,
                        index=step.lineage_index,
                        latest_step_id=step.step_id,
                        branch_from_step_id=(
                            step.route_base_step_id
                            if step.route_type == "branch"
                            else None
                        ),
                    )
                    lineages[step.lineage_id] = lineage
                else:
                    lineage.latest_step_id = step.step_id
            session = Session(
                session_id=prepared.session_id,
                instance_id=prepared.instance_id,
                steps=steps,
                history_rewritten=prepared.history_rewritten,
                lineages=lineages,
                steps_by_id={step.step_id: step for step in steps},
            )
            records = self._builder.build_records(
                session=session,
                trajectory_id=prepared.trajectory_id,
                instance_id=prepared.instance_id,
                finalization_id=prepared.finalization_id,
                label=prepared.label,
                segment_view=selected_view,
                sealed_at=prepared.sealed_at,
            )
            expected_segment_count = (
                prepared.num_lineage_segments
                if selected_view == "lineage"
                else prepared.num_timeline_segments
            )
            if len(records) != expected_segment_count:
                raise RuntimeError(
                    "trajectory manifest segment count does not match StepRecords"
                )
            records = self._store._normalize_records(records)
            records, payload_keys = await self._write_training_payloads(
                prepared,
                records,
            )
            prepared = replace(
                prepared,
                training_payload_keys=payload_keys,
            )
            await self._store.write_manifest(prepared)
        except Exception:
            await self._store.clear_training_payloads(payload_keys)
            await self._store.write_manifest(
                replace(
                    prepared,
                    state="SEALED",
                    training_payload_keys=None,
                )
            )
            raise
        return {
            "success": bool(records),
            "mode": "trajectory",
            "data": records,
            "meta_info": {
                "transfer_queue_enabled": True,
                "transfer_queue_ready": True,
                "transfer_queue_store_id": self._store.store_id,
                "manifest_state": "PREPARED",
            },
            "drained": False,
        }

    async def ack(
        self,
        *,
        trajectory_id: str,
    ) -> dict[str, Any]:
        manifest = await self._store.read_manifest(trajectory_id)
        if manifest is None:
            return {"success": True, "state": "DRAINED"}
        if manifest.state == "DRAINING":
            draining = manifest
        else:
            if manifest.state != "PREPARED":
                raise RuntimeError("trajectory has not been prepared for training")
            draining = manifest
        return await self._drain(draining)

    async def ack_steps(
        self,
        *,
        trajectory_id: str,
    ) -> dict[str, Any]:
        manifest = await self._store.read_manifest(trajectory_id)
        if manifest is None:
            return {"success": True, "state": "DRAINED"}
        if manifest.state != "PREPARED":
            raise RuntimeError("trajectory has not been prepared for training")
        if manifest.step_refs:
            await self._store.clear_steps(manifest.step_refs)
            await self._store.write_manifest(replace(manifest, step_refs=[]))
        return {"success": True, "state": "PREPARED"}

    async def release(self, *, trajectory_id: str) -> dict[str, Any]:
        manifest = await self._store.read_manifest(trajectory_id)
        if manifest is None:
            return {"success": True, "state": "DRAINED"}
        return await self._drain(manifest)

    async def status(self) -> dict[str, Any]:
        return {
            "store_id": self._store.store_id,
            "config_fingerprint": self._config_fingerprint,
        }


class TransportCoordinator:
    def __init__(
        self,
        *,
        config_path: str,
        store_id: str,
        build_config: dict[str, Any],
        config_fingerprint: str,
    ):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from ray.util.state import list_actors

        self._store = TransferQueueTrajectoryStore.from_config(
            config_path,
            store_id=store_id,
        )
        self._build_config = dict(build_config)
        self._config_fingerprint = config_fingerprint
        actors = list_actors(
            filters=[("state", "=", "ALIVE")],
            limit=10000,
            detail=True,
            raise_on_missing_output=False,
        )
        storage_node_ids = sorted(
            {
                actor.node_id
                for actor in actors
                if (actor.name or "").startswith("TransferQueueStorageUnit#")
                and actor.node_id
            }
        )
        if not storage_node_ids:
            storage_node_ids = [ray.get_runtime_context().get_node_id()]

        assembler_actor = ray.remote(max_concurrency=1)(TrajectoryAssembler)
        self._assembler_names: list[str] = []
        assembler_handles = []
        for index, node_id in enumerate(storage_node_ids):
            name = f"{ASSEMBLER_NAME_PREFIX}{index}"
            handle = assembler_actor.options(
                name=name,
                namespace=TRANSPORT_NAMESPACE,
                lifetime="detached",
                get_if_exists=True,
                max_restarts=-1,
                max_task_retries=-1,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            ).remote(
                store_id=store_id,
                build_config=build_config,
                config_fingerprint=config_fingerprint,
                cleanup_owner=index == 0,
            )
            self._assembler_names.append(name)
            assembler_handles.append(handle)
        statuses = ray.get([handle.status.remote() for handle in assembler_handles])
        if any(
            status["store_id"] != store_id
            or status["config_fingerprint"] != config_fingerprint
            for status in statuses
        ):
            raise RuntimeError(
                "existing Dressage trajectory assembler configuration does not match"
            )

    def info(self) -> dict[str, Any]:
        return {
            "store_id": self._store.store_id,
            "build_config": dict(self._build_config),
            "config_fingerprint": self._config_fingerprint,
            "ray_namespace": TRANSPORT_NAMESPACE,
            "coordinator_name": COORDINATOR_NAME,
            "assembler_names": list(self._assembler_names),
        }

    def route(self, trajectory_id: str) -> str:
        return assembler_name_for(trajectory_id, self._assembler_names)

    async def register_training_batch(
        self,
        *,
        batch_id: int,
        trajectory_ids: list[str],
        payload_keys: list[str],
    ) -> dict[str, Any]:
        await self._cleanup_expired_training_batches(time.time())
        normalized_trajectories = sorted(set(trajectory_ids))
        normalized_payloads = sorted(set(payload_keys))
        existing = await self._store.read_training_batch(batch_id)
        if existing is not None:
            if (
                existing["trajectory_ids"] != normalized_trajectories
                or existing["payload_keys"] != normalized_payloads
            ):
                raise RuntimeError(
                    "training batch is already registered with different payloads"
                )
            return {"success": True, "registered": False}
        await self._store.write_training_batch(
            batch_id=batch_id,
            trajectory_ids=normalized_trajectories,
            payload_keys=normalized_payloads,
            created_at=time.time(),
        )
        return {"success": True, "registered": True}

    async def _clear_training_batch(
        self,
        batch_id: int,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        import ray

        for trajectory_id in batch["trajectory_ids"]:
            assembler = ray.get_actor(
                self.route(trajectory_id),
                namespace=TRANSPORT_NAMESPACE,
            )
            await assembler.ack.remote(
                trajectory_id=trajectory_id,
            )
        await self._store.clear_training_payloads(batch["payload_keys"])
        await self._store.clear_training_batch_record(batch_id)
        return {"success": True, "state": "DRAINED"}

    async def release_training_trajectories(
        self,
        *,
        trajectory_ids: list[str],
    ) -> dict[str, Any]:
        import ray

        normalized = sorted(set(trajectory_ids))
        for trajectory_id in normalized:
            assembler = ray.get_actor(
                self.route(trajectory_id),
                namespace=TRANSPORT_NAMESPACE,
            )
            await assembler.release.remote(trajectory_id=trajectory_id)
        return {"success": True, "released": len(normalized)}

    async def _cleanup_expired_training_batches(self, now: float) -> None:
        for batch in await self._store.list_training_batches():
            if (
                now - float(batch["created_at"])
                >= TRAINING_PAYLOAD_RETENTION_SECONDS
            ):
                await self._clear_training_batch(int(batch["batch_id"]), batch)

    async def clear_training_batch(self, *, batch_id: int) -> dict[str, Any]:
        batch = await self._store.read_training_batch(batch_id)
        if batch is None:
            return {"success": True, "state": "MISSING"}
        return await self._clear_training_batch(batch_id, batch)


def start_transport_coordinator(
    *,
    config_path: str,
    build_config: TrajectoryBuildConfig,
) -> dict[str, Any]:
    import ray

    if not ray.is_initialized():
        ray.init(address="auto")
    fingerprint = transport_config_fingerprint(config_path, build_config)
    coordinator_actor = ray.remote(max_concurrency=1)(TransportCoordinator)
    coordinator = coordinator_actor.options(
        name=COORDINATOR_NAME,
        namespace=TRANSPORT_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
        max_restarts=-1,
        max_task_retries=-1,
    ).remote(
        config_path=config_path,
        store_id=uuid.uuid4().hex,
        build_config=build_config.to_dict(),
        config_fingerprint=fingerprint,
    )
    info = ray.get(coordinator.info.remote())
    if info["config_fingerprint"] != fingerprint:
        raise RuntimeError(
            "existing Dressage transport coordinator configuration does not match"
        )
    return info
