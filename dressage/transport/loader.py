"""Node-local TransferQueue loader shared by Megatron ranks."""

from __future__ import annotations

import hashlib
from typing import Any

from .assembler import (
    COORDINATOR_NAME,
    TRAINING_LOADER_NAME_PREFIX,
    TRANSPORT_NAMESPACE,
)
from .payload import TrainingPayloadRef
from .store import TransferQueueTrajectoryStore


class TrainingPayloadLoader:
    def __init__(self, *, store_id: str):
        self._store = TransferQueueTrajectoryStore.from_existing(store_id=store_id)
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._read_count = 0

    async def load(
        self,
        *,
        batch_id: int,
        payload_refs: list[dict[str, Any]],
        include_routed_experts: bool,
        include_logprobs: bool,
    ) -> Any:
        import ray
        from slime.utils.misc import Box

        refs = [TrainingPayloadRef.from_dict(value) for value in payload_refs]
        cache_key = (
            batch_id,
            tuple(ref.payload_key for ref in refs),
            include_routed_experts,
            include_logprobs,
        )
        object_ref = self._cache.get(cache_key)
        if object_ref is None:
            payloads = await self._store.read_training_payloads(
                payload_refs,
                include_routed_experts=include_routed_experts,
                include_logprobs=include_logprobs,
            )
            object_ref = ray.put(payloads)
            self._cache[cache_key] = object_ref
            self._read_count += 1
        return Box(object_ref)

    def release(self, *, batch_id: int) -> None:
        self._cache = {
            key: value
            for key, value in self._cache.items()
            if key[0] != batch_id
        }

    def status(self) -> dict[str, Any]:
        return {
            "store_id": self._store.store_id,
            "cached_batches": len(self._cache),
            "read_count": self._read_count,
        }


def get_node_local_loader() -> Any:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    coordinator = ray.get_actor(
        COORDINATOR_NAME,
        namespace=TRANSPORT_NAMESPACE,
    )
    store_id = ray.get(coordinator.info.remote())["store_id"]
    node_id = ray.get_runtime_context().get_node_id()
    suffix = hashlib.sha256(f"{store_id}:{node_id}".encode("utf-8")).hexdigest()[:16]
    name = f"{TRAINING_LOADER_NAME_PREFIX}{suffix}"
    loader_actor = ray.remote(max_concurrency=1, num_cpus=0)(
        TrainingPayloadLoader
    )
    loader = loader_actor.options(
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
    ).remote(store_id=store_id)
    status = ray.get(loader.status.remote())
    if status["store_id"] != store_id:
        raise RuntimeError(
            "existing Dressage training payload loader store ID does not match"
        )
    return loader
