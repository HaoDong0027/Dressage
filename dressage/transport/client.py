"""Direct Rollout-to-Assembler trajectory reads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


async def prepare_trajectory(
    proxy_client: Any,
    finalization_result: dict[str, Any],
    *,
    trajectory_id: str,
    instance_id: str | None,
    segment_view: str | None = None,
    drain: bool,
) -> dict[str, Any]:
    handle = finalization_result.get("transport")
    if not isinstance(handle, dict):
        read_kwargs = {
            "trajectory_id": trajectory_id,
            "instance_id": instance_id,
            "drain": drain,
        }
        if segment_view is not None:
            read_kwargs["segment_view"] = segment_view
        return await proxy_client.read_trajectory(**read_kwargs)

    import ray

    assembler = ray.get_actor(
        handle["assembler_name"],
        namespace=handle["ray_namespace"],
    )
    prepare_kwargs = {
        "trajectory_id": trajectory_id,
        "instance_id": instance_id,
        "segment_view": segment_view,
    }
    payload = await assembler.prepare.remote(
        **prepare_kwargs,
    )
    if payload.get("data"):
        await assembler.ack_steps.remote(
            trajectory_id=trajectory_id,
        )
    return payload


async def release_lazy_samples(samples: Iterable[Any]) -> None:
    from .payload import (
        LAZY_TRAJECTORY_METADATA_KEY,
        TRAINING_PAYLOAD_METADATA_KEY,
    )

    trajectory_ids = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        value = metadata.get(TRAINING_PAYLOAD_METADATA_KEY)
        if value is not None:
            trajectory_ids.append(str(value["trajectory_id"]))
        owner_id = metadata.get(LAZY_TRAJECTORY_METADATA_KEY)
        if owner_id is not None:
            trajectory_ids.append(str(owner_id))
    trajectory_ids = sorted(set(trajectory_ids))
    if not trajectory_ids:
        return

    import ray

    from .assembler import COORDINATOR_NAME, TRANSPORT_NAMESPACE

    coordinator = ray.get_actor(COORDINATOR_NAME, namespace=TRANSPORT_NAMESPACE)
    await coordinator.release_training_trajectories.remote(
        trajectory_ids=trajectory_ids,
    )


def register_training_batch(
    batch_id: int,
    trajectory_ids: list[str],
    payload_keys: list[str],
) -> dict[str, Any]:
    import ray

    from .assembler import COORDINATOR_NAME, TRANSPORT_NAMESPACE

    coordinator = ray.get_actor(
        COORDINATOR_NAME,
        namespace=TRANSPORT_NAMESPACE,
    )
    return ray.get(
        coordinator.register_training_batch.remote(
            batch_id=batch_id,
            trajectory_ids=trajectory_ids,
            payload_keys=payload_keys,
        )
    )


def clear_training_batch(batch_id: int) -> dict[str, Any]:
    import ray

    from .assembler import COORDINATOR_NAME, TRANSPORT_NAMESPACE

    coordinator = ray.get_actor(
        COORDINATOR_NAME,
        namespace=TRANSPORT_NAMESPACE,
    )
    return ray.get(coordinator.clear_training_batch.remote(batch_id=batch_id))
