"""TransferQueue assembler state-machine tests."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dressage.transport.assembler import (
    TrajectoryAssembler,
    TransportCoordinator,
    assembler_name_for,
    transport_config_fingerprint,
)
from dressage.transport.builder import TrajectoryBuildConfig, TrajectoryBuilder
from dressage.transport.client import (
    prepare_trajectory,
    release_lazy_samples,
)
from dressage.transport.manifest import TrajectoryManifest
from dressage.transport.payload import (
    LAZY_TRAJECTORY_METADATA_KEY,
    TRAINING_PAYLOAD_METADATA_KEY,
    TrainingPayloadRef,
)
from dressage.transport.store import TransferQueueTrajectoryStore
from tests.test_transfer_queue_store import FakeTransferQueue, make_step


def build_config() -> TrajectoryBuildConfig:
    return TrajectoryBuildConfig(
        token_build_mode="tito",
        token_build_model="qwen3.5",
        model_mask_type=None,
        tokenizer_path=None,
        record_token_versions=False,
        mask_nonlast_version_tokens=False,
    )


def test_assembler_routing_is_stable():
    names = [
        "DressageTrajectoryAssembler#0",
        "DressageTrajectoryAssembler#1",
    ]

    assert assembler_name_for("trajectory", names) == assembler_name_for(
        "trajectory",
        names,
    )


def test_transport_fingerprint_includes_manifest_schema(monkeypatch, tmp_path):
    config_path = tmp_path / "transfer_queue.yaml"
    config_path.write_text("controller: {}\n", encoding="utf-8")
    current = transport_config_fingerprint(str(config_path), build_config())

    monkeypatch.setattr(
        "dressage.transport.assembler.MANIFEST_SCHEMA_VERSION",
        "dressage.transport.manifest/test-next",
    )

    assert transport_config_fingerprint(str(config_path), build_config()) != current


def test_tito_builder_preserves_lineage_and_timeline_r3_views():
    from dressage.proxy.session_manager import Lineage, Session

    create_step = replace(
        make_step(0),
        step_id="step-1",
        lineage_id="lineage-1",
        lineage_index=0,
        route_type="create",
        concat_token_ids=[1, 2, 3],
        concat_response_logprobs=[0.0, 0.0, -0.1],
        concat_response_mask=[0, 0, 1],
        concat_context_token_count=2,
        concat_output_token_count=1,
        all_token_ids=[1, 2, 3],
        prompt_token_ids=[1, 2],
        response_token_ids=[3],
        response_logprobs=[-0.1],
        response_routed_experts_chunks=[{"data": "create", "row_count": 2}],
    )
    append_step = replace(
        make_step(1),
        step_id="step-2",
        lineage_id="lineage-1",
        lineage_index=0,
        route_type="append",
        route_base_step_id="step-1",
        concat_token_ids=[4, 5],
        concat_response_logprobs=[0.0, -0.2],
        concat_response_mask=[0, 1],
        concat_context_token_count=1,
        concat_output_token_count=1,
        all_token_ids=[1, 2, 3, 4, 5],
        prompt_token_ids=[1, 2, 3, 4],
        response_token_ids=[5],
        response_logprobs=[-0.2],
        response_routed_experts_chunks=[{"data": "append", "row_count": 2}],
    )
    branch_step = replace(
        make_step(2),
        step_id="step-3",
        lineage_id="lineage-2",
        lineage_index=1,
        route_type="branch",
        route_base_step_id="step-1",
        concat_token_ids=[1, 2, 6],
        concat_response_logprobs=[0.0, 0.0, -0.3],
        concat_response_mask=[0, 0, 1],
        concat_context_token_count=2,
        concat_output_token_count=1,
        all_token_ids=[1, 2, 6],
        prompt_token_ids=[1, 2],
        response_token_ids=[6],
        response_logprobs=[-0.3],
        response_routed_experts_chunks=[{"data": "branch", "row_count": 2}],
    )
    session = Session(
        session_id="trajectory",
        instance_id="instance",
        steps=[create_step, append_step, branch_step],
        lineages={
            "lineage-1": Lineage("lineage-1", 0, "step-2"),
            "lineage-2": Lineage("lineage-2", 1, "step-3", "step-1"),
        },
        steps_by_id={
            step.step_id: step
            for step in (create_step, append_step, branch_step)
        },
    )
    builder = TrajectoryBuilder(build_config())

    lineage = builder.build_records(
        session=session,
        trajectory_id="trajectory",
        instance_id="instance",
        finalization_id="finalization",
        label=1,
        segment_view="lineage",
    )
    timeline = builder.build_records(
        session=session,
        trajectory_id="trajectory",
        instance_id="instance",
        finalization_id="finalization",
        label=1,
        segment_view="timeline",
    )

    assert len(lineage) == 2
    assert len(timeline) == 3
    assert [
        item["row_count"] for item in lineage[0]["routed_experts_chunks"]
    ] == [2, 2]
    assert [
        item["row_count"] for item in timeline[1]["routed_experts_chunks"]
    ] == [2, 2]
    assert [item["row_count"] for item in timeline[2]["routed_experts_chunks"]] == [2]


async def create_assembler(monkeypatch):
    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="assembler")
    monkeypatch.setattr(
        TransferQueueTrajectoryStore,
        "from_existing",
        classmethod(lambda cls, **kwargs: store),
    )
    assembler = TrajectoryAssembler(
        store_id="assembler",
        build_config=build_config().to_dict(),
        config_fingerprint="fingerprint",
        cleanup_owner=False,
    )
    refs = [
        await store.write_step(
            session_id="trajectory",
            step_index=index,
            step=replace(
                make_step(index),
                response_routed_experts_chunks=[],
            ),
        )
        for index in range(2)
    ]
    manifest = TrajectoryManifest(
        session_id="trajectory",
        trajectory_id="trajectory",
        instance_id="instance",
        finalization_id="finalization",
        step_refs=refs,
        num_steps=2,
        num_turns=2,
        num_lineage_segments=1,
        num_timeline_segments=2,
        history_rewritten=False,
        label=1,
        build_config=build_config().to_dict(),
        config_fingerprint="fingerprint",
        sealed_at=10.0,
        retention_seconds=86400.0,
    )
    await store.write_manifest(manifest)
    return tq, store, assembler


@pytest.mark.asyncio
async def test_prepare_claims_the_whole_trajectory_and_ack_clears_steps(monkeypatch):
    tq, store, assembler = await create_assembler(monkeypatch)

    first = await assembler.prepare(
        trajectory_id="trajectory",
        instance_id="instance",
    )
    second = await assembler.prepare(
        trajectory_id="trajectory",
        instance_id="instance",
        segment_view="timeline",
    )

    assert isinstance(first["data"][0]["timestamp"], float)
    assert "training_payload_ref" in first["data"][0]
    assert second["data"] == []
    assert second["drained"] is True

    assert (
        await assembler.ack(
            trajectory_id="trajectory",
        )
    )["state"] == "DRAINED"
    assert tq.partitions[store.step_partition] == {}
    assert await store.read_manifest("trajectory") is None
    assert (
        await assembler.ack(
            trajectory_id="trajectory",
        )
    )["state"] == "DRAINED"


@pytest.mark.asyncio
async def test_prepare_failure_restores_sealed_manifest(monkeypatch):
    tq, store, assembler = await create_assembler(monkeypatch)
    tq.partitions[store.step_partition].pop("trajectory:1")

    with pytest.raises(RuntimeError, match="unexpected item count"):
        await assembler.prepare(
            trajectory_id="trajectory",
        )

    manifest = await store.read_manifest("trajectory")
    assert manifest.state == "SEALED"


@pytest.mark.asyncio
async def test_release_clears_unconsumed_sealed_trajectory(monkeypatch):
    tq, store, assembler = await create_assembler(monkeypatch)

    result = await assembler.release(trajectory_id="trajectory")

    assert result["state"] == "DRAINED"
    assert tq.partitions[store.step_partition] == {}
    assert await store.read_manifest("trajectory") is None


@pytest.mark.asyncio
async def test_drain_failure_keeps_manifest_for_retry(monkeypatch):
    _, store, assembler = await create_assembler(monkeypatch)
    clear_training_payloads = store.clear_training_payloads

    async def fail_clear_training_payloads(payload_keys):
        raise RuntimeError("clear failed")

    monkeypatch.setattr(
        store,
        "clear_training_payloads",
        fail_clear_training_payloads,
    )
    with pytest.raises(RuntimeError, match="clear failed"):
        await assembler.release(trajectory_id="trajectory")

    assert (await store.read_manifest("trajectory")).state == "DRAINING"

    monkeypatch.setattr(
        store,
        "clear_training_payloads",
        clear_training_payloads,
    )
    assert (await assembler.release(trajectory_id="trajectory"))["state"] == "DRAINED"
    assert await store.read_manifest("trajectory") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_method", ["ack", "release"])
async def test_lazy_prepare_moves_heavy_fields_to_training_payload(
    monkeypatch,
    drain_method,
):
    import base64

    import numpy as np

    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="assembler")
    monkeypatch.setattr(
        TransferQueueTrajectoryStore,
        "from_existing",
        classmethod(lambda cls, **kwargs: store),
    )
    assembler = TrajectoryAssembler(
        store_id="assembler",
        build_config=build_config().to_dict(),
        config_fingerprint="fingerprint",
        cleanup_owner=False,
    )
    routed_experts = np.arange(4, dtype=np.int32).reshape(1, 4)
    step = replace(
        make_step(0),
        response_routed_experts_chunks=[
            {
                "data": base64.b64encode(routed_experts.tobytes()).decode("ascii"),
                "row_count": 1,
            }
        ],
    )
    step_ref = await store.write_step(
        session_id="trajectory",
        step_index=0,
        step=step,
    )
    await store.write_manifest(
        TrajectoryManifest(
            session_id="trajectory",
            trajectory_id="trajectory",
            instance_id="instance",
            finalization_id="finalization",
            step_refs=[step_ref],
            num_steps=1,
            num_turns=1,
            num_lineage_segments=1,
            num_timeline_segments=1,
            history_rewritten=False,
            label=1,
            build_config=build_config().to_dict(),
            config_fingerprint="fingerprint",
            sealed_at=10.0,
            retention_seconds=86400.0,
        )
    )

    payload = await assembler.prepare(
        trajectory_id="trajectory",
    )
    record = payload["data"][0]
    payload_ref = record["training_payload_ref"]

    assert "full_logprobs" not in record
    assert "routed_experts_chunks" not in record
    stored = await store.read_training_payloads(
        [payload_ref],
        include_routed_experts=True,
        include_logprobs=True,
    )
    assert stored[0]["routed_experts"].shape == (1, 4)
    assert stored[0]["full_logprobs"].shape == (2,)

    await assembler.ack_steps(
        trajectory_id="trajectory",
    )
    await assembler.ack_steps(
        trajectory_id="trajectory",
    )

    manifest = await store.read_manifest("trajectory")
    assert manifest.step_refs == []
    assert tq.partitions[store.step_partition] == {}
    assert tq.partitions[store.training_payload_partition]
    consumed = await assembler.prepare(
        trajectory_id="trajectory",
    )
    assert consumed["data"] == []
    assert consumed["drained"] is True

    if drain_method == "ack":
        await assembler.ack(
            trajectory_id="trajectory",
        )
    else:
        await assembler.release(trajectory_id="trajectory")

    assert tq.partitions[store.step_partition] == {}
    assert tq.partitions[store.training_payload_partition] == {}
    assert await store.read_manifest("trajectory") is None


@pytest.mark.asyncio
async def test_active_manifest_and_payload_expire_after_retention(monkeypatch):
    tq, store, assembler = await create_assembler(monkeypatch)
    assembler._cleanup_owner = True
    manifest = await store.read_manifest("trajectory")
    await store.write_manifest(
        replace(
            manifest,
            training_payload_keys=["payload"],
        )
    )
    tq.partitions[store.training_payload_partition] = {
        "payload": {"payload": {}}
    }

    await assembler._cleanup_expired(86411.0)

    assert await store.read_manifest("trajectory") is None
    assert tq.partitions[store.step_partition] == {}
    assert tq.partitions[store.training_payload_partition] == {}


@pytest.mark.asyncio
async def test_training_batch_clear_acks_once_and_is_idempotent(monkeypatch):
    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="assembler")
    coordinator = TransportCoordinator.__new__(TransportCoordinator)
    coordinator._store = store
    coordinator._assembler_names = ["DressageTrajectoryAssembler#0"]
    calls = []

    class Method:
        async def remote(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "state": "DRAINED"}

    actor = SimpleNamespace(ack=Method())
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get_actor=lambda name, namespace: actor,
        ),
    )

    await coordinator.register_training_batch(
        batch_id=7,
        trajectory_ids=["trajectory", "trajectory"],
        payload_keys=["payload", "payload"],
    )
    first = await coordinator.clear_training_batch(batch_id=7)
    second = await coordinator.clear_training_batch(batch_id=7)

    assert first["state"] == "DRAINED"
    assert second["state"] == "MISSING"
    assert calls == [
        {
            "trajectory_id": "trajectory",
        }
    ]


@pytest.mark.asyncio
async def test_release_lazy_samples_deduplicates_trajectories(monkeypatch):
    calls = []

    class Method:
        async def remote(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "state": "DRAINED"}

    coordinator = SimpleNamespace(release_training_trajectories=Method())
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get_actor=lambda name, namespace: coordinator,
        ),
    )
    payload_ref = TrainingPayloadRef(
        payload_key="trajectory:finalization:0",
        trajectory_id="trajectory",
        segment_id="segment",
        token_count=2,
        response_start=1,
        response_length=1,
        routed_experts_shape=[1, 4],
    ).to_dict()
    samples = [
        SimpleNamespace(metadata={TRAINING_PAYLOAD_METADATA_KEY: payload_ref}),
        SimpleNamespace(metadata={TRAINING_PAYLOAD_METADATA_KEY: payload_ref}),
        SimpleNamespace(metadata={LAZY_TRAJECTORY_METADATA_KEY: "handoff-only"}),
        SimpleNamespace(metadata={}),
    ]

    await release_lazy_samples(samples)

    assert calls == [{"trajectory_ids": ["handoff-only", "trajectory"]}]


@pytest.mark.asyncio
async def test_direct_client_keeps_non_transfer_queue_read():
    calls = []

    class Proxy:
        async def read_trajectory(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "data": [{"tokens": [1]}]}

    payload = await prepare_trajectory(
        Proxy(),
        {"success": True},
        trajectory_id="trajectory",
        instance_id="instance",
        segment_view="lineage",
        drain=True,
    )

    assert payload["data"] == [{"tokens": [1]}]
    assert calls == [
        {
            "trajectory_id": "trajectory",
            "instance_id": "instance",
            "segment_view": "lineage",
            "drain": True,
        }
    ]


@pytest.mark.asyncio
async def test_direct_client_releases_lazy_steps_after_prepare(monkeypatch):
    calls = []

    class Method:
        def __init__(self, name):
            self.name = name

        async def remote(self, **kwargs):
            calls.append((self.name, kwargs))
            return {"success": True, "data": [{"tokens": [1]}]}

    actor = SimpleNamespace(
        prepare=Method("prepare"),
        ack_steps=Method("ack_steps"),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get_actor=lambda name, namespace: actor,
        ),
    )
    handle = {
        "trajectory_id": "trajectory",
        "assembler_name": "DressageTrajectoryAssembler#0",
        "ray_namespace": "dressage_transport",
    }

    payload = await prepare_trajectory(
        object(),
        {"transport": handle},
        trajectory_id="trajectory",
        instance_id="instance",
        drain=True,
    )

    assert payload["data"] == [{"tokens": [1]}]
    assert calls == [
        (
            "prepare",
            {
                "trajectory_id": "trajectory",
                "instance_id": "instance",
                "segment_view": None,
            },
        ),
        (
            "ack_steps",
            {
                "trajectory_id": "trajectory",
            },
        ),
    ]


@pytest.mark.skipif(
    os.environ.get("DRESSAGE_RUN_TRANSFER_QUEUE_E2E") != "1",
    reason="requires a real Ray and TransferQueue runtime",
)
def test_real_transfer_queue_two_storage_units(tmp_path):
    ray = pytest.importorskip("ray")
    pytest.importorskip("transfer_queue")
    from ray.util.state import list_actors

    from dressage.transport.assembler import start_transport_coordinator

    config_path = tmp_path / "transfer_queue.yaml"
    config_path.write_text(
        """controller:
  polling_mode: true
backend:
  storage_backend: SimpleStorage
  SimpleStorage:
    total_storage_size: null
    num_data_storage_units: 2
""",
        encoding="utf-8",
    )
    ray.init(num_cpus=8)
    try:
        info = start_transport_coordinator(
            config_path=str(config_path),
            build_config=build_config(),
        )
        actors = list_actors(
            filters=[("state", "=", "ALIVE")],
            limit=10000,
            detail=True,
            raise_on_missing_output=False,
        )
        storage_units = [
            actor
            for actor in actors
            if (actor.name or "").startswith("TransferQueueStorageUnit#")
        ]
        assembler_nodes = {
            actor.node_id for actor in actors if actor.name in info["assembler_names"]
        }
        storage_nodes = {actor.node_id for actor in storage_units}

        assert len(storage_units) == 2
        assert assembler_nodes == storage_nodes
    finally:
        ray.shutdown()
