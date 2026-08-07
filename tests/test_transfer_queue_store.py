"""TransferQueue trajectory store tests."""

from __future__ import annotations

import pytest

from dressage.proxy.session_manager import StepRecord
from dressage.transport import TrajectoryManifest, TransferQueueTrajectoryStore


class FakeTensorDict:
    def __init__(self, columns):
        self._columns = columns
        self.batch_size = [len(next(iter(columns.values())))]

    def __getitem__(self, key):
        return self._columns[key]


class FakeTransferQueue:
    def __init__(self, *, raise_on_missing=False):
        self.partitions = {}
        self.raise_on_missing = raise_on_missing

    async def async_kv_put(self, *, key, partition_id, fields):
        self.partitions.setdefault(partition_id, {})[key] = dict(fields)

    async def async_kv_batch_get(self, *, keys, partition_id, select_fields):
        requested = [keys] if isinstance(keys, str) else keys
        partition = self.partitions.get(partition_id, {})
        if self.raise_on_missing and not any(key in partition for key in requested):
            raise ValueError("keys or partition were not found!")
        fields = [select_fields] if isinstance(select_fields, str) else select_fields
        return FakeTensorDict(
            {
                field: [
                    partition[key][field]
                    for key in requested
                    if key in partition
                ]
                for field in fields
            }
        )

    async def async_kv_clear(self, *, keys, partition_id):
        requested = [keys] if isinstance(keys, str) else keys
        partition = self.partitions.get(partition_id, {})
        for key in requested:
            partition.pop(key, None)

    async def async_kv_list(self, *, partition_id):
        return {
            partition_id: {key: {} for key in self.partitions.get(partition_id, {})}
        }


def make_step(index: int) -> StepRecord:
    return StepRecord(
        turn_id=f"turn-{index}",
        request_messages=[{"role": "user", "content": f"question-{index}"}],
        normalized_request_messages=[{"role": "user", "content": f"question-{index}"}],
        prompt_token_ids=[index],
        prompt_token_logprobs=[0.0],
        snapshot_token_ids=[index, index + 1],
        response_token_ids=[index + 1],
        response_logprobs=[-0.1],
        all_token_ids=[index, index + 1],
        all_logprobs=[0.0, -0.1],
        input_token_texts=[str(index)],
        output_token_texts=[str(index + 1)],
        messages_snapshot=[
            {"role": "user", "content": f"question-{index}"},
            {"role": "assistant", "content": f"answer-{index}"},
        ],
        raw_response_text=f"answer-{index}",
        concat_token_ids=[index, index + 1],
        concat_response_logprobs=[0.0, -0.1],
        concat_response_mask=[0, 1],
        response_routed_experts_chunks=[{"data": f"chunk-{index}", "row_count": 1}],
        finish_reason="length" if index else "stop",
        request_version="v1",
        response_version="v1",
        response_versions=["v1"],
    )


@pytest.mark.asyncio
async def test_step_record_round_trip_keeps_all_fields_and_order():
    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="test")
    steps = [make_step(0), make_step(1)]

    refs = [
        await store.write_step(
            session_id="session",
            step_index=index,
            step=step,
        )
        for index, step in enumerate(steps)
    ]

    assert await store.read_steps(refs) == steps
    assert store.stats()["step_refs"] == 2

    await store.discard_step_routed_experts(refs[0])

    stored_steps = await store.read_steps(refs)
    assert stored_steps[0].response_routed_experts_chunks == []
    assert stored_steps[1].response_routed_experts_chunks == [
        {"data": "chunk-1", "row_count": 1}
    ]

    await store.clear_steps(refs)

    assert store.stats()["step_refs"] == 0


@pytest.mark.asyncio
async def test_read_steps_rejects_missing_transfer_queue_data():
    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="test")
    ref = await store.write_step(
        session_id="session",
        step_index=0,
        step=make_step(0),
    )
    tq.partitions[store.step_partition].pop(ref)

    with pytest.raises(RuntimeError, match="unexpected item count"):
        await store.read_steps([ref])


@pytest.mark.asyncio
async def test_manifest_round_trip_does_not_depend_on_process_local_index():
    tq = FakeTransferQueue(raise_on_missing=True)
    writer = TransferQueueTrajectoryStore(tq, store_id="test")
    manifest = TrajectoryManifest(
        session_id="session",
        trajectory_id="trajectory",
        instance_id="instance",
        finalization_id="finalization",
        step_refs=["session:0"],
        num_steps=1,
        num_turns=1,
        num_lineage_segments=1,
        num_timeline_segments=1,
        history_rewritten=False,
        label=1,
        build_config={"token_build_mode": "tito"},
        config_fingerprint="fingerprint",
        sealed_at=1.0,
        retention_seconds=86400.0,
    )

    await writer.write_manifest(manifest)
    reader = TransferQueueTrajectoryStore(tq, store_id="test")

    assert await reader.read_manifest("trajectory") == manifest
    assert await reader.list_manifests() == [manifest]

    await reader.clear_manifest("trajectory")

    assert await reader.read_manifest("trajectory") is None


def test_manifest_rejects_previous_schema_version():
    with pytest.raises(
        RuntimeError,
        match="manifest schema does not match",
    ):
        TrajectoryManifest.from_dict(
            {
                "schema_version": "dressage.transport.manifest/v2",
            }
        )


@pytest.mark.asyncio
async def test_training_payload_round_trip_uses_tensor_fields():
    import torch

    from dressage.transport.payload import TrainingPayloadRef

    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="test")
    payload_ref = TrainingPayloadRef(
        payload_key="trajectory:finalization:0",
        trajectory_id="trajectory",
        segment_id="segment",
        token_count=3,
        response_start=1,
        response_length=2,
        routed_experts_shape=[2, 4],
        batch_id=7,
    ).to_dict()
    routed_experts = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    logprobs = torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32)

    await store.write_training_payload(
        payload_ref=payload_ref,
        routed_experts=routed_experts,
        full_logprobs=logprobs,
    )
    payloads = await store.read_training_payloads(
        [payload_ref],
        include_routed_experts=True,
        include_logprobs=True,
    )

    assert torch.equal(payloads[0]["routed_experts"], routed_experts)
    assert torch.equal(payloads[0]["full_logprobs"], logprobs)

    await store.clear_training_payloads([payload_ref["payload_key"]])

    assert tq.partitions[store.training_payload_partition] == {}


@pytest.mark.asyncio
async def test_training_payload_views_are_detached_from_jagged_batch_storage():
    import torch

    from dressage.transport.payload import TrainingPayloadRef

    refs = [
        TrainingPayloadRef(
            payload_key=f"trajectory:finalization:{index}",
            trajectory_id="trajectory",
            segment_id=f"segment-{index}",
            token_count=rows + 1,
            response_start=1,
            response_length=rows,
            routed_experts_shape=[rows, 4],
            batch_id=7,
        ).to_dict()
        for index, rows in enumerate((2, 4))
    ]
    routed_experts = torch.nested.as_nested_tensor(
        [
            torch.arange(8, dtype=torch.int32).reshape(2, 4),
            torch.arange(16, dtype=torch.int32).reshape(4, 4),
        ],
        layout=torch.jagged,
    )
    logprobs = torch.nested.as_nested_tensor(
        [torch.zeros(3), torch.zeros(5)],
        layout=torch.jagged,
    )

    class JaggedTransferQueue:
        async def async_kv_batch_get(self, **kwargs):
            return FakeTensorDict(
                {
                    "payload": [
                        {
                            "schema_version": ref["schema_version"],
                            "payload_key": ref["payload_key"],
                            "trajectory_id": ref["trajectory_id"],
                            "segment_id": ref["segment_id"],
                            "token_count": ref["token_count"],
                            "routed_experts_shape": ref["routed_experts_shape"],
                        }
                        for ref in refs
                    ],
                    "routed_experts": routed_experts,
                    "full_logprobs": logprobs,
                }
            )

    store = TransferQueueTrajectoryStore(JaggedTransferQueue(), store_id="test")
    payloads = await store.read_training_payloads(
        refs,
        include_routed_experts=True,
        include_logprobs=True,
    )

    for payload in payloads:
        for field in ("routed_experts", "full_logprobs"):
            tensor = payload[field]
            assert tensor.untyped_storage().nbytes() == (
                tensor.numel() * tensor.element_size()
            )


@pytest.mark.asyncio
async def test_training_batch_round_trip_and_clear_are_idempotent():
    tq = FakeTransferQueue()
    store = TransferQueueTrajectoryStore(tq, store_id="test")

    assert await store.read_training_batch(3) is None

    await store.write_training_batch(
        batch_id=3,
        trajectory_ids=["trajectory"],
        payload_keys=["payload"],
        created_at=10.0,
    )

    assert (await store.read_training_batch(3))["trajectory_ids"] == [
        "trajectory"
    ]
    assert (await store.read_training_batch(3))["payload_keys"] == ["payload"]

    await store.clear_training_batch_record(3)
    await store.clear_training_batch_record(3)

    assert await store.read_training_batch(3) is None


@pytest.mark.asyncio
async def test_training_batch_missing_after_list_returns_none():
    tq = FakeTransferQueue(raise_on_missing=True)
    store = TransferQueueTrajectoryStore(tq, store_id="test")

    async def stale_list(*, partition_id):
        return {partition_id: {"3": {}}}

    tq.async_kv_list = stale_list

    assert await store.read_training_batch(3) is None
