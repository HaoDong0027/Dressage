"""TransferQueue proxy integration tests."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from dressage.proxy.server import create_app, parse_args
from dressage.proxy.session_manager import SessionManager
from dressage.proxy.trajectory_store import TrajectoryStore
from dressage.transport import (
    TrajectoryBuildConfig,
    TransferQueueRuntime,
    TransferQueueTrajectoryStore,
)
from tests.test_proxy import FakeSGLangClient, FakeTokenizer, make_response


class FakeTransferQueue:
    def __init__(self, *, fail_put: bool = False):
        self.partitions = {}
        self.fail_put = fail_put
        self.get_calls = 0

    async def async_kv_put(self, *, key, partition_id, fields):
        if self.fail_put:
            raise RuntimeError("forced TransferQueue put failure")
        self.partitions.setdefault(partition_id, {})[key] = fields["payload"]

    async def async_kv_batch_get(self, *, keys, partition_id, select_fields):
        self.get_calls += 1
        assert select_fields == "payload"
        requested = [keys] if isinstance(keys, str) else keys
        payloads = [
            self.partitions[partition_id][key]
            for key in requested
            if key in self.partitions.get(partition_id, {})
        ]
        return type(
            "FakeTensorDict",
            (),
            {
                "batch_size": [len(payloads)],
                "__getitem__": lambda _, field: payloads,
            },
        )()

    async def async_kv_clear(self, *, keys, partition_id):
        requested = [keys] if isinstance(keys, str) else keys
        for key in requested:
            self.partitions.get(partition_id, {}).pop(key, None)


def make_client(
    tq: FakeTransferQueue,
    *responses,
    token_build_mode: str = "snapshot",
):
    session_manager = SessionManager()
    trajectory_store = TrajectoryStore(min_group_size=1, group_timeout=0.0)
    build_config = TrajectoryBuildConfig(
        token_build_mode=token_build_mode,
        token_build_model="qwen3_5",
        model_mask_type=(
            None if token_build_mode == "tito" else "qwen3_5"
        ),
        tokenizer_path=None,
        record_token_versions=False,
        mask_nonlast_version_tokens=False,
    ).to_dict()
    runtime = TransferQueueRuntime(
        TransferQueueTrajectoryStore(
            tq,
            store_id="proxy-test",
        ),
        transport_info={
            "config_fingerprint": "fingerprint",
            "ray_namespace": "dressage_transport",
            "assembler_names": ["DressageTrajectoryAssembler#0"],
            "build_config": build_config,
        },
    )
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=session_manager,
        trajectory_store=trajectory_store,
        transfer_queue_runtime=runtime,
        sglang_client=FakeSGLangClient(list(responses)),
        model_mask_type="qwen3_5",
        model_tool_call_type="hermes",
        token_build_mode=token_build_mode,
        token_build_model="qwen3_5",
        tito_model="qwen3_5" if token_build_mode == "tito" else None,
    )
    return TestClient(app), session_manager, trajectory_store, runtime


def test_transfer_queue_mode_keeps_proxy_session_light():
    tq = FakeTransferQueue()
    client, session_manager, trajectory_store, runtime = make_client(
        tq,
        make_response("done", finish_reason="length"),
    )

    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-session", "X-Instance-Id": "tq-instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )

    assert generated.status_code == 200
    session = session_manager.get_session("tq-session")
    assert session is not None
    assert len(session.steps) == 1
    assert session.steps[0].all_token_ids == []
    assert session.steps[0].response_routed_experts_chunks == []
    assert session.steps[0].normalized_messages_snapshot
    assert runtime.step_count(session) == 1
    assert session.latest_step.normalized_messages_snapshot[-1]["content"] == "done"
    assert trajectory_store.stats()["total_items"] == 0

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "tq-session", "instance_id": "tq-instance"},
    )
    direct_read = client.post(
        "/trajectory/read",
        json={"trajectory_id": "tq-session"},
    )

    assert finalized.status_code == 200
    assert finalized.json()["num_steps"] == 1
    assert finalized.json()["transport"]["assembler_name"] == (
        "DressageTrajectoryAssembler#0"
    )
    assert direct_read.status_code == 400
    assert runtime.store.manifest_partition in tq.partitions


def test_transfer_queue_mode_preserves_multistep_tito_state():
    tq = FakeTransferQueue()
    client, session_manager, _, runtime = make_client(
        tq,
        make_response("first"),
        make_response("second"),
        token_build_mode="tito",
    )
    first_messages = [{"role": "user", "content": "one"}]
    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-tito", "X-Instance-Id": "tq-instance"},
        json={"model": "fake-model", "messages": first_messages},
    )
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-tito", "X-Instance-Id": "tq-instance"},
        json={
            "model": "fake-model",
            "messages": [
                *first_messages,
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "two"},
            ],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    session = session_manager.get_session("tq-tito")
    assert session is not None
    assert len(session.steps) == 2
    assert all(step.concat_token_ids == [] for step in session.steps)
    assert runtime.step_count(session) == 2
    assert session.latest_step.normalized_messages_snapshot[-1]["content"] == "second"
    assert runtime.current_segment_prefix_tokens(
        session,
        session.steps[-1].lineage_id,
    )

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "tq-tito", "instance_id": "tq-instance"},
    )
    direct_read = client.post(
        "/trajectory/read",
        json={"trajectory_id": "tq-tito"},
    )

    assert finalized.status_code == 200
    assert finalized.json()["num_steps"] == 2
    assert direct_read.status_code == 400
    manifest = tq.partitions[runtime.store.manifest_partition]["tq-tito"]
    assert manifest["num_steps"] == 2
    assert manifest["build_config"]["token_build_mode"] == "tito"


def test_transfer_queue_append_uses_memory_and_branch_reads_only_its_base():
    tq = FakeTransferQueue()
    client, session_manager, _, _ = make_client(
        tq,
        make_response("first"),
        make_response("second"),
        make_response("branch"),
        token_build_mode="tito",
    )
    first_messages = [{"role": "user", "content": "one"}]
    client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-branch", "X-Instance-Id": "instance"},
        json={"model": "fake-model", "messages": first_messages},
    )
    client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-branch", "X-Instance-Id": "instance"},
        json={
            "model": "fake-model",
            "messages": [
                *first_messages,
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "two"},
            ],
        },
    )

    assert tq.get_calls == 0
    branched = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-branch", "X-Instance-Id": "instance"},
        json={
            "model": "fake-model",
            "messages": [
                *first_messages,
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "alternate"},
            ],
        },
    )

    assert branched.status_code == 200
    assert tq.get_calls == 1
    session = session_manager.get_session("tq-branch")
    assert session.steps[-1].route_type == "branch"
    assert session.steps[-1].route_base_step_id == session.steps[0].step_id


def test_discard_session_clears_active_transfer_queue_steps():
    tq = FakeTransferQueue()
    client, session_manager, _, runtime = make_client(tq, make_response("done"))
    client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "discard", "X-Instance-Id": "instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )
    tq.partitions.setdefault(runtime.store.manifest_partition, {})["discard"] = {
        "state": "SEALED"
    }

    discarded = client.post("/session/discard", json={"session_id": "discard"})

    assert discarded.json() == {"success": True}
    assert tq.partitions[runtime.store.step_partition] == {}
    assert tq.partitions[runtime.store.manifest_partition] == {}
    assert session_manager.get_session("discard") is None


def test_transfer_queue_step_write_failure_does_not_commit_runtime_state():
    client, session_manager, _, runtime = make_client(
        FakeTransferQueue(fail_put=True),
        make_response("done"),
    )

    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-fail", "X-Instance-Id": "tq-instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )

    assert generated.status_code == 503
    assert generated.json()["detail"]["error"] == "transfer_queue_step_write_failed"
    session = session_manager.get_session("tq-fail")
    assert session is not None
    assert session.steps == []
    assert runtime.step_count(session) == 0


def test_transfer_queue_local_index_failure_rolls_back_remote_step(monkeypatch):
    tq = FakeTransferQueue()
    client, session_manager, _, runtime = make_client(
        tq,
        make_response("done"),
    )

    def fail_commit_step(**_):
        raise RuntimeError("forced local index failure")

    monkeypatch.setattr(runtime, "commit_step", fail_commit_step)
    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-index-fail", "X-Instance-Id": "instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )

    assert generated.status_code == 503
    session = session_manager.get_session("tq-index-fail")
    assert session is not None
    assert session.steps == []
    assert runtime.step_count(session) == 0
    assert tq.partitions[runtime.store.step_partition] == {}


@pytest.mark.asyncio
async def test_request_cancellation_during_transfer_queue_commit_keeps_step_indexed():
    class BlockingTransferQueue(FakeTransferQueue):
        def __init__(self):
            super().__init__()
            self.write_started = asyncio.Event()
            self.allow_write = asyncio.Event()

        async def async_kv_put(self, *, key, partition_id, fields):
            if key.endswith(":0"):
                self.write_started.set()
                await self.allow_write.wait()
            await super().async_kv_put(
                key=key,
                partition_id=partition_id,
                fields=fields,
            )

    tq = BlockingTransferQueue()
    client, session_manager, _, runtime = make_client(
        tq,
        make_response("done"),
    )
    transport = httpx.ASGITransport(app=client.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        request_task = asyncio.create_task(
            async_client.post(
                "/v1/chat/completions",
                headers={
                    "X-Session-Id": "tq-cancel",
                    "X-Instance-Id": "instance",
                },
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "go"}],
                },
            )
        )
        await tq.write_started.wait()
        request_task.cancel()
        tq.allow_write.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    session = session_manager.get_session("tq-cancel")
    assert session is not None
    assert len(session.steps) == 1
    assert runtime.step_count(session) == 1
    assert len(tq.partitions[runtime.store.step_partition]) == 1


def test_transfer_queue_finalize_failure_preserves_active_session():
    tq = FakeTransferQueue()
    client, session_manager, _, runtime = make_client(tq, make_response("done"))
    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "tq-final-fail", "X-Instance-Id": "tq-instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )
    assert generated.status_code == 200
    tq.fail_put = True
    client = TestClient(client.app, raise_server_exceptions=False)

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "tq-final-fail", "instance_id": "tq-instance"},
    )

    assert finalized.status_code == 500
    session = session_manager.get_session("tq-final-fail")
    assert session is not None
    assert runtime.step_count(session) == 1
    assert session_manager.get_finalization_result("tq-final-fail") is None


def test_assembler_finalize_seals_manifest_without_reading_steps(monkeypatch):
    tq = FakeTransferQueue()
    session_manager = SessionManager()
    store = TransferQueueTrajectoryStore(tq, store_id="assembler-test")
    runtime = TransferQueueRuntime(
        store,
        transport_info={
            "config_fingerprint": "fingerprint",
            "ray_namespace": "dressage_transport",
            "assembler_names": ["DressageTrajectoryAssembler#0"],
            "build_config": TrajectoryBuildConfig(
                token_build_mode="snapshot",
                token_build_model="qwen3_5",
                model_mask_type="qwen3_5",
                tokenizer_path=None,
                record_token_versions=False,
                mask_nonlast_version_tokens=False,
            ).to_dict(),
        },
    )
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=session_manager,
        trajectory_store=TrajectoryStore(),
        transfer_queue_runtime=runtime,
        sglang_client=FakeSGLangClient([make_response("done")]),
        model_mask_type="qwen3_5",
        model_tool_call_type="hermes",
        token_build_mode="snapshot",
        token_build_model="qwen3_5",
    )
    client = TestClient(app)

    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "assembler", "X-Instance-Id": "instance"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )
    finalized = client.post(
        "/session/finalize",
        json={"session_id": "assembler", "instance_id": "instance"},
    )

    assert generated.status_code == 200
    assert finalized.status_code == 200
    result = finalized.json()
    assert "data_path" not in result["transport"]
    assert result["num_segments"] == 1
    assert tq.partitions[store.step_partition]
    manifest = tq.partitions[store.manifest_partition]["assembler"]
    assert manifest["state"] == "SEALED"
    assert not hasattr(store, "trajectory_partition")
    assert tq.get_calls == 0
    assert store.stats()["step_refs"] == 0
    assert session_manager.get_session("assembler") is None

    calls = []

    class Method:
        async def remote(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get_actor=lambda name, namespace: SimpleNamespace(release=Method())
        ),
    )
    assert client.post(
        "/session/discard",
        json={"session_id": "assembler"},
    ).json() == {"success": True}
    assert calls == [{"trajectory_id": "assembler"}]


def test_parse_args_requires_transfer_queue_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--tokenizer-path",
            "fake-tokenizer",
            "--enable-transfer-queue",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_accepts_transfer_queue_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--tokenizer-path",
            "fake-tokenizer",
            "--enable-transfer-queue",
            "--transfer-queue-config",
            "transfer-queue.yaml",
        ],
    )

    args = parse_args()

    assert args.enable_transfer_queue is True
    assert args.transfer_queue_config == "transfer-queue.yaml"
    assert not hasattr(args, "transfer_queue_data_path")
    assert not hasattr(args, "transfer_queue_lease_timeout_seconds")
    assert args.transfer_queue_retention_seconds == 86400.0
