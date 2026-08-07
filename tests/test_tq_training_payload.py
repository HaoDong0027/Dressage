"""Training-side lazy TransferQueue payload tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("slime.backends.megatron_utils.actor")

from dressage.training import tq_megatron_actor
from dressage.training.tq_megatron_actor import hydrate_training_payloads
from dressage.transport.payload import TrainingPayloadRef


def test_hydrate_training_payloads_restores_r3_after_truncation():
    payload_ref = TrainingPayloadRef(
        payload_key="trajectory:finalization:0",
        trajectory_id="trajectory",
        segment_id="segment",
        token_count=3,
        response_start=1,
        response_length=2,
        routed_experts_shape=[3, 4],
        batch_id=7,
    ).to_dict()
    routed_experts = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
    }

    hydrate_training_payloads(
        SimpleNamespace(num_layers=2, moe_router_topk=2),
        rollout_data,
        [payload_ref],
        [
            {
                "payload_ref": payload_ref,
                "routed_experts": routed_experts,
            }
        ],
        include_routed_experts=True,
        include_logprobs=False,
    )

    restored = rollout_data["rollout_routed_experts"][0]
    assert restored.shape == (2, 2, 2)
    assert torch.equal(restored.reshape(2, 4), routed_experts[:2])


def test_hydrate_training_payloads_restores_response_logprobs(monkeypatch):
    payload_ref = TrainingPayloadRef(
        payload_key="trajectory:finalization:0",
        trajectory_id="trajectory",
        segment_id="segment",
        token_count=3,
        response_start=1,
        response_length=2,
        routed_experts_shape=[3, 4],
        batch_id=7,
    ).to_dict()
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
    }
    monkeypatch.setattr(
        tq_megatron_actor,
        "slice_log_prob_with_cp",
        lambda values, total_length, response_length: values,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    hydrate_training_payloads(
        SimpleNamespace(num_layers=2, moe_router_topk=2),
        rollout_data,
        [payload_ref],
        [
            {
                "payload_ref": payload_ref,
                "full_logprobs": torch.tensor(
                    [0.0, -0.1, -0.2, -0.3],
                    dtype=torch.float32,
                ),
            }
        ],
        include_routed_experts=False,
        include_logprobs=True,
    )

    assert torch.equal(
        rollout_data["rollout_log_probs"][0],
        torch.tensor([-0.1, -0.2], dtype=torch.float32),
    )


def test_hydrate_training_payloads_fills_removed_sample_to_expected_lengths(
    monkeypatch,
):
    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [2],
    }

    def assert_logprob_length(values, total_length, response_length):
        assert len(values) == response_length
        return values

    monkeypatch.setattr(
        tq_megatron_actor,
        "slice_log_prob_with_cp",
        assert_logprob_length,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    hydrate_training_payloads(
        SimpleNamespace(num_layers=2, moe_router_topk=2),
        rollout_data,
        [None],
        [],
        include_routed_experts=True,
        include_logprobs=True,
    )

    assert rollout_data["rollout_routed_experts"][0].shape == (3, 2, 2)
    assert rollout_data["rollout_log_probs"][0].shape == (2,)


def test_tq_actor_hydrates_all_removed_shard_without_loading_tq(monkeypatch):
    rollout_data = {
        "prompt": [None],
        "total_lengths": [4],
        "response_lengths": [2],
    }
    monkeypatch.setattr(
        tq_megatron_actor.MegatronTrainRayActor,
        "_get_rollout_data",
        lambda self, rollout_data_ref: dict(rollout_data),
    )
    monkeypatch.setattr(
        tq_megatron_actor,
        "get_node_local_loader",
        lambda: pytest.fail("An all-removed shard must not access TransferQueue"),
    )
    monkeypatch.setattr(
        tq_megatron_actor,
        "slice_log_prob_with_cp",
        lambda values, total_length, response_length: values,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    actor = object.__new__(tq_megatron_actor.TQMegatronTrainRayActor)
    actor.args = SimpleNamespace(
        use_rollout_routing_replay=True,
        num_layers=2,
        moe_router_topk=2,
    )
    actor._tq_batch_id = 7

    result = actor._get_rollout_data(None)

    assert "prompt" not in result
    assert result["rollout_routed_experts"][0].shape == (3, 2, 2)
    assert result["rollout_log_probs"][0].shape == (2,)
