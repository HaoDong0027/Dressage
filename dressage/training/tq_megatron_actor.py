"""Megatron actor that hydrates heavy rollout fields from TransferQueue."""

from __future__ import annotations

from typing import Any

import ray
import torch
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

from dressage.transport.loader import get_node_local_loader
from dressage.transport.payload import TrainingPayloadRef


def hydrate_training_payloads(
    args: Any,
    rollout_data: dict[str, Any],
    payload_refs: list[dict[str, Any] | None],
    payloads: list[dict[str, Any]],
    *,
    include_routed_experts: bool,
    include_logprobs: bool,
) -> None:
    if sum(ref is not None for ref in payload_refs) != len(payloads):
        raise ValueError("TransferQueue payload count does not match training references")

    routed_experts = []
    rollout_logprobs = []
    payload_iterator = iter(payloads)
    for index, value in enumerate(payload_refs):
        if value is None:
            total_length = int(rollout_data["total_lengths"][index])
            response_length = int(rollout_data["response_lengths"][index])
            if include_routed_experts:
                routed_experts.append(
                    torch.zeros(
                        (
                            total_length - 1,
                            int(args.num_layers),
                            int(args.moe_router_topk),
                        ),
                        dtype=torch.int32,
                    )
                )
            if include_logprobs:
                rollout_logprobs.append(
                    torch.zeros(response_length, dtype=torch.float32)
                )
            continue

        payload = next(payload_iterator)
        ref = TrainingPayloadRef.from_dict(value)
        if payload["payload_ref"] != ref.to_dict():
            raise ValueError("TransferQueue payload reference does not match")
        if ref.token_count != int(rollout_data["total_lengths"][index]):
            raise ValueError("TransferQueue token count does not match rollout data")
        if ref.response_length != int(rollout_data["response_lengths"][index]):
            raise ValueError("TransferQueue response length does not match rollout data")
        if ref.response_start + ref.response_length != ref.token_count:
            raise ValueError("TransferQueue response range does not match token count")

        if include_routed_experts:
            num_layers = int(args.num_layers)
            topk = int(args.moe_router_topk)
            expected_rows = ref.token_count - 1
            values = torch.as_tensor(payload["routed_experts"], dtype=torch.int32)
            if list(values.shape) != ref.routed_experts_shape:
                raise ValueError(
                    "TransferQueue routed_experts shape does not match reference"
                )
            if values.ndim != 2 or values.shape[1] != num_layers * topk:
                raise ValueError(
                    "TransferQueue routed_experts width does not match model"
                )
            if values.shape[0] < expected_rows:
                raise ValueError(
                    "TransferQueue routed_experts rows do not match tokens"
                )
            routed_experts.append(
                values[:expected_rows].reshape(expected_rows, num_layers, topk)
            )

        if include_logprobs:
            values = torch.as_tensor(
                payload["full_logprobs"],
                dtype=torch.float32,
            ).reshape(-1)
            if values.numel() < ref.token_count:
                raise ValueError("TransferQueue logprob length does not match tokens")
            rollout_logprobs.append(
                values[
                    ref.response_start : ref.response_start + ref.response_length
                ]
            )

    if include_routed_experts:
        rollout_data["rollout_routed_experts"] = routed_experts
    if include_logprobs:
        device = torch.cuda.current_device()
        rollout_data["rollout_log_probs"] = [
            slice_log_prob_with_cp(logprob, total_length, response_length).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for logprob, total_length, response_length in zip(
                rollout_logprobs,
                rollout_data["total_lengths"],
                rollout_data["response_lengths"],
                strict=True,
            )
        ]


class TQMegatronTrainRayActor(MegatronTrainRayActor):
    """Load R3 and rollout log-probabilities after slime loads light fields."""

    def _get_rollout_data(self, rollout_data_ref):
        rollout_data = super()._get_rollout_data(rollout_data_ref)
        raw_refs = rollout_data.pop("prompt", None)
        if raw_refs is None:
            return rollout_data
        payload_refs = [
            None if value is None else TrainingPayloadRef.from_dict(value).to_dict()
            for value in raw_refs
        ]
        batch_ids = {
            TrainingPayloadRef.from_dict(value).batch_id
            for value in payload_refs
            if value is not None
        }
        if batch_ids and batch_ids != {self._tq_batch_id}:
            raise ValueError("TransferQueue training batch ID does not match rollout")

        include_routed_experts = bool(
            getattr(self.args, "use_rollout_routing_replay", False)
        )
        include_logprobs = True

        stored_refs = [value for value in payload_refs if value is not None]
        payloads = []
        if stored_refs:
            if not hasattr(self, "_tq_loader"):
                self._tq_loader = get_node_local_loader()
            boxed_ref = ray.get(
                self._tq_loader.load.remote(
                    batch_id=self._tq_batch_id,
                    payload_refs=stored_refs,
                    include_routed_experts=include_routed_experts,
                    include_logprobs=include_logprobs,
                )
            )
            payloads = ray.get(boxed_ref.inner)
        hydrate_training_payloads(
            self.args,
            rollout_data,
            payload_refs,
            payloads,
            include_routed_experts=include_routed_experts,
            include_logprobs=include_logprobs,
        )
        return rollout_data

    def train(self, rollout_id: int, rollout_data_ref, external_data=None):
        self._tq_batch_id = rollout_id
        try:
            return super().train(
                rollout_id,
                rollout_data_ref,
                external_data=external_data,
            )
        finally:
            if hasattr(self, "_tq_loader"):
                ray.get(
                    self._tq_loader.release.remote(
                        batch_id=rollout_id,
                    )
                )
