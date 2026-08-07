"""Public-compatible trajectory assembly for TransferQueue workers."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dressage.proxy.session_manager import Session, StepRecord

_INPUT_TOKEN_VERSION = "-1"


@dataclass(frozen=True)
class TrajectoryBuildConfig:
    token_build_mode: Literal["snapshot", "tito"]
    token_build_model: str
    model_mask_type: str | None
    tokenizer_path: str | None
    record_token_versions: bool
    mask_nonlast_version_tokens: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrajectoryBuildConfig":
        return cls(**value)


def _ordered_turn_ids(steps: list[Any]) -> list[str]:
    turn_ids: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step.turn_id in seen:
            continue
        seen.add(step.turn_id)
        turn_ids.append(step.turn_id)
    return turn_ids


def split_session_into_lineage_segments(session: "Session") -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    sorted_lineages = sorted(session.lineages.values(), key=lambda item: item.index)
    for lineage in sorted_lineages:
        lineage_steps = [
            step for step in session.steps if step.lineage_id == lineage.id
        ]
        current_steps: list[Any] = []
        current_segment_reasons: list[str] = ["initial"]

        def flush_segment() -> None:
            nonlocal current_steps, current_segment_reasons
            if not current_steps:
                return
            segments.append(
                {
                    "steps": current_steps,
                    "segment_reason": current_segment_reasons[0],
                    "segment_reasons": list(current_segment_reasons),
                    "turn_ids": _ordered_turn_ids(current_steps),
                    "lineage_id": lineage.id,
                    "lineage_index": lineage.index,
                    "branch_from_step_id": lineage.branch_from_step_id,
                }
            )
            current_steps = []
            current_segment_reasons = ["initial"]

        for step in lineage_steps:
            if step.lineage_segment_boundary_before and current_steps:
                flush_segment()
                current_segment_reasons = list(
                    step.lineage_segment_reasons_before or ["initial"]
                )
            current_steps.append(step)
        flush_segment()
    return segments


def split_session_into_timeline_segments(session: "Session") -> list[dict[str, Any]]:
    return [
        {
            "steps": [step],
            "segment_reason": step.segment_reason_before or "initial",
            "segment_reasons": list(step.segment_reasons_before or ["initial"]),
            "turn_ids": [step.turn_id],
            "lineage_id": step.lineage_id,
            "lineage_index": step.lineage_index,
            "route_type": step.route_type,
            "route_base_step_id": step.route_base_step_id,
        }
        for step in session.steps
    ]


def _normalize_logprobs_to_length(
    values: list[float],
    token_count: int,
) -> tuple[list[float], bool]:
    invalid = len(values) != token_count
    normalized: list[float] = []
    for index in range(token_count):
        if index >= len(values):
            normalized.append(0.0)
            continue
        try:
            normalized.append(float(values[index]))
        except (TypeError, ValueError):
            normalized.append(0.0)
            invalid = True
    return normalized, invalid


def _lineage_routed_experts_chunks_until(
    session: "Session",
    target_step: "StepRecord",
) -> list[dict[str, Any]]:
    lineage_steps = [
        step for step in session.steps if step.lineage_id == target_step.lineage_id
    ]
    target_index = next(
        index for index, step in enumerate(lineage_steps) if step.step_id == target_step.step_id
    )
    start_index = 0
    for index, step in enumerate(lineage_steps[: target_index + 1]):
        if step.lineage_segment_boundary_before:
            start_index = index
    return [
        dict(chunk)
        for step in lineage_steps[start_index : target_index + 1]
        for chunk in step.response_routed_experts_chunks
    ]


class TrajectoryBuilder:
    def __init__(self, config: TrajectoryBuildConfig):
        self.config = config

    def build_records(
        self,
        *,
        session: "Session",
        trajectory_id: str,
        instance_id: str,
        finalization_id: str,
        label: Any | None,
        segment_view: str | None = None,
        sealed_at: float | None = None,
    ) -> list[dict[str, Any]]:
        timestamp = time.time() if sealed_at is None else sealed_at
        expected_view = "lineage" if self.config.token_build_mode == "tito" else "timeline"
        selected_view = segment_view or expected_view
        if selected_view not in {"lineage", "timeline"}:
            raise ValueError("segment_view must be 'lineage' or 'timeline'")
        if self.config.token_build_mode == "snapshot" and selected_view != "timeline":
            return []

        if selected_view == "lineage":
            segments = split_session_into_lineage_segments(session)
        else:
            segments = split_session_into_timeline_segments(session)

        records: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            if selected_view == "lineage":
                record = self._build_lineage_record(
                    session=session,
                    trajectory_id=trajectory_id,
                    segment=segment,
                    segment_index=segment_index,
                    segment_count=len(segments),
                    instance_id=instance_id,
                    label=label,
                    finalization_id=finalization_id,
                    timestamp=timestamp,
                )
            else:
                record = self._build_timeline_record(
                    session=session,
                    trajectory_id=trajectory_id,
                    segment=segment,
                    segment_index=segment_index,
                    segment_count=len(segments),
                    instance_id=instance_id,
                    label=label,
                    finalization_id=finalization_id,
                    timestamp=timestamp,
                )
            record["extra_info"].update(
                {
                    "finalization_complete": True,
                    "finalization_id": finalization_id,
                    "segment_view": selected_view,
                    "token_build_mode": self.config.token_build_mode,
                }
            )
            records.append(record)
        return records

    @staticmethod
    def _uid(finalization_id: str, segment_view: str, segment_index: int) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"dressage:{finalization_id}:{segment_view}:{segment_index}",
            )
        )

    def _build_timeline_record(
        self,
        *,
        session: "Session",
        trajectory_id: str,
        segment: dict[str, Any],
        segment_index: int,
        segment_count: int,
        instance_id: str,
        label: Any | None,
        finalization_id: str,
        timestamp: float,
    ) -> dict[str, Any]:
        step = segment["steps"][-1]
        tokens = list(step.all_token_ids)
        prompt_len = min(len(step.prompt_token_ids), len(tokens))
        response_len = max(0, len(tokens) - prompt_len)
        output_logprobs, invalid = _normalize_logprobs_to_length(
            list(step.response_logprobs),
            response_len,
        )
        full_logprobs = [0.0] * prompt_len + output_logprobs
        full_loss_mask = [0] * prompt_len + [1] * response_len

        extra_info = {
            "alignment_method": "snapshot_step",
            "token_build_mode": self.config.token_build_mode,
            "segment_view": "timeline",
            "context_token_count": prompt_len,
            "output_token_count": response_len,
            "num_steps": 1,
            "num_turns": len(segment["turn_ids"]),
            "turn_ids": segment["turn_ids"],
            "step_id": step.step_id,
            "lineage_id": step.lineage_id,
            "lineage_index": step.lineage_index,
            "route_type": step.route_type,
            "route_base_step_id": step.route_base_step_id,
            "timestamp": str(timestamp),
            "history_rewritten": session.history_rewritten,
            "segment_reason": segment["segment_reason"],
            "segment_reasons": segment["segment_reasons"],
            "trajectory_num_segments": segment_count,
        }
        if invalid:
            extra_info["snapshot_logprobs_invalid"] = True
        if self.config.mask_nonlast_version_tokens:
            extra_info["mask_nonlast_version_tokens"] = True

        record: dict[str, Any] = {
            "uid": self._uid(finalization_id, "timeline", segment_index),
            "trajectory_id": trajectory_id,
            "turn_id": step.turn_id,
            "instance_id": instance_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "messages": step.messages_snapshot,
            "tools": step.tools,
            "tokens": tokens,
            "full_logprobs": full_logprobs,
            "full_loss_mask": full_loss_mask,
            "aligned_response_length": sum(full_loss_mask),
            "label": label,
            "finish_reason": step.finish_reason,
            "timestamp": timestamp,
            "extra_info": extra_info,
        }
        if self.config.record_token_versions:
            response_version = (
                step.response_version
                or (step.response_versions[-1] if step.response_versions else None)
                or step.request_version
                or "unknown"
            )
            record["full_versions"] = [_INPUT_TOKEN_VERSION] * prompt_len + [
                str(response_version)
            ] * response_len

        chunks = (
            _lineage_routed_experts_chunks_until(session, step)
            if self.config.token_build_mode == "tito"
            else [dict(chunk) for chunk in step.response_routed_experts_chunks]
        )
        if chunks:
            record["routed_experts_chunks"] = chunks
        return record

    def _build_lineage_record(
        self,
        *,
        session: "Session",
        trajectory_id: str,
        segment: dict[str, Any],
        segment_index: int,
        segment_count: int,
        instance_id: str,
        label: Any | None,
        finalization_id: str,
        timestamp: float,
    ) -> dict[str, Any]:
        steps = segment["steps"]
        base_step = steps[-1]
        turn_ids = segment["turn_ids"]
        tokens = [token for step in steps for token in step.concat_token_ids]
        full_logprobs = [
            value for step in steps for value in step.concat_response_logprobs
        ]
        full_loss_mask = [
            value for step in steps for value in step.concat_response_mask
        ]
        if not (len(tokens) == len(full_logprobs) == len(full_loss_mask)):
            raise RuntimeError(
                "tito segment arrays are not aligned: "
                f"tokens={len(tokens)}, full_logprobs={len(full_logprobs)}, "
                f"full_loss_mask={len(full_loss_mask)}"
            )

        full_versions = [
            version for step in steps for version in step.concat_versions
        ]
        if self.config.record_token_versions and len(full_versions) != len(tokens):
            raise RuntimeError(
                "tito segment versions do not match the token count"
            )

        context_token_count = sum(
            step.concat_context_token_count for step in steps
        )
        output_token_count = sum(step.concat_output_token_count for step in steps)
        extra_info = {
            "alignment_method": "tito",
            "token_build_mode": "tito",
            "segment_view": "lineage",
            "context_token_count": context_token_count,
            "context_delta_token_count": context_token_count,
            "output_token_count": output_token_count,
            "num_steps": len(steps),
            "num_turns": len(turn_ids),
            "turn_ids": turn_ids,
            "lineage_id": segment["lineage_id"],
            "lineage_index": segment["lineage_index"],
            "branch_from_step_id": segment["branch_from_step_id"],
            "step_ids": [step.step_id for step in steps],
            "timestamp": str(timestamp),
            "history_rewritten": session.history_rewritten,
            "segment_reason": segment["segment_reason"],
            "segment_reasons": segment["segment_reasons"],
            "trajectory_num_segments": segment_count,
        }
        if any(step.concat_logprobs_invalid for step in steps):
            extra_info["tito_logprobs_invalid"] = True
        if any(step.concat_incremental_tokenization_failed for step in steps):
            extra_info["tito_incremental_tokenization_failed"] = True
        if self.config.mask_nonlast_version_tokens:
            extra_info["mask_nonlast_version_tokens"] = True

        record: dict[str, Any] = {
            "uid": self._uid(finalization_id, "lineage", segment_index),
            "trajectory_id": trajectory_id,
            "turn_id": turn_ids[-1],
            "instance_id": instance_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "messages": base_step.messages_snapshot,
            "tools": base_step.tools,
            "tokens": tokens,
            "full_logprobs": full_logprobs,
            "full_loss_mask": full_loss_mask,
            "aligned_response_length": sum(full_loss_mask),
            "label": label,
            "finish_reason": base_step.finish_reason,
            "timestamp": timestamp,
            "extra_info": extra_info,
        }
        if self.config.record_token_versions:
            record["full_versions"] = full_versions

        chunks = [
            dict(chunk)
            for step in steps
            for chunk in step.response_routed_experts_chunks
        ]
        if chunks:
            record["routed_experts_chunks"] = chunks
        return record
