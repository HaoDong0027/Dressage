"""Transport runtime and proxy integration helpers."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dressage.proxy.session_manager import Session, SessionManager, StepRecord

from .assembler import assembler_name_for
from .builder import split_session_into_lineage_segments
from .manifest import TrajectoryManifest
from .store import TransferQueueTrajectoryStore

logger = logging.getLogger(__name__)

_NON_REAL_TOKEN_VERSIONS = {"", "-1", "unknown", "none"}


def _real_token_version(value: Any) -> str | None:
    if value is None:
        return None
    version = str(value)
    if version.strip().lower() in _NON_REAL_TOKEN_VERSIONS:
        return None
    return version


@dataclass
class TITOState:
    """Lightweight state required by online proxy decisions."""

    lineage_segment_prefix_tokens: dict[str, list[int]] = field(default_factory=dict)
    real_versions: set[str] = field(default_factory=set)
    ordered_response_versions: list[str] = field(default_factory=list)

    def update(self, step: StepRecord) -> None:
        prefix_tokens = self.lineage_segment_prefix_tokens.get(step.lineage_id)
        if prefix_tokens is None or step.lineage_segment_boundary_before:
            self.lineage_segment_prefix_tokens[step.lineage_id] = list(
                step.concat_token_ids
            )
        else:
            prefix_tokens.extend(step.concat_token_ids)

        for value in [
            *step.response_versions,
            step.response_version,
            step.request_version,
        ]:
            version = _real_token_version(value)
            if version is not None:
                self.real_versions.add(version)

        seen = set(self.ordered_response_versions)
        for value in step.response_versions:
            version = _real_token_version(value)
            if version is None or version in seen:
                continue
            self.ordered_response_versions.append(version)
            seen.add(version)


@dataclass
class _SessionState:
    session_created_at: float
    step_refs: list[str] = field(default_factory=list)
    step_refs_by_id: dict[str, str] = field(default_factory=dict)
    tito_state: TITOState = field(default_factory=TITOState)

    @property
    def step_count(self) -> int:
        return len(self.step_refs)


class TransferQueueRuntime:
    """Own all TransferQueue-specific state outside the proxy core classes."""

    def __init__(
        self,
        store: TransferQueueTrajectoryStore,
        *,
        transport_info: dict[str, Any],
        retention_seconds: float = 86400.0,
    ):
        self.store = store
        self.transport_info = transport_info
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}
        self._pending_step_cleanup: set[str] = set()

    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        store_id: str,
        transport_info: dict[str, Any],
        retention_seconds: float = 86400.0,
    ) -> TransferQueueRuntime:
        return cls(
            TransferQueueTrajectoryStore.from_config(
                config_path,
                store_id=store_id,
            ),
            transport_info=transport_info,
            retention_seconds=retention_seconds,
        )

    def _get_state(self, session: Session) -> _SessionState | None:
        with self._lock:
            state = self._sessions.get(session.session_id)
            if state is None or state.session_created_at != session.created_at:
                return None
            return state

    def has_steps(self, session: Session) -> bool:
        state = self._get_state(session)
        return state is not None and state.step_count > 0

    def step_count(self, session: Session) -> int:
        state = self._get_state(session)
        return 0 if state is None else state.step_count

    def real_versions(self, session: Session) -> set[str]:
        state = self._get_state(session)
        return set() if state is None else set(state.tito_state.real_versions)

    def ordered_response_versions(self, session: Session) -> list[str]:
        state = self._get_state(session)
        if state is None:
            return []
        return list(state.tito_state.ordered_response_versions)

    def current_segment_prefix_tokens(
        self,
        session: Session,
        lineage_id: str,
    ) -> list[int]:
        state = self._get_state(session)
        if state is None:
            return []
        return list(
            state.tito_state.lineage_segment_prefix_tokens.get(lineage_id, [])
        )

    @staticmethod
    def _build_step_record(values: dict[str, Any]) -> StepRecord:
        from dressage.proxy.session_manager import StepRecord

        return StepRecord(
            turn_id=values["turn_id"],
            request_messages=values["request_messages"],
            normalized_request_messages=values["normalized_request_messages"],
            prompt_token_ids=values["prompt_token_ids"],
            prompt_token_logprobs=values["prompt_token_logprobs"],
            snapshot_token_ids=values["snapshot_token_ids"],
            response_token_ids=values["response_token_ids"],
            response_logprobs=values["response_logprobs"],
            response_versions=list(values.get("response_versions") or []),
            all_token_ids=values["all_token_ids"],
            all_logprobs=values["all_logprobs"],
            all_versions=list(values.get("all_versions") or []),
            prompt_versions=list(values.get("prompt_versions") or []),
            input_token_texts=values["input_token_texts"],
            output_token_texts=values["output_token_texts"],
            messages_snapshot=values["messages"],
            raw_response_text=values["raw_response_text"],
            step_id=values["step_id"],
            lineage_id=values["lineage_id"],
            lineage_index=int(values.get("lineage_index", 0)),
            route_type=values.get("route_type", "create"),
            route_base_step_id=values.get("route_base_step_id"),
            normalized_messages_snapshot=list(
                values.get("normalized_messages_snapshot") or values["messages"]
            ),
            snapshot_rendered=values.get("snapshot_rendered", ""),
            snapshot_rendered_len=values.get("snapshot_rendered_len", 0),
            snapshot_tools_hash=values.get("snapshot_tools_hash"),
            lineage_segment_boundary_before=values.get(
                "lineage_segment_boundary_before",
                False,
            ),
            lineage_segment_reasons_before=list(
                values.get("lineage_segment_reasons_before") or []
            ),
            all_logprobs_invalid=values.get("all_logprobs_invalid", False),
            concat_token_ids=list(values.get("concat_token_ids") or []),
            concat_response_logprobs=list(values.get("concat_response_logprobs") or []),
            concat_response_mask=list(values.get("concat_response_mask") or []),
            concat_versions=list(values.get("concat_versions") or []),
            concat_context_token_count=values.get("concat_context_token_count", 0),
            concat_output_token_count=values.get("concat_output_token_count", 0),
            concat_logprobs_invalid=values.get("concat_logprobs_invalid", False),
            concat_incremental_tokenization_failed=values.get(
                "concat_incremental_tokenization_failed",
                False,
            ),
            response_routed_experts_chunks=[
                dict(item)
                for item in (values.get("response_routed_experts_chunks") or [])
            ],
            tools=values.get("tools"),
            segment_boundary_before=values.get("segment_boundary_before", False),
            rewrite_reason=values.get("rewrite_reason"),
            segment_reason_before=values.get("segment_reason_before"),
            segment_reasons_before=list(values.get("segment_reasons_before") or []),
            finish_reason=values.get("finish_reason", "stop"),
            request_version=values.get("request_version"),
            response_version=values.get("response_version"),
        )

    @staticmethod
    def build_proxy_projection(step: StepRecord) -> StepRecord:
        return replace(
            step,
            request_messages=[],
            normalized_request_messages=[],
            prompt_token_ids=[],
            prompt_token_logprobs=[],
            snapshot_token_ids=[],
            response_token_ids=[],
            response_logprobs=[],
            response_versions=[],
            all_token_ids=[],
            all_logprobs=[],
            all_versions=[],
            prompt_versions=[],
            input_token_texts=[],
            output_token_texts=[],
            messages_snapshot=[],
            raw_response_text="",
            concat_token_ids=[],
            concat_response_logprobs=[],
            concat_response_mask=[],
            concat_versions=[],
            concat_context_token_count=0,
            concat_output_token_count=0,
            concat_logprobs_invalid=False,
            concat_incremental_tokenization_failed=False,
            response_routed_experts_chunks=[],
        )

    async def persist_step(
        self,
        *,
        session: Session,
        **values: Any,
    ) -> tuple[str, StepRecord]:
        step = self._build_step_record(values)
        step_ref = await self.store.write_step(
            session_id=session.session_id,
            step_index=len(session.steps),
            step=step,
        )
        return step_ref, step

    def commit_step(
        self,
        *,
        session: Session,
        step_ref: str,
        step: StepRecord,
    ) -> None:
        with self._lock:
            state = self._sessions.setdefault(
                session.session_id,
                _SessionState(session_created_at=session.created_at),
            )
            prefix_tokens = {
                lineage_id: (
                    list(tokens)
                    if lineage_id == step.lineage_id
                    else tokens
                )
                for lineage_id, tokens in (
                    state.tito_state.lineage_segment_prefix_tokens.items()
                )
            }
            updated_tito_state = TITOState(
                lineage_segment_prefix_tokens=prefix_tokens,
                real_versions=set(state.tito_state.real_versions),
                ordered_response_versions=list(
                    state.tito_state.ordered_response_versions
                ),
            )
            updated_tito_state.update(step)
            state.step_refs.append(step_ref)
            state.step_refs_by_id[step.step_id] = step_ref
            state.tito_state = updated_tito_state
        session.last_active = time.time()

    async def abort_step(self, step_ref: str) -> None:
        await self._clear_step_refs([step_ref])

    async def read_step(self, session: Session, step_id: str) -> StepRecord:
        state = self._get_state(session)
        if state is None or step_id not in state.step_refs_by_id:
            raise KeyError(f"TransferQueue StepRecord {step_id!r} was not found")
        return (await self.store.read_steps([state.step_refs_by_id[step_id]]))[0]

    async def seal_session(
        self,
        *,
        session: Session,
        instance_id: str,
        finalization_id: str,
        label: Any,
        build_config: dict[str, Any],
    ) -> dict[str, Any]:
        if build_config != self.transport_info["build_config"]:
            raise RuntimeError(
                "proxy trajectory build configuration does not match coordinator"
            )
        state = self._get_state(session)
        if state is None or not state.step_refs:
            raise RuntimeError("session has no TransferQueue StepRecords")
        manifest = TrajectoryManifest(
            session_id=session.session_id,
            trajectory_id=session.session_id,
            instance_id=instance_id,
            finalization_id=finalization_id,
            step_refs=list(state.step_refs),
            num_steps=state.step_count,
            num_turns=len(session.turn_ids),
            num_lineage_segments=(
                len(split_session_into_lineage_segments(session))
                if build_config["token_build_mode"] == "tito"
                else 0
            ),
            num_timeline_segments=len(session.steps),
            history_rewritten=session.history_rewritten,
            label=label,
            build_config=build_config,
            config_fingerprint=self.transport_info["config_fingerprint"],
            sealed_at=time.time(),
            retention_seconds=self.retention_seconds,
        )
        await self.store.write_manifest(manifest)
        assembler_name = assembler_name_for(
            session.session_id,
            self.transport_info["assembler_names"],
        )
        return {
            "schema_version": "dressage.transport/v2",
            "backend": "transfer_queue",
            "store_id": self.store.store_id,
            "trajectory_id": session.session_id,
            "finalization_id": finalization_id,
            "ray_namespace": self.transport_info["ray_namespace"],
            "assembler_name": assembler_name,
        }

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    async def _clear_step_refs(self, step_refs: list[str]) -> None:
        if not step_refs:
            return
        try:
            await self.store.clear_steps(step_refs)
        except Exception:
            with self._lock:
                self._pending_step_cleanup.update(step_refs)
            logger.exception(
                "Failed to clear TransferQueue StepRecords; cleanup will be retried"
            )
        else:
            with self._lock:
                self._pending_step_cleanup.difference_update(step_refs)

    async def discard_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            await self._clear_step_refs(list(state.step_refs))
            await self.store.clear_manifest(session_id)
            return
        import ray

        assembler = ray.get_actor(
            assembler_name_for(
                session_id,
                self.transport_info["assembler_names"],
            ),
            namespace=self.transport_info["ray_namespace"],
        )
        await assembler.release.remote(trajectory_id=session_id)

    def handoff_session(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            self.store.forget_steps(state.step_refs)

    async def cleanup_expired(self, session_manager: SessionManager) -> None:
        with self._lock:
            session_states = list(self._sessions.items())
        expired_states = [
            (session_id, state)
            for session_id, state in session_states
            if (
                (session := session_manager.get_session(session_id)) is None
                or session.created_at != state.session_created_at
            )
        ]
        with self._lock:
            refs = list(self._pending_step_cleanup)
            expired_session_ids = []
            for session_id, state in expired_states:
                if self._sessions.get(session_id) is state:
                    self._sessions.pop(session_id)
                    expired_session_ids.append(session_id)
                    refs.extend(state.step_refs)
        await self._clear_step_refs(list(dict.fromkeys(refs)))
        for session_id in expired_session_ids:
            await self.store.clear_manifest(session_id)

    async def close(self, session_manager: SessionManager) -> None:
        await self.cleanup_expired(session_manager)
        with self._lock:
            session_ids = list(self._sessions)
            refs = [
                step_ref
                for state in self._sessions.values()
                for step_ref in state.step_refs
            ]
            self._sessions.clear()
        await self._clear_step_refs(refs)
        for session_id in session_ids:
            await self.store.clear_manifest(session_id)
