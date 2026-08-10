"""dressage_claw rollout dispatch: standard blackbox flow plus grader input staging."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from dressage.config import proxy_url
from dressage.paddock.blackbox.execute_hooks import (
    execute_blackbox_cmds_for_stage,
    parse_blackbox_execute_cmds,
)
from dressage.paddock.blackbox.failures import (
    HARVESTABLE_AGENT_ERRORS,
    agent_response_text,
    expected_abort_from_call_agent_exception,
    failure_from_call_agent_exception,
    failure_from_payload_state,
    record_blackbox_abort_for_retry,
    record_agent_early_stop_metadata,
    record_agent_failure_metadata,
)
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    merge_backend_options,
    normalize_blackbox_type,
)
from dressage.paddock.lifecycle import (
    exception_summary as _exception_summary,
    schedule_terminate_paddock,
)
from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _instance_id,
    set_status as _set_status,
)
from dressage.rollout.artifacts.writer import DEFAULT_WRITER as _ARTIFACT_WRITER
from dressage.rollout.generate.runtime import (
    discard_proxy_session_best_effort,
    get_paddock_from_env,
    get_proxy_client,
    maybe_await,
    paddock_env_args_from_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# agent_messages.json helpers (grader input)
# ---------------------------------------------------------------------------


def _build_fallback_messages_b64(prompt: Any, response: str) -> str:
    """Build a prompt-only agent_messages payload as fallback."""
    import base64

    payload = json.dumps(
        {
            "prompt": str(prompt),
            "response": response,
            "messages": [{"role": "user", "content": str(prompt)}],
        },
        ensure_ascii=False,
    )
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


def _build_agent_messages_json(
    trajectory_payload: dict[str, Any],
    *,
    prompt: Any,
    response: str,
) -> tuple[str, str]:
    """Build agent_messages.json content from Proxy trajectory data.

    Returns (json_string, updated_response).
    Raises ValueError if no trajectory segments or messages.
    """
    segments = trajectory_payload.get("data") or []
    if not segments:
        raise ValueError("trajectory has no segments")

    last_segment = segments[-1]
    messages = last_segment.get("messages") or []
    if not messages:
        raise ValueError("last segment has no messages")

    updated_response = response
    if not updated_response:
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if msg.get("tool_calls"):
                continue
            content = msg.get("content")
            if content:
                updated_response = str(content)
                break

    json_str = json.dumps(
        {"prompt": str(prompt), "response": updated_response, "messages": messages},
        ensure_ascii=False,
    )
    return json_str, updated_response


async def _write_agent_messages_to_sandbox(
    paddock: Any,
    state: Any,
    *,
    session_id: str,
    json_content: str,
) -> None:
    """Write agent_messages.json to sandbox, chunked to avoid shell arg limits."""
    import base64
    import shlex

    b64_str = base64.b64encode(json_content.encode("utf-8")).decode("ascii")
    chunk_size = 50000
    chunks = [b64_str[i:i + chunk_size] for i in range(0, len(b64_str), chunk_size)]

    await maybe_await(paddock.execute_cmd(
        state, session_id=session_id,
        cmd=f"mkdir -p /tmp/omni_grader && printf %s {shlex.quote(chunks[0])} > /tmp/omni_grader/agent_messages.b64",
        timeout=10.0,
    ))
    for chunk in chunks[1:]:
        await maybe_await(paddock.execute_cmd(
            state, session_id=session_id,
            cmd=f"printf %s {shlex.quote(chunk)} >> /tmp/omni_grader/agent_messages.b64",
            timeout=10.0,
        ))
    await maybe_await(paddock.execute_cmd(
        state, session_id=session_id,
        cmd="base64 -d /tmp/omni_grader/agent_messages.b64 > /tmp/omni_grader/agent_messages.json",
        timeout=10.0,
    ))


async def _fetch_agent_messages_b64(
    paddock: Any,
    state: Any,
    *,
    session_id: str,
    prompt: Any,
    response: str,
) -> str:
    """Fetch conversation_history from BBS and return base64-encoded payload."""
    import base64
    import shlex

    bbs_port = os.environ.get("BBS_PORT", "")
    if bbs_port:
        url = f"http://127.0.0.1:{bbs_port}/v1/sessions/{session_id}?include_history=true"
        cmd = f"curl -sf {shlex.quote(url)}"
    else:
        url = f"http://127.0.0.1:${{BBS_PORT:-31000}}/v1/sessions/{session_id}?include_history=true"
        cmd = f'curl -sf "{url}"'
    execute_cmd = getattr(paddock, "execute_cmd", None)
    if execute_cmd is None:
        raise RuntimeError(
            f"paddock does not support execute_cmd; cannot fetch "
            f"conversation_history for session_id={session_id}"
        )
    result = await maybe_await(
        execute_cmd(state, session_id=session_id, cmd=cmd, timeout=15.0)
    )
    stdout = result.get("stdout", "") if isinstance(result, dict) else getattr(result, "stdout", "")
    if not stdout or not stdout.strip():
        raise RuntimeError(
            f"BBS session {session_id} returned empty response; "
            "cannot build grader messages"
        )
    session_data = json.loads(stdout)
    history = session_data.get("conversation_history") if isinstance(session_data, dict) else None
    if not isinstance(history, list) or not history:
        raise RuntimeError(
            f"BBS session {session_id} returned empty conversation_history; "
            "cannot build grader messages"
        )
    payload = json.dumps(
        {"prompt": str(prompt), "response": response, "messages": history},
        ensure_ascii=False,
    )
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


async def _write_stage_files(paddock: Any, state: Any, metadata: dict[str, Any], stage: str) -> None:
    """Write {stage}_files into the sandbox via paddock.write_files()."""
    files = metadata.get(f"{stage}_files")
    if not files:
        return
    await maybe_await(paddock.write_files(state, files=files))


# ---------------------------------------------------------------------------
# Standard blackbox_dispatch helpers (kept identical)
# ---------------------------------------------------------------------------


def _chat_messages_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": str(prompt)}]


def _ensure_blackbox_session_id(sample: Any) -> str:
    session_id = getattr(sample, "session_id", None)
    if session_id is None:
        session_id = str(uuid.uuid4())
    session_id = str(session_id)

    if not session_id.startswith("bbs-"):
        session_id = f"bbs-{session_id}"
        sample.session_id = session_id

    return session_id


def _backend_options_for_register(
    *,
    args: Any,
    metadata: dict[str, Any],
    blackbox_type: str,
) -> Any:
    backend_options = metadata.get("backend_options")
    return merge_backend_options(blackbox_type, backend_options, args=args)


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Run one blackbox sandbox rollout and write proxy data back to Sample."""
    del sampling_params
    if evaluation:
        raise ValueError("blackbox_dispatch does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata.pop("blackbox_error", None)
    metadata.pop("blackbox_error_log_path", None)
    metadata.pop("blackbox_expected_abort", None)
    metadata.pop("blackbox_agent_early_stop", None)
    metadata.pop("blackbox_agent_early_stop_kind", None)
    metadata["execute_cmds"] = []
    session_id = _ensure_blackbox_session_id(sample)
    instance_id = _instance_id(sample)
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id
    blackbox_type = normalize_blackbox_type(
        metadata.get("blackbox_type") or DEFAULT_BLACKBOX_TYPE
    )

    extra_env_args = None
    if "blackbox_type" in metadata or blackbox_type != DEFAULT_BLACKBOX_TYPE:
        extra_env_args = {"blackbox_type": blackbox_type}
    env_args = paddock_env_args_from_metadata(
        metadata,
        extra_env_args=extra_env_args,
    )
    paddock = None
    proxy_client = None
    state = None
    initialized = False
    agent_response = ""
    try:
        execute_cmd_schedule = parse_blackbox_execute_cmds(
            metadata.get("blackbox_execute_cmds")
        )
        backend_options = _backend_options_for_register(
            args=args,
            metadata=metadata,
            blackbox_type=blackbox_type,
        )
        paddock = get_paddock_from_env(allow_whitebox_mode=False)
        proxy_client = get_proxy_client()
        state = await maybe_await(
            paddock.init(
                session_id,
                metadata.get("env_type"),
                env_args,
            )
        )
        initialized = True
        # system_prompt_file must exist at register time.
        await _write_stage_files(paddock, state, metadata, "before_agent")
        if not hasattr(paddock, "register_agent"):
            raise TypeError(f"{type(paddock).__name__} does not implement register_agent")
        await maybe_await(
            paddock.register_agent(
                state,
                instance_id=instance_id,
                session_id=session_id,
                router_url=proxy_url(),
                blackbox_type=blackbox_type,
                backend_options=backend_options,
                system_prompt_file=metadata.get("system_prompt_file") or None,
            )
        )
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="before_agent",
        )
        call_payload: Any = None
        call_succeeded = False
        try:
            call_payload = await maybe_await(
                paddock.call_agent(
                    state,
                    session_id=session_id,
                    messages=_chat_messages_from_prompt(sample.prompt),
                    metadata={"source": "dressage", **metadata},
                )
            )
            call_succeeded = True
        except Exception as exc:
            if agent_failure := failure_from_call_agent_exception(exc):
                record_agent_failure_metadata(metadata, agent_failure)
                if agent_failure.kind in HARVESTABLE_AGENT_ERRORS:
                    record_agent_early_stop_metadata(metadata, agent_failure)
                    logger.info(
                        "harvesting blackbox rollout after agent early stop: "
                        "session_id=%s kind=%s",
                        session_id,
                        agent_failure.kind,
                    )
                else:
                    raise agent_failure from exc
            else:
                raise

        if call_succeeded:
            agent_response = agent_response_text(call_payload)
            if agent_failure := failure_from_payload_state(
                call_payload,
                agent_response=agent_response,
            ):
                record_agent_failure_metadata(metadata, agent_failure)
                raise agent_failure

        # Finalize early: the grader needs the full conversation below.
        try:
            await proxy_client.finalize_session(
                session_id, instance_id=instance_id, label=getattr(sample, "label", None)
            )
            trajectory_payload = await proxy_client.read_trajectory(
                trajectory_id=session_id,
                instance_id=instance_id,
                drain=True,
            )
        except Exception:
            logger.warning(
                "finalize/read_trajectory failed for session_id=%s; "
                "will try BBS fallback for grader",
                session_id,
                exc_info=True,
            )
            trajectory_payload = {"data": []}

        # Prefer proxy trajectory, then BBS history, then prompt-only.
        agent_messages_json: str
        try:
            agent_messages_json, agent_response = _build_agent_messages_json(
                trajectory_payload,
                prompt=sample.prompt,
                response=agent_response,
            )
        except Exception:
            try:
                import base64 as _b64
                _b64_payload = await _fetch_agent_messages_b64(
                    paddock, state, session_id=session_id,
                    prompt=sample.prompt, response=agent_response,
                )
                agent_messages_json = _b64.b64decode(_b64_payload).decode("utf-8")
            except Exception:
                logger.warning(
                    "failed to build agent_messages from both proxy and BBS "
                    "for session_id=%s; using fallback prompt-only payload",
                    session_id,
                    exc_info=True,
                )
                import base64 as _b64
                _b64_payload = _build_fallback_messages_b64(
                    sample.prompt, agent_response,
                )
                agent_messages_json = _b64.b64decode(_b64_payload).decode("utf-8")

        await _write_agent_messages_to_sandbox(
            paddock, state, session_id=session_id, json_content=agent_messages_json,
        )

        # Grader files injected after agent finishes (agent must not see them).
        await _write_stage_files(paddock, state, metadata, "after_agent")

        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="after_agent",
        )

        try:
            await _ARTIFACT_WRITER.write_session_payload(
                trajectory_payload,
                session_id=session_id,
                instance_id=instance_id,
            )
        except Exception:
            logger.warning(
                "failed to write trajectory payload log for session_id=%s",
                session_id,
                exc_info=True,
            )
        segments = trajectory_payload.get("data") or []
        base_metadata_for_logs = dict(metadata)
        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=args,
            agent_response=agent_response,
            session_id=session_id,
            instance_id=instance_id,
        )
        log_template = sample
        try:
            await _ARTIFACT_WRITER.write_segment_samples(
                log_template,
                args=args,
                segments=segments,
                base_metadata=base_metadata_for_logs,
                session_id=session_id,
                instance_id=instance_id,
                agent_response=agent_response,
            )
        except Exception:
            logger.warning(
                "failed to write sample logs for session_id=%s",
                session_id,
                exc_info=True,
            )
        return result
    except Exception as exc:
        expected_abort = expected_abort_from_call_agent_exception(exc)
        if expected_abort is None:
            logger.warning(
                "blackbox rollout failed for session_id=%s: %s",
                session_id,
                _exception_summary(exc),
            )
            try:
                error_log_path = await _ARTIFACT_WRITER.write_error(
                    exc,
                    sample=sample,
                    metadata=dict(metadata),
                    session_id=session_id,
                    instance_id=instance_id,
                    blackbox_type=blackbox_type,
                    env_args=dict(env_args),
                    state=state,
                    agent_response=agent_response,
                )
                if error_log_path is not None:
                    metadata["blackbox_error_log_path"] = str(error_log_path)
            except Exception:
                logger.warning(
                    "failed to write trajectory error log for session_id=%s",
                    session_id,
                    exc_info=True,
                )
            record_blackbox_abort_for_retry(metadata, session_id, exc)
        else:
            metadata["blackbox_expected_abort"] = expected_abort
        multi_segment.mark_aborted_no_grad(
            sample, session_id=session_id, instance_id=instance_id
        )
        _set_status(sample, "ABORTED")
        await discard_proxy_session_best_effort(
            session_id,
            proxy_client=proxy_client,
        )
        return sample
    finally:
        if initialized and paddock is not None:
            schedule_terminate_paddock(
                paddock,
                session_id=session_id,
                env_args=env_args,
            )
