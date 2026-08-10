"""Routed-expert payload normalization."""

from __future__ import annotations

from typing import Any

try:
    import pybase64 as _base64
except ImportError:
    import base64 as _base64


def canonicalize_routed_experts(
    raw_chunks: list[dict[str, Any]],
    *,
    target_start: int,
    target_count: int,
) -> list[dict[str, Any]]:
    """Keep only the routed-expert rows required by the recorded step."""

    if target_start < 0 or target_count < 0:
        raise ValueError("routed-expert target range must be non-negative")
    if not raw_chunks or target_count == 0:
        return []

    target_end = target_start + target_count
    row_width: int | None = None
    emitted_rows = 0
    result: list[dict[str, Any]] = []

    for chunk in raw_chunks:
        prefix_count = int(chunk["prefix_token_count"])
        output_count = int(chunk["output_token_count"])
        raw_row_count = prefix_count + output_count - 1
        valid_start = 0 if chunk.get("is_first_chunk") else prefix_count - 1
        overlap_start = max(valid_start, target_start)
        overlap_end = min(raw_row_count, target_end)
        if overlap_start >= overlap_end:
            continue

        raw = _base64.b64decode(str(chunk["data"]).encode("ascii"))
        if raw_row_count <= 0 or len(raw) % raw_row_count != 0:
            raise ValueError(
                "routed-expert payload size does not match its expected row width"
            )
        current_row_width = len(raw) // raw_row_count
        if row_width is None:
            row_width = current_row_width
        elif current_row_width != row_width:
            raise ValueError("routed-expert chunks have inconsistent row widths")

        selected_rows = overlap_end - overlap_start
        result.append(
            {
                "data": _base64.b64encode(
                    raw[
                        overlap_start * current_row_width :
                        overlap_end * current_row_width
                    ]
                ).decode("ascii"),
                "row_count": selected_rows,
            }
        )
        emitted_rows += selected_rows

    if emitted_rows != target_count:
        raise ValueError(
            "routed-expert chunks do not cover the requested target range: "
            f"expected={target_count}, actual={emitted_rows}"
        )
    return result
