from __future__ import annotations

import base64
from array import array

import pytest

from dressage.proxy.routed_experts import canonicalize_routed_experts


def _encode(values: list[int]) -> str:
    encoded = array("i", values)
    assert encoded.itemsize == 4
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode(chunks: list[dict]) -> list[int]:
    values: list[int] = []
    for chunk in chunks:
        decoded = array("i")
        decoded.frombytes(base64.b64decode(chunk["data"]))
        values.extend(decoded)
    return values


def _raw_partial_chunks() -> list[dict]:
    return [
        {
            "data": _encode([10, 11, 12, 13]),
            "prefix_token_count": 3,
            "output_token_count": 2,
            "is_first_chunk": True,
        },
        {
            "data": _encode([20, 21, 22, 23, 24, 25]),
            "prefix_token_count": 5,
            "output_token_count": 2,
            "is_first_chunk": False,
        },
    ]


def test_canonicalize_routed_experts_removes_partial_prefix() -> None:
    chunks = canonicalize_routed_experts(
        _raw_partial_chunks(),
        target_start=0,
        target_count=6,
    )

    assert [chunk["row_count"] for chunk in chunks] == [4, 2]
    assert _decode(chunks) == [10, 11, 12, 13, 24, 25]


def test_canonicalize_routed_experts_intersects_step_delta() -> None:
    chunks = canonicalize_routed_experts(
        _raw_partial_chunks(),
        target_start=3,
        target_count=3,
    )

    assert [chunk["row_count"] for chunk in chunks] == [1, 2]
    assert _decode(chunks) == [13, 24, 25]


def test_canonicalize_routed_experts_preserves_complete_rows() -> None:
    chunks = canonicalize_routed_experts(
        [
            {
                "data": _encode([10, 110, 11, 111, 12, 112, 13, 113]),
                "prefix_token_count": 3,
                "output_token_count": 2,
                "is_first_chunk": True,
            },
            {
                "data": _encode(
                    [20, 120, 21, 121, 22, 122, 23, 123, 24, 124, 25, 125]
                ),
                "prefix_token_count": 5,
                "output_token_count": 2,
                "is_first_chunk": False,
            },
        ],
        target_start=3,
        target_count=3,
    )

    assert [chunk["row_count"] for chunk in chunks] == [1, 2]
    assert _decode(chunks) == [13, 113, 24, 124, 25, 125]


def test_canonicalize_routed_experts_rejects_invalid_row_width() -> None:
    chunks = [
        {
            "data": base64.b64encode(b"abc").decode("ascii"),
            "prefix_token_count": 2,
            "output_token_count": 1,
            "is_first_chunk": True,
        }
    ]

    with pytest.raises(ValueError, match="row width"):
        canonicalize_routed_experts(chunks, target_start=0, target_count=2)
