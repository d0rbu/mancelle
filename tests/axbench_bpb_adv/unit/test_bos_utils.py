"""Tests for AxBench-BPB-Adv BOS-padded adversarial token guards."""

from __future__ import annotations

import pytest

from steering.eval.axbench_bpb_adv.bos_utils import (
    assert_bos_padded_tokens,
    assert_no_bos_payload,
    bos_padded_tokens,
    strip_bos_padded_tokens,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def test_bos_padded_tokens_builds_valid_view() -> None:
    assert bos_padded_tokens([10, 11], bos_id=7, context="ctx") == [7, 10, 11]


def test_bos_padded_tokens_rejects_payload_starting_with_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        bos_padded_tokens([7, 11], bos_id=7, context="ctx")


def test_bos_padded_tokens_rejects_non_adjacent_payload_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        bos_padded_tokens([10, 7], bos_id=7, context="ctx")


def test_strip_bos_padded_tokens_returns_payload() -> None:
    assert strip_bos_padded_tokens([7, 10, 11], bos_id=7, context="ctx") == [
        10,
        11,
    ]


def test_strip_bos_padded_tokens_rejects_missing_bos_prefix() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="does not start with BOS"):
        strip_bos_padded_tokens([10, 11], bos_id=7, context="ctx")


def test_strip_bos_padded_tokens_rejects_double_bos_prefix() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        strip_bos_padded_tokens([7, 7, 11], bos_id=7, context="ctx")


def test_strip_bos_padded_tokens_rejects_non_adjacent_payload_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        strip_bos_padded_tokens([7, 10, 7], bos_id=7, context="ctx")


def test_assert_bos_padded_tokens_returns_normalized_list() -> None:
    assert assert_bos_padded_tokens((7, 10, 11), bos_id=7, context="ctx") == [
        7,
        10,
        11,
    ]


def test_assert_no_bos_payload_rejects_all_payload_bos_positions() -> None:
    with pytest.raises(AxBenchBpbAdvError, match=r"position\(s\) \[1, 3\]"):
        assert_no_bos_payload([10, 7, 11, 7], bos_id=7, context="ctx")
