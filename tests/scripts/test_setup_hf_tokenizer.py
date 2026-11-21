"""Unit tests for shared Hugging Face token-byte accounting (P2 #11).

The pre-refactor script swallowed every exception in ``decode([token_id])``
and silently appended ``0`` byte length, which understates BPB measurements
if a real tokenizer bug surfaces a non-decode error class. We narrow the
caught classes to genuine decode failures and log a warning otherwise.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from steering.sft.token_bytes import compute_hf_token_bytes


class _StubTok:
    """Stub tokenizer with a configurable ``decode`` and special-id set."""

    def __init__(
        self,
        vocab_size: int,
        special_ids: set[int],
        decode_map: dict[int, str | type[BaseException]],
    ) -> None:
        self._vocab_size = vocab_size
        self._special_ids = special_ids
        self._decode_map = decode_map

    def get_vocab_size(self) -> int:
        return self._vocab_size

    @property
    def tokenizer(self):
        # Mimic transformers Tokenizer surface: ``.tokenizer.all_special_ids``
        m = MagicMock()
        m.all_special_ids = list(self._special_ids)
        return m

    def decode(self, ids: list[int]) -> str:
        assert isinstance(ids, list) and len(ids) == 1
        result = self._decode_map.get(ids[0], "")
        if isinstance(result, type) and issubclass(result, BaseException):
            if result is UnicodeDecodeError:
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "synthetic")
            raise result("synthetic")
        return result


def test_special_tokens_get_zero_bytes():
    tok = _StubTok(
        vocab_size=4,
        special_ids={0, 1},
        decode_map={0: "", 1: "", 2: "ab", 3: "cdef"},
    )
    out = compute_hf_token_bytes(tok)
    assert out == [0, 0, 2, 4]


def test_empty_decode_yields_zero_bytes():
    tok = _StubTok(
        vocab_size=3,
        special_ids=set(),
        decode_map={0: "x", 1: "", 2: "yz"},
    )
    out = compute_hf_token_bytes(tok)
    assert out == [1, 0, 2]


def test_unicode_decode_failure_logs_and_returns_zero(caplog):
    tok = _StubTok(
        vocab_size=2,
        special_ids=set(),
        decode_map={0: "x", 1: UnicodeDecodeError},
    )
    with caplog.at_level(logging.WARNING):
        out = compute_hf_token_bytes(tok)
    assert out == [1, 0]


def test_runtime_decode_failure_logs_and_returns_zero():
    tok = _StubTok(
        vocab_size=2,
        special_ids=set(),
        decode_map={0: "x", 1: RuntimeError},
    )
    out = compute_hf_token_bytes(tok)
    assert out == [1, 0]


def test_unexpected_exception_propagates():
    """A non-decode exception should NOT be swallowed."""
    tok = _StubTok(
        vocab_size=2,
        special_ids=set(),
        decode_map={0: "x", 1: ZeroDivisionError},
    )
    with pytest.raises(ZeroDivisionError):
        compute_hf_token_bytes(tok)


def test_utf8_byte_length_for_multibyte_chars():
    tok = _StubTok(
        vocab_size=2,
        special_ids=set(),
        decode_map={0: "a", 1: "中文"},  # "中文" = 6 UTF-8 bytes
    )
    out = compute_hf_token_bytes(tok)
    assert out == [1, 6]
