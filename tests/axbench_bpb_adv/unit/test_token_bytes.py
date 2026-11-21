"""Unit tests for ``_build_token_bytes`` (the BPB byte-weight table).

The byte-weight of every vocab token feeds the bits-per-byte normalisation. A
token whose decode fails must NOT be silently assigned a zero byte-weight (that
under-counts bytes and inflates BPB downstream with no error) -- it must fail
loudly, naming the offending token id.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.scoring import _build_token_bytes


class _FakeTokenizer:
    """Minimal tokenizer stub: ``decode([id])`` returns the mapped string, or
    raises if the mapped value is an exception instance."""

    def __init__(self, decoded: dict[int, object]) -> None:
        self._decoded = decoded

    def get_vocab_size(self) -> int:
        return len(self._decoded)

    def decode(self, ids: list[int]) -> str:
        (token_id,) = ids
        value = self._decoded[token_id]
        if isinstance(value, Exception):
            raise value
        return str(value)


def test_build_token_bytes_counts_utf8_byte_lengths() -> None:
    # "a" = 1 byte, "é" = 2 bytes, "€" = 3 bytes in UTF-8.
    weights = _build_token_bytes(_FakeTokenizer({0: "a", 1: "é", 2: "€"}))
    assert weights.dtype == t.int64
    assert weights.tolist() == [1, 2, 3]


def test_build_token_bytes_raises_loudly_on_undecodable_token() -> None:
    tok = _FakeTokenizer({0: "ok", 1: ValueError("boom"), 2: "fine"})
    with pytest.raises(AxBenchBpbAdvError, match="token id 1"):
        _build_token_bytes(tok)
