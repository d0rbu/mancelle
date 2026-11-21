"""Shared helpers for cross-tokenizer truncation and segment validation tests.

Provides:
- SimpleTokenizer: per-character tokenizer (ID = ord(c), NO BOS — see contract
  notes on ``SimpleTokenizer`` below).
- _make_seg_features: build seg_features tuples from char ranges

Slice 02 owns this file (Phase 0 map §1, design spec Appendix A). Tests
that compare cross-tokenizer behavior must pin the contract differences
between this helper and production ``RustBPETokenizer``: the helper is
deliberately BOS-free so char positions equal ``ord(c)`` directly.
"""

from __future__ import annotations

import torch as t


class SimpleTokenizer:
    """Tokenizer where each ASCII character is a separate token (ID = ord(c)).

    Empty-string contract (slice-02 audit + X2 §3 broken-fixture #6):
        ``encode("") -> []`` and ``decode([]) -> ""``. The helper does
        NOT prepend a BOS token. This intentionally diverges from
        production ``RustBPETokenizer.encode("")`` (which may emit a BOS
        or raise) so that tests asserting char-position == ``ord(c)``
        keep that one-to-one mapping. Cross-tokenizer tests that need to
        emulate a BOS-prepending tokenizer must do so explicitly (e.g.
        prepend ``[0]`` in the test) and assert against the resulting
        offset shift.

    Per-character mapping:
        ``encode(c) -> [ord(c)]``. Multibyte characters become a single
        token whose id equals the codepoint (NOT a byte-fallback run).
        Tests that need byte-fallback semantics should use a different
        helper (e.g. the ``_ByteTokenizer`` in
        ``tests/steering/test_segmentation_property.py``).

    BOS id:
        ``get_bos_token_id() -> 0``. Fixed; downstream consumers
        (``axbench_bpb`` seg_map tests) rely on it.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids)

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        # IDs are Unicode codepoints via ord(); cover the BMP (0–65535).
        return 65536

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts)}
        return {"input_ids": [self.encode(text) for text in texts]}


def _make_seg_features(
    char_ranges: list[tuple[int, int]],
    n_features: int = 2,
) -> tuple[list[t.Tensor], list[t.Tensor], list[int], list[int]]:
    """Build seg_features tuple from character ranges."""
    fids = [
        t.arange(n_features, dtype=t.int64) + (i + 1) * 10
        for i in range(len(char_ranges))
    ]
    mags = [t.ones(n_features, dtype=t.int64) for _ in range(len(char_ranges))]
    char_starts = [s for s, _ in char_ranges]
    char_ends = [e for _, e in char_ranges]
    return fids, mags, char_starts, char_ends
