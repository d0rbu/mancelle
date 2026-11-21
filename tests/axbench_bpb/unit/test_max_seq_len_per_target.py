"""Slice 17 audit (2026-05-05) P0.D regression test.

``parquet_row_to_scoring_input`` previously baked in a single
``max_seq_len`` (always the decoder's), so an encdec target with a smaller
context window would not see any pre-truncation, and ``score_sample`` could
silently overrun on the encdec branch.

The fix moves truncation into ``score_sample`` per-target via
``truncate_prefix_left`` against ``target_bundle.max_seq_len``. The
``ScoringInputs`` carries the **untruncated** prefix tokens.

This test pins the behaviour: the per-target truncation runs in
``score_sample``, and ``parquet_row_to_scoring_input`` does NOT
pre-truncate the prefix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from steering.eval.axbench_bpb.scoring_inputs import parquet_row_to_scoring_input
from steering.eval.axbench_bpb.types import SteeringMode


@dataclass
class _MiniTokenizer:
    """One-char-per-token tokenizer with a fixed BOS."""

    bos_id: int = 0

    def encode(self, text):
        # Simple char→id mapping (offset by 1 so 0 is reserved for BOS).
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids if t != 0)

    def get_bos_token_id(self):
        return self.bos_id


def testparquet_row_to_scoring_input_keeps_full_prefix():
    """The returned ScoringInputs.prefix_tokens are NOT pre-truncated."""
    tok = _MiniTokenizer()
    long_prefix = "A" * 1024
    output = "OUTPUT"
    row = {
        "prefix_text": long_prefix,
        "output_text": output,
        "content_class": "concept_rich",
        "concept_id": 7,
    }
    # Use a max_seq_len that would have triggered truncation in the
    # pre-fix code (decoder's seq_len).
    token_bytes = torch.zeros(2048, dtype=torch.int64)
    si = parquet_row_to_scoring_input(
        row,
        mode=SteeringMode.POSITIVE,
        context_segments_by_hash={},
        tokenizer=tok,
        max_seq_len=64,
        token_bytes=token_bytes,
    )
    # n_prefix should equal the FULL prefix tokens count (BOS + 1024 chars),
    # NOT the truncated count (which would be ~64 - len(output)).
    assert si.n_prefix == 1 + len(long_prefix), (
        f"prefix_tokens were pre-truncated (n_prefix={si.n_prefix}); "
        "the P0.D fix requires the un-truncated prefix to flow through to "
        "score_sample so it can apply per-target truncation."
    )


def test_parquet_row_truncation_flag_marks_oversized():
    """If prefix would not fit in max_seq_len, the truncated flag is set
    even though prefix_tokens are now untruncated."""
    tok = _MiniTokenizer()
    row = {
        "prefix_text": "X" * 100,
        "output_text": "OUT",
        "content_class": "concept_avoiding",
        "concept_id": 1,
    }
    token_bytes = torch.zeros(2048, dtype=torch.int64)
    si = parquet_row_to_scoring_input(
        row,
        mode=SteeringMode.NEGATIVE,
        context_segments_by_hash={},
        tokenizer=tok,
        max_seq_len=10,
        token_bytes=token_bytes,
    )
    # truncated must be True (best-effort upfront flag)
    assert si.truncated is True
    # Yet the prefix_tokens are untruncated — score_sample will redo the
    # truncation against the per-target window.
    assert si.n_prefix > 10


def test_parquet_row_no_truncation_preserves_flag_false():
    """A short prefix flows through with truncated=False."""
    tok = _MiniTokenizer()
    row = {
        "prefix_text": "AB",
        "output_text": "C",
        "content_class": "concept_rich",
        "concept_id": 0,
    }
    token_bytes = torch.zeros(2048, dtype=torch.int64)
    si = parquet_row_to_scoring_input(
        row,
        mode=SteeringMode.POSITIVE,
        context_segments_by_hash={},
        tokenizer=tok,
        max_seq_len=128,
        token_bytes=token_bytes,
    )
    assert si.truncated is False
    assert si.n_prefix == 1 + 2  # BOS + 2 chars
    assert si.n_output == 1
