"""Integration tests for the tokenizer-mismatch defence in
``TargetAwareEncoderDecoderEvalWrapper._align_row_context_segments``.

Mirrors the seg_map.py defence: a small mismatch between the supplied
``context_segments`` char_end and the consumer tokenizer's decoded
prefix length should be absorbed by ``align_segments_to_decoded_prefix``
(site-scoped counter ``segmentation.char_end_extended.encdec_eval_context``
with ``direction="forward"`` or ``direction="reverse"``); a large mismatch
should raise ``ValueError``.

We invoke the private ``_align_row_context_segments`` method directly with
a minimal stub-tokenizer + ContextFeatureSegment input, sidestepping
the full ``TargetAwareEncoderDecoderEvalWrapper`` construction (which
requires a real EncoderDecoderGPT model). Post the alignment lift,
alignment lives in ``_align_row_context_segments`` rather than inside
``_fill_context_assignments`` (see commit message for rationale).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.target_conditioning import ContextFeatureSegment


@dataclass
class _StubTokenizer:
    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def get_bos_token_id(self):
        return ord("<")


def _align_row_under_stub(tokenizer, context_tokens, context_segments):
    """Invoke TargetAwareEncoderDecoderEvalWrapper._align_row_context_segments
    by binding the bound-method descriptor to a SimpleNamespace stub.

    Why descriptor-protocol binding instead of mock.Mock(spec=...)?
    The full TargetAwareEncoderDecoderEvalWrapper construction requires a
    real EncoderDecoderGPT model and SafetensorsContextFeatureProvider.
    The method we're testing reads only ``self.tokenizer``, so a
    SimpleNamespace stub + descriptor binding lets us invoke the method
    without standing up the rest of the wrapper. Mock(spec=Class) would
    also work but would obscure that the test is exercising the actual
    production method, not a mock surface.
    """
    stub = SimpleNamespace(tokenizer=tokenizer)
    bound = TargetAwareEncoderDecoderEvalWrapper._align_row_context_segments.__get__(
        stub, TargetAwareEncoderDecoderEvalWrapper
    )
    return bound(context_tokens, context_segments)  # ty: ignore[invalid-argument-type]


def test_align_row_no_mismatch_returns_unchanged_no_counter(patch_counter):
    """char_end matches decoded len → no extension, no counter."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C")]
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
    )
    out = _align_row_under_stub(tok, tokens, segs)
    assert out == segs
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert canon == []


def test_align_row_forward_within_window_extends(patch_counter):
    """char_end < decoded by <= window → forward extension + counter."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5  # decoded = 5 chars
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
    )
    out = _align_row_under_stub(tok, tokens, segs)
    # The single segment's char_end is extended forward to match decoded length.
    assert out[-1].char_end == 5
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 2


def test_align_row_forward_over_window_raises():
    """char_end < decoded by > window → ValueError."""
    tok = _StubTokenizer()
    decoded = 1 + 20 + 1  # 22
    tokens = [ord("A")] * decoded
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=1),
    )
    with pytest.raises(ValueError, match="[Ff]orward.*window|exceeds"):
        _align_row_under_stub(tok, tokens, segs)


def test_align_row_reverse_within_window_clips(patch_counter):
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B")]  # decoded = 2
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=5),
    )
    out = _align_row_under_stub(tok, tokens, segs)
    # The (sole) segment is clipped reverse to match decoded length.
    assert out[-1].char_end == 2
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 3


def test_align_row_reverse_over_window_raises():
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B")]
    over = 2 + 20 + 1
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=over),
    )
    with pytest.raises(ValueError, match="[Rr]everse.*window|exceeds"):
        _align_row_under_stub(tok, tokens, segs)


def test_align_row_empty_tokens_returns_unchanged(patch_counter):
    """Empty context_tokens → method returns immediately, no counter."""
    tok = _StubTokenizer()
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
    )
    out = _align_row_under_stub(tok, [], segs)
    assert out == segs
    assert [c for c in patch_counter if "char_end_extended" in c[0]] == []


def test_align_row_reverse_filter_multi_segment_drops_tail(patch_counter):
    """Reverse-direction filtering drops tail segments past decoded_len.

    Verifies that the alignment helper's filter is tail-only (preserves
    a PREFIX of kept_indices), which is the load-bearing invariant for
    M3's seg_map / encoder slot routing: post-lift the encoder slots
    [0, M) are populated by _pack_encoder_inputs from POST-aligned
    segments, and the seg_map values from
    compute_context_token_segment_assignments are also in [0, M).
    Filter-is-tail ensures slot k in [0, M) semantically matches the
    encoder slot k built upstream.
    """
    tok = _StubTokenizer()
    decoded_len = 4
    tokens = [ord("A")] * decoded_len
    # 3 segments: [(0,3),(3,6),(6,9)]. decoded_len=4 -> kept=[0,1],
    # last clipped to 4. Index 2 dropped (char_start=6 > 4).
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    out = _align_row_under_stub(tok, tokens, segs)
    # Tail (seg index 2) dropped. Seg index 1 clipped to char_end=4.
    assert len(out) == 2
    assert out[1].char_end == 4
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 5  # overall_delta = 9 - 4
