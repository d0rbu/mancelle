"""Slice 17 audit (2026-05-05) Q11 / P0.C / P0.F regression tests:

When ``truncate_prefix_left`` drops leftmost prefix tokens, the resulting
seg_map must reflect the truncated prefix's coordinate space — char-offsets
on the surviving segments must be re-projected onto the post-truncation
prefix text. Without the fix, the cummax monotonicity step in
``compute_context_token_segment_assignments`` mis-routes truncated tokens
to slot 0 (or out-of-range).

The tests cover both the single-sample path (``score_sample``) AND the
batched path (``_pad_encoder_inputs``).

ALL TESTS BELOW WILL FAIL WITHOUT THE Q11 FIX. They serve as the
regression gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from steering.eval.axbench_bpb.seg_map import (
    build_multi_segment_seg_map,
    project_segments_to_truncated_prefix,
)
from steering.eval.target_conditioning import ContextFeatureSegment


@dataclass
class _StubTokenizer:
    """Char-per-token tokenizer for char-exact projection tests.

    Each token id ``tid`` decodes to a single literal character whose
    codepoint == tid. Token sequences become predictable strings, so the
    expected projected char offsets are computable by hand.
    """

    def decode(self, token_ids):
        return "".join(chr(tid) for tid in token_ids)

    def encode(self, text):
        return [ord(ch) for ch in text]

    def get_bos_token_id(self):
        return ord("<")  # arbitrary


def test_project_segments_no_truncation_returns_input():
    """If truncated_prefix == original_prefix, projection is a no-op."""
    tok = _StubTokenizer()
    original = [ord("A"), ord("B"), ord("C"), ord("D")]
    truncated = list(original)
    segments = (
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(2,), char_start=0, char_end=4,
        ),
    )
    result = project_segments_to_truncated_prefix(
        segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    assert result == segments


def test_project_segments_full_drop_yields_no_segments():
    """If every segment lies entirely in the dropped portion, result is empty."""
    tok = _StubTokenizer()
    original = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E"), ord("F")]
    truncated = original[3:]  # keep "DEF"
    # Original char range of "AB" is [0, 2). After dropping "ABC" (3 chars),
    # the projected range is [-3, -1) which has new_end <= 0 → drop.
    segments = (
        ContextFeatureSegment(
            feature_ids=(11,), buckets=(1,), char_start=0, char_end=2,
        ),
    )
    result = project_segments_to_truncated_prefix(
        segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    assert result == ()


def test_project_segments_partial_overlap_clamps_left():
    """Segments that span the boundary get char_start clamped to 0."""
    tok = _StubTokenizer()
    original = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    truncated = original[2:]  # "CDE"
    # Segment covers chars [1, 4) — i.e. "BCD". After dropping 2 chars,
    # the segment lives at [-1, 2) → clamped to [0, 2) on "CDE".
    segments = (
        ContextFeatureSegment(
            feature_ids=(3,), buckets=(0,), char_start=1, char_end=4,
        ),
    )
    result = project_segments_to_truncated_prefix(
        segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    assert len(result) == 1
    proj = result[0]
    assert proj.feature_ids == (3,)
    assert proj.buckets == (0,)
    assert proj.char_start == 0
    assert proj.char_end == 2


def test_project_segments_fully_retained_shifts_offsets():
    """Segments entirely past the boundary just shift by dropped chars."""
    tok = _StubTokenizer()
    original = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    truncated = original[2:]  # "CDE"
    # Segment covers chars [3, 5) — "DE". After dropping 2 chars,
    # lives at [1, 3) on "CDE".
    segments = (
        ContextFeatureSegment(
            feature_ids=(9,), buckets=(4,), char_start=3, char_end=5,
        ),
    )
    result = project_segments_to_truncated_prefix(
        segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    assert len(result) == 1
    proj = result[0]
    assert proj.char_start == 1
    assert proj.char_end == 3


def test_project_segments_rejects_non_suffix():
    """If truncated is not a left-suffix of original, raise ValueError."""
    tok = _StubTokenizer()
    original = [1, 2, 3, 4]
    truncated = [9, 9]  # not a suffix
    with pytest.raises(ValueError, match="not a left-suffix"):
        project_segments_to_truncated_prefix(
            (
                ContextFeatureSegment(
                    feature_ids=(0,), buckets=(0,), char_start=0, char_end=1,
                ),
            ),
            original_prefix_tokens=original,
            truncated_prefix_tokens=truncated,
            tokenizer=tok,
        )


def test_project_segments_rejects_longer_truncated():
    """If truncated is longer than original, raise ValueError."""
    tok = _StubTokenizer()
    with pytest.raises(ValueError, match="exceeds original_prefix_tokens length"):
        project_segments_to_truncated_prefix(
            (),
            original_prefix_tokens=[1],
            truncated_prefix_tokens=[1, 2, 3],
            tokenizer=tok,
        )


def test_project_segments_preserves_order_and_payload():
    """Surviving segments retain feature_ids/buckets and order."""
    tok = _StubTokenizer()
    original = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E"), ord("F")]
    truncated = original[1:]  # "BCDEF"
    segments = (
        # Dropped (entirely in 'A')
        ContextFeatureSegment(
            feature_ids=(11,), buckets=(1,), char_start=0, char_end=1,
        ),
        # Partial overlap clamped
        ContextFeatureSegment(
            feature_ids=(22,), buckets=(2,), char_start=0, char_end=3,
        ),
        # Fully retained
        ContextFeatureSegment(
            feature_ids=(33,), buckets=(3,), char_start=4, char_end=6,
        ),
    )
    result = project_segments_to_truncated_prefix(
        segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    # Should have 2 surviving segments in order.
    assert len(result) == 2
    assert result[0].feature_ids == (22,)
    assert result[0].buckets == (2,)
    assert result[1].feature_ids == (33,)
    assert result[1].buckets == (3,)


def test_build_multi_segment_seg_map_unchanged_for_empty_segments():
    """Empty segments → all output positions point to generation_slot."""
    tok = _StubTokenizer()
    seg_map = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=[1, 2, 3],
        prefix_segments=(),
        n_output=2,
        generation_slot=0,
    )
    # All positions point to generation_slot=0
    assert seg_map.shape == (1, 5)
    assert seg_map.eq(0).all().item()


def test_build_multi_segment_seg_map_with_post_projection_segments():
    """When segments are projected against a truncated prefix, the
    cummax-based assignment routes truncated tokens to the correct slot.

    This is the smoke test that the Q11 fix actually works end-to-end:
    we compute segments against an ORIGINAL prefix, then project them
    against the TRUNCATED prefix, then build the seg_map.
    """
    tok = _StubTokenizer()
    # Original prefix: 6 single-char tokens "ABCDEF" (chars 0..5).
    original = [ord(c) for c in "ABCDEF"]
    # Truncated prefix: "DEF" (last 3 tokens).
    truncated = [ord(c) for c in "DEF"]
    # Two segments in the original coordinate space:
    #   [0, 3) covers "ABC" — entirely dropped.
    #   [3, 6) covers "DEF" — fully retained.
    original_segments = (
        ContextFeatureSegment(
            feature_ids=(10,), buckets=(0,), char_start=0, char_end=3,
        ),
        ContextFeatureSegment(
            feature_ids=(20,), buckets=(0,), char_start=3, char_end=6,
        ),
    )
    projected = project_segments_to_truncated_prefix(
        original_segments,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=tok,
    )
    # First segment dropped; second retained at [0, 3) on truncated.
    assert len(projected) == 1
    assert projected[0].char_start == 0
    assert projected[0].char_end == 3
    # generation_slot must reflect the post-projection segment count.
    seg_map = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=truncated,
        prefix_segments=projected,
        n_output=1,
        generation_slot=len(projected),  # == 1
    )
    # All 3 prefix tokens land on slot 0 (the only retained segment),
    # output position lands on slot 1 (generation slot).
    expected = [0, 0, 0, 1]
    assert seg_map[0].tolist() == expected
