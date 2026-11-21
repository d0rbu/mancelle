"""Hypothesis property tests for ``project_segments_to_truncated_prefix``.

The slice-17 audit (2026-05-05) Q11 fix re-projects ``ContextFeatureSegment``
char offsets onto a left-truncated prefix. These tests assert invariants the
projection MUST satisfy for any valid (original_prefix, truncated_prefix,
segments) tuple, not just the hand-picked unit cases in
``test_seg_map_truncation.py``.

Invariants tested:
  P1. NO-OP: when truncated == original, projection returns input unchanged.
  P2. SUFFIX GUARD: when truncated is a left-suffix, projection succeeds.
  P3. NON-SUFFIX REJECTION: when truncated is NOT a left-suffix, raises ValueError.
  P4. CHAR-OFFSET BOUNDS: every surviving segment lives in [0, len_truncated_chars].
  P5. SHIFT INVARIANT: surviving segment widths shrink by at most ``dropped_chars``;
      fully-retained segments preserve their original width exactly.
  P6. ORDER PRESERVATION: surviving segments maintain their input order.
  P7. PAYLOAD PRESERVATION: surviving segments preserve ``feature_ids`` and ``buckets``.
  P8. DROP COMPLETENESS: any segment with ``char_end <= dropped_chars`` is dropped.
  P9. RETENTION COMPLETENESS: any segment with ``char_start >= dropped_chars``
      survives (assuming non-degenerate width).
  P10. IDEMPOTENCY: applying the projection twice with the same arguments is a no-op
      (since the second call has truncated == "original" from its perspective).
  P11. EMPTY SEGMENTS: any (orig, trunc) pair with empty segments returns empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.seg_map import project_segments_to_truncated_prefix
from steering.eval.target_conditioning import ContextFeatureSegment


@dataclass
class _CharTok:
    """One-codepoint-per-token tokenizer for predictable char offsets."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def get_bos_token_id(self) -> int:
        return ord("<")


_TOK = _CharTok()


# --- strategies ---


def _segments_strategy(max_char: int) -> st.SearchStrategy:
    """Generate non-empty, possibly-overlapping segment tuples within [0, max_char]."""
    if max_char < 1:
        return st.just(())
    return st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=max_char - 1),  # char_start
            st.integers(min_value=1, max_value=max_char),       # char_end
        )
        .filter(lambda se: se[0] < se[1])
        .flatmap(
            lambda se: st.tuples(
                st.just(se[0]),
                st.just(se[1]),
                st.integers(min_value=0, max_value=10000),  # feature_id
                st.integers(min_value=0, max_value=255),    # bucket
            )
        ),
        min_size=0,
        max_size=8,
    ).map(
        lambda items: tuple(
            ContextFeatureSegment(
                feature_ids=(fid,),
                buckets=(bkt,),
                char_start=cs,
                char_end=ce,
            )
            for (cs, ce, fid, bkt) in items
        )
    )


_PRINTABLE_CHAR = st.integers(min_value=ord("A"), max_value=ord("Z"))


@st.composite
def _suffix_pair(draw):
    """Generate (original_prefix_tokens, truncated_prefix_tokens) where truncated is a left-suffix."""
    original = draw(st.lists(_PRINTABLE_CHAR, min_size=1, max_size=20))
    drop = draw(st.integers(min_value=0, max_value=len(original)))
    truncated = list(original[drop:])
    return original, truncated


# --- properties ---


@settings(max_examples=200, derandomize=True, suppress_health_check=[HealthCheck.too_slow])
@given(
    pair=_suffix_pair(),
    segments=st.data(),
)
def test_p1_p2_p4_p5_p6_p7_p8_p9_p11_combined(pair: tuple[list[int], list[int]], segments) -> None:
    """One omnibus property check covering the safety invariants of the projection."""
    original, truncated = pair
    n_original = len(original)
    n_truncated = len(truncated)
    n_dropped = n_original - n_truncated
    # original_prefix_tokens here are 1 char each (we use _CharTok).
    # So len_original_chars == n_original; dropped_chars == n_dropped.
    len_original_chars = n_original
    len_truncated_chars = n_truncated
    dropped_chars = n_dropped

    # Generate segments AFTER we know max_char.
    segs: tuple[ContextFeatureSegment, ...] = segments.draw(
        _segments_strategy(max_char=len_original_chars)
    )

    result = project_segments_to_truncated_prefix(
        segs,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=_TOK,
    )

    # P11 — empty segments → empty result.
    if not segs:
        assert result == ()
        return

    # P1 — no-op when no truncation.
    if dropped_chars == 0:
        assert result == segs
        return

    # P4 — char-offset bounds.
    for proj in result:
        assert 0 <= proj.char_start <= len_truncated_chars
        assert 0 <= proj.char_end <= len_truncated_chars
        assert proj.char_start < proj.char_end, "non-empty range required"

    # Match each surviving projected segment back to its input.
    surviving_inputs: list[ContextFeatureSegment] = []
    for orig in segs:
        if orig.char_end <= dropped_chars:
            # P8 — fully dropped.
            continue
        surviving_inputs.append(orig)

    assert len(result) == len(surviving_inputs), (
        f"survival count mismatch: result={len(result)} expected={len(surviving_inputs)}"
    )

    # P6 (order preservation) and P7 (payload preservation).
    for proj, orig in zip(result, surviving_inputs, strict=True):
        assert proj.feature_ids == orig.feature_ids
        assert proj.buckets == orig.buckets

    # P5 (shift / clamp invariant) and P9 (retention completeness).
    for proj, orig in zip(result, surviving_inputs, strict=True):
        # Expected char_start: max(orig.char_start - dropped_chars, 0).
        expected_start = max(orig.char_start - dropped_chars, 0)
        # Expected char_end: orig.char_end - dropped_chars (always positive since orig.char_end > dropped_chars).
        expected_end = orig.char_end - dropped_chars
        assert proj.char_start == expected_start
        assert proj.char_end == expected_end
        # Width shrinks by at most dropped_chars.
        original_width = orig.char_end - orig.char_start
        new_width = proj.char_end - proj.char_start
        assert new_width <= original_width
        # Fully-retained (char_start >= dropped_chars) preserves width.
        if orig.char_start >= dropped_chars:
            assert new_width == original_width


@settings(max_examples=100, derandomize=True)
@given(
    original=st.lists(_PRINTABLE_CHAR, min_size=2, max_size=15),
    bogus_truncated=st.lists(
        st.integers(min_value=ord("a"), max_value=ord("z")),  # disjoint alphabet
        min_size=1,
        max_size=10,
    ),
)
def test_p3_non_suffix_rejection(original, bogus_truncated) -> None:
    """If ``truncated`` is not a left-suffix of ``original``, projection raises ValueError.

    We use a disjoint alphabet (lowercase) for the bogus truncated, so the
    suffix check is guaranteed to fail (provided len(bogus_truncated) <= len(original)).
    The validator also raises if truncated is longer than original.
    """
    if len(bogus_truncated) > len(original):
        # Must raise length-mismatch ValueError.
        with pytest.raises(ValueError, match="exceeds original_prefix_tokens length"):
            project_segments_to_truncated_prefix(
                (
                    ContextFeatureSegment(
                        feature_ids=(0,), buckets=(0,), char_start=0, char_end=1,
                    ),
                ),
                original_prefix_tokens=list(original),
                truncated_prefix_tokens=list(bogus_truncated),
                tokenizer=_TOK,
            )
        return

    # The suffix check only fires when 0 < n_truncated < n_original.
    # Equal-length: n_dropped == 0 → fast no-op return regardless of contents.
    # Empty truncated: n_truncated == 0 → suffix check skipped per the helper.
    # So we only test the non-suffix rejection in the strict-shorter case.
    if len(bogus_truncated) == 0 or len(bogus_truncated) >= len(original):
        return

    # Otherwise: alphabets are disjoint, so the suffix check fails.
    with pytest.raises(ValueError, match="not a left-suffix"):
        project_segments_to_truncated_prefix(
            (
                ContextFeatureSegment(
                    feature_ids=(0,), buckets=(0,), char_start=0, char_end=1,
                ),
            ),
            original_prefix_tokens=list(original),
            truncated_prefix_tokens=list(bogus_truncated),
            tokenizer=_TOK,
        )


@settings(max_examples=100, derandomize=True)
@given(pair=_suffix_pair(), segments=st.data())
def test_p10_idempotency(pair, segments) -> None:
    """Projecting twice (second call with truncated == truncated) is a no-op.

    Specifically: project(segs, orig, trunc) → result. Then
    project(result, trunc, trunc) MUST return the same result, since the
    second call has dropped_chars == 0 and triggers the no-op fast path.
    """
    original, truncated = pair
    segs = segments.draw(_segments_strategy(max_char=len(original)))

    once = project_segments_to_truncated_prefix(
        segs,
        original_prefix_tokens=original,
        truncated_prefix_tokens=truncated,
        tokenizer=_TOK,
    )
    twice = project_segments_to_truncated_prefix(
        once,
        original_prefix_tokens=truncated,
        truncated_prefix_tokens=truncated,
        tokenizer=_TOK,
    )
    assert twice == once
