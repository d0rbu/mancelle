"""Slice 14 audit X2: ``filter_segments_to_target`` invariants.

Pin the rebased-segment monotonicity / contiguity / partial-coverage
extension rules. Tests cover (a) the canonical happy path, (b) gaps and
overlaps reject, (c) partial coverage within the canonical
``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS = 20`` tolerance is absorbed,
(d) partial coverage past 20 is rejected — directly testing the Q12
behaviour change relative to the previous local 32-char window.
"""

from __future__ import annotations

import pytest

from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.target_segments import filter_segments_to_target
from steering.segmentation import CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS


def _seg(start: int, end: int) -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=(0,), buckets=(1,), char_start=start, char_end=end
    )


def test_canonical_value_is_20() -> None:
    """Pin the canonical value of the cross-tokenizer end-marker window."""
    assert CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS == 20


def test_happy_path_three_segments_tile_target() -> None:
    """Three contiguous segments fully cover the target → rebased to [0,T)."""
    segs = [_seg(100, 110), _seg(110, 120), _seg(120, 130)]
    result = filter_segments_to_target(
        segs, target_char_start=100, target_char_end=130
    )
    assert len(result) == 3
    assert result[0].char_start == 0 and result[0].char_end == 10
    assert result[1].char_start == 10 and result[1].char_end == 20
    assert result[2].char_start == 20 and result[2].char_end == 30


def test_partial_coverage_within_tolerance_extends_last_segment() -> None:
    """Partial coverage within the 20-char window absorbs into last seg."""
    # Segments cover [100, 115); target is [100, 125). Gap = 10 ≤ 20.
    segs = [_seg(100, 110), _seg(110, 115)]
    result = filter_segments_to_target(
        segs, target_char_start=100, target_char_end=125
    )
    # Last segment extends to target_len = 25.
    assert result[-1].char_end == 25


def test_partial_coverage_at_exactly_20_absorbs() -> None:
    """A 20-char gap is exactly at the tolerance boundary and absorbs."""
    segs = [_seg(100, 105)]
    result = filter_segments_to_target(
        segs, target_char_start=100, target_char_end=125
    )
    assert result[-1].char_end == 25


def test_partial_coverage_past_20_rejects() -> None:
    """A 21-char gap exceeds the canonical 20-char window → ValueError.

    Slice 14 audit Q12 user-attention behaviour change: the previous
    local 32-char window in target_segments would have absorbed gaps in
    [21, 32]. Now they reject (matching extraction + SFT data prep).
    """
    segs = [_seg(100, 104)]  # gap = 26 chars.
    with pytest.raises(ValueError, match="[Pp]artial|[Ff]orward|exceeds"):
        filter_segments_to_target(
            segs, target_char_start=100, target_char_end=130
        )


@pytest.mark.parametrize("gap", [21, 25, 30, 32])
def test_partial_coverage_in_old_32_window_now_rejects(gap: int) -> None:
    """Q12 behaviour change pin: char-windows in [21, 32] no longer absorb.

    Previously the local 32 in target_segments accepted these. Now the
    canonical 20 in segmentation rejects.
    """
    target_len = 30
    last_end = target_len - gap  # so gap chars uncovered.
    assert gap > CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    segs = [_seg(100, 100 + last_end)] if last_end > 0 else [_seg(100, 101)]
    with pytest.raises(ValueError, match="[Pp]artial|[Ff]orward|exceeds"):
        filter_segments_to_target(
            segs, target_char_start=100, target_char_end=100 + target_len
        )


def test_gap_between_segments_rejects() -> None:
    """Internal gaps reject regardless of tolerance."""
    segs = [_seg(100, 105), _seg(110, 115)]
    with pytest.raises(ValueError, match="gap"):
        filter_segments_to_target(
            segs, target_char_start=100, target_char_end=115
        )


def test_overlap_between_segments_rejects() -> None:
    """Internal overlap rejects regardless of tolerance."""
    segs = [_seg(100, 110), _seg(105, 120)]
    with pytest.raises(ValueError, match="overlap"):
        filter_segments_to_target(
            segs, target_char_start=100, target_char_end=120
        )


def test_no_segments_overlap_rejects() -> None:
    """No overlapping segments → ValueError."""
    segs = [_seg(0, 10), _seg(50, 60)]
    with pytest.raises(ValueError, match="no segments overlap"):
        filter_segments_to_target(
            segs, target_char_start=20, target_char_end=40
        )


def test_first_segment_must_start_at_zero_after_rebase() -> None:
    """If the rebased first segment does not start at 0, reject."""
    segs = [_seg(105, 130)]  # target = [100, 130) → rebased start = 5.
    with pytest.raises(ValueError, match="not covered"):
        filter_segments_to_target(
            segs, target_char_start=100, target_char_end=130
        )


def test_target_range_must_be_positive() -> None:
    segs = [_seg(0, 10)]
    with pytest.raises(ValueError, match="positive"):
        filter_segments_to_target(segs, target_char_start=10, target_char_end=10)
    with pytest.raises(ValueError, match="positive"):
        filter_segments_to_target(segs, target_char_start=20, target_char_end=10)


def test_segment_clipped_to_target_bounds() -> None:
    """Segments that overhang the target are clipped on both sides."""
    segs = [_seg(50, 200)]  # target = [100, 150).
    result = filter_segments_to_target(
        segs, target_char_start=100, target_char_end=150
    )
    assert len(result) == 1
    assert result[0].char_start == 0
    assert result[0].char_end == 50
