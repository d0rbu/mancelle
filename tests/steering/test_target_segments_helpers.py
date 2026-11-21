"""PR-C: pure helpers for target-real conditioning.

Covers: filter_segments_to_target (range overlap + char rebase),
pool_segments_to_dict (BUCKET_MAX, BUCKET_MEAN), and PoolingOp parsing.
"""

from __future__ import annotations

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.target_segments import (
    filter_segments_to_target,
    pool_segments_to_dict,
)

# ---- PoolingOp ----

def test_pooling_op_str_round_trip():
    assert PoolingOp("bucket_max") is PoolingOp.BUCKET_MAX
    assert PoolingOp("bucket_mean") is PoolingOp.BUCKET_MEAN


def test_pooling_op_invalid_value_raises():
    with pytest.raises(ValueError):
        PoolingOp("bucket_median")


# ---- filter_segments_to_target ----

def _seg(fids, buckets, char_start, char_end):
    return ContextFeatureSegment(
        feature_ids=tuple(fids), buckets=tuple(buckets),
        char_start=char_start, char_end=char_end,
    )


def test_filter_excess_coverage_clipped_to_target_range():
    """Extraction covers chars [0, 100]; target is [40, 90].
    Result keeps segments overlapping [40, 90), rebased to [0, 50)."""
    segs = (
        _seg([1], [1], 0, 30),     # before target — drop
        _seg([2], [2], 30, 50),    # straddles start — keep, rebase to (0, 10)
        _seg([3], [3], 50, 80),    # inside target — keep, rebase to (10, 40)
        _seg([4], [4], 80, 100),   # straddles end — keep, rebase to (40, 50)
    )
    out = filter_segments_to_target(segs, target_char_start=40, target_char_end=90)
    assert len(out) == 3
    assert out[0].feature_ids == (2,) and out[0].char_start == 0 and out[0].char_end == 10
    assert out[1].feature_ids == (3,) and out[1].char_start == 10 and out[1].char_end == 40
    assert out[2].feature_ids == (4,) and out[2].char_start == 40 and out[2].char_end == 50


def test_filter_segments_within_target_range_passthrough_with_rebase():
    """Target is [10, 60]; segments cover [10, 25), [25, 40), [40, 60)."""
    segs = (
        _seg([1], [1], 10, 25),
        _seg([2], [2], 25, 40),
        _seg([3], [3], 40, 60),
    )
    out = filter_segments_to_target(segs, target_char_start=10, target_char_end=60)
    assert [(s.char_start, s.char_end) for s in out] == [(0, 15), (15, 30), (30, 50)]


def test_filter_partial_coverage_within_tolerance_extends_last_segment():
    """Mirrors the canonical CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS rule:
    small uncovered tail (<= tolerance) extends the last segment."""
    segs = (
        _seg([1], [1], 0, 28),  # covers 28/30 chars; 2 uncovered (within tolerance).
    )
    out = filter_segments_to_target(segs, target_char_start=0, target_char_end=30)
    assert len(out) == 1
    assert out[0].char_start == 0 and out[0].char_end == 30  # extended


def test_filter_partial_coverage_too_large_raises():
    """Large uncovered tail -> ValueError."""
    segs = (_seg([1], [1], 0, 10),)
    with pytest.raises(ValueError, match="[Pp]artial|[Ff]orward|exceeds"):
        filter_segments_to_target(segs, target_char_start=0, target_char_end=100)


def test_filter_no_overlap_raises():
    segs = (_seg([1], [1], 0, 10),)
    with pytest.raises(ValueError, match="no segments overlap"):
        filter_segments_to_target(segs, target_char_start=50, target_char_end=80)


def test_filter_validates_segment_tiling():
    """Overlapping segments after rebase -> ValueError."""
    segs = (
        _seg([1], [1], 10, 30),
        _seg([2], [2], 20, 40),  # overlap with previous
    )
    with pytest.raises(ValueError, match="overlap|tiling"):
        filter_segments_to_target(segs, target_char_start=0, target_char_end=40)


# ---- pool_segments_to_dict ----

def test_pool_bucket_max_overlapping_features():
    """Feature 1: max(2, 5) = 5. Feature 2: max(3, 1) = 3. Feature 3: 7."""
    segs = (
        _seg([1, 2], [2, 3], 0, 10),
        _seg([1, 3], [5, 7], 10, 20),
        _seg([2], [1], 20, 30),
    )
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MAX)
    assert out == {1: 5, 2: 3, 3: 7}


def test_pool_bucket_mean_rounded_to_nearest_int():
    """Feature 1: mean(2, 4, 6) = 4. Feature 2: mean(3) = 3."""
    segs = (
        _seg([1], [2], 0, 5),
        _seg([1, 2], [4, 3], 5, 10),
        _seg([1], [6], 10, 15),
    )
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MEAN)
    assert out == {1: 4, 2: 3}


def test_pool_empty_segments_returns_empty_dict():
    out = pool_segments_to_dict((), op=PoolingOp.BUCKET_MAX)
    assert out == {}


def test_pool_single_segment_round_trip():
    segs = (_seg([5, 7, 9], [1, 2, 3], 0, 10),)
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MAX)
    assert out == {5: 1, 7: 2, 9: 3}


def test_segment_rejects_negative_feature_id_before_pooling():
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        _seg([-1], [1], 0, 10)


def test_segment_rejects_negative_bucket_before_pooling():
    with pytest.raises(BeartypeCallHintParamViolation, match="buckets"):
        _seg([1], [-1], 0, 10)


# ---- BUCKET_MEAN rounding invariants ----

@pytest.mark.parametrize("pair,expected", [
    ((0, 1), 1),   # 0.5 -> rounds UP to 1 (not banker's 0)
    ((1, 2), 2),   # 1.5 -> rounds UP to 2 (not banker's 2, same here — but (1,2)=1.5->2)
    ((2, 3), 3),   # 2.5 -> rounds UP to 3 (not banker's 2)
])
def test_pool_bucket_mean_half_rounds_up_not_banker(pair, expected):
    """BUCKET_MEAN uses round-half-up, not Python's banker's rounding.

    Python's built-in round() uses banker's rounding (round-half-to-even):
      round(0.5) == 0, round(1.5) == 2, round(2.5) == 2.
    We require round-half-up instead:
      0.5 -> 1, 1.5 -> 2, 2.5 -> 3.
    """
    fid = 42
    segs = tuple(_seg([fid], [v], i * 10, (i + 1) * 10) for i, v in enumerate(pair))
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MEAN)
    assert out[fid] == expected, (
        f"pair={pair}: expected {expected}, got {out[fid]} "
        f"(banker's rounding would give {round(sum(pair) / len(pair))})"
    )


def test_pool_bucket_mean_three_value_average_even():
    """Feature 1: mean(2, 4, 6) == 4 (integer, no rounding needed)."""
    segs = (
        _seg([1], [2], 0, 5),
        _seg([1], [4], 5, 10),
        _seg([1], [6], 10, 15),
    )
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MEAN)
    assert out == {1: 4}


def test_pool_bucket_mean_two_value_average_exact():
    """Feature 1: mean(2, 4) == 3 (exact integer, no rounding needed)."""
    segs = (
        _seg([1], [2], 0, 5),
        _seg([1], [4], 5, 10),
    )
    out = pool_segments_to_dict(segs, op=PoolingOp.BUCKET_MEAN)
    assert out == {1: 3}
