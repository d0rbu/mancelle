"""Unit tests for BOS-masked pooling primitives."""

from typing import cast

import pytest
import torch as t

from errors import BosOnlySegmentError
from steering.extract_features import pool_one_segment, pool_segments_batched_gpu
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode


def test_pool_one_segment_none_mask_unchanged():
    acts = t.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 5.0]])
    assert t.equal(pool_one_segment(acts, PoolingMode.max_pool),
                   acts.max(dim=0).values)
    assert t.equal(pool_one_segment(acts, PoolingMode.mean_pool),
                   acts.mean(dim=0))
    assert t.equal(pool_one_segment(acts, PoolingMode.last_token), acts[-1])


def test_pool_one_segment_mask_max_excludes_row0():
    acts = t.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 0.0]])
    mask = t.tensor([False, True, True])
    out = pool_one_segment(acts, PoolingMode.max_pool, keep_mask=mask)
    assert t.equal(out, t.tensor([3.0, 2.0]))


def test_pool_one_segment_mask_mean_excludes_row0():
    acts = t.tensor([[9.0, 9.0], [1.0, 3.0], [3.0, 1.0]])
    mask = t.tensor([False, True, True])
    out = pool_one_segment(acts, PoolingMode.mean_pool, keep_mask=mask)
    assert t.allclose(out, t.tensor([2.0, 2.0]))


def test_pool_one_segment_mask_last_token_skips_trailing_bos():
    acts = t.tensor([[1.0], [5.0], [9.0]])
    mask = t.tensor([True, True, False])  # last row is <bos>
    out = pool_one_segment(acts, PoolingMode.last_token, keep_mask=mask)
    assert t.equal(out, t.tensor([5.0]))


def test_pool_one_segment_all_bos_raises():
    acts = t.tensor([[9.0, 9.0]])
    mask = t.tensor([False])
    with pytest.raises(BosOnlySegmentError):
        pool_one_segment(acts, PoolingMode.max_pool, keep_mask=mask)


def _acts(seq, nfeat=2):
    g = t.arange(1, seq * nfeat + 1, dtype=t.float32).reshape(seq, nfeat)
    return g


def test_psbg_none_mask_is_unchanged_fastpath():
    acts = _acts(6)
    segs = [(0, 3), (3, 6)]
    for pm in (PoolingMode.max_pool, PoolingMode.mean_pool, PoolingMode.last_token):
        a = pool_segments_batched_gpu(acts, segs, pm, SegmentMode.fixed_tokens)
        b = pool_segments_batched_gpu(acts, segs, pm, SegmentMode.fixed_tokens,
                                      keep_mask=t.ones(6, dtype=t.bool))
        assert t.allclose(a, b), pm


def test_psbg_mask_drops_bos_seg0_max():
    acts = t.tensor([[99.0, 99.0], [1.0, 2.0], [4.0, 3.0],
                     [5.0, 6.0], [7.0, 8.0], [1.0, 1.0]])
    segs = [(0, 3), (3, 6)]
    mask = t.tensor([False, True, True, True, True, True])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.max_pool,
                                    SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.equal(out, t.tensor([[4.0, 3.0], [7.0, 8.0]]))


def test_psbg_mask_mean_counts_exclude_bos():
    acts = t.tensor([[99.0, 99.0], [1.0, 3.0], [3.0, 1.0],
                     [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    segs = [(0, 3), (3, 6)]
    mask = t.tensor([False, True, True, True, True, True])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.mean_pool,
                                    SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.allclose(out, t.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_psbg_mask_short_last_segment_with_bos():
    acts = t.tensor([[50.0], [2.0], [4.0]])
    segs = [(0, 3)]
    mask = t.tensor([False, True, True])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.mean_pool,
                                    SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.allclose(out, t.tensor([[3.0]]))


def test_psbg_all_bos_segment_raises():
    acts = t.tensor([[9.0], [9.0]])
    segs = [(0, 2)]
    mask = t.tensor([False, False])
    with pytest.raises(BosOnlySegmentError):
        pool_segments_batched_gpu(acts, segs, PoolingMode.max_pool,
                                  SegmentMode.fixed_tokens, keep_mask=mask)


def test_psbg_last_token_mask_picks_last_non_bos():
    acts = t.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [9.0]])
    segs = [(0, 3), (3, 6)]
    mask = t.tensor([True, True, True, True, True, False])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.last_token,
                                    SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.equal(out, t.tensor([[3.0], [5.0]]))


def test_psbg_multiple_bos_in_one_segment_all_modes():
    # seg0 = rows0..3 with TWO bos (rows 0 and 2); seg1 = rows4,5 clean
    acts = t.tensor([[9.0, 9.0], [1.0, 4.0], [8.0, 8.0], [3.0, 2.0],
                     [5.0, 5.0], [7.0, 1.0]])
    segs = [(0, 4), (4, 6)]
    mask = t.tensor([False, True, False, True, True, True])
    # seg0 kept rows = {1:[1,4], 3:[3,2]} ; seg1 kept = {4:[5,5],5:[7,1]}
    mx = pool_segments_batched_gpu(acts, segs, PoolingMode.max_pool,
                                   SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.equal(mx, t.tensor([[3.0, 4.0], [7.0, 5.0]]))
    mn = pool_segments_batched_gpu(acts, segs, PoolingMode.mean_pool,
                                   SegmentMode.fixed_tokens, keep_mask=mask)
    assert t.allclose(mn, t.tensor([[2.0, 3.0], [6.0, 3.0]]))
    lt = pool_segments_batched_gpu(acts, segs, PoolingMode.last_token,
                                   SegmentMode.fixed_tokens, keep_mask=mask)
    # last kept index seg0 = row3 -> [3,2]; seg1 = row5 -> [7,1]
    assert t.equal(lt, t.tensor([[3.0, 2.0], [7.0, 1.0]]))


def test_pool_one_segment_multiple_bos():
    acts = t.tensor([[9.0], [2.0], [9.0], [4.0]])
    mask = t.tensor([False, True, False, True])
    assert t.equal(pool_one_segment(acts, PoolingMode.max_pool, keep_mask=mask),
                   t.tensor([4.0]))
    assert t.allclose(pool_one_segment(acts, PoolingMode.mean_pool, keep_mask=mask),
                      t.tensor([3.0]))
    assert t.equal(pool_one_segment(acts, PoolingMode.last_token, keep_mask=mask),
                   t.tensor([4.0]))


# ---------------------------------------------------------------------------
# Hardening tests: all-True≡no-mask (incl bf16), all-modes BosOnly,
# length/unknown-mode guards, variable-segment fallback
# ---------------------------------------------------------------------------

def test_pool_one_segment_all_true_mask_equals_no_mask():
    """all-True keep_mask must be bit-identical to keep_mask=None (float32)."""
    acts = t.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 5.0]])
    mask_all_true = t.ones(3, dtype=t.bool)
    for pm in list(PoolingMode):
        result_masked = pool_one_segment(acts, pm, keep_mask=mask_all_true)
        result_unmasked = pool_one_segment(acts, pm)
        assert t.equal(result_masked, result_unmasked), (
            f"PoolingMode.{pm.name}: all-True mask != no-mask in float32"
        )


def test_pool_one_segment_all_true_mask_equals_no_mask_bf16():
    """all-True keep_mask must be bit-identical to keep_mask=None in bf16.

    This pins the F1 fix: all-True must fall through to acts.mean()/acts.max()/
    acts[-1] rather than sum/count, which diverges from .mean() in bf16 due to
    reduction-order differences.
    """
    acts = t.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 5.0]]).bfloat16()
    mask_all_true = t.ones(3, dtype=t.bool)
    for pm in list(PoolingMode):
        result_masked = pool_one_segment(acts, pm, keep_mask=mask_all_true)
        result_unmasked = pool_one_segment(acts, pm)
        assert t.equal(result_masked, result_unmasked), (
            f"PoolingMode.{pm.name}: all-True mask != no-mask in bf16"
        )


def test_pool_segments_batched_gpu_all_true_mask_equals_no_mask():
    """all-True keep_mask in psbg must be allclose to keep_mask=None."""
    acts = t.tensor([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
        [7.0, 8.0], [9.0, 0.0], [1.0, 3.0],
    ])
    segs = [(0, 3), (3, 6)]
    mask_all_true = t.ones(6, dtype=t.bool)
    for pm in list(PoolingMode):
        result_masked = pool_segments_batched_gpu(
            acts, segs, pm, SegmentMode.fixed_tokens, keep_mask=mask_all_true
        )
        result_unmasked = pool_segments_batched_gpu(
            acts, segs, pm, SegmentMode.fixed_tokens
        )
        assert t.allclose(result_masked, result_unmasked), (
            f"PoolingMode.{pm.name}: all-True mask != no-mask in psbg"
        )


def test_pool_one_segment_keep_mask_wrong_length_raises():
    """keep_mask with wrong length raises ValueError mentioning 'keep_mask length'."""
    acts = t.ones(3, 2)
    mask = t.tensor([True, False])  # length 2, acts has 3 rows
    with pytest.raises(ValueError, match="keep_mask length"):
        pool_one_segment(acts, PoolingMode.max_pool, keep_mask=mask)


@pytest.mark.parametrize("pm", list(PoolingMode))
def test_pool_one_segment_all_bos_raises_all_modes(pm: PoolingMode):
    """All-False keep_mask raises BosOnlySegmentError for every PoolingMode."""
    acts = t.tensor([[9.0, 9.0]])
    mask = t.tensor([False])
    with pytest.raises(BosOnlySegmentError):
        pool_one_segment(acts, pm, keep_mask=mask)


@pytest.mark.parametrize("pm", list(PoolingMode))
def test_psbg_all_bos_raises_all_modes(pm: PoolingMode):
    """All-False segment mask raises BosOnlySegmentError in psbg for every mode."""
    acts = t.tensor([[9.0], [9.0]])
    segs = [(0, 2)]
    mask = t.tensor([False, False])
    with pytest.raises(BosOnlySegmentError):
        pool_segments_batched_gpu(
            acts, segs, pm, SegmentMode.fixed_tokens, keep_mask=mask
        )


def test_psbg_all_bos_error_message_has_segment_index():
    """BosOnlySegmentError message must contain the segment index."""
    # Multi-segment case: segment at index 2 is all-bos
    acts = t.tensor([[1.0], [2.0], [3.0]])
    segs = [(0, 1), (1, 2), (2, 3)]
    mask = t.tensor([True, True, False])
    with pytest.raises(BosOnlySegmentError, match=r"Segment 2"):
        pool_segments_batched_gpu(
            acts, segs, PoolingMode.max_pool,
            SegmentMode.fixed_tokens, keep_mask=mask,
        )

    # Single-segment case: segment at index 0
    acts1 = t.tensor([[7.0]])
    segs1 = [(0, 1)]
    mask1 = t.tensor([False])
    with pytest.raises(BosOnlySegmentError, match=r"Segment 0"):
        pool_segments_batched_gpu(
            acts1, segs1, PoolingMode.max_pool,
            SegmentMode.fixed_tokens, keep_mask=mask1,
        )


def test_pool_one_segment_unknown_pooling_with_partial_mask_raises():
    """A bogus pooling value on the masked branch raises ValueError."""
    acts = t.tensor([[1.0, 1.0], [2.0, 2.0]])
    # partial mask (not all-True, not all-False) routes to masked branch
    mask = t.tensor([True, False])
    bogus_pm = cast(PoolingMode, "bogus")
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        pool_one_segment(acts, bogus_pm, keep_mask=mask)


# ---------------------------------------------------------------------------
# Arbitrary-layout compatibility fallback tests. Turn-delimited production
# spans are contiguous and vectorized, but this helper historically accepted
# overlapping spans. Those calls must remain on the loop because one token can
# belong to multiple segments. ``last_token`` returns before either route.
# ---------------------------------------------------------------------------

_FALLBACK_MODE = SegmentMode.turn_delimited_fixed_tokens


def test_psbg_fallback_max_with_mask():
    """Fallback path: max_pool with partial keep_mask, hand-derived values.

    acts = [[1,2],[3,4],[5,6]], segs=[(0,2),(1,3)], mask=[F,T,T]
    Seg0 rows0,1: keep row1=[3,4] only (row0 masked) → masked_fill row0→0 →
        max([0,0],[3,4]) = [3,4]
    Seg1 rows1,2: keep both → max([3,4],[5,6]) = [5,6]
    """
    acts = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segs = [(0, 2), (1, 3)]
    mask = t.tensor([False, True, True])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.max_pool, _FALLBACK_MODE,
                                    keep_mask=mask)
    assert t.equal(out, t.tensor([[3.0, 4.0], [5.0, 6.0]]))


def test_psbg_fallback_mean_with_mask():
    """Fallback path: mean_pool with partial keep_mask, hand-derived values.

    acts = [[1,2],[3,4],[5,6]], segs=[(0,2),(1,3)], mask=[F,T,T]
    Seg0: keep row1=[3,4] only → sum=[3,4]/count=1 → [3,4]
    Seg1: keep rows1,2=[3,4],[5,6] → sum=[8,10]/count=2 → [4,5]
    """
    acts = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segs = [(0, 2), (1, 3)]
    mask = t.tensor([False, True, True])
    out = pool_segments_batched_gpu(acts, segs, PoolingMode.mean_pool, _FALLBACK_MODE,
                                    keep_mask=mask)
    assert t.equal(out, t.tensor([[3.0, 4.0], [4.0, 5.0]]))


def test_psbg_fallback_all_bos_raises():
    """Fallback path: fully-masked segment raises BosOnlySegmentError."""
    acts = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segs = [(0, 2), (1, 3)]
    # Seg1 rows1,2 are both masked → BosOnlySegmentError
    mask = t.tensor([True, False, False])
    with pytest.raises(BosOnlySegmentError):
        pool_segments_batched_gpu(acts, segs, PoolingMode.max_pool, _FALLBACK_MODE,
                                  keep_mask=mask)


def test_psbg_fallback_unknown_pooling_raises():
    """Fallback path: bogus pooling hits the else-raise guard."""
    acts = t.tensor([[1.0, 2.0], [3.0, 4.0]])
    segs = [(0, 2)]
    bogus_pm = cast(PoolingMode, "bogus")
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        pool_segments_batched_gpu(acts, segs, bogus_pm, _FALLBACK_MODE)
