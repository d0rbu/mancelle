"""Hand-computed correctness tests for pooling modes and sparse extraction.

Verifies pool_segments_batched_gpu and extract_sparse_gpu produce exact expected
results for hand-crafted inputs, covering the fixed-token and turn-delimited
routes plus edge cases and cross-route equivalence. The arbitrary-span
compatibility fallback is covered in ``test_ignore_bos_pooling.py`` and
``test_turn_pooling_vectorized.py``; no test in this module reaches it.

Run with:
    uv run pytest tests/test_pooling_correctness.py -v
"""

import torch as t

from steering.extract_features import (
    extract_sparse_gpu,
    pool_segments_batched_gpu,
)
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens


def _hand_activations_10x4():
    """10 tokens, 4 features -- fully deterministic, no randomness.

    Returns (sae_acts, segments) where segments partition into seg_size=4:
        [(0,4), (4,8), (8,10)]   -- last segment has 2 tokens.
    """
    sae_acts = t.tensor(
        [
            [1.0, 0.0, 3.0, 0.0],  # token 0
            [0.0, 2.0, 0.0, 4.0],  # token 1
            [5.0, 0.0, 1.0, 0.0],  # token 2
            [0.0, 6.0, 0.0, 2.0],  # token 3
            [3.0, 0.0, 7.0, 0.0],  # token 4
            [0.0, 8.0, 0.0, 1.0],  # token 5
            [2.0, 0.0, 9.0, 0.0],  # token 6
            [0.0, 4.0, 0.0, 3.0],  # token 7
            [6.0, 0.0, 2.0, 0.0],  # token 8
            [0.0, 5.0, 0.0, 7.0],  # token 9
        ],
        dtype=t.float32,
    )
    segments = [(0, 4), (4, 8), (8, 10)]
    return sae_acts, segments


def _hand_activations_8x3_uniform():
    """8 tokens, 3 features -- all segments of size 4 (uniform, no short tail).

    Returns (sae_acts, segments) where segments are [(0,4), (4,8)].
    """
    sae_acts = t.tensor(
        [
            [1.0, 2.0, 3.0],  # token 0
            [4.0, 5.0, 6.0],  # token 1
            [7.0, 8.0, 9.0],  # token 2
            [10.0, 11.0, 12.0],  # token 3
            [2.0, 3.0, 4.0],  # token 4
            [5.0, 6.0, 7.0],  # token 5
            [8.0, 9.0, 10.0],  # token 6
            [11.0, 12.0, 13.0],  # token 7
        ],
        dtype=t.float32,
    )
    segments = [(0, 4), (4, 8)]
    return sae_acts, segments


# ---------------------------------------------------------------------------
# 1. Hand-computed max_pool tests
# ---------------------------------------------------------------------------


class TestMaxPoolHandComputed:
    """Verify max_pool against hand-computed expected values."""

    def test_uniform_segments(self):
        sae_acts, segments = _hand_activations_8x3_uniform()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        # Segment 0: max over tokens 0-3
        #   feat 0: max(1,4,7,10)  = 10
        #   feat 1: max(2,5,8,11)  = 11
        #   feat 2: max(3,6,9,12)  = 12
        expected_seg0 = t.tensor([10.0, 11.0, 12.0])
        # Segment 1: max over tokens 4-7
        #   feat 0: max(2,5,8,11)  = 11
        #   feat 1: max(3,6,9,12)  = 12
        #   feat 2: max(4,7,10,13) = 13
        expected_seg1 = t.tensor([11.0, 12.0, 13.0])

        assert t.allclose(result[0], expected_seg0)
        assert t.allclose(result[1], expected_seg1)

    def test_short_last_segment(self):
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        # Segment 0 (tokens 0-3): max along dim=0
        #   feat 0: max(1,0,5,0) = 5
        #   feat 1: max(0,2,0,6) = 6
        #   feat 2: max(3,0,1,0) = 3
        #   feat 3: max(0,4,0,2) = 4
        expected_seg0 = t.tensor([5.0, 6.0, 3.0, 4.0])
        # Segment 1 (tokens 4-7):
        #   feat 0: max(3,0,2,0) = 3
        #   feat 1: max(0,8,0,4) = 8
        #   feat 2: max(7,0,9,0) = 9
        #   feat 3: max(0,1,0,3) = 3
        expected_seg1 = t.tensor([3.0, 8.0, 9.0, 3.0])
        # Segment 2 (tokens 8-9 only):
        #   feat 0: max(6,0) = 6
        #   feat 1: max(0,5) = 5
        #   feat 2: max(2,0) = 2
        #   feat 3: max(0,7) = 7
        expected_seg2 = t.tensor([6.0, 5.0, 2.0, 7.0])

        assert t.allclose(result[0], expected_seg0)
        assert t.allclose(result[1], expected_seg1)
        assert t.allclose(result[2], expected_seg2)

    def test_padded_zeros_dont_affect_max(self):
        """Since SAE activations are non-negative, zero padding cannot inflate max."""
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        # Manually compute the max over *only* the actual tokens for the last segment
        manual_last_max = sae_acts[8:10].max(dim=0).values
        assert t.allclose(result[2], manual_last_max), (
            f"Padding zeros corrupted max_pool: got {result[2]}, expected {manual_last_max}"
        )


# ---------------------------------------------------------------------------
# 2. Hand-computed mean_pool tests
# ---------------------------------------------------------------------------


class TestMeanPoolHandComputed:
    """Verify mean_pool divides by actual token count, not padded length."""

    def test_uniform_segments(self):
        sae_acts, segments = _hand_activations_8x3_uniform()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        # Segment 0: mean over tokens 0-3
        #   feat 0: (1+4+7+10)/4  = 5.5
        #   feat 1: (2+5+8+11)/4  = 6.5
        #   feat 2: (3+6+9+12)/4  = 7.5
        expected_seg0 = t.tensor([5.5, 6.5, 7.5])
        # Segment 1: mean over tokens 4-7
        #   feat 0: (2+5+8+11)/4  = 6.5
        #   feat 1: (3+6+9+12)/4  = 7.5
        #   feat 2: (4+7+10+13)/4 = 8.5
        expected_seg1 = t.tensor([6.5, 7.5, 8.5])

        assert t.allclose(result[0], expected_seg0)
        assert t.allclose(result[1], expected_seg1)

    def test_short_last_segment_divides_by_actual_count(self):
        """CRITICAL: last segment of 2 tokens must divide by 2, not by seg_size=4."""
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        # Segment 2 has only tokens 8 and 9
        #   feat 0: (6+0)/2 = 3.0    NOT (6+0+0+0)/4 = 1.5
        #   feat 1: (0+5)/2 = 2.5    NOT (0+5+0+0)/4 = 1.25
        #   feat 2: (2+0)/2 = 1.0    NOT (2+0+0+0)/4 = 0.5
        #   feat 3: (0+7)/2 = 3.5    NOT (0+7+0+0)/4 = 1.75
        expected_seg2 = t.tensor([3.0, 2.5, 1.0, 3.5])
        wrong_if_padded = t.tensor([1.5, 1.25, 0.5, 1.75])

        assert t.allclose(result[2], expected_seg2, atol=1e-6), (
            f"mean_pool divided by padded length: got {result[2]}, expected {expected_seg2}"
        )
        assert not t.allclose(result[2], wrong_if_padded, atol=1e-6), (
            "mean_pool incorrectly divided by seg_size instead of actual count"
        )

    def test_full_segments_match_naive_mean(self):
        """Full-sized segments should give the same result as sae_acts[s:e].mean(dim=0)."""
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        for i, (s, e) in enumerate(segments):
            naive_mean = sae_acts[s:e].mean(dim=0)
            assert t.allclose(result[i], naive_mean, atol=1e-6), (
                f"Segment {i}: fast-path mean {result[i]} != naive mean {naive_mean}"
            )


# ---------------------------------------------------------------------------
# 3. Hand-computed last_token tests
# ---------------------------------------------------------------------------


class TestLastTokenHandComputed:
    """Verify last_token picks the correct token from each segment."""

    def test_each_segment_gets_its_last_token(self):
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.last_token, SegmentMode.fixed_tokens
        )
        # Segment 0: last token is index 3
        assert t.allclose(result[0], sae_acts[3])
        # Segment 1: last token is index 7
        assert t.allclose(result[1], sae_acts[7])
        # Segment 2: last token is index 9 (NOT a padding position)
        assert t.allclose(result[2], sae_acts[9])

    def test_short_last_segment_uses_correct_token(self):
        """Last segment (8,10) should use token 9, not token 11 or padded position."""
        sae_acts, segments = _hand_activations_10x4()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.last_token, SegmentMode.fixed_tokens
        )
        expected = t.tensor([0.0, 5.0, 0.0, 7.0])
        assert t.allclose(result[2], expected), (
            f"Short last segment took wrong token: got {result[2]}, expected {expected}"
        )

    def test_uniform_segments(self):
        sae_acts, segments = _hand_activations_8x3_uniform()
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.last_token, SegmentMode.fixed_tokens
        )
        # Segment 0: last token is index 3 → [10, 11, 12]
        assert t.allclose(result[0], t.tensor([10.0, 11.0, 12.0]))
        # Segment 1: last token is index 7 → [11, 12, 13]
        assert t.allclose(result[1], t.tensor([11.0, 12.0, 13.0]))


# ---------------------------------------------------------------------------
# 4. Sparse extraction correctness
# ---------------------------------------------------------------------------


class TestSparseExtractionCorrectness:
    """Verify extract_sparse_gpu invariants on hand-crafted pooled tensors."""

    def test_all_magnitudes_positive(self):
        """No zero-valued magnitudes should be returned."""
        pooled = t.tensor(
            [
                [0.0, 3.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        ids, mags = extract_sparse_gpu(pooled)
        for seg_idx in range(3):
            if len(mags[seg_idx]) > 0:
                assert (mags[seg_idx] > 0).all(), (
                    f"Segment {seg_idx}: found non-positive magnitude"
                )

    def test_feature_ids_within_range(self):
        """All feature IDs must be in [0, n_features)."""
        t.manual_seed(42)
        n_feat = 64
        pooled = t.rand(5, n_feat)
        pooled[pooled < 0.6] = 0.0
        ids, mags = extract_sparse_gpu(pooled)
        for seg_idx in range(5):
            if len(ids[seg_idx]) > 0:
                assert (ids[seg_idx] >= 0).all()
                assert (ids[seg_idx] < n_feat).all(), (
                    f"Segment {seg_idx}: feature ID out of range"
                )

    def test_magnitudes_sorted_descending(self):
        """Magnitudes within each segment must be in descending order."""
        pooled = t.tensor(
            [
                [1.0, 5.0, 3.0, 0.0, 2.0, 4.0],
                [9.0, 0.0, 7.0, 0.0, 8.0, 0.0],
            ]
        )
        ids, mags = extract_sparse_gpu(pooled)
        # Segment 0: non-zero are 1,5,3,2,4 → sorted desc: 5,4,3,2,1
        assert mags[0].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]
        # Segment 1: non-zero are 9,7,8 → sorted desc: 9,8,7
        assert mags[1].tolist() == [9.0, 8.0, 7.0]

    def test_correct_feature_ids_returned(self):
        """Feature IDs must correspond to the correct columns."""
        pooled = t.tensor([[0.0, 3.0, 0.0, 1.0, 5.0]])
        ids, mags = extract_sparse_gpu(pooled)
        # Sorted by magnitude desc: 5.0 (col 4), 3.0 (col 1), 1.0 (col 3)
        assert ids[0].tolist() == [4, 1, 3]
        assert mags[0].tolist() == [5.0, 3.0, 1.0]

    def test_mix_of_zero_and_nonzero(self):
        """Segments with a mix of zeros return only the non-zero entries."""
        pooled = t.tensor(
            [
                [0.0, 0.0, 7.0],
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
        ids, mags = extract_sparse_gpu(pooled)
        # Segment 0: only feature 2
        assert ids[0].tolist() == [2]
        assert mags[0].tolist() == [7.0]
        # Segment 1: all features, sorted desc: 3(col2), 2(col1), 1(col0)
        assert ids[1].tolist() == [2, 1, 0]
        assert mags[1].tolist() == [3.0, 2.0, 1.0]
        # Segment 2: empty
        assert len(ids[2]) == 0
        assert len(mags[2]) == 0


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestPoolingEdgeCases:
    """Boundary conditions for pooling and sparse extraction."""

    def test_single_token_segments(self):
        """Each segment is a single token -- all pooling modes return that token."""
        sae_acts = t.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        segments = [(0, 1), (1, 2), (2, 3)]
        for mode in PoolingMode:
            result = pool_segments_batched_gpu(
                sae_acts, segments, mode, SegmentMode.fixed_tokens
            )
            assert result.shape == (3, 2)
            for i in range(3):
                assert t.allclose(result[i], sae_acts[i]), (
                    f"mode={mode.value}, seg {i}: {result[i]} != {sae_acts[i]}"
                )

    def test_single_segment_covering_entire_sequence(self):
        """One segment spanning all tokens."""
        t.manual_seed(42)
        sae_acts = t.rand(20, 8)
        segments = [(0, 20)]
        # max_pool
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        assert t.allclose(result[0], sae_acts.max(dim=0).values)
        # mean_pool
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        assert t.allclose(result[0], sae_acts.mean(dim=0))
        # last_token
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.last_token, SegmentMode.fixed_tokens
        )
        assert t.allclose(result[0], sae_acts[19])

    def test_all_zero_activations_sparse_extraction(self):
        """All-zero pooled tensor yields empty sparse results for every segment."""
        pooled = t.zeros(5, 16)
        ids, mags = extract_sparse_gpu(pooled)
        assert len(ids) == 5
        assert len(mags) == 5
        for i in range(5):
            assert len(ids[i]) == 0
            assert len(mags[i]) == 0

    def test_all_nonzero_activations_dense_sparse(self):
        """Fully non-zero tensor returns every feature for every segment."""
        pooled = t.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        ids, mags = extract_sparse_gpu(pooled)
        for i in range(2):
            assert len(ids[i]) == 3, (
                f"Segment {i}: expected 3 features, got {len(ids[i])}"
            )
            assert len(mags[i]) == 3

    def test_many_segments(self):
        """1000 segments -- stress test that counts and shapes remain correct."""
        t.manual_seed(42)
        n_segs = 1000
        seg_size = 4
        n_tokens = n_segs * seg_size
        n_feat = 32
        sae_acts = t.rand(n_tokens, n_feat)
        segments = [(i * seg_size, (i + 1) * seg_size) for i in range(n_segs)]

        for mode in PoolingMode:
            result = pool_segments_batched_gpu(
                sae_acts, segments, mode, SegmentMode.fixed_tokens
            )
            assert result.shape == (n_segs, n_feat), (
                f"mode={mode.value}: expected ({n_segs}, {n_feat}), got {result.shape}"
            )

    def test_many_segments_sparse_extraction(self):
        """1000 segments -- sparse extraction returns the right number of lists."""
        t.manual_seed(42)
        n_segs = 1000
        pooled = t.rand(n_segs, 64)
        pooled[pooled < 0.7] = 0.0
        ids, mags = extract_sparse_gpu(pooled)
        assert len(ids) == n_segs
        assert len(mags) == n_segs
        total_nnz = sum(len(m) for m in mags)
        expected_nnz = (pooled > 0).sum().item()
        assert total_nnz == expected_nnz

    def test_all_zero_pooling_then_sparse(self):
        """End-to-end: pool all-zero activations, then sparse extract."""
        sae_acts = t.zeros(8, 4)
        segments = [(0, 4), (4, 8)]
        for mode in [PoolingMode.max_pool, PoolingMode.mean_pool]:
            pooled = pool_segments_batched_gpu(
                sae_acts, segments, mode, SegmentMode.fixed_tokens
            )
            ids, mags = extract_sparse_gpu(pooled)
            for i in range(2):
                assert len(ids[i]) == 0
                assert len(mags[i]) == 0


# ---------------------------------------------------------------------------
# 6. Fixed-path vs turn-path equivalence
# ---------------------------------------------------------------------------


class TestFixedTurnPathEquivalence:
    """Verify fixed reshape and turn segmented reductions agree."""

    def test_max_pool_equivalence_uniform(self):
        sae_acts, segments = _hand_activations_8x3_uniform()
        fast = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        turn = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, TURN_MODE
        )
        assert t.equal(fast, turn), (
            f"max_pool fixed vs turn mismatch:\nfixed={fast}\nturn={turn}"
        )

    def test_mean_pool_equivalence_uniform(self):
        sae_acts, segments = _hand_activations_8x3_uniform()
        fast = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        turn = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, TURN_MODE
        )
        assert t.allclose(fast, turn, atol=1e-6), (
            f"mean_pool fixed vs turn mismatch:\nfixed={fast}\nturn={turn}"
        )

    def test_max_pool_equivalence_short_last(self):
        sae_acts, segments = _hand_activations_10x4()
        fast = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        turn = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, TURN_MODE
        )
        assert t.equal(fast, turn), (
            f"max_pool short-last fixed vs turn mismatch:\nfixed={fast}\nturn={turn}"
        )

    def test_mean_pool_equivalence_short_last(self):
        sae_acts, segments = _hand_activations_10x4()
        fast = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        turn = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, TURN_MODE
        )
        assert t.allclose(fast, turn, atol=1e-6), (
            f"mean_pool short-last fixed vs turn mismatch:\nfixed={fast}\nturn={turn}"
        )

    def test_equivalence_random_large(self):
        """Random 100-token, 16-feature tensor with seg_size=8, last segment short."""
        t.manual_seed(42)
        n_tokens = 100
        n_feat = 16
        seg_size = 8
        sae_acts = t.rand(n_tokens, n_feat)
        n_full = n_tokens // seg_size
        segments = [(i * seg_size, (i + 1) * seg_size) for i in range(n_full)]
        remainder = n_tokens % seg_size
        if remainder > 0:
            segments.append((n_full * seg_size, n_tokens))

        for mode in [PoolingMode.max_pool, PoolingMode.mean_pool]:
            fast = pool_segments_batched_gpu(
                sae_acts, segments, mode, SegmentMode.fixed_tokens
            )
            turn = pool_segments_batched_gpu(
                sae_acts, segments, mode, TURN_MODE
            )
            if mode == PoolingMode.max_pool:
                # Max selects an existing value on both routes, so tolerance
                # would hide a real divergence. Its siblings above are already
                # pinned with t.equal.
                assert t.equal(fast, turn), (
                    f"mode={mode.value}: fixed vs turn mismatch on random data"
                )
            else:
                assert t.allclose(fast, turn, atol=1e-6), (
                    f"mode={mode.value}: fixed vs turn mismatch on random data"
                )
