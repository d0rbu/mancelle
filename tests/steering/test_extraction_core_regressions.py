"""Regression tests for extraction-core edge cases."""

from __future__ import annotations

import pytest
import torch as t

from steering.extract_features import (
    extract_sparse_gpu,
    pool_segments_batched_gpu,
)
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode


class TestExtractSparseGpuAllZeroNoAliasing:
    """The all-zero fast path must return independent tensors per segment."""

    def test_extract_sparse_gpu_all_zero_no_aliasing(self):
        n_segs = 4
        n_feat = 8
        pooled = t.zeros(n_segs, n_feat, dtype=t.float32)

        ids, mags = extract_sparse_gpu(pooled)
        assert len(ids) == n_segs
        assert len(mags) == n_segs

        for i in range(n_segs):
            for j in range(i + 1, n_segs):
                assert ids[i] is not ids[j]
                assert mags[i] is not mags[j]

        ids[0].resize_(3)
        ids[0][0] = 42
        ids[0][1] = 7
        ids[0][2] = 99
        for k in range(1, n_segs):
            assert ids[k].numel() == 0

        mags[1].resize_(2)
        mags[1][0] = 1.5
        mags[1][1] = 2.5
        for k in range(n_segs):
            if k == 1:
                continue
            assert mags[k].numel() == 0

    def test_extract_sparse_gpu_all_zero_still_emits_n_segs_entries(self):
        for n_segs in (1, 3, 10, 128):
            pooled = t.zeros(n_segs, 16, dtype=t.float32)
            ids, mags = extract_sparse_gpu(pooled)
            assert len(ids) == n_segs
            assert len(mags) == n_segs
            for i in range(n_segs):
                assert ids[i].numel() == 0
                assert mags[i].numel() == 0
                assert ids[i].dtype == t.int32
                assert mags[i].dtype == t.float32


class TestMaxPoolFastPathRejectsNegativeActivations:
    """Zero-padding in the fixed-token max-pool fast path requires nonnegative activations."""

    def test_max_pool_fast_path_rejects_negative_activations(self):
        sae_acts = t.full((10, 4), -1.0, dtype=t.float32)
        segments = [(0, 4), (4, 8), (8, 10)]

        with pytest.raises(AssertionError, match=r"(?i)non-?negative"):
            pool_segments_batched_gpu(
                sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
            )

    def test_max_pool_fast_path_accepts_non_negative(self):
        sae_acts = t.rand(10, 4, dtype=t.float32)
        segments = [(0, 4), (4, 8), (8, 10)]

        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        assert result.shape == (3, 4)
