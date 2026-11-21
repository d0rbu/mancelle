"""Tests for ``pool_fp_logits``: pool feature-prediction head outputs over a
span to produce a single prediction vector.

Used by self-conditioned generation: after K decoded tokens, pool the
head's per-token logits over the K positions to get one (n_sae,) prediction
that drives the encoder for the NEXT segment.
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits


class TestPoolFpLogitsLastToken:
    """``mode=last_token`` reads position -1 of the time dimension."""

    def test_basic_3d(self):
        # (B=1, T=4, n_sae=3) — pick T=-1 row
        fp_logits = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]
        )
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)
        assert pooled.shape == (1, 3)
        assert torch.allclose(pooled, torch.tensor([[1.0, 1.1, 1.2]]))

    def test_batch_dim_preserved(self):
        # (B=2, T=2, n_sae=2)
        fp_logits = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)
        assert pooled.shape == (2, 2)
        assert torch.allclose(pooled, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))

    def test_single_token_span(self):
        # T=1 — last_token == only_token
        fp_logits = torch.tensor([[[0.5, 0.7]]])
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)
        assert torch.allclose(pooled, torch.tensor([[0.5, 0.7]]))


class TestPoolFpLogitsFullSegment:
    """``mode=full_segment`` averages over the time dimension."""

    def test_mean_over_time(self):
        fp_logits = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]]
        )
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.FULL_SEGMENT)
        assert pooled.shape == (1, 2)
        # column 0: (0+2+4+6)/4 = 3.0; column 1: (1+3+5+7)/4 = 4.0
        assert torch.allclose(pooled, torch.tensor([[3.0, 4.0]]))

    def test_single_token_mean_equals_value(self):
        fp_logits = torch.tensor([[[1.0, 2.0]]])
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.FULL_SEGMENT)
        assert torch.allclose(pooled, torch.tensor([[1.0, 2.0]]))


class TestPoolFpLogitsMaxPool:
    """``mode=max_pool`` takes per-feature max across the time dimension."""

    def test_per_feature_max(self):
        # (B=1, T=3, n_sae=4) — column-wise max
        fp_logits = torch.tensor(
            [[[1.0, 5.0, 0.0, -1.0],
              [3.0, 2.0, 9.0, -2.0],
              [4.0, 1.0, 8.0, -3.0]]]
        )
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.MAX_POOL)
        assert pooled.shape == (1, 4)
        assert torch.equal(pooled, torch.tensor([[4.0, 5.0, 9.0, -1.0]]))

    def test_single_token_equals_value(self):
        fp_logits = torch.tensor([[[0.5, 0.7]]])
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.MAX_POOL)
        assert torch.allclose(pooled, torch.tensor([[0.5, 0.7]]))

    def test_batch_dim_preserved(self):
        fp_logits = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]],
             [[5.0, 6.0], [-1.0, -2.0]]]
        )
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.MAX_POOL)
        assert pooled.shape == (2, 2)
        assert torch.equal(pooled, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))


class TestPoolFpLogitsBestTokenForbidden:
    """``mode=best_token`` is forbidden at inference: it needs future targets
    to compute the per-token loss whose argmin selects the 'best' token."""

    def test_raises(self):
        fp_logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
            pool_fp_logits(fp_logits, FeaturePredLossMode.BEST_TOKEN)


class TestPoolFpLogitsContract:
    """Contract pins for callers."""

    def test_dtype_preserved(self):
        fp_logits = torch.tensor([[[0.1, 0.2]]], dtype=torch.float32)
        pooled = pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)
        assert pooled.dtype == torch.float32

    def test_2d_input_rejected(self):
        # The contract is (B, T, n_sae). 2D input is ambiguous.
        fp_logits = torch.tensor([[0.1, 0.2]])
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)

    def test_empty_time_rejected(self):
        # T=0 has no token to pool from.
        fp_logits = torch.empty((1, 0, 4))
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            pool_fp_logits(fp_logits, FeaturePredLossMode.LAST_TOKEN)
