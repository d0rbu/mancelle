"""Slice 07 — S-07-02 P1: feature-percentile threshold inclusivity.

Map §6 S-07-02 + manifest §7 slice 7.

Post-fix: feature-percentile thresholding uses ``>=`` (inclusive). Rationale:
the percentile threshold IS the boundary value — for a percentile-derived
threshold, the boundary itself represents the score at the cutoff, and
excluding it adds an arbitrary off-by-one.

Tests below pin the post-fix INCLUSIVE behaviour. They will fail against the
pre-fix `>` semantic (red-line the bug).

Seed source: deterministic — these are pure-tensor tests with no RNG.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.thresholding import (
    apply_feature_percentile,
    apply_feature_percentile_segment_topk,
)


class TestApplyFeaturePercentileInclusive:
    """After the fix, exact-threshold magnitudes are KEPT."""

    def test_exact_threshold_value_kept(self):
        """Magnitude exactly at the per-feature threshold must be KEPT (>=)."""
        precomputed = t.tensor([0.5], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.6, 0.4], dtype=t.float32)
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, precomputed,
        )
        assert len(out_ids) == 2, (
            "expected exact-threshold value (0.5) AND above-threshold (0.6) to "
            "be kept under inclusive (>=) semantics"
        )
        kept = set(out_mags.tolist())
        assert any(abs(m - 0.5) < 1e-6 for m in kept)
        assert any(abs(m - 0.6) < 1e-6 for m in kept)

    def test_all_same_value_kept(self):
        """All-equal magnitudes at the threshold are KEPT (>=)."""
        precomputed = t.tensor([0.5], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.5, 0.5, 0.5], dtype=t.float64)
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, precomputed,
        )
        assert len(out_ids) == 4

    def test_below_threshold_excluded(self):
        """Below-threshold magnitudes are still excluded."""
        precomputed = t.tensor([0.5], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.4, 0.49, 0.499], dtype=t.float32)
        out_ids, _ = apply_feature_percentile(
            feature_ids, feature_mags, precomputed,
        )
        assert len(out_ids) == 0


class TestCombinedInclusivity:
    """apply_feature_percentile_segment_topk's stage-1 percentile filter
    must use the same inclusive semantic as the standalone helper.
    """

    def test_combined_keeps_threshold_boundary(self):
        precomputed = t.tensor([0.5, 0.5], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 1, 1], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.6, 0.5, 0.7], dtype=t.float32)
        out_ids, _ = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=10,
        )
        assert len(out_ids) == 4, (
            "all 4 features satisfy >= 0.5; stage-2 cap (k=10) leaves all"
        )


class TestPropertyInclusivity:
    """For random configs, post-filter is exactly {i : mags[i] >= thresholds[fids[i]]}."""

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_property_matches_inclusive_predicate(self, seed: int):
        rng = t.Generator().manual_seed(seed)
        n_feat = 8
        n_act = 32
        precomputed = t.rand(n_feat, generator=rng, dtype=t.float64)
        feature_ids = t.randint(0, n_feat, (n_act,), generator=rng, dtype=t.int64)
        feature_mags = t.rand(n_act, generator=rng, dtype=t.float32)
        # Force 1/3 of activations to land EXACTLY on their threshold.
        for i in range(0, n_act, 3):
            feature_mags[i] = float(precomputed[feature_ids[i]].item())

        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, precomputed,
        )
        # Re-compute oracle with INCLUSIVE semantic.
        oracle_mask = t.zeros(n_act, dtype=t.bool)
        for i in range(n_act):
            fid = int(feature_ids[i].item())
            mag = float(feature_mags[i].item())
            thr = float(precomputed[fid].item())
            oracle_mask[i] = mag >= thr  # INCLUSIVE
        expected_ids = feature_ids[oracle_mask].tolist()
        assert out_ids.tolist() == expected_ids
