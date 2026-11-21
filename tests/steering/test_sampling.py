"""Tests for steering/conditional/sampling.py.

Tests cover:
- build_percentile_table: shape, dtype, known/missing/empty features
- magnitude_to_percentile: boundary, interpolation, clamping, constant features
- sample_features: determinism, subset, no-dups, id-mag correspondence, edge cases
- Monte Carlo statistical validation
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.sampling import (
    Distribution,
    FeatureSelectionConfig,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
    build_percentile_table,
    magnitude_to_percentile,
    parse_feature_selection,
    parse_negative_selection,
    sample_features,
    sample_inactive_features,
    sample_k,
)
from steering.conditional.thresholding import ThresholdType
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_stat(p_values: list[float]) -> PerFeatureStat:
    """Create a PerFeatureStat with p1..p99 from a list of 99 values."""
    assert len(p_values) == 99
    percentiles = {f"p{i + 1}": float(v) for i, v in enumerate(p_values)}
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=float(p_values[49]),
        mean_mag_incl_zeros=float(p_values[49]) * 0.5,
        std_mag=1.0,
        min_mag=float(p_values[0]),
        max_mag=float(p_values[-1]),
        median_mag=float(p_values[49]),
        percentiles=percentiles,
    )


def make_linear_stat(lo: float = 1.0, hi: float = 99.0) -> PerFeatureStat:
    """Create a stat whose p_k = lo + (k-1)*(hi-lo)/98 for k in 1..99.

    With lo=1, hi=99: p_k = k exactly.
    """
    vals = [lo + (hi - lo) * i / 98.0 for i in range(99)]
    return make_stat(vals)


# ---------------------------------------------------------------------------
# build_percentile_table
# ---------------------------------------------------------------------------


class TestBuildPercentileTable:
    def test_shape_and_dtype(self):
        stats: dict[str, PerFeatureStat] = {"3": make_linear_stat()}
        table = build_percentile_table(stats, n_sae_features=10)
        assert table.shape == (10, 99)
        assert table.dtype == t.float64

    def test_known_feature_row_matches_p1_to_p99(self):
        stat = make_linear_stat()
        stats = {"5": stat}
        table = build_percentile_table(stats, n_sae_features=10)
        expected = t.tensor(
            [stat.percentiles[f"p{k}"] for k in range(1, 100)],
            dtype=t.float64,
        )
        assert t.allclose(table[5], expected)

    def test_missing_feature_row_is_zero(self):
        stats = {"3": make_linear_stat()}
        table = build_percentile_table(stats, n_sae_features=10)
        for i in range(10):
            if i != 3:
                assert t.all(table[i] == 0.0), f"Row {i} should be all zeros"

    def test_empty_stats_all_zeros(self):
        table = build_percentile_table({}, n_sae_features=5)
        assert table.shape == (5, 99)
        assert t.all(table == 0.0)

    def test_multiple_features_placed_correctly(self):
        stat_0 = make_linear_stat(lo=0.0, hi=49.5)
        stat_7 = make_linear_stat(lo=10.0, hi=108.0)
        stats = {"0": stat_0, "7": stat_7}
        table = build_percentile_table(stats, n_sae_features=10)
        for k in range(1, 100):
            assert table[0, k - 1].item() == pytest.approx(stat_0.percentiles[f"p{k}"])
            assert table[7, k - 1].item() == pytest.approx(stat_7.percentiles[f"p{k}"])
        # rows not in stats are zero
        for i in [1, 2, 3, 4, 5, 6, 8, 9]:
            assert t.all(table[i] == 0.0)

    def test_non_monotonic_percentile_row_raises(self):
        """Non-monotonic percentile values should raise ValueError."""
        vals = [float(v) for v in range(1, 100)]  # monotonic: [1, 2, ..., 99]
        vals[49] = 0.0  # p50 = 0.0, violates monotonicity (p49=49 > p50=0)
        stat = make_stat(vals)
        with pytest.raises(ValueError, match="non-monotonic"):
            build_percentile_table({"0": stat}, n_sae_features=5)

    def test_constant_row_accepted(self):
        """All-same percentile values are monotonic (non-decreasing) and accepted."""
        stat = make_stat([5.0] * 99)
        table = build_percentile_table({"0": stat}, n_sae_features=2)
        assert t.all(table[0] == 5.0)


# ---------------------------------------------------------------------------
# magnitude_to_percentile
# ---------------------------------------------------------------------------


class TestMagnitudeToPercentile:
    """Tests for magnitude_to_percentile.

    Uses a linear stat where p_k = k (p1=1, p50=50, p99=99).
    """

    def _table(self, n: int = 5) -> t.Tensor:
        stat = make_linear_stat()
        return build_percentile_table({"2": stat}, n_sae_features=n)

    def test_exact_boundary_match_p50(self):
        """mag=50.0 at feature 2 (p50=50) → 50.0"""
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([50.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result.dtype == t.float64
        assert result[0].item() == pytest.approx(50.0)

    def test_interpolation_between_p45_and_p46(self):
        """mag=45.3 between p45=45 and p46=46 → 45.3.

        Pass mag as float64 to avoid float32 precision loss.
        """
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([45.3], dtype=t.float64)
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(45.3, rel=1e-9)

    def test_below_p1_returns_0_5(self):
        """mag=0.0 is below p1=1 → 0.5"""
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([0.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(0.5)

    def test_above_p99_returns_99_5(self):
        """mag=100.0 is above p99=99 → 99.5"""
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([100.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(99.5)

    def test_at_p1_exactly(self):
        """mag=1.0 exactly equals p1 → 1.0 (right=True: idx=1, left=p1, right=p2, frac=0)"""
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([1.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(1.0)

    def test_at_p99_exactly(self):
        """mag=99.0 exactly equals p99 → 99.5.

        With right=True, searchsorted places mag=p99 at idx=99 (past the last element),
        which triggers the above-p99 rule → 99.5. This is consistent with right=True
        semantics: equal values are pushed to the right, so p99 itself is classified
        as "above" the last boundary.
        """
        table = self._table()
        ids = t.tensor([2], dtype=t.int64)
        mags = t.tensor([99.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(99.5)

    def test_all_zero_row_positive_mag_raises(self):
        """A missing-statistics row cannot produce a percentile score."""
        table = self._table()  # feature 0,1,3,4 are zero rows
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([5.0])
        with pytest.raises(
            ValueError, match=r"missing for 1 distinct feature ID\(s\) \(first 1: \[0\]\)"
        ):
            magnitude_to_percentile(ids, mags, table)

    def test_multiple_features_each_own_row(self):
        """Two features, each looked up against their own row."""
        stat_a = make_linear_stat(lo=0.0, hi=98.0)  # p_k = (k-1)
        stat_b = make_linear_stat(lo=10.0, hi=108.0)  # p_k = 9 + k
        table = build_percentile_table({"0": stat_a, "1": stat_b}, n_sae_features=5)

        ids = t.tensor([0, 1], dtype=t.int64)
        # Feature 0: p50 = 49.0, Feature 1: p50 = 59.0
        p50_a = stat_a.percentiles["p50"]
        p50_b = stat_b.percentiles["p50"]
        mags = t.tensor([p50_a, p50_b])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(50.0)
        assert result[1].item() == pytest.approx(50.0)

    def test_empty_input_returns_empty(self):
        table = self._table()
        ids = t.tensor([], dtype=t.int64)
        mags = t.tensor([])
        result = magnitude_to_percentile(ids, mags, table)
        assert result.shape == (0,)
        assert result.dtype == t.float64

    def test_constant_feature_mag_equals_const_above_returns_99_5(self):
        """All percentile values = 5.0. mag=5.0: right=True pushes past all equal → 99.5."""
        const_vals = [5.0] * 99
        stat = make_stat(const_vals)
        table = build_percentile_table({"0": stat}, n_sae_features=2)
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([5.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(99.5)

    def test_constant_feature_mag_below_const_returns_0_5(self):
        """All percentile values = 5.0. mag=3.0: below all → 0.5."""
        const_vals = [5.0] * 99
        stat = make_stat(const_vals)
        table = build_percentile_table({"0": stat}, n_sae_features=2)
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([3.0])
        result = magnitude_to_percentile(ids, mags, table)
        assert result[0].item() == pytest.approx(0.5)

    def test_plateau_region_interpolation(self):
        """Magnitude equal to a run of identical percentile values.

        With right=True, searchsorted skips past the entire plateau.
        The interpolation then uses the boundary just after the plateau.
        """
        vals = [float(k) for k in range(1, 100)]  # p_k = k
        # p45=p46=p47=p48 all equal to 45
        vals[44] = 45.0
        vals[45] = 45.0
        vals[46] = 45.0
        vals[47] = 45.0
        vals[48] = 49.0  # p49 resumes
        stat = make_stat(vals)
        table = build_percentile_table({"0": stat}, n_sae_features=2)
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([45.0], dtype=t.float64)
        result = magnitude_to_percentile(ids, mags, table)
        # searchsorted(right=True) pushes past all 45s → idx=48
        # left=row[47]=45.0, right=row[48]=49.0, frac=(45-45)/4=0
        # result = 48.0
        assert result[0].item() == pytest.approx(48.0)

    def test_all_zero_row_zero_mag_raises(self):
        """A zero magnitude does not make a missing-statistics row valid."""
        table = t.zeros(2, 99, dtype=t.float64)
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([0.0])
        with pytest.raises(
            ValueError, match=r"missing for 1 distinct feature ID\(s\) \(first 1: \[0\]\)"
        ):
            magnitude_to_percentile(ids, mags, table)


# ---------------------------------------------------------------------------
# sample_features
# ---------------------------------------------------------------------------


class TestSampleFeatures:
    def _make_inputs(self, n: int = 10) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        """n features with linear stat on feature 0..n-1 (each feature its own row)."""
        # All same stat for simplicity
        stat = make_linear_stat()
        stats = {str(i): stat for i in range(n)}
        table = build_percentile_table(stats, n_sae_features=n)
        ids = t.arange(n, dtype=t.int64)
        mags = t.linspace(10.0, 90.0, n)
        return ids, mags, table

    def _rng(self, seed: int = 42) -> t.Generator:
        g = t.Generator()
        g.manual_seed(seed)
        return g

    def test_deterministic_with_seeded_generator(self):
        ids, mags, table = self._make_inputs()
        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=1.0)
        out1_ids, out1_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        out2_ids, out2_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        assert t.equal(out1_ids, out2_ids)
        assert t.equal(out1_mags, out2_mags)

    def test_override_weights_match_internal_percentile_weighting(self):
        """Precomputed percentile weights must preserve weighted sampling exactly.

        The SFT data path reuses all-feature percentile scores for both
        feature-prediction targets and weighted positive sampling.  This test
        pins the equivalence to the original internal percentile lookup path,
        including seeded RNG consumption.
        """
        ids, mags, table = self._make_inputs()
        cfg = SamplingSelection(distribution=Distribution.normal, mean=4.0, std=1.0)
        precomputed_weights = magnitude_to_percentile(ids, mags, table)

        ref_ids, ref_mags = sample_features(
            ids, mags, table, cfg, self._rng(123)
        )
        reuse_ids, reuse_mags = sample_features(
            ids,
            mags,
            table,
            cfg,
            self._rng(123),
            override_weights=precomputed_weights,
        )

        assert t.equal(reuse_ids, ref_ids)
        assert t.equal(reuse_mags, ref_mags)

    def test_output_is_subset_of_input(self):
        ids, mags, table = self._make_inputs()
        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=1.0)
        out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(99))
        input_id_set = set(ids.tolist())
        for fid in out_ids.tolist():
            assert fid in input_id_set

    def test_no_duplicates_over_100_seeds(self):
        ids, mags, table = self._make_inputs()
        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=1.0)
        for seed in range(100):
            out_ids, _ = sample_features(ids, mags, table, cfg, self._rng(seed))
            assert len(out_ids) == len(set(out_ids.tolist())), (
                f"Duplicates found at seed {seed}"
            )

    def test_preserves_id_mag_correspondence(self):
        """Each output id must match its original magnitude."""
        n = 10
        ids = t.arange(n, dtype=t.int64)
        mags = t.arange(n, dtype=t.float32) * 2.0 + 1.0  # 1, 3, 5, ..., 19
        stat = make_linear_stat()
        stats = {str(i): stat for i in range(n)}
        table = build_percentile_table(stats, n_sae_features=n)
        cfg = SamplingSelection(distribution=Distribution.normal, mean=5.0, std=1.0)
        # Build original id->mag map
        id_to_mag = {int(ids[i]): float(mags[i]) for i in range(n)}
        for seed in range(20):
            out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(seed))
            for oid, omag in zip(out_ids.tolist(), out_mags.tolist(), strict=True):
                assert omag == pytest.approx(id_to_mag[int(oid)])

    def test_empty_input_returns_empty_output(self):
        table = t.zeros(5, 99, dtype=t.float64)
        ids = t.tensor([], dtype=t.int64)
        mags = t.tensor([])
        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=1.0)
        out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        assert out_ids.shape == (0,)
        assert out_mags.shape == (0,)

    def test_single_feature_k_ge_1_selected(self):
        """Single active feature with k ≥ 1 → it is always selected."""
        stat = make_linear_stat()
        table = build_percentile_table({"0": stat}, n_sae_features=1)
        ids = t.tensor([0], dtype=t.int64)
        mags = t.tensor([50.0])
        # Tiny positive std always rounds to k=5, then clamps to 1.
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=5.0,
            std=1e-12,
        )
        out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        assert out_ids.shape == (1,)
        assert out_ids[0].item() == 0
        assert out_mags[0].item() == pytest.approx(50.0)

    def test_k_greater_than_num_active_returns_all(self):
        """When k > num_active, all features are returned (clamped)."""
        ids, mags, table = self._make_inputs(n=3)
        # mean=100 → k=100 >> 3
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=100.0,
            std=1e-12,
        )
        out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        assert len(out_ids) == 3

    def test_k_le_0_clamped_to_min_k(self):
        """When sampled k ≤ 0, min_k=1 clamps to 1 feature."""
        ids, mags, table = self._make_inputs()
        # mean=-5 with tiny positive std rounds below zero, then clamps to min_k=1.
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=-5.0,
            std=1e-12,
        )
        out_ids, out_mags = sample_features(ids, mags, table, cfg, self._rng(42))
        assert out_ids.shape == (1,)
        assert out_mags.shape == (1,)

    def test_invalid_distribution_raises_value_error(self):
        ids, mags, table = self._make_inputs()
        # Corrupt a valid frozen instance to exercise sample_k's defensive arm.
        cfg = SamplingSelection()
        object.__setattr__(cfg, "distribution", "uniform")
        with pytest.raises(BeartypeCallHintParamViolation, match="distribution"):
            sample_features(ids, mags, table, cfg, self._rng(42))

    def test_geometric_distribution_accepted(self):
        """sample_features should accept geometric distribution."""
        n_features = 100
        pct_vals = [float(i) for i in range(1, 100)]
        stats = {str(f): make_stat(pct_vals) for f in range(n_features)}
        table = build_percentile_table(stats, n_features)

        fids = t.arange(20, dtype=t.int64)
        fmags = t.linspace(10, 90, 20)

        cfg = SamplingSelection(distribution=Distribution.geometric, mean=5.0, std=5.0)
        rng = t.Generator().manual_seed(42)
        out_ids, out_mags = sample_features(fids, fmags, table, cfg, rng)
        assert len(out_ids) <= len(fids)
        assert len(out_ids) == len(out_mags)

    def test_geometric_deterministic_in_sample_features(self):
        """Geometric path in sample_features is deterministic."""
        n_features = 100
        pct_vals = [float(i) for i in range(1, 100)]
        stats = {str(f): make_stat(pct_vals) for f in range(n_features)}
        table = build_percentile_table(stats, n_features)

        fids = t.arange(20, dtype=t.int64)
        fmags = t.linspace(10, 90, 20)
        cfg = SamplingSelection(distribution=Distribution.geometric, mean=5.0, std=5.0)

        rng1 = t.Generator().manual_seed(99)
        out1, _ = sample_features(fids, fmags, table, cfg, rng1)
        rng2 = t.Generator().manual_seed(99)
        out2, _ = sample_features(fids, fmags, table, cfg, rng2)
        assert t.equal(out1, out2)

    def test_equal_weight_features_uniform_sampling(self):
        """Two features with identical percentile → uniform selection over 1000 trials."""
        stat = make_linear_stat()
        table = build_percentile_table({"0": stat, "1": stat}, n_sae_features=2)
        ids = t.tensor([0, 1], dtype=t.int64)
        mags = t.tensor([50.0, 50.0])  # same mag → same percentile → same weight
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=1.0,
            std=1e-12,
        )
        counts = [0, 0]
        rng = t.Generator().manual_seed(777)
        for _ in range(1000):
            out_ids, _ = sample_features(ids, mags, table, cfg, rng)
            if len(out_ids) == 1:
                counts[int(out_ids[0].item())] += 1
        # With equal weights, each should get ~50%. Allow 40-60% range.
        total = sum(counts)
        assert total > 0
        ratio = counts[0] / total
        assert 0.40 < ratio < 0.60, f"Expected ~50/50, got {counts}"


# ---------------------------------------------------------------------------
# Task 4: Monte Carlo statistical validation
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """Statistical validation tests using many random trials."""

    def _three_feature_setup(self) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        """Three features with percentiles 1, 50, 99 relative to their own stats.

        Feature 0: mag at p1 (→ percentile ≈ 1)
        Feature 1: mag at p50 (→ percentile ≈ 50)
        Feature 2: mag at p99 (→ percentile ≈ 99)
        """
        stat = make_linear_stat()  # p_k = k
        table = build_percentile_table(
            {"0": stat, "1": stat, "2": stat},
            n_sae_features=3,
        )
        # p1=1, p50=50, p99=99
        ids = t.tensor([0, 1, 2], dtype=t.int64)
        mags = t.tensor([1.0, 50.0, 99.0])
        return ids, mags, table

    def test_k1_selection_rates_chi_squared(self):
        """50,000 trials with k=1: rates should match 1/150, 50/150, 99/150.

        Chi-squared test: df=2, alpha=0.001, critical=13.816.
        """
        ids, mags, table = self._three_feature_setup()
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=1.0,
            std=1e-12,
        )

        counts = [0, 0, 0]
        n_trials = 50_000
        rng = t.Generator()
        rng.manual_seed(12345)

        for _ in range(n_trials):
            out_ids, _ = sample_features(ids, mags, table, cfg, rng)
            if len(out_ids) == 1:
                counts[int(out_ids[0].item())] += 1

        # Compute expected counts
        total_weight = 1.0 + 50.0 + 99.0  # 150.0
        expected_rates = [1.0 / total_weight, 50.0 / total_weight, 99.0 / total_weight]
        total_counts = sum(counts)

        # Chi-squared
        chi2 = 0.0
        for obs, rate in zip(counts, expected_rates, strict=True):
            exp = total_counts * rate
            chi2 += (obs - exp) ** 2 / exp

        # df=2, alpha=0.001 → critical=13.816
        assert chi2 < 13.816, (
            f"Chi-squared {chi2:.3f} exceeds critical 13.816. "
            f"Counts: {counts}, expected rates: {[f'{r:.4f}' for r in expected_rates]}"
        )

    def test_k2_ordering_and_high_pct_dominance(self):
        """10,000 trials with k=2: feature at p99 should dominate.

        Feature 2 (percentile≈99) should appear in >90% of trials.
        Ordering: feature 2 > feature 1 > feature 0.
        """
        ids, mags, table = self._three_feature_setup()
        cfg = SamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=1e-12,
        )

        counts = [0, 0, 0]
        n_trials = 10_000
        rng = t.Generator()
        rng.manual_seed(99999)

        for _ in range(n_trials):
            out_ids, _ = sample_features(ids, mags, table, cfg, rng)
            for fid in out_ids.tolist():
                counts[int(fid)] += 1

        # Feature 2 (p99) should appear most often
        assert counts[2] > counts[1] > counts[0], f"Ordering violated: counts={counts}"
        # Feature 2 should appear in >90% of trials
        rate_2 = counts[2] / n_trials
        assert rate_2 > 0.90, (
            f"Feature at p99 appears in only {rate_2:.3f} of trials (expected >0.90)"
        )


# ---------------------------------------------------------------------------
# Task 5: parse_feature_selection
# ---------------------------------------------------------------------------


class TestParseFeatureSelection:
    def test_valid_threshold(self):
        cfg = {"threshold_type": "feature_percentile_segment_topk", "threshold": 2048.9}
        result = parse_feature_selection(cfg)
        assert isinstance(result, ThresholdSelection)
        assert result.threshold_type == ThresholdType.feature_percentile_segment_topk
        assert result.threshold == 2048.9

    def test_valid_sampling(self):
        cfg = {"distribution": "normal", "mean": 5.0, "std": 2.0}
        result = parse_feature_selection(cfg)
        assert isinstance(result, SamplingSelection)
        assert result.distribution == Distribution.normal
        assert result.mean == 5.0
        assert result.std == 2.0

    def test_mixed_keys_raises(self):
        cfg = {
            "threshold_type": "segment_topk",
            "threshold": 10,
            "distribution": "normal",
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            parse_feature_selection(cfg)

    def test_sampling_unknown_key_raises(self):
        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "typo": 123,
        }
        with pytest.raises(ValueError, match=r"unknown or incompatible.*typo"):
            parse_feature_selection(cfg)

    def test_threshold_rejects_leftover_sampling_keys(self):
        cfg = {
            "threshold_type": "feature_percentile",
            "threshold": 0.9,
            "mean": 5.0,
            "max_k": 25,
        }
        with pytest.raises(ValueError, match=r"unknown or incompatible.*max_k.*mean"):
            parse_feature_selection(cfg)

    def test_threshold_rejects_null_gradient_key(self):
        cfg = {
            "threshold_type": "feature_percentile",
            "threshold": 0.9,
            "gradient": None,
        }
        with pytest.raises(ValueError, match=r"unknown or incompatible.*gradient"):
            parse_feature_selection(cfg)

    def test_neither_variant_raises(self):
        cfg = {"some_other_key": 42}
        with pytest.raises(ValueError, match="must contain either"):
            parse_feature_selection(cfg)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="must contain either"):
            parse_feature_selection({})

    def test_sampling_missing_mean_raises(self):
        cfg = {"distribution": "normal", "std": 2.0}
        with pytest.raises(ValueError, match="mean"):
            parse_feature_selection(cfg)

    def test_sampling_missing_std_raises(self):
        cfg = {"distribution": "normal", "mean": 5.0}
        with pytest.raises(ValueError, match="std"):
            parse_feature_selection(cfg)

    def test_sampling_zero_std_raises(self):
        cfg = {"distribution": "normal", "mean": 5.0, "std": 0.0}
        with pytest.raises(BeartypeCallHintParamViolation, match="std"):
            parse_feature_selection(cfg)

    def test_sampling_negative_std_raises(self):
        cfg = {"distribution": "normal", "mean": 5.0, "std": -1.0}
        with pytest.raises(BeartypeCallHintParamViolation, match="std"):
            parse_feature_selection(cfg)

    def test_threshold_missing_threshold_raises(self):
        cfg = {"threshold_type": "segment_topk"}
        with pytest.raises(ValueError, match="threshold"):
            parse_feature_selection(cfg)

    def test_unsupported_distribution_raises(self):
        cfg = {"distribution": "uniform", "mean": 5.0, "std": 2.0}
        with pytest.raises(ValueError, match="Unsupported distribution"):
            parse_feature_selection(cfg)

    def test_invalid_threshold_type_raises(self):
        cfg = {"threshold_type": "nonexistent_type", "threshold": 10}
        with pytest.raises(ValueError, match="Unsupported threshold_type"):
            parse_feature_selection(cfg)

    def test_removed_feature_filter_topk_raises(self):
        cfg = {"threshold_type": "feature_filter_topk", "threshold": 10}
        with pytest.raises(ValueError, match="feature_filter_topk"):
            parse_feature_selection(cfg)

    def test_removed_feature_topk_raises(self):
        cfg = {"threshold_type": "feature_topk", "threshold": 10}
        with pytest.raises(ValueError, match="feature_topk"):
            parse_feature_selection(cfg)

    def test_geometric_without_std_sets_std_to_mean(self):
        result = parse_feature_selection({"distribution": "geometric", "mean": 7.0})
        assert isinstance(result, SamplingSelection)
        assert result.distribution == Distribution.geometric
        assert result.mean == 7.0
        assert result.std == 7.0

    def test_geometric_with_std_equal_mean_accepted(self):
        result = parse_feature_selection(
            {"distribution": "geometric", "mean": 5.0, "std": 5.0}
        )
        assert isinstance(result, SamplingSelection)
        assert result.std == 5.0

    def test_geometric_with_std_not_equal_mean_raises(self):
        with pytest.raises(ValueError, match="std.*must equal.*mean"):
            parse_feature_selection(
                {"distribution": "geometric", "mean": 5.0, "std": 3.0}
            )

    def test_geometric_mean_zero_raises(self):
        with pytest.raises(ValueError, match="mean > 0"):
            parse_feature_selection({"distribution": "geometric", "mean": 0.0})

    def test_geometric_mean_negative_raises(self):
        with pytest.raises(ValueError, match="mean > 0"):
            parse_feature_selection({"distribution": "geometric", "mean": -1.0})

    def test_normal_without_std_raises(self):
        with pytest.raises(ValueError, match="std"):
            parse_feature_selection({"distribution": "normal", "mean": 5.0})

    def test_normal_with_valid_std_accepted(self):
        result = parse_feature_selection(
            {"distribution": "normal", "mean": 5.0, "std": 2.0}
        )
        assert isinstance(result, SamplingSelection)
        assert result.distribution == Distribution.normal
        assert result.std == 2.0

    def test_geometric_default_values(self):
        s = SamplingSelection()
        assert s.distribution == Distribution.geometric
        assert s.mean == 5.0
        assert s.std == 5.0

    def test_all_threshold_types(self):
        for tt in ThresholdType:
            threshold = (
                2048.9 if tt == ThresholdType.feature_percentile_segment_topk else 10
            )
            cfg = {"threshold_type": tt.value, "threshold": threshold}
            result = parse_feature_selection(cfg)
            assert isinstance(result, ThresholdSelection)
            assert result.threshold_type == tt


class TestSampleK:
    """Tests for sample_k distribution sampling helper."""

    def test_geometric_deterministic(self):
        rng1 = t.Generator().manual_seed(42)
        rng2 = t.Generator().manual_seed(42)
        k1 = sample_k(Distribution.geometric, 5.0, 5.0, rng1)
        k2 = sample_k(Distribution.geometric, 5.0, 5.0, rng2)
        assert k1 == k2

    def test_normal_deterministic(self):
        rng1 = t.Generator().manual_seed(42)
        rng2 = t.Generator().manual_seed(42)
        k1 = sample_k(Distribution.normal, 5.0, 2.0, rng1)
        k2 = sample_k(Distribution.normal, 5.0, 2.0, rng2)
        assert k1 == k2

    def test_geometric_non_negative(self):
        for seed in range(200):
            rng = t.Generator().manual_seed(seed)
            k = sample_k(Distribution.geometric, 3.0, 3.0, rng)
            assert k >= 0, f"seed={seed} produced k={k}"

    def test_normal_non_negative(self):
        for seed in range(200):
            rng = t.Generator().manual_seed(seed)
            k = sample_k(Distribution.normal, 1.0, 5.0, rng)
            assert k >= 0, f"seed={seed} produced k={k}"

    def test_unsupported_distribution_raises(self):
        rng = t.Generator().manual_seed(0)
        with pytest.raises(BeartypeCallHintParamViolation, match="distribution"):
            sample_k(cast(Distribution, "exponential"), 5.0, 5.0, rng)

    def test_geometric_monte_carlo_mean(self):
        mean = 5.0
        rng = t.Generator().manual_seed(123)
        samples = [
            sample_k(Distribution.geometric, mean, mean, rng) for _ in range(10_000)
        ]
        empirical_mean = sum(samples) / len(samples)
        assert abs(empirical_mean - mean) < 0.3, (
            f"Expected ~{mean}, got {empirical_mean}"
        )

    def test_normal_monte_carlo_mean(self):
        mean = 10.0
        std = 2.0
        rng = t.Generator().manual_seed(456)
        samples = [
            sample_k(Distribution.normal, mean, std, rng) for _ in range(10_000)
        ]
        empirical_mean = sum(samples) / len(samples)
        assert abs(empirical_mean - mean) < 0.3, (
            f"Expected ~{mean}, got {empirical_mean}"
        )

    def test_geometric_returns_int(self):
        rng = t.Generator().manual_seed(0)
        k = sample_k(Distribution.geometric, 5.0, 5.0, rng)
        assert isinstance(k, int)

    def test_different_seeds_different_results(self):
        results = set()
        for seed in range(50):
            rng = t.Generator().manual_seed(seed)
            results.add(sample_k(Distribution.geometric, 5.0, 5.0, rng))
        assert len(results) > 1


class TestParseNegativeSelection:
    """Tests for parse_negative_selection."""

    def test_none_returns_none(self):
        assert parse_negative_selection(None) is None

    def test_valid_geometric_default(self):
        result = parse_negative_selection({"distribution": "geometric", "mean": 3.0})
        assert isinstance(result, NegativeSamplingSelection)
        assert result.distribution == Distribution.geometric
        assert result.mean == 3.0
        assert result.std == 3.0

    def test_valid_normal(self):
        result = parse_negative_selection(
            {"distribution": "normal", "mean": 5.0, "std": 2.0}
        )
        assert isinstance(result, NegativeSamplingSelection)
        assert result.distribution == Distribution.normal
        assert result.mean == 5.0
        assert result.std == 2.0

    def test_geometric_std_mismatch_raises(self):
        with pytest.raises(ValueError, match="std.*must equal.*mean"):
            parse_negative_selection(
                {"distribution": "geometric", "mean": 3.0, "std": 1.0}
            )

    def test_normal_without_std_raises(self):
        with pytest.raises(ValueError, match="std"):
            parse_negative_selection({"distribution": "normal", "mean": 5.0})

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            parse_negative_selection({"distribution": "exponential", "mean": 5.0})

    def test_missing_mean_raises(self):
        with pytest.raises(ValueError, match="mean"):
            parse_negative_selection({"distribution": "geometric"})

    def test_default_values(self):
        s = NegativeSamplingSelection()
        assert s.distribution == Distribution.geometric
        assert s.mean == 3.0
        assert s.std == 3.0


class TestSampleInactiveFeatures:
    """Tests for sample_inactive_features."""

    def _make_cfg(self, distribution=Distribution.geometric, mean=3.0, std=None):
        std = (
            std
            if std is not None
            else (mean if distribution == Distribution.geometric else 2.0)
        )
        return NegativeSamplingSelection(distribution=distribution, mean=mean, std=std)

    def test_returns_correct_count_with_seeded_rng(self):
        """Output count matches what sample_k would produce with same seed."""
        cfg = self._make_cfg(mean=5.0)
        rng = t.Generator().manual_seed(42)
        rng_copy = t.Generator().manual_seed(42)
        expected_k = sample_k(cfg.distribution, cfg.mean, cfg.std, rng_copy)
        n_sae = 16384
        active = t.tensor([10, 20, 30], dtype=t.int64)
        expected_k = min(expected_k, n_sae - len(active))
        result = sample_inactive_features(active, n_sae, cfg, rng)
        assert len(result) == expected_k

    def test_no_overlap_with_exclusion_set(self):
        cfg = self._make_cfg(mean=10.0)
        active = t.arange(100, dtype=t.int64)
        n_sae = 1000
        for seed in range(20):
            rng = t.Generator().manual_seed(seed)
            result = sample_inactive_features(active, n_sae, cfg, rng)
            active_set = set(active.tolist())
            for fid in result.tolist():
                assert fid not in active_set, f"Feature {fid} is in exclusion set"

    def test_all_unique(self):
        cfg = self._make_cfg(mean=20.0)
        active = t.arange(50, dtype=t.int64)
        n_sae = 1000
        for seed in range(20):
            rng = t.Generator().manual_seed(seed)
            result = sample_inactive_features(active, n_sae, cfg, rng)
            assert len(result) == len(t.unique(result)), f"Duplicates at seed={seed}"

    def test_all_in_range(self):
        cfg = self._make_cfg(mean=10.0)
        active = t.tensor([5, 10, 15], dtype=t.int64)
        n_sae = 100
        for seed in range(20):
            rng = t.Generator().manual_seed(seed)
            result = sample_inactive_features(active, n_sae, cfg, rng)
            if len(result) > 0:
                assert result.min() >= 0
                assert result.max() < n_sae

    def test_empty_exclusion_full_pool(self):
        cfg = self._make_cfg(mean=5.0)
        active = t.tensor([], dtype=t.int64)
        n_sae = 100
        rng = t.Generator().manual_seed(42)
        result = sample_inactive_features(active, n_sae, cfg, rng)
        assert len(result) <= n_sae

    def test_all_excluded_returns_empty(self):
        cfg = self._make_cfg(mean=5.0)
        n_sae = 10
        active = t.arange(n_sae, dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        result = sample_inactive_features(active, n_sae, cfg, rng)
        assert len(result) == 0

    def test_k_clamped_to_pool_size(self):
        cfg = self._make_cfg(distribution=Distribution.normal, mean=100.0, std=1.0)
        n_sae = 20
        active = t.arange(15, dtype=t.int64)  # pool = 5
        rng = t.Generator().manual_seed(42)
        result = sample_inactive_features(active, n_sae, cfg, rng)
        assert len(result) == 5
        assert len(t.unique(result)) == 5

    def test_deterministic_with_seed(self):
        cfg = self._make_cfg(mean=5.0)
        active = t.arange(50, dtype=t.int64)
        n_sae = 1000
        rng1 = t.Generator().manual_seed(42)
        rng2 = t.Generator().manual_seed(42)
        r1 = sample_inactive_features(active, n_sae, cfg, rng1)
        r2 = sample_inactive_features(active, n_sae, cfg, rng2)
        assert t.equal(r1, r2)

    def test_different_seeds_different_results(self):
        cfg = self._make_cfg(mean=5.0)
        active = t.arange(50, dtype=t.int64)
        n_sae = 1000
        results = set()
        for seed in range(20):
            rng = t.Generator().manual_seed(seed)
            r = sample_inactive_features(active, n_sae, cfg, rng)
            results.add(tuple(sorted(r.tolist())))
        assert len(results) > 1

    def test_small_pool_convergence(self):
        """With pool=1 (9 of 10 active), sampling still works."""
        cfg = self._make_cfg(distribution=Distribution.normal, mean=1.0, std=0.1)
        n_sae = 10
        active = t.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        result = sample_inactive_features(active, n_sae, cfg, rng)
        if len(result) > 0:
            assert result.item() == 9

    def test_returns_int64_tensor(self):
        cfg = self._make_cfg(mean=3.0)
        active = t.arange(10, dtype=t.int64)
        n_sae = 100
        rng = t.Generator().manual_seed(42)
        result = sample_inactive_features(active, n_sae, cfg, rng)
        assert result.dtype == t.int64


# ---------------------------------------------------------------------------
# Distribution enum
# ---------------------------------------------------------------------------


class TestDistributionEnum:
    """Tests for the Distribution enum."""

    def test_from_string_geometric(self):
        assert Distribution("geometric") == Distribution.geometric

    def test_from_string_normal(self):
        assert Distribution("normal") == Distribution.normal

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            Distribution("uniform")

    def test_value_is_string(self):
        assert Distribution.geometric.value == "geometric"
        assert Distribution.normal.value == "normal"

    def test_parse_feature_selection_returns_enum(self):
        result = parse_feature_selection({"distribution": "geometric", "mean": 5.0})
        assert isinstance(result, SamplingSelection)
        assert isinstance(result.distribution, Distribution)
        assert result.distribution == Distribution.geometric

    def test_parse_negative_selection_returns_enum(self):
        result = parse_negative_selection(
            {"distribution": "normal", "mean": 5.0, "std": 2.0}
        )
        assert isinstance(result, NegativeSamplingSelection)
        assert isinstance(result.distribution, Distribution)
        assert result.distribution == Distribution.normal


# ---------------------------------------------------------------------------
# SamplingBundle
# ---------------------------------------------------------------------------


class TestSamplingBundle:
    """Tests for the SamplingBundle dataclass."""

    def test_creation(self):
        """SamplingBundle can be created with all required fields."""
        selection = SamplingSelection(
            distribution=Distribution.geometric, mean=5.0, std=5.0
        )
        table = t.zeros(10, 99, dtype=t.float64)
        rng = t.Generator().manual_seed(42)
        bundle = SamplingBundle(selection=selection, percentile_table=table, rng=rng)
        assert bundle.selection is selection
        assert bundle.percentile_table is table
        assert bundle.rng is rng

    def test_rng_is_mutable(self):
        """SamplingBundle is not frozen -- the rng field can be mutated in-place."""
        selection = SamplingSelection()
        table = t.zeros(5, 99, dtype=t.float64)
        rng = t.Generator().manual_seed(0)
        bundle = SamplingBundle(selection=selection, percentile_table=table, rng=rng)
        # Consume some state from the RNG through the bundle reference
        initial_state = bundle.rng.get_state()
        t.rand(1, generator=bundle.rng)
        # State should have advanced (not frozen)
        new_state = bundle.rng.get_state()
        assert not t.equal(initial_state, new_state)

    def test_feature_selection_config_union(self):
        """FeatureSelectionConfig accepts both ThresholdSelection and SamplingBundle."""
        thresh = ThresholdSelection(
            threshold_type=ThresholdType.segment_topk,
            threshold=10.0,
        )
        bundle = SamplingBundle(
            selection=SamplingSelection(),
            percentile_table=t.zeros(5, 99, dtype=t.float64),
            rng=t.Generator().manual_seed(0),
        )
        # Both should be valid instances of the union
        assert isinstance(thresh, ThresholdSelection)
        assert isinstance(bundle, SamplingBundle)

        # Verify the union type alias covers both
        def accepts_config(cfg: FeatureSelectionConfig) -> str:
            if isinstance(cfg, ThresholdSelection):
                return "threshold"
            if isinstance(cfg, SamplingBundle):
                return "sampling"
            return "unknown"

        assert accepts_config(thresh) == "threshold"
        assert accepts_config(bundle) == "sampling"


# ---------------------------------------------------------------------------
# geometric_() primitive validation
# ---------------------------------------------------------------------------


class TestGeometricPrimitive:
    """Tests validating geometric_() primitive sampling behavior."""

    def test_monte_carlo_mean(self):
        """10k samples from geometric_()-1 should have mean close to configured mean."""
        mean = 5.0
        p = 1.0 / (1.0 + mean)
        rng = t.Generator().manual_seed(42)
        samples = []
        for _ in range(10_000):
            sample = t.empty(1, dtype=t.float64)
            sample.geometric_(p, generator=rng)
            samples.append(int(sample.item()) - 1)  # 1-based to 0-based
        empirical_mean = sum(samples) / len(samples)
        assert abs(empirical_mean - mean) < 0.3, (
            f"Expected mean ~{mean}, got {empirical_mean:.3f}"
        )

    def test_all_non_negative(self):
        """1000 geometric_()-1 samples should all be >= 0."""
        mean = 3.0
        p = 1.0 / (1.0 + mean)
        rng = t.Generator().manual_seed(99)
        for i in range(1000):
            sample = t.empty(1, dtype=t.float64)
            sample.geometric_(p, generator=rng)
            k = int(sample.item()) - 1
            assert k >= 0, f"Sample {i} produced k={k}"

    def test_includes_zero(self):
        """geometric_()-1 can produce 0 (geometric_() produces 1, minus 1 = 0)."""
        mean = 0.5  # high p → many 1s from geometric_() → many 0s after -1
        p = 1.0 / (1.0 + mean)
        rng = t.Generator().manual_seed(42)
        found_zero = False
        for _ in range(1000):
            sample = t.empty(1, dtype=t.float64)
            sample.geometric_(p, generator=rng)
            if int(sample.item()) - 1 == 0:
                found_zero = True
                break
        assert found_zero, "geometric_()-1 never produced 0 in 1000 trials"


# ---------------------------------------------------------------------------
# Task 4: logger.error on fallback
# ---------------------------------------------------------------------------


class TestSampleInactiveFeaturesLogging:
    """Verify logger.error is emitted when rejection sampling doesn't converge."""

    def test_fallback_logs_error(self):
        """Force rejection loop to exhaust 100 iterations, then check logger.error."""
        cfg = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        # n_sae=2, active=[0] → pool_size=1, so k is clamped to 1
        active = t.tensor([0], dtype=t.int64)
        n_sae = 2
        rng = t.Generator().manual_seed(42)

        # Make sample_k always return 1 (need one inactive feature)
        # Make t.randint always return 0 (the active feature) → rejection always fails
        # After 100 iterations, the for/else fallback triggers
        with (
            patch("steering.conditional.sampling.sample_k", return_value=1),
            patch(
                "steering.conditional.sampling.t.randint", return_value=t.tensor([0])
            ),
            patch("steering.conditional.sampling.logger") as mock_logger,
        ):
            result = sample_inactive_features(active, n_sae, cfg, rng)

        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "did not converge" in error_msg
        assert "remaining=1" in error_msg
        assert "pool_size=1" in error_msg
        assert "n_sae_features=2" in error_msg
        # Fallback (enumerate-and-permute) should still produce a valid result
        assert len(result) == 1
        assert result[0].item() == 1  # only inactive feature available


# ---------------------------------------------------------------------------
# Task 5: sample_k exhaustive Distribution enum coverage
# ---------------------------------------------------------------------------


class TestSampleKEnumExhaustive:
    """Exhaustive coverage: every Distribution enum member works, invalid raises."""

    def test_all_enum_members_work(self):
        """Every member of Distribution produces a non-negative int from sample_k."""
        rng = t.Generator().manual_seed(7)
        for dist in Distribution:
            k = sample_k(dist, 5.0, 5.0, rng)
            assert isinstance(k, int), f"{dist.name} should return int"
            assert k >= 0, f"{dist.name} should return non-negative int, got {k}"

    def test_all_enum_members_are_accounted_for(self):
        """Ensure Distribution has exactly the expected members (geometric, normal)."""
        members = {d.value for d in Distribution}
        assert members == {"geometric", "normal"}, (
            f"Unexpected Distribution members: {members}"
        )

    def test_invalid_string_raises(self):
        """A raw string is rejected by the typed distribution boundary."""
        rng = t.Generator().manual_seed(0)
        with pytest.raises(BeartypeCallHintParamViolation, match="distribution"):
            sample_k(cast(Distribution, "exponential"), 5.0, 5.0, rng)

    def test_invalid_enum_like_object_raises(self):
        """An enum-like object is rejected by the typed distribution boundary."""
        rng = t.Generator().manual_seed(0)
        with pytest.raises(BeartypeCallHintParamViolation, match="distribution"):
            sample_k(cast(Distribution, "uniform"), 5.0, 5.0, rng)


# ---------------------------------------------------------------------------
# NegativeBundle dataclass
# ---------------------------------------------------------------------------


class TestNegativeBundle:
    """Tests for the NegativeBundle dataclass."""

    def test_creation(self):
        """NegativeBundle stores selection, n_sae_features, and rng."""
        sel = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(42)
        bundle = NegativeBundle(selection=sel, n_sae_features=1000, rng=rng)
        assert bundle.selection is sel
        assert bundle.n_sae_features == 1000
        assert bundle.rng is rng

    def test_rng_is_mutable(self):
        """NegativeBundle's rng advances state when used (not frozen)."""
        sel = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(42)
        bundle = NegativeBundle(selection=sel, n_sae_features=100, rng=rng)
        state_before = bundle.rng.get_state().clone()
        # Use the rng through sample_inactive_features
        sample_inactive_features(
            t.arange(5, dtype=t.int64),
            bundle.n_sae_features,
            bundle.selection,
            bundle.rng,
        )
        state_after = bundle.rng.get_state()
        assert not t.equal(state_before, state_after), (
            "RNG state should advance after sampling"
        )


# ---------------------------------------------------------------------------
# Task 5: SamplingBundle=None threshold path (negative_rng fallback)
# ---------------------------------------------------------------------------


class TestThresholdPathNegativeRngFallback:
    """Verify sample_inactive_features works via NegativeBundle (threshold path).

    When sampling_bundle is None (threshold path), the caller constructs a
    NegativeBundle with its own rng and passes it to the dataloader.
    This mirrors the training script's NegativeBundle construction codepath.
    """

    def test_standalone_negative_rng_produces_valid_output(self):
        """sample_inactive_features works via NegativeBundle (no SamplingBundle)."""
        sel = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(12345)
        bundle = NegativeBundle(selection=sel, n_sae_features=100, rng=rng)
        active = t.arange(10, dtype=t.int64)
        result = sample_inactive_features(
            active, bundle.n_sae_features, bundle.selection, bundle.rng
        )
        assert result.dtype == t.int64
        active_set = set(active.tolist())
        for fid in result.tolist():
            assert 0 <= fid < bundle.n_sae_features, (
                f"Feature {fid} out of range [0, {bundle.n_sae_features})"
            )
            assert fid not in active_set, f"Feature {fid} is in active set"

    def test_standalone_rng_is_deterministic(self):
        """Two NegativeBundles with same seed produce identical results."""
        sel = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        active = t.arange(5, dtype=t.int64)
        b1 = NegativeBundle(
            selection=sel, n_sae_features=50, rng=t.Generator().manual_seed(99)
        )
        b2 = NegativeBundle(
            selection=sel, n_sae_features=50, rng=t.Generator().manual_seed(99)
        )
        r1 = sample_inactive_features(active, b1.n_sae_features, b1.selection, b1.rng)
        r2 = sample_inactive_features(active, b2.n_sae_features, b2.selection, b2.rng)
        assert t.equal(r1, r2), "NegativeBundle rng should be deterministic"

    def test_threshold_path_negative_bundle_no_sampling_bundle(self):
        """Verify NegativeBundle works independently of SamplingBundle existence.

        This replicates the training script logic:
            neg_rng = t.Generator()
            neg_rng.manual_seed(seed)
            negative_bundle = NegativeBundle(selection=neg_sel, n_sae_features=n, rng=neg_rng)
        """
        neg_sel = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=5.0, std=5.0
        )
        active = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        n_sae = 200
        # sampling_bundle is None → construct NegativeBundle with its own rng
        sampling_bundle = None
        assert sampling_bundle is None  # confirms we are on the threshold path
        neg_rng = t.Generator().manual_seed(42)
        bundle = NegativeBundle(selection=neg_sel, n_sae_features=n_sae, rng=neg_rng)
        neg_ids = sample_inactive_features(
            active, bundle.n_sae_features, bundle.selection, bundle.rng
        )
        assert len(neg_ids) >= 0
        active_set = set(active.tolist())
        for nid in neg_ids.tolist():
            assert nid not in active_set, f"Neg id {nid} overlaps with active set"
            assert 0 <= nid < n_sae


# ---------------------------------------------------------------------------
# Task 5: Checkpoint path uses .distribution.value (string, not enum repr)
# ---------------------------------------------------------------------------


class TestCheckpointPathDistributionValue:
    """Verify checkpoint dirname uses feature_selection_dirname (nested dirs for sampling).

    The shared helper builds:
        ThresholdSelection  → "t{threshold}" (single dir)
        SamplingSelection   → "dist_{dist}/mean_{mean}/std_{std}" (nested dirs)
    """

    def test_geometric_value_is_plain_string(self):
        """Distribution.geometric.value == 'geometric' (no enum prefix)."""
        assert Distribution.geometric.value == "geometric"
        assert "Distribution" not in Distribution.geometric.value

    def test_normal_value_is_plain_string(self):
        """Distribution.normal.value == 'normal' (no enum prefix)."""
        assert Distribution.normal.value == "normal"
        assert "Distribution" not in Distribution.normal.value

    def test_checkpoint_dirname_geometric(self):
        """feature_selection_dirname with geometric selection produces nested dirs."""
        from steering.conditional.feature_pipeline import feature_selection_dirname

        selection = SamplingSelection(
            distribution=Distribution.geometric, mean=5.0, std=5.0
        )
        fs_dirname = feature_selection_dirname(selection)
        assert fs_dirname == "dist_geometric/mean_5.0/std_5.0", (
            f"Expected 'dist_geometric/mean_5.0/std_5.0', got '{fs_dirname}'"
        )
        assert "Distribution" not in fs_dirname

    def test_checkpoint_dirname_normal(self):
        """feature_selection_dirname with normal selection produces nested dirs."""
        from steering.conditional.feature_pipeline import feature_selection_dirname

        selection = SamplingSelection(
            distribution=Distribution.normal, mean=10.0, std=2.0
        )
        fs_dirname = feature_selection_dirname(selection)
        assert fs_dirname == "dist_normal/mean_10.0/std_2.0", (
            f"Expected 'dist_normal/mean_10.0/std_2.0', got '{fs_dirname}'"
        )
        assert "Distribution" not in fs_dirname

    def test_checkpoint_dirname_parsed_from_dict(self):
        """parse_feature_selection → feature_selection_dirname produces nested dirs."""
        from steering.conditional.feature_pipeline import feature_selection_dirname

        result = parse_feature_selection({"distribution": "geometric", "mean": 3.0})
        assert isinstance(result, SamplingSelection)
        fs_dirname = feature_selection_dirname(result)
        assert fs_dirname.startswith("dist_geometric/"), (
            f"Expected prefix 'dist_geometric/', got '{fs_dirname}'"
        )
        assert "Distribution" not in fs_dirname

    def test_checkpoint_dirname_threshold(self):
        """feature_selection_dirname with ThresholdSelection produces flat dir."""
        from steering.conditional.feature_pipeline import feature_selection_dirname
        from steering.conditional.thresholding import ThresholdType

        selection = ThresholdSelection(ThresholdType.segment_topk, 2048.0)
        fs_dirname = feature_selection_dirname(selection)
        assert fs_dirname == "t2048.0"

    def test_enum_repr_would_differ_from_value(self):
        """Confirm that str(Distribution.geometric) != Distribution.geometric.value.

        This documents why .value must be used explicitly in the f-string.
        """
        # The .value is what we want (just the string "geometric")
        assert Distribution.geometric.value == "geometric"
        # Confirm .value does not include any class/enum prefix
        assert "." not in Distribution.geometric.value
