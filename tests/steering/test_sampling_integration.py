"""Integration tests for stochastic feature sampling → discretization pipeline."""

import torch as t

from steering.conditional.discretization import discretize_activations
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
    build_percentile_table,
    sample_features,
    sample_inactive_features,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
)
from steering.stats_types import PerFeatureStat


def _make_stat(pcts: dict[str, float]) -> PerFeatureStat:
    return PerFeatureStat(
        count=100,
        pct_segments_active=0.5,
        mean_mag=1.0,
        mean_mag_incl_zeros=0.5,
        std_mag=0.5,
        min_mag=0.1,
        max_mag=5.0,
        median_mag=1.0,
        percentiles=pcts,
    )


class TestSamplingDiscretizationPipeline:
    """Verify sampling output feeds correctly into discretization."""

    def test_sampled_features_discretize_correctly(self):
        """Sampled features should produce valid bucket indices."""
        n_features = 100
        num_buckets = 10
        n_boundaries = num_buckets - 1

        pcts = {f"p{i}": float(i) for i in range(1, 100)}
        stats = {str(f): _make_stat(pcts) for f in range(n_features)}
        table = build_percentile_table(stats, n_features)

        boundaries = t.zeros(n_features, n_boundaries, dtype=t.float64)
        for f in range(n_features):
            boundaries[f] = t.linspace(10, 90, n_boundaries, dtype=t.float64)

        fids = t.tensor([10, 20, 30, 40, 50], dtype=t.int64)
        fmags = t.tensor([50.0, 75.0, 25.0, 90.0, 10.0], dtype=t.float64)

        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=0.5)
        rng = t.Generator().manual_seed(42)

        sampled_ids, sampled_mags = sample_features(fids, fmags, table, cfg, rng)

        if len(sampled_ids) > 0:
            buckets = discretize_activations(
                sampled_ids,
                sampled_mags.float(),
                boundaries,
            )
            assert (buckets >= 0).all()
            assert (buckets < num_buckets).all()
            assert len(buckets) == len(sampled_ids)

    def test_empty_sampling_discretizes_to_empty(self):
        """When sampling returns no features, discretization handles it."""
        boundaries = t.zeros(10, 9, dtype=t.float64)
        fids = t.tensor([], dtype=t.int64)
        fmags = t.tensor([], dtype=t.float32)
        buckets = discretize_activations(fids, fmags, boundaries)
        assert len(buckets) == 0

    def test_sampled_ids_are_subset(self):
        """Sampled feature IDs must be a subset of input IDs."""
        n_features = 50
        pcts = {f"p{i}": float(i) for i in range(1, 100)}
        stats = {str(f): _make_stat(pcts) for f in range(n_features)}
        table = build_percentile_table(stats, n_features)

        fids = t.tensor([5, 15, 25, 35, 45], dtype=t.int64)
        fmags = t.tensor([10.0, 50.0, 90.0, 30.0, 70.0], dtype=t.float64)

        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=1.0)
        input_set = set(fids.tolist())

        for seed in range(50):
            rng = t.Generator().manual_seed(seed)
            out_ids, _ = sample_features(fids, fmags, table, cfg, rng)
            for oid in out_ids.tolist():
                assert oid in input_set, f"Sampled ID {oid} not in input {input_set}"


class TestNegativeSamplingIntegration:
    """Integration tests for positive + negative feature sampling → discretization."""

    def _setup(self, n_features=100, num_buckets=10):
        pcts = {f"p{i}": float(i) for i in range(1, 100)}
        stats = {str(f): _make_stat(pcts) for f in range(n_features)}
        table = build_percentile_table(stats, n_features)
        # Boundaries for positive buckets only (num_buckets - 1 positive)
        n_pos_buckets = num_buckets - 1
        n_boundaries = n_pos_buckets - 1
        boundaries = t.zeros(n_features, n_boundaries, dtype=t.float64)
        for f in range(n_features):
            boundaries[f] = t.linspace(10, 90, n_boundaries, dtype=t.float64)
        return stats, table, boundaries, n_features, num_buckets

    def test_positive_sampling_plus_negative_correct_buckets(self):
        stats, table, boundaries, n_features, num_buckets = self._setup()
        fids = t.arange(20, dtype=t.int64)
        fmags = t.linspace(10, 90, 20)
        pos_cfg = SamplingSelection(distribution=Distribution.normal, mean=5.0, std=1.0)
        neg_cfg = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(42)
        pos_ids, pos_mags = sample_features(fids, fmags, table, pos_cfg, rng)
        neg_ids = sample_inactive_features(fids, n_features, neg_cfg, rng)
        if len(pos_ids) > 0:
            pos_buckets = discretize_activations(pos_ids, pos_mags.float(), boundaries)
            assert (pos_buckets >= 0).all()
            assert (pos_buckets < num_buckets - 1).all()
        if len(neg_ids) > 0:
            neg_buckets = t.full((len(neg_ids),), num_buckets - 1, dtype=t.int64)
            assert (neg_buckets == num_buckets - 1).all()
        if len(pos_ids) > 0 and len(neg_ids) > 0:
            pos_set = set(pos_ids.tolist())
            neg_set = set(neg_ids.tolist())
            assert pos_set.isdisjoint(neg_set)

    def test_positive_threshold_plus_negative_correct_buckets(self):
        stats, table, boundaries, n_features, num_buckets = self._setup()
        fids = t.arange(50, dtype=t.int64)
        fmags = t.linspace(1, 99, 50)
        thresh_cfg = ThresholdConfig(ThresholdType.segment_topk, 10.0)
        pos_ids, pos_mags = apply_threshold(thresh_cfg, fids, fmags)
        neg_cfg = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(42)
        neg_ids = sample_inactive_features(fids, n_features, neg_cfg, rng)
        if len(pos_ids) > 0:
            pos_buckets = discretize_activations(pos_ids, pos_mags.float(), boundaries)
            assert (pos_buckets >= 0).all()
            assert (pos_buckets < num_buckets - 1).all()
        if len(neg_ids) > 0:
            raw_set = set(fids.tolist())
            for nid in neg_ids.tolist():
                assert nid not in raw_set

    def test_negative_features_no_overlap_with_raw_active(self):
        stats, table, boundaries, n_features, num_buckets = self._setup()
        raw_fids = t.arange(30, dtype=t.int64)
        neg_cfg = NegativeSamplingSelection(mean=5.0, std=5.0)
        rng = t.Generator().manual_seed(42)
        neg_ids = sample_inactive_features(raw_fids, n_features, neg_cfg, rng)
        raw_set = set(raw_fids.tolist())
        for nid in neg_ids.tolist():
            assert nid not in raw_set


class TestBinaryBucketMode:
    """Tests for num_buckets=2 binary mode where effective_buckets=1.

    When num_buckets=2 and negative_selection is enabled, effective_buckets = 2 - 1 = 1.
    With only 1 positive bucket, boundaries is an empty tensor (n, 0) and
    discretize_activations maps all positive features to bucket 0.
    Negative features get bucket 1 (num_buckets - 1 = 1).
    """

    def test_two_buckets_positive_all_zero(self):
        """With empty boundaries, positive features all get bucket 0."""
        n_features = 50
        fids = t.tensor([3, 7, 15, 42], dtype=t.int64)
        fmags = t.tensor([1.5, 3.0, 0.8, 5.2], dtype=t.float32)
        # effective_buckets=1 → boundaries shape (n, 0) → all bucket 0
        boundaries = t.zeros(n_features, 0, dtype=t.float64)
        buckets = discretize_activations(fids, fmags, boundaries)
        assert len(buckets) == len(fids)
        assert (buckets == 0).all(), (
            f"Expected all bucket 0 for positive features, got {buckets}"
        )

    def test_two_buckets_negative_all_one(self):
        """Negative features get bucket 1 (num_buckets - 1) in binary mode."""
        num_buckets = 2
        n_sae_features = 100
        raw_fids = t.arange(10, dtype=t.int64)
        neg_cfg = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0
        )
        rng = t.Generator().manual_seed(42)
        neg_ids = sample_inactive_features(raw_fids, n_sae_features, neg_cfg, rng)
        assert len(neg_ids) > 0, "Expected at least one negative feature"
        neg_buckets = t.full((len(neg_ids),), num_buckets - 1, dtype=t.int64)
        assert (neg_buckets == 1).all(), (
            f"Expected all bucket 1 for negative features, got {neg_buckets}"
        )

    def test_combined_binary_signal(self):
        """Concatenated positive (bucket 0) + negative (bucket 1) buckets are correct."""
        num_buckets = 2
        n_sae_features = 200
        n_features = 200
        # Positive features
        pos_fids = t.tensor([5, 10, 20], dtype=t.int64)
        pos_mags = t.tensor([2.0, 4.0, 1.0], dtype=t.float32)
        # effective_buckets=1 → empty boundaries → all bucket 0
        boundaries = t.zeros(n_features, 0, dtype=t.float64)
        pos_buckets = discretize_activations(pos_fids, pos_mags, boundaries)
        # Negative features
        neg_cfg = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=2.0, std=2.0
        )
        rng = t.Generator().manual_seed(99)
        neg_ids = sample_inactive_features(pos_fids, n_sae_features, neg_cfg, rng)
        neg_buckets = t.full((len(neg_ids),), num_buckets - 1, dtype=t.int64)
        # Concatenate
        all_fids = t.cat([pos_fids.to(t.int64), neg_ids])
        all_buckets = t.cat([pos_buckets, neg_buckets])
        assert len(all_fids) == len(pos_fids) + len(neg_ids)
        assert len(all_buckets) == len(all_fids)
        # First part should be 0 (positive), second part should be 1 (negative)
        assert (all_buckets[: len(pos_fids)] == 0).all()
        assert (all_buckets[len(pos_fids) :] == 1).all()
        # No overlap between positive and negative feature IDs
        pos_set = set(pos_fids.tolist())
        neg_set = set(neg_ids.tolist())
        assert pos_set.isdisjoint(neg_set)

    def test_effective_buckets_one_uses_empty_boundaries(self):
        """Verify that with num_buckets=2 and negative_selection, boundaries is empty (n, 0)."""
        num_buckets = 2
        negative_selection = NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
        )
        effective_buckets = (
            num_buckets - 1 if negative_selection is not None else num_buckets
        )
        assert effective_buckets == 1
        # Binary mode: boundaries shape is (n, 0)
        n_features = 100
        boundaries = t.zeros(n_features, 0, dtype=t.float64)
        assert boundaries.shape == (n_features, 0)
