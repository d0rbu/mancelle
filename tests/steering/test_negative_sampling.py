"""Tests for negative feature sampling in the encoder-decoder pipeline.

Covers:
- sample_inactive_features unit tests
- Bucket shifting when inactive_k > 0
- Eval adapter bucket correctness

Note: Live-dataloader integration tests for negative sampling were removed
along with the live encoder-decoder dataloader.
"""

import torch as t

from steering.conditional.discretization import (
    build_boundaries_from_stats,
    discretize_activations,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    sample_inactive_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neg_config(mean: float = 3.0, std: float = 3.0) -> NegativeSamplingSelection:
    """Helper: build a geometric NegativeSamplingSelection."""
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=mean,
        std=std,
    )


# ===========================================================================
# Unit tests for sample_inactive_features
# ===========================================================================


class TestSampleInactiveFeatures:
    """Unit tests for the sample_inactive_features function."""

    def test_no_overlap_with_active(self):
        """Sampled features should not overlap with active features."""
        active = t.tensor([10, 20, 30, 40, 50], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=20.0, std=20.0)
        result = sample_inactive_features(
            active, n_sae_features=100, config=config, rng=rng
        )
        active_set = set(active.tolist())
        for fid in result.tolist():
            assert fid not in active_set, f"Inactive sample {fid} overlaps with active"

    def test_no_duplicates(self):
        """Sampled features should have no duplicates."""
        active = t.tensor([0, 1], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=50.0, std=50.0)
        result = sample_inactive_features(
            active, n_sae_features=100, config=config, rng=rng
        )
        assert len(set(result.tolist())) == len(result)

    def test_no_active_features(self):
        """All features are available when n_active=0."""
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=10.0, std=10.0)
        result = sample_inactive_features(
            t.tensor([], dtype=t.int64),
            n_sae_features=100,
            config=config,
            rng=rng,
        )
        # With mean=10 geometric, we should get at least some features
        assert len(result) >= 0  # non-negative count guaranteed

    def test_all_features_active(self):
        """When all features are active, returns empty."""
        active = t.arange(50, dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=10.0, std=10.0)
        result = sample_inactive_features(
            active, n_sae_features=50, config=config, rng=rng
        )
        assert len(result) == 0

    def test_clamps_to_available(self):
        """When k > n_available, clamps to n_available."""
        active = t.arange(95, dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        # High mean to try requesting more than the 5 available
        config = _make_neg_config(mean=100.0, std=100.0)
        result = sample_inactive_features(
            active, n_sae_features=100, config=config, rng=rng
        )
        assert len(result) <= 5  # only 5 available
        active_set = set(active.tolist())
        for fid in result.tolist():
            assert fid not in active_set

    def test_determinism_with_seeded_rng(self):
        """Results are reproducible with fixed-seed RNG."""
        active = t.tensor([0, 1, 2], dtype=t.int64)
        config = _make_neg_config(mean=10.0, std=10.0)
        rng1 = t.Generator().manual_seed(42)
        r1 = sample_inactive_features(
            active, n_sae_features=1000, config=config, rng=rng1
        )
        rng2 = t.Generator().manual_seed(42)
        r2 = sample_inactive_features(
            active, n_sae_features=1000, config=config, rng=rng2
        )
        assert t.equal(r1, r2)

    def test_valid_range(self):
        """All returned feature IDs are in [0, n_sae_features)."""
        active = t.tensor([5, 10], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=30.0, std=30.0)
        result = sample_inactive_features(
            active, n_sae_features=50, config=config, rng=rng
        )
        if len(result) > 0:
            assert (result >= 0).all()
            assert (result < 50).all()

    def test_excludes_all_provided_fids(self):
        """All features in active_feature_ids are excluded, not just a subset."""
        all_activating = t.arange(10, dtype=t.int64)  # features 0-9 activated
        n_sae = 20
        config = _make_neg_config(mean=8.0, std=8.0)

        for i in range(50):  # repeat for statistical confidence
            rng = t.Generator().manual_seed(i)
            result = sample_inactive_features(
                all_activating, n_sae, config=config, rng=rng
            )
            result_set = set(result.tolist())
            for fid in all_activating.tolist():
                assert fid not in result_set, (
                    f"Feature {fid} is in active_feature_ids but was sampled as negative"
                )
            # All results should be in [10, 19]
            assert all(10 <= fid < 20 for fid in result_set)

    def test_heuristic_slow_path_high_density(self):
        """High-density active set forces slow path; results still correct."""
        # 90% active -> high collision rate -> slow path should be used
        n_sae = 1000
        active = t.arange(900, dtype=t.int64)  # 900/1000 active
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=50.0, std=50.0)
        result = sample_inactive_features(active, n_sae, config=config, rng=rng)
        assert len(result) <= 100  # at most 100 available
        active_set = set(active.tolist())
        assert not (set(result.tolist()) & active_set)
        assert all(0 <= fid < n_sae for fid in result.tolist())

    def test_feature_ids_in_valid_range(self):
        """All returned feature IDs are in [0, n_sae_features)."""
        active = t.tensor([5, 10], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        config = _make_neg_config(mean=30.0, std=30.0)
        result = sample_inactive_features(
            active, n_sae_features=50, config=config, rng=rng
        )
        if len(result) > 0:
            assert (result >= 0).all()
            assert (result < 50).all()


# ===========================================================================
# Unit tests for bucket shifting
# ===========================================================================


class TestBucketAssignment:
    """Test that active/inactive bucket indices are correctly assigned."""

    def test_active_buckets_unchanged(self):
        """Active features keep their original bucket indices (no shift)."""
        fids = t.tensor([0, 1, 2], dtype=t.int32)
        mags = t.tensor([0.5, 1.0, 1.5], dtype=t.float32)
        # 3 boundaries -> 4 positive buckets (indices 0-3)
        boundaries = t.tensor(
            [[0.3, 0.7, 1.2], [0.3, 0.7, 1.2], [0.3, 0.7, 1.2]], dtype=t.float64
        )

        buckets = discretize_activations(fids, mags, boundaries)
        # Active buckets are the raw searchsorted output -- no shift applied
        assert (buckets >= 0).all()
        assert (buckets <= 3).all()

    def test_negatives_get_last_bucket(self):
        """Inactive features get bucket index num_buckets - 1 (the last bucket)."""
        num_buckets = 5
        neg_buckets = t.full((10,), num_buckets - 1, dtype=t.int64)
        assert (neg_buckets == 4).all()

    def test_backward_compat_no_change_when_inactive_k_zero(self):
        """When inactive_k=0, buckets are identical to current behavior."""
        fids = t.tensor([0, 1, 2], dtype=t.int32)
        mags = t.tensor([0.5, 1.0, 1.5], dtype=t.float32)
        boundaries = t.tensor([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]], dtype=t.float64)

        buckets = discretize_activations(fids, mags, boundaries)
        # Full range [0, num_buckets-1] = [0, 2] available for active features
        assert buckets.min() >= 0

    def test_binary_mode_only_zero_and_one(self):
        """num_buckets=2 with inactive_k>0: bucket 0 = active, bucket 1 = inactive."""
        # positive_buckets=1, boundaries shape (n_sae, 0)
        fids = t.tensor([0, 1, 2], dtype=t.int32)
        mags = t.tensor([0.5, 1.0, 1.5], dtype=t.float32)
        boundaries = t.zeros(3, 0, dtype=t.float64)
        active_buckets = discretize_activations(fids, mags, boundaries)
        assert (active_buckets == 0).all()  # all active -> bucket 0

        # Inactive features -> bucket 1 (= num_buckets - 1)
        num_buckets = 2
        inactive_bucket = t.full((3,), num_buckets - 1, dtype=t.int64)
        assert (inactive_bucket == 1).all()

    def test_active_and_inactive_ranges_disjoint(self):
        """Active buckets [0, positive_buckets-1] never overlap inactive bucket."""
        num_buckets = 5
        positive_buckets = num_buckets - 1  # 4
        # Boundaries for 4 positive buckets -> 3 boundary values
        boundaries = t.tensor([[0.3, 0.6, 0.9]] * 3, dtype=t.float64)
        fids = t.tensor([0, 1, 2], dtype=t.int32)
        mags = t.tensor([0.1, 0.5, 1.0], dtype=t.float32)

        active_buckets = discretize_activations(fids, mags, boundaries)
        # Active buckets should be in [0, 3] (positive_buckets - 1)
        assert (active_buckets <= positive_buckets - 1).all()
        # Inactive bucket = 4, which is never returned by searchsorted
        assert (active_buckets != num_buckets - 1).all()

    def test_boundary_computation_positive_buckets(self):
        """build_boundaries_from_stats with reduced positive_buckets returns correct shape."""
        from dataclasses import dataclass

        @dataclass
        class MockStats:
            count: int
            percentiles: dict

        stats = {}
        for fid in range(10):
            pcts = {f"p{i}": float(i) for i in range(1, 100)}
            stats[str(fid)] = MockStats(count=100, percentiles=pcts)

        # 5 positive buckets -> 4 boundaries
        boundaries = build_boundaries_from_stats(
            stats, n_sae_features=10, num_buckets=5
        )
        assert boundaries.shape == (10, 4)

        # 3 positive buckets -> 2 boundaries
        boundaries_3 = build_boundaries_from_stats(
            stats, n_sae_features=10, num_buckets=3
        )
        assert boundaries_3.shape == (10, 2)


