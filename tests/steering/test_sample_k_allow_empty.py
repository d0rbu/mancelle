"""Tests for sample_k min_k/max_k range parameters (steering/conditional/sampling.py).

min_k controls the minimum sampled value (replaces allow_empty).
max_k controls the maximum sampled value (exclusive).
For geometric: both → (sample % range) + min_k; min_k only → sample + min_k (offset).
For normal: clamp to [min_k, max_k - 1].
"""

import torch

from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    sample_features,
    sample_k,
)


class TestSampleKRange:
    """Verify sample_k respects min_k/max_k."""

    def test_geometric_min_k_1_never_returns_0(self):
        """Geometric with min_k=1 never returns 0."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.geometric, 1.0, 1.0, rng, min_k=1)
            for _ in range(1000)
        ]
        assert all(k >= 1 for k in results), (
            f"min_k=1 returned k=0: min={min(results)}"
        )

    def test_geometric_min_k_0_can_return_0(self):
        """Geometric with min_k=0 can return 0."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.geometric, 1.0, 1.0, rng, min_k=0)
            for _ in range(1000)
        ]
        assert 0 in results, "min_k=0 should occasionally return 0"

    def test_normal_min_k_1_never_returns_0(self):
        """Normal with min_k=1 never returns 0."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.normal, 0.5, 1.0, rng, min_k=1)
            for _ in range(1000)
        ]
        assert all(k >= 1 for k in results), (
            f"Normal min_k=1 returned k<1: min={min(results)}"
        )

    def test_normal_min_k_0_can_return_0(self):
        """Normal with min_k=0 and low mean can return 0."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.normal, 0.5, 2.0, rng, min_k=0)
            for _ in range(1000)
        ]
        assert 0 in results, "Normal min_k=0 with low mean should return 0"

    def test_geometric_min_k_offset_only(self):
        """Geometric with min_k only (no max_k) applies offset without modding."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k = 3
        results = [
            sample_k(Distribution.geometric, 5.0, 5.0, rng, min_k=min_k)
            for _ in range(1000)
        ]
        assert all(k >= min_k for k in results), (
            f"min_k={min_k} returned k < {min_k}: min={min(results)}"
        )
        # With offset, values can exceed mean + min_k significantly
        assert max(results) > min_k + 10, (
            "Offset should allow large values without modding"
        )

    def test_geometric_both_min_max_bounded(self):
        """Geometric with both min_k and max_k stays in [min_k, max_k)."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 3, 10
        results = [
            sample_k(Distribution.geometric, 5.0, 5.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(1000)
        ]
        assert all(min_k <= k < max_k for k in results), (
            f"Out of range: min={min(results)}, max={max(results)}, "
            f"expected [{min_k}, {max_k})"
        )

    def test_normal_both_min_max_bounded(self):
        """Normal with both min_k and max_k clamps to [min_k, max_k-1]."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 2, 8
        results = [
            sample_k(Distribution.normal, 5.0, 3.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(1000)
        ]
        assert all(min_k <= k <= max_k - 1 for k in results), (
            f"Out of range: min={min(results)}, max={max(results)}, "
            f"expected [{min_k}, {max_k - 1}]"
        )

    def test_geometric_large_mean_min_k_1(self):
        """Even with very large mean, min_k=1 still guarantees k>=1."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.geometric, 100.0, 100.0, rng, min_k=1)
            for _ in range(100)
        ]
        assert all(k >= 1 for k in results)

    def test_geometric_max_k_only(self):
        """Geometric with max_k only (min_k=0) caps values."""
        rng = torch.Generator()
        rng.manual_seed(42)
        max_k = 5
        results = [
            sample_k(Distribution.geometric, 10.0, 10.0, rng, min_k=0, max_k=max_k)
            for _ in range(1000)
        ]
        assert all(0 <= k < max_k for k in results), (
            f"Out of range: min={min(results)}, max={max(results)}, expected [0, {max_k})"
        )


class TestSampleFeaturesRange:
    """Integration: sample_features with min_k/max_k."""

    def test_sample_features_min_k_1_always_returns_at_least_1(self):
        """With min_k=1, sample_features always selects at least 1 feature."""
        rng = torch.Generator()
        rng.manual_seed(42)
        n_features = 10
        pct_table = torch.rand(n_features, 99, dtype=torch.float64).cumsum(dim=1)
        feature_ids = torch.arange(5, dtype=torch.int64)
        feature_mags = torch.rand(5) + 0.01
        config = SamplingSelection(
            distribution=Distribution.geometric, mean=1.0, std=1.0,
            min_k=1, max_k=5,
        )
        for _ in range(100):
            ids, mags = sample_features(feature_ids, feature_mags, pct_table, config, rng)
            assert len(ids) >= 1, (
                "sample_features returned 0 features with min_k=1"
            )

    def test_sample_features_min_k_0_can_return_0(self):
        """With min_k=0, sample_features can return 0 features."""
        rng = torch.Generator()
        rng.manual_seed(42)
        n_features = 10
        pct_table = torch.rand(n_features, 99, dtype=torch.float64).cumsum(dim=1)
        feature_ids = torch.arange(5, dtype=torch.int64)
        feature_mags = torch.rand(5) + 0.01
        config = SamplingSelection(
            distribution=Distribution.geometric, mean=0.5, std=0.5,
            min_k=0, max_k=3,
        )
        empty_count = 0
        for _ in range(500):
            ids, _ = sample_features(feature_ids, feature_mags, pct_table, config, rng)
            if len(ids) == 0:
                empty_count += 1
        assert empty_count > 0, "min_k=0 never returned empty result in 500 tries"
