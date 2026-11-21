import pytest
import torch as t

from steering.conditional.discretization import (
    discretize_activations,
    percentile_to_bucket,
)
from tests.steering._discretization_reference import compute_bucket_boundaries


class TestComputeBucketBoundaries:
    """Per-feature percentile boundaries for discretization."""

    def test_4_buckets(self):
        # Feature 0 has activations [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # 4 buckets -> 3 boundaries at p25, p50, p75
        feature_ids = t.tensor([0] * 8, dtype=t.int32)
        feature_mags = t.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=t.float32
        )
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        assert boundaries.shape == (1, 3)  # 1 feature, 3 boundaries
        # Boundaries should be monotonically increasing
        assert boundaries[0, 0] < boundaries[0, 1] < boundaries[0, 2]

    def test_2_buckets(self):
        feature_ids = t.tensor([0] * 4, dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=2)
        assert boundaries.shape == (1, 1)  # 1 boundary (median)

    def test_multiple_features(self):
        feature_ids = t.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=t.int32)
        feature_mags = t.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=t.float32
        )
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        assert boundaries.shape == (2, 2)  # 2 features, 2 boundaries each

    def test_feature_with_single_activation(self):
        # Single activation -> all boundaries are that value
        feature_ids = t.tensor([0], dtype=t.int32)
        feature_mags = t.tensor([0.5], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        assert boundaries.shape[0] == 1
        assert boundaries.shape[1] == 2
        # All boundaries should equal the single activation value
        assert boundaries[0, 0].item() == pytest.approx(0.5)
        assert boundaries[0, 1].item() == pytest.approx(0.5)

    def test_empty_input(self):
        feature_ids = t.tensor([], dtype=t.int32)
        feature_mags = t.tensor([], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        assert boundaries.shape == (0, 3)

    def test_num_buckets_must_be_at_least_2(self):
        feature_ids = t.tensor([0], dtype=t.int32)
        feature_mags = t.tensor([0.5], dtype=t.float32)
        with pytest.raises(AssertionError, match="Need at least 2 buckets"):
            compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=1)

    def test_sparse_feature_ids(self):
        # Feature IDs are not contiguous: 0 and 5
        feature_ids = t.tensor([0, 0, 5, 5], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9, 0.2, 0.8], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=2)
        # Shape should be (6, 1) to cover feature IDs 0-5
        assert boundaries.shape == (6, 1)
        # Features 1-4 should have zero boundaries (no activations)
        for fid in range(1, 5):
            assert boundaries[fid, 0].item() == 0.0
        # Feature 0 and 5 should have non-zero boundaries
        assert boundaries[0, 0].item() > 0
        assert boundaries[5, 0].item() > 0

    def test_length_mismatch_raises(self):
        feature_ids = t.tensor([0, 1], dtype=t.int32)
        feature_mags = t.tensor([0.5], dtype=t.float32)
        with pytest.raises(AssertionError):
            compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=2)

    def test_boundaries_are_float64(self):
        feature_ids = t.tensor([0] * 4, dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        assert boundaries.dtype == t.float64


class TestDiscretizeActivations:
    """Map activation magnitudes to bucket indices."""

    def test_basic(self):
        # Feature 0 boundaries: [0.3, 0.6]  (3 buckets)
        # Activation 0.1 -> bucket 0, 0.4 -> bucket 1, 0.8 -> bucket 2
        boundaries = t.tensor([[0.3, 0.6]], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.4, 0.8], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.tolist() == [0, 1, 2]

    def test_boundary_values(self):
        # Exact boundary value -> goes to higher bucket (searchsorted right)
        boundaries = t.tensor([[0.5]], dtype=t.float64)
        feature_ids = t.tensor([0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.6], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets[0].item() == 1  # at boundary -> higher bucket
        assert buckets[1].item() == 1

    def test_multiple_features(self):
        # Feature 0 boundaries: [0.3], Feature 1 boundaries: [0.7]
        boundaries = t.tensor([[0.3], [0.7]], dtype=t.float64)
        feature_ids = t.tensor([0, 1], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.5], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets[0].item() == 1  # 0.5 > 0.3
        assert buckets[1].item() == 0  # 0.5 < 0.7

    def test_empty(self):
        boundaries = t.tensor([], dtype=t.float64).reshape(0, 0)
        feature_ids = t.tensor([], dtype=t.int32)
        feature_mags = t.tensor([], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert len(buckets) == 0

    def test_unknown_feature_raises_value_error(self):
        # Feature ID beyond boundaries length -> ValueError
        boundaries = t.tensor([[0.5]], dtype=t.float64)
        feature_ids = t.tensor([5], dtype=t.int32)
        feature_mags = t.tensor([0.8], dtype=t.float32)
        with pytest.raises(ValueError, match="exceed boundaries size"):
            discretize_activations(feature_ids, feature_mags, boundaries)

    def test_returns_int64(self):
        boundaries = t.tensor([[0.5]], dtype=t.float64)
        feature_ids = t.tensor([0], dtype=t.int32)
        feature_mags = t.tensor([0.3], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.dtype == t.int64

    def test_all_below_first_boundary(self):
        # All activations below the first boundary -> all bucket 0
        boundaries = t.tensor([[0.5, 0.8]], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.tolist() == [0, 0, 0]

    def test_all_above_last_boundary(self):
        # All activations above the last boundary -> all highest bucket
        boundaries = t.tensor([[0.1, 0.2]], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.6, 0.7], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.tolist() == [2, 2, 2]

    def test_roundtrip_with_compute_boundaries(self):
        """Integration: compute boundaries then discretize."""
        feature_ids = t.tensor([0] * 100, dtype=t.int32)
        feature_mags = t.linspace(0.0, 1.0, 100)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        # All buckets should be in [0, 3]
        assert buckets.min().item() >= 0
        assert buckets.max().item() <= 3
        # Should have some distribution across buckets
        unique_buckets = buckets.unique()
        assert len(unique_buckets) >= 3  # at least 3 of 4 buckets used

    def test_length_mismatch_raises(self):
        boundaries = t.tensor([[0.5]], dtype=t.float64)
        feature_ids = t.tensor([0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.3], dtype=t.float32)
        with pytest.raises(AssertionError):
            discretize_activations(feature_ids, feature_mags, boundaries)


class TestPercentileToBucket:
    """Map predicted percentile values to discrete bucket indices.

    Buckets are equally-spaced percentile bands by construction
    (compute_bucket_boundaries uses linspace(1/B, 1-1/B, B-1)). Therefore
    bucket = floor(percentile * num_buckets) clamped to [0, num_buckets-1].
    """

    def test_basic_mapping_4_buckets(self):
        # 4 buckets: [0, 0.25) -> 0, [0.25, 0.5) -> 1, [0.5, 0.75) -> 2, [0.75, 1.0] -> 3
        percentiles = t.tensor([0.0, 0.1, 0.24999, 0.25, 0.3, 0.5, 0.74, 0.75, 0.99, 1.0])
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        assert buckets.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]

    def test_p_equals_one_clamps_to_last_bucket(self):
        # p = 1.0 would naturally produce floor(1.0 * 4) = 4, but must clamp to 3.
        percentiles = t.tensor([1.0])
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        assert buckets.tolist() == [3]

    def test_p_equals_zero_maps_to_bucket_zero(self):
        percentiles = t.tensor([0.0])
        buckets = percentile_to_bucket(percentiles, num_buckets=10)
        assert buckets.tolist() == [0]

    def test_returns_int64(self):
        percentiles = t.tensor([0.5])
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        assert buckets.dtype == t.int64

    def test_empty_input(self):
        percentiles = t.tensor([], dtype=t.float32)
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        assert buckets.shape == (0,)
        assert buckets.dtype == t.int64

    def test_2d_input_preserves_shape(self):
        # Provider may pass (n_seg, n_sae)-shaped tensors
        percentiles = t.tensor([[0.0, 0.5], [0.25, 0.999]])
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        assert buckets.shape == (2, 2)
        assert buckets.tolist() == [[0, 2], [1, 3]]

    def test_all_bucket_counts_in_range(self):
        # Exhaustive: for several num_buckets, sweep p and assert in [0, B-1].
        ps = t.tensor([0.0, 0.0001, 0.5, 0.9999, 1.0])
        for B in [2, 4, 10, 16]:
            buckets = percentile_to_bucket(ps, num_buckets=B)
            assert buckets.min().item() >= 0
            assert buckets.max().item() <= B - 1
            # p=0 -> 0; p=1 -> B-1
            assert buckets[0].item() == 0
            assert buckets[-1].item() == B - 1

    def test_two_buckets(self):
        # Below p=0.5 -> 0, at-or-above 0.5 -> 1
        percentiles = t.tensor([0.0, 0.49, 0.5, 0.99, 1.0])
        buckets = percentile_to_bucket(percentiles, num_buckets=2)
        assert buckets.tolist() == [0, 0, 1, 1, 1]

    def test_negative_or_above_one_clamped_or_rejected(self):
        # The provider should already sigmoid logits; out-of-range values
        # indicate a bug. We accept clamping behavior to keep the function
        # robust under bf16/fp32 numerical drift around p=1.0.
        percentiles = t.tensor([-0.001, 1.0001])
        buckets = percentile_to_bucket(percentiles, num_buckets=4)
        # Negative -> floor(negative * 4) is e.g. -1; we expect clamp to 0.
        # Slightly-over-1 -> 4; clamp to 3.
        assert buckets.tolist() == [0, 3]
