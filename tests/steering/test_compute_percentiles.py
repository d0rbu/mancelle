import pytest
import torch as t

from steering.conditional.discretization import (
    compute_percentiles,
)


class TestComputePercentiles:
    """Test compute_percentiles: maps raw magnitudes to [0, 1] via piecewise linear interpolation."""

    def test_empty_input(self):
        fids = t.tensor([], dtype=t.int32)
        fmags = t.tensor([], dtype=t.float32)
        bounds = t.zeros(10, 3, dtype=t.float64)  # 4 buckets, 3 boundaries
        result = compute_percentiles(fids, fmags, bounds)
        assert len(result) == 0
        assert result.dtype == t.float32

    def test_midpoint_of_bucket(self):
        """A magnitude exactly between two boundaries should get the midpoint percentile."""
        bounds = t.zeros(1, 3, dtype=t.float64)
        bounds[0] = t.tensor([1.0, 2.0, 3.0], dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([1.5], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert result.shape == (1,)
        assert abs(result[0].item() - 0.375) < 1e-5

    def test_below_first_boundary(self):
        """Magnitude below first boundary interpolates from 0."""
        bounds = t.zeros(1, 3, dtype=t.float64)
        bounds[0] = t.tensor([2.0, 4.0, 6.0], dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([1.0], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert abs(result[0].item() - 0.125) < 1e-5

    def test_above_last_boundary(self):
        """Magnitude above last boundary interpolates toward 1."""
        bounds = t.zeros(1, 3, dtype=t.float64)
        bounds[0] = t.tensor([2.0, 4.0, 6.0], dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([7.0], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert 0.75 < result[0].item() <= 1.0

    def test_at_boundary(self):
        """Magnitude exactly at a boundary gets the boundary's percentile."""
        bounds = t.zeros(1, 3, dtype=t.float64)
        bounds[0] = t.tensor([2.0, 4.0, 6.0], dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([4.0], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert abs(result[0].item() - 0.50) < 1e-5

    def test_multiple_features(self):
        """Different features use their own boundaries."""
        bounds = t.zeros(2, 3, dtype=t.float64)
        bounds[0] = t.tensor([1.0, 2.0, 3.0], dtype=t.float64)
        bounds[1] = t.tensor([10.0, 20.0, 30.0], dtype=t.float64)
        fids = t.tensor([0, 1], dtype=t.int32)
        fmags = t.tensor([2.0, 20.0], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert abs(result[0].item() - 0.50) < 1e-5
        assert abs(result[1].item() - 0.50) < 1e-5

    def test_output_in_unit_interval(self):
        """All outputs should be in [0, 1]."""
        bounds = t.zeros(5, 9, dtype=t.float64)
        for i in range(5):
            bounds[i] = t.linspace(1.0, 9.0, 9, dtype=t.float64) * (i + 1)
        fids = t.randint(0, 5, (100,), dtype=t.int32)
        fmags = t.randn(100).abs() * 50
        result = compute_percentiles(fids, fmags, bounds)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_zero_boundaries_feature(self):
        """Feature with all-zero boundaries (no activations in corpus)."""
        bounds = t.zeros(1, 3, dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([5.0], dtype=t.float32)
        result = compute_percentiles(fids, fmags, bounds)
        assert not result.isnan().any()
        assert (result >= 0.0).all() and (result <= 1.0).all()

    def test_out_of_range_feature_id(self):
        """Feature ID >= n_features in boundaries should raise."""
        bounds = t.zeros(5, 3, dtype=t.float64)
        fids = t.tensor([0, 10], dtype=t.int32)
        fmags = t.tensor([2.0, 5.0], dtype=t.float32)
        with pytest.raises(ValueError, match="exceed boundaries size"):
            compute_percentiles(fids, fmags, bounds)

    def test_consistency_with_discretize(self):
        """Percentiles should be monotonically consistent with bucket indices."""
        from steering.conditional.discretization import discretize_activations

        bounds = t.zeros(1, 9, dtype=t.float64)
        bounds[0] = t.linspace(1.0, 9.0, 9, dtype=t.float64)
        fids = t.zeros(20, dtype=t.int32)
        fmags = t.linspace(0.0, 10.0, 20, dtype=t.float32)
        buckets = discretize_activations(fids, fmags, bounds)
        pcts = compute_percentiles(fids, fmags, bounds)
        order = buckets.argsort(stable=True)
        sorted_pcts = pcts[order]
        for i in range(1, len(sorted_pcts)):
            assert sorted_pcts[i] >= sorted_pcts[i - 1] - 1e-6
