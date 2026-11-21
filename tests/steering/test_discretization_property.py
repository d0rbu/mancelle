"""Slice 05 audit — property-based tests for discretization primitives.

Pin documented invariants on ``compute_bucket_boundaries``,
``discretize_activations``, ``compute_percentiles``, and
``build_boundaries_from_stats``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.discretization import (
    DEGENERATE_MIDPOINT,
    build_boundaries_from_stats,
    compute_percentiles,
    discretize_activations,
)
from tests.steering._discretization_reference import compute_bucket_boundaries

# ---------------------------------------------------------------------------
# compute_bucket_boundaries
# ---------------------------------------------------------------------------


class TestComputeBucketBoundariesShape:
    @pytest.mark.parametrize("num_buckets", [2, 16, 128])
    def test_shape_for_known_inputs(self, num_buckets):
        feature_ids = t.tensor([0, 0, 0, 1, 1, 2], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5, 1.0, 0.2, 0.4, 0.7])
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        assert boundaries.shape == (3, num_buckets - 1)
        assert boundaries.dtype == t.float64


class TestSingleActivationDegenerate:
    def test_single_activation_returns_constant_boundaries(self):
        # Slice 05 P2 #9: features with one activation get all-equal boundaries.
        feature_ids = t.tensor([7], dtype=t.int32)
        feature_mags = t.tensor([1.5])
        num_buckets = 8
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        # Feature 7 must have all boundaries equal to 1.5.
        assert (boundaries[7] == 1.5).all()
        # Features 0-6 are unseen — all-zero (default).
        assert (boundaries[:7] == 0).all()


class TestPropertyMonotonic:
    @settings(
        derandomize=True, max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        num_buckets=st.integers(min_value=2, max_value=64),
        n_activations=st.integers(min_value=2, max_value=50),
    )
    def test_boundaries_non_decreasing_per_feature(self, num_buckets, n_activations):
        # Single feature with many activations: boundaries must be monotonic.
        feature_ids = t.zeros(n_activations, dtype=t.int32)
        feature_mags = t.linspace(0.1, 10.0, n_activations).float()
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        feat_boundaries = boundaries[0]
        diffs = feat_boundaries[1:] - feat_boundaries[:-1]
        assert (diffs >= -1e-9).all(), (
            f"Non-monotonic boundaries: {feat_boundaries.tolist()}"
        )


# ---------------------------------------------------------------------------
# discretize_activations
# ---------------------------------------------------------------------------


class TestDiscretizeActivationsRange:
    @pytest.mark.parametrize("num_buckets", [2, 4, 16, 128])
    def test_bucket_indices_in_range(self, num_buckets):
        feature_ids = t.tensor([0, 0, 0, 1, 1, 2], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5, 1.0, 0.2, 0.4, 0.7])
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert (buckets >= 0).all()
        assert (buckets < num_buckets).all()
        assert buckets.shape == (6,)
        assert buckets.dtype == t.int64


class TestDiscretizeBoundaryAtRight:
    """``searchsorted(..., right=True)`` puts values *exactly* at a boundary
    in the higher bucket. Pin this contract.
    """

    def test_value_at_boundary_goes_higher(self):
        # 1 feature, 4 activations → 3 boundaries
        feature_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.0, 1.0, 2.0, 3.0])
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        # boundaries[0] = quantile(mags, [0.25, 0.5, 0.75]) = ~[0.75, 1.5, 2.25]
        # Test: a magnitude exactly at boundary[1] (1.5) goes into bucket >= 2
        test_fids = t.tensor([0], dtype=t.int32)
        test_mags = t.tensor([float(boundaries[0, 1])])
        bucket = discretize_activations(test_fids, test_mags, boundaries)
        assert bucket.item() >= 2, (
            f"Magnitude at boundary should go into higher bucket, got {bucket}"
        )


class TestDiscretizeRaisesOnOutOfRange:
    def test_raises_on_id_exceeding_boundaries(self):
        boundaries = t.zeros(3, 4, dtype=t.float64)
        with pytest.raises(ValueError, match="exceed boundaries size"):
            discretize_activations(
                t.tensor([5], dtype=t.int32),
                t.tensor([0.5]),
                boundaries,
            )


class TestDiscretizeEmptyInput:
    def test_empty_returns_empty(self):
        result = discretize_activations(
            t.tensor([], dtype=t.int32),
            t.tensor([]),
            t.zeros(10, 5),
        )
        assert result.shape == (0,)
        assert result.dtype == t.int64


# ---------------------------------------------------------------------------
# compute_percentiles
# ---------------------------------------------------------------------------


class TestComputePercentilesRange:
    @settings(
        derandomize=True, max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        num_buckets=st.integers(min_value=2, max_value=32),
        n_activations=st.integers(min_value=4, max_value=20),
    )
    def test_percentiles_in_unit_interval(self, num_buckets, n_activations):
        feature_ids = t.zeros(n_activations, dtype=t.int32)
        feature_mags = t.rand(n_activations) * 10.0 + 0.1
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        percentiles = compute_percentiles(feature_ids, feature_mags, boundaries)
        assert (percentiles >= 0.0).all()
        assert (percentiles <= 1.0).all()
        assert percentiles.dtype == t.float32


class TestComputePercentilesDegenerate:
    def test_single_bucket_returns_midpoint(self):
        # n_boundaries == 0 → all magnitudes map to DEGENERATE_MIDPOINT.
        feature_ids = t.tensor([0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 1.5])
        # single-bucket boundaries: [n_features, 0]
        boundaries = t.zeros(1, 0, dtype=t.float64)
        result = compute_percentiles(feature_ids, feature_mags, boundaries)
        assert (result == DEGENERATE_MIDPOINT).all()


class TestComputePercentilesEmpty:
    def test_empty_returns_empty(self):
        result = compute_percentiles(
            t.tensor([], dtype=t.int32),
            t.tensor([]),
            t.zeros(10, 5),
        )
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# build_boundaries_from_stats
# ---------------------------------------------------------------------------


@dataclass
class _MockStats:
    percentiles: dict[str, float]


class TestBuildBoundariesPctKeys:
    """Pin the integer-truncation of percentile keys
    ``f"p{int(100 * (i + 1) / num_buckets)}"`` (slice 05 P2 #12).

    For the practical range ``num_buckets ∈ {2, 4, 6, 8, 10, 16, 25, 50}``
    no key collisions occur. Document the pathological cases for
    ``num_buckets >= 100`` where ``int(...)`` truncation can produce
    duplicates — but the assert in ``build_boundaries_from_stats`` will
    surface a missing-key error, so latent silent failure is precluded.
    """

    @pytest.mark.parametrize("num_buckets", [2, 4, 6, 8, 10, 16, 25, 50])
    def test_pct_keys_unique_for_reasonable_num_buckets(self, num_buckets):
        n_boundaries = num_buckets - 1
        keys = [f"p{int(100 * (i + 1) / num_buckets)}" for i in range(n_boundaries)]
        assert len(keys) == len(set(keys)), (
            f"int-truncation collision at num_buckets={num_buckets}: keys={keys}"
        )

    def test_keys_for_num_buckets_2(self):
        # For num_buckets=2, n_boundaries=1 → key="p50".
        keys = [f"p{int(100 * (i + 1) / 2)}" for i in range(1)]
        assert keys == ["p50"]

    @pytest.mark.parametrize(
        ("num_buckets", "legacy_keys"),
        [
            (2, [50]),
            (3, [33, 66]),
            (8, [12, 25, 37, 50, 62, 75, 87]),
            (11, [9, 18, 27, 36, 45, 54, 63, 72, 81, 90]),
        ],
    )
    def test_legacy_int_boundaries_are_frozen(
        self,
        num_buckets: int,
        legacy_keys: list[int],
    ):
        stats = {
            "0": _MockStats(
                percentiles={f"p{k}": float(k) for k in range(1, 100)}
            )
        }

        actual = build_boundaries_from_stats(
            stats,
            n_sae_features=1,
            num_buckets=num_buckets,
        )

        assert t.equal(
            actual,
            t.tensor([legacy_keys], dtype=t.float64),
        )

    def test_shifted_range_round_boundaries_are_frozen(self):
        stats = {
            "0": _MockStats(
                percentiles={f"p{k}": float(k) for k in range(1, 100)}
            )
        }

        actual = build_boundaries_from_stats(
            stats,
            n_sae_features=1,
            num_buckets=4,
            threshold_percentile=0.37,
        )

        assert t.equal(
            actual,
            t.tensor([[58.0, 78.0, 99.0]], dtype=t.float64),
        )


class TestBuildBoundariesFromStatsOk:
    """Smoke test: with full pct keys p1..p99 present, the function
    successfully returns a [n_sae, num_buckets-1] boundary tensor.
    """

    def test_smoke_path(self):
        n_sae = 5
        num_buckets = 4
        per_feature_stats = {
            str(i): _MockStats(
                percentiles={f"p{k}": float(k) * 0.1 + i for k in range(1, 100)},
            )
            for i in range(n_sae)
        }
        result = build_boundaries_from_stats(
            per_feature_stats, n_sae_features=n_sae, num_buckets=num_buckets,
        )
        assert result.shape == (n_sae, num_buckets - 1)
        assert result.dtype == t.float64

    def test_threshold_percentile_path(self):
        # When threshold_percentile is set the keys come from ``round(...)``.
        n_sae = 3
        num_buckets = 4
        per_feature_stats = {
            str(i): _MockStats(
                percentiles={f"p{k}": float(k) * 0.1 + i for k in range(1, 100)},
            )
            for i in range(n_sae)
        }
        result = build_boundaries_from_stats(
            per_feature_stats,
            n_sae_features=n_sae,
            num_buckets=num_buckets,
            threshold_percentile=0.9,
        )
        assert result.shape == (n_sae, num_buckets - 1)
