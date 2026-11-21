"""Hardcore edge-case tests for sample_k (steering/conditional/sampling.py).

Covers degenerate ranges, distribution boundary behaviour, modular arithmetic
correctness, clamping guarantees, statistical coverage, and determinism.
"""

from collections import Counter

import torch

from steering.conditional.sampling import (
    Distribution,
    sample_k,
)


class TestSingleValueRange:
    """When min_k == max_k - 1 the range has exactly one legal value."""

    def test_geometric_single_value_always_returns_min_k(self):
        """Geometric with min_k == max_k - 1 always returns min_k."""
        rng = torch.Generator()
        rng.manual_seed(99)
        min_k, max_k = 7, 8
        results = {
            sample_k(Distribution.geometric, 5.0, 5.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(500)
        }
        assert results == {min_k}, (
            f"Expected always {min_k}, got unique values: {results}"
        )

    def test_normal_single_value_always_returns_min_k(self):
        """Normal with min_k == max_k - 1 always returns min_k."""
        rng = torch.Generator()
        rng.manual_seed(99)
        min_k, max_k = 4, 5
        results = {
            sample_k(Distribution.normal, 100.0, 50.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(500)
        }
        assert results == {min_k}, (
            f"Expected always {min_k}, got unique values: {results}"
        )


class TestZeroOneRange:
    """min_k=0, max_k=1 — the only legal value is 0."""

    def test_geometric_zero_one_always_zero(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        results = {
            sample_k(Distribution.geometric, 3.0, 3.0, rng, min_k=0, max_k=1)
            for _ in range(500)
        }
        assert results == {0}

    def test_normal_zero_one_always_zero(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        results = {
            sample_k(Distribution.normal, 50.0, 20.0, rng, min_k=0, max_k=1)
            for _ in range(500)
        }
        assert results == {0}


class TestGeometricModularArithmetic:
    """Geometric with both min_k and max_k uses (sample % range) + min_k."""

    def test_full_range_coverage(self):
        """Every value in [min_k, max_k) appears at least once over many samples."""
        rng = torch.Generator()
        rng.manual_seed(1)
        min_k, max_k = 2, 7
        results = set()
        for _ in range(10_000):
            results.add(
                sample_k(Distribution.geometric, 5.0, 5.0, rng, min_k=min_k, max_k=max_k)
            )
        expected = set(range(min_k, max_k))
        assert results == expected, (
            f"Missing values: {expected - results}, extra: {results - expected}"
        )

    def test_statistical_uniformish_coverage(self):
        """Each value in [min_k, max_k) appears at least once in 10000 samples.

        Geometric mod a small range should hit every bucket.
        """
        rng = torch.Generator()
        rng.manual_seed(7)
        min_k, max_k = 0, 5
        counts = Counter()
        n = 10_000
        for _ in range(n):
            counts[
                sample_k(Distribution.geometric, 3.0, 3.0, rng, min_k=min_k, max_k=max_k)
            ] += 1
        for val in range(min_k, max_k):
            assert counts[val] >= 1, (
                f"Value {val} never appeared in {n} samples"
            )


class TestGeometricOffsetOnly:
    """Geometric with min_k > 0 and no max_k — offset without modular wrap."""

    def test_unbounded_growth(self):
        """Without max_k, values can grow well beyond min_k."""
        rng = torch.Generator()
        rng.manual_seed(12)
        min_k = 0
        results = [
            sample_k(Distribution.geometric, 20.0, 20.0, rng, min_k=min_k)
            for _ in range(2000)
        ]
        assert max(results) > 30, (
            f"Expected large values with mean=20, max seen={max(results)}"
        )

    def test_min_k_5_all_above(self):
        """min_k=5 with no max_k: every sample >= 5."""
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.geometric, 5.0, 5.0, rng, min_k=5)
            for _ in range(2000)
        ]
        assert all(k >= 5 for k in results), (
            f"min_k=5 violated: min={min(results)}"
        )

    def test_min_k_0_no_max_raw_geometric(self):
        """min_k=0, no max_k: raw geometric (matches old allow_empty=True).

        Values start at 0 and the distribution is unbounded.
        """
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(Distribution.geometric, 3.0, 3.0, rng, min_k=0)
            for _ in range(5000)
        ]
        assert 0 in results, "Raw geometric should include 0"
        assert max(results) > 10, (
            f"Raw geometric with mean=3 should produce values > 10 over 5000 samples, "
            f"max seen={max(results)}"
        )


class TestNormalClamping:
    """Normal distribution with extreme mean/std still respects bounds."""

    def test_extreme_low_mean_clamped_to_min_k(self):
        """Mean far below min_k: all samples clamp to min_k."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 10, 20
        results = [
            sample_k(Distribution.normal, -100.0, 0.1, rng, min_k=min_k, max_k=max_k)
            for _ in range(500)
        ]
        assert all(k == min_k for k in results), (
            f"Expected all {min_k}, got unique: {set(results)}"
        )

    def test_extreme_high_mean_clamped_to_max_k_minus_1(self):
        """Mean far above max_k: all samples clamp to max_k - 1."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 0, 5
        results = [
            sample_k(Distribution.normal, 1000.0, 0.1, rng, min_k=min_k, max_k=max_k)
            for _ in range(500)
        ]
        assert all(k == max_k - 1 for k in results), (
            f"Expected all {max_k - 1}, got unique: {set(results)}"
        )

    def test_large_std_stays_in_bounds(self):
        """Very wide normal still stays in [min_k, max_k - 1]."""
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 3, 15
        results = [
            sample_k(Distribution.normal, 8.0, 1000.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(5000)
        ]
        assert all(min_k <= k <= max_k - 1 for k in results), (
            f"Out of bounds: min={min(results)}, max={max(results)}"
        )


class TestMinK0MaxK2:
    """min_k=0, max_k=2: only 0 or 1 are valid."""

    def test_geometric_only_0_or_1(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        results = set()
        for _ in range(5000):
            results.add(
                sample_k(Distribution.geometric, 2.0, 2.0, rng, min_k=0, max_k=2)
            )
        assert results <= {0, 1}, f"Got unexpected values: {results - {0, 1}}"
        assert results == {0, 1}, f"Expected both 0 and 1, got {results}"

    def test_normal_only_0_or_1(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 0, 2
        results = set()
        for _ in range(5000):
            results.add(
                sample_k(Distribution.normal, 0.5, 1.0, rng, min_k=min_k, max_k=max_k)
            )
        assert results <= {0, 1}, f"Got unexpected values: {results - {0, 1}}"


class TestDeterminism:
    """Same seed produces exactly the same sequence."""

    def test_geometric_deterministic(self):
        n = 50
        rng1 = torch.Generator()
        rng1.manual_seed(777)
        seq1 = [
            sample_k(Distribution.geometric, 3.0, 3.0, rng1, min_k=1, max_k=10)
            for _ in range(n)
        ]
        rng2 = torch.Generator()
        rng2.manual_seed(777)
        seq2 = [
            sample_k(Distribution.geometric, 3.0, 3.0, rng2, min_k=1, max_k=10)
            for _ in range(n)
        ]
        assert seq1 == seq2

    def test_normal_deterministic(self):
        n = 50
        rng1 = torch.Generator()
        rng1.manual_seed(555)
        seq1 = [
            sample_k(Distribution.normal, 5.0, 2.0, rng1, min_k=0, max_k=15)
            for _ in range(n)
        ]
        rng2 = torch.Generator()
        rng2.manual_seed(555)
        seq2 = [
            sample_k(Distribution.normal, 5.0, 2.0, rng2, min_k=0, max_k=15)
            for _ in range(n)
        ]
        assert seq1 == seq2


class TestLargeRange:
    """Large range to exercise modular arithmetic on bigger spans."""

    def test_geometric_large_range_bounded(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        min_k, max_k = 100, 500
        results = [
            sample_k(Distribution.geometric, 10.0, 10.0, rng, min_k=min_k, max_k=max_k)
            for _ in range(5000)
        ]
        assert all(min_k <= k < max_k for k in results), (
            f"Out of bounds: min={min(results)}, max={max(results)}"
        )
        # Should see a spread of values
        unique = set(results)
        assert len(unique) > 50, (
            f"Expected > 50 unique values in range of 400, got {len(unique)}"
        )
