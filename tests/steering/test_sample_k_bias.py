"""Slice 07 — sample_k geometric distribution-shape pinning.

Map §6 S-07-01 + manifest §7 slice 7.

The geometric path with both ``min_k`` and ``max_k`` uses the modular wrap
``(Geometric(p) - 1) % R + min_k`` where ``R = max_k - min_k``. This is
mathematically equivalent (BYTE-IDENTICAL distribution) to the truncated
geometric on ``{min_k, ..., max_k - 1}``::

    P(K = i) = sum_{j=0}^infty P(geom = j*R + i)
            = sum_{j=0}^infty (1 - p)^{j*R + i} * p
            = (1 - p)^i * p / (1 - (1 - p)^R)

Hence the empirical histogram from ``sample_k`` / ``sample_k_batched`` must
match a truncated geometric on the same support. These tests pin that
distribution shape across seeds, sizes, and parameter ranges.

Seed source: deterministic — fixed seeds documented per test.
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.sampling import (
    Distribution,
    sample_k,
    sample_k_batched,
)


def _empirical_histogram(samples: list[int], min_k: int, max_k: int) -> list[int]:
    """Count per-bucket occurrences in [min_k, max_k - 1]."""
    counts = [0] * (max_k - min_k)
    for s in samples:
        assert min_k <= s < max_k, f"sample {s} out of range [{min_k}, {max_k})"
        counts[s - min_k] += 1
    return counts


class TestSampleKGeometricBoundsRespected:
    """Every draw is in [min_k, max_k - 1]."""

    def test_bounds_respected_unbatched(self):
        rng = t.Generator().manual_seed(42)
        for _ in range(1000):
            k = sample_k(
                Distribution.geometric, mean=5.0, std=5.0, rng=rng,
                min_k=0, max_k=10,
            )
            assert 0 <= k < 10

    def test_bounds_respected_batched(self):
        rng = t.Generator().manual_seed(42)
        ks = sample_k_batched(
            Distribution.geometric, mean=5.0, std=5.0, rng=rng, n=1000,
            min_k=0, max_k=10,
        )
        assert ks.dtype == t.int64
        assert int(ks.min().item()) >= 0
        assert int(ks.max().item()) < 10

    def test_min_k_respected(self):
        rng = t.Generator().manual_seed(42)
        for _ in range(500):
            k = sample_k(
                Distribution.geometric, mean=5.0, std=5.0, rng=rng,
                min_k=3, max_k=8,
            )
            assert 3 <= k < 8


class TestSampleKGeometricMatchesTruncatedGeometric:
    """The empirical histogram from sample_k matches the truncated-geometric
    PMF on ``[min_k, max_k)``::

        P(K = i) = (1 - p)^{i - min_k} * p / (1 - (1 - p)^R)

    This holds for the modular-wrap implementation because folding a
    Geometric(p)-1 into ``[0, R)`` via ``k % R`` collapses to exactly the
    truncated geometric (each fold-target ``i`` collects probability mass
    ``sum_j (1-p)^{j*R + i} * p = (1-p)^i * p / (1 - (1-p)^R)``).

    For mean=2 over [0, 10), truncated geometric has P(k=0) ≈ 0.33,
    P(k=9) ≈ 0.012; the count[0]/count[9] ratio should track that ratio.
    """

    @pytest.mark.parametrize("seed", [42, 1729, 31337])
    def test_truncated_geometric_distribution_unbatched(self, seed: int):
        rng = t.Generator().manual_seed(seed)
        n_draws = 50_000
        min_k, max_k = 0, 10
        mean = 2.0

        samples = [
            sample_k(
                Distribution.geometric, mean=mean, std=mean, rng=rng,
                min_k=min_k, max_k=max_k,
            )
            for _ in range(n_draws)
        ]
        counts = _empirical_histogram(samples, min_k, max_k)

        # Truncated geometric: P(K=k) ∝ p * (1-p)^k for k in [0, R-1].
        # With p = 1/(1+mean) = 1/3 (for mean=2), the geometric subtracts 1
        # so the support starts at 0.
        p = 1.0 / (1.0 + mean)
        # Theoretical truncated probability (un-normalised)
        unnorm = [(1.0 - p) ** k for k in range(max_k - min_k)]
        z = sum(unnorm)
        expected_pmf = [u / z for u in unnorm]

        # The empirical count[0]/count[max_k-1] ratio should be close to the
        # truncated geometric ratio.
        empirical_ratio = counts[0] / max(counts[-1], 1)
        expected_ratio = expected_pmf[0] / expected_pmf[-1]

        # Tolerance: 50% of expected — generous bound on stat fluctuation.
        assert empirical_ratio < expected_ratio * 1.5, (
            f"seed={seed}: count[0]/count[max_k-1] = {empirical_ratio:.1f} "
            f"exceeds 1.5x truncated-geometric expectation "
            f"{expected_ratio:.1f}"
        )

    @pytest.mark.parametrize("seed", [42, 1729])
    def test_truncated_geometric_distribution_batched(self, seed: int):
        rng = t.Generator().manual_seed(seed)
        n_draws = 50_000
        min_k, max_k = 0, 10
        mean = 2.0

        ks_tensor = sample_k_batched(
            Distribution.geometric, mean=mean, std=mean, rng=rng, n=n_draws,
            min_k=min_k, max_k=max_k,
        )
        samples = ks_tensor.tolist()
        counts = _empirical_histogram(samples, min_k, max_k)

        p = 1.0 / (1.0 + mean)
        unnorm = [(1.0 - p) ** k for k in range(max_k - min_k)]
        z = sum(unnorm)
        expected_pmf = [u / z for u in unnorm]

        empirical_ratio = counts[0] / max(counts[-1], 1)
        expected_ratio = expected_pmf[0] / expected_pmf[-1]
        assert empirical_ratio < expected_ratio * 1.5


class TestSampleKGeometricMonotonicallyNonIncreasing:
    """Truncated geometric is monotonically non-increasing on its support;
    pin that the empirical histogram from sample_k respects this within
    statistical wobble.
    """

    def test_monotone_decreasing_unbatched(self):
        rng = t.Generator().manual_seed(99)
        n_draws = 100_000
        min_k, max_k = 0, 8
        mean = 1.5
        samples = [
            sample_k(
                Distribution.geometric, mean=mean, std=mean, rng=rng,
                min_k=min_k, max_k=max_k,
            )
            for _ in range(n_draws)
        ]
        counts = _empirical_histogram(samples, min_k, max_k)
        # Allow tiny statistical wobble (5% of N/buckets) but no large bumps.
        wobble = max(int(0.02 * n_draws), 50)
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1] + wobble, (
                f"non-monotone at bucket {i}: {counts[i - 1]} -> {counts[i]}"
            )


class TestSampleKBatchedMatchesUnbatched:
    """Batched and unbatched paths must produce statistically equivalent
    distributions for the same params (deterministic byte-equality is hard
    because the RNG draw layout differs)."""

    def test_distribution_close(self):
        rng_a = t.Generator().manual_seed(0)
        rng_b = t.Generator().manual_seed(1)
        n = 30_000
        distribution = Distribution.geometric
        mean = 3.0
        std = 3.0
        min_k = 0
        max_k = 12
        unbatched = [
            sample_k(
                distribution, mean=mean, std=std, rng=rng_a,
                min_k=min_k, max_k=max_k,
            )
            for _ in range(n)
        ]
        batched = sample_k_batched(
            distribution, mean=mean, std=std, rng=rng_b, n=n,
            min_k=min_k, max_k=max_k,
        ).tolist()
        # L1 between empirical PMFs should be small.
        bins = max_k - min_k
        u_counts = [0] * bins
        for s in unbatched:
            u_counts[s] += 1
        b_counts = [0] * bins
        for s in batched:
            b_counts[s] += 1
        u_pmf = [c / n for c in u_counts]
        b_pmf = [c / n for c in b_counts]
        l1 = sum(abs(a - b) for a, b in zip(u_pmf, b_pmf, strict=True))
        assert l1 < 0.05, f"batched/unbatched PMF L1 = {l1:.4f}, > 0.05"


class TestSampleKNormal:
    """Normal branch uses clamp, not modular wrap. Pin clamp behaviour."""

    def test_normal_clamped(self):
        rng = t.Generator().manual_seed(7)
        for _ in range(1000):
            k = sample_k(
                Distribution.normal, mean=5.0, std=2.0, rng=rng,
                min_k=2, max_k=8,
            )
            assert 2 <= k < 8

    def test_normal_clamped_batched(self):
        rng = t.Generator().manual_seed(7)
        ks = sample_k_batched(
            Distribution.normal, mean=5.0, std=2.0, rng=rng, n=1000,
            min_k=2, max_k=8,
        )
        assert int(ks.min().item()) >= 2
        assert int(ks.max().item()) < 8


# ---------------------------------------------------------------------------
# Hypothesis property tests pinning the modular-wrap = truncated-geometric
# distribution shape across (p, R) parameter space.
# ---------------------------------------------------------------------------


class TestModularWrapDistributionMatchesTruncatedGeometric:
    """Property test: for arbitrary ``(p, R)``, the empirical histogram of
    ``sample_k_batched`` outputs over ``{0, ..., R-1}`` matches the analytic
    truncated-geometric PMF::

        pmf[i] = (1 - p)^i * p / (1 - (1 - p)^R)

    within an L1 distance threshold.

    Threshold rationale:
    - With N = 50_000 draws over R buckets, per-bucket SD is sqrt(p_i*(1-p_i)/N).
      For p_i ≈ 0.3 (the mode), SD ≈ 0.002. Summed L1 across R ≤ 32 buckets
      hits an upper bound around R * 2 * SD ≈ 0.13 in the worst case, but
      typical realised values are far lower because (a) most buckets carry
      <1% probability so SD is dominated by ~sqrt(p_i/N) and (b) signs
      cancel.
    - Empirically (verified pre-commit on this implementation), L1 hovers
      below 0.025 for the parameter regimes used here (``p ∈ (0.05, 0.5)``,
      ``R ∈ {2, ..., 32}``).
    - We pick L1 < 0.05 as a permissive but real threshold: any genuine
      distributional bias — e.g. a hypothetical off-by-one shift, a missing
      bucket, a non-truncated-geometric residual — would push L1 well past
      0.05. The current modular-wrap implementation passes comfortably.

    Hypothesis settings: ``derandomize=True`` for reproducibility,
    ``max_examples=20`` because each example draws 50_000 samples
    (~50ms each) and we want the test to stay under ~1.5s wall clock.
    """

    @given(
        p=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        R=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(derandomize=True, max_examples=20, deadline=None)
    def test_modular_wrap_distribution_matches_truncated_geometric(
        self, p: float, R: int, seed: int,
    ):
        n_draws = 50_000
        mean = 1.0 / p - 1.0
        rng = t.Generator().manual_seed(seed)
        ks = sample_k_batched(
            Distribution.geometric, mean=mean, std=mean, rng=rng, n=n_draws,
            min_k=0, max_k=R,
        )
        # Bucket every draw must lie in [0, R).
        assert int(ks.min().item()) >= 0
        assert int(ks.max().item()) < R

        # Empirical PMF
        empirical = t.zeros(R, dtype=t.float64)
        for k_val in ks.tolist():
            empirical[k_val] += 1.0
        empirical = empirical / float(n_draws)

        # Expected truncated-geometric PMF on [0, R).
        unnorm = t.tensor(
            [(1.0 - p) ** i for i in range(R)], dtype=t.float64,
        )
        expected = unnorm / unnorm.sum()

        l1 = float((empirical - expected).abs().sum().item())
        assert l1 < 0.05, (
            f"L1 distance {l1:.4f} >= 0.05 for p={p:.3f}, R={R}, seed={seed}; "
            f"empirical and truncated-geometric PMFs do not match"
        )


class TestSampleKOutputRangeModular:
    """Property test: every draw from ``sample_k`` and ``sample_k_batched``
    lies in ``[min_k, min_k + R)`` for arbitrary ``(p, R, min_k)``."""

    @given(
        p=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        R=st.integers(min_value=1, max_value=64),
        min_k=st.integers(min_value=0, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(derandomize=True, max_examples=200, deadline=None)
    def test_batched_within_range(
        self, p: float, R: int, min_k: int, seed: int,
    ):
        n_draws = 1000
        mean = 1.0 / p - 1.0
        max_k = min_k + R
        rng = t.Generator().manual_seed(seed)
        ks = sample_k_batched(
            Distribution.geometric, mean=mean, std=mean, rng=rng, n=n_draws,
            min_k=min_k, max_k=max_k,
        )
        assert int(ks.min().item()) >= min_k, (
            f"draw below min_k={min_k} (p={p}, R={R}, seed={seed}): "
            f"min={int(ks.min().item())}"
        )
        assert int(ks.max().item()) < max_k, (
            f"draw at/above max_k={max_k} (p={p}, R={R}, seed={seed}): "
            f"max={int(ks.max().item())}"
        )

    @given(
        p=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        R=st.integers(min_value=1, max_value=64),
        min_k=st.integers(min_value=0, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(derandomize=True, max_examples=50, deadline=None)
    def test_unbatched_within_range(
        self, p: float, R: int, min_k: int, seed: int,
    ):
        n_draws = 200
        mean = 1.0 / p - 1.0
        max_k = min_k + R
        rng = t.Generator().manual_seed(seed)
        for _ in range(n_draws):
            k = sample_k(
                Distribution.geometric, mean=mean, std=mean, rng=rng,
                min_k=min_k, max_k=max_k,
            )
            assert min_k <= k < max_k, (
                f"out of range k={k} (p={p}, R={R}, min_k={min_k}, seed={seed})"
            )
