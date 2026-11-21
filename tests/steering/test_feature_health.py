"""Tests for steering.feature_health — degeneracy check logic."""

from __future__ import annotations

import random

import pytest
import torch as t

from errors import DegenerateFeatureStatisticsError
from steering.feature_health import (
    MIN_SEGMENTS_FOR_DEGENERACY_CHECK,
    _interpolated_median,
    assert_feature_statistics_not_degenerate,
    firing_health,
)


def test_degenerate_raises():
    with pytest.raises(DegenerateFeatureStatisticsError, match="issue #549"):
        assert_feature_statistics_not_degenerate(
            median_firing_frequency=1.0, n_features_firing_gt_99pct=6000,
            n_features_seen=7000, n_sae_features=65536, n_segments=130_000,
            allow_degenerate=False,
        )


def test_healthy_passes():
    assert_feature_statistics_not_degenerate(
        median_firing_frequency=0.005, n_features_firing_gt_99pct=21,
        n_features_seen=57000, n_sae_features=65536, n_segments=20_000,
        allow_degenerate=False,
    )


def test_min_segments_gate_suppresses_check():
    # tiny corpus → no hard-fail even if median freq is high
    assert_feature_statistics_not_degenerate(
        median_firing_frequency=1.0, n_features_firing_gt_99pct=10,
        n_features_seen=12, n_sae_features=64,
        n_segments=MIN_SEGMENTS_FOR_DEGENERACY_CHECK - 1, allow_degenerate=False,
    )


def test_allow_degenerate_escape_hatch():
    assert_feature_statistics_not_degenerate(
        median_firing_frequency=1.0, n_features_firing_gt_99pct=6000,
        n_features_seen=7000, n_sae_features=65536, n_segments=130_000,
        allow_degenerate=True,
    )


# ---------------------------------------------------------------------------
# Coverage-gap tests (issue #549 hardening audit)
# ---------------------------------------------------------------------------


def test_ubiquitous_features_gate_raises_when_median_diluted():
    """The ubiquitous-feature signal fires even when median firing frequency is
    below the threshold, provided the fraction of always-on features is high.

    Scenario: median=0.10 (< 0.5 threshold) but 6000/8000 = 75% of seen
    features fire in >99% of segments — well above DEGENERATE_UBIQUITOUS_FEATURE_FRACTION.

    Non-vacuity: removing the ``ubiquitous_degenerate`` branch from the
    ``if median_degenerate or ubiquitous_degenerate:`` guard makes this call
    pass silently without raising, and the test fails.
    """
    with pytest.raises(DegenerateFeatureStatisticsError):
        assert_feature_statistics_not_degenerate(
            median_firing_frequency=0.10,
            n_features_firing_gt_99pct=6000,
            n_features_seen=8000,
            n_sae_features=65536,
            n_segments=130_000,
            allow_degenerate=False,
        )


def test_ubiquitous_features_gate_passes_when_few_always_on():
    """Healthy corpus: median=0.03, only 83/16306 features (≈0.5%) fire in
    >99% of segments — well below DEGENERATE_UBIQUITOUS_FEATURE_FRACTION (50%).

    Non-vacuity: if we changed ``n_features_firing_gt_99pct`` to 8500 (> 50%
    of 16306), the ubiquitous gate would fire and raise instead.
    """
    assert_feature_statistics_not_degenerate(
        median_firing_frequency=0.03,
        n_features_firing_gt_99pct=83,
        n_features_seen=16306,
        n_sae_features=65536,
        n_segments=130_000,
        allow_degenerate=False,
    )


# ---------------------------------------------------------------------------
# firing_health single-source + interpolated median + >= boundary (Round 2 / F5)
# ---------------------------------------------------------------------------


def test_firing_health_returns_median_and_gt99_count():
    # 4 seen features over 100 segments: freqs [1.0, 1.0, 0.5, 0.01]
    counts = t.tensor([100, 100, 50, 1])
    median, n_gt99 = firing_health(counts, n_segments=100)
    assert median == pytest.approx(0.75)   # interpolated median of [0.01,0.5,1.0,1.0]
    assert n_gt99 == 2                      # the two freq==1.0 features


def test_firing_health_empty_and_zero_segments():
    assert firing_health(t.tensor([], dtype=t.long), 1000) == (0.0, 0)
    assert firing_health(t.tensor([100]), 0) == (0.0, 0)


def test_median_uses_interpolation_not_torch_lower_median():
    # even count: torch.median would return the LOWER middle (0.001), the true
    # interpolated median is ~0.5005. counts=[1,1,1000,1000] over 1000 segments.
    counts = t.tensor([1, 1, 1000, 1000])
    median = firing_health(counts, n_segments=1000)[0]
    assert median == pytest.approx((0.001 + 1.0) / 2, abs=1e-6)


def test_interpolated_median_byte_identical_to_quantile():
    # FUZZ firing_health's exact domain (freq = integer counts / n_segments) for
    # odd AND even lengths, asserting EXACT equality to torch.quantile(x, 0.5).
    # A handful of cherry-picked vectors hide the bug: torch.quantile's lerp
    # >=0.5 branch is `upper - (upper-lower)*0.5`, NOT `lower + (upper-lower)*0.5`
    # — the wrong arm diverges ~1 ULP on ~4% of even-count inputs, which would
    # silently change the persisted median_feature_firing_frequency and could
    # flip a borderline degeneracy verdict at the 0.5 threshold.
    rng = random.Random(0xC0FFEE)
    for _ in range(3000):
        k = rng.choice([1, 2, 3, 4, 5, 6, 8, 16, 50, 257])
        n_segments = rng.randint(1000, 5_000_000)
        counts = t.tensor(
            [rng.randint(1, n_segments) for _ in range(k)], dtype=t.int64
        )
        freq = counts.to(t.float64) / float(n_segments)
        assert _interpolated_median(freq) == float(t.quantile(freq, 0.5)), (
            f"median mismatch for counts={counts.tolist()} n_segments={n_segments}"
        )


def test_interpolated_median_no_size_ceiling():
    # torch.quantile raises ("input tensor is too large") above 2**24 elements;
    # _interpolated_median routes around it via torch.kthvalue. This guard stays in
    # the DEFAULT suite (NOT a @slow lane) — a revert to torch.quantile would crash
    # right here on the precise #549 ceiling the median change exists to dodge, and
    # the default fuzz above only exercises tiny n. float32 keeps the allocation
    # modest (~67MB) and kthvalue is O(n) with no sort, so this is cheap enough to
    # run every time (the prior @slow marker removed the only ceiling guard).
    n = 2**24 + 1  # one past torch.quantile's ceiling
    assert _interpolated_median(t.ones(n, dtype=t.float32)) == 1.0


def test_ubiquitous_gate_fires_at_exactly_half():  # the >= boundary
    # exactly 50% of seen features always-on, median diluted below 0.5
    with pytest.raises(DegenerateFeatureStatisticsError):
        assert_feature_statistics_not_degenerate(
            median_firing_frequency=0.10, n_features_firing_gt_99pct=5000,
            n_features_seen=10000, n_sae_features=65536, n_segments=130000,
            allow_degenerate=False,
        )
