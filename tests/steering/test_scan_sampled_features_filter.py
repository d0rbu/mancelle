"""Pin the percentile-scale invariant for scan_sampled_features (map B-17).

The filter ``pct_scores >= 1.0`` in ``scan_sampled_features`` correctly
excludes only below-p1 magnitudes, which
``magnitude_to_percentile`` maps to 0.5. This test pins the scale of
``magnitude_to_percentile`` (0-100) so a future refactor that changes the
scale to [0, 1] would produce a loud test failure rather than a silent
"keeps everything" filter.

The test is intentionally narrow: it asserts the SCALE contract of
``magnitude_to_percentile`` (slice 7's territory) and the FILTER semantic
of ``scan_sampled_features`` against synthetic data — not the full
end-to-end scan pipeline (covered elsewhere).
"""

from __future__ import annotations

import torch as t

from steering.conditional.percentile_table import (
    build_percentile_table,
    magnitude_to_percentile,
)
from steering.stats_types import PerFeatureStat


def _stats_for_feature(fid: int) -> PerFeatureStat:
    # Linear ramp p1=1.0, p2=2.0, ..., p99=99.0
    pcts = {f"p{i}": float(i) for i in range(1, 100)}
    return PerFeatureStat(
        count=100,
        pct_segments_active=0.5,
        mean_mag=50.0,
        mean_mag_incl_zeros=25.0,
        std_mag=25.0,
        min_mag=0.0,
        max_mag=100.0,
        median_mag=50.0,
        percentiles=pcts,
        histogram=None,
    )


class TestPercentileScaleInvariant:
    """Pin the scale of ``magnitude_to_percentile`` to [0.5, 99.5]."""

    def test_below_p1_maps_to_half(self) -> None:
        """Magnitude strictly below p1 → 0.5 (the "below" sentinel)."""
        stats = {"0": _stats_for_feature(0)}
        table = build_percentile_table(stats, n_sae_features=1)
        out = magnitude_to_percentile(
            t.tensor([0], dtype=t.int64),
            t.tensor([0.5], dtype=t.float32),
            table,
        )
        assert out.item() == 0.5

    def test_above_p99_maps_to_99_5(self) -> None:
        """Magnitude at or above p99 → 99.5."""
        stats = {"0": _stats_for_feature(0)}
        table = build_percentile_table(stats, n_sae_features=1)
        out = magnitude_to_percentile(
            t.tensor([0], dtype=t.int64),
            t.tensor([200.0], dtype=t.float32),
            table,
        )
        assert out.item() == 99.5

    def test_at_p1_maps_to_one(self) -> None:
        """Magnitude exactly at p1 → 1.0 (in-range, not the 0.5 below sentinel)."""
        stats = {"0": _stats_for_feature(0)}
        table = build_percentile_table(stats, n_sae_features=1)
        out = magnitude_to_percentile(
            t.tensor([0], dtype=t.int64),
            t.tensor([1.0], dtype=t.float32),
            table,
        )
        # at p1 → idx=1 (right=True), interpolation puts result at idx=1.0
        assert out.item() == 1.0

    def test_no_value_falls_in_open_unit_interval(self) -> None:
        """The output of magnitude_to_percentile is NEVER in (0.0, 1.0).

        Either it's the 0.5 below-p1 sentinel, or it's 1.0+. Pins the
        invariant that ``pct_scores >= 1.0`` filters EXACTLY the below-p1
        values (no values land in (0.5, 1.0) for the linear-ramp stats).
        """
        stats = {"0": _stats_for_feature(0)}
        table = build_percentile_table(stats, n_sae_features=1)
        # Sample a range of magnitudes
        mags = t.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 50.0, 99.0, 99.5, 200.0], dtype=t.float32)
        ids = t.zeros_like(mags, dtype=t.int64)
        out = magnitude_to_percentile(ids, mags, table)
        for v in out.tolist():
            # Either below (=0.5) or in [1.0, 99.5] — never in (0.5, 1.0)
            assert v == 0.5 or v >= 1.0


class TestScanSampledFeaturesKeepMask:
    """Pin the filter semantics on a synthetic shard."""

    def test_filter_excludes_below_p1_values(self) -> None:
        """Only entries with magnitude >= p1 survive the keep_mask."""
        stats = {"0": _stats_for_feature(0), "1": _stats_for_feature(1)}
        table = build_percentile_table(stats, n_sae_features=2)

        # Build a synthetic batch: half below p1, half above
        feature_ids = t.tensor([0, 1, 0, 1], dtype=t.int64)
        feature_mags = t.tensor([0.5, 0.5, 50.0, 50.0], dtype=t.float32)
        pct_scores = magnitude_to_percentile(feature_ids, feature_mags, table)
        keep_mask = pct_scores >= 1.0

        # First two are below p1 → 0.5 → drop
        # Last two are at p50 → 50.0 → keep
        assert keep_mask.tolist() == [False, False, True, True]

    def test_filter_keeps_values_at_or_above_p1(self) -> None:
        """Magnitude exactly at p1 → kept (>= 1.0 boundary)."""
        stats = {"0": _stats_for_feature(0)}
        table = build_percentile_table(stats, n_sae_features=1)

        # p1 = 1.0 in our linear-ramp stats
        ids = t.tensor([0, 0, 0], dtype=t.int64)
        mags = t.tensor([0.999, 1.0, 1.001], dtype=t.float32)
        pct_scores = magnitude_to_percentile(ids, mags, table)
        keep_mask = pct_scores >= 1.0

        assert keep_mask.tolist() == [False, True, True]
