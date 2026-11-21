"""Tests for viz._charts pure chart functions."""

from __future__ import annotations

import dataclasses
from typing import Any

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from steering.segmentation import SegmentMode
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)
from viz._charts import (
    count_distribution,
    count_vs_magnitude,
    features_per_segment,
    features_per_token_ratio,
    magnitude_distribution,
    percentile_heatmap,
    percentile_surface_3d,
    segment_length_distribution,
)
from viz.plot_features import CHART_REGISTRY


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures: synthetic ExtractionStatistics dataclasses
# ---------------------------------------------------------------------------

PERCENTILE_KEYS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "p99"]


def _make_per_feature_stats(
    n: int = 100,
    *,
    with_histograms: bool = False,
) -> dict[str, PerFeatureStat]:
    """Generate synthetic per_feature_stats dict with n features."""
    stats: dict[str, PerFeatureStat] = {}
    for i in range(n):
        count = (i + 1) * 10  # 10, 20, ..., 1000
        mean_mag = 0.5 + i * 0.01
        histogram: dict[str, object] | None = (
            {"counts": [count // 50 + 1] * 50} if with_histograms else None
        )
        stats[str(i)] = PerFeatureStat(
            count=count,
            pct_segments_active=round(count / 1000 * 100, 4),
            mean_mag=mean_mag,
            mean_mag_incl_zeros=mean_mag * 0.5,
            std_mag=mean_mag * 0.2,
            min_mag=mean_mag * 0.1,
            max_mag=mean_mag * 3.0,
            median_mag=mean_mag * 0.9,
            percentiles={
                k: mean_mag * (0.1 + 0.09 * j) for j, k in enumerate(PERCENTILE_KEYS)
            },
            histogram=histogram,
        )
    return stats


def _make_count_percentiles() -> dict[str, float | int]:
    """Matching count percentiles for the synthetic data."""
    return {
        "p10": 100,
        "p20": 200,
        "p30": 300,
        "p40": 400,
        "p50": 500,
        "p60": 600,
        "p70": 700,
        "p80": 800,
        "p90": 900,
        "p99": 990,
    }


def _make_mag_histogram_bin_edges() -> list[float]:
    """Generate 51 bin edges for magnitude histograms."""
    return [round(i * 0.06, 6) for i in range(51)]  # 0.0, 0.06, ..., 3.0


def _make_histogram() -> dict[str, Any]:
    """Build a generic histogram dict."""
    return {
        "bin_edges": [float(i * 4) for i in range(51)],
        "counts": [20] * 50,
    }


_DEFAULT_METADATA = MetadataStats(
    pooling_mode="mean_pool",
    segment_mode=SegmentMode.fixed_tokens,
    segment_size=128,
    context_window=1024,
    layer=0,
    n_sae_features=0,
    generated_at="",
)

_DEFAULT_CORPUS = CorpusStats(
    n_shards_processed=0,
    n_docs=0,
    n_tokens_gpt2=0,
    mean_tokens_per_doc=0.0,
    median_tokens_per_doc=0,
)

_DEFAULT_STORAGE = StorageStats(
    total_bytes=0,
    mean_bytes_per_doc=0.0,
    mean_bytes_per_token=0.0,
)

_DEFAULT_EXTRAPOLATION = Extrapolation(
    target_tokens=0,
    estimated_docs=0,
    estimated_storage_gb=0.0,
    estimated_wall_time_hours=0.0,
)


def _make_stats(
    *,
    n_features: int = 100,
    with_histograms: bool = False,
    with_segment_length: bool = False,
    with_features_per_token: bool = False,
    n_segments: int = 1000,
) -> ExtractionStatistics:
    """Build a complete ExtractionStatistics for testing."""
    per_feature = _make_per_feature_stats(n_features, with_histograms=with_histograms)

    seg_length_histogram: dict[str, object] | None = None
    seg_length_percentiles: dict[str, float] = {}
    if with_segment_length:
        seg_length_histogram = {
            "bin_edges": [float(i * 3) for i in range(51)],
            "counts": [20] * 50,
        }
        seg_length_percentiles = {
            "p1": 5.0,
            "p5": 15.0,
            "p10": 30.0,
            "p25": 60.0,
            "p50": 75.0,
            "p75": 90.0,
            "p90": 120.0,
            "p95": 135.0,
            "p99": 148.0,
        }

    fpt_histogram: dict[str, object] | None = None
    fpt_percentiles: dict[str, float] = {}
    if with_features_per_token:
        fpt_histogram = {
            "bin_edges": [round(i * 0.2, 4) for i in range(51)],
            "counts": [20] * 50,
        }
        fpt_percentiles = {
            "p1": 0.3,
            "p5": 1.0,
            "p10": 2.0,
            "p25": 3.5,
            "p50": 5.0,
            "p75": 6.5,
            "p90": 8.0,
            "p95": 9.0,
            "p99": 9.8,
        }

    features = SegmentFeatureStats(
        n_segments=n_segments,
        n_features_seen=n_features,
        mean_active_per_segment=500.0,
        median_active_per_segment=450,
        p90_active_per_segment=800,
        max_active_per_segment=1000,
        min_active_per_segment=100,
        n_total_sae_features=0,
        pct_features_active=0.0,
        count_percentiles=_make_count_percentiles(),
        magnitude_stats={
            "mean": 0.5,
            "std": 0.1,
            "min": 0.01,
            "max": 3.0,
            "percentiles": {},
        },
        active_per_segment_histogram=_make_histogram() if with_histograms else None,
        active_per_segment_percentiles={"p10": 100.0, "p50": 500.0, "p90": 900.0},
        mag_histogram_bin_edges=_make_mag_histogram_bin_edges()
        if with_histograms
        else None,
        segment_length_histogram=seg_length_histogram,
        segment_length_percentiles=seg_length_percentiles or None,
        features_per_token_histogram=fpt_histogram,
        features_per_token_percentiles=fpt_percentiles or None,
    )

    return ExtractionStatistics(
        metadata=_DEFAULT_METADATA,
        corpus=_DEFAULT_CORPUS,
        features=features,
        storage=_DEFAULT_STORAGE,
        per_feature_stats=per_feature,
        extrapolation_10b=_DEFAULT_EXTRAPOLATION,
    )


# ---------------------------------------------------------------------------
# Tests: count_distribution
# ---------------------------------------------------------------------------


class TestCountDistribution:
    def test_returns_figure(self):
        fig = count_distribution(_make_stats())
        assert isinstance(fig, Figure)

    def test_log_scale_sets_xscale(self):
        fig = count_distribution(_make_stats(), log_scale=True)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"

    def test_linear_scale(self):
        fig = count_distribution(_make_stats(), log_scale=False)
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"

    def test_has_percentile_lines(self):
        fig = count_distribution(_make_stats())
        ax = fig.axes[0]
        lines = list(ax.get_lines())
        assert len(lines) >= 3

    def test_has_title_and_labels(self):
        fig = count_distribution(_make_stats())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "% of Segments Active"
        assert ax.get_ylabel() == "Number of Features"
        assert "1,000 segments" in ax.get_title()

    def test_title_contains_segment_count(self):
        fig = count_distribution(_make_stats(n_segments=284269))
        ax = fig.axes[0]
        assert "284,269 segments" in ax.get_title()

    def test_percentile_lines_are_percentages(self):
        stats = _make_stats()
        n_seg = stats.features.n_segments
        cp = stats.features.count_percentiles
        fig = count_distribution(stats)
        ax = fig.axes[0]
        # The p10 line should be at cp["p10"] / n_seg * 100 = 100/1000*100 = 10.0
        lines = ax.get_lines()
        xdata_values = [line.get_xdata()[0] for line in lines]  # ty: ignore[not-subscriptable]
        expected_p10_pct = cp["p10"] / n_seg * 100
        assert any(abs(x - expected_p10_pct) < 0.01 for x in xdata_values)  # ty: ignore[unsupported-operator]


# ---------------------------------------------------------------------------
# Tests: count_vs_magnitude
# ---------------------------------------------------------------------------


class TestCountVsMagnitude:
    def test_returns_figure(self):
        fig = count_vs_magnitude(_make_stats())
        assert isinstance(fig, Figure)

    def test_log_x(self):
        fig = count_vs_magnitude(_make_stats(), log_x=True)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"

    def test_linear_x(self):
        fig = count_vs_magnitude(_make_stats(), log_x=False)
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"

    def test_has_colorbar(self):
        fig = count_vs_magnitude(_make_stats())
        assert len(fig.axes) >= 2

    def test_has_title_and_labels(self):
        fig = count_vs_magnitude(_make_stats())
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""


# ---------------------------------------------------------------------------
# Tests: percentile_heatmap (magnitude density or legacy)
# ---------------------------------------------------------------------------


class TestPercentileHeatmap:
    def test_returns_figure_legacy(self):
        fig = percentile_heatmap(_make_stats())
        assert isinstance(fig, Figure)

    def test_returns_figure_with_histograms(self):
        fig = percentile_heatmap(_make_stats(with_histograms=True))
        assert isinstance(fig, Figure)

    def test_density_mode_has_correct_title(self):
        fig = percentile_heatmap(_make_stats(with_histograms=True))
        ax = fig.axes[0]
        assert "Density" in ax.get_title()

    def test_legacy_mode_without_bin_edges(self):
        # Provide stats with no mag_histogram_bin_edges to trigger legacy path
        stats = _make_stats()
        fig = percentile_heatmap(stats)
        ax = fig.axes[0]
        assert "Magnitude Profiles" in ax.get_title()

    def test_n_features_caps_at_available(self):
        fig = percentile_heatmap(_make_stats(n_features=5), n_features=21)
        assert isinstance(fig, Figure)

    def test_has_colorbar(self):
        fig = percentile_heatmap(_make_stats())
        assert len(fig.axes) >= 2

    def test_log_scale_density(self):
        fig = percentile_heatmap(_make_stats(with_histograms=True), log_scale=True)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Tests: percentile_surface_3d (magnitude density or legacy)
# ---------------------------------------------------------------------------


class TestPercentileSurface3d:
    def test_returns_figure_legacy(self):
        fig = percentile_surface_3d(_make_stats())
        assert isinstance(fig, Figure)

    def test_returns_figure_with_histograms(self):
        fig = percentile_surface_3d(_make_stats(with_histograms=True))
        assert isinstance(fig, Figure)

    def test_has_3d_axes(self):
        fig = percentile_surface_3d(_make_stats())
        ax = fig.axes[0]
        assert ax.name == "3d"

    def test_density_mode_has_correct_labels(self):
        fig = percentile_surface_3d(_make_stats(with_histograms=True))
        ax = fig.axes[0]
        assert "Density" in ax.get_title()
        assert "Magnitude" in ax.get_ylabel()

    def test_legacy_mode_labels(self):
        fig = percentile_surface_3d(_make_stats())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Magnitude Percentile"
        assert ax.get_ylabel() == "Feature (by count)"
        assert ax.get_zlabel() == "Normalized Magnitude"  # ty: ignore[unresolved-attribute]

    def test_log_scale_density(self):
        fig = percentile_surface_3d(_make_stats(with_histograms=True), log_scale=True)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Tests: features_per_segment
# ---------------------------------------------------------------------------


class TestFeaturesPerSegment:
    def _stats_with_histogram(self) -> ExtractionStatistics:
        base = _make_stats(with_histograms=True)
        features = dataclasses.replace(
            base.features,
            n_segments=1000,
            mean_active_per_segment=50.0,
            median_active_per_segment=48,
            p90_active_per_segment=85,
            max_active_per_segment=200,
            min_active_per_segment=10,
            active_per_segment_histogram={
                "bin_edges": [float(i * 4) for i in range(51)],
                "counts": [20] * 50,
            },
            active_per_segment_percentiles={
                "p1": 2.0,
                "p5": 10.0,
                "p10": 20.0,
                "p25": 37.5,
                "p50": 48.0,
                "p75": 62.5,
                "p90": 85.0,
                "p95": 95.0,
                "p99": 198.0,
            },
        )
        return dataclasses.replace(base, features=features)

    def _stats_without_histogram(self) -> ExtractionStatistics:
        base = _make_stats()
        features = dataclasses.replace(
            base.features,
            n_segments=500,
            mean_active_per_segment=30.0,
            median_active_per_segment=28,
            p90_active_per_segment=55,
            max_active_per_segment=100,
            min_active_per_segment=5,
            active_per_segment_histogram=None,
        )
        return dataclasses.replace(base, features=features)

    def test_returns_figure_with_histogram(self):
        fig = features_per_segment(self._stats_with_histogram())
        assert isinstance(fig, Figure)

    def test_returns_figure_without_histogram(self):
        fig = features_per_segment(self._stats_without_histogram())
        assert isinstance(fig, Figure)

    def test_has_title_with_segment_count(self):
        fig = features_per_segment(self._stats_with_histogram())
        ax = fig.axes[0]
        assert "1,000 segments" in ax.get_title()

    def test_has_axis_labels(self):
        fig = features_per_segment(self._stats_with_histogram())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Number of Active Features"
        assert ax.get_ylabel() == "Number of Segments"

    def test_has_reference_lines(self):
        fig = features_per_segment(self._stats_with_histogram())
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 3  # mean, median, p90

    def test_has_bars_with_histogram(self):
        fig = features_per_segment(self._stats_with_histogram())
        ax = fig.axes[0]
        assert len(ax.patches) == 50  # 50 bars from 50 bin counts

    def test_fallback_without_histogram_shows_text(self):
        fig = features_per_segment(self._stats_without_histogram())
        ax = fig.axes[0]
        # No bars expected, but text should be present
        assert len(ax.patches) == 0
        texts = ax.texts
        assert len(texts) >= 1

    def test_custom_n_bins_accepted(self):
        """n_bins is accepted but only used if histogram is rebuilt (future use)."""
        fig = features_per_segment(self._stats_with_histogram(), n_bins=25)
        assert isinstance(fig, Figure)

    def test_rejects_non_numeric_histogram_edges(self):
        stats = self._stats_with_histogram()
        malformed_histogram: dict[str, object] = {
            "bin_edges": ["not-a-number"],
            "counts": [],
        }
        malformed = dataclasses.replace(
            stats,
            features=dataclasses.replace(
                stats.features,
                active_per_segment_histogram=malformed_histogram,
            ),
        )

        with pytest.raises(
            TypeError,
            match=r"active_per_segment_histogram\.bin_edges must be a list of numbers",
        ):
            features_per_segment(malformed)

    def test_rejects_mismatched_histogram_lengths(self):
        stats = self._stats_with_histogram()
        malformed_histogram: dict[str, object] = {
            "bin_edges": [0.0, 1.0, 2.0],
            "counts": [1],
        }
        malformed = dataclasses.replace(
            stats,
            features=dataclasses.replace(
                stats.features,
                active_per_segment_histogram=malformed_histogram,
            ),
        )

        with pytest.raises(
            ValueError,
            match="must have one more bin edge than count",
        ):
            features_per_segment(malformed)


# ---------------------------------------------------------------------------
# Tests: magnitude_distribution
# ---------------------------------------------------------------------------


def _make_stats_with_low_activity(
    n: int = 100,
    n_low: int = 20,
    threshold: float = 1.0,
) -> ExtractionStatistics:
    """Build stats with some low-activity features in per_feature_stats."""
    stats = _make_stats(n_features=n)
    pfs = dict(stats.per_feature_stats)
    for i in range(n_low):
        pfs[str(i)] = dataclasses.replace(
            pfs[str(i)], pct_segments_active=threshold * 0.5
        )
    return dataclasses.replace(stats, per_feature_stats=pfs)


class TestMagnitudeDistribution:
    def test_returns_figure(self):
        fig = magnitude_distribution(_make_stats())
        assert isinstance(fig, Figure)

    def test_has_two_subplots(self):
        fig = magnitude_distribution(_make_stats())
        axes = fig.axes
        assert len(axes) == 2

    def test_right_panel_filters_low_activity(self):
        stats = _make_stats_with_low_activity(n=100, n_low=20)
        fig = magnitude_distribution(stats, n_bins=10, low_activity_threshold_pct=1.0)
        ax_all, ax_high = fig.axes

        # Left subplot should have all 100 features worth of data
        all_patches = ax_all.patches
        total_all = sum(p.get_height() for p in all_patches)  # ty: ignore[unresolved-attribute]
        assert total_all == 100

        # Right subplot should exclude 20 low-activity features -> 80 total
        high_patches = ax_high.patches
        total_high = sum(p.get_height() for p in high_patches)  # ty: ignore[unresolved-attribute]
        assert total_high == 80

    def test_title_shows_counts(self):
        stats = _make_stats_with_low_activity(n=100, n_low=20)
        fig = magnitude_distribution(stats, low_activity_threshold_pct=1.0)
        ax_all, ax_high = fig.axes
        assert "N=100" in ax_all.get_title()
        assert "N=80" in ax_high.get_title()

    def test_right_panel_title_shows_threshold(self):
        fig = magnitude_distribution(_make_stats(), low_activity_threshold_pct=2.5)
        _, ax_high = fig.axes
        assert "2.5%" in ax_high.get_title()

    def test_has_title_and_labels(self):
        fig = magnitude_distribution(_make_stats())
        ax_all, ax_high = fig.axes
        assert ax_all.get_xlabel() == "Mean Magnitude"
        assert ax_all.get_ylabel() == "Number of Features"
        assert "All Features" in ax_all.get_title()
        assert "High-Activity" in ax_high.get_title()

    def test_log_y_sets_yscale(self):
        fig = magnitude_distribution(_make_stats(), log_y=True)
        for ax in fig.axes:
            assert ax.get_yscale() == "log"

    def test_linear_y_default(self):
        fig = magnitude_distribution(_make_stats())
        for ax in fig.axes:
            assert ax.get_yscale() == "linear"


# ---------------------------------------------------------------------------
# Tests: segment_length_distribution
# ---------------------------------------------------------------------------


class TestSegmentLengthDistribution:
    def _stats_with_seg_length(self) -> ExtractionStatistics:
        return _make_stats(with_segment_length=True)

    def _stats_without_seg_length(self) -> ExtractionStatistics:
        return _make_stats()

    def test_returns_figure_with_data(self):
        fig = segment_length_distribution(self._stats_with_seg_length())
        assert isinstance(fig, Figure)

    def test_returns_figure_without_data(self):
        fig = segment_length_distribution(self._stats_without_seg_length())
        assert isinstance(fig, Figure)

    def test_fallback_shows_text(self):
        fig = segment_length_distribution(self._stats_without_seg_length())
        ax = fig.axes[0]
        assert len(ax.texts) >= 1

    def test_has_bars_with_histogram(self):
        fig = segment_length_distribution(self._stats_with_seg_length())
        ax = fig.axes[0]
        assert len(ax.patches) == 50

    def test_has_percentile_lines(self):
        fig = segment_length_distribution(self._stats_with_seg_length())
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 3  # p10, p50, p90

    def test_has_labels(self):
        fig = segment_length_distribution(self._stats_with_seg_length())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Segment Length (tokens)"
        assert ax.get_ylabel() == "Number of Segments"

    def test_metadata_annotation(self):
        stats = self._stats_with_seg_length()
        fig = segment_length_distribution(stats)
        ax = fig.axes[0]
        # Should have metadata text annotation
        texts = ax.texts
        assert any("segment_mode" in t.get_text() for t in texts)


# ---------------------------------------------------------------------------
# Tests: features_per_token_ratio
# ---------------------------------------------------------------------------


class TestFeaturesPerTokenRatio:
    def _stats_with_fpt(self) -> ExtractionStatistics:
        return _make_stats(with_features_per_token=True)

    def _stats_without_fpt(self) -> ExtractionStatistics:
        return _make_stats()

    def test_returns_figure_with_data(self):
        fig = features_per_token_ratio(self._stats_with_fpt())
        assert isinstance(fig, Figure)

    def test_returns_figure_without_data(self):
        fig = features_per_token_ratio(self._stats_without_fpt())
        assert isinstance(fig, Figure)

    def test_fallback_shows_text(self):
        fig = features_per_token_ratio(self._stats_without_fpt())
        ax = fig.axes[0]
        assert len(ax.texts) >= 1

    def test_has_bars_with_histogram(self):
        fig = features_per_token_ratio(self._stats_with_fpt())
        ax = fig.axes[0]
        assert len(ax.patches) == 50

    def test_has_percentile_lines(self):
        fig = features_per_token_ratio(self._stats_with_fpt())
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 3  # p10, p50, p90

    def test_has_labels(self):
        fig = features_per_token_ratio(self._stats_with_fpt())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Features per Token"
        assert ax.get_ylabel() == "Number of Segments"

    def test_log_x(self):
        fig = features_per_token_ratio(self._stats_with_fpt(), log_x=True)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"

    def test_linear_x_default(self):
        fig = features_per_token_ratio(self._stats_with_fpt())
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"


# ---------------------------------------------------------------------------
# Tests: orchestrator (load_stats, chart registry)
# ---------------------------------------------------------------------------


class TestLoadStats:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            from viz.plot_features import load_stats

            load_stats("/nonexistent/path/stats.safetensors")


class TestChartRegistry:
    def test_has_all_charts(self):
        assert set(CHART_REGISTRY.keys()) == {
            "count_distribution",
            "count_vs_magnitude",
            "features_per_segment",
            "magnitude_distribution",
            "percentile_heatmap",
            "percentile_surface_3d",
            "segment_length_distribution",
            "features_per_token_ratio",
        }

    def test_all_values_are_callable(self):
        for name, fn in CHART_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Tests: end-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test: generate all charts from synthetic data, verify PNGs."""

    def _make_full_stats(self) -> ExtractionStatistics:
        """Build a complete ExtractionStatistics for all chart types."""
        return _make_stats(
            n_features=100,
            with_histograms=True,
            with_segment_length=True,
            with_features_per_token=True,
        )

    def test_all_charts_produce_pngs(self, tmp_path):
        stats = self._make_full_stats()

        fig1 = count_distribution(stats)
        fig2 = count_vs_magnitude(stats)
        fig3 = features_per_segment(stats)
        fig4 = magnitude_distribution(stats)
        fig5 = percentile_heatmap(stats)
        fig6 = percentile_surface_3d(stats)
        fig7 = segment_length_distribution(stats)
        fig8 = features_per_token_ratio(stats)

        figs = [
            ("count_distribution", fig1),
            ("count_vs_magnitude", fig2),
            ("features_per_segment", fig3),
            ("magnitude_distribution", fig4),
            ("percentile_heatmap", fig5),
            ("percentile_surface_3d", fig6),
            ("segment_length_distribution", fig7),
            ("features_per_token_ratio", fig8),
        ]
        for name, fig in figs:
            out = tmp_path / f"{name}.png"
            fig.savefig(out, dpi=72)
            assert out.exists()
            assert out.stat().st_size > 0
            # Verify PNG magic bytes
            with open(out, "rb") as f:
                assert f.read(4) == b"\x89PNG"


# ---------------------------------------------------------------------------
# Tests: chart validation
# ---------------------------------------------------------------------------


class TestChartValidation:
    def test_count_distribution_empty_per_feature_stats(self):
        """Empty per_feature_stats should not crash, just produce an empty chart."""
        stats = _make_stats(n_features=0)
        fig = count_distribution(stats)
        assert isinstance(fig, Figure)

    def test_count_vs_magnitude_empty_per_feature_stats(self):
        """Empty per_feature_stats should not crash."""
        stats = _make_stats(n_features=0)
        fig = count_vs_magnitude(stats)
        assert isinstance(fig, Figure)

    def test_magnitude_distribution_empty_per_feature_stats(self):
        """Empty per_feature_stats should not crash."""
        stats = _make_stats(n_features=0)
        fig = magnitude_distribution(stats)
        assert isinstance(fig, Figure)
