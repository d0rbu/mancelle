"""Pure chart functions for feature statistics visualization.

Each function takes an ExtractionStatistics dataclass and config params,
returns a matplotlib Figure.  No I/O — the caller handles saving.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import matplotlib

from steering.stats_types import (
    ExtractionStatistics,
    PerFeatureStat,
    SegmentFeatureStats,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for '3d' projection

# ---------------------------------------------------------------------------
# Multi-panel group helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HistogramSeries:
    """Numeric histogram values narrowed from the serialized stats payload."""

    bin_edges: list[float]
    counts: list[int]


def _histogram_series(
    histogram: dict[str, object],
    *,
    context: str,
) -> _HistogramSeries:
    """Validate and narrow a corpus-level histogram before plotting it."""
    raw_edges = histogram.get("bin_edges")
    raw_counts = histogram.get("counts")
    if not isinstance(raw_edges, list) or any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in raw_edges
    ):
        raise TypeError(f"{context}.bin_edges must be a list of numbers")
    if not isinstance(raw_counts, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_counts
    ):
        raise TypeError(f"{context}.counts must be a list of integers")
    if len(raw_edges) != len(raw_counts) + 1:
        raise ValueError(
            f"{context} must have one more bin edge than count, got "
            f"{len(raw_edges)} edges and {len(raw_counts)} counts"
        )
    numeric_edges = cast(list[int | float], raw_edges)
    integer_counts = cast(list[int], raw_counts)
    return _HistogramSeries(
        bin_edges=[float(value) for value in numeric_edges],
        counts=list(integer_counts),
    )


def _per_feature_histogram_counts(
    histogram: dict[str, object],
    *,
    context: str,
) -> list[int]:
    """Validate and narrow the counts-only per-feature histogram payload."""
    raw_counts = histogram.get("counts")
    if not isinstance(raw_counts, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_counts
    ):
        raise TypeError(f"{context}.counts must be a list of integers")
    return list(cast(list[int], raw_counts))


def _prepend_title(ax: Axes, prefix: str) -> None:
    """Prepend a group label to the axes title."""
    existing = ax.get_title()
    if existing:
        ax.set_title(f"{prefix} — {existing}")
    else:
        ax.set_title(prefix)


def _build_grouped_figure(
    render_fn: Callable,
    default_kwargs: dict[str, Any],
    slices: dict[str, dict] | None,
    extract_kwargs_fn: Callable[[str, dict], dict[str, Any]],
    *,
    figsize_single: tuple[float, float] = (10, 6),
    ncols: int = 1,
) -> Figure:
    """Build single- or multi-panel figure with one row per slice.

    Args:
        render_fn: Callable(fig, ax_or_axes, **kwargs) that draws onto given axes.
            For ncols=1, receives a single Axes. For ncols>1, receives a row of Axes.
        default_kwargs: kwargs for the default panel.
        slices: slice key string -> entry dict, or None for single panel.
            Keys are used directly as panel labels. Insertion order is preserved.
        extract_kwargs_fn: (slice_key, slice_entry) -> kwargs for render_fn.
        figsize_single: (w, h) for a single panel.
        ncols: number of columns per row.
    """
    if not slices:
        fig, axes = plt.subplots(1, ncols, figsize=figsize_single, squeeze=False)
        ax = axes[0, 0] if ncols == 1 else axes[0, :]
        render_fn(fig, ax, **default_kwargs)
        fig.tight_layout()
        return fig

    slice_keys = list(slices.keys())
    n_rows = 1 + len(slice_keys)
    w, h = figsize_single
    fig, all_axes = plt.subplots(n_rows, ncols, figsize=(w, h * n_rows), squeeze=False)

    # Row 0: default
    ax = all_axes[0, 0] if ncols == 1 else all_axes[0, :]
    render_fn(fig, ax, **default_kwargs)
    _prepend_title(all_axes[0, 0], "default")

    # Slice rows
    for i, key in enumerate(slice_keys):
        ax = all_axes[i + 1, 0] if ncols == 1 else all_axes[i + 1, :]
        kwargs = extract_kwargs_fn(key, slices[key])
        render_fn(fig, ax, **kwargs)
        _prepend_title(all_axes[i + 1, 0], key)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Render functions (draw onto given axes, no figure creation)
# ---------------------------------------------------------------------------


def _render_count_distribution(
    fig: Figure,
    ax: Axes,
    *,
    per_feature_stats: dict[str, PerFeatureStat],
    count_percentiles: dict[str, int],
    n_segments: int,
    log_scale: bool,
    n_bins: int,
) -> None:
    """Render count distribution histogram onto given axes."""
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")

    pcts = [feat.pct_segments_active for feat in per_feature_stats.values()]

    if log_scale and pcts:
        lo = math.log10(max(1e-6, min(pcts)))
        hi = math.log10(max(pcts))
        bins = [10 ** (lo + i * (hi - lo) / (n_bins - 1)) for i in range(n_bins)]
        ax.set_xscale("log")
    else:
        bins = n_bins

    ax.hist(pcts, bins=bins, edgecolor="black", alpha=0.7)

    highlight = {"p10": "green", "p50": "orange", "p90": "red"}
    for label, color in highlight.items():
        if label in count_percentiles:
            pct_val = count_percentiles[label] / n_segments * 100
            ax.axvline(pct_val, color=color, linestyle="--", linewidth=1.5, label=label)

    ax.set_xlabel("% of Segments Active")
    ax.set_ylabel("Number of Features")
    ax.set_title(f"Feature Activation Distribution (N={n_segments:,} segments)")
    if count_percentiles:
        ax.legend()


def _render_count_vs_magnitude(
    fig: Figure,
    ax: Axes,
    *,
    per_feature_stats: dict[str, PerFeatureStat],
    log_x: bool,
    log_y: bool,
    alpha: float,
) -> None:
    """Render count vs magnitude scatter onto given axes."""
    counts = []
    mean_mags = []
    std_mags = []
    has_std = all(hasattr(feat, "std_mag") for feat in per_feature_stats.values())

    for feat in per_feature_stats.values():
        counts.append(feat.count)
        mean_mags.append(feat.mean_mag)
        if has_std:
            std_mags.append(feat.std_mag)

    if has_std and std_mags:
        scatter = ax.scatter(
            counts,
            mean_mags,
            c=std_mags,
            alpha=alpha,
            s=10,
            cmap="plasma",
        )
        fig.colorbar(scatter, ax=ax, label="Std Magnitude")
    else:
        ax.scatter(counts, mean_mags, alpha=alpha, s=10, color="steelblue")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Feature Activation Count")
    ax.set_ylabel("Mean Magnitude")
    ax.set_title("Count vs Mean Magnitude")


def _render_features_per_segment(
    fig: Figure,
    ax: Axes,
    *,
    feature_stats: SegmentFeatureStats,
) -> None:
    """Render features-per-segment distribution onto given axes."""
    n_segments = feature_stats.n_segments
    histogram = feature_stats.active_per_segment_histogram

    if histogram is not None:
        series = _histogram_series(
            histogram,
            context="active_per_segment_histogram",
        )
        bin_edges = series.bin_edges
        counts = series.counts
        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(counts))]
        centers = [bin_edges[i] + widths[i] / 2 for i in range(len(counts))]
        ax.bar(
            centers,
            counts,
            width=widths,
            edgecolor="black",
            alpha=0.7,
            align="center",
        )
    else:
        text_lines = [
            f"Mean: {feature_stats.mean_active_per_segment}",
            f"Median: {feature_stats.median_active_per_segment}",
            f"P90: {feature_stats.p90_active_per_segment}",
            f"Min: {feature_stats.min_active_per_segment}",
            f"Max: {feature_stats.max_active_per_segment}",
        ]
        ax.text(
            0.5,
            0.5,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    highlight = {
        "mean": ("blue", feature_stats.mean_active_per_segment),
        "median": ("orange", feature_stats.median_active_per_segment),
        "p90": ("red", feature_stats.p90_active_per_segment),
    }
    for label, (color, value) in highlight.items():
        if value is not None:
            ax.axvline(
                float(value),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=label,
            )

    ax.set_xlabel("Number of Active Features")
    ax.set_ylabel("Number of Segments")
    ax.set_title(f"Features Active per Segment (N={n_segments:,} segments)")
    ax.legend()


def _render_magnitude_distribution(
    fig: Figure,
    axes,
    *,
    per_feature_stats: dict[str, PerFeatureStat],
    n_bins: int,
    log_y: bool,
    low_activity_threshold_pct: float,
) -> None:
    """Render magnitude distribution side-by-side panels onto given axes row."""
    ax_all, ax_high = axes[0], axes[1]

    all_mags = [feat.mean_mag for feat in per_feature_stats.values()]
    high_activity_mags = [
        feat.mean_mag
        for feat in per_feature_stats.values()
        if feat.pct_segments_active >= low_activity_threshold_pct
    ]

    n_all = len(all_mags)
    n_high = len(high_activity_mags)

    ax_all.hist(all_mags, bins=n_bins, edgecolor="black", alpha=0.7)
    ax_all.set_xlabel("Mean Magnitude")
    ax_all.set_ylabel("Number of Features")
    ax_all.set_title(f"All Features (N={n_all})")
    if log_y:
        ax_all.set_yscale("log")

    if high_activity_mags:
        ax_high.hist(high_activity_mags, bins=n_bins, edgecolor="black", alpha=0.7)
    ax_high.set_xlabel("Mean Magnitude")
    ax_high.set_ylabel("Number of Features")
    ax_high.set_title(
        f"High-Activity Only (>={low_activity_threshold_pct}% segments, N={n_high})",
    )
    if log_y:
        ax_high.set_yscale("log")


def _render_features_per_token_ratio(
    fig: Figure,
    ax: Axes,
    *,
    feature_stats: SegmentFeatureStats,
    log_x: bool,
) -> None:
    """Render features-per-token ratio histogram onto given axes."""
    histogram = feature_stats.features_per_token_histogram
    percentiles = feature_stats.features_per_token_percentiles

    if histogram is not None:
        series = _histogram_series(
            histogram,
            context="features_per_token_histogram",
        )
        bin_edges = series.bin_edges
        counts = series.counts
        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(counts))]
        centers = [bin_edges[i] + widths[i] / 2 for i in range(len(counts))]
        ax.bar(
            centers,
            counts,
            width=widths,
            edgecolor="black",
            alpha=0.7,
            align="center",
        )

        if log_x:
            ax.set_xscale("log")

        if percentiles:
            pct_lines = {"p10": "green", "p50": "orange", "p90": "red"}
            for label, color in pct_lines.items():
                if label in percentiles:
                    ax.axvline(
                        percentiles[label],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        label=label,
                    )

        ax.set_xlabel("Features per Token")
        ax.set_ylabel("Number of Segments")
        ax.set_title("Features-per-Token Ratio Distribution")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Features-per-token data not available\n(only computed for fixed_tokens mode)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        ax.set_title("Features-per-Token Ratio Distribution")


def _render_density_heatmap_on_ax(
    fig: Figure,
    ax: Axes,
    sampled: list[tuple[str, PerFeatureStat]],
    bin_edges: list[float],
    colormap: str,
    log_scale: bool,
) -> None:
    """Render magnitude density heatmap onto a given axes."""
    n_bins = len(bin_edges) - 1
    n_sample = len(sampled)

    matrix = np.zeros((n_bins, n_sample))
    feature_labels = []
    for col, (feat_id, feat_data) in enumerate(sampled):
        hist = feat_data.histogram
        if hist is not None:
            counts = _per_feature_histogram_counts(
                hist,
                context=f"per_feature_stats[{feat_id!r}].histogram",
            )
            total = sum(counts)
            if total > 0:
                matrix[:, col] = [c / total for c in counts]
        feature_labels.append(f"{feat_id}\n(n={feat_data.count})")

    if log_scale:
        matrix = np.log1p(matrix)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=colormap,
        origin="lower",
        extent=(-0.5, n_sample - 0.5, bin_edges[0], bin_edges[-1]),
    )
    fig.colorbar(im, ax=ax, label="Density" + (" (log1p)" if log_scale else ""))

    ax.set_xticks(range(n_sample))
    ax.set_xticklabels(feature_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Feature (sampled by count)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Per-Feature Magnitude Density Heatmap")


# ---------------------------------------------------------------------------
# Public chart functions
# ---------------------------------------------------------------------------


def count_distribution(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    log_scale: bool = True,
    n_bins: int = 50,
) -> Figure:
    """Histogram of feature activation as percentage of segments (log-scale x-axis)."""
    per_feature_stats = stats.per_feature_stats
    count_percentiles = stats.features.count_percentiles
    n_segments = stats.features.n_segments

    default_kwargs = {
        "per_feature_stats": per_feature_stats,
        "count_percentiles": count_percentiles,
        "n_segments": n_segments,
        "log_scale": log_scale,
        "n_bins": n_bins,
    }

    def extract(key, entry):
        return {
            "per_feature_stats": entry["per_feature"],
            "count_percentiles": {},
            "n_segments": n_segments,
            "log_scale": log_scale,
            "n_bins": n_bins,
        }

    return _build_grouped_figure(
        _render_count_distribution,
        default_kwargs,
        slices,
        extract,
        figsize_single=(10, 6),
    )


def count_vs_magnitude(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    log_x: bool = True,
    log_y: bool = False,
    alpha: float = 0.3,
) -> Figure:
    """Scatter plot: feature count vs mean magnitude, colored by std_mag."""
    per_feature_stats = stats.per_feature_stats

    default_kwargs = {
        "per_feature_stats": per_feature_stats,
        "log_x": log_x,
        "log_y": log_y,
        "alpha": alpha,
    }

    def extract(key, entry):
        return {
            "per_feature_stats": entry["per_feature"],
            "log_x": log_x,
            "log_y": log_y,
            "alpha": alpha,
        }

    return _build_grouped_figure(
        _render_count_vs_magnitude,
        default_kwargs,
        slices,
        extract,
        figsize_single=(10, 6),
    )


def _sample_features_by_count(
    per_feature_stats: dict[str, PerFeatureStat],
    n_features: int,
) -> list[tuple[str, PerFeatureStat]]:
    """Sample features evenly across the count distribution."""
    sorted_features = sorted(
        per_feature_stats.items(),
        key=lambda kv: kv[1].count,
    )
    n_available = len(sorted_features)
    n_sample = min(n_features, n_available)
    if n_sample <= 1:
        indices = [0]
    else:
        indices = [
            round(i * (n_available - 1) / (n_sample - 1)) for i in range(n_sample)
        ]
    return [sorted_features[i] for i in indices]


def percentile_heatmap(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    n_features: int = 21,
    colormap: str = "plasma",
    log_scale: bool = False,
) -> Figure:
    """Magnitude density heatmap: X=features, Y=magnitude bins, Color=density.

    Uses per-feature histogram data when available, falls back to percentile-based
    heatmap for backwards compatibility.
    """
    per_feature_stats = stats.per_feature_stats
    feature_stats = stats.features

    bin_edges = feature_stats.mag_histogram_bin_edges

    sampled = _sample_features_by_count(per_feature_stats, n_features)

    has_histograms = bin_edges is not None and any(
        feat.histogram is not None for _, feat in sampled
    )

    if has_histograms:
        return _magnitude_density_heatmap(sampled, bin_edges, colormap, log_scale)
    return _percentile_heatmap_legacy(sampled, colormap)


def _magnitude_density_heatmap(
    sampled: list[tuple[str, PerFeatureStat]],
    bin_edges: list[float],
    colormap: str,
    log_scale: bool,
) -> Figure:
    """Density heatmap using per-feature histogram data (single panel)."""
    n_sample = len(sampled)
    fig, ax = plt.subplots(figsize=(max(10, n_sample * 0.6), 8))
    _render_density_heatmap_on_ax(fig, ax, sampled, bin_edges, colormap, log_scale)
    fig.tight_layout()
    return fig


def _percentile_heatmap_legacy(
    sampled: list[tuple[str, PerFeatureStat]],
    colormap: str,
) -> Figure:
    """Legacy percentile-based heatmap for backwards compatibility."""
    percentile_keys = [
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p99",
    ]
    n_sample = len(sampled)
    matrix = []
    feature_labels = []
    for feat_id, feat_data in sampled:
        row = [feat_data.percentiles.get(k, 0.0) for k in percentile_keys]
        matrix.append(row)
        feature_labels.append(f"{feat_id} (n={feat_data.count})")

    normalized = []
    for row in matrix:
        lo = min(row)
        hi = max(row)
        span = hi - lo if hi != lo else 1.0
        normalized.append([(v - lo) / span for v in row])

    fig, ax = plt.subplots(figsize=(12, max(6, n_sample * 0.4)))
    im = ax.imshow(normalized, aspect="auto", cmap=colormap)
    fig.colorbar(im, ax=ax, label="Normalized Magnitude")

    ax.set_xticks(range(len(percentile_keys)))
    ax.set_xticklabels(percentile_keys)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=8)

    ax.set_xlabel("Magnitude Percentile")
    ax.set_title("Feature Magnitude Profiles (sampled across count distribution)")

    return fig


def percentile_surface_3d(
    stats: ExtractionStatistics,
    *,
    n_features: int = 21,
    colormap: str = "plasma",
    elevation: int = 30,
    azimuth: int = 45,
    log_scale: bool = False,
) -> Figure:
    """3D surface: X=feature index, Y=magnitude (bin centers), Z=density.

    Uses per-feature histogram data when available, falls back to percentile-based
    surface. No slices support (3D subplots don't stack well).
    """
    per_feature_stats = stats.per_feature_stats
    feature_stats = stats.features

    bin_edges = feature_stats.mag_histogram_bin_edges

    sampled = _sample_features_by_count(per_feature_stats, n_features)

    has_histograms = bin_edges is not None and any(
        feat.histogram is not None for _, feat in sampled
    )

    if has_histograms:
        return _magnitude_density_surface(
            sampled,
            bin_edges,
            colormap,
            elevation,
            azimuth,
            log_scale,
        )
    return _percentile_surface_legacy(sampled, colormap, elevation, azimuth)


def _magnitude_density_surface(
    sampled: list[tuple[str, PerFeatureStat]],
    bin_edges: list[float],
    colormap: str,
    elevation: int,
    azimuth: int,
    log_scale: bool,
) -> Figure:
    """Density surface using per-feature histogram data."""
    n_bins = len(bin_edges) - 1
    n_sample = len(sampled)
    bin_centers = np.array(
        [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)],
    )

    Z = np.zeros((n_bins, n_sample))
    for col, (feat_id, feat_data) in enumerate(sampled):
        hist = feat_data.histogram
        if hist is not None:
            counts = _per_feature_histogram_counts(
                hist,
                context=f"per_feature_stats[{feat_id!r}].histogram",
            )
            total = sum(counts)
            if total > 0:
                Z[:, col] = [c / total for c in counts]

    if log_scale:
        Z = np.log1p(Z)

    x = np.arange(n_sample)
    X, Y = np.meshgrid(x, bin_centers)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=colormap)

    ax.set_xlabel("Feature (by count)")
    ax.set_ylabel("Magnitude")
    ax.set_zlabel("Density" + (" (log1p)" if log_scale else ""))
    ax.set_title("Per-Feature Magnitude Density — 3D Surface")
    ax.view_init(elev=elevation, azim=azimuth)

    return fig


def _percentile_surface_legacy(
    sampled: list[tuple[str, PerFeatureStat]],
    colormap: str,
    elevation: int,
    azimuth: int,
) -> Figure:
    """Legacy percentile-based 3D surface for backwards compatibility."""
    percentile_keys = [
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p99",
    ]
    n_sample = len(sampled)
    matrix = []
    for _feat_id, feat_data in sampled:
        row = [feat_data.percentiles.get(k, 0.0) for k in percentile_keys]
        matrix.append(row)

    normalized = []
    for row in matrix:
        lo = min(row)
        hi = max(row)
        span = hi - lo if hi != lo else 1.0
        normalized.append([(v - lo) / span for v in row])

    Z = np.array(normalized)
    x = np.arange(len(percentile_keys))
    y = np.arange(n_sample)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=colormap)

    ax.set_xlabel("Magnitude Percentile")
    ax.set_ylabel("Feature (by count)")
    ax.set_zlabel("Normalized Magnitude")
    ax.set_title("Feature Magnitude Profiles — 3D Surface")

    ax.set_xticks(range(len(percentile_keys)))
    ax.set_xticklabels(percentile_keys, fontsize=7)
    ax.view_init(elev=elevation, azim=azimuth)

    return fig


def features_per_segment(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    n_bins: int = 50,
) -> Figure:
    """Bar chart showing distribution of how many features activate per segment."""
    feature_stats = stats.features

    default_kwargs = {"feature_stats": feature_stats}

    def extract(key, entry):
        return {"feature_stats": entry["segment"]}

    return _build_grouped_figure(
        _render_features_per_segment,
        default_kwargs,
        slices,
        extract,
        figsize_single=(10, 6),
    )


def magnitude_distribution(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    n_bins: int = 50,
    log_y: bool = False,
    low_activity_threshold_pct: float = 1.0,
) -> Figure:
    """Side-by-side histograms: all features vs high-activity features only.

    Left panel: all features. Right panel: only features with
    pct_segments_active >= low_activity_threshold_pct.
    """
    per_feature_stats = stats.per_feature_stats

    n_all = len(per_feature_stats)
    n_high = sum(
        1
        for f in per_feature_stats.values()
        if f.pct_segments_active >= low_activity_threshold_pct
    )
    if n_all == n_high:
        logger.warning(
            f"All {n_all} features have pct_segments_active >= "
            f"{low_activity_threshold_pct}%. "
            "Both panels will be identical. Consider raising "
            "low_activity_threshold_pct.",
        )

    default_kwargs = {
        "per_feature_stats": per_feature_stats,
        "n_bins": n_bins,
        "log_y": log_y,
        "low_activity_threshold_pct": low_activity_threshold_pct,
    }

    def extract(key, entry):
        return {
            "per_feature_stats": entry["per_feature"],
            "n_bins": n_bins,
            "log_y": log_y,
            "low_activity_threshold_pct": low_activity_threshold_pct,
        }

    fig = _build_grouped_figure(
        _render_magnitude_distribution,
        default_kwargs,
        slices,
        extract,
        figsize_single=(14, 6),
        ncols=2,
    )
    if not slices:
        fig.suptitle("Feature Mean Magnitude Distribution")
        fig.tight_layout()
    return fig


def segment_length_distribution(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    n_bins: int = 50,
) -> Figure:
    """Histogram of segment token lengths with percentile lines and metadata annotation."""
    feature_stats = stats.features
    metadata = stats.metadata

    histogram = feature_stats.segment_length_histogram
    percentiles = feature_stats.segment_length_percentiles

    fig, ax = plt.subplots(figsize=(10, 6))

    if histogram is not None:
        series = _histogram_series(
            histogram,
            context="segment_length_histogram",
        )
        bin_edges = series.bin_edges
        counts = series.counts

        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(counts))]
        centers = [bin_edges[i] + widths[i] / 2 for i in range(len(counts))]
        ax.bar(
            centers,
            counts,
            width=widths,
            edgecolor="black",
            alpha=0.7,
            align="center",
        )

        if percentiles:
            pct_lines = {"p10": "green", "p50": "orange", "p90": "red"}
            for label, color in pct_lines.items():
                if label in percentiles:
                    ax.axvline(
                        percentiles[label],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        label=label,
                    )

        ax.set_xlabel("Segment Length (tokens)")
        ax.set_ylabel("Number of Segments")
        ax.set_title("Segment Token Length Distribution")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Segment length data not available\n(only computed for fixed_tokens mode)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        ax.set_title("Segment Token Length Distribution")

    if metadata is not None:
        meta_lines = []
        for key in ("segment_mode", "pooling_mode", "segment_size", "context_window"):
            val = getattr(metadata, key, None)
            if val is not None:
                meta_lines.append(f"{key}: {val}")
        if meta_lines:
            ax.text(
                0.98,
                0.98,
                "\n".join(meta_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightyellow",
                    "alpha": 0.8,
                },
            )

    return fig


def features_per_token_ratio(
    stats: ExtractionStatistics,
    *,
    slices: dict[str, dict] | None = None,
    n_bins: int = 50,
    log_x: bool = False,
) -> Figure:
    """Histogram of (#active_features / #tokens_in_segment) ratio."""
    feature_stats = stats.features

    default_kwargs = {"feature_stats": feature_stats, "log_x": log_x}

    def extract(key, entry):
        return {"feature_stats": entry["segment"], "log_x": log_x}

    return _build_grouped_figure(
        _render_features_per_token_ratio,
        default_kwargs,
        slices,
        extract,
        figsize_single=(10, 6),
    )
