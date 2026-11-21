"""Cross-axis comparison visualizations for grid sweep results.

Loads all feature_statistics.json files from a grid output directory
and creates comparison plots showing how SAE feature distributions
vary across different parameter axes (layer, activation type, pooling,
segment mode, SAE width, activation percentile).

Usage:
    uv run python -m viz.cross_axis_plots
    uv run python -m viz.cross_axis_plots grid_dir=out/data/my_grid
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib
from omegaconf import DictConfig

from steering.stats_io import glob_statistics, load_statistics
from steering.stats_types import ExtractionStatistics

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


@dataclass
class StatsEntry:
    """A single stats file with its grid coordinates."""

    layer: int
    activation: str
    pooling: str
    segment: str
    width: str
    percentile: float
    stats: ExtractionStatistics
    path: str


def discover_stats_files(grid_dir: str) -> list[StatsEntry]:
    """Walk the grid output directory and load all feature_statistics.json files."""
    entries = []
    grid_path = Path(grid_dir)

    for stats_file in glob_statistics(grid_path):
        rel = stats_file.relative_to(grid_path)
        parts = str(rel).split("/")

        # Parse directory structure: layer=X/activation=Y/pooling=Z/segment=W/width=V/percentile=P
        coords = {}
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                coords[key] = val

        if not all(
            k in coords
            for k in (
                "layer",
                "activation",
                "pooling",
                "segment",
                "width",
                "percentile",
            )
        ):
            logger.warning(f"Skipping {stats_file}: incomplete coordinates {coords}")
            continue

        try:
            stats = load_statistics(stats_file)
        except (ValueError, OSError) as e:
            logger.warning(f"Skipping {stats_file}: {e}")
            continue

        entries.append(
            StatsEntry(
                layer=int(coords["layer"]),
                activation=coords["activation"],
                pooling=coords["pooling"],
                segment=coords["segment"],
                width=coords["width"],
                percentile=float(coords["percentile"]),
                stats=stats,
                path=str(stats_file),
            )
        )

    logger.info(f"Loaded {len(entries)} stats files from {grid_dir}")
    return entries


def _extract_metric(entry: StatsEntry, metric: str) -> float | None:
    """Extract a numeric metric from a stats entry."""
    features = entry.stats.features
    corpus = entry.stats.corpus

    metric_map = {
        "mean_active_per_segment": features.mean_active_per_segment,
        "median_active_per_segment": features.median_active_per_segment,
        "p90_active_per_segment": features.p90_active_per_segment,
        "max_active_per_segment": features.max_active_per_segment,
        "pct_features_active": features.pct_features_active,
        "n_features_seen": features.n_features_seen,
        "n_segments": features.n_segments,
        "n_docs": corpus.n_docs,
        "n_tokens_gpt2": corpus.n_tokens_gpt2,
        "mean_tokens_per_doc": corpus.mean_tokens_per_doc,
    }

    # Also extract magnitude stats
    mag_stats = features.magnitude_stats or {}
    metric_map.update(
        {
            "mean_magnitude": mag_stats.get("mean"),
            "median_magnitude": mag_stats.get("median"),
            "p90_magnitude": mag_stats.get("p90"),
            "std_magnitude": mag_stats.get("std"),
        }
    )

    value = metric_map.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _filter_entries(
    entries: list[StatsEntry],
    *,
    percentile: float = 0.0,
    **kwargs,
) -> list[StatsEntry]:
    """Filter entries by fixed coordinate values."""
    filtered = [e for e in entries if e.percentile == percentile]
    for key, val in kwargs.items():
        filtered = [e for e in filtered if getattr(e, key) == val]
    return filtered


def _get_unique_values(entries: list[StatsEntry], attr: str) -> list:
    """Get sorted unique values for a given attribute."""
    vals = sorted({getattr(e, attr) for e in entries})
    return vals


# ---------------------------------------------------------------------------
# Comparison chart generators
# ---------------------------------------------------------------------------


def plot_metric_by_layer(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Bar chart comparing a metric across layers, grouped by activation type and width."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        logger.warning(f"No entries for metric_by_layer ({metric})")
        return

    layers = _get_unique_values(filtered, "layer")
    poolings = _get_unique_values(filtered, "pooling")

    for pooling in poolings:
        pool_entries = [e for e in filtered if e.pooling == pooling]
        groups = defaultdict(dict)
        for e in pool_entries:
            label = f"{e.activation}/{e.width}"
            groups[label][e.layer] = _extract_metric(e, metric)

        if not groups:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(layers))
        n_groups = len(groups)
        width = 0.8 / max(n_groups, 1)

        for i, (label, layer_vals) in enumerate(sorted(groups.items())):
            vals = [layer_vals.get(layer, 0) or 0 for layer in layers]
            offset = (i - n_groups / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label, alpha=0.85)

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} by Layer\n(pooling={pooling}, segment={segment}, percentile={percentile})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(layer) for layer in layers])
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        fname = f"{metric}_by_layer_pool={pooling}_seg={segment}_pct={percentile}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_metric_by_activation_type(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Grouped bar chart comparing metric across activation types."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        return

    activations = _get_unique_values(filtered, "activation")
    _get_unique_values(filtered, "layer")
    _get_unique_values(filtered, "width")
    poolings = _get_unique_values(filtered, "pooling")

    for pooling in poolings:
        pool_entries = [e for e in filtered if e.pooling == pooling]
        groups = defaultdict(dict)
        for e in pool_entries:
            label = f"L{e.layer}/{e.width}"
            groups[label][e.activation] = _extract_metric(e, metric)

        if not groups:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(activations))
        n_groups = len(groups)
        width = 0.8 / max(n_groups, 1)

        for i, (label, act_vals) in enumerate(sorted(groups.items())):
            vals = [act_vals.get(a, 0) or 0 for a in activations]
            offset = (i - n_groups / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label, alpha=0.85)

        ax.set_xlabel("Activation Type")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} by Activation Type\n(pooling={pooling}, segment={segment})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(activations)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        fname = f"{metric}_by_activation_pool={pooling}_seg={segment}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_metric_by_pooling(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Grouped bar chart comparing metric across pooling modes."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        return

    poolings = _get_unique_values(filtered, "pooling")
    groups = defaultdict(dict)
    for e in filtered:
        label = f"L{e.layer}/{e.activation}/{e.width}"
        groups[label][e.pooling] = _extract_metric(e, metric)

    if not groups:
        return

    # Pick top 6 groups by mean value for readability
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: np.mean([v for v in kv[1].values() if v is not None]),
        reverse=True,
    )[:6]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(poolings))
    n_groups = len(sorted_groups)
    width = 0.8 / max(n_groups, 1)

    for i, (label, pool_vals) in enumerate(sorted_groups):
        vals = [pool_vals.get(p, 0) or 0 for p in poolings]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, alpha=0.85)

    ax.set_xlabel("Pooling Mode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"{metric.replace('_', ' ').title()} by Pooling Mode\n(segment={segment}, top 6 combos)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(poolings)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    fname = f"{metric}_by_pooling_seg={segment}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {fname}")


def plot_metric_by_width(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Grouped bar chart comparing metric across SAE widths."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        return

    widths = _get_unique_values(filtered, "width")
    poolings = _get_unique_values(filtered, "pooling")

    for pooling in poolings:
        pool_entries = [e for e in filtered if e.pooling == pooling]
        groups = defaultdict(dict)
        for e in pool_entries:
            label = f"L{e.layer}/{e.activation}"
            groups[label][e.width] = _extract_metric(e, metric)

        if not groups:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(widths))
        n_groups = len(groups)
        w = 0.8 / max(n_groups, 1)

        for i, (label, width_vals) in enumerate(sorted(groups.items())):
            vals = [width_vals.get(ww, 0) or 0 for ww in widths]
            offset = (i - n_groups / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=label, alpha=0.85)

        ax.set_xlabel("SAE Width")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} by SAE Width\n(pooling={pooling}, segment={segment})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(widths)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        fname = f"{metric}_by_width_pool={pooling}_seg={segment}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_metric_by_percentile(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    segment: str = "fixed_tokens",
) -> None:
    """Line chart showing how a metric changes with activation percentile filtering."""
    filtered = [e for e in entries if e.segment == segment]
    if not filtered:
        return

    percentiles = sorted({e.percentile for e in filtered})
    if len(percentiles) < 2:
        return

    poolings = _get_unique_values(filtered, "pooling")

    for pooling in poolings:
        pool_entries = [e for e in filtered if e.pooling == pooling]
        groups = defaultdict(dict)
        for e in pool_entries:
            label = f"L{e.layer}/{e.activation}/{e.width}"
            groups[label][e.percentile] = _extract_metric(e, metric)

        if not groups:
            continue

        # Pick top 6 groups for readability
        sorted_groups = sorted(
            groups.items(),
            key=lambda kv: max((v for v in kv[1].values() if v is not None), default=0),
            reverse=True,
        )[:6]

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, pct_vals in sorted_groups:
            xs = sorted(pct_vals.keys())
            ys = [pct_vals[x] or 0 for x in xs]
            ax.plot(xs, ys, "o-", label=label, alpha=0.8, markersize=5)

        ax.set_xlabel("Activation Percentile")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} vs Activation Percentile\n(pooling={pooling}, segment={segment}, top 6 combos)"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"{metric}_by_percentile_pool={pooling}_seg={segment}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_segment_mode_comparison(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
) -> None:
    """Grouped bar chart comparing metric across segment modes."""
    filtered = _filter_entries(entries, percentile=percentile)
    if not filtered:
        return

    segments = _get_unique_values(filtered, "segment")
    if len(segments) < 2:
        return

    poolings = _get_unique_values(filtered, "pooling")

    for pooling in poolings:
        pool_entries = [e for e in filtered if e.pooling == pooling]
        groups = defaultdict(dict)
        for e in pool_entries:
            label = f"L{e.layer}/{e.activation}/{e.width}"
            groups[label][e.segment] = _extract_metric(e, metric)

        if not groups:
            continue

        # Pick top 6 groups
        sorted_groups = sorted(
            groups.items(),
            key=lambda kv: np.mean([v for v in kv[1].values() if v is not None]),
            reverse=True,
        )[:6]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(segments))
        n_groups = len(sorted_groups)
        w = 0.8 / max(n_groups, 1)

        for i, (label, seg_vals) in enumerate(sorted_groups):
            vals = [seg_vals.get(s, 0) or 0 for s in segments]
            offset = (i - n_groups / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=label, alpha=0.85)

        ax.set_xlabel("Segment Mode")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} by Segment Mode\n(pooling={pooling}, top 6 combos)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(segments)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()

        fname = f"{metric}_by_segment_pool={pooling}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_heatmap_layer_x_activation(
    entries: list[StatsEntry],
    metric: str,
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Heatmap: layer (y-axis) vs activation type (x-axis), one per (pooling, width)."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        return

    layers = _get_unique_values(filtered, "layer")
    activations = _get_unique_values(filtered, "activation")
    poolings = _get_unique_values(filtered, "pooling")
    widths = _get_unique_values(filtered, "width")

    for pooling in poolings:
        for width in widths:
            subset = [e for e in filtered if e.pooling == pooling and e.width == width]
            if not subset:
                continue

            matrix = np.full((len(layers), len(activations)), np.nan)
            for e in subset:
                li = layers.index(e.layer)
                ai = activations.index(e.activation)
                val = _extract_metric(e, metric)
                if val is not None:
                    matrix[li, ai] = val

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
            fig.colorbar(im, ax=ax, label=metric.replace("_", " ").title())

            ax.set_xticks(range(len(activations)))
            ax.set_xticklabels(activations)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([str(layer) for layer in layers])
            ax.set_xlabel("Activation Type")
            ax.set_ylabel("Layer")
            ax.set_title(
                f"{metric.replace('_', ' ').title()}\n"
                f"(pooling={pooling}, width={width}, segment={segment})"
            )

            # Annotate cells
            for i in range(len(layers)):
                for j in range(len(activations)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text = f"{val:.1f}" if val > 10 else f"{val:.2f}"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="white" if val > np.nanmedian(matrix) else "black",
                        )

            fig.tight_layout()
            fname = f"{metric}_heatmap_layer_x_act_pool={pooling}_w={width}_seg={segment}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {fname}")


def plot_summary_dashboard(
    entries: list[StatsEntry],
    output_dir: Path,
    *,
    percentile: float = 0.0,
    segment: str = "fixed_tokens",
) -> None:
    """Create a 2x2 summary dashboard with key metrics across axes."""
    filtered = _filter_entries(entries, percentile=percentile, segment=segment)
    if not filtered:
        return

    metrics = [
        ("mean_active_per_segment", "Mean Active Features/Segment"),
        ("pct_features_active", "% Features Active"),
        ("mean_magnitude", "Mean Magnitude"),
        ("median_active_per_segment", "Median Active Features/Segment"),
    ]

    layers = _get_unique_values(filtered, "layer")
    activations = _get_unique_values(filtered, "activation")

    for pooling in _get_unique_values(filtered, "pooling"):
        pool_entries = [e for e in filtered if e.pooling == pooling]
        widths = _get_unique_values(pool_entries, "width")

        for width in widths:
            subset = [e for e in pool_entries if e.width == width]
            if not subset:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            for idx, (metric_key, metric_label) in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]

                for act in activations:
                    act_entries = sorted(
                        [e for e in subset if e.activation == act],
                        key=lambda e: e.layer,
                    )
                    xs = [e.layer for e in act_entries]
                    ys = [_extract_metric(e, metric_key) or 0 for e in act_entries]
                    ax.plot(xs, ys, "o-", label=act, markersize=8, linewidth=2)

                ax.set_xlabel("Layer")
                ax.set_ylabel(metric_label)
                ax.set_title(metric_label)
                ax.legend()
                ax.grid(True, alpha=0.3)
                if layers:
                    ax.set_xticks(layers)

            fig.suptitle(
                f"Feature Distribution Summary\n"
                f"(pooling={pooling}, width={width}, segment={segment}, percentile={percentile})",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()

            fname = f"summary_dashboard_pool={pooling}_w={width}_seg={segment}_pct={percentile}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {fname}")


def generate_all_cross_axis_plots(grid_dir: str, output_dir: str) -> None:
    """Generate comprehensive cross-axis comparison plots."""
    entries = discover_stats_files(grid_dir)
    if not entries:
        logger.error("No stats files found — cannot generate plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    segments = _get_unique_values(entries, "segment")
    percentiles = sorted({e.percentile for e in entries})

    # Key metrics to compare
    key_metrics = [
        "mean_active_per_segment",
        "median_active_per_segment",
        "pct_features_active",
        "mean_magnitude",
        "median_magnitude",
        "n_features_seen",
    ]

    logger.info(
        f"Generating cross-axis plots for {len(key_metrics)} metrics across {len(segments)} segment modes"
    )

    for metric in key_metrics:
        logger.info(f"Generating plots for: {metric}")

        for seg in segments:
            # By layer
            plot_metric_by_layer(entries, metric, out, segment=seg)

            # By activation type
            plot_metric_by_activation_type(entries, metric, out, segment=seg)

            # By pooling mode
            plot_metric_by_pooling(entries, metric, out, segment=seg)

            # By SAE width
            plot_metric_by_width(entries, metric, out, segment=seg)

            # Heatmap layer x activation
            plot_heatmap_layer_x_activation(entries, metric, out, segment=seg)

        # By segment mode
        plot_segment_mode_comparison(entries, metric, out)

        # By percentile
        for seg in segments:
            plot_metric_by_percentile(entries, metric, out, segment=seg)

    # Summary dashboards
    for seg in segments:
        for pct in percentiles:
            plot_summary_dashboard(entries, out, percentile=pct, segment=seg)

    logger.info(f"Cross-axis plot generation complete. Output: {output_dir}")


@hydra.main(
    config_path="../configs/viz", config_name="cross_axis_plots", version_base=None
)
def main(cfg: DictConfig) -> None:
    generate_all_cross_axis_plots(cfg.grid_dir, cfg.output_dir)


if __name__ == "__main__":
    main()
