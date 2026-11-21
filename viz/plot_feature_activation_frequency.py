"""Histogram of per-feature segment-activation frequency.

Reads ``feature_statistics.safetensors`` for a feature directory, computes
``count / n_segments`` for every active feature, and plots a log-binned
histogram. Outputs a PNG + a CSV (bin counts) + a text summary listing
how many features exceed common thresholds.

The metric is fraction of POST-POOLING SEGMENTS active (extraction pools
``segment_size`` tokens per segment before SAE encoding) — not fraction of
tokens. The per-token frequency is not recoverable from the statistics
artifact. See ``docs/reference/configuration.md``.

Usage:
    uv run -m viz.plot_feature_activation_frequency -- \\
        feature_dir=path/to/feature_dir
"""
from __future__ import annotations

import csv
from pathlib import Path

import hydra
import matplotlib

# Headless backend — safe in tests and on CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from steering.stats_io import find_statistics_path, load_statistics  # noqa: E402
from steering.stats_types import ExtractionStatistics  # noqa: E402


def compute_frequency_histogram(
    stats: ExtractionStatistics, *, n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, edges, counts)`` for the active-feature frequencies.

    - ``freqs[i] = stats.per_feature_stats[fid_i].count /
      stats.features.n_segments`` in iteration order over per_feature_stats.
    - ``edges`` is a log-spaced array of ``n_bins + 1`` bin edges from
      ``1 / n_segments`` (the smallest possible non-zero freq) to ``1.0``.
    - ``counts[i]`` is the number of active features whose freq falls in the
      i-th bin (numpy histogram conventions; the last bin is right-closed).

    ``n_segments == 0`` ⇒ returns three empty arrays (no histogram possible).
    Inactive features (count == 0) are absent from ``per_feature_stats`` by
    construction and so are absent from the histogram; their count is
    reported separately by ``threshold_summary`` callers via
    ``len(stats.per_feature_stats)`` vs ``stats.metadata.n_sae_features``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    n_segments = int(stats.features.n_segments)
    if n_segments <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty.astype(np.int64)
    freqs = np.array(
        [pfs.count / n_segments for pfs in stats.per_feature_stats.values()],
        dtype=np.float64,
    )
    lo = 1.0 / n_segments
    edges = np.logspace(np.log10(lo), 0.0, n_bins + 1)
    counts, _ = np.histogram(freqs, bins=edges)
    return freqs, edges, counts


def threshold_summary(
    freqs: np.ndarray, *, threshold_lines: list[float]
) -> list[tuple[float, int]]:
    """For each threshold, how many active features have freq strictly >."""
    return [(float(t), int((freqs > t).sum())) for t in threshold_lines]


def plot_frequency_histogram(
    stats: ExtractionStatistics,
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    threshold_lines: list[float],
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 6))
    if counts.size:
        widths = np.diff(edges)
        ax.bar(
            edges[:-1],
            counts,
            width=widths,
            align="edge",
            edgecolor="black",
            linewidth=0.3,
            color="#4c72b0",
        )
    ax.set_xscale("log")
    ax.set_xlabel(
        "per-feature activation frequency  (count / n_segments;  "
        "fraction of POST-POOLING segments active — see "
        "docs/reference/configuration.md)"
    )
    ax.set_ylabel("number of features")
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.3)
    ymax = float(max(counts.max() if counts.size else 1, 1))
    for thr in threshold_lines:
        ax.axvline(thr, linestyle="--", color="firebrick", alpha=0.7)
        ax.text(
            thr,
            ymax * 0.95,
            f" thr={thr}",
            rotation=90,
            color="firebrick",
            va="top",
            fontsize=8,
        )
    n_active = len(stats.per_feature_stats)
    n_total = stats.metadata.n_sae_features
    n_inactive = n_total - n_active
    summary = (
        f"n_sae_features = {n_total:,}\n"
        f"n_active       = {n_active:,} "
        f"({n_active / max(n_total, 1) * 100:.2f}%)\n"
        f"n_inactive     = {n_inactive:,} (freq = 0; excluded from hist)\n"
        f"n_segments     = {stats.features.n_segments:,}\n"
        f"pooling        = {stats.metadata.pooling_mode} "
        f"(segment_size = {stats.metadata.segment_size})"
    )
    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    return fig


def write_csv(
    csv_path: Path, edges: np.ndarray, counts: np.ndarray
) -> None:
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "n_features"])
        for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
            w.writerow([f"{float(lo):.6e}", f"{float(hi):.6e}", int(c)])


def run(cfg: DictConfig) -> dict[str, str]:
    """Compute + save the histogram. Returns the output paths."""
    feature_dir = Path(cfg.feature_dir)
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise FileNotFoundError(
            f"feature_statistics.safetensors not found in {feature_dir}"
        )
    stats = load_statistics(stats_path)

    n_bins = int(cfg.get("n_bins", 50))
    threshold_lines = [
        float(t) for t in cfg.get("threshold_lines", [0.001, 0.01, 0.1, 0.5])
    ]
    out_dir = Path(cfg.output_dir) if cfg.get("output_dir") else feature_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / cfg.get(
        "output_name", "feature_activation_frequency_hist.png"
    )
    csv_path = out_dir / (Path(png_path).stem + ".csv")

    freqs, edges, counts = compute_frequency_histogram(stats, n_bins=n_bins)
    title = cfg.get("title") or (
        f"Feature activation frequency — {feature_dir}"
    )
    fig = plot_frequency_histogram(
        stats, edges, counts, threshold_lines=threshold_lines, title=title
    )
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    write_csv(csv_path, edges, counts)

    logger.info(f"Saved histogram to {png_path}")
    logger.info(f"Saved bin counts to {csv_path}")
    logger.info(
        f"n_sae_features={stats.metadata.n_sae_features:,}  "
        f"n_active={len(stats.per_feature_stats):,}  "
        f"n_segments={stats.features.n_segments:,}  "
        f"pooling={stats.metadata.pooling_mode}"
    )
    for thr, n in threshold_summary(freqs, threshold_lines=threshold_lines):
        logger.info(
            f"  features with count/n_segments > {thr}: {n:,}"
        )
    return {"png": str(png_path), "csv": str(csv_path)}


@hydra.main(
    version_base=None,
    config_path="../configs/extraction",
    config_name="plot_feature_activation_frequency",
)
def main(cfg: DictConfig) -> None:  # pragma: no cover — thin Hydra wrapper
    run(cfg)


if __name__ == "__main__":
    main()
