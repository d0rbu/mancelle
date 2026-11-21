"""Hydra entry point for feature statistics visualization.

Usage:
    python -m viz.plot_features
    python -m viz.plot_features stats_path=path/to/stats.json output_dir=fig
    python -m viz.plot_features charts=[count_distribution,count_vs_magnitude]
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig

from steering.stats_io import load_statistics
from steering.stats_types import ExtractionStatistics
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

CHART_REGISTRY: dict[str, Callable[..., Figure]] = {
    "count_distribution": count_distribution,
    "count_vs_magnitude": count_vs_magnitude,
    "features_per_segment": features_per_segment,
    "magnitude_distribution": magnitude_distribution,
    "percentile_heatmap": percentile_heatmap,
    "percentile_surface_3d": percentile_surface_3d,
    "segment_length_distribution": segment_length_distribution,
    "features_per_token_ratio": features_per_token_ratio,
}


def load_stats(stats_path: str) -> ExtractionStatistics:
    """Load feature statistics from safetensors."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    return load_statistics(path)


@hydra.main(
    config_path="../configs/extraction", config_name="plot_features", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Generate feature visualization PNGs."""
    stats = load_stats(cfg.stats_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = list(CHART_REGISTRY.keys()) if cfg.charts == "all" else list(cfg.charts)

    for chart_name in charts:
        if chart_name not in CHART_REGISTRY:
            logger.warning(f"Unknown chart: {chart_name}, skipping")
            continue

        logger.info(f"Generating {chart_name}...")
        chart_fn = CHART_REGISTRY[chart_name]
        chart_cfg = cfg.get(chart_name, {})
        chart_kwargs = dict(chart_cfg) if chart_cfg else {}
        fig = chart_fn(stats, **chart_kwargs)

        out_path = output_dir / f"{chart_name}.png"
        fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
