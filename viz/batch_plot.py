"""Batch generation of per-combo charts for all stats in a grid output.

Walks the grid directory, finds all feature_statistics.json files,
and generates the standard chart suite for each one.

Usage:
    uv run python -m viz.batch_plot
    uv run python -m viz.batch_plot grid_dir=out/data/my_grid dpi=300
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from steering.stats_io import glob_statistics, load_statistics
from viz._charts import (
    count_distribution,
    count_vs_magnitude,
    features_per_segment,
    features_per_token_ratio,
    magnitude_distribution,
    percentile_heatmap,
    segment_length_distribution,
)

CHART_FNS = {
    "count_distribution": count_distribution,
    "count_vs_magnitude": count_vs_magnitude,
    "features_per_segment": features_per_segment,
    "magnitude_distribution": magnitude_distribution,
    "percentile_heatmap": percentile_heatmap,
    "segment_length_distribution": segment_length_distribution,
    "features_per_token_ratio": features_per_token_ratio,
}


def batch_generate(grid_dir: str, output_dir: str, dpi: int = 150) -> None:
    """Generate all charts for every stats file in the grid directory."""
    grid_path = Path(grid_dir)
    out_base = Path(output_dir)

    stats_files = glob_statistics(grid_path)
    logger.info(f"Found {len(stats_files)} stats files in {grid_dir}")

    for stats_file in stats_files:
        rel = stats_file.relative_to(grid_path).parent
        combo_out = out_base / rel
        combo_out.mkdir(parents=True, exist_ok=True)

        try:
            stats = load_statistics(stats_file)
        except (ValueError, OSError) as e:
            logger.warning(f"Skipping {stats_file}: {e}")
            continue

        # Skip charts that need per_feature_stats if the data is empty
        has_per_feature = bool(stats.per_feature_stats)
        charts_needing_per_feature = {
            "count_distribution",
            "count_vs_magnitude",
            "magnitude_distribution",
            "percentile_heatmap",
        }

        for chart_name, chart_fn in CHART_FNS.items():
            if not has_per_feature and chart_name in charts_needing_per_feature:
                continue
            try:
                fig = chart_fn(stats)
                out_path = combo_out / f"{chart_name}.png"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate {chart_name} for {rel}: {e}")

        logger.info(f"Generated charts for {rel}")

    logger.info(f"Batch plot generation complete. Output: {output_dir}")


@hydra.main(config_path="../configs/viz", config_name="batch_plot", version_base=None)
def main(cfg: DictConfig) -> None:
    batch_generate(cfg.grid_dir, cfg.output_dir, dpi=cfg.dpi)


if __name__ == "__main__":
    main()
