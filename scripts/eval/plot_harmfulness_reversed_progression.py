"""Plot reverse-order harmfulness progression charts.

Usage:
    uv run python -m scripts.eval.plot_harmfulness_reversed_progression
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig

from steering.eval.harmfulness_progression import (
    DEFAULT_CONFIG_LABEL_MAP,
    DEFAULT_REVERSED_COND_SOURCE_MAP,
    DEFAULT_REVERSED_STAGES,
    METRIC_SPECS,
    harmfulness_plot_signals,
    load_harmfulness_records,
    render_progression_chart,
)


def _main(cfg: DictConfig) -> None:
    """Core logic, separated from Hydra wiring so tests can invoke directly."""
    base_dir = Path(cfg.base_dir)
    out_dir = Path(
        cfg.output_dir
        if cfg.output_dir
        else f"fig/harmfulness_reversed_progression/{cfg.model_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_harmfulness_records(
        base_dir,
        model_tag=cfg.model_tag,
        # Coerce: cfg.num_harmful may be a string when resolved via Hydra env-var
        # interpolation. The library compares int-vs-int, so a string would
        # silently filter out all records and produce blank charts.
        num_harmful=int(cfg.num_harmful),
        config_label_map=DEFAULT_CONFIG_LABEL_MAP,
        safety_steering=cfg.get("safety_steering"),
    )

    signal_order, signal_styles = harmfulness_plot_signals(records)

    logger.info(f"Generating reverse-order progression plots for {cfg.model_tag}...")

    for spec in METRIC_SPECS:
        fig = render_progression_chart(
            records,
            source_to_stage=DEFAULT_REVERSED_COND_SOURCE_MAP,
            stage_labels=DEFAULT_REVERSED_STAGES,
            signal_order=signal_order,
            signal_styles=signal_styles,
            metric=spec.metric,
            ylabel=spec.ylabel,
            fmt=spec.fmt,
            title=(
                f"{cfg.model_tag}: Reverse Conditioned SFT Progression - {spec.ylabel}"
            ),
        )
        out_path = out_dir / f"reverse_conditioned_{spec.suffix}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {out_path}")

    logger.info(f"\nAll plots saved to {out_dir}/")


@hydra.main(
    config_path="../../configs/plot",
    config_name="harmfulness_reversed_progression",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _main(cfg)


if __name__ == "__main__":
    main()
