"""Plot cross-lineage harmfulness progression: d13_base + d12_baseline + standalone.

Usage:
    uv run python -m scripts.eval.plot_harmfulness_standalone_progression
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
    DEFAULT_SFT_STEPS,
    METRIC_SPECS,
    harmfulness_plot_signals,
    load_harmfulness_records,
    load_progression_data,
    render_standalone_progression,
)


def _main(cfg: DictConfig) -> None:
    """Core logic, separated from Hydra wiring so tests can invoke directly."""
    base_dir = Path(cfg.base_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_harmfulness_records(
        base_dir,
        model_tag=cfg.model_tag,
        num_harmful=cfg.num_harmful,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP,
        safety_steering=cfg.get("safety_steering"),
    )
    d13_prog = load_progression_data(
        Path(cfg.d13_progression_path), min_step=int(cfg.min_step)
    )
    d12_prog = load_progression_data(
        Path(cfg.d12_progression_path), min_step=int(cfg.min_step),
        conditioned_config=cfg.get("d12_conditioned_config"),
    )

    signal_order, signal_styles = harmfulness_plot_signals(records)

    logger.info("Generating standalone progression plots...")
    for spec in METRIC_SPECS:
        fig = render_standalone_progression(
            records,
            d13_progression=d13_prog,
            d12_progression=d12_prog,
            metric=spec.metric,
            ylabel=spec.ylabel,
            title=f"Harmfulness: Pretrain → Safety SFT → Cap SFT ({spec.ylabel})",
            last_pretrain_step=int(cfg.last_pretrain_step),
            sft_steps=DEFAULT_SFT_STEPS,
            signal_order=signal_order,
            signal_styles=signal_styles,
        )
        out_path = out_dir / f"progression_{spec.suffix}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {out_path}")

    logger.info(f"\nAll plots saved to {out_dir}/")


@hydra.main(
    config_path="../../configs/plot",
    config_name="harmfulness_standalone_progression",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _main(cfg)


if __name__ == "__main__":
    main()
