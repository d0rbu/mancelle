"""Plot harmfulness progression charts for conditioned SFT branches.

Usage:
    uv run python -m scripts.eval.plot_harmfulness_progression
    uv run python -m scripts.eval.plot_harmfulness_progression model_tag=d13_base
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig

from steering.eval.harmfulness_progression import (
    DEFAULT_COND_SOURCE_MAP,
    DEFAULT_CONFIG_LABEL_MAP,
    DEFAULT_STAGES,
    METRIC_SPECS,
    HarmfulnessRecord,
    SignalStyle,
    harmfulness_plot_signals,
    load_harmfulness_records,
    load_safety_progression_runs,
    render_progression_chart,
)


def save_harmfulness_progression_plots(
    records: Sequence[HarmfulnessRecord],
    output_dir: Path,
    *,
    source_to_stage: Mapping[str, int],
    stage_labels: Sequence[str],
    filename_prefix: str,
    title_template: str,
    signal_order: Sequence[str] | None = None,
    signal_styles: Mapping[str, SignalStyle] | None = None,
) -> None:
    """Render and close every shared harmfulness metric chart."""

    if signal_order is None or signal_styles is None:
        signal_order, signal_styles = harmfulness_plot_signals(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in METRIC_SPECS:
        figure = render_progression_chart(
            records,
            source_to_stage=source_to_stage,
            stage_labels=stage_labels,
            signal_order=signal_order,
            signal_styles=signal_styles,
            metric=spec.metric,
            ylabel=spec.ylabel,
            fmt=spec.fmt,
            title=title_template.format(ylabel=spec.ylabel),
        )
        output_path = output_dir / f"{filename_prefix}_{spec.suffix}.png"
        try:
            figure.savefig(output_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(figure)
        logger.info(f"  Saved {output_path}")


def _main(cfg: DictConfig) -> None:
    """Core logic, separated from Hydra wiring so tests can invoke directly."""
    base_dir = Path(cfg.base_dir)
    out_dir = Path(
        cfg.output_dir if cfg.output_dir else f"fig/harmfulness_progression/{cfg.model_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    progression_paths = cfg.get("progression_paths")
    if progression_paths:
        if cfg.get("safety_steering") is not None:
            raise ValueError("progression_paths plots all strength receipts; safety_steering filter must be null")
        records, source_to_stage = load_safety_progression_runs(
            [Path(path) for path in progression_paths], model_tag=cfg.model_tag)
        stage_labels = tuple(source_to_stage)
    else:
        records = load_harmfulness_records(
            base_dir,
            model_tag=cfg.model_tag,
            num_harmful=cfg.num_harmful,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP,
            safety_steering=cfg.get("safety_steering"),
        )
        source_to_stage, stage_labels = DEFAULT_COND_SOURCE_MAP, DEFAULT_STAGES

    logger.info(f"Generating progression plots for {cfg.model_tag}...")
    save_harmfulness_progression_plots(
        records,
        out_dir,
        source_to_stage=source_to_stage,
        stage_labels=stage_labels,
        filename_prefix="conditioned",
        title_template=f"{cfg.model_tag}: Conditioned SFT Progression - {{ylabel}}",
    )

    logger.info(f"\nAll plots saved to {out_dir}/")


@hydra.main(
    config_path="../../configs/plot",
    config_name="harmfulness_progression",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _main(cfg)


if __name__ == "__main__":
    main()
