"""Replay autointerp trial-audit rows against prompt variants."""

from __future__ import annotations

import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from steering.eval.autointerp_filter.prompt_calibration import (
    run_prompt_calibration_from_config,
)
from steering.eval.autointerp_filter.types import AutoInterpConfigError


@hydra.main(
    version_base=None,
    config_path="../../configs/eval",
    config_name="autointerp_prompt_calibration",
)
def main(cfg: DictConfig) -> int:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        logger.error(f"Expected mapping config, got {type(raw).__name__}")
        return 2
    try:
        run_prompt_calibration_from_config({str(k): v for k, v in raw.items()})
    except AutoInterpConfigError as exc:
        logger.error(f"Autointerp prompt calibration config failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
