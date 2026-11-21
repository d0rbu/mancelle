"""Validate a counter continuation without opening a model or provider."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.pipeline import run_pipeline


@hydra.main(
    config_path="../../../configs/eval/axbench_adv", config_name="axbench_adv", version_base=None
)
def main(cfg: DictConfig) -> None:
    receipt = run_pipeline(build_config(cfg), counter_resume_preflight_only=True)
    logger.info("Counter replay preflight passed; no native completion manifest: {}", receipt)


if __name__ == "__main__":
    main()
