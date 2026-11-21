"""Fit and run native decoder DiffMean/LinearProbe counter-steering."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.native_counter import run_native_counter


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_adv",
    config_name="native_counter",
)
def main(cfg: DictConfig) -> None:
    run_native_counter(cfg)


if __name__ == "__main__":
    main()
