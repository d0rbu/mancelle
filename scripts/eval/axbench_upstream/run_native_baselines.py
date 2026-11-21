"""Fit and generate native decoder DiffMean, LinearProbe, and LoRA."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_upstream.native_baselines import run_native_baselines


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_upstream",
    config_name="native_baselines",
)
def main(cfg: DictConfig) -> None:
    run_native_baselines(cfg)


if __name__ == "__main__":
    main()
