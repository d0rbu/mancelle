"""Run native DiffMean, LinearProbe, and LoRA through AxBench-Adv."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.native_methods import run_native_method_adv


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_adv",
    config_name="native_methods",
)
def main(cfg: DictConfig) -> None:
    run_native_method_adv(cfg)


if __name__ == "__main__":
    main()
