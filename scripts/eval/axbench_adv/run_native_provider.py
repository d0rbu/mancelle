"""Consume authorized native Adv CPU provider request artifacts."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.native_provider import run_provider


@hydra.main(version_base=None, config_path="../../../configs/eval/axbench_adv", config_name="native_provider")
def main(cfg: DictConfig) -> None:
    run_provider(cfg)


if __name__ == "__main__":
    main()
