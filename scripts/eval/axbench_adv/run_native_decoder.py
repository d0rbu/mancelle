"""Run target-only native decoder GEPA through a durable provider mailbox."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.native_decoder import run_native_adv


@hydra.main(version_base=None, config_path="../../../configs/eval/axbench_adv", config_name="native_decoder")
def main(cfg: DictConfig) -> None:
    run_native_adv(cfg)


if __name__ == "__main__":
    main()
