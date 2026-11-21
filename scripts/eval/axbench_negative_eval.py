"""Evaluate signed steering against a frozen natural-chat negative bank."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_negative.evaluate import run_negative_evaluation


@hydra.main(version_base=None, config_path="../../configs/eval/axbench_negative", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_negative_evaluation(cfg)


if __name__ == "__main__":
    main()
