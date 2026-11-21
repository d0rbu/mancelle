"""Build naturally elicited, SAE-verified negative AxBench references."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_negative.build import run_negative_build


@hydra.main(version_base=None, config_path="../../configs/eval/axbench_negative", config_name="build")
def main(cfg: DictConfig) -> None:
    run_negative_build(cfg)


if __name__ == "__main__":
    main()
