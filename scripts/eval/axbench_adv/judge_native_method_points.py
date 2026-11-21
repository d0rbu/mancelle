"""Run final AxBench judges over completed native-method Adv points."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.native_method_judge import run_native_method_final_judge


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_adv",
    config_name="native_method_final_judge",
)
def main(cfg: DictConfig) -> None:
    run_native_method_final_judge(cfg)


if __name__ == "__main__":
    main()
