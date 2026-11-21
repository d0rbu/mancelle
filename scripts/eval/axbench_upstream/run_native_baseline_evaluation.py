"""Judge one native decoder standalone baseline artifact."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_upstream.native_baseline_judge import (
    run_native_baseline_evaluation,
)


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_upstream",
    config_name="native_baseline_evaluate",
)
def main(cfg: DictConfig) -> None:
    run_native_baseline_evaluation(cfg)


if __name__ == "__main__":
    main()
