"""Judge only NativeDecoderPromptSteering generations."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_upstream.native_judge import run_native_evaluation


@hydra.main(
    version_base=None,
    config_path="../../../configs/eval/axbench_upstream",
    config_name="native_evaluate",
)
def main(cfg: DictConfig) -> None:
    run_native_evaluation(cfg)


if __name__ == "__main__":
    main()
