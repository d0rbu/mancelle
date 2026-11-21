"""Hydra entry point for native decoder SFT generation-based AxBench."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_upstream.native_decoder import run_native_decoder


@hydra.main(version_base=None, config_path="../../../configs/eval/axbench_upstream", config_name="native_decoder")
def main(cfg: DictConfig) -> None:
    run_native_decoder(cfg)


if __name__ == "__main__":
    main()
