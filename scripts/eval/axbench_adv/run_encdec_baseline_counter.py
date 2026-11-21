"""Run the explicit encoder-decoder baseline latent counter campaign."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_adv.encdec_baseline_counter import run_encdec_baseline_counter


@hydra.main(
    config_path="../../../configs/eval/axbench_adv",
    config_name="encdec_baseline_counter",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    result = run_encdec_baseline_counter(cfg)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
