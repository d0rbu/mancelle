"""Precompute intruder-detection autointerp scores and a feature whitelist.

Usage:
    uv run -m scripts.preprocess.precompute_autointerp_filter -- \
        feature_dir=path/to/features output_dir=out/autointerp/my_run
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.autointerp_filter.cli import run_from_config


@hydra.main(
    version_base=None,
    config_path="../../configs/extraction",
    config_name="precompute_autointerp_filter",
)
def main(cfg: DictConfig) -> None:
    # @hydra.main discards the task function's return value; raising SystemExit
    # is what propagates a nonzero exit code to the shell.
    raise SystemExit(run_from_config(cfg))


if __name__ == "__main__":
    main()
