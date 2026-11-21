"""Prepare an immutable original-message UltraChat source for negative AxBench."""

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from steering.eval.axbench_negative.source import prepare_ultrachat_source


@hydra.main(
    version_base=None, config_path="../../configs/eval/axbench_negative", config_name="prepare"
)
def main(cfg: DictConfig) -> None:
    manifest = prepare_ultrachat_source(
        output_dir=Path(cfg.output_dir),
        row_count=cfg.row_count,
        snapshot_dir=None if cfg.snapshot_dir is None else Path(cfg.snapshot_dir),
        revision=cfg.revision,
        expected_parquet_sha256=cfg.expected_parquet_sha256,
    )
    logger.info(
        "Prepared {} original test conversations at {}", manifest["source"]["rows"], cfg.output_dir
    )


if __name__ == "__main__":
    main()
