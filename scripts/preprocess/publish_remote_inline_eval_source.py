"""Publish the certified sparse source for remote clustered inline evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from steering.remote_inline_eval import publish_remote_inline_eval_source


def run(cfg: DictConfig) -> int:
    feature_dir = Path(str(cfg.feature_dir)).expanduser()
    data_dir = str(cfg.data_dir) if cfg.data_dir is not None else None
    fingerprint_cache_dir = (
        Path(str(cfg.fingerprint_cache_dir)).expanduser()
        if cfg.fingerprint_cache_dir is not None
        else None
    )
    certificate = publish_remote_inline_eval_source(
        feature_dir=feature_dir,
        data_dir=data_dir,
        stem=str(cfg.stem),
        fingerprint_cache_dir=fingerprint_cache_dir,
    )
    raw = certificate["raw_feature"]
    token = certificate["token_boundary"]
    parquet = certificate["parquet"]
    logger.info(
        "Published certified sparse remote inline-eval source: stem={} "
        "source_bytes={} aggregate={}",
        certificate["stem"],
        raw["byte_length"] + token["byte_length"] + parquet["byte_length"],
        certificate["aggregate_sha256"],
    )
    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="publish_remote_inline_eval_source",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
