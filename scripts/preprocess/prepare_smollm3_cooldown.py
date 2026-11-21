"""Prepare deterministic SmolLM3 cooldown token/index shards."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig

from nanochat.hydra_cli import parse_config_name_and_overrides
from steering.smollm3_cooldown import load_source_tokenizer, prepare_dataset


def load_config(argv: list[str]) -> DictConfig:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
    config_name, overrides = parse_config_name_and_overrides(
        argv,
        default_config_name="smollm3_cooldown",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(sys.argv[1:] if argv is None else argv)
    tokenizer = load_source_tokenizer(local_files_only=bool(cfg.source.local_files_only))
    manifest = prepare_dataset(
        parquet_dir=Path(str(cfg.data.parquet_dir)),
        output_dir=Path(str(cfg.data.prepared_dir)),
        tokenizer=tokenizer,
        batch_size=int(cfg.data.tokenizer_batch_size),
    )
    logger.info(f"SmolLM3 cooldown dataset is complete: {manifest}")


if __name__ == "__main__":
    main()
