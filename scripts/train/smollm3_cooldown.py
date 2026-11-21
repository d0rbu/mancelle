"""Torchrun entrypoint for the Transformers-native SmolLM3 cooldown."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from nanochat.hydra_cli import parse_config_name_and_overrides
from steering.smollm3_cooldown import run_training


def load_config(argv: list[str]) -> DictConfig:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
    config_name, overrides = parse_config_name_and_overrides(
        argv,
        default_config_name="smollm3_cooldown",
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    run_training(load_config(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
