"""Convert an autointerp ranked artifact into the chat TUI's description JSON."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from scripts.chat_tui.descriptions import (
    descriptions_from_autointerp_ranked,
    write_offline_descriptions,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/extraction",
    config_name="build_chat_tui_descriptions",
)
def main(cfg: DictConfig) -> int:
    ranked_yaml_path = Path(cfg.ranked_yaml_path)
    output_path = Path(cfg.output_path)
    descriptions = descriptions_from_autointerp_ranked(ranked_yaml_path)
    write_offline_descriptions(descriptions, output_path)
    logger.info(
        f"wrote {len(descriptions)} feature descriptions from {ranked_yaml_path} to {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
