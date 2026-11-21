"""Validate and certify packed rows as compact FP-target source data."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from steering.conditional.encoder_decoder.fp_target_source import (
    certify_fp_target_source,
    fp_target_source_certificate_digest,
)
from steering.refined_types import PositiveInt


def run(cfg: DictConfig) -> int:
    packed_row_dir = Path(str(cfg.packed_row_dir)).expanduser()
    certificate = certify_fp_target_source(
        packed_row_dir,
        n_sae_features=PositiveInt.parse(int(cfg.n_sae_features)),
    )
    digest = fp_target_source_certificate_digest(certificate)
    logger.info(
        "Certified compact FP target source: dir={} rows={} segments={} "
        "features={} digest={}",
        packed_row_dir,
        certificate["rows_checked"],
        certificate["segments_checked"],
        certificate["feature_entries_checked"],
        digest,
    )
    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="validate_fp_target_source",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
