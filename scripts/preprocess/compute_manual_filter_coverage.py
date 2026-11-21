"""Compute per-segment coverage for a manual autointerp feature filter."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from steering.eval.autointerp_filter.manual_coverage import (
    compute_manual_filter_coverage,
    dump_manual_coverage_report,
)
from steering.eval.autointerp_filter.types import AutoInterpConfigError


@hydra.main(
    version_base=None,
    config_path="../../configs/extraction",
    config_name="compute_manual_filter_coverage",
)
def main(cfg: DictConfig) -> int:
    if cfg.feature_dir is None:
        raise AutoInterpConfigError("feature_dir is required")
    if cfg.selection_yaml_path is None:
        raise AutoInterpConfigError("selection_yaml_path is required")
    report = compute_manual_filter_coverage(
        feature_dir=str(cfg.feature_dir),
        selection_yaml_path=str(cfg.selection_yaml_path),
        exclude_last_shard=bool(cfg.exclude_last_shard),
    )
    print(dump_manual_coverage_report(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
