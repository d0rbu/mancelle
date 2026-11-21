"""Shared helpers for the preprocess pipeline scripts.

Consolidates code that was previously duplicated across the three precompute
scripts (``precompute_percentiles.py``, ``precompute_token_boundaries.py``,
``precompute_packed_rows.py``). The duplicated body is the feature-directory
resolution: either honour an explicit ``cfg.feature_dir`` override or build
the canonical structured path from the SAE config.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    resolve_output_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth


def resolve_feature_dir(cfg: DictConfig) -> Path:
    """Resolve the feature directory from an extraction config.

    Honours an explicit ``cfg.feature_dir`` override; otherwise builds the
    canonical path via ``steering.feature_paths.resolve_output_dir`` from the
    structured SAE config (``activation_type``, ``pooling``, ``sae_width``,
    ``segment_size``, ``layer``, ``model``, ``dataset_phase``,
    ``feature_root``).

    Returns:
        The feature directory path. Existence is **not** checked here — the
        caller decides what to do for missing dirs.
    """
    if cfg.feature_dir is not None:
        return Path(cfg.feature_dir)

    activation_type = ActivationType(cfg.activation_type)
    pooling = PoolingMode(cfg.pooling)
    width = SAEWidth(cfg.sae_width)
    return resolve_output_dir(
        output_dir=None,
        feature_root=cfg.feature_root,
        segment_size=cfg.segment_size,
        layer=cfg.layer,
        activation_type=activation_type,
        pooling=pooling,
        width=width,
        model=cfg.model,
        dataset_phase=cfg.dataset_phase,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
    )
