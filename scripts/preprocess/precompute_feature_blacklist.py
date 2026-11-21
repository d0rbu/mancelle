"""Standalone pre-generation of the activation-frequency feature blacklist.

Thin Hydra wrapper around ``steering.conditional.freq_blacklist``. Single
process. Producer for PR 471's ``feature_mask`` — see
docs/pipelines/feature-extraction.md. Reads the precomputed
``feature_statistics.safetensors`` (run ``compute_statistics`` first).

Usage:
    uv run -m scripts.preprocess.precompute_feature_blacklist \
        feature_dir=path/to/features feature_mask_freq_min=1e-5 feature_mask_freq_max=0.5
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from steering.conditional.freq_blacklist import (
    generate_or_load_blacklist,
    resolve_blacklist_bounds,
)
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    parse_feature_dir,
    resolve_output_dir,
)
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAEWidth,
)
from steering.stats_io import find_statistics_path, load_statistics
from steering.stats_types import ExtractionStatistics


def _resolve_feature_dir(cfg: DictConfig) -> str:
    if cfg.feature_dir is not None:
        return str(cfg.feature_dir)
    return str(
        resolve_output_dir(
            output_dir=None,
            feature_root=cfg.feature_root,
            segment_size=cfg.segment_size,
            layer=cfg.sae.layer,
            activation_type=ActivationType(cfg.sae.activation_type),
            pooling=PoolingMode(cfg.pooling),
            width=SAEWidth(cfg.sae.width),
            model=cfg.sae.model,
            dataset_phase=cfg.dataset_phase,
            ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
        )
    )


def _resolve_n_sae_features_and_statistics(
    cfg: DictConfig, feature_dir: str,
) -> tuple[int, ExtractionStatistics | None]:
    """Resolve width and retain one complete statistics snapshot per invocation."""
    explicit = cfg.n_sae_features
    if explicit is not None:
        return int(explicit), None
    stats_path = find_statistics_path(feature_dir)
    if stats_path is not None:
        statistics = load_statistics(stats_path)
        return int(statistics.metadata.n_sae_features), statistics
    return SAE_WIDTH_TO_NUM_FEATURES[parse_feature_dir(feature_dir).width], None


def run(cfg: DictConfig) -> str | None:
    """Generate (or reuse) the blacklist. Returns the txt path, or ``None``
    when disabled or the blacklist is empty."""
    bounds = resolve_blacklist_bounds(cfg)
    if bounds is None:
        logger.info("feature_mask_freq_min/max both null — feature disabled, nothing to do.")
        return None
    feature_dir = _resolve_feature_dir(cfg)
    n_sae_features, statistics = _resolve_n_sae_features_and_statistics(cfg, feature_dir)
    logger.info(
        f"Generating freq-blacklist from feature_statistics: dir={feature_dir} "
        f"n_sae_features={n_sae_features} freq_min={bounds.min} "
        f"freq_max={bounds.max}"
    )
    out = generate_or_load_blacklist(
        feature_dir,
        n_sae_features,
        freq_min=bounds.min,
        freq_max=bounds.max,
        ddp_rank=0,
        barrier_fn=None,
        force=bool(cfg.force),
        **({"_statistics": statistics} if statistics is not None else {}),
    )
    if out is None:
        logger.info("Empty blacklist (no feature falls outside the kept interval) — no-op.")
        return None
    txt, result = out
    # Neutral wording: generate_or_load may have (re)written OR reused an
    # up-to-date artifact — both yield the same deterministic bytes.
    logger.info(
        f"Blacklist ready: {len(result.ids)} feature ids at {txt} "
        f"({result.n_segments} segments, {result.n_active_features} active "
        f"features in statistics)"
    )
    return str(txt)


@hydra.main(
    version_base=None,
    config_path="../../configs/extraction",
    config_name="precompute_feature_blacklist",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
