"""Shared feature selection pipeline helpers.

Extracts the threshold-vs-sampling dispatch pattern into reusable functions
so that all scripts (training, eval, profiling, conversion, SFT) can support
both ThresholdSelection and SamplingSelection without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from beartype import beartype
from omegaconf import OmegaConf

from steering.conditional.delta import FeatureDeltaConfig
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask, resolve_feature_mask
from steering.conditional.sampling import (
    FeatureSelection,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
    build_percentile_table,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    build_thresholds_from_stats,
    extract_threshold_percentile,
)
from steering.refined_types import PositiveIntLike
from steering.removed_features import reject_feature_clustering_config
from steering.stats_types import PerFeatureStat

DEFAULT_THRESHOLD_FEATURE_ESTIMATE = 512


def feature_selection_dirname(fs: FeatureSelection) -> str:
    """Return checkpoint subdirectory path component for a feature selection config.

    ThresholdSelection → "t{threshold}" (single dir)
    SamplingSelection  → "dist_{dist}/mean_{mean}/std_{std}" (nested)
    """
    if isinstance(fs, ThresholdSelection):
        return f"t{fs.threshold}"
    if isinstance(fs, SamplingSelection):
        return f"dist_{fs.distribution.value}/mean_{fs.mean}/std_{fs.std}"
    raise TypeError(f"Unsupported feature selection type: {type(fs).__name__}")


def append_gradient_leaf(path: str, fs: FeatureSelection) -> str:
    """Append the gradient-attribution checkpoint leaf when configured.

    Mirrors ``append_mask_leaf`` (I2): a non-gradient selection returns
    ``path`` byte-identically, so legacy checkpoint paths never change. A
    gradient-ranked run gets a distinct leaf so it can never clobber its
    percentile-ranked twin: ``grad_{mode}_mf{min_final}[_fp32]``.
    """
    gradient = getattr(fs, "gradient", None)
    if gradient is None:
        return path
    leaf = f"grad_{gradient.mode.value}_mf{gradient.min_final_segment_tokens}"
    if gradient.model_dtype == "float32":
        leaf += "_fp32"
    return str(Path(path) / leaf)


@dataclass
class ThresholdSelectionConfig:
    """Resolved threshold pipeline state."""

    selection: ThresholdSelection
    precomputed_thresholds: t.Tensor | None  # None when threshold_type doesn't need it


@dataclass
class DeltaBundle:
    """Delta conditioning config + precomputed percentile table."""

    config: FeatureDeltaConfig
    percentile_table: t.Tensor  # [n_sae_features, 99] float64


@beartype
def active_bucket_count(
    num_buckets: PositiveIntLike,
    *,
    reserves_negative_bucket: bool,
) -> PositiveIntLike:
    """Return the bucket count available to active (positive) features.

    A negative-sampling checkpoint reserves the final encoder bucket for an
    explicitly inactive feature. Positive features must therefore be
    discretized into ``num_buckets - 1`` buckets. Without that reservation,
    they use the full encoder bucket space.

    ``@beartype`` owns both unary contracts: ``num_buckets`` is a strict
    positive integer through :class:`~steering.refined_types.PositiveIntLike`,
    and ``reserves_negative_bucket`` is a strict boolean. This helper is the
    authority used by :func:`build_feature_pipeline`; callers that reconstruct
    a checkpoint's selection contract should use the same rule rather than
    open-coding it.

    Raises:
        ValueError: Reserving the final bucket would leave no bucket for active
            features.
    """
    if not reserves_negative_bucket:
        return num_buckets
    if num_buckets < 2:
        raise ValueError(
            f"num_buckets={num_buckets} cannot reserve a negative/inactive "
            "bucket: the reservation leaves effective_buckets=0 — an empty "
            "positive bucket space. Use num_buckets >= 2 for a checkpoint "
            "with negative sampling."
        )
    return num_buckets - 1


@dataclass
class FeaturePipelineConfig:
    """Resolved feature pipeline state — everything needed to process features."""

    selection_config: ThresholdSelectionConfig | SamplingBundle
    boundaries: t.Tensor  # discretization boundaries (both paths need this)
    negative_bundle: NegativeBundle | None
    effective_buckets: int
    delta_bundle: DeltaBundle | None = None
    feature_mask: FeatureMask | None = None


def negative_sampling_mask(
    pipeline: FeaturePipelineConfig,
    consolidation: FeatureClusterConsolidation | None = None,
) -> FeatureMask | None:
    """The single feature mask negative samplers may draw from.

    ``consolidation is None`` (every unclustered caller, including the live
    pretrain dataloader, which rejects clusters outright — see
    ``_validate_feature_clusters_training_config``) returns
    ``pipeline.feature_mask`` byte-identically.

    When clustering IS active the plain mask is the wrong pool: it still
    contains every consolidated-away cluster sibling, and a negative drawn from
    one of those reaches a compacted encoder that has no row for it and trips
    ``FeatureClusterConsolidation.assert_representatives_only``. The
    representative-only pool is ``effective_negative_mask()``, whose allowed
    universe is exactly ``cluster_allowed_universe`` — the same call the
    clustered precompute writer makes
    (``scripts/preprocess/precompute_selected_features.py``), which is where
    every clustered pretraining run's negatives actually came from.
    """
    if consolidation is not None:
        return consolidation.effective_negative_mask()
    return pipeline.feature_mask


def append_negative_features(
    fids: t.Tensor,
    buckets: t.Tensor,
    pcts: t.Tensor,
    neg_fids: t.Tensor,
    *,
    effective_buckets: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Append sampled inactive features to one segment's positive selection.

    The single owner of the "negatives ride in the reserved inactive row"
    encoding: every negative id is written with bucket ``effective_buckets``
    (``num_buckets - 1``) and percentile 0.0, appended after the positives. The
    packed-row writer's ``enc_mags[...] = effective_buckets``
    (``steering/packed_batch_io.py``) is the same contract on the precomputed
    side.

    Callers: the live pretrain dataloader
    (``steering/conditional/encoder_decoder/dataloader.py``) and conditioned SFT
    (``scripts/train/conditional_encdec_sft.py``). Both must produce the same
    encoding, so neither re-derives it.

    Returns ``(fids, buckets, pcts)`` unchanged when ``neg_fids`` is empty, so a
    no-negatives run is bit-identical to not calling this at all.
    """
    if neg_fids.numel() == 0:
        return fids, buckets, pcts
    neg_buckets = t.full((len(neg_fids),), effective_buckets, dtype=t.int64)
    return (
        t.cat([fids.to(t.int64), neg_fids]),
        t.cat([buckets, neg_buckets]),
        t.cat([pcts, t.zeros(len(neg_fids), dtype=t.float32)]),
    )


def build_feature_pipeline(
    feature_selection: FeatureSelection,
    negative_selection: NegativeSamplingSelection | None,
    per_feature_stats: dict[str, PerFeatureStat],
    n_sae_features: int,
    num_buckets: int,
    seed: int = 42,
    ddp_rank: int = 0,
    feature_delta: FeatureDeltaConfig | None = None,
    feature_mask: FeatureMask | None = None,
) -> FeaturePipelineConfig:
    """Build the complete feature pipeline config from parsed selections.

    Handles threshold vs sampling dispatch, RNG seeding, boundary/table construction.
    Shared RNG invariant: sampling path shares rng between SamplingBundle and
    NegativeBundle; threshold path gives NegativeBundle its own rng.
    """
    # --- Cross-validation ---
    if isinstance(feature_selection, SamplingSelection):
        if feature_delta is not None and feature_selection.delta_weights is None:
            raise ValueError(
                "SamplingSelection requires delta_weights to be set when "
                "feature_delta is enabled. Set delta_weights to 'percentile_delta' "
                "or 'absolute_percentile' in the feature_selection config."
            )
        if feature_delta is None and feature_selection.delta_weights is not None:
            raise ValueError(
                "SamplingSelection has delta_weights set but feature_delta is None. "
                "Either set feature_delta config or remove delta_weights."
            )

    effective_buckets = active_bucket_count(
        num_buckets,
        reserves_negative_bucket=negative_selection is not None,
    )

    negative_bundle: NegativeBundle | None = None
    selection_config: ThresholdSelectionConfig | SamplingBundle
    percentile_table: t.Tensor | None = None

    if isinstance(feature_selection, ThresholdSelection):
        threshold_cfg = ThresholdConfig(
            feature_selection.threshold_type,
            feature_selection.threshold,
        )
        needs_feature_thresholds = feature_selection.threshold_type in (
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        )
        threshold_pct = (
            extract_threshold_percentile(threshold_cfg)
            if needs_feature_thresholds
            else None
        )

        precomputed_thresholds: t.Tensor | None = None

        if effective_buckets == 1:
            # Binary mode: bucket 0 = active, bucket 1 = inactive.
            boundaries = t.zeros(n_sae_features, 0, dtype=t.float64)
        else:
            boundaries = build_boundaries_from_stats(
                per_feature_stats,
                n_sae_features,
                effective_buckets,
                threshold_percentile=threshold_pct,
            )

        if needs_feature_thresholds:
            precomputed_thresholds = build_thresholds_from_stats(
                per_feature_stats,
                n_sae_features,
                feature_selection.threshold_type,
                feature_selection.threshold,
            )

        # Build percentile table if delta conditioning requires it
        if feature_delta is not None:
            percentile_table = build_percentile_table(per_feature_stats, n_sae_features)

        selection_config = ThresholdSelectionConfig(
            selection=feature_selection,
            precomputed_thresholds=precomputed_thresholds,
        )

        # Threshold path: NegativeBundle gets its own separate RNG
        if negative_selection is not None:
            neg_rng = t.Generator().manual_seed(seed + ddp_rank)
            negative_bundle = NegativeBundle(
                selection=negative_selection,
                n_sae_features=n_sae_features,
                rng=neg_rng,
                feature_mask=feature_mask,
            )

    elif isinstance(feature_selection, SamplingSelection):
        rng = t.Generator().manual_seed(seed + ddp_rank)

        percentile_table = build_percentile_table(per_feature_stats, n_sae_features)

        if effective_buckets == 1:
            boundaries = t.zeros(n_sae_features, 0, dtype=t.float64)
        else:
            boundaries = build_boundaries_from_stats(
                per_feature_stats,
                n_sae_features,
                effective_buckets,
                threshold_percentile=None,
            )

        selection_config = SamplingBundle(
            selection=feature_selection,
            percentile_table=percentile_table,
            rng=rng,
        )

        # Sampling path: NegativeBundle shares RNG with SamplingBundle
        if negative_selection is not None:
            negative_bundle = NegativeBundle(
                selection=negative_selection,
                n_sae_features=n_sae_features,
                rng=rng,
                feature_mask=feature_mask,
            )

    else:
        raise TypeError(
            f"Unsupported feature selection type: {type(feature_selection).__name__}"
        )

    # Build DeltaBundle if delta conditioning is enabled
    delta_bundle: DeltaBundle | None = None
    if feature_delta is not None:
        assert percentile_table is not None  # ensured by branches above
        delta_bundle = DeltaBundle(
            config=feature_delta,
            percentile_table=percentile_table,
        )

    return FeaturePipelineConfig(
        selection_config=selection_config,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        delta_bundle=delta_bundle,
        feature_mask=feature_mask,
    )


def build_feature_pipeline_from_cfg_block(
    cfg_block: Mapping[str, Any],
    *,
    per_feature_stats: dict[str, PerFeatureStat],
    n_sae_features: int,
    num_buckets: int,
    seed: int = 42,
    ddp_rank: int = 0,
    use_negative_selection: bool = True,
    feature_delta: FeatureDeltaConfig | None = None,
    feature_mask: FeatureMask | None = None,
) -> FeaturePipelineConfig:
    """Parse feature_selection / negative_selection from a cfg block and build
    the pipeline.

    cfg_block: OmegaConf DictConfig or plain dict carrying at least
        ``feature_selection``. ``negative_selection`` is optional and defaults
        to None when missing or null.

    use_negative_selection: when False, force ``negative_selection=None`` even
        if cfg_block.negative_selection is set (matches
        completion_steering_eval's "eval doesn't use negative sampling"
        stance).

    feature_delta: pre-parsed FeatureDeltaConfig (or None). The helper does
        NOT parse feature_delta from the cfg block because 3 of 5 callers
        don't use it; explicit construction at the call site is clearer.
        Use parse_feature_delta(...) at the call site if needed.
    """
    if not isinstance(cfg_block, Mapping):
        raise TypeError(
            f"cfg_block must be a mapping, got {type(cfg_block).__name__}"
        )
    reject_feature_clustering_config(cfg_block, "feature pipeline")

    fs_raw = cfg_block.get("feature_selection")
    if fs_raw is None:
        raise TypeError(
            "cfg_block must define 'feature_selection'; got None or missing key"
        )
    fs_d = (
        OmegaConf.to_container(fs_raw, resolve=True)
        if OmegaConf.is_config(fs_raw)
        else fs_raw
    )
    if not isinstance(fs_d, dict):
        raise TypeError(
            f"cfg_block.feature_selection must resolve to a mapping; got "
            f"{type(fs_d).__name__}"
        )
    feature_selection = parse_feature_selection(fs_d)

    if use_negative_selection:
        neg_raw = cfg_block.get("negative_selection")
        if neg_raw is None:
            negative_selection = None
        else:
            neg_d = (
                OmegaConf.to_container(neg_raw, resolve=True)
                if OmegaConf.is_config(neg_raw)
                else neg_raw
            )
            if neg_d is not None and not isinstance(neg_d, dict):
                raise TypeError(
                    f"cfg_block.negative_selection must resolve to a mapping "
                    f"or null; got {type(neg_d).__name__}"
                )
            negative_selection = parse_negative_selection(neg_d)
    else:
        negative_selection = None

    if feature_mask is None:
        feature_mask = resolve_feature_mask(cfg_block, n_sae_features)

    return build_feature_pipeline(
        feature_selection=feature_selection,
        negative_selection=negative_selection,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        seed=seed,
        ddp_rank=ddp_rank,
        feature_delta=feature_delta,
        feature_mask=feature_mask,
    )


def avg_features_per_segment_estimate(
    feature_selection: FeatureSelection,
    negative_selection: NegativeSamplingSelection | None = None,
) -> int:
    """Estimate avg features per segment for FLOPs and embedding table sizing.

    Includes both positive and negative (inactive) features when
    ``negative_selection`` is provided.
    """
    if isinstance(feature_selection, ThresholdSelection):
        if feature_selection.threshold_type in (
            ThresholdType.segment_topk,
            ThresholdType.feature_percentile_segment_topk,
        ):
            pos = int(feature_selection.threshold) // 2
        else:
            pos = DEFAULT_THRESHOLD_FEATURE_ESTIMATE
    elif isinstance(feature_selection, SamplingSelection):
        # When max_k is None, sample_k offsets by min_k: effective_mean = mean + min_k
        pos = max(1, int(feature_selection.mean) + feature_selection.min_k)
    else:
        raise TypeError(
            f"Unsupported feature selection type: {type(feature_selection).__name__}"
        )
    if negative_selection is not None:
        neg = int(negative_selection.mean) + negative_selection.min_k
    else:
        neg = 0
    return max(1, pos + neg)
