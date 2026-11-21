"""Shared AxBench pure-model selection and example-local random streams.

Cadence belongs to the caller: AxBench keeps its historical defaults, whereas
ChatCORE resolves its separately measured checkpoint cadence. The provider owns
compact-row translation through the model's exact FeatureVocab.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final

import torch as t

from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfCondScoreInterpolation,
    SelfCondScorePolicy,
    selection_without_delta,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import active_bucket_count
from steering.conditional.sampling import (
    SamplingBundle,
    SamplingSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.refined_types import PositiveInt, UnitInterval
from steering.segmentation import SegmentMode

SELF_CONDITIONING_SEED_VERSION: Final = "chatcore_self_conditioning_sha256_v1"


class SeedDomain(StrEnum):
    TOKENS = "tokens"
    FEATURE_SELECTION = "feature_selection"


def self_conditioning_example_seeds(
    *,
    base_seed: int,
    task_name: str,
    original_problem_index: int,
    prompt_token_key: tuple[int, ...],
    sample_index: int,
) -> tuple[int, int]:
    """Hash the canonical task name and exact rendered key, independent of work order.

    Callers supply the task's canonical name, not an alias or a shard label.
    There is deliberately no rank, shard offset, job, or iteration argument.
    """
    def derive(domain: SeedDomain) -> int:
        payload = {
            "version": SELF_CONDITIONING_SEED_VERSION,
            "domain": domain.value,
            "base_seed": base_seed,
            "task_name": task_name,
            "original_problem_index": original_problem_index,
            "prompt_token_key": prompt_token_key,
            "sample_index": sample_index,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    return derive(SeedDomain.TOKENS), derive(SeedDomain.FEATURE_SELECTION)


def build_self_conditioning_config(
    *,
    model: Any,
    positive_selection: SamplingSelection,
    percentile_table: t.Tensor,
    n_sae_features: int,
    feature_mask: FeatureMask | None,
    consolidation: FeatureClusterConsolidation | None,
    segment_mode: SegmentMode,
    segment_size: int,
    policy: SelfCondScorePolicy | None,
    selection_seed: int,
) -> SelfCondConfig:
    """The common construction, preserving every historical AxBench field."""
    fp_config = model.feature_pred_config
    if fp_config is None:
        raise ValueError("SelfConditioning requires checkpoint feature_prediction metadata")
    if FeaturePredLossMode(fp_config.loss_mode) is not FeaturePredLossMode.MAX_POOL:
        raise ValueError("SelfConditioning requires a MAX_POOL feature-prediction head")
    rng = t.Generator(device="cpu")
    rng.manual_seed(int(selection_seed))
    positive = selection_without_delta(positive_selection)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(segment_size),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(selection=positive, percentile_table=percentile_table, rng=rng),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(
            active_bucket_count(int(model.encoder.config.num_buckets), reserves_negative_bucket=True)
        ),
        n_sae_features=PositiveInt.parse(n_sae_features),
        segment_mode=segment_mode,
        feature_mask=feature_mask,
        cluster_consolidation=consolidation,
        score_policy=policy,
    )


def build_pure_model_self_conditioning_config(
    *,
    model: Any,
    checkpoint_meta: Mapping[str, Any],
    n_sae_features: int,
    feature_mask: FeatureMask | None,
    consolidation: FeatureClusterConsolidation | None,
    segment_mode: SegmentMode,
    segment_size: int,
    selection_seed: int,
    percentile_table: t.Tensor,
) -> SelfCondConfig:
    """Build the two-sided weight-zero endpoint, never the policy=None arm.

    Count distributions use the pretraining/SFT metadata fallback. The exact
    effective vocabulary must match the loaded model; the existing provider
    translates its selected compact rows to original IDs and assigns buckets
    zero/one. No reference answer or concept score is consumed here.
    """
    if model.feature_pred_head is None:
        raise ValueError("SelfConditioning requires a feature-prediction head")
    if model.encoder.config.num_buckets != 2:
        raise ValueError("SelfConditioning requires two encoder buckets")
    conditioning = read_checkpoint_conditioning(checkpoint_meta)
    raw_positive = conditioning.positive_selection
    if not isinstance(raw_positive, Mapping):
        raise ValueError("SelfConditioning requires checkpoint feature_selection mapping")
    positive = parse_feature_selection(dict(raw_positive))
    if not isinstance(positive, SamplingSelection):
        raise ValueError("SelfConditioning requires sampling-based output selection")
    negative = parse_negative_selection(conditioning.negative_selection)
    if negative is None:
        raise ValueError("SelfConditioning requires checkpoint negative_selection")
    universe = consolidation if consolidation is not None else feature_mask
    vocab = effective_feature_vocab(universe, n_sae_features)
    if model.feature_vocab != vocab:
        raise ValueError("SelfConditioning requires the exact checkpoint FeatureVocab")
    width = n_sae_features if vocab is None else int(vocab.size)
    return build_self_conditioning_config(
        model=model,
        positive_selection=positive,
        percentile_table=percentile_table,
        n_sae_features=n_sae_features,
        feature_mask=feature_mask,
        consolidation=consolidation,
        segment_mode=segment_mode,
        segment_size=segment_size,
        policy=SelfCondScoreInterpolation(
            weight=UnitInterval.parse(0),
            target_scores=t.zeros(width, dtype=t.float32),
            negative_selection=negative,
        ),
        selection_seed=selection_seed,
    )
