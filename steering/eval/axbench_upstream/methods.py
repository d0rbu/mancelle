"""Corrected dense-DiffMean SelfConditioning adapter for AxBench upstream."""

from __future__ import annotations

import gc
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import torch as t
from loguru import logger

from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    resolve_and_assert_model_feature_contract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.delta import DeltaMode, DeltaSamplingWeights, parse_feature_delta
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfCondScoreMasking,
    SelfCondScorePolicy,
    rank_masked_candidate_pools,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.encoder_decoder.sft_selection import (
    SegmentSelectionContext,
    select_conversation_segments,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import (
    MissingPercentileError,
    MissingPercentileResolver,
    check_percentile_coverage,
    enforce_percentile_coverage,
)
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    build_percentile_table,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.eval.axbench_base.concept_feature_mapping import _batched_dense_sae_max_pool
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from steering.eval.chat_generation import ChatGenerationResult, render_prompt_for_chat_completion
from steering.eval.context_cache_build import (
    build_context_entry_from_extracted_segments,
    build_context_entry_from_selected_segments,
)
from steering.eval.context_cache_io import ContextCacheEntry
from steering.eval.context_enumeration import context_text_for_extraction
from steering.eval.dense_feature_targets import (
    dense_diffmean_from_means,
    dense_mean_percentiles,
    load_dense_extraction_stack,
)
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.self_conditioning_policy import (
    build_self_conditioning_config as build_shared_self_conditioning_config,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SingleEntryContextProvider,
)
from steering.eval.turn_context_contract import (
    build_turn_selection_context as _build_turn_prompt_selection_context,
)
from steering.eval.turn_context_contract import (
    load_turn_statistics_snapshot as _load_turn_statistics_snapshot,
)
from steering.extract_features import (
    extract_document_features,
)
from steering.feature_paths import PoolingMode
from steering.refined_types import NonNegativeInt, PositiveInt, UnitInterval
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    coerce_config_l0,
    coerce_sae_width,
    resolve_sae,
)
from steering.segmentation import (
    SegmentMode,
    resolve_bos_token_id,
    truncate_to_training_window,
)
from steering.sft.segment_mode_identity import (
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
    segmentation_identity_drift,
)
from steering.sft.turn_spans import rendered_text_and_turn_char_starts
from steering.stats_io import (
    content_sha256,
    load_statistics,
    verify_statistics_sha256,
)
from steering.stats_types import ExtractionStatistics

TEMPERATURE = 1.0
TOP_K = 50
SELF_CONDITIONING_POLICY = "prompt_last_segment_max_pool_logit_rank_diffmean_lerp_v2"
SELF_CONDITIONING_MASKED_POLICY = "prompt_last_segment_max_pool_logit_rank_diffmean_sign_rank_v3"
SELF_CONDITIONING_FACTORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
PROMPT_SEGMENT_SIZE = 128
PROMPT_CONTEXT_WINDOW = 2048

# fit_concept_dense_diffmeans's gc.collect()+empty_cache() release interval
# (in concepts). Measured unfixed growth was ~0.25GiB/concept (~10GiB ->
# ~19.5GiB allocated over 39 concepts before OOM, c50 scale) -- releasing
# every concept eliminates that but costs ~32% more wall-clock for this
# stage (1.39s/concept unfixed pace -> 1.84s/concept, measured via direct
# A/B). Releasing only every N concepts still bounds the worst-case
# accumulation to N * 0.25GiB before each release (1.25GiB at N=5) against
# a multi-GiB safety margin to the GPU ceiling, while cutting the number of
# gc.collect()/empty_cache() calls (and their overhead) by N x.
_MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS = 5


@dataclass(frozen=True)
class PreparedPromptContext:
    """Exact rendered prompt key, stop token, and transient extracted context."""

    token_ids: tuple[int, ...]
    assistant_end_id: int
    segments: tuple[ContextFeatureSegment, ...]
    segment_mode: SegmentMode = SegmentMode.fixed_tokens
    token_assignments: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SelfConditioningGenerationRequest:
    """One independently configured and seeded project generation row."""

    policy: SelfCondScorePolicy
    selection_seed: int
    token_seed: int


@dataclass(frozen=True)
class MethodSetup:
    """Checkpoint-derived feature contracts shared by fitting and generation."""

    target_meta_path: Path
    meta: dict[str, Any]
    sae_model: str
    sae_layer: int
    resolved_sae: Any
    n_sae_features: int
    feature_contract: CheckpointFeatureContract
    prompt_feature_contract: CheckpointFeatureContract
    consolidation: FeatureClusterConsolidation | None
    prompt_consolidation: FeatureClusterConsolidation | None
    feature_vocab: FeatureVocab | None
    allowed_ids: t.Tensor | None
    output_selection: SamplingSelection
    output_negative_selection: NegativeSamplingSelection
    prompt_pipeline: FeaturePipelineConfig
    statistics: ExtractionStatistics
    percentile_table: t.Tensor
    segment_mode: SegmentMode
    prompt_segment_size: int
    prompt_context_window: int
    turn_self_cond_segment_size_rustbpe: int | None
    prompt_selection_context: SegmentSelectionContext | None
    prompt_statistics_raw_sha256: str | None = None

    @property
    def feature_mask(self):
        return self.feature_contract.feature_mask


def _load_pretrain_user_config(
    pretrain_meta_path: Path,
    *,
    target_user_config: dict[str, Any],
) -> dict[str, Any]:
    """Load and return ``pretrain_meta_path``'s ``user_config``, model_tag-checked.

    Shared by every SFT-phase fallback below: each reads a different field
    that the SFT trainer failed to (re-)record faithfully, but all of them
    must come from the *same* pretrain checkpoint the target actually
    started from, so the model_tag cross-check lives here once.
    """
    pretrain_meta_path = Path(pretrain_meta_path)
    pretrain_meta = json.loads(pretrain_meta_path.read_text(encoding="utf-8"))
    if not isinstance(pretrain_meta, dict):
        raise ValueError(f"pretrain metadata is not a JSON object: {pretrain_meta_path}")
    pretrain_user_config = pretrain_meta.get("user_config")
    if not isinstance(pretrain_user_config, dict):
        raise ValueError(f"pretrain metadata has no user_config mapping: {pretrain_meta_path}")
    target_model_tag = target_user_config.get("model_tag")
    pretrain_model_tag = pretrain_user_config.get("model_tag")
    if target_model_tag is None or target_model_tag != pretrain_model_tag:
        raise ValueError(
            "pretrain_meta_path model_tag does not match the target checkpoint's own "
            f"model_tag: target={target_model_tag!r}, pretrain={pretrain_model_tag!r}"
        )
    return pretrain_user_config


def _conditioning_block(
    meta: dict[str, Any],
    *,
    target_meta_path: Path,
    pretrain_meta_path: Path | None,
) -> dict[str, Any]:
    """Resolve the checkpoint's prompt-conditioning snapshot.

    Only the clustered pretrain-time training path writes
    ``user_config.conditioning`` (context_cache_dir, seed, feature_selection,
    negative_selection, feature_mask, feature_clusters, stats_source_sha256,
    identity): it is a frozen snapshot for reproducible eval-time prompt
    construction. An SFT-phase checkpoint fine-tunes weights only — its
    conditioning space is unchanged from the pretrain checkpoint it started
    from — but the SFT trainer never writes this block. When it's absent,
    the caller must supply ``pretrain_meta_path`` pointing at that exact
    pretrain checkpoint's own metadata; every field this function returns
    is subsequently cross-checked against the target checkpoint's own
    top-level user_config (feature_mask/feature_clusters in
    ``_prompt_feature_contract_metadata``, n_sae_features/num_buckets here in
    ``load_method_setup``), so a mismatched or unrelated pretrain checkpoint
    still fails loud downstream.
    """
    user_config = meta.get("user_config")
    conditioning = user_config.get("conditioning") if isinstance(user_config, dict) else None
    if isinstance(conditioning, dict):
        return conditioning
    if pretrain_meta_path is None:
        raise ValueError(
            f"target metadata has no user_config.conditioning mapping: {target_meta_path}; "
            "this looks like an SFT-phase checkpoint (fine-tunes weights only, does not "
            "carry its own frozen prompt-conditioning snapshot) — pass pretrain_meta_path "
            "pointing at the pretrain checkpoint metadata it started from"
        )
    pretrain_user_config = _load_pretrain_user_config(
        pretrain_meta_path, target_user_config=user_config or {}
    )
    pretrain_conditioning = pretrain_user_config.get("conditioning")
    if not isinstance(pretrain_conditioning, dict):
        raise ValueError(
            f"pretrain metadata has no user_config.conditioning mapping: {pretrain_meta_path}"
        )
    return pretrain_conditioning


def resolve_raw_sae_config(
    user_config: dict[str, Any],
    *,
    pretrain_meta_path: Path | None,
) -> Any:
    """Return the checkpoint's sae config, always from pretrain when SFT-phase.

    The SAE identity (which feature space the encoder conditions on) is
    fixed at pretrain time and cannot change during SFT — SFT only
    fine-tunes decoder/cross-attention weights. In practice the SFT trainer
    has been observed to record a stale/default ``user_config.sae`` instead
    of faithfully copying the pretrain checkpoint's own (e.g. recording
    gemma-2-2b/l8/16k for a checkpoint whose encoder actually conditions on
    gemma-3-4b-pt/l22/65k). ``pretrain_meta_path`` is an explicit signal
    this is an SFT-phase checkpoint, so when it's supplied, always prefer
    the pretrain checkpoint's own sae config over the target's.
    """
    if pretrain_meta_path is None:
        return user_config.get("sae")
    pretrain_user_config = _load_pretrain_user_config(
        pretrain_meta_path, target_user_config=user_config
    )
    return pretrain_user_config.get("sae")


def _normalize_feature_reference(value: Any) -> Any:
    """Normalize a feature_mask/feature_clusters reference for cross-checking.

    Compares by content-identifying fields (mode + path basename) rather than
    the full absolute path: the same artifact is legitimately staged under
    different absolute paths across environments (e.g. a local repo-relative
    pretrain path vs an Engaging scratch path for an SFT run), and these
    artifacts are named uniquely by content (schema-versioned/timestamped
    filenames), so basename equality is a reliable identity check without
    requiring both files to be locally co-accessible.
    """
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    path = normalized.get("path")
    if isinstance(path, str) and path:
        normalized["path"] = Path(path).name
    return normalized


def _prompt_feature_contract_metadata(
    meta: dict[str, Any],
    user_config: dict[str, Any],
    conditioning: dict[str, Any],
) -> dict[str, Any]:
    """Bind nested prompt markers to the checkpoint's canonical contract."""

    for field in ("feature_mask", "feature_clusters"):
        if _normalize_feature_reference(conditioning.get(field)) != _normalize_feature_reference(
            user_config.get(field)
        ):
            raise ValueError(
                f"checkpoint conditioning.{field} disagrees with the canonical "
                f"top-level {field}; the prompt feature universe is ambiguous"
            )
    return {
        "feature_mask_canonical": meta.get("feature_mask_canonical"),
        "feature_clusters_descriptor": meta.get("feature_clusters_descriptor"),
        "feature_clusters_canonical": meta.get("feature_clusters_canonical"),
        "user_config": conditioning,
    }


def _validate_statistics_identity(
    conditioning: dict[str, Any],
    path: Path,
    *,
    require_exact_statistics_identity: bool = True,
) -> str:
    """Confirm ``path`` is the statistics file the checkpoint was trained with.

    Strict validation accepts the canonical content hash or an exact legacy
    raw-file hash through the shared statistics verifier. The returned identity
    remains the canonical content hash. An explicit relaxed mode retains the
    existing operator-asserted compatibility path for historical re-saves.
    """
    expected = conditioning.get("stats_source_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(
            "target metadata requires a 64-character lowercase conditioning.stats_source_sha256"
        )
    actual = content_sha256(path)
    if actual != expected:
        message = (
            "feature statistics do not match the target checkpoint's recorded "
            f"conditioning.stats_source_sha256: expected={expected}, actual content_sha256="
            f"{actual} for {path}"
        )
        if not require_exact_statistics_identity:
            logger.warning(
                f"{message} — proceeding anyway because "
                "require_exact_statistics_identity=False was explicitly set (operator-asserted "
                "trust, not a verified match; likely because this checkpoint's recorded hash "
                "predates content_sha256)"
            )
        else:
            try:
                verify_statistics_sha256(path, expected, surface="target checkpoint")
            except ValueError as exc:
                raise ValueError(message) from exc
    return actual


def _validate_target_policy(
    raw_delta: Any,
    selection: SamplingSelection,
) -> None:
    if not isinstance(raw_delta, dict):
        raise ValueError("target metadata has no feature_delta mapping")
    feature_delta = parse_feature_delta(raw_delta)
    if feature_delta is None or feature_delta.mode is not DeltaMode.splice_counterfactual:
        raise ValueError("AxBench methods require feature_delta.mode=splice_counterfactual")
    if selection.delta_weights is not DeltaSamplingWeights.percentile_delta:
        raise ValueError(
            "AxBench methods require feature_selection.delta_weights=percentile_delta; "
            f"got {selection.delta_weights!r}"
        )


def _is_under_context_independent_no_bos(path: Path) -> bool:
    """Whether ``path`` descends from a .../context_independent/no_bos/ directory.

    The simple feature-directory layout puts the statistics file directly at
    .../context_independent/no_bos/feature_statistics.safetensors. The
    spliced-counterfactual layout nests further phase-refinement directories
    underneath (e.g. .../no_bos/splice/cut_frac0.15/seed42/), so checking only
    the immediate parent/grandparent misses it. Walk every adjacent ancestor
    pair instead of just the nearest one, so any depth of nesting below
    no_bos/ still counts as the same feature family.
    """
    ancestors = list(path.parents)
    return any(
        parent.name == "no_bos" and grandparent.name == "context_independent"
        for parent, grandparent in pairwise(ancestors)
    )


def _validate_statistics_contract(
    statistics: ExtractionStatistics,
    *,
    path: Path,
    sae_model: str,
    sae_layer: int,
    n_sae_features: int,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: int = PROMPT_SEGMENT_SIZE,
    context_window: int = PROMPT_CONTEXT_WINDOW,
) -> None:
    """Bind statistics metadata to the geometry the checkpoint declares.

    Turn mode reads the context window OFF these statistics (whose bytes the
    lineage digest pins) and proves their feature family through the lineage,
    so neither the window nor the canonical path is re-asserted there.
    """
    metadata = statistics.metadata
    expected: dict[str, Any] = {
        "pooling_mode": "max_pool",
        "segment_mode": segment_mode,
        "segment_size": segment_size,
        "layer": sae_layer,
        "n_sae_features": n_sae_features,
    }
    if segment_mode is SegmentMode.fixed_tokens:
        expected["context_window"] = context_window
    for field, value in expected.items():
        observed = getattr(metadata, field)
        if observed != value:
            raise ValueError(f"feature statistics {field}={observed!r}, expected {value!r}")
    if not statistics.ignore_bos_features:
        raise ValueError("prompt feature statistics must record ignore_bos_features=true")
    if segment_mode is SegmentMode.fixed_tokens and not _is_under_context_independent_no_bos(path):
        raise ValueError(
            "prompt feature statistics must come from a context_independent/no_bos feature family"
        )
    if sae_model != "gemma-3-4b-pt" or sae_layer != 22 or n_sae_features != 65_536:
        raise ValueError(
            "AxBench SelfConditioning requires Gemma-3-4B-PT residual SAE layer 22 width 65,536"
        )


def _validate_turn_lineage_contract(
    lineage: OnlineSAELineage,
    *,
    sae_release: SAERelease,
    sae_model: str,
    sae_layer: int,
    activation_type: ActivationType,
    sae_width: Any,
    sae_l0: Any,
    n_sae_features: int,
    num_buckets: int,
) -> None:
    """Bind turn prompt extraction to the exact conditioned-SFT source.

    ``lineage.source_segment_mode`` is deliberately absent from the comparison:
    ``OnlineSAELineage.from_checkpoint_meta`` derives it from the very same
    ``resolve_checkpoint_sft_segment_mode`` call that gates this branch, so a
    conjunct on it could never fail. A disagreeing witness is already rejected
    one layer down by that resolver's conflicting-witness branch.
    """

    expected = {
        "model": sae_model,
        "release": sae_release.value,
        "activation_type": activation_type.value,
        "layer": sae_layer,
        "width": sae_width.value,
        "l0": getattr(sae_l0, "value", sae_l0),
        "n_sae_features": n_sae_features,
        "num_buckets": num_buckets,
    }
    observed = {
        "model": lineage.model,
        "release": lineage.release,
        "activation_type": lineage.activation_type,
        "layer": lineage.layer,
        "width": lineage.width,
        "l0": lineage.l0,
        "n_sae_features": lineage.n_sae_features,
        "num_buckets": lineage.num_buckets,
    }
    if observed != expected:
        raise ValueError(
            "turn-delimited online-SAE lineage conflicts with the target checkpoint "
            f"feature contract: expected={expected!r}, observed={observed!r}. SFT-phase "
            "checkpoints record a default user_config.sae; pass pretrain_meta_path so the "
            "SAE identity comes from the pretrain checkpoint."
        )


def _resolve_checkpoint_phase_field(meta: dict, user_config: dict, field: str) -> Any:
    """Return ``user_config[field]``, falling back to ``meta.pretrain_conditioning[field]``.

    A pretrain checkpoint records fields like ``negative_selection`` and
    ``feature_delta`` at the top level of ``user_config``. An SFT-phase
    checkpoint inherits them from the pretrain checkpoint it started from
    without re-recording them there; the SFT trainer instead nests that
    inheritance under ``pretrain_conditioning`` (see
    ``scripts/train/conditional_encdec_sft.py``). Fall back to that nested
    location only when the top-level field is absent.
    """
    raw = user_config.get(field)
    if raw is not None:
        return raw
    pretrain_conditioning = meta.get("pretrain_conditioning")
    if isinstance(pretrain_conditioning, dict):
        return pretrain_conditioning.get(field)
    return None


def resolve_raw_negative_selection(meta: dict, user_config: dict) -> dict | None:
    """Return the checkpoint's negative_selection mapping, checkpoint-phase aware."""
    return _resolve_checkpoint_phase_field(meta, user_config, "negative_selection")


def resolve_raw_feature_delta(meta: dict, user_config: dict) -> dict | None:
    """Return the checkpoint's feature_delta mapping, checkpoint-phase aware."""
    return _resolve_checkpoint_phase_field(meta, user_config, "feature_delta")


def resolve_raw_feature_selection(meta: dict, user_config: dict) -> Any:
    """Return the checkpoint's feature_selection mapping with delta_weights resolved.

    An SFT-phase checkpoint inherits its output feature-selection SHAPE
    (distribution/mean/std/min_k/max_k/mode) from the pretrain checkpoint and
    records it verbatim at the top level, but ``delta_weights`` specifically
    records whether delta-based selection built the PRETRAIN checkpoint's
    training targets -- not applicable to SFT's own run, so the SFT trainer
    records it as null there while nesting the real (pretrain) value under
    ``pretrain_conditioning.delta_weights``. Patch ``delta_weights`` from that
    nested location only when the top-level value is missing.
    """
    raw_selection = user_config.get("feature_selection")
    if not isinstance(raw_selection, dict) or raw_selection.get("delta_weights") is not None:
        return raw_selection
    pretrain_conditioning = meta.get("pretrain_conditioning")
    if not isinstance(pretrain_conditioning, dict):
        return raw_selection
    delta_weights = pretrain_conditioning.get("delta_weights")
    if delta_weights is None:
        return raw_selection
    return {**raw_selection, "delta_weights": delta_weights}


def load_method_setup(
    target_meta_path: Path,
    feature_statistics_path: Path,
    *,
    pretrain_meta_path: Path | None = None,
    require_exact_statistics_identity: bool = True,
) -> MethodSetup:
    """Resolve exact checkpoint, statistics, prompt, and output feature policies."""

    target_meta_path = Path(target_meta_path)
    meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not isinstance(meta.get("user_config"), dict):
        raise ValueError(f"target metadata has no user_config mapping: {target_meta_path}")
    user_config = meta["user_config"]
    segment_mode = resolve_checkpoint_sft_segment_mode(
        meta,
        checkpoint_dir=str(target_meta_path.parent),
    )
    turn_self_cond_segment_size_rustbpe = resolve_turn_self_cond_segment_size_rustbpe(
        meta,
        checkpoint_dir=str(target_meta_path.parent),
    )
    turn_lineage = (
        OnlineSAELineage.from_checkpoint_meta(meta)
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens
        else None
    )
    # A raw conditioned-SFT export records extraction identity in its typed
    # lineage while user_config.sae can be an unrelated Hydra default. With no
    # pretrain witness the lineage is the sole authority; a supplied
    # pretrain_meta_path is read and cross-checked exactly as for every other
    # checkpoint (never accepted and ignored). Post-training exports always
    # keep the independent config/lineage cross-check.
    lineage_is_sole_witness = (
        turn_lineage is not None
        and "encoder_config" in meta
        and "selfcond" not in meta
        and pretrain_meta_path is None
    )
    sae_config: dict[str, Any]
    if lineage_is_sole_witness:
        assert turn_lineage is not None
        sae_config = {
            "model": turn_lineage.model,
            "release": turn_lineage.release,
            "activation_type": turn_lineage.activation_type,
            "layer": turn_lineage.layer,
            "width": turn_lineage.width,
            "l0": turn_lineage.l0,
        }
    else:
        sae_config = resolve_raw_sae_config(user_config, pretrain_meta_path=pretrain_meta_path)
    if not isinstance(sae_config, dict):
        raise ValueError(f"target metadata has no user_config.sae mapping: {target_meta_path}")
    try:
        sae_release = SAERelease(str(sae_config["release"]))
        sae_model = str(sae_config["model"])
        sae_layer = int(sae_config["layer"])
        activation_type = ActivationType(str(sae_config["activation_type"]))
        sae_width = coerce_sae_width(sae_config["width"])
        sae_l0 = coerce_config_l0(sae_release, sae_config.get("l0"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed user_config.sae in {target_meta_path}: {exc}") from exc
    if activation_type is not ActivationType.residual:
        raise ValueError("AxBench SelfConditioning requires a residual SAE")
    n_sae_features = SAE_WIDTH_TO_NUM_FEATURES[sae_width]
    resolved_sae = resolve_sae(
        model=sae_model,
        release=sae_release,
        activation_type=activation_type,
        layer=sae_layer,
        width=sae_width,
        l0=sae_l0,
    )
    if turn_lineage is not None:
        _validate_turn_lineage_contract(
            turn_lineage,
            sae_release=sae_release,
            sae_model=sae_model,
            sae_layer=sae_layer,
            activation_type=activation_type,
            sae_width=sae_width,
            sae_l0=sae_l0,
            n_sae_features=n_sae_features,
            num_buckets=2,
        )

    raw_selection = resolve_raw_feature_selection(meta, user_config)
    if not isinstance(raw_selection, dict):
        raise ValueError("target metadata has no feature_selection mapping")
    output_selection = parse_feature_selection(raw_selection)
    if not isinstance(output_selection, SamplingSelection):
        raise ValueError("AxBench SelfConditioning requires sampling-based output selection")
    _validate_target_policy(resolve_raw_feature_delta(meta, user_config), output_selection)
    output_negative_selection = parse_negative_selection(
        resolve_raw_negative_selection(meta, user_config)
    )
    if output_negative_selection is None:
        raise ValueError("AxBench SelfConditioning requires checkpoint negative_selection")
    if user_config.get("num_buckets") != 2:
        raise ValueError(
            "AxBench SelfConditioning requires two buckets (active bucket 0, inactive bucket 1)"
        )

    feature_contract = resolve_checkpoint_feature_contract(
        meta,
        n_sae_features=PositiveInt.parse(n_sae_features),
        surface="AxBench upstream checkpoint",
        source_path=str(target_meta_path),
    )
    feature_vocab = effective_feature_vocab(feature_contract.universe, n_sae_features)
    recorded_vocab = meta.get("feature_vocab")
    actual_vocab = (
        None
        if feature_vocab is None
        else {
            "n_sae_features": feature_vocab.n_sae_features,
            "vocab_size": feature_vocab.size,
            "sha256": feature_vocab.fingerprint(),
        }
    )
    if recorded_vocab != actual_vocab:
        raise ValueError(
            "target checkpoint feature vocabulary does not match its canonical "
            f"mask/cluster contract: recorded={recorded_vocab!r}, derived={actual_vocab!r}"
        )
    allowed_ids = None if feature_vocab is None else feature_vocab.new_to_old_tensor(device="cpu")

    if lineage_is_sole_witness and "conditioning" not in user_config:
        # Only fill an ABSENT snapshot from the lineage; a recorded one (even a
        # malformed one) keeps its existing validation and cross-checks.
        assert turn_lineage is not None
        conditioning = {
            "n_sae_features": turn_lineage.n_sae_features,
            "num_buckets": turn_lineage.num_buckets,
            "feature_mask": user_config.get("feature_mask"),
            "feature_clusters": user_config.get("feature_clusters"),
        }
    else:
        conditioning = _conditioning_block(
            meta,
            target_meta_path=target_meta_path,
            pretrain_meta_path=pretrain_meta_path,
        )
    prompt_feature_contract = resolve_checkpoint_feature_contract(
        _prompt_feature_contract_metadata(meta, user_config, conditioning),
        n_sae_features=PositiveInt.parse(n_sae_features),
        surface="AxBench upstream prompt conditioning",
        source_path=str(target_meta_path),
    )
    prompt_feature_vocab = effective_feature_vocab(
        prompt_feature_contract.universe,
        n_sae_features,
    )
    if prompt_feature_vocab != feature_vocab:
        raise ValueError(
            "checkpoint prompt conditioning feature universe does not equal the "
            "prediction-head/encoder feature vocabulary"
        )

    feature_statistics_path = Path(feature_statistics_path)
    if turn_lineage is None:
        _validate_statistics_identity(
            conditioning,
            feature_statistics_path,
            require_exact_statistics_identity=require_exact_statistics_identity,
        )
        statistics = load_statistics(feature_statistics_path)
    else:
        if not require_exact_statistics_identity:
            raise ValueError(
                "turn-delimited checkpoints have no relaxed statistics identity mode: the "
                "online-SAE lineage pins the statistics file's RAW sha256, so "
                "require_exact_statistics_identity=False would be silently ignored"
            )
        statistics = _load_turn_statistics_snapshot(
            feature_statistics_path,
            expected_raw_sha256=turn_lineage.statistics_sha256,
        )
    if turn_lineage is None:
        prompt_segment_size = PROMPT_SEGMENT_SIZE
        prompt_context_window = PROMPT_CONTEXT_WINDOW
    else:
        prompt_segment_size = turn_lineage.source_segment_size
        prompt_context_window = statistics.metadata.context_window
        if (
            isinstance(prompt_context_window, bool)
            or not isinstance(prompt_context_window, int)
            or prompt_context_window <= 0
        ):
            raise ValueError(
                "turn-delimited feature statistics context_window must be a positive integer"
            )
    _validate_statistics_contract(
        statistics,
        path=feature_statistics_path,
        sae_model=sae_model,
        sae_layer=sae_layer,
        n_sae_features=n_sae_features,
        segment_mode=segment_mode,
        segment_size=prompt_segment_size,
        context_window=prompt_context_window,
    )
    percentile_table = build_percentile_table(
        statistics.per_feature_stats,
        n_sae_features,
    )
    if conditioning.get("n_sae_features") != n_sae_features:
        raise ValueError("checkpoint prompt conditioning SAE width disagrees with the SAE")
    if conditioning.get("num_buckets") != 2:
        raise ValueError("checkpoint prompt conditioning must use two buckets")
    conditioning_seed = (
        conditioning.get("seed") if turn_lineage is None else turn_lineage.selection_seed
    )
    if not isinstance(conditioning_seed, int) or isinstance(conditioning_seed, bool):
        raise ValueError("checkpoint prompt conditioning must record an integer seed")
    prompt_pipeline_source: Any = conditioning
    if turn_lineage is not None:
        prompt_pipeline_source = {
            "feature_selection": dict(turn_lineage.feature_selection),
            "negative_selection": (
                None
                if turn_lineage.negative_selection is None
                else dict(turn_lineage.negative_selection)
            ),
        }
    prompt_pipeline = build_feature_pipeline_from_cfg_block(
        prompt_pipeline_source,
        per_feature_stats=statistics.per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=2,
        seed=conditioning_seed,
        feature_mask=prompt_feature_contract.feature_mask,
    )
    if not isinstance(prompt_pipeline.selection_config, SamplingBundle):
        raise ValueError("checkpoint prompt conditioning must use sampling selection")
    if prompt_pipeline.negative_bundle is None:
        raise ValueError("checkpoint prompt conditioning must sample inactive features")
    if prompt_pipeline.effective_buckets != 1:
        raise ValueError("checkpoint prompt positives must encode as bucket 0 only")
    prompt_selection_context = (
        None
        if turn_lineage is None
        else _build_turn_prompt_selection_context(
            pipeline=prompt_pipeline,
            missing_percentile_policy=turn_lineage.missing_percentile_policy,
            consolidation=prompt_feature_contract.consolidation,
        )
    )

    return MethodSetup(
        target_meta_path=target_meta_path,
        meta=meta,
        sae_model=sae_model,
        sae_layer=sae_layer,
        resolved_sae=resolved_sae,
        n_sae_features=n_sae_features,
        feature_contract=feature_contract,
        prompt_feature_contract=prompt_feature_contract,
        consolidation=feature_contract.consolidation,
        prompt_consolidation=prompt_feature_contract.consolidation,
        feature_vocab=feature_vocab,
        allowed_ids=allowed_ids,
        output_selection=output_selection,
        output_negative_selection=output_negative_selection,
        prompt_pipeline=prompt_pipeline,
        statistics=statistics,
        percentile_table=percentile_table,
        segment_mode=segment_mode,
        prompt_segment_size=prompt_segment_size,
        prompt_context_window=prompt_context_window,
        turn_self_cond_segment_size_rustbpe=turn_self_cond_segment_size_rustbpe,
        prompt_selection_context=prompt_selection_context,
        prompt_statistics_raw_sha256=(
            None if turn_lineage is None else turn_lineage.statistics_sha256
        ),
    )


def load_extraction_stack(setup: MethodSetup, device: str) -> tuple[Any, Any]:
    """Keep existing callers on the shared dense extraction loader."""
    return load_dense_extraction_stack(sae_model=setup.sae_model, sae_layer=setup.sae_layer,
        resolved_sae=setup.resolved_sae, device=device)


def build_extracted_context_entry(
    setup: MethodSetup,
    extraction_model: Any,
    sae: Any,
    nanochat_tokenizer: Any,
    *,
    token_ids: Sequence[int],
    source: str,
) -> ContextCacheEntry:
    """Extract, select, and cross-tokenizer-align one exact rendered row."""

    exact_tokens = tuple(int(token_id) for token_id in token_ids)
    if not exact_tokens:
        raise ValueError(f"{source}: token tuple must be nonempty")
    segment_mode = setup.segment_mode
    segment_size = int(setup.prompt_segment_size)
    context_window = int(setup.prompt_context_window)
    model_context_window = int(extraction_model.cfg.n_ctx)
    if context_window > model_context_window:
        raise ValueError(
            f"{source}: extraction context window {context_window} exceeds "
            f"the extraction model context length {model_context_window}"
        )
    turn_starts: tuple[int, ...] | None = None
    if segment_mode is SegmentMode.fixed_tokens:
        context_text = context_text_for_extraction(nanochat_tokenizer, exact_tokens)
    else:
        context_text, turn_starts = rendered_text_and_turn_char_starts(
            exact_tokens,
            nanochat_tokenizer,
        )
    _kept, full_gemma_tokens = truncate_to_training_window(
        context_text,
        extraction_model.tokenizer,
        context_window,
        extraction_model.cfg.n_ctx,
    )
    if full_gemma_tokens > context_window:
        raise ValueError(
            f"{source} has {full_gemma_tokens} Gemma tokens, exceeding "
            f"the extraction window {context_window}"
        )
    extracted = extract_document_features(
        model=extraction_model,
        sae=sae,
        text=context_text,
        layer=setup.sae_layer,
        context_window=context_window,
        pooling=PoolingMode.max_pool,
        segment_mode=segment_mode,
        segment_size=segment_size,
        hook_name=setup.resolved_sae.hook_name,
        context_independent_features=True,
        ignore_bos_features=True,
        turn_char_starts=turn_starts,
    )
    feature_ids, magnitudes, positions, char_starts, char_ends, n_tokens = extracted
    if n_tokens != full_gemma_tokens:
        raise RuntimeError(
            f"{source}: extraction token count changed between preflight and extraction: "
            f"{full_gemma_tokens} != {n_tokens}"
        )
    bos_id = resolve_bos_token_id(extraction_model.tokenizer)
    if bos_id is None:
        raise ValueError(f"{source}: Gemma tokenizer has no BOS token")
    source_bos_char_len = len(extraction_model.tokenizer.decode([bos_id]))
    if source_bos_char_len <= 0:
        raise ValueError(f"{source}: Gemma tokenizer BOS decodes to empty text")

    prompt_selection_context = setup.prompt_selection_context
    if segment_mode is SegmentMode.fixed_tokens:
        entry = build_context_entry_from_extracted_segments(
            token_ids=exact_tokens,
            segment_feature_ids=feature_ids,
            segment_feature_magnitudes=magnitudes,
            segment_char_starts=char_starts,
            segment_char_ends=char_ends,
            pipeline=setup.prompt_pipeline,
            n_sae_features=PositiveInt.parse(setup.n_sae_features),
            num_buckets=PositiveInt.parse(2),
            tokenizer=nanochat_tokenizer,
            source_bos_char_len=NonNegativeInt.parse(source_bos_char_len),
            feature_clusters=setup.prompt_consolidation,
            source=source,
        )
    else:
        if prompt_selection_context is None:
            raise ValueError(f"{source}: turn-delimited prompt selection context is missing")
        raw_segments = [
            SegmentFeatures(
                segment_idx=segment_index,
                position=int(position),
                feature_ids=ids.to(t.int32),
                feature_mags=mags.to(t.float32),
                char_start=int(char_start),
                char_end=int(char_end),
            )
            for segment_index, (ids, mags, position, char_start, char_end) in enumerate(
                zip(
                    feature_ids,
                    magnitudes,
                    positions,
                    char_starts,
                    char_ends,
                    strict=True,
                )
            )
        ]
        selected_segments = select_conversation_segments(
            raw_segments,
            ctx=prompt_selection_context,
            example_idx=0,
        )
        entry = build_context_entry_from_selected_segments(
            token_ids=exact_tokens,
            segment_feature_ids=[segment.feature_ids for segment in selected_segments],
            segment_buckets=[segment.buckets for segment in selected_segments],
            segment_char_starts=[segment.char_start for segment in selected_segments],
            segment_char_ends=[segment.char_end for segment in selected_segments],
            tokenizer=nanochat_tokenizer,
            source_bos_char_len=NonNegativeInt.parse(source_bos_char_len),
            source=source,
        )
    return entry


def prepare_prompt_context(
    setup: MethodSetup,
    base_model: Any,
    sae: Any,
    nanochat_tokenizer: Any,
    *,
    token_ids: Sequence[int],
    assistant_end_id: int,
) -> PreparedPromptContext:
    """Extract and align one exact rendered prompt entirely in memory."""

    entry = build_extracted_context_entry(
        setup,
        base_model,
        sae,
        nanochat_tokenizer,
        token_ids=token_ids,
        source="AxBench in-memory prompt",
    )
    if not entry.segments:
        raise ValueError("prepared prompt has no valid real feature segments")
    return PreparedPromptContext(
        token_ids=entry.token_ids,
        assistant_end_id=int(assistant_end_id),
        segments=entry.segments,
        segment_mode=entry.segment_mode,
        token_assignments=entry.token_assignments,
    )


def _diffmean_missing_percentile(
    setup: MethodSetup,
    required_ids: t.Tensor,
) -> MissingPercentileResolver:
    """Resolve uncovered DiffMean ids exactly as the turn-mode SFT selector does.

    In TURN mode the percentile table is the checkpoint's own SFT-domain
    statistics, digest-bound to the online-SAE lineage, while the DiffMean
    contrast texts come from the AxBench concept corpus: a mask-allowed feature
    the SFT domain never activated has an all-zero percentile row, which
    ``magnitude_to_percentile`` refuses to query. The lineage records what to do
    about that — ``missing_percentile_policy`` — and the selector that conditions
    the prompts already honours it through
    ``prompt_selection_context.missing_percentile``, so the fit reads the SAME
    resolver object rather than re-deriving a policy string. The coverage report
    is raised (``error``) or logged (``lowest``) up front so the operator sees
    the complete uncovered population instead of whichever id was looked up
    first.

    In FIXED mode there is no lineage and therefore no recorded policy: the
    ``error`` resolver delegates verbatim to ``magnitude_to_percentile``, and no
    coverage check runs, so every emitted value stays byte-identical.
    """

    context = setup.prompt_selection_context
    if context is None:
        return MissingPercentileResolver.error()
    policy = context.missing_percentile.policy
    report = check_percentile_coverage(
        percentile_table=setup.percentile_table,
        required_ids=required_ids,
        table_name="turn-delimited prompt statistics percentile table",
        required_set_name="dense DiffMean mask-allowed feature ids",
    )
    try:
        enforce_percentile_coverage([report], policy)
    except MissingPercentileError as error:
        raise MissingPercentileError(
            "AxBench turn-delimited dense DiffMean cannot rank every mask-allowed "
            f"feature under missing_percentile_policy={policy.value!r}: {error}"
        ) from error
    return context.missing_percentile


def fit_dense_diffmean(
    setup: MethodSetup,
    base_model: Any,
    sae: Any,
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    *,
    concept_id: int,
    extraction_batch_size: int = 8,
    pool_texts: Callable[[Sequence[str]], t.Tensor] | None = None,
) -> t.Tensor:
    """Fit the complete percentile-rank contrast in exact FP-head row order."""

    if not positive_texts or not negative_texts:
        raise ValueError("dense DiffMean requires nonempty positive and negative classes")
    if extraction_batch_size <= 0:
        raise ValueError("dense DiffMean extraction_batch_size must be positive")
    def pool(texts: Sequence[str]) -> t.Tensor:
        if pool_texts is not None:
            return pool_texts(texts)
        return _batched_dense_sae_max_pool(
            list(texts), base_model, sae, sae_layer=setup.sae_layer,
            tokenizer_id=f"google/{setup.sae_model}", batch_size=extraction_batch_size,
        )

    positive, negative = (pool(texts) for texts in (positive_texts, negative_texts))
    if setup.percentile_table.shape[0] != setup.n_sae_features:
        raise ValueError("dense DiffMean statistics width disagrees with setup")
    consolidation = setup.consolidation
    allowed_ids = (t.nonzero(consolidation.allowed_universe, as_tuple=False).squeeze(1)
                   if consolidation is not None else t.arange(setup.n_sae_features, dtype=t.int64)
                   if setup.allowed_ids is None else setup.allowed_ids)
    missing = _diffmean_missing_percentile(setup, allowed_ids)
    positive_mean, negative_mean = (
        dense_mean_percentiles(activations, percentile_table=setup.percentile_table,
            allowed_ids=allowed_ids, consolidation=consolidation, missing=missing,
            label=f"dense DiffMean {polarity}")
        for polarity, activations in (("positive", positive), ("negative", negative))
    )
    return dense_diffmean_from_means(positive_mean, negative_mean,
        feature_vocab=setup.feature_vocab, label=f"dense DiffMean concept_id={concept_id}")


def load_target_model(setup: MethodSetup, device: str) -> tuple[Any, Any]:
    """Load and bind the target checkpoint to the setup's exact row vocabulary."""

    try:
        step = int(setup.meta["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target metadata has no integer step: {setup.target_meta_path}") from exc
    checkpoint_dir = str(setup.target_meta_path.parent)
    model, tokenizer, loaded_meta = build_encdec_model(
        checkpoint_dir,
        step,
        device,
        ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_upstream_methods"),
    )
    drifted = segmentation_identity_drift(loaded_meta, setup.meta)
    if drifted:
        raise ValueError(
            "loaded target model turn-segmentation metadata differs from the "
            f"prevalidated AxBench setup: {drifted!r}"
        )
    contract = resolve_and_assert_model_feature_contract(
        model,
        loaded_meta,
        checkpoint_dir=checkpoint_dir,
        surface="AxBench upstream",
        missing_metadata_label="canonical feature_mask/feature_clusters metadata",
        missing_metadata_guidance="Use the checkpoint's matching metadata generation.",
    )
    loaded_clusters = contract.resolved_clusters
    setup_clusters = setup.feature_contract.resolved_clusters
    clusters_match = (loaded_clusters is None and setup_clusters is None) or (
        loaded_clusters is not None
        and setup_clusters is not None
        and loaded_clusters.definition == setup_clusters.definition
        and loaded_clusters.descriptor == setup_clusters.descriptor
    )
    if (
        contract.feature_mask != setup.feature_contract.feature_mask
        or not clusters_match
        or model.feature_vocab != setup.feature_vocab
    ):
        raise ValueError("loaded target model feature contract or vocabulary differs from setup")
    return model, tokenizer


def _validate_self_conditioning_factor(factor: float, *, owner: str) -> float:
    """Validate one declared-grid factor for either strength mapper."""

    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError(
            f"{owner} strength must be a real number from the declared "
            f"0..6 factor grid; got {factor!r}"
        )
    value = float(factor)
    if not math.isfinite(value) or value not in SELF_CONDITIONING_FACTORS:
        raise ValueError(
            f"{owner} strength must be a finite member of the declared "
            f"0..6 factor grid; got {factor!r}"
        )
    return value


def self_conditioning_weight(factor: float) -> UnitInterval:
    value = _validate_self_conditioning_factor(factor, owner="SelfConditioning")
    return UnitInterval.parse(value / 6.0)


def self_conditioning_masked_pool_limit(
    strength: float,
    *,
    n_candidates: int,
    mean: float,
) -> int:
    """Map a continuous strength to a deterministic ranked-pool cap."""

    maximum_strength = max(SELF_CONDITIONING_FACTORS)
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise TypeError(f"strength must be a finite real number, got {strength!r}")
    strength_value = float(strength)
    if not math.isfinite(strength_value) or not 0.0 <= strength_value <= maximum_strength:
        raise ValueError(
            f"strength must be finite and lie in [0, {maximum_strength}], got {strength!r}"
        )
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, Integral):
        raise TypeError(f"n_candidates must be an integer, got {n_candidates!r}")
    candidate_count = int(n_candidates)
    if candidate_count <= 0:
        raise ValueError(f"n_candidates must be greater than zero, got {n_candidates!r}")
    if isinstance(mean, bool) or not isinstance(mean, Real):
        raise TypeError(f"mean must be a finite real number, got {mean!r}")
    mean_value = float(mean)
    if not math.isfinite(mean_value) or mean_value <= 0.0:
        raise ValueError(f"mean must be finite and strictly positive, got {mean!r}")

    tau = strength_value / maximum_strength
    real_limit = candidate_count * (1.0 - tau) + 2.0 * mean_value * tau
    rounded_limit = round(real_limit)
    return min(candidate_count, max(1, rounded_limit))


def _masked_head_rows(
    setup: MethodSetup,
    diffmean_scores: t.Tensor,
) -> tuple[t.Tensor, t.Tensor | None, int]:
    """Resolve aligned original IDs, optional keep rows, and eligible width N."""

    row_original_ids = (
        t.arange(int(setup.n_sae_features), dtype=t.int64)
        if setup.allowed_ids is None
        else setup.allowed_ids.to(dtype=t.int64, device="cpu")
    )
    target_width = int(diffmean_scores.numel())
    if diffmean_scores.ndim != 1 or int(row_original_ids.numel()) != target_width:
        raise ValueError(
            "masked candidate ids and dense diff-mean widths disagree: "
            f"{int(row_original_ids.numel())} != {target_width}"
        )
    keep_rows = (
        None
        if setup.feature_mask is None
        else setup.feature_mask.keep_mask(row_original_ids).to(dtype=t.bool, device="cpu")
    )
    n_candidates = target_width if keep_rows is None else int(keep_rows.sum().item())
    if n_candidates <= 0:
        raise ValueError("SelfConditioningMasked has no eligible head rows after the feature mask")
    return row_original_ids, keep_rows, n_candidates


def assert_masked_diffmean_feasible(
    setup: MethodSetup,
    diffmean_scores: t.Tensor,
    *,
    concept_id: int,
) -> None:
    """Fail a sign-infeasible SelfConditioningMasked sweep before model load."""

    _, keep_rows, n_candidates = _masked_head_rows(setup, diffmean_scores)
    rank_masked_candidate_pools(
        diffmean_scores.detach().to(dtype=t.float32, device="cpu"),
        positive_pool_limit=n_candidates,
        negative_pool_limit=n_candidates,
        keep_rows=keep_rows,
        positive_min_k=int(setup.output_selection.min_k),
        negative_min_k=int(setup.output_negative_selection.min_k),
        owner=f"SelfConditioningMasked preflight (concept_id={concept_id})",
    )


def build_self_conditioning_masked_policy(
    setup: MethodSetup,
    diffmean_scores: t.Tensor,
    factor: float,
) -> SelfCondScoreMasking:
    """Build the shared AxBench rank-based masked policy for one factor."""

    strength = _validate_self_conditioning_factor(factor, owner="SelfConditioningMasked")
    _, _, n_candidates = _masked_head_rows(setup, diffmean_scores)
    positive_limit = self_conditioning_masked_pool_limit(
        strength,
        n_candidates=n_candidates,
        mean=float(setup.output_selection.mean),
    )
    negative_limit = self_conditioning_masked_pool_limit(
        strength,
        n_candidates=n_candidates,
        mean=float(setup.output_negative_selection.mean),
    )
    return SelfCondScoreMasking(
        target_scores=diffmean_scores,
        negative_selection=setup.output_negative_selection,
        positive_pool_limit=PositiveInt.parse(positive_limit),
        negative_pool_limit=PositiveInt.parse(negative_limit),
    )


def build_self_conditioning_config(
    setup: MethodSetup,
    model: Any,
    policy: SelfCondScorePolicy | None,
    *,
    selection_seed: int,
) -> SelfCondConfig:
    """Build the factor-invariant SelfConditioning config for one score policy.

    Every call seeds a fresh CPU generator from ``selection_seed`` so a
    per-generation config never inherits advanced RNG state from an earlier
    cell.
    """

    segment_mode = setup.segment_mode
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        # ``resolve_turn_self_cond_segment_size_rustbpe`` already rejected every
        # non-positive-integer cadence when the setup was built, and
        # ``PositiveInt.parse`` below re-pins positivity; only the ``None`` arm
        # of the declared ``int | None`` needs narrowing here.
        cadence = setup.turn_self_cond_segment_size_rustbpe
        if cadence is None:
            raise ValueError(
                "turn-delimited SelfConditioning requires the RustBPE "
                "cadence resolved from checkpoint post-training metadata"
            )
        self_cond_segment_size = cadence
    else:
        self_cond_segment_size = int(model.dec_config.sequence_len)
    return build_shared_self_conditioning_config(
        model=model,
        positive_selection=setup.output_selection,
        percentile_table=setup.percentile_table,
        n_sae_features=setup.n_sae_features,
        feature_mask=setup.feature_mask,
        consolidation=setup.consolidation,
        segment_mode=segment_mode,
        segment_size=self_cond_segment_size,
        policy=policy,
        selection_seed=selection_seed,
    )


def assert_prepared_prompt_matches(
    tokenizer: Any,
    prompt: str,
    prepared: PreparedPromptContext,
) -> None:
    if not prepared.token_ids:
        raise ValueError("prepared prompt token tuple must be nonempty")
    rerendered = render_prompt_for_chat_completion(tokenizer, prompt)
    if tuple(rerendered.token_ids) != prepared.token_ids:
        raise ValueError("target-tokenizer rerender changed the prepared prompt token tuple")
    if rerendered.assistant_end_id != prepared.assistant_end_id:
        raise ValueError("target-tokenizer rerender changed the assistant-end stop token")
    if prepared.token_ids[0] != int(tokenizer.get_bos_token_id()):
        raise ValueError("prepared prompt does not begin with the target nanochat BOS")


def _validate_self_conditioning_generation_inputs(
    model: Any,
    prepared: PreparedPromptContext,
    *,
    max_new_tokens: int,
) -> None:
    """Validate the shared self-conditioning generation contract."""

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if len(prepared.token_ids) + max_new_tokens > int(model.dec_config.sequence_len):
        raise ValueError("prepared prompt plus generation budget exceeds decoder sequence length")
    if not prepared.token_ids:
        raise ValueError("prepared prompt token tuple must be nonempty")
    if not prepared.segments:
        raise ValueError("prepared prompt must contain at least one real feature segment")


def _generate_self_conditioning(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    policy: SelfCondScorePolicy | None,
    *,
    selection_seed: int,
    seed: int,
    max_new_tokens: int,
    profile_driver: Literal["project", "encdec"],
    reuse_single_token_mask: bool = False,
) -> str:
    """Generate one scalar project or encoder-decoder row."""

    return generate_self_conditioning_result(
        setup, model, tokenizer, prepared, policy,
        selection_seed=selection_seed, seed=seed, max_new_tokens=max_new_tokens,
        profile_driver=profile_driver, reuse_single_token_mask=reuse_single_token_mask,
    ).text


def generate_self_conditioning_result(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    policy: SelfCondScorePolicy | None,
    *,
    selection_seed: int,
    seed: int,
    max_new_tokens: int,
    profile_driver: Literal["project", "encdec"] = "project",
    reuse_single_token_mask: bool = False,
) -> ChatGenerationResult:
    """Use the production scalar engine and retain generated-token evidence."""

    _validate_self_conditioning_generation_inputs(
        model,
        prepared,
        max_new_tokens=max_new_tokens,
    )
    with profile_region(
        "axbench_upstream.generation.engine_construction",
        driver=profile_driver,
        prompt_tokens=len(prepared.token_ids),
    ):
        config = build_self_conditioning_config(
            setup,
            model,
            policy,
            selection_seed=selection_seed,
        )
        provider = SingleEntryContextProvider(
            prepared.token_ids,
            prepared.segments,
            segment_mode=prepared.segment_mode,
            token_assignments=prepared.token_assignments,
        )
        engine = build_engine_self_conditioned(
            model,
            tokenizer,
            next(model.parameters()).device,
            config,
            context_provider=provider,
        )
    mask_context = model.reuse_single_token_packed_masks() if reuse_single_token_mask else nullcontext()
    with mask_context:
        with profile_region(
            "axbench_upstream.generation.stream",
            driver=profile_driver,
            prompt_tokens=len(prepared.token_ids),
            max_new_tokens=max_new_tokens,
        ):
            generated = list(
                engine.generate_stream(
                    list(prepared.token_ids),
                    max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    seed=seed,
                    stop_token_ids={
                        int(tokenizer.get_bos_token_id()),
                        prepared.assistant_end_id,
                    },
                )
            )
    with profile_region(
        "axbench_upstream.generation.tokenizer_decode",
        driver=profile_driver,
        generated_tokens=len(generated),
    ):
        return ChatGenerationResult(
            tokenizer.decode(generated), tuple(generated), len(generated) == max_new_tokens,
        )


def generate_self_conditioning(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    policy: SelfCondScorePolicy,
    *,
    selection_seed: int,
    seed: int,
    max_new_tokens: int,
    reuse_single_token_mask: bool = False,
) -> str:
    """Generate once from the exact prepared tokens under one score policy.

    The config (and its selection RNG) is built inside the call so every
    generation cell starts from a freshly seeded generator.
    """

    return _generate_self_conditioning(
        setup,
        model,
        tokenizer,
        prepared,
        policy,
        selection_seed=selection_seed,
        seed=seed,
        max_new_tokens=max_new_tokens,
        profile_driver="project",
        reuse_single_token_mask=reuse_single_token_mask,
    )


def _should_release_masked_memory(concept_index: int, total_concepts: int) -> bool:
    """Decide whether ``fit_concept_dense_diffmeans``'s per-concept memory
    release should fire at ``concept_index`` (0-based) out of
    ``total_concepts``. Pure arithmetic -- deliberately CUDA-free so the
    release cadence itself is unit-testable without a GPU. Fires every
    ``_MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS`` concepts and always on the
    last concept, so the worst-case unreleased accumulation is bounded and
    a run's final memory state is always clean regardless of how
    ``total_concepts`` divides by the interval. See the constant's
    docstring for the release-cost/leak-bound tradeoff.
    """
    is_last_concept = concept_index + 1 == total_concepts
    return (
        concept_index % _MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS
        == _MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS - 1
        or is_last_concept
    )


def generate_self_conditioning_batch(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    requests: Sequence[SelfConditioningGenerationRequest],
    *,
    max_new_tokens: int,
) -> tuple[str, ...]:
    """Batch fixed-token project cells sharing one exact prepared prompt.

    Every row receives a fresh config, selection generator, wrapper, scalar
    prefill, and runtime snapshot. The shared decoder is entered only after
    those policy-specific S0 states have been frozen, and only for a budget
    that the inference engine proves cannot consume the next conditioning
    boundary.

    Turn-delimited callers must use :func:`generate_self_conditioning` once
    per row so every generated conditioning refresh and structural stop is
    handled by the scalar runtime.
    """

    _validate_self_conditioning_generation_inputs(
        model,
        prepared,
        max_new_tokens=max_new_tokens,
    )
    rows = tuple(requests)
    if len(rows) < 2:
        raise ValueError(
            "generate_self_conditioning_batch requires at least two rows; "
            "use generate_self_conditioning for a scalar cell"
        )
    if setup.segment_mode is not SegmentMode.fixed_tokens:
        raise ValueError(
            "generate_self_conditioning_batch supports only fixed_tokens; "
            "turn-delimited generation requires the scalar cell dispatcher"
        )

    prefills = []
    engine = None
    with profile_region(
        "axbench_upstream.generation.batch_prefills",
        driver="project",
        physical_batch_size=len(rows),
        prompt_tokens=len(prepared.token_ids),
    ):
        for request in rows:
            config = build_self_conditioning_config(
                setup,
                model,
                request.policy,
                selection_seed=request.selection_seed,
            )
            provider = SingleEntryContextProvider(
                prepared.token_ids,
                prepared.segments,
                segment_mode=prepared.segment_mode,
                token_assignments=prepared.token_assignments,
            )
            engine = build_engine_self_conditioned(
                model,
                tokenizer,
                next(model.parameters()).device,
                config,
                context_provider=provider,
            )
            prefills.append(
                engine.prepare_self_conditioned_prefill(
                    list(prepared.token_ids),
                    max_new_tokens=max_new_tokens,
                )
            )
    assert engine is not None

    with profile_region(
        "axbench_upstream.generation.batch_stream",
        driver="project",
        physical_batch_size=len(rows),
        prompt_tokens=len(prepared.token_ids),
        max_new_tokens=max_new_tokens,
    ):
        generated = engine.generate_self_conditioned_batch(
            prefills,
            seeds=[request.token_seed for request in rows],
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            stop_token_ids={
                int(tokenizer.get_bos_token_id()),
                prepared.assistant_end_id,
            },
        )

    with profile_region(
        "axbench_upstream.generation.tokenizer_decode_batch",
        driver="project",
        physical_batch_size=len(rows),
    ):
        return tuple(tokenizer.decode(tokens) for tokens in generated)


def fit_concept_dense_diffmeans(
    setup: MethodSetup,
    base_model: Any,
    sae: Any,
    training: dict[int, ConceptTrainingTexts],
    *,
    extraction_batch_size: int,
    reuse_duplicate_texts: bool = False,
) -> dict[int, t.Tensor]:
    if type(reuse_duplicate_texts) is not bool:
        raise ValueError("reuse_duplicate_texts must be bool")
    if reuse_duplicate_texts and extraction_batch_size != 1:
        raise ValueError("reuse_duplicate_texts requires extraction_batch_size=1")
    pool_texts: Callable[[Sequence[str]], t.Tensor] | None = None
    if reuse_duplicate_texts:
        # A batch-one extraction is independent of neighboring texts. Retain
        # only rows that recur, avoiding a full-corpus activation cache on GPU.
        counts = Counter(
            text for examples in training.values()
            for text in (*examples.positive, *examples.negative)
        )
        repeated = {text for text, count in counts.items() if count > 1}
        cached: dict[str, t.Tensor] = {}

        def pool_texts(texts: Sequence[str]) -> t.Tensor:
            missing = list(dict.fromkeys(text for text in texts if text not in cached))
            fresh = _batched_dense_sae_max_pool(
                missing, base_model, sae, sae_layer=setup.sae_layer,
                tokenizer_id=f"google/{setup.sae_model}", batch_size=1,
            ) if missing else None
            rows = {text: fresh[index] for index, text in enumerate(missing)} if fresh is not None else {}
            for text in missing:
                if text in repeated:
                    cached[text] = rows[text].detach().clone()
            if len(missing) == len(texts):
                assert fresh is not None
                return fresh
            return t.stack([cached[text] if text in cached else rows[text] for text in texts])

    results: dict[int, t.Tensor] = {}
    for concept_index, (concept_id, examples) in enumerate(training.items()):
        results[concept_id] = fit_dense_diffmean(
            setup,
            base_model,
            sae,
            examples.positive,
            examples.negative,
            concept_id=concept_id,
            extraction_batch_size=extraction_batch_size,
            **({"pool_texts": pool_texts} if pool_texts is not None else {}),
        )
        if t.cuda.is_available():
            # Per-concept forwards go through nnterp's trace()/save()/stop(),
            # whose intervention graph nodes hold parent<->child references
            # (a reference cycle). CPython's generational GC is triggered by
            # ALLOCATION COUNT, not size, so this workload's few-but-huge
            # tensor allocations under-trigger it: cycle-trapped activation
            # tensors accumulate across concepts instead of being freed
            # immediately by refcounting. Confirmed empirically: allocated
            # (not just reserved) CUDA memory climbed ~10GiB -> ~19.5GiB
            # monotonically over concepts 1-39 before OOM, at c50 scale.
            # gc.collect() breaks the cycles so empty_cache() can actually
            # return the freed blocks to the allocator -- releasing only
            # every _MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS bounds the
            # worst-case accumulation instead of paying the release cost on
            # every concept (see the constant's docstring for the tradeoff).
            if _should_release_masked_memory(concept_index, len(training)):
                gc.collect()
                t.cuda.empty_cache()
            allocated = t.cuda.memory_allocated() / 1024**3
            reserved = t.cuda.memory_reserved() / 1024**3
            logger.debug(
                "fit_concept_dense_diffmeans: concept {}/{} (id={}) "
                "cuda_allocated={:.2f}GiB cuda_reserved={:.2f}GiB",
                concept_index + 1,
                len(training),
                concept_id,
                allocated,
                reserved,
            )
    return results


__all__ = [
    "SELF_CONDITIONING_FACTORS",
    "SELF_CONDITIONING_MASKED_POLICY",
    "SELF_CONDITIONING_POLICY",
    "MethodSetup",
    "PreparedPromptContext",
    "SelfConditioningGenerationRequest",
    "assert_masked_diffmean_feasible",
    "assert_prepared_prompt_matches",
    "build_self_conditioning_config",
    "build_self_conditioning_masked_policy",
    "fit_concept_dense_diffmeans",
    "fit_dense_diffmean",
    "generate_self_conditioning",
    "generate_self_conditioning_batch",
    "load_extraction_stack",
    "load_method_setup",
    "load_target_model",
    "prepare_prompt_context",
    "self_conditioning_masked_pool_limit",
    "self_conditioning_weight",
]
