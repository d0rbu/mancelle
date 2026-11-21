"""
Train encoder-decoder conditional model with SAE feature cross-attention. From root directory:

python -m scripts.train.conditional_encdec_train feature_dir=path/to/features

or distributed as:

torchrun --nproc_per_node=8 -m scripts.train.conditional_encdec_train -- feature_dir=path/to/features

Override feature selection settings:
python -m scripts.train.conditional_encdec_train feature_dir=path/to/features \
    '~feature_selection' \
    +feature_selection.threshold_type=segment_topk \
    +feature_selection.threshold=10
"""

import gc
import json
import math
import os
import re
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import torch as t
import wandb
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

import nanochat.flash_attention as fa_module
from errors import (
    FeatureClustersError,
    FeatureMaskError,
    FeatureVocabError,
    GradientAttributionError,
    GradientRankingUnsupportedError,
)
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print_banner,
)
from nanochat.dataset import DatasetSplit, list_parquet_files
from nanochat.flash_attention import FlashAttnImpl
from nanochat.fp8_utils import apply_fp8_training, disable_fp8
from nanochat.gpt import GPTConfig
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.eval.base_eval import evaluate_core
from steering._common.build_meta import resolve_gpt_dimensions
from steering._common.compile_helpers import resolve_compile_dynamic
from steering._common.ddp import ddp_barrier
from steering._common.hydra_cli import parse_config_name_and_overrides
from steering._common.resume import resolve_resume_step
from steering.conditional.compaction_summary import (
    format_compaction_summary,
    no_compaction_reason,
)
from steering.conditional.delta import (
    DeltaMode,
    parse_feature_delta,
)
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.conditioning_cooldown import (
    apply_conditioning_cooldown,
    parse_conditioning_cooldown,
    resolve_conditioning_cooldown,
    validate_conditioning_cooldown_resume,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    parse_feature_prediction_config,
    reject_removed_feature_prediction_keys,
    serialize_feature_prediction_config,
    validate_feature_prediction_resume,
)
from steering.conditional.encoder_decoder.dataloader import (
    DEFAULT_BUFFER_SIZE,
    ClosableBatchPrefetcher,
    PrecomputedValidationSetupCache,
    _prefetch_wrapper,
    build_dataloader_state_envelope,
    encoder_decoder_data_loader_with_state,
    format_loader_health,
    loader_health_delta,
    loader_health_scope,
    loader_health_snapshot,
    precomputed_data_loader,
    resolve_train_prefetch_batches,
    resolve_train_shard_lookahead,
    select_dataloader_rank_state,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    validate_fp_target_source_certificate,
    validate_fp_target_source_certificate_remote,
)
from steering.conditional.encoder_decoder.loss_eval import evaluate_encdec_bpb_conditioned
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.encoder_decoder.precomputed_cuda_staging import (
    PrecomputedCudaStagingLoader,
    validate_precomputed_cuda_staging_eligibility,
)
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingClusterVectors,
    maybe_initialize_encoder_feature_embeddings_from_sae,
    parse_encoder_embedding_cluster_vectors,
)
from steering.conditional.encoder_decoder.sm12_fa4 import admit_project_attention
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    append_feature_clusters_leaf,
    assert_feature_clusters_selection_metadata,
    assert_resume_feature_clusters_match,
    canonical_feature_clusters_descriptor,
    feature_clusters_checkpoint_metadata,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import (
    FeatureMask,
    append_mask_leaf,
    assert_resume_feature_mask_matches,
    canonical_dict,
    resolve_feature_mask,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    append_gradient_leaf,
    avg_features_per_segment_estimate,
    build_feature_pipeline,
    build_feature_pipeline_from_cfg_block,
    feature_selection_dirname,
)
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
    feature_vocab_from_state_dict,
)
from steering.conditional.freq_blacklist import maybe_apply_frequency_blacklist
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.selection_artifact_validation import (
    validate_complete_selection_dir,
)
from steering.conditional.splice_counterfactual import (
    packed_row_fingerprint,
    tensor_content_fingerprint,
)
from steering.conditional.splice_selection_batching import (
    canonical_splice_selection_batching,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    build_thresholds_from_stats,
    extract_threshold_percentile,
)
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.clustered_inline_cache import (
    assert_clustered_inline_cache_ddp_consistent,
    prepare_clustered_inline_eval_cache,
)
from steering.eval.context_cache_io import (
    CacheIdentity,
    SafetensorsContextFeatureProvider,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.inference_engine import build_engine_for_features, build_wrapper_for_features
from steering.eval.inline_steering_eval import (
    InlineEvalCache,
    SteeringEvalConfig,
    prepare_inline_eval_cache,
    restore_rng_state,
    save_rng_state,
)
from steering.eval.inline_steering_eval import (
    evaluate_steering as evaluate_steering_inline,
)
from steering.eval.probe_model import ProbeModel, load_probe_model
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    resolve_output_dir,
    verify_feature_dir,
)
from steering.feature_shard_splice import PARQUET_DIRNAME, PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    assert_resume_gradient_matches,
    canonical_gradient_selection_dict,
    resolve_gradient_spec_from_feature_selection,
)

# nnsight patches torch.Tensor.backward globally at import time (imported
# transitively via steering.eval.inline_steering_eval → steering.extract_features
# → nnterp → nnsight).  The wrapper adds overhead to every backward pass.
# Restore the original before training starts.
from steering.nnsight_compat import unpatch_nnsight
from steering.node_shard_cache import NodeLocalShardCache, resolve_train_node_cache
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    OverlayKind,
    glob_overlay_shards,
    read_manifest,
)
from steering.packed_batch_io import compute_selection_hash
from steering.profiling.rank_skew import log_rank_duration_skew
from steering.profiling.startup import StartupTrace
from steering.refined_types import PositiveInt
from steering.remote_body_prefetch import RemoteBodyPrefetcher
from steering.remote_inline_eval import CertifiedRemoteInlineEvalSource
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    REMOTE_RUN_META_KEY,
    REMOTE_SEMANTIC_META_KEY,
    BucketBackend,
    HfBucketBackend,
    RemoteCheckpointPublisher,
    RemoteCheckpointStore,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    assert_checkpoint_remote_run,
    build_remote_semantic_fingerprint_v2,
    build_remote_semantic_identity,
    ddp_collective_result,
    ddp_create_checkpoint_publisher,
    ddp_drain_checkpoint_publisher,
    ddp_poll_checkpoint_publisher,
    ddp_publish_saved_checkpoint,
    ddp_rank0_action,
    validate_remote_training_eligibility,
)
from steering.removed_features import reject_feature_clustering_config
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
)
from steering.segmentation import compute_tokenizer_checksum
from steering.stats_io import (
    content_sha256,
    find_statistics_path,
    load_percentile_checksum,
    load_statistics,
    load_tokenizer_checksum,
)
from steering.stats_types import ExtractionStatistics, PerFeatureStat

unpatch_nnsight()


_SPLICE_CUT_FRACTION_LEAF_RE = re.compile(r"^cut_frac(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_SPLICE_SEED_LEAF_RE = re.compile(
    r"^seed-?\d+(?:_opposite_percentile_pool(?:[2-9]|[1-9]\d+))?$"
)


def _startup_phase(trace: StartupTrace | None, name: str):
    """Return a startup phase context without adding a disabled-path cost."""
    return trace.phase(name) if trace is not None else nullcontext()


def _startup_skip(
    trace: StartupTrace | None,
    name: str,
    *,
    fields: Mapping[str, object] | None = None,
) -> None:
    if trace is not None:
        trace.skip_phase(name, fields=fields)


@dataclass(frozen=True)
class EffectiveDataPath:
    """The final training-data route selected for one config invocation."""

    packed_rows: bool
    route: str
    selection: str
    source: str
    reason: str


def _describe_effective_data_path(
    *,
    packed_row_dir: Path | None,
    selection_dir: Path | None,
    ignored_packed_rows_for_delta: bool,
    use_precomputed_batches: bool,
    remote_certified: bool,
) -> EffectiveDataPath:
    """Classify the route after all shared discovery and per-config overrides."""
    if packed_row_dir is not None:
        return EffectiveDataPath(
            packed_rows=True,
            route="precomputed",
            selection=("selection_shards" if selection_dir is not None else "fused_base_csr"),
            source="remote_certified" if remote_certified else "local",
            reason="selected",
        )
    if ignored_packed_rows_for_delta:
        reason = "feature_delta"
    elif not use_precomputed_batches:
        reason = "precomputed_disabled"
    else:
        reason = "packed_rows_unavailable"
    return EffectiveDataPath(
        packed_rows=False,
        route="assembled",
        selection="live_feature_pipeline",
        source="remote_certified" if remote_certified else "local",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Dataclasses for modular entry-points (sweep runner / standalone)
# ---------------------------------------------------------------------------


@dataclass
class SharedState:
    """Everything identical across sweep configs — built once, reused many times."""

    tokenizer: Any
    token_bytes: t.Tensor
    base_vocab: int
    # Global/live source index. Remote packed streaming deliberately leaves
    # this unset; its one certified inline-eval source is kept separately.
    feature_index: FeatureIndex | None
    feature_stats: ExtractionStatistics
    per_feature_stats: dict[str, PerFeatureStat]
    n_sae_features: PositiveInt
    resolved_feature_clusters: ResolvedFeatureClusters | None
    feature_clusters: FeatureClusterConsolidation | None
    percentile_checksum: str | None
    model: Any  # compiled model (OptimizedModule or EncoderDecoderGPT)
    orig_model: EncoderDecoderGPT
    cached_context_provider: SafetensorsContextFeatureProvider | None
    device: t.device
    device_type: str
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    extraction_segment_size: PositiveInt
    cfg: DictConfig
    gpu_peak_flops: float
    boundaries: t.Tensor
    d_ref: float  # precomputed D_REF for scaling laws
    b_ref: int  # precomputed B_REF
    tokenizer_checksum: str | None
    # Steering eval (constant across configs)
    steering_eval_cfg: SteeringEvalConfig | None
    steering_probe_model: ProbeModel | None
    feature_dir: str  # needed by steering eval
    packed_row_dir: Path | None
    selection_dir: Path | None
    original_val_feature_dir: Path | None
    original_val_packed_row_dir: Path | None
    original_val_selection_dir: Path | None
    # Live dual validation (issue #596, splice_counterfactual): the base-dir
    # FeatureIndex the delta-free val/* pass reads. None for packed dual val
    # and single-val runs.
    original_val_feature_index: FeatureIndex | None
    # A DEDICATED FeatureIndex over the derived dir for the val_spliced/* pass.
    # The train loader prefetches on a background thread using
    # shared.feature_index, and FeatureIndex.lookup's LRU-1 cache swap is not
    # thread-safe — reusing that same instance for validation races (wrong-shard
    # segments, caught loudly by the #596 sidecar cross-check). A separate
    # instance keeps shared.feature_index single-threaded. None otherwise.
    spliced_val_feature_index: FeatureIndex | None
    # Startup-verified VAL dirs (gradient/static-cluster runs; None otherwise). The
    # eval loop reads THESE instead of re-deriving the paths — independent
    # re-derivations are exactly how the verified-at-startup invariant could
    # silently drift from what the loop actually loads.
    val_packed_row_dir: Path | None
    val_selection_dir: Path | None
    max_seg_size: int | None
    pad_F: int | None  # static feature dim from selection config max_k (fused path)
    live_pad_n_seg: int | None  # static segment dim for live dataloader path
    fp_target_layout: FeaturePredictionTargetLayout | None
    fp_target_source_certificate_digest: str | None
    remote_training: RemoteTrainingConfig | None = None
    remote_backend: BucketBackend | None = None
    packed_materializer: RemotePackedRowMaterializer | None = None
    packed_identity_materializer: RemotePackedRowMaterializer | None = None
    remote_derived_feature_fingerprint: str | None = None
    remote_inline_eval_source: CertifiedRemoteInlineEvalSource | None = None
    remote_materializers: dict[str, RemoteSelectionMaterializer] = field(default_factory=dict)
    remote_run_fingerprint: str | None = None
    remote_resolved_training_config: Mapping[str, Any] | None = None
    remote_selection_contracts: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class PackedSelectionDiscovery:
    """Filesystem discovery results shared by pretraining and post-training."""

    train_packed_dir: Path | None
    train_selection_dir: Path | None
    val_packed_dir: Path | None
    val_selection_dir: Path | None
    original_val_feature_dir: Path | None
    original_val_packed_dir: Path | None
    original_val_selection_dir: Path | None
    original_val_feature_index: FeatureIndex | None
    spliced_val_feature_index: FeatureIndex | None
    max_seg_size: int | None
    pad_n_seg: int | None
    pad_f: int | None
    selection_hash: str | None
    packed_materializer: RemotePackedRowMaterializer | None = None
    packed_identity_materializer: RemotePackedRowMaterializer | None = None
    remote_derived_feature_fingerprint: str | None = None
    remote_materializers: dict[str, RemoteSelectionMaterializer] = field(default_factory=dict)


class DualValidationMode(Enum):
    none = "none"
    live = "live"
    packed = "packed"


@dataclass(frozen=True)
class ValidationEvalSpec:
    """One conditioned-BPB validation pass and its metric namespace."""

    metric_prefix: str
    log_label: str
    packed_row_dir: Path | None
    selection_dir: Path | None
    data_dir: str | None = None
    feature_index: FeatureIndex | None = None  # None ⇒ shared.feature_index
    pipeline: FeaturePipelineConfig | None = None  # None ⇒ the main feature_pipeline
    splice_feature_dir: Path | None = None
    remote_derived_feature_fingerprint: str | None = None


def _sel_coverage_missing(packed_dir: Path, sel_dir: Path) -> list[str]:
    """Packed-shard stems with no selection-shard twin (the loader would
    silently fall back to fused live selection for exactly these)."""
    packed = {p.stem.split(".")[0] for p in glob_overlay_shards(packed_dir)}
    sel = {p.stem.split(".")[0] for p in glob_overlay_shards(sel_dir)}
    return sorted(packed - sel)


def discover_val_packed_row_dir(train_packed_row_dir: Path) -> Path | None:
    """Discover precomputed validation shards as a sibling to train shards."""
    train_dir = Path(train_packed_row_dir)
    val_candidate = train_dir.parent / f"val_{train_dir.name}"
    if val_candidate.is_dir() and glob_overlay_shards(val_candidate):
        return val_candidate
    return None


def _packed_row_train_dir(
    feature_dir: str | Path,
    *,
    percentile_checksum: str,
    tokenizer_checksum: str,
    context_len: int,
    max_train_shards: int | None,
) -> Path:
    shards_str = f"s{max_train_shards}" if max_train_shards is not None else "sall"
    return (
        Path(feature_dir)
        / "pct"
        / percentile_checksum[:CHECKSUM_PREFIX_LEN]
        / "tok"
        / tokenizer_checksum[:CHECKSUM_PREFIX_LEN]
        / f"c{context_len}"
        / shards_str
    )


def _assert_remote_packed_row_manifest_identity(
    packed_row_dir: Path,
    *,
    percentile_checksum: str,
    tokenizer_checksum: str,
    label: str,
) -> None:
    """Require exact full checksums before remote mode trusts packed rows.

    Packed-row directory names contain only checksum prefixes. Remote packed
    training intentionally does not stage the raw percentile overlay, so its
    manifest is the fail-loud certificate that the rows were produced from the
    exact percentile table and tokenizer recorded in feature statistics.
    """
    manifest = read_manifest(packed_row_dir)
    metadata = manifest.get("metadata") if manifest is not None else None
    expected = {
        "checksum": f"{percentile_checksum}:{tokenizer_checksum}",
        "kind": OverlayKind.PACKED_ROW.value,
        "pct_checksum": percentile_checksum,
        "tok_checksum": tokenizer_checksum,
    }
    actual = {
        "checksum": manifest.get("checksum") if manifest is not None else None,
        "kind": manifest.get("kind") if manifest is not None else None,
        "pct_checksum": (metadata.get("pct_checksum") if isinstance(metadata, Mapping) else None),
        "tok_checksum": (metadata.get("tok_checksum") if isinstance(metadata, Mapping) else None),
    }
    if actual != expected:
        raise RemoteTrainingError(
            f"{label} packed-row manifest identity mismatch at {packed_row_dir}: "
            f"expected {expected!r}, got {actual!r}. Regenerate and republish "
            "the packed-row family from the current feature statistics and tokenizer."
        )


def _auto_original_val_feature_dir(feature_dir: str | Path) -> Path | None:
    """Resolve the base of canonical random or opposite-percentile splice paths."""
    path = Path(feature_dir)
    if path.parent.parent.name != "splice":
        return None
    if not _SPLICE_CUT_FRACTION_LEAF_RE.fullmatch(path.parent.name) or not (
        _SPLICE_SEED_LEAF_RE.fullmatch(path.name)
    ):
        raise ValueError(
            "feature_dir appears to be under a splice output directory but does "
            "not match the canonical <base>/splice/cut_frac{cut_fraction}/seed{seed} layout: "
            f"{path}. Set original_val_feature_dir explicitly for custom "
            "splice output dirs."
        )
    return path.parent.parent.parent


def _resolve_original_val_feature_dir(cfg: DictConfig, feature_dir: str | Path) -> Path | None:
    override = cfg.get("original_val_feature_dir", None)
    original = (
        Path(str(override)).expanduser()
        if override is not None
        else _auto_original_val_feature_dir(feature_dir)
    )
    if original is None:
        return None
    active = Path(feature_dir).expanduser()
    if original == active:
        raise ValueError(
            "original_val_feature_dir resolves to the active feature_dir. "
            "It must point at the unspliced/base feature directory."
        )
    if not original.is_dir():
        raise FileNotFoundError(
            f"original validation feature directory not found: {original}. "
            "For a custom splice output_dir, set original_val_feature_dir to "
            "the base feature directory that owns the original validation shard."
        )
    return original


def _load_packed_row_checksums(
    feature_dir: Path,
    *,
    fallback_tokenizer_checksum: str | None,
) -> tuple[str, str]:
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise FileNotFoundError(
            f"feature_statistics.safetensors not found in {feature_dir}; "
            "run compute_statistics before packed-row validation."
        )
    percentile_checksum = load_percentile_checksum(stats_path)
    if percentile_checksum is None:
        raise ValueError(
            f"No percentile checksum found in {stats_path}; run "
            "precompute_percentiles for this feature directory before "
            "packed-row validation."
        )
    tokenizer_checksum = load_tokenizer_checksum(stats_path) or fallback_tokenizer_checksum
    if tokenizer_checksum is None:
        raise ValueError(
            f"No tokenizer checksum found in {stats_path} and no fallback "
            "checksum was provided. Run precompute_token_boundaries for this "
            "feature directory before packed-row validation."
        )
    return percentile_checksum, tokenizer_checksum


def _assert_gradient_val_selection_dir(
    *,
    val_packed_row_dir: Path,
    selection_hash: str,
    configured_gradient: dict[str, Any] | None,
    label: str,
) -> Path:
    val_sel = val_packed_row_dir / "sel" / selection_hash
    if not (val_sel.is_dir() and glob_overlay_shards(val_sel)):
        raise GradientRankingUnsupportedError(
            f"feature_selection.gradient requires {label} selection shards at "
            f"{val_sel}; run precompute_selected_features with "
            f"packed_row_dir={val_packed_row_dir}."
        )
    val_sel_manifest = read_manifest(val_sel)
    val_manifest_gradient = (val_sel_manifest or {}).get("metadata", {}).get("gradient_attribution")
    val_gradient_canonical = (
        {k: v for k, v in val_manifest_gradient.items() if k != "overlay_checksum"}
        if val_manifest_gradient is not None
        else None
    )
    if val_gradient_canonical != configured_gradient:
        raise GradientAttributionError(
            f"{label} sel/ manifest gradient_attribution mismatch for {val_sel}: "
            f"manifest has {val_gradient_canonical!r}, configured "
            f"{configured_gradient!r}. Re-run precompute_selected_features for "
            "that val dir."
        )
    missing_val = _sel_coverage_missing(val_packed_row_dir, val_sel)
    if missing_val:
        raise GradientRankingUnsupportedError(
            f"feature_selection.gradient: {label} selection shards at {val_sel} "
            f"cover only part of the val packed dir ({len(missing_val)} "
            f"uncovered, e.g. {missing_val[:3]}). Re-run "
            f"precompute_selected_features with packed_row_dir={val_packed_row_dir}."
        )
    return val_sel


def _ensure_clustered_original_val_selection_dir(
    *,
    current: Path | None,
    original_val_packed_row_dir: Path,
    selection_hash: str,
    configured_mask: dict[str, Any] | None,
    feature_clusters: FeatureClusterConsolidation | None,
    splice_counterfactual: bool,
) -> Path | None:
    """Make clustered packed dual validation selection-complete on both splits."""
    if feature_clusters is None or current is not None:
        return current
    if splice_counterfactual:
        raise FeatureClustersError(
            "clustered splice_counterfactual dual validation did not resolve its "
            "delta-free original/base selection identity"
        )
    return validate_complete_selection_dir(
        packed_row_dir=original_val_packed_row_dir,
        selection_hash=selection_hash,
        configured_mask=configured_mask,
        label="static feature clusters original/base val",
        feature_clusters=feature_clusters,
        delta_config=None,
    )


def _build_validation_eval_specs(
    *,
    packed_row_dir: Path | None,
    active_val_packed_row_dir: Path | None,
    active_val_selection_dir: Path | None,
    original_val_packed_row_dir: Path | None,
    original_val_selection_dir: Path | None,
    live_dual: bool = False,
    original_val_feature_index=None,
    spliced_val_feature_index=None,
    original_val_data_dir: str | None = None,
    spliced_data_dir: str | None = None,
    live_data_dir: str | None = None,
    delta_free_pipeline=None,
    active_splice_feature_dir: Path | None = None,
    remote_derived_feature_fingerprint: str | None = None,
) -> list[ValidationEvalSpec]:
    if live_dual:
        # Live dual validation (issue #596, splice_counterfactual): both passes
        # use live loaders. val/* runs the delta-free twin pipeline over the
        # original/base FeatureIndex + text; val_spliced/* runs the main
        # feature_pipeline (spec.pipeline=None) over the derived _parquets, on a
        # DEDICATED derived-dir FeatureIndex (never shared.feature_index — that
        # instance is concurrently mutated by the train loader's prefetch thread).
        return [
            ValidationEvalSpec(
                metric_prefix="val",
                log_label="validation (original/base)",
                packed_row_dir=None,
                selection_dir=None,
                data_dir=original_val_data_dir,
                feature_index=original_val_feature_index,
                pipeline=delta_free_pipeline,
            ),
            ValidationEvalSpec(
                metric_prefix="val_spliced",
                log_label="validation (spliced)",
                packed_row_dir=None,
                selection_dir=None,
                data_dir=spliced_data_dir,
                feature_index=spliced_val_feature_index,
            ),
        ]
    if original_val_packed_row_dir is not None:
        if active_val_packed_row_dir is None:
            raise ValueError(
                "spliced dual validation requires both original and spliced "
                "validation packed-row dirs."
            )
        return [
            ValidationEvalSpec(
                metric_prefix="val",
                log_label="original validation",
                packed_row_dir=original_val_packed_row_dir,
                selection_dir=original_val_selection_dir,
                pipeline=delta_free_pipeline,
            ),
            ValidationEvalSpec(
                metric_prefix="val_spliced",
                log_label="spliced validation",
                packed_row_dir=active_val_packed_row_dir,
                selection_dir=active_val_selection_dir,
                splice_feature_dir=active_splice_feature_dir,
                remote_derived_feature_fingerprint=(remote_derived_feature_fingerprint),
            ),
        ]
    if active_val_packed_row_dir is not None or packed_row_dir is None:
        return [
            ValidationEvalSpec(
                metric_prefix="val",
                log_label="validation",
                packed_row_dir=active_val_packed_row_dir,
                selection_dir=active_val_selection_dir,
                data_dir=live_data_dir if active_val_packed_row_dir is None else None,
                remote_derived_feature_fingerprint=(remote_derived_feature_fingerprint),
            )
        ]
    return []


def _validation_bpb_log(prefix: str, metrics) -> dict[str, float]:
    return {f"{prefix}/bpb": metrics.bpb_conditioned}


def _log_validation_feature_stats(
    *,
    wandb_run: Any,
    prefix: str,
    metrics,
    cfg: DictConfig,
    master_process: bool,
) -> None:
    if not master_process:
        return
    wandb_run.log(
        {f"{prefix}/feature_density_histogram": wandb.Histogram(metrics.features_per_segment)}
    )
    wandb_run.summary[f"{prefix}/avg_features_per_segment"] = metrics.avg_features_per_segment
    wandb_run.summary[f"{prefix}/avg_features_per_batch"] = metrics.avg_features_per_batch
    fps_t = t.tensor(metrics.features_per_segment, dtype=t.float32)
    wandb_run.summary[f"{prefix}/features_per_segment_min"] = fps_t.min().item()
    wandb_run.summary[f"{prefix}/features_per_segment_p25"] = fps_t.quantile(0.25).item()
    wandb_run.summary[f"{prefix}/features_per_segment_median"] = fps_t.median().item()
    wandb_run.summary[f"{prefix}/features_per_segment_p75"] = fps_t.quantile(0.75).item()
    wandb_run.summary[f"{prefix}/features_per_segment_max"] = fps_t.max().item()
    logger.info(
        f"{prefix} feature density: avg={metrics.avg_features_per_segment:.1f}/seg, "
        f"median={fps_t.median().item():.0f}, "
        f"range=[{fps_t.min().item():.0f}, {fps_t.max().item():.0f}]"
    )
    if metrics.magnitude_buckets:
        mb_t = t.tensor(metrics.magnitude_buckets, dtype=t.float32)
        wandb_run.log(
            {f"{prefix}/magnitude_bucket_histogram": wandb.Histogram(metrics.magnitude_buckets)}
        )
        wandb_run.summary[f"{prefix}/magnitude_bucket_mean"] = mb_t.mean().item()
        wandb_run.summary[f"{prefix}/magnitude_bucket_median"] = mb_t.median().item()
        wandb_run.summary[f"{prefix}/magnitude_bucket_p25"] = mb_t.quantile(0.25).item()
        wandb_run.summary[f"{prefix}/magnitude_bucket_p75"] = mb_t.quantile(0.75).item()
        wandb_run.summary[f"{prefix}/magnitude_bucket_min"] = int(mb_t.min().item())
        wandb_run.summary[f"{prefix}/magnitude_bucket_max"] = int(mb_t.max().item())
        wandb_run.summary[f"{prefix}/magnitude_bucket_total_activations"] = len(
            metrics.magnitude_buckets
        )
        for bucket_idx in range(cfg.num_buckets):
            count = int((mb_t == bucket_idx).sum().item())
            pct = count / len(metrics.magnitude_buckets) * 100
            wandb_run.summary[f"{prefix}/magnitude_bucket_{bucket_idx}_count"] = count
            wandb_run.summary[f"{prefix}/magnitude_bucket_{bucket_idx}_pct"] = round(pct, 2)
        logger.info(
            f"{prefix} magnitude buckets: mean={mb_t.mean().item():.2f}, "
            f"median={mb_t.median().item():.0f}, "
            f"range=[{int(mb_t.min().item())}, {int(mb_t.max().item())}], "
            f"n={len(metrics.magnitude_buckets)}"
        )


def _build_validation_loader(
    *,
    spec: ValidationEvalSpec,
    cfg: DictConfig,
    shared: SharedState,
    feature_pipeline: FeaturePipelineConfig,
    tokenizer: Any,
    device: t.device,
    node_cache: NodeLocalShardCache | None = None,
    validation_setup_cache: PrecomputedValidationSetupCache | None = None,
):
    if spec.packed_row_dir is None:
        feature_index = (
            spec.feature_index
            if spec.feature_index is not None
            else _require_global_feature_index(
                shared.feature_index,
                surface=f"{spec.log_label} live validation",
            )
        )
        return encoder_decoder_data_loader_with_state(
            tokenizer=tokenizer,
            B=cfg.device_batch_size,
            T=cfg.max_seq_len,
            split=DatasetSplit.VAL,
            feature_index=feature_index,
            pipeline=spec.pipeline or feature_pipeline,
            segment_size=shared.extraction_segment_size,
            decoder_segment_size=None,
            device=str(device),
            resume_state_dict=None,
            data_dir=spec.data_dir,
            label_dropout=0.0,
            max_shards=cfg.get("max_train_shards", None),
            pad_F=shared.pad_F,
            pad_n_seg=shared.live_pad_n_seg,
            next_segment_conditioning_offset=cfg.get("next_segment_conditioning_offset", False),
        )
    remote_materializer = (
        shared.remote_materializers.get(str(spec.selection_dir.resolve()))
        if spec.selection_dir is not None
        else None
    )
    if remote_materializer is not None:
        remote_materializer.request_startup_window()
        remote_materializer.check_worker_errors()
    # Validation drops this bare generator after eval_steps; lookahead would leak per eval cycle.
    return precomputed_data_loader(
        packed_row_dir=spec.packed_row_dir,
        B=cfg.device_batch_size,
        device=str(device),
        feature_pipeline=spec.pipeline or feature_pipeline,
        selection_dir=spec.selection_dir,
        effective_buckets=feature_pipeline.effective_buckets,
        pad_F=shared.pad_F,
        ddp_rank=shared.ddp_rank,
        ddp_world_size=shared.ddp_world_size,
        splice_feature_dir=spec.splice_feature_dir,
        feature_clusters=shared.feature_clusters,
        remote_materializer=remote_materializer,
        remote_derived_feature_fingerprint=(spec.remote_derived_feature_fingerprint),
        node_cache=node_cache,
        validation_setup_cache=(
            validation_setup_cache
            if remote_materializer is None and spec.selection_dir is not None
            else None
        ),
    )


def _validation_setup_cache_for_spec(
    spec: ValidationEvalSpec,
    caches: dict[str, PrecomputedValidationSetupCache],
) -> PrecomputedValidationSetupCache | None:
    """Own one process-local immutable setup for each packed validation spec."""
    if spec.packed_row_dir is None or spec.selection_dir is None:
        return None
    setup_cache = caches.get(spec.metric_prefix)
    if setup_cache is None:
        setup_cache = PrecomputedValidationSetupCache()
        caches[spec.metric_prefix] = setup_cache
    return setup_cache


@contextmanager
def _validation_loader_health(
    *,
    step: int,
    spec: ValidationEvalSpec,
    ddp_rank: int,
) -> Iterator[None]:
    """Report packed-base loads caused synchronously by one validation pass."""
    if spec.packed_row_dir is None:
        yield
        return

    completed = False
    with loader_health_scope() as health:
        try:
            yield
            completed = True
        finally:
            outcome = "complete" if completed else "error"
            logger.info(
                "loader health eval: "
                f"step={step:05d} split={spec.metric_prefix} rank={ddp_rank} "
                f"outcome={outcome} | "
                f"{format_loader_health(health, include_max=True)}"
            )


def _log_loader_health_gate(
    *,
    step: int,
    gate: str,
    ddp_rank: int,
    previous: Mapping[str, float],
) -> dict[str, float]:
    """Log process-local additive loader health since the previous gate."""
    current = loader_health_snapshot()
    delta = loader_health_delta(previous, current)
    logger.info(
        "loader health gate: "
        f"step={step:05d} gate={gate} rank={ddp_rank} | "
        f"{format_loader_health(delta, include_max=False)}"
    )
    return current


@contextmanager
def _remote_checkpoint_save_rank_skew(
    *,
    step: int,
    world_size: int,
    rank: int,
) -> Iterator[None]:
    """Save locally, align failures, then gather remote-branch duration skew."""
    started = time.monotonic()
    local_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        local_error = exc
    elapsed_s = time.monotonic() - started
    ddp_collective_result(
        local_error=local_error,
        world_size=world_size,
        rank=rank,
    )
    # The failure consensus above is load-bearing: no rank may enter this
    # second collective unless every rank completed its local save.
    log_rank_duration_skew(
        f"checkpoint save (step {step})",
        elapsed_s,
        rank=rank,
        world_size=world_size,
    )


def _require_global_feature_index(
    feature_index: FeatureIndex | None,
    *,
    surface: str,
) -> FeatureIndex:
    if feature_index is None:
        raise RuntimeError(
            f"{surface} requires the global raw FeatureIndex, but remote packed "
            "streaming intentionally staged no complete raw feature family"
        )
    return feature_index


def _cfg_dict(value: Any, *, name: str) -> dict[str, Any]:
    """Resolve an OmegaConf mapping and reject non-mapping shapes."""
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"{name} must resolve to a mapping, got {type(resolved).__name__}")
    return cast(dict[str, Any], resolved)


def _optional_cfg_dict(cfg: DictConfig, key: str) -> dict[str, Any] | None:
    """Resolve an optional OmegaConf mapping from *cfg*."""
    if cfg.get(key) is None:
        return None
    return _cfg_dict(cfg[key], name=key)


def _resolved_user_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolved runtime config without producer-only schema fields."""
    user_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    reject_removed_feature_prediction_keys(user_config)
    if user_config.get("feature_prediction") is not None:
        user_config["feature_prediction"] = serialize_feature_prediction_config(
            parse_feature_prediction_config(user_config["feature_prediction"])
        )
    # packed_row_dir exists in the shared schema solely so the selection
    # producer can use typo-checked ``packed_row_dir=...`` syntax. Training
    # auto-discovers packed rows and must never publish this producer input as
    # checkpoint or W&B metadata.
    user_config.pop("packed_row_dir", None)
    # selection_source_cache_shards / selection_sidecar_cache_shards (issue
    # #740) size the precompute-only FeatureIndex/SpliceProvenanceIndex
    # resident caches inside scripts.preprocess.precompute_selected_features.
    # Training itself never reads either key -- same producer-only rationale
    # as packed_row_dir above.
    user_config.pop("selection_source_cache_shards", None)
    user_config.pop("selection_sidecar_cache_shards", None)
    # selection_num_workers / selection_resume / selection_packed_spool_dir are
    # producer-only execution controls for precompute_selected_features;
    # training never reads them and their values do not identify an experiment.
    user_config.pop("selection_num_workers", None)
    user_config.pop("selection_resume", None)
    user_config.pop("selection_packed_spool_dir", None)
    # selection_shared_store / selection_shared_store_capacity (issue #740 PR-F) are
    # producer-only: they toggle + size the shared /dev/shm source store in
    # precompute_selected_features. Training never reads either.
    user_config.pop("selection_shared_store", None)
    user_config.pop("selection_shared_store_capacity", None)
    # remote_preselection is an offline producer transport. It never changes
    # training, checkpoint identity, W&B experiment identity, or selection
    # semantics; the final manifest carries only the existing consumer proofs.
    user_config.pop("remote_preselection", None)
    # Conditional semantic producer mode: only active multi-segment splice
    # batching is retained. Legacy null/1 and every non-splice pipeline omit
    # the key, matching the conditional selection-hash/manifest contract.
    if (
        canonical_splice_selection_batching(
            cast(dict[str, Any] | None, user_config.get("feature_delta")),
            user_config.get("selection_splice_batch_size"),
        )
        is None
    ):
        user_config.pop("selection_splice_batch_size", None)
    # Conditional mode key: absent/null preserves the exact legacy runtime,
    # checkpoint, and W&B metadata contract.
    if user_config.get("remote_training") is None:
        user_config.pop("remote_training", None)
    if parse_conditioning_cooldown(user_config.get("conditioning_cooldown")) is None:
        user_config.pop("conditioning_cooldown", None)
    return user_config


def _checkpoint_user_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolved checkpoint config with disabled conditional keys omitted."""
    user_config = _resolved_user_config(cfg)
    # Hydra removes its reserved top-level runtime stanza from ordinary
    # ``compose()`` results, but keeps the same stanza when a training config is
    # packaged below another key (for example the topology migrator's
    # ``destination_training``). It is process orchestration, never checkpoint
    # identity, so normalize both composition shapes to the native training
    # checkpoint contract.
    user_config.pop("hydra", None)
    # Loader lookahead and node-local caching are execution-only.
    for key in (
        "train_prefetch_batches",
        "train_shard_lookahead",
        "train_node_cache_dir",
        "train_node_cache_max_bytes",
        "train_node_cache_reserve_bytes",
    ):
        user_config.pop(key, None)
    # Remote checkpoint publication is operational: changing sync/async mode
    # must not change checkpoint bytes or resume identity. Keep it in the
    # resolved W&B config, but remove it from a copied checkpoint mapping.
    remote_training = user_config.get("remote_training")
    if isinstance(remote_training, dict):
        checkpoint_remote_training = dict(remote_training)
        checkpoint_remote_training.pop("async_checkpoint_upload", None)
        checkpoint_remote_training.pop("body_prefetch", None)
        user_config["remote_training"] = checkpoint_remote_training
    # The shipped Hydra config declares feature_clusters: null. Absence is the
    # backward-compatible checkpoint contract; do not turn disabled clustering
    # into a metadata schema change by persisting an unconditional null key.
    if user_config.get("feature_clusters") is None:
        user_config.pop("feature_clusters", None)
    # `mean` cluster vectors are the semantic DEFAULT. Keep the explicit value in
    # W&B's resolved snapshot, but do not change the checkpoint user_config schema
    # for any semantically-default run — judged by the same parser the initializer
    # uses, so "MEAN"/"null"/None cannot leak a schema key a lowercase spelling
    # would not. The explicit `representative` opt-out IS retained: it is the one
    # spelling that changes behavior relative to the default. A value the parser
    # rejects raises here, before any checkpoint is written, with the
    # initializer's own error text.
    if (
        parse_encoder_embedding_cluster_vectors(
            user_config.get("initialize_encoder_embeddings_cluster_vectors")
        )
        is EncoderEmbeddingClusterVectors.MEAN
    ):
        user_config.pop("initialize_encoder_embeddings_cluster_vectors", None)
    return user_config


def _reject_precompute_only_packed_row_dir(cfg: DictConfig) -> None:
    """Reject a producer-only path that encoder-decoder training ignores."""
    packed_row_dir = cfg.get("packed_row_dir")
    if packed_row_dir is None:
        return
    raise ValueError(
        "packed_row_dir is only consumed by "
        "scripts.preprocess.precompute_selected_features. "
        "scripts.train.conditional_encdec_train auto-discovers packed rows "
        "from feature_dir; remove the packed_row_dir override from training."
    )


def _splice_counterfactual_enabled(cfg) -> bool:
    fd = _optional_cfg_dict(cfg, "feature_delta")
    parsed = parse_feature_delta(fd)
    return parsed is not None and parsed.mode is DeltaMode.splice_counterfactual


def _pipeline_uses_splice_counterfactual(
    feature_pipeline: FeaturePipelineConfig,
) -> bool:
    bundle = feature_pipeline.delta_bundle
    return bundle is not None and bundle.config.mode is DeltaMode.splice_counterfactual


def _resolve_splice_counterfactual_data_dir(cfg: DictConfig, feature_dir: str) -> None:
    """D8 (issue #596): splice_counterfactual requires the derived dir's own
    parquets as the text source. Auto-derive data_dir when unset; warn on a
    non-canonical explicit value; validate the sidecar dir exists."""
    if not _splice_counterfactual_enabled(cfg):
        return
    prov_dir = Path(feature_dir) / PROVENANCE_DIRNAME
    if not prov_dir.is_dir() or not any(prov_dir.glob("*.json")):
        raise ValueError(
            f"feature_delta.mode=splice_counterfactual requires feature_dir to be "
            f"a spliced derived directory with provenance sidecars; {prov_dir} is "
            f"missing or empty. Run scripts.preprocess.splice_feature_shards on "
            f"the base dir first, then set feature_dir=<derived> and "
            f"data_dir=<derived>/{PARQUET_DIRNAME}."
        )
    canonical = Path(feature_dir) / PARQUET_DIRNAME
    configured = cfg.get("data_dir", None)
    if configured is None:
        if not canonical.is_dir():
            raise ValueError(
                f"cannot auto-derive data_dir: {canonical} does not exist. The "
                f"splice step writes derived parquets there; re-run it or pass "
                f"data_dir= explicitly."
            )
        with open_dict(cfg):
            cfg.data_dir = str(canonical)
        logger.info(f"splice_counterfactual: auto-derived data_dir={cfg.data_dir}")
    elif Path(str(configured)).resolve() != canonical.resolve():
        logger.warning(
            f"splice_counterfactual: data_dir={configured} is not the canonical "
            f"{canonical}; the per-shard n_docs guard will fail fast on a wrong "
            f"pairing."
        )


def _dual_validation_mode(
    *,
    original_val_feature_dir,
    splice_cf_enabled: bool,
    use_precomputed: bool = True,
) -> DualValidationMode:
    """Route dual validation between disabled, live, and packed modes."""
    if original_val_feature_dir is None:
        return DualValidationMode.none
    if splice_cf_enabled:
        return DualValidationMode.packed if use_precomputed else DualValidationMode.live
    return DualValidationMode.packed


def _require_splice_cf_dual_val(cfg, feature_dir, original_val_feature_dir) -> None:
    """splice_counterfactual ALWAYS runs dual validation (issue #596, review
    finding [2]): if the base/original feature dir cannot be resolved (feature_dir
    is not the canonical splice layout AND original_val_feature_dir is unset),
    raise at startup rather than silently degrading to a single-val pass."""
    if _splice_counterfactual_enabled(cfg) and original_val_feature_dir is None:
        raise ValueError(
            "feature_delta.mode=splice_counterfactual requires dual validation, but "
            "no base/original feature directory could be resolved: feature_dir "
            f"({feature_dir}) is not under the canonical "
            "<base>/splice/cut_frac*/seed* layout and original_val_feature_dir is "
            "not set. Set original_val_feature_dir=<base feature dir> explicitly so "
            "val/* can evaluate the unspliced baseline (val_spliced/* uses the "
            "derived dir)."
        )


def _assert_live_val_pair_aligned(
    feature_index: FeatureIndex,
    data_dir: str | None,
    max_shards: int | None,
    label: str,
) -> None:
    """Fail fast if a live-validation feature dir does not pair with its text
    corpus. The val split is the last parquet; its row count must equal that
    shard's feature-doc count, else the val loader would raise a confusing
    KeyError mid-training (issue #596, review finding [1]). The spliced pass has
    an equivalent per-parquet guard inside the loader; the base/original pass
    (unspliced, no sidecar) gets this startup check."""
    parquets = list_parquet_files(data_dir)
    if max_shards is not None:
        parquets = parquets[:max_shards]
    if not parquets:
        raise ValueError(f"{label}: no parquet files found in data_dir={data_dir}.")
    val_parquet = parquets[-1]
    val_stem = Path(val_parquet).stem
    if val_stem not in feature_index.shard_stems:
        raise ValueError(
            f"{label}: validation parquet {val_stem!r} (from {data_dir}) has no "
            f"matching feature shard in {feature_index.feature_dir}. The text "
            f"corpus does not pair with the feature directory."
        )
    n_rows = pq.ParquetFile(val_parquet).metadata.num_rows
    n_docs = len(feature_index.doc_ids_for_shard(val_stem))
    if n_rows != n_docs:
        raise ValueError(
            f"{label}: validation shard {val_stem!r} has {n_docs} feature docs but "
            f"its parquet {val_parquet} has {n_rows} rows — the text corpus "
            f"({data_dir}) does not pair with the feature directory "
            f"({feature_index.feature_dir})."
        )


def _build_delta_free_val_pipeline(
    cfg,
    *,
    per_feature_stats,
    n_sae_features: int,
    num_buckets: int,
    seed: int,
    feature_mask,
) -> FeaturePipelineConfig:
    """Delta-free twin of the main pipeline for the original/base val pass
    (D12): identical selection/negatives/mask, feature_delta=None, and
    delta_weights stripped (the base val shard has no splice points — pure
    absolute-percentile selection)."""
    fs_d = _cfg_dict(cfg.feature_selection, name="feature_selection")
    fs_d = dict(fs_d)
    fs_d.pop("delta_weights", None)
    feature_selection = parse_feature_selection(fs_d)
    neg_raw = _optional_cfg_dict(cfg, "negative_selection")
    negative_selection = parse_negative_selection(neg_raw) if neg_raw is not None else None
    return build_feature_pipeline(
        feature_selection=feature_selection,
        negative_selection=negative_selection,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        seed=seed,
        # ``seed`` already includes the caller's rank offset. Applying rank a
        # second time here would desynchronize this twin from the main pipeline.
        ddp_rank=0,
        feature_delta=None,
        feature_mask=feature_mask,
    )


def _packed_row_inputs_for_pipeline(
    *,
    packed_row_dir: Path | None,
    selection_dir: Path | None,
    feature_pipeline: FeaturePipelineConfig,
) -> tuple[Path | None, Path | None, bool]:
    """Return packed-row inputs, falling back to live loading for feature deltas."""
    if packed_row_dir is not None and feature_pipeline.delta_bundle is not None:
        if _pipeline_uses_splice_counterfactual(feature_pipeline):
            if selection_dir is None:
                raise ValueError(
                    "feature_delta.mode=splice_counterfactual with packed rows "
                    "requires a precomputed selection_dir. Run "
                    "precompute_selected_features for this packed_row_dir, or "
                    "set use_precomputed_batches=false to use the live loader."
                )
            return packed_row_dir, selection_dir, False
        return None, None, True
    return packed_row_dir, selection_dir, False


def build_encdec_meta_model_from_training_cfg(
    cfg: DictConfig,
    *,
    base_vocab: int,
    n_sae_features: int,
    depth: int | None = None,
    feature_vocab: FeatureVocab | None = None,
) -> EncoderDecoderGPT:
    """Build the encoder-decoder model shape from the training config on meta.

    This is the production model-construction path without any surrounding
    process, tokenizer, feature-discovery, FP8, compile, or eval setup.

    ``feature_vocab`` (issue #695) sizes the encoder's feature-indexed tables to the
    feature_mask's allowed universe. ``None`` ⇒ full width, byte-identical to pre-#695.

    Construction on the ``meta`` device is LOAD-BEARING for byte-exactness, not just an
    optimisation: ``nn.Embedding.reset_parameters()`` draws ``numel()`` randoms on a real
    device, so a compact table would draw fewer than a full one and diverge the RNG
    stream before ``init_weights()`` (which slice-inits) ever runs.
    """
    decoder_depth = int(cfg.depth if depth is None else depth)
    model_dim, num_heads = resolve_gpt_dimensions(
        depth=decoder_depth,
        aspect_ratio=cfg.aspect_ratio,
        head_dim=cfg.head_dim,
    )
    dec_config = GPTConfig(
        sequence_len=cfg.max_seq_len,
        vocab_size=base_vocab,
        n_layer=decoder_depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=cfg.window_pattern,
    )
    enc_config = EncoderConfig(
        n_embd=cfg.encoder.n_embd,
        depth=cfg.encoder.depth,
        n_head=cfg.encoder.n_head,
        n_kv_head=cfg.encoder.n_kv_head,
        head_dim=cfg.encoder.head_dim,
    )
    xattn_config = CrossAttentionConfig(
        frequency=cfg.cross_attention.frequency,
        n_head=cfg.cross_attention.n_head,
        n_kv_head=cfg.cross_attention.n_kv_head,
    )
    reject_removed_feature_prediction_keys(cfg)
    fp_config = parse_feature_prediction_config(cfg.get("feature_prediction"))
    with t.device("meta"):
        return EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=n_sae_features,
            num_buckets=cfg.num_buckets,
            feature_pred_config=fp_config,
            feature_vocab=feature_vocab,
        )


def resolve_static_pad_f_from_training_cfg(cfg: DictConfig) -> int | None:
    """Resolve the static feature dimension implied by sampling max_k bounds."""
    feature_selection_raw = _cfg_dict(cfg.feature_selection, name="feature_selection")
    feature_selection = parse_feature_selection(feature_selection_raw)
    if not isinstance(feature_selection, SamplingSelection):
        return None
    if feature_selection.max_k is None:
        return None

    negative_selection_raw = _optional_cfg_dict(cfg, "negative_selection")
    negative_selection = parse_negative_selection(negative_selection_raw)
    negative_max_k = 0
    if negative_selection is not None:
        if negative_selection.max_k is None:
            return None
        negative_max_k = negative_selection.max_k

    pad_f = feature_selection.max_k + negative_max_k
    if cfg.fp8:
        pad_f = ((pad_f + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    return pad_f


def resolve_live_pad_n_seg_from_training_cfg(
    cfg: DictConfig,
    *,
    pad_f: int | None,
) -> int | None:
    """Resolve the live dataloader's static segment dimension."""
    if pad_f is None:
        return None
    single_doc_segs = math.ceil((cfg.max_seq_len + 1) / cfg.segment_size)
    return math.ceil(1.5 * single_doc_segs)


@dataclass
class TrainResult:
    """Return type from train_one_config()."""

    model_tag: str
    val_bpb: float | None
    min_val_bpb: float
    total_training_time: float
    peak_memory_mib: float
    startup_time: float
    error: str | None = None


def _scaling_parameter_count(model: Any) -> int:
    counts = model.num_scaling_params()
    return int(
        counts["transformer_matrices"]
        + counts["lm_head"]
        + counts["wte"]
        + counts["value_embeds"]
        + counts["scalars"]
    )


def _resolve_total_batch_size(
    configured_total_batch_size: Any,
    *,
    model: Any,
    target_param_data_ratio: Any,
    d_ref: float,
    b_ref: int,
) -> int:
    if (
        type(configured_total_batch_size) is not int
        or configured_total_batch_size == 0
        or configured_total_batch_size < -1
    ):
        raise ValueError(
            "total_batch_size must be a positive strict int or -1 for auto, "
            f"got {configured_total_batch_size!r}"
        )
    if configured_total_batch_size > 0:
        return configured_total_batch_size
    scaling_params = _scaling_parameter_count(model)
    target_tokens = int(target_param_data_ratio * scaling_params)
    batch_size_ratio = target_tokens / d_ref
    predicted_batch_size = b_ref * batch_size_ratio**0.383
    return 2 ** round(math.log2(predicted_batch_size))


def _resolve_training_iterations(
    cfg: DictConfig,
    *,
    num_flops_per_token: float,
    total_batch_size: int,
    target_tokens: int,
) -> int:
    """The existing horizon precedence, shared with pre-restoration cooldown checks."""
    if cfg.num_iterations > 0:
        return cfg.num_iterations
    if cfg.target_flops > 0:
        return round(cfg.target_flops / (num_flops_per_token * total_batch_size))
    if cfg.target_param_data_ratio > 0:
        return target_tokens // total_batch_size
    raise ValueError("No training horizon specified")


# ---------------------------------------------------------------------------
# _build_cached_context_provider(): SafetensorsContextFeatureProvider setup
# ---------------------------------------------------------------------------


def _build_cached_context_provider(
    cfg,
    *,
    stats_path: Path | None = None,
    original_stats_path: Path | None = None,
    n_sae_features: int | None = None,
    num_buckets: int | None = None,
    feature_mask: FeatureMask | None = None,
    feature_clusters: FeatureClusterConsolidation | None = None,
    feature_vocab: FeatureVocab | None = None,
) -> SafetensorsContextFeatureProvider | None:
    """Construct the SafetensorsContextFeatureProvider for in-training CORE
    eval and sample generation.

    Returns None when cfg.conditioning.context_cache_dir is unset (operator
    opted into cached-mode evals only by setting that knob; missing-knob
    fail-fast is enforced separately in _validate_eval_config).
    """
    cond_cfg = cfg.get("conditioning", None)
    if cond_cfg is None:
        return None
    cache_root = cond_cfg.get("context_cache_dir", None)
    if cache_root is None:
        return None
    identity_cfg = cond_cfg.get("identity", None)
    if identity_cfg is None:
        raise ValueError(
            "conditioning.context_cache_dir is set but conditioning.identity "
            "is missing. Identity is required to validate the cache manifest "
            "(model_id/sae_id/layer/pooling/feature_selection_hash)."
        )
    identity = cache_identity_from_config(identity_cfg, cond_cfg)
    model_max_seq_len = int(
        cfg.get(
            "model_max_seq_len",
            cfg.get("max_seq_len", cfg.get("sequence_len", DEFAULT_CONTEXT_WINDOW)),
        )
    )
    cache_dir = context_length_cache_dir(
        cache_root,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    if feature_clusters is None:
        # Literal legacy provider contract for all unclustered runs.
        expected_n_sae_features = int(cond_cfg.get("n_sae_features", 16384))
        expected_num_buckets = int(cond_cfg.get("num_buckets", 2))
        expected_metadata = {"model_max_seq_len": model_max_seq_len}
        expected_feature_vocab = None
    else:
        if stats_path is None or n_sae_features is None or num_buckets is None:
            raise ValueError(
                "clustered cached-context setup requires resolved stats_path, "
                "n_sae_features, and num_buckets"
            )
        if feature_clusters.feature_mask != feature_mask:
            raise FeatureClustersError(
                "cached-context clusters were derived from a different FeatureMask"
            )
        expected_vocab = effective_feature_vocab(
            feature_clusters,
            n_sae_features,
        )
        if expected_vocab != feature_vocab:
            raise FeatureVocabError("cached-context FeatureVocab differs from the model vocabulary")
        for key, resolved_value in (
            ("n_sae_features", n_sae_features),
            ("num_buckets", num_buckets),
        ):
            configured_value = cond_cfg.get(key, None)
            if configured_value is not None and int(configured_value) != resolved_value:
                raise ValueError(
                    f"conditioning.{key}={configured_value!r} does not match "
                    f"the resolved model value {resolved_value}"
                )

        prompt_selection = dict(_cfg_dict(cfg.feature_selection, name="feature_selection"))
        # Codex cross-review fix (Finding 1, HIGH): RETAIN delta_weights as
        # an explicit null rather than popping the key entirely, but only
        # for sampling-shaped selections ('distribution' present).
        # compute_selection_hash (below) hashes raw key PRESENCE, not the
        # parsed value -- parse_feature_selection treats an absent key and
        # delta_weights=None identically (steering/conditional/sampling.py),
        # but popping the key here made derived_hash disagree with
        # cache_identity.py's clustered branch (which retains the null key,
        # matching this exact packed-row-producer convention) for every
        # clustered config, permanently failing the cross-check below.
        # delta_weights is a SamplingSelection-only concept -- parse_
        # feature_selection is a closed per-arm schema, so injecting it
        # into a threshold-shaped dict raises "unknown key" before the
        # isinstance(..., SamplingSelection) check below ever runs,
        # breaking the intended "prompt caches support sampling selection
        # only" rejection message for threshold configs.
        if "distribution" in prompt_selection:
            prompt_selection["delta_weights"] = None
        if not isinstance(
            parse_feature_selection(prompt_selection),
            SamplingSelection,
        ):
            raise FeatureClustersError(
                "clustered CORE/sample prompt caches support sampling selection "
                "only; threshold cache semantics are not implemented"
            )
        negative_selection = _optional_cfg_dict(cfg, "negative_selection")
        configured_stats_sha256 = cond_cfg.get("stats_source_sha256", None)
        if configured_stats_sha256 is not None:
            active_stats_sha256 = content_sha256(Path(stats_path))
            if str(configured_stats_sha256) != active_stats_sha256:
                original_stats_sha256 = (
                    None
                    if original_stats_path is None
                    else content_sha256(Path(original_stats_path))
                )
                if str(configured_stats_sha256) != original_stats_sha256:
                    raise ValueError(
                        "conditioning.stats_source_sha256 does not match either "
                        "the active training statistics source or the resolved "
                        "original/base statistics source: "
                        f"configured={configured_stats_sha256!r}, "
                        f"active={active_stats_sha256!r}, "
                        f"original={original_stats_sha256!r}"
                    )
                assert original_stats_path is not None
                stats_path = original_stats_path
                logger.info(
                    "Clustered prompt-cache statistics identity resolved from "
                    f"the original/base feature directory: {stats_path}"
                )
        stats_source_sha256 = content_sha256(Path(stats_path))
        # D-seed (PR #739 Track D) corollary site: this cross-check
        # re-derives the clustered prompt-cache identity from the
        # TOP-LEVEL training config (as a consistency check that
        # conditioning's own copy of the selection/mask/cluster policy
        # genuinely matches training's) and must use the exact same
        # real-seed, delta_config=None formula
        # steering/eval/cache_identity.py's clustered branch uses for
        # `identity` above -- otherwise this cross-check would compare a
        # real-seed hash against a permanently-stale legacy hardcoded
        # seed=0 one and never agree. Uses the training script's own
        # established cfg.get("seed", 42) idiom (matching this same
        # file's other compute_selection_hash call, the inline-steering
        # site further below) rather than a strict int(cfg.seed), since
        # this is an existing call site being corrected in place, not a
        # new strict-validation surface. stats_source_sha256 remains
        # computed above for clustered_context_cache_metadata's stamping
        # (below) -- it is intentionally NOT part of this hash payload;
        # statistics identity is independently pinned by the clustered
        # feature contract + cache manifest (see cache_identity.py's
        # clustered-branch comment).
        derived_hash = compute_selection_hash(
            positive_selection=prompt_selection,
            negative_selection=negative_selection,
            delta_config=None,
            num_buckets=num_buckets,
            seed=int(cfg.get("seed", 42)),
            ddp_rank=0,
            feature_mask=canonical_dict(feature_mask),
            feature_clusters=canonical_feature_clusters_descriptor(feature_clusters.resolved),
            splice_batch_size=None,
        )
        if identity.feature_selection_hash != derived_hash:
            raise ValueError(
                "conditioning.identity.feature_selection_hash does not match "
                "the delta-free clustered prompt-cache identity derived from "
                "the top-level training selection/statistics/mask/clusters: "
                f"configured={identity.feature_selection_hash}, derived={derived_hash}"
            )
        identity = CacheIdentity(
            model_id=identity.model_id,
            sae_id=identity.sae_id,
            layer=identity.layer,
            pooling=identity.pooling,
            feature_selection_hash=derived_hash,
        )
        expected_n_sae_features = n_sae_features
        expected_num_buckets = num_buckets
        expected_metadata = {
            "model_max_seq_len": model_max_seq_len,
            **clustered_context_cache_metadata(
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                feature_vocab=feature_vocab,
                stats_source_sha256=stats_source_sha256,
            ),
        }
        expected_feature_vocab = feature_vocab
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=expected_n_sae_features,
        expected_num_buckets=expected_num_buckets,
        expected_metadata=expected_metadata,
        expected_feature_vocab=expected_feature_vocab,
    )


def _load_steering_probe_model(cfg, device) -> ProbeModel:
    """Load the in-training probe with the extraction SAE identity.

    Hydra represents Gemma-Scope-2's L0 tier as a plain string. The resolver
    requires the typed tier, so this config boundary uses the same
    release-aware coercion as feature extraction.
    """
    sae_release = SAERelease(cfg.sae.release)
    return load_probe_model(
        model_name=cfg.sae.model,
        layer=cfg.sae.layer,
        activation_type=cfg.sae.activation_type,
        sae_release=cfg.sae.release,
        sae_width=cfg.sae.width,
        device=device,
        l0=coerce_config_l0(sae_release, cfg.sae.get("l0")),
    )


# ---------------------------------------------------------------------------
# _validate_eval_config(): hard-fail rule for missing context_cache_dir
# ---------------------------------------------------------------------------


def _validate_eval_config(cfg) -> None:
    """Hard-fail at startup if in-training evals are enabled without a cache.

    No silent fallback: in-training encoder-decoder evals require full-context
    cached SAE features.
    """
    needs_cache = int(cfg.get("core_metric_every", 0)) > 0 or int(cfg.get("sample_every", 0)) > 0
    cond = cfg.get("conditioning", None)
    cache_dir = None if cond is None else cond.get("context_cache_dir", None)
    if needs_cache and cache_dir is None:
        raise ValueError(
            f"core_metric_every={cfg.get('core_metric_every', 0)} and "
            f"sample_every={cfg.get('sample_every', 0)} require "
            f"conditioning.context_cache_dir to point at a cache built by "
            f"scripts.eval.build_eval_context_cache (with suites including "
            f"'core' and 'samples'). Either set the cache dir or set both "
            f"core_metric_every=0 (or any value <= 0) and sample_every=0 (or any value <= 0)."
        )


def _validate_feature_clusters_training_config(cfg: DictConfig) -> None:
    """Fail before model/data work when clustered pretraining would escape V1."""
    if cfg.get("feature_clusters", None) is None:
        return
    if not bool(cfg.get("use_precomputed_batches", True)):
        raise FeatureClustersError(
            "feature_clusters requires use_precomputed_batches=true; live/fused "
            "feature selection does not implement cluster semantics"
        )
    if bool(cfg.get("next_segment_conditioning_offset", False)):
        raise FeatureClustersError(
            "feature_clusters cannot use next_segment_conditioning_offset because "
            "that mode is unsupported with precomputed packed rows"
        )

    feature_selection_raw = _cfg_dict(
        cfg.feature_selection,
        name="feature_selection",
    )
    if resolve_gradient_spec_from_feature_selection(feature_selection_raw) is not None:
        raise FeatureClustersError(
            "feature_clusters cannot be combined with gradient-attribution ranking in V1"
        )
    feature_selection = parse_feature_selection(feature_selection_raw)
    feature_delta = parse_feature_delta(_optional_cfg_dict(cfg, "feature_delta"))
    if feature_delta is not None and feature_delta.mode is not DeltaMode.splice_counterfactual:
        raise FeatureClustersError(
            "feature_clusters with packed preselection supports feature_delta=null or "
            "feature_delta.mode=splice_counterfactual only"
        )
    steering_eval_every = int(cfg.get("steering_eval_every", 0))
    if steering_eval_every > 0 and (
        feature_delta is None or feature_delta.mode is not DeltaMode.splice_counterfactual
    ):
        raise FeatureClustersError(
            "clustered in-training steering currently requires "
            "feature_delta.mode=splice_counterfactual and its precomputed "
            "validation selection/provenance artifacts"
        )
    if feature_delta is not None and not isinstance(feature_selection, SamplingSelection):
        raise FeatureClustersError(
            "feature_clusters with splice_counterfactual supports sampling selection only in V1"
        )
    if isinstance(feature_selection, ThresholdSelection) and (
        feature_selection.threshold_type
        not in {
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        }
    ):
        raise FeatureClustersError(
            "feature_clusters ordinary preselection supports only percentile-based "
            "thresholds; raw-magnitude representative semantics are undefined"
        )


def _feature_index_tokenizer_checksum_for_training(
    cfg: DictConfig,
    tokenizer_checksum: str | None,
) -> str | None:
    """Avoid requiring an unused raw token overlay in clustered packed mode.

    Static clusters have no live/fused fallback: startup later requires exact,
    complete packed-row and selection manifests, and those manifests bind the
    tokenizer checksum.  Conditioned-BPB validation also consumes packed rows,
    while CORE/sample evaluation consumes the separately validated context
    cache.  With steering disabled, the source ``FeatureIndex`` is therefore
    never asked for token boundaries.  Clustered inline steering does join its
    selected segments through that index, so steering-enabled runs retain the
    checksum and its fail-loud overlay guard.  Every live/unclustered mode keeps
    the existing guard as well.
    """
    if (
        cfg.get("feature_clusters", None) is not None
        and bool(cfg.get("use_precomputed_batches", True))
        and int(cfg.get("steering_eval_every", 0)) <= 0
    ):
        return None
    return tokenizer_checksum


def _feature_index_percentile_checksum_for_training(
    cfg: DictConfig,
    remote_training: RemoteTrainingConfig | None,
    percentile_checksum: str | None,
) -> str | None:
    """Avoid validating an intentionally unstaged raw percentile overlay.

    Remote eligibility has already restricted this route to static clustered,
    precomputed training. Those consumers read percentile-derived values from
    authenticated packed rows and the statistics table, never from the source
    ``FeatureIndex`` percentile overlay. Local and live modes retain the
    existing complete-family integrity guard.
    """
    if (
        remote_training is not None
        and cfg.get("feature_clusters", None) is not None
        and cfg.get("use_precomputed_batches") is True
    ):
        return None
    return percentile_checksum


def _feature_index_required_token_boundary_stems_for_training(
    cfg: DictConfig,
) -> frozenset[str] | None:
    """Bound clustered static steering to its one validation source shard.

    Precomputed clustered training and validation consume packed token rows;
    the raw ``FeatureIndex`` token overlay is used only by the startup inline
    steering join. That join is deliberately restricted to the final derived
    validation parquet. Requiring token overlays for every training shard
    would add hundreds of unused payloads and contradict the bounded two-pass
    cache contract. All live and unclustered callers retain ``None``, which
    means complete-family coverage exactly as before.
    """
    if not (
        cfg.get("feature_clusters", None) is not None
        and bool(cfg.get("use_precomputed_batches", True))
        and int(cfg.get("steering_eval_every", 0)) > 0
    ):
        return None
    parquet_paths = list_parquet_files(cfg.get("data_dir", None))
    if not parquet_paths:
        raise FileNotFoundError(
            "clustered inline steering cannot resolve its derived validation "
            "parquet for token-boundary overlay validation"
        )
    return frozenset({Path(parquet_paths[-1]).stem})


# ---------------------------------------------------------------------------
# _run_sample_generation(): per-(mode, prompt) sample generation helper
# ---------------------------------------------------------------------------


def _run_sample_generation(
    *,
    orig_model,
    tokenizer,
    device,
    sample_prompts,
    cached_context_provider,
) -> None:
    """Run sample generation with real per-segment prompt features.

    Caller is responsible for model.eval()/train() framing and master_process gating.
    """
    logger.info("Samples (cached context):")
    for prompt in sample_prompts:
        engine = build_engine_for_features(
            orig_model,
            tokenizer,
            device,
            features=None,
            context_provider=cached_context_provider,
        )
        with disable_fp8(orig_model):
            sample_text = engine.generate(
                prompt,
                max_new_tokens=16,
                temperature=0.0,
            )
        logger.info(sample_text)


# ---------------------------------------------------------------------------
# build_shared_state(): one-time setup shared across all sweep configs
# ---------------------------------------------------------------------------


def discover_packed_selection(
    *,
    feature_dir: Path,
    original_val_feature_dir: Path | None,
    feature_selection: dict[str, Any],
    negative_selection: dict[str, Any] | None,
    feature_delta: dict[str, Any] | None,
    num_buckets: int,
    seed: int,
    feature_mask: dict[str, Any] | None,
    feature_clusters: dict[str, Any] | None,
    n_sae_features: int,
    use_precomputed_batches: bool,
    percentile_checksum: str | None,
    tokenizer_checksum: str | None,
    context_len: int,
    segment_size: int,
    max_train_shards: int | None,
    compile_static: bool | str,
    fp8: bool,
    eval_every: int,
    original_val_data_dir: str | None,
    boundary_fingerprint: str | None,
    selection_splice_batch_size: object = None,
    remote_training: RemoteTrainingConfig | None = None,
    feature_root: Path | None = None,
    remote_backend: BucketBackend | None = None,
) -> PackedSelectionDiscovery:
    """Discover packed rows and selection overlays without process/model setup."""
    feature_dir_str = str(feature_dir)
    cfg = OmegaConf.create(
        {
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
            "feature_delta": feature_delta,
            "num_buckets": num_buckets,
            "seed": seed,
            "selection_splice_batch_size": selection_splice_batch_size,
            "feature_mask": feature_mask,
            "feature_clusters": feature_clusters,
            "use_precomputed_batches": use_precomputed_batches,
            "max_train_shards": max_train_shards,
            "max_seq_len": context_len,
            "segment_size": segment_size,
            "compile_static": compile_static,
            "fp8": fp8,
            "eval_every": eval_every,
            "original_val_feature_dir": (
                str(original_val_feature_dir) if original_val_feature_dir is not None else None
            ),
            "original_val_data_dir": original_val_data_dir,
        }
    )
    assert isinstance(cfg, DictConfig)
    remote_materializers: dict[str, RemoteSelectionMaterializer] = {}
    packed_materializer: RemotePackedRowMaterializer | None = None
    packed_identity_materializer: RemotePackedRowMaterializer | None = None
    remote_packed_streaming = (
        remote_training is not None and remote_training.train_packed is not None
    )
    if remote_training is not None:
        if feature_root is None:
            raise RemoteTrainingError("remote packed-selection discovery requires feature_root")
        if remote_backend is None:
            remote_backend = HfBucketBackend(remote_training.bucket_id)
    if remote_packed_streaming:
        if find_statistics_path(feature_dir) is None:
            raise FileNotFoundError(
                "remote train packed streaming requires pre-staged derived "
                f"feature statistics in {feature_dir}; copy "
                "feature_statistics.safetensors before launch"
            )

    def _remote_materializer(
        *,
        packed_dir: Path,
        selection_checksum: str,
        label: str,
        train_packed_materializer: RemotePackedRowMaterializer | None = None,
    ) -> RemoteSelectionMaterializer | None:
        if remote_training is None:
            return None
        assert feature_root is not None
        assert remote_backend is not None
        selection_path = packed_dir / "sel" / selection_checksum
        key = str(selection_path.resolve())
        prior = remote_materializers.get(key)
        if prior is not None:
            return prior
        if train_packed_materializer is not None:
            packed_fingerprint = train_packed_materializer.packed_row_fingerprint
            packed_stems = train_packed_materializer.stems()
        else:
            packed_paths = glob_overlay_shards(packed_dir)
            if not packed_paths:
                raise RemoteTrainingError(
                    f"remote_training requires local packed rows for {label}: {packed_dir}"
                )
            packed_fingerprint = packed_row_fingerprint(
                packed_dir,
                cache_dir=packed_dir,
            )
            packed_stems = {path.stem.split(".")[0] for path in packed_paths}
        materializer = RemoteSelectionMaterializer.discover(
            backend=remote_backend,
            remote=remote_training,
            local_selection_dir=selection_path,
            feature_root=feature_root,
            expected_selection_checksum=selection_checksum,
            packed_row_fingerprint=packed_fingerprint,
            packed_stems=packed_stems,
            label=label,
        )
        remote_materializers[key] = materializer
        return materializer

    discovery_feature_mask = resolve_feature_mask(cfg, n_sae_features)
    resolved_feature_clusters = resolve_feature_clusters(cfg, n_sae_features)
    feature_cluster_state = (
        FeatureClusterConsolidation.build(
            resolved_feature_clusters,
            discovery_feature_mask,
        )
        if resolved_feature_clusters is not None
        else None
    )

    # --- Discover precomputed packed rows and selection shards ---
    packed_row_dir = None
    selection_dir = None
    val_packed_row_dir = None
    val_selection_dir = None
    resolved_original_val_feature_dir = _resolve_original_val_feature_dir(cfg, feature_dir_str)
    _require_splice_cf_dual_val(cfg, feature_dir_str, resolved_original_val_feature_dir)
    original_val_packed_row_dir = None
    original_val_selection_dir = None
    original_val_feature_index: FeatureIndex | None = None
    spliced_val_feature_index: FeatureIndex | None = None
    max_seg_size = None
    use_precomputed = bool(cfg.get("use_precomputed_batches", True))
    feature_delta_raw_for_discovery = _optional_cfg_dict(cfg, "feature_delta")
    feature_delta_for_discovery = parse_feature_delta(feature_delta_raw_for_discovery)
    splice_cf_for_discovery = (
        feature_delta_for_discovery is not None
        and feature_delta_for_discovery.mode is DeltaMode.splice_counterfactual
    )
    selection_hash: str | None = None
    # Gradient-attribution ranking is offline-only: it REQUIRES precomputed
    # packed rows + sel/ shards (for the train AND val splits). Every soft
    # fallback below becomes a hard error when it is configured.
    discovery_grad_spec = resolve_gradient_spec_from_feature_selection(
        _cfg_dict(cfg.feature_selection, name="feature_selection")
    )
    configured_gradient = (
        canonical_gradient_selection_dict(discovery_grad_spec)
        if discovery_grad_spec is not None
        else None
    )
    if discovery_grad_spec is not None and not use_precomputed:
        raise GradientRankingUnsupportedError(
            "feature_selection.gradient requires use_precomputed_batches=true "
            "(gradient ranking is offline-only; the live dataloader cannot "
            "rank by gradient attribution)."
        )
    if discovery_grad_spec is not None and feature_delta_raw_for_discovery is not None:
        raise GradientRankingUnsupportedError(
            "feature_selection.gradient cannot be combined with feature_delta "
            "(delta conditioning is live-only; gradient ranking is "
            "precompute-only)."
        )
    if (
        use_precomputed
        and feature_delta_raw_for_discovery is not None
        and not splice_cf_for_discovery
    ):
        logger.info(
            "feature_delta is enabled; skipping precomputed packed-row discovery "
            "because packed rows do not preserve the live dataloader's "
            "document-history state."
        )
    elif use_precomputed and percentile_checksum and tokenizer_checksum:
        candidate = _packed_row_train_dir(
            feature_dir_str,
            percentile_checksum=percentile_checksum,
            tokenizer_checksum=tokenizer_checksum,
            context_len=cfg.max_seq_len,
            max_train_shards=max_train_shards,
        )
        feature_selection_raw = _cfg_dict(cfg.feature_selection, name="feature_selection")
        negative_selection_raw = _optional_cfg_dict(cfg, "negative_selection")
        selection_hash = compute_selection_hash(
            positive_selection=feature_selection_raw,
            negative_selection=negative_selection_raw,
            delta_config=feature_delta_raw_for_discovery,
            num_buckets=cfg.num_buckets,
            seed=cfg.get("seed", 42),
            ddp_rank=0,
            feature_mask=canonical_dict(discovery_feature_mask),
            feature_clusters=canonical_feature_clusters_descriptor(resolved_feature_clusters),
            splice_batch_size=cfg.get("selection_splice_batch_size"),
        )
        if remote_packed_streaming:
            assert remote_training is not None
            assert remote_training.train_packed is not None
            assert feature_root is not None
            assert remote_backend is not None
            packed_materializer = RemotePackedRowMaterializer.discover(
                backend=remote_backend,
                remote=remote_training,
                local_packed_dir=candidate,
                feature_root=feature_root,
                cache_shards=remote_training.body_prefetch.cache_shards,
                label="remote train packed rows",
            )
            packed_identity_materializer = packed_materializer
            if not packed_materializer.base_shard_stems:
                raise RemoteTrainingError(
                    "remote train packed manifest has no base_shard_stems; "
                    "regenerate and republish the packed family"
                )
        if candidate.is_dir() and (
            glob_overlay_shards(candidate) or packed_materializer is not None
        ):
            if (
                remote_training is not None
                and packed_identity_materializer is None
                and (candidate / REMOTE_PACKED_INVENTORY_FILENAME).is_file()
            ):
                assert feature_root is not None
                assert remote_backend is not None
                packed_identity_materializer = RemotePackedRowMaterializer.discover(
                    backend=remote_backend,
                    remote=remote_training,
                    local_packed_dir=candidate,
                    feature_root=feature_root,
                    cache_shards=2,
                    label="remote train packed identity",
                )
            packed_row_dir = candidate
            manifest = read_manifest(packed_row_dir)
            if remote_training is not None:
                _assert_remote_packed_row_manifest_identity(
                    packed_row_dir,
                    percentile_checksum=percentile_checksum,
                    tokenizer_checksum=tokenizer_checksum,
                    label="remote train",
                )
            meta = manifest.get("metadata", {}) if manifest else {}
            max_seg_size = meta.get("max_seg_size")
            logger.info(
                f"Found precomputed packed rows: {packed_row_dir} (max_seg_size={max_seg_size})"
            )

            # ANTI-DESYNC: this selection-hash is the discovery-side ``compute_selection_hash`` call at:
            #   1. scripts/preprocess/precompute_selected_features.py
            #   2. scripts/train/conditional_encdec_train.py:discover_packed_selection (THIS site)
            #   3. scripts/eval/base_eval.py:_resolve_bpb_selection_dir
            #   4. steering/eval/feature_selection_hash.py:compute_canonical_feature_selection_hash (cache identity)
            # All four must agree on feature-mask, cluster, and splice-batching semantics.
            # Change ALL FOUR or none.
            sel_candidate = packed_row_dir / "sel" / selection_hash
            train_remote_materializer = _remote_materializer(
                packed_dir=packed_row_dir,
                selection_checksum=selection_hash,
                label="remote train selection",
                train_packed_materializer=packed_identity_materializer,
            )
            if sel_candidate.is_dir() and (
                glob_overlay_shards(sel_candidate) or train_remote_materializer is not None
            ):
                selection_dir = sel_candidate
                # Defense-in-depth (I4): assert the sel/ manifest records the same
                # feature_mask as the configured one.  Guards hash collisions and
                # moved/mislabeled sel/ dirs.
                # Back-compat (I5): legacy manifest with no feature_mask key + no
                # configured mask → manifest_mask is None, configured canonical is
                # None → None == None → passes, no error.
                sel_manifest = read_manifest(sel_candidate)
                manifest_mask = (
                    sel_manifest.get("metadata", {}).get("feature_mask") if sel_manifest else None
                )
                configured_canonical = canonical_dict(discovery_feature_mask)
                if manifest_mask != configured_canonical:
                    raise FeatureMaskError(
                        f"sel/ manifest feature_mask mismatch for {sel_candidate}: "
                        f"manifest has {manifest_mask!r}, configured {configured_canonical!r}. "
                        "This guards against a hash collision or a moved/mislabeled sel/ dir. "
                        "Re-run precompute_selected_features with the current config."
                    )
                if sel_manifest is not None:
                    assert_feature_clusters_selection_metadata(
                        feature_cluster_state,
                        sel_manifest.get("metadata", {}),
                        surface=f"sel/ manifest {sel_candidate}",
                    )
                manifest_gradient = (
                    sel_manifest.get("metadata", {}).get("gradient_attribution")
                    if sel_manifest
                    else None
                )
                configured_gradient = (
                    canonical_gradient_selection_dict(discovery_grad_spec)
                    if discovery_grad_spec is not None
                    else None
                )
                manifest_gradient_canonical = (
                    {
                        key: value
                        for key, value in manifest_gradient.items()
                        if key != "overlay_checksum"
                    }
                    if manifest_gradient is not None
                    else None
                )
                if manifest_gradient_canonical != configured_gradient:
                    raise GradientAttributionError(
                        f"sel/ manifest gradient_attribution mismatch for "
                        f"{sel_candidate}: manifest has "
                        f"{manifest_gradient_canonical!r}, configured "
                        f"{configured_gradient!r}. Re-run "
                        "precompute_selected_features with the current config."
                    )
                if discovery_grad_spec is not None:
                    # Fail at startup (not mid-run) if ANY shard would have
                    # to live-select. Existence is not enough: the loader
                    # pairs sel shards to base shards PER STEM and silently
                    # falls back to the fused live path for uncovered stems —
                    # which the central guard turns into a
                    # GradientRankingUnsupportedError hours into training.
                    # Compare stem SETS for the train and val splits.
                    missing_train = _sel_coverage_missing(packed_row_dir, sel_candidate)
                    if missing_train:
                        raise GradientRankingUnsupportedError(
                            f"feature_selection.gradient: selection shards at "
                            f"{sel_candidate} cover only part of the packed "
                            f"dir ({len(missing_train)} base shards uncovered, "
                            f"e.g. {missing_train[:3]}) — an interrupted or "
                            f"out-of-date precompute. Re-run "
                            f"precompute_selected_features (it skips existing "
                            f"shards)."
                        )
                    val_candidate = candidate.parent / f"val_{candidate.name}"
                    val_sel = val_candidate / "sel" / selection_hash
                    if not (val_candidate.is_dir() and glob_overlay_shards(val_candidate)):
                        raise GradientRankingUnsupportedError(
                            f"feature_selection.gradient requires precomputed VAL "
                            f"packed rows at {val_candidate} (the in-training "
                            f"validation loader cannot live-select)."
                        )
                    if not (val_sel.is_dir() and glob_overlay_shards(val_sel)):
                        raise GradientRankingUnsupportedError(
                            f"feature_selection.gradient requires VAL selection "
                            f"shards at {val_sel}; run "
                            f"precompute_selected_features with "
                            f"packed_row_dir={val_candidate}."
                        )
                    # The val sel manifest must record the same gradient
                    # block (the train-side I4 twin), and coverage must be
                    # 1:1 there too.
                    val_sel_manifest = read_manifest(val_sel)
                    val_manifest_gradient = (
                        (val_sel_manifest or {}).get("metadata", {}).get("gradient_attribution")
                    )
                    val_gradient_canonical = (
                        {
                            key: value
                            for key, value in val_manifest_gradient.items()
                            if key != "overlay_checksum"
                        }
                        if val_manifest_gradient is not None
                        else None
                    )
                    if val_gradient_canonical != configured_gradient:
                        raise GradientAttributionError(
                            f"VAL sel/ manifest gradient_attribution mismatch "
                            f"for {val_sel}: manifest has "
                            f"{val_gradient_canonical!r}, configured "
                            f"{configured_gradient!r}. Re-run "
                            f"precompute_selected_features for the val dir."
                        )
                    missing_val = _sel_coverage_missing(val_candidate, val_sel)
                    if missing_val:
                        raise GradientRankingUnsupportedError(
                            f"feature_selection.gradient: VAL selection shards "
                            f"at {val_sel} cover only part of the val packed "
                            f"dir ({len(missing_val)} uncovered, e.g. "
                            f"{missing_val[:3]}). Re-run "
                            f"precompute_selected_features with "
                            f"packed_row_dir={val_candidate}."
                        )
                    val_packed_row_dir = val_candidate
                    val_selection_dir = val_sel
                if splice_cf_for_discovery:
                    selection_dir = validate_complete_selection_dir(
                        packed_row_dir=packed_row_dir,
                        selection_hash=selection_hash,
                        configured_mask=configured_canonical,
                        label="splice_counterfactual train",
                        feature_clusters=feature_cluster_state,
                        delta_config=feature_delta_raw_for_discovery,
                        splice_feature_dir=Path(feature_dir_str),
                        boundary_fingerprint=boundary_fingerprint,
                        splice_batch_size=selection_splice_batch_size,
                        remote_materializer=train_remote_materializer,
                        packed_materializer=packed_identity_materializer,
                        remote_packed_streaming=(packed_identity_materializer is not None),
                    )
                logger.info(f"Found selection shards: {selection_dir}")
            else:
                if splice_cf_for_discovery:
                    raise FileNotFoundError(
                        "feature_delta.mode=splice_counterfactual with "
                        "use_precomputed_batches=true requires TRAIN selection "
                        f"shards at {sel_candidate}; run "
                        "precompute_selected_features with "
                        f"packed_row_dir={candidate}."
                    )
                if discovery_grad_spec is not None:
                    raise GradientRankingUnsupportedError(
                        f"feature_selection.gradient requires precomputed "
                        f"selection shards, but none exist at {sel_candidate}. "
                        f"Run precompute_selected_features with this config "
                        f"(after precompute_gradient_attribution)."
                    )
                logger.info(f"No selection shards at {sel_candidate}, using base features")
        else:
            if splice_cf_for_discovery:
                raise FileNotFoundError(
                    "feature_delta.mode=splice_counterfactual with "
                    "use_precomputed_batches=true requires precomputed TRAIN "
                    f"packed rows at {candidate}; run precompute_packed_rows "
                    "for the derived feature directory."
                )
            if discovery_grad_spec is not None:
                raise GradientRankingUnsupportedError(
                    f"feature_selection.gradient requires precomputed packed "
                    f"rows, but none exist at {candidate}."
                )
            logger.info(f"No precomputed packed rows at {candidate}")

    # Static clustering is an offline preselection transform. It has no fused
    # or live fallback: require a complete, identity-matched train artifact at
    # startup even for ordinary (delta-free) training.
    if feature_cluster_state is not None:
        if packed_row_dir is None or selection_hash is None:
            raise FileNotFoundError(
                "feature_clusters requires precomputed TRAIN packed rows and "
                "selection shards for the exact mask/cluster/config identity; "
                "run precompute_packed_rows and precompute_selected_features "
                "with this training config"
            )
        selection_dir = validate_complete_selection_dir(
            packed_row_dir=packed_row_dir,
            selection_hash=selection_hash,
            configured_mask=canonical_dict(discovery_feature_mask),
            label="static feature clusters train",
            feature_clusters=feature_cluster_state,
            delta_config=feature_delta_raw_for_discovery,
            splice_feature_dir=(Path(feature_dir_str) if splice_cf_for_discovery else None),
            boundary_fingerprint=(boundary_fingerprint if splice_cf_for_discovery else None),
            splice_batch_size=selection_splice_batch_size,
            remote_materializer=_remote_materializer(
                packed_dir=packed_row_dir,
                selection_checksum=selection_hash,
                label="remote train selection",
                train_packed_materializer=packed_identity_materializer,
            ),
            packed_materializer=packed_identity_materializer,
            remote_packed_streaming=(packed_identity_materializer is not None),
        )

    # Route dual validation. splice_counterfactual runs BOTH val passes:
    # packed when use_precomputed_batches=true, live otherwise.
    dual_mode = _dual_validation_mode(
        original_val_feature_dir=resolved_original_val_feature_dir,
        splice_cf_enabled=_splice_counterfactual_enabled(cfg),
        use_precomputed=use_precomputed,
    )
    # Any non-none dual mode implies original_val_feature_dir is not None;
    # the explicit `is not None` keeps the type-narrowing the old guard provided
    # for the untouched #606 packed body below.
    if dual_mode is DualValidationMode.packed and resolved_original_val_feature_dir is not None:
        if not use_precomputed:
            raise ValueError(
                "Spliced dual validation requires precomputed packed rows for "
                "both the derived and original/base feature directories. Set "
                "use_precomputed_batches=true or use a non-spliced feature_dir."
            )
        if feature_delta_raw_for_discovery is not None and not splice_cf_for_discovery:
            raise ValueError(
                "Spliced dual validation requires precomputed packed rows, but "
                "feature_delta is live-dataloader-only in conditional_encdec_train. "
                "Set feature_delta=null for feature-shard splice training."
            )
        if packed_row_dir is None or selection_hash is None:
            raise FileNotFoundError(
                "Spliced dual validation requires precomputed packed rows for "
                f"the active derived feature_dir ({feature_dir_str}). Run "
                "precompute_packed_rows for the derived feature directory and "
                "for original_val_feature_dir before training."
            )
        active_val_packed_row_dir = val_packed_row_dir or discover_val_packed_row_dir(
            packed_row_dir
        )
        if active_val_packed_row_dir is None:
            expected = packed_row_dir.parent / f"val_{packed_row_dir.name}"
            raise FileNotFoundError(
                "Spliced dual validation requires the derived/spliced val "
                f"packed rows at {expected}; run precompute_packed_rows for "
                "the derived feature directory."
            )
        val_packed_row_dir = active_val_packed_row_dir
        if remote_training is not None:
            assert percentile_checksum is not None
            assert tokenizer_checksum is not None
            _assert_remote_packed_row_manifest_identity(
                val_packed_row_dir,
                percentile_checksum=percentile_checksum,
                tokenizer_checksum=tokenizer_checksum,
                label="remote derived validation",
            )
        configured_canonical = canonical_dict(discovery_feature_mask)
        if splice_cf_for_discovery:
            derived_val_remote_materializer = _remote_materializer(
                packed_dir=val_packed_row_dir,
                selection_checksum=selection_hash,
                label="remote derived validation selection",
            )
            val_selection_dir = validate_complete_selection_dir(
                packed_row_dir=val_packed_row_dir,
                selection_hash=selection_hash,
                configured_mask=configured_canonical,
                label="splice_counterfactual derived val_spliced",
                feature_clusters=feature_cluster_state,
                delta_config=feature_delta_raw_for_discovery,
                splice_feature_dir=Path(feature_dir_str),
                boundary_fingerprint=boundary_fingerprint,
                splice_batch_size=selection_splice_batch_size,
                remote_materializer=derived_val_remote_materializer,
                remote_packed_streaming=(derived_val_remote_materializer is not None),
            )

        original_pct_checksum, original_tok_checksum = _load_packed_row_checksums(
            resolved_original_val_feature_dir,
            fallback_tokenizer_checksum=tokenizer_checksum,
        )
        assert percentile_checksum is not None
        assert tokenizer_checksum is not None
        if original_pct_checksum != percentile_checksum:
            raise ValueError(
                "original_val_feature_dir percentile checksum does not match "
                f"the active derived feature_dir: {original_pct_checksum[:16]} "
                f"!= {percentile_checksum[:16]}. Recompute percentiles/stats so "
                "base and spliced dirs use the same feature-percentile table."
            )
        if original_tok_checksum != tokenizer_checksum:
            raise ValueError(
                "original_val_feature_dir tokenizer checksum does not match "
                f"the active derived feature_dir: {original_tok_checksum[:16]} "
                f"!= {tokenizer_checksum[:16]}. Regenerate token-boundaries and "
                "packed rows with the current nanochat tokenizer."
            )

        original_train_packed_row_dir = _packed_row_train_dir(
            resolved_original_val_feature_dir,
            percentile_checksum=original_pct_checksum,
            tokenizer_checksum=original_tok_checksum,
            context_len=cfg.max_seq_len,
            max_train_shards=max_train_shards,
        )
        original_val_packed_row_dir = discover_val_packed_row_dir(original_train_packed_row_dir)
        if original_val_packed_row_dir is None:
            expected = (
                original_train_packed_row_dir.parent / f"val_{original_train_packed_row_dir.name}"
            )
            raise FileNotFoundError(
                "Spliced dual validation requires original/base val packed "
                f"rows at {expected}; run precompute_packed_rows for "
                f"original_val_feature_dir={resolved_original_val_feature_dir}."
            )
        if remote_training is not None:
            _assert_remote_packed_row_manifest_identity(
                original_val_packed_row_dir,
                percentile_checksum=original_pct_checksum,
                tokenizer_checksum=original_tok_checksum,
                label="remote original/base validation",
            )
        if discovery_grad_spec is not None:
            original_val_selection_dir = _assert_gradient_val_selection_dir(
                val_packed_row_dir=original_val_packed_row_dir,
                selection_hash=selection_hash,
                configured_gradient=configured_gradient,
                label="ORIGINAL VAL",
            )
        if splice_cf_for_discovery:
            base_val_feature_selection_raw = dict(feature_selection_raw)
            base_val_feature_selection_raw["delta_weights"] = None
            negative_selection_raw = _optional_cfg_dict(cfg, "negative_selection")
            base_val_selection_hash = compute_selection_hash(
                positive_selection=base_val_feature_selection_raw,
                negative_selection=negative_selection_raw,
                delta_config=None,
                num_buckets=cfg.num_buckets,
                seed=cfg.get("seed", 42),
                ddp_rank=0,
                feature_mask=configured_canonical,
                feature_clusters=canonical_feature_clusters_descriptor(resolved_feature_clusters),
                splice_batch_size=cfg.get("selection_splice_batch_size"),
            )
            original_val_selection_dir = validate_complete_selection_dir(
                packed_row_dir=original_val_packed_row_dir,
                selection_hash=base_val_selection_hash,
                configured_mask=configured_canonical,
                label="splice_counterfactual original/base val",
                feature_clusters=feature_cluster_state,
                delta_config=None,
                splice_batch_size=selection_splice_batch_size,
                remote_materializer=_remote_materializer(
                    packed_dir=original_val_packed_row_dir,
                    selection_checksum=base_val_selection_hash,
                    label="remote base validation selection",
                ),
            )
        original_val_selection_dir = _ensure_clustered_original_val_selection_dir(
            current=original_val_selection_dir,
            original_val_packed_row_dir=original_val_packed_row_dir,
            selection_hash=selection_hash,
            configured_mask=configured_canonical,
            feature_clusters=feature_cluster_state,
            splice_counterfactual=splice_cf_for_discovery,
        )
        logger.info(
            "Spliced dual validation enabled: "
            f"val/* uses {original_val_packed_row_dir}; "
            f"val_spliced/* uses {val_packed_row_dir}"
        )
    elif dual_mode is DualValidationMode.live:
        # Live dual validation (issue #596, splice_counterfactual). Both val
        # passes use live loaders (no packed rows / selection dirs). Reuse the
        # #606 base-stats + checksum-equality gates (same raise wording), then
        # build the base-dir FeatureIndex the delta-free val/* pass reads.
        assert resolved_original_val_feature_dir is not None  # guaranteed by dual_mode
        original_pct_checksum, original_tok_checksum = _load_packed_row_checksums(
            resolved_original_val_feature_dir,
            fallback_tokenizer_checksum=tokenizer_checksum,
        )
        assert percentile_checksum is not None
        assert tokenizer_checksum is not None
        if original_pct_checksum != percentile_checksum:
            raise ValueError(
                "original_val_feature_dir percentile checksum does not match "
                f"the active derived feature_dir: {original_pct_checksum[:16]} "
                f"!= {percentile_checksum[:16]}. Recompute percentiles/stats so "
                "base and spliced dirs use the same feature-percentile table."
            )
        if original_tok_checksum != tokenizer_checksum:
            raise ValueError(
                "original_val_feature_dir tokenizer checksum does not match "
                f"the active derived feature_dir: {original_tok_checksum[:16]} "
                f"!= {tokenizer_checksum[:16]}. Regenerate token-boundaries and "
                "packed rows with the current nanochat tokenizer."
            )
        original_val_feature_index = FeatureIndex(
            str(resolved_original_val_feature_dir),
            percentile_checksum=original_pct_checksum,
            tokenizer_checksum=original_tok_checksum,
        )
        # Dedicated index for the val_spliced/* pass over the DERIVED dir. It must
        # NOT be shared.feature_index: the train loader's prefetch thread mutates
        # that instance's LRU-1 cache concurrently and FeatureIndex.lookup is not
        # thread-safe.
        spliced_val_feature_index = FeatureIndex(
            feature_dir_str,
            percentile_checksum=percentile_checksum,
            tokenizer_checksum=tokenizer_checksum,
        )
        # Fail fast if the base val corpus does not pair with the base feature
        # dir (else a KeyError surfaces mid-training at the first val pass). The
        # spliced pass is guarded inside the loader (sidecar n_docs == rows).
        _assert_live_val_pair_aligned(
            original_val_feature_index,
            cfg.get("original_val_data_dir", None),
            max_train_shards,
            "splice_counterfactual base validation (val/*)",
        )
        logger.info(
            "Live dual validation enabled (splice_counterfactual): val/* uses "
            f"the delta-free twin over base dir {resolved_original_val_feature_dir}; "
            f"val_spliced/* uses the counterfactual pipeline over {feature_dir_str}."
        )

    if feature_cluster_state is not None and int(cfg.get("eval_every", 0)) > 0:
        assert packed_row_dir is not None
        assert selection_hash is not None
        if val_packed_row_dir is None:
            val_packed_row_dir = discover_val_packed_row_dir(packed_row_dir)
        if val_packed_row_dir is None:
            expected = packed_row_dir.parent / f"val_{packed_row_dir.name}"
            raise FileNotFoundError(
                "feature_clusters with eval_every>0 requires precomputed VAL "
                f"packed rows at {expected}; clustered validation never falls "
                "back to live or fused selection"
            )
        if val_selection_dir is None:
            val_remote_materializer = _remote_materializer(
                packed_dir=val_packed_row_dir,
                selection_checksum=selection_hash,
                label="remote validation selection",
            )
            val_selection_dir = validate_complete_selection_dir(
                packed_row_dir=val_packed_row_dir,
                selection_hash=selection_hash,
                configured_mask=canonical_dict(discovery_feature_mask),
                label="static feature clusters val",
                feature_clusters=feature_cluster_state,
                delta_config=feature_delta_raw_for_discovery,
                splice_feature_dir=(Path(feature_dir_str) if splice_cf_for_discovery else None),
                boundary_fingerprint=(boundary_fingerprint if splice_cf_for_discovery else None),
                splice_batch_size=selection_splice_batch_size,
                remote_materializer=val_remote_materializer,
                remote_packed_streaming=(remote_packed_streaming and splice_cf_for_discovery),
            )

    # --- Compute static pad_F from selection config max_k bounds (fused path) ---
    # When selection shards are absent but the config defines max_k for positive
    # and negative sampling, we can compute a static upper bound on the feature
    # dimension. This enables dynamic=False for torch.compile.
    pad_f: int | None = None
    if selection_dir is None and cfg.get("compile_static", "auto") is not False:
        pad_f = resolve_static_pad_f_from_training_cfg(cfg)
        if pad_f is not None:
            feature_selection_raw_for_padding = _cfg_dict(
                cfg.feature_selection,
                name="feature_selection",
            )
            feature_selection_parsed = parse_feature_selection(feature_selection_raw_for_padding)
            assert isinstance(feature_selection_parsed, SamplingSelection)
            assert feature_selection_parsed.max_k is not None
            negative_selection_raw_for_padding = _optional_cfg_dict(
                cfg,
                "negative_selection",
            )
            negative_selection_parsed = parse_negative_selection(negative_selection_raw_for_padding)
            negative_max_k = 0
            if negative_selection_parsed is not None:
                assert negative_selection_parsed.max_k is not None
                negative_max_k = negative_selection_parsed.max_k
            if cfg.fp8:
                logger.info(
                    f"Static pad_F={pad_f} from config (pos_max={feature_selection_parsed.max_k}, "
                    f"neg_max={negative_max_k}, aligned to FP8_ALIGNMENT={FP8_ALIGNMENT})"
                )
            else:
                logger.info(
                    f"Static pad_F={pad_f} from config (pos_max={feature_selection_parsed.max_k}, "
                    f"neg_max={negative_max_k})"
                )

    # Compute pad_n_seg for static shapes (live path needs this from config
    # since there's no manifest; precomputed path gets it from manifest)
    live_pad_n_seg = resolve_live_pad_n_seg_from_training_cfg(cfg, pad_f=pad_f)

    remote_derived_feature_fingerprint: str | None = None
    if remote_training is not None and splice_cf_for_discovery:
        if remote_packed_streaming and packed_materializer is None:
            raise RemoteTrainingError(
                "remote train packed streaming did not produce a verified packed materializer"
            )
        required_splice_selection_dirs = {
            "train": selection_dir,
            "derived_val": val_selection_dir,
        }
        resolved_fingerprints: dict[str, str] = {}
        for label, resolved_selection_dir in required_splice_selection_dirs.items():
            if resolved_selection_dir is None:
                raise RemoteTrainingError(
                    "remote splice-counterfactual training requires train and "
                    f"derived-val selection families; missing {label}"
                )
            selection_materializer = remote_materializers.get(str(resolved_selection_dir.resolve()))
            if selection_materializer is None:
                raise RemoteTrainingError(
                    "remote splice-counterfactual training did not verify the remote "
                    f"{label} selection manifest: {resolved_selection_dir}"
                )
            splice_contract = selection_materializer.identity_contract.get("splice_counterfactual")
            fingerprint = (
                splice_contract.get("derived_feature_fingerprint")
                if isinstance(splice_contract, Mapping)
                else None
            )
            if (
                not isinstance(fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
            ):
                raise RemoteTrainingError(
                    f"remote {label} selection manifest has malformed "
                    "splice_counterfactual.derived_feature_fingerprint: "
                    f"{fingerprint!r}"
                )
            resolved_fingerprints[label] = fingerprint
        if resolved_fingerprints["train"] != resolved_fingerprints["derived_val"]:
            raise RemoteTrainingError(
                "remote train and derived-val selection manifests disagree on "
                "splice_counterfactual.derived_feature_fingerprint: "
                f"train={resolved_fingerprints['train']}, "
                f"derived_val={resolved_fingerprints['derived_val']}"
            )
        remote_derived_feature_fingerprint = resolved_fingerprints["train"]

    return PackedSelectionDiscovery(
        train_packed_dir=packed_row_dir,
        train_selection_dir=selection_dir,
        val_packed_dir=val_packed_row_dir,
        val_selection_dir=val_selection_dir,
        original_val_feature_dir=resolved_original_val_feature_dir,
        original_val_packed_dir=original_val_packed_row_dir,
        original_val_selection_dir=original_val_selection_dir,
        original_val_feature_index=original_val_feature_index,
        spliced_val_feature_index=spliced_val_feature_index,
        max_seg_size=max_seg_size,
        pad_n_seg=live_pad_n_seg,
        pad_f=pad_f,
        selection_hash=selection_hash,
        packed_materializer=packed_materializer,
        packed_identity_materializer=packed_identity_materializer,
        remote_derived_feature_fingerprint=(remote_derived_feature_fingerprint),
        remote_materializers=remote_materializers,
    )


def build_shared_state(
    cfg: DictConfig,
    *,
    gpu_index: int | None = None,
    startup_trace: StartupTrace | None = None,
) -> SharedState:
    """Encapsulate ALL one-time setup: DDP, tokenizer, features, model, eval wrappers.

    Args:
        cfg: Hydra config.
        gpu_index: If set, pin to this CUDA device (for multi-GPU config parallelism).
    """
    with _startup_phase(startup_trace, "startup_validation"):
        _reject_precompute_only_packed_row_dir(cfg)
        reject_feature_clustering_config(cfg, "conditional_encdec_train shared setup")
        _validate_feature_clusters_training_config(cfg)
        _validate_eval_config(cfg)  # Phase 2 PR-A: fail fast on missing cache
        remote_training = RemoteTrainingConfig.parse(
            _optional_cfg_dict(cfg, "remote_training")
        )
        if remote_training is not None:
            resolved_cfg_for_remote = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(resolved_cfg_for_remote, dict)
            validate_remote_training_eligibility(
                cast(Mapping[str, Any], resolved_cfg_for_remote), remote_training
            )
        remote_backend: BucketBackend | None = (
            HfBucketBackend(remote_training.bucket_id)
            if remote_training is not None
            else None
        )
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        # Encoder-decoder models have more unique parameter shapes than base GPT,
        # which causes the Muon optimizer's compiled step to recompile for each shape.
        # Raise the cache limit to accommodate all shape groups (bounded by model architecture).
        t._dynamo.config.cache_size_limit = 64

    # --- Compute init and device detection ---
    with _startup_phase(startup_trace, "device_ddp_init"):
        device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
        # Override device if gpu_index specified (config parallelism, not DDP)
        if gpu_index is not None and device_type == "cuda" and not ddp:
            device = t.device("cuda", gpu_index)
            t.cuda.set_device(device)
        master_process = ddp_rank == 0  # noqa: F841 — kept for symmetry with standalone tail
        if device_type == "cuda":
            gpu_device_name = t.cuda.get_device_name(device)
            gpu_peak_flops = get_peak_flops(gpu_device_name)
            logger.info(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
        else:
            gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

    # --- Flash Attention override ---
    with _startup_phase(startup_trace, "attention_backend"):
        if cfg.attn_impl != FlashAttnImpl.AUTO:
            fa_module._override_impl = FlashAttnImpl(cfg.attn_impl)
            logger.info(f"Attention implementation override: {cfg.attn_impl}")
        if device_type == "cuda":
            admit_project_attention()

        # Flash Attention status
        active_impl = fa_module._active_impl()
        if active_impl == FlashAttnImpl.FA4:
            backend = "FLASH (FA4)" if fa_module.HAS_FA4_BACKEND else "Triton"
            logger.info(f"Using FlexAttention ({backend} backend)")
        elif active_impl == FlashAttnImpl.FA3:
            logger.info("Using Flash Attention 3 (Hopper GPU detected)")
        else:
            logger.warning("Flash Attention not available, using PyTorch SDPA fallback")
            logger.warning("Training will be less efficient without flash attention")
            if cfg.window_pattern != "L":
                logger.warning(
                    f"SDPA has limited support for sliding window attention (window_pattern='{cfg.window_pattern}')."
                )

    # --- Tokenizer ---
    with _startup_phase(startup_trace, "tokenizer"):
        logger.info("Loading tokenizer...")
        tokenizer = get_tokenizer()
        token_bytes = get_token_bytes(device=device)
        base_vocab = tokenizer.get_vocab_size()
        logger.info(f"Base vocab size: {base_vocab:,}")

    # --- Feature index ---
    with _startup_phase(startup_trace, "feature_source_resolution"):
        if cfg.feature_dir is not None:
            feature_dir = cfg.feature_dir
        else:
            feature_dir = str(
                resolve_output_dir(
                    output_dir=None,
                    feature_root=cfg.feature_root,
                    segment_size=cfg.segment_size,
                    layer=cfg.sae.layer,
                    activation_type=ActivationType(cfg.sae.activation_type),
                    pooling=PoolingMode(cfg.pooling),
                    width=SAEWidth(cfg.sae.width),
                    model=cfg.sae.model,
                    dataset_phase=cfg.get("dataset_phase", None),
                    ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
                )
            )
            logger.info(f"Auto-resolved feature_dir: {feature_dir}")

        remote_packed_streaming = (
            remote_training is not None and remote_training.train_packed is not None
        )
        if remote_packed_streaming and cfg.get("verify_features", False):
            raise RemoteTrainingError(
                "remote packed streaming does not stage the complete raw feature "
                "family required by verify_features=true; keep verify_features=false "
                "and use the certified sparse inline-eval source contract"
            )
        if cfg.get("verify_features", False):
            verify_feature_dir(feature_dir)
            logger.info("Feature directory verification passed")
        else:
            logger.info("Skipping feature directory verification (verify_features=false)")

        # D8 (issue #596): splice_counterfactual auto-derives cfg.data_dir from the
        # derived feature_dir's own _parquets. MUST run before the val-shard pin
        # block (reads cfg.data_dir), _resolve_original_val_feature_dir, and every
        # other cfg.data_dir consumer downstream.
        _resolve_splice_counterfactual_data_dir(cfg, feature_dir)

    with _startup_phase(startup_trace, "statistics_checksums"):
        logger.info(f"Scanning feature shards in {feature_dir}...")
        feature_prefetch = cfg.get("feature_prefetch", False)
        stats_path = find_statistics_path(feature_dir)
        assert stats_path is not None, (
            f"feature_statistics not found in {feature_dir}. "
            f"Run feature extraction with statistics computation first."
        )
        percentile_checksum = load_percentile_checksum(stats_path)
        if percentile_checksum is not None:
            logger.info(f"Found precomputed percentile checksum: {percentile_checksum[:16]}...")
        tokenizer_checksum = load_tokenizer_checksum(stats_path)
        if tokenizer_checksum is None:
            tokenizer_checksum = compute_tokenizer_checksum(tokenizer)
            logger.info(f"Computed tokenizer checksum: {tokenizer_checksum[:16]}...")
        else:
            logger.info(f"Found precomputed tokenizer checksum: {tokenizer_checksum[:16]}...")

        # Statistics are the complete-family metadata authority in remote packed
        # mode, where no global raw FeatureIndex is constructed.
        logger.info(f"Loading feature statistics from {stats_path}...")
        feature_stats = load_statistics(stats_path)
        per_feature_stats = feature_stats.per_feature_stats
        n_sae_features = PositiveInt.parse(feature_stats.metadata.n_sae_features)

    feature_index: FeatureIndex | None
    remote_inline_eval_source: CertifiedRemoteInlineEvalSource | None = None
    if remote_packed_streaming:
        extraction_segment_size = PositiveInt.parse(feature_stats.metadata.segment_size)
        if extraction_segment_size != cfg.segment_size:
            raise RemoteTrainingError(
                "remote packed feature statistics segment_size mismatch: "
                f"statistics={extraction_segment_size}, config={cfg.segment_size}"
            )
        feature_index = None
        _startup_skip(
            startup_trace,
            "feature_index_overlays",
            fields={"source": "remote_certified"},
        )
        logger.info(
            "Remote packed streaming: skipping global raw FeatureIndex; "
            "packed/selection manifests authenticate training and validation"
        )
        if int(cfg.get("steering_eval_every", 0)) > 0:
            assert remote_training is not None
            assert remote_backend is not None
            if tokenizer_checksum is None:
                raise RemoteTrainingError(
                    "remote clustered inline steering requires a tokenizer checksum"
                )
            remote_inline_eval_source = CertifiedRemoteInlineEvalSource.discover(
                backend=remote_backend,
                remote=remote_training,
                feature_root=Path(str(cfg.feature_root)),
                feature_dir=Path(feature_dir),
                data_dir=cfg.get("data_dir"),
                tokenizer_checksum=tokenizer_checksum,
                segment_size=extraction_segment_size,
            )
    else:
        feature_index_percentile_checksum = _feature_index_percentile_checksum_for_training(
            cfg,
            remote_training,
            percentile_checksum,
        )
        if percentile_checksum is not None and feature_index_percentile_checksum is None:
            logger.info(
                "Remote static clustered packed mode: percentile identity will be "
                "validated from exact packed-row manifests; the raw FeatureIndex "
                "percentile overlay is not required"
            )
        feature_index_tokenizer_checksum = _feature_index_tokenizer_checksum_for_training(
            cfg,
            tokenizer_checksum,
        )
        if tokenizer_checksum is not None and feature_index_tokenizer_checksum is None:
            logger.info(
                "Static clustered packed mode: tokenizer identity will be validated "
                "from packed-row/selection manifests; the raw FeatureIndex token "
                "boundary overlay is not required"
            )
        with _startup_phase(startup_trace, "feature_index_overlays"):
            feature_index = FeatureIndex(
                feature_dir,
                prefetch=feature_prefetch,
                percentile_checksum=feature_index_percentile_checksum,
                tokenizer_checksum=feature_index_tokenizer_checksum,
                required_token_boundary_stems=(
                    _feature_index_required_token_boundary_stems_for_training(cfg)
                ),
            )
        assert feature_index.segment_size is not None, "No feature shards found"
        extraction_segment_size = PositiveInt.parse(feature_index.segment_size)
        assert extraction_segment_size == cfg.segment_size, (
            f"segment_size mismatch: config has {cfg.segment_size} but "
            f"extraction features used {extraction_segment_size}"
        )
        if feature_prefetch:
            logger.info("Feature shard prefetching enabled")
    doc_prefetch_depth = cfg.get("doc_prefetch_depth", 0)
    if doc_prefetch_depth > 0:
        logger.info(f"Document batch prefetching enabled (depth={doc_prefetch_depth})")

    with _startup_phase(startup_trace, "boundaries_and_feature_identity"):
        # Pin the validation shard so it doesn't evict training shards from the LRU cache
        max_train_shards_cfg = cfg.get("max_train_shards", None)
        if max_train_shards_cfg is not None and feature_index is not None:
            all_parquet_paths = list_parquet_files(cfg.get("data_dir", None))
            limited_parquets = all_parquet_paths[: int(max_train_shards_cfg)]
            if limited_parquets:
                val_stem = Path(limited_parquets[-1]).stem
                if val_stem in feature_index._shard_paths:
                    feature_index.pin_shard(val_stem)
                    logger.info(f"Pinned validation shard: {val_stem}")

        # --- Char-level segment assignment (always) ---
        if feature_index is not None:
            assert feature_index.has_char_boundaries, (
                "Character-level segment assignment requires "
                "segment_char_starts/segment_char_ends in feature shards. Run the "
                "migration script to add character boundaries."
            )
        elif remote_inline_eval_source is not None:
            assert remote_inline_eval_source.feature_index.has_char_boundaries
        logger.info(
            f"Using character-level segment assignment (segment_size={extraction_segment_size})"
        )

        # --- Auto activation-frequency blacklist (producer for feature_mask) ---
        # Reads feature_statistics (already loaded above as feature_stats) and
        # mutates cfg.feature_mask BEFORE the first resolve_feature_mask below so
        # the discovery selection-hash, the checkpoint mask leaf, and the feature
        # pipeline all observe it consistently (train_one_config reads this same
        # cfg). No-op (cfg byte-identically untouched ⇒ PR 471 I1/I2/I5 hold)
        # when both feature_mask_freq_min/max are null, when the computed blacklist is
        # empty, or when an explicit feature_mask is already configured.
        maybe_apply_frequency_blacklist(
            cfg,
            feature_dir,
            n_sae_features,
            ddp_rank=ddp_rank,
            barrier_fn=ddp_barrier,
        )

        # --- Boundaries for base config ---
        negative_selection_raw = _optional_cfg_dict(cfg, "negative_selection")
        negative_selection = parse_negative_selection(negative_selection_raw)
        effective_buckets = (
            cfg.num_buckets - 1 if negative_selection is not None else cfg.num_buckets
        )
        if effective_buckets == 1:
            boundaries = t.zeros(n_sae_features, 0, dtype=t.float64)
        else:
            # Use the threshold percentile from the base config for boundary building
            feature_selection_raw = _cfg_dict(cfg.feature_selection, name="feature_selection")
            feature_selection = parse_feature_selection(feature_selection_raw)
            if isinstance(feature_selection, ThresholdSelection):
                threshold_cfg = ThresholdConfig(
                    feature_selection.threshold_type, feature_selection.threshold
                )
                threshold_pct = extract_threshold_percentile(threshold_cfg)
            else:
                threshold_pct = None
            boundaries = build_boundaries_from_stats(
                per_feature_stats,
                n_sae_features,
                effective_buckets,
                threshold_percentile=threshold_pct,
            )

        logger.info(f"Vocab size (no extension needed): {base_vocab:,}")

    # --- Encoder feature-dictionary compaction (issue #695) ---
    # The mask must be resolved BEFORE the model is built, because it sizes the
    # encoder's feature-indexed tables. The ordering already works:
    # maybe_apply_frequency_blacklist (above) has already mutated cfg.feature_mask.
    #
    # The vocab is a PURE function of (mask, clusters, n_sae_features), so the existing
    # bl_/wl_<16hex> and fc_<64hex> checkpoint leaves already pin it — no new hash
    # component (I1 holds).
    shared_feature_mask = resolve_feature_mask(cfg, n_sae_features)
    resolved_feature_clusters = resolve_feature_clusters(cfg, n_sae_features)
    feature_clusters = (
        FeatureClusterConsolidation.build(
            resolved_feature_clusters,
            shared_feature_mask,
        )
        if resolved_feature_clusters is not None
        else None
    )
    shared_feature_vocab = effective_feature_vocab(
        feature_clusters if feature_clusters is not None else shared_feature_mask,
        n_sae_features,
    )
    fp_target_layout = (
        FeaturePredictionTargetLayout.build(
            n_sae_features=n_sae_features,
            feature_mask=shared_feature_mask,
            feature_clusters=feature_clusters,
            feature_vocab=shared_feature_vocab,
        )
        if cfg.get("feature_prediction", None) is not None
        else None
    )

    with _startup_phase(startup_trace, "model_materialization"):
        logger.info("Building model on meta device...")
        model = build_encdec_meta_model_from_training_cfg(
            cfg,
            base_vocab=base_vocab,
            n_sae_features=n_sae_features,
            feature_vocab=shared_feature_vocab,
        )
        if fp_target_layout is not None and (
            model.feature_prediction_width != fp_target_layout.output_width
        ):
            raise FeatureVocabError(
                "feature-prediction head width does not match the derived target "
                f"layout: head={model.feature_prediction_width}, "
                f"targets={fp_target_layout.output_width}"
            )
        logger.info(
            format_compaction_summary(
                model,
                no_vocab_reason=no_compaction_reason(
                    shared_feature_mask,
                    n_sae_features,
                    feature_clusters=feature_clusters,
                ),
                consolidated_away=(
                    int(feature_clusters.consolidated_away.sum().item())
                    if feature_clusters is not None
                    else None
                ),
            )
        )
        model_config = model.dec_config
        model_config_kwargs = asdict(model_config)
        logger.info(f"Decoder config:\n{json.dumps(model_config_kwargs, indent=2)}")
        enc_fe = model.encoder.config
        logger.info(
            f"Encoder config (resolved): n_embd={enc_fe.n_embd}, n_layer={enc_fe.n_layer}, "
            f"n_head={enc_fe.n_head}, n_kv_head={enc_fe.n_kv_head}, head_dim={enc_fe.head_dim}"
        )
        logger.info(f"Materializing model on {device}...")
        model.to_empty(device=device)
        logger.info("Initializing weights...")
        model.init_weights()
        logger.info("Model initialized.")

    # --- FP8 training ---
    if cfg.fp8:
        if device_type != "cuda":
            logger.warning("FP8 training requires CUDA, ignoring --fp8 flag")
        else:
            apply_fp8_training(model, recipe=cfg.fp8_recipe)
            encoder_fp8 = sum(
                1
                for n, m in model.named_modules()
                if "Float8" in type(m).__name__ and n.startswith("encoder.")
            )
            decoder_fp8 = sum(
                1
                for n, m in model.named_modules()
                if "Float8" in type(m).__name__ and n.startswith("transformer.")
            )
            xattn_fp8 = sum(
                1
                for n, m in model.named_modules()
                if "Float8" in type(m).__name__ and n.startswith("cross_attn_layers.")
            )
            lm_head_fp8 = 1 if "Float8" in type(model.lm_head).__name__ else 0
            logger.info(
                f"FP8 breakdown: encoder={encoder_fp8}, decoder={decoder_fp8}, xattn={xattn_fp8}, lm_head={lm_head_fp8}"
            )
            assert encoder_fp8 > 0, (
                "No encoder layers converted to FP8. Encoder dims may be too small. "
                "Check that encoder n_embd >= 128 and is divisible by 16."
            )

    # --- Discover precomputed packed rows and selection shards ---
    feature_delta_raw_for_discovery = _optional_cfg_dict(cfg, "feature_delta")
    feature_delta_for_discovery = parse_feature_delta(feature_delta_raw_for_discovery)
    discovery_boundary_fingerprint = (
        tensor_content_fingerprint(boundaries)
        if feature_delta_for_discovery is not None
        and feature_delta_for_discovery.mode is DeltaMode.splice_counterfactual
        else None
    )
    original_val_override = cfg.get("original_val_feature_dir", None)
    with _startup_phase(startup_trace, "packed_selection_artifacts"):
        discovery = discover_packed_selection(
            feature_dir=Path(feature_dir),
            original_val_feature_dir=(
                Path(str(original_val_override)) if original_val_override is not None else None
            ),
            feature_selection=_cfg_dict(
                cfg.feature_selection,
                name="feature_selection",
            ),
            negative_selection=_optional_cfg_dict(cfg, "negative_selection"),
            feature_delta=feature_delta_raw_for_discovery,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.get("seed", 42)),
            feature_mask=canonical_dict(shared_feature_mask),
            feature_clusters=_optional_cfg_dict(cfg, "feature_clusters"),
            n_sae_features=n_sae_features,
            use_precomputed_batches=bool(cfg.get("use_precomputed_batches", True)),
            percentile_checksum=percentile_checksum,
            tokenizer_checksum=tokenizer_checksum,
            context_len=int(cfg.max_seq_len),
            segment_size=int(cfg.segment_size),
            max_train_shards=cfg.get("max_train_shards", None),
            compile_static=cfg.get("compile_static", "auto"),
            fp8=bool(cfg.fp8),
            eval_every=int(cfg.get("eval_every", 0)),
            original_val_data_dir=cfg.get("original_val_data_dir", None),
            boundary_fingerprint=discovery_boundary_fingerprint,
            selection_splice_batch_size=cfg.get("selection_splice_batch_size"),
            remote_training=remote_training,
            feature_root=Path(str(cfg.feature_root)),
            remote_backend=remote_backend,
        )
    packed_row_dir = discovery.train_packed_dir
    selection_dir = discovery.train_selection_dir
    val_packed_row_dir = discovery.val_packed_dir
    val_selection_dir = discovery.val_selection_dir
    original_val_feature_dir = discovery.original_val_feature_dir
    original_val_packed_row_dir = discovery.original_val_packed_dir
    original_val_selection_dir = discovery.original_val_selection_dir
    original_val_feature_index = discovery.original_val_feature_index
    spliced_val_feature_index = discovery.spliced_val_feature_index
    max_seg_size = discovery.max_seg_size
    pad_F = discovery.pad_f
    live_pad_n_seg = discovery.pad_n_seg
    packed_materializer = discovery.packed_materializer
    packed_identity_materializer = discovery.packed_identity_materializer
    remote_derived_feature_fingerprint = discovery.remote_derived_feature_fingerprint
    remote_run_fingerprint: str | None = None
    if remote_training is not None:
        required_dirs = {
            "train": selection_dir,
            "derived_val": val_selection_dir,
            "base_val": original_val_selection_dir,
        }
        missing_dirs = [label for label, path in required_dirs.items() if path is None]
        if missing_dirs:
            raise RemoteTrainingError(
                "remote_training requires all three local packed/selection families; "
                f"missing selection families={missing_dirs}"
            )
        contracts: dict[str, Mapping[str, Any]] = {}
        for label, selection_path in required_dirs.items():
            assert selection_path is not None
            materializer = discovery.remote_materializers.get(str(selection_path.resolve()))
            if materializer is None:
                raise RemoteTrainingError(
                    f"remote_training did not verify a remote manifest for {label}: "
                    f"{selection_path}"
                )
            contracts[label] = materializer.identity_contract
        if remote_derived_feature_fingerprint is not None:
            if remote_training.train_packed is not None and packed_materializer is None:
                raise RemoteTrainingError(
                    "remote_training.train_packed did not verify the train packed-row inventory"
                )
            startup_fingerprints: dict[str, str] = {}
            for label in ("train", "derived_val"):
                splice_contract = contracts[label].get("splice_counterfactual")
                fingerprint = (
                    splice_contract.get("derived_feature_fingerprint")
                    if isinstance(splice_contract, Mapping)
                    else None
                )
                if (
                    not isinstance(fingerprint, str)
                    or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
                ):
                    raise RemoteTrainingError(
                        f"remote {label} startup contract has malformed "
                        "derived_feature_fingerprint: "
                        f"{fingerprint!r}"
                    )
                startup_fingerprints[label] = fingerprint
            if startup_fingerprints["train"] != startup_fingerprints["derived_val"]:
                raise RemoteTrainingError(
                    "remote train and derived-val manifests disagree on "
                    "derived_feature_fingerprint at startup"
                )
            if remote_derived_feature_fingerprint != startup_fingerprints["train"]:
                raise RemoteTrainingError(
                    "remote derived-feature fingerprint changed between "
                    "discovery and startup contract assembly"
                )
            if remote_inline_eval_source is not None:
                remote_inline_eval_source.assert_derived_feature_fingerprint(
                    startup_fingerprints["train"]
                )
        resolved_semantic_cfg = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved_semantic_cfg, dict)
        remote_run_fingerprint = build_remote_semantic_fingerprint_v2(
            resolved_training_config=cast(Mapping[str, Any], resolved_semantic_cfg),
            world_size=ddp_world_size,
            selection_contracts=contracts,
        )
        logger.info(
            "remote_training startup contract verified: "
            f"semantic_fingerprint_v2={remote_run_fingerprint} "
            f"checkpoint_lineage={remote_run_fingerprint} "
            "compatibility_mode=direct_v2"
        )

    fp_target_source_certificate_digest: str | None = None
    if fp_target_layout is not None and fp_target_layout.is_compact:
        if packed_row_dir is None:
            raise ValueError(
                "compact feature prediction requires precomputed packed rows. "
                "Run scripts.preprocess.precompute_packed_rows and set "
                "use_precomputed_batches=true."
            )
        if packed_identity_materializer is not None:
            _, fp_target_source_certificate_digest = validate_fp_target_source_certificate_remote(
                packed_row_dir,
                n_sae_features=n_sae_features,
                inventory=packed_identity_materializer.inventory,
            )
        else:
            _, fp_target_source_certificate_digest = validate_fp_target_source_certificate(
                packed_row_dir,
                n_sae_features=n_sae_features,
            )

    # --- Compile ---
    # The packed encoder and cross-attention own separate dynamic compile
    # boundaries, so their data-dependent token axis does not make the decoder
    # graph dynamic. Static outer shapes are available when selection shards
    # exist or the fused path resolved a fixed feature extent.
    static_shapes_available = (selection_dir is not None) or (pad_F is not None)
    orig_model = model
    if cfg.compile:
        with _startup_phase(startup_trace, "compile_wrapper"):
            compile_static = cfg.get("compile_static", "auto")
            dynamic = resolve_compile_dynamic(
                compile_static,
                mode="pretrain",
                static_shapes_available=static_shapes_available,
            )
            logger.info(
                f"Compiling model with torch.compile(dynamic={dynamic})... "
                f"[compile_static={compile_static}]"
            )
            model = t.compile(model, dynamic=dynamic)
            logger.info("Model compiled.")
    else:
        _startup_skip(
            startup_trace,
            "compile_wrapper",
            fields={"configured": False},
        )

    is_compiled = isinstance(model, t._dynamo.eval_frame.OptimizedModule)
    logger.info(
        f"[COMPILE DIAGNOSTIC] cfg.compile={cfg.compile} | model is OptimizedModule: {is_compiled} | model type: {type(model).__name__}"
    )

    # --- Cached context provider for full-context CORE eval and sample gen ---
    conditioning_cfg = cfg.get("conditioning", None)
    needs_clustered_context_cache = (
        feature_clusters is not None
        and conditioning_cfg is not None
        and conditioning_cfg.get("context_cache_dir", None) is not None
    )
    original_stats_path = (
        find_statistics_path(original_val_feature_dir)
        if needs_clustered_context_cache and original_val_feature_dir is not None
        else None
    )
    with _startup_phase(startup_trace, "eval_contexts"):
        cached_context_provider = _build_cached_context_provider(
            cfg,
            stats_path=Path(stats_path),
            original_stats_path=original_stats_path,
            n_sae_features=n_sae_features,
            num_buckets=int(cfg.num_buckets),
            feature_mask=shared_feature_mask,
            feature_clusters=feature_clusters,
            feature_vocab=shared_feature_vocab,
        )

    # --- Scaling law constants ---
    def get_scaling_params(m):
        params_counts = m.num_scaling_params()
        scaling_params = (
            params_counts["transformer_matrices"]
            + params_counts["lm_head"]
            + params_counts["wte"]
            + params_counts["value_embeds"]
            + params_counts["scalars"]
        )
        return scaling_params

    # Scaling-law reference. Built WITH the vocab for consistency; it makes no
    # difference to the number that is actually used, because get_scaling_params sums
    # only decoder tensors (transformer_matrices, lm_head, wte, decoder value_embeds,
    # scalars) and EXCLUDES the encoder entirely. Compaction therefore does not change
    # the training-token horizon or the batch-size/LR schedule.
    d12_ref = build_encdec_meta_model_from_training_cfg(
        cfg,
        base_vocab=base_vocab,
        n_sae_features=n_sae_features,
        depth=12,
        feature_vocab=shared_feature_vocab,
    )
    d_ref = cfg.target_param_data_ratio * get_scaling_params(d12_ref)
    b_ref = 2**19

    # --- Steering eval (constant across configs) ---
    steering_eval_cfg: SteeringEvalConfig | None = None
    steering_probe_model: ProbeModel | None = None
    if cfg.steering_eval_every > 0:
        steering_eval_cfg = SteeringEvalConfig(
            n_eval_docs=cfg.steering_eval.n_eval_docs,
            samples_per_condition=cfg.steering_eval.samples_per_condition,
            temperature=cfg.steering_eval.temperature,
            top_k_sampling=cfg.steering_eval.top_k_sampling,
            seed=cfg.steering_eval.seed,
        )
        probe_device = "cpu" if cfg.get("offload_steering_eval_feature_model", False) else device
        steering_probe_model = _load_steering_probe_model(cfg, probe_device)
        logger.info(f"Steering eval probe model loaded on {probe_device}")

    return SharedState(
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        base_vocab=base_vocab,
        feature_index=feature_index,
        feature_stats=feature_stats,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        resolved_feature_clusters=resolved_feature_clusters,
        feature_clusters=feature_clusters,
        percentile_checksum=percentile_checksum,
        tokenizer_checksum=tokenizer_checksum,
        model=model,
        orig_model=orig_model,
        cached_context_provider=cached_context_provider,
        device=device,
        device_type=device_type,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        extraction_segment_size=extraction_segment_size,
        cfg=cfg,
        gpu_peak_flops=gpu_peak_flops,
        boundaries=boundaries,
        d_ref=d_ref,
        b_ref=b_ref,
        steering_eval_cfg=steering_eval_cfg,
        steering_probe_model=steering_probe_model,
        feature_dir=feature_dir,
        packed_row_dir=packed_row_dir,
        selection_dir=selection_dir,
        original_val_feature_dir=original_val_feature_dir,
        original_val_packed_row_dir=original_val_packed_row_dir,
        original_val_selection_dir=original_val_selection_dir,
        original_val_feature_index=original_val_feature_index,
        spliced_val_feature_index=spliced_val_feature_index,
        val_packed_row_dir=val_packed_row_dir,
        val_selection_dir=val_selection_dir,
        max_seg_size=max_seg_size,
        pad_F=pad_F,
        live_pad_n_seg=live_pad_n_seg,
        fp_target_layout=fp_target_layout,
        fp_target_source_certificate_digest=(fp_target_source_certificate_digest),
        remote_training=remote_training,
        remote_backend=remote_backend,
        packed_materializer=packed_materializer,
        packed_identity_materializer=packed_identity_materializer,
        remote_derived_feature_fingerprint=(remote_derived_feature_fingerprint),
        remote_inline_eval_source=remote_inline_eval_source,
        remote_materializers=discovery.remote_materializers,
        remote_run_fingerprint=remote_run_fingerprint,
        remote_resolved_training_config=(
            cast(Mapping[str, Any], resolved_semantic_cfg)
            if remote_run_fingerprint is not None
            else None
        ),
        remote_selection_contracts=(contracts if remote_run_fingerprint is not None else None),
    )


# ---------------------------------------------------------------------------
# _maybe_run_inline_steering_eval(): per-step trigger for inline steering eval
# ---------------------------------------------------------------------------


def _maybe_run_inline_steering_eval(
    *,
    step: int,
    last_step: bool,
    cfg: DictConfig,
    shared: "SharedState",
    model,  # possibly torch.compile-wrapped, for .eval()/.train() toggling
    orig_model,  # raw EncoderDecoderGPT, for evaluate_steering_inline + disable_fp8
    feature_dir,
    steering_threshold_cfg,
    steering_precomputed_thresholds,
    steering_percentile_table,
    tokenizer,
    device,
    device_type: str,
    boundaries,
    ddp_rank: int,
    ddp_world_size: int,
    master_process: bool,
    wandb_run,
    inline_eval_cache: InlineEvalCache | None,
    feature_mask=None,
) -> InlineEvalCache | None:
    """Run one steering-eval trigger check. Builds the cache on first fire.

    Returns the (possibly-updated) ``inline_eval_cache``, which the caller
    threads back into its loop variable. When the trigger does not fire,
    returns the input unchanged.

    """
    # Trigger check — same predicate as the existing in-loop block
    if not (
        cfg.steering_eval_every > 0
        and shared.steering_eval_cfg is not None
        and (last_step or (step > 0 and step % cfg.steering_eval_every == 0))
    ):
        return inline_eval_cache

    # Build cache on first fire within this config
    if inline_eval_cache is None:
        logger.info(
            f"Building inline steering eval cache "
            f"(n_eval_docs={shared.steering_eval_cfg.n_eval_docs}, "
            f"seed={shared.steering_eval_cfg.seed})..."
        )
        cache_t0 = time.monotonic()
        inline_eval_cache = prepare_inline_eval_cache(
            feature_dir=feature_dir,
            threshold_cfg=steering_threshold_cfg,
            precomputed_thresholds=steering_precomputed_thresholds,
            data_dir=cfg.get("data_dir"),
            n_eval_docs=shared.steering_eval_cfg.n_eval_docs,
            seed=shared.steering_eval_cfg.seed,
            feature_mask=feature_mask,
        )
        logger.info(
            f"Built inline steering eval cache in "
            f"{time.monotonic() - cache_t0:.2f}s "
            f"(n_docs_selected={len(inline_eval_cache.selected_docs)})"
        )

    logger.info(f"Step {step:05d} | Starting steering evaluation...")
    model.eval()  # toggles the compiled wrapper (and underlying module)
    eval_device = t.device(device)

    offload_probe = bool(
        cfg.get("offload_steering_eval_feature_model", False)
        and shared.steering_probe_model is not None
    )
    rng_state: dict | None = None
    try:
        if offload_probe:
            assert shared.steering_probe_model is not None
            shared.steering_probe_model.to(eval_device)

        rng_state = save_rng_state(eval_device)
        with disable_fp8(orig_model):
            steering_metrics = evaluate_steering_inline(
                model=orig_model,
                feature_dir=feature_dir,
                tokenizer=tokenizer,
                device=eval_device,
                boundaries=boundaries,
                threshold_cfg=steering_threshold_cfg,
                precomputed_thresholds=steering_precomputed_thresholds,
                cfg=shared.steering_eval_cfg,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                probe_model=shared.steering_probe_model,
                percentile_table=steering_percentile_table,
                feature_index=(
                    None
                    if inline_eval_cache.clustered_payload is not None
                    else shared.feature_index
                ),
                eval_cache=inline_eval_cache,
                meta=None,
                feature_mask=feature_mask,
            )
    finally:
        if rng_state is not None:
            restore_rng_state(rng_state, eval_device)
        if offload_probe:
            assert shared.steering_probe_model is not None
            shared.steering_probe_model.to("cpu")
            if device_type == "cuda":
                t.cuda.empty_cache()
        model.train()  # restore the compiled wrapper even when eval fails

    if master_process:
        wandb_run.log(steering_metrics, step=step)
        logger.info(
            f"Step {step:05d} | Steering eval done in "
            f"{steering_metrics.get('steering/eval_time_seconds', 0):.1f}s | "
            f"roundtrip_signal={steering_metrics.get('steering/roundtrip_steering_signal_pct', 0):.4f} | "
            f"specificity={steering_metrics.get('steering/roundtrip_specificity_pct', 0):.4f}"
        )
    return inline_eval_cache


# ---------------------------------------------------------------------------
# train_one_config(): per-config training work
# ---------------------------------------------------------------------------


def _assert_not_pre_effective_compaction_checkpoint(
    configured_vocab: FeatureVocab | None,
    feature_mask: FeatureMask | None,
    feature_clusters: FeatureClusterConsolidation | None,
    model_data: Mapping[str, Any],
    *,
    checkpoint_dir: str,
) -> None:
    """Diagnose the one clustered vocabulary mismatch whose migration is knowable.

    Before effective-feature compaction, a clustered checkpoint allocated encoder rows
    from :meth:`FeatureVocab.from_mask` and therefore retained every mask-allowed member
    that consolidation made unreachable.  The cluster leaf and metadata are otherwise
    identical to a current run, so the ordinary vocabulary guard can only call this a
    generic mismatch.  Comparing the stored vocabulary to that exact legacy constructor
    identifies the format transition without guessing.

    This guard intentionally runs only after cluster metadata has matched.  A changed
    cluster definition owns the clearer cluster-content diagnostic, and an unrelated
    vocabulary or SAE-width mismatch falls through to
    :func:`assert_checkpoint_vocab_compatible`.
    """
    if feature_clusters is None:
        return
    checkpoint_n_sae, checkpoint_vocab = feature_vocab_from_state_dict(model_data)
    mask_only_vocab = FeatureVocab.from_mask(
        feature_mask,
        feature_clusters.n_sae_features,
    )
    if (
        checkpoint_n_sae == feature_clusters.n_sae_features
        and checkpoint_vocab == mask_only_vocab
        and checkpoint_vocab != configured_vocab
    ):
        checkpoint_size = checkpoint_n_sae if checkpoint_vocab is None else checkpoint_vocab.size
        effective_size = (
            feature_clusters.n_sae_features if configured_vocab is None else configured_vocab.size
        )
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} was written by pre-effective-compaction code: "
            f"its encoder has the mask-only vocabulary ({checkpoint_size} rows), while "
            f"the configured feature_mask and feature_clusters produce an effective "
            f"vocabulary of {effective_size} rows. The extra rows were allocated for "
            "consolidated-away cluster members and cannot be loaded into the current "
            "model.\n"
            "START FRESH. In principle an offline migration could gather only the "
            "effective rows from every feature-indexed parameter and its optimizer "
            "state, but no tool in this repository performs that reshape.\n"
            "(One caveat: this diagnosis compares against the CURRENTLY configured "
            "feature_mask. If the checkpoint was actually trained with a DIFFERENT "
            "mask whose allowed universe happens to coincide with this one's, the "
            "true cause is that mask change — recover the original run's config "
            "before trusting the migration reading.)"
        )


def _assert_compact_fp_checkpoint_compatible(
    layout: FeaturePredictionTargetLayout | None,
    source_certificate_digest: str | None,
    model_data: Mapping[str, Any],
    meta_data: Mapping[str, Any],
    *,
    checkpoint_dir: str,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> None:
    """Reject compact-FP shape/source drift before ``assign=True`` mutates state."""
    keys = {
        FEATURE_VOCAB_META_KEY,
        "feature_prediction_width",
        "fp_target_source_certificate_sha256",
    }
    saved_dataloader_state = meta_data.get("dataloader_state_dict")
    if saved_dataloader_state is not None and not isinstance(saved_dataloader_state, Mapping):
        raise ValueError(f"checkpoint {checkpoint_dir} has a non-mapping dataloader_state_dict")
    dataloader_state = select_dataloader_rank_state(
        saved_dataloader_state,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    if dataloader_state is None:
        dataloader_state = {}
    present = keys.intersection(dataloader_state)
    if layout is None or not layout.is_compact:
        if present:
            raise ValueError(
                f"checkpoint {checkpoint_dir} contains compact-FP dataloader "
                "metadata but the configured run is full-width or has feature "
                f"prediction disabled: present={sorted(present)}"
            )
        return

    if present != keys:
        raise ValueError(
            f"checkpoint {checkpoint_dir} has missing/partial compact-FP "
            f"dataloader metadata: expected={sorted(keys)}, present={sorted(present)}"
        )
    if source_certificate_digest is None:
        raise ValueError("compact-FP resume requires a validated packed-source certificate digest")
    assert layout.feature_vocab is not None
    expected_vocab_metadata = layout.feature_vocab.metadata_dict()
    if dataloader_state[FEATURE_VOCAB_META_KEY] != expected_vocab_metadata:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} compact-FP vocabulary metadata does "
            "not match the configured effective vocabulary: checkpoint="
            f"{dataloader_state[FEATURE_VOCAB_META_KEY]!r}, "
            f"configured={expected_vocab_metadata!r}"
        )
    if dataloader_state["feature_prediction_width"] != layout.output_width:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} compact-FP width "
            f"{dataloader_state['feature_prediction_width']!r} does not match "
            f"configured width {layout.output_width}"
        )
    if dataloader_state["fp_target_source_certificate_sha256"] != source_certificate_digest:
        raise ValueError(
            f"checkpoint {checkpoint_dir} was trained from a different packed "
            "FP-target source certificate"
        )

    weight = model_data.get("feature_pred_head.weight")
    bias = model_data.get("feature_pred_head.bias")
    if not isinstance(weight, t.Tensor) or weight.ndim != 2:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} compact-FP head weight is missing or malformed"
        )
    if not isinstance(bias, t.Tensor) or bias.ndim != 1:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} compact-FP head bias is missing or malformed"
        )
    if weight.shape[0] != layout.output_width or bias.shape[0] != layout.output_width:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} compact-FP head width is inconsistent "
            f"with the configured vocabulary size {layout.output_width}: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )


def _close_training_background_workers(
    *,
    batch_prefetch_loader: ClosableBatchPrefetcher | None = None,
    cuda_staging_loader: PrecomputedCudaStagingLoader | None,
    remote_body_prefetcher: RemoteBodyPrefetcher | None = None,
    checkpoint_publisher: RemoteCheckpointPublisher | None,
    active_error: BaseException | None,
) -> None:
    failures: list[tuple[str, BaseException]] = []
    outer_producers_terminal = True
    for label, worker in (
        ("remote batch prefetch loader", batch_prefetch_loader),
        ("CUDA staging loader", cuda_staging_loader),
    ):
        if worker is None:
            continue
        try:
            worker.close()
        except BaseException as exc:
            failures.append((label, exc))
        try:
            outer_producers_terminal = bool(worker.is_terminal()) and outer_producers_terminal
        except BaseException as exc:
            outer_producers_terminal = False
            failures.append((f"{label} terminal-state check", exc))
    body_worker_terminal = remote_body_prefetcher is None
    if remote_body_prefetcher is not None:
        try:
            remote_body_prefetcher.stop_worker()
        except BaseException as exc:
            failures.append(("rank-striped remote body worker", exc))
        try:
            body_worker_terminal = remote_body_prefetcher.worker_is_terminal()
        except BaseException as exc:
            body_worker_terminal = False
            failures.append(("rank-striped remote body worker terminal-state check", exc))
    if checkpoint_publisher is not None:
        try:
            checkpoint_publisher.close()
        except BaseException as exc:
            failures.append(("remote checkpoint publisher", exc))
    if remote_body_prefetcher is not None and outer_producers_terminal and body_worker_terminal:
        try:
            remote_body_prefetcher.release_session()
        except BaseException as exc:
            failures.append(("remote body cache session", exc))
    elif remote_body_prefetcher is not None:
        retained_lock_error = RemoteTrainingError(
            "remote body cache session lock was retained because background "
            "work was not proven terminal"
        )
        failures.append(("remote body cache session", retained_lock_error))
        logger.opt(exception=retained_lock_error).error(str(retained_lock_error))

    if not failures:
        return
    if active_error is not None:
        for label, failure in failures:
            active_error.add_note(
                f"{label} also failed during teardown: {type(failure).__name__}: {failure}"
            )
            logger.opt(exception=failure).error(f"{label} failed during teardown")
        return

    primary_label, primary = failures[0]
    for label, failure in failures[1:]:
        primary.add_note(
            f"{label} also failed during teardown: {type(failure).__name__}: {failure}"
        )
    logger.opt(exception=primary).error(f"{primary_label} failed during teardown")
    raise primary


def train_one_config(
    cfg: DictConfig,
    shared: SharedState,
    wandb_run: Any,
    *,
    startup_trace: StartupTrace | None = None,
) -> TrainResult:
    """Run a single training config using shared state. Returns TrainResult."""
    _reject_precompute_only_packed_row_dir(cfg)
    reject_feature_clustering_config(cfg, "conditional_encdec_train")
    _validate_feature_clusters_training_config(cfg)
    startup_t0 = time.time()
    loader_health_gate_cursor = loader_health_snapshot()

    # Unpack shared state for local convenience
    tokenizer = shared.tokenizer
    token_bytes = shared.token_bytes
    feature_index = shared.feature_index
    per_feature_stats = shared.per_feature_stats
    n_sae_features = shared.n_sae_features
    feature_dir = shared.feature_dir
    boundaries = shared.boundaries
    model = shared.model
    orig_model = shared.orig_model
    cached_context_provider = shared.cached_context_provider
    device = shared.device
    device_type = shared.device_type
    ddp_rank = shared.ddp_rank
    ddp_world_size = shared.ddp_world_size
    gpu_peak_flops = shared.gpu_peak_flops
    D_REF = shared.d_ref
    B_REF = shared.b_ref

    master_process = ddp_rank == 0
    synchronize = t.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = t.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    with _startup_phase(startup_trace, "config_identity"):
        # The fast sweep builds a fresh per-config DictConfig, so the startup cfg
        # mutation from build_shared_state is not visible here. Reapply the
        # splice_counterfactual data_dir derivation before any per-config snapshot,
        # feature pipeline, dataloader, or validation spec reads cfg.data_dir.
        _resolve_splice_counterfactual_data_dir(cfg, feature_dir)

        # --- Auto activation-frequency blacklist (per-config, producer) ---
        # MUST also run here, not only in build_shared_state: the amortized fast
        # sweep (run_encdec_sweep_fast.py) calls build_shared_state(base_cfg) once
        # but train_one_config(per_cfg) with a FRESH per-config DictConfig, so the
        # build_shared_state mutation is not visible here. Re-applying makes the
        # checkpoint-mask leaf + feature pipeline consistent with the masked
        # discovery selection-hash. Idempotent + cheap: in the standalone /
        # subprocess-sweep path cfg.feature_mask is already set (→
        # SKIPPED_USER_MASK, no-op); reading feature_statistics is a single small
        # file. Disabled/empty/user-mask paths leave cfg byte-identical (I1/I2/I5).
        # Runs BEFORE the user_config snapshot below so the persisted checkpoint
        # metadata + wandb config record the auto-applied feature_mask (the
        # fast-sweep per_cfg would otherwise log a misleading feature_mask: null).
        maybe_apply_frequency_blacklist(
            cfg,
            feature_dir,
            n_sae_features,
            ddp_rank=ddp_rank,
            barrier_fn=ddp_barrier,
        )

        user_config = _checkpoint_user_config(cfg)

        # --- Parse feature selection/negative/delta from this config ---
        feature_selection_raw = _cfg_dict(cfg.feature_selection, name="feature_selection")
        feature_selection = parse_feature_selection(feature_selection_raw)

    # --- Checkpoint dir (hoisted early for resume support) ---
    base_dir = get_base_dir()
    output_dirname = cfg.model_tag if cfg.model_tag else f"d{cfg.depth}"
    # n_sae_features is available from shared.n_sae_features (line ~902 above).
    # resolve_feature_mask returns None when no mask configured → append_mask_leaf
    # returns the path unchanged (I2: byte-identical legacy checkpoint path).
    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    resolved_feature_clusters = resolve_feature_clusters(cfg, n_sae_features)
    shared_cluster_descriptor = canonical_feature_clusters_descriptor(
        shared.resolved_feature_clusters
    )
    per_config_cluster_descriptor = canonical_feature_clusters_descriptor(resolved_feature_clusters)
    if per_config_cluster_descriptor != shared_cluster_descriptor:
        raise FeatureClustersError(
            "this config's feature_clusters resolves to different semantics than "
            "the shared model/data state. Static cluster content cannot vary across "
            "configs in one fast sweep; use the subprocess sweep for separate runs"
        )

    # The encoder's tables were sized from the SHARED config's mask, but this is a
    # per-config DictConfig (the fast sweep composes a fresh one). The sweep provably
    # cannot vary the mask — its grid is a 4-axis product over feature_selection,
    # negative_selection, feature_delta and fp_weight, and training_overrides is one
    # flat list applied identically to base and every per-config. This assert exists so
    # that if that ever changes, we fail LOUDLY instead of silently training against a
    # dictionary the model was not built for.
    per_config_clusters = (
        FeatureClusterConsolidation.build(
            resolved_feature_clusters,
            feature_mask,
        )
        if resolved_feature_clusters is not None
        else None
    )
    per_config_vocab = effective_feature_vocab(
        per_config_clusters if per_config_clusters is not None else feature_mask,
        n_sae_features,
    )
    if per_config_vocab != orig_model.feature_vocab:
        raise FeatureVocabError(
            "this config's feature_mask and/or feature_clusters resolve to a different "
            "encoder vocabulary than the shared model was built with "
            f"(config: {None if per_config_vocab is None else per_config_vocab.size} "
            f"features, model: "
            f"{None if orig_model.feature_vocab is None else orig_model.feature_vocab.size}). "
            "The encoder's embedding tables are sized by the vocabulary, so it cannot "
            "vary across configs sharing one model. Use the subprocess sweep "
            "(run_encdec_sweep.py), which builds a fresh model per config."
        )

    checkpoint_dir = append_gradient_leaf(
        append_mask_leaf(
            os.path.join(
                base_dir,
                "conditional_encdec_checkpoints",
                output_dirname,
                f"s{cfg.segment_size}",
                feature_selection_dirname(feature_selection),
            ),
            feature_mask,
        ),
        feature_selection,
    )
    checkpoint_dir = append_feature_clusters_leaf(
        checkpoint_dir,
        resolved_feature_clusters,
    )
    remote_checkpoint_store: RemoteCheckpointStore | None = None
    checkpoint_lineage = shared.remote_run_fingerprint
    if shared.remote_training is not None:
        if (
            shared.remote_backend is None
            or shared.remote_run_fingerprint is None
            or shared.remote_resolved_training_config is None
            or shared.remote_selection_contracts is None
        ):
            raise RemoteTrainingError(
                "remote training shared state is missing its verified semantic inputs"
            )
        current_device_batch_size = cfg.get("device_batch_size")
        current_context_length = cfg.get("max_seq_len")
        if (
            type(current_device_batch_size) is not int
            or current_device_batch_size <= 0
            or type(current_context_length) is not int
            or current_context_length <= 0
        ):
            raise RemoteTrainingError(
                "remote training requires positive strict device_batch_size "
                "and max_seq_len before checkpoint restoration"
            )
        try:
            current_total_batch_size = _resolve_total_batch_size(
                cfg.get("total_batch_size"),
                model=orig_model,
                target_param_data_ratio=cfg.target_param_data_ratio,
                d_ref=shared.d_ref,
                b_ref=shared.b_ref,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteTrainingError(
                "remote training could not resolve total_batch_size before checkpoint restoration"
            ) from exc
        world_microbatch_tokens = (
            current_device_batch_size * current_context_length * ddp_world_size
        )
        if current_total_batch_size % world_microbatch_tokens != 0:
            raise RemoteTrainingError(
                "total_batch_size must be divisible by "
                "device_batch_size * max_seq_len * ddp_world_size before "
                f"checkpoint restoration; got {current_total_batch_size} and "
                f"{world_microbatch_tokens}"
            )
        remote_checkpoint_store = RemoteCheckpointStore(
            backend=shared.remote_backend,
            remote=shared.remote_training,
            checkpoint_dir=Path(checkpoint_dir),
            checkpoint_root=Path(base_dir) / "conditional_encdec_checkpoints",
            remote_run_fingerprint=shared.remote_run_fingerprint,
            world_size=ddp_world_size,
            semantic_fingerprint_v2=shared.remote_run_fingerprint,
            resolved_training_config=shared.remote_resolved_training_config,
            selection_contracts=shared.remote_selection_contracts,
        )
    model_config = orig_model.dec_config
    model_config_kwargs = asdict(model_config)

    # Parameter shapes and config determine accounting, independently of weight
    # initialization or checkpoint assignment. Resolve once before resume checks.
    negative_selection_raw = _optional_cfg_dict(cfg, "negative_selection")
    negative_selection = parse_negative_selection(negative_selection_raw)
    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    avg_features_per_segment = avg_features_per_segment_estimate(
        feature_selection, negative_selection
    )
    flops_segment_size = cfg.segment_size
    num_flops_per_token = model.estimate_flops(avg_features_per_segment, flops_segment_size)
    num_scaling_params = _scaling_parameter_count(model)
    target_tokens = int(cfg.target_param_data_ratio * num_scaling_params)
    total_batch_size = _resolve_total_batch_size(
        cfg.total_batch_size,
        model=model,
        target_param_data_ratio=cfg.target_param_data_ratio,
        d_ref=D_REF,
        b_ref=B_REF,
    )
    num_iterations = _resolve_training_iterations(
        cfg,
        num_flops_per_token=num_flops_per_token,
        total_batch_size=total_batch_size,
        target_tokens=target_tokens,
    )

    # Resolve the new consumer-side schedule before restoring model/optimizer
    # state. The disabled branch retains the existing loader/RNG/metadata path.
    cooldown_config = parse_conditioning_cooldown(cfg.get("conditioning_cooldown"))
    cooldown = None
    if cooldown_config is not None:
        if cfg.label_dropout != 0.0:
            raise ValueError("conditioning_cooldown requires label_dropout=0.0; two dropout policies cannot be combined")
        if shared.original_val_feature_dir is not None and not (Path(feature_dir) / PROVENANCE_DIRNAME).is_dir():
            raise ValueError("conditioning_cooldown requires splice provenance for this known spliced feature directory")
        cooldown = resolve_conditioning_cooldown(
            cooldown_config,
            num_iterations=num_iterations,
            total_batch_size=total_batch_size,
            seed=cfg.get("seed", 42),
            world_size=ddp_world_size,
            device_batch_size=cfg.device_batch_size,
            max_seq_len=cfg.max_seq_len,
            device_type=device_type,
        )
        assert cooldown is not None
        cooldown_window_tokens = (cooldown.end_step - cooldown.start_step + 1) * cooldown.total_batch_size
        logger.info(
            f"Conditioning cooldown: updates {cooldown.start_step}..{cooldown.end_step} "
            f"inclusive, retention {cooldown.start_fraction:.6f}->{cooldown.end_fraction:.6f}, "
            f"granularity={cooldown.granularity}, {cooldown_window_tokens:,} nominal tokens "
            f"({cooldown.total_batch_size:,}/update)"
        )

    # --- Resume from checkpoint ---
    if (
        remote_checkpoint_store is not None
        and shared.remote_training is not None
        and shared.remote_training.resume_latest
    ):

        def _restore_latest_remote_checkpoint() -> dict[str, Any] | None:
            latest = remote_checkpoint_store.latest_complete_step()
            if latest is not None:
                remote_checkpoint_store.restore(latest)
                return {
                    "step": latest,
                    "checkpoint_lineage": remote_checkpoint_store.remote_run_fingerprint,
                    "compatibility_mode": remote_checkpoint_store.compatibility_mode,
                }
            return None

        remote_resume = ddp_rank0_action(
            action=_restore_latest_remote_checkpoint,
            world_size=ddp_world_size,
            rank=ddp_rank,
        )
        if remote_resume is None:
            resume_step_or_none = None
        elif not isinstance(remote_resume, Mapping):
            raise RemoteTrainingError("remote checkpoint compatibility result was malformed")
        else:
            resume_step_or_none = remote_resume.get("step")
            checkpoint_lineage = remote_resume.get("checkpoint_lineage")
            compatibility_mode = remote_resume.get("compatibility_mode")
            if (
                type(resume_step_or_none) is not int
                or not isinstance(checkpoint_lineage, str)
                or compatibility_mode not in {"direct_v2", "legacy_v1_migrated"}
            ):
                raise RemoteTrainingError("remote checkpoint compatibility result was invalid")
            remote_checkpoint_store.remote_run_fingerprint = checkpoint_lineage
            remote_checkpoint_store.compatibility_mode = compatibility_mode
            logger.info(
                "remote_training resume identity: "
                f"semantic_fingerprint_v2={shared.remote_run_fingerprint} "
                f"checkpoint_lineage={checkpoint_lineage} "
                f"compatibility_mode={compatibility_mode}"
            )
    else:
        resume_step_or_none = resolve_resume_step(int(cfg.get("resume_from_step", -1)))
    resuming = resume_step_or_none is not None
    resume_step = resume_step_or_none if resuming else -1
    configured_fp_target_layout = (
        shared.fp_target_layout if cfg.get("feature_prediction", None) is not None else None
    )
    dataloader_resume_state: dict[str, Any] | None = None
    if resuming:
        logger.info(f"Resuming from step {resume_step}")
        with _startup_phase(startup_trace, "checkpoint_resume"):
            model_data, optimizer_data, meta_data = load_checkpoint(
                checkpoint_dir,
                resume_step,
                device,
                load_optimizer=True,
                rank=ddp_rank,
                report_rank_skew=True,
                broadcast_model=True,
            )
        validate_conditioning_cooldown_resume(cooldown, meta_data)
        if checkpoint_lineage is not None:
            assert_checkpoint_remote_run(
                meta_data,
                checkpoint_lineage,
            )
        dataloader_resume_state = select_dataloader_rank_state(
            meta_data.get("dataloader_state_dict"),
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        if dataloader_resume_state is not None:
            dataloader_resume_state = dict(dataloader_resume_state)
            dataloader_resume_state.setdefault(
                "device_batch_size",
                meta_data.get("device_batch_size"),
            )
        if meta_data.get("conditioning_cooldown") is not None and (
            dataloader_resume_state is None
            or "conditioning_splice_identity" not in dataloader_resume_state
            or "conditioning_splice_mask" not in dataloader_resume_state
        ):
            raise ValueError(
                "active conditioning_cooldown checkpoint requires conditioning_splice_identity "
                "and conditioning_splice_mask in each saved rank's dataloader state"
            )
        validate_feature_prediction_resume(meta_data, orig_model.feature_pred_config)
        # Validate semantic lineage before assign=True mutates the shared model.
        # A cluster mismatch must leave model parameters and persistent vocab
        # buffers untouched, especially in the in-process fast sweep.
        assert_resume_feature_clusters_match(
            resolved_feature_clusters,
            meta_data,
            checkpoint_dir=checkpoint_dir,
        )
        assert_resume_feature_mask_matches(
            feature_mask,
            meta_data,
            checkpoint_dir=checkpoint_dir,
        )
        assert_resume_feature_vocab_matches(
            orig_model.feature_vocab,
            meta_data,
            checkpoint_dir=checkpoint_dir,
        )
        assert_resume_gradient_matches(
            getattr(feature_selection, "gradient", None),
            meta_data,
            checkpoint_dir=checkpoint_dir,
        )
        _assert_not_pre_effective_compaction_checkpoint(
            orig_model.feature_vocab,
            feature_mask,
            per_config_clusters,
            model_data,
            checkpoint_dir=checkpoint_dir,
        )
        # BEFORE load_state_dict: PyTorch would raise a bare "size mismatch" on a
        # legacy (uncompacted) masked checkpoint, which tells an operator nothing.
        # Raise our own, naming the escape hatches. Also catches the two cases where
        # the shapes happen to agree but the dictionaries do not (issue #695 §4.9.3).
        assert_checkpoint_vocab_compatible(
            orig_model.feature_vocab, model_data, checkpoint_dir=checkpoint_dir
        )
        _assert_compact_fp_checkpoint_compatible(
            configured_fp_target_layout,
            shared.fp_target_source_certificate_digest,
            model_data,
            meta_data,
            checkpoint_dir=checkpoint_dir,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        orig_model.load_state_dict(model_data, strict=True, assign=True)
        # assign=True REPLACES the persistent vocab buffer while the derived map keeps
        # whatever init_weights() produced. Verify they still agree.
        orig_model.assert_vocab_consistent()
        del model_data
    else:
        _startup_skip(
            startup_trace,
            "checkpoint_resume",
            fields={"resumed": False},
        )

    if negative_selection is not None:
        if cfg.num_buckets < 2:
            raise ValueError(
                f"Negative sampling requires num_buckets >= 2 (at least 1 positive + 1 negative), "
                f"got num_buckets={cfg.num_buckets}."
            )
        logger.info(
            f"Negative sampling: {negative_selection.distribution.value}(mean={negative_selection.mean}, std={negative_selection.std})"
        )

    feature_delta_raw = _optional_cfg_dict(cfg, "feature_delta")
    feature_delta = parse_feature_delta(feature_delta_raw)
    if feature_delta is not None:
        logger.info(
            f"Feature delta conditioning: mode={feature_delta.mode.value}, neg_delta_ratio={feature_delta.neg_delta_ratio}"
        )

    # --- Build feature pipeline (delegates parse + build to shared helper) ---
    rng_seed = (cfg.get("seed", 42) * 1000003 + ddp_rank) % (2**63)
    with _startup_phase(startup_trace, "feature_pipeline_setup"):
        feature_pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=cfg.num_buckets,
            seed=rng_seed,
            ddp_rank=0,
            feature_delta=feature_delta,
            feature_mask=feature_mask,
        )

    # Delta-free twin pipeline for the original/base val/* pass of dual
    # validation. Built once here (mirroring the main pipeline's construction
    # args) and reused every eval; None for every other config.
    delta_free_val_pipeline: FeaturePipelineConfig | None = None
    if (
        feature_delta is not None
        and feature_delta.mode is DeltaMode.splice_counterfactual
        and (
            shared.original_val_feature_index is not None
            or shared.original_val_packed_row_dir is not None
        )
    ):
        delta_free_val_pipeline = _build_delta_free_val_pipeline(
            cfg,
            per_feature_stats=per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=cfg.num_buckets,
            seed=rng_seed,
            feature_mask=feature_pipeline.feature_mask,
        )

    # --- Build threshold config for steering eval ---
    # The steering eval's scan+threshold pipeline needs a ThresholdConfig.
    # For SamplingSelection configs, we use segment_topk with a large k as a
    # passthrough (keeps all features). For ThresholdSelection, use the actual config.
    if isinstance(feature_selection, ThresholdSelection):
        steering_threshold_cfg: ThresholdConfig | None = ThresholdConfig(
            feature_selection.threshold_type, feature_selection.threshold
        )
        steering_precomputed_thresholds: t.Tensor | None = None
        if feature_selection.threshold_type in (
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        ):
            steering_precomputed_thresholds = build_thresholds_from_stats(
                per_feature_stats,
                n_sae_features,
                feature_selection.threshold_type,
                feature_selection.threshold,
            )
    else:
        # SamplingSelection: use segment_topk capped at max_k (or mean+4*std as fallback)
        # so the eval probe reference matches what the model could have been conditioned on.
        assert isinstance(feature_selection, SamplingSelection)
        if feature_selection.max_k is not None:
            steering_topk = feature_selection.max_k
        else:
            steering_topk = int(feature_selection.mean + 4 * feature_selection.std)
        steering_threshold_cfg = ThresholdConfig(ThresholdType.segment_topk, steering_topk)
        steering_precomputed_thresholds = None

    # Build percentile table for steering eval percentile metrics
    steering_percentile_table: t.Tensor | None = build_percentile_table(
        per_feature_stats, n_sae_features
    )

    # Legacy steering builds its thresholded cache lazily on first cadence.
    # Clustered steering instead validates and joins the delta-aware offline
    # validation selection before the first training batch is consumed.
    inline_eval_cache: InlineEvalCache | None = None
    if shared.feature_clusters is not None and cfg.steering_eval_every > 0:
        if shared.steering_eval_cfg is None:
            raise RuntimeError("clustered steering is enabled but shared steering config is absent")
        if shared.val_packed_row_dir is None or shared.val_selection_dir is None:
            raise FileNotFoundError(
                "clustered steering requires the derived validation packed rows "
                "and exact precomputed selection discovered at startup"
            )
        clustered_selection_hash = compute_selection_hash(
            positive_selection=feature_selection_raw,
            negative_selection=negative_selection_raw,
            delta_config=feature_delta_raw,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.get("seed", 42)),
            ddp_rank=0,
            feature_mask=canonical_dict(feature_pipeline.feature_mask),
            feature_clusters=canonical_feature_clusters_descriptor(resolved_feature_clusters),
            splice_batch_size=cfg.get("selection_splice_batch_size"),
        )
        if shared.val_selection_dir.name != clustered_selection_hash:
            raise FeatureClustersError(
                "clustered steering validation selection differs from the "
                "current training selection identity: "
                f"discovered={shared.val_selection_dir.name!r}, "
                f"configured={clustered_selection_hash!r}"
            )
        logger.info(
            "Building clustered inline steering cache from validated derived "
            f"selection {clustered_selection_hash[:16]}..."
        )
        clustered_cache_t0 = time.monotonic()
        remote_inline_source = shared.remote_inline_eval_source
        inline_feature_index = (
            remote_inline_source.feature_index
            if remote_inline_source is not None
            else _require_global_feature_index(
                feature_index,
                surface="clustered inline steering cache construction",
            )
        )
        clustered_build = prepare_clustered_inline_eval_cache(
            feature_dir=Path(feature_dir),
            packed_row_dir=shared.val_packed_row_dir,
            selection_dir=shared.val_selection_dir,
            selection_hash=clustered_selection_hash,
            feature_index=inline_feature_index,
            data_dir=cfg.get("data_dir"),
            n_eval_docs=shared.steering_eval_cfg.n_eval_docs,
            seed=shared.steering_eval_cfg.seed,
            model_max_seq_len=int(cfg.max_seq_len),
            effective_buckets=feature_pipeline.effective_buckets,
            feature_mask=feature_pipeline.feature_mask,
            feature_clusters=shared.feature_clusters,
            feature_vocab=orig_model.feature_vocab,
            delta_config=feature_delta_raw,
            boundary_fingerprint=tensor_content_fingerprint(boundaries),
            splice_batch_size=cfg.get("selection_splice_batch_size"),
            validation_shard_stem=(
                remote_inline_source.stem if remote_inline_source is not None else None
            ),
            remote_derived_feature_fingerprint=(
                shared.remote_derived_feature_fingerprint
                if remote_inline_source is not None
                else None
            ),
            source_certificate_fingerprint=(
                remote_inline_source.aggregate_sha256 if remote_inline_source is not None else None
            ),
        )
        if remote_inline_source is not None:
            remote_inline_source.validate_signatures()
        assert_clustered_inline_cache_ddp_consistent(
            clustered_build.payload,
            world_size=ddp_world_size,
        )
        inline_eval_cache = InlineEvalCache(
            selected_docs=clustered_build.selected_docs,
            selection_metrics=clustered_build.selection_metrics,
            doc_texts=clustered_build.doc_texts,
            clustered_payload=clustered_build.payload,
        )
        if remote_inline_source is not None:
            # The immutable clustered payload is the sole cadence-time
            # consumer. Evict the multi-GB decoded shard before training while
            # retaining the authenticated sparse catalog for a later config in
            # the in-process fast sweep.
            inline_feature_index.drop_except(())
        logger.info(
            "Built clustered inline steering cache in "
            f"{time.monotonic() - clustered_cache_t0:.2f}s "
            f"(n_docs_selected={len(inline_eval_cache.selected_docs)}, "
            f"digest={clustered_build.payload.digest[:16]}...)"
        )

    # --- Patch feature prediction weight if configured ---
    fp_cfg = cfg.get("feature_prediction")
    if fp_cfg is not None and orig_model.feature_pred_config is not None:
        orig_model.feature_pred_config.weight = fp_cfg.weight
    if fp_cfg is not None and shared.fp_target_layout is None:
        raise ValueError(
            "this config enables feature_prediction, but the shared model was "
            "built without a feature-prediction head. Use the subprocess sweep "
            "or build shared state from a config with feature_prediction enabled."
        )
    fp_target_layout = configured_fp_target_layout

    # --- Wire feature_mask into the model's FP-loss gate ---
    # Set on orig_model (the unwrapped EncoderDecoderGPT that executes forward).
    # model may be a compiled OptimizedModule; orig_model is always the raw module.
    # Defaults to None (byte-identical no-op when no mask is configured, invariant I5).
    orig_model.feature_mask = feature_pipeline.feature_mask

    # --- Reseed global RNG ---
    t.manual_seed(rng_seed)
    if device_type == "cuda":
        t.cuda.manual_seed(rng_seed)

    # --- Reinit weights (skip when resuming — weights loaded from checkpoint) ---
    if not resuming:
        orig_model.init_weights()
        maybe_initialize_encoder_feature_embeddings_from_sae(
            orig_model,
            cfg,
            seed=int(cfg.get("seed", 42)),
            feature_clusters=shared.feature_clusters,
        )

    # --- Scaling laws: param counts, training horizon, batch size, LR ---
    logger.info("Parameter counts:")
    for key, value in param_counts.items():
        logger.info(f"{key:24s}: {value:,}")
    logger.info(
        f"Estimated FLOPs per token: {num_flops_per_token:e} "
        f"(avg_features_per_segment={avg_features_per_segment}, segment_size={flops_segment_size})"
    )

    # Batch size
    if cfg.total_batch_size == -1:
        logger.info(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    tokens_per_fwdbwd = cfg.device_batch_size * cfg.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    train_prefetch_batches = resolve_train_prefetch_batches(
        cfg.get("train_prefetch_batches"),
        PositiveInt.parse(grad_accum_steps),
    )
    train_shard_lookahead = resolve_train_shard_lookahead(cfg.get("train_shard_lookahead"))
    node_cache_config = resolve_train_node_cache(
        cfg.get("train_node_cache_dir"),
        cfg.get("train_node_cache_max_bytes"),
        cfg.get("train_node_cache_reserve_bytes"),
    )
    node_cache = (
        NodeLocalShardCache(node_cache_config) if node_cache_config is not None else None
    )

    # LR correction
    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio**0.5
        logger.info(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
        )

    # Weight decay
    weight_decay_scaled = (
        cfg.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    )
    if weight_decay_scaled != cfg.weight_decay:
        logger.info(
            f"Scaling weight decay from {cfg.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {cfg.depth}"
        )

    # --- Optimizer ---
    with _startup_phase(startup_trace, "optimizer_setup"):
        logger.info("Setting up optimizer...")
        optimizer = model.setup_optimizer(
            unembedding_lr=cfg.unembedding_lr * batch_lr_scale,
            embedding_lr=cfg.embedding_lr * batch_lr_scale,
            scalar_lr=cfg.scalar_lr * batch_lr_scale,
            adam_betas=(cfg.adam_beta1, cfg.adam_beta2),
            matrix_lr=cfg.matrix_lr * batch_lr_scale,
            weight_decay=weight_decay_scaled,
            encoder_embedding_lr=cfg.encoder_embedding_lr * batch_lr_scale,
            encoder_matrix_lr=cfg.encoder_matrix_lr * batch_lr_scale,
            xattn_matrix_lr=cfg.xattn_matrix_lr * batch_lr_scale,
        )

        # --- Load optimizer state from checkpoint ---
        if resuming:
            optimizer.load_state_dict(optimizer_data)
            del optimizer_data

    # --- Dataloader ---
    # --- Use precomputed packed rows from shared state (discovered in build_shared_state) ---
    with _startup_phase(startup_trace, "effective_data_path"):
        packed_row_dir, selection_dir, ignored_packed_rows_for_delta = (
            _packed_row_inputs_for_pipeline(
                packed_row_dir=shared.packed_row_dir,
                selection_dir=shared.selection_dir,
                feature_pipeline=feature_pipeline,
            )
        )
    if ignored_packed_rows_for_delta:
        logger.info(
            "feature_delta is enabled for this training config; ignoring "
            f"precomputed packed rows at {shared.packed_row_dir} and using "
            "the live dataloader."
        )

    effective_data_path = _describe_effective_data_path(
        packed_row_dir=packed_row_dir,
        selection_dir=selection_dir,
        ignored_packed_rows_for_delta=ignored_packed_rows_for_delta,
        use_precomputed_batches=bool(cfg.get("use_precomputed_batches", True)),
        remote_certified=shared.remote_training is not None,
    )
    if startup_trace is not None:
        startup_trace.event(
            "encdec_startup_data_path",
            packed_rows=effective_data_path.packed_rows,
            route=effective_data_path.route,
            selection=effective_data_path.selection,
            source=effective_data_path.source,
            reason=effective_data_path.reason,
            resolution_s=(startup_trace.phase_elapsed("packed_selection_artifacts")),
            decision_s=startup_trace.phase_elapsed("effective_data_path"),
        )

    if packed_row_dir is not None and cfg.get("next_segment_conditioning_offset", False):
        raise ValueError(
            "next_segment_conditioning_offset is not supported with precomputed packed rows. "
            "Use the live dataloader (remove packed row shards or set use_precomputed_batches=false)."
        )

    logger.info("Building train dataloader...")
    batch_prefetch_loader: ClosableBatchPrefetcher | None = None
    cuda_staging_loader: PrecomputedCudaStagingLoader | None = None
    remote_body_prefetcher: RemoteBodyPrefetcher | None = None
    remote_read_cancel_event: threading.Event | None = None
    if packed_row_dir is None:
        logger.warning(
            "No precomputed packed row shards found — falling back to live dataloader. "
            + (
                "fp_targets built from all extracted features. "
                if fp_target_layout is not None
                else "feature prediction is disabled; fp_targets will not be built. "
            )
            + f"Expected packed rows under: {feature_dir}/pct/..."
        )
        with _startup_phase(startup_trace, "dataloader_build"):
            train_loader = encoder_decoder_data_loader_with_state(
                tokenizer=tokenizer,
                B=cfg.device_batch_size,
                T=cfg.max_seq_len,
                split=DatasetSplit.TRAIN,
                feature_index=_require_global_feature_index(
                    feature_index,
                    surface="live training dataloader",
                ),
                pipeline=feature_pipeline,
                segment_size=shared.extraction_segment_size,
                decoder_segment_size=None,
                device=str(device),
                data_dir=cfg.get("data_dir"),
                resume_state_dict=dataloader_resume_state,
                tokenizer_batch_size=cfg.get("tokenizer_batch_size", 128),
                buffer_size=DEFAULT_BUFFER_SIZE,
                prefetch_batches=train_prefetch_batches,
                label_dropout=cfg.label_dropout,
                fp_target_layout=fp_target_layout,
                max_shards=cfg.get("max_train_shards", None),
                pad_F=shared.pad_F,
                pad_n_seg=shared.live_pad_n_seg,
                doc_prefetch_depth=cfg.get("doc_prefetch_depth", 0),
                next_segment_conditioning_offset=cfg.get("next_segment_conditioning_offset", False),
                **({"include_splice_mask": True} if cooldown is not None else {}),
            )
    else:
        train_remote_materializer = (
            shared.remote_materializers.get(str(selection_dir.resolve()))
            if selection_dir is not None
            else None
        )
        cuda_staging_config = (
            shared.remote_training.train_packed.cuda_staging
            if shared.remote_training is not None
            and shared.remote_training.train_packed is not None
            else None
        )
        loader_device = device
        expected_staging_pad_f: int | None = None
        expected_staging_n_seg: int | None = None
        if cuda_staging_config is not None:
            expected_staging_pad_f = (
                train_remote_materializer.max_features_per_segment
                if train_remote_materializer is not None
                else None
            )
            if shared.packed_materializer is not None:
                shared.packed_materializer.assert_manifest_current()
            packed_manifest = read_manifest(packed_row_dir)
            packed_metadata = packed_manifest.get("metadata", {}) if packed_manifest else {}
            expected_staging_n_seg = (
                packed_metadata.get("max_n_seg") if isinstance(packed_metadata, Mapping) else None
            )
            validate_precomputed_cuda_staging_eligibility(
                device=device,
                label_dropout=cfg.label_dropout,
                selection_dir=selection_dir,
                remote_materializer=train_remote_materializer,
                packed_materializer=shared.packed_materializer,
                resume_state_dict=dataloader_resume_state,
                expected_batch_size=cfg.device_batch_size,
                expected_context_len=cfg.max_seq_len,
                expected_pad_f=expected_staging_pad_f,
                expected_n_seg=expected_staging_n_seg,
            )
            loader_device = t.device("cpu")
        if shared.remote_training is not None:
            if selection_dir is None:
                raise RemoteTrainingError(
                    "remote body prefetch requires the verified train selection directory"
                )
            remote_body_prefetcher = RemoteBodyPrefetcher.start(
                remote=shared.remote_training,
                cache_root=Path(str(cfg.feature_root)),
                rank=shared.ddp_rank,
                world_size=shared.ddp_world_size,
                train_selection_dir=selection_dir,
                selection_materializers=list(shared.remote_materializers.values()),
                packed_materializer=shared.packed_materializer,
                resume_state_dict=dataloader_resume_state,
            )
            remote_read_cancel_event = remote_body_prefetcher.cancel_event
        try:
            with _startup_phase(startup_trace, "dataloader_build"):
                inner_loader = precomputed_data_loader(
                    packed_row_dir=packed_row_dir,
                    B=cfg.device_batch_size,
                    device=str(loader_device),
                    label_dropout=cfg.label_dropout,
                    selection_dir=selection_dir,
                    effective_buckets=feature_pipeline.effective_buckets,
                    feature_pipeline=feature_pipeline,
                    fp_target_layout=fp_target_layout,
                    pad_F=shared.pad_F,
                    ddp_rank=shared.ddp_rank,
                    ddp_world_size=shared.ddp_world_size,
                    resume_state_dict=dataloader_resume_state,
                    splice_feature_dir=(
                        Path(feature_dir)
                        if _pipeline_uses_splice_counterfactual(feature_pipeline)
                        or (
                            cooldown is not None
                            and (
                                (Path(feature_dir) / PROVENANCE_DIRNAME).exists()
                                or shared.original_val_feature_dir is not None
                            )
                        )
                        else None
                    ),
                    feature_clusters=shared.feature_clusters,
                    remote_materializer=train_remote_materializer,
                    packed_materializer=shared.packed_materializer,
                    packed_identity_materializer=shared.packed_identity_materializer,
                    remote_derived_feature_fingerprint=(shared.remote_derived_feature_fingerprint),
                    remote_read_cancel_event=remote_read_cancel_event,
                    train_shard_lookahead=train_shard_lookahead,
                    node_cache=node_cache,
                    **({"include_splice_mask": True} if cooldown is not None else {}),
                )
            if cuda_staging_config is None:
                if train_prefetch_batches == 0:
                    train_loader = inner_loader
                elif remote_body_prefetcher is None:
                    train_loader = _prefetch_wrapper(
                        inner_loader,
                        train_prefetch_batches,
                        use_cuda=t.device(loader_device).type == "cuda",
                    )
                else:
                    assert remote_read_cancel_event is not None
                    batch_prefetch_loader = ClosableBatchPrefetcher(
                        inner_loader,
                        PositiveInt.parse(train_prefetch_batches),
                        stop_callback=remote_read_cancel_event.set,
                        failure_check=remote_body_prefetcher.check_worker_errors,
                    )
                    train_loader = batch_prefetch_loader
            else:
                assert expected_staging_pad_f is not None
                assert expected_staging_n_seg is not None
                assert remote_read_cancel_event is not None
                cuda_staging_loader = PrecomputedCudaStagingLoader(
                    inner_loader,
                    device=device,
                    host_queue_bytes=cuda_staging_config.host_queue_bytes,
                    pinned_slots=cuda_staging_config.pinned_slots,
                    expected_batch_size=int(cfg.device_batch_size),
                    expected_context_len=int(cfg.max_seq_len),
                    expected_pad_f=expected_staging_pad_f,
                    expected_n_seg=expected_staging_n_seg,
                    remote_read_cancel_event=remote_read_cancel_event,
                )
                train_loader = cuda_staging_loader
        except BaseException as construction_error:
            _close_training_background_workers(
                batch_prefetch_loader=batch_prefetch_loader,
                cuda_staging_loader=cuda_staging_loader,
                remote_body_prefetcher=remote_body_prefetcher,
                checkpoint_publisher=None,
                active_error=construction_error,
            )
            raise
    checkpoint_publisher: RemoteCheckpointPublisher | None = None
    training_error: BaseException | None = None
    try:
        logger.info("Train dataloader built")
        logger.info("Fetching first training batch...")
        with _startup_phase(startup_trace, "first_batch_fetch"):
            (
                x,
                y,
                enc_fids,
                enc_mags,
                enc_mask,
                _enc_pcts,
                seg_map,
                seg_doc_id,
                fp_targets,
                dataloader_state_dict,
            ) = next(train_loader)
        logger.info(
            f"First batch ready: x={list(x.shape)}, enc_fids={list(enc_fids.shape)}, "
            f"seg_map={list(seg_map.shape)}"
        )

        # --- Training iterations ---
        with _startup_phase(startup_trace, "pre_first_update_control"):
            if cfg.num_iterations > 0:
                logger.info(f"Using user-provided number of iterations: {num_iterations:,}")
            elif cfg.target_flops > 0:
                logger.info(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
            elif cfg.target_param_data_ratio > 0:
                logger.info(
                    f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
                )
            total_tokens = total_batch_size * num_iterations
            logger.info(f"Total number of training tokens: {total_tokens:,}")
            logger.info(
                "Tokens : Scaling params ratio: "
                f"{total_batch_size * num_iterations / num_scaling_params:.2f}"
            )
            logger.info(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

            # LR schedule
            def get_lr_multiplier(it):
                warmup_iters = round(cfg.warmup_ratio * num_iterations)
                warmdown_iters = round(cfg.warmdown_ratio * num_iterations)
                if it < warmup_iters:
                    return (it + 1) / warmup_iters
                if it <= num_iterations - warmdown_iters:
                    return 1.0
                progress = (num_iterations - it) / warmdown_iters
                return progress * 1.0 + (1 - progress) * cfg.final_lr_frac

            def get_muon_momentum(it):
                frac = min(it / 300, 1)
                momentum = (1 - frac) * 0.85 + frac * 0.95
                return momentum

            def get_weight_decay(it):
                return weight_decay_scaled * (1 - it / num_iterations)

            # --- Loop state ---
            if not resuming:
                step = 0
                val_bpb = None
                min_val_bpb = float("inf")
                smooth_train_loss = 0
                smooth_fp_loss = 0.0
                total_training_time = 0
            else:
                step = meta_data["step"]
                loop_state = meta_data["loop_state"]
                val_bpb = meta_data.get("val_bpb")
                min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
                smooth_train_loss = loop_state.get("smooth_train_loss", 0)
                smooth_fp_loss = loop_state.get("smooth_fp_loss", 0.0)
                total_training_time = loop_state.get("total_training_time", 0)
                logger.info(f"Resumed at step {step}, smooth_train_loss={smooth_train_loss:.4f}")
        feature_stats_logged = False
        max_n_seg_seen = 0
        max_f_max_seen = 0
        results: dict = {}

        startup_time = time.time() - startup_t0

        logger.info(
            "Tokens / micro-batch / rank: "
            f"{cfg.device_batch_size} x {cfg.max_seq_len} = {tokens_per_fwdbwd:,}"
        )
        logger.info(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
        logger.info(
            f"Total batch size {total_batch_size:,} => "
            f"gradient accumulation steps: {grad_accum_steps}"
        )

        detect_anomaly = cfg.get("detect_anomaly", False)
        if detect_anomaly:
            logger.info("torch.autograd.set_detect_anomaly(True) — backward will be slower")
            t.autograd.set_detect_anomaly(True)

        # Chrome trace profiler setup
        profile_trace_step = cfg.get("profile_trace", None)
        profile_trace_n = cfg.get("profile_trace_steps", 3)
        _chrome_profiler: Any | None = None
        if profile_trace_step is not None:
            trace_dir = Path(cfg.get("output_dir", "out")) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Will capture torch.profiler trace at steps "
                f"{profile_trace_step}-{profile_trace_step + profile_trace_n - 1} "
                f"→ {trace_dir}"
            )

        logger.info(
            f"Entering training loop (starting at step {step}, {num_iterations:,} total iterations)"
        )

        # H10 fail-fast: the feature_mask MUST be wired onto the model before the
        # first (compile-traced) forward. If a refactor moves the set below the
        # training loop, torch.compile would bake feature_mask=None into the graph
        # and silently disable masking. This guard makes that regression loud.
        # No-mask path (I5): feature_pipeline.feature_mask is None,
        # orig_model.feature_mask is None → None is None → passes. Zero hot-path
        # cost; runs once before the loop.
        if orig_model.feature_mask is not feature_pipeline.feature_mask:
            raise RuntimeError(
                "feature_mask not wired onto orig_model before the training loop "
                f"(orig_model.feature_mask={orig_model.feature_mask!r}, "
                f"expected feature_pipeline.feature_mask={feature_pipeline.feature_mask!r}). "
                "It must be set before the first torch.compile-traced forward."
            )

        async_checkpoint_upload = (
            remote_checkpoint_store is not None
            and shared.remote_training is not None
            and shared.remote_training.async_checkpoint_upload
        )
        checkpoint_publication_pending = False
        validation_setup_caches: dict[str, PrecomputedValidationSetupCache] = {}
        if async_checkpoint_upload:
            checkpoint_publisher = ddp_create_checkpoint_publisher(
                store=remote_checkpoint_store,
                world_size=ddp_world_size,
                rank=ddp_rank,
            )
        while True:
            if remote_body_prefetcher is not None:
                remote_body_prefetcher.check_worker_errors()
            last_step = step == num_iterations
            flops_so_far = num_flops_per_token * total_batch_size * step

            if checkpoint_publication_pending:
                checkpoint_publication_pending = ddp_poll_checkpoint_publisher(
                    publisher=checkpoint_publisher,
                    world_size=ddp_world_size,
                    rank=ddp_rank,
                )

            # --- Eval ---
            # Skip the resume-step validation: it re-measures the bpb already
            # recorded at save time, and running it BEFORE the first training
            # step recompile-thrashes (B1 prewarm vs B14 eval guards; 224
            # recompile events, 731s eval, 600s NCCL watchdog — 2026-08-27)
            # on every resume. The next scheduled eval reuses the warm
            # training graphs exactly like a fresh start does.
            skip_resume_eval = resuming and step == resume_step and not last_step
            if skip_resume_eval and cfg.eval_every > 0 and step % cfg.eval_every == 0:
                logger.info(f"Step {step:05d} | Skipping resume-step validation")
            if (
                cfg.eval_every > 0
                and (last_step or step % cfg.eval_every == 0)
                and not skip_resume_eval
            ):
                # Gradient runs use the startup-VERIFIED val dir from
                # SharedState; a fresh discovery here could silently diverge
                # from what the startup gate checked.
                active_val_packed_row_dir = shared.val_packed_row_dir or (
                    discover_val_packed_row_dir(packed_row_dir)
                    if packed_row_dir is not None
                    else None
                )
                validation_specs = _build_validation_eval_specs(
                    packed_row_dir=packed_row_dir,
                    active_val_packed_row_dir=active_val_packed_row_dir,
                    active_val_selection_dir=shared.val_selection_dir,
                    original_val_packed_row_dir=shared.original_val_packed_row_dir,
                    original_val_selection_dir=shared.original_val_selection_dir,
                    live_dual=(shared.original_val_feature_index is not None),
                    original_val_feature_index=shared.original_val_feature_index,
                    spliced_val_feature_index=shared.spliced_val_feature_index,
                    original_val_data_dir=cfg.get("original_val_data_dir", None),
                    spliced_data_dir=cfg.get("data_dir", None),
                    live_data_dir=cfg.get("data_dir", None),
                    delta_free_pipeline=delta_free_val_pipeline,
                    active_splice_feature_dir=Path(shared.feature_dir),
                    remote_derived_feature_fingerprint=(shared.remote_derived_feature_fingerprint),
                )
                if not validation_specs and packed_row_dir is not None:
                    logger.warning(
                        f"No precomputed val shards found (expected at "
                        f"{packed_row_dir.parent / f'val_{packed_row_dir.name}'}). "
                        f"Skipping validation."
                    )
                elif validation_specs:
                    logger.info(
                        f"Step {step:05d} | Starting validation ({cfg.eval_tokens:,} tokens)..."
                    )
                    model.eval()
                    eval_steps = cfg.eval_tokens // (
                        cfg.device_batch_size * cfg.max_seq_len * ddp_world_size
                    )
                    log_dict = {
                        "step": step,
                        "total_training_flops": flops_so_far,
                        "total_training_time": total_training_time,
                    }
                    for spec in validation_specs:
                        if remote_body_prefetcher is not None:
                            remote_body_prefetcher.check_worker_errors()
                        with _validation_loader_health(
                            step=step,
                            spec=spec,
                            ddp_rank=ddp_rank,
                        ):
                            validation_setup_cache = _validation_setup_cache_for_spec(
                                spec,
                                validation_setup_caches,
                            )
                            val_loader = _build_validation_loader(
                                spec=spec,
                                cfg=cfg,
                                shared=shared,
                                feature_pipeline=feature_pipeline,
                                tokenizer=tokenizer,
                                device=device,
                                node_cache=node_cache,
                                validation_setup_cache=validation_setup_cache,
                            )
                            with disable_fp8(orig_model):
                                metrics = evaluate_encdec_bpb_conditioned(
                                    orig_model, val_loader, eval_steps, token_bytes, device
                                )
                        logger.info(
                            f"Step {step:05d} | {spec.log_label} bpb: {metrics.bpb_conditioned:.6f}"
                        )
                        log_dict.update(_validation_bpb_log(spec.metric_prefix, metrics))
                        if spec.metric_prefix == "val":
                            val_bpb = metrics.bpb_conditioned
                            if val_bpb < min_val_bpb:
                                min_val_bpb = val_bpb
                        if not feature_stats_logged:
                            _log_validation_feature_stats(
                                wandb_run=wandb_run,
                                prefix=spec.metric_prefix,
                                metrics=metrics,
                                cfg=cfg,
                                master_process=master_process,
                            )
                    if not feature_stats_logged:
                        feature_stats_logged = True
                    wandb_run.log(log_dict)
                    model.train()
                    if device_type == "cuda":
                        t.cuda.empty_cache()
                loader_health_gate_cursor = _log_loader_health_gate(
                    step=step,
                    gate="eval",
                    ddp_rank=ddp_rank,
                    previous=loader_health_gate_cursor,
                )

            # CORE metric evaluation with full-context cached conditioning.
            if cfg.core_metric_every > 0 and (
                last_step or (step > 0 and step % cfg.core_metric_every == 0)
            ):
                logger.info(f"Step {step:05d} | Starting CORE metric evaluation...")
                model.eval()
                try:
                    wrapper = build_wrapper_for_features(
                        orig_model,
                        tokenizer,
                        features=None,
                        context_provider=cached_context_provider,
                    )
                    with disable_fp8(orig_model):
                        core_results = evaluate_core(
                            wrapper,
                            tokenizer,
                            device,
                            max_per_task=cfg.core_metric_max_per_task,
                        )
                    logger.info(
                        f"Step {step:05d} | CORE (cached): {core_results['core_metric']:.4f}"
                    )
                    wandb_run.log(
                        {
                            "step": step,
                            "total_training_flops": flops_so_far,
                            "core_metric/cached": core_results["core_metric"],
                            "centered_results/cached": core_results["centered_results"],
                        }
                    )
                    results = core_results
                finally:
                    model.train()

            # Sampling with full-context cached conditioning.
            if (
                cfg.sample_every > 0
                and master_process
                and (last_step or (step > 0 and step % cfg.sample_every == 0))
            ):
                model.eval()
                try:
                    _run_sample_generation(
                        orig_model=orig_model,
                        tokenizer=tokenizer,
                        device=device,
                        sample_prompts=cfg.sample_prompts,
                        cached_context_provider=cached_context_provider,
                    )
                finally:
                    model.train()

            # --- Steering eval (all ranks participate, work partitioned internally) ---
            inline_eval_cache = _maybe_run_inline_steering_eval(
                step=step,
                last_step=last_step,
                cfg=cfg,
                shared=shared,
                model=model,
                orig_model=orig_model,
                feature_dir=feature_dir,
                steering_threshold_cfg=steering_threshold_cfg,
                steering_precomputed_thresholds=steering_precomputed_thresholds,
                steering_percentile_table=steering_percentile_table,
                tokenizer=tokenizer,
                device=device,
                device_type=device_type,
                boundaries=boundaries,
                feature_mask=feature_pipeline.feature_mask,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                master_process=master_process,
                wandb_run=wandb_run,
                inline_eval_cache=inline_eval_cache,
            )

            # Save checkpoint
            should_save_checkpoint = last_step or (
                step > 0 and cfg.save_every > 0 and step % cfg.save_every == 0
            )
            if remote_checkpoint_store is not None and resuming and step == resume_step:
                should_save_checkpoint = False
                logger.info(
                    f"Step {step:05d} | Remote resume: preserving the committed "
                    "checkpoint group until after the next training update"
                )
            if cooldown is not None and resuming and step == resume_step:
                # In particular, do not overwrite a legacy source checkpoint
                # with cooldown provenance before any cooldown update ran.
                should_save_checkpoint = False
            if should_save_checkpoint:
                if remote_body_prefetcher is not None:
                    remote_body_prefetcher.check_worker_errors()
                if checkpoint_publication_pending:
                    ddp_drain_checkpoint_publisher(
                        publisher=checkpoint_publisher,
                        world_size=ddp_world_size,
                        rank=ddp_rank,
                    )
                    checkpoint_publication_pending = False
                # All ranks compute an identical should_save_checkpoint (last_step,
                # save_every, and the remote-resume suppression are all rank-symmetric),
                # so every rank enters this block together. Gather the NEXT preloaded
                # batch's local cursor/RNG state before rank-local checkpoint persistence.
                dataloader_state_envelope = build_dataloader_state_envelope(
                    dataloader_state_dict,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                logger.info(f"Step {step:05d} | Saving checkpoint to {checkpoint_dir}...")
                # B6 canonical persistence: persist the resolved canonical form
                # (mode + sorted ids, NO path reference) so cross-host resumes can
                # verify mask identity without re-resolving the original path.
                # No-mask path (I2/I5): key is absent → meta byte-identical to pre-B6.
                _save_meta: dict = {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "device_batch_size": cfg.device_batch_size,
                    "max_seq_len": cfg.max_seq_len,
                    "dataloader_state_dict": dataloader_state_envelope,
                    "loop_state": {
                        "min_val_bpb": min_val_bpb,
                        "smooth_train_loss": smooth_train_loss,
                        "smooth_fp_loss": smooth_fp_loss,
                        "total_training_time": total_training_time,
                    },
                }
                if cooldown is not None:
                    _save_meta["conditioning_cooldown"] = cooldown.metadata_dict()
                if orig_model.feature_pred_config is not None:
                    effective_fp = serialize_feature_prediction_config(
                        orig_model.feature_pred_config
                    )
                    _save_meta["feature_prediction_config"] = effective_fp
                    _save_meta["user_config"] = {**user_config, "feature_prediction": effective_fp}
                if feature_mask is not None:
                    _save_meta["feature_mask_canonical"] = canonical_dict(feature_mask)
                # Mirror the encoder vocabulary into meta as a small fingerprint. The
                # persistent buffers in the state dict are the load-bearing record (the
                # checkpoint is self-describing); this exists so tooling can compare a
                # checkpoint's vocabulary without loading a multi-GB .pt. Guarded
                # insert, so a no-mask run's meta stays byte-identical (I2/I5).
                if orig_model.feature_vocab is not None:
                    _save_meta[FEATURE_VOCAB_META_KEY] = orig_model.feature_vocab.metadata_dict()
                _save_grad_spec = getattr(feature_selection, "gradient", None)
                if _save_grad_spec is not None:
                    # Includes the code-versioned score_reduction /
                    # split_policy_version constants the checkpoint-dir leaf
                    # omits — the resume assert pins the full lineage.
                    # Absent when no gradient ⇒ meta byte-identical (I2/I5).
                    _save_meta["gradient_attribution_canonical"] = (
                        canonical_gradient_selection_dict(_save_grad_spec)
                    )
                _save_meta.update(feature_clusters_checkpoint_metadata(resolved_feature_clusters))
                if shared.remote_run_fingerprint is not None:
                    if checkpoint_lineage is None:
                        raise RemoteTrainingError(
                            "remote checkpoint save is missing its resolved lineage"
                        )
                    _save_meta[REMOTE_RUN_META_KEY] = checkpoint_lineage
                    _save_meta[REMOTE_SEMANTIC_META_KEY] = build_remote_semantic_identity(
                        shared.remote_run_fingerprint
                    )
                if remote_checkpoint_store is None:
                    save_checkpoint(
                        checkpoint_dir,
                        step,
                        orig_model.state_dict(),
                        optimizer.state_dict(),
                        _save_meta,
                        rank=ddp_rank,
                        report_rank_skew=True,
                    )
                else:
                    with _remote_checkpoint_save_rank_skew(
                        step=step,
                        world_size=ddp_world_size,
                        rank=ddp_rank,
                    ):
                        save_checkpoint(
                            checkpoint_dir,
                            step,
                            orig_model.state_dict(),
                            optimizer.state_dict(),
                            _save_meta,
                            rank=ddp_rank,
                        )
                    checkpoint_publication_pending = ddp_publish_saved_checkpoint(
                        store=remote_checkpoint_store,
                        publisher=checkpoint_publisher,
                        async_upload=async_checkpoint_upload,
                        step=step,
                        world_size=ddp_world_size,
                        rank=ddp_rank,
                    )
                loader_health_gate_cursor = _log_loader_health_gate(
                    step=step,
                    gate="save",
                    ddp_rank=ddp_rank,
                    previous=loader_health_gate_cursor,
                )

            # Termination
            if last_step:
                if checkpoint_publication_pending:
                    ddp_drain_checkpoint_publisher(
                        publisher=checkpoint_publisher,
                        world_size=ddp_world_size,
                        rank=ddp_rank,
                    )
                    checkpoint_publication_pending = False
                break

            # --- Training step ---
            if step == 0 and cfg.compile:
                logger.info(
                    "Step 0: first forward pass will trigger torch.compile graph tracing "
                    "(this may take 30-90s)..."
                )

            # Start Chrome trace profiler on target step
            use_trace = (
                profile_trace_step is not None
                and profile_trace_step <= step <= profile_trace_step + profile_trace_n - 1
            )
            if use_trace and step == profile_trace_step:
                _chrome_profiler = t.profiler.profile(
                    activities=[
                        t.profiler.ProfilerActivity.CPU,
                        t.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                    experimental_config=t.profiler._ExperimentalConfig(profile_all_threads=True),
                )
                _chrome_profiler.__enter__()

            # Intentionally do not synchronize before starting the timer. Loader
            # event handoff already orders batch writes before the consumer stream,
            # while a device-wide barrier here would also wait for async staging.
            # The post-optimizer synchronize below remains inside dt.
            t0 = time.time()
            data_wait_s = 0.0
            # AUTOCAST_OFFPATTERN: conditional_encdec_train deliberately runs without an
            # ambient torch.amp.autocast. The encoder-decoder stack relies on embeddings
            # being cast to COMPUTE_DTYPE at the Linear layer boundary (see nanochat/gpt.py
            # Linear.forward), which is sufficient for bf16 matmul throughput. Adding
            # autocast here would double-cast and complicate the FP8 / gradient-accumulation
            # logic. See docs/architecture.md "Mixed Precision Contract".
            conditioning_fraction = cooldown.fraction_at(step) if cooldown is not None else None
            for _micro_step in range(grad_accum_steps):
                max_n_seg_seen = max(max_n_seg_seen, enc_fids.shape[1])
                max_f_max_seen = max(max_f_max_seen, enc_fids.shape[2])
                training_enc_mask = enc_mask
                if cooldown is not None:
                    assert conditioning_fraction is not None
                    if (
                        "conditioning_splice_mask" not in dataloader_state_dict
                        or "conditioning_splice_identity" not in dataloader_state_dict
                    ):
                        raise ValueError(
                            "conditioning_cooldown requires explicit loader conditioning_splice_mask "
                            "and conditioning_splice_identity"
                        )
                    conditioning_splice_mask = t.as_tensor(
                        dataloader_state_dict["conditioning_splice_mask"],
                        device=enc_mask.device,
                    )
                    training_enc_mask = apply_conditioning_cooldown(
                        enc_mask,
                        seg_doc_id,
                        fraction=conditioning_fraction,
                        granularity=cooldown.granularity,
                        seed=cooldown.seed,
                        step=step,
                        rank=ddp_rank,
                        micro_step=_micro_step,
                        splice_mask=conditioning_splice_mask,
                    )
                with t.profiler.record_function("forward"):
                    result = model(
                        x,
                        y,
                        enc_fids,
                        enc_mags,
                        training_enc_mask,
                        seg_map,
                        seg_doc_id=seg_doc_id,
                        static_max_seg_size=shared.max_seg_size,
                        fp_targets=fp_targets,
                    )
                if isinstance(result, ForwardResult):
                    loss = result.loss
                    train_loss = result.lm_loss.detach()
                    train_fp_loss = result.feature_pred_loss.detach()
                else:
                    loss = result
                    train_loss = loss.detach()
                    train_fp_loss = None
                loss = loss / grad_accum_steps
                with t.profiler.record_function("backward"):
                    loss.backward()
                with t.profiler.record_function("data_loading"):
                    data_wait_started = time.time()
                    (
                        x,
                        y,
                        enc_fids,
                        enc_mags,
                        enc_mask,
                        _enc_pcts,
                        seg_map,
                        seg_doc_id,
                        fp_targets,
                        dataloader_state_dict,
                    ) = next(train_loader)
                    data_wait_s += time.time() - data_wait_started
                    if remote_body_prefetcher is not None:
                        remote_body_prefetcher.check_worker_errors()
            # Step optimizer
            lrm = get_lr_multiplier(step)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            with t.profiler.record_function("optimizer_step"):
                optimizer.step()
                model.zero_grad(set_to_none=True)
            train_loss_f = train_loss.item()
            synchronize()

            # Stop Chrome trace profiler and export (rank 0 only to avoid rename race)
            if use_trace and step == profile_trace_step + profile_trace_n - 1:
                assert _chrome_profiler is not None
                _chrome_profiler.__exit__(None, None, None)
                if master_process:
                    trace_path = trace_dir / f"trace_step{profile_trace_step}.json"
                    _chrome_profiler.export_chrome_trace(str(trace_path))
                    logger.info(f"Exported Chrome trace to {trace_path}")
                    logger.info(
                        f"To view the trace:\n"
                        f"  1. On this machine:  python -m http.server 9001 -d {trace_dir}\n"
                        f"  2. On your laptop:   ssh -L 9001:localhost:9001 <this-machine>\n"
                        f"  3. Open https://ui.perfetto.dev/ and paste: http://localhost:9001/{trace_path.name}"
                    )
                _chrome_profiler = None

            t1 = time.time()
            dt = t1 - t0
            if startup_trace is not None:
                # This is intentionally adjacent to the existing post-sync
                # timer boundary: first_step_s is exactly the legacy dt, while
                # time_to_first_step_s uses the startup trace's wall clock.
                startup_trace.observe_step(
                    update_step=step,
                    step_s=dt,
                    compile_configured=bool(cfg.compile),
                )
            if step == 0 and cfg.compile:
                logger.info(
                    f"Step 0 complete (torch.compile graph tracing + first step): {dt:.1f}s"
                )

            # Logging
            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            if train_fp_loss is not None:
                smooth_fp_loss = ema_beta * smooth_fp_loss + (1 - ema_beta) * train_fp_loss.item()
            pct_done = 100 * step / num_iterations
            tok_per_sec = int(total_batch_size / dt)
            flops_per_sec = num_flops_per_token * total_batch_size / dt
            mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
            if step > 10:
                total_training_time += dt
            steps_done = step - 10
            if steps_done > 0:
                avg_time_per_step = total_training_time / steps_done
                remaining_steps = num_iterations - step
                eta_seconds = remaining_steps * avg_time_per_step
                eta_str = f" | eta: {eta_seconds / 60:.1f}m"
            else:
                eta_str = ""
            epoch = dataloader_state_dict["epoch"]
            logger.info(
                f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | dw: {data_wait_s * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | enc: {max_n_seg_seen}×{max_f_max_seen} | total time: {total_training_time / 60:.2f}m{eta_str}"
            )
            if step % 100 == 0:
                log_data = {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/data_wait_s": data_wait_s,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": epoch,
                }
                if train_fp_loss is not None:
                    log_data["train/feature_pred_loss"] = train_fp_loss.item()
                if conditioning_fraction is not None:
                    log_data["train/conditioning_fraction"] = conditioning_fraction
                    log_data["train/conditioning_dropout"] = 1.0 - conditioning_fraction
                wandb_run.log(log_data)

            # State update
            step += 1

            # GC management
            if step == 1:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif step % 5000 == 0:
                gc.collect()

    except BaseException as exc:
        training_error = exc
        if startup_trace is not None:
            startup_trace.finish(status="error")
        raise
    finally:
        try:
            _close_training_background_workers(
                batch_prefetch_loader=batch_prefetch_loader,
                cuda_staging_loader=cuda_staging_loader,
                remote_body_prefetcher=remote_body_prefetcher,
                checkpoint_publisher=checkpoint_publisher,
                active_error=training_error,
            )
        finally:
            # Restore GC
            gc.enable()
            if hasattr(gc, "unfreeze"):
                gc.unfreeze()

    # --- Clean up model grad buffers ---
    model.zero_grad(set_to_none=True)

    # --- Report ---
    peak_memory_mib = get_max_memory() / 1024 / 1024
    logger.info(f"Peak memory usage: {peak_memory_mib:.2f}MiB")
    logger.info(f"Total training time: {total_training_time / 60:.2f}m")
    if val_bpb is not None:
        logger.info(f"Minimum validation bpb: {min_val_bpb:.6f}")

    get_report(checkpoint_dir=checkpoint_dir).log(
        section="Encoder-decoder conditional model training",
        data=[
            user_config,
            {
                "Number of parameters": num_params,
                "Number of FLOPs per token": f"{num_flops_per_token:e}",
                "Calculated number of iterations": num_iterations,
                "Number of training tokens": total_tokens,
                "Tokens : Scaling params ratio": total_batch_size
                * num_iterations
                / num_scaling_params,
                "DDP world size": ddp_world_size,
                "warmup_ratio": cfg.warmup_ratio,
                "warmdown_ratio": cfg.warmdown_ratio,
                "final_lr_frac": cfg.final_lr_frac,
                "SAE features": n_sae_features,
                "Feature selection": (
                    f"{feature_selection.threshold_type.value}/{feature_selection.threshold}"
                    if isinstance(feature_selection, ThresholdSelection)
                    else f"sampling({feature_selection.distribution.value}, mean={feature_selection.mean}, std={feature_selection.std})"
                ),
                "Encoder config": OmegaConf.to_container(cfg.encoder, resolve=True),
                "Cross-attention config": OmegaConf.to_container(cfg.cross_attention, resolve=True),
            },
            {
                "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
                "Final validation bpb": val_bpb,
                "CORE metric estimate": results.get("core_metric", None),
                "MFU %": f"{mfu:.2f}%",
                "Total training flops": f"{flops_so_far:e}",
                "Total training time": f"{total_training_time / 60:.2f}m",
                "Peak memory usage": f"{peak_memory_mib:.2f}MiB",
            },
        ],
    )

    if startup_trace is not None:
        startup_trace.finish(
            status=(
                "complete"
                if startup_trace.first_update_step is not None
                else "no_training_step"
            )
        )

    return TrainResult(
        model_tag=output_dirname,
        val_bpb=val_bpb,
        min_val_bpb=min_val_bpb,
        total_training_time=total_training_time,
        peak_memory_mib=peak_memory_mib,
        startup_time=startup_time,
    )


# -----------------------------------------------------------------------------
# Entry point (guarded so the module can be safely imported in tests)

if __name__ == "__main__":
    startup_trace = StartupTrace()
    try:
        # Load Hydra config (compose API for torchrun compatibility)
        with _startup_phase(startup_trace, "hydra_config"):
            _config_dir = str(
                Path(__file__).resolve().parent.parent.parent / "configs" / "training"
            )
            _config_name, _overrides = parse_config_name_and_overrides(
                sys.argv[1:],
                default_config_name="conditional_encdec_train",
            )
            with initialize_config_dir(config_dir=_config_dir, version_base=None):
                cfg = compose(config_name=_config_name, overrides=_overrides)
        print_banner()
        shared = build_shared_state(cfg, startup_trace=startup_trace)

        # wandb init (standalone mode)
        with _startup_phase(startup_trace, "wandb_setup"):
            user_config = _resolved_user_config(cfg)
            master_process = shared.ddp_rank == 0
            use_dummy_wandb = cfg.run == "dummy" or not master_process
            wandb_run = (
                DummyWandb()
                if use_dummy_wandb
                else wandb.init(project="nanochat", name=cfg.run, config=user_config)
            )

        result = train_one_config(
            cfg,
            shared,
            wandb_run,
            startup_trace=startup_trace,
        )
        wandb_run.finish()
        compute_cleanup()

        if result.error:
            logger.error(f"Training failed: {result.error}")
            sys.exit(1)
    except BaseException:
        startup_trace.finish(status="error")
        raise
