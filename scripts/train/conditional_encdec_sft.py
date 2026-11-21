"""
Encoder-decoder SFT finetuning.

Provides `make_sft_data_generator` which produces batches for encoder-decoder
SFT training using BOS-bestfit packing over full nanochat-rendered sequences.

Every non-padding target token is trained on, and encoder inputs are expected
to carry full-sequence SFT features.

Run as:
    python -m scripts.train.conditional_encdec_sft dataset=safety model_tag=my_model

Or distributed:
    torchrun --nproc_per_node=8 -m scripts.train.conditional_encdec_sft -- dataset=safety model_tag=my_model
"""

import copy
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.distributed as dist
import wandb
from beartype import beartype
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

from errors import DataLoadError, FeatureClustersError, FeatureVocabError
from nanochat import flash_attention
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
)
from nanochat.gpt import GPTConfig, Linear
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.preprocess.convert_sft_to_parquet import (
    prepare_messages_for_full_nanochat_render,
    render_messages_to_full_nanochat_text,
)
from steering._common.compile_helpers import resolve_compile_dynamic
from steering._common.ddp import (
    broadcast_bitmap_from_rank_zero,
    rank_synchronized_stage,
    sync_stop_and_progress,
)
from steering._common.lr_schedules import sft_warmup_flat_decay
from steering._common.resume import resolve_resume_step
from steering._common.strict_json import string_keyed_mapping
from steering.conditional.checkpoint_conditioning import (
    assert_conditioning_supported,
    assert_pretrain_conditioning_recoverable,
    read_checkpoint_conditioning,
)
from steering.conditional.compaction_summary import format_compaction_summary
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    reject_removed_feature_prediction_keys,
    validate_feature_prediction_architecture,
    validate_feature_prediction_head_layers,
)
from steering.conditional.encoder_decoder.config import (
    feature_prediction_config_from_metadata as _resolve_fp_config_from_meta,
)
from steering.conditional.encoder_decoder.config import (
    serialize_feature_prediction_config as _serialize_fp_config,
)
from steering.conditional.encoder_decoder.dataloader import (
    ClosableBatchPrefetcher,
    SegFeatures,
    _pad_segments,
    _zero_masked_fp_targets,
    build_dataloader_state_envelope,
    select_dataloader_rank_state,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    FullSequenceSegmentRouting,
    build_full_sequence_segment_routing,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    compute_full_coverage_gap as _compute_full_coverage_gap,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    source_aligned_char_offsets as _source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.loss_eval import (
    accumulate_bpb_nats_bytes,
    reduce_bpb,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.encoder_decoder.sft_features import (
    SFTFeatureLoadReport,
    assert_sft_source_bos_char_len_matches,
    load_sft_features,
    scan_required_feature_id_bitmap,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    ONLINE_SAE_LINEAGE_KEY,
    build_online_sae_lineage_metadata,
)
from steering.conditional.encoder_decoder.sft_prediction_groups import (
    steered_prediction_groups,
)
from steering.conditional.encoder_decoder.sft_selection import (
    SegmentSelectionContext,
    select_conversation_segments,
)
from steering.conditional.encoder_decoder.sm12_fa4 import admit_project_attention
from steering.conditional.feature_clusters import (
    CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY,
    ClusteredThresholdSelectionSemantics,
    FeatureClusterConsolidation,
    append_feature_clusters_leaf,
    assert_resume_clustered_threshold_semantics_match,
    assert_resume_feature_clusters_match,
    feature_clusters_checkpoint_metadata,
    reject_feature_clusters_metadata,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import (
    FeatureMask,
    append_mask_leaf,
    assert_resume_feature_mask_matches,
    canonical_dict,
    resolve_feature_mask,
)
from steering.conditional.feature_pipeline import (
    ThresholdSelectionConfig,
    build_feature_pipeline,
    feature_selection_dirname,
    negative_sampling_mask,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
    feature_vocab_from_state_dict,
)
from steering.conditional.missing_percentile import (
    MISSING_PERCENTILE_POLICY_META_KEY,
    MissingPercentilePolicy,
    MissingPercentileResolver,
    assert_resume_missing_percentile_policy_matches,
    check_percentile_coverage,
    combined_coverage,
    enforce_percentile_coverage,
    parse_missing_percentile_policy,
    percentile_table_coverage,
)
from steering.conditional.percentile_table import (
    build_percentile_table as build_full_percentile_table,
)
from steering.conditional.sampling import (
    NegativeBundle,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
)
from steering.feature_io import glob_features
from steering.feature_paths import (
    STEERED_JSONL_PLACEHOLDER,
    build_feature_dir,
    parse_feature_dir,
    sft_phase_for,
)
from steering.profiling.markers import (
    TimingRecorder,
    timed_region,
    timing_tags,
    use_recorder,
)
from steering.refined_types import (
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveFiniteNumber,
    PositiveInt,
    PositiveIntLike,
)
from steering.removed_features import reject_feature_clustering_config
from steering.segmentation import SegmentMode
from steering.sft.clipping import apply_gradient_clipping, resolve_grad_clip_max_norm
from steering.sft.iteration import resolve_sft_loop_position
from steering.sft.loss_mask import (
    LossMaskMode,
    apply_completion_only_row_masks,
    assert_min_one_valid_target,
    assert_resume_sft_loss_mask_matches,
    parse_loss_mask_mode,
    render_conversation_with_mask,
    supervised_target_count,
    truncate_conv_and_mask,
)
from steering.sft.mixture_fingerprint import (
    MixtureFingerprint,
    compute_mixture_fingerprint,
    format_mixture_fingerprint_mismatch,
    sample_alignment_probe_indices,
)
from steering.sft.segment_mode_identity import (
    SFT_SEGMENT_MODE_META_KEY,
    assert_resume_sft_segment_mode_matches,
)
from steering.sft.turn_spans import ChatSpecialTokens, TurnRole, legacy_turn_spans
from steering.sft_schema import SFTSplit, sft_split_dir
from steering.stats_io import find_statistics_path, load_statistics
from tasks.mixtures import (
    STEERED_MIXTURES,
    SFTMixture,
    build_task_mixtures,
    sft_val_family,
    steered_mixture_overrides,
)
from tasks.steered import SteeredConversations, SteeredTaskMixture

SFT_CURSOR_SEMANTICS_BUFFERED_V2 = "buffered_v2"
SFT_FEATURE_PIPELINE_SEED = 42

# One conversation packed into a row: (conv_start, conv_end, seg_features,
# logical turn starts in source coordinates — None unless the conversation is
# steered —, buffer cursor position, prepared conditioned-input routing).
type _PackedConversation = tuple[
    int, int, list[SegFeatures], tuple[int, ...] | None, int, FullSequenceSegmentRouting
]


def _validate_sft_dataloader_state(
    state: Mapping[str, Any],
    *,
    ddp_rank: int,
    ddp_world_size: int,
    buffer_size: int,
    device_batch_size: int,
    sequence_length: int,
    grad_accum_steps: int,
    min_loss_tokens: int,
    loss_mask_mode: LossMaskMode,
) -> dict[str, Any]:
    """Validate and normalize one rank's buffered SFT cursor state."""
    expected_geometry: dict[str, int | str] = {
        "ddp_rank": ddp_rank,
        "ddp_world_size": ddp_world_size,
        "buffer_size": buffer_size,
        "device_batch_size": device_batch_size,
        "sequence_length": sequence_length,
        "grad_accum_steps": grad_accum_steps,
        "min_loss_tokens": min_loss_tokens,
        "loss_mask_mode": loss_mask_mode.value,
    }
    for key, expected in expected_geometry.items():
        actual = state[key]
        if type(actual) is not type(expected) or actual != expected:
            raise DataLoadError(
                f"conditioned SFT resume requires {key}={expected!r}, but the "
                f"checkpoint records {actual!r}"
            )

    consumed = NonNegativeInt.parse(state["consumed"])
    if consumed % ddp_world_size != 0:
        raise DataLoadError(
            "conditioned SFT dataloader state consumed must be divisible by "
            f"ddp_world_size={ddp_world_size}, got {consumed}"
        )
    next_cursor = NonNegativeInt.parse(state["next_cursor"])
    if next_cursor % ddp_world_size != ddp_rank:
        raise DataLoadError(
            f"conditioned SFT dataloader state next_cursor={next_cursor} does not "
            f"belong to rank {ddp_rank} under world size {ddp_world_size}"
        )
    buffered = state.get("buffered_cursor_positions")
    if not isinstance(buffered, list):
        raise DataLoadError(
            "conditioned SFT dataloader state buffered_cursor_positions must be a list"
        )
    if len(buffered) > buffer_size:
        raise DataLoadError(
            "conditioned SFT dataloader state contains more buffered positions "
            f"({len(buffered)}) than buffer_size={buffer_size}"
        )
    normalized_buffered: list[int] = []
    for value in buffered:
        position = NonNegativeInt.parse(value)
        if position >= next_cursor:
            raise DataLoadError(
                "conditioned SFT buffered cursor positions must precede next_cursor; "
                f"got position {position}, next_cursor {next_cursor}"
            )
        if position % ddp_world_size != ddp_rank:
            raise DataLoadError(
                f"conditioned SFT buffered cursor position {position} does not "
                f"belong to rank {ddp_rank} under world size {ddp_world_size}"
            )
        if normalized_buffered and position <= normalized_buffered[-1]:
            raise DataLoadError(
                "conditioned SFT buffered cursor positions must be strictly increasing"
            )
        normalized_buffered.append(position)

    attempted = (next_cursor - ddp_rank) // ddp_world_size
    accounted = consumed // ddp_world_size + len(normalized_buffered)
    if accounted > attempted:
        raise DataLoadError(
            "conditioned SFT dataloader state accounts for more consumed and "
            f"buffered conversations ({accounted}) than cursor attempts ({attempted})"
        )

    return {
        **expected_geometry,
        "consumed": consumed,
        "next_cursor": next_cursor,
        "buffered_cursor_positions": normalized_buffered,
    }


@dataclass
class SFTState:
    """Mutable state shared between the data generator and training loop.

    The generator updates ``consumed`` as conversations are popped from
    the packing buffer and publishes ``dataloader_state`` after constructing
    each batch. The training loop reads the former for progress and commits a
    snapshot of the latter only after the optimizer step succeeds.
    """

    consumed: int = field(default=0)
    dataloader_state: dict[str, Any] | None = field(default=None)
    dataset_size: int | None = field(default=None, init=False)
    artifact_coverage: dict[str, Any] | None = field(default=None, init=False)
    # Set by a ``single_pass`` generator once every covered document has been
    # served AND the packing buffer has drained. Only meaningful for the
    # validation generator; the training generator wraps forever by design.
    exhausted: bool = field(default=False, init=False)


@dataclass(frozen=True)
class _SFTStateSnapshot:
    """Generator state attached to one fully materialized SFT batch."""

    consumed: int
    dataloader_state: dict[str, Any] | None
    dataset_size: int | None
    artifact_coverage: dict[str, Any] | None
    exhausted: bool

    @classmethod
    def capture(cls, state: SFTState) -> "_SFTStateSnapshot":
        return cls(
            consumed=state.consumed,
            dataloader_state=copy.deepcopy(state.dataloader_state),
            dataset_size=state.dataset_size,
            artifact_coverage=copy.deepcopy(state.artifact_coverage),
            exhausted=state.exhausted,
        )

    def publish(self, state: SFTState) -> None:
        state.consumed = self.consumed
        state.dataloader_state = copy.deepcopy(self.dataloader_state)
        state.dataset_size = self.dataset_size
        state.artifact_coverage = copy.deepcopy(self.artifact_coverage)
        state.exhausted = self.exhausted


@dataclass(frozen=True)
class _PrefetchedSFTBatch:
    batch: tuple[Any, ...]
    state: _SFTStateSnapshot
    ready: Any | None
    timing_rows: tuple[dict[str, Any], ...] | None


def _prefetched_sft_batches(
    inner: Iterator[tuple[Any, ...]],
    *,
    producer_state: SFTState,
    producer_stream: Any | None,
    batch_count: int,
    timing_enabled: bool,
    timing_rank: int,
    timing_step: int,
    first_micro_step: int,
) -> Iterator[_PrefetchedSFTBatch]:
    """Attach state, CUDA readiness, and private timing to worker-built batches."""
    stream_context = (
        nullcontext() if producer_stream is None else t.cuda.stream(producer_stream)
    )
    with stream_context:
        batch_iterator = iter(inner)
        for micro_step in range(first_micro_step, first_micro_step + batch_count):
            batch_recorder = TimingRecorder() if timing_enabled else None
            try:
                if batch_recorder is None:
                    batch = next(batch_iterator)
                else:
                    with use_recorder(batch_recorder):
                        with timing_tags(
                            rank=timing_rank,
                            step=timing_step,
                            micro_step=micro_step,
                        ):
                            batch = next(batch_iterator)
            except StopIteration:
                return
            ready = None
            if producer_stream is not None:
                ready = t.cuda.Event(blocking=False)
                ready.record(producer_stream)
            yield _PrefetchedSFTBatch(
                batch=batch,
                state=_SFTStateSnapshot.capture(producer_state),
                ready=ready,
                timing_rows=(
                    None if batch_recorder is None else tuple(batch_recorder.rows())
                ),
            )


def _delivered_sft_batches(
    prefetched: Iterator[Any],
    *,
    consumer_state: SFTState,
    cuda_device: t.device | None,
    timing_recorder: TimingRecorder | None,
) -> Iterator[tuple[Any, ...]]:
    """Publish one batch's timing and matching state only when it is delivered."""
    for item in prefetched:
        if not isinstance(item, _PrefetchedSFTBatch):
            raise RuntimeError(f"unexpected SFT prefetch item: {item!r}")
        if item.ready is not None:
            consumer_stream = t.cuda.current_stream(cuda_device)
            item.ready.wait(consumer_stream)
            for value in item.batch:
                if getattr(value, "is_cuda", False):
                    value.record_stream(consumer_stream)
        if item.timing_rows is not None:
            if timing_recorder is None:
                raise RuntimeError("prefetched SFT timing rows have no destination recorder")
            for row in item.timing_rows:
                timing_recorder.append(
                    row["name"],
                    row["elapsed_ns"],
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"name", "elapsed_ns"}
                    },
                )
        item.state.publish(consumer_state)
        yield item.batch


def _resolve_sft_prefetch_batches(value: object) -> int:
    """Validate the SFT prefetch depth without coercing bools/floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"sft_prefetch_batches must be a strict non-negative int, got {value!r}"
        )
    return value


@contextmanager
def _sft_step_batch_iterator(
    inner: Iterator[tuple[Any, ...]],
    *,
    producer_state: SFTState,
    consumer_state: SFTState,
    batch_count: int,
    prefetch_batches: int,
    cuda_device: str | t.device | None,
    timing_recorder: TimingRecorder | None = None,
    timing_rank: int = 0,
    timing_step: int = 0,
    first_micro_step: int = 0,
):
    """Yield a finite iterator for one optimizer step's remaining batches.

    Feature selection and validation intentionally share RNGs. Limiting the
    producer to exactly ``batch_count`` items guarantees every RNG draw belongs
    to a batch consumed before validation or checkpointing can run.
    """
    if batch_count < 0:
        raise ValueError(f"batch_count must be non-negative, got {batch_count}")
    if prefetch_batches == 0 or batch_count == 0:
        yield inner
        return
    device = None if cuda_device is None else t.device(cuda_device)
    if device is not None and device.type != "cuda":
        raise ValueError(
            f"cuda_device must resolve to a CUDA device or None, got {cuda_device!r}"
        )
    producer_stream = None if device is None else t.cuda.Stream(device=device)
    producer = _prefetched_sft_batches(
        iter(islice(inner, batch_count)),
        producer_state=producer_state,
        producer_stream=producer_stream,
        batch_count=batch_count,
        timing_enabled=timing_recorder is not None,
        timing_rank=timing_rank,
        timing_step=timing_step,
        first_micro_step=first_micro_step,
    )
    prefetcher = ClosableBatchPrefetcher(
        producer,
        PositiveInt.parse(prefetch_batches),
        stop_callback=lambda: None,
        failure_check=lambda: None,
    )
    try:
        yield _delivered_sft_batches(
            prefetcher,
            consumer_state=consumer_state,
            cuda_device=device,
            timing_recorder=timing_recorder,
        )
    finally:
        prefetcher.close()


class _SFTValidationLoader(Iterator[Any]):
    """Validation iterator carrying the covered-document state it advances.

    ``evaluate_encdec_bpb`` requires this exact type rather than duck-typing a
    ``state`` attribute off whatever it is handed: before PR #1311 the factory
    returned a bare generator, and a refactor that reintroduced one would have
    silently restored wrap-around validation with no error at all (review
    finding 10).
    """

    def __init__(self, generator: Iterator[Any], state: SFTState) -> None:
        self._generator = generator
        self.state = state

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return next(self._generator)


_UNSUPPORTED_SFT_PRETRAIN_KEYS = (
    "adam_beta1",
    "adam_beta2",
    "aspect_ratio",
    "attn_impl",
    "conditioning_cooldown",
    "core_metric_every",
    "core_metric_max_per_task",
    "depth",
    "final_lr_frac",
    "fp8",
    "fp8_recipe",
    "head_dim",
    "sample_every",
    "scalar_lr",
    "target_flops",
    "target_param_data_ratio",
    "warmdown_ratio",
    "warmup_ratio",
    "window_pattern",
)


def _warn_if_forced_dynamic_breaks_fa4(compile_dynamic: bool | None) -> None:
    """Loudly flag the known-bad force-dynamic + FA4 combination.

    ``compile_static=false`` remains a legitimate escape hatch under the
    Triton/SDPA attention paths, but with the flex FLASH (CuteDSL) decoder
    backend active, torch>=2.13 rejects the symbolic mask_mod capture that
    whole-graph dynamic compilation produces, crashing the first compiled
    forward.  Warn at resolve time so the operator sees it before the
    doomed compile burns the remaining startup.
    """
    if compile_dynamic is not True:
        return
    if flash_attention._use_flex() and flash_attention.has_fa4_backend():
        logger.warning(
            "compile_static=false forces whole-graph dynamic compilation while "
            "the flex FLASH (CuteDSL) decoder backend is active; torch>=2.13 "
            "rejects the symbolic mask_mod capture this produces ('NYI: "
            "score_mod or mask_mod captures a dynamic scalar') at the first "
            "compiled forward. Use compile_static=auto (the default), or "
            "override NANOCHAT_ATTN_IMPL if force-dynamic is required."
        )


def _reject_unsupported_sft_pretrain_keys(cfg: Mapping[str, Any]) -> None:
    """Reject pretraining-only keys that Hydra can append or inherit."""
    for key in _UNSUPPORTED_SFT_PRETRAIN_KEYS:
        if key in cfg:
            raise ValueError(
                f"conditional_encdec_sft does not honour pretrain-only key {key!r}. "
                "Use scripts/train/conditional_encdec_train.py for FP8 and other "
                "pretraining-loop controls. Architecture values are reconstructed "
                "from checkpoint metadata by conditioned SFT."
            )


@t.no_grad()
@beartype
def evaluate_encdec_bpb(
    model,
    batches: _SFTValidationLoader,
    steps: int,
    token_bytes: t.Tensor,
) -> float:
    """Evaluate bits-per-byte for an encoder-decoder model.

    Like ``nanochat.loss_eval.evaluate_bpb`` but accepts SFT batches from
    ``make_sft_data_generator`` instead of plain ``(x, y)`` pairs.
    ``fp_targets`` and optional ``seg_doc_id`` metadata are discarded — eval
    measures the decoder LM loss only and does not feed the feature-prediction
    head.

    ``batches`` is annotated ``_SFTValidationLoader`` and the annotation is
    ENFORCED by ``@beartype`` — this function must not accept a duck.  The
    covered-document bound lives on that type; ``_build_sft_val_loader``
    returned a bare generator before PR #1311, and a refactor that reintroduced
    one would silently restore wrap-around validation, re-scoring documents
    already folded into the BPB sums with no error at all (review finding 10).
    ``@beartype`` raises ``BeartypeCallHintParamViolation`` there, so the
    contract cannot be stripped by deleting a hand-rolled ``isinstance``.

    The BPB arithmetic itself is pretraining's:
    ``accumulate_bpb_nats_bytes`` per batch and ``reduce_bpb`` at the end, the
    same two helpers ``evaluate_encdec_bpb_conditioned`` calls.  SFT reproduced
    them inline, which is how ``val/bpb`` and ``val/bpb_selfcond`` would have
    drifted apart and stopped being comparable on one plot.

    The loop stops once the validation generator has served every covered
    document.  That decision is made COLLECTIVELY: each rank's generator packs
    a different number of conversations per batch, so a rank-local ``break``
    would leave the loop's iteration count rank-dependent and desynchronise
    every subsequent collective — including the aligned batch-construction
    stage. ``sync_last_step`` all-reduces the flag first and stops only once
    EVERY rank is exhausted, so every rank agrees before any other collective
    in the iteration without dropping the tail of a slower rank's slice.
    Already-exhausted ranks keep participating with the generator's fully
    masked padded batches, which contribute neither nats nor bytes. The bound
    itself stays SFT-specific: pretraining's eval is a plain ``for _ in
    range(eval_steps)`` with no early exit, so there is nothing to share.
    """
    val_state = batches.state
    device = model.get_device()
    total_nats = t.tensor(0.0, dtype=t.float32, device=device)
    total_bytes = t.tensor(0, dtype=t.int64, device=device)
    batch_iter = iter(batches)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ddp = world_size > 1
    for _ in range(steps):
        if sync_last_step(val_state.exhausted, device, ddp):
            break
        with rank_synchronized_stage(
            "conditioned SFT validation batch construction",
            device=device,
        ):
            batch = next(batch_iter)
        if len(batch) == 8:
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets, _seg_doc_id = batch
        else:
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
        loss2d = model(
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            loss_reduction="none",
        )
        batch_nats, batch_bytes = accumulate_bpb_nats_bytes(
            loss2d.reshape(-1),
            targets.reshape(-1),
            token_bytes,
        )
        total_nats += batch_nats
        total_bytes += batch_bytes

    return reduce_bpb(total_nats, total_bytes)


def sync_last_step(last_step: bool, device: str, ddp: bool) -> bool:
    """Synchronize the last_step flag across all DDP ranks.

    Uses ``ReduceOp.MIN`` so that all ranks stop only after EVERY rank has
    finished. Exhausted ranks keep yielding fully masked padding until then;
    stopping when any rank finishes would silently drop slower ranks' tails.
    """
    if not ddp:
        return last_step
    last_step_tensor = t.tensor(last_step, dtype=t.int32, device=device)
    dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MIN)
    return bool(last_step_tensor.item())


def _should_run_val_eval(*, has_val: bool, eval_every: int, step: int, last_step: bool) -> bool:
    """Validation runs only when enabled; eval_every=0 is a total opt-out (#1278)."""
    return has_val and eval_every > 0 and (last_step or step % eval_every == 0)


def create_wandb_run(run_name: str, master_process: bool, user_config: dict):
    """Create a wandb run or DummyWandb for non-logging contexts."""
    if run_name == "dummy" or not master_process:
        return DummyWandb()
    return wandb.init(project="nanochat-encdec-sft", name=run_name, config=user_config)


def _validate_sft_lr_schedule(
    *, warmup_steps: object, num_iterations: object
) -> tuple[int, int | None]:
    """Validate schedule identity before device/model initialization or restore."""
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError(
            f"lr_warmup_steps must be a nonnegative integer (0 disables); got {warmup_steps!r}"
        )
    if type(num_iterations) is not int or (num_iterations != -1 and num_iterations <= 0):
        raise ValueError(
            "num_iterations must be -1 or a positive integer optimizer-step count; "
            f"got {num_iterations!r}"
        )
    if num_iterations == -1:
        return warmup_steps, None
    if 5 * warmup_steps >= 4 * num_iterations:
        raise ValueError(
            "lr_warmup_steps must finish strictly before 80% decay: "
            f"require 5 * lr_warmup_steps < 4 * num_iterations; "
            f"got lr_warmup_steps={warmup_steps}, num_iterations={num_iterations}. "
            "Reduce lr_warmup_steps, or set lr_warmup_steps=0 to disable warmup."
        )
    return warmup_steps, num_iterations


def _assert_resume_sft_lr_schedule_matches(
    meta: dict, *, warmup_steps: int, num_iterations: int | None
) -> None:
    """Pin both schedule axes; a missing legacy warmup means the historical zero."""
    saved = meta.get("user_config")
    if not isinstance(saved, dict):
        raise ValueError("SFT resume requires a user_config mapping with num_iterations")
    saved_warmup, saved_iterations = _validate_sft_lr_schedule(
        warmup_steps=saved.get("lr_warmup_steps", 0),
        num_iterations=saved.get("num_iterations"),
    )
    if warmup_steps != saved_warmup or num_iterations != saved_iterations:
        raise ValueError(
            "SFT resume LR schedule mismatch: exact continuation requires "
            f"lr_warmup_steps={saved_warmup} and num_iterations={saved['num_iterations']}. "
            "A legacy checkpoint without lr_warmup_steps requires lr_warmup_steps=0. "
            "To change the schedule, start a fresh SFT phase from model weights."
        )


def _setup_sft_optimizer(model, cfg):
    """Build the SFT optimizer and establish every schedule base LR."""
    fp_head_lr = cfg.get("fp_head_lr")
    if fp_head_lr is not None:
        fp_head_lr = PositiveFiniteNumber.parse(fp_head_lr) * cfg.lr_multiplier
    fp_head_matrix_lr = (
        PositiveFiniteNumber.parse(cfg.fp_head_matrix_lr) * cfg.lr_multiplier
    )
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.unembedding_lr * cfg.lr_multiplier,
        embedding_lr=cfg.embedding_lr * cfg.lr_multiplier,
        matrix_lr=cfg.matrix_lr * cfg.lr_multiplier,
        weight_decay=cfg.weight_decay,
        encoder_embedding_lr=cfg.encoder_embedding_lr * cfg.lr_multiplier,
        encoder_matrix_lr=cfg.encoder_matrix_lr * cfg.lr_multiplier,
        xattn_matrix_lr=cfg.xattn_matrix_lr * cfg.lr_multiplier,
        fp_head_lr=fp_head_lr,
        fp_head_matrix_lr=fp_head_matrix_lr,
        muon_parameter_packing=cfg.get("muon_parameter_packing", False),
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * cfg.init_lr_frac
        group["initial_lr"] = group["lr"]
    return optimizer


def _resolve_sft_parquet_dir(override: str | None, base_dir: str, dataset: str) -> str:
    """Resolve the SFT conversation parquet directory.

    This is where ``make_sft_data_generator`` looks for the sharded
    full-nanochat-text parquets produced by
    ``scripts/preprocess/convert_sft_to_parquet.py``. The loader requires the
    ``original_index`` column to re-key features back to task examples.

    Args:
        override: the value of ``cfg.sft_parquet_dir``.  ``None``
            (the default) means "use ``{base_dir}/sft_data/{dataset}``";
            any other string is returned as-is so operators can point at
            arbitrary filesystem locations (shared NFS mounts,
            scratch dirs, alternate mixtures, etc.).
        base_dir: the run's base directory (``get_base_dir()``),
            used only when ``override is None`` to construct the
            default parquet path.
        dataset: the run's ``cfg.dataset`` string, used only when
            ``override is None``.

    Returns:
        An absolute filesystem path to the SFT parquet directory.
    """
    if override is not None:
        return override
    return os.path.join(base_dir, "sft_data", dataset)


def _resolve_sft_val_parquet_dir(override: str | None, base_dir: str, dataset: str) -> str:
    """Resolve validation parquet independently from the train conversion."""
    if override is not None:
        return override
    train_dir = os.path.join(base_dir, "sft_data", dataset)
    return str(sft_split_dir(train_dir, SFTSplit.VAL))


def _resolve_val_feature_dir(
    override: str | None, feature_dir: str, dataset: str, val_dataset: str
) -> str:
    """Derive the validation feature path through the canonical path builder.

    ``dataset`` identifies the TRAIN phase the feature dir must carry;
    ``val_dataset`` is its validation family (``tasks.mixtures.sft_val_family``),
    which is what names the val phase.  They differ only for the steered
    mixtures, which validate on standard's own val artifacts.
    """
    if override is not None:
        return override
    try:
        components = parse_feature_dir(feature_dir)
    except ValueError as exc:
        raise DataLoadError(
            f"Cannot derive val_feature_dir from feature_dir={feature_dir!r}: {exc}. "
            "Set val_feature_dir explicitly or use a canonical feature path."
        ) from exc

    try:
        train_phase = sft_phase_for(dataset, SFTSplit.TRAIN)
        val_phase = sft_phase_for(val_dataset, SFTSplit.VAL)
    except ValueError:
        raise DataLoadError(
            f"Cannot derive val_feature_dir for dataset={dataset!r}: no matching "
            "train/val DatasetPhase pair exists. Set val_feature_dir explicitly."
        ) from None
    if components.dataset_phase is not train_phase:
        raise DataLoadError(
            f"Cannot derive val_feature_dir for dataset={dataset!r}: "
            f"feature_dir has phase={components.dataset_phase.value!r}, expected "
            f"{train_phase.value!r}. Set val_feature_dir explicitly if intentional."
        )

    root = Path(feature_dir)
    component_count = 7
    if components.context_independent_features:
        component_count += 1
    if components.ignore_bos_features:
        component_count += 1
    if components.segment_mode is not SegmentMode.fixed_tokens:
        component_count += 1
    for _ in range(component_count):
        root = root.parent

    return str(
        build_feature_dir(
            root=root,
            segment_size=components.segment_size,
            layer=components.layer,
            activation_type=components.activation_type,
            pooling=components.pooling,
            width=components.width,
            model=components.model,
            dataset_phase=val_phase,
            context_independent_features=components.context_independent_features,
            ignore_bos_features=components.ignore_bos_features,
            segment_mode=components.segment_mode,
        )
    )


def _task_steered_jsonl(task: object) -> str | None:
    """The steered JSONL a built mixture came from, for recovery commands."""
    steered = task.tasks[-1] if isinstance(task, SteeredTaskMixture) else None
    return steered.filepath if isinstance(steered, SteeredConversations) else None


def _sft_convert_command(dataset: str, split: SFTSplit, steered_jsonl: str | None) -> str:
    """The conversion command for one mixture/split, override included.

    A steered mixture cannot convert without its JSONL, so the command always
    carries ``steered_jsonl=`` — the operator's own path when the caller knows
    it, the placeholder otherwise — rather than a line that fails on paste.
    """
    command = (
        "uv run python -m scripts.preprocess.convert_sft_to_parquet "
        f"mixture={dataset} split={split.value}"
    )
    if dataset not in STEERED_MIXTURES:
        return command
    return f"{command} steered_jsonl={steered_jsonl or STEERED_JSONL_PLACEHOLDER}"


def _sft_artifact_recovery_instructions(
    dataset: str, *, split: SFTSplit, steered_jsonl: str | None = None
) -> str:
    convert = _sft_convert_command(dataset, split, steered_jsonl)
    try:
        phase = sft_phase_for(dataset, split)
    except ValueError:
        return (
            "Recovery commands:\n"
            f"  {convert}\n"
            "  Feature extraction command omitted: dataset has no known SFT phase.\n"
            "No separate statistics run is needed for val: train statistics govern "
            "validation discretization. Set eval_every=0 to explicitly opt out of validation."
        )
    return (
        "Recovery commands:\n"
        f"  {convert}\n"
        "  uv run torchrun --standalone --nproc_per_node=4 "
        f"-m steering.extract_features dataset_phase={phase.value}\n"
        "No separate statistics run is needed for val: train statistics govern "
        "validation discretization. Set eval_every=0 to explicitly opt out of validation."
    )


def _infer_sft_mixture_from_parquet_dir(parquet_dir: str) -> str:
    """Infer a supported mixture without parsing the validation suffix."""

    parquet_path = Path(parquet_dir)
    for mixture in SFTMixture:
        train_candidate = parquet_path.with_name(mixture.value)
        for split in SFTSplit:
            if sft_split_dir(train_candidate, split) == parquet_path:
                return mixture.value
    return parquet_path.name


def _require_sft_val_artifacts(*, dataset: str, parquet_dir: str, feature_dir: str) -> None:
    """Fail at startup unless both validation artifact families are populated."""
    parquet_path = Path(parquet_dir)
    feature_path = Path(feature_dir)
    parquet_ready = parquet_path.is_dir() and any(parquet_path.glob("shard_*.parquet"))
    feature_ready = feature_path.is_dir() and bool(glob_features(feature_path))
    if parquet_ready and feature_ready:
        return
    raise DataLoadError(
        "Conditioned SFT validation artifacts are missing or empty. "
        f"Resolved val parquet dir: {parquet_dir}. "
        f"Resolved val feature dir: {feature_dir}. "
        f"parquet_ready={parquet_ready}, feature_ready={feature_ready}.\n"
        + _sft_artifact_recovery_instructions(dataset, split=SFTSplit.VAL)
    )


def _resolve_and_require_sft_val_artifacts(
    *,
    has_val: bool,
    eval_every: int,
    val_parquet_override: str | None,
    val_feature_override: str | None,
    base_dir: str,
    feature_dir: str,
    dataset: str,
    val_dataset: str,
) -> tuple[str | None, str | None]:
    """Resolve + preflight val artifacts only when validation is enabled.

    Every val path and every recovery command names ``val_dataset`` (the
    mixture's validation family), while ``dataset`` still identifies the train
    phase the feature dir must carry.
    """
    if not has_val or eval_every <= 0:
        return None, None
    parquet_dir = _resolve_sft_val_parquet_dir(
        val_parquet_override,
        base_dir,
        val_dataset,
    )
    resolved_feature_dir = _resolve_val_feature_dir(
        val_feature_override,
        feature_dir,
        dataset,
        val_dataset,
    )
    _require_sft_val_artifacts(
        dataset=val_dataset,
        parquet_dir=parquet_dir,
        feature_dir=resolved_feature_dir,
    )
    return parquet_dir, resolved_feature_dir


@beartype
def _configure_min_loss_tokens(value: PositiveIntLike) -> int:
    """Validate + log the per-conversation supervised-token floor."""
    validated = int(value)
    logger.info(
        f"min_loss_tokens={validated}: conversations with fewer "
        "supervised loss tokens after truncation are dropped (train "
        "and validation use the same threshold)"
    )
    return validated


@beartype
def _configure_alignment_probe_samples(value: NonNegativeIntLike) -> int:
    """Validate + log the deterministic startup content-probe sample count."""
    validated = int(value)
    if validated == 0:
        logger.info("alignment_probe_samples=0: SFT startup content probe disabled")
    else:
        logger.info(
            f"alignment_probe_samples={validated}: train and validation generators "
            "will byte-compare deterministic task/parquet samples at startup"
        )
    return validated


def _assert_sft_mixture_fingerprint_matches_task(
    *,
    store,
    task,
    mixture_name: str,
    expected_split: SFTSplit,
    feature_dir: str,
    parquet_dir: str,
    alignment_probe_samples: int,
) -> MixtureFingerprint | None:
    """Apply the legacy/half-migrated/mismatch transition matrix at startup."""
    parquet_fingerprint = getattr(store, "parquet_mixture_fingerprint", None)
    feature_fingerprint = getattr(store, "feature_mixture_fingerprint", None)
    if parquet_fingerprint is None and feature_fingerprint is None:
        if alignment_probe_samples == 0:
            raise DataLoadError(
                "conditioned SFT legacy SFT artifacts have no mixture fingerprint "
                "on either side and alignment_probe_samples=0, leaving no alignment "
                "verification at all; regenerate the parquet and feature artifacts "
                "in fresh output directories to get fingerprints, or set "
                "alignment_probe_samples > 0"
            )
        fresh_parquet_dir = f"out/data/sft_data/{mixture_name}_fingerprinted"
        convert = _sft_convert_command(
            mixture_name, expected_split, _task_steered_jsonl(task)
        )
        logger.warning(
            "conditioned SFT mixture fingerprint is absent from both parquet "
            "conversion metadata and feature shard metadata; using the "
            "documented legacy escape hatch. Regenerate with "
            f"`{convert} output_dir={fresh_parquet_dir}` and re-extract features "
            "into a fresh output directory. Alignment protection for this run "
            "rests on the independent startup content probe alone."
        )
        return None
    if (parquet_fingerprint is None) != (feature_fingerprint is None):
        raise DataLoadError(
            "conditioned SFT mixture fingerprint present on exactly one side "
            "(half-migrated state is not legacy): "
            f"parquet_fingerprint={parquet_fingerprint!r}, "
            f"feature_shard_fingerprint={feature_fingerprint!r}; "
            f"feature_dir={feature_dir!r}; parquet_dir={parquet_dir!r}"
        )
    assert parquet_fingerprint is not None
    assert feature_fingerprint is not None
    if parquet_fingerprint != feature_fingerprint:
        raise DataLoadError(
            "conditioned SFT parquet-vs-feature mixture fingerprint mismatch: "
            + format_mixture_fingerprint_mismatch(
                "parquet_fingerprint",
                parquet_fingerprint,
                "feature_shard_fingerprint",
                feature_fingerprint,
            )
        )

    task_fingerprint = compute_mixture_fingerprint(
        task,
        mixture_name=mixture_name,
        split=expected_split,
    )
    recorded_length = int(parquet_fingerprint["len_mixture"])
    if recorded_length != len(task):
        raise DataLoadError(
            "conditioned SFT task mixture fingerprint mismatch: coverage-length "
            f"guard recorded len_mixture={recorded_length} but len(task)={len(task)}; "
            + format_mixture_fingerprint_mismatch(
                "store_fingerprint",
                parquet_fingerprint,
                "task_fingerprint",
                task_fingerprint,
            )
        )
    if parquet_fingerprint != task_fingerprint:
        raise DataLoadError(
            "conditioned SFT task mixture fingerprint mismatch: "
            + format_mixture_fingerprint_mismatch(
                "store_fingerprint",
                parquet_fingerprint,
                "task_fingerprint",
                task_fingerprint,
            )
        )
    return parquet_fingerprint


def _prepare_sft_render_messages(
    conversation: dict,
    *,
    surface: str,
    original_index: int,
) -> list[dict]:
    """Return the converter's rendered message slice, or fail loud.

    ``convert_sft_to_parquet`` truncates every conversation after its last
    assistant message and drops rows with no assistant turn or an empty final
    assistant response.  The parquet ``text`` and the extracted segment char
    offsets are both built from that slice, so every live render here must use
    the same slice — a trailing user turn otherwise adds at least 22 decoder
    characters that no feature segment covers (issue #1465).

    A converter-skipped row cannot appear in a consistent artifact family: the
    converter never writes one, and a legacy family containing one already dies
    on the missing-features guard before any render.  Reaching one here means
    the live task drifted from the conversion behind a composition-only mixture
    fingerprint, so raise rather than silently skipping the row.
    """
    messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
    if messages is None:
        raise DataLoadError(
            f"conditioned SFT {surface} cannot render "
            f"original_index={original_index}: the live task row is skipped by "
            f"the converter ({skip_reason}), so no parquet text or feature row "
            "can correspond to it. Regenerate the matched parquet and feature "
            "artifacts from the live task mixture."
        )
    return messages


def _run_sft_content_alignment_probe(
    *,
    store,
    task,
    tokenizer,
    alignment_probe_samples: int,
    probe_indices: Sequence[int] | None = None,
) -> list[int]:
    """Byte-compare deterministic task renders with sampled parquet rows."""
    candidates: Sequence[int] = range(len(task)) if probe_indices is None else probe_indices
    sample_positions = sample_alignment_probe_indices(
        len(candidates), alignment_probe_samples
    )
    indices = [candidates[position] for position in sample_positions]
    if not indices:
        logger.info("conditioned SFT startup content probe disabled (0 samples)")
        return indices
    load_texts = getattr(store, "load_texts", None)
    if not callable(load_texts):
        raise DataLoadError(
            "conditioned SFT startup content probe cannot run: feature store has "
            "no sampled parquet text lookup"
        )
    parquet_texts = load_texts(indices)
    for original_index in indices:
        messages = _prepare_sft_render_messages(
            task[original_index],
            surface="startup content probe",
            original_index=original_index,
        )
        task_text = render_messages_to_full_nanochat_text(tokenizer, messages)
        parquet_text = parquet_texts.get(original_index)
        if parquet_text is None:
            raise DataLoadError(
                "conditioned SFT startup content probe missing parquet text for "
                f"original_index={original_index}"
            )
        if task_text.encode("utf-8") != parquet_text.encode("utf-8"):
            raise DataLoadError(
                "conditioned SFT content alignment probe mismatch at "
                f"original_index={original_index}: "
                f"task_text_prefix={task_text[:80]!r}; "
                f"parquet_text_prefix={parquet_text[:80]!r}"
            )
    logger.info(
        f"conditioned SFT startup content probe matched {len(indices)} rows: {indices}"
    )
    return indices


def _build_sft_checkpoint_dir(checkpoint_dir: str, dataset: str) -> str:
    """Return the absolute path of the SFT checkpoint leaf directory.

    SFT is conditioned-only, so runs land in
    ``{dataset}_sft_conditioned_checkpoints``.
    """
    return os.path.join(checkpoint_dir, f"{dataset}_sft_conditioned_checkpoints")


_REMOVED_SFT_FREQUENCY_KEYS = (
    "feature_mask_freq_min",
    "feature_mask_freq_max",
)
_PERCENTILE_ONLY_CLUSTER_THRESHOLD_TYPES = frozenset(
    {
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    }
)


def _reject_removed_sft_frequency_keys(cfg: Mapping[str, Any]) -> None:
    """Reject removed frequency-producer knobs after Hydra composition."""
    for key in _REMOVED_SFT_FREQUENCY_KEYS:
        if key in cfg:
            raise ValueError(
                f"conditional_encdec_sft no longer accepts {key}: SFT consumes "
                "the base checkpoint's exact explicit feature_mask. Recover it "
                "with: uv run python -m scripts.analysis.dump_feature_mask "
                "<checkpoint_dir>"
            )


def _validate_feature_clusters_sft_selection(
    cfg: Mapping[str, Any],
    feature_selection: ThresholdSelection | SamplingSelection,
) -> None:
    """Reject raw-magnitude threshold semantics in clustered SFT."""
    if cfg.get("feature_clusters") is None or not isinstance(feature_selection, ThresholdSelection):
        return
    if feature_selection.threshold_type in _PERCENTILE_ONLY_CLUSTER_THRESHOLD_TYPES:
        return
    supported = sorted(item.value for item in _PERCENTILE_ONLY_CLUSTER_THRESHOLD_TYPES)
    raise FeatureClustersError(
        f"Threshold type {feature_selection.threshold_type!r} requires raw feature "
        "magnitudes, but clustered SFT selection only has percentiles; raw-magnitude "
        f"representative semantics are undefined. Supported types: {supported}."
    )


def _save_sft_checkpoint(
    *,
    dry_run: bool,
    checkpoint_dir: str,
    step: int,
    model: t.nn.Module,
    optimizer: t.optim.Optimizer,
    meta: dict[str, Any],
    rank: int,
) -> None:
    """Persist one SFT checkpoint unless checkpoint output is disabled."""
    if dry_run:
        return
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        optimizer.state_dict(),
        meta,
        rank=rank,
    )


def _assert_sft_universe_matches_checkpoint_vocab(
    feature_universe: FeatureClusterConsolidation | FeatureMask | None,
    feature_mask: FeatureMask | None,
    checkpoint_vocab: FeatureVocab | None,
    n_sae_features: int,
    *,
    checkpoint_dir: str,
    base_meta: Mapping[str, Any] | None,
) -> None:
    """The SFT effective universe must EQUAL the base checkpoint's universe.

    Under compaction the vocabulary is a property of the **checkpoint**, not of the
    SFT config: the encoder has one embedding row per allowed feature and nothing
    else. SFT inherits that dictionary and can neither create nor widen it.

    Two deliberate policy choices, both owner decisions:

    * **Equality, not subset.** A subset would be structurally safe — every id would
      still have a row — so this is a stricter policy than the structural minimum. It
      is the consistent one: it matches ``assert_resume_feature_mask_matches``, which
      already demands exact equality on resume, and the documented rule that a model
      trained with a ``feature_mask`` must be evaluated with the same one.
    * **Nothing is inferred.** ``feature_mask: null`` against a compacted checkpoint
      RAISES; the checkpoint's vocabulary is never silently adopted as the SFT
      universe.

    Uncompacted clustered checkpoints always raise. A pre-compaction masked base may
    continue only when its canonical mask metadata exactly matches the configured
    compacting mask; ordinary unmasked legacy checkpoints have no such key and remain
    a byte-identical no-op.
    """
    sft_vocab = effective_feature_vocab(feature_universe, n_sae_features)
    clustered = isinstance(feature_universe, FeatureClusterConsolidation)
    configured_mask_canonical = canonical_dict(feature_mask)
    meta = base_meta or {}

    if checkpoint_vocab is None:
        if clustered:
            clustered_universe = feature_universe
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} has no compacted feature vocabulary, "
                "but configured feature_clusters "
                f"descriptor={clustered_universe.resolved.descriptor_dict()!r} with "
                f"feature_mask={configured_mask_canonical!r}. Cluster-trained bases are "
                "compacted by construction, so SFT cannot verify or introduce clusters "
                "against this uncompacted checkpoint."
            )
        if sft_vocab is not None:
            if "feature_mask_canonical" not in meta:
                raise FeatureVocabError(
                    f"base checkpoint {checkpoint_dir} has no compacted feature "
                    "vocabulary, and the configured feature_mask cannot be verified "
                    "against this pre-compaction checkpoint because canonical mask "
                    "metadata is absent. SFT cannot introduce or guess a compacting "
                    "mask. Recover the checkpoint contract with: uv run python -m "
                    f"scripts.analysis.dump_feature_mask {checkpoint_dir}"
                )
            if meta["feature_mask_canonical"] != configured_mask_canonical:
                raise FeatureVocabError(
                    f"base checkpoint {checkpoint_dir} feature_mask_canonical metadata "
                    "disagrees with the configured SFT feature_mask: checkpoint records "
                    f"{meta['feature_mask_canonical']!r}, configured "
                    f"{configured_mask_canonical!r}. SFT inherits the base checkpoint's "
                    "feature universe; present canonical metadata must agree "
                    "bidirectionally, even for a full-width state dict."
                )
            logger.info(
                f"base checkpoint {checkpoint_dir} is a pre-compaction masked base; "
                "canonical mask metadata matches the configured mask"
            )
            return
        if (
            "feature_mask_canonical" in meta
            and meta["feature_mask_canonical"] != configured_mask_canonical
        ):
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} feature_mask_canonical metadata "
                f"disagrees with the configured SFT feature_mask: checkpoint records "
                f"{meta['feature_mask_canonical']!r}, configured "
                f"{configured_mask_canonical!r}. SFT inherits the base checkpoint's "
                "feature universe; present canonical metadata must agree "
                "bidirectionally, even for a full-width state dict."
            )
        return

    if sft_vocab != checkpoint_vocab:
        # ``effective_feature_vocab`` returns None for BOTH "no mask" and an
        # identity effective universe. Neither can describe a compacted encoder.
        if sft_vocab is None and not clustered:
            configured = (
                "no feature_mask is configured"
                if feature_mask is None
                else f"the configured feature_mask (mode={feature_mask.mode.value}) allows "
                f"ALL {n_sae_features} features"
            )
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} is COMPACTED — its encoder has one "
                f"embedding row for each of {checkpoint_vocab.size} allowed features (of "
                f"{checkpoint_vocab.n_sae_features}) and no row for any other feature — but "
                f"{configured}. The checkpoint's vocabulary is never silently adopted as "
                "the SFT universe: configure the SAME feature_mask the checkpoint was "
                "trained with. Recover it with: uv run python -m "
                f"scripts.analysis.dump_feature_mask {checkpoint_dir}"
            )
        if sft_vocab is None:
            assert isinstance(feature_universe, FeatureClusterConsolidation)
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} is COMPACTED with "
                f"{checkpoint_vocab.size} feature rows, but configured feature_clusters "
                f"descriptor={feature_universe.resolved.descriptor_dict()!r} is identity "
                f"after feature_mask={configured_mask_canonical!r} and therefore expects "
                f"all {n_sae_features} rows. The cluster descriptor participates in the "
                "effective SFT universe, which must EQUAL the checkpoint vocabulary. "
                "Recover the checkpoint contract with: uv run python -m "
                f"scripts.analysis.dump_feature_mask {checkpoint_dir}"
            )

        missing = sorted(set(sft_vocab.original_ids) - set(checkpoint_vocab.original_ids))
        extra = sorted(set(checkpoint_vocab.original_ids) - set(sft_vocab.original_ids))
        cluster_participation = (
            " Static feature_clusters descriptor "
            f"{feature_universe.resolved.descriptor_dict()!r} participates in this "
            "effective universe."
            if clustered
            else ""
        )
        mask_description = (
            f"the mask (mode={feature_mask.mode.value})"
            if feature_mask is not None
            else "the effective no-mask universe"
        )
        raise FeatureVocabError(
            f"the SFT feature_mask does not match the vocabulary of base checkpoint "
            f"{checkpoint_dir}: {mask_description} allows "
            f"{sft_vocab.size} features, the checkpoint's encoder was trained on "
            f"{checkpoint_vocab.size}. "
            f"{len(missing)} allowed by the mask are absent from the checkpoint's "
            f"vocabulary (e.g. {missing[:8]}); {len(extra)} in the checkpoint's vocabulary "
            f"are excluded by the mask (e.g. {extra[:8]}). "
            "The allowed universe must EQUAL the checkpoint's vocabulary — a subset is not "
            "enough, because a compacted checkpoint's rows only mean anything relative to "
            "the exact mask it was trained with."
            f"{cluster_participation} Recover that mask with: uv run python -m "
            f"scripts.analysis.dump_feature_mask {checkpoint_dir}"
        )

    if "feature_mask_canonical" in meta:
        if meta["feature_mask_canonical"] != configured_mask_canonical:
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} feature_mask_canonical metadata "
                f"disagrees with the configured SFT feature_mask despite an equal "
                f"state-dict vocabulary: checkpoint records "
                f"{meta['feature_mask_canonical']!r}, configured "
                f"{configured_mask_canonical!r}. Writers must never be weaker than "
                "readers; refusing the hand-edited or inconsistent artifact."
            )
    else:
        logger.info(
            f"base checkpoint {checkpoint_dir} is compacted but "
            "feature_mask_canonical metadata is absent; using the state-dict feature "
            "vocabulary as the authority"
        )


_FEATURE_PREDICTION_WIDTH_META_KEY = "feature_prediction_width"


def _feature_prediction_layout_checkpoint_metadata(
    layout: FeaturePredictionTargetLayout | None,
) -> dict[str, int]:
    """Return the SFT checkpoint mirror of an active FP target width."""
    if layout is None:
        return {}
    return {_FEATURE_PREDICTION_WIDTH_META_KEY: layout.output_width}


def _assert_resume_feature_prediction_layout_matches(
    layout: FeaturePredictionTargetLayout | None,
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Reject missing, partial, or changed SFT compact-FP resume identity.

    Old full-width FP checkpoints predate the width mirror and remain valid.
    Compact FP has no such compatibility case: before issue #1276 SFT refused
    to create those checkpoints, so a compact resume must carry both the
    vocabulary fingerprint and the target width written by this implementation.
    """
    meta = checkpoint_meta or {}
    width_present = _FEATURE_PREDICTION_WIDTH_META_KEY in meta
    vocab_present = "feature_vocab" in meta

    if layout is None:
        if width_present:
            raise ValueError(
                f"SFT resume checkpoint {checkpoint_dir} records "
                f"{_FEATURE_PREDICTION_WIDTH_META_KEY} but the base checkpoint "
                "has no feature-prediction head"
            )
        return

    if layout.is_compact:
        compact_fp_keys = {
            "feature_vocab",
            _FEATURE_PREDICTION_WIDTH_META_KEY,
        }
        present = compact_fp_keys.intersection(meta)
        if present != compact_fp_keys:
            raise ValueError(
                f"SFT resume checkpoint {checkpoint_dir} compact-FP metadata is "
                "missing or partial: expected keys="
                f"{sorted(compact_fp_keys)}, present={sorted(present)}. "
                "A pre-#1276 compact-FP SFT checkpoint cannot exist because the "
                "old trainer refused to create one; start fresh or restore the "
                "original complete checkpoint metadata."
            )
    else:
        if vocab_present:
            raise ValueError(
                f"SFT resume checkpoint {checkpoint_dir} records a compact feature "
                "vocabulary but the configured FP target layout is full-width"
            )
        if not width_present:
            # Compatibility with SFT checkpoints written before the width mirror.
            return

    saved_width = meta[_FEATURE_PREDICTION_WIDTH_META_KEY]
    if saved_width != layout.output_width:
        raise FeatureVocabError(
            f"SFT resume checkpoint {checkpoint_dir} feature-prediction width "
            f"mismatch: checkpoint={saved_width!r}, configured={layout.output_width}"
        )


_LINEAGE_PROBE_FIELDS: tuple[str, ...] = (
    "release",
    "model",
    "activation_type",
    "layer",
    "width",
    "l0",
)


def base_sae_config(
    base_meta: Mapping[str, Any],
    base_user_config: Mapping[str, Any],
    *,
    sft_user_config: Mapping[str, Any],
) -> Any:
    """Return the SAE block the base checkpoint's encoder was actually trained against.

    Precedence: the base checkpoint's published online-SAE lineage (a chained
    conditioned-SFT base), then the base ``user_config.sae`` (a pretrain base),
    then this run's own ``sae`` block. A chained base's ``user_config.sae`` is that
    run's Hydra default and may name a different probe than its features; the
    lineage block is the authoritative record, and the downstream lineage builder
    still rejects any block that disagrees with the canonical feature path.
    """
    lineage = base_meta.get(ONLINE_SAE_LINEAGE_KEY)
    if isinstance(lineage, Mapping):
        missing = [field for field in _LINEAGE_PROBE_FIELDS if field not in lineage]
        if missing:
            raise ValueError(
                f"base checkpoint online-SAE lineage lacks probe fields {missing}; "
                "cannot resolve the SAE block for a chained conditioned SFT"
            )
        resolved = {field: lineage[field] for field in _LINEAGE_PROBE_FIELDS}
        recorded = base_user_config.get("sae")
        if isinstance(recorded, Mapping) and any(
            recorded.get(key) != value for key, value in resolved.items() if key in recorded
        ):
            logger.warning(
                "base checkpoint user_config.sae differs from its online-SAE lineage; "
                f"using the lineage: recorded={dict(recorded)}, lineage={resolved}"
            )
        return resolved
    recorded = base_user_config.get("sae")
    if recorded is not None:
        return recorded
    return sft_user_config.get("sae")


def _checkpoint_user_config_with_fp(
    user_config: dict[str, Any],
    fp_config: FeaturePredictionConfig | None,
) -> dict[str, Any]:
    checkpoint_user_config = dict(user_config)
    for optional_key in (
        "feature_clusters",
        "val_feature_dir",
        "val_sft_parquet_dir",
    ):
        if checkpoint_user_config.get(optional_key) is None:
            checkpoint_user_config.pop(optional_key, None)
    fp_config_dict = _serialize_fp_config(fp_config)
    if fp_config_dict is not None:
        checkpoint_user_config["feature_prediction"] = fp_config_dict
    return checkpoint_user_config


def _restore_loop_state(
    sft_meta: dict | None,
) -> tuple[int, float, float, float, float, Mapping[str, Any] | None]:
    """Restore ``(step, smooth_train_loss, smooth_fp_loss, min_val_bpb,
    total_training_time, sft_dataloader_state)`` from a loaded SFT checkpoint
    meta, or return fresh defaults.

    Factored out of ``main()`` so the restoration logic can be unit
    tested without spinning up the full training loop.  The meta dict
    is produced by ``save_checkpoint`` in this same module and matches
    the ``loop_state`` schema written during training.  If any of the
    required keys is missing we raise ``KeyError`` — we'd rather fail
    loud than silently reset the state and let the user start over
    wondering why their loss jumped.

    Returns:
        A tuple of ``(step, smooth_train_loss, smooth_fp_loss,
        min_val_bpb, total_training_time, sft_dataloader_state)`` — the values
        needed to rebuild ``SFTState`` and the training loop counters.
    """
    if sft_meta is None:
        return 0, 0.0, 0.0, float("inf"), 0.0, None
    step = int(sft_meta["step"])
    loop = sft_meta["loop_state"]
    cursor_semantics = loop.get("sft_cursor_semantics")
    if cursor_semantics != SFT_CURSOR_SEMANTICS_BUFFERED_V2:
        raise DataLoadError(
            "conditioned SFT resume requires loop_state.sft_cursor_semantics="
            f"{SFT_CURSOR_SEMANTICS_BUFFERED_V2!r}; checkpoint records "
            f"{cursor_semantics!r}. Historical checkpoints do not record the "
            "ordered best-fit buffer independently for every DDP rank, so their "
            "next conversation cannot be reconstructed exactly. Start a fresh "
            "SFT run from the source model."
        )
    dataloader_state = loop["sft_dataloader_state"]
    if not isinstance(dataloader_state, Mapping):
        raise DataLoadError(
            "conditioned SFT loop_state.sft_dataloader_state must be a mapping"
        )
    artifact_coverage = loop["sft_artifact_coverage"]
    if not isinstance(artifact_coverage, Mapping):
        raise DataLoadError(
            "conditioned SFT buffered_v2 loop_state.sft_artifact_coverage "
            "must be a mapping"
        )
    return (
        step,
        float(loop["smooth_train_loss"]),
        float(loop["smooth_fp_loss"]),
        float(loop["min_val_bpb"]),
        float(loop["total_training_time"]),
        dataloader_state,
    )


def _assert_sft_resume_artifact_coverage(
    sft_meta: Mapping[str, Any],
    current_coverage: Mapping[str, Any] | None,
) -> None:
    """Reject resuming an SFT cursor against a different artifact coverage.

    This generic guard retains the historical warning-only behavior for callers
    inspecting pre-#1311 metadata. The conditioned-SFT ``buffered_v2`` restore
    boundary separately requires the key before this helper is reached, because
    every genuine v2 writer records it. A key that is present must match exactly
    (review finding 3, PR #1311).
    """
    loop_state = sft_meta.get("loop_state")
    saved_coverage = (
        loop_state.get("sft_artifact_coverage")
        if isinstance(loop_state, Mapping)
        else None
    )
    if saved_coverage is None:
        logger.warning(
            "conditioned SFT resume checkpoint has no sft_artifact_coverage "
            "(written before the key existed); the data cursor cannot be "
            f"verified against the current coverage {current_coverage!r}. "
            "Confirm the parquet/feature artifacts are the ones this run was "
            "started on."
        )
        return
    if not isinstance(saved_coverage, Mapping):
        raise DataLoadError(
            "conditioned SFT resume checkpoint has a malformed "
            f"sft_artifact_coverage: {saved_coverage!r} is not a mapping. "
            "Restart from the pretrain checkpoint and create a fresh SFT run."
        )
    if current_coverage is None or dict(saved_coverage) != dict(current_coverage):
        raise DataLoadError(
            "conditioned SFT resume artifact coverage mismatch: "
            f"checkpoint={dict(saved_coverage)!r}, current={current_coverage!r}. "
            "The SFT cursor is tied to its artifact family; restart from the "
            "pretrain checkpoint or use the exact parquet/feature artifacts."
        )


ProfileMarkerSwitch = bool | Callable[[], bool]


@dataclass(frozen=True)
class _SFTStartupPhaseRecord:
    """One always-on, rank-0-visible conditioned-SFT startup phase."""

    name: str
    elapsed_ns: int
    status: str
    extra: tuple[tuple[str, object], ...]
    include_in_sum: bool = True


class _SFTStartupTrace:
    """Minimal startup ledger independent of optional live-loop profiling.

    The trace deliberately owns no model/data state and never creates a
    profiler marker.  It is therefore safe to pass through the lazy generator
    without changing its outputs or its RNG/data-loader behavior.
    """

    def __init__(self, *, start_ns: int | None = None) -> None:
        self.start_ns = time.perf_counter_ns() if start_ns is None else int(start_ns)
        self._records: list[_SFTStartupPhaseRecord] = []
        self._active = False
        self._first_step_elapsed_ns: int | None = None
        self._startup_phase_sum_ns: int | None = None
        self._time_to_first_step_ns: int | None = None
        self._startup_summary_logged = False
        self.ddp_rank: int | None = None
        self.ddp_world_size: int | None = None

    @property
    def phase_sum_ns(self) -> int:
        return sum(
            record.elapsed_ns for record in self._records if record.include_in_sum
        )

    @property
    def first_step_elapsed_ns(self) -> int | None:
        return self._first_step_elapsed_ns

    def set_ddp(self, *, rank: int, world_size: int) -> None:
        self.ddp_rank = int(rank)
        self.ddp_world_size = int(world_size)

    @staticmethod
    def _format_value(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def _log_record(self, record: _SFTStartupPhaseRecord) -> None:
        fields = [
            "SFT startup",
            f"phase={record.name}",
            f"status={record.status}",
            f"elapsed_s={record.elapsed_ns / 1_000_000_000:.6f}",
        ]
        fields.extend(
            f"{key}={self._format_value(value)}" for key, value in record.extra
        )
        logger.info(" ".join(fields))

    def record(
        self,
        name: str,
        elapsed_ns: int,
        *,
        status: str = "completed",
        extra: Mapping[str, object] | None = None,
        include_in_sum: bool = True,
    ) -> _SFTStartupPhaseRecord:
        if not name or any(character.isspace() for character in name):
            raise ValueError(f"startup phase name must be one token, got {name!r}")
        if elapsed_ns < 0:
            raise ValueError(f"startup phase elapsed_ns must be non-negative, got {elapsed_ns}")
        if status not in {"completed", "skipped", "failed"}:
            raise ValueError(f"unsupported startup phase status: {status!r}")
        normalized_extra = tuple(
            sorted((str(key), value) for key, value in (extra or {}).items())
        )
        record = _SFTStartupPhaseRecord(
            name=name,
            elapsed_ns=int(elapsed_ns),
            status=status,
            extra=normalized_extra,
            include_in_sum=include_in_sum,
        )
        self._records.append(record)
        self._log_record(record)
        return record

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        extra: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
        include_in_sum: bool = True,
    ) -> Iterator[None]:
        """Measure one non-overlapping leaf phase and preserve exceptions."""
        if self._active:
            raise RuntimeError(
                f"conditioned SFT startup phases must not overlap; attempted {name!r}"
            )
        self._active = True
        started_ns = time.perf_counter_ns()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self._active = False
            if extra is None:
                resolved_extra = None
            elif isinstance(extra, Mapping):
                resolved_extra = string_keyed_mapping(extra)
                if resolved_extra is None:
                    raise TypeError("conditioned SFT startup metadata keys must be strings")
            else:
                resolved_extra = extra()
            self.record(
                name,
                time.perf_counter_ns() - started_ns,
                status=status,
                extra=resolved_extra,
                include_in_sum=include_in_sum,
            )

    def record_first_step(
        self,
        *,
        completed_ns: int,
        step_elapsed_ns: int,
        step: int | None = None,
    ) -> None:
        if self._first_step_elapsed_ns is not None:
            raise RuntimeError("conditioned SFT first-step timing was recorded twice")
        if completed_ns < self.start_ns:
            raise ValueError("first-step completion precedes startup start")
        if step_elapsed_ns < 0:
            raise ValueError("first-step elapsed time must be non-negative")
        self._first_step_elapsed_ns = int(step_elapsed_ns)
        self._startup_summary_logged = True
        phase_sum_ns = self.phase_sum_ns
        total_ns = completed_ns - self.start_ns
        self._startup_phase_sum_ns = phase_sum_ns
        self._time_to_first_step_ns = total_ns
        self.record(
            "first_optimizer_step",
            step_elapsed_ns,
            extra={} if step is None else {"step": step},
            include_in_sum=False,
        )
        self._log_summary(
            summary="time_to_first_step",
            status="completed",
            time_to_first_step_ns=total_ns,
            phase_sum_ns=phase_sum_ns,
            first_step_ns=step_elapsed_ns,
            post_first_step_mean_ns=None,
            post_first_step_steps=0,
        )

    def log_no_step_summary(self, *, completed_ns: int | None = None) -> None:
        if self._startup_summary_logged:
            return
        end_ns = time.perf_counter_ns() if completed_ns is None else int(completed_ns)
        self._startup_summary_logged = True
        self._log_summary(
            summary="time_to_first_step",
            status="not_reached",
            time_to_first_step_ns=end_ns - self.start_ns,
            phase_sum_ns=self.phase_sum_ns,
            first_step_ns=None,
            post_first_step_mean_ns=None,
            post_first_step_steps=0,
        )

    def _log_summary(
        self,
        *,
        summary: str,
        status: str,
        time_to_first_step_ns: int,
        phase_sum_ns: int,
        first_step_ns: int | None,
        post_first_step_mean_ns: float | None,
        post_first_step_steps: int,
    ) -> None:
        residual_ns = time_to_first_step_ns - phase_sum_ns - (first_step_ns or 0)
        fields: dict[str, object] = {
            "summary": summary,
            "status": status,
            "time_to_first_step_s": time_to_first_step_ns / 1_000_000_000,
            "startup_phase_sum_s": phase_sum_ns / 1_000_000_000,
            "unattributed_startup_s": residual_ns / 1_000_000_000,
            "first_step_warmup_s": (
                None if first_step_ns is None else first_step_ns / 1_000_000_000
            ),
            "post_first_step_mean_s": (
                None
                if post_first_step_mean_ns is None
                else post_first_step_mean_ns / 1_000_000_000
            ),
            "post_first_step_steps": post_first_step_steps,
        }
        if self.ddp_rank is not None:
            fields["ddp_rank"] = self.ddp_rank
        if self.ddp_world_size is not None:
            fields["ddp_world_size"] = self.ddp_world_size
        formatted = " ".join(
            f"{key}={self._format_value(value)}" for key, value in sorted(fields.items())
        )
        logger.info(f"SFT startup summary {formatted}")

    def log_steady_state(
        self,
        *,
        time_to_first_step_ns: int,
        post_first_step_total_ns: int,
        post_first_step_steps: int,
    ) -> None:
        mean_ns = (
            post_first_step_total_ns / post_first_step_steps
            if post_first_step_steps > 0
            else None
        )
        time_to_first_step_ns = (
            self._time_to_first_step_ns
            if self._time_to_first_step_ns is not None
            else time_to_first_step_ns
        )
        self._log_summary(
            summary="steady_state",
            status="completed" if self.first_step_elapsed_ns is not None else "not_reached",
            time_to_first_step_ns=time_to_first_step_ns,
            phase_sum_ns=(
                self._startup_phase_sum_ns
                if self._startup_phase_sum_ns is not None
                else self.phase_sum_ns
            ),
            first_step_ns=self.first_step_elapsed_ns,
            post_first_step_mean_ns=mean_ns,
            post_first_step_steps=post_first_step_steps,
        )


@beartype
@dataclass(frozen=True)
class SFTProfileSettings:
    """Refined profiling coordinates validated by ``@beartype``."""

    output_dir: Path
    trace_step: NonNegativeInt | None
    trace_steps: PositiveInt
    timing_enabled: bool
    timing_sync_cuda: bool


def _resolve_sft_profile_settings(cfg) -> SFTProfileSettings:
    """Parse dynamic profiling config into the beartyped settings record."""
    raw_trace_step = cfg.profile_trace
    try:
        trace_step = None if raw_trace_step is None else NonNegativeInt.parse(int(raw_trace_step))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"profile_trace must be a non-negative integer or null, got {raw_trace_step!r}"
        ) from exc

    raw_trace_steps = cfg.profile_trace_steps
    try:
        trace_steps = PositiveInt.parse(int(raw_trace_steps))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"profile_trace_steps must be a positive integer, got {raw_trace_steps!r}"
        ) from exc

    return SFTProfileSettings(
        output_dir=Path(cfg.profile_output_dir),
        trace_step=trace_step,
        trace_steps=trace_steps,
        timing_enabled=bool(cfg.profile_timing),
        timing_sync_cuda=bool(cfg.profile_timing_sync_cuda),
    )


def _profile_markers_are_enabled(profile_markers: ProfileMarkerSwitch) -> bool:
    if callable(profile_markers):
        return bool(profile_markers())
    return bool(profile_markers)


@contextmanager
def _maybe_timed_region(
    name: str,
    *,
    profile_markers: ProfileMarkerSwitch,
    extra: Mapping[str, Any] | None = None,
):
    if not _profile_markers_are_enabled(profile_markers):
        yield
        return
    with timed_region(name, sync_cuda=False, extra=extra):
        yield


def _sft_feature_load_report_extra(
    report: SFTFeatureLoadReport,
    *,
    source: str,
) -> dict[str, object]:
    """Return bounded feature-store telemetry without touching artifact data."""
    extra: dict[str, object] = {"source": source}
    if report.cache_hit is not None:
        extra["cache"] = "hit" if report.cache_hit else "miss"
    if report.feature_shards is not None:
        extra["feature_shards"] = report.feature_shards
    if report.parquet_shards is not None:
        extra["parquet_shards"] = report.parquet_shards
    if report.documents is not None:
        extra["documents"] = report.documents
    return extra


def _load_sft_features_for_startup(
    feature_dir: str,
    parquet_dir: str,
    *,
    startup_trace: _SFTStartupTrace | None,
    source: str,
):
    """Load SFT features and attribute the existing work when tracing startup."""
    if startup_trace is None:
        # Preserve the exact legacy call path, including test doubles and cache
        # behavior, when no startup trace is requested.
        return load_sft_features(feature_dir, parquet_dir)

    report = SFTFeatureLoadReport()
    with startup_trace.phase(
        "sft_feature_store_load",
        extra=lambda: _sft_feature_load_report_extra(report, source=source),
    ):
        return load_sft_features(feature_dir, parquet_dir, report=report)


@beartype
def _assert_sft_feature_dir_segment_mode(
    feature_dir: str,
    artifact_mode: SegmentMode,
) -> None:
    """Cross-check canonical path identity against discovered shard metadata."""
    resolved_path = str(Path(feature_dir).resolve())
    try:
        parsed = parse_feature_dir(resolved_path)
    except ValueError as error:
        logger.warning(
            "Non-canonical conditioned-SFT feature directory accepted; the "
            "path-encoded segmentation mode cannot be cross-checked: "
            f"{feature_dir!r} (resolves to {resolved_path!r}; {error})"
        )
        return
    if parsed.segment_mode is artifact_mode:
        return
    raise DataLoadError(
        "Conditioned SFT feature directory leaf disagrees with the shard "
        f"metadata: directory {feature_dir!r} encodes "
        f"{parsed.segment_mode.value!r}, artifacts record "
        f"{artifact_mode.value!r}; re-extract into the canonical output_dir."
    )


@beartype
def _finalize_sft_segment_mode(
    train_mode: SegmentMode,
    val_mode: SegmentMode | None,
    *,
    train_feature_dir: str,
    val_feature_dir: str | None,
    resuming: bool,
    checkpoint_meta: Mapping[str, Any] | None,
    checkpoint_dir: str,
) -> SegmentMode:
    """Validate, log, and resume-check discovered SFT segmentation identity."""
    if val_mode is not None and val_mode is not train_mode:
        raise DataLoadError(
            "Conditioned SFT validation features must be extracted in the same "
            "segmentation mode as training: "
            f"train feature directory {train_feature_dir!r} records "
            f"{train_mode.value!r}, validation feature directory "
            f"{val_feature_dir!r} records {val_mode.value!r}."
        )
    _assert_sft_feature_dir_segment_mode(train_feature_dir, train_mode)
    # A resolved validation directory without a loaded validation store means
    # the run has no validation documents, so there is no artifact mode to
    # cross-check its leaf against.
    if val_feature_dir is not None and val_mode is not None:
        _assert_sft_feature_dir_segment_mode(val_feature_dir, val_mode)
    logger.info(f"SFT feature segmentation mode: {train_mode.value}")
    if resuming:
        assert_resume_sft_segment_mode_matches(
            train_mode,
            checkpoint_meta,
            checkpoint_dir=checkpoint_dir,
        )
    return train_mode


@contextmanager
def _maybe_sft_startup_phase(
    startup_trace: _SFTStartupTrace | None,
    name: str,
    *,
    extra: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
):
    """Keep the generator's non-telemetry path completely unchanged."""
    phase = (
        nullcontext()
        if startup_trace is None
        else startup_trace.phase(name, extra=extra)
    )
    with phase:
        yield


def _percentile_coverage_preflight(
    *,
    coverage_sources: list[tuple[str, Mapping[int, Any], Sequence[int]]],
    n_sae_features: int,
    feature_mask: FeatureMask | None,
    fp_percentile_table: t.Tensor | None,
    sft_sampling_bundle: SamplingBundle | None,
    cluster_threshold_percentile_table: t.Tensor | None,
    missing_percentile_policy: MissingPercentilePolicy,
    device: str | t.device,
    startup_trace: _SFTStartupTrace | None,
) -> MissingPercentileResolver:
    """Scan required ids ONCE (rank 0), broadcast, and enforce table coverage.

    Extracted from ``main`` (issue #1495) so the rank-once scan, the coverage
    verdicts, and the resolver construction are unit- and two-rank-testable
    without driving the whole training startup. The caller wraps this in
    ``rank_synchronized_stage("percentile coverage", ...)``. The only
    collectives inside are the aligned rank-0 scan broadcast; every rank
    reaches them together because everything before them is deterministic
    local work on identically configured inputs.
    """
    coverage_scan_extra: dict[str, object] = {"sources": len(coverage_sources)}
    with _maybe_sft_startup_phase(
        startup_trace, "sft_percentile_id_scan", extra=lambda: coverage_scan_extra
    ):

        def _scan_all_sources() -> t.Tensor:
            bitmap = t.zeros(n_sae_features, dtype=t.bool)
            for source_name, store, indices in coverage_sources:
                bitmap |= scan_required_feature_id_bitmap(
                    store,
                    indices,
                    n_sae_features=n_sae_features,
                    source=source_name,
                )
            return bitmap

        # UNCONDITIONAL, even with zero sources: the helper's collectives must
        # exist on every rank regardless of local state so no rank can pair
        # its outer stage all-reduce with a peer's inner one.
        queried_ids = broadcast_bitmap_from_rank_zero(
            _scan_all_sources,
            numel=n_sae_features,
            device=device,
            stage="sft_percentile_id_scan",
        )
        raw_required = t.nonzero(queried_ids, as_tuple=False).squeeze(1)
        if feature_mask is None:
            masked_required = raw_required
        else:
            masked_required = raw_required[feature_mask.keep_mask(raw_required)]
        coverage_scan_extra["raw_ids"] = int(raw_required.numel())
        coverage_scan_extra["masked_ids"] = int(masked_required.numel())

    coverage_extra: dict[str, object] = {
        "tables": 0,
        "policy": missing_percentile_policy.value,
        "sources": len(coverage_sources),
    }
    resolver = MissingPercentileResolver.error()
    with _maybe_sft_startup_phase(
        startup_trace, "sft_percentile_coverage", extra=lambda: coverage_extra
    ):
        # Each table is checked against the id set it is actually queried
        # with; iteration order matches the historical statement order.
        table_checks = (
            (
                fp_percentile_table,
                raw_required,
                "fp_percentile_table",
                "raw pre-selection segment feature ids",
            ),
            (
                None if sft_sampling_bundle is None else sft_sampling_bundle.percentile_table,
                masked_required,
                "sampling_bundle.percentile_table",
                "post-feature-mask selection candidates",
            ),
            (
                cluster_threshold_percentile_table,
                masked_required,
                "cluster_threshold_percentile_table",
                "post-feature-mask selection candidates",
            ),
        )
        coverage_reports = []
        coverage_bitmaps = []
        for table, required_ids, table_name, required_set_name in table_checks:
            if table is None:
                continue
            coverage_reports.append(
                check_percentile_coverage(
                    percentile_table=table,
                    required_ids=required_ids,
                    table_name=table_name,
                    required_set_name=required_set_name,
                )
            )
            coverage_bitmaps.append(percentile_table_coverage(table))
        coverage_extra["tables"] = len(coverage_reports)
        if coverage_reports:
            enforce_percentile_coverage(coverage_reports, missing_percentile_policy)
        if missing_percentile_policy is MissingPercentilePolicy.LOWEST:
            if not coverage_bitmaps:
                raise ValueError(
                    "missing_percentile_policy=lowest was configured but this run "
                    "queries no percentile table at all (no feature-prediction "
                    "head and no percentile-based conditioning selection). "
                    "Remove the override."
                )
            resolver = MissingPercentileResolver(
                policy=MissingPercentilePolicy.LOWEST,
                covered=combined_coverage(coverage_bitmaps),
            )
            logger.warning(
                "conditioned SFT is running as a PILOT "
                "(missing_percentile_policy=lowest): val/bpb and min_val_bpb "
                "from this run are NOT comparable to a run trained under "
                "missing_percentile_policy=error. The policy is recorded in "
                "every checkpoint and a resume under a different policy is "
                "rejected."
            )
    return resolver


def _build_sft_val_loader(
    *,
    val_task,
    tokenizer,
    B: int,
    T: int,
    device: str | t.device,
    ddp_rank: int,
    ddp_world_size: int,
    feature_dir: str | None,
    expected_feature_dir: str | None,
    parquet_dir: str | None,
    threshold_cfg,
    boundaries,
    precomputed_thresholds,
    sampling_bundle,
    loss_mask_mode: LossMaskMode,
    min_loss_tokens: int,
    n_sae_features: int | None = None,
    fp_percentile_table: t.Tensor | None = None,
    cluster_threshold_percentile_table: t.Tensor | None = None,
    negative_bundle: NegativeBundle | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    feature_mask: FeatureMask | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
    missing_percentile: MissingPercentileResolver | None = None,
    feature_source_bos_char_len: int = 0,
    profile_markers: ProfileMarkerSwitch = False,
    startup_trace: _SFTStartupTrace | None = None,
    alignment_probe_samples: int = 8,
    mixture_name: str = "unknown",
    return_seg_doc_id: bool = False,
) -> _SFTValidationLoader:
    """Construct a fresh validation data generator.

    The validation eval loop rebuilds its generator on every
    evaluation so the cursor restarts at the beginning of the val
    task each time — this is what the in-line ``build_val_loader()``
    closure used to do.  Extracting it as a module-level factory lets
    the unit tests verify the plumbing (feature dir passthrough,
    fresh ``SFTState()``) without spinning up
    ``main()``.

    Kwargs-only so the call site stays self-documenting at the
    otherwise-long argument list.
    """
    if (
        feature_dir is None
        or expected_feature_dir is None
        or Path(feature_dir).resolve() != Path(expected_feature_dir).resolve()
    ):
        raise DataLoadError(
            "Conditioned SFT validation feature-dir identity mismatch: "
            f"feature_dir={feature_dir!r}, expected_feature_dir={expected_feature_dir!r}. "
            "build_val_loader must pass the resolved validation feature directory."
        )
    val_state = SFTState()
    generator = make_sft_data_generator(
        task=val_task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device=device,
        sft_state=val_state,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        threshold_cfg=threshold_cfg,
        boundaries=boundaries,
        precomputed_thresholds=precomputed_thresholds,
        sampling_bundle=sampling_bundle,
        loss_mask_mode=loss_mask_mode,
        min_loss_tokens=min_loss_tokens,
        n_sae_features=n_sae_features,
        fp_percentile_table=fp_percentile_table,
        cluster_threshold_percentile_table=cluster_threshold_percentile_table,
        # Train and validation must condition identically: the same bundle
        # object, so validation draws negatives from the same distribution and
        # the same (shared) RNG the train generator advances.
        negative_bundle=negative_bundle,
        fp_target_layout=fp_target_layout,
        feature_mask=feature_mask,
        consolidation=consolidation,
        missing_percentile=missing_percentile,
        feature_source_bos_char_len=feature_source_bos_char_len,
        profile_markers=profile_markers,
        startup_trace=startup_trace,
        alignment_probe_samples=alignment_probe_samples,
        mixture_name=mixture_name,
        return_seg_doc_id=return_seg_doc_id,
        # Validation BPB must be measured over DISTINCT documents: one pass
        # over the covered set, padding instead of wrapping.
        single_pass=True,
        expected_split=SFTSplit.VAL,
    )
    return _SFTValidationLoader(generator, val_state)


@beartype
def _steered_positions_for_task(task) -> frozenset[int] | None:
    """Dataset positions of the steered rows, or ``None`` for a plain mixture.

    They ARE the values the generator serves as ``original_index``, so the FP
    prediction-group split needs no second marker. Validation never carries
    steered rows (steered mixtures validate on standard's family).
    """
    if not isinstance(task, SteeredTaskMixture):
        return None
    positions = task.steered_positions
    # The percentage cannot divide by zero: ``SteeredTaskMixture.__init__``
    # rejects ``steered_count < 1`` and ``len(task)`` is base rows PLUS that
    # count, so an empty steered mixture is unconstructible.  Guarding here
    # would only add a branch no artifact can reach.
    logger.info(
        f"Steered mixture: {len(positions):,} steered rows in {len(task):,} "
        f"({len(positions) / len(task):.2%})"
    )
    return positions


@beartype
def _assert_steered_segment_mode(
    steered_positions: frozenset[int] | None,
    segment_mode: SegmentMode,
    dataset: str,
) -> None:
    """Steered rows need turn boundaries for their prediction-group split."""
    if steered_positions is None or segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        return
    raise DataLoadError(
        f"steered mixture {dataset!r} requires turn-delimited SFT features "
        "(a …/turn_delimited feature family) so the prediction-group split can land "
        f"on the assistant turn; the loaded stores are segment_mode={segment_mode.value!r}"
    )


def _truncate_seg_features(
    seg_features: list[SegFeatures] | None,
    total_chars: int,
) -> list[SegFeatures] | None:
    """Truncate segment features to match a token-truncated conversation.

    Keeps all segments whose ``char_start < total_chars`` (they overlap
    with the surviving text) plus **one extra segment** past the boundary
    for feature-prediction targets.  The extra segment is marked with
    ``prediction_target_only=True``.

    The last in-range segment's ``char_end`` is clipped to ``total_chars``
    so downstream full-conversation routing sees consistent geometry.

    **Character gap**: When the extra segment's ``char_start`` exceeds the
    clipped ``char_end`` of the last in-range segment, there is a gap in
    character coverage (e.g., last in-range clipped to char 128 but the
    extra starts at char 150).  This is intentional — the extra segment
    is only used as a feature-prediction target in the encoder, not for
    cross-attention conditioning of decoder positions.  The gap never
    reaches decoder routing because prediction-target-only
    segments are filtered out before that call.

    Returns ``None`` when no segments survive (all segments are past the
    truncation boundary, or input is None/empty).
    """
    if not seg_features:
        return None

    in_range: list[SegFeatures] = []
    extra: SegFeatures | None = None

    for seg in seg_features:
        if seg.char_start < total_chars:
            in_range.append(seg)
        elif extra is None:
            # First segment past boundary → keep as prediction target
            extra = replace(seg, prediction_target_only=True)

    if not in_range:
        return None

    # Clip the last in-range segment's char_end
    last = in_range[-1]
    if last.char_end > total_chars:
        in_range[-1] = replace(last, char_end=total_chars)

    if extra is not None:
        in_range.append(extra)

    return in_range


@dataclass(frozen=True)
class PreparedSftExample:
    """One truncated conversation and its conditioned-input ownership."""

    ids: list[int]
    loss_mask: list[int]
    original_len: int
    was_truncated: bool
    n_loss_tokens: int
    segments: list[SegFeatures]
    routing: FullSequenceSegmentRouting | None


def prepare_sft_example(
    *,
    ids: Sequence[int],
    loss_mask: Sequence[int],
    row_capacity: int,
    loss_mask_mode: LossMaskMode,
    min_loss_tokens: int,
    select_segments: Callable[[], list[SegFeatures]],
    tokenizer: Any,
    feature_source_bos_char_len: int,
    example_idx: int,
    profile_markers: bool | Callable[[], bool] = False,
) -> PreparedSftExample:
    """Prepare a generator/cadence row with one shared skip and routing contract.

    Selection is lazy so skipped conversations consume no selection RNG. The
    final rendered token is a target, not a conditioned input, even when another
    conversation follows it in the packing row. Prediction-only segments remain
    available to the generator but never participate in token routing.
    """
    ids, mask, original_len, was_truncated = truncate_conv_and_mask(
        list(ids), list(loss_mask), row_capacity
    )
    n_loss_tokens = (
        supervised_target_count(mask)
        if loss_mask_mode is LossMaskMode.COMPLETION_ONLY
        else max(0, len(ids) - 1)
    )
    if n_loss_tokens < min_loss_tokens:
        return PreparedSftExample(ids, mask, original_len, was_truncated, n_loss_tokens, [], None)
    with _maybe_timed_region("sft/feature_select_discretize", profile_markers=profile_markers):
        segments = select_segments()
    if not segments:
        raise ValueError(
            f"conditioned SFT example {example_idx}: feature row has no segments. "
            "Re-extract full-sequence SFT features over the rendered task mixture before training."
        )
    with _maybe_timed_region("sft/coverage_gap", profile_markers=profile_markers):
        gap = _compute_full_coverage_gap(
            ids, segments, tokenizer, source_bos_char_len=feature_source_bos_char_len
        )
    if gap > 20:
        raise ValueError(
            f"conditioned SFT example {example_idx}: feature coverage ends {gap} chars "
            "before the rendered conversation. Re-extract full-sequence SFT features "
            "with the same rendered task mixture and tokenization settings before training."
        )
    if was_truncated:
        with _maybe_timed_region("sft/truncate_seg_features", profile_markers=profile_markers):
            total_chars = _source_aligned_char_offsets(
                tokenizer, ids, source_bos_char_len=feature_source_bos_char_len
            )[-1]
            clipped = _truncate_seg_features(segments, total_chars)
        if clipped is None:
            raise ValueError(
                f"conditioned SFT example {example_idx}: truncation removed every feature "
                "segment. Re-extract full-sequence SFT features with matching rendering "
                "and tokenization settings before training."
            )
        segments = clipped
    in_range = [segment for segment in segments if not segment.prediction_target_only]
    with _maybe_timed_region("sft/build_conv_segments", profile_markers=profile_markers):
        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=(
                [segment.feature_ids for segment in in_range],
                [segment.buckets for segment in in_range],
                [segment.char_start for segment in in_range],
                [segment.char_end for segment in in_range],
            ),
            source_bos_char_len=feature_source_bos_char_len,
        )
    return PreparedSftExample(
        ids,
        mask,
        original_len,
        was_truncated,
        n_loss_tokens,
        segments,
        FullSequenceSegmentRouting(
            routing.per_seg_fids, routing.per_seg_buckets, routing.seg_map[:-1]
        ),
    )


def _log_skipped_conv(
    *,
    example_idx: int,
    original_len: int,
    truncated_len: int,
    n_loss_tokens: int,
    min_loss_tokens: int,
) -> None:
    """Log a conversation skipped for too few supervised loss tokens."""
    logger.debug(
        f"Skipping conv {example_idx}: {original_len} original tokens, "
        f"{truncated_len} after truncation, {n_loss_tokens} supervised loss tokens "
        f"< {min_loss_tokens} minimum"
    )


def _log_truncated_conv(*, example_idx: int, original_len: int, truncated_len: int) -> None:
    """Emit a debug log when a conversation is truncated but accepted."""
    logger.debug(f"Truncated conv {example_idx}: {original_len} -> {truncated_len} tokens")


def _log_periodic_summary(*, processed: int, skipped: int, min_loss_tokens: int) -> None:
    """Emit a warning summary every 1000 conversations if any were skipped."""
    if processed % 1000 == 0 and processed > 0 and skipped > 0:
        logger.warning(
            f"SFT buffer: skipped {skipped}/{processed} conversations "
            f"with <{min_loss_tokens} supervised loss tokens after truncation"
        )


def _stack_padded_encoder_inputs(
    row_enc_fids: list[t.Tensor],
    row_enc_mags: list[t.Tensor],
    row_enc_masks: list[t.Tensor],
    *,
    device: str | t.device,
    use_cuda: bool,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Stack per-row encoder tensors with one host-to-device transfer per field.

    ``_pad_segments`` returns CPU tensors because the preceding SFT feature
    alignment and discretization path is CPU-bound.  Building the final batch
    directly on CUDA and assigning one row at a time launches multiple tiny H2D
    copies per micro-batch.  Fill a CPU staging tensor instead, then transfer
    each of fids, magnitudes, and masks once.
    """
    batch_n_seg = max(f.shape[0] for f in row_enc_fids)
    batch_f_max = max(1, max(f.shape[1] for f in row_enc_fids))
    batch_size = len(row_enc_fids)

    enc_fids_cpu = t.zeros(
        batch_size,
        batch_n_seg,
        batch_f_max,
        dtype=t.int64,
        pin_memory=use_cuda,
    )
    enc_mags_cpu = t.zeros(
        batch_size,
        batch_n_seg,
        batch_f_max,
        dtype=t.int64,
        pin_memory=use_cuda,
    )
    enc_mask_cpu = t.zeros(
        batch_size,
        batch_n_seg,
        batch_f_max,
        dtype=t.bool,
        pin_memory=use_cuda,
    )

    for row_idx in range(batch_size):
        ns = row_enc_fids[row_idx].shape[0]
        nf = row_enc_fids[row_idx].shape[1]
        enc_fids_cpu[row_idx, :ns, :nf] = row_enc_fids[row_idx]
        enc_mags_cpu[row_idx, :ns, :nf] = row_enc_mags[row_idx]
        enc_mask_cpu[row_idx, :ns, :nf] = row_enc_masks[row_idx]

    if str(device) == "cpu":
        return enc_fids_cpu, enc_mags_cpu, enc_mask_cpu

    return (
        enc_fids_cpu.to(device=device, non_blocking=use_cuda),
        enc_mags_cpu.to(device=device, non_blocking=use_cuda),
        enc_mask_cpu.to(device=device, non_blocking=use_cuda),
    )


def _resolve_sft_parquet_reference_indices(
    sft_features: Mapping[int, Any],
    *,
    conversion_coverage: int,
    feature_dir: str,
    parquet_dir: str,
) -> list[int]:
    """Return the parquet's ``original_index`` set as the reference documents.

    The reference set is read from the parquet shards themselves, NOT derived
    from the feature store, so the coverage comparison below is a real check
    rather than a restatement of how the store was built.  Both directions are
    enforced: feature rows with no parquet row are rejected here, and parquet
    rows with no feature row are rejected by the caller's ``missing_indices``
    guard.
    """
    parquet_indices = getattr(sft_features, "parquet_original_indices", None)
    if not isinstance(parquet_indices, frozenset | set):
        raise DataLoadError(
            "conditioned SFT feature store does not expose "
            "parquet_original_indices, so the converted document set cannot be "
            "established independently of the feature index. Rebuild the store "
            f"via load_sft_features({feature_dir!r}, {parquet_dir!r})."
        )
    out_of_prefix = sorted(
        index
        for index in parquet_indices
        if not isinstance(index, int) or index < 0 or index >= conversion_coverage
    )
    if out_of_prefix:
        raise DataLoadError(
            "conditioned SFT parquet original_index values fall outside the "
            f"converted prefix [0, {conversion_coverage}): {out_of_prefix[:10]}. "
            "Regenerate the matched artifacts."
        )
    reference_indices = sorted(int(index) for index in parquet_indices)
    if not reference_indices:
        raise DataLoadError(
            "conditioned SFT conversion coverage is non-empty but the parquet "
            f"at {parquet_dir} indexes no documents. Regenerate the matched "
            "parquet and feature artifacts."
        )
    orphan_feature_indices = sorted(
        int(index) for index in sft_features if int(index) not in parquet_indices
    )
    if orphan_feature_indices:
        raise DataLoadError(
            f"conditioned SFT feature store at {feature_dir} indexes "
            f"{len(orphan_feature_indices)} documents that the parquet at "
            f"{parquet_dir} does not contain: {orphan_feature_indices[:10]}. "
            "Features and parquet must cover exactly the same original_index "
            "set; regenerate the matched artifact family."
        )
    return reference_indices


def _resolve_sft_dataset_indices(
    sft_features: Mapping[int, Any],
    *,
    task_size: int,
    feature_dir: str,
    parquet_dir: str,
) -> tuple[int | None, Sequence[int]]:
    """Resolve the covered document set the run will actually consume.

    A capped conversion stores a covered prefix of the full, fingerprinted
    mixture.  The conversion metadata records the attempted prefix length; the
    parquet's own ``original_index`` column is the authoritative set of rows
    that survived conversion.  ``convert_sft_to_parquet`` legitimately drops
    ``no_assistant`` / ``empty_response`` conversations, so that set is a subset
    of the attempted prefix even for an UNCAPPED run — demanding
    ``range(task_size)`` there would make a normal full-mixture conversion
    fatal, with a "re-extract features" instruction no re-extraction can satisfy
    (review findings 6 + 14, PR #1311).  Legacy artifacts have no coverage field
    and retain the historical full-mixture requirement.

    Extracted from the generator body so the STARTUP percentile-coverage
    preflight resolves exactly the same document set the generator will walk,
    rather than an approximation of it.
    """
    conversion_coverage = getattr(sft_features, "conversion_total_conversations", None)
    if conversion_coverage is None:
        return None, range(task_size)
    if (
        isinstance(conversion_coverage, bool)
        or not isinstance(conversion_coverage, int)
        or conversion_coverage <= 0
        or conversion_coverage > task_size
    ):
        raise DataLoadError(
            "conditioned SFT conversion coverage is invalid: "
            f"total_conversations={conversion_coverage!r}, live task length="
            f"{task_size}. Regenerate the matched conversion artifacts."
        )
    dataset_indices = _resolve_sft_parquet_reference_indices(
        sft_features,
        conversion_coverage=conversion_coverage,
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
    )
    logger.info(
        "Conditioned SFT artifact coverage: "
        f"{len(dataset_indices):,} parquet documents from converted prefix "
        f"{conversion_coverage:,}/{task_size:,}"
    )
    return conversion_coverage, dataset_indices


def _make_sft_data_generator_impl(
    task,
    tokenizer,
    B: int,
    T: int,
    device: str | t.device,
    sft_state: SFTState,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    buffer_size: int = 100,
    feature_dir: str | None = None,
    parquet_dir: str | None = None,
    threshold_cfg: ThresholdConfig | None = None,
    boundaries: t.Tensor | None = None,
    precomputed_thresholds: t.Tensor | None = None,
    sampling_bundle: SamplingBundle | None = None,
    min_loss_tokens: int = 8,
    n_sae_features: int | None = None,
    fp_percentile_table: t.Tensor | None = None,
    cluster_threshold_percentile_table: t.Tensor | None = None,
    *,
    grad_accum_steps: int = 1,
    negative_bundle: NegativeBundle | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    feature_mask: FeatureMask | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
    missing_percentile: MissingPercentileResolver | None = None,
    return_seg_doc_id: bool = False,
    feature_source_bos_char_len: int = 0,
    profile_markers: ProfileMarkerSwitch = False,
    startup_trace: _SFTStartupTrace | None = None,
    loss_mask_mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    alignment_probe_samples: int = 8,
    mixture_name: str = "unknown",
    single_pass: bool = False,
    expected_split: SFTSplit,
    steered_positions: frozenset[int] | None = None,
):
    """BOS-bestfit SFT data generator for encoder-decoder models.

    Draws conversations from a Task, tokenizes them with the chat template,
    packs them into rows using best-fit-with-padding, and builds LM targets.
    ``loss_mask_mode`` chooses full-sequence or completion-only supervision.

    Builds encoder inputs with segment-aligned features from pre-extracted
    full-conversation SAE activations.

    When an FP width (the production ``fp_target_layout`` or legacy
    ``n_sae_features`` argument) and ``fp_percentile_table`` are both set,
    also builds a dense ``(B, batch_n_seg, feature_prediction_width)``
    ``fp_targets`` tensor scattering all-feature percentiles for each segment.
    Used by the next-segment feature-prediction auxiliary loss
    (``EncoderDecoderGPT(feature_pred_config=...)``).  When either is
    ``None``, ``fp_targets`` is yielded as ``None`` and the model receives
    no fp signal (the head's contribution to the combined loss is zero).

    ``feature_mask`` filters positive conditioning candidates before
    thresholding/sampling. Full-width FP targets zero masked columns; compact
    layouts drop masked sources because those columns do not exist.
    ``feature_mask is None`` keeps the full-width legacy path byte-identical.

    ``consolidation`` folds mask-surviving cluster siblings to their fixed
    representative with max full-distribution percentile before selection and
    the FP layout applies the same max fold at the target boundary. Clustered
    threshold selection obtains those ranks from
    ``cluster_threshold_percentile_table``; it never interprets post-threshold
    bucket boundaries as a second, boundary-relative percentile currency.
    ``None`` leaves the legacy unclustered tensor path byte-identical.

    ``negative_bundle`` (the ``NegativeBundle`` the checkpoint's pretraining
    ``negative_selection`` produces) makes every segment also carry a sampled
    set of INACTIVE features in the reserved bucket ``effective_buckets``,
    exactly as pretraining did. ``None`` — an unconditioned lineage — leaves
    every emitted tensor bit-identical to the positives-only path. Negatives
    ride in the encoder input only: they have no decoder tokens, so ``seg_map``,
    the char offsets, and ``_truncate_seg_features`` are unaffected. They also
    never enter ``all_feature_ids`` / ``all_feature_pcts``, which carry the
    pre-selection ACTIVE set that builds ``fp_targets``.

    ``steered_positions`` (the dataset positions of a ``SteeredTaskMixture``'s
    steered conversations) makes ``seg_doc_id`` open a NEW prediction group at
    each steered conversation's first assistant-turn segment, so the FP loss
    never predicts the deliberately perturbed response features from the user
    turn. It requires turn-delimited features and leaves every non-steered row
    byte-identical. ``None`` skips the whole mechanism, including the logical
    turn recovery its span check needs.

    ``single_pass=True`` (validation) forbids wrapping: the cursor walks its
    rank's slice of the covered documents exactly once, then pads instead of
    re-serving an index, and sets ``sft_state.exhausted`` when the packing
    buffer has drained. Without it the final validation batch packed wrapped
    duplicates into the BPB sums, so "BPB over distinct documents" was simply
    untrue (review finding 4, PR #1311). Training keeps ``single_pass=False``
    and wraps for multi-epoch runs.

    Yields:
        (inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets)
        when FP is inactive and ``return_seg_doc_id=False``, or
        (inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id)
        when ``return_seg_doc_id=True`` or FP targets are present. FP-active
        batches always include ``seg_doc_id`` because the model requires it to
        mask cross-document next-segment targets.
        - inputs: (B, T) int32 token IDs
        - targets: (B, T) int64 targets (-1 = masked)
        - enc_fids: (B, n_seg, F_max) int64 feature IDs
        - enc_mags: (B, n_seg, F_max) int64 magnitude buckets
        - enc_mask: (B, n_seg, F_max) bool mask
        - seg_map: (B, T) int64 position-to-segment mapping
        - fp_targets: (B, n_seg, feature_prediction_width) float32 dense targets,
          or ``None`` when fp is inactive
        - seg_doc_id: (B, n_seg) int64 FP prediction-group id per segment; -1
          for padding/empty segments. Segment ``s`` may predict ``s+1`` only
          within one group (one packed conversation, split again at a steered
          conversation's first assistant-turn segment).
    """
    task_size = len(task)
    assert task_size > 0, "Task has no examples"
    sft_state.consumed = NonNegativeInt.parse(sft_state.consumed)
    if sft_state.dataloader_state is None and sft_state.consumed != 0:
        raise DataLoadError(
            "conditioned SFT cannot resume from SFTState.consumed alone; an exact "
            "resume requires buffered_v2 dataloader state"
        )
    normalized_resume_state = (
        None
        if sft_state.dataloader_state is None
        else _validate_sft_dataloader_state(
            sft_state.dataloader_state,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            buffer_size=buffer_size,
            device_batch_size=B,
            sequence_length=T,
            grad_accum_steps=grad_accum_steps,
            min_loss_tokens=min_loss_tokens,
            loss_mask_mode=loss_mask_mode,
        )
    )
    if (
        normalized_resume_state is not None
        and sft_state.consumed != normalized_resume_state["consumed"]
    ):
        raise DataLoadError(
            "conditioned SFT SFTState.consumed disagrees with its dataloader "
            f"state: {sft_state.consumed} != {normalized_resume_state['consumed']}"
        )
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "make_sft_data_generator accepts fp_target_layout or the legacy "
            "n_sae_features argument, not both"
        )
    if fp_target_layout is not None and fp_target_layout.is_compact:
        if fp_target_layout.feature_mask != feature_mask:
            raise ValueError(
                "fp_target_layout was derived from a different feature_mask than the SFT generator"
            )
        if fp_target_layout.feature_clusters != consolidation:
            raise ValueError(
                "fp_target_layout was derived from different feature_clusters than the SFT generator"
            )
    # ONE contract for every percentile consumer below. ``None`` resolves to the
    # production ``error`` policy, which substitutes nothing and leaves every
    # emitted tensor byte-identical to a run without this plumbing.
    missing_percentile_resolver = (
        MissingPercentileResolver.error() if missing_percentile is None else missing_percentile
    )
    if precomputed_thresholds is not None:
        # Uncovered features carry a 0.0 per-feature threshold, which ADMITS
        # every positive magnitude. Under ``lowest`` that would contradict the
        # rank-0.5 floor the percentile-space selectors give the same id.
        precomputed_thresholds = missing_percentile_resolver.mask_precomputed_thresholds(
            precomputed_thresholds
        )
    row_capacity = T + 1  # +1 for target at last position

    bos_token = tokenizer.get_bos_token_id()
    # ``specials is not None`` IS the steered gate: recovering logical turns
    # needs ``encode_special``, which the synthetic tokenizers of non-steered
    # harnesses do not have, so a non-steered run must never reach for it.
    specials = None if steered_positions is None else ChatSpecialTokens.from_tokenizer(tokenizer)
    steered_row_positions = steered_positions or frozenset()

    # Load full-sequence features once and exclude rows missing extraction data.
    sft_features: Mapping[int, list[SegmentFeatures]] | None = None
    assert feature_dir is not None, "feature_dir required for conditioned SFT"
    assert parquet_dir is not None, "parquet_dir required for conditioned SFT"
    assert threshold_cfg is not None or sampling_bundle is not None, (
        "threshold_cfg or sampling_bundle required for conditioned SFT"
    )
    assert boundaries is not None, "boundaries required for conditioned SFT"
    # ORDERING CONSTRAINT: this frozen snapshot must be built AFTER every
    # rebind of the values it captures — in particular
    # ``missing_percentile_resolver`` and the
    # ``mask_precomputed_thresholds`` rebind of ``precomputed_thresholds``
    # just above. The dataclass copies references at construction time, so
    # hoisting it earlier would silently ship the pre-policy thresholds to
    # every conversation while the rest of the generator used the masked ones.
    selection_ctx = SegmentSelectionContext(
        threshold_cfg=threshold_cfg,
        precomputed_thresholds=precomputed_thresholds,
        sampling_bundle=sampling_bundle,
        negative_bundle=negative_bundle,
        boundaries=boundaries,
        fp_percentile_table=fp_percentile_table,
        cluster_threshold_percentile_table=cluster_threshold_percentile_table,
        feature_mask=feature_mask,
        consolidation=consolidation,
        missing_percentile=missing_percentile_resolver,
    )
    with _maybe_timed_region("sft/load_features", profile_markers=profile_markers):
        sft_features = _load_sft_features_for_startup(
            feature_dir,
            parquet_dir,
            startup_trace=startup_trace,
            source=expected_split.value,
        )
    store_segment_mode = getattr(sft_features, "segment_mode", None)
    if specials is not None and store_segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
        raise DataLoadError(
            "steered conversations require turn-delimited SFT features "
            f"(store segment_mode={store_segment_mode}). Feature dir: {feature_dir}."
        )
    actual_split = getattr(sft_features, "conversion_split", None)
    conversion_mixture = getattr(sft_features, "conversion_mixture", None)
    coverage_extra: dict[str, object] = {"source": expected_split.value}
    with _maybe_sft_startup_phase(
        startup_trace,
        "sft_artifact_identity",
        extra=lambda: coverage_extra,
    ):
        if expected_split is SFTSplit.TRAIN and actual_split is None:
            logger.info(
                f"SFT parquet metadata at {parquet_dir} has no split field; "
                "accepting it as a legacy train conversion"
            )
        elif actual_split is not expected_split:
            recovery_dataset = (
                conversion_mixture
                if isinstance(conversion_mixture, str)
                else _infer_sft_mixture_from_parquet_dir(parquet_dir)
            )
            actual_split_value = (
                actual_split.value if isinstance(actual_split, SFTSplit) else actual_split
            )
            raise DataLoadError(
                "conditioned SFT parquet split mismatch: "
                f"expected_split={expected_split.value!r}, conversion_split={actual_split_value!r}, "
                f"conversion_mixture={conversion_mixture!r}. "
                f"Feature dir: {feature_dir}. Parquet dir: {parquet_dir}.\n"
                + _sft_artifact_recovery_instructions(
                    recovery_dataset,
                    split=expected_split,
                    steered_jsonl=_task_steered_jsonl(task),
                )
            )
        _assert_sft_mixture_fingerprint_matches_task(
            store=sft_features,
            task=task,
            mixture_name=mixture_name,
            expected_split=expected_split,
            feature_dir=feature_dir,
            parquet_dir=parquet_dir,
            alignment_probe_samples=alignment_probe_samples,
        )
    # A capped conversion stores a covered prefix of the full, fingerprinted
    # mixture.  The conversion metadata records the attempted prefix length;
    # the parquet's own ``original_index`` column is the authoritative set of
    # rows that survived conversion.  ``convert_sft_to_parquet`` legitimately
    # drops ``no_assistant`` / ``empty_response`` conversations, so that set is
    # a subset of the attempted prefix even for an UNCAPPED run — demanding
    # ``range(task_size)`` there would make a normal full-mixture conversion
    # fatal, with a "re-extract features" instruction no re-extraction can
    # satisfy (review findings 6 + 14, PR #1311).
    # Legacy artifacts have no coverage field and retain the historical
    # full-mixture requirement below.
    with _maybe_sft_startup_phase(
        startup_trace,
        "sft_document_coverage",
        extra=lambda: coverage_extra,
    ):
        conversion_coverage, dataset_indices = _resolve_sft_dataset_indices(
            sft_features,
            task_size=task_size,
            feature_dir=feature_dir,
            parquet_dir=parquet_dir,
        )
        dataset_size = len(dataset_indices)
        coverage_extra["documents"] = dataset_size
        if conversion_coverage is not None:
            coverage_extra["conversion_prefix"] = conversion_coverage
    rank_dataset_size = len(range(ddp_rank, dataset_size, ddp_world_size))
    if specials is not None:
        # Steered rows are evenly spaced at DATASET level, so a rank's
        # ``[ddp_rank::ddp_world_size]`` stride can still be unevenly steered.
        # That is accepted; the per-rank share is logged so it is visible.
        rank_indices = dataset_indices[ddp_rank::ddp_world_size]
        steered_in_stride = sum(1 for index in rank_indices if index in steered_row_positions)
        logger.info(
            f"Steered SFT coverage: rank {ddp_rank}/{ddp_world_size} serves "
            f"{steered_in_stride}/{len(rank_indices)} steered documents"
        )
    if single_pass and dataset_size < ddp_world_size:
        # Each rank walks ``dataset_indices[ddp_rank::ddp_world_size]`` exactly
        # once; with fewer documents than ranks some rank owns none and could
        # only produce an empty batch. Refuse rather than emit one.
        raise DataLoadError(
            f"conditioned SFT single-pass generator covers {dataset_size} "
            f"documents across {ddp_world_size} ranks; every rank needs at "
            "least one covered document. Use a larger validation conversion or "
            "fewer ranks."
        )
    sft_state.dataset_size = dataset_size
    # The resume cursor is only meaningful against the exact artifacts it was
    # advanced over. Pin BOTH sides: the parquet/task-mixture identity and the
    # feature-side identity, so re-pointing ``feature_dir`` at a different
    # extraction of the same mixture (other layer, SAE width or pooling) is
    # rejected rather than silently resumed (review finding 7, PR #1311).
    # The shard counts are deliberately NOT recorded: ``SFTConversionMeta``
    # types both as required ints and ``load_sft_features`` already rejects
    # any inequality, so they are always equal and carry no identity.
    sft_state.artifact_coverage = {
        "dataset_size": dataset_size,
        "conversion_total_conversations": conversion_coverage,
        "parquet_mixture_fingerprint": getattr(
            sft_features, "parquet_mixture_fingerprint", None
        ),
        "feature_mixture_fingerprint": getattr(
            sft_features, "feature_mixture_fingerprint", None
        ),
    }

    # Check feature coverage before training. Full-sequence SFT requires one
    # extracted feature row per converted task example; missing rows in legacy
    # artifacts usually indicate a stale/misaligned parquet shard or a failed
    # extraction pass. Do not silently train on a filtered full-mixture dataset.
    with _maybe_sft_startup_phase(
        startup_trace,
        "sft_feature_row_coverage",
        extra=lambda: coverage_extra,
    ):
        missing_indices = [i for i in dataset_indices if i not in sft_features]
        coverage_extra["missing_documents"] = len(missing_indices)
    if missing_indices:
        missing_frac = len(missing_indices) / dataset_size
        raise ValueError(
            f"conditioned SFT: {len(missing_indices)}/{dataset_size} "
            f"({missing_frac:.2%}) examples missing features from "
            f"{feature_dir!r}. First 10 missing: {missing_indices[:10]}. "
            "Re-extract features over the full SFT task mixture before training."
        )
    alignment_extra: dict[str, object] = {
        "source": expected_split.value,
        "candidates": len(dataset_indices),
        "requested": alignment_probe_samples,
        "sampled": max(0, min(len(dataset_indices), alignment_probe_samples)),
        "matched": 0,
    }
    sampled_indices: list[int] = []
    with _maybe_sft_startup_phase(
        startup_trace,
        "sft_alignment_probe",
        extra=lambda: alignment_extra,
    ):
        sampled_indices = _run_sft_content_alignment_probe(
            store=sft_features,
            task=task,
            tokenizer=tokenizer,
            alignment_probe_samples=alignment_probe_samples,
            probe_indices=dataset_indices,
        )
        alignment_extra["matched"] = len(sampled_indices)

    # Buffer entries retain their logical striped cursor position. Checkpoints
    # replay the still-buffered positions in this same order before advancing
    # from ``next_cursor``; rematerializing only from a consumed count loses
    # that information because best-fit packing pops entries out of order.
    # Turn starts are the conversation's source-coordinate logical turn
    # starts, present ONLY for a steered conversation — so ``is not None`` is
    # also the per-conversation steered flag, with no second marker to drift.
    # Prepared routing travels with the row through best-fit packing.
    conv_buffer: list[
        tuple[
            int,
            list[int],
            list[int],
            list[SegFeatures],
            tuple[int, ...] | None,
            FullSequenceSegmentRouting,
        ]
    ] = []
    if normalized_resume_state is None:
        cursor = ddp_rank + sft_state.consumed
        replay_positions: deque[int] = deque()
        consumed = sft_state.consumed
    else:
        cursor = normalized_resume_state["next_cursor"]
        replay_positions = deque(normalized_resume_state["buffered_cursor_positions"])
        consumed = normalized_resume_state["consumed"]
        sft_state.consumed = consumed
    _trunc_processed = 0
    _trunc_skipped = 0
    _first_pass_logged = False
    steered_seen = 0
    steered_skipped_min_loss_tokens = 0
    steered_without_assistant_segment = 0
    initial_batch_started_ns = (
        time.perf_counter_ns() if startup_trace is not None else None
    )
    initial_batch_recorded = False

    def _log_first_pass_summary() -> None:
        nonlocal _first_pass_logged
        if _first_pass_logged or _trunc_processed < rank_dataset_size:
            return
        level = logger.warning if _trunc_skipped else logger.info
        level(
            f"SFT buffer first pass: skipped {_trunc_skipped}/"
            f"{_trunc_processed} conversations with "
            f"<{min_loss_tokens} supervised loss tokens after truncation "
            f"(min_loss_tokens={min_loss_tokens})"
        )
        if specials is not None:
            # ``steered_without_assistant_segment`` is a PACKING count while the
            # trigger above is a RENDER count, so it trails by at most
            # ``buffer_size`` conversations out of a whole first pass.
            # The counters and trigger both describe this rank's local share,
            # including renders skipped by min_loss_tokens. They reset when
            # a generator is constructed, including checkpoint resume.
            logger.info(
                f"SFT steered rows first pass ({_trunc_processed} conversations "
                f"rendered on this rank): {steered_seen} seen, "
                f"{steered_skipped_min_loss_tokens} skipped by min_loss_tokens, "
                f"{steered_without_assistant_segment} packed with no assistant-turn segment"
            )
        _first_pass_logged = True

    def refill_buffer():
        nonlocal cursor, _trunc_processed, _trunc_skipped, _first_pass_logged
        nonlocal steered_seen, steered_skipped_min_loss_tokens
        # _trunc_processed/_trunc_skipped grow monotonically across all
        # refill_buffer() calls for the lifetime of the generator.  The
        # periodic summary fires at every 1000th conversation processed.
        consecutive_skips = 0
        while len(conv_buffer) < buffer_size:
            replaying = bool(replay_positions)
            if single_pass and not replaying and cursor >= dataset_size:
                # Bounded (validation) pass: this rank has served its whole
                # slice. Never wrap — ``dataset_indices[cursor % dataset_size]``
                # would re-emit an index already scored in this eval pass.
                # ``exhausted`` only latches once the buffer has ALSO drained,
                # so buffered-but-unpacked conversations are not dropped.
                if not conv_buffer:
                    sft_state.exhausted = True
                return
            if consecutive_skips >= dataset_size:
                # Full dataset cycle without finding a valid conversation.
                # Using dataset_size (not ceil(dataset_size/world_size))
                # because with non-divisible DDP a rank can cycle through
                # more than its "natural share" of indices before repeating.
                # DDP propagation happens at the aligned ``next(batch)``
                # boundary via ``rank_synchronized_stage``. Do not put a
                # collective here: local refill loops can diverge by rank.
                raise RuntimeError(
                    f"SFT buffer starvation: scanned all {dataset_size} "
                    f"conversations without finding one with >= {min_loss_tokens} "
                    "supervised loss tokens after truncation to "
                    f"{row_capacity} tokens. "
                    f"The min_loss_tokens config knob is set to "
                    f"{min_loss_tokens}; lower it, increase max_seq_len "
                    f"(null uses the checkpoint's sequence_len), or check "
                    f"that the dataset has conversations with enough "
                    f"rendered tokens. (ddp_rank={ddp_rank})"
                )
            if replaying:
                logical_position = replay_positions.popleft()
            else:
                logical_position = cursor
                cursor += ddp_world_size
            example_idx = dataset_indices[logical_position % dataset_size]

            conversation = task[example_idx]
            with _maybe_timed_region("sft/render_conversation", profile_markers=profile_markers):
                # Render the converter's slice, not the raw row: the parquet
                # ``text`` and every segment char offset stop at the last
                # assistant turn, so a trailing user turn would add decoder
                # text no feature segment covers (issue #1465).
                messages = _prepare_sft_render_messages(
                    conversation,
                    surface="data generator",
                    original_index=example_idx,
                )
                ids, loss_mask = render_conversation_with_mask(tokenizer, {"messages": messages})

            # --- Locate the steered assistant turn on the FULL render ---
            # Truncation rewrites ``ids`` but never a segment's ``char_start``,
            # so the turn boundary the prediction-group split keys on must be
            # read here, before the render is cut down to ``row_capacity``.
            turn_src_starts: tuple[int, ...] | None = None
            if specials is not None and example_idx in steered_row_positions:
                spans = legacy_turn_spans(ids, specials)
                roles = tuple(span.role for span in spans)
                if roles != (TurnRole.USER, TurnRole.ASSISTANT):
                    raise DataLoadError(
                        f"conditioned SFT steered conversation original_index={example_idx} "
                        "must render exactly one user turn followed by one assistant turn; "
                        f"got roles {roles}"
                    )
                # Turn 0 starts at source char 0: extraction absorbs the source
                # model's BOS into it. Target-space span start ``j`` is
                # BOS-inclusive token index ``j + 1``.
                # Keep the literal 0 — do NOT "fix" it to the first non-BOS
                # token's offset. It is only ever a lower bound, and it is the
                # bound that makes a segment starting before turn 0 map to turn
                # -1 and fail loud; ``steered_prediction_groups`` also rejects
                # a first start that is not 0.
                turn_src_starts = (
                    0,
                    _source_aligned_char_offsets(
                        tokenizer, ids, source_bos_char_len=feature_source_bos_char_len
                    )[spans[1].start + 1],
                )
                steered_seen += 1

            assert sft_features is not None
            prepared = prepare_sft_example(
                ids=ids,
                loss_mask=loss_mask,
                row_capacity=row_capacity,
                loss_mask_mode=loss_mask_mode,
                min_loss_tokens=min_loss_tokens,
                select_segments=lambda example_idx=example_idx: select_conversation_segments(
                    sft_features[example_idx], ctx=selection_ctx, example_idx=example_idx
                ),
                tokenizer=tokenizer,
                feature_source_bos_char_len=feature_source_bos_char_len,
                example_idx=example_idx,
                profile_markers=profile_markers,
            )
            ids, loss_mask = prepared.ids, prepared.loss_mask
            original_len, was_truncated = prepared.original_len, prepared.was_truncated
            n_loss_tokens = prepared.n_loss_tokens
            _trunc_processed += 1
            if n_loss_tokens < min_loss_tokens:
                if replaying:
                    raise DataLoadError(
                        "conditioned SFT buffered conversation no longer satisfies "
                        f"min_loss_tokens={min_loss_tokens} at logical cursor "
                        f"{logical_position}; the checkpoint inputs changed"
                    )
                _trunc_skipped += 1
                if turn_src_starts is not None:
                    steered_skipped_min_loss_tokens += 1
                _log_skipped_conv(
                    example_idx=example_idx,
                    original_len=original_len,
                    truncated_len=len(ids),
                    n_loss_tokens=n_loss_tokens,
                    min_loss_tokens=min_loss_tokens,
                )
                _log_periodic_summary(
                    processed=_trunc_processed,
                    skipped=_trunc_skipped,
                    min_loss_tokens=min_loss_tokens,
                )
                _log_first_pass_summary()
                consecutive_skips += 1
                continue

            consecutive_skips = 0

            if was_truncated:
                _log_truncated_conv(
                    example_idx=example_idx,
                    original_len=original_len,
                    truncated_len=len(ids),
                )

            _log_periodic_summary(
                processed=_trunc_processed,
                skipped=_trunc_skipped,
                min_loss_tokens=min_loss_tokens,
            )
            _log_first_pass_summary()

            assert prepared.routing is not None
            seg_features_list = prepared.segments
            conv_buffer.append(
                (
                    logical_position,
                    ids,
                    loss_mask,
                    seg_features_list,
                    turn_src_starts,
                    prepared.routing,
                )
            )

    use_cuda = str(device).startswith("cuda")

    while True:
        rows: list[list[int]] = []
        row_loss_masks: list[list[int]] = []
        row_lengths: list[int] = []
        # Track conversation boundaries per row for feature building.
        row_convs: list[list[_PackedConversation]] = []

        for _ in range(B):
            row_ids: list[int] = []
            row_loss_mask: list[int] = []
            convs_in_row: list[_PackedConversation] = []
            padded = False

            while len(row_ids) < row_capacity:
                with _maybe_timed_region("sft/refill_buffer", profile_markers=profile_markers):
                    refill_buffer()

                remaining = row_capacity - len(row_ids)

                # Best-fit: find largest conversation that fits
                best_idx = -1
                best_len = 0
                for i, (
                    _position,
                    ids,
                    _loss_mask,
                    _seg_features_list,
                    _turns,
                    _routing,
                ) in enumerate(conv_buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx = i
                        best_len = len(ids)

                if best_idx >= 0:
                    _position, ids, loss_mask, seg_feats, turns, routing = conv_buffer.pop(best_idx)
                    consumed += ddp_world_size
                    sft_state.consumed = consumed
                    if single_pass and cursor >= dataset_size and not conv_buffer:
                        # Latch as soon as the LAST covered conversation is
                        # packed, so the eval loop stops without first asking
                        # for an entirely padded batch.
                        sft_state.exhausted = True
                    conv_start = len(row_ids)
                    row_ids.extend(ids)
                    row_loss_mask.extend(loss_mask)
                    conv_end = len(row_ids)
                    convs_in_row.append(
                        (conv_start, conv_end, seg_feats, turns, _position, routing)
                    )
                else:
                    # Pad remainder
                    content_len = len(row_ids)
                    pad_len = remaining
                    row_ids.extend([bos_token] * pad_len)
                    row_loss_mask.extend([0] * pad_len)
                    padded = True
                    break

            if padded:
                # A fully padded row is legitimate ONLY in a bounded validation
                # pass whose covered documents ran out mid-batch: every target
                # in it is -1, so it contributes nothing to the BPB sums. In
                # every other mode it means refill_buffer is broken.
                if content_len == 0 and not (single_pass and sft_state.exhausted):
                    raise RuntimeError(
                        "Empty row: no conversation was packed before padding. "
                        "After truncation, all buffer conversations should fit "
                        "row_capacity. This indicates a bug in refill_buffer."
                    )
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)

            rows.append(row_ids[:row_capacity])
            row_loss_masks.append(row_loss_mask[:row_capacity])
            row_convs.append(convs_in_row)

        # Build tensors
        with _maybe_timed_region("sft/build_text_tensors", profile_markers=profile_markers):
            batch_tensor = t.tensor(rows, dtype=t.long, pin_memory=use_cuda)

            # Slicing ``batch_tensor`` yields non-contiguous views (stride keeps
            # the original row pitch ``T+1`` while shape is ``(B, T)``).  The
            # ``.to(dtype=...)`` calls happen to copy for ``inputs`` (int64 →
            # int32 forces a new tensor) but are no-ops for ``targets``
            # (already int64, same device on CPU runs).  That leaves ``targets``
            # as a non-contiguous slice which crashes ``targets.view(-1)`` inside
            # ``F.cross_entropy``.  In production on GPU the
            # ``.to(device="cuda")`` copy hid the bug; the tiny CPU integration
            # tests expose it.  Force contiguity explicitly so neither code path
            # depends on an implicit PyTorch optimisation.
            inputs = (
                batch_tensor[:, :-1]
                .to(device=device, dtype=t.int32, non_blocking=use_cuda)
                .contiguous()
            )
            targets = (
                batch_tensor[:, 1:]
                .to(device=device, dtype=t.int64, non_blocking=use_cuda)
                .contiguous()
            )

            # Padding is always ignored, independent of the configured loss mode.
            # ``max(..., 0)`` covers the fully padded row a bounded validation
            # pass can emit after its covered documents run out: the whole row
            # is masked instead of wrapping to a slice-from-the-end.
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, max(content_len - 1, 0) :] = -1
            if loss_mask_mode is LossMaskMode.COMPLETION_ONLY:
                apply_completion_only_row_masks(targets, row_loss_masks)
                # Fully padded rows carry no supervision by construction; the
                # all-masked-row guard applies to rows that hold content. The
                # common case keeps the original tensor (and therefore the
                # original row indices in the error message).
                content_rows = [i for i, length in enumerate(row_lengths) if length > 0]
                assert_min_one_valid_target(
                    targets if len(content_rows) == B else targets[content_rows]
                )

        # Build encoder inputs
        fp_targets_batch: t.Tensor | None = None
        # Build per-row encoder inputs from packed conversations
        row_enc_fids: list[t.Tensor] = []
        row_enc_mags: list[t.Tensor] = []
        row_enc_masks: list[t.Tensor] = []
        row_seg_maps: list[t.Tensor] = []
        row_seg_doc_ids: list[list[int]] = []
        # Track per-row, per-segment all-feature data for fp_targets.
        # row_all_pairs[row_idx] is a list of (all_fids, all_pcts) tuples
        # ordered identically to the encoder segments (segment 0 = empty).
        row_all_pairs: list[list[tuple[t.Tensor | None, t.Tensor | None]]] = []

        for row_idx in range(B):
            all_seg_fids: list[t.Tensor] = [t.tensor([], dtype=t.int64)]  # segment 0: empty
            all_seg_buckets: list[t.Tensor] = [t.tensor([], dtype=t.int64)]
            all_seg_pcts: list[t.Tensor] = [t.tensor([], dtype=t.float32)]
            row_doc_ids: list[int] = [-1]
            row_pairs: list[tuple[t.Tensor | None, t.Tensor | None]] = [
                (None, None)  # segment 0: empty fp data
            ]
            row_seg_map = t.zeros(T, dtype=t.int64)
            global_seg = 1
            # FP prediction-group id: +1 at every conversation start and, for a
            # steered conversation, +1 more at its first assistant-turn
            # segment. Without steered rows it equals ``doc_idx`` exactly.
            group_id = -1

            # rows are row_capacity = T+1 tokens; inputs use [:-1] so positions [0, T)
            for (
                conv_start,
                conv_end,
                conv_seg_feats,
                turn_src_starts,
                position,
                routing,
            ) in row_convs[row_idx]:
                group_id += 1
                # Extract this conversation's ids in the inputs view [:-1].
                # conv_start/conv_end are in the row_capacity coordinate system
                # For inputs, we use positions [0, T) which is row[:-1]
                inp_start = conv_start
                inp_end = min(conv_end, row_capacity) - 1  # -1 because inputs = row[:-1]
                if inp_start >= T or inp_end <= inp_start:
                    logger.debug(
                        f"Skipping conversation with invalid range: "
                        f"inp_start={inp_start}, inp_end={inp_end}, T={T}"
                    )
                    continue

                # Recover the in-range and prediction-only features for FP targets.
                # Filter out prediction_target_only segments — those are
                # kept past the truncation boundary for feature prediction
                # targets but have no text in the decoder. They are appended
                # to the encoder input separately after routing.
                extra_pred_seg: SegFeatures | None = None
                in_range = [sf for sf in conv_seg_feats if not sf.prediction_target_only]
                pred_only = [sf for sf in conv_seg_feats if sf.prediction_target_only]
                assert len(pred_only) <= 1, (
                    f"Expected at most 1 prediction_target_only segment per "
                    f"conversation, got {len(pred_only)}"
                )
                extra_pred_seg = pred_only[0] if pred_only else None
                if not in_range:
                    raise ValueError(
                        "conditioned SFT conversation has no in-range feature "
                        "segments after truncation"
                    )
                local_fids = routing.per_seg_fids
                local_buckets = routing.per_seg_buckets
                local_seg_map = routing.seg_map
                assert len(local_seg_map) == inp_end - inp_start

                # Remap local segments to global: local 0 → global 0, local 1+ → global_seg+
                #
                # Writing the ``local == 0`` positions as an explicit 0 is
                # equivalent to the per-position loop that skipped them:
                # ``row_seg_map`` is zero-initialized and each conversation's
                # ``[inp_start, inp_end)`` span is disjoint from every other's
                # (``inp_end = conv_end - 1 < conv_end <=`` the next
                # ``inp_start``), so nothing else has written there.
                n_local_segs = len(local_fids)
                local_len = len(local_seg_map)
                remapped = t.where(
                    local_seg_map > 0,
                    local_seg_map + (global_seg - 1),
                    t.zeros((), dtype=t.int64),
                )
                row_seg_map[inp_start : inp_start + local_len] = remapped

                # Append feature segments (skip local segment 0).
                # Full-conversation routing consumes the in-range source
                # list (via ``seg_feats_tuple``) in order, so local
                # segment ``s+1`` corresponds to ``in_range[s-1]`` for
                # the purposes of looking up all-feature fp data. Guard
                # against shorter source lists (shouldn't happen given
                # the assertion in full-conversation routing but defensive).
                in_range_for_fp = [sf for sf in conv_seg_feats if not sf.prediction_target_only]
                # Steered conversations split into two prediction groups at the
                # first assistant-turn segment; every other conversation is one
                # group, so ``seg_groups is None`` keeps the legacy ids exactly.
                seg_groups: list[int] | None = None
                if turn_src_starts is not None:
                    seg_char_starts = [
                        sf.char_start for sf in in_range_for_fp[: n_local_segs - 1]
                    ]
                    if extra_pred_seg is not None:
                        seg_char_starts.append(extra_pred_seg.char_start)
                    try:
                        seg_groups = steered_prediction_groups(
                            seg_char_starts, turn_src_starts, base_group=group_id
                        )
                    except RuntimeError as error:
                        raise RuntimeError(
                            "conditioned SFT example "
                            f"{dataset_indices[position % dataset_size]}: {error}"
                        ) from error
                    if not seg_groups or seg_groups[-1] == group_id:
                        steered_without_assistant_segment += 1
                for s in range(1, n_local_segs):
                    all_seg_fids.append(local_fids[s])
                    all_seg_buckets.append(local_buckets[s])
                    all_seg_pcts.append(t.zeros(len(local_fids[s]), dtype=t.float32))
                    src_idx = s - 1
                    # Bounds check: full-conversation routing guarantees
                    # ``n_local_segs - 1 <= len(in_range_for_fp)`` because
                    # it consumes the in-range source list in order and
                    # only ever drops trailing segments via the
                    # excess-coverage filter. If that contract ever
                    # changes, the assertion fails loud at the first
                    # affected batch instead of silently zeroing the
                    # fp targets for the orphaned segments.
                    assert src_idx < len(in_range_for_fp), (
                        f"local segment {s} has no source in_range entry "
                        f"(src_idx={src_idx}, len(in_range_for_fp)="
                        f"{len(in_range_for_fp)}). Full-conversation routing "
                        f"contract violated: produced more segments than "
                        f"the in-range source list contains."
                    )
                    src = in_range_for_fp[src_idx]
                    row_pairs.append((src.all_feature_ids, src.all_feature_pcts))
                    row_doc_ids.append(group_id if seg_groups is None else seg_groups[src_idx])
                global_seg += max(0, n_local_segs - 1)

                # Append the prediction-target-only extra segment (at
                # most one per conversation) to the encoder input.
                # It has features but no decoder positions — seg_map
                # is NOT updated.  Note: a character gap exists between
                # the clipped last in-range segment and this extra
                # segment (e.g. clipped to char 128 but extra starts
                # at char 150).  This is intentional — the extra segment
                # is only used as a feature-prediction target, not for
                # cross-attention conditioning of decoder positions.
                if extra_pred_seg is not None:
                    all_seg_fids.append(extra_pred_seg.feature_ids)
                    all_seg_buckets.append(extra_pred_seg.buckets)
                    all_seg_pcts.append(extra_pred_seg.percentiles)
                    global_seg += 1
                    row_pairs.append(
                        (extra_pred_seg.all_feature_ids, extra_pred_seg.all_feature_pcts)
                    )
                    row_doc_ids.append(group_id if seg_groups is None else seg_groups[-1])
                if seg_groups:
                    group_id = seg_groups[-1]

            # Pad and build per-row tensors
            with _maybe_timed_region("sft/pad_segments", profile_markers=profile_markers):
                fids_row, mags_row, _, mask_row, seg_map_padded = _pad_segments(
                    all_seg_fids,
                    all_seg_buckets,
                    all_seg_pcts,
                    row_seg_map,
                )
            row_enc_fids.append(fids_row)
            row_enc_mags.append(mags_row)
            row_enc_masks.append(mask_row)
            row_seg_maps.append(seg_map_padded)
            row_all_pairs.append(row_pairs)
            row_seg_doc_ids.append(row_doc_ids)

        # Stack across rows, padding n_seg and F_max to batch max
        with _maybe_timed_region("sft/stack_encoder_inputs", profile_markers=profile_markers):
            enc_fids, enc_mags, enc_mask_out = _stack_padded_encoder_inputs(
                row_enc_fids,
                row_enc_mags,
                row_enc_masks,
                device=device,
                use_cuda=use_cuda,
            )
            batch_n_seg = enc_fids.shape[1]

        seg_map = t.stack(row_seg_maps).to(device)
        seg_doc_id_cpu = t.full((B, batch_n_seg), -1, dtype=t.int64)
        for row_idx, doc_ids in enumerate(row_seg_doc_ids):
            n_row_seg = min(len(doc_ids), batch_n_seg)
            if n_row_seg > 0:
                seg_doc_id_cpu[row_idx, :n_row_seg] = t.tensor(doc_ids[:n_row_seg], dtype=t.int64)
        seg_doc_id = (
            seg_doc_id_cpu.to(device, non_blocking=use_cuda)
            if str(device) != "cpu"
            else seg_doc_id_cpu
        )

        # Build dense fp_targets by scattering all-feature percentiles into
        # [B, batch_n_seg, feature_prediction_width]. Active only when the
        # caller provides an FP width and fp_percentile_table (the fp head
        # needs both to interpret the targets).
        #
        # We build the dense tensor on CPU first and only transfer it
        # to ``device`` once at the end, which avoids one HtoD transfer
        # per (row, segment) pair and keeps the inner scatter loop
        # device-local. The SegFeatures objects populated by
        # ``refill_buffer`` carry their CPU-side ``all_feature_ids`` /
        # ``all_feature_pcts`` tensors, so this matches where the data
        # naturally lives.
        fp_source_width = (
            fp_target_layout.n_sae_features if fp_target_layout is not None else n_sae_features
        )
        if fp_source_width is not None and fp_percentile_table is not None:
            with _maybe_timed_region("sft/build_fp_targets", profile_markers=profile_markers):
                fp_output_width = (
                    fp_target_layout.output_width
                    if fp_target_layout is not None
                    else fp_source_width
                )
                fp_targets_cpu = t.zeros(
                    B,
                    batch_n_seg,
                    fp_output_width,
                    dtype=t.float32,
                )
                for row_idx in range(B):
                    pairs = row_all_pairs[row_idx]
                    for seg_i, (fids, pcts) in enumerate(pairs):
                        if seg_i >= batch_n_seg:
                            break
                        if fids is not None and pcts is not None and len(fids) > 0:
                            source_ids = fids.to(t.int64)
                            if fp_target_layout is None or not fp_target_layout.is_compact:
                                fp_targets_cpu[row_idx, seg_i, source_ids] = pcts.to(
                                    fp_targets_cpu.dtype
                                )
                            else:
                                if bool(((source_ids < 0) | (source_ids >= fp_source_width)).any()):
                                    bad_id = int(
                                        source_ids[
                                            (source_ids < 0) | (source_ids >= fp_source_width)
                                        ][0].item()
                                    )
                                    raise ValueError(
                                        f"SFT FP source feature ID {bad_id} is outside "
                                        f"[0, {fp_source_width})"
                                    )
                                original_to_output = fp_target_layout.original_to_output_row
                                assert original_to_output is not None
                                output_rows = original_to_output[source_ids]
                                keep = output_rows >= 0
                                fp_targets_cpu[row_idx, seg_i].scatter_reduce_(
                                    0,
                                    output_rows[keep],
                                    pcts[keep].to(fp_targets_cpu.dtype),
                                    reduce="amax",
                                    include_self=True,
                                )
                if fp_target_layout is None or not fp_target_layout.is_compact:
                    fp_targets_cpu = _zero_masked_fp_targets(
                        fp_targets_cpu,
                        feature_mask,
                        fp_source_width,
                    )
                fp_targets_batch = (
                    fp_targets_cpu.to(device, non_blocking=use_cuda)
                    if str(device) != "cpu"
                    else fp_targets_cpu
                )

        # Publish only after every tensor has been built successfully. The
        # caller may now train this exact batch; the training loop snapshots
        # this produced state only after its optimizer step succeeds.
        sft_state.dataloader_state = {
            "ddp_rank": ddp_rank,
            "ddp_world_size": ddp_world_size,
            "consumed": consumed,
            "next_cursor": cursor,
            "buffered_cursor_positions": [entry[0] for entry in conv_buffer],
            "buffer_size": buffer_size,
            "device_batch_size": B,
            "sequence_length": T,
            "grad_accum_steps": grad_accum_steps,
            "min_loss_tokens": min_loss_tokens,
            "loss_mask_mode": loss_mask_mode.value,
        }

        if startup_trace is not None and not initial_batch_recorded:
            assert initial_batch_started_ns is not None
            startup_trace.record(
                "sft_initial_batch",
                time.perf_counter_ns() - initial_batch_started_ns,
                extra={
                    "source": expected_split.value,
                    "documents": dataset_size,
                    "batch_rows": B,
                    "sequence_length": T,
                    "segments": int(batch_n_seg),
                },
            )
            initial_batch_recorded = True

        if return_seg_doc_id or fp_targets_batch is not None:
            yield (
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask_out,
                seg_map,
                fp_targets_batch,
                seg_doc_id,
            )
        else:
            yield inputs, targets, enc_fids, enc_mags, enc_mask_out, seg_map, fp_targets_batch


@beartype
def make_sft_data_generator(
    task,
    tokenizer,
    B: int,
    T: int,
    device: str | t.device,
    sft_state: SFTState,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    buffer_size: int = 100,
    feature_dir: str | None = None,
    parquet_dir: str | None = None,
    threshold_cfg: ThresholdConfig | None = None,
    boundaries: t.Tensor | None = None,
    precomputed_thresholds: t.Tensor | None = None,
    sampling_bundle: SamplingBundle | None = None,
    min_loss_tokens: int = 8,
    n_sae_features: int | None = None,
    fp_percentile_table: t.Tensor | None = None,
    cluster_threshold_percentile_table: t.Tensor | None = None,
    *,
    grad_accum_steps: int = 1,
    negative_bundle: NegativeBundle | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    feature_mask: FeatureMask | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
    missing_percentile: MissingPercentileResolver | None = None,
    return_seg_doc_id: bool = False,
    feature_source_bos_char_len: int = 0,
    profile_markers: ProfileMarkerSwitch = False,
    startup_trace: _SFTStartupTrace | None = None,
    loss_mask_mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    alignment_probe_samples: int = 8,
    mixture_name: str = "unknown",
    single_pass: bool = False,
    expected_split: SFTSplit = SFTSplit.TRAIN,
    steered_positions: frozenset[int] | None = None,
):
    """Validate the public SFT generator contract before iteration starts."""

    if ddp_world_size <= 0 or not 0 <= ddp_rank < ddp_world_size:
        raise DataLoadError(
            "conditioned SFT requires ddp_world_size > 0 and "
            "0 <= ddp_rank < ddp_world_size; "
            f"got ddp_rank={ddp_rank}, ddp_world_size={ddp_world_size}"
        )

    if consolidation is not None:
        consolidation.assert_representatives_only(
            t.nonzero(
                consolidation.cluster_allowed_universe,
                as_tuple=False,
            ).squeeze(1),
            surface="conditional_encdec_sft configured representative universe",
        )
        if isinstance(threshold_cfg, ThresholdConfig):
            if cluster_threshold_percentile_table is None:
                cluster_threshold_percentile_table = fp_percentile_table
            if cluster_threshold_percentile_table is None:
                raise FeatureClustersError(
                    "clustered threshold SFT requires a full-distribution "
                    "percentile table through cluster_threshold_percentile_table "
                    "or fp_percentile_table"
                )

    # A clustered run MUST draw negatives from the representative-only pool.
    # ``negative_sampling_mask(pipeline)`` alone is cluster-blind, so a bundle
    # still carrying the plain feature mask would sample consolidated-away
    # siblings the compacted encoder has no row for. Caught here, at the public
    # boundary, rather than mid-stream on whichever conversation first draws
    # one.
    if negative_bundle is not None and consolidation is not None:
        expected_negative_mask = consolidation.effective_negative_mask()
        if negative_bundle.feature_mask != expected_negative_mask:
            raise FeatureClustersError(
                "conditional_encdec_sft negative_bundle carries feature_mask="
                f"{negative_bundle.feature_mask!r}, but a clustered run must "
                "sample negatives from the representative-only pool "
                f"{expected_negative_mask!r} "
                "(FeatureClusterConsolidation.effective_negative_mask()). Build "
                "it with negative_sampling_mask(pipeline, consolidation)."
            )

    return _make_sft_data_generator_impl(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device=device,
        sft_state=sft_state,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        buffer_size=buffer_size,
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        threshold_cfg=threshold_cfg,
        boundaries=boundaries,
        precomputed_thresholds=precomputed_thresholds,
        sampling_bundle=sampling_bundle,
        min_loss_tokens=min_loss_tokens,
        n_sae_features=n_sae_features,
        fp_percentile_table=fp_percentile_table,
        cluster_threshold_percentile_table=cluster_threshold_percentile_table,
        grad_accum_steps=grad_accum_steps,
        negative_bundle=negative_bundle,
        fp_target_layout=fp_target_layout,
        feature_mask=feature_mask,
        consolidation=consolidation,
        missing_percentile=missing_percentile,
        return_seg_doc_id=return_seg_doc_id,
        feature_source_bos_char_len=feature_source_bos_char_len,
        profile_markers=profile_markers,
        startup_trace=startup_trace,
        loss_mask_mode=loss_mask_mode,
        alignment_probe_samples=alignment_probe_samples,
        mixture_name=mixture_name,
        single_pass=single_pass,
        expected_split=expected_split,
        steered_positions=steered_positions,
    )


def main():
    """Full SFT training loop for encoder-decoder models.

    Uses Hydra compose API for torchrun compatibility. Loads a pretrained
    encoder-decoder checkpoint and finetunes on a Task dataset with
    BOS-bestfit packing and the configured SFT loss mask.
    """
    startup_trace = _SFTStartupTrace()
    with startup_trace.phase("hydra_config"):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        t._dynamo.config.cache_size_limit = 32

        # -------------------------------------------------------------------------
        # Hydra config loading (compose API for torchrun compatibility)
        _config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
        _config_name, _overrides = parse_config_name_and_overrides(
            sys.argv[1:], default_config_name="conditional_encdec_sft"
        )
        with initialize_config_dir(config_dir=_config_dir, version_base=None):
            cfg = compose(config_name=_config_name, overrides=_overrides)
        _reject_removed_sft_frequency_keys(cfg)
        _reject_unsupported_sft_pretrain_keys(cfg)
        if "lr_warmup_frac" in cfg:
            raise ValueError("lr_warmup_frac is unsupported; use integer lr_warmup_steps")
        lr_warmup_steps, num_iterations = _validate_sft_lr_schedule(
            warmup_steps=cfg.lr_warmup_steps, num_iterations=cfg.num_iterations
        )
        logger.info(f"SFT LR warmup: {lr_warmup_steps} optimizer updates (0 disables)")
        if num_iterations is None and lr_warmup_steps > 0:
            logger.info(
                "Epoch LR warmup overlap is checked during execution using synchronized "
                "committed progress; the optimizer-step horizon is unknown."
            )
        reject_feature_clustering_config(cfg, "conditional_encdec_sft")
        reject_removed_feature_prediction_keys(cfg)
        requested_fp_block = FeaturePredictionBlock(cfg.feature_prediction_block)
        requested_fp_depth = validate_feature_prediction_head_layers(
            cfg.feature_prediction_head_layers
        )
        try:
            dataset_mixture = SFTMixture(cfg.dataset)
        except ValueError:
            raise ValueError(
                f"Unknown dataset: {cfg.dataset}. Available: {sorted(m.value for m in SFTMixture)}"
            ) from None
        user_config_raw = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(user_config_raw, dict)
        user_config = _checkpoint_user_config_with_fp(
            cast(dict[str, Any], user_config_raw),
            None,
        )
        loss_mask_mode = parse_loss_mask_mode(cfg.sft_loss_mask)
        missing_percentile_policy = parse_missing_percentile_policy(
            cfg.missing_percentile_policy
        )
        logger.info(f"SFT loss mask mode: {loss_mask_mode.value}")

        feature_selection_raw = OmegaConf.to_container(cfg.feature_selection, resolve=True)
        assert isinstance(feature_selection_raw, dict)
        feature_selection = parse_feature_selection(cast(dict[str, Any], feature_selection_raw))
        _validate_feature_clusters_sft_selection(cfg, feature_selection)
        assert_no_gradient_ranking(
            feature_selection, "conditional_encdec_sft (SFT is live-selection only)"
        )
    # -------------------------------------------------------------------------
    # Compute init
    with startup_trace.phase("compute_init"):
        device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
        master_process = ddp_rank == 0
        if device_type == "cuda":
            admit_project_attention()

        autocast_ctx = (
            t.amp.autocast(device_type=device_type, dtype=t.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        )
        synchronize = t.cuda.synchronize if device_type == "cuda" else lambda: None
    startup_trace.set_ddp(rank=ddp_rank, world_size=ddp_world_size)

    # -------------------------------------------------------------------------
    # Load pretrained checkpoint and extract configs from metadata
    base_dir = get_base_dir()
    assert cfg.model_tag is not None, (
        "model_tag is required for SFT (set via CLI: model_tag=my_model)"
    )
    if getattr(cfg, "base_checkpoint_dir", None) is not None:
        checkpoint_dir = cfg.base_checkpoint_dir
        logger.info(f"Using explicit checkpoint dir: {checkpoint_dir}")
    else:
        checkpoint_dir = os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            cfg.model_tag,
            f"s{cfg.segment_size}",
            feature_selection_dirname(feature_selection),
        )
    # sft_checkpoint_dir is computed after n_sae_features is known (below,
    # after load_checkpoint). It may include an optional feature_mask leaf
    # between the pretrain dir and the SFT-phase leaf. Initialise to a
    # sentinel so references before that point cause a clear NameError.
    sft_checkpoint_dir: str  # deferred: assigned after load_checkpoint — n_sae_features (needed by resolve_feature_mask for the optional mask leaf) is not known until then

    # Resume flag: ``-1`` means fresh start (default, inherited from
    # ``base_train.yaml``); any positive integer means resume from that
    # step of an earlier SFT run.  ``0`` and negatives other than ``-1``
    # are rejected inside the helper.
    resume_step = resolve_resume_step(int(cfg.resume_from_step))
    resuming = resume_step is not None

    # Always load the pretrained base checkpoint first — it is the sole
    # source of truth for the model architecture (decoder GPTConfig,
    # encoder ratios, cross-attention frequency) regardless of whether
    # this is a fresh SFT run or a resume.
    with startup_trace.phase(
        "base_checkpoint_load",
        extra={"model_step": cfg.model_step},
    ):
        base_model_data, _, meta = load_checkpoint(
            checkpoint_dir, cfg.model_step, device, load_optimizer=False
        )

    # Extract decoder config from checkpoint metadata.
    # max_seq_len=null (default) → use checkpoint's sequence_len.
    # max_seq_len=<int> → override (e.g. for shorter sequences on smaller GPUs).
    dec_config_kwargs = meta["model_config"]
    checkpoint_seq_len = dec_config_kwargs["sequence_len"]
    if cfg.max_seq_len is not None:
        dec_config_kwargs["sequence_len"] = cfg.max_seq_len
        logger.info(
            f"Sequence length: {cfg.max_seq_len} (overridden from checkpoint's {checkpoint_seq_len})"
        )
    else:
        # Backfill cfg.max_seq_len so all downstream code (grad accum,
        # data generator T, eval tokens) uses the resolved value.
        OmegaConf.update(cfg, "max_seq_len", checkpoint_seq_len)
        logger.info(f"Sequence length: {checkpoint_seq_len} (from checkpoint)")
    dec_config = GPTConfig(**dec_config_kwargs)

    # Extract encoder/xattn config from checkpoint metadata.
    # Pretrain checkpoints store these under user_config.encoder / .cross_attention.
    # SFT checkpoints (used when chaining, e.g. safety → capabilities) store
    # them under top-level encoder_config / xattn_config keys instead, since
    # the SFT user_config doesn't include pretrain-specific fields.
    pretrain_cfg = meta["user_config"]
    if "encoder" in pretrain_cfg:
        enc_config = EncoderConfig(**pretrain_cfg["encoder"])
        xattn_config = CrossAttentionConfig(**pretrain_cfg["cross_attention"])
    elif "encoder_config" in meta:
        enc_config = EncoderConfig(**meta["encoder_config"])
        xattn_config = CrossAttentionConfig(**meta["xattn_config"])
    else:
        raise KeyError(
            "Checkpoint metadata missing encoder/cross_attention config. "
            "Expected either user_config.encoder (pretrain checkpoint) or "
            "encoder_config (SFT checkpoint). Is this an encoder-decoder checkpoint?"
        )

    # num_buckets: use SFT override if set, else checkpoint value.
    # Pretrain checkpoints have it under user_config; SFT checkpoints under top-level.
    _pretrain_buckets = pretrain_cfg.get("num_buckets")
    num_buckets_default = (
        _pretrain_buckets if _pretrain_buckets is not None else meta.get("num_buckets", 2)
    )
    num_buckets = cfg.num_buckets if cfg.num_buckets is not None else num_buckets_default

    # Recover n_sae_features + the compacted feature vocabulary from the base
    # checkpoint's state dict ALONE — the checkpoint is self-describing (issue #695).
    #
    # This MUST NOT go back to reading feature_embed.weight.shape[0]: once the
    # encoder's dictionary is compacted that row count is V, not n_sae_features, and
    # SFT would silently run with a redefined feature space — corrupting the frequency
    # blacklist, the resolved feature_mask, live feature selection, and every consumer
    # that bounds-checks ORIGINAL feature ids against
    # model.encoder.config.n_sae_features.
    #
    # The vocabulary is a property of the CHECKPOINT, not of the SFT config: SFT
    # inherits it and can neither create nor widen it (see the equality guard below).
    # A legacy checkpoint carries no vocab buffers and yields None ⇒ full-width
    # encoder, byte-identical to pre-#695.
    with startup_trace.phase("checkpoint_identity"):
        n_sae_features, _sft_feature_vocab = feature_vocab_from_state_dict(base_model_data)
        base_has_fp_weight = "feature_pred_head.weight" in base_model_data
        base_has_fp_bias = "feature_pred_head.bias" in base_model_data
        if base_has_fp_weight != base_has_fp_bias:
            raise FeatureVocabError(
                f"base checkpoint {checkpoint_dir} has a partial feature-prediction "
                "head state (weight/bias presence differs)"
            )

    logger.info(
        f"Loaded checkpoint metadata: dec={dec_config.n_layer}L/{dec_config.n_embd}d, "
        f"enc={enc_config.depth}/{enc_config.n_embd}, "
        f"xattn_freq={xattn_config.frequency}, "
        f"n_sae_features={n_sae_features}, num_buckets={num_buckets}"
    )
    # The compaction summary is logged once the model exists (below): its
    # feature-indexed parameter saving — the number that decides whether the run fits in
    # VRAM — needs the RESOLVED encoder config and the real value-embedding count, neither
    # of which the ratio-shaped enc_config carries.

    # Now that n_sae_features is known, resolve the optional feature_mask and
    # feature_clusters, then build the SFT checkpoint dir. The mask and cluster
    # leaves are inserted between the pretrain dir and the SFT-phase leaf:
    #   .../s<seg>/t<thr>/[bl_<hash>/][fc_<hash>/]<dataset>_sft_..._checkpoints/
    # checkpoint_dir (pretrain, no leaves) is kept as-is for loading above.
    # Only the auto-built path gets the leaves; the base_checkpoint_dir
    # override branch is left untouched (per task spec).
    selection_identity_extra: dict[str, object] = {
        "n_sae_features": n_sae_features,
    }
    with startup_trace.phase(
        "selection_identity",
        extra=lambda: selection_identity_extra,
    ):
        _sft_feature_mask = resolve_feature_mask(cfg, n_sae_features)
        resolved_clusters = resolve_feature_clusters(cfg, n_sae_features)
        consolidation = (
            FeatureClusterConsolidation.build(resolved_clusters, _sft_feature_mask)
            if resolved_clusters is not None
            else None
        )
        selection_identity_extra["mask_allowed_features"] = (
            None
            if _sft_feature_mask is None
            else (
                len(_sft_feature_mask.ids)
                if _sft_feature_mask.mode.value == "whitelist"
                else n_sae_features - len(_sft_feature_mask.ids)
            )
        )
        selection_identity_extra["clustered"] = resolved_clusters is not None
        if consolidation is not None:
            selection_identity_extra["cluster_representatives"] = int(
                consolidation.cluster_allowed_universe.sum().item()
            )

    if resolved_clusters is None:
        reject_feature_clusters_metadata(
            meta,
            f"conditional_encdec_sft base checkpoint {checkpoint_dir} "
            "(run has feature_clusters disabled)",
        )
    else:
        assert_resume_feature_clusters_match(
            resolved_clusters,
            meta,
            checkpoint_dir=checkpoint_dir,
        )

    # Fail at startup, not 40 minutes into the run. This run's effective mask/cluster
    # universe must EQUAL the base checkpoint's vocabulary. A pre-compaction masked
    # base is allowed only when its canonical mask metadata agrees; compacted
    # state-dict buffers remain authoritative.
    _assert_sft_universe_matches_checkpoint_vocab(
        consolidation if consolidation is not None else _sft_feature_mask,
        _sft_feature_mask,
        _sft_feature_vocab,
        n_sae_features,
        checkpoint_dir=checkpoint_dir,
        base_meta=meta,
    )

    # The conditioning contract the checkpoint's ENCODER was pretrained under.
    # ``num_buckets`` alone does not describe it: a checkpoint trained with
    # ``negative_selection`` reserved bucket ``num_buckets - 1`` for inactive
    # features and discretised positives into the remaining
    # ``effective_buckets = num_buckets - 1`` space.  Conditioned SFT reproduces
    # that ENCODING exactly and emits negatives into the reserved row (see the
    # feature-pipeline block below).  The DELTA axis is what it cannot
    # reproduce, and the policy for it is HARDCODED — not a config knob — so
    # every run behaves identically: splice_counterfactual warns, prev_segment
    # and cum_max raise.
    #
    # A chained base (an SFT checkpoint) carries its own SFT ``user_config``, so
    # the contract is read from the ``pretrain_conditioning`` block SFT
    # re-publishes; a pre-fix SFT checkpoint has none and is rejected rather
    # than silently read as unconditioned.
    #
    # Ordering: this sits AFTER the mask/cluster/vocabulary contract guards
    # above and BEFORE any W&B, profile, tokenizer or dataset work.  A base
    # checkpoint whose feature UNIVERSE disagrees with this run is a more
    # fundamental mismatch than how that universe was conditioned, so it must
    # be the error the operator sees; parsing an unusable checkpoint's
    # negative_selection first would shadow it.
    assert_pretrain_conditioning_recoverable(
        meta,
        checkpoint_label=f"base checkpoint {checkpoint_dir}",
    )
    pretrain_conditioning = read_checkpoint_conditioning(meta)
    assert_conditioning_supported(
        pretrain_conditioning,
        checkpoint_label=f"base checkpoint {checkpoint_dir}",
    )
    pretrain_negative_selection = parse_negative_selection(
        dict(pretrain_conditioning.negative_selection)
        if pretrain_conditioning.negative_selection is not None
        else None
    )

    fp_target_layout = (
        FeaturePredictionTargetLayout.build(
            n_sae_features=PositiveInt.parse(n_sae_features),
            # Uncompacted base => the literal full-width legacy layout. The
            # pre-compaction masked base (see
            # _assert_sft_universe_matches_checkpoint_vocab) keeps a compacting
            # mask against a full-width state dict; the mask still reaches the
            # generator and _zero_masked_fp_targets, but it does not describe
            # this checkpoint's row space.
            feature_mask=_sft_feature_mask if _sft_feature_vocab is not None else None,
            feature_clusters=consolidation if _sft_feature_vocab is not None else None,
            feature_vocab=_sft_feature_vocab,
        )
        if base_has_fp_weight
        else None
    )

    if getattr(cfg, "base_checkpoint_dir", None) is not None:
        # Explicit override — use the checkpoint path as-is, but the resolved
        # semantic mask still applies to data selection, FP loss, and metadata.
        sft_checkpoint_dir = _build_sft_checkpoint_dir(checkpoint_dir, cfg.dataset)
    else:
        # Auto-built path: append the mask leaf between the pretrain dir and
        # the cluster and SFT leaves.
        sft_checkpoint_dir = _build_sft_checkpoint_dir(
            append_feature_clusters_leaf(
                append_mask_leaf(checkpoint_dir, _sft_feature_mask),
                resolved_clusters,
            ),
            cfg.dataset,
        )

    # If resuming, load the previously-saved SFT checkpoint so we can
    # overwrite the base decoder weights with the SFT-finetuned weights
    # and restore optimizer + loop state.
    sft_model_data: dict | None = None
    sft_optimizer_data: dict | None = None
    sft_meta: dict | None = None
    if resuming:
        logger.info(f"Resuming SFT from step {resume_step} at {sft_checkpoint_dir}")
        with startup_trace.phase(
            "sft_resume_checkpoint_load",
            extra={"resume_step": resume_step},
        ):
            sft_model_data, sft_optimizer_data, sft_meta = load_checkpoint(
                sft_checkpoint_dir,
                resume_step,
                device,
                load_optimizer=True,
                rank=ddp_rank,
            )
        if resolved_clusters is None:
            reject_feature_clusters_metadata(
                sft_meta,
                f"conditional_encdec_sft resume checkpoint {sft_checkpoint_dir} "
                "(run has feature_clusters disabled)",
            )
        else:
            assert_resume_feature_clusters_match(
                resolved_clusters,
                sft_meta,
                checkpoint_dir=sft_checkpoint_dir,
            )
        assert_resume_feature_mask_matches(
            _sft_feature_mask,
            sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        assert_resume_sft_loss_mask_matches(
            loss_mask_mode,
            sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        assert_resume_missing_percentile_policy_matches(
            missing_percentile_policy,
            sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        # Defence in depth: same mask ⇒ same vocab, so the guard above already covers
        # the common case. This one also covers checkpoints written by paths that
        # record no feature_mask_canonical, and it compares n_sae_features — which the
        # bl_/wl_ leaf hash does NOT cover (it hashes only mode + ids).
        assert_resume_feature_vocab_matches(
            _sft_feature_vocab,
            sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        _assert_resume_feature_prediction_layout_matches(
            fp_target_layout,
            sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        _assert_resume_sft_lr_schedule_matches(
            sft_meta, warmup_steps=lr_warmup_steps, num_iterations=num_iterations
        )

    # Resolve and validate the saved architecture before external logging or
    # dataset construction. Fresh SFT may grow a linear head only after its
    # source weights have loaded strictly below; resume never grows a head.
    fp_meta = sft_meta if sft_meta is not None else meta
    fp_config = _resolve_fp_config_from_meta(fp_meta)
    validate_feature_prediction_architecture(
        fp_config, requested_fp_block, requested_fp_depth, resuming=resuming
    )

    # External logging, profiling outputs, tokenizer loading, and dataset
    # construction occur only after the config-derived resume-identity guards.
    # The segmentation-mode guard is the one exception: its configured value is
    # discovered from feature artifacts loaded later, but it still fails before
    # the first batch is constructed.
    with startup_trace.phase("wandb_setup"):
        wandb_run = create_wandb_run(cfg.run, master_process, user_config)

    profile_settings = _resolve_sft_profile_settings(cfg)
    profile_output_dir = profile_settings.output_dir
    profile_trace_step = profile_settings.trace_step
    profile_trace_steps = profile_settings.trace_steps
    profile_timing_sync_cuda = profile_settings.timing_sync_cuda
    timing_recorder = TimingRecorder() if profile_settings.timing_enabled else None
    chrome_profiler = None
    if profile_settings.timing_enabled or profile_trace_step is not None:
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"SFT profiling enabled: timing={profile_settings.timing_enabled}, "
            f"trace_step={profile_trace_step}, output_dir={profile_output_dir}"
        )

    @contextmanager
    def profile_region(name: str, **extra):
        """Emit a trace marker and optionally append a timing CSV row."""
        if timing_recorder is None:
            if chrome_profiler is None:
                yield
                return
            with timed_region(name, sync_cuda=False, extra=extra):
                yield
            return
        tags = {"rank": ddp_rank, **extra}
        with use_recorder(timing_recorder):
            with timing_tags(**tags):
                with timed_region(name, sync_cuda=profile_timing_sync_cuda):
                    yield

    def profile_markers_enabled() -> bool:
        return timing_recorder is not None or chrome_profiler is not None

    # Every accepted SFTMixture member defines a non-empty validation split;
    # the steered members borrow standard's (``sft_val_family``). Preflight its
    # artifacts before tokenizer and task construction; eval_every=0 remains
    # the explicit total opt-out.
    val_dataset = sft_val_family(dataset_mixture).value
    with startup_trace.phase(
        "validation_artifact_paths",
        extra={"enabled": bool(cfg.eval_every > 0)},
    ):
        resolved_sft_val_parquet_dir, resolved_val_feature_dir = (
            _resolve_and_require_sft_val_artifacts(
                has_val=True,
                eval_every=cfg.eval_every,
                val_parquet_override=cfg.val_sft_parquet_dir,
                val_feature_override=cfg.val_feature_dir,
                base_dir=base_dir,
                feature_dir=cfg.feature_dir,
                dataset=cfg.dataset,
                val_dataset=val_dataset,
            )
        )

    with startup_trace.phase("tokenizer_load"):
        tokenizer = get_tokenizer()
    # Deliberately defaults-only apart from steered_jsonl: the parquet
    # converter builds the SAME mixture with build_mixture defaults
    # (convert_sft_to_parquet.py), and conditioned feature lookup is keyed by
    # position in the resulting TaskMixture (original_index). Honouring
    # mixture-sizing overrides here would silently desynchronise every
    # extracted feature row from its conversation. steered_jsonl is the one
    # permitted override because a steered mixture cannot be built without it
    # and the converter reads the same key from its own config. Do NOT thread
    # any other cfg knob through this call.
    mixture_extra: dict[str, object] = {"dataset": str(dataset_mixture)}
    with startup_trace.phase(
        "mixture_build",
        extra=lambda: mixture_extra,
    ):
        task, val_task = build_task_mixtures(
            dataset_mixture,
            **steered_mixture_overrides(dataset_mixture, cfg.steered_jsonl),
        )
        mixture_extra["train_documents"] = len(task)
        mixture_extra["val_documents"] = 0 if val_task is None else len(val_task)
    steered_positions = _steered_positions_for_task(task)
    has_val = val_task is not None and len(val_task) > 0
    if val_task is not None:
        logger.info(f"Dataset: {cfg.dataset} ({len(task):,} train, {len(val_task):,} val examples)")
    else:
        logger.info(f"Dataset: {cfg.dataset} ({len(task):,} train, no val split)")

    # Build model from checkpoint configs, then load weights.  If
    # resuming, prefer the SFT-saved weights over the base ones so the
    # model picks up right where it left off.
    #
    # The FeaturePredictionConfig comes from the checkpoint metadata: a
    # pretrain checkpoint that trained with ``feature_prediction`` set
    # carries the knob in ``user_config.feature_prediction``; an SFT
    # checkpoint preserves that key and also carries a top-level
    # ``feature_prediction_config`` fallback for older chained-SFT
    # metadata. ``_resolve_fp_config_from_meta`` only answers whether the
    # checkpoint has an fp_head. Whether the auxiliary loss is active is
    # gated later by conditioned SFT plus a populated percentile table.
    #
    # When resuming a conditioned SFT, prefer the SFT-meta's fp config
    # over the pretrain meta's so the run picks up the exact same head
    # configuration it saved. When starting a fresh SFT the base
    # checkpoint meta is the only source of truth available.
    if fp_config is not None:
        # ``fp_config is not None`` ⇒ the checkpoint carries an
        # ``feature_pred_head`` and we must build the module for a
        # strict state_dict load. Whether the auxiliary loss is
        # *computed* during this run is a separate gate (conditioned
        # SFT + populated ``fp_percentile_table``, see below).
        logger.info(
            f"Feature-prediction loss active: weight={fp_config.weight}, "
            f"layer={fp_config.layer}, mode={fp_config.loss_mode.value}, "
            f"fn={fp_config.loss_fn.value}, block={fp_config.block.value}, head_layers={fp_config.head_layers}"
        )
    model_extra: dict[str, object] = {
        "n_sae_features": n_sae_features,
        "compact_vocab_rows": (
            None if _sft_feature_vocab is None else _sft_feature_vocab.size
        ),
    }
    with startup_trace.phase(
        "model_construct",
        extra=lambda: model_extra,
    ):
        with t.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                feature_pred_config=fp_config,
                feature_vocab=_sft_feature_vocab,
            )
    if fp_target_layout is not None and (
        model.feature_prediction_width != fp_target_layout.output_width
    ):
        raise FeatureVocabError(
            "feature-prediction head width does not match the SFT target layout: "
            f"head={model.feature_prediction_width}, "
            f"targets={fp_target_layout.output_width}"
        )
    # SFT INHERITS its vocabulary from the base checkpoint, so "no feature_mask is
    # configured" (the trainer's default reason) would be a lie here: an SFT run may well
    # configure a mask against a legacy checkpoint and still get a full-width encoder.
    logger.info(
        format_compaction_summary(
            model,
            vocab_source="inherited from the base checkpoint",
            no_vocab_reason="the base checkpoint is not compacted",
        )
    )
    with startup_trace.phase("model_materialize"):
        model.to_empty(device=device)
        model.init_buffers()
    # ``assert_vocab_consistent()`` after EVERY load: ``assign=True`` REPLACES the persistent
    # ``feature_vocab_ids`` tensor with the checkpoint's. If a checkpoint's vocabulary differs
    # from the configured one the shapes can still agree — and then every embedding row silently
    # means a different feature. (There is no separate derived map; ``_to_compact`` computes the
    # remap from ``feature_vocab_ids`` per-forward, so the ids buffer is the whole invariant.)
    #
    # ``assert_checkpoint_vocab_compatible()`` BEFORE every load, and it has to be before:
    # ``assert_vocab_consistent()`` below runs AFTER, but a compacted-vs-full-width mismatch
    # blows up INSIDE ``load_state_dict`` first, as PyTorch's bare "Unexpected key(s) ...
    # feature_vocab_ids / size mismatch for encoder.feature_embed.weight" — which names
    # neither compaction nor the recovery command. These two loads were the only production
    # encdec loads missing the guard that ``conditional_encdec_train.py`` already applies.
    if sft_model_data is not None:
        with startup_trace.phase("model_state_load", extra={"source": "sft_resume"}):
            assert_checkpoint_vocab_compatible(
                model.feature_vocab, sft_model_data, checkpoint_dir=sft_checkpoint_dir
            )
            model.load_state_dict(sft_model_data, strict=True, assign=True)
            model.assert_vocab_consistent()
        del sft_model_data
        del base_model_data
        logger.info(f"Loaded SFT checkpoint from {sft_checkpoint_dir} at step {resume_step}")
    else:
        # NO assert_checkpoint_vocab_compatible here, deliberately. It would be TAUTOLOGICAL:
        # ``model.feature_vocab`` IS this checkpoint's vocabulary by construction — it is
        # ``feature_vocab_from_state_dict(base_model_data)`` (above), handed straight to the
        # constructor — so the guard would re-run the same pure function on the same dict and
        # compare the result with itself. Every branch is unreachable, while it costs a full
        # re-parse of V ids (61,835 for the shipped band) on every SFT start. A dead guard on
        # the most dangerous load in the file is worse than none: it reads as protection.
        #
        # What genuinely needs verifying is (a) that the SFT's CONFIGURED mask agrees with the
        # checkpoint's vocabulary — ``_assert_sft_universe_matches_checkpoint_vocab``, above — and
        # (b) that the load landed correctly, which is ``assert_vocab_consistent()`` below.
        # The sibling call in the resume branch IS a real check: it compares against
        # ``sft_model_data``, a different checkpoint.
        with startup_trace.phase("model_state_load", extra={"source": "base"}):
            model.load_state_dict(base_model_data, strict=True, assign=True)
            model.assert_vocab_consistent()
        del base_model_data
        logger.info(f"Loaded pretrained checkpoint from {checkpoint_dir}")

    if not resuming and fp_config is not None and (
        fp_config.block, fp_config.head_layers
    ) != (requested_fp_block, requested_fp_depth):
        with startup_trace.phase("feature_prediction_head_upgrade"):
            if requested_fp_depth:
                model.feature_pred_head = ResidualFeaturePredictionHead(
                    cast(Linear, model.feature_pred_head),
                    requested_fp_depth,
                    requested_fp_block,
                    decoder_config=dec_config,
                )
            fp_config = replace(fp_config, block=requested_fp_block, head_layers=requested_fp_depth)
            model.feature_pred_config = fp_config
        logger.info(
            f"Upgraded feature-prediction head to {requested_fp_depth} {requested_fp_block.value} blocks"
        )

    orig_model = model  # keep uncompiled reference for saving
    # Eagerly build the decoder self-attention BlockMasks BEFORE compiling:
    # a cache miss inside the compiled forward rebuilds them in-graph with a
    # SymInt window capture, which the torch>=2.13 flex FLASH (CuteDSL)
    # backend rejects at lowering.  Train and validation loaders share the
    # same fixed (device_batch_size, sequence_len) decoder geometry, so this
    # covers every decoder flex call of the run.  See
    # EncoderDecoderGPT.prewarm_decoder_block_masks for the full mechanism.
    with startup_trace.phase("decoder_block_mask_prewarm"):
        model.prewarm_decoder_block_masks(
            cfg.device_batch_size, model.dec_config.sequence_len
        )
    # Resolve ``cfg.compile_static`` via the shared helper so the SFT and
    # pretrain scripts can never drift on the truth table.  In ``mode='sft'``,
    # ``"auto"`` resolves to ``dynamic=None`` (auto-dynamic): decoder rows are
    # a fixed ``(device_batch_size, max_seq_len)`` shape, and the packed
    # encoder / cross-attention tensors that vary per batch are already
    # ``mark_dynamic``'d, so only they compile symbolically.  Concrete decoder
    # dims are required by the flex-attention FLASH (CuteDSL) backend on
    # torch >= 2.13.  See ``steering/_common/compile_helpers.py``.
    compile_dynamic = resolve_compile_dynamic(
        cfg.compile_static,
        mode="sft",
    )
    _warn_if_forced_dynamic_breaks_fa4(compile_dynamic)
    with startup_trace.phase(
        "compile_setup",
        extra={"dynamic": compile_dynamic},
    ):
        model = cast(EncoderDecoderGPT, t.compile(model, dynamic=compile_dynamic))

    # --- Wire feature_mask into the model's FP-loss gate ---
    # Set on orig_model (unwrapped EncoderDecoderGPT) post-compile; model may be
    # a compiled OptimizedModule. The single resolved mask returns None when
    # no mask is configured — invariant I5 means None is a byte-identical no-op
    # in compute_feature_pred_loss.
    # Bound to _h10_sft_mask so the H10 fail-fast guard below can identity-check it.
    _h10_sft_mask = _sft_feature_mask
    orig_model.feature_mask = _h10_sft_mask

    # -------------------------------------------------------------------------
    # Optimizer setup with both global LR multipliers applied before the loop.
    # Bind persistent packing to the final raw model before the lazy compiled
    # wrapper executes its first forward.
    with startup_trace.phase("optimizer_and_runtime_setup"):
        optimizer = _setup_sft_optimizer(orig_model, cfg)
    # Restore optimizer moments on resume.  ``load_checkpoint`` returned
    # this rank's shard of the optimizer state (DistMuonAdamW uses ZeRO-2
    # sharding), so restoring is a per-rank operation.
    if sft_optimizer_data is not None:
        optimizer.load_state_dict(sft_optimizer_data)
        del sft_optimizer_data
        logger.info("Restored optimizer state from SFT checkpoint")

    # Resolve once at startup; the loop uses only this validated value.
    grad_clip_max_norm = resolve_grad_clip_max_norm(cfg.grad_clip_max_norm)
    min_loss_tokens: int = _configure_min_loss_tokens(cfg.min_loss_tokens)
    alignment_probe_samples: int = _configure_alignment_probe_samples(
        cfg.alignment_probe_samples
    )

    # -------------------------------------------------------------------------
    # Batch size calculations
    tokens_per_fwdbwd = cfg.device_batch_size * cfg.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert cfg.total_batch_size % world_tokens_per_fwdbwd == 0, (
        f"total_batch_size ({cfg.total_batch_size}) must be divisible by "
        f"world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd})"
    )
    grad_accum_steps = cfg.total_batch_size // world_tokens_per_fwdbwd
    logger.info(
        f"Tokens / micro-batch / rank: {cfg.device_batch_size} x {cfg.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    logger.info(
        f"Total batch size {cfg.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
    )

    # -------------------------------------------------------------------------
    # Feature conditioning setup.
    feature_boundaries: t.Tensor | None = None
    threshold_cfg: ThresholdConfig | None = None
    precomputed_thresholds: t.Tensor | None = None
    sft_sampling_bundle: SamplingBundle | None = None

    assert cfg.feature_dir is not None, "feature_dir is required for conditioned SFT"

    # Load TRAIN feature statistics and build bucket boundaries. Discretisation
    # is a property of the conditioning scheme trained under, not of the eval
    # split, so validation deliberately reuses these tables and never rebuilds
    # them from val features (issue #1278).
    stats_path = find_statistics_path(cfg.feature_dir)
    assert stats_path is not None, (
        f"feature_statistics not found in {cfg.feature_dir}. "
        f"Run feature extraction with statistics computation first."
    )
    with startup_trace.phase("statistics_load"):
        feature_stats = load_statistics(stats_path)
    per_feature_stats = feature_stats.per_feature_stats
    # The encoder was trained against the base checkpoint's SAE contract.
    # SFT's own Hydra defaults can describe a different probe and are not
    # authoritative lineage for a chained consumer; neither is a chained
    # base's own ``user_config.sae`` (that run's Hydra default), so a base that
    # publishes online-SAE lineage is read from that lineage.
    raw_sae_config = base_sae_config(meta, pretrain_cfg, sft_user_config=user_config)
    raw_feature_selection = user_config.get("feature_selection")
    if not isinstance(raw_sae_config, Mapping) or not isinstance(
        raw_feature_selection, Mapping
    ):
        raise ValueError(
            "conditioned SFT must retain mapping-valued sae and feature_selection "
            "config to publish online-SAE lineage"
        )
    # Conditioned SFT's discretisation MUST reproduce the bucket space the
    # checkpoint's encoder was PRETRAINED on, so it calls the same production
    # owner pretraining calls — ``build_feature_pipeline`` — with the
    # checkpoint's own ``negative_selection``.
    #
    # When pretraining ran with ``negative_selection``, positives were
    # discretised into ``effective_buckets = num_buckets - 1`` and bucket
    # ``num_buckets - 1`` was reserved as the "feature inactive" row (the
    # negative writer in ``steering/packed_batch_io.py`` sets
    # ``enc_mags = effective_buckets`` there).  Passing the raw ``num_buckets``
    # here instead — i.e. dropping ``negative_selection=`` — builds boundaries
    # over the full space and sends high-magnitude positives straight into that
    # inactive row: silent conditioning corruption, invisible in the loss.  That
    # is exactly the bug this call fixes, so the argument is load-bearing.
    #
    # ``build_feature_pipeline`` also owns the ``effective_buckets == 1``
    # binary-mode branch that yields a zero-column boundaries tensor (the
    # production d12/l22 ``num_buckets=2`` configuration).  Re-deriving any of
    # this locally is what let the two drift in the first place.
    #
    # ``feature_delta=None``: SFT has no reference segment to difference
    # against.  The delta gap is policed at startup by
    # ``assert_conditioning_supported`` (splice_counterfactual warns, the
    # cross-segment modes raise).
    #
    # Model construction below keeps the RAW ``num_buckets`` — the encoder's
    # magnitude embedding table must stay the checkpoint's width or the strict
    # state-dict load fails.  Only the DATA encoding is reduced.
    with startup_trace.phase("feature_pipeline_setup"):
        feature_pipeline = build_feature_pipeline(
            feature_selection=feature_selection,
            negative_selection=pretrain_negative_selection,
            per_feature_stats=per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            seed=42,
            ddp_rank=ddp_rank,
            feature_delta=None,
            feature_mask=_sft_feature_mask,
        )
    feature_boundaries = feature_pipeline.boundaries

    if isinstance(feature_pipeline.selection_config, ThresholdSelectionConfig):
        assert isinstance(feature_selection, ThresholdSelection)
        threshold_type = feature_selection.threshold_type
        threshold_cfg = ThresholdConfig(threshold_type, feature_selection.threshold)
        precomputed_thresholds = feature_pipeline.selection_config.precomputed_thresholds
        if precomputed_thresholds is not None:
            logger.info(f"Built {threshold_type.value} thresholds from cached statistics")

        logger.info(
            f"Feature conditioning enabled: {cfg.feature_dir}, "
            f"threshold={threshold_type.value}/{feature_selection.threshold}, "
            f"buckets={num_buckets} "
            f"(effective_buckets={feature_pipeline.effective_buckets})"
        )
    else:
        assert isinstance(feature_selection, SamplingSelection)
        sft_sampling_bundle = feature_pipeline.selection_config
        logger.info(
            f"Feature conditioning enabled (sampling): {cfg.feature_dir}, "
            f"distribution={feature_selection.distribution.value}, "
            f"mean={feature_selection.mean}, std={feature_selection.std}, "
            f"buckets={num_buckets} "
            f"(effective_buckets={feature_pipeline.effective_buckets})"
        )

    # Negative (inactive) features, when the checkpoint was pretrained with
    # them.  ``negative_sampling_mask`` is the shared resolver: unclustered it
    # is ``pipeline.feature_mask``; clustered it is the representative-only
    # ``effective_negative_mask()``, because the compacted encoder has no row
    # for a consolidated-away sibling.  ``replace`` keeps the SAME
    # ``t.Generator`` object, so the sampling/negative RNG stays shared exactly
    # as ``build_feature_pipeline`` set it up.
    sft_negative_bundle: NegativeBundle | None = None
    if feature_pipeline.negative_bundle is not None:
        sft_negative_bundle = replace(
            feature_pipeline.negative_bundle,
            feature_mask=negative_sampling_mask(feature_pipeline, consolidation),
        )
        logger.info(
            "Negative (inactive) feature emission enabled from the base "
            f"checkpoint's negative_selection={pretrain_negative_selection!r}: "
            f"every segment also carries sampled inactive features in bucket "
            f"{feature_pipeline.effective_buckets} "
            f"(negative pool mask={sft_negative_bundle.feature_mask!r})"
        )

    # -------------------------------------------------------------------------
    # Feature-prediction percentile table.
    # When fp_config is active, map each segment's raw SAE feature magnitudes
    # to percentile scores so the dataloader can scatter all-feature
    # percentiles into the dense ``fp_targets`` tensor. Reuse the sampling
    # bundle's table when present so the sampling-mode path and the fp-target
    # path share exactly one percentile statistics object. Otherwise build a
    # fresh table from the feature_dir's statistics.
    #
    # If the checkpoint has no feature-prediction head, leave
    # ``fp_percentile_table`` at ``None`` so the forward skips the auxiliary
    # loss branch.
    fp_percentile_table: t.Tensor | None = None
    if fp_config is not None:
        if sft_sampling_bundle is not None:
            fp_percentile_table = sft_sampling_bundle.percentile_table
        else:
            with startup_trace.phase(
                "percentile_table_setup",
                extra={"table": "fp_percentile_table"},
            ):
                fp_percentile_table = build_full_percentile_table(
                    per_feature_stats,
                    n_sae_features,
                )
        logger.info(
            f"Built fp_percentile_table: shape={tuple(fp_percentile_table.shape)}, "
            f"dtype={fp_percentile_table.dtype}"
        )

    # Cluster consolidation must compare siblings in one shared currency.
    # Threshold selection therefore uses the same full-distribution percentile
    # ranks as clustered sampling and the FP head. Reuse the FP table when it
    # exists; headless SFT still builds the table because conditioning itself
    # needs it. The old path interpolated over post-threshold bucket boundaries,
    # producing a second tail-relative rank and effectively applying the
    # threshold twice.
    cluster_threshold_percentile_table: t.Tensor | None = None
    if consolidation is not None and threshold_cfg is not None:
        # This run is clustered + threshold-selected, so the checkpoint being
        # resumed must have been stamped with the same selection-percentile
        # currency (full-distribution rank, PR #1321). Neither the cluster
        # nor the loss-mask/feature-vocab resume guards above observe this
        # axis: a pre-#1321 checkpoint would otherwise resume cleanly and
        # silently switch selection semantics mid-run.
        if resuming:
            assert_resume_clustered_threshold_semantics_match(
                sft_meta,
                surface=f"conditional_encdec_sft resume checkpoint {sft_checkpoint_dir}",
            )
        cluster_threshold_percentile_table = (
            fp_percentile_table
            if fp_percentile_table is not None
            else None
        )
        if cluster_threshold_percentile_table is None:
            with startup_trace.phase(
                "percentile_table_setup",
                extra={"table": "cluster_threshold_percentile_table"},
            ):
                cluster_threshold_percentile_table = build_full_percentile_table(
                    per_feature_stats,
                    n_sae_features,
                )
        logger.info(
            "Clustered threshold selection uses full-distribution percentiles: "
            f"shape={tuple(cluster_threshold_percentile_table.shape)}, "
            f"dtype={cluster_threshold_percentile_table.dtype}"
        )

    # -------------------------------------------------------------------------
    # Calculate num_iterations
    dataset_size = len(task)

    # ``sft_state`` is created before the data generator so the two can
    # share the committed cursor state. Recover the entire loop state
    # tuple up front; the rank-local dataloader state seeds ``SFTState``,
    # and the scalars (step, smooth loss, min val bpb, total time) are
    # unpacked into locals right before the training loop below.  On
    # a fresh run (``sft_meta is None``) the helper returns all-zero
    # defaults with ``min_val_bpb = +inf``.
    (
        resumed_step,
        resumed_smooth_train_loss,
        resumed_smooth_fp_loss,
        resumed_min_val_bpb,
        resumed_total_training_time,
        resumed_dataloader_envelope,
    ) = _restore_loop_state(sft_meta)
    try:
        resumed_dataloader_state = select_dataloader_rank_state(
            resumed_dataloader_envelope,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
    except ValueError as exc:
        raise DataLoadError(f"invalid conditioned SFT dataloader checkpoint: {exc}") from exc
    resumed_consumed = (
        0
        if resumed_dataloader_state is None
        else NonNegativeInt.parse(resumed_dataloader_state.get("consumed"))
    )
    sft_state = SFTState(
        consumed=resumed_consumed,
        dataloader_state=resumed_dataloader_state,
    )
    sft_prefetch_batches = _resolve_sft_prefetch_batches(cfg.sft_prefetch_batches)

    # -------------------------------------------------------------------------
    # Data generator.
    # The SFT conversation parquet directory is resolved once via
    # ``_resolve_sft_parquet_dir``. When ``cfg.sft_parquet_dir`` is
    # left at its default (``None``), the resolver uses
    # ``{base_dir}/sft_data/{dataset}``. Operators who point at a shared
    # mount, scratch dir, or alternate mixture set the config explicitly
    # and the override wins.
    resolved_sft_parquet_dir = _resolve_sft_parquet_dir(cfg.sft_parquet_dir, base_dir, cfg.dataset)

    # Segmentation is checkpoint identity even when no percentile table is in
    # use, so discover both stores unconditionally before the optional coverage
    # preflight. The strong references also keep the weak loader cache warm for
    # the generator and validation loader.
    val_store = None
    with rank_synchronized_stage("SFT feature segmentation", device=device):
        train_store = _load_sft_features_for_startup(
            cfg.feature_dir,
            resolved_sft_parquet_dir,
            startup_trace=startup_trace,
            source=SFTSplit.TRAIN.value,
        )
        if has_val and cfg.eval_every > 0:
            assert resolved_val_feature_dir is not None
            assert resolved_sft_val_parquet_dir is not None
            val_store = _load_sft_features_for_startup(
                resolved_val_feature_dir,
                resolved_sft_val_parquet_dir,
                startup_trace=startup_trace,
                source=SFTSplit.VAL.value,
            )
        # This resume-identity guard intentionally runs after the others: the
        # configured mode comes from the feature artifacts, not a config knob.
        sft_segment_mode = _finalize_sft_segment_mode(
            train_store.segment_mode,
            None if val_store is None else val_store.segment_mode,
            train_feature_dir=cfg.feature_dir,
            val_feature_dir=resolved_val_feature_dir,
            resuming=resuming,
            checkpoint_meta=sft_meta,
            checkpoint_dir=sft_checkpoint_dir,
        )
        _assert_steered_segment_mode(steered_positions, sft_segment_mode, str(cfg.dataset))
        bos_checks = [(SFTSplit.TRAIN, cfg.feature_dir, resolved_sft_parquet_dir)]
        if val_store is not None:
            assert resolved_val_feature_dir is not None
            assert resolved_sft_val_parquet_dir is not None
            bos_checks.append(
                (SFTSplit.VAL, resolved_val_feature_dir, resolved_sft_val_parquet_dir)
            )
        for split, split_feature_dir, split_parquet_dir in bos_checks:
            with _maybe_sft_startup_phase(
                startup_trace,
                "sft_source_bos_validation",
                extra={"source": split.value},
            ):
                assert_sft_source_bos_char_len_matches(
                    split_feature_dir,
                    split_parquet_dir,
                    configured=int(cfg.feature_source_bos_char_len),
                    source=split.value,
                )
    online_sae_lineage_metadata = build_online_sae_lineage_metadata(
        feature_dir=str(cfg.feature_dir),
        statistics_path=stats_path,
        segment_mode=sft_segment_mode,
        sae_config=raw_sae_config,
        feature_selection=raw_feature_selection,
        negative_selection=(
            None
            if pretrain_conditioning.negative_selection is None
            else dict(pretrain_conditioning.negative_selection)
        ),
        missing_percentile_policy=missing_percentile_policy.value,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        selection_seed=SFT_FEATURE_PIPELINE_SEED,
    )

    # -------------------------------------------------------------------------
    # Percentile-coverage preflight (issue #1311 / PR #1316).
    #
    # ``magnitude_to_percentile`` still refuses an all-zero row — that primitive
    # is untouched and remains the backstop. What it cannot do is tell the
    # operator how big the hole is: it fires on whichever missing id happens to
    # be looked up first, minutes into setup. This stage builds the coverage
    # bitmap up front and reports the COMPLETE distinct uncovered population per
    # table, then applies the configured policy.
    #
    # Each table is validated against the set it is actually queried with, and
    # those sets are NOT the same:
    #   * ``fp_percentile_table`` is queried with the RAW pre-selection ids
    #     (``full_fids = seg.feature_ids``), before feature-mask filtering and
    #     before cluster consolidation;
    #   * ``sft_sampling_bundle.percentile_table`` and
    #     ``cluster_threshold_percentile_table`` are queried with the ids that
    #     SURVIVE the feature mask, still pre-consolidation (the clustered path
    #     consolidates only after the lookup, and passes the consolidated
    #     percentiles to ``sample_features`` as override weights).
    # They coincide only when ``fp_percentile_table is
    # sft_sampling_bundle.percentile_table``, which is why they are checked
    # independently rather than through a shared approximation.
    #
    # Store loads + document coverage are aligned SEPARATELY from the scan
    # stage below: the scan stage opens with a collective (the rank-0 scan
    # broadcast), so a rank-local artifact failure must be resolved before any
    # rank enters it — otherwise the failed rank's stage-end all-reduce would
    # pair with its peers' scan all-reduce and the survivors would block in a
    # later collective (issue #1495).
    coverage_sources: list[tuple[str, Mapping[int, Any], Sequence[int]]] = []
    with rank_synchronized_stage("percentile coverage sources", device=device):
        if (
            fp_percentile_table is not None
            or sft_sampling_bundle is not None
            or cluster_threshold_percentile_table is not None
        ):
            with _maybe_sft_startup_phase(
                startup_trace,
                "sft_percentile_document_coverage",
                extra={"source": SFTSplit.TRAIN.value},
            ):
                _train_coverage, train_indices = _resolve_sft_dataset_indices(
                    train_store,
                    task_size=len(task),
                    feature_dir=cfg.feature_dir,
                    parquet_dir=resolved_sft_parquet_dir,
                )
            coverage_sources.append(("train", train_store, train_indices))
            if has_val and cfg.eval_every > 0:
                # Validation conditions on train statistics by design (#1278),
                # so the VAL mixture's active ids must be covered by the same
                # tables. A hole here is exactly the capped-statistics failure.
                assert val_task is not None  # has_val implies a non-empty val task
                assert resolved_val_feature_dir is not None
                assert resolved_sft_val_parquet_dir is not None
                assert val_store is not None
                with _maybe_sft_startup_phase(
                    startup_trace,
                    "sft_percentile_document_coverage",
                    extra={"source": SFTSplit.VAL.value},
                ):
                    _val_coverage, val_indices = _resolve_sft_dataset_indices(
                        val_store,
                        task_size=len(val_task),
                        feature_dir=resolved_val_feature_dir,
                        parquet_dir=resolved_sft_val_parquet_dir,
                    )
                coverage_sources.append(("val", val_store, val_indices))
    # Do not ``del`` coverage_sources: it keeps the weak-value store cache warm.
    with rank_synchronized_stage("percentile coverage", device=device):
        missing_percentile_resolver = _percentile_coverage_preflight(
            coverage_sources=coverage_sources,
            n_sae_features=n_sae_features,
            feature_mask=_sft_feature_mask,
            fp_percentile_table=fp_percentile_table,
            sft_sampling_bundle=sft_sampling_bundle,
            cluster_threshold_percentile_table=cluster_threshold_percentile_table,
            missing_percentile_policy=missing_percentile_policy,
            device=device,
            startup_trace=startup_trace,
        )

    producer_state = (
        SFTState(
            consumed=resumed_consumed,
            dataloader_state=copy.deepcopy(resumed_dataloader_state),
        )
        if sft_prefetch_batches > 0
        else sft_state
    )
    raw_train_gen = make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=cfg.device_batch_size,
        T=cfg.max_seq_len,
        device=device,
        sft_state=producer_state,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        grad_accum_steps=grad_accum_steps,
        feature_dir=cfg.feature_dir,
        parquet_dir=resolved_sft_parquet_dir,
        threshold_cfg=threshold_cfg,
        boundaries=feature_boundaries,
        precomputed_thresholds=precomputed_thresholds,
        sampling_bundle=sft_sampling_bundle,
        loss_mask_mode=loss_mask_mode,
        min_loss_tokens=min_loss_tokens,
        fp_percentile_table=fp_percentile_table,
        cluster_threshold_percentile_table=cluster_threshold_percentile_table,
        negative_bundle=sft_negative_bundle,
        fp_target_layout=fp_target_layout,
        feature_mask=_h10_sft_mask,
        consolidation=consolidation,
        missing_percentile=missing_percentile_resolver,
        return_seg_doc_id=fp_percentile_table is not None,
        feature_source_bos_char_len=cfg.feature_source_bos_char_len,
        profile_markers=profile_markers_enabled,
        startup_trace=startup_trace,
        alignment_probe_samples=alignment_probe_samples,
        mixture_name=str(cfg.dataset),
        expected_split=SFTSplit.TRAIN,
        steered_positions=steered_positions,
    )
    if sft_prefetch_batches > 0:
        logger.info(
            "SFT batch prefetch enabled: "
            f"depth={sft_prefetch_batches}, optimizer-step-bounded producer"
        )

    # Validation data generator (lazy — created fresh each eval) consumes the
    # validation mixture's independently converted parquet and feature shards.

    def build_val_loader():
        return _build_sft_val_loader(
            val_task=val_task,
            tokenizer=tokenizer,
            B=cfg.device_batch_size,
            T=cfg.max_seq_len,
            device=device,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            feature_dir=resolved_val_feature_dir,
            expected_feature_dir=resolved_val_feature_dir,
            parquet_dir=resolved_sft_val_parquet_dir,
            threshold_cfg=threshold_cfg,
            boundaries=feature_boundaries,
            precomputed_thresholds=precomputed_thresholds,
            sampling_bundle=sft_sampling_bundle,
            loss_mask_mode=loss_mask_mode,
            min_loss_tokens=min_loss_tokens,
            fp_percentile_table=fp_percentile_table,
            cluster_threshold_percentile_table=cluster_threshold_percentile_table,
            negative_bundle=sft_negative_bundle,
            fp_target_layout=fp_target_layout,
            feature_mask=_h10_sft_mask,
            consolidation=consolidation,
            missing_percentile=missing_percentile_resolver,
            feature_source_bos_char_len=cfg.feature_source_bos_char_len,
            profile_markers=profile_markers_enabled,
            startup_trace=startup_trace,
            alignment_probe_samples=alignment_probe_samples,
            mixture_name=val_dataset,
        )

    token_bytes = get_token_bytes(device=device) if has_val and cfg.eval_every > 0 else None

    # Muon momentum warmup
    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    # -------------------------------------------------------------------------
    # Training loop state
    # The initial batch is allowed to publish artifact coverage before loop
    # entry, but its produced cursor is not committed until a successful
    # optimizer step trains it.
    step = resumed_step
    committed_consumed = resumed_consumed
    committed_dataloader_state = resumed_dataloader_state
    smooth_train_loss = resumed_smooth_train_loss
    smooth_fp_loss = resumed_smooth_fp_loss
    min_val_bpb = resumed_min_val_bpb
    total_training_time = resumed_total_training_time
    first_step_seen = False
    first_step_time_to_completion_ns: int | None = None
    post_first_step_total_ns = 0
    post_first_step_steps = 0
    if sft_meta is not None:
        logger.info(
            f"Resumed at step {step}, consumed={sft_state.consumed}, "
            f"smooth_loss={smooth_train_loss:.4f}, "
            f"min_val_bpb={min_val_bpb:.4f}"
        )
    with profile_region("sft/initial_data_loading", step=step):
        with rank_synchronized_stage(
            "conditioned SFT training batch construction",
            device=device,
        ):
            pending_batch = next(raw_train_gen)
            if producer_state is not sft_state:
                _SFTStateSnapshot.capture(producer_state).publish(sft_state)
    if sft_meta is not None:
        _assert_sft_resume_artifact_coverage(sft_meta, sft_state.artifact_coverage)
    if sft_state.dataset_size is None or sft_state.artifact_coverage is None:
        raise RuntimeError(
            "conditioned SFT generator did not publish artifact coverage before "
            "the first training batch"
        )
    dataset_size = sft_state.dataset_size
    if num_iterations is not None:
        logger.info(
            f"Training for {num_iterations:,} iterations "
            f"({dataset_size:,} indexed examples)"
        )
    else:
        logger.info(
            f"Training for 1 epoch ({dataset_size:,} indexed examples, "
            "consumption-based stopping)"
        )

    # H10 fail-fast: the feature_mask MUST be wired onto the model before the
    # first (compile-traced) forward. If a refactor moves the set below the
    # training loop, torch.compile would bake feature_mask=None into the graph
    # and silently disable masking. This guard makes that regression loud.
    # No-mask path (I5): _h10_sft_mask is None, orig_model.feature_mask is None
    # → None is None → passes. Zero hot-path cost; runs once before the loop.
    if orig_model.feature_mask is not _h10_sft_mask:
        raise RuntimeError(
            "feature_mask not wired onto orig_model before the SFT training loop "
            f"(orig_model.feature_mask={orig_model.feature_mask!r}, "
            f"expected _h10_sft_mask={_h10_sft_mask!r}). "
            "It must be set before the first torch.compile-traced forward."
        )

    while True:
        position = resolve_sft_loop_position(
            completed_steps=step,
            num_iterations=-1 if num_iterations is None else num_iterations,
            committed_consumed=committed_consumed,
            dataset_size=dataset_size,
        )
        last_step, progress = sync_stop_and_progress(
            last_step=position.last_step,
            progress=position.progress,
            device=device,
        )
        # Check even the terminal boundary, before evaluation or publication.
        # Every rank uses the same committed progress from the collective above.
        if (
            num_iterations is None
            and lr_warmup_steps > 0
            and step <= lr_warmup_steps
            and progress >= 0.8
        ):
            raise ValueError(
                "SFT epoch LR warmup must finish strictly before 80% decay; "
                f"completed_steps={step}, lr_warmup_steps={lr_warmup_steps}, progress={progress}. "
                "Reduce lr_warmup_steps (0 disables warmup)."
            )

        # Validation evaluation (skipped when no val split available).
        if _should_run_val_eval(
            has_val=has_val,
            eval_every=cfg.eval_every,
            step=step,
            last_step=last_step,
        ):
            model.eval()
            eval_steps = cfg.eval_tokens // (
                cfg.device_batch_size * cfg.max_seq_len * ddp_world_size
            )
            # ``token_bytes`` is built under exactly this predicate above.
            assert token_bytes is not None
            val_loader = build_val_loader()
            with autocast_ctx:
                with profile_region("sft/validation_bpb", step=step):
                    val_bpb = evaluate_encdec_bpb(model, val_loader, eval_steps, token_bytes)
            log_payload: dict[str, float | int] = {
                "step": step,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
            logger.info(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log(log_payload)
            model.train()

        # Checkpoint save — every rank participates so optimizer state is
        # fully persisted.  ``save_checkpoint`` writes the model and meta
        # only from rank 0 internally, while optimizer state is sharded
        # per rank (matches pretraining's ``conditional_encdec_train.py``
        # save path). ``loop_state`` carries the scalar training state, exact
        # per-rank committed dataloader state, and the artifact coverage that
        # cursor belongs to.
        if last_step or (cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0):
            with profile_region("sft/checkpoint_save", step=step):
                if committed_dataloader_state is None:
                    raise RuntimeError(
                        "conditioned SFT cannot checkpoint before a training batch "
                        "has been committed"
                    )
                sft_dataloader_envelope = build_dataloader_state_envelope(
                    committed_dataloader_state,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                fp_config_dict = _serialize_fp_config(fp_config)
                checkpoint_user_config = _checkpoint_user_config_with_fp(
                    user_config,
                    fp_config,
                )
                # B6 canonical persistence: persist the resolved canonical form
                # (mode + sorted ids, NO path reference) so cross-host resumes can
                # verify mask identity without re-resolving the original path.
                # No-mask path (I2/I5): key is absent → meta byte-identical to pre-B6.
                #
                # A steered run needs NO extra checkpoint key: ``user_config``
                # already echoes ``dataset`` and ``steered_jsonl``, which is
                # what defines the interleaving, and ``sft_artifact_coverage``
                # already pins that interleaving by fingerprint on resume.
                _sft_save_meta: dict = {
                    "step": step,
                    "dataset": cfg.dataset,
                    "sft_loss_mask": loss_mask_mode.value,
                    SFT_SEGMENT_MODE_META_KEY: sft_segment_mode.value,
                    # A `lowest` pilot's val/bpb must never be silently
                    # comparable to a production number, so the resolved policy
                    # travels with the checkpoint and a resume under a different
                    # one is rejected.
                    MISSING_PERCENTILE_POLICY_META_KEY: missing_percentile_policy.value,
                    "user_config": checkpoint_user_config,
                    "model_config": asdict(orig_model.dec_config),
                    # Encoder/xattn configs for checkpoint chaining — allows
                    # a subsequent SFT run (e.g. capabilities from safety) to
                    # reconstruct the model without the pretrain checkpoint.
                    "encoder_config": asdict(enc_config),
                    "xattn_config": asdict(xattn_config),
                    "num_buckets": num_buckets,
                    # Persist the fp config so a chained SFT run
                    # (``resume_from_step`` or a fresh SFT starting from
                    # this checkpoint) can reconstruct the fp_head
                    # without ancestor metadata. Both mirrors contain the
                    # actual post-upgrade head configuration.
                    # ``_resolve_fp_config_from_meta`` gives the current
                    # checkpoint's ``user_config.feature_prediction`` mirror
                    # precedence over this fallback.
                    "feature_prediction_config": fp_config_dict,
                    "min_val_bpb": min_val_bpb,
                    "loop_state": {
                        "sft_cursor_semantics": SFT_CURSOR_SEMANTICS_BUFFERED_V2,
                        "smooth_train_loss": smooth_train_loss,
                        "smooth_fp_loss": smooth_fp_loss,
                        "min_val_bpb": min_val_bpb,
                        "total_training_time": total_training_time,
                        "sft_consumed": committed_consumed,
                        "sft_dataloader_state": sft_dataloader_envelope,
                        "sft_artifact_coverage": sft_state.artifact_coverage,
                    },
                }
                # Re-publish the PRETRAINING conditioning contract this run
                # inherited. The SFT ``user_config`` describes the SFT run, not
                # the pretraining that shaped the encoder, so without this a
                # chained SFT (safety -> capabilities) would read its base as
                # unconditioned and silently rebuild the wrong bucket space.
                # Empty (hence meta byte-identical to pre-change) for a base
                # with no negative_selection and no delta ranking.
                _sft_save_meta.update(pretrain_conditioning.metadata_dict())
                _sft_save_meta.update(online_sae_lineage_metadata)
                if _sft_feature_mask is not None:
                    _sft_save_meta["feature_mask_canonical"] = canonical_dict(_sft_feature_mask)
                _sft_save_meta.update(feature_clusters_checkpoint_metadata(resolved_clusters))
                # Clustered-threshold runs stamp the selection-percentile
                # currency (PR #1321 full-distribution rank) so a resume
                # guard can reject a pre-#1321 checkpoint that computed
                # selection_percentiles from post-threshold, boundary-
                # relative bucket ranks instead. Sampling and unclustered
                # runs never stamp this key.
                if consolidation is not None and threshold_cfg is not None:
                    _sft_save_meta[CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY] = (
                        ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK.value
                    )
                # The compacted encoder's dictionary (#695). A fingerprint, not the id
                # list — the load-bearing record is the persistent buffer in the
                # state dict; this is the mirror tooling and the resume guard read.
                # Absent when not compacted ⇒ the meta is byte-identical to pre-#695.
                if _sft_feature_vocab is not None:
                    _sft_save_meta["feature_vocab"] = _sft_feature_vocab.metadata_dict()
                _sft_save_meta.update(
                    _feature_prediction_layout_checkpoint_metadata(fp_target_layout)
                )
                _save_sft_checkpoint(
                    dry_run=bool(cfg.dry_run),
                    checkpoint_dir=sft_checkpoint_dir,
                    step=step,
                    model=orig_model,
                    optimizer=optimizer,
                    meta=_sft_save_meta,
                    rank=ddp_rank,
                )

        if last_step:
            break

        # Training step
        monotonic_step_started_ns = time.perf_counter_ns()
        synchronize()
        use_trace = (
            profile_trace_step is not None
            and profile_trace_step <= step < profile_trace_step + profile_trace_steps
        )
        if use_trace and step == profile_trace_step:
            activities = [t.profiler.ProfilerActivity.CPU]
            if device_type == "cuda" and t.cuda.is_available():
                activities.append(t.profiler.ProfilerActivity.CUDA)
            chrome_profiler = t.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                experimental_config=t.profiler._ExperimentalConfig(profile_all_threads=True),
            )
            chrome_profiler.__enter__()
        t0 = time.time()
        train_fp_loss_micro: t.Tensor | None = None
        prefetched_batch_count = grad_accum_steps - int(pending_batch is not None)
        first_prefetched_micro_step = int(pending_batch is not None)
        with _sft_step_batch_iterator(
            raw_train_gen,
            producer_state=producer_state,
            consumer_state=sft_state,
            batch_count=prefetched_batch_count,
            prefetch_batches=sft_prefetch_batches,
            cuda_device=device if device_type == "cuda" else None,
            timing_recorder=timing_recorder,
            timing_rank=ddp_rank,
            timing_step=step,
            first_micro_step=first_prefetched_micro_step,
        ) as step_train_gen:
            for micro_step in range(grad_accum_steps):
                if pending_batch is None:
                    with profile_region("sft/data_loading", step=step, micro_step=micro_step):
                        with rank_synchronized_stage(
                            "conditioned SFT training batch construction",
                            device=device,
                        ):
                            batch = next(step_train_gen)
                else:
                    batch = pending_batch
                    pending_batch = None
                if len(batch) == 8:
                    (
                        inputs,
                        targets,
                        enc_fids,
                        enc_mags,
                        enc_mask,
                        seg_map,
                        fp_targets,
                        seg_doc_id,
                    ) = batch
                else:
                    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets = batch
                    seg_doc_id = None
                with autocast_ctx:
                    with profile_region("sft/forward", step=step, micro_step=micro_step):
                        result = model(
                            inputs,
                            targets,
                            enc_fids,
                            enc_mags,
                            enc_mask,
                            seg_map,
                            fp_targets=fp_targets,
                            seg_doc_id=seg_doc_id,
                        )
                if isinstance(result, ForwardResult):
                    loss = result.loss
                    train_loss = result.lm_loss.detach()
                    step_fp_loss = result.feature_pred_loss.detach()
                    # Accumulate fp loss across micro-steps so the eventual
                    # smoothed value reflects the whole optimizer step,
                    # mirroring the way ``loss`` aggregates LM loss across
                    # accumulation steps before the optimizer step.
                    train_fp_loss_micro = (
                        step_fp_loss
                        if train_fp_loss_micro is None
                        else train_fp_loss_micro + step_fp_loss
                    )
                else:
                    loss = result
                    train_loss = loss.detach()
                loss = loss / grad_accum_steps
                with profile_region("sft/backward", step=step, micro_step=micro_step):
                    loss.backward()

        # Gradient norm logging per module
        if step % 100 == 0:
            with profile_region("sft/grad_norm_logging", step=step):

                def grad_norm(params):
                    grads = [p.grad for p in params if p.grad is not None]
                    if not grads:
                        return 0.0
                    return t.cat([g.flatten() for g in grads]).norm().item()

                dec_gnorm = grad_norm(model.transformer.parameters())
                enc_gnorm = grad_norm(model.encoder.parameters())
                xattn_gnorm = grad_norm(model.cross_attn_layers.parameters())
            logger.info(
                f"Gradient norms — dec: {dec_gnorm:.4f}, "
                f"enc: {enc_gnorm:.4f}, xattn: {xattn_gnorm:.4f}"
            )

        # Update LR and momentum
        lrm = sft_warmup_flat_decay(
            progress, completed_steps=step, warmup_steps=lr_warmup_steps
        )
        muon_momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
        total_norm: t.Tensor | None = None
        if grad_clip_max_norm is not None:
            with profile_region("sft/grad_clip", step=step):
                with rank_synchronized_stage("conditioned SFT gradient clipping", device=device):
                    total_norm = apply_gradient_clipping(
                        orig_model.parameters(),
                        grad_clip_max_norm,
                    )
        with profile_region("sft/optimizer_step", step=step):
            optimizer.step()
            model.zero_grad(set_to_none=True)
        synchronize()
        monotonic_step_elapsed_ns = time.perf_counter_ns() - monotonic_step_started_ns
        if not first_step_seen:
            first_step_seen = True
            first_step_completed_ns = time.perf_counter_ns()
            first_step_time_to_completion_ns = first_step_completed_ns - startup_trace.start_ns
            startup_trace.record_first_step(
                completed_ns=first_step_completed_ns,
                step_elapsed_ns=monotonic_step_elapsed_ns,
                step=step + 1,
            )
        else:
            post_first_step_total_ns += monotonic_step_elapsed_ns
            post_first_step_steps += 1
        if (
            chrome_profiler is not None
            and use_trace
            and step == profile_trace_step + profile_trace_steps - 1
        ):
            chrome_profiler.__exit__(None, None, None)
            if master_process:
                trace_path = profile_output_dir / f"sft_trace_step{profile_trace_step}.json"
                chrome_profiler.export_chrome_trace(str(trace_path))
                logger.info(f"Exported SFT Chrome trace to {trace_path}")
            chrome_profiler = None
        t1 = time.time()
        dt = t1 - t0

        # Logging
        step += 1
        committed_consumed = sft_state.consumed
        if sft_state.dataloader_state is None:
            raise RuntimeError(
                "conditioned SFT generator did not publish dataloader state "
                "for the completed optimizer step"
            )
        committed_dataloader_state = dict(sft_state.dataloader_state)
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        debiased = smooth_train_loss / (1 - ema_beta**step)
        debiased_fp: float | None = None
        if train_fp_loss_micro is not None:
            avg_fp = train_fp_loss_micro.item() / max(1, grad_accum_steps)
            smooth_fp_loss = ema_beta * smooth_fp_loss + (1 - ema_beta) * avg_fp
            debiased_fp = smooth_fp_loss / (1 - ema_beta**step)
        if step > 10:
            total_training_time += dt
        pct = 100 * progress
        iter_str = f"{num_iterations}" if num_iterations is not None else "epoch"
        log_msg = (
            f"step {step:05d}/{iter_str} ({pct:.1f}%) | "
            f"loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms"
        )
        if debiased_fp is not None:
            log_msg += f" | fp_loss: {debiased_fp:.6f}"
        logger.info(log_msg)
        if step % 10 == 0:
            wandb_payload = {
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased,
                "train/lrm": lrm,
                "train/dt": dt,
            }
            if debiased_fp is not None:
                wandb_payload["train/feature_pred_loss"] = debiased_fp
            if total_norm is not None:
                wandb_payload["train/pre_clip_grad_norm"] = float(total_norm.item())
            wandb_run.log(wandb_payload)

    if first_step_time_to_completion_ns is None:
        startup_trace.log_no_step_summary()
    else:
        startup_trace.log_steady_state(
            time_to_first_step_ns=first_step_time_to_completion_ns,
            post_first_step_total_ns=post_first_step_total_ns,
            post_first_step_steps=post_first_step_steps,
        )
    logger.info(
        f"Training complete. {committed_consumed:,} / {dataset_size:,} conversations "
        f"consumed in {step} steps. Total time: {total_training_time / 60:.2f}m"
    )
    if chrome_profiler is not None:
        chrome_profiler.__exit__(None, None, None)
        if master_process:
            trace_path = profile_output_dir / f"sft_trace_step{profile_trace_step}_partial.json"
            chrome_profiler.export_chrome_trace(str(trace_path))
            logger.info(f"Exported partial SFT Chrome trace to {trace_path}")
    if timing_recorder is not None:
        timing_path = profile_output_dir / f"sft_timing_rank{ddp_rank}.csv"
        timing_recorder.to_csv(timing_path)
        logger.info(f"Exported SFT timing CSV to {timing_path}")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
