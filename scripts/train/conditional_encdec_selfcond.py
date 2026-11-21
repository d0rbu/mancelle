"""On-policy self-conditioning post-training for encoder-decoder checkpoints.

Run single-process::

    python -m scripts.train.conditional_encdec_selfcond model_tag=<tag>

Run distributed::

    torchrun --nproc_per_node=8 -m scripts.train.conditional_encdec_selfcond -- model_tag=<tag>
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import torch as t
import torch.distributed as dist
import wandb
from beartype import beartype
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

import nanochat.flash_attention as fa_module
from errors import (
    FeatureClustersError,
    FeatureMaskError,
    FeatureVocabError,
    SelfCondTrainingError,
)
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
)
from nanochat.flash_attention import FlashAttnImpl
from nanochat.fp8_utils import disable_fp8
from nanochat.gpt import GPTConfig, OptimizerKind
from nanochat.hydra_cli import parse_config_name_and_overrides
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.train.conditional_encdec_sft import (
    _FEATURE_PREDICTION_WIDTH_META_KEY,
    SFT_CURSOR_SEMANTICS_BUFFERED_V2,
    SFT_FEATURE_PIPELINE_SEED,
    SFTState,
    _build_sft_val_loader,
    _load_sft_features_for_startup,
    _prepare_sft_render_messages,
    _resolve_and_require_sft_val_artifacts,
    _resolve_sft_parquet_dir,
    evaluate_encdec_bpb,
    make_sft_data_generator,
    sync_last_step,
)
from scripts.train.conditional_encdec_sft import (
    _SFTStartupTrace as _SFTStartupTraceBase,
)
from scripts.train.conditional_encdec_train import (
    discover_packed_selection,
    discover_val_packed_row_dir,
)
from steering._common.ddp import rank_synchronized_stage
from steering._common.resume import resolve_resume_step
from steering._common.strict_json import canonical_json
from steering.conditional.checkpoint_conditioning import (
    PRETRAIN_CONDITIONING_META_KEY,
    read_checkpoint_conditioning,
)
from steering.conditional.delta import parse_feature_delta
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.conditioning_cooldown import parse_conditioning_cooldown
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredLossMode,
    reject_removed_feature_prediction_keys,
    validate_feature_prediction_resume,
)
from steering.conditional.encoder_decoder.config import (
    feature_prediction_config_from_metadata as _resolve_fp_config_from_meta,
)
from steering.conditional.encoder_decoder.config import (
    serialize_feature_prediction_config as _serialize_fp_config,
)
from steering.conditional.encoder_decoder.dataloader import (
    EncDecTrainBatch,
    _prefetch_wrapper,
    build_dataloader_state_envelope,
    precomputed_data_loader,
    resolve_train_prefetch_batches,
    select_dataloader_rank_state,
)
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.loss_eval import (
    accumulate_bpb_nats_bytes,
    evaluate_encdec_bpb_conditioned,
    reduce_bpb,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    VAL_DATA_SEED_SENTINEL,
    VAL_ROLLOUT_STEP_SENTINEL,
    ConditioningBatch,
    SelfCondBatch,
    SelfCondDataSource,
    SelfCondStepMetrics,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    freeze_for_head_only,
    materialize_head_zero_grads,
    resolve_recorded_num_buckets,
    resolve_selfcond_count_configs,
    rollout,
    selfcond_loss,
    selfcond_step_seed,
    validate_selfcond_prerequisites,
)
from steering.conditional.encoder_decoder.sft_features import (
    assert_sft_source_bos_char_len_matches,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.encoder_decoder.turn_cadence import (
    measure_turn_self_cond_segment_size_rustbpe,
)
from steering.conditional.feature_clusters import (
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
    FeatureClusterConsolidation,
    resolve_feature_clusters_from_checkpoint_meta,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    feature_mask_from_checkpoint_meta,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    build_feature_pipeline,
    build_feature_pipeline_from_cfg_block,
    feature_selection_dirname,
    negative_sampling_mask,
)
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    feature_vocab_from_state_dict,
)
from steering.conditional.missing_percentile import (
    MISSING_PERCENTILE_POLICY_META_KEY,
    MissingPercentilePolicy,
    MissingPercentileResolver,
    combined_coverage,
    parse_missing_percentile_policy,
    percentile_table_coverage,
)
from steering.conditional.percentile_table import (
    build_percentile_table as build_full_percentile_table,
)
from steering.conditional.sampling import (
    SELFCOND_ACTION_RANKING_POLICY,
    NegativeBundle,
    SamplingBundle,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import ThresholdConfig
from steering.feature_paths import parse_feature_dir, sft_phase_for
from steering.profiling.markers import (
    TimingRecorder,
    timed_region,
    timing_tags,
    use_profiler_markers,
    use_recorder,
)
from steering.refined_types import (
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveFloat,
    PositiveInt,
    PositiveIntLike,
    ZeroIntLike,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from steering.sft.loss_mask import (
    LossMaskMode,
    parse_loss_mask_mode,
    render_conversation_with_mask,
)
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    SFT_SEGMENT_MODE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    assert_resume_sft_segment_mode_matches,
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
)
from steering.sft_schema import SFTSplit
from steering.stats_io import (
    find_statistics_path,
    load_percentile_checksum,
    load_statistics,
    load_tokenizer_checksum,
)
from tasks.mixtures import STEERED_MIXTURES, SFTMixture, build_task_mixtures
from tasks.spellingbee import ensure_word_list_available

IDENTITY_KEYS = (
    "feature_dir",
    "feature_selection",
    "negative_selection",
    "seed",
    "segment_size",
    "sae",
    "num_buckets",
    "pooling",
    "dataset_phase",
)
STRUCTURAL_IDENTITY_KEYS = frozenset({"segment_size", "num_buckets"})

# SFT-mode self-conditioning shares conditioned SFT's cursor contract rather
# than inventing a parallel one: it persists BOTH the committed cursor
# (``sft_consumed``) and the per-rank buffered dataloader envelope
# (``sft_dataloader_state``), which is exactly what ``buffered_v2`` denotes.
# The generator refuses a consumed-only resume, so carrying the envelope is
# what makes SFT-mode resume exact (it also closes the repeat/skip slop that
# the committed-cursor-only contract had — see #1444).
SELFCOND_SFT_CURSOR_SEMANTICS = SFT_CURSOR_SEMANTICS_BUFFERED_V2
_SFT_ONLY_SELFCOND_KEYS = (
    "sft_dataset",
    "sft_feature_dir",
    "sft_parquet_dir",
    "val_sft_feature_dir",
    "val_sft_parquet_dir",
)
_FEATURE_MASK_CANONICAL_KEY = "feature_mask_canonical"
_CANONICAL_CLUSTER_SOURCE_SENTINEL = "<checkpoint-canonical-feature-clusters>"


@dataclass(frozen=True)
class _StartupPhase:
    """One host-wall startup phase."""

    name: str
    status: str
    elapsed_s: float
    cumulative_s: float
    metadata: dict[str, Any]


class _SelfCondStartupTrace(_SFTStartupTraceBase):
    """Rank-aware, deterministic startup diagnostics for self-conditioning.

    The trace is deliberately independent of tensors, CUDA synchronization,
    RNG, and the optional profiling-marker machinery.  Before distributed
    initialization, records are buffered so only rank zero emits them once
    ``compute_init`` has established the rank.  This keeps the early Hydra
    phases observable without making every DDP rank log them.
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._clock = time.perf_counter if clock is None else clock
        self._start = float(self._clock())
        self._records: list[_StartupPhase] = []
        self._phase_intervals: list[tuple[float, float]] = []
        self._pending_payloads: list[dict[str, Any]] = []
        self._master_process: bool | None = None
        self._summary_emitted = False
        self._first_step_total_s: float | None = None

    @property
    def total_elapsed_s(self) -> float:
        return max(0.0, float(self._clock()) - self._start)

    def set_master_process(self, master_process: bool) -> None:
        """Select the emitting rank and flush buffered pre-DDP phases."""

        self._master_process = master_process
        if master_process:
            for payload in self._pending_payloads:
                logger.info(f"selfcond_startup {canonical_json(payload)}")
        self._pending_payloads.clear()

    @staticmethod
    def _safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        try:
            resolved = dict(metadata)
            extra = resolved.pop("extra", None)
            if callable(extra):
                extra = extra()
            if isinstance(extra, Mapping):
                resolved.update(extra)
            elif extra is not None:
                resolved["extra"] = extra
            return resolved
        except BaseException as error:
            return {"metadata_error_type": type(error).__name__}

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        extra: Mapping[str, Any] | Any = None,
        include_in_sum: bool = True,
        **metadata: Any,
    ) -> Iterator[None]:
        """Record one phase, preserving the original exception on failure."""

        if extra is not None:
            metadata["extra"] = extra
        phase_start = float(self._clock())
        try:
            yield
        except BaseException as error:
            elapsed_s = max(0.0, float(self._clock()) - phase_start)
            self._phase_intervals.append((phase_start, phase_start + elapsed_s))
            resolved_metadata = self._safe_metadata(metadata)
            self._record(
                _StartupPhase(
                    name=name,
                    status="error",
                    elapsed_s=elapsed_s,
                    cumulative_s=self.total_elapsed_s,
                    metadata={
                        **resolved_metadata,
                        "error_type": type(error).__name__,
                    },
                )
            )
            raise
        else:
            elapsed_s = max(0.0, float(self._clock()) - phase_start)
            self._phase_intervals.append((phase_start, phase_start + elapsed_s))
            resolved_metadata = self._safe_metadata(metadata)
            self._record(
                _StartupPhase(
                    name=name,
                    status="ok",
                    elapsed_s=elapsed_s,
                    cumulative_s=self.total_elapsed_s,
                    metadata=resolved_metadata,
                )
            )

    def record(
        self,
        name: str,
        elapsed_ns: int,
        *,
        status: str = "completed",
        extra: Mapping[str, Any] | None = None,
        include_in_sum: bool = True,
    ) -> Any:
        """Accept the shared SFT generator's measured leaf-phase callback."""

        elapsed_s = max(0.0, float(elapsed_ns) / 1_000_000_000)
        phase_end = float(self._clock())
        self._phase_intervals.append((phase_end - elapsed_s, phase_end))
        self._record(
            _StartupPhase(
                name=name,
                status={"completed": "ok", "failed": "error"}.get(status, status),
                elapsed_s=elapsed_s,
                cumulative_s=self.total_elapsed_s,
                metadata=self._safe_metadata({"extra": extra}),
            )
        )

    def skipped(self, name: str, **metadata: Any) -> None:
        """Record an intentional non-applicable phase without timing work."""

        self._record(
            _StartupPhase(
                name=name,
                status="skipped",
                elapsed_s=0.0,
                cumulative_s=self.total_elapsed_s,
                metadata=metadata,
            )
        )

    def event(self, name: str, **metadata: Any) -> None:
        """Emit an instantaneous diagnostic with the current cumulative time."""

        self._emit(
            {
                "kind": "event",
                "event": name,
                "cumulative_s": self.total_elapsed_s,
                **metadata,
            }
        )

    def mark_first_step(self) -> None:
        """Freeze the main-entry-to-first-step wall time once."""

        if self._first_step_total_s is None:
            self._first_step_total_s = self.total_elapsed_s

    def complete(
        self,
        *,
        first_step_s: float | None,
        steady_state_s: list[float],
    ) -> None:
        """Emit the successful first-step/steady-state summary once."""

        total_s = (
            self._first_step_total_s
            if self._first_step_total_s is not None
            else self.total_elapsed_s
        )
        self._emit_summary(
            status="ok",
            scope=(
                "main_entry_to_first_completed_step"
                if first_step_s is not None
                else "main_entry_to_training_end_without_completed_step"
            ),
            total_s=total_s,
            first_step_s=first_step_s,
            steady_state_s=steady_state_s,
        )

    def failed(self, error: BaseException) -> None:
        """Emit a non-masking failure summary exactly once."""

        self._emit_summary(
            status="error",
            scope="main_entry_to_failure",
            total_s=self.total_elapsed_s,
            error_type=type(error).__name__,
        )

    def _record(self, record: _StartupPhase) -> None:
        self._records.append(record)
        self._log_phase(record)

    def _non_overlapping_phase_s(self) -> float:
        """Union nested phase intervals so the summary cannot double-count."""

        intervals = sorted(self._phase_intervals)
        if not intervals:
            return 0.0
        total = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                total += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        return total + end - start

    def _emit_summary(
        self,
        *,
        status: str,
        scope: str,
        total_s: float,
        first_step_s: float | None = None,
        steady_state_s: list[float] | None = None,
        error_type: str | None = None,
    ) -> None:
        if self._summary_emitted:
            return
        self._summary_emitted = True
        phase_s = self._non_overlapping_phase_s()
        payload: dict[str, Any] = {
            "kind": "summary",
            "status": status,
            "scope": scope,
            "total_s": total_s,
            "named_phase_s": phase_s,
            "unattributed_s": max(0.0, total_s - phase_s),
        }
        if error_type is not None:
            payload["error_type"] = error_type
        else:
            steady_state = [] if steady_state_s is None else steady_state_s
            payload.update(
                {
                    "first_step_s": first_step_s,
                    "steady_state_mean_s": (
                        sum(steady_state) / len(steady_state) if steady_state else None
                    ),
                    "steady_state_step_count": len(steady_state),
                }
            )
        self._emit(payload)

    def _log_phase(self, record: _StartupPhase) -> None:
        self._emit(
            {
                "kind": "phase",
                "phase": record.name,
                "status": record.status,
                "elapsed_s": record.elapsed_s,
                "cumulative_s": record.cumulative_s,
                **record.metadata,
            }
        )

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._master_process is True:
            logger.info(f"selfcond_startup {canonical_json(payload)}")
        elif self._master_process is None:
            self._pending_payloads.append(payload)

    def ensure_failure_visible(self) -> None:
        """Flush a pre-DDP failure as rank zero when no rank was established."""

        if self._master_process is None:
            rank_raw = os.environ.get("RANK")
            try:
                master_process = rank_raw is None or int(rank_raw) == 0
            except ValueError:
                # An unrelated launcher variable must not make diagnostics
                # disappear; compute_init remains the authority when startup
                # reaches distributed initialization.
                master_process = True
            self.set_master_process(master_process)


def _startup_phase(
    startup_trace: _SelfCondStartupTrace | None,
    name: str,
    **metadata: Any,
) -> AbstractContextManager[None]:
    """Return a no-op context unless self-conditioning startup tracing is active."""

    if startup_trace is None:
        return nullcontext()
    return startup_trace.phase(name, **metadata)


@dataclass(frozen=True)
class _SelfCondProfileSettings:
    """Strict, side-effect-free profiling configuration."""

    output_dir: Path
    trace_step: NonNegativeInt | None
    trace_steps: PositiveInt
    timing_enabled: bool

    @property
    def enabled(self) -> bool:
        return self.timing_enabled or self.trace_step is not None


def _resolve_selfcond_profile_settings(cfg: Mapping[str, Any]) -> _SelfCondProfileSettings:
    """Validate profiling values without coercing booleans or numeric strings."""

    raw_timing = cfg.get("profile_timing", False)
    if not isinstance(raw_timing, bool):
        raise ValueError(f"profile_timing must be a boolean, got {raw_timing!r}")

    raw_trace_step = cfg.get("profile_trace")
    if raw_trace_step is None:
        trace_step = None
    elif isinstance(raw_trace_step, bool) or not isinstance(raw_trace_step, int):
        raise ValueError(
            f"profile_trace must be a non-negative integer or null, got {raw_trace_step!r}"
        )
    else:
        try:
            trace_step = NonNegativeInt.parse(raw_trace_step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"profile_trace must be a non-negative integer or null, got {raw_trace_step!r}"
            ) from error

    raw_trace_steps = cfg.get("profile_trace_steps", 1)
    if isinstance(raw_trace_steps, bool) or not isinstance(raw_trace_steps, int):
        raise ValueError(f"profile_trace_steps must be a positive integer, got {raw_trace_steps!r}")
    try:
        trace_steps = PositiveInt.parse(raw_trace_steps)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"profile_trace_steps must be a positive integer, got {raw_trace_steps!r}"
        ) from error

    raw_output_dir = cfg.get(
        "profile_output_dir",
        "out/profiles/encdec_selfcond",
    )
    if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
        raise ValueError(f"profile_output_dir must be a non-empty string, got {raw_output_dir!r}")
    if raw_timing and trace_step is not None:
        raise ValueError(
            "profile_timing and profile_trace cannot be enabled together; "
            "run timing, trace, and throughput as separate benchmark arms"
        )

    return _SelfCondProfileSettings(
        output_dir=Path(raw_output_dir),
        trace_step=trace_step,
        trace_steps=trace_steps,
        timing_enabled=raw_timing,
    )


def _validate_selfcond_profile_window(
    settings: _SelfCondProfileSettings,
    *,
    start_step: int,
    num_iterations: int,
) -> None:
    """Require the complete absolute trace window to execute in this run."""

    if settings.trace_step is None:
        return
    trace_start = int(settings.trace_step)
    trace_end = trace_start + int(settings.trace_steps)
    if trace_start < start_step:
        raise ValueError(
            f"profile_trace={trace_start} precedes resume start_step={start_step}; "
            "choose an absolute step that this run will execute"
        )
    if trace_end > num_iterations:
        raise ValueError(
            "profile trace window must fit completely inside the training run: "
            f"[{trace_start}, {trace_end}) is outside "
            f"[start_step={start_step}, num_iterations={num_iterations})"
        )


class _SelfCondProfileSession:
    """Own one rank's optional timing recorder or bounded Chrome trace."""

    def __init__(
        self,
        settings: _SelfCondProfileSettings,
        *,
        rank: int,
        device_type: str,
    ) -> None:
        self.settings = settings
        self.rank = rank
        self.device_type = device_type
        self._timing_recorder = TimingRecorder() if settings.timing_enabled else None
        self._active_profiler: Any = None
        self._trace_exported = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def markers_enabled(self) -> bool:
        """True only while timing rows or the bounded trace can consume labels."""

        return self._timing_recorder is not None or self._active_profiler is not None

    def prepare_output_dir(self) -> None:
        """Create diagnostics output only for an explicitly enabled run."""

        if self.enabled:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def should_start_trace(self, step: int) -> bool:
        return self.settings.trace_step is not None and step == self.settings.trace_step

    def should_stop_trace(self, step: int) -> bool:
        return (
            self.settings.trace_step is not None
            and step == self.settings.trace_step + self.settings.trace_steps - 1
        )

    @contextmanager
    def recording(self, step: int) -> Iterator[None]:
        """Activate this rank and absolute-step tags for timing rows."""

        if self._timing_recorder is None:
            if self._active_profiler is not None:
                with use_profiler_markers():
                    yield
                return
            yield
            return
        with use_recorder(self._timing_recorder):
            with timing_tags(rank=self.rank, step=step):
                yield

    def start_trace(self, step: int) -> None:
        """Start the configured trace exactly once at its absolute first step."""

        if not self.should_start_trace(step):
            raise RuntimeError(f"trace cannot start at unconfigured step {step}")
        if self._active_profiler is not None or self._trace_exported:
            raise RuntimeError("self-conditioning trace was already started")
        activities = [t.profiler.ProfilerActivity.CPU]
        if self.device_type == "cuda" and t.cuda.is_available():
            activities.append(t.profiler.ProfilerActivity.CUDA)
        profiler = t.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            experimental_config=t.profiler._ExperimentalConfig(profile_all_threads=True),
        )
        self._active_profiler = profiler
        profiler.__enter__()

    def stop_and_export_trace(self, step: int) -> Path:
        """Stop and export one complete rank-local trace at the configured end."""

        if not self.should_stop_trace(step):
            raise RuntimeError(f"trace cannot stop at unconfigured step {step}")
        if self._active_profiler is None:
            raise RuntimeError("self-conditioning trace is not active")
        profiler = self._active_profiler
        self._active_profiler = None
        profiler.__exit__(None, None, None)
        path = self._trace_path(partial=False)
        profiler.export_chrome_trace(str(path))
        self._trace_exported = True
        logger.info(f"Exported self-conditioning Chrome trace to {path}")
        return path

    def finalize(self) -> Path | None:
        """Export timing output and assert the configured trace completed."""

        if self._active_profiler is not None:
            raise RuntimeError("training ended while the self-conditioning trace was active")
        if self.settings.trace_step is not None and not self._trace_exported:
            raise RuntimeError("configured self-conditioning trace was never exported")
        if self._timing_recorder is None:
            return None
        path = self.settings.output_dir / f"selfcond_timing_rank{self.rank}.csv"
        self._timing_recorder.to_csv(path)
        logger.info(f"Exported self-conditioning timing CSV to {path}")
        return path

    def close_on_error(self, error: BaseException) -> None:
        """Best-effort local diagnostics cleanup that never masks ``error``."""

        if self._active_profiler is not None:
            profiler = self._active_profiler
            self._active_profiler = None
            try:
                profiler.__exit__(type(error), error, error.__traceback__)
                path = self._trace_path(partial=True)
                profiler.export_chrome_trace(str(path))
                logger.warning(
                    f"Exported partial self-conditioning Chrome trace after failure to {path}"
                )
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original
                logger.warning(
                    "Could not export partial self-conditioning trace after "
                    f"{type(error).__name__}: {cleanup_error}"
                )
        if self._timing_recorder is not None:
            try:
                path = self.settings.output_dir / (f"selfcond_timing_rank{self.rank}_partial.csv")
                self._timing_recorder.to_csv(path)
                logger.warning(
                    f"Exported partial self-conditioning timing CSV after failure to {path}"
                )
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original
                logger.warning(
                    "Could not export partial self-conditioning timings after "
                    f"{type(error).__name__}: {cleanup_error}"
                )

    def _trace_path(self, *, partial: bool) -> Path:
        trace_step = self.settings.trace_step
        assert trace_step is not None
        suffix = "_partial" if partial else ""
        return self.settings.output_dir / (
            f"selfcond_trace_step{trace_step}_rank{self.rank}{suffix}.json"
        )


@dataclass(frozen=True)
class SelfCondSftSources:
    """Resolved SFT artifacts and generator policy recovered from the base."""

    dataset: SFTMixture
    train_parquet_dir: str
    train_feature_dir: str
    val_parquet_dir: str | None
    val_feature_dir: str | None
    loss_mask_mode: LossMaskMode
    min_loss_tokens: int
    feature_source_bos_char_len: int
    alignment_probe_samples: int
    segment_mode: SegmentMode
    # ERROR is the safe default (the fail-loud policy); the resolver always
    # sets this explicitly from the base checkpoint's recorded value.
    missing_percentile_policy: MissingPercentilePolicy = MissingPercentilePolicy.ERROR


@dataclass(frozen=True)
class SelfCondSftData:
    """Resolved train-statistics state shared by SFT train and validation."""

    pipeline: FeaturePipelineConfig
    threshold_cfg: ThresholdConfig | None
    precomputed_thresholds: t.Tensor | None
    sampling_bundle: SamplingBundle | None
    boundaries: t.Tensor
    tokenizer_checksum: str
    percentile_checksum: str | None
    feature_mask: FeatureMask | None = None
    consolidation: FeatureClusterConsolidation | None = None
    cluster_threshold_percentile_table: t.Tensor | None = None
    missing_percentile: MissingPercentileResolver | None = None


@dataclass(frozen=True)
class SelfCondSftEvalConfig:
    """Loop geometry and data seed for one SFT validation cycle."""

    B: int
    T: int
    eval_steps: int
    seed: int


def build_selfcond_sft_data(
    *,
    sources: SelfCondSftSources,
    base_meta: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any],
    n_sae_features: int,
    num_buckets: int,
    ddp_rank: int,
    tokenizer: Any,
    state_feature_vocab: FeatureVocab | None = None,
    startup_trace: _SelfCondStartupTrace | None = None,
) -> SelfCondSftData:
    """Rebuild an SFT base checkpoint's live conditioning pipeline.

    SFT features reuse the checkpoint's train statistics and selection policy.
    Negative sampling is mandatory for a self-conditioning-eligible base, and
    the production pipeline owns both inactive-bucket encoding and the shared
    sampling/negative RNG invariant.

    ``state_feature_vocab`` is the AUTHORITATIVE compaction record recovered
    from the checkpoint's state dict (``feature_vocab_from_state_dict``).
    Canonical-metadata reconstruction must agree with it id-for-id: metadata
    is a mirror, and a mirror can be hand-edited while the encoder's rows
    cannot (review finding).
    """
    with _startup_phase(
        startup_trace,
        "sft_statistics_load",
        feature_dir=sources.train_feature_dir,
    ):
        stats_path = find_statistics_path(sources.train_feature_dir)
        if stats_path is None:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft requires feature_statistics.safetensors "
                f"under train_feature_dir={sources.train_feature_dir!r}. Run SFT "
                "feature extraction with statistics computation first."
            )
        statistics = load_statistics(stats_path)
    assert_checkpoint_feature_universe(
        n_sae_features,
        statistics.metadata.n_sae_features,
    )

    feature_mask: FeatureMask | None = None
    consolidation: FeatureClusterConsolidation | None = None
    metadata = base_meta or {}
    compact_identity_keys = (
        FEATURE_VOCAB_META_KEY,
        _FEATURE_MASK_CANONICAL_KEY,
        FEATURE_CLUSTERS_DESCRIPTOR_KEY,
        FEATURE_CLUSTERS_CANONICAL_KEY,
    )
    user_config = metadata.get("user_config")
    recorded_clusters = (
        user_config.get("feature_clusters") if isinstance(user_config, Mapping) else None
    )
    compact_meta_present = (
        any(key in metadata for key in compact_identity_keys) or recorded_clusters is not None
    )
    # Detection binds to the AUTHORITATIVE state dict, not metadata presence:
    # a physically compact checkpoint whose canonical mirrors were stripped
    # must fail HERE, not later at the encoder's OOV assert; and canonical
    # mirrors on an uncompacted state dict are equally corrupt (review
    # finding).
    if state_feature_vocab is not None and not compact_meta_present:
        raise SelfCondTrainingError(
            "selfcond.data_source=sft compact checkpoint identity is corrupt: the "
            "state dict carries a compact FeatureVocab "
            f"(size={state_feature_vocab.size}) but the metadata records none of "
            f"the canonical keys {compact_identity_keys}. Restore the original "
            "complete SFT checkpoint metadata."
        )
    if state_feature_vocab is None and compact_meta_present:
        raise SelfCondTrainingError(
            "selfcond.data_source=sft compact checkpoint identity is corrupt: "
            "canonical compaction metadata is recorded but the state dict has no "
            "FeatureVocab buffers. Restore the matching checkpoint artifacts."
        )
    if compact_meta_present:
        missing_keys: list[str] = []
        for key in (FEATURE_VOCAB_META_KEY, _FEATURE_MASK_CANONICAL_KEY):
            if key not in metadata or metadata[key] is None:
                missing_keys.append(key)

        clusters_active = (
            recorded_clusters is not None
            or FEATURE_CLUSTERS_DESCRIPTOR_KEY in metadata
            or FEATURE_CLUSTERS_CANONICAL_KEY in metadata
        )
        if clusters_active:
            for key in (
                FEATURE_CLUSTERS_DESCRIPTOR_KEY,
                FEATURE_CLUSTERS_CANONICAL_KEY,
            ):
                if key not in metadata or metadata[key] is None:
                    missing_keys.append(key)
        if missing_keys:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                f"missing or null canonical metadata keys={missing_keys}. Restore "
                "the original complete SFT checkpoint metadata; config paths cannot "
                "reconstruct compact identity."
            )
        if not isinstance(metadata[FEATURE_VOCAB_META_KEY], Mapping):
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                f"{FEATURE_VOCAB_META_KEY} must be a fingerprint mapping, got "
                f"{type(metadata[FEATURE_VOCAB_META_KEY]).__name__}."
            )

        try:
            with _startup_phase(startup_trace, "sft_feature_mask_reconstruction"):
                feature_mask = feature_mask_from_checkpoint_meta(metadata)
        except FeatureMaskError as error:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint has invalid "
                f"{_FEATURE_MASK_CANONICAL_KEY}: {error}"
            ) from error
        if feature_mask is None:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                f"{_FEATURE_MASK_CANONICAL_KEY} did not reconstruct a FeatureMask."
            )

        cluster_source_path = _CANONICAL_CLUSTER_SOURCE_SENTINEL
        if isinstance(recorded_clusters, Mapping) and recorded_clusters.get("path") is not None:
            cluster_source_path = str(recorded_clusters["path"])
        try:
            with _startup_phase(startup_trace, "sft_cluster_metadata_resolution"):
                resolved_clusters = resolve_feature_clusters_from_checkpoint_meta(
                    metadata,
                    surface="self-conditioning SFT base checkpoint",
                    source_path=cluster_source_path,
                )
        except FeatureClustersError as error:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint has invalid canonical "
                f"feature cluster metadata: {error}"
            ) from error
        if resolved_clusters is not None:
            with _startup_phase(startup_trace, "sft_cluster_consolidation"):
                consolidation = FeatureClusterConsolidation.build(
                    resolved_clusters,
                    feature_mask,
                )
        with _startup_phase(startup_trace, "sft_effective_feature_vocab"):
            reconstructed_vocab = effective_feature_vocab(
                consolidation if consolidation is not None else feature_mask,
                n_sae_features,
            )
        if reconstructed_vocab is None:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                "canonical mask/clusters reconstruct an uncompacted identity vocabulary."
            )
        if metadata[FEATURE_VOCAB_META_KEY] != reconstructed_vocab.metadata_dict():
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                f"{FEATURE_VOCAB_META_KEY}={metadata[FEATURE_VOCAB_META_KEY]!r} "
                "does not match the vocabulary reconstructed from canonical mask/cluster "
                f"metadata ({reconstructed_vocab.metadata_dict()!r})."
            )
        # The state dict is the authority: metadata mirrors can be hand-edited
        # consistently with each other while the encoder's actual rows differ.
        # FeatureVocab equality compares the full original-id tuple, so a
        # same-width different-ids forgery cannot pass (review finding).
        assert state_feature_vocab is not None
        if reconstructed_vocab != state_feature_vocab:
            raise SelfCondTrainingError(
                "selfcond.data_source=sft compact checkpoint identity is corrupt: "
                "the vocabulary reconstructed from canonical mask/cluster metadata "
                f"(size={reconstructed_vocab.size}, "
                f"sha256={reconstructed_vocab.metadata_dict()['sha256']}) does not "
                "equal the state dict's FeatureVocab "
                f"(size={state_feature_vocab.size}, "
                f"sha256={state_feature_vocab.metadata_dict()['sha256']}). The "
                "encoder's rows are the authority; restore matching artifacts."
            )

    feature_selection = parse_feature_selection(cast(dict[str, Any], identity["feature_selection"]))
    negative_selection = parse_negative_selection(
        cast(dict[str, Any] | None, identity.get("negative_selection"))
    )
    with _startup_phase(startup_trace, "sft_conditioning_pipeline_reconstruction"):
        pipeline = build_feature_pipeline(
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            per_feature_stats=statistics.per_feature_stats,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            seed=SFT_FEATURE_PIPELINE_SEED,
            ddp_rank=ddp_rank,
            feature_delta=None,
            feature_mask=feature_mask,
        )
    if pipeline.negative_bundle is None:
        raise SelfCondTrainingError(
            "C4: selfcond.data_source=sft requires the SFT base checkpoint's "
            "resolved negative_selection so inactive features use the reserved "
            "encoder bucket. Re-run SFT from a self-conditioning-eligible base."
        )
    pipeline = replace(
        pipeline,
        negative_bundle=replace(
            pipeline.negative_bundle,
            feature_mask=negative_sampling_mask(pipeline, consolidation),
        ),
    )

    threshold_cfg: ThresholdConfig | None = None
    precomputed_thresholds: t.Tensor | None = None
    sampling_bundle: SamplingBundle | None = None
    if isinstance(pipeline.selection_config, ThresholdSelectionConfig):
        selection = pipeline.selection_config.selection
        threshold_cfg = ThresholdConfig(
            selection.threshold_type,
            selection.threshold,
        )
        precomputed_thresholds = pipeline.selection_config.precomputed_thresholds
    else:
        sampling_bundle = pipeline.selection_config

    cluster_threshold_percentile_table: t.Tensor | None = None
    if consolidation is not None and threshold_cfg is not None:
        with _startup_phase(startup_trace, "sft_percentile_table_load"):
            cluster_threshold_percentile_table = build_full_percentile_table(
                statistics.per_feature_stats,
                n_sae_features,
            )

    # Honor the base checkpoint's recorded missing-percentile policy (SFT
    # conditioning identity). ERROR keeps the default fail-loud resolver;
    # LOWEST rebuilds the same coverage-bounded substitution resolver the SFT
    # trainer runs under, over exactly the tables this run queries.
    missing_percentile: MissingPercentileResolver | None = None
    if sources.missing_percentile_policy is MissingPercentilePolicy.LOWEST:
        coverage_bitmaps = [
            percentile_table_coverage(table)
            for table in (
                sampling_bundle.percentile_table if sampling_bundle is not None else None,
                cluster_threshold_percentile_table,
            )
            if table is not None
        ]
        if not coverage_bitmaps:
            raise SelfCondTrainingError(
                "SFT base checkpoint records missing_percentile_policy='lowest' "
                "but this run queries no percentile table (threshold selection "
                "without clusters); the recorded policy cannot be honored. "
                "Repair the checkpoint metadata."
            )
        with _startup_phase(
            startup_trace,
            "sft_percentile_coverage_resolver",
            policy=MissingPercentilePolicy.LOWEST.value,
        ):
            missing_percentile = MissingPercentileResolver(
                policy=MissingPercentilePolicy.LOWEST,
                covered=combined_coverage(coverage_bitmaps),
            )

    tokenizer_checksum = load_tokenizer_checksum(stats_path)
    if tokenizer_checksum is None:
        tokenizer_checksum = compute_tokenizer_checksum(tokenizer)
    return SelfCondSftData(
        pipeline=pipeline,
        threshold_cfg=threshold_cfg,
        precomputed_thresholds=precomputed_thresholds,
        sampling_bundle=sampling_bundle,
        boundaries=pipeline.boundaries,
        tokenizer_checksum=tokenizer_checksum,
        # The statistics artifact is the source of the discretization
        # boundaries and the percentile table; without its checksum a
        # regenerated feature_statistics.safetensors could resume the cursor
        # on a silently different conditioning distribution (review finding).
        percentile_checksum=load_percentile_checksum(stats_path),
        feature_mask=feature_mask,
        consolidation=consolidation,
        cluster_threshold_percentile_table=cluster_threshold_percentile_table,
        missing_percentile=missing_percentile,
    )


def clone_val_bundles(
    sampling_bundle: SamplingBundle | None,
    negative_bundle: NegativeBundle,
    *,
    seed: int,
) -> tuple[SamplingBundle | None, NegativeBundle]:
    """Fresh identically-seeded bundle clones for one deterministic eval pass.

    One generator object is shared between both clones, preserving
    ``build_feature_pipeline``'s shared-RNG invariant.
    """
    rng = t.Generator().manual_seed(seed)
    cloned_sampling = replace(sampling_bundle, rng=rng) if sampling_bundle is not None else None
    return cloned_sampling, replace(negative_bundle, rng=rng)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _user_config(meta: Mapping[str, Any]) -> dict[str, Any]:
    raw = meta.get("user_config", {})
    if not isinstance(raw, dict):
        raise SelfCondTrainingError(
            "checkpoint metadata user_config must be a mapping for self-conditioning"
        )
    return cast(dict[str, Any], raw)


# The conditioned-SFT trainer's own default for rows too short to supervise.
DEFAULT_MIN_LOSS_TOKENS: Final = 8


def resolve_selfcond_data_source(
    selfcond_block: Mapping[str, Any],
) -> SelfCondDataSource:
    """Resolve the selected data family and reject mode-inapplicable keys."""
    raw = selfcond_block.get("data_source")
    try:
        source = SelfCondDataSource(str(raw))
    except ValueError as error:
        raise SelfCondTrainingError(
            f"selfcond.data_source={raw!r} is not a valid data source; expected one of "
            f"{[member.value for member in SelfCondDataSource]}"
        ) from error

    if source is SelfCondDataSource.PRETRAINING:
        stray = [key for key in _SFT_ONLY_SELFCOND_KEYS if selfcond_block.get(key) is not None]
        if stray:
            raise SelfCondTrainingError(
                f"selfcond.data_source=pretraining but SFT-only keys are set: {stray}. "
                "Set selfcond.data_source=sft or remove them — silently ignoring them "
                "is not an option."
            )
    return source


def _turn_cadence_renderer(
    task: Any, tokenizer: Any
) -> Callable[[int], tuple[list[int], list[int]]]:
    """Render one mixture row exactly as the conditioned-SFT generator does.

    The cadence estimator must see the SAME RustBPE ids the trainer trains on,
    so it goes through the converter's message slice and the shared
    ``render_conversation_with_mask`` rather than re-encoding parquet text.
    """

    def render(original_index: int) -> tuple[list[int], list[int]]:
        messages = _prepare_sft_render_messages(
            task[original_index],
            surface="turn cadence measurement",
            original_index=original_index,
        )
        return render_conversation_with_mask(tokenizer, {"messages": messages})

    return render


def assert_turn_cadence_is_not_configured(
    selfcond_block: Mapping[str, Any],
    *,
    data_source: SelfCondDataSource,
    segment_mode: SegmentMode,
) -> None:
    """Reject a configured cadence and a turn base fed pretraining rows.

    The generated-assistant cadence is MEASURED from the training artifacts by
    :func:`measure_turn_self_cond_segment_size_rustbpe`, so any presence of the
    key in the config -- including an explicit null -- is an operator trying to
    declare a value the run does not accept.
    """
    if TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in selfcond_block:
        raise SelfCondTrainingError(
            f"selfcond.{TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY} is not a "
            "config knob: the generated-assistant cadence is measured from the "
            "training artifacts, not configured. Remove the override."
        )
    if (
        segment_mode is SegmentMode.turn_delimited_fixed_tokens
        and data_source is not SelfCondDataSource.SFT
    ):
        raise SelfCondTrainingError(
            "a turn-delimited SFT checkpoint cannot be post-trained with "
            "selfcond.data_source=pretraining; select the checkpoint's SFT "
            "conversation source"
        )


def _resolve_selfcond_online_sae_lineage(
    checkpoint_meta: Mapping[str, Any],
    *,
    segment_mode: SegmentMode,
    checkpoint_dir: str,
    purpose: str,
) -> OnlineSAELineage | None:
    """Parse every present lineage and require it for turn-mode deployment."""

    if ONLINE_SAE_LINEAGE_META_KEY not in checkpoint_meta:
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
            raise SelfCondTrainingError(
                f"{purpose} at {checkpoint_dir} is turn-delimited but has no "
                f"{ONLINE_SAE_LINEAGE_META_KEY!r} block. Exact online extraction "
                "and selection provenance is required for post-training and inference."
            )
        return None
    try:
        return OnlineSAELineage.from_checkpoint_meta(checkpoint_meta)
    except ValueError as error:
        raise SelfCondTrainingError(
            f"{purpose} at {checkpoint_dir} has malformed online-SAE lineage: {error}"
        ) from error


def assert_resume_online_sae_lineage_matches(
    base_lineage: OnlineSAELineage | None,
    resume_lineage: OnlineSAELineage | None,
    *,
    segment_mode: SegmentMode,
) -> None:
    """Keep turn-mode online extraction identity immutable across resume."""

    if segment_mode is SegmentMode.turn_delimited_fixed_tokens and resume_lineage != base_lineage:
        raise SelfCondTrainingError(
            "resume checkpoint online-SAE lineage disagrees with the turn-delimited base checkpoint"
        )


def assert_sft_shaped_base(
    base_meta: Mapping[str, Any],
    base_checkpoint_dir: str,
) -> None:
    """Require the resolved base to be an SFT checkpoint leaf (D1).

    Tag-level discovery ends at the pretraining checkpoint family, so SFT mode
    cannot infer an SFT leaf from ``model_tag``. The shape check mirrors the
    checkpoint-conditioning reader: SFT metadata has no ``user_config.encoder``
    and records ``encoder_config`` plus ``dataset`` at the top level.
    """
    user_config = base_meta.get("user_config")
    is_sft_shaped = (
        isinstance(user_config, dict)
        and "encoder" not in user_config
        and "encoder_config" in base_meta
        and "dataset" in base_meta
    )
    if not is_sft_shaped:
        raise SelfCondTrainingError(
            "selfcond.data_source=sft v1 requires an SFT checkpoint leaf, but "
            f"base_checkpoint_dir={base_checkpoint_dir!r} is not SFT-shaped. "
            "Pass the SFT leaf explicitly via base_checkpoint_dir=<sft-leaf>; "
            "model_tag discovery cannot locate SFT leaves."
        )


def _assert_sft_feature_dir_structure(
    feature_dir: str,
    *,
    dataset: SFTMixture,
    split: SFTSplit,
    segment_size: int,
    segment_mode: SegmentMode,
    role: str,
    was_explicit: bool,
) -> bool:
    """Phase/segment structural guard on the SYMLINK-RESOLVED feature path.

    The guard parses ``Path(feature_dir).resolve()`` — a lexically canonical
    SFT path symlinked into a pretraining tree must be caught by its REAL
    phase, not its name (review finding). Returns True when the directory was
    structurally verified; False when an EXPLICIT non-canonical directory was
    accepted through the D8 escape hatch with a loud warning. A non-canonical
    RECOVERED or DERIVED directory raises: it is corrupt identity, not an
    operator choice.
    """
    resolved_path = str(Path(feature_dir).resolve())
    try:
        components = parse_feature_dir(resolved_path)
    except ValueError as error:
        if was_explicit:
            logger.warning(
                f"non-canonical SFT {role} feature dir accepted; phase isolation "
                f"cannot be verified: {feature_dir!r} (resolves to {resolved_path!r})"
            )
            return False
        raise SelfCondTrainingError(
            f"The SFT {role} feature directory {feature_dir!r} (resolved: "
            f"{resolved_path!r}) is non-canonical. Recovered/derived feature "
            "directories are part of the checkpoint's structural identity; use a "
            "modern SFT checkpoint with a canonical feature path or pass an "
            "explicit override."
        ) from error
    expected_phase = sft_phase_for(dataset.value, split)
    if components.dataset_phase is not expected_phase:
        raise SelfCondTrainingError(
            f"SFT {role} feature directory {feature_dir!r} (resolved: "
            f"{resolved_path!r}) has dataset "
            f"phase={components.dataset_phase.value!r}, but dataset "
            f"{dataset.value!r} requires phase={expected_phase.value!r}. Use the "
            f"matching SFT {split.value.upper()} feature directory; pretraining "
            "features cannot be used in SFT mode."
        )
    if components.segment_size != segment_size:
        raise SelfCondTrainingError(
            f"SFT {role} feature directory {feature_dir!r} has "
            f"segment_size={components.segment_size}, but the runtime "
            f"checkpoint requires segment_size={segment_size}. Use matching "
            "SFT features."
        )
    if components.segment_mode is not segment_mode:
        raise SelfCondTrainingError(
            f"SFT {role} feature directory {feature_dir!r} has "
            f"segment_mode={components.segment_mode.value!r}, but the SFT base "
            f"checkpoint records segment_mode={segment_mode.value!r}. Use "
            "features extracted in the checkpoint's segmentation mode."
        )
    return True


def resolve_selfcond_sft_sources(
    selfcond_block: Mapping[str, Any],
    base_meta: Mapping[str, Any],
    base_dir: str,
    *,
    base_checkpoint_dir: str,
    eval_every: int,
    segment_size: int,
) -> SelfCondSftSources:
    """Resolve SFT train/val sources without falling back to pretraining data.

    Checkpoint-recovered feature directories must be canonical because they are
    part of the checkpoint's structural identity. An explicit non-canonical
    train directory is the sole escape hatch; when validation is enabled it
    requires explicit val directories because neither split can be derived or
    structurally verified from that path.
    """
    assert_sft_shaped_base(base_meta, base_checkpoint_dir)
    # The recorded missing-percentile policy is conditioning IDENTITY, exactly
    # like the loss mask: SFT itself REQUIRES policy continuity on resume
    # (assert_resume_missing_percentile_policy_matches) rather than rejecting.
    # A `lowest` pilot base therefore continues under `lowest` — with the
    # SFT trainer's own NON-COMPARABLE warning — instead of being rejected
    # (the owner's l22 production family is a lowest-policy run).
    recorded_policy_raw = base_meta.get(
        MISSING_PERCENTILE_POLICY_META_KEY,
        MissingPercentilePolicy.ERROR.value,
    )
    missing_percentile_policy = parse_missing_percentile_policy(recorded_policy_raw)
    if missing_percentile_policy is MissingPercentilePolicy.LOWEST:
        logger.warning(
            "SFT base checkpoint records missing_percentile_policy='lowest' "
            "(PILOT): self-conditioning continues under the same policy so "
            "conditioning matches the SFT run exactly, but val/bpb and "
            "val/bpb_selfcond from this run are NOT comparable to a run "
            "trained under missing_percentile_policy=error. The policy is "
            "persisted in selfcond_data and resume under a different policy "
            "is rejected."
        )
    if read_checkpoint_conditioning(base_meta).negative_selection is None:
        # Pre-empt the generic C4 raise with the remedy an operator actually
        # needs here: legacy thin SFT checkpoints predate the
        # pretrain_conditioning block and cannot seed SFT-mode
        # self-conditioning (review finding — the real d12_fp safety SFT leaf
        # in the artifact bucket is exactly this shape).
        raise SelfCondTrainingError(
            f"SFT base checkpoint {base_checkpoint_dir!r} has no recoverable "
            "negative_selection conditioning metadata (no user_config."
            "negative_selection and no pretrain_conditioning block). Legacy "
            "SFT checkpoints written before the pretrain_conditioning re-publish "
            "cannot seed self-conditioning (C4). Re-run conditional_encdec_sft "
            "from the eligible pretraining base with current code to mint a "
            "usable SFT leaf."
        )

    explicit_dataset = selfcond_block.get("sft_dataset")
    recorded_dataset = base_meta["dataset"]
    dataset_raw = recorded_dataset if explicit_dataset is None else explicit_dataset
    try:
        dataset = SFTMixture(str(dataset_raw))
    except ValueError as error:
        raise SelfCondTrainingError(
            f"SFT dataset {dataset_raw!r} is invalid; expected one of "
            f"{[member.value for member in SFTMixture]}. Set "
            "selfcond.sft_dataset to a supported mixture or repair the SFT "
            "checkpoint's recorded dataset."
        ) from error
    if explicit_dataset is not None and str(recorded_dataset) != dataset.value:
        raise SelfCondTrainingError(
            f"selfcond.sft_dataset={dataset.value!r} disagrees with the SFT base "
            f"checkpoint's recorded dataset={recorded_dataset!r}. Cross-dataset "
            "self-conditioning is unsupported in v1; use a matching SFT checkpoint."
        )
    if dataset in STEERED_MIXTURES:
        # A steered SFT run opens a NEW feature-prediction group at each steered
        # conversation's assistant turn; the rollout would have to reproduce that
        # split slot-for-slot to stay aligned with the objective the head was
        # trained under. Out of scope for v1 — reject HERE, before any artifact
        # preflight or store load, so nothing downstream has to know about it.
        raise SelfCondTrainingError(
            "self-conditioning SFT mode does not support steered mixtures "
            f"(dataset={dataset.value!r})"
        )

    user_config = _user_config(base_meta)
    try:
        segment_mode = resolve_checkpoint_sft_segment_mode(
            base_meta,
            checkpoint_dir=base_checkpoint_dir,
        )
    except ValueError as error:
        raise SelfCondTrainingError(str(error)) from error
    if SFT_SEGMENT_MODE_META_KEY not in base_meta:
        logger.warning(
            f"SFT base checkpoint has no {SFT_SEGMENT_MODE_META_KEY} at the top level; "
            f"the strict checkpoint-witness resolver selected "
            f"{segment_mode.value!r} from remaining metadata or the legacy "
            f"{SegmentMode.fixed_tokens.value!r} default."
        )
    explicit_feature_dir = selfcond_block.get("sft_feature_dir")
    feature_dir_was_explicit = explicit_feature_dir is not None
    train_feature_raw = (
        explicit_feature_dir if feature_dir_was_explicit else user_config.get("feature_dir")
    )
    if train_feature_raw is None:
        raise SelfCondTrainingError(
            "selfcond.data_source=sft could not resolve a train feature directory: "
            "set selfcond.sft_feature_dir explicitly or use an SFT checkpoint whose "
            "user_config.feature_dir is recorded."
        )
    train_feature_dir = str(train_feature_raw)

    train_guard_verified = _assert_sft_feature_dir_structure(
        train_feature_dir,
        dataset=dataset,
        split=SFTSplit.TRAIN,
        segment_size=segment_size,
        segment_mode=segment_mode,
        role="train",
        was_explicit=feature_dir_was_explicit,
    )
    if (
        not train_guard_verified
        and eval_every > 0
        and (
            selfcond_block.get("val_sft_feature_dir") is None
            or selfcond_block.get("val_sft_parquet_dir") is None
        )
    ):
        raise SelfCondTrainingError(
            "An explicit non-canonical selfcond.sft_feature_dir cannot derive "
            "validation identity when eval_every > 0. Set both "
            "selfcond.val_sft_feature_dir and selfcond.val_sft_parquet_dir "
            "explicitly, or use a canonical train feature directory."
        )

    explicit_train_parquet = selfcond_block.get("sft_parquet_dir")
    recorded_train_parquet = user_config.get("sft_parquet_dir")
    train_parquet_override = (
        explicit_train_parquet if explicit_train_parquet is not None else recorded_train_parquet
    )
    train_parquet_dir = _resolve_sft_parquet_dir(
        None if train_parquet_override is None else str(train_parquet_override),
        base_dir,
        dataset.value,
    )

    loss_mask_raw = base_meta.get(
        "sft_loss_mask",
        LossMaskMode.FULL_SEQUENCE.value,
    )
    if "sft_loss_mask" not in base_meta:
        logger.warning(
            "SFT base checkpoint has no sft_loss_mask; using the legacy "
            f"default {LossMaskMode.FULL_SEQUENCE.value!r}."
        )
    loss_mask_mode = parse_loss_mask_mode(loss_mask_raw)
    min_loss_tokens = int(user_config.get("min_loss_tokens", DEFAULT_MIN_LOSS_TOKENS))
    feature_source_bos_char_len = int(user_config.get("feature_source_bos_char_len", 5))
    alignment_probe_samples = int(user_config.get("alignment_probe_samples", 8))

    # Validation-source precedence mirrors the train side: explicit config
    # override > the SFT checkpoint's RECORDED val locations (the real l22
    # family records non-default val_sft_parquet_dir/val_feature_dir; deriving
    # by convention would resolve a DIFFERENT parquet family and fail the
    # mixture-fingerprint gate) > convention-derived defaults.
    val_parquet_raw = selfcond_block.get("val_sft_parquet_dir")
    if val_parquet_raw is None:
        val_parquet_raw = user_config.get("val_sft_parquet_dir")
    val_feature_explicit = selfcond_block.get("val_sft_feature_dir")
    val_feature_raw = (
        val_feature_explicit
        if val_feature_explicit is not None
        else user_config.get("val_feature_dir")
    )
    val_parquet_dir, val_feature_dir = _resolve_and_require_sft_val_artifacts(
        has_val=True,
        eval_every=eval_every,
        val_parquet_override=(None if val_parquet_raw is None else str(val_parquet_raw)),
        val_feature_override=(None if val_feature_raw is None else str(val_feature_raw)),
        base_dir=base_dir,
        feature_dir=train_feature_dir,
        dataset=dataset.value,
        val_dataset=dataset.value,
    )
    if val_feature_dir is not None:
        # The VAL feature directory gets the SAME structural guard as train:
        # an explicit val override pointing at a pretraining tree was
        # previously accepted outright (review finding). Derived dirs come
        # from a verified canonical train path, so a non-canonical resolution
        # there is corrupt derivation and raises (was_explicit only for a
        # real operator override).
        _assert_sft_feature_dir_structure(
            val_feature_dir,
            dataset=dataset,
            split=SFTSplit.VAL,
            segment_size=segment_size,
            segment_mode=segment_mode,
            role="validation",
            # The D8 escape hatch keys on OPERATOR-explicit overrides only: a
            # val dir recovered from checkpoint metadata is structural
            # identity and must stay strict.
            was_explicit=val_feature_explicit is not None,
        )

    for label, value in (
        ("dataset", dataset.value),
        ("train_parquet_dir", train_parquet_dir),
        ("train_feature_dir", train_feature_dir),
        ("val_parquet_dir", val_parquet_dir),
        ("val_feature_dir", val_feature_dir),
        ("loss_mask_mode", loss_mask_mode.value),
        ("segment_mode", segment_mode.value),
        ("min_loss_tokens", min_loss_tokens),
        ("feature_source_bos_char_len", feature_source_bos_char_len),
        ("alignment_probe_samples", alignment_probe_samples),
    ):
        logger.info(f"Resolved selfcond SFT {label}={value!r}")

    return SelfCondSftSources(
        dataset=dataset,
        train_parquet_dir=train_parquet_dir,
        train_feature_dir=train_feature_dir,
        val_parquet_dir=val_parquet_dir,
        val_feature_dir=val_feature_dir,
        loss_mask_mode=loss_mask_mode,
        min_loss_tokens=min_loss_tokens,
        feature_source_bos_char_len=feature_source_bos_char_len,
        alignment_probe_samples=alignment_probe_samples,
        missing_percentile_policy=missing_percentile_policy,
        segment_mode=segment_mode,
    )


def explicit_override_roots(overrides: list[str]) -> set[str]:
    """Return top-level config keys explicitly named on the Hydra CLI."""
    roots: set[str] = set()
    for override in overrides:
        key = override.split("=", maxsplit=1)[0].lstrip("+~")
        if key:
            roots.add(key.split(".", maxsplit=1)[0])
    return roots


def resolve_base_checkpoint_dir(
    config: Mapping[str, Any],
    base_dir: str | Path,
) -> str:
    """Resolve the base checkpoint path and enforce runtime tag requiredness."""
    explicit_dir = config.get("base_checkpoint_dir")
    if explicit_dir is not None:
        return str(explicit_dir)
    model_tag = config.get("model_tag")
    if model_tag is None:
        raise SelfCondTrainingError(
            "model_tag is required for self-conditioning post-training when "
            "base_checkpoint_dir is not set (set model_tag=<tag> or "
            "base_checkpoint_dir=<explicit-path>)"
        )
    selection_raw = _plain(config.get("feature_selection"))
    if not isinstance(selection_raw, dict):
        raise SelfCondTrainingError("feature_selection must be a mapping")
    selection = parse_feature_selection(selection_raw)
    return str(
        Path(base_dir)
        / "conditional_encdec_checkpoints"
        / str(model_tag)
        / f"s{int(config['segment_size'])}"
        / feature_selection_dirname(selection)
    )


_CHECKPOINT_LEAF_PREFIXES = ("bl_", "wl_", "grad_", "fc_")

# Validation loaders are constructed per eval, consume exactly eval_steps
# batches, and are then dropped. This is an INVARIANT, not a tunable: any
# prefetch wrapping would leave the producer daemon thread blocked on its
# queue forever, leaking the thread plus a buffered device batch every eval
# cycle (pretraining's _build_validation_loader is bare for the same reason).
VAL_EVAL_PREFETCH_BATCHES: ZeroIntLike = 0


def resolve_adam_betas(cfg: Any) -> tuple[float, float]:
    """Adam betas come from THIS run's composed config, like every sibling
    training script (`conditional_encdec_train.py` reads ``cfg.adam_beta*``
    directly). Reading them from the base checkpoint's recorded user_config
    instead would silently ignore a CLI ``adam_beta1=``/``adam_beta2=``
    override — the composed config carries these keys via base_train.yaml, so
    an override MUST take effect (review finding). The encoder/xattn LR knobs
    remain meta-sourced because they are NOT part of this config surface."""
    return float(cfg.adam_beta1), float(cfg.adam_beta2)


@beartype
def resolve_grad_accum_steps(
    device_batch_size: PositiveInt,
    sequence_len: PositiveInt,
    ddp_world_size: PositiveInt,
    total_batch_size: int,
) -> PositiveInt:
    """Derive grad accumulation from refined positive token-budget factors."""
    micro_tokens = device_batch_size * sequence_len * ddp_world_size
    if total_batch_size <= 0:
        total_batch_size = micro_tokens
    if total_batch_size % micro_tokens:
        raise SelfCondTrainingError(
            f"total_batch_size={total_batch_size} must be divisible by per-step "
            f"micro tokens={micro_tokens}"
        )
    return PositiveInt.parse(total_batch_size // micro_tokens)


def resolve_dataloader_resume(resume_meta: dict | None) -> dict | None:
    """Return the checkpoint's loader cursor, raising loudly when it is absent.

    Every checkpoint this script writes persists ``dataloader_state_dict``; a
    resume checkpoint without a well-formed one is a hand-repaired or
    corrupted artifact, and silently restarting the cursor from epoch 1 /
    batch 0 would train on already-seen data without any signal (fail-loud
    doctrine — review finding). Fresh starts (``resume_meta is None``) simply
    return None.
    """
    if resume_meta is None:
        return None
    state = resume_meta.get("dataloader_state_dict")
    if not isinstance(state, dict) or not state:
        raise SelfCondTrainingError(
            "resume checkpoint metadata has no well-formed dataloader_state_dict "
            f"(got {type(state).__name__}); refusing to silently restart the "
            "loader cursor from scratch. Repair the checkpoint meta or start fresh "
            "with resume_from_step=-1."
        )
    return state


def assert_training_window(start_step: int, num_iterations: int) -> None:
    """A run whose loop body would never execute is a misconfiguration.

    Covers fresh ``num_iterations=0`` and resuming at/past ``num_iterations``
    (e.g. a checkpoint saved under a larger previous budget): without this the
    script would no-op silently and still log a completion message.
    """
    if start_step >= num_iterations:
        raise SelfCondTrainingError(
            f"nothing to train: start step {start_step} >= num_iterations "
            f"{num_iterations}. Raise num_iterations (resume continues from the "
            "checkpoint's recorded step) or start fresh."
        )


def assert_base_dir_has_checkpoint(base_checkpoint_dir: str | Path) -> None:
    """Turn a leaf-nested checkpoint into an eligibility error, not a 404.

    The tag-based locator assembles the legacy leaf-less pretraining path.
    Runs trained with a feature mask, gradient ranking, or feature clusters
    write checkpoints under an appended ``bl_/wl_/grad_/fc_`` identity leaf.
    Tag-based discovery cannot choose that leaf, so a bare FileNotFoundError
    would misdirect the user into path debugging. The operator must name the
    exact leaf; source-specific eligibility is validated later.
    """
    directory = Path(base_checkpoint_dir)
    if list(directory.glob("model_*.pt")):
        return
    leaf_dirs = sorted(
        str(child.name)
        for child in directory.glob("*")
        if child.is_dir()
        and child.name.startswith(_CHECKPOINT_LEAF_PREFIXES)
        and list(child.glob("model_*.pt"))
    )
    if leaf_dirs:
        raise SelfCondTrainingError(
            f"no checkpoint at {directory}, but leaf checkpoint dir(s) {leaf_dirs} exist "
            "beneath it. Those leaves carry feature-mask, gradient-ranking, or "
            "feature-cluster identity, so the model_tag locator cannot select one. "
            "Pass the exact leaf via base_checkpoint_dir=; the selected data source "
            "will then apply its own compaction/selection eligibility checks."
        )
    raise SelfCondTrainingError(
        f"no encoder-decoder checkpoint found at {directory} (no model_*.pt). "
        "Check model_tag/segment_size/feature_selection, or pass "
        "base_checkpoint_dir= explicitly."
    )


def resolve_val_dirs(
    discovery: Any,
    eval_every: int,
) -> tuple[Path | None, Path | None]:
    """Resolve the validation packed/selection dirs, failing loud when evals are on.

    ``discover_packed_selection`` only fills ``val_packed_dir`` on the special
    branches v1 rejects; on the plain path pretraining resolves the val sibling
    AT EVAL TIME via ``discover_val_packed_row_dir``. Without this mirror the
    in-loop eval guard is permanently dead and val metrics silently never run
    (review finding — the fail-loud raise makes that state impossible).
    """
    val_packed_dir = discovery.val_packed_dir
    if val_packed_dir is None and discovery.train_packed_dir is not None:
        val_packed_dir = discover_val_packed_row_dir(discovery.train_packed_dir)
    if eval_every > 0 and val_packed_dir is None:
        raise SelfCondTrainingError(
            "eval_every > 0 but no validation packed rows were found (no "
            f"val_* sibling of {discovery.train_packed_dir}). Either precompute "
            "validation packed rows (precompute_packed_rows writes the val_ "
            "sibling) or set eval_every=0 explicitly — silently skipping "
            "validation is not an option."
        )
    return val_packed_dir, discovery.val_selection_dir


def resolve_checkpoint_num_buckets(meta: Mapping[str, Any]) -> int:
    """Read the recorded bucket count without an unsafe fallback.

    Delegates to the module-level resolver so the user_config-over-top-level
    precedence rule has exactly ONE implementation shared with the C3
    validation (review finding: three independent copies could drift).
    """
    return resolve_recorded_num_buckets(meta)


def resolve_identity_config(
    config: Mapping[str, Any],
    meta: Mapping[str, Any],
    explicit_keys: set[str],
    *,
    log: Any = logger,
) -> dict[str, Any]:
    """Apply R14: checkpoint identity by default, loud explicit overrides."""
    user = _user_config(meta)
    resolved: dict[str, Any] = {}
    for key in IDENTITY_KEYS:
        recorded = _plain(user.get(key, meta.get(key)))
        configured = _plain(config.get(key))
        if key in explicit_keys and configured is not None:
            if key in STRUCTURAL_IDENTITY_KEYS and recorded is not None and configured != recorded:
                raise SelfCondTrainingError(
                    f"explicit {key}={configured!r} disagrees with checkpoint metadata "
                    f"{recorded!r}; structural identity cannot be overridden"
                )
            if recorded is not None and configured != recorded:
                log.warning(
                    f"Explicit self-conditioning override for checkpoint identity key "
                    f"{key}: checkpoint={recorded!r}, override={configured!r}"
                )
            resolved[key] = configured
        elif key in explicit_keys and recorded is not None:
            # An explicit `key=null` override cannot beat the checkpoint value
            # (identity keys have no meaningful null state) — but silently
            # ignoring the user's directive violates loudness (review finding).
            log.warning(
                f"Explicit null override for checkpoint identity key {key} has no "
                f"effect; the checkpoint's recorded value {recorded!r} is used"
            )
            resolved[key] = recorded
        elif recorded is not None:
            resolved[key] = recorded
        elif configured is not None:
            resolved[key] = configured
    return resolved


def resolve_selfcond_train_config(
    meta: Mapping[str, Any],
    identity: Mapping[str, Any],
    selfcond: Mapping[str, Any],
    n_sae_features: int,
) -> SelfCondTrainConfig:
    """Resolve pool/count inheritance into the immutable runtime config."""
    fp_config = _resolve_fp_config_from_meta(meta)
    if fp_config is None:
        raise SelfCondTrainingError(
            "C1: checkpoint metadata does not contain a recoverable feature_prediction config"
        )
    pool_raw = selfcond.get("pool_mode")
    pool_mode = fp_config.loss_mode if pool_raw is None else FeaturePredLossMode(str(pool_raw))
    feature_selection = identity.get("feature_selection")
    negative_selection = identity.get("negative_selection")
    if not isinstance(feature_selection, dict):
        raise SelfCondTrainingError(
            "checkpoint feature_selection metadata is required to inherit positive counts"
        )
    if negative_selection is not None and not isinstance(negative_selection, dict):
        raise SelfCondTrainingError("checkpoint negative_selection metadata must be a mapping")
    positive_override = _plain(selfcond.get("positive_counts"))
    negative_override = _plain(selfcond.get("negative_counts"))
    positive, negative = resolve_selfcond_count_configs(
        feature_selection=feature_selection,
        negative_selection=negative_selection,
        positive_override=positive_override,
        negative_override=negative_override,
    )
    cfg = SelfCondTrainConfig(
        pool_mode=pool_mode,
        train_scope=SelfCondTrainScope(str(selfcond["train_scope"])),
        ste_form=SteForm(str(selfcond["ste_form"])),
        positive_counts=positive,
        negative_counts=negative,
        seed=int(identity.get("seed", 42)),
    )
    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    max_selected = (cfg.positive_counts.max_k - 1) + (cfg.negative_counts.max_k - 1)
    if max_selected > n_sae_features:
        raise SelfCondTrainingError(
            f"maximum self-conditioning count sum {max_selected} exceeds "
            f"n_sae_features={n_sae_features}"
        )
    return cfg


def validate_selfcond_startup(
    model: EncoderDecoderGPT,
    meta: dict,
    selfcond_cfg: SelfCondTrainConfig,
    n_sae_features: int,
    runtime_cfg: Mapping[str, Any],
    *,
    data_source: SelfCondDataSource = SelfCondDataSource.PRETRAINING,
) -> None:
    """Validate unwired script surfaces and the complete C1-C8 matrix."""
    for key in ("core_metric_every", "sample_every", "steering_eval_every"):
        if int(runtime_cfg.get(key, 0)) > 0:
            raise SelfCondTrainingError(
                f"{key} is not wired for self-conditioning post-training v1; set it <= 0"
            )
    if float(runtime_cfg.get("label_dropout", 0.0)) != 0.0:
        raise SelfCondTrainingError("C5: label_dropout must equal 0")
    if parse_conditioning_cooldown(runtime_cfg.get("conditioning_cooldown")) is not None:
        raise SelfCondTrainingError("C5: conditioning_cooldown is not wired for self-conditioning")
    if bool(runtime_cfg.get("compile", False)):
        raise SelfCondTrainingError(
            "compile=true is unsupported in self-conditioning v1 (three distinct graphs)"
        )
    validate_selfcond_prerequisites(
        model,
        meta,
        selfcond_cfg,
        n_sae_features,
        data_source=data_source,
    )


def assert_checkpoint_feature_universe(
    checkpoint_n_sae_features: int,
    statistics_n_sae_features: int,
) -> None:
    """Fail before data construction if checkpoint and statistics disagree."""
    if checkpoint_n_sae_features != statistics_n_sae_features:
        raise FeatureVocabError(
            "checkpoint n_sae_features disagrees with feature statistics: "
            f"{checkpoint_n_sae_features} != {statistics_n_sae_features}"
        )


def _serialize_counts(counts: Any) -> dict[str, Any]:
    return {
        "distribution": counts.distribution.value,
        "mean": counts.mean,
        "std": counts.std,
        "min_k": counts.min_k,
        "max_k": counts.max_k,
    }


def assert_resume_selfcond_matches(
    resume_meta: Mapping[str, Any],
    cfg: SelfCondTrainConfig,
    *,
    data_source: SelfCondDataSource = SelfCondDataSource.PRETRAINING,
) -> None:
    """Refuse to resume under a different rollout policy than the checkpoint's.

    ``train_scope``/``ste_form`` self-protect through the checkpoint directory
    name, but ``pool_mode``, the count distributions, and the action seed do not
    — a changed override would silently continue training under a different
    policy than the one that produced the resumed weights. Mirrors the
    defense-in-depth of SFT's ``assert_resume_feature_mask_matches``.
    """
    recorded = resume_meta.get("selfcond")
    if not isinstance(recorded, Mapping):
        raise SelfCondTrainingError(
            "resume checkpoint metadata has no 'selfcond' block; this checkpoint was "
            "not produced by self-conditioning post-training and cannot be resumed here"
        )
    if "data_source" not in recorded:
        logger.info(
            "Legacy self-conditioning resume checkpoint has no selfcond.data_source; "
            "defaulting the recorded source to 'pretraining'. Checkpoints written "
            "before source-aware SFT support necessarily used pretraining data."
        )
    recorded_data_source = recorded.get("data_source", SelfCondDataSource.PRETRAINING.value)
    current: dict[str, Any] = {
        "data_source": data_source.value,
        "action_ranking_policy": SELFCOND_ACTION_RANKING_POLICY,
        "train_scope": cfg.train_scope.value,
        "ste_form": cfg.ste_form.value,
        "pool_mode": cfg.pool_mode.value,
        "positive_counts": _serialize_counts(cfg.positive_counts),
        "negative_counts": _serialize_counts(cfg.negative_counts),
        "seed": cfg.seed,
    }
    mismatches = {
        key: (
            recorded_data_source if key == "data_source" else recorded.get(key),
            value,
        )
        for key, value in current.items()
        if (recorded_data_source if key == "data_source" else recorded.get(key)) != value
    }
    if mismatches:
        rendered = "\n".join(
            f"  {key}: checkpoint={old!r} current={new!r}" for key, (old, new) in mismatches.items()
        )
        raise SelfCondTrainingError(
            "resume selfcond policy mismatch — the resumed checkpoint was trained under a "
            "different self-conditioning policy than the current config resolves:\n"
            f"{rendered}\n"
            "Re-specify the original selfcond settings (or start a fresh run) rather than "
            "silently switching policies mid-run."
        )


def resolve_sft_mode_resume_state(
    resume_meta: Mapping[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    """Return the SFT cursor and optional buffered loader state."""
    loop_state = resume_meta.get("loop_state")
    if not isinstance(loop_state, Mapping):
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume requires a loop_state mapping with "
            "buffered cursor metadata; start a fresh run if the checkpoint was "
            "not written by source-aware self-conditioning."
        )
    cursor_semantics = loop_state.get("sft_cursor_semantics")
    if cursor_semantics != SELFCOND_SFT_CURSOR_SEMANTICS:
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume requires "
            "loop_state.sft_cursor_semantics="
            f"{SELFCOND_SFT_CURSOR_SEMANTICS!r}; checkpoint records "
            f"{cursor_semantics!r}. Only the buffered cursor is safe to resume; "
            "start a fresh run from the SFT base checkpoint."
        )
    sft_consumed = loop_state.get("sft_consumed")
    if not isinstance(sft_consumed, int) or isinstance(sft_consumed, bool) or sft_consumed < 0:
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume requires a non-negative integer "
            f"loop_state.sft_consumed; checkpoint records {sft_consumed!r}. "
            "Repair the checkpoint metadata or start a fresh run."
        )
    # A non-zero cursor without the buffered envelope cannot be resumed
    # exactly: the generator rejects it rather than silently replaying or
    # skipping the conversations that were mid-buffer at save time.
    dataloader_state = loop_state.get("sft_dataloader_state")
    if sft_consumed != 0 and not isinstance(dataloader_state, Mapping):
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume requires "
            "loop_state.sft_dataloader_state (the per-rank buffered cursor "
            f"envelope) alongside sft_consumed={sft_consumed}; checkpoint "
            f"records {dataloader_state!r}. Start a fresh run from the SFT "
            "base checkpoint."
        )
    return sft_consumed, (dict(dataloader_state) if isinstance(dataloader_state, Mapping) else None)


def assert_resume_selfcond_data_matches(
    resume_meta: Mapping[str, Any],
    selfcond_data: Mapping[str, Any],
) -> None:
    """Require an exact SFT artifact/scalar identity match on resume."""
    recorded = resume_meta.get("selfcond_data")
    if not isinstance(recorded, Mapping):
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume checkpoint has no well-formed "
            f"selfcond_data identity block (got {type(recorded).__name__}). Every "
            "SFT-mode checkpoint is source-aware; use its original artifacts or "
            "start a fresh run from the SFT base checkpoint."
        )
    recorded_dict = dict(recorded)
    current_dict = dict(selfcond_data)
    if recorded_dict == current_dict:
        return
    keys = sorted(recorded_dict.keys() | current_dict.keys())
    mismatch_lines: list[str] = []
    for key in keys:
        recorded_has_key = key in recorded_dict
        current_has_key = key in current_dict
        if recorded_has_key and current_has_key and recorded_dict[key] == current_dict[key]:
            continue
        recorded_value = recorded_dict[key] if recorded_has_key else "<absent>"
        current_value = current_dict[key] if current_has_key else "<absent>"
        mismatch_lines.append(f"  {key}: checkpoint={recorded_value!r} current={current_value!r}")
    rendered = "\n".join(mismatch_lines)
    raise SelfCondTrainingError(
        "SFT-mode self-conditioning resume selfcond_data mismatch — the data "
        "cursor and policy are tied to the exact recorded dataset, directories, "
        "scalars, and tokenizer checksum:\n"
        f"{rendered}\n"
        "Restore the recorded SFT artifacts/settings or start a fresh run from the SFT base."
    )


def assert_resume_sft_artifact_coverage(
    resume_meta: Mapping[str, Any],
    current_coverage: Mapping[str, Any] | None,
) -> None:
    """Require exact generator-published SFT artifact coverage on resume."""
    loop_state = resume_meta.get("loop_state")
    recorded = loop_state.get("sft_artifact_coverage") if isinstance(loop_state, Mapping) else None
    if not isinstance(recorded, Mapping):
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume checkpoint has no well-formed "
            "loop_state.sft_artifact_coverage mapping. There is no legacy "
            "tolerance for SFT-mode checkpoints; start a fresh run if this "
            "source identity cannot be recovered."
        )
    if not isinstance(current_coverage, Mapping):
        raise SelfCondTrainingError(
            "The current SFT generator did not publish a well-formed "
            "sft_artifact_coverage mapping. Refusing to resume an unverified "
            "cursor; check the parquet and feature artifacts."
        )
    recorded_dict = dict(recorded)
    current_dict = dict(current_coverage)
    if recorded_dict != current_dict:
        raise SelfCondTrainingError(
            "SFT-mode self-conditioning resume sft_artifact_coverage mismatch: "
            f"checkpoint={recorded_dict!r}, current={current_dict!r}. The SFT "
            "cursor is tied to this exact artifact family; restore the recorded "
            "parquet/features or start a fresh run."
        )


# Top-level identity blocks the SFT base records that a self-conditioning
# checkpoint must RE-PUBLISH verbatim: every downstream consumer reads them from
# the checkpoint it was handed, never from that checkpoint's base.
#   * ``steering/eval/axbench_upstream/methods.py``
#     ``resolve_raw_feature_delta`` / ``resolve_raw_feature_selection`` fall back
#     to ``pretrain_conditioning`` when ``user_config`` does not carry the field.
#   * ``steering/conditional/feature_clusters.py``
#     ``resolve_feature_clusters_from_checkpoint_meta`` requires BOTH cluster
#     blocks and rejects a checkpoint carrying only one.
#   * ``resolve_checkpoint_feature_contract`` / ``effective_feature_vocab`` read
#     ``feature_mask_canonical`` and ``feature_vocab`` (AxBench validates the
#     recorded vocabulary against the derived one, so an absent block is a hard
#     mismatch rather than a silent identity).
#   * the SFT/self-conditioning resume guards and ``assert_sft_shaped_base`` read
#     ``dataset``, ``sft_loss_mask``, ``missing_percentile_policy`` and
#     ``feature_prediction_width``.
# ``step`` is deliberately NOT inherited: it is the self-conditioning step, and
# ``build_selfcond_checkpoint_meta`` writes it from ``loop_state`` instead so
# ``run_methods._target_checkpoint_identity`` can locate ``model_{step:06d}.pt``.
INHERITED_BASE_META_KEYS: Final[tuple[str, ...]] = (
    "dataset",
    "sft_loss_mask",
    MISSING_PERCENTILE_POLICY_META_KEY,
    PRETRAIN_CONDITIONING_META_KEY,
    "conditioning_cooldown",
    "feature_mask_canonical",
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_VOCAB_META_KEY,
    _FEATURE_PREDICTION_WIDTH_META_KEY,
)


def build_selfcond_checkpoint_meta(
    *,
    model: EncoderDecoderGPT,
    base_meta: Mapping[str, Any],
    identity: Mapping[str, Any],
    cfg: SelfCondTrainConfig,
    dataloader_state_dict: dict | None,
    loop_state: Mapping[str, Any],
    base_checkpoint_dir: str,
    data_source: SelfCondDataSource = SelfCondDataSource.PRETRAINING,
    selfcond_data: dict[str, Any] | None = None,
    turn_self_cond_segment_size_rustbpe: int | None = None,
) -> dict[str, Any]:
    """Build the R15 chaining metadata for one self-conditioning checkpoint."""
    original_fp = _resolve_fp_config_from_meta(base_meta)
    if original_fp is None:
        raise SelfCondTrainingError("cannot save self-conditioning metadata without FP config")
    effective_fp = replace(original_fp, loss_mode=cfg.pool_mode)
    user_config = dict(_user_config(base_meta))
    user_config.update(identity)
    user_config["feature_prediction"] = _serialize_fp_config(effective_fp)
    try:
        base_segment_mode = resolve_checkpoint_sft_segment_mode(
            base_meta,
            checkpoint_dir=base_checkpoint_dir,
        )
    except ValueError as error:
        raise SelfCondTrainingError(str(error)) from error
    selfcond_meta: dict[str, Any] = {
        "action_ranking_policy": SELFCOND_ACTION_RANKING_POLICY,
        "train_scope": cfg.train_scope.value,
        "ste_form": cfg.ste_form.value,
        "pool_mode": cfg.pool_mode.value,
        "positive_counts": _serialize_counts(cfg.positive_counts),
        "negative_counts": _serialize_counts(cfg.negative_counts),
        "seed": cfg.seed,
        "original_head_loss_mode": original_fp.loss_mode.value,
        "data_source": data_source.value,
    }
    meta: dict[str, Any] = {
        # The self-conditioning step, NOT the base's: consumers locate the
        # weights as ``model_{step:06d}.pt`` beside this metadata file.
        "step": int(loop_state["step"]),
        "model_config": asdict(model.dec_config),
        "encoder_config": asdict(model.enc_config),
        "xattn_config": asdict(model.xattn_config),
        "num_buckets": model.encoder.config.num_buckets,
        "feature_prediction_config": _serialize_fp_config(effective_fp),
        "user_config": user_config,
        "base_checkpoint_dir": base_checkpoint_dir,
        "selfcond": selfcond_meta,
    }
    for inherited_key in INHERITED_BASE_META_KEYS:
        if inherited_key in base_meta:
            meta[inherited_key] = deepcopy(base_meta[inherited_key])
    if (
        base_segment_mode is SegmentMode.turn_delimited_fixed_tokens
        or SFT_SEGMENT_MODE_META_KEY in base_meta
    ):
        meta[SFT_SEGMENT_MODE_META_KEY] = base_segment_mode.value
    if turn_self_cond_segment_size_rustbpe is not None:
        meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] = turn_self_cond_segment_size_rustbpe
    if ONLINE_SAE_LINEAGE_META_KEY in base_meta:
        meta[ONLINE_SAE_LINEAGE_META_KEY] = deepcopy(base_meta[ONLINE_SAE_LINEAGE_META_KEY])
    if dataloader_state_dict is not None:
        meta["dataloader_state_dict"] = dataloader_state_dict
    meta["loop_state"] = dict(loop_state)
    if selfcond_data is not None:
        meta["selfcond_data"] = dict(selfcond_data)
    # Re-validates the CONSTRUCTED meta only; main() owns the base-metadata
    # gate (it parses the same base lineage before any model or data activity).
    try:
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir=base_checkpoint_dir,
        )
    except ValueError as error:
        raise SelfCondTrainingError(str(error)) from error
    return meta


def _create_wandb_run(run_name: str, master_process: bool, config: dict[str, Any]) -> Any:
    if run_name == "dummy" or not master_process:
        return DummyWandb()
    return wandb.init(project="nanochat-encdec-selfcond", name=run_name, config=config)


def _lr_multiplier(step: int, num_iterations: int) -> float:
    progress = step / max(num_iterations, 1)
    return 1.0 if progress < 0.8 else max(0.0, 1.0 - (progress - 0.8) / 0.2)


@dataclass(frozen=True)
class SelfCondTrainStepResult:
    """Detached outputs from one complete optimizer step."""

    loss: float
    metrics: SelfCondStepMetrics
    lrm: float


@beartype
def run_selfcond_train_step(
    model: EncoderDecoderGPT,
    optimizer: t.optim.Optimizer,
    batches: Iterable[ConditioningBatch],
    cfg: SelfCondTrainConfig,
    *,
    global_step: int,
    ddp_rank: int,
    num_iterations: PositiveInt,
    grad_clip_max_norm: PositiveFloat | None,
    static_max_seg_size: int | None,
    grad_accum_steps: PositiveIntLike = 1,
    full_weight_decay: float | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
) -> SelfCondTrainStepResult:
    """Execute one production rollout/backward/update step.

    ``@beartype`` owns the unary iteration, accumulation, and clipping domains;
    the body retains optimizer and batch-sequence relationships.
    """
    with timed_region("selfcond/zero_grad_start"):
        model.zero_grad(set_to_none=True)
    action_rng = t.Generator().manual_seed(selfcond_step_seed(cfg.seed, ddp_rank, global_step))
    batch_iter = iter(batches)
    metrics: SelfCondStepMetrics | None = None
    step_loss = 0.0
    for _micro_step in range(grad_accum_steps):
        with timing_tags(micro_step=_micro_step):
            with timed_region("selfcond/data_loading"):
                try:
                    batch = next(batch_iter)
                except StopIteration as error:
                    raise SelfCondTrainingError(
                        "training-step batch iterable ended before grad_accum_steps batches"
                    ) from error
            # AUTOCAST_OFFPATTERN: pretraining convention uses explicit module
            # casts; rollout, pass-1, and pass-2 intentionally share that exact
            # dtype path.
            with timed_region("selfcond/rollout"):
                if consolidation is None:
                    result = rollout(
                        model,
                        batch,
                        cfg,
                        action_rng,
                        static_max_seg_size,
                    )
                else:
                    result = rollout(
                        model,
                        batch,
                        cfg,
                        action_rng,
                        static_max_seg_size,
                        consolidation=consolidation,
                    )
            with timed_region("selfcond/loss_forward"):
                loss, metrics = selfcond_loss(
                    model,
                    batch,
                    result,
                    cfg,
                    static_max_seg_size,
                )
            with timed_region("selfcond/backward"):
                (loss / grad_accum_steps).backward()
            with timed_region("selfcond/materialize_head_zero_grads"):
                materialize_head_zero_grads(model)
            with timed_region("selfcond/loss_scalar"):
                step_loss += float(loss.detach().item()) / grad_accum_steps

    if grad_clip_max_norm is not None:
        with timed_region("selfcond/grad_clip"):
            t.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
    with timed_region("selfcond/optimizer_step"):
        lrm = _lr_multiplier(global_step, num_iterations)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if (
                cfg.train_scope is SelfCondTrainScope.FULL
                and group.get("kind") == OptimizerKind.MUON
            ):
                group["momentum"] = 0.85 + 0.10 * min(global_step / 300, 1.0)
                if full_weight_decay is not None:
                    group["weight_decay"] = full_weight_decay * (1.0 - global_step / num_iterations)
        optimizer.step()
    with timed_region("selfcond/zero_grad_end"):
        model.zero_grad(set_to_none=True)
    assert metrics is not None
    return SelfCondTrainStepResult(loss=step_loss, metrics=metrics, lrm=lrm)


@beartype
def _make_precomputed_loader(
    *,
    packed_dir: Path,
    selection_dir: Path | None,
    batch_size: PositiveInt,
    pipeline: FeaturePipelineConfig,
    device: str,
    pad_f: int | None,
    ddp_rank: int,
    ddp_world_size: int,
    resume_state_dict: dict | None,
    prefetch_batches: NonNegativeIntLike,
):
    inner = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=batch_size,
        feature_pipeline=pipeline,
        device=device,
        label_dropout=0.0,
        selection_dir=selection_dir,
        effective_buckets=pipeline.effective_buckets,
        pad_F=pad_f,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
    )
    if prefetch_batches <= 0:
        # Validation loaders MUST stay unwrapped: the prefetch producer is a
        # daemon thread over an infinite generator, and an eval that consumes
        # exactly eval_steps batches then drops the loader would leak the
        # thread plus its queued device batch every eval cycle (review
        # finding; pretraining's _build_validation_loader is also bare).
        return inner
    return _prefetch_wrapper(inner, prefetch_batches, use_cuda=device.startswith("cuda"))


class _TrainLoaderLookahead:
    """Keep the next unconsumed batch available for exact checkpoint state.

    ``precomputed_data_loader`` state describes the batch it accompanies.  A
    checkpoint taken after training on that batch must therefore persist the
    following batch's state, not the last-consumed state.
    """

    def __init__(self, loader: Iterable[Any]) -> None:
        self._loader = iter(loader)
        self._next_batch = EncDecTrainBatch.from_tuple(next(self._loader))

    @property
    def next_state_dict(self) -> dict:
        return self._next_batch.state_dict

    def pop(self) -> EncDecTrainBatch:
        current = self._next_batch
        self._next_batch = EncDecTrainBatch.from_tuple(next(self._loader))
        return current


@t.no_grad()
def evaluate_selfcond_bpb(
    model: EncoderDecoderGPT,
    val_loader: Any,
    eval_steps: int,
    token_bytes: t.Tensor,
    cfg: SelfCondTrainConfig,
    *,
    ddp_rank: int,
    static_max_seg_size: int | None,
) -> float:
    """Evaluate rollout-conditioned CE as bits per byte with a fixed seed."""
    device = model.get_device()
    total_nats = t.tensor(0.0, dtype=t.float32, device=device)
    total_bytes = t.tensor(0, dtype=t.int64, device=device)
    rng = t.Generator().manual_seed(
        selfcond_step_seed(cfg.seed, ddp_rank, VAL_ROLLOUT_STEP_SENTINEL)
    )
    for _ in range(eval_steps):
        batch = EncDecTrainBatch.from_tuple(next(val_loader))
        result = rollout(model, batch, cfg, rng, static_max_seg_size)
        loss = model(
            batch.inputs,
            batch.targets,
            result.conditioning.fids,
            result.conditioning.mags,
            result.conditioning.mask,
            batch.seg_map,
            loss_reduction="none",
            static_max_seg_size=static_max_seg_size,
        ).reshape(-1)
        # THE shared accumulation contract — same helper val/bpb uses, so the
        # two metrics stay comparable on one plot (review finding).
        batch_nats, batch_bytes = accumulate_bpb_nats_bytes(
            loss, batch.targets.reshape(-1), token_bytes
        )
        total_nats += batch_nats
        total_bytes += batch_bytes
    return reduce_bpb(total_nats, total_bytes)


@t.no_grad()
def evaluate_selfcond_bpb_sft(
    model: EncoderDecoderGPT,
    val_loader: Any,
    eval_steps: int,
    token_bytes: t.Tensor,
    cfg: SelfCondTrainConfig,
    *,
    ddp_rank: int,
    device: str,
    consolidation: FeatureClusterConsolidation | None = None,
) -> float:
    """Evaluate rollout-conditioned BPB over one bounded SFT validation pass."""
    total_nats = t.tensor(0.0, dtype=t.float32, device=device)
    total_bytes = t.tensor(0, dtype=t.int64, device=device)
    rng = t.Generator().manual_seed(
        selfcond_step_seed(cfg.seed, ddp_rank, VAL_ROLLOUT_STEP_SENTINEL)
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ddp = world_size > 1
    batch_iter = iter(val_loader)
    for _ in range(eval_steps):
        if sync_last_step(val_loader.state.exhausted, device, ddp):
            break
        with rank_synchronized_stage(
            "selfcond SFT validation batch construction",
            device=device,
        ):
            batch = SelfCondBatch.from_sft(next(batch_iter))
        if consolidation is None:
            result = rollout(model, batch, cfg, rng, None)
        else:
            result = rollout(
                model,
                batch,
                cfg,
                rng,
                None,
                consolidation=consolidation,
            )
        loss = model(
            batch.inputs,
            batch.targets,
            result.conditioning.fids,
            result.conditioning.mags,
            result.conditioning.mask,
            batch.seg_map,
            loss_reduction="none",
        ).reshape(-1)
        batch_nats, batch_bytes = accumulate_bpb_nats_bytes(
            loss,
            batch.targets.reshape(-1),
            token_bytes,
        )
        total_nats += batch_nats
        total_bytes += batch_bytes
    return reduce_bpb(total_nats, total_bytes)


def run_sft_validation_cycle(
    *,
    model: EncoderDecoderGPT,
    sources: SelfCondSftSources,
    sft_data: SelfCondSftData,
    val_task: Any,
    tokenizer: Any,
    cfg_eval: SelfCondSftEvalConfig,
    selfcond_cfg: SelfCondTrainConfig,
    ddp_rank: int,
    ddp_world_size: int,
    device: str,
    token_bytes: t.Tensor,
) -> tuple[float, float]:
    """Evaluate plain and rollout BPB on identical fresh-seeded SFT streams."""
    if sources.val_feature_dir is None or sources.val_parquet_dir is None:
        raise SelfCondTrainingError(
            "SFT validation requires resolved val_feature_dir and val_parquet_dir. "
            "Enable validation only after providing the matching SFT VAL artifacts."
        )
    negative_bundle = sft_data.pipeline.negative_bundle
    if negative_bundle is None:
        raise SelfCondTrainingError(
            "C4: SFT validation requires the base checkpoint's negative_selection; "
            "the train and validation conditioning policies must be identical."
        )

    data_seed = selfcond_step_seed(
        cfg_eval.seed,
        ddp_rank,
        VAL_DATA_SEED_SENTINEL,
    )
    sampling1, negative1 = clone_val_bundles(
        sft_data.sampling_bundle,
        negative_bundle,
        seed=data_seed,
    )
    sampling2, negative2 = clone_val_bundles(
        sft_data.sampling_bundle,
        negative_bundle,
        seed=data_seed,
    )

    def build_loader(
        sampling_bundle: SamplingBundle | None,
        eval_negative_bundle: NegativeBundle,
    ):
        return _build_sft_val_loader(
            val_task=val_task,
            tokenizer=tokenizer,
            B=cfg_eval.B,
            T=cfg_eval.T,
            device=device,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            feature_dir=sources.val_feature_dir,
            expected_feature_dir=sources.val_feature_dir,
            parquet_dir=sources.val_parquet_dir,
            threshold_cfg=sft_data.threshold_cfg,
            boundaries=sft_data.boundaries,
            precomputed_thresholds=sft_data.precomputed_thresholds,
            sampling_bundle=sampling_bundle,
            loss_mask_mode=sources.loss_mask_mode,
            min_loss_tokens=sources.min_loss_tokens,
            # No n_sae_features / fp table: FP targets stay OFF in validation
            # exactly as in the SFT-mode train generator — the two must
            # condition identically (review finding: the asymmetric arg was
            # inert but unexplained).
            fp_percentile_table=None,
            cluster_threshold_percentile_table=(sft_data.cluster_threshold_percentile_table),
            negative_bundle=eval_negative_bundle,
            feature_mask=sft_data.feature_mask,
            consolidation=sft_data.consolidation,
            missing_percentile=sft_data.missing_percentile,
            feature_source_bos_char_len=sources.feature_source_bos_char_len,
            alignment_probe_samples=sources.alignment_probe_samples,
            mixture_name=sources.dataset.value,
            return_seg_doc_id=True,
        )

    plain_loader = build_loader(sampling1, negative1)
    selfcond_loader = build_loader(sampling2, negative2)
    val_bpb = evaluate_encdec_bpb(
        model,
        plain_loader,
        cfg_eval.eval_steps,
        token_bytes,
    )
    val_bpb_selfcond = evaluate_selfcond_bpb_sft(
        model,
        selfcond_loader,
        cfg_eval.eval_steps,
        token_bytes,
        selfcond_cfg,
        ddp_rank=ddp_rank,
        device=device,
        consolidation=sft_data.consolidation,
    )
    return val_bpb, val_bpb_selfcond


def _architecture_from_meta(
    meta: Mapping[str, Any],
) -> tuple[GPTConfig, EncoderConfig, CrossAttentionConfig]:
    dec = GPTConfig(**dict(cast(Mapping[str, Any], meta["model_config"])))
    user = _user_config(meta)
    if "encoder" in user:
        enc = EncoderConfig(**dict(cast(Mapping[str, Any], user["encoder"])))
        xattn = CrossAttentionConfig(**dict(cast(Mapping[str, Any], user["cross_attention"])))
    elif "encoder_config" in meta:
        enc = EncoderConfig(**dict(cast(Mapping[str, Any], meta["encoder_config"])))
        xattn = CrossAttentionConfig(**dict(cast(Mapping[str, Any], meta["xattn_config"])))
    else:
        raise SelfCondTrainingError(
            "checkpoint metadata lacks encoder/cross_attention architecture"
        )
    return dec, enc, xattn


def main() -> None:
    """Load a base checkpoint and execute self-conditioning post-training."""
    startup_trace = _SelfCondStartupTrace()
    try:
        with _startup_phase(startup_trace, "config_and_hydra_resolution"):
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
            config_name, overrides = parse_config_name_and_overrides(
                sys.argv[1:], default_config_name="conditional_encdec_selfcond"
            )
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides)
            raw_config = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(raw_config, dict)
            user_cli_config = cast(dict[str, Any], raw_config)
            reject_removed_feature_prediction_keys(user_cli_config)
            explicit_keys = explicit_override_roots(overrides)
            profile_settings = _resolve_selfcond_profile_settings(user_cli_config)

        # Full-scope Muon groups span more unique shapes than Dynamo's default
        # cache permits; match the encoder-decoder pretraining loop's bound.
        with _startup_phase(startup_trace, "runtime_overrides"):
            t._dynamo.config.cache_size_limit = 64
            if cfg.attn_impl != FlashAttnImpl.AUTO:
                fa_module._override_impl = FlashAttnImpl(cfg.attn_impl)
                logger.info(f"Attention implementation override: {cfg.attn_impl}")

        with _startup_phase(startup_trace, "distributed_device_initialization"):
            device_type = (
                autodetect_device_type() if cfg.device_type == "" else str(cfg.device_type)
            )
            ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    except BaseException as error:
        startup_trace.failed(error)
        startup_trace.ensure_failure_visible()
        raise
    master_process = ddp_rank == 0
    startup_trace.set_master_process(master_process)
    profile_session: _SelfCondProfileSession | None = None

    def profile_markers_enabled() -> bool:
        return profile_session is not None and profile_session.markers_enabled

    try:
        with _startup_phase(startup_trace, "checkpoint_resolution_and_metadata_load"):
            base_checkpoint_dir = resolve_base_checkpoint_dir(user_cli_config, get_base_dir())
            if cfg.base_checkpoint_dir is not None:
                logger.info(f"Using explicit base checkpoint dir: {base_checkpoint_dir}")
            assert_base_dir_has_checkpoint(base_checkpoint_dir)
            base_model_data, _, base_meta = load_checkpoint(
                base_checkpoint_dir,
                cfg.model_step,
                device,
                load_optimizer=False,
            )
        with _startup_phase(startup_trace, "checkpoint_identity_seed_resolution"):
            identity = resolve_identity_config(user_cli_config, base_meta, explicit_keys)
            base_conditioning = read_checkpoint_conditioning(base_meta)
        if (
            identity.get("negative_selection") is None
            and base_conditioning.negative_selection is not None
        ):
            identity["negative_selection"] = dict(base_conditioning.negative_selection)
        # C5 must gate the RESOLVED identity, not just the checkpoint meta: an
        # explicit feature_selection override carrying a gradient block would
        # otherwise bypass validation and drive gradient-ranked discovery that
        # v1 forbids (review finding).
        resolved_feature_selection = identity.get("feature_selection")
        if (
            isinstance(resolved_feature_selection, dict)
            and resolved_feature_selection.get("gradient") is not None
        ):
            raise SelfCondTrainingError(
                "C5: the RESOLVED feature_selection carries a gradient-attribution "
                "block (via an explicit override); gradient-ranked conditioning is "
                "unsupported by self-conditioning post-training v1."
            )
        num_buckets = resolve_checkpoint_num_buckets(base_meta)
        identity["num_buckets"] = num_buckets
        n_sae_features, feature_vocab = feature_vocab_from_state_dict(base_model_data)
        dec_config, enc_config, xattn_config = _architecture_from_meta(base_meta)
        fp_config = _resolve_fp_config_from_meta(base_meta)
        selfcond_block = cast(dict[str, Any], _plain(cfg.selfcond))
        data_source = resolve_selfcond_data_source(selfcond_block)
        try:
            base_sft_segment_mode = resolve_checkpoint_sft_segment_mode(
                base_meta,
                checkpoint_dir=base_checkpoint_dir,
            )
        except ValueError as error:
            raise SelfCondTrainingError(str(error)) from error
        assert_turn_cadence_is_not_configured(
            selfcond_block,
            data_source=data_source,
            segment_mode=base_sft_segment_mode,
        )
        # Measured from the TRAIN artifacts once the SFT store and task mixture
        # are loaded; stays None for fixed-token and pretraining sources.
        turn_self_cond_segment_size_rustbpe: int | None = None
        base_online_sae_lineage = _resolve_selfcond_online_sae_lineage(
            base_meta,
            segment_mode=base_sft_segment_mode,
            checkpoint_dir=base_checkpoint_dir,
            purpose="self-conditioning base checkpoint",
        )
        if (
            data_source is SelfCondDataSource.PRETRAINING
            and feature_vocab is not None
            and fp_config is not None
        ):
            raise NotImplementedError(
                "conditional_encdec_selfcond with data_source=pretraining does not "
                "support checkpoints with both a compact FeatureVocab and a "
                "feature-prediction head. The packed-row path has no compact target "
                "contract; use an uncompacted checkpoint or data_source=sft with an "
                "exact compact source contract."
            )
        if data_source is SelfCondDataSource.SFT:
            if cfg.get("train_prefetch_batches") is not None:
                raise SelfCondTrainingError(
                    "selfcond.data_source=sft requires train_prefetch_batches=null: "
                    "prefetch would decouple consumed conversations from trained "
                    "batches and violate the buffered_v2 cursor contract."
                )
        selfcond_cfg = resolve_selfcond_train_config(
            base_meta,
            identity,
            selfcond_block,
            n_sae_features,
        )
        output_dir = str(
            Path(base_checkpoint_dir)
            / (
                f"selfcond_{SELFCOND_ACTION_RANKING_POLICY}_"
                f"{selfcond_cfg.train_scope.value}_{selfcond_cfg.ste_form.value}_checkpoints"
            )
        )
        if data_source is SelfCondDataSource.PRETRAINING:
            try:
                assert_sft_shaped_base(base_meta, base_checkpoint_dir)
            except SelfCondTrainingError:
                pass
            else:
                raise SelfCondTrainingError(
                    "this base is an SFT checkpoint; set selfcond.data_source=sft "
                    "so training consumes its recorded "
                    "SFT conversations rather than attempting packed-row discovery."
                )

        resume_step = resolve_resume_step(int(cfg.resume_from_step))
        resume_model_data: dict | None = None
        resume_optimizer_data: dict | None = None
        resume_meta: dict | None = None
        if resume_step is not None:
            with _startup_phase(startup_trace, "resume_checkpoint_load"):
                resume_model_data, resume_optimizer_data, resume_meta = load_checkpoint(
                    output_dir,
                    resume_step,
                    device,
                    load_optimizer=True,
                    rank=ddp_rank,
                )
                assert_resume_selfcond_matches(
                    resume_meta,
                    selfcond_cfg,
                    data_source=data_source,
                )
                try:
                    assert_resume_sft_segment_mode_matches(
                        base_sft_segment_mode,
                        resume_meta,
                        checkpoint_dir=output_dir,
                    )
                except ValueError as error:
                    raise SelfCondTrainingError(str(error)) from error
                assert_resume_online_sae_lineage_matches(
                    base_online_sae_lineage,
                    _resolve_selfcond_online_sae_lineage(
                        resume_meta,
                        segment_mode=base_sft_segment_mode,
                        checkpoint_dir=output_dir,
                        purpose="self-conditioning resume checkpoint",
                    ),
                    segment_mode=base_sft_segment_mode,
                )
        loop = resume_meta.get("loop_state", {}) if resume_meta is not None else {}
        start_step = int(loop.get("step", resume_step if resume_step is not None else 0))
        assert_training_window(start_step, int(cfg.num_iterations))
        _validate_selfcond_profile_window(
            profile_settings,
            start_step=start_step,
            num_iterations=int(cfg.num_iterations),
        )

        with _startup_phase(startup_trace, "model_construction_and_checkpoint_restore"):
            with t.device("meta"):
                model = EncoderDecoderGPT(
                    dec_config,
                    enc_config,
                    xattn_config,
                    n_sae_features=n_sae_features,
                    num_buckets=num_buckets,
                    feature_pred_config=fp_config,
                    feature_vocab=feature_vocab,
                )
            model.to_empty(device=device)
            model.init_buffers()
            if resume_meta is not None:
                validate_feature_prediction_resume(resume_meta, model.feature_pred_config)
            model.load_state_dict(
                resume_model_data if resume_model_data is not None else base_model_data,
                strict=True,
                assign=True,
            )
            # Release the superseded/copied checkpoint state dicts — keeping them
            # referenced for the whole run pins a full model-size of device memory
            # (review finding). assign=True adopted the loaded tensors we keep.
            del base_model_data
            resume_model_data = None
            model.assert_vocab_consistent()
            model.feature_mask = None
            validate_selfcond_startup(
                model,
                base_meta,
                selfcond_cfg,
                n_sae_features,
                user_cli_config,
                data_source=data_source,
            )

        tokenizer: Any
        token_bytes: t.Tensor
        meta_user: dict[str, Any]
        feature_pipeline: FeaturePipelineConfig | None = None
        discovery: Any = None
        val_packed_dir: Path | None = None
        val_selection_dir: Path | None = None
        train_lookahead: _TrainLoaderLookahead | None = None
        sources: SelfCondSftSources | None = None
        # Held for the run: the weak feature-store cache re-indexes unless these
        # startup-loaded stores remain strongly reachable.
        sft_feature_stores: dict[SFTSplit, Any] = {}
        sft_data: SelfCondSftData | None = None
        val_task: Any = None
        sft_state: SFTState | None = None
        train_gen: Iterator[Any] | None = None
        pending_sft_batch: tuple | list | None = None
        selfcond_data_dict: dict[str, Any] | None = None
        committed_consumed = 0

        if data_source is SelfCondDataSource.PRETRAINING:
            feature_dir_raw = identity.get("feature_dir")
            if feature_dir_raw is None:
                raise SelfCondTrainingError(
                    "checkpoint metadata lacks feature_dir; pass +feature_dir=<path> explicitly"
                )
            feature_dir = Path(str(feature_dir_raw))
            stats_path = find_statistics_path(feature_dir)
            if stats_path is None:
                raise SelfCondTrainingError(
                    f"feature_statistics.safetensors not found under {feature_dir}"
                )
            with _startup_phase(
                startup_trace,
                "pretraining_statistics_and_percentile_metadata",
                feature_dir=str(feature_dir),
            ):
                statistics = load_statistics(stats_path)
            assert_checkpoint_feature_universe(
                n_sae_features,
                statistics.metadata.n_sae_features,
            )
            tokenizer = get_tokenizer()
            token_bytes = get_token_bytes(device=device)
            percentile_checksum = load_percentile_checksum(stats_path)
            tokenizer_checksum = load_tokenizer_checksum(stats_path)
            if tokenizer_checksum is None:
                tokenizer_checksum = compute_tokenizer_checksum(tokenizer)

            runtime_data_cfg = dict(identity)
            runtime_data_cfg["feature_clusters"] = _user_config(base_meta).get("feature_clusters")
            runtime_data_cfg["feature_mask"] = _user_config(base_meta).get("feature_mask")
            feature_delta_raw = base_conditioning.feature_delta
            with _startup_phase(
                startup_trace,
                "pretraining_conditioning_pipeline_reconstruction",
            ):
                feature_pipeline = build_feature_pipeline_from_cfg_block(
                    runtime_data_cfg,
                    per_feature_stats=statistics.per_feature_stats,
                    n_sae_features=n_sae_features,
                    num_buckets=num_buckets,
                    seed=int(identity.get("seed", 42)),
                    ddp_rank=ddp_rank,
                    feature_delta=parse_feature_delta(feature_delta_raw),
                    feature_mask=None,
                )
            meta_user = _user_config(base_meta)
            original_val_feature_dir = meta_user.get("original_val_feature_dir")
            with _startup_phase(
                startup_trace,
                "pretraining_packed_selection_discovery",
                feature_dir=str(feature_dir),
            ):
                discovery = discover_packed_selection(
                    feature_dir=feature_dir,
                    original_val_feature_dir=(
                        Path(str(original_val_feature_dir))
                        if original_val_feature_dir is not None
                        else None
                    ),
                    feature_selection=cast(dict[str, Any], identity["feature_selection"]),
                    negative_selection=cast(
                        dict[str, Any] | None,
                        identity.get("negative_selection"),
                    ),
                    feature_delta=cast(dict[str, Any] | None, feature_delta_raw),
                    num_buckets=num_buckets,
                    seed=int(identity.get("seed", 42)),
                    feature_mask=None,
                    feature_clusters=cast(
                        dict[str, Any] | None,
                        meta_user.get("feature_clusters"),
                    ),
                    n_sae_features=n_sae_features,
                    use_precomputed_batches=bool(meta_user.get("use_precomputed_batches", True)),
                    percentile_checksum=percentile_checksum,
                    tokenizer_checksum=tokenizer_checksum,
                    context_len=dec_config.sequence_len,
                    segment_size=int(identity["segment_size"]),
                    max_train_shards=meta_user.get("max_train_shards"),
                    compile_static=False,
                    fp8=False,
                    eval_every=int(cfg.eval_every),
                    original_val_data_dir=meta_user.get("original_val_data_dir"),
                    boundary_fingerprint=None,
                    selection_splice_batch_size=meta_user.get("selection_splice_batch_size"),
                )
            if discovery.train_packed_dir is None:
                raise SelfCondTrainingError(
                    "C6: matching precomputed TRAIN packed rows are unavailable for this checkpoint"
                )
            val_packed_dir, val_selection_dir = resolve_val_dirs(
                discovery,
                int(cfg.eval_every),
            )
            static_max_seg_size = discovery.max_seg_size
        else:
            tokenizer = get_tokenizer()
            token_bytes = get_token_bytes(device=device)
            meta_user = _user_config(base_meta)
            with _startup_phase(startup_trace, "sft_source_resolution"):
                assert_sft_shaped_base(base_meta, base_checkpoint_dir)
                sources = resolve_selfcond_sft_sources(
                    selfcond_block,
                    base_meta,
                    get_base_dir(),
                    base_checkpoint_dir=base_checkpoint_dir,
                    eval_every=int(cfg.eval_every),
                    segment_size=int(identity["segment_size"]),
                )
            with rank_synchronized_stage(
                "selfcond SFT feature segmentation",
                device=device,
            ):
                store_specs = [
                    (
                        SFTSplit.TRAIN,
                        sources.train_feature_dir,
                        sources.train_parquet_dir,
                    )
                ]
                if int(cfg.eval_every) > 0:
                    assert sources.val_feature_dir is not None
                    assert sources.val_parquet_dir is not None
                    store_specs.append(
                        (
                            SFTSplit.VAL,
                            sources.val_feature_dir,
                            sources.val_parquet_dir,
                        )
                    )
                for split, feature_dir, parquet_dir in store_specs:
                    store = _load_sft_features_for_startup(
                        feature_dir,
                        parquet_dir,
                        startup_trace=startup_trace,
                        source=split.value,
                    )
                    if store.segment_mode is not sources.segment_mode:
                        raise SelfCondTrainingError(
                            f"selfcond.data_source=sft {split.value} feature store "
                            f"segmentation mismatch for {feature_dir!r}: checkpoint "
                            f"records {sources.segment_mode.value!r}, loaded store "
                            f"records {store.segment_mode.value!r}."
                        )
                    with _startup_phase(
                        startup_trace,
                        "sft_source_bos_validation",
                        source=split.value,
                    ):
                        assert_sft_source_bos_char_len_matches(
                            feature_dir,
                            parquet_dir,
                            configured=sources.feature_source_bos_char_len,
                            source=split.value,
                        )
                    sft_feature_stores[split] = store
            sft_data = build_selfcond_sft_data(
                sources=sources,
                base_meta=base_meta,
                identity=identity,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                ddp_rank=ddp_rank,
                tokenizer=tokenizer,
                state_feature_vocab=feature_vocab,
                startup_trace=startup_trace,
            )
            if sources.dataset in {SFTMixture.STANDARD, SFTMixture.CAPABILITIES}:
                with _startup_phase(
                    startup_trace,
                    "sft_word_list_asset_preflight",
                    dataset=sources.dataset.value,
                ):
                    word_list_path = ensure_word_list_available(allow_network=False)
                startup_trace.event(
                    "sft_word_list_asset_ready",
                    path=str(word_list_path),
                    dataset=sources.dataset.value,
                )
            else:
                startup_trace.skipped(
                    "sft_word_list_asset_preflight",
                    dataset=sources.dataset.value,
                    reason="selected SFT mixture does not include spelling tasks",
                )
            with _startup_phase(
                startup_trace,
                "sft_mixture_construction",
                dataset=sources.dataset.value,
            ):
                train_task, val_task = build_task_mixtures(sources.dataset)
            if int(cfg.eval_every) > 0 and (val_task is None or len(val_task) == 0):
                raise SelfCondTrainingError(
                    f"selfcond.data_source=sft dataset={sources.dataset.value!r} "
                    "has no validation task mixture while eval_every > 0. "
                    "Provide a dataset with validation coverage or set eval_every=0."
                )
            if sources.segment_mode is SegmentMode.turn_delimited_fixed_tokens:
                with _startup_phase(
                    startup_trace,
                    "turn_self_cond_cadence_measurement",
                    feature_dir=sources.train_feature_dir,
                ):
                    measurement = measure_turn_self_cond_segment_size_rustbpe(
                        segments_by_document=sft_feature_stores[SFTSplit.TRAIN],
                        rendered=_turn_cadence_renderer(train_task, tokenizer),
                        tokenizer=tokenizer,
                        segment_size=int(identity["segment_size"]),
                        row_capacity=int(dec_config.sequence_len) + 1,
                        loss_mask_mode=sources.loss_mask_mode,
                        min_loss_tokens=sources.min_loss_tokens,
                        feature_source_bos_char_len=sources.feature_source_bos_char_len,
                        feature_dir=sources.train_feature_dir,
                        parquet_dir=sources.train_parquet_dir,
                    )
                turn_self_cond_segment_size_rustbpe = int(measurement.cadence)
                logger.info(
                    "Measured generated-assistant cadence "
                    f"{TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY}="
                    f"{turn_self_cond_segment_size_rustbpe} from "
                    f"{int(measurement.windows)} full assistant windows in "
                    f"{int(measurement.documents)} conversations "
                    f"(mean={float(measurement.mean):.4f}, "
                    f"min={int(measurement.minimum)}, "
                    f"max={int(measurement.maximum)})"
                )
                if resume_meta is not None:
                    try:
                        recorded_cadence = resolve_turn_self_cond_segment_size_rustbpe(
                            resume_meta,
                            checkpoint_dir=output_dir,
                        )
                    except ValueError as error:
                        raise SelfCondTrainingError(str(error)) from error
                    if recorded_cadence != turn_self_cond_segment_size_rustbpe:
                        raise SelfCondTrainingError(
                            f"{TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY} "
                            f"mismatch on resume from {output_dir}: checkpoint "
                            f"records {recorded_cadence!r}, the current TRAIN "
                            "artifacts measure "
                            f"{turn_self_cond_segment_size_rustbpe!r}. Resume "
                            "against the artifacts the checkpoint was trained "
                            "on or start a fresh run."
                        )
            resumed_consumed, resumed_dataloader_state = (
                resolve_sft_mode_resume_state(resume_meta) if resume_meta is not None else (0, None)
            )
            selfcond_data_dict = {
                "dataset": sources.dataset.value,
                "train_parquet_dir": sources.train_parquet_dir,
                "train_feature_dir": sources.train_feature_dir,
                "val_parquet_dir": sources.val_parquet_dir,
                "val_feature_dir": sources.val_feature_dir,
                "sft_loss_mask": sources.loss_mask_mode.value,
                SFT_SEGMENT_MODE_META_KEY: sources.segment_mode.value,
                "min_loss_tokens": sources.min_loss_tokens,
                "feature_source_bos_char_len": sources.feature_source_bos_char_len,
                "tokenizer_checksum": sft_data.tokenizer_checksum,
                # The stats artifact governs boundaries + percentile tables;
                # its checksum pins the conditioning distribution across
                # resumes (review finding).
                "percentile_checksum": sft_data.percentile_checksum,
                # The LIVE selection policy is data identity too: an explicit
                # feature_selection override (e.g. weighted vs
                # top_k_percentile) changes conditioning semantics without
                # touching dirs, counts, scope, or STE form (review finding).
                "feature_selection": _plain(identity.get("feature_selection")),
                "negative_selection": _plain(identity.get("negative_selection")),
                # A `lowest` pilot's metrics must never silently compare to an
                # `error` run's; the recorded policy travels with the selfcond
                # checkpoint and resume under a different one is rejected.
                "missing_percentile_policy": sources.missing_percentile_policy.value,
            }
            if sft_data.feature_mask is not None:
                mask_canonical = sft_data.feature_mask.canonical_dict()
                selfcond_data_dict.update(
                    {
                        "feature_mask_canonical_sha256": {
                            "sha256": hashlib.sha256(
                                canonical_json(mask_canonical).encode("utf-8")
                            ).hexdigest(),
                            "mode": sft_data.feature_mask.mode.value,
                            "id_count": len(sft_data.feature_mask.ids),
                        },
                        "feature_clusters_descriptor": _plain(
                            base_meta.get(FEATURE_CLUSTERS_DESCRIPTOR_KEY)
                        ),
                        "feature_vocab": _plain(base_meta[FEATURE_VOCAB_META_KEY]),
                    }
                )
            if resume_meta is not None:
                assert_resume_selfcond_data_matches(
                    resume_meta,
                    selfcond_data_dict,
                )
            sft_state = SFTState(
                consumed=resumed_consumed,
                dataloader_state=resumed_dataloader_state,
            )
            train_gen = make_sft_data_generator(
                task=train_task,
                tokenizer=tokenizer,
                B=int(cfg.device_batch_size),
                T=dec_config.sequence_len,
                device=str(device),
                sft_state=sft_state,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                feature_dir=sources.train_feature_dir,
                parquet_dir=sources.train_parquet_dir,
                threshold_cfg=sft_data.threshold_cfg,
                boundaries=sft_data.boundaries,
                precomputed_thresholds=sft_data.precomputed_thresholds,
                sampling_bundle=sft_data.sampling_bundle,
                min_loss_tokens=sources.min_loss_tokens,
                fp_percentile_table=None,
                cluster_threshold_percentile_table=(sft_data.cluster_threshold_percentile_table),
                negative_bundle=sft_data.pipeline.negative_bundle,
                feature_mask=sft_data.feature_mask,
                consolidation=sft_data.consolidation,
                missing_percentile=sft_data.missing_percentile,
                return_seg_doc_id=True,
                feature_source_bos_char_len=sources.feature_source_bos_char_len,
                startup_trace=startup_trace,
                loss_mask_mode=sources.loss_mask_mode,
                alignment_probe_samples=sources.alignment_probe_samples,
                mixture_name=sources.dataset.value,
                single_pass=False,
                expected_split=SFTSplit.TRAIN,
                profile_markers=(profile_markers_enabled if profile_settings.enabled else False),
            )
            startup_trace.event(
                "sft_initial_batch_materialization_started",
                dataset=sources.dataset.value,
                rank_synchronized=True,
            )
            with _startup_phase(
                startup_trace,
                "sft_initial_batch_materialization",
                dataset=sources.dataset.value,
            ):
                with rank_synchronized_stage(
                    "selfcond SFT initial training batch construction",
                    device=device,
                ):
                    pending_sft_batch = next(train_gen)
            startup_trace.skipped(
                "sft_validation_artifact_index",
                reason="validation generator is rebuilt lazily at evaluation time",
                val_feature_dir=sources.val_feature_dir,
                val_parquet_dir=sources.val_parquet_dir,
            )
            if sft_state.artifact_coverage is None:
                raise SelfCondTrainingError(
                    "The SFT training generator did not publish "
                    "sft_artifact_coverage after its initial batch. Refusing to "
                    "train without a cursor-bound artifact identity."
                )
            if resume_meta is not None:
                assert_resume_sft_artifact_coverage(
                    resume_meta,
                    sft_state.artifact_coverage,
                )
            committed_consumed = resumed_consumed
            static_max_seg_size = None
            logger.info(
                "Self-conditioning data_source=sft "
                f"dataset={sources.dataset.value} "
                f"train_parquet={sources.train_parquet_dir} "
                f"train_features={sources.train_feature_dir} "
                f"val_parquet={sources.val_parquet_dir} "
                f"val_features={sources.val_feature_dir}"
            )

        with _startup_phase(startup_trace, "optimizer_construction"):
            if selfcond_cfg.train_scope is SelfCondTrainScope.HEAD_ONLY:
                freeze_for_head_only(model)
                optimizer = build_head_only_optimizer(
                    model,
                    unembedding_lr=float(cfg.unembedding_lr) * float(cfg.lr_multiplier),
                    adam_betas=resolve_adam_betas(cfg),
                )
            else:
                optimizer = model.setup_optimizer(
                    unembedding_lr=float(cfg.unembedding_lr) * float(cfg.lr_multiplier),
                    embedding_lr=float(cfg.embedding_lr) * float(cfg.lr_multiplier),
                    matrix_lr=float(cfg.matrix_lr) * float(cfg.lr_multiplier),
                    scalar_lr=float(cfg.scalar_lr) * float(cfg.lr_multiplier),
                    weight_decay=float(cfg.weight_decay),
                    adam_betas=resolve_adam_betas(cfg),
                    encoder_embedding_lr=float(meta_user.get("encoder_embedding_lr", 0.2))
                    * float(cfg.lr_multiplier),
                    encoder_matrix_lr=float(meta_user.get("encoder_matrix_lr", 0.02))
                    * float(cfg.lr_multiplier),
                    xattn_matrix_lr=float(meta_user.get("xattn_matrix_lr", 0.02))
                    * float(cfg.lr_multiplier),
                )
        if resume_optimizer_data is not None:
            with _startup_phase(startup_trace, "optimizer_resume_state_load"):
                optimizer.load_state_dict(resume_optimizer_data)
                resume_optimizer_data = None  # deep-copied by load_state_dict; release

        grad_accum_steps = resolve_grad_accum_steps(
            PositiveInt.parse(int(cfg.device_batch_size)),
            PositiveInt.parse(dec_config.sequence_len),
            PositiveInt.parse(ddp_world_size),
            int(cfg.total_batch_size),
        )
        if data_source is SelfCondDataSource.PRETRAINING:
            assert discovery is not None
            assert discovery.train_packed_dir is not None
            assert feature_pipeline is not None
            dataloader_resume = select_dataloader_rank_state(
                resolve_dataloader_resume(resume_meta),
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
            )
            with _startup_phase(
                startup_trace,
                "pretraining_loader_setup",
                packed_dir=str(discovery.train_packed_dir),
            ):
                train_loader = _make_precomputed_loader(
                    packed_dir=discovery.train_packed_dir,
                    selection_dir=discovery.train_selection_dir,
                    batch_size=PositiveInt.parse(int(cfg.device_batch_size)),
                    pipeline=feature_pipeline,
                    device=str(device),
                    pad_f=discovery.pad_f,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                    resume_state_dict=dataloader_resume,
                    prefetch_batches=resolve_train_prefetch_batches(
                        cfg.get("train_prefetch_batches"), grad_accum_steps
                    ),
                )
                train_lookahead = _TrainLoaderLookahead(train_loader)

        with _startup_phase(startup_trace, "wandb_setup"):
            wandb_run = _create_wandb_run(str(cfg.run), master_process, user_cli_config)
        startup_trace.skipped(
            "compile_and_graph_warmup",
            reason="compile=false is required by self-conditioning v1",
        )
        _loaded_smooth_loss = loop.get("smooth_train_loss")
        smooth_loss = float(_loaded_smooth_loss) if _loaded_smooth_loss is not None else 0.0
        min_val_bpb = float(loop.get("min_val_bpb", float("inf")))
        min_val_bpb_selfcond = float(loop.get("min_val_bpb_selfcond", float("inf")))
        total_training_time = float(loop.get("total_training_time", 0.0))
        profile_session = _SelfCondProfileSession(
            profile_settings,
            rank=ddp_rank,
            device_type=device_type,
        )
        if profile_session.enabled:
            with rank_synchronized_stage(
                "selfcond profiling output setup",
                device=device,
            ):
                profile_session.prepare_output_dir()
            logger.info(
                "Self-conditioning profiling enabled: "
                f"timing={profile_settings.timing_enabled}, "
                f"trace_step={profile_settings.trace_step}, "
                f"trace_steps={profile_settings.trace_steps}, "
                f"output_dir={profile_settings.output_dir}, rank={ddp_rank}"
            )
        wrote_checkpoint = False
        model.train()
        first_step_elapsed: float | None = None
        steady_state_elapsed: list[float] = []

        for step in range(start_step, int(cfg.num_iterations)):
            step_start = time.perf_counter()

            if profile_session.should_start_trace(step):
                with rank_synchronized_stage(
                    "selfcond profiling trace start",
                    device=device,
                ):
                    profile_session.start_trace(step)

            def step_batches() -> Iterator[ConditioningBatch]:
                nonlocal pending_sft_batch
                for _micro_step in range(grad_accum_steps):
                    if data_source is SelfCondDataSource.PRETRAINING:
                        assert train_lookahead is not None
                        yield train_lookahead.pop()
                    else:
                        if pending_sft_batch is not None:
                            raw, pending_sft_batch = pending_sft_batch, None
                        else:
                            assert train_gen is not None
                            with rank_synchronized_stage(
                                "selfcond SFT training batch construction",
                                device=device,
                            ):
                                raw = next(train_gen)
                        yield SelfCondBatch.from_sft(raw)

            with _startup_phase(
                startup_trace if first_step_elapsed is None else None,
                "first_optimizer_step",
                step=step,
            ):
                with profile_session.recording(step):
                    with timed_region("selfcond/train_step"):
                        train_result = run_selfcond_train_step(
                            model,
                            optimizer,
                            step_batches(),
                            selfcond_cfg,
                            global_step=step,
                            ddp_rank=ddp_rank,
                            num_iterations=PositiveInt.parse(int(cfg.num_iterations)),
                            grad_clip_max_norm=(
                                PositiveFloat.parse(float(cfg.grad_clip_max_norm))
                                if cfg.grad_clip_max_norm is not None
                                else None
                            ),
                            static_max_seg_size=static_max_seg_size,
                            grad_accum_steps=grad_accum_steps,
                            full_weight_decay=float(cfg.weight_decay),
                            consolidation=(
                                sft_data.consolidation if sft_data is not None else None
                            ),
                        )
            if profile_session.should_stop_trace(step):
                with rank_synchronized_stage(
                    "selfcond profiling trace export",
                    device=device,
                ):
                    profile_session.stop_and_export_trace(step)
            if data_source is SelfCondDataSource.SFT:
                assert sft_state is not None
                committed_consumed = sft_state.consumed
            step_loss = train_result.loss
            metrics = train_result.metrics
            lrm = train_result.lrm
            elapsed = time.perf_counter() - step_start
            total_training_time += elapsed
            is_first_step = first_step_elapsed is None
            if is_first_step:
                first_step_elapsed = elapsed
                startup_trace.mark_first_step()
            else:
                steady_state_elapsed.append(elapsed)
            # Seed the EMA from the first observed loss only on a FRESH start;
            # on resume, blend into the checkpoint's loaded smooth_train_loss
            # (pretraining/SFT semantics) instead of discarding it.
            if step == start_step and _loaded_smooth_loss is None:
                smooth_loss = step_loss
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * step_loss
            if master_process:
                # Display the smoothed loss like the pretraining/SFT siblings
                # (their `loss:` field and wandb `train/loss` are the smoothed
                # series); the raw per-step loss stays available alongside it.
                logger.info(
                    f"step {step + 1:05d} | loss {smooth_loss:.6f} | "
                    f"raw {step_loss:.6f} | "
                    f"pred {metrics.frac_predicted_slots:.3f} | "
                    f"k {metrics.mean_k:.2f} | l {metrics.mean_l:.2f} | "
                    f"{elapsed * 1000:.1f} ms"
                )
                timing_payload: dict[str, float | int | str] = {
                    "timing/step_seconds": elapsed,
                }
                if is_first_step:
                    timing_payload["timing/first_step_seconds"] = elapsed
                else:
                    timing_payload["timing/steady_state_step_seconds"] = elapsed
                wandb_run.log(
                    {
                        "step": step + 1,
                        "train/loss": smooth_loss,
                        "train/raw_loss": step_loss,
                        "train/frac_predicted_slots": metrics.frac_predicted_slots,
                        "train/mean_k": metrics.mean_k,
                        "train/mean_l": metrics.mean_l,
                        "train/mean_selected_pos_score": metrics.mean_selected_pos_score,
                        "train/lrm": lrm,
                        **timing_payload,
                    },
                    step=step + 1,
                )

            completed_step = step + 1
            run_eval = int(cfg.eval_every) > 0 and (
                completed_step % int(cfg.eval_every) == 0
                or completed_step == int(cfg.num_iterations)
            )
            if run_eval:
                if data_source is SelfCondDataSource.PRETRAINING:
                    # val_packed_dir is guaranteed non-None here: resolve_val_dirs
                    # raised at startup if eval_every > 0 with no val rows, so the
                    # dead-eval state (guard silently never firing) is impossible.
                    assert val_packed_dir is not None
                    assert feature_pipeline is not None
                    eval_steps = max(
                        1,
                        int(cfg.eval_tokens)
                        // (int(cfg.device_batch_size) * dec_config.sequence_len * ddp_world_size),
                    )
                    real_val_loader = _make_precomputed_loader(
                        packed_dir=val_packed_dir,
                        selection_dir=val_selection_dir,
                        batch_size=PositiveInt.parse(int(cfg.device_batch_size)),
                        pipeline=feature_pipeline,
                        device=str(device),
                        pad_f=discovery.pad_f,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                        resume_state_dict=None,
                        prefetch_batches=VAL_EVAL_PREFETCH_BATCHES,
                    )
                    selfcond_val_loader = _make_precomputed_loader(
                        packed_dir=val_packed_dir,
                        selection_dir=val_selection_dir,
                        batch_size=PositiveInt.parse(int(cfg.device_batch_size)),
                        pipeline=feature_pipeline,
                        device=str(device),
                        pad_f=discovery.pad_f,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                        resume_state_dict=None,
                        prefetch_batches=VAL_EVAL_PREFETCH_BATCHES,
                    )
                    model.eval()
                    with profile_session.recording(step):
                        with timed_region("selfcond/validation"):
                            with disable_fp8(model):
                                real_metrics = evaluate_encdec_bpb_conditioned(
                                    model,
                                    real_val_loader,
                                    eval_steps,
                                    token_bytes,
                                    device,
                                )
                                val_bpb_selfcond = evaluate_selfcond_bpb(
                                    model,
                                    selfcond_val_loader,
                                    eval_steps,
                                    token_bytes,
                                    selfcond_cfg,
                                    ddp_rank=ddp_rank,
                                    static_max_seg_size=discovery.max_seg_size,
                                )
                    model.train()
                    val_bpb = real_metrics.bpb_conditioned
                    # `or` (not a .get default): resolve_identity_config may
                    # materialize the key with an explicit None (review finding).
                    dataset_identity = str(identity.get("dataset_phase") or "pretraining")
                else:
                    assert sources is not None
                    assert sft_data is not None
                    assert val_task is not None
                    eval_steps = max(
                        1,
                        int(cfg.eval_tokens)
                        // (int(cfg.device_batch_size) * dec_config.sequence_len * ddp_world_size),
                    )
                    model.eval()
                    with profile_session.recording(step):
                        with timed_region("selfcond/validation"):
                            with disable_fp8(model):
                                val_bpb, val_bpb_selfcond = run_sft_validation_cycle(
                                    model=model,
                                    sources=sources,
                                    sft_data=sft_data,
                                    val_task=val_task,
                                    tokenizer=tokenizer,
                                    cfg_eval=SelfCondSftEvalConfig(
                                        B=int(cfg.device_batch_size),
                                        T=dec_config.sequence_len,
                                        eval_steps=eval_steps,
                                        seed=selfcond_cfg.seed,
                                    ),
                                    selfcond_cfg=selfcond_cfg,
                                    ddp_rank=ddp_rank,
                                    ddp_world_size=ddp_world_size,
                                    device=str(device),
                                    token_bytes=token_bytes,
                                )
                    model.train()
                    dataset_identity = sources.dataset.value
                min_val_bpb = min(min_val_bpb, val_bpb)
                min_val_bpb_selfcond = min(min_val_bpb_selfcond, val_bpb_selfcond)
                if master_process:
                    logger.info(
                        f"step {completed_step:05d} | "
                        f"data_source={data_source.value} | "
                        f"dataset={dataset_identity} | "
                        f"val/bpb {val_bpb:.6f} | "
                        f"val/bpb_selfcond {val_bpb_selfcond:.6f}"
                    )
                    wandb_run.log(
                        {
                            "step": completed_step,
                            "data_source": data_source.value,
                            "dataset": dataset_identity,
                            "val/bpb": val_bpb,
                            "val/bpb_selfcond": val_bpb_selfcond,
                        },
                        step=completed_step,
                    )

            run_save = int(cfg.save_every) > 0 and (
                completed_step % int(cfg.save_every) == 0
                or completed_step == int(cfg.num_iterations)
            )
            if run_save:
                if data_source is SelfCondDataSource.PRETRAINING:
                    assert train_lookahead is not None
                    # Every rank enters run_save. The lookahead is the next
                    # unconsumed precomputed batch, matching pretraining semantics.
                    dataloader_state_envelope = build_dataloader_state_envelope(
                        train_lookahead.next_state_dict,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )
                    save_meta = build_selfcond_checkpoint_meta(
                        model=model,
                        base_meta=base_meta,
                        identity=identity,
                        cfg=selfcond_cfg,
                        dataloader_state_dict=dataloader_state_envelope,
                        loop_state={
                            "step": completed_step,
                            "smooth_train_loss": smooth_loss,
                            "min_val_bpb": min_val_bpb,
                            "min_val_bpb_selfcond": min_val_bpb_selfcond,
                            "total_training_time": total_training_time,
                        },
                        base_checkpoint_dir=base_checkpoint_dir,
                        turn_self_cond_segment_size_rustbpe=(turn_self_cond_segment_size_rustbpe),
                    )
                    with profile_session.recording(step):
                        with timed_region("selfcond/checkpoint_save"):
                            save_checkpoint(
                                output_dir,
                                completed_step,
                                model.state_dict(),
                                optimizer.state_dict(),
                                save_meta,
                                rank=ddp_rank,
                            )
                else:
                    assert sft_state is not None
                    assert selfcond_data_dict is not None
                    assert sft_state.artifact_coverage is not None
                    save_meta = build_selfcond_checkpoint_meta(
                        model=model,
                        base_meta=base_meta,
                        identity=identity,
                        cfg=selfcond_cfg,
                        data_source=data_source,
                        selfcond_data=selfcond_data_dict,
                        turn_self_cond_segment_size_rustbpe=(turn_self_cond_segment_size_rustbpe),
                        dataloader_state_dict=None,
                        loop_state={
                            "step": completed_step,
                            "smooth_train_loss": smooth_loss,
                            "min_val_bpb": min_val_bpb,
                            "min_val_bpb_selfcond": min_val_bpb_selfcond,
                            "total_training_time": total_training_time,
                            "sft_cursor_semantics": (SELFCOND_SFT_CURSOR_SEMANTICS),
                            "sft_consumed": committed_consumed,
                            # The buffered envelope is what makes the cursor
                            # exactly resumable; ``buffered_v2`` asserts both
                            # are present, and the generator refuses a
                            # consumed-only resume.
                            "sft_dataloader_state": (sft_state.dataloader_state),
                            "sft_artifact_coverage": (sft_state.artifact_coverage),
                        },
                        base_checkpoint_dir=base_checkpoint_dir,
                    )
                    with profile_session.recording(step):
                        with timed_region("selfcond/checkpoint_save"):
                            save_checkpoint(
                                output_dir,
                                completed_step,
                                model.state_dict(),
                                optimizer.state_dict(),
                                save_meta,
                                rank=ddp_rank,
                            )
                wrote_checkpoint = True

        if profile_session.enabled:
            with rank_synchronized_stage(
                "selfcond profiling finalization",
                device=device,
            ):
                profile_session.finalize()
        if master_process:
            if wrote_checkpoint:
                logger.info(
                    "Self-conditioning complete. Downstream consumers must use the "
                    f"explicit checkpoint directory: {output_dir}"
                )
            else:
                logger.warning(
                    "Self-conditioning finished WITHOUT writing any checkpoint "
                    f"(save_every={int(cfg.save_every)}, num_iterations="
                    f"{int(cfg.num_iterations)}); nothing under {output_dir} was "
                    "produced by this run."
                )
        startup_trace.complete(
            first_step_s=first_step_elapsed,
            steady_state_s=steady_state_elapsed,
        )
        wandb_run.finish()
    except BaseException as error:
        if profile_session is not None:
            profile_session.close_on_error(error)
        startup_trace.failed(error)
        startup_trace.ensure_failure_visible()
        raise
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
