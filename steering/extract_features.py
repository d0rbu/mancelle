"""Segment-granular SAE feature extraction from FineWeb-Edu parquet shards.

Loads a TransformerLens model + SAE (GPT-2/GemmaScope/etc.), processes
individual documents from parquet files, and extracts non-zero SAE features
per segment in CSR-style sparse format.

Uses Hydra for configuration. Run with:
    uv run python -m steering.extract_features
"""

from __future__ import annotations

import collections
import functools
import gc
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from beartype import beartype
from loguru import logger
from nnterp import StandardizedTransformer
from omegaconf import DictConfig
from sae_lens import SAE
from torch.profiler import record_function
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

from errors import (
    AcceleratedBackendError,
    BosOnlySegmentError,
    CompileGateError,
    DataLoadError,
    EmptyExtractionSegmentError,
    ExtractionError,
    ModelWeightLoadError,
    RowPackGateError,
    ValidationError,
)
from nanochat.common import get_peak_flops
from nanochat.dataset import list_parquet_files
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.axbench_base.profiler import record_counter
from steering.extraction_identity import ExtractionSourceIdentity, extraction_documents_sha256
from steering.feature_equivalence import (
    BASELINE_RELATIVE_SLACK,
    compiled_passes_ideal_math_gate,
    feature_equivalence_metrics,
)
from steering.feature_io import (
    ExtractionBackend,
    MetadataKey,
    assert_extraction_backend_compatible,
    assert_l0_tier_compatible,
    empty_segment_indices,
    feature_path,
    feature_path_exists,
    glob_features,
    read_shard_metadata,
    save_features,
    warn_ignore_bos_cfg_key_ignored,
)
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    PoolingMode,
    check_sft_data_ready,
    resolve_data_dir,
    resolve_output_dir,
)
from steering.gemma3_loading import load_gemma3_text_decoder
from steering.models import (
    EXTRACTION_MODELS,
    resolve_sae_max_context,
    resolve_tokenizer,
)
from steering.native_collector import NativeCollector, build_native_model
from steering.parquet_io import select_shard_paths
from steering.profiling.extraction import extraction_profiler
from steering.profiling.markers import timed_region
from steering.refined_types import (
    NonBlankStr,
    NonNegativeFiniteNumber,
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveFiniteNumberLike,
    PositiveInt,
    UnitInterval,
    ZstdCompressionLevel,
)
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    l0_tier_for_resolved,
    resolve_sae,
)
from steering.sae_resolver import (
    coerce_config_l0 as _coerce_config_l0,
)
from steering.segmentation import (
    SegmentMode,
    batch_truncate_to_training_window,
    resolve_bos_token_id,
    segment_document,
    segment_mode_from_metadata,
    truncate_to_training_window,
    turn_segments_for_document,
)
from steering.sft.mixture_fingerprint import (
    MIXTURE_FINGERPRINT_KEY,
    SFT_MIXTURE_FINGERPRINT_METADATA_KEY,
    mixture_fingerprint_json,
    normalize_mixture_fingerprint,
)
from steering.sft_schema import SFTConversionMeta, legacy_sft_columns
from steering.shard_validation import validate_shard, validate_shard_data
from steering.staging import BackgroundSaver, coordinate_free_space_check
from steering.staging.prefetcher import StagedPrefetcher
from steering.staging.relocate import raise_destination_space_error
from steering.staging.saver import StagedSaver
from steering.stats_io import find_statistics_path, load_statistics_metadata


class Gemma3LoadError(RuntimeError):
    """Raised when the nnterp-wrapped Gemma 3 model fails to build."""


def _load_sft_conversion_fingerprint_json(data_dir: str | Path) -> str | None:
    """Read an SFT conversion fingerprint without treating legacy data as SFT."""
    meta_path = Path(data_dir) / "conversion_meta.json"
    if not meta_path.is_file():
        return None
    try:
        raw_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Cannot read SFT conversion metadata {meta_path}: {exc}") from exc
    try:
        metadata = SFTConversionMeta.parse(raw_metadata)
    except ValueError as exc:
        raise DataLoadError(f"Invalid SFT conversion metadata {meta_path}: {exc}") from exc
    if metadata.format != "full_nanochat_text":
        raise DataLoadError(
            f"Legacy SFT feature extraction requires format='full_nanochat_text', "
            f"got {metadata.format!r} at {meta_path}. Structured rollout artifacts "
            "use a separate namespace and consumer."
        )
    fingerprint_raw = metadata.raw.get(MIXTURE_FINGERPRINT_KEY)
    if fingerprint_raw is None:
        return None
    try:
        fingerprint = normalize_mixture_fingerprint(
            fingerprint_raw,
            source=f"{meta_path} mixture_fingerprint",
        )
    except ValueError as exc:
        raise DataLoadError(str(exc)) from exc
    return mixture_fingerprint_json(fingerprint)


def _assert_existing_output_dir_compatible(
    output_dir: str | Path,
    requested_mode: SegmentMode,
    source_fingerprint_json: str | None,
) -> None:
    """Forbid resume from shards with incompatible geometry or migration state.

    Each shard header is read once, in shard order, and its segment mode is
    checked before the next header is read, so a mode mismatch on an earlier
    shard is reported before a corrupt later header.
    """
    existing: list[tuple[Path, dict[str, str]]] = []
    for shard_path in glob_features(output_dir):
        metadata = read_shard_metadata(shard_path)
        existing.append((shard_path, metadata))
        recorded_mode = segment_mode_from_metadata(metadata, source=shard_path)
        if recorded_mode is not requested_mode:
            raise ValueError(
                f"Feature extraction cannot resume from shard {shard_path}: recorded "
                f"segment_mode={recorded_mode.value!r}, requested "
                f"segment_mode={requested_mode.value!r}. Use a fresh output_dir."
            )

    stats_path = find_statistics_path(output_dir)
    if stats_path is not None:
        metadata = load_statistics_metadata(stats_path)
        recorded_mode = segment_mode_from_metadata(metadata, source=stats_path)
        if recorded_mode is not requested_mode:
            raise ValueError(
                f"Feature extraction cannot resume from statistics {stats_path}: "
                f"recorded segment_mode={recorded_mode.value!r}, requested "
                f"segment_mode={requested_mode.value!r}. Use a fresh output_dir."
            )

    for shard_path, metadata in existing:
        shard_fingerprint_json = metadata.get(SFT_MIXTURE_FINGERPRINT_METADATA_KEY)
        if shard_fingerprint_json == source_fingerprint_json:
            continue
        if source_fingerprint_json is None:
            state = "source parquet has no fingerprint but existing shard does"
        elif shard_fingerprint_json is None:
            state = "source parquet has a fingerprint but existing shard does not"
        else:
            state = "source parquet and existing shard fingerprints differ"
        raise DataLoadError(
            f"SFT feature extraction cannot resume in {output_dir}: {state}. "
            f"Source fingerprint={source_fingerprint_json!r}; "
            f"shard {shard_path} fingerprint={shard_fingerprint_json!r}. "
            "Use a fresh output_dir and re-extract the complete conversion."
        )


@dataclass(frozen=True)
class PrefetchedShard:
    """Pre-loaded shard data from a background thread (Phase 1 output).

    Contains the CPU-only data loading results ready for the GPU pipeline in
    process_shard().
    """

    all_text_chunks: list[list[str]]
    all_turn_start_chunks: list[list[list[int]] | None]
    parquet_path: str
    shard_name: str
    num_row_groups: int
    memory_bytes: int  # approximate text memory footprint in bytes


def _validate_extraction_parquet_schema(
    *,
    parquet_path: str,
    schema_names: list[str],
    dataset_phase: str | None,
    segment_mode: SegmentMode,
) -> None:
    """Validate the one supported feature-extraction parquet shape."""
    phase_label = (
        "SFT" if dataset_phase is not None and dataset_phase != "base" else "feature extraction"
    )
    legacy_cols = legacy_sft_columns(schema_names)
    if legacy_cols:
        raise ValueError(
            f"{phase_label} parquet {parquet_path} contains removed SFT legacy "
            f"columns {sorted(legacy_cols)}. Feature extraction consumes rendered "
            "'text' shards only; regenerate with "
            "scripts/preprocess/convert_sft_to_parquet.py or remove the stale columns."
        )
    if "text" not in schema_names:
        raise ValueError(
            f"{phase_label} parquet {parquet_path} must have a 'text' column. "
            f"Found: {schema_names}."
        )
    if (
        segment_mode is SegmentMode.turn_delimited_fixed_tokens
        and "turn_char_starts" not in schema_names
    ):
        raise ValueError(
            f"Turn-delimited feature extraction parquet {parquet_path} must have a "
            "'turn_char_starts' column; re-run "
            "scripts/preprocess/convert_sft_to_parquet into a fresh output_dir."
        )


def _read_text_and_turn_chunks(
    pf: pq.ParquetFile,
    row_group_indices: Iterable[int],
    cpu_batch_size: int,
    segment_mode: SegmentMode,
) -> tuple[list[list[str]], list[list[list[int]] | None]]:
    """Read row groups into aligned text and optional turn-boundary chunks."""
    all_text_chunks: list[list[str]] = []
    all_turn_start_chunks: list[list[list[int]] | None] = []
    for rg_idx in row_group_indices:
        with record_function("read_row_group"):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            turn_starts = (
                rg.column("turn_char_starts").to_pylist()
                if segment_mode is SegmentMode.turn_delimited_fixed_tokens
                else None
            )
            for chunk_start in range(0, len(texts), cpu_batch_size):
                chunk = slice(chunk_start, chunk_start + cpu_batch_size)
                all_text_chunks.append(texts[chunk])
                all_turn_start_chunks.append(
                    None if turn_starts is None else turn_starts[chunk]
                )
    return all_text_chunks, all_turn_start_chunks


def prefetch_shard_data(
    parquet_path: str,
    cpu_batch_size: int,
    dataset_phase: str | None = None,
    *,
    source_path: str | None = None,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> PrefetchedShard:
    """Load and preprocess a shard's parquet data (CPU-only, thread-safe).

    This is Phase 1 of process_shard() extracted into a standalone function
    for shard-level pipelining. It reads the parquet file, splits texts into
    cpu_batch_size chunks, and returns a PrefetchedShard ready for Phase 2.

    Args:
        parquet_path: Path to the parquet shard file to READ (may be an ephemeral
            local staged copy).
        cpu_batch_size: Number of documents per CPU chunk.
        dataset_phase: Dataset phase (None or "base" for pretraining, else SFT).
        source_path: Durable path reported in schema-validation errors, the debug
            log, and on the returned shard's ``parquet_path``/``shard_name``. Defaults
            to ``parquet_path``; set it when reading a staged copy so diagnostics name
            the source rather than the soon-deleted scratch file.

    Returns:
        PrefetchedShard with all_text_chunks.
    """
    diag_path = source_path if source_path is not None else parquet_path
    pf = pq.ParquetFile(parquet_path)
    shard_name = Path(diag_path).stem

    _validate_extraction_parquet_schema(
        parquet_path=diag_path,
        schema_names=pf.schema_arrow.names,
        dataset_phase=dataset_phase,
        segment_mode=segment_mode,
    )

    all_text_chunks, all_turn_start_chunks = _read_text_and_turn_chunks(
        pf,
        range(pf.num_row_groups),
        cpu_batch_size,
        segment_mode,
    )

    memory_bytes = sum(sum(len(text) for text in chunk) for chunk in all_text_chunks)
    logger.debug(
        "Prefetched shard {}: {} chunks, {} row groups, ~{:.1f} MB text",
        shard_name,
        len(all_text_chunks),
        pf.num_row_groups,
        memory_bytes / (1024 * 1024),
    )

    return PrefetchedShard(
        all_text_chunks=all_text_chunks,
        all_turn_start_chunks=all_turn_start_chunks,
        parquet_path=diag_path,
        shard_name=shard_name,
        num_row_groups=pf.num_row_groups,
        memory_bytes=memory_bytes,
    )


def _staged_prefetch_load(
    source_path: str,
    local_path: str,
    *,
    cpu_batch_size: int,
    dataset_phase: str | None,
    segment_mode: SegmentMode,
) -> PrefetchedShard:
    """``StagedPrefetcher`` ``load_fn`` adapter: read the LOCAL staged copy but
    keep the ``PrefetchedShard``'s identity pointing at the original (network)
    source path.

    ``StagedPrefetcher`` copies ``source_path`` into local scratch (one
    sequential bulk read) and calls this with ``(source_path, local_path)``. We
    read the local copy but pass ``source_path`` so that schema-validation errors,
    the debug log line, and the returned shard's ``parquet_path``/``shard_name``
    all name the durable source rather than the ephemeral scratch file (which is
    deleted the moment this returns).
    """
    return prefetch_shard_data(
        local_path,
        cpu_batch_size,
        dataset_phase=dataset_phase,
        source_path=source_path,
        segment_mode=segment_mode,
    )


@dataclass(frozen=True)
class ShardResult:
    n_docs: int
    n_tokens_gpt2: int
    n_total_features: int
    elapsed_seconds: float
    output_path: str
    bytes_on_disk: int  # always 0 when a background saver is used: file
    # is written in the background, so the size
    # isn't available by the time process_shard returns


# A shard can take tens of minutes, so INFO must carry durable progress when the
# process is running under Slurm (where stderr is redirected and tqdm is not a
# useful log signal).  Keep this operational contract local and fixed rather
# than adding another Hydra/config identity knob.
_EXTRACTION_HEARTBEAT_DOCS = 1_000
_EXTRACTION_HEARTBEAT_SECONDS = 60.0


def _tqdm_disabled() -> bool:
    """Disable terminal progress bars when stderr is redirected to a log file."""
    return not sys.stderr.isatty()


class _ShardProgress:
    """Rate-limited, durable progress reporting for one extraction shard.

    Emission is driven by completed extraction work rather than a background
    timer.  This means a heartbeat is evidence that the extraction loop is
    making progress, not merely that the Python process is still alive.  The
    time threshold is a fallback for workloads whose documents are unusually
    expensive; it is checked at the same work boundaries as the document
    threshold so the helper cannot outlive or race a shard run.
    """

    def __init__(self, shard_name: str, total_docs: int) -> None:
        self.shard_name = shard_name
        self.total_docs = total_docs
        self._started_at = time.monotonic()
        self._last_emitted_at = self._started_at
        self._next_doc_milestone = _EXTRACTION_HEARTBEAT_DOCS

    def emit(
        self,
        *,
        docs_done: int,
        segments_done: int,
        phase: str = "extract",
        force: bool = False,
    ) -> bool:
        """Emit a heartbeat when count/time cadence is met; return whether emitted."""
        now = time.monotonic()
        count_due = docs_done >= self._next_doc_milestone
        time_due = now - self._last_emitted_at >= _EXTRACTION_HEARTBEAT_SECONDS
        if not force and not count_due and not time_due:
            return False

        if self.total_docs == 0:
            progress = 1.0
        else:
            progress = min(max(docs_done / self.total_docs, 0.0), 1.0)
        logger.info(
            "extract_features heartbeat: shard={} phase={} docs_done={} "
            "docs_total={} progress={:.3f} segments_done={} elapsed_s={:.1f}",
            self.shard_name,
            phase,
            docs_done,
            self.total_docs,
            progress,
            segments_done,
            now - self._started_at,
        )
        self._last_emitted_at = now
        while self._next_doc_milestone <= docs_done:
            self._next_doc_milestone += _EXTRACTION_HEARTBEAT_DOCS
        return True


@dataclass
class ShardData:
    offsets: t.Tensor
    feature_ids: t.Tensor
    feature_mags: t.Tensor
    doc_ids: t.Tensor
    segment_indices: t.Tensor
    segment_positions: t.Tensor
    doc_token_counts: t.Tensor
    n_docs: int
    n_segments: int
    layer: int
    context_window: int
    pooling: str
    segment_size: int
    segment_char_starts: t.Tensor
    segment_char_ends: t.Tensor
    segment_mode: SegmentMode = SegmentMode.fixed_tokens
    context_independent_features: bool = False
    ignore_bos_features: bool = False

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ShardData:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class ContextIndependentSegmentInput:
    """One segment prepared for context-independent forward pass.

    Invariant: ``len(model_input) == original_length + (1 if bos_injected else 0)``.
    The ``doc_idx``/``seg_idx`` pair uniquely identifies the segment's position
    within the source chunk; used to scatter batched sparse results back into
    per-document order.

    ``keep_mask`` is the per-segment non-``<bos>`` keep-mask over the segment's
    real tokens (``None`` when ignore_bos_features is off).  Aligned 1:1 with
    ``real_acts`` after injected-BOS slicing: in the injected case
    ``real_acts=acts[1:]`` so ``len(real_acts)==len(seg)``; in the natural-BOS
    case ``real_acts=acts`` so ``len(real_acts)==len(seg)`` as well.
    """

    doc_idx: int
    seg_idx: int
    model_input: list[int]
    bos_injected: bool
    original_length: int
    keep_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class RowpackGateProbe:
    """Production-shaped row-pack gate input surface.

    ``token_id_lists`` are the exact sequences forwarded by the native model.
    ``segments_by_sequence`` are spans inside those forwarded sequences that
    become extracted rows after SAE encoding. ``keep_masks_by_sequence`` are
    optional full-sequence BOS-exclusion masks aligned to ``token_id_lists``.
    """

    token_id_lists: list[list[int]]
    segments_by_sequence: list[list[tuple[int, int]]]
    keep_masks_by_sequence: list[tuple[bool, ...] | None]
    context_independent_features: bool


def _validate_doc_accounting(
    doc_token_counts: list[int],
    doc_segment_info: list,
    doc_counter: int,
) -> None:
    """Fail loud if the per-doc accumulators desync from ``doc_counter``.

    Enforces the shard<->parquet alignment the conditioning pipeline depends on:
    every written doc advances ``doc_counter`` exactly once and contributes exactly
    one ``doc_token_counts`` / ``doc_segment_info`` row, across BOTH the GPU-batched
    and context-independent extraction paths.

    Typed ``RuntimeError`` rather than ``assert`` so the check survives ``python -O``
    (assertions are stripped at -O level), matching ``_validate_csr_invariants``.
    """
    if len(doc_token_counts) != doc_counter:
        raise RuntimeError(
            f"doc accounting desync: len(doc_token_counts)={len(doc_token_counts)} "
            f"!= doc_counter={doc_counter}"
        )
    if len(doc_segment_info) != doc_counter:
        raise RuntimeError(
            f"doc accounting desync: len(doc_segment_info)={len(doc_segment_info)} "
            f"!= doc_counter={doc_counter}"
        )


def _validate_csr_invariants(
    offsets: t.Tensor,
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    n_segments: int,
    doc_ids: t.Tensor,
    segment_indices: t.Tensor,
    segment_positions: t.Tensor,
    segment_char_starts: t.Tensor,
    segment_char_ends: t.Tensor,
) -> None:
    """Pre-save CSR shape + monotonicity validation.

    Implemented as typed ``RuntimeError`` raises rather than ``assert`` so
    the checks survive ``python -O`` (assertions are stripped at -O level).
    See slice 01 audit (suspected-bug #27).
    """
    if int(offsets[0].item()) != 0:
        raise RuntimeError(f"offsets[0] must be 0, got {int(offsets[0].item())}")
    if int(offsets[-1].item()) != len(feature_ids):
        raise RuntimeError(
            f"offsets[-1]={int(offsets[-1].item())} != len(feature_ids)={len(feature_ids)}"
        )
    if len(offsets) != n_segments + 1:
        raise RuntimeError(f"len(offsets)={len(offsets)} != n_segments+1={n_segments + 1}")
    if not (offsets[1:] >= offsets[:-1]).all():
        raise RuntimeError("offsets must be monotonically non-decreasing")
    empties = empty_segment_indices(offsets)
    if empties:
        i = empties[0]
        # doc_ids length is not validated until the check below, so guard the
        # lookup; -1 is a sentinel meaning "doc id unavailable (malformed
        # doc_ids)" — the empty-segment error is the more informative failure
        # to surface first.
        doc = int(doc_ids[i].item()) if i < len(doc_ids) else -1
        raise EmptyExtractionSegmentError(
            f"Base extraction emitted segment {i} (doc {doc}) with zero active "
            f"features (offsets[{i}]==offsets[{i + 1}]={int(offsets[i].item())}); "
            f"this indicates a pooling/threshold/serialization or source-alignment "
            f"problem — see issue #522."
        )
    if len(doc_ids) != n_segments:
        raise RuntimeError(f"len(doc_ids)={len(doc_ids)} != n_segments={n_segments}")
    if len(segment_indices) != n_segments:
        raise RuntimeError(
            f"len(segment_indices)={len(segment_indices)} != n_segments={n_segments}"
        )
    if len(segment_positions) != n_segments:
        raise RuntimeError(
            f"len(segment_positions)={len(segment_positions)} != n_segments={n_segments}"
        )
    if len(segment_char_starts) != n_segments:
        raise RuntimeError(
            f"len(segment_char_starts)={len(segment_char_starts)} != n_segments={n_segments}"
        )
    if len(segment_char_ends) != n_segments:
        raise RuntimeError(
            f"len(segment_char_ends)={len(segment_char_ends)} != n_segments={n_segments}"
        )
    if len(feature_ids) != len(feature_mags):
        raise RuntimeError(
            f"len(feature_ids)={len(feature_ids)} != len(feature_mags)={len(feature_mags)}"
        )


class Backend(Enum):
    """Inference backend for feature extraction."""

    auto = "auto"  # nnterp; hard-abort if unavailable (use transformerlens for TL)
    nnterp = "nnterp"  # nnsight-based (faster, GPU-only)
    transformerlens = "transformerlens"  # TransformerLens (CPU/GPU)
    native = "native"  # home-grown raw-HF + forward-hook (GPU, Gemma only)


def resolved_extraction_backend(
    nnterp_model: object | None, native_model: object | None,
) -> ExtractionBackend:
    """Name the execution objects passed to the batched extraction dispatcher."""
    if native_model is not None:
        if nnterp_model is not None:
            raise ValueError("Extraction must have only one accelerated backend")
        return ExtractionBackend.native
    if nnterp_model is not None:
        return ExtractionBackend.nnterp
    return ExtractionBackend.transformerlens


def _require_cuda_for_native(backend: Backend, device: str) -> bool:
    """Return True iff the run must abort: backend=native on a non-CUDA device.

    The native backend is GPU-only (raw-HF + forward hooks); on a non-CUDA
    device its build branch is skipped, so without this guard ``main()`` would
    silently fall back to the CPU-resident TransformerLens model despite the
    user explicitly requesting ``native``. Pure decision so the abort can be
    unit-tested without mocking ``main()``'s model-loading machinery.
    """
    return backend == Backend.native and not device.startswith("cuda")


def _accelerated_backend_error(
    backend_name: str, phase: str, cause: BaseException | None
) -> AcceleratedBackendError:
    """Build the single, uniform fail-fast error for an unavailable/failed
    accelerated backend. ``phase`` is "build" or "forward". The cause is
    chained onto ``__cause__`` so the root error is preserved."""
    err = AcceleratedBackendError(
        f"accelerated backend {backend_name!r} failed during {phase}: "
        f"{cause!r}. Refusing to silently fall back to TransformerLens "
        f"(slower, numerically divergent). Set backend=transformerlens to "
        f"run TransformerLens intentionally, or fix the {backend_name} backend."
    )
    err.__cause__ = cause
    return err


def _build_accelerated_backend(
    backend: Backend,
    model,
    device: str,
    model_dtype,
    cfg,
) -> tuple[StandardizedTransformer | None, NativeCollector | None]:
    """Build the accelerated backend per ``backend``, or hard-raise.

    Returns ``(nn_model, native_model)``. ``backend=transformerlens`` returns
    ``(None, None)`` (no accelerated build — the only intentional TL route).
    Any unavailability/build-failure for ``native``/``nnterp``/``auto``
    raises ``AcceleratedBackendError`` (never silent TL). Extracted from
    ``main()`` so it is directly unit-testable without ``@hydra.main``.
    """
    if _require_cuda_for_native(backend, device):
        raise _accelerated_backend_error(
            "native",
            "build",
            RuntimeError(
                f"native backend requires a CUDA device but got {device!r} "
                "(set backend=transformerlens to run on this device)"
            ),
        )
    if backend == Backend.native and device.startswith("cuda"):
        # Native extraction never forwards through the already-loaded
        # TransformerLens model; keep only its tokenizer/cfg on CPU before
        # loading the separate HF native collector to avoid co-resident
        # TL+native peaks on 24 GB cards.
        model.to("cpu")
        t.cuda.empty_cache()
        native_model = build_native_model(
            cfg.model, device=device, model_dtype=model_dtype, layer=cfg.layer
        )
        if native_model is None:
            raise _accelerated_backend_error(
                "native",
                "build",
                RuntimeError("build_native_model returned None"),
            )
        return None, native_model
    if backend in (Backend.nnterp, Backend.auto) and device.startswith("cuda"):
        nn_model = build_nnterp_model(model, device=device, model_dtype=model_dtype)
        if nn_model is None:
            raise _accelerated_backend_error(
                "nnterp",
                "build",
                RuntimeError("build_nnterp_model returned None"),
            )
        return nn_model, None
    # Any non-TL backend reaching here couldn't establish an
    # accelerated backend (non-CUDA nnterp/auto; native+cpu already
    # raised at the pre-flight above).
    if backend != Backend.transformerlens:
        raise _accelerated_backend_error(
            backend.value,
            "build",
            RuntimeError(
                f"backend={backend.value!r} requires a CUDA device but got "
                f"{device!r}; set backend=transformerlens to run on this device"
            ),
        )
    return None, None


def load_sae_and_model(
    model_name: str = "gpt2",
    layer: int = 8,
    device: str = "cpu",
    *,
    sae_release: str | None = None,
    sae_id: str | None = None,
    resolved: ResolvedSAE | None = None,
    offload_transformer_tail: bool = False,
) -> tuple[HookedTransformer, SAE, object]:
    """Load a TransformerLens model and SAE from SAELens.

    Two loading paths:
      1. Raw coordinates (sae_release + sae_id), used by shard validation and
         focused diagnostics whose identities come from artifact metadata.
      2. Resolved coordinates, used by extraction and evaluation pipelines.
    Exactly one path must be supplied.

    The hook_name comes from either resolved.hook_name or the loaded sae.cfg.hook_name.

    Returns:
        (model, sae, tokenizer) tuple.
    """
    offload_layer = NonNegativeInt.parse(layer) if offload_transformer_tail else None
    if resolved is not None:
        if sae_release is not None or sae_id is not None:
            raise ValueError("resolved cannot be combined with sae_release or sae_id")
        release = resolved.release
        resolved_sae_id = resolved.sae_id
    else:
        if sae_release is None or sae_id is None:
            raise ValueError(
                "Must provide either resolved (ResolvedSAE) or both sae_release and sae_id"
            )
        release = sae_release
        resolved_sae_id = sae_id

    model = HookedTransformer.from_pretrained(model_name, device=device)
    if offload_layer is not None:
        _offload_transformer_tail_for_early_stop(
            model,
            last_required_layer=offload_layer,
            device=device,
        )

    sae_obj = SAE.from_pretrained(
        release=release,
        sae_id=resolved_sae_id,
        device=device,
    )

    return model, sae_obj, model.tokenizer


def _resolve_extraction_sae(
    cfg: DictConfig,
    *,
    model_name: str,
    activation_type: ActivationType,
    layer: int,
) -> ResolvedSAE:
    """Resolve the one supported extraction SAE configuration shape."""
    if "sae_id" in cfg:
        raise ValueError(
            "The extraction sae_id config key has been removed; configure "
            "sae.release, sae.num_features, and sae.l0 instead."
        )
    if cfg.sae is None:
        raise ValueError("Extraction requires a structured sae config")
    return resolve_sae(
        model=model_name,
        release=SAERelease(cfg.sae.release),
        activation_type=activation_type,
        layer=layer,
        width=SAEWidth(cfg.sae.num_features),
        l0=_coerce_config_l0(SAERelease(cfg.sae.release), cfg.sae.get("l0")),
    )


@beartype
def _offload_transformer_tail_for_early_stop(
    model: object,
    *,
    last_required_layer: NonNegativeInt,
    device: str | t.device,
) -> None:
    """Move blocks after an extraction hook to CPU before loading its SAE.

    Every production activation extractor calls TransformerLens with
    ``stop_at_layer=last_required_layer + 1``.  Blocks after that boundary are
    unreachable, yet a large Gemma model normally keeps them resident while a
    wide SAE is loaded.  This opt-in lifetime optimization moves exactly those
    unreachable blocks to CPU; executed blocks, embeddings, hook activations,
    and SAE parameters are unchanged.

    The helper is intentionally not automatic.  Callers that may perform a
    full model forward must retain the default fully-resident loader path.
    """

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("cannot offload TransformerLens tail: model has no blocks collection")
    n_layers = len(blocks)
    if not 0 <= last_required_layer < n_layers:
        raise ValueError(
            "cannot offload TransformerLens tail: last_required_layer="
            f"{last_required_layer} is outside [0, {n_layers})"
        )
    for block in blocks[last_required_layer + 1 :]:
        block.to("cpu")
    if t.device(device).type == "cuda" and t.cuda.is_available():
        t.cuda.empty_cache()


def build_nnterp_model(
    tl_model: HookedTransformer | None = None,
    device: str | t.device = "cuda",
    model_dtype: t.dtype | None = t.float32,
    *,
    model_name: str | None = None,
) -> StandardizedTransformer | None:
    """Build an nnterp-wrapped HuggingFace model, dispatching by architecture.

    Returns the nnterp StandardizedTransformer, or None if the architecture
    is not supported for nnterp wrapping.

    ``model_name`` may be provided explicitly (e.g. for Gemma 3, which is
    loaded directly from HF without a TransformerLens companion); otherwise
    it is taken from ``tl_model.cfg.model_name``.
    """
    if model_name is None:
        if tl_model is None:
            raise ValueError(
                "build_nnterp_model requires either a tl_model or an explicit model_name"
            )
        model_name = tl_model.cfg.model_name
    if model_name.startswith("gemma-3"):
        try:
            return _build_nnterp_gemma3(
                model_name=model_name,
                tl_model=tl_model,
                device=device,
                model_dtype=model_dtype,
            )
        except Gemma3LoadError:
            raise
        except ModelWeightLoadError:
            raise
        except Exception as exc:
            logger.warning(f"nnterp Gemma-3 build failed: {exc}; returning None.")
            return None
    # gpt2/gemma-2 retain their original signatures (HookedTransformer required,
    # str device, non-None dtype). Narrow the dispatcher's wider types here.
    if tl_model is None:
        raise ValueError(
            f"build_nnterp_model({model_name!r}) requires a tl_model for non-gemma-3 architectures."
        )
    device_str = str(device) if isinstance(device, t.device) else device
    if model_name == "gpt2":
        return _build_nnterp_gpt2(tl_model, device_str)
    if model_name.startswith("gemma-2"):
        dtype_for_gemma2 = model_dtype if model_dtype is not None else t.float32
        try:
            return _build_nnterp_gemma2(tl_model, device_str, model_dtype=dtype_for_gemma2)
        except RuntimeError as e:
            logger.warning(f"nnterp Gemma-2 build failed: {e}; returning None.")
            return None
    logger.warning(f"No nnterp support for {model_name}; returning None.")
    return None


def _build_nnterp_gpt2(
    tl_model: HookedTransformer,
    device: str = "cuda",
) -> StandardizedTransformer:
    """Build an nnterp-wrapped HuggingFace GPT-2 with TransformerLens weight transforms.

    TransformerLens applies fold_ln, center_writing_weights, center_unembed when
    loading GPT-2, changing intermediate activations. The SAE was trained on these
    transformed activations. This function applies the same transforms to a
    HuggingFace GPT-2 model and wraps it with nnterp for early stopping via
    tracer.stop(), which is ~1.9x faster than TransformerLens stop_at_layer.
    """
    d_model = tl_model.cfg.d_model
    n_layers = tl_model.cfg.n_layers

    tl_sd = tl_model.state_dict()
    hf_sd = {}

    # Embeddings (same shape)
    hf_sd["wte.weight"] = tl_sd["embed.W_E"].clone()
    hf_sd["wpe.weight"] = tl_sd["pos_embed.W_pos"].clone()

    for layer_idx in range(n_layers):
        tp, hp = f"blocks.{layer_idx}", f"h.{layer_idx}"
        # LayerNorm: identity (weights folded into attn/MLP by TransformerLens)
        hf_sd[f"{hp}.ln_1.weight"] = t.ones(d_model)
        hf_sd[f"{hp}.ln_1.bias"] = t.zeros(d_model)
        hf_sd[f"{hp}.ln_2.weight"] = t.ones(d_model)
        hf_sd[f"{hp}.ln_2.bias"] = t.zeros(d_model)
        # Attention QKV: [n_heads, d_model, d_head] -> [d_model, 3*d_model]
        w_Q = tl_sd[f"{tp}.attn.W_Q"].permute(1, 0, 2).reshape(d_model, -1)
        w_K = tl_sd[f"{tp}.attn.W_K"].permute(1, 0, 2).reshape(d_model, -1)
        w_V = tl_sd[f"{tp}.attn.W_V"].permute(1, 0, 2).reshape(d_model, -1)
        hf_sd[f"{hp}.attn.c_attn.weight"] = t.cat([w_Q, w_K, w_V], dim=1)
        b_Q = tl_sd[f"{tp}.attn.b_Q"].reshape(-1)
        b_K = tl_sd[f"{tp}.attn.b_K"].reshape(-1)
        b_V = tl_sd[f"{tp}.attn.b_V"].reshape(-1)
        hf_sd[f"{hp}.attn.c_attn.bias"] = t.cat([b_Q, b_K, b_V])
        # Attention output: [n_heads, d_head, d_model] -> [d_model, d_model]
        hf_sd[f"{hp}.attn.c_proj.weight"] = tl_sd[f"{tp}.attn.W_O"].reshape(-1, d_model)
        hf_sd[f"{hp}.attn.c_proj.bias"] = tl_sd[f"{tp}.attn.b_O"].clone()
        # MLP (same shapes)
        hf_sd[f"{hp}.mlp.c_fc.weight"] = tl_sd[f"{tp}.mlp.W_in"].clone()
        hf_sd[f"{hp}.mlp.c_fc.bias"] = tl_sd[f"{tp}.mlp.b_in"].clone()
        hf_sd[f"{hp}.mlp.c_proj.weight"] = tl_sd[f"{tp}.mlp.W_out"].clone()
        hf_sd[f"{hp}.mlp.c_proj.bias"] = tl_sd[f"{tp}.mlp.b_out"].clone()

    hf_sd["ln_f.weight"] = t.ones(d_model)
    hf_sd["ln_f.bias"] = t.zeros(d_model)

    hf_lm = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_lm_sd = {f"transformer.{k}": v for k, v in hf_sd.items()}
    hf_lm_sd["lm_head.weight"] = hf_sd["wte.weight"].clone()
    hf_lm.load_state_dict(hf_lm_sd, strict=False)
    hf_lm = cast(Any, hf_lm).to(device).eval()

    nn_model = StandardizedTransformer(hf_lm)
    logger.info("Built nnterp GPT-2 model with TransformerLens weight transforms (~1.9x faster)")
    return nn_model


def _build_nnterp_gemma2(
    tl_model: HookedTransformer,
    device: str = "cuda",
    model_dtype: t.dtype = t.float32,
) -> StandardizedTransformer:
    """Build an nnterp-wrapped HuggingFace Gemma-2 model.

    GemmaScope SAEs were trained on vanilla Gemma-2 activations. TransformerLens
    for Gemma-2 doesn't apply aggressive weight transforms (no fold_ln for
    RMSNorm), so we can load the HuggingFace model directly.

    The TL model is offloaded to CPU before loading the HF copy to avoid OOM
    on 24GB cards.  Verification runs on CPU.  After this function returns,
    the TL model remains on CPU (only used for its tokenizer).
    """
    tl_name = tl_model.cfg.model_name
    hf_name = EXTRACTION_MODELS[tl_name].hf_model_id

    # Offload TL model to CPU to make room for the HF copy on GPU
    tl_model.to("cpu")
    t.cuda.empty_cache()
    logger.info("Offloaded TransformerLens model to CPU before loading HF model")

    # Always load in float32 for verification, then convert if needed.
    # Loading directly in bf16 causes dtype mismatches in nnterp/nnsight.
    hf_model_any = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=t.float32,
        ),
    )
    hf_model = hf_model_any.to(device).eval()
    nn_model = StandardizedTransformer(hf_model)

    # Verify on CPU (5 tokens — fast, avoids moving TL back to GPU).
    # Higher tolerance because CPU-vs-GPU float32 introduces small diffs.
    _verify_logits_match(tl_model, nn_model, atol=0.1)

    # Convert to requested dtype after verification
    if model_dtype != t.float32:
        nn_model.model.to(model_dtype)
        logger.info(f"Converted nnterp model to {model_dtype}")

    logger.info(f"Built nnterp Gemma-2 model (direct HuggingFace load, dtype={model_dtype})")
    return nn_model


def _build_nnterp_gemma3(
    *,
    model_name: str,
    tl_model: HookedTransformer | None,
    device: str | t.device,
    model_dtype: t.dtype | None = None,
) -> StandardizedTransformer:
    """Build an nnterp-wrapped HuggingFace Gemma-3 model.

    Mirrors :func:`_build_nnterp_gemma2`. Loads the text decoder via
    ``load_gemma3_text_decoder``, which correctly remaps VLM (4b/12b/27b)
    checkpoints (text weights under ``language_model.model.*``) into a
    ``Gemma3ForCausalLM`` — a plain ``Gemma3ForCausalLM.from_pretrained``
    would silently random-init them (issue #549). nnterp then wraps the
    resulting text-only decoder.

    The argument-order follows the keyword-only convention to make the call
    site self-documenting and to align with the public ``build_nnterp_model``
    signature.

    The ``attn_implementation="sdpa"`` kwarg is pinned explicitly to avoid
    routing onto an FA4 path that hasn't been exercised on sm_120
    (Blackwell consumer / RTX PRO 6000). See peer-agent flag in
    ``dev/agent-handoff/gemma3-gemmascope2-extraction.md`` (tick 3).
    """
    if model_name not in EXTRACTION_MODELS:
        raise Gemma3LoadError(f"Model {model_name!r} is not registered in EXTRACTION_MODELS.")
    cfg = EXTRACTION_MODELS[model_name]

    # Offload the TransformerLens companion to CPU before loading the HF
    # Gemma-3 copy. For a >=4B model the fp32 TL model (~16 GB) would
    # double-resident with this bf16 HF build and OOM a 24 GB GPU. Mirrors
    # _build_nnterp_gemma2's pre-load offload. tl_model is unused below and
    # is None only in unit tests.
    if tl_model is not None:
        tl_model.to("cpu")
        t.cuda.empty_cache()
        logger.info("Offloaded TransformerLens model to CPU before loading HF Gemma-3 model")

    try:
        hf_model = load_gemma3_text_decoder(
            cfg.hf_model_id,
            dtype=model_dtype if model_dtype is not None else t.float32,
            attn_implementation="sdpa",
        )
    except ModelWeightLoadError:
        raise  # FATAL — see issue #549
    except Exception as exc:
        raise Gemma3LoadError(
            f"Failed to load {model_name} (HF: {cfg.hf_model_id}). "
            f"Likely cause: missing HF_TOKEN with access to google/gemma-3 "
            f"(gated). Original: {exc!r}"
        ) from exc

    # resolve_tokenizer is called for parity with other builders and to
    # surface a fast HF_TOKEN/gated-repo error before the heavier nnterp wrap.
    resolve_tokenizer(model_name)
    nnterp_model = StandardizedTransformer(hf_model)
    target_device = device if isinstance(device, t.device) else t.device(device)
    nnterp_model.to(target_device)
    if model_dtype is not None:
        nnterp_model.model.to(model_dtype)
        logger.info(f"Converted nnterp Gemma-3 model to {model_dtype}")
    logger.info(
        f"Built nnterp Gemma-3 model {model_name} (direct HuggingFace load, dtype={model_dtype})"
    )
    return nnterp_model


def _verify_logits_match(
    tl_model: HookedTransformer,
    nn_model: StandardizedTransformer,
    atol: float = 1e-3,
) -> None:
    """Assert nnterp model produces same logits as TL model on test input.

    Runs both models on CPU so the TL model doesn't need to be on GPU.
    """
    test_tokens = t.tensor([[1, 2, 3, 4, 5]], device="cpu")
    with t.no_grad():
        tl_logits = tl_model(test_tokens)
        # nnterp model may be on GPU — move input there, then bring logits back
        nn_device = next(nn_model.parameters()).device
        nn_logits = nn_model(test_tokens.to(nn_device)).logits.cpu()
    if not t.allclose(tl_logits, nn_logits, atol=atol):
        max_diff = (tl_logits - nn_logits).abs().max().item()
        raise RuntimeError(
            f"nnterp logits diverge from TransformerLens (max diff={max_diff:.6f}); returning None."
        )


class ActivationCapture:
    """Captures activations from a TransformerLens hook point during forward pass.

    Usage:
        capture = ActivationCapture()
        handle = model.blocks[layer].hook_resid_pre.register_forward_hook(capture.hook_fn)
        model(tokens)
        activations = capture.activation  # [batch, seq_len, d_model]
        handle.remove()

    The captured tensor is detached from the autograd graph and cloned so
    that downstream consumers cannot accidentally mutate the source tensor
    or pin storage that the model still owns. (Slice 01 audit: suspected-bug
    #18.)
    """

    def __init__(self):
        self.activation: t.Tensor | None = None

    def hook_fn(self, module, input, output):
        if isinstance(output, t.Tensor):
            self.activation = output.detach().clone()
        else:
            # Some hook points return tuples (e.g. attention outputs);
            # take the first tensor element.
            inner = output[0] if isinstance(output, (tuple, list)) else output
            self.activation = inner.detach().clone() if isinstance(inner, t.Tensor) else inner


def _get_activations_tl(
    model: HookedTransformer,
    tokens: t.Tensor,
    layer: int,
    hook_name: str | None = None,
    attention_mask: t.Tensor | None = None,
) -> t.Tensor:
    """Get activations using TransformerLens run_with_cache.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs.
        layer: Layer index (used for stop_at_layer fallback).
        hook_name: TransformerLens hook name. If provided, uses run_with_cache
            with names_filter. Otherwise falls back to stop_at_layer for residual.
        attention_mask: Optional attention mask for padded batches.

    Returns:
        Activation tensor from the specified hook point.
    """
    if hook_name is not None:
        # stop_at_layer=layer+1 avoids running unnecessary later layers
        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
            attention_mask=attention_mask,
            stop_at_layer=layer + 1,
        )
        # Detach + clone for the same reason ``ActivationCapture.hook_fn``
        # does (Slice 01 audit, suspected-bug #18): TransformerLens caches
        # share storage with the model's autograd graph, so a downstream
        # consumer that resizes / writes in-place could either corrupt the
        # forward graph or pin enormous tensors the model still owns.
        return cache[hook_name].detach().clone()
    # Legacy fallback: residual stream via stop_at_layer
    return model(tokens, stop_at_layer=layer, attention_mask=attention_mask).detach().clone()


def _get_activations_nnterp(
    nnterp_model: StandardizedTransformer,
    tokens: t.Tensor,
    layer: int,
    activation_type: ActivationType = ActivationType.residual,
    hook_name: str | None = None,
) -> t.Tensor:
    """Get activations using nnterp tracer.stop() (early halt after needed layer).

    For residual activations, the correct ``layers_output`` index depends on
    the TransformerLens hook point:

    - ``hook_resid_post`` of layer L  -> ``layers_output[L]``
      (output *after* block L processes the residual stream)
    - ``hook_resid_pre`` of layer L   -> ``layers_output[L - 1]``
      (output of block L-1 = input *to* block L)

    When *hook_name* is ``None`` (legacy GPT-2 path), defaults to the
    ``hook_resid_pre`` convention (``layers_output[layer - 1]``).

    For non-residual activation types, attempts model-specific module access.

    Raises:
        ExtractionError: If the activation type is not supported via nnterp
            or capture fails.
    """
    try:
        with nnterp_model.trace(tokens) as tracer:
            if activation_type == ActivationType.residual:
                # Determine correct layers_output index from hook_name
                if hook_name is not None and "hook_resid_post" in hook_name:
                    layer_idx = layer  # output of layer L
                else:
                    layer_idx = layer - 1  # input to layer L = output of L-1
                acts = nnterp_model.layers_output[layer_idx].save()
            elif activation_type == ActivationType.mlp:
                acts = nnterp_model.model.layers[layer].mlp.output.save()
            elif activation_type == ActivationType.attention:
                if hook_name is not None and "hook_attn_out" in hook_name:
                    # Post-projection [batch, seq, d_model] — matches
                    # gemma-scope-2's `attn_out/` SAEs.
                    acts = nnterp_model.model.layers[layer].self_attn.output.save()
                else:
                    # Pre-projection hook_z: nnterp doesn't expose this
                    # shape; raise ExtractionError (caller decides handling).
                    raise ExtractionError(
                        f"nnterp does not support pre-projection hook_z for "
                        f"{activation_type.value} at layer {layer}. Use the "
                        f"TransformerLens backend or pass hook_name with "
                        f"`hook_attn_out` for the post-projection SAE."
                    )
            else:
                raise ValueError(f"Unknown activation type: {activation_type}")
            tracer.stop()
        # Centralized unwrap: HF attention modules (e.g. Gemma3 self_attn
        # for the gemma-scope-2 `hook_attn_out` path) return an
        # ``(hidden_states, attn_weights)`` tuple. Normalize here so the
        # `-> t.Tensor` contract holds for *every* consumer — including
        # `probe_activations`, which does not unwrap at its call site.
        # Mirrors the isinstance idiom used at the extract_features call sites.
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        return acts
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(
            f"nnterp capture failed for {activation_type.value} at layer {layer}: "
            f"{type(e).__name__}: {e}"
        ) from e


def _get_activations_native(
    native_model: NativeCollector,
    tokens: t.Tensor,
    layer: int,
    activation_type: ActivationType = ActivationType.residual,
    hook_name: str | None = None,
) -> t.Tensor:
    """Adapter: native collector → same contract as _get_activations_nnterp.

    Raises ExtractionError on any failure. How a caller handles that is
    caller-defined (e.g. _get_sae_activations_batched converts it to a hard
    AcceleratedBackendError — no silent TransformerLens fallback).

    MECHANISM (Phase-2 PR-1): the native collector no longer registers a
    forward hook. ``NativeCollector.capture`` runs a STRUCTURALLY-TRUNCATED
    forward — the inner Gemma{2,3} decoder stack is sliced to layers
    ``0..layer`` and its final ``norm`` is replaced with ``nn.Identity()``,
    so the truncated model's ``last_hidden_state`` IS ``resid_post(layer)``,
    byte-identical (max_abs == 0) to the removed Phase-1 full-forward-hook
    capture (residual-only by design — mlp/attention/wrong-layer raise a
    typed ExtractionError). See steering/native_collector.py (the
    ``capture`` / ``_truncated_forward`` / ``build_native_model`` truncation
    build).

    DELIBERATELY MASKLESS — not an omitted ``attention_mask``. We do NOT
    forward an attention_mask into ``NativeCollector.capture`` (whose
    signature accepts ``attention_mask=None`` but deliberately ignores it —
    the truncated forward is itself maskless, see the ``capture`` signature
    + maskless rationale at steering/native_collector.py:98-135), so the
    raw-HF/SDPA truncated forward runs maskless. This is intentional and
    required:
      1. Causal, decoder-only LM with RIGHT-padding (real tokens at front,
         pads appended — see the ``padded[i, : len(ids)]`` fill in
         _get_sae_activations_batched, steering/extract_features.py:1168):
         the causal mask already forbids any position attending to a later
         one, so trailing pad positions cannot influence the attention at
         earlier REAL-token positions. Real-token activations are therefore
         invariant to right-padding in exact arithmetic; in fp32 only
         batched-vs-sequential reduction-order noise (~1e-6, independent of
         padding) remains — well within downstream tolerance — while
         native-vs-nnterp stays exactly bit-identical (same maskless
         forward, §2).
      2. Bit-exact cross-backend parity contract: native MUST be
         byte-identical to the production nnterp backend. nnterp also runs
         maskless raw-HF/SDPA, so threading a mask here (while nnterp stays
         maskless) would break that guarantee. The truncated forward is
         likewise maskless, so native ≡ nnterp bit-identical is preserved.
         Enforced by
         tests/steering/native/conftest_parity.py::assert_bit_identical
         (torch.equal, max_abs == 0.0) and exercised by the slow residual
         gate tests/steering/native/test_native_integration.py
         ::test_native_matches_nnterp_and_tl_residual (the maskless-vs-SDPA
         drift floor is fp32 numerical noise, not a correctness gap).
      3. Pad-position activations are never consumed: they are stripped by
         true sequence length before pooling/SAE-encode — see the
         ``sae_acts[i, : lengths[i]]`` per-doc slice in
         _get_sae_activations_batched (steering/extract_features.py:1234).
      4. Mirrors the maskless contract of _get_activations_nnterp
         (steering/extract_features.py:815, the production backend — no
         attention_mask parameter at all) exactly. NOT _get_activations_tl
         (steering/extract_features.py:777), which threads its own mask.
    """
    if hook_name is None:
        hook_name = f"blocks.{layer}.hook_resid_post"
    # Maskless by design (see docstring §1-4): no attention_mask is threaded
    # into capture(); the causal mask + right-pad invariance + downstream
    # length-strip make it correct, and bit-exact nnterp parity requires it.
    return native_model.capture(tokens, layer, hook_name=hook_name, activation_type=activation_type)


def maybe_gc_cuda(device: t.device, threshold: float = 0.8) -> None:
    """Run gc.collect() + empty_cache() if GPU memory exceeds threshold.

    The nnterp/nnsight tracer retains cyclic references to GPU tensors that
    Python's reference counter cannot free. Only gc.collect() (cycle detection)
    breaks these references. However, gc.collect() is expensive (~4x throughput
    cost if called every batch), so we only trigger it when GPU utilization
    exceeds a configurable fraction of total VRAM.

    Uses ``t.cuda.mem_get_info`` which reports *device-wide* free/total memory,
    correctly accounting for other processes sharing the GPU.

    Args:
        device: Torch device. No-op if not CUDA.
        threshold: Fraction of total VRAM (0.0-1.0). GC triggers when
            device-wide usage / total memory > threshold.
    """
    if device.type != "cuda":
        return
    free, total = t.cuda.mem_get_info(device)
    utilization = 1.0 - (free / total)
    if utilization > threshold:
        gc.collect()
        t.cuda.empty_cache()


def _infer_activation_type_from_hook_name(
    hook_name: str | None,
) -> ActivationType:
    """Infer ``ActivationType`` from a hook-name substring.

    ``None`` keeps the historical residual-stream default (legacy GPT-2
    path). Order matters — ``hook_attn_out`` and ``hook_z`` both indicate
    attention, but the routes downstream differ (post- vs pre-projection).

    Recognized substrings:

    - ``hook_resid`` (any) → residual
    - ``hook_mlp`` (any) → mlp
    - ``hook_z`` (any) → attention (pre-projection)
    - ``hook_attn_out`` (any) → attention (post-projection)
    """
    if hook_name is None:
        return ActivationType.residual
    if "hook_z" in hook_name:
        return ActivationType.attention
    if "hook_attn_out" in hook_name:
        return ActivationType.attention
    if "hook_mlp" in hook_name:
        return ActivationType.mlp
    if "hook_resid" in hook_name:
        return ActivationType.residual
    raise ValueError(
        f"Cannot infer activation type from hook_name {hook_name!r}; "
        f"recognized substrings: 'hook_resid', 'hook_mlp', 'hook_z', 'hook_attn_out'."
    )


def _get_sae_activations(
    model: HookedTransformer,
    sae: SAE,
    tokens: t.Tensor,
    layer: int,
    hook_name: str | None = None,
) -> t.Tensor:
    """Run tokens through model and SAE, return SAE activations.

    Args:
        model: HookedTransformer.
        sae: SAE model.
        tokens: [seq_len] token IDs (long tensor, on device).
        layer: Transformer layer.
        hook_name: TransformerLens hook name. Defaults to resid_pre for backward compat.

    Returns:
        [seq_len, n_features] SAE activation tensor (no grad).
    """
    if hook_name is None:
        hook_name = f"blocks.{layer}.hook_resid_pre"
    device = next(sae.parameters()).device
    tokens_batched = tokens.unsqueeze(0)
    with t.no_grad():
        with record_function("model_forward"):
            activations = None
            native_model = getattr(model, "native_model", None)
            if native_model is not None:
                try:
                    activations = _get_activations_native(
                        native_model,
                        tokens_batched.to(device),
                        layer,
                        activation_type=_infer_activation_type_from_hook_name(
                            hook_name,
                        ),
                        hook_name=hook_name,
                    )
                except ExtractionError as exc:
                    raise _accelerated_backend_error("native", "forward", exc) from exc
            if activations is None:
                nnterp_model = getattr(model, "nnterp_model", None)
                if nnterp_model is not None:
                    try:
                        activations = _get_activations_nnterp(
                            nnterp_model,
                            tokens_batched.to(device),
                            layer,
                            activation_type=_infer_activation_type_from_hook_name(
                                hook_name,
                            ),
                            hook_name=hook_name,
                        )
                    except ExtractionError as exc:
                        raise _accelerated_backend_error("nnterp", "forward", exc) from exc
            if activations is None:
                # No accelerated backend on the model -> explicit
                # backend=transformerlens path (the ONLY TL route).
                try:
                    tl_device = next(model.parameters()).device
                except (AttributeError, StopIteration):
                    tl_device = tokens.device
                activations = _get_activations_tl(
                    model,
                    tokens_batched.to(tl_device),
                    layer,
                    hook_name=hook_name,
                )
            if isinstance(activations, (tuple, list)):
                activations = activations[0]
            if activations.device != device:
                activations = activations.to(device)
        with record_function("sae_encode"):
            # For 4D attention hook_z, squeeze batch dim but keep heads separate
            act = activations.squeeze(0)  # [seq, d_model] or [seq, nh, dh]
            sae_out = sae.encode(act)  # [seq_len, n_features]
        if sae_out.ndim != 2:
            raise RuntimeError(f"SAE output ndim {sae_out.ndim} != 2")
        if sae_out.shape[0] != len(tokens):
            raise RuntimeError(
                f"SAE output seq_len {sae_out.shape[0]} != input tokens {len(tokens)}"
            )
        if sae_out.requires_grad:
            raise RuntimeError("SAE output should not require grad")
        return sae_out


def _get_sae_activations_batched(
    model: HookedTransformer,
    sae: SAE,
    token_id_lists: list[list[int]],
    layer: int,
    nnterp_model: StandardizedTransformer | None = None,
    native_model: NativeCollector | None = None,
    hook_name: str | None = None,
    activation_type: ActivationType = ActivationType.residual,
) -> list[t.Tensor]:
    """Run multiple documents through model + SAE in a single forward pass.

    Pads sequences to the max length in the batch, runs one batched forward
    pass, then strips padding from each result.

    Args:
        model: HookedTransformer.
        sae: SAE model.
        token_id_lists: List of token ID lists, one per document.
        layer: Transformer layer.
        nnterp_model: Optional nnterp backend for faster inference.
        native_model: Optional home-grown native collector. Tried BEFORE
            nnterp; on ExtractionError raises AcceleratedBackendError
            (no silent TransformerLens fallback). TransformerLens runs
            only when no accelerated backend is configured (the explicit
            backend=transformerlens path, both native and nnterp absent).
        hook_name: TransformerLens hook name for activation capture.
        activation_type: Which activation site to capture.

    Returns:
        List of [seq_len_i, n_features] tensors, one per document.
    """
    assert len(token_id_lists) > 0, "token_id_lists must be non-empty"
    assert all(len(ids) > 0 for ids in token_id_lists), "All token lists must be non-empty"
    device = next(sae.parameters()).device
    lengths = [len(ids) for ids in token_id_lists]
    max_seq = max(lengths)

    # SDPA row-packing (S2): when the per-run correctness gate verified packing
    # for this native (model, layer, dtype), pack the ragged batch into ONE
    # sequence via `capture_packed`, skipping the wasted pad-token compute.
    # ALWAYS-ON for native when verified (no per-batch pad-fraction guard); the
    # gate hard-raises `RowPackGateError` if packing isn't ideal-math-equivalent,
    # so a verified flag means packing is proven safe for the run. Residual-only
    # (the native collector's contract). The unpacked path below is retained for
    # non-native backends and as the gate's reference.
    if native_model is not None and getattr(native_model, "_rowpack_verified", False):
        with t.no_grad():
            with record_function("model_forward_packed"):
                try:
                    per_doc_acts = native_model.capture_packed(
                        token_id_lists,
                        layer,
                        hook_name=hook_name,
                        activation_type=activation_type,
                    )
                except ExtractionError as exc:
                    raise _accelerated_backend_error("native", "forward", exc) from exc
            with record_function("sae_encode_packed"):
                # One flat encode (matches the padded path's encode batching),
                # then split per doc — packed acts are already stripped.
                flat_sae = sae.encode(t.cat(per_doc_acts, dim=0))
                packed_out: list[t.Tensor] = []
                offset = 0
                for length in lengths:
                    packed_out.append(flat_sae[offset : offset + length])
                    offset += length
        return packed_out

    # Pad to max-in-batch and build attention mask
    padded = t.zeros(len(token_id_lists), max_seq, dtype=t.long, device=device)
    attention_mask = t.zeros(len(token_id_lists), max_seq, dtype=t.long, device=device)
    for i, ids in enumerate(token_id_lists):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
        attention_mask[i, : len(ids)] = 1

    with t.no_grad():
        with record_function("model_forward_batched"):
            activations = None
            if native_model is not None:
                try:
                    activations = _get_activations_native(
                        native_model,
                        padded,
                        layer,
                        activation_type=activation_type,
                        hook_name=hook_name,
                    )
                except ExtractionError as exc:
                    raise _accelerated_backend_error("native", "forward", exc) from exc
            if activations is None and nnterp_model is not None:
                try:
                    activations = _get_activations_nnterp(
                        nnterp_model,
                        padded,
                        layer,
                        activation_type=activation_type,
                        hook_name=hook_name,
                    )
                except ExtractionError as exc:
                    raise _accelerated_backend_error("nnterp", "forward", exc) from exc
            if activations is None:
                # No accelerated backend configured -> explicit
                # backend=transformerlens path (both models absent). This is
                # the ONLY route to TransformerLens.
                tl_device = next(model.parameters()).device
                activations = _get_activations_tl(
                    model,
                    padded.to(tl_device),
                    layer,
                    hook_name=hook_name,
                    attention_mask=attention_mask.to(tl_device),
                )
                activations = activations.to(device)

            # Handle tuple returns from attention modules
            if isinstance(activations, (tuple, list)):
                activations = activations[0]

        with record_function("sae_encode_batched"):
            if activations.ndim == 4:
                # Attention hook_z: [batch, seq, n_heads, d_head]
                # Flatten batch*seq but keep n_heads, d_head separate for SAE
                batch, seq, n_heads, d_head = activations.shape
                flat_acts = activations.reshape(batch * seq, n_heads, d_head)
                flat_sae = sae.encode(flat_acts)
                sae_acts = flat_sae.view(batch, seq, -1)
            else:
                # Residual / MLP: [batch, seq, d_model]
                flat_acts = activations.view(-1, activations.shape[-1])
                flat_sae = sae.encode(flat_acts)
                sae_acts = flat_sae.view(len(token_id_lists), max_seq, -1)

    # Strip padding: return list of [len_i, n_features] tensors
    return [sae_acts[i, : lengths[i]] for i in range(len(token_id_lists))]


# A long, multi-paragraph essay of distinct coherent English sentences (NOT a
# single token spammed) used to build the compile-gate probe. Tokenized with
# the real gemma tokenizers it is ~1.2k tokens — ≳ the slow Layer-4 gate's
# realistic corpus and ~2.4× gemma-3-1b-pt's published sliding_window (512), so
# the dynamic compiled graph is probed on a PRODUCTION-representative ragged
# batch (mixed sliding + full attention at scale), not a vacuous toy. Kept
# in-module (production code must not import from tests/); the slow Task-7 gate
# certifies real-model numerics on its own realistic corpus.
_COMPILE_PROBE_LONG_DOC = (
    "Language models learn statistical regularities from vast text corpora. "
    "During pretraining the network repeatedly predicts the next token given "
    "every previous one, and through this deceptively simple objective it "
    "gradually internalizes syntax, morphology, semantics, factual world "
    "knowledge, discourse structure, and even rudimentary multi-step "
    "reasoning. The optimization landscape is shaped by the data: rare "
    "constructions are learned slowly, frequent idioms quickly, and the "
    "model's eventual competence is a smeared average over everything it has "
    "read. Researchers who want to understand what such a network has "
    "actually learned cannot simply inspect its billions of weights; the "
    "representations are distributed, superposed, and only loosely aligned "
    "with human concepts. Sparse autoencoders offer one tractable lens. The "
    "idea is to take a dense activation vector from somewhere inside the "
    "transformer and decompose it into a sparse, overcomplete sum of learned "
    "feature directions, each of which ideally corresponds to a single "
    "interpretable concept the model tracks. A single residual-stream "
    "position, read this way, can light up dozens of features at once: one "
    "for the grammatical subject, another for the topic of the sentence, "
    "others for sentiment, register, or the presence of code. Cataloguing "
    "which features fire, where they fire, and how strongly is the daily work "
    "of mechanistic interpretability, and it only yields trustworthy answers "
    "when the activations being decomposed were captured faithfully in the "
    "first place. A subtle bug in the capture path silently poisons every "
    "downstream feature, and no amount of clever analysis recovers from "
    "corrupted inputs. Attention is the mechanism by which a transformer "
    "routes information between positions. In a causal decoder every position "
    "attends only to earlier ones, never to the future, so any padding "
    "appended to the right of a shorter sequence is provably unable to "
    "influence the activations at genuine token positions, a property the "
    "production batching code leans on heavily when it right-pads a ragged "
    "batch to a common length and then strips each row back to its true "
    "length afterwards. Some layers further restrict attention with a sliding "
    "window, allowing each position to see only a fixed number of recent "
    "tokens, while other layers attend globally across the whole context. "
    "Correctly interleaving these two regimes, in exactly the published "
    "pattern, is essential for long-context fidelity, and only a sequence "
    "strictly longer than the window forces the windowed and full causal "
    "masks numerically apart, so only a genuinely long document actually "
    "exercises that boundary. Numerical reproducibility across execution "
    "modes is its own deep subtlety: the same model, run through a different "
    "compiled kernel or a different reduction order, can emit values that "
    "differ in the last few bits, and those tiny differences propagate "
    "through the sparse autoencoder into the reported feature magnitudes. A "
    "faithful compiled forward must therefore stay within a small, explicitly "
    "documented, empirically measured envelope of the trusted eager forward, "
    "verified on realistic ragged production-scale input rather than a "
    "handful of hand-picked synthetic identifiers, which is exactly what this "
    "self-verifying gate exists to guarantee before a single shard is written."
)

# A short doc so the probe batch is genuinely RAGGED (the longer doc dwarfs
# it ⇒ many right-pad positions on row 0 ⇒ the maskless/right-pad/strip path
# is exercised representatively, not as a no-op uniform batch).
_COMPILE_PROBE_SHORT_DOC = "The cat sat on the mat."


def _tokenize_native_gate_probe_docs(tokenizer) -> list[list[int]]:
    """Tokenize the deterministic native-gate probe corpus with ``tokenizer``."""
    return [
        tokenizer(text, padding=False, add_special_tokens=True)["input_ids"]
        for text in (_COMPILE_PROBE_LONG_DOC, _COMPILE_PROBE_SHORT_DOC)
    ]


def _build_rowpack_probe_docs(model_name: str) -> list[list[int]]:
    """The shared native-gate probe as per-doc token-id LISTS.

    Built with the model's OWN gated-HF tokenizer (``resolve_tokenizer`` →
    ``AutoTokenizer.from_pretrained(hf_id)``), ``add_special_tokens=True`` and
    ``padding=False`` (the production worker's chunk tokenization). The compile
    gate consumes these as whole documents. The row-pack gate wraps the same
    corpus in ``RowpackGateProbe`` so it proves production segment rows.
    """
    tokenizer = resolve_tokenizer(model_name)
    return _tokenize_native_gate_probe_docs(tokenizer)


def _build_compile_probe_ids(model_name: str, device: str | t.device) -> t.Tensor:
    """A deterministic, production-shaped ragged probe-id batch on ``device``.

    Built with the model's OWN gated-HF tokenizer (the exact source production
    uses, ``resolve_tokenizer`` → ``AutoTokenizer.from_pretrained(hf_id)``),
    ``add_special_tokens=True`` / ``padding=False`` (mirroring the production
    worker's chunk tokenization), then RIGHT-padded with id 0 to the max length
    EXACTLY as ``_get_sae_activations_batched`` does
    (``padded[i, : len(ids)] = ids``). batch=2 with a long (~1.2k-token) doc
    and a one-line doc ⇒ genuinely ragged, ≳ the slow gate's realistic corpus
    length, > gemma-3's published ``sliding_window`` (512) — so the compiled
    ``dynamic=True`` graph is validated on the SAME maskless / right-pad /
    mixed-sliding-+-full-attention regime production traffic stresses (a toy
    probe would make the gate vacuous exactly where bf16 corruption lives:
    deep gemma-3-bf16, large-magnitude, long-context).
    """
    docs = _build_rowpack_probe_docs(model_name)
    lengths = [len(ids) for ids in docs]
    max_seq = max(lengths)
    # Right-pad with id 0 — byte-identical to _get_sae_activations_batched.
    padded = t.zeros(len(docs), max_seq, dtype=t.long, device=device)
    for i, ids in enumerate(docs):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    return padded


def _build_rowpack_gate_probe(
    model_name: str,
    *,
    segment_size: int,
    context_independent_features: bool,
    ignore_bos_features: bool,
) -> RowpackGateProbe:
    """Build the native row-pack gate probe on the production row surface.

    Context-dependent mode forwards whole docs and pools their fixed-token
    segment rows. Context-independent mode forwards the same per-segment model
    inputs production builds, including per-segment BOS injection when needed,
    and pools only the real segment span after any injected BOS.
    """
    tokenizer = resolve_tokenizer(model_name)
    docs = _tokenize_native_gate_probe_docs(tokenizer)
    precomputed = _precompute_batch_cpu_work(
        [(ids, len(ids)) for ids in docs],
        tokenizer,
        SegmentMode.fixed_tokens,
        segment_size,
    )

    bos_token_id: int | None = None
    if context_independent_features or ignore_bos_features:
        bos_token_id = resolve_bos_token_id(tokenizer)
        if bos_token_id is None:
            raise ValueError(
                "row-pack gate requires tokenizer to define bos_token_id when "
                "context_independent_features or ignore_bos_features is enabled "
                f"(model={model_name})"
            )

    if context_independent_features:
        assert bos_token_id is not None
        ci_inputs = _prepare_context_independent_segment_inputs(
            chunk_token_info=[(ids, len(ids)) for ids in docs],
            chunk_precomputed=precomputed,
            bos_token_id=bos_token_id,
            ignore_bos_features=ignore_bos_features,
        )
        token_id_lists: list[list[int]] = []
        segments_by_sequence: list[list[tuple[int, int]]] = []
        keep_masks_by_sequence: list[tuple[bool, ...] | None] = []
        for ci_input in ci_inputs:
            token_id_lists.append(ci_input.model_input)
            row_start = 1 if ci_input.bos_injected else 0
            row_end = row_start + ci_input.original_length
            segments_by_sequence.append([(row_start, row_end)])
            if ci_input.keep_mask is None:
                keep_masks_by_sequence.append(None)
            elif ci_input.bos_injected:
                keep_masks_by_sequence.append((False,) + ci_input.keep_mask)
            else:
                keep_masks_by_sequence.append(ci_input.keep_mask)
        return RowpackGateProbe(
            token_id_lists=token_id_lists,
            segments_by_sequence=segments_by_sequence,
            keep_masks_by_sequence=keep_masks_by_sequence,
            context_independent_features=True,
        )

    return RowpackGateProbe(
        token_id_lists=docs,
        segments_by_sequence=[segments for segments, _n, _cs, _ce in precomputed],
        keep_masks_by_sequence=[
            tuple(tok != bos_token_id for tok in ids)
            if ignore_bos_features and bos_token_id is not None
            else None
            for ids in docs
        ],
        context_independent_features=False,
    )


def _sparse_proj_from_stripped(
    stripped: list[t.Tensor],
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Per-doc single-segment max-pool → (ids, mags) — the gate's oracle inputs.

    Each ``stripped[i]`` is a per-doc ``[Lᵢ, f]`` SAE-feature tensor; the
    projection mirrors the slow Layer-4 gate's ``_sparse``
    (test_native_integration.py): a single-segment ``max(dim=0)`` pool,
    ``(pooled > 0)`` feature ids, and the co-indexed magnitudes.
    """
    ids: list[t.Tensor] = []
    mags: list[t.Tensor] = []
    for seg in stripped:
        pooled = seg.max(dim=0).values  # single-segment max-pool proxy
        nz = (pooled > 0).nonzero().flatten()
        ids.append(nz.to(t.int32).cpu())
        mags.append(pooled[nz].float().cpu())
    return ids, mags


def _gate_sparse_features(
    sae: SAE, acts: t.Tensor, lengths: list[int]
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """SAE-encode + per-doc strip + single-segment max-pool → (ids, mags).

    The encode/flatten/strip half is byte-identical to the residual branch of
    ``_get_sae_activations_batched`` (steering/extract_features.py:1334-1339):
    ``flat = acts.view(-1, d); feats = sae.encode(flat);
    feats.view(B, S, -1)``, then ``feats[i, : lengths[i]]`` per doc — the EXACT
    transform the slow tests assert through, NOT a re-derived one. The
    (ids, mags) projection (``_sparse_proj_from_stripped``) then mirrors the
    slow Layer-4 gate's ``_sparse`` — so the gate's oracle inputs are produced
    identically to the slow gate's.
    """
    flat = acts.view(-1, acts.shape[-1])
    feats = sae.encode(flat).view(len(lengths), acts.shape[1], -1)
    stripped = [feats[i, : lengths[i]] for i in range(len(lengths))]
    return _sparse_proj_from_stripped(stripped)


def _rowpack_gate_sparse_from_sae_sequences(
    sae_sequences: list[t.Tensor],
    probe: RowpackGateProbe,
    pooling: PoolingMode,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Project SAE activations to the exact row surface extraction writes."""
    assert len(sae_sequences) == len(probe.token_id_lists), (
        f"SAE sequence count {len(sae_sequences)} != probe sequence count "
        f"{len(probe.token_id_lists)}"
    )
    pooled_rows: list[t.Tensor] = []
    for seq_idx, (
        seq_acts,
        segments,
        keep_mask_tuple,
    ) in enumerate(
        zip(
            sae_sequences,
            probe.segments_by_sequence,
            probe.keep_masks_by_sequence,
            strict=True,
        )
    ):
        assert seq_acts.shape[0] == len(probe.token_id_lists[seq_idx]), (
            f"rowpack gate seq {seq_idx}: activations length {seq_acts.shape[0]} "
            f"!= token length {len(probe.token_id_lists[seq_idx])}"
        )
        keep_mask = None
        if keep_mask_tuple is not None:
            keep_mask = t.tensor(keep_mask_tuple, dtype=t.bool, device=seq_acts.device)
            assert keep_mask.shape[0] == seq_acts.shape[0], (
                f"rowpack gate seq {seq_idx}: keep_mask length "
                f"{keep_mask.shape[0]} != activation rows {seq_acts.shape[0]}"
            )
            if bool(keep_mask.all()):
                keep_mask = None

        if probe.context_independent_features:
            assert len(segments) == 1, (
                "context-independent rowpack gate expects one pooled row per "
                f"forwarded sequence, got {len(segments)} for seq {seq_idx}"
            )
            start, end = segments[0]
            segment_keep_mask = keep_mask[start:end] if keep_mask is not None else None
            pooled_rows.append(
                pool_one_segment(seq_acts[start:end], pooling, keep_mask=segment_keep_mask)
            )
            continue

        pooled = pool_segments_batched_gpu(
            seq_acts,
            segments,
            pooling,
            SegmentMode.fixed_tokens,
            keep_mask=keep_mask,
        )
        pooled_rows.extend(pooled.unbind(dim=0))

    if not pooled_rows:
        return [], []
    return extract_sparse_gpu(t.stack(pooled_rows, dim=0))


def _rowpack_gate_sparse_features(
    sae: SAE,
    acts: t.Tensor,
    probe: RowpackGateProbe,
    pooling: PoolingMode,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """SAE-encode a padded native capture once, then project production rows."""
    lengths = [len(ids) for ids in probe.token_id_lists]
    flat = acts.view(-1, acts.shape[-1])
    feats = sae.encode(flat).view(len(lengths), acts.shape[1], -1)
    stripped = [feats[i, : lengths[i]] for i in range(len(lengths))]
    return _rowpack_gate_sparse_from_sae_sequences(stripped, probe, pooling)


def _rowpack_gate_sparse_features_packed(
    sae: SAE,
    per_sequence_acts: list[t.Tensor],
    probe: RowpackGateProbe,
    pooling: PoolingMode,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """SAE-encode packed native residuals once, then project production rows."""
    lengths = [a.shape[0] for a in per_sequence_acts]
    flat = t.cat(per_sequence_acts, dim=0)
    feats = sae.encode(flat)
    stripped: list[t.Tensor] = []
    offset = 0
    for length in lengths:
        stripped.append(feats[offset : offset + length])
        offset += length
    return _rowpack_gate_sparse_from_sae_sequences(stripped, probe, pooling)


def _format_compile_gate_envelope(
    measurements: dict[str, float],
    *,
    is_bf16: bool,
) -> tuple[str, str]:
    """Pure formatter for the ideal-math-relative compile-gate envelope.

    Returns ``(measured, gate)``: the human-readable measured-vs-gate strings
    the PASS ``logger.info`` and the fail-raise ``CompileGateError`` both
    reuse (so their f-string skeletons are not duplicated).

    ``measurements`` is the dict produced by
    ``compiled_passes_ideal_math_gate`` — the EXACT same dict the gate's
    predicate returned — so the printed envelope can never drift from the
    enforced decision.  Both ``measured`` and ``gate`` carry the four
    documented numbers (``j_baseline_vs_ref``, ``j_compiled_vs_ref``,
    ``slack``, ``margin``) plus the dtype tag.
    """
    j_baseline = measurements["j_baseline_vs_ref"]
    j_compiled = measurements["j_compiled_vs_ref"]
    slack = measurements["slack"]
    margin = measurements["margin"]
    dtype_name = "bf16" if is_bf16 else "fp32"
    measured = (
        f"j_baseline_vs_ref={j_baseline:.6f} "
        f"j_compiled_vs_ref={j_compiled:.6f} "
        f"slack={slack:.6f} margin={margin:.6f}"
    )
    gate = (
        f"j_compiled_vs_ref>=j_baseline_vs_ref-slack "
        f"({j_baseline:.6f}-{slack:.6f}={j_baseline - slack:.6f}) "
        f"[{dtype_name}]"
    )
    return measured, gate


def _native_compile_gate(
    native_model: NativeCollector,
    sae: SAE,
    *,
    probe_input_ids: t.Tensor,
    model_dtype: t.dtype,
) -> None:
    """Arm+probe the compiled forward; ACTIVATE it iff the compiled config is
    no worse than the bf16-eager baseline relative to the fp32-eager
    (ideal-math) reference, within ``BASELINE_RELATIVE_SLACK``.

    Fail-loud, and structurally incapable of leaving the collector in a
    compiled-but-unverified state:

      * pass -> ``native_model.set_compiled_active(True)`` + ``logger.info``
                (measurements: j_baseline_vs_ref, j_compiled_vs_ref, slack,
                margin);
      * fail -> ``set_compiled_active(False)`` (already restored by step 4),
                then ``raise CompileGateError`` (message carries the same four
                measurements).

    There is no eager-fallback: a gate-fail HARD-RAISES so an extraction
    cannot silently emit potentially-corrupted activations. The ONLY PASS
    path is the predicate returning True; the ONLY FAIL path is the
    explicit raise — no try/except masking, no silent fallback. This is the
    "structurally incapable of leaving compiled-but-unverified" invariant.

    The pass/fail decision is the production GATE predicate
    ``steering.feature_equivalence.compiled_passes_ideal_math_gate`` — an
    ideal-math-relative check anchored on the fp32-eager reference. Empirical
    evidence (4090, FineWeb-Edu shard 00388, gemma-3-1b-pt @7 bf16) showed
    that the prior bf16-eager-relative absolute-threshold gate spuriously
    rejected the emF dyn compiled config (j=0.990) that is in fact
    CLOSER to ideal-math than bf16-eager itself (j=0.989).  The new gate uses
    fp32-eager as the reference, and the SLACK band cleanly admits compiled
    configs no worse than bf16-eager + noise while still rejecting the
    known-corrupting default-Inductor case (~0.019 worse than baseline).

    Steps:
      1. Build a fresh fp32 reference: load the model in fp32, cast the SAE
         to fp32, capture features. Drop the fp32 model/sae references
         immediately after capture so the bf16 probes do not co-resident a
         fresh fp32 stack (~+4 GB transient is acceptable; we explicitly
         ``gc.collect()`` + ``torch.cuda.empty_cache()`` before step 3).
      2. ``enable_compiled()`` arms the compiled base (idempotent).
      3. ``set_compiled_active(False)``; capture the bf16-eager baseline
         features.
      4. ``set_compiled_active(True)``; capture the compiled features inside a
         ``try/finally`` that restores ``set_compiled_active(False)`` — the
         step that preserves the never-compiled-but-unverified invariant.
      5. Apply ``compiled_passes_ideal_math_gate``: on PASS,
         ``set_compiled_active(True)`` + ``logger.info``; on FAIL, raise
         ``CompileGateError``.

    Failure modes raised as ``CompileGateError`` with a clear message:
      * SAE class does not support ``.to(torch.float32)``
        (e.g. ``.to`` raises AttributeError/TypeError);
      * fp32 model build fails (typically OOM at load time);
      * fp32 reference returns empty features (corpus too sparse — gate
        cannot decide).
    """
    lengths = [int((row != 0).sum().item()) for row in probe_input_ids]
    hook_name = f"blocks.{native_model.truncated_layer}.hook_resid_post"

    # --- Step 1: fp32-eager reference (ideal-math approximation). --------- #
    # Build a fresh, untruncated-then-truncated fp32 model + fp32 SAE. The
    # device matches the bf16 model so probe inputs do not need to move; the
    # truncated layer matches so the captured residual is at the same depth.
    ref_device = native_model.device
    try:
        ref_native = build_native_model(
            native_model.model_name,
            ref_device,
            t.float32,
            native_model.truncated_layer,
        )
    except ModelWeightLoadError:
        raise  # #549 fatal — never re-typed to CompileGateError
    except Exception as e:
        raise CompileGateError(
            f"ideal-math gate FAILED to build fp32 reference for "
            f"{native_model.model_name}@{native_model.truncated_layer}: "
            f"{type(e).__name__}: {e} (typically OOM at load time); the "
            f"ideal-math reference is required and the gate cannot proceed"
        ) from e
    if ref_native is None:
        raise CompileGateError(
            f"ideal-math gate FAILED: build_native_model returned None for "
            f"the fp32 reference of {native_model.model_name}@"
            f"{native_model.truncated_layer}; the ideal-math reference is "
            f"required and the gate cannot proceed"
        )
    # Cast the SAE to fp32. Fail loud if the SAE class does not support it.
    try:
        ref_sae = sae.to(t.float32)
    except Exception as e:
        # Tear down the ref_native we just built before raising.
        del ref_native
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
        raise CompileGateError(
            f"ideal-math gate FAILED: SAE class "
            f"{type(sae).__name__} does not support .to(torch.float32) "
            f"({type(e).__name__}: {e}); the ideal-math gate requires an "
            f"fp32 SAE and cannot proceed"
        ) from e
    # The production SAE is now fp32 (``sae.to`` is in-place — ``ref_sae`` IS
    # ``sae``). ANY exception before the restore below MUST put the production
    # dtype back, else a caller that catches the gate's raise would reuse a
    # silently-fp32 SAE. The fp32 ref ``capture()`` is the gate's most
    # OOM-prone op (the docstring anticipates OOM here), so the restore +
    # ref_native teardown live in a ``finally`` that runs on success AND raise.
    try:
        ref_acts = ref_native.capture(
            probe_input_ids,
            ref_native.truncated_layer,
            hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
        ref_ids, _ = _gate_sparse_features(ref_sae, ref_acts, lengths)
    finally:
        # Cast the production SAE BACK to model_dtype (we only borrowed it at
        # fp32 for the ref capture) and drop the fp32 model so the bf16
        # baseline + compiled probes do not co-resident the fp32 stack.
        sae.to(model_dtype)
        del ref_native
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
    del ref_sae, ref_acts
    # Non-vacuity: ideal-math ref must produce SOME features or the predicate
    # would compare against an empty set (trivially j=1 vs anything else =
    # 0 → garbage decision).
    if not any(seg.numel() > 0 for seg in ref_ids):
        raise CompileGateError(
            f"ideal-math gate FAILED: the fp32-eager reference produced no "
            f"features for {native_model.model_name}@"
            f"{native_model.truncated_layer} on the probe corpus "
            f"(corpus too sparse?); cannot decide gate without a non-empty "
            f"reference"
        )

    # --- Step 2: arm the compiled forward (idempotent; does NOT activate). - #
    native_model.enable_compiled()

    # --- Step 3: bf16-eager baseline. ------------------------------------- #
    native_model.set_compiled_active(False)
    baseline_acts = native_model.capture(
        probe_input_ids,
        native_model.truncated_layer,
        hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    baseline_ids, _ = _gate_sparse_features(sae, baseline_acts, lengths)

    # --- Step 4: compiled candidate. Restore eager IMMEDIATELY in `finally` -
    # so the only code path that can leave the collector compiled-active is
    # the explicit step-5 pass branch — preserves S1's never-half-verified
    # invariant even if the compiled probe throws mid-call.
    native_model.set_compiled_active(True)
    try:
        compiled_acts = native_model.capture(
            probe_input_ids,
            native_model.truncated_layer,
            hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
    finally:
        native_model.set_compiled_active(False)
    compiled_ids, _ = _gate_sparse_features(sae, compiled_acts, lengths)

    # --- Step 5: ideal-math-relative decision. ---------------------------- #
    passed, measurements = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=BASELINE_RELATIVE_SLACK,
    )
    is_bf16 = model_dtype == t.bfloat16
    dtype_name = "bf16" if is_bf16 else "fp32"
    measured, gate = _format_compile_gate_envelope(measurements, is_bf16=is_bf16)

    # The ONLY set_compiled_active(True) on a PASS path; the ONLY exit on
    # FAIL is a loud raise (no silent eager-fallback, no try/except mask).
    if passed:
        native_model.set_compiled_active(True)
        logger.info(
            f"[compile-gate] {native_model.model_name}@"
            f"{native_model.truncated_layer} {dtype_name}: compiled forward "
            f"PASSED the ideal-math-relative gate "
            f"(measured {measured} vs gate {gate}) — compiled path ACTIVATED"
        )
        return

    # Failed — collector is already EAGER (step-4 finally). Hard-raise so the
    # run cannot silently proceed on a potentially-corrupted compiled path
    # AND cannot silently fall back to eager (which would mask a real
    # correctness regression). The four measurements are surfaced in the
    # message so a future operator can reconstruct the decision from logs.
    raise CompileGateError(
        f"compile-gate FAIL for {native_model.model_name}@"
        f"{native_model.truncated_layer} {dtype_name}: compiled forward "
        f"degrades vs the fp32-eager (ideal-math) reference MORE than "
        f"bf16-eager itself does, by more than slack="
        f"{measurements['slack']:.6f} "
        f"(measured {measured} vs gate {gate}); refusing to emit "
        f"potentially-corrupted activations and refusing to silently "
        f"fall back to eager (the gate is mandatory — fix the cell or "
        f"add a targeted graph-break before shipping)"
    )


def _native_rowpack_gate(
    native_model: NativeCollector,
    sae: SAE,
    *,
    probe: RowpackGateProbe,
    pooling: PoolingMode,
    model_dtype: t.dtype,
) -> None:
    """Probe the PACKED forward; ACTIVATE row-packing iff the packed forward is
    no worse than the unpacked ``model_dtype`` path relative to the fp32-eager
    (ideal-math) reference, within ``BASELINE_RELATIVE_SLACK``.

    The row-packing twin of ``_native_compile_gate`` — same ideal-math-relative
    predicate (``compiled_passes_ideal_math_gate``), same fp32-reference build +
    SAE-dtype-restore ``try/finally``, same fail-loud no-fallback contract — but
    it compares PACKED vs UNPACKED instead of compiled vs eager:

      * reference = fp32-eager UNPACKED  (ideal-math approximation)
      * baseline  = ``model_dtype`` UNPACKED (the padded path packing replaces)
      * candidate = ``model_dtype`` PACKED   (``capture_packed``)

    The probe projects the SAME rows production extraction emits: fixed-token
    segment rows in context-dependent mode, and per-segment context-independent
    rows with injected BOS outside the pooled span plus optional natural-BOS
    masking. Note the predicate's ``baseline_*`` args carry the UNPACKED
    features and its ``compiled_*`` args carry the PACKED features (the
    predicate is generic "candidate vs baseline relative to reference"; the
    names are historical).

    pass -> ``native_model.set_rowpack_verified(True)`` + ``logger.info``;
    fail -> ``raise RowPackGateError``. Row-packing is ALWAYS-ON with no
    unpacked production fallback, so a fail HARD-RAISES rather than silently
    emit row-pack-corrupted activations. The verified flag is cleared at entry
    so a rerun cannot preserve stale pass state after a later failure. The fp32
    reference ``capture`` is the most OOM-prone op, so the SAE-dtype restore +
    ref teardown live in a ``finally`` that runs on success AND raise (the same
    silent-fp32-SAE guard ``_native_compile_gate`` carries).
    """
    # A rerun after a previous pass must not leave stale verified state behind
    # if any later reference, baseline, packed, or predicate step fails.
    native_model.set_rowpack_verified(False)

    token_id_lists = probe.token_id_lists
    lengths = [len(d) for d in token_id_lists]
    hook_name = f"blocks.{native_model.truncated_layer}.hook_resid_post"
    device = native_model.device
    # The padded form for the unpacked baseline + fp32 reference — right-pad
    # id 0 EXACTLY as ``_get_sae_activations_batched`` / the compile probe does.
    padded = t.zeros(len(token_id_lists), max(lengths), dtype=t.long, device=device)
    for i, ids in enumerate(token_id_lists):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    n_probe_rows = sum(len(segments) for segments in probe.segments_by_sequence)
    ignore_bos_features = any(mask is not None for mask in probe.keep_masks_by_sequence)

    # --- Step 1: fp32-eager UNPACKED reference (ideal-math). -------------- #
    try:
        ref_native = build_native_model(
            native_model.model_name,
            device,
            t.float32,
            native_model.truncated_layer,
        )
    except ModelWeightLoadError:
        raise  # #549 fatal — never re-typed to RowPackGateError
    except Exception as e:
        raise RowPackGateError(
            f"rowpack gate FAILED to build fp32 reference for "
            f"{native_model.model_name}@{native_model.truncated_layer}: "
            f"{type(e).__name__}: {e} (typically OOM at load time); the "
            f"ideal-math reference is required and the gate cannot proceed"
        ) from e
    if ref_native is None:
        raise RowPackGateError(
            f"rowpack gate FAILED: build_native_model returned None for the "
            f"fp32 reference of {native_model.model_name}@"
            f"{native_model.truncated_layer}; the ideal-math reference is "
            f"required and the gate cannot proceed"
        )
    try:
        ref_sae = sae.to(t.float32)
    except Exception as e:
        del ref_native
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
        raise RowPackGateError(
            f"rowpack gate FAILED: SAE class {type(sae).__name__} does not "
            f"support .to(torch.float32) ({type(e).__name__}: {e}); the "
            f"ideal-math gate requires an fp32 SAE and cannot proceed"
        ) from e
    try:
        ref_acts = ref_native.capture(
            padded,
            ref_native.truncated_layer,
            hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
        ref_ids, _ = _rowpack_gate_sparse_features(ref_sae, ref_acts, probe, pooling)
    finally:
        sae.to(model_dtype)
        del ref_native
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
    del ref_sae, ref_acts
    if not any(seg.numel() > 0 for seg in ref_ids):
        raise RowPackGateError(
            f"rowpack gate FAILED: the fp32-eager reference produced no "
            f"features for {native_model.model_name}@"
            f"{native_model.truncated_layer} on the probe corpus "
            f"(corpus too sparse?); cannot decide gate without a non-empty "
            f"reference"
        )

    # --- Step 2: model_dtype UNPACKED baseline (the padded path). --------- #
    baseline_acts = native_model.capture(
        padded,
        native_model.truncated_layer,
        hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    baseline_ids, _ = _rowpack_gate_sparse_features(sae, baseline_acts, probe, pooling)

    # --- Step 3: model_dtype PACKED candidate. ---------------------------- #
    packed_acts = native_model.capture_packed(
        token_id_lists,
        native_model.truncated_layer,
        hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    packed_ids, _ = _rowpack_gate_sparse_features_packed(sae, packed_acts, probe, pooling)

    # --- Step 4: ideal-math-relative decision (baseline=unpacked, ---------- #
    #             compiled=packed). --------------------------------------- #
    passed, measurements = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=packed_ids,
        slack=BASELINE_RELATIVE_SLACK,
    )
    is_bf16 = model_dtype == t.bfloat16
    dtype_name = "bf16" if is_bf16 else "fp32"
    measured, gate = _format_compile_gate_envelope(measurements, is_bf16=is_bf16)
    j_packed_vs_unpacked, count_rdiff_packed_vs_unpacked = feature_equivalence_metrics(
        baseline_ids, packed_ids
    )
    diagnostics = (
        f"j_packed_vs_unpacked={j_packed_vs_unpacked:.6f} "
        f"count_rdiff_packed_vs_unpacked="
        f"{count_rdiff_packed_vs_unpacked:.6f} "
        f"n_probe_sequences={len(token_id_lists)} "
        f"n_probe_rows={n_probe_rows} "
        f"max_probe_len={max(lengths)} "
        f"pooling={pooling.value} "
        f"context_independent_features="
        f"{probe.context_independent_features} "
        f"ignore_bos_features={ignore_bos_features}"
    )

    if passed:
        native_model.set_rowpack_verified(True)
        logger.info(
            f"[rowpack-gate] {native_model.model_name}@"
            f"{native_model.truncated_layer} {dtype_name}: packed forward "
            f"PASSED the ideal-math-relative gate "
            f"(measured {measured} vs gate {gate}; {diagnostics}) "
            f"— row-packing ACTIVATED"
        )
        return

    raise RowPackGateError(
        f"rowpack-gate FAIL for {native_model.model_name}@"
        f"{native_model.truncated_layer} {dtype_name}: packed forward degrades "
        f"vs the fp32-eager (ideal-math) reference MORE than unpacked "
        f"{dtype_name} itself does, by more than slack="
        f"{measurements['slack']:.6f} (measured {measured} vs gate {gate}; "
        f"{diagnostics}); "
        f"refusing to emit potentially row-pack-corrupted activations "
        f"and refusing to silently fall back to the unpacked path "
        f"(the gate is mandatory — packing is always-on)"
    )


def _run_compile_gate(
    cfg: DictConfig,
    native_model: NativeCollector | None,
    sae: SAE,
    device: str | t.device,
    model_dtype: t.dtype,
) -> None:
    """Mandatory self-verifying torch.compile gate. Strictly downstream of
    _build_accelerated_backend and NATIVE-ONLY: it turns the already-correct
    eager native collector into a verified-compiled one, or hard-aborts the
    run if the compiled path fails the SAE feature-equivalence oracle.

    No-op only when ``native_model is None`` (i.e. the accelerated backend
    did not dispatch native for this run — e.g. a non-truncatable model or
    a non-resid_post hook). When native IS in play, the gate ALWAYS runs;
    there is no opt-out and no eager-fallback (S3 lifted every currently-
    tested cell over the 0.99 gate, so silent fallback became failure-
    masking, not safety). The probe is built with the model's real
    tokenizer at the production dtype the SAE was just converted to so the
    dtype-aware oracle matches production precision exactly.
    """
    if native_model is not None:
        probe_ids = _build_compile_probe_ids(cfg.model, device)
        _native_compile_gate(
            native_model,
            sae,
            probe_input_ids=probe_ids,
            model_dtype=model_dtype,
        )


def _run_rowpack_gate(
    cfg: DictConfig,
    native_model: NativeCollector | None,
    sae: SAE,
    model_dtype: t.dtype,
    *,
    pooling: PoolingMode,
    segment_size: int,
    context_independent_features: bool,
    ignore_bos_features: bool,
) -> None:
    """Mandatory self-verifying SDPA row-packing gate. NATIVE-ONLY, run
    strictly AFTER ``_run_compile_gate`` so the gate's unpacked baseline and
    packed candidate share the run's compile state (clean packing-vs-unpacked
    comparison, no compile/packing confound).

    Proves the packed forward (``capture_packed``) is ideal-math-equivalent to
    the unpacked path for this (model, layer, dtype) and ACTIVATES always-on
    packing (``set_rowpack_verified(True)`` — consumed by
    ``_get_sae_activations_batched``), or HARD-ABORTS the run
    (``RowPackGateError``) if not. No opt-out, no per-batch pad-fraction guard,
    no fallback: packing is always-on once verified. No-op only when
    ``native_model is None``. Probe = the production-representative ragged
    corpus wrapped into the exact row surface extraction emits for the resolved
    runtime settings. ``device`` is kept for call-site parity with
    ``_run_compile_gate``; the gate builds its padded baseline on
    ``native_model.device`` internally.
    """
    if native_model is not None:
        probe = _build_rowpack_gate_probe(
            cfg.model,
            segment_size=segment_size,
            context_independent_features=context_independent_features,
            ignore_bos_features=ignore_bos_features,
        )
        _native_rowpack_gate(
            native_model,
            sae,
            probe=probe,
            pooling=pooling,
            model_dtype=model_dtype,
        )


def _context_independent_forward_and_pool_batch(
    model: HookedTransformer,
    sae: SAE,
    batch: list[ContextIndependentSegmentInput],
    layer: int,
    pooling: PoolingMode,
    hook_name: str | None,
    activation_type: ActivationType | None,
    nnterp_model: StandardizedTransformer | None,
    native_model: NativeCollector | None = None,
) -> t.Tensor:
    """Run one forward pass on a batch of context-independent segments; apply BOS drop and pool.

    Returns [len(batch), n_features] on the SAE's device. Asserts per-sequence
    activation length parity to catch padding/trim bugs immediately.
    """
    assert len(batch) >= 1, "batch must be non-empty"
    model_inputs = [ci_input.model_input for ci_input in batch]
    # _get_sae_activations_batched handles variable lengths with attention masks
    # and returns list of [seq_len_i, n_features] unpadded tensors.
    sae_acts_list = _get_sae_activations_batched(
        model,
        sae,
        model_inputs,
        layer=layer,
        nnterp_model=nnterp_model,
        native_model=native_model,
        hook_name=hook_name,
        activation_type=activation_type if activation_type is not None else ActivationType.residual,
    )
    pooled_rows = []
    for ci_input, acts in zip(batch, sae_acts_list, strict=True):
        expected_len = ci_input.original_length + (1 if ci_input.bos_injected else 0)
        assert acts.shape[0] == expected_len, (
            f"Context-independent activation length mismatch for (doc={ci_input.doc_idx}, "
            f"seg={ci_input.seg_idx}): got {acts.shape[0]}, expected {expected_len} "
            f"(original_length={ci_input.original_length}, bos_injected={ci_input.bos_injected})"
        )
        real_acts = acts[1:] if ci_input.bos_injected else acts
        assert real_acts.shape[0] == ci_input.original_length, (
            f"Post-drop length {real_acts.shape[0]} != original_length "
            f"{ci_input.original_length} for (doc={ci_input.doc_idx}, seg={ci_input.seg_idx})"
        )
        seg_keep_mask = None
        if ci_input.keep_mask is not None:
            seg_keep_mask = t.tensor(ci_input.keep_mask, dtype=t.bool, device=real_acts.device)
        pooled_rows.append(pool_one_segment(real_acts, pooling, keep_mask=seg_keep_mask))
    return t.stack(pooled_rows, dim=0)


@beartype
@dataclass(frozen=True)
class AutoBatchSearchConfig:
    """Unary domains for the extraction auto-tuner's search surface."""

    max_gpu_batch_size: PositiveInt
    max_cpu_batch_size: PositiveInt
    min_gpu_batch_size: PositiveInt
    min_cpu_batch_size: PositiveInt


def find_optimal_batch_sizes(
    model,
    sae,
    layer: int,
    context_window: int,
    nnterp_model=None,
    native_model: NativeCollector | None = None,
    max_gpu_batch_size: int = 512,
    max_cpu_batch_size: int = 512,
    min_gpu_batch_size: int = 1,
    min_cpu_batch_size: int = 1,
    n_trials: int = 3,
    n_docs: int = 32,
    pooling: PoolingMode = PoolingMode.mean_pool,
    segment_size: int = 128,
    gc_mem_threshold: float = 0.95,
    hook_name: str | None = None,
    activation_type: ActivationType = ActivationType.residual,
    parquet_paths: list[str] | None = None,
    context_independent_features: bool = False,
    ignore_bos_features: bool = False,
) -> tuple[PositiveInt, PositiveInt]:
    """Find optimal (gpu_batch_size, cpu_batch_size) via 2D grid search.

    Tests all power-of-2 (gpu_bs, cpu_bs) pairs where gpu_bs <= cpu_bs,
    profiling end-to-end process_shard throughput at each combination.

    Args:
        model: HookedTransformer.
        sae: SAE model.
        layer: Residual stream layer.
        context_window: Token sequence length.
        nnterp_model: Optional nnterp backend.
        native_model: Optional native collector backend.
        max_gpu_batch_size: Upper bound for GPU batch search.
        max_cpu_batch_size: Upper bound for CPU batch search.
        min_gpu_batch_size: Lower bound for GPU batch search.
        min_cpu_batch_size: Lower bound for CPU batch search.
        n_trials: Number of timed runs per combo (takes median).
        n_docs: Number of docs in benchmark parquet.
        pooling: Pooling mode.
        segment_size: Segment size.
        gc_mem_threshold: GC threshold for the profiled runs.
        parquet_paths: Optional list of parquet file paths to sample real
            documents from. When provided, n_docs documents are randomly
            sampled from the first shard. Falls back to synthetic data on
            error or when None/empty.

    Returns:
        (gpu_batch_size, cpu_batch_size) tuple, both powers of 2.

    Raises:
        BeartypeCallHintParamViolation: if a batch-size bound is non-positive,
            non-integral, or a ``bool``.
        ValueError: if a min/max range is empty.
    """
    search = AutoBatchSearchConfig(
        max_gpu_batch_size=cast(Any, max_gpu_batch_size),
        max_cpu_batch_size=cast(Any, max_cpu_batch_size),
        min_gpu_batch_size=cast(Any, min_gpu_batch_size),
        min_cpu_batch_size=cast(Any, min_cpu_batch_size),
    )
    max_gpu_batch_size = search.max_gpu_batch_size
    max_cpu_batch_size = search.max_cpu_batch_size
    min_gpu_batch_size = search.min_gpu_batch_size
    min_cpu_batch_size = search.min_cpu_batch_size
    if min_gpu_batch_size > max_gpu_batch_size:
        raise ValueError(
            f"min_gpu_batch_size ({min_gpu_batch_size}) must be "
            f"<= max_gpu_batch_size ({max_gpu_batch_size})"
        )
    if min_cpu_batch_size > max_cpu_batch_size:
        raise ValueError(
            f"min_cpu_batch_size ({min_cpu_batch_size}) must be "
            f"<= max_cpu_batch_size ({max_cpu_batch_size})"
        )

    # Enumerate power-of-2 candidates within [min, max] before touching the
    # model/SAE so invalid ranges fail cleanly during CLI validation.
    # GPU: every power of 2 (stride 1 on exponent)
    gpu_candidates = [
        2**i
        for i in range(max_gpu_batch_size.bit_length())
        if min_gpu_batch_size <= 2**i <= max_gpu_batch_size
    ]
    if not gpu_candidates:
        raise ValueError(
            "No GPU batch-size candidates in range: expected at least one "
            f"power of 2 between min_gpu_batch_size={min_gpu_batch_size} "
            f"and max_gpu_batch_size={max_gpu_batch_size}"
        )

    # CPU: every other power of 2 (stride 2 on exponent) — coarser since
    # CPU batch size has less impact on throughput than GPU batch size.
    cpu_start_exp = max(0, (min_cpu_batch_size - 1).bit_length())
    cpu_candidates = [
        2**i
        for i in range(cpu_start_exp, max_cpu_batch_size.bit_length(), 2)
        if min_cpu_batch_size <= 2**i <= max_cpu_batch_size
    ]
    if not cpu_candidates:
        raise ValueError(
            "No CPU batch-size candidates in range: expected at least one "
            f"power of 2 between min_cpu_batch_size={min_cpu_batch_size} "
            f"and max_cpu_batch_size={max_cpu_batch_size}"
        )

    # Only test pairs where gpu_batch_size <= cpu_batch_size.
    valid_pairs = [
        (gpu_bs, cpu_bs)
        for gpu_bs, cpu_bs in itertools.product(gpu_candidates, cpu_candidates)
        if gpu_bs <= cpu_bs
    ]
    if not valid_pairs:
        raise ValueError(
            "No valid batch-size candidate pairs: require gpu_batch_size <= "
            "cpu_batch_size, but every GPU candidate is larger than every CPU "
            f"candidate (GPU candidates={gpu_candidates}, "
            f"CPU candidates={cpu_candidates})"
        )

    device = next(sae.parameters()).device
    # Build benchmark parquet — prefer real data, fall back to synthetic
    tmp_dir = tempfile.mkdtemp(prefix="auto_batch_2d_")
    parquet_path = os.path.join(tmp_dir, "tuning_shard.parquet")

    if parquet_paths:
        # Sample n_docs random documents from the first real shard
        try:
            table = pq.read_table(parquet_paths[0], columns=["text"])
            all_texts = table.column("text").to_pylist()
            if len(all_texts) >= n_docs:
                sampled_texts = random.sample(all_texts, n_docs)
            else:
                sampled_texts = all_texts
            logger.info(
                f"Auto-tuner: sampled {len(sampled_texts)} docs from "
                f"{Path(parquet_paths[0]).name} (random sample)"
            )
            pq.write_table(pa.table({"text": sampled_texts}), parquet_path)
        except Exception as e:
            logger.warning(
                f"Auto-tuner: failed to read parquet ({e}), falling back to synthetic data"
            )
            base_text = "The quick brown fox jumps over the lazy dog. " * 50
            texts = [base_text[: 200 + i * 40] for i in range(n_docs)]
            pq.write_table(pa.table({"text": texts}), parquet_path)
    else:
        # Synthetic fallback
        base_text = "The quick brown fox jumps over the lazy dog. " * 50
        texts = [base_text[: 200 + i * 40] for i in range(n_docs)]
        pq.write_table(pa.table({"text": texts}), parquet_path)

    # Warmup
    warmup_ids = [list(range(min(context_window, 32)))]
    _get_sae_activations_batched(
        model,
        sae,
        warmup_ids,
        layer=layer,
        nnterp_model=nnterp_model,
        native_model=native_model,
        hook_name=hook_name,
        activation_type=activation_type,
    )
    if device.type == "cuda":
        t.cuda.synchronize()

    best_gpu_bs = min_gpu_batch_size
    best_cpu_bs = min_cpu_batch_size
    best_throughput = 0.0

    logger.info(
        f"2D grid search: {len(gpu_candidates)} GPU x {len(cpu_candidates)} CPU candidates ({len(valid_pairs)} valid pairs)"
    )

    for gpu_bs, cpu_bs in valid_pairs:
        try:
            if device.type == "cuda":
                t.cuda.empty_cache()

            times = []
            for trial in range(n_trials):
                output_dir = os.path.join(tmp_dir, f"out_g{gpu_bs}_c{cpu_bs}_t{trial}")
                os.makedirs(output_dir, exist_ok=True)

                if device.type == "cuda":
                    t.cuda.synchronize()
                start = time.perf_counter()
                process_shard(
                    parquet_path,
                    model,
                    sae,
                    layer=layer,
                    output_dir=output_dir,
                    context_window=context_window,
                    pooling=pooling,
                    gpu_batch_size=gpu_bs,
                    cpu_batch_size=cpu_bs,
                    gc_mem_threshold=gc_mem_threshold,
                    nnterp_model=nnterp_model,
                    native_model=native_model,
                    # The tuning parquet is rebuilt text-only (no turn_char_starts).
                    segment_mode=SegmentMode.fixed_tokens,
                    segment_size=segment_size,
                    hook_name=hook_name,
                    activation_type=activation_type,
                    context_independent_features=context_independent_features,
                    ignore_bos_features=ignore_bos_features,
                )
                if device.type == "cuda":
                    t.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Clear cache after every trial to avoid fragmentation
                if device.type == "cuda":
                    t.cuda.empty_cache()

            median_time = sorted(times)[len(times) // 2]
            throughput = n_docs / median_time

            logger.info(
                f"  GPU={gpu_bs:>4d} CPU={cpu_bs:>4d}: "
                f"{median_time * 1000:8.2f} ms | {throughput:8.1f} docs/s"
            )

            if throughput > best_throughput:
                best_throughput = throughput
                best_gpu_bs = gpu_bs
                best_cpu_bs = cpu_bs

        except (RuntimeError, t.cuda.OutOfMemoryError):
            logger.warning(f"  GPU={gpu_bs:>4d} CPU={cpu_bs:>4d}: OOM, skipping")
            if device.type == "cuda":
                t.cuda.empty_cache()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if best_throughput == 0.0:
        raise RuntimeError(
            f"All {len(valid_pairs)} batch size combinations failed (OOM). "
            f"Reduce max_gpu_batch_size (was {max_gpu_batch_size}) or "
            f"max_cpu_batch_size (was {max_cpu_batch_size})."
        )

    logger.info(
        f"Auto batch sizes: gpu={best_gpu_bs}, cpu={best_cpu_bs} ({best_throughput:.1f} docs/s)"
    )
    logger.info(
        f"To skip auto-tuning, set: gpu_batch_size={best_gpu_bs} cpu_batch_size={best_cpu_bs}"
    )
    return PositiveInt.parse(best_gpu_bs), PositiveInt.parse(best_cpu_bs)


def pool_one_segment(
    acts: t.Tensor,
    pooling: PoolingMode,
    keep_mask: t.Tensor | None = None,
) -> t.Tensor:
    """Pool a single segment's activations to a feature vector.

    When ``keep_mask`` is provided (length == ``acts.shape[0]``), positions
    that are ``False`` (``<bos>`` tokens, ignore_bos_features) are excluded
    from the reduction. An all-``False`` mask raises ``BosOnlySegmentError``.
    SAE outputs are post-ReLU non-negative, so masked-to-zero rows cannot beat
    the max of the kept rows.

    An **all-``True``** ``keep_mask`` excludes zero tokens and therefore MUST
    produce exactly the no-mask result: it falls through to the unmasked
    reductions below rather than the masked ``sum/count`` formulas. This keeps
    no-op masks bit-identical to ``keep_mask=None`` even in bf16/fp16 — where
    ``acts.sum(0)/n`` differs from ``acts.mean(0)`` by reduction order — while
    the ``keep_mask=None`` path itself stays byte-identical to legacy. Only a
    segment that actually contains a ``<bos>`` takes the masked branch.
    """
    assert acts.ndim == 2, f"acts must be 2D, got {acts.ndim}D"
    assert acts.shape[0] >= 1, (
        f"pool_one_segment requires at least 1 token, got seq_len={acts.shape[0]}"
    )
    if keep_mask is not None:
        if keep_mask.shape[0] != acts.shape[0]:
            raise ValueError(f"keep_mask length {keep_mask.shape[0]} != acts rows {acts.shape[0]}")
        kept = keep_mask.to(t.bool)
        if not bool(kept.any()):
            raise BosOnlySegmentError(
                "Context-independent segment is entirely <bos>; "
                "ignore_bos_features removed every poolable token."
            )
        if not bool(kept.all()):
            if pooling == PoolingMode.last_token:
                rel = int(kept.shape[0] - 1 - kept.flip(0).int().argmax().item())
                return acts[rel]
            if pooling == PoolingMode.max_pool:
                return acts.masked_fill(~kept.unsqueeze(-1), 0.0).max(dim=0).values
            if pooling == PoolingMode.mean_pool:
                kept_f = kept.unsqueeze(-1).to(acts.dtype)
                return (acts * kept_f).sum(dim=0) / kept.sum().to(acts.dtype)
            raise ValueError(f"Unknown pooling mode: {pooling}")
    if pooling == PoolingMode.last_token:
        return acts[-1]
    if pooling == PoolingMode.max_pool:
        return acts.max(dim=0).values
    if pooling == PoolingMode.mean_pool:
        return acts.mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def pool_segments_batched_gpu(
    sae_acts: t.Tensor,
    segments: list[tuple[int, int]],
    pooling: PoolingMode,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    keep_mask: t.Tensor | None = None,
) -> t.Tensor:
    """Pool SAE activations for all segments at once on GPU.

    Args:
        sae_acts: [seq_len, n_features] SAE activations on GPU.
        segments: List of (start, end) tuples.
        pooling: Pooling mode.
        segment_mode: Segmentation mode. fixed_tokens uses fast reshape path
            (segments guaranteed uniform by construction). Contiguous
            multi-segment layouts from other modes use segmented reductions;
            arbitrary overlapping, gapped, or reordered layouts retain the
            per-segment fallback.
        keep_mask: Optional [seq_len] bool tensor. ``False`` positions
            (``<bos>`` tokens under ignore_bos_features) are excluded from
            every reduction. A segment whose tokens are all ``False`` raises
            ``BosOnlySegmentError``. When ``None`` the function is
            byte-identical to its pre-keep_mask behaviour on the fixed-token
            and per-segment routes (hard contract): no extra ops touch those
            legacy code paths and results are exactly the legacy formulas.
            The contiguous multi-segment route reassociates the mean sum, so
            its float32 mean may differ from the legacy formula in the low
            bits (max, and float16/bfloat16 mean, stay bit-exact). An
            all-``True`` mask is normalized to ``None`` on every route.

    Returns:
        [n_segments, n_features] pooled tensor on GPU.
    """
    assert sae_acts.ndim == 2, f"sae_acts must be 2D, got {sae_acts.ndim}D"
    assert len(segments) > 0, "segments list must be non-empty"

    n_feat = sae_acts.shape[1]
    n_segs = len(segments)
    for i, (start, end) in enumerate(segments):
        assert 0 <= start < end <= sae_acts.shape[0], (
            f"Segment {i} ({start}, {end}) out of bounds for sae_acts with {sae_acts.shape[0]} rows"
        )

    if keep_mask is not None:
        if keep_mask.shape[0] != sae_acts.shape[0]:
            raise ValueError(
                f"keep_mask length {keep_mask.shape[0]} != sae_acts rows {sae_acts.shape[0]}"
            )
        keep_mask = keep_mask.to(t.bool)
        if bool(keep_mask.all()):
            keep_mask = None

    if pooling == PoolingMode.last_token:
        if keep_mask is None:
            last_indices = t.tensor(
                [end - 1 for _, end in segments],
                dtype=t.long,
                device=sae_acts.device,
            )
        else:
            idx: list[int] = []
            for i, (start, end) in enumerate(segments):
                m = keep_mask[start:end].to(t.bool)
                if not bool(m.any()):
                    raise BosOnlySegmentError(
                        f"Segment {i} [{start},{end}) is entirely <bos>; "
                        f"ignore_bos_features removed every poolable token "
                        f"(e.g. segment_size=1 on pretraining)."
                    )
                rel = int(m.shape[0] - 1 - m.flip(0).int().argmax().item())
                idx.append(start + rel)
            last_indices = t.tensor(idx, dtype=t.long, device=sae_acts.device)
        return sae_acts[last_indices]

    # Fast path: fixed_tokens segments are contiguous and uniform by construction.
    # Last segment may be shorter (tail of sequence).
    if segment_mode == SegmentMode.fixed_tokens:
        seg_size = segments[0][1] - segments[0][0]
        base = segments[0][0]
        last_seg_len = segments[-1][1] - segments[-1][0]
        trimmed_len = n_segs * seg_size
        block = sae_acts[base : base + trimmed_len]
        if block.shape[0] < trimmed_len:
            block = t.nn.functional.pad(block, (0, 0, 0, trimmed_len - block.shape[0]))
        reshaped = block.reshape(n_segs, seg_size, n_feat)

        if keep_mask is None:
            if pooling == PoolingMode.max_pool:
                # Short-last-segment padding is zero.  The padded positions
                # only fail to dominate the max when all activations are >= 0
                # (true for post-ReLU SAE outputs).  Assert loudly so a
                # future SAE family that emits negative values trips the
                # check instead of silently returning 0 as the "max" of an
                # all-negative segment.
                assert sae_acts.min().item() >= 0, (
                    "pool_segments_batched_gpu max_pool assumes non-negative "
                    "input (post-ReLU SAE outputs); short-last-segment zero "
                    "padding would otherwise mask negative activations."
                )
                return reshaped.max(dim=1).values
            if pooling == PoolingMode.mean_pool:
                summed = reshaped.sum(dim=1)
                counts = t.full(
                    (n_segs, 1),
                    seg_size,
                    dtype=sae_acts.dtype,
                    device=sae_acts.device,
                )
                counts[-1, 0] = last_seg_len
                return summed / counts
        else:
            mask_block = keep_mask[base : base + trimmed_len].to(t.bool)
            if mask_block.shape[0] < trimmed_len:
                mask_block = t.nn.functional.pad(mask_block, (0, trimmed_len - mask_block.shape[0]))
            mask2d = mask_block.reshape(n_segs, seg_size)
            kept_per_seg = mask2d.sum(dim=1)
            if bool((kept_per_seg == 0).any()):
                bad = int((kept_per_seg == 0).nonzero(as_tuple=True)[0][0].item())
                raise BosOnlySegmentError(
                    f"Segment {bad} is entirely <bos>; ignore_bos_features "
                    f"removed every poolable token (e.g. segment_size=1 on "
                    f"pretraining)."
                )
            if pooling == PoolingMode.max_pool:
                assert sae_acts.min().item() >= 0, (
                    "pool_segments_batched_gpu max_pool assumes non-negative "
                    "input (post-ReLU SAE outputs)."
                )
                return reshaped.masked_fill(~mask2d.unsqueeze(-1), 0.0).max(dim=1).values
            if pooling == PoolingMode.mean_pool:
                mask3d = mask2d.unsqueeze(-1).to(reshaped.dtype)
                summed = (reshaped * mask3d).sum(dim=1)
                counts = kept_per_seg.to(sae_acts.dtype).unsqueeze(1)
                return summed / counts

    # Turn-delimited segments are variable-sized but still form a contiguous
    # tiling. Vectorize that production layout while retaining the loop below.
    # The reachable production fallback is a single-segment document: one
    # reduction already, so segment_reduce cannot remove a launch and would
    # only add layout construction. The overlapping/gapped/reordered branch is
    # exercised by this helper's tests, which one segmented reduction cannot
    # represent; keeping it avoids imposing a new precondition on callers.
    contiguous_tiling = n_segs > 1 and all(
        left_end == right_start
        for (_, left_end), (right_start, _) in itertools.pairwise(segments)
    )
    segment_reduce_dtype = sae_acts.dtype in (
        t.float16,
        t.bfloat16,
        t.float32,
        t.float64,
    )
    if (
        contiguous_tiling
        and segment_reduce_dtype
        and pooling in (PoolingMode.max_pool, PoolingMode.mean_pool)
    ):
        if pooling == PoolingMode.max_pool:
            assert sae_acts.min().item() >= 0, (
                "pool_segments_batched_gpu max_pool assumes non-negative "
                "input (post-ReLU SAE outputs)."
            )

        base = segments[0][0]
        block_end = segments[-1][1]
        segment_layout = t.tensor(
            [(start - base, end - start) for start, end in segments],
            dtype=t.long,
            device=sae_acts.device,
        )
        segment_offsets = segment_layout[:, 0]
        segment_lengths = segment_layout[:, 1]
        block = sae_acts[base:block_end]

        kept_per_seg: t.Tensor | None = None
        mask_block: t.Tensor | None = None
        if keep_mask is not None:
            mask_block = keep_mask[base:block_end]
            kept_prefix = t.nn.functional.pad(mask_block.to(t.long).cumsum(0), (1, 0))
            kept_per_seg = (
                kept_prefix[segment_offsets + segment_lengths]
                - kept_prefix[segment_offsets]
            )
            empty_segments = kept_per_seg == 0
            if bool(empty_segments.any()):
                bad = int(empty_segments.nonzero(as_tuple=True)[0][0].item())
                start, end = segments[bad]
                raise BosOnlySegmentError(
                    f"Segment {bad} [{start},{end}) is entirely <bos>; "
                    f"ignore_bos_features removed every poolable token."
                )

        if pooling == PoolingMode.max_pool:
            if mask_block is not None:
                block = block.masked_fill(~mask_block.unsqueeze(-1), 0.0)
            # ``segment_reduce`` scans each segment in order, so like
            # ``max(dim=0)`` it retains the first maximum's representation —
            # including the sign bit when +0.0 and -0.0 tie. That equality is
            # pinned by the signed-zero tests rather than repaired at runtime:
            # an earlier scatter_reduce(amax) draft needed a fix-up here
            # because atomics can pick either tie, but this reduction cannot.
            return t.segment_reduce(
                block,
                "max",
                lengths=segment_lengths,
            )

        if mask_block is None:
            source = block
            counts = segment_lengths
        else:
            source = block * mask_block.unsqueeze(-1).to(block.dtype)
            assert kept_per_seg is not None
            counts = kept_per_seg
        low_precision_mean = sae_acts.dtype in (t.float16, t.bfloat16)
        summed = t.segment_reduce(
            source.float() if low_precision_mean else source,
            "sum",
            lengths=segment_lengths,
        )
        if low_precision_mean:
            if mask_block is not None:
                # Preserve the legacy masked formula's rounding point: its
                # per-segment sum is stored in the input dtype before division.
                summed = summed.to(sae_acts.dtype)
                return summed / counts.to(sae_acts.dtype).unsqueeze(1)
            return (summed / counts.float().unsqueeze(1)).to(sae_acts.dtype)
        return summed / counts.to(sae_acts.dtype).unsqueeze(1)

    # Compatibility fallback: arbitrary layouts and single-segment reductions.
    if pooling == PoolingMode.max_pool:
        assert sae_acts.min().item() >= 0, (
            "pool_segments_batched_gpu max_pool assumes non-negative "
            "input (post-ReLU SAE outputs)."
        )
        result = t.empty(n_segs, n_feat, device=sae_acts.device, dtype=sae_acts.dtype)
        for i, (start, end) in enumerate(segments):
            chunk = sae_acts[start:end]
            if keep_mask is not None:
                m = keep_mask[start:end].to(t.bool)
                if not bool(m.any()):
                    raise BosOnlySegmentError(
                        f"Segment {i} [{start},{end}) is entirely <bos>; "
                        f"ignore_bos_features removed every poolable token."
                    )
                chunk = chunk.masked_fill(~m.unsqueeze(-1), 0.0)
            result[i] = chunk.max(dim=0).values
        return result

    if pooling == PoolingMode.mean_pool:
        result = t.empty(n_segs, n_feat, device=sae_acts.device, dtype=sae_acts.dtype)
        for i, (start, end) in enumerate(segments):
            chunk = sae_acts[start:end]
            if keep_mask is None:
                result[i] = chunk.mean(dim=0)
            else:
                m = keep_mask[start:end].to(t.bool)
                if not bool(m.any()):
                    raise BosOnlySegmentError(
                        f"Segment {i} [{start},{end}) is entirely <bos>; "
                        f"ignore_bos_features removed every poolable token."
                    )
                mf = m.unsqueeze(-1).to(chunk.dtype)
                result[i] = (chunk * mf).sum(dim=0) / m.sum().to(chunk.dtype)
        return result

    raise ValueError(f"Unknown pooling mode: {pooling}")


def extract_sparse_gpu(pooled_batch: t.Tensor) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Extract non-zero features from batched pooled activations.

    Performs sparse extraction on GPU (nonzero, gather, bincount) then
    transfers results to pinned CPU memory via DMA. Pinned D2H is ~2x
    faster than the default pageable transfer path.

    Args:
        pooled_batch: [n_segments, n_features] on GPU (or CPU).

    Returns:
        (all_feat_ids, all_feat_mags) lists of CPU tensors, one per segment.
    """
    assert pooled_batch.ndim == 2, f"pooled_batch must be 2D, got {pooled_batch.ndim}D"
    n_segs = pooled_batch.shape[0]

    # Batch nonzero: [nnz, 2] with (row_idx, col_idx) pairs
    nz = (pooled_batch > 0).nonzero()

    if nz.shape[0] == 0:
        # NOTE: each list element must be an INDEPENDENT tensor. ``[x] * n``
        # aliases a single tensor n times, so any downstream in-place mutation
        # (``resize_``, pin_memory, future compaction) corrupts every segment
        # simultaneously — a latent data-corruption trap.
        return (
            [t.empty(0, dtype=t.int32) for _ in range(n_segs)],
            [t.empty(0, dtype=t.float32) for _ in range(n_segs)],
        )

    feat_ids = nz[:, 1].to(t.int32)
    mags = pooled_batch[nz[:, 0], nz[:, 1]].to(t.float32)
    seg_lengths = t.bincount(nz[:, 0].int(), minlength=n_segs)

    # Vectorized within-segment sort by magnitude descending.
    # Note: save_features() will re-sort by feature ID ascending for compression.
    # Two-pass stable sort on the device: first by -magnitude, then by segment
    # index. The stable second pass preserves within-segment magnitude ordering.
    # This replaces a Python loop over n_segs individual argsort calls.
    mag_order = mags.argsort(descending=True, stable=True)
    seg_order = nz[:, 0][mag_order].argsort(stable=True)
    final_order = mag_order[seg_order]
    feat_ids = feat_ids[final_order]
    mags = mags[final_order]

    # Pinned D2H: ~2x faster than pageable .cpu() transfers
    if pooled_batch.is_cuda:
        cat_ids = t.empty_like(feat_ids, device="cpu", pin_memory=True)
        cat_ids.copy_(feat_ids)
        cat_mags = t.empty_like(mags, device="cpu", pin_memory=True)
        cat_mags.copy_(mags)
    else:
        cat_ids = feat_ids
        cat_mags = mags

    # Split on CPU by per-segment counts (already sorted within each segment)
    all_ids = list(cat_ids.split(seg_lengths.tolist()))
    all_mags = list(cat_mags.split(seg_lengths.tolist()))

    assert len(all_ids) == n_segs, f"Expected {n_segs} id lists, got {len(all_ids)}"
    assert len(all_mags) == n_segs, f"Expected {n_segs} mag lists, got {len(all_mags)}"
    return all_ids, all_mags


def extract_document_features(
    model,
    sae,
    text: str,
    layer: int = 8,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    pooling: PoolingMode = PoolingMode.mean_pool,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: int | None = None,
    hook_name: str | None = None,
    context_independent_features: bool = False,
    ignore_bos_features: bool = False,
    turn_char_starts: Sequence[int] | None = None,
) -> tuple[list[t.Tensor], list[t.Tensor], list[int], list[int], list[int], int]:
    """Extract per-segment SAE features from a single document.

    See module docstring for full semantics. The ``context_independent_features``
    flag switches each segment to its own forward pass with BOS injection per
    the usual rule (inject iff ``seg[0] != bos_token_id``).

    Segment character offsets are always document-relative. SFT data reaches
    this function as full nanochat-rendered text, so it shares the same
    segmentation contract as pretraining.
    """
    if segment_size is None:
        segment_size = context_window

    device = next(sae.parameters()).device
    sae_max_context = model.cfg.n_ctx

    with record_function("tokenize"):
        kept_ids, n_tokens = _truncate_to_training_window(
            text,
            model.tokenizer,
            context_window,
            sae_max_context,
        )
    if not kept_ids:
        raise ValueError(f"Document produced 0 kept tokens: {text[:100]!r}")

    ignore_bos_bos_id: int | None = None
    if ignore_bos_features:
        ignore_bos_bos_id = resolve_bos_token_id(model.tokenizer)
        if ignore_bos_bos_id is None:
            raise ValueError(
                "ignore_bos_features=True requires tokenizer to define "
                f"bos_token_id (model={model.cfg.model_name})"
            )

    batch = [(kept_ids, n_tokens)]
    precomputed = _precompute_batch_cpu_work(
        batch,
        model.tokenizer,
        segment_mode,
        segment_size,
        texts=[text] if segment_mode is SegmentMode.turn_delimited_fixed_tokens else None,
        turn_char_starts=(
            [turn_char_starts] if turn_char_starts is not None else None
        ),
    )
    segments, _, char_starts, char_ends = precomputed[0]

    if context_independent_features:
        bos_token_id = getattr(model.tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError(
                f"context_independent_features=True requires tokenizer to "
                f"define bos_token_id (model={model.cfg.model_name})"
            )
        if segment_size + 1 > model.cfg.n_ctx:
            raise ValueError(
                f"segment_size+1={segment_size + 1} exceeds model context "
                f"{model.cfg.n_ctx}; BOS injection would overflow"
            )
        context_independent_inputs = _prepare_context_independent_segment_inputs(
            chunk_token_info=[(kept_ids, n_tokens)],
            chunk_precomputed=[(segments, n_tokens, char_starts, char_ends)],
            bos_token_id=bos_token_id,
            ignore_bos_features=ignore_bos_features,
        )
        if not context_independent_inputs:
            return [], [], [], char_starts, char_ends, n_tokens
        with record_function("context_independent_forward_and_pool"):
            native_model = getattr(model, "native_model", None)
            nnterp_model = getattr(model, "nnterp_model", None)
            pooled = _context_independent_forward_and_pool_batch(
                model,
                sae,
                context_independent_inputs,
                layer=layer,
                pooling=pooling,
                hook_name=hook_name,
                activation_type=ActivationType.residual,
                nnterp_model=nnterp_model,
                native_model=native_model,
            )
            all_feat_ids, all_feat_mags = extract_sparse_gpu(pooled)
    else:
        native_model = getattr(model, "native_model", None)
        nnterp_model = getattr(model, "nnterp_model", None)
        if native_model is not None or nnterp_model is not None:
            sae_acts = _get_sae_activations_batched(
                model=model,
                sae=sae,
                token_id_lists=[kept_ids],
                layer=layer,
                nnterp_model=nnterp_model,
                native_model=native_model,
                hook_name=hook_name,
                activation_type=_infer_activation_type_from_hook_name(hook_name),
            )[0]
        else:
            kept_tensor = t.tensor(kept_ids, dtype=t.long, device=device)
            sae_acts = _get_sae_activations(
                model,
                sae,
                kept_tensor,
                layer,
                hook_name=hook_name,
            )
        doc_keep_mask = None
        if ignore_bos_features:
            assert ignore_bos_bos_id is not None
            doc_keep_mask = t.tensor(kept_ids, device=sae_acts.device) != ignore_bos_bos_id
        with record_function("pool_and_extract"):
            pooled = pool_segments_batched_gpu(
                sae_acts,
                segments,
                pooling,
                segment_mode,
                keep_mask=doc_keep_mask,
            )
            all_feat_ids, all_feat_mags = extract_sparse_gpu(pooled)

    positions = [start for start, _ in segments]
    assert len(all_feat_ids) == len(segments)
    assert len(all_feat_mags) == len(segments)
    assert len(positions) == len(segments)
    assert len(char_starts) == len(segments)
    assert len(char_ends) == len(segments)
    return all_feat_ids, all_feat_mags, positions, char_starts, char_ends, n_tokens


def estimate_extraction_flops_per_token(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    context_window: int,
) -> int:
    """Estimate forward-only FLOPs per token for partial model (layers 0..layer) + SAE.

    Uses 2 FLOPs per matmul parameter (forward only, no backward) plus
    attention QK^T and attn*V FLOPs scaled by context window length.
    Based on the PaLM paper formulation (arxiv.org/abs/2204.02311).
    """
    # Matmul params: all 2D+ weight tensors in layers 0..layer
    matmul_params = 0
    for i in range(layer + 1):
        matmul_params += sum(p.numel() for p in model.blocks[i].parameters() if p.ndim >= 2)
    # SAE encoder + decoder matmul params
    sae_matmul_params = sum(p.numel() for p in sae.parameters() if p.ndim >= 2)
    # Attention FLOPs: 4 * n_heads * d_head * seq_len per layer
    # (2 for QK^T + 2 for attn*V, forward only)
    attn_flops = (layer + 1) * 4 * model.cfg.n_heads * model.cfg.d_head * context_window
    return 2 * (matmul_params + sae_matmul_params) + attn_flops


def _prepare_context_independent_segment_inputs(
    chunk_token_info: list[tuple[list[int], int]],
    chunk_precomputed: list[tuple[list[tuple[int, int]], int, list[int], list[int]]],
    bos_token_id: int,
    ignore_bos_features: bool = False,
) -> list[ContextIndependentSegmentInput]:
    """Flatten per-doc segments into a list of ContextIndependentSegmentInput for batched context-independent forward.

    For each (doc, segment) pair: slice the segment tokens from ``kept_ids``,
    prepend BOS iff the first token is not already BOS, and record everything
    the downstream forward-and-pool step needs to reattribute results.

    Pure CPU; thread-safe. Preserves (doc_idx, seg_idx) ordering (docs outer,
    segs inner). Zero-segment docs (e.g. SFT truncation) contribute nothing.
    """
    result: list[ContextIndependentSegmentInput] = []
    for doc_idx, (
        (segments_d, _n_tokens_d, _cs_d, _ce_d),
        (kept_ids_d, _n_full_d),
    ) in enumerate(zip(chunk_precomputed, chunk_token_info, strict=True)):
        for seg_idx, (start, end) in enumerate(segments_d):
            seg = kept_ids_d[start:end]
            bos_injected = seg[0] != bos_token_id
            model_input = ([bos_token_id] + seg) if bos_injected else list(seg)
            keep_mask = tuple(tok != bos_token_id for tok in seg) if ignore_bos_features else None
            result.append(
                ContextIndependentSegmentInput(
                    doc_idx=doc_idx,
                    seg_idx=seg_idx,
                    model_input=model_input,
                    bos_injected=bos_injected,
                    original_length=end - start,
                    keep_mask=keep_mask,
                )
            )
    return result


def _precompute_batch_cpu_work(
    batch: list[tuple[list[int], int]],
    tokenizer,
    segment_mode: SegmentMode,
    segment_size: int,
    *,
    texts: list[str] | None = None,
    turn_char_starts: Sequence[Sequence[int]] | None = None,
) -> list[tuple[list[tuple[int, int]], int, list[int], list[int]]]:
    """Pre-compute segments + char offsets for a batch (CPU only, thread-safe).

    Runs segment_document for each document in the batch.
    These are pure CPU operations that can safely run in a background thread
    while the GPU processes the forward pass.

    Returns:
        List of (segments, n_tokens, char_starts, char_ends).
    """
    refined_segment_size = PositiveInt.parse(segment_size)
    if segment_mode is SegmentMode.fixed_tokens:
        if turn_char_starts is not None:
            raise ValueError("fixed_tokens segmentation does not accept turn_char_starts")
        if texts is not None:
            raise ValueError("fixed_tokens segmentation does not accept texts")
    else:
        if texts is None or turn_char_starts is None:
            raise ValueError(
                "turn_delimited_fixed_tokens segmentation requires texts and turn_char_starts"
            )
        if len(texts) != len(batch) or len(turn_char_starts) != len(batch):
            raise ValueError(
                "turn-delimited batch/text/boundary length mismatch: "
                f"{len(batch)}, {len(texts)}, {len(turn_char_starts)}"
            )
    results = []
    for doc_index, (kept_ids, n_tokens) in enumerate(batch):
        if segment_mode is SegmentMode.fixed_tokens:
            segments, char_starts, char_ends = segment_document(
                kept_ids, tokenizer, segment_mode, refined_segment_size
            )
        else:
            assert texts is not None
            assert turn_char_starts is not None
            segments, char_starts, char_ends, tail_drift = turn_segments_for_document(
                kept_ids,
                tokenizer,
                texts[doc_index],
                turn_char_starts[doc_index],
                refined_segment_size,
            )
            if tail_drift is not None:
                record_counter(
                    "segmentation.turn_boundary.tail_drift",
                    1,
                    delta=tail_drift,
                    direction="tail",
                )
        if not segments:
            raise ValueError(
                f"Segmentation produced no segments for {n_tokens} tokens (mode={segment_mode})"
            )
        results.append((segments, n_tokens, char_starts, char_ends))
    return results


def _truncate_to_training_window(
    text: str,
    sae_tokenizer,
    context_window: int,
    sae_max_context: int,
) -> tuple[list[int], int]:
    """Truncate a document to fit within the context window using the SAE tokenizer.

    Tokenizes ``text`` with the SAE tokenizer and keeps at most
    ``min(context_window, sae_max_context)`` tokens from the head.

    Returns:
        (kept_sae_ids, full_n_sae_tokens).
    """
    return truncate_to_training_window(text, sae_tokenizer, context_window, sae_max_context)


def _batch_truncate_to_training_window(
    texts: list[str],
    sae_tokenizer,
    context_window: int,
    sae_max_context: int,
    *,
    chunk_size: int = 2048,
) -> list[tuple[list[int], int]]:
    """Batched, chunked equivalent of per-text :func:`_truncate_to_training_window`.

    Issues ONE tokenizer call per ``chunk_size`` texts (amortizing the
    Python->tokenizer FFI and letting fast tokenizers thread internally) instead
    of one call per text. The result is BIT-IDENTICAL to calling
    :func:`_truncate_to_training_window` on each text:

    * ``padding=False`` => each text is encoded independently (no cross-text
      interaction), so the i-th batched encoding equals the standalone encoding.
    * Empty texts map to ``([], 0)`` WITHOUT consulting the tokenizer, exactly as
      the per-text path does — they are filtered out of the batched call so a
      tokenizer's ``""`` output (e.g. a lone BOS) can never leak in.

    Returns one ``(kept_ids, full_n_tokens)`` per input text, in order. Chunking
    bounds peak tokenizer memory (a shard may hold tens of thousands of docs).
    """
    return batch_truncate_to_training_window(
        texts, sae_tokenizer, context_window, sae_max_context, chunk_size=chunk_size
    )


def _tokenize_chunk(
    sae_tokenizer,
    texts: list[str],
    context_window: int,
    sae_max_context: int,
) -> list[tuple[list[int], int]]:
    """Tokenize and head-truncate a chunk of texts using the SAE tokenizer.

    Truncates each document to ``min(context_window, sae_max_context)`` SAE
    tokens from the head.

    Returns:
        List of (kept_sae_ids, full_n_sae_tokens) tuples.
    """
    result: list[tuple[list[int], int]] = []
    for text in texts:
        kept_ids, n_full = _truncate_to_training_window(
            text,
            sae_tokenizer,
            context_window,
            sae_max_context,
        )
        if len(kept_ids) <= 0 and text:
            raise ValueError(f"Document produced 0 kept SAE tokens: {text[:100]!r}")
        result.append((kept_ids, n_full))
    return result


def _tokenize_and_segment_chunk(
    sae_tokenizer,
    texts: list[str],
    context_window: int,
    segment_mode: SegmentMode,
    segment_size: int,
    sae_max_context: int,
    simulate_delay_s: float = 0.0,
    *,
    turn_char_starts: list[list[int]] | None = None,
) -> tuple[
    list[tuple[list[int], int]],
    list[tuple[list[tuple[int, int]], int, list[int], list[int]]],
]:
    """Tokenize a chunk and pre-compute segments + char offsets.

    Combines _tokenize_chunk and _precompute_batch_cpu_work so that all CPU
    work for a chunk runs together in a background thread, leaving the GPU
    batch loop free of Python CPU work (and GIL contention).
    Thread-safe.

    Returns:
        (token_info, precomputed) where token_info is from _tokenize_chunk
        and precomputed is from _precompute_batch_cpu_work.
    """
    token_info = _tokenize_chunk(
        sae_tokenizer,
        texts,
        context_window,
        sae_max_context,
    )
    precomputed = _precompute_batch_cpu_work(
        token_info,
        sae_tokenizer,
        segment_mode,
        segment_size,
        texts=texts if segment_mode is SegmentMode.turn_delimited_fixed_tokens else None,
        turn_char_starts=turn_char_starts,
    )
    if simulate_delay_s > 0:
        time.sleep(simulate_delay_s)
    return token_info, precomputed


def _is_picklable(obj) -> bool:
    """Check if an object can be pickled (for ProcessPoolExecutor compatibility)."""
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


# Module-level globals for ProcessPoolExecutor worker processes.
# Set by _init_worker at pool creation time, read by _tokenize_and_segment_chunk_worker.
_worker_tokenizer = None
_worker_context_window: int = 0
_worker_segment_mode: str = ""
_worker_segment_size: int = 0
_worker_simulate_delay_s: float = 0.0
_worker_sae_max_context: int = 0


def _init_worker(
    tokenizer,
    context_window: int,
    segment_mode: str,
    segment_size: int,
    sae_max_context: int,
    simulate_delay_s: float = 0.0,
):
    """Initialize worker process globals. Called once per worker at pool creation."""
    global _worker_tokenizer, _worker_context_window, _worker_segment_mode
    global _worker_segment_size, _worker_simulate_delay_s
    global _worker_sae_max_context
    _worker_tokenizer = tokenizer
    _worker_context_window = context_window
    _worker_segment_mode = segment_mode
    _worker_segment_size = segment_size
    _worker_simulate_delay_s = simulate_delay_s
    _worker_sae_max_context = sae_max_context


def _tokenize_and_segment_chunk_worker(
    texts: list[str],
    turn_char_starts: list[list[int]] | None = None,
) -> tuple[
    list[tuple[list[int], int]],
    list[tuple[list[tuple[int, int]], int, list[int], list[int]]],
]:
    """Worker-process entry point: reads globals set by _init_worker."""
    return _tokenize_and_segment_chunk(
        _worker_tokenizer,
        texts,
        _worker_context_window,
        SegmentMode(_worker_segment_mode),
        _worker_segment_size,
        sae_max_context=_worker_sae_max_context,
        simulate_delay_s=_worker_simulate_delay_s,
        turn_char_starts=turn_char_starts,
    )


def _run_context_independent_chunk(
    *,
    model: HookedTransformer,
    sae: SAE,
    chunk_token_info: list[tuple[list[int], int]],
    chunk_precomputed: list[tuple[list[tuple[int, int]], int, list[int], list[int]]],
    layer: int,
    pooling: PoolingMode,
    gpu_batch_size: int,
    hook_name: str | None,
    activation_type: ActivationType | None,
    nnterp_model: StandardizedTransformer | None,
    native_model: NativeCollector | None = None,
    bos_token_id: int,
    ignore_bos_features: bool = False,
    device: t.device,
    gc_mem_threshold: float,
    timer: Any | None,
    # Mutated-in-place accumulators (shared with cd path):
    all_feature_ids: list[t.Tensor],
    all_feature_mags: list[t.Tensor],
    doc_segment_info: list,
    doc_token_counts: list[int],
    doc_counter: int,
) -> int:
    """Context-independent counterpart to the cd GPU batch loop for one chunk.

    Flattens (doc, segment) pairs -> batches ``gpu_batch_size`` segments per
    forward pass -> pools each segment with BOS-drop -> scatters sparse results
    back into per-doc accumulators. Returns the new ``doc_counter`` after
    processing this chunk's docs.
    """

    def _region(name: str):
        return timer.region(name) if timer is not None else nullcontext()

    with _region("context_independent_flatten"):
        context_independent_inputs = _prepare_context_independent_segment_inputs(
            chunk_token_info,
            chunk_precomputed,
            bos_token_id=bos_token_id,
            ignore_bos_features=ignore_bos_features,
        )

    # Scatter target keyed by (doc_idx, seg_idx).
    per_segment_sparse: dict[tuple[int, int], tuple[t.Tensor, t.Tensor]] = {}

    for bs in range(0, len(context_independent_inputs), gpu_batch_size):
        batch = context_independent_inputs[bs : bs + gpu_batch_size]
        with _region("model_forward_batched"):
            pooled = _context_independent_forward_and_pool_batch(
                model,
                sae,
                batch,
                layer=layer,
                pooling=pooling,
                hook_name=hook_name,
                activation_type=activation_type,
                nnterp_model=nnterp_model,
                native_model=native_model,
            )
        with _region("extract_sparse"):
            batch_ids, batch_mags = extract_sparse_gpu(pooled)
        for ci_input, fid, fmag in zip(batch, batch_ids, batch_mags, strict=True):
            per_segment_sparse[(ci_input.doc_idx, ci_input.seg_idx)] = (fid, fmag)
        with _region("gc_cuda"):
            maybe_gc_cuda(device, threshold=gc_mem_threshold)

    # Reassembly.
    assert len(per_segment_sparse) == len(context_independent_inputs), (
        f"Context-independent reassembly incomplete: have {len(per_segment_sparse)} sparse "
        f"entries, expected {len(context_independent_inputs)}"
    )
    seg_count_by_doc: dict[int, int] = collections.Counter(
        doc_idx for (doc_idx, _) in per_segment_sparse
    )
    with _region("result_accumulate"):
        for doc_idx, (segments_d, n_tokens_d, cs_d, ce_d) in enumerate(
            chunk_precomputed,
        ):
            n_segs_d = len(segments_d)
            assert seg_count_by_doc.get(doc_idx, 0) == n_segs_d, (
                f"doc {doc_idx}: have {seg_count_by_doc.get(doc_idx, 0)} "
                f"sparse entries, expected {n_segs_d} segments"
            )
            for seg_idx in range(n_segs_d):
                fid, fmag = per_segment_sparse[(doc_idx, seg_idx)]
                all_feature_ids.append(fid)
                all_feature_mags.append(fmag)
            positions = [s for s, _ in segments_d]
            doc_segment_info.append(
                (doc_counter, n_segs_d, positions, cs_d, ce_d),
            )
            doc_counter += 1
            doc_token_counts.append(n_tokens_d)
    return doc_counter


def stamp_l0_tier(shard_dict: dict, run_l0_tier: L0Tier | None) -> None:
    """Stamp this run's gemma-scope-2 L0 tier into a shard dict in place.

    This is the *exact* production injection used right before every shard
    write (both the async-saver and the direct ``save_features`` paths).
    ``save_features`` writes any non-tensor str value in the dict to the
    safetensors metadata generically, so the plain ``L0Tier.value`` string
    is sufficient for ``read_l0_tier_from_shard`` to round-trip it back.

    For v1 / legacy runs (``run_l0_tier is None``) the ``l0_tier`` key is
    left absent, preserving legacy L0 metadata — we never write an empty
    string or the literal ``"None"``.
    """
    if run_l0_tier is not None:
        shard_dict["l0_tier"] = run_l0_tier.value


def process_shard(
    parquet_path: str,
    model: HookedTransformer,
    sae: SAE,
    layer: int = 8,
    output_dir: str = ".",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    pooling: PoolingMode = PoolingMode.mean_pool,
    shard_position: int = 0,
    gpu_batch_size: int = 8,
    cpu_batch_size: int = 64,
    gc_mem_threshold: float = 0.95,
    nnterp_model: StandardizedTransformer | None = None,
    native_model: NativeCollector | None = None,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: int | None = None,
    hook_name: str | None = None,
    activation_type: ActivationType = ActivationType.residual,
    timer: Any | None = None,
    pipeline_post: bool = True,
    _simulate_cpu_delay_s: float = 0.0,
    dataset_phase: str | None = None,
    background_saver: BackgroundSaver | None = None,
    prefetched: PrefetchedShard | None = None,
    zstd_level: int | None = None,
    context_independent_features: bool = False,
    ignore_bos_features: bool = False,
    run_l0_tier: L0Tier | None = None,
    sft_mixture_fingerprint_json: str | None = None,
    extraction_source_identity: ExtractionSourceIdentity | None = None,
) -> ShardResult:
    """Process all documents in a parquet shard with segment-granular extraction.

    Args:
        timer: Optional FunctionTimer for profiling. When provided, wraps key
            regions with timing instrumentation: model_forward_batched,
            segment_and_pool, extract_sparse, gc_cuda, batch_prep,
            result_accumulate, tokenize_wait.
        pipeline_post: When True, pre-compute CPU work (segmentation + char
            offsets) for batch N+1 in a background thread while the GPU
            processes batch N. Expected ~1.45x speedup on GPU-bound workloads.
        _simulate_cpu_delay_s: Artificial per-chunk delay (seconds) added to
            tokenization workers. Simulates the effect of a faster GPU (like
            H100) on a slower machine where the GPU would otherwise keep up
            with the CPU. For profiling only.
        prefetched: Pre-loaded shard data (from ``prefetch_shard_data``, e.g.
            via the staged input prefetcher). When provided, skips Phase 1
            (parquet read + text chunking) and uses the pre-loaded data
            directly. When None, loads data inline (original behavior).
        zstd_level: Compression level (1-22) for output shards. Defaults to
            the module-level constant in ``feature_io``.
        run_l0_tier: The gemma-scope-2 L0Tier for this run, or ``None`` for
            v1 / legacy runs. When not None it is stamped into every shard's
            metadata (``shard_dict["l0_tier"]``); when None the key is left
            absent to preserve legacy L0 metadata.
        sft_mixture_fingerprint_json: Canonical SFT conversion identity copied
            into the existing feature-shard metadata surface. ``None`` omits
            this fingerprint key.
        extraction_source_identity: Resolved coordinates used to load the model
            and SAE. Legacy/direct callers may omit them; their outputs cannot
            qualify for consumers requiring authenticated extraction identity.

    Returns:
        ShardResult with profiling info.
    """
    extraction_backend = resolved_extraction_backend(nnterp_model, native_model)
    if segment_size is None:
        segment_size = context_window

    sae_max_context = model.cfg.n_ctx

    # Context-independent / ignore-bos precondition resolution (one-shot per
    # shard). bos resolution covers BOTH flags; the BOS-injection overflow
    # check and the equivalence info-log stay context-independent-only.
    bos_token_id: int | None = None
    if context_independent_features or ignore_bos_features:
        bos_token_id = resolve_bos_token_id(model.tokenizer)
        if bos_token_id is None:
            raise ValueError(
                "context_independent_features/ignore_bos_features require the "
                f"tokenizer to define bos_token_id (model={model.cfg.model_name})"
            )
    if context_independent_features:
        # segment_size was normalized to context_window above when None.
        if segment_size + 1 > sae_max_context:
            raise ValueError(
                f"segment_size+1={segment_size + 1} exceeds model context "
                f"{sae_max_context}; BOS injection would overflow"
            )
        if segment_size >= context_window:
            logger.info(
                "context_independent_features=True but segment_size "
                f"({segment_size}) >= context_window ({context_window}); "
                "feature values will be semantically equivalent to "
                "context_dependent extraction."
            )

    def _region(name: str):
        return timer.region(name) if timer is not None else nullcontext()

    start_time = time.time()
    shard_name = Path(parquet_path).stem
    logger.info(
        "extract_features shard start: shard={} context_independent_features={} "
        "pipeline_post={}",
        shard_name,
        context_independent_features,
        pipeline_post,
    )

    all_feature_ids: list[t.Tensor] = []
    all_feature_mags: list[t.Tensor] = []
    doc_token_counts: list[int] = []
    # Per-document: (n_segments, positions, char_starts, char_ends)
    doc_segment_info: list[
        tuple[int, int, list[int], list[int], list[int]]
    ] = []  # (doc_id, n_segs, positions, char_starts, char_ends)

    doc_counter = 0

    device = next(sae.parameters()).device

    if prefetched is not None:
        # Use pre-loaded data from the staged input prefetcher (skip Phase 1)
        all_text_chunks = prefetched.all_text_chunks
        all_turn_start_chunks = prefetched.all_turn_start_chunks
    else:
        # Phase 1: inline data loading (original path)
        pf = pq.ParquetFile(parquet_path)

        rg_bar = tqdm(
            range(pf.num_row_groups),
            desc=f"{shard_name} row groups",
            position=shard_position + 1,
            leave=False,
            total=pf.num_row_groups,
            disable=_tqdm_disabled(),
        )

        # Build flat list of text chunks across all row groups for pipelining.
        # This lets us overlap tokenization of chunk N+1 with GPU processing
        # of chunk N.
        _validate_extraction_parquet_schema(
            parquet_path=parquet_path,
            schema_names=pf.schema_arrow.names,
            dataset_phase=dataset_phase,
            segment_mode=segment_mode,
        )

        all_text_chunks, all_turn_start_chunks = _read_text_and_turn_chunks(
            pf,
            rg_bar,
            cpu_batch_size,
            segment_mode,
        )

    shard_progress = _ShardProgress(
        shard_name,
        total_docs=sum(len(texts) for texts in all_text_chunks),
    )

    # Pipeline: tokenize (+ optionally segment) chunk N+1 on CPU while
    # GPU processes chunk N.
    if pipeline_post and all_text_chunks:
        # Chunk-level pipelining: background process/thread does tokenization
        # AND segmentation/char-offsets together, so the GPU batch loop has
        # zero Python CPU work (no GIL contention during forward pass).
        # ProcessPoolExecutor eliminates GIL contention entirely (separate
        # process = separate GIL). Falls back to ThreadPoolExecutor when the
        # tokenizer isn't picklable (e.g. MagicMock in tests).
        use_process_pool = _is_picklable(model.tokenizer)
        if use_process_pool:
            executor_cm = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    model.tokenizer,
                    context_window,
                    segment_mode.value,
                    segment_size,
                    sae_max_context,
                    _simulate_cpu_delay_s,
                ),
            )

            def _submit(texts, turn_starts):
                return executor_cm.submit(
                    _tokenize_and_segment_chunk_worker, texts, turn_starts
                )
        else:
            executor_cm = ThreadPoolExecutor(max_workers=1)

            def _submit(texts, turn_starts):
                return executor_cm.submit(
                    _tokenize_and_segment_chunk,
                    model.tokenizer,
                    texts,
                    context_window,
                    segment_mode,
                    segment_size,
                    sae_max_context=sae_max_context,
                    simulate_delay_s=_simulate_cpu_delay_s,
                    turn_char_starts=turn_starts,
                )

        with executor_cm as tok_pool:
            pending_future = _submit(
                all_text_chunks[0],
                all_turn_start_chunks[0],
            )

            for chunk_idx in range(len(all_text_chunks)):
                with _region("tokenize_wait"):
                    with record_function("tokenize_chunk"):
                        chunk_token_info, chunk_precomputed = pending_future.result()

                assert len(chunk_token_info) == len(chunk_precomputed), (
                    f"Worker returned mismatched lengths: "
                    f"{len(chunk_token_info)} token_info vs {len(chunk_precomputed)} precomputed"
                )
                assert len(chunk_token_info) == len(all_text_chunks[chunk_idx]), (
                    f"Worker returned {len(chunk_token_info)} results for "
                    f"{len(all_text_chunks[chunk_idx])} texts"
                )

                if chunk_idx + 1 < len(all_text_chunks):
                    pending_future = _submit(
                        all_text_chunks[chunk_idx + 1],
                        all_turn_start_chunks[chunk_idx + 1],
                    )

                if context_independent_features:
                    assert bos_token_id is not None
                    # Pretraining seg-0 BOS precondition check (context-independent-only, loud).
                    if dataset_phase in (None, "base"):
                        for kept_ids, _ in chunk_token_info:
                            if kept_ids and kept_ids[0] != bos_token_id:
                                raise AssertionError(
                                    f"Pretraining doc has kept_ids[0]={kept_ids[0]} "
                                    f"!= bos_token_id={bos_token_id}; context-independent assumes "
                                    "auto-BOS prepend at extraction."
                                )
                    doc_counter = _run_context_independent_chunk(
                        model=model,
                        sae=sae,
                        chunk_token_info=chunk_token_info,
                        chunk_precomputed=chunk_precomputed,
                        layer=layer,
                        pooling=pooling,
                        gpu_batch_size=gpu_batch_size,
                        hook_name=hook_name,
                        activation_type=activation_type,
                        nnterp_model=nnterp_model,
                        native_model=native_model,
                        bos_token_id=bos_token_id,
                        ignore_bos_features=ignore_bos_features,
                        device=device,
                        gc_mem_threshold=gc_mem_threshold,
                        timer=timer,
                        all_feature_ids=all_feature_ids,
                        all_feature_mags=all_feature_mags,
                        doc_segment_info=doc_segment_info,
                        doc_token_counts=doc_token_counts,
                        doc_counter=doc_counter,
                    )
                    shard_progress.emit(
                        docs_done=doc_counter,
                        segments_done=len(all_feature_ids),
                    )
                else:
                    batch_starts = list(range(0, len(chunk_token_info), gpu_batch_size))
                    doc_bar = tqdm(
                        batch_starts,
                        desc=f"Chunk {chunk_idx + 1}/{len(all_text_chunks)}",
                        position=shard_position + 2,
                        leave=False,
                        disable=_tqdm_disabled(),
                    )

                    for batch_start in doc_bar:
                        with _region("batch_prep"):
                            batch_tok = chunk_token_info[batch_start : batch_start + gpu_batch_size]
                            batch_pre = chunk_precomputed[
                                batch_start : batch_start + gpu_batch_size
                            ]
                            assert len(batch_tok) == len(batch_pre)
                            batch_kept_ids = [kept_ids for kept_ids, _ in batch_tok]

                        with _region("model_forward_batched"):
                            sae_acts_list = _get_sae_activations_batched(
                                model,
                                sae,
                                batch_kept_ids,
                                layer=layer,
                                nnterp_model=nnterp_model,
                                native_model=native_model,
                                hook_name=hook_name,
                                activation_type=activation_type,
                            )

                        # GPU-only: pool using pre-computed segments
                        with _region("segment_and_pool"):
                            batch_pooled_gpu: list[t.Tensor] = []
                            batch_seg_counts: list[int] = []
                            batch_positions: list[list[int]] = []
                            batch_char_starts: list[list[int]] = []
                            batch_char_ends: list[list[int]] = []

                            for (
                                segments,
                                n_tokens,
                                char_starts,
                                char_ends,
                            ), sae_acts, kept_ids_doc in zip(
                                batch_pre,
                                sae_acts_list,
                                batch_kept_ids,
                                strict=True,
                            ):
                                with record_function("pool_segments"):
                                    doc_keep_mask = None
                                    if ignore_bos_features:
                                        assert bos_token_id is not None
                                        doc_keep_mask = (
                                            t.tensor(kept_ids_doc, device=sae_acts.device)
                                            != bos_token_id
                                        )
                                    pooled = pool_segments_batched_gpu(
                                        sae_acts,
                                        segments,
                                        pooling,
                                        segment_mode,
                                        keep_mask=doc_keep_mask,
                                    )
                                batch_pooled_gpu.append(pooled)
                                batch_seg_counts.append(len(segments))
                                batch_positions.append([start for start, _ in segments])
                                batch_char_starts.append(char_starts)
                                batch_char_ends.append(char_ends)
                                doc_token_counts.append(n_tokens)

                        if batch_pooled_gpu:
                            all_pooled = t.cat(batch_pooled_gpu, dim=0)
                            assert all_pooled.shape[0] == sum(batch_seg_counts), (
                                f"Pooled rows {all_pooled.shape[0]} != "
                                f"sum(seg_counts) {sum(batch_seg_counts)}"
                            )
                            with _region("extract_sparse"):
                                with record_function("extract_sparse"):
                                    batch_ids, batch_mags = extract_sparse_gpu(all_pooled)
                            del all_pooled
                        else:
                            batch_ids = []
                            batch_mags = []
                        del batch_pooled_gpu

                        with _region("result_accumulate"):
                            seg_offset = 0
                            for n_segs, positions, cstarts, cends in zip(
                                batch_seg_counts,
                                batch_positions,
                                batch_char_starts,
                                batch_char_ends,
                                strict=True,
                            ):
                                doc_ids = batch_ids[seg_offset : seg_offset + n_segs]
                                doc_mags = batch_mags[seg_offset : seg_offset + n_segs]
                                all_feature_ids.extend(doc_ids)
                                all_feature_mags.extend(doc_mags)
                                doc_segment_info.append(
                                    (doc_counter, n_segs, positions, cstarts, cends),
                                )
                                doc_counter += 1
                                seg_offset += n_segs

                        with _region("gc_cuda"):
                            maybe_gc_cuda(device, threshold=gc_mem_threshold)
                        shard_progress.emit(
                            docs_done=doc_counter,
                            segments_done=len(all_feature_ids),
                        )

    elif all_text_chunks:
        # Sequential path: tokenize-only pipeline, inline CPU segmentation
        with ThreadPoolExecutor(max_workers=1) as tok_pool:
            pending_future = tok_pool.submit(
                _tokenize_chunk,
                model.tokenizer,
                all_text_chunks[0],
                context_window,
                sae_max_context,
            )

            for chunk_idx in range(len(all_text_chunks)):
                with _region("tokenize_wait"):
                    with record_function("tokenize_chunk"):
                        chunk_token_info = pending_future.result()

                if chunk_idx + 1 < len(all_text_chunks):
                    pending_future = tok_pool.submit(
                        _tokenize_chunk,
                        model.tokenizer,
                        all_text_chunks[chunk_idx + 1],
                        context_window,
                        sae_max_context,
                    )

                # Pre-compute segments + document-relative char offsets.
                chunk_precomputed = _precompute_batch_cpu_work(
                    chunk_token_info,
                    model.tokenizer,
                    segment_mode,
                    segment_size,
                    texts=(
                        all_text_chunks[chunk_idx]
                        if segment_mode is SegmentMode.turn_delimited_fixed_tokens
                        else None
                    ),
                    turn_char_starts=(
                        all_turn_start_chunks[chunk_idx]
                    ),
                )

                if context_independent_features:
                    assert bos_token_id is not None
                    # Pretraining seg-0 BOS precondition check (context-independent-only, loud).
                    if dataset_phase in (None, "base"):
                        for kept_ids, _ in chunk_token_info:
                            if kept_ids and kept_ids[0] != bos_token_id:
                                raise AssertionError(
                                    f"Pretraining doc has kept_ids[0]={kept_ids[0]} "
                                    f"!= bos_token_id={bos_token_id}; context-independent assumes "
                                    "auto-BOS prepend at extraction."
                                )
                    doc_counter = _run_context_independent_chunk(
                        model=model,
                        sae=sae,
                        chunk_token_info=chunk_token_info,
                        chunk_precomputed=chunk_precomputed,
                        layer=layer,
                        pooling=pooling,
                        gpu_batch_size=gpu_batch_size,
                        hook_name=hook_name,
                        activation_type=activation_type,
                        nnterp_model=nnterp_model,
                        native_model=native_model,
                        bos_token_id=bos_token_id,
                        ignore_bos_features=ignore_bos_features,
                        device=device,
                        gc_mem_threshold=gc_mem_threshold,
                        timer=timer,
                        all_feature_ids=all_feature_ids,
                        all_feature_mags=all_feature_mags,
                        doc_segment_info=doc_segment_info,
                        doc_token_counts=doc_token_counts,
                        doc_counter=doc_counter,
                    )
                    shard_progress.emit(
                        docs_done=doc_counter,
                        segments_done=len(all_feature_ids),
                    )
                else:
                    batch_starts = list(range(0, len(chunk_token_info), gpu_batch_size))
                    doc_bar = tqdm(
                        batch_starts,
                        desc=f"Chunk {chunk_idx + 1}/{len(all_text_chunks)}",
                        position=shard_position + 2,
                        leave=False,
                        disable=_tqdm_disabled(),
                    )

                    for batch_start in doc_bar:
                        with _region("batch_prep"):
                            batch_tok = chunk_token_info[batch_start : batch_start + gpu_batch_size]
                            batch_pre = chunk_precomputed[
                                batch_start : batch_start + gpu_batch_size
                            ]
                            assert len(batch_tok) == len(batch_pre)
                            batch_kept_ids = [kept_ids for kept_ids, _ in batch_tok]

                        with _region("model_forward_batched"):
                            sae_acts_list = _get_sae_activations_batched(
                                model,
                                sae,
                                batch_kept_ids,
                                layer=layer,
                                nnterp_model=nnterp_model,
                                native_model=native_model,
                                hook_name=hook_name,
                                activation_type=activation_type,
                            )

                        # GPU-only: pool using pre-computed segments
                        with _region("segment_and_pool"):
                            batch_pooled_gpu: list[t.Tensor] = []
                            batch_seg_counts: list[int] = []
                            batch_positions: list[list[int]] = []
                            batch_char_starts: list[list[int]] = []
                            batch_char_ends: list[list[int]] = []

                            for (
                                segments,
                                n_tokens,
                                char_starts,
                                char_ends,
                            ), sae_acts, kept_ids_doc in zip(
                                batch_pre,
                                sae_acts_list,
                                batch_kept_ids,
                                strict=True,
                            ):
                                with record_function("pool_segments"):
                                    doc_keep_mask = None
                                    if ignore_bos_features:
                                        assert bos_token_id is not None
                                        doc_keep_mask = (
                                            t.tensor(kept_ids_doc, device=sae_acts.device)
                                            != bos_token_id
                                        )
                                    pooled = pool_segments_batched_gpu(
                                        sae_acts,
                                        segments,
                                        pooling,
                                        segment_mode,
                                        keep_mask=doc_keep_mask,
                                    )
                                batch_pooled_gpu.append(pooled)
                                batch_seg_counts.append(len(segments))
                                batch_positions.append([start for start, _ in segments])
                                batch_char_starts.append(char_starts)
                                batch_char_ends.append(char_ends)
                                doc_token_counts.append(n_tokens)

                        if batch_pooled_gpu:
                            all_pooled = t.cat(batch_pooled_gpu, dim=0)
                            assert all_pooled.shape[0] == sum(batch_seg_counts), (
                                f"Pooled rows {all_pooled.shape[0]} != "
                                f"sum(seg_counts) {sum(batch_seg_counts)}"
                            )
                            with _region("extract_sparse"):
                                with record_function("extract_sparse"):
                                    batch_ids, batch_mags = extract_sparse_gpu(all_pooled)
                            del all_pooled
                        else:
                            batch_ids = []
                            batch_mags = []
                        del batch_pooled_gpu

                        with _region("result_accumulate"):
                            seg_offset = 0
                            for n_segs, positions, cstarts, cends in zip(
                                batch_seg_counts,
                                batch_positions,
                                batch_char_starts,
                                batch_char_ends,
                                strict=True,
                            ):
                                doc_ids = batch_ids[seg_offset : seg_offset + n_segs]
                                doc_mags = batch_mags[seg_offset : seg_offset + n_segs]
                                all_feature_ids.extend(doc_ids)
                                all_feature_mags.extend(doc_mags)
                                doc_segment_info.append(
                                    (doc_counter, n_segs, positions, cstarts, cends),
                                )
                                doc_counter += 1
                                seg_offset += n_segs

                        with _region("gc_cuda"):
                            maybe_gc_cuda(device, threshold=gc_mem_threshold)
                        shard_progress.emit(
                            docs_done=doc_counter,
                            segments_done=len(all_feature_ids),
                        )

    _validate_doc_accounting(doc_token_counts, doc_segment_info, doc_counter)
    shard_progress.emit(
        docs_done=doc_counter,
        segments_done=len(all_feature_ids),
        phase="extract_done",
        force=True,
    )

    # Vectorized offset computation via torch.cumsum
    with record_function("build_csr_metadata"):
        feat_counts = t.tensor(
            [len(feature_ids) for feature_ids in all_feature_ids],
            dtype=t.int64,
        )
        offsets = t.cat([t.zeros(1, dtype=t.int64), feat_counts.cumsum(0)])
        total_features = int(offsets[-1].item())

        # Vectorized metadata construction
        n_total_segments = len(all_feature_ids)
        doc_ids_list: list[int] = []
        seg_indices_list: list[int] = []
        seg_positions_list: list[int] = []
        seg_char_starts_list: list[int] = []
        seg_char_ends_list: list[int] = []
        for doc_id, n_segs, positions, cstarts, cends in doc_segment_info:
            doc_ids_list.extend([doc_id] * n_segs)
            seg_indices_list.extend(range(n_segs))
            seg_positions_list.extend(positions)
            seg_char_starts_list.extend(cstarts)
            seg_char_ends_list.extend(cends)

    # Save in CSR format with segment metadata
    with record_function("save_shard"):
        shard_data = ShardData(
            offsets=offsets,
            feature_ids=(
                t.cat(all_feature_ids) if all_feature_ids else t.tensor([], dtype=t.int32)
            ),
            feature_mags=(
                t.cat(all_feature_mags) if all_feature_mags else t.tensor([], dtype=t.float32)
            ),
            doc_ids=t.tensor(doc_ids_list, dtype=t.int32),
            segment_indices=t.tensor(seg_indices_list, dtype=t.int32),
            segment_positions=t.tensor(seg_positions_list, dtype=t.int32),
            doc_token_counts=t.tensor(doc_token_counts, dtype=t.int32),
            n_docs=doc_counter,
            n_segments=n_total_segments,
            layer=layer,
            context_window=context_window,
            pooling=pooling.value,
            segment_mode=segment_mode,
            segment_size=segment_size,
            segment_char_starts=t.tensor(seg_char_starts_list, dtype=t.int32),
            segment_char_ends=t.tensor(seg_char_ends_list, dtype=t.int32),
            context_independent_features=context_independent_features,
            ignore_bos_features=ignore_bos_features,
        )
        # Loud self-consistency check (catches a refactor that decouples the two).
        if shard_data.context_independent_features != context_independent_features:
            raise RuntimeError(
                "ShardData.context_independent_features does not match "
                "process_shard arg — refactor bug"
            )
        if shard_data.ignore_bos_features != ignore_bos_features:
            raise RuntimeError(
                "ShardData.ignore_bos_features does not match process_shard arg — refactor bug"
            )
        # CSR invariant checks before save (typed exceptions, not assert: -O safe)
        _validate_csr_invariants(
            offsets=shard_data.offsets,
            feature_ids=shard_data.feature_ids,
            feature_mags=shard_data.feature_mags,
            n_segments=shard_data.n_segments,
            doc_ids=shard_data.doc_ids,
            segment_indices=shard_data.segment_indices,
            segment_positions=shard_data.segment_positions,
            segment_char_starts=shard_data.segment_char_starts,
            segment_char_ends=shard_data.segment_char_ends,
        )
        shard_dict = shard_data.to_dict()
        if extraction_source_identity is not None:
            shard_dict.update(extraction_source_identity.metadata())
            shard_dict[MetadataKey.N_SAE_FEATURES] = int(sae.cfg.d_sae)
            shard_dict[MetadataKey.EXTRACTION_DOCUMENTS_SHA256] = extraction_documents_sha256(
                text for chunk in all_text_chunks for text in chunk
            )
        if sft_mixture_fingerprint_json is not None:
            shard_dict[SFT_MIXTURE_FINGERPRINT_METADATA_KEY] = sft_mixture_fingerprint_json
        # Stamp the gemma-scope-2 L0 tier so the shard is distinguishable
        # from a v1 (tier-less) shard and the cross-tier overwrite guard
        # (``assert_l0_tier_compatible``) has something to compare against.
        # This single call covers BOTH the async-saver and direct
        # save_features paths below; v1 / legacy runs stay tier-less.
        stamp_l0_tier(shard_dict, run_l0_tier)
        shard_dict["extraction_backend"] = extraction_backend.value
        out_stem = os.path.join(output_dir, f"{shard_name}_features")

        # Validate in-memory before either save path (fast — no disk I/O).
        validate_start = time.time()
        val_result = validate_shard_data(shard_dict)
        validate_elapsed = time.time() - validate_start
        if not val_result.valid:
            raise ValidationError(f"Validation failed for {shard_name}: {val_result.violations}")

        # Save (possibly in background)
        save_start = time.time()
        logger.info(
            "extract_features shard save start: shard={} docs={} staged={}",
            shard_name,
            doc_counter,
            background_saver is not None,
        )
        if background_saver is not None:
            background_saver.submit(shard_dict, out_stem)
            output_path = str(feature_path(output_dir, shard_name))
            save_elapsed = 0.0  # non-blocking
        else:
            output_path = str(save_features(shard_dict, out_stem, zstd_level=zstd_level))
            save_elapsed = time.time() - save_start

            # Validate the saved shard (file-based, legacy path)
            validate_start = time.time()
            val_result = validate_shard(output_path)
            validate_elapsed = time.time() - validate_start
            if not val_result.valid:
                raise ValidationError(
                    f"Validation failed for {output_path}: {val_result.violations}"
                )

    logger.info(
        f"Between-shard overhead: save={save_elapsed:.1f}s, "
        f"validate={validate_elapsed:.1f}s, total={save_elapsed + validate_elapsed:.1f}s"
    )

    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) if background_saver is None else 0

    return ShardResult(
        n_docs=doc_counter,
        n_tokens_gpt2=sum(doc_token_counts),
        n_total_features=int(total_features),
        elapsed_seconds=elapsed,
        output_path=output_path,
        bytes_on_disk=file_size,
    )


@beartype
@dataclass(frozen=True)
class ProcessShardParams:
    model: Any
    sae: Any
    layer: NonNegativeInt
    output_dir: str
    context_window: PositiveInt
    pooling: PoolingMode
    gpu_batch_size: PositiveInt
    cpu_batch_size: PositiveInt
    gc_mem_threshold: UnitInterval
    nnterp_model: Any
    native_model: Any
    segment_mode: SegmentMode
    segment_size: PositiveInt | None
    hook_name: str | None
    activation_type: ActivationType
    dataset_phase: str | None
    zstd_level: ZstdCompressionLevel | None
    context_independent_features: bool
    ignore_bos_features: bool
    run_l0_tier: L0Tier | None
    sft_mixture_fingerprint_json: str | None = None
    extraction_source_identity: ExtractionSourceIdentity | None = None


@beartype
@dataclass(frozen=True)
class ExtractionStagingConfig:
    """Typed staging values resolved once at the Hydra ingress boundary."""

    scratch_dir: str
    output_depth: NonNegativeInt
    input_depth: NonNegativeInt
    min_free_gb: NonNegativeFiniteNumber
    est_shard_gb: PositiveFiniteNumber


@beartype
@dataclass(frozen=True)
class ExtractionRuntimeConfig:
    """Scalar extraction configuration after dynamic Hydra/env parsing."""

    rank: NonNegativeInt
    world_size: PositiveInt
    local_rank: NonNegativeInt
    model_name: NonBlankStr
    layer: NonNegativeInt
    context_window: PositiveInt
    segment_size: PositiveInt | None
    gc_mem_threshold: UnitInterval
    zstd_level: ZstdCompressionLevel | None
    num_shards: NonNegativeInt
    start_shard: NonNegativeInt
    gpu_batch_size: PositiveInt | Literal["auto"]
    cpu_batch_size: PositiveInt | Literal["auto"]
    min_gpu_batch_size: PositiveInt
    min_cpu_batch_size: PositiveInt
    max_gpu_batch_size: PositiveInt
    max_cpu_batch_size: PositiveInt
    staging: ExtractionStagingConfig


def _safe_int_env(name: str, default: int) -> int:
    """``int(os.environ[name])`` with a fallback: an unset / empty / non-numeric value
    yields ``default`` instead of raising ``ValueError`` before any shard is processed."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer {}={!r}; using {}", name, raw, default)
        return default


def _scratch_run_key(output_dir: str) -> str:
    """Stable per-run scratch discriminator derived from the absolute ``output_dir``.

    Concurrent extractions to different output_dirs on one node (same global rank,
    same ``staging.scratch_dir``) use disjoint base subtrees. The configured shard
    window is added by :func:`_scratch_range_key` below, so disjoint windows for one
    output_dir are isolated as well. Resuming the same output_dir and window reuses
    the same key, so a crashed run's leftovers are swept on restart. Tradeoff
    (deliberate, strictly safer than the cross-run sweep it replaces): a run
    hard-killed before its exit sweep whose output_dir/window is then never resumed
    leaves up to ``output_depth × shard_size`` of staged files that no other run
    will sweep. Keyed on ``abspath`` (CWD-stable, symlink-sensitive) — the standard
    absolute-output_dir resume path is unaffected.
    """
    digest = hashlib.sha1(os.path.abspath(output_dir).encode()).hexdigest()[:12]
    return f"run_{digest}"


@beartype
def _scratch_range_key(shard_start: NonNegativeInt, shard_end: NonNegativeInt) -> str:
    """Return the canonical scratch component for an exclusive shard window."""
    return f"range_{int(shard_start)}_{int(shard_end)}"


@beartype
def run_extraction_loop(
    params: ProcessShardParams,
    parquet_paths: list[str],
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    shard_start: NonNegativeInt,
    shard_end: NonNegativeInt,
    staging_cfg: ExtractionStagingConfig,
    profile_enabled: bool,
    profile_output: str,
    flops_per_token: PositiveFiniteNumberLike,
    gpu_peak_flops: float,
) -> list[ShardResult]:
    """Run the per-shard extraction loop.

    Output is staged save->move when ``staging_cfg.output_depth>=1`` (else
    synchronous direct save). Input is N-deep parquet->scratch->read staging via
    :class:`StagedPrefetcher` when ``staging_cfg.input_depth>=1`` (else inline
    ``process_shard`` read with ``prefetched=None``). Scratch is namespaced by
    the absolute output directory, the configured exclusive shard window, and
    rank: ``run_<output>/range_<start>_<end>/rank{rank}/{out,in}``. Thus
    disjoint range jobs targeting one output directory cannot sweep each
    other's in-flight files. Decompressed output contents are identical across
    both staging depths."""
    rank_prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""
    results: list[ShardResult] = []
    # Pre-filter already-extracted shards up front so resume never STAGES or reads them:
    # a staged prefetch otherwise copies+parses each done shard's full parquet before any
    # in-loop skip could discard it (issue #564 review). Per-rank assignment means no other
    # rank creates this rank's shards, so this is race-free.
    pending_paths = [
        p for p in parquet_paths if not feature_path_exists(params.output_dir, Path(p).stem)
    ]
    if pending_paths:
        assert_extraction_backend_compatible(
            params.output_dir,
            resolved_extraction_backend(params.nnterp_model, params.native_model),
        )
    n_skipped = len(parquet_paths) - len(pending_paths)
    if n_skipped:
        logger.info(f"{rank_prefix}Skipping {n_skipped} already-extracted shard(s)")
    shard_bar = tqdm(
        pending_paths,
        desc="Shards",
        position=0,
        total=len(pending_paths),
        disable=_tqdm_disabled(),
    )
    sidecar = f"{profile_output}_rank{rank}"
    output_depth = staging_cfg.output_depth
    input_depth = staging_cfg.input_depth
    local_world_size = NonNegativeInt.parse(_safe_int_env("LOCAL_WORLD_SIZE", world_size))
    # Per-output, per-range scratch subtree. The range is the configured
    # [start_shard, num_shards) window, not this rank's striped parquet_paths;
    # every rank in one launch must share the same parent while sibling range
    # jobs remain disjoint.
    run_scratch = (
        Path(staging_cfg.scratch_dir)
        / _scratch_run_key(params.output_dir)
        / _scratch_range_key(shard_start, shard_end)
    )
    if output_depth >= 1:
        destination_dir = Path(params.output_dir)
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise_destination_space_error(destination_dir, e)
            raise
        run_scratch.mkdir(parents=True, exist_ok=True)
        # Local rank 0 is the SOLE free-space checker on the shared scratch FS; the other
        # local ranks wait for its verdict and fail if it reports a shortage. Per-rank
        # checks raced on staggered model-load finishes (no torch.distributed group here),
        # so this gates all ranks on rank 0. Per-launch run_id avoids stale-resume reads.
        coordinate_free_space_check(
            run_scratch,
            local_rank=local_rank,
            run_id=os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("MASTER_PORT")
            or "default",
            output_depth=output_depth,
            est_shard_gb=staging_cfg.est_shard_gb,
            local_world_size=local_world_size,
            min_free_gb=staging_cfg.min_free_gb,
            destination_dir=destination_dir,
        )
        saver: StagedSaver | None = StagedSaver(
            scratch_dir=run_scratch / f"rank{rank}" / "out",
            final_dir=params.output_dir,
            depth=PositiveInt.parse(output_depth),
            zstd_level=params.zstd_level,
            min_free_gb=staging_cfg.min_free_gb,
        )
    else:
        saver = None
    if input_depth >= 1:
        load_fn = functools.partial(
            _staged_prefetch_load,
            cpu_batch_size=params.cpu_batch_size,
            dataset_phase=params.dataset_phase,
            segment_mode=params.segment_mode,
        )
        prefetcher: StagedPrefetcher | None = StagedPrefetcher(
            scratch_dir=run_scratch / f"rank{rank}" / "in",
            depth=PositiveInt.parse(input_depth),
            load_fn=load_fn,
        )
    else:
        prefetcher = None
    # M1: enter saver FIRST, then prefetcher, then profiler LAST so that a
    # profiler __enter__ failure (profile=true) still unwinds the already-
    # registered saver/prefetcher. The sidecar never reads saver/prefetcher
    # state, so teardown order is otherwise irrelevant.
    with ExitStack() as stack:
        if saver is not None:
            stack.enter_context(saver)
        if prefetcher is not None:
            stack.enter_context(prefetcher)
        profile = stack.enter_context(
            extraction_profiler(
                enabled=profile_enabled,
                rank=rank,
                local_rank=local_rank,
                sidecar_path=sidecar,
            )
        )
        if prefetcher is not None:
            # Prime <= input_depth in flight; the loop then get()s before each
            # submit() so the FIFO never exceeds depth. Only pending (not-yet-extracted)
            # shards are staged — done shards were filtered out above.
            for j in range(min(input_depth, len(pending_paths))):
                prefetcher.submit(pending_paths[j])
        for i, parquet_path in enumerate(shard_bar):
            with timed_region("input_stall", sync_cuda=False):
                if prefetcher is not None:
                    prefetched = prefetcher.get()
                else:
                    prefetched = None  # process_shard does the inline read
            if prefetcher is not None:
                nxt = i + input_depth
                if nxt < len(pending_paths):
                    prefetcher.submit(pending_paths[nxt])
            # pending_paths are all not-yet-extracted (pre-filtered), so process every one.
            with timed_region("compute", sync_cuda=True):
                result = process_shard(
                    parquet_path,
                    params.model,
                    params.sae,
                    layer=params.layer,
                    output_dir=params.output_dir,
                    context_window=params.context_window,
                    pooling=params.pooling,
                    shard_position=0,
                    gpu_batch_size=params.gpu_batch_size,
                    cpu_batch_size=params.cpu_batch_size,
                    gc_mem_threshold=params.gc_mem_threshold,
                    nnterp_model=params.nnterp_model,
                    native_model=params.native_model,
                    segment_mode=params.segment_mode,
                    segment_size=params.segment_size,
                    hook_name=params.hook_name,
                    activation_type=params.activation_type,
                    dataset_phase=params.dataset_phase,
                    background_saver=saver,
                    prefetched=prefetched,
                    zstd_level=params.zstd_level,
                    context_independent_features=params.context_independent_features,
                    ignore_bos_features=params.ignore_bos_features,
                    run_l0_tier=params.run_l0_tier,
                    sft_mixture_fingerprint_json=params.sft_mixture_fingerprint_json,
                    extraction_source_identity=params.extraction_source_identity,
                )
            if profile is not None:
                profile.mark_shard_done()
            results.append(result)
            tok_per_sec = result.n_tokens_gpt2 / max(result.elapsed_seconds, 1e-9)
            mfu = 100 * flops_per_token * tok_per_sec / gpu_peak_flops
            shard_bar.set_postfix(
                docs=result.n_docs,
                time=f"{result.elapsed_seconds:.0f}s",
                mfu=f"{mfu:.1f}%",
                inflight=saver.in_flight if saver is not None else 0,
            )
        # Drain the staged save pipeline INSIDE the profiler region (saver/prefetcher
        # are entered first, so they tear down AFTER the profiler stops). Without this
        # the final in-flight save+relocate — GPU-idle, network-bound — drains after
        # wall_seconds is latched, optimistically biasing gpu_idle_fraction. No-op
        # when staging is off (saver is None).
        if saver is not None:
            with timed_region("save_drain", sync_cuda=False):
                saver.drain()
    # Clean up THIS rank's scratch subtree (empty after its saver/prefetcher swept their
    # files) so range-keyed scratch doesn't leak empty rank{rank}/{out,in} trees. Per-rank
    # (NOT the shared run_scratch) so a fast-finishing rank or sibling range never deletes
    # another rank's or range's in-flight scratch.
    if saver is not None or prefetcher is not None:
        shutil.rmtree(run_scratch / f"rank{rank}", ignore_errors=True)
    return results


def _warn_if_path_flag_mismatch(
    output_dir: str,
    context_independent_flag: bool,
    ignore_bos_flag: bool = False,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> None:
    """Soft-warn when an explicit output_dir's trailing leaves disagree with
    the flags. Loud read-time enforcement is verify_feature_dir."""
    parts = Path(output_dir).parts
    present: dict[str, bool] = {}
    for leaf in ("turn_delimited", "no_bos", "context_independent"):
        present[leaf] = bool(parts and parts[-1] == leaf)
        if present[leaf]:
            parts = parts[:-1]
    rows = (
        (
            "context_independent_features",
            context_independent_flag,
            context_independent_flag,
            "context_independent",
        ),
        ("ignore_bos_features", ignore_bos_flag, ignore_bos_flag, "no_bos"),
        (
            "segment_mode",
            segment_mode.value,
            segment_mode is SegmentMode.turn_delimited_fixed_tokens,
            "turn_delimited",
        ),
    )
    for label, display, flag, leaf in rows:
        if flag != present[leaf]:
            logger.warning(
                f"{label}={display} but output_dir {leaf} leaf present="
                f"{present[leaf]} — possible mismatch. output_dir={output_dir!r}"
            )


# The conditioned SFT trainer keeps ``max_seq_len`` *nanochat* tokens of every
# conversation and requires the feature shard to cover that prefix (its
# coverage gate tolerates a 20-char gap). Extraction truncates to
# ``context_window`` tokens of the *feature model*, whose tokenizer can need more
# tokens than nanochat's for the same text (Gemma splits digits, long
# multi-turn rows), so an SFT window equal to the trainer's ``max_seq_len``
# leaves long rows uncovered (#1634). Production SFT shards use 2048 for the
# trainer's default max_seq_len=1024; the rule is context_window >= 2 x max_seq_len.
SFT_RECOMMENDED_MIN_CONTEXT_WINDOW = 2048


def _log_sft_context_window_advice(context_window: int, dataset_phase: str | None) -> None:
    """Warn when an SFT extraction window is likely to trip the trainer's coverage gate."""
    if dataset_phase in (None, "base"):
        return
    rule = (
        "the conditioned SFT trainer keeps max_seq_len nanochat tokens per conversation "
        "and the feature model's tokenizer can need more tokens than nanochat for the same "
        "text; use context_window >= 2 x the trainer's max_seq_len (production: 2048 for "
        "max_seq_len=1024, 4096 for 2048) or long rows fail the trainer's coverage gate (#1634)"
    )
    if context_window < SFT_RECOMMENDED_MIN_CONTEXT_WINDOW:
        logger.warning(
            f"SFT extraction context_window={context_window} is below "
            f"{SFT_RECOMMENDED_MIN_CONTEXT_WINDOW} (2 x the trainer default "
            f"max_seq_len=1024): {rule}"
        )
    else:
        logger.info(
            f"SFT extraction context_window={context_window} meets the 2048 floor for the "
            f"trainer default max_seq_len=1024; scale it for a larger max_seq_len: {rule}"
        )


def _validate_segment_mode_dataset_phase(
    segment_mode: SegmentMode,
    dataset_phase: str | None,
) -> None:
    """Reject turn segmentation for pretraining documents, which have no turns."""
    if (
        segment_mode is SegmentMode.turn_delimited_fixed_tokens
        and dataset_phase in (None, "base")
    ):
        raise ValueError(
            "turn-delimited segmentation requires an SFT dataset_phase; "
            "pretraining documents have no turns"
        )


def _resolve_ignore_bos_features(cfg: DictConfig) -> bool:
    """Resolve the producer's ``ignore_bos_features`` value: always ``True``.

    BOS-feature exclusion is frozen ALWAYS-ON in the extraction producer (``<bos>``
    is a large attention-sink outlier; excluding every ``<bos>``-id token from
    per-segment pooling is the only supported extraction behaviour). It is no
    longer a user-facing config option.

    A value still present in the resolved config — e.g. via
    ``+ignore_bos_features=...`` or a stale ``local@_global_`` override — is
    IGNORED and reported via ``logger.error`` (loud but non-fatal) so the user
    notices the dead key. ``"... in cfg"`` is the OmegaConf-struct-safe presence
    check; ``getattr`` cannot distinguish a present ``false`` from an absent key.

    The compute/format/consumer layers keep their ``ignore_bos_features``
    parameter (default ``False``, both values reachable) so legacy artifacts
    extracted with the flag off remain readable — and re-extractable by
    ``validate_features`` with their recorded flag.
    """
    if "ignore_bos_features" in cfg:
        warn_ignore_bos_cfg_key_ignored(
            cfg.ignore_bos_features,
            resolved=PRODUCER_IGNORE_BOS_FEATURES,
        )
    return PRODUCER_IGNORE_BOS_FEATURES


@beartype
def _assert_context_window_within_sae_max(
    model: NonBlankStr,
    context_window: PositiveInt,
) -> None:
    """Fail loud at startup when ``context_window`` exceeds the SAE truncation cap.

    Extraction head-truncates every document to ``min(context_window, sae_max)`` (via
    ``batch_truncate_to_training_window``) but records the RAW ``context_window`` in
    each shard's metadata. An over-set ``context_window`` therefore produces shards
    whose recorded window exceeds reality, silently poisoning downstream consumers
    (e.g. ``precompute_token_boundaries`` fully-covered classification, the
    feature-shard splice). We refuse to start rather than clamp/warn/record ``min()``.

    ``resolve_sae_max_context`` reads the cap weight-free from the HF ``AutoConfig``
    (no model load), so this is the earliest possible loud failure. It matches the
    ``model.cfg.n_ctx`` extraction actually truncates with for the supported models.
    """
    sae_max_context = resolve_sae_max_context(model)
    if context_window > sae_max_context:
        raise ValueError(
            f"context_window={context_window} exceeds the SAE max context "
            f"({sae_max_context}) for model {model!r}: extraction would silently "
            f"truncate every document to {sae_max_context} tokens while recording "
            f"context_window={context_window} in shard metadata, which is forbidden. "
            f"Set context_window <= {sae_max_context}."
        )


@hydra.main(
    config_path="../configs/extraction",
    config_name="extract_features",
    version_base=None,
)
def main(cfg: DictConfig):
    # Multi-process support: compatible with torchrun env vars
    staging_cfg = ExtractionStagingConfig(
        scratch_dir=str(cfg.staging.scratch_dir),
        output_depth=cfg.staging.output_depth,
        input_depth=cfg.staging.input_depth,
        min_free_gb=cfg.staging.min_free_gb,
        est_shard_gb=cfg.staging.est_shard_gb,
    )
    gpu_batch_size_cfg: PositiveInt | Literal["auto"] = (
        "auto" if str(cfg.gpu_batch_size) == "auto" else cfg.gpu_batch_size
    )
    cpu_batch_size_cfg: PositiveInt | Literal["auto"] = (
        "auto" if str(cfg.cpu_batch_size) == "auto" else cfg.cpu_batch_size
    )
    runtime_cfg = ExtractionRuntimeConfig(
        rank=cast(Any, int(os.environ.get("RANK", "0"))),
        world_size=cast(Any, int(os.environ.get("WORLD_SIZE", "1"))),
        local_rank=cast(Any, int(os.environ.get("LOCAL_RANK", "0"))),
        model_name=cfg.model,
        layer=cfg.layer,
        context_window=cfg.context_window,
        segment_size=(None if cfg.segment_size is None else cfg.segment_size),
        gc_mem_threshold=cfg.gc_mem_threshold,
        zstd_level=(None if cfg.get("zstd_level", None) is None else cfg.zstd_level),
        num_shards=cfg.num_shards,
        start_shard=cfg.start_shard,
        gpu_batch_size=gpu_batch_size_cfg,
        cpu_batch_size=cpu_batch_size_cfg,
        min_gpu_batch_size=cast(Any, getattr(cfg, "min_gpu_batch_size", 1)),
        min_cpu_batch_size=cast(Any, getattr(cfg, "min_cpu_batch_size", 1)),
        max_gpu_batch_size=cast(Any, getattr(cfg, "max_gpu_batch_size", 512)),
        max_cpu_batch_size=cast(Any, getattr(cfg, "max_cpu_batch_size", 512)),
        staging=staging_cfg,
    )
    rank = runtime_cfg.rank
    world_size = runtime_cfg.world_size
    local_rank = runtime_cfg.local_rank
    model_name = runtime_cfg.model_name
    layer = runtime_cfg.layer
    context_window = runtime_cfg.context_window
    segment_size = runtime_cfg.segment_size
    gc_mem_threshold = runtime_cfg.gc_mem_threshold
    zstd_level = runtime_cfg.zstd_level
    device = f"cuda:{local_rank}" if cfg.device.startswith("cuda") else cfg.device

    # Fail loud BEFORE the model load if context_window exceeds the SAE truncation cap:
    # extraction would silently truncate yet record the raw context_window in metadata.
    _assert_context_window_within_sae_max(model_name, context_window)

    # Resolve dataset phase
    dataset_phase = getattr(cfg, "dataset_phase", None)
    segment_mode = SegmentMode(str(cfg.segment_mode))
    _validate_segment_mode_dataset_phase(segment_mode, dataset_phase)
    _log_sft_context_window_advice(context_window, dataset_phase)
    context_independent_features = bool(getattr(cfg, "context_independent_features", False))
    ignore_bos_features = _resolve_ignore_bos_features(cfg)

    # Resolve data directory
    data_dir = str(resolve_data_dir(cfg.data_dir, dataset_phase))

    # Check SFT data readiness
    if dataset_phase is not None and dataset_phase != "base":
        ok, msg = check_sft_data_ready(data_dir, dataset_phase)
        if not ok:
            logger.error(msg)
            return

    # Resolve output directory (auto-construct or explicit override)
    activation_type = ActivationType(cfg.activation_type)
    pooling = PoolingMode(cfg.pooling)

    if cfg.output_dir is not None:
        output_dir = cfg.output_dir
    else:
        width = SAEWidth(cfg.sae.num_features) if cfg.sae is not None else None
        if width is None:
            raise ValueError("Cannot auto-construct output_dir without sae.num_features")
        output_dir = str(
            resolve_output_dir(
                output_dir=None,
                feature_root=cfg.feature_root,
                segment_size=segment_size or context_window,
                layer=layer,
                activation_type=activation_type,
                pooling=pooling,
                width=width,
                model=model_name,
                dataset_phase=dataset_phase,
                context_independent_features=context_independent_features,
                ignore_bos_features=ignore_bos_features,
                segment_mode=segment_mode,
            )
        )

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise_destination_space_error(Path(output_dir), e)
        raise
    _warn_if_path_flag_mismatch(
        output_dir,
        context_independent_features,
        ignore_bos_features,
        segment_mode,
    )

    sft_mixture_fingerprint_json = _load_sft_conversion_fingerprint_json(data_dir)
    _assert_existing_output_dir_compatible(
        output_dir,
        segment_mode,
        sft_mixture_fingerprint_json,
    )

    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        logger.error(
            "No parquet files found. Download shards first: python -m nanochat.dataset -n 2"
        )
        return

    all_in_window = select_shard_paths(
        parquet_paths,
        runtime_cfg.num_shards,
        runtime_cfg.start_shard,
        rank=NonNegativeInt.parse(0),
        world_size=PositiveInt.parse(1),
    )
    n_shards_total = len(all_in_window)
    parquet_paths = all_in_window[rank::world_size]
    n_shards = len(parquet_paths)

    if n_shards == 0:
        logger.info(f"[rank {rank}/{world_size}] No shards assigned, exiting")
        return

    rank_prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""
    logger.info(
        f"{rank_prefix}Processing {n_shards} shard(s) (of {n_shards_total} total) — "
        f"pooling={pooling.value}, "
        f"segment={segment_mode.value} (size={segment_size or context_window}), "
        f"model={model_name}, activation={activation_type.value}, "
        f"context_independent_features={context_independent_features}, "
        f"ignore_bos_features={ignore_bos_features}"
    )

    resolved = _resolve_extraction_sae(
        cfg,
        model_name=model_name,
        activation_type=activation_type,
        layer=layer,
    )
    hook_name = resolved.hook_name
    model, sae, tokenizer = load_sae_and_model(
        model_name,
        layer,
        device,
        resolved=resolved,
    )
    logger.info(
        f"Resolved SAE: release={resolved.release}, sae_id={resolved.sae_id}, "
        f"hook={resolved.hook_name}"
    )

    # ── L0-tier protection ────────────────────────────────────────────────
    # gemma-scope-2 runs stamp their L0Tier into every shard; v1 runs stay
    # tier-less (key absent, preserving legacy L0 metadata).
    # Guard the output dir *before any write* so a cross-tier (or v1↔v2)
    # overwrite into an existing dir fails fast instead of silently mixing
    # incompatible shards. Resume/append into a matching-tier dir is allowed.
    run_l0_tier: L0Tier | None = l0_tier_for_resolved(resolved)
    assert_l0_tier_compatible(output_dir, run_l0_tier)
    if run_l0_tier is not None:
        logger.info(f"L0 tier for this run: {run_l0_tier.value} (gemma-scope-2)")

    # Resolve model dtype (float32 or bfloat16)
    dtype_str = getattr(cfg, "model_dtype", "float32")
    model_dtype = {"float32": t.float32, "bfloat16": t.bfloat16}[dtype_str]

    # Build the accelerated backend (or hard-abort); TL only via explicit backend=transformerlens.
    backend = Backend(getattr(cfg, "backend", "auto"))
    nn_model, native_model = _build_accelerated_backend(backend, model, device, model_dtype, cfg)

    # Offload the TransformerLens model to CPU when an accelerated backend
    # (nnterp or native) is active. Native already did this before its HF load
    # inside _build_accelerated_backend to avoid co-resident TL+native peaks;
    # nnterp may do the same internally for Gemma-2. This post-build call is
    # therefore idempotent and keeps the tokenizer/cfg available from CPU.
    if nn_model is not None or native_model is not None:
        model.to("cpu")
        t.cuda.empty_cache()

    # Convert SAE to match model dtype for full bf16 pipeline
    if model_dtype != t.float32:
        sae.to(model_dtype)
        logger.info(f"Converted SAE to {model_dtype}")

    # Mandatory self-verifying torch.compile gate. Strictly downstream of
    # _build_accelerated_backend and NATIVE-ONLY (see _run_compile_gate's
    # docstring for the full rationale); no-op only when native_model is None.
    _run_compile_gate(cfg, native_model, sae, device, model_dtype)

    # Mandatory self-verifying SDPA row-packing gate — runs AFTER the compile
    # gate so its unpacked baseline + packed candidate share the run's compile
    # state. On PASS it activates always-on packing for the native batched
    # forward; on FAIL it hard-aborts (RowPackGateError). NATIVE-ONLY; no-op
    # when native_model is None.
    _run_rowpack_gate(
        cfg,
        native_model,
        sae,
        model_dtype,
        pooling=pooling,
        segment_size=segment_size or context_window,
        context_independent_features=context_independent_features,
        ignore_bos_features=ignore_bos_features,
    )

    # MFU estimation: forward-only FLOPs for partial model + SAE
    flops_per_token = estimate_extraction_flops_per_token(
        model,
        sae,
        layer,
        context_window,
    )
    gpu_peak_flops = (
        get_peak_flops(t.cuda.get_device_name()) if device.startswith("cuda") else float("inf")
    )
    logger.info(
        f"{rank_prefix}FLOPs/token: {flops_per_token:.2e} "
        f"(model layers 0..{cfg.layer} + SAE, forward only)"
    )

    # Resolve gpu_batch_size and cpu_batch_size
    gpu_auto = runtime_cfg.gpu_batch_size == "auto"
    cpu_auto = runtime_cfg.cpu_batch_size == "auto"

    if gpu_auto or cpu_auto:
        logger.info("Auto-tuning batch sizes...")
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
            logger.info(
                "Auto-tuner probes fixed windows of the same segment_size: turn "
                "pooling now uses the same segmented reduction, so throughput is "
                "representative and peak activation memory is identical for "
                "max_pool. float16/bfloat16 mean_pool additionally accumulates in "
                "float32, so its peak exceeds the probe by one float32 copy of the "
                "document block"
            )
        auto_gpu, auto_cpu = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=context_window,
            nnterp_model=nn_model,
            native_model=native_model,
            pooling=pooling,
            segment_size=segment_size or context_window,
            gc_mem_threshold=gc_mem_threshold,
            hook_name=hook_name,
            activation_type=activation_type,
            parquet_paths=parquet_paths,
            min_gpu_batch_size=runtime_cfg.min_gpu_batch_size,
            min_cpu_batch_size=runtime_cfg.min_cpu_batch_size,
            max_gpu_batch_size=runtime_cfg.max_gpu_batch_size,
            max_cpu_batch_size=runtime_cfg.max_cpu_batch_size,
            context_independent_features=context_independent_features,
            ignore_bos_features=ignore_bos_features,
        )
        gpu_batch_size = auto_gpu if gpu_auto else runtime_cfg.gpu_batch_size
        cpu_batch_size = auto_cpu if cpu_auto else runtime_cfg.cpu_batch_size
    else:
        gpu_batch_size = runtime_cfg.gpu_batch_size
        cpu_batch_size = runtime_cfg.cpu_batch_size

    logger.info(f"Batch sizes: gpu={gpu_batch_size}, cpu={cpu_batch_size}")

    params = ProcessShardParams(
        model=model,
        sae=sae,
        layer=layer,
        output_dir=output_dir,
        context_window=context_window,
        pooling=pooling,
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        gc_mem_threshold=gc_mem_threshold,
        nnterp_model=nn_model,
        native_model=native_model,
        segment_mode=segment_mode,
        segment_size=segment_size,
        hook_name=hook_name,
        activation_type=activation_type,
        dataset_phase=dataset_phase,
        zstd_level=zstd_level,
        context_independent_features=context_independent_features,
        ignore_bos_features=ignore_bos_features,
        run_l0_tier=run_l0_tier,
        sft_mixture_fingerprint_json=sft_mixture_fingerprint_json,
        extraction_source_identity=ExtractionSourceIdentity(str(model_name), resolved.release, resolved.sae_id),
    )
    total_start = time.time()
    all_results = run_extraction_loop(
        params,
        parquet_paths,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        shard_start=runtime_cfg.start_shard,
        shard_end=runtime_cfg.num_shards,
        staging_cfg=runtime_cfg.staging,
        profile_enabled=bool(cfg.get("profile", False)),
        profile_output=cfg.get("profile_output", "out/traces/extraction_profile"),
        flops_per_token=flops_per_token,
        gpu_peak_flops=gpu_peak_flops,
    )
    total_elapsed = time.time() - total_start
    total_docs = sum(result.n_docs for result in all_results)
    total_tokens = sum(result.n_tokens_gpt2 for result in all_results)
    logger.info(
        f"{rank_prefix}Total: {total_docs:,} docs, {total_tokens:,} tokens in {total_elapsed:.1f}s"
    )
    if total_docs > 0:
        tok_per_sec = total_tokens / total_elapsed
        mfu = 100 * flops_per_token * tok_per_sec / gpu_peak_flops
        logger.info(
            f"{rank_prefix}  {total_elapsed / total_docs * 1000:.1f} ms/doc | "
            f"{total_elapsed / total_tokens * 1e6:.1f} us/token | "
            f"{tok_per_sec:,.0f} tok/s | mfu: {mfu:.1f}%"
        )


if __name__ == "__main__":
    main()
