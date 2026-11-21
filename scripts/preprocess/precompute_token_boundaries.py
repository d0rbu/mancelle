"""Precompute nanochat-tokenizer token boundaries as overlay shards.

For each base feature shard, computes per-segment (token_start, token_end)
ranges and writes them to ``tok/<checksum[:16]>/seg_boundaries/shard_NNNNN.safetensors.zst``.
Base shards are never modified.

The overlay directory name encodes the tokenizer checksum, and a ``manifest.json``
stores the full checksum for exact validation.

Usage::

    uv run python -m scripts.preprocess.precompute_token_boundaries

    # Override specific params:
    uv run python -m scripts.preprocess.precompute_token_boundaries layer=12 sae_width=32k

    # Explicit feature dir:
    uv run python -m scripts.preprocess.precompute_token_boundaries feature_dir=path/to/features

    # Resume (skip shards with existing overlay):
    uv run python -m scripts.preprocess.precompute_token_boundaries resume=true
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from errors import EmptyExtractionSegmentError
from nanochat.dataset import DATA_DIR, list_parquet_files
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess._common import resolve_feature_dir
from scripts.preprocess._tok_boundary_profile import ShardTiming, TokenBoundaryProfile
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    empty_segment_indices,
    glob_features,
    load_features,
    read_shard_metadata,
    shard_stem_from_path,
)
from steering.feature_paths import SegmentSizeMismatch, parse_feature_dir, verify_segment_size
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayIncompleteError,
    OverlayKind,
    _compute_base_hash,
    discover_overlay,
    expected_overlay_shard_stems,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    validate_token_boundary_overlay_shard,
    write_manifest,
)
from steering.parquet_io import (
    detect_source_bos_char_len_from_data as _detect_source_bos_char_len_from_data,
)
from steering.parquet_io import read_parquet_texts
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.segmentation import (
    CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    SegmentMode,
    assign_tokens_to_segments,
    compute_tokenizer_checksum,
    get_path_stats,
    segment_mode_from_metadata,
    token_to_char_offsets,
)
from steering.stats_io import find_statistics_path, save_tokenizer_checksum
from steering.token_boundary_remote import (
    AuthenticatedTokenBoundarySource,
    TokenBoundaryRemoteConfig,
    remote_output_lock,
)
from steering.token_boundary_source import load_token_boundary_source_shard_stable

_DOC_BOUNDARY_TASK_CHUNK_SIZE = 64
_WORKER_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

# Tensor fields each load path actually consumes (issue #529). The loader is ALWAYS called with
# these keys (subset loading is unconditional), so it skips materializing the large
# feature_ids/feature_mags tensors while keeping the #522 non-empty guard (it needs ``offsets``)
# and the boundary computation byte-identical to a full load.
_PRECOMPUTE_SHARD_LOAD_KEYS: set[str] = {
    TensorKey.OFFSETS,
    TensorKey.DOC_IDS,
    TensorKey.SEGMENT_INDICES,
    TensorKey.SEGMENT_CHAR_STARTS,
    TensorKey.SEGMENT_CHAR_ENDS,
}
_DETECT_BOS_LOAD_KEYS: set[str] = {
    TensorKey.DOC_IDS,
    TensorKey.SEGMENT_CHAR_ENDS,
    TensorKey.DOC_TOKEN_COUNTS,
}


class MissingParquetInputError(FileNotFoundError):
    """One or more pending nonempty token-overlay shards lack source parquet."""


class TokenBoundaryAlignmentError(ValueError):
    """A feature shard, parquet, or tokenizer batch has incompatible document rows."""


class _BoundaryTokenizer(Protocol):
    """Tokenizer capability required by the process-pool boundary workers."""

    def encode(self, text: str, /, *, prepend: int) -> list[int]: ...

    def decode(self, token_ids: list[int], /) -> str: ...


def _raise_for_missing_pending_parquets(
    pending_nonempty: list[Path],
    parquet_by_stem: dict[str, Path],
) -> None:
    """Fail once for every selected pending shard without a parquet input."""
    missing_stems = [
        shard_stem_from_path(path)
        for path in pending_nonempty
        if shard_stem_from_path(path) not in parquet_by_stem
    ]
    if missing_stems:
        raise MissingParquetInputError(
            f"{len(missing_stems)} pending nonempty token-overlay shard(s) "
            "lack source parquet inputs; representative "
            f"stems={missing_stems[:5]}. No output was written."
        )

# ---------------------------------------------------------------------------
# Worker functions for parallel precomputation (must be module-level for spawn)
# ---------------------------------------------------------------------------

_worker_tokenizer: _BoundaryTokenizer | None = None
_worker_source_bos_char_len: int = 0
_worker_source_segment_size: int = 0
_worker_max_nanochat_tokens_per_segment: int | None = None
_worker_min_nanochat_tokens_per_segment: int | None = None


def _init_worker(
    tokenizer: _BoundaryTokenizer,
    source_bos_char_len: int,
    source_segment_size: int,
    max_nanochat_tokens_per_segment: int | None = None,
    min_nanochat_tokens_per_segment: int | None = None,
):
    global _worker_tokenizer
    global _worker_source_bos_char_len, _worker_source_segment_size
    global _worker_max_nanochat_tokens_per_segment
    global _worker_min_nanochat_tokens_per_segment
    _worker_tokenizer = tokenizer
    _worker_source_bos_char_len = source_bos_char_len
    _worker_source_segment_size = source_segment_size
    _worker_max_nanochat_tokens_per_segment = max_nanochat_tokens_per_segment
    _worker_min_nanochat_tokens_per_segment = min_nanochat_tokens_per_segment


def _apply_worker_thread_env() -> dict[str, str | None]:
    """Cap native math-library threads inherited by spawn children.

    Returns the previous values (None = unset) so the caller can restore them
    once the pool is shut down. The mutation must span the pool's whole
    lifetime — ProcessPoolExecutor spawns workers lazily on first submit, not
    at construction — so restoration belongs in run()'s ``finally`` after
    ``shutdown()``, never immediately after pool creation (review finding,
    issue #1269).
    """
    saved: dict[str, str | None] = {
        key: os.environ.get(key) for key in _WORKER_THREAD_ENV_VARS
    }
    os.environ.update(_WORKER_THREAD_ENV_VARS)
    return saved


def _restore_worker_thread_env(saved: dict[str, str | None]) -> None:
    """Undo ``_apply_worker_thread_env`` (delete keys that were unset)."""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _init_worker_process(
    tokenizer: _BoundaryTokenizer,
    source_bos_char_len: int,
    source_segment_size: int,
    max_nanochat_tokens_per_segment: int | None = None,
    min_nanochat_tokens_per_segment: int | None = None,
) -> None:
    """Initialize a pool worker with one torch/native thread per process.

    This initializer is pool-only: applying ``set_num_interop_threads`` to the parent
    process after prior parallel work has started raises. The old uncapped setup measured
    1,144 runnable threads on a 32-CPU cpuset (issue #1269), so spawned workers must each
    cap torch intra-op/inter-op and native BLAS/OpenMP threads to one.
    """
    t.set_num_threads(1)
    try:
        t.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise RuntimeError(
            "token-boundary worker could not cap torch inter-op threads to 1; "
            "something in the spawn bootstrap (e.g. an unguarded main module) "
            "already configured or started inter-op parallelism. The #1269 "
            "oversubscription fix requires capped workers — refusing to run "
            "uncapped."
        ) from exc
    _init_worker(
        tokenizer,
        source_bos_char_len,
        source_segment_size,
        max_nanochat_tokens_per_segment,
        min_nanochat_tokens_per_segment,
    )


def _accumulate_doc_stats(
    agg: dict[str, dict[str, int | float]], doc_stats: dict
) -> None:
    """Merge one worker's per-document offset-path and compute timings."""
    for path, stats in doc_stats["paths"].items():
        target = agg[path]
        target["count"] += stats["count"]
        target["total_ms"] += stats["total_ms"]
    totals = agg.setdefault(
        "_totals",
        {"worker_compute_seconds": 0.0, "map_wait_seconds": 0.0, "scatter_seconds": 0.0},
    )
    totals["worker_compute_seconds"] = float(totals["worker_compute_seconds"]) + float(
        doc_stats["compute_seconds"]
    )


def _compute_doc_boundaries_from_tokens(args):
    t_doc = time.perf_counter()
    doc_id, text_tokens, seg_char_starts, seg_char_ends = args
    if _worker_tokenizer is None:
        raise RuntimeError("token-boundary worker was used before initialization")
    tok_starts, tok_ends = compute_doc_token_boundaries_from_tokens(
        text_tokens, seg_char_starts, seg_char_ends,
        _worker_tokenizer,
        source_bos_char_len=_worker_source_bos_char_len,
    )
    try:
        validate_segment_sizes(
            tok_starts,
            tok_ends,
            _worker_source_segment_size,
            max_nanochat_tokens_per_segment=_worker_max_nanochat_tokens_per_segment,
            min_nanochat_tokens_per_segment=_worker_min_nanochat_tokens_per_segment,
        )
    except ValueError as exc:
        raise ValueError(f"doc_id={doc_id}: {exc}") from exc
    return (
        doc_id,
        tok_starts,
        tok_ends,
        {
            "paths": get_path_stats(),
            "compute_seconds": time.perf_counter() - t_doc,
        },
    )


def _boundaries_from_offsets(
    text_tokens: list[int],
    seg_char_starts: list[int],
    seg_char_ends: list[int],
    tokenizer: _BoundaryTokenizer,
    source_bos_char_len: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Compute token-boundary ranges from decoded character offsets."""
    n_tokens = len(text_tokens)
    n_segs = len(seg_char_starts)
    if n_tokens == 0 or n_segs == 0:
        return t.zeros(n_segs, dtype=t.int32), t.zeros(n_segs, dtype=t.int32)

    # Compute char offsets on the full token sequence (including BOS) to
    # preserve decoding context, then shift text-token offsets from nanochat
    # BOS coordinates into source-model BOS coordinates.
    char_offsets = token_to_char_offsets(tokenizer, text_tokens)
    nanochat_bos_len = char_offsets[1] if len(char_offsets) > 1 else 0
    delta = source_bos_char_len - nanochat_bos_len
    if delta != 0:
        char_offsets = [char_offsets[0]] + [o + delta for o in char_offsets[1:]]

    tok_starts_full = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends_full = t.tensor(char_offsets[1:], dtype=t.int64)

    # Zero-width tokens (start == end) arise from intermediate bytes of
    # multi-byte UTF-8 characters in byte-fallback tokenization.  Expand each
    # zero-width token to share the end of the next non-zero-width token so
    # that assign_tokens_to_segments sees a valid interval.
    zero_width = tok_starts_full == tok_ends_full
    if zero_width.any():
        expanded_ends = tok_ends_full.clone()
        for i in range(len(expanded_ends) - 2, -1, -1):
            if expanded_ends[i] == tok_starts_full[i]:
                expanded_ends[i] = expanded_ends[i + 1]
        tok_ends_full = expanded_ends

    # Segments only cover the source model's context window. Nanochat may
    # tokenize text beyond that range — truncate to tokens that start before
    # the last segment boundary.
    seg_range_end = seg_char_ends[-1]
    in_range = tok_starts_full < seg_range_end
    n_in_range = int(in_range.sum().item())
    tok_starts = tok_starts_full[:n_in_range]
    tok_ends = tok_ends_full[:n_in_range]

    seg_starts_t = t.tensor(seg_char_starts, dtype=t.int64)
    seg_ends_t = t.tensor(seg_char_ends, dtype=t.int64)

    assignments = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts_t, seg_ends_t)
    assignments = t.cummax(assignments, dim=0).values

    assignments = _fix_trailing_empty_last_segment(
        assignments,
        seg_char_starts,
        seg_char_ends,
    )

    # Extract per-segment token ranges from monotonic assignments.
    # Detect contiguous run boundaries instead of scanning per-segment.
    return _extract_ranges_from_monotonic(assignments, n_segs, n_in_range)


def _fix_trailing_empty_last_segment(
    assignments: t.Tensor,
    seg_char_starts: list[int],
    seg_char_ends: list[int],
) -> t.Tensor:
    """Repair the degenerate tiny-trailing-segment case.

    After ``cummax`` monotonicity enforcement it is possible for the last
    nanochat token to stay assigned to the penultimate source segment,
    leaving the final source segment empty. This happens for documents
    with e.g. ``N*segment_size + k`` source tokens (small ``k``) where the
    trailing source segment is just a few characters wide — narrow enough
    that no nanochat token naturally lands in it after tokenization
    overhead.

    When the last source segment's char width is within
    ``_TINY_TRAILING_SEG_MAX_CHARS``, force-assign the last nanochat token
    to it. Any other empty-segment pattern is deliberately left untouched
    so ``_extract_ranges_from_monotonic`` will raise a clear error — those
    patterns indicate a real anomaly upstream (wrong segment widths,
    misaligned tokenization) and should surface for investigation rather
    than being silently papered over.

    Returns the (possibly mutated) assignments tensor. Mutates in place
    when a repair is applied.
    """
    n_segs = len(seg_char_starts)
    if len(assignments) == 0 or n_segs < 2:
        return assignments
    max_assigned = int(assignments[-1].item())
    if max_assigned >= n_segs - 1:
        return assignments
    last_seg_chars = seg_char_ends[-1] - seg_char_starts[-1]
    if last_seg_chars <= _TINY_TRAILING_SEG_MAX_CHARS:
        assignments[-1] = n_segs - 1
    return assignments


def _extract_ranges_from_monotonic(
    assignments: t.Tensor, n_segs: int, n_tokens: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Extract per-segment [start, end) ranges from monotonic assignments.

    assignments must be non-decreasing (post-cummax) with exactly n_segs
    distinct contiguous runs covering segments 0..n_segs-1. Every segment
    must have at least one token assigned.

    Returns two int32 tensors of length n_segs.

    Raises ValueError if any segment has zero tokens.
    """
    if n_tokens == 0:
        raise ValueError(f"Cannot extract ranges from 0 tokens for {n_segs} segments")

    # Find contiguous run boundaries
    changes = (assignments[1:] != assignments[:-1]).nonzero(as_tuple=True)[0] + 1
    boundaries = t.cat([t.tensor([0]), changes, t.tensor([n_tokens])])
    run_starts = boundaries[:-1].to(t.int32)
    run_ends = boundaries[1:].to(t.int32)
    n_runs = len(run_starts)

    if n_runs != n_segs:
        run_seg_ids = assignments[boundaries[:-1]].tolist()
        missing = [s for s in range(n_segs) if s not in run_seg_ids]
        raise ValueError(
            f"Expected {n_segs} contiguous runs but found {n_runs}. "
            f"Segments with no tokens: {missing}. "
            f"Every extraction segment must cover at least one token."
        )

    # Scatter runs into output tensors (run_seg_ids == [0, 1, ..., n_segs-1])
    token_starts = run_starts
    token_ends = run_ends

    return token_starts, token_ends


# Absolute char-width threshold for the tiny-trailing-segment repair.
# A trailing source segment narrower than this receives the last nanochat
# token if it would otherwise be empty (e.g. N*segment_size+k Gemma tokens
# producing a final segment holding just the last 1-2 Gemma tokens worth
# of chars). Anything wider than this falls through to the extractor and
# raises as a real anomaly.
#
# Unified on the canonical ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` —
# this is the same concept as the partial-coverage absorption window on
# the consumer side (full-sequence SFT routing and target-real eval):
# tolerate up to one end-of-turn marker's worth of char residue at the
# segment boundary.
_TINY_TRAILING_SEG_MAX_CHARS = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS

# Bounds for nanochat segment size relative to source segment size.
# Tiered upper bounds for nanochat segment size relative to source segment size.
# Gemma (256K vocab) is more efficient than nanochat (32K vocab), especially
# on CJK/Korean text where nanochat falls back to byte-level encoding.
# Lower bound is a hard limit (< 0.5x always indicates a real problem).
_SEG_LO_RAISE = 0.5
_SEG_HI_WARN = 4.0
_SEG_HI_ERROR = 5.0
_SEG_HI_RAISE = 6.0


def validate_segment_sizes(
    token_starts: t.Tensor | list[int],
    token_ends: t.Tensor | list[int],
    source_segment_size: int,
    max_nanochat_tokens_per_segment: int | None = None,
    min_nanochat_tokens_per_segment: int | None = None,
) -> None:
    """Tiered validation of nanochat segment sizes vs source segment size.

    Only the last segment is exempt (commonly shorter due to partial fill at
    end of document). All other segments, including the first, are checked
    against three tiers: warn, error, raise.

    Args:
        max_nanochat_tokens_per_segment: If set, overrides the multiplier-based
            upper hard limit (default ``source_segment_size * _SEG_HI_RAISE``)
            with this absolute token count. Useful for small source segment
            sizes (e.g. 32) where the default 6x multiplier is too tight given
            fixed per-segment tokenization overhead.
        min_nanochat_tokens_per_segment: If set, overrides the multiplier-based
            lower hard bound (default ``source_segment_size * _SEG_LO_RAISE``)
            with this absolute token count. Useful for small source segment
            sizes where the ``_fix_trailing_empty_last_segment`` repair can
            legitimately produce 1-token segments. The upper bound and the
            warn/error tiers are unaffected.
    """
    n = len(token_starts)
    if n <= 1:
        return  # single segment is always the last
    lo_raise = (
        min_nanochat_tokens_per_segment
        if min_nanochat_tokens_per_segment is not None
        else int(source_segment_size * _SEG_LO_RAISE)
    )
    hi_warn = int(source_segment_size * _SEG_HI_WARN)
    hi_error = int(source_segment_size * _SEG_HI_ERROR)
    hi_raise = (
        max_nanochat_tokens_per_segment
        if max_nanochat_tokens_per_segment is not None
        else int(source_segment_size * _SEG_HI_RAISE)
    )
    for i in range(n - 1):  # all segments except the last
        seg_len = int(token_ends[i]) - int(token_starts[i])
        if seg_len == 0:
            continue
        if seg_len < lo_raise:
            raise ValueError(
                f"Segment {i}/{n}: nanochat token count {seg_len} below "
                f"minimum {lo_raise} (source_segment_size={source_segment_size})"
            )
        if seg_len > hi_raise:
            raise ValueError(
                f"Segment {i}/{n}: nanochat token count {seg_len} exceeds "
                f"hard limit {hi_raise} (source_segment_size={source_segment_size})"
            )
        if seg_len > hi_error:
            logger.error(
                f"Segment {i}/{n}: nanochat token count {seg_len} exceeds "
                f"error threshold {hi_error} (source_segment_size={source_segment_size})"
            )
        elif seg_len > hi_warn:
            logger.warning(
                f"Segment {i}/{n}: nanochat token count {seg_len} exceeds "
                f"warn threshold {hi_warn} (source_segment_size={source_segment_size})"
            )


def compute_doc_token_boundaries_from_tokens(
    text_tokens: list[int],
    seg_char_starts: list[int],
    seg_char_ends: list[int],
    tokenizer,
    source_bos_char_len: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Compute per-segment token boundaries from pre-tokenized tokens.

    Expects tokens already produced by batch encoding in the main process.
    """
    return _boundaries_from_offsets(
        text_tokens, seg_char_starts, seg_char_ends,
        tokenizer, source_bos_char_len,
    )


def precompute_shard(
    shard_path: Path,
    parquet_path: Path | None,
    tokenizer,
    bos_token: int,
    output_dir: Path,
    shard_stem: str,
    source_bos_char_len: int,
    source_segment_size: int,
    dry_run: bool = False,
    zstd_level: int | None = None,
    max_nanochat_tokens_per_segment: int | None = None,
    min_nanochat_tokens_per_segment: int | None = None,
    timing: ShardTiming | None = None,
    worker_pool: ProcessPoolExecutor | None = None,
    load_keys: set[str] | None = None,
    *,
    preloaded_data: Mapping[str, Any] | None = None,
) -> bool:
    """Precompute token boundaries for a single shard and write as overlay.

    Reads the base shard (immutable), computes boundaries, and writes to
    ``output_dir/shard_stem.safetensors.zst``. The base shard is never modified.

    Args:
        output_dir: Overlay directory (e.g., tok/<hash>/seg_boundaries/).
        shard_stem: Stem name for the output file.
        worker_pool: ``None`` ⇒ compute sequentially in this process (1 worker);
            otherwise dispatch per-doc work to this run-level pool — its
            ``max_workers`` IS the parallelism, so there is no separate worker
            count here.
        zstd_level: Compression level (1-22). Defaults to overlay_io.ZSTD_LEVEL.

    Returns True if the overlay shard was written (or would be on dry_run).
    """
    t0 = time.monotonic()
    data = (
        load_features(shard_path, keys=load_keys)
        if preloaded_data is None
        else preloaded_data
    )
    t_load = time.monotonic() - t0

    segment_mode = segment_mode_from_metadata(data, source=shard_path)
    if segment_mode is not SegmentMode.fixed_tokens:
        raise ValueError(
            f"token-boundary precompute supports fixed_tokens shards only (segment_mode={segment_mode.value!r})"
        )

    n_segments = int(data["n_segments"])
    if timing is not None:
        if preloaded_data is None:
            timing.load_features_seconds = t_load
            timing.load_bytes = shard_path.stat().st_size
        timing.n_segments = n_segments
    if n_segments == 0:
        logger.info(f"  {shard_path.name}: empty shard, skipping")
        return False

    if parquet_path is None:
        raise MissingParquetInputError(
            f"Pending nonempty token-overlay shard {shard_stem} has no "
            "source parquet input."
        )

    empties = empty_segment_indices(data["offsets"])
    if empties:
        i = empties[0]
        raise EmptyExtractionSegmentError(
            f"precompute_token_boundaries: base shard {shard_path.name} segment "
            f"{i} has zero features; base shards must be non-empty (issue #522)."
        )

    doc_ids = data["doc_ids"]
    seg_indices = data["segment_indices"]
    seg_char_starts = data["segment_char_starts"]
    seg_char_ends = data["segment_char_ends"]

    # Load parquet to get raw text (row-group order = extraction doc order)
    t1 = time.monotonic()
    all_texts = read_parquet_texts(parquet_path)
    t_parquet = time.monotonic() - t1
    if timing is not None:
        timing.parquet_load_seconds = t_parquet
        timing.parquet_bytes = parquet_path.stat().st_size

    metadata_n_docs = int(data["n_docs"])
    doc_ids_list = doc_ids.tolist()
    observed_doc_min = min(doc_ids_list)
    observed_doc_max = max(doc_ids_list)
    alignment_context = (
        f"feature_path={shard_path}, parquet_path={parquet_path}, "
        f"parquet_rows={len(all_texts)}, metadata_n_docs={metadata_n_docs}, "
        f"observed_doc_id_range=[{observed_doc_min}, {observed_doc_max}]"
    )
    if len(all_texts) != metadata_n_docs:
        raise TokenBoundaryAlignmentError(
            "Token-boundary feature/parquet row-count mismatch: " + alignment_context
        )
    if observed_doc_min < 0 or observed_doc_max >= metadata_n_docs:
        raise TokenBoundaryAlignmentError(
            "Token-boundary feature doc_id is outside [0, n_docs): "
            + alignment_context
        )

    # Batch-encode all documents in the main process before distributing to
    # workers. This amortizes Python->Rust FFI overhead and lets tiktoken use
    # internal threading (RustBPE encode_ordinary_batch).
    t_enc = time.monotonic()
    all_token_lists = tokenizer.encode(all_texts, prepend=bos_token)
    t_batch_enc = time.monotonic() - t_enc
    if timing is not None:
        timing.batch_encode_seconds = t_batch_enc
    if len(all_token_lists) != len(all_texts):
        raise TokenBoundaryAlignmentError(
            "Token-boundary tokenizer batch-length mismatch: "
            + alignment_context
            + f", tokenizer_outputs={len(all_token_lists)}"
        )
    logger.info(f"    Batch-encoded {len(all_texts)} docs in {t_batch_enc:.1f}s")

    # Group segments by doc_id
    seg_indices_list = seg_indices.tolist()
    char_starts_list = seg_char_starts.tolist()
    char_ends_list = seg_char_ends.tolist()

    # Group segments by doc_id for efficient processing
    doc_segments: dict[int, list[int]] = defaultdict(list)
    for seg_i in range(n_segments):
        doc_segments[doc_ids_list[seg_i]].append(seg_i)

    # Sort each doc's segments by segment_idx
    for seg_indices_for_doc in doc_segments.values():
        seg_indices_for_doc.sort(key=lambda i: seg_indices_list[i])

    # Build work items: (doc_id, token_ids, seg_char_starts, seg_char_ends)
    work_items = []
    for doc_id, seg_indices_for_doc in doc_segments.items():
        doc_char_starts = [char_starts_list[i] for i in seg_indices_for_doc]
        doc_char_ends = [char_ends_list[i] for i in seg_indices_for_doc]
        work_items.append(
            (doc_id, all_token_lists[doc_id], doc_char_starts, doc_char_ends)
        )

    n_docs = len(work_items)

    # Compute token boundaries (parallel or sequential)
    t2 = time.monotonic()
    token_starts = t.zeros(n_segments, dtype=t.int32)
    token_ends = t.zeros(n_segments, dtype=t.int32)
    agg_paths: dict[str, dict[str, int | float]] = {
        path: {"count": 0, "total_ms": 0.0}
        for path in ("fast", "slow", "small_n")
    }
    agg_paths["_totals"] = {
        "worker_compute_seconds": 0.0,
        "map_wait_seconds": 0.0,
        "scatter_seconds": 0.0,
    }

    if worker_pool is None:
        # Sequential (1 worker, in-process): init the module-level worker globals here.
        _init_worker(
            tokenizer,
            source_bos_char_len,
            source_segment_size,
            max_nanochat_tokens_per_segment,
            min_nanochat_tokens_per_segment,
        )
        # Drain any counts a long-lived in-process caller accumulated before
        # this shard: the per-doc drain below must attribute only this shard's
        # offset calls, matching fresh spawn workers (review finding, #1269).
        get_path_stats()
        for doc_count, item in enumerate(work_items):
            if doc_count % max(1, n_docs // 10) == 0:
                elapsed = time.monotonic() - t2
                docs_per_sec = doc_count / elapsed if elapsed > 0 else 0
                remaining = (n_docs - doc_count) / docs_per_sec if docs_per_sec > 0 else 0
                logger.info(
                    f"    doc {doc_count}/{n_docs} ({doc_count*100//n_docs}%) "
                    f"| {docs_per_sec:.1f} docs/s | ETA {remaining:.0f}s"
                )
            doc_id, doc_tok_starts, doc_tok_ends, doc_stats = (
                _compute_doc_boundaries_from_tokens(item)
            )
            t_scatter = time.perf_counter()
            for j, seg_i in enumerate(doc_segments[doc_id]):
                token_starts[seg_i] = doc_tok_starts[j]
                token_ends[seg_i] = doc_tok_ends[j]
            _accumulate_doc_stats(agg_paths, doc_stats)
            agg_paths["_totals"]["scatter_seconds"] = float(
                agg_paths["_totals"]["scatter_seconds"]
            ) + (time.perf_counter() - t_scatter)
    else:
        # Parallel: dispatch each doc to the caller's run-level pool (created once in run() and
        # reused across every shard; its max_workers is the parallelism). The per-doc computation
        # is a pure function of its inputs and results are scattered by seg index, so the output
        # is byte-identical to the sequential path.
        logger.info(f"    Dispatching {n_docs} docs to the worker pool")

        t_map_wait = 0.0
        t_scatter = 0.0
        results_iter = worker_pool.map(
            _compute_doc_boundaries_from_tokens,
            work_items,
            chunksize=_DOC_BOUNDARY_TASK_CHUNK_SIZE,
        )
        while True:
            t_wait0 = time.perf_counter()
            try:
                doc_id, doc_tok_starts, doc_tok_ends, doc_stats = next(results_iter)
            except StopIteration:
                t_map_wait += time.perf_counter() - t_wait0
                break
            t_map_wait += time.perf_counter() - t_wait0
            t_sc0 = time.perf_counter()
            for j, seg_i in enumerate(doc_segments[doc_id]):
                token_starts[seg_i] = doc_tok_starts[j]
                token_ends[seg_i] = doc_tok_ends[j]
            _accumulate_doc_stats(agg_paths, doc_stats)
            t_scatter += time.perf_counter() - t_sc0

        agg_paths["_totals"]["map_wait_seconds"] = t_map_wait
        agg_paths["_totals"]["scatter_seconds"] = t_scatter

    t_compute = time.monotonic() - t2
    if timing is not None:
        timing.tokenize_seconds = t_compute
        timing.offset_paths = {
            path: agg_paths[path] for path in ("fast", "slow", "small_n")
        }
        timing.worker_compute_seconds = float(
            agg_paths["_totals"]["worker_compute_seconds"]
        )
        timing.map_wait_seconds = float(agg_paths["_totals"]["map_wait_seconds"])
        timing.scatter_seconds = float(agg_paths["_totals"]["scatter_seconds"])

    if dry_run:
        logger.info(
            f"  [DRY-RUN] {shard_stem}: {n_segments} segments, "
            f"load={t_load:.1f}s, parquet={t_parquet:.1f}s, compute={t_compute:.1f}s"
        )
        return True

    t3 = time.monotonic()
    out_path = overlay_shard_path(output_dir, shard_stem)
    save_overlay_shard(
        {
            TensorKey.SEGMENT_TOKEN_STARTS: token_starts,
            TensorKey.SEGMENT_TOKEN_ENDS: token_ends,
        },
        out_path,
        zstd_level=zstd_level,
    )
    t_save = time.monotonic() - t3
    if timing is not None:
        timing.overlay_save_seconds = t_save
        timing.overlay_bytes = out_path.stat().st_size
    t_total = time.monotonic() - t0

    logger.info(
        f"  {shard_stem}: {n_segments} segments in {t_total:.0f}s "
        f"(load={t_load:.0f}s/{shard_path.stat().st_size >> 20}MB, "
        f"parquet={t_parquet:.0f}s, compute={t_compute:.0f}s, "
        f"save={t_save:.0f}s/{out_path.stat().st_size >> 10}KB)"
    )
    return True


def _detect_source_bos_char_len(
    shard_path: Path,
    parquet_path: Path,
    *,
    load_keys: set[str] | None = None,
) -> int:
    """Load a local raw shard and delegate to the shared BOS detector."""
    return _detect_source_bos_char_len_from_data(
        load_features(shard_path, keys=load_keys),
        parquet_path,
    )


def _run_local(cfg: DictConfig) -> int:
    """Main entry point for precompute pipeline, usable from Hydra or tests."""
    feature_dir = resolve_feature_dir(cfg)
    if not feature_dir.is_dir():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1

    # Cross-stage segment_size check (Wave 3D task 4).
    verify_segment_size(feature_dir, expected_segment_size=int(cfg.segment_size))

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.warning(
            f"No feature_statistics.safetensors in {feature_dir} — "
            "tokenizer checksum will not be saved to stats metadata"
        )

    # Load tokenizer and compute checksum
    logger.info("Loading nanochat tokenizer...")
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    checksum = compute_tokenizer_checksum(tokenizer)
    logger.info(f"Tokenizer checksum: {checksum[:16]}...")

    dry_run = bool(cfg.get("dry_run", False))
    resume = bool(cfg.get("resume", False))
    force = bool(cfg.get("force", False))
    profile = bool(cfg.get("profile", False))
    # The chrome trace runs the first shard SEQUENTIALLY under torch.profiler
    # (with_stack=True): on a real production shard its event buffer reaches
    # tens of GB and the process gets OOM-killed, and the sequential first
    # shard distorts worker-sweep benchmarks. profile_trace=false keeps the
    # JSON sidecar while skipping the trace entirely (issue #1269).
    profile_trace = bool(cfg.get("profile_trace", True))
    profile_summary_output = cfg.get("profile_summary_output", None)
    # #529 optimizations are unconditional (each proven byte-identical to the reference path):
    # always subset-load only the segmentation fields each load path consumes.
    shard_load_keys = _PRECOMPUTE_SHARD_LOAD_KEYS
    bos_load_keys = _DETECT_BOS_LOAD_KEYS
    n_workers = int(cfg.get("workers", 0))
    zstd_level = cfg.get("zstd_level", None)
    shard_start = int(cfg.get("shard_start", 0))
    shard_end = int(cfg.get("shard_end", -1))
    max_nanochat_tokens_raw = cfg.get("max_nanochat_tokens_per_segment", None)
    max_nanochat_tokens_per_segment = (
        None if max_nanochat_tokens_raw is None else int(max_nanochat_tokens_raw)
    )
    if max_nanochat_tokens_per_segment is not None:
        logger.info(
            f"Override hard limit: max_nanochat_tokens_per_segment="
            f"{max_nanochat_tokens_per_segment}"
        )
    min_nanochat_tokens_raw = cfg.get("min_nanochat_tokens_per_segment", None)
    min_nanochat_tokens_per_segment = (
        None if min_nanochat_tokens_raw is None else int(min_nanochat_tokens_raw)
    )
    if min_nanochat_tokens_per_segment is not None:
        logger.info(
            f"Override lower bound: min_nanochat_tokens_per_segment="
            f"{min_nanochat_tokens_per_segment}"
        )
    if n_workers == 0:
        n_workers = (os.cpu_count() or 2) // 2
    logger.info(f"Workers: {n_workers}")

    if force and resume:
        logger.warning("Both force=true and resume=true set; force takes precedence")
        resume = False

    profiler = (
        TokenBoundaryProfile(
            config={
                "feature_dir": str(feature_dir),
                "workers": n_workers,
                "resume": resume,
                "force": force,
                "zstd_level": zstd_level,
            }
        )
        if profile
        else None
    )

    # Resolve overlay directory
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    logger.info(f"Overlay directory: {output}")

    # Find feature shards before discovery so expected coverage is derived only
    # from the current base shards and never persisted in the manifest.
    all_shard_paths = glob_features(feature_dir)
    logger.info(f"Found {len(all_shard_paths)} feature shards in {feature_dir}")
    if not all_shard_paths:
        logger.warning("No shards found, nothing to do.")
        return 0
    expected_stems = expected_overlay_shard_stems(
        all_shard_paths, OverlayKind.SEG_BOUNDARY
    )

    # Authenticate an existing v2 manifest against the current base bytes before
    # any output, statistics, or cache write. Legacy v1 remains accepted by the
    # ordinary discovery contract (without base authentication).
    existing_overlay = None
    if not force:
        existing_overlay = discover_overlay(
            feature_dir,
            OverlayKind.SEG_BOUNDARY,
            checksum,
            base_shard_paths=all_shard_paths,
            cache_dir=None,
        )
    if existing_overlay is not None and not force:
        logger.info(f"Overlay already exists at {existing_overlay}")
        missing_stems = [
            stem
            for stem in expected_stems
            if not overlay_shard_path(existing_overlay, stem).is_file()
        ]
        if not missing_stems and not resume:
            if not dry_run and stats_path is not None:
                save_tokenizer_checksum(stats_path, checksum)
                logger.info(f"Saved tokenizer checksum to {stats_path}")
            return 0
        if not resume:
            raise OverlayIncompleteError(
                f"Token-boundary overlay at {existing_overlay} is incomplete: "
                f"missing {len(missing_stems)} expected shard(s). "
                "Use resume=true to fill them."
            )

    # Find feature shards and parquet files
    shard_paths = all_shard_paths

    # Apply shard range filter
    if shard_end < 0:
        shard_end = len(shard_paths)
    shard_paths = shard_paths[shard_start:shard_end]
    logger.info(f"Processing shard range [{shard_start}, {shard_end}) → {len(shard_paths)} shards")
    if not shard_paths:
        logger.info("Selected shard range is empty, nothing to do.")
        return 0
    selected_expected_stems = set(
        expected_overlay_shard_stems(shard_paths, OverlayKind.SEG_BOUNDARY)
    )

    data_dir = cfg.get("data_dir", None)
    parquet_paths = list_parquet_files(data_dir)
    parquet_by_stem = {Path(p).stem: Path(p) for p in parquet_paths}
    logger.info(f"Found {len(parquet_paths)} parquet files")

    # Resolve pending nonempty work and validate resume-present payloads before
    # any manifest, cache, payload, statistics, profiler, or worker-pool write.
    resume_present_stems: set[str] = set()
    pending_nonempty: list[Path] = []
    for shard_path in shard_paths:
        stem = shard_stem_from_path(shard_path)
        if stem not in selected_expected_stems:
            continue
        payload_path = overlay_shard_path(output, stem)
        if resume and not force and payload_path.is_file():
            n_segments = int(
                read_shard_metadata(shard_path)[MetadataKey.N_SEGMENTS]
            )
            validate_token_boundary_overlay_shard(
                payload_path, expected_n_segments=n_segments
            )
            resume_present_stems.add(stem)
        else:
            pending_nonempty.append(shard_path)
    _raise_for_missing_pending_parquets(pending_nonempty, parquet_by_stem)

    # Preserve the reference schema-v2 write and base-hash cache behavior for
    # fresh and explicit-force production. Existing valid v1/v2 manifests are
    # never restamped by resume.
    if not dry_run and (existing_overlay is None or force):
        t_pre = time.monotonic()
        base_hash_result = _compute_base_hash(
            all_shard_paths, cache_dir=feature_dir
        )
        write_manifest(
            output,
            checksum,
            OverlayKind.SEG_BOUNDARY,
            base_shard_paths=all_shard_paths,
            precomputed_base_hash=base_hash_result.hexdigest,
        )
        pre_pass_seconds = time.monotonic() - t_pre
        if profiler is not None:
            profiler.manifest_pre_pass = {
                "seconds": pre_pass_seconds,
                "bytes_read": base_hash_result.bytes_read,
                "cache_hit": base_hash_result.cache_hit,
                "n_shards_hashed": len(all_shard_paths),
            }
        logger.info(
            f"Wrote manifest to {output} "
            f"(base-hash {'cache hit' if base_hash_result.cache_hit else 'computed'}, "
            f"{base_hash_result.bytes_read >> 20} MB read)"
        )

    # Detect source-model geometry only when a shard actually needs work. The
    # preflight above proves this reference shard has a matching parquet, so no
    # model-specific BOS guess is necessary. On a fully completed resume these
    # values are never consumed and no worker pool needs to be started.
    source_bos_char_len: int | None = None
    source_segment_size: int | None = None
    persistent_pool = None
    saved_thread_env: dict[str, str | None] | None = None
    if pending_nonempty:
        reference_shard = pending_nonempty[0]
        reference_stem = shard_stem_from_path(reference_shard)
        reference_parquet = parquet_by_stem[reference_stem]
        source_bos_char_len = _detect_source_bos_char_len(
            reference_shard, reference_parquet, load_keys=bos_load_keys
        )
        nanochat_bos_char_len = len(tokenizer.decode([bos_token]))
        logger.info(
            f"BOS char lengths: source={source_bos_char_len}, "
            f"nanochat={nanochat_bos_char_len}, delta={source_bos_char_len - nanochat_bos_char_len}"
        )
        # Read the first selected shard's segment size from its safetensors
        # header only. Feature extraction has always recorded this metadata;
        # absence is corrupt input, not a reason to silently assume 128.
        first_shard_meta = read_shard_metadata(shard_paths[0])
        source_segment_size = int(first_shard_meta[MetadataKey.SEGMENT_SIZE])
        assert source_segment_size > 0, (
            f"non-positive segment_size {source_segment_size} from {shard_paths[0]}"
        )
        logger.info(f"Source segment size: {source_segment_size}")

        # Pool workers across pending shards. The worker init args are detected
        # once and constant across shards, so one spawn-context pool serves the
        # whole run instead of re-importing modules per shard.
        if n_workers > 1:
            saved_thread_env = _apply_worker_thread_env()
            t_pool = time.monotonic()
            try:
                persistent_pool = ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_init_worker_process,
                    initargs=(
                        tokenizer,
                        source_bos_char_len,
                        source_segment_size,
                        max_nanochat_tokens_per_segment,
                        min_nanochat_tokens_per_segment,
                    ),
                )
            except BaseException:
                # The shard-loop finally below never runs if construction
                # itself raises; do not leak the capped env to the caller.
                _restore_worker_thread_env(saved_thread_env)
                raise
            if profiler is not None:
                profiler.worker_pool = {
                    "spawn_count": 1,
                    "total_init_seconds": time.monotonic() - t_pool,
                    "thread_caps": {
                        "torch_num_threads": 1,
                        "torch_num_interop_threads": 1,
                        **_WORKER_THREAD_ENV_VARS,
                    },
                }
            logger.info(f"Pooling {n_workers} workers across pending shards")

    n_processed = 0
    n_skipped = 0
    n_total = len(shard_paths)
    t_start = time.monotonic()

    # try/finally so the spawn-context pool is shut down and the profile sidecar is flushed
    # even when a shard raises mid-loop (e.g. validate_segment_sizes ValueError) — otherwise
    # workers leak (when run() is reused in-process) and all collected timings are lost.
    try:
        for i, shard_path in enumerate(shard_paths):
            stem = shard_stem_from_path(shard_path)

            # Resume: skip if overlay shard already exists
            if stem in resume_present_stems:
                overlay_path = overlay_shard_path(output, stem)
                n_segments = int(
                    read_shard_metadata(shard_path)[MetadataKey.N_SEGMENTS]
                )
                validate_token_boundary_overlay_shard(
                    overlay_path, expected_n_segments=n_segments
                )
                n_skipped += 1
                if n_skipped <= 5 or n_skipped % 50 == 0:
                    logger.info(f"  [{i+1}/{n_total}] Skipped {stem} (overlay exists)")
                continue

            parquet_path = parquet_by_stem.get(stem)
            if parquet_path is None:
                if stem in selected_expected_stems:
                    raise MissingParquetInputError(
                        f"Pending nonempty token-overlay shard {stem} has no "
                        "source parquet input after preflight."
                    )
                logger.warning(f"  [{i+1}/{n_total}] No parquet file for shard {stem}, skipping")
                continue

            assert source_bos_char_len is not None
            assert source_segment_size is not None

            timing = ShardTiming(stem=stem, n_segments=0) if profiler is not None else None

            # Profile the first shard if requested
            should_profile = profile and profile_trace and n_processed == 0
            if should_profile:
                trace_dir = Path("out/traces")
                trace_dir.mkdir(parents=True, exist_ok=True)
                # `or` (not get-default): the YAML ships `profile_output: null`, so cfg.get
                # returns None when present-as-null → Path(None) would crash on `profile=true`.
                trace_path = Path(
                    cfg.get("profile_output") or str(trace_dir / "tok_boundary_trace.json")
                )
                logger.info(f"  Profiling first shard → {trace_path}")
                with torch_profile(
                    activities=[ProfilerActivity.CPU],
                    with_stack=True,
                    record_shapes=True,
                ) as prof:
                    precompute_shard(
                        shard_path, parquet_path, tokenizer, bos_token,
                        output, stem,
                        dry_run=dry_run, zstd_level=zstd_level,
                        source_bos_char_len=source_bos_char_len,
                        source_segment_size=source_segment_size,
                        max_nanochat_tokens_per_segment=max_nanochat_tokens_per_segment,
                        min_nanochat_tokens_per_segment=min_nanochat_tokens_per_segment,
                        timing=timing,
                        worker_pool=None,  # trace the first shard sequentially
                        load_keys=shard_load_keys,
                    )
                prof.export_chrome_trace(str(trace_path))
                logger.info(f"  Chrome trace saved to {trace_path}")
                logger.info("  Open in chrome://tracing or https://ui.perfetto.dev")
            else:
                precompute_shard(
                    shard_path, parquet_path, tokenizer, bos_token,
                    output, stem,
                    dry_run=dry_run, zstd_level=zstd_level,
                    source_bos_char_len=source_bos_char_len,
                    source_segment_size=source_segment_size,
                    max_nanochat_tokens_per_segment=max_nanochat_tokens_per_segment,
                    min_nanochat_tokens_per_segment=min_nanochat_tokens_per_segment,
                    timing=timing,
                    worker_pool=persistent_pool,
                    load_keys=shard_load_keys,
                )
            n_processed += 1
            if profiler is not None and timing is not None:
                profiler.add_shard(timing)

            elapsed = time.monotonic() - t_start
            remaining = n_total - i - 1
            if n_processed > 0:
                avg_per_shard = elapsed / (n_processed + n_skipped)
                eta_s = remaining * avg_per_shard
                eta_str = f"{eta_s/60:.0f}m" if eta_s > 60 else f"{eta_s:.0f}s"
            else:
                eta_str = "?"
            logger.info(
                f"  [{i+1}/{n_total}] {stem} done "
                f"(processed={n_processed}, skipped={n_skipped}, "
                f"elapsed={elapsed/60:.1f}m, ETA={eta_str})"
            )
    finally:
        if persistent_pool is not None:
            persistent_pool.shutdown()
        if saved_thread_env is not None:
            # Only after shutdown(): workers spawn lazily, so the env caps must
            # cover the pool's entire lifetime (review finding, issue #1269).
            _restore_worker_thread_env(saved_thread_env)
        if profiler is not None:
            # spawn_count + total_init_seconds were set at pool creation when a pool was built
            # (n_workers > 1); a sequential run (no pool) leaves the default spawn_count=0.
            profiler.totals = {
                "wall_seconds": time.monotonic() - t_start,
                "n_processed": n_processed,
                "n_skipped": n_skipped,
            }
            sidecar_path = Path(
                profile_summary_output or "out/traces/tok_boundary_profile.json"
            )
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.write(sidecar_path)
            logger.info(f"Profile sidecar → {sidecar_path}")

    elapsed_total = time.monotonic() - t_start
    logger.info(
        f"Done: {n_processed} processed, {n_skipped} skipped in {elapsed_total/60:.1f}m "
        f"(overlay={output}, checksum={checksum[:16]}...)"
    )

    if not dry_run:
        for shard_path in shard_paths:
            stem = shard_stem_from_path(shard_path)
            if stem not in selected_expected_stems:
                continue
            payload_path = overlay_shard_path(output, stem)
            if not payload_path.is_file():
                raise OverlayIncompleteError(
                    f"Token-boundary overlay at {output} is incomplete for "
                    f"shard range [{shard_start}, {shard_end}): missing {stem}."
                )
            n_segments = int(
                read_shard_metadata(shard_path)[MetadataKey.N_SEGMENTS]
            )
            validate_token_boundary_overlay_shard(
                payload_path, expected_n_segments=n_segments
            )

    if not dry_run and stats_path is not None:
        save_tokenizer_checksum(stats_path, checksum)
        logger.info(f"Saved tokenizer checksum to {stats_path}")

    return 0


def _remote_enabled(cfg: DictConfig) -> bool:
    remote = cfg.get("remote", None)
    if remote is None:
        return False
    if not hasattr(remote, "get"):
        raise ValueError("remote must be a configuration mapping")
    enabled = remote.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"remote.enabled must be a bool, got {enabled!r}")
    return enabled


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve()
    resolved_right = right.resolve()
    return (
        resolved_left == resolved_right
        or resolved_left.is_relative_to(resolved_right)
        or resolved_right.is_relative_to(resolved_left)
    )


def _resolve_remote_transport_config(
    cfg: DictConfig,
    *,
    feature_dir: Path,
    output: Path,
    data_dir: Path,
) -> TokenBoundaryRemoteConfig:
    remote = cfg.remote
    if remote.staging_dir is None:
        raise ValueError("remote.staging_dir is required when remote.enabled=true")
    staging_dir = Path(str(remote.staging_dir)).absolute()
    if any(
        _paths_overlap(staging_dir, protected)
        for protected in (feature_dir, output, data_dir)
    ):
        raise ValueError(
            "remote.staging_dir must be disjoint from feature_dir, the overlay "
            "output, and data_dir"
        )
    local_features_root = Path(str(remote.local_features_root))
    remote_features_root = NormalizedRemotePrefix.parse(
        str(remote.remote_features_root)
    )
    prefix = resolve_remote_feature_prefix(
        feature_dir,
        local_features_root=local_features_root,
        remote_features_root=remote_features_root,
        surface="feature_dir",
    )
    expected_raw = remote.expected_shard_count
    if expected_raw is None:
        expected_count = None
    elif (
        isinstance(expected_raw, bool)
        or not isinstance(expected_raw, int)
        or expected_raw <= 0
    ):
        raise ValueError(
            "remote.expected_shard_count must be null or a positive integer, "
            f"got {expected_raw!r}"
        )
    else:
        expected_count = int(expected_raw)
    return TokenBoundaryRemoteConfig(
        bucket_id=HfBucketId.parse(str(remote.bucket_id)),
        remote_prefix=prefix,
        staging_dir=staging_dir,
        expected_shard_count=expected_count,
        retry_timeout_seconds=remote.retry_timeout_seconds,
        retry_interval_seconds=remote.retry_interval_seconds,
    )


def _compact_shard_metadata(
    *,
    index: int,
    shard_path: Path,
    data: Mapping[str, Any],
    expected_segment_size: int,
) -> tuple[str, int, int]:
    expected_name = f"shard_{index:05d}_features.safetensors.zst"
    if shard_path.name != expected_name:
        raise ValueError(
            "compact token-boundary scheduler returned a noncanonical shard: "
            f"index={index}, expected={expected_name}, got={shard_path.name}"
        )
    n_segments = int(data["n_segments"])
    segment_size = int(data["segment_size"])
    if segment_size != expected_segment_size:
        raise SegmentSizeMismatch(
            f"segment_size mismatch for compact source shard {shard_path}: "
            f"cfg/path={expected_segment_size}, compact={segment_size}"
        )
    return shard_stem_from_path(shard_path), n_segments, segment_size


def _remote_overlay_preflight(
    *,
    output: Path,
    checksum: str,
    selected_stems: set[str],
    selected_complete_family: bool,
    parquet_by_stem: dict[str, Path],
    dry_run: bool,
    resume: bool,
    force: bool,
) -> tuple[bool, set[str]]:
    """Classify pending work without trusting unauthenticated payload residue."""
    manifest = read_manifest(output)
    manifest_checksum = manifest.get("checksum", "") if manifest is not None else ""
    has_matching_manifest = bool(
        isinstance(manifest_checksum, str)
        and manifest_checksum
        and manifest_checksum.startswith(checksum)
    )
    if not dry_run and not has_matching_manifest and not selected_complete_family:
        unselected_payloads = sorted(
            shard_stem_from_path(path)
            for path in glob_overlay_shards(output)
            if shard_stem_from_path(path) not in selected_stems
        )
        if unselected_payloads:
            raise OverlayIncompleteError(
                "remote fresh partial-range run found unauthenticated unselected "
                "canonical payloads; use the complete shard range or remove the "
                f"orphaned payloads first: {unselected_payloads[:5]}"
            )
    if dry_run or force or not has_matching_manifest:
        tentative_work_stems = set(selected_stems)
    elif resume:
        tentative_work_stems = {
            stem
            for stem in selected_stems
            if not overlay_shard_path(output, stem).is_file()
        }
    else:
        tentative_work_stems = set()
    missing_parquets = sorted(tentative_work_stems - set(parquet_by_stem))
    if missing_parquets:
        raise MissingParquetInputError(
            f"{len(missing_parquets)} tentatively pending remote token-overlay "
            "shard(s) lack local parquet inputs; representative "
            f"stems={missing_parquets[:5]}. No semantic output was written."
        )
    return has_matching_manifest, tentative_work_stems


def _run_remote(cfg: DictConfig) -> int:
    wall_start = time.monotonic()
    feature_dir = resolve_feature_dir(cfg)
    if not feature_dir.is_dir():
        logger.error(f"Logical feature directory not found: {feature_dir}")
        return 1
    configured_segment_size = int(cfg.segment_size)
    parsed_feature_dir = parse_feature_dir(feature_dir)
    if parsed_feature_dir.segment_size != configured_segment_size:
        raise SegmentSizeMismatch(
            "remote token-boundary segment_size mismatch before transport: "
            f"cfg={configured_segment_size}, path={parsed_feature_dir.segment_size}"
        )

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.warning(
            f"No feature_statistics.safetensors in {feature_dir} — "
            "tokenizer checksum will not be saved to stats metadata"
        )

    logger.info("Loading nanochat tokenizer...")
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    checksum = compute_tokenizer_checksum(tokenizer)
    logger.info(f"Tokenizer checksum: {checksum[:16]}...")
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)

    raw_data_dir = cfg.get("data_dir", None)
    data_dir = Path(DATA_DIR if raw_data_dir is None else str(raw_data_dir))
    remote_config = _resolve_remote_transport_config(
        cfg,
        feature_dir=feature_dir,
        output=output,
        data_dir=data_dir,
    )
    transport = AuthenticatedTokenBoundarySource.snapshot(remote_config)

    dry_run = bool(cfg.get("dry_run", False))
    resume = bool(cfg.get("resume", False))
    force = bool(cfg.get("force", False))
    if force and resume:
        logger.warning("Both force=true and resume=true set; force takes precedence")
        resume = False
    profile = bool(cfg.get("profile", False))
    profile_trace = bool(cfg.get("profile_trace", True))
    profile_summary_output = cfg.get("profile_summary_output", None)
    n_workers = int(cfg.get("workers", 0))
    if n_workers == 0:
        n_workers = (os.cpu_count() or 2) // 2
    zstd_level = cfg.get("zstd_level", None)
    shard_start = int(cfg.get("shard_start", 0))
    shard_end = int(cfg.get("shard_end", -1))
    max_tokens_raw = cfg.get("max_nanochat_tokens_per_segment", None)
    max_tokens = None if max_tokens_raw is None else int(max_tokens_raw)
    min_tokens_raw = cfg.get("min_nanochat_tokens_per_segment", None)
    min_tokens = None if min_tokens_raw is None else int(min_tokens_raw)
    logger.info(f"Workers: {n_workers}")

    profiler = (
        TokenBoundaryProfile(
            config={
                "feature_dir": str(feature_dir),
                "workers": n_workers,
                "resume": resume,
                "force": force,
                "zstd_level": zstd_level,
                "remote": True,
                "remote_prefix": str(remote_config.remote_prefix),
                "compact_prefix": str(transport.compact_prefix),
            }
        )
        if profile
        else None
    )

    inventory = transport.inventory
    if shard_end < 0:
        shard_end = len(inventory)
    selected_inventory = inventory[shard_start:shard_end]
    if force and selected_inventory != inventory:
        raise ValueError(
            "remote force=true requires the complete sealed shard range; "
            f"selected [{shard_start}, {shard_end}) covers "
            f"{len(selected_inventory)} of {len(inventory)} shards"
        )
    selected_indices = {entry.index for entry in selected_inventory}
    logger.info(
        f"Processing remote shard range [{shard_start}, {shard_end}) → "
        f"{len(selected_inventory)} selected of {len(inventory)} total shards"
    )

    parquet_paths = list_parquet_files(data_dir)
    parquet_by_stem = {Path(path).stem: Path(path) for path in parquet_paths}
    logger.info(f"Found {len(parquet_paths)} local parquet files")

    selected_stems = {
        shard_stem_from_path(Path(entry.info.path)) for entry in selected_inventory
    }
    has_matching_manifest, tentative_work_stems = _remote_overlay_preflight(
        output=output,
        checksum=checksum,
        selected_stems=selected_stems,
        selected_complete_family=selected_inventory == inventory,
        parquet_by_stem=parquet_by_stem,
        dry_run=dry_run,
        resume=resume,
        force=force,
    )

    lock_path = feature_dir / f".token-boundary-remote-{checksum[:16]}.lock"
    candidate_parent = output.parent
    candidate_parent.mkdir(parents=True, exist_ok=True)
    candidate_dir = Path(
        tempfile.mkdtemp(prefix=".seg-boundary-candidates.", dir=candidate_parent)
    )
    candidate_paths: dict[str, Path] = {}
    n_segments_by_stem: dict[str, int] = {}
    expected_stems: set[str] = set()
    persistent_pool: ProcessPoolExecutor | None = None
    saved_thread_env: dict[str, str | None] | None = None
    source_bos_char_len: int | None = None
    n_processed = 0
    n_skipped = 0
    run_status = "failed"
    transport_profile: dict[str, object] = {}

    try:
        with remote_output_lock(lock_path), transport:
            # The first pass fails before staging/download. Repeat under the
            # output lock so a run that published during preflight cannot make
            # this invocation classify stale manifest/payload state.
            has_matching_manifest, tentative_work_stems = _remote_overlay_preflight(
                output=output,
                checksum=checksum,
                selected_stems=selected_stems,
                selected_complete_family=selected_inventory == inventory,
                parquet_by_stem=parquet_by_stem,
                dry_run=dry_run,
                resume=resume,
                force=force,
            )
            try:
                for position, prepared in enumerate(transport.stream()):
                    compact_load_started = time.monotonic()
                    compact_data = load_token_boundary_source_shard_stable(
                        prepared.local_path,
                        expected_identity=prepared.file_identity,
                    )
                    compact_load_seconds = time.monotonic() - compact_load_started
                    stem, n_segments, source_segment_size = _compact_shard_metadata(
                        index=prepared.index,
                        shard_path=prepared.local_path,
                        data=compact_data,
                        expected_segment_size=configured_segment_size,
                    )
                    n_segments_by_stem[stem] = n_segments
                    if n_segments > 0:
                        expected_stems.add(stem)

                    if prepared.index not in selected_indices or n_segments == 0:
                        del compact_data
                        continue
                    canonical_path = overlay_shard_path(output, stem)
                    should_compute = stem in tentative_work_stems
                    if not should_compute:
                        if canonical_path.is_file():
                            validate_token_boundary_overlay_shard(
                                canonical_path,
                                expected_n_segments=n_segments,
                            )
                            n_skipped += 1
                        del compact_data
                        continue

                    parquet_path = parquet_by_stem[stem]
                    if source_bos_char_len is None:
                        source_bos_char_len = _detect_source_bos_char_len_from_data(
                            compact_data,
                            parquet_path,
                        )
                        nanochat_bos_char_len = len(tokenizer.decode([bos_token]))
                        logger.info(
                            f"BOS char lengths: source={source_bos_char_len}, "
                            f"nanochat={nanochat_bos_char_len}, "
                            f"delta={source_bos_char_len - nanochat_bos_char_len}"
                        )
                        if n_workers > 1:
                            saved_thread_env = _apply_worker_thread_env()
                            pool_started = time.monotonic()
                            try:
                                persistent_pool = ProcessPoolExecutor(
                                    max_workers=n_workers,
                                    mp_context=mp.get_context("spawn"),
                                    initializer=_init_worker_process,
                                    initargs=(
                                        tokenizer,
                                        source_bos_char_len,
                                        source_segment_size,
                                        max_tokens,
                                        min_tokens,
                                    ),
                                )
                            except BaseException:
                                _restore_worker_thread_env(saved_thread_env)
                                saved_thread_env = None
                                raise
                            if profiler is not None:
                                profiler.worker_pool = {
                                    "spawn_count": 1,
                                    "total_init_seconds": time.monotonic() - pool_started,
                                    "thread_caps": {
                                        "torch_num_threads": 1,
                                        "torch_num_interop_threads": 1,
                                        **_WORKER_THREAD_ENV_VARS,
                                    },
                                }

                    assert source_bos_char_len is not None
                    timing = (
                        ShardTiming(stem=stem, n_segments=0)
                        if profiler is not None
                        else None
                    )
                    if timing is not None:
                        timing.load_features_seconds = compact_load_seconds
                        timing.load_bytes = prepared.remote_identity.byte_length
                        timing.n_segments = n_segments
                    should_trace = profile and profile_trace and n_processed == 0
                    if should_trace:
                        trace_path = Path(
                            cfg.get("profile_output", None)
                            or "out/traces/tok_boundary_trace.json"
                        )
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        with torch_profile(
                            activities=[ProfilerActivity.CPU],
                            with_stack=True,
                            record_shapes=True,
                        ) as prof:
                            precompute_shard(
                                prepared.local_path,
                                parquet_path,
                                tokenizer,
                                bos_token,
                                candidate_dir,
                                stem,
                                source_bos_char_len,
                                source_segment_size,
                                dry_run=False,
                                zstd_level=zstd_level,
                                max_nanochat_tokens_per_segment=max_tokens,
                                min_nanochat_tokens_per_segment=min_tokens,
                                timing=timing,
                                worker_pool=None,
                                preloaded_data=compact_data,
                            )
                        prof.export_chrome_trace(str(trace_path))
                    else:
                        precompute_shard(
                            prepared.local_path,
                            parquet_path,
                            tokenizer,
                            bos_token,
                            candidate_dir,
                            stem,
                            source_bos_char_len,
                            source_segment_size,
                            dry_run=False,
                            zstd_level=zstd_level,
                            max_nanochat_tokens_per_segment=max_tokens,
                            min_nanochat_tokens_per_segment=min_tokens,
                            timing=timing,
                            worker_pool=persistent_pool,
                            preloaded_data=compact_data,
                        )
                    candidate_path = overlay_shard_path(candidate_dir, stem)
                    validate_token_boundary_overlay_shard(
                        candidate_path,
                        expected_n_segments=n_segments,
                    )
                    candidate_paths[stem] = candidate_path
                    n_processed += 1
                    if profiler is not None and timing is not None:
                        profiler.add_shard(timing)
                    del compact_data
                    logger.info(
                        f"  [{position + 1}/{len(inventory)}] {stem} candidate ready"
                    )
            finally:
                if persistent_pool is not None:
                    persistent_pool.shutdown()
                    persistent_pool = None
                if saved_thread_env is not None:
                    _restore_worker_thread_env(saved_thread_env)
                    saved_thread_env = None

            base_hash = transport.manifest.base_shard_content_hash
            if profiler is not None:
                profiler.manifest_pre_pass = {
                    "seconds": time.monotonic() - wall_start,
                    "bytes_read": 0,
                    "cache_hit": True,
                    "n_shards_hashed": len(inventory),
                }

            if has_matching_manifest and not force:
                authenticated = discover_overlay(
                    feature_dir,
                    OverlayKind.SEG_BOUNDARY,
                    checksum,
                    precomputed_base_hash=base_hash,
                )
                if authenticated != output:
                    raise RuntimeError(
                        f"existing token-boundary manifest could not be authenticated: {output}"
                    )

            canonical_missing = sorted(
                stem
                for stem in expected_stems
                if not overlay_shard_path(output, stem).is_file()
            )
            if has_matching_manifest and not force and not resume and canonical_missing:
                raise OverlayIncompleteError(
                    f"Token-boundary overlay at {output} is incomplete: missing "
                    f"{len(canonical_missing)} expected shard(s). Use resume=true to fill them."
                )

            selected_expected = expected_stems & selected_stems
            for stem in sorted(selected_expected):
                validation_path = candidate_paths.get(
                    stem,
                    overlay_shard_path(output, stem),
                )
                if not validation_path.is_file():
                    raise OverlayIncompleteError(
                        f"Token-boundary overlay candidate coverage is missing {stem}"
                    )
                validate_token_boundary_overlay_shard(
                    validation_path,
                    expected_n_segments=n_segments_by_stem[stem],
                )

            transport.final_inventory_fence()

            if not dry_run:
                output.mkdir(parents=True, exist_ok=True)
                manifest_path = output / MANIFEST_FILENAME
                if force and manifest_path.exists():
                    manifest_path.unlink()
                for stem, candidate_path in sorted(candidate_paths.items()):
                    os.replace(candidate_path, overlay_shard_path(output, stem))
                if force or not has_matching_manifest:
                    write_manifest(
                        output,
                        checksum,
                        OverlayKind.SEG_BOUNDARY,
                        precomputed_base_hash=base_hash,
                    )

                for stem in sorted(selected_expected):
                    validate_token_boundary_overlay_shard(
                        overlay_shard_path(output, stem),
                        expected_n_segments=n_segments_by_stem[stem],
                    )
                if stats_path is not None:
                    save_tokenizer_checksum(stats_path, checksum)
                    logger.info(f"Saved tokenizer checksum to {stats_path}")

        run_status = "success"
        transport_profile = transport.profile_payload()
    finally:
        active_exception = sys.exception()
        candidate_cleanup_error: BaseException | None = None
        if persistent_pool is not None:
            persistent_pool.shutdown()
        if saved_thread_env is not None:
            _restore_worker_thread_env(saved_thread_env)
        try:
            shutil.rmtree(candidate_dir)
        except FileNotFoundError:
            pass
        except BaseException as caught:
            candidate_cleanup_error = caught
            run_status = "failed"
            if active_exception is not None:
                active_exception.add_note(
                    f"candidate overlay cleanup also failed: {caught!r}"
                )
        if not transport_profile:
            transport_profile = transport.profile_payload()
        if profiler is not None:
            profiler.remote_transport = transport_profile
            profiler.totals = {
                "wall_seconds": time.monotonic() - wall_start,
                "n_processed": n_processed,
                "n_skipped": n_skipped,
                "status": run_status,
                "dry_run": dry_run,
            }
            sidecar_path = Path(
                profile_summary_output or "out/traces/tok_boundary_profile.json"
            )
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.write(sidecar_path)
            logger.info(f"Profile sidecar → {sidecar_path}")
        if candidate_cleanup_error is not None and active_exception is None:
            raise candidate_cleanup_error

    logger.info(
        f"Done: {n_processed} processed, {n_skipped} validated in "
        f"{(time.monotonic() - wall_start) / 60:.1f}m "
        f"(overlay={output}, checksum={checksum[:16]}..., remote=true)"
    )
    return 0


def run(cfg: DictConfig) -> int:
    """Dispatch to the literal local runner or the bounded remote runner."""
    return _run_remote(cfg) if _remote_enabled(cfg) else _run_local(cfg)


@hydra.main(
    config_path="../../configs/extraction",
    config_name="precompute_token_boundaries",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
