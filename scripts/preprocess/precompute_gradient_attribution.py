"""Precompute gradient-attribution scores as base-aligned overlay shards.

For each base feature shard, re-derives every document's SAE tokens from the
source parquet and verifies the stored token-count/segment geometry. Base
shards do not yet carry the token-stream identity witness needed to authenticate
the re-tokenized ids, so the producer currently fails before scoring rather
than risk a misaligned overlay. The eventual scoring path runs the full-model
forward + per-unit backward schedule of
:class:`steering.gradient_attribution.GradientAttributionCollector`, and
writes one fp16 score per stored (segment, feature) to
``{feature_dir}/grad/<checksum[:16]>/shard_NNNNN.safetensors.zst``.
Base shards are never modified.

Each grad shard is self-contained for downstream joins: alongside the flat
``grad_scores`` it stores verified copies of the base CSR ``offsets`` and a
per-doc segment cumsum ``doc_seg_offsets``, so consumers (the selection
precompute) never load base shards.

The overlay directory name encodes the SHA-256 of the canonical gradient
descriptor (mode, merge threshold, model/SAE identity, dtype, reduction +
split-policy version); ``manifest.json`` stores the full checksum, the
descriptor, and the base-shard content hash (schema v2 staleness guard).

Usage::

    uv run python -m scripts.preprocess.precompute_gradient_attribution \
        attribution_mode=future_half_split

    # Multi-GPU (round-robin shard striping, resumable):
    torchrun --standalone --nproc_per_node=4 \
        -m scripts.preprocess.precompute_gradient_attribution resume=true
"""

from __future__ import annotations

import gc
import os
import sys
import time
from collections import Counter
from pathlib import Path

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig
from sae_lens import SAE
from transformers import AutoConfig

from errors import EmptyExtractionSegmentError, GradientAttributionError
from scripts.preprocess._common import resolve_feature_dir
from steering.feature_io import (
    FeatureDirectoryMismatchError,
    assert_uniform_ignore_bos,
    assert_uniform_l0_tier,
    empty_segment_indices,
    glob_features,
    load_features,
    read_shard_metadata,
    shard_stem_from_path,
)
from steering.feature_paths import (
    FeatureDirectoryComponents,
    PoolingMode,
    parse_feature_dir,
    resolve_data_dir,
)
from steering.gradient_attribution import (
    GradientAttributionCollector,
    build_gradient_model,
    decoder_linearity_gate,
)
from steering.gradient_overlay import (
    GRAD_DOC_SEG_OFFSETS_KEY,
    GRAD_OFFSETS_KEY,
    GRAD_SCORES_KEY,
    MIN_SCOREABLE_TOKENS,
    GradientAttributionMode,
    canonical_gradient_descriptor,
    compute_gradient_overlay_checksum,
)
from steering.models import EXTRACTION_MODELS, max_position_embeddings, resolve_tokenizer
from steering.overlay_io import (
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    expected_overlay_shard_stems,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    resolve_single_percentile_overlay,
    save_overlay_shard,
    validate_percentile_overlay_shard,
    write_manifest,
)
from steering.parquet_io import read_parquet_texts
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    l0_tier_for_resolved,
    resolve_sae,
)
from steering.segmentation import (
    SegmentMode,
    batch_truncate_to_training_window,
    resolve_bos_token_id,
    segment_mode_from_metadata,
    segment_tokens,
)
from steering.shard_alignment import (
    AlignmentViolation,
    canonical_doc_blocks,
    verify_document_alignment,
)

_DTYPES = {"bfloat16": t.bfloat16, "float32": t.float32}
_CUDA_DIAGNOSTIC_HISTOGRAM_ENTRY_LIMIT = 14


def _live_cuda_hist() -> list[tuple[tuple[tuple[int, ...], str], int]]:
    """Count live CUDA tensors by ``(shape, dtype)`` for diagnostics."""
    counts: Counter[tuple[tuple[int, ...], str]] = Counter()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, t.Tensor) and obj.is_cuda:
                counts[(tuple(obj.shape), str(obj.dtype))] += 1
        except Exception:  # pragma: no cover - defensive over gc churn
            continue
    return counts.most_common(_CUDA_DIAGNOSTIC_HISTOGRAM_ENTRY_LIMIT)


def _require_positive_batch_size(value: object, *, caller: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GradientAttributionError(
            f"{caller}: batch_size must be a non-bool integer greater than "
            f"zero, got {value!r}."
        )
    return value


def _assert_residual_feature_dir(
    feature_dir: Path,
) -> FeatureDirectoryComponents:
    """Parse a canonical feature path and require exact residual identity."""
    try:
        components = parse_feature_dir(feature_dir)
    except ValueError as exc:
        raise GradientAttributionError(
            f"explicit feature_dir must be a canonical feature path with a "
            f"parseable a_res component; got {feature_dir}: {exc}"
        ) from exc
    if components.activation_type is not ActivationType.residual:
        raise GradientAttributionError(
            f"the canonical feature path encodes activation_type="
            f"{components.activation_type.value!r}, but gradient attribution "
            f"is residual-only: {feature_dir}."
        )
    return components


def _parse_required_int_metadata(
    metadata: dict[str, str], field: str, *, shard_path: Path
) -> int:
    raw = metadata.get(field)
    if raw is None:
        raise GradientAttributionError(
            f"{shard_path.name} is missing required {field!r} metadata."
        )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise GradientAttributionError(
            f"{shard_path.name} has malformed {field}={raw!r}; expected an integer."
        ) from exc
    if str(value) != raw:
        raise GradientAttributionError(
            f"{shard_path.name} has non-canonical {field}={raw!r}; expected "
            f"canonical decimal metadata."
        )
    return value


def _parse_required_bool_metadata(
    metadata: dict[str, str], field: str, *, shard_path: Path
) -> bool:
    raw = metadata.get(field)
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw is None:
        raise GradientAttributionError(
            f"{shard_path.name} is missing required {field!r} metadata."
        )
    raise GradientAttributionError(
        f"{shard_path.name} has malformed {field}={raw!r}; expected "
        f"exactly 'true' or 'false'."
    )


def _preflight_base_shard_identity(
    feature_dir: Path, shard_paths: list[Path], cfg: DictConfig
) -> bool:
    """Validate canonical path plus required identity metadata on every shard."""
    components = _assert_residual_feature_dir(feature_dir)
    expected = {
        "segment_size": int(cfg.segment_size),
        "layer": int(cfg.layer),
        "pooling": PoolingMode(cfg.pooling),
        "sae_width": SAEWidth(cfg.sae_width),
        "model": str(cfg.model),
    }
    path_values = {
        "segment_size": components.segment_size,
        "layer": components.layer,
        "pooling": components.pooling,
        "sae_width": components.width,
        "model": components.model,
    }
    path_mismatches = [
        f"{field}: path={path_values[field]!r}, config={value!r}"
        for field, value in expected.items()
        if path_values[field] != value
    ]
    if path_mismatches:
        raise GradientAttributionError(
            f"canonical feature path identity mismatch for {feature_dir}:\n  - "
            + "\n  - ".join(path_mismatches)
        )

    required_values: dict[Path, dict[str, object]] = {}
    for shard_path in shard_paths:
        metadata = read_shard_metadata(shard_path)
        raw_pooling = metadata.get("pooling")
        if raw_pooling is None:
            raise GradientAttributionError(
                f"{shard_path.name} is missing required 'pooling' metadata."
            )
        try:
            pooling = PoolingMode(raw_pooling)
        except ValueError as exc:
            raise GradientAttributionError(
                f"{shard_path.name} has malformed pooling={raw_pooling!r}."
            ) from exc
        values: dict[str, object] = {
            "layer": _parse_required_int_metadata(
                metadata, "layer", shard_path=shard_path
            ),
            "context_window": _parse_required_int_metadata(
                metadata, "context_window", shard_path=shard_path
            ),
            "context_independent_features": _parse_required_bool_metadata(
                metadata, "context_independent_features", shard_path=shard_path
            ),
            "pooling": pooling,
            "segment_size": _parse_required_int_metadata(
                metadata, "segment_size", shard_path=shard_path
            ),
        }
        required_values[shard_path] = values

    reference_path = shard_paths[0]
    reference = required_values[reference_path]
    for shard_path, values in required_values.items():
        for field, reference_value in reference.items():
            if values[field] != reference_value:
                raise GradientAttributionError(
                    f"{field} disagrees across base shards: "
                    f"{reference_path.name}={reference_value!r}, "
                    f"{shard_path.name}={values[field]!r}."
                )

    config_expectations: dict[str, object] = {
        "layer": int(cfg.layer),
        "context_window": int(cfg.context_window),
        "context_independent_features": False,
        "pooling": PoolingMode(cfg.pooling),
        "segment_size": int(cfg.segment_size),
    }
    for field, expected_value in config_expectations.items():
        if reference[field] != expected_value:
            raise GradientAttributionError(
                f"{field} mismatch: config/path require {expected_value!r}, "
                f"base shards record {reference[field]!r}."
            )
    if components.context_independent_features is not bool(
        reference["context_independent_features"]
    ):
        raise GradientAttributionError(
            f"context_independent_features mismatch: canonical path records "
            f"{components.context_independent_features}, base shards record "
            f"{reference['context_independent_features']}."
        )

    try:
        return assert_uniform_ignore_bos(shard_paths)
    except (FeatureDirectoryMismatchError, ValueError) as exc:
        raise GradientAttributionError(str(exc)) from exc


def _preflight_base_doc_blocks(shard_paths: list[Path]) -> None:
    """Require canonical document blocks before constructing scoring objects."""
    for shard_path in shard_paths:
        data = load_features(shard_path, keys={"doc_ids"})
        try:
            n_docs = data["n_docs"]
            n_segments = data["n_segments"]
        except KeyError as exc:
            raise GradientAttributionError(
                f"{shard_path.name} is missing required {exc.args[0]!r} metadata."
            ) from exc
        boundaries = _doc_seg_offsets(data["doc_ids"], n_docs)
        if int(boundaries[-1]) != n_segments:
            raise GradientAttributionError(
                f"{shard_path.name}: canonical doc block total "
                f"{int(boundaries[-1])} != n_segments={n_segments}."
            )


def _doc_seg_offsets(doc_ids: t.Tensor, n_docs: int) -> t.Tensor:
    """Per-doc segment-count cumsum [n_docs + 1] (int32) from the shard's
    ``doc_ids`` array. Base shards order segments by doc, then ordinal.

    Adapts the shared pure alignment core to the tensor/error contract used by
    gradient overlays.
    """
    result = canonical_doc_blocks(doc_ids, n_docs)
    if isinstance(result, AlignmentViolation):
        raise GradientAttributionError(
            f"invalid base-shard doc_ids ({result.code}): {result.detail}"
        )
    return t.tensor(result.boundaries, dtype=t.int32)


def _verify_doc_alignment(
    *,
    shard_name: str,
    doc_idx: int,
    n_full_tokens: int,
    segments: list[tuple[int, int]],
    data: dict,
    seg_lo: int,
    seg_hi: int,
) -> None:
    """Fail loud if re-derived token counts/segments disagree with the base shard.

    This proves geometry, not token-id identity; the caller guards that
    deferred witness gap before scoring.

    Adapts the shared pure alignment core to ``GradientAttributionError``.
    """
    violation = verify_document_alignment(
        doc_idx=doc_idx,
        n_full_tokens=n_full_tokens,
        segments=segments,
        stored_doc_token_counts=data["doc_token_counts"],
        stored_segment_positions=data["segment_positions"],
        stored_segment_indices=data["segment_indices"],
        seg_lo=seg_lo,
        seg_hi=seg_hi,
    )
    if violation is not None:
        raise GradientAttributionError(
            f"{shard_name}: {violation.detail}; refusing to attribute "
            f"misaligned tokens."
        )


def _raise_for_unwitnessed_base_token_stream_identity() -> None:
    raise NotImplementedError(
        "verifying base-shard token-stream identity is not implemented "
        "(deferred; it requires a base-shard identity witness). Token counts "
        "and segment geometry cannot prove that re-tokenization produced the "
        "ids used for feature extraction. Use percentile-based selection by "
        "leaving feature_selection.gradient unset until that witness exists."
    )


def _resolve_pct_overlay(
    feature_dir: Path, base_shard_paths: list[Path]
) -> tuple[Path, str]:
    """The percentile overlay ``(dir, full checksum)`` the score is weighted by.

    REQUIRED: the stored score is ``pct * sensitivity``. Delegates to the shared
    strict resolver (:func:`steering.overlay_io.resolve_single_percentile_overlay`)
    so the producer and the gradient SELECTION resolve the SAME percentile
    identity — that identity is now part of the grad overlay's content hash, so
    a different percentile overlay re-addresses the ``grad/<hash>/`` dir."""
    try:
        candidate, checksum = resolve_single_percentile_overlay(feature_dir)
    except OverlayStaleError as exc:
        raise GradientAttributionError(
            f"{exc} The gradient score is pct * sensitivity."
        ) from exc
    discovered = discover_overlay(
        feature_dir,
        OverlayKind.PERCENTILE,
        checksum,
        base_shard_paths=base_shard_paths,
    )
    if discovered is None or discovered != candidate:
        raise OverlayStaleError(
            f"percentile overlay {candidate} could not be rediscovered through "
            f"its manifest checksum {checksum}; rebuild it."
        )

    expected_stems = expected_overlay_shard_stems(
        base_shard_paths, OverlayKind.PERCENTILE
    )
    actual_stems = [
        shard_stem_from_path(path) for path in glob_overlay_shards(discovered)
    ]
    if actual_stems != expected_stems:
        missing = sorted(set(expected_stems) - set(actual_stems))
        extra = sorted(set(actual_stems) - set(expected_stems))
        raise OverlayStaleError(
            f"percentile overlay coverage mismatch at {discovered}: "
            f"missing={missing}, extra={extra}; rebuild the overlay."
        )
    for base_path, stem in zip(base_shard_paths, expected_stems, strict=True):
        base_offsets = load_features(base_path, keys={"offsets"})["offsets"]
        expected_numel = int(base_offsets[-1])
        validate_percentile_overlay_shard(
            overlay_shard_path(discovered, stem), expected_numel=expected_numel
        )
    return discovered, checksum


def precompute_shard(
    shard_path: Path,
    parquet_path: Path,
    collector: GradientAttributionCollector,
    tokenizer,
    *,
    context_window: int,
    sae_max_context: int,
    segment_size: int,
    batch_size: int,
    output_dir: Path,
    shard_stem: str,
    pct_overlay_dir: Path,
    dry_run: bool = False,
    zstd_level: int | None = None,
) -> bool:
    """Score one base shard and write its grad overlay shard.

    The collector yields the gradient SENSITIVITY per (segment, feature); this
    function multiplies by the feature's activation PERCENTILE (from
    ``pct_overlay_dir``, uint8 / 255, CSR-aligned 1:1 with the base feature_ids)
    to form the stored ``pct * sensitivity`` score."""
    batch_size = _require_positive_batch_size(
        batch_size, caller="precompute_shard"
    )
    refined_segment_size = PositiveInt.parse(segment_size)
    t0 = time.monotonic()
    data = load_features(shard_path)
    segment_mode = segment_mode_from_metadata(data, source=shard_path)
    if segment_mode is not SegmentMode.fixed_tokens:
        raise GradientAttributionError(
            f"gradient-attribution precompute supports fixed_tokens base shards only (segment_mode={segment_mode.value!r})"
        )
    empties = empty_segment_indices(data["offsets"])
    if empties:
        raise EmptyExtractionSegmentError(
            f"precompute_gradient_attribution: base shard {shard_path.name} "
            f"segment {empties[0]} has zero features (issue #522)."
        )

    n_docs = int(data["n_docs"])
    n_segments = int(data["n_segments"])
    offsets = data["offsets"].to(t.int64)
    feature_ids = data["feature_ids"].to(t.int64)
    doc_seg_offsets = _doc_seg_offsets(data["doc_ids"], n_docs)
    if int(doc_seg_offsets[-1]) != n_segments:
        raise GradientAttributionError(
            f"{shard_stem}: doc_seg_offsets[-1]={int(doc_seg_offsets[-1])} != "
            f"n_segments={n_segments}"
        )

    texts = read_parquet_texts(parquet_path)
    if len(texts) != n_docs:
        raise GradientAttributionError(
            f"{shard_stem}: parquet {parquet_path.name} has {len(texts)} docs "
            f"but the base shard recorded {n_docs}; wrong data_dir?"
        )

    flat_scores = t.zeros(len(feature_ids), dtype=t.float16)
    t_compute0 = time.monotonic()

    # First pass (CPU): re-tokenize + verify every doc, collecting the
    # per-doc inputs. Scores are written back through ABSOLUTE base-CSR
    # slices, so the GPU processing order is free — see the sort below.
    doc_inputs: list[tuple[list[int], list[tuple[int, int]], list[t.Tensor], list[tuple[int, int]]]] = []
    # Batch-tokenize the shard's docs up front (chunked): one tokenizer call per
    # chunk amortizes the Python->tokenizer FFI and lets fast tokenizers thread
    # internally, instead of one call per doc on this CPU pass that gates the GPU
    # pass. Bit-identical to per-doc tokenization (padding=False => independent
    # encodings). Geometry drift is caught by _verify_doc_alignment below;
    # equal-geometry token-id drift reaches the explicit identity guard.
    tokenized = batch_truncate_to_training_window(
        texts[:n_docs], tokenizer, context_window, sae_max_context
    )
    for doc_idx in range(n_docs):
        kept_ids, n_full = tokenized[doc_idx]
        if len(kept_ids) < MIN_SCOREABLE_TOKENS:
            raise GradientAttributionError(
                f"{shard_stem} doc {doc_idx}: {len(kept_ids)} token(s) — a "
                f"gradient-attributable doc needs >= {MIN_SCOREABLE_TOKENS} "
                f"tokens (token 0 is never predicted, so a 0/1-token doc has "
                f"no LM loss). The base extraction should not emit such docs."
            )
        segments = segment_tokens(
            kept_ids,
            SegmentMode.fixed_tokens,
            refined_segment_size,
        )
        seg_lo = int(doc_seg_offsets[doc_idx])
        seg_hi = int(doc_seg_offsets[doc_idx + 1])
        _verify_doc_alignment(
            shard_name=shard_stem,
            doc_idx=doc_idx,
            n_full_tokens=n_full,
            segments=segments,
            data=data,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
        )
        doc_fid_rows: list[t.Tensor] = []
        doc_slices: list[tuple[int, int]] = []
        for s in range(seg_lo, seg_hi):
            f_lo, f_hi = int(offsets[s]), int(offsets[s + 1])
            doc_fid_rows.append(feature_ids[f_lo:f_hi])
            doc_slices.append((f_lo, f_hi))
        doc_inputs.append((kept_ids, segments, doc_fid_rows, doc_slices))

    if doc_inputs:
        # GA-001: joining re-tokenized ids to base CSR rows needs a token-stream
        # witness; counts and segment coordinates alone cannot authenticate it.
        _raise_for_unwitnessed_base_token_stream_identity()

    # Second pass (GPU): batches of length-SORTED docs. Right-padded maskless
    # batches pay for every pad row in the forward AND every per-unit
    # backward; corpus doc lengths are heavy-tailed, so raw-order batches of
    # 4 waste ~30-50% of compute on pads. The sort is stable and
    # deterministic (reruns stay byte-identical), and batch composition only
    # perturbs scores within the documented pad/permutation-invariance
    # envelope — no byte contract spans batch shapes (CUDA overlays are
    # already declared non-bit-reproducible).
    order = sorted(
        range(n_docs), key=lambda d: len(doc_inputs[d][0]), reverse=True
    )
    processed_doc_counts = [0] * n_docs

    # Opt-in per-batch VRAM probe (GRAD_MEM_DEBUG=1). Logs CUDA *allocated*
    # (live tensors) vs *reserved* (cached blocks) so a real cross-batch leak
    # (allocated climbs) is distinguishable from caching-allocator
    # fragmentation (allocated flat, reserved climbs) — the latter is the
    # expected failure mode here: gemma-3's 262k-vocab lm_head makes each
    # group's logit tensor large and SIZE-VARYING across the length-sorted
    # batches, which the default allocator caches per-size and cannot reuse.
    mem_debug = bool(os.environ.get("GRAD_MEM_DEBUG", "").strip()) and t.cuda.is_available()

    def _flush(batch_idx: int, batch_order: list[int]) -> None:
        scored = collector.score_documents(
            [doc_inputs[d][0] for d in batch_order],
            [doc_inputs[d][1] for d in batch_order],
            [doc_inputs[d][2] for d in batch_order],
        )
        for d, doc_rows in zip(batch_order, scored, strict=True):
            for row, (f_lo, f_hi) in zip(doc_rows, doc_inputs[d][3], strict=True):
                if row.numel() != (f_hi - f_lo):  # pragma: no cover - guard
                    raise GradientAttributionError(
                        f"{shard_stem}: score row length {row.numel()} != "
                        f"stored feature count {f_hi - f_lo}"
                    )
                flat_scores[f_lo:f_hi] = row
            processed_doc_counts[d] += 1
        if mem_debug:
            t.cuda.synchronize()
            max_len = max(len(doc_inputs[d][0]) for d in batch_order)
            logger.info(
                f"  [mem] {shard_stem} b{batch_idx} bs={len(batch_order)} "
                f"maxlen={max_len} alloc={t.cuda.memory_allocated() >> 20}MiB "
                f"reserved={t.cuda.memory_reserved() >> 20}MiB"
            )
            if batch_idx % 100 == 0:
                logger.info(
                    f"  [mem-hist] b{batch_idx} live CUDA tensors "
                    f"(shape,dtype)xN: {_live_cuda_hist()}"
                )

    for batch_idx, lo in enumerate(range(0, n_docs, batch_size)):
        _flush(batch_idx, order[lo : lo + batch_size])
    coverage_errors = [
        (doc_idx, count)
        for doc_idx, count in enumerate(processed_doc_counts)
        if count != 1
    ]
    if coverage_errors:
        raise GradientAttributionError(
            f"{shard_stem}: document processing coverage is not exactly once: "
            f"{coverage_errors[:10]}"
        )
    t_compute = time.monotonic() - t_compute0

    # Bake in the activation-PERCENTILE weight: stored score = pct * sensitivity
    # (pct = feature_percentiles uint8 / 255 in [0,1], CSR-aligned 1:1 with the
    # base feature_ids — same order flat_scores was filled in). magnitude_to_
    # percentile floors at 0.5 ("below p1"), which encodes to uint8 == 1, so a
    # stored (fired) feature always has pct >= 1/255 > 0; the score>0 eligibility
    # set is therefore preserved in exact arithmetic (pct*sens > 0 iff sens > 0).
    # The sole edge is a near-zero sensitivity whose fp16 product underflows to 0
    # — a benign, essentially-zero-signal feature that ranking would drop anyway.
    # fp32 multiply then fp16 so the producer matches the bit-exact expected
    # (collector_fp16 * pct), underflow included.
    pct_uint8 = validate_percentile_overlay_shard(
        overlay_shard_path(pct_overlay_dir, shard_stem),
        expected_numel=len(flat_scores),
    )
    flat_scores = (
        flat_scores.to(t.float32) * (pct_uint8.to(t.float32) / 255.0)
    ).to(t.float16)

    zero_frac = float((flat_scores == 0).float().mean()) if len(flat_scores) else 0.0
    if dry_run:
        logger.info(
            f"  [DRY-RUN] {shard_stem}: {len(flat_scores):,} scores "
            f"(zero-fraction {zero_frac:.3f}, compute={t_compute:.1f}s)"
        )
        return True

    out_path = overlay_shard_path(output_dir, shard_stem)
    save_overlay_shard(
        {
            GRAD_SCORES_KEY: flat_scores,
            GRAD_OFFSETS_KEY: offsets,
            GRAD_DOC_SEG_OFFSETS_KEY: doc_seg_offsets,
        },
        out_path,
        zstd_level=zstd_level,
    )
    logger.info(
        f"  {shard_stem}: {len(flat_scores):,} scores in "
        f"{time.monotonic() - t0:.0f}s (compute={t_compute:.0f}s, "
        f"zero-fraction {zero_frac:.3f}, max {float(flat_scores.max()) if len(flat_scores) else 0:.3g})"
    )
    return True


def run(cfg: DictConfig) -> int:
    """Run from a complete Hydra config or equivalent fixture with patched builders."""
    feature_dir = resolve_feature_dir(cfg)
    if not feature_dir.is_dir():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1
    batch_size = _require_positive_batch_size(
        cfg.batch_size, caller="precompute_gradient_attribution.run"
    )
    segment_size = int(cfg.segment_size)
    if segment_size < MIN_SCOREABLE_TOKENS:
        raise GradientAttributionError(
            f"gradient attribution requires segment_size >= "
            f"{MIN_SCOREABLE_TOKENS}, got {segment_size} (a 1-token segment "
            f"has no own predictions)."
        )

    activation_type = ActivationType(cfg.activation_type)
    if activation_type is not ActivationType.residual:
        raise GradientAttributionError(
            f"gradient attribution is residual-only; got "
            f"activation_type={activation_type.value!r}."
        )
    mode = GradientAttributionMode(cfg.attribution_mode)
    min_final = int(cfg.min_final_segment_tokens)
    if min_final < MIN_SCOREABLE_TOKENS:
        raise GradientAttributionError(
            f"min_final_segment_tokens must be >= {MIN_SCOREABLE_TOKENS}, "
            f"got {min_final}. {MIN_SCOREABLE_TOKENS} merges only the "
            f"unscoreable 1-token finals; raise it to merge longer noisy finals."
        )
    all_shard_paths = glob_features(feature_dir)
    if not all_shard_paths:
        logger.error(f"No feature shards found in {feature_dir}")
        return 1

    if "ignore_bos_features" in cfg:
        raise GradientAttributionError(
            "ignore_bos_features is obsolete for the gradient producer; remove "
            "ignore_bos_features from the config/CLI. BOS handling is inherited "
            "from the validated base shards."
        )
    ignore_bos = _preflight_base_shard_identity(
        feature_dir, all_shard_paths, cfg
    )
    _preflight_base_doc_blocks(all_shard_paths)

    # The stored score is pct * sensitivity. Validate manifest staleness,
    # exact coverage, and every percentile payload before resolving weights.
    pct_overlay_dir, pct_checksum = _resolve_pct_overlay(
        feature_dir, all_shard_paths
    )
    logger.info(
        f"Percentile overlay: {pct_overlay_dir.name} ({pct_checksum[:16]}...)"
    )

    sae_width = SAEWidth(cfg.sae_width)
    resolved = resolve_sae(
        model=cfg.model,
        release=SAERelease(cfg.sae.release),
        activation_type=activation_type,
        layer=int(cfg.layer),
        width=sae_width,
        l0=coerce_config_l0(SAERelease(cfg.sae.release), cfg.sae.l0),
    )
    expected_hook_name = f"blocks.{int(cfg.layer)}.hook_resid_post"
    if resolved.hook_name != expected_hook_name:
        raise GradientAttributionError(
            f"resolved SAE hook {resolved.hook_name!r} does not exactly match "
            f"the required {expected_hook_name!r}."
        )
    try:
        base_l0_tier = assert_uniform_l0_tier(all_shard_paths)
    except (FeatureDirectoryMismatchError, ValueError) as exc:
        raise GradientAttributionError(str(exc)) from exc
    resolved_l0_tier = l0_tier_for_resolved(resolved)
    if base_l0_tier != resolved_l0_tier:
        raise GradientAttributionError(
            f"l0_tier mismatch: base shards record "
            f"{getattr(base_l0_tier, 'value', base_l0_tier)!r}, but the "
            f"resolved SAE requires "
            f"{getattr(resolved_l0_tier, 'value', resolved_l0_tier)!r}."
        )

    descriptor = canonical_gradient_descriptor(
        attribution_mode=mode.value,
        model=cfg.model,
        model_dtype=str(cfg.model_dtype),
        layer=int(cfg.layer),
        sae_release=str(cfg.sae.release),
        sae_width=str(cfg.sae_width),
        sae_l0=cfg.sae.l0,
        activation_type=activation_type.value,
        ignore_bos_features=ignore_bos,
        min_final_segment_tokens=min_final,
        percentile_checksum=pct_checksum,
    )
    checksum = compute_gradient_overlay_checksum(descriptor)
    logger.info(f"Gradient overlay checksum: {checksum[:16]}... ({descriptor})")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    rank_prefix = f"[rank {rank}/{world_size}] " if world_size > 1 else ""

    device_str = cfg.device
    if device_str == "auto":
        device = (
            t.device(f"cuda:{local_rank % t.cuda.device_count()}")
            if t.cuda.is_available()
            else t.device("cpu")
        )
    else:
        device = t.device(device_str)
    model_dtype = _DTYPES[str(cfg.model_dtype)]
    logger.info(f"{rank_prefix}Using device: {device}, dtype: {model_dtype}")

    dry_run = bool(cfg.dry_run)
    resume = bool(cfg.resume)
    force = bool(cfg.force)
    zstd_level = cfg.zstd_level
    if force and resume:
        logger.warning("Both force=true and resume=true set; force takes precedence")
        resume = False

    output = overlay_dir(feature_dir, OverlayKind.GRADIENT, checksum)
    # Staleness gate FIRST (skipped under force, which rewrites everything):
    # discover_overlay(base_shard_paths=...) raises OverlayStaleError when the
    # existing manifest's base-shard content hash no longer matches the
    # CURRENT base shards — re-extracted bases must never be silently resumed
    # over (nor have their manifest re-stamped, which would launder the stale
    # score shards past the consumer's gate).
    existing = (
        discover_overlay(
            feature_dir,
            OverlayKind.GRADIENT,
            checksum,
            base_shard_paths=all_shard_paths,
        )
        if not force
        else None
    )
    # The score = pct * sensitivity, and the percentile_checksum is now part of
    # the content hash (canonical_gradient_descriptor), so a DIFFERENT pct
    # overlay re-addresses grad/<hash>/ — a force-rebuild against a new pct lands
    # in a NEW dir instead of silently overwriting these scores in place. No
    # runtime pct-staleness comparison is needed.
    if existing is not None and not force and not resume:
        missing = [
            shard_stem_from_path(p)
            for p in all_shard_paths
            if not overlay_shard_path(existing, shard_stem_from_path(p)).exists()
        ]
        if not missing:
            logger.info(
                f"Overlay already complete at {existing} "
                f"({len(all_shard_paths)} shards). Use force=true to rewrite."
            )
            return 0
        if len(missing) < len(all_shard_paths):
            # Genuinely partial (an interrupted run): exit NON-zero so
            # orchestration cannot mistake it for success.
            logger.error(
                f"Overlay at {existing} is missing {len(missing)} of "
                f"{len(all_shard_paths)} shards (e.g. {missing[:3]}). Use "
                f"resume=true to fill the gaps, or force=true to rewrite all."
            )
            return 1
        # Manifest present but ZERO shards: a concurrent fresh DDP launch
        # (rank 0 stamps the manifest before any stripe computes) — proceed
        # as a fresh run rather than short-circuiting this rank's stripe.
        logger.info(
            f"{rank_prefix}Manifest present with no shards at {existing}; "
            f"treating as a fresh run."
        )

    data_dir = Path(resolve_data_dir(cfg.data_dir, cfg.dataset_phase))
    if not data_dir.is_dir():
        logger.error(f"Source parquet dir not found: {data_dir}")
        return 1
    shard_paths = all_shard_paths[rank::world_size]
    parquets: dict[str, Path] = {}
    for shard_path in shard_paths:
        stem = shard_stem_from_path(shard_path)
        if resume and not force and overlay_shard_path(output, stem).exists():
            continue
        parquet_path = data_dir / f"{stem}.parquet"
        if not parquet_path.is_file():
            raise GradientAttributionError(
                f"source parquet for {stem} not found at {parquet_path}; the "
                f"grad producer must read the SAME data the base shards were "
                f"extracted from."
            )
        parquets[stem] = parquet_path

    # --- Heavy builds (only after every cheap artifact/path preflight) ---- #
    sae = SAE.from_pretrained(
        release=resolved.release, sae_id=resolved.sae_id, device=str(device)
    )
    expected_d_sae = SAE_WIDTH_TO_NUM_FEATURES[sae_width]
    actual_d_sae = int(getattr(sae.cfg, "d_sae", -1))
    if actual_d_sae != expected_d_sae:
        raise GradientAttributionError(
            f"loaded SAE width mismatch: the feature dir / config say "
            f"sae_width={sae_width.value} ({expected_d_sae} features) but the "
            f"loaded SAE has d_sae={actual_d_sae}. Scoring this dir's shards "
            f"with a different-width SAE would persist silently wrong scores."
        )
    if model_dtype != t.float32:
        sae.to(model_dtype)
    decoder_linearity_gate(sae, expected_hook_name=expected_hook_name)

    tokenizer = resolve_tokenizer(cfg.model)
    bos_token_id = resolve_bos_token_id(tokenizer)
    if ignore_bos and bos_token_id is None:
        raise GradientAttributionError(
            "ignore_bos_features=true but the tokenizer has no bos_token_id."
        )
    sae_max_context = max_position_embeddings(
        AutoConfig.from_pretrained(EXTRACTION_MODELS[cfg.model].hf_model_id)
    )

    model = build_gradient_model(cfg.model, device, model_dtype)
    collector = GradientAttributionCollector(
        model,
        sae,
        layer=int(cfg.layer),
        mode=mode,
        min_final_segment_tokens=min_final,
        bos_token_id=bos_token_id,
        ignore_bos_features=ignore_bos,
    )

    # The manifest is written only after every preflight and constructor gate.
    # Resume keeps the existing manifest byte-for-byte; nonzero ranks never
    # write it, preserving the established DDP protocol.
    if not dry_run and rank == 0 and (force or read_manifest(output) is None):
        write_manifest(
            output,
            checksum,
            OverlayKind.GRADIENT,
            metadata={
                **descriptor,
                "segment_size": int(cfg.segment_size),
                "context_window": int(cfg.context_window),
            },
            base_shard_paths=all_shard_paths,
        )
        logger.info(f"{rank_prefix}Wrote manifest to {output}")

    logger.info(
        f"{rank_prefix}{len(shard_paths)} shards assigned "
        f"(of {len(all_shard_paths)} total)"
    )

    n_processed = 0
    n_skipped = 0
    t_start = time.monotonic()
    for i, shard_path in enumerate(shard_paths):
        stem = shard_stem_from_path(shard_path)
        if resume and not force and overlay_shard_path(output, stem).exists():
            n_skipped += 1
            logger.info(f"  [{i + 1}/{len(shard_paths)}] Skipped {stem} (exists)")
            continue
        precompute_shard(
            shard_path,
            parquets[stem],
            collector,
            tokenizer,
            context_window=int(cfg.context_window),
            sae_max_context=sae_max_context,
            segment_size=int(cfg.segment_size),
            batch_size=batch_size,
            output_dir=output,
            shard_stem=stem,
            pct_overlay_dir=pct_overlay_dir,
            dry_run=dry_run,
            zstd_level=zstd_level,
        )
        n_processed += 1
        elapsed = time.monotonic() - t_start
        logger.info(
            f"  [{i + 1}/{len(shard_paths)}] {stem} done "
            f"(processed={n_processed}, skipped={n_skipped}, "
            f"elapsed={elapsed / 60:.1f}m)"
        )

    logger.info(
        f"{rank_prefix}Done: {n_processed} processed, {n_skipped} skipped "
        f"(overlay={output}, checksum={checksum[:16]}...)"
    )
    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="precompute_gradient_attribution",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
