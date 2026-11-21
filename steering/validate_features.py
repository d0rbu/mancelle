"""Validation for segment-granular SAE feature shard files.

Checks CSR invariants, segment metadata consistency, and data integrity.

Usage as CLI:
    python -m steering.validate_features <path_or_dir>
"""

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig
from transformer_lens import HookedTransformer

from errors import ValidationError
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.extract_features import (
    Backend,
    PoolingMode,
    _build_accelerated_backend,
    extract_document_features,
    load_sae_and_model,
)
from steering.feature_io import (
    ExtractionBackendProvenance,
    FeatureDirectoryMismatchError,
    assert_uniform_ignore_bos,
    glob_features,
    load_features,
    read_extraction_backend_provenance,
    shard_stem_from_path,
)
from steering.feature_paths import resolve_data_dir
from steering.models import resolve_sae_max_context, resolve_tokenizer
from steering.overlay_io import (
    COMPRESSED_EXT,
    OverlayKind,
    discover_overlay,
    expected_overlay_shard_stems,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    validate_percentile_overlay_shard,
)
from steering.percentile_precompute import magnitude_to_percentile_chunked
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    segment_document,
    segment_mode_from_metadata,
    truncate_to_training_window,
    turn_segments_for_document,
)
from steering.sft_schema import legacy_sft_columns
from steering.shard_validation import (
    ValidationResult,
    validate_shard_data,
)
from steering.stats_io import (
    find_statistics_path,
    load_percentile_checksum,
    load_statistics,
)

# Public API of validate_features.py — only names defined in this module.
# Primitives (REQUIRED_FIELDS, ValidationResult, assert_csr_invariants,
# validate_shard, validate_shard_data) live in steering.shard_validation;
# import them from there directly.
__all__ = [
    "_select_spot_check_docs",
    "validate_feature_directory",
    "validate_shard_activations",
    "validate_shard_against_source",
    "validate_shard_segments",
]


def _turn_char_starts_column(
    segment_mode: SegmentMode,
    table: pa.Table,
) -> list[list[int]] | None:
    if segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
        return None
    if "turn_char_starts" not in table.schema.names:
        raise ValidationError(
            "Parquet missing 'turn_char_starts' column for turn-delimited shard"
        )
    return table.column("turn_char_starts").to_pylist()


def validate_shard_segments(data: dict) -> ValidationResult:
    """Validate segment tiling, char boundaries, and document grouping."""
    violations: list[str] = []

    if "segment_char_starts" not in data:
        violations.append("Missing required field: segment_char_starts")
    if "segment_char_ends" not in data:
        violations.append("Missing required field: segment_char_ends")
    if violations:
        return ValidationResult(valid=False, violations=violations)

    n_segments = data["n_segments"]
    n_docs = data["n_docs"]
    doc_ids = data["doc_ids"]
    seg_indices = data["segment_indices"]
    seg_positions = data["segment_positions"]
    char_starts = data["segment_char_starts"]
    char_ends = data["segment_char_ends"]
    doc_token_counts = data["doc_token_counts"]
    segment_size = data["segment_size"]
    context_window = data["context_window"]
    try:
        segment_mode = segment_mode_from_metadata(data, source="feature shard")
    except ValueError as exc:
        return ValidationResult(valid=False, violations=[str(exc)])

    actual_doc_ids = sorted({int(doc_id) for doc_id in doc_ids.tolist()})
    expected_doc_ids = list(range(n_docs))
    if actual_doc_ids != expected_doc_ids:
        violations.append(
            f"Shard document IDs {actual_doc_ids} != expected range(n_docs) "
            f"{expected_doc_ids}; every document must have at least one segment"
        )

    if n_segments == 0:
        if not (doc_token_counts > 0).all():
            violations.append("Some doc_token_counts are <= 0")
        return ValidationResult(valid=len(violations) == 0, violations=violations)

    if not (doc_token_counts > 0).all():
        bad = (doc_token_counts <= 0).nonzero(as_tuple=True)[0]
        violations.append(f"doc_token_counts <= 0 at indices: {bad.tolist()}")

    if n_segments > 1:
        diffs = doc_ids[1:].to(t.int64) - doc_ids[:-1].to(t.int64)
        backwards = (diffs < 0).nonzero(as_tuple=True)[0]
        if len(backwards) > 0:
            violations.append(
                f"doc_ids not monotonically non-decreasing at positions: "
                f"{(backwards + 1).tolist()}"
            )

    if (char_starts < 0).any():
        bad = (char_starts < 0).nonzero(as_tuple=True)[0]
        violations.append(f"Negative segment_char_starts at indices: {bad.tolist()}")

    if (char_ends < char_starts).any():
        bad = (char_ends < char_starts).nonzero(as_tuple=True)[0]
        violations.append(
            f"segment_char_ends < segment_char_starts at indices: {bad.tolist()}"
        )

    unique_doc_ids = doc_ids.unique().tolist()

    for doc_id in unique_doc_ids:
        mask = doc_ids == doc_id
        doc_seg_indices = seg_indices[mask]
        doc_positions = seg_positions[mask]
        doc_char_starts = char_starts[mask]
        doc_char_ends = char_ends[mask]
        n_segs = int(mask.sum().item())

        expected_indices = t.arange(n_segs, dtype=t.int32)
        if not t.equal(doc_seg_indices, expected_indices):
            violations.append(
                f"Doc {doc_id}: segment_indices {doc_seg_indices.tolist()} "
                f"!= expected {expected_indices.tolist()}"
            )

        if n_segs > 1:
            pos_diffs = doc_positions[1:].to(t.int64) - doc_positions[:-1].to(t.int64)
            if not (pos_diffs > 0).all():
                violations.append(
                    f"Doc {doc_id}: segment_positions not strictly increasing: "
                    f"{doc_positions.tolist()}"
                )

        if segment_mode is SegmentMode.fixed_tokens and n_segs > 1:
            p0 = int(doc_positions[0].item())
            expected_positions = t.tensor(
                [p0 + i * segment_size for i in range(n_segs)],
                dtype=t.int32,
            )
            if not t.equal(doc_positions, expected_positions):
                violations.append(
                    f"Doc {doc_id}: positions {doc_positions.tolist()} "
                    f"!= expected fixed_tokens spacing {expected_positions.tolist()}"
                )

        if doc_id < len(doc_token_counts):
            doc_tokens = int(doc_token_counts[doc_id].item())
            # Same kept-length derivation as compute_segment_lengths: a
            # non-positive context_window means "keep the whole document".
            upper_bound = min(doc_tokens, context_window) if context_window > 0 else doc_tokens
            last_pos = int(doc_positions[-1].item())
            if last_pos >= upper_bound:
                violations.append(
                    f"Doc {doc_id}: last segment position {last_pos} >= "
                    f"min(doc_tokens={doc_tokens}, context_window={context_window}) "
                    f"= {upper_bound}"
                )
            # Mode-independent per-segment length bound, derived exactly as
            # compute_segment_lengths derives lengths.
            global_indices = mask.nonzero(as_tuple=True)[0]
            for i in range(n_segs):
                position = int(doc_positions[i].item())
                end = int(doc_positions[i + 1].item()) if i + 1 < n_segs else upper_bound
                length = end - position
                if not 0 < length <= segment_size:
                    violations.append(
                        f"Doc {doc_id}: segment {int(global_indices[i])} at position "
                        f"{position} has length {length}; expected 0 < length <= "
                        f"segment_size={segment_size}"
                    )

        if n_segs > 1:
            for i in range(n_segs - 1):
                ce = int(doc_char_ends[i].item())
                cs_next = int(doc_char_starts[i + 1].item())
                if ce != cs_next:
                    violations.append(
                        f"Doc {doc_id}: char boundary tiling gap at segment {i}: "
                        f"char_end={ce} != next char_start={cs_next}"
                    )
                    break

        if n_segs > 1:
            cs_diffs = doc_char_starts[1:].to(t.int64) - doc_char_starts[:-1].to(
                t.int64
            )
            if not (cs_diffs >= 0).all():
                violations.append(
                    f"Doc {doc_id}: segment_char_starts not non-decreasing: "
                    f"{doc_char_starts.tolist()}"
                )

        first_pos = int(doc_positions[0].item())
        first_char_start = int(doc_char_starts[0].item())
        if first_pos == 0 and first_char_start != 0:
            violations.append(
                f"Doc {doc_id}: first segment position is 0 (pretraining) "
                f"but char_start is {first_char_start}, expected 0"
            )

    # Cross-doc invariants
    if n_segments > 0:
        if doc_ids.max().item() >= n_docs:
            violations.append(f"doc_id {doc_ids.max().item()} >= n_docs {n_docs}")
        if doc_ids.min().item() < 0:
            violations.append(f"Negative doc_id: {doc_ids.min().item()}")

    total_segs_accounted = sum(
        int((doc_ids == did).sum().item()) for did in unique_doc_ids
    )
    if total_segs_accounted != n_segments:
        violations.append(
            f"Total segments from doc groups ({total_segs_accounted}) "
            f"!= n_segments ({n_segments})"
        )

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def validate_shard_against_source(
    shard_data: dict,
    parquet_path: str | Path,
    tokenizer,
    context_window: int,
    segment_size: int,
    sae_max_context: int | None = None,
) -> ValidationResult:
    """Cross-reference a feature shard against its source parquet file.

    Runs on every document: checks document count and token counts, then
    rederives production segmentation and compares every boundary exactly.

    Args:
        shard_data: Loaded shard dict.
        parquet_path: Path to source parquet file.
        tokenizer: Tokenizer compatible with the extraction model.
        context_window: Max tokens per forward pass.
        segment_size: Tokens per segment.
        sae_max_context: Model's max context (n_ctx). If provided,
            effective limit is min(context_window, sae_max_context).
    """
    refined_segment_size = PositiveInt.parse(segment_size)
    violations: list[str] = []
    parquet_path = Path(parquet_path)
    try:
        segment_mode = segment_mode_from_metadata(
            shard_data,
            source=f"feature shard for {parquet_path}",
        )
    except ValueError as exc:
        return ValidationResult(valid=False, violations=[str(exc)])

    # Load parquet
    pf = pq.ParquetFile(str(parquet_path))
    table = pf.read()
    schema_names = table.schema.names

    has_text = "text" in schema_names
    legacy_cols = legacy_sft_columns(schema_names)

    if legacy_cols:
        violations.append(
            "Parquet uses a legacy SFT schema "
            f"({sorted(legacy_cols)} in {schema_names}). Feature validation "
            "now supports only full-sequence 'text' shards; rebuild SFT "
            "parquet as full_nanochat_text."
        )
        return ValidationResult(valid=False, violations=violations)

    if not has_text:
        violations.append(f"Parquet missing 'text' column. Found: {schema_names}")
        return ValidationResult(valid=False, violations=violations)

    texts = table.column("text").to_pylist()
    try:
        all_turn_char_starts = _turn_char_starts_column(segment_mode, table)
    except ValidationError as exc:
        return ValidationResult(valid=False, violations=[str(exc)])

    n_parquet_docs = table.num_rows
    n_shard_docs = shard_data["n_docs"]

    # Document count
    if n_parquet_docs != n_shard_docs:
        violations.append(
            f"Document count mismatch: parquet has {n_parquet_docs}, "
            f"shard has {n_shard_docs}"
        )
        return ValidationResult(valid=False, violations=violations)

    doc_token_counts = shard_data["doc_token_counts"]
    doc_ids = shard_data["doc_ids"]
    seg_indices = shard_data["segment_indices"]
    seg_positions = shard_data["segment_positions"]
    char_starts = shard_data["segment_char_starts"]
    char_ends = shard_data["segment_char_ends"]

    for doc_id in range(n_parquet_docs):
        text = texts[doc_id]
        if text is None:
            violations.append(
                f"Doc {doc_id}: source parquet has null text and cannot be validated"
            )
            continue

        kept_ids, n_full_tokens = truncate_to_training_window(
            text,
            tokenizer,
            context_window,
            sae_max_context,
        )

        # Token count check
        stored_count = int(doc_token_counts[doc_id].item())
        if stored_count != n_full_tokens:
            violations.append(
                f"Doc {doc_id}: token count mismatch: "
                f"re-tokenized={n_full_tokens}, stored={stored_count}"
            )
            continue  # Can't check further if token counts disagree

        if not kept_ids:
            violations.append(
                f"Doc {doc_id}: source produced zero kept tokens and zero segments"
            )
            continue

        if segment_mode is SegmentMode.fixed_tokens:
            expected_segments, expected_char_starts, expected_char_ends = (
                segment_document(
                    kept_ids,
                    tokenizer,
                    segment_mode,
                    refined_segment_size,
                )
            )
        else:
            assert all_turn_char_starts is not None
            try:
                expected_segments, expected_char_starts, expected_char_ends, _ = (
                    turn_segments_for_document(
                        kept_ids,
                        tokenizer,
                        text,
                        all_turn_char_starts[doc_id],
                        refined_segment_size,
                    )
                )
            except ValueError as exc:
                violations.append(f"Doc {doc_id}: turn segmentation failed: {exc}")
                continue
        expected_indices = list(range(len(expected_segments)))
        expected_positions = [start for start, _ in expected_segments]

        mask = doc_ids == doc_id
        doc_seg_indices = seg_indices[mask].tolist()
        doc_seg_positions = seg_positions[mask].tolist()
        doc_char_starts = char_starts[mask].tolist()
        doc_char_ends = char_ends[mask].tolist()
        n_segs = int(mask.sum().item())

        if n_segs != len(expected_segments):
            violations.append(
                f"Doc {doc_id}: segment count {n_segs} != production "
                f"segment count {len(expected_segments)}"
            )
        if doc_seg_indices != expected_indices:
            violations.append(
                f"Doc {doc_id}: segment_indices {doc_seg_indices} != production "
                f"{expected_indices}"
            )
        if doc_seg_positions != expected_positions:
            violations.append(
                f"Doc {doc_id}: segment_positions {doc_seg_positions} != production "
                f"{expected_positions}"
            )
        if doc_char_starts != expected_char_starts:
            violations.append(
                f"Doc {doc_id}: segment_char_starts {doc_char_starts} != production "
                f"{expected_char_starts}"
            )
        if doc_char_ends != expected_char_ends:
            violations.append(
                f"Doc {doc_id}: segment_char_ends {doc_char_ends} != production "
                f"{expected_char_ends}"
            )

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def _normalize_identity_value(value):
    """Return the persisted scalar for enums and leave plain values unchanged."""
    return getattr(value, "value", value)


def _validate_statistics_identity(
    stats,
    reference_meta: dict[str, object],
    n_sae_features: int | None,
    violations: list[str],
) -> None:
    """Require statistics to describe the same extraction identity as shards."""
    checks = (
        ("pooling", stats.metadata.pooling_mode, reference_meta["pooling"]),
        (
            "segment_mode",
            stats.metadata.segment_mode,
            reference_meta["segment_mode"],
        ),
        (
            "segment_size",
            stats.metadata.segment_size,
            reference_meta["segment_size"],
        ),
        (
            "context_window",
            stats.metadata.context_window,
            reference_meta["context_window"],
        ),
        ("layer", stats.metadata.layer, reference_meta["layer"]),
        ("l0_tier", stats.l0_tier, reference_meta["l0_tier"]),
        (
            "ignore_bos_features",
            stats.ignore_bos_features,
            reference_meta["ignore_bos_features"],
        ),
    )
    for field, stats_value, shard_value in checks:
        normalized_stats = _normalize_identity_value(stats_value)
        normalized_shard = _normalize_identity_value(shard_value)
        if normalized_stats != normalized_shard:
            violations.append(
                f"Statistics {field}={normalized_stats!r} != "
                f"shard {field}={normalized_shard!r}"
            )

    if (
        n_sae_features is not None
        and stats.metadata.n_sae_features != n_sae_features
    ):
        violations.append(
            f"Statistics n_sae_features={stats.metadata.n_sae_features} != "
            f"explicit n_sae_features={n_sae_features}"
        )


def _statistics_shard_domain(
    shard_totals: list[tuple[Path, int, int]],
    n_shards_processed: int,
    violations: list[str],
) -> tuple[list[tuple[Path, int, int]], str] | None:
    """Resolve the two shard domains emitted by the statistics producer."""
    n_base_shards = len(shard_totals)
    if n_shards_processed == n_base_shards:
        return shard_totals, "all base shards"
    if n_base_shards > 1 and n_shards_processed == n_base_shards - 1:
        return (
            shard_totals[:-1],
            f"all base shards except final {shard_totals[-1][0].name}",
        )

    violations.append(
        f"Statistics n_shards={n_shards_processed} does not match a supported "
        f"statistics domain for {n_base_shards} base shards (expected "
        f"{n_base_shards} for all shards"
        + (
            f" or {n_base_shards - 1} for the producer's final-shard exclusion"
            if n_base_shards > 1
            else ""
        )
        + ")"
    )
    return None


def validate_feature_directory(
    feature_dir: str | Path,
    parquet_dir: str | Path | None = None,
    tokenizer=None,
    context_window: int | None = None,
    segment_size: int | None = None,
    sae_max_context: int | None = None,
    n_sae_features: int | None = None,
) -> ValidationResult:
    """Validate all shards in a feature directory for consistency.

    Checks cross-shard metadata consistency, shard completeness,
    and optionally runs per-shard structural + source validation.

    Args:
        feature_dir: Directory containing feature shard files.
        parquet_dir: If provided, check shard-parquet correspondence
            and run source validation on every document.
        tokenizer: Required when parquet_dir is provided.
        context_window: Override; read from first shard if None.
        segment_size: Override; read from first shard if None.
        sae_max_context: Model's max context for source validation.
        n_sae_features: If provided, check that all feature IDs are in
            [0, n_sae_features). Reinstates the heredoc upper-bound check
            that was dropped when Step 0b switched to validate_features.
    """
    violations: list[str] = []
    feature_dir = Path(feature_dir)

    if parquet_dir is not None and tokenizer is None:
        raise ValueError(
            "tokenizer is required when parquet_dir is provided; source "
            "validation cannot be downgraded to stem-only correspondence"
        )

    shard_paths = glob_features(feature_dir)

    if not shard_paths:
        violations.append(f"No shard files found in {feature_dir}")
        return ValidationResult(valid=False, violations=violations)

    try:
        provenance = read_extraction_backend_provenance(shard_paths)
    except (FeatureDirectoryMismatchError, ValueError) as exc:
        return ValidationResult(valid=False, violations=[str(exc)])
    provenance.warn_if_incomplete(feature_dir)

    try:
        assert_uniform_ignore_bos(shard_paths)
    except (FeatureDirectoryMismatchError, ValueError) as exc:
        violations.append(str(exc))

    # Shard-parquet correspondence (stems only — no data needed)
    parquet_stems: set[str] | None = None
    if parquet_dir is not None:
        parquet_dir = Path(parquet_dir)
        parquet_stems = {Path(p).stem for p in sorted(parquet_dir.glob("*.parquet"))}
        shard_stems = {shard_stem_from_path(p) for p in shard_paths}

        for stem in sorted(parquet_stems - shard_stems):
            violations.append(f"Missing feature shard for parquet: {stem}")
        for stem in sorted(shard_stems - parquet_stems):
            violations.append(f"Orphan feature shard (no parquet): {stem}")

    # Process shards one at a time to avoid OOM on large directories
    CONSISTENCY_KEYS = [
        "layer",
        "segment_size",
        "context_window",
        "pooling",
        "segment_mode",
        "context_independent_features",
        "ignore_bos_features",
        "l0_tier",
    ]
    reference_meta: dict[str, object] = {}
    shard_totals: list[tuple[Path, int, int]] = []
    n_shards = len(shard_paths)
    run_source_validation = parquet_dir is not None

    logger.info(
        f"Validating {n_shards} shards (source_validation={run_source_validation})"
    )
    t0 = time.monotonic()

    for i, path in enumerate(shard_paths):
        data = load_features(str(path))

        # Per-shard structural validation
        result = validate_shard_data(data, n_sae_features=n_sae_features)
        if not result.valid:
            for v in result.violations:
                violations.append(f"[{path.name}] {v}")

        result = validate_shard_segments(data)
        if not result.valid:
            for v in result.violations:
                violations.append(f"[{path.name}] {v}")

        # Metadata consistency
        if i == 0:
            for key in CONSISTENCY_KEYS:
                reference_meta[key] = data.get(key)
            if context_window is None:
                context_window = data.get("context_window")
            if segment_size is None:
                segment_size = data.get("segment_size")
        else:
            for key in CONSISTENCY_KEYS:
                ref_val = reference_meta[key]
                cur_val = data.get(key)
                if cur_val != ref_val:
                    violations.append(
                        f"Metadata mismatch: {path.name} has {key}={cur_val}, "
                        f"but first shard has {key}={ref_val}"
                    )

        shard_totals.append(
            (path, int(data["n_docs"]), int(data["n_segments"]))
        )

        # Source validation (in same loop so we can free shard data)
        if (
            run_source_validation
            and context_window is not None
            and segment_size is not None
        ):
            stem = shard_stem_from_path(path)
            pq_path = parquet_dir / f"{stem}.parquet"
            if pq_path.exists():
                result = validate_shard_against_source(
                    data,
                    pq_path,
                    tokenizer,
                    context_window=context_window,
                    segment_size=segment_size,
                    sae_max_context=sae_max_context,
                )
                if not result.valid:
                    for v in result.violations:
                        violations.append(f"[{path.name}] {v}")

        del data

        if (i + 1) % 10 == 0 or (i + 1) == n_shards:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (n_shards - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  CPU validation: {i + 1}/{n_shards} shards "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                f"{len(violations)} violations)"
            )

    # Statistics consistency
    stats_path = find_statistics_path(feature_dir)
    if stats_path is not None:
        stats = load_statistics(stats_path)
        _validate_statistics_identity(
            stats,
            reference_meta,
            n_sae_features,
            violations,
        )
        domain = _statistics_shard_domain(
            shard_totals,
            stats.corpus.n_shards_processed,
            violations,
        )
        if domain is not None:
            domain_shards, domain_label = domain
            domain_docs = sum(n_docs for _, n_docs, _ in domain_shards)
            domain_segments = sum(
                n_segments for _, _, n_segments in domain_shards
            )
            if stats.corpus.n_docs != domain_docs:
                violations.append(
                    f"Statistics n_docs={stats.corpus.n_docs} != "
                    f"sum across {domain_label}={domain_docs}"
                )
            if stats.features.n_segments != domain_segments:
                violations.append(
                    f"Statistics n_segments={stats.features.n_segments} != "
                    f"sum across {domain_label}={domain_segments}"
                )

        # Percentile checksum validation
        try:
            _validate_percentile_checksums(
                feature_dir, stats_path, stats, shard_paths, violations,
            )
        except Exception as e:
            violations.append(f"Percentile validation error: {e}")

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def _select_spot_check_shards(shard_paths: list[Path], n_spot_check_shards: int) -> list[Path]:
    """Pick first / middle / last shards (deduplicated) for percentile spot-check.

    Replaces the original middle-only selection (slice-3 P1-4): a malformed
    shard at index 0 or N-1 was previously never inspected.

    Args:
        shard_paths: All shard files in load order.
        n_spot_check_shards: Maximum shards to inspect. ``<= 0`` returns []
            (caller short-circuits). ``>= len(shard_paths)`` returns all.

    Returns:
        Deduplicated, order-preserving list of shards to spot-check.
    """
    n = len(shard_paths)
    if n_spot_check_shards <= 0 or n == 0:
        return []
    if n_spot_check_shards >= n:
        return list(shard_paths)

    # Default coverage: first + last + middle (deduplicated, order-preserved).
    candidates: list[int] = [0]
    if n >= 2:
        candidates.append(n - 1)
    if n_spot_check_shards >= 3 and n >= 3:
        # Insert the middle so we cover both ends and the interior.
        candidates.append(n // 2)
    # Truncate to requested count while preserving the seen-order.
    seen: set[int] = set()
    picked: list[int] = []
    for idx in candidates:
        if idx not in seen and len(picked) < n_spot_check_shards:
            seen.add(idx)
            picked.append(idx)
    picked.sort()
    return [shard_paths[i] for i in picked]


def _spot_check_one_shard(
    base_shard: Path,
    overlay_shard: Path,
    pct_table: t.Tensor,
    violations: list[str],
    n_spot_check_segments: int,
    rng: random.Random,
    *,
    producer_device: t.device,
    max_abs_bin_difference: int,
) -> None:
    """Inner per-shard spot-check helper. Mutates ``violations``."""
    data = load_features(str(base_shard))

    fids = data["feature_ids"]
    fmags = data["feature_mags"]
    offsets = data["offsets"]
    n_segments = int(data["n_segments"])
    stored_pcts = validate_percentile_overlay_shard(
        overlay_shard,
        expected_numel=fids.numel(),
    )

    if n_segments == 0 or len(fids) == 0:
        del data
        return

    check_segs = rng.sample(range(n_segments), min(n_spot_check_segments, n_segments))

    for seg_i in check_segs:
        s = int(offsets[seg_i].item())
        e = int(offsets[seg_i + 1].item())
        if e <= s:
            continue

        seg_fids = fids[s:e].to(t.int64)
        seg_fmags = fmags[s:e]
        seg_stored = stored_pcts[s:e]

        ref_uint8 = magnitude_to_percentile_chunked(
            seg_fids,
            seg_fmags,
            pct_table,
            device=producer_device,
        )

        absolute_difference = (seg_stored.to(t.int16) - ref_uint8.to(t.int16)).abs()
        invalid = absolute_difference > max_abs_bin_difference
        if bool(invalid.any().item()):
            n_diff = int(invalid.sum().item())
            violations.append(
                f"[{base_shard.name}] Segment {seg_i}: {n_diff}/{e-s} percentile "
                "values don't match recomputation under the producer contract "
                f"(device={producer_device.type}, allowed_bin_difference="
                f"{max_abs_bin_difference})"
            )

    del data


def _percentile_validation_producer(
    manifest: dict,
    violations: list[str],
) -> tuple[t.device, int] | None:
    """Resolve the numerical producer contract for percentile spot-checking.

    Legacy/local manifests have no ``metadata.producer`` and retain exact CPU
    validation. Remote CUDA manifests are recomputed exactly only on the same
    runtime/GPU contract. A validator without that producer may use CPU with a
    one-bin tolerance: the direct-gather float64 lookup has a pinned legitimate
    CPU/CUDA quantization-boundary difference of exactly one uint8 bin.
    """
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict) or "producer" not in metadata:
        return t.device("cpu"), 0
    producer = metadata["producer"]
    expected_keys = {
        "device_type",
        "torch_version",
        "cuda_runtime",
        "gpu_name",
        "gpu_capability",
    }
    if not isinstance(producer, dict) or set(producer) != expected_keys:
        violations.append("Percentile manifest producer contract is malformed")
        return None
    device_type = producer["device_type"]
    torch_version = producer["torch_version"]
    if (
        device_type not in {"cpu", "cuda"}
        or not isinstance(torch_version, str)
        or not torch_version
    ):
        violations.append("Percentile manifest producer contract is malformed")
        return None
    cuda_runtime = producer["cuda_runtime"]
    gpu_name = producer["gpu_name"]
    gpu_capability = producer["gpu_capability"]
    if device_type == "cpu":
        if cuda_runtime is not None or gpu_name is not None or gpu_capability is not None:
            violations.append("Percentile manifest CPU producer contract is malformed")
            return None
        return t.device("cpu"), 0
    if (
        not isinstance(cuda_runtime, str)
        or not cuda_runtime
        or not isinstance(gpu_name, str)
        or not gpu_name
        or not isinstance(gpu_capability, list)
        or len(gpu_capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in gpu_capability
        )
    ):
        violations.append("Percentile manifest CUDA producer contract is malformed")
        return None
    same_cuda_contract = (
        t.cuda.is_available()
        and torch_version == t.__version__
        and cuda_runtime == t.version.cuda
        and gpu_name == t.cuda.get_device_name(0)
        and gpu_capability == list(t.cuda.get_device_capability(0))
    )
    if same_cuda_contract:
        return t.device("cuda:0"), 0
    logger.warning(
        "Percentile overlay was produced under a different CUDA contract; "
        "validating on CPU with the documented one-bin cross-device tolerance"
    )
    return t.device("cpu"), 1


def _validate_percentile_checksums(
    feature_dir: Path,
    stats_path: Path,
    stats,
    shard_paths: list[Path],
    violations: list[str],
    n_spot_check_docs: int = 3,
    n_spot_check_shards: int = 3,
) -> None:
    """Validate the checksum-addressed percentile overlay and spot-check values.

    Checks:
    1. Statistics file has a percentile checksum.
    2. The exact current-layout overlay has matching identity and complete
       base-shard coverage.
    3. For a deterministic subset of documents in spot-checked shards (default
       first / middle / last; controlled by ``n_spot_check_shards``),
       recomputes percentiles from the percentile table and verifies they
       match the stored uint8 values.

    Args:
        feature_dir: directory containing the immutable base shards and
            checksum-addressed overlays.
        stats_path: path to the corpus statistics file.
        stats: loaded statistics object.
        shard_paths: every shard in the directory (in load order).
        violations: mutated with any failure messages.
        n_spot_check_docs: number of segments inspected per spot-checked shard.
        n_spot_check_shards: number of shards to inspect (slice-3 P1-4 fix:
            previously hard-coded to the middle shard only). Default 3:
            covers first + last + middle.
    """
    stats_checksum = load_percentile_checksum(stats_path)
    if stats_checksum is None:
        # No precomputed percentiles — nothing to validate
        return

    logger.info(
        f"Validating precomputed percentile checksums (expected={stats_checksum[:16]}...)"
    )

    # Verify the stored checksum matches what we'd compute from the statistics
    n_sae = stats.metadata.n_sae_features
    pct_table = build_percentile_table(stats.per_feature_stats, n_sae)
    computed_checksum = compute_percentile_checksum(pct_table)

    if stats_checksum != computed_checksum:
        violations.append(
            f"Statistics percentile_checksum ({stats_checksum[:16]}...) does not match "
            f"recomputed checksum ({computed_checksum[:16]}...) — statistics file may "
            f"have been regenerated without rerunning precompute_percentiles"
        )
        return

    percentile_dir = discover_overlay(
        feature_dir,
        OverlayKind.PERCENTILE,
        stats_checksum,
        base_shard_paths=shard_paths,
        cache_dir=feature_dir,
    )
    if percentile_dir is None:
        violations.append(
            f"Percentile overlay for statistics checksum "
            f"{stats_checksum[:16]}... is missing or has a mismatched checksum "
            f"at {overlay_dir(feature_dir, OverlayKind.PERCENTILE, stats_checksum)}"
        )
        return

    manifest = read_manifest(percentile_dir)
    if (
        manifest is None
        or manifest.get("kind") != OverlayKind.PERCENTILE.value
        or manifest.get("schema_version") != 2
        or not isinstance(manifest.get("base_shard_content_hash"), str)
    ):
        violations.append(
            f"Percentile overlay {percentile_dir} must have a schema-v2 "
            "percentile manifest with a base_shard_content_hash"
        )
        return

    producer_validation = _percentile_validation_producer(manifest, violations)
    if producer_validation is None:
        return
    producer_device, max_abs_bin_difference = producer_validation

    expected_stems = set(
        expected_overlay_shard_stems(shard_paths, OverlayKind.PERCENTILE)
    )
    actual_stems = {
        path.name.removesuffix(COMPRESSED_EXT)
        for path in glob_overlay_shards(percentile_dir)
    }
    for stem in sorted(expected_stems - actual_stems):
        violations.append(
            f"Missing percentile overlay shard for base shard {stem}"
        )
    for stem in sorted(actual_stems - expected_stems):
        violations.append(
            f"Orphan percentile overlay shard without base shard {stem}"
        )

    # Every overlay payload must align structurally with its immutable base
    # shard, even when only a deterministic subset is value-spot-checked.
    for base_shard in shard_paths:
        stem = shard_stem_from_path(base_shard)
        if stem not in actual_stems:
            continue
        data = load_features(str(base_shard))
        validate_percentile_overlay_shard(
            overlay_shard_path(percentile_dir, stem),
            expected_numel=data["feature_ids"].numel(),
        )
        del data

    # Spot-check first/middle/last shards (deduplicated). Slice-3 P1-4 fix:
    # the previous middle-only selection silently passed corruption at
    # index 0 or len-1.
    rng = random.Random(42)
    spot_shards = _select_spot_check_shards(shard_paths, n_spot_check_shards)
    for base_shard in spot_shards:
        stem = shard_stem_from_path(base_shard)
        if stem not in actual_stems:
            continue
        _spot_check_one_shard(
            base_shard,
            overlay_shard_path(percentile_dir, stem),
            pct_table,
            violations,
            n_spot_check_docs,
            rng,
            producer_device=producer_device,
            max_abs_bin_difference=max_abs_bin_difference,
        )


def _select_spot_check_docs(
    doc_token_counts: t.Tensor,
    n_docs: int,
    n_spot_check: int,
) -> list[int]:
    """Select document indices for GPU spot-checking.

    Selection strategy:
    - n=0: empty list
    - n=1: [first]
    - n=2: [first, last]
    - n>=3: [first, last] + (n-2) sampled biased by doc_token_counts
    """
    if n_spot_check <= 0 or n_docs == 0:
        return []
    if n_spot_check == 1:
        return [0]
    if n_docs == 1:
        return [0]
    if n_spot_check == 2 or n_docs == 2:
        return [0, n_docs - 1]

    selected = {0, n_docs - 1}
    remaining = [i for i in range(n_docs) if i not in selected]
    n_extra = min(n_spot_check - 2, len(remaining))

    if n_extra > 0:
        weights = doc_token_counts[remaining].float()
        weights = weights.clamp(min=1.0)
        indices = t.multinomial(weights, n_extra, replacement=False)
        for idx in indices:
            selected.add(remaining[idx.item()])

    return sorted(selected)


def validate_shard_activations(
    shard_data: dict,
    parquet_path: str | Path,
    model,
    sae,
    layer: int,
    context_window: int,
    segment_size: int,
    pooling: str,
    n_spot_check: int = 0,
    hook_name: str | None = None,
    feature_count_rtol: float = 0.01,
    jaccard_threshold: float = 0.99,
) -> ValidationResult:
    """Spot-check shard by re-running documents through LLM + SAE.

    Compares extracted features against fresh extraction results.

    Because bfloat16 inference is non-deterministic across runs, features
    near the JumpReLU activation threshold may flip on/off between the
    original extraction and this re-run.  Rather than requiring an exact
    match, we allow small differences controlled by two parameters:

    Args:
        feature_count_rtol: Maximum relative difference in feature counts
            per segment before flagging a violation (default 1%).
        jaccard_threshold: Minimum Jaccard similarity (intersection / union)
            of feature ID sets per segment (default 0.99 = 99% overlap).
    """
    violations: list[str] = []
    parquet_path = Path(parquet_path)

    if n_spot_check <= 0:
        return ValidationResult(valid=True, violations=[])

    # Default hook_name to the SAE's configured hook. Validation must
    # re-extract from the SAME hook point as the original extraction; otherwise
    # features diverge (stored shards use the SAE's real hook, e.g.
    # ``blocks.8.hook_resid_post`` for GemmaScope, while
    # ``_get_sae_activations`` would otherwise default to
    # ``blocks.{layer}.hook_resid_pre`` — a different activation site
    # producing wholly different features).
    #
    # SAELens stores the hook on ``sae.cfg.metadata.hook_name``; older /
    # legacy SAEs may expose it directly on ``sae.cfg.hook_name``. Probe
    # both locations.
    if hook_name is None:
        cfg = getattr(sae, "cfg", None)
        hook_name = getattr(cfg, "hook_name", None)
        if hook_name is None:
            metadata = getattr(cfg, "metadata", None)
            hook_name = getattr(metadata, "hook_name", None)

    pf = pq.ParquetFile(str(parquet_path))
    table = pf.read()
    texts = table.column("text").to_pylist()
    try:
        segment_mode = segment_mode_from_metadata(
            shard_data,
            source=f"feature shard for {parquet_path}",
        )
    except ValueError as exc:
        return ValidationResult(valid=False, violations=[str(exc)])
    try:
        all_turn_char_starts = _turn_char_starts_column(segment_mode, table)
    except ValidationError as exc:
        return ValidationResult(valid=False, violations=[str(exc)])

    n_docs = shard_data["n_docs"]
    context_independent_flag = bool(shard_data.get("context_independent_features", False))
    ignore_bos_flag = bool(shard_data.get("ignore_bos_features", False))
    doc_token_counts = shard_data["doc_token_counts"]
    doc_ids = shard_data["doc_ids"]
    offsets = shard_data["offsets"]
    stored_feat_ids = shard_data["feature_ids"]
    stored_feat_mags = shard_data["feature_mags"]

    selected = _select_spot_check_docs(doc_token_counts, n_docs, n_spot_check)

    for doc_id in selected:
        if doc_id >= len(texts):
            violations.append(
                f"Doc {doc_id}: index out of range for parquet ({len(texts)} docs)"
            )
            continue

        text = texts[doc_id]

        try:
            feat_ids, feat_mags, _, _, _, _ = extract_document_features(
                model,
                sae,
                text,
                layer=layer,
                context_window=context_window,
                pooling=PoolingMode(pooling),
                segment_mode=segment_mode,
                segment_size=segment_size,
                hook_name=hook_name,
                context_independent_features=context_independent_flag,
                ignore_bos_features=ignore_bos_flag,
                turn_char_starts=(
                    None
                    if all_turn_char_starts is None
                    else all_turn_char_starts[doc_id]
                ),
            )
        except Exception as e:
            violations.append(f"Doc {doc_id}: extraction failed: {e}")
            continue

        mask = doc_ids == doc_id
        seg_global_indices = mask.nonzero(as_tuple=True)[0]
        n_stored_segs = len(seg_global_indices)
        n_fresh_segs = len(feat_ids)

        if n_stored_segs != n_fresh_segs:
            violations.append(
                f"Doc {doc_id}: segment count mismatch: "
                f"stored={n_stored_segs}, fresh={n_fresh_segs}"
            )
            continue

        for seg_i in range(n_stored_segs):
            global_idx = seg_global_indices[seg_i].item()
            s = int(offsets[global_idx].item())
            e = int(offsets[global_idx + 1].item())
            stored_ids = stored_feat_ids[s:e]
            stored_mags = stored_feat_mags[s:e]

            fresh_ids = feat_ids[seg_i]
            fresh_mags = feat_mags[seg_i]

            if len(stored_ids) > 0:
                stored_order = stored_ids.argsort()
                stored_ids = stored_ids[stored_order]
                stored_mags = stored_mags[stored_order]
            if len(fresh_ids) > 0:
                fresh_order = fresh_ids.argsort()
                fresh_ids = fresh_ids[fresh_order]
                fresh_mags = fresh_mags[fresh_order]

            n_stored = len(stored_ids)
            n_fresh = len(fresh_ids)
            denom = max(n_stored, n_fresh, 1)
            count_rdiff = abs(n_stored - n_fresh) / denom

            if count_rdiff > feature_count_rtol:
                violations.append(
                    f"Doc {doc_id}, seg {seg_i}: feature count mismatch: "
                    f"stored={n_stored}, fresh={n_fresh} "
                    f"(rdiff={count_rdiff:.4f} > rtol={feature_count_rtol})"
                )
                continue

            stored_set = set(stored_ids.tolist())
            fresh_set = set(fresh_ids.tolist())
            intersection = len(stored_set & fresh_set)
            union = len(stored_set | fresh_set)
            jaccard = intersection / union if union > 0 else 1.0

            if jaccard < jaccard_threshold:
                violations.append(
                    f"Doc {doc_id}, seg {seg_i}: feature ID mismatch "
                    f"(jaccard={jaccard:.4f} < threshold={jaccard_threshold})"
                )
                continue

            # Compare magnitudes only for shared features
            if intersection > 0:
                common_ids = sorted(stored_set & fresh_set)
                common_tensor = t.tensor(common_ids, dtype=stored_ids.dtype)
                stored_mask = t.isin(stored_ids, common_tensor)
                fresh_mask = t.isin(fresh_ids, common_tensor)
                s_mags = stored_mags[stored_mask]
                f_mags = fresh_mags[fresh_mask]
                # Re-sort by feature ID for alignment
                s_order = stored_ids[stored_mask].argsort()
                f_order = fresh_ids[fresh_mask].argsort()
                s_mags = s_mags[s_order]
                f_mags = f_mags[f_order]
                if not t.allclose(s_mags, f_mags, rtol=1e-2, atol=1e-5):
                    max_abs_err = (s_mags - f_mags).abs().max().item()
                    max_rel_err = (
                        ((s_mags - f_mags).abs() / f_mags.clamp(min=1e-8)).max().item()
                    )
                    violations.append(
                        f"Doc {doc_id}, seg {seg_i}: magnitude mismatch "
                        f"(max_abs_err={max_abs_err:.6f}, max_rel_err={max_rel_err:.4f})"
                    )

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def _resolve_n_spot_check(cfg: DictConfig) -> int:
    """Use zero unless a positive CLI spot-check count was explicitly requested."""
    configured = int(cfg.get("n_spot_check", 0))
    if not HydraConfig.initialized():
        return configured

    task_overrides = HydraConfig.get().overrides.task
    explicitly_overridden = any(
        override.split("=", 1)[0].lstrip("+") == "n_spot_check"
        for override in task_overrides
    )
    return configured if explicitly_overridden else 0


def _preflight_spot_check_prerequisites(
    cfg: DictConfig,
    parquet_dir: Path | None,
    n_spot_check: int,
) -> None:
    """Fail before I/O when requested activation replay cannot run."""
    if n_spot_check < 0:
        raise ValueError(
            f"n_spot_check must be non-negative, got {n_spot_check}"
        )
    if n_spot_check == 0:
        return

    prerequisites = (
        ("parquet_dir", parquet_dir),
        ("model", cfg.get("model")),
        ("sae_release", cfg.get("sae_release")),
        ("sae_id", cfg.get("sae_id")),
    )
    missing = [name for name, value in prerequisites if not value]
    if missing:
        raise ValueError(
            f"n_spot_check={n_spot_check} requires parquet_dir, model, "
            f"sae_release, and sae_id; missing: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class SpotCheckBackendSelection:
    """Replay choice and the limitations that must accompany its results."""

    backend: Backend
    summary: str
    needs_warning: bool

    def log(self) -> None:
        if self.needs_warning:
            logger.warning(self.summary)
        else:
            logger.info(self.summary)


def resolve_spot_check_backend(
    configured: str | None,
    provenance: ExtractionBackendProvenance,
) -> SpotCheckBackendSelection:
    """Metadata-first replay; an explicit config may intentionally disagree."""
    if configured is None:
        if provenance.backend is not None and provenance.unknown_shards:
            raise ValueError(
                "Mixed known/unknown extraction_backend provenance: "
                f"{provenance.unknown_shards}/{provenance.n_shards} shards are unknown. "
                "GPU spot-checks require an explicit backend=auto|nnterp|transformerlens; "
                "the known shards' backend cannot be inferred for the legacy shards."
            )
        backend = Backend(provenance.backend.value) if provenance.backend else Backend.auto
    else:
        backend = Backend(configured)
    if backend is Backend.native:
        raise ValueError(
            "validate_features spot-checks cannot replay backend=native: the native "
            "collector is layer-bound at build time. Use an explicit backend=nnterp "
            "or transformerlens for a cross-backend comparison, or n_spot_check=0."
        )
    effective = Backend.nnterp if backend is Backend.auto else backend
    mismatch = provenance.backend is not None and provenance.backend.value != effective.value
    limitations = []
    if mismatch:
        limitations.append("BACKEND MISMATCH: intentional cross-backend comparison")
    if provenance.unknown_shards:
        limitations.append("legacy backend match is UNVERIFIED")
    summary = (
        f"Spot-check backend provenance: recorded={provenance.backend or 'unknown'}, "
        f"configured={configured if configured is not None else 'null (metadata-first)'}, "
        f"effective={effective.value}, "
        f"unknown_shards={provenance.unknown_shards}/{provenance.n_shards}"
    )
    if limitations:
        summary += "; " + "; ".join(limitations)
    return SpotCheckBackendSelection(backend, summary, bool(limitations))


def attach_spot_check_backend(
    model: HookedTransformer,
    *,
    backend: Backend,
    device: str,
    model_dtype: t.dtype,
) -> str:
    """Give the GPU spot-check the SAME activation backend extraction used.

    ``extract_document_features`` forwards through ``model.nnterp_model`` (or
    ``model.native_model``) when the attribute is present and only otherwise
    through the TransformerLens model itself. Extraction attaches the backend
    its ``backend`` config selects (``auto`` = nnterp on CUDA); a spot-check
    that recomputes through TransformerLens instead compares two different
    bf16 implementations and reports their drift as a Jaccard failure (#1633).

    ``native`` is rejected: that collector is bound to one layer at build time,
    while the validator reads the layer from each shard's metadata. ``auto`` /
    ``nnterp`` need a CUDA device, exactly as in extraction: on a CPU-only host
    pass ``backend=transformerlens`` explicitly (the build raises otherwise).

    Returns the label of the backend that will run the recompute.
    """
    if backend is Backend.native:
        raise ValueError(
            "validate_features spot-checks support backend=auto|nnterp|transformerlens; "
            "the native collector is layer-bound at build time and the validator reads "
            "the layer per shard. Re-run with backend=nnterp (auto) or transformerlens."
        )
    # _build_accelerated_backend reads cfg.model / cfg.layer on the native branch
    # only, which is rejected above; the nnterp build takes the model name from
    # the loaded TL model's cfg. The cfg argument is therefore never read here.
    nn_model, _native_model = _build_accelerated_backend(
        backend,
        model,
        device,
        model_dtype,
        SimpleNamespace(model=None, layer=None),
    )
    if nn_model is not None:
        # Plain attribute, never a registered submodule: the offload below must
        # move the TL weights only, whatever type the backend wrapper has.
        object.__setattr__(model, "nnterp_model", nn_model)
        # Keep only the tokenizer/cfg of the TL model resident, as extraction does.
        model.to("cpu")
        if device.startswith("cuda"):
            t.cuda.empty_cache()
        return "nnterp"
    if model_dtype != t.float32:
        model.to(model_dtype)
    return "transformerlens"


@hydra.main(
    config_path="../configs/extraction",
    config_name="validate_features",
    version_base=None,
)
def main(cfg: DictConfig):
    feature_dir = Path(cfg.feature_dir)
    if cfg.parquet_dir:
        parquet_dir = Path(cfg.parquet_dir)
    elif getattr(cfg, "resolve_parquet_default", False):
        # Only base-phase (dataset_phase=None) is wired via
        # resolve_parquet_default today; an SFT-phase validate that opts in
        # must also set dataset_phase or it will resolve to base_data.
        parquet_dir = resolve_data_dir(
            None, getattr(cfg, "dataset_phase", None)
        )
        logger.info(
            f"resolve_parquet_default=true: using nanochat default parquet "
            f"dir {parquet_dir}"
        )
    else:
        parquet_dir = None
    n_spot_check = _resolve_n_spot_check(cfg)

    if parquet_dir is not None and not cfg.get("model"):
        raise ValueError(
            "model is required when parquet_dir is provided so a source "
            "tokenizer can be resolved"
        )
    _preflight_spot_check_prerequisites(
        cfg,
        parquet_dir,
        n_spot_check,
    )

    backend_selection = None
    if n_spot_check > 0:
        shard_paths = glob_features(feature_dir)
        if not shard_paths:
            raise ValueError(f"No shard files found in {feature_dir}")
        # Read all headers before allocating weights. Missing provenance is
        # unknown, while malformed values and conflicting known values fail.
        provenance = read_extraction_backend_provenance(shard_paths)
        backend_selection = resolve_spot_check_backend(cfg.get("backend"), provenance)
        backend_selection.log()

    # For source validation, the tokenizer comes from the model.
    # When model is specified, load via TransformerLens.
    model = None
    sae = None
    tokenizer = None
    sae_max_context = None
    spot_check_backend: str | None = None

    # Resolve model dtype for GPU spot-checks (must match extraction dtype)
    dtype_str = getattr(cfg, "model_dtype", "bfloat16")
    model_dtype = {"float32": t.float32, "bfloat16": t.bfloat16}[dtype_str]

    if cfg.model is not None:
        if backend_selection is not None:
            device = "cuda" if t.cuda.is_available() else "cpu"
            logger.info(f"Loading model + SAE on {device} (dtype={dtype_str})")
            model, sae, tokenizer = load_sae_and_model(
                model_name=cfg.model,
                layer=0,  # layer comes from shard metadata per shard
                device=device,
                sae_release=cfg.sae_release,
                sae_id=cfg.sae_id,
            )
            sae_max_context = model.cfg.n_ctx
            spot_check_backend = attach_spot_check_backend(
                model,
                backend=backend_selection.backend,
                device=device,
                model_dtype=model_dtype,
            )
            if model_dtype != t.float32:
                sae.to(model_dtype)
            logger.info(
                f"Spot-check activation backend: {spot_check_backend} "
                f"(dtype={dtype_str})"
            )
        else:
            # Model only — source validation needs just the tokenizer, not
            # the weights. resolve_tokenizer is weight-free (AutoTokenizer)
            # and is the pipeline-faithful Gemma tokenizer; fall back to a
            # full TL load only for models it doesn't know (e.g. gpt2).
            try:
                tokenizer = resolve_tokenizer(cfg.model)
            except ValueError:
                tl_model = HookedTransformer.from_pretrained(
                    cfg.model, device="cpu"
                )
                tokenizer = tl_model.tokenizer
                sae_max_context = tl_model.cfg.n_ctx
                del tl_model
            else:
                sae_max_context = resolve_sae_max_context(cfg.model)

    # CPU validation: Layers 1-3
    n_sae_features = getattr(cfg, "n_sae_features", None)
    result = validate_feature_directory(
        feature_dir,
        parquet_dir=parquet_dir,
        tokenizer=tokenizer,
        sae_max_context=sae_max_context,
        n_sae_features=n_sae_features,
    )

    # Print CPU results
    if result.valid:
        logger.info(f"[PASS] All shards in {feature_dir} valid (CPU checks)")
    else:
        logger.error(f"[FAIL] {len(result.violations)} violations found:")
        for v in result.violations:
            logger.error(f"  - {v}")

    # GPU activation spot-checks (Layer 4)
    if (
        parquet_dir is not None
        and n_spot_check > 0
        and model is not None
        and sae is not None
    ):
        shard_paths = glob_features(feature_dir)
        logger.info(
            f"Running GPU activation spot-checks "
            f"(n={n_spot_check} docs/shard, {len(shard_paths)} shards, "
            f"backend={spot_check_backend})..."
        )
        gpu_t0 = time.monotonic()
        for gi, path in enumerate(shard_paths):
            data = load_features(str(path))
            stem = shard_stem_from_path(path)
            pq_path = parquet_dir / f"{stem}.parquet"
            if not pq_path.exists():
                del data
                continue
            act_result = validate_shard_activations(
                data,
                pq_path,
                model,
                sae,
                layer=data["layer"],
                context_window=data["context_window"],
                segment_size=data["segment_size"],
                pooling=data["pooling"],
                n_spot_check=n_spot_check,
                feature_count_rtol=cfg.feature_count_rtol,
                jaccard_threshold=cfg.jaccard_threshold,
            )
            del data
            status = "PASS" if act_result.valid else "FAIL"
            logger.info(f"  [{status}] {path.name} ({gi + 1}/{len(shard_paths)})")
            for v in act_result.violations:
                logger.error(f"    - {v}")
            if not act_result.valid:
                result = ValidationResult(
                    valid=False,
                    violations=result.violations + act_result.violations,
                )
        gpu_elapsed = time.monotonic() - gpu_t0
        logger.info(f"GPU spot-checks complete in {gpu_elapsed:.0f}s")
        assert backend_selection is not None
        backend_selection.log()

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
