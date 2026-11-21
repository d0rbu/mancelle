"""Validate precomputed token-boundary overlays for correctness.

Two layers of validation are offered:

1. **Structural**: inspect only the overlay tensors and the base shard's
   (doc_id, segment_idx) layout. Catches shape/length mismatches, empty
   segments, non-monotonic ranges, negative starts, non-contiguous
   same-doc ranges, and first-segment-of-doc not starting at token 0.

2. **Deep**: re-tokenize each document from the source parquet with the
   current nanochat tokenizer, recompute per-segment boundaries via
   ``compute_doc_token_boundaries_from_tokens``, and compare element-wise
   with the overlay. Any divergence — including boundaries that were
   silently patched by a previously-removed repair path — is reported.

Both layers accumulate every issue they see (no fail-fast unless
``--fail-fast`` is set) so one pass over the data produces a complete
report.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess._common import resolve_feature_dir
from scripts.preprocess.precompute_token_boundaries import (
    _detect_source_bos_char_len,
    compute_doc_token_boundaries_from_tokens,
)
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    glob_features,
    load_features,
    read_shard_metadata,
    shard_stem_from_path,
)
from steering.overlay_io import (
    OverlayKind,
    discover_overlay,
    overlay_shard_path,
    validate_token_boundary_overlay_shard,
)
from steering.parquet_io import read_parquet_texts
from steering.segmentation import compute_tokenizer_checksum


class IssueKind(StrEnum):
    """Categorical label for a token-boundary overlay issue."""

    LENGTH_MISMATCH = "length_mismatch"
    EMPTY_SEGMENT = "empty_segment"
    NON_MONOTONIC = "non_monotonic"
    NEGATIVE_START = "negative_start"
    NON_CONTIGUOUS_WITHIN_DOC = "non_contiguous_within_doc"
    FIRST_SEGMENT_NONZERO = "first_segment_nonzero"
    RECOMPUTE_MISMATCH = "recompute_mismatch"
    RECOMPUTE_EXCEPTION = "recompute_exception"


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding.

    ``shard_stem`` identifies the shard (e.g. ``shard_00123``).
    ``doc_id``/``segment_idx`` are ``None`` for whole-shard issues such
    as length mismatch.
    """

    kind: IssueKind
    shard_stem: str
    doc_id: int | None
    segment_idx: int | None
    details: str


@dataclass
class ShardValidationResult:
    shard_stem: str
    n_segments: int
    n_docs: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_shard_structural(
    shard_stem: str,
    doc_ids: t.Tensor,
    segment_indices: t.Tensor,
    overlay_token_starts: t.Tensor,
    overlay_token_ends: t.Tensor,
) -> list[ValidationIssue]:
    """Structurally validate a single shard's overlay tensors.

    Args:
        shard_stem: Shard identifier for issue reporting.
        doc_ids: ``(n_segments,)`` int tensor, doc index per segment.
        segment_indices: ``(n_segments,)`` int tensor, segment index
            within each doc.
        overlay_token_starts: ``(n_segments,)`` int tensor from overlay.
        overlay_token_ends: ``(n_segments,)`` int tensor from overlay.

    Returns:
        List of ``ValidationIssue`` — empty when the overlay is structurally
        sound. The function never raises; every inconsistency becomes an
        issue.
    """
    issues: list[ValidationIssue] = []
    n_segments_base = int(doc_ids.numel())

    if overlay_token_starts.numel() != n_segments_base:
        return [
            ValidationIssue(
                kind=IssueKind.LENGTH_MISMATCH,
                shard_stem=shard_stem,
                doc_id=None,
                segment_idx=None,
                details=(
                    f"overlay token_starts length {overlay_token_starts.numel()} "
                    f"!= base shard n_segments {n_segments_base}"
                ),
            )
        ]
    if overlay_token_ends.numel() != n_segments_base:
        return [
            ValidationIssue(
                kind=IssueKind.LENGTH_MISMATCH,
                shard_stem=shard_stem,
                doc_id=None,
                segment_idx=None,
                details=(
                    f"overlay token_ends length {overlay_token_ends.numel()} "
                    f"!= base shard n_segments {n_segments_base}"
                ),
            )
        ]

    if n_segments_base == 0:
        return issues

    doc_ids_list = doc_ids.tolist()
    seg_indices_list = segment_indices.tolist()
    starts_list = overlay_token_starts.tolist()
    ends_list = overlay_token_ends.tolist()

    # Collect per-doc (seg_idx, row_idx) pairs so contiguity checks can
    # walk each doc's segments in seg_idx order regardless of shard layout.
    doc_rows: dict[int, list[tuple[int, int]]] = {}
    for row_idx, (doc_id, seg_idx) in enumerate(zip(doc_ids_list, seg_indices_list, strict=True)):
        doc_rows.setdefault(int(doc_id), []).append((int(seg_idx), row_idx))
    for rows in doc_rows.values():
        rows.sort(key=lambda pair: pair[0])

    for row_idx in range(n_segments_base):
        start = int(starts_list[row_idx])
        end = int(ends_list[row_idx])
        doc_id = int(doc_ids_list[row_idx])
        seg_idx = int(seg_indices_list[row_idx])

        if start < 0:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.NEGATIVE_START,
                    shard_stem=shard_stem,
                    doc_id=doc_id,
                    segment_idx=seg_idx,
                    details=f"token_start={start}",
                )
            )
        if end < start:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.NON_MONOTONIC,
                    shard_stem=shard_stem,
                    doc_id=doc_id,
                    segment_idx=seg_idx,
                    details=f"token_end={end} < token_start={start}",
                )
            )
        if end == start:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.EMPTY_SEGMENT,
                    shard_stem=shard_stem,
                    doc_id=doc_id,
                    segment_idx=seg_idx,
                    details=(
                        f"token_start={start} == token_end={end}; "
                        "segment has no tokens"
                    ),
                )
            )

    for doc_id, rows in doc_rows.items():
        if not rows:
            continue
        first_seg_idx, first_row_idx = rows[0]
        first_start = int(starts_list[first_row_idx])
        if first_seg_idx == 0 and first_start != 0:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.FIRST_SEGMENT_NONZERO,
                    shard_stem=shard_stem,
                    doc_id=doc_id,
                    segment_idx=first_seg_idx,
                    details=(
                        f"doc's first segment starts at token {first_start}, "
                        "expected 0 (BOS must be in the first segment)"
                    ),
                )
            )
        for (prev_seg, prev_row), (next_seg, next_row) in itertools.pairwise(rows):
            prev_end = int(ends_list[prev_row])
            next_start = int(starts_list[next_row])
            if next_start != prev_end:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.NON_CONTIGUOUS_WITHIN_DOC,
                        shard_stem=shard_stem,
                        doc_id=doc_id,
                        segment_idx=next_seg,
                        details=(
                            f"seg {next_seg} starts at token {next_start} "
                            f"but previous seg {prev_seg} ended at token {prev_end}"
                        ),
                    )
                )

    return issues


def validate_shard_deep(
    shard_stem: str,
    doc_ids: t.Tensor,
    segment_indices: t.Tensor,
    segment_char_starts: t.Tensor,
    segment_char_ends: t.Tensor,
    overlay_token_starts: t.Tensor,
    overlay_token_ends: t.Tensor,
    texts: list[str],
    tokenizer,
    bos_token: int,
    source_bos_char_len: int,
) -> list[ValidationIssue]:
    """Re-tokenize every doc and compare the recomputed token boundaries to
    the overlay.

    The recomputation uses the currently-installed
    ``compute_doc_token_boundaries_from_tokens``, so any divergence — including
    boundaries that a previously-removed repair path patched over — is
    reported. If the current code raises for a doc (e.g. a middle segment is
    missing), a ``RECOMPUTE_EXCEPTION`` issue is emitted instead of the
    exception propagating.

    Length mismatches are not re-checked here; callers should run
    ``validate_shard_structural`` first.

    Args:
        shard_stem: Shard identifier for issue reporting.
        doc_ids: ``(n_segments,)`` int tensor, doc index per segment.
        segment_indices: ``(n_segments,)`` int tensor, segment index
            within each doc.
        segment_char_starts: ``(n_segments,)`` int tensor, source-model
            character start offset for each segment (BOS-prepended coords).
        segment_char_ends: ``(n_segments,)`` int tensor, source-model
            character end offset for each segment (BOS-prepended coords).
        overlay_token_starts: ``(n_segments,)`` int tensor from overlay.
        overlay_token_ends: ``(n_segments,)`` int tensor from overlay.
        texts: One string per document index. ``len(texts)`` must cover
            every ``doc_id`` seen in the shard.
        tokenizer: Nanochat tokenizer instance.
        bos_token: BOS token id to prepend during encoding.
        source_bos_char_len: Char length of the source (Gemma) BOS token in
            the same coordinate frame as ``segment_char_starts``.

    Returns:
        List of ``ValidationIssue``. Empty if every segment's recomputed
        (token_start, token_end) matches the overlay exactly.
    """
    issues: list[ValidationIssue] = []
    n_segments = int(doc_ids.numel())
    if n_segments == 0:
        return issues

    doc_ids_list = doc_ids.tolist()
    seg_indices_list = segment_indices.tolist()
    char_starts_list = segment_char_starts.tolist()
    char_ends_list = segment_char_ends.tolist()
    ov_starts_list = overlay_token_starts.tolist()
    ov_ends_list = overlay_token_ends.tolist()

    # Group by doc and sort by seg_idx so the recomputed boundaries line
    # up with overlay rows regardless of in-shard ordering.
    doc_rows: dict[int, list[tuple[int, int]]] = {}
    for row_idx, (doc_id, seg_idx) in enumerate(zip(doc_ids_list, seg_indices_list, strict=True)):
        doc_rows.setdefault(int(doc_id), []).append((int(seg_idx), row_idx))
    for rows in doc_rows.values():
        rows.sort(key=lambda pair: pair[0])

    for doc_id, rows in doc_rows.items():
        doc_char_starts = [char_starts_list[row] for _, row in rows]
        doc_char_ends = [char_ends_list[row] for _, row in rows]
        text_tokens = tokenizer.encode(texts[doc_id], prepend=bos_token)
        try:
            expected_starts, expected_ends = compute_doc_token_boundaries_from_tokens(
                text_tokens,
                doc_char_starts,
                doc_char_ends,
                tokenizer,
                source_bos_char_len=source_bos_char_len,
            )
        except Exception as exc:  # noqa: BLE001 — every failure mode must be captured
            issues.append(
                ValidationIssue(
                    kind=IssueKind.RECOMPUTE_EXCEPTION,
                    shard_stem=shard_stem,
                    doc_id=doc_id,
                    segment_idx=None,
                    details=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        expected_starts_list = expected_starts.tolist()
        expected_ends_list = expected_ends.tolist()
        for local_idx, (seg_idx, row_idx) in enumerate(rows):
            actual_start = int(ov_starts_list[row_idx])
            actual_end = int(ov_ends_list[row_idx])
            exp_start = int(expected_starts_list[local_idx])
            exp_end = int(expected_ends_list[local_idx])
            if actual_start != exp_start or actual_end != exp_end:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.RECOMPUTE_MISMATCH,
                        shard_stem=shard_stem,
                        doc_id=doc_id,
                        segment_idx=seg_idx,
                        details=(
                            f"expected [{exp_start}, {exp_end}) "
                            f"got [{actual_start}, {actual_end})"
                        ),
                    )
                )

    return issues


def validate_shard(
    shard_stem: str,
    doc_ids: t.Tensor,
    segment_indices: t.Tensor,
    segment_char_starts: t.Tensor,
    segment_char_ends: t.Tensor,
    overlay_token_starts: t.Tensor,
    overlay_token_ends: t.Tensor,
    n_docs: int,
    deep: bool,
    texts: list[str] | None,
    tokenizer,
    bos_token: int | None,
    source_bos_char_len: int | None,
) -> ShardValidationResult:
    """Run structural and (optionally) deep validation for a single shard.

    Structural checks always run first; if any structural issue is found
    (e.g. a length mismatch) the deep stage is skipped, because deep
    validation assumes the overlay tensors have the same length as the
    base shard.

    ``deep=True`` requires ``texts``, ``tokenizer``, ``bos_token`` and
    ``source_bos_char_len`` to be provided; otherwise a ``ValueError``
    is raised (this is a programming error, not a data issue).
    """
    if deep and texts is None:
        raise ValueError("deep=True requires texts to be provided")

    result = ShardValidationResult(
        shard_stem=shard_stem,
        n_segments=int(doc_ids.numel()),
        n_docs=n_docs,
    )
    structural_issues = validate_shard_structural(
        shard_stem,
        doc_ids,
        segment_indices,
        overlay_token_starts,
        overlay_token_ends,
    )
    result.issues.extend(structural_issues)
    if not deep:
        return result
    # Structural failures that would invalidate deep assumptions
    # (length mismatch) short-circuit the deep stage. Other structural
    # issues are informational and don't block the recompute comparison.
    if any(i.kind is IssueKind.LENGTH_MISMATCH for i in structural_issues):
        return result
    assert texts is not None  # narrowed for the type checker
    assert bos_token is not None
    assert source_bos_char_len is not None
    deep_issues = validate_shard_deep(
        shard_stem=shard_stem,
        doc_ids=doc_ids,
        segment_indices=segment_indices,
        segment_char_starts=segment_char_starts,
        segment_char_ends=segment_char_ends,
        overlay_token_starts=overlay_token_starts,
        overlay_token_ends=overlay_token_ends,
        texts=texts,
        tokenizer=tokenizer,
        bos_token=bos_token,
        source_bos_char_len=source_bos_char_len,
    )
    result.issues.extend(deep_issues)
    return result


def iter_issues_by_kind(
    issues: Iterable[ValidationIssue],
) -> dict[IssueKind, list[ValidationIssue]]:
    """Group issues by kind for summary reporting."""
    out: dict[IssueKind, list[ValidationIssue]] = {}
    for issue in issues:
        out.setdefault(issue.kind, []).append(issue)
    return out


def _load_shard_inputs(
    shard_path: Path,
    parquet_path: Path,
) -> tuple[dict, list[str], int]:
    """Load base shard, parquet texts, and detect source BOS char length."""
    base_data = load_features(shard_path)
    all_texts = read_parquet_texts(parquet_path)
    source_bos_char_len = _detect_source_bos_char_len(shard_path, parquet_path)
    return base_data, all_texts, source_bos_char_len


def validate_shard_from_paths(
    shard_path: Path,
    parquet_path: Path,
    overlay_path: Path,
    tokenizer,
    bos_token: int,
    deep: bool,
) -> ShardValidationResult:
    """Disk-backed wrapper around :func:`validate_shard`.

    Loads the base shard, parquet texts and overlay tensors, detects the
    source BOS char length from the first few docs, and runs structural
    (and optionally deep) validation.
    """
    shard_stem = shard_stem_from_path(shard_path)
    base_data, texts, source_bos_char_len = _load_shard_inputs(shard_path, parquet_path)
    overlay_token_starts, overlay_token_ends = validate_token_boundary_overlay_shard(
        overlay_path, expected_n_segments=int(base_data[MetadataKey.N_SEGMENTS])
    )
    return validate_shard(
        shard_stem=shard_stem,
        doc_ids=base_data[TensorKey.DOC_IDS],
        segment_indices=base_data[TensorKey.SEGMENT_INDICES],
        segment_char_starts=base_data[TensorKey.SEGMENT_CHAR_STARTS],
        segment_char_ends=base_data[TensorKey.SEGMENT_CHAR_ENDS],
        overlay_token_starts=overlay_token_starts,
        overlay_token_ends=overlay_token_ends,
        n_docs=int(base_data[MetadataKey.N_DOCS]),
        deep=deep,
        texts=texts if deep else None,
        tokenizer=tokenizer,
        bos_token=bos_token if deep else None,
        source_bos_char_len=source_bos_char_len if deep else None,
    )


def _log_shard_result(result: ShardValidationResult, max_issues: int) -> None:
    if result.ok:
        logger.info(
            f"  {result.shard_stem}: OK ({result.n_segments:,} segments, "
            f"{result.n_docs:,} docs)"
        )
        return
    grouped = iter_issues_by_kind(result.issues)
    kind_summary = ", ".join(f"{k.value}={len(v)}" for k, v in grouped.items())
    logger.error(
        f"  {result.shard_stem}: {len(result.issues)} issue(s) "
        f"[{kind_summary}] (n_segments={result.n_segments:,})"
    )
    for issue in result.issues[:max_issues]:
        locator_parts = []
        if issue.doc_id is not None:
            locator_parts.append(f"doc={issue.doc_id}")
        if issue.segment_idx is not None:
            locator_parts.append(f"seg={issue.segment_idx}")
        locator = " ".join(locator_parts) or "(shard-level)"
        logger.error(f"    [{issue.kind.value}] {locator}: {issue.details}")
    if len(result.issues) > max_issues:
        logger.error(f"    ... and {len(result.issues) - max_issues} more")


def run(cfg: DictConfig) -> int:
    """Entry point for the validator. Returns an exit code.

    0 = all shards passed; 1 = one or more issues (including missing
    overlay / missing parquet / feature dir not found).
    """
    try:
        feature_dir = resolve_feature_dir(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to resolve feature dir: {exc}")
        return 1
    if not feature_dir.is_dir():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1

    logger.info("Loading nanochat tokenizer...")
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    checksum = compute_tokenizer_checksum(tokenizer)
    logger.info(f"Tokenizer checksum: {checksum[:16]}...")

    deep = bool(cfg.get("deep", True))
    shard_start = int(cfg.get("shard_start", 0))
    shard_end = int(cfg.get("shard_end", -1))
    max_shards_raw = cfg.get("max_shards", None)
    max_shards = None if max_shards_raw is None else int(max_shards_raw)
    fail_fast = bool(cfg.get("fail_fast", False))
    max_issues_per_shard = int(cfg.get("max_issues_per_shard", 20))

    overlay_root = discover_overlay(
        feature_dir, OverlayKind.SEG_BOUNDARY, checksum
    )
    if overlay_root is None:
        logger.error(
            f"No token boundary overlay matching tokenizer checksum "
            f"{checksum[:16]}... under {feature_dir / 'tok'}"
        )
        return 1
    logger.info(f"Overlay directory: {overlay_root}")

    shard_paths = glob_features(str(feature_dir))
    if not shard_paths:
        logger.error(f"No feature shards found in {feature_dir}")
        return 1

    if shard_end < 0:
        shard_end = len(shard_paths)
    selected = shard_paths[shard_start:shard_end]
    if max_shards is not None:
        selected = selected[:max_shards]
    selected = [
        path
        for path in selected
        if int(read_shard_metadata(path)[MetadataKey.N_SEGMENTS]) > 0
    ]
    logger.info(
        f"Validating {len(selected)} shard(s) "
        f"(range [{shard_start}, {shard_end}), max_shards={max_shards}, deep={deep})"
    )

    data_dir = cfg.get("data_dir", None)
    parquet_paths = list_parquet_files(data_dir)
    parquet_by_stem = {Path(p).stem: Path(p) for p in parquet_paths}

    n_shards_ok = 0
    n_shards_failed = 0
    n_missing_overlay = 0
    n_missing_parquet = 0
    total_issues = 0

    for shard_path_str in selected:
        shard_path = Path(shard_path_str)
        stem = shard_stem_from_path(shard_path)
        overlay_path = overlay_shard_path(overlay_root, stem)
        if not overlay_path.exists():
            logger.error(f"  {stem}: overlay shard missing at {overlay_path}")
            n_missing_overlay += 1
            if fail_fast:
                break
            continue
        parquet_path = parquet_by_stem.get(stem)
        if deep and parquet_path is None:
            logger.error(f"  {stem}: no parquet file found for deep validation")
            n_missing_parquet += 1
            if fail_fast:
                break
            continue
        try:
            if parquet_path is not None:
                result = validate_shard_from_paths(
                    shard_path=shard_path,
                    parquet_path=parquet_path,
                    overlay_path=overlay_path,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    deep=deep,
                )
            else:
                base_data = load_features(shard_path)
                overlay_token_starts, overlay_token_ends = (
                    validate_token_boundary_overlay_shard(
                        overlay_path,
                        expected_n_segments=int(base_data[MetadataKey.N_SEGMENTS]),
                    )
                )
                result = validate_shard(
                    shard_stem=stem,
                    doc_ids=base_data[TensorKey.DOC_IDS],
                    segment_indices=base_data[TensorKey.SEGMENT_INDICES],
                    segment_char_starts=base_data[TensorKey.SEGMENT_CHAR_STARTS],
                    segment_char_ends=base_data[TensorKey.SEGMENT_CHAR_ENDS],
                    overlay_token_starts=overlay_token_starts,
                    overlay_token_ends=overlay_token_ends,
                    n_docs=int(base_data[MetadataKey.N_DOCS]),
                    deep=False,
                    texts=None,
                    tokenizer=None,
                    bos_token=None,
                    source_bos_char_len=None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"  {stem}: validator crashed: {exc}")
            n_shards_failed += 1
            if fail_fast:
                break
            continue

        _log_shard_result(result, max_issues_per_shard)
        if result.ok:
            n_shards_ok += 1
        else:
            n_shards_failed += 1
            total_issues += len(result.issues)
            if fail_fast:
                break

    logger.info("=" * 60)
    logger.info(
        f"Summary: ok={n_shards_ok} failed={n_shards_failed} "
        f"missing_overlay={n_missing_overlay} missing_parquet={n_missing_parquet} "
        f"total_issues={total_issues}"
    )
    if n_shards_failed or n_missing_overlay or n_missing_parquet:
        return 1
    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="validate_token_boundary_overlays",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
