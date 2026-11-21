"""Pure base-shard document/segment alignment checks.

The gradient-overlay producer and feature-shard splicer adapt these results to
their own return containers and exception types. This module does no I/O and
does not know either caller's error hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as t

_INTEGRAL_DTYPES = frozenset(
    {
        t.uint8,
        t.uint16,
        t.uint32,
        t.uint64,
        t.int8,
        t.int16,
        t.int32,
        t.int64,
    }
)


@dataclass(frozen=True)
class AlignmentViolation:
    """Structured reason a shard cannot be aligned to source documents."""

    code: str
    detail: str


@dataclass(frozen=True)
class CanonicalDocBlocks:
    """Canonical per-document segment counts and cumulative boundaries."""

    counts: tuple[int, ...]
    boundaries: tuple[int, ...]


def canonical_doc_blocks(
    doc_ids: t.Tensor, n_docs: int
) -> CanonicalDocBlocks | AlignmentViolation:
    """Validate canonical ascending document blocks and derive boundaries."""
    if isinstance(n_docs, bool) or not isinstance(n_docs, int) or n_docs < 0:
        return AlignmentViolation(
            "invalid_n_docs", f"n_docs must be a non-negative integer, got {n_docs!r}"
        )
    if doc_ids.ndim != 1:
        return AlignmentViolation(
            "doc_ids_rank", f"doc_ids must be one-dimensional, got shape={tuple(doc_ids.shape)}"
        )
    if doc_ids.dtype not in _INTEGRAL_DTYPES:
        return AlignmentViolation(
            "doc_ids_dtype", f"doc_ids must have an integral dtype, got {doc_ids.dtype}"
        )

    ids = doc_ids.to(t.int64)
    if ids.numel() == 0:
        if n_docs == 0:
            return CanonicalDocBlocks(counts=(), boundaries=(0,))
        return AlignmentViolation(
            "missing_doc_block",
            f"doc_ids is empty but n_docs={n_docs}; every expected document must own a nonzero block",
        )
    if n_docs == 0:
        return AlignmentViolation(
            "doc_id_out_of_range",
            f"doc_ids contains {ids.numel()} row(s) but n_docs=0",
        )

    min_id = int(ids.min())
    max_id = int(ids.max())
    if min_id < 0 or max_id >= n_docs:
        return AlignmentViolation(
            "doc_id_out_of_range",
            f"doc_ids range [{min_id}, {max_id}] is outside [0, {n_docs})",
        )

    counts_tensor = t.bincount(ids, minlength=n_docs)
    missing = (counts_tensor == 0).nonzero(as_tuple=False).flatten().tolist()
    if missing:
        return AlignmentViolation(
            "missing_doc_block",
            f"expected document ids {missing} have no segment rows",
        )

    canonical = t.repeat_interleave(
        t.arange(n_docs, dtype=t.int64, device=ids.device), counts_tensor
    )
    if not t.equal(ids, canonical):
        mismatch = int((ids != canonical).nonzero(as_tuple=False)[0])
        return AlignmentViolation(
            "noncanonical_doc_order",
            f"doc_ids are not canonical ascending repeated blocks; first mismatch at row "
            f"{mismatch}: found {int(ids[mismatch])}, expected {int(canonical[mismatch])}",
        )

    counts = tuple(int(value) for value in counts_tensor.tolist())
    boundaries = (0, *tuple(int(value) for value in counts_tensor.cumsum(0).tolist()))
    return CanonicalDocBlocks(counts=counts, boundaries=boundaries)


def verify_document_alignment(
    *,
    doc_idx: int,
    n_full_tokens: int,
    segments: list[tuple[int, int]],
    stored_doc_token_counts: t.Tensor,
    stored_segment_positions: t.Tensor,
    stored_segment_indices: t.Tensor,
    seg_lo: int,
    seg_hi: int,
) -> AlignmentViolation | None:
    """Compare one re-tokenized document with its canonical stored rows."""
    stored_full = int(stored_doc_token_counts[doc_idx])
    if n_full_tokens != stored_full:
        return AlignmentViolation(
            "token_count_drift",
            f"doc {doc_idx}: re-tokenized to {n_full_tokens} tokens but the base shard "
            f"recorded {stored_full} — tokenizer or source-data drift",
        )
    n_segs = seg_hi - seg_lo
    if len(segments) != n_segs:
        return AlignmentViolation(
            "segment_count_drift",
            f"doc {doc_idx}: derived {len(segments)} segments but the base shard stores {n_segs}",
        )
    derived_positions = [start for start, _ in segments]
    stored_positions = stored_segment_positions[seg_lo:seg_hi].tolist()
    if derived_positions != stored_positions:
        return AlignmentViolation(
            "segment_position_drift",
            f"doc {doc_idx}: segment positions drifted "
            f"(derived {derived_positions[:5]}..., stored {stored_positions[:5]}...)",
        )
    stored_indices = stored_segment_indices[seg_lo:seg_hi].tolist()
    expected_indices = list(range(n_segs))
    if stored_indices != expected_indices:
        return AlignmentViolation(
            "segment_ordinal_drift",
            f"doc {doc_idx}: segment_indices are not 0..{n_segs - 1}",
        )
    return None
