"""Shared fixtures for feature-shard splice tests.

``make_base_shard`` builds a CSR-valid base feature-shard dict from a real
tokenizer + real segmentation (``segment_document``) with fabricated-but-valid,
per-segment-distinct features. The features are distinct per segment so that
feature-preservation through the splice can be verified by an exact ``t.equal``
join via provenance. ``doc_token_counts`` stores ``n_full`` (the full
pre-truncation length) exactly as ``extract_features`` does.
"""

from __future__ import annotations

import torch as t

from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    batch_truncate_to_training_window,
    segment_document,
)
from steering.shard_validation import validate_shard_data


def _fabricate_fids(gi: int, n_features: int) -> t.Tensor:
    """A distinct, ascending, non-empty feature-id set for global segment ``gi``."""
    count = 1 + (gi % 4)  # 1..4 features
    base = (gi * 37) % max(1, n_features - 8 * count)
    raw = sorted({(base + k * 5) % n_features for k in range(count)})
    return t.tensor(raw, dtype=t.int32)


def _fabricate_fmags(gi: int, n: int) -> t.Tensor:
    return t.tensor([1.0 + gi * 0.01 + k * 0.1 for k in range(n)], dtype=t.float32)


def make_base_shard(
    texts: list[str],
    tokenizer,
    segment_size: int,
    *,
    context_window: int = 2048,
    sae_max_context: int | None = None,
    n_features: int = 4096,
    layer: int = 8,
    pooling: str = "max_pool",
    ignore_bos_features: bool = True,
    context_independent_features: bool = False,
) -> dict:
    """Build a CSR-valid base feature-shard dict (validated before returning)."""
    truncated = batch_truncate_to_training_window(
        list(texts), tokenizer, context_window, sae_max_context
    )
    all_fids: list[t.Tensor] = []
    all_fmags: list[t.Tensor] = []
    offsets: list[int] = [0]
    doc_ids: list[int] = []
    seg_indices: list[int] = []
    seg_positions: list[int] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    doc_token_counts: list[int] = []
    gi = 0
    for d, (kept_ids, n_full) in enumerate(truncated):
        if not kept_ids:
            raise ValueError(f"make_base_shard: doc {d} produced 0 tokens")
        segments, cs, ce = segment_document(
            kept_ids,
            tokenizer,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(segment_size),
        )
        doc_token_counts.append(n_full)  # n_full, matching extract_features
        for j, (s, _e) in enumerate(segments):
            fids = _fabricate_fids(gi, n_features)
            all_fids.append(fids)
            all_fmags.append(_fabricate_fmags(gi, len(fids)))
            offsets.append(offsets[-1] + len(fids))
            doc_ids.append(d)
            seg_indices.append(j)
            seg_positions.append(s)
            char_starts.append(cs[j])
            char_ends.append(ce[j])
            gi += 1

    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.cat(all_fids).to(t.int32),
        "feature_mags": t.cat(all_fmags).to(t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": len(texts),
        "n_segments": gi,
        "layer": layer,
        "context_window": context_window,
        "pooling": pooling,
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_independent_features": context_independent_features,
        "ignore_bos_features": ignore_bos_features,
    }
    result = validate_shard_data(shard)
    if not result.valid:
        raise AssertionError(f"make_base_shard produced an invalid shard: {result.violations}")
    return shard


def doc_seg_ranges(shard: dict) -> list[tuple[int, int]]:
    """Per-doc (seg_lo, seg_hi) global-row ranges (base shards are doc-ordered)."""
    counts = t.bincount(shard["doc_ids"].to(t.int64), minlength=int(shard["n_docs"]))
    bounds = [0, *t.cumsum(counts, 0).tolist()]
    return [(bounds[d], bounds[d + 1]) for d in range(int(shard["n_docs"]))]
