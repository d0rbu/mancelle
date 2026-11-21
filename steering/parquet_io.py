"""Shared parquet text reading (the extraction-order contract).

Every doc-indexed artifact in the repo (base feature shards, packed rows,
token-boundary overlays, gradient overlays) keys documents by their position
in ROW-GROUP iteration order of the source parquet shard. All consumers must
share this exact iteration — a site switching to ``pq.read_table`` or
filtering rows would silently re-index documents only there. This module is
deliberately tiny and is the only sanctioned reader.

Kept separate from :mod:`steering.feature_io` so importing the (ubiquitous)
feature helpers does not pull in pyarrow.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from beartype import beartype

from steering.refined_types import NonNegativeInt, PositiveInt


def read_parquet_texts(parquet_path: str | Path) -> list[str]:
    """All texts of a parquet shard in row-group order (= extraction order)."""
    pf = pq.ParquetFile(parquet_path)
    texts: list[str] = []
    for rg_idx in range(pf.num_row_groups):
        texts.extend(pf.read_row_group(rg_idx).column("text").to_pylist())
    return texts


def detect_source_bos_char_len_from_data(
    data: Mapping[str, Any],
    parquet_path: Path,
) -> int:
    """Detect the feature source's decoded BOS character length.

    For a document whose source-token count is below ``context_window`` (and
    is therefore fully covered by segments), the source BOS character length
    is ``last_segment_char_end - len(original_text)``. Only the first parquet
    row group is inspected so callers can make one bounded startup probe.
    """
    doc_ids = data["doc_ids"]
    seg_char_ends = data["segment_char_ends"]
    doc_token_counts = data["doc_token_counts"]
    context_window = int(data["context_window"])

    parquet_file = pq.ParquetFile(parquet_path)
    first_row_group = parquet_file.read_row_group(0)
    texts = first_row_group.column("text").to_pylist()

    n_docs_to_check = min(int(doc_token_counts.shape[0]), len(texts))
    for doc_id in range(n_docs_to_check):
        if doc_token_counts[doc_id].item() >= context_window:
            continue
        mask = doc_ids == doc_id
        last_segment_end = int(seg_char_ends[mask][-1].item())
        bos_char_len = last_segment_end - len(texts[doc_id])
        if bos_char_len >= 0:
            return bos_char_len

    raise ValueError(
        f"Could not auto-detect source BOS char length from {parquet_path}. "
        "No short document found in the first row group with a computable BOS length."
    )


_SHARD_INDEX_RE = re.compile(r"shard_(\d+)")


def shard_index_from_path(path: str | Path) -> int | None:
    """Parse the integer shard index from a ``shard_NNNNN`` filename, else None.

    Full-stem match: a name merely CONTAINING ``shard_NNNNN`` (e.g. ``data_shard_5``
    or ``shard_5_backup``) does NOT falsely parse to that index — the canonical
    shard identity is the whole stem ``shard_<digits>``. Shared single source of
    truth for extraction shard-window selection."""
    m = _SHARD_INDEX_RE.fullmatch(Path(path).stem)
    return int(m.group(1)) if m else None


@beartype
def select_shard_paths(
    parquet_paths: list[str],
    num_shards: NonNegativeInt,
    start_shard: NonNegativeInt,
    rank: NonNegativeInt,
    world_size: PositiveInt,
) -> list[str]:
    """Select shards whose index is in ``[start_shard, num_shards)``, distributed
    round-robin across DDP ranks.

    Shards are identified by the index parsed from their ``shard_NNNNN`` filename —
    the canonical shard identity that also seeds per-shard determinism and names
    feature outputs. For a contiguous ``shard_00000..`` directory this equals
    slicing ``parquet_paths[start_shard:num_shards]``; for a NON-contiguous
    directory (gaps in the numbering) it correctly selects by shard NUMBER, which
    list-position slicing would get wrong.

    Stray non-shard files (e.g. a custom or val parquet) are simply not shards and
    are excluded — one stray file cannot revert the whole selection. ONLY when NO
    path is ``shard_NNNNN`` does it fall back to the historical position-based
    window, so a fully non-shard corpus is unaffected.

    Args:
        parquet_paths: All available parquet file paths, sorted.
        num_shards: Upper bound on shard index (exclusive).
        start_shard: Lower bound on shard index (inclusive). Default 0.
        rank: DDP rank of this process.
        world_size: Total number of DDP processes.

    Returns:
        The subset of paths assigned to this rank, in ascending shard-index order.

    Raises:
        BeartypeCallHintParamViolation: If a scalar violates its refined type.
        ValueError: If rank is greater than or equal to world_size.
    """
    if rank >= world_size:
        raise ValueError(
            f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
        )
    indexed = [
        (i, p)
        for i, p in ((shard_index_from_path(p), p) for p in parquet_paths)
        if i is not None
    ]
    if indexed:
        # Canonical: select shard_NNNNN files by parsed index (correct under gaps
        # and non-zero starts). Stray non-shard files are excluded -- they are not
        # shards -- so one stray file can't revert the whole selection. Index-ordered.
        ordered = sorted(indexed, key=lambda ip: ip[0])
        selected = [p for i, p in ordered if start_shard <= i < num_shards]
    else:
        # NO shard_NNNNN files at all -> historical position-based window so a fully
        # non-shard corpus (custom/val parquets) is unaffected.
        end = min(num_shards, len(parquet_paths))
        selected = parquet_paths[start_shard:end]
    return selected[rank::world_size]
