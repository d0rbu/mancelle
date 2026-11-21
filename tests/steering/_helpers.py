"""Shared fixtures for slice-03 audit tests.

Mirrors the canonical builders in ``tests/steering/test_validate_features.py``
so we can construct minimal valid + intentionally-corrupt shards without
re-implementing the on-disk format. Importing the fixtures here (rather than
from the existing test module) keeps the audit tests independent of any
churn in the production-test module structure (per X2's "KEEP existing"
recommendation, the existing tests are not modified by this slice).
"""

from __future__ import annotations

from pathlib import Path

import torch as t
import zstandard
from safetensors.torch import save as safetensors_save

from steering.feature_io import COMPRESSED_EXT, save_features


def make_valid_shard_data(
    n_docs: int = 3,
    segs_per_doc: int = 2,
    feats_per_seg: int = 5,
    seed: int = 0,
) -> dict:
    """Create a valid in-memory shard data dict with deterministic content.

    Dtypes match the production extraction contract
    (offsets:int64, feature_ids:int32, feature_mags:float32, doc_ids:int32).
    """
    n_segments = n_docs * segs_per_doc
    n_features_total = n_segments * feats_per_seg

    g = t.Generator().manual_seed(seed)
    offsets = [i * feats_per_seg for i in range(n_segments + 1)]
    # Use a sorted-per-segment id pattern so the on-disk delta encoding is
    # well-defined (slice-3 G1: ids are ascending within a segment).
    feature_ids_list: list[int] = []
    for _ in range(n_segments):
        seg_ids = sorted(t.randperm(1024, generator=g)[:feats_per_seg].tolist())
        feature_ids_list.extend(seg_ids)
    feature_ids = t.tensor(feature_ids_list, dtype=t.int32)
    feature_mags = (t.rand(n_features_total, generator=g, dtype=t.float32).abs() + 0.01)

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        for s in range(segs_per_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 16)
            seg_char_ends.append(s * 16 + 16)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([1000] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def save_valid_shard(tmp_path: Path, name: str = "valid", **kwargs) -> tuple[str, dict]:
    """Build + save a valid shard via the production save_features pipeline.

    Returns (path_to_compressed_safetensors, original_in_memory_dict).
    """
    data = make_valid_shard_data(**kwargs)
    path = save_features(data, tmp_path / f"{name}_features")
    return str(path), data


def save_raw_shard(tmp_path: Path, name: str, data: dict) -> str:
    """Write a .safetensors.zst file directly, bypassing dtype optimization.

    Used to construct intentionally-corrupt files for asymmetry testing.
    Tensors are stored as-is (no sort / delta-encode / dtype downcast).
    """
    path = tmp_path / f"{name}_features{COMPRESSED_EXT}"
    tensors: dict[str, t.Tensor] = {}
    metadata: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, t.Tensor):
            tensors[key] = value
        else:
            metadata[key] = str(value)
    sf_bytes = safetensors_save(tensors, metadata=metadata)
    cctx = zstandard.ZstdCompressor(level=3)
    path.write_bytes(cctx.compress(sf_bytes))
    return str(path)
