"""Build a minimal AxBench-Base output skeleton for end-to-end tests.

Produces:
- run_root/candidate_concepts.json (12 entries)
- run_root/data/validation/shard_NNNNN.parquet (3 shards × 8 docs)
- run_root/features/shard_NNNNN_features.safetensors (3 shards)
- run_root/feature_statistics.safetensors (16 features × 99 percentiles)

Sized for fast pytest execution while still exercising every code path
that branches on shard count or feature count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from safetensors.torch import save_file


def _write_candidate_concepts(path: Path) -> None:
    entries = [
        {
            "feature_index": i,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": f"fake concept {i}",
            "corpus_count": 100 + i * 10,
            "description_source": "neuronpedia",
        }
        for i in range(12)
    ]
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write_validation_parquet(path: Path, *, shard_idx: int) -> None:
    """8 docs of varied length so candidate scanning has signal."""
    docs = [
        f"shard {shard_idx} doc {d}: " + ("hello world. " * (3 + d))
        for d in range(8)
    ]
    table = pa.table({"text": docs})
    pq.write_table(table, path)


def _write_feature_shard(path: Path, *, shard_idx: int, n_features: int) -> None:
    """A CSR-packed feature shard with 8 segments matching 8 docs."""
    n_segments = 8
    # Each segment activates feature_id == shard_idx (so per-shard feature
    # coverage is non-trivial and predictable).
    feature_ids = t.full((n_segments,), shard_idx, dtype=t.int32)
    offsets = t.arange(0, n_segments + 1, dtype=t.int64)
    doc_ids = t.arange(n_segments, dtype=t.int64)
    segment_char_ends = t.tensor([20 + i * 5 for i in range(n_segments)],
                                  dtype=t.int64)
    save_file(
        {
            "feature_ids": feature_ids,
            "offsets": offsets,
            "doc_ids": doc_ids,
            "segment_char_ends": segment_char_ends,
        },
        str(path),
    )


def _write_feature_statistics(path: Path, *, n_features: int) -> None:
    """A 99-bin percentile table for n_features."""
    pct_table = t.linspace(0.0, 1.0, 99).unsqueeze(0).expand(n_features, 99)
    save_file({"percentile_table": pct_table.contiguous()}, str(path))


def build_tiny_run_dir(tmp_path: Path, *, n_features: int = 16) -> dict:
    """Materialise a complete tiny AxBench run dir under ``tmp_path``."""
    run_root = tmp_path / "axbench_run"
    (run_root / "data" / "validation").mkdir(parents=True, exist_ok=True)
    (run_root / "features").mkdir(parents=True, exist_ok=True)

    candidate_path = run_root / "candidate_concepts.json"
    _write_candidate_concepts(candidate_path)

    val_parquets = []
    for s in range(3):
        p = run_root / "data" / "validation" / f"shard_{s:05d}.parquet"
        _write_validation_parquet(p, shard_idx=s)
        val_parquets.append(p)

    feat_shards = []
    for s in range(3):
        p = run_root / "features" / f"shard_{s:05d}_features.safetensors"
        _write_feature_shard(p, shard_idx=s, n_features=n_features)
        feat_shards.append(p)

    stats_path = run_root / "feature_statistics.safetensors"
    _write_feature_statistics(stats_path, n_features=n_features)

    return {
        "run_root": run_root,
        "candidate_concepts_path": candidate_path,
        "validation_parquets": val_parquets,
        "feature_shards": feat_shards,
        "feature_statistics_path": stats_path,
        "n_features": n_features,
    }


__all__ = ["build_tiny_run_dir"]
