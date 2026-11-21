"""Smoke + consistency tests for the tiny_axbench_run_dir fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from safetensors.torch import load_file


def test_tiny_run_dir_has_expected_files(tiny_axbench_run_dir):
    info = tiny_axbench_run_dir
    assert isinstance(info["run_root"], Path)
    assert info["run_root"].is_dir()
    assert info["candidate_concepts_path"].is_file()
    assert info["feature_statistics_path"].is_file()
    assert len(info["validation_parquets"]) == 3
    assert len(info["feature_shards"]) == 3
    for p in info["validation_parquets"] + info["feature_shards"]:
        assert p.is_file()


def test_tiny_run_dir_candidate_concepts_count(tiny_axbench_run_dir):
    """The hand-built candidate_concepts.json has exactly 12 entries."""
    info = tiny_axbench_run_dir
    data = json.loads(info["candidate_concepts_path"].read_text())
    assert len(data) == 12
    for entry in data:
        assert {"feature_index", "layer", "sae_source", "description",
                "corpus_count", "description_source"}.issubset(set(entry))


def test_tiny_run_dir_validation_parquet_shape(tiny_axbench_run_dir):
    """Each validation parquet has 8 docs."""
    info = tiny_axbench_run_dir
    for p in info["validation_parquets"]:
        table = pq.read_table(p)
        assert table.num_rows == 8
        assert "text" in table.column_names


def test_tiny_run_dir_internal_consistency(tiny_axbench_run_dir):
    """The (parquet, feature_shard) pairs must be internally consistent
    so that downstream tests using this fixture don't silently pass on
    garbage data:

    * ``feature_ids.max() < n_features`` — feature ids cannot exceed
      the SAE feature count.
    * ``len(offsets) == n_segments + 1`` — CSR-packed offsets array
      always has ``n_segments + 1`` entries (one per segment plus a
      trailing total).
    * ``doc_ids.max() < parquet.num_rows`` — every segment's
      ``doc_id`` indexes into the parquet's text column.
    * ``segment_char_ends.max() <= len(text[doc_id])`` for every
      (segment, doc) pair — the segment cannot end past the document
      it indexes.
    """
    info = tiny_axbench_run_dir
    n_features = info["n_features"]
    for parquet_path, shard_path in zip(
        info["validation_parquets"],
        info["feature_shards"], strict=True,
    ):
        table = pq.read_table(parquet_path)
        n_rows = table.num_rows
        texts = table.column("text").to_pylist()
        shard = load_file(str(shard_path))
        feature_ids = shard["feature_ids"]
        offsets = shard["offsets"]
        doc_ids = shard["doc_ids"]
        seg_ends = shard["segment_char_ends"]

        n_segments = doc_ids.numel()
        assert (
            int(feature_ids.max().item()) < n_features
        ), (
            f"{shard_path}: feature_ids.max()={int(feature_ids.max())} "
            f"must be < n_features={n_features}"
        )
        assert offsets.numel() == n_segments + 1, (
            f"{shard_path}: len(offsets)={offsets.numel()} must be "
            f"n_segments + 1 = {n_segments + 1}"
        )
        assert int(doc_ids.max().item()) < n_rows, (
            f"{shard_path}: doc_ids.max()={int(doc_ids.max())} must be "
            f"< parquet rows={n_rows}"
        )
        assert int(doc_ids.min().item()) >= 0
        for seg_i in range(n_segments):
            doc_i = int(doc_ids[seg_i].item())
            seg_end = int(seg_ends[seg_i].item())
            assert seg_end <= len(texts[doc_i]), (
                f"{shard_path}: segment_char_ends[{seg_i}]={seg_end} > "
                f"len(text[{doc_i}])={len(texts[doc_i])}"
            )
            assert seg_end >= 0


def test_tiny_run_dir_offsets_are_monotonically_increasing(tiny_axbench_run_dir):
    """CSR-packed offsets must be monotonically non-decreasing — a
    common bug in shard generators that swaps two segments leaves the
    offsets table in a state downstream readers cannot decode."""
    info = tiny_axbench_run_dir
    for shard_path in info["feature_shards"]:
        offsets = load_file(str(shard_path))["offsets"]
        prev = -1
        for i in range(offsets.numel()):
            cur = int(offsets[i].item())
            assert cur >= prev, (
                f"{shard_path}: offsets[{i}]={cur} < offsets[{i - 1}]={prev}"
            )
            prev = cur


def test_tiny_run_dir_feature_statistics_shape(tiny_axbench_run_dir):
    """``feature_statistics.safetensors`` has a ``percentile_table`` of
    shape ``(n_features, 99)``."""
    info = tiny_axbench_run_dir
    stats = load_file(str(info["feature_statistics_path"]))
    assert "percentile_table" in stats
    table = stats["percentile_table"]
    assert tuple(table.shape) == (info["n_features"], 99)
