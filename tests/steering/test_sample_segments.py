"""Tests for viz.sample_segments interpretability sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
import yaml

from steering.extract_features import ShardData
from steering.feature_io import save_features
from viz.sample_segments import (
    find_eligible_segments,
    load_shard,
    read_doc_texts,
    sample_segments,
    write_yaml,
)

# ---------------------------------------------------------------------------
# Helpers: build synthetic shard data
# ---------------------------------------------------------------------------


def _make_shard(
    n_docs: int = 5,
    segments_per_doc: int = 4,
    features_per_segment: int = 10,
    segment_size: int = 128,
    context_window: int = 1024,
) -> ShardData:
    """Build a synthetic ShardData for testing."""
    n_segments = n_docs * segments_per_doc
    n_features_total = n_segments * features_per_segment

    # CSR offsets: uniform features per segment
    offsets = t.arange(0, n_features_total + 1, features_per_segment, dtype=t.int64)

    # Feature IDs and magnitudes (sorted descending per segment)
    feature_ids_list = []
    feature_mags_list = []
    for _ in range(n_segments):
        ids = t.randint(0, 24576, (features_per_segment,), dtype=t.int32)
        mags = t.rand(features_per_segment, dtype=t.float32) + 0.01
        sort_idx = mags.argsort(descending=True)
        feature_ids_list.append(ids[sort_idx])
        feature_mags_list.append(mags[sort_idx])

    # Doc metadata
    doc_ids_list = []
    seg_indices_list = []
    seg_positions_list = []
    for doc_id in range(n_docs):
        for seg_idx in range(segments_per_doc):
            doc_ids_list.append(doc_id)
            seg_indices_list.append(seg_idx)
            seg_positions_list.append(seg_idx * segment_size)

    # Character boundaries (approximate: segment_size tokens ~= segment_size * 4 chars)
    seg_char_starts = t.tensor(
        [pos * 4 for pos in seg_positions_list],
        dtype=t.int32,
    )
    seg_char_ends = t.tensor(
        [(pos + segment_size) * 4 for pos in seg_positions_list],
        dtype=t.int32,
    )

    return ShardData(
        offsets=offsets,
        feature_ids=t.cat(feature_ids_list),
        feature_mags=t.cat(feature_mags_list),
        doc_ids=t.tensor(doc_ids_list, dtype=t.int32),
        segment_indices=t.tensor(seg_indices_list, dtype=t.int32),
        segment_positions=t.tensor(seg_positions_list, dtype=t.int32),
        doc_token_counts=t.full((n_docs,), context_window, dtype=t.int32),
        n_docs=n_docs,
        n_segments=n_segments,
        layer=8,
        context_window=context_window,
        pooling="mean_pool",
        segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
        segment_size=segment_size,
        segment_char_starts=seg_char_starts,
        segment_char_ends=seg_char_ends,
    )


def _save_shard(shard: ShardData, path: Path) -> Path:
    """Save shard to disk using compressed format."""
    return save_features(shard.to_dict(), str(path))


# ---------------------------------------------------------------------------
# Tests: find_eligible_segments
# ---------------------------------------------------------------------------


class TestFindEligibleSegments:
    def test_excludes_last_segment_per_doc(self):
        shard = _make_shard(n_docs=3, segments_per_doc=4)
        eligible = find_eligible_segments(shard)

        # 3 docs x 4 segments = 12 total, 3 last segments excluded => 9
        assert len(eligible) == 9

        # Verify none of the eligible are the last segment (index 3)
        for idx in eligible:
            assert shard.segment_indices[idx].item() < 3

    def test_single_segment_doc_has_no_eligible(self):
        shard = _make_shard(n_docs=2, segments_per_doc=1)
        eligible = find_eligible_segments(shard)
        assert len(eligible) == 0

    def test_two_segment_doc(self):
        shard = _make_shard(n_docs=2, segments_per_doc=2)
        eligible = find_eligible_segments(shard)
        # 2 docs x 1 eligible each = 2
        assert len(eligible) == 2
        for idx in eligible:
            assert shard.segment_indices[idx].item() == 0

    def test_returns_global_indices(self):
        shard = _make_shard(n_docs=3, segments_per_doc=4)
        eligible = find_eligible_segments(shard)

        # All returned indices should be valid global indices
        for idx in eligible:
            assert 0 <= idx < shard.n_segments


# ---------------------------------------------------------------------------
# Tests: load_shard round-trip
# ---------------------------------------------------------------------------


class TestLoadShard:
    def test_round_trip(self, tmp_path: Path):
        original = _make_shard(n_docs=2, segments_per_doc=3)
        actual_path = _save_shard(original, tmp_path / "test_features")

        loaded = load_shard(str(actual_path))
        assert loaded.n_docs == original.n_docs
        assert loaded.n_segments == original.n_segments
        assert t.equal(loaded.offsets, original.offsets)

        # After save/load, feature IDs are sorted ascending within each segment
        # and magnitudes lose some precision (float32 -> float16 -> float32).
        # Verify per-segment ID sets match and magnitudes are close.
        for i in range(loaded.n_segments):
            s, e = loaded.offsets[i].item(), loaded.offsets[i + 1].item()
            orig_ids = original.feature_ids[s:e]
            loaded_ids = loaded.feature_ids[s:e]
            assert set(orig_ids.tolist()) == set(loaded_ids.tolist())
            # Loaded IDs should be sorted ascending
            if len(loaded_ids) > 1:
                assert (loaded_ids[:-1] <= loaded_ids[1:]).all()
            # Magnitudes should be approximately equal (float16 precision)
            orig_mags = original.feature_mags[s:e]
            loaded_mags = loaded.feature_mags[s:e]
            assert t.allclose(
                orig_mags.sort().values, loaded_mags.sort().values, atol=1e-3
            )


# ---------------------------------------------------------------------------
# Tests: read_doc_texts
# ---------------------------------------------------------------------------


class TestReadDocTexts:
    def test_reads_requested_docs(self, tmp_path: Path):
        # Create a tiny parquet with a few rows
        import pyarrow as pa
        import pyarrow.parquet as pq

        texts = [f"Document number {i} with some content." for i in range(10)]
        table = pa.table({"text": texts})
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, str(parquet_path), row_group_size=3)

        result = read_doc_texts(str(parquet_path), {0, 4, 9})
        assert len(result) == 3
        assert result[0] == texts[0]
        assert result[4] == texts[4]
        assert result[9] == texts[9]

    def test_empty_request(self, tmp_path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"text": ["hello"]})
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, str(parquet_path))

        result = read_doc_texts(str(parquet_path), set())
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: write_yaml
# ---------------------------------------------------------------------------


class TestWriteYaml:
    def test_writes_valid_yaml(self, tmp_path: Path):
        data = {
            "metadata": {"n_samples": 1},
            "samples": [
                {
                    "context": "Line one.\nLine two.\nLine three.",
                    "segment": "Short text.",
                    "features": [{"id": 42, "magnitude": 1.5}],
                }
            ],
        }
        out_path = tmp_path / "test.yaml"
        write_yaml(data, out_path)

        loaded = yaml.safe_load(out_path.read_text())
        assert loaded["metadata"]["n_samples"] == 1
        assert loaded["samples"][0]["features"][0]["id"] == 42

    def test_multiline_strings_use_block_style(self, tmp_path: Path):
        data = {"text": "line 1\nline 2\nline 3"}
        out_path = tmp_path / "test.yaml"
        write_yaml(data, out_path)

        raw = out_path.read_text()
        # Block scalar indicator should be present
        assert "|" in raw


# ---------------------------------------------------------------------------
# Tests: sample_segments (integration with synthetic data)
# ---------------------------------------------------------------------------


class TestSampleSegments:
    @pytest.fixture()
    def synthetic_env(self, tmp_path: Path):
        """Set up a synthetic shard + parquet pair for testing."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        n_docs = 5
        segments_per_doc = 8  # 1024 / 128 = 8 segments
        segment_size = 128
        context_window = 1024

        # Generate documents long enough to produce full tails
        # GPT-2 tokenizer: ~1 token per 4 chars, so 6000 chars => ~1500 tokens
        texts = [f"This is document {i}. " * 300 for i in range(n_docs)]
        table = pa.table({"text": texts})
        parquet_path = tmp_path / "shard_00000.parquet"
        pq.write_table(table, str(parquet_path))

        shard = _make_shard(
            n_docs=n_docs,
            segments_per_doc=segments_per_doc,
            features_per_segment=20,
            segment_size=segment_size,
            context_window=context_window,
        )
        actual_path = _save_shard(shard, tmp_path / "shard_00000_features")

        return {
            "shard_path": str(actual_path),
            "parquet_path": str(parquet_path),
            "n_docs": n_docs,
            "segments_per_doc": segments_per_doc,
            "context_window": context_window,
        }

    def test_returns_correct_structure(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=5,
            top_k=10,
            seed=42,
        )

        assert "metadata" in result
        assert "samples" in result
        assert result["metadata"]["n_samples"] == len(result["samples"])
        assert result["metadata"]["top_k_features"] == 10

    def test_samples_have_required_fields(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=3,
            top_k=5,
            seed=42,
        )

        required_fields = {
            "global_segment_id",
            "doc_id",
            "segment_index",
            "n_segments_in_doc",
            "segment_token_position",
            "context_tokens",
            "segment_tokens",
            "context",
            "segment",
            "next_segment",
            "n_active_features",
            "features",
        }
        for sample in result["samples"]:
            assert required_fields.issubset(sample.keys())

    def test_features_limited_by_top_k(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=3,
            top_k=5,
            seed=42,
        )

        for sample in result["samples"]:
            assert len(sample["features"]) <= 5

    def test_features_sorted_descending(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=3,
            top_k=10,
            seed=42,
        )

        for sample in result["samples"]:
            mags = [f["magnitude"] for f in sample["features"]]
            assert mags == sorted(mags, reverse=True)

    def test_segment_not_last_in_doc(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=10,
            top_k=5,
            seed=42,
        )

        for sample in result["samples"]:
            # segment_index should be less than (n_segments_in_doc - 1)
            assert sample["segment_index"] < sample["n_segments_in_doc"] - 1

    def test_context_is_nonempty_for_non_first_segments(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=20,
            top_k=5,
            seed=42,
        )

        for sample in result["samples"]:
            if sample["segment_index"] > 0:
                assert len(sample["context"]) > 0

    def test_deterministic_with_same_seed(self, synthetic_env):
        kwargs = {
            "shard_path": synthetic_env["shard_path"],
            "parquet_path": synthetic_env["parquet_path"],
            "n_samples": 5,
            "top_k": 5,
            "seed": 123,
        }
        r1 = sample_segments(**kwargs)
        r2 = sample_segments(**kwargs)

        # Same global segment IDs sampled
        ids1 = [s["global_segment_id"] for s in r1["samples"]]
        ids2 = [s["global_segment_id"] for s in r2["samples"]]
        assert ids1 == ids2

    def test_different_seeds_give_different_samples(self, synthetic_env):
        base = {
            "shard_path": synthetic_env["shard_path"],
            "parquet_path": synthetic_env["parquet_path"],
            "n_samples": 10,
            "top_k": 5,
        }
        r1 = sample_segments(**base, seed=1)
        r2 = sample_segments(**base, seed=2)

        ids1 = {s["global_segment_id"] for s in r1["samples"]}
        ids2 = {s["global_segment_id"] for s in r2["samples"]}
        # Very unlikely to be identical with different seeds
        assert ids1 != ids2

    def test_n_samples_capped_at_eligible(self, synthetic_env):
        result = sample_segments(
            shard_path=synthetic_env["shard_path"],
            parquet_path=synthetic_env["parquet_path"],
            n_samples=99999,
            top_k=5,
            seed=42,
        )
        # 5 docs x 8 segments, 5 last excluded => 35 eligible
        assert len(result["samples"]) <= 35
