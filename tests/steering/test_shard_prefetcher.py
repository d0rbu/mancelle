"""Tests for shard-level prefetching: PrefetchedShard and prefetch_shard_data.

Verifies that:
1. prefetch_shard_data produces identical results to inline loading
2. process_shard with prefetched data produces bitwise-identical output
3. Chunk boundary correctness with uneven doc counts
4. Edge cases: empty shards, single-doc shards
5. Default inline processing (prefetched=None)

Run with:
    uv run pytest tests/test_shard_prefetcher.py -v --model-scale=tiny
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import (
    PoolingMode,
    prefetch_shard_data,
    process_shard,
)
from steering.feature_io import load_features

# Re-use tiny model fixtures from integration tests
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning transforms data into predictions.",
    "Quantum computing harnesses quantum mechanical phenomena.",
    "The Pythagorean theorem relates the sides of right triangles.",
    "Photosynthesis converts light energy into chemical energy.",
    "DNA stores genetic information in sequences of nucleotides.",
    "Neural networks learn hierarchical representations of data.",
    "The speed of light in vacuum is approximately 300000 km per second.",
    "Algorithms provide step-by-step procedures for calculations.",
    "Entropy measures the disorder or randomness in a system.",
]


def _make_parquet(path: str, texts: list[str]) -> str:
    """Create a minimal parquet file with a text column."""
    table = pa.table({"text": texts})
    pq.write_table(table, path)
    return path


def _make_parquet_with_aco(
    path: str, texts: list[str], aco_list: list[int]
) -> str:
    """Create a legacy SFT parquet file with assistant_char_offset columns."""
    table = pa.table({"text": texts, "assistant_char_offset": aco_list})
    pq.write_table(table, path)
    return path


@pytest.fixture(scope="module")
def tiny_model():
    return TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")


@pytest.fixture(scope="module")
def tiny_sae():
    return TinySAE(d_model=64, n_features=256, device="cpu")


@pytest.fixture(scope="module")
def layer():
    return 8


# ---------------------------------------------------------------------------
# 1. prefetch_shard_data unit test — exact text content and chunking
# ---------------------------------------------------------------------------


class TestPrefetchShardDataUnit:
    def test_prefetched_chunks_contain_exact_expected_texts(self, tmp_path):
        """Verify all_text_chunks contains exactly the expected texts in order."""
        path = _make_parquet(str(tmp_path / "test.parquet"), SAMPLE_TEXTS)
        result = prefetch_shard_data(path, cpu_batch_size=4)

        # With 10 docs and batch_size=4: chunks of [4, 4, 2]
        assert len(result.all_text_chunks) == 3
        assert result.all_text_chunks[0] == SAMPLE_TEXTS[0:4]
        assert result.all_text_chunks[1] == SAMPLE_TEXTS[4:8]
        assert result.all_text_chunks[2] == SAMPLE_TEXTS[8:10]

    def test_prefetched_metadata_is_correct(self, tmp_path):
        """Verify PrefetchedShard metadata fields."""
        path = _make_parquet(str(tmp_path / "meta.parquet"), SAMPLE_TEXTS[:5])
        result = prefetch_shard_data(path, cpu_batch_size=3)

        assert result.parquet_path == path
        assert result.shard_name == "meta"
        assert result.num_row_groups >= 1
        assert result.memory_bytes == sum(len(text) for text in SAMPLE_TEXTS[:5])

    def test_chunk_size_equals_one(self, tmp_path):
        """Each chunk should contain exactly 1 doc when cpu_batch_size=1."""
        texts = SAMPLE_TEXTS[:3]
        path = _make_parquet(str(tmp_path / "bs1.parquet"), texts)
        result = prefetch_shard_data(path, cpu_batch_size=1)

        assert len(result.all_text_chunks) == 3
        for i, chunk in enumerate(result.all_text_chunks):
            assert chunk == [texts[i]]

    def test_chunk_size_exceeds_doc_count(self, tmp_path):
        """Single chunk when cpu_batch_size > number of docs."""
        texts = SAMPLE_TEXTS[:3]
        path = _make_parquet(str(tmp_path / "big_bs.parquet"), texts)
        result = prefetch_shard_data(path, cpu_batch_size=100)

        assert len(result.all_text_chunks) == 1
        assert result.all_text_chunks[0] == texts


# ---------------------------------------------------------------------------
# 2. Prefetched data matches inline loading exactly
# ---------------------------------------------------------------------------


class TestPrefetchMatchesInlineLoading:
    def test_prefetched_data_matches_inline_loading_exactly(self, tmp_path):
        """prefetch_shard_data must produce identical chunks to inline parquet reading."""
        texts = SAMPLE_TEXTS
        path = _make_parquet(str(tmp_path / "parity.parquet"), texts)
        cpu_batch_size = 3

        # Prefetch path
        prefetched = prefetch_shard_data(path, cpu_batch_size=cpu_batch_size)

        # Inline path (replicate Phase 1 of process_shard exactly)
        pf = pq.ParquetFile(path)
        inline_text_chunks: list[list[str]] = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rg_texts = rg.column("text").to_pylist()
            for chunk_start in range(0, len(rg_texts), cpu_batch_size):
                inline_text_chunks.append(
                    rg_texts[chunk_start : chunk_start + cpu_batch_size]
                )

        # Exact match: same number of chunks
        assert len(prefetched.all_text_chunks) == len(inline_text_chunks)

        # Exact match: same texts in same order in each chunk
        for i in range(len(inline_text_chunks)):
            assert prefetched.all_text_chunks[i] == inline_text_chunks[i], (
                f"Chunk {i} text mismatch: "
                f"prefetched={prefetched.all_text_chunks[i]} "
                f"inline={inline_text_chunks[i]}"
            )

    def test_sft_prefetch_rejects_legacy_aco_column(self, tmp_path):
        """SFT extraction rejects legacy assistant-offset parquet shards."""
        texts = SAMPLE_TEXTS[:4]
        aco_list = [5, 10, 15, 20]
        path = _make_parquet_with_aco(
            str(tmp_path / "aco.parquet"), texts, aco_list
        )

        with pytest.raises(ValueError, match="legacy columns"):
            prefetch_shard_data(path, cpu_batch_size=2, dataset_phase="sft")

    def test_sft_prefetch_rejects_legacy_loss_mask_column(self, tmp_path):
        """SFT extraction rejects legacy assistant loss-mask parquet shards."""
        path = str(tmp_path / "loss_mask.parquet")
        pq.write_table(
            pa.table({
                "text": SAMPLE_TEXTS[:2],
                "assistant_loss_mask": [0, 1],
            }),
            path,
        )

        with pytest.raises(ValueError, match="legacy columns"):
            prefetch_shard_data(path, cpu_batch_size=2, dataset_phase="sft")

    def test_prefetch_rejects_legacy_aco_column_for_all_phases(self, tmp_path):
        """Legacy SFT columns are not accepted as generic text metadata."""
        texts = SAMPLE_TEXTS[:2]
        path = _make_parquet_with_aco(str(tmp_path / "aco.parquet"), texts, [5, 10])

        with pytest.raises(ValueError, match="removed SFT legacy columns"):
            prefetch_shard_data(path, cpu_batch_size=2, dataset_phase=None)


# ---------------------------------------------------------------------------
# 3. process_shard output parity (bitwise identical)
# ---------------------------------------------------------------------------


class TestProcessShardOutputParity:
    """Gold standard: process_shard with prefetched data == inline loading."""

    def test_prefetched_output_is_bitwise_identical_to_inline(
        self, tiny_model, tiny_sae, layer, tmp_path
    ):
        """process_shard(prefetched=data) must produce bitwise-identical tensors
        to process_shard(prefetched=None)."""
        texts = [
            "This is document one about mathematics and algebra.",
            "Document two discusses natural language processing techniques.",
            "The third document covers physics and quantum mechanics.",
            "Fourth document is about cooking recipes and ingredients.",
            "Fifth document describes machine learning algorithms.",
        ]
        path = _make_parquet(str(tmp_path / "parity.parquet"), texts)
        cpu_batch_size = 2

        # Inline path
        result_inline = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "inline"),
            context_window=1024,
            pooling=PoolingMode.mean_pool,
            cpu_batch_size=cpu_batch_size,
        )

        # Prefetched path
        prefetched = prefetch_shard_data(path, cpu_batch_size=cpu_batch_size)
        result_prefetched = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "prefetched"),
            context_window=1024,
            pooling=PoolingMode.mean_pool,
            cpu_batch_size=cpu_batch_size,
            prefetched=prefetched,
        )

        # Compare ShardResult scalars
        assert result_inline.n_docs == result_prefetched.n_docs
        assert result_inline.n_tokens_gpt2 == result_prefetched.n_tokens_gpt2
        assert result_inline.n_total_features == result_prefetched.n_total_features

        # Load saved tensors and compare bitwise
        data_inline = load_features(result_inline.output_path)
        data_prefetched = load_features(result_prefetched.output_path)

        for key in (
            "offsets",
            "feature_ids",
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "doc_token_counts",
            "segment_char_starts",
            "segment_char_ends",
        ):
            assert t.equal(data_inline[key], data_prefetched[key]), (
                f"{key}: tensors differ. "
                f"inline shape={data_inline[key].shape}, "
                f"prefetched shape={data_prefetched[key].shape}"
            )

        assert t.equal(data_inline["feature_mags"], data_prefetched["feature_mags"]), (
            "feature_mags: tensors differ (bitwise). "
            f"max diff={(data_inline['feature_mags'] - data_prefetched['feature_mags']).abs().max()}"
        )

        for key in ("n_docs", "n_segments", "layer", "context_window", "pooling"):
            assert data_inline[key] == data_prefetched[key], (
                f"{key}: {data_inline[key]} != {data_prefetched[key]}"
            )

    @pytest.mark.parametrize("pooling_mode", list(PoolingMode))
    def test_parity_across_all_pooling_modes(
        self, tiny_model, tiny_sae, layer, tmp_path, pooling_mode
    ):
        """Output parity holds for all pooling modes."""
        texts = ["Hello world of testing.", "Another test document here."]
        path = _make_parquet(str(tmp_path / f"pool_{pooling_mode.value}.parquet"), texts)

        result_inline = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "inline"),
            context_window=1024,
            pooling=pooling_mode,
        )

        prefetched = prefetch_shard_data(path, cpu_batch_size=64)
        result_prefetched = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "prefetched"),
            context_window=1024,
            pooling=pooling_mode,
            prefetched=prefetched,
        )

        data_inline = load_features(result_inline.output_path)
        data_prefetched = load_features(result_prefetched.output_path)
        for key in ("offsets", "feature_ids", "feature_mags", "doc_ids"):
            assert t.equal(data_inline[key], data_prefetched[key]), (
                f"{key} differs for pooling={pooling_mode.value}"
            )


# ---------------------------------------------------------------------------
# 4. SFT format parity
# ---------------------------------------------------------------------------


class TestSftFormatPrefetchParity:
    """Verify SFT prefetching uses the full-sequence text parquet contract."""

    def _make_sft_parquet(self, path: str, texts: list[str]) -> str:
        """Create full-sequence SFT parquet with only rendered text."""
        table = pa.table({"text": texts, "original_index": list(range(len(texts)))})
        pq.write_table(table, path)
        return path

    def test_sft_prefetch_chunks_match_inline(self, tmp_path):
        """SFT parquet chunks are read exactly from the rendered text column."""
        texts = [
            "<|user|>What is 2+2?<|assistant|>The answer is 4.",
            "<|user|>Tell me about Python.<|assistant|>Python is a programming language.",
            "<|user|>What is ML?<|assistant|>Machine learning is a subfield of AI.",
        ]
        path = self._make_sft_parquet(str(tmp_path / "sft.parquet"), texts)
        cpu_batch_size = 2

        # Prefetch path
        prefetched = prefetch_shard_data(
            path,
            cpu_batch_size=cpu_batch_size,
            dataset_phase="sft",
        )

        pf = pq.ParquetFile(path)
        inline_text_chunks: list[list[str]] = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rg_texts = rg.column("text").to_pylist()
            for chunk_start in range(0, len(rg_texts), cpu_batch_size):
                inline_text_chunks.append(
                    rg_texts[chunk_start : chunk_start + cpu_batch_size]
                )

        # Verify exact parity
        assert len(prefetched.all_text_chunks) == len(inline_text_chunks)
        for i in range(len(inline_text_chunks)):
            assert prefetched.all_text_chunks[i] == inline_text_chunks[i], (
                f"SFT chunk {i} text mismatch"
            )

    def test_sft_prefetch_rejects_messages_response_schema(self, tmp_path):
        """Old messages/response shards must be rebuilt before extraction."""
        table = pa.table(
            {
                "messages": ['[{"role": "user", "content": "Hi"}]'],
                "response": ["Hello."],
            }
        )
        path = str(tmp_path / "legacy_sft.parquet")
        pq.write_table(table, path)

        with pytest.raises(ValueError, match="removed SFT legacy columns"):
            prefetch_shard_data(path, cpu_batch_size=64, dataset_phase="sft")

    def test_sft_prefetch_rejects_text_plus_messages_response_schema(self, tmp_path):
        """A text column does not make mixed legacy SFT schemas acceptable."""
        table = pa.table(
            {
                "text": ["rendered"],
                "messages": ['[{"role": "user", "content": "Hi"}]'],
                "response": ["Hello."],
            }
        )
        path = str(tmp_path / "mixed_legacy_sft.parquet")
        pq.write_table(table, path)

        with pytest.raises(ValueError, match="legacy columns"):
            prefetch_shard_data(path, cpu_batch_size=64, dataset_phase="sft")


# ---------------------------------------------------------------------------
# 5. Chunk boundary correctness with uneven division
# ---------------------------------------------------------------------------


class TestChunkBoundaryCorrectness:
    def test_uneven_division_last_chunk_has_correct_remainder(self, tmp_path):
        """10 docs with cpu_batch_size=3: chunks of [3, 3, 3, 1]."""
        path = _make_parquet(str(tmp_path / "uneven.parquet"), SAMPLE_TEXTS)
        prefetched = prefetch_shard_data(path, cpu_batch_size=3)

        assert len(prefetched.all_text_chunks) == 4
        assert len(prefetched.all_text_chunks[0]) == 3
        assert len(prefetched.all_text_chunks[1]) == 3
        assert len(prefetched.all_text_chunks[2]) == 3
        assert len(prefetched.all_text_chunks[3]) == 1

        # Verify exact content of last chunk
        assert prefetched.all_text_chunks[3] == [SAMPLE_TEXTS[9]]

        # Compare against inline loading
        pf = pq.ParquetFile(path)
        inline_chunks: list[list[str]] = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            rg_texts = rg.column("text").to_pylist()
            for chunk_start in range(0, len(rg_texts), 3):
                inline_chunks.append(rg_texts[chunk_start : chunk_start + 3])

        assert len(prefetched.all_text_chunks) == len(inline_chunks)
        for i in range(len(inline_chunks)):
            assert prefetched.all_text_chunks[i] == inline_chunks[i]

    def test_exact_division_no_remainder(self, tmp_path):
        """6 docs with cpu_batch_size=3: exactly [3, 3]."""
        texts = SAMPLE_TEXTS[:6]
        path = _make_parquet(str(tmp_path / "exact.parquet"), texts)
        prefetched = prefetch_shard_data(path, cpu_batch_size=3)

        assert len(prefetched.all_text_chunks) == 2
        assert prefetched.all_text_chunks[0] == texts[0:3]
        assert prefetched.all_text_chunks[1] == texts[3:6]


# ---------------------------------------------------------------------------
# 6. Empty shard
# ---------------------------------------------------------------------------


class TestEmptyShard:
    def test_empty_shard_prefetch_produces_empty_chunks(self, tmp_path):
        """Parquet with 0 rows produces empty chunk lists."""
        path = _make_parquet(str(tmp_path / "empty.parquet"), [])
        prefetched = prefetch_shard_data(path, cpu_batch_size=64)

        assert prefetched.all_text_chunks == []
        assert prefetched.memory_bytes == 0

    def test_empty_shard_process_shard_parity(
        self, tiny_model, tiny_sae, layer, tmp_path
    ):
        """Empty shard: inline and prefetched paths produce identical results."""
        path = _make_parquet(str(tmp_path / "empty.parquet"), [])

        result_inline = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "inline"),
            context_window=1024,
        )

        prefetched = prefetch_shard_data(path, cpu_batch_size=64)
        result_prefetched = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "prefetched"),
            context_window=1024,
            prefetched=prefetched,
        )

        assert result_inline.n_docs == result_prefetched.n_docs == 0


# ---------------------------------------------------------------------------
# 7. Single doc shard
# ---------------------------------------------------------------------------


class TestSingleDocShard:
    def test_single_doc_prefetch_produces_one_chunk(self, tmp_path):
        """Single-doc shard produces exactly one chunk with one text."""
        path = _make_parquet(str(tmp_path / "single.parquet"), [SAMPLE_TEXTS[0]])
        prefetched = prefetch_shard_data(path, cpu_batch_size=64)

        assert len(prefetched.all_text_chunks) == 1
        assert prefetched.all_text_chunks[0] == [SAMPLE_TEXTS[0]]

    def test_single_doc_process_shard_parity(
        self, tiny_model, tiny_sae, layer, tmp_path
    ):
        """Single-doc shard: inline and prefetched paths are bitwise identical."""
        path = _make_parquet(str(tmp_path / "single.parquet"), [SAMPLE_TEXTS[0]])

        result_inline = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "inline"),
            context_window=1024,
        )

        prefetched = prefetch_shard_data(path, cpu_batch_size=64)
        result_prefetched = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path / "prefetched"),
            context_window=1024,
            prefetched=prefetched,
        )

        data_inline = load_features(result_inline.output_path)
        data_prefetched = load_features(result_prefetched.output_path)

        for key in (
            "offsets",
            "feature_ids",
            "feature_mags",
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "doc_token_counts",
            "segment_char_starts",
            "segment_char_ends",
        ):
            assert t.equal(data_inline[key], data_prefetched[key]), (
                f"Single-doc: {key} differs"
            )


# ---------------------------------------------------------------------------
# 13. Inline processing without prefetched data
# ---------------------------------------------------------------------------


class TestInlineProcessing:
    def test_process_shard_with_none_prefetched_works(
        self, tiny_model, tiny_sae, layer, tmp_path
    ):
        """process_shard(prefetched=None) still runs the inline text-loading path."""
        texts = [
            "Backward compat test document one.",
            "Backward compat test document two.",
            "Backward compat test document three.",
        ]
        path = _make_parquet(str(tmp_path / "compat.parquet"), texts)

        result = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path),
            context_window=1024,
            pooling=PoolingMode.mean_pool,
        )

        assert result.n_docs == 3
        assert result.n_tokens_gpt2 > 0
        assert Path(result.output_path).exists()

        data = load_features(result.output_path)
        assert data["n_docs"] == 3
        assert data["offsets"][0].item() == 0
        assert data["offsets"][-1].item() == len(data["feature_ids"])

    def test_process_shard_without_prefetched_kwarg_works(
        self, tiny_model, tiny_sae, layer, tmp_path
    ):
        """Calling process_shard without the prefetched kwarg at all still works."""
        texts = ["Test doc."]
        path = _make_parquet(str(tmp_path / "no_kwarg.parquet"), texts)

        # Call without prefetched keyword — should default to None
        result = process_shard(
            path,
            tiny_model,
            tiny_sae,
            layer=layer,
            output_dir=str(tmp_path),
        )
        assert result.n_docs == 1
