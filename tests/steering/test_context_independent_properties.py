"""Property-based tests for context-independent extraction.

Uses Hypothesis to fuzz (n_docs, doc_len, seg_size, pooling, gpu_batch_size)
and verify production matches reference.

Run with:
    uv run pytest tests/test_context_independent_properties.py -v --model-scale=tiny
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import hypothesis.strategies as st
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from hypothesis import given, settings

from steering.extract_features import PoolingMode, process_shard
from steering.feature_io import load_features
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE


@pytest.fixture(scope="module")
def model_sae():
    model = TinyHookedTransformer(d_model=64, vocab_size=256, device="cpu")
    sae = TinySAE(d_model=64, n_features=128, device="cpu")
    return model, sae


def _run_context_independent(
    tmp_dir: str, texts: list[str], seg_size: int, pooling: PoolingMode,
    gpu_bs: int, model, sae,
) -> dict:
    path = str(Path(tmp_dir) / "shard.parquet")
    pq.write_table(pa.table({"text": texts}), path)
    result = process_shard(
        path, model, sae,
        layer=8, output_dir=tmp_dir,
        context_window=128, pooling=pooling,
        segment_size=seg_size, gpu_batch_size=gpu_bs,
        context_independent_features=True,
    )
    return load_features(result.output_path)


class TestContextIndependentProperties:
    @given(
        n_docs=st.integers(min_value=1, max_value=5),
        min_len=st.integers(min_value=1, max_value=5),
        seg_size=st.integers(min_value=1, max_value=16),
        pooling=st.sampled_from(list(PoolingMode)),
        gpu_bs=st.sampled_from([1, 2, 4, 8]),
        seed=st.integers(0, 2**31),
    )
    @settings(max_examples=15, deadline=30000)
    def test_context_independent_invariant_to_gpu_batch_size(
        self, model_sae, n_docs, min_len, seg_size, pooling, gpu_bs, seed,
    ):
        """Running context-independent with different gpu_batch_sizes produces identical feature_ids
        (floats may differ within batch-GEMM tolerance)."""
        model, sae = model_sae
        t.manual_seed(seed)
        texts = [
            "x" * (min_len + i * 3 + seed % 7) for i in range(n_docs)
        ]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = _run_context_independent(d1, texts, seg_size, pooling, 1, model, sae)
            r2 = _run_context_independent(d2, texts, seg_size, pooling, gpu_bs, model, sae)
            assert t.equal(r1["feature_ids"], r2["feature_ids"])
            assert t.allclose(r1["feature_mags"], r2["feature_mags"], atol=2e-4)
            assert t.equal(r1["offsets"], r2["offsets"])

    @given(
        seg_size=st.integers(min_value=1, max_value=16),
        pooling=st.sampled_from(list(PoolingMode)),
    )
    @settings(max_examples=10, deadline=30000)
    def test_context_independent_segment_metadata_matches_cd(self, model_sae, seg_size, pooling):
        """context-independent and context-dependent produce identical segment metadata for the same input."""
        model, sae = model_sae
        texts = ["document alpha beta gamma", "second doc here"]
        with tempfile.TemporaryDirectory() as dcd, tempfile.TemporaryDirectory() as dcontext_independent:
            pcd = str(Path(dcd) / "shard.parquet")
            pcontext_independent = str(Path(dcontext_independent) / "shard.parquet")
            for p in (pcd, pcontext_independent):
                pq.write_table(pa.table({"text": texts}), p)
            cd = process_shard(
                pcd, model, sae, layer=8, output_dir=dcd,
                context_window=128, pooling=pooling,
                segment_size=seg_size, gpu_batch_size=4,
            )
            context_independent_result = process_shard(
                pcontext_independent, model, sae, layer=8, output_dir=dcontext_independent,
                context_window=128, pooling=pooling,
                segment_size=seg_size, gpu_batch_size=4,
                context_independent_features=True,
            )
            dc = load_features(cd.output_path)
            context_independent_data = load_features(context_independent_result.output_path)
            for key in ["doc_ids", "segment_indices", "segment_positions",
                        "segment_char_starts", "segment_char_ends",
                        "doc_token_counts"]:
                assert t.equal(dc[key], context_independent_data[key]), (
                    f"metadata key {key} differs for seg_size={seg_size}, "
                    f"pooling={pooling.value}"
                )
