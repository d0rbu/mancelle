"""Performance benchmarks for steering.extract_features.

Uses pytest-benchmark to track regressions. Requires --model-scale=tiny for CI.

Run with:
    uv run pytest tests/steering/test_performance.py -m benchmark -v --model-scale=tiny
    uv run pytest tests/steering/test_performance.py -m benchmark --benchmark-autosave
    uv run pytest tests/steering/test_performance.py -m benchmark --benchmark-compare
    uv run pytest tests/ --benchmark-disable                        # skip in normal runs
"""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.compute_statistics import compute_statistics
from steering.extract_features import (
    PoolingMode,
    _get_sae_activations,
    _get_sae_activations_batched,
    extract_document_features,
    find_optimal_batch_sizes,
    process_shard,
)
from steering.feature_io import save_features

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Tiny model fixtures (mirror test_extract_features_integration.py)
# ---------------------------------------------------------------------------


class TinyHookPoint(t.nn.Module):
    """Minimal hook point that supports register_forward_hook."""

    def forward(self, x):
        return x


class TinyBlock(t.nn.Module):
    """Minimal transformer block with hook_resid_pre."""

    def __init__(self):
        super().__init__()
        self.hook_resid_pre = TinyHookPoint()


class TinySAE:
    """Minimal SAE substitute for benchmarks."""

    def __init__(self, d_model: int = 64, n_features: int = 256, device: str = "cpu"):
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = t.randn(d_model, n_features, device=device) * 0.1
        self.b_enc = t.randn(n_features, device=device) * 0.01
        self.device = device
        self.cfg = type("obj", (object,), {"d_sae": n_features})()
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x @ self.W_enc + self.b_enc)

    def parameters(self):
        yield self._dummy_param


class TinyHookedTransformer:
    """Minimal HookedTransformer substitute for benchmarks."""

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._device = device
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))
        self.tokenizer = MagicMock()
        self.tokenizer.encode = self._encode
        self.tokenizer.decode = lambda tids: "".join(
            chr(t % 128) if t < 128 else f"<{t}>" for t in tids
        )

        def _tokenizer_call(texts_or_text, **kw):
            if isinstance(texts_or_text, str):
                return {"input_ids": self._encode(texts_or_text)}
            return {"input_ids": [self._encode(text) for text in texts_or_text]}

        self.tokenizer.side_effect = _tokenizer_call

        self.cfg = type("obj", (object,), {"n_ctx": 2048, "model_name": "gpt2"})()

        self._embedding = t.randn(vocab_size, d_model, device=device) * 0.1
        self.blocks = [TinyBlock() for _ in range(12)]

    def _encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text] if text else []

    def parameters(self):
        yield self._dummy_param

    def run_with_cache(
        self,
        tokens: t.Tensor,
        names_filter: list[str],
        attention_mask: t.Tensor | None = None,
        stop_at_layer: int | None = None,
    ):
        del attention_mask, stop_at_layer
        seq_len = tokens.shape[1]
        token_ids = tokens[0].clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]
        cache = {name: activations.unsqueeze(0) for name in names_filter}
        logits = t.zeros(1, seq_len, self.vocab_size, device=self._device)
        return logits, cache

    def __call__(self, tokens, stop_at_layer=None, **kwargs):
        """Forward pass that triggers hooks on blocks.

        Supports stop_at_layer to match TransformerLens API:
        if stop_at_layer=N, runs blocks 0..N-1 and returns the residual stream.
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        B, seq_len = tokens.shape
        token_ids = tokens.clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]  # [B, seq, d_model]
        end = stop_at_layer if stop_at_layer is not None else len(self.blocks)
        for block in self.blocks[:end]:
            activations = block.hook_resid_pre(activations)
        if stop_at_layer is not None:
            return activations
        return t.zeros(B, seq_len, self.vocab_size, device=self._device)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    return TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")


@pytest.fixture(scope="module")
def tiny_sae():
    return TinySAE(d_model=64, n_features=256, device="cpu")


def _make_text(n_chars: int) -> str:
    """Generate repeating text of approximately n_chars length."""
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, n_chars // len(base))
    return (base * repeats)[:n_chars]


def _make_synthetic_feature_file(
    path: str,
    n_docs: int = 100,
    features_per_doc: int = 50,
    n_total_features: int = 24576,
):
    """Create a synthetic feature shard file with valid CSR structure."""
    offsets = [0]
    all_ids = []
    all_mags = []
    doc_token_counts = []

    for _ in range(n_docs):
        ids = t.randint(0, n_total_features, (features_per_doc,), dtype=t.int32)
        mags = t.rand(features_per_doc, dtype=t.float32).abs() + 0.01
        sort_idx = mags.argsort(descending=True)
        all_ids.append(ids[sort_idx])
        all_mags.append(mags[sort_idx])
        offsets.append(offsets[-1] + features_per_doc)
        doc_token_counts.append(t.randint(100, 5000, (1,)).item())

    save_data = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.cat(all_ids) if all_ids else t.tensor([], dtype=t.int32),
        "feature_mags": t.cat(all_mags) if all_mags else t.tensor([], dtype=t.float32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": n_docs,
        "doc_ids": t.arange(n_docs, dtype=t.int32),
        "segment_indices": t.zeros(n_docs, dtype=t.int32),
        "segment_positions": t.zeros(n_docs, dtype=t.int32),
        "n_segments": n_docs,
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
    }
    save_features(save_data, path)
    return save_data


def _make_synthetic_parquet(path: str, texts: Sequence[str]):
    table = pa.table({"text": texts})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# extract_document_features benchmarks
# ---------------------------------------------------------------------------


class TestExtractDocumentPerformance:
    """Benchmark extract_document_features across pooling modes and text lengths."""

    @pytest.mark.parametrize(
        "n_chars", [100, 500, 2000], ids=["100ch", "500ch", "2000ch"]
    )
    @pytest.mark.parametrize("pooling", [PoolingMode.mean_pool, PoolingMode.max_pool])
    def test_extract_document(self, benchmark, tiny_model, tiny_sae, pooling, n_chars):
        text = _make_text(n_chars)
        feat_ids, feat_mags, positions, char_starts, char_ends, n_tokens = benchmark(
            extract_document_features,
            tiny_model,
            tiny_sae,
            text,
            layer=8,
            context_window=128,
            pooling=pooling,
        )
        assert n_tokens > 0
        assert len(feat_ids) == len(feat_mags) == len(positions)
        assert len(char_starts) == len(char_ends) == len(positions)


# ---------------------------------------------------------------------------
# _get_sae_activations benchmarks
# ---------------------------------------------------------------------------


class TestGetSAEActivationsPerformance:
    """Benchmark _get_sae_activations at different sequence lengths."""

    @pytest.mark.parametrize(
        "seq_len", [32, 128, 512], ids=["32tok", "128tok", "512tok"]
    )
    def test_get_sae_activations(self, benchmark, tiny_model, tiny_sae, seq_len):
        tokens = t.randint(0, 1000, (seq_len,), dtype=t.long)
        result = benchmark(_get_sae_activations, tiny_model, tiny_sae, tokens, layer=8)
        assert result.shape == (seq_len, tiny_sae.n_features)
        assert result.dtype == t.float32


# ---------------------------------------------------------------------------
# process_shard benchmarks
# ---------------------------------------------------------------------------


class TestProcessShardPerformance:
    """Benchmark process_shard on a small synthetic parquet."""

    @pytest.fixture
    def parquet_20docs(self, tmp_path):
        texts = [_make_text(200 + i * 50) for i in range(20)]
        path = str(tmp_path / "bench_shard.parquet")
        _make_synthetic_parquet(path, texts)
        return path

    def test_process_shard(
        self, benchmark, tiny_model, tiny_sae, parquet_20docs, tmp_path
    ):
        output_dir = str(tmp_path / "bench_output")
        Path(output_dir).mkdir(exist_ok=True)
        result = benchmark(
            process_shard,
            parquet_20docs,
            tiny_model,
            tiny_sae,
            layer=8,
            output_dir=output_dir,
            context_window=128,
            pooling=PoolingMode.mean_pool,
        )
        assert result.n_docs == 20
        assert Path(result.output_path).is_file()


# ---------------------------------------------------------------------------
# process_shard batched benchmarks (various batch sizes)
# ---------------------------------------------------------------------------


class TestProcessShardBatchedPerformance:
    """Benchmark process_shard with different batch sizes to measure batching impact."""

    @pytest.fixture
    def parquet_40docs(self, tmp_path):
        texts = [_make_text(200 + i * 20) for i in range(40)]
        path = str(tmp_path / "bench_shard_batched.parquet")
        _make_synthetic_parquet(path, texts)
        return path

    @pytest.mark.parametrize("batch_size", [1, 8, 32], ids=["B1", "B8", "B32"])
    def test_process_shard_batched(
        self,
        benchmark,
        tiny_model,
        tiny_sae,
        parquet_40docs,
        tmp_path,
        batch_size,
    ):
        output_dir = str(tmp_path / f"bench_output_b{batch_size}")
        Path(output_dir).mkdir(exist_ok=True)
        result = benchmark(
            process_shard,
            parquet_40docs,
            tiny_model,
            tiny_sae,
            layer=8,
            output_dir=output_dir,
            context_window=128,
            pooling=PoolingMode.mean_pool,
            gpu_batch_size=batch_size,
        )
        assert result.n_docs == 40
        assert Path(result.output_path).is_file()


# ---------------------------------------------------------------------------
# process_shard 2D batch size benchmarks (gpu_batch_size x cpu_batch_size)
# ---------------------------------------------------------------------------


class TestDualBatchSizePerformance:
    """Benchmark process_shard with different (gpu_batch, cpu_batch) combos."""

    @pytest.fixture
    def parquet_40docs(self, tmp_path):
        texts = [_make_text(200 + i * 20) for i in range(40)]
        path = str(tmp_path / "bench_dual_batch.parquet")
        _make_synthetic_parquet(path, texts)
        return path

    @pytest.mark.parametrize(
        "gpu_bs,cpu_bs",
        [(1, 1), (4, 4), (8, 8), (4, 16), (8, 32)],
        ids=["G1C1", "G4C4", "G8C8", "G4C16", "G8C32"],
    )
    def test_dual_batch_throughput(
        self,
        benchmark,
        tiny_model,
        tiny_sae,
        parquet_40docs,
        tmp_path,
        gpu_bs,
        cpu_bs,
    ):
        output_dir = str(tmp_path / f"bench_g{gpu_bs}_c{cpu_bs}")
        Path(output_dir).mkdir(exist_ok=True)
        result = benchmark(
            process_shard,
            parquet_40docs,
            tiny_model,
            tiny_sae,
            layer=8,
            output_dir=output_dir,
            context_window=128,
            pooling=PoolingMode.mean_pool,
            gpu_batch_size=gpu_bs,
            cpu_batch_size=cpu_bs,
        )
        assert result.n_docs == 40
        assert Path(result.output_path).is_file()


# ---------------------------------------------------------------------------
# _get_sae_activations_batched benchmarks
# ---------------------------------------------------------------------------


class TestBatchedForwardPerformance:
    """Benchmark _get_sae_activations_batched at different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16], ids=["B1", "B4", "B16"])
    def test_batched_activations(self, benchmark, tiny_model, tiny_sae, batch_size):
        seq_len = 128
        token_id_lists = [list(range(seq_len)) for _ in range(batch_size)]
        result = benchmark(
            _get_sae_activations_batched,
            tiny_model,
            tiny_sae,
            token_id_lists,
            layer=8,
        )
        assert len(result) == batch_size
        assert all(acts.shape == (seq_len, tiny_sae.n_features) for acts in result)


# ---------------------------------------------------------------------------
# CUDA-only benchmarks (skipped if no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
class TestCUDAPerformance:
    """CUDA-specific benchmarks requiring real GPU."""

    @pytest.fixture(scope="class")
    def cuda_model_and_sae(self):
        """Load real GPT-2 + SAE on CUDA for benchmarking."""
        try:
            from steering.extract_features import load_sae_and_model

            model, sae, _ = load_sae_and_model(device="cuda")
            return model, sae
        except Exception:
            pytest.skip("Failed to load real model")

    @pytest.mark.slow
    def test_stop_at_layer_raw_cuda(self, benchmark, cuda_model_and_sae):
        """Raw GPT-2 forward pass (no SAE) for reference."""
        model, _ = cuda_model_and_sae
        tokens = t.randint(0, 50257, (8, 1024), dtype=t.long, device="cuda")

        def fn():
            with t.no_grad():
                model(tokens, stop_at_layer=8)
            t.cuda.synchronize()

        benchmark(fn)

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16], ids=["B1", "B4", "B8", "B16"])
    def test_batched_activations_cuda(self, benchmark, cuda_model_and_sae, batch_size):
        """Batched SAE activations on GPU at various batch sizes."""
        model, sae = cuda_model_and_sae
        tokenizer = model.tokenizer
        # Use real-ish text to get varied lengths
        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        token_id_lists = [
            tokenizer.encode(base_text[: 200 + i * 50])[-1024:]
            for i in range(batch_size)
        ]

        def fn():
            _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
            t.cuda.synchronize()

        benchmark(fn)

    @pytest.mark.slow
    def test_sequential_vs_batched_cuda(self, benchmark, cuda_model_and_sae):
        """Compare sequential _get_sae_activations vs batched on GPU with 8 docs."""
        model, sae = cuda_model_and_sae
        tokenizer = model.tokenizer
        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        token_id_lists = [
            tokenizer.encode(base_text[: 200 + i * 80])[-1024:] for i in range(8)
        ]

        def fn():
            _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
            t.cuda.synchronize()

        benchmark(fn)

    @pytest.mark.slow
    def test_sequential_activations_cuda(self, benchmark, cuda_model_and_sae):
        """Sequential (one-at-a-time) SAE activations on GPU for baseline comparison."""
        model, sae = cuda_model_and_sae
        tokenizer = model.tokenizer
        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        token_id_lists = [
            tokenizer.encode(base_text[: 200 + i * 80])[-1024:] for i in range(8)
        ]

        def fn():
            from steering.extract_features import _get_sae_activations

            device = next(model.parameters()).device
            for ids in token_id_lists:
                tokens = t.tensor(ids, dtype=t.long, device=device)
                _get_sae_activations(model, sae, tokens, layer=8)
            t.cuda.synchronize()

        benchmark(fn)

    @pytest.mark.slow
    def test_find_optimal_batch_sizes_cuda(self, cuda_model_and_sae):
        """2D auto batch size search on CUDA returns valid, usable result."""
        model, sae = cuda_model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=8,
            context_window=1024,
            max_gpu_batch_size=64,
            max_cpu_batch_size=64,
        )
        assert isinstance(gpu_bs, int) and gpu_bs >= 1
        assert isinstance(cpu_bs, int) and cpu_bs >= 1
        assert gpu_bs & (gpu_bs - 1) == 0
        assert cpu_bs & (cpu_bs - 1) == 0
        assert gpu_bs <= cpu_bs
        # Verify GPU batch size actually works
        token_id_lists = [list(range(1024)) for _ in range(gpu_bs)]
        results = _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
        assert len(results) == gpu_bs
        t.cuda.synchronize()

    @pytest.mark.slow
    def test_process_shard_cuda(self, benchmark, cuda_model_and_sae, tmp_path):
        """End-to-end process_shard on GPU with real model."""
        model, sae = cuda_model_and_sae
        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        texts = [base_text[: 200 + i * 100] for i in range(20)]
        parquet_path = str(tmp_path / "cuda_bench.parquet")
        _make_synthetic_parquet(parquet_path, texts)
        output_dir = str(tmp_path / "cuda_output")
        Path(output_dir).mkdir(exist_ok=True)

        def fn():
            process_shard(
                parquet_path,
                model,
                sae,
                layer=8,
                output_dir=output_dir,
                context_window=1024,
                pooling=PoolingMode.mean_pool,
                gpu_batch_size=8,
            )
            t.cuda.synchronize()

        benchmark(fn)


# ---------------------------------------------------------------------------
# compute_statistics benchmarks
# ---------------------------------------------------------------------------


class TestComputeStatisticsPerformance:
    """Benchmark compute_statistics on synthetic shard files."""

    @pytest.fixture
    def stats_dir(self, tmp_path):
        for i in range(3):
            _make_synthetic_feature_file(
                str(tmp_path / f"shard_{i:05d}_features"),
                n_docs=100,
                features_per_doc=50,
            )
        return str(tmp_path)

    def test_compute_statistics(self, benchmark, stats_dir, tmp_path):
        output_path = str(tmp_path / "bench_stats.json")
        stats = benchmark(compute_statistics, stats_dir, output_path)
        assert stats.corpus.n_shards_processed == 3
        assert stats.corpus.n_docs == 300


# ---------------------------------------------------------------------------
# Batch size granularity investigation (power-of-2 vs non-power-of-2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
class TestBatchSizeGranularity:
    """Profile power-of-2 vs non-power-of-2 batch sizes to see if fine-grained search helps."""

    @pytest.fixture(scope="class")
    def cuda_model_and_sae(self):
        try:
            from steering.extract_features import load_sae_and_model

            model, sae, _ = load_sae_and_model(device="cuda")
            return model, sae
        except Exception:
            pytest.skip("Failed to load real model")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32],
        ids=[f"B{b}" for b in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32]],
    )
    def test_batch_size_throughput(self, benchmark, cuda_model_and_sae, batch_size):
        """Measure throughput at each batch size to see if non-powers-of-2 help."""
        model, sae = cuda_model_and_sae
        token_id_lists = [list(range(1024)) for _ in range(batch_size)]

        def fn():
            _get_sae_activations_batched(model, sae, token_id_lists, layer=8)
            t.cuda.synchronize()

        benchmark(fn)
