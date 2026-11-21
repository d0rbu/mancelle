"""Tests for the batch-size auto-tuner in steering.extract_features.

Run with:
    uv run pytest tests/test_auto_batch_tuner.py -v --model-scale=tiny
"""

from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import find_optimal_batch_sizes

# Import tiny model fixtures from the integration test module
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)


@pytest.fixture
def tiny_model():
    return TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")


@pytest.fixture
def tiny_sae():
    return TinySAE(d_model=64, n_features=256, device="cpu")


class TestSyntheticFallback:
    """When parquet_paths=None, behavior is unchanged (generates synthetic data)."""

    def test_returns_valid_result_without_parquet_paths(self, tiny_model, tiny_sae):
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert isinstance(gpu_bs, int) and gpu_bs >= 1
        assert isinstance(cpu_bs, int) and cpu_bs >= 1

    def test_parquet_paths_none_uses_synthetic(self, tiny_model, tiny_sae):
        """Explicitly passing parquet_paths=None should use synthetic data."""
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            parquet_paths=None,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs >= 1 and cpu_bs >= 1


class TestRealDataSampling:
    """When parquet_paths is provided, documents are sampled from actual parquet."""

    def test_uses_parquet_data_when_provided(self, tiny_model, tiny_sae, tmp_path):
        """Auto-tuner should read from the provided parquet, not generate synthetic."""
        long_text = "Real document content for benchmarking. " * 100
        texts = [long_text] * 8
        parquet_path = str(tmp_path / "shard_0000.parquet")
        pq.write_table(pa.table({"text": texts}), parquet_path)

        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            parquet_paths=[parquet_path],
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs >= 1 and cpu_bs >= 1

    def test_falls_back_to_synthetic_on_empty_parquet_list(self, tiny_model, tiny_sae):
        """Empty parquet_paths list should fall back to synthetic data."""
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            parquet_paths=[],
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs >= 1 and cpu_bs >= 1

    def test_falls_back_to_synthetic_on_corrupt_parquet(self, tiny_model, tiny_sae, tmp_path):
        """If parquet file can't be read, fall back to synthetic data."""
        bad_path = str(tmp_path / "corrupt.parquet")
        with open(bad_path, "w") as f:
            f.write("not a parquet file")
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            parquet_paths=[bad_path],
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs >= 1 and cpu_bs >= 1


class TestGridBounds:
    """Verify min/max batch size bounds correctly constrain the grid."""

    def test_min_gpu_excludes_small_sizes(self, tiny_model, tiny_sae):
        """min_gpu_batch_size=4 should exclude gpu_bs=1,2."""
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            min_gpu_batch_size=4,
            max_gpu_batch_size=8,
            max_cpu_batch_size=8,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs >= 4, f"gpu_bs={gpu_bs} below min=4"

    def test_min_cpu_excludes_small_sizes(self, tiny_model, tiny_sae):
        """min_cpu_batch_size=4 should exclude cpu_bs=1,2."""
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            min_cpu_batch_size=4,
            max_gpu_batch_size=8,
            max_cpu_batch_size=8,
            n_trials=1,
            n_docs=4,
        )
        assert cpu_bs >= 4, f"cpu_bs={cpu_bs} below min=4"

    def test_max_gpu_caps_upper_bound(self, tiny_model, tiny_sae):
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=8,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs <= 4

    def test_max_cpu_caps_upper_bound(self, tiny_model, tiny_sae):
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            max_gpu_batch_size=8,
            max_cpu_batch_size=4,
            n_trials=1,
            n_docs=4,
        )
        assert cpu_bs <= 4

    def test_result_validity_invariants(self, tiny_model, tiny_sae):
        """Returned sizes: powers of 2, gpu <= cpu, within [min, max]."""
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            tiny_model,
            tiny_sae,
            layer=8,
            context_window=128,
            min_gpu_batch_size=2,
            min_cpu_batch_size=2,
            max_gpu_batch_size=16,
            max_cpu_batch_size=16,
            n_trials=1,
            n_docs=4,
        )
        assert gpu_bs & (gpu_bs - 1) == 0, f"gpu_bs={gpu_bs} not power of 2"
        assert cpu_bs & (cpu_bs - 1) == 0, f"cpu_bs={cpu_bs} not power of 2"
        assert gpu_bs <= cpu_bs
        assert 2 <= gpu_bs <= 16
        assert 2 <= cpu_bs <= 16


class TestOOMHandling:
    """Verify OOM resilience in the benchmark loop."""

    def test_skips_oom_combo_and_continues(self, tiny_model, tiny_sae):
        """If one (gpu_bs, cpu_bs) pair OOMs, tuner should skip it and find another."""
        from steering import extract_features as ef_module
        original_process_shard = ef_module.process_shard

        def oom_on_large_gpu(parquet_path, model, sae, **kwargs):
            if kwargs.get("gpu_batch_size", 1) >= 4:
                raise t.cuda.OutOfMemoryError("CUDA out of memory")
            return original_process_shard(parquet_path, model, sae, **kwargs)

        with patch.object(ef_module, "process_shard", side_effect=oom_on_large_gpu):
            gpu_bs, cpu_bs = find_optimal_batch_sizes(
                tiny_model,
                tiny_sae,
                layer=8,
                context_window=128,
                max_gpu_batch_size=8,
                max_cpu_batch_size=8,
                n_trials=1,
                n_docs=4,
            )
        # Should have found a working combo with gpu_bs < 4
        assert gpu_bs < 4
        assert cpu_bs >= 1

    def test_all_oom_raises_error(self, tiny_model, tiny_sae):
        """If every combo OOMs, function should raise RuntimeError."""
        from steering import extract_features as ef_module

        def always_oom(*args, **kwargs):
            raise t.cuda.OutOfMemoryError("CUDA out of memory")

        with patch.object(ef_module, "process_shard", side_effect=always_oom):
            with pytest.raises(RuntimeError, match="All .* batch size combinations failed"):
                find_optimal_batch_sizes(
                    tiny_model,
                    tiny_sae,
                    layer=8,
                    context_window=128,
                    max_gpu_batch_size=4,
                    max_cpu_batch_size=4,
                    n_trials=1,
                    n_docs=4,
                )

    def test_oom_logs_warning(self, tiny_model, tiny_sae):
        """OOM should produce a log message with 'OOM, skipping'."""
        import io

        from loguru import logger

        from steering import extract_features as ef_module

        def always_oom(*args, **kwargs):
            raise t.cuda.OutOfMemoryError("CUDA out of memory")

        # Capture loguru output
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            with patch.object(ef_module, "process_shard", side_effect=always_oom):
                with pytest.raises(RuntimeError):
                    find_optimal_batch_sizes(
                        tiny_model,
                        tiny_sae,
                        layer=8,
                        context_window=128,
                        max_gpu_batch_size=2,
                        max_cpu_batch_size=2,
                        n_trials=1,
                        n_docs=4,
                    )
            log_text = log_output.getvalue()
            assert "OOM, skipping" in log_text, f"Expected 'OOM, skipping' in log, got: {log_text}"
        finally:
            logger.remove(handler_id)

    def test_empty_cache_called_after_every_trial(self, tiny_model, tiny_sae):
        """torch.cuda.empty_cache() should be called after every trial and on OOM."""
        from unittest.mock import PropertyMock

        from steering import extract_features as ef_module

        def oom_on_large_gpu(parquet_path, model, sae, **kwargs):
            if kwargs.get("gpu_batch_size", 1) >= 4:
                raise t.cuda.OutOfMemoryError("CUDA out of memory")
            # For small gpu_bs, just return successfully (no-op)
            return None

        # Mock the SAE's device to report "cuda" so the empty_cache branches execute
        fake_device = t.device("cuda")

        with patch.object(ef_module, "process_shard", side_effect=oom_on_large_gpu), \
             patch("torch.cuda.empty_cache") as mock_empty_cache, \
             patch("torch.cuda.synchronize"), \
             patch.object(ef_module, "_get_sae_activations_batched"), \
             patch.object(type(tiny_sae._dummy_param), "device", new_callable=PropertyMock, return_value=fake_device):
            gpu_bs, cpu_bs = find_optimal_batch_sizes(
                tiny_model,
                tiny_sae,
                layer=8,
                context_window=128,
                max_gpu_batch_size=8,
                max_cpu_batch_size=8,
                n_trials=1,
                n_docs=4,
            )
        # empty_cache is called: once after each successful trial + once per OOM combo
        assert mock_empty_cache.call_count >= 1, (
            f"Expected empty_cache to be called, got {mock_empty_cache.call_count} calls"
        )
        assert gpu_bs >= 1


class TestDocumentLength:
    """Verify sampled documents produce realistic token counts."""

    def test_sampled_docs_fill_context_window(self, tiny_model, tiny_sae, tmp_path):
        """When given long docs, the auto-tuner's benchmark parquet should contain them."""
        # Create parquet with long documents (each ~4000 chars -> 4000 tokens with TinyHookedTransformer)
        long_text = "A" * 4000
        texts = [long_text] * 10
        parquet_path = str(tmp_path / "shard.parquet")
        pq.write_table(pa.table({"text": texts}), parquet_path)

        # Spy on pq.write_table to capture what the auto-tuner writes as benchmark data
        original_write = pq.write_table
        written_tables = []

        def spy_write(table, path, *args, **kwargs):
            written_tables.append(table)
            return original_write(table, path, *args, **kwargs)

        with patch("pyarrow.parquet.write_table", side_effect=spy_write):
            find_optimal_batch_sizes(
                tiny_model,
                tiny_sae,
                layer=8,
                context_window=2048,
                parquet_paths=[parquet_path],
                max_gpu_batch_size=2,
                max_cpu_batch_size=2,
                n_trials=1,
                n_docs=4,
            )

        # The first write_table call is the auto-tuner's benchmark parquet
        assert len(written_tables) >= 1, "Auto-tuner should have written a benchmark parquet"
        bench_texts = written_tables[0].column("text").to_pylist()

        for doc in bench_texts:
            tokens = tiny_model.tokenizer.encode(doc)
            # TinyHookedTransformer: 1 token per char, 4000 chars > 2048 context_window
            assert len(tokens) >= 2048, (
                f"Sampled doc too short: {len(tokens)} tokens, expected >= 2048"
            )
