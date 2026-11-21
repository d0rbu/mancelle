"""Performance regression baselines for the encoder-decoder training system.

These are hang-detection guards, not tight regression bounds. Hardware varies
too much for specific thresholds. The value is: (1) smoke-testing the full
training path, (2) printing timing data for manual comparison.

Run with:
    uv run pytest tests/test_encdec_performance.py -v -m "benchmark" -s --tb=short
"""

import resource
import time

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from tests.encdec_test_helpers import (
    SEQ_LEN,
    build_tiny_model,
    make_precomputed_dataloader,
    setup_optimizer_simple,
    train_steps,
)

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Module-level autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module.

    FlexAttention does not support backward on CPU. We patch the module-level
    override directly rather than using an env var, because the flash_attention
    module may already be imported by earlier tests in the suite.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# TestThroughput
# ---------------------------------------------------------------------------


class TestThroughput:
    """Throughput benchmarks — detect hangs in the full training path."""

    def test_training_throughput(self, tmp_path):
        """Build model, warm up 2 steps, measure 10 steps. Assert > 100 tok/sec."""
        model = build_tiny_model()
        optimizer = setup_optimizer_simple(model)
        dataloader = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)

        # Warm up
        train_steps(model, optimizer, dataloader, n_steps=2)

        # Measure
        B = 2
        T = SEQ_LEN
        n_measure = 10
        t0 = time.perf_counter()
        train_steps(model, optimizer, dataloader, n_steps=n_measure)
        elapsed = time.perf_counter() - t0

        tokens_processed = n_measure * B * T
        tok_per_sec = tokens_processed / elapsed
        print(f"\n[test_training_throughput] {tok_per_sec:.1f} tok/sec over {n_measure} steps")
        assert tok_per_sec > 100, (
            f"Training throughput too low: {tok_per_sec:.1f} tok/sec (expected > 100)"
        )

    def test_dataloader_throughput(self, tmp_path):
        """Create dataloader, warm up 2 batches, measure 20 batches. Assert > 1 batch/sec."""
        dataloader = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)

        # Warm up
        for _ in range(2):
            next(dataloader)

        # Measure
        n_measure = 20
        t0 = time.perf_counter()
        for _ in range(n_measure):
            next(dataloader)
        elapsed = time.perf_counter() - t0

        batches_per_sec = n_measure / elapsed
        print(
            f"\n[test_dataloader_throughput] {batches_per_sec:.2f} batches/sec "
            f"over {n_measure} batches"
        )
        assert batches_per_sec > 1, (
            f"Dataloader throughput too low: {batches_per_sec:.2f} batches/sec (expected > 1)"
        )


# ---------------------------------------------------------------------------
# TestTiming
# ---------------------------------------------------------------------------


class TestTiming:
    """Per-pass timing tests — detect hangs, print timing for comparison."""

    def test_forward_pass_time(self, tmp_path):
        """Warm up 1 forward pass, measure 10 passes. Assert avg < 5000 ms."""
        model = build_tiny_model()
        model.eval()
        dataloader = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)

        def _forward_once():
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = next(
                dataloader
            )
            with torch.no_grad():
                result = model(
                    x, y, enc_fids, enc_mags, enc_mask, seg_map,
                    seg_doc_id=seg_doc_id,
                )
            return result

        # Warm up
        _forward_once()

        # Measure
        n_measure = 10
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _forward_once()
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000
        print(f"\n[test_forward_pass_time] avg forward pass: {avg_ms:.1f} ms over {n_measure} passes")
        assert avg_ms < 5000, (
            f"Forward pass too slow: {avg_ms:.1f} ms (expected < 5000 ms)"
        )

    def test_backward_pass_time(self, tmp_path):
        """Warm up 1 fwd+bwd, measure 10 fwd+bwd. Assert avg < 10000 ms."""
        model = build_tiny_model()
        model.train()
        optimizer = setup_optimizer_simple(model)
        dataloader = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)

        def _fwd_bwd_once():
            optimizer.zero_grad(set_to_none=True)
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = next(
                dataloader
            )
            result = model(
                x, y, enc_fids, enc_mags, enc_mask, seg_map,
                seg_doc_id=seg_doc_id,
            )
            if hasattr(result, "loss"):
                loss = result.loss
            else:
                loss = result
            loss.backward()

        # Warm up
        _fwd_bwd_once()

        # Measure
        n_measure = 10
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _fwd_bwd_once()
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000
        print(
            f"\n[test_backward_pass_time] avg fwd+bwd: {avg_ms:.1f} ms over {n_measure} passes"
        )
        assert avg_ms < 10000, (
            f"Forward+backward too slow: {avg_ms:.1f} ms (expected < 10000 ms)"
        )

    def test_memory_usage_bounded(self, tmp_path):
        """Run 5 training steps and assert resident-set growth < 2048 MB."""
        model = build_tiny_model()
        optimizer = setup_optimizer_simple(model)
        dataloader = make_precomputed_dataloader(tmp_path, B=2, n_rows=32)

        before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        train_steps(model, optimizer, dataloader, n_steps=5)
        after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # On Linux ru_maxrss is in kilobytes; on macOS it is in bytes.
        import platform
        if platform.system() == "Darwin":
            delta_mb = (after_kb - before_kb) / (1024 * 1024)
        else:
            delta_mb = (after_kb - before_kb) / 1024

        print(f"\n[test_memory_usage_bounded] RSS delta: {delta_mb:.1f} MB over 5 training steps")
        assert delta_mb < 2048, (
            f"Memory growth too large: {delta_mb:.1f} MB (expected < 2048 MB)"
        )
