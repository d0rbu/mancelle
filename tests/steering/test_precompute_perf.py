"""Performance benchmark for token boundary precomputation.

Run with: uv run pytest tests/test_precompute_perf.py -v -s --benchmark
"""

from __future__ import annotations

import time

import pytest
import torch as t

from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from tests.conftest import MockTokenizer


def _make_doc_data(n_tokens: int, n_segs: int):
    """Create realistic synthetic document data."""
    total_chars = n_tokens * 4  # ~4 chars per token
    seg_size = total_chars // n_segs
    seg_starts = t.tensor([i * seg_size for i in range(n_segs)], dtype=t.int64)
    seg_ends = t.tensor(
        [total_chars if i == n_segs - 1 else (i + 1) * seg_size for i in range(n_segs)],
        dtype=t.int64,
    )
    tok_starts = t.arange(0, total_chars, 4, dtype=t.int64)[:n_tokens]
    tok_ends = (tok_starts + 4).clamp(max=total_chars)
    return tok_starts, tok_ends, seg_starts, seg_ends


def _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends):
    """O(n×m) overlap matrix reference."""
    if len(tok_starts) == 0:
        return t.tensor([], dtype=t.int64)
    lo = t.max(tok_starts.unsqueeze(1), seg_starts.unsqueeze(0))
    hi = t.min(tok_ends.unsqueeze(1), seg_ends.unsqueeze(0))
    overlap = (hi - lo).clamp(min=0)
    return overlap.argmax(dim=1).to(t.int64)


@pytest.mark.benchmark
class TestPrecomputePerformance:
    """Benchmark the full per-document pipeline."""

    def test_assign_tokens_speedup(self):
        """Measure assign_tokens speedup across realistic sizes."""
        configs = [
            (500, 4, "small_doc"),
            (1000, 8, "medium_doc"),
            (2000, 16, "large_doc"),
            (3000, 16, "xlarge_doc"),
        ]
        N = 500
        print("\n--- assign_tokens_to_segments benchmark ---")
        for n_tok, n_seg, label in configs:
            tok_s, tok_e, seg_s, seg_e = _make_doc_data(n_tok, n_seg)

            t0 = time.perf_counter()
            for _ in range(N):
                assign_tokens_to_segments(tok_s, tok_e, seg_s, seg_e)
            t_new = (time.perf_counter() - t0) / N

            t0 = time.perf_counter()
            for _ in range(N):
                _assign_matrix_reference(tok_s, tok_e, seg_s, seg_e)
            t_ref = (time.perf_counter() - t0) / N

            # Verify correctness
            assert t.equal(
                assign_tokens_to_segments(tok_s, tok_e, seg_s, seg_e),
                _assign_matrix_reference(tok_s, tok_e, seg_s, seg_e),
            )
            speedup = t_ref / t_new
            print(f"  {label:12s} ({n_tok:4d}tok × {n_seg:2d}seg): "
                  f"new={t_new*1000:.3f}ms  ref={t_ref*1000:.3f}ms  speedup={speedup:.1f}x")

    def test_full_pipeline_per_doc(self):
        """Benchmark full per-document pipeline: char_offsets + assign + cummax + extract."""
        tok = MockTokenizer()
        n_tokens = 1500
        n_segs = 12

        # Prepare token IDs
        text = "a" * n_tokens  # ASCII text = fast path for char offsets
        token_ids = tok.encode(text)
        n_tokens = len(token_ids)

        # Prepare segment char boundaries (tiling)
        chars_per_seg = n_tokens // n_segs
        seg_char_starts = [i * chars_per_seg for i in range(n_segs)]
        seg_char_ends = [
            n_tokens if i == n_segs - 1 else (i + 1) * chars_per_seg
            for i in range(n_segs)
        ]

        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            char_offsets = token_to_char_offsets(tok, token_ids)  # ty: ignore[invalid-argument-type]
            tok_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
            tok_ends = t.tensor(char_offsets[1:], dtype=t.int64)
            seg_s = t.tensor(seg_char_starts, dtype=t.int64)
            seg_e = t.tensor(seg_char_ends, dtype=t.int64)
            assignments = assign_tokens_to_segments(tok_starts, tok_ends, seg_s, seg_e)
            assignments = t.cummax(assignments, dim=0).values
        t_total = (time.perf_counter() - t0) / N

        print(f"\n--- Full per-doc pipeline ({n_tokens} tokens, {n_segs} segs) ---")
        print(f"  Total: {t_total*1000:.2f}ms/doc")
        print(f"  Estimated per-shard (53K docs): {53000 * t_total:.1f}s")

    def test_shard_estimate(self):
        """Estimate full shard precompute time from per-doc measurements."""
        tok = MockTokenizer()

        # Simulate a distribution of document sizes
        doc_sizes = [200, 500, 500, 1000, 1000, 1000, 1500, 2000, 2000, 3000]

        total_time = 0.0
        for n_tok in doc_sizes:
            n_segs = max(1, n_tok // 128)
            text = "a" * n_tok
            token_ids = tok.encode(text)

            t0 = time.perf_counter()
            N = 50
            for _ in range(N):
                char_offsets = token_to_char_offsets(tok, token_ids)  # ty: ignore[invalid-argument-type]
                tok_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
                tok_ends = t.tensor(char_offsets[1:], dtype=t.int64)
                seg_s = t.tensor([i * (n_tok // n_segs) for i in range(n_segs)], dtype=t.int64)
                seg_e = t.tensor(
                    [n_tok if i == n_segs - 1 else (i + 1) * (n_tok // n_segs) for i in range(n_segs)],
                    dtype=t.int64,
                )
                assign_tokens_to_segments(tok_starts, tok_ends, seg_s, seg_e)
            per_call = (time.perf_counter() - t0) / N
            total_time += per_call

        avg_per_doc = total_time / len(doc_sizes)
        estimated_shard = avg_per_doc * 53000

        print("\n--- Shard estimate (mixed doc sizes) ---")
        print(f"  Avg per-doc: {avg_per_doc*1000:.3f}ms")
        print(f"  Estimated per-shard (53K docs): {estimated_shard:.1f}s")
        print(f"  Estimated full run (500 shards): {estimated_shard * 500 / 3600:.1f}h")
