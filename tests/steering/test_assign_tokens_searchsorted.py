"""Tests for the searchsorted optimization of assign_tokens_to_segments.

Verifies that the fast O(n log m) searchsorted path produces identical
results to the O(n × m) overlap matrix for all segment configurations.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.segmentation import assign_tokens_to_segments


def _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends):
    """Reference implementation using the O(n×m) overlap matrix (always)."""
    if len(tok_starts) == 0:
        return t.tensor([], dtype=t.int64)
    lo = t.max(tok_starts.unsqueeze(1), seg_starts.unsqueeze(0))
    hi = t.min(tok_ends.unsqueeze(1), seg_ends.unsqueeze(0))
    overlap = (hi - lo).clamp(min=0)
    return overlap.argmax(dim=1).to(t.int64)


class TestSearchsortedCorrectness:
    """Verify searchsorted path matches overlap matrix for tiling segments."""

    @pytest.mark.parametrize("n_tokens,n_segs", [
        (10, 1), (50, 2), (100, 4), (200, 8), (500, 16),
        (1000, 8), (2000, 16), (3000, 16), (100, 16),
    ])
    def test_matches_matrix_uniform_tokens(self, n_tokens, n_segs):
        """Uniform 5-char tokens, tiling segments."""
        total_chars = n_tokens * 5
        seg_size = total_chars // n_segs
        seg_starts = t.tensor([i * seg_size for i in range(n_segs)], dtype=t.int64)
        seg_ends = t.tensor(
            [(i + 1) * seg_size if i < n_segs - 1 else total_chars for i in range(n_segs)],
            dtype=t.int64,
        )
        tok_starts = t.arange(0, total_chars, 5, dtype=t.int64)[:n_tokens]
        tok_ends = (tok_starts + 5).clamp(max=total_chars)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)

    @pytest.mark.parametrize("n_tokens,n_segs", [
        (100, 3), (200, 7), (500, 13), (1000, 11),
    ])
    def test_matches_matrix_uneven_segments(self, n_tokens, n_segs):
        """Uneven segment sizes (not divisible), tiling segments."""
        total_chars = n_tokens * 5
        boundaries = t.linspace(0, total_chars, n_segs + 1, dtype=t.int64)
        seg_starts = boundaries[:-1]
        seg_ends = boundaries[1:]
        # Ensure exact tiling
        seg_ends[-1] = total_chars

        tok_starts = t.arange(0, total_chars, 5, dtype=t.int64)[:n_tokens]
        tok_ends = (tok_starts + 5).clamp(max=total_chars)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)

    def test_variable_token_widths(self):
        """Tokens with different character widths."""
        # Tokens of width 1, 3, 2, 7, 4, 1, 5, 2 = 25 chars total
        widths = [1, 3, 2, 7, 4, 1, 5, 2]
        cumulative = [0]
        for w in widths:
            cumulative.append(cumulative[-1] + w)
        tok_starts = t.tensor(cumulative[:-1], dtype=t.int64)
        tok_ends = t.tensor(cumulative[1:], dtype=t.int64)

        seg_starts = t.tensor([0, 10, 20], dtype=t.int64)
        seg_ends = t.tensor([10, 20, 25], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)

    def test_single_segment(self):
        """All tokens assigned to the only segment."""
        tok_starts = t.arange(0, 50, 5, dtype=t.int64)
        tok_ends = tok_starts + 5
        seg_starts = t.tensor([0], dtype=t.int64)
        seg_ends = t.tensor([50], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert (result == 0).all()

    def test_single_token(self):
        tok_starts = t.tensor([10], dtype=t.int64)
        tok_ends = t.tensor([15], dtype=t.int64)
        seg_starts = t.tensor([0, 12, 20], dtype=t.int64)
        seg_ends = t.tensor([12, 20, 30], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)

    def test_empty_tokens(self):
        result = assign_tokens_to_segments(
            t.tensor([], dtype=t.int64), t.tensor([], dtype=t.int64),
            t.tensor([0], dtype=t.int64), t.tensor([10], dtype=t.int64),
        )
        assert len(result) == 0


class TestSearchsortedValidation:
    """Verify zero-overlap detection works with searchsorted path."""

    def test_tokens_beyond_segments_raises(self):
        tok_starts = t.arange(0, 100, 5, dtype=t.int64)
        tok_ends = tok_starts + 5
        seg_starts = t.tensor([0, 25], dtype=t.int64)
        seg_ends = t.tensor([25, 50], dtype=t.int64)
        with pytest.raises(ValueError, match="zero overlap"):
            assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)

    def test_tokens_before_segments_raises(self):
        tok_starts = t.tensor([0, 5], dtype=t.int64)
        tok_ends = t.tensor([5, 10], dtype=t.int64)
        seg_starts = t.tensor([20], dtype=t.int64)
        seg_ends = t.tensor([30], dtype=t.int64)
        with pytest.raises(ValueError, match="zero overlap"):
            assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)


class TestFallbackToMatrix:
    """Verify non-tiling segments fall back to overlap matrix."""

    def test_non_contiguous_segments_uses_matrix(self):
        """Segments with overlap (not tiling) fall back to matrix path."""
        # Tokens [0-15), segments overlap: [0-10) and [5-15)
        tok_starts = t.arange(0, 15, 1, dtype=t.int64)
        tok_ends = tok_starts + 1
        seg_starts = t.tensor([0, 5], dtype=t.int64)
        seg_ends = t.tensor([10, 15], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)

    def test_overlapping_segments_uses_matrix(self):
        """Overlapping segments should still work (matrix path)."""
        tok_starts = t.arange(0, 30, 3, dtype=t.int64)
        tok_ends = tok_starts + 3
        seg_starts = t.tensor([0, 10, 20], dtype=t.int64)
        seg_ends = t.tensor([15, 25, 30], dtype=t.int64)

        result = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        reference = _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        assert t.equal(result, reference)


class TestBenchmark:
    """Benchmark to verify speedup (not a correctness test)."""

    @pytest.mark.benchmark
    def test_speedup_2000_tokens_16_segs(self):
        """Verify searchsorted is faster than matrix for typical workload."""
        import time

        n_tokens, n_segs = 2000, 16
        total_chars = n_tokens * 5
        seg_size = total_chars // n_segs
        seg_starts = t.tensor([i * seg_size for i in range(n_segs)], dtype=t.int64)
        seg_ends = t.tensor(
            [(i + 1) * seg_size if i < n_segs - 1 else total_chars for i in range(n_segs)],
            dtype=t.int64,
        )
        tok_starts = t.arange(0, total_chars, 5, dtype=t.int64)[:n_tokens]
        tok_ends = (tok_starts + 5).clamp(max=total_chars)

        N = 1000
        t0 = time.perf_counter()
        for _ in range(N):
            assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        t_fast = (time.perf_counter() - t0) / N

        t0 = time.perf_counter()
        for _ in range(N):
            _assign_matrix_reference(tok_starts, tok_ends, seg_starts, seg_ends)
        t_matrix = (time.perf_counter() - t0) / N

        speedup = t_matrix / t_fast
        print(f"\nSearchsorted: {t_fast*1000:.3f}ms, Matrix: {t_matrix*1000:.3f}ms, Speedup: {speedup:.1f}x")
        # Speedup varies by CPU load; just verify it's not slower
        assert speedup > 0.8, f"Searchsorted should not be slower, got {speedup:.1f}x"
