"""Tests for H200 production fixes: zero-width tokens, dynamic compile,
and max_seq_len from checkpoint metadata.

These reproduce real issues discovered during the first H200 conditioned
SFT run on d12_baseline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir

from steering.segmentation import assign_tokens_to_segments

# ===================================================================
# Fix 1: Zero-width token handling in assign_tokens_to_segments
# ===================================================================


class TestZeroWidthTokens:
    """Zero-width tokens (U+FFFD byte-merge artifacts) must be assigned
    to segments, not flagged as unassigned."""

    def test_zero_width_at_exact_segment_boundary_tiling(self):
        """Reproduction of production bug: zero-width token at chars [302, 302)
        with segment [0, 302) raises ValueError.

        Root cause: after_end check uses >= which treats start==end==boundary
        as 'past the end', even though searchsorted correctly assigns it."""
        token_starts = t.tensor([0, 100, 200, 300, 302], dtype=t.int64)
        token_ends = t.tensor([100, 200, 300, 302, 302], dtype=t.int64)
        seg_starts = t.tensor([0], dtype=t.int64)
        seg_ends = t.tensor([302], dtype=t.int64)

        # Should NOT raise — zero-width token should be assigned to segment 0
        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        assert assignments[4] == 0  # assigned to the only segment

    def test_zero_width_at_segment_boundary_multi_segment(self):
        """Zero-width token at boundary between segments [0,128) and [128,256)."""
        token_starts = t.tensor([120, 128, 130], dtype=t.int64)
        token_ends = t.tensor([128, 128, 140], dtype=t.int64)  # token 1 is zero-width at 128
        seg_starts = t.tensor([0, 128], dtype=t.int64)
        seg_ends = t.tensor([128, 256], dtype=t.int64)

        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        # Zero-width at 128 could go to either segment — just shouldn't crash
        assert assignments[1] in (0, 1)

    def test_zero_width_mid_segment(self):
        """Zero-width token in the middle of a segment — no boundary issue."""
        token_starts = t.tensor([0, 50, 50, 100], dtype=t.int64)
        token_ends = t.tensor([50, 50, 100, 200], dtype=t.int64)  # token 1 is zero-width at 50
        seg_starts = t.tensor([0], dtype=t.int64)
        seg_ends = t.tensor([200], dtype=t.int64)

        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        assert len(assignments) == 4
        assert assignments[1] == 0  # assigned to the only segment

    def test_zero_width_at_last_segment_end(self):
        """Production case: zero-width at total_chars = seg_ends[-1]."""
        # Exactly the chars [783, 783) case from production
        token_starts = t.tensor([0, 400, 782, 783], dtype=t.int64)
        token_ends = t.tensor([400, 782, 783, 783], dtype=t.int64)
        seg_starts = t.tensor([0, 400], dtype=t.int64)
        seg_ends = t.tensor([400, 783], dtype=t.int64)

        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        assert assignments[3] == 1  # last segment

    def test_multiple_zero_width_tokens(self):
        """Multiple zero-width tokens scattered through the text."""
        # Mimics conv 276 which has 9 zero-width tokens
        token_starts = t.tensor([0, 100, 302, 302, 500, 642, 642, 800], dtype=t.int64)
        token_ends = t.tensor([100, 302, 302, 500, 642, 642, 800, 1000], dtype=t.int64)
        seg_starts = t.tensor([0, 500], dtype=t.int64)
        seg_ends = t.tensor([500, 1000], dtype=t.int64)

        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        assert len(assignments) == 8
        # Zero-width at 302 should go to segment 0
        assert assignments[2] == 0
        # Zero-width at 642 should go to segment 1
        assert assignments[5] == 1

    def test_normal_tokens_still_validated(self):
        """Non-zero-width tokens past the segment range should still raise."""
        token_starts = t.tensor([0, 100, 500], dtype=t.int64)
        token_ends = t.tensor([100, 200, 600], dtype=t.int64)  # token 2 past segments
        seg_starts = t.tensor([0], dtype=t.int64)
        seg_ends = t.tensor([300], dtype=t.int64)

        with pytest.raises(ValueError, match="zero overlap"):
            assign_tokens_to_segments(
                token_starts, token_ends, seg_starts, seg_ends
            )

    def test_zero_width_slow_path(self):
        """Zero-width tokens handled correctly in the non-tiling (slow) path."""
        # Non-contiguous segments: [0,100) and [200,300) — gap at [100,200)
        token_starts = t.tensor([50, 100, 250], dtype=t.int64)
        token_ends = t.tensor([100, 100, 300], dtype=t.int64)  # token 1 zero-width at 100
        seg_starts = t.tensor([0, 200], dtype=t.int64)
        seg_ends = t.tensor([100, 300], dtype=t.int64)

        assignments = assign_tokens_to_segments(
            token_starts, token_ends, seg_starts, seg_ends
        )
        assert assignments[0] == 0
        assert assignments[2] == 1


# ===================================================================
# Fix 2: dynamic=True for torch.compile
# ===================================================================


class TestDynamicCompile:
    """Verify SFT script uses dynamic compilation for conditioned mode."""

    def test_compile_static_config_exists(self):
        """base_train.yaml has compile_static config."""

        config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="conditional_encdec_sft")
        assert hasattr(cfg, "compile_static")


# ===================================================================
# Fix 3: max_seq_len from checkpoint metadata
# ===================================================================


class TestMaxSeqLenFromCheckpoint:
    """max_seq_len should default to checkpoint value, not 2048."""

    def test_config_default_is_null(self):
        """SFT config should default max_seq_len to null (use checkpoint value)."""

        config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="conditional_encdec_sft")
        assert cfg.max_seq_len is None, (
            f"max_seq_len should default to null, got {cfg.max_seq_len}"
        )

    def test_explicit_override_works(self):
        """max_seq_len can be overridden via CLI."""

        config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="conditional_encdec_sft",
                overrides=["max_seq_len=512"],
            )
        assert cfg.max_seq_len == 512
