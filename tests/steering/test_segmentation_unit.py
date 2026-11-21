"""Unit tests for slice 02 production fixes (B-2, B-5, B-12).

B-2: ``assign_tokens_to_segments`` must reject non-monotonic ``seg_char_ends``
     instead of silently mishandling via ``searchsorted`` on a non-sorted needle.
B-5: ``_segment_fixed`` (and ``segment_tokens``) must reject ``segment_size <= 0``
     instead of silently producing 0 segments.
B-12: ``compute_tokenizer_checksum`` must include the tokenizer's ``bos_token_id``
     so a BOS-id-only mismatch flips the checksum.

Tests are TDD: written to fail against the pre-fix production code, then pass
once the fixes land in the same PR.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import (
    _segment_fixed,
    assign_tokens_to_segments,
    compute_tokenizer_checksum,
    segment_tokens,
)

# ---------------------------------------------------------------------------
# B-2: monotonicity check on contiguous-tiling fast path
# ---------------------------------------------------------------------------


class TestAssignTokensMonotonicity:
    """B-2 (P1) — ``assign_tokens_to_segments`` validates segment monotonicity.

    The pre-fix fast path (``is_tiling`` short-circuit) assumed every
    contiguously-tiled input also had monotonically non-decreasing
    ``seg_char_ends``. A pathological caller with
    ``seg_char_starts=[10, 20]`` and ``seg_char_ends=[20, 5]`` satisfied the
    tiling check but ``searchsorted`` requires sorted needles to give
    correct results.

    Production (``_segment_fixed`` + extraction) never produces such
    inputs, so this is a defensive contract guard that goes from "correct
    by upstream invariant" to "explicitly enforced".
    """

    def test_non_monotone_ends_raises(self) -> None:
        # Tiling holds (seg_char_starts[1:] == seg_char_ends[:-1]) but ends
        # are not non-decreasing.
        seg_char_starts = t.tensor([10, 20], dtype=t.int64)
        seg_char_ends = t.tensor([20, 5], dtype=t.int64)
        token_char_starts = t.tensor([12], dtype=t.int64)
        token_char_ends = t.tensor([15], dtype=t.int64)

        with pytest.raises(ValueError, match="monoton"):
            assign_tokens_to_segments(
                token_char_starts,
                token_char_ends,
                seg_char_starts,
                seg_char_ends,
            )

    def test_non_monotone_ends_with_three_segments_raises(self) -> None:
        # Tiling DOES hold (seg_char_starts[1:] == seg_char_ends[:-1] →
        # [20, 15] == [20, 15]) but seg_char_ends [20, 15, 18] is not
        # monotonically non-decreasing (20 > 15). The fast-path
        # ``searchsorted`` would silently misbehave; the B-2 guard raises.
        seg_char_starts = t.tensor([10, 20, 15], dtype=t.int64)
        seg_char_ends = t.tensor([20, 15, 18], dtype=t.int64)
        token_char_starts = t.tensor([12], dtype=t.int64)
        token_char_ends = t.tensor([14], dtype=t.int64)

        with pytest.raises(ValueError, match="monoton"):
            assign_tokens_to_segments(
                token_char_starts,
                token_char_ends,
                seg_char_starts,
                seg_char_ends,
            )

    def test_non_tiling_non_monotone_uses_slow_path(self) -> None:
        # Tiling does NOT hold here: seg_char_starts[1:] == [25] but
        # seg_char_ends[:-1] == [20]. This routes to the slow O(n*m)
        # path which is monotonicity-agnostic — the B-2 guard is
        # gated on ``is_tiling`` and stays out of the way.
        seg_char_starts = t.tensor([10, 25], dtype=t.int64)
        seg_char_ends = t.tensor([20, 18], dtype=t.int64)
        token_char_starts = t.tensor([12], dtype=t.int64)
        token_char_ends = t.tensor([14], dtype=t.int64)

        result = assign_tokens_to_segments(
            token_char_starts,
            token_char_ends,
            seg_char_starts,
            seg_char_ends,
        )
        # Slow path: [12, 14) overlaps seg 0 [10, 20) by 2 chars; no
        # overlap with seg 1 [25, 18) (clamped to 0). argmax=0.
        assert result.tolist() == [0]

    def test_monotone_input_unchanged(self) -> None:
        """Regression: well-formed monotone tiling continues to work."""
        seg_char_starts = t.tensor([0, 10, 20], dtype=t.int64)
        seg_char_ends = t.tensor([10, 20, 30], dtype=t.int64)
        token_char_starts = t.tensor([0, 5, 15, 25], dtype=t.int64)
        token_char_ends = t.tensor([5, 10, 20, 30], dtype=t.int64)

        result = assign_tokens_to_segments(
            token_char_starts,
            token_char_ends,
            seg_char_starts,
            seg_char_ends,
        )
        assert result.tolist() == [0, 0, 1, 2]


# ---------------------------------------------------------------------------
# B-5: segment_size <= 0 must raise
# ---------------------------------------------------------------------------


class TestSegmentSizePositive:
    """B-5 (P2) — ``_segment_fixed`` rejects non-positive ``segment_size``.

    Pre-fix: ``range(token_offset, n, 0)`` raises ``ValueError``
    organically (Python). But ``range(token_offset, n, -1)`` with
    ``token_offset < n`` produces an empty list, and ``[(start, end)]``
    pairing then degenerates silently to ``[(n,)]`` zip-shorter — no zip
    output at all, so callers see an empty segment list. This is a silent
    failure mode.

    Refined boundary: ``PositiveInt`` plus ``beartype`` reject invalid values
    at both the public ``segment_tokens`` surface and private helper.
    """

    def test_segment_size_zero_raises(self) -> None:
        bad_segment_size: Any = 0
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            _segment_fixed(
                n=10,
                segment_size=bad_segment_size,
                token_offset=NonNegativeInt.parse(0),
            )

    def test_segment_size_negative_raises(self) -> None:
        bad_segment_size: Any = -1
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            _segment_fixed(
                n=10,
                segment_size=bad_segment_size,
                token_offset=NonNegativeInt.parse(0),
            )

    def test_segment_size_negative_via_public_api(self) -> None:
        bad_segment_size: Any = -3
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            segment_tokens(
                token_ids=[1, 2, 3, 4, 5],
                segment_size=bad_segment_size,
            )

    def test_positive_segment_size_unchanged(self) -> None:
        """Regression: well-formed positive segment_size keeps tiling."""
        result = _segment_fixed(
            n=10,
            segment_size=PositiveInt.parse(4),
            token_offset=NonNegativeInt.parse(0),
        )
        assert result == [(0, 4), (4, 8), (8, 10)]


# ---------------------------------------------------------------------------
# B-12: tokenizer checksum includes BOS id
# ---------------------------------------------------------------------------


class _ChecksumTokenizer:
    """Minimal tokenizer surface for checksum tests.

    ``encode`` is identical for two instances; only ``bos_token_id`` differs.
    Pre-fix checksum collides; post-fix checksum disagrees.
    """

    def __init__(self, bos_token_id: int) -> None:
        self.bos_token_id = bos_token_id

    def encode(self, text: str) -> list[int]:
        # Minimal deterministic encoder: per-char ord modulo 256.
        return [ord(c) % 256 for c in text]


class TestTokenizerChecksumIncludesBOS:
    """B-12 (P3) — checksum incorporates ``bos_token_id``.

    Two tokenizers with identical ``encode`` but different BOS ids must
    produce different checksums. Otherwise a Gemma-IT vs Gemma-base swap
    that only differs in BOS would slip through cross-tokenizer guards.
    """

    def test_different_bos_changes_checksum(self) -> None:
        tok_a = _ChecksumTokenizer(bos_token_id=2)
        tok_b = _ChecksumTokenizer(bos_token_id=3)

        sum_a = compute_tokenizer_checksum(tok_a)
        sum_b = compute_tokenizer_checksum(tok_b)

        assert sum_a != sum_b, (
            "Tokenizers with identical encode() but different bos_token_id "
            "must produce different checksums (B-12)."
        )

    def test_same_bos_same_checksum(self) -> None:
        """Regression: deterministic for identical tokenizers."""
        tok_a = _ChecksumTokenizer(bos_token_id=2)
        tok_b = _ChecksumTokenizer(bos_token_id=2)

        sum_a = compute_tokenizer_checksum(tok_a)
        sum_b = compute_tokenizer_checksum(tok_b)

        assert sum_a == sum_b

    def test_checksum_is_hex_sha256(self) -> None:
        """Regression: still 64-char lowercase hex SHA-256."""
        tok = _ChecksumTokenizer(bos_token_id=1)
        cs = compute_tokenizer_checksum(tok)
        assert len(cs) == 64
        assert all(c in "0123456789abcdef" for c in cs)
