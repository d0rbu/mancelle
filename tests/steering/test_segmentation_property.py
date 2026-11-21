"""Property-based tests for slice 02 invariants (B-3, B-6, B-9, B-10, B-11).

Hypothesis configured per design spec §7: ``derandomize=True`` and a fixed
``max_examples=200``. Seeds are explicit at the suite level — Hypothesis's
``derandomize=True`` makes runs reproducible without needing per-test seeds.

Items covered:
- B-3 (P2): pin ``get_path_stats()`` reset-on-read contract.
- B-6 (P2): pin zero-width-token-after-last-segment absorption behavior.
- B-10 (P3): ``compute_context_token_segment_assignments`` returns all-zeros
  when ``context_segments`` is empty.
- B-11 (P2): ``_token_to_char_offsets_progressive`` always emits monotone
  non-decreasing offsets even on long byte-fallback runs.
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.segmentation import (
    _token_to_char_offsets_progressive,
    assign_tokens_to_segments,
    compute_context_token_segment_assignments,
    get_path_stats,
    token_to_char_offsets,
)

# Suite-level Hypothesis profile (derandomized, deterministic, CI-friendly).
HYPOTHESIS_SETTINGS = settings(
    derandomize=True,
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Helper — minimal tokenizer surface used by the property tests
# ---------------------------------------------------------------------------


class _ByteTokenizer:
    """Per-byte tokenizer with byte-fallback semantics for Hypothesis runs.

    encode(text): one token per byte (utf-8) keyed by byte value.
    decode([byte0, byte1, ...]): concatenates the bytes back to a string.
    Multi-byte characters thus span multiple tokens whose decoded prefix
    contains a U+FFFD until the final byte arrives — mirroring Gemma's
    byte-fallback behavior on rare scripts.
    """

    bos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def get_bos_token_id(self) -> int:
        return self.bos_token_id


# ---------------------------------------------------------------------------
# B-3 — get_path_stats reset-on-read contract
# ---------------------------------------------------------------------------


class TestGetPathStatsResetOnRead:
    """``get_path_stats()`` is documented to RESET counters on every call.

    This is a side-effecting "getter" — slice-02 audit B-3 (P2). Two parallel
    callers cannot read stats independently; a future profiling tool that
    invokes the function twice will see zeros on the second call. We pin the
    contract as currently documented (a rename to ``pop_path_stats`` is
    deferred per the slice-02 plan).
    """

    def test_two_consecutive_calls_show_reset(self) -> None:
        # Drain whatever was there from prior tests.
        _ = get_path_stats()
        # Drive a known path: small_n triggers progressive on n <= 16.
        token_to_char_offsets(_ByteTokenizer(), [ord("a"), ord("b"), ord("c")])
        first = get_path_stats()
        second = get_path_stats()

        # First call must show at least one increment in some bucket.
        total_first = sum(b["count"] for b in first.values())
        assert total_first >= 1, (
            f"first call should record path stats; got {first}"
        )
        # Second call (no work in between) must be zero everywhere.
        total_second = sum(b["count"] for b in second.values())
        assert total_second == 0, (
            f"second call should reset to zero; got {second}"
        )

    def test_path_stats_keys_are_stable(self) -> None:
        # Public contract: keys are always {"fast", "slow", "small_n"}.
        stats = get_path_stats()
        assert set(stats.keys()) == {"fast", "slow", "small_n"}
        for bucket in stats.values():
            assert set(bucket.keys()) == {"count", "mean_ms", "total_ms"}


# ---------------------------------------------------------------------------
# B-6 — zero-width token after last segment
# ---------------------------------------------------------------------------


class TestZeroWidthTokenAfterLastSegment:
    """A zero-width token whose position falls beyond the last segment's end
    is silently absorbed into the last segment (clamped via ``searchsorted``
    + ``.clamp(max=n_segs - 1)``). Slice-02 B-6 (P2) — undocumented but
    intentional. We pin the behavior so a future change surfaces explicitly.
    """

    def test_zero_width_at_doc_end_assigned_to_last_segment(self) -> None:
        seg_char_starts = t.tensor([0, 5], dtype=t.int64)
        seg_char_ends = t.tensor([5, 10], dtype=t.int64)
        # Three real tokens covering the segments + a final zero-width
        # token positioned EXACTLY at the document end (10).
        token_char_starts = t.tensor([0, 5, 10], dtype=t.int64)
        token_char_ends = t.tensor([5, 10, 10], dtype=t.int64)

        result = assign_tokens_to_segments(
            token_char_starts,
            token_char_ends,
            seg_char_starts,
            seg_char_ends,
        )
        assert result.tolist() == [0, 1, 1]

    def test_zero_width_past_doc_end_clamped_to_last_segment(self) -> None:
        seg_char_starts = t.tensor([0, 5], dtype=t.int64)
        seg_char_ends = t.tensor([5, 10], dtype=t.int64)
        token_char_starts = t.tensor([0, 5, 12], dtype=t.int64)
        token_char_ends = t.tensor([5, 10, 12], dtype=t.int64)

        result = assign_tokens_to_segments(
            token_char_starts,
            token_char_ends,
            seg_char_starts,
            seg_char_ends,
        )
        # Position 12 > seg_char_ends[-1] = 10. Zero-width token is
        # silently clamped to n_segs - 1 = 1. This is the behavior the
        # B-6 finding documents.
        assert result.tolist() == [0, 1, 1]


# ---------------------------------------------------------------------------
# B-10 — empty context_segments returns all-zeros assignments
# ---------------------------------------------------------------------------


class TestComputeContextTokenSegmentAssignmentsEmpty:
    """B-10 (P3): zero context_segments → zero-tensor of token-length.

    The contract is documented in ``segmentation.py`` but not exercised by
    a unit test.
    """

    def test_empty_segments_with_tokens_returns_zeros(self) -> None:
        result = compute_context_token_segment_assignments(
            tokenizer=_ByteTokenizer(),
            context_tokens=[1, 2, 3, 4, 5],
            context_segments=(),
        )
        assert result.dtype == t.long
        assert result.shape == (5,)
        assert int(result.sum().item()) == 0

    def test_empty_tokens_with_empty_segments_returns_empty_tensor(self) -> None:
        result = compute_context_token_segment_assignments(
            tokenizer=_ByteTokenizer(),
            context_tokens=[],
            context_segments=(),
        )
        assert result.dtype == t.long
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# B-11 — progressive walkback always produces monotone offsets
# ---------------------------------------------------------------------------


@HYPOTHESIS_SETTINGS
@given(
    text=st.text(
        alphabet=st.characters(
            min_codepoint=0x0021,  # exclude NUL/control chars
            max_codepoint=0x024F,  # Latin extended; mostly multi-byte
        ),
        min_size=0,
        max_size=80,
    )
)
def test_progressive_offsets_are_non_decreasing(text: str) -> None:
    """B-11 (P2): walkback must never produce a decreasing offset.

    Construct token sequences via the byte-fallback ``_ByteTokenizer``;
    multi-byte characters yield long byte-fallback runs in the
    progressive path. The contract: ``offsets[i+1] >= offsets[i]`` for
    every ``i``.
    """
    tok = _ByteTokenizer()
    token_ids = tok.encode(text)
    offsets = _token_to_char_offsets_progressive(tok, token_ids)
    assert len(offsets) == len(token_ids) + 1
    assert offsets[0] == 0
    for i in range(len(offsets) - 1):
        assert offsets[i + 1] >= offsets[i], (
            f"offsets non-monotone at index {i}: {offsets[i]} -> "
            f"{offsets[i + 1]}; offsets={offsets}; text={text!r}"
        )
    # The final offset must equal the decoded length.
    assert offsets[-1] == len(tok.decode(token_ids))


@HYPOTHESIS_SETTINGS
@given(
    text=st.text(
        alphabet=st.characters(
            min_codepoint=0x0021,
            max_codepoint=0x024F,
        ),
        min_size=0,
        max_size=80,
    )
)
def test_progressive_matches_token_to_char_offsets_router(text: str) -> None:
    """The public ``token_to_char_offsets`` (with all routing) must always
    agree with the canonical progressive reference on byte tokenizers.

    This is the slice-02 invariant that motivated the chunked path's
    bail-to-progressive fallback: chunked must be either correct (per-token
    decode == full decode) or it must defer to progressive.
    """
    tok = _ByteTokenizer()
    token_ids = tok.encode(text)
    public = token_to_char_offsets(tok, token_ids)
    progressive = _token_to_char_offsets_progressive(tok, token_ids)
    assert public == progressive


# ---------------------------------------------------------------------------
# Module-level guard — ensure tests don't accidentally rely on side-effects.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drain_path_stats():
    """Drain ``_path_stats`` before each test to keep B-3 tests deterministic.

    Untyped on purpose: ``yield`` makes this a generator fixture, which ty
    cannot annotate as ``-> None``.
    """
    _ = get_path_stats()
    yield
    _ = get_path_stats()
