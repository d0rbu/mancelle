"""PR-C: build_target_for_real_config dispatcher.

- PooledTargetRealConfig -> SingleTargetSegment(features=pool_segments_to_dict(...))
- LockstepTargetRealConfig -> MultiTargetSegments(segments=..., token_assignments=...)

Char-frame note: target_char_start points at the first target character.
filter_segments_to_target rebases cache-frame chars to target-relative frame.
The lockstep dispatcher maps target-overlapping tokens directly and lets a
single leading autoregressive position inherit the first target assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from steering.eval.context_cache_io import ContextCacheEntry, ContextFeatureSegment
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    MultiTargetSegments,
    PooledTargetRealConfig,
    SingleTargetSegment,
)
from steering.eval.target_real_dispatch import build_target_for_real_config
from tests.conftest import build_tiny_tokenizer


def _entry_with_segments(token_ids, segments):
    return ContextCacheEntry(
        token_ids=tuple(int(x) for x in token_ids),
        segments=tuple(segments),
    )


# ---- Pooled dispatch ----

def test_pooled_dispatch_returns_single_target_segment():
    """Pure pooling — no conversation routing involved. char ranges align directly."""
    tokenizer = build_tiny_tokenizer()
    cfg = PooledTargetRealConfig(
        name="pooled", cache_dir=Path("/tmp/x"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    full_token_ids = [0, 1, 2, 3, 4]
    entry = _entry_with_segments(
        full_token_ids,
        [
            ContextFeatureSegment(feature_ids=(1,), buckets=(2,), char_start=0, char_end=10),
            ContextFeatureSegment(feature_ids=(1, 2), buckets=(3, 1), char_start=10, char_end=20),
        ],
    )
    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=0,
        target_char_end=20,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[0, 0, 1, 1, 1],
        tokenizer=tokenizer,
    )
    assert isinstance(target, SingleTargetSegment)
    # Pooled BUCKET_MAX: feature 1 -> max(2, 3) = 3; feature 2 -> 1.
    assert target.features == {1: 3, 2: 1}


def test_pooled_dispatch_with_bucket_mean():
    tokenizer = build_tiny_tokenizer()
    cfg = PooledTargetRealConfig(
        name="pooled", cache_dir=Path("/tmp/x"), pooling_op=PoolingOp.BUCKET_MEAN,
    )
    full_token_ids = [0, 1, 2, 3]
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(2,), char_start=0, char_end=8),
        ContextFeatureSegment(feature_ids=(1,), buckets=(4,), char_start=8, char_end=16),
    ])
    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=0, target_char_end=16,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[0, 1, 1, 1],
        tokenizer=tokenizer,
    )
    assert isinstance(target, SingleTargetSegment)
    assert target.features == {1: 3}  # mean(2, 4) = 3


# ---- Lockstep dispatch ----

def test_lockstep_dispatch_returns_multi_target_segments():
    """Synthetic 3-segment lockstep over a 4-token response.

    Conversation: [BOS, 'a', 'b', 'c', 'd', 'e'] = 6 tokens. 'a' is the
    start marker (per loss_mask convention); 'bcde' is the response (4 chars).
    Cache segments cover chars [1, 2), [2, 4), [4, 5) of the full sequence.
    target_char_start=1 (start of response), target_char_end=5.
    After filter, segments rebased to [0, 1), [1, 3), [3, 4) — 4 chars total,
    matches response-token chars exactly.
    """
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="lockstep", cache_dir=Path("/tmp/x"))
    full_token_ids = list(tokenizer("abcde", prepend=tokenizer.get_bos_token_id()))
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=1, char_end=2),
        ContextFeatureSegment(feature_ids=(2,), buckets=(2,), char_start=2, char_end=4),
        ContextFeatureSegment(feature_ids=(3,), buckets=(3,), char_start=4, char_end=5),
    ])
    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=1,    # post-start-marker
        target_char_end=5,
        full_token_ids=full_token_ids,  # ty: ignore[invalid-argument-type]
        target_token_loss_mask=[0] + [1] * (len(full_token_ids) - 1),
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    assert len(target.segments) == 3
    # The leading "a" position predicts the first target token, so it inherits
    # the first target assignment before the four response-token assignments.
    assert target.token_assignments.tolist() == [0, 0, 1, 1, 2]
    # All assignments are 0-indexed and fit in [0, 3).
    assert int(target.token_assignments.max().item()) < 3
    assert int(target.token_assignments.min().item()) >= 0
    # Monotonic non-decreasing (MultiTargetSegments __post_init__ enforces this).


def test_lockstep_one_letter_response_collapses_to_n_eq_1():
    """ARC/MMLU one-letter responses produce N=1 lockstep config.

    Conversation: [BOS, 'a' (start marker), 'X' (the answer letter)] = 3 tokens.
    target_char_start=1 (post-marker), target_char_end=2.
    Cache segment at char [1, 2) → after filter, [0, 1) — matches the single
    response token's 1-char span.
    """
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="lockstep_categorical", cache_dir=Path("/tmp/x"))
    full_token_ids = list(tokenizer("aX", prepend=tokenizer.get_bos_token_id()))
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(2,), char_start=1, char_end=2),
    ])
    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=1, target_char_end=2,
        full_token_ids=full_token_ids,  # ty: ignore[invalid-argument-type]
        target_token_loss_mask=[0] + [1] * (len(full_token_ids) - 1),
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    assert len(target.segments) == 1
    assert target.segments[0] == {1: 2}
    assert target.token_assignments.tolist() == [0, 0]


def test_lockstep_allows_trailing_non_overlap_terminator_position():
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="lockstep", cache_dir=Path("/tmp/x"))
    full_token_ids = list(tokenizer("abcde", prepend=tokenizer.get_bos_token_id()))
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=1, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(2,), char_start=3, char_end=4),
    ])
    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=1,
        target_char_end=4,
        full_token_ids=full_token_ids,  # ty: ignore[invalid-argument-type]
        # "a" is a leading predictor position, "e" is a trailing terminator
        # position. Neither overlaps the target chars, and both are assigned to
        # the nearest target segment.
        target_token_loss_mask=[0, 1, 1, 1, 1, 1],
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    assert target.token_assignments.tolist() == [0, 0, 0, 1, 1]


def test_lockstep_zero_width_internal_token_inherits_segment():
    """A zero-width token between target-overlapping positions must NOT abort.

    A multi-byte UTF-8 character mid-response tokenizes into several tokens whose
    intermediate pieces contribute no characters, so ``token_to_char_offsets``
    renders them as zero-width spans (equal consecutive offsets). Such a token
    carries no target overlap by construction; it must inherit the surrounding
    segment assignment (as the loss/categorical path tolerates it) rather than
    being flagged as an internal non-overlap.

    Regression: the lockstep dispatcher previously raised on these, crashing
    target_real eval on realistic chat data containing emoji/CJK characters
    mid-response. (Under monotonic char offsets a *positive*-width internal
    non-overlap is impossible — only zero-width tokens ever reach that branch —
    so the prior hard error could only ever be a false positive.)
    """
    class _MultiByteTokenizer:
        # Token 99 decodes to "" — a multi-byte UTF-8 intermediate token that
        # token_to_char_offsets renders as a zero-width span.
        def decode(self, token_ids):
            return "".join("" if int(tok) == 99 else "x" for tok in token_ids)

    cfg = LockstepTargetRealConfig(name="lockstep", cache_dir=Path("/tmp/x"))
    # chars: x x <zero-width> x  ->  char_offsets [0, 1, 2, 2, 3]
    full_token_ids = [1, 2, 99, 3]
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=0, char_end=1),
        ContextFeatureSegment(feature_ids=(2,), buckets=(2,), char_start=1, char_end=2),
        ContextFeatureSegment(feature_ids=(3,), buckets=(3,), char_start=2, char_end=3),
    ])

    target = build_target_for_real_config(
        cfg=cfg, entry=entry,
        target_char_start=0,
        target_char_end=3,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[1, 1, 1, 1],
        tokenizer=_MultiByteTokenizer(),
    )
    assert isinstance(target, MultiTargetSegments)
    # The zero-width token (index 2) inherits the preceding segment (1); the
    # real tokens map to their own segments. Monotonic, in-bounds.
    assert target.token_assignments.tolist() == [0, 1, 1, 2]
    assert len(target.segments) == 3


def test_lockstep_expands_internal_forced_span_without_post_tool_shift():
    """Forced tool output consumes positions that lockstep must represent.

    The target span has seven absolute decoder positions.  Only positions
    1,2,6,7 are routed/scored (mask=1); positions 3,4,5 model the forced
    ``<output_start>result<output_end>`` span.  The old compressed vector was
    ``[0, 1, 5, 6]`` and therefore made post-tool positions use the wrong
    target segments.  Every position from the first through last routed token
    must now receive a forward-filled assignment.
    """

    class _OneCharTokenizer:
        def decode(self, token_ids):
            return "".join(chr(ord("a") + int(token)) for token in token_ids)

    tokenizer = _OneCharTokenizer()
    cfg = LockstepTargetRealConfig(name="lockstep-tool-gap", cache_dir=Path("/tmp/x"))
    full_token_ids = list(range(8))
    entry = _entry_with_segments(
        full_token_ids,
        [
            ContextFeatureSegment(
                feature_ids=(idx,),
                buckets=(idx % 4,),
                char_start=idx,
                char_end=idx + 1,
            )
            for idx in range(1, 8)
        ],
    )

    target = build_target_for_real_config(
        cfg=cfg,
        entry=entry,
        target_char_start=1,
        target_char_end=8,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[0, 1, 1, 0, 0, 0, 1, 1],
        tokenizer=tokenizer,
    )

    assert isinstance(target, MultiTargetSegments)
    assert target.token_assignments.tolist() == [0, 1, 1, 1, 1, 5, 6]
    assert target.token_assignments.numel() == 7
    assert target.token_assignments.tolist() == sorted(target.token_assignments.tolist())


@pytest.mark.parametrize(
    ("target_mask", "expected"),
    [
        ([0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1]),
        ([0, 0, 1, 1, 1, 0, 0], [0, 1, 1]),
        ([0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 2]),
    ],
)
def test_lockstep_contiguous_routes_preserve_the_legacy_assignment_vector(
    target_mask: list[int], expected: list[int]
) -> None:
    """When no absolute gap exists, expansion must be a value-identical no-op."""
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="legacy-parity", cache_dir=Path("/tmp/x"))
    full_token_ids = cast(
        list[int],
        tokenizer("abcdef", prepend=tokenizer.get_bos_token_id()),
    )
    entry = _entry_with_segments(
        full_token_ids,
        [
            ContextFeatureSegment((1,), (1,), 1, 2),
            ContextFeatureSegment((2,), (2,), 2, 4),
            ContextFeatureSegment((3,), (3,), 4, 6),
        ],
    )
    target = build_target_for_real_config(
        cfg=cfg,
        entry=entry,
        target_char_start=1,
        target_char_end=6,
        full_token_ids=full_token_ids,
        target_token_loss_mask=target_mask,
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    assert target.token_assignments.tolist() == expected
    assert target.token_assignments.numel() == sum(target_mask)


def test_lockstep_forced_gap_assignment_is_forward_filled_only_until_next_geometry():
    """A forced span inherits the preceding slot and stops at the next route."""
    class _OneCharTokenizer:
        def decode(self, token_ids):
            return "".join(chr(ord("a") + int(token)) for token in token_ids)

    tokenizer = _OneCharTokenizer()
    full_token_ids = list(range(7))
    entry = _entry_with_segments(
        full_token_ids,
        [
            ContextFeatureSegment((10,), (1,), 2, 3),
            ContextFeatureSegment((11,), (2,), 3, 5),
            ContextFeatureSegment((12,), (3,), 5, 6),
        ],
    )
    target = build_target_for_real_config(
        cfg=LockstepTargetRealConfig(name="gap-boundary", cache_dir=Path("/tmp/x")),
        entry=entry,
        target_char_start=2,
        target_char_end=6,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[0, 0, 1, 0, 1, 1, 0],
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    # Absolute positions 2..5 become relative [0..3]; position 3 is forced.
    assert target.token_assignments.tolist() == [0, 0, 1, 2]


@pytest.mark.parametrize(
    "mask",
    [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    ],
)
def test_lockstep_bad_or_empty_masks_fail_loudly(mask: list[int]):
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="lockstep", cache_dir=Path("/tmp/x"))
    full_token_ids = cast(
        list[int],
        tokenizer("abcd", prepend=tokenizer.get_bos_token_id()),
    )
    entry = _entry_with_segments(
        full_token_ids,
        [ContextFeatureSegment((1,), (1,), 0, 4)],
    )
    if sum(mask) == 0:
        expected = "at least one target_token_loss_mask=1"
    else:
        expected = "length mismatch"
    with pytest.raises(ValueError, match=expected):
        build_target_for_real_config(
            cfg=cfg,
            entry=entry,
            target_char_start=0,
            target_char_end=4,
            full_token_ids=full_token_ids,
            target_token_loss_mask=mask,
            tokenizer=tokenizer,
        )


def test_lockstep_dispatch_propagates_filter_error():
    """If filter_segments_to_target raises (e.g. no overlap), so does the dispatcher."""
    tokenizer = build_tiny_tokenizer()
    cfg = LockstepTargetRealConfig(name="x", cache_dir=Path("/tmp/x"))
    full_token_ids = [0, 1, 2]
    entry = _entry_with_segments(full_token_ids, [
        ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=0, char_end=2),
    ])
    with pytest.raises(ValueError, match="no segments overlap"):
        build_target_for_real_config(
            cfg=cfg, entry=entry,
            target_char_start=10, target_char_end=20,
            full_token_ids=full_token_ids,
            target_token_loss_mask=[0, 1, 1],
            tokenizer=tokenizer,
        )


def test_dispatcher_raises_typeerror_on_unknown_variant():
    """Custom non-LockstepTargetRealConfig/PooledTargetRealConfig subclasses
    must hit the default arm and raise TypeError. This makes the
    default arm in the match statement honest."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeTargetRealConfig:
        name: str
        cache_dir: Path

    cfg = FakeTargetRealConfig(name="fake", cache_dir=Path("/tmp/x"))
    entry = _entry_with_segments(
        (0, 1, 2),
        [ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=0, char_end=2)],
    )
    with pytest.raises(TypeError, match="unknown TargetRealConfig variant"):
        build_target_for_real_config(
            cfg=cfg, entry=entry,  # ty: ignore[invalid-argument-type]
            target_char_start=0, target_char_end=2,
            full_token_ids=[0, 1, 2],
            target_token_loss_mask=[0, 1, 1],
            tokenizer=None,
        )
