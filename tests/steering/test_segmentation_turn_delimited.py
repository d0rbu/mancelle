"""Turn-delimited segmentation and shared GRPO chunking regressions."""

from __future__ import annotations

import itertools
import subprocess
import sys
from bisect import bisect_right

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from nanochat.tokenizer import LogicalTurnSpan
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    _segment_fixed,
    segment_document,
    segment_tokens,
    segment_turn_delimited,
    turn_token_starts,
)
from steering.sft import structured_rollout
from steering.sft.structured_rollout import TargetSegment


class _CharacterTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.parametrize(
    ("turn_starts", "n", "segment_size", "expected"),
    [
        ([0, 3, 11], 13, 4, [(0, 3), (3, 7), (7, 11), (11, 13)]),
        ([0, 4, 12], 16, 4, [(0, 4), (4, 8), (8, 12), (12, 16)]),
        ([0, 2], 7, 8, [(0, 2), (2, 7)]),
        ([0, 2], 5, 1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    ],
)
def test_segment_turn_delimited_cases(
    turn_starts: list[int],
    n: int,
    segment_size: int,
    expected: list[tuple[int, int]],
) -> None:
    assert segment_turn_delimited(turn_starts, n, segment_size) == expected


@pytest.mark.parametrize(
    ("turn_starts", "n", "match"),
    [
        ([], 4, "turn_starts must be non-empty"),
        ([1], 4, r"turn_starts\[0\] must be 0"),
        ([0, 2, 2], 4, "turn_starts must be strictly increasing"),
        ([0, 3, 2], 4, "turn_starts must be strictly increasing"),
        ([0, 4], 4, "turn_starts values must be less than n"),
    ],
)
def test_segment_turn_delimited_rejects_invalid_starts(
    turn_starts: list[int], n: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        segment_turn_delimited(turn_starts, n, 2)


@pytest.mark.parametrize(
    ("char_offsets", "turn_char_starts", "match"),
    [
        ([], [0], "^char_offsets must be non-empty$"),
        ([1, 4], [0], r"^char_offsets\[0\] must be 0$"),
        ([0, 5, 3, 9], [0, 2], "^char_offsets must be non-decreasing$"),
        ([0, 5, 9], [], "^turn_char_starts must be non-empty$"),
        ([0, 5, 9], [2], r"^turn_char_starts\[0\] must be 0$"),
        ([0, 5, 9], [0, 4, 4], "^turn_char_starts must be strictly increasing$"),
        (
            [0, 5, 10, 15],
            [0, 12, 6],
            "^turn_char_starts must be strictly increasing$",
        ),
        (
            [0, 5, 10, 15],
            [0, 7, 4],
            "^turn_char_starts must be strictly increasing$",
        ),
        (
            [0, 5, 10, 15],
            [0, 100, 6],
            "^turn_char_starts must be strictly increasing$",
        ),
    ],
)
def test_turn_token_starts_rejects_invalid_inputs(
    char_offsets: list[int],
    turn_char_starts: list[int],
    match: str,
) -> None:
    """Pin the producer invariant tested by test_monotonically_nondecreasing.

    Flat runs remain legal, as test_turn_token_starts_uses_first_offset_in_flat_run
    demonstrates.
    """
    with pytest.raises(ValueError, match=match):
        turn_token_starts(char_offsets, turn_char_starts, bos_char_len=0)


@pytest.mark.parametrize("n", range(1, 201))
@pytest.mark.parametrize("segment_size", [1, 2, 3, 5, 7, 16, 64])
def test_single_turn_matches_fixed_segmentation(n: int, segment_size: int) -> None:
    assert segment_turn_delimited([0], n, segment_size) == _segment_fixed(
        n, PositiveInt.parse(segment_size)
    )


@pytest.mark.parametrize(
    ("char_offsets", "turn_char_starts", "bos_char_len", "expected"),
    [
        pytest.param(
            [0, 5, 8, 12, 16],
            [0, 4],
            5,
            [0, 3],
            id="straddling_token_to_previous_turn",
        ),
        pytest.param([0, 5, 8, 12], [0, 3], 5, [0, 2], id="exact_boundary"),
        pytest.param(
            [0, 5, 8, 12],
            [0, 20],
            5,
            [0],
            id="drops_boundary_beyond_kept_tokens",
        ),
        pytest.param(
            [0, 5, 8, 12],
            [0, 3, 7],
            5,
            [0, 2],
            id="drops_at_exactly_window_end",
        ),
        pytest.param(
            [0, 3, 7],
            [0, 5],
            0,
            [0],
            id="drops_boundary_straddling_final_token",
        ),
        pytest.param(
            [0, 5, 8, 12, 16],
            [0, 3, 7],
            5,
            [0, 2, 3],
            id="absorbs_bos_into_first_turn",
        ),
        pytest.param(
            [0, 3, 3, 3, 7, 9],
            [0, 3],
            0,
            [0, 1],
            id="uses_first_offset_in_flat_run",
        ),
    ],
)
def test_turn_token_starts_cases(
    char_offsets: list[int],
    turn_char_starts: list[int],
    bos_char_len: int,
    expected: list[int],
) -> None:
    starts = turn_token_starts(char_offsets, turn_char_starts, bos_char_len=bos_char_len)
    assert starts == expected
    if char_offsets == [0, 3, 3, 3, 7, 9]:
        assert char_offsets[starts[1] - 1] < 3 <= char_offsets[starts[1]]


@pytest.mark.parametrize(
    ("char_offsets", "turn_char_starts", "match"),
    [
        pytest.param(
            [0, 5, 10, 15],
            [0, 3, 4],
            "turn 2 received zero tokens",
            id="rejects_zero_token_turn",
        ),
        pytest.param(
            [0, 5, 10, 15],
            [0, 6, 7],
            "turn 2 received zero tokens",
            id="rejects_two_boundaries_at_same_token",
        ),
    ],
)
def test_turn_token_starts_rejects(
    char_offsets: list[int], turn_char_starts: list[int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        turn_token_starts(char_offsets, turn_char_starts, bos_char_len=0)


@st.composite
def _aligned_boundaries(draw):
    widths = draw(st.lists(st.integers(0, 8), min_size=2, max_size=30))
    widths[-1] = draw(st.integers(1, 8))
    offsets = [0, *itertools.accumulate(widths)]
    bos_char_len = draw(st.integers(0, 6))
    usable_token_indices = [
        index
        for index in range(1, len(widths))
        if offsets[index - 1] < offsets[index] and offsets[index] > bos_char_len
    ]
    assume(usable_token_indices)
    token_indices = draw(
        st.lists(
            st.sampled_from(usable_token_indices),
            min_size=1,
            unique=True,
            max_size=min(8, len(usable_token_indices)),
        ).map(sorted)
    )
    boundaries = [
        draw(st.integers(max(offsets[index - 1] + 1, bos_char_len + 1), offsets[index]))
        for index in token_indices
    ]
    turn_char_starts = [boundary - bos_char_len for boundary in boundaries]
    trailing_boundary = draw(
        st.integers(
            max(
                offsets[-2] + 1,
                boundaries[-1] + 1 if boundaries else 0,
                bos_char_len + 1,
            ),
            offsets[-1] + 1,
        )
    )
    trailing_turn_char_start = trailing_boundary - bos_char_len
    return (
        offsets,
        bos_char_len,
        token_indices,
        boundaries,
        [0, *turn_char_starts, trailing_turn_char_start],
    )


@given(case=_aligned_boundaries(), segment_size=st.integers(1, 12))
def test_turn_token_starts_exactness_property(case, segment_size: int) -> None:
    char_offsets, bos_char_len, expected_token_starts, boundaries, turn_char_starts = case
    starts = turn_token_starts(
        char_offsets,
        turn_char_starts,
        bos_char_len=bos_char_len,
    )
    assert starts == [0, *expected_token_starts]
    assert len(starts) < len(turn_char_starts)

    for token_start, boundary in zip(starts[1:], boundaries, strict=True):
        assert char_offsets[token_start] >= boundary
        assert char_offsets[token_start - 1] < boundary

    windows = segment_turn_delimited(starts, len(char_offsets) - 1, segment_size)
    turn_boundaries = [0, *boundaries]
    for window_start, _ in windows:
        turn_index = bisect_right(starts, window_start) - 1
        assert char_offsets[window_start] >= turn_boundaries[turn_index]
        if turn_index + 1 < len(turn_boundaries):
            assert char_offsets[window_start] < turn_boundaries[turn_index + 1]


def _legacy_segment_tokens(
    token_ids: list[int], segment_size: int, token_offset: int
) -> list[tuple[int, int]]:
    n = len(token_ids)
    assert n > 0
    assert token_offset < n
    return list(itertools.pairwise(list(range(token_offset, n, segment_size)) + [n]))


@given(
    n=st.integers(1, 300),
    segment_size=st.integers(1, 80),
    data=st.data(),
)
def test_segment_tokens_fixed_path_is_unchanged(
    n: int, segment_size: int, data: st.DataObject
) -> None:
    token_offset = data.draw(st.integers(0, n - 1))
    ids = list(range(n))
    assert segment_tokens(
        ids,
        SegmentMode.fixed_tokens,
        segment_size,
        token_offset,
    ) == _legacy_segment_tokens(ids, segment_size, token_offset)


def test_segment_dispatch_contracts() -> None:
    ids = [1000, 1001, 1002, 1003]
    assert segment_tokens(
        ids,
        SegmentMode.turn_delimited_fixed_tokens,
        2,
        turn_starts=[0, 3],
    ) == [(0, 2), (2, 3), (3, 4)]

    with pytest.raises(ValueError, match="requires turn_starts"):
        segment_tokens(ids, SegmentMode.turn_delimited_fixed_tokens, 2)
    with pytest.raises(ValueError, match="requires token_offset == 0"):
        segment_tokens(
            ids,
            SegmentMode.turn_delimited_fixed_tokens,
            2,
            1,
            turn_starts=[0, 3],
        )
    with pytest.raises(ValueError, match="requires token_offset == 0"):
        segment_tokens(
            ids,
            SegmentMode.turn_delimited_fixed_tokens,
            2,
            len(ids),
            turn_starts=[0, 3],
        )
    with pytest.raises(ValueError, match="does not accept turn_starts"):
        segment_tokens(ids, SegmentMode.fixed_tokens, 2, turn_starts=[0])


def test_segment_document_dispatches_turn_starts() -> None:
    segments, char_starts, char_ends = segment_document(
        [1000, 1001, 1002, 1003],
        _CharacterTokenizer(),
        SegmentMode.turn_delimited_fixed_tokens,
        PositiveInt.parse(2),
        turn_starts=[0, 3],
    )
    assert segments == [(0, 2), (2, 3), (3, 4)]
    assert char_starts == [0, 2, 3]
    assert char_ends == [2, 3, 4]


# Verbatim oracle bodies from steering/sft/structured_rollout.py at a18873736.
def _legacy_fixed_segments(length: int, segment_length: int) -> tuple[TargetSegment, ...]:
    return tuple(
        TargetSegment(
            index=index,
            start=start,
            end=min(start + segment_length, length),
            role=None,
        )
        for index, start in enumerate(range(0, length, segment_length))
    )


def _legacy_turn_segments(
    turns: list[LogicalTurnSpan],
    segment_length: int,
) -> tuple[TargetSegment, ...]:
    result = []
    for turn in turns:
        for start in range(turn.start, turn.end, segment_length):
            result.append(
                TargetSegment(
                    index=len(result),
                    start=start,
                    end=min(start + segment_length, turn.end),
                    role=turn.role,
                )
            )
    return tuple(result)


@st.composite
def _logical_turn_layouts(draw):
    lengths = draw(st.lists(st.integers(1, 600), min_size=1, max_size=8))
    roles = draw(
        st.lists(
            st.sampled_from(["user", "assistant", "tool"]),
            min_size=len(lengths),
            max_size=len(lengths),
        )
    )
    starts = [0, *itertools.accumulate(lengths)]
    return [
        LogicalTurnSpan(role, starts[index], starts[index + 1]) for index, role in enumerate(roles)
    ]


@given(
    turns=_logical_turn_layouts(),
    segment_length=st.integers(1, 300),
    fixed_length=st.integers(0, 600),
)
def test_grpo_segment_helpers_match_legacy_oracles(
    turns: list[LogicalTurnSpan], segment_length: int, fixed_length: int
) -> None:
    assert structured_rollout._turn_segments(turns, segment_length) == _legacy_turn_segments(
        turns, segment_length
    )
    assert structured_rollout._fixed_segments(
        fixed_length, segment_length
    ) == _legacy_fixed_segments(fixed_length, segment_length)


@pytest.mark.parametrize(
    ("turns", "target_length", "message"),
    [
        (
            [LogicalTurnSpan("user", 1, 2)],
            2,
            r"logical_turns\[0\] does not tile from cursor 0: \[1, 2\)",
        ),
        (
            [LogicalTurnSpan("user", 0, 1), LogicalTurnSpan("assistant", 1, 1)],
            1,
            r"logical_turns\[1\] does not tile from cursor 1: \[1, 1\)",
        ),
        (
            [LogicalTurnSpan("user", 0, 3), LogicalTurnSpan("assistant", 2, 4)],
            4,
            r"logical_turns\[1\] does not tile from cursor 3: \[2, 4\)",
        ),
        (
            [LogicalTurnSpan("user", 0, 0)],
            1,
            r"logical_turns\[0\] does not tile from cursor 0: \[0, 0\)",
        ),
    ],
)
def test_build_conversation_environment_rejects_invalid_tilings(
    turns: list[LogicalTurnSpan],
    target_length: int,
    message: str,
) -> None:
    with pytest.raises(structured_rollout.StructuredRolloutError, match=message):
        structured_rollout.build_conversation_environment(
            structured_rollout.StructuredRolloutRow(
                target_token_ids=tuple(range(target_length)),
                target_mask=(True,) * target_length,
                logical_turns=tuple(turns),
                text="",
                original_index=0,
            ),
            bos_token_id=0,
            context_length=64,
            segment_length=2,
            segmentation_mode=structured_rollout.SegmentationMode.TURN_DELIMITED_FIXED_LENGTH,
        )


def test_turn_span_modules_import_without_cycle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import steering.sft.structured_rollout; "
            "import steering.sft.turn_spans; "
            "assert 'steering.eval.axbench_base.profiler' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
