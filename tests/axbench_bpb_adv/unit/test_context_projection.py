from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.eval.axbench_bpb_adv.context import project_fixed_context_segments
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.target_conditioning import ContextFeatureSegment


class _CharTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


class _VisibleBosTokenizer:
    bos_token_id = 0

    def decode(self, ids: list[int]) -> str:
        return "".join(
            "<BOS>" if int(token_id) == self.bos_token_id else chr(int(token_id))
            for token_id in ids
        )


class _Issue737FailureShapeTokenizer:
    """Compact replay of the real 2192/2197 clustered-cache failure geometry."""

    _WIDTHS = {
        0: 5,
        1: 582,
        2: 627,
        3: 668,
        4: 311,
        5: 4,
        11: 582,
        12: 627,
        13: 668,
        14: 311,
        15: 4,
    }

    def decode(self, ids: list[int]) -> str:
        return "".join("x" * self._WIDTHS[int(token_id)] for token_id in ids)


_ISSUE_728_REPLAY = (
    Path(__file__).parents[1] / "fixtures" / "issue_728_projector_replay.json"
)


def _segment(
    feature_ids: tuple[int, ...],
    *,
    char_start: int,
    char_end: int,
) -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=feature_ids,
        buckets=tuple(1 for _ in feature_ids),
        char_start=char_start,
        char_end=char_end,
    )


def test_issue_728_saved_projector_replay_matches_pre_regression_oracle() -> None:
    payload = json.loads(_ISSUE_728_REPLAY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["cases"]) == 32
    assert sum(case["source_tokens"][1:].count(0) > 0 for case in payload["cases"]) == 10

    for case in payload["cases"]:
        source_segments = tuple(
            ContextFeatureSegment(
                feature_ids=tuple(raw["feature_ids"]),
                buckets=tuple(raw["buckets"]),
                char_start=raw["char_start"],
                char_end=raw["char_end"],
            )
            for raw in case["source_segments"]
        )
        projected = project_fixed_context_segments(
            tokenizer=_VisibleBosTokenizer(),
            source_tokens=case["source_tokens"],
            attacked_tokens=case["attacked_tokens"],
            source_segments=source_segments,
        )
        assert [segment.feature_ids for segment in projected] == [
            tuple(raw["feature_ids"]) for raw in case["source_segments"]
        ], case["id"]
        assert [segment.buckets for segment in projected] == [
            tuple(raw["buckets"]) for raw in case["source_segments"]
        ], case["id"]
        assert [(segment.char_start, segment.char_end) for segment in projected] == [
            tuple(span) for span in case["expected_projected_spans"]
        ], case["id"]


def test_fixed_context_preserves_features_and_projects_attacked_char_spans() -> None:
    source = (
        _segment((1, 2), char_start=0, char_end=2),
        _segment((3,), char_start=2, char_end=4),
    )

    out = project_fixed_context_segments(
        tokenizer=_CharTokenizer(),
        source_tokens=[97, 98, 99, 100],
        attacked_tokens=[97, 98, 120, 121],
        source_segments=source,
    )

    assert [segment.feature_ids for segment in out] == [(1, 2), (3,)]
    assert [segment.buckets for segment in out] == [(1, 1), (1,)]
    assert [(segment.char_start, segment.char_end) for segment in out] == [(0, 2), (2, 4)]


def test_fixed_context_projects_bos_inclusive_segments_in_visible_bos_coordinates() -> None:
    source = (_segment((1,), char_start=0, char_end=len("<BOS>ab")),)

    out = project_fixed_context_segments(
        tokenizer=_VisibleBosTokenizer(),
        source_tokens=[0, 97, 98],
        attacked_tokens=[0, 120, 121],
        source_segments=source,
    )

    assert [segment.feature_ids for segment in out] == [(1,)]
    assert [(segment.char_start, segment.char_end) for segment in out] == [
        (0, len("<BOS>xy"))
    ]


def test_fixed_context_bos_position_shift_keeps_later_segments_aligned() -> None:
    source = (
        _segment((1,), char_start=0, char_end=len("<BOS>ab")),
        _segment((2,), char_start=len("<BOS>ab"), char_end=len("<BOS>abcd")),
    )

    out = project_fixed_context_segments(
        tokenizer=_VisibleBosTokenizer(),
        source_tokens=[0, 97, 98, 99, 100],
        attacked_tokens=[0, 120, 121, 109, 110],
        source_segments=source,
    )

    assert [segment.feature_ids for segment in out] == [(1,), (2,)]
    assert [(segment.char_start, segment.char_end) for segment in out] == [
        (0, len("<BOS>xy")),
        (len("<BOS>xy"), len("<BOS>xymn")),
    ]


def test_issue_737_real_failure_geometry_requires_bos_inclusive_final_bound() -> None:
    tokenizer = _Issue737FailureShapeTokenizer()
    source_tokens = [0, 1, 2, 3, 4, 5]
    attacked_tokens = [0, 11, 12, 13, 14, 15]
    source = (
        _segment((1,), char_start=0, char_end=587),
        _segment((2,), char_start=587, char_end=1214),
        _segment((3,), char_start=1214, char_end=1882),
        _segment((4,), char_start=1882, char_end=2197),
    )

    projected = project_fixed_context_segments(
        tokenizer=tokenizer,
        source_tokens=source_tokens,
        attacked_tokens=attacked_tokens,
        source_segments=source,
    )
    assert [(segment.char_start, segment.char_end) for segment in projected] == [
        (0, 587),
        (587, 1214),
        (1214, 1882),
        (1882, 2197),
    ]

    stale = (*source[:-1], _segment((4,), char_start=1882, char_end=2192))
    with pytest.raises(ValueError, match=r"\[2193, 2197\).+\[0, 2192\)"):
        project_fixed_context_segments(
            tokenizer=tokenizer,
            source_tokens=source_tokens,
            attacked_tokens=attacked_tokens,
            source_segments=stale,
        )


def test_fixed_context_rejects_bos_stripped_segments_with_visible_bos_tokens() -> None:
    with pytest.raises(ValueError, match="zero overlap"):
        project_fixed_context_segments(
            tokenizer=_VisibleBosTokenizer(),
            source_tokens=[0, 97, 98],
            attacked_tokens=[0, 120, 121],
            source_segments=(_segment((1,), char_start=0, char_end=2),),
        )


# ---------------------------------------------------------------------------
# project_fixed_context_segments defensive guards. These are the SOLE surviving
# projection after the recomputed_context removal (#506); its guards must hold.
# ---------------------------------------------------------------------------


def test_fixed_context_rejects_length_changing_edit() -> None:
    # Non-length-preserving token edits violate the fixed-projection contract.
    with pytest.raises(
        AxBenchBpbAdvError, match="length-preserving token edits"
    ):
        project_fixed_context_segments(
            tokenizer=_CharTokenizer(),
            source_tokens=[97, 98],
            attacked_tokens=[97],
            source_segments=(_segment((1,), char_start=0, char_end=2),),
        )


def test_fixed_context_rejects_empty_source_segments() -> None:
    with pytest.raises(
        AxBenchBpbAdvError, match="non-empty source segments"
    ):
        project_fixed_context_segments(
            tokenizer=_CharTokenizer(),
            source_tokens=[97, 98],
            attacked_tokens=[97, 98],
            source_segments=(),
        )


def test_fixed_context_rejects_zero_width_segment_at_prompt_end() -> None:
    # The trailing source token (98 -> "b") is replaced by an empty-decoding
    # token (id 0 -> ""), length-preserving. The second source segment owns that
    # trailing token, so its projected span collapses to zero width AT THE END of
    # the decoded attacked prompt (decoded_len <= start) -> the end-of-prompt raise.
    source = (
        _segment((1,), char_start=0, char_end=1),
        _segment((2,), char_start=1, char_end=2),
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="zero-width segment at end of prompt",
    ):
        project_fixed_context_segments(
            tokenizer=_CharTokenizer(),
            source_tokens=[97, 98],
            attacked_tokens=[97, 0],
            source_segments=source,
        )


def test_fixed_context_recovers_interior_zero_width_then_detects_overlap() -> None:
    # An INTERIOR source token (98 -> "b" at position 2) is replaced by an
    # empty-decoding token (id 0 -> ""), length-preserving. The middle source
    # segment owns only that now-empty token, so its projected span is zero width
    # but NOT at the prompt end (decoded_len > start) -> the recovery branch sets
    # end = start + 1. That recovered right edge then overlaps the following
    # segment (whose offsets collapsed onto the same char), so projection raises
    # the overlap guard. Both the recovery and overlap branches execute here.
    source = (
        _segment((1,), char_start=0, char_end=1),
        _segment((2,), char_start=1, char_end=2),
        _segment((3,), char_start=2, char_end=3),
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="overlapping segments",
    ):
        project_fixed_context_segments(
            tokenizer=_CharTokenizer(),
            source_tokens=[97, 98, 99],
            attacked_tokens=[97, 0, 99],
            source_segments=source,
        )
