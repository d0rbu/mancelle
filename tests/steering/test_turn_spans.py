"""Logical-turn recovery from legacy nanochat conversation renders."""

from __future__ import annotations

import pytest

import steering.sft.turn_spans as turn_spans_module
from nanochat.tokenizer import LogicalTurnSpan
from steering.segmentation import token_to_char_offsets
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    TurnSpanError,
    legacy_turn_spans,
    rendered_text_and_turn_char_starts,
    turn_char_starts,
)
from tests.conftest import get_tokenizer_or_skip
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer

_SHARED_RENDER_CORPUS = [
    {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "calculate"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "python", "text": "1 + 1"},
                    {"type": "python_output", "text": "2"},
                    {"type": "text", "text": "after"},
                ],
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "two calls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": "1 + 1"},
                    {"type": "python_output", "text": "2"},
                    {"type": "text", "text": "then"},
                    {"type": "python", "text": "2 + 2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "done"},
                ],
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "empty first"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "python", "text": "1"},
                    {"type": "python_output", "text": "1"},
                    {"type": "text", "text": "done"},
                ],
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "python first"},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": "3 + 3"},
                    {"type": "python_output", "text": "6"},
                    {"type": "text", "text": "six"},
                ],
            },
        ]
    },
]


def _assert_matches_structured_renderer(tokenizer) -> None:
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    for conversation in _SHARED_RENDER_CORPUS:
        ids, _ = tokenizer.render_conversation(conversation)
        structured = tokenizer.render_structured_conversation(conversation)
        assert ids == list(structured.token_ids)
        assert legacy_turn_spans(ids, specials) == structured.logical_turns


def test_character_tokenizer_spans_match_structured_renderer() -> None:
    _assert_matches_structured_renderer(_CharacterTokenizer())


def test_real_tokenizer_spans_match_structured_renderer() -> None:
    _assert_matches_structured_renderer(get_tokenizer_or_skip())


def test_system_message_uses_legacy_merged_user_span() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    ids, _ = tokenizer.render_conversation(conversation)
    targets = ids[1:]
    assistant_start = targets.index(tokenizer.encode_special("<assistant_start>"))
    assert legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer)) == (
        LogicalTurnSpan("user", 0, assistant_start),
        LogicalTurnSpan("assistant", assistant_start, len(targets)),
    )


def test_python_output_last_gets_trailing_assistant_end_span() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": "1 + 1"},
                    {"type": "python_output", "text": "2"},
                ],
            },
        ]
    }
    ids, _ = tokenizer.render_conversation(conversation)
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    assert [span.role for span in spans] == ["user", "assistant", "tool", "assistant"]
    assert spans[-1] == LogicalTurnSpan("assistant", len(ids) - 2, len(ids) - 1)


def test_back_to_back_tool_outputs_merge_into_one_tool_span() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "python", "text": "1"},
                    {"type": "python_output", "text": "a"},
                    {"type": "python_output", "text": "b"},
                    {"type": "text", "text": "done"},
                ],
            },
        ]
    }
    ids, _ = tokenizer.render_conversation(conversation)
    assert legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer)) == (
        LogicalTurnSpan("user", 0, 3),
        LogicalTurnSpan("assistant", 3, 7),
        LogicalTurnSpan("tool", 7, 13),
        LogicalTurnSpan("assistant", 13, 18),
    )
    with pytest.raises(ValueError, match="orphan native python_output part"):
        tokenizer.render_structured_conversation(conversation)


def test_truncated_render_closes_last_partial_turn() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "a long answer"},
        ]
    }
    full_ids, _ = tokenizer.render_conversation(conversation)
    assistant_position = full_ids.index(tokenizer.encode_special("<assistant_start>"))
    ids, _ = tokenizer.render_conversation(
        conversation,
        max_tokens=assistant_position + 2,
    )
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    assert spans[-1].role == "assistant"
    assert spans[-1].end == len(ids) - 1


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        ([], "must contain a BOS token"),
        ([2, 2], "must start with BOS token"),
        ([1], "must have <user_start> immediately after BOS"),
        ([1, 4], "must have <user_start> immediately after BOS"),
    ],
)
def test_legacy_turn_spans_rejects_invalid_prefix(ids: list[int], match: str) -> None:
    with pytest.raises(TurnSpanError, match=match):
        legacy_turn_spans(ids, ChatSpecialTokens(1, 2, 4, 8, 9))


def test_turn_char_starts_character_offsets_match_hand_calculation() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    ids, _ = tokenizer.render_conversation(conversation)
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    offsets = token_to_char_offsets(tokenizer, ids[1:])
    assert turn_char_starts(spans, offsets) == (
        0,
        len("<user_start>q<user_end>"),
    )


def test_turn_char_starts_real_tokenizer_matches_offsets() -> None:
    tokenizer = get_tokenizer_or_skip()
    conversation = _SHARED_RENDER_CORPUS[2]
    ids, _ = tokenizer.render_conversation(conversation)
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    assert len(spans) == 4
    offsets = token_to_char_offsets(tokenizer, ids[1:])
    expected = tuple(len(tokenizer.decode(ids[1 : 1 + span.start])) for span in spans)
    assert turn_char_starts(spans, offsets) == expected


def test_rendered_text_and_turn_char_starts_uses_exact_bos_free_render() -> None:
    tokenizer = _CharacterTokenizer()
    conversation = _SHARED_RENDER_CORPUS[1]
    ids, _ = tokenizer.render_conversation(conversation)

    text, starts = rendered_text_and_turn_char_starts(ids, tokenizer)

    assert text == tokenizer.decode(ids[1:])
    spans = legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))
    assert starts == tuple(len(tokenizer.decode(ids[1 : span.start + 1])) for span in spans)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ([0], "one boundary per BOS-free token"),
        ([0, 2, 1, 3], "must be non-decreasing"),
    ],
)
def test_rendered_text_and_turn_char_starts_rejects_malformed_offsets(
    monkeypatch: pytest.MonkeyPatch,
    replacement: list[int],
    match: str,
) -> None:
    tokenizer = _CharacterTokenizer()
    ids, _ = tokenizer.render_conversation({"messages": [{"role": "user", "content": "q"}]})
    monkeypatch.setattr(
        turn_spans_module,
        "token_to_char_offsets",
        lambda *_args, **_kwargs: replacement,
    )

    with pytest.raises(TurnSpanError, match=match):
        rendered_text_and_turn_char_starts(ids, tokenizer)


# kills: indexing the offsets inside rendered_text_and_turn_char_starts instead
# of routing them through turn_char_starts. Only the FIRST offset is corrupted,
# so the length, monotonicity and decoded-coverage guards all still pass and
# the rejection can only come from the callee.
def test_rendered_text_and_turn_char_starts_rejects_offsets_not_starting_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTokenizer()
    ids, _ = tokenizer.render_conversation({"messages": [{"role": "user", "content": "q"}]})
    offsets = token_to_char_offsets(tokenizer, ids[1:])
    shifted = [1, *offsets[1:]]
    monkeypatch.setattr(
        turn_spans_module,
        "token_to_char_offsets",
        lambda *_args, **_kwargs: shifted,
    )

    with pytest.raises(TurnSpanError, match=r"char_offsets\[0\] must be 0"):
        rendered_text_and_turn_char_starts(ids, tokenizer)


def test_rendered_text_and_turn_char_starts_rejects_decoded_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTokenizer()
    ids, _ = tokenizer.render_conversation({"messages": [{"role": "user", "content": "q"}]})
    offsets = token_to_char_offsets(tokenizer, ids[1:])
    offsets[-1] += 1
    monkeypatch.setattr(
        turn_spans_module,
        "token_to_char_offsets",
        lambda *_args, **_kwargs: offsets,
    )

    with pytest.raises(TurnSpanError, match="do not cover the exact decoded text"):
        rendered_text_and_turn_char_starts(ids, tokenizer)


@pytest.mark.parametrize(
    ("spans", "char_offsets", "match"),
    [
        ((LogicalTurnSpan("user", 0, 1),), [], "char_offsets must be non-empty"),
        ((LogicalTurnSpan("user", 0, 1),), [3, 7], r"char_offsets\[0\] must be 0"),
        (
            (LogicalTurnSpan("user", 0, 1), LogicalTurnSpan("assistant", 2, 3)),
            [0, 4],
            "logical turn start is outside char_offsets",
        ),
        (
            (LogicalTurnSpan("user", 0, 1), LogicalTurnSpan("assistant", 1, 2)),
            [0, 0, 1],
            "must be strictly increasing",
        ),
    ],
)
def test_turn_char_starts_rejects_invalid_offsets(
    spans: tuple[LogicalTurnSpan, ...],
    char_offsets: list[int],
    match: str,
) -> None:
    with pytest.raises(TurnSpanError, match=match):
        turn_char_starts(spans, char_offsets)


def test_turn_char_starts_rejects_spans_not_starting_at_token_zero() -> None:
    with pytest.raises(TurnSpanError, match="must start at token 0"):
        turn_char_starts(
            (LogicalTurnSpan("user", 2, 4),),
            [0, 1, 2, 3, 4],
        )


def test_turn_char_starts_rejects_empty_spans() -> None:
    with pytest.raises(TurnSpanError, match="must not be empty"):
        turn_char_starts((), [0, 1, 2])


def test_legacy_turn_spans_structural_invariant_raises(monkeypatch) -> None:
    original = turn_spans_module.LogicalTurnSpan

    def shift_first_span(*, role: str, start: int, end: int) -> LogicalTurnSpan:
        return original(role=role, start=start + 1, end=end)

    monkeypatch.setattr(turn_spans_module, "LogicalTurnSpan", shift_first_span)
    with pytest.raises(TurnSpanError, match="must start at token 0"):
        legacy_turn_spans([1, 2, 10], ChatSpecialTokens(1, 2, 4, 8, 9))


def test_chat_special_tokens_from_character_tokenizer() -> None:
    assert ChatSpecialTokens.from_tokenizer(_CharacterTokenizer()) == ChatSpecialTokens(
        bos=1,
        user_start=2,
        assistant_start=4,
        output_start=8,
        output_end=9,
    )
