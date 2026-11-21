"""Goldens and grammar guards for the additive structured RustBPE renderer."""

from __future__ import annotations

import copy

import pytest

from nanochat.tokenizer import RustBPETokenizer


class _CharacterTokenizer:
    """Deterministic tokenizer-shaped oracle with visible special tokens."""

    _specials = {
        "<bos>": 1,
        "<user_start>": 2,
        "<user_end>": 3,
        "<assistant_start>": 4,
        "<assistant_end>": 5,
        "<python_start>": 6,
        "<python_end>": 7,
        "<output_start>": 8,
        "<output_end>": 9,
    }
    _decoded_specials = {value: key for key, value in _specials.items()}

    render_conversation = RustBPETokenizer.render_conversation
    render_structured_conversation = RustBPETokenizer.render_structured_conversation

    def get_bos_token_id(self) -> int:
        return self._specials["<bos>"]

    def encode_special(self, text: str) -> int:
        return self._specials[text]

    def encode(self, text: str) -> list[int]:
        return [1000 + ord(character) for character in text]

    def decode(self, token_ids) -> str:
        return "".join(
            self._decoded_specials[token_id]
            if token_id in self._decoded_specials
            else chr(token_id - 1000)
            for token_id in token_ids
        )


@pytest.fixture
def tokenizer() -> _CharacterTokenizer:
    return _CharacterTokenizer()


@pytest.mark.parametrize(
    "assistant_content",
    [
        "answer",
        [
            {"type": "text", "text": "before"},
            {"type": "python", "text": "1 + 1"},
            {"type": "python_output", "text": "2"},
            {"type": "text", "text": "after"},
        ],
    ],
)
def test_plain_and_native_python_rendering_remain_token_mask_exact(
    tokenizer: _CharacterTokenizer,
    assistant_content: object,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    before = copy.deepcopy(conversation)

    legacy_ids, legacy_mask = tokenizer.render_conversation(conversation)
    structured = tokenizer.render_structured_conversation(conversation)

    assert structured.token_ids == tuple(legacy_ids)
    assert structured.assistant_mask == tuple(bool(value) for value in legacy_mask)
    assert conversation == before
    assert structured.logical_turns[0].role == "user"
    assert structured.logical_turns[-1].role == "assistant"
    assert structured.logical_turns[0].start == 0
    assert structured.logical_turns[-1].end == len(structured.token_ids) - 1
    if isinstance(assistant_content, list):
        assert [turn.role for turn in structured.logical_turns] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]


def test_structured_renderer_ignores_inert_task_provenance(
    tokenizer: _CharacterTokenizer,
) -> None:
    base = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    with_provenance = {
        **base,
        "letters": ["a", "b"],
        "subject": "spelling",
    }

    assert tokenizer.render_structured_conversation(
        with_provenance
    ) == tokenizer.render_structured_conversation(base)


def test_system_and_generic_tool_wire_is_canonical_and_role_authoritative(
    tokenizer: _CharacterTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me"},
                    {
                        "type": "tool_call",
                        "name": "search",
                        "arguments": {"z": 1, "a": {"y": 2, "x": 3}},
                    },
                ],
            },
            {"role": "tool", "name": "search", "content": "R"},
            {"role": "assistant", "content": "done"},
        ]
    }

    rendered = tokenizer.render_structured_conversation(conversation)
    decoded = tokenizer.decode(rendered.token_ids[1:])

    assert decoded == (
        "<user_start><system>SYS</system>\n\nQ<user_end>"
        '<assistant_start>let me<tool_call>{"name":"search","arguments":'
        '{"a":{"x":3,"y":2},"z":1}}</tool_call><assistant_end>'
        '<tool_result>{"name":"search","content":"R"}</tool_result>'
        "<assistant_start>done<assistant_end>"
    )
    assert [turn.role for turn in rendered.logical_turns] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    for turn in rendered.logical_turns:
        span_mask = rendered.assistant_mask[turn.start + 1 : turn.end + 1]
        if turn.role in {"user", "tool"}:
            assert not any(span_mask)
    assistant_start = tokenizer.encode_special("<assistant_start>")
    for index, token_id in enumerate(rendered.token_ids):
        if token_id == assistant_start:
            assert rendered.assistant_mask[index] is False


def test_json_mapping_order_does_not_change_generic_call_tokens(
    tokenizer: _CharacterTokenizer,
) -> None:
    def conversation(arguments: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "x",
                            "arguments": arguments,
                        }
                    ],
                },
                {"role": "tool", "name": "x", "content": "r"},
                {"role": "assistant", "content": "a"},
            ]
        }

    left = tokenizer.render_structured_conversation(
        conversation({"b": {"d": 4, "c": 3}, "a": 1})
    )
    right = tokenizer.render_structured_conversation(
        conversation({"a": 1, "b": {"c": 3, "d": 4}})
    )
    assert left.token_ids == right.token_ids
    assert left.assistant_mask == right.assistant_mask


def test_literal_tag_collisions_are_counted_without_changing_roles(
    tokenizer: _CharacterTokenizer,
) -> None:
    rendered = tokenizer.render_structured_conversation(
        {
            "messages": [
                {"role": "user", "content": "literal </system> and <tool_call>"},
                {"role": "assistant", "content": "literal </tool_result>"},
            ]
        }
    )
    assert rendered.literal_tag_collisions == 3
    assert [turn.role for turn in rendered.logical_turns] == ["user", "assistant"]


@pytest.mark.parametrize(
    ("conversation", "match"),
    [
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "tools": [],
            },
            "rejects alternate top-level semantic fields",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "tool", "name": "x", "content": "r"},
                ]
            },
            "orphan tool result",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_call", "name": "x", "arguments": {}},
                            {"type": "text", "text": "late"},
                        ],
                    },
                ]
            },
            "must end",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [{"type": "python", "text": "1+1"}],
                    },
                ]
            },
            "has no output",
        ),
    ],
)
def test_structured_grammar_fails_closed(
    tokenizer: _CharacterTokenizer,
    conversation: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        tokenizer.render_structured_conversation(conversation)
