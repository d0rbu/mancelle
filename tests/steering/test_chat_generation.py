"""Contract tests for the shared chat-generation seams (issue #1336).

These tests deliberately exercise the project-owned helper rather than
reimplementing prompt logic in each consumer. The tokenizer tests pin exact
render parity and fail-loud capability errors.
"""

from __future__ import annotations

from typing import Any

import pytest

from steering.eval.chat_generation import (
    ChatCompletionPrompt,
    empty_assistant_conversation,
    render_chat_completion,
    render_prompt_for_chat_completion,
    require_chat_render_support,
)
from tests.conftest import MockTokenizer


class _RecordingTokenizer:
    """Deterministic chat tokenizer with explicit call/aliasing observability."""

    def __init__(self) -> None:
        self.rendered: list[object] = []

    def render_for_completion(self, conversation: object) -> list[int]:
        self.rendered.append(conversation)
        return [101, 202, 303]

    def encode_special(self, token: str) -> int:
        assert token == "<assistant_end>"
        return 909


class _GeneratorTokenizer(_RecordingTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.ids = [11, 22, 33]

    def render_for_completion(self, conversation: object):
        self.rendered.append(conversation)
        yield from self.ids


def test_empty_assistant_conversation_has_canonical_shape_and_fresh_state() -> None:
    first = empty_assistant_conversation("first")
    second = empty_assistant_conversation("second")

    assert first == {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},
        ]
    }
    assert second["messages"][0]["content"] == "second"
    assert first is not second
    assert first["messages"] is not second["messages"]
    first["messages"][0]["content"] = "mutated"
    assert second["messages"][0]["content"] == "second"


@pytest.mark.parametrize("bad_prompt", [None, 1, b"bytes", object()])
def test_empty_assistant_conversation_rejects_non_string_prompt(bad_prompt: Any) -> None:
    with pytest.raises(TypeError, match="chat completion prompt must be str"):
        empty_assistant_conversation(bad_prompt)


def test_render_prompt_uses_empty_assistant_and_returns_exact_rendered_ids() -> None:
    tokenizer = _RecordingTokenizer()

    rendered = render_prompt_for_chat_completion(tokenizer, "hello")

    assert isinstance(rendered, ChatCompletionPrompt)
    assert rendered.token_ids == [101, 202, 303]
    assert rendered.assistant_end_id == 909
    assert len(tokenizer.rendered) == 1
    assert tokenizer.rendered[0] == empty_assistant_conversation("hello")


def test_render_materializes_generator_and_owns_a_fresh_token_id_list() -> None:
    tokenizer = _GeneratorTokenizer()

    first = render_prompt_for_chat_completion(tokenizer, "one")
    second = render_prompt_for_chat_completion(tokenizer, "two")

    assert first.token_ids == [11, 22, 33]
    assert second.token_ids == [11, 22, 33]
    assert first.token_ids is not second.token_ids
    first.token_ids.append(44)
    assert second.token_ids == [11, 22, 33]


def test_render_populated_task_conversation_preserves_tokenizer_parity() -> None:
    tokenizer = MockTokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "metadata": {"answer": "4"},
    }
    expected = tokenizer.render_for_completion(conversation)

    rendered = render_chat_completion(tokenizer, conversation)

    assert rendered.token_ids == expected
    assert rendered.assistant_end_id == tokenizer.encode_special("<assistant_end>")
    # RustBPE/MockTokenizer render_for_completion deep-copies and pops its
    # final assistant message; the helper must not mutate task truth/metadata.
    assert conversation["messages"][-1]["content"] == "4"
    assert conversation["metadata"] == {"answer": "4"}


@pytest.mark.parametrize(
    ("tokenizer", "match"),
    [
        (object(), "render_for_completion and encode_special"),
        (type("NoSpecial", (), {"render_for_completion": lambda self, _: []})(), "render_for_completion and encode_special"),
        (type("NoRender", (), {"encode_special": lambda self, _: 1})(), "render_for_completion and encode_special"),
    ],
)
def test_chat_capability_gate_is_fail_loud(tokenizer: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        require_chat_render_support(tokenizer)
    with pytest.raises(ValueError, match=match):
        render_prompt_for_chat_completion(tokenizer, "prompt")


class _BadEndTokenizer(_RecordingTokenizer):
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        super().__init__()
        self.result = result
        self.error = error

    def encode_special(self, token: str) -> Any:
        assert token == "<assistant_end>"
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("bad_id", [None, True, False, "909", 9.0, [], {}])
def test_render_rejects_missing_or_non_integer_assistant_end(bad_id: Any) -> None:
    with pytest.raises(ValueError, match="requires tokenizer.encode_special") as exc:
        render_prompt_for_chat_completion(_BadEndTokenizer(result=bad_id), "prompt")
    assert "assistant-end stop" in str(exc.value)


def test_render_preserves_assistant_end_exception_chain() -> None:
    original = RuntimeError("vocabulary lookup failed")
    with pytest.raises(ValueError, match="RuntimeError: vocabulary lookup failed") as exc:
        render_prompt_for_chat_completion(_BadEndTokenizer(error=original), "prompt")
    assert exc.value.__cause__ is original


@pytest.mark.parametrize(
    "original",
    [ValueError("bad token"), TypeError("bad type"), AttributeError("missing"), KeyError("end")],
)
def test_render_wraps_all_established_assistant_end_lookup_errors(
    original: Exception,
) -> None:
    with pytest.raises(ValueError) as exc:
        render_prompt_for_chat_completion(
            _BadEndTokenizer(error=original),
            "prompt",
        )
    assert exc.value.__cause__ is original
    assert type(original).__name__ in str(exc.value)


def test_render_does_not_hide_unrelated_renderer_errors() -> None:
    class BrokenRenderer(_RecordingTokenizer):
        def render_for_completion(self, conversation: object) -> list[int]:
            raise AssertionError("renderer invariant")

    with pytest.raises(AssertionError, match="renderer invariant"):
        render_prompt_for_chat_completion(BrokenRenderer(), "prompt")
