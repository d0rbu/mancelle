"""Pin that ``tasks.anthropic_hh._parse_hh_conversation`` produces correct
output regardless of the orphan ``re.split`` call at line 43 (P3 #2).

The orphan ``re.split`` call is dead code — its return value is discarded.
Removing it must not change any observable behaviour.
"""

from __future__ import annotations

import pytest

from errors import ParseError
from tasks.anthropic_hh import _parse_hh_conversation


def test_basic_two_turn_conversation_starts_without_double_newline():
    text = "Human: hi\n\nAssistant: hello"
    msgs = _parse_hh_conversation(text)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_basic_two_turn_with_leading_double_newline():
    text = "\n\nHuman: hi\n\nAssistant: hello"
    msgs = _parse_hh_conversation(text)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_multi_turn_alternation():
    text = "Human: q1\n\nAssistant: a1\n\nHuman: q2\n\nAssistant: a2"
    msgs = _parse_hh_conversation(text)
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2"]


def test_unexpected_format_raises_parse_error():
    with pytest.raises(ParseError, match=r"unexpected format"):
        _parse_hh_conversation("not a hh format")


def test_existing_anthropic_hh_error_handling_still_passes():
    # Smoke check: any structurally invalid input raises ParseError.
    # (The exact error is downstream of the regex; we don't pin it here
    # because the regex's whitespace-matching swallows newline separators
    # in some pathological inputs — out of scope for slice 6.)
    with pytest.raises(ParseError):
        _parse_hh_conversation("Random text with no human/assistant markers")


def test_starts_with_assistant_raises_first_role_check():
    # A conversation that starts with "Assistant:" violates the
    # "first message must be user" rule.
    text = "Assistant: hi\n\nHuman: hello"
    with pytest.raises(ParseError, match=r"does not start with user"):
        _parse_hh_conversation(text)


def test_ends_with_human_raises_last_role_check():
    text = "Human: q1\n\nAssistant: a1\n\nHuman: q2"
    with pytest.raises(ParseError, match=r"does not end with assistant"):
        _parse_hh_conversation(text)
