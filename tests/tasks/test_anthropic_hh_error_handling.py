"""Tests for error handling in anthropic_hh task."""

import pytest

from errors import ParseError


class TestAnthropicHHParse:
    """_parse_hh_conversation must raise ParseError with specific reasons."""

    def test_valid_conversation_parses(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        text = "\n\nHuman: Hello\n\nAssistant: Hi there"
        result = _parse_hh_conversation(text)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_human_prefix_added(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        text = "Human: Hello\n\nAssistant: Hi there"
        result = _parse_hh_conversation(text)
        assert len(result) == 2

    def test_unexpected_format_raises(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        with pytest.raises(ParseError, match="unexpected format"):
            _parse_hh_conversation("Random text with no markers")

    def test_empty_content_raises(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        # Last Assistant turn has no content (nothing after the colon)
        with pytest.raises(ParseError, match="empty message content"):
            _parse_hh_conversation("\n\nHuman: hello\n\nAssistant: hi\n\nHuman: world\n\nAssistant:")

    def test_too_few_messages_raises(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        # Single Human turn with content, but only one message
        with pytest.raises(ParseError, match="fewer than 2 messages"):
            _parse_hh_conversation("\n\nHuman: just one message here")

    def test_wrong_first_role_raises(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        with pytest.raises(ParseError, match="does not start with user"):
            _parse_hh_conversation("\n\nAssistant: hello\n\nHuman: hi\n\nAssistant: hey")

    def test_wrong_last_role_raises(self):
        from tasks.anthropic_hh import _parse_hh_conversation
        with pytest.raises(ParseError, match="does not end with assistant"):
            _parse_hh_conversation("\n\nHuman: hello\n\nAssistant: hi\n\nHuman: bye")
