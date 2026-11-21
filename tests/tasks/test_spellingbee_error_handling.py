"""Tests for error handling in spellingbee task."""

import pytest

from errors import DataLoadError, ParseError


class TestSpellingBeeExtractAnswer:
    """spellingbee.extract_answer must raise ParseError instead of returning None."""

    def test_valid_answer_extracted(self):
        from tasks.spellingbee import extract_answer
        assert extract_answer("The answer is #### 42") == "42"

    def test_comma_stripped(self):
        from tasks.spellingbee import extract_answer
        assert extract_answer("#### 1,234") == "1234"

    def test_no_marker_raises_parse_error(self):
        from tasks.spellingbee import extract_answer
        with pytest.raises(ParseError, match="No answer marker found"):
            extract_answer("There is no marker here")

    def test_empty_string_raises_parse_error(self):
        from tasks.spellingbee import extract_answer
        with pytest.raises(ParseError, match="No answer marker found"):
            extract_answer("")


class TestSpellingBeeEvaluate:
    """spellingbee.evaluate must handle ParseError correctly."""

    @staticmethod
    def _make_conversation(answer_text: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "How many r in strawberry?"},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
            ],
        }

    def test_pred_unparseable_returns_zero(self):
        from tasks.spellingbee import SpellingBee
        task = SpellingBee.__new__(SpellingBee)
        conv = self._make_conversation("#### 3")
        result = task.evaluate(conv, "also no marker")
        assert result == 0

    def test_correct_answer_returns_one(self):
        from tasks.spellingbee import SpellingBee
        task = SpellingBee.__new__(SpellingBee)
        conv = self._make_conversation("#### 3")
        result = task.evaluate(conv, "#### 3")
        assert result == 1

    def test_reference_unparseable_raises(self):
        from tasks.spellingbee import SpellingBee
        task = SpellingBee.__new__(SpellingBee)
        conv = self._make_conversation("broken reference")
        with pytest.raises(DataLoadError, match="Ground truth"):
            task.evaluate(conv, "#### 3")
