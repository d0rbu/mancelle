"""Tests for error handling in task loaders."""

import pytest

from errors import DataLoadError, ParseError


class TestGSM8KExtractAnswer:
    """gsm8k.extract_answer must raise ParseError instead of returning None."""

    def test_valid_answer_extracted(self):
        from tasks.gsm8k import extract_answer
        assert extract_answer("The answer is #### 42") == "42"

    def test_comma_stripped(self):
        from tasks.gsm8k import extract_answer
        assert extract_answer("#### 1,234") == "1234"

    def test_negative_answer(self):
        from tasks.gsm8k import extract_answer
        assert extract_answer("#### -7") == "-7"

    def test_no_marker_raises_parse_error(self):
        from tasks.gsm8k import extract_answer
        with pytest.raises(ParseError, match="No answer marker found"):
            extract_answer("There is no marker here")

    def test_empty_string_raises_parse_error(self):
        from tasks.gsm8k import extract_answer
        with pytest.raises(ParseError, match="No answer marker found"):
            extract_answer("")


class TestGSM8KEvaluate:
    """gsm8k.evaluate must handle ParseError correctly.

    evaluate() takes (conversation, assistant_response) where conversation
    is a dict with messages[]. The assistant message content is a list of parts.
    """

    @staticmethod
    def _make_conversation(answer_text: str) -> dict:
        """Build a minimal GSM8K conversation with the given answer text."""
        return {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
            ],
        }

    def test_both_unparseable_returns_zero(self):
        """Previously None==None returned 1 (correct). Now must return 0."""
        from tasks.gsm8k import GSM8K
        task = GSM8K.__new__(GSM8K)
        conv = self._make_conversation("no marker here")
        result = task.evaluate(conv, "also no marker")
        assert result == 0, "Both unparseable must be incorrect, not silently correct"

    def test_correct_answer_returns_one(self):
        from tasks.gsm8k import GSM8K
        task = GSM8K.__new__(GSM8K)
        conv = self._make_conversation("The answer is #### 4")
        result = task.evaluate(conv, "I think it's #### 4")
        assert result == 1

    def test_wrong_answer_returns_zero(self):
        from tasks.gsm8k import GSM8K
        task = GSM8K.__new__(GSM8K)
        conv = self._make_conversation("The answer is #### 4")
        result = task.evaluate(conv, "#### 5")
        assert result == 0

    def test_reference_unparseable_raises_data_load_error(self):
        """If the ground truth lacks a marker, that's a data integrity issue."""
        from tasks.gsm8k import GSM8K
        task = GSM8K.__new__(GSM8K)
        conv = self._make_conversation("broken reference without marker")
        with pytest.raises(DataLoadError, match="Ground truth"):
            task.evaluate(conv, "#### 4")
