"""Regression tests for wave3b/8 Fix B: `tasks.customjson` missing-file behavior.

The bug: when `identity_conversations.jsonl` is absent, `CustomJSON.__init__`
logs a warning and yields zero conversations silently, so the STANDARD SFT
mixture loses its identity data without any loud signal.

The fix introduces an `allow_missing: bool = False` parameter. Default raises
`DataLoadError`; explicit `allow_missing=True` logs a warning and yields zero.

These tests construct real JSONL files under `tmp_path` — no mocks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from errors import DataLoadError
from tasks.customjson import CustomJSON


def _write_valid_jsonl(path: Path, n: int = 2) -> None:
    lines: list[str] = []
    for i in range(n):
        conv = [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ]
        lines.append(json.dumps(conv))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCustomJsonMissingFile:
    """Behavior when the configured JSONL file does not exist."""

    def test_customjson_missing_file_raises(self, tmp_path: Path) -> None:
        """Default (allow_missing=False): missing file raises DataLoadError
        with the path in the message."""
        missing = tmp_path / "nope.jsonl"
        assert not missing.exists()
        with pytest.raises(DataLoadError) as excinfo:
            CustomJSON(filepath=str(missing))
        msg = str(excinfo.value)
        assert str(missing) in msg
        assert "not found" in msg.lower() or "missing" in msg.lower() or "empty" in msg.lower()

    def test_customjson_allow_missing_yields_zero(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """allow_missing=True + missing file: zero conversations, warning logged."""
        missing = tmp_path / "absent.jsonl"
        assert not missing.exists()
        # Loguru emits through its own handler; propagate to std logging so
        # pytest's caplog can see the message.
        from loguru import logger as loguru_logger

        class _PropagateHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                logging.getLogger(record.name).handle(record)

        sink_id = loguru_logger.add(
            _PropagateHandler(), level="WARNING", format="{message}"
        )
        try:
            with caplog.at_level(logging.WARNING):
                task = CustomJSON(filepath=str(missing), allow_missing=True)
        finally:
            loguru_logger.remove(sink_id)
        assert task.num_examples() == 0
        assert len(task) == 0
        # Warning must mention the path.
        combined = " ".join(rec.getMessage() for rec in caplog.records)
        assert str(missing) in combined

    def test_customjson_empty_file_raises(self, tmp_path: Path) -> None:
        """Empty file with default args → DataLoadError."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert empty.exists()
        with pytest.raises(DataLoadError):
            CustomJSON(filepath=str(empty))

    def test_customjson_valid_file_loads(self, tmp_path: Path) -> None:
        """A valid 2-line JSONL returns 2 conversations."""
        valid = tmp_path / "valid.jsonl"
        _write_valid_jsonl(valid, n=2)
        task = CustomJSON(filepath=str(valid))
        assert task.num_examples() == 2
        conv0 = task.get_example(0)
        assert "messages" in conv0
        assert conv0["messages"][0]["role"] == "user"
        assert conv0["messages"][0]["content"] == "question 0"
        assert conv0["messages"][1]["role"] == "assistant"
        assert conv0["messages"][1]["content"] == "answer 0"


# --- Wave 3C.6 additions: loader-contract coverage ------------------------


class TestCustomJsonLoaderContract:
    """Loader contract: 3-row happy path, multi-turn, blank-line skipping, malformed input."""

    def test_exact_rendered_output_three_rows(self, tmp_path: Path) -> None:
        """Lock in the exact conversations returned for a 3-row fixture."""
        jsonl = tmp_path / "fixture.jsonl"
        rows = [
            [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am an AI."},
            ],
            [
                {"role": "user", "content": "Multi-turn start"},
                {"role": "assistant", "content": "Reply 1"},
                {"role": "user", "content": "Follow-up"},
                {"role": "assistant", "content": "Reply 2"},
            ],
            [
                {"role": "user", "content": "Third row"},
                {"role": "assistant", "content": "Last."},
            ],
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        task = CustomJSON(filepath=str(jsonl))
        assert task.num_examples() == 3
        got = [task.get_example(i) for i in range(3)]
        expected = [{"messages": r} for r in rows]
        assert got == expected

    def test_schema_shape(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "f.jsonl"
        _write_valid_jsonl(jsonl, n=1)
        task = CustomJSON(filepath=str(jsonl))
        conv = task.get_example(0)
        assert set(conv.keys()) == {"messages"}

    def test_multi_turn_alternating_roles_accepted(self, tmp_path: Path) -> None:
        """4-turn conversation with strictly alternating user/assistant passes."""
        jsonl = tmp_path / "multi.jsonl"
        conv = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
        jsonl.write_text(json.dumps(conv) + "\n", encoding="utf-8")
        task = CustomJSON(filepath=str(jsonl))
        assert task.num_examples() == 1
        assert task.get_example(0)["messages"] == conv

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        """Blank/whitespace-only lines don't produce empty conversations."""
        jsonl = tmp_path / "blanks.jsonl"
        conv = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        jsonl.write_text(
            "\n\n" + json.dumps(conv) + "\n   \n" + json.dumps(conv) + "\n",
            encoding="utf-8",
        )
        task = CustomJSON(filepath=str(jsonl))
        assert task.num_examples() == 2

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON line: loud exception, not silent empty loader."""
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("{this is not json}\n", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            CustomJSON(filepath=str(jsonl))

    def test_too_few_messages_raises(self, tmp_path: Path) -> None:
        """Filter path: conversation with < 2 messages rejected."""
        jsonl = tmp_path / "short.jsonl"
        jsonl.write_text(
            json.dumps([{"role": "user", "content": "only one"}]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="at least 2 messages"):
            CustomJSON(filepath=str(jsonl))

    def test_wrong_leading_role_raises(self, tmp_path: Path) -> None:
        """Filter path: must alternate starting with user."""
        jsonl = tmp_path / "wrong_role.jsonl"
        bad = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "uh"},
        ]
        jsonl.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="role assistant but should be user"):
            CustomJSON(filepath=str(jsonl))

    def test_missing_role_field_raises(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "missing.jsonl"
        bad = [{"content": "hi"}, {"role": "assistant", "content": "yo"}]
        jsonl.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="missing 'role'"):
            CustomJSON(filepath=str(jsonl))

    def test_non_object_message_raises(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "non_object_message.jsonl"
        jsonl.write_text(
            json.dumps(["not an object", {"role": "assistant", "content": "yo"}]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="Message 0 must be an object"):
            CustomJSON(filepath=str(jsonl))

    def test_non_list_conversation_raises(self, tmp_path: Path) -> None:
        """Top-level JSON must be a list; object at top level is rejected."""
        jsonl = tmp_path / "obj.jsonl"
        jsonl.write_text(
            json.dumps({"role": "user", "content": "x"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(AssertionError, match="Expected list of messages"):
            CustomJSON(filepath=str(jsonl))
