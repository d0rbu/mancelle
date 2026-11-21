"""Regression tests for empty multi-part response rows being skipped.

The bug: structured tool-use messages with empty ``python`` parts can still
render to non-empty tool markers. The converter must inspect raw part text
before nanochat rendering and skip rows where every part is empty.

These tests construct real conversations, run the real conversion, then count
rows in the actual parquet output (per task requirement: no writer mocks).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.preprocess.convert_sft_to_parquet import (
    _is_empty_multipart_response,
    convert_mixture_to_parquet,
)


class FakeMixture:
    """Minimal in-memory mixture with `__len__`/`__getitem__` duck-typed interface.

    ``convert_mixture_to_parquet`` annotates ``mixture: SFTConversationSource``
    (a structural ``Protocol``), which only requires ``len(...)`` and
    ``mixture[i]`` — duck-typed, and satisfied by this fixture directly under
    both ``@beartype`` (runtime) and ``ty`` (static) without a cast.
    """

    def __init__(self, conversations: list[dict]) -> None:
        self.conversations = conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        return self.conversations[idx]


class FakeTokenizer:
    """Tokenizer stub that preserves visible message and tool markers in text."""

    _bos_id = 101
    _specials = {
        "<user_start>": 100_001,
        "<assistant_start>": 100_002,
        "<output_start>": 100_003,
        "<output_end>": 100_004,
    }
    _decoded_specials = {
        100_001: "<user_start>",
        100_002: "<assistant_start>",
        100_003: "<python_output_start>",
        100_004: "<python_output_end>",
    }

    def render_conversation(self, conversation):
        ids = [self._bos_id]
        for message in conversation["messages"]:
            role = message["role"]
            ids.append(self._specials[f"<{role}_start>"])
            content = message["content"]
            if isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "text")
                    text = part.get("text", "")
                    if part_type == "python":
                        rendered = f"<python_start>{text}<python_end>"
                        ids.extend(ord(character) for character in rendered)
                    elif part_type == "python_output":
                        ids.append(self._specials["<output_start>"])
                        ids.extend(ord(character) for character in text)
                        ids.append(self._specials["<output_end>"])
                    else:
                        ids.extend(ord(character) for character in text)
            else:
                ids.extend(ord(character) for character in str(content))
            ids.extend(ord(character) for character in f"<{role}_end>")
        return ids, [0] * len(ids)

    def get_bos_token_id(self):
        return self._bos_id

    def encode_special(self, name):
        return self._specials[name]

    def decode(self, ids):
        return "".join(
            self._decoded_specials[token_id]
            if token_id in self._decoded_specials
            else chr(token_id)
            for token_id in ids
        )


@pytest.fixture(autouse=True)
def stub_converter_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these parquet tests independent of the local nanochat tokenizer artifact."""
    monkeypatch.setattr(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        lambda: FakeTokenizer(),
    )


def _run(mixture: FakeMixture, out: Path) -> None:
    convert_mixture_to_parquet(mixture, out, mixture_name="test")


def _count_rows(parquet_dir: Path) -> int:
    total = 0
    for pf in sorted(parquet_dir.glob("*.parquet")):
        total += pq.read_table(pf).num_rows
    return total


def _read_texts(parquet_dir: Path) -> list[str]:
    texts: list[str] = []
    for pf in sorted(parquet_dir.glob("*.parquet")):
        texts.extend(pq.read_table(pf).column("text").to_pylist())
    return texts


def _read_original_indices(parquet_dir: Path) -> list[int]:
    indices: list[int] = []
    for pf in sorted(parquet_dir.glob("*.parquet")):
        indices.extend(pq.read_table(pf).column("original_index").to_pylist())
    return indices


class TestIsEmptyMultipartResponse:
    """Unit tests for the `_is_empty_multipart_response` helper."""

    def test_all_empty_strings_is_empty(self) -> None:
        parts = [
            {"type": "python", "text": ""},
            {"type": "python_output", "text": ""},
        ]
        assert _is_empty_multipart_response(parts) is True

    def test_whitespace_only_is_empty(self) -> None:
        parts = [
            {"type": "text", "text": "   \n\t"},
            {"type": "python", "text": "  "},
        ]
        assert _is_empty_multipart_response(parts) is True

    def test_any_nonempty_is_not_empty(self) -> None:
        parts = [
            {"type": "python", "text": ""},
            {"type": "text", "text": "answer"},
        ]
        assert _is_empty_multipart_response(parts) is False

    def test_empty_list_is_empty(self) -> None:
        assert _is_empty_multipart_response([]) is True

    def test_missing_text_key_is_empty(self) -> None:
        parts = [{"type": "python"}]
        assert _is_empty_multipart_response(parts) is True


class TestConversionSkipsEmptyMultipart:
    """Integration-style tests: run real conversion, count rows in parquet output."""

    def test_empty_multipart_response_is_skipped(self, tmp_path: Path) -> None:
        """A conversation with only-empty python parts is NOT written to parquet.

        Without the fix, the nanochat renderer can emit non-empty tool markers
        for the otherwise empty response, so the row sneaks into parquet.
        """
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Count 'p' in 'python'"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "python", "text": ""},
                            {"type": "python_output", "text": ""},
                        ],
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ]
        _run(FakeMixture(convs), tmp_path)
        rows = _count_rows(tmp_path)
        assert rows == 1, (
            f"Expected 1 row (the non-empty one). Got {rows}."
        )
        # The surviving row should be the plain string conversation at idx=1.
        texts = _read_texts(tmp_path)
        assert len(texts) == 1
        assert "A2" in texts[0]
        assert _read_original_indices(tmp_path) == [1]

    def test_whitespace_only_multipart_is_skipped(self, tmp_path: Path) -> None:
        """Multi-part response where every text is whitespace only → skipped."""
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Q0"},
                    {"role": "assistant", "content": "A0"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "whitespace"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "   \n\t"},
                            {"type": "python", "text": "  "},
                        ],
                    },
                ]
            },
        ]
        _run(FakeMixture(convs), tmp_path)
        rows = _count_rows(tmp_path)
        assert rows == 1
        texts = _read_texts(tmp_path)
        assert len(texts) == 1
        assert "A0" in texts[0]
        assert _read_original_indices(tmp_path) == [0]

    def test_nonempty_multipart_is_kept(self, tmp_path: Path) -> None:
        """At least one non-empty text in multi-part response → row written."""
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Count 'a' in 'banana'"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Let me count. "},
                            {"type": "python", "text": "'banana'.count('a')"},
                            {"type": "python_output", "text": "3"},
                            {"type": "text", "text": "\nThe answer is 3."},
                        ],
                    },
                ]
            },
        ]
        _run(FakeMixture(convs), tmp_path)
        rows = _count_rows(tmp_path)
        assert rows == 1
        texts = _read_texts(tmp_path)
        assert len(texts) == 1
        assert "<python_start>" in texts[0]
        assert "'banana'.count('a')" in texts[0]

    def test_empty_parts_list_is_skipped(self, tmp_path: Path) -> None:
        """A list-typed content equal to `[]` is treated as empty and skipped."""
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Q0"},
                    {"role": "assistant", "content": []},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
        ]
        _run(FakeMixture(convs), tmp_path)
        rows = _count_rows(tmp_path)
        assert rows == 1
        texts = _read_texts(tmp_path)
        assert len(texts) == 1
        assert "A1" in texts[0]
        assert _read_original_indices(tmp_path) == [1]
