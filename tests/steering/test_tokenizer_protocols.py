"""Tests for shared structural tokenizer contracts."""

from __future__ import annotations

import pytest

from steering.tokenizer_protocols import (
    BosTokenizer,
    ChatTokenizer,
    CompletionTokenizer,
    PrefixedTextTokenizer,
    TextTokenizer,
    require_bos_tokenizer,
    require_chat_tokenizer,
    require_completion_tokenizer,
    require_text_tokenizer,
)


class _TextTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _ChatTokenizer(_TextTokenizer):
    def encode_special(self, text: str) -> int:
        return len(text)

    def get_bos_token_id(self) -> int:
        return 1


class _BosTokenizer(_TextTokenizer):
    def encode(self, text: str, *, prepend: int | None = None) -> list[int]:
        prefix = [] if prepend is None else [prepend]
        return prefix + list(text.encode())

    def get_bos_token_id(self) -> int:
        return 1


class _CompletionTokenizer:
    def encode_special(self, text: str) -> int:
        return len(text)

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode()

    def render_for_completion(self, conversation: object) -> list[int]:
        return [1]


def test_text_tokenizer_narrowing_preserves_identity() -> None:
    tokenizer = _TextTokenizer()

    assert isinstance(tokenizer, TextTokenizer)
    assert require_text_tokenizer(tokenizer, surface="test") is tokenizer


def test_chat_tokenizer_narrowing_preserves_identity() -> None:
    tokenizer = _ChatTokenizer()

    assert isinstance(tokenizer, ChatTokenizer)
    assert require_chat_tokenizer(tokenizer, surface="test") is tokenizer


def test_bos_tokenizer_narrowing_preserves_identity() -> None:
    tokenizer = _BosTokenizer()

    assert isinstance(tokenizer, PrefixedTextTokenizer)
    assert isinstance(tokenizer, BosTokenizer)
    assert require_bos_tokenizer(tokenizer, surface="test") is tokenizer


def test_completion_tokenizer_narrowing_preserves_identity() -> None:
    tokenizer = _CompletionTokenizer()

    assert isinstance(tokenizer, CompletionTokenizer)
    assert require_completion_tokenizer(tokenizer, surface="test") is tokenizer


def test_text_tokenizer_rejects_missing_decode() -> None:
    class _EncodeOnly:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    with pytest.raises(TypeError, match=r"sample requires tokenizer\.encode"):
        require_text_tokenizer(_EncodeOnly(), surface="sample")


def test_chat_tokenizer_rejects_plain_text_tokenizer() -> None:
    with pytest.raises(TypeError, match="chat requires encode/decode"):
        require_chat_tokenizer(_TextTokenizer(), surface="chat")


def test_bos_tokenizer_rejects_missing_bos_method() -> None:
    with pytest.raises(TypeError, match="sample requires encode/decode"):
        require_bos_tokenizer(_TextTokenizer(), surface="sample")


def test_completion_tokenizer_rejects_missing_renderer() -> None:
    with pytest.raises(TypeError, match="rl requires encode_special"):
        require_completion_tokenizer(_ChatTokenizer(), surface="rl")
