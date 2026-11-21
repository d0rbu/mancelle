"""Structural tokenizer contracts for project-owned callers.

The vendored nanochat loaders can return either RustBPE or Transformers
tokenizers, while extraction utilities also use Hugging Face tokenizers
directly. These protocols describe only the methods shared by the callers
below and keep those boundaries independent of a concrete implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextTokenizer(Protocol):
    """Tokenizer surface needed for plain-text round trips."""

    def encode(self, text: str, /) -> list[int]: ...

    def decode(self, token_ids: list[int], /) -> str: ...


@runtime_checkable
class ChatTokenizer(TextTokenizer, Protocol):
    """Text tokenizer with nanochat chat-special-token support."""

    def encode_special(self, text: str, /) -> int: ...

    def get_bos_token_id(self) -> int: ...


@runtime_checkable
class PrefixedTextTokenizer(Protocol):
    """Text tokenizer supporting an explicit prepended token ID."""

    def encode(
        self,
        text: str,
        /,
        *,
        prepend: int | None = None,
    ) -> list[int]: ...

    def decode(self, token_ids: list[int], /) -> str: ...


@runtime_checkable
class BosTokenizer(PrefixedTextTokenizer, Protocol):
    """Prefixed text tokenizer that can resolve its BOS token ID."""

    def get_bos_token_id(self) -> int: ...


@runtime_checkable
class CompletionTokenizer(Protocol):
    """Tokenizer surface needed by nanochat completion-based RL."""

    def encode_special(self, text: str, /) -> int: ...

    def decode(self, token_ids: list[int], /) -> str: ...

    def render_for_completion(self, conversation: object, /) -> list[int]: ...


def require_text_tokenizer(tokenizer: object, *, surface: str) -> TextTokenizer:
    """Validate and narrow an object to the plain-text tokenizer contract."""
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError(
            f"{surface} requires tokenizer.encode(text) and tokenizer.decode(ids); "
            f"got {type(tokenizer).__name__}"
        )
    return tokenizer


def require_chat_tokenizer(tokenizer: object, *, surface: str) -> ChatTokenizer:
    """Validate and narrow an object to the nanochat tokenizer contract."""
    if not isinstance(tokenizer, ChatTokenizer):
        raise TypeError(
            f"{surface} requires encode/decode, encode_special, and "
            f"get_bos_token_id; got {type(tokenizer).__name__}"
        )
    return tokenizer


def require_bos_tokenizer(tokenizer: object, *, surface: str) -> BosTokenizer:
    """Validate and narrow an object to the BOS-prefix tokenizer contract."""
    if not isinstance(tokenizer, BosTokenizer):
        raise TypeError(
            f"{surface} requires encode/decode and get_bos_token_id; got {type(tokenizer).__name__}"
        )
    return tokenizer


def require_completion_tokenizer(tokenizer: object, *, surface: str) -> CompletionTokenizer:
    """Validate and narrow an object to the completion-rendering contract."""
    if not isinstance(tokenizer, CompletionTokenizer):
        raise TypeError(
            f"{surface} requires encode_special, decode, and "
            f"render_for_completion; got {type(tokenizer).__name__}"
        )
    return tokenizer
