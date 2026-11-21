"""Shared fixtures for completion TUI tests.

Provides a word-level mock tokenizer with multi-character tokens,
essential for testing retokenization boundary snapping.
"""

from __future__ import annotations

import pytest


class WordMockTokenizer:
    """Greedy longest-match tokenizer with a small fixed vocabulary.

    Produces multi-character tokens so that token boundaries don't coincide
    with every character position — required for meaningful retokenization tests.

    Key vocabulary for the spec test case:
    - "hello" (5 chars) and "world" (5 chars) are full tokens
    - "hel" (3 chars) and "tlo" (3 chars) are tokens
    - So "hello world" → ["hello", " ", "world"]
    - And "heltlo world" → ["hel", "tlo", " ", "world"]
    """

    _MULTI_CHAR_TOKENS: list[str] = [
        "hello",
        "world",
        "quick",
        "brown",
        "jumps",
        "there",
        "split",
        "test",
        "the",
        "hel",
        "tlo",
        "foo",
        "bar",
        "baz",
        "fox",
        "cat",
        "dog",
        "big",
        "end",
    ]

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

        # Multi-char tokens get low IDs
        for i, tok in enumerate(self._MULTI_CHAR_TOKENS):
            self._token_to_id[tok] = i
            self._id_to_token[i] = tok

        # Single ASCII chars fill the rest
        next_id = len(self._MULTI_CHAR_TOKENS)
        for code in range(32, 127):
            ch = chr(code)
            if ch not in self._token_to_id:
                self._token_to_id[ch] = next_id
                self._id_to_token[next_id] = ch
                next_id += 1

        # Sort by length descending for greedy matching
        self._sorted_tokens = sorted(
            self._token_to_id.keys(), key=len, reverse=True
        )

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match tokenization."""
        tokens: list[int] = []
        i = 0
        while i < len(text):
            for tok in self._sorted_tokens:
                if text[i : i + len(tok)] == tok:
                    tokens.append(self._token_to_id[tok])
                    i += len(tok)
                    break
            else:
                # Fallback for chars outside ASCII printable (e.g., Unicode)
                fallback_id = 10000 + ord(text[i])
                self._id_to_token[fallback_id] = text[i]
                tokens.append(fallback_id)
                i += 1
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        parts: list[str] = []
        for tid in ids:
            if tid in self._id_to_token:
                parts.append(self._id_to_token[tid])
            elif tid >= 10000:
                parts.append(chr(tid - 10000))
        return "".join(parts)


@pytest.fixture()
def word_tokenizer() -> WordMockTokenizer:
    """A word-level mock tokenizer with multi-character tokens."""
    return WordMockTokenizer()
