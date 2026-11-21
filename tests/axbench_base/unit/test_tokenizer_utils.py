"""BOS ownership and exact labeled-pair tokenization tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.tokenizer_utils import (
    assert_bos_prefix,
    prepend_owned_bos,
    tokenize_labeled_pair,
)
from steering.eval.axbench_base.types import AxBenchBaseError

# ---------------------------------------------------------------------------
# Stub tokenizer
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Minimal nanochat-shaped tokenizer for unit tests.

    ``encode(text)`` returns deterministic fake token ids derived from
    the character codepoints; when ``prepend`` is provided, the id is
    inserted at position 0 (exactly like nanochat's
    :class:`RustBPETokenizer`). ``get_bos_token_id`` returns
    ``self._bos``, which can be set to ``None`` to simulate a tokenizer
    that has no BOS id.
    """

    def __init__(self, bos_id: int | None = 77) -> None:
        self._bos = bos_id

    def get_bos_token_id(self) -> int | None:
        return self._bos

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        # Deterministic "tokenization": one token per codepoint, ids in
        # a range (100..) that avoids colliding with a BOS id of 77.
        # Tests can inject a literal BOS-at-the-start by crafting a
        # string whose first character is chr(77).
        ids = [100 + (ord(c) % 50) for c in text]
        # Special-case: let tests simulate "text already starts with BOS"
        # by using the sentinel character chr(0); maps to the BOS id.
        result: list[int] = []
        for c in text:
            if ord(c) == 0 and self._bos is not None:
                result.append(self._bos)
            else:
                result.append(100 + (ord(c) % 50))
        # Avoid unused-variable lint
        del ids
        if prepend is not None:
            return [prepend, *result]
        return result


class PairTokenizer:
    """Tiny table tokenizer with context-sensitive progressive decodes."""

    def __init__(
        self,
        *,
        encoded: list[int],
        prefix_decodes: dict[tuple[int, ...], str],
        bos_id: int = 77,
    ) -> None:
        self.encoded = encoded
        self.prefix_decodes = prefix_decodes
        self.bos_id = bos_id
        self.encoded_texts: list[str] = []

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return list(self.encoded)

    def decode(self, tokens: list[int]) -> str:
        return self.prefix_decodes[tuple(tokens)]


# ---------------------------------------------------------------------------
# Explicit owned-BOS helpers
# ---------------------------------------------------------------------------


def test_owned_bos_preserves_equal_payload_id() -> None:
    tok = StubTokenizer(bos_id=77)
    assert prepend_owned_bos(tok, [77, 4, 77]) == [77, 77, 4, 77]


@given(st.lists(st.integers(min_value=0, max_value=100_000), max_size=100))
def test_owned_bos_insertion_is_positionally_reversible(payload: list[int]) -> None:
    tok = StubTokenizer(bos_id=77)
    model_tokens = prepend_owned_bos(tok, payload)
    assert model_tokens[0] == 77
    assert model_tokens[1:] == payload


# ---------------------------------------------------------------------------
# Exact combined-text output boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "output", "encoded", "decodes", "expected_boundary"),
    [
        ("abc", "XYZ", [10, 11], {(10,): "abc", (10, 11): "abcXYZ"}, 2),
        # Token 11 spans characters 2:6, crossing output start at character 3.
        ("abc", "XYZ", [10, 11], {(10,): "ab", (10, 11): "abcXYZ"}, 2),
        ("café", "☕", [20, 21], {(20,): "café", (20, 21): "café☕"}, 2),
    ],
)
def test_tokenize_labeled_pair_exact_boundary(
    prefix: str,
    output: str,
    encoded: list[int],
    decodes: dict[tuple[int, ...], str],
    expected_boundary: int,
) -> None:
    tok = PairTokenizer(encoded=encoded, prefix_decodes=decodes)
    tokens, boundary = tokenize_labeled_pair(tok, prefix, output)
    assert tokens == [77, *encoded]
    assert boundary == expected_boundary
    assert tok.encoded_texts == [prefix + output]


def test_tokenize_labeled_pair_rejects_non_roundtripping_input() -> None:
    tok = PairTokenizer(encoded=[10], prefix_decodes={(10,): "different"})
    with pytest.raises(AxBenchBaseError, match="does not round-trip"):
        tokenize_labeled_pair(tok, "prefix", "output")


def test_tokenize_labeled_pair_rejects_empty_output() -> None:
    tok = PairTokenizer(encoded=[10], prefix_decodes={(10,): "prefix"})
    with pytest.raises(AxBenchBaseError, match="output_text must be non-empty"):
        tokenize_labeled_pair(tok, "prefix", "")


@pytest.mark.parametrize(
    ("prefix", "output", "message"),
    [
        pytest.param(
            None, "output", "prefix_text must be a string", id="null-prefix"
        ),
        pytest.param(
            "prefix", None, "output_text must be a string", id="null-output"
        ),
    ],
)
def test_tokenize_labeled_pair_rejects_non_string_text_without_coercion(
    prefix: object,
    output: object,
    message: str,
) -> None:
    tok = PairTokenizer(encoded=[10], prefix_decodes={(10,): "unused"})
    unchecked_tokenize = cast(
        Callable[[object, object, object], tuple[list[int], int]],
        tokenize_labeled_pair,
    )
    with pytest.raises(AxBenchBaseError, match=message):
        unchecked_tokenize(tok, prefix, output)


# ---------------------------------------------------------------------------
# assert_bos_prefix
# ---------------------------------------------------------------------------


def test_assert_bos_prefix_passes_on_valid() -> None:
    """A list that starts with [bos_id, ...non-bos...] passes."""
    assert_bos_prefix([77, 1, 2, 3], bos_id=77)  # must not raise


def test_assert_bos_prefix_empty_raises() -> None:
    """An empty list has no BOS prefix -> raise."""
    with pytest.raises(AxBenchBaseError, match="empty"):
        assert_bos_prefix([], bos_id=77)


def test_assert_bos_prefix_missing_bos_raises() -> None:
    """First token != BOS -> raise."""
    with pytest.raises(AxBenchBaseError, match="does not start with BOS"):
        assert_bos_prefix([100, 1, 2], bos_id=77)


def test_assert_bos_prefix_double_bos_raises() -> None:
    """First two tokens both BOS -> raise."""
    with pytest.raises(AxBenchBaseError, match="double BOS"):
        assert_bos_prefix([77, 77, 1, 2], bos_id=77)
