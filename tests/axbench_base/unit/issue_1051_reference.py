"""Frozen pre-#1051 text helpers for exact differential tests.

These functions are copied from production commit
``cd30e9911edfc5c33e186cc025c39ff1bbbc6807``. They are deliberately
independent of the new shared modules: editing them to follow production would
destroy their value as pre-refactor oracles.
"""

from __future__ import annotations

import re
import unicodedata

_MIN_REPEATED_PHRASE_WORDS = 8
_MIN_REPEATED_PHRASE_CHARS = 48
_MIN_SOURCE_ECHO_WORDS = 7
_MIN_SOURCE_ECHO_CHARS = 32
_MIN_SOURCE_ECHO_ALNUM_CHARS = 32


def legacy_window_prefix_text(
    full_text: str,
    seg_end_char: int,
    max_prefix_chars: int,
) -> str:
    """Pre-#1051 Stage 0c prefix window implementation."""
    start = max(0, seg_end_char - max_prefix_chars)
    return full_text[start:seg_end_char]


def legacy_repeated_phrase(text: str) -> str | None:
    """Pre-#1051 Stage 0c repeated-phrase detector."""
    words = re.findall(r"[^\W_]+", text.casefold())
    max_width = len(words) // 2
    for width in range(max_width, _MIN_REPEATED_PHRASE_WORDS - 1, -1):
        seen: dict[tuple[str, ...], int] = {}
        for start in range(len(words) - width + 1):
            phrase_words = tuple(words[start : start + width])
            prior_start = seen.get(phrase_words)
            if prior_start is None:
                seen[phrase_words] = start
            elif start - prior_start >= width:
                phrase = " ".join(phrase_words)
                if len(phrase) >= _MIN_REPEATED_PHRASE_CHARS:
                    return phrase
    return None


def legacy_source_phrase_echo(
    source_text: str,
    continuation: str,
) -> str | None:
    """Pre-#1051 Stage 0c source-echo detector."""
    source_words = re.findall(r"[^\W_]+", source_text.casefold())
    continuation_words = re.findall(r"[^\W_]+", continuation.casefold())
    max_width = min(len(source_words), len(continuation_words))
    for width in range(max_width, _MIN_SOURCE_ECHO_WORDS - 1, -1):
        continuation_phrases = {
            tuple(continuation_words[start : start + width])
            for start in range(len(continuation_words) - width + 1)
        }
        for start in range(len(source_words) - width + 1):
            phrase_words = tuple(source_words[start : start + width])
            if phrase_words not in continuation_phrases:
                continue
            phrase = " ".join(phrase_words)
            if len(phrase) >= _MIN_SOURCE_ECHO_CHARS:
                return phrase
    normalized_source = "".join(
        char for char in unicodedata.normalize("NFKC", source_text).casefold() if char.isalnum()
    )
    normalized_continuation = "".join(
        char for char in unicodedata.normalize("NFKC", continuation).casefold() if char.isalnum()
    )
    if (
        len(normalized_source) >= _MIN_SOURCE_ECHO_ALNUM_CHARS
        and len(normalized_continuation) >= _MIN_SOURCE_ECHO_ALNUM_CHARS
    ):
        source_windows = {
            normalized_source[start : start + _MIN_SOURCE_ECHO_ALNUM_CHARS]
            for start in range(len(normalized_source) - _MIN_SOURCE_ECHO_ALNUM_CHARS + 1)
        }
        for start in range(len(normalized_continuation) - _MIN_SOURCE_ECHO_ALNUM_CHARS + 1):
            candidate = normalized_continuation[start : start + _MIN_SOURCE_ECHO_ALNUM_CHARS]
            if candidate in source_windows:
                return candidate
    return None
