"""Shared deterministic guards for LLM-generated continuation text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from steering.eval.axbench_base.types import AxBenchBaseError

MIN_PREFIX_SUFFIX_ECHO_CHARS = 16
PREFIX_ECHO_CHARS = 96
PREFIX_ECHO_SEPARATORS = ("\r\n\r\n", "\n\n", "\r\n", "\n", " ", "")
_MIN_REPEATED_PHRASE_WORDS = 8
_MIN_REPEATED_PHRASE_CHARS = 48
_MIN_SOURCE_ECHO_WORDS = 7
_MIN_SOURCE_ECHO_CHARS = 32
_MIN_SOURCE_ECHO_ALNUM_CHARS = 32


class ContinuationProtocolError(AxBenchBaseError):
    """A provider response violated the authenticated continuation protocol."""


@dataclass(frozen=True)
class ResponseEnvelope:
    """Nonce-delimited response boundary for one provider attempt."""

    begin: str
    end: str
    split: str | None = None


@dataclass(frozen=True)
class PrefixEchoFraming:
    """The exact framing prefix selected for one authenticated response."""

    text: str
    exact_echo: str
    used_left_truncation: bool


def make_response_envelope(
    kind: str,
    identity: object,
    *,
    untrusted_texts: Sequence[str],
    protocol_version: int,
    semantic_identity: Callable[[object], str],
    error_subject: str,
    error_type: type[AxBenchBaseError] = ContinuationProtocolError,
) -> ResponseEnvelope:
    """Build deterministic markers guaranteed absent from untrusted inputs."""
    normalized_kind = re.sub(r"[^A-Z0-9_]", "_", kind.upper())
    for salt in range(10_000):
        digest = semantic_identity(
            {
                "protocol_version": protocol_version,
                "kind": normalized_kind,
                "identity": identity,
                "salt": salt,
            }
        )[:24].upper()
        begin = f"[[AXBENCH_{normalized_kind}_BEGIN_{digest}]]"
        end = f"[[AXBENCH_{normalized_kind}_END_{digest}]]"
        truncated_end = end[:-1]
        if all(
            begin not in text and end not in text and truncated_end not in text
            for text in untrusted_texts
        ):
            return ResponseEnvelope(begin=begin, end=end)
    raise error_type(
        f"Could not derive {error_subject} protocol markers absent from the "
        "untrusted prompt inputs after 10000 salts"
    )


def parse_response_envelope(
    response: str,
    envelope: ResponseEnvelope,
    *,
    tolerate_end_marker_variants: bool = False,
    error_subject: str,
    counter_prefix: str | None = None,
    record_counter_fn: Callable[..., Any] | None = None,
    error_type: type[AxBenchBaseError] = ContinuationProtocolError,
) -> str:
    """Extract one well-ordered envelope with only audited end variants."""

    def _record(name: str, value: int, **attrs: Any) -> None:
        if counter_prefix is not None and record_counter_fn is not None:
            record_counter_fn(f"{counter_prefix}.{name}", value, **attrs)

    if not isinstance(response, str):
        raise error_type(
            f"{error_subject} response must be a string, got {type(response).__name__}"
        )
    begin_count = response.count(envelope.begin)
    end_count = response.count(envelope.end)
    if begin_count != 1:
        raise error_type(
            f"{error_subject} response must contain exactly one matching begin and "
            f"end marker, found begin={begin_count}, end={end_count}"
        )
    begin_at = response.index(envelope.begin)
    payload_at = begin_at + len(envelope.begin)
    recognized_end = envelope.end
    if end_count:
        end_at = response.index(envelope.end)
        if end_count > 1 and tolerate_end_marker_variants:
            _record("normalized_duplicate_end_marker_chatter", end_count - 1)
        elif end_count != 1:
            raise error_type(
                f"{error_subject} response must contain exactly one matching begin and "
                f"end marker, found begin={begin_count}, end={end_count}"
            )
    elif tolerate_end_marker_variants:
        recognized_end = envelope.end[:-1]
        truncated_count = response.count(recognized_end)
        if not truncated_count:
            raise error_type(
                f"{error_subject} response must contain exactly one matching begin and "
                f"end marker, found begin={begin_count}, end={end_count}"
            )
        end_at = response.index(recognized_end)
        _record("normalized_truncated_end_marker", 1)
        if truncated_count > 1:
            _record("normalized_duplicate_end_marker_chatter", truncated_count - 1)
    else:
        raise error_type(
            f"{error_subject} response must contain exactly one matching begin and "
            f"end marker, found begin={begin_count}, end={end_count}"
        )
    if end_at < payload_at:
        raise error_type(
            f"{error_subject} response end marker occurs before its begin marker"
        )
    if begin_at > 0 or end_at + len(recognized_end) < len(response):
        _record("outside_envelope_chatter", 1)
    return response[payload_at:end_at]


def prefix_suffix_echo_chars(
    prefix_text: str,
    continuation: str,
    *,
    min_chars: int = MIN_PREFIX_SUFFIX_ECHO_CHARS,
) -> int:
    """Return the longest substantial prefix-suffix echo at output start."""
    continuation_body = continuation.lstrip()
    max_overlap = min(len(prefix_text), len(continuation_body))
    for overlap in range(max_overlap, min_chars - 1, -1):
        if prefix_text.endswith(continuation_body[:overlap]):
            return overlap
    return 0


def prefix_echo_text(
    prefix_text: str,
    *,
    max_chars: int = PREFIX_ECHO_CHARS,
) -> str:
    """Return the exact local suffix the provider must authenticate."""
    return prefix_text[-max_chars:]


def match_prefix_echo_framing(
    payload: str,
    prefix_text: str,
    *,
    min_chars: int = MIN_PREFIX_SUFFIX_ECHO_CHARS,
    max_chars: int = PREFIX_ECHO_CHARS,
) -> PrefixEchoFraming | None:
    """Select the first exact audited separator/echo framing candidate."""
    exact_echo = prefix_echo_text(prefix_text, max_chars=max_chars)
    echo_candidates = [(exact_echo, False)]
    if len(exact_echo) > min_chars:
        echo_candidates.append((exact_echo[1:], True))
    for echo, used_left_truncation in echo_candidates:
        for separator in PREFIX_ECHO_SEPARATORS:
            candidate = separator + echo
            if payload.startswith(candidate):
                return PrefixEchoFraming(
                    text=candidate,
                    exact_echo=exact_echo,
                    used_left_truncation=used_left_truncation,
                )
    return None


def repeated_phrase(text: str) -> str | None:
    """Return one substantial repeated word sequence, normalized for noise.

    Exact phrase loops are a common provider degeneration. Case, punctuation,
    and whitespace are presentation details, so detection operates on
    case-folded Unicode words. Short refrains remain allowed; a refusal needs
    at least eight words and 48 normalized characters repeated without
    overlap.
    """
    words = re.findall(r"[^\W_]+", text.casefold())
    max_width = len(words) // 2
    for width in range(max_width, _MIN_REPEATED_PHRASE_WORDS - 1, -1):
        seen: dict[tuple[str, ...], int] = {}
        for start in range(len(words) - width + 1):
            phrase_words = tuple(words[start : start + width])
            prior_start = seen.get(phrase_words)
            if prior_start is None:
                # Keep the earliest occurrence. Replacing it with an
                # overlapping later occurrence could hide a subsequent
                # non-overlapping repeat.
                seen[phrase_words] = start
            elif start - prior_start >= width:
                phrase = " ".join(phrase_words)
                if len(phrase) >= _MIN_REPEATED_PHRASE_CHARS:
                    return phrase
    return None


def source_phrase_echo(source_text: str, continuation: str) -> str | None:
    """Return a substantial normalized phrase copied from source context.

    Provider failures do not always echo the source at the continuation
    boundary. Live Ling output has repeated a long source phrase after an
    otherwise clean paragraph, which bypasses a prefix-only overlap check.
    Case, punctuation, and whitespace are presentation details, so compare
    case-folded Unicode words. Seven words and 32 normalized characters is
    deliberately conservative: rejection sampling can discard an occasional
    natural refrain, while a leaked source copy must never enter an accepted
    token window.
    """
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
    # Whitespace-delimited words do not cover CJK and other scripts without
    # reliable word boundaries. A normalized alphanumeric window closes that
    # gap and also catches punctuation-heavy variants of long source strings.
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


def validate_no_source_echo_or_repeated_phrase(
    continuation: str,
    *,
    source_text: str,
    subject: str,
) -> None:
    """Reject substantial phrase loops or normalized source copies.

    Repeated phrases intentionally take precedence over source echoes, matching
    the Stage 0c protocol contract and its existing diagnostics exactly.
    """
    phrase_loop = repeated_phrase(continuation)
    if phrase_loop is not None:
        raise AxBenchBaseError(
            f"{subject} contains a repeated phrase loop: {phrase_loop!r}"
        )
    source_echo = source_phrase_echo(source_text, continuation)
    if source_echo is not None:
        raise AxBenchBaseError(
            f"{subject} copied a substantial source phrase: {source_echo!r}"
        )
