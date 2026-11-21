"""Exact differential proof for the Stage 0c protocol-helper extraction."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections.abc import Callable, Sequence
from typing import Any

import pytest

import steering.eval.axbench_base.stage0c_api_generation as current
from steering.eval.axbench_base.synth.continuation_guards import (
    ResponseEnvelope,
    prefix_echo_text,
    prefix_suffix_echo_chars,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def _frozen_semantic_identity(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _frozen_make_envelope(
    kind: str,
    identity: object,
    *,
    untrusted_texts: Sequence[str],
) -> ResponseEnvelope:
    normalized_kind = re.sub(r"[^A-Z0-9_]", "_", kind.upper())
    for salt in range(10_000):
        digest = _frozen_semantic_identity(
            {
                "protocol_version": current.STAGE0C_API_PROTOCOL_VERSION,
                "kind": normalized_kind,
                "identity": identity,
                "salt": salt,
            }
        )[:24].upper()
        begin = f"[[AXBENCH_{normalized_kind}_BEGIN_{digest}]]"
        end = f"[[AXBENCH_{normalized_kind}_END_{digest}]]"
        if all(
            begin not in text
            and end not in text
            and end[:-1] not in text
            for text in untrusted_texts
        ):
            return ResponseEnvelope(begin=begin, end=end)
    raise AxBenchBaseError(
        "Could not derive Stage 0c API protocol markers absent from the "
        "untrusted prompt inputs after 10000 salts"
    )


def _frozen_prefix_suffix_echo_chars(
    prefix_text: str,
    continuation: str,
) -> int:
    continuation_body = continuation.lstrip()
    max_overlap = min(len(prefix_text), len(continuation_body))
    for overlap in range(max_overlap, 15, -1):
        if prefix_text.endswith(continuation_body[:overlap]):
            return overlap
    return 0


def _frozen_parse_envelope(
    response: str,
    envelope: ResponseEnvelope,
    *,
    tolerate_end_marker_variants: bool,
    record: Callable[..., Any],
) -> str:
    if not isinstance(response, str):
        raise AxBenchBaseError(
            "Stage 0c API response must be a string, got "
            f"{type(response).__name__}"
        )
    begin_count = response.count(envelope.begin)
    end_count = response.count(envelope.end)
    if begin_count != 1:
        raise AxBenchBaseError(
            "Stage 0c API response must contain exactly one matching begin and "
            f"end marker, found begin={begin_count}, end={end_count}"
        )
    begin_at = response.index(envelope.begin)
    payload_at = begin_at + len(envelope.begin)
    recognized_end = envelope.end
    if end_count:
        end_at = response.index(envelope.end)
        if end_count > 1 and tolerate_end_marker_variants:
            record(
                "stage0c.api.normalized_duplicate_end_marker_chatter",
                end_count - 1,
            )
        elif end_count != 1:
            raise AxBenchBaseError(
                "Stage 0c API response must contain exactly one matching "
                "begin and end marker, found "
                f"begin={begin_count}, end={end_count}"
            )
    elif tolerate_end_marker_variants:
        recognized_end = envelope.end[:-1]
        truncated_count = response.count(recognized_end)
        if not truncated_count:
            raise AxBenchBaseError(
                "Stage 0c API response must contain exactly one matching "
                "begin and end marker, found "
                f"begin={begin_count}, end={end_count}"
            )
        end_at = response.index(recognized_end)
        record("stage0c.api.normalized_truncated_end_marker", 1)
        if truncated_count > 1:
            record(
                "stage0c.api.normalized_duplicate_end_marker_chatter",
                truncated_count - 1,
            )
    else:
        raise AxBenchBaseError(
            "Stage 0c API response must contain exactly one matching begin and "
            f"end marker, found begin={begin_count}, end={end_count}"
        )
    if end_at < payload_at:
        raise AxBenchBaseError(
            "Stage 0c API response end marker occurs before its begin marker"
        )
    if begin_at > 0 or end_at + len(recognized_end) < len(response):
        record("stage0c.api.outside_envelope_chatter", 1)
    return response[payload_at:end_at]


def _outcome(call: Callable[[], str]) -> tuple[str, str, str]:
    try:
        return ("return", type(result := call()).__name__, result)
    except Exception as exc:  # noqa: BLE001 - differential captures taxonomy
        return ("raise", type(exc).__name__, str(exc))


def test_extracted_envelope_derivation_is_exact_for_collisions_and_unicode() -> None:
    identities = [
        {"feature": 7, "seed": 9},
        {"nested": ["日本語", {"attempt": 3}]},
        ["list", 1, None, False],
    ]
    for index, identity in enumerate(identities):
        initial = _frozen_make_envelope(
            f"continuation-{index}",
            identity,
            untrusted_texts=[],
        )
        for untrusted in (
            [],
            ["ordinary source"],
            [initial.begin],
            [initial.end],
            [initial.end[:-1]],
        ):
            assert current._make_envelope(
                f"continuation-{index}",
                identity,
                untrusted_texts=untrusted,
            ) == _frozen_make_envelope(
                f"continuation-{index}",
                identity,
                untrusted_texts=untrusted,
            )


def test_extracted_prefix_helpers_are_exact_over_seeded_text_matrix() -> None:
    rng = random.Random(0x1067)
    alphabet = string.ascii_letters + string.digits + " \t\r\né日"
    for _ in range(500):
        prefix = "".join(
            rng.choice(alphabet) for _ in range(rng.randrange(0, 180))
        )
        continuation = "".join(
            rng.choice(alphabet) for _ in range(rng.randrange(0, 180))
        )
        assert prefix_echo_text(prefix) == prefix[-96:]
        assert prefix_suffix_echo_chars(
            prefix,
            continuation,
        ) == _frozen_prefix_suffix_echo_chars(prefix, continuation)


@pytest.mark.parametrize("tolerate", [False, True])
def test_extracted_envelope_parser_matches_returns_errors_and_counters(
    monkeypatch: pytest.MonkeyPatch,
    tolerate: bool,
) -> None:
    envelope = ResponseEnvelope("<BEGIN>", "<END>")
    responses: list[Any] = [
        "<BEGIN>payload<END>",
        "before<BEGIN>payload<END>after",
        "<BEGIN>payload<END><END>",
        "<BEGIN>payload<END",
        "<END><BEGIN>payload",
        "<BEGIN><BEGIN>payload<END>",
        "payload<END>",
        123,
    ]
    for response in responses:
        expected_counters: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        actual_counters: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def _expected_record(
            *args: Any,
            _target: list[tuple[tuple[Any, ...], dict[str, Any]]] = (
                expected_counters
            ),
            **kwargs: Any,
        ) -> None:
            _target.append((args, kwargs))

        def _actual_record(
            *args: Any,
            _target: list[tuple[tuple[Any, ...], dict[str, Any]]] = (
                actual_counters
            ),
            **kwargs: Any,
        ) -> None:
            _target.append((args, kwargs))

        def _expected_call(response: Any = response) -> str:
            return _frozen_parse_envelope(
                response,
                envelope,
                tolerate_end_marker_variants=tolerate,
                record=_expected_record,
            )

        def _actual_call(response: Any = response) -> str:
            return current._parse_envelope(
                response,
                envelope,
                tolerate_end_marker_variants=tolerate,
            )

        monkeypatch.setattr(current, "record_counter", _actual_record)
        expected = _outcome(_expected_call)
        actual = _outcome(_actual_call)
        assert actual == expected
        assert actual_counters == expected_counters
