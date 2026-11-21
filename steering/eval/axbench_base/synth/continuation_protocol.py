"""Authenticated request/response protocol for Stage 1 LLM continuations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from steering.eval.axbench_base.profiler import record_counter
from steering.eval.axbench_base.synth.continuation_guards import (
    ContinuationProtocolError,
    ResponseEnvelope,
    make_response_envelope,
    match_prefix_echo_framing,
    parse_response_envelope,
    prefix_echo_text,
    prefix_suffix_echo_chars,
    validate_no_source_echo_or_repeated_phrase,
)
from steering.eval.axbench_base.synth.response_parser import LlmSpecialTokenRefusal
from steering.eval.axbench_base.templates import render_untrusted_template
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.segmentation import resolve_bos_token_id, special_token_ids

STAGE1_CONTINUATION_PROTOCOL_VERSION = 1
_DISALLOWED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Stage1ContinuationRequest:
    """One deterministic attempt-specific Stage 1 prompt and response envelope."""

    prompt: str
    envelope: ResponseEnvelope


def _semantic_identity(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Stage 1 continuation identity is not canonical-JSON serializable: {exc}"
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_stage1_continuation_request(
    template: str,
    *,
    label: str,
    kind: str,
    identity: object,
    prefix_text: str,
    description: str,
) -> Stage1ContinuationRequest:
    """Render one attempt-specific prompt ending at its authenticated suffix."""
    if not isinstance(prefix_text, str) or not prefix_text:
        raise AxBenchBaseError(f"{label} prefix_text must be a non-empty string")
    if not template.endswith("{{PREFIX_SUFFIX}}\n"):
        raise AxBenchBaseError(
            f"{label} template must end with the trusted "
            "{{PREFIX_SUFFIX}} placeholder and exactly one newline"
        )
    envelope = make_response_envelope(
        kind,
        identity,
        untrusted_texts=(prefix_text, description),
        protocol_version=STAGE1_CONTINUATION_PROTOCOL_VERSION,
        semantic_identity=_semantic_identity,
        error_subject="Stage 1 continuation",
    )
    echo = prefix_echo_text(prefix_text)
    prompt = render_untrusted_template(
        template,
        label=label,
        BEGIN_MARKER=envelope.begin,
        END_MARKER=envelope.end,
        PREFIX_TEXT=prefix_text,
        PREFIX_SUFFIX=echo,
        ECHO_CHARS=len(echo),
        DESCRIPTION=description,
    )
    if not prompt.endswith(echo + "\n"):
        raise AxBenchBaseError(
            f"{label} rendered prompt does not end with the exact trusted "
            "prefix suffix and one newline"
        )
    return Stage1ContinuationRequest(prompt=prompt[:-1], envelope=envelope)


def _validate_continuation_text(
    continuation: str,
    *,
    prefix_text: str,
    subject: str,
) -> None:
    if not continuation or not continuation.strip():
        raise ContinuationProtocolError(f"{subject} is empty")
    if len(prefix_text) >= 32 and prefix_text in continuation:
        raise ContinuationProtocolError(
            f"{subject} repeated the locally retained candidate prefix"
        )
    suffix_echo_chars = prefix_suffix_echo_chars(prefix_text, continuation)
    if suffix_echo_chars:
        raise ContinuationProtocolError(
            f"{subject} repeated {suffix_echo_chars} character(s) from the "
            "candidate prefix suffix"
        )
    if _DISALLOWED_CONTROL_RE.search(continuation):
        raise ContinuationProtocolError(
            f"{subject} contains disallowed control characters"
        )
    if "AXBENCH_" in continuation:
        raise ContinuationProtocolError(
            f"{subject} contains a protocol-like marker"
        )
    try:
        validate_no_source_echo_or_repeated_phrase(
            continuation,
            source_text=prefix_text,
            subject=subject,
        )
    except AxBenchBaseError as exc:
        raise ContinuationProtocolError(str(exc)) from exc


def parse_stage1_continuation_response(
    *,
    response: str,
    envelope: ResponseEnvelope,
    prefix_text: str,
    output_length: int,
    tokenizer: Any,
    subject: str,
    normalization_counter_prefix: str | None = None,
    normalization_counter_attrs: dict[str, Any] | None = None,
) -> str:
    """Authenticate, strip, guard, tokenize, and revalidate one response."""

    def _record_normalization(name: str, value: int, **attrs: Any) -> None:
        if normalization_counter_prefix is None:
            return
        merged_attrs = dict(normalization_counter_attrs or {})
        merged_attrs.update(attrs)
        record_counter(name, value, **merged_attrs)

    payload = parse_response_envelope(
        response,
        envelope,
        tolerate_end_marker_variants=True,
        error_subject=subject,
        counter_prefix=normalization_counter_prefix,
        record_counter_fn=_record_normalization,
    )
    framing = match_prefix_echo_framing(payload, prefix_text)
    if framing is None:
        raise ContinuationProtocolError(
            f"{subject} must repeat the exact local prefix suffix immediately "
            "after the begin-marker newline and append continuation data "
            "directly at the fixed suffix boundary"
        )
    if framing.used_left_truncation:
        _record_normalization(
            f"{normalization_counter_prefix}.normalized_left_truncated_prefix_echo",
            1,
        )
    if framing.text not in {
        "\n" + framing.exact_echo,
        "\r\n" + framing.exact_echo,
    }:
        _record_normalization(
            f"{normalization_counter_prefix}.normalized_prefix_echo_framing",
            1,
        )
    framed_continuation = payload[len(framing.text) :]
    continuation = framed_continuation.rstrip(" \t\r\n")
    final_separator = framed_continuation[len(continuation) :]
    if final_separator != " ":
        _record_normalization(
            f"{normalization_counter_prefix}.normalized_end_marker_separator",
            1,
            separator_chars=len(final_separator),
        )
    _validate_continuation_text(
        continuation,
        prefix_text=prefix_text,
        subject=subject,
    )

    response_tokens = tokenizer.encode(continuation, add_special_tokens=False)
    polluting = special_token_ids(
        tokenizer,
        resolve_bos_token_id(tokenizer),
    ).intersection(response_tokens)
    if polluting:
        raise LlmSpecialTokenRefusal(
            f"{subject} encodes special token id(s) {sorted(polluting)} even "
            "with add_special_tokens=False"
        )
    decoded = tokenizer.decode(response_tokens[:output_length]).rstrip(" \t\r\n")
    _validate_continuation_text(
        decoded,
        prefix_text=prefix_text,
        subject=f"{subject} stored token window",
    )
    return decoded


def parse_stage1_continuation_or_none(
    *,
    response: str,
    request: Stage1ContinuationRequest,
    prefix_text: str,
    output_length: int,
    tokenizer: Any,
    subject: str,
    log_label: str,
    protocol_counter: str,
    special_token_counter: str,
    counter_attrs: dict[str, Any],
) -> str | None:
    """Return parsed text, or record and warn for retryable output failures."""
    try:
        return parse_stage1_continuation_response(
            response=response,
            envelope=request.envelope,
            prefix_text=prefix_text,
            output_length=output_length,
            tokenizer=tokenizer,
            subject=subject,
            normalization_counter_prefix=protocol_counter.removesuffix(
                "_failures"
            ),
            normalization_counter_attrs=counter_attrs,
        )
    except ContinuationProtocolError as exc:
        record_counter(protocol_counter, 1, **counter_attrs)
        # DEBUG, not WARNING. Protocol rejection is the NORMAL mechanism of
        # rejection sampling, not an anomaly: an observed run emitted ~150 of
        # these per minute (11,044 in one stage), which buries the genuinely
        # rare WARNINGs an operator needs to see and makes the log unusable at
        # default verbosity. The per-attempt detail stays available under
        # NANOCHAT_LOG_LEVEL=DEBUG, and the aggregate is always retained in
        # `record_counter` plus the per-concept termination summary.
        logger.debug(
            f"{log_label}: synth response protocol failure; skipping attempt: {exc}"
        )
        return None
    except LlmSpecialTokenRefusal as exc:
        record_counter(special_token_counter, 1, **counter_attrs)
        # Same reasoning: an expected rejection-sampling outcome, counted
        # rather than shouted.
        logger.debug(
            f"{log_label}: synth response refused, skipping attempt: {exc}"
        )
        return None
