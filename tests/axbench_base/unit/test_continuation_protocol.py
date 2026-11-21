import json
from typing import Any

import pytest

from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.synth.continuation_guards import (
    PREFIX_ECHO_SEPARATORS,
    ContinuationProtocolError,
    prefix_echo_text,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    Stage1ContinuationRequest,
    build_stage1_continuation_request,
    parse_stage1_continuation_or_none,
    parse_stage1_continuation_response,
)
from steering.eval.axbench_base.synth.response_parser import (
    LlmSpecialTokenRefusal,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base.stage1_protocol_helpers import TEST_STAGE1_TEMPLATE


class _CharTokenizer:
    bos_token_id = 0
    all_special_ids = [0]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 1) for token_id in token_ids)


class _SpecialAwareTokenizer(_CharTokenizer):
    all_special_ids = [0, 999]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [999 if char == "§" else ord(char) + 1 for char in text]


class _PostDecodeCopyTokenizer(_CharTokenizer):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del text, add_special_tokens
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        del token_ids
        return "0123456789abcdef copied after decoding"


def _request(
    *,
    prefix: str,
    identity: object | None = None,
    description: str = "ordinary description",
) -> Stage1ContinuationRequest:
    return build_stage1_continuation_request(
        TEST_STAGE1_TEMPLATE,
        label="test continuation",
        kind="stage1b_rich",
        identity={"attempt": 0} if identity is None else identity,
        prefix_text=prefix,
        description=description,
    )


def _response(
    request: Stage1ContinuationRequest,
    *,
    prefix: str,
    continuation: str,
    separator: str = "\n",
    left_truncation: int = 0,
    end: str | None = None,
) -> str:
    echo = prefix_echo_text(prefix)[left_truncation:]
    return (
        request.envelope.begin
        + separator
        + echo
        + continuation
        + " "
        + (request.envelope.end if end is None else end)
    )


def _parse(
    request: Stage1ContinuationRequest,
    *,
    prefix: str,
    response: str,
    output_length: int = 10_000,
    tokenizer: Any | None = None,
) -> str:
    return parse_stage1_continuation_response(
        response=response,
        envelope=request.envelope,
        prefix_text=prefix,
        output_length=output_length,
        tokenizer=_CharTokenizer() if tokenizer is None else tokenizer,
        subject="test response",
    )


def test_request_is_deterministic_attempt_scoped_and_ends_at_exact_suffix() -> None:
    prefix = ("long trusted context " * 10) + "arbitrary midword bou"
    first = _request(prefix=prefix, identity={"attempt": 0, "seed": 7})
    repeat = _request(prefix=prefix, identity={"seed": 7, "attempt": 0})
    next_attempt = _request(prefix=prefix, identity={"attempt": 1, "seed": 7})

    assert first == repeat
    assert first.envelope != next_attempt.envelope
    assert first.prompt.endswith(prefix_echo_text(prefix))
    assert not first.prompt.endswith("\n")


def test_request_rejects_empty_prefix_because_it_cannot_authenticate_an_echo() -> None:
    with pytest.raises(AxBenchBaseError, match="prefix_text must be a non-empty string"):
        _request(prefix="")


@pytest.mark.parametrize(
    "template",
    [
        TEST_STAGE1_TEMPLATE.removesuffix("\n"),
        TEST_STAGE1_TEMPLATE + "\n",
        TEST_STAGE1_TEMPLATE + "trailing text\n",
    ],
)
def test_request_rejects_template_drift_after_trusted_suffix(template: str) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="must end with the trusted.*PREFIX_SUFFIX",
    ):
        build_stage1_continuation_request(
            template,
            label="test continuation",
            kind="stage1b_rich",
            identity={"attempt": 0},
            prefix_text="trusted prefix",
            description="ordinary description",
        )


@pytest.mark.parametrize("identity", [{"not-json": {1, 2}}, {"nan": float("nan")}])
def test_request_rejects_noncanonical_json_identity(identity: object) -> None:
    with pytest.raises(AxBenchBaseError, match="canonical-JSON serializable"):
        _request(prefix="trusted prefix", identity=identity)


def test_untrusted_placeholder_and_marker_shaped_text_remains_literal_data() -> None:
    prefix = "source {{END_MARKER}} [[AXBENCH_FAKE_BEGIN_ABC]] tail"
    description = "description {{PREFIX_SUFFIX}}"
    request = _request(prefix=prefix, description=description)

    assert prefix in request.prompt
    assert description in request.prompt
    assert request.prompt.endswith(prefix_echo_text(prefix))
    assert request.envelope.begin not in prefix
    assert request.envelope.end not in prefix


@pytest.mark.parametrize("separator", PREFIX_ECHO_SEPARATORS)
@pytest.mark.parametrize("left_truncation", [0, 1])
def test_all_audited_separator_and_echo_variants_preserve_continuation_exactly(
    separator: str,
    left_truncation: int,
) -> None:
    prefix = ("trusted context before boundary. " * 6) + "The microbio"
    continuation = "logical record continues."
    request = _request(prefix=prefix)

    assert _parse(
        request,
        prefix=prefix,
        response=_response(
            request,
            prefix=prefix,
            continuation=continuation,
            separator=separator,
            left_truncation=left_truncation,
        ),
    ) == continuation


@pytest.mark.parametrize(
    "continuation",
    [
        " leading space remains data.",
        "\nleading newline remains data.",
        ", punctuation remains attached.",
        "logical continuation of a midword boundary.",
    ],
)
def test_only_terminal_framing_whitespace_is_removed(
    continuation: str,
) -> None:
    prefix = ("trusted context " * 8) + "microbio"
    request = _request(prefix=prefix)

    assert _parse(
        request,
        prefix=prefix,
        response=_response(
            request,
            prefix=prefix,
            continuation=continuation + "\t\r\n",
        ),
    ) == continuation


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda req, prefix: prefix_echo_text(prefix) + "safe prose " + req.envelope.end,
        lambda req, prefix: req.envelope.begin
        + "\n"
        + prefix_echo_text(prefix)
        + "safe prose",
        lambda req, prefix: req.envelope.end
        + req.envelope.begin
        + "\n"
        + prefix_echo_text(prefix)
        + "safe prose",
        lambda req, prefix: req.envelope.begin
        + req.envelope.begin
        + "\n"
        + prefix_echo_text(prefix)
        + "safe prose "
        + req.envelope.end,
    ],
)
def test_missing_duplicate_or_misordered_envelope_is_rejected(
    response_factory,
) -> None:
    prefix = "trusted prefix long enough for framing"
    request = _request(prefix=prefix)
    with pytest.raises(ContinuationProtocolError):
        _parse(
            request,
            prefix=prefix,
            response=response_factory(request, prefix),
        )


def test_duplicate_end_chatter_uses_first_end_as_stage0c_does() -> None:
    prefix = "trusted prefix long enough for framing"
    request = _request(prefix=prefix)
    response = (
        _response(
            request,
            prefix=prefix,
            continuation="accepted prose.",
        )
        + request.envelope.end
    )
    assert _parse(request, prefix=prefix, response=response) == "accepted prose."


def test_exact_one_character_end_marker_truncation_is_accepted() -> None:
    prefix = "trusted prefix long enough for framing"
    request = _request(prefix=prefix)
    response = _response(
        request,
        prefix=prefix,
        continuation="accepted prose.",
        end=request.envelope.end[:-1],
    )
    assert _parse(request, prefix=prefix, response=response) == "accepted prose."


def test_two_character_end_marker_truncation_is_rejected() -> None:
    prefix = "trusted prefix long enough for framing"
    request = _request(prefix=prefix)
    with pytest.raises(ContinuationProtocolError):
        _parse(
            request,
            prefix=prefix,
            response=_response(
                request,
                prefix=prefix,
                continuation="accepted prose.",
                end=request.envelope.end[:-2],
            ),
        )


@pytest.mark.parametrize("mutation", ["missing", "changed", "right", "left2"])
def test_unaudited_echo_mutations_are_rejected(mutation: str) -> None:
    prefix = ("long context " * 12) + "boundary"
    request = _request(prefix=prefix)
    echo = prefix_echo_text(prefix)
    mutated = {
        "missing": "",
        "changed": "X" + echo[1:],
        "right": echo[:-1],
        "left2": echo[2:],
    }[mutation]
    response = (
        request.envelope.begin
        + "\n"
        + mutated
        + "safe forward prose. "
        + request.envelope.end
    )
    with pytest.raises(ContinuationProtocolError, match="exact local prefix"):
        _parse(request, prefix=prefix, response=response)


def test_short_prefix_uses_the_complete_prefix_as_echo() -> None:
    prefix = "short"
    request = _request(prefix=prefix)
    assert _parse(
        request,
        prefix=prefix,
        response=_response(
            request,
            prefix=prefix,
            continuation=" continuation.",
        ),
    ) == " continuation."


@pytest.mark.parametrize(("overlap", "accepted"), [(15, True), (16, False)])
def test_residual_prefix_suffix_overlap_boundary(
    overlap: int,
    accepted: bool,
) -> None:
    suffix = "abcdefghijklmnop"[-overlap:]
    prefix = ("unrelated context " * 8) + suffix
    continuation = suffix + " then entirely new prose."
    request = _request(prefix=prefix)
    response = _response(
        request,
        prefix=prefix,
        continuation=continuation,
    )
    if accepted:
        assert _parse(request, prefix=prefix, response=response) == continuation
    else:
        with pytest.raises(ContinuationProtocolError, match="repeated 16"):
            _parse(request, prefix=prefix, response=response)


@pytest.mark.parametrize(
    "continuation",
    [
        ("full retained source text " * 4),
        (
            "one two three four five six seven eight nine ten "
            "one two three four five six seven eight nine ten"
        ),
        "\x00disallowed control",
        "[[AXBENCH_FAKE_MARKER]]",
    ],
)
def test_copy_loop_control_and_protocol_pollution_are_rejected(
    continuation: str,
) -> None:
    prefix = ("full retained source text " * 4) + "boundary"
    request = _request(prefix=prefix)
    with pytest.raises(ContinuationProtocolError):
        _parse(
            request,
            prefix=prefix,
            response=_response(
                request,
                prefix=prefix,
                continuation=continuation,
            ),
        )


def test_full_response_special_token_is_checked_before_token_truncation() -> None:
    prefix = "trusted prefix long enough for framing"
    request = _request(prefix=prefix)
    response = _response(
        request,
        prefix=prefix,
        continuation="safe start followed later by § pollution",
    )
    with pytest.raises(LlmSpecialTokenRefusal):
        _parse(
            request,
            prefix=prefix,
            response=response,
            output_length=4,
            tokenizer=_SpecialAwareTokenizer(),
        )


def test_decoded_stored_window_is_revalidated() -> None:
    prefix = ("trusted context " * 6) + "0123456789abcdef"
    request = _request(prefix=prefix)
    with pytest.raises(ContinuationProtocolError, match="stored token window"):
        _parse(
            request,
            prefix=prefix,
            response=_response(
                request,
                prefix=prefix,
                continuation="safe provider prose",
            ),
            output_length=1,
            tokenizer=_PostDecodeCopyTokenizer(),
        )


def test_accepted_protocol_normalizations_are_attributed_to_attempt(
    tmp_path,
) -> None:
    prefix = ("trusted context " * 8) + "boundary"
    request = _request(prefix=prefix)
    response = (
        "provider preamble"
        + request.envelope.begin
        + " "
        + prefix_echo_text(prefix)[1:]
        + "accepted prose.\t"
        + request.envelope.end[:-1]
        + "provider epilogue"
    )
    profile_dir = tmp_path / "profile"

    with AxBenchProfiler(profile_dir):
        parsed = parse_stage1_continuation_or_none(
            response=response,
            request=request,
            prefix_text=prefix,
            output_length=10_000,
            tokenizer=_CharTokenizer(),
            subject="test response",
            log_label="test response",
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": 7,
                "attempt": 2,
                "content_class": "concept_rich",
            },
        )

    assert parsed == "accepted prose."
    events = [
        json.loads(line)
        for line in (profile_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    counters = {
        event["name"]: (event["value"], event["attrs"])
        for event in events
        if event["type"] == "counter"
    }
    expected_attrs = {
        "concept_id": 7,
        "attempt": 2,
        "content_class": "concept_rich",
    }
    assert counters == {
        "stage1b.continuation_protocol.normalized_truncated_end_marker": (
            1.0,
            expected_attrs,
        ),
        "stage1b.continuation_protocol.outside_envelope_chatter": (
            1.0,
            expected_attrs,
        ),
        "stage1b.continuation_protocol.normalized_left_truncated_prefix_echo": (
            1.0,
            expected_attrs,
        ),
        "stage1b.continuation_protocol.normalized_prefix_echo_framing": (
            1.0,
            expected_attrs,
        ),
        "stage1b.continuation_protocol.normalized_end_marker_separator": (
            1.0,
            {**expected_attrs, "separator_chars": 1},
        ),
    }
