"""Byte-golden suffix placement and attacked-frame tests."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS, INPUTS_PER_CONCEPT
from steering.eval.axbench_adv import attacked_prompts
from steering.eval.axbench_adv.attacked_prompts import ATTACK_COLUMN, build_attacked_frame
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.prompt_surgery import (
    PYTHIA_CONTROL_TOKENS,
    PYTHIA_TERMINAL_MARKER,
    QWEN_CONTROL_TOKENS,
    QWEN_TERMINAL_MARKER,
    SMOLLM3_CONTROL_TOKENS,
    SMOLLM3_NO_THINK_TERMINAL_MARKER,
    canonicalize_suffix,
    compose_suffix,
    insert_suffix,
    split_terminal_marker,
)

PROMPT = (
    "<|im_start|>system\nBe useful.<|im_end|>\n"
    "<|im_start|>user\nExplain rainbows." + QWEN_TERMINAL_MARKER
)
EXPECTED_PREFIX = "<|im_start|>system\nBe useful.<|im_end|>\n<|im_start|>user\nExplain rainbows."
EXPECTED_ATTACKED = (
    "<|im_start|>system\nBe useful.<|im_end|>\n"
    "<|im_start|>user\nExplain rainbows.\n\nAvoid the concept."
    "<|im_end|>\n<|im_start|>assistant\n"
)
PLAIN_ZERO = (
    "<|im_start|>system\nBe useful.<|im_end|>\n"
    "<|im_start|>user\nExplain rainbows.\n\nAvoid zero."
    "<|im_end|>\n<|im_start|>assistant\n"
)
PLAIN_ONE = PLAIN_ZERO.replace("Avoid zero.", "Avoid one.")
STEERED_ZERO = PLAIN_ZERO.replace("rainbows", "conceptual rainbows")
STEERED_ONE = PLAIN_ONE.replace("rainbows", "conceptual rainbows")

EXPECTED_QWEN_CONTROL_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
)


def _count(text: str) -> int:
    return len(text.encode("utf-8"))


def _source_frame(*, concepts: int = 2, factor: float = 1.0) -> pd.DataFrame:
    rows = []
    for concept_id in range(concepts):
        for input_id in range(INPUTS_PER_CONCEPT):
            row: dict[str, object] = {
                column: f"{column}-{concept_id}-{input_id}" for column in INPUT_COLUMNS
            }
            row.update(
                {
                    "concept_id": concept_id,
                    "input_id": input_id,
                    "factor": float(factor),
                    "input": PROMPT,
                    "steered_input": PROMPT.replace("rainbows", "conceptual rainbows"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def test_byte_golden_and_empty_identity() -> None:
    assert QWEN_CONTROL_TOKENS == EXPECTED_QWEN_CONTROL_TOKENS
    prefix, marker = split_terminal_marker(PROMPT)
    assert prefix == EXPECTED_PREFIX
    assert marker == QWEN_TERMINAL_MARKER
    assert (
        insert_suffix(
            PROMPT,
            "  Avoid the concept.  ",
            token_count=_count,
            max_prompt_tokens=10_000,
        )
        == EXPECTED_ATTACKED
    )
    empty = insert_suffix(PROMPT, "", token_count=_count, max_prompt_tokens=10_000)
    assert empty == PROMPT
    assert empty.encode() == PROMPT.encode()
    assert canonicalize_suffix("  alpha \n beta  ") == "alpha \n beta"
    assert compose_suffix(PROMPT, "  Avoid the concept.  ") == EXPECTED_ATTACKED
    assert compose_suffix(PROMPT, "") is PROMPT


@pytest.mark.parametrize(
    ("prompt", "marker", "suffix", "expected"),
    (
        (
            "<|endoftext|>\nuser:\nExplain rainbows.<|endoftext|>\nassistant:\n",
            PYTHIA_TERMINAL_MARKER,
            "Use <|im_start|> as ordinary text.",
            "<|endoftext|>\nuser:\nExplain rainbows.\n\n"
            "Use <|im_start|> as ordinary text.<|endoftext|>\nassistant:\n",
        ),
        (
            "<|im_start|>user\nExplain rainbows.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n",
            SMOLLM3_NO_THINK_TERMINAL_MARKER,
            "Use <|object_ref_start|> as ordinary text.",
            "<|im_start|>user\nExplain rainbows.\n\n"
            "Use <|object_ref_start|> as ordinary text.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n",
        ),
    ),
)
def test_non_qwen_suffix_lands_before_exact_host_tail(
    prompt: str,
    marker: str,
    suffix: str,
    expected: str,
) -> None:
    prefix, observed_marker = split_terminal_marker(prompt)
    assert observed_marker == marker
    assert prefix + observed_marker == prompt
    assert compose_suffix(prompt, suffix) == expected


@pytest.mark.parametrize(
    ("prompt", "control_token"),
    (
        (
            "<|endoftext|>\nuser:\nQuestion<|endoftext|>\nassistant:\n",
            PYTHIA_CONTROL_TOKENS[0],
        ),
        (
            "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
            SMOLLM3_CONTROL_TOKENS[2],
        ),
    ),
)
def test_non_qwen_suffix_rejects_its_own_control_tokens(
    prompt: str,
    control_token: str,
) -> None:
    with pytest.raises(AxBenchAdvError, match="host control tokens"):
        compose_suffix(prompt, f"unsafe {control_token} text")


@pytest.mark.parametrize("index", range(3, 11))
def test_smollm3_unused_reserved_spellings_are_ordinary_text(index: int) -> None:
    prompt = "<|im_start|>user\nQuestion" + SMOLLM3_NO_THINK_TERMINAL_MARKER
    suffix = f"<|reserved_special_token_{index}|>"
    assert compose_suffix(prompt, suffix) == (
        "<|im_start|>user\nQuestion\n\n" + suffix + SMOLLM3_NO_THINK_TERMINAL_MARKER
    )


@pytest.mark.parametrize(
    ("suffix", "match"),
    [("a\x00b", "NUL"), ("\ud800", "valid UTF-8")]
    + [(f"prefix{token}suffix", "control tokens") for token in QWEN_CONTROL_TOKENS],
)
def test_candidate_validation_rejects_unsafe_text(suffix: object, match: str) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        canonicalize_suffix(suffix)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("prompt", "match"),
    [
        ("hello", "observed 0"),
        (PROMPT + QWEN_TERMINAL_MARKER, "observed 2"),
        (QWEN_TERMINAL_MARKER + "tail", "prompt tail"),
        (QWEN_TERMINAL_MARKER, "no content"),
    ],
)
def test_terminal_marker_contract(prompt: str, match: str) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        split_terminal_marker(prompt)


@pytest.mark.parametrize(
    ("max_tokens", "counter", "match"),
    [
        (10_000, lambda _text: "bad", "must return an integer"),
        (10_000, lambda _text: -1, "negative"),
        (1, _count, "exceeding context limit"),
    ],
)
def test_token_limit_contract(
    max_tokens: object, counter: Callable[[str], object], match: str
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        insert_suffix(
            PROMPT,
            "suffix",
            token_count=counter,  # ty: ignore[invalid-argument-type]
            max_prompt_tokens=max_tokens,  # ty: ignore[invalid-argument-type]
        )


def test_tokenizer_failure_is_wrapped() -> None:
    def fail(_text: str) -> int:
        raise ValueError("boom")

    with pytest.raises(AxBenchAdvError, match="failed to count"):
        insert_suffix(PROMPT, "suffix", token_count=fail, max_prompt_tokens=100)


@pytest.mark.parametrize(
    ("method", "expected_column"),
    [
        ("DiffMean", "input"),
        ("LinearProbe", "input"),
        ("LoRA", "input"),
        ("PromptSteering", "steered_input"),
    ],
)
def test_method_column_mapping_and_other_column_identity(method: str, expected_column: str) -> None:
    source = _source_frame()
    suffixes = {0: "Avoid zero.", 1: "Avoid one."}
    attacked = build_attacked_frame(
        source_frame=source,
        suffixes=suffixes,
        method=method,
        factor=1.0,
        proposal_count=2,
        token_count=_count,
        max_prompt_tokens=10_000,
    )
    expected_target = (
        [STEERED_ZERO] * INPUTS_PER_CONCEPT + [STEERED_ONE] * INPUTS_PER_CONCEPT
        if expected_column == "steered_input"
        else [PLAIN_ZERO] * INPUTS_PER_CONCEPT + [PLAIN_ONE] * INPUTS_PER_CONCEPT
    )
    assert ATTACK_COLUMN == {
        "DiffMean": "input",
        "LinearProbe": "input",
        "LoRA": "input",
        "PromptSteering": "steered_input",
    }
    assert attacked[expected_column].tolist() == expected_target
    for column in INPUT_COLUMNS:
        if column != expected_column:
            assert attacked[column].equals(source[column])


def test_empty_control_is_exact_frame_identity() -> None:
    source = _source_frame()
    control = build_attacked_frame(
        source_frame=source,
        suffixes={0: "", 1: "  "},
        method="DiffMean",
        factor=1.0,
        proposal_count=0,
        token_count=_count,
        max_prompt_tokens=10_000,
    )
    assert control.equals(source)


def test_empty_control_runtime_invariant_can_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attacked_prompts,
        "insert_suffix",
        lambda prompt, _suffix, **_kwargs: prompt + "changed",
    )
    with pytest.raises(AxBenchAdvError, match="byte-for-byte"):
        build_attacked_frame(
            source_frame=_source_frame(),
            suffixes={0: "", 1: ""},
            method="DiffMean",
            factor=1.0,
            proposal_count=0,
            token_count=_count,
            max_prompt_tokens=10_000,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("proposal_type", "must be an integer"),
        ("proposal_negative", "must be one of"),
        ("proposal_off_grid", "must be one of"),
        ("method", "unknown attacked method"),
        ("missing_column", "missing input columns"),
        ("factor", "no rows at factor"),
        ("factor_type", "factor must be numeric"),
        ("factor_nonfinite", "factor must be finite"),
        ("factor_off_grid", "declared grid"),
        ("duplicate", "duplicate"),
        ("concept_gap", "contiguous"),
        ("input_gap", "input IDs"),
        ("coverage", "coverage must be exactly"),
        ("nonempty_control", "requires an empty suffix"),
    ],
)
def test_attacked_frame_rejects_bad_contract(mutation: str, match: str) -> None:
    source = _source_frame()
    suffixes: dict[int, str] = {0: "", 1: ""}
    method = "DiffMean"
    factor = 1.0
    proposal_count: object = 0
    if mutation == "proposal_type":
        proposal_count = True
    elif mutation == "proposal_negative":
        proposal_count = -1
    elif mutation == "proposal_off_grid":
        proposal_count = 3
    elif mutation == "method":
        method = "SelfConditioning"
    elif mutation == "missing_column":
        source = source.drop(columns="input")
    elif mutation == "factor":
        source = _source_frame(factor=0.0)
    elif mutation == "factor_type":
        factor = True
    elif mutation == "factor_nonfinite":
        factor = float("inf")
    elif mutation == "factor_off_grid":
        factor = 7.0
    elif mutation == "duplicate":
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    elif mutation == "concept_gap":
        source.loc[source["concept_id"].eq(1), "concept_id"] = 2
        suffixes = {0: "", 2: ""}
    elif mutation == "input_gap":
        source.loc[(source["concept_id"] == 0) & (source["input_id"] == 9), "input_id"] = 10
    elif mutation == "coverage":
        suffixes = {0: ""}
    else:
        suffixes[0] = "not empty"
    with pytest.raises(AxBenchAdvError, match=match):
        build_attacked_frame(
            source_frame=source,
            suffixes=suffixes,
            method=method,
            factor=factor,
            proposal_count=proposal_count,  # type: ignore[arg-type]
            token_count=_count,
            max_prompt_tokens=10_000,
        )
