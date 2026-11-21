from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.roundtrip_budgeting import (
    RoundtripGenerationBudgetError,
    extraction_token_ids_and_char_ends,
    reference_segment_generation_budget,
)
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets


def _segment_size(value: int) -> PositiveInt:
    return PositiveInt.parse(value)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids)

    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        input_ids = self.encode(text)
        out: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [
                (index, index + 1) for index in range(len(input_ids))
            ]
        return out


class _NestedCharTokenizer(_CharTokenizer):
    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        flat = super().__call__(
            text,
            padding=padding,
            return_offsets_mapping=return_offsets_mapping,
        )
        out: dict[str, Any] = {"input_ids": [flat["input_ids"]]}
        if return_offsets_mapping:
            out["offset_mapping"] = [flat["offset_mapping"]]
        return out


class _MismatchedOffsetTokenizer(_CharTokenizer):
    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        input_ids = self.encode(text)
        out: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(0, len(text))]
        return out


class _ChunkTargetTokenizer:
    _pieces = {
        1: "ab",
        2: "cd",
        3: "ef",
        4: "gh",
        9: "abcde",
    }

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)


class _BoundaryMergeTokenizer:
    """Tokenizer where continuation tokens decode differently after the prefix."""

    _decoded = {
        (1,): "a",
        (2,): "LONG",
        (2, 3): "LONGc",
        (2, 3, 4): "LONGcd",
        (1, 2): "ab",
        (1, 2, 3): "abc",
        (1, 2, 3, 4): "abcd",
    }

    def decode(self, ids: Sequence[int]) -> str:
        return self._decoded[tuple(int(token_id) for token_id in ids)]


class _BoundaryMergeExtractionTokenizer:
    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        if text == "a":
            input_ids = [1]
            offsets = [(0, 1)]
        elif text == "ab":
            input_ids = [9]
            offsets = [(0, 2)]
        else:
            raise AssertionError(f"unexpected text {text!r}")
        out: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class _RecordingCharTokenizer(_CharTokenizer):
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        self.seen_texts.append(text)
        return super().__call__(
            text,
            padding=padding,
            return_offsets_mapping=return_offsets_mapping,
        )


class _PieceTokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = dict(pieces)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._pieces[int(token_id)] for token_id in ids)


class _BosPrefixTokenizer(_CharTokenizer):
    def decode(self, ids: Sequence[int]) -> str:
        return "".join(
            "<bos>" if int(token_id) == 0 else chr(int(token_id)) for token_id in ids
        )


class _BosBoundaryMergeTokenizer:
    """Target tokenizer with visible BOS and context-dependent continuation text."""

    _decoded = {
        (0,): "<bos>",
        (0, 1): "<bos>a",
        (0, 1, 2): "<bos>ab",
        (1,): "a",
        (1, 2): "ab",
        (2,): "LONG",
    }

    def decode(self, ids: Sequence[int]) -> str:
        return self._decoded[tuple(int(token_id) for token_id in ids)]


def _char_prefix_tokens(prefix_text: str) -> list[int]:
    return _CharTokenizer().encode(prefix_text)


def _assert_budget_context_consistent(
    *,
    budget_max_new_tokens: int,
    prefix_tokens: Sequence[int],
    output_tokens: Sequence[int],
    target_tokenizer: Any,
    reference_char_count: int,
) -> None:
    full_tokens = [*prefix_tokens, *output_tokens]
    offsets = token_to_char_offsets(target_tokenizer, list(full_tokens))
    prefix_text = str(target_tokenizer.decode(list(prefix_tokens)))
    output_starts = offsets[len(prefix_tokens) : len(prefix_tokens) + len(output_tokens)]
    relative_starts = [int(start) - len(prefix_text) for start in output_starts]
    expected = sum(start < reference_char_count for start in relative_starts)
    assert budget_max_new_tokens == expected
    assert expected > 0
    assert relative_starts[expected - 1] < reference_char_count
    if expected < len(relative_starts):
        assert relative_starts[expected] >= reference_char_count


def test_extraction_token_ids_and_char_ends_accepts_batched_offsets() -> None:
    token_ids, char_ends = extraction_token_ids_and_char_ends(
        "abc",
        _NestedCharTokenizer(),
    )

    assert token_ids == [97, 98, 99]
    assert char_ends == [1, 2, 3]


def test_extraction_token_ids_and_char_ends_falls_back_on_bad_offsets() -> None:
    token_ids, char_ends = extraction_token_ids_and_char_ends(
        "abcd",
        _MismatchedOffsetTokenizer(),
    )

    assert token_ids == [97, 98, 99, 100]
    assert char_ends == [1, 2, 3, 4]


@settings(max_examples=500, derandomize=True)
@given(st.integers(min_value=1, max_value=2_000))
def test_reference_budget_matches_character_token_segment_geometry(
    segment_size: int,
) -> None:
    prefix_text = "abcde"
    output_tokens = [ord("x")] * 200
    budget = reference_segment_generation_budget(
        prefix_text=prefix_text,
        prefix_tokens=_char_prefix_tokens(prefix_text),
        reference_output_tokens=output_tokens,
        target_tokenizer=_CharTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(segment_size),
    )
    total_extraction_tokens = len(prefix_text) + len(output_tokens)
    expected_segment_end = min(
        total_extraction_tokens,
        ((len(prefix_text) // segment_size) + 1) * segment_size,
    )
    expected_reference_tokens = expected_segment_end - len(prefix_text)
    assert budget.segment_end_extraction_token_count == expected_segment_end
    assert budget.reference_char_count == expected_reference_tokens
    assert budget.max_new_tokens == expected_reference_tokens


def test_reference_budget_generates_remaining_tokens_to_segment_end() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="abcde",
        prefix_tokens=_char_prefix_tokens("abcde"),
        reference_output_tokens=[102, 103, 104, 105, 106],
        target_tokenizer=_CharTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(4),
    )

    assert budget.prefix_extraction_token_count == 5
    assert budget.segment_end_extraction_token_count == 8
    assert budget.reference_char_count == 3
    assert budget.output_tokens == (102, 103, 104)
    assert budget.max_new_tokens == 3
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=_char_prefix_tokens("abcde"),
        output_tokens=[102, 103, 104, 105, 106],
        target_tokenizer=_CharTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_uses_full_segment_when_prefix_on_boundary() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="abcd",
        prefix_tokens=_char_prefix_tokens("abcd"),
        reference_output_tokens=[101, 102, 103, 104, 105],
        target_tokenizer=_CharTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(4),
    )

    assert budget.prefix_extraction_token_count == 4
    assert budget.segment_end_extraction_token_count == 8
    assert budget.reference_char_count == 4
    assert budget.output_tokens == (101, 102, 103, 104)
    assert budget.max_new_tokens == 4
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=_char_prefix_tokens("abcd"),
        output_tokens=[101, 102, 103, 104, 105],
        target_tokenizer=_CharTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_allows_decoded_bos_before_prefix_text() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="abcd",
        prefix_tokens=[0, *_char_prefix_tokens("abcd")],
        reference_output_tokens=[101, 102, 103, 104, 105],
        target_tokenizer=_BosPrefixTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(4),
    )

    assert budget.prefix_extraction_token_count == 4
    assert budget.segment_end_extraction_token_count == 8
    assert budget.reference_char_count == 4
    assert budget.output_tokens == (101, 102, 103, 104)
    assert budget.max_new_tokens == 4
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[0, *_char_prefix_tokens("abcd")],
        output_tokens=[101, 102, 103, 104, 105],
        target_tokenizer=_BosPrefixTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_tokenizes_bos_free_prefix_plus_contextual_suffix_once() -> None:
    extraction_tokenizer = _RecordingCharTokenizer()

    budget = reference_segment_generation_budget(
        prefix_text="a",
        prefix_tokens=[0, 1],
        reference_output_tokens=[2],
        target_tokenizer=_BosBoundaryMergeTokenizer(),
        extraction_tokenizer=extraction_tokenizer,
        reference_segment_size=_segment_size(2),
    )

    assert extraction_tokenizer.seen_texts == ["ab"]
    assert budget.prefix_extraction_token_count == 1
    assert budget.segment_end_extraction_token_count == 2
    assert budget.reference_char_count == 1
    assert budget.output_tokens == (2,)
    assert budget.max_new_tokens == 1
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[0, 1],
        output_tokens=[2],
        target_tokenizer=_BosBoundaryMergeTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_rejects_prefix_text_that_does_not_match_tokens() -> None:
    with pytest.raises(RoundtripGenerationBudgetError, match="prefix_text"):
        reference_segment_generation_budget(
            prefix_text="abx",
            prefix_tokens=_char_prefix_tokens("abc"),
            reference_output_tokens=[100],
            target_tokenizer=_CharTokenizer(),
            extraction_tokenizer=_CharTokenizer(),
            reference_segment_size=_segment_size(4),
        )


def test_reference_budget_converts_extraction_boundary_to_target_tokens() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="abcde",
        prefix_tokens=[9],
        reference_output_tokens=[1, 2, 3, 4],
        target_tokenizer=_ChunkTargetTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(8),
    )

    assert budget.prefix_extraction_token_count == 5
    assert budget.segment_end_extraction_token_count == 8
    assert budget.reference_char_count == 3
    assert budget.output_tokens == (1, 2)
    assert budget.max_new_tokens == 2
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[9],
        output_tokens=[1, 2, 3, 4],
        target_tokenizer=_ChunkTargetTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_uses_contextual_target_token_offsets() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="a",
        prefix_tokens=[1],
        reference_output_tokens=[2, 3, 4],
        target_tokenizer=_BoundaryMergeTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(3),
    )

    assert budget.prefix_extraction_token_count == 1
    assert budget.segment_end_extraction_token_count == 3
    assert budget.reference_char_count == 2
    assert budget.output_tokens == (2, 3)
    assert budget.max_new_tokens == 2
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[1],
        output_tokens=[2, 3, 4],
        target_tokenizer=_BoundaryMergeTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_counts_prefix_tokens_from_full_extraction_text() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="a",
        prefix_tokens=[97],
        reference_output_tokens=[98],
        target_tokenizer=_CharTokenizer(),
        extraction_tokenizer=_BoundaryMergeExtractionTokenizer(),
        reference_segment_size=_segment_size(1),
    )

    assert budget.prefix_extraction_token_count == 0
    assert budget.segment_end_extraction_token_count == 1
    assert budget.reference_char_count == 1
    assert budget.output_tokens == (98,)
    assert budget.max_new_tokens == 1


def test_reference_budget_excludes_token_starting_at_segment_boundary() -> None:
    pieces = {
        10: "prefix",
        11: "ab",
        12: "cd",
        13: "ef",
    }
    budget = reference_segment_generation_budget(
        prefix_text="prefix",
        prefix_tokens=[10],
        reference_output_tokens=[11, 12, 13],
        target_tokenizer=_PieceTokenizer(pieces),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(len("prefix") + len("abcd")),
    )

    assert budget.reference_char_count == len("abcd")
    assert budget.output_tokens == (11, 12)
    assert budget.max_new_tokens == 2
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[10],
        output_tokens=[11, 12, 13],
        target_tokenizer=_PieceTokenizer(pieces),
        reference_char_count=budget.reference_char_count,
    )


@pytest.mark.parametrize(
    (
        "prefix_id",
        "prefix_text",
        "output_pieces",
        "reference_char_count",
        "expected_output_tokens",
    ),
    [
        (
            100,
            "नमस्ते ",
            {
                101: "भारत",
                102: " १२३",
                103: " में",
            },
            len("भारत "),
            (101, 102),
        ),
        (
            200,
            "தமிழ் ",
            {
                201: "எண்",
                202: " 123",
                203: " முடிவு",
            },
            len("எண் 1"),
            (201, 202),
        ),
        (
            300,
            "Order ",
            {
                301: "123",
                302: "45",
                303: "६७",
                304: "890",
            },
            len("12345६"),
            (301, 302, 303),
        ),
        (
            400,
            "한국어 ",
            {
                401: "텍스트",
                402: "입니다",
                403: " 2026",
            },
            len("텍스트입"),
            (401, 402),
        ),
    ],
    ids=[
        "hindi_devanagari",
        "tamil",
        "mixed_ascii_devanagari_numbers",
        "korean_hangul",
    ],
)
def test_reference_budget_handles_multilingual_and_numeric_text(
    prefix_id: int,
    prefix_text: str,
    output_pieces: dict[int, str],
    reference_char_count: int,
    expected_output_tokens: tuple[int, ...],
) -> None:
    target_tokenizer = _PieceTokenizer({prefix_id: prefix_text, **output_pieces})
    output_tokens = tuple(output_pieces)

    budget = reference_segment_generation_budget(
        prefix_text=prefix_text,
        prefix_tokens=[prefix_id],
        reference_output_tokens=output_tokens,
        target_tokenizer=target_tokenizer,
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(len(prefix_text) + reference_char_count),
    )

    assert budget.prefix_extraction_token_count == len(prefix_text)
    assert budget.reference_char_count == reference_char_count
    assert budget.segment_end_extraction_token_count == (
        len(prefix_text) + reference_char_count
    )
    assert budget.output_tokens == expected_output_tokens
    assert budget.max_new_tokens == len(expected_output_tokens)
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[prefix_id],
        output_tokens=output_tokens,
        target_tokenizer=target_tokenizer,
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_counts_zero_width_contextual_output_tokens() -> None:
    class _ZeroWidthTokenizer:
        _decoded = {
            (1,): "a",
            (2,): "",
            (2, 3): "b",
            (1, 2): "a",
            (1, 2, 3): "ab",
        }

        def decode(self, ids: Sequence[int]) -> str:
            return self._decoded[tuple(int(token_id) for token_id in ids)]

    budget = reference_segment_generation_budget(
        prefix_text="a",
        prefix_tokens=[1],
        reference_output_tokens=[2, 3],
        target_tokenizer=_ZeroWidthTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(2),
    )

    assert budget.reference_char_count == 1
    assert budget.output_tokens == (2, 3)
    assert budget.max_new_tokens == 2
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=[1],
        output_tokens=[2, 3],
        target_tokenizer=_ZeroWidthTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


def test_reference_budget_rejects_non_prefix_stable_contextual_decode() -> None:
    class _NonPrefixStableTokenizer:
        _decoded = {
            (1,): "left",
            (2,): "right",
            (1, 2): "merged",
        }

        def decode(self, ids: Sequence[int]) -> str:
            return self._decoded[tuple(int(token_id) for token_id in ids)]

    with pytest.raises(RoundtripGenerationBudgetError, match="full target context"):
        reference_segment_generation_budget(
            prefix_text="left",
            prefix_tokens=[1],
            reference_output_tokens=[2],
            target_tokenizer=_NonPrefixStableTokenizer(),
            extraction_tokenizer=_CharTokenizer(),
            reference_segment_size=_segment_size(2),
        )


def test_reference_budget_caps_at_short_reference() -> None:
    budget = reference_segment_generation_budget(
        prefix_text="abcde",
        prefix_tokens=_char_prefix_tokens("abcde"),
        reference_output_tokens=[102, 103],
        target_tokenizer=_CharTokenizer(),
        extraction_tokenizer=_CharTokenizer(),
        reference_segment_size=_segment_size(4),
    )

    assert budget.segment_end_extraction_token_count == 7
    assert budget.reference_char_count == 2
    assert budget.output_tokens == (102, 103)
    assert budget.max_new_tokens == 2
    _assert_budget_context_consistent(
        budget_max_new_tokens=budget.max_new_tokens,
        prefix_tokens=_char_prefix_tokens("abcde"),
        output_tokens=[102, 103],
        target_tokenizer=_CharTokenizer(),
        reference_char_count=budget.reference_char_count,
    )


@settings(max_examples=500, derandomize=True)
@given(st.one_of(st.integers(max_value=0), st.booleans()))
def test_reference_budget_rejects_invalid_segment_size(
    invalid_segment_size: int,
) -> None:
    runtime_value: Any = invalid_segment_size
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="reference_segment_size",
    ):
        reference_segment_generation_budget(
            prefix_text="abc",
            prefix_tokens=_char_prefix_tokens("abc"),
            reference_output_tokens=[100],
            target_tokenizer=_CharTokenizer(),
            extraction_tokenizer=_CharTokenizer(),
            reference_segment_size=runtime_value,
        )


def test_reference_budget_rejects_empty_prefix_tokens() -> None:
    with pytest.raises(RoundtripGenerationBudgetError, match="prefix_tokens"):
        reference_segment_generation_budget(
            prefix_text="abc",
            prefix_tokens=[],
            reference_output_tokens=[100],
            target_tokenizer=_CharTokenizer(),
            extraction_tokenizer=_CharTokenizer(),
            reference_segment_size=_segment_size(4),
        )
