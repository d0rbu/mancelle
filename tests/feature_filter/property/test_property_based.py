"""Hypothesis properties for sampler + parser."""
from itertools import groupby

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.interp_judge import (
    INTERP_JUDGE_MAX_SCORE,
    InterpJudgeRating,
    parse_interp_rating,
)
from tests.feature_filter.fixtures.payload_builder import build_bin_structured_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_bin_sampler_determinism(samples_per_bin: int) -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    a = sample_by_bins(payload, samples_per_bin=samples_per_bin, tokenizer=_Tok())
    b = sample_by_bins(payload, samples_per_bin=samples_per_bin, tokenizer=_Tok())
    assert a == b


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_bin_sampler_output_length_bound(samples_per_bin: int) -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    out = sample_by_bins(payload, samples_per_bin=samples_per_bin, tokenizer=_Tok())
    assert len(out) <= 5 * samples_per_bin


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_bin_sampler_peak_first_weakly_decreasing(samples_per_bin: int) -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    out = sample_by_bins(payload, samples_per_bin=samples_per_bin, tokenizer=_Tok())
    idxs = [s.bin_index for s in out]
    assert idxs == sorted(idxs, reverse=True)


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_bin_sampler_max_value_descending_within_band(samples_per_bin: int) -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    out = sample_by_bins(payload, samples_per_bin=samples_per_bin, tokenizer=_Tok())
    for _, group in groupby(out, key=lambda s: s.bin_index):
        mv = [s.max_value for s in group]
        assert mv == sorted(mv, reverse=True)


@given(st.text(max_size=500))
@settings(max_examples=100)
def test_parser_never_returns_out_of_range(text: str) -> None:
    r = parse_interp_rating(text)
    if r is not None:
        assert 0 <= r.interpretability <= INTERP_JUDGE_MAX_SCORE
        assert 0 <= r.consistency <= INTERP_JUDGE_MAX_SCORE


@given(st.integers(0, INTERP_JUDGE_MAX_SCORE), st.integers(0, INTERP_JUDGE_MAX_SCORE))
@settings(max_examples=200)
def test_bracketed_roundtrip(i: int, c: int) -> None:
    assert parse_interp_rating(
        f"Interpretability: [[{i}]]\nConsistency: [[{c}]]"
    ) == InterpJudgeRating(i, c)


@given(st.integers(0, INTERP_JUDGE_MAX_SCORE), st.integers(0, INTERP_JUDGE_MAX_SCORE))
@settings(max_examples=200)
def test_bare_roundtrip(i: int, c: int) -> None:
    assert parse_interp_rating(
        f"Interpretability: {i}\nConsistency: {c}"
    ) == InterpJudgeRating(i, c)


@given(
    st.integers(0, INTERP_JUDGE_MAX_SCORE),
    st.integers(0, INTERP_JUDGE_MAX_SCORE),
    st.sampled_from(["INTERPRETABILITY", "Interpretability", "interpretability"]),
    st.sampled_from(["CONSISTENCY", "Consistency", "consistency"]),
)
@settings(max_examples=100)
def test_case_mix_roundtrip(i: int, c: int, label_a: str, label_b: str) -> None:
    assert parse_interp_rating(
        f"{label_a}: [[{i}]]\n{label_b}: [[{c}]]"
    ) == InterpJudgeRating(i, c)


@given(st.integers(INTERP_JUDGE_MAX_SCORE + 1, 9999), st.integers(0, INTERP_JUDGE_MAX_SCORE))
@settings(max_examples=200)
def test_out_of_range_always_none(i: int, c: int) -> None:
    assert parse_interp_rating(
        f"Interpretability: [[{i}]]\nConsistency: [[{c}]]"
    ) is None


@given(
    st.integers(0, 4), st.integers(1, 9),
    st.integers(0, INTERP_JUDGE_MAX_SCORE),
)
@settings(max_examples=100)
def test_decimal_always_none(integer_part: int, frac_part: int, c: int) -> None:
    assert parse_interp_rating(
        f"Interpretability: [[{integer_part}.{frac_part}]]\nConsistency: [[{c}]]"
    ) is None


@given(
    st.integers(0, INTERP_JUDGE_MAX_SCORE),
    st.integers(0, INTERP_JUDGE_MAX_SCORE),
    st.sampled_from(["[[", "[", "]]", "]"]),
)
@settings(max_examples=200)
def test_half_open_brackets_always_none(i: int, c: int, bracket: str) -> None:
    bad = f"{bracket}{i}" if bracket.startswith("[") else f"{i}{bracket}"
    assert parse_interp_rating(
        f"Interpretability: {bad}\nConsistency: [[{c}]]"
    ) is None
