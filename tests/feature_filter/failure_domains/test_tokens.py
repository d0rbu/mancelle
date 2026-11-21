"""Token edge cases: empty tokens, single-char, extremely long, CJK/emoji, whitespace, marker chars."""
from __future__ import annotations

import pytest

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.interp_judge import (
    InterpJudgeConfig,
    build_judge_prompt,
    make_interp_judge,
)
from steering.eval.feature_filter.neuronpedia_activations import FeatureActivationPayload
from steering.eval.feature_filter.types import (
    ActivatingExample,
    BinSample,
    NoNonzeroActivationsError,
    PromptTooLongError,
)
from tests.feature_filter.fixtures.payload_builder import build_entry


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _mk(text: str, tokens: list[str] | None = None, peak: int = 0) -> BinSample:
    toks = tokens if tokens is not None else list(text)
    ex = ActivatingExample(
        tokens=toks,
        text=text,
        max_value=1.0,
        max_value_token_index=peak,
        peak_char_offset=peak,
    )
    return BinSample(
        bin_index=4,
        band_label="band 5/5 (peak firing)",
        is_absent=False,
        max_value=1.0,
        example=ex,
    )


def _payload_from(entries) -> FeatureActivationPayload:
    return FeatureActivationPayload(
        feature_id=1, description="d", entries=entries,
        freq_hist_bar_heights=[1.0], freq_hist_bar_values=[0.0],
        frac_nonzero=0.01, max_act_approx=10.0,
    )


def test_empty_token_list_entry_still_samples() -> None:
    """Sampler tolerates entries whose token list is empty when max_value > 0."""
    entries = [
        build_entry(max_value=0.5, bin_min=0.0, bin_max=2.0, tokens=[]),
        build_entry(max_value=3.0, bin_min=2.0, bin_max=4.0, tokens=["a"]),
        build_entry(max_value=5.0, bin_min=4.0, bin_max=6.0, tokens=["b"]),
        build_entry(max_value=7.0, bin_min=6.0, bin_max=8.0, tokens=["c"]),
        build_entry(max_value=9.0, bin_min=8.0, bin_max=10.0, tokens=["d"]),
    ]
    payload = _payload_from(entries)
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_Tok())
    bin0 = [s for s in samples if s.bin_index == 0]
    assert bin0[0].example.text == ""
    assert bin0[0].max_value == 0.5


def test_single_char_tokens_produce_marked_snippet() -> None:
    """Single-char tokens render exactly, and the peak is wrapped in <<...>>."""
    prompt = build_judge_prompt(
        "desc",
        [_mk("abc", tokens=["a", "b", "c"], peak=1)],
        window_chars=0,
    )
    assert "<<b>>" in prompt


def test_extremely_long_token_triggers_prompt_too_long() -> None:
    """A token longer than ``max_prompt_chars`` makes ``make_interp_judge`` raise PromptTooLongError."""
    cfg = InterpJudgeConfig(max_prompt_chars=500, example_window_chars=0, parse_max_retries=0)
    huge = "x" * 10_000
    samples = [_mk(huge, tokens=[huge])]

    class _Client:
        _cfg = type("_C", (), {"model": "mock"})()

        def complete(self, *a, **kw):  # pragma: no cover - should never be called
            raise AssertionError("judge must abort before calling the LLM")

    judge = make_interp_judge(_Client(), cfg)
    with pytest.raises(PromptTooLongError):
        judge("desc", samples)


def test_cjk_and_emoji_tokens_pass_through_snippet() -> None:
    """Non-ASCII tokens (CJK, emoji) survive prompt construction verbatim."""
    text = "漢字" + "😀"
    prompt = build_judge_prompt(
        "desc",
        [_mk(text, tokens=["漢", "字", "😀"], peak=2)],
        window_chars=0,
    )
    assert "漢" in prompt
    assert "😀" in prompt
    assert "<<😀>>" in prompt


def test_all_zero_tokens_triggers_no_nonzero() -> None:
    """Entries whose max_value == 0 and bin_min == -1 (sentinel) get filtered out,
    raising NoNonzeroActivationsError."""
    entries = [
        build_entry(max_value=0.0, bin_min=-1.0, bin_max=0.0, tokens=["a"]),
        build_entry(max_value=0.0, bin_min=-1.0, bin_max=0.0, tokens=["b"]),
    ]
    payload = _payload_from(entries)
    with pytest.raises(NoNonzeroActivationsError):
        sample_by_bins(payload, samples_per_bin=3, tokenizer=_Tok())
