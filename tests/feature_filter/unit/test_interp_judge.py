"""Interp judge prompt construction and retry policy."""
from dataclasses import dataclass as _dataclass
from unittest.mock import MagicMock

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.feature_filter.interp_judge import (
    InterpJudgeConfig,
    build_judge_prompt,
    format_band_blocks,
    make_interp_judge,
    marked_snippet,
)
from steering.eval.feature_filter.types import ActivatingExample, BinSample, PromptTooLongError


def _mk_sample(bin_index: int, mv: float, text: str, peak: int = 0,
               is_absent: bool = False) -> BinSample:
    ex = ActivatingExample(
        tokens=list(text), text=text, max_value=mv,
        max_value_token_index=peak, peak_char_offset=peak,
    )
    label_map = {
        4: "band 5/5 (peak firing)",
        3: "band 4/5 (very high firing)",
        2: "band 3/5 (high firing)",
        1: "band 2/5 (moderate firing)",
        0: "band 1/5 (absent — feature did not fire)" if is_absent
           else "band 1/5 (weak firing)",
    }
    return BinSample(
        bin_index=bin_index, band_label=label_map[bin_index],
        is_absent=is_absent, max_value=mv, example=ex,
    )


def test_format_band_blocks_single_band() -> None:
    samples = [_mk_sample(4, 9.5, "hello world", peak=6)]
    out = format_band_blocks(samples, window_chars=0)
    assert "band 5/5 (peak firing) — 1 examples:" in out
    assert "max_value=9.50" in out
    assert "<<w>>" in out


def test_format_band_blocks_groups_consecutive_band() -> None:
    samples = [
        _mk_sample(4, 9.5, "a", peak=0),
        _mk_sample(4, 8.0, "b", peak=0),
    ]
    out = format_band_blocks(samples, window_chars=0)
    assert "band 5/5 (peak firing) — 2 examples:" in out
    assert "[1]" in out and "[2]" in out


def test_format_band_blocks_absent_no_marker() -> None:
    s = _mk_sample(0, 0.0, "blank text window", peak=0, is_absent=True)
    out = format_band_blocks([s], window_chars=0)
    assert "Negative baselines (feature did not fire)" in out
    assert "<<" not in out
    assert "max_value=0 (absent)" in out


def test_format_band_blocks_separates_absent_into_negative_baselines() -> None:
    samples = [
        _mk_sample(4, 9.5, "peak example", peak=0),
        _mk_sample(0, 0.0, "blank text window", peak=0, is_absent=True),
    ]
    out = format_band_blocks(samples, window_chars=0)
    assert "Negative baselines (feature did not fire)" in out
    assert "band 1/5" not in out
    assert "band 5/5 (peak firing)" in out
    # absent block is last
    assert out.index("band 5/5") < out.index("Negative baselines")


def test_format_band_blocks_weak_firing_still_grouped_under_band_1_5() -> None:
    s = _mk_sample(0, 0.3, "weak text", peak=0, is_absent=False)
    out = format_band_blocks([s], window_chars=0)
    assert "band 1/5 (weak firing)" in out
    assert "Negative baselines" not in out


def test_format_band_blocks_rejects_unsorted_input() -> None:
    samples = [
        _mk_sample(2, 5.0, "a", peak=0),
        _mk_sample(4, 9.0, "b", peak=0),
    ]
    with pytest.raises(ValueError, match="peak-first"):
        format_band_blocks(samples, window_chars=0)


def test_build_prompt_substitutes_band_blocks() -> None:
    s = _mk_sample(4, 3.0, "fox jumps", peak=0)
    prompt = build_judge_prompt(description="animals", samples=[s], window_chars=0)
    assert "animals" in prompt
    assert "Respond in EXACTLY" in prompt
    assert "band 5/5" in prompt
    assert "{BAND_BLOCKS}" not in prompt
    assert "{DESCRIPTION}" not in prompt


def _stub_client(responses: list[str], model: str = "openai/gpt-5-nano") -> MagicMock:
    client = MagicMock()
    cfg_cls = _dataclass(type("C", (), {"model": model, "requests_per_minute": 60}))
    client._cfg = cfg_cls
    client.complete.side_effect = responses
    return client


def test_judge_happy_path_returns_rating() -> None:
    client = _stub_client(["Interpretability: [[3]]\nConsistency: [[5]]"])
    judge = make_interp_judge(client, InterpJudgeConfig())
    sample = _mk_sample(4, 1.0, "hi", 0)
    out = judge("the letter i", [sample])
    assert out is not None
    assert out.interpretability == 3


def test_judge_retry_exhausted_returns_none() -> None:
    client = _stub_client(["garbage"] * 4)
    judge = make_interp_judge(client, InterpJudgeConfig(parse_max_retries=3))
    sample = _mk_sample(4, 1.0, "hi", 0)
    out = judge("concept", [sample])
    assert out is None
    assert client.complete.call_count == 4


def test_judge_distinct_cache_keys_per_retry() -> None:
    client = _stub_client(["garbage", "Interpretability: [[2]]\nConsistency: [[4]]"])
    judge = make_interp_judge(client, InterpJudgeConfig(parse_max_retries=3))
    sample = _mk_sample(4, 1.0, "hi", 0)
    judge("concept", [sample])
    keys = [c.kwargs["cache_key"] for c in client.complete.call_args_list]
    assert keys[0] != keys[1]


def test_judge_transport_failure_returns_none() -> None:
    client = MagicMock()
    client._cfg = _dataclass(type("C", (), {"model": "openai/gpt-5-nano"}))
    client.complete.side_effect = AxBenchBaseError("upstream 500")
    judge = make_interp_judge(client, InterpJudgeConfig())
    sample = _mk_sample(4, 1.0, "hi", 0)
    assert judge("concept", [sample]) is None


def test_judge_parse_max_retries_negative_raises() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="parse_max_retries"):
        InterpJudgeConfig(parse_max_retries=-1)


def test_marked_snippet_long_text_windowed() -> None:
    # Force the long-text window branch (len(text) > 2*window_chars)
    long_text = "x" * 200
    ex = ActivatingExample(
        tokens=["y"], text=long_text, max_value=1.0,
        max_value_token_index=0, peak_char_offset=100,
    )
    out = marked_snippet(ex, window_chars=20)
    assert "<<" in out and ">>" in out


def test_marked_snippet_pos_out_of_range_returns_full() -> None:
    ex = ActivatingExample(
        tokens=["y"], text="hello", max_value=1.0,
        max_value_token_index=0, peak_char_offset=999,
    )
    out = marked_snippet(ex, window_chars=0)
    assert out == "hello"


def test_marked_snippet_token_does_not_match_at_pos() -> None:
    # Token text different from char at pos → fallback to single-char mark
    ex = ActivatingExample(
        tokens=["Z"], text="hello", max_value=1.0,
        max_value_token_index=0, peak_char_offset=2,
    )
    out = marked_snippet(ex, window_chars=0)
    assert "<<l>>" in out  # single char marked


def test_parse_interp_rating_out_of_range_returns_none() -> None:
    # Triggers else branch in _last_int_for (value > INTERP_JUDGE_MAX_SCORE)
    from steering.eval.feature_filter.interp_judge import parse_interp_rating
    assert parse_interp_rating("Interpretability: 99\nConsistency: 3") is None


def test_parse_interp_rating_with_decimal_returns_none() -> None:
    # Triggers the "." in raw branch (last = None, continue)
    from steering.eval.feature_filter.interp_judge import parse_interp_rating
    assert parse_interp_rating("Interpretability: 3.5\nConsistency: 3") is None


def test_judge_prompt_too_long() -> None:
    client = _stub_client([])
    cfg = InterpJudgeConfig(max_prompt_chars=10)
    judge = make_interp_judge(client, cfg)
    sample = _mk_sample(4, 1.0, "a long example that will blow the 10-char cap", 0)
    with pytest.raises(PromptTooLongError):
        judge("concept", [sample])


def test_interp_judge_config_defaults_raised_for_bands() -> None:
    from steering.eval.feature_filter.interp_judge import InterpJudgeConfig
    cfg = InterpJudgeConfig()
    assert cfg.max_examples_in_prompt == 25
    assert cfg.max_prompt_chars == 32_000
