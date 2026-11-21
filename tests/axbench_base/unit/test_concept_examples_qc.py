"""Unit tests for ``concept_examples_qc`` (review-fix F + G)."""

from __future__ import annotations

import dataclasses
import json
import random
import threading
import time
from dataclasses import asdict
from typing import Any

import pytest

from steering.eval.axbench_base import concept_examples_qc
from steering.eval.axbench_base.concept_examples_qc import (
    ConceptExamplesQC,
    QCFeatureScore,
    QCReport,
    _fetch_activating_examples_raw,
)
from steering.eval.axbench_base.concept_examples_qc import (
    make_llm_concept_examples_judge as _make_llm_concept_examples_judge,
)
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.synth.response_parser import (
    get_extraction_tokenizer,
)
from steering.eval.axbench_base.types import AxBenchBaseError, QcOutcome
from steering.eval.feature_filter.types import ActivatingExample
from tests.axbench_base._loguru_capture import capture_loguru_info
from tests.runtime_attribute_helpers import set_runtime_attribute

_TEST_JUDGE_MODEL = "test/concept-examples-judge"


def make_llm_concept_examples_judge(
    llm_client: LLMJudgeClient,
    *args: Any,
    model: str = _TEST_JUDGE_MODEL,
    **kwargs: Any,
):
    """Test adapter that supplies the production-required model identity."""
    return _make_llm_concept_examples_judge(
        llm_client,
        *args,
        model=model,
        **kwargs,
    )


def _shuffled_candidates(
    candidates: list[tuple[int, str]],
    seed: int,
) -> list[tuple[int, str]]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled

# ---------------------------------------------------------------------------
# Stub tokenizer that mirrors SentencePiece / byte-fallback semantics
# ---------------------------------------------------------------------------


class _SentencePieceStubTokenizer:
    """Stub that emulates SentencePiece whitespace marker semantics.

    Tokens starting with ``▁`` (U+2581) begin a new word (a space is
    inserted before the token's remaining characters). Tokens without
    ``▁`` are subword continuations (joined directly without a space).
    Byte-fallback tokens ``<0xNN>`` are decoded to the corresponding
    byte via ``chr(int(..., 16))``.

    Unlike a previous iteration of this stub, the leading space is NOT
    stripped — this matches the real Gemma (and LLaMA-family) tokenizer
    behavior, where ``convert_tokens_to_string(['▁Hello'])`` returns
    ``' Hello'`` with a leading space.
    """

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        parts: list[str] = []
        for tok in tokens:
            if tok.startswith("▁"):
                parts.append(" " + tok[1:])
            elif tok.startswith("<0x") and tok.endswith(">"):
                hex_val = tok[3:-1]
                try:
                    parts.append(chr(int(hex_val, 16)))
                except ValueError:
                    parts.append(tok)
            else:
                parts.append(tok)
        return "".join(parts)


def _stub_tokenizer() -> _SentencePieceStubTokenizer:
    return _SentencePieceStubTokenizer()


# ---------------------------------------------------------------------------
# ActivatingExample.example_text
# ---------------------------------------------------------------------------


def _make_example(
    *,
    text: str,
    peak_char_offset: int = 0,
    tokens: list[str] | None = None,
    max_value: float = 1.0,
    max_value_token_index: int = 0,
) -> ActivatingExample:
    return ActivatingExample(
        tokens=tokens if tokens is not None else list(text),
        text=text,
        max_value=max_value,
        max_value_token_index=max_value_token_index,
        peak_char_offset=peak_char_offset,
    )


def test_example_text_sentencepiece_space_marker():
    """SentencePiece-marked tokens give spaces in the text, not U+2581."""
    tok = _stub_tokenizer()
    tokens = ["▁Hello", "▁world"]
    text = tok.convert_tokens_to_string(tokens)
    assert text == " Hello world"
    ex = _make_example(text=text, tokens=tokens, max_value_token_index=0, peak_char_offset=0)
    assert "▁" not in ex.example_text()
    assert ex.example_text() == " Hello world"


def test_example_text_subword_continuation():
    """Subword continuations are concatenated directly (no space)."""
    tok = _stub_tokenizer()
    tokens = ["▁run", "ning", "▁fast"]
    text = tok.convert_tokens_to_string(tokens)
    # First token keeps leading space after ▁, next is continuation, next is new word
    assert text == " running fast"
    # peak_char_offset for max_value_token_index=1 is the length of
    # convert_tokens_to_string(['▁run']) → ' run' → 4.
    peak = len(tok.convert_tokens_to_string(tokens[:1]))
    assert peak == 4
    ex = _make_example(
        text=text,
        tokens=tokens,
        max_value_token_index=1,
        peak_char_offset=peak,
    )
    # The full text is short (13 chars) → no windowing.
    assert ex.example_text(window_chars=120) == " running fast"


def test_example_text_byte_fallback_newline():
    """Byte-fallback tokens like <0x0A> are decoded to the actual byte."""
    tok = _stub_tokenizer()
    tokens = ["hello", "<0x0A>", "world"]
    text = tok.convert_tokens_to_string(tokens)
    assert text == "hello\nworld"
    ex = _make_example(text=text, tokens=tokens)
    assert "\n" in ex.example_text()


def test_example_text_byte_fallback_invalid_hex_kept_literal():
    """Malformed <0xGG>-style tokens fall through to literal preservation."""
    tok = _stub_tokenizer()
    tokens = ["x", "<0xGG>", "y"]
    text = tok.convert_tokens_to_string(tokens)
    assert text == "x<0xGG>y"


def test_example_text_windowing_around_peak():
    """Window of +/- window_chars around peak_char_offset, with both ellipses."""
    text = "a" * 200 + "PEAK" + "a" * 200
    ex = _make_example(text=text, peak_char_offset=200)
    result = ex.example_text(window_chars=30)
    assert result.startswith("...")
    assert result.endswith("...")
    # The window is [170, 230] around peak=200.
    expected_core = text[170:230]
    assert expected_core in result
    assert result == "..." + expected_core + "..."


def test_example_text_no_windowing_when_short():
    """When text is shorter than 2*window_chars, return it unchanged."""
    text = "short enough"
    ex = _make_example(text=text, peak_char_offset=5)
    assert ex.example_text(window_chars=120) == text


def test_example_text_window_clips_left_only():
    """Peak near end of text → clip at end, trailing suffix is empty."""
    text = "a" * 100 + "PEAK"
    ex = _make_example(text=text, peak_char_offset=102)
    result = ex.example_text(window_chars=30)
    # start = max(0, 102-30) = 72 → leading "..."
    # end = min(104, 102+30) = 104 → no trailing "..."
    assert result.startswith("...")
    assert not result.endswith("...")
    assert result == "..." + text[72:104]


def test_example_text_window_clips_right_only():
    """Peak near start of text → clip at start, leading prefix is empty."""
    text = "PEAK" + "a" * 200
    ex = _make_example(text=text, peak_char_offset=2)
    result = ex.example_text(window_chars=30)
    # start = max(0, 2-30) = 0 → no leading "..."
    # end = min(204, 2+30) = 32 → trailing "..."
    assert not result.startswith("...")
    assert result.endswith("...")
    assert result == text[0:32] + "..."


def test_example_text_window_zero_returns_full_text():
    """window_chars=0 short-circuits to the full text regardless of length."""
    text = "a" * 1000
    ex = _make_example(text=text, peak_char_offset=500)
    assert ex.example_text(window_chars=0) == text


def test_example_text_window_negative_returns_full_text():
    """Negative window_chars is treated the same as 0 (no windowing)."""
    text = "a" * 1000
    ex = _make_example(text=text, peak_char_offset=500)
    assert ex.example_text(window_chars=-5) == text


def test_example_text_gemma_actual_behavior_via_stub():
    """The motivating Gemma example must NOT contain U+2581."""
    tok = _stub_tokenizer()
    tokens = ["1", "7", "▁to", "▁I", "-", "7", "9", "5", "▁from", "▁Gold"]
    text = tok.convert_tokens_to_string(tokens)
    assert text == "17 to I-795 from Gold"
    assert "▁" not in text
    ex = _make_example(text=text, tokens=tokens)
    assert "▁" not in ex.example_text()


# ---------------------------------------------------------------------------
# _fetch_activating_examples_raw
# ---------------------------------------------------------------------------


class _FakeFeature:
    """Stand-in for ``neuronpedia.SAEFeature`` returned values."""

    def __init__(self, json_data: Any) -> None:
        self.jsonData = json_data


def _install_fake_sae_feature(monkeypatch, feature_map):
    """Replace ``SAEFeature.get`` with a classmethod returning feature_map[(model,source,fid)]."""

    def _fake_get(cls, model_id, sae_source, feature_id_str):
        key = (model_id, sae_source, feature_id_str)
        if key not in feature_map:
            raise KeyError(key)
        return feature_map[key]

    monkeypatch.setattr(
        concept_examples_qc.SAEFeature,
        "get",
        classmethod(_fake_get),
    )


def test_fetch_activating_examples_raw_sorts_by_max_value(monkeypatch):
    """Unsorted activations should be returned in descending max_value order."""
    payload = {
        "activations": [
            {"tokens": ["a", "b"], "maxValue": 0.5, "maxValueTokenIndex": 0},
            {"tokens": ["c", "d"], "maxValue": 2.0, "maxValueTokenIndex": 1},
            {"tokens": ["e", "f"], "maxValue": 1.0, "maxValueTokenIndex": 0},
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("gemma-2-2b", "8-gemmascope-res-16k", "42"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "gemma-2-2b", "8-gemmascope-res-16k", 42, n_examples=2,
        tokenizer=_stub_tokenizer(),
    )
    assert len(result) == 2
    assert [s.max_value for s in result] == [2.0, 1.0]
    assert result[0].tokens == ["c", "d"]
    assert result[0].text == "cd"
    assert result[0].max_value_token_index == 1


def test_fetch_activating_examples_raw_handles_jsondata_string(monkeypatch):
    """When jsonData is a JSON string, it should be json.loads'd."""
    payload = {
        "activations": [
            {"tokens": ["▁hi"], "maxValue": 1.5, "maxValueTokenIndex": 0},
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("model_x", "src_x", "7"): _FakeFeature(json.dumps(payload))},
    )
    result = _fetch_activating_examples_raw(
        "model_x", "src_x", 7, n_examples=10, tokenizer=_stub_tokenizer(),
    )
    assert len(result) == 1
    assert result[0].tokens == ["▁hi"]
    # Real Gemma keeps the leading space from the ▁ marker; so does our stub.
    assert result[0].text == " hi"
    assert result[0].max_value == 1.5


def test_fetch_activating_examples_raw_handles_jsondata_dict(monkeypatch):
    """When jsonData is already a dict, it should be used directly."""
    payload = {
        "activations": [
            {"tokens": ["x"], "maxValue": 0.1, "maxValueTokenIndex": 0},
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "1"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "m", "s", 1, n_examples=1, tokenizer=_stub_tokenizer(),
    )
    assert len(result) == 1
    assert result[0].tokens == ["x"]
    assert result[0].text == "x"


def test_fetch_activating_examples_raw_rejects_non_list_activations(monkeypatch):
    """activations field that isn't a list raises AxBenchBaseError."""
    payload = {"activations": {"not": "a list"}}
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "5"): _FakeFeature(payload)},
    )
    with pytest.raises(AxBenchBaseError, match="activations field is not a list"):
        _fetch_activating_examples_raw(
            "m", "s", 5, n_examples=1, tokenizer=_stub_tokenizer(),
        )


def test_fetch_activating_examples_raw_logs_and_skips_empty_token_entries(monkeypatch):
    """A malformed Neuronpedia entry with empty tokens is skipped (external data —
    don't abort the run) but logged at ERROR so it's visible, not silently dropped."""
    payload = {
        "activations": [
            {"tokens": [], "maxValue": 99.0, "maxValueTokenIndex": 0},
            {"tokens": ["real"], "maxValue": 0.5, "maxValueTokenIndex": 0},
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "2"): _FakeFeature(payload)},
    )
    with capture_loguru_info() as captured:
        result = _fetch_activating_examples_raw(
            "m", "s", 2, n_examples=5, tokenizer=_stub_tokenizer(),
        )
    assert len(result) == 1
    assert result[0].tokens == ["real"]
    assert any("activation entry with no tokens" in line for line in captured), (
        f"malformed empty-token entry must be logged at ERROR: {captured!r}"
    )


def test_fetch_activating_examples_raw_missing_activations_field(monkeypatch):
    """When ``activations`` key is absent, returns an empty list."""
    payload: dict = {}
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "3"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "m", "s", 3, n_examples=5, tokenizer=_stub_tokenizer(),
    )
    assert result == []


def test_fetch_activating_examples_raw_logs_and_skips_non_dict_entries(monkeypatch):
    """A non-dict (malformed) Neuronpedia entry is skipped (external data) but
    logged at ERROR so it's visible, not silently dropped."""
    payload = {
        "activations": [
            "not a dict",
            {"tokens": ["ok"], "maxValue": 1.0, "maxValueTokenIndex": 0},
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "4"): _FakeFeature(payload)},
    )
    with capture_loguru_info() as captured:
        result = _fetch_activating_examples_raw(
            "m", "s", 4, n_examples=5, tokenizer=_stub_tokenizer(),
        )
    assert len(result) == 1
    assert result[0].tokens == ["ok"]
    assert any("non-dict activation entry" in line for line in captured), (
        f"malformed non-dict entry must be logged at ERROR: {captured!r}"
    )


def test_fetch_activating_examples_raw_clamps_out_of_range_max_idx(monkeypatch):
    """``maxValueTokenIndex`` values outside ``[0, len-1]`` must be clamped."""
    payload = {
        "activations": [
            {
                "tokens": ["a", "b", "c"],
                "maxValue": 1.0,
                "maxValueTokenIndex": -5,
            },
            {
                "tokens": ["d", "e", "f"],
                "maxValue": 0.5,
                "maxValueTokenIndex": 9999,
            },
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "9"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "m", "s", 9, n_examples=5, tokenizer=_stub_tokenizer(),
    )
    assert len(result) == 2
    # Sorted by max_value descending.
    assert result[0].max_value_token_index == 0  # clamped from -5
    assert result[0].peak_char_offset == 0
    assert result[1].max_value_token_index == 2  # clamped from 9999 to len-1
    # peak_char_offset for clamped idx=2 is length of convert_tokens_to_string(['d','e'])
    assert result[1].peak_char_offset == 2


def test_fetch_activating_examples_raw_computes_peak_char_offset(monkeypatch):
    """Peak char offset reflects the decoded text up to the max token index."""
    payload = {
        "activations": [
            {
                "tokens": ["▁run", "ning", "▁fast"],
                "maxValue": 1.0,
                "maxValueTokenIndex": 2,
            },
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("m", "s", "11"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "m", "s", 11, n_examples=1, tokenizer=_stub_tokenizer(),
    )
    # Full text = ' running fast' (13 chars)
    # tokens[:2] → ' running' (8 chars)
    # So peak_char_offset = 8
    assert result[0].text == " running fast"
    assert result[0].peak_char_offset == 8


# ---------------------------------------------------------------------------
# ConceptExamplesQC.filter
# ---------------------------------------------------------------------------


def _stub_fetch(mapping):
    """Return a FetchFn that returns ``mapping[fid]`` (or raises if it's an Exception).

    Matches the new 5-arg FetchFn signature (model_id, sae_source, fid,
    n_examples, tokenizer).
    """

    def _fn(model_id, sae_source, fid, n_examples, tokenizer):
        value = mapping[fid]
        if isinstance(value, Exception):
            raise value
        return value

    return _fn


def _make_ex(
    max_value: float = 1.0,
    tokens: list[str] | None = None,
    text: str | None = None,
) -> ActivatingExample:
    tokens_actual = tokens or ["t"]
    text_actual = text if text is not None else "".join(tokens_actual)
    return ActivatingExample(
        tokens=tokens_actual,
        text=text_actual,
        max_value=max_value,
        max_value_token_index=0,
        peak_char_offset=0,
    )


def _make_qc(judge, **kwargs) -> ConceptExamplesQC:
    defaults: dict[str, Any] = {
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "n_examples": 3,
        "judge": judge,
        "tokenizer": _stub_tokenizer(),
        "min_score": 1,
    }
    defaults.update(kwargs)
    return ConceptExamplesQC(**defaults)


def test_concept_examples_qc_keeps_high_scores():
    """Concepts with judge score >= min_score are kept, others dropped."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: [_make_ex(2.0)],
    }

    def _judge(description, examples):
        # fid 1 → score 2, fid 2 → score 0
        return 2 if examples[0].max_value == 1.0 else 0

    qc = _make_qc(
        judge=_judge,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    result, report = qc.filter([(1, "desc1"), (2, "desc2")])
    assert result == [(1, "desc1")]
    assert report.n_total == 2
    assert report.n_accepted == 1
    assert report.n_rejected == 1


def test_concept_examples_qc_fetch_failure_skips_concept():
    """Fetch errors skip the concept and are not fatal."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: RuntimeError("network down"),
        3: [_make_ex(1.0)],
    }
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    result, report = qc.filter([(1, "d1"), (2, "d2"), (3, "d3")])
    assert [fid for fid, _ in result] == [1, 3]
    assert report.n_fetch_errors == 1
    assert report.n_accepted == 2


def test_concept_examples_qc_empty_examples_skips_concept():
    """No examples returned → concept is dropped."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: [],
    }
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    result, report = qc.filter([(1, "d1"), (2, "d2")])
    assert [fid for fid, _ in result] == [1]
    assert report.n_empty_examples == 1


def test_concept_examples_qc_judge_raises_treated_as_zero():
    """Judge callable raising → treated as score 0 and dropped (if min_score > 0)."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: [_make_ex(1.0)],
    }

    def _judge(description, examples):
        if description == "fail":
            raise RuntimeError("boom")
        return 2

    qc = _make_qc(
        judge=_judge,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    result, report = qc.filter([(1, "ok"), (2, "fail")])
    assert result == [(1, "ok")]
    assert report.n_judge_errors == 1


def test_concept_examples_qc_uses_default_fetch_fn_when_none():
    """Constructing without fetch_fn falls back to the module-level raw fetcher."""
    qc = _make_qc(
        judge=lambda d, e: 0,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=None,
    )
    # We don't call filter; we just verify the default was wired in.
    assert qc._fetch_fn is _fetch_activating_examples_raw


# ---------------------------------------------------------------------------
# QCReport structure and stats
# ---------------------------------------------------------------------------


def test_qc_report_counts_acceptances():
    """Three-all-accepted: per-feature entries all have outcome=accepted."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in (1, 2, 3)}
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    survivors, report = qc.filter([(1, "a"), (2, "b"), (3, "c")])
    assert len(survivors) == 3
    assert report.n_total == 3
    assert report.n_accepted == 3
    assert report.n_rejected == 0
    assert report.n_fetch_errors == 0
    assert report.n_judge_errors == 0
    assert report.n_empty_examples == 0
    assert all(entry.outcome == "accepted" for entry in report.per_feature)
    assert all(entry.score == 2 for entry in report.per_feature)
    assert all(entry.n_examples == 1 for entry in report.per_feature)
    assert report.score_histogram == {2: 3}


def test_qc_report_counts_rejections():
    """Mixed score → rejected entries populated with outcome=rejected."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in (1, 2, 3)}
    scores_by_fid = {1: 2, 2: 0, 3: 1}

    def _judge(description, examples):
        return scores_by_fid[int(description)]

    qc = _make_qc(
        judge=_judge,
        model_id="m",
        sae_source="s",
        n_examples=1,
        min_score=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    survivors, report = qc.filter([(1, "1"), (2, "2"), (3, "3")])
    assert sorted(fid for fid, _ in survivors) == [1, 3]
    assert report.n_accepted == 2
    assert report.n_rejected == 1
    rejected = [e for e in report.per_feature if e.outcome == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].feature_index == 2
    assert rejected[0].score == 0


def test_qc_report_tracks_fetch_errors():
    """Fetch errors appear in both the summary count and per_feature."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: RuntimeError("neuronpedia 404"),
    }
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    _, report = qc.filter([(1, "d1"), (2, "d2")])
    assert report.n_fetch_errors == 1
    entry = next(e for e in report.per_feature if e.feature_index == 2)
    assert entry.outcome == "fetch_error"
    assert entry.score is None
    assert entry.n_examples == 0
    assert entry.error_message is not None
    assert "neuronpedia 404" in entry.error_message


def test_qc_report_tracks_empty_examples():
    """Empty-example concepts appear in n_empty_examples and per_feature."""
    examples_by_fid = {
        1: [],
        2: [_make_ex(1.0)],
    }
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    _, report = qc.filter([(1, "d1"), (2, "d2")])
    assert report.n_empty_examples == 1
    entry = next(e for e in report.per_feature if e.feature_index == 1)
    assert entry.outcome == "empty_examples"
    assert entry.score is None
    assert entry.n_examples == 0
    assert entry.error_message is None


def test_qc_report_tracks_judge_errors():
    """Judge-raising concepts appear in n_judge_errors and are dropped."""
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: [_make_ex(1.0)],
    }

    def _judge(description, examples):
        if description == "bad":
            raise RuntimeError("judge blew up")
        return 2

    qc = _make_qc(
        judge=_judge,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    survivors, report = qc.filter([(1, "ok"), (2, "bad")])
    assert [fid for fid, _ in survivors] == [1]
    assert report.n_judge_errors == 1
    entry = next(e for e in report.per_feature if e.feature_index == 2)
    assert entry.outcome == "judge_error"
    assert entry.score is None
    assert entry.n_examples == 1
    assert entry.error_message is not None
    assert "judge blew up" in entry.error_message
    # The histogram must NOT include the judge-error concept's fake 0 score.
    # Only the successful judge call (score=2 for fid=1) should appear.
    assert report.score_histogram == {2: 1}


def test_qc_report_score_histogram():
    """Score histogram counts real judge scores only, not judge errors."""
    # 5 real scores + 1 judge error → histogram should have 5 entries,
    # not 6.
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(1, 7)}
    real_scores = {1: 0, 2: 1, 3: 2, 4: 2, 5: 1}  # fid 6 raises

    def _judge(description, examples):
        fid = int(description)
        if fid == 6:
            raise RuntimeError("judge down")
        return real_scores[fid]

    qc = _make_qc(
        judge=_judge,
        model_id="m",
        sae_source="s",
        n_examples=1,
        min_score=1,
        fetch_fn=_stub_fetch(examples_by_fid),
    )
    _, report = qc.filter([(i, str(i)) for i in range(1, 7)])
    # Histogram: 0→1, 1→2, 2→2. Judge error (fid 6) is NOT in the histogram.
    assert report.score_histogram == {0: 1, 1: 2, 2: 2}
    assert report.n_judge_errors == 1
    # Total score_histogram counts should equal (n_total - n_fetch_errors -
    # n_judge_errors - n_empty_examples), i.e. the number of actual judge
    # calls.
    total_in_histogram = sum(report.score_histogram.values())
    assert total_in_histogram == (
        report.n_total
        - report.n_fetch_errors
        - report.n_judge_errors
        - report.n_empty_examples
    )


def test_qc_report_returned_from_filter_is_tuple():
    """filter() must return a 2-tuple of (list, QCReport)."""
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch({1: [_make_ex(1.0)]}),
    )
    out = qc.filter([(1, "d")])
    assert isinstance(out, tuple)
    assert len(out) == 2
    survivors, report = out
    assert isinstance(survivors, list)
    assert isinstance(report, QCReport)


def test_qc_report_is_serializable():
    """QCReport round-trips via asdict + json.dumps + json.loads."""
    qc = _make_qc(
        judge=lambda d, e: 2,
        model_id="m",
        sae_source="s",
        n_examples=1,
        fetch_fn=_stub_fetch({1: [_make_ex(1.0)], 2: [_make_ex(2.0)]}),
    )
    _, report = qc.filter([(1, "a"), (2, "b")])
    payload = asdict(report)
    # dict keys in score_histogram survive json round-trip (as strings,
    # which is fine — the consumer coerces).
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["n_total"] == 2
    assert loaded["n_accepted"] == 2
    assert len(loaded["per_feature"]) == 2
    assert loaded["per_feature"][0]["outcome"] == "accepted"


def test_qc_feature_score_is_frozen_dataclass():
    """QCFeatureScore rejects attribute assignment (frozen)."""
    entry = QCFeatureScore(
        feature_index=1,
        description="d",
        outcome=QcOutcome.ACCEPTED,
        score=2,
        n_examples=1,
        error_message=None,
    )
    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. Frozen
    # dataclass mutation raises ``FrozenInstanceError``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(entry, "outcome", "rejected")


def test_qc_report_is_frozen_dataclass():
    """QCReport rejects attribute assignment (frozen)."""
    report = QCReport(
        n_total=0,
        n_accepted=0,
        n_rejected=0,
        n_fetch_errors=0,
        n_judge_errors=0,
        n_empty_examples=0,
        score_histogram={},
        per_feature=[],
        judge_min_score=1,
    )
    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. Frozen
    # dataclass mutation raises ``FrozenInstanceError``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(report, "n_total", 1)


# ---------------------------------------------------------------------------
# make_llm_concept_examples_judge
# ---------------------------------------------------------------------------


class _StubLLMClient(LLMJudgeClient):
    def __init__(self, responses, raise_for=None):
        self.responses = list(responses)
        self.raise_for = raise_for
        self.calls: list[list[dict[str, str]]] = []
        self.cache_keys: list[str | None] = []

    def complete(self, messages, *, cache_key=None):
        self.calls.append(messages)
        self.cache_keys.append(cache_key)
        if self.raise_for is not None:
            raise self.raise_for
        return self.responses.pop(0)


def test_make_llm_concept_examples_judge_parses_rating():
    """Well-formed "Rating: [[N]]" responses are parsed to an int score."""
    client = _StubLLMClient(
        responses=["The description matches the examples well. Rating: [[2]]"],
    )
    judge = make_llm_concept_examples_judge(client)
    examples = [_make_ex(1.0, tokens=["hello", " world"], text="hello world")]
    score = judge("greeting phrases", examples)
    assert score == 2
    # Prompt was sent as a single user message containing the template.
    assert len(client.calls) == 1
    assert client.calls[0][0]["role"] == "user"
    assert "hello world" in client.calls[0][0]["content"]
    assert "greeting phrases" in client.calls[0][0]["content"]


def test_make_llm_concept_examples_judge_uses_neutral_profiler_names(tmp_path):
    client = _StubLLMClient(responses=["Rating: [[2]]"])
    judge = make_llm_concept_examples_judge(client)

    with AxBenchProfiler(tmp_path / "profile"):
        score = judge("desc", [_make_ex(1.0)])

    assert score == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "profile" / "events.jsonl").read_text().splitlines()
    ]
    names = [event.get("name", "") for event in events]
    assert "concept_examples_qc.llm_judge" in names
    assert "concept_examples_qc.judge_success" in names
    assert not any(name.startswith("stage0a.") for name in names)


def test_make_llm_concept_examples_judge_parse_failure_after_retries_returns_none():
    """After exhausting parse retries, the judge returns None (NOT score 0).

    Conflating "parse failed" with "judge said 0" is the bug that
    silently rejected 16k features in a 9.6-hour production run.
    The contract is explicit: None means "could not parse", and the
    caller (ConceptExamplesQC.filter) must treat it as a judge_error
    rather than a rubric-zero rating."""
    # All 4 attempts (1 initial + 3 retries) return unparseable content.
    client = _StubLLMClient(
        responses=[
            "Missing the marker 1",
            "Missing the marker 2",
            "Missing the marker 3",
            "Missing the marker 4",
        ],
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score is None
    # All attempts fired (parse_max_retries + 1 = 4).
    assert len(client.calls) == 4


def test_make_llm_concept_examples_judge_parse_failure_then_success_returns_score():
    """If the first call can't be parsed but a retry succeeds, the
    retry's score is returned -- this is what fixes the production
    bug (judge drops double brackets ~17% of the time; one retry
    usually catches the correct format)."""
    client = _StubLLMClient(
        responses=[
            "Ambiguous prose with no rating",
            "Rating: [[2]]",  # retry succeeds
        ],
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score == 2
    # Fired exactly 2 calls: 1 initial + 1 retry.
    assert len(client.calls) == 2


def test_make_llm_concept_examples_judge_bare_rating_parses():
    """The whole point of this PR: bare `Rating: N` (5/30 empirical
    frequency) must parse without needing a retry."""
    client = _StubLLMClient(responses=["Rating: 2"])
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score == 2
    assert len(client.calls) == 1  # no retry needed


def test_make_llm_concept_examples_judge_retry_uses_distinct_cache_keys():
    """Per-attempt cache keys ensure retries actually hit the API,
    even when the LLM client caches the first (bad) response.

    Without per-attempt keys, a cached bad response would be returned
    on every retry and the loop would uselessly re-parse the same
    content N times, giving up after retries and returning None."""
    client = _StubLLMClient(
        responses=["bad1", "bad2", "Rating: [[1]]"],
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    judge("desc", [_make_ex(1.0)])
    assert len(client.cache_keys) == 3
    # All three keys are distinct so the LLM cache can't short-circuit
    # a retry into re-reading a bad response.
    assert len(set(client.cache_keys)) == 3


def test_make_llm_concept_examples_judge_llm_transport_error_returns_none():
    """A transport-level failure (AxBenchBaseError from the LLM client)
    propagates as None to the caller. The upstream client already
    handled 429/5xx retries; further retries here would just hammer
    a broken service."""
    client = _StubLLMClient(
        responses=[],
        raise_for=AxBenchBaseError("openrouter down"),
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score is None


def test_make_llm_concept_examples_judge_negative_retries_rejects_at_construction():
    """A negative ``parse_max_retries`` (from a Hydra config typo, say)
    would silently turn into ``range(0)`` -> zero LLM calls -> 100%
    reject rate, which looks identical to the original production bug
    this PR exists to fix. The constructor must refuse.

    Catching it at construction time gives an operator a clear
    ``ValueError`` with the bad value in the message, rather than a
    silent 3-hour run that marks every concept as ``judge_error``."""
    client = _StubLLMClient(responses=[])
    with pytest.raises(ValueError, match="parse_max_retries"):
        make_llm_concept_examples_judge(client, parse_max_retries=-1)
    with pytest.raises(ValueError, match="parse_max_retries"):
        make_llm_concept_examples_judge(client, parse_max_retries=-5)


def test_make_llm_concept_examples_judge_zero_retries_works():
    """``parse_max_retries=0`` means exactly ONE attempt. On parse
    failure, return None immediately (no retry). On success, return
    the score. Verifies the boundary between the negative-rejection
    path and the normal retry path."""
    # Success on first attempt.
    client1 = _StubLLMClient(responses=["Rating: [[2]]"])
    judge1 = make_llm_concept_examples_judge(client1, parse_max_retries=0)
    assert judge1("desc", [_make_ex(1.0)]) == 2
    assert len(client1.calls) == 1

    # Parse failure on first attempt -> None (no retry).
    client2 = _StubLLMClient(responses=["nonsense"])
    judge2 = make_llm_concept_examples_judge(client2, parse_max_retries=0)
    assert judge2("desc", [_make_ex(1.0)]) is None
    assert len(client2.calls) == 1


def test_make_llm_concept_examples_judge_cache_key_namespaces_on_model():
    """Two judges constructed from clients with DIFFERENT underlying
    LLM models must use DIFFERENT cache keys so that an A/B comparison
    between e.g. gpt-5-nano and gpt-4o-mini on the same ``output_root``
    doesn't silently serve the wrong model's cached response."""
    client_a = _StubLLMClient(responses=["Rating: [[2]]"])
    client_b = _StubLLMClient(responses=["Rating: [[2]]"])
    judge_a = make_llm_concept_examples_judge(
        client_a,
        model="openai/gpt-5-nano",
    )
    judge_b = make_llm_concept_examples_judge(
        client_b,
        model="openai/gpt-4o-mini",
    )
    judge_a("same-description", [_make_ex(1.0, text="same-text")])
    judge_b("same-description", [_make_ex(1.0, text="same-text")])
    # Same description + same prompt + DIFFERENT models -> DIFFERENT
    # cache keys (so future runs can't collide across configs).
    assert client_a.cache_keys[0] != client_b.cache_keys[0]

def test_make_llm_concept_examples_judge_truncates_examples_in_prompt():
    """``max_examples_in_prompt`` caps how many snippets are sent to the LLM."""
    client = _StubLLMClient(responses=["Rating: [[1]]"])
    judge = make_llm_concept_examples_judge(
        client, max_examples_in_prompt=2,
    )
    examples = [
        _make_ex(3.0, text="alpha"),
        _make_ex(2.0, text="beta"),
        _make_ex(1.0, text="gamma"),
    ]
    judge("desc", examples)
    content = client.calls[0][0]["content"]
    assert "alpha" in content
    assert "beta" in content
    assert "gamma" not in content


def test_make_llm_concept_examples_judge_rejects_out_of_range_score():
    """An out-of-rubric score triggers retry. If every attempt is out
    of range, return None so the caller can mark the feature invalid."""
    client = _StubLLMClient(
        responses=["Rating: [[5]]"] * 4,  # 1 initial + 3 retries
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score is None


def test_make_llm_concept_examples_judge_rejects_huge_score():
    """Astronomically-large scores are out-of-range -> None."""
    client = _StubLLMClient(
        responses=["Rating: [[9999]]"] * 4,
    )
    judge = make_llm_concept_examples_judge(client, parse_max_retries=3)
    score = judge("desc", [_make_ex(1.0)])
    assert score is None


def test_make_llm_concept_examples_judge_accepts_boundary_scores():
    """Scores at the rubric boundaries (0, 1, 2) should be accepted."""
    for rubric_score in (0, 1, 2):
        client = _StubLLMClient(responses=[f"Rating: [[{rubric_score}]]"])
        judge = make_llm_concept_examples_judge(client)
        score = judge("desc", [_make_ex(1.0)])
        assert score == rubric_score


def test_make_llm_concept_examples_judge_forwards_cache_key():
    """The judge must build a deterministic cache_key per (description, prompt)."""
    client = _StubLLMClient(
        responses=["Rating: [[2]]", "Rating: [[1]]"],
    )
    judge = make_llm_concept_examples_judge(client)
    judge("description-A", [_make_ex(1.0, text="ex-a")])
    judge("description-B", [_make_ex(1.0, text="ex-b")])
    # Both calls received a non-None cache_key.
    assert len(client.cache_keys) == 2
    assert client.cache_keys[0] is not None
    assert client.cache_keys[1] is not None
    assert len(client.cache_keys[0]) == 64  # sha256 hex
    # Two different descriptions -> two different cache keys.
    assert client.cache_keys[0] != client.cache_keys[1]


def test_make_llm_concept_examples_judge_cache_key_deterministic():
    """Re-invoking the judge with identical inputs yields the same cache key."""
    client_a = _StubLLMClient(responses=["Rating: [[2]]"])
    client_b = _StubLLMClient(responses=["Rating: [[2]]"])
    judge_a = make_llm_concept_examples_judge(client_a)
    judge_b = make_llm_concept_examples_judge(client_b)
    example = _make_ex(1.0, text="fixed-text")
    judge_a("same-description", [example])
    judge_b("same-description", [example])
    assert client_a.cache_keys[0] == client_b.cache_keys[0]


# ---------------------------------------------------------------------------
# Real-tokenizer integration (slow — requires Gemma tokenizer download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_gemma_tokenizer_detokenizes_neuronpedia_tokens():
    """End-to-end: use the real Gemma tokenizer to detokenize a Neuronpedia-
    shaped token list. Verifies that our stub-based tests aren't hiding a
    real-world divergence.
    """
    tokenizer = get_extraction_tokenizer("google/gemma-2-2b")
    # Exact payload seen in Neuronpedia responses for feature 100.
    tokens = ["1", "7", "▁to", "▁I", "-", "7", "9", "5", "▁from", "▁Gold"]
    text = tokenizer.convert_tokens_to_string(tokens)
    # No SentencePiece markers leak through.
    assert "▁" not in text
    # Leading space is NOT stripped by Gemma — adjust any assumption.
    assert text == "17 to I-795 from Gold"


@pytest.mark.slow
def test_real_gemma_tokenizer_preserves_leading_space():
    """When the first token starts with ``▁``, the real tokenizer keeps the
    leading space. This is the behavior the stub was updated to match."""
    tokenizer = get_extraction_tokenizer("google/gemma-2-2b")
    assert tokenizer.convert_tokens_to_string(["▁Hello"]) == " Hello"
    assert (
        tokenizer.convert_tokens_to_string(["▁run", "ning", "▁fast"])
        == " running fast"
    )


@pytest.mark.slow
def test_fetch_activating_examples_raw_with_real_tokenizer(monkeypatch):
    """Full ``_fetch_activating_examples_raw`` path with the real Gemma tokenizer.

    Uses a fake Neuronpedia payload (so no network) but threads it through
    the real ``convert_tokens_to_string`` to verify the peak_char_offset
    arithmetic lands at the right place in real-tokenizer output.
    """
    tokenizer = get_extraction_tokenizer("google/gemma-2-2b")
    payload = {
        "activations": [
            {
                "tokens": ["▁The", "▁quick", "▁brown", "▁fox"],
                "maxValue": 1.5,
                "maxValueTokenIndex": 3,
            },
        ],
    }
    _install_fake_sae_feature(
        monkeypatch,
        {("gemma-2-2b", "8-gemmascope-res-16k", "1"): _FakeFeature(payload)},
    )
    result = _fetch_activating_examples_raw(
        "gemma-2-2b",
        "8-gemmascope-res-16k",
        1,
        n_examples=1,
        tokenizer=tokenizer,
    )
    assert len(result) == 1
    full = tokenizer.convert_tokens_to_string(
        ["▁The", "▁quick", "▁brown", "▁fox"]
    )
    prefix = tokenizer.convert_tokens_to_string(["▁The", "▁quick", "▁brown"])
    assert result[0].text == full
    assert result[0].peak_char_offset == len(prefix)
    # Slicing the text at peak_char_offset gives us "▁fox" reconstructed.
    assert result[0].text[result[0].peak_char_offset :] == full[len(prefix) :]


# ---------------------------------------------------------------------------
# Early termination via target_surviving
#
# These tests drive the fix for the Stage 0a bottleneck where ``filter``
# serially fetched + judged all 16k+ candidates even when the caller only
# needed ``max_concepts=5``. Expected production impact: ~70h → seconds.
# ---------------------------------------------------------------------------


def _counting_fetch(mapping: dict[int, Any]) -> tuple[Any, list[int]]:
    """Return ``(fetch_fn, fids_called)``.

    ``fids_called`` is appended to on every invocation (thread-safe via GIL
    on list.append; we don't need a lock for this test-only counter).
    """
    fids_called: list[int] = []
    lock = threading.Lock()

    def _fn(model_id, sae_source, fid, n_examples, tokenizer):
        with lock:
            fids_called.append(fid)
        value = mapping[fid]
        if isinstance(value, Exception):
            raise value
        return value

    return _fn, fids_called


def test_filter_backward_compat_no_target_processes_all():
    """``target_surviving=None`` → all candidates are processed (pre-fix contract)."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(10)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, report = qc.filter(candidates)
    assert len(survivors) == 10
    assert len(called) == 10  # every candidate's fetch_fn was invoked
    assert report.n_accepted == 10


def test_filter_target_surviving_stops_early_when_target_reached():
    """``target_surviving=3`` → stop after enough survivors are collected."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(100)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=4,
    )
    candidates = [(i, f"d{i}") for i in range(100)]
    survivors, _report = qc.filter(candidates, target_surviving=3)
    # Got at least target survivors.
    assert len(survivors) >= 3
    # Did not process all 100 — that would defeat the whole point.
    # With batch=4 and all-pass, at most 4 fetches are needed for 3 survivors.
    assert len(called) <= 4
    assert len(called) < 100


def test_filter_target_surviving_with_rejections_continues_until_target():
    """Rejected concepts don't count toward target; filter keeps going."""
    # Pattern: even fids pass (score 2), odd fids fail (score 0).
    # To get 3 survivors, we need to process at least 5 even candidates
    # (= 10 candidates in 0..9). With batch=10 and early-term after a batch,
    # one batch of 10 suffices to get 5 even → 5 survivors ≥ 3.
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(100)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2 if int(d[1:]) % 2 == 0 else 0,
        model_id="m", sae_source="s", n_examples=1,
        fetch_fn=fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(100)]
    survivors, report = qc.filter(candidates, target_surviving=3)
    assert len(survivors) >= 3
    # All accepted entries should be even (by construction).
    assert all(int(desc[1:]) % 2 == 0 for _fid, desc in survivors)
    # Rejections should count, but we stopped processing early.
    assert len(called) < 100


def test_filter_target_surviving_none_pass_processes_all():
    """Zero survivors case: no early-term; we process every candidate."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(20)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 0,  # always reject
        model_id="m", sae_source="s", n_examples=1,
        fetch_fn=fetch_fn, max_workers=5,
    )
    candidates = [(i, f"d{i}") for i in range(20)]
    survivors, report = qc.filter(candidates, target_surviving=3)
    assert survivors == []
    # No survivors means we exhausted the candidate list.
    assert len(called) == 20
    assert report.n_rejected == 20


def test_filter_target_surviving_zero_processes_nothing():
    """``target_surviving=0`` → trivially satisfied, no work done."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(10)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, _report = qc.filter(candidates, target_surviving=0)
    assert survivors == []
    assert called == []


def test_filter_target_surviving_negative_raises():
    """Negative ``target_surviving`` is always a programmer error — fail loudly."""
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_stub_fetch({1: [_make_ex(1.0)]}),
    )
    with pytest.raises(ValueError):
        qc.filter([(1, "d")], target_surviving=-1)


def test_filter_target_larger_than_candidates_processes_all():
    """``target_surviving > len(candidates)`` degrades to "process all"."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(5)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn,
    )
    candidates = [(i, f"d{i}") for i in range(5)]
    survivors, _report = qc.filter(candidates, target_surviving=100)
    assert len(survivors) == 5
    assert len(called) == 5


def test_filter_fetch_errors_do_not_count_toward_target():
    """Fetch errors are classified as errors, not survivors."""
    # fids 0-4 raise; fids 5-9 pass. Target=3 needs fids 5, 6, 7.
    examples_by_fid: dict[int, Any] = {
        i: (RuntimeError("net down") if i < 5 else [_make_ex(1.0)])
        for i in range(10)
    }
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, report = qc.filter(candidates, target_surviving=3)
    assert len(survivors) >= 3
    # All survivors must come from the passing half.
    assert all(fid >= 5 for fid, _d in survivors)
    # Errors + processed count must cover at least fids 0..7 (or more)
    # to yield 3 survivors. We tolerate batch-overshoot past 8.
    assert report.n_fetch_errors >= 3


def test_filter_judge_errors_do_not_count_toward_target():
    """Judge errors are classified as errors, not survivors."""
    # Description d0..d4 → judge raises; d5..d9 → judge returns 2.
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(10)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)

    def _judge(desc, examples):
        if int(desc[1:]) < 5:
            raise RuntimeError("judge barfed")
        return 2

    qc = _make_qc(
        judge=_judge, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, report = qc.filter(candidates, target_surviving=3)
    assert len(survivors) >= 3
    assert all(fid >= 5 for fid, _d in survivors)
    assert report.n_judge_errors >= 3


def test_filter_empty_examples_do_not_count_toward_target():
    """Concepts that return zero examples are classified as empty, not survivors."""
    examples_by_fid: dict[int, Any] = {
        i: ([] if i < 5 else [_make_ex(1.0)]) for i in range(10)
    }
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, report = qc.filter(candidates, target_surviving=3)
    assert len(survivors) >= 3
    assert all(fid >= 5 for fid, _d in survivors)
    assert report.n_empty_examples >= 3


# ---------------------------------------------------------------------------
# Determinism with caller-owned candidate ordering
# ---------------------------------------------------------------------------


def test_filter_pre_shuffled_order_is_deterministic():
    """The same caller-owned order yields the same survivors."""
    # All candidates pass the judge; target=5 forces early-term.
    # Without the shuffle, survivors would be fids 0..4 in both runs.
    # WITH the shuffle, survivors are a pseudo-random 5-subset — but
    # identical across calls for the same seed.
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(50)}

    def _run():
        fetch_fn, _called = _counting_fetch(examples_by_fid)
        qc = _make_qc(
            judge=lambda d, e: 2, model_id="m", sae_source="s",
            n_examples=1, fetch_fn=fetch_fn, max_workers=10,
        )
        candidates = [(i, f"d{i}") for i in range(50)]
        survivors, _r = qc.filter(
            _shuffled_candidates(candidates, 42),
            target_surviving=5,
        )
        return survivors

    a = _run()
    b = _run()
    assert a == b
    # And it's NOT the trivial input order (unless we got catastrophically
    # unlucky — probability ~1/50!).
    assert [fid for fid, _ in a] != list(range(5))


def test_filter_different_input_orders_produce_different_survivors():
    """Different seeds → (usually) different survivor subset.

    With 50 candidates all passing, picking 5 via shuffle, the odds of
    two different seeds picking identical subsets AND identical orders
    are astronomically low. We assert inequality.
    """
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(50)}

    def _run(seed):
        fetch_fn, _called = _counting_fetch(examples_by_fid)
        qc = _make_qc(
            judge=lambda d, e: 2, model_id="m", sae_source="s",
            n_examples=1, fetch_fn=fetch_fn, max_workers=10,
        )
        candidates = [(i, f"d{i}") for i in range(50)]
        survivors, _r = qc.filter(
            _shuffled_candidates(candidates, seed),
            target_surviving=5,
        )
        return survivors

    assert _run(1) != _run(2)


def test_filter_preserves_input_order():
    """Survivors appear in the caller-provided order."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(20)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=5,
    )
    candidates = [(i, f"d{i}") for i in range(20)]
    survivors, _report = qc.filter(candidates)
    assert [fid for fid, _ in survivors] == list(range(20))


def test_filter_pre_shuffled_input_does_not_drop_candidates():
    """A caller-provided permutation loses or duplicates no candidate."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(20)}
    fetch_fn, called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=5,
    )
    candidates = [(i, f"d{i}") for i in range(20)]
    survivors, _report = qc.filter(_shuffled_candidates(candidates, 7))
    # Everyone processed exactly once.
    assert sorted(called) == list(range(20))
    assert len(survivors) == 20
    # Survivor set == input set (ignoring order).
    assert {fid for fid, _ in survivors} == set(range(20))


# ---------------------------------------------------------------------------
# Parallelism via max_workers
# ---------------------------------------------------------------------------


def test_filter_fetches_in_parallel_via_max_workers():
    """With ``max_workers=4`` and a 4-thread barrier inside fetch_fn,
    all four fetches run simultaneously — the barrier only clears if true
    parallelism is happening. If the filter is serial, the barrier blocks
    forever (caught by a tight timeout)."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(4)}
    barrier = threading.Barrier(4, timeout=5.0)

    def _fetch_fn(model_id, sae_source, fid, n_examples, tokenizer):
        # All 4 threads must reach this barrier before any proceeds.
        barrier.wait()
        return examples_by_fid[fid]

    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_fetch_fn, max_workers=4,
    )
    candidates = [(i, f"d{i}") for i in range(4)]
    # If serial, barrier.wait() blocks the first fetch forever, eventually
    # raising BrokenBarrierError after the 5.0s timeout.
    survivors, _report = qc.filter(candidates)
    assert len(survivors) == 4


def test_filter_respects_max_workers_cap():
    """With ``max_workers=3`` and 10 candidates, at most 3 fetches in flight.

    We track the peak concurrent count via a lock-protected counter. A
    serial implementation would show peak=1; a parallel impl must never
    exceed 3.
    """
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(10)}
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def _fetch_fn(model_id, sae_source, fid, n_examples, tokenizer):
        with lock:
            state["in_flight"] += 1
            if state["in_flight"] > state["peak"]:
                state["peak"] = state["in_flight"]
        try:
            # Give sibling threads a chance to start.
            time.sleep(0.02)
            return examples_by_fid[fid]
        finally:
            with lock:
                state["in_flight"] -= 1

    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_fetch_fn, max_workers=3,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    qc.filter(candidates)
    # Peak should be > 1 (parallel) and <= 3 (cap honored).
    assert state["peak"] > 1
    assert state["peak"] <= 3


def test_filter_default_max_workers_is_reasonable():
    """Default ``max_workers`` matches the Neuronpedia concurrent default (10)."""
    # This is a contract test — if we ever change the default, this test
    # must be updated consciously (not silently).
    def _judge(description, examples):
        return 0

    qc = ConceptExamplesQC(
        model_id="m", sae_source="s", n_examples=1,
        judge=_judge, tokenizer=_stub_tokenizer(),
    )
    assert qc._max_workers == 10


# ---------------------------------------------------------------------------
# QCReport accuracy under early termination
# ---------------------------------------------------------------------------


def test_filter_report_has_n_processed_field():
    """``QCReport`` gains an ``n_processed`` field populated by ``filter``."""
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_stub_fetch({1: [_make_ex(1.0)]}),
    )
    _survivors, report = qc.filter([(1, "d")])
    # Field exists and is correct.
    assert hasattr(report, "n_processed")
    assert report.n_processed == 1


def test_filter_report_n_processed_equals_per_feature_length():
    """``n_processed`` is a shorthand for ``len(per_feature)``."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(5)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(5)]
    _survivors, report = qc.filter(candidates, target_surviving=2)
    assert report.n_processed == len(report.per_feature)


def test_filter_report_n_processed_less_than_n_total_on_early_term():
    """Under early termination, ``n_processed < n_total`` is expected."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(100)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=4,
    )
    candidates = [(i, f"d{i}") for i in range(100)]
    _survivors, report = qc.filter(candidates, target_surviving=3)
    assert report.n_total == 100
    assert report.n_processed < 100
    assert report.n_processed >= 3


def test_filter_report_n_rejected_excludes_unprocessed():
    """Under early termination, ``n_rejected`` reflects only actual rejections,
    not "remaining unprocessed candidates"."""
    # All 100 candidates would pass; target=3 makes us stop after ~1 batch.
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(100)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=5,
    )
    candidates = [(i, f"d{i}") for i in range(100)]
    _survivors, report = qc.filter(candidates, target_surviving=3)
    # Zero actual rejections; the 95+ unprocessed are NOT counted here.
    assert report.n_rejected == 0
    # The accounting invariant over processed entries:
    assert (
        report.n_accepted
        + report.n_rejected
        + report.n_fetch_errors
        + report.n_judge_errors
        + report.n_empty_examples
    ) == report.n_processed


def test_filter_report_per_feature_only_covers_processed():
    """``per_feature`` must not include entries for unprocessed candidates."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(50)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=5,
    )
    candidates = [(i, f"d{i}") for i in range(50)]
    _survivors, report = qc.filter(candidates, target_surviving=3)
    assert len(report.per_feature) < 50
    # The per_feature fids must be a subset of the input fids.
    feat_ids = {entry.feature_index for entry in report.per_feature}
    assert feat_ids.issubset(set(range(50)))


# ---------------------------------------------------------------------------
# Interaction: target_surviving + caller-owned ordering + errors
# ---------------------------------------------------------------------------


def test_filter_early_term_with_pre_shuffled_input_is_deterministic():
    """Pre-shuffled input plus early termination remains deterministic."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(50)}

    def _run():
        fetch_fn, _called = _counting_fetch(examples_by_fid)
        qc = _make_qc(
            judge=lambda d, e: 2, model_id="m", sae_source="s",
            n_examples=1, fetch_fn=fetch_fn, max_workers=7,
        )
        candidates = [(i, f"d{i}") for i in range(50)]
        survivors, _r = qc.filter(
            _shuffled_candidates(candidates, 123),
            target_surviving=5,
        )
        return survivors

    assert _run() == _run()


def test_filter_report_returned_even_on_early_term():
    """Stop-early must still return a populated ``QCReport`` tuple."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(20)}
    fetch_fn, _called = _counting_fetch(examples_by_fid)
    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=fetch_fn, max_workers=3,
    )
    candidates = [(i, f"d{i}") for i in range(20)]
    out = qc.filter(candidates, target_surviving=2)
    assert isinstance(out, tuple) and len(out) == 2
    survivors, report = out
    assert isinstance(report, QCReport)
    assert report.n_accepted >= 2
    assert report.n_processed == len(report.per_feature)


def test_filter_backward_compat_existing_callers_unchanged():
    """Regression guard: callers that don't pass new kwargs see no change.

    Uses the simplest ``_make_qc`` construction (no ``max_workers`` or
    ``target_surviving`` kwarg) — the shape of the
    survivor list and the QCReport aggregate counts must match the
    pre-fix contract exactly.
    """
    examples_by_fid = {
        1: [_make_ex(1.0)],
        2: [_make_ex(1.0)],
        3: [_make_ex(1.0)],
    }
    qc = _make_qc(
        judge=lambda d, e: 2 if int(d[1:]) != 2 else 0,
        model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_stub_fetch(examples_by_fid),
    )
    survivors, report = qc.filter([(1, "d1"), (2, "d2"), (3, "d3")])
    assert survivors == [(1, "d1"), (3, "d3")]
    assert report.n_total == 3
    assert report.n_accepted == 2
    assert report.n_rejected == 1


# ---------------------------------------------------------------------------
# Thread-safety: fetch_fn concurrent invocations don't corrupt per_feature
# ---------------------------------------------------------------------------


def test_filter_per_feature_order_stable_under_parallel_fetch():
    """Even when fetches complete out of order, ``per_feature`` entries
    within a processed batch are consistent with the submitted order
    (== shuffled candidate order, so downstream consumers can cross-
    reference by feature_index)."""
    examples_by_fid = {i: [_make_ex(1.0)] for i in range(10)}

    def _fetch_fn(model_id, sae_source, fid, n_examples, tokenizer):
        # Even fids are slow, odd fids are fast → completion order
        # != submission order.
        if fid % 2 == 0:
            time.sleep(0.03)
        return examples_by_fid[fid]

    qc = _make_qc(
        judge=lambda d, e: 2, model_id="m", sae_source="s",
        n_examples=1, fetch_fn=_fetch_fn, max_workers=10,
    )
    candidates = [(i, f"d{i}") for i in range(10)]
    survivors, report = qc.filter(candidates)
    # Survivors preserve input order regardless of fetch completion order.
    assert [fid for fid, _ in survivors] == list(range(10))
    # And per_feature matches the submission order too.
    assert [e.feature_index for e in report.per_feature] == list(range(10))


# ---------------------------------------------------------------------------
# QcOutcome StrEnum smoke test
# ---------------------------------------------------------------------------


def test_qc_outcome_is_strenum_with_5_members():
    """``QcOutcome`` is a :class:`StrEnum` so members compare equal to
    their string value AND iterate as the canonical 5-element set.

    The persisted QC-report JSON schema relies on the bare string values
    (``"accepted"``, ``"rejected"``, ...) — this test pins both the
    member-set sentinel and the str-equality contract that lets bare
    string comparisons in production code keep working transparently.
    """
    assert set(QcOutcome) == {
        QcOutcome.ACCEPTED,
        QcOutcome.REJECTED,
        QcOutcome.FETCH_ERROR,
        QcOutcome.JUDGE_ERROR,
        QcOutcome.EMPTY_EXAMPLES,
    }
    # str-equality contract (StrEnum mixin guarantees this)
    assert QcOutcome.ACCEPTED == "accepted"
    assert QcOutcome.REJECTED == "rejected"
    assert QcOutcome.FETCH_ERROR == "fetch_error"
    assert QcOutcome.JUDGE_ERROR == "judge_error"
    assert QcOutcome.EMPTY_EXAMPLES == "empty_examples"
    # str(member) emits the bare string value (json.dumps relies on this)
    assert str(QcOutcome.ACCEPTED) == "accepted"
    # And value lookup round-trips
    assert QcOutcome("accepted") is QcOutcome.ACCEPTED


def test_qc_outcome_json_serializes_to_string_value():
    """``json.dumps(QcOutcome.X)`` emits the bare string value.

    The persisted ``concept_examples_qc_report.json`` and the Stage 0c
    checkpoint fingerprint dict rely on this — switching from
    :class:`typing.Literal` to :class:`StrEnum` must NOT change the
    on-disk byte representation.
    """
    assert json.dumps({"k": QcOutcome.ACCEPTED}) == '{"k": "accepted"}'
    assert json.dumps({"k": QcOutcome.FETCH_ERROR}) == '{"k": "fetch_error"}'
