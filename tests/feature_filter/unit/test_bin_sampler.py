"""Tests for bin_sampler.sample_by_bins."""
from itertools import groupby

import pytest

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.neuronpedia_activations import FeatureActivationPayload
from steering.eval.feature_filter.types import (
    BinSample,
    InvalidHistogramError,
    NoNonzeroActivationsError,
)
from tests.feature_filter.fixtures.payload_builder import build_bin_structured_payload, build_entry


def _payload_from_entries(entries, feature_id=7):
    return FeatureActivationPayload(
        feature_id=feature_id, description="x", entries=entries,
        freq_hist_bar_heights=[1.0], freq_hist_bar_values=[0.0],
        frac_nonzero=0.01, max_act_approx=10.0,
    )


class _FakeTokenizer:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def test_happy_path_5_bins_3_samples_each() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())
    assert len(samples) == 15
    assert all(isinstance(s, BinSample) for s in samples)


def test_peak_first_order() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())
    bin_indices = [s.bin_index for s in samples]
    assert bin_indices == sorted(bin_indices, reverse=True)
    assert set(bin_indices) == {0, 1, 2, 3, 4}


def test_samples_within_bin_descending_max_value() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())
    for bin_idx, group in groupby(samples, key=lambda s: s.bin_index):
        group_list = list(group)
        mvs = [s.max_value for s in group_list]
        assert mvs == sorted(mvs, reverse=True), f"bin {bin_idx} not desc: {mvs}"


def test_sentinel_minus_one_entries_excluded() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5, overall_topk=20)
    samples = sample_by_bins(payload, samples_per_bin=5, tokenizer=_FakeTokenizer())
    assert len(samples) == 25
    # Positive check: no sample carries a token from the sentinel bucket
    # (fixture tags sentinel entries with "top{i}"), and no sample has
    # bin_index=-1 or bin_min=-1 in its provenance.
    for s in samples:
        assert 0 <= s.bin_index <= 4
        assert not any(tok.startswith("top") for tok in s.example.tokens)


def test_band_labels_correct() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=1, tokenizer=_FakeTokenizer())
    by_idx = {s.bin_index: s.band_label for s in samples}
    assert by_idx[4] == "band 5/5 (peak firing)"
    assert by_idx[3] == "band 4/5 (very high firing)"
    assert by_idx[2] == "band 3/5 (high firing)"
    assert by_idx[1] == "band 2/5 (moderate firing)"
    assert by_idx[0] == "band 1/5 (absent — feature did not fire)"


def test_band_label_weak_firing_when_bin_zero_mv_positive() -> None:
    entries = [
        build_entry(max_value=0.3, bin_min=0.0, bin_max=2.0, tokens=["w"]),
        build_entry(max_value=3.0, bin_min=2.0, bin_max=4.0, tokens=["a"]),
        build_entry(max_value=5.0, bin_min=4.0, bin_max=6.0, tokens=["b"]),
        build_entry(max_value=7.0, bin_min=6.0, bin_max=8.0, tokens=["c"]),
        build_entry(max_value=9.0, bin_min=8.0, bin_max=10.0, tokens=["d"]),
    ]
    payload = FeatureActivationPayload(
        feature_id=3, description="x", entries=entries,
        freq_hist_bar_heights=[1.0], freq_hist_bar_values=[0.0],
        frac_nonzero=0.01, max_act_approx=10.0,
    )
    samples = sample_by_bins(payload, samples_per_bin=1, tokenizer=_FakeTokenizer())
    bin0 = [s for s in samples if s.bin_index == 0]
    assert len(bin0) == 1
    assert bin0[0].is_absent is False
    assert bin0[0].band_label == "band 1/5 (weak firing)"


def test_partial_bins_raise_invalid_histogram() -> None:
    entries = [
        build_entry(max_value=1.0, bin_min=0.0, bin_max=2.0),
        build_entry(max_value=4.5, bin_min=4.0, bin_max=6.0),
    ]
    payload = _payload_from_entries(entries)
    with pytest.raises(InvalidHistogramError, match="expected 5 non-sentinel bins"):
        sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())


def test_all_bins_empty_after_sentinel_filter_raises() -> None:
    entries = [
        build_entry(max_value=9.0, bin_min=-1.0, bin_max=10.0),
        build_entry(max_value=8.0, bin_min=-1.0, bin_max=10.0),
    ]
    payload = _payload_from_entries(entries)
    with pytest.raises(NoNonzeroActivationsError, match="non-sentinel"):
        sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())


def test_samples_per_bin_greater_than_available_returns_all() -> None:
    entries = [
        build_entry(max_value=1.0, bin_min=0.0, bin_max=2.0),
        build_entry(max_value=0.5, bin_min=0.0, bin_max=2.0),
        build_entry(max_value=3.0, bin_min=2.0, bin_max=4.0),
        build_entry(max_value=5.0, bin_min=4.0, bin_max=6.0),
        build_entry(max_value=7.0, bin_min=6.0, bin_max=8.0),
        build_entry(max_value=9.0, bin_min=8.0, bin_max=10.0),
    ]
    payload = _payload_from_entries(entries)
    samples = sample_by_bins(payload, samples_per_bin=5, tokenizer=_FakeTokenizer())
    # bin 0 has 2 entries; other 4 bins have 1 each → 6 total
    assert len(samples) == 6


def test_samples_per_bin_one() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=1, tokenizer=_FakeTokenizer())
    assert len(samples) == 5
    assert sorted(s.bin_index for s in samples) == [0, 1, 2, 3, 4]


def test_absent_detection_in_bin_zero() -> None:
    payload = build_bin_structured_payload(max_act=10.0, per_bin=5)
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_FakeTokenizer())
    bin0 = [s for s in samples if s.bin_index == 0]
    assert all(s.is_absent for s in bin0)
    assert all(s.band_label == "band 1/5 (absent — feature did not fire)" for s in bin0)


def test_weak_firing_label_when_bin_zero_has_positive_mv() -> None:
    entries = [
        build_entry(max_value=0.5, bin_min=0.0, bin_max=2.0),
        build_entry(max_value=0.0, bin_min=0.0, bin_max=2.0),
        build_entry(max_value=3.0, bin_min=2.0, bin_max=4.0),
        build_entry(max_value=5.0, bin_min=4.0, bin_max=6.0),
        build_entry(max_value=7.0, bin_min=6.0, bin_max=8.0),
        build_entry(max_value=9.0, bin_min=8.0, bin_max=10.0),
    ]
    payload = _payload_from_entries(entries)
    samples = sample_by_bins(payload, samples_per_bin=2, tokenizer=_FakeTokenizer())
    labels = [s.band_label for s in samples]
    assert "band 1/5 (weak firing)" in labels
    assert "band 1/5 (absent — feature did not fire)" in labels
