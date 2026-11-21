"""Histogram / bin structure failure domain."""
import pytest

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
)
from steering.eval.feature_filter.types import (
    InvalidHistogramError,
    NoNonzeroActivationsError,
)


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


def _payload(entries):
    return FeatureActivationPayload(
        feature_id=1, description="x", entries=entries,
        freq_hist_bar_heights=[1.0], freq_hist_bar_values=[0.0],
        frac_nonzero=0.01, max_act_approx=10.0,
    )


def _e(mv: float, lo: float, hi: float) -> ActivationEntry:
    return ActivationEntry(
        tokens=["t"], values=[mv], max_value=mv, max_value_token_index=0,
        bin_min=lo, bin_max=hi, bin_contains=0.0,
    )


def test_only_sentinel_entries_raises() -> None:
    with pytest.raises(NoNonzeroActivationsError):
        sample_by_bins(_payload([_e(5, -1, 10), _e(4, -1, 10)]),
                       samples_per_bin=3, tokenizer=_Tok())


def test_empty_payload_raises() -> None:
    with pytest.raises(NoNonzeroActivationsError):
        sample_by_bins(_payload([]), samples_per_bin=3, tokenizer=_Tok())


def test_single_bin_raises() -> None:
    with pytest.raises(InvalidHistogramError, match="expected 5 non-sentinel bins"):
        sample_by_bins(
            _payload([_e(1.0, 0, 2), _e(0.5, 0, 2)]),
            samples_per_bin=3, tokenizer=_Tok(),
        )


def test_four_bins_raises() -> None:
    with pytest.raises(InvalidHistogramError, match="expected 5 non-sentinel bins"):
        sample_by_bins(_payload([
            _e(0.5, 0, 2), _e(3.0, 2, 4), _e(5.0, 4, 6), _e(7.0, 6, 8),
        ]), samples_per_bin=1, tokenizer=_Tok())


def test_negative_bin_min_other_than_sentinel_still_used() -> None:
    # Include all 5 bins; one has a non-sentinel negative bin_min.
    out = sample_by_bins(_payload([
        _e(1.0, -0.5, 2.0),
        _e(3.0, 2.0, 4.0),
        _e(5.0, 4.0, 6.0),
        _e(7.0, 6.0, 8.0),
        _e(9.0, 8.0, 10.0),
    ]), samples_per_bin=1, tokenizer=_Tok())
    assert len(out) == 5
