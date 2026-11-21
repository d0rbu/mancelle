"""Neuronpedia pathologies: empty activations, histogram length mismatch,
all-zero values, tolerated non-monotone binContains."""
import pytest

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
    _parse_payload,
)
from steering.eval.feature_filter.types import (
    InvalidHistogramError,
    NoNonzeroActivationsError,
)


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


def _payload(*, entries=None, heights=None, values=None, frac=0.01) -> FeatureActivationPayload:
    entries = entries if entries is not None else [
        ActivationEntry(tokens=["t"], values=[1.0], max_value=1.0,
                        max_value_token_index=0, bin_min=0.0, bin_max=10.0, bin_contains=0.0),
    ]
    return FeatureActivationPayload(
        feature_id=1, description="d", entries=entries,
        freq_hist_bar_heights=heights if heights is not None else [0.99, 0.01],
        freq_hist_bar_values=values if values is not None else [0.0, 5.0],
        frac_nonzero=frac, max_act_approx=10.0,
    )


def test_empty_activations_list_raises() -> None:
    p = _payload(entries=[])
    with pytest.raises(NoNonzeroActivationsError):
        sample_by_bins(p, samples_per_bin=3, tokenizer=_Tok())


def test_histogram_length_mismatch_raises() -> None:
    # Histogram validation happens at parse time.
    raw = {
        "activations": [],
        "freq_hist_data_bar_heights": [1.0],
        "freq_hist_data_bar_values": [0.0, 5.0],
        "frac_nonzero": 0.01,
    }
    with pytest.raises(InvalidHistogramError):
        _parse_payload(raw, 1)


def test_all_sentinel_entries_raises() -> None:
    # Parser-level 'all-zero histogram' has no exception path, but the sampler's
    # equivalent: every entry is the sentinel `bin_min=-1` bucket → NoNonzero.
    sentinel_entries = [
        ActivationEntry(tokens=["t"], values=[1.0], max_value=1.0,
                        max_value_token_index=0, bin_min=-1.0, bin_max=10.0, bin_contains=0.0),
    ]
    p = _payload(entries=sentinel_entries, heights=[1.0, 0.0], values=[0.0, 0.0])
    with pytest.raises(NoNonzeroActivationsError):
        sample_by_bins(p, samples_per_bin=3, tokenizer=_Tok())


def test_non_monotone_bin_contains_tolerated() -> None:
    # binContains is per-entry metadata; the sampler does not require monotonicity,
    # so a bad value should be carried through without crashing.
    entries = [
        ActivationEntry(tokens=["a"], values=[1.0], max_value=1.0,
                        max_value_token_index=0, bin_min=0.0, bin_max=2.0, bin_contains=-5.0),
        ActivationEntry(tokens=["b"], values=[3.0], max_value=3.0,
                        max_value_token_index=0, bin_min=2.0, bin_max=4.0, bin_contains=0.1),
        ActivationEntry(tokens=["c"], values=[5.0], max_value=5.0,
                        max_value_token_index=0, bin_min=4.0, bin_max=6.0, bin_contains=0.1),
        ActivationEntry(tokens=["d"], values=[7.0], max_value=7.0,
                        max_value_token_index=0, bin_min=6.0, bin_max=8.0, bin_contains=0.1),
        ActivationEntry(tokens=["e"], values=[9.0], max_value=9.0,
                        max_value_token_index=0, bin_min=8.0, bin_max=10.0, bin_contains=0.1),
    ]
    p = _payload(entries=entries)
    samples = sample_by_bins(p, samples_per_bin=3, tokenizer=_Tok())
    assert len(samples) == 5
