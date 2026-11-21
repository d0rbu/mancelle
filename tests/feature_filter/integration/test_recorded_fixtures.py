"""Integration: run bin_sampler against real captured Neuronpedia fixtures."""
import json
from pathlib import Path

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


def _load(name: str) -> FeatureActivationPayload:
    raw = json.loads((FIXTURES / name).read_text())
    entries = [ActivationEntry(**e) for e in raw["entries"]]
    return FeatureActivationPayload(
        feature_id=raw["feature_id"],
        description=raw["description"],
        entries=entries,
        freq_hist_bar_heights=raw["freq_hist_bar_heights"],
        freq_hist_bar_values=raw["freq_hist_bar_values"],
        frac_nonzero=raw["frac_nonzero"],
        max_act_approx=raw["max_act_approx"],
    )


def test_feature_42_five_bands_fifteen_samples() -> None:
    payload = _load("np_feature_gemma_res16k_42.json")
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_Tok())
    assert len(samples) == 15
    assert {s.bin_index for s in samples} == {0, 1, 2, 3, 4}
    peak = next(s for s in samples if s.bin_index == 4)
    assert peak.max_value > 40.0
    bin0 = [s for s in samples if s.bin_index == 0]
    assert all(s.is_absent for s in bin0)


def test_feature_100_mixed_weak_and_absent_bin_zero() -> None:
    payload = _load("np_feature_gemma_res16k_100.json")
    samples = sample_by_bins(payload, samples_per_bin=3, tokenizer=_Tok())
    assert len(samples) == 15
    bin0 = [s for s in samples if s.bin_index == 0]
    has_weak = any((not s.is_absent) and s.max_value > 0 for s in bin0)
    has_absent = any(s.is_absent for s in bin0)
    assert has_weak and has_absent
