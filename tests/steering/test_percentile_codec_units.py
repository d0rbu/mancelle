"""Slice 04 — codec unit-contract + corpus-vs-feature percentile method drift.

Map P3 #6 / §3.5: ``uint8_to_percentile(percentile_to_uint8(p))`` does NOT round-
trip on the same scale. Encoded input is a percent rank in ``[0.5, 99.5]``;
decoded output is a normalized weight in ``[0, 1]``. The slice 04 invariant
"``m → p → uint8 → p → m``" is rank-preserving but lossy by uint8 quantization.

Map P2 #2 / §3.2: ``_percentile_dict`` (corpus-level) uses NEAREST-RANK percentiles
while per-feature percentiles use LINEAR INTERPOLATION. This pinpoints the
documented divergence so a future "harmonization" doesn't silently change either.
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.compute_statistics import _percentile_dict
from steering.conditional.percentile_table import (
    percentile_to_uint8,
    uint8_to_percentile,
)

# ---------------------------------------------------------------------------
# Codec unit contract (T4)
# ---------------------------------------------------------------------------


class TestUint8PercentileUnits:
    """The encoder→decoder round-trip is rank-preserving but RANGE-DROPPING."""

    def test_decoded_values_are_in_unit_interval(self) -> None:
        """``uint8_to_percentile`` returns values in ``[0, 1]`` (normalized weight)."""
        encoded = t.arange(0, 256, dtype=t.uint8)
        decoded = uint8_to_percentile(encoded)
        assert decoded.dtype == t.float32
        assert decoded.min().item() == pytest.approx(0.0)
        assert decoded.max().item() == pytest.approx(1.0)
        assert (decoded >= 0.0).all()
        assert (decoded <= 1.0).all()

    def test_encoded_then_decoded_is_normalized_weight_not_percentile(self) -> None:
        """A percentile ``99.5`` encoded → decoded is ``≈1.0``, NOT ``99.5``.

        Pins map P3 #6: the slice 04 invariant `m → p → uint8 → p → m` is mis-stated.
        Decoding produces a normalized weight; the encoded percentile cannot be
        recovered in ``[0.5, 99.5]`` units.
        """
        pcts = t.tensor([0.5, 50.0, 99.5], dtype=t.float64)
        encoded = percentile_to_uint8(pcts)
        decoded = uint8_to_percentile(encoded)
        # The decoded ``99.5`` is ≈ 1.0, NOT 99.5
        assert decoded[-1].item() < 1.5, (
            "uint8_to_percentile output is normalized weight in [0, 1], "
            "not a percent rank in [0.5, 99.5]"
        )
        # Asserting rank order is preserved.
        assert decoded[0] < decoded[1] < decoded[2]

    def test_round_trip_rank_preserving_on_full_range(self) -> None:
        """Random monotone percentiles preserve ascending order through encode→decode."""
        pcts = t.tensor([0.5, 1.0, 25.0, 50.0, 75.0, 99.0, 99.5], dtype=t.float64)
        decoded = uint8_to_percentile(percentile_to_uint8(pcts))
        # Strictly ascending after encode→decode.
        assert (decoded[1:] >= decoded[:-1]).all(), (
            f"rank order must be preserved; got {decoded.tolist()}"
        )

    @given(
        pct_a=st.floats(min_value=0.5, max_value=99.5, allow_nan=False, allow_infinity=False),
        pct_b=st.floats(min_value=0.5, max_value=99.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, derandomize=True)
    def test_codec_property_rank_preserving(self, pct_a: float, pct_b: float) -> None:
        """Hypothesis: pct_a < pct_b implies decoded_a <= decoded_b after codec."""
        # uint8 quantization can collapse adjacent percentiles to the same bucket;
        # the contract is monotone (≤), not strictly increasing.
        pcts = t.tensor([pct_a, pct_b], dtype=t.float64)
        decoded = uint8_to_percentile(percentile_to_uint8(pcts))
        if pct_a <= pct_b:
            assert decoded[0].item() <= decoded[1].item(), (
                f"codec must be monotone: pct_a={pct_a}, pct_b={pct_b}, "
                f"decoded={decoded.tolist()}"
            )
        else:
            assert decoded[0].item() >= decoded[1].item()


# ---------------------------------------------------------------------------
# Corpus-level (nearest-rank) vs per-feature (linear-interp) drift (T6)
# ---------------------------------------------------------------------------


class TestCorpusPercentileDictNearestRank:
    """Documents that ``_percentile_dict`` uses NEAREST-RANK percentiles.

    Adjacent percentile buckets can have IDENTICAL values for small ``n`` because
    nearest-rank truncates ``int(p/100 * n)`` to the nearest sorted index — small
    increases of ``p`` may not advance the index at all.
    """

    def test_nearest_rank_collapses_adjacent_percentiles_for_small_n(self) -> None:
        """With ``n=10`` and percentiles ``(50, 51, 52, 53, 54)`` — all buckets share ``idx=5``."""
        data = t.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=t.float64)
        # All five percentiles map to int(p/100 * 10) for p in (50, 51, 52, 53, 54)
        # → index 5 → value 5.
        percentiles = (50, 51, 52, 53, 54)
        result = _percentile_dict(data, percentiles)
        values = list(result.values())
        # Nearest-rank produces equal values across adjacent percentiles for tiny n.
        assert all(v == values[0] for v in values), (
            f"nearest-rank percentiles must equal each other for small n; got {result}"
        )

    def test_returns_dict_with_p_keys(self) -> None:
        """Returned dict uses ``p<int>`` keys exactly matching the input integers."""
        data = t.arange(100, dtype=t.float64)
        result = _percentile_dict(data, (1, 50, 99))
        assert set(result.keys()) == {"p1", "p50", "p99"}

    def test_as_int_truncates(self) -> None:
        """``as_int=True`` casts values to int, never rounds."""
        data = t.tensor([0.5, 1.5, 2.5, 3.5, 4.5], dtype=t.float64)
        result = _percentile_dict(data, (50,), as_int=True)
        # Nearest rank: idx = int(0.5 * 5) = 2 → 2.5 → int(2.5) = 2.
        assert result == {"p50": 2}

    def test_round_digits_truncates(self) -> None:
        """``round_digits`` rounds float results to the requested precision."""
        data = t.tensor([0.123456789, 1.0, 2.0], dtype=t.float64)
        result = _percentile_dict(data, (50,), round_digits=4)
        # value at idx=int(0.5*3)=1 → 1.0; rounded to 4 digits.
        assert result == {"p50": pytest.approx(1.0, abs=1e-9)}

    def test_corpus_vs_feature_method_drift_documented(self) -> None:
        """Pin the documented divergence: nearest-rank vs linear-interp.

        The corpus path uses ``_percentile_dict`` (nearest-rank). The per-feature
        path uses linear interpolation through ``compute_per_feature_percentiles``
        / ``vectorized_pass2_percentiles``. This test asserts the NEAREST-RANK
        producer; if a future refactor switches the corpus path to interpolation,
        downstream consumers (viz, plot tooling) that compare corpus-level
        ``magnitude_stats.percentiles`` against per-feature ``percentiles`` will
        break — that switch MUST update this test.
        """
        data = t.tensor([0.0, 1.0, 2.0, 3.0], dtype=t.float64)
        # Linear interpolation at p=50 over 4 points → 1.5.
        # Nearest-rank: idx = int(0.5 * 4) = 2 → value 2.0.
        result = _percentile_dict(data, (50,))
        assert result["p50"] == pytest.approx(2.0), (
            "_percentile_dict must use nearest-rank (truncating idx); "
            "if this changes the corpus-vs-feature drift documentation in "
            "compute_statistics.py:_percentile_dict and downstream viz consumers "
            "must be updated"
        )
