"""Tests for steering/conditional/percentile_table.py.

Tests cover:
- build_percentile_table: shape, dtype, monotonicity validation, missing features
- magnitude_to_percentile: boundary, interpolation, clamping, empty input, constant features
- compute_percentile_checksum: determinism, sensitivity to changes
- percentile_to_uint8 / uint8_to_percentile: round-trip encoding, boundary values
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    normalized_percentile_to_uint8,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stat(p_values: list[float]) -> PerFeatureStat:
    """Create a PerFeatureStat with p1..p99 from a list of 99 values."""
    assert len(p_values) == 99
    percentiles = {f"p{i + 1}": float(v) for i, v in enumerate(p_values)}
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=float(p_values[49]),
        mean_mag_incl_zeros=float(p_values[49]) * 0.5,
        std_mag=1.0,
        min_mag=float(p_values[0]),
        max_mag=float(p_values[-1]),
        median_mag=float(p_values[49]),
        percentiles=percentiles,
    )


def _make_linear_stat(lo: float = 1.0, hi: float = 99.0) -> PerFeatureStat:
    """Create a stat whose p_k = lo + (k-1)*(hi-lo)/98 for k in 1..99.

    With lo=1, hi=99: p_k = k exactly.
    """
    vals = [lo + (hi - lo) * i / 98.0 for i in range(99)]
    return _make_stat(vals)


# ---------------------------------------------------------------------------
# build_percentile_table
# ---------------------------------------------------------------------------


class TestBuildPercentileTable:
    """Construct [n_features, 99] lookup table from per-feature stats."""

    def test_shape_and_dtype(self):
        stats = {"3": _make_linear_stat()}
        table = build_percentile_table(stats, n_sae_features=10)
        assert table.shape == (10, 99)
        assert table.dtype == t.float64

    def test_known_feature_row(self):
        """Feature row matches the p1..p99 values from the stat."""
        stats = {"5": _make_linear_stat(lo=10.0, hi=50.0)}
        table = build_percentile_table(stats, n_sae_features=10)
        expected = t.tensor(
            [10.0 + (50.0 - 10.0) * i / 98.0 for i in range(99)], dtype=t.float64
        )
        t.testing.assert_close(table[5], expected)

    def test_missing_feature_is_zero(self):
        """Features not in per_feature_stats get all-zero rows."""
        stats = {"1": _make_linear_stat()}
        table = build_percentile_table(stats, n_sae_features=5)
        assert (table[0] == 0).all()
        assert (table[2] == 0).all()
        assert (table[3] == 0).all()
        assert (table[4] == 0).all()
        assert table[1].sum() > 0

    def test_empty_stats(self):
        """Empty stats dict produces all-zero table."""
        table = build_percentile_table({}, n_sae_features=3)
        assert table.shape == (3, 99)
        assert (table == 0).all()

    @pytest.mark.parametrize("percentile", range(1, 100))
    def test_nan_at_every_percentile_is_rejected(self, percentile):
        stat = _make_linear_stat()
        stat.percentiles[f"p{percentile}"] = float("nan")
        with pytest.raises(ValueError, match="non-monotonic percentile values"):
            build_percentile_table({"0": stat}, n_sae_features=1)

    def test_non_monotonic_raises(self):
        """Non-monotonic percentile values should raise ValueError."""
        bad_vals = list(range(1, 100))
        bad_vals[50] = bad_vals[49] - 1  # break monotonicity
        stats = {"0": _make_stat([float(v) for v in bad_vals])}
        with pytest.raises(ValueError, match="non-monotonic"):
            build_percentile_table(stats, n_sae_features=1)

    @pytest.mark.parametrize(
        "fid,missing,nonmonotonic,error,message",
        [
            ("bad", True, True, ValueError, "invalid literal for int"),
            ("2", True, True, KeyError, "^'p1'$"),
            ("2", False, True, ValueError, "Feature 2 has non-monotonic percentile values"),
            ("2", False, False, IndexError, "index 2 is out of bounds"),
        ],
    )
    def test_validation_order(self, fid, missing, nonmonotonic, error, message):
        stat = _make_linear_stat()
        if nonmonotonic:
            stat.percentiles["p2"] = 0.0
        if missing:
            del stat.percentiles["p1"], stat.percentiles["p99"]
        with pytest.raises(error, match=message):
            build_percentile_table({"0": _make_linear_stat(), fid: stat}, n_sae_features=1)

    def test_constant_feature_monotonic(self):
        """Constant percentiles (all same value) are considered monotonic."""
        vals = [5.0] * 99
        stats = {"0": _make_stat(vals)}
        table = build_percentile_table(stats, n_sae_features=1)
        assert (table[0] == 5.0).all()

    def test_multiple_features(self):
        """Multiple features are placed in correct rows."""
        stats = {
            "0": _make_linear_stat(1.0, 10.0),
            "3": _make_linear_stat(100.0, 200.0),
        }
        table = build_percentile_table(stats, n_sae_features=5)
        assert table[0, 0].item() == pytest.approx(1.0)
        assert table[0, 98].item() == pytest.approx(10.0)
        assert table[3, 0].item() == pytest.approx(100.0)
        assert table[3, 98].item() == pytest.approx(200.0)
        assert (table[1] == 0).all()
        assert (table[2] == 0).all()

    def test_missing_row_representation_and_checksum_are_frozen(self):
        """Absent features remain zero rows in the checksum input tensor."""
        table = build_percentile_table(
            {"1": _make_linear_stat()},
            n_sae_features=3,
        )
        expected = t.zeros(3, 99, dtype=t.float64)
        expected[1] = t.arange(1, 100, dtype=t.float64)

        assert t.equal(table, expected)
        assert compute_percentile_checksum(table) == (
            "3fcb0fad60d892de3a4567974d52e2b31d3a66b5ee22607b94dfe3598b7d6454"
        )


# ---------------------------------------------------------------------------
# magnitude_to_percentile
# ---------------------------------------------------------------------------


class TestMagnitudeToPercentile:
    """Map feature magnitudes to percentile scores via interpolation."""

    def _simple_table(self):
        """Single-feature table where p_k = k for k in 1..99."""
        stats = {"0": _make_linear_stat()}
        return build_percentile_table(stats, n_sae_features=1)

    def test_below_p1_returns_0_5(self):
        """Magnitude below p1 returns 0.5."""
        table = self._simple_table()
        fids = t.tensor([0])
        mags = t.tensor([0.0])  # below p1=1.0
        result = magnitude_to_percentile(fids, mags, table)
        assert result.item() == pytest.approx(0.5)

    def test_above_p99_returns_99_5(self):
        """Magnitude above p99 returns 99.5."""
        table = self._simple_table()
        fids = t.tensor([0])
        mags = t.tensor([100.0])  # above p99=99.0
        result = magnitude_to_percentile(fids, mags, table)
        assert result.item() == pytest.approx(99.5)

    def test_exact_p1_returns_1(self):
        """Magnitude exactly at p1 returns 1.0 (right=True puts it in-range)."""
        table = self._simple_table()
        fids = t.tensor([0])
        mags = t.tensor([1.0])  # exactly at p1
        result = magnitude_to_percentile(fids, mags, table)
        assert result.item() == pytest.approx(1.0, abs=0.1)

    def test_interpolation_midpoint(self):
        """Magnitude halfway between two percentile boundaries gives correct interpolation."""
        table = self._simple_table()
        fids = t.tensor([0])
        # p50 = 50.0, p51 = 51.0 in linear table. Midpoint at 50.5 → percentile ~50.5
        mags = t.tensor([50.5])
        result = magnitude_to_percentile(fids, mags, table)
        assert 50.0 < result.item() < 51.0

    def test_empty_input(self):
        """Empty feature_ids returns empty result."""
        table = self._simple_table()
        fids = t.zeros(0, dtype=t.int64)
        mags = t.zeros(0)
        result = magnitude_to_percentile(fids, mags, table)
        assert result.shape == (0,)

    def test_batch_of_features(self):
        """Multiple features mapped simultaneously."""
        stats = {
            "0": _make_linear_stat(1.0, 99.0),
            "1": _make_linear_stat(10.0, 200.0),
        }
        table = build_percentile_table(stats, n_sae_features=2)
        fids = t.tensor([0, 0, 1])
        mags = t.tensor([0.0, 50.0, 105.0])  # below p1, mid, mid
        result = magnitude_to_percentile(fids, mags, table)
        assert result.shape == (3,)
        assert result[0].item() == pytest.approx(0.5)  # below p1
        assert 40 < result[1].item() < 60  # roughly middle
        assert 40 < result[2].item() < 60  # roughly middle

    def test_constant_feature_midpoint(self):
        """Constant percentiles (all same value) use frac=0.5 for matching magnitude."""
        vals = [5.0] * 99
        stats = {"0": _make_stat(vals)}
        table = build_percentile_table(stats, n_sae_features=1)
        fids = t.tensor([0])
        mags = t.tensor([5.0])  # exactly at all percentiles
        result = magnitude_to_percentile(fids, mags, table)
        # Should get some valid percentile (not NaN)
        assert t.isfinite(result[0])

    def test_output_range(self):
        """All output values are in [0.5, 99.5]."""
        table = self._simple_table()
        fids = t.zeros(100, dtype=t.int64)
        mags = t.linspace(-10, 200, 100)
        result = magnitude_to_percentile(fids, mags, table)
        assert (result >= 0.5).all()
        assert (result <= 99.5).all()

    def test_monotonic_output(self):
        """Increasing magnitudes produce non-decreasing percentiles."""
        table = self._simple_table()
        fids = t.zeros(50, dtype=t.int64)
        mags = t.linspace(0, 100, 50)
        result = magnitude_to_percentile(fids, mags, table)
        diffs = result[1:] - result[:-1]
        assert (diffs >= -1e-10).all(), "Percentile output is not monotonic"

# ---------------------------------------------------------------------------
# compute_percentile_checksum
# ---------------------------------------------------------------------------


class TestComputePercentileChecksum:
    """SHA-256 checksum of a percentile table."""

    def test_deterministic(self):
        """Same table always produces the same checksum."""
        stats = {"0": _make_linear_stat()}
        table = build_percentile_table(stats, n_sae_features=2)
        c1 = compute_percentile_checksum(table)
        c2 = compute_percentile_checksum(table)
        assert c1 == c2

    def test_hex_string_format(self):
        """Checksum is a 64-character hex string (SHA-256)."""
        table = t.zeros(3, 99, dtype=t.float64)
        checksum = compute_percentile_checksum(table)
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_different_tables_different_checksums(self):
        """Changing one value changes the checksum."""
        table1 = t.zeros(3, 99, dtype=t.float64)
        table2 = table1.clone()
        table2[0, 0] = 1.0
        c1 = compute_percentile_checksum(table1)
        c2 = compute_percentile_checksum(table2)
        assert c1 != c2

    def test_clone_same_checksum(self):
        """Cloned table has same checksum as original."""
        stats = {"0": _make_linear_stat(), "5": _make_linear_stat(10.0, 50.0)}
        table = build_percentile_table(stats, n_sae_features=10)
        c_orig = compute_percentile_checksum(table)
        c_clone = compute_percentile_checksum(table.clone())
        assert c_orig == c_clone

    def test_non_contiguous_same_checksum(self):
        """Non-contiguous tensor produces same checksum as contiguous (forced contiguous internally)."""
        table = t.randn(5, 99, dtype=t.float64)
        # Create non-contiguous view via transpose + transpose back
        table_nc = table.t().t()
        # Both should be contiguous after internal .contiguous() call
        c1 = compute_percentile_checksum(table)
        c2 = compute_percentile_checksum(table_nc)
        assert c1 == c2


# ---------------------------------------------------------------------------
# percentile_to_uint8 / uint8_to_percentile
# ---------------------------------------------------------------------------


class TestPercentileUint8Encoding:
    """Round-trip encoding: percentile → uint8 → float."""

    def test_percentile_to_uint8_boundary_low(self):
        """Percentile 0.5 maps to a small uint8 value."""
        pct = t.tensor([0.5], dtype=t.float64)
        encoded = percentile_to_uint8(pct)
        assert encoded.dtype == t.uint8
        assert encoded.item() >= 0
        assert encoded.item() <= 5  # 0.5/100*255 ≈ 1.275 → rounds to 1

    def test_percentile_to_uint8_boundary_high(self):
        """Percentile 99.5 maps to a high uint8 value."""
        pct = t.tensor([99.5], dtype=t.float64)
        encoded = percentile_to_uint8(pct)
        assert encoded.item() >= 250  # 99.5/100*255 ≈ 253.7

    def test_percentile_to_uint8_midpoint(self):
        """Percentile 50.0 maps to ~127-128."""
        pct = t.tensor([50.0], dtype=t.float64)
        encoded = percentile_to_uint8(pct)
        assert 125 <= encoded.item() <= 130

    def test_uint8_to_percentile_range(self):
        """Decoded values are in [0, 1]."""
        encoded = t.tensor([0, 1, 127, 128, 254, 255], dtype=t.uint8)
        decoded = uint8_to_percentile(encoded)
        assert decoded.dtype == t.float32
        assert (decoded >= 0.0).all()
        assert (decoded <= 1.0).all()

    def test_uint8_to_percentile_0_gives_0(self):
        assert uint8_to_percentile(t.tensor([0], dtype=t.uint8)).item() == 0.0

    def test_uint8_to_percentile_255_gives_1(self):
        assert uint8_to_percentile(t.tensor([255], dtype=t.uint8)).item() == 1.0

    def test_normalized_percentile_to_uint8_round_trip(self):
        """Normalized shard weights encode as the inverse of uint8_to_percentile."""
        values = t.tensor([0.0, 0.5, 128.0 / 255.0, 1.0], dtype=t.float32)
        encoded = normalized_percentile_to_uint8(values)
        assert encoded.tolist() == [0, 128, 128, 255]
        decoded = uint8_to_percentile(encoded)
        t.testing.assert_close(
            decoded,
            t.tensor([0.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]),
            atol=0.0,
            rtol=0.0,
        )

    def test_normalized_uint8_codec_is_exact_for_every_code(self):
        """Decoding and re-encoding preserves every possible stored byte."""
        encoded = t.arange(256, dtype=t.uint8)
        reencoded = normalized_percentile_to_uint8(uint8_to_percentile(encoded))
        assert t.equal(reencoded, encoded)

    def test_round_trip_approximate(self):
        """Encode then decode is approximately identity (within quantization error)."""
        pcts = t.tensor([0.5, 10.0, 25.0, 50.0, 75.0, 90.0, 99.5], dtype=t.float64)
        encoded = percentile_to_uint8(pcts)
        decoded = uint8_to_percentile(encoded)
        # Decoded is in [0,1] range, original is in [0.5, 99.5] range
        # After encoding: uint8 = round(pct/100 * 255)
        # After decoding: decoded = uint8/255
        # So decoded ≈ pct/100
        expected_normalized = pcts.float() / 100.0
        t.testing.assert_close(decoded, expected_normalized, atol=0.01, rtol=0.01)

    def test_batch_encoding(self):
        """Multiple values encode correctly."""
        pcts = t.linspace(0.5, 99.5, 20, dtype=t.float64)
        encoded = percentile_to_uint8(pcts)
        assert encoded.shape == (20,)
        assert encoded.dtype == t.uint8
        # Should be monotonically non-decreasing
        diffs = encoded[1:].int() - encoded[:-1].int()
        assert (diffs >= 0).all()

    def test_clamp_below_zero(self):
        """Values below 0 are clamped to uint8 0."""
        pcts = t.tensor([-1.0], dtype=t.float64)
        encoded = percentile_to_uint8(pcts)
        assert encoded.item() == 0

    def test_clamp_above_max(self):
        """Values above 100 are clamped to uint8 255."""
        pcts = t.tensor([200.0], dtype=t.float64)
        encoded = percentile_to_uint8(pcts)
        assert encoded.item() == 255
