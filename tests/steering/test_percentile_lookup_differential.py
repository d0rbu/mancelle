"""Differential coverage for the issue #1269 percentile lookup optimization."""

from __future__ import annotations

import math

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.percentile_table import (
    magnitude_to_percentile,
    percentile_to_uint8,
)


# Frozen pre-#1269 reference implementation (copied from percentile_table.py @ 78c841725). Do NOT "fix" or update; the differential is the point.
def _reference_magnitude_to_percentile(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    percentile_table: t.Tensor,
) -> t.Tensor:
    n = feature_ids.shape[0]
    if n == 0:
        return t.zeros(0, dtype=t.float64)

    rows = percentile_table[feature_ids.long()]  # [N, 99] float64
    missing_rows = (rows == 0).all(dim=1)
    if missing_rows.any():
        missing_feature_ids = feature_ids[missing_rows].long()
        raise ValueError(
            "Percentile statistics are missing for feature IDs "
            f"{missing_feature_ids.tolist()[:5]}; all-zero percentile rows "
            "cannot be queried."
        )
    mags = feature_mags.to(t.float64).unsqueeze(1)  # [N, 1]

    # searchsorted with right=True: idx in [0, 99]
    # idx=0  → mag < p1 (below all)
    # idx=99 → mag >= p99 (above all)
    # idx in [1, 98] → p_idx <= mag < p_{idx+1}
    idx = t.searchsorted(rows, mags, right=True).squeeze(1)  # [N]

    result = t.empty(n, dtype=t.float64, device=feature_ids.device)

    below_mask = idx == 0
    above_mask = idx >= 99
    interp_mask = ~below_mask & ~above_mask

    result[below_mask] = 0.5
    result[above_mask] = 99.5

    if interp_mask.any():
        i_idx = idx[interp_mask]  # values in [1, 98]
        i_rows = rows[interp_mask]  # [M, 99]
        i_mags = feature_mags.to(t.float64)[interp_mask]  # [M]

        # left_val = p_{i_idx}, right_val = p_{i_idx+1}
        # row is 0-indexed: row[k] = p_{k+1}, so p_{i_idx} = row[i_idx - 1]
        left_val = i_rows.gather(1, (i_idx - 1).unsqueeze(1)).squeeze(1)  # p_{idx}
        right_val = i_rows.gather(1, i_idx.unsqueeze(1)).squeeze(1)  # p_{idx+1}

        denom = right_val - left_val
        # When left_val == right_val (constant region), use frac=0.5 (midpoint)
        frac = t.where(
            denom == 0.0,
            t.full_like(denom, 0.5),
            (i_mags - left_val) / denom,
        )
        result[interp_mask] = i_idx.to(t.float64) + frac

    return result


def _assert_equal(fids: t.Tensor, mags: t.Tensor, table: t.Tensor) -> None:
    actual = magnitude_to_percentile(fids, mags, table)
    expected = _reference_magnitude_to_percentile(fids, mags, table)
    assert t.equal(actual, expected)


class TestDifferentialExactEquality:
    """New output equals the frozen reference bitwise for float64 and uint8."""

    @pytest.mark.parametrize("dtype", [t.float32, t.float64])
    def test_dense_grid_single_feature(self, dtype: t.dtype) -> None:
        table = t.arange(1, 100, dtype=t.float64).unsqueeze(0)
        mags = t.linspace(-5, 105, 2201, dtype=dtype)
        fids = t.zeros(mags.shape[0], dtype=t.int64)
        actual = magnitude_to_percentile(fids, mags, table)
        expected = _reference_magnitude_to_percentile(fids, mags, table)
        assert t.equal(actual, expected)
        assert t.equal(percentile_to_uint8(actual), percentile_to_uint8(expected))

    def test_every_threshold_and_midpoint(self) -> None:
        row = t.arange(1, 100, dtype=t.float64)
        mags = t.cat((row, (row[:-1] + row[1:]) / 2))
        _assert_equal(t.zeros(mags.shape[0], dtype=t.int64), mags, row.unsqueeze(0))

    @pytest.mark.parametrize(
        ("row", "mags"),
        [
            ([1.0] * 50 + [2.0] * 49, [0.0, 1.0, 1.5, 2.0, 3.0]),
            ([5.0] * 99, [4.0, 5.0, 6.0]),
            ([0.0] * 98 + [1.0], [-1.0, 0.0, 0.5, 1.0, 2.0]),
        ],
    )
    def test_plateau_rows(self, row: list[float], mags: list[float]) -> None:
        mag_tensor = t.tensor(mags, dtype=t.float64)
        _assert_equal(
            t.zeros(len(mags), dtype=t.int64),
            mag_tensor,
            t.tensor(row, dtype=t.float64).unsqueeze(0),
        )

    def test_negative_value_rows(self) -> None:
        table = t.linspace(-50.0, -1.0, 99, dtype=t.float64).unsqueeze(0)
        mags = t.linspace(-60.0, 5.0, 1000, dtype=t.float64)
        _assert_equal(t.zeros(mags.shape[0], dtype=t.int64), mags, table)

    def test_special_float_mags(self) -> None:
        table = t.arange(1, 100, dtype=t.float64).unsqueeze(0)
        mags = t.tensor(
            [math.nan, math.inf, -math.inf, 0.0, -0.0, 5e-324, 1e308],
            dtype=t.float64,
        )
        _assert_equal(t.zeros(mags.shape[0], dtype=t.int64), mags, table)

    def test_mixed_feature_batch(self) -> None:
        generator = t.Generator().manual_seed(1269)
        table = t.stack(
            (
                t.arange(1, 100, dtype=t.float64),
                t.linspace(-100, -1, 99, dtype=t.float64),
                t.full((99,), 5.0, dtype=t.float64),
                t.cat((t.zeros(98, dtype=t.float64), t.ones(1, dtype=t.float64))),
                t.linspace(0.25, 10.0, 99, dtype=t.float64),
                t.linspace(-10.0, 10.0, 99, dtype=t.float64),
                t.arange(1, 100, dtype=t.float64).square(),
                t.linspace(1e-12, 1e12, 99, dtype=t.float64),
            )
        )
        fids = t.randint(0, 8, (10_000,), generator=generator)
        mags = t.randn(10_000, generator=generator, dtype=t.float64) * 100
        _assert_equal(fids, mags, table)

    def test_empty_input_device_and_dtype(self) -> None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        result = magnitude_to_percentile(
            t.empty(0, dtype=t.int64, device=device),
            t.empty(0, device=device),
            t.ones((1, 99), dtype=t.float64, device=device),
        )
        assert result.shape == (0,)
        assert result.dtype == t.float64
        assert result.device.type == "cpu"


class TestMissingRowParity:
    """The RAISING inputs stay identical to the frozen reference.

    The message deliberately diverges from the frozen reference (PR #1311):
    the old text named up to five raw hits and said nothing about what to do
    about them. The current text reports the DISTINCT missing population and
    names the actionable fix (rebuild statistics over the full ACTIVE feature
    set, not the allowed representatives). Divergence in the *set of inputs
    that raise* would still be a regression, so that part stays differential.
    """

    def test_missing_row_message_names_distinct_ids(self) -> None:
        table = t.zeros((4, 99), dtype=t.float64)
        table[0] = t.arange(1, 100, dtype=t.float64)
        table[2] = t.arange(1, 100, dtype=t.float64)
        fids = t.tensor([0, 1, 3])
        mags = t.ones(3)
        match = (
            r"Percentile statistics are missing for 2 distinct feature ID\(s\) "
            r"\(first 2: \[1, 3\]\); all-zero percentile rows cannot be queried\."
        )
        with pytest.raises(ValueError, match=match) as actual:
            magnitude_to_percentile(fids, mags, table)
        # The frozen reference rejects exactly the same input.
        with pytest.raises(ValueError):
            _reference_magnitude_to_percentile(fids, mags, table)
        assert "RAW pre-selection feature ids" in str(actual.value)
        assert "not only the allowed representatives" in str(actual.value)

    def test_distinct_count_counts_the_whole_missing_population(self) -> None:
        """Repeated hits collapse; the reported count is the distinct one."""
        table = t.zeros((8, 99), dtype=t.float64)
        fids = t.tensor([7, 1, 2, 3, 4, 5, 6, 7, 7, 7])
        mags = t.ones(fids.shape[0])
        with pytest.raises(ValueError) as actual:
            magnitude_to_percentile(fids, mags, table)
        with pytest.raises(ValueError):
            _reference_magnitude_to_percentile(fids, mags, table)
        message = str(actual.value)
        assert "7 distinct feature ID(s)" in message
        assert "(first 5: [1, 2, 3, 4, 5])" in message


class TestInvalidInputContracts:
    """Invalid inputs must fail loud — never silently broadcast or misroute.

    Review findings (#1269): the gather-based lanes would happily broadcast a
    mismatched-magnitude shape that the reference's searchsorted rejected, and
    a NaN-poisoned table (unconstructible via build_percentile_table, whose
    monotonicity check rejects NaN rows) would silently diverge from
    searchsorted's NaN-comparison routing. Both now raise.
    """

    def test_shape_mismatch_raises_not_broadcasts(self) -> None:
        table = t.arange(1, 100, dtype=t.float64).repeat(2, 1)
        fids = t.tensor([0, 1])
        with pytest.raises(ValueError, match=r"feature_mags shape \(1,\) != feature_ids shape \(2,\)"):
            magnitude_to_percentile(fids, t.tensor([1000.0]), table)
        # The frozen reference also refused this input (searchsorted shape error).
        with pytest.raises((ValueError, RuntimeError)):
            _reference_magnitude_to_percentile(fids, t.tensor([1000.0]), table)

    def test_nan_probed_table_edge_raises(self) -> None:
        table = t.arange(1, 100, dtype=t.float64).unsqueeze(0)
        table[0, 98] = math.nan
        with pytest.raises(ValueError, match="NaN in probed entries"):
            magnitude_to_percentile(t.tensor([0]), t.tensor([50.0]), table)

    def test_nan_probed_table_interior_raises(self) -> None:
        table = t.arange(1, 100, dtype=t.float64).unsqueeze(0)
        table[0, 49] = math.nan
        with pytest.raises(ValueError, match="NaN in probed entries"):
            magnitude_to_percentile(t.tensor([0]), t.tensor([0.0]), table)

    def test_nan_never_probed_is_tolerated(self) -> None:
        # A NaN the binary search never touches for this query cannot affect
        # the result; the guard is probe-scoped by design (full-row scans are
        # exactly the [N, 99] traffic this rewrite removes). For mag=1.5 on
        # the [1..99] row the probe path is {49, 24, 12, 6, 3, 1, 0} plus the
        # first/last sentinels and interp endpoints {0, 1}; index 90 is never
        # touched, and the reference's own probes skip it too.
        table = t.arange(1, 100, dtype=t.float64).unsqueeze(0)
        table[0, 90] = math.nan
        fids = t.tensor([0])
        mags = t.tensor([1.5])
        actual = magnitude_to_percentile(fids, mags, table)
        expected = _reference_magnitude_to_percentile(fids, mags, table)
        assert t.equal(actual, expected)


@st.composite
def _table_queries(draw: st.DrawFn) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    n_features = draw(st.integers(min_value=1, max_value=8))
    row_strategy = st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
        min_size=99,
        max_size=99,
    ).map(sorted)
    rows = draw(st.lists(row_strategy, min_size=n_features, max_size=n_features))
    table = t.tensor(rows, dtype=t.float64)
    valid_fids = [i for i, row in enumerate(rows) if any(value != 0.0 for value in row)]
    n = draw(st.integers(min_value=0, max_value=512))
    if n == 0 or not valid_fids:
        return table, t.empty(0, dtype=t.int64), t.empty(0, dtype=t.float64)
    fids = draw(st.lists(st.sampled_from(valid_fids), min_size=n, max_size=n))
    finite_or_special = st.floats(
        allow_nan=True,
        allow_infinity=True,
        width=64,
    )
    row_values = st.sampled_from([value for row in rows for value in row])
    mags = draw(st.lists(finite_or_special | row_values, min_size=n, max_size=n))
    return table, t.tensor(fids, dtype=t.int64), t.tensor(mags, dtype=t.float64)


class TestHypothesisDifferential:
    @given(_table_queries())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_tables_and_mags(self, case: tuple[t.Tensor, t.Tensor, t.Tensor]) -> None:
        table, fids, mags = case
        _assert_equal(fids, mags, table)


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA parity needs a GPU")
class TestCudaParity:
    def test_cuda_matches_cpu_reference(self) -> None:
        table = t.stack(
            (
                t.arange(1, 100, dtype=t.float64),
                t.linspace(-10, 10, 99, dtype=t.float64),
            )
        )
        fids = t.tensor([0, 1, 1, 0], dtype=t.int64)
        mags = t.tensor([math.nan, -2.5, math.inf, 45.5], dtype=t.float64)
        actual = magnitude_to_percentile(fids.cuda(), mags.cuda(), table.cuda()).cpu()
        expected = _reference_magnitude_to_percentile(fids, mags, table)
        assert t.equal(actual, expected)
