"""Exact tests for the bounded-state feature-statistics reducers.

These tests intentionally retain sort/expansion-based reference calculations.
The production reducer is only useful if it is an exact change of
representation, so randomized tests use ordinary equality throughout: no
``allclose``, percentile tolerance, or histogram tolerance is permitted.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

import steering.exact_statistics as exact_statistics_module
from steering.exact_statistics import (
    INT64_MAX,
    MAX_EXACT_FLOAT64_INTEGER,
    POSITIVE_FINITE_F16_COUNT,
    ExactFeaturePercentiles,
    ExactMagnitudeCounts,
    ExactStatisticsError,
    WeightedDistribution,
    _read_cgroup_available_bytes,
    count_table_nbytes,
    effective_available_memory_bytes,
    encode_f16_count_keys,
    require_count_table_memory,
    shared_histogram_edges,
)

PERCENTILES = tuple(range(1, 100))


def decode_f16_count_keys(keys: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Decode count keys for round-trip tests of the production encoder."""
    if keys.ndim != 1 or keys.dtype != t.int64:
        raise ExactStatisticsError(
            f"keys must be a 1-D int64 tensor, got shape={tuple(keys.shape)} dtype={keys.dtype}"
        )
    if keys.numel() == 0:
        return (
            t.empty(0, dtype=t.int64, device=keys.device),
            t.empty(0, dtype=t.float32, device=keys.device),
        )
    if bool((keys < 0).any().item()):
        raise ExactStatisticsError("composite count keys must be nonnegative")
    feature_ids = t.div(keys, POSITIVE_FINITE_F16_COUNT, rounding_mode="floor")
    bit_patterns = (keys.remainder(POSITIVE_FINITE_F16_COUNT) + 1).to(t.int16)
    feature_mags = bit_patterns.contiguous().view(t.float16).to(t.float32)
    return feature_ids, feature_mags


def _positive_f16_values(bits: t.Tensor) -> t.Tensor:
    """Interpret positive finite float16 bit patterns as loaded float32."""
    return bits.to(t.int16).contiguous().view(t.float16).to(t.float32)


def _f16(values: list[float]) -> t.Tensor:
    """Round values exactly as shard persistence does, then load as float32."""
    return t.tensor(values, dtype=t.float16).to(t.float32)


def _empty_counts(n_features: int, *, total_count: int = 0) -> ExactMagnitudeCounts:
    return ExactMagnitudeCounts(
        t.zeros((n_features, POSITIVE_FINITE_F16_COUNT), dtype=t.int64),
        total_count=total_count,
    )


def _sort_reference(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    occurrence: t.Tensor,
    mag_subsample: t.Tensor,
    global_mag_min: float,
    *,
    percentiles: tuple[int, ...] = PERCENTILES,
    n_hist_bins: int = 50,
) -> tuple[dict[int, ExactFeaturePercentiles], list[float] | None]:
    """Independent copy of the pre-change sort-based numerical contract."""
    active_ids = (occurrence > 0).nonzero(as_tuple=True)[0]
    if active_ids.numel() == 0:
        return {}, None

    if mag_subsample.numel() > 0:
        sample_f64 = mag_subsample.to(t.float64)
        mag_lo = global_mag_min
        k99 = max(1, int(len(sample_f64) * 0.99))
        mag_p99 = sample_f64.kthvalue(k99).values.item()
        mag_hi = max(mag_p99, mag_lo + 1e-6)
    else:
        mag_lo, mag_hi = 0.0, 1.0
    edges = t.linspace(mag_lo, mag_hi, n_hist_bins + 1, dtype=t.float64)
    fractions = t.tensor([p / 100.0 for p in percentiles], dtype=t.float64)

    result: dict[int, ExactFeaturePercentiles] = {}
    for fid_tensor in active_ids:
        fid = int(fid_tensor.item())
        sorted_mags = feature_mags[feature_ids == fid].to(t.float64).sort().values
        count = len(sorted_mags)
        assert count == int(occurrence[fid].item())

        continuous = (count - 1) * fractions
        lower = continuous.floor().to(t.int64)
        upper = continuous.ceil().to(t.int64).clamp(max=count - 1)
        frac = continuous - lower.to(t.float64)
        values = sorted_mags[lower] + frac * (sorted_mags[upper] - sorted_mags[lower])

        median_continuous = (count - 1) * 0.5
        median_lower = int(median_continuous)
        median_upper = min(median_lower + 1, count - 1)
        median_fraction = median_continuous - median_lower
        median = sorted_mags[median_lower].item() + median_fraction * (
            sorted_mags[median_upper].item() - sorted_mags[median_lower].item()
        )

        clamped = sorted_mags.clamp(edges[0], edges[-1])
        bin_ids = t.bucketize(clamped, edges[1:]).clamp(max=n_hist_bins - 1)
        histogram = t.zeros(n_hist_bins, dtype=t.int64)
        histogram.scatter_add_(0, bin_ids, t.ones_like(bin_ids))

        result[fid] = ExactFeaturePercentiles(
            percentiles={
                f"p{p}": round(values[index].item(), 6) for index, p in enumerate(percentiles)
            },
            median=round(median, 6),
            histogram=histogram.tolist(),
        )

    return result, [round(value, 6) for value in edges.tolist()]


def _expanded_nearest_rank(
    values: Sequence[int | float],
    percentiles: tuple[int, ...],
    *,
    round_digits: int | None = None,
    as_int: bool = False,
) -> dict[str, int | float]:
    sorted_values = sorted(values)
    result: dict[str, int | float] = {}
    for percentile in percentiles:
        rank = min(int(percentile / 100 * len(sorted_values)), len(sorted_values) - 1)
        value = sorted_values[rank]
        if as_int:
            result[f"p{percentile}"] = int(value)
        elif round_digits is not None:
            result[f"p{percentile}"] = round(float(value), round_digits)
        else:
            result[f"p{percentile}"] = float(value)
    return result


class TestCountTableSizing:
    @pytest.mark.parametrize(
        ("n_features", "expected"),
        [
            (1, 31_743 * 8),
            (16_384, 4_160_618_496),
            (65_536, 16_642_473_984),
            (1_048_576, 266_279_583_744),
        ],
    )
    def test_exact_bytes(self, n_features: int, expected: int) -> None:
        assert count_table_nbytes(n_features) == expected

    @pytest.mark.parametrize("value", [0, -1, -65_536])
    def test_rejects_nonpositive_width(self, value: int) -> None:
        with pytest.raises(ExactStatisticsError, match="must be positive"):
            count_table_nbytes(value)

    @pytest.mark.parametrize("value", [True, False, 1.0, "65536", None])
    def test_rejects_noninteger_width(self, value: object) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            count_table_nbytes(cast(Any, value))

    def test_admission_returns_exact_required_bytes(self) -> None:
        required = count_table_nbytes(7)
        assert require_count_table_memory(7, available_bytes=required) == required

    def test_admission_rejects_one_byte_short(self) -> None:
        required = count_table_nbytes(7)
        with pytest.raises(ExactStatisticsError) as exc_info:
            require_count_table_memory(7, available_bytes=required - 1)
        message = str(exc_info.value)
        assert "width=7" in message
        assert f"required={required:,} bytes" in message
        assert f"available={required - 1:,} bytes" in message
        assert "Refusing to allocate or restore magnitude spills" in message

    @pytest.mark.parametrize("value", [-1, -10_000])
    def test_admission_rejects_negative_availability(self, value: int) -> None:
        with pytest.raises(ExactStatisticsError, match="must be nonnegative"):
            require_count_table_memory(1, available_bytes=value)

    @pytest.mark.parametrize("value", [True, 1.5, "100", None])
    def test_admission_rejects_explicit_noninteger_availability(self, value: object) -> None:
        if value is None:
            return
        with pytest.raises(TypeError, match="available_bytes must be an int"):
            require_count_table_memory(1, available_bytes=cast(Any, value))


class TestCgroupMemoryDetection:
    @staticmethod
    def _write_limit(group: Path, *, maximum: str, current: str) -> None:
        group.mkdir(parents=True, exist_ok=True)
        (group / "memory.max").write_text(maximum)
        (group / "memory.current").write_text(current)

    def _write_v2_group(
        self,
        tmp_path: Path,
        *,
        cgroup_line: str = "0::/research/job-7\n",
        maximum: str = "1000",
        current: str = "300",
    ) -> tuple[Path, Path]:
        proc = tmp_path / "proc-self-cgroup"
        proc.write_text(cgroup_line)
        root = tmp_path / "cgroup"
        parent = root / "research"
        group = root / "research" / "job-7"
        self._write_limit(root, maximum="max", current="0")
        self._write_limit(parent, maximum="max", current="0")
        self._write_limit(group, maximum=maximum, current=current)
        return proc, root

    def test_reads_remaining_v2_limit(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path)
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 700
        )

    def test_saturates_at_zero_when_current_exceeds_limit(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(
            tmp_path,
            maximum="200",
            current="300",
        )
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 0
        )

    def test_unlimited_group_returns_none(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="max")
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            is None
        )

    def test_finite_parent_constrains_unlimited_leaf(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="max")
        self._write_limit(root / "research", maximum="900", current="350")

        assert _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=root) == 550

    def test_tightest_remaining_limit_wins_across_every_level(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="1000", current="100")
        self._write_limit(root / "research", maximum="800", current="300")
        self._write_limit(root, maximum="2000", current="700")

        assert _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=root) == 500

    def test_leaf_wins_when_ancestor_headroom_is_larger(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="600", current="300")
        self._write_limit(root / "research", maximum="1000", current="200")
        self._write_limit(root, maximum="2000", current="500")

        assert _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=root) == 300

    def test_ancestor_over_limit_saturates_effective_headroom_at_zero(
        self,
        tmp_path: Path,
    ) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="max")
        self._write_limit(root / "research", maximum="200", current="300")

        assert _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=root) == 0

    @pytest.mark.parametrize(
        "cgroup_line",
        [
            "",
            "12:memory:/legacy\n",
            "malformed\n",
            "0:cpu:/not-unified\n",
        ],
    )
    def test_non_v2_membership_returns_none(
        self,
        tmp_path: Path,
        cgroup_line: str,
    ) -> None:
        proc = tmp_path / "proc-self-cgroup"
        proc.write_text(cgroup_line)
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=tmp_path / "missing",
            )
            is None
        )

    @pytest.mark.parametrize(
        ("maximum", "current"),
        [("invalid", "1"), ("100", "invalid")],
    )
    def test_malformed_memory_files_return_none(
        self,
        tmp_path: Path,
        maximum: str,
        current: str,
    ) -> None:
        proc, root = self._write_v2_group(
            tmp_path,
            maximum=maximum,
            current=current,
        )
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            is None
        )

    def test_malformed_ancestor_memory_file_returns_none(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path)
        self._write_limit(root / "research", maximum="invalid", current="1")

        assert _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=root) is None

    def test_membership_cannot_escape_cgroup_mount(self, tmp_path: Path) -> None:
        proc = tmp_path / "proc-self-cgroup"
        proc.write_text("0::/../../outside\n")

        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc,
                cgroup_root=tmp_path / "cgroup",
            )
            is None
        )

    def test_effective_availability_uses_smaller_cgroup_value(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path)
        assert (
            effective_available_memory_bytes(
                host_available_bytes=900,
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 700
        )

    def test_effective_availability_uses_host_when_smaller(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path)
        assert (
            effective_available_memory_bytes(
                host_available_bytes=600,
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 600
        )

    def test_effective_availability_honors_tighter_parent(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="max")
        self._write_limit(root / "research", maximum="800", current="300")

        assert (
            effective_available_memory_bytes(
                host_available_bytes=900,
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 500
        )

    def test_effective_availability_uses_host_for_unlimited_group(self, tmp_path: Path) -> None:
        proc, root = self._write_v2_group(tmp_path, maximum="max")
        assert (
            effective_available_memory_bytes(
                host_available_bytes=600,
                proc_cgroup_path=proc,
                cgroup_root=root,
            )
            == 600
        )

    def test_effective_availability_rejects_negative_host(self, tmp_path: Path) -> None:
        with pytest.raises(ExactStatisticsError, match="must be nonnegative"):
            effective_available_memory_bytes(
                host_available_bytes=-1,
                proc_cgroup_path=tmp_path / "missing",
            )

    @pytest.mark.parametrize("host", [True, 1.5, "100", []])
    def test_effective_availability_rejects_noninteger_host(
        self,
        tmp_path: Path,
        host: object,
    ) -> None:
        with pytest.raises(TypeError, match="host_available_bytes must be an int"):
            effective_available_memory_bytes(
                host_available_bytes=cast(Any, host),
                proc_cgroup_path=tmp_path / "missing",
            )


class TestFloat16KeyCodec:
    def test_exhaustive_positive_finite_domain_is_strictly_ordered(self) -> None:
        bits = t.arange(1, POSITIVE_FINITE_F16_COUNT + 1, dtype=t.int32)
        values = _positive_f16_values(bits)

        assert len(values) == 31_743
        assert bool(t.isfinite(values).all().item())
        assert bool((values > 0).all().item())
        assert bool((values[1:] > values[:-1]).all().item())
        assert values[0].item() == 2**-24
        assert values[-1].item() == 65_504.0

    def test_exhaustive_domain_round_trips_through_composite_keys(self) -> None:
        bits = t.arange(1, POSITIVE_FINITE_F16_COUNT + 1, dtype=t.int32)
        values = _positive_f16_values(bits)
        ids = t.zeros(len(values), dtype=t.int32)

        keys = encode_f16_count_keys(ids, values, n_sae_features=1)
        decoded_ids, decoded_values = decode_f16_count_keys(keys)

        assert t.equal(keys, t.arange(POSITIVE_FINITE_F16_COUNT, dtype=t.int64))
        assert t.equal(decoded_ids, ids.to(t.int64))
        assert t.equal(decoded_values, values)

    def test_feature_id_is_major_key_component(self) -> None:
        bits = t.tensor([1, 2, 0x7BFE, 0x7BFF], dtype=t.int32)
        values = _positive_f16_values(bits)
        ids = t.tensor([0, 1, 65_534, 65_535], dtype=t.int32)

        keys = encode_f16_count_keys(ids, values, n_sae_features=65_536)
        decoded_ids, decoded_values = decode_f16_count_keys(keys)

        expected = ids.to(t.int64) * POSITIVE_FINITE_F16_COUNT + bits - 1
        assert t.equal(keys, expected)
        assert t.equal(decoded_ids, ids.to(t.int64))
        assert t.equal(decoded_values, values)

    def test_empty_codec_preserves_device_and_dtypes(self) -> None:
        keys = encode_f16_count_keys(
            t.empty(0, dtype=t.int32),
            t.empty(0, dtype=t.float32),
            n_sae_features=1,
        )
        ids, mags = decode_f16_count_keys(keys)
        assert keys.shape == ids.shape == mags.shape == (0,)
        assert keys.dtype == ids.dtype == t.int64
        assert mags.dtype == t.float32
        assert keys.device == ids.device == mags.device == t.device("cpu")

    @pytest.mark.parametrize(
        ("ids", "mags", "match"),
        [
            (t.zeros((1, 1), dtype=t.int32), t.ones(1), "must both be 1-D"),
            (t.zeros(1, dtype=t.int32), t.ones((1, 1)), "must both be 1-D"),
            (t.zeros(2, dtype=t.int32), t.ones(1), "shape"),
            (t.zeros(1, dtype=t.int64), t.ones(1), "dtype int32"),
            (t.zeros(1, dtype=t.int16), t.ones(1), "dtype int32"),
            (t.zeros(1, dtype=t.int32), t.ones(1, dtype=t.float64), "dtype float32"),
            (t.zeros(1, dtype=t.int32), t.ones(1, dtype=t.float16), "dtype float32"),
        ],
    )
    def test_rejects_invalid_tensor_contracts(
        self,
        ids: t.Tensor,
        mags: t.Tensor,
        match: str,
    ) -> None:
        with pytest.raises(ExactStatisticsError, match=match):
            encode_f16_count_keys(ids, mags, n_sae_features=1)

    @pytest.mark.parametrize("feature_id", [-2, -1, 1, 2**31 - 1])
    def test_rejects_out_of_range_feature_ids(self, feature_id: int) -> None:
        with pytest.raises(ExactStatisticsError, match="outside the declared"):
            encode_f16_count_keys(
                t.tensor([feature_id], dtype=t.int32),
                t.tensor([1.0], dtype=t.float32),
                n_sae_features=1,
            )

    @pytest.mark.parametrize("magnitude", [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan])
    def test_rejects_nonpositive_or_nonfinite_magnitudes(self, magnitude: float) -> None:
        with pytest.raises(ExactStatisticsError, match="positive finite"):
            encode_f16_count_keys(
                t.tensor([0], dtype=t.int32),
                t.tensor([magnitude], dtype=t.float32),
                n_sae_features=1,
            )

    @pytest.mark.parametrize("magnitude", [0.1, 1.0001, 65_503.0, 70_000.0, 1e-9])
    def test_rejects_values_not_exactly_representable_as_float16(self, magnitude: float) -> None:
        with pytest.raises(ExactStatisticsError, match="non-float16-exact"):
            encode_f16_count_keys(
                t.tensor([0], dtype=t.int32),
                t.tensor([magnitude], dtype=t.float32),
                n_sae_features=1,
            )

    @pytest.mark.parametrize(
        "keys",
        [
            t.zeros((1, 1), dtype=t.int64),
            t.zeros(1, dtype=t.int32),
            t.zeros(1, dtype=t.float32),
        ],
    )
    def test_decode_rejects_invalid_tensor_contract(self, keys: t.Tensor) -> None:
        with pytest.raises(ExactStatisticsError, match="1-D int64"):
            decode_f16_count_keys(keys)

    def test_decode_rejects_negative_key(self) -> None:
        with pytest.raises(ExactStatisticsError, match="nonnegative"):
            decode_f16_count_keys(t.tensor([0, -1], dtype=t.int64))

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_exhaustive_cuda_codec_matches_cpu_bit_for_bit(self) -> None:
        bits = t.arange(
            1,
            POSITIVE_FINITE_F16_COUNT + 1,
            dtype=t.int32,
            device="cuda",
        )
        values = bits.to(t.int16).contiguous().view(t.float16).to(t.float32)
        ids = t.arange(len(bits), dtype=t.int32, device="cuda").remainder(257)

        cuda_keys = encode_f16_count_keys(ids, values, n_sae_features=257)
        decoded_ids, decoded_values = decode_f16_count_keys(cuda_keys)
        cpu_keys = encode_f16_count_keys(ids.cpu(), values.cpu(), n_sae_features=257)

        assert t.equal(cuda_keys.cpu(), cpu_keys)
        assert t.equal(decoded_ids, ids.to(t.int64))
        assert t.equal(decoded_values, values)


class TestExactMagnitudeCountsUpdates:
    def test_allocate_has_exact_shape_dtype_device_and_zero_state(self) -> None:
        required = count_table_nbytes(3)
        reducer = ExactMagnitudeCounts.allocate(3, available_bytes=required)
        assert reducer.counts.shape == (3, POSITIVE_FINITE_F16_COUNT)
        assert reducer.counts.dtype == t.int64
        assert reducer.counts.device == t.device("cpu")
        assert reducer.total_count == 0
        assert reducer.n_sae_features == 3
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.parametrize(
        "counts",
        [
            t.zeros(31_743, dtype=t.int64),
            t.zeros((0, 31_743), dtype=t.int64),
            t.zeros((1, 31_742), dtype=t.int64),
            t.zeros((1, 31_743), dtype=t.int32),
            t.zeros((1, 31_743), dtype=t.float64),
            t.zeros((31_743, 2), dtype=t.int64).transpose(0, 1),
        ],
    )
    def test_constructor_rejects_invalid_table_contract(self, counts: t.Tensor) -> None:
        with pytest.raises(ExactStatisticsError, match="counts must be a contiguous CPU int64"):
            ExactMagnitudeCounts(counts)

    @pytest.mark.parametrize("total", [-1, INT64_MAX + 1, True, 1.5])
    def test_constructor_rejects_invalid_total(self, total: object) -> None:
        with pytest.raises(ExactStatisticsError, match="total_count"):
            ExactMagnitudeCounts(
                t.zeros((1, POSITIVE_FINITE_F16_COUNT), dtype=t.int64),
                total_count=cast(Any, total),
            )

    def test_add_counts_duplicates_across_chunks_exactly(self) -> None:
        reducer = _empty_counts(4)
        ids = t.tensor([3, 0, 3, 1, 3, 0, 1, 1, 3], dtype=t.int32)
        mags = _f16([2, 1, 2, 0.5, 4, 1, 0.5, 8, 2])

        reducer.add(ids, mags, chunk_size=2)

        assert reducer.total_count == len(ids)
        keys = encode_f16_count_keys(ids, mags, n_sae_features=4)
        expected = Counter(keys.tolist())
        flat = reducer.counts.view(-1)
        assert t.count_nonzero(flat).item() == len(expected)
        for key, multiplicity in expected.items():
            assert flat[key].item() == multiplicity
        assert reducer.counts.sum().item() == len(ids)

    def test_multiple_add_calls_equal_one_concatenated_add(self) -> None:
        ids = t.tensor([0, 1, 0, 2, 1, 2, 2, 0], dtype=t.int32)
        mags = _f16([1, 2, 3, 4, 2, 6, 4, 1])
        split = _empty_counts(3)
        whole = _empty_counts(3)

        split.add(ids[:3], mags[:3], chunk_size=1)
        split.add(ids[3:6], mags[3:6], chunk_size=2)
        split.add(ids[6:], mags[6:], chunk_size=17)
        whole.add(ids, mags, chunk_size=len(ids))

        assert split.total_count == whole.total_count == len(ids)
        assert t.equal(split.counts, whole.counts)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("chunk_size", [1, 2, 5, 11, 64])
    def test_cuda_reduction_matches_cpu_for_every_chunk_boundary(
        self,
        chunk_size: int,
    ) -> None:
        ids = t.tensor(
            [3, 0, 3, 1, 3, 0, 1, 1, 3, 2, 2],
            dtype=t.int32,
        )
        mags = _positive_f16_values(
            t.tensor(
                [1, 1, 0x7BFF, 0x0400, 0x3C00, 0x03FF, 0x3C00, 1, 1, 0x7BFF, 0x0400],
                dtype=t.int32,
            )
        )
        cpu = _empty_counts(4)
        cuda = _empty_counts(4)

        cpu.add(ids, mags, chunk_size=chunk_size)
        cuda.add(
            ids,
            mags,
            chunk_size=chunk_size,
            reduction_device=t.device("cuda"),
        )

        assert cuda.total_count == cpu.total_count == len(ids)
        assert t.equal(cuda.counts, cpu.counts)
        assert ids.device.type == mags.device.type == "cpu"

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        ("ids", "mags", "match"),
        [
            (t.tensor([-1], dtype=t.int32), _f16([1]), "outside the declared"),
            (t.tensor([4], dtype=t.int32), _f16([1]), "outside the declared"),
            (t.tensor([0], dtype=t.int32), t.tensor([0.0]), "positive finite"),
            (t.tensor([0], dtype=t.int32), t.tensor([math.inf]), "positive finite"),
            (t.tensor([0], dtype=t.int32), t.tensor([0.1]), "non-float16-exact"),
            (t.tensor([0], dtype=t.int64), _f16([1]), "dtype int32"),
        ],
    )
    def test_cuda_reduction_rejects_invalid_first_chunk_before_mutation(
        self,
        ids: t.Tensor,
        mags: t.Tensor,
        match: str,
    ) -> None:
        reducer = _empty_counts(4)

        with pytest.raises(ExactStatisticsError, match=match):
            reducer.add(ids, mags, reduction_device=t.device("cuda"))

        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cpu_and_cuda_add_calls_compose_into_the_same_exact_state(self) -> None:
        ids = t.tensor([0, 1, 0, 2, 1, 2, 2, 0], dtype=t.int32)
        mags = _f16([1, 2, 3, 4, 2, 6, 4, 1])
        mixed = _empty_counts(3)
        reference = _empty_counts(3)

        mixed.add(ids[:3], mags[:3], chunk_size=2)
        mixed.add(
            ids[3:6],
            mags[3:6],
            chunk_size=1,
            reduction_device=t.device("cuda"),
        )
        mixed.add(ids[6:], mags[6:], chunk_size=17)
        reference.add(ids, mags, chunk_size=len(ids))

        assert mixed.total_count == reference.total_count == len(ids)
        assert t.equal(mixed.counts, reference.counts)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sparse_reduction_returns_owned_cpu_int64_tensors(self) -> None:
        ids = t.tensor([2, 0, 2, 1, 2, 0], dtype=t.int32)
        mags = _f16([4, 1, 4, 2, 8, 1])

        unique_keys, multiplicities = exact_statistics_module._cuda_unique_f16_counts(
            ids,
            mags,
            n_sae_features=3,
        )

        expected = Counter(encode_f16_count_keys(ids, mags, n_sae_features=3).tolist())
        assert unique_keys.device.type == "cpu"
        assert multiplicities.device.type == "cpu"
        assert unique_keys.dtype == multiplicities.dtype == t.int64
        assert unique_keys.is_contiguous()
        assert multiplicities.is_contiguous()
        assert dict(zip(unique_keys.tolist(), multiplicities.tolist(), strict=True)) == expected

    def test_empty_add_is_identity(self) -> None:
        reducer = _empty_counts(2)
        reducer.add(
            t.empty(0, dtype=t.int32),
            t.empty(0, dtype=t.float32),
            chunk_size=1,
        )
        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.parametrize(
        "reduction_device",
        ["cpu", "cuda", t.device("mps"), t.device("cuda:0"), "CPU", "", None, 1],
    )
    def test_rejects_unknown_reduction_device_before_mutation(
        self,
        reduction_device: object,
    ) -> None:
        reducer = _empty_counts(1)
        with pytest.raises(ExactStatisticsError, match="reduction_device"):
            reducer.add(
                t.tensor([0], dtype=t.int32),
                _f16([1]),
                reduction_device=cast(Any, reduction_device),
            )
        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    def test_rejects_unavailable_cuda_before_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reducer = _empty_counts(1)
        monkeypatch.setattr(exact_statistics_module.t.cuda, "is_available", lambda: False)

        with pytest.raises(ExactStatisticsError, match="available CUDA"):
            reducer.add(
                t.tensor([0], dtype=t.int32),
                _f16([1]),
                reduction_device=t.device("cuda"),
            )

        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.parametrize(
        ("ids", "mags", "match"),
        [
            (t.empty(0, dtype=t.int64), t.empty(0), "dtype int32"),
            (t.empty(0, dtype=t.int16), t.empty(0), "dtype int32"),
            (t.empty(0, dtype=t.int32), t.empty(0, dtype=t.float64), "dtype float32"),
            (t.empty((0, 1), dtype=t.int32), t.empty((0, 1)), "must both be 1-D"),
        ],
    )
    def test_empty_add_still_validates_tensor_contract(
        self,
        ids: t.Tensor,
        mags: t.Tensor,
        match: str,
    ) -> None:
        reducer = _empty_counts(1)
        with pytest.raises(ExactStatisticsError, match=match):
            reducer.add(ids, mags)
        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.parametrize("chunk_size", [0, -1, -100])
    def test_rejects_nonpositive_chunk_size(self, chunk_size: int) -> None:
        with pytest.raises(ExactStatisticsError, match="must be positive"):
            _empty_counts(1).add(
                t.empty(0, dtype=t.int32),
                t.empty(0, dtype=t.float32),
                chunk_size=chunk_size,
            )

    @pytest.mark.parametrize("chunk_size", [True, 1.5, "1", None])
    def test_rejects_noninteger_chunk_size(self, chunk_size: object) -> None:
        with pytest.raises(TypeError, match="chunk_size must be an int"):
            _empty_counts(1).add(
                t.empty(0, dtype=t.int32),
                t.empty(0, dtype=t.float32),
                chunk_size=cast(Any, chunk_size),
            )

    def test_rejects_int64_total_overflow_before_mutation(self) -> None:
        reducer = _empty_counts(1, total_count=INT64_MAX)
        with pytest.raises(ExactStatisticsError, match="overflow"):
            reducer.add(t.tensor([0], dtype=t.int32), _f16([1]))
        assert reducer.total_count == INT64_MAX
        assert t.count_nonzero(reducer.counts).item() == 0

    def test_cuda_overflow_fails_before_launching_reduction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reducer = _empty_counts(1, total_count=INT64_MAX)
        monkeypatch.setattr(exact_statistics_module.t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            exact_statistics_module,
            "_cuda_unique_f16_counts",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CUDA reduction launched")
            ),
        )

        with pytest.raises(ExactStatisticsError, match="overflow"):
            reducer.add(
                t.tensor([0], dtype=t.int32),
                _f16([1]),
                reduction_device=t.device("cuda"),
            )

        assert reducer.total_count == INT64_MAX
        assert t.count_nonzero(reducer.counts).item() == 0

    def test_cuda_out_of_memory_propagates_without_cpu_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reducer = _empty_counts(1)
        monkeypatch.setattr(exact_statistics_module.t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            exact_statistics_module,
            "_cuda_unique_f16_counts",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                t.cuda.OutOfMemoryError("deliberate CUDA OOM")
            ),
        )

        with pytest.raises(t.cuda.OutOfMemoryError, match="deliberate CUDA OOM"):
            reducer.add(
                t.tensor([0], dtype=t.int32),
                _f16([1]),
                reduction_device=t.device("cuda"),
            )

        assert reducer.total_count == 0
        assert t.count_nonzero(reducer.counts).item() == 0

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_rejects_cuda_shard_tensors_before_update(self) -> None:
        reducer = _empty_counts(1)
        ids = t.tensor([0], dtype=t.int32, device="cuda")
        mags = t.tensor([1.0], dtype=t.float32, device="cuda")
        with pytest.raises(ExactStatisticsError, match="requires CPU"):
            reducer.add(ids, mags)
        assert reducer.total_count == 0


class TestSharedHistogramEdges:
    def test_empty_sample_preserves_legacy_default(self) -> None:
        edges = shared_histogram_edges(
            t.empty(0, dtype=t.float32),
            global_mag_min=0.0,
            n_hist_bins=4,
        )
        assert t.equal(edges, t.linspace(0.0, 1.0, 5, dtype=t.float64))

    def test_uses_legacy_floor_kth_rank_not_torch_quantile(self) -> None:
        sample = t.arange(1, 102, dtype=t.float32)
        edges = shared_histogram_edges(
            sample,
            global_mag_min=0.5,
            n_hist_bins=2,
        )
        assert edges.tolist() == [0.5, 49.75, 99.0]
        assert t.quantile(sample.to(t.float64), 0.99).item() == 100.0

    def test_constant_sample_forces_positive_range(self) -> None:
        sample = t.full((100,), 2.0, dtype=t.float32)
        edges = shared_histogram_edges(
            sample,
            global_mag_min=2.0,
            n_hist_bins=2,
        )
        assert edges[0].item() == 2.0
        assert edges[-1].item() == 2.000001
        assert bool((edges[1:] > edges[:-1]).all().item())

    @pytest.mark.parametrize("minimum", [0.0, -1.0, math.inf, -math.inf, math.nan])
    def test_nonempty_sample_requires_positive_finite_minimum(self, minimum: float) -> None:
        with pytest.raises(ExactStatisticsError, match="positive and finite"):
            shared_histogram_edges(
                t.tensor([1.0]),
                global_mag_min=minimum,
            )

    @pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
    def test_rejects_nonfinite_sample(self, invalid: float) -> None:
        with pytest.raises(ExactStatisticsError, match="non-finite"):
            shared_histogram_edges(
                t.tensor([1.0, invalid]),
                global_mag_min=1.0,
            )

    def test_rejects_nonvector_sample(self) -> None:
        with pytest.raises(ExactStatisticsError, match="must be 1-D"):
            shared_histogram_edges(t.ones((1, 1)), global_mag_min=1.0)

    @pytest.mark.parametrize("n_bins", [0, -1])
    def test_rejects_nonpositive_bin_count(self, n_bins: int) -> None:
        with pytest.raises(ExactStatisticsError, match="must be positive"):
            shared_histogram_edges(
                t.ones(1),
                global_mag_min=1.0,
                n_hist_bins=n_bins,
            )

    @pytest.mark.parametrize("n_bins", [True, 1.5, "50", None])
    def test_rejects_noninteger_bin_count(self, n_bins: object) -> None:
        with pytest.raises(TypeError, match="n_hist_bins must be an int"):
            shared_histogram_edges(
                t.ones(1),
                global_mag_min=1.0,
                n_hist_bins=cast(Any, n_bins),
            )

    @pytest.mark.parametrize(
        "sample",
        [
            t.ones(1, dtype=t.float16),
            t.ones(1, dtype=t.float64),
            t.ones(1, dtype=t.int64),
        ],
    )
    def test_rejects_non_float32_sample(self, sample: t.Tensor) -> None:
        with pytest.raises(ExactStatisticsError, match="CPU float32"):
            shared_histogram_edges(sample, global_mag_min=1.0)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_rejects_cuda_sample(self) -> None:
        with pytest.raises(ExactStatisticsError, match="CPU float32"):
            shared_histogram_edges(
                t.ones(1, dtype=t.float32, device="cuda"),
                global_mag_min=1.0,
            )


class TestExactMagnitudeFinalization:
    @pytest.mark.parametrize("n_hist_bins", [1, 17])
    def test_full_float16_domain_with_weighted_and_inactive_features(
        self, n_hist_bins: int
    ) -> None:
        values = _positive_f16_values(t.arange(1, POSITIVE_FINITE_F16_COUNT + 1))
        ids = t.cat((t.zeros(len(values), dtype=t.int32), t.full((5,), 2, dtype=t.int32)))
        mags = t.cat((values, _f16([0.5, 1, 1, 5, 65504])))
        occurrence = t.bincount(ids.to(t.int64), minlength=4)
        reducer = _empty_counts(4)
        reducer.add(ids, mags)
        sample = t.empty(0, dtype=t.float32)
        percentiles = (100, 0, 1, 50, 99, 50)
        watched = (reducer.counts, occurrence, sample)
        copies = tuple(value.clone() for value in watched)
        versions = tuple(value._version for value in watched)
        rng = t.get_rng_state().clone()

        actual = reducer.finalize(
            occurrence=occurrence, mag_subsample=sample, global_mag_min=0.0,
            percentiles=percentiles, n_hist_bins=n_hist_bins,
        )
        expected = _sort_reference(
            ids, mags, occurrence, sample, 0.0,
            percentiles=percentiles, n_hist_bins=n_hist_bins,
        )

        assert actual == expected
        assert set(actual[0]) == {0, 2}
        assert all(t.equal(value, copy) for value, copy in zip(watched, copies, strict=True))
        assert tuple(value._version for value in watched) == versions
        assert t.equal(t.get_rng_state(), rng)

    def test_histogram_bins_are_recomputed_for_each_finalize_call(self) -> None:
        ids = t.tensor([0, 0, 0, 0, 0, 0, 0, 2, 2, 2], dtype=t.int32)
        mags = _f16([0.5, 1, 2, 3, 4, 5, 65504, 1, 1, 65504])
        occurrence = t.bincount(ids.to(t.int64), minlength=4)
        reducer = _empty_counts(4)
        reducer.add(ids, mags)
        counts_before = reducer.counts.clone()
        occurrence_before = occurrence.clone()
        versions = (reducer.counts._version, occurrence._version)
        rng = t.get_rng_state().clone()
        histograms = []

        for sample, minimum in (
            (_f16([1] + [5] * 100), 1.0),
            (t.empty(0, dtype=t.float32), 0.0),
        ):
            sample_before = sample.clone()
            sample_version = sample._version
            actual = reducer.finalize(
                occurrence=occurrence, mag_subsample=sample, global_mag_min=minimum,
                percentiles=(25, 50, 75), n_hist_bins=4,
            )
            expected = _sort_reference(
                ids, mags, occurrence, sample, minimum,
                percentiles=(25, 50, 75), n_hist_bins=4,
            )
            assert actual == expected
            assert t.equal(sample, sample_before) and sample._version == sample_version
            histograms.append(actual[0][0].histogram)

        assert histograms[0] != histograms[1]
        assert t.equal(reducer.counts, counts_before)
        assert t.equal(occurrence, occurrence_before)
        assert (reducer.counts._version, occurrence._version) == versions
        assert t.equal(t.get_rng_state(), rng)

    def test_hand_computed_percentiles_median_and_histogram(self) -> None:
        ids = t.tensor([0, 0, 0, 0, 1, 1, 1], dtype=t.int32)
        mags = _f16([1, 1, 3, 9, 2, 4, 8])
        occurrence = t.bincount(ids.to(t.int64), minlength=2)
        reducer = _empty_counts(2)
        reducer.add(ids, mags, chunk_size=3)

        result, edges = reducer.finalize(
            occurrence=occurrence,
            mag_subsample=mags,
            global_mag_min=1.0,
            percentiles=(0, 25, 50, 75, 100),
            n_hist_bins=4,
        )

        assert result[0].percentiles == {
            "p0": 1.0,
            "p25": 1.0,
            "p50": 2.0,
            "p75": 4.5,
            "p100": 9.0,
        }
        assert result[0].median == 2.0
        assert sum(result[0].histogram) == 4
        assert result[1].percentiles == {
            "p0": 2.0,
            "p25": 3.0,
            "p50": 4.0,
            "p75": 6.0,
            "p100": 8.0,
        }
        assert result[1].median == 4.0
        assert sum(result[1].histogram) == 3
        assert edges is not None
        assert len(edges) == 5

    def test_exact_internal_edge_values_follow_bucketize_convention(self) -> None:
        ids = t.zeros(5, dtype=t.int32)
        mags = _f16([1, 2, 3, 4, 5])
        reducer = _empty_counts(1)
        reducer.add(ids, mags)

        result, edges = reducer.finalize(
            occurrence=t.tensor([5], dtype=t.int64),
            mag_subsample=_f16([1] + [5] * 100),
            global_mag_min=1.0,
            percentiles=(50,),
            n_hist_bins=4,
        )

        assert edges == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert result[0].histogram == [2, 1, 1, 1]

    def test_empty_table_returns_empty_detail_and_no_edges(self) -> None:
        result, edges = _empty_counts(2).finalize(
            occurrence=t.zeros(2, dtype=t.int64),
            mag_subsample=t.empty(0, dtype=t.float32),
            global_mag_min=0.0,
            percentiles=PERCENTILES,
        )
        assert result == {}
        assert edges is None

    def test_detects_occurrence_total_mismatch(self) -> None:
        reducer = _empty_counts(1)
        reducer.add(t.tensor([0], dtype=t.int32), _f16([1]))
        with pytest.raises(ExactStatisticsError, match="occurrence total 2"):
            reducer.finalize(
                occurrence=t.tensor([2], dtype=t.int64),
                mag_subsample=_f16([1]),
                global_mag_min=1.0,
                percentiles=(50,),
            )

    def test_detects_per_feature_count_mismatch_even_when_total_matches(self) -> None:
        reducer = _empty_counts(2)
        reducer.add(t.tensor([0, 1], dtype=t.int32), _f16([1, 2]))
        with pytest.raises(ExactStatisticsError, match="feature 1"):
            reducer.finalize(
                occurrence=t.tensor([0, 2], dtype=t.int64),
                mag_subsample=_f16([1, 2]),
                global_mag_min=1.0,
                percentiles=(50,),
            )

    def test_detects_negative_table_value_during_integrity_audit(self) -> None:
        reducer = _empty_counts(1)
        reducer.counts[0, 0] = -1
        with pytest.raises(ExactStatisticsError, match="negative values"):
            reducer.finalize(
                occurrence=t.zeros(1, dtype=t.int64),
                mag_subsample=t.empty(0, dtype=t.float32),
                global_mag_min=0.0,
                percentiles=(50,),
            )

    def test_integrity_audit_checks_every_bounded_row_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(exact_statistics_module, "COUNT_AUDIT_ROW_CHUNK_SIZE", 1)
        reducer = _empty_counts(3)
        reducer.counts[2, -1] = -1

        with pytest.raises(ExactStatisticsError, match="negative values"):
            reducer.finalize(
                occurrence=t.zeros(3, dtype=t.int64),
                mag_subsample=t.empty(0, dtype=t.float32),
                global_mag_min=0.0,
                percentiles=(50,),
            )

    def test_detects_tracked_total_mismatch_during_integrity_audit(self) -> None:
        reducer = _empty_counts(1, total_count=1)
        with pytest.raises(ExactStatisticsError, match="count-table sum 0"):
            reducer.finalize(
                occurrence=t.tensor([1], dtype=t.int64),
                mag_subsample=_f16([1]),
                global_mag_min=1.0,
                percentiles=(50,),
            )

    def test_positive_occurrence_with_empty_row_has_typed_error(self) -> None:
        reducer = _empty_counts(2, total_count=1)
        reducer.counts[1, 0] = 1
        with pytest.raises(ExactStatisticsError, match="feature 0.*row is empty"):
            reducer.finalize(
                occurrence=t.tensor([1, 0], dtype=t.int64),
                mag_subsample=_f16([1]),
                global_mag_min=1.0,
                percentiles=(50,),
            )

    @pytest.mark.parametrize(
        "occurrence",
        [
            t.zeros((1, 1), dtype=t.int64),
            t.zeros(1, dtype=t.int32),
            t.zeros(1, dtype=t.float64),
            t.zeros(2, dtype=t.int64),
        ],
    )
    def test_rejects_invalid_occurrence_contract(self, occurrence: t.Tensor) -> None:
        with pytest.raises(ExactStatisticsError, match="occurrence"):
            _empty_counts(1).finalize(
                occurrence=occurrence,
                mag_subsample=t.empty(0),
                global_mag_min=0.0,
                percentiles=(50,),
            )

    def test_rejects_negative_occurrence(self) -> None:
        with pytest.raises(ExactStatisticsError, match="nonnegative"):
            _empty_counts(1).finalize(
                occurrence=t.tensor([-1], dtype=t.int64),
                mag_subsample=t.empty(0),
                global_mag_min=0.0,
                percentiles=(50,),
            )

    @pytest.mark.parametrize("percentiles", [(-1,), (101,), (50.0,), (True,)])
    def test_rejects_invalid_percentiles(self, percentiles: tuple[object, ...]) -> None:
        with pytest.raises(ExactStatisticsError, match="percentiles"):
            _empty_counts(1).finalize(
                occurrence=t.zeros(1, dtype=t.int64),
                mag_subsample=t.empty(0),
                global_mag_min=0.0,
                percentiles=cast(Any, percentiles),
            )

    @pytest.mark.parametrize("n_bins", [True, 1.5, "50", None])
    def test_finalize_rejects_noninteger_histogram_bin_count(
        self,
        n_bins: object,
    ) -> None:
        with pytest.raises(TypeError, match="n_hist_bins must be an int"):
            _empty_counts(1).finalize(
                occurrence=t.zeros(1, dtype=t.int64),
                mag_subsample=t.empty(0),
                global_mag_min=0.0,
                percentiles=(50,),
                n_hist_bins=cast(Any, n_bins),
            )

    def test_randomized_tie_heavy_results_equal_sort_reference(self) -> None:
        rng = random.Random(20260811)
        value_domain = _positive_f16_values(
            t.tensor([1, 2, 3, 17, 255, 1024, 12_345, 24_000, 31_743])
        )

        for case_index in range(100):
            n_features = rng.randint(1, 5)
            n_activations = rng.randint(1, 500)
            generator = t.Generator().manual_seed(10_000 + case_index)
            ids = t.randint(
                0,
                n_features,
                (n_activations,),
                dtype=t.int32,
                generator=generator,
            )
            choices = t.randint(
                0,
                len(value_domain),
                (n_activations,),
                generator=generator,
            )
            mags = value_domain[choices]
            permutation = t.randperm(n_activations, generator=generator)
            ids = ids[permutation]
            mags = mags[permutation]
            occurrence = t.bincount(ids.to(t.int64), minlength=n_features)
            sample_step = rng.randint(1, 9)
            sample = mags[::sample_step].clone()
            minimum = float(mags.min().item())

            reducer = _empty_counts(n_features)
            reducer.add(ids, mags, chunk_size=rng.randint(1, 47))
            actual = reducer.finalize(
                occurrence=occurrence,
                mag_subsample=sample,
                global_mag_min=minimum,
                percentiles=PERCENTILES,
            )
            expected = _sort_reference(
                ids,
                mags,
                occurrence,
                sample,
                minimum,
            )

            assert actual == expected, f"randomized parity case {case_index}"


class TestWeightedDistribution:
    @pytest.mark.parametrize("population", [-1, INT64_MAX + 1, True, 1.5, "1"])
    def test_constructor_rejects_invalid_population_size(self, population: object) -> None:
        with pytest.raises(ExactStatisticsError, match="population_size"):
            WeightedDistribution(population_size=cast(Any, population))

    @pytest.mark.parametrize(
        "counts",
        [
            Counter({True: 1}),
            Counter({"1": 1}),
            Counter({None: 1}),
            Counter({math.inf: 1}),
            Counter({-math.inf: 1}),
            Counter({math.nan: 1}),
        ],
    )
    def test_constructor_rejects_invalid_counter_keys(self, counts: Counter) -> None:
        with pytest.raises(ExactStatisticsError, match="key must"):
            WeightedDistribution(counts=cast(Any, counts), population_size=1)

    @pytest.mark.parametrize(
        "counts",
        [
            Counter({1: 0}),
            Counter({1: -1}),
            Counter({1: True}),
            Counter({1: 1.5}),
            Counter({1: "1"}),
        ],
    )
    def test_constructor_rejects_invalid_multiplicities(self, counts: Counter) -> None:
        with pytest.raises(ExactStatisticsError, match="multiplicities"):
            WeightedDistribution(counts=cast(Any, counts), population_size=1)

    def test_constructor_rejects_counter_population_mismatch(self) -> None:
        with pytest.raises(ExactStatisticsError, match="does not match"):
            WeightedDistribution(counts=Counter({1: 2}), population_size=3)

    def test_add_tracks_exact_multiplicities_and_population(self) -> None:
        distribution = WeightedDistribution()
        distribution.add([3, 1, 3, 2, 3, 1])
        assert distribution.counts == Counter({3: 3, 1: 2, 2: 1})
        assert distribution.population_size == 6
        assert distribution.minimum() == 1
        assert distribution.maximum() == 3

    def test_add_weighted_merges_with_unweighted_add(self) -> None:
        distribution = WeightedDistribution()
        distribution.add([0.5, 1.0])
        distribution.add_weighted(0.5, 3)
        distribution.add_weighted(2.0, 2)
        distribution.add_weighted(99.0, 0)
        assert distribution.counts == Counter({0.5: 4, 2.0: 2, 1.0: 1})
        assert distribution.population_size == 7

    def test_histogram_preserves_legacy_float_count_scalar_type(self) -> None:
        distribution = WeightedDistribution()
        distribution.add([1, 1, 2, 3])

        histogram = distribution.histogram(n_bins=3)

        assert histogram is not None
        assert histogram.counts == [2.0, 1.0, 1.0]
        assert all(type(count) is float for count in histogram.counts)

    @pytest.mark.parametrize("value", [True, "1", None, [], object()])
    def test_add_rejects_non_numeric_values(self, value: object) -> None:
        distribution = WeightedDistribution()
        with pytest.raises(ExactStatisticsError, match="must be int or float"):
            distribution.add([1, cast(Any, value)])
        assert distribution.population_size == 0
        assert distribution.counts == Counter()

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_add_rejects_nonfinite_values_atomically(self, value: float) -> None:
        distribution = WeightedDistribution()
        with pytest.raises(ExactStatisticsError, match="must be finite"):
            distribution.add([1.0, value, 2.0])
        assert distribution.population_size == 0
        assert distribution.counts == Counter()

    @pytest.mark.parametrize("multiplicity", [-1, -100])
    def test_add_weighted_rejects_negative_multiplicity(self, multiplicity: int) -> None:
        with pytest.raises(ExactStatisticsError, match="nonnegative"):
            WeightedDistribution().add_weighted(1, multiplicity)

    @pytest.mark.parametrize("multiplicity", [True, 1.5, "2", None])
    def test_add_weighted_rejects_noninteger_multiplicity(self, multiplicity: object) -> None:
        with pytest.raises(TypeError, match="multiplicity must be an int"):
            WeightedDistribution().add_weighted(1, cast(Any, multiplicity))

    def test_population_overflow_is_rejected_before_mutation(self) -> None:
        distribution = WeightedDistribution(
            counts=Counter({1: INT64_MAX}), population_size=INT64_MAX
        )
        with pytest.raises(ExactStatisticsError, match="overflow"):
            distribution.add_weighted(2, 1)
        assert distribution.counts == Counter({1: INT64_MAX})
        assert distribution.population_size == INT64_MAX

    @pytest.mark.parametrize("method", ["minimum", "maximum"])
    def test_extrema_are_undefined_for_empty_distribution(self, method: str) -> None:
        with pytest.raises(ExactStatisticsError, match="undefined"):
            getattr(WeightedDistribution(), method)()

    def test_nearest_rank_matches_expanded_reference_at_every_percentile(self) -> None:
        values = [9, 1, 1, 3, 3, 3, 7, 9, 9, 9]
        distribution = WeightedDistribution()
        distribution.add(values)
        percentiles = tuple(range(101))
        assert distribution.percentile_dict(percentiles) == _expanded_nearest_rank(
            values,
            percentiles,
        )
        assert distribution.percentile_dict(
            percentiles,
            as_int=True,
        ) == _expanded_nearest_rank(values, percentiles, as_int=True)

    def test_rounded_nearest_rank_matches_expanded_float_reference(self) -> None:
        values = [0.1, 0.1, 1 / 3, 0.75, 0.75, 10 / 3]
        distribution = WeightedDistribution()
        distribution.add(values)
        percentiles = (1, 5, 10, 25, 50, 75, 90, 95, 99)
        assert distribution.percentile_dict(
            percentiles,
            round_digits=4,
        ) == _expanded_nearest_rank(
            values,
            percentiles,
            round_digits=4,
        )

    def test_empty_percentiles_and_histogram_match_guarded_legacy_calls(self) -> None:
        distribution = WeightedDistribution()
        assert distribution.percentile_dict((1, 50, 99)) == {}
        assert distribution.histogram() is None

    @pytest.mark.parametrize("percentiles", [(-1,), (101,), (50.0,), (True,)])
    def test_rejects_invalid_percentiles(self, percentiles: tuple[object, ...]) -> None:
        distribution = WeightedDistribution()
        distribution.add([1])
        with pytest.raises(ExactStatisticsError, match="percentiles"):
            distribution.percentile_dict(cast(Any, percentiles))

    @pytest.mark.parametrize("n_bins", [0, -1])
    def test_histogram_rejects_nonpositive_bin_count(self, n_bins: int) -> None:
        distribution = WeightedDistribution()
        distribution.add([1])
        with pytest.raises(ExactStatisticsError, match="must be positive"):
            distribution.histogram(n_bins=n_bins)

    @pytest.mark.parametrize("n_bins", [True, 1.5, "50", None])
    def test_histogram_rejects_noninteger_bin_count(self, n_bins: object) -> None:
        distribution = WeightedDistribution()
        distribution.add([1])
        with pytest.raises(TypeError, match="n_bins must be an int"):
            distribution.histogram(n_bins=cast(Any, n_bins))

    def test_weighted_histogram_matches_expanded_torch_histogram_exactly(self) -> None:
        values = [-3.0, -3.0, -1.0, 0.0, 0.0, 0.0, 2.5, 9.0, 9.0]
        distribution = WeightedDistribution()
        distribution.add(values)

        actual = distribution.histogram(n_bins=7)
        expected_counts, expected_edges = t.histogram(
            t.tensor(values, dtype=t.float64),
            bins=7,
        )

        assert actual is not None
        assert actual.counts == expected_counts.tolist()
        assert actual.bin_edges == [round(value, 4) for value in expected_edges.tolist()]

    def test_constant_value_histogram_matches_torch_range_expansion(self) -> None:
        distribution = WeightedDistribution()
        distribution.add_weighted(7, 13)
        expected_counts, expected_edges = t.histogram(
            t.full((13,), 7.0, dtype=t.float64),
            bins=5,
        )
        actual = distribution.histogram(n_bins=5)
        assert actual is not None
        assert actual.counts == expected_counts.tolist()
        assert actual.bin_edges == [round(value, 4) for value in expected_edges.tolist()]

    def test_histogram_rejects_population_above_float64_exact_integer_limit(self) -> None:
        distribution = WeightedDistribution(
            counts=Counter({1: MAX_EXACT_FLOAT64_INTEGER + 1}),
            population_size=MAX_EXACT_FLOAT64_INTEGER + 1,
        )
        with pytest.raises(ExactStatisticsError, match=r"above 2\*\*53"):
            distribution.histogram()

    def test_counter_population_corruption_is_detected(self) -> None:
        with pytest.raises(ExactStatisticsError, match="does not match"):
            WeightedDistribution(counts=Counter({1: 2}), population_size=3)

    def test_exact_integer_sum(self) -> None:
        distribution = WeightedDistribution()
        distribution.add_weighted(3, 5)
        distribution.add_weighted(-2, 4)
        assert distribution.exact_integer_sum() == 7

    def test_exact_integer_sum_rejects_float_keys_even_if_integral(self) -> None:
        distribution = WeightedDistribution()
        distribution.add_weighted(3.0, 5)
        with pytest.raises(ExactStatisticsError, match="requires integer"):
            distribution.exact_integer_sum()

    def test_randomized_percentiles_histograms_and_sums_match_expansion(self) -> None:
        rng = random.Random(917_503)
        percentiles = tuple(range(101))
        for case_index in range(500):
            domain = sorted({rng.randint(-50, 50) for _ in range(rng.randint(1, 15))})
            values: list[int] = []
            distribution = WeightedDistribution()
            for value in domain:
                multiplicity = rng.randint(1, 30)
                distribution.add_weighted(value, multiplicity)
                values.extend([value] * multiplicity)

            assert distribution.population_size == len(values), case_index
            assert distribution.exact_integer_sum() == sum(values), case_index
            assert distribution.minimum() == min(values), case_index
            assert distribution.maximum() == max(values), case_index
            assert distribution.percentile_dict(
                percentiles,
                as_int=True,
            ) == _expanded_nearest_rank(values, percentiles, as_int=True), case_index

            n_bins = rng.randint(1, 50)
            actual_histogram = distribution.histogram(n_bins=n_bins)
            expected_counts, expected_edges = t.histogram(
                t.tensor(values, dtype=t.float64),
                bins=n_bins,
            )
            assert actual_histogram is not None
            assert actual_histogram.counts == expected_counts.tolist(), case_index
            assert actual_histogram.bin_edges == [
                round(value, 4) for value in expected_edges.tolist()
            ], case_index


class TestDirectCountUpdates:
    @pytest.mark.parametrize("chunk_size", [1, 3, 19])
    def test_preloaded_cells_keep_int64_modular_updates(self, chunk_size: int) -> None:
        reducer = _empty_counts(2)
        reducer.counts[0, 0] = -(2**63)
        reducer.counts[1, 0x3C00 - 1] = INT64_MAX - 1
        expected = reducer.counts.clone()
        expected[0, 0] = -(2**63) + 1
        expected[1, 0x3C00 - 1] = -(2**63) + 1
        ids = t.tensor([1, 1, 0, 1], dtype=t.int32)
        mags = _positive_f16_values(t.tensor([0x3C00, 0x3C00, 1, 0x3C00]))
        copies = (ids.clone(), mags.clone())
        versions = (ids._version, mags._version)
        rng = t.get_rng_state().clone()

        reducer.add(ids, mags, chunk_size=chunk_size)

        assert t.equal(reducer.counts, expected)
        assert reducer.total_count == 4
        assert t.equal(ids, copies[0]) and t.equal(mags, copies[1])
        assert (ids._version, mags._version) == versions
        assert t.equal(t.get_rng_state(), rng)

    @pytest.mark.parametrize("chunk_size", [1, 2])
    def test_invalid_later_chunk_preserves_only_completed_updates(self, chunk_size: int) -> None:
        reducer = _empty_counts(2)
        ids = t.tensor([0, 0, 2], dtype=t.int32)
        mags = _f16([1, 1, 1])
        expected = reducer.counts.clone()
        expected[0, 0x3C00 - 1] = 2
        copies = (ids.clone(), mags.clone())
        versions = (ids._version, mags._version)
        rng = t.get_rng_state().clone()

        with pytest.raises(ExactStatisticsError, match="outside the declared SAE dictionary"):
            reducer.add(ids, mags, chunk_size=chunk_size)

        assert t.equal(reducer.counts, expected)
        assert reducer.total_count == 0
        assert t.equal(ids, copies[0]) and t.equal(mags, copies[1])
        assert (ids._version, mags._version) == versions
        assert t.equal(t.get_rng_state(), rng)

    @pytest.mark.parametrize("chunk_size", [1, 2])
    def test_aliased_magnitudes_are_validated_at_each_chunk(self, chunk_size: int) -> None:
        reducer = _empty_counts(1)
        # The first magnitude is float16 bit pattern 5. Its count-cell update
        # changes the second aliased magnitude from 1 to a non-float16-exact value.
        reducer.counts[0, 3] = 0x34A00000
        reducer.counts[0, 4] = 0x3F800000
        mags = reducer.counts.view(t.float32).flatten()[6:10:2]
        ids = t.tensor([0, 0], dtype=t.int32)
        expected = reducer.counts.clone()
        expected[0, 4] = 0x3F800001
        if chunk_size == 2:
            expected[0, 0x3C00 - 1] = 1
            reducer.add(ids, mags, chunk_size=chunk_size)
        else:
            with pytest.raises(ExactStatisticsError, match="non-float16-exact input"):
                reducer.add(ids, mags, chunk_size=chunk_size)

        assert t.equal(reducer.counts, expected)
        assert reducer.total_count == (2 if chunk_size == 2 else 0)
        assert mags.view(t.int32).tolist() == [0x34A00000, 0x3F800001]
        assert ids.tolist() == [0, 0]


class TestReducerAlgebra:
    """Metamorphic proofs that partitioning and ordering cannot change state."""

    def _population(self) -> tuple[t.Tensor, t.Tensor]:
        generator = t.Generator().manual_seed(88_021)
        domain = _positive_f16_values(
            t.tensor([1, 2, 7, 31, 255, 1024, 4096, 12_345, 24_000, 31_743])
        )
        feature_ids = t.randint(0, 8, (2_000,), dtype=t.int32, generator=generator)
        choices = t.randint(0, len(domain), (2_000,), generator=generator)
        return feature_ids, domain[choices]

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 31, 127, 2_000, 10_000])
    def test_count_table_is_invariant_to_chunk_size(self, chunk_size: int) -> None:
        feature_ids, feature_mags = self._population()
        reference = _empty_counts(8)
        actual = _empty_counts(8)
        reference.add(feature_ids, feature_mags, chunk_size=len(feature_ids))
        actual.add(feature_ids, feature_mags, chunk_size=chunk_size)
        assert actual.total_count == reference.total_count
        assert t.equal(actual.counts, reference.counts)

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_count_table_is_invariant_to_activation_order(self, seed: int) -> None:
        feature_ids, feature_mags = self._population()
        reference = _empty_counts(8)
        reference.add(feature_ids, feature_mags, chunk_size=37)
        permutation = t.randperm(
            len(feature_ids),
            generator=t.Generator().manual_seed(seed),
        )
        actual = _empty_counts(8)
        actual.add(
            feature_ids[permutation],
            feature_mags[permutation],
            chunk_size=53,
        )
        assert actual.total_count == reference.total_count
        assert t.equal(actual.counts, reference.counts)

    @pytest.mark.parametrize("split_count", [1, 2, 3, 7, 19, 100])
    def test_count_table_is_invariant_to_shard_partitioning(
        self,
        split_count: int,
    ) -> None:
        feature_ids, feature_mags = self._population()
        reference = _empty_counts(8)
        partitioned = _empty_counts(8)
        reference.add(feature_ids, feature_mags, chunk_size=61)
        boundaries = t.linspace(0, len(feature_ids), split_count + 1).to(t.int64)
        for start, end in itertools.pairwise(boundaries):
            partitioned.add(
                feature_ids[int(start.item()) : int(end.item())],
                feature_mags[int(start.item()) : int(end.item())],
                chunk_size=17,
            )
        assert partitioned.total_count == reference.total_count
        assert t.equal(partitioned.counts, reference.counts)

    @pytest.mark.parametrize(
        "percentiles",
        [
            (0,),
            (100,),
            (0, 100),
            (1, 50, 99),
            tuple(range(0, 101, 5)),
            tuple(reversed(range(1, 100))),
        ],
    )
    def test_finalization_matches_sort_for_arbitrary_percentile_requests(
        self,
        percentiles: tuple[int, ...],
    ) -> None:
        feature_ids, feature_mags = self._population()
        occurrence = t.bincount(feature_ids.to(t.int64), minlength=8)
        reducer = _empty_counts(8)
        reducer.add(feature_ids, feature_mags, chunk_size=29)
        sample = feature_mags[::11].clone()
        minimum = float(feature_mags.min().item())
        actual = reducer.finalize(
            occurrence=occurrence,
            mag_subsample=sample,
            global_mag_min=minimum,
            percentiles=percentiles,
            n_hist_bins=17,
        )
        expected = _sort_reference(
            feature_ids,
            feature_mags,
            occurrence,
            sample,
            minimum,
            percentiles=percentiles,
            n_hist_bins=17,
        )
        assert actual == expected

    @pytest.mark.parametrize("n_unused", [1, 10, 100])
    def test_unused_feature_rows_do_not_change_active_results(self, n_unused: int) -> None:
        feature_ids, feature_mags = self._population()
        occurrence = t.bincount(feature_ids.to(t.int64), minlength=8)
        narrow = _empty_counts(8)
        wide = _empty_counts(8 + n_unused)
        narrow.add(feature_ids, feature_mags)
        wide.add(feature_ids, feature_mags)
        sample = feature_mags[::13]
        minimum = float(feature_mags.min().item())
        narrow_result = narrow.finalize(
            occurrence=occurrence,
            mag_subsample=sample,
            global_mag_min=minimum,
            percentiles=PERCENTILES,
        )
        wide_result = wide.finalize(
            occurrence=t.cat([occurrence, t.zeros(n_unused, dtype=t.int64)]),
            mag_subsample=sample,
            global_mag_min=minimum,
            percentiles=PERCENTILES,
        )
        assert wide_result == narrow_result

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_weighted_distribution_is_invariant_to_partition_and_merge(
        self,
        seed: int,
    ) -> None:
        rng = random.Random(seed)
        values = [rng.randint(-20, 20) for _ in range(1_000)]
        direct = WeightedDistribution()
        direct.add(values)

        left = WeightedDistribution()
        right = WeightedDistribution()
        split = rng.randint(0, len(values))
        left.add(values[:split])
        right.add(values[split:])
        merged = WeightedDistribution()
        for value, multiplicity in left.counts.items():
            merged.add_weighted(value, multiplicity)
        for value, multiplicity in right.counts.items():
            merged.add_weighted(value, multiplicity)

        assert merged.counts == direct.counts
        assert merged.population_size == direct.population_size
        assert merged.exact_integer_sum() == direct.exact_integer_sum()
        assert merged.percentile_dict(tuple(range(101)), as_int=True) == (
            direct.percentile_dict(tuple(range(101)), as_int=True)
        )
        for n_bins in (1, 2, 3, 7, 50):
            assert merged.histogram(n_bins=n_bins) == direct.histogram(n_bins=n_bins)
