"""Exact bounded-state reducers for feature statistics.

Production feature shards persist magnitudes as float16 and load them as
float32.  The positive finite float16 domain is finite and totally ordered by
its raw bit pattern, so an integer count table over ``(feature_id, bits)`` is
an exact replacement for retaining and sorting every activation magnitude.

This module is intentionally a leaf: it knows nothing about Hydra, feature
paths, shard transport, or statistics serialization.  That keeps the
numerical contract independently testable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import torch as t
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from steering.stats_schema import HistogramData

POSITIVE_FINITE_F16_COUNT = 0x7C00 - 1
DEFAULT_COUNT_CHUNK_SIZE = 16_000_000
COUNT_AUDIT_ROW_CHUNK_SIZE = 128
MAX_EXACT_FLOAT64_INTEGER = 2**53
INT64_MAX = t.iinfo(t.int64).max
_CPU_DEVICE = t.device("cpu")
_CUDA_DEVICE = t.device("cuda")
_TENSOR_GETITEM = t.Tensor.__getitem__
_TENSOR_ITEM = t.Tensor.item
_TENSOR_TOLIST = t.Tensor.tolist
_BUILTIN_ROUND = round


class ExactStatisticsError(ValueError):
    """An input or resource cannot satisfy the exact-reducer contract."""


@dataclass(frozen=True)
class ExactFeaturePercentiles:
    """Exact per-feature order statistics reconstructed from integer counts."""

    percentiles: dict[str, float]
    median: float
    histogram: list[int]


def count_table_nbytes(n_sae_features: int) -> int:
    """Return the exact dense int64 count-table allocation size."""
    if isinstance(n_sae_features, bool) or not isinstance(n_sae_features, int):
        raise TypeError(
            "n_sae_features must be an int, "
            f"got {type(n_sae_features).__name__}: {n_sae_features!r}"
        )
    if n_sae_features <= 0:
        raise ExactStatisticsError(f"n_sae_features must be positive, got {n_sae_features}")
    return n_sae_features * POSITIVE_FINITE_F16_COUNT * t.int64.itemsize


def _read_cgroup_available_bytes(
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """Return effective remaining cgroup-v2 memory across the full hierarchy.

    Every finite ``memory.max`` from the process' leaf cgroup through the mount
    root constrains the allocation. ``memory.current`` at each level includes
    the memory charged to its descendants, so the minimum ``max - current`` is
    the process' effective remaining hard-limit headroom.
    """
    try:
        cgroup_text = proc_cgroup_path.read_text()
    except OSError:
        return None

    relative: str | None = None
    for line in cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            relative = path
            break
    if relative is None:
        return None

    relative_path = Path(relative.lstrip("/"))
    if ".." in relative_path.parts:
        return None

    root = cgroup_root.resolve(strict=False)
    group = (root / relative_path).resolve(strict=False)
    if not group.is_relative_to(root):
        return None

    remaining_limits: list[int] = []
    while True:
        try:
            raw_max = (group / "memory.max").read_text().strip()
        except OSError:
            return None

        if raw_max != "max":
            try:
                maximum = int(raw_max)
                current = int((group / "memory.current").read_text().strip())
            except (OSError, ValueError):
                return None
            remaining_limits.append(max(0, maximum - current))

        if group == root:
            break
        group = group.parent

    return min(remaining_limits) if remaining_limits else None


def effective_available_memory_bytes(
    *,
    host_available_bytes: int | None = None,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int:
    """Return current memory headroom, respecting a finite cgroup-v2 limit."""
    if host_available_bytes is not None and (
        isinstance(host_available_bytes, bool) or not isinstance(host_available_bytes, int)
    ):
        raise TypeError(
            "host_available_bytes must be an int, "
            f"got {type(host_available_bytes).__name__}: {host_available_bytes!r}"
        )
    host_available = (
        int(psutil.virtual_memory().available)
        if host_available_bytes is None
        else host_available_bytes
    )
    if host_available < 0:
        raise ExactStatisticsError(
            f"host available memory must be nonnegative, got {host_available}"
        )
    cgroup_available = _read_cgroup_available_bytes(
        proc_cgroup_path=proc_cgroup_path,
        cgroup_root=cgroup_root,
    )
    return min(host_available, cgroup_available) if cgroup_available is not None else host_available


def require_count_table_memory(
    n_sae_features: int,
    *,
    available_bytes: int | None = None,
) -> int:
    """Fail before allocation when the count table alone cannot fit.

    This is a necessary admission check, not a claim about total peak RSS: a
    caller must separately account for shard-load and application transients.
    """
    required = count_table_nbytes(n_sae_features)
    available = effective_available_memory_bytes() if available_bytes is None else available_bytes
    if isinstance(available, bool) or not isinstance(available, int):
        raise TypeError(
            f"available_bytes must be an int, got {type(available).__name__}: {available!r}"
        )
    if available < 0:
        raise ExactStatisticsError(f"available_bytes must be nonnegative, got {available}")
    if required > available:
        raise ExactStatisticsError(
            "exact float16 feature-statistics count table does not fit current "
            f"memory headroom: width={n_sae_features:,}, "
            f"required={required:,} bytes ({required / 2**30:.3f} GiB), "
            f"available={available:,} bytes ({available / 2**30:.3f} GiB). "
            "This check covers only the table; shard-load transients require "
            "additional headroom. Refusing to allocate or restore magnitude spills."
        )
    return required


def encode_f16_count_keys(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    *,
    n_sae_features: int,
) -> t.Tensor:
    """Validate and encode one bounded activation chunk into composite keys."""
    if feature_ids.ndim != 1 or feature_mags.ndim != 1:
        raise ExactStatisticsError(
            "feature_ids and feature_mags must both be 1-D, got "
            f"{tuple(feature_ids.shape)} and {tuple(feature_mags.shape)}"
        )
    if feature_ids.shape != feature_mags.shape:
        raise ExactStatisticsError(
            f"feature_ids shape {tuple(feature_ids.shape)} != "
            f"feature_mags shape {tuple(feature_mags.shape)}"
        )
    if feature_ids.dtype != t.int32:
        raise ExactStatisticsError(
            f"feature_ids must have loaded-shard dtype int32, got {feature_ids.dtype}"
        )
    if feature_mags.dtype != t.float32:
        raise ExactStatisticsError(
            f"feature_mags must have loaded-shard dtype float32, got {feature_mags.dtype}"
        )
    count_table_nbytes(n_sae_features)
    if feature_ids.numel() == 0:
        return t.empty(0, dtype=t.int64, device=feature_ids.device)

    min_id = int(feature_ids.min().item())
    max_id = int(feature_ids.max().item())
    if min_id < 0 or max_id >= n_sae_features:
        raise ExactStatisticsError(
            "feature IDs lie outside the declared SAE dictionary: "
            f"min={min_id}, max={max_id}, valid=[0, {n_sae_features})"
        )
    finite = t.isfinite(feature_mags)
    positive = feature_mags > 0
    if not bool(finite.all().item()) or not bool(positive.all().item()):
        invalid = (~finite) | (~positive)
        first = int(invalid.nonzero(as_tuple=True)[0][0].item())
        raise ExactStatisticsError(
            "exact float16 statistics require positive finite magnitudes; "
            f"feature_mags[{first}]={feature_mags[first].item()!r}"
        )

    mags_f16 = feature_mags.to(t.float16)
    roundtrip = mags_f16.to(t.float32)
    exact = roundtrip == feature_mags
    if not bool(exact.all().item()):
        first = int((~exact).nonzero(as_tuple=True)[0][0].item())
        raise ExactStatisticsError(
            "exact float16 statistics cannot encode non-float16-exact input; "
            f"feature_mags[{first}]={feature_mags[first].item()!r} round-trips "
            f"as {roundtrip[first].item()!r}"
        )

    bit_patterns = mags_f16.contiguous().view(t.int16).to(t.int64)
    min_bits = int(bit_patterns.min().item())
    max_bits = int(bit_patterns.max().item())
    if min_bits < 1 or max_bits > POSITIVE_FINITE_F16_COUNT:
        raise ExactStatisticsError(
            "positive finite float16 bit pattern outside canonical domain: "
            f"min=0x{min_bits:04x}, max=0x{max_bits:04x}"
        )
    return feature_ids.to(t.int64) * POSITIVE_FINITE_F16_COUNT + (bit_patterns - 1)


def _cuda_unique_f16_counts(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    *,
    n_sae_features: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Reduce one CPU shard chunk on CUDA and return sparse CPU counts."""
    cuda_feature_ids = feature_ids.to("cuda")
    cuda_feature_mags = feature_mags.to("cuda")
    keys = encode_f16_count_keys(
        cuda_feature_ids,
        cuda_feature_mags,
        n_sae_features=n_sae_features,
    )
    unique_keys, multiplicities = t.unique(keys, return_counts=True)
    return unique_keys.cpu(), multiplicities.cpu()


@dataclass
class ExactMagnitudeCounts:
    """Dense exact counts over feature ID and positive finite float16 value."""

    counts: t.Tensor
    total_count: int = 0

    @classmethod
    def allocate(
        cls,
        n_sae_features: int,
        *,
        available_bytes: int | None = None,
    ) -> ExactMagnitudeCounts:
        require_count_table_memory(
            n_sae_features,
            available_bytes=available_bytes,
        )
        try:
            counts = t.zeros(
                (n_sae_features, POSITIVE_FINITE_F16_COUNT),
                dtype=t.int64,
            )
        except (MemoryError, RuntimeError) as exc:
            required = count_table_nbytes(n_sae_features)
            raise ExactStatisticsError(
                "failed to allocate exact float16 count table after admission: "
                f"shape=({n_sae_features}, {POSITIVE_FINITE_F16_COUNT}), "
                f"bytes={required:,}"
            ) from exc
        return cls(counts=counts)

    def __post_init__(self) -> None:
        if (
            self.counts.ndim != 2
            or self.counts.shape[0] <= 0
            or self.counts.shape[1] != POSITIVE_FINITE_F16_COUNT
            or self.counts.dtype != t.int64
            or self.counts.device.type != "cpu"
            or not self.counts.is_contiguous()
        ):
            raise ExactStatisticsError(
                "counts must be a contiguous CPU int64 tensor with shape "
                f"[n_features, {POSITIVE_FINITE_F16_COUNT}], got "
                f"shape={tuple(self.counts.shape)} dtype={self.counts.dtype} "
                f"device={self.counts.device}"
            )
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
            or self.total_count > INT64_MAX
        ):
            raise ExactStatisticsError(
                f"total_count must lie in [0, {INT64_MAX}], got {self.total_count}"
            )

    @property
    def n_sae_features(self) -> int:
        return int(self.counts.shape[0])

    def add(
        self,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        *,
        chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
        reduction_device: t.device = _CPU_DEVICE,
    ) -> None:
        """Add one shard exactly, using bounded temporary chunks."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an int, got {type(chunk_size).__name__}: {chunk_size!r}"
            )
        if chunk_size <= 0:
            raise ExactStatisticsError(f"chunk_size must be positive, got {chunk_size}")
        if not isinstance(reduction_device, t.device):
            raise ExactStatisticsError(
                "reduction_device must be a torch.device, "
                f"got {reduction_device!r}"
            )
        if reduction_device not in (_CPU_DEVICE, _CUDA_DEVICE):
            raise ExactStatisticsError(
                "reduction_device must be torch.device('cpu') or "
                f"torch.device('cuda'), got {reduction_device!r}"
            )
        if reduction_device == _CUDA_DEVICE and not t.cuda.is_available():
            raise ExactStatisticsError(
                "reduction_device=torch.device('cuda') requires an available CUDA device"
            )
        if feature_ids.device.type != "cpu" or feature_mags.device.type != "cpu":
            raise ExactStatisticsError(
                "ExactMagnitudeCounts.add requires CPU shard tensors; "
                f"got {feature_ids.device} and {feature_mags.device}"
            )
        if feature_ids.shape != feature_mags.shape:
            raise ExactStatisticsError(
                f"feature_ids shape {tuple(feature_ids.shape)} != "
                f"feature_mags shape {tuple(feature_mags.shape)}"
            )
        n = int(feature_ids.numel())
        if self.total_count + n > INT64_MAX:
            raise ExactStatisticsError(
                "activation count would overflow int64: "
                f"current={self.total_count}, incoming={n}, max={INT64_MAX}"
            )
        if n == 0:
            encode_f16_count_keys(
                feature_ids,
                feature_mags,
                n_sae_features=self.n_sae_features,
            )
            return

        flat_counts = self.counts.view(-1)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            if reduction_device == _CPU_DEVICE:
                keys = encode_f16_count_keys(
                    feature_ids[start:end],
                    feature_mags[start:end],
                    n_sae_features=self.n_sae_features,
                )
                unique_keys, multiplicities = keys, t.ones_like(keys)
                del keys
            else:
                unique_keys, multiplicities = _cuda_unique_f16_counts(
                    feature_ids[start:end],
                    feature_mags[start:end],
                    n_sae_features=self.n_sae_features,
                )
            represented = int(multiplicities.sum().item())
            if represented != end - start:
                raise RuntimeError(
                    "composite-key reduction lost activations: "
                    f"chunk={start}:{end}, represented={represented}"
                )
            flat_counts.index_add_(0, unique_keys, multiplicities)
            del unique_keys, multiplicities
        self.total_count += n

    def finalize(
        self,
        *,
        occurrence: t.Tensor,
        mag_subsample: t.Tensor,
        global_mag_min: float,
        percentiles: tuple[int, ...],
        n_hist_bins: int = 50,
    ) -> tuple[dict[int, ExactFeaturePercentiles], list[float] | None]:
        """Reconstruct exact order statistics and histogram counts."""
        if occurrence.ndim != 1 or occurrence.dtype != t.int64:
            raise ExactStatisticsError(
                f"occurrence must be 1-D int64, got {tuple(occurrence.shape)} {occurrence.dtype}"
            )
        if occurrence.device.type != "cpu":
            raise ExactStatisticsError(f"occurrence must be on CPU, got {occurrence.device}")
        if len(occurrence) != self.n_sae_features:
            raise ExactStatisticsError(
                f"occurrence width {len(occurrence)} != count-table width {self.n_sae_features}"
            )
        if bool((occurrence < 0).any().item()):
            raise ExactStatisticsError("occurrence counts must be nonnegative")
        occurrence_total = int(occurrence.sum().item())
        if occurrence_total != self.total_count:
            raise ExactStatisticsError(
                f"occurrence total {occurrence_total} != exact-table total {self.total_count}"
            )
        if isinstance(n_hist_bins, bool) or not isinstance(n_hist_bins, int):
            raise TypeError(
                f"n_hist_bins must be an int, got {type(n_hist_bins).__name__}: {n_hist_bins!r}"
            )
        if n_hist_bins <= 0:
            raise ExactStatisticsError(f"n_hist_bins must be positive, got {n_hist_bins}")
        if any(
            isinstance(p, bool) or not isinstance(p, int) or p < 0 or p > 100 for p in percentiles
        ):
            raise ExactStatisticsError(
                f"percentiles must be integer values in [0, 100], got {percentiles!r}"
            )

        # Audit by bounded row windows. A whole-table ``self.counts < 0``
        # temporary would itself consume almost 2 GiB at 65k width and could
        # turn a successful admission into an avoidable finalization OOM.
        table_total = 0
        for start in range(0, self.n_sae_features, COUNT_AUDIT_ROW_CHUNK_SIZE):
            rows = self.counts[start : start + COUNT_AUDIT_ROW_CHUNK_SIZE]
            if bool((rows < 0).any().item()):
                raise ExactStatisticsError("count table contains negative values")
            table_total += int(rows.sum().item())
        if table_total != self.total_count:
            raise ExactStatisticsError(
                f"count-table sum {table_total} != tracked total {self.total_count}"
            )

        active_ids = (occurrence > 0).nonzero(as_tuple=True)[0]
        if active_ids.numel() == 0:
            if self.total_count != 0:
                raise ExactStatisticsError(
                    "count table is nonempty but occurrence has no active features"
                )
            return {}, None

        shared_edges = shared_histogram_edges(
            mag_subsample,
            global_mag_min=global_mag_min,
            n_hist_bins=n_hist_bins,
        )
        pct_fracs = t.tensor(
            [p / 100.0 for p in percentiles],
            dtype=t.float64,
        )
        native_percentiles = type(percentiles) is tuple and all(
            type(p) is int for p in percentiles
        )
        result: dict[int, ExactFeaturePercentiles] = {}
        histogram_bin_ids: t.Tensor | None = None

        for fid in active_ids.tolist():
            row = self.counts[fid]
            columns = (row > 0).nonzero(as_tuple=True)[0]
            if columns.numel() == 0:
                raise ExactStatisticsError(
                    f"feature {fid}: occurrence is positive but exact count row is empty"
                )
            multiplicities = row[columns]
            cumulative = multiplicities.cumsum(0)
            count = int(cumulative[-1].item())
            expected_count = int(occurrence[fid].item())
            if count != expected_count:
                raise ExactStatisticsError(
                    f"feature {fid}: exact count {count} != occurrence {expected_count}"
                )

            bit_patterns = (columns + 1).to(t.int16)
            values = bit_patterns.contiguous().view(t.float16).to(t.float64)

            continuous_idx = (count - 1) * pct_fracs
            lower_idx = continuous_idx.floor().to(t.int64)
            upper_idx = continuous_idx.ceil().to(t.int64).clamp(max=count - 1)
            frac = continuous_idx - lower_idx.to(t.float64)
            lower_pos = t.searchsorted(cumulative, lower_idx, right=True)
            upper_pos = t.searchsorted(cumulative, upper_idx, right=True)
            quantile_values = values[lower_pos] + frac * (values[upper_pos] - values[lower_pos])

            median_continuous = (count - 1) * 0.5
            median_lower = int(median_continuous)
            median_upper = min(median_lower + 1, count - 1)
            median_lower_pos = int(
                t.searchsorted(
                    cumulative,
                    t.tensor(median_lower, dtype=t.int64),
                    right=True,
                ).item()
            )
            median_upper_pos = int(
                t.searchsorted(
                    cumulative,
                    t.tensor(median_upper, dtype=t.int64),
                    right=True,
                ).item()
            )
            median_frac = median_continuous - median_lower
            median_value = values[median_lower_pos].item() + median_frac * (
                values[median_upper_pos].item() - values[median_lower_pos].item()
            )

            if histogram_bin_ids is None:
                domain_values = t.arange(
                    1, POSITIVE_FINITE_F16_COUNT + 1, dtype=t.int16, device=columns.device
                ).view(t.float16).to(t.float64)
                histogram_bin_ids = t.bucketize(
                    domain_values.clamp(shared_edges[0], shared_edges[-1]), shared_edges[1:]
                ).clamp(max=n_hist_bins - 1)
            bin_idx = histogram_bin_ids[columns]
            histogram = t.zeros(n_hist_bins, dtype=t.int64)
            histogram.index_add_(0, bin_idx, multiplicities)
            if int(histogram.sum().item()) != count:
                raise RuntimeError(f"feature {fid}: histogram does not preserve count {count}")

            # Resolve the live constructor before evaluating its arguments.
            feature_percentiles_type = ExactFeaturePercentiles
            if (
                native_percentiles
                and type(quantile_values) is t.Tensor
                and not t.overrides.has_torch_function((quantile_values,))
                and not is_in_torch_dispatch_mode()
                and not t._is_functional_tensor(quantile_values)
                and round is _BUILTIN_ROUND
                and t.Tensor.__getitem__ is _TENSOR_GETITEM
                and t.Tensor.item is _TENSOR_ITEM
                and t.Tensor.tolist is _TENSOR_TOLIST
                and quantile_values.device.type == "cpu"
                and quantile_values.dtype is t.float64
                and quantile_values.layout is t.strided
                and tuple(quantile_values.shape) == (len(percentiles),)
                and not quantile_values.requires_grad
                and not dict.__len__(vars(quantile_values))
            ):
                quantile_row = quantile_values.tolist()
                pcts = {
                    f"p{p}": round(quantile_row[j], 6) for j, p in enumerate(percentiles)
                }
            else:
                pcts = {
                    f"p{p}": round(quantile_values[j].item(), 6) for j, p in enumerate(percentiles)
                }
            result[fid] = feature_percentiles_type(
                percentiles=pcts,
                median=round(median_value, 6),
                histogram=histogram.tolist(),
            )

        return result, [round(value, 6) for value in shared_edges.tolist()]


def shared_histogram_edges(
    mag_subsample: t.Tensor,
    *,
    global_mag_min: float,
    n_hist_bins: int = 50,
) -> t.Tensor:
    """Reproduce the streaming implementation's unrounded shared edges."""
    if mag_subsample.ndim != 1:
        raise ExactStatisticsError(f"mag_subsample must be 1-D, got {tuple(mag_subsample.shape)}")
    if mag_subsample.dtype != t.float32 or mag_subsample.device.type != "cpu":
        raise ExactStatisticsError(
            "mag_subsample must be a CPU float32 tensor, got "
            f"dtype={mag_subsample.dtype} device={mag_subsample.device}"
        )
    if isinstance(n_hist_bins, bool) or not isinstance(n_hist_bins, int):
        raise TypeError(
            f"n_hist_bins must be an int, got {type(n_hist_bins).__name__}: {n_hist_bins!r}"
        )
    if n_hist_bins <= 0:
        raise ExactStatisticsError(f"n_hist_bins must be positive, got {n_hist_bins}")
    if mag_subsample.numel() == 0:
        mag_lo, mag_hi = 0.0, 1.0
    else:
        if not math.isfinite(global_mag_min) or global_mag_min <= 0:
            raise ExactStatisticsError(
                "global_mag_min must be positive and finite for a nonempty "
                f"sample, got {global_mag_min!r}"
            )
        sample_f64 = mag_subsample.to(t.float64)
        if not bool(t.isfinite(sample_f64).all().item()):
            raise ExactStatisticsError("mag_subsample contains non-finite values")
        mag_lo = global_mag_min
        k99 = max(1, int(len(sample_f64) * 0.99))
        mag_p99 = sample_f64.kthvalue(k99).values.item()
        mag_hi = max(mag_p99, mag_lo + 1e-6)
    return t.linspace(
        mag_lo,
        mag_hi,
        n_hist_bins + 1,
        dtype=t.float64,
    )


WeightedValue = int | float


@dataclass
class WeightedDistribution:
    """An exact finite population represented by value multiplicities."""

    counts: Counter[WeightedValue] = field(default_factory=Counter)
    population_size: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size < 0
            or self.population_size > INT64_MAX
        ):
            raise ExactStatisticsError(
                f"population_size must lie in [0, {INT64_MAX}], got {self.population_size!r}"
            )
        for value, multiplicity in self.counts.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExactStatisticsError(
                    f"weighted-distribution key must be int or float, got {value!r}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ExactStatisticsError(
                    f"weighted-distribution key must be finite, got {value!r}"
                )
            if (
                isinstance(multiplicity, bool)
                or not isinstance(multiplicity, int)
                or multiplicity <= 0
            ):
                raise ExactStatisticsError(
                    "weighted-distribution multiplicities must be positive ints, "
                    f"got {value!r}: {multiplicity!r}"
                )
        if sum(self.counts.values()) != self.population_size:
            raise ExactStatisticsError(
                "weighted-distribution counter total does not match tracked population"
            )

    def add(self, values: Iterable[WeightedValue]) -> None:
        materialized = list(values)
        for value in materialized:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExactStatisticsError(
                    "weighted-distribution values must be int or float, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ExactStatisticsError(
                    f"weighted-distribution values must be finite, got {value!r}"
                )
        if self.population_size + len(materialized) > INT64_MAX:
            raise ExactStatisticsError(
                "weighted-distribution population would overflow int64: "
                f"current={self.population_size}, incoming={len(materialized)}"
            )
        self.counts.update(materialized)
        self.population_size += len(materialized)

    def add_weighted(self, value: WeightedValue, multiplicity: int) -> None:
        if isinstance(multiplicity, bool) or not isinstance(multiplicity, int):
            raise TypeError(
                f"multiplicity must be an int, got {type(multiplicity).__name__}: {multiplicity!r}"
            )
        if multiplicity < 0:
            raise ExactStatisticsError(f"multiplicity must be nonnegative, got {multiplicity}")
        if multiplicity == 0:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExactStatisticsError(
                "weighted-distribution values must be int or float, "
                f"got {type(value).__name__}: {value!r}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ExactStatisticsError(
                f"weighted-distribution values must be finite, got {value!r}"
            )
        if self.population_size + multiplicity > INT64_MAX:
            raise ExactStatisticsError(
                "weighted-distribution population would overflow int64: "
                f"current={self.population_size}, incoming={multiplicity}"
            )
        self.counts[value] += multiplicity
        self.population_size += multiplicity

    def _sorted_tensors(self) -> tuple[t.Tensor, t.Tensor]:
        if self.population_size == 0:
            return (
                t.empty(0, dtype=t.float64),
                t.empty(0, dtype=t.int64),
            )
        if sum(self.counts.values()) != self.population_size:
            raise ExactStatisticsError(
                "weighted-distribution counter total does not match tracked population"
            )
        ordered = sorted(self.counts.items(), key=lambda item: item[0])
        values = t.tensor([item[0] for item in ordered], dtype=t.float64)
        multiplicities = t.tensor([item[1] for item in ordered], dtype=t.int64)
        return values, multiplicities

    def percentile_dict(
        self,
        percentiles: tuple[int, ...],
        *,
        round_digits: int | None = None,
        as_int: bool = False,
    ) -> dict[str, float | int]:
        """Reproduce ``compute_statistics._percentile_dict`` by weight."""
        if self.population_size == 0:
            return {}
        if any(
            isinstance(p, bool) or not isinstance(p, int) or p < 0 or p > 100 for p in percentiles
        ):
            raise ExactStatisticsError(
                f"percentiles must be integer values in [0, 100], got {percentiles!r}"
            )
        values, multiplicities = self._sorted_tensors()
        cumulative = multiplicities.cumsum(0)
        result: dict[str, float | int] = {}
        for percentile in percentiles:
            rank = min(
                int(percentile / 100 * self.population_size),
                self.population_size - 1,
            )
            position = int(
                t.searchsorted(
                    cumulative,
                    t.tensor(rank, dtype=t.int64),
                    right=True,
                ).item()
            )
            value = values[position].item()
            if as_int:
                result[f"p{percentile}"] = int(value)
            elif round_digits is not None:
                result[f"p{percentile}"] = round(value, round_digits)
            else:
                result[f"p{percentile}"] = value
        return result

    def histogram(self, *, n_bins: int = 50) -> HistogramData | None:
        """Reproduce ``torch.histogram(expanded_values)`` without expansion."""
        if self.population_size == 0:
            return None
        if isinstance(n_bins, bool) or not isinstance(n_bins, int):
            raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}: {n_bins!r}")
        if n_bins <= 0:
            raise ExactStatisticsError(f"n_bins must be positive, got {n_bins}")
        if self.population_size > MAX_EXACT_FLOAT64_INTEGER:
            raise ExactStatisticsError(
                "weighted torch.histogram cannot represent every integer count "
                f"exactly above 2**53; population={self.population_size}"
            )
        values, multiplicities = self._sorted_tensors()
        counts_f64, edges = t.histogram(
            values,
            bins=n_bins,
            weight=multiplicities.to(t.float64),
        )
        counts_i64 = counts_f64.to(t.int64)
        if not t.equal(counts_i64.to(t.float64), counts_f64):
            raise ExactStatisticsError("weighted histogram produced a non-integral count")
        if int(counts_i64.sum().item()) != self.population_size:
            raise ExactStatisticsError(
                "weighted histogram does not preserve the population: "
                f"histogram={int(counts_i64.sum().item())}, "
                f"population={self.population_size}"
            )
        return HistogramData(
            bin_edges=[round(value, 4) for value in edges.tolist()],
            # Preserve the historical ``torch.histogram`` serialization type:
            # counts are mathematically integral but were emitted as floats.
            counts=counts_f64.tolist(),
        )

    def exact_integer_sum(self) -> int:
        """Return the exact population sum, requiring integer-valued keys."""
        total = 0
        for value, multiplicity in self.counts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ExactStatisticsError(
                    f"exact_integer_sum requires integer values, got {value!r}"
                )
            total += value * multiplicity
        return total

    def minimum(self) -> WeightedValue:
        """Return the least represented value."""
        if self.population_size == 0:
            raise ExactStatisticsError("minimum is undefined for an empty distribution")
        return min(self.counts)

    def maximum(self) -> WeightedValue:
        """Return the greatest represented value."""
        if self.population_size == 0:
            raise ExactStatisticsError("maximum is undefined for an empty distribution")
        return max(self.counts)
