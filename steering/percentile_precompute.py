"""Shared tensor computation for percentile-overlay producers."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch as t

from errors import EmptyExtractionSegmentError
from steering.conditional.percentile_table import (
    _magnitude_to_percentile,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.feature_io import (
    EncodedPercentileFeatures,
    TensorKey,
    decode_feature_id_deltas,
    empty_segment_indices,
)

# Keep these bounds identical to the original local producer.  The lookup is a
# direct-gather binary search; the larger CUDA chunk amortizes kernel launches.
CPU_CHUNK_SIZE = 1_000_000
CUDA_CHUNK_SIZE = 10_000_000


@dataclass
class PercentileChunkTiming:
    """Internal phase attribution for lazily decoded percentile chunks."""

    decode_seconds: float = 0.0
    lookup_seconds: float = 0.0


def _lookup_percentile_chunk(
    feature_ids: t.Tensor,
    feature_magnitudes: t.Tensor,
    percentile_table: t.Tensor,
    *,
    device: t.device,
    _table_is_finite: bool = False,
) -> t.Tensor:
    if device.type == "cuda":
        feature_ids = feature_ids.to(device)
        feature_magnitudes = feature_magnitudes.to(device)
    percentiles = (
        _magnitude_to_percentile(
            feature_ids, feature_magnitudes, percentile_table, table_is_finite=True
        )
        if _table_is_finite and device.type != "cuda"
        else magnitude_to_percentile(feature_ids, feature_magnitudes, percentile_table)
    )
    return percentile_to_uint8(percentiles)


def magnitude_to_percentile_chunked(
    feature_ids: t.Tensor,
    feature_magnitudes: t.Tensor,
    percentile_table: t.Tensor,
    *,
    device: t.device | str = "cpu",
    _table_is_finite: bool = False,
) -> t.Tensor:
    """Return the legacy chunked uint8 percentile encoding on CPU.

    Loaded feature IDs retain their restored dtype through CUDA transfer; the
    lookup normalizes them to int64 on its current device.
    """
    n_values = len(feature_ids)
    if n_values == 0:
        return t.tensor([], dtype=t.uint8)

    resolved_device = t.device(device)
    use_cuda = resolved_device.type == "cuda"
    chunk_size = CUDA_CHUNK_SIZE if use_cuda else CPU_CHUNK_SIZE
    table_on_device = percentile_table.to(resolved_device) if use_cuda else percentile_table

    if n_values <= chunk_size:
        return _lookup_percentile_chunk(
            feature_ids,
            feature_magnitudes,
            table_on_device,
            device=resolved_device,
            _table_is_finite=_table_is_finite,
        ).cpu()

    parts: list[t.Tensor] = []
    for start in range(0, n_values, chunk_size):
        end = min(start + chunk_size, n_values)
        parts.append(
            _lookup_percentile_chunk(
                feature_ids[start:end],
                feature_magnitudes[start:end],
                table_on_device,
                device=resolved_device,
                _table_is_finite=_table_is_finite,
            ).cpu()
        )
    return t.cat(parts)


def _segment_aligned_chunk_spans(
    offsets: t.Tensor,
    *,
    chunk_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield complete-segment spans with ``chunk_size`` as a soft value bound."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    boundaries = offsets.tolist()
    segment_start = 0
    final_boundary = len(boundaries) - 1
    while segment_start < final_boundary:
        value_start = boundaries[segment_start]
        target = value_start + chunk_size
        segment_end = bisect_right(
            boundaries,
            target,
            lo=segment_start + 1,
        ) - 1
        if segment_end == segment_start:
            # One segment may exceed the soft bound. Keep it intact so its
            # delta stream remains independently decodable.
            segment_end += 1
        value_end = boundaries[segment_end]
        yield segment_start, segment_end, value_start, value_end
        segment_start = segment_end


def _validate_encoded_percentile_features(
    encoded: EncodedPercentileFeatures,
    *,
    source_name: str,
) -> None:
    offsets = encoded.offsets
    feature_ids = encoded.feature_id_deltas
    feature_magnitudes = encoded.feature_magnitudes
    if offsets.ndim != 1 or feature_ids.ndim != 1 or feature_magnitudes.ndim != 1:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} requires rank-1 "
            "offsets, feature_ids, and feature_mags"
        )
    if feature_ids.shape != feature_magnitudes.shape:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} has feature_ids "
            f"shape {tuple(feature_ids.shape)} but feature_mags shape "
            f"{tuple(feature_magnitudes.shape)}"
        )
    if offsets.numel() == 0 or int(offsets[0].item()) != 0:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} offsets must "
            "be non-empty and start at 0"
        )
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} offsets must be monotonic"
        )
    if int(offsets[-1].item()) != encoded.n_values:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} offsets[-1] "
            f"must match feature_ids length ({int(offsets[-1].item())} != "
            f"{encoded.n_values})"
        )
    empty_segments = empty_segment_indices(offsets)
    if empty_segments:
        raise EmptyExtractionSegmentError(
            f"precompute_percentiles: base shard {source_name} segment "
            f"{empty_segments[0]} has zero features; base shards must be "
            "non-empty (issue #522)."
        )


def _decode_encoded_percentile_span(
    encoded: EncodedPercentileFeatures,
    span: tuple[int, int, int, int],
) -> tuple[int, int, t.Tensor, t.Tensor, float]:
    """Decode one complete-segment span without touching CUDA state."""
    segment_start, segment_end, value_start, value_end = span
    started = time.monotonic()
    relative_offsets = (
        encoded.offsets[segment_start : segment_end + 1] - value_start
    )
    feature_ids = decode_feature_id_deltas(
        encoded.feature_id_deltas[value_start:value_end],
        relative_offsets,
    )
    feature_magnitudes = encoded.feature_magnitudes[value_start:value_end]
    return (
        value_start,
        value_end,
        feature_ids,
        feature_magnitudes,
        time.monotonic() - started,
    )


def compute_percentile_overlay_values_from_encoded(
    encoded: EncodedPercentileFeatures,
    percentile_table: t.Tensor,
    *,
    device: t.device | str = "cpu",
    source_name: str,
    timing: PercentileChunkTiming | None = None,
) -> t.Tensor:
    """Decode complete segment spans and write their percentile bytes in place."""
    _validate_encoded_percentile_features(encoded, source_name=source_name)
    if encoded.n_values == 0:
        return t.empty(0, dtype=t.uint8)

    resolved_device = t.device(device)
    chunk_size = (
        CUDA_CHUNK_SIZE if resolved_device.type == "cuda" else CPU_CHUNK_SIZE
    )
    lookup_started = time.monotonic() if timing is not None else 0.0
    table_on_device = (
        percentile_table.to(resolved_device)
        if resolved_device.type == "cuda"
        else percentile_table
    )
    if timing is not None:
        timing.lookup_seconds += time.monotonic() - lookup_started

    result = t.empty(encoded.n_values, dtype=t.uint8)

    def consume(
        decoded: tuple[int, int, t.Tensor, t.Tensor, float],
    ) -> None:
        value_start, value_end, feature_ids, feature_magnitudes, decode_seconds = decoded
        if timing is not None:
            timing.decode_seconds += decode_seconds

        lookup_started = time.monotonic() if timing is not None else 0.0
        values = _lookup_percentile_chunk(
            feature_ids,
            feature_magnitudes,
            table_on_device,
            device=resolved_device,
        )
        result[value_start:value_end].copy_(values)
        if timing is not None:
            timing.lookup_seconds += time.monotonic() - lookup_started

    spans = iter(_segment_aligned_chunk_spans(encoded.offsets, chunk_size=chunk_size))
    if resolved_device.type != "cuda":
        for span in spans:
            consume(_decode_encoded_percentile_span(encoded, span))
        return result

    # Keep exactly one decoded CPU chunk ahead of CUDA. The worker never
    # initializes CUDA, and the executor's context joins it on every exit.
    first_span = next(spans)
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="percentile-chunk-decode",
    ) as executor:
        pending = executor.submit(_decode_encoded_percentile_span, encoded, first_span)
        for span in spans:
            decoded = pending.result()
            pending = executor.submit(_decode_encoded_percentile_span, encoded, span)
            consume(decoded)
        consume(pending.result())

    return result


def compute_percentile_overlay_values(
    feature_data: Mapping[str, object],
    percentile_table: t.Tensor,
    *,
    device: t.device | str = "cpu",
    source_name: str,
    _table_is_finite: bool = False,
) -> t.Tensor:
    """Validate one base shard and compute its rank-1 uint8 overlay tensor."""
    offsets = feature_data[TensorKey.OFFSETS]
    feature_ids = feature_data[TensorKey.FEATURE_IDS]
    feature_magnitudes = feature_data[TensorKey.FEATURE_MAGS]
    if not isinstance(offsets, t.Tensor):
        raise TypeError(f"{source_name}: offsets must be a tensor")
    if not isinstance(feature_ids, t.Tensor):
        raise TypeError(f"{source_name}: feature_ids must be a tensor")
    if not isinstance(feature_magnitudes, t.Tensor):
        raise TypeError(f"{source_name}: feature_mags must be a tensor")

    empty_segments = empty_segment_indices(offsets)
    if empty_segments:
        index = empty_segments[0]
        raise EmptyExtractionSegmentError(
            f"precompute_percentiles: base shard {source_name} segment {index} "
            "has zero features; base shards must be non-empty (issue #522)."
        )
    if feature_ids.ndim != 1 or feature_magnitudes.ndim != 1:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} requires rank-1 "
            "feature_ids and feature_mags"
        )
    if feature_ids.shape != feature_magnitudes.shape:
        raise ValueError(
            f"precompute_percentiles: base shard {source_name} has feature_ids "
            f"shape {tuple(feature_ids.shape)} but feature_mags shape "
            f"{tuple(feature_magnitudes.shape)}"
        )

    return magnitude_to_percentile_chunked(
        feature_ids,
        feature_magnitudes,
        percentile_table,
        device=device,
        _table_is_finite=_table_is_finite,
    )


__all__ = [
    "CPU_CHUNK_SIZE",
    "CUDA_CHUNK_SIZE",
    "PercentileChunkTiming",
    "compute_percentile_overlay_values",
    "compute_percentile_overlay_values_from_encoded",
    "magnitude_to_percentile_chunked",
]
