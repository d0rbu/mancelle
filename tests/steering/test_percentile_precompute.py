"""Characterization tests for the shared percentile computation helper."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save as safetensors_save_bytes

import steering.percentile_precompute as percentile_module
from errors import EmptyExtractionSegmentError
from steering.feature_io import (
    EncodedPercentileFeatures,
    TensorKey,
    decode_feature_safetensors_bytes,
    decode_percentile_feature_safetensors_bytes,
    parse_percentile_feature_safetensors_bytes,
)
from steering.overlay_io import save_overlay_shard
from steering.percentile_precompute import (
    PercentileChunkTiming,
    compute_percentile_overlay_values,
    compute_percentile_overlay_values_from_encoded,
    magnitude_to_percentile_chunked,
)


def _table(width: int = 65_536) -> t.Tensor:
    base = t.arange(1, 100, dtype=t.float64) / 100
    return base.repeat(width, 1)


def _encoded_lookup_fixture() -> bytes:
    return safetensors_save_bytes(
        {
            "offsets": t.tensor([0, 3, 6], dtype=t.int32),
            "feature_ids": t.tensor(
                [0, 1, 65_534, 2, 0, 1],
                dtype=t.uint16,
            ),
            "feature_mags": t.tensor(
                [0.01, 0.5, 0.99, float("inf"), float("nan"), 0.005],
                dtype=t.float16,
            ),
        },
        metadata={"segment_size": "128", "n_segments": "2"},
    )


def _assert_narrow_decode_output_matches_generic_reference(device: str) -> None:
    payload = _encoded_lookup_fixture()
    generic = decode_feature_safetensors_bytes(payload)
    narrow = decode_percentile_feature_safetensors_bytes(payload)
    encoded = parse_percentile_feature_safetensors_bytes(payload)
    original_deltas = encoded.feature_id_deltas.clone()
    original_magnitudes = encoded.feature_magnitudes.clone()

    expected = compute_percentile_overlay_values(
        generic,
        _table(),
        device=device,
        source_name="generic-reference",
    )
    narrow_reference = compute_percentile_overlay_values(
        narrow,
        _table(),
        device=device,
        source_name="narrow-decode",
    )
    observed = compute_percentile_overlay_values_from_encoded(
        encoded,
        _table(),
        device=device,
        source_name="chunked-decode",
    )

    assert generic["feature_mags"].dtype == t.float32
    assert narrow["feature_mags"].dtype == t.float16
    assert t.equal(narrow_reference, expected)
    assert t.equal(observed, expected)
    assert t.equal(encoded.feature_id_deltas, original_deltas)
    assert t.equal(
        encoded.feature_magnitudes.view(t.int16),
        original_magnitudes.view(t.int16),
    )


def test_narrow_decode_output_matches_generic_cpu_reference() -> None:
    _assert_narrow_decode_output_matches_generic_reference("cpu")


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_narrow_decode_output_matches_generic_cuda_reference() -> None:
    _assert_narrow_decode_output_matches_generic_reference("cuda")


def _multi_segment_encoded_fixture() -> tuple[bytes, list[int]]:
    decoded_ids = [0, 2, 1, 3, 5, 4, 0, 1, 2, 3]
    return (
        safetensors_save_bytes(
            {
                "offsets": t.tensor([0, 2, 5, 6, 10], dtype=t.int32),
                "feature_ids": t.tensor(
                    [0, 2, 1, 2, 2, 4, 0, 1, 1, 1],
                    dtype=t.uint16,
                ),
                "feature_mags": t.linspace(0.05, 0.95, 10, dtype=t.float16),
            },
            metadata={"segment_size": "128", "n_segments": "4"},
        ),
        decoded_ids,
    )


def test_segment_aligned_chunk_spans_respect_complete_segments_and_soft_bound() -> None:
    offsets = t.tensor([0, 2, 5, 6, 10], dtype=t.int64)
    assert list(
        percentile_module._segment_aligned_chunk_spans(offsets, chunk_size=4)
    ) == [
        (0, 1, 0, 2),
        (1, 3, 2, 6),
        (3, 4, 6, 10),
    ]
    assert list(
        percentile_module._segment_aligned_chunk_spans(
            t.tensor([0, 6, 8], dtype=t.int64),
            chunk_size=4,
        )
    ) == [
        (0, 1, 0, 6),
        (1, 2, 6, 8),
    ]


def test_encoded_cpu_path_decodes_complete_segments_into_preallocated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, decoded_ids = _multi_segment_encoded_fixture()
    generic = decode_feature_safetensors_bytes(payload)
    encoded = parse_percentile_feature_safetensors_bytes(payload)
    original_deltas = encoded.feature_id_deltas.clone()
    lookup = percentile_module.magnitude_to_percentile
    expected = compute_percentile_overlay_values(
        generic,
        _table(6),
        device="cpu",
        source_name="full-reference",
    )
    observed_chunks: list[t.Tensor] = []

    def observe_lookup(
        ids: t.Tensor,
        mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        observed_chunks.append(ids.clone())
        assert ids.device.type == "cpu"
        assert ids.dtype == t.int32
        return lookup(ids, mags, percentile_table)

    def reject_cat(*args, **kwargs):
        raise AssertionError("chunked percentile output must not call torch.cat")

    timing = PercentileChunkTiming()
    monkeypatch.setattr(percentile_module, "CPU_CHUNK_SIZE", 4)
    monkeypatch.setattr(percentile_module, "magnitude_to_percentile", observe_lookup)
    monkeypatch.setattr(percentile_module.t, "cat", reject_cat)
    observed = compute_percentile_overlay_values_from_encoded(
        encoded,
        _table(6),
        device="cpu",
        source_name="chunked",
        timing=timing,
    )

    assert [chunk.numel() for chunk in observed_chunks] == [2, 4, 4]
    assert [value for chunk in observed_chunks for value in chunk.tolist()] == decoded_ids
    assert t.equal(observed, expected)
    assert t.equal(encoded.feature_id_deltas, original_deltas)
    assert timing.decode_seconds > 0
    assert timing.lookup_seconds > 0


def test_encoded_empty_shard_returns_without_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = EncodedPercentileFeatures(
        offsets=t.tensor([0], dtype=t.int64),
        feature_id_deltas=t.tensor([], dtype=t.uint16),
        feature_magnitudes=t.tensor([], dtype=t.float16),
        metadata={},
    )

    def reject_lookup(*args, **kwargs):
        raise AssertionError("empty encoded shard must not initialize lookup")

    monkeypatch.setattr(percentile_module, "_lookup_percentile_chunk", reject_lookup)
    observed = compute_percentile_overlay_values_from_encoded(
        encoded,
        _table(1),
        device="cuda",
        source_name="empty",
    )
    assert observed.dtype == t.uint8
    assert observed.numel() == 0


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        (t.tensor([], dtype=t.int64), "non-empty and start at 0"),
        (t.tensor([1, 10], dtype=t.int64), "non-empty and start at 0"),
        (t.tensor([0, 8, 7, 10], dtype=t.int64), "monotonic"),
        (t.tensor([0, 9], dtype=t.int64), "must match feature_ids length"),
    ],
)
def test_encoded_path_rejects_invalid_offsets(
    offsets: t.Tensor,
    message: str,
) -> None:
    encoded = EncodedPercentileFeatures(
        offsets=offsets,
        feature_id_deltas=t.zeros(10, dtype=t.uint16),
        feature_magnitudes=t.ones(10, dtype=t.float16),
        metadata={},
    )
    with pytest.raises(ValueError, match=message):
        compute_percentile_overlay_values_from_encoded(
            encoded,
            _table(1),
            source_name="bad-offsets",
        )


def test_encoded_path_rejects_empty_base_segment() -> None:
    encoded = EncodedPercentileFeatures(
        offsets=t.tensor([0, 1, 1], dtype=t.int64),
        feature_id_deltas=t.tensor([0], dtype=t.uint16),
        feature_magnitudes=t.tensor([0.5], dtype=t.float16),
        metadata={},
    )
    with pytest.raises(EmptyExtractionSegmentError, match="segment 1"):
        compute_percentile_overlay_values_from_encoded(
            encoded,
            _table(1),
            source_name="empty-segment",
        )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_encoded_cuda_path_transfers_only_decoded_segment_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, decoded_ids = _multi_segment_encoded_fixture()
    generic = decode_feature_safetensors_bytes(payload)
    encoded = parse_percentile_feature_safetensors_bytes(payload)
    original_deltas = encoded.feature_id_deltas.clone()
    lookup = percentile_module.magnitude_to_percentile
    decode_span = percentile_module._decode_encoded_percentile_span
    expected = compute_percentile_overlay_values(
        generic,
        _table(6),
        device="cuda",
        source_name="full-reference",
    )
    observed_chunks: list[t.Tensor] = []
    decode_threads: list[str] = []

    def observe_decode(*args, **kwargs):
        decode_threads.append(threading.current_thread().name)
        return decode_span(*args, **kwargs)

    def observe_lookup(
        ids: t.Tensor,
        mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        observed_chunks.append(ids.detach().cpu().clone())
        assert ids.device.type == "cuda"
        assert ids.dtype == t.int32
        assert mags.device.type == "cuda"
        assert mags.dtype == t.float16
        return lookup(ids, mags, percentile_table)

    monkeypatch.setattr(percentile_module, "CUDA_CHUNK_SIZE", 4)
    monkeypatch.setattr(
        percentile_module,
        "_decode_encoded_percentile_span",
        observe_decode,
    )
    monkeypatch.setattr(percentile_module, "magnitude_to_percentile", observe_lookup)
    observed = compute_percentile_overlay_values_from_encoded(
        encoded,
        _table(6),
        device="cuda",
        source_name="chunked-cuda",
    )

    assert [chunk.numel() for chunk in observed_chunks] == [2, 4, 4]
    assert [value for chunk in observed_chunks for value in chunk.tolist()] == decoded_ids
    assert decode_threads
    assert all(name.startswith("percentile-chunk-decode") for name in decode_threads)
    assert t.equal(observed, expected)
    assert t.equal(encoded.feature_id_deltas, original_deltas)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_encoded_cuda_lookup_failure_joins_decode_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ = _multi_segment_encoded_fixture()
    encoded = parse_percentile_feature_safetensors_bytes(payload)

    def fail_lookup(*args, **kwargs):
        raise RuntimeError("injected lookup failure")

    monkeypatch.setattr(percentile_module, "CUDA_CHUNK_SIZE", 4)
    monkeypatch.setattr(percentile_module, "_lookup_percentile_chunk", fail_lookup)
    with pytest.raises(RuntimeError, match="injected lookup failure"):
        compute_percentile_overlay_values_from_encoded(
            encoded,
            _table(6),
            device="cuda",
            source_name="faulted-cuda",
        )

    assert not any(
        thread.name.startswith("percentile-chunk-decode")
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize(
    ("feature_ids", "magnitudes"),
    [
        (t.tensor([], dtype=t.int32), t.tensor([], dtype=t.float32)),
        (
            t.tensor([0, 0, 1, 65_535], dtype=t.int32),
            t.tensor([0.01, 0.5, 0.5, 0.99], dtype=t.float32),
        ),
        (
            t.tensor([2, 2, 2, 3], dtype=t.int32),
            t.tensor([0.5, 0.5, float("inf"), float("nan")], dtype=t.float32),
        ),
    ],
)
def test_shared_computation_has_legacy_shape_dtype_and_determinism(
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
) -> None:
    offsets = (
        t.tensor([0], dtype=t.int64)
        if feature_ids.numel() == 0
        else t.tensor([0, feature_ids.numel()], dtype=t.int64)
    )
    data = {
        "offsets": offsets,
        "feature_ids": feature_ids,
        "feature_mags": magnitudes,
    }
    first = compute_percentile_overlay_values(
        data,
        _table(),
        device="cpu",
        source_name="characterization",
    )
    second = compute_percentile_overlay_values(
        data,
        _table(),
        device="cpu",
        source_name="characterization",
    )
    assert first.dtype == t.uint8
    assert first.shape == feature_ids.shape
    assert t.equal(first, second)


def test_shared_computation_rejects_empty_base_segment() -> None:
    data = {
        "offsets": t.tensor([0, 1, 1], dtype=t.int64),
        "feature_ids": t.tensor([0], dtype=t.int32),
        "feature_mags": t.tensor([0.2], dtype=t.float32),
    }
    with pytest.raises(EmptyExtractionSegmentError, match="segment 1"):
        compute_percentile_overlay_values(
            data,
            _table(1),
            source_name="empty-segment",
        )


def test_shared_computation_rejects_shape_mismatch() -> None:
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([0], dtype=t.int32),
        "feature_mags": t.tensor([0.2, 0.3], dtype=t.float32),
    }
    with pytest.raises(ValueError, match="feature_ids shape"):
        compute_percentile_overlay_values(
            data,
            _table(1),
            source_name="shape-mismatch",
        )


@pytest.mark.parametrize("field", ["offsets", "feature_ids", "feature_mags"])
def test_shared_computation_rejects_non_tensor_fields(field: str) -> None:
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([0], dtype=t.int32),
        "feature_mags": t.tensor([0.2], dtype=t.float32),
    }
    data[field] = "not-a-tensor"
    with pytest.raises(TypeError, match="must be a tensor"):
        compute_percentile_overlay_values(data, _table(1), source_name="bad-field")


def test_shared_computation_rejects_non_rank_one_fields() -> None:
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([[0]], dtype=t.int32),
        "feature_mags": t.tensor([[0.2]], dtype=t.float32),
    }
    with pytest.raises(ValueError, match="requires rank-1"):
        compute_percentile_overlay_values(data, _table(1), source_name="bad-rank")


def test_shared_computation_preserves_results_across_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_ids = t.tensor([0, 1, 0, 1, 0], dtype=t.int32)
    magnitudes = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float16)
    table = _table(2)
    expected = percentile_module.magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cpu",
    )
    monkeypatch.setattr(percentile_module, "CPU_CHUNK_SIZE", 2)
    observed = percentile_module.magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cpu",
    )
    assert t.equal(observed, expected)


def test_pinned_cpu_quantization_boundary() -> None:
    table = t.arange(1, 100, dtype=t.float64).unsqueeze(0) * 0.1
    observed = magnitude_to_percentile_chunked(
        t.tensor([0], dtype=t.int64),
        t.tensor([1.0], dtype=t.float32),
        table,
        device="cpu",
    )
    assert observed.tolist() == [26]


def test_cpu_legacy_payload_golden_covers_extreme_lookup_semantics(
    tmp_path: Path,
) -> None:
    table = t.zeros((65_536, 99), dtype=t.float64)
    table[0] = t.arange(1, 100, dtype=t.float64) * 0.1
    table[1] = 1.0
    table[2] = t.linspace(0.0, 65_504.0, 99, dtype=t.float64)
    table[65_535] = t.arange(1, 100, dtype=t.float64)
    feature_data = {
        "offsets": t.tensor([0, 4, 8], dtype=t.int64),
        "feature_ids": t.tensor(
            [0, 0, 1, 2, 65_535, 1, 0, 2],
            dtype=t.int32,
        ),
        "feature_mags": t.tensor(
            [
                1.0,
                2**-24,
                1.0,
                65_504.0,
                float("inf"),
                float("nan"),
                float("inf"),
                0.0,
            ],
            dtype=t.float16,
        ),
    }
    values = compute_percentile_overlay_values(
        feature_data,
        table,
        device="cpu",
        source_name="legacy-golden",
    )
    assert values.tolist() == [26, 1, 254, 254, 254, 254, 254, 3]
    payload = save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES.value: values},
        tmp_path / "legacy-golden",
        zstd_level=12,
    )
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == (
        "0f6595461e475cce9671c12b5d1f3a6ed246d10acd50aa3e106fc85d6eeba388"
    )


def test_shared_computation_rejects_missing_statistics_rows() -> None:
    feature_data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.0], dtype=t.float16),
    }
    with pytest.raises(ValueError, match="statistics are missing"):
        compute_percentile_overlay_values(
            feature_data,
            t.zeros((4, 99), dtype=t.float64),
            device="cpu",
            source_name="missing-row",
        )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_pinned_cuda_boundary_differs_by_one_bin_from_cpu() -> None:
    table = t.arange(1, 100, dtype=t.float64).unsqueeze(0) * 0.1
    feature_ids = t.tensor([0], dtype=t.int64)
    magnitudes = t.tensor([1.0], dtype=t.float16)
    assert magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cpu",
    ).tolist() == [26]
    assert magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cuda",
    ).tolist() == [25]


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_chunked_path_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    feature_ids = t.tensor([0, 1, 0], dtype=t.int32)
    magnitudes = t.tensor([0.1, 0.2, 0.3], dtype=t.float16)
    table = _table(2)
    expected = magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cuda",
    )
    monkeypatch.setattr(percentile_module, "CUDA_CHUNK_SIZE", 2)
    observed = magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cuda",
    )
    assert t.equal(observed, expected)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_single_chunk_transfers_int32_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_ids = t.tensor([0, 1, 0, 1], dtype=t.int32)
    original_ids = feature_ids.clone()
    magnitudes = t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float16)
    table = _table(2)
    lookup = percentile_module.magnitude_to_percentile
    expected = percentile_module.percentile_to_uint8(
        lookup(
            feature_ids.to(t.int64).to("cuda"),
            magnitudes.to("cuda"),
            table.to("cuda"),
        )
    ).cpu()
    observed_chunks: list[t.Tensor] = []

    def observe_lookup(
        ids: t.Tensor,
        mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        observed_chunks.append(ids.detach().cpu().clone())
        assert ids.device.type == "cuda"
        assert ids.dtype == t.int32
        return lookup(ids, mags, percentile_table)

    monkeypatch.setattr(percentile_module, "magnitude_to_percentile", observe_lookup)
    observed = percentile_module.magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cuda",
    )

    assert len(observed_chunks) == 1
    assert t.equal(observed_chunks[0], feature_ids)
    assert t.equal(observed, expected)
    assert feature_ids.dtype == t.int32
    assert feature_ids.device.type == "cpu"
    assert feature_ids.shape == original_ids.shape
    assert t.equal(feature_ids, original_ids)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_multi_chunk_transfers_int32_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_ids = t.tensor([0, 1, 0, 1, 0], dtype=t.int32)
    original_ids = feature_ids.clone()
    magnitudes = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float16)
    table = _table(2)
    lookup = percentile_module.magnitude_to_percentile
    expected = percentile_module.percentile_to_uint8(
        lookup(
            feature_ids.to(t.int64).to("cuda"),
            magnitudes.to("cuda"),
            table.to("cuda"),
        )
    ).cpu()
    observed_chunks: list[t.Tensor] = []

    def observe_lookup(
        ids: t.Tensor,
        mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        observed_chunks.append(ids.detach().cpu().clone())
        assert ids.device.type == "cuda"
        assert ids.dtype == t.int32
        return lookup(ids, mags, percentile_table)

    monkeypatch.setattr(percentile_module, "CUDA_CHUNK_SIZE", 2)
    monkeypatch.setattr(percentile_module, "magnitude_to_percentile", observe_lookup)
    observed = percentile_module.magnitude_to_percentile_chunked(
        feature_ids,
        magnitudes,
        table,
        device="cuda",
    )

    assert [chunk.numel() for chunk in observed_chunks] == [2, 2, 1]
    assert t.equal(t.cat(observed_chunks), feature_ids)
    assert t.equal(observed, expected)
    assert feature_ids.dtype == t.int32
    assert feature_ids.device.type == "cpu"
    assert feature_ids.shape == original_ids.shape
    assert t.equal(feature_ids, original_ids)


@pytest.mark.parametrize("chunk_size", [2, 100])
def test_certified_cpu_lookup_preserves_chunked_uint8_values(monkeypatch, chunk_size):
    table = t.arange(1, 100, dtype=t.float64).repeat(2, 1)
    ids = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.int32)
    mags = t.tensor([0.5, 1, 49.5, 99, float("nan"), 100], dtype=t.float32)
    originals = [x.clone() for x in (ids, mags, table)]
    versions = [x._version for x in (ids, mags, table)]
    monkeypatch.setattr(percentile_module, "CPU_CHUNK_SIZE", chunk_size)
    expected = t.tensor([1, 3, 126, 254, 254, 254], dtype=t.uint8)
    assert t.equal(magnitude_to_percentile_chunked(ids, mags, table), expected)
    assert t.equal(magnitude_to_percentile_chunked(ids, mags, table, _table_is_finite=True), expected)
    for original, current in zip(originals, (ids, mags, table), strict=True):
        assert t.equal(original.contiguous().view(t.uint8), current.contiguous().view(t.uint8))
    assert versions == [x._version for x in (ids, mags, table)]


def test_borrowed_table_default_still_checks_nan_before_search():
    table = t.arange(1, 100, dtype=t.float64).reshape(1, 99)
    table[0, 49] = float("nan")
    with pytest.raises(ValueError, match="NaN in probed entries"):
        magnitude_to_percentile_chunked(t.tensor([0]), t.tensor([50.]), table)


def test_certified_chunked_lookup_keeps_interpolation_gradients(monkeypatch):
    ids = t.tensor([0, 1, 0], dtype=t.int32)
    mags = t.tensor([1.5, 20.25, 70.75], dtype=t.float64, requires_grad=True)
    table = t.arange(1, 100, dtype=t.float64).repeat(2, 1).requires_grad_()
    checked = percentile_module.magnitude_to_percentile(ids, mags, table)
    expected_grads = t.autograd.grad(checked.sum(), (mags, table))
    original = percentile_module._magnitude_to_percentile
    chunks = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        chunks.append(result)
        return result

    monkeypatch.setattr(percentile_module, "CPU_CHUNK_SIZE", 2)
    monkeypatch.setattr(percentile_module, "_magnitude_to_percentile", observe)
    encoded = magnitude_to_percentile_chunked(ids, mags, table, _table_is_finite=True)
    assert len(chunks) == 2
    assert t.equal(encoded, percentile_module.percentile_to_uint8(checked))
    actual_grads = t.autograd.grad(t.cat(chunks).sum(), (mags, table))
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        t.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decoded_empty_input_skips_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_lookup(*args, **kwargs):
        raise AssertionError("empty input must not initialize percentile lookup")

    monkeypatch.setattr(percentile_module, "magnitude_to_percentile", reject_lookup)
    monkeypatch.setattr(percentile_module, "_magnitude_to_percentile", reject_lookup)
    result = magnitude_to_percentile_chunked(
        t.empty(0, dtype=t.int32),
        t.empty(0, dtype=t.float32),
        _table(1),
        device="cuda",
        _table_is_finite=True,
    )
    assert result.shape == (0,)
    assert result.dtype == t.uint8
    assert result.device.type == "cpu"


@pytest.mark.parametrize("table_is_finite", [False, True])
def test_decoded_single_chunk_preserves_lookup_input_identity(
    monkeypatch: pytest.MonkeyPatch,
    table_is_finite: bool,
) -> None:
    ids = t.tensor([0, 1, 0], dtype=t.int32)
    magnitudes = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
    table = _table(2)
    name = "_magnitude_to_percentile" if table_is_finite else "magnitude_to_percentile"
    lookup = getattr(percentile_module, name)
    calls = 0

    def observe(feature_ids, feature_magnitudes, percentile_table, **kwargs):
        nonlocal calls
        assert feature_ids is ids
        assert feature_magnitudes is magnitudes
        assert percentile_table is table
        assert kwargs == ({"table_is_finite": True} if table_is_finite else {})
        calls += 1
        return lookup(feature_ids, feature_magnitudes, percentile_table, **kwargs)

    monkeypatch.setattr(percentile_module, name, observe)
    result = magnitude_to_percentile_chunked(
        ids, magnitudes, table, _table_is_finite=table_is_finite
    )
    assert calls == 1
    assert result.shape == ids.shape
    assert result.dtype == t.uint8
    assert result.device.type == "cpu"
