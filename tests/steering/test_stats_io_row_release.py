"""Consumed percentile rows must stay owned and preserve reconstruction behavior."""

from __future__ import annotations

import sys

import pytest
import torch as t
from torch.overrides import TorchFunctionMode

from steering import stats_io

SCALAR_FIELDS = (
    "pct_segments_active", "mean_mag", "mean_mag_incl_zeros", "std_mag",
    "min_mag", "max_mag", "median_mag",
)


def _fixture():
    tensors = {
        "feature_ids": t.tensor([0, 1], dtype=t.int32),
        "counts": t.tensor([3, 4]),
        "percentile_values": t.tensor([[.125, .25], [.5, .75]], dtype=t.float64),
    }
    tensors.update({key: t.tensor([.125, .5], dtype=t.float64) for key in SCALAR_FIELDS})
    return tensors, {"percentile_names": "p1,p2", "n_sae_features": "4"}


def _assert_percentiles(stats):
    assert stats.per_feature_stats["0"].percentiles == {"p1": .125, "p2": .25}
    assert stats.per_feature_stats["1"].percentiles == {"p1": .5, "p2": .75}


def test_public_statistics_round_trip(tmp_path):
    tensors, metadata = _fixture()
    original = stats_io._reconstruct_statistics(tensors, metadata)
    path = stats_io.save_statistics(original, tmp_path / "statistics")
    loaded = stats_io.load_statistics(path)
    assert loaded == original
    _assert_percentiles(loaded)


@pytest.mark.parametrize("kind", ["native", "grad", "view"])
def test_reconstruction_preserves_native_tensor_storage(kind):
    tensors, metadata = _fixture()
    if kind == "grad":
        tensors["percentile_values"].requires_grad_()
    elif kind == "view":
        tensors["percentile_values"] = t.tensor(
            [[9., .125, 8., .25], [9., .5, 8., .75]], dtype=t.float64
        )[:, 1::2]
    value = tensors["percentile_values"]
    before = value.detach().clone()
    state = (value.data_ptr(), value.storage_offset(), value.stride(), value._version, value.requires_grad)
    _assert_percentiles(stats_io._reconstruct_statistics(tensors, metadata))
    assert t.equal(value, before)
    assert state == (value.data_ptr(), value.storage_offset(), value.stride(), value._version, value.requires_grad)


@pytest.mark.parametrize("mode", [False, True])
def test_borrowed_rows_from_override_are_not_modified(mode):
    tensors, metadata = _fixture()
    target = tensors["percentile_values"]
    rows = [[.125, .25], [.5, .75]]
    original_tolist = t.Tensor.tolist

    class BorrowedRows(TorchFunctionMode):
        def __torch_function__(self, func, types, args: tuple[object, ...] = (), kwargs=None):
            if func is original_tolist and args[0] is target:
                return rows
            return func(*args, **(kwargs or {}))

    if mode:
        with BorrowedRows():
            stats = stats_io._reconstruct_statistics(tensors, metadata)
    else:
        target.tolist = lambda: rows
        stats = stats_io._reconstruct_statistics(tensors, metadata)
    _assert_percentiles(stats)
    assert rows == [[.125, .25], [.5, .75]]


@pytest.mark.parametrize("fail", [False, True])
def test_temporary_mapping_receiver_lifetime(fail):
    tensors, metadata = _fixture()
    live = []
    events = []

    class Receiver:
        def __init__(self):
            live.append("receiver")

        @property
        def tolist(self):
            events.append("property")
            if fail:
                return [].pop

            def detached_callable():
                events.append(("call", bool(live)))
                return [[float(bool(live)), .25], [.5, .75]]

            return detached_callable

        def __del__(self):
            live.remove("receiver")
            events.append("destroy")

    class TemporaryMapping(dict):
        def __getitem__(self, key):
            if key == "percentile_values":
                return Receiver()
            return super().__getitem__(key)

    tensors = TemporaryMapping(tensors)
    if fail:
        with pytest.raises(IndexError, match="pop from empty list") as caught:
            stats_io._reconstruct_statistics(tensors, metadata)
        # Holding the exception/traceback must not retain the temporary receiver.
        assert caught.value is not None
        assert events == ["property", "destroy"]
    else:
        stats = stats_io._reconstruct_statistics(tensors, metadata)
        assert stats.per_feature_stats["0"].percentiles["p1"] == 0.
        assert events == ["property", "destroy", ("call", False)]
    assert not live


@pytest.mark.parametrize("names", ["", "p1"])
def test_scalar_percentiles_preserve_empty_names_and_error_priority(names):
    tensors, metadata = _fixture()
    tensors["percentile_values"] = t.tensor(.125, dtype=t.float64)
    metadata["percentile_names"] = names
    if names:
        with pytest.raises(TypeError, match="not subscriptable"):
            stats_io._reconstruct_statistics(tensors, metadata)
    else:
        stats = stats_io._reconstruct_statistics(tensors, metadata)
        assert all(not feature.percentiles for feature in stats.per_feature_stats.values())
    del tensors["counts"]
    with pytest.raises(KeyError, match="counts"):
        stats_io._reconstruct_statistics(tensors, metadata)


def test_owned_rows_are_consumed_without_changing_result():
    tensors, metadata = _fixture()
    reconstruct = stats_io._reconstruct_statistics
    observed = []
    previous = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is reconstruct.__code__ and event == "return":
            rows = frame.f_locals["percentile_values"]
            observed.append((frame.f_locals["release_percentile_rows"], [row is None for row in rows]))
        return trace

    try:
        sys.settrace(trace)
        result = reconstruct(tensors, metadata)
    finally:
        sys.settrace(previous)
    _assert_percentiles(result)
    assert observed == [(True, [True, True])]
    assert sys.gettrace() is previous
