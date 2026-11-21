"""Regression checks for native quantile-row result conversion."""
from __future__ import annotations

import contextlib
import sys

import pytest
import torch as t
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import steering.compute_statistics as statistics_module


class _ScalarPercentile(int):
    """Select the existing scalar path without changing arithmetic or formatting."""


class _ScalarPercentileTuple(tuple):
    """Select the scalar path even when the requested percentile tuple is empty."""


class _FunctionTrace(TorchFunctionMode):
    def __init__(self, trace):
        self.trace = trace

    def __torch_function__(self, func, types, args: tuple[object, ...] = (), kwargs=None):
        self.trace.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


class _DispatchTrace(TorchDispatchMode):
    def __init__(self, trace):
        self.trace = trace

    def __torch_dispatch__(self, func, types, args: tuple[object, ...] = (), kwargs=None):
        self.trace.append(str(func))
        return func(*args, **(kwargs or {}))


def _inputs(*, view=False, requires_grad=False):
    ids = t.tensor([0, 0, 2, 2, 7, 7], dtype=t.int32)
    mags = t.tensor([2.0, 8.0, 3.0, 9.0, 4.0, 10.0], dtype=t.float32)
    if view:
        ids = t.stack((ids, ids), dim=-1)[:, 0]
        mags = t.stack((mags, mags), dim=-1)[:, 0]
    return ids, mags.requires_grad_(requires_grad)


def _observe(ids, mags, percentiles, *, saved=None):
    before = (ids.clone(), mags.detach().clone(), ids._version, mags._version)
    function = statistics_module.compute_per_feature_percentiles
    row_calls = []
    prior = sys.getprofile()

    def profile(frame, event, arg):
        if (
            event == "c_call"
            and frame.f_code is function.__code__
            and getattr(arg, "__name__", None) == "tolist"
        ):
            receiver = getattr(arg, "__self__", None)
            if (
                type(receiver) is t.Tensor
                and receiver.ndim == 1
                and receiver.dtype is t.float64
                and tuple(receiver.shape) == (len(percentiles),)
            ):
                row_calls.append(tuple(receiver.shape))

    def pack(value):
        if saved is not None:
            plain = value.detach().contiguous()
            saved.append((str(plain.dtype), tuple(plain.shape), plain.numpy().tobytes()))
        return value

    try:
        sys.setprofile(profile)
        with t.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = function(ids, mags, percentiles)
    finally:
        sys.setprofile(prior)
    assert sys.getprofile() is prior
    assert t.equal(ids, before[0]) and t.equal(mags.detach(), before[1])
    assert (ids._version, mags._version) == before[2:]
    return result, row_calls


@pytest.mark.parametrize("percentiles", [(1, 50, 99), (99, 50, 99, 1), ()])
@pytest.mark.parametrize("view", [False, True])
def test_native_rows_match_scalar_percentile_path(percentiles, view):
    ids, mags = _inputs(view=view)
    expected, old_rows = _observe(
        ids, mags, _ScalarPercentileTuple(_ScalarPercentile(p) for p in percentiles)
    )
    actual, rows = _observe(ids, mags, percentiles)
    assert actual == expected
    assert old_rows == []
    assert rows == [(len(percentiles),)] * 3
    assert list(actual[0]) == [0, 2, 7]
    if 50 in percentiles:
        assert [actual[0][fid].percentiles["p50"] for fid in (0, 2, 7)] == [5.0, 6.0, 7.0]
    for record in actual[0].values():
        assert record.histogram is not None
        assert sum(record.histogram) == 2


@pytest.mark.parametrize("mode_type", [_FunctionTrace, _DispatchTrace])
def test_active_modes_keep_scalar_observable_operations(mode_type):
    ids, mags = _inputs()
    plain = (1, 50, 99)
    trace = []
    with mode_type(trace):
        actual, rows = _observe(ids, mags, plain)
    expected, _ = _observe(ids, mags, tuple(_ScalarPercentile(p) for p in plain))
    assert actual == expected and rows == []
    assert trace


def test_gradient_intermediate_keeps_scalar_saved_payloads():
    ids, mags = _inputs(requires_grad=True)
    expected_saved, actual_saved = [], []
    expected, _ = _observe(
        ids, mags, tuple(_ScalarPercentile(p) for p in (1, 50, 99)), saved=expected_saved
    )
    actual, rows = _observe(ids, mags, (1, 50, 99), saved=actual_saved)
    assert actual == expected and rows == []
    assert actual_saved == expected_saved and actual_saved
    assert mags.grad is None  # Forward saved payload proof, not a backward assertion.


def test_constructor_entered_mode_is_checked_before_next_row(monkeypatch):
    ids, mags = _inputs()
    original = statistics_module.FeaturePercentiles
    trace = []
    calls = []
    with contextlib.ExitStack() as stack:
        def constructor(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            if len(calls) == 1:
                stack.enter_context(_FunctionTrace(trace))
            return result

        monkeypatch.setattr(statistics_module, "FeaturePercentiles", constructor)
        result, rows = _observe(ids, mags, (1, 50, 99))
    assert rows == [(3,)]
    assert list(result[0]) == [0, 2, 7] and len(calls) == 3
    assert [result[0][fid].percentiles["p50"] for fid in (0, 2, 7)] == [5.0, 6.0, 7.0]
    assert trace and not t.overrides.has_torch_function(())


def test_custom_round_keeps_callback_path(monkeypatch):
    ids, mags = _inputs()
    calls = []

    def custom_round(value, digits):
        calls.append((value, digits))
        return round(value + 0.125, digits)

    monkeypatch.setattr(statistics_module, "round", custom_round, raising=False)
    result, rows = _observe(ids, mags, (1, 50, 99))
    assert rows == [] and calls
    assert result[0][0].percentiles["p50"] == 5.125
