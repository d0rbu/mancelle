"""Differential checks for the exact reducer's percentile-row conversion.

The tuple subclass selects the literal scalar path while keeping the requested
values unchanged. The ignored qualification harness also runs these scenarios
against the complete, pinned pre-change module.
"""

from __future__ import annotations

import builtins
import contextlib

import pytest
import torch as t
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import steering.exact_statistics as exact


class _ScalarTuple(tuple):
    """Request the unchanged scalar conversion, including for an empty tuple."""


class _FunctionTrace(TorchFunctionMode):
    def __init__(self, trace):
        self.trace = trace

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.trace.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


class _DispatchTrace(TorchDispatchMode):
    def __init__(self, trace):
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.trace.append(str(func))
        return func(*args, **(kwargs or {}))


def _inputs(module=None, *, extreme=False):
    module = exact if module is None else module
    ids = t.tensor([0, 0, 0, 1, 1, 1], dtype=t.int32)
    if extreme:
        mags = t.tensor([1, 2, 0x3C00, 1, 0x3C00, 0x7BFF], dtype=t.int16)
        mags = mags.view(t.float16).to(t.float32)
    else:
        mags = t.tensor([1, 3, 9, 2, 4, 8], dtype=t.float32)
    reducer = module.ExactMagnitudeCounts(
        t.zeros((2, module.POSITIVE_FINITE_F16_COUNT), dtype=t.int64)
    )
    reducer.add(ids, mags)
    return reducer, {
        "occurrence": t.tensor([3, 3], dtype=t.int64),
        "mag_subsample": mags,
        "global_mag_min": float(mags.min().item()),
        "n_hist_bins": 4,
    }


def _normalize(result):
    records, edges = result
    return {
        fid: (list(row.percentiles.items()), row.median, row.histogram)
        for fid, row in records.items()
    }, edges


def _observe(reducer, kwargs, percentiles):
    counts = reducer.counts.clone()
    occurrence = kwargs["occurrence"].clone()
    sample = kwargs["mag_subsample"].clone()
    versions = (reducer.counts._version, kwargs["occurrence"]._version,
                kwargs["mag_subsample"]._version)
    try:
        result = ("ok", _normalize(reducer.finalize(percentiles=percentiles, **kwargs)))
    except (ValueError, TypeError, RuntimeError, IndexError) as exc:
        result = ("error", type(exc).__name__, str(exc))
    assert t.equal(reducer.counts, counts)
    assert t.equal(kwargs["occurrence"], occurrence)
    assert t.equal(kwargs["mag_subsample"], sample)
    assert versions == (reducer.counts._version, kwargs["occurrence"]._version,
                        kwargs["mag_subsample"]._version)
    return result


@pytest.mark.parametrize("percentiles", [(), (0, 100), (99, 1, 99), (25, 75), (100, 50, 0)])
@pytest.mark.parametrize("extreme", [False, True])
def test_native_rows_match_scalar_values_order_and_inputs(percentiles, extreme):
    reducer, kwargs = _inputs(extreme=extreme)
    actual = _observe(reducer, kwargs, percentiles)
    assert actual == _observe(reducer, kwargs, _ScalarTuple(percentiles))
    assert actual[0] == "ok"
    for fid, (items, median, histogram) in actual[1][0].items():
        assert [key for key, _ in items] == list(dict.fromkeys(f"p{p}" for p in percentiles))
        assert sum(histogram) == 3
        if not extreme:
            assert median == (3.0 if fid == 0 else 4.0)


@pytest.mark.parametrize("mode_type", [_FunctionTrace, _DispatchTrace])
@pytest.mark.parametrize("between_rows", [False, True])
def test_live_modes_preserve_scalar_operations(monkeypatch, mode_type, between_rows):
    def run(percentiles):
        reducer, kwargs = _inputs()
        trace, calls = [], []
        with monkeypatch.context() as patch, contextlib.ExitStack() as stack:
            constructor = exact.ExactFeaturePercentiles

            def construct(*args, **kw):
                value = constructor(*args, **kw)
                calls.append(value)
                if between_rows and len(calls) == 1:
                    stack.enter_context(mode_type(trace))
                return value

            patch.setattr(exact, "ExactFeaturePercentiles", construct)
            if not between_rows:
                stack.enter_context(mode_type(trace))
            result = reducer.finalize(percentiles=percentiles, **kwargs)
        return _normalize(result), trace

    actual, trace = run((1, 50, 99))
    expected, old_trace = run(_ScalarTuple((1, 50, 99)))
    assert actual == expected and trace == old_trace and trace


@pytest.mark.parametrize("method", ["__getitem__", "item", "tolist"])
def test_method_replacement_between_rows_preserves_callbacks(monkeypatch, method):
    def run(percentiles):
        reducer, kwargs = _inputs()
        trace, calls = [], []
        original_method = getattr(t.Tensor, method)
        constructor = exact.ExactFeaturePercentiles
        with monkeypatch.context() as patch:
            def replacement(receiver, *args, **kw):
                trace.append((method, tuple(receiver.shape)))
                return original_method(receiver, *args, **kw)

            def construct(*args, **kw):
                result = constructor(*args, **kw)
                calls.append(result)
                if len(calls) == 1:
                    patch.setattr(t.Tensor, method, replacement)
                return result

            patch.setattr(exact, "ExactFeaturePercentiles", construct)
            result = reducer.finalize(percentiles=percentiles, **kwargs)
        return _normalize(result), trace

    assert run((1, 50, 99)) == run(_ScalarTuple((1, 50, 99)))


@pytest.mark.parametrize("trigger", ["format", "round"])
def test_constructor_is_resolved_before_argument_callbacks(monkeypatch, trigger):
    def run():
        reducer, kwargs = _inputs()
        events = []
        constructor = exact.ExactFeaturePercentiles
        with monkeypatch.context() as patch:
            def new_constructor(*args, **kw):
                events.append("new")
                return constructor(*args, **kw)

            def rebind():
                events.append("rebind")
                patch.setattr(exact, "ExactFeaturePercentiles", new_constructor)

            class Percentile(int):
                def __format__(self, spec):
                    rebind()
                    return format(int(self), spec)

            def custom_round(value, digits):
                rebind()
                return builtins.round(value + 0.125, digits)

            def old_constructor(*args, **kw):
                events.append("old")
                return constructor(*args, **kw)

            patch.setattr(exact, "ExactFeaturePercentiles", old_constructor)
            if trigger == "round":
                patch.setattr(exact, "round", custom_round, raising=False)
            pcts = (Percentile(25), Percentile(75)) if trigger == "format" else (25, 75)
            result = reducer.finalize(percentiles=pcts, **kwargs)
        return _normalize(result), events

    result, events = run()
    assert [event for event in events if event != "rebind"] == ["old", "new"]
    assert events[0] == "rebind"
    if trigger == "round":
        assert result[0][0][1] == 3.125


def test_setting_self_percentiles_does_not_replace_local_request(monkeypatch):
    reducer, kwargs = _inputs()
    constructor = exact.ExactFeaturePercentiles
    calls = []

    class Percentile(int):
        def __format__(self, spec):
            raise AssertionError("self.percentiles is not the finalize argument")

    def construct(*args, **kw):
        calls.append(kw)
        reducer.percentiles = (Percentile(50),)
        return constructor(*args, **kw)

    monkeypatch.setattr(exact, "ExactFeaturePercentiles", construct)
    result = reducer.finalize(percentiles=(25, 75), **kwargs)
    assert len(calls) == 2
    assert all(list(row.percentiles) == ["p25", "p75"] for row in result[0].values())


@pytest.mark.parametrize("kind", ["subclass", "attribute", "method", "rank", "gradient", "functional"])
def test_nonordinary_intermediate_rows_match_scalar_path(monkeypatch, kind):
    class Row(t.Tensor):
        def tolist(self):
            raise AssertionError("subclass must retain scalar reads")

    def run(percentiles):
        reducer, kwargs = _inputs()
        original_add = t.Tensor.__add__
        injected = []

        def add(receiver, other):
            result = original_add(receiver, other)
            if result.dtype == t.float64 and tuple(result.shape) == (3,):
                injected.append(True)
                if kind == "subclass":
                    result = result.as_subclass(Row)
                elif kind == "attribute":
                    result.sentinel = 1  # ty: ignore[unresolved-attribute]  # Deliberate dynamic fixture.
                elif kind == "method":
                    def forbidden():
                        raise AssertionError("instance override must retain scalar reads")
                    result.tolist = forbidden  # ty: ignore[invalid-assignment]  # Deliberate override.
                elif kind == "rank":
                    result = result.reshape(1, -1)
                elif kind == "gradient":
                    result.requires_grad_(True)
                else:
                    result = t._to_functional_tensor(result)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(t.Tensor, "__add__", add)
            result = _observe(reducer, kwargs, percentiles)
        assert injected
        return result

    actual = run((1, 50, 99))
    assert actual == run(_ScalarTuple((1, 50, 99)))
    assert actual[0] == ("error" if kind == "rank" else "ok")


@pytest.mark.parametrize("invalid", ["width", "negative", "total", "dtype", "percentile", "bins"])
def test_invalid_paths_match_literal_scalar_errors(invalid):
    reducer, kwargs = _inputs()
    percentiles = (1, 50, 99)
    if invalid == "width":
        kwargs["occurrence"] = kwargs["occurrence"][:1]
    elif invalid == "negative":
        kwargs["occurrence"] = t.tensor([-1, 7], dtype=t.int64)
    elif invalid == "total":
        kwargs["occurrence"] = t.tensor([3, 2], dtype=t.int64)
    elif invalid == "dtype":
        kwargs["occurrence"] = kwargs["occurrence"].float()
    elif invalid == "percentile":
        percentiles = (101,)
    else:
        kwargs["n_hist_bins"] = 0
    actual = _observe(reducer, kwargs, percentiles)
    assert actual[0] == "error"
    assert actual == _observe(reducer, kwargs, _ScalarTuple(percentiles))
