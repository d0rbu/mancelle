"""Rotary substitution is local to verified attention instances and scoring."""

from __future__ import annotations

import dis
import importlib.util
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from types import FunctionType
from typing import Any

import pytest
import torch as t
from transformers import LlamaConfig, Qwen2Config, SmolLM3Config
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention

from scripts.eval.axbench_adv.vendored import gcg_mlp_scoring as scoring

KINDS = (LlamaAttention, Qwen2Attention, SmolLM3Attention)
CONFIGS = {
    LlamaAttention: LlamaConfig,
    Qwen2Attention: Qwen2Config,
    SmolLM3Attention: SmolLM3Config,
}


def _attention(kind: Any = SmolLM3Attention) -> Any:
    config = CONFIGS[kind](
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return kind(config, layer_idx=0).eval()


def _inputs(model: Any, *, device: str = "cpu", dtype: t.dtype = t.float32) -> dict[str, Any]:
    x = t.randn(2, 3, 32, device=device, dtype=dtype, requires_grad=True)
    angles = t.randn(1, 3, model.head_dim, device=device, dtype=t.float32)
    return {
        "hidden_states": x,
        "position_embeddings": (angles.cos().to(dtype), angles.sin().to(dtype)),
        "attention_mask": None,
    }


@contextmanager
def _context(model: t.nn.Module, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scoring, "_has_build_prerequisites", lambda: True)
    if scoring.silu_mul_ is None:
        monkeypatch.setattr(scoring, "silu_mul_", lambda *_args: pytest.fail("unexpected MLP"))
    if scoring.rotary_pair is None:
        monkeypatch.setattr(scoring, "rotary_pair", lambda *_args: None)
    with scoring.fused_scoring_mlps(model):
        yield


def _plain_native(kind: Any) -> bool:
    return isinstance(scoring._ROTARY_GLOBALS[kind][0], FunctionType)


def _require_plain_native() -> None:
    if not _plain_native(SmolLM3Attention):
        pytest.skip("this environment exposes an external rotary kernel wrapper")


@pytest.mark.parametrize("kind", KINDS)
def test_native_code_defaults_globals_and_cpu_backward(
    kind: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _attention(kind)
    inputs = _inputs(model)
    x = inputs["hidden_states"]
    native = model.forward.__func__
    namespace = dict(native.__globals__)
    expected = model(**inputs)
    expected_gradient = t.autograd.grad(expected[0].sum(), x)[0]
    monkeypatch.setattr(scoring, "rotary_pair", lambda *_args: None)
    with _context(model, monkeypatch):
        actual_forward = model.forward.__func__
        if _plain_native(kind):
            assert actual_forward is not native
            assert actual_forward.__code__ is native.__code__
            assert actual_forward.__defaults__ is native.__defaults__
            assert actual_forward.__kwdefaults__ is native.__kwdefaults__
            assert actual_forward.__annotations__ is native.__annotations__
            assert inspect.signature(actual_forward) == inspect.signature(native)
            assert actual_forward.__globals__ is not native.__globals__
            assert actual_forward.__globals__.keys() == namespace.keys()
            assert all(
                value is actual_forward.__globals__[name]
                for name, value in namespace.items()
                if name != "apply_rotary_pos_emb"
            )
            assert not native.__closure__
            assert not any(
                instruction.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"}
                for instruction in dis.get_instructions(native)
            )
        else:
            assert actual_forward is native
        actual = model(**inputs)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        t.testing.assert_close(
            t.autograd.grad(actual[0].sum(), x)[0], expected_gradient, rtol=0, atol=0
        )
    assert "forward" not in model.__dict__
    assert model.forward.__func__ is native
    assert native.__globals__.keys() == namespace.keys()
    assert all(native.__globals__[name] is value for name, value in namespace.items())


def test_scoped_instance_leaves_other_model_and_module_global_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_plain_native()
    first, second = _attention(), _attention()
    native = first.forward.__func__
    rotary = native.__globals__["apply_rotary_pos_emb"]
    calls = []

    def count(*args: Any) -> None:
        calls.append(args[0].shape)
        return None

    monkeypatch.setattr(scoring, "rotary_pair", count)
    with t.no_grad(), _context(first, monkeypatch):
        first(**_inputs(first))
        assert len(calls) == 1
        second(**_inputs(second))
        assert len(calls) == 1
        assert second.forward.__func__ is native
        assert native.__globals__["apply_rotary_pos_emb"] is rotary
    assert first.forward.__func__ is second.forward.__func__ is native


def test_nested_context_and_failure_restore_all_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_plain_native()
    first, second = _attention(), _attention()
    model = t.nn.ModuleList([first, second])
    with pytest.raises(RuntimeError, match="body failure"), _context(model, monkeypatch):
        outer = first.forward
        with scoring.fused_scoring_mlps(model):
            assert first.forward is outer
        assert first.forward is outer
        raise RuntimeError("body failure")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__
    original = scoring._rotary_attention_forward

    def fail_on_second(module: t.nn.Module) -> Any:
        if module is second:
            raise RuntimeError("entry failure")
        return original(module)

    monkeypatch.setattr(scoring, "_rotary_attention_forward", fail_on_second)
    with pytest.raises(RuntimeError, match="entry failure"), _context(model, monkeypatch):
        pytest.fail("entry must fail before yielding")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__


@pytest.mark.parametrize("location", ["self", "q_proj", "k_proj", "v_proj", "o_proj"])
@pytest.mark.parametrize("change", ["instance", "class", "compiled", "hook"])
def test_custom_forwards_and_hooks_keep_native_attention(
    location: str, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _attention()
    owner = model if location == "self" else getattr(model, location)
    inputs = _inputs(model)
    original = owner.forward
    handle = None
    if change == "instance":
        owner.forward = original
    elif change == "class":
        native = type(owner).forward
        monkeypatch.setattr(
            type(owner), "forward", lambda self, *args, **kwargs: native(self, *args, **kwargs)
        )
    elif change == "compiled":
        owner._compiled_call_impl = original
    else:
        handle = owner.register_forward_pre_hook(lambda *_args: None)
    expected_forward = model.forward
    try:
        expected = model(**inputs)
        with _context(t.nn.ModuleList([model]), monkeypatch):
            assert model.forward == expected_forward
            t.testing.assert_close(model(**inputs), expected, rtol=0, atol=0)
        assert model.forward == expected_forward
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("name", ["apply_rotary_pos_emb", "rotate_half"])
def test_changed_rotary_globals_preserve_native_dispatch(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _attention()
    native = model.forward.__func__
    original = native.__globals__[name]
    calls = []

    def replacement(*args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setitem(native.__globals__, name, replacement)
    with _context(model, monkeypatch):
        assert model.forward.__func__ is native
        model(**_inputs(model))
    assert calls
    assert native.__globals__[name] is replacement


def test_global_snapshot_refreshes_between_scoring_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_plain_native()
    model = _attention()
    native = model.forward.__func__
    original = native.__globals__["eager_attention_forward"]
    calls = []

    def replacement(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)

    with _context(model, monkeypatch):
        assert model.forward.__func__.__globals__["eager_attention_forward"] is original
    monkeypatch.setitem(native.__globals__, "eager_attention_forward", replacement)
    with _context(model, monkeypatch):
        assert model.forward.__func__.__globals__["eager_attention_forward"] is replacement
        model(**_inputs(model))
    assert calls == [True]


@pytest.mark.parametrize(
    "case", ["subclass", "projection", "weight", "bias", "wrapped_rotary", "missing_kernel"]
)
def test_unsupported_modules_are_not_rebound(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomAttention(SmolLM3Attention):
        pass

    class CustomLinear(t.nn.Linear):
        pass

    class CustomParameter(t.nn.Parameter):
        pass

    model = _attention()
    if case == "subclass":
        model = CustomAttention(model.config, layer_idx=0)
    elif case == "projection":
        model.q_proj = CustomLinear(32, 32)
    elif case == "weight":
        model.q_proj.weight = CustomParameter(model.q_proj.weight.detach())
    elif case == "bias":
        model.q_proj.bias = CustomParameter(t.zeros(32))
    elif case == "wrapped_rotary":
        native, half = scoring._ROTARY_GLOBALS[type(model)]
        wrapper = t.nn.Identity()
        monkeypatch.setitem(scoring._ROTARY_GLOBALS, type(model), (wrapper, half))
        monkeypatch.setitem(type(model).forward.__globals__, "apply_rotary_pos_emb", wrapper)
        assert native is not wrapper
    else:
        monkeypatch.setattr(scoring, "rotary_pair", None)
    assert scoring._rotary_attention_forward(model) is None


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_cpu_rotary_fallback_keeps_native_values_and_gradients(grad_enabled: bool) -> None:
    if scoring.rotary_pair is None:
        pytest.skip("requires Triton import")
    native = scoring._ROTARY_GLOBALS[SmolLM3Attention][0]
    q = t.randn(2, 4, 3, 8, requires_grad=True)
    k = t.randn(2, 2, 3, 8, requires_grad=True)
    cos, sin = t.randn(1, 3, 8), t.randn(1, 3, 8)
    with t.set_grad_enabled(grad_enabled):
        expected = native(q, k, cos, sin)
        actual = scoring._scoring_rotary(native, q=q, k=k, cos=cos, sin=sin)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        if grad_enabled:
            for i, value in enumerate((q, k)):
                t.testing.assert_close(
                    t.autograd.grad(actual[i].sum(), value)[0],
                    t.autograd.grad(expected[i].sum(), value)[0],
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("missing", ["triton", "legacy_silu_class"])
def test_optional_import_paths_preserve_native_execution(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_find_spec = importlib.util.find_spec
    if missing == "triton":
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: None if name == "triton" else original_find_spec(name),
        )
    else:
        monkeypatch.delattr(scoring.activations, "SiLUActivation", raising=False)
    spec = importlib.util.spec_from_file_location("_optional_gcg_scoring", scoring.__file__)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    if missing == "triton":
        assert isolated.silu_mul_ is isolated.rms_postscale is isolated.rotary_pair is None
    else:
        assert isolated._SILU_FORWARDS == {t.nn.SiLU: t.nn.SiLU.forward}
    model = _attention()
    native = model.forward.__func__
    inputs = _inputs(model)
    expected = model(**inputs)
    expected_gradient = t.autograd.grad(expected[0].sum(), inputs["hidden_states"])[0]
    with isolated.fused_scoring_mlps(model):
        actual = model(**inputs)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        t.testing.assert_close(
            t.autograd.grad(actual[0].sum(), inputs["hidden_states"])[0],
            expected_gradient,
            rtol=0,
            atol=0,
        )
    assert "forward" not in model.__dict__ and model.forward.__func__ is native
