"""Scoring fusion must preserve eager numerics and caller-owned modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from weakref import ref

import pytest
import torch as t
import torch.nn.functional as F
from torch.nn.modules import module as module_hooks
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import LlamaConfig, Qwen2Config, SmolLM3Config
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.smollm3.modeling_smollm3 import SmolLM3MLP

from scripts.eval.axbench_adv.vendored import gcg_mlp_scoring as scoring


def _mlp(kind: type[t.nn.Module] = SmolLM3MLP) -> Any:
    configs: dict[type[t.nn.Module], Any] = {
        LlamaMLP: LlamaConfig,
        Qwen2MLP: Qwen2Config,
        SmolLM3MLP: SmolLM3Config,
    }
    config = configs[kind]
    return kind(
        config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_act="silu",
        )
    )


@contextmanager
def _context(model: t.nn.Module, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Exercise restoration/selection on CPU-only test hosts without launching a kernel.
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scoring, "_has_build_prerequisites", lambda: True)
    if scoring.silu_mul_ is None:
        monkeypatch.setattr(scoring, "silu_mul_", lambda a, b: F.silu(a) * b)
    with scoring.fused_scoring_mlps(model):
        yield


@pytest.mark.parametrize("kind", [LlamaMLP, Qwen2MLP, SmolLM3MLP])
def test_cpu_fallback_and_forward_restoration(
    kind: type[t.nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _mlp(kind)
    x = t.randn(2, 3, 16)
    original = model.forward.__func__
    expected = model(x)
    with t.no_grad(), _context(model, monkeypatch):
        assert "forward" in model.__dict__
        t.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert "forward" not in model.__dict__
    assert model.forward.__func__ is original


@pytest.mark.parametrize("location", ["self", "act_fn", "gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize(
    "registration",
    [
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
    ],
)
def test_local_hooks_keep_the_original_forward(
    location: str, registration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _mlp()
    owner = model if location == "self" else getattr(model, location)
    handle = getattr(owner, registration)(lambda *args: None)
    try:
        assert not scoring._eligible(model)
        with _context(model, monkeypatch):
            assert "forward" not in model.__dict__
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_global_hooks_keep_the_original_forward(
    registration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _mlp()
    handle = getattr(module_hooks, registration)(lambda *args: None)
    try:
        with _context(model, monkeypatch):
            assert "forward" not in model.__dict__
    finally:
        handle.remove()


def test_parent_hooks_observe_original_children(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _mlp()
    parent = t.nn.Sequential(child)
    seen = []

    def inspect_child(_module: t.nn.Module, _args: Any) -> None:
        seen.append(child.forward.__func__)
        assert "forward" not in child.__dict__

    handle = parent.register_forward_pre_hook(inspect_child)
    try:
        with t.no_grad(), _context(parent, monkeypatch):
            parent(t.randn(2, 3, 16))
        assert seen == [type(child).forward]
    finally:
        handle.remove()


@pytest.mark.parametrize("location", ["self", "act_fn", "gate_proj", "up_proj", "down_proj"])
def test_instance_forward_overrides_are_preserved(
    location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _mlp()
    owner = model if location == "self" else getattr(model, location)
    original = owner.forward
    owner.forward = original
    with _context(model, monkeypatch):
        assert owner.forward is original
        if owner is not model:
            assert "forward" not in model.__dict__
    assert owner.forward is original


@pytest.mark.parametrize("location", ["self", "act_fn", "gate_proj"])
def test_class_forward_overrides_are_preserved(
    location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _mlp()
    owner = model if location == "self" else getattr(model, location)
    original = type(owner).forward

    def replacement(self: t.nn.Module, x: t.Tensor) -> t.Tensor:
        return original(self, x)

    monkeypatch.setattr(type(owner), "forward", replacement)
    assert not scoring._eligible(model)
    with _context(model, monkeypatch):
        assert "forward" not in model.__dict__
        assert owner.forward.__func__ is replacement


def test_other_activations_and_custom_projections_are_not_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomLinear(t.nn.Linear):
        pass

    models = [_mlp() for _ in range(4)]
    models[0].act_fn = t.nn.GELU()
    models[1].act_fn = t.nn.SiLU(inplace=True)
    models[2].gate_proj = CustomLinear(16, 32)
    models[3]._compiled_call_impl = lambda *args: None
    for model in models:
        assert not scoring._eligible(model)
        with _context(model, monkeypatch):
            assert "forward" not in model.__dict__


def test_restores_after_body_and_partial_entry_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _mlp(), _mlp()
    model = t.nn.Sequential(first, second)
    with pytest.raises(RuntimeError, match="body failure"), _context(model, monkeypatch):
        assert "forward" in first.__dict__ and "forward" in second.__dict__
        raise RuntimeError("body failure")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__
    original_eligible = scoring._eligible

    def fail_on_second(module: t.nn.Module) -> bool:
        if module is second:
            raise RuntimeError("entry failure")
        return original_eligible(module)

    monkeypatch.setattr(scoring, "_eligible", fail_on_second)
    with pytest.raises(RuntimeError, match="entry failure"), _context(model, monkeypatch):
        pytest.fail("entry should fail before yielding")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__


def test_nested_context_and_missing_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _mlp()
    with _context(model, monkeypatch):
        outer = model.forward
        with scoring.fused_scoring_mlps(model):
            assert model.forward is outer
        assert model.forward is outer
    assert "forward" not in model.__dict__
    monkeypatch.setattr(scoring, "silu_mul_", None)
    with scoring.fused_scoring_mlps(model):
        assert "forward" not in model.__dict__


def test_no_cuda_keeps_original_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _mlp()
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    with scoring.fused_scoring_mlps(model):
        assert "forward" not in model.__dict__


@pytest.mark.parametrize("missing", ["Python.h", "pyconfig.h", "compiler"])
def test_missing_build_prerequisites_preserve_eager(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("Python.h", "pyconfig.h"):
        if name != missing:
            (tmp_path / name).touch()
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(
        scoring.shutil, "which", lambda _name: None if missing == "compiler" else "/cc"
    )
    monkeypatch.setattr(
        scoring.sysconfig, "get_paths", lambda **_kwargs: {"include": str(tmp_path)}
    )
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    assert not scoring._has_build_prerequisites()
    model = _mlp()
    x = t.randn(2, 3, 16)
    expected = model(x)
    with scoring.fused_scoring_mlps(model):
        assert "forward" not in model.__dict__
        t.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("compiler", ["gcc", "clang", "custom"])
def test_build_prerequisites_follow_triton_lookup(
    compiler: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("Python.h", "pyconfig.h"):
        (tmp_path / name).touch()
    executable = tmp_path / compiler
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CC", raising=False)
    if compiler == "custom":
        monkeypatch.setenv("CC", str(executable))
    monkeypatch.setattr(scoring.sysconfig, "get_default_scheme", lambda: "posix_local")
    schemes = []

    def paths(*, scheme: str) -> dict[str, str]:
        schemes.append(scheme)
        return {"include": str(tmp_path)}

    monkeypatch.setattr(scoring.sysconfig, "get_paths", paths)
    assert scoring._has_build_prerequisites()
    assert schemes == ["posix_prefix"]
    monkeypatch.setenv("CC", "missing-compiler")
    assert not scoring._has_build_prerequisites()
    monkeypatch.setenv("CC", "")
    assert not scoring._has_build_prerequisites()


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_custom_parameters_keep_original_projection_semantics(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CustomParameter(t.nn.Parameter):
        pass

    model = _mlp()
    value = model.gate_proj.weight.detach() if name == "weight" else t.zeros(32)
    setattr(model.gate_proj, name, CustomParameter(value))
    assert not scoring._eligible(model)
    with _context(model, monkeypatch):
        assert "forward" not in model.__dict__


@pytest.mark.parametrize("mode_type", [TorchFunctionMode, TorchDispatchMode])
def test_torch_modes_observe_original_forward(
    mode_type: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Observe(mode_type):
        def __torch_function__(
            self, func: Any, _types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            calls.append(func)
            return func(*args, **(kwargs or {}))

        def __torch_dispatch__(
            self, func: Any, _types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            calls.append(func)
            return func(*args, **(kwargs or {}))

    model = _mlp()
    x = t.randn(2, 3, 16)
    expected = model(x)
    with Observe(), _context(model, monkeypatch):
        assert "forward" not in model.__dict__
        actual = model(x)
    assert calls
    t.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_jit_trace_uses_original_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []

    class Traced(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _mlp()

        def forward(self, x: t.Tensor) -> t.Tensor:
            with scoring.fused_scoring_mlps(self.mlp):
                if t.jit.is_tracing():
                    observed.append("forward" not in self.mlp.__dict__)
                return self.mlp(x)

    model = Traced()
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scoring, "_has_build_prerequisites", lambda: True)
    traced = t.jit.trace(model, t.randn(2, 3, 16), check_trace=False)
    x = t.randn(2, 3, 16)
    assert observed == [True]
    t.testing.assert_close(traced(x), model.mlp(x), rtol=0, atol=0)


def test_kernel_requires_disabled_autograd_and_cuda_operands() -> None:
    if scoring.silu_mul_ is None:
        pytest.skip("requires Triton import")
    empty = t.empty(0, dtype=t.bfloat16)
    with t.enable_grad(), pytest.raises(RuntimeError, match="disabled autograd"):
        scoring.silu_mul_(empty, empty)
    with t.no_grad(), pytest.raises(ValueError, match="BF16 CUDA operands"):
        scoring.silu_mul_(empty, empty)


def _require_cuda() -> None:
    if (
        not t.cuda.is_available()
        or scoring.silu_mul_ is None
        or not scoring._has_build_prerequisites()
    ):
        pytest.skip("requires CUDA, Triton, and compiler/Python headers")


@pytest.mark.slow
@pytest.mark.parametrize("kind", [LlamaMLP, Qwen2MLP, SmolLM3MLP])
def test_cuda_mlp_matches_eager_and_restores_gradients(
    kind: type[t.nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cuda()
    model = _mlp(kind).cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    original_kernel = scoring.silu_mul_
    calls = []

    def record(gate: t.Tensor, up: t.Tensor) -> t.Tensor:
        calls.append(gate.shape)
        assert original_kernel is not None
        return original_kernel(gate, up)

    monkeypatch.setattr(scoring, "silu_mul_", record)
    with t.no_grad():
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            actual = model(x)
    assert len(calls) == 1
    t.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls.clear()
    x.requires_grad_(True)
    expected_gradient = t.autograd.grad(model(x).float().sum(), x)[0]
    with scoring.fused_scoring_mlps(model):
        actual_gradient = t.autograd.grad(model(x).float().sum(), x)[0]
    assert not calls
    t.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("kind", [LlamaMLP, Qwen2MLP, SmolLM3MLP])
def test_cuda_projection_biases_and_native_silu(kind: type[t.nn.Module]) -> None:
    _require_cuda()
    model = _mlp(kind).cuda().bfloat16()
    model.act_fn = t.nn.SiLU()
    for projection in (model.gate_proj, model.up_proj, model.down_proj):
        projection.bias = t.nn.Parameter(
            t.randn(projection.out_features, device="cuda", dtype=t.bfloat16)
        )
    assert scoring._eligible(model)
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    with t.no_grad():
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            actual = model(x)
    t.testing.assert_close(actual, expected, rtol=0, atol=0)
    x.requires_grad_(True)
    expected_gradient = t.autograd.grad(model(x).float().sum(), x)[0]
    with scoring.fused_scoring_mlps(model):
        actual_gradient = t.autograd.grad(model(x).float().sum(), x)[0]
    t.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.slow
def test_tensor_subclasses_retain_eager_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class Subclass(t.Tensor):
        pass

    _require_cuda()
    model = _mlp().cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16).as_subclass(Subclass)
    before = x.clone()
    monkeypatch.setattr(scoring, "silu_mul_", lambda *args: pytest.fail("unexpected fusion"))
    with t.no_grad():
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            actual = model(x)
    assert type(actual) is Subclass
    t.testing.assert_close(actual, expected, rtol=0, atol=0)
    t.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [t.float32, t.float16])
def test_other_cuda_dtypes_preserve_eager(dtype: t.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    model = _mlp().cuda().to(dtype)
    x = t.randn(2, 3, 16, device="cuda", dtype=dtype)
    monkeypatch.setattr(scoring, "silu_mul_", lambda *args: pytest.fail("unexpected fusion"))
    with t.no_grad():
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            t.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.slow
def test_pre_ampere_capability_keeps_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    model = _mlp().cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (7, 5))
    monkeypatch.setattr(scoring, "silu_mul_", lambda *args: pytest.fail("unexpected fusion"))
    with t.no_grad():
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            t.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.slow
def test_cuda_autocast_preserves_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    model = _mlp().cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    monkeypatch.setattr(scoring, "silu_mul_", lambda *args: pytest.fail("unexpected fusion"))
    with t.no_grad(), t.autocast("cuda", dtype=t.float16):
        expected = model(x)
        with scoring.fused_scoring_mlps(model):
            t.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.slow
def test_projection_hook_mutations_retain_eager_order() -> None:
    _require_cuda()
    model = _mlp().cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    captured = []
    gate_hook = model.gate_proj.register_forward_hook(lambda _m, _a, out: captured.append(out))

    def mutate_gate(_module: t.nn.Module, _args: Any) -> None:
        captured[-1].zero_()

    up_hook = model.up_proj.register_forward_pre_hook(mutate_gate)
    try:
        with t.no_grad():
            expected = model(x)
            with scoring.fused_scoring_mlps(model):
                actual = model(x)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        gate_hook.remove()
        up_hook.remove()


@pytest.mark.slow
def test_only_the_reused_gate_survives_until_down_projection() -> None:
    _require_cuda()
    model = _mlp().cuda().bfloat16()
    x = t.randn(2, 3, 16, device="cuda", dtype=t.bfloat16)
    raw_outputs = []
    observed = []

    def remember(_module: t.nn.Module, _args: Any, output: t.Tensor) -> None:
        raw_outputs.append(ref(output))

    def check_lifetime(_module: t.nn.Module, _args: Any) -> None:
        observed.append([value() is None for value in raw_outputs])
        assert _args[0] is raw_outputs[0]()

    handles = [
        model.gate_proj.register_forward_hook(remember),
        model.up_proj.register_forward_hook(remember),
        model.down_proj.register_forward_pre_hook(check_lifetime),
    ]
    try:
        # Directly test the private forward's allocation lifetime; ordinary
        # context dispatch deliberately falls back when these hooks are present.
        with t.no_grad():
            scoring._scoring_forward(model, x)
        assert observed == [[False, True]]
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.slow
@pytest.mark.parametrize("case", ["dtype", "shape", "strides", "device"])
def test_kernel_rejects_unsupported_operands(case: str) -> None:
    _require_cuda()
    assert scoring.silu_mul_ is not None
    gate = t.ones(8, device="cuda", dtype=t.bfloat16)
    up = t.ones_like(gate)
    if case == "dtype":
        up = up.float()
    elif case == "shape":
        up = up[:7]
    elif case == "strides":
        gate, up = gate[::2], up[::2]
    else:
        up = up.cpu()
    with t.no_grad(), pytest.raises(ValueError, match="BF16 CUDA operands"):
        scoring.silu_mul_(gate, up)


@pytest.mark.slow
def test_kernel_matches_every_bf16_operand_pair() -> None:
    _require_cuda()
    assert scoring.silu_mul_ is not None
    bits = t.arange(65536, dtype=t.int32).to(t.int16).view(t.bfloat16).cuda()
    up = bits.unsqueeze(0).expand(128, -1).contiguous()
    with t.no_grad():
        for begin in range(0, 65536, 128):
            gate = bits[begin : begin + 128].unsqueeze(1).expand(-1, 65536).contiguous()
            expected = F.silu(gate) * up
            actual = scoring.silu_mul_(gate, up)
            both_nan = t.isnan(expected) & t.isnan(actual)
            assert bool(((expected.view(t.int16) == actual.view(t.int16)) | both_nan).all()), begin


@pytest.mark.slow
@pytest.mark.parametrize("count", [0, 1, 17, 127, 129, 1023, 1024, 1025])
def test_kernel_tail_masks(count: int) -> None:
    _require_cuda()
    assert scoring.silu_mul_ is not None
    storage = t.randn(count + 32, device="cuda", dtype=t.bfloat16)
    gate = storage[16 : 16 + count]
    up = t.randn_like(gate)
    if count:
        gate[-1], up[-1] = 1, 3
    storage_before, up_before = storage.clone(), up.clone()
    expected = F.silu(gate) * up
    with t.no_grad():
        actual = scoring.silu_mul_(gate, up)
    assert actual is gate
    t.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert t.equal(storage[:16].view(t.int16), storage_before[:16].view(t.int16))
    assert t.equal(storage[16 + count :].view(t.int16), storage_before[16 + count :].view(t.int16))
    assert t.equal(up.view(t.int16), up_before.view(t.int16))


@pytest.mark.slow
def test_kernel_offsets_beyond_signed_int32() -> None:
    _require_cuda()
    assert scoring.silu_mul_ is not None
    count = (1 << 31) + 1025
    free, _total = t.cuda.mem_get_info()
    if free < count * 4 + (1 << 30):
        pytest.skip("requires 9 GiB free for two large BF16 operands")
    gate = t.ones(count, device="cuda", dtype=t.bfloat16)
    up = t.full_like(gate, 3)
    indices = t.tensor([0, 1023, (1 << 30) - 1, (1 << 31) - 1, 1 << 31, count - 1], device="cuda")
    gate[indices] = t.arange(indices.numel(), device="cuda", dtype=t.bfloat16) - 2
    expected = F.silu(gate[indices]) * up[indices]
    with t.no_grad():
        actual = scoring.silu_mul_(gate, up)
    assert actual is gate
    t.testing.assert_close(actual[indices], expected, rtol=0, atol=0)
    assert bool(up[indices].eq(3).all())
