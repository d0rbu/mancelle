"""Temporarily fuse supported eager pointwise operations during GCG scoring."""

from __future__ import annotations

import os
import shutil
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial, update_wrapper
from importlib.util import find_spec
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any

import torch as t
from torch.nn.modules import module as module_hooks
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import is_in_torch_dispatch_mode
from transformers import activations
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Attention,
    SmolLM3MLP,
    SmolLM3RMSNorm,
)

if find_spec("triton") is not None:
    from scripts.eval.axbench_adv.vendored.gcg_rms_kernel import rms_postscale
    from scripts.eval.axbench_adv.vendored.gcg_rotary_kernel import rotary_pair
    from scripts.eval.axbench_adv.vendored.gcg_silu_kernel import silu_mul_
else:
    rms_postscale = None
    rotary_pair = None
    silu_mul_ = None


_MLP_FORWARDS = {kind: kind.forward for kind in (LlamaMLP, Qwen2MLP, SmolLM3MLP)}
_NORM_FORWARDS = {kind: kind.forward for kind in (LlamaRMSNorm, Qwen2RMSNorm, SmolLM3RMSNorm)}
_SILU_FORWARDS = {t.nn.SiLU: t.nn.SiLU.forward}
if hasattr(activations, "SiLUActivation"):
    _SILU_FORWARDS[activations.SiLUActivation] = activations.SiLUActivation.forward
_LINEAR_FORWARD = t.nn.Linear.forward
_ATTENTION_FORWARDS = {
    kind: kind.forward for kind in (LlamaAttention, Qwen2Attention, SmolLM3Attention)
}
_ROTARY_GLOBALS: dict[type[t.nn.Module], tuple[Any, Any]] = {
    kind: (
        getattr(forward, "__globals__", {}).get("apply_rotary_pos_emb"),
        getattr(forward, "__globals__", {}).get("rotate_half"),
    )
    for kind, forward in _ATTENTION_FORWARDS.items()
}


def _has_build_prerequisites() -> bool:
    """Match the compiler and Python include lookup used by Triton 3.1/3.3."""
    compiler = os.environ.get("CC")
    if compiler is None:
        compiler = shutil.which("gcc") or shutil.which("clang")
    if not compiler or shutil.which(compiler) is None:
        return False
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    include = Path(sysconfig.get_paths(scheme=scheme)["include"])
    return (include / "Python.h").is_file() and (include / "pyconfig.h").is_file()


def _has_hooks(module: t.nn.Module) -> bool:
    return bool(
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
    )


def _unmodified(module: t.nn.Module, forward: Any) -> bool:
    return (
        "forward" not in module.__dict__
        and getattr(module.forward, "__func__", None) is forward
        and module._compiled_call_impl is None
        and not _has_hooks(module)
    )


def _eligible(mlp: t.nn.Module) -> bool:
    forward = _MLP_FORWARDS.get(type(mlp))
    if forward is None or not _unmodified(mlp, forward):
        return False
    activation: Any = mlp.act_fn
    activation_forward = _SILU_FORWARDS.get(type(activation))
    if (
        activation_forward is None
        or not _unmodified(activation, activation_forward)
        or getattr(activation, "inplace", False)
    ):
        return False
    return all(
        type(projection) is t.nn.Linear
        and type(projection.weight) is t.nn.Parameter
        and (projection.bias is None or type(projection.bias) is t.nn.Parameter)
        and _unmodified(projection, _LINEAR_FORWARD)
        for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    )


def _scoring_forward(mlp: Any, x: t.Tensor) -> t.Tensor:
    if (
        t.is_grad_enabled()
        or t.is_autocast_enabled()
        or type(x) is not t.Tensor
        or not x.is_cuda
        or x.dtype != t.bfloat16
        or t.cuda.get_device_capability(x.device)[0] < 8
    ):
        return _MLP_FORWARDS[type(mlp)](mlp, x)
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    # Native, unhooked projections produce private outputs. Reuse the gate
    # allocation instead of creating an output that deterministic mode fills.
    assert silu_mul_ is not None
    activated = silu_mul_(gate, up)
    del gate, up
    return mlp.down_proj(activated)


def _eligible_norm(norm: t.nn.Module) -> bool:
    forward = _NORM_FORWARDS.get(type(norm))
    return (
        forward is not None
        and _unmodified(norm, forward)
        and type(norm.weight) is t.nn.Parameter
        and type(norm.variance_epsilon) in (int, float)
    )


def _scoring_norm_forward(norm: Any, hidden_states: t.Tensor) -> t.Tensor:
    if (
        t.is_grad_enabled()
        or t.is_autocast_enabled()
        or type(hidden_states) is not t.Tensor
        or not hidden_states.is_cuda
        or hidden_states.dtype != t.bfloat16
        or hidden_states.ndim == 0
        or norm.weight.device != hidden_states.device
        or norm.weight.dtype != t.bfloat16
        or norm.weight.shape != (hidden_states.shape[-1],)
        or not hidden_states.is_contiguous()
        or not norm.weight.is_contiguous()
        or t.cuda.get_device_capability(hidden_states.device)[0] < 8
    ):
        return _NORM_FORWARDS[type(norm)](norm, hidden_states)
    assert rms_postscale is not None
    return rms_postscale(hidden_states, norm.weight, norm.variance_epsilon)


def _scoring_rotary(
    native: Any,
    q: t.Tensor,
    k: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[t.Tensor, t.Tensor]:
    assert rotary_pair is not None
    result = rotary_pair(q, k, cos, sin, unsqueeze_dim)
    if result is None:
        return native(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    return result


def _rotary_attention_forward(attention: t.nn.Module) -> Any:
    forward = _ATTENTION_FORWARDS.get(type(attention))
    if (
        rotary_pair is None
        or not isinstance(forward, FunctionType)
        or not _unmodified(attention, forward)
        or forward.__closure__ is not None
        or hasattr(forward, "__wrapped__")
        or "apply_rotary_pos_emb" not in forward.__code__.co_names
    ):
        return None
    namespace = forward.__globals__
    rotary, rotate_half = _ROTARY_GLOBALS[type(attention)]
    if (
        not isinstance(rotary, FunctionType)
        or not isinstance(rotate_half, FunctionType)
        or hasattr(rotary, "__wrapped__")
        or rotary.__globals__ is not namespace
        or namespace.get("apply_rotary_pos_emb") is not rotary
        or namespace.get("rotate_half") is not rotate_half
        or not all(
            type(projection) is t.nn.Linear
            and type(projection.weight) is t.nn.Parameter
            and (projection.bias is None or type(projection.bias) is t.nn.Parameter)
            and _unmodified(projection, _LINEAR_FORWARD)
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
            )
        )
    ):
        return None
    # Keep the native attention code and all other bindings. A fresh globals
    # mapping confines rotary replacement to this instance, without modifying
    # Transformers globals or freezing their values across scoring requests.
    replacement = FunctionType(
        forward.__code__,
        {**namespace, "apply_rotary_pos_emb": partial(_scoring_rotary, rotary)},
        name=forward.__name__,
        argdefs=forward.__defaults__,
    )
    replacement.__kwdefaults__ = forward.__kwdefaults__
    return update_wrapper(replacement, forward)


@contextmanager
def fused_scoring_mlps(model: t.nn.Module) -> Iterator[None]:
    """Fuse MLP, RMS and rotary arithmetic; restore all forwards on exit."""
    if (
        silu_mul_ is None
        or not t.cuda.is_available()
        or not _has_build_prerequisites()
        or t.jit.is_tracing()
        or _get_current_function_mode() is not None
        or is_in_torch_dispatch_mode()
        or "forward" in model.__dict__
        or model._compiled_call_impl is not None
        or module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_backward_hooks
        or module_hooks._global_backward_pre_hooks
    ):
        yield
        return
    modules = tuple(model.modules())
    # An ancestor hook can inspect or change a child forward or activation.
    # Keep all such user instrumentation on the original eager path.
    if any(_has_hooks(module) for module in modules):
        yield
        return
    replacements = []
    try:
        for module in modules:
            if _eligible(module):
                module.forward = MethodType(_scoring_forward, module)
                replacements.append(module)
            elif _eligible_norm(module):
                module.forward = MethodType(_scoring_norm_forward, module)
                replacements.append(module)
            elif (forward := _rotary_attention_forward(module)) is not None:
                module.forward = MethodType(forward, module)
                replacements.append(module)
        yield
    finally:
        for module in reversed(replacements):
            del module.forward
