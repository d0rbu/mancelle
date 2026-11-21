"""Finite native-state guard for optional group-local RLVR hidden replay.

No forwards, mutations, hooks or persistent allocations are installed here.
Capture before native computation; match after successful capture and before every hit.
Only ordinary native CUDA head-only models are admitted. Model .data writes,
unregistered independently versioned aliases and concurrent mutation are outside
the published frozen-staging contract; mutable prompt contents are the cache's
separate responsibility. Head values/versions stay live, while its structure,
storage/layout and trainable scope must stay fixed.
Normal compiler/mask-cache population is operational state. Deliberate external
writes to those cache contents, compiled closures or the Torch/F namespace are
unsupported; this is a finite native guard, not a Python runtime sandbox.
Module construction/deserialization and Dynamo generation tagging do not affect
the existing frozen model forward and are excluded from class comparisons.
Installed framework source files are neither read nor authenticated.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, fields
from types import FunctionType, ModuleType
from typing import Any

import torch as t

import nanochat.flash_attention as flash
import nanochat.gpt as gpt
from steering.conditional.encoder_decoder import config as configs
from steering.conditional.encoder_decoder import cross_attention as cross
from steering.conditional.encoder_decoder import encoder as enc
from steering.conditional.encoder_decoder import feature_prediction_head as heads
from steering.conditional.encoder_decoder import frozen_decoder as staging
from steering.conditional.encoder_decoder import model as models
from steering.conditional.encoder_decoder import packed_encoder as packed
from steering.conditional.encoder_decoder.frozen_decoder import _ParameterState
from steering.conditional.encoder_decoder.typed_module_list import TypedModuleList
from steering.conditional.feature_vocab import FeatureVocab

_MISSING = object()
_REGISTRIES = ("_parameters", "_buffers", "_modules")
_MODULE_LIFECYCLE_FIELDS = frozenset({"__init__", "__setstate__", "___needs_generation_tag_patch"})
_MODULE_FIELDS: dict[type, tuple[str, ...]] = {
    models.EncoderDecoderGPT: (
        "dec_config", "enc_config", "xattn_config", "feature_pred_config", "enc_dim",
        "window_sizes", "rotary_seq_len", "_feature_pred_layer_idx", "_n_sae_features",
        "_post_cross_attention_eval_callback",
    ),
    enc.FeatureEncoder: ("config", "feature_vocab", "_vocab_oov_msg"),
    enc.EncoderSelfAttention: (
        "n_head", "n_kv_head", "n_embd", "head_dim", "q_dim", "kv_dim", "ve_gate_channels", "ve_gate",
    ),
    gpt.CausalSelfAttention: (
        "layer_idx", "n_head", "n_kv_head", "n_embd", "head_dim", "q_dim", "kv_dim",
        "ve_gate_channels", "ve_gate",
    ),
    cross.CrossAttentionLayer: (
        "config", "n_head", "n_kv_head", "head_dim", "dec_n_embd", "enc_n_embd", "kv_dim",
    ),
    gpt.Linear: ("in_features", "out_features"),
    t.nn.Embedding: (
        "num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type",
        "scale_grad_by_freq", "sparse",
    ),
    TypedModuleList: ("_module_type", "_item_label"),
    heads.ResidualFeaturePredictionHead: ("in_features", "out_features", "block"),
    enc.EncoderBlock: (), enc.EncoderMLP: (), gpt.Block: (), gpt.MLP: (), t.nn.ModuleDict: (),
}
_CONFIG_TYPES = (
    gpt.GPTConfig, configs.EncoderConfig, configs.CrossAttentionConfig,
    configs.FeaturePredictionConfig, enc.FeatureEncoderConfig,
    cross.CrossAttentionLayerConfig, FeatureVocab,
)
_ENUM_TYPES = (configs.FeaturePredLossMode, configs.FeaturePredLossFn,
               configs.FeaturePredictionBlock, flash.FlashAttnImpl)
_COMMON_FIELDS = frozenset({
    "training", "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
    "_is_full_backward_hook", "_compiled_call_impl",
})
_NONCOMPUTE_FIELDS = {
    models.EncoderDecoderGPT: frozenset({"feature_mask"}),
    enc.FeatureEncoder: frozenset({"_feature_embed_full_avg_row_norm"}),
}


@dataclass(frozen=True)
class NativeBinding:
    """A definition-time native binding; never infer the baseline on admission."""

    namespace: ModuleType
    name: str
    expected: Any
    code: Any

    @classmethod
    def capture(cls, namespace: ModuleType, name: str, expected: Any) -> NativeBinding:
        return cls(namespace, name, expected, getattr(expected, "__code__", None))

    def matches(self) -> bool:
        current = vars(self.namespace).get(self.name, _MISSING)
        return current is self.expected and getattr(current, "__code__", None) is self.code


def _value_signature(value: Any) -> tuple[Any, ...]:
    functions = (value,) if type(value) is FunctionType else (
        (value.fget, value.fset, value.fdel) if type(value) is property else ()
    )
    return (id(value), tuple(
        (id(fn), id(fn.__code__), id(fn.__defaults__), id(fn.__kwdefaults__))
        for fn in functions if type(fn) is FunctionType
    ))


@dataclass(frozen=True)
class _ClassState:
    kind: type
    bases_mro: tuple[type, ...]
    bindings: tuple[tuple[str, Any, tuple[Any, ...]], ...]

    @classmethod
    def capture(cls, kind: type) -> _ClassState:
        # Static lookup does not execute descriptors or custom instance access.
        return cls(kind, type.__getattribute__(kind, "__mro__"), tuple(
            (name, value, _value_signature(value))
            for name, value in type.__getattribute__(kind, "__dict__").items()
            if not (name == "__slotnames__" and type(value) in (list, tuple) and not value)
            and not (kind is t.nn.Module and name in _MODULE_LIFECYCLE_FIELDS)
        ))

    def matches(self) -> bool:
        current = self.capture(self.kind)
        return (current.bases_mro == self.bases_mro
                and tuple((name, signature) for name, _value, signature in current.bindings)
                == tuple((name, signature) for name, _value, signature in self.bindings))


# Snapshot known native project bindings at import, before any optional session.
# This is only Python metadata: no model inspection, private import or source IO.
_NATIVE_CLASSES = {
    base: _ClassState.capture(base)
    for kind in (*_MODULE_FIELDS, *_CONFIG_TYPES, *_ENUM_TYPES, staging._StagedLinear,
                 packed.PackedEncoderLayout, packed.PackedEncoderOutput)
    for base in type.__getattribute__(kind, "__mro__")
}
_PRIVATE_NAMES = {
    "torch.overrides": ("_get_current_function_mode_stack",),
    "torch.utils._python_dispatch": ("_get_current_dispatch_mode_stack",),
    "torch._C._autograd": ("_top_saved_tensors_default_hooks",),
}
_PRIVATE_MODULES = {name: sys.modules.get(name) for name in _PRIVATE_NAMES}
_PRIVATE_BINDINGS = {
    name: NativeBinding.capture(module, name, vars(module).get(name, _MISSING))
    for module_name, names in _PRIVATE_NAMES.items()
    if type(module := _PRIVATE_MODULES[module_name]) is ModuleType
    for name in names
}


def _private_binding_reason() -> str | None:
    for name, module in _PRIVATE_MODULES.items():
        if type(module) is not ModuleType or sys.modules.get(name) is not module:
            return "unsupported_framework_module:" + name
    for binding in _PRIVATE_BINDINGS.values():
        if binding.expected is _MISSING or not callable(binding.expected):
            return "unsupported_framework_capability:" + binding.namespace.__name__ + "." + binding.name
        if not binding.matches():
            return "framework_binding_changed:" + binding.namespace.__name__ + "." + binding.name
    return None


_NATIVE_GLOBAL_NAMES: tuple[tuple[ModuleType, tuple[str, ...]], ...] = (
    (models, ("norm", "COMPUTE_DTYPE", "ResidualFeaturePredictionHead", "validate_packed_cross_attention_inputs",
              "build_packed_cross_attention_block_mask")),
    (enc, ("norm", "COMPUTE_DTYPE", "pack_encoder_inputs", "mark_packed_encoder_output_dynamic",
           "flex_attention", "_compiled_encode_packed_dynamic")),
    (cross, ("norm", "flex_attention", "_dispatch_packed_cross_attention_dynamic",
             "_compiled_packed_cross_attention_dynamic", "build_block_mask_from_dense_routing",
             "mark_packed_block_mask_dynamic")),
    (packed, ("pack_encoder_inputs", "pack_encoder_layout", "_validate_dense_mask",
              "_pack_tensor", "_mark_packed_inputs_dynamic", "_validate_packed_output_shape",
              "_build_packed_encoder_output",
              "build_packed_encoder_block_mask", "build_block_mask_from_dense_routing",
              "mark_packed_block_mask_dynamic", "mark_packed_encoder_output_dynamic",
              "append_packed_encoder_segment", "PACKED_ENCODER_ALIGNMENT", "PACKED_ENCODER_BLOCK_SIZE")),
    (gpt, ("norm", "apply_rotary_emb", "COMPUTE_DTYPE", "flash_attn")),
    (heads, ("norm", "Block")),
    (flash, ("flash_attn_func", "_use_flex", "_use_fa3", "_sdpa_attention",
             "_get_block_mask", "_compiled_flex", "_ensure_flash_compat",
             "_causal_mask", "_make_causal_sliding_mask", "_create_block_mask")),
)
_NATIVE_GLOBALS = tuple(
    NativeBinding.capture(module, name, vars(module)[name])
    for module, names in _NATIVE_GLOBAL_NAMES for name in names
)


def _execution_reason() -> str | None:
    reason = _private_binding_reason()
    if reason is not None:
        return reason
    for kind, state in _NATIVE_CLASSES.items():
        if not state.matches():
            return "native_class_changed:" + kind.__qualname__
    for binding in _NATIVE_GLOBALS:
        if not binding.matches():
            return "native_callable_changed:" + binding.namespace.__name__ + "." + binding.name
    if t.is_inference_mode_enabled():
        return "inference_mode"
    if t.is_autocast_enabled("cuda") or t.is_autocast_enabled("cpu"):
        return "autocast"
    if (_PRIVATE_BINDINGS["_get_current_dispatch_mode_stack"].expected()
            or _PRIVATE_BINDINGS["_get_current_function_mode_stack"].expected()):
        return "tensor_dispatch_mode"
    if _PRIVATE_BINDINGS["_top_saved_tensors_default_hooks"].expected(True):
        return "saved_tensor_hooks"
    if any(bool(value) for name, value in vars(t.nn.modules.module).items()
           if name.startswith("_global_") and "hooks" in name):
        return "global_module_hooks"
    return None


def _literal(value: Any) -> Any:
    if value is _MISSING:
        return ("missing",)
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    if type(value) in _ENUM_TYPES:
        return (type(value), value.value)
    if type(value) is type and value in _NATIVE_CLASSES:
        return ("native_type", value)
    if type(value) in (tuple, list):
        # Only windows and immutable vocabulary IDs need sequence metadata.
        if any(type(item) in (tuple, list) and any(type(v) not in (bool, int) for v in item)
               for item in value):
            raise ValueError("unsupported_nested_metadata")
        return (type(value), tuple(_literal(item) for item in value))
    if type(value) in _CONFIG_TYPES:
        data = object.__getattribute__(value, "__dict__")
        names = tuple(field.name for field in fields(type(value)))
        if set(data) != set(names):
            raise ValueError("unsupported_config_fields")
        return (type(value), id(value), tuple((name, _literal(data[name])) for name in names))
    raise ValueError("unsupported_metadata:" + type(value).__qualname__)


def _module_metadata(module: t.nn.Module) -> tuple[Any, ...]:
    data = object.__getattribute__(module, "__dict__")
    kind = type(module)
    if kind not in _MODULE_FIELDS:
        raise ValueError("unsupported_module:" + kind.__qualname__)
    if data.get("training") is not False:
        raise ValueError("module_training")
    if data.get("_compiled_call_impl") is not None:
        raise ValueError("compiled_module_call")
    if any(bool(value) for name, value in data.items() if "hooks" in name):
        raise ValueError("module_hooks")
    allowed = _COMMON_FIELDS | set(_MODULE_FIELDS[kind]) | _NONCOMPUTE_FIELDS.get(kind, frozenset())
    for name in data:
        if name not in allowed and "hooks" not in name and not (kind is gpt.Linear and name == "forward"):
            raise ValueError("unsupported_module_attribute:" + name)
    if kind is t.nn.Embedding and data["max_norm"] is not None:
        raise ValueError("embedding_max_norm")
    if kind is models.EncoderDecoderGPT and data["_post_cross_attention_eval_callback"] is not None:
        raise ValueError("post_cross_attention_callback")
    return (
        tuple((name, _literal(data.get(name, _MISSING))) for name in _MODULE_FIELDS[kind]),
        tuple(sorted(data["_non_persistent_buffers_set"])), data.get("_is_full_backward_hook"),
        id(data.get("forward", _MISSING)),
    )


@dataclass(frozen=True)
class _ModuleState:
    name: str
    module: t.nn.Module
    kind: type
    metadata: tuple[Any, ...]
    registries: tuple[Any, ...]

    @classmethod
    def capture(cls, name: str, module: t.nn.Module) -> _ModuleState:
        metadata = _module_metadata(module)
        data = object.__getattribute__(module, "__dict__")
        registries = []
        for key in _REGISTRIES:
            registry = data[key]
            if type(registry) not in (dict, OrderedDict):
                raise ValueError("unsupported_registry:" + name + "." + key)
            registries.append((key, registry, tuple(registry.items())))
        return cls(name, module, type(module), metadata, tuple(registries))

    def matches(self) -> bool:
        if type(self.module) is not self.kind:
            return False
        data = object.__getattribute__(self.module, "__dict__")
        if _module_metadata(self.module) != self.metadata:
            return False
        for key, registry, items in self.registries:
            current = data[key]
            if current is not registry or tuple(current) != tuple(name for name, _ in items):
                return False
            if any(current[name] is not value for name, value in items):
                return False
        return True


def _tensor_layout(tensor: t.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (id(tensor), tensor.requires_grad, storage._cdata, storage.data_ptr(), storage.nbytes(),
            int(tensor.storage_offset()), tuple(tensor.shape), tuple(tensor.stride()), tensor.device, tensor.dtype)


@dataclass(frozen=True)
class _StagingState:
    module: gpt.Linear
    entry: staging._StagedLinear
    forward: Any
    original_forward: Any
    owner: object
    report: staging.StagingReport
    shadows: tuple[tuple[str, _ParameterState], ...]

    @classmethod
    def capture(cls, model: t.nn.Module, module: gpt.Linear) -> _StagingState | None:
        forward = object.__getattribute__(module, "__dict__").get("forward", _MISSING)
        if forward is _MISSING:
            return None
        entry = getattr(forward, "__self__", None)
        if type(entry) is not staging._StagedLinear or getattr(forward, "__func__", None) is not staging._StagedLinear.forward:
            raise ValueError("unsupported_linear_override")
        with staging._OWNER_LOCK:
            owner = staging._OWNERS.get(model)
            if owner is None or staging._OWNERS.get(module) is not owner:
                raise ValueError("staging_owner_missing")
        original = entry.original_forward
        if getattr(original, "__self__", None) is not module or getattr(original, "__func__", None) is not gpt.Linear.forward:
            raise ValueError("staging_original_forward")
        shadows = []
        for name in ("compute_weight", "compute_bias"):
            tensor = getattr(entry, name)
            if tensor is not None:
                if type(tensor) is not t.Tensor or tensor.requires_grad:
                    raise ValueError("staging_shadow_type")
                shadows.append((name, _ParameterState.capture(tensor)))
        result = cls(module, entry, forward, original, owner, entry.report, tuple(shadows))
        if not result.matches(model):
            raise ValueError("staging_state_invalid")
        return result

    def matches(self, model: t.nn.Module) -> bool:
        entry = self.entry
        data = object.__getattribute__(self.module, "__dict__")
        if data.get("forward") is not self.forward or entry.module is not self.module:
            return False
        with staging._OWNER_LOCK:
            if staging._OWNERS.get(model) is not self.owner or staging._OWNERS.get(self.module) is not self.owner:
                return False
        if (entry.original_forward is not self.original_forward or entry.invalidated
                or entry.report is not self.report or not self.report.applied or self.report.cleaned_up
                or entry.compute_dtype != t.bfloat16 or entry.compute_weight is None):
            return False
        if not entry.weight_state.matches(self.module.weight):
            return False
        if entry.bias_state is None:
            if self.module.bias is not None or entry.compute_bias is not None:
                return False
        elif not entry.bias_state.matches(self.module.bias) or entry.compute_bias is None:
            return False
        return all(type(getattr(entry, name)) is t.Tensor and state.matches(getattr(entry, name)) and state.dtype == t.bfloat16
                   and state.device == entry.weight_state.device for name, state in self.shadows)


def _ambient(device: t.device) -> tuple[Any, ...]:
    matmul = t.backends.cuda.matmul
    return (
        t.are_deterministic_algorithms_enabled(), t.is_deterministic_algorithms_warn_only_enabled(),
        t.get_float32_matmul_precision(), matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction,
        t.backends.cuda.flash_sdp_enabled(), t.backends.cuda.math_sdp_enabled(),
        t.backends.cuda.mem_efficient_sdp_enabled(), t.backends.cuda.cudnn_sdp_enabled(),
        t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        flash._override_impl, flash.HAS_FA3, flash.HAS_FA4_BACKEND, flash.HAS_FLEX_ATTENTION,
        id(flash._fa3), flash._BLOCK_M, flash._BLOCK_N, flash._FLEX_DECODING_QUERY_LENGTH_CUTOFF,
        tuple(enc._PACKED_ENCODER_FLEX_KERNEL_OPTIONS.items()), tuple(cross._PACKED_FLEX_KERNEL_OPTIONS.items()),
        t.cuda.current_stream(device).cuda_stream, t.cuda.current_device(),
    )


class NativeFrozenState:
    """Immutable-source baseline with permanent invalidation on first mismatch.

    ``capture`` returns ``(None, reason)`` on unsupported scope. ``matches`` accepts
    ordinary grad-enabled and no-grad calls: the caller owns head-only replay
    gating, while inference/autocast/hooks/foreign dispatch modes are refused.
    """

    def __init__(self, model: models.EncoderDecoderGPT, *, policy_dependencies: tuple[NativeBinding, ...]):
        self.model = model
        self.policy_dependencies = policy_dependencies
        self.modules: tuple[_ModuleState, ...] = ()
        self.frozen: tuple[tuple[str, _ParameterState], ...] = ()
        self.head_layouts: tuple[tuple[t.Tensor, tuple[Any, ...]], ...] = ()
        self.staged: tuple[_StagingState, ...] = ()
        self.device = t.device("cpu")
        self.ambient: tuple[Any, ...] = ()
        self.invalid_reason: str | None = None

    @classmethod
    def capture(cls, model: Any, *, policy_dependencies: tuple[NativeBinding, ...]) -> tuple[NativeFrozenState | None, str]:
        try:
            reason = _execution_reason()
            if reason is not None:
                return None, reason
            if type(model) is not models.EncoderDecoderGPT:
                return None, "unsupported_model"
            if t.distributed.is_initialized() and t.distributed.get_world_size() != 1:
                return None, "unsupported_world_size"
            if any(module.COMPUTE_DTYPE != t.bfloat16 for module in (models, enc, gpt, staging)):
                return None, "compute_dtype_not_bf16"
            if type(policy_dependencies) is not tuple or any(type(binding) is not NativeBinding for binding in policy_dependencies):
                return None, "unsupported_policy_dependencies"
            result = cls(model, policy_dependencies=policy_dependencies)
            result._capture()
            good, reason = result.matches()
            return (result, "admitted") if good else (None, reason)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            return None, "unsupported_state:" + str(error)

    def _capture(self) -> None:
        pending = [("", self.model)]
        seen = set()
        modules = []
        tensors: dict[int, tuple[str, t.Tensor]] = {}
        while pending:
            name, module = pending.pop()
            if id(module) in seen:
                raise ValueError("module_registered_multiple_paths:" + name)
            seen.add(id(module))
            state = _ModuleState.capture(name, module)
            modules.append(state)
            data = object.__getattribute__(module, "__dict__")
            for key in ("_parameters", "_buffers"):
                for child_name, tensor in data[key].items():
                    if tensor is None:
                        continue
                    if type(tensor) not in (t.Tensor, t.nn.Parameter) or t.is_inference(tensor):
                        raise ValueError("unsupported_tensor:" + name + "." + child_name)
                    tensors.setdefault(id(tensor), (name + "." + child_name, tensor))
            pending.extend((name + "." + key, child) for key, child in data["_modules"].items() if child is not None)
        self.modules = tuple(modules)
        head = object.__getattribute__(self.model, "__dict__")["_modules"].get("feature_pred_head")
        if type(head) not in (gpt.Linear, heads.ResidualFeaturePredictionHead):
            raise ValueError("unsupported_head")
        head_tensors = {id(parameter): parameter for parameter in head.parameters()}
        if not head_tensors or any(not tensor.requires_grad for tensor in head_tensors.values()):
            raise ValueError("head_not_fully_trainable")
        if any(tensor.requires_grad and identity not in head_tensors for identity, (_, tensor) in tensors.items()):
            raise ValueError("not_head_only")
        # No head module/parameter may also be registered under the frozen trunk.
        for state in modules:
            for key, _registry, items in state.registries:
                if key in ("_parameters", "_buffers") and not state.name.startswith(".feature_pred_head"):
                    if any(value is not None and id(value) in head_tensors for _, value in items):
                        raise ValueError("head_parameter_registered_in_trunk")
        devices = {tensor.device for _, tensor in tensors.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("model_not_on_one_cuda_device")
        self.device = next(iter(devices))
        ranges = []
        for name, tensor in tensors.values():
            storage = tensor.untyped_storage()
            begin, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            if any(begin < last and first < end for first, last in ranges):
                raise ValueError("registered_storage_overlap:" + name)
            ranges.append((begin, end))
        self.frozen = tuple((name, _ParameterState.capture(tensor)) for identity, (name, tensor) in tensors.items()
                            if identity not in head_tensors)
        self.head_layouts = tuple((tensor, _tensor_layout(tensor)) for tensor in head_tensors.values())
        staged = []
        for state in modules:
            if type(state.module) is gpt.Linear:
                item = _StagingState.capture(self.model, state.module)
                if item is not None:
                    if not state.name.startswith((".transformer.h.", ".lm_head")):
                        raise ValueError("staged_linear_outside_decoder")
                    staged.append(item)
        self.staged = tuple(staged)
        self.ambient = _ambient(self.device)

    def matches(self) -> tuple[bool, str]:
        if self.invalid_reason is not None:
            return False, self.invalid_reason
        try:
            reason = _execution_reason()
            if reason is None:
                for binding in self.policy_dependencies:
                    if not binding.matches():
                        reason = "native_callable_changed:" + binding.namespace.__name__ + "." + binding.name
                        break
            if reason is None and _ambient(self.device) != self.ambient:
                reason = "ambient_execution_changed"
            if reason is None:
                for state in self.modules:
                    if not state.matches():
                        reason = "module_state_changed:" + state.name
                        break
            if reason is None:
                for name, state in self.frozen:
                    if type(state.parameter) not in (t.Tensor, t.nn.Parameter) or not state.matches(state.parameter):
                        reason = "frozen_tensor_changed:" + name
                        break
            if reason is None and any(type(tensor) is not t.nn.Parameter or _tensor_layout(tensor) != layout
                                      for tensor, layout in self.head_layouts):
                reason = "head_structure_or_trainable_scope_changed"
            if reason is None and any(not state.matches(self.model) for state in self.staged):
                reason = "decoder_staging_changed"
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            reason = "unsupported_state:" + str(error)
        if reason is not None:
            self.invalid_reason = reason
            return False, reason
        return True, "matched"
