"""Scoped BF16 shadows for frozen encoder-decoder trunk linears.

Enter only after checkpoint loading and trainable-scope configuration. Parameters,
state_dict entries and optimizer ownership stay unchanged. Ordinary native Linear
instances under transformer.h and lm_head alone are eligible; feature-head,
encoder and cross-attention modules retain their native implementation.

Distinct registered parameters or buffers sharing a selected tensor's whole
storage disable staging, including frozen aliases and distinct selected tensors.
Repeated registrations of the same tensor are safe because they share a version
counter. While active, the caller must keep the model registry/trainable scope
fixed and must not write via .data or unregistered independently versioned aliases.
Versioned writes through selected parameters invalidate their shadows.

Entry, cleanup and any inactive fallback are explicit. Shadow memory admission is
an estimate, not an OOM guarantee. No shadow tensors, method overrides, or runtime
admission state are serialized into checkpoints.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import cast
from weakref import WeakKeyDictionary

import torch as t
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Linear

_MISSING = object()
_OWNERS: WeakKeyDictionary[t.nn.Module, object] = WeakKeyDictionary()
_OWNER_LOCK = RLock()

DEFAULT_MIN_FREE_BYTES = 4 * 1024**3


def frozen_decoder_staging_settings(cfg: Mapping) -> dict[str, int]:
    """Resolve the two optional performance controls without loading a model."""
    resolved = {
        "max_shadow_bytes": cfg.get("frozen_decoder_max_shadow_bytes", 0),
        "min_free_bytes": cfg.get("frozen_decoder_min_free_bytes", DEFAULT_MIN_FREE_BYTES),
    }
    for name, value in resolved.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"frozen_decoder_{name} must be a nonnegative integer")
    return resolved


@dataclass
class StagingReport:
    applied: bool = False
    reason: str = "not_entered"
    module_names: tuple[str, ...] = ()
    planned_shadow_bytes: int = 0
    staged_shadow_bytes: int = 0
    staging_seconds: float = 0.0
    # Driver-free memory excludes reusable blocks retained by the allocator.
    admission_free_bytes: int = 0
    admission_allocator_reserved_bytes: int = 0
    admission_allocator_allocated_bytes: int = 0
    admission_reusable_reserved_bytes: int = 0
    admission_estimated_available_bytes: int = 0
    storage_overlap_registered_names: tuple[str, ...] = ()
    fallback_calls: int = 0
    invalidated_modules: int = 0
    cleaned_up: bool = False


@dataclass(frozen=True)
class _ParameterState:
    parameter: t.Tensor
    version: int
    storage_identity: int
    storage_pointer: int
    storage_bytes: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: t.device
    dtype: t.dtype

    @classmethod
    def capture(cls, parameter: t.Tensor) -> _ParameterState:
        storage = parameter.untyped_storage()
        return cls(
            parameter=parameter,
            version=parameter._version,
            storage_identity=storage._cdata,
            storage_pointer=storage.data_ptr(),
            storage_bytes=storage.nbytes(),
            storage_offset=int(parameter.storage_offset()),
            shape=tuple(parameter.shape),
            stride=tuple(parameter.stride()),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def matches(self, parameter: t.Tensor | None) -> bool:
        if parameter is not self.parameter or parameter.requires_grad:
            return False
        try:
            storage = parameter.untyped_storage()
            return (
                parameter._version == self.version
                and storage._cdata == self.storage_identity
                and storage.data_ptr() == self.storage_pointer
                and storage.nbytes() == self.storage_bytes
                and parameter.storage_offset() == self.storage_offset
                and tuple(parameter.shape) == self.shape
                and tuple(parameter.stride()) == self.stride
                and parameter.device == self.device
                and parameter.dtype == self.dtype
            )
        except RuntimeError:
            return False


@dataclass
class _StagedLinear:
    module: Linear
    original_forward: Callable[[t.Tensor], t.Tensor]
    weight_state: _ParameterState
    bias_state: _ParameterState | None
    compute_weight: t.Tensor | None
    compute_bias: t.Tensor | None
    compute_dtype: t.dtype
    report: StagingReport
    invalidated: bool = False

    def forward(self, x: t.Tensor) -> t.Tensor:
        valid = (
            not self.invalidated
            and type(self.module) is Linear
            and self.weight_state.matches(self.module.weight)
            and (
                self.module.bias is None
                if self.bias_state is None
                else self.bias_state.matches(self.module.bias)
            )
        )
        if not valid:
            if not self.invalidated:
                self.invalidated = True
                self.compute_weight = self.compute_bias = None
                self.report.invalidated_modules += 1
            self.report.fallback_calls += 1
            return self.original_forward(x)
        if x.device != self.weight_state.device or x.dtype != self.compute_dtype:
            self.report.fallback_calls += 1
            return self.original_forward(x)
        assert self.compute_weight is not None
        # A shadow can be consumed on a stream other than its staging stream.
        # Register that use so allocator reuse remains safe after context exit.
        stream = t.cuda.current_stream(x.device)
        self.compute_weight.record_stream(stream)
        if self.compute_bias is not None:
            self.compute_bias.record_stream(stream)
        return F.linear(x, self.compute_weight, self.compute_bias)


def _decoder_linears(model: t.nn.Module) -> list[tuple[str, Linear]]:
    transformer = getattr(model, "transformer", None)
    head = getattr(model, "feature_pred_head", None)
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(transformer, t.nn.ModuleDict) or "h" not in transformer:
        raise ValueError("Expected encoder-decoder transformer.h modules")
    if not isinstance(head, t.nn.Module) or not isinstance(lm_head, t.nn.Module):
        raise ValueError("Expected separate feature_pred_head and lm_head modules")
    head_parameters = {id(parameter) for parameter in head.parameters()}
    if any(
        parameter.requires_grad and id(parameter) not in head_parameters
        for parameter in model.parameters()
    ):
        raise ValueError("Frozen decoder staging requires the feature-head-only scope")
    excluded = {id(module) for module in head.modules()}
    for name in ("encoder", "cross_attn_layers"):
        root = getattr(model, name, None)
        if isinstance(root, t.nn.Module):
            excluded.update(id(module) for module in root.modules())
    result: list[tuple[str, Linear]] = []
    seen: set[int] = set()
    for prefix, root in (("transformer.h", transformer["h"]), ("lm_head", lm_head)):
        for suffix, module in root.named_modules():
            if type(module) is not Linear or id(module) in seen or id(module) in excluded:
                continue
            seen.add(id(module))
            name = prefix + ("." + suffix if suffix else "")
            if module.__dict__.get("forward", _MISSING) is not _MISSING:
                raise ValueError(f"Refusing an existing instance forward override: {name}")
            if getattr(module.forward, "__func__", None) is not Linear.forward:
                raise ValueError(f"Expected the ordinary native Linear forward: {name}")
            parameters = (module.weight,) if module.bias is None else (module.weight, module.bias)
            if any(parameter.requires_grad for parameter in parameters):
                raise ValueError(f"Refusing a trainable decoder linear: {name}")
            result.append((name, module))
    return result


@contextmanager
def stage_frozen_decoder_linears(
    model: t.nn.Module,
    *,
    max_shadow_bytes: int,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    train_feature_head_only: bool,
    world_size: int,
) -> Iterator[StagingReport]:
    """Stage FP32 decoder weights as BF16 within an explicit memory allowance.

    max_shadow_bytes bounds persistent shadows. min_free_bytes is the caller's
    reserved headroom for future KV, activation, replay and compiler allocations.
    Admission estimates available bytes as driver-free plus allocator-reserved
    minus allocated bytes (clamped at zero for the reusable-reserve term). The
    reserve can be fragmented, pending stream use, or otherwise unreusable for a
    particular request; concurrent allocations can also change the snapshot.
    This is an admission estimate, not an OOM guarantee. It does not empty the
    allocator cache. Choose headroom from measured workload needs.
    Budget/device/dtype inadmission yields an inactive report without mutation.
    A staging-only CUDA OOM also yields inactive after releasing partial shadows.
    """
    for name, value in (("max_shadow_bytes", max_shadow_bytes), ("min_free_bytes", min_free_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(train_feature_head_only) is not bool:
        raise ValueError("train_feature_head_only must be a boolean")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    report = StagingReport()
    entries: list[tuple[_StagedLinear, Callable[[t.Tensor], t.Tensor]]] = []
    shadows: dict[int, t.Tensor] = {}
    claimed: list[t.nn.Module] = []
    owner = object()
    try:
        if max_shadow_bytes == 0:
            report.reason = "disabled"
            yield report
            return
        if not train_feature_head_only:
            report.reason = "full_policy"
            yield report
            return
        if world_size != 1:
            report.reason = "unsupported_world_size"
            yield report
            return
        if COMPUTE_DTYPE != t.bfloat16:
            report.reason = "native_compute_dtype_not_bf16"
            yield report
            return
        with _OWNER_LOCK:
            if model in _OWNERS:
                raise RuntimeError("A staging context already owns this model")
            linears = _decoder_linears(model)
            claimed = [model, *(module for _, module in linears)]
            if any(module in _OWNERS for module in claimed):
                raise RuntimeError("A staging context already owns a selected decoder linear")
            for module in claimed:
                _OWNERS[module] = owner
        report.module_names = tuple(name for name, _ in linears)
        parameters = {
            id(parameter): parameter
            for _, module in linears
            for parameter in (module.weight, module.bias)
            if parameter is not None
        }
        if not parameters:
            report.reason = "no_eligible_linears"
            yield report
            return
        devices = {parameter.device for parameter in parameters.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            report.reason = "decoder_not_on_one_cuda_device"
            yield report
            return
        if any(
            parameter.dtype != t.float32 or not parameter.is_contiguous()
            for parameter in parameters.values()
        ):
            report.reason = "decoder_weights_not_contiguous_fp32"
            yield report
            return
        device = next(iter(devices))
        selected_storage_ranges: list[tuple[int, int, int]] = []
        for parameter in parameters.values():
            storage = parameter.untyped_storage()
            start = storage.data_ptr()
            selected_storage_ranges.append((id(parameter), start, start + storage.nbytes()))
        registered_tensors = (
            *(("parameter", name, tensor) for name, tensor in model.named_parameters(remove_duplicate=False)),
            *(("buffer", name, tensor) for name, tensor in model.named_buffers(remove_duplicate=False)),
        )
        overlapping_names: list[str] = []
        for kind, name, tensor in registered_tensors:
            if tensor.device != device:
                continue
            try:
                storage = tensor.untyped_storage()
                start = storage.data_ptr()
                end = start + storage.nbytes()
            except RuntimeError:
                report.reason = "unsupported_registered_storage"
                yield report
                return
            # A distinct registered tensor can share storage while maintaining
            # an independent version counter, even when both tensors are frozen.
            # Whole-storage rejection also covers disjoint views conservatively.
            if any(id(tensor) != identity and start < selected_end and selected_start < end
                   for identity, selected_start, selected_end in selected_storage_ranges):
                overlapping_names.append(f"{kind}:{name}")
        if overlapping_names:
            report.storage_overlap_registered_names = tuple(overlapping_names)
            report.reason = "selected_storage_overlaps_registered_tensor"
            yield report
            return
        report.planned_shadow_bytes = 2 * sum(parameter.numel() for parameter in parameters.values())
        if report.planned_shadow_bytes > max_shadow_bytes:
            report.reason = "shadow_budget_exceeded"
            yield report
            return
        free_bytes, _total_bytes = t.cuda.mem_get_info(device)
        reserved_bytes = t.cuda.memory_reserved(device)
        allocated_bytes = t.cuda.memory_allocated(device)
        reusable_reserved_bytes = max(0, reserved_bytes - allocated_bytes)
        estimated_available_bytes = free_bytes + reusable_reserved_bytes
        report.admission_free_bytes = free_bytes
        report.admission_allocator_reserved_bytes = reserved_bytes
        report.admission_allocator_allocated_bytes = allocated_bytes
        report.admission_reusable_reserved_bytes = reusable_reserved_bytes
        report.admission_estimated_available_bytes = estimated_available_bytes
        if estimated_available_bytes < report.planned_shadow_bytes + min_free_bytes:
            report.reason = "insufficient_estimated_memory_with_headroom"
            yield report
            return
        t.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            # These must be ordinary tensors even if a diagnostic caller enters
            # under inference_mode; F.linear may need them for an input gradient.
            with t.inference_mode(False), t.no_grad():
                for identity, parameter in parameters.items():
                    shadows[identity] = parameter.detach().to(dtype=t.bfloat16)
            t.cuda.synchronize(device)
        except t.OutOfMemoryError:
            shadows.clear()
            report.reason = "shadow_allocation_oom"
            report.staging_seconds = time.perf_counter() - started
            yield report
            return
        report.staged_shadow_bytes = report.planned_shadow_bytes
        for _, module in linears:
            entry = _StagedLinear(
                module=module,
                original_forward=cast(Callable[[t.Tensor], t.Tensor], module.forward),
                weight_state=_ParameterState.capture(module.weight),
                bias_state=None if module.bias is None else _ParameterState.capture(module.bias),
                compute_weight=shadows[id(module.weight)],
                compute_bias=None if module.bias is None else shadows[id(module.bias)],
                compute_dtype=t.bfloat16,
                report=report,
            )
            entries.append((entry, entry.forward))
        shadows.clear()
        for entry, forward in entries:
            entry.module.forward = forward  # ty: ignore[invalid-assignment]
        report.staging_seconds = time.perf_counter() - started
        report.applied = True
        report.reason = "staged"
        yield report
    finally:
        for entry, forward in reversed(entries):
            if entry.module.__dict__.get("forward", _MISSING) is forward:
                delattr(entry.module, "forward")
            entry.invalidated = True
            entry.compute_weight = entry.compute_bias = None
        shadows.clear()
        with _OWNER_LOCK:
            for module in claimed:
                if _OWNERS.get(module) is owner:
                    del _OWNERS[module]
        report.cleaned_up = True
