"""Persistent Muon parameter storage without changing optimizer arithmetic.

Construct after loading/materializing the raw model, before training forwards.
Parameter objects, optimizer group order, and upstream checkpoint state stay
unchanged. Packed buffers are private, ephemeral storage; reconstruct after a
model device/dtype move, assign=True load, parameter replacement, group change,
or distributed process-group change. Ordinary assign=False model loads and
same-topology upstream optimizer checkpoints remain compatible.

Both adapters call the original nanochat fused kernels and inherit AdamW.
The distributed adapter also inherits all reductions and the three-phase step;
it keeps gather send/receive buffers disjoint and preserves padded chunk sizes.
After a partial distributed failure it invalidates Parameter versions, retains
pending gather storage, and rejects further optimizer use. Outstanding
collectives may still write the model; failure is not a safe recovery boundary.
"""

import inspect
import weakref
from typing import Any

import torch as t
import torch.distributed as dist

import nanochat.optim as upstream


def _fingerprint(parameter: t.Tensor) -> tuple[Any, ...]:
    return (
        id(parameter),
        tuple(parameter.shape),
        tuple(parameter.stride()),
        parameter.dtype,
        parameter.device,
        parameter.storage_offset(),
        parameter.untyped_storage().data_ptr(),
        parameter.untyped_storage().nbytes(),
        parameter.requires_grad,
    )


def _validate_source_groups(groups: list[dict[str, Any]]) -> None:
    """Reject aliasing before rebinding any parameter; disjoint saved views work."""
    identities = set()
    intervals: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    for group in groups:
        if group.get("kind") not in ("muon", "adamw"):
            raise ValueError("packed optimizer only supports upstream Muon/AdamW groups")
        parameters = group["params"]
        if group["kind"] == "muon" and not parameters:
            raise ValueError("empty Muon group")
        for parameter in parameters:
            if not isinstance(parameter, t.nn.Parameter) or not parameter.is_leaf:
                raise ValueError("packed optimizer requires leaf Parameter objects")
            if id(parameter) in identities:
                raise ValueError("duplicate/tied parameter in optimizer groups")
            identities.add(id(parameter))
            if (
                parameter.layout != t.strided
                or parameter.device.type not in ("cpu", "cuda")
                or parameter.numel() == 0
            ):
                raise ValueError("unsupported parameter layout/device/empty shape")
            storage = parameter.untyped_storage()
            key = (parameter.device, storage.data_ptr())
            begin = parameter.storage_offset() * parameter.element_size()
            span = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(parameter.shape, parameter.stride(), strict=True)
            )
            end = begin + span * parameter.element_size()
            if any(
                begin < other_end and other_begin < end
                for other_begin, other_end in intervals.get(key, [])
            ):
                raise ValueError("overlapping/tied parameter storage")
            intervals.setdefault(key, []).append((begin, end))
            if group["kind"] == "muon" and (
                parameter.ndim != 2
                or not parameter.is_contiguous()
                or parameter.dtype not in (t.float32, t.bfloat16)
                or not parameter.requires_grad
                or parameter.grad is not None
            ):
                raise ValueError(
                    "Muon packing requires contiguous trainable FP32/BF16 matrices with no existing gradients"
                )
        if (
            group["kind"] == "muon"
            and len({(tuple(value.shape), value.dtype, value.device) for value in parameters}) != 1
        ):
            raise ValueError("Muon group shape/dtype/device must be uniform")


def _validate_owner(
    owner: t.nn.Module, groups: list[dict[str, Any]]
) -> tuple[tuple[str, int], ...]:
    if not isinstance(owner, t.nn.Module) or isinstance(
        getattr(owner, "_orig_mod", None), t.nn.Module
    ):
        raise ValueError("packed optimizer requires the raw owning model")
    # Include duplicate paths and parameters excluded from optimizer groups:
    # rebinding a trainable matrix must not silently break another model alias.
    named = tuple(owner.named_parameters(remove_duplicate=False))
    _validate_source_groups([{"kind": "adamw", "params": [value for _, value in named]}])
    owned = tuple((name, id(value)) for name, value in named)
    owner_ids = {identity for _, identity in owned}
    if any(id(value) not in owner_ids for group in groups for value in group["params"]):
        raise ValueError("optimizer contains parameters outside its owning model")
    return owned


class _PackedParameters:
    """Own the shared storage and reject changes that would detach model weights."""

    def __init__(self, groups: list[dict[str, Any]], owner: t.nn.Module, world_size: int = 1):
        _validate_source_groups(groups)
        self._owner_parameters = _validate_owner(owner, groups)
        self._owner = weakref.ref(owner)
        self.buffers: dict[int, t.Tensor] = {}
        with t.no_grad():
            for group in groups:
                if group["kind"] != "muon":
                    continue
                parameters = group["params"]
                first = parameters[0]
                count = ((len(parameters) + world_size - 1) // world_size) * world_size
                if count == len(parameters):
                    packed = t.stack(parameters)
                else:
                    packed = t.zeros(count, *first.shape, dtype=first.dtype, device=first.device)
                    packed[: len(parameters)].copy_(t.stack(parameters))
                for index, parameter in enumerate(parameters):
                    parameter.data = packed[index]
                # Rebinding must also invalidate a forward saved before construction.
                t.autograd.graph.increment_version(parameters)
                self.buffers[id(first)] = packed
        self._bindings = self._current_bindings(groups)

    @staticmethod
    def _current_bindings(groups: list[dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(
            (group["kind"], tuple(_fingerprint(value) for value in group["params"]))
            for group in groups
        )

    def validate(self, groups: list[dict[str, Any]]) -> None:
        owner = self._owner()
        if (
            owner is None
            or tuple(
                (name, id(value)) for name, value in owner.named_parameters(remove_duplicate=False)
            )
            != self._owner_parameters
        ):
            raise RuntimeError(
                "packed optimizer owning model Parameters changed; reconstruct optimizer"
            )
        if self._current_bindings(groups) != self._bindings:
            raise RuntimeError(
                "parameter storage/shape/dtype/device or group membership/order/kind changed; "
                "reconstruct optimizer"
            )


class PackedMuonAdamW(upstream.MuonAdamW):
    """Same update/state contract, with persistent Muon parameter stacks only."""

    def __init__(self, param_groups: list[dict[str, Any]], *, owner: t.nn.Module):
        super().__init__([{**group, "params": list(group["params"])} for group in param_groups])
        self._packing = _PackedParameters(self.param_groups, owner)

    def _validate_bindings(self) -> None:
        self._packing.validate(self.param_groups)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._validate_bindings()
        super().load_state_dict(state_dict)
        self._validate_bindings()

    def state_dict(self) -> dict[str, Any]:
        self._validate_bindings()
        return super().state_dict()

    @t.no_grad()
    def step(self):
        # Validate every group before AdamW or Muon can make a partial update.
        self._validate_bindings()
        # Torch wraps each optimizer class's step in hooks at construction.
        # Reuse the upstream dispatch without invoking those hooks a second time.
        inspect.unwrap(upstream.MuonAdamW.step)(self)

    def _step_muon(self, group: dict[str, Any]) -> None:
        parameters = group["params"]
        first = parameters[0]
        packed = self._packing.buffers[id(first)]
        state = self.state[first]
        count, shape, device, dtype = len(parameters), first.shape, first.device, first.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = t.zeros(count, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (count, shape[-2], 1) if shape[-2] >= shape[-1] else (count, 1, shape[-1])
            state["second_momentum_buffer"] = t.zeros(state_shape, dtype=dtype, device=device)
        gradients = t.stack([parameter.grad for parameter in parameters])
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        try:
            upstream.muon_step_fused(
                gradients,
                packed,
                state["momentum_buffer"],
                state["second_momentum_buffer"],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                -1 if shape[-2] >= shape[-1] else -2,
            )
        finally:
            # Parameter.data assignment retains separate version counters; a
            # packed in-place kernel alone would bypass saved-tensor protection.
            t.autograd.graph.increment_version(parameters)


class DistPackedMuonAdamW(upstream.DistMuonAdamW):
    """Packed Muon storage with upstream sharding and collective ordering."""

    def __init__(self, param_groups: list[dict[str, Any]], *, owner: t.nn.Module):
        self._rank, self._world_size, self._process_group = self._distributed_identity()
        self._failed = False
        super().__init__([{**group, "params": list(group["params"])} for group in param_groups])
        self._packing = _PackedParameters(self.param_groups, owner, self._world_size)
        self._pending_muon: list[dict[str, Any]] = []

    @staticmethod
    def _distributed_identity() -> tuple[int, int, Any]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed packing requires an initialized process group")
        return dist.get_rank(), dist.get_world_size(), dist.group.WORLD

    def _validate_bindings(self) -> None:
        if self._failed:
            raise RuntimeError("distributed packed optimizer is poisoned after a failed step")
        rank, world, process_group = self._distributed_identity()
        if (rank, world) != (
            self._rank,
            self._world_size,
        ) or process_group is not self._process_group:
            raise RuntimeError(
                "distributed rank/world/process group changed; reconstruct optimizer"
            )
        self._packing.validate(self.param_groups)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._validate_bindings()
        super().load_state_dict(state_dict)
        self._validate_bindings()

    def state_dict(self) -> dict[str, Any]:
        self._validate_bindings()
        return super().state_dict()

    @t.no_grad()
    def step(self):
        self._validate_bindings()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if (
                    gradient is None
                    or gradient.layout != t.strided
                    or gradient.shape != parameter.shape
                    or gradient.dtype != parameter.dtype
                    or gradient.device != parameter.device
                ):
                    raise ValueError(
                        "distributed optimizer requires a matching dense gradient for every parameter"
                    )
        try:
            # Skip the upstream hook wrapper; this class already invokes it.
            inspect.unwrap(upstream.DistMuonAdamW.step)(self)
        except BaseException:
            # Invalidate saved tensors even when owned storage changed before a
            # later kernel/collective failed. No parameter reads or waits here.
            self._failed = True
            t.autograd.graph.increment_version(
                [
                    value
                    for group in self.param_groups
                    if group["kind"] == "muon"
                    for value in group["params"]
                ]
            )
            raise
        self._pending_muon.clear()

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        info["future"].wait()
        params = group["params"]
        chunk_size, grad_chunk = info["chunk_size"], info["grad_chunk"]
        first = params[0]
        shape, device, dtype = first.shape, first.device, first.dtype
        start = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start))
        state = self.state[first]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = t.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = t.zeros(state_shape, dtype=dtype, device=device)
        packed = self._packing.buffers[id(first)]
        updated_params = t.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = packed[start : start + num_owned]
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            upstream.muon_step_fused(
                grad_chunk[:num_owned],
                owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                -1 if shape[-2] >= shape[-1] else -2,
            )
            updated_params[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
        # Deliberately disjoint send/receive buffers: no backend-specific
        # in-place all-gather assumption. Keep the send alive through completion.
        pending = {"updated_params": updated_params, "params": params}
        self._pending_muon.append(pending)
        pending["future"] = dist.all_gather_into_tensor(
            packed, updated_params, async_op=True
        ).get_future()
        gather_list.append(pending)

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                t.autograd.graph.increment_version(info["params"])
