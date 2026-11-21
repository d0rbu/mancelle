"""Conditioned rsLoRA fitting for the AxBench encoder-decoder baselines."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from nanochat.atomic_io import atomic_write_bytes
from steering.eval.axbench_base.baselines.hooks import _get_block
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    FitRowConditioning,
    conditioned_forward_inputs,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    RenderedFitRow,
    targets_from_mask,
)
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    load_safetensors_file_snapshot,
)

LORA_RANK = 4
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LORA_LR = 0.0009
LORA_WEIGHT_DECAY = 0.0
LORA_EPOCHS = 24
LORA_BATCH_ROWS = 6
LORA_GRAD_ACCUM = 6

LORA_SEED_DOMAIN = "axbench-upstream-encdec-lora-v1"
_SEED_MODULUS = 2**31 - 1
_LORA_STATE_KEY = re.compile(r"^layer_(\d+)_attn_c_proj_lora_([AB])$")


class RsLoRAAdapter(nn.Module):
    """A cast-on-use fp32 rsLoRA wrapper around one decoder projection."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        linear = cast(Any, base)
        in_features = int(linear.in_features)
        out_features = int(linear.out_features)
        device = linear.weight.device
        self.base = base
        self.lora_A = nn.Parameter(
            t.empty((LORA_RANK, in_features), dtype=t.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            t.empty((out_features, LORA_RANK), dtype=t.float32, device=device)
        )
        self.dropout = nn.Dropout(LORA_DROPOUT)
        self.scaling = LORA_ALPHA / math.sqrt(LORA_RANK)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: t.Tensor) -> t.Tensor:
        result = self.base(x)
        result_dtype = result.dtype
        x_up = self.dropout(x.to(dtype=self.lora_A.dtype))
        delta = F.linear(F.linear(x_up, self.lora_A), self.lora_B) * self.scaling
        return result + delta.to(dtype=result_dtype)


@dataclass(frozen=True)
class LoRATrainingRow:
    """One rendered supervised row paired with its real feature conditioning."""

    rendered: RenderedFitRow
    conditioning: FitRowConditioning


@dataclass(frozen=True)
class AttachedAdapterRef:
    """The exact parent/original/adapter objects for one reversible swap."""

    layer: int
    parent: nn.Module
    original: nn.Module
    adapter: RsLoRAAdapter


def attach_encdec_lora(model: nn.Module, layers: Sequence[int]) -> list[AttachedAdapterRef]:
    """Attach adapters to ``layers``, rolling back every attach on failure."""

    refs: list[AttachedAdapterRef] = []
    try:
        for layer in layers:
            block = _get_block(model, layer)
            parent = cast(nn.Module, block.attn)
            original = cast(nn.Module, parent.c_proj)
            adapter = RsLoRAAdapter(original)
            adapter.train(parent.training)
            parent.c_proj = adapter
            refs.append(
                AttachedAdapterRef(
                    layer=layer,
                    parent=parent,
                    original=original,
                    adapter=adapter,
                )
            )
    except BaseException:
        for ref in reversed(refs):
            ref.parent.c_proj = ref.original
        raise
    return refs


def detach_encdec_lora(model: nn.Module, refs: Sequence[AttachedAdapterRef]) -> None:
    """Restore the original projection objects captured during attachment."""

    del model
    for ref in reversed(refs):
        ref.parent.c_proj = ref.original


def _derive_lora_seed(base_seed: int, concept_id: int) -> int:
    fields = (LORA_SEED_DOMAIN, str(base_seed), str(concept_id))
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def _state_from_refs(refs: Sequence[AttachedAdapterRef]) -> dict[str, t.Tensor]:
    state: dict[str, t.Tensor] = {}
    for ref in refs:
        prefix = f"layer_{ref.layer}_attn_c_proj_lora_"
        state[f"{prefix}A"] = ref.adapter.lora_A.detach().to(device="cpu").contiguous().clone()
        state[f"{prefix}B"] = ref.adapter.lora_B.detach().to(device="cpu").contiguous().clone()
    return state


def train_encdec_lora(
    model: nn.Module,
    rows: Sequence[LoRATrainingRow],
    *,
    layers: Sequence[int],
    base_seed: int,
    concept_id: int,
    surface: str,
    epochs: int = LORA_EPOCHS,
) -> dict[str, t.Tensor]:
    """Fit one concept's adapters with corrected vendored accumulation semantics.

    Vendored AxBench floors the scheduler horizon, which silently freezes the
    learning rate at zero mid-training when row counts are not divisible by 36.
    We use ceil so linear decay spans the real optimizer steps.
    """

    if not rows:
        raise ValueError(f"{surface}: LoRA training rows must not be empty")
    if epochs <= 0:
        raise ValueError(f"{surface}: LoRA training epochs must be positive")
    for row in rows:
        if row.rendered.ids != row.conditioning.token_ids:
            raise ValueError(f"{surface}: rendered ids must equal conditioning token_ids")

    derived_seed = _derive_lora_seed(base_seed, concept_id)
    # Dropout draws from the global RNG (deliberate — nn.Dropout takes no
    # generator); snapshot the caller's RNG state so training is a sealed
    # boundary rather than an observable process mutation.
    entry_rng_state = t.get_rng_state()
    device = next(model.parameters()).device
    entry_device_rng_state: t.Tensor | None = None
    if device.type == "cuda":
        entry_device_rng_state = t.cuda.get_rng_state(device)
    elif device.type == "mps":
        entry_device_rng_state = t.mps.get_rng_state()
    entry_training = model.training
    base_parameters = list(model.named_parameters())
    requires_grad_at_entry = {
        id(parameter): parameter.requires_grad for _name, parameter in base_parameters
    }
    refs: list[AttachedAdapterRef] = []
    # Track OUR body's outcome directly: `sys.exc_info()` also reflects an
    # exception the CALLER is currently handling, which must not downgrade a
    # success-path leaked-gradient failure to a warning.
    body_succeeded = False
    try:
        # Seed only the CPU generator and the model's active accelerator. A
        # process may own other CUDA devices whose streams are unrelated to
        # this fit and must remain untouched.
        t.random.default_generator.manual_seed(derived_seed)
        if device.type == "cuda":
            with t.cuda.device(device):
                t.cuda.manual_seed(derived_seed)
        elif device.type == "mps":
            t.mps.manual_seed(derived_seed)
        shuffle_generator = t.Generator(device="cpu")
        shuffle_generator.manual_seed(derived_seed)
        refs = attach_encdec_lora(model, layers)
        for _name, parameter in base_parameters:
            parameter.requires_grad_(False)
        model.train()
        adapter_parameters = [
            parameter for ref in refs for parameter in (ref.adapter.lora_A, ref.adapter.lora_B)
        ]
        optimizer = t.optim.AdamW(
            adapter_parameters,
            lr=LORA_LR,
            weight_decay=LORA_WEIGHT_DECAY,
        )
        n_batches = math.ceil(len(rows) / LORA_BATCH_ROWS)
        num_training_steps = epochs * max(1, math.ceil(n_batches / LORA_GRAD_ACCUM))
        scheduler = t.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(0.0, 1.0 - step / num_training_steps),
        )
        optimizer.zero_grad()
        prepared_rows: list[
            tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, int]
        ] = []
        for row_index, row in enumerate(rows):
            with profile_region(
                "axbench_upstream.encdec_lora.conditioned_forward_inputs",
                concept_id=concept_id,
                row_index=row_index,
            ):
                enc_feature_ids, enc_magnitudes, enc_mask, seg_map = conditioned_forward_inputs(
                    row.conditioning,
                    device,
                )
            idx = t.tensor([row.conditioning.token_ids], dtype=t.int64, device=device)
            target_values = targets_from_mask(row.rendered.ids, row.rendered.mask)
            targets = t.tensor([target_values], dtype=t.int64, device=device)
            prepared_rows.append(
                (
                    idx,
                    enc_feature_ids,
                    enc_magnitudes,
                    enc_mask,
                    seg_map,
                    targets,
                    sum(target != -1 for target in target_values),
                )
            )

        for _epoch in range(epochs):
            order = t.randperm(len(rows), generator=shuffle_generator).tolist()
            for step, batch_start in enumerate(range(0, len(rows), LORA_BATCH_ROWS)):
                batch_indices = order[batch_start : batch_start + LORA_BATCH_ROWS]
                batch_loss: t.Tensor | None = None
                supervised_tokens = 0
                for row_index in batch_indices:
                    (
                        idx,
                        enc_feature_ids,
                        enc_magnitudes,
                        enc_mask,
                        seg_map,
                        targets,
                        row_supervised_tokens,
                    ) = prepared_rows[row_index]
                    supervised_tokens += row_supervised_tokens
                    with profile_region(
                        "axbench_upstream.encdec_lora.forward",
                        concept_id=concept_id,
                        epoch=_epoch,
                        batch=step,
                        row_index=row_index,
                    ):
                        row_loss = model(
                            idx,
                            targets,
                            enc_feature_ids=enc_feature_ids,
                            enc_magnitudes=enc_magnitudes,
                            enc_mask=enc_mask,
                            seg_map=seg_map,
                            loss_reduction="sum",
                            encoder_strict=True,
                        )
                    if not isinstance(row_loss, t.Tensor):
                        raise TypeError(
                            "AxBench encdec-baseline fit row loss forward must return a tensor, "
                            f"got {type(row_loss).__name__}"
                        )
                    batch_loss = row_loss if batch_loss is None else batch_loss + row_loss
                if supervised_tokens == 0:
                    raise ValueError(f"{surface}: LoRA batch has no supervised tokens")
                if batch_loss is None:
                    raise RuntimeError(f"{surface}: LoRA batch unexpectedly has no rows")
                loss = batch_loss / supervised_tokens / LORA_GRAD_ACCUM
                if not bool(t.isfinite(loss).item()):
                    raise ValueError(f"{surface}: non-finite rsLoRA loss")
                with profile_region(
                    "axbench_upstream.encdec_lora.backward",
                    concept_id=concept_id,
                    epoch=_epoch,
                    batch=step,
                ):
                    loss.backward()

                is_accumulation_boundary = (step + 1) % LORA_GRAD_ACCUM == 0
                is_epoch_end = (step + 1) == n_batches
                if is_accumulation_boundary or is_epoch_end:
                    with profile_region(
                        "axbench_upstream.encdec_lora.optimizer_step",
                        concept_id=concept_id,
                        epoch=_epoch,
                        batch=step,
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

        state = _state_from_refs(refs)
        body_succeeded = True
        return state
    finally:
        t.set_rng_state(entry_rng_state)
        if entry_device_rng_state is not None:
            if device.type == "cuda":
                t.cuda.set_rng_state(entry_device_rng_state, device=device)
            else:
                t.mps.set_rng_state(entry_device_rng_state)
        if refs:
            detach_encdec_lora(model, refs)
        for _name, parameter in base_parameters:
            parameter.requires_grad_(requires_grad_at_entry[id(parameter)])
        model.train(entry_training)
        base_grad_names: list[str] = []
        for name, parameter in base_parameters:
            if parameter.grad is not None:
                parameter.grad = None
                base_grad_names.append(name)
        if base_grad_names:
            message = (
                "non-adapter parameter retained a gradient after frozen LoRA training: "
                + ", ".join(base_grad_names)
            )
            if body_succeeded:
                raise RuntimeError(message)
            logger.warning(message)


def save_lora(
    path: Path,
    state: dict[str, t.Tensor],
    metadata: dict[str, Any],
) -> None:
    """Write a portable adapter-only safetensors artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: value.detach().to(dtype=t.float32, device="cpu").contiguous()
        for key, value in state.items()
    }
    atomic_write_bytes(
        deterministic_safetensors_bytes(
            tensors,
            metadata={key: str(value) for key, value in metadata.items()},
        ),
        path,
    )


def load_lora(path: Path) -> tuple[dict[str, t.Tensor], dict[str, Any]]:
    """Load adapter tensors and string-valued safetensors metadata."""

    path = Path(path)
    state, loaded_metadata = load_safetensors_file_snapshot(path)
    if not state:
        raise ValueError(f"{path}: LoRA state must contain at least one adapter pair")
    pairs: dict[int, set[str]] = {}
    for key, tensor in state.items():
        match = _LORA_STATE_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"{path}: invalid LoRA state key {key!r}")
        layer = int(match.group(1))
        pairs.setdefault(layer, set()).add(match.group(2))
        if (
            tensor.dtype is not t.float32
            or tensor.device.type != "cpu"
            or tensor.ndim != 2
            or tensor.numel() == 0
            or not tensor.is_contiguous()
            or not bool(t.isfinite(tensor).all())
        ):
            raise ValueError(
                f"{path}: LoRA tensor {key!r} must be a finite contiguous CPU float32 matrix"
            )
    incomplete = sorted(layer for layer, matrices in pairs.items() if matrices != {"A", "B"})
    if incomplete:
        raise ValueError(f"{path}: LoRA layers have incomplete A/B pairs: {incomplete}")
    metadata: dict[str, Any] = dict(loaded_metadata or {})
    return state, metadata


def validate_lora_state(
    model: nn.Module,
    state: dict[str, t.Tensor],
    *,
    layers: Sequence[int],
) -> None:
    """Require an adapter state to match the target model's exact projections."""

    expected_shapes: dict[str, tuple[int, int]] = {}
    for layer in layers:
        block = cast(Any, _get_block(model, layer))
        projection = block.attn.c_proj
        prefix = f"layer_{layer}_attn_c_proj_lora_"
        expected_shapes[f"{prefix}A"] = (LORA_RANK, int(projection.in_features))
        expected_shapes[f"{prefix}B"] = (int(projection.out_features), LORA_RANK)
    if set(state) != set(expected_shapes):
        raise ValueError(
            "LoRA state keys do not match the target model: "
            f"missing={sorted(set(expected_shapes) - set(state))}, "
            f"unexpected={sorted(set(state) - set(expected_shapes))}"
        )
    for key, expected_shape in expected_shapes.items():
        tensor = state[key]
        if (
            tensor.dtype is not t.float32
            or tensor.device.type != "cpu"
            or tensor.shape != expected_shape
            or not tensor.is_contiguous()
            or not bool(t.isfinite(tensor).all())
        ):
            raise ValueError(
                f"LoRA state tensor {key!r} violates the target-model contract: "
                f"expected_shape={expected_shape}, actual_shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, device={tensor.device}, "
                f"contiguous={tensor.is_contiguous()}"
            )


def apply_lora_state(
    refs: Sequence[AttachedAdapterRef],
    state: dict[str, t.Tensor],
) -> None:
    """Copy one exact adapter state into an attached set of LoRA modules."""

    expected = {
        f"layer_{ref.layer}_attn_c_proj_lora_{matrix}" for ref in refs for matrix in ("A", "B")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            "LoRA state keys do not match attached adapters: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, tensor in state.items():
        if not bool(t.isfinite(tensor).all().item()):
            raise ValueError(f"LoRA state tensor {key!r} contains non-finite values")
    with t.no_grad():
        for ref in refs:
            prefix = f"layer_{ref.layer}_attn_c_proj_lora_"
            for matrix, parameter in (
                ("A", ref.adapter.lora_A),
                ("B", ref.adapter.lora_B),
            ):
                value = state[f"{prefix}{matrix}"]
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"{prefix}{matrix} shape {tuple(value.shape)} does not match "
                        f"adapter shape {tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=t.float32))


__all__ = (
    "LORA_ALPHA",
    "LORA_BATCH_ROWS",
    "LORA_DROPOUT",
    "LORA_EPOCHS",
    "LORA_GRAD_ACCUM",
    "LORA_LR",
    "LORA_RANK",
    "LORA_WEIGHT_DECAY",
    "AttachedAdapterRef",
    "LoRATrainingRow",
    "RsLoRAAdapter",
    "apply_lora_state",
    "attach_encdec_lora",
    "detach_encdec_lora",
    "load_lora",
    "save_lora",
    "train_encdec_lora",
    "validate_lora_state",
)
