"""Shared LoRA attach/detach helpers for ``nanochat`` decoder blocks.

Both the LoRA training loop (``training.py``, Task 24) and the
generation-time loader (``loader.py``, Task 25) need to wrap the same
four ``nanochat.gpt.Linear`` sublayers on a ``nanochat.gpt.Block`` with
``LoRAAdapter``, and later restore the originals byte-for-byte.  This
module centralises that mechanism so both sites share a single
implementation.

Per-block sublayers we wrap with a ``LoRAAdapter``
---------------------------------------------------
Each entry in :data:`_ATTACH_PATHS` is a ``(parent_attribute_name,
child_attribute_name)`` pair on a ``nanochat.gpt.Block``. Order is fixed
so the safetensors keys written by ``training.py`` and read by
``loader.py`` stay in lockstep:

  * ``attn.c_qkv``  — the fused query/key/value projection.
  * ``attn.c_proj`` — the post-attention output projection.
  * ``mlp.c_fc``   — the MLP up-projection.
  * ``mlp.c_proj`` — the MLP down-projection.

Attach contract
---------------
:func:`_attach_lora_to_model` replaces each target sublayer with a fresh
``LoRAAdapter`` (initialised with ``lora_B = 0`` so the adapter forward
is bit-exact equal to the base forward at attach time), returning a dict
keyed by ``(layer_idx, parent_name, child_name)`` mapping to the original
``nn.Linear`` module and its exact pre-attach ``requires_grad`` state.
Attachment is transactional: an exception while constructing or installing
any adapter restores every original module and trainability flag.

Detach contract
---------------
:func:`_detach_lora_from_model` takes the dict returned by
:func:`_attach_lora_to_model` and ``setattr``-restores every original
module. It also restores each base parameter's original ``requires_grad``
value, undoing the freezing that ``LoRAAdapter.__init__`` performed at
attach time without assuming that the base model was trainable. After the
call the model is byte-for-byte identical to the pre-attach state (modulo
any weights the caller mutated between attach and detach).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch.nn as nn

from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.types import AxBenchBaseError

__all__ = (
    "_ATTACH_PATHS",
    "_AttachedAdapterRef",
    "_OriginalLinearState",
    "_attach_lora_to_model",
    "_collect_attached_adapters",
    "_detach_lora_from_model",
)


# Per-block sublayers we wrap with a LoRAAdapter. Each entry is a
# ``(parent_attribute_name, child_attribute_name)`` pair on a
# ``nanochat.gpt.Block``. Order is fixed so tests and loaders see a
# stable layout in the saved safetensors keys.
_ATTACH_PATHS: tuple[tuple[str, str], ...] = (
    ("attn", "c_qkv"),
    ("attn", "c_proj"),
    ("mlp", "c_fc"),
    ("mlp", "c_proj"),
)


@dataclass(frozen=True)
class _AttachedAdapterRef:
    """One LoRAAdapter attached to a ``(layer, parent, child)`` location.

    Used internally by :func:`_collect_attached_adapters` to build the
    save dict (training) and to cross-reference loaded weights with
    attached modules (loader). Kept as a frozen dataclass (rather than a
    tuple) so tests and future extensions can address fields by name.
    """

    layer: int
    parent_name: str
    child_name: str
    adapter: LoRAAdapter


@dataclass(frozen=True)
class _OriginalLinearState:
    """Original module plus its exact pre-attach trainability state."""

    module: nn.Linear
    requires_grad: tuple[tuple[str, bool], ...]


def _restore_original_linear(state: _OriginalLinearState) -> None:
    """Restore every parameter's snapshotted ``requires_grad`` value."""
    current = dict(state.module.named_parameters())
    expected_names = {name for name, _ in state.requires_grad}
    if set(current) != expected_names:
        raise AxBenchBaseError(
            "LoRA base parameter structure changed while an adapter was "
            f"attached: expected={sorted(expected_names)}, "
            f"actual={sorted(current)}."
        )
    for name, requires_grad in state.requires_grad:
        current[name].requires_grad = requires_grad


def _attach_lora_to_model(
    model: nn.Module,
    *,
    layer: int,
    lora_cfg: LoraConfig,
) -> dict[tuple[int, str, str], _OriginalLinearState]:
    """Wrap each ``_ATTACH_PATHS`` sublayer at ``layer`` with a LoRAAdapter.

    Returns a dict keyed by ``(layer_idx, parent_name, child_name)``
    mapping to the original module state that was wrapped, so
    :func:`_detach_lora_from_model` can restore the model to its initial
    state by direct ``setattr``.

    Side-effect: every wrapped sublayer is replaced in-place via
    ``setattr(parent, child_name, adapter)``. The parent module's
    ``forward`` method now sees a ``LoRAAdapter`` instead of a
    ``Linear``; both behave identically (``LoRAAdapter`` delegates to its
    wrapped ``base``), so no other code needs to change.
    """
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    block = blocks[layer]

    original: dict[tuple[int, str, str], _OriginalLinearState] = {}
    locations: list[tuple[str, str, nn.Module, nn.Linear]] = []
    for parent_name, child_name in _ATTACH_PATHS:
        parent = getattr(block, parent_name)
        base_linear = getattr(parent, child_name)
        if not isinstance(base_linear, nn.Linear):
            raise AxBenchBaseError(
                f"Expected nn.Linear at layer={layer} "
                f"{parent_name}.{child_name}, got "
                f"{type(base_linear).__name__}."
            )
        key = (layer, parent_name, child_name)
        original[key] = _OriginalLinearState(
            module=base_linear,
            requires_grad=tuple(
                (name, parameter.requires_grad)
                for name, parameter in base_linear.named_parameters()
            ),
        )
        locations.append((parent_name, child_name, parent, base_linear))

    adapters: list[tuple[nn.Module, str, LoRAAdapter]] = []
    try:
        for _parent_name, child_name, parent, base_linear in locations:
            adapter = LoRAAdapter(
                base=base_linear,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
            )
            # LoRAAdapter initialises ``lora_A`` / ``lora_B`` on CPU via
            # ``torch.empty`` / ``torch.zeros``. Move it to the base layer's
            # device before installing it in the model graph.
            adapter = adapter.to(base_linear.weight.device)
            adapters.append((parent, child_name, adapter))
        for parent, child_name, adapter in adapters:
            setattr(parent, child_name, adapter)
    except BaseException:
        # Construction freezes base parameters before the adapter is
        # installed. Restore both structure and trainability even when an
        # exception occurs partway through construction or installation.
        for parent_name, child_name, parent, _base_linear in locations:
            state = original[(layer, parent_name, child_name)]
            setattr(parent, child_name, state.module)
            _restore_original_linear(state)
        raise
    return original


def _detach_lora_from_model(
    model: nn.Module,
    *,
    original: dict[tuple[int, str, str], _OriginalLinearState],
) -> None:
    """Restore the wrapped sublayers to their original ``Linear`` modules.

    For every key in ``original`` (the dict returned by
    :func:`_attach_lora_to_model`), set the corresponding sublayer back
    to the original module via ``setattr``. After this call the model is
    byte-for-byte identical to the pre-attach state, except for any base
    weights that may have moved due to optimizer steps (LoRA only trains
    ``lora_A``/``lora_B``, so base weights are unchanged).
    """
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    for (layer_idx, parent_name, child_name), state in original.items():
        block = blocks[layer_idx]
        parent = getattr(block, parent_name)
        _restore_original_linear(state)
        setattr(parent, child_name, state.module)


def _collect_attached_adapters(
    model: nn.Module,
    *,
    layer: int,
) -> list[_AttachedAdapterRef]:
    """Walk the attached sublayers and return their ``LoRAAdapter`` refs.

    Used after :func:`_attach_lora_to_model` to build the optimizer
    param list (training) or to look up the adapter for weight
    assignment (loader). Kept as a separate helper so tests can
    introspect what got attached.
    """
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    block = blocks[layer]
    refs: list[_AttachedAdapterRef] = []
    for parent_name, child_name in _ATTACH_PATHS:
        parent = getattr(block, parent_name)
        adapter = getattr(parent, child_name)
        if not isinstance(adapter, LoRAAdapter):
            raise AxBenchBaseError(
                f"Expected LoRAAdapter at layer={layer} "
                f"{parent_name}.{child_name}, got {type(adapter).__name__}. "
                "Did you call _attach_lora_to_model first?"
            )
        refs.append(
            _AttachedAdapterRef(
                layer=layer,
                parent_name=parent_name,
                child_name=child_name,
                adapter=adapter,
            )
        )
    return refs
