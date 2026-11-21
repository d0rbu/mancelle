"""Unit tests for the LoRA adapter loader (Task 25, axbench_base Stage 3b).

Covers
``steering.eval.axbench_base.baselines.lora.loader.attach_lora_from_disk``:

  * Attach ``LoRAAdapter`` wrappers onto the four nanochat ``Linear``
    sublayers of a target block at a given layer, loading their
    ``lora_A`` / ``lora_B`` weights from a safetensors file written by
    Task 24 (``training._save_lora_adapter``).
  * Return a ``detach_fn`` that restores the original ``nn.Linear``
    modules byte-for-byte (identity preserved).
  * Idempotent detach: calling ``detach_fn`` twice must not raise.
  * Missing required keys raise a clean ``AxBenchBaseError``.
  * Nonexistent files and corrupt files raise clear errors.
  * Loaded adapter weights match the saved weights bit-for-bit.
  * Loaded adapter weights are cast to the base Linear's weight dtype.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch as t
import torch.nn as nn
from safetensors.torch import load_file, save_file

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from nanochat.gpt import Linear as NanochatLinear
from steering.eval.axbench_base.baselines.lora._attach import _ATTACH_PATHS
from steering.eval.axbench_base.baselines.lora.loader import (
    attach_lora_from_disk,
)
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.baselines.lora.training import (
    _attach_lora_to_model,
    _collect_attached_adapters,
    _detach_lora_from_model,
    _save_lora_adapter,
)
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.types import AxBenchBaseError


@pytest.fixture(autouse=True)
def _force_sdpa() -> Iterator[None]:
    """Force SDPA attention backend for every test in this module.

    FlexAttention does not support backward on CPU; we force SDPA so
    the tiny GPT fixture's ``forward`` runs on CPU without crashing.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Tiny GPT fixture (mirrors test_lora_training.py)
# ---------------------------------------------------------------------------


TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_BOS_ID: int = 1
TINY_LAYER: int = 0  # which block to attach adapters to


def _make_tiny_gpt(seed: int = 0xD24F) -> GPT:
    """Return a tiny CPU GPT in eval mode, fully initialised."""
    t.manual_seed(seed)
    config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _tiny_lora_cfg(rank: int = 2, alpha: float = 4.0) -> LoraConfig:
    return LoraConfig(rank=rank, alpha=alpha)


def _train_and_save_adapter(
    model: nn.Module,
    out_path: Path,
    *,
    layer: int = TINY_LAYER,
    lora_cfg: LoraConfig | None = None,
    randomise_weights: bool = True,
) -> dict[str, t.Tensor]:
    """Attach a fresh adapter set, mutate the LoRA weights, and save.

    This does NOT run any real training — it just fills ``lora_A`` /
    ``lora_B`` with small random values so the loaded adapter has
    non-trivial weights (and thus the loader can verify that weights
    were actually installed, not just zero-initialised).

    Detaches before returning so the caller's model is left byte-for-
    byte identical to the pre-attach state.

    Returns the dict of key -> tensor that was persisted, for the
    ``test_loaded_weights_match_saved_weights`` assertion.
    """
    cfg = lora_cfg or _tiny_lora_cfg()
    original = _attach_lora_to_model(model, layer=layer, lora_cfg=cfg)
    try:
        refs = _collect_attached_adapters(model, layer=layer)
        if randomise_weights:
            t.manual_seed(0xBEEF)
            for ref in refs:
                with t.no_grad():
                    ref.adapter.lora_A.copy_(
                        t.randn_like(ref.adapter.lora_A) * 0.05
                    )
                    ref.adapter.lora_B.copy_(
                        t.randn_like(ref.adapter.lora_B) * 0.05
                    )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_lora_adapter(
            refs=refs,
            out_path=out_path,
            metadata={"layer": str(layer)},
        )
    finally:
        _detach_lora_from_model(model, original=original)
    return load_file(str(out_path))


def _make_bos_inputs() -> t.Tensor:
    """Tiny CPU input tensor for forward-pass tests."""
    return t.tensor([[TINY_BOS_ID, 2, 3, 4, 5]], dtype=t.long)


# ---------------------------------------------------------------------------
# Happy-path attach/detach
# ---------------------------------------------------------------------------


class TestAttachDetachHappyPath:
    def test_attach_then_detach_restores_identity(self, tmp_path: Path) -> None:
        """After detach, every attn/mlp linear is the SAME object (id())
        as it was before attach."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()

        # Capture pre-attach identities.
        block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        before_ids: dict[tuple[str, str], int] = {}
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            before_ids[(parent_name, child_name)] = id(
                getattr(parent, child_name)
            )

        # Save an adapter file to disk.
        adapter_path = tmp_path / "concept_0042.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        assert len(adapters) == 4
        for adapter in adapters:
            assert isinstance(adapter, LoRAAdapter)

        detach_fn()

        block_after = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block_after, parent_name)
            current = getattr(parent, child_name)
            assert id(current) == before_ids[(parent_name, child_name)], (
                f"{parent_name}.{child_name} identity changed after detach"
            )
            assert isinstance(current, NanochatLinear)
            assert not isinstance(current, LoRAAdapter)

    def test_attach_returns_adapters_in_attach_paths_order(
        self, tmp_path: Path,
    ) -> None:
        """The returned adapters list must be ordered matching ``_ATTACH_PATHS``
        so callers can zip them against the names if they need to."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        try:
            block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
            for adapter, (parent_name, child_name) in zip(
                adapters, _ATTACH_PATHS, strict=True,
            ):
                parent = getattr(block, parent_name)
                assert getattr(parent, child_name) is adapter
        finally:
            detach_fn()

    def test_attach_only_touches_requested_layer(self, tmp_path: Path) -> None:
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        other_layer = 1
        assert other_layer != TINY_LAYER

        _adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        try:
            other_block = cast(nn.ModuleList, model.transformer["h"])[other_layer]
            for parent_name, child_name in _ATTACH_PATHS:
                parent = getattr(other_block, parent_name)
                current = getattr(parent, child_name)
                assert not isinstance(current, LoRAAdapter), (
                    f"layer {other_layer}.{parent_name}.{child_name} should "
                    "have been untouched"
                )
        finally:
            detach_fn()


# ---------------------------------------------------------------------------
# Forward pass sanity
# ---------------------------------------------------------------------------


class TestForwardWithAdapter:
    def test_forward_uses_adapter_while_attached(self, tmp_path: Path) -> None:
        """Output after attach (with non-zero lora_B) must differ from the
        pre-attach output, and match the base output again after detach."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        idx = _make_bos_inputs()
        with t.no_grad():
            base_logits = model(idx).clone()

        _adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        try:
            with t.no_grad():
                adapted_logits = model(idx).clone()
        finally:
            detach_fn()

        # After detach we should see the base model forward again, exactly.
        with t.no_grad():
            restored_logits = model(idx).clone()

        # Loaded adapter has lora_A, lora_B != 0, so the output must differ.
        assert not t.allclose(base_logits, adapted_logits, atol=1e-6), (
            "Forward output should differ while adapter attached; either "
            "lora_A or lora_B was not installed correctly."
        )
        # And restoring should produce the EXACT same logits as before.
        assert t.allclose(base_logits, restored_logits, atol=1e-6), (
            "After detach, model forward must match pre-attach forward."
        )


# ---------------------------------------------------------------------------
# Detach idempotency
# ---------------------------------------------------------------------------


class TestDetachIdempotent:
    def test_detach_is_idempotent(self, tmp_path: Path) -> None:
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        _adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        detach_fn()
        # Calling again must not raise, and must leave the model in the
        # same (already-detached) state.
        detach_fn()

        block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            current = getattr(parent, child_name)
            assert not isinstance(current, LoRAAdapter)


# ---------------------------------------------------------------------------
# Missing / corrupt / nonexistent adapter files
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    @pytest.mark.parametrize("suffix", ["lora_A", "lora_B"])
    def test_missing_adapter_key_raises_clear_error(
        self, tmp_path: Path, suffix: str,
    ) -> None:
        """If the safetensors file is missing either the A or the B key for
        any wrapped linear, loader must raise ``AxBenchBaseError`` with a
        clear message and leave the model untouched."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        saved = _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        missing_key = f"layer_{TINY_LAYER}_attn_c_qkv_{suffix}"
        assert missing_key in saved
        broken = {k: v for k, v in saved.items() if k != missing_key}
        save_file(broken, str(adapter_path))

        with pytest.raises(AxBenchBaseError, match="Missing adapter key"):
            attach_lora_from_disk(
                model, adapter_path, lora_cfg, layer=TINY_LAYER,
            )
        # Model must still be in its original (non-adapter) state.
        block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            assert not isinstance(getattr(parent, child_name), LoRAAdapter)

    def test_attach_with_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loader must raise a clear error when the adapter path doesn't
        exist. We accept either ``AxBenchBaseError`` or ``FileNotFoundError``;
        both surface the problem unambiguously."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "does_not_exist.pt"

        with pytest.raises((AxBenchBaseError, FileNotFoundError)):
            attach_lora_from_disk(
                model, adapter_path, lora_cfg, layer=TINY_LAYER,
            )

    def test_attach_with_corrupt_file_raises(self, tmp_path: Path) -> None:
        """Feeding a non-safetensors blob should raise a clear error."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "corrupt.pt"
        adapter_path.write_bytes(b"this is definitely not a safetensors file")

        with pytest.raises(AxBenchBaseError, match="Failed to load"):
            attach_lora_from_disk(
                model, adapter_path, lora_cfg, layer=TINY_LAYER,
            )


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtypeMatch:
    def test_adapter_dtype_matches_base(self, tmp_path: Path) -> None:
        """Loaded adapter weights must be in the same dtype as the base
        Linear's ``weight`` tensor."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        try:
            for adapter in adapters:
                base_dtype = adapter.base.weight.dtype
                assert adapter.lora_A.dtype == base_dtype, (
                    f"lora_A dtype {adapter.lora_A.dtype} != "
                    f"base dtype {base_dtype}"
                )
                assert adapter.lora_B.dtype == base_dtype, (
                    f"lora_B dtype {adapter.lora_B.dtype} != "
                    f"base dtype {base_dtype}"
                )
        finally:
            detach_fn()


# ---------------------------------------------------------------------------
# Exact weight preservation
# ---------------------------------------------------------------------------


class TestLoadedWeightsMatchSaved:
    def test_loaded_weights_match_saved_weights(self, tmp_path: Path) -> None:
        """Save → load → compare lora_A / lora_B tensors bit-for-bit (in
        the base dtype)."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        saved = _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        try:
            for adapter, (parent_name, child_name) in zip(
                adapters, _ATTACH_PATHS, strict=True,
            ):
                prefix = (
                    f"layer_{TINY_LAYER}_{parent_name}_{child_name}"
                )
                expected_A = saved[f"{prefix}_lora_A"].to(
                    dtype=adapter.base.weight.dtype,
                )
                expected_B = saved[f"{prefix}_lora_B"].to(
                    dtype=adapter.base.weight.dtype,
                )
                assert t.equal(adapter.lora_A.detach().cpu(), expected_A), (
                    f"{prefix}: lora_A does not match saved tensor"
                )
                assert t.equal(adapter.lora_B.detach().cpu(), expected_B), (
                    f"{prefix}: lora_B does not match saved tensor"
                )
        finally:
            detach_fn()


# ---------------------------------------------------------------------------
# Path-string compatibility
# ---------------------------------------------------------------------------


class TestAcceptsPathLike:
    def test_attach_accepts_pathlib_path(self, tmp_path: Path) -> None:
        """``attach_lora_from_disk`` must accept ``Path`` objects (not just
        strings)."""
        model = _make_tiny_gpt()
        lora_cfg = _tiny_lora_cfg()
        adapter_path = tmp_path / "concept.pt"
        _train_and_save_adapter(model, adapter_path, lora_cfg=lora_cfg)

        assert isinstance(adapter_path, Path)
        _adapters, detach_fn = attach_lora_from_disk(
            model, adapter_path, lora_cfg, layer=TINY_LAYER,
        )
        detach_fn()
