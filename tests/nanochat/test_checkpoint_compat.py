"""Tests for backward-compatible loading of old checkpoints with separate c_q/c_k/c_v weights."""

from __future__ import annotations

import torch

from nanochat.checkpoint_manager import _fuse_qkv_weights
from nanochat.gpt import GPT, GPTConfig


def _make_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=128,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )


def _build_old_state_dict(config: GPTConfig) -> dict[str, torch.Tensor]:
    """Build a state dict with separate c_q/c_k/c_v (old checkpoint format).

    Takes a real model's state dict and splits each layer's c_qkv.weight
    into the old c_q, c_k, c_v format.
    """
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=config.vocab_size)
    model = model.to_empty(device="cpu")
    model.init_weights()
    state = model.state_dict()

    head_dim = config.n_embd // config.n_head
    q_dim = config.n_head * head_dim
    kv_dim = config.n_kv_head * head_dim

    new_state: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.endswith(".attn.c_qkv.weight"):
            prefix = key.removesuffix("c_qkv.weight")
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            new_state[f"{prefix}c_q.weight"] = q
            new_state[f"{prefix}c_k.weight"] = k
            new_state[f"{prefix}c_v.weight"] = v
        else:
            new_state[key] = val
    return new_state


class TestFuseQkvWeights:
    """Unit tests for _fuse_qkv_weights remapping."""

    def test_fuses_separate_qkv_into_single_key(self):
        config = _make_config()
        old_state = _build_old_state_dict(config)
        # Verify old format has c_q/c_k/c_v and no c_qkv
        assert any("c_q.weight" in k for k in old_state)
        assert not any("c_qkv.weight" in k for k in old_state)

        _fuse_qkv_weights(old_state)

        # After fusion: c_qkv present, c_q/c_k/c_v gone
        assert all("c_q.weight" not in k for k in old_state)
        assert all("c_k.weight" not in k for k in old_state)
        assert all("c_v.weight" not in k for k in old_state)
        assert any("c_qkv.weight" in k for k in old_state)

    def test_fused_weights_load_correctly(self):
        """Split→fuse round-trip should produce a loadable state dict identical to the original."""
        config = _make_config()

        # Build one model, grab its state as the reference
        with torch.device("meta"):
            ref_model = GPT(config, pad_vocab_size_to=config.vocab_size)
        ref_model = ref_model.to_empty(device="cpu")
        ref_model.init_weights()
        ref_model.eval()
        ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

        # Simulate old checkpoint: split c_qkv -> c_q, c_k, c_v from the same weights
        head_dim = config.n_embd // config.n_head
        q_dim = config.n_head * head_dim
        kv_dim = config.n_kv_head * head_dim
        old_state: dict[str, torch.Tensor] = {}
        for key, val in ref_state.items():
            if key.endswith(".attn.c_qkv.weight"):
                prefix = key.removesuffix("c_qkv.weight")
                q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
                old_state[f"{prefix}c_q.weight"] = q
                old_state[f"{prefix}c_k.weight"] = k
                old_state[f"{prefix}c_v.weight"] = v
            else:
                old_state[key] = val

        # Fuse and load into a fresh model
        _fuse_qkv_weights(old_state)

        with torch.device("meta"):
            new_model = GPT(config, pad_vocab_size_to=config.vocab_size)
        new_model = new_model.to_empty(device="cpu")
        new_model.init_weights()
        new_model.eval()
        new_model.load_state_dict(old_state, strict=True, assign=True)

        # The fused weights must exactly match the originals
        for key in ref_state:
            assert torch.equal(ref_state[key], new_model.state_dict()[key]), (
                f"Mismatch on {key}"
            )

    def test_no_op_on_already_fused_state_dict(self):
        """If the state dict already has c_qkv, _fuse_qkv_weights should be a no-op."""
        config = _make_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=config.vocab_size)
        model = model.to_empty(device="cpu")
        model.init_weights()
        state = model.state_dict()
        original_keys = set(state.keys())

        _fuse_qkv_weights(state)

        assert set(state.keys()) == original_keys

    def test_fused_shape_is_correct(self):
        config = _make_config()
        head_dim = config.n_embd // config.n_head
        expected_rows = (config.n_head + 2 * config.n_kv_head) * head_dim

        old_state = _build_old_state_dict(config)
        _fuse_qkv_weights(old_state)

        for key, val in old_state.items():
            if key.endswith(".attn.c_qkv.weight"):
                assert val.shape == (expected_rows, config.n_embd), (
                    f"{key}: expected ({expected_rows}, {config.n_embd}), got {val.shape}"
                )

    def test_works_with_gkv_heads(self):
        """Test with grouped-query attention (n_kv_head < n_head)."""
        config = GPTConfig(
            sequence_len=128,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )
        old_state = _build_old_state_dict(config)
        _fuse_qkv_weights(old_state)

        head_dim = config.n_embd // config.n_head
        expected_rows = (config.n_head + 2 * config.n_kv_head) * head_dim

        # Should load without error
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=config.vocab_size)
        model = model.to_empty(device="cpu")
        model.init_weights()
        model.load_state_dict(old_state, strict=True, assign=True)

        for key, val in old_state.items():
            if key.endswith(".attn.c_qkv.weight"):
                assert val.shape[0] == expected_rows
