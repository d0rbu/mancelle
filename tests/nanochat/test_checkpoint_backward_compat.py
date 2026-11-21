"""Tests for backward compatibility patching in checkpoint loading.

Covers:
- _patch_missing_config_keys: adds default window_pattern for old checkpoints
- _patch_missing_keys: adds default resid_lambdas / x0_lambdas tensors
- _recover_model_config: recovers model_config from parent or user_config
- _recover_parent_user_config: loads user_config from parent checkpoint
- End-to-end: build_model with an old-format checkpoint
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nanochat.checkpoint_manager import (
    _patch_missing_config_keys,
    _patch_missing_keys,
    _recover_model_config,
    _recover_parent_user_config,
    build_model,
    save_checkpoint,
)
from nanochat.gpt import GPT, GPTConfig
from tests.conftest import (
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    MockTokenizer,
    _make_gpt,
    _make_gpt_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_old_state_dict(config: GPTConfig) -> dict[str, torch.Tensor]:
    """Build a state dict with separate c_q/c_k/c_v (old format) and no lambda params."""
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=config.vocab_size)
    model = model.to_empty(device="cpu")
    model.init_weights()
    state = model.state_dict()

    head_dim = config.n_embd // config.n_head
    q_dim = config.n_head * head_dim
    kv_dim = config.n_kv_head * head_dim

    old_state: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.endswith(".attn.c_qkv.weight"):
            prefix = key.removesuffix("c_qkv.weight")
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            old_state[f"{prefix}c_q.weight"] = q
            old_state[f"{prefix}c_k.weight"] = k
            old_state[f"{prefix}c_v.weight"] = v
        elif key in ("resid_lambdas", "x0_lambdas"):
            # Old checkpoints did not have these
            continue
        else:
            old_state[key] = val
    return old_state


# ===================================================================
# 1. _patch_missing_config_keys
# ===================================================================


class TestPatchMissingConfigKeys:
    """Tests for _patch_missing_config_keys."""

    def test_missing_window_pattern_defaults_to_L(self):
        cfg: dict = {"n_layer": 4, "n_head": 2}
        _patch_missing_config_keys(cfg)
        assert cfg["window_pattern"] == "L"

    def test_existing_window_pattern_not_overwritten(self):
        cfg: dict = {"window_pattern": "SSSL", "n_layer": 4}
        _patch_missing_config_keys(cfg)
        assert cfg["window_pattern"] == "SSSL"

    def test_single_char_pattern_preserved(self):
        cfg: dict = {"window_pattern": "S"}
        _patch_missing_config_keys(cfg)
        assert cfg["window_pattern"] == "S"

    def test_idempotent(self):
        cfg: dict = {"n_layer": 4}
        _patch_missing_config_keys(cfg)
        assert cfg["window_pattern"] == "L"
        _patch_missing_config_keys(cfg)
        assert cfg["window_pattern"] == "L"

    def test_other_keys_not_affected(self):
        cfg: dict = {"n_layer": 4, "n_head": 8, "n_embd": 512}
        original = cfg.copy()
        _patch_missing_config_keys(cfg)
        for key in original:
            assert cfg[key] == original[key], f"Key {key} was modified"

    def test_empty_dict(self):
        cfg: dict = {}
        _patch_missing_config_keys(cfg)
        assert cfg == {"window_pattern": "L"}


# ===================================================================
# 2. _patch_missing_keys
# ===================================================================


class TestPatchMissingKeys:
    """Tests for _patch_missing_keys."""

    def test_missing_resid_lambdas_defaults_to_ones(self):
        config = _make_gpt_config()
        model_data: dict = {}
        _patch_missing_keys(model_data, config)
        assert "resid_lambdas" in model_data
        expected = torch.ones(config.n_layer)
        assert torch.equal(model_data["resid_lambdas"], expected)

    def test_missing_x0_lambdas_defaults_to_zeros(self):
        config = _make_gpt_config()
        model_data: dict = {}
        _patch_missing_keys(model_data, config)
        assert "x0_lambdas" in model_data
        expected = torch.zeros(config.n_layer)
        assert torch.equal(model_data["x0_lambdas"], expected)

    def test_existing_resid_lambdas_not_overwritten(self):
        config = _make_gpt_config()
        custom = torch.tensor([0.5, 0.8])
        model_data = {"resid_lambdas": custom}
        _patch_missing_keys(model_data, config)
        assert torch.equal(model_data["resid_lambdas"], custom)

    def test_existing_x0_lambdas_not_overwritten(self):
        config = _make_gpt_config()
        custom = torch.tensor([0.3, 0.7])
        model_data = {"x0_lambdas": custom}
        _patch_missing_keys(model_data, config)
        assert torch.equal(model_data["x0_lambdas"], custom)

    def test_shape_matches_n_layer(self):
        for n_layer in (1, 2, 4, 8):
            config = GPTConfig(
                sequence_len=128,
                vocab_size=128,
                n_layer=n_layer,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="L",
            )
            model_data: dict = {}
            _patch_missing_keys(model_data, config)
            assert model_data["resid_lambdas"].shape == (n_layer,)
            assert model_data["x0_lambdas"].shape == (n_layer,)

    def test_default_values_mean_identity_behavior(self):
        """resid_lambdas=1 means identity scaling, x0_lambdas=0 means disabled."""
        config = _make_gpt_config()
        model_data: dict = {}
        _patch_missing_keys(model_data, config)
        # All ones => residual stream is unscaled (identity)
        assert torch.all(model_data["resid_lambdas"] == 1.0)
        # All zeros => no x0 skip connection (disabled)
        assert torch.all(model_data["x0_lambdas"] == 0.0)

    def test_patched_state_dict_loads_strict(self):
        """Patched state dict loads into a GPT model with strict=True."""
        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=config.vocab_size)
        model = model.to_empty(device="cpu")
        model.init_weights()
        state = model.state_dict()

        # Remove lambdas to simulate old checkpoint
        del state["resid_lambdas"]
        del state["x0_lambdas"]

        # Patch them back
        _patch_missing_keys(state, config)

        # Load into a fresh model -- must not raise
        with torch.device("meta"):
            model2 = GPT(config, pad_vocab_size_to=config.vocab_size)
        model2 = model2.to_empty(device="cpu")
        model2.init_weights()
        model2.load_state_dict(state, strict=True, assign=True)

    def test_patched_vs_explicit_output_identical(self):
        """Model with patched default lambdas produces the same output as one with explicit defaults."""
        config = _make_gpt_config()

        # Model A: explicit default lambdas
        with torch.device("meta"):
            model_a = GPT(config, pad_vocab_size_to=config.vocab_size)
        model_a = model_a.to_empty(device="cpu")
        model_a.init_weights()
        model_a.eval()
        state_a = model_a.state_dict()

        # Model B: patched lambdas (remove then patch)
        state_b = {k: v.clone() for k, v in state_a.items()}
        state_b["resid_lambdas"] = torch.ones(config.n_layer)
        state_b["x0_lambdas"] = torch.zeros(config.n_layer)

        with torch.device("meta"):
            model_b = GPT(config, pad_vocab_size_to=config.vocab_size)
        model_b = model_b.to_empty(device="cpu")
        model_b.init_weights()
        model_b.eval()
        model_b.load_state_dict(state_b, strict=True, assign=True)

        # Same input => same output
        torch.manual_seed(42)
        ids = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out_a = model_a(ids)
            out_b = model_b(ids)
        assert torch.allclose(out_a, out_b, atol=1e-3), "Outputs differ"

    def test_idempotent(self):
        config = _make_gpt_config()
        model_data: dict = {}
        _patch_missing_keys(model_data, config)
        first_resid = model_data["resid_lambdas"].clone()
        first_x0 = model_data["x0_lambdas"].clone()
        _patch_missing_keys(model_data, config)
        assert torch.equal(model_data["resid_lambdas"], first_resid)
        assert torch.equal(model_data["x0_lambdas"], first_x0)


# ===================================================================
# 3. _recover_model_config
# ===================================================================


class TestRecoverModelConfig:
    """Tests for _recover_model_config."""

    def test_strategy1_parent_has_model_config(self, tmp_path):
        """When the parent directory has a checkpoint with model_config, use it."""
        parent_dir = tmp_path / "parent"
        expected_config = {
            "sequence_len": 256,
            "vocab_size": 32768,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 128,
            "window_pattern": "SL",
        }
        # Save a parent checkpoint with model_config
        parent_meta = {"model_config": expected_config, "user_config": {"depth": 4}}
        save_checkpoint(str(parent_dir), 0, {"dummy": torch.zeros(1)}, None, parent_meta)

        # Child checkpoint directory is inside parent
        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir(parents=True)
        child_meta: dict = {"user_config": {"depth": 4}}

        result = _recover_model_config(str(sft_dir), child_meta)
        assert result == expected_config

    def test_strategy1_parent_no_checkpoint_files_falls_through_to_strategy2(self, tmp_path):
        """When the parent directory exists but has no model_*.pt files,
        find_last_step_or_none returns None (issue #443 fix) and _recover_model_config
        falls through to Strategy 2 (reconstruct from user_config)."""
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()

        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir()
        child_meta = {"user_config": {"depth": 2, "aspect_ratio": 32, "head_dim": 32}}

        result = _recover_model_config(str(sft_dir), child_meta)
        # Strategy 2: reconstructed from user_config, not raised.
        assert result["n_layer"] == 2
        assert result["n_embd"] == 2 * 32

    def test_strategy1_parent_meta_no_model_config_falls_through(self, tmp_path):
        """Parent has a meta file but without model_config key -> fall through."""
        parent_dir = tmp_path / "parent"
        # Save parent checkpoint without model_config in meta
        parent_meta: dict = {"user_config": {"depth": 4}}
        save_checkpoint(str(parent_dir), 0, {"dummy": torch.zeros(1)}, None, parent_meta)

        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir()
        child_meta = {
            "user_config": {
                "depth": 4,
                "aspect_ratio": 64,
                "head_dim": 128,
                "max_seq_len": 1024,
                "window_pattern": "SSL",
            }
        }

        result = _recover_model_config(str(sft_dir), child_meta)
        # Should have used strategy 2
        assert result["n_layer"] == 4
        assert result["n_embd"] == 4 * 64
        assert result["sequence_len"] == 1024
        assert result["window_pattern"] == "SSL"

    def test_strategy2_reconstruct_all_fields(self, tmp_path):
        """Reconstruct model_config from user_config with all fields present."""
        # Use a path whose parent doesn't exist so os.path.isdir returns False,
        # skipping strategy 1 entirely.
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        meta = {
            "user_config": {
                "depth": 8,
                "aspect_ratio": 32,
                "head_dim": 64,
                "max_seq_len": 512,
                "window_pattern": "SSSL",
            }
        }
        result = _recover_model_config(sft_dir, meta)
        n_embd = 8 * 32  # 256
        n_head = n_embd // 64  # 4
        assert result == {
            "sequence_len": 512,
            "vocab_size": 32768,
            "n_layer": 8,
            "n_head": n_head,
            "n_kv_head": n_head,
            "n_embd": n_embd,
            "window_pattern": "SSSL",
        }

    def test_strategy2_defaults(self, tmp_path):
        """Reconstruct with all defaults when user_config is empty."""
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        meta: dict = {"user_config": {}}
        result = _recover_model_config(sft_dir, meta)
        # Defaults: depth=26, aspect_ratio=64, head_dim=128
        n_embd = 26 * 64  # 1664
        n_head = n_embd // 128  # 13
        assert result["n_layer"] == 26
        assert result["n_embd"] == n_embd
        assert result["n_head"] == n_head
        assert result["n_kv_head"] == n_head
        assert result["sequence_len"] == 2048
        assert result["window_pattern"] == "SSSL"

    def test_strategy2_no_user_config(self, tmp_path):
        """When user_config key is missing entirely, use all defaults."""
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        meta: dict = {}
        result = _recover_model_config(sft_dir, meta)
        assert result["n_layer"] == 26

    def test_reconstructed_config_creates_valid_gptconfig(self, tmp_path):
        """The reconstructed dict should be valid kwargs for GPTConfig."""
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        meta = {
            "user_config": {
                "depth": 4,
                "aspect_ratio": 32,
                "head_dim": 64,
                "max_seq_len": 256,
            }
        }
        result = _recover_model_config(sft_dir, meta)
        # Should not raise
        gpt_config = GPTConfig(**result)
        assert gpt_config.n_layer == 4
        assert gpt_config.n_embd == 128
        assert gpt_config.n_head == 2

    def test_vocab_size_hardcoded_32768(self, tmp_path):
        """Strategy 2 always sets vocab_size to 32768 (the project tokenizer size)."""
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        # Even if user_config has a different value, vocab_size is hardcoded
        meta = {"user_config": {"depth": 2, "aspect_ratio": 32, "head_dim": 32}}
        result = _recover_model_config(sft_dir, meta)
        assert result["vocab_size"] == 32768

    def test_strategy1_uses_latest_parent_step(self, tmp_path):
        """When parent has multiple steps, use the latest one."""
        parent_dir = tmp_path / "parent"

        # Save two checkpoints at different steps
        config_v1 = {"sequence_len": 128, "vocab_size": 32768, "n_layer": 2}
        config_v2 = {"sequence_len": 256, "vocab_size": 32768, "n_layer": 4}

        save_checkpoint(
            str(parent_dir), 0, {"dummy": torch.zeros(1)}, None,
            {"model_config": config_v1},
        )
        save_checkpoint(
            str(parent_dir), 100, {"dummy": torch.zeros(1)}, None,
            {"model_config": config_v2},
        )

        sft_dir = parent_dir / "sft_checkpoints"
        sft_dir.mkdir()
        # find_last_step returns step 100, so we should get config_v2
        result = _recover_model_config(str(sft_dir), {})
        assert result == config_v2


# ===================================================================
# 4. _recover_parent_user_config
# ===================================================================


class TestRecoverParentUserConfig:
    """Tests for _recover_parent_user_config."""

    def test_parent_has_user_config(self, tmp_path):
        parent_dir = tmp_path / "parent"
        user_config = {"depth": 4, "aspect_ratio": 64, "head_dim": 128}
        save_checkpoint(
            str(parent_dir), 0, {"dummy": torch.zeros(1)}, None,
            {"user_config": user_config},
        )

        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir()

        result = _recover_parent_user_config(str(sft_dir))
        assert result == user_config

    def test_parent_no_user_config_raises(self, tmp_path):
        parent_dir = tmp_path / "parent"
        save_checkpoint(
            str(parent_dir), 0, {"dummy": torch.zeros(1)}, None,
            {"model_config": {"n_layer": 4}},
        )

        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Cannot recover"):
            _recover_parent_user_config(str(sft_dir))

    def test_parent_exists_but_no_checkpoints_raises_domain_error(self, tmp_path):
        """Parent directory exists but has no model_*.pt files -- after issue #443 fix,
        find_last_step_or_none returns None and _recover_parent_user_config raises the
        domain-level FileNotFoundError (not find_last_step's internal message)."""
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        sft_dir = parent_dir / "standard_sft_checkpoints"
        sft_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Cannot recover encoder/cross_attention config"):
            _recover_parent_user_config(str(sft_dir))

    def test_no_parent_directory_raises(self, tmp_path):
        """When the parent path is not a real directory, raises 'Cannot recover'."""
        # sft_dir whose dirname does not exist as a directory
        sft_dir = str(tmp_path / "nonexistent_parent" / "sft")
        # Don't create the directories -- parent doesn't exist
        # os.path.isdir returns False -> parent_step is None -> raises
        with pytest.raises(FileNotFoundError, match="Cannot recover"):
            _recover_parent_user_config(sft_dir)

    def test_trailing_slash_handled(self, tmp_path):
        """checkpoint_dir with trailing slash should still find the parent."""
        parent_dir = tmp_path / "parent"
        user_config = {"depth": 2}
        save_checkpoint(
            str(parent_dir), 0, {"dummy": torch.zeros(1)}, None,
            {"user_config": user_config},
        )

        sft_dir = parent_dir / "sft_checkpoints"
        sft_dir.mkdir()

        # Pass with trailing slash
        result = _recover_parent_user_config(str(sft_dir) + "/")
        assert result == user_config


# ===================================================================
# 5. End-to-end: build_model with old checkpoint format
# ===================================================================


class TestBuildModelOldFormat:
    """End-to-end test: build_model loads an old-format checkpoint with all patches."""

    def test_build_model_patches_everything(self, tmp_path):
        """Create a checkpoint missing window_pattern, lambdas, and with
        separate c_q/c_k/c_v. Verify build_model patches everything and
        loads successfully."""
        config = _make_gpt_config()

        # Build a reference model to get weights
        ref_model = _make_gpt()
        ref_state = ref_model.state_dict()

        # Simulate an old checkpoint:
        # 1) Split c_qkv -> c_q, c_k, c_v
        # 2) Remove resid_lambdas and x0_lambdas
        # 3) Remove window_pattern from the config
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
            elif key in ("resid_lambdas", "x0_lambdas"):
                continue  # skip -- old checkpoint
            else:
                old_state[key] = val

        # Meta without window_pattern
        model_config = {
            "sequence_len": TINY_SEQ_LEN,
            "vocab_size": TINY_VOCAB,
            "n_layer": TINY_N_LAYER,
            "n_head": TINY_N_HEAD,
            "n_kv_head": TINY_N_KV_HEAD,
            "n_embd": TINY_N_EMBD,
            # no window_pattern
        }
        meta = {"model_config": model_config}

        checkpoint_dir = str(tmp_path / "old_checkpoint")
        save_checkpoint(checkpoint_dir, 0, old_state, None, meta)

        # Patch get_tokenizer to return MockTokenizer
        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            model, tokenizer, loaded_meta = build_model(
                checkpoint_dir, 0, torch.device("cpu"), "eval"
            )

        # Model loaded successfully
        assert isinstance(model, GPT)
        assert model.config.window_pattern == "L"
        assert model.config.n_layer == TINY_N_LAYER

        # Verify model can run a forward pass
        ids = torch.randint(0, TINY_VOCAB, (1, 16))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 16, TINY_VOCAB)

    def test_build_model_preserves_existing_values(self, tmp_path):
        """build_model must not overwrite existing window_pattern or lambda values."""
        config = GPTConfig(
            sequence_len=TINY_SEQ_LEN,
            vocab_size=TINY_VOCAB,
            n_layer=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            n_embd=TINY_N_EMBD,
            window_pattern="L",
        )
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=config.vocab_size)
        model = model.to_empty(device="cpu")
        model.init_weights()
        state = model.state_dict()

        # Set custom lambda values
        state["resid_lambdas"] = torch.full((TINY_N_LAYER,), 0.42)
        state["x0_lambdas"] = torch.full((TINY_N_LAYER,), 0.13)

        meta = {
            "model_config": {
                "sequence_len": TINY_SEQ_LEN,
                "vocab_size": TINY_VOCAB,
                "n_layer": TINY_N_LAYER,
                "n_head": TINY_N_HEAD,
                "n_kv_head": TINY_N_KV_HEAD,
                "n_embd": TINY_N_EMBD,
                "window_pattern": "L",
            }
        }

        checkpoint_dir = str(tmp_path / "checkpoint")
        save_checkpoint(checkpoint_dir, 0, state, None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_model(
                checkpoint_dir, 0, torch.device("cpu"), "eval"
            )

        # Verify custom values survived
        assert torch.allclose(
            loaded_model.resid_lambdas, torch.full((TINY_N_LAYER,), 0.42)
        )
        assert torch.allclose(
            loaded_model.x0_lambdas, torch.full((TINY_N_LAYER,), 0.13)
        )

    def test_build_model_strips_orig_mod_prefix(self, tmp_path):
        """build_model strips _orig_mod. prefix from torch.compile keys."""
        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=config.vocab_size)
        model = model.to_empty(device="cpu")
        model.init_weights()
        state = model.state_dict()

        # Prepend _orig_mod. to all keys (simulating torch.compile checkpoint)
        compiled_state = {f"_orig_mod.{k}": v for k, v in state.items()}

        meta = {
            "model_config": {
                "sequence_len": TINY_SEQ_LEN,
                "vocab_size": TINY_VOCAB,
                "n_layer": TINY_N_LAYER,
                "n_head": TINY_N_HEAD,
                "n_kv_head": TINY_N_KV_HEAD,
                "n_embd": TINY_N_EMBD,
                "window_pattern": "L",
            }
        }

        checkpoint_dir = str(tmp_path / "compiled_ckpt")
        save_checkpoint(checkpoint_dir, 0, compiled_state, None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_model(
                checkpoint_dir, 0, torch.device("cpu"), "eval"
            )

        # Model loaded successfully despite _orig_mod. prefix
        ids = torch.randint(0, TINY_VOCAB, (1, 16))
        with torch.no_grad():
            logits = loaded_model(ids)
        assert logits.shape == (1, 16, TINY_VOCAB)
