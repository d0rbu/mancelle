"""Tests for checkpoint save/load roundtrip correctness."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from nanochat.checkpoint_manager import (
    build_encdec_model,
    build_model,
    find_last_step,
    load_checkpoint,
    save_checkpoint,
)
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from tests.conftest import (
    MockTokenizer,
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    _make_encdec,
    _make_gpt,
    _make_gpt_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_config_dict() -> dict:
    """Return a model_config dict matching the tiny GPT config fields."""
    return asdict(_make_gpt_config())


def _meta_data(step: int = 0, val_bpb: float = 3.14) -> dict:
    """Return a representative metadata dict."""
    return {
        "step": step,
        "model_config": _model_config_dict(),
        "user_config": {
            "depth": TINY_N_LAYER,
            "aspect_ratio": TINY_N_EMBD // TINY_N_LAYER,
            "head_dim": TINY_HEAD_DIM,
            "max_seq_len": TINY_SEQ_LEN,
        },
        "val_bpb": val_bpb,
    }


def _encdec_meta_data(step: int = 0, val_bpb: float = 2.71) -> dict:
    """Return metadata for an EncoderDecoderGPT checkpoint."""
    return {
        "step": step,
        "model_config": _model_config_dict(),
        "user_config": {
            "depth": TINY_N_LAYER,
            "aspect_ratio": TINY_N_EMBD // TINY_N_LAYER,
            "head_dim": TINY_HEAD_DIM,
            "max_seq_len": TINY_SEQ_LEN,
            "encoder": {
                "n_embd": TINY_N_EMBD,
                "depth": TINY_N_LAYER,
                "n_head": TINY_N_HEAD,
                "n_kv_head": TINY_N_KV_HEAD,
                "head_dim": TINY_HEAD_DIM,
            },
            "cross_attention": {
                "frequency": 1,
            },
            "num_buckets": TINY_NUM_BUCKETS,
        },
        "val_bpb": val_bpb,
    }


# ---------------------------------------------------------------------------
# 1. GPT save -> load roundtrip (bit-exact)
# ---------------------------------------------------------------------------


class TestGPTRoundtrip:
    """Save a tiny GPT state_dict, load it back, verify every tensor matches."""

    def test_model_state_dict_roundtrip(self, tmp_path):
        model = _make_gpt()
        original_sd = model.state_dict()
        step = 0
        meta = _meta_data(step=step)

        save_checkpoint(str(tmp_path), step, original_sd, None, meta, rank=0)
        loaded_sd, opt_data, loaded_meta = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )

        assert opt_data is None
        assert set(loaded_sd.keys()) == set(original_sd.keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], loaded_sd[key]), (
                f"Mismatch on key {key}"
            )


# ---------------------------------------------------------------------------
# 2. Optimizer state roundtrip
# ---------------------------------------------------------------------------


class TestOptimizerRoundtrip:
    """Save model + optimizer state, reload, verify optimizer tensors match."""

    def test_optimizer_state_roundtrip(self, tmp_path):
        model = _make_gpt()

        # Build a synthetic AdamW-style optimizer state dict with momentum
        # buffers and step counts. We cannot run a forward/backward pass on
        # CPU because FlexAttention does not support backward there.
        original_opt_sd = {
            "state": {},
            "param_groups": [
                {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.0,
                 "eps": 1e-8, "params": []},
            ],
        }
        for idx, p in enumerate(model.parameters()):
            original_opt_sd["param_groups"][0]["params"].append(idx)
            original_opt_sd["state"][idx] = {
                "step": torch.tensor(5.0),
                "exp_avg": torch.randn_like(p),
                "exp_avg_sq": torch.rand_like(p),
            }

        step = 1
        meta = _meta_data(step=step)

        save_checkpoint(
            str(tmp_path), step, model.state_dict(), original_opt_sd, meta, rank=0
        )
        _, loaded_opt_sd, _ = load_checkpoint(
            str(tmp_path), step, torch.device("cpu"), load_optimizer=True
        )

        assert loaded_opt_sd is not None
        # Compare param_groups (non-tensor metadata)
        assert len(loaded_opt_sd["param_groups"]) == len(
            original_opt_sd["param_groups"]
        )
        # Compare state entries (momentum buffers, step counts, etc.)
        assert set(loaded_opt_sd["state"].keys()) == set(
            original_opt_sd["state"].keys()
        )
        for param_id in original_opt_sd["state"]:
            orig_entry = original_opt_sd["state"][param_id]
            loaded_entry = loaded_opt_sd["state"][param_id]
            for field in orig_entry:
                orig_val = orig_entry[field]
                loaded_val = loaded_entry[field]
                if isinstance(orig_val, torch.Tensor):
                    assert torch.equal(orig_val, loaded_val), (
                        f"Optimizer state mismatch: param {param_id}, field {field}"
                    )
                else:
                    assert orig_val == loaded_val, (
                        f"Optimizer scalar mismatch: param {param_id}, field {field}"
                    )


# ---------------------------------------------------------------------------
# 3. Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    """Verify all metadata fields survive the JSON roundtrip."""

    def test_metadata_roundtrip(self, tmp_path):
        step = 42
        meta = _meta_data(step=step, val_bpb=1.2345)
        # Create a minimal state dict to satisfy save_checkpoint
        dummy_sd = {"weight": torch.zeros(2)}

        save_checkpoint(str(tmp_path), step, dummy_sd, None, meta, rank=0)
        _, _, loaded_meta = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )

        assert loaded_meta["step"] == step
        assert loaded_meta["val_bpb"] == pytest.approx(1.2345)
        assert loaded_meta["model_config"] == meta["model_config"]
        assert loaded_meta["user_config"] == meta["user_config"]

    def test_metadata_extra_fields_preserved(self, tmp_path):
        step = 0
        meta = _meta_data(step=step)
        meta["custom_field"] = "hello"
        meta["nested"] = {"a": 1, "b": [2, 3]}
        dummy_sd = {"weight": torch.zeros(2)}

        save_checkpoint(str(tmp_path), step, dummy_sd, None, meta, rank=0)
        _, _, loaded_meta = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )

        assert loaded_meta["custom_field"] == "hello"
        assert loaded_meta["nested"] == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# 4. EncoderDecoderGPT roundtrip
# ---------------------------------------------------------------------------


class TestEncoderDecoderRoundtrip:
    """Save an EncoderDecoderGPT, load it back, verify all tensors match."""

    def test_encdec_state_dict_roundtrip(self, tmp_path):
        model = _make_encdec()
        original_sd = model.state_dict()
        step = 0
        meta = _encdec_meta_data(step=step)

        save_checkpoint(str(tmp_path), step, original_sd, None, meta, rank=0)
        loaded_sd, _, loaded_meta = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )

        assert set(loaded_sd.keys()) == set(original_sd.keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], loaded_sd[key]), (
                f"Mismatch on key {key}"
            )

    def test_encoder_params_preserved(self, tmp_path):
        """Verify encoder-specific parameters survive the roundtrip."""
        model = _make_encdec()
        original_sd = model.state_dict()
        step = 0
        meta = _encdec_meta_data(step=step)

        save_checkpoint(str(tmp_path), step, original_sd, None, meta, rank=0)
        loaded_sd, _, _ = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )

        # Check key encoder params explicitly
        encoder_keys = [k for k in original_sd if k.startswith("encoder.")]
        assert len(encoder_keys) > 0, "No encoder keys found in state dict"
        for key in encoder_keys:
            assert key in loaded_sd, f"Encoder key {key} missing after load"
            assert torch.equal(original_sd[key], loaded_sd[key]), (
                f"Encoder key {key} mismatch"
            )

        # Specifically check the feature embedding
        assert torch.equal(
            original_sd["encoder.feature_embed.weight"],
            loaded_sd["encoder.feature_embed.weight"],
        )

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_build_encdec_model_roundtrip(self, mock_get_tok, tmp_path):
        """Full build_encdec_model reconstruction from checkpoint."""
        mock_get_tok.return_value = MockTokenizer()
        model = _make_encdec()
        original_sd = model.state_dict()
        step = 0
        meta = _encdec_meta_data(step=step)

        save_checkpoint(str(tmp_path), step, original_sd, None, meta, rank=0)
        rebuilt_model, tokenizer, loaded_meta = build_encdec_model(
            str(tmp_path),
            step,
            torch.device("cpu"),
            "eval",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_roundtrip"),
        )

        rebuilt_sd = rebuilt_model.state_dict()
        for key in original_sd:
            assert key in rebuilt_sd, f"Key {key} missing in rebuilt model"
            assert torch.equal(original_sd[key], rebuilt_sd[key]), (
                f"Mismatch on key {key} after build_encdec_model"
            )


# ---------------------------------------------------------------------------
# 5. Step resolution (find_last_step)
# ---------------------------------------------------------------------------


class TestStepResolution:
    """Verify find_last_step discovers the highest checkpoint step."""

    def test_find_last_step_returns_highest(self, tmp_path):
        dummy_sd = {"w": torch.zeros(2)}
        for step in [0, 5, 10]:
            save_checkpoint(str(tmp_path), step, dummy_sd, None, _meta_data(step=step))

        assert find_last_step(str(tmp_path)) == 10

    def test_load_checkpoint_step_none_loads_latest(self, tmp_path):
        dummy_sd = {"w": torch.zeros(2)}
        for step in [0, 5, 10]:
            meta = _meta_data(step=step)
            save_checkpoint(str(tmp_path), step, dummy_sd, None, meta)

        loaded_sd, _, loaded_meta = load_checkpoint(
            str(tmp_path), None, torch.device("cpu")
        )
        assert loaded_meta["step"] == 10

    def test_find_last_step_single_checkpoint(self, tmp_path):
        dummy_sd = {"w": torch.zeros(2)}
        save_checkpoint(str(tmp_path), 7, dummy_sd, None, _meta_data(step=7))

        assert find_last_step(str(tmp_path)) == 7

    def test_find_last_step_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_last_step(str(tmp_path))


# ---------------------------------------------------------------------------
# 6. bfloat16 -> float32 conversion
# ---------------------------------------------------------------------------


class TestBfloat16Conversion:
    """Verify build_model converts bfloat16 tensors to float32 on CPU."""

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_bf16_to_f32_on_cpu(self, mock_get_tok, tmp_path):
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        sd = model.state_dict()

        # Cast a subset of tensors to bfloat16 (simulating GPU-trained model)
        bf16_sd = {}
        for k, v in sd.items():
            if v.is_floating_point():
                bf16_sd[k] = v.bfloat16()
            else:
                bf16_sd[k] = v

        step = 0
        meta = _meta_data(step=step)
        save_checkpoint(str(tmp_path), step, bf16_sd, None, meta, rank=0)

        rebuilt_model, _, _ = build_model(
            str(tmp_path), step, torch.device("cpu"), "eval"
        )
        rebuilt_sd = rebuilt_model.state_dict()

        # All floating-point tensors should now be float32 on CPU
        for key, val in rebuilt_sd.items():
            if val.is_floating_point():
                assert val.dtype == torch.float32, (
                    f"Key {key} has dtype {val.dtype}, expected float32"
                )

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_bf16_conversion_preserves_values(self, mock_get_tok, tmp_path):
        """Values should survive the bf16 -> f32 roundtrip within tolerance."""
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        original_sd = model.state_dict()

        bf16_sd = {}
        for k, v in original_sd.items():
            if v.is_floating_point():
                bf16_sd[k] = v.bfloat16()
            else:
                bf16_sd[k] = v

        step = 0
        meta = _meta_data(step=step)
        save_checkpoint(str(tmp_path), step, bf16_sd, None, meta, rank=0)

        rebuilt_model, _, _ = build_model(
            str(tmp_path), step, torch.device("cpu"), "eval"
        )
        rebuilt_sd = rebuilt_model.state_dict()

        for key in original_sd:
            if original_sd[key].is_floating_point():
                # bfloat16 has ~3 decimal digits of precision, so tolerance is loose
                assert torch.allclose(
                    original_sd[key].float(),
                    rebuilt_sd[key].float(),
                    atol=1e-2,
                    rtol=1e-2,
                ), f"Value mismatch on {key} after bf16 roundtrip"


# ---------------------------------------------------------------------------
# 7. _orig_mod prefix stripping
# ---------------------------------------------------------------------------


class TestOrigModPrefixStripping:
    """Verify build_model strips _orig_mod. prefix from torch.compile keys."""

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_orig_mod_prefix_stripped(self, mock_get_tok, tmp_path):
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        sd = model.state_dict()

        # Simulate torch.compile prefix on every key
        compiled_sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

        step = 0
        meta = _meta_data(step=step)
        save_checkpoint(str(tmp_path), step, compiled_sd, None, meta, rank=0)

        rebuilt_model, _, _ = build_model(
            str(tmp_path), step, torch.device("cpu"), "eval"
        )
        rebuilt_sd = rebuilt_model.state_dict()

        # All keys in the rebuilt model should match the original (no prefix)
        assert set(rebuilt_sd.keys()) == set(sd.keys())
        for key in sd:
            assert torch.equal(sd[key], rebuilt_sd[key]), (
                f"Mismatch on {key} after _orig_mod stripping"
            )


# ---------------------------------------------------------------------------
# 8. Multi-step checkpoint directory
# ---------------------------------------------------------------------------


class TestMultiStepCheckpoints:
    """Verify saving/loading multiple checkpoint steps in one directory."""

    def test_find_last_step_picks_highest(self, tmp_path):
        dummy_sd = {"w": torch.zeros(2)}
        steps = [0, 3, 7, 15, 100]
        for step in steps:
            save_checkpoint(str(tmp_path), step, dummy_sd, None, _meta_data(step=step))

        assert find_last_step(str(tmp_path)) == 100

    def test_load_specific_step(self, tmp_path):
        """Each step's data should be independently loadable."""
        for step in [0, 5, 10]:
            # Save a distinct tensor at each step so we can verify the right one loads
            sd = {"w": torch.full((2,), float(step))}
            meta = _meta_data(step=step)
            save_checkpoint(str(tmp_path), step, sd, None, meta)

        # Load step=5 specifically
        loaded_sd, _, loaded_meta = load_checkpoint(
            str(tmp_path), 5, torch.device("cpu")
        )
        assert loaded_meta["step"] == 5
        assert torch.equal(loaded_sd["w"], torch.full((2,), 5.0))

        # Load step=0 specifically
        loaded_sd, _, loaded_meta = load_checkpoint(
            str(tmp_path), 0, torch.device("cpu")
        )
        assert loaded_meta["step"] == 0
        assert torch.equal(loaded_sd["w"], torch.full((2,), 0.0))

    def test_load_latest_ignores_earlier_steps(self, tmp_path):
        for step in [0, 5, 10]:
            sd = {"w": torch.full((2,), float(step))}
            meta = _meta_data(step=step)
            save_checkpoint(str(tmp_path), step, sd, None, meta)

        loaded_sd, _, loaded_meta = load_checkpoint(
            str(tmp_path), None, torch.device("cpu")
        )
        assert loaded_meta["step"] == 10
        assert torch.equal(loaded_sd["w"], torch.full((2,), 10.0))


# ---------------------------------------------------------------------------
# 9. build_model full roundtrip
# ---------------------------------------------------------------------------


class TestBuildModelRoundtrip:
    """Full end-to-end: save a GPT checkpoint, rebuild via build_model."""

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_build_model_matches_original(self, mock_get_tok, tmp_path):
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        original_sd = model.state_dict()
        step = 0
        meta = _meta_data(step=step)

        save_checkpoint(str(tmp_path), step, original_sd, None, meta, rank=0)
        rebuilt_model, tokenizer, loaded_meta = build_model(
            str(tmp_path), step, torch.device("cpu"), "eval"
        )

        rebuilt_sd = rebuilt_model.state_dict()
        # Every key in the original should appear in the rebuilt model
        for key in original_sd:
            assert key in rebuilt_sd, f"Key {key} missing in rebuilt model"
            assert torch.equal(original_sd[key], rebuilt_sd[key]), (
                f"Mismatch on key {key}"
            )

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_build_model_eval_mode(self, mock_get_tok, tmp_path):
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        step = 0
        meta = _meta_data(step=step)
        save_checkpoint(str(tmp_path), step, model.state_dict(), None, meta)

        rebuilt, _, _ = build_model(
            str(tmp_path), step, torch.device("cpu"), "eval"
        )
        assert not rebuilt.training

    @patch("nanochat.checkpoint_manager.get_tokenizer")
    def test_build_model_train_mode(self, mock_get_tok, tmp_path):
        mock_get_tok.return_value = MockTokenizer()
        model = _make_gpt()
        step = 0
        meta = _meta_data(step=step)
        save_checkpoint(str(tmp_path), step, model.state_dict(), None, meta)

        rebuilt, _, _ = build_model(
            str(tmp_path), step, torch.device("cpu"), "train"
        )
        assert rebuilt.training


# ---------------------------------------------------------------------------
# 10. Rank-specific optimizer sharding
# ---------------------------------------------------------------------------


class TestRankSpecificOptimizer:
    """Optimizer files are saved per-rank; verify rank isolation."""

    def test_rank0_and_rank1_saved_separately(self, tmp_path):
        dummy_sd = {"w": torch.zeros(2)}
        opt_r0 = {"state": {0: {"step": torch.tensor(1)}}, "param_groups": []}
        opt_r1 = {"state": {0: {"step": torch.tensor(2)}}, "param_groups": []}
        meta = _meta_data()
        step = 0

        save_checkpoint(str(tmp_path), step, dummy_sd, opt_r0, meta, rank=0)
        save_checkpoint(str(tmp_path), step, dummy_sd, opt_r1, meta, rank=1)

        _, loaded_r0, _ = load_checkpoint(
            str(tmp_path), step, torch.device("cpu"), load_optimizer=True, rank=0
        )
        _, loaded_r1, _ = load_checkpoint(
            str(tmp_path), step, torch.device("cpu"), load_optimizer=True, rank=1
        )

        assert torch.equal(
            loaded_r0["state"][0]["step"], torch.tensor(1)
        )
        assert torch.equal(
            loaded_r1["state"][0]["step"], torch.tensor(2)
        )
