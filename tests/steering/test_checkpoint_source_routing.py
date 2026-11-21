"""Tests for conditioned SFT segment routing, _is_encdec_root,
decoder-to-encoder-decoder conversion, and build_encdec_model config recovery."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanochat.checkpoint_manager import (
    _find_checkpoint_leaf,
    _is_encdec_root,
    build_encdec_model,
    get_checkpoint_dir,
    save_checkpoint,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    MockTokenizer,
    _make_encdec,
    _make_gpt,
    _make_gpt_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_checkpoint(checkpoint_dir: str, step: int = 0) -> None:
    """Create minimal GPT checkpoint files (model, meta) at the given dir."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    config = _make_gpt_config()
    model = _make_gpt()
    meta = {
        "step": step,
        "model_config": {
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "window_pattern": config.window_pattern,
        },
    }
    save_checkpoint(checkpoint_dir, step, model.state_dict(), None, meta, rank=0)


def _create_fake_encdec_checkpoint(
    checkpoint_dir: str,
    step: int = 0,
    include_encoder_config: bool = True,
    include_num_buckets: bool = True,
    negative_selection: dict | None = None,
) -> None:
    """Create minimal EncoderDecoderGPT checkpoint files."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = _make_encdec()
    config = _make_gpt_config()
    user_config: dict = {}
    if include_encoder_config:
        user_config["encoder"] = {
            "n_embd": TINY_N_EMBD,
            "depth": TINY_N_LAYER,
            "n_head": TINY_N_HEAD,
            "n_kv_head": TINY_N_KV_HEAD,
            "head_dim": TINY_HEAD_DIM,
        }
        user_config["cross_attention"] = {"frequency": 1}
    if include_num_buckets:
        user_config["num_buckets"] = TINY_NUM_BUCKETS
    if negative_selection is not None:
        user_config["negative_selection"] = negative_selection
    meta = {
        "step": step,
        "model_config": {
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "window_pattern": config.window_pattern,
        },
        "user_config": user_config,
    }
    save_checkpoint(checkpoint_dir, step, model.state_dict(), None, meta, rank=0)


@pytest.fixture(autouse=True)
def _mock_tokenizer():
    """Mock get_tokenizer to avoid needing real tokenizer files."""
    mock_tok = MockTokenizer()
    with patch("nanochat.checkpoint_manager.get_tokenizer", return_value=mock_tok):
        yield


# ===========================================================================
# 1. Conditioned SFT segment routing
# ===========================================================================


class TestConditionedRouting:
    """Test checkpoint routing entries for conditioned SFT segments."""

    def test_safety_cond_first_segment(self, tmp_path: Path) -> None:
        """safety_cond -> base_checkpoints/tag/safety_sft_conditioned_checkpoints"""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("safety_cond", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "safety_sft_conditioned_checkpoints",
        )
        assert result == expected

    def test_capabilities_cond_first_segment(self, tmp_path: Path) -> None:
        """capabilities_cond -> base_checkpoints/tag/capabilities_sft_conditioned_checkpoints"""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("capabilities_cond", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "capabilities_sft_conditioned_checkpoints",
        )
        assert result == expected

    def test_encdec_safety_cond_chain(self, tmp_path: Path) -> None:
        """encdec/safety_cond -> conditional_encdec_checkpoints/tag/safety_sft_conditioned_checkpoints"""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "conditional_encdec_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("encdec/safety_cond", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            "d26",
            "safety_sft_conditioned_checkpoints",
        )
        assert result == expected

    def test_safety_cond_rl_chain(self, tmp_path: Path) -> None:
        """safety_cond/rl -> base_checkpoints/tag/safety_sft_conditioned_checkpoints/standard_rl_checkpoints"""
        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("safety_cond/rl", model_tag="d26")
        expected = os.path.join(
            base_dir,
            "base_checkpoints",
            "d26",
            "safety_sft_conditioned_checkpoints",
            "standard_rl_checkpoints",
        )
        assert result == expected


# ===========================================================================
# 2. _is_encdec_root tests
# ===========================================================================


class TestIsEncdecRoot:
    """Test _is_encdec_root detects encdec source chains correctly."""

    def test_encdec_alone(self) -> None:
        assert _is_encdec_root("encdec") is True

    def test_encdec_with_chain(self) -> None:
        assert _is_encdec_root("encdec/safety") is True

    def test_base_is_not_encdec(self) -> None:
        assert _is_encdec_root("base") is False

    def test_sft_is_not_encdec(self) -> None:
        assert _is_encdec_root("sft") is False

    def test_safety_rl_is_not_encdec(self) -> None:
        assert _is_encdec_root("safety/rl") is False


# ===========================================================================
# 3. Decoder-to-EncoderDecoder conversion verification
# ===========================================================================


class TestDecoderToEncoderDecoderConversion:
    """Simulate and verify the logic from convert_decoder_to_encdec.py."""

    def test_strict_false_missing_keys_are_encoder_and_xattn(self) -> None:
        """Loading a GPT state dict into EncoderDecoderGPT with strict=False
        should have missing keys ONLY for encoder + cross-attention parameters."""
        gpt = _make_gpt()
        decoder_state = gpt.state_dict()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            encdec = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
            )
        encdec.to_empty(device="cpu")
        encdec.init_weights()

        missing, unexpected = encdec.load_state_dict(
            decoder_state, strict=False, assign=True
        )

        # All missing keys must be encoder.* or cross_attn_layers.*
        for key in missing:
            assert key.startswith(
                ("encoder.", "cross_attn_layers.")
            ), f"Missing key {key!r} is not encoder or cross-attention"

    def test_strict_false_no_unexpected_keys(self) -> None:
        """No unexpected keys when loading a GPT state dict into EncoderDecoderGPT."""
        gpt = _make_gpt()
        decoder_state = gpt.state_dict()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            encdec = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
            )
        encdec.to_empty(device="cpu")
        encdec.init_weights()

        _, unexpected = encdec.load_state_dict(
            decoder_state, strict=False, assign=True
        )

        assert unexpected == [], f"Unexpected keys: {unexpected}"

    def test_decoder_weights_match_after_load(self) -> None:
        """All decoder weights should match the original GPT state dict after loading."""
        gpt = _make_gpt()
        decoder_state = gpt.state_dict()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            encdec = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
            )
        encdec.to_empty(device="cpu")
        encdec.init_weights()

        encdec.load_state_dict(decoder_state, strict=False, assign=True)

        encdec_state = encdec.state_dict()
        for key, value in decoder_state.items():
            assert key in encdec_state, (
                f"Decoder key {key!r} missing from EncoderDecoderGPT state dict"
            )
            assert t.equal(encdec_state[key], value), (
                f"Decoder weight {key!r} does not match after loading"
            )

    def test_encoder_weights_are_fresh_not_from_decoder(self) -> None:
        """Encoder weights should be freshly initialized, not copied from the decoder."""
        gpt = _make_gpt()
        decoder_state = gpt.state_dict()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)

        # Build and init first to capture fresh init weights
        with t.device("meta"):
            encdec = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
            )
        encdec.to_empty(device="cpu")
        encdec.init_weights()

        # Snapshot encoder weights before loading decoder state
        pre_load_encoder = {
            k: v.clone()
            for k, v in encdec.state_dict().items()
            if k.startswith("encoder.")
        }

        # Load decoder weights (should NOT affect encoder)
        encdec.load_state_dict(decoder_state, strict=False, assign=True)

        post_load_encoder = {
            k: v for k, v in encdec.state_dict().items() if k.startswith("encoder.")
        }

        # Encoder weights should be unchanged (still the fresh init values)
        for key, pre_load_value in pre_load_encoder.items():
            assert key in post_load_encoder, (
                f"Encoder key {key!r} missing after load"
            )
            assert t.equal(pre_load_value, post_load_encoder[key]), (
                f"Encoder weight {key!r} was modified by decoder load"
            )

    def test_missing_keys_include_encoder_feature_embed(self) -> None:
        """The missing keys should include encoder.feature_embed.weight (critical for inference)."""
        gpt = _make_gpt()
        decoder_state = gpt.state_dict()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with t.device("meta"):
            encdec = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
            )
        encdec.to_empty(device="cpu")
        encdec.init_weights()

        missing, _ = encdec.load_state_dict(
            decoder_state, strict=False, assign=True
        )

        assert "encoder.feature_embed.weight" in missing


# ===========================================================================
# 4. EncoderDecoderGPT config recovery for SFT checkpoints
# ===========================================================================


class TestBuildEncdecModelConfigRecovery:
    """Test that build_encdec_model recovers encoder/cross_attention config from parent."""

    def test_recovery_from_parent_checkpoint(self, tmp_path: Path) -> None:
        """SFT child without encoder config recovers from parent."""
        # Create parent with full config
        parent_dir = str(tmp_path / "parent")
        _create_fake_encdec_checkpoint(parent_dir, step=0)

        # Create child WITHOUT encoder/cross_attention in user_config
        child_dir = str(tmp_path / "parent" / "child_sft")
        _create_fake_encdec_checkpoint(
            child_dir, step=100, include_encoder_config=False
        )

        model, _, meta = build_encdec_model(
            child_dir,
            step=100,
            device=t.device("cpu"),
            phase="eval",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_routing"),
        )

        assert isinstance(model, EncoderDecoderGPT)
        # Verify the recovered config is present in user_config
        assert "encoder" in meta["user_config"]
        assert "cross_attention" in meta["user_config"]
        assert meta["user_config"]["encoder"]["n_embd"] == TINY_N_EMBD
        assert meta["user_config"]["cross_attention"]["frequency"] == 1

    def test_num_buckets_recovered_from_parent(self, tmp_path: Path) -> None:
        """num_buckets is recovered from parent when missing in child."""
        # Create parent with num_buckets
        parent_dir = str(tmp_path / "parent")
        _create_fake_encdec_checkpoint(parent_dir, step=0)

        # Create child without num_buckets
        child_dir = str(tmp_path / "parent" / "child_sft")
        _create_fake_encdec_checkpoint(
            child_dir,
            step=100,
            include_encoder_config=False,
            include_num_buckets=False,
        )

        model, _, meta = build_encdec_model(
            child_dir,
            step=100,
            device=t.device("cpu"),
            phase="eval",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_routing"),
        )

        assert isinstance(model, EncoderDecoderGPT)
        assert meta["user_config"]["num_buckets"] == TINY_NUM_BUCKETS

    def test_negative_selection_recovered_from_parent(self, tmp_path: Path) -> None:
        """negative_selection is recovered from an encdec ancestor when child omits it."""
        parent_dir = str(tmp_path / "parent")
        _create_fake_encdec_checkpoint(
            parent_dir,
            step=0,
            negative_selection={"distribution": "geometric", "mean": 2.0},
        )

        child_dir = str(tmp_path / "parent" / "child_sft")
        _create_fake_encdec_checkpoint(
            child_dir,
            step=100,
            include_encoder_config=False,
        )

        model, _, meta = build_encdec_model(
            child_dir,
            step=100,
            device=t.device("cpu"),
            phase="eval",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_routing"),
        )

        assert isinstance(model, EncoderDecoderGPT)
        assert meta["user_config"]["negative_selection"] == {
            "distribution": "geometric",
            "mean": 2.0,
        }

    def test_child_with_full_config_does_not_need_parent(self, tmp_path: Path) -> None:
        """Child checkpoint with full encoder config does not consult parent."""
        child_dir = str(tmp_path / "standalone")
        _create_fake_encdec_checkpoint(child_dir, step=0)

        model, _, meta = build_encdec_model(
            child_dir,
            step=0,
            device=t.device("cpu"),
            phase="eval",
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_routing"),
        )

        assert isinstance(model, EncoderDecoderGPT)
        assert meta["user_config"]["encoder"]["n_embd"] == TINY_N_EMBD

    def test_no_parent_raises_when_config_missing(self, tmp_path: Path) -> None:
        """Missing encoder config with no parent checkpoint raises FileNotFoundError.

        _recover_parent_user_config traverses up one directory.  When the parent
        directory has no checkpoint files, find_last_step raises FileNotFoundError.
        When the parent directory does not even exist (nested deeper),
        _recover_parent_user_config itself raises with "Cannot recover".
        Either way a FileNotFoundError surfaces.
        """
        # Nest deep enough that the parent dir has no checkpoint files at all
        orphan_dir = str(tmp_path / "a" / "b" / "orphan")
        _create_fake_encdec_checkpoint(
            orphan_dir, step=0, include_encoder_config=False
        )

        with pytest.raises(FileNotFoundError):
            build_encdec_model(
                orphan_dir,
                step=0,
                device=t.device("cpu"),
                phase="eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_routing"),
            )


# ===========================================================================
# 5. _find_checkpoint_leaf edge cases
# ===========================================================================


class TestFindCheckpointLeafEdgeCases:
    """Test _find_checkpoint_leaf with deeper nesting and multiple candidates."""

    def test_returns_first_alphabetically(self, tmp_path: Path) -> None:
        """Multiple nested paths: returns first (alphabetically sorted)."""
        leaf_a = tmp_path / "a_config"
        leaf_b = tmp_path / "b_config"
        _create_fake_checkpoint(str(leaf_a), step=0)
        _create_fake_checkpoint(str(leaf_b), step=0)

        result = _find_checkpoint_leaf(str(tmp_path))
        assert result == str(leaf_a)

    def test_deeply_nested_three_levels(self, tmp_path: Path) -> None:
        """3+ levels of nesting: s128/t2048.9/run1/"""
        leaf = tmp_path / "s128" / "t2048.9" / "run1"
        _create_fake_checkpoint(str(leaf), step=0)

        result = _find_checkpoint_leaf(str(tmp_path))
        assert result == str(leaf)

    def test_mixed_some_subdirs_have_checkpoints(self, tmp_path: Path) -> None:
        """Some subdirs have checkpoints, some are empty dirs."""
        (tmp_path / "empty1").mkdir()
        (tmp_path / "empty2" / "also_empty").mkdir(parents=True)
        leaf = tmp_path / "real" / "nested"
        _create_fake_checkpoint(str(leaf), step=0)

        result = _find_checkpoint_leaf(str(tmp_path))
        assert result == str(leaf)

    def test_checkpoint_at_root_preferred_over_nested(self, tmp_path: Path) -> None:
        """If root_dir itself has checkpoint files, return root even if subdirs also have them."""
        _create_fake_checkpoint(str(tmp_path), step=0)
        nested = tmp_path / "sub"
        _create_fake_checkpoint(str(nested), step=1)

        result = _find_checkpoint_leaf(str(tmp_path))
        assert result == str(tmp_path)


# ===========================================================================
# 6. Encdec auto-discovery via _find_checkpoint_leaf in get_checkpoint_dir
# ===========================================================================


class TestEncdecAutoDiscovery:
    """Test that get_checkpoint_dir auto-discovers nested encdec structures
    when segment_size and feature_selection are both None."""

    def test_auto_discovery_finds_nested_checkpoint(self, tmp_path: Path) -> None:
        """Nested encdec structure is auto-discovered when no segment_size/feature_selection."""
        base_dir = str(tmp_path)
        tag_dir = tmp_path / "conditional_encdec_checkpoints" / "d26"
        leaf = tag_dir / "s128" / "t2048.9"
        _create_fake_checkpoint(str(leaf), step=0)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("encdec", model_tag="d26")

        assert result == str(leaf)

    def test_no_auto_discovery_when_segment_size_given(self, tmp_path: Path) -> None:
        """Auto-discovery is skipped when segment_size is explicitly provided."""
        base_dir = str(tmp_path)
        tag_dir = tmp_path / "conditional_encdec_checkpoints" / "d26"
        tag_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("encdec", model_tag="d26", segment_size=128)

        # Should just append s128 without traversing further
        assert result == os.path.join(
            base_dir, "conditional_encdec_checkpoints", "d26", "s128"
        )

    def test_auto_discovery_with_multiple_nested_paths(self, tmp_path: Path) -> None:
        """When multiple nested paths exist, auto-discovery picks the first alphabetically."""
        base_dir = str(tmp_path)
        tag_dir = tmp_path / "conditional_encdec_checkpoints" / "d26"
        leaf_a = tag_dir / "s064" / "t100.0"
        leaf_b = tag_dir / "s128" / "t2048.9"
        _create_fake_checkpoint(str(leaf_a), step=0)
        _create_fake_checkpoint(str(leaf_b), step=0)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            result = get_checkpoint_dir("encdec", model_tag="d26")

        # Depth-first alphabetical: s064/t100.0 comes before s128/t2048.9
        assert result == str(leaf_a)
