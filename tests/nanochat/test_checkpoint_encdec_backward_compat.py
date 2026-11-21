"""Regression tests for encdec-specific checkpoint backward-compat (P0s).

Covers P0s in ``build_encdec_model`` that were silent in the original
backward-compat coverage (which only exercised ``build_model``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nanochat.checkpoint_manager import (
    build_encdec_model,
    save_checkpoint,
)
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
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
    TINY_SEQ_LEN,
    TINY_VOCAB,
    MockTokenizer,
    _make_encdec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encdec_user_config() -> dict:
    """Return a complete encdec user_config matching _make_encdec() fixture."""
    return {
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
    }


def _encdec_model_config() -> dict:
    """Return a model_config dict for tiny EncoderDecoderGPT."""
    return {
        "sequence_len": TINY_SEQ_LEN,
        "vocab_size": TINY_VOCAB,
        "n_layer": TINY_N_LAYER,
        "n_head": TINY_N_HEAD,
        "n_kv_head": TINY_N_KV_HEAD,
        "n_embd": TINY_N_EMBD,
        "window_pattern": "L",
    }


def _feature_prediction_config_dict(
    *,
    weight: float = 0.25,
    layer: int = -1,
    loss_mode: str = "full_segment",
    loss_fn: str = "bce",
) -> dict:
    """Return JSON-serialisable feature-prediction config metadata."""
    return {
        "block": "mlp",
        "head_layers": 0,
        "weight": weight,
        "layer": layer,
        "loss_mode": loss_mode,
        "loss_fn": loss_fn,
    }


def _feature_prediction_config(
    *,
    weight: float = 0.25,
    layer: int = -1,
    loss_mode: FeaturePredLossMode = FeaturePredLossMode.FULL_SEGMENT,
    loss_fn: FeaturePredLossFn = FeaturePredLossFn.BCE,
) -> FeaturePredictionConfig:
    """Return a FeaturePredictionConfig matching _feature_prediction_config_dict."""
    return FeaturePredictionConfig(
        weight=weight,
        layer=layer,
        loss_mode=loss_mode,
        loss_fn=loss_fn,
    )


def _make_encdec_with_feature_prediction(
    fp_config: FeaturePredictionConfig | None = None,
) -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT whose state dict carries the aux head."""
    dec_config = GPTConfig(**_encdec_model_config())
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=1)
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
        feature_pred_config=fp_config or _feature_prediction_config(),
    )
    model.init_weights()
    model.eval()
    return model


def _split_qkv_state(
    state: dict[str, torch.Tensor], config: GPTConfig
) -> dict[str, torch.Tensor]:
    """Return a state dict with every ``.attn.c_qkv.weight`` split into
    separate c_q/c_k/c_v keys (simulating an old-format checkpoint).

    Only attention layers whose key ends in ``.attn.c_qkv.weight`` are
    split — cross-attention's standalone ``c_q.weight`` (which has no
    c_k/c_v sibling) is intentionally ignored by the suffix filter.
    """
    head_dim = config.n_embd // config.n_head
    q_dim = config.n_head * head_dim
    kv_dim = config.n_kv_head * head_dim

    split: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.endswith(".attn.c_qkv.weight"):
            prefix = key.removesuffix("c_qkv.weight")
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            split[f"{prefix}c_q.weight"] = q
            split[f"{prefix}c_k.weight"] = k
            split[f"{prefix}c_v.weight"] = v
        else:
            split[key] = val
    return split


# ===================================================================
# QKV fuse for encdec (P0 #2)
# ===================================================================


class TestBuildEncdecModelQKVFuse:
    """build_encdec_model must fuse split c_q/c_k/c_v like build_model does."""

    def test_encdec_with_split_qkv_loads_successfully(self, tmp_path):
        """Old-format encdec checkpoint with split c_q/c_k/c_v on BOTH the
        decoder (transformer.h.N.attn.*) and the encoder
        (encoder.blocks.N.attn.*) must load via build_encdec_model."""
        model = _make_encdec()
        current_state = model.state_dict()

        # Sanity check: the fresh model has BOTH decoder and encoder c_qkv.
        assert any(
            k.startswith("transformer.h.") and k.endswith(".attn.c_qkv.weight")
            for k in current_state
        ), "fixture expected to have decoder c_qkv"
        assert any(
            k.startswith("encoder.blocks.") and k.endswith(".attn.c_qkv.weight")
            for k in current_state
        ), "fixture expected to have encoder c_qkv"

        dec_config = GPTConfig(**_encdec_model_config())
        old_state = _split_qkv_state(current_state, dec_config)

        # Sanity check: split keys exist for BOTH decoder and encoder,
        # and no fused keys remain.
        assert any(
            k.startswith("transformer.h.") and k.endswith(".attn.c_q.weight")
            for k in old_state
        )
        assert any(
            k.startswith("encoder.blocks.") and k.endswith(".attn.c_q.weight")
            for k in old_state
        )
        assert not any(k.endswith(".attn.c_qkv.weight") for k in old_state)

        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        checkpoint_dir = str(tmp_path / "encdec_old_qkv")
        save_checkpoint(checkpoint_dir, 0, old_state, None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                checkpoint_dir,
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert isinstance(loaded_model, EncoderDecoderGPT)
        # The rebuilt model must use fused c_qkv.
        rebuilt = loaded_model.state_dict()
        assert any(k.endswith(".attn.c_qkv.weight") for k in rebuilt)

    def test_encdec_fused_qkv_weights_are_correct(self, tmp_path):
        """After fusion, the rebuilt c_qkv weight must equal cat([q, k, v], dim=0)."""
        model = _make_encdec()
        current_state = model.state_dict()
        dec_config = GPTConfig(**_encdec_model_config())
        old_state = _split_qkv_state(current_state, dec_config)

        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        checkpoint_dir = str(tmp_path / "encdec_fusion_check")
        save_checkpoint(checkpoint_dir, 0, old_state, None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                checkpoint_dir,
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        rebuilt = loaded_model.state_dict()
        # The fused c_qkv must round-trip bit-exact.
        for key, original_val in current_state.items():
            if key.endswith(".attn.c_qkv.weight"):
                assert key in rebuilt, f"Missing fused key {key} after load"
                assert torch.equal(rebuilt[key], original_val), (
                    f"Fused c_qkv weight for {key} does not match original"
                )


# ===================================================================
# num_buckets recovery for encdec (P0 #3)
# ===================================================================


class TestBuildEncdecRecoversNumBuckets:
    """num_buckets recovery must run unconditionally when the key is missing.

    The pre-fix code gated the entire parent-config recovery block (including
    num_buckets) by ``if "encoder" not in user_cfg or "cross_attention" not
    in user_cfg``. A child checkpoint that had both encoder and cross_attention
    but lacked num_buckets would skip the recovery block and hit
    ``num_buckets = user_cfg["num_buckets"]`` with a raw ``KeyError``.
    """

    def test_num_buckets_missing_with_enc_and_xattn_present_recovers(self, tmp_path):
        """Child has encoder+cross_attention but lacks num_buckets; parent has it.
        Pre-fix: KeyError, because recovery was gated by the enc/xattn check.
        Post-fix: num_buckets recovered from parent, load succeeds.
        """
        # Parent checkpoint has full user_config including num_buckets.
        parent_dir = tmp_path / "parent_tag"
        model = _make_encdec()
        parent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        save_checkpoint(
            str(parent_dir), 0, model.state_dict(), None, parent_meta
        )

        # Child lives under parent/standard_sft_checkpoints/
        child_dir = parent_dir / "standard_sft_checkpoints"
        child_user_cfg = _encdec_user_config()
        del child_user_cfg["num_buckets"]
        # Trigger precondition: both encoder AND cross_attention present.
        assert "encoder" in child_user_cfg
        assert "cross_attention" in child_user_cfg
        assert "num_buckets" not in child_user_cfg

        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": child_user_cfg,
        }
        save_checkpoint(str(child_dir), 0, model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            # Pre-fix raised KeyError('num_buckets') at ``num_buckets =
            # user_cfg[...]``. Post-fix: num_buckets is recovered from the
            # parent (without touching the now-unused enc/xattn values).
            loaded_model, _, _ = build_encdec_model(
                str(child_dir),
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert isinstance(loaded_model, EncoderDecoderGPT)

    def test_num_buckets_missing_without_parent_raises_value_error(self, tmp_path):
        """When num_buckets cannot be recovered from any source, raise a clean
        ValueError with an actionable message — not a bare KeyError."""
        # No parent; child lacks num_buckets entirely. Place the child
        # under an orphan parent directory that has no sibling checkpoint.
        child_dir = tmp_path / "orphan" / "standard_sft_checkpoints"
        model = _make_encdec()
        child_user_cfg = _encdec_user_config()
        del child_user_cfg["num_buckets"]
        assert "encoder" in child_user_cfg
        assert "cross_attention" in child_user_cfg

        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": child_user_cfg,
        }
        save_checkpoint(str(child_dir), 0, model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            # Must raise ValueError (NOT KeyError) with a message that names
            # num_buckets so the user can act on it.
            with pytest.raises(ValueError, match="num_buckets"):
                build_encdec_model(
                    str(child_dir),
                    0,
                    torch.device("cpu"),
                    "eval",
                    eval_conditioning=EvalConditioning.deferred(
                        "test_checkpoint_encdec_backward_compat"
                    ),
                )


# ===================================================================
# Feature-prediction metadata recovery for encdec SFT chains
# ===================================================================


class TestBuildEncdecRecoversFeaturePrediction:
    """Feature-prediction head reconstruction must follow checkpoint weights.

    Conditional SFT checkpoints can inherit a feature-prediction head from the
    encoder-decoder pretrain state dict while omitting the corresponding config
    from their local meta. The loader must rebuild the head only when the head
    weights are actually present, and it must recover metadata from both old and
    current meta layouts.
    """

    def test_direct_user_config_feature_prediction_builds_head(self, tmp_path):
        """A direct checkpoint with user_config.feature_prediction loads strictly."""
        fp_meta = _feature_prediction_config_dict(weight=0.3, layer=-1)
        model = _make_encdec_with_feature_prediction()
        user_cfg = _encdec_user_config()
        user_cfg["feature_prediction"] = fp_meta
        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": user_cfg,
        }
        checkpoint_dir = str(tmp_path / "encdec_fp_user_config")
        save_checkpoint(checkpoint_dir, 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                checkpoint_dir,
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is not None
        assert loaded_model.feature_pred_config is not None
        assert loaded_model.feature_pred_config.weight == fp_meta["weight"]
        assert loaded_model.feature_pred_config.layer == fp_meta["layer"]

    def test_direct_top_level_feature_prediction_config_builds_head(self, tmp_path):
        """A direct checkpoint with top-level feature_prediction_config loads.

        This pins the metadata shape written by conditional encoder-decoder
        training, where SFT-style metas can carry auxiliary-head config outside
        user_config.
        """
        fp_meta = _feature_prediction_config_dict(
            weight=0.4,
            layer=0,
            loss_mode="last_token",
            loss_fn="mse",
        )
        model = _make_encdec_with_feature_prediction()
        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
            "feature_prediction_config": fp_meta,
        }
        checkpoint_dir = str(tmp_path / "encdec_fp_top_level")
        save_checkpoint(checkpoint_dir, 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                checkpoint_dir,
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is not None
        assert loaded_model.feature_pred_config is not None
        assert loaded_model.feature_pred_config.weight == fp_meta["weight"]
        assert loaded_model.feature_pred_config.layer == fp_meta["layer"]
        assert loaded_model.feature_pred_config.loss_mode == fp_meta["loss_mode"]
        assert loaded_model.feature_pred_config.loss_fn == fp_meta["loss_fn"]

    def test_child_sft_recovers_feature_prediction_from_parent_user_config(
        self, tmp_path
    ):
        """A child SFT meta can recover aux-head config from a parent user_config."""
        fp_meta = _feature_prediction_config_dict(weight=0.5, layer=-1)
        model = _make_encdec_with_feature_prediction()
        parent_user_cfg = _encdec_user_config()
        parent_user_cfg["feature_prediction"] = fp_meta
        parent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": parent_user_cfg,
        }
        parent_dir = tmp_path / "parent_with_fp_user_config"
        save_checkpoint(str(parent_dir), 0, model.state_dict(), None, parent_meta)

        child_dir = parent_dir / "standard_sft_checkpoints"
        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        save_checkpoint(str(child_dir), 0, model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                str(child_dir),
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is not None
        assert loaded_model.feature_pred_config is not None
        assert loaded_model.feature_pred_config.weight == fp_meta["weight"]

    def test_child_sft_recovers_feature_prediction_from_parent_top_level_config(
        self, tmp_path
    ):
        """A child SFT meta can recover aux-head config from parent top-level meta."""
        fp_meta = _feature_prediction_config_dict(weight=0.6, layer=0)
        model = _make_encdec_with_feature_prediction()
        parent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
            "feature_prediction_config": fp_meta,
        }
        parent_dir = tmp_path / "parent_with_fp_top_level"
        save_checkpoint(str(parent_dir), 0, model.state_dict(), None, parent_meta)

        child_dir = parent_dir / "standard_sft_checkpoints"
        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        save_checkpoint(str(child_dir), 0, model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                str(child_dir),
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is not None
        assert loaded_model.feature_pred_config is not None
        assert loaded_model.feature_pred_config.weight == fp_meta["weight"]

    def test_closest_ancestor_feature_prediction_wins(self, tmp_path):
        """When several ancestors have aux-head metadata, use the closest one."""
        grandparent_fp = _feature_prediction_config_dict(weight=0.1, layer=-1)
        parent_fp = _feature_prediction_config_dict(weight=0.7, layer=0)
        model = _make_encdec_with_feature_prediction()

        grandparent_user_cfg = _encdec_user_config()
        grandparent_user_cfg["feature_prediction"] = grandparent_fp
        grandparent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": grandparent_user_cfg,
        }
        grandparent_dir = tmp_path / "pretrain"
        save_checkpoint(
            str(grandparent_dir), 0, model.state_dict(), None, grandparent_meta
        )

        parent_dir = grandparent_dir / "standard_sft_checkpoints"
        parent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
            "feature_prediction_config": parent_fp,
        }
        save_checkpoint(str(parent_dir), 0, model.state_dict(), None, parent_meta)

        child_dir = parent_dir / "capabilities_sft_checkpoints"
        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        save_checkpoint(str(child_dir), 0, model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                str(child_dir),
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is not None
        assert loaded_model.feature_pred_config is not None
        assert loaded_model.feature_pred_config.weight == parent_fp["weight"]
        assert loaded_model.feature_pred_config.layer == parent_fp["layer"]

    @pytest.mark.parametrize("structural_missing", [False, True])
    @pytest.mark.parametrize("legacy_mirror", [False, True])
    @pytest.mark.parametrize("has_head", [False, True])
    def test_ancestor_mirrors_are_validated_only_when_recovering_a_head(
        self, tmp_path, structural_missing, legacy_mirror, has_head
    ):
        parent_model = _make_encdec_with_feature_prediction()
        fp_meta = _feature_prediction_config_dict()
        other_mirror = dict(fp_meta)
        if legacy_mirror:
            other_mirror.pop("block")
            other_mirror["mlp_layers"] = other_mirror.pop("head_layers")
        else:
            other_mirror["block"] = "transformer"
        parent_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": {**_encdec_user_config(), "feature_prediction": fp_meta},
            "feature_prediction_config": other_mirror,
        }
        parent_dir = tmp_path / "parent"
        save_checkpoint(str(parent_dir), 0, parent_model.state_dict(), None, parent_meta)
        child_model = parent_model if has_head else _make_encdec()
        child_dir = parent_dir / "sft_checkpoints"
        child_meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": {} if structural_missing else _encdec_user_config(),
        }
        save_checkpoint(str(child_dir), 0, child_model.state_dict(), None, child_meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            if has_head:
                with pytest.raises(ValueError, match="removed|conflicting.*mirrors"):
                    build_encdec_model(
                        str(child_dir), 0, torch.device("cpu"), "eval",
                        eval_conditioning=EvalConditioning.deferred(
                            "test_checkpoint_encdec_backward_compat"
                        ),
                    )
            else:
                loaded, _, _ = build_encdec_model(
                    str(child_dir), 0, torch.device("cpu"), "eval",
                    eval_conditioning=EvalConditioning.deferred(
                        "test_checkpoint_encdec_backward_compat"
                    ),
                )
                assert loaded.feature_pred_head is None
                for name, value in child_model.state_dict().items():
                    assert torch.equal(loaded.state_dict()[name], value), name

    def test_head_without_recoverable_feature_prediction_raises_value_error(
        self, tmp_path
    ):
        """Head weights without local or ancestor config fail before strict load."""
        model = _make_encdec_with_feature_prediction()
        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
        }
        checkpoint_dir = str(tmp_path / "orphan_fp_head_without_config")
        save_checkpoint(checkpoint_dir, 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            with pytest.raises(
                ValueError,
                match="feature_pred_head\\.weight.*feature_prediction",
            ):
                build_encdec_model(
                    checkpoint_dir,
                    0,
                    torch.device("cpu"),
                    "eval",
                    eval_conditioning=EvalConditioning.deferred(
                        "test_checkpoint_encdec_backward_compat"
                    ),
                )

    def test_feature_prediction_config_without_head_is_ignored(self, tmp_path):
        """Metadata alone must not create a head missing from the state dict."""
        model = _make_encdec()
        assert "feature_pred_head.weight" not in model.state_dict()
        meta = {
            "step": 0,
            "model_config": _encdec_model_config(),
            "user_config": _encdec_user_config(),
            "feature_prediction_config": _feature_prediction_config_dict(),
        }
        checkpoint_dir = str(tmp_path / "fp_config_without_head")
        save_checkpoint(checkpoint_dir, 0, model.state_dict(), None, meta)

        with patch(
            "nanochat.checkpoint_manager.get_tokenizer",
            return_value=MockTokenizer(),
        ):
            loaded_model, _, _ = build_encdec_model(
                checkpoint_dir,
                0,
                torch.device("cpu"),
                "eval",
                eval_conditioning=EvalConditioning.deferred("test_checkpoint_encdec_backward_compat"),
            )

        assert loaded_model.feature_pred_head is None
        assert loaded_model.feature_pred_config is None
