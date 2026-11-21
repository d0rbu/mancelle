import pytest
import torch as t

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult

TINY_VOCAB = 128
TINY_N_EMBD = 64
TINY_N_LAYER = 4  # need >= 2 layers for layer indexing tests
TINY_N_HEAD = 2
TINY_N_KV_HEAD = 2
TINY_HEAD_DIM = 32
TINY_N_SAE = 32
TINY_NUM_BUCKETS = 4
B, T, N_SEG, F_MAX = 2, 16, 4, 5

_HAS_CUDA = t.cuda.is_available()
_DEVICE = "cuda" if _HAS_CUDA else "cpu"
_REQUIRES_CUDA = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")


def _make_model(fp_config=None, device=_DEVICE):
    dec_config = GPTConfig(
        sequence_len=T,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=2,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=2)
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=TINY_N_SAE,
        num_buckets=TINY_NUM_BUCKETS,
        feature_pred_config=fp_config,
    )
    model = model.to(device)
    model.init_weights()
    return model


def _make_inputs(device=_DEVICE):
    idx = t.randint(0, TINY_VOCAB, (B, T), device=device)
    targets = t.randint(0, TINY_VOCAB, (B, T), device=device)
    enc_fids = t.randint(0, TINY_N_SAE, (B, N_SEG, F_MAX), device=device)
    enc_mags = t.randint(0, TINY_NUM_BUCKETS, (B, N_SEG, F_MAX), device=device)
    enc_mask = t.ones(B, N_SEG, F_MAX, dtype=t.bool, device=device)
    seg_map = t.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        ],
        device=device,
    )
    # All segments from same doc (simple case)
    seg_doc_id = t.zeros(B, N_SEG, dtype=t.long, device=device)
    # Dense feature-prediction targets (B, n_seg, n_sae): required
    # for the model to return ForwardResult since the fp-head gates
    # on fp_targets is not None.
    fp_targets = t.rand(B, N_SEG, TINY_N_SAE, device=device)
    return (
        idx,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id,
        fp_targets,
    )


def _autocast_ctx():
    """Return autocast context matching the training script (bf16 on CUDA, noop on CPU)."""
    if _HAS_CUDA:
        return t.amp.autocast(device_type="cuda", dtype=t.bfloat16)
    return t.amp.autocast(device_type="cpu", enabled=False)


class TestFeaturePredForward:
    @_REQUIRES_CUDA
    def test_returns_forward_result(self):
        model = _make_model(FeaturePredictionConfig())
        (
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        ) = _make_inputs()
        with _autocast_ctx():
            result = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
        assert isinstance(result, ForwardResult)
        assert result.loss.shape == ()
        assert result.lm_loss.shape == ()
        assert result.feature_pred_loss.shape == ()

    @_REQUIRES_CUDA
    def test_combined_loss(self):
        weight = 0.5
        model = _make_model(FeaturePredictionConfig(weight=weight))
        (
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        ) = _make_inputs()
        with _autocast_ctx():
            result = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
        expected = result.lm_loss + weight * result.feature_pred_loss
        assert t.allclose(result.loss, expected, atol=1e-5)

    @_REQUIRES_CUDA
    def test_no_config_returns_scalar(self):
        model = _make_model(fp_config=None)
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, _, _ = _make_inputs()
        with _autocast_ctx():
            result = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        assert isinstance(result, t.Tensor)
        assert result.shape == ()

    @_REQUIRES_CUDA
    def test_no_fp_targets_skips_fp_loss(self):
        """When fp_targets is omitted, skip feature prediction (for eval)."""
        model = _make_model(FeaturePredictionConfig())
        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, _, _ = _make_inputs()
        with _autocast_ctx():
            result = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        # Should fall back to scalar loss
        assert isinstance(result, t.Tensor)

    @_REQUIRES_CUDA
    def test_gradient_flows_to_decoder(self):
        model = _make_model(FeaturePredictionConfig())
        model.train()
        (
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        ) = _make_inputs()
        with _autocast_ctx():
            result = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
        result.loss.backward()
        # Check gradients flow to decoder blocks
        for block in model.transformer.h:
            for p in block.parameters():
                if p.requires_grad:
                    assert p.grad is not None, "Missing gradient on decoder block"

    @_REQUIRES_CUDA
    def test_gradient_flows_to_head(self):
        model = _make_model(FeaturePredictionConfig())
        model.train()
        (
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        ) = _make_inputs()
        with _autocast_ctx():
            result = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
        result.loss.backward()
        assert model.feature_pred_head.weight.grad is not None
        assert model.feature_pred_head.weight.grad.abs().sum() > 0

    @_REQUIRES_CUDA
    def test_logits_mode_unchanged(self):
        """When targets=None, returns logits regardless of feature_pred_config."""
        model = _make_model(FeaturePredictionConfig())
        idx, _, enc_fids, enc_mags, enc_mask, seg_map, _, _ = _make_inputs()
        with _autocast_ctx():
            logits = model(idx, None, enc_fids, enc_mags, enc_mask, seg_map)
        assert logits.shape == (B, T, TINY_VOCAB)


class TestLayerIndexing:
    def test_layer_minus_one(self):
        model = _make_model(FeaturePredictionConfig(layer=-1), device="cpu")
        assert model._feature_pred_layer_idx == TINY_N_LAYER - 1

    def test_layer_minus_two(self):
        model = _make_model(FeaturePredictionConfig(layer=-2), device="cpu")
        assert model._feature_pred_layer_idx == TINY_N_LAYER - 2

    def test_layer_zero(self):
        model = _make_model(FeaturePredictionConfig(layer=0), device="cpu")
        assert model._feature_pred_layer_idx == 0

    def test_out_of_range_raises(self):
        with pytest.raises(AssertionError):
            _make_model(FeaturePredictionConfig(layer=TINY_N_LAYER), device="cpu")


class TestOptimizerIntegration:
    def test_all_params_grouped(self):
        model = _make_model(FeaturePredictionConfig(), device="cpu")
        optimizer = model.setup_optimizer()
        grouped_params = sum(len(g["params"]) for g in optimizer.param_groups)
        total_params = len(list(model.parameters()))
        assert grouped_params == total_params, (
            f"Grouped {grouped_params} but model has {total_params} parameters"
        )

    def test_head_in_adamw_group(self):
        model = _make_model(FeaturePredictionConfig(), device="cpu")
        optimizer = model.setup_optimizer()
        head_weight_id = id(model.feature_pred_head.weight)
        head_bias_id = id(model.feature_pred_head.bias)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) == head_weight_id or id(p) == head_bias_id:
                    assert group["kind"] == "adamw", (
                        "Head params should be in AdamW group"
                    )
                    return
        pytest.fail("Head params not found in any optimizer group")


class TestLossModes:
    @_REQUIRES_CUDA
    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    @pytest.mark.parametrize("loss_fn", list(FeaturePredLossFn))
    def test_all_mode_fn_combinations(self, loss_mode, loss_fn):
        model = _make_model(
            FeaturePredictionConfig(loss_mode=loss_mode, loss_fn=loss_fn)
        )
        model.train()
        (
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        ) = _make_inputs()
        with _autocast_ctx():
            result = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )
        assert isinstance(result, ForwardResult)
        assert not t.isnan(result.loss)
        assert not t.isnan(result.feature_pred_loss)


class TestEstimateFlops:
    def test_flops_includes_feature_pred_head(self):
        """estimate_flops should include the feature prediction head when enabled."""
        model_with = _make_model(FeaturePredictionConfig(), device="cpu")
        model_without = _make_model(fp_config=None, device="cpu")

        flops_with = model_with.estimate_flops(
            avg_features_per_segment=100, segment_size=4
        )
        flops_without = model_without.estimate_flops(
            avg_features_per_segment=100, segment_size=4
        )

        # The feature prediction head adds 6 * (n_embd * n_sae + n_sae) FLOPs
        expected_head_flops = 6 * model_with.feature_pred_head.weight.numel()
        assert flops_with == flops_without + expected_head_flops

    def test_flops_no_head_unchanged(self):
        """estimate_flops should be unchanged when feature prediction is disabled."""
        model = _make_model(fp_config=None, device="cpu")
        flops = model.estimate_flops(avg_features_per_segment=100, segment_size=4)
        assert flops > 0
