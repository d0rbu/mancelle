"""Integration tests for the encoder-decoder conditional pretraining system.

Tests real component interaction: encoder signal influence, segment isolation,
dataloader/model compatibility, and end-to-end pipeline correctness.
"""

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.model import ForwardResult
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    build_model_with_feature_prediction,
    build_tiny_model,
    make_precomputed_dataloader,
)
from tests.feature_prediction_helpers import dense_targets_from_sparse

# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend (FlexAttention doesn't support backward on CPU)."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(n_sae=N_SAE_FEATURES, B=2, T=SEQ_LEN, n_seg=4, F=5, num_buckets=NUM_BUCKETS):
    """Create a minimal synthetic batch for direct model tests."""
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    enc_fids = torch.randint(0, n_sae, (B, n_seg, F), dtype=torch.long)
    enc_mags = torch.randint(0, num_buckets, (B, n_seg, F), dtype=torch.long)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    enc_pcts = torch.rand(B, n_seg, F)
    seg_size = T // n_seg
    seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // seg_size
    seg_map = seg_map.clamp(max=n_seg - 1)
    seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
    return idx, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id


def _perturb_c_proj(model):
    """Perturb c_proj weights so gradients can flow through cross-attention."""
    for xattn in model.cross_attn_layers.values():
        xattn.c_proj.weight.data.fill_(0.01)
    for block in model.encoder.blocks:
        block.attn.c_proj.weight.data.fill_(0.01)
        block.mlp.c_proj.weight.data.fill_(0.01)
    for block in model.transformer.h:
        block.attn.c_proj.weight.data.fill_(0.01)
        block.mlp.c_proj.weight.data.fill_(0.01)


# ---------------------------------------------------------------------------
# TestEncoderSignal
# ---------------------------------------------------------------------------


class TestEncoderSignal:
    def test_encoder_output_influences_logits(self):
        """Forward with real enc_mask vs all-zeros enc_mask produces different logits."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.eval()

        idx, _, enc_fids, enc_mags, enc_mask, _, seg_map, _ = _make_batch()
        zero_mask = torch.zeros_like(enc_mask)

        with torch.no_grad():
            logits_real = model(idx, None, enc_fids, enc_mags, enc_mask, seg_map)
            logits_zero = model(idx, None, enc_fids, enc_mags, zero_mask, seg_map)

        diff = (logits_real - logits_zero).norm().item()
        assert diff > 1e-3, f"Logit difference too small: {diff}"

    def test_different_features_produce_different_logits(self):
        """Shifting feature IDs produces different logits."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.eval()

        n_sae = N_SAE_FEATURES
        idx, _, enc_fids, enc_mags, enc_mask, _, seg_map, _ = _make_batch(n_sae=n_sae)
        enc_fids_shifted = (enc_fids + 50) % n_sae

        with torch.no_grad():
            logits_orig = model(idx, None, enc_fids, enc_mags, enc_mask, seg_map)
            logits_shifted = model(idx, None, enc_fids_shifted, enc_mags, enc_mask, seg_map)

        diff = (logits_orig - logits_shifted).norm().item()
        assert diff > 1e-3, f"Logit difference too small after shifting features: {diff}"

    def test_encoder_gradients_nonzero(self):
        """After backward, all encoder named parameters have nonzero gradients."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.train()

        idx, targets, enc_fids, enc_mags, enc_mask, _, seg_map, _ = _make_batch()
        loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
                assert param.grad.abs().sum().item() > 0, f"Zero gradient for encoder.{name}"

    def test_cross_attention_kv_gradients_flow(self):
        """After backward, all cross_attn_layers[*].c_kv.weight have nonzero gradients."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.train()

        idx, targets, enc_fids, enc_mags, enc_mask, _, seg_map, _ = _make_batch()
        loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        for layer_key, layer in model.cross_attn_layers.items():
            grad = layer.c_kv.weight.grad
            assert grad is not None, f"No gradient for cross_attn_layers[{layer_key}].c_kv.weight"
            assert grad.abs().sum().item() > 0, (
                f"Zero gradient for cross_attn_layers[{layer_key}].c_kv.weight"
            )

    def test_feature_prediction_gradient_flows_to_encoder(self):
        """Feature prediction auxiliary loss adds extra gradient signal to the decoder.

        The FP head reads from the decoder hidden state at a captured layer, so
        the FP loss gradient flows through the decoder blocks (and cross-attention)
        back towards the encoder. We verify this by confirming that at least one
        decoder transformer block parameter receives a different gradient when FP
        is enabled vs disabled.
        """
        # Build two models with identical weights except the FP head
        model_fp = build_model_with_feature_prediction()
        model_no_fp = build_tiny_model()

        # Copy matching weights from model_fp to model_no_fp
        fp_state = model_fp.state_dict()
        no_fp_state = model_no_fp.state_dict()
        shared_keys = set(no_fp_state.keys()) & set(fp_state.keys())
        for k in shared_keys:
            no_fp_state[k] = fp_state[k].clone()
        model_no_fp.load_state_dict(no_fp_state)

        _perturb_c_proj(model_fp)
        _perturb_c_proj(model_no_fp)
        model_fp.train()
        model_no_fp.train()

        idx, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id = _make_batch()

        # Build fp_targets from the batch features (for test purposes)
        fp_targets = dense_targets_from_sparse(
            enc_fids.flatten(0, 1),
            enc_pcts.flatten(0, 1),
            enc_mask.flatten(0, 1),
            N_SAE_FEATURES,
        ).view(enc_fids.shape[0], -1, N_SAE_FEATURES)

        # Forward + backward with FP
        result_fp = model_fp(
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )
        assert isinstance(result_fp, ForwardResult), "Expected ForwardResult with FP enabled"
        result_fp.loss.backward()

        # Forward + backward without FP (same inputs)
        loss_no_fp = model_no_fp(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss_no_fp.backward()

        # The FP head gradient flows through the decoder hidden state where it was
        # captured. At least one decoder transformer block parameter should receive a
        # different gradient when FP is active (due to the extra gradient path via FP loss).
        any_differ = False
        fp_params = dict(model_fp.transformer.h.named_parameters())
        no_fp_params = dict(model_no_fp.transformer.h.named_parameters())
        for name, fp_param in fp_params.items():
            g_fp = fp_param.grad
            g_nf = no_fp_params[name].grad
            if g_fp is not None and g_nf is not None:
                if not torch.allclose(g_fp, g_nf, atol=1e-7):
                    any_differ = True
                    break

        assert any_differ, (
            "Decoder transformer block gradients are identical with and without "
            "feature prediction — FP should add extra gradient signal via the "
            "captured hidden state."
        )


# ---------------------------------------------------------------------------
# TestSegmentIsolation
# ---------------------------------------------------------------------------


class TestSegmentIsolation:
    def test_segment_feature_modification_localized(self):
        """Modifying segment 1's features affects seg1 tokens more than seg0 tokens."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.eval()

        B, T, n_seg, F = 1, SEQ_LEN, 4, 5
        idx, _, enc_fids, enc_mags, enc_mask, _, seg_map, _ = _make_batch(
            B=B, T=T, n_seg=n_seg, F=F
        )

        # Baseline logits
        with torch.no_grad():
            logits_base = model(idx, None, enc_fids, enc_mags, enc_mask, seg_map)

        # Modify segment 1's feature IDs
        enc_fids_mod = enc_fids.clone()
        enc_fids_mod[0, 1, :] = (enc_fids_mod[0, 1, :] + 50) % N_SAE_FEATURES

        with torch.no_grad():
            logits_mod = model(idx, None, enc_fids_mod, enc_mags, enc_mask, seg_map)

        # Compute per-position L2 difference
        diff = (logits_mod - logits_base).norm(dim=-1).squeeze(0)  # (T,)

        # Identify tokens belonging to seg0 and seg1
        seg0_mask = seg_map[0] == 0
        seg1_mask = seg_map[0] == 1

        if seg0_mask.sum() == 0 or seg1_mask.sum() == 0:
            pytest.skip("Batch doesn't have enough segments")

        avg_diff_seg0 = diff[seg0_mask].mean().item()
        avg_diff_seg1 = diff[seg1_mask].mean().item()

        assert avg_diff_seg1 > avg_diff_seg0, (
            f"Segment 1 modification should affect seg1 tokens more than seg0: "
            f"seg1={avg_diff_seg1:.5f}, seg0={avg_diff_seg0:.5f}"
        )

    def test_cross_document_feature_leakage(self):
        """Modifying doc1's features does not change doc0's per-token losses."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.eval()

        B, T, n_seg, F = 1, SEQ_LEN, 4, 5
        idx, targets, enc_fids, enc_mags, enc_mask, _, seg_map, seg_doc_id = _make_batch(
            B=B, T=T, n_seg=n_seg, F=F
        )

        # Create two-doc setup: segs 0,1 from doc0; segs 2,3 from doc1
        seg_doc_id = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

        # Find token positions belonging to each doc via seg_map
        doc0_segs = (seg_doc_id[0] == 0).nonzero(as_tuple=True)[0].tolist()
        doc1_segs = (seg_doc_id[0] == 1).nonzero(as_tuple=True)[0].tolist()
        if len(doc0_segs) == 0 or len(doc1_segs) == 0:
            pytest.skip("Batch doesn't have at least 2 documents")

        doc0_tok_mask = torch.zeros(T, dtype=torch.bool)
        for s in doc0_segs:
            doc0_tok_mask |= seg_map[0] == s

        if doc0_tok_mask.sum() == 0:
            pytest.skip("No tokens found for doc0")

        # Baseline: per-token loss with loss_reduction="none"
        with torch.no_grad():
            loss_base = model(
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                loss_reduction="none",
            )  # (B*T,)
        loss_base = loss_base.view(B, T)[0]  # (T,)

        # Modify doc1's features
        enc_fids_mod = enc_fids.clone()
        for s in doc1_segs:
            enc_fids_mod[0, s, :] = (enc_fids_mod[0, s, :] + 50) % N_SAE_FEATURES

        with torch.no_grad():
            loss_mod = model(
                idx,
                targets,
                enc_fids_mod,
                enc_mags,
                enc_mask,
                seg_map,
                loss_reduction="none",
            )
        loss_mod = loss_mod.view(B, T)[0]  # (T,)

        # Doc0's per-token losses should be unchanged
        diff = (loss_mod[doc0_tok_mask] - loss_base[doc0_tok_mask]).abs().max().item()
        assert diff < 1e-5, (
            f"Doc0 losses changed when doc1 features were modified: max_diff={diff:.2e}"
        )


# ---------------------------------------------------------------------------
# TestDataloaderModelIntegration
# ---------------------------------------------------------------------------


class TestDataloaderModelIntegration:
    def test_shape_compatibility_multiple_batches(self, tmp_path):
        """5 batches from dataloader are all processable by model.forward()."""
        model = build_tiny_model()
        model.eval()

        dl = make_precomputed_dataloader(tmp_path, B=2)

        for _ in range(5):
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )
            with torch.no_grad():
                result = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            loss_val = result.item() if isinstance(result, torch.Tensor) else result.loss.item()
            assert torch.isfinite(torch.tensor(loss_val)), f"Non-finite loss: {loss_val}"

    def test_seg_map_no_oob(self, tmp_path):
        """seg_map values must be within [0, n_seg) for 5 batches."""
        dl = make_precomputed_dataloader(tmp_path, B=2)

        for _ in range(5):
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )
            n_seg = enc_fids.shape[1]
            assert seg_map.max().item() < n_seg, (
                f"seg_map max {seg_map.max().item()} >= n_seg {n_seg}"
            )
            assert seg_map.min().item() >= 0, f"seg_map min {seg_map.min().item()} < 0"

    def test_enc_mask_has_true_per_real_segment(self, tmp_path):
        """For segments where seg_doc_id >= 0, enc_mask.sum(dim=-1) > 0."""
        dl = make_precomputed_dataloader(tmp_path, B=2)

        for _ in range(5):
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )
            # For real segments (doc_id >= 0), enc_mask should have at least one True
            real_seg_mask = seg_doc_id >= 0  # (B, n_seg)
            if real_seg_mask.any():
                feat_counts = enc_mask.sum(dim=-1)  # (B, n_seg)
                real_feat_counts = feat_counts[real_seg_mask]
                assert (real_feat_counts > 0).all(), (
                    "Some real segments have no active features in enc_mask"
                )


# ---------------------------------------------------------------------------
# TestPipelineEndToEnd
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    def test_all_loss_components_finite(self, tmp_path):
        """Model with feature prediction: all 3 loss components finite and > 0."""
        model = build_model_with_feature_prediction()
        model.eval()

        dl = make_precomputed_dataloader(tmp_path, B=2)

        x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, fp_targets, _ = next(dl)

        with torch.no_grad():
            result = model(
                x,
                y,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id=seg_doc_id,
                fp_targets=fp_targets,
            )

        assert isinstance(result, ForwardResult), f"Expected ForwardResult, got {type(result)}"

        for name, val in [
            ("loss", result.loss),
            ("lm_loss", result.lm_loss),
            ("feature_pred_loss", result.feature_pred_loss),
        ]:
            assert torch.isfinite(val), f"{name} is not finite: {val.item()}"
            assert val.item() > 0, f"{name} is not > 0: {val.item()}"


# ---------------------------------------------------------------------------
# TestNegativeSamplingIntegration
# ---------------------------------------------------------------------------


def _make_pipeline_with_negatives(n_sae_features, num_buckets):
    """Build a FeaturePipelineConfig with negative sampling enabled.

    When negatives are present, effective_buckets = num_buckets - 1.
    Positive features use buckets 0..effective_buckets-1;
    negative features are assigned bucket index effective_buckets.
    """
    effective_buckets = num_buckets - 1
    # Boundaries partition effective_buckets positive buckets
    # (need effective_buckets - 1 internal boundaries).
    boundaries = (
        torch.linspace(
            1 / effective_buckets,
            1.0 - 1 / effective_buckets,
            effective_buckets - 1,
            dtype=torch.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )
    neg_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
        ),
        n_sae_features=n_sae_features,
        rng=torch.Generator().manual_seed(42),
    )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=neg_bundle,
        effective_buckets=effective_buckets,
    )


class TestNegativeSamplingIntegration:
    """End-to-end tests for negative feature sampling through the full model."""

    def test_negative_sampling_forward_backward(self, tmp_path):
        """Model forward/backward with negative sampling produces finite loss
        and gradients flow to magnitude_embed and feature_embed."""
        model = build_tiny_model()
        _perturb_c_proj(model)
        model.train()

        pipeline = _make_pipeline_with_negatives(N_SAE_FEATURES, NUM_BUCKETS)
        dl = make_precomputed_dataloader(tmp_path, B=2, pipeline=pipeline)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = (
                next(dl)
            )
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
            loss.backward()
            optimizer.step()

        # After the last backward, check gradients on embedding tables
        for name, param in [
            ("encoder.feature_embed", model.encoder.feature_embed.weight),
            ("encoder.magnitude_embed", model.encoder.magnitude_embed.weight),
        ]:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"
