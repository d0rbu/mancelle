"""Full training-loop smoke test using the live dataloader fixture.

This is the closest in-tree analog of the user-facing
``conditional_encdec_train.py`` script: it exercises the live
dataloader (real synthetic feature shards + parquets + tokenizer
checksums + FeatureIndex + ThresholdSelectionConfig pipeline + fp_targets
build) and a real encoder-decoder model with the ``MAX_POOL``
feature-prediction head, training with ``AdamW`` for several real
batches and asserting the loss path works end-to-end.

This is the "FULL TRAINING smoke" the user requested — it exercises every
component the production training script touches (model + dataloader +
optimizer + autograd + fp_targets) other than the steering-eval probe
and wandb hooks.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.dataset import DatasetSplit
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.dataloader import (
    encoder_decoder_data_loader_with_state,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_index import FeatureIndex
from steering.refined_types import PositiveInt
from tests.steering.test_live_dataloader import (
    _build_threshold_pipeline,
    _make_synthetic_data,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _build_tiny_encdec_for_full_loop(n_sae_features: int):
    """Build a tiny EncoderDecoderGPT with MAX_POOL fp head matching the
    production model surface."""
    dec_config = GPTConfig(
        sequence_len=128,  # Match _T from test_live_dataloader (default _T=64)
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=64, depth=2, n_head=2, n_kv_head=2, head_dim=32,
    )
    xattn_config = CrossAttentionConfig(frequency=2, n_head=4, n_kv_head=4)
    fp_config = FeaturePredictionConfig(
        weight=0.5, layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=n_sae_features,
        num_buckets=10,
        feature_pred_config=fp_config,
    )
    model.init_weights()
    return model


class TestMaxPoolFullTrainingLoopSmoke:
    """End-to-end smoke: live dataloader + tiny encdec + MAX_POOL fp head
    + AdamW + several training steps."""

    def test_full_training_loop_runs_with_max_pool(self, tmp_path):
        torch.manual_seed(0)
        # Build synthetic data (parquets + feature shards + token boundary
        # overlay + percentile overlay) — same fixture exercised by
        # test_live_dataloader.py's 144 passing tests.
        env = _make_synthetic_data(tmp_path, n_docs=20, n_segs_per_doc=2,
                                    n_feats_per_seg=4)

        # Build the FeatureIndex over the real synthetic shards.
        feature_index = FeatureIndex(
            env["feature_dir"],
            tokenizer_checksum=env["tokenizer_checksum"],
            percentile_checksum=env.get("percentile_checksum"),
        )
        # Production-style threshold pipeline.
        pipeline = _build_threshold_pipeline(threshold=2)

        # Build the real model with MAX_POOL fp head.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_tiny_encdec_for_full_loop(
            n_sae_features=feature_index.n_sae_features
            if hasattr(feature_index, "n_sae_features")
            else 200,  # ty: ignore[invalid-argument-type]
        ).to(device)
        model.train()

        # Live dataloader exactly as the production training script
        # constructs it — feature_index, pipeline, segment_size,
        # n_sae_features, etc.
        gen = encoder_decoder_data_loader_with_state(
            tokenizer=env["tokenizer"],
            B=2,
            T=64,
            split=DatasetSplit.TRAIN,
            feature_index=feature_index,
            pipeline=pipeline,
            segment_size=PositiveInt.parse(20),
            decoder_segment_size=None,
            device=str(device),
            tokenizer_batch_size=64,
            buffer_size=20,
            data_dir=env["data_dir"],
            prefetch_batches=0,  # synchronous to keep the test deterministic
            label_dropout=0.0,
            max_shards=None,
            n_sae_features=200,
            pad_F=None,
            pad_n_seg=None,
            doc_prefetch_depth=0,
        )

        # Optimizer (matches production: AdamW + per-step zero_grad/backward/step)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        n_steps = 5
        losses = []
        fp_losses = []
        for step in range(n_steps):
            batch = next(gen)
            (x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map,
             seg_doc_id, fp_targets, _state_dict) = batch
            opt.zero_grad()
            result = model(
                x.to(device), y.to(device),
                enc_feature_ids=enc_fids.to(device),
                enc_magnitudes=enc_mags.to(device),
                enc_mask=enc_mask.to(device),
                seg_map=seg_map.to(device),
                seg_doc_id=seg_doc_id.to(device),
                fp_targets=fp_targets.to(device) if fp_targets is not None else None,
            )
            assert torch.isfinite(result.loss), f"non-finite loss at step {step}"
            assert torch.isfinite(result.lm_loss), f"non-finite lm_loss at step {step}"
            assert torch.isfinite(result.feature_pred_loss), (
                f"non-finite fp_loss at step {step}"
            )
            result.loss.backward()
            opt.step()
            losses.append(result.loss.item())
            fp_losses.append(result.feature_pred_loss.item())

        # Sanity checks
        assert len(losses) == n_steps
        assert all(loss > 0 for loss in losses), "all losses positive"
        # The fp head's gradient flows correctly (its weights have grad)
        assert model.feature_pred_head.weight.grad is not None
        assert model.feature_pred_head.weight.grad.abs().sum().item() > 0
