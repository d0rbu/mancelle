from __future__ import annotations

import copy

import pytest
import torch as t

from nanochat.gpt import GPTConfig
from steering.conditional.checkpoint_conditioning import (
    PRETRAIN_CONDITIONING_META_KEY,
    read_checkpoint_conditioning,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_checkpoint import (
    validate_grpo_checkpoint_eligibility,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    assert_grpo_gradients_ready,
    configure_grpo_trainable_parameters,
)
from tests.encdec_test_helpers import build_tiny_model


def _model() -> EncoderDecoderGPT:
    model = EncoderDecoderGPT(
        GPTConfig(
            sequence_len=16,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            window_pattern="L",
        ),
        EncoderConfig(n_embd=16, depth=1, n_head=1, n_kv_head=1, head_dim=16),
        CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2),
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    model.init_weights()
    return model


def _metadata() -> dict:
    return {
        "user_config": {"dataset": "standard", "label_dropout": 0.0},
        "encoder_config": {},
        "xattn_config": {},
        "num_buckets": 2,
        "feature_prediction_config": {
            "block": "mlp",
            "head_layers": 0,
            "weight": 0.1,
            "layer": -1,
            "loss_mode": "max_pool",
            "loss_fn": "bce",
        },
        PRETRAIN_CONDITIONING_META_KEY: {
            "positive_selection": {
                "distribution": "geometric",
                "mean": 1.0,
                "std": 1.0,
                "min_k": 1,
                "max_k": 3,
                "mode": "weighted",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 1.0,
                "std": 1.0,
                "min_k": 0,
                "max_k": 2,
            },
            "feature_delta": None,
            "delta_weights": None,
            "label_dropout": 0.0,
            "next_segment_conditioning_offset": False,
        },
    }


def test_strict_checkpoint_eligibility_resolves_checkpoint_counts() -> None:
    eligibility = validate_grpo_checkpoint_eligibility(
        _model(),
        _metadata(),
        checkpoint_dir="/checkpoint",
        action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        seed=4,
    )
    assert eligibility.capture_layer == 1
    assert eligibility.cross_attention_layers == (1,)
    assert eligibility.feature_count == 8
    assert eligibility.policy.positive_counts.max_k == 3
    assert eligibility.policy.negative_counts.max_k == 2


def test_conditioning_lineage_republishes_all_grpo_eligibility_fields() -> None:
    republished = read_checkpoint_conditioning(_metadata()).metadata_dict()
    lineage = republished[PRETRAIN_CONDITIONING_META_KEY]
    assert lineage["positive_selection"]["distribution"] == "geometric"
    assert lineage["negative_selection"]["distribution"] == "geometric"
    assert lineage["label_dropout"] == 0.0
    assert lineage["next_segment_conditioning_offset"] is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda metadata: metadata.pop(PRETRAIN_CONDITIONING_META_KEY),
            "pretrain_conditioning",
        ),
        (
            lambda metadata: metadata["feature_prediction_config"].update(loss_mode="full_segment"),
            "max_pool",
        ),
        (
            lambda metadata: metadata.update(num_buckets=3),
            "exactly two buckets",
        ),
        (
            lambda metadata: metadata["user_config"].update(label_dropout=0.2),
            "dropout",
        ),
        (
            lambda metadata: metadata[PRETRAIN_CONDITIONING_META_KEY].update(
                positive_selection=None
            ),
            "positive",
        ),
        (
            lambda metadata: (
                metadata[PRETRAIN_CONDITIONING_META_KEY].pop("label_dropout"),
                metadata["user_config"].pop("label_dropout"),
            ),
            "dropout lineage",
        ),
        (
            lambda metadata: metadata[PRETRAIN_CONDITIONING_META_KEY].pop(
                "next_segment_conditioning_offset"
            ),
            "offset lineage",
        ),
        (
            lambda metadata: metadata[PRETRAIN_CONDITIONING_META_KEY].update(
                next_segment_conditioning_offset=True
            ),
            "offset is unsupported",
        ),
        (
            lambda metadata: metadata[PRETRAIN_CONDITIONING_META_KEY]["positive_selection"].update(
                distribution="normal"
            ),
            "geometric",
        ),
        (
            lambda metadata: metadata[PRETRAIN_CONDITIONING_META_KEY]["negative_selection"].update(
                distribution="normal"
            ),
            "geometric",
        ),
    ],
)
def test_checkpoint_eligibility_rejects_unrecoverable_lineage(mutate, match: str) -> None:
    metadata = copy.deepcopy(_metadata())
    mutate(metadata)
    with pytest.raises((SFTGRPOError, ValueError), match=match):
        validate_grpo_checkpoint_eligibility(
            _model(),
            metadata,
            checkpoint_dir="/checkpoint",
            action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
            seed=4,
        )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_all_masked_cuda_policy_path_has_exact_zero_connected_gradients() -> None:
    """Every optimized ancestor remains in the graph for canonical empty slots."""
    feature_prediction = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = (
        build_tiny_model(
            n_sae_features=8,
            num_buckets=2,
            feature_pred_config=feature_prediction,
            n_layer=2,
            n_embd=64,
            seq_len=16,
            xattn_frequency=2,
        )
        .cuda()
        .eval()
    )
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=False)
    inputs = t.tensor([[1, 2, 3, 4]], dtype=t.int64, device="cuda")
    feature_ids = t.zeros((1, 2, 1), dtype=t.int64, device="cuda")
    magnitudes = t.zeros_like(feature_ids)
    mask = t.zeros_like(feature_ids, dtype=t.bool)
    seg_map = t.tensor([[0, 0, 1, 1]], dtype=t.int64, device="cuda")

    logits = model(
        inputs,
        None,
        feature_ids,
        magnitudes,
        mask,
        seg_map,
        fp_logits_only=True,
    )
    (logits.float().sum() * 0.0).backward()

    assert_grpo_gradients_ready(model, expected_manifest=manifest)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert t.count_nonzero(parameter.grad).item() == 0
