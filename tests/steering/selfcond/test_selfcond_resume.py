"""T-I3: byte-exact self-conditioning checkpoint and resume contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_selfcond import (
    build_selfcond_checkpoint_meta,
)
from scripts.train.conditional_encdec_selfcond import (
    run_selfcond_train_step as _run_selfcond_train_step,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    freeze_for_head_only,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    SFT_SEGMENT_MODE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
)
from tests.steering.selfcond.helpers import make_batch, tiny_head_model


def run_selfcond_train_step(*args: Any, **kwargs: Any) -> Any:
    """Pass raw resume values through the production beartype boundary."""
    return _run_selfcond_train_step(*args, **kwargs)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _config() -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=0,
            max_k=1,
        ),
        seed=901,
    )


def _base_meta() -> dict:
    return {
        "user_config": {
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
                "loss_fn": "bce",
            },
            "feature_selection": {
                "distribution": "geometric",
                "mean": 1.0,
                "std": 1.0,
                "min_k": 1,
                "max_k": 2,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 1.0,
                "std": 1.0,
                "min_k": 0,
                "max_k": 1,
            },
            "seed": 901,
            "segment_size": 128,
            "num_buckets": 2,
        }
    }


def _identity() -> dict:
    return {
        "feature_dir": "/tmp/features",
        "feature_selection": _base_meta()["user_config"]["feature_selection"],
        "negative_selection": _base_meta()["user_config"]["negative_selection"],
        "seed": 901,
        "segment_size": 128,
        "num_buckets": 2,
    }


def _model_and_optimizer(seed: int):
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=seed)
    freeze_for_head_only(model)
    optimizer = build_head_only_optimizer(
        model,
        unembedding_lr=0.004,
        adam_betas=(0.8, 0.95),
    )
    return model, optimizer


def _run_steps(model, optimizer, cfg, batch, steps: range) -> list[float]:
    losses: list[float] = []
    for step in steps:
        result = run_selfcond_train_step(
            model,
            optimizer,
            [batch],
            cfg,
            global_step=step,
            ddp_rank=0,
            num_iterations=4,
            grad_clip_max_norm=1.0,
            static_max_seg_size=None,
        )
        losses.append(result.loss)
    return losses


def test_save_load_continue_is_bitwise_equal_and_metadata_is_chainable(
    tmp_path: Path,
) -> None:
    cfg = _config()
    batch = make_batch([[3]], T=9, F=3, n_sae=8, seed=902)

    continuous_model, continuous_optimizer = _model_and_optimizer(903)
    continuous_losses = _run_steps(
        continuous_model,
        continuous_optimizer,
        cfg,
        batch,
        range(4),
    )

    interrupted_model, interrupted_optimizer = _model_and_optimizer(903)
    interrupted_losses = _run_steps(
        interrupted_model,
        interrupted_optimizer,
        cfg,
        batch,
        range(2),
    )
    checkpoint_dir = tmp_path / "selfcond_head_only_dense_checkpoints"
    meta = build_selfcond_checkpoint_meta(
        model=interrupted_model,
        base_meta=_base_meta(),
        identity=_identity(),
        cfg=cfg,
        dataloader_state_dict={"cursor": 2},
        loop_state={
            "step": 2,
            "smooth_train_loss": interrupted_losses[-1],
            "min_val_bpb": 4.0,
            "min_val_bpb_selfcond": 3.5,
            "total_training_time": 1.25,
        },
        base_checkpoint_dir="/tmp/base-checkpoint",
    )
    save_checkpoint(
        str(checkpoint_dir),
        2,
        interrupted_model.state_dict(),
        interrupted_optimizer.state_dict(),
        meta,
        rank=0,
    )

    model_data, optimizer_data, loaded_meta = load_checkpoint(
        str(checkpoint_dir),
        2,
        "cpu",
        load_optimizer=True,
        rank=0,
    )
    assert optimizer_data is not None
    resumed_model, resumed_optimizer = _model_and_optimizer(999)
    resumed_model.load_state_dict(model_data, strict=True)
    resumed_optimizer.load_state_dict(optimizer_data)
    interrupted_losses.extend(_run_steps(resumed_model, resumed_optimizer, cfg, batch, range(2, 4)))

    assert interrupted_losses == continuous_losses
    assert resumed_model.state_dict().keys() == continuous_model.state_dict().keys()
    for name, value in continuous_model.state_dict().items():
        assert t.equal(value, resumed_model.state_dict()[name]), name

    assert loaded_meta["model_config"] == meta["model_config"]
    assert loaded_meta["encoder_config"] == meta["encoder_config"]
    assert loaded_meta["xattn_config"] == meta["xattn_config"]
    assert loaded_meta["num_buckets"] == 2
    assert loaded_meta["feature_prediction_config"]["loss_mode"] == "last_token"
    assert loaded_meta["user_config"]["feature_prediction"]["loss_mode"] == "last_token"
    assert loaded_meta["selfcond"] == {
        "data_source": "pretraining",
        "action_ranking_policy": "pooled_fp_logits_v1",
        "train_scope": "head_only",
        "ste_form": "dense",
        "pool_mode": "last_token",
        "positive_counts": {
            "distribution": "geometric",
            "mean": 1.0,
            "std": 1.0,
            "min_k": 1,
            "max_k": 2,
        },
        "negative_counts": {
            "distribution": "geometric",
            "mean": 1.0,
            "std": 1.0,
            "min_k": 0,
            "max_k": 1,
        },
        "seed": 901,
        "original_head_loss_mode": "max_pool",
    }
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in loaded_meta["selfcond"]
    assert SFT_SEGMENT_MODE_META_KEY not in loaded_meta
    assert ONLINE_SAE_LINEAGE_META_KEY not in loaded_meta
    assert loaded_meta["dataloader_state_dict"] == {"cursor": 2}
    assert loaded_meta["loop_state"]["step"] == 2
    assert loaded_meta["base_checkpoint_dir"] == "/tmp/base-checkpoint"
    profile_keys = {
        "profile_timing",
        "profile_trace",
        "profile_trace_steps",
        "profile_output_dir",
    }
    assert profile_keys.isdisjoint(loaded_meta)
    assert profile_keys.isdisjoint(loaded_meta["user_config"])
    assert profile_keys.isdisjoint(loaded_meta["selfcond"])
