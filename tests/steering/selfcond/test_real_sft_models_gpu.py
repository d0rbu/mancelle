"""Opt-in real-checkpoint GPU qualification for self-conditioning training."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as flash_attention_module
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_selfcond import (
    _architecture_from_meta,
    _resolve_fp_config_from_meta,
    run_selfcond_train_step,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    freeze_for_head_only,
)
from steering.conditional.feature_vocab import feature_vocab_from_state_dict
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.profiling.markers import TimingRecorder, timed_region, timing_tags, use_recorder
from steering.refined_types import PositiveFloat, PositiveInt
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch


@dataclass(frozen=True)
class _RealCase:
    model_size: str
    scope: SelfCondTrainScope
    ste_form: SteForm
    segments: int
    parameter_bounds: tuple[int, int]


_CASES = (
    _RealCase(
        model_size="100m",
        scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        segments=8,
        parameter_bounds=(80_000_000, 120_000_000),
    ),
    _RealCase(
        model_size="400m",
        scope=SelfCondTrainScope.FULL,
        ste_form=SteForm.SELECTED,
        segments=4,
        parameter_bounds=(350_000_000, 450_000_000),
    ),
)


def _real_root() -> Path:
    return Path(
        os.environ.get(
            "SELFCOND_REAL_MODEL_ROOT",
            "out/real_selfcond_validation",
        )
    )


def _config(case: _RealCase) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=case.scope,
        ste_form=case.ste_form,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=12.0,
            std=12.0,
            min_k=1,
            max_k=49,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=12.0,
            std=12.0,
            min_k=0,
            max_k=49,
        ),
        seed=4700,
    )


def _load_real_model(
    checkpoint_dir: Path,
    device: str,
) -> tuple[EncoderDecoderGPT, dict[str, Any], Any]:
    state, _optimizer_state, meta = load_checkpoint(
        str(checkpoint_dir),
        749,
        device,
        load_optimizer=False,
    )
    n_sae_features, feature_vocab = feature_vocab_from_state_dict(state)
    assert feature_vocab is not None
    dec_config, enc_config, xattn_config = _architecture_from_meta(meta)
    fp_config = _resolve_fp_config_from_meta(meta)
    assert fp_config is not None
    with t.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=n_sae_features,
            num_buckets=int(meta["num_buckets"]),
            feature_pred_config=fp_config,
            feature_vocab=feature_vocab,
        )
    model.to_empty(device=device)
    model.init_buffers()
    model.load_state_dict(state, strict=True, assign=True)
    model.feature_mask = None
    model.assert_vocab_consistent()
    return model, meta, feature_vocab


def _real_batch(
    *,
    case: _RealCase,
    sequence_len: int,
    n_sae_features: int,
    feature_vocab: Any,
    device: str,
) -> SelfcondBatch:
    batch = make_batch(
        [[case.segments]],
        T=sequence_len,
        F=96,
        n_sae=n_sae_features,
        seed=4701,
    )
    allowed = feature_vocab.new_to_old_tensor(device="cpu")
    batch.enc_fids.copy_(allowed[batch.enc_fids.remainder(feature_vocab.size)])
    original_to_compact = feature_vocab.old_to_new_tensor(device="cpu")
    assert bool((original_to_compact[batch.enc_fids] >= 0).all())
    return SelfcondBatch(
        **{
            name: getattr(batch, name).to(device)
            for name in batch.__dataclass_fields__
        }
    )


def _optimizer(
    case: _RealCase,
    model: EncoderDecoderGPT,
    meta: dict[str, Any],
) -> t.optim.Optimizer:
    if case.scope is SelfCondTrainScope.HEAD_ONLY:
        freeze_for_head_only(model)
        return build_head_only_optimizer(
            model,
            unembedding_lr=0.004,
            adam_betas=(0.8, 0.95),
        )
    user_config = meta["user_config"]
    return model.setup_optimizer(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        scalar_lr=0.5,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        encoder_embedding_lr=float(user_config.get("encoder_embedding_lr", 0.2)),
        encoder_matrix_lr=float(user_config.get("encoder_matrix_lr", 0.02)),
        xattn_matrix_lr=float(user_config.get("xattn_matrix_lr", 0.02)),
    )


@pytest.mark.very_slow
@pytest.mark.skipif(
    os.environ.get("RUN_SELFCOND_REAL_GPU") != "1" or not t.cuda.is_available(),
    reason="requires RUN_SELFCOND_REAL_GPU=1, CUDA, and downloaded d0rb checkpoints",
)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.model_size)
def test_real_sft_checkpoint_full_context_step_is_finite_profiled_and_updates_scope(
    case: _RealCase,
    record_property,
) -> None:
    checkpoint_dir = _real_root() / case.model_size
    required = {
        checkpoint_dir / "model_000749.pt",
        checkpoint_dir / "meta_000749.json",
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    assert not missing, f"missing exact d0rb real-checkpoint files: {missing}"

    flash_attention_module._override_impl = FlashAttnImpl.SDPA
    t._dynamo.config.cache_size_limit = 64
    device = "cuda:0"
    model, meta, feature_vocab = _load_real_model(checkpoint_dir, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert case.parameter_bounds[0] <= parameter_count <= case.parameter_bounds[1]
    assert model.dec_config.sequence_len == 1024
    assert feature_vocab.n_sae_features == 65_536
    assert feature_vocab.size == 2_074
    optimizer = _optimizer(case, model, meta)
    batch = _real_batch(
        case=case,
        sequence_len=model.dec_config.sequence_len,
        n_sae_features=feature_vocab.n_sae_features,
        feature_vocab=feature_vocab,
        device=device,
    )

    observed_names = [
        "feature_pred_head.weight",
        "transformer.h.0.attn.c_qkv.weight",
        "encoder.blocks.0.attn.c_qkv.weight",
        "cross_attn_layers.0.c_q.weight",
    ]
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in observed_names}
    t.cuda.reset_peak_memory_stats()
    recorder = TimingRecorder()
    started = t.cuda.Event(enable_timing=True)
    finished = t.cuda.Event(enable_timing=True)
    started.record()
    with use_recorder(recorder), timing_tags(rank=0, step=0):
        with timed_region("selfcond/train_step"):
            result = run_selfcond_train_step(
                model,
                optimizer,
                [batch],
                _config(case),
                global_step=0,
                ddp_rank=0,
                num_iterations=PositiveInt.parse(1),
                grad_clip_max_norm=PositiveFloat.parse(1.0),
                static_max_seg_size=None,
                full_weight_decay=0.0,
            )
    finished.record()
    t.cuda.synchronize()

    assert math.isfinite(result.loss)
    assert result.metrics.frac_predicted_slots == pytest.approx(
        (case.segments - 1) / case.segments
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {row["name"] for row in recorder.rows()} >= {
        "selfcond/train_step",
        "selfcond/rollout/model_forward",
        "selfcond/pass1/model_forward",
        "selfcond/pass2/lm_forward",
        "selfcond/backward",
        "selfcond/optimizer_step",
    }
    changed = {
        name
        for name in observed_names
        if not t.equal(parameters[name].detach(), before[name])
    }
    if case.scope is SelfCondTrainScope.HEAD_ONLY:
        assert changed == {"feature_pred_head.weight"}
    else:
        assert changed == set(observed_names)

    record_property("parameter_count", parameter_count)
    record_property("sequence_len", model.dec_config.sequence_len)
    record_property("segments", case.segments)
    record_property("elapsed_ms", started.elapsed_time(finished))
    record_property("peak_allocated_bytes", t.cuda.max_memory_allocated())
    record_property("peak_reserved_bytes", t.cuda.max_memory_reserved())
