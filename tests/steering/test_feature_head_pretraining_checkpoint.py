"""The real pretraining loop saves and resumes a learned transformer head."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

import nanochat.checkpoint_manager as checkpoints
import nanochat.flash_attention as attention
import scripts.train.conditional_encdec_train as pretrain
from nanochat.checkpoint_manager import ModelPhase
from nanochat.common import DummyWandb
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.feature_prediction import FeaturePredictionTargetLayout
from steering.conditional.eval_conditioning import EvalConditioning
from steering.refined_types import PositiveInt
from tests.steering.selfcond.helpers import make_batch
from tests.steering.test_encdec_train_vocab_wiring import _cfg, _shared


def test_pretraining_saves_effective_head_metadata_and_resumes_learned_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    cfg = _cfg(hydra_compose, None)
    for key, value in {
        "model_tag": "tiny-transformer-head",
        "depth": 2, "aspect_ratio": 16, "head_dim": 16, "window_pattern": "L",
        "encoder": {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16},
        "cross_attention.frequency": 1,
        "max_seq_len": 8, "device_batch_size": 1, "total_batch_size": 8,
        "num_iterations": 1, "eval_every": 0, "core_metric_every": 0,
        "sample_every": 0, "steering_eval_every": 0, "save_every": 1,
        "compile": False, "feature_delta": None, "feature_selection.delta_weights": None,
        "feature_prediction": {"block": "transformer", "head_layers": 2, "weight": 0.25},
        "initialize_encoder_embeddings_from_sae": None,
    }.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    model = pretrain.build_encdec_meta_model_from_training_cfg(
        cfg, base_vocab=128, n_sae_features=64, feature_vocab=None,
    )
    model.to_empty(device="cpu")
    shared = _shared(model, cfg, str(tmp_path))
    shared.per_feature_stats = {}
    monkeypatch.setattr(shared, "feature_index", SimpleNamespace())
    shared.fp_target_layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(64), feature_mask=None,
        feature_clusters=None, feature_vocab=None,
    )
    batch = make_batch([[2]], T=8, F=2, n_sae=64, seed=19)
    targets = t.zeros(1, 2, 64)
    targets[0, 1, [3, 17]] = 0.8
    resumed_cursors = []

    def batches(**kwargs):
        cursor = kwargs["resume_state_dict"]
        resumed_cursors.append(copy.deepcopy(cursor))
        offset = 0 if cursor is None else cursor["batch"]
        while True:
            yield (
                batch.inputs, batch.targets, batch.enc_fids, batch.enc_mags,
                batch.enc_mask, batch.enc_pcts, batch.seg_map, batch.seg_doc_id,
                targets, {"epoch": 1, "batch": offset, "ddp_world_size": 1},
            )
            offset += 1

    monkeypatch.setattr(pretrain, "encoder_decoder_data_loader_with_state", batches)
    monkeypatch.setattr(pretrain, "get_report", lambda **_kwargs: SimpleNamespace(log=lambda **_fields: None))
    # Exercise the actual optimizer arithmetic without allocating compiler caches.
    with t.compiler.set_stance("force_eager"):
        pretrain.train_one_config(cfg, shared, DummyWandb())
    metadata_path, = tmp_path.rglob("meta_000001.json")
    meta = json.loads(metadata_path.read_text())
    expected = {
        "weight": 0.25, "layer": -1, "loss_mode": "max_pool", "loss_fn": "bce",
        "block": "transformer", "head_layers": 2,
    }
    assert meta["feature_prediction_config"] == expected
    assert meta["user_config"]["feature_prediction"] == expected
    assert meta["loop_state"]["smooth_fp_loss"] > 0
    first_state = copy.deepcopy(model.state_dict())
    for index in range(2):
        for sublayer in ("attn", "mlp"):
            assert first_state[f"feature_pred_head.residual_blocks.{index}.{sublayer}.c_proj.weight"].count_nonzero()

    monkeypatch.setattr(checkpoints, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 128))
    loaded, _, _ = checkpoints.build_encdec_model(
        str(metadata_path.parent), 1, t.device("cpu"), ModelPhase.TRAIN,
        eval_conditioning=EvalConditioning.not_evaluating(),
    )
    assert all(t.equal(value, loaded.state_dict()[name]) for name, value in first_state.items())
    assert loaded.feature_pred_config == model.feature_pred_config

    # Destroy in-memory weights: continuing successfully must restore model and
    # optimizer state from the artifact written by the actual first invocation.
    with t.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    cfg.resume_from_step = 1
    cfg.num_iterations = 2
    with t.compiler.set_stance("force_eager"):
        pretrain.train_one_config(cfg, shared, DummyWandb())
    assert resumed_cursors == [None, {"epoch": 1, "batch": 1, "ddp_world_size": 1, "device_batch_size": 1}]
    resumed_meta = json.loads(metadata_path.with_name("meta_000002.json").read_text())
    assert resumed_meta["feature_prediction_config"] == expected
    assert resumed_meta["user_config"]["feature_prediction"] == expected
    assert all(t.isfinite(parameter).all() for parameter in model.parameters())
    assert model.feature_pred_head is not None
    assert not t.equal(model.feature_pred_head.weight, first_state["feature_pred_head.weight"])
