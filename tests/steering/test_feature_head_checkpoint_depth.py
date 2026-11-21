"""Residual feature-head architecture survives every checkpoint consumer."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

import nanochat.checkpoint_manager as checkpoints
import nanochat.flash_attention as attention
from nanochat.checkpoint_manager import ModelPhase
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import MLP, Linear, OptimizerKind
from scripts.train import conditional_encdec_selfcond as selfcond
from scripts.train import conditional_encdec_sft as sft
from scripts.train import conditional_encdec_sft_grpo as grpo
from scripts.train.conditional_encdec_train import build_encdec_meta_model_from_training_cfg
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    feature_prediction_config_from_metadata,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_vocab import FeatureVocab
from tests.encdec_test_helpers import N_SAE_FEATURES, NUM_BUCKETS, build_tiny_model
from tests.scripts.test_encdec_sft_segment_mode_startup import _patched_main_runtime
from tests.steering.selfcond.helpers import make_batch
from tests.steering.selfcond.test_selfcond_resume import _config, _identity

READERS = (
    sft._resolve_fp_config_from_meta,
    selfcond._resolve_fp_config_from_meta,
    grpo._feature_prediction_from_meta,
)


def test_grpo_requires_a_head_even_when_shared_metadata_parser_accepts_headless():
    assert feature_prediction_config_from_metadata({}) is None
    with pytest.raises(SFTGRPOError, match="does not recover feature-prediction lineage"):
        grpo._feature_prediction_from_meta({})


def _raw_config(**extra: object) -> dict:
    return {
        "weight": 0.1,
        "layer": -1,
        "loss_mode": "max_pool",
        "loss_fn": "bce",
        "block": "mlp",
        "head_layers": 0,
        **extra,
    }


def _metadata(model: EncoderDecoderGPT) -> dict:
    fp = sft._serialize_fp_config(model.feature_pred_config)
    return {
        "model_config": asdict(model.dec_config),
        "encoder_config": asdict(model.enc_config),
        "xattn_config": asdict(model.xattn_config),
        "num_buckets": model.encoder.config.num_buckets,
        "feature_prediction_config": fp,
        "user_config": {
            "feature_prediction": copy.deepcopy(fp),
            "num_buckets": model.encoder.config.num_buckets,
            "segment_size": 128,
        },
        **({"feature_vocab": model.feature_vocab.metadata_dict()} if model.feature_vocab else {}),
    }


def test_historical_headless_sft_needs_no_metadata_conversion(tmp_path, monkeypatch):
    model = build_tiny_model()
    assert model.feature_pred_config is None
    meta = _metadata(model)
    meta["user_config"]["feature_prediction_mlp_layers"] = 0
    assert sft._resolve_fp_config_from_meta(meta) is None
    checkpoints.save_checkpoint(str(tmp_path), 2, model.state_dict(), None, meta, rank=0)
    monkeypatch.setattr(checkpoints, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: model.dec_config.vocab_size))
    restored, _, _ = checkpoints.build_encdec_model(
        str(tmp_path), 2, t.device("cpu"), ModelPhase.TRAIN,
        eval_conditioning=EvalConditioning.not_evaluating(),
    )
    assert restored.feature_pred_config is None and restored.feature_pred_head is None
    assert all(t.equal(tensor, restored.state_dict()[name]) for name, tensor in model.state_dict().items())
    for invalid in (None, False, True, 1, "0", 0.0):
        meta["user_config"]["feature_prediction_mlp_layers"] = invalid
        with pytest.raises(ValueError, match="removed"):
            sft._resolve_fp_config_from_meta(meta)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("location", ["user_config", "top_level"])
def test_manual_readers_retain_explicit_saved_architecture(reader, depth, location):
    raw = _raw_config(head_layers=depth)
    meta = (
        {"user_config": {"feature_prediction": raw}}
        if location == "user_config"
        else {"feature_prediction_config": raw}
    )
    resolved = reader(meta)
    assert resolved is not None
    assert resolved.head_layers == depth
    assert resolved.block.value == "mlp"


@pytest.mark.parametrize("reader", READERS)
def test_manual_readers_reject_conflicting_mirrors(reader):
    with pytest.raises(ValueError, match="conflicting.*mirrors"):
        reader({
            "user_config": {"feature_prediction": _raw_config(head_layers=2)},
            "feature_prediction_config": _raw_config(head_layers=1),
        })


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("depth", [True, False, -1, 1.0, "2", None])
def test_checkpoint_depth_is_never_coerced(reader, depth):
    with pytest.raises(ValueError, match="head_layers"):
        reader({"feature_prediction_config": _raw_config(head_layers=depth)})


@pytest.mark.parametrize("saved", [None, 0, 1, 2])
@pytest.mark.parametrize("requested", [0, 1, 2])
@pytest.mark.parametrize("resuming", [False, True])
def test_sft_depth_transition_contract(saved, requested, resuming):
    config = None if saved is None else FeaturePredictionConfig(head_layers=saved)
    permitted = (
        requested == 0 if saved is None else requested == saved or (not resuming and saved == 0)
    )
    if permitted:
        sft.validate_feature_prediction_architecture(config, FeaturePredictionBlock.MLP, requested, resuming=resuming)
    else:
        with pytest.raises(ValueError, match="feature_prediction_head_layers|headless checkpoint"):
            sft.validate_feature_prediction_architecture(config, FeaturePredictionBlock.MLP, requested, resuming=resuming)


@pytest.mark.parametrize("raw", ["true", "false", "-1", "1.0", "'2'", "null"])
def test_bad_hydra_depth_fails_before_compute_initialization(monkeypatch, raw):
    monkeypatch.setattr(sft.sys, "argv", ["sft", f"feature_prediction_head_layers={raw}"])
    monkeypatch.setattr(
        sft,
        "compute_init",
        lambda *_: pytest.fail("invalid depth reached distributed/device initialization"),
    )
    with pytest.raises(ValueError, match="head_layers"):
        sft.main()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
def test_both_sft_checkpoint_mirrors_and_selfcond_pool_override_retain_depth(depth, block):
    model = build_tiny_model(feature_pred_config=FeaturePredictionConfig(block=block, head_layers=depth))
    meta = _metadata(model)
    saved_user = sft._checkpoint_user_config_with_fp({}, model.feature_pred_config)
    assert saved_user["feature_prediction"] == meta["feature_prediction_config"]
    assert saved_user["feature_prediction"]["head_layers"] == depth
    chained = selfcond.build_selfcond_checkpoint_meta(
        model=model,
        base_meta=meta,
        identity=_identity(),
        cfg=_config(),
        dataloader_state_dict=None,
        loop_state={"step": 2},
        base_checkpoint_dir="/test/base",
    )
    assert chained["feature_prediction_config"]["head_layers"] == depth
    assert chained["feature_prediction_config"]["block"] == block.value
    assert chained["user_config"]["feature_prediction"] == chained["feature_prediction_config"]
    assert chained["feature_prediction_config"]["loss_mode"] == "last_token"
    assert meta["feature_prediction_config"]["loss_mode"] == "max_pool"


@pytest.mark.parametrize("base_depth,resume_depth", [(0, 2), (2, 0), (1, 2), (2, 1)])
def test_selfcond_resume_strict_load_rejects_different_tensor_depths(base_depth, resume_depth):
    base = build_tiny_model(feature_pred_config=FeaturePredictionConfig(head_layers=base_depth))
    resume = build_tiny_model(feature_pred_config=FeaturePredictionConfig(head_layers=resume_depth))
    # Self-conditioning constructs the base architecture before loading resume tensors.
    with pytest.raises(RuntimeError, match="(Unexpected|Missing) key"):
        _restore(base, _metadata(base), resume.state_dict(), selfcond._resolve_fp_config_from_meta)


def _restore(model: EncoderDecoderGPT, raw: dict, state: dict, reader) -> EncoderDecoderGPT:
    with t.device("meta"):
        restored = EncoderDecoderGPT(
            model.dec_config,
            model.enc_config,
            model.xattn_config,
            n_sae_features=model.encoder.config.n_sae_features,
            num_buckets=model.encoder.config.num_buckets,
            feature_pred_config=reader(raw),
            feature_vocab=model.feature_vocab,
        )
    restored.to_empty(device="cpu")
    restored.init_buffers()
    restored.load_state_dict(state, strict=True, assign=True)
    restored.assert_vocab_consistent()
    return restored


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
def test_learned_weights_restore_exactly_through_all_consumers(
    tmp_path, monkeypatch, compact, depth, block
):
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    vocab = FeatureVocab(n_sae_features=8, original_ids=(1, 4, 7)) if compact else None
    model = build_tiny_model(
        n_sae_features=8,
        feature_vocab=vocab,
        feature_pred_config=FeaturePredictionConfig(block=block, head_layers=depth),
    )
    assert model.feature_pred_head is not None
    # Nonzero learned residuals make omission/reinitialization observable.
    with t.no_grad():
        for name, parameter in model.feature_pred_head.named_parameters():
            if "c_proj" in name:
                parameter.uniform_(-0.05, 0.05)
    state = model.state_dict()
    meta = _metadata(model)
    hidden = t.randn(2, 5, model.dec_config.n_embd)
    expected = model.apply_feature_prediction_head(hidden)
    checkpoints.save_checkpoint(str(tmp_path), 2, state, None, meta, rank=0)
    tokenizer = SimpleNamespace(get_vocab_size=lambda: model.dec_config.vocab_size)
    monkeypatch.setattr(checkpoints, "get_tokenizer", lambda: tokenizer)
    generic, _, _ = checkpoints.build_encdec_model(
        str(tmp_path),
        2,
        t.device("cpu"),
        ModelPhase.TRAIN,
        eval_conditioning=EvalConditioning.not_evaluating(),
    )
    restored = [
        generic,
        grpo._build_model(state, meta, device="cpu"),
        _restore(model, meta, state, sft._resolve_fp_config_from_meta),
        _restore(model, meta, state, selfcond._resolve_fp_config_from_meta),
    ]
    for actual in restored:
        assert actual.feature_pred_config is not None
        assert actual.feature_pred_head is not None
        assert actual.feature_pred_config.head_layers == depth
        assert actual.feature_pred_config.block == block
        assert actual.state_dict().keys() == state.keys()
        for name, value in actual.state_dict().items():
            assert t.equal(value, state[name]), name
        assert t.equal(actual.apply_feature_prediction_head(hidden), expected)


@pytest.mark.parametrize(
    "corruption",
    ["legacy_depth", "wrong_depth", "missing_block", "extra_block", "partial_projection"],
)
@pytest.mark.parametrize("reader", READERS)
def test_strict_restore_rejects_architecture_tensor_disagreement(reader, corruption):
    model = build_tiny_model(feature_pred_config=FeaturePredictionConfig(head_layers=2))
    state = dict(model.state_dict())
    meta = _metadata(model)
    if corruption == "legacy_depth":
        meta["user_config"]["feature_prediction"].pop("head_layers")
    elif corruption == "wrong_depth":
        meta["user_config"]["feature_prediction"]["head_layers"] = 1
    elif corruption == "missing_block":
        state.pop("feature_pred_head.residual_blocks.1.c_proj.weight")
    elif corruption == "extra_block":
        state["feature_pred_head.residual_blocks.2.c_fc.weight"] = state[
            "feature_pred_head.residual_blocks.0.c_fc.weight"
        ]
    else:
        state.pop("feature_pred_head.bias")
    with pytest.raises((RuntimeError, ValueError), match="state_dict|head_layers|conflicting"):
        _restore(model, meta, state, reader)


def test_pretraining_config_builder_retains_depth():
    cfg = OmegaConf.create(
        {
            "depth": 2,
            "aspect_ratio": 32,
            "head_dim": 32,
            "max_seq_len": 16,
            "window_pattern": "L",
            "num_buckets": 2,
            "encoder": {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16},
            "cross_attention": {"frequency": 1, "n_head": 2, "n_kv_head": 2},
            "feature_prediction": _raw_config(head_layers=2),
        }
    )
    model = build_encdec_meta_model_from_training_cfg(cfg, base_vocab=32, n_sae_features=8)
    assert model.feature_pred_config is not None and model.feature_pred_config.head_layers == 2
    assert "feature_pred_head.residual_blocks.1.c_proj.weight" in model.state_dict()


@pytest.mark.parametrize("source_depth,requested_block,requested_depth", [
    (0, "mlp", 2), (2, "mlp", 2), (0, "transformer", 0),
])
def test_main_upgrades_after_load_before_optimizer_and_saves_resolved_depth(
    tmp_path, monkeypatch, source_depth, requested_block, requested_depth,
):
    with _patched_main_runtime(tmp_path, monkeypatch) as harness:
        source, _, meta = sft.load_checkpoint(str(tmp_path / "base_checkpoint"), None, "cpu")
        fp_model = build_tiny_model(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            n_layer=2,
            seq_len=64,
            xattn_frequency=1,
            feature_pred_config=FeaturePredictionConfig(head_layers=source_depth),
        )
        if source_depth:
            with t.no_grad():
                for block in fp_model.feature_pred_head.residual_blocks:
                    block.c_proj.weight.fill_(0.01)
        source.update(
            {
                key: value
                for key, value in fp_model.state_dict().items()
                if key.startswith("feature_pred_head.")
            }
        )
        meta["user_config"]["feature_prediction"] = _raw_config(head_layers=source_depth)
        expected_source = copy.deepcopy(source)
        monkeypatch.setattr(sft, "load_checkpoint", lambda *_args, **_kwargs: (source, None, meta))
        monkeypatch.setattr(sft.sys, "argv", [
            *sft.sys.argv,
            f"feature_prediction_block={requested_block}",
            f"feature_prediction_head_layers={requested_depth}",
        ])
        original_setup = sft._setup_sft_optimizer
        observed = []

        def setup(model, cfg):
            assert model.feature_pred_config.head_layers == requested_depth
            assert model.feature_pred_config.block.value == requested_block
            if requested_depth == 0:
                assert isinstance(model.feature_pred_head, Linear)
            for name, value in expected_source.items():
                assert t.equal(model.state_dict()[name], value), name
            optimizer = original_setup(model, cfg)
            head_ids = {id(parameter) for parameter in model.feature_pred_head.parameters()}
            grouped_ids = [
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ]
            assert head_ids.issubset(grouped_ids)
            assert len(grouped_ids) == len(set(grouped_ids))
            observed.append(model)
            return optimizer

        monkeypatch.setattr(sft, "_setup_sft_optimizer", setup)
        sft.main()
        assert len(observed) == 1
        assert harness.saved_meta
        for saved in harness.saved_meta:
            assert saved["feature_prediction_config"]["head_layers"] == requested_depth
            assert saved["feature_prediction_config"]["block"] == requested_block
            assert saved["user_config"]["feature_prediction"] == saved["feature_prediction_config"]


def test_main_headless_upgrade_rejected_before_dataset_build(tmp_path, monkeypatch):
    with _patched_main_runtime(tmp_path, monkeypatch):
        monkeypatch.setattr(sft.sys, "argv", [*sft.sys.argv, "feature_prediction_head_layers=2"])
        monkeypatch.setattr(
            sft,
            "build_task_mixtures",
            lambda *_args, **_kwargs: pytest.fail("headless upgrade reached dataset build"),
        )
        with pytest.raises(ValueError, match="headless checkpoint"):
            sft.main()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_actual_sft_optimizer_learns_every_block_and_resumes_exactly(
    monkeypatch, hydra_compose, depth, compact
):
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "training"
    cfg = hydra_compose(str(config_dir), "conditional_encdec_sft")
    t.manual_seed(1681)
    vocab = FeatureVocab(n_sae_features=8, original_ids=(1, 4, 7)) if compact else None
    model = build_tiny_model(
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(),
        feature_vocab=vocab,
    )
    model.feature_pred_head = ResidualFeaturePredictionHead(model.feature_pred_head, depth)
    model.feature_pred_config = replace(model.feature_pred_config, head_layers=depth)
    optimizer = sft._setup_sft_optimizer(model, cfg)
    assert all(parameter.requires_grad for parameter in model.parameters())
    grouped_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped_ids) == sorted(id(p) for p in model.parameters())
    head = model.feature_pred_head
    head_ids = {id(p) for p in head.parameters()}
    grouped_head = [
        (parameter, group)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in head_ids
    ]
    assert len(grouped_head) == len(head_ids)
    assert {id(parameter) for parameter, _group in grouped_head} == head_ids
    block_matrix_ids = {
        id(parameter) for parameter in head.residual_blocks.parameters() if parameter.ndim == 2
    }
    for parameter, group in grouped_head:
        if id(parameter) in block_matrix_ids:
            assert group["kind"] == OptimizerKind.MUON
            assert group["lr"] == pytest.approx(0.02)
        else:
            assert group["kind"] == OptimizerKind.ADAMW
            assert group["lr"] == pytest.approx(
                0.004 * (model.dec_config.n_embd / 768) ** -0.5
            )
    width = model.feature_prediction_width
    batch = make_batch([[2]], T=8, F=3, n_sae=width, seed=1681)
    if vocab is not None:
        batch.enc_fids = t.tensor(vocab.original_ids)[batch.enc_fids]
    targets = t.linspace(0.05, 0.95, 2 * width).reshape(1, 2, width)

    def backward(current, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        loss = current(
            batch.inputs,
            batch.targets,
            enc_feature_ids=batch.enc_fids,
            enc_magnitudes=batch.enc_mags,
            seg_map=batch.seg_map,
            enc_mask=batch.enc_mask,
            fp_targets=targets,
            seg_doc_id=batch.seg_doc_id,
        ).loss
        loss.backward()
        return loss.detach().clone()

    def step(current_optimizer):
        # Execute the real optimizer math eagerly; GPU tests qualify compilation.
        with t.compiler.set_stance("force_eager"):
            current_optimizer.step()

    original_inputs = []
    for block in head.residual_blocks:
        assert isinstance(block, MLP)
        original_inputs.append(block.c_fc.weight.detach().clone())
    for step_index in range(2):
        backward(model, optimizer)
        for block in head.residual_blocks:
            assert isinstance(block, MLP)
            for parameter in block.parameters():
                assert parameter.grad is not None
                assert t.isfinite(parameter.grad).all()
            assert block.c_fc.weight.grad is not None and block.c_proj.weight.grad is not None
            assert (t.count_nonzero(block.c_fc.weight.grad) == 0) == (step_index == 0)
            assert t.count_nonzero(block.c_proj.weight.grad) > 0
        step(optimizer)
    for block, initial in zip(head.residual_blocks, original_inputs, strict=True):
        assert isinstance(block, MLP)
        assert not t.equal(block.c_fc.weight, initial)
    state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    metadata = _metadata(model)
    uninterrupted_loss = backward(model, optimizer)
    step(optimizer)
    resumed = grpo._build_model(state, metadata, device="cpu")
    resumed_optimizer = sft._setup_sft_optimizer(resumed, cfg)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_loss = backward(resumed, resumed_optimizer)
    step(resumed_optimizer)
    assert t.equal(resumed_loss, uninterrupted_loss)
    for name, value in model.state_dict().items():
        assert t.equal(value, resumed.state_dict()[name]), name
    expected_optimizer = optimizer.state_dict()
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    t.testing.assert_close(actual_optimizer["state"], expected_optimizer["state"], rtol=0, atol=0)
