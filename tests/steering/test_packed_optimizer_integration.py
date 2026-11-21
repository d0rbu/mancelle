"""Packing admission and real encoder-decoder training behavior on CPU oracles."""

import copy
from dataclasses import asdict, replace

import pytest
import torch as t
from omegaconf import OmegaConf

import nanochat.flash_attention as flash_attention
import nanochat.optim as upstream
import steering.conditional.encoder_decoder.model as model_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.optim import MuonAdamW
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.encoder_decoder.config import FeaturePredictionBlock
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.model import ForwardResult
from steering.conditional.encoder_decoder.packed_optimizer import PackedMuonAdamW
from tests.scripts.test_encdec_sft_warmup import runtime as sft_runtime
from tests.steering.test_packed_encoder_decoder_optimizer import assert_equal
from tests.steering.test_residual_feature_head_integration import _batch, _model


@pytest.mark.parametrize("block", ["mlp", "transformer"])
@t.random.fork_rng()
def test_packing_preserves_model_outputs_gradients_and_evolving_updates(monkeypatch, block):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    t.manual_seed(491)
    reference = _model(depth=2, block=block, compact=True)
    # Exercise the expanded head beyond its function-preserving initialization.
    with t.no_grad():
        for parameter in reference.feature_pred_head.residual_blocks.parameters():
            parameter.normal_(0, 0.03)
    packed = copy.deepcopy(reference)
    identities = {name: id(parameter) for name, parameter in packed.named_parameters()}
    reference_optimizer = reference.setup_optimizer(weight_decay=0.1, fp_head_matrix_lr=0.013)
    packed_optimizer = packed.setup_optimizer(
        weight_decay=0.1,
        fp_head_matrix_lr=0.013,
        muon_parameter_packing=True,
    )
    assert type(reference_optimizer) is MuonAdamW
    assert type(packed_optimizer) is PackedMuonAdamW
    assert {name: id(parameter) for name, parameter in packed.named_parameters()} == identities
    assert_equal(reference.state_dict(), packed.state_dict())
    assert_equal(reference_optimizer.state_dict(), packed_optimizer.state_dict())
    batch = _batch(reference)
    with t.compiler.set_stance("force_eager"):
        for _ in range(3):
            reference_result = reference(**batch)
            packed_result = packed(**batch)
            assert isinstance(reference_result, ForwardResult)
            assert isinstance(packed_result, ForwardResult)
            assert_equal(reference_result, packed_result)
            reference_result.loss.backward()
            packed_result.loss.backward()
            for (name, before), (other_name, after) in zip(
                reference.named_parameters(),
                packed.named_parameters(),
                strict=True,
            ):
                assert name == other_name
                assert_equal(before.grad, after.grad)
            reference_optimizer.step()
            packed_optimizer.step()
            assert_equal(reference.state_dict(), packed.state_dict())
            assert_equal(reference_optimizer.state_dict(), packed_optimizer.state_dict())
            reference_optimizer.zero_grad(set_to_none=True)
            packed_optimizer.zero_grad(set_to_none=True)


def test_trainable_only_packing_retains_frozen_trunk_and_head_group_ownership():
    model = _model(depth=2, block="transformer")
    model.requires_grad_(False)
    model.feature_pred_head.requires_grad_(True)
    head_ids = {id(parameter) for parameter in model.feature_pred_head.parameters()}
    trunk_storage = {
        name: parameter.data_ptr()
        for name, parameter in model.named_parameters()
        if id(parameter) not in head_ids
    }
    optimizer = model.setup_optimizer(trainable_only=True, muon_parameter_packing=True)
    assert type(optimizer) is PackedMuonAdamW
    assert {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    } == head_ids
    assert {
        name: parameter.data_ptr()
        for name, parameter in model.named_parameters()
        if id(parameter) not in head_ids
    } == trunk_storage


def test_distributed_dispatch_binds_raw_model_before_training(monkeypatch):
    model = _model(depth=2)
    calls = []

    def factory(groups, *, owner):
        calls.append(owner)
        return MuonAdamW(groups)

    monkeypatch.setattr(model_module, "get_dist_info", lambda: (True, 0, 0, 2))
    monkeypatch.setattr(model_module, "DistPackedMuonAdamW", factory)
    optimizer = model.setup_optimizer(muon_parameter_packing=True)
    assert calls == [model]
    assert all(group["initial_lr"] == group["lr"] for group in optimizer.param_groups)


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_packing_flag_rejects_non_boolean_before_changing_storage(value):
    model = _model(depth=2)
    storage = [parameter.data_ptr() for parameter in model.parameters()]
    with pytest.raises(ValueError, match="muon_parameter_packing must be a boolean"):
        model.setup_optimizer(muon_parameter_packing=value)
    assert [parameter.data_ptr() for parameter in model.parameters()] == storage


def test_default_meta_optimizer_inventory_does_not_require_materialized_storage():
    model = _model(depth=2).to("meta")
    optimizer = model.setup_optimizer()
    assert type(optimizer) is MuonAdamW
    assert all(
        parameter.is_meta for group in optimizer.param_groups for parameter in group["params"]
    )


@pytest.mark.parametrize("packing", [False, True])
@t.random.fork_rng()
def test_sft_lazy_compile_checkpoint_and_cross_packing_resume(monkeypatch, tmp_path, packing):
    """Real SFT setup/save/load seams; CPU eager kernels and Dynamo eager backend."""
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    for name in ("muon_step_fused", "adamw_step_fused"):
        monkeypatch.setattr(upstream, name, getattr(upstream, name)._torchdynamo_orig_callable)
    t.manual_seed(491)
    base = _model(compact=True)
    raw = copy.deepcopy(base).to("meta")
    raw.to_empty(device="cpu")
    raw.init_buffers()
    raw.load_state_dict(copy.deepcopy(base.state_dict()), strict=True, assign=True)
    raw.assert_vocab_consistent()
    raw.feature_pred_head = ResidualFeaturePredictionHead(
        raw.feature_pred_head,
        2,
        FeaturePredictionBlock.TRANSFORMER,
        decoder_config=raw.dec_config,
    )
    raw.feature_pred_config = replace(
        raw.feature_pred_config, block=FeaturePredictionBlock.TRANSFORMER, head_layers=2
    )
    cfg = OmegaConf.create(
        {
            "unembedding_lr": 0.004,
            "embedding_lr": 0.3,
            "matrix_lr": 0.02,
            "fp_head_matrix_lr": 0.013,
            "weight_decay": 0.1,
            "encoder_embedding_lr": 0.2,
            "encoder_matrix_lr": 0.02,
            "xattn_matrix_lr": 0.02,
            "lr_multiplier": 0.25,
            "init_lr_frac": 0.5,
            "lr_warmup_steps": 2,
            "num_iterations": 3,
            "muon_parameter_packing": packing,
        }
    )
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def schedule(current, step):
        multiplier = sft.sft_warmup_flat_decay(
            step / cfg.num_iterations,
            completed_steps=step,
            warmup_steps=cfg.lr_warmup_steps,
        )
        assert multiplier == (0.5, 1.0, 1.0)[step]
        for group in current.param_groups:
            group["lr"] = group["initial_lr"] * multiplier

    # SFT creates its lazy wrapper before binding the optimizer to the raw model.
    compiled = t.compile(raw, backend=backend, dynamic=None)
    assert not graphs
    optimizer = sft._setup_sft_optimizer(raw, cfg)
    assert type(optimizer) is (PackedMuonAdamW if packing else MuonAdamW)
    batch = _batch(raw)
    compiled(**batch).loss.backward()
    schedule(optimizer, 0)
    optimizer.step()
    compiled.zero_grad(set_to_none=True)
    assert graphs
    meta = {
        "feature_prediction_config": asdict(raw.feature_pred_config),
        "user_config": OmegaConf.to_container(cfg),
    }
    sft._save_sft_checkpoint(
        dry_run=False,
        checkpoint_dir=str(tmp_path),
        step=1,
        model=raw,
        optimizer=optimizer,
        meta=meta,
        rank=0,
    )
    model_state, optimizer_state, loaded_meta = sft.load_checkpoint(
        str(tmp_path), 1, t.device("cpu"), load_optimizer=True, rank=0
    )
    sft._assert_resume_sft_lr_schedule_matches(
        loaded_meta,
        warmup_steps=cfg.lr_warmup_steps,
        num_iterations=cfg.num_iterations,
    )
    resumed = _model(depth=2, block="transformer", compact=True).to("meta")
    resumed.to_empty(device="cpu")
    resumed.init_buffers()
    resumed.load_state_dict(model_state, strict=True, assign=True)
    resumed.assert_vocab_consistent()
    assert sft._resolve_fp_config_from_meta(loaded_meta) == raw.feature_pred_config
    resumed_compiled = t.compile(resumed, backend=backend, dynamic=None)
    cfg.muon_parameter_packing = not packing
    resumed_optimizer = sft._setup_sft_optimizer(resumed, cfg)
    assert type(resumed_optimizer) is (MuonAdamW if packing else PackedMuonAdamW)
    resumed_optimizer.load_state_dict(optimizer_state)
    assert_equal(resumed.state_dict(), raw.state_dict())
    assert_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    for step in range(1, cfg.num_iterations):
        original_result = compiled(**batch)
        resumed_result = resumed_compiled(**batch)
        assert_equal(resumed_result, original_result)
        original_result.loss.backward()
        resumed_result.loss.backward()
        assert_equal(
            [parameter.grad for parameter in resumed.parameters()],
            [parameter.grad for parameter in raw.parameters()],
        )
        schedule(optimizer, step)
        schedule(resumed_optimizer, step)
        optimizer.step()
        resumed_optimizer.step()
        assert_equal(resumed.state_dict(), raw.state_dict())
        assert_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
        compiled.zero_grad(set_to_none=True)
        resumed_compiled.zero_grad(set_to_none=True)


@pytest.mark.parametrize("packing", [False, True])
@t.random.fork_rng()
def test_sft_main_warmup_checkpoint_resumes_with_opposite_packing(tmp_path, packing):
    options = (
        "num_iterations=6",
        "lr_warmup_steps=2",
        "total_batch_size=256",
        "save_every=1",
    )
    with sft_runtime(
        tmp_path / "full",
        head="transformer",
        overrides=(*options, f"muon_parameter_packing={str(packing).lower()}"),
    ) as full:
        sft.main()
    with sft_runtime(
        tmp_path / "resumed",
        head="transformer",
        resume_dir=full.checkpoint_dir,
        overrides=(
            *options,
            "resume_from_step=1",
            f"muon_parameter_packing={str(not packing).lower()}",
        ),
    ) as resumed:
        sft.main()
    assert type(full.optimizers[0]) is (PackedMuonAdamW if packing else MuonAdamW)
    assert type(resumed.optimizers[0]) is (MuonAdamW if packing else PackedMuonAdamW)
    packed_run = full if packing else resumed
    assert packed_run.optimizers[0]._packing._owner() is packed_run.models[0]
    assert len(full.rates) == 6
    assert resumed.rates == full.rates[1:]
    assert resumed.positions == full.positions[1:]
    assert_equal(resumed.models[0].state_dict(), full.models[0].state_dict())
    assert_equal(resumed.optimizers[0].state_dict(), full.optimizers[0].state_dict())
    assert_equal(
        {k: v for k, v in resumed.saves[-1]["loop_state"].items() if k != "total_training_time"},
        {k: v for k, v in full.saves[-1]["loop_state"].items() if k != "total_training_time"},
    )
