"""Checkpoint identities must describe the bytes actually admitted by RLVR."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.train import conditional_encdec_rlvr as trainer
from steering.conditional.encoder_decoder import rlvr_runtime as runtime
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.segmentation import SegmentMode
from tests.scripts.test_encdec_rlvr_runtime import _settings


class _ResumeModel(t.nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = t.nn.Parameter(t.tensor([weight]))
        self.lm_head = t.nn.Linear(1, 1, bias=False)
        self.feature_pred_config = None

    def get_device(self):
        return self.weight.device

    def assert_vocab_consistent(self):
        return None


def _resume_case(monkeypatch, tmp_path, *, rank_rng):
    output = tmp_path / "checkpoint"
    output.mkdir()
    source = _ResumeModel(2.0)
    source_optimizer = t.optim.SGD([source.weight], lr=0.1, momentum=0.9)
    source.weight.grad = t.tensor([3.0])
    source_optimizer.step()
    target = _ResumeModel(-1.0)
    target_optimizer = t.optim.SGD([target.weight], lr=0.1, momentum=0.9)
    target_optimizer.param_groups[0]["initial_lr"] = 0.1
    identity = {"sha256": "checkpoint-read-fixture"}
    t.save(source.state_dict(), output / "model_000001.pt")
    t.save(source_optimizer.state_dict(), output / "optim_000001_rank0.pt")
    metadata = {
        "step": 1,
        runtime.RLVR_META_KEY: {
            "completed_optimizer_step": 1,
            "resume_identity": identity,
            "rank_rng": rank_rng,
            "wandb": {},
        },
    }
    (output / "meta_000001.json").write_text(json.dumps(metadata))
    (output / "rlvr_complete_000001.json").write_text(
        json.dumps({"step": 1, "world_size": 1})
    )
    cfg = OmegaConf.create(
        {
            **_settings(
                global_prompts_per_step=2,
                group_size=2,
                num_iterations=1,
                eval_every=1,
                save_every=1,
                resume_from_step=1,
            ),
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "output_dir": str(output),
            "resume_checkpoint_dir": None,
            "wandb": {**_settings()["wandb"], "enabled": False},
        }
    )
    monkeypatch.setattr(
        trainer,
        "compute_init_for_checkpoint_resume",
        lambda *_args, **_kwargs: (True, 0, 0, 1, t.device("cpu")),
    )
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: None)
    monkeypatch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "sdpa")
    monkeypatch.setattr(
        trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        trainer, "load_rlvr_inputs", lambda *_args, **_kwargs: SimpleNamespace(model=target)
    )
    monkeypatch.setattr(
        trainer, "_build_optimizer", lambda *_args, **_kwargs: target_optimizer
    )
    monkeypatch.setattr(trainer, "DistMuonAdamW", t.optim.SGD)
    monkeypatch.setattr(trainer, "resume_identity", lambda *_args, **_kwargs: identity)
    return cfg, target, target_optimizer, source, source_optimizer, output


def _valid_rank_rng():
    local = {**runtime.capture_rank_rng(t.device("cpu")), "ddp_world_size": 1}
    return {"ddp_world_size": 1, "rank_states": [local]}


@pytest.mark.parametrize("replacement", ["model", "metadata"])
def test_base_checkpoint_replacement_during_load_is_rejected_before_model_build(
    monkeypatch, tmp_path, replacement,
):
    model_path = tmp_path / "model_000001.pt"
    meta_path = tmp_path / "meta_000001.json"
    t.save({"weight": t.tensor([1.0])}, model_path)
    metadata = {"step": 1}
    meta_path.write_text(json.dumps(metadata))
    native_load = trainer.load_checkpoint
    seen_weights = []

    def load_replaced(*args, **kwargs):
        if replacement == "model":
            new_path = tmp_path / "replacement.pt"
            t.save({"weight": t.tensor([2.0])}, new_path)
            new_path.replace(model_path)
        state, optimizer, loaded_meta = native_load(*args, **kwargs)
        seen_weights.append(state["weight"].item())
        if replacement == "metadata":
            meta_path.write_text(json.dumps({**metadata, "changed": True}))
        return state, optimizer, loaded_meta

    def build_model(*_args, **_kwargs):
        pytest.fail("replacement passed checkpoint admission")

    monkeypatch.setattr(trainer, "load_checkpoint", load_replaced)
    monkeypatch.setattr(trainer, "_build_model", build_model)
    monkeypatch.setattr(trainer, "resolve_checkpoint_sft_segment_mode", lambda *_args, **_kwargs: SegmentMode.turn_delimited_fixed_tokens)
    monkeypatch.setattr(trainer, "resolve_turn_self_cond_segment_size_rustbpe", lambda *_args, **_kwargs: 33)
    monkeypatch.setattr(trainer, "resolve_next_segment_conditioning_offset", lambda *_args: 0)

    with pytest.raises(ValueError, match=f"checkpoint {replacement} changed"):
        trainer.load_rlvr_inputs(
            SimpleNamespace(base_checkpoint_dir=str(tmp_path), model_step=1),
            device=t.device("cpu"), rank=0,
        )
    assert seen_weights == ([2.0] if replacement == "model" else [1.0])


@pytest.mark.parametrize("recorded_step", [2, True, "1", None])
def test_base_checkpoint_rejects_present_step_disagreement(monkeypatch, tmp_path, recorded_step):
    (tmp_path / "meta_000001.json").write_text(json.dumps({"step": recorded_step}))
    (tmp_path / "model_000001.pt").write_bytes(b"not yet loaded")

    def load(*_args, **_kwargs):
        pytest.fail("step disagreement must fail before loading model weights")

    monkeypatch.setattr(trainer, "load_checkpoint", load)
    with pytest.raises(SFTGRPOError, match="metadata step disagrees"):
        trainer.load_rlvr_inputs(
            SimpleNamespace(base_checkpoint_dir=str(tmp_path), model_step=1),
            device=t.device("cpu"), rank=0,
        )


def test_resume_admission_loads_stable_real_files(monkeypatch, tmp_path):
    cfg, target, target_optimizer, source, source_optimizer, _ = _resume_case(
        monkeypatch, tmp_path, rank_rng=_valid_rank_rng()
    )

    trainer.train(cfg)

    assert t.equal(target.weight, source.weight)
    assert t.equal(
        target_optimizer.state[target.weight]["momentum_buffer"],
        source_optimizer.state[source.weight]["momentum_buffer"],
    )


@pytest.mark.parametrize("replacement", ["model", "metadata", "optimizer"])
def test_resume_admission_rejects_replacement_before_state_application(
    monkeypatch, tmp_path, replacement,
):
    cfg, target, target_optimizer, _, _, output = _resume_case(
        monkeypatch, tmp_path, rank_rng=_valid_rank_rng()
    )
    native_load = trainer.load_checkpoint

    def load_replaced(*args, **kwargs):
        replacement_path = output / f"replacement-{replacement}"
        if replacement == "model":
            t.save(_ResumeModel(9.0).state_dict(), replacement_path)
            replacement_path.replace(output / "model_000001.pt")
        elif replacement == "metadata":
            metadata = json.loads((output / "meta_000001.json").read_text())
            replacement_path.write_text(json.dumps({**metadata, "replacement": True}))
            replacement_path.replace(output / "meta_000001.json")
        else:
            model = _ResumeModel(4.0)
            optimizer = t.optim.SGD([model.weight], lr=0.7, momentum=0.5)
            t.save(optimizer.state_dict(), replacement_path)
            replacement_path.replace(output / "optim_000001_rank0.pt")
        return native_load(*args, **kwargs)

    monkeypatch.setattr(trainer, "load_checkpoint", load_replaced)
    with pytest.raises(ValueError, match=f"checkpoint {replacement} changed"):
        trainer.train(cfg)

    assert target.weight.item() == -1.0
    assert not target_optimizer.state
    assert target_optimizer.param_groups[0]["lr"] == 0.1
    assert target_optimizer.param_groups[0]["momentum"] == 0.9


def test_resume_admission_rejects_loaded_metadata_disagreement(monkeypatch, tmp_path):
    cfg, target, target_optimizer, _, _, _ = _resume_case(
        monkeypatch, tmp_path, rank_rng=_valid_rank_rng()
    )
    native_load = trainer.load_checkpoint

    def load_disagreed(*args, **kwargs):
        state, optimizer, metadata = native_load(*args, **kwargs)
        return state, optimizer, {**metadata, "loaded-only": True}

    monkeypatch.setattr(trainer, "load_checkpoint", load_disagreed)
    with pytest.raises(SFTGRPOError, match="checkpoint metadata changed"):
        trainer.train(cfg)

    assert target.weight.item() == -1.0
    assert not target_optimizer.state
    assert target_optimizer.param_groups[0]["lr"] == 0.1
    assert target_optimizer.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("rank_rng", [None, pytest.param({}, id="legacy-flat")])
def test_resume_admission_rejects_missing_rank_rng_envelope(
    monkeypatch, tmp_path, rank_rng,
):
    cfg, target, target_optimizer, _, _, _ = _resume_case(
        monkeypatch, tmp_path, rank_rng=rank_rng
    )

    with pytest.raises(SFTGRPOError, match="rank_rng.*rank-state envelope"):
        trainer.train(cfg)

    assert target.weight.item() == -1.0
    assert not target_optimizer.state
