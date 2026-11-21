"""Independent contract tests for the encoder-decoder RLVR runtime utilities."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.tokenizer import get_tokenizer
from scripts.benchmark import benchmark_encdec_rlvr as benchmark
from scripts.eval import build_eval_context_cache as cache_builder
from scripts.eval import encdec_rlvr as evaluator
from scripts.preprocess import prepare_encdec_rlvr as prepare_script
from scripts.train import conditional_encdec_rlvr as trainer
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.checkpoint_conditioning import PRETRAIN_CONDITIONING_META_KEY
from steering.conditional.encoder_decoder import rlvr_runtime as runtime
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.frozen_decoder import StagingReport
from steering.conditional.encoder_decoder.rlvr_data import (
    RLVR_DATASET_REVISIONS,
    bundle_manifest_sha256,
)
from steering.conditional.encoder_decoder.rlvr_replay_cache import (
    HiddenReplayReport,
    HiddenReplaySettings,
    hidden_replay_settings,
)
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.turn_context_contract import (
    canonical_turn_payload,
)
from steering.segmentation import SegmentMode
from tests.scripts.test_conditional_encdec_sft_grpo_e2e import _save_base_checkpoint
from tests.steering.test_rlvr_data import _sources
from tests.steering.test_turn_context_cache_build import synthetic_shards
from tests.steering.test_turn_context_contract import checkpoint_fixture
from tests.torchrun_helpers import run_torchrun

_DYNAMO_CONFIG_KEYS = (
    "recompile_limit",
    "accumulated_recompile_limit",
    "fail_on_recompile_limit_hit",
    "suppress_errors",
)


class _MalformedRNGResumeModel(t.nn.Module):
    feature_pred_config = None

    def __init__(self):
        super().__init__()
        self.weight = t.nn.Parameter(t.tensor([1.0]))
        self.lm_head = t.nn.Linear(1, 1, bias=False)

    def get_device(self):
        return self.weight.device

    def assert_vocab_consistent(self):
        return None


@pytest.fixture(autouse=True)
def _restore_dynamo_config():
    original = {key: getattr(t._dynamo.config, key) for key in _DYNAMO_CONFIG_KEYS}
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(t._dynamo.config, key, value)


def _rows(count: int = 64) -> dict[str, tuple[str, ...]]:
    return {
        task: tuple(f"{task}:{index}" for index in range(count))
        for task in runtime.RLVR_TASKS
    }


def _settings(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "global_prompts_per_step": 8,
        "group_size": 4,
        "num_iterations": 20,
        "eval_every": 5,
        "save_every": 5,
        "validation_per_task": 32,
        "mc_max_tokens": 16,
        "gsm8k_max_tokens": 512,
        "seed": 42,
        "validation_seed": 314159,
        "temperature": 1.0,
        "grad_clip_max_norm": 1.0,
        "lr_multiplier": 1.0,
        "init_lr_frac": 0.05,
        "embedding_lr": 0.2,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.02,
        "encoder_embedding_lr": 0.2,
        "encoder_matrix_lr": 0.02,
        "xattn_matrix_lr": 0.02,
        "top_k": None,
        "weight_decay": 0.0,
        "lr_decay_start": 0.8,
        "resume_from_step": -1,
        "device_type": "cpu",
        "ntp": None,
        "kl_beta": None,
        "entropy_weight": None,
        "train_lm_head": None,
        "wandb": {
            "enabled": True,
            "mode": "online",
            "project": "test",
            "entity": None,
            "name": None,
            "new_run": False,
        },
    }
    result.update(overrides)
    return result


@pytest.fixture
def restore_cuda_sdpa_state():
    sdpa = {
        "cudnn": t.backends.cuda.cudnn_sdp_enabled(),
        "flash": t.backends.cuda.flash_sdp_enabled(),
        "math": t.backends.cuda.math_sdp_enabled(),
        "memory_efficient": t.backends.cuda.mem_efficient_sdp_enabled(),
        "math_reduction": t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }
    deterministic = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        t.backends.cuda.enable_cudnn_sdp(sdpa["cudnn"])
        t.backends.cuda.enable_flash_sdp(sdpa["flash"])
        t.backends.cuda.enable_math_sdp(sdpa["math"])
        t.backends.cuda.enable_mem_efficient_sdp(sdpa["memory_efficient"])
        t.backends.cuda.allow_fp16_bf16_reduction_math_sdp(sdpa["math_reduction"])
        t.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _resume_identity_loaded(device_type: str = "cpu") -> trainer.RLVRLoaded:
    return cast(
        trainer.RLVRLoaded,
        SimpleNamespace(
            model=SimpleNamespace(get_device=lambda: t.device(device_type)),
            base_identity={"step": 1116},
            bundle=SimpleNamespace(identity_sha256="bundle"),
            turn_contract=SimpleNamespace(semantic={"contract": "turn"}),
            cadence=33,
            policy=object(),
            compiler_config={"recompile_limit": 8193},
        ),
    )


def test_head_only_default_is_opt_in_and_invalid_scope_precedes_admission(monkeypatch):
    config_path = (
        Path(__file__).resolve().parents[2] / "configs/training/conditional_encdec_rlvr.yaml"
    )
    assert OmegaConf.load(config_path).train_feature_head_only is False
    monkeypatch.setattr(
        trainer,
        "compute_init_for_checkpoint_resume",
        lambda _device, *, resume_step: (True, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: None)
    monkeypatch.setattr(
        trainer,
        "load_rlvr_inputs",
        lambda *_args, **_kwargs: pytest.fail("invalid scope reached model/artifact admission"),
    )
    for invalid in (None, 0, "false"):
        with pytest.raises(SFTGRPOError, match="train_feature_head_only must be a boolean"):
            trainer.train(OmegaConf.create(_settings(train_feature_head_only=invalid)))


@pytest.mark.parametrize("invalid", [None, 0, "false"])
def test_benchmark_rejects_invalid_scope_before_output_or_admission(monkeypatch, tmp_path, invalid):
    output = tmp_path / "benchmark"
    cfg = OmegaConf.create({
        **_settings(train_feature_head_only=invalid),
        "compiler_reset": None, "warmup_steps": 0, "measure_steps": 1,
        "output_dir": str(output),
    })
    for name in ("compute_init", "load_rlvr_inputs"):
        monkeypatch.setattr(
            benchmark, name,
            lambda *_args, **_kwargs: pytest.fail("invalid benchmark scope reached runtime/model admission"),
        )
    with pytest.raises(SFTGRPOError, match="train_feature_head_only must be a boolean"):
        benchmark.benchmark(cfg)
    assert not output.exists()


def test_hidden_replay_defaults_are_opt_in():
    config_path = Path(__file__).resolve().parents[2] / "configs/training/conditional_encdec_rlvr.yaml"
    configured = OmegaConf.load(config_path)
    assert isinstance(configured, DictConfig)
    assert configured.hidden_replay_max_cache_bytes == 0
    assert configured.hidden_replay_min_free_bytes == 4 * 1024**3
    assert configured.hidden_replay_history_encoder is False
    assert configured.hidden_replay_prompt_encoder is False
    assert hidden_replay_settings(configured) == HiddenReplaySettings()
    assert hidden_replay_settings({}) == HiddenReplaySettings(max_cache_bytes=0, min_free_bytes=4 * 1024**3)


@pytest.mark.parametrize("field", ["hidden_replay_max_cache_bytes", "hidden_replay_min_free_bytes"])
@pytest.mark.parametrize("invalid", [True, False, -1, 1.5, "0", None])
def test_hidden_replay_invalid_settings_fail_before_admission(field, invalid):
    with pytest.raises(SFTGRPOError, match=field):
        runtime.validate_rlvr_settings(_settings(**{field: invalid}), world_size=1)


@pytest.mark.parametrize("invalid", [0, 1, -1, 1.5, "false", "true", None])
def test_history_encoder_requires_boolean_before_admission(invalid):
    with pytest.raises(SFTGRPOError, match="hidden_replay_history_encoder"):
        runtime.validate_rlvr_settings(_settings(hidden_replay_history_encoder=invalid), world_size=1)


@pytest.mark.parametrize("invalid", [0, 1, -1, 1.5, "false", "true", None])
def test_prompt_encoder_requires_boolean_before_admission(invalid):
    with pytest.raises(SFTGRPOError, match="hidden_replay_prompt_encoder"):
        runtime.validate_rlvr_settings(_settings(hidden_replay_prompt_encoder=invalid), world_size=1)


def test_prompt_encoder_requires_history_reuse_before_admission():
    with pytest.raises(SFTGRPOError, match="hidden_replay_history_encoder"):
        runtime.validate_rlvr_settings(_settings(hidden_replay_prompt_encoder=True), world_size=1)
    configured = _settings(hidden_replay_prompt_encoder=True, hidden_replay_history_encoder=True)
    runtime.validate_rlvr_settings(configured, world_size=1)
    assert hidden_replay_settings(configured).prompt_encoder is True


def test_prompt_encoder_setting_is_bound_to_resume_identity(monkeypatch):
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    cfg = _settings(hidden_replay_max_cache_bytes=16 * 1024**2,
                    hidden_replay_history_encoder=True, hidden_replay_prompt_encoder=False)
    kwargs: dict[str, Any] = {"world_size": 1, "attention_impl": "sdpa", "deterministic": True,
                              "parameter_manifest": ("feature_pred_head.weight",)}
    loaded = _resume_identity_loaded()
    recorded = trainer.resume_identity(loaded, cfg, **kwargs)
    assert recorded["config"]["hidden_replay_prompt_encoder"] is False
    expected = trainer.resume_identity(loaded, {**cfg, "hidden_replay_prompt_encoder": True}, **kwargs)
    assert expected["config"]["hidden_replay_prompt_encoder"] is True
    with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
        trainer.assert_grpo_resume_identity(expected, recorded, caller="RLVR")


@pytest.mark.parametrize("field,changed", [
    ("hidden_replay_max_cache_bytes", 1024), ("hidden_replay_min_free_bytes", 0),
    ("hidden_replay_history_encoder", True),
])
def test_hidden_replay_settings_are_bound_to_resume_identity(monkeypatch, field, changed):
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    cfg = _settings(hidden_replay_max_cache_bytes=0, hidden_replay_min_free_bytes=4 * 1024**3,
                    hidden_replay_history_encoder=False)
    kwargs: dict[str, Any] = {"world_size": 1, "attention_impl": "sdpa", "deterministic": True,
                              "parameter_manifest": ("feature_pred_head.weight",)}
    loaded = _resume_identity_loaded()
    recorded = trainer.resume_identity(loaded, cfg, **kwargs)
    assert recorded["config"]["hidden_replay_max_cache_bytes"] == 0
    assert recorded["config"]["hidden_replay_min_free_bytes"] == 4 * 1024**3
    assert recorded["config"]["hidden_replay_history_encoder"] is False
    expected = trainer.resume_identity(loaded, {**cfg, field: changed}, **kwargs)
    assert expected["config"][field] == changed
    with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
        trainer.assert_grpo_resume_identity(expected, recorded, caller="RLVR")


def test_hidden_replay_failed_worker_logging_preserves_native_exception(monkeypatch):
    logged = []
    monkeypatch.setattr(trainer.logger, "info", lambda _message, value: logged.append(value))
    report = HiddenReplayReport(max_owned_tensor_objects_alive_after_cleanup=1, groups_failed=1)
    settings = HiddenReplaySettings(max_cache_bytes=1024, min_free_bytes=0)
    with pytest.raises(RuntimeError, match="native behavior mismatch"):
        try:
            raise RuntimeError("native behavior mismatch")
        finally:
            trainer._log_hidden_replay_worker(report, settings, first_step=0, completed=0,
                                              completed_successfully=False)
    assert len(logged) == 1
    assert logged[0]["worker_completed_without_error"] is False
    assert logged[0]["max_owned_tensor_objects_alive_after_cleanup"] == 1
    with pytest.raises(RuntimeError, match="retained group state"):
        trainer._log_hidden_replay_worker(report, settings, first_step=0, completed=1)


def test_scope_identity_preserves_legacy_config_and_uses_the_manifest(monkeypatch):
    # Compare scopes under the same admitted source/runtime identity; the
    # pre-existing Git/source pins continue to reject cross-revision resumes.
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    loaded = _resume_identity_loaded()
    kwargs: dict[str, Any] = {"world_size": 1, "attention_impl": "sdpa", "deterministic": True}
    manifest = ("trunk.weight", "feature_pred_head.weight")
    legacy = trainer.resume_identity(loaded, _settings(), parameter_manifest=manifest, **kwargs)
    for flag in (False, True):
        actual = trainer.resume_identity(
            loaded,
            _settings(train_feature_head_only=flag),
            parameter_manifest=manifest,
            **kwargs,
        )
        assert actual == legacy  # The flag adds no duplicate identity field.
    assert "train_feature_head_only" not in legacy["config"]
    head_only = trainer.resume_identity(
        loaded,
        _settings(train_feature_head_only=True),
        parameter_manifest=("feature_pred_head.weight",),
        **kwargs,
    )
    for expected, recorded in ((legacy, head_only), (head_only, legacy)):
        with pytest.raises(
            SFTGRPOError, match="RLVR resume identity mismatch.*same training scope"
        ):
            trainer.assert_grpo_resume_identity(expected, recorded, caller="RLVR")


@pytest.mark.parametrize("field", ["git_commit", "source_sha256"])
def test_scope_compatibility_does_not_bypass_runtime_source_identity(monkeypatch, field):
    code = {"git_commit": "a" * 40, "source_sha256": "b" * 64}
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: dict(code))
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    loaded = _resume_identity_loaded()
    kwargs: dict[str, Any] = {
        "world_size": 1,
        "parameter_manifest": ("weight",),
        "attention_impl": "sdpa",
        "deterministic": True,
    }
    recorded = trainer.resume_identity(loaded, _settings(), **kwargs)
    code[field] = "c" * len(code[field])
    expected = trainer.resume_identity(loaded, _settings(train_feature_head_only=False), **kwargs)
    with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
        trainer.assert_grpo_resume_identity(expected, recorded, caller="RLVR")


@pytest.mark.parametrize(
    ("device", "capability", "active", "hard_deterministic", "kv_backend", "expected"),
    (
        ("cuda:3", (9, 0), "fa4", False, "sdpa", False),
        ("cuda:3", (9, 1), "fa4", False, "sdpa", True),
        ("cuda:3", (9, 0), "sdpa", False, "sdpa", True),
        ("cuda:3", (9, 0), "fa4", True, "sdpa", True),
        ("cuda:3", (9, 0), "fa4", False, "fa3", True),
        ("cpu", (9, 0), "fa4", False, "sdpa", True),
    ),
)
def test_prepare_rlvr_runtime_only_disables_cudnn_for_sm90_fa4_sdpa(
    monkeypatch,
    restore_cuda_sdpa_state,
    device: str,
    capability: tuple[int, int],
    active: str,
    hard_deterministic: bool,
    kv_backend: str,
    expected: bool,
) -> None:
    del restore_cuda_sdpa_state
    t.backends.cuda.enable_cudnn_sdp(True)
    t.use_deterministic_algorithms(hard_deterministic, warn_only=True)
    preserved = {
        "flash": t.backends.cuda.flash_sdp_enabled(),
        "math": t.backends.cuda.math_sdp_enabled(),
        "memory_efficient": t.backends.cuda.mem_efficient_sdp_enabled(),
        "math_reduction": t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "deterministic": t.are_deterministic_algorithms_enabled(),
        "warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
    }
    capability_devices: list[t.device] = []

    def get_capability(actual_device: t.device) -> tuple[int, int]:
        capability_devices.append(actual_device)
        return capability

    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_active_impl",
        lambda: trainer.native_flash_attention.FlashAttnImpl(active),
    )
    monkeypatch.setattr(trainer, "_resolved_token_kv_backend", lambda: kv_backend)

    trainer._prepare_rlvr_runtime(t.device(device))

    assert t.backends.cuda.cudnn_sdp_enabled() is expected
    assert {
        "flash": t.backends.cuda.flash_sdp_enabled(),
        "math": t.backends.cuda.math_sdp_enabled(),
        "memory_efficient": t.backends.cuda.mem_efficient_sdp_enabled(),
        "math_reduction": t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "deterministic": t.are_deterministic_algorithms_enabled(),
        "warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
    } == preserved
    assert capability_devices == ([] if device == "cpu" else [t.device(device)])


def test_load_rlvr_inputs_prepares_runtime_before_reading_config(monkeypatch) -> None:
    prepared: list[t.device] = []
    device = t.device("cpu")
    monkeypatch.setattr(trainer, "_prepare_rlvr_runtime", prepared.append)

    class ExpectedConfigRead(RuntimeError):
        pass

    class Config:
        @property
        def base_checkpoint_dir(self):
            assert prepared == [device]
            raise ExpectedConfigRead

    with pytest.raises(ExpectedConfigRead):
        trainer.load_rlvr_inputs(Config(), device=device, rank=0)


def test_resume_identity_tracks_resolved_token_kv_backend(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_override_impl",
        trainer.native_flash_attention.FlashAttnImpl.FA4,
    )
    loaded = _resume_identity_loaded()

    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", False)
    sdpa = trainer.resume_identity(
        loaded,
        _settings(),
        world_size=1,
        parameter_manifest=("weight",),
        attention_impl="fa4",
        deterministic=False,
    )
    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", True)
    fa3 = trainer.resume_identity(
        loaded,
        _settings(),
        world_size=1,
        parameter_manifest=("weight",),
        attention_impl="fa4",
        deterministic=False,
    )

    assert sdpa["runtime"]["token_kv_backend"] == "sdpa"
    assert fa3["runtime"]["token_kv_backend"] == "fa3"
    assert sdpa["runtime"]["token_kv_sdpa"] is None
    assert fa3["runtime"]["token_kv_sdpa"] is None
    assert sdpa["sha256"] != fa3["sha256"]
    for identity in (sdpa, fa3):
        identity.pop("sha256")
        identity["runtime"].pop("token_kv_backend")
    assert sdpa == fa3  # The prior identity could not detect this runtime change.


def test_resume_identity_ignores_unused_fa3_when_token_kv_is_forced_sdpa(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_override_impl",
        trainer.native_flash_attention.FlashAttnImpl.SDPA,
    )
    loaded = _resume_identity_loaded()
    identities = []
    for has_fa3 in (False, True):
        monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", has_fa3)
        identities.append(
            trainer.resume_identity(
                loaded,
                _settings(),
                world_size=1,
                parameter_manifest=("weight",),
                attention_impl="sdpa",
                deterministic=True,
            )
        )

    assert identities[0]["runtime"]["token_kv_backend"] == "sdpa"
    assert identities[0] == identities[1]


def test_resume_identity_records_effective_cuda_sdpa_flags(
    monkeypatch, restore_cuda_sdpa_state
) -> None:
    del restore_cuda_sdpa_state
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    monkeypatch.setattr(trainer.t.cuda, "get_device_name", lambda _device=None: "mock-cuda")
    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_override_impl",
        trainer.native_flash_attention.FlashAttnImpl.FA4,
    )
    t.backends.cuda.enable_cudnn_sdp(True)
    t.backends.cuda.enable_flash_sdp(True)
    t.backends.cuda.enable_math_sdp(False)
    t.backends.cuda.enable_mem_efficient_sdp(True)
    t.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    loaded = _resume_identity_loaded("cuda")

    enabled = trainer.resume_identity(
        loaded,
        _settings(),
        world_size=1,
        parameter_manifest=("weight",),
        attention_impl="fa4",
        deterministic=False,
    )
    t.backends.cuda.enable_cudnn_sdp(False)
    disabled = trainer.resume_identity(
        loaded,
        _settings(),
        world_size=1,
        parameter_manifest=("weight",),
        attention_impl="fa4",
        deterministic=False,
    )

    assert enabled["runtime"]["token_kv_sdpa"] == {
        "cudnn_enabled": True,
        "flash_enabled": True,
        "math_enabled": False,
        "memory_efficient_enabled": True,
        "fp16_bf16_math_reduction_allowed": True,
    }
    assert disabled["runtime"]["token_kv_sdpa"] == {
        **enabled["runtime"]["token_kv_sdpa"],
        "cudnn_enabled": False,
    }
    assert enabled["sha256"] != disabled["sha256"]


@pytest.mark.parametrize(("device_type", "has_fa3"), (("cpu", False), ("cuda", True)))
def test_resume_identity_ignores_sdpa_flags_without_cuda_sdpa(
    monkeypatch, restore_cuda_sdpa_state, device_type: str, has_fa3: bool
) -> None:
    del restore_cuda_sdpa_state
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    monkeypatch.setattr(trainer.t.cuda, "get_device_name", lambda _device=None: "mock-cuda")
    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", has_fa3)
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_override_impl",
        trainer.native_flash_attention.FlashAttnImpl.FA4,
    )
    loaded = _resume_identity_loaded(device_type)
    identities = []
    for cudnn_enabled in (True, False):
        t.backends.cuda.enable_cudnn_sdp(cudnn_enabled)
        identities.append(
            trainer.resume_identity(
                loaded,
                _settings(),
                world_size=1,
                parameter_manifest=("weight",),
                attention_impl="fa4",
                deterministic=False,
            )
        )

    assert identities[0]["runtime"]["token_kv_sdpa"] is None
    assert identities[0] == identities[1]


def test_stateless_prompt_selection_is_topology_and_global_rng_independent() -> None:
    rows = _rows()
    serial = [
        runtime.select_training_row(rows, seed=91, step=7, ordinal=ordinal)
        for ordinal in range(128)
    ]
    random.seed(123)
    for _ in range(100):
        random.random()
    even = {
        ordinal: runtime.select_training_row(rows, seed=91, step=7, ordinal=ordinal)
        for ordinal in range(0, 128, 2)
    }
    odd = {
        ordinal: runtime.select_training_row(rows, seed=91, step=7, ordinal=ordinal)
        for ordinal in range(1, 128, 2)
    }
    reconstructed = [even.get(ordinal, odd.get(ordinal)) for ordinal in range(128)]
    assert reconstructed == serial
    assert serial != [
        runtime.select_training_row(rows, seed=91, step=8, ordinal=ordinal)
        for ordinal in range(128)
    ]


def test_task_first_sampling_is_uniform_over_tasks_not_source_sizes() -> None:
    rows = _rows(1)
    rows["GSM8K"] = tuple(f"GSM8K:{index}" for index in range(1000))
    selected = [
        runtime.select_training_row(rows, seed=17, step=0, ordinal=ordinal)
        for ordinal in range(20_000)
    ]
    counts = {
        task: sum(row.startswith(f"{task}:") for row in selected)
        for task in runtime.RLVR_TASKS
    }
    assert all(abs(count - 5_000) < 250 for count in counts.values())


@pytest.mark.parametrize(
    "rows",
    [
        {task: (task,) for task in runtime.RLVR_TASKS[:-1]},
        {task: (() if task == "MMLU" else (task,)) for task in runtime.RLVR_TASKS},
        {**{task: (task,) for task in runtime.RLVR_TASKS}, "extra": ("extra",)},
    ],
)
def test_training_selection_rejects_incomplete_or_ambiguous_task_sets(rows) -> None:
    with pytest.raises(SFTGRPOError):
        runtime.select_training_row(rows, seed=1, step=0, ordinal=0)


def test_validation_panel_is_fixed_exact_size_and_without_replacement() -> None:
    rows = _rows()
    panel = runtime.validation_panel(rows, seed=33, per_task=32)
    random.seed(999)
    for _ in range(20):
        random.random()
    assert runtime.validation_panel(rows, seed=33, per_task=32) == panel
    assert len(panel) == 4 * 32
    for task_index, task in enumerate(runtime.RLVR_TASKS):
        task_panel = panel[task_index * 32 : (task_index + 1) * 32]
        assert len(set(task_panel)) == 32
        assert all(row.startswith(f"{task}:") for row in task_panel)
    assert runtime.validation_panel(rows, seed=34, per_task=32) != panel


def test_validation_panel_none_preserves_all_heldout_rows_in_task_order() -> None:
    rows = _rows(3)
    assert runtime.validation_panel(rows, seed=33, per_task=None) == tuple(
        row for task in runtime.RLVR_TASKS for row in rows[task]
    )


def test_validation_panel_rejects_missing_task_with_domain_error() -> None:
    rows = _rows()
    del rows["GSM8K"]
    with pytest.raises(SFTGRPOError):
        runtime.validation_panel(rows, seed=33, per_task=32)


def test_settings_accept_confirmed_budgets_and_two_gpu_geometry() -> None:
    runtime.validate_rlvr_settings(_settings(), world_size=2)
    runtime.validate_rlvr_settings(_settings(validation_per_task=None), world_size=2)


@pytest.mark.parametrize(
    ("overrides", "world_size"),
    [
        ({"mc_max_tokens": 0}, 2),
        ({"gsm8k_max_tokens": 0}, 2),
        ({"temperature": 0.0}, 2),
        ({"top_k": 0}, 2),
        ({"global_prompts_per_step": 7}, 2),
        ({"ntp": 0.0}, 2),
        ({"train_lm_head": False}, 2),
        ({"wandb": {**_settings()["wandb"], "enabled": 1}}, 2),
        ({"wandb": {**_settings()["wandb"], "new_run": 0}}, 2),
        ({"wandb": {**_settings()["wandb"], "mode": "disabled"}}, 2),
    ],
)
def test_settings_reject_contract_drift(overrides: dict[str, object], world_size: int) -> None:
    """Reject invalid geometry and any non-null adjacent objective."""

    with pytest.raises(SFTGRPOError):
        runtime.validate_rlvr_settings(_settings(**overrides), world_size=world_size)


def test_clip_happens_locally_before_optimizer_step(monkeypatch) -> None:
    model = t.nn.Linear(1, 1, bias=False)
    model.weight.grad = t.tensor([[2.0]])
    events: list[tuple[str, float]] = []

    def readiness(actual_model, *, expected_manifest) -> None:
        assert actual_model is model
        assert expected_manifest == ("weight",)
        events.append(("ready", float(model.weight.grad.item())))

    class Optimizer:
        def step(self) -> None:
            events.append(("step", float(model.weight.grad.item())))

    monkeypatch.setattr(runtime, "assert_grpo_gradients_ready", readiness)
    norm = runtime.clip_and_step(
        model,
        Optimizer(),
        parameter_manifest=("weight",),
        max_norm=1.0,
        device=t.device("cpu"),
    )
    assert norm == pytest.approx(2.0)
    assert events[0] == ("ready", 2.0)
    assert events[1][0] == "step"
    assert events[1][1] == pytest.approx(1.0)


def _gloo_local_mean_worker(
    rank: int,
    init_file: str,
    queue,
    fail_during_collection: bool,
) -> None:
    """Two-rank oracle for aligned failure and local-mean-before-average order."""

    t.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    optimizer_called = False
    try:
        model = t.nn.Linear(1, 1, bias=False)
        with t.no_grad():
            model.weight.zero_()

        class AveragingOptimizer:
            def step(self) -> None:
                nonlocal optimizer_called
                optimizer_called = True
                t.distributed.all_reduce(model.weight.grad)
                model.weight.grad.div_(2)
                with t.no_grad():
                    model.weight.sub_(model.weight.grad)

        coefficients = ((1.0, 3.0), (5.0, 7.0))[rank]
        with rank_synchronized_stage("test collection", device=t.device("cpu")):
            if fail_during_collection and rank == 0:
                raise RuntimeError("rank-zero collection failed")
            # This is the trainer's accepted mean-per-local-prompt scale:
            # each prompt contributes 1 / local_prompts before DDP averaging.
            for coefficient in coefficients:
                (model.weight.sum() * coefficient / len(coefficients)).backward()
        local_gradient = float(model.weight.grad.item())
        norm = runtime.clip_and_step(
            model,
            AveragingOptimizer(),
            parameter_manifest=("weight",),
            max_norm=10.0,
            device=t.device("cpu"),
        )
        queue.put(
            {
                "rank": rank,
                "status": "ok",
                "local_gradient": local_gradient,
                "norm": norm,
                "weight": float(model.weight.item()),
                "optimizer_called": optimizer_called,
            }
        )
    except Exception as exc:  # noqa: BLE001 - child reports the synchronized result
        queue.put(
            {
                "rank": rank,
                "status": "error",
                "error": str(exc),
                "optimizer_called": optimizer_called,
            }
        )
    finally:
        t.distributed.destroy_process_group()


@pytest.mark.parametrize("fail_during_collection", [False, True])
def test_two_rank_gloo_local_prompt_mean_and_aligned_failure(
    tmp_path: Path,
    fail_during_collection: bool,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "gloo_init")
    processes = [
        context.Process(
            target=_gloo_local_mean_worker,
            args=(rank, init_file, queue, fail_during_collection),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    results = sorted((queue.get(timeout=5) for _ in processes), key=lambda row: row["rank"])

    if fail_during_collection:
        assert [row["status"] for row in results] == ["error", "error"]
        assert all("rank-zero collection failed" in row["error"] for row in results)
        assert all(not row["optimizer_called"] for row in results)
    else:
        assert [row["local_gradient"] for row in results] == pytest.approx([2.0, 6.0])
        assert [row["norm"] for row in results] == pytest.approx([2.0, 6.0])
        assert [row["weight"] for row in results] == pytest.approx([-4.0, -4.0])
        assert all(row["optimizer_called"] for row in results)


def _gloo_uneven_validation_worker(rank: int, init_file: str, queue) -> None:
    t.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=3,
    )
    try:
        rows = tuple(
            SimpleNamespace(
                task=task,
                row_id=f"{task}:0",
                prompt_token_ids=(task_index + 1,),
                oracle={"answer": str(task_index % 2)},
            )
            for task_index, task in enumerate(runtime.RLVR_TASKS)
        )

        def group(_loaded, row, **_kwargs):
            reward = float(runtime.RLVR_TASKS.index(row.task) % 2)
            return SimpleNamespace(
                trajectories=(
                    SimpleNamespace(
                        reward=reward,
                        token_ids=(*row.prompt_token_ids, 99),
                        stop_reason="eos",
                    ),
                )
            )

        trainer._group = group
        trainer.grade_completion_result = lambda *_args, **_kwargs: SimpleNamespace(
            valid_format=True,
            parsed_answer="parsed",
        )
        loaded = SimpleNamespace(
            model=SimpleNamespace(get_device=lambda: t.device("cpu")),
            bundle=SimpleNamespace(validation=rows),
            tokenizer=SimpleNamespace(decode=lambda values: ":".join(map(str, values))),
        )
        metrics, samples = trainer.evaluate_rlvr(
            loaded,
            cfg=SimpleNamespace(validation_seed=7, validation_per_task=1),
            rank=rank,
            world_size=3,
        )
        queue.put({"rank": rank, "metrics": metrics, "samples": samples})
    finally:
        t.distributed.destroy_process_group()


def test_three_rank_gloo_uneven_validation_partition_aggregates_once(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "gloo_validation_init")
    processes = [
        context.Process(
            target=_gloo_uneven_validation_worker,
            args=(rank, init_file, queue),
        )
        for rank in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    results = sorted((queue.get(timeout=5) for _ in processes), key=lambda row: row["rank"])
    assert [len(result["samples"]) for result in results] == [4, 4, 4]
    assert all(result["metrics"] == results[0]["metrics"] for result in results)
    metrics = results[0]["metrics"]
    assert metrics["validation/accuracy"] == pytest.approx(0.5)
    for task in runtime.RLVR_TASKS:
        assert metrics[f"validation/{task}/count"] == 1
        assert metrics[f"validation/{task}/parse_rate"] == 1


def test_cpu_rng_state_survives_json_round_trip() -> None:
    random.seed(41)
    t.manual_seed(42)
    state = json.loads(json.dumps(runtime.capture_rank_rng(t.device("cpu"))))
    expected_python = [random.random() for _ in range(4)]
    expected_torch = t.rand(4)
    for _ in range(10):
        random.random()
        t.rand(1)
    runtime.restore_rank_rng(state, t.device("cpu"))
    assert [random.random() for _ in range(4)] == expected_python
    assert t.equal(t.rand(4), expected_torch)


def _gloo_malformed_rng_resume_worker(rank: int, init_file: str, output: str, queue) -> None:
    t.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    model = _MalformedRNGResumeModel()
    optimizer = t.optim.SGD([model.weight], lr=0.1)
    optimizer.param_groups[0]["initial_lr"] = 0.1
    cfg = OmegaConf.create(
        {
            **_settings(
                global_prompts_per_step=2,
                group_size=2,
                num_iterations=2,
                eval_every=1,
                save_every=1,
                resume_from_step=1,
            ),
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "output_dir": output,
            "resume_checkpoint_dir": None,
            "wandb": {**_settings()["wandb"], "enabled": False},
        }
    )
    rows = tuple(
        SimpleNamespace(task=task, row_id=f"{task}:0") for task in runtime.RLVR_TASKS
    )
    collection_started = False

    def fail_group(*_args, **_kwargs):
        nonlocal collection_started
        collection_started = True
        raise RuntimeError("rollout entered after a peer RNG restoration failure")

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                trainer,
                "compute_init_for_checkpoint_resume",
                lambda *_args, **_kwargs: (True, rank, rank, 2, t.device("cpu")),
            )
            patch.setattr(trainer, "compute_cleanup", lambda: None)
            patch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "sdpa")
            patch.setattr(
                trainer,
                "_configure_deterministic_algorithms",
                lambda **_kwargs: True,
            )
            patch.setattr(
                trainer,
                "load_rlvr_inputs",
                lambda *_args, **_kwargs: SimpleNamespace(
                    model=model, bundle=SimpleNamespace(train=rows)
                ),
            )
            patch.setattr(
                trainer, "_build_optimizer", lambda *_args, **_kwargs: optimizer
            )
            patch.setattr(trainer, "DistMuonAdamW", t.optim.SGD)
            patch.setattr(
                trainer,
                "resume_identity",
                lambda *_args, **_kwargs: {"sha256": "malformed-rng-fixture"},
            )
            patch.setattr(trainer, "_reset_step_compiler_state", lambda: None)
            patch.setattr(trainer, "_group", fail_group)
            try:
                trainer.train(cfg)
            except Exception as exc:  # noqa: BLE001 - child reports aligned failure
                queue.put(
                    {
                        "rank": rank,
                        "collection_started": collection_started,
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            else:
                queue.put(
                    {
                        "rank": rank,
                        "collection_started": collection_started,
                        "exception": None,
                        "message": "",
                    }
                )
    finally:
        if t.distributed.is_initialized():
            t.distributed.destroy_process_group()


@pytest.mark.skipif(
    not t.distributed.is_available() or not t.distributed.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_gloo_malformed_rank_rng_aborts_before_rollout(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    output.mkdir()
    source = _MalformedRNGResumeModel()
    optimizer = t.optim.SGD([source.weight], lr=0.1)
    t.save(source.state_dict(), output / "model_000001.pt")
    for rank in range(2):
        t.save(optimizer.state_dict(), output / f"optim_000001_rank{rank}.pt")
    rank_states = [
        {**runtime.capture_rank_rng(t.device("cpu")), "ddp_world_size": 2}
        for _ in range(2)
    ]
    rank_states[0]["torch_cpu"] = []
    metadata = {
        "step": 1,
        runtime.RLVR_META_KEY: {
            "completed_optimizer_step": 1,
            "resume_identity": {"sha256": "malformed-rng-fixture"},
            "rank_rng": {"ddp_world_size": 2, "rank_states": rank_states},
            "wandb": {},
        },
    }
    (output / "meta_000001.json").write_text(json.dumps(metadata))
    (output / "rlvr_complete_000001.json").write_text(
        json.dumps({"step": 1, "world_size": 2})
    )

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = str(tmp_path / "gloo_rng_init")
    processes = [
        context.Process(
            target=_gloo_malformed_rng_resume_worker,
            args=(rank, init_file, str(output), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=60) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    results.sort(key=lambda row: row["rank"])
    assert [row["exception"] for row in results] == [
        "AlignedRankFailure",
        "AlignedRankFailure",
    ]
    assert all(not row["collection_started"] for row in results)
    assert all("RLVR RNG restoration" in row["message"] for row in results)
    assert all("CPUGeneratorImplState" in row["message"] for row in results)


def _write_checkpoint_files(directory: Path, step: int, world_size: int) -> None:
    for path in runtime.checkpoint_files(directory, step, world_size):
        path.write_bytes(b"complete")


def test_checkpoint_requires_last_marker_and_every_rank_file(tmp_path: Path) -> None:
    step = 3
    (tmp_path / "rlvr_complete_000003.json").write_text(
        json.dumps({"step": step, "world_size": 2})
    )
    with pytest.raises(SFTGRPOError, match="incomplete"):
        runtime.assert_complete_checkpoint(tmp_path, step, 2)
    _write_checkpoint_files(tmp_path, step, 2)
    runtime.assert_complete_checkpoint(tmp_path, step, 2)
    (tmp_path / "optim_000003_rank1.pt").unlink()
    with pytest.raises(SFTGRPOError, match="incomplete"):
        runtime.assert_complete_checkpoint(tmp_path, step, 2)


def test_checkpoint_marker_is_topology_specific(tmp_path: Path) -> None:
    _write_checkpoint_files(tmp_path, 3, 2)
    (tmp_path / "rlvr_complete_000003.json").write_text(
        json.dumps({"step": 3, "world_size": 1})
    )
    with pytest.raises(SFTGRPOError, match="this topology"):
        runtime.assert_complete_checkpoint(tmp_path, 3, 2)


class _FakeRun:
    def __init__(self, *, step: int = 0, fail_logs: int = 0):
        self.step = step
        self.fail_logs = fail_logs
        self.logged: list[tuple[dict[str, object], int]] = []
        self.metric_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.finish_codes: list[int] = []

    def define_metric(self, *args, **kwargs) -> None:
        self.metric_calls.append((args, kwargs))

    def log(self, payload, *, step: int) -> None:
        if self.fail_logs:
            self.fail_logs -= 1
            raise RuntimeError("simulated network failure")
        self.logged.append((payload, step))
        self.step = step + 1

    def finish(self, *, exit_code: int = 0) -> None:
        self.finish_codes.append(exit_code)


class _FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


def _wandb_cfg(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "enabled": True,
        "mode": "online",
        "project": "project",
        "entity": "entity",
        "name": "name",
    }
    result.update(overrides)
    return result


def _install_fake_wandb(monkeypatch, run: _FakeRun, init_calls: list[dict[str, object]]) -> None:
    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    monkeypatch.setattr(runtime.wandb, "init", init)
    monkeypatch.setattr(runtime, "generate_id", lambda: "generated")
    monkeypatch.setattr(runtime.wandb, "Table", _FakeTable)


def test_wandb_rank_zero_reuses_id_and_deduplicates_completed_steps(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun(step=5)
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    resumed = {
        "id": "prior",
        "project": "project",
        "entity": "entity",
        "mode": "online",
        "last_logged_step": 3,
    }
    logger = runtime.RLVRWandb(
        _wandb_cfg(), rank=0, output_dir=tmp_path, run_config={"x": 1}, resumed=resumed
    )
    assert init_calls[0]["id"] == "prior"
    assert init_calls[0]["resume"] == "must"
    assert logger.state["last_logged_step"] == 4
    logger.log(4, {"reward": 0.0})
    logger.log(5, {"reward": 1.0})
    logger.log(5, {"reward": 2.0})
    assert [(payload["reward"], step) for payload, step in run.logged] == [(1.0, 5)]
    assert json.loads((tmp_path / "wandb_state.json").read_text())["last_logged_step"] == 5


def test_wandb_initialization_failure_finishes_run_as_failed(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run, [])
    monkeypatch.setattr(
        runtime,
        "configure_wandb_step_metric",
        lambda _run: (_ for _ in ()).throw(RuntimeError("metric setup failed")),
    )

    with pytest.raises(RuntimeError, match="metric setup failed"):
        runtime.RLVRWandb(
            _wandb_cfg(),
            rank=0,
            output_dir=tmp_path,
            run_config={"identity": "test"},
        )

    assert run.finish_codes == [1]


def test_wandb_new_run_records_parent_without_resuming(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun()
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    resumed = {
        "id": "prior",
        "project": "old-project",
        "entity": "old-entity",
        "mode": "offline",
        "last_logged_step": 99,
    }
    logger = runtime.RLVRWandb(
        _wandb_cfg(), rank=0, output_dir=tmp_path, run_config={}, resumed=resumed, new_run=True
    )
    assert init_calls[0]["id"] == "generated"
    assert init_calls[0]["resume"] == "never"
    assert init_calls[0]["config"]["parent_run_id"] == "prior"
    assert logger.state["last_logged_step"] == -1


def test_wandb_offline_reuses_identity_without_online_resume_protocol(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun(step=3)
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    resumed = {
        "id": "offline-prior",
        "project": "project",
        "entity": "entity",
        "mode": "offline",
        "last_logged_step": 1,
    }
    logger = runtime.RLVRWandb(
        _wandb_cfg(mode="offline"), rank=0, output_dir=tmp_path,
        run_config={}, resumed=resumed,
    )
    assert init_calls[0]["id"] == "offline-prior"
    assert init_calls[0]["resume"] is None
    assert logger.state["last_logged_step"] == 2


def test_wandb_resume_identity_drift_fails_before_initialization(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runtime.wandb,
        "init",
        lambda **_kwargs: pytest.fail("invalid resume must fail before wandb.init"),
    )
    resumed = {
        "id": "prior",
        "project": "different",
        "entity": "entity",
        "mode": "online",
        "last_logged_step": 2,
    }
    with pytest.raises(SFTGRPOError, match="project/entity/mode changed"):
        runtime.RLVRWandb(
            _wandb_cfg(), rank=0, output_dir=tmp_path,
            run_config={}, resumed=resumed,
        )
    assert not (tmp_path / "wandb_state.json").exists()


@pytest.mark.slow
def test_real_wandb_offline_sdk_resume_uses_persisted_dedup_floor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the installed SDK rather than the fake run-step model."""

    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = _wandb_cfg(mode="offline", entity=None, name="rlvr-offline-smoke")
    first = runtime.RLVRWandb(cfg, rank=0, output_dir=tmp_path, run_config={"phase": 1})
    first.log(0, {"reward": 0.0})
    first.log(1, {"reward": 1.0})
    resumed = dict(first.state)
    first.finish()

    assert resumed["last_logged_step"] == 1
    second = runtime.RLVRWandb(
        cfg,
        rank=0,
        output_dir=tmp_path,
        run_config={"phase": 2},
        resumed=resumed,
    )
    # Offline W&B does not implement cloud resume; the authenticated local
    # state remains the dedup authority even if the SDK counter restarts.
    assert second.state["id"] == resumed["id"]
    assert second.state["last_logged_step"] == 1
    second.log(1, {"reward": 99.0})
    assert second.state["last_logged_step"] == 1
    second.log(2, {"reward": 0.5})
    assert second.state["last_logged_step"] == 2
    second.finish()
    assert json.loads((tmp_path / "wandb_state.json").read_text())["last_logged_step"] == 2


def test_wandb_log_failure_does_not_advance_dedup_state(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun(fail_logs=1)
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    logger = runtime.RLVRWandb(
        _wandb_cfg(), rank=0, output_dir=tmp_path, run_config={}
    )
    with pytest.raises(RuntimeError, match="network failure"):
        logger.log(0, {"reward": 1.0})
    assert logger.state["last_logged_step"] == -1
    assert json.loads((tmp_path / "wandb_state.json").read_text())["last_logged_step"] == -1
    logger.log(0, {"reward": 1.0})
    assert len(run.logged) == 1


def test_wandb_sample_table_has_row_and_string_bounds(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun()
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    logger = runtime.RLVRWandb(
        _wandb_cfg(), rank=0, output_dir=tmp_path, run_config={}
    )
    sample = ["GSM8K", "id", "q" * 5000, "completion", "1", "1", 1, "eos"]
    logger.log(0, {}, [sample] * 8)
    table = run.logged[0][0]["validation/samples"]
    assert len(table.data) == 8
    assert len(table.data[0][2]) == 4096
    with pytest.raises(SFTGRPOError, match="at most eight"):
        logger.log(1, {}, [sample] * 9)
    with pytest.raises(SFTGRPOError, match="fixed table schema"):
        logger.log(1, {}, [["too", "short"]])
    assert len(run.logged) == 1


def test_wandb_metric_payload_cannot_override_canonical_step(monkeypatch, tmp_path: Path) -> None:
    run = _FakeRun()
    init_calls: list[dict[str, object]] = []
    _install_fake_wandb(monkeypatch, run, init_calls)
    logger = runtime.RLVRWandb(
        _wandb_cfg(), rank=0, output_dir=tmp_path, run_config={}
    )
    logger.log(7, {"step": 999, "reward": 1.0})
    payload, explicit_step = run.logged[0]
    assert explicit_step == 7
    assert payload["step"] == 7


def test_wandb_nonzero_rank_is_a_side_effect_free_dummy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runtime.wandb,
        "init",
        lambda **_kwargs: pytest.fail("nonzero rank must not initialize W&B"),
    )
    logger = runtime.RLVRWandb(
        _wandb_cfg(mode="invalid"), rank=1, output_dir=tmp_path, run_config={}
    )
    logger.log(0, {"reward": 1.0})
    logger.finish()
    assert not (tmp_path / "wandb_state.json").exists()


def test_group_uses_task_specific_confirmed_token_budgets(monkeypatch) -> None:
    observed: list[tuple[int, int]] = []

    def collect(*_args, group_size, max_tokens, **_kwargs):
        observed.append((group_size, max_tokens))
        return object()

    monkeypatch.setattr(trainer, "collect_rlvr_group", collect)
    loaded = SimpleNamespace(
        model=object(),
        engine=object(),
        policy=object(),
        source_tokenizer=object(),
        tokenizer=SimpleNamespace(decode=lambda _tokens: "completion"),
    )
    cfg = SimpleNamespace(gsm8k_max_tokens=512, mc_max_tokens=16, temperature=1.0, top_k=None)
    for task in ("GSM8K", "ARC-Challenge"):
        row = SimpleNamespace(task=task, prompt_token_ids=(1,), oracle={"answer": "A"})
        trainer._group(loaded, row, cfg=cfg, group_size=3, seed=7)
    assert observed == [(3, 512), (3, 16)]


@pytest.mark.parametrize("tasks", [runtime.RLVR_TASKS, runtime.RLVR_TASKS[:-1], ("MMLU",)])
def test_validation_evaluation_is_task_macro_averaged_and_samples_are_bounded(
    monkeypatch,
    tasks,
) -> None:
    rows = tuple(
        SimpleNamespace(
            task=task,
            row_id=f"{task}:{index}",
            prompt_token_ids=(task_index + 1, index + 1),
            oracle={"answer": str(task_index % 2)},
        )
        for task_index, task in enumerate(runtime.RLVR_TASKS)
        for index in range(3)
    )
    group_calls: list[tuple[str, int]] = []

    def group(_loaded, row, *, cfg, group_size, seed, replay_cache=None):
        assert replay_cache is None
        del cfg, seed
        group_calls.append((row.row_id, group_size))
        reward = float(runtime.RLVR_TASKS.index(row.task) % 2)
        trajectory = SimpleNamespace(
            reward=reward,
            token_ids=(*row.prompt_token_ids, 99),
            stop_reason="eos",
        )
        return SimpleNamespace(trajectories=(trajectory,))

    monkeypatch.setattr(trainer, "_group", group)
    monkeypatch.setattr(
        trainer,
        "grade_completion_result",
        lambda *_args, **_kwargs: SimpleNamespace(valid_format=True, parsed_answer="parsed"),
    )
    loaded = SimpleNamespace(
        model=SimpleNamespace(get_device=lambda: t.device("cpu")),
        bundle=SimpleNamespace(validation=rows),
        tokenizer=SimpleNamespace(decode=lambda values: ":".join(map(str, values))),
    )
    cfg = SimpleNamespace(validation_seed=19, validation_per_task=2, tasks=tasks)
    metrics, samples = trainer.evaluate_rlvr(loaded, cfg=cfg, rank=0, world_size=1)

    assert len(group_calls) == 8
    assert all(group_size == 1 for _, group_size in group_calls)
    assert len(samples) == 8
    assert len({(row[0], row[1]) for row in samples}) == 8
    assert all(len(row) == 8 for row in samples)
    expected = {
        task: float(index % 2) for index, task in enumerate(runtime.RLVR_TASKS)
    }
    for task in runtime.RLVR_TASKS:
        assert metrics[f"validation/{task}/accuracy"] == expected[task]
        assert metrics[f"validation/{task}/parse_rate"] == 1
        assert metrics[f"validation/{task}/count"] == 2
    assert metrics["validation/accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "turn_contract",
        "cadence",
        "feature_action",
        "objective",
        "generation",
        "base",
        "bundle",
        "seed",
        "sha256",
        "runtime",
        "head_legacy",
        "head_missing",
        "head_conflicting",
        "head_block",
        "head_depth",
        "checkpoint_model_during_load",
        "checkpoint_metadata_during_load",
        "checkpoint_model_during_hash",
        "checkpoint_model_before_publication",
    ],
)
@pytest.mark.parametrize("head_only", [False, True])
def test_standalone_evaluator_admits_complete_rlvr_identity_and_publishes_once(
    monkeypatch,
    tmp_path: Path,
    mutation: str | None,
    head_only: bool,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    output = tmp_path / "evaluation.json"
    step = 4
    base_identity = {
        "step": 1116,
        "model_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
    }
    config_identity = {
        "seed": 42,
        "validation_seed": 314159,
        "temperature": 1.0,
        "top_k": None,
        "mc_max_tokens": 16,
        "gsm8k_max_tokens": 512,
        "tasks": ["ARC-Easy", "ARC-Challenge", "MMLU"],
    }
    compiler_identity = {
        "derived_minimum": 8193,
        "window_variants": 2,
        "sequence_len": 2048,
        "recompile_limit": 8193,
        "accumulated_recompile_limit": 8193,
        "fail_on_recompile_limit_hit": True,
        "suppress_errors": False,
    }
    identity: dict[str, object] = {
        "base_checkpoint": base_identity,
        "bundle": "c" * 64,
        "turn_contract": {"contract": "expected"},
        "cadence": 33,
        "feature_action": {"action": "expected"},
        "objective": runtime.RLVR_OBJECTIVE_VERSION,
        "generation": runtime.RLVR_GENERATION_VERSION,
        "config": config_identity,
        "world_size": 2,
        "parameter_manifest": ["feature_pred_head.weight"] if head_only else ["trunk.weight", "feature_pred_head.weight"],
        "runtime": {
            "device": "cuda",
            "attention": "fa4",
            "compiler": compiler_identity,
        },
    }
    if mutation == "turn_contract":
        identity["turn_contract"] = {"contract": "wrong"}
    elif mutation == "cadence":
        identity["cadence"] = 32
    elif mutation == "feature_action":
        identity["feature_action"] = {"action": "wrong"}
    elif mutation == "objective":
        identity["objective"] = "wrong"
    elif mutation == "generation":
        identity["generation"] = "canonical_full_prefix_turn_calculator_v1"
    elif mutation == "base":
        identity["base_checkpoint"] = {**base_identity, "model_sha256": "f" * 64}
    elif mutation == "bundle":
        identity["bundle"] = "f" * 64
    elif mutation == "seed":
        identity["config"] = {**config_identity, "seed": 99}
    identity["sha256"] = hashlib.sha256(canonical_turn_payload(identity)).hexdigest()
    if mutation == "sha256":
        identity["sha256"] = "f" * 64
    elif mutation == "runtime":
        identity["runtime"] = {"device": "cpu", "attention": "sdpa"}
    head_config = {"block": "transformer", "head_layers": 0}
    head_metadata = {
        "user_config": {"feature_prediction": dict(head_config)},
        "feature_prediction_config": dict(head_config),
    }
    if mutation == "head_missing":
        head_metadata = {}
    elif mutation == "head_conflicting":
        head_metadata["feature_prediction_config"]["head_layers"] = 1
    elif mutation in {"head_legacy", "head_block", "head_depth"}:
        for raw in (
            head_metadata["user_config"]["feature_prediction"],
            head_metadata["feature_prediction_config"],
        ):
            if mutation == "head_legacy":
                raw.pop("block")
                raw["mlp_layers"] = raw.pop("head_layers")
            elif mutation == "head_block":
                raw["block"] = "mlp"
            else:
                raw["head_layers"] = 1
    for path in runtime.checkpoint_files(checkpoint_dir, step, 2):
        path.write_bytes(b"checkpoint")
    (checkpoint_dir / f"meta_{step:06d}.json").write_text(
        json.dumps(
            {
                **head_metadata,
                runtime.RLVR_META_KEY: {
                    "completed_optimizer_step": step,
                    "resume_identity": identity,
                }
            }
        )
    )
    (checkpoint_dir / f"rlvr_complete_{step:06d}.json").write_text(
        json.dumps({"step": step, "world_size": 2})
    )
    cfg = OmegaConf.create(
        {
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "rlvr_checkpoint_dir": str(checkpoint_dir),
            "rlvr_step": step,
            "output_path": str(output),
            "device_type": "cpu",
            **config_identity,
            "validation_per_task": 32,
        }
    )

    class Model:
        feature_pred_config = FeaturePredictionConfig(block=FeaturePredictionBlock.TRANSFORMER, head_layers=0)

        def __init__(self):
            self.loaded = False

        def load_state_dict(self, state, *, strict, assign):
            assert state == {"weight": "trained"}
            assert strict and not assign
            self.loaded = True

        def assert_vocab_consistent(self):
            assert self.loaded

    stable_calls: list[int] = []
    model = Model()
    loaded = SimpleNamespace(
        model=model,
        base_identity=base_identity,
        bundle=SimpleNamespace(identity_sha256="c" * 64),
        turn_contract=SimpleNamespace(semantic={"contract": "expected"}),
        cadence=33,
        policy=object(),
        # Evaluation may inherit a higher local compiler ceiling than training.
        compiler_config={
            **compiler_identity,
            "recompile_limit": 100_000,
            "accumulated_recompile_limit": 100_000,
        },
        cache_snapshot=SimpleNamespace(assert_stable=lambda: stable_calls.append(1)),
    )
    cleanup_calls: list[int] = []
    evaluation_calls: list[int] = []
    monkeypatch.setattr(evaluator, "compute_init", lambda _device: (True, 0, 0, 1, "cpu"))
    monkeypatch.setattr(evaluator, "compute_cleanup", lambda: cleanup_calls.append(1))
    monkeypatch.setattr(evaluator, "_assert_attention_impl", lambda **_kwargs: "sdpa")
    monkeypatch.setattr(evaluator, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(evaluator, "load_rlvr_inputs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(evaluator, "feature_action_identity", lambda _policy: {"action": "expected"})
    def load_checkpoint(*_args, **_kwargs):
        metadata = json.loads((checkpoint_dir / f"meta_{step:06d}.json").read_text())
        if mutation == "checkpoint_model_during_load":
            replacement = checkpoint_dir / "replacement.pt"
            replacement.write_bytes(b"replacement model")
            replacement.replace(checkpoint_dir / f"model_{step:06d}.pt")
        elif mutation == "checkpoint_metadata_during_load":
            (checkpoint_dir / f"meta_{step:06d}.json").write_text("{}")
        return {"weight": "trained"}, None, metadata

    monkeypatch.setattr(evaluator, "load_checkpoint", load_checkpoint)
    if mutation == "checkpoint_model_during_hash":
        original_hash = evaluator._sha256_file

        def hash_then_replace(path):
            digest = original_hash(path)
            path.write_bytes(b"replacement after hashing")
            return digest

        monkeypatch.setattr(evaluator, "_sha256_file", hash_then_replace)

    def evaluate(*_args, **_kwargs):
        evaluation_calls.append(1)
        if mutation == "checkpoint_model_before_publication":
            (checkpoint_dir / f"model_{step:06d}.pt").write_bytes(b"replacement during evaluation")
        return {"validation/accuracy": 0.75}, [["sample"]]

    monkeypatch.setattr(evaluator, "evaluate_rlvr", evaluate)
    if mutation is not None and mutation.startswith("head_"):
        with pytest.raises(ValueError, match="removed|presence|conflicting|does not match"):
            evaluator.evaluate(cfg)
        assert not model.loaded and not evaluation_calls and not output.exists()
        assert cleanup_calls == [1]
        return
    if mutation is not None and mutation.startswith("checkpoint_"):
        with pytest.raises(ValueError, match="checkpoint (model|metadata) changed"):
            evaluator.evaluate(cfg)
        assert not output.exists()
        assert evaluation_calls == ([1] if mutation.endswith("before_publication") else [])
        assert cleanup_calls == [1]
        return
    if mutation is not None:
        expected = (
            "base checkpoint or data bundle"
            if mutation in {"base", "bundle"}
            else "resume identity digest"
            if mutation in {"sha256", "runtime"}
            else f"{mutation} disagrees"
        )
        with pytest.raises(SFTGRPOError, match=expected):
            evaluator.evaluate(cfg)
        assert not output.exists()
        assert not evaluation_calls
        assert not model.loaded
    else:
        evaluator.evaluate(cfg)
        published = json.loads(output.read_text())
        assert published["checkpoint"]["step"] == step
        assert published["checkpoint"]["resume_identity_sha256"] == identity["sha256"]
        assert published["bundle_sha256"] == "c" * 64
        assert published["metrics"] == {"validation/accuracy": 0.75}
        assert evaluation_calls == [1]
        assert stable_calls == [1]
        assert model.loaded
    assert cleanup_calls == [1]


@pytest.mark.parametrize("orphan_rlvr_step", [False, True])
def test_standalone_evaluator_base_checkpoint_path_is_explicit_and_immutable(
    monkeypatch,
    tmp_path: Path,
    orphan_rlvr_step: bool,
) -> None:
    output = tmp_path / "base-evaluation.json"
    base_identity = {
        "step": 1116,
        "model_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
    }
    loaded = SimpleNamespace(
        base_identity=base_identity,
        bundle=SimpleNamespace(identity_sha256="c" * 64),
        cache_snapshot=SimpleNamespace(assert_stable=lambda: None),
    )
    cleanup_calls: list[int] = []
    evaluation_calls: list[int] = []
    cfg = OmegaConf.create(
        {
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "rlvr_checkpoint_dir": None,
            "rlvr_step": 4 if orphan_rlvr_step else None,
            "output_path": str(output),
            "device_type": "cpu",
            "seed": 42,
            "validation_seed": 314159,
            "temperature": 1.0,
            "top_k": None,
            "mc_max_tokens": 16,
            "gsm8k_max_tokens": 512,
            "validation_per_task": 32,
        }
    )

    monkeypatch.setattr(evaluator, "compute_init", lambda _device: (True, 0, 0, 1, "cpu"))
    monkeypatch.setattr(evaluator, "compute_cleanup", lambda: cleanup_calls.append(1))
    monkeypatch.setattr(evaluator, "_assert_attention_impl", lambda **_kwargs: "sdpa")
    monkeypatch.setattr(evaluator, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(evaluator, "load_rlvr_inputs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        evaluator,
        "load_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load RLVR")),
    )

    def evaluate(*_args, **_kwargs):
        evaluation_calls.append(1)
        return {"validation/accuracy": 0.25}, [["base-sample"]]

    monkeypatch.setattr(evaluator, "evaluate_rlvr", evaluate)
    if orphan_rlvr_step:
        with pytest.raises(SFTGRPOError, match="rlvr_step requires"):
            evaluator.evaluate(cfg)
        assert not output.exists()
        assert not evaluation_calls
    else:
        evaluator.evaluate(cfg)
        published = json.loads(output.read_text())
        assert published["checkpoint"] == base_identity
        assert published["bundle_sha256"] == "c" * 64
        assert published["metrics"] == {"validation/accuracy": 0.25}
        assert evaluation_calls == [1]
    assert cleanup_calls == [1]


@pytest.mark.parametrize("tasks", [None, runtime.RLVR_TASKS[:-1], ("MMLU",), runtime.RLVR_TASKS[::-1]])
@pytest.mark.parametrize("unparseable", [(), ("GSM8K",), runtime.RLVR_TASKS])
def test_one_step_trainer_stage_integration_uses_prompt_mean_and_commits_checkpoint(
    monkeypatch,
    tmp_path: Path,
    tasks,
    unparseable,
) -> None:
    """Exercise orchestration seams without loading a model, cache, network, or GPU."""

    events: list[object] = []
    output = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            **_settings(
                global_prompts_per_step=2,
                group_size=2,
                num_iterations=1,
                eval_every=1,
                save_every=1,
            ),
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "output_dir": str(output),
            "resume_checkpoint_dir": None,
            "wandb": {
                "enabled": True,
                "mode": "online",
                "project": "test",
                "entity": None,
                "name": None,
                "new_run": False,
            },
        }
    )
    if tasks is not None:
        cfg.tasks = list(tasks)
    expected_tasks = runtime.RLVR_TASKS if tasks is None else tasks
    warnings: list[str] = []
    monkeypatch.setattr(trainer.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))

    class Model(t.nn.Module):
        feature_pred_config = None

        def get_device(self):
            return t.device("cpu")

    class Optimizer:
        def __init__(self):
            self.param_groups = [{"lr": 1.0, "initial_lr": 1.0}]
            self.steps = 0

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            events.append("zero_grad")

        def step(self) -> None:
            self.steps += 1
            events.append("optimizer_step")

        def state_dict(self):
            return {"steps": self.steps}

    class Snapshot:
        def assert_stable(self) -> None:
            events.append("cache_stable")

    rows = tuple(
        SimpleNamespace(task=task, row_id=f"{task}:0", prompt_token_ids=(1, 2))
        for task in runtime.RLVR_TASKS
    )
    loaded = SimpleNamespace(
        model=Model(),
        bundle=SimpleNamespace(train=rows, validation=rows),
        cache_snapshot=Snapshot(),
        base_meta={"source": "base"},
        policy=object(),
        tokenizer=SimpleNamespace(decode=lambda tokens: "decoded:" + ",".join(map(str, tokens))),
    )
    optimizer = Optimizer()

    class Telemetry:
        instances: list[Telemetry] = []

        def __init__(self, *_args, **_kwargs):
            self.state = {"id": "run", "last_logged_step": -1}
            self.logs: list[tuple[int, dict[str, object], list[object]]] = []
            self.finish_codes: list[int] = []
            self.instances.append(self)

        def log(self, step, metrics, samples=()):
            self.logs.append((step, dict(metrics), list(samples)))
            self.state["last_logged_step"] = step

        def finish(self, *, exit_code: int = 0):
            self.finish_codes.append(exit_code)

    def fake_group(_loaded, row, *, cfg, group_size, seed, replay_cache=None):
        assert replay_cache is None
        del cfg, seed
        events.append(("collect", row.row_id, group_size))
        action = SimpleNamespace(
            policy_trace=SimpleNamespace(positive_rows=(1,), negative_rows=(2,)),
        )
        trajectory = lambda reward: SimpleNamespace(  # noqa: E731
            reward=reward,
            token_ids=(*row.prompt_token_ids, 3),
            sampled_mask=(0, 0, 1),
            live_actions=(action,),
            actions=(action,),
            stop_reason="eos",
        )
        return SimpleNamespace(
            trajectories=(trajectory(1.0), trajectory(0.0)),
            advantages=(1.0, -1.0),
        )

    loss_scales: list[float] = []

    def fake_backward(_model, _group, *, policy_config, loss_scale, train_feature_head_only, replay_cache=None):
        assert replay_cache is None
        del policy_config
        assert train_feature_head_only is False
        loss_scales.append(loss_scale)
        events.append("backward")
        return SimpleNamespace(loss=0.25, ratio_min=1.0, ratio_max=1.0, live_decisions=2)

    saved_meta: list[dict[str, object]] = []

    def fake_save(directory, step, model_data, optimizer_data, meta_data, *, rank):
        del model_data, optimizer_data
        assert rank == 0
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for path in runtime.checkpoint_files(directory, step, 1):
            path.write_bytes(b"complete")
        saved_meta.append(meta_data)
        events.append("save")

    def fake_clip(_model, actual_optimizer, **_kwargs):
        events.append("clip")
        actual_optimizer.step()
        return 0.5

    monkeypatch.setattr(
        trainer,
        "compute_init_for_checkpoint_resume",
        lambda _device, *, resume_step: (True, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "sdpa")
    monkeypatch.setattr(trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(trainer, "load_rlvr_inputs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(trainer, "configure_grpo_trainable_parameters", lambda *_args, **_kwargs: ("weight",))
    monkeypatch.setattr(trainer, "_build_optimizer", lambda *_args, **_kwargs: optimizer)
    monkeypatch.setattr(trainer, "DistMuonAdamW", Optimizer)
    monkeypatch.setattr(trainer, "resume_identity", lambda *_args, **_kwargs: {"sha256": "identity"})
    monkeypatch.setattr(trainer, "RLVRWandb", Telemetry)
    validation_metrics = {"validation/accuracy": 0.5}
    for task in runtime.RLVR_TASKS:
        validation_metrics[f"validation/{task}/parse_rate"] = int(task not in unparseable)
        validation_metrics[f"validation/{task}/count"] = 2
    monkeypatch.setattr(trainer, "evaluate_rlvr", lambda *_args, **_kwargs: (dict(validation_metrics), []))
    monkeypatch.setattr(trainer, "_group", fake_group)
    monkeypatch.setattr(
        trainer,
        "grade_completion_result",
        lambda *_args, **_kwargs: SimpleNamespace(valid_format=True),
    )
    monkeypatch.setattr(trainer, "backward_rlvr_group", fake_backward)
    monkeypatch.setattr(trainer, "clip_and_step", fake_clip)
    monkeypatch.setattr(trainer, "_reset_step_compiler_state", lambda: events.append("reset"))
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save)

    trainer.train(cfg)

    assert loss_scales == [0.5, 0.5]
    assert events.index("clip") > max(i for i, event in enumerate(events) if event == "backward")
    assert events.index("optimizer_step") > events.index("clip")
    assert optimizer.steps == 1
    assert [entry[0] for entry in Telemetry.instances[0].logs] == [0, 1]
    step_metrics = Telemetry.instances[0].logs[-1][1]
    assert step_metrics["train/reward"] == pytest.approx(0.5)
    assert step_metrics["train/reward_std"] == pytest.approx(0.5)
    assert step_metrics["train/trajectories"] == 4
    assert step_metrics["train/groups"] == 2
    assert step_metrics["train/zero_variance_groups"] == 0
    assert step_metrics["train/generated_tokens"] == 4
    assert step_metrics["train/forced_tokens"] == 0
    assert step_metrics["train/live_actions"] == 4
    assert step_metrics["train/unused_actions"] == 0
    assert step_metrics["train/positive_features"] == 4
    assert step_metrics["train/negative_features"] == 4
    assert step_metrics["train/parse_rate"] == 1
    assert step_metrics["train/stop_eos"] == 4
    assert step_metrics["train/policy_loss"] == pytest.approx(0.25)
    assert step_metrics["train/ratio_min"] == 1
    assert step_metrics["train/ratio_max"] == 1
    assert step_metrics["train/gradient_norm_max_local"] == pytest.approx(0.5)
    collected = [cast(str, event[1]).split(":")[0] for event in events if isinstance(event, tuple) and event[0] == "collect"]
    assert set(collected) <= set(expected_tasks)
    for task in runtime.RLVR_TASKS:
        assert step_metrics[f"train/{task}/count"] == 2 * collected.count(task)
        assert step_metrics[f"validation/{task}/count"] == 2
    assert sum(
        step_metrics.get(f"train/{task}/count", 0)
        for task in runtime.RLVR_TASKS
    ) == 4
    assert Telemetry.instances[0].finish_codes == [0]
    assert json.loads((output / "rlvr_complete_000001.json").read_text()) == {
        "step": 1,
        "world_size": 1,
    }
    loop = cast(dict[str, Any], saved_meta[0][runtime.RLVR_META_KEY])
    assert loop["completed_optimizer_step"] == 1
    assert loop["rank_rng"]["ddp_world_size"] == 1
    assert len(loop["rank_rng"]["rank_states"]) == 1
    assert loop["user_config"]["tasks"] == list(expected_tasks)
    assert len(warnings) == len(unparseable)
    for task, warning in zip(unparseable, warnings, strict=True):
        assert f"{task}: 0/2 completions" in warning
        assert f"training_enabled={task in expected_tasks}" in warning
        assert "validation/samples" in warning
        assert "docs/pipelines/encoder-decoder-rlvr.md" in warning


@pytest.mark.parametrize("finish_raises", [False, True])
@pytest.mark.parametrize("staging", [False, True])
def test_trainer_marks_failed_run_and_always_cleans_up(
    monkeypatch,
    tmp_path: Path,
    finish_raises: bool,
    staging: bool,
) -> None:
    events: list[object] = []
    cfg = OmegaConf.create(
        {
            **_settings(
                global_prompts_per_step=1,
                group_size=2,
                num_iterations=1,
                eval_every=1,
                save_every=1,
                train_feature_head_only=staging,
                frozen_decoder_max_shadow_bytes=64 * 1024**2 if staging else 0,
                frozen_decoder_min_free_bytes=0,
            ),
            "base_checkpoint_dir": "/unused/base",
            "model_step": 1116,
            "bundle_dir": "/unused/bundle",
            "stats_path": "/unused/stats",
            "output_dir": str(tmp_path / "failed"),
            "resume_checkpoint_dir": None,
        }
    )

    class Model(t.nn.Module):
        feature_pred_config = None

        def get_device(self):
            return t.device("cpu")

    class Optimizer:
        param_groups: list[dict[str, object]] = []

    rows = tuple(
        SimpleNamespace(task=task, row_id=f"{task}:0")
        for task in runtime.RLVR_TASKS
    )
    loaded = SimpleNamespace(
        model=Model(),
        bundle=SimpleNamespace(train=rows),
    )

    @contextmanager
    def staged_context(model, *, max_shadow_bytes, min_free_bytes,
                       train_feature_head_only, world_size):
        assert model is loaded.model
        assert max_shadow_bytes == 64 * 1024**2 and min_free_bytes == 0
        assert train_feature_head_only is True and world_size == 1
        events.append("stage")
        report = StagingReport(applied=True, reason="staged")
        try:
            yield report
        finally:
            report.cleaned_up = True
            events.append("unstage")

    class Telemetry:
        state = {"id": "failed-run", "last_logged_step": -1}

        def __init__(self, *_args, **_kwargs):
            pass

        def finish(self, *, exit_code: int = 0):
            events.append(("finish", exit_code))
            if finish_raises:
                raise RuntimeError("telemetry finish failed")

    monkeypatch.setattr(
        trainer,
        "compute_init_for_checkpoint_resume",
        lambda _device, *, resume_step: (True, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "sdpa")
    monkeypatch.setattr(trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(trainer, "load_rlvr_inputs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        trainer,
        "configure_grpo_trainable_parameters",
        lambda *_args, **_kwargs: ("weight",),
    )
    monkeypatch.setattr(trainer, "_build_optimizer", lambda *_args, **_kwargs: Optimizer())
    monkeypatch.setattr(trainer, "DistMuonAdamW", Optimizer)
    monkeypatch.setattr(trainer, "resume_identity", lambda *_args, **_kwargs: {"sha256": "id"})
    monkeypatch.setattr(trainer, "RLVRWandb", Telemetry)
    monkeypatch.setattr(trainer, "stage_frozen_decoder_linears", staged_context)
    monkeypatch.setattr(
        trainer,
        "evaluate_rlvr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("initial evaluation failed")),
    )

    expected = "telemetry finish failed" if finish_raises else "initial evaluation failed"
    with pytest.raises(RuntimeError, match=expected):
        trainer.train(cfg)

    assert events == (["stage", "unstage"] if staging else []) + [("finish", 1), "cleanup"]


@pytest.mark.parametrize("head_only", [False, True])
@pytest.mark.parametrize("tasks", [runtime.RLVR_TASKS, runtime.RLVR_TASKS[:-1]])
def test_trainer_checkpoint_resume_restores_real_model_optimizer_and_frozen_head(
    monkeypatch,
    tmp_path: Path,
    restore_cuda_sdpa_state,
    head_only: bool,
    tasks,
) -> None:
    """Resumed and ordinary steps preserve SGD momentum and compiler lifetime."""
    del restore_cuda_sdpa_state

    source_output = tmp_path / "resume_run"
    fork_output = tmp_path / "fork_run"

    def config(
        *,
        resume_from_step: int,
        output_dir: Path = source_output,
        resume_checkpoint_dir: Path | None = None,
        new_run: bool = False,
    ):
        return OmegaConf.create(
            {
                **_settings(
                    global_prompts_per_step=2,
                    group_size=2,
                    num_iterations=2,
                    eval_every=2,
                    save_every=1,
                    resume_from_step=resume_from_step,
                    train_feature_head_only=head_only,
                    grad_clip_max_norm=2.0,
                    tasks=list(tasks),
                ),
                "base_checkpoint_dir": "/unused/base",
                "model_step": 1116,
                "bundle_dir": "/unused/bundle",
                "stats_path": "/unused/stats",
                "output_dir": str(output_dir),
                "resume_checkpoint_dir": (
                    None if resume_checkpoint_dir is None else str(resume_checkpoint_dir)
                ),
                "wandb": {
                    "enabled": True,
                    "mode": "online",
                    "project": "test",
                    "entity": None,
                    "name": None,
                    "new_run": new_run,
                },
            }
        )

    class TinyModel(t.nn.Module):
        feature_pred_config = FeaturePredictionConfig()

        def __init__(self):
            super().__init__()
            self.weight = t.nn.Parameter(t.tensor([1.0]))
            self.feature_pred_head = t.nn.Linear(1, 1, bias=False)
            self.lm_head = t.nn.Linear(1, 1, bias=False)
            with t.no_grad():
                self.feature_pred_head.weight.fill_(1.0)
                self.lm_head.weight.fill_(7.0)

        def get_device(self):
            return self.weight.device

        def assert_vocab_consistent(self):
            return None

    class Snapshot:
        def assert_stable(self):
            return None

    rows = tuple(
        SimpleNamespace(task=task, row_id=f"{task}:0", prompt_token_ids=(1, 2))
        for task in runtime.RLVR_TASKS
    )
    models: list[TinyModel] = []

    def load_inputs(*_args, **_kwargs):
        model = TinyModel()
        models.append(model)
        return SimpleNamespace(
            model=model,
            bundle=SimpleNamespace(train=rows, validation=rows),
            cache_snapshot=Snapshot(),
            base_meta={"source": "tiny-base"},
            policy=object(),
            tokenizer=SimpleNamespace(decode=lambda _tokens: "A"),
        )

    optimizers: list[t.optim.SGD] = []

    def build_optimizer(model, *_args, **_kwargs):
        optimizer = t.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
            momentum=0.9,
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        optimizers.append(optimizer)
        return optimizer

    action = SimpleNamespace(
        policy_trace=SimpleNamespace(positive_rows=(1,), negative_rows=(2,)),
    )

    def group_for_row(_loaded, row, **_kwargs):
        def trajectory(reward: float):
            return SimpleNamespace(
                reward=reward,
                token_ids=(*row.prompt_token_ids, 3),
                sampled_mask=(0, 0, 1),
                live_actions=(action,),
                actions=(action,),
                stop_reason="eos",
            )

        return SimpleNamespace(
            trajectories=(trajectory(1.0), trajectory(0.0)),
            advantages=(1.0, -1.0),
        )

    def backward(model, _group, *, policy_config, loss_scale, train_feature_head_only, replay_cache=None):
        assert replay_cache is None
        del policy_config
        assert train_feature_head_only is head_only
        loss = model.weight.sum() + model.feature_pred_head.weight.sum()
        (loss * loss_scale).backward()
        return SimpleNamespace(
            loss=float(loss.detach()),
            ratio_min=1.0,
            ratio_max=1.0,
            live_decisions=2,
        )

    class Telemetry:
        instances: list[Telemetry] = []

        def __init__(self, _cfg, *, resumed=None, new_run=False, **_kwargs):
            if new_run:
                self.state = {
                    "id": "tiny-fork",
                    "project": "test",
                    "entity": None,
                    "mode": "online",
                    "last_logged_step": -1,
                    "parent_run_id": resumed["id"],
                }
            else:
                self.state = (
                    dict(resumed)
                    if resumed is not None
                    else {
                        "id": "tiny-run",
                        "project": "test",
                        "entity": None,
                        "mode": "online",
                        "last_logged_step": -1,
                        "parent_run_id": None,
                    }
                )
            self.logged: list[int] = []
            self.finish_codes: list[int] = []
            self.instances.append(self)

        def log(self, step, _metrics, _samples=()):
            if step > self.state["last_logged_step"]:
                self.logged.append(step)
                self.state["last_logged_step"] = step

        def finish(self, *, exit_code: int = 0):
            self.finish_codes.append(exit_code)

    class ReexecRequested(RuntimeError):
        pass

    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", False)
    monkeypatch.setattr(
        trainer.native_flash_attention,
        "_override_impl",
        trainer.native_flash_attention.FlashAttnImpl.FA4,
    )
    init_resume_steps: list[int] = []
    monkeypatch.setattr(
        trainer,
        "compute_init_for_checkpoint_resume",
        lambda _device, *, resume_step: init_resume_steps.append(resume_step)
        or (True, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: None)
    monkeypatch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "fa4")
    monkeypatch.setattr(trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(trainer, "load_rlvr_inputs", load_inputs)
    monkeypatch.setattr(trainer, "_build_optimizer", build_optimizer)
    monkeypatch.setattr(trainer, "DistMuonAdamW", t.optim.SGD)
    t.backends.cuda.enable_cudnn_sdp(False)
    t.backends.cuda.enable_flash_sdp(True)
    monkeypatch.setattr(
        trainer,
        "resume_identity",
        lambda _loaded, cfg, **kwargs: {
            "token_kv_backend": trainer._resolved_token_kv_backend(),
            "token_kv_sdpa": trainer._token_kv_sdpa_identity(t.device("cuda")),
            "parameter_manifest": list(kwargs["parameter_manifest"]),
            "tasks": list(runtime.resolve_rlvr_tasks(cfg.get("tasks", runtime.RLVR_TASKS))),
        },
    )
    monkeypatch.setattr(trainer, "RLVRWandb", Telemetry)
    evaluation_calls: list[int] = []
    warnings: list[str] = []
    monkeypatch.setattr(trainer.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))

    def evaluate(*_args, **_kwargs):
        evaluation_calls.append(1)
        metrics = {"validation/accuracy": 0.5}
        for task in runtime.RLVR_TASKS:
            metrics[f"validation/{task}/parse_rate"] = 0
            metrics[f"validation/{task}/count"] = 2
        return metrics, []

    monkeypatch.setattr(trainer, "evaluate_rlvr", evaluate)
    monkeypatch.setattr(trainer, "_group", group_for_row)
    monkeypatch.setattr(trainer, "backward_rlvr_group", backward)
    monkeypatch.setattr(
        trainer,
        "grade_completion_result",
        lambda *_args, **_kwargs: SimpleNamespace(valid_format=True),
    )
    compiler_resets: list[int] = []
    monkeypatch.setattr(
        trainer, "_reset_step_compiler_state", lambda: compiler_resets.append(1)
    )
    monkeypatch.setattr(
        trainer,
        "_reexec_from_checkpoint",
        lambda step: (_ for _ in ()).throw(ReexecRequested(str(step))),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conditional_encdec_rlvr.py",
            "resume_checkpoint_dir=/stale/source",
            "wandb.new_run=true",
            "keep=this",
        ],
    )
    with pytest.raises(ReexecRequested, match="1"):
        trainer.train(config(resume_from_step=-1))
    assert sys.argv == [
        "conditional_encdec_rlvr.py",
        "keep=this",
        "resume_checkpoint_dir=null",
        "wandb.new_run=false",
    ]
    assert models[0].weight.item() == pytest.approx(1.0 if head_only else 0.9)
    assert models[0].feature_pred_head.weight.item() == pytest.approx(0.9)
    assert models[0].lm_head.weight.item() == pytest.approx(7.0)
    assert not models[0].lm_head.weight.requires_grad
    runtime.assert_complete_checkpoint(source_output, 1, 1)

    changed_scope = config(resume_from_step=1)
    changed_scope.train_feature_head_only = not head_only
    with pytest.raises(SFTGRPOError, match="trainable parameter manifest.*same training scope"):
        trainer.train(changed_scope)
    assert models[-1].feature_pred_head.weight.item() == 1.0  # No resumed weights installed.
    assert not optimizers[-1].state
    models.pop()
    optimizers.pop()

    for changed_tasks in (tasks[1:], tasks[::-1]):
        changed = config(resume_from_step=1)
        changed.tasks = list(changed_tasks)
        with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
            trainer.train(changed)
        assert models[-1].weight.item() == 1.0
        assert models[-1].feature_pred_head.weight.item() == 1.0
        assert not optimizers[-1].state
        models.pop()
        optimizers.pop()

    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", True)
    with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
        trainer.train(config(resume_from_step=1))
    models.pop()
    optimizers.pop()
    monkeypatch.setattr(trainer.native_flash_attention, "HAS_FA3", False)

    t.backends.cuda.enable_flash_sdp(False)
    with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
        trainer.train(config(resume_from_step=1))
    models.pop()
    optimizers.pop()
    t.backends.cuda.enable_flash_sdp(True)

    metadata_path = source_output / "meta_000001.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    metadata[runtime.RLVR_META_KEY]["resume_identity"].pop("token_kv_sdpa")
    for invalid_metadata in (metadata, {**metadata, runtime.RLVR_META_KEY: None}):
        metadata_path.write_text(json.dumps(invalid_metadata))
        with pytest.raises(SFTGRPOError, match="RLVR resume identity mismatch"):
            trainer.train(config(resume_from_step=1))
        assert models[-1].feature_pred_head.weight.item() == 1.0
        assert not optimizers[-1].state
        models.pop()
        optimizers.pop()
    metadata_path.write_bytes(metadata_bytes)

    trainer.train(
        config(
            resume_from_step=1,
            output_dir=fork_output,
            resume_checkpoint_dir=source_output,
            new_run=True,
        )
    )
    assert len(models) == 2
    assert models[1].weight.item() == pytest.approx(1.0 if head_only else 0.71)
    assert models[1].feature_pred_head.weight.item() == pytest.approx(0.71)
    assert models[1].lm_head.weight.item() == pytest.approx(7.0)
    assert not models[1].lm_head.weight.requires_grad
    assert optimizers[1].state[models[1].feature_pred_head.weight]["momentum_buffer"].item() == pytest.approx(1.9)
    if head_only:
        assert models[1].weight.grad is None and models[1].weight not in optimizers[1].state
    else:
        assert optimizers[1].state[models[1].weight]["momentum_buffer"].item() == pytest.approx(1.9)
    assert not (source_output / "rlvr_complete_000002.json").exists()
    runtime.assert_complete_checkpoint(fork_output, 2, 1)
    saved_state, saved_optimizer, saved_meta = trainer.load_checkpoint(
        str(fork_output), 2, t.device("cpu"), load_optimizer=True, rank=0
    )
    assert saved_state["weight"].item() == pytest.approx(1.0 if head_only else 0.71)
    assert saved_state["feature_pred_head.weight"].item() == pytest.approx(0.71)
    assert saved_state["lm_head.weight"].item() == pytest.approx(7.0)
    t.testing.assert_close(saved_optimizer, optimizers[1].state_dict(), rtol=0, atol=0)
    assert saved_meta[runtime.RLVR_META_KEY]["completed_optimizer_step"] == 2
    assert Telemetry.instances[0].logged == [0, 1]
    assert Telemetry.instances[1].logged == [2]
    assert Telemetry.instances[0].finish_codes == [0]
    assert Telemetry.instances[1].finish_codes == [0]
    assert Telemetry.instances[1].state["id"] == "tiny-fork"
    assert Telemetry.instances[1].state["parent_run_id"] == "tiny-run"
    assert len(evaluation_calls) == 2
    assert len(warnings) == 4  # Initial validation only, never periodic or resumed validation.

    # Restarting the already-complete final checkpoint is a validated no-op:
    # checkpoint/model/optimizer identity is admitted, then no W&B run,
    # evaluation, training metric, or optimizer step is repeated. Even with a
    # positive cache budget, worker completion reports zero work and no session.
    # This orchestration fixture mocks resume_identity; settings pinning is
    # covered separately by test_hidden_replay_settings_are_bound_to_resume_identity.
    complete_cfg = config(resume_from_step=2, output_dir=fork_output)
    complete_cfg.hidden_replay_max_cache_bytes = 1024
    complete_cfg.hidden_replay_min_free_bytes = 0
    completion_reports = []

    def capture_completion(message, *args):
        if message == "RLVR hidden-replay worker complete: {}":
            completion_reports.append(args[0])

    with monkeypatch.context() as complete_patch:
        complete_patch.setattr(trainer.logger, "info", capture_completion)
        complete_patch.setattr(
            trainer, "rlvr_group_replay_cache",
            lambda *_args, **_kwargs: pytest.fail("Completed resume created a replay-cache session"),
        )
        trainer.train(complete_cfg)
    assert len(completion_reports) == 1
    completion = completion_reports[0]
    assert completion["requested"] == {
        "max_cache_bytes": 1024, "min_free_bytes": 0, "history_encoder": False, "prompt_encoder": False,
    }
    assert completion["worker_start_step"] == 2
    assert completion["worker_optimizer_updates"] == 0
    assert completion["worker_completed_without_error"] is True
    assert completion["cleaned_up"] is True and completion["applied"] is False
    for name in ("requested_groups", "inactive_groups", "active_groups", "admitted_groups",
                 "groups_collected", "groups_replayed", "groups_completed", "groups_closed",
                 "groups_failed", "groups_abandoned", "admitted_entries", "current_bytes",
                 "peak_bytes", "owned_tensor_objects_alive_after_cleanup",
                 "max_owned_tensor_objects_alive_after_cleanup"):
        assert completion[name] == 0, (name, completion)
    assert len(models) == 3
    assert models[2].weight.item() == pytest.approx(1.0 if head_only else 0.71)
    assert models[2].feature_pred_head.weight.item() == pytest.approx(0.71)
    assert len(Telemetry.instances) == 2
    assert len(evaluation_calls) == 2
    assert init_resume_steps == [-1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    assert len(compiler_resets) == (0 if head_only else 2)

    # Ordinary optimizer steps retain head-only compiled kernels. Full-policy
    # training still resets every step, and both scopes retain checkpoint parity.
    compiler_resets.clear()
    uninterrupted_output = tmp_path / "uninterrupted_run"
    uninterrupted_cfg = config(resume_from_step=-1, output_dir=uninterrupted_output)
    uninterrupted_cfg.save_every = 2
    trainer.train(uninterrupted_cfg)
    assert len(compiler_resets) == (0 if head_only else 2)
    assert Telemetry.instances[-1].logged == [0, 1, 2]
    runtime.assert_complete_checkpoint(uninterrupted_output, 2, 1)
    t.testing.assert_close(models[-1].state_dict(), saved_state, rtol=0, atol=0)
    t.testing.assert_close(optimizers[-1].state_dict(), saved_optimizer, rtol=0, atol=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_real_tokenizer_rlvr_case(
    tmp_path: Path, monkeypatch, *, head_depth: int = 0, head_block: str = "mlp"
) -> SimpleNamespace:
    """Build the smallest fully authenticated RLVR input accepted by production."""

    tokenizer_base = Path(os.environ.get(
        "NANOCHAT_BASE_DIR", "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/nanochat_cache"
    ))
    if not (tokenizer_base / "tokenizer" / "tokenizer.pkl").is_file():
        pytest.skip("real cached RustBPE tokenizer is unavailable")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tokenizer_base))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    tokenizer = get_tokenizer()
    try:
        source_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-4b-pt", local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - optional git-ignored test artifact
        pytest.skip(f"real cached Gemma tokenizer is unavailable: {exc}")

    base_dir = tmp_path / "base"
    initial_state = _save_base_checkpoint(
        base_dir,
        sequence_len=640,
        vocab_size=tokenizer.get_vocab_size(),
        head_depth=head_depth,
        head_block=head_block,
    )
    model_path = base_dir / "model_000001.pt"
    meta_path = base_dir / "meta_000001.json"
    meta = json.loads(meta_path.read_text())

    lineage_meta, _, stats_path = checkpoint_fixture(
        tmp_path / "lineage",
        size=32,
        window=4096,
        seed=42,
        width=8,
        buckets=2,
    )
    positive = meta[PRETRAIN_CONDITIONING_META_KEY]["positive_selection"]
    positive["mode"] = "top_k_percentile"
    lineage = lineage_meta["structured_sft_online_sae_lineage_v1"]
    lineage["feature_selection"] = dict(positive)
    lineage["negative_selection"] = dict(
        meta[PRETRAIN_CONDITIONING_META_KEY]["negative_selection"]
    )
    lineage["missing_percentile_policy"] = "lowest"
    meta.update(
        {
            "sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
            "turn_self_cond_segment_size_rustbpe": 33,
            "structured_sft_online_sae_lineage_v1": lineage,
            "feature_vocab": None,
        }
    )
    meta["user_config"].update(lineage_meta["user_config"])
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    bundle_dir = tmp_path / "bundle"
    expected_counts = dict.fromkeys(runtime.RLVR_TASKS, 1)
    preparation_cfg = OmegaConf.create({
        "output_dir": str(bundle_dir),
        "checkpoint_meta_path": str(meta_path),
        "base_checkpoint_path": str(model_path),
        "base_checkpoint_sha256": _sha256_file(model_path),
        "stats_path": str(stats_path),
        "model_max_seq_len": 640,
        "revisions": dict(RLVR_DATASET_REVISIONS),
        "expected_train_counts": expected_counts,
        "expected_validation_counts": expected_counts,
        "expected_removed_counts": dict.fromkeys(runtime.RLVR_TASKS, 0),
        "train_caps": None,
        "validation_caps": None,
    })
    # Only dataset acquisition is replaced: preparation, both real tokenizers,
    # raw-shard conversion, finalization and admission are production code.
    monkeypatch.setattr(
        prepare_script, "load_pinned_sources",
        lambda: (_sources(), _sources(), dict.fromkeys(runtime.RLVR_TASKS, 0)),
    )
    preparation_path = prepare_script.prepare_data_from_config(preparation_cfg)
    preparation = json.loads(preparation_path.read_text())
    records = cache_builder.load_context_records(bundle_dir / "contexts.jsonl")
    synthetic_shards(
        tmp_path / "features", records, source_tokenizer, size=32, split=1, window=4096,
    )
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs/eval"),
        version_base=None,
    ):
        conversion_cfg = compose(config_name="build_eval_context_cache_turn")
    OmegaConf.set_struct(conversion_cfg.metadata, False)
    conversion_cfg.merge_with({
        "checkpoint_meta_path": str(meta_path),
        "contexts_file": str(bundle_dir / "contexts.jsonl"),
        "context_parquet_dir": str(bundle_dir / "context_parquet"),
        "feature_dir": str(tmp_path / "features"),
        "output_dir": str(bundle_dir / "context_cache"),
        "stats_path": str(stats_path),
        "model_max_seq_len": 640,
        "evals": "chat_rlvr",
        "metadata": {
            "rlvr_data_sha256": preparation["bundle_sha256"],
            "rlvr_base_model_sha256": _sha256_file(model_path),
            "rlvr_base_metadata_sha256": _sha256_file(meta_path),
        },
    })
    cache_builder.build_cache_from_config(conversion_cfg)
    prepare_script.finalize_from_config(preparation_cfg)
    return SimpleNamespace(
        base_dir=base_dir,
        bundle_dir=bundle_dir,
        stats_path=stats_path,
        initial_state=initial_state,
        preparation_cfg=preparation_cfg,
        conversion_cfg=conversion_cfg,
        load_cfg=SimpleNamespace(
            base_checkpoint_dir=str(base_dir), model_step=1, seed=0,
            stats_path=str(stats_path), bundle_dir=str(bundle_dir),
            mc_max_tokens=16, gsm8k_max_tokens=512,
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize("head_block,head_depth", [("mlp", 0), ("mlp", 2), ("transformer", 2)])
def test_real_tokenizers_tiny_checkpoint_and_schema5_bundle_admit_offline(
    tmp_path: Path,
    monkeypatch,
    head_depth,
    head_block,
) -> None:
    case = _build_real_tokenizer_rlvr_case(tmp_path, monkeypatch, head_depth=head_depth, head_block=head_block)
    original_init = ResidualFeaturePredictionHead.init_residual_weights

    def init_meta_only(head):
        assert head.weight.is_meta, "restoration initialized real residual weights"
        original_init(head)

    monkeypatch.setattr(
        ResidualFeaturePredictionHead, "init_residual_weights", init_meta_only,
    )
    loaded = trainer.load_rlvr_inputs(
        SimpleNamespace(
            base_checkpoint_dir=str(case.base_dir),
            model_step=1,
            seed=0,
            stats_path=str(case.stats_path),
            bundle_dir=str(case.bundle_dir),
            mc_max_tokens=16,
            gsm8k_max_tokens=512,
        ),
        device=t.device("cpu"),
        rank=0,
    )
    assert loaded.cadence == 33
    assert loaded.model.feature_pred_config.head_layers == head_depth
    for name, expected in case.initial_state.items():
        assert t.equal(loaded.model.state_dict()[name], expected), name
    # Independent readout from the saved tensors detects skipped residual math
    # as well as missing blocks and load-time initialization.
    hidden = t.randn(2, 3, loaded.model.dec_config.n_embd)
    expected_logits = hidden
    for index in range(head_depth):
        prefix = f"feature_pred_head.residual_blocks.{index}."
        branch = F.rms_norm(expected_logits, (hidden.shape[-1],))
        branch = F.relu(F.linear(branch, case.initial_state[prefix + "c_fc.weight"])).square()
        expected_logits = expected_logits + F.linear(branch, case.initial_state[prefix + "c_proj.weight"])
    expected_logits = F.linear(
        expected_logits, case.initial_state["feature_pred_head.weight"],
        case.initial_state["feature_pred_head.bias"],
    )
    assert t.equal(loaded.model.feature_pred_head(hidden), expected_logits)
    assert len(loaded.bundle.train) == len(runtime.RLVR_TASKS)
    assert len(loaded.bundle.validation) == len(runtime.RLVR_TASKS)
    assert len(loaded.provider) == len(runtime.RLVR_TASKS)
    assert loaded.model.encoder.config.n_sae_features == 8
    assert loaded.model.encoder.config.num_buckets == 2
    assert loaded.compiler_config["derived_minimum"] == (
        1
        + 2
        * len(set(loaded.model.window_sizes))
        * loaded.model.dec_config.sequence_len
    )
    assert loaded.compiler_config["fail_on_recompile_limit_hit"] is True
    with pytest.raises(SFTGRPOError, match="prompt plus requested answer budget"):
        trainer.load_rlvr_inputs(
            SimpleNamespace(
                base_checkpoint_dir=str(case.base_dir),
                model_step=1,
                seed=0,
                stats_path=str(case.stats_path),
                bundle_dir=str(case.bundle_dir),
                mc_max_tokens=16,
                gsm8k_max_tokens=640,
            ),
            device=t.device("cpu"),
            rank=0,
        )


@pytest.mark.slow
@pytest.mark.parametrize("mutation", ["missing_block", "extra_block", "wrong_depth", "legacy_depth", "invalid_depth", "vocabulary"])
def test_rlvr_depth2_base_rejects_architecture_disagreement(tmp_path, monkeypatch, mutation):
    case = _build_real_tokenizer_rlvr_case(tmp_path, monkeypatch, head_depth=2)
    model_path = case.base_dir / "model_000001.pt"
    meta_path = case.base_dir / "meta_000001.json"
    metadata = json.loads(meta_path.read_text())
    state = dict(case.initial_state)
    if mutation == "missing_block":
        state.pop("feature_pred_head.residual_blocks.1.c_proj.weight")
    elif mutation == "extra_block":
        state["feature_pred_head.residual_blocks.2.c_proj.weight"] = state["feature_pred_head.residual_blocks.1.c_proj.weight"]
    elif mutation == "wrong_depth":
        metadata["feature_prediction_config"]["head_layers"] = 1
    elif mutation == "legacy_depth":
        metadata["feature_prediction_config"].pop("head_layers")
    elif mutation == "invalid_depth":
        metadata["feature_prediction_config"]["head_layers"] = "2"
    else:
        metadata["feature_vocab"] = FeatureVocab(8, (0, 1, 2, 3, 4, 7)).metadata_dict()
    t.save(state, model_path)
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises((RuntimeError, ValueError), match="key|head_layers|vocab|feature"):
        trainer.load_rlvr_inputs(case.load_cfg, device=t.device("cpu"), rank=0)


@pytest.mark.slow
@pytest.mark.parametrize("pin", ["rlvr_data_sha256", "rlvr_base_model_sha256", "rlvr_base_metadata_sha256"])
def test_real_tokenizer_finalization_rejects_stale_depth2_producer_pins(tmp_path, monkeypatch, pin):
    case = _build_real_tokenizer_rlvr_case(tmp_path, monkeypatch, head_depth=2)
    manifest = json.loads((case.bundle_dir / "manifest.json").read_text())
    (case.bundle_dir / "manifest.json").unlink()
    cache_manifest = case.bundle_dir / manifest["context_cache"]["path"] / "manifest.json"
    cache = json.loads(cache_manifest.read_text())
    cache["metadata"][pin] = "f" * 64
    cache_manifest.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="metadata"):
        prepare_script.finalize_from_config(case.preparation_cfg)
    assert not (case.bundle_dir / "manifest.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize(("section", "key", "message"), [
    ("base_checkpoint", "model_sha256", "base checkpoint authority"),
    ("base_checkpoint", "metadata_sha256", "base metadata authority"),
    ("statistics", "sha256", "statistics authority"),
    ("tokenizers", "decoder_sha256", "decoder tokenizer checksum"),
    ("tokenizers", "extraction_sha256", "extraction tokenizer checksum"),
    (None, "unfinalized", "cannot read RLVR manifest"),
    (None, "shard", "cache shard hash mismatch"),
    (None, "rebound_base", "base checkpoint authority"),
])
def test_depth2_bundle_admission_rejects_changed_authorities(tmp_path, monkeypatch, section, key, message):
    case = _build_real_tokenizer_rlvr_case(tmp_path, monkeypatch, head_depth=2)
    manifest_path = case.bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if section is not None:
        manifest[section][key] = "f" * 64
        manifest["bundle_sha256"] = bundle_manifest_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest))
    elif key == "unfinalized":
        manifest_path.unlink()
    elif key == "rebound_base":
        # An internally consistent relabel of the cache and bundle must still
        # fail against the independently admitted checkpoint's actual bytes.
        cache_path = case.bundle_dir / manifest["context_cache"]["path"] / "manifest.json"
        cache_manifest = json.loads(cache_path.read_text())
        for field in ("model_sha256", "metadata_sha256"):
            manifest["base_checkpoint"][field] = "f" * 64
            cache_manifest["metadata"][f"rlvr_base_{field}"] = "f" * 64
        cache_path.write_text(json.dumps(cache_manifest))
        manifest["context_cache"]["manifest_sha256"] = _sha256_file(cache_path)
        manifest["bundle_sha256"] = bundle_manifest_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest))
    else:
        cache = manifest["context_cache"]
        shard = case.bundle_dir / cache["path"] / cache["shards"][0]["file"]
        with shard.open("ab") as handle:
            handle.write(b"altered")
    with pytest.raises(ValueError, match=message):
        trainer.load_rlvr_inputs(case.load_cfg, device=t.device("cpu"), rank=0)


def _assert_frozen_decoder_worker_lifetimes(
    output: str,
    expected_workers: tuple[tuple[int, int], ...],
) -> None:
    """Check real worker setup/cleanup reports, including checkpoint reexecs."""
    events = []
    for line in output.splitlines():
        for phase in ("setup", "complete"):
            marker = f"RLVR frozen-decoder worker {phase}: "
            if marker in line:
                report = ast.literal_eval(line.split(marker, 1)[1])
                assert isinstance(report, dict)
                events.append((phase, report))
    assert len(events) == 2 * len(expected_workers), events
    for index, (start_step, updates) in enumerate(expected_workers):
        (setup_phase, setup), (complete_phase, complete) = events[2 * index : 2 * index + 2]
        assert (setup_phase, complete_phase) == ("setup", "complete")
        for report in (setup, complete):
            assert report["applied"] is True and report["reason"] == "staged"
            assert report["fallback_calls"] == report["invalidated_modules"] == 0
            assert report["worker_start_step"] == start_step
            assert report["planned_shadow_bytes"] == report["staged_shadow_bytes"] > 0
            assert report["setup_seconds"] >= report["staging_seconds"] >= 0
        assert setup["cleaned_up"] is False
        assert complete["cleaned_up"] is True
        assert complete["module_names"] == setup["module_names"]
        assert complete["setup_seconds"] == setup["setup_seconds"]
        assert complete["worker_optimizer_updates"] == updates
        assert complete["cleanup_seconds"] >= 0
        assert complete["setup_seconds_per_update"] == pytest.approx(
            setup["setup_seconds"] / updates
        )


def _assert_hidden_replay_worker_lifetimes(output: str, expected_workers: tuple[tuple[int, int], ...]) -> None:
    marker = "RLVR hidden-replay worker complete: "
    reports = [ast.literal_eval(line.split(marker, 1)[1]) for line in output.splitlines() if marker in line]
    assert len(reports) == len(expected_workers), reports
    for report, (start_step, updates) in zip(reports, expected_workers, strict=True):
        assert report["worker_start_step"] == start_step and report["worker_optimizer_updates"] == updates
        assert report["worker_completed_without_error"] is True
        assert report["requested"] == {
            "max_cache_bytes": 16 * 1024**2, "min_free_bytes": 0, "history_encoder": True, "prompt_encoder": True,
        }
        assert report["requested_groups"] == report["groups_collected"] == report["groups_replayed"] == updates
        assert report["groups_completed"] == report["groups_closed"] == report["admitted_groups"] == updates
        assert report["inactive_groups"] == report["active_groups"] == report["groups_abandoned"] == report["groups_failed"] == 0
        assert report["applied"] is True and report["cleaned_up"] is True
        assert report["replay_hits"] == report["replay_calls"] > 0
        assert report["replay_misses"] == report["invalidations"] == report["optional_copy_ooms"] == 0
        assert report["current_bytes"] == report["current_hidden_bytes"] == report["current_prompt_bytes"] == 0
        assert report["current_canonical_prompt_bytes"] == 0
        assert report["owned_tensor_objects_alive_after_cleanup"] == report["max_owned_tensor_objects_alive_after_cleanup"] == 0
        assert 0 < report["peak_hidden_bytes"] <= report["peak_bytes"] <= 16 * 1024**2


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_block,head_depth", [("mlp", 0), ("mlp", 2), ("transformer", 2)])
@pytest.mark.parametrize("head_only", [False, True])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("tasks", [runtime.RLVR_TASKS, runtime.RLVR_TASKS[:-1]])
def test_tiny_real_tokenizer_cli_reexec_resumes_and_freezes_lm_head(
    tmp_path: Path,
    monkeypatch,
    head_only: bool,
    world_size,
    tasks,
    head_depth,
    head_block,
) -> None:
    if t.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices are required")
    case = _build_real_tokenizer_rlvr_case(tmp_path, monkeypatch, head_depth=head_depth, head_block=head_block)
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "scripts.train.conditional_encdec_rlvr",
        "--",
        f"base_checkpoint_dir={case.base_dir}",
        "model_step=1",
        f"bundle_dir={case.bundle_dir}",
        f"stats_path={case.stats_path}",
        f"output_dir={output}",
        f"global_prompts_per_step={world_size}",
        "group_size=2",
        "num_iterations=3",
        f"tasks=[{','.join(tasks)}]",
        "device_type=cuda",
        "seed=0",
        "eval_every=1",
        "save_every=1",
        "wandb.enabled=false",
    ]
    if head_only:
        command.append("train_feature_head_only=true")
    # Reuse the existing single-GPU depth-2 case to cover staged worker restarts.
    # Other cases retain the default-off and full-policy coverage.
    staging = head_only and head_depth == 2 and world_size == 1
    if staging:
        command.extend([
            "frozen_decoder_max_shadow_bytes=67108864",
            "frozen_decoder_min_free_bytes=0",
            "hidden_replay_max_cache_bytes=16777216",
            "hidden_replay_min_free_bytes=0",
            "hidden_replay_history_encoder=true",
            "hidden_replay_prompt_encoder=true",
        ])
    environment = os.environ.copy()
    environment.update(
        {
            "NANOCHAT_ATTN_IMPL": "sdpa",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    result = run_torchrun(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for step in (1, 2, 3):
        runtime.assert_complete_checkpoint(output, step, world_size)
    trained, optimizer, meta = load_checkpoint(
        str(output), 3, t.device("cpu"), load_optimizer=True, rank=0
    )
    assert trained.keys() == case.initial_state.keys()
    t.testing.assert_close(
        trained["lm_head.weight"],
        case.initial_state["lm_head.weight"],
        atol=0,
        rtol=0,
    )
    assert optimizer is not None
    assert optimizer.keys() == {"state", "param_groups"}
    assert meta["feature_prediction_config"]["head_layers"] == head_depth
    assert meta["feature_prediction_config"]["block"] == head_block
    assert meta["user_config"]["feature_prediction"] == meta["feature_prediction_config"]
    loop = meta[runtime.RLVR_META_KEY]
    assert loop.keys() == {
        "version", "completed_optimizer_step", "resume_identity", "rank_rng", "wandb", "user_config",
    }
    manifest = loop["resume_identity"]["parameter_manifest"]
    head_names = [name for name in case.initial_state if name.startswith("feature_pred_head.")]
    assert len(head_names) == 2 + (4 if head_block == "transformer" else 2) * head_depth
    if head_only:
        assert manifest == head_names
        for name, initial in case.initial_state.items():
            if not name.startswith("feature_pred_head."):
                assert t.equal(trained[name], initial), name
    assert loop["completed_optimizer_step"] == 3
    assert loop["user_config"]["tasks"] == list(tasks)
    assert loop["resume_identity"]["config"]["tasks"] == list(tasks)
    combined_output = result.stdout + result.stderr
    assert "RLVR step=1" in combined_output
    assert "RLVR step=2" in combined_output
    assert "RLVR step=3" in combined_output
    _assert_frozen_decoder_worker_lifetimes(
        combined_output, ((0, 1), (1, 1), (2, 1)) if staging else ()
    )
    _assert_hidden_replay_worker_lifetimes(
        combined_output, ((0, 1), (1, 1), (2, 1)) if staging else ()
    )
    if staging:
        for config in (loop["user_config"], loop["resume_identity"]["config"]):
            assert config["frozen_decoder_max_shadow_bytes"] == 64 * 1024**2
            assert config["frozen_decoder_min_free_bytes"] == 0
            assert config["hidden_replay_max_cache_bytes"] == 16 * 1024**2
            assert config["hidden_replay_min_free_bytes"] == 0
            assert config["hidden_replay_history_encoder"] is True
            assert config["hidden_replay_prompt_encoder"] is True

    fork_output = tmp_path / "fork"
    fork_command = [arg for arg in command if not arg.startswith("output_dir=")]
    fork_command.extend([
        f"output_dir={fork_output}", f"resume_checkpoint_dir={output}",
        "resume_from_step=1", "wandb.new_run=true",
    ])
    fork = run_torchrun(
        fork_command, cwd=Path(__file__).resolve().parents[2], env=environment,
        timeout=1200,
    )
    assert fork.returncode == 0, fork.stdout + fork.stderr
    _assert_frozen_decoder_worker_lifetimes(
        fork.stdout + fork.stderr, ((1, 1), (2, 1)) if staging else ()
    )
    _assert_hidden_replay_worker_lifetimes(
        fork.stdout + fork.stderr, ((1, 1), (2, 1)) if staging else ()
    )
    fork_state, fork_optimizer, fork_meta = load_checkpoint(
        str(fork_output), 3, t.device("cpu"), load_optimizer=True, rank=0,
    )
    for name, expected in trained.items():
        assert t.equal(fork_state[name], expected), name
    assert fork_optimizer is not None
    assert fork_state.keys() == trained.keys()
    assert fork_optimizer.keys() == optimizer.keys()
    assert fork_meta[runtime.RLVR_META_KEY].keys() == loop.keys()
    assert fork_optimizer["param_groups"] == optimizer["param_groups"]
    for key, state in optimizer["state"].items():
        for name, expected in state.items():
            actual = fork_optimizer["state"][key][name]
            assert t.equal(actual, expected) if isinstance(expected, t.Tensor) else actual == expected
    assert fork_meta[runtime.RLVR_META_KEY]["rank_rng"] == meta[runtime.RLVR_META_KEY]["rank_rng"]

    evaluation_path = tmp_path / "evaluation.json"
    evaluation_command = [
        sys.executable, "-m", "scripts.eval.encdec_rlvr",
        f"base_checkpoint_dir={case.base_dir}", "model_step=1",
        f"bundle_dir={case.bundle_dir}", f"stats_path={case.stats_path}",
        f"rlvr_checkpoint_dir={output}", "rlvr_step=3", "seed=0", "device_type=cuda",
        f"output_path={evaluation_path}",
    ]
    evaluation = subprocess.run(
        evaluation_command, cwd=Path(__file__).resolve().parents[2], env=environment,
        text=True, capture_output=True, timeout=600, check=False,
    )
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    result = json.loads(evaluation_path.read_text())
    assert result["checkpoint"]["step"] == 3
    assert all(result["metrics"][f"validation/{task}/count"] == 1 for task in runtime.RLVR_TASKS)

    # Saving only at the end avoids reexec; optimizer updates must be identical.
    uninterrupted = tmp_path / "uninterrupted"
    reference_command = [
        f"output_dir={uninterrupted}"
        if arg.startswith("output_dir=")
        else "save_every=3"
        if arg == "save_every=1"
        else arg
        for arg in command
    ]
    reference = run_torchrun(
        reference_command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        timeout=1200,
    )
    assert reference.returncode == 0, reference.stdout + reference.stderr
    _assert_frozen_decoder_worker_lifetimes(
        reference.stdout + reference.stderr, ((0, 3),) if staging else ()
    )
    _assert_hidden_replay_worker_lifetimes(
        reference.stdout + reference.stderr, ((0, 3),) if staging else ()
    )
    runtime.assert_complete_checkpoint(uninterrupted, 3, world_size)
    reference_state, reference_optimizer, reference_meta = load_checkpoint(
        str(uninterrupted), 3, t.device("cpu"), load_optimizer=True, rank=0
    )
    assert reference_meta[runtime.RLVR_META_KEY].keys() == loop.keys()
    t.testing.assert_close(trained, reference_state, atol=0, rtol=0)
    for rank in range(world_size):
        if rank:
            _, optimizer, _ = load_checkpoint(
                str(output), 3, t.device("cpu"), load_optimizer=True, rank=rank
            )
            _, reference_optimizer, _ = load_checkpoint(
                str(uninterrupted), 3, t.device("cpu"), load_optimizer=True, rank=rank
            )
        assert optimizer is not None and reference_optimizer is not None
        if head_only:
            groups = optimizer["param_groups"]
            parameter_ids = [parameter for group in groups for parameter in group["params"]]
            assert len(parameter_ids) == len(set(parameter_ids)) == len(head_names)
            expected_state = set()
            for group in groups:
                assert group["kind"] in ("adamw", "muon")
                owners = group["params"][:1] if group["kind"] == "muon" else group["params"]
                expected_state.update(owners)
                if group["kind"] == "muon":
                    state = optimizer["state"][owners[0]]
                    rows = (len(group["params"]) + world_size - 1) // world_size
                    assert state["momentum_buffer"].shape[0] == rows
                    assert state["second_momentum_buffer"].shape[0] == rows
            assert set(optimizer["state"]) == expected_state
        assert optimizer["param_groups"] == reference_optimizer["param_groups"]
        t.testing.assert_close(optimizer["state"], reference_optimizer["state"], atol=0, rtol=0)
