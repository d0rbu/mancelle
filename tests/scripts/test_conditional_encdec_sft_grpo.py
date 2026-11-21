from __future__ import annotations

import inspect
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue, OmegaConf

from nanochat.checkpoint_manager import meta_looks_encdec
from scripts.train import conditional_encdec_sft_grpo as trainer
from steering.conditional.encoder_decoder import sm12_fa4
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    serialize_feature_prediction_config,
)
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    ONLINE_SAE_LINEAGE_KEY,
    build_online_sae_lineage_metadata,
)
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    assert_grpo_resume_identity,
    build_grpo_resume_identity,
)
from steering.conditional.encoder_decoder.sft_grpo_training import GRPOPolicyConfig
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import SFT_SEGMENT_MODE_META_KEY
from steering.sft.structured_dataset import StagedEnvironmentBatch
from tests.steering.selfcond.test_sft_grpo_training import TinySequentialPolicy

CONFIG_DIR = str(Path("configs/training").resolve())


def _compose(name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name, overrides=overrides or [])


def _policy() -> GRPOPolicyConfig:
    return GRPOPolicyConfig(
        ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        SamplingSelection(mean=1, std=1, min_k=1, max_k=2),
        NegativeSamplingSelection(mean=1, std=1, min_k=0, max_k=1),
        seed=1,
    )


def test_base_config_keeps_hardware_geometry_mandatory() -> None:
    cfg = _compose(
        "conditional_encdec_sft_grpo",
        [
            "base_checkpoint_dir=/base",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
        ],
    )
    for key in (
        "global_conversations_per_step",
        "group_size",
        "device_conversation_batch_size",
        "trajectory_microbatch_size",
        "num_iterations",
    ):
        with pytest.raises(MissingMandatoryValue):
            OmegaConf.select(cfg, key, throw_on_missing=True)


def test_pilot_pins_requested_group_and_empty_initial_conditioning() -> None:
    cfg = _compose(
        "conditional_encdec_sft_grpo_pilot",
        [
            "base_checkpoint_dir=/base",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
            "global_conversations_per_step=2",
            "device_conversation_batch_size=1",
            "trajectory_microbatch_size=1",
            "num_iterations=1",
        ],
    )
    assert cfg.group_size == 256
    assert cfg.initial_conditioning_mode == "none"
    assert cfg.sae_loaded is False
    assert cfg.kl_beta == 0
    assert cfg.segment_length == 256


@pytest.mark.parametrize(
    "name",
    [
        "conditional_encdec_sft_grpo",
        "conditional_encdec_sft_grpo_pilot",
        "conditional_encdec_sft_grpo_chatcore",
    ],
)
def test_head_only_config_is_inherited_and_enabled_by_default(name: str) -> None:
    assert _compose(name).train_feature_head_only is True
    assert _compose(name, ["train_feature_head_only=false"]).train_feature_head_only is False


@pytest.mark.parametrize("raw", [None, 0, 1, 0.0, "false", "true", [], {}])
def test_head_only_config_requires_a_boolean(raw: object) -> None:
    with pytest.raises(SFTGRPOError, match="must be a boolean"):
        trainer.resolve_train_feature_head_only({"train_feature_head_only": raw})


def test_head_only_config_absence_and_false_preserve_legacy_scopes() -> None:
    for ntp in (None, {"loss_weight": 1.0}):
        assert trainer.resolve_train_feature_head_only({"ntp": ntp}) is False
        assert (
            trainer.resolve_train_feature_head_only({"train_feature_head_only": False, "ntp": ntp})
            is False
        )
    assert (
        trainer.resolve_train_feature_head_only({"train_feature_head_only": True, "ntp": None})
        is True
    )


@pytest.mark.parametrize("ntp", [False, True, 0, "enabled", [], {}, {"loss_weight": 1.0}])
def test_head_only_rejects_every_non_null_ntp_block(ntp: object) -> None:
    with pytest.raises(SFTGRPOError, match="requires ntp=null.*no differentiable path"):
        trainer.resolve_train_feature_head_only({"train_feature_head_only": True, "ntp": ntp})


@pytest.mark.parametrize(
    "override",
    [
        "ntp={}",
        "ntp=false",
        "ntp={trajectories_per_group:1,loss_weight:1.0,decay_start:0.8,fp_head_lr:null}",
    ],
)
def test_head_only_ntp_rejection_precedes_runtime_initialization(monkeypatch, override):
    monkeypatch.setattr(trainer, "_reexec_for_deterministic_kernels", lambda: None)

    def unreachable(*_args, **_kwargs):
        pytest.fail("head-only/NTP validation must precede runtime construction")

    for name in (
        "compute_init_for_checkpoint_resume",
        "load_checkpoint",
        "_build_model",
        "_build_optimizer",
        "get_tokenizer",
        "load_structured_rollout_artifact",
    ):
        monkeypatch.setattr(trainer, name, unreachable)
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "trainer.py",
            "base_checkpoint_dir=/base",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
            "global_conversations_per_step=1",
            "group_size=2",
            "device_conversation_batch_size=1",
            "trajectory_microbatch_size=1",
            "num_iterations=1",
            override,
        ],
    )
    with pytest.raises(SFTGRPOError, match="requires ntp=null.*integer/detached"):
        trainer.main()


@pytest.mark.parametrize("fault", [None, "identity"])
def test_main_restores_head_scope_before_optimizer_and_cursor_state(monkeypatch, fault):
    events = []
    identities = []
    logs = []
    model = TinySequentialPolicy().eval()
    fp_config = FeaturePredictionConfig()
    monkeypatch.setattr(model, "feature_pred_config", fp_config, raising=False)
    monkeypatch.setattr(model, "dec_config", SimpleNamespace(sequence_len=4), raising=False)
    monkeypatch.setattr(model, "assert_vocab_consistent", lambda: None, raising=False)
    real_load = model.load_state_dict
    real_identity = trainer.build_grpo_resume_identity
    expected = ("feature_pred_head.weight", "feature_pred_head.bias")

    class Optimizer:
        def __init__(self):
            events.append("build_optimizer")
            self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
            assert {id(p) for p in self.parameters} == {
                id(p) for p in model.feature_pred_head.parameters()
            }

        def load_state_dict(self, state):
            assert state == {"saved": True}
            assert (
                tuple(name for name, p in model.named_parameters() if p.requires_grad) == expected
            )
            assert {id(p) for p in self.parameters} == {
                id(p) for p in model.feature_pred_head.parameters()
            }
            events.append("optimizer_load")

    def load_model(state, **kwargs):
        assert kwargs == {"strict": True, "assign": False}
        events.append("model_load")
        return real_load(state, **kwargs)

    def identity(**kwargs):
        result = real_identity(**kwargs)
        identities.append(result)
        return result

    def checkpoint(*_args, **kwargs):
        if not kwargs["load_optimizer"]:
            return model.state_dict(), None, {}
        recorded = dict(identities[0])
        if fault == "identity":
            recorded["parameter_manifest"] = ["lm_head.weight"]
        return (
            model.state_dict(),
            {"saved": True},
            {
                "feature_prediction_config": serialize_feature_prediction_config(fp_config),
                trainer.GRPO_LOOP_META_KEY: {
                    "completed_optimizer_step": 1,
                    "initial_conditioning_mode": "none",
                    "sae_loaded": False,
                    "resume_identity": recorded,
                    "train_cursor": {},
                    "validation_cursor": {},
                }
            },
        )

    def cursor_state(*_args, **_kwargs):
        events.append("cursor")
        return None

    def compute_init(_device, *, resume_step):
        assert resume_step == 1
        return True, 0, 0, 1, "cpu"

    monkeypatch.setattr(trainer, "_reexec_for_deterministic_kernels", lambda: None)
    monkeypatch.setattr(trainer.logger, "info", logs.append)
    monkeypatch.setattr(trainer, "compute_init_for_checkpoint_resume", compute_init)
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: None)
    monkeypatch.setattr(trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(trainer, "load_checkpoint", checkpoint)
    monkeypatch.setattr(trainer, "_resolved_checkpoint_sha256", lambda *_args: "a" * 64)
    monkeypatch.setattr(trainer, "_build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        trainer,
        "validate_grpo_checkpoint_eligibility",
        lambda *_args, **_kwargs: SimpleNamespace(policy=_policy()),
    )
    monkeypatch.setattr(model, "load_state_dict", load_model)
    monkeypatch.setattr(trainer, "DistMuonAdamW", Optimizer)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda *_args, **_kwargs: Optimizer())
    monkeypatch.setattr(
        trainer,
        "get_tokenizer",
        lambda: SimpleNamespace(decode=lambda _ids: "", get_bos_token_id=lambda: 0),
    )
    monkeypatch.setattr(trainer, "rustbpe_tokenizer_sha256", lambda _tokenizer: "b" * 64)
    monkeypatch.setattr(
        trainer,
        "load_structured_rollout_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(identity_sha256="c" * 64),
    )
    monkeypatch.setattr(trainer, "build_grpo_resume_identity", identity)
    monkeypatch.setattr(trainer, "select_dataloader_rank_state", cursor_state)
    monkeypatch.setattr(trainer, "StructuredEnvironmentCursor", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "")
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "trainer.py",
            "base_checkpoint_dir=/base",
            "model_step=1",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
            "global_conversations_per_step=1",
            "group_size=2",
            "device_conversation_batch_size=1",
            "trajectory_microbatch_size=1",
            "num_iterations=1",
            "resume_from_step=1",
            "device_type=cpu",
        ],
    )
    if fault is None:
        trainer.main()
        assert events == [
            "build_optimizer",
            "model_load",
            "optimizer_load",
            "cursor",
            "cursor",
        ]
        startup = next(message for message in logs if "GRPO startup:" in message)
        trainable = sum(p.numel() for p in model.feature_pred_head.parameters())
        total = sum(p.numel() for p in model.parameters())
        assert "train_feature_head_only=true" in startup
        assert f"trainable_parameters={trainable}, total_parameters={total}," in startup
    else:
        with pytest.raises(SFTGRPOError, match="same training scope"):
            trainer.main()
        assert "optimizer_load" not in events and "cursor" not in events
        assert "model_load" not in events


def test_timing_profile_defaults_are_disabled_and_strict(tmp_path: Path) -> None:
    cfg = _compose(
        "conditional_encdec_sft_grpo",
        [
            "base_checkpoint_dir=/base",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
        ],
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert trainer._resolve_timing_profile(resolved) == (
        False,
        Path("out/profiles/encdec_sft_grpo"),
    )
    assert trainer._resolve_timing_profile(
        {
            "profile_timing": True,
            "profile_output_dir": str(tmp_path),
        }
    ) == (True, tmp_path)
    with pytest.raises(trainer.SFTGRPOError, match="profile_timing must be a boolean"):
        trainer._resolve_timing_profile({"profile_timing": 1})
    with pytest.raises(trainer.SFTGRPOError, match="must be a non-empty string"):
        trainer._resolve_timing_profile({"profile_output_dir": "  "})


def test_trainer_is_separate_uncompiled_and_has_no_ddp_wrapper() -> None:
    source = inspect.getsource(trainer)
    assert "DistributedDataParallel" not in source
    assert "torch.compile" not in source
    assert "OnlineSAEInitialConditioningProvider" in source
    assert "DistMuonAdamW" in source
    assert "trainable_only=True" in source
    assert "weight_decay=0.0" in source


def test_resume_identity_pins_optimization_and_rejects_drift() -> None:
    identity = build_grpo_resume_identity(
        base_checkpoint_sha256="a" * 64,
        train_artifact_sha256="b" * 64,
        validation_artifact_sha256="c" * 64,
        geometry=GRPOBatchGeometry(2, 4, 1, 1, 2),
        parameter_manifest=("x",),
        policy=_policy(),
        segmentation={"segment_length": 256},
        initial_conditioning_mode="none",
        sae_loaded=False,
        kl_beta=0,
        reference_sha256=None,
        optimization={"num_iterations": 100, "weight_decay": 0.0},
    )
    assert_grpo_resume_identity(identity, identity)
    drifted = dict(identity)
    drifted["optimization"] = {"num_iterations": 101, "weight_decay": 0.0}
    with pytest.raises(ValueError, match="resume identity mismatch"):
        assert_grpo_resume_identity(identity, drifted)
    drifted = {
        **identity,
        "parameter_manifest": ["feature_pred_head.weight", "feature_pred_head.bias"],
    }
    with pytest.raises(SFTGRPOError, match="trainable parameter manifest.*same training scope"):
        assert_grpo_resume_identity(identity, drifted)
    with pytest.raises(SFTGRPOError, match="trainable parameter manifest.*same training scope"):
        assert_grpo_resume_identity(drifted, identity)


def test_checkpoint_payload_preserves_base_loader_shape() -> None:
    base = {
        "model_config": {"sequence_len": 16},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
        "feature_prediction_config": {"loss_mode": "max_pool", "block": "mlp", "head_layers": 0},
    }
    meta = trainer._checkpoint_meta(
        base,
        step=3,
        identity={"sha256": "x"},
        train_cursor_envelope={"ddp_world_size": 1, "rank_states": [{}]},
        validation_cursor_envelope={"ddp_world_size": 1, "rank_states": [{}]},
        user_config={"initial_conditioning_mode": "none"},
        initial_conditioning_mode=trainer.InitialConditioningMode.NONE,
        sae_loaded=False,
    )
    assert meta["model_config"] == base["model_config"]
    assert meta["encoder_config"] == base["encoder_config"]
    assert meta[trainer.GRPO_LOOP_META_KEY]["completed_optimizer_step"] == 3
    assert meta[trainer.GRPO_LOOP_META_KEY]["sae_loaded"] is False
    assert meta_looks_encdec(meta)


def test_local_rollout_work_counts_trajectory_expansion() -> None:
    staged = SimpleNamespace(
        environments=(
            SimpleNamespace(targets=(1, 2, 3)),
            SimpleNamespace(targets=(4, 5)),
        )
    )
    assert trainer._local_rollout_work(cast(StagedEnvironmentBatch, staged), group_size=256) == (
        512,
        1280,
    )


def test_deterministic_kernel_mode_is_a_noop_when_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NANOCHAT_ATTN_IMPL": "sdpa",
    }
    monkeypatch.setattr(trainer, "_deterministic_kernel_env", lambda: required)
    for name, value in required.items():
        monkeypatch.setenv(name, value)

    def unexpected_exec(*_args: object) -> None:
        raise AssertionError("execve must not run for the canonical environment")

    monkeypatch.setattr(trainer.os, "execve", unexpected_exec)
    trainer._reexec_for_deterministic_kernels()


def test_deterministic_kernel_mode_reexecs_with_pinned_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NANOCHAT_ATTN_IMPL": "fa4",
    }
    monkeypatch.setattr(trainer, "_deterministic_kernel_env", lambda: required)
    for name in required:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(trainer.sys, "executable", "/python")
    monkeypatch.setitem(trainer.sys.modules, "__main__", SimpleNamespace(__spec__=None))
    monkeypatch.setattr(trainer.sys, "argv", ["trainer.py", "group_size=4"])
    captured: dict[str, object] = {}

    def capture_exec(
        executable: str,
        argv: list[str],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            executable=executable,
            argv=argv,
            environment=environment,
        )

    monkeypatch.setattr(trainer.os, "execve", capture_exec)
    with pytest.raises(trainer.SFTGRPOError, match="re-exec unexpectedly returned"):
        trainer._reexec_for_deterministic_kernels()
    assert captured["executable"] == "/python"
    assert captured["argv"] == ["/python", "trainer.py", "group_size=4"]
    environment = cast(dict[str, str], captured["environment"])
    assert {name: environment[name] for name in required} == required


def test_each_step_resets_non_checkpointed_compiler_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def capture_reset() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(trainer.t.compiler, "reset", capture_reset)
    trainer._reset_step_compiler_state()
    assert calls == 1


def test_checkpoint_reexec_replaces_resume_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer.sys, "executable", "/python")
    monkeypatch.setitem(trainer.sys.modules, "__main__", SimpleNamespace(__spec__=None))
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        ["trainer.py", "group_size=4", "resume_from_step=-1", "save_every=10"],
    )
    assert trainer._checkpoint_reexec_argv(10) == [
        "/python",
        "trainer.py",
        "group_size=4",
        "save_every=10",
        "resume_from_step=10",
    ]
    with pytest.raises(trainer.SFTGRPOError, match="positive completed step"):
        trainer._checkpoint_reexec_argv(0)


@pytest.mark.parametrize(
    "module_name",
    ["scripts.train.conditional_encdec_sft_grpo", "scripts.train.conditional_encdec_rlvr"],
)
def test_worker_reexec_preserves_module_import_path(
    monkeypatch: pytest.MonkeyPatch, module_name: str,
) -> None:
    monkeypatch.setitem(trainer.sys.modules, "__main__", SimpleNamespace(
        __spec__=SimpleNamespace(name=module_name),
    ))
    monkeypatch.setattr(trainer.sys, "executable", "/python")
    monkeypatch.setattr(trainer.sys, "argv", ["/repo/scripts/train/entry.py", "--", "resume_from_step=-1"])
    assert trainer._python_reexec_argv(trainer.sys.argv) == [
        "/python", "-m", module_name, "--", "resume_from_step=-1",
    ]
    assert trainer._checkpoint_reexec_argv(7) == [
        "/python", "-m", module_name, "--", "resume_from_step=7",
    ]


def test_worker_reexec_preserves_file_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(trainer.sys.modules, "__main__", SimpleNamespace(__spec__=None))
    monkeypatch.setattr(trainer.sys, "executable", "/python")
    assert trainer._python_reexec_argv(["/entry.py", "--", "a=1"]) == [
        "/python", "/entry.py", "--", "a=1",
    ]


def test_checkpoint_reexec_synchronizes_cleans_up_and_execs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(trainer.sys, "executable", "/python")
    monkeypatch.setitem(trainer.sys.modules, "__main__", SimpleNamespace(__spec__=None))
    monkeypatch.setattr(trainer.sys, "argv", ["trainer.py", "resume_from_step=-1"])
    monkeypatch.setattr(trainer.t.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer.t.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.t.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: calls.append("cleanup"))

    def capture_exec(
        executable: str,
        argv: list[str],
        environment: dict[str, str],
    ) -> None:
        calls.append((executable, argv, environment))

    monkeypatch.setattr(trainer.os, "execve", capture_exec)
    with pytest.raises(trainer.SFTGRPOError, match="re-exec unexpectedly returned"):
        trainer._reexec_from_checkpoint(7)
    assert calls[:2] == ["barrier", "cleanup"]
    executable, argv, environment = cast(tuple[object, object, object], calls[2])
    assert executable == "/python"
    assert argv == ["/python", "trainer.py", "resume_from_step=7"]
    assert isinstance(environment, dict)


@pytest.mark.parametrize(
    ("attention_impl", "capability", "expected"),
    (
        ("sdpa", (8, 9), True),
        ("fa4", (9, 0), False),
        ("fa4", (10, 0), True),
        ("fa4", (12, 0), False),
        ("fa4", (12, 1), False),
        ("sdpa", (12, 0), True),
    ),
)
def test_cuda_deterministic_mode_matches_attention_backend_and_hardware(
    monkeypatch: pytest.MonkeyPatch,
    attention_impl: str,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    deterministic: list[bool] = []
    precisions: list[str] = []
    monkeypatch.setattr(
        trainer.t,
        "use_deterministic_algorithms",
        deterministic.append,
    )
    monkeypatch.setattr(
        trainer.t,
        "set_float32_matmul_precision",
        precisions.append,
    )
    monkeypatch.setattr(
        trainer.t.cuda,
        "get_device_capability",
        lambda: capability,
    )
    assert (
        trainer._configure_deterministic_algorithms(
            device_type="cuda",
            attention_impl=attention_impl,
        )
        is expected
    )
    assert deterministic == [expected]
    assert precisions == ["highest"]


def test_cpu_deterministic_mode_remains_hard(monkeypatch: pytest.MonkeyPatch) -> None:
    deterministic: list[bool] = []
    monkeypatch.setattr(
        trainer.t,
        "use_deterministic_algorithms",
        deterministic.append,
    )
    assert trainer._configure_deterministic_algorithms(
        device_type="cpu",
        attention_impl="cpu",
    )
    assert deterministic == [True]


def test_attention_backend_resolves_from_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trainer.t.cuda, "is_available", lambda: False)
    assert trainer._target_attention_impl() == "sdpa"
    monkeypatch.setattr(trainer.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", lambda: (8, 9))
    assert trainer._target_attention_impl() == "sdpa"
    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(trainer, "has_fa4_backend", lambda: False)
    with pytest.raises(trainer.SFTGRPOError, match="requires FA4/CuTE"):
        trainer._target_attention_impl()
    monkeypatch.setattr(trainer, "has_fa4_backend", lambda: True)
    assert trainer._target_attention_impl() == "fa4"


@pytest.mark.parametrize("expected", ["fa4", "sdpa"])
def test_attention_backend_identity_is_strict(monkeypatch: pytest.MonkeyPatch, expected: str) -> None:
    assert trainer._assert_attention_impl(device_type="cpu") == "cpu"
    monkeypatch.setattr(trainer, "_target_attention_impl", lambda: expected)
    installed = []
    monkeypatch.setattr(trainer, "admit_project_attention", lambda **_kwargs: installed.append(True))
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", "sdpa" if expected == "fa4" else "fa4")
    with pytest.raises(trainer.SFTGRPOError, match=f"requires attention backend '{expected}'"):
        trainer._assert_attention_impl(device_type="cuda")
    assert installed == []
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", expected)
    assert trainer._assert_attention_impl(device_type="cuda") == expected
    assert installed == ([True] if expected == "fa4" else [])


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_sm12_selects_native_fa4_without_probing_block_sparse_fa4(monkeypatch, capability):
    monkeypatch.setattr(trainer.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(trainer, "has_fa4_backend", lambda: pytest.fail("SM12 must not probe block-sparse FA4"))
    installed = []
    monkeypatch.setattr(trainer, "admit_project_attention", lambda **_kwargs: installed.append(True))
    assert trainer._target_attention_impl() == "fa4"
    assert trainer._deterministic_kernel_env()["NANOCHAT_ATTN_IMPL"] == "fa4"
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", "fa4")
    monkeypatch.setattr(sm12_fa4.flash_attention, "_override_impl", sm12_fa4.flash_attention.FlashAttnImpl.FA4)
    assert trainer._assert_attention_impl(device_type="cuda") == "fa4"
    assert installed == [True]
    monkeypatch.setattr(trainer, "admit_project_attention", sm12_fa4.admit_project_attention)
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    with pytest.raises(trainer.SFTGRPOError, match="requires native FA4/CuTe"):
        trainer._assert_attention_impl(device_type="cuda")


def test_main_cleans_up_when_native_fa4_admission_fails(monkeypatch):
    cfg = _compose("conditional_encdec_sft_grpo", [
        "base_checkpoint_dir=/unused", "train_artifact_dir=/unused", "validation_artifact_dir=/unused",
        "global_conversations_per_step=1", "group_size=2", "device_conversation_batch_size=1",
        "trajectory_microbatch_size=1", "num_iterations=1", "device_type=cuda",
    ])
    cleaned = []
    monkeypatch.setattr(trainer, "_reexec_for_deterministic_kernels", lambda: None)
    monkeypatch.setattr(trainer.sys, "argv", ["conditional_encdec_sft_grpo.py"])
    monkeypatch.setattr(trainer, "compose", lambda **_kwargs: cfg)
    monkeypatch.setattr(trainer, "compute_init_for_checkpoint_resume", lambda *_args, **_kwargs: (True, 0, 0, 1, "cuda"))
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: cleaned.append(True))
    monkeypatch.setattr(trainer, "load_checkpoint", lambda *_args, **_kwargs: pytest.fail("loaded model before admission"))
    monkeypatch.setattr(trainer.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", "fa4")
    monkeypatch.setattr(sm12_fa4.flash_attention, "_override_impl", sm12_fa4.flash_attention.FlashAttnImpl.FA4)
    with pytest.raises(trainer.SFTGRPOError, match="requires native FA4/CuTe"):
        trainer.main()
    assert cleaned == [True]


def _online_sae_base_meta(
    feature_root: Path,
    *,
    create: bool,
    mode: SegmentMode = SegmentMode.turn_delimited_fixed_tokens,
) -> dict[str, object]:
    """Build a conditioned-SFT base meta carrying an online-SAE lineage.

    ``create`` mints the canonical feature tree and statistics artifact so
    ``validate_paths()`` would succeed; otherwise the recorded training-host
    tree is absent, which is the ordinary remote-staging shape.
    """
    feature_dir = build_feature_dir(
        root=feature_root,
        segment_size=256,
        layer=22,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_65K,
        model="gemma-3-4b-pt",
        dataset_phase=DatasetPhase.STANDARD_SFT,
        context_independent_features=True,
        ignore_bos_features=True,
        segment_mode=mode,
    )
    statistics_path = feature_dir / "feature_statistics.safetensors"
    feature_dir.mkdir(parents=True)
    statistics_path.write_bytes(b"pinned-statistics")
    lineage = build_online_sae_lineage_metadata(
        feature_dir=str(feature_dir),
        statistics_path=statistics_path,
        segment_mode=mode,
        sae_config={
            "model": "gemma-3-4b-pt",
            "release": "gemmascope-2",
            "activation_type": "residual",
            "layer": 22,
            "width": "65k",
        },
        feature_selection={"mode": "top_k_percentile"},
        negative_selection=None,
        missing_percentile_policy="error",
        n_sae_features=65_536,
        num_buckets=2,
        selection_seed=42,
    )
    if not create:
        shutil.rmtree(feature_root)
    meta: dict[str, object] = {
        SFT_SEGMENT_MODE_META_KEY: mode.value,
        "model_config": {"sequence_len": 256},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
    }
    meta.update(lineage)
    return meta


def _drive_main_to_initial_conditioning(
    monkeypatch: pytest.MonkeyPatch,
    base_meta: dict[str, object],
    *,
    resume_step: int = -1,
) -> None:
    """Run the real ``main()`` up to initial-conditioning construction.

    Only checkpoint/model/optimizer/tokenizer construction is stubbed; the
    online-SAE lineage parse and its policy gate stay live.
    """

    class _FakeOptimizer:
        pass

    monkeypatch.setattr(trainer, "_reexec_for_deterministic_kernels", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_configure_deterministic_algorithms",
        lambda **_kwargs: True,
    )
    def init(_device: str, *, resume_step: int) -> tuple[None, int, int, int, str]:
        assert resume_step == expected_resume_step
        return None, 0, 0, 1, "cpu"

    expected_resume_step = resume_step
    monkeypatch.setattr(trainer, "compute_init_for_checkpoint_resume", init)
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: None)
    monkeypatch.setattr(
        trainer,
        "load_checkpoint",
        lambda *_args, **_kwargs: ({}, None, base_meta),
    )
    monkeypatch.setattr(trainer, "_resolved_checkpoint_sha256", lambda *_a: "a" * 64)
    monkeypatch.setattr(trainer, "_build_model", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        trainer,
        "validate_grpo_checkpoint_eligibility",
        lambda *_a, **_k: SimpleNamespace(policy=_policy(), feature_contract=object()),
    )
    monkeypatch.setattr(
        trainer,
        "configure_grpo_trainable_parameters",
        lambda *_a, **_k: ("x",),
    )
    monkeypatch.setattr(trainer, "DistMuonAdamW", _FakeOptimizer)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda *_a, **_k: _FakeOptimizer())
    monkeypatch.setattr(
        trainer,
        "get_tokenizer",
        lambda: SimpleNamespace(decode=lambda ids: ""),
    )
    monkeypatch.setattr(trainer, "rustbpe_tokenizer_sha256", lambda _t: "b" * 64)

    def _unreachable(*_args: object, **_kwargs: object) -> None:
        pytest.fail("main() must fail before online bootstrap in this harness")

    monkeypatch.setattr(trainer, "build_online_selection_context_factory", _unreachable)
    monkeypatch.setattr(trainer, "OnlineSAEInitialConditioningProvider", _unreachable)
    monkeypatch.setattr(trainer, "load_structured_rollout_artifact", _unreachable)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "")
    monkeypatch.setattr(
        trainer.sys,
        "argv",
        [
            "trainer.py",
            "base_checkpoint_dir=/base",
            "train_artifact_dir=/train",
            "validation_artifact_dir=/val",
            "model_step=0",
            "global_conversations_per_step=1",
            "group_size=2",
            "device_conversation_batch_size=1",
            "trajectory_microbatch_size=1",
            "num_iterations=1",
            f"resume_from_step={resume_step}",
            "initial_conditioning_mode=online_sae",
            "sae_loaded=true",
            "device_type=cpu",
        ],
    )
    trainer.main()


# kills: deleting `online_lineage.assert_grpo_supported()` from main(), or
# moving validate_paths() ahead of that gate. The recorded training-host tree
# is absent here, so that ordering reports FileNotFoundError on a path instead
# of the fixed-only policy error.
def test_online_sae_turn_rejection_precedes_training_host_path_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_meta = _online_sae_base_meta(tmp_path / "absent", create=False)
    assert not (tmp_path / "absent").exists()

    with pytest.raises(
        trainer.SFTGRPOError,
        match=r"supports fixed_tokens.*turn_delimited_fixed_tokens",
    ):
        _drive_main_to_initial_conditioning(monkeypatch, base_meta, resume_step=7)


# kills: dropping `online_lineage.validate_paths()` from main() after the policy
# gate. A fixed base passes the gate; with its recorded training-host tree
# absent, the run must stop on the missing statistics path rather than reach
# online bootstrap.
def test_online_sae_fixed_base_validates_training_host_paths_after_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_meta = _online_sae_base_meta(
        tmp_path / "absent",
        create=False,
        mode=SegmentMode.fixed_tokens,
    )
    lineage = cast(dict[str, object], base_meta[ONLINE_SAE_LINEAGE_KEY])
    absent_root = tmp_path / "absent"
    assert str(lineage["statistics_path"]).startswith(str(absent_root))
    assert not absent_root.exists()

    # validate_paths() is the only step that touches the filesystem; strict
    # resolution names the first missing ancestor, i.e. the absent root.
    with pytest.raises(FileNotFoundError, match=re.escape(str(absent_root))):
        _drive_main_to_initial_conditioning(monkeypatch, base_meta)
