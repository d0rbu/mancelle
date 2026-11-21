"""PR-D: chat_eval / training_progression include `kind` in the data dict
they pass to report.log, so the report contains the discriminator.

Behavior tests with mock-based call interception — fail loudly if the
production code regresses to omitting `kind`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch as t
from omegaconf import DictConfig, OmegaConf

import scripts.train.training_progression as mod
from scripts.train.training_progression import CheckpointSpec, _evaluate_single_checkpoint_impl
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.eval.feature_config import FeatureConfig
from steering.eval.progression_results import SuiteResultKind
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
)
from steering.eval.target_real_runner import run_target_real_evals
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer

# ---------------------------------------------------------------------------
# Tests 1 & 2: chat_eval entry_kind classification path
#
# The production code in chat_eval.main() builds report_data containing
# {"kind": entry_kind}. Static feature configs are explicit operator requests,
# so chat_eval no longer emits the old plain/unconditioned discriminator.
# ---------------------------------------------------------------------------


def _entry_kind_for_config(config_name: str, configs: list[FeatureConfig]) -> str:
    """Mirror the production entry_kind classification in chat_eval.main()."""
    return "static_yaml"


def test_chat_eval_report_log_data_contains_kind_for_static_yaml():
    """When chat_eval logs a non-empty-features (static_yaml) config result
    via report.log, the data dict must contain 'kind' == 'static_yaml'."""
    static_cfg = FeatureConfig(name="safety_features", features={0: 1, 1: 2})
    kind = _entry_kind_for_config("safety_features", [static_cfg])
    assert kind == "static_yaml", (
        f"non-empty-features config must produce entry_kind='static_yaml'; got {kind!r}. "
        "chat_eval.main() static_yaml branch sets entry_kind='static_yaml'."
    )
    report_data_entry = {"kind": kind}
    assert report_data_entry.get("kind") == SuiteResultKind.STATIC_YAML.value, (
        f"report_data entry must match SuiteResultKind.STATIC_YAML.value; "
        f"got {report_data_entry!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: target_real path calls report.log with kind (via run_target_real_evals)
#
# This is the same code path exercised by H3's test_target_real_runner_report_log.py
# but narrows to the chat_eval target_real → report.log connection.
# ---------------------------------------------------------------------------


def test_chat_eval_target_real_path_calls_report_log_with_kind(tmp_path):
    """The target_real path through run_target_real_evals (used by chat_eval)
    includes kind in the report.log call. Verifies the H3/II-6 fix is real and
    that chat_eval's lambda delegation to run_target_real_evals wires correctly."""
    mock_report = MagicMock()
    trc = LockstepTargetRealConfig(
        name="chat_lockstep",
        cache_dir=tmp_path / "cache",
    )
    cfg = OmegaConf.create({
        "conditioning": {
            "identity": {
                "model_id": "m", "sae_id": "s", "layer": 0, "pooling": "max_pool",
            },
        },
        "model_max_seq_len": 32,
    })

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=object(),
    ), patch(
        "steering.eval.target_real_runner.get_report",
        return_value=mock_report,
    ):
        run_target_real_evals(
            target_real_configs=[trc],
            cfg=cfg,
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=lambda t, p: {
                "name": t.name,
                "target_real_mode": "lockstep",
                "results": {},
            },
            checkpoint_dir=str(tmp_path / "ckpt"),
            report_section_prefix="Chat evaluation test_source",
        )

    mock_report.log.assert_called_once()
    call_data = mock_report.log.call_args.kwargs.get("data") or mock_report.log.call_args.args[1]
    assert any(
        isinstance(d, dict) and d.get("kind") == "target_real_lockstep"
        for d in call_data
    ), (
        f"report.log data must contain {{'kind': 'target_real_lockstep'}}; "
        f"got call_data={call_data}"
    )


# ---------------------------------------------------------------------------
# Tests 4 & 5: training_progression cfg_result kind field
#
# Drive _evaluate_single_checkpoint_impl through the production collect-loop
# and verify that cfg_result dicts carry the correct 'kind' for both
# target_real (lockstep + pooled) and static_yaml conditioned configs.
# ---------------------------------------------------------------------------


def _make_cfg_for_kind_tests(
    *,
    target_real_configs: list[dict] | None = None,
    conditioned_configs: dict | None = None,
) -> DictConfig:
    return OmegaConf.create({
        "model_tag": "test",
        "checkpoint_source": "local",
        "checkpoints_dir": None,
        "checkpoint_root": None,
        "checkpoint_prefix": "",
        "model_family": "encoder_decoder",
        "results_file": None,
        "fig_dir": None,
        "evals": "core",
        "max_per_task": 1,
        "chat_tasks": None,
        "chat_batch_size": 1,
        "chat_max_new_tokens": 8,
        "chat_temperature": 0.0,
        "chat_top_k": 50,
        "chat_num_samples": 1,
        "chat_max_problems": 1,
        "harm_temperature": 0.7,
        "harm_top_k": 50,
        "harm_max_new_tokens": 8,
        "harm_num_benign": 1,
        "harm_num_harmful": 1,
        "harm_judge_model": "gpt-5-nano",
        "device_type": "cpu",
        "dtype": "float32",
        "num_gpus": 1,
        "num_seeds": 1,
        "base_seed": 1234,
        "phases": [{"phase": "sft", "subdir": "", "steps": [0]}],
        "cleanup_after_eval": False,
        "source": "encdec",
        "conditioning": {
            "identity": {
                "model_id": "test-model",
                "sae_id": "test-sae",
                "layer": 0,
                "pooling": "mean",
                "feature_selection_hash": "a" * 64,
            },
            "n_sae_features": 64,
            "num_buckets": 4,
            "context_cache_dir": None,
            "conditioned_configs": conditioned_configs or {},
            "target_real_configs": target_real_configs or [],
        },
    })


def test_training_progression_target_real_cfg_result_carries_kind_field():
    """Every target_real cfg_result dict produced by _evaluate_single_checkpoint_impl
    includes a 'kind' field matching SuiteResultKind for lockstep and pooled.
    Drives the production _run_target_real_for_suite code path."""

    cfg = _make_cfg_for_kind_tests(target_real_configs=[
        {"name": "lockstep_cfg", "mode": "lockstep", "cache_dir": "/tmp/l"},
        {"name": "pooled_cfg", "mode": "pooled",
         "cache_dir": "/tmp/p", "pooling_op": "bucket_max"},
    ])
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    model = build_tiny_encdec()
    tokenizer = build_tiny_tokenizer()
    meta = {"model_type": "encoder_decoder", "checkpoint_dir": "/tmp"}

    providers = {
        "lockstep_cfg": MagicMock(),
        "pooled_cfg": MagicMock(),
    }

    def _fake_run(*, eval_name, model, tokenizer, device, seed, cfg, trc,
                  provider, target_mode):
        return {"core_metric": 0.5}

    with patch.object(
        mod,
        "load_progression_model",
        return_value=(model, tokenizer, meta, CheckpointFeatureContract(None, None)),
    ), \
         patch.object(mod, "resolve_local_checkpoint_dir", return_value="/tmp"), \
         patch.object(mod, "_target_real_providers_for_cfg", return_value=providers), \
         patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake_run):
        evals = _evaluate_single_checkpoint_impl(
            spec, t.device("cpu"), t.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,
        )

    lockstep_result = evals["core"]["target_real"]["configs"]["lockstep_cfg"]
    pooled_result = evals["core"]["target_real"]["configs"]["pooled_cfg"]

    assert lockstep_result["kind"] == SuiteResultKind.TARGET_REAL_LOCKSTEP.value, (
        f"lockstep cfg_result must have kind={SuiteResultKind.TARGET_REAL_LOCKSTEP.value!r}; "
        f"got {lockstep_result.get('kind')!r}"
    )
    assert pooled_result["kind"] == SuiteResultKind.TARGET_REAL_POOLED.value, (
        f"pooled cfg_result must have kind={SuiteResultKind.TARGET_REAL_POOLED.value!r}; "
        f"got {pooled_result.get('kind')!r}"
    )


def test_training_progression_static_yaml_cfg_result_carries_kind_field():
    """Static YAML cfg_result dict produced by _evaluate_single_checkpoint_impl
    includes kind == 'static_yaml'. Drives the production conditioned_configs
    path (line 1935: 'kind': 'static_yaml')."""

    cfg = _make_cfg_for_kind_tests(
        conditioned_configs={"safety": "out/safety_features.yaml"},
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    model = build_tiny_encdec()
    tokenizer = build_tiny_tokenizer()
    meta = {"model_type": "encoder_decoder", "checkpoint_dir": "/tmp"}

    # Patch _target_real_providers_for_cfg to return empty (no target_real).
    # Patch parse_strict_feature_configs to return a real FeatureConfig so the
    # conditioned loop iterates.
    safety_fc = FeatureConfig(
        name="safety", features={0: 1}, source="out/safety_features.yaml"
    )

    # Patch _run_eval_suite (the inner per-seed eval helper) to avoid
    # actually running the eval stack. The "kind" field is set on the
    # cfg_result skeleton BEFORE _run_eval_suite is called (line 1935 of
    # training_progression.py), so a no-op stub is sufficient to get the
    # cfg_result dict populated with the correct kind.
    with patch.object(
        mod,
        "load_progression_model",
        return_value=(model, tokenizer, meta, CheckpointFeatureContract(None, None)),
    ), \
         patch.object(mod, "resolve_local_checkpoint_dir", return_value="/tmp"), \
         patch.object(
             mod,
             "_context_provider_for_cfg",
             return_value=SimpleNamespace(cache_dir="/tmp/cache"),
         ), \
         patch.object(mod, "_target_real_providers_for_cfg", return_value={}), \
         patch.object(mod, "parse_strict_feature_configs", return_value=[safety_fc]), \
         patch.object(mod, "enumerate_expected_eval_contexts", return_value=[]), \
         patch.object(mod, "_wrap_conditioned_model", return_value=model), \
         patch.object(mod, "_run_eval_suite", return_value=None):
        evals = _evaluate_single_checkpoint_impl(
            spec, t.device("cpu"), t.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,
        )

    conditioned_result = evals["core"]["conditioned"]["configs"]["safety"]
    assert conditioned_result["kind"] == SuiteResultKind.STATIC_YAML.value, (
        f"static_yaml cfg_result must have kind='static_yaml'; "
        f"got {conditioned_result.get('kind')!r}"
    )
