"""PR-D: resume-from-state skips already-completed target_real seeds.

Behavior tests (PR-D Batch H2): drive ``_evaluate_single_checkpoint_impl``
through the production code path with a pre-populated existing entry and
assert the seed-skip semantics. The bypass-detection version of these
tests lives in ``test_progression_collect_loop_target_real.py``; this file
focuses on the resume-specific edge cases (mid-write recovery, multiple
trcs, full vs partial resume).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import scripts.train.training_progression as mod
from scripts.train.training_progression import (
    CheckpointSpec,
    _evaluate_single_checkpoint_impl,
)
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.eval.progression_results import SuiteResultKind
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer


def _make_cfg(target_real_configs):
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
        "num_seeds": 3,
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
            "conditioned_configs": {},
            "target_real_configs": target_real_configs,
        },
    })


@pytest.fixture()
def stub_load_progression_model():
    model = build_tiny_encdec()
    tokenizer = build_tiny_tokenizer()
    meta = {"model_type": "encoder_decoder", "checkpoint_dir": "/tmp"}

    def _stub(checkpoint_dir, step, device, cfg):
        return model, tokenizer, meta, CheckpointFeatureContract(None, None)

    with patch.object(mod, "load_progression_model", side_effect=_stub):
        with patch.object(mod, "resolve_local_checkpoint_dir", return_value="/tmp"):
            yield model, tokenizer


@pytest.fixture()
def stub_target_real_provider():
    providers: dict[str, MagicMock] = {}

    def _stub(cfg, *, model_max_seq_len, suite=None, target_real_configs=None):
        # H7 fix: production now passes parsed configs directly.
        if target_real_configs is not None:
            for trc in target_real_configs:
                providers[str(trc.name)] = MagicMock()
            return providers
        cond = cfg.get("conditioning", {})
        for trc in list(cond.get("target_real_configs", []) or []):
            providers[str(trc["name"])] = MagicMock()
        return providers

    with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_stub):
        yield providers


# ---------------------------------------------------------------------------
# Behavior: pre-populated seed dict -> only missing seed runs.
# ---------------------------------------------------------------------------


def test_resume_skips_completed_seed_runs_only_missing(
    stub_load_progression_model, stub_target_real_provider,
):
    """Pre-populated results with seed 1234 done; collect runs only seed 2234."""
    cfg = _make_cfg([
        {"name": "pooled1", "mode": "pooled",
         "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
    ])
    cfg.num_seeds = 2
    seeds = [1234, 2234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    existing_entry = {
        "step": 0, "phase": "sft",
        "evals": {
            "core": {
                "target_real": {
                    "configs": {
                        "pooled1": {
                            "kind": SuiteResultKind.TARGET_REAL_POOLED.value,
                            "cache_dir": "/tmp/c1",
                            "seeds": {"1234": {"core_metric": 0.99}},
                        },
                    },
                },
            },
        },
    }

    seen_seeds: list[int] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        seen_seeds.append(int(seed))
        return {"core_metric": 0.42}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg,
            existing_entry=existing_entry, work_keys=None,
        )

    assert seen_seeds == [2234]
    cfg_result = evals["core"]["target_real"]["configs"]["pooled1"]
    assert cfg_result["seeds"] == {
        "1234": {"core_metric": 0.99},
        "2234": {"core_metric": 0.42},
    }


# ---------------------------------------------------------------------------
# Behavior: empty seed dict -> all seeds run.
# ---------------------------------------------------------------------------


def test_resume_empty_results_runs_all_seeds(
    stub_load_progression_model, stub_target_real_provider,
):
    """No existing entry -> all configured seeds run."""
    cfg = _make_cfg([
        {"name": "pooled1", "mode": "pooled",
         "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
    ])
    cfg.num_seeds = 3
    seeds = [1234, 2234, 3234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    seen: list[int] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        seen.append(int(seed))
        return {"core_metric": 0.42}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,
        )

    assert sorted(seen) == sorted(seeds)


# ---------------------------------------------------------------------------
# Behavior: multiple trcs, partial resume on one — independent skip per trc.
# ---------------------------------------------------------------------------


def test_resume_independent_per_trc(
    stub_load_progression_model, stub_target_real_provider,
):
    """Two trcs: trc_a has seed 1234 done, trc_b is empty.
    Only trc_a/2234 + trc_b/{1234,2234} run."""
    cfg = _make_cfg([
        {"name": "trc_a", "mode": "pooled",
         "cache_dir": "/tmp/a", "pooling_op": "bucket_max"},
        {"name": "trc_b", "mode": "lockstep", "cache_dir": "/tmp/b"},
    ])
    cfg.num_seeds = 2
    seeds = [1234, 2234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    existing_entry = {
        "step": 0, "phase": "sft",
        "evals": {
            "core": {
                "target_real": {
                    "configs": {
                        "trc_a": {
                            "kind": SuiteResultKind.TARGET_REAL_POOLED.value,
                            "cache_dir": "/tmp/a",
                            "seeds": {"1234": {"core_metric": 0.5}},
                        },
                    },
                },
            },
        },
    }

    seen: list[tuple[str, int]] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        seen.append((trc.name, int(seed)))
        return {"core_metric": 0.0}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg,
            existing_entry=existing_entry, work_keys=None,
        )

    assert sorted(seen) == sorted([
        ("trc_a", 2234),
        ("trc_b", 1234),
        ("trc_b", 2234),
    ])
    cfg_a = evals["core"]["target_real"]["configs"]["trc_a"]
    cfg_b = evals["core"]["target_real"]["configs"]["trc_b"]
    assert set(cfg_a["seeds"].keys()) == {"1234", "2234"}
    assert set(cfg_b["seeds"].keys()) == {"1234", "2234"}
