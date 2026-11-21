"""End-to-end integration test: drive training_progression's collect-loop
with auto-resolve target_real_configs (cache_dir missing) and a non-1024
context model. Verifies that:

1. ``parse_target_real_configs_from_cfg`` is called with
   ``model_max_seq_len=int(model.dec_config.sequence_len)`` (the bug fix
   from PR-E — previously this kwarg didn't exist and the auto-resolver
   defaulted to 1024).
2. The auto-resolved cache_dirs contain ``c{seq_len}/`` matching the
   model's actual context length, NOT the legacy 1024 fallback.
3. ``_target_real_providers_for_cfg`` receives the same model_max_seq_len
   so the cache_dir and provider's expected_metadata stay consistent.

This is the load-bearing e2e test for the PR-E latent-bug fix. A
regression here would manifest as a confusing cache-validation error at
provider load time (cache says model_max_seq_len=2048, provider expects
1024). This test catches it pre-merge by inspecting the call kwargs
flowing through the production collect-loop.
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
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer


def _make_cfg_with_auto_resolve(
    target_real_configs: list[dict] | None = None,
):
    """Build a minimal cfg whose target_real_configs LACK explicit cache_dirs
    so the auto-resolver fires. This exercises the code path that PR-E
    fixed (model_max_seq_len plumbing through parse_target_real_configs_from_cfg)."""
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
                "model_id": "gemma-2-2b",
                "sae_id": "gemmascope-residual-l8-16k",
                "layer": 8,
                "pooling": "max_pool",
                "feature_selection_hash": "f" * 64,
            },
            "n_sae_features": 64,
            "num_buckets": 4,
            "context_cache_dir": None,
            "conditioned_configs": {},
            "target_real_configs": target_real_configs or [],
        },
    })


@pytest.fixture()
def fake_base_dir(tmp_path, monkeypatch):
    """Redirect get_base_dir so auto-resolved paths land under tmp_path."""
    import steering.eval.eval_cache_paths as paths_mod
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    return str(tmp_path)


@pytest.fixture()
def model_with_seq_len_2048():
    """Build the standard tiny encdec, then monkey-patch dec_config.sequence_len
    to 2048 so the test exercises a non-default context length without
    paying the cost of building a full 2048-context model."""
    model = build_tiny_encdec()
    # The production code reads model.dec_config.sequence_len; override it
    # post-construction. dec_config is a dataclass; __setattr__ may be
    # frozen, so we bypass via __dict__.
    object.__setattr__(model.dec_config, "sequence_len", 2048)
    return model


@pytest.fixture()
def stub_load_progression_model_with_seq_len_2048(model_with_seq_len_2048):
    """Stub model load to return the seq_len=2048 model."""
    model = model_with_seq_len_2048
    tokenizer = build_tiny_tokenizer()
    meta = {"model_type": "encoder_decoder", "checkpoint_dir": "/tmp"}

    def _stub(checkpoint_dir, step, device, cfg):
        return model, tokenizer, meta, CheckpointFeatureContract(None, None)

    with patch.object(mod, "load_progression_model", side_effect=_stub):
        with patch.object(mod, "resolve_local_checkpoint_dir", return_value="/tmp"):
            yield model, tokenizer


# ---------------------------------------------------------------------------
# E2E: collect-loop with auto-resolve + 2048-context model
# ---------------------------------------------------------------------------


def test_collect_loop_threads_seq_len_through_auto_resolve(
    fake_base_dir, stub_load_progression_model_with_seq_len_2048,
):
    """Drive the full collect-loop with auto-resolve target_real_configs
    and a 2048-context model. Capture the kwargs passed to
    parse_target_real_configs_from_cfg and assert model_max_seq_len=2048."""
    cfg = _make_cfg_with_auto_resolve(
        target_real_configs=[
            {"name": "pooled", "mode": "pooled", "pooling_op": "bucket_max"},
            {"name": "lockstep", "mode": "lockstep"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    # Capture call kwargs while letting the real parser run.
    real_parse = mod.parse_target_real_configs_from_cfg
    captured_parse_calls: list[dict] = []

    def _spy_parse(cfg_arg, **kwargs):
        captured_parse_calls.append({"cfg": cfg_arg, **kwargs})
        return real_parse(cfg_arg, **kwargs)

    # Capture _target_real_providers_for_cfg kwargs too for the consistency
    # invariant (both calls must use the same model_max_seq_len).
    captured_provider_calls: list[dict] = []

    def _spy_providers(cfg_arg, *, model_max_seq_len, suite=None,
                       target_real_configs=None):
        captured_provider_calls.append({
            "model_max_seq_len": model_max_seq_len,
            "suite": suite,
            "target_real_configs": target_real_configs,
        })
        return {"pooled": MagicMock(), "lockstep": MagicMock()}

    def _fake_run_target_real(*, eval_name, model, tokenizer, device, seed,
                              cfg, trc, provider, target_mode):
        return {"core_metric": 0.5, "results": {}, "centered_results": {}}

    with patch.object(mod, "parse_target_real_configs_from_cfg", side_effect=_spy_parse):
        with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_spy_providers):
            with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake_run_target_real):
                _evaluate_single_checkpoint_impl(
                    spec, torch.device("cpu"), torch.float32,
                    ["core"], seeds, cfg, existing_entry=None, work_keys=None,
                )

    # Invariant 1: parse_target_real_configs_from_cfg called with seq_len=2048.
    assert len(captured_parse_calls) == 1, (
        f"parse_target_real_configs_from_cfg should be called exactly once "
        f"per checkpoint; got {len(captured_parse_calls)} calls"
    )
    parse_kwargs = captured_parse_calls[0]
    assert parse_kwargs["model_max_seq_len"] == 2048, (
        f"PR-E bug fix: production must thread "
        f"model.dec_config.sequence_len (=2048) to "
        f"parse_target_real_configs_from_cfg, not the default 1024. "
        f"Got: {parse_kwargs['model_max_seq_len']}"
    )
    assert parse_kwargs["suite"] == "sft_target", (
        "suite must be derived from spec.phase ('sft') + '_target'"
    )

    # Invariant 2: _target_real_providers_for_cfg called with same seq_len.
    # If the two disagree, cache_dir resolution will not match the provider's
    # expected_metadata and downstream cache validation will fail.
    assert len(captured_provider_calls) == 1
    assert captured_provider_calls[0]["model_max_seq_len"] == 2048, (
        "model_max_seq_len divergence between parse + provider calls — "
        "cache identity will not validate at provider load time"
    )

    # Invariant 3: the parsed configs returned to production have cache_dirs
    # under c2048/, NOT c1024/. This is the operator-visible bug fix.
    parsed_configs = captured_provider_calls[0]["target_real_configs"]
    assert parsed_configs is not None
    for trc in parsed_configs:
        cd = str(trc.cache_dir)
        assert "/c2048/" in cd, (
            f"Auto-resolved cache_dir for {trc.name!r} did not include "
            f"/c2048/: {cd}. The parser failed to thread the model's "
            f"sequence_len through to default_target_cache_dir."
        )
        assert "/c1024/" not in cd, (
            f"Regression: 2048-context model resolved to /c1024/ "
            f"({trc.name!r} -> {cd})"
        )


def test_collect_loop_with_explicit_cache_dir_ignores_seq_len(
    fake_base_dir, stub_load_progression_model_with_seq_len_2048,
):
    """When all target_real_configs entries have explicit cache_dir,
    the auto-resolver does not need to consult model_max_seq_len. Verify
    that explicit cache_dirs flow through verbatim and the 2048-context
    model doesn't accidentally rewrite them."""
    explicit = "/operator/supplied/cache_dir/c1024"
    cfg = _make_cfg_with_auto_resolve(
        target_real_configs=[
            {"name": "explicit", "mode": "pooled",
             "cache_dir": explicit, "pooling_op": "bucket_max"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    captured_provider_calls: list[dict] = []

    def _spy_providers(cfg_arg, *, model_max_seq_len, suite=None,
                       target_real_configs=None):
        captured_provider_calls.append({
            "model_max_seq_len": model_max_seq_len,
            "target_real_configs": target_real_configs,
        })
        return {"explicit": MagicMock()}

    def _fake_run(*, eval_name, model, tokenizer, device, seed, cfg, trc,
                  provider, target_mode):
        return {"core_metric": 0.5}

    with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_spy_providers):
        with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake_run):
            _evaluate_single_checkpoint_impl(
                spec, torch.device("cpu"), torch.float32,
                ["core"], seeds, cfg, existing_entry=None, work_keys=None,
            )

    # The provider call still receives model_max_seq_len=2048 (consistency
    # invariant — it MAY use it for the provider's expected_metadata).
    assert captured_provider_calls[0]["model_max_seq_len"] == 2048
    # But the explicit cache_dir flows through unchanged.
    parsed = captured_provider_calls[0]["target_real_configs"]
    assert str(parsed[0].cache_dir) == explicit, (
        "Explicit cache_dir must not be rewritten by the auto-resolver"
    )
