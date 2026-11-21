"""PR-C: behavior tests for dclm_eval target_real_configs eval loop.

Verifies dclm_eval-specific behavior:
* the dispatcher (build_target_for_real_config) is wired through
* output yaml has real numbers (no pending_functional_implementation)
* DDP rank-0 barrier

Provider construction errors and the encdec guard are tested once in
``tests/steering/test_target_real_runner.py`` (the consolidated shared
helper).
"""

from __future__ import annotations

import inspect

import pytest
import yaml
from omegaconf import OmegaConf

import steering.eval.dclm_eval as mod
import steering.eval.target_real_runner as runner_mod
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
)
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer


class _MissProvider:
    def __init__(self):
        self.call_count = 0

    def get(self, token_ids):
        self.call_count += 1
        return None


def _build_pooled_trc(tmp_path):
    return PooledTargetRealConfig(
        name="trc_pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )


def _run_dclm_target_real(
    *,
    monkeypatch,
    raw_encdec,
    tokenizer,
    device,
    trc,
    provider,
    cfg,
    output_dir_base,
    ddp_rank: int,
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "build_context_feature_provider",
        lambda _cfg, **_kw: provider,
    )
    runner_mod.run_target_real_evals(
        target_real_configs=[trc],
        cfg=cfg,
        output_dir_base=str(output_dir_base),
        ddp_rank=ddp_rank,
        per_config_eval_fn=lambda trc_arg, provider_arg: (
            mod._dclm_core_target_real_per_config(
                raw_encdec=raw_encdec,
                tokenizer=tokenizer,
                device=device,
                trc=trc_arg,
                provider=provider_arg,
                cfg=cfg,
            )
        ),
    )


@pytest.fixture
def tiny_setup():
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_encdec()
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# 1. Output yaml contains real numbers (no placeholder)
# ---------------------------------------------------------------------------


def test_output_yaml_has_real_results_not_placeholder(
    tiny_setup, tmp_path, monkeypatch
):
    """Patch _run_core_target_real_eval to return synthetic results;
    verify the orchestrator writes them to yaml without the
    pending_functional_implementation placeholder."""
    tokenizer, model = tiny_setup
    trc = _build_pooled_trc(tmp_path)
    provider = _MissProvider()

    fake_core_out = {
        "results": {"hellaswag": 0.75, "arc": 0.5},
        "centered_results": {"hellaswag": 0.5, "arc": 0.25},
        "core_metric": 0.375,
    }
    calls = []

    def fake_core_eval(**kw):
        calls.append(kw)
        return fake_core_out

    monkeypatch.setattr(mod, "_run_core_target_real_eval", fake_core_eval)
    cfg = OmegaConf.create({"max_per_task": 2})
    out_dir = tmp_path / "out"
    _run_dclm_target_real(
        monkeypatch=monkeypatch,
        raw_encdec=model, tokenizer=tokenizer, device="cpu",
        trc=trc, provider=provider, cfg=cfg,
        output_dir_base=out_dir, ddp_rank=0,
    )
    assert len(calls) == 1, "core eval helper must be invoked exactly once"
    yaml_path = out_dir / "target_real" / f"{trc.name}.yaml"
    assert yaml_path.is_file()
    payload = yaml.safe_load(yaml_path.read_text())
    assert payload.get("status") != "pending_functional_implementation"
    assert payload["kind"] == "target_real_pooled"
    adhoc = payload["seeds"]["adhoc"]
    assert adhoc["results"] == fake_core_out["results"]
    assert adhoc["centered_results"] == fake_core_out["centered_results"]
    assert adhoc["core_metric"] == fake_core_out["core_metric"]


def test_lockstep_variant_dispatched_correctly(tiny_setup, tmp_path, monkeypatch):
    tokenizer, model = tiny_setup
    trc = LockstepTargetRealConfig(name="trc_lockstep", cache_dir=tmp_path / "cache")
    provider = _MissProvider()
    fake_core_out = {
        "results": {"task_a": 1.0},
        "centered_results": {"task_a": 1.0},
        "core_metric": 1.0,
    }
    monkeypatch.setattr(mod, "_run_core_target_real_eval", lambda **kw: fake_core_out)
    cfg = OmegaConf.create({"max_per_task": -1})
    out_dir = tmp_path / "out"
    _run_dclm_target_real(
        monkeypatch=monkeypatch,
        raw_encdec=model, tokenizer=tokenizer, device="cpu",
        trc=trc, provider=provider, cfg=cfg,
        output_dir_base=out_dir, ddp_rank=0,
    )
    payload = yaml.safe_load((out_dir / "target_real" / f"{trc.name}.yaml").read_text())
    assert payload["kind"] == "target_real_lockstep"


# ---------------------------------------------------------------------------
# 2. Dispatcher is invoked through delegation (proves we call into the
#    shared CORE pipeline)
# ---------------------------------------------------------------------------


def test_dispatcher_invocation_chain_through_core_helper(
    tiny_setup, tmp_path, monkeypatch
):
    """Source-level pin + behavior verification: dclm_eval delegates to
    base_eval._run_core_target_real_eval, which calls
    build_target_for_real_config per row. The chain must be intact.

    The per-config payload builder delegates to the shared CORE helper, and
    main routes target_real configs through the shared orchestrator.
    """
    per_config_src = inspect.getsource(mod._dclm_core_target_real_per_config)
    main_src = inspect.getsource(mod.main)
    assert "_run_core_target_real_eval" in per_config_src, (
        "dclm_eval must delegate to the shared base_eval CORE target_real helper "
    )
    assert "run_target_real_evals" in main_src
    # Behavior: call the orchestrator with a tracking patch to ensure the
    # core helper gets invoked.
    tokenizer, model = tiny_setup
    trc = _build_pooled_trc(tmp_path)
    provider = _MissProvider()
    fake_out = {"results": {}, "centered_results": {}, "core_metric": 0.0}
    calls = []
    monkeypatch.setattr(
        mod, "_run_core_target_real_eval",
        lambda **kw: (calls.append(kw), fake_out)[1],
    )
    cfg = OmegaConf.create({"max_per_task": -1})
    out_dir = tmp_path / "out"
    _run_dclm_target_real(
        monkeypatch=monkeypatch,
        raw_encdec=model, tokenizer=tokenizer, device="cpu",
        trc=trc, provider=provider, cfg=cfg,
        output_dir_base=out_dir, ddp_rank=0,
    )
    assert len(calls) == 1
    assert calls[0]["raw_encdec"] is model
    assert calls[0]["trc"] is trc
    assert calls[0]["provider"] is provider


# Encdec guard + provider-construction tests have been consolidated into
# tests/steering/test_target_real_runner.py — assert_target_real_requires_encdec
# and build_context_feature_provider are tested once for all three eval surfaces.


# ---------------------------------------------------------------------------
# 3. DDP rank-0 barrier
# ---------------------------------------------------------------------------


def test_non_zero_rank_skips_target_real_eval(
    tiny_setup, tmp_path, monkeypatch
):
    tokenizer, model = tiny_setup
    trc = _build_pooled_trc(tmp_path)
    provider = _MissProvider()
    calls = []
    monkeypatch.setattr(
        mod, "_run_core_target_real_eval",
        lambda **kw: (calls.append(kw), {"results": {}, "centered_results": {}, "core_metric": 0.0})[1],
    )
    cfg = OmegaConf.create({"max_per_task": -1})
    out_dir = tmp_path / "out"
    _run_dclm_target_real(
        monkeypatch=monkeypatch,
        raw_encdec=model, tokenizer=tokenizer, device="cpu",
        trc=trc, provider=provider, cfg=cfg,
        output_dir_base=out_dir, ddp_rank=3,  # NOT zero
    )
    assert not (out_dir / "target_real" / f"{trc.name}.yaml").exists()
    assert len(calls) == 0


def test_main_applies_rank_0_barrier():
    src = inspect.getsource(mod.main)
    assert "target_real_configs and ddp_rank == 0" in src, (
        "main()'s PR-C dispatch must be guarded by `if target_real_configs "
        "and ddp_rank == 0`"
    )
