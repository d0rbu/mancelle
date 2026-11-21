"""Unit tests for scripts.train.conditional_encdec_train._maybe_run_inline_steering_eval.

These tests mock evaluate_steering_inline and prepare_inline_eval_cache at
the module level and call the helper directly, bypassing the real training
loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_train as train_mod
from scripts.train.conditional_encdec_train import _maybe_run_inline_steering_eval
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.inline_steering_eval import InlineEvalCache, SteeringEvalConfig

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _MockModel:
    """Minimal model with eval/train toggling and parameter tracking."""

    def __init__(self):
        self._is_training = True
        self.eval_calls = 0
        self.train_calls = 0

    @property
    def training(self) -> bool:
        return self._is_training

    def eval(self):
        self._is_training = False
        self.eval_calls += 1
        return self

    def train(self, mode: bool = True):
        self._is_training = mode
        self.train_calls += 1
        return self


class _MockFeatureIndex:
    """Marker class used to verify pass-through in kwargs."""


class _MockProbeModel:
    def __init__(self):
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self


class _MockWandbRun:
    def __init__(self):
        self.log_calls: list[tuple[dict, int]] = []

    def log(self, metrics: dict, step: int):
        self.log_calls.append((metrics, step))


@dataclass
class _FakeSharedState:
    feature_index: _MockFeatureIndex
    steering_probe_model: _MockProbeModel | None
    steering_eval_cfg: SteeringEvalConfig | None


@pytest.fixture
def fake_shared():
    return _FakeSharedState(
        feature_index=_MockFeatureIndex(),
        steering_probe_model=_MockProbeModel(),
        steering_eval_cfg=SteeringEvalConfig(n_eval_docs=3, samples_per_condition=1),
    )


@pytest.fixture
def fake_cfg():
    return OmegaConf.create({
        "steering_eval_every": 250,
        "offload_steering_eval_feature_model": False,
        "data_dir": None,
    })


@pytest.fixture
def fake_threshold_cfg():
    return ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=3.0)


@pytest.fixture
def stub_helpers(monkeypatch):
    """Install counting stubs for evaluate_steering_inline and prepare_inline_eval_cache.

    Returns a dict tracking call counts and captured kwargs.
    """
    state: dict[str, Any] = {
        "eval_calls": 0,
        "eval_last_kwargs": None,
        "prepare_calls": 0,
        "prepare_last_kwargs": None,
    }

    def _stub_eval(**kwargs):
        state["eval_calls"] += 1
        state["eval_last_kwargs"] = kwargs
        return {"steering/eval_time_seconds": 0.123, "steering/eval_cache_hit": 1}

    def _stub_prepare(**kwargs):
        state["prepare_calls"] += 1
        state["prepare_last_kwargs"] = kwargs
        return InlineEvalCache(
            selected_docs=[(0, 1)],
            selection_metrics={"steering/n_docs_available": 3, "steering/n_docs_selected": 1},
            doc_texts={0: "hello"},
        )

    monkeypatch.setattr(train_mod, "evaluate_steering_inline", _stub_eval)
    monkeypatch.setattr(train_mod, "prepare_inline_eval_cache", _stub_prepare)

    # disable_fp8 is a context manager; replace with a no-op
    class _NoopCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(train_mod, "disable_fp8", lambda m: _NoopCtx())

    # save_rng_state / restore_rng_state — no-ops
    monkeypatch.setattr(train_mod, "save_rng_state", lambda device: {})
    monkeypatch.setattr(train_mod, "restore_rng_state", lambda state, device: None)

    return state


def _run_helper(
    *,
    step,
    last_step,
    cfg,
    shared,
    threshold_cfg,
    inline_eval_cache=None,
    master_process=True,
    wandb_run=None,
):
    """Construct the full kwarg list for the helper under test and call it.

    Returns a flat (updated_cache, model) tuple so tests can destructure
    both cleanly: ``updated_cache, model = _run_helper(...)``.
    """
    model = _MockModel()
    updated_cache = _maybe_run_inline_steering_eval(
        step=step,
        last_step=last_step,
        cfg=cfg,
        shared=shared,
        model=model,
        orig_model=model,  # same object in tests — real train_one_config passes orig_model
        feature_dir="/tmp/ignored",
        steering_threshold_cfg=threshold_cfg,
        steering_precomputed_thresholds=None,
        steering_percentile_table=None,
        tokenizer=None,
        device=torch.device("cpu"),
        device_type="cpu",
        boundaries=torch.zeros(0),
        ddp_rank=0,
        ddp_world_size=1,
        master_process=master_process,
        wandb_run=wandb_run or _MockWandbRun(),
        inline_eval_cache=inline_eval_cache,
    )
    return updated_cache, model


# ---------------------------------------------------------------------------
# Trigger-logic tests
# ---------------------------------------------------------------------------


def test_helper_noop_when_trigger_does_not_fire(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """At step 1 with steering_eval_every=250, the helper does nothing."""
    updated_cache, _ = _run_helper(
        step=1, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        inline_eval_cache=None,
    )
    assert updated_cache is None
    assert stub_helpers["eval_calls"] == 0
    assert stub_helpers["prepare_calls"] == 0


def test_helper_fires_on_matching_step(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """At step 250 (divisible by steering_eval_every=250), helper fires."""
    updated_cache, _ = _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        inline_eval_cache=None,
    )
    assert updated_cache is not None
    assert stub_helpers["eval_calls"] == 1
    assert stub_helpers["prepare_calls"] == 1


def test_helper_fires_on_last_step(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """last_step=True fires regardless of step number."""
    updated_cache, _ = _run_helper(
        step=123, last_step=True,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        inline_eval_cache=None,
    )
    assert updated_cache is not None
    assert stub_helpers["eval_calls"] == 1


def test_helper_disabled_when_every_zero(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """steering_eval_every=0 disables the helper entirely."""
    fake_cfg.steering_eval_every = 0
    updated_cache, _ = _run_helper(
        step=250, last_step=True,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    assert updated_cache is None
    assert stub_helpers["eval_calls"] == 0


def test_helper_noop_when_steering_eval_cfg_none(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """When shared.steering_eval_cfg is None, helper does not fire."""
    fake_shared.steering_eval_cfg = None
    updated_cache, _ = _run_helper(
        step=250, last_step=True,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    assert updated_cache is None
    assert stub_helpers["eval_calls"] == 0


# ---------------------------------------------------------------------------
# Cache lifecycle tests
# ---------------------------------------------------------------------------


def test_helper_builds_cache_on_first_fire(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """Starting with inline_eval_cache=None, the first fire calls prepare_inline_eval_cache."""
    updated_cache, _ = _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        inline_eval_cache=None,
    )
    assert stub_helpers["prepare_calls"] == 1
    assert isinstance(updated_cache, InlineEvalCache)


def test_helper_reuses_cache_on_subsequent_fires(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """Firing 5 times in a row triggers prepare_inline_eval_cache only once."""
    cache = None
    for step in [250, 500, 750, 1000, 1250]:
        cache, _ = _run_helper(
            step=step, last_step=False,
            cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
            inline_eval_cache=cache,
        )

    assert stub_helpers["prepare_calls"] == 1, (
        f"prepare_inline_eval_cache called {stub_helpers['prepare_calls']} times, expected 1"
    )
    assert stub_helpers["eval_calls"] == 5


def test_helper_fresh_cache_per_config(fake_cfg, fake_threshold_cfg, stub_helpers):
    """Two sequential runs starting from inline_eval_cache=None each build their own cache."""
    shared_a = _FakeSharedState(
        feature_index=_MockFeatureIndex(),
        steering_probe_model=_MockProbeModel(),
        steering_eval_cfg=SteeringEvalConfig(),
    )
    shared_b = _FakeSharedState(
        feature_index=_MockFeatureIndex(),
        steering_probe_model=_MockProbeModel(),
        steering_eval_cfg=SteeringEvalConfig(),
    )
    _run_helper(step=250, last_step=False, cfg=fake_cfg, shared=shared_a, threshold_cfg=fake_threshold_cfg, inline_eval_cache=None)
    _run_helper(step=250, last_step=False, cfg=fake_cfg, shared=shared_b, threshold_cfg=fake_threshold_cfg, inline_eval_cache=None)

    assert stub_helpers["prepare_calls"] == 2
    assert stub_helpers["eval_calls"] == 2


# ---------------------------------------------------------------------------
# Kwarg wiring tests
# ---------------------------------------------------------------------------


def test_helper_passes_feature_index_kwarg(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """evaluate_steering_inline must be called with feature_index=shared.feature_index."""
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    kwargs = stub_helpers["eval_last_kwargs"]
    assert kwargs is not None
    assert kwargs["feature_index"] is fake_shared.feature_index


def test_helper_passes_eval_cache_kwarg(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """evaluate_steering_inline must be called with eval_cache equal to the computed cache."""
    updated_cache, _ = _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        inline_eval_cache=None,
    )
    kwargs = stub_helpers["eval_last_kwargs"]
    assert kwargs is not None
    assert kwargs["eval_cache"] is updated_cache


def test_helper_forwards_all_expected_kwargs(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """evaluate_steering_inline receives every kwarg the original in-loop block passed.

    Guard against accidental drops (e.g., forgetting to thread tokenizer
    or boundaries in a future refactor).
    The set assertion uses >= so that adding NEW kwargs in a future commit
    won't break this test.
    """
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    kwargs = stub_helpers["eval_last_kwargs"]
    assert kwargs is not None

    expected_kwargs = {
        "model",
        "feature_dir",
        "tokenizer",
        "device",
        "boundaries",
        "threshold_cfg",
        "precomputed_thresholds",
        "cfg",
        "ddp_rank",
        "ddp_world_size",
        "probe_model",
        "percentile_table",
        "feature_index",  # P0.1
        "eval_cache",     # P0.2 + P0.3
    }
    actual = set(kwargs.keys())
    missing = expected_kwargs - actual
    assert not missing, (
        f"evaluate_steering_inline missing expected kwargs: {missing}\n"
        f"actual kwargs: {sorted(actual)}"
    )
    assert kwargs["device"] == torch.device("cpu")


def test_unclustered_run_forwards_meta_none(
    fake_cfg, fake_shared, fake_threshold_cfg, stub_helpers
):
    """The inline eval helper forwards meta=None."""
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    kwargs = stub_helpers["eval_last_kwargs"]
    assert kwargs is not None
    assert kwargs["meta"] is None


# ---------------------------------------------------------------------------
# Side-effect preservation tests
# ---------------------------------------------------------------------------


def test_helper_preserves_model_train_eval_state(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """On fire, helper calls model.eval() at least once and leaves it in train mode."""
    _cache, model = _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    assert model.eval_calls >= 1, "helper should have called model.eval() during eval"
    assert model.train_calls >= 1, "helper should have called model.train() after eval"
    assert model.training, "helper should leave the model in training mode"


def test_helper_logs_wandb_metrics_on_master(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """On fire with master_process=True, wandb_run.log is called once with the metrics."""
    wandb_run = _MockWandbRun()
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        master_process=True,
        wandb_run=wandb_run,
    )
    assert len(wandb_run.log_calls) == 1
    metrics, step = wandb_run.log_calls[0]
    assert step == 250
    assert "steering/eval_time_seconds" in metrics


def test_helper_does_not_log_wandb_on_non_master(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """On fire with master_process=False, wandb_run.log is NOT called."""
    wandb_run = _MockWandbRun()
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
        master_process=False,
        wandb_run=wandb_run,
    )
    assert wandb_run.log_calls == []


def test_helper_offload_toggle_moves_probe_model(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """When offload_steering_eval_feature_model=True, helper moves probe model
    to the eval device before the eval and back to CPU after."""
    fake_cfg.offload_steering_eval_feature_model = True
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    # Must have moved the probe model to the eval device first, then back to "cpu"
    assert fake_shared.steering_probe_model is not None
    assert fake_shared.steering_probe_model.to_calls == ["cpu", "cpu"], (
        f"Expected [eval_device, cpu] = ['cpu', 'cpu'] for fake cpu device, "
        f"got {fake_shared.steering_probe_model.to_calls}"
    )


def test_helper_offload_toggle_disabled_by_default(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """When offload_steering_eval_feature_model=False (default), helper does
    NOT move the probe model at all."""
    # Ensure default is False
    assert fake_cfg.offload_steering_eval_feature_model is False
    _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    assert fake_shared.steering_probe_model is not None
    assert fake_shared.steering_probe_model.to_calls == [], (
        f"Probe model should not be moved when offload disabled, "
        f"got to_calls={fake_shared.steering_probe_model.to_calls}"
    )


def test_helper_offload_toggle_handles_none_probe_model(fake_shared, fake_cfg, fake_threshold_cfg, stub_helpers):
    """When shared.steering_probe_model is None, offload toggle must not crash
    regardless of the flag value. Guards against a NoneType.to() refactor bug."""
    fake_shared.steering_probe_model = None
    fake_cfg.offload_steering_eval_feature_model = True
    # Should not raise
    updated_cache, _ = _run_helper(
        step=250, last_step=False,
        cfg=fake_cfg, shared=fake_shared, threshold_cfg=fake_threshold_cfg,
    )
    assert updated_cache is not None


def test_helper_restores_rng_model_mode_and_probe_after_failure(
    fake_shared,
    fake_cfg,
    fake_threshold_cfg,
    stub_helpers,
    monkeypatch,
):
    fake_cfg.offload_steering_eval_feature_model = True
    restored: list[tuple[dict, torch.device]] = []
    monkeypatch.setattr(
        train_mod,
        "evaluate_steering_inline",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("eval failed")),
    )
    monkeypatch.setattr(train_mod, "save_rng_state", lambda device: {"saved": True})
    monkeypatch.setattr(
        train_mod,
        "restore_rng_state",
        lambda state, device: restored.append((state, device)),
    )
    cache = InlineEvalCache(
        selected_docs=[(0, 1)],
        selection_metrics={},
        doc_texts={0: "hello"},
    )
    model = _MockModel()

    with pytest.raises(ValueError, match="eval failed"):
        _maybe_run_inline_steering_eval(
            step=250,
            last_step=False,
            cfg=fake_cfg,
            shared=fake_shared,
            model=model,
            orig_model=model,
            feature_dir="/tmp/ignored",
            steering_threshold_cfg=fake_threshold_cfg,
            steering_precomputed_thresholds=None,
            steering_percentile_table=None,
            tokenizer=None,
            device=torch.device("cpu"),
            device_type="cpu",
            boundaries=torch.zeros(0),
            ddp_rank=0,
            ddp_world_size=1,
            master_process=True,
            wandb_run=_MockWandbRun(),
            inline_eval_cache=cache,
        )

    assert model.training
    assert restored == [({"saved": True}, torch.device("cpu"))]
    assert fake_shared.steering_probe_model is not None
    assert fake_shared.steering_probe_model.to_calls == ["cpu", "cpu"]
