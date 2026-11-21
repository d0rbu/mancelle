"""PR-D Batch H2: production collect-loop actually runs target_real evals.

Bypass-detection behavior tests. The tests pin that the production
collect-loop:
  CI-1: actually invokes a real per-suite target_real eval helper for
        every (eval_suite, trc, seed) triple — not a no-op stub.
  CI-5: target_real entries are keyed under ``evals[eval_name]``
        (per-suite), NOT at the top of ``evals`` — multi-suite runs
        must not conflate.
  CI-6: the seed loop iterates ``_eval_seeds_for_suite(eval_name, seeds)``
        not the full sweep ``seeds`` list (resume parity with
        conditioned_configs path).

Each test drives the production code path through
``_evaluate_single_checkpoint_impl`` (stubbing model load + per-eval
helper as needed) and asserts behavior on the resulting ``evals`` dict.
A regression to a stub would fail loudly — the asserts inspect the
side-effects of the real production loop, not source text.
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
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
)
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer

# ---------------------------------------------------------------------------
# Test scaffolding: make _evaluate_single_checkpoint_impl drivable in CI by
# stubbing the model-load step + the per-eval helpers, but keep the
# *production* control flow (the iteration over target_real_configs +
# the schema + the error-handling) live.
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    eval_suites: str = "core",
    target_real_configs: list[dict] | None = None,
    conditioned_configs: dict | None = None,
) -> OmegaConf:
    """Build a minimal cfg for collect-loop tests.

    Note: ``conditioned_configs`` defaults to {} so the test exercises the
    target_real-only path (no static-yaml configs to iterate). Tests that
    want to exercise the conditioned + target_real interleave should pass
    a non-empty dict.
    """
    return OmegaConf.create({
        "model_tag": "test",
        "checkpoint_source": "local",
        "checkpoints_dir": None,
        "checkpoint_root": None,
        "checkpoint_prefix": "",
        "model_family": "encoder_decoder",
        "results_file": None,
        "fig_dir": None,
        "evals": eval_suites,
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
    })  # ty: ignore[invalid-return-type]


@pytest.fixture()
def stub_load_progression_model():
    """Stub the heavy model-load step. The model is a real (tiny)
    EncoderDecoderGPT so the wrapper-build and isinstance checks behave
    as in production."""
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
    """Stub _target_real_providers_for_cfg to return a dict of mock
    providers (one per trc name) keyed by config name."""
    providers: dict[str, MagicMock] = {}

    def _stub(cfg, *, model_max_seq_len, suite=None, target_real_configs=None):
        # H7 fix: production now passes parsed configs directly.
        if target_real_configs is not None:
            for trc in target_real_configs:
                providers[str(trc.name)] = MagicMock()
            return providers
        cond = cfg.get("conditioning", {})
        target_real_raw = list(cond.get("target_real_configs", []) or [])
        for trc in target_real_raw:
            providers[str(trc["name"])] = MagicMock()
        return providers

    with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_stub):
        yield providers


# ---------------------------------------------------------------------------
# CI-1: production loop invokes a real per-suite eval helper
# ---------------------------------------------------------------------------


def test_collect_loop_calls_run_eval_suite_target_real_per_seed(
    stub_load_progression_model, stub_target_real_provider,
):
    """CI-1: the production target_real loop must invoke
    :func:`_run_eval_suite_target_real` once per (eval_suite, trc, seed)
    triple. A regression to a no-op stub would yield zero call_count.

    This is the load-bearing bypass-detection test: it asserts on the
    side effect of the production code, not on source text."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
            {"name": "lockstep1", "mode": "lockstep",
             "cache_dir": "/tmp/c2"},
        ],
    )
    cfg.num_seeds = 2  # ty: ignore[unresolved-attribute]
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]  # ty: ignore[unresolved-attribute]

    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    call_log: list[tuple[str, str, int]] = []

    def _fake_run_target_real(*, eval_name, model, tokenizer, device, seed,
                              cfg, trc, provider, target_mode):
        call_log.append((eval_name, trc.name, int(seed)))
        return {"core_metric": 0.5, "results": {}, "centered_results": {}}

    with patch.object(
        mod, "_run_eval_suite_target_real", side_effect=_fake_run_target_real,
    ):
        _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    # 1 suite x 2 trcs x 2 seeds = 4 calls.
    assert len(call_log) == 4, (
        f"Expected 4 _run_eval_suite_target_real calls (1 suite x 2 trcs "
        f"x 2 seeds), got {len(call_log)}: {call_log}. The collect-loop "
        f"may have regressed to a no-op stub."
    )
    triples_seen = set(call_log)
    expected = {
        ("core", "pooled1", seeds[0]),
        ("core", "pooled1", seeds[1]),
        ("core", "lockstep1", seeds[0]),
        ("core", "lockstep1", seeds[1]),
    }
    assert triples_seen == expected


def test_collect_loop_populates_seeds_dict_with_real_results(
    stub_load_progression_model, stub_target_real_provider,
):
    """CI-1 corollary: the seeds dict must contain the eval result returned
    by ``_run_eval_suite_target_real``, not stay empty."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    expected_result = {"core_metric": 0.42, "results": {"task_a": 0.5}}

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return expected_result

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    cfg_result = evals["core"]["target_real"]["configs"]["pooled1"]
    assert cfg_result["seeds"] != {}, (
        "seeds dict is empty — the collect-loop did not write the eval "
        "result. Did the no-op stub regress?"
    )
    assert cfg_result["seeds"]["1234"] == expected_result


# ---------------------------------------------------------------------------
# CI-5: target_real keyed per eval_name, not top-level
# ---------------------------------------------------------------------------


def test_collect_loop_keys_target_real_per_eval_name_not_top_level(
    stub_load_progression_model, stub_target_real_provider,
):
    """CI-5: target_real must be keyed under ``evals[eval_name]``,
    NOT at the top level. A multi-suite run that conflates the two would
    silently drop one suite's results."""
    cfg = _make_cfg(
        eval_suites="core,chat",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    # Each suite + trc + seed gets a unique result so we can verify they
    # don't get conflated.
    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"_suite_marker": eval_name, "_seed": int(seed)}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core", "chat"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    # Top-level ``evals['target_real']`` must NOT exist (the regression that
    # this test pins).
    assert "target_real" not in evals, (
        "evals['target_real'] exists at top level — CI-5 regression. "
        "target_real must be keyed under evals[eval_name]."
    )

    # Both suites must each have their own target_real entry.
    assert "core" in evals and "target_real" in evals["core"]
    assert "chat" in evals and "target_real" in evals["chat"]

    core_entry = evals["core"]["target_real"]["configs"]["pooled1"]["seeds"]
    chat_entry = evals["chat"]["target_real"]["configs"]["pooled1"]["seeds"]
    assert core_entry["1234"]["_suite_marker"] == "core"
    assert chat_entry["1234"]["_suite_marker"] == "chat"


# ---------------------------------------------------------------------------
# CI-6: seed loop iterates _eval_seeds_for_suite, not full sweep seeds
# ---------------------------------------------------------------------------


def test_collect_loop_uses_eval_seeds_for_suite_not_full_sweep_seeds(
    stub_load_progression_model, stub_target_real_provider,
):
    """CI-6: the chat suite gets ``seeds[:1]`` (single seed for chat); the
    target_real loop must respect that just like the conditioned_configs
    loop does. A regression to ``for seed in seeds:`` would emit 3 chat
    seeds where there should be only 1."""
    cfg = _make_cfg(
        eval_suites="chat",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
        ],
    )
    cfg.num_seeds = 3  # ty: ignore[unresolved-attribute]
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]  # ty: ignore[unresolved-attribute]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    call_log: list[int] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        call_log.append(int(seed))
        return {"chat_metric": 0.0}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["chat"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    # chat suite = single seed — _eval_seeds_for_suite returns seeds[:1].
    assert len(call_log) == 1, (
        f"Expected 1 chat seed call (chat is single-seed), got {len(call_log)} "
        f"({call_log}). The target_real loop is iterating the full sweep seed "
        f"list — CI-6 regression."
    )
    assert call_log == [seeds[0]]


def test_collect_loop_pooled_does_not_raise(
    stub_load_progression_model, stub_target_real_provider,
):
    """Pooled target-real configs run through the collect loop."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"core_metric": 0.5, "results": {}, "centered_results": {}}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    cfg_result = evals["core"]["target_real"]["configs"]["pooled1"]
    assert "error" not in cfg_result
    assert cfg_result["seeds"] == {"1234": {
        "core_metric": 0.5, "results": {}, "centered_results": {},
    }}


# ---------------------------------------------------------------------------
# Resume guard: pre-populated seeds are skipped.
# ---------------------------------------------------------------------------


def test_collect_loop_skips_already_done_seeds(
    stub_load_progression_model, stub_target_real_provider,
):
    """The per-seed resume guard mirrors the conditioned_configs path: a
    pre-populated seeds dict for a target_real config means the loop
    skips that seed."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/c1", "pooling_op": "bucket_max"},
        ],
    )
    cfg.num_seeds = 2  # ty: ignore[unresolved-attribute]
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]  # ty: ignore[unresolved-attribute]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    # Simulate seed 1234 already done.
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

    call_log: list[int] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        call_log.append(int(seed))
        return {"core_metric": 0.5}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=existing_entry,  # ty: ignore[invalid-argument-type]
            work_keys=None,
        )

    # Only the missing seed (2234) should run.
    assert call_log == [2234], (
        f"Expected only seed 2234 to run (1234 already done in existing "
        f"entry), got {call_log}."
    )
    cfg_result = evals["core"]["target_real"]["configs"]["pooled1"]
    # Both seeds present in final dict.
    assert set(cfg_result["seeds"].keys()) == {"1234", "2234"}
    # Existing seed preserved.
    assert cfg_result["seeds"]["1234"] == {"core_metric": 0.99}


# ---------------------------------------------------------------------------
# Schema: kind / cache_dir are recorded.
# ---------------------------------------------------------------------------


def test_collect_loop_records_kind_cache_dir(
    stub_load_progression_model, stub_target_real_provider,
):
    """The cfg_result schema records kind/cache_dir for variant dispatch."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "pooled1", "mode": "pooled",
             "cache_dir": "/tmp/poolc", "pooling_op": "bucket_max"},
            {"name": "lockstep1", "mode": "lockstep",
             "cache_dir": "/tmp/lockc"},
        ],
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"core_metric": 0.5}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,  # ty: ignore[invalid-argument-type]
        )

    pooled = evals["core"]["target_real"]["configs"]["pooled1"]
    assert pooled["kind"] == SuiteResultKind.TARGET_REAL_POOLED.value
    assert pooled["cache_dir"] == "/tmp/poolc"

    lockstep = evals["core"]["target_real"]["configs"]["lockstep1"]
    assert lockstep["kind"] == SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    assert lockstep["cache_dir"] == "/tmp/lockc"


# ---------------------------------------------------------------------------
# Dispatch sanity (kept from prior file): isinstance variants
# ---------------------------------------------------------------------------


def test_lockstep_config_maps_to_suite_result_kind_lockstep():
    """LockstepTargetRealConfig -> SuiteResultKind.TARGET_REAL_LOCKSTEP.value."""
    trc = LockstepTargetRealConfig(name="lockstep", cache_dir="/tmp/lock")  # ty: ignore[invalid-argument-type]
    if isinstance(trc, LockstepTargetRealConfig):
        kind = SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    elif isinstance(trc, PooledTargetRealConfig):
        kind = SuiteResultKind.TARGET_REAL_POOLED.value
    else:
        raise TypeError(
            f"unexpected TargetRealConfig variant: {type(trc).__name__}"
        )
    assert kind == "target_real_lockstep"


def test_pooled_config_maps_to_suite_result_kind_pooled():
    """PooledTargetRealConfig -> SuiteResultKind.TARGET_REAL_POOLED.value."""
    trc = PooledTargetRealConfig(
        name="pooled", cache_dir="/tmp/pool", pooling_op="bucket_max",  # ty: ignore[invalid-argument-type]
    )
    if isinstance(trc, LockstepTargetRealConfig):
        kind = SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    elif isinstance(trc, PooledTargetRealConfig):
        kind = SuiteResultKind.TARGET_REAL_POOLED.value
    else:
        raise TypeError(
            f"unexpected TargetRealConfig variant: {type(trc).__name__}"
        )
    assert kind == "target_real_pooled"


def test_unknown_trc_variant_raises_type_error():
    """Unknown variant raises TypeError (exhaustive check)."""
    class UnknownTRC:
        name = "unknown"
        cache_dir = "/tmp"

    trc = UnknownTRC()
    with pytest.raises(TypeError, match="TargetRealConfig variant"):
        if isinstance(trc, LockstepTargetRealConfig):
            raise AssertionError("should not reach")
        if isinstance(trc, PooledTargetRealConfig):
            raise AssertionError("should not reach")
        raise TypeError(
            f"unexpected TargetRealConfig variant: {type(trc).__name__}"
        )
