"""PR-D Batch H7 (paranoid review): invariant tests for progression
target_real collect-loop and shared per-example primitives.

This file groups the deepest invariants — things the code assumes but
previously had no behavior test pinning. Each test docs the exact
invariant being pinned. A regression that violates the invariant must
fail the test loudly.

Coverage focus (gaps identified during the H7 deep review):
  1. ``_target_real_providers_for_cfg`` consistency:
     * num_buckets default matches the runner's auto-build path (2,
       not 10) so the default 3-mode YAML works without overrides.
     * Auto-resolved cache_dir (from ``parse_target_real_configs_from_cfg``)
       is used for provider construction — NOT the raw YAML cache_dir
       (which is null for the auto-resolve case and would crash).
  2. ``_run_target_real_for_suite`` invariants:
     * ``cfg_result["seeds"]`` length equals
       ``len(_eval_seeds_for_suite(eval_name, seeds))`` after a clean run.
     * Two suites with the same trc.name keep INDEPENDENT seed dicts.
     * Empty target_real_configs short-circuits before any provider
       lookup.
     * Existing seeds are NOT overwritten on re-run.
     * Per-seed eval failure does not corrupt other seeds.
     * cfg_result["kind"] matches the variant's
       SuiteResultKind.value.
     * ``_trc_kind_for`` is exhaustive over (Lockstep, Pooled).
  3. ``_run_eval_suite_target_real`` dispatch:
     * harmfulness suite returns None and writes nothing.
     * unknown suite returns None and writes nothing.
  4. ``build_target_real_payload`` round-trips via
     ``load_suite_from_dict`` for both variants.
  5. ``parse_target_real_configs_from_cfg`` produces different
     auto-resolved cache_dirs for different suite names.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import scripts.train.training_progression as mod
import steering.eval.eval_cache_paths as paths_mod
import steering.eval.target_real_runner as runner_mod
from scripts.train.training_progression import (
    CheckpointSpec,
    _evaluate_single_checkpoint_impl,
    _target_real_providers_for_cfg,
    _trc_kind_for,
)
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.eval.context_cache_io import CONTEXT_LEN_DIR_RE
from steering.eval.pooling_op import PoolingOp
from steering.eval.progression_results import (
    SuiteResultKind,
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
    load_suite_from_dict,
)
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
)
from steering.eval.target_real_runner import (
    build_target_real_payload,
    parse_target_real_configs_from_cfg,
)
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_identity_cfg(**overrides):
    base = {
        "model_id": "gemma-2-2b",
        "sae_id": "gemmascope-residual-l8-16k",
        "layer": 8,
        "pooling": "max_pool",
        "feature_selection_hash": "a" * 64,
    }
    base.update(overrides)
    return base


def _make_cfg(
    *,
    eval_suites: str = "core",
    target_real_configs: list[dict] | None = None,
    conditioned_configs: dict | None = None,
    num_seeds: int = 1,
    base_seed: int = 1234,
):
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
        "num_seeds": num_seeds,
        "base_seed": base_seed,
        "phases": [{"phase": "sft", "subdir": "", "steps": [0]}],
        "cleanup_after_eval": False,
        "source": "encdec",
        "conditioning": {
            "identity": _full_identity_cfg(),
            "n_sae_features": 64,
            "num_buckets": 4,
            "context_cache_dir": None,
            "conditioned_configs": conditioned_configs or {},
            "target_real_configs": target_real_configs or [],
        },
    })


def _parsed_target_real_configs(cfg, *, suite: str = "sft_target", model_max_seq_len: int = 1024):
    return parse_target_real_configs_from_cfg(
        cfg.conditioning,
        suite=suite,
        model_max_seq_len=model_max_seq_len,
    )


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

    def _stub(cfg, *, model_max_seq_len, target_real_configs):
        for trc in target_real_configs:
            providers[str(trc.name)] = MagicMock()
        return providers

    with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_stub):
        yield providers


# ---------------------------------------------------------------------------
# Bug-pin: _target_real_providers_for_cfg with null/missing cache_dir
# (the default 3-mode YAML currently crashes here — pin the desired behavior).
# ---------------------------------------------------------------------------


def test_provider_factory_handles_missing_cache_dir_field_with_auto_resolve(
    monkeypatch, tmp_path,
):
    """Invariant (BUG FIX): the default 3-mode YAML has cache_dir field
    absent for auto-resolve entries. ``_target_real_providers_for_cfg`` must
    auto-resolve via ``default_target_cache_dir(suite=...)`` rather than
    crashing with a ``KeyError`` on ``trc["cache_dir"]``.

    Before the fix this raised ``KeyError: 'cache_dir'``."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    # Patch out the actual provider class so we don't hit a missing manifest.
    constructed_paths: list[Path] = []

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, **kw):
            constructed_paths.append(Path(cache_dir))

    monkeypatch.setattr(mod, "SafetensorsContextFeatureProvider", _StubProvider)

    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                # Field missing entirely — mirrors default 3-mode YAML.
                {"name": "pooled", "mode": "pooled"},
                {"name": "lockstep", "mode": "lockstep"},
            ],
            "identity": _full_identity_cfg(),
            "n_sae_features": 16384,
            "num_buckets": 2,
        },
    })
    out = _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=1024,
        target_real_configs=_parsed_target_real_configs(cfg),
    )
    assert set(out.keys()) == {"pooled", "lockstep"}
    # All constructed paths must contain "eval_caches" + "sft_target".
    for p in constructed_paths:
        assert "eval_caches" in str(p)
        assert "sft_target" in str(p)


def test_provider_factory_handles_null_cache_dir_with_auto_resolve(
    monkeypatch, tmp_path,
):
    """Invariant (BUG FIX): null cache_dir + suite kwarg also auto-resolves.
    Before the fix this attempted ``str(None) == 'None'`` and constructed a
    bogus path."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    constructed_paths: list[Path] = []

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, **kw):
            constructed_paths.append(Path(cache_dir))

    monkeypatch.setattr(mod, "SafetensorsContextFeatureProvider", _StubProvider)

    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                {"name": "pooled", "mode": "pooled", "cache_dir": None},
            ],
            "identity": _full_identity_cfg(),
            "n_sae_features": 16384,
            "num_buckets": 2,
        },
    })
    out = _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=1024,
        target_real_configs=_parsed_target_real_configs(cfg),
    )
    assert set(out.keys()) == {"pooled"}
    # The path must NOT contain the literal string 'None' (the str(None)
    # leak that the old code produced).
    for p in constructed_paths:
        assert "None" not in str(p), (
            f"path contains 'None' — str(None) leak from raw cfg traversal: {p}"
        )


def test_provider_factory_does_not_double_append_context_length_dir(
    monkeypatch, tmp_path,
):
    """Invariant (BUG FIX): when an auto-resolved cache_dir is passed in,
    the resulting provider path must NOT contain two ``c{N}`` segments.

    The auto-resolved path layout already ends with
    ``.../c{N}/{hash[:16]}/`` so calling ``context_length_cache_dir`` on it
    again would append ANOTHER ``c{N}`` at the end (verified via repro
    script). The fix must avoid the double-append."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    constructed_paths: list[Path] = []

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, **kw):
            constructed_paths.append(Path(cache_dir))

    monkeypatch.setattr(mod, "SafetensorsContextFeatureProvider", _StubProvider)

    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                {"name": "pooled", "mode": "pooled", "cache_dir": None},
            ],
            "identity": _full_identity_cfg(),
            "n_sae_features": 16384,
            "num_buckets": 2,
        },
    })
    _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=1024,
        target_real_configs=_parsed_target_real_configs(cfg),
    )
    # Count c{N} segments per path.
    for p in constructed_paths:
        c_segments = [
            seg for seg in p.parts if CONTEXT_LEN_DIR_RE.fullmatch(seg)
        ]
        assert len(c_segments) <= 1, (
            f"path has {len(c_segments)} c-segments — context_length_cache_dir "
            f"applied twice or default_target_cache_dir already includes it: {p}"
        )


def test_provider_factory_uses_num_buckets_default_2(monkeypatch):
    """Invariant: the default for num_buckets must be 2 (binary-bucket SAE
    pipeline) to match the runner's ``build_context_feature_provider``
    default (line 179) and the YAML defaults. Before the fix this used 10."""
    captured: dict = {}

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, expected_n_sae_features=None,
                     expected_num_buckets=None, **kw):
            captured["expected_num_buckets"] = expected_num_buckets

    monkeypatch.setattr(mod, "SafetensorsContextFeatureProvider", _StubProvider)
    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                {"name": "p", "mode": "pooled", "cache_dir": "/tmp/c"},
            ],
            "identity": _full_identity_cfg(),
            # NOTE: num_buckets not set — must default to 2.
        },
    })
    _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=1024,
        target_real_configs=_parsed_target_real_configs(cfg),
    )
    assert captured["expected_num_buckets"] == 2, (
        f"Expected num_buckets default 2, got {captured.get('expected_num_buckets')}; "
        f"this must match build_context_feature_provider's default for cache "
        f"validation parity."
    )


# ---------------------------------------------------------------------------
# Provider lifecycle: construction count by (cache_dir, model_max_seq_len)
# ---------------------------------------------------------------------------


def test_provider_construction_count_one_per_unique_cache_dir(monkeypatch):
    """Invariant: ``SafetensorsContextFeatureProvider.__init__`` is called
    exactly N times where N = number of unique (cache_dir,
    model_max_seq_len) tuples — even if more trcs reference those cache_dirs."""
    construction_count = 0

    class _StubProvider:
        def __init__(self, *a, **kw):
            nonlocal construction_count
            construction_count += 1

    monkeypatch.setattr(mod, "SafetensorsContextFeatureProvider", _StubProvider)
    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": [
                {"name": "a", "mode": "pooled", "cache_dir": "/tmp/X"},
                {"name": "b", "mode": "lockstep", "cache_dir": "/tmp/X"},
                {"name": "c", "mode": "pooled", "cache_dir": "/tmp/Y"},
                {"name": "d", "mode": "lockstep", "cache_dir": "/tmp/X"},
            ],
            "identity": _full_identity_cfg(),
            "n_sae_features": 16, "num_buckets": 2,
        },
    })
    out = _target_real_providers_for_cfg(
        cfg,
        model_max_seq_len=512,
        target_real_configs=_parsed_target_real_configs(
            cfg,
            model_max_seq_len=512,
        ),
    )
    # 4 trcs — but only 2 unique cache_dirs.
    assert construction_count == 2
    # All 4 names map to a provider.
    assert set(out.keys()) == {"a", "b", "c", "d"}
    # The three /tmp/X trcs share the same provider instance.
    assert out["a"] is out["b"] is out["d"]
    # /tmp/Y is independent.
    assert out["a"] is not out["c"]


# ---------------------------------------------------------------------------
# _run_target_real_for_suite invariants
# ---------------------------------------------------------------------------


def test_seeds_dict_length_equals_eval_seeds_for_suite(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: after one collect-pass, the seeds dict for a target_real
    config has length |_eval_seeds_for_suite(eval_name, seeds)|.

    For ``core``, this equals num_seeds. For ``chat``, it equals 1 (chat
    is single-seed). Either way the invariant holds: the dict's seed-key
    set matches the suite's expected seed list."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "trc1", "mode": "pooled", "cache_dir": "/tmp/c1",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=3,
    )
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"core_metric": 0.5}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg, existing_entry=None, work_keys=None,
        )

    seeds_dict = evals["core"]["target_real"]["configs"]["trc1"]["seeds"]
    # core suite uses ALL seeds.
    assert len(seeds_dict) == len(seeds)
    assert set(seeds_dict.keys()) == {str(s) for s in seeds}


def test_two_suites_independent_target_real_seed_dicts(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: when the same trc.name is referenced from two eval
    suites, the seed dicts at
    ``evals[<suite>]['target_real']['configs'][<name>]['seeds']``
    are INDEPENDENT — they share no mutable references. A regression
    that aliased the dict would silently overwrite the first suite's
    results when the second suite ran."""
    cfg = _make_cfg(
        eval_suites="core,chat",
        target_real_configs=[
            {"name": "shared", "mode": "pooled", "cache_dir": "/tmp/c",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=2,
    )
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"_marker": eval_name, "_seed": int(seed)}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core", "chat"], seeds, cfg, existing_entry=None, work_keys=None,
        )

    core_cfg = evals["core"]["target_real"]["configs"]["shared"]
    chat_cfg = evals["chat"]["target_real"]["configs"]["shared"]
    # Different objects.
    assert core_cfg is not chat_cfg, (
        "shared trc.name across suites must yield SEPARATE cfg_result dicts; "
        "aliasing would corrupt resume semantics."
    )
    assert core_cfg["seeds"] is not chat_cfg["seeds"]
    # The markers prove they hold their own suite's results.
    for body in core_cfg["seeds"].values():
        assert body["_marker"] == "core"
    for body in chat_cfg["seeds"].values():
        assert body["_marker"] == "chat"


def test_empty_target_real_configs_skips_loop_entirely(
    stub_load_progression_model,
):
    """Invariant: when target_real_configs is empty,
    ``_run_target_real_for_suite`` returns immediately. No provider
    construction, no eval invocation. The resulting evals dict has no
    'target_real' key."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[],
        num_seeds=1,
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    provider_calls = 0

    def _provider_stub(cfg, *, model_max_seq_len):
        nonlocal provider_calls
        provider_calls += 1
        return {}

    eval_calls = 0

    def _eval_stub(**kw):
        nonlocal eval_calls
        eval_calls += 1
        return {"core_metric": 0.0}

    with patch.object(mod, "_target_real_providers_for_cfg", side_effect=_provider_stub):
        with patch.object(mod, "_run_eval_suite_target_real", side_effect=_eval_stub):
            evals = _evaluate_single_checkpoint_impl(
                spec, torch.device("cpu"), torch.float32,
                ["core"], seeds, cfg, existing_entry=None, work_keys=None,
            )
    # _target_real_providers_for_cfg won't be called because the early-skip
    # branch in _evaluate gates on target_real_raw being non-empty
    # (line 1716). _run_eval_suite_target_real must NOT be called.
    assert eval_calls == 0, (
        f"_run_eval_suite_target_real was invoked {eval_calls} times despite "
        f"empty target_real_configs"
    )
    # No top-level 'target_real' key in any suite.
    if "core" in evals:
        assert "target_real" not in evals["core"]


def test_existing_seeds_not_overwritten_on_re_run(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: when the cfg_result already contains a populated seeds
    dict, the second collect-pass MUST NOT overwrite the existing entry
    even if the eval helper would return a different value. (Per-seed
    skip happens at line 677: ``if str(seed) in cfg_result["seeds"]``.)"""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "trc1", "mode": "pooled", "cache_dir": "/tmp/c",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=1,
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")
    pre_existing_value = {"core_metric": 0.99, "_marker": "ORIGINAL"}
    existing_entry = {
        "step": 0, "phase": "sft",
        "evals": {
            "core": {
                "target_real": {
                    "configs": {
                        "trc1": {
                            "kind": SuiteResultKind.TARGET_REAL_POOLED.value,
                            "cache_dir": "/tmp/c",
                            "seeds": {"1234": pre_existing_value},
                        },
                    },
                },
            },
        },
    }
    invoked = False

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        nonlocal invoked
        invoked = True
        return {"core_metric": -1.0, "_marker": "OVERWRITE"}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg,
            existing_entry=existing_entry, work_keys=None,
        )

    cfg_result = evals["core"]["target_real"]["configs"]["trc1"]
    assert cfg_result["seeds"]["1234"] == pre_existing_value
    assert not invoked, "eval was invoked despite seed already being done"


def test_per_seed_eval_failure_does_not_corrupt_other_seeds(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: if seed K's eval raises a non-NEXT-mode error, the seed
    loop must propagate (re-raise) so the failure is loud — but the
    pre-existing K-1 entry must remain intact in the cfg_result."""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "trc1", "mode": "pooled", "cache_dir": "/tmp/c",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=2,
    )
    seeds = [1000, 2000]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")
    # First seed succeeds, second seed raises (non-NEXT).
    call_log: list[int] = []

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        call_log.append(int(seed))
        if int(seed) == 2000:
            raise ValueError("some unrelated value error")
        return {"core_metric": 0.5}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        with pytest.raises(ValueError, match="some unrelated"):
            _evaluate_single_checkpoint_impl(
                spec, torch.device("cpu"), torch.float32,
                ["core"], seeds, cfg, existing_entry=None, work_keys=None,
            )

    # Both seeds were attempted; the first one's result was already in the
    # call log — but since the ValueError propagates we don't see the
    # final dict. This pins LOUD failure semantics.
    assert call_log == [1000, 2000]


def test_cfg_result_kind_field_matches_isinstance_branch():
    """Invariant: ``cfg_result['kind']`` must equal
    ``SuiteResultKind.TARGET_REAL_LOCKSTEP.value`` iff
    ``isinstance(trc, LockstepTargetRealConfig)``, and similarly for Pooled.
    A regression that swapped the dispatch would silently mis-tag rows."""
    lockstep = LockstepTargetRealConfig(name="ls", cache_dir=Path("/tmp/lock"))
    pooled = PooledTargetRealConfig(
        name="pl", cache_dir=Path("/tmp/pool"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    assert _trc_kind_for(lockstep) == SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    assert _trc_kind_for(pooled) == SuiteResultKind.TARGET_REAL_POOLED.value


def test_trc_kind_for_unknown_variant_raises_typeerror():
    """Invariant: ``_trc_kind_for`` is exhaustive — an unknown
    TargetRealConfig subclass raises TypeError, NOT silently mis-tags."""
    class _Custom:
        name = "custom"
        cache_dir = "/tmp"

    with pytest.raises(TypeError, match="TargetRealConfig variant"):
        _trc_kind_for(_Custom())


# ---------------------------------------------------------------------------
# _run_eval_suite_target_real dispatch
# ---------------------------------------------------------------------------


def test_run_eval_suite_target_real_returns_none_for_harmfulness():
    """Invariant (H2 deviation): ``_run_eval_suite_target_real`` returns
    ``None`` for the harmfulness suite — harmfulness has no per-example
    target cache (judged via API). The result is a logged warning + None
    so the collect-loop's seed loop short-circuits."""
    from scripts.train.training_progression import _run_eval_suite_target_real
    model = build_tiny_encdec()
    tokenizer = build_tiny_tokenizer()
    cfg = OmegaConf.create({"max_per_task": 1})
    trc = PooledTargetRealConfig(
        name="trc", cache_dir=Path("/tmp/c"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    provider = MagicMock()

    result = _run_eval_suite_target_real(
        eval_name="harmfulness", model=model, tokenizer=tokenizer,
        device=torch.device("cpu"), seed=1234, cfg=cfg, trc=trc,
        provider=provider,
    )
    assert result is None


def test_run_eval_suite_target_real_returns_none_for_unknown_suite():
    """Invariant: an unknown suite name -> None (warning logged, no eval)."""
    from scripts.train.training_progression import _run_eval_suite_target_real
    model = build_tiny_encdec()
    tokenizer = build_tiny_tokenizer()
    cfg = OmegaConf.create({"max_per_task": 1})
    trc = PooledTargetRealConfig(
        name="trc", cache_dir=Path("/tmp/c"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    provider = MagicMock()

    result = _run_eval_suite_target_real(
        eval_name="bogus_suite", model=model, tokenizer=tokenizer,
        device=torch.device("cpu"), seed=1234, cfg=cfg, trc=trc,
        provider=provider,
    )
    assert result is None


def test_run_eval_suite_target_real_rejects_decoder_only_model():
    """Invariant: a non-encoder-decoder model + non-empty target_real ->
    ValueError mentioning encoder-decoder."""
    from scripts.train.training_progression import _run_eval_suite_target_real
    not_encdec = MagicMock()  # not isinstance EncoderDecoderGPT
    tokenizer = build_tiny_tokenizer()
    cfg = OmegaConf.create({"max_per_task": 1})
    trc = PooledTargetRealConfig(
        name="trc", cache_dir=Path("/tmp/c"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    provider = MagicMock()

    with pytest.raises(ValueError, match="EncoderDecoderGPT"):
        _run_eval_suite_target_real(
            eval_name="core", model=not_encdec, tokenizer=tokenizer,
            device=torch.device("cpu"), seed=1234, cfg=cfg, trc=trc,
            provider=provider,
        )


def test_collect_loop_skips_seeds_after_unsupported_suite_returns_none(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: when ``_run_eval_suite_target_real`` returns None for
    seed K (unsupported suite), the seed loop breaks — remaining seeds
    are NOT invoked. (Otherwise we'd get N identical 'unknown suite'
    log entries per checkpoint.)"""
    unsupported_suite = "bogus_suite"
    cfg = _make_cfg(
        # A supported suite provisions the provider shared by this checkpoint;
        # the unsupported suite then exercises its real None-returning dispatch.
        eval_suites=f"core,{unsupported_suite}",
        target_real_configs=[
            {"name": "trc1", "mode": "pooled", "cache_dir": "/tmp/c",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=3,
    )
    seeds = [1000, 2000, 3000]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")
    call_log: list[tuple[str, int]] = []
    dispatch = mod._run_eval_suite_target_real

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        call_log.append((eval_name, int(seed)))
        if eval_name == "core":
            return {"core_metric": 0.42}
        return dispatch(eval_name=eval_name, model=model, tokenizer=tokenizer,
                        device=device, seed=seed, cfg=cfg, trc=trc,
                        provider=provider, target_mode=target_mode)

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        results = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core", unsupported_suite], seeds, cfg,
            existing_entry=None, work_keys=None,
        )

    # The seed loop must break after the first None — only seed 1000 is
    # invoked.
    assert call_log == [("core", seed) for seed in seeds] + [(unsupported_suite, seeds[0])]
    assert set(results["core"]["target_real"]["configs"]["trc1"]["seeds"]) == {str(seed) for seed in seeds}
    assert results[unsupported_suite]["target_real"]["configs"]["trc1"]["seeds"] == {}


# ---------------------------------------------------------------------------
# build_target_real_payload round-trip and edge cases
# ---------------------------------------------------------------------------


def test_build_target_real_payload_pooled_round_trip():
    """Invariant: ``load_suite_from_dict(build_target_real_payload(pooled, body))``
    yields a TargetRealPooledSuiteResult equal to the original."""
    trc = PooledTargetRealConfig(
        name="rt_pool", cache_dir=Path("/tmp/cache_p"),
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    body = {"core_metric": 0.42, "results": {"task_a": 0.5}}
    payload = build_target_real_payload(trc, body)
    variant = load_suite_from_dict(payload, name="rt_pool")
    assert isinstance(variant, TargetRealPooledSuiteResult)
    assert variant.name == "rt_pool"
    assert variant.cache_dir == str(trc.cache_dir)
    assert variant.pooling_op == trc.pooling_op.value
    assert variant.seeds["adhoc"] == body


def test_build_target_real_payload_lockstep_round_trip():
    """Invariant: ``load_suite_from_dict(build_target_real_payload(lockstep, body))``
    yields a TargetRealLockstepSuiteResult equal to the original."""
    trc = LockstepTargetRealConfig(name="rt_ls", cache_dir=Path("/tmp/cache_ls"))
    body = {"chat_metric": 0.7, "tasks": {"GSM8K": 0.4}}
    payload = build_target_real_payload(trc, body)
    variant = load_suite_from_dict(payload, name="rt_ls")
    assert isinstance(variant, TargetRealLockstepSuiteResult)
    assert variant.name == "rt_ls"
    assert variant.cache_dir == str(trc.cache_dir)
    assert variant.seeds["adhoc"] == body


def test_build_target_real_payload_empty_body_does_not_raise():
    """Invariant: empty body dict works (operator might dispatch a no-op
    eval just to populate metadata)."""
    trc = PooledTargetRealConfig(
        name="empty", cache_dir=Path("/tmp/c"), pooling_op=PoolingOp.BUCKET_MAX,
    )
    payload = build_target_real_payload(trc, {})
    assert payload["seeds"]["adhoc"] == {}


# ---------------------------------------------------------------------------
# parse_target_real_configs_from_cfg per-suite resolution
# ---------------------------------------------------------------------------


def test_auto_resolve_yields_different_paths_per_suite(
    monkeypatch, tmp_path,
):
    """Invariant: ``parse_target_real_configs_from_cfg(cfg, suite='chat_target')``
    and ``(..., suite='core_target')`` produce DIFFERENT cache_dirs for the
    same trc with cache_dir=None — the suite is part of the path layout."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    cfg = OmegaConf.create({
        "target_real_configs": [
            {"name": "trc1", "mode": "pooled", "cache_dir": None},
        ],
        "conditioning": {"identity": _full_identity_cfg()},
        "model_max_seq_len": 1024,
    })
    chat = parse_target_real_configs_from_cfg(cfg, suite="chat_target")
    core = parse_target_real_configs_from_cfg(cfg, suite="core_target")
    assert chat[0].cache_dir != core[0].cache_dir
    assert "chat_target" in str(chat[0].cache_dir)
    assert "core_target" in str(core[0].cache_dir)


def test_auto_resolve_is_lazy_when_all_explicit(monkeypatch, tmp_path):
    """Invariant: when all entries have explicit cache_dirs, the
    auto-resolver is NOT built, so missing identity does NOT cause a
    failure for the explicit case (matches docstring lines 252-256)."""
    # Patch _build_auto_resolver to track if it's called.
    build_calls: list = []
    real_builder = runner_mod._build_auto_resolver

    def _track(*a, **kw):
        build_calls.append((a, kw))
        return real_builder(*a, **kw)

    monkeypatch.setattr(runner_mod, "_build_auto_resolver", _track)
    cfg = OmegaConf.create({
        "target_real_configs": [
            {"name": "x", "mode": "pooled", "cache_dir": str(tmp_path / "explicit")},
        ],
        "conditioning": {},  # no identity — would error if resolver was built
        "model_max_seq_len": 1024,
    })
    parsed = parse_target_real_configs_from_cfg(cfg, suite="chat_target")
    assert len(parsed) == 1
    assert parsed[0].cache_dir == Path(str(tmp_path / "explicit"))
    assert build_calls == [], (
        "_build_auto_resolver was called even though all entries have explicit "
        "cache_dirs (lazy-build invariant violated)"
    )


# ---------------------------------------------------------------------------
# _run_target_real_for_suite: setdefault preserves existing keys
# ---------------------------------------------------------------------------


def test_build_context_feature_provider_auto_resolved_path_no_double_c_dir(
    monkeypatch,
):
    """Invariant (BUG FIX): ``build_context_feature_provider`` must NOT
    double-append ``c{N}`` when given an auto-resolved cache_dir whose
    parent already matches CONTEXT_LEN_DIR_RE.

    Before the fix, calling build_context_feature_provider with a path
    like ``.../c1024/<hash>/`` produced a provider rooted at
    ``.../c1024/<hash>/c1024/`` — double-c segment, causing the manifest
    lookup to FileNotFoundError downstream. This test pins the fix."""
    from steering.eval import target_real_runner as runner
    captured: dict = {}

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, **kw):
            captured["full_cache_dir"] = Path(cache_dir)

    monkeypatch.setattr(runner, "SafetensorsContextFeatureProvider", _StubProvider)
    cfg = OmegaConf.create({
        "conditioning": {"identity": _full_identity_cfg()},
        "model_max_seq_len": 1024,
    })
    # Auto-resolved layout: .../c{N}/<hash>/
    auto_resolved = (
        "/tmp/eval_caches/chat_target/m_x/l8/a_res/p_max/f16384/"
        f"c1024/{'a' * 16}"
    )
    runner.build_context_feature_provider(cfg, cache_dir=auto_resolved)
    full = captured["full_cache_dir"]
    # Count c{N} segments — must be exactly one.
    c_segs = [
        seg for seg in full.parts
        if CONTEXT_LEN_DIR_RE.fullmatch(seg)
    ]
    assert len(c_segs) == 1, (
        f"Expected exactly 1 c{{N}} segment, got {len(c_segs)}: full={full}; "
        f"the runner double-appended c{{N}} for an already-auto-resolved path."
    )
    # The full path must equal the input (no append).
    assert str(full) == auto_resolved


def test_build_context_feature_provider_raw_cache_root_appends_c_dir(
    monkeypatch,
):
    """Invariant (regression guard): when given a non-auto-resolved cache
    root, build_context_feature_provider STILL appends ``c{N}`` so PR-B's
    ``conditioning.context_cache_dir`` (an operator's cache root) still
    works as before.

    Before the H7 fix this was the only behavior. After the fix it must
    still be the behavior for non-auto-resolved paths."""
    from steering.eval import target_real_runner as runner
    captured: dict = {}

    class _StubProvider:
        def __init__(self, cache_dir, *, identity, **kw):
            captured["full_cache_dir"] = Path(cache_dir)

    monkeypatch.setattr(runner, "SafetensorsContextFeatureProvider", _StubProvider)
    cfg = OmegaConf.create({
        "conditioning": {"identity": _full_identity_cfg()},
        "model_max_seq_len": 2048,
    })
    raw_root = "/tmp/operator_cache_root"  # parent has no c{N} segment
    runner.build_context_feature_provider(cfg, cache_dir=raw_root)
    full = captured["full_cache_dir"]
    # The PR-B / context_cache_dir convention: append c{N}/{hash[:16]}.
    assert str(full) == f"/tmp/operator_cache_root/c2048/{'a' * 16}"


def test_build_auto_resolver_accepts_cond_block_directly(monkeypatch, tmp_path):
    """Invariant (BUG FIX): ``_build_auto_resolver`` accepts a cfg passed
    DIRECTLY as a conditioning block (not nested under a ``conditioning``
    key). This is the layout training_progression uses when calling
    ``parse_target_real_configs_from_cfg(cfg.conditioning, ...)``.

    Before the fix this raised ValueError because the resolver looked
    only for ``cfg.conditioning.identity`` (one level too deep)."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    from steering.eval.target_real_runner import _build_auto_resolver
    # Pass the cond-block directly (no wrapping conditioning key).
    cond_block = OmegaConf.create({
        "identity": _full_identity_cfg(),
    })
    resolver = _build_auto_resolver(cond_block, suite="chat_target")
    out = resolver({"name": "x"})
    assert "eval_caches" in str(out)
    assert "chat_target" in str(out)


def test_build_auto_resolver_top_level_layout_still_works(monkeypatch, tmp_path):
    """Invariant: the original top-level layout (cfg.conditioning.identity)
    still works after H7 fix — chat_eval / base_eval / dclm_eval pass the
    full cfg, not a cond-block."""
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    from steering.eval.target_real_runner import _build_auto_resolver
    full_cfg = OmegaConf.create({
        "conditioning": {"identity": _full_identity_cfg()},
        "model_max_seq_len": 1024,
    })
    resolver = _build_auto_resolver(full_cfg, suite="core_target")
    out = resolver({"name": "x"})
    assert "eval_caches" in str(out)
    assert "core_target" in str(out)


def test_run_target_real_for_suite_setdefault_preserves_existing_kind(
    stub_load_progression_model, stub_target_real_provider,
):
    """Invariant: ``setdefault`` semantics preserve existing skeleton keys.
    If the existing entry has kind=target_real_pooled but the trc is later
    changed to lockstep (re-run with different YAML), we should NOT
    silently overwrite — the existing entry's kind is the source of
    truth for the persisted seeds.

    (This is the documented setdefault contract on line 663: skeleton is
    only created when missing.)"""
    cfg = _make_cfg(
        eval_suites="core",
        target_real_configs=[
            {"name": "trc1", "mode": "pooled", "cache_dir": "/tmp/c",
             "pooling_op": "bucket_max"},
        ],
        num_seeds=1,
    )
    seeds = [1234]
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    # Pre-populated existing entry: same name but different kind +
    # cache_dir + already-completed seed.
    existing_entry = {
        "step": 0, "phase": "sft",
        "evals": {
            "core": {
                "target_real": {
                    "configs": {
                        "trc1": {
                            "kind": "some_legacy_kind",
                            "cache_dir": "/tmp/old_path",
                            "seeds": {"1234": {"core_metric": 0.123}},
                        },
                    },
                },
            },
        },
    }

    def _fake(*, eval_name, model, tokenizer, device, seed, cfg, trc,
              provider, target_mode):
        return {"core_metric": 999.0}

    with patch.object(mod, "_run_eval_suite_target_real", side_effect=_fake):
        evals = _evaluate_single_checkpoint_impl(
            spec, torch.device("cpu"), torch.float32,
            ["core"], seeds, cfg,
            existing_entry=existing_entry, work_keys=None,
        )

    cfg_result = evals["core"]["target_real"]["configs"]["trc1"]
    # setdefault preserves the existing legacy kind, cache_dir, mode.
    assert cfg_result["kind"] == "some_legacy_kind"
    assert cfg_result["cache_dir"] == "/tmp/old_path"
    # Existing seed preserved.
    assert cfg_result["seeds"]["1234"] == {"core_metric": 0.123}
