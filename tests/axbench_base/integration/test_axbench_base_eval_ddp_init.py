"""Regression tests for the AxBench-Base CLI distributed init wiring.

Three reviewers caught that ``scripts/eval/axbench_base_eval._run`` was
silently producing ``rank=0, world_size=1`` for every process when
launched under ``torchrun --nproc_per_node=N``: the CLI never called
``compute_init`` so the process group was never initialised, and the
``_resolve_ddp_coordinates`` helper short-circuited to ``(0, 1)``.

The fix calls :func:`nanochat.common.compute_init` BEFORE
:func:`_resolve_ddp_coordinates` so each rank sees its own rank /
world_size, and pairs it with :func:`nanochat.common.compute_cleanup`
in the outer ``finally`` so the process group is destroyed even when
``_dispatch_stage`` raises.

These tests pin the contracts:

1. Under a real 2-rank ``gloo_multiprocess`` spawn that emulates
   torchrun's ``RANK`` / ``WORLD_SIZE`` env vars, each rank's
   :class:`RuntimeState` is constructed with a DIFFERENT
   ``ddp_rank`` and the same ``ddp_world_size == 2``.

2. Single-GPU (no torchrun env vars) does not call
   ``init_process_group`` — the call falls through to a plain
   ``torch.device(device_type)`` and the process group remains
   uninitialised.

3. ``compute_cleanup`` is invoked even when ``_dispatch_stage``
   raises, so the process group is always destroyed before ``_run``
   returns. Verified with a spy on ``destroy_process_group``.

4. Non-Stage-0c stages run on rank 0 only under DDP. A worker that
   passes through ``_dispatch_stage`` with a non-rank-aware stage
   verifies only rank 0 invoked the stage handler; rank 1 returned
   without doing the work (rank-0 gate from PR-4 review fix #2).

5. Every resource-acquisition boundary after configuration validation
   releases ``RuntimeState`` and the distributed process group exactly
   once, including failures from compute initialization, profiler setup,
   context entry/exit, dispatch, and state closure.
"""
# Spawn targets cannot use pytest's monkeypatch fixture, and direct assignment
# conflicts with the production function types under ty.
# ruff: noqa: B010

from __future__ import annotations

import os
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch.distributed as dist
from omegaconf import OmegaConf

from scripts.eval import axbench_base_eval as cli
from steering.eval.axbench_base import profiler as profiler_mod
from steering.eval.axbench_base import runtime_state as rs_mod
from steering.eval.axbench_base.negative_prefixes import (
    _broadcast_resume_decision,
)
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.types import AxBenchBaseError, Stage
from tests.axbench_base import _multiprocess_helpers as mp_helpers


def _build_minimal_cfg(
    tmp_path_str: str,
    *,
    stage: str = "0a",
    output_root: str | None = None,
    run_tag: str | None = "test-ddp-init",
) -> Any:
    """Build a ``DictConfig``-shaped object minimally sufficient for ``_run``.

    The CLI only reads a small subset of cfg fields before dispatching:
    feature_dir, feature_statistics_path, decoder_only, encdec, device,
    output_root, run_tag, stage, run_negative_mode. Everything else
    short-circuits before touching heavy paths because we stub the
    stage handler.
    """
    return OmegaConf.create({
        "feature_dir": str(Path(tmp_path_str) / "features" / "s128"),
        "feature_statistics_path": str(Path(tmp_path_str) / "stats.safetensors"),
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cpu",
        "output_root": output_root or str(Path(tmp_path_str) / "eval_out"),
        "run_tag": run_tag,
        "stage": stage,
        "run_negative_mode": False,
        "max_concepts": 1,
        "n_verified_prefixes_per_concept_target": 1,
        "verification_length": 4,
    })


def _stub_validate_env(stages: Any, *, cfg: Any) -> None:
    """No-op env validator: prevents the test from caring about API keys."""
    del stages, cfg


def _stub_runtime_state_close(self: Any) -> None:
    """No-op RuntimeState.close so we don't touch any real heavy objects."""
    del self


class _MockCfg:
    """Minimal attribute config for direct ``_dispatch_stage`` tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.output_root = kwargs.pop("output_root", None)
        for k, v in kwargs.items():
            setattr(self, k, v)


def _ddp_init_target(rank: int, world_size: int, tmp_path_str: str) -> dict[str, Any]:
    """Worker target: invoke ``_run`` with stubbed dispatch + env validate.

    Returns the (ddp_rank, ddp_world_size) the RuntimeState saw.

    Worker imports must work after a ``spawn`` fork, but every module
    we touch is already imported in the parent (top-of-file imports
    above) and re-imported in the child by name when this function is
    looked up — so the helpers reach modules at module scope without
    in-function imports.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    seen: dict[str, Any] = {
        "stages_invoked": [], "ddp_rank": None, "ddp_world_size": None,
    }

    def _spy_run_single_stage(cfg: Any, stage: Any, state: Any) -> None:
        del cfg
        seen["stages_invoked"].append(str(Stage.coerce(stage).value))
        seen["ddp_rank"] = int(state.ddp_rank)
        seen["ddp_world_size"] = int(state.ddp_world_size)

    # Use Stage 0c (PREPARE_NEGATIVE_PREFIXES) — the rank-aware stage.
    # Both ranks invoke the stage handler so the spy records each
    # rank's RuntimeState ddp_rank / ddp_world_size.
    cfg = _build_minimal_cfg(tmp_path_str, stage="0c")
    # Stub the env validator and the per-stage handler. We do NOT stub
    # compute_init / compute_cleanup — those are the unit under test.
    setattr(cli, "validate_env_for_stages", _stub_validate_env)
    setattr(cli, "_run_single_stage", _spy_run_single_stage)
    rs_mod.RuntimeState.close = _stub_runtime_state_close
    # Config validation is outside this wiring test. Keep the real
    # ``cfg_to_dataclass`` conversion so the fixture follows newly added
    # defaulted config fields instead of drifting from the CLI contract.
    setattr(cli, "validate_config", lambda cfg_obj: None)
    # Skip dependency check (we stubbed the stage handler).
    setattr(cli, "check_stage_dependencies", lambda c, s: None)

    cli._run(cfg)
    return seen


def test_run_under_torchrun_assigns_distinct_ranks(tmp_path, gloo_multiprocess):
    """When 2 worker processes are spawned with the gloo backend (which
    emulates torchrun's RANK / WORLD_SIZE / LOCAL_RANK env vars), each
    rank's RuntimeState must see its own ``ddp_rank`` and a shared
    ``ddp_world_size == 2``. Without compute_init() being called inside
    ``_run`` this would silently collapse to (0, 1) on every rank."""
    results = gloo_multiprocess(2, _ddp_init_target, str(tmp_path))
    rank0 = results[0]
    rank1 = results[1]

    assert rank0["ddp_rank"] == 0, f"rank 0 saw ddp_rank={rank0['ddp_rank']!r}"
    assert rank1["ddp_rank"] == 1, f"rank 1 saw ddp_rank={rank1['ddp_rank']!r}"
    assert rank0["ddp_world_size"] == 2
    assert rank1["ddp_world_size"] == 2


def _ddp_autotag_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
) -> dict[str, Any]:
    del world_size
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    base_time = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)

    class _RankSkewedDateTime:
        @classmethod
        def now(cls, *, tz: Any) -> datetime:
            del cls
            return (base_time + timedelta(seconds=rank)).astimezone(tz)

    seen: dict[str, Any] = {}

    def _spy_run_single_stage(cfg: Any, stage: Any, state: Any) -> None:
        del stage
        seen["run_tag"] = cfg.run_tag
        seen["output_root"] = str(state.output_paths().output_root)

    cfg = _build_minimal_cfg(
        tmp_path_str,
        stage="0c",
        run_tag=None,
    )
    setattr(cli, "_DateTime", _RankSkewedDateTime)
    setattr(cli, "validate_env_for_stages", _stub_validate_env)
    setattr(cli, "_run_single_stage", _spy_run_single_stage)
    setattr(cli, "validate_config", lambda cfg_obj: None)
    setattr(cli, "check_stage_dependencies", lambda c, s: None)
    rs_mod.RuntimeState.close = _stub_runtime_state_close  # type: ignore[assignment]

    cli._run(cfg)
    return seen


def test_ddp_autogenerated_tag_is_broadcast_from_rank_zero_despite_clock_skew(
    tmp_path: Path,
    gloo_multiprocess: Any,
) -> None:
    results = gloo_multiprocess(2, _ddp_autotag_target, str(tmp_path))

    assert results[0]["run_tag"] == results[1]["run_tag"]
    assert results[0]["run_tag"].startswith("20260729-120000-")
    assert results[0]["output_root"] == results[1]["output_root"]
    assert results[0]["output_root"] == str(tmp_path / "eval_out")


def _ddp_mismatched_root_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
) -> dict[str, Any]:
    del world_size
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    dispatched = False

    def _unexpected_dispatch(cfg: Any, stage: Any, state: Any) -> None:
        del cfg, stage, state
        nonlocal dispatched
        dispatched = True

    cfg = _build_minimal_cfg(
        tmp_path_str,
        stage="0c",
        output_root=str(Path(tmp_path_str) / f"rank-{rank}-root"),
        run_tag="mismatched-root",
    )
    setattr(cli, "validate_env_for_stages", _stub_validate_env)
    setattr(cli, "_run_single_stage", _unexpected_dispatch)
    setattr(cli, "validate_config", lambda cfg_obj: None)
    setattr(cli, "check_stage_dependencies", lambda c, s: None)
    rs_mod.RuntimeState.close = _stub_runtime_state_close  # type: ignore[assignment]

    try:
        cli._run(cfg)
    except AxBenchBaseError as exc:
        return {
            "error": str(exc),
            "dispatched": dispatched,
            "root_exists": (
                Path(tmp_path_str) / f"rank-{rank}-root/mismatched-root"
            ).exists(),
        }
    raise AssertionError("rank-varying output roots were accepted")


def test_ddp_rank_varying_output_roots_fail_collectively_before_writes(
    tmp_path: Path,
    gloo_multiprocess: Any,
) -> None:
    results = gloo_multiprocess(2, _ddp_mismatched_root_target, str(tmp_path))

    assert all(
        "DDP ranks resolved different output roots" in result["error"]
        for result in results
    )
    assert all(result["dispatched"] is False for result in results)
    assert all(result["root_exists"] is False for result in results)


def test_ddp_output_root_agreement_accepts_identical_rank_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path)))
    assert cfg.output_root is not None
    monkeypatch.setattr(cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)

    def gather_identical(output: list[Any], local: Any) -> None:
        output[:] = [local, dict(local)]

    monkeypatch.setattr(cli.dist, "all_gather_object", gather_identical)

    cli._assert_ddp_output_root_agreement(cfg, world_size=2)


def test_ddp_output_root_agreement_reports_resolution_failure_collectively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path)))
    monkeypatch.setattr(cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        cli.paths,
        "effective_output_root",
        lambda _root: (_ for _ in ()).throw(OSError("unresolvable root")),
    )

    def gather_errors(output: list[Any], local: Any) -> None:
        output[:] = [local, dict(local)]

    monkeypatch.setattr(cli.dist, "all_gather_object", gather_errors)

    with pytest.raises(
        AxBenchBaseError,
        match="could not resolve output_root.*unresolvable root",
    ):
        cli._assert_ddp_output_root_agreement(cfg, world_size=2)


def test_ddp_output_root_agreement_rejects_missing_or_different_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path)))
    assert cfg.output_root is not None
    monkeypatch.setattr(cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)
    gathered_payloads: list[list[Any]] = [
        [None, None],
        [
            {
                "configured_output_root": cfg.output_root,
                "effective_output_root": str(Path(cfg.output_root)),
                "resolved_output_root": str(Path(cfg.output_root).resolve()),
                "error": None,
            },
            {
                "configured_output_root": "different",
                "effective_output_root": "different",
                "resolved_output_root": "different",
                "error": None,
            },
        ],
    ]

    def gather_next(output: list[Any], _local: Any) -> None:
        output[:] = gathered_payloads.pop(0)

    monkeypatch.setattr(cli.dist, "all_gather_object", gather_next)

    with pytest.raises(AxBenchBaseError, match="could not resolve output_root"):
        cli._assert_ddp_output_root_agreement(cfg, world_size=2)
    with pytest.raises(AxBenchBaseError, match="different output roots"):
        cli._assert_ddp_output_root_agreement(cfg, world_size=2)


def test_ddp_run_tag_synchronization_covers_generated_and_explicit_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path), run_tag="local"))
    monkeypatch.setattr(cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)

    def broadcast_rank0(payload: list[Any], *, src: int) -> None:
        assert src == 0
        payload[0] = "rank-zero-tag"

    monkeypatch.setattr(cli.dist, "broadcast_object_list", broadcast_rank0)
    synchronized = cli._synchronize_ddp_run_tag(
        cfg,
        rank=1,
        world_size=2,
        generated=True,
    )
    assert synchronized.run_tag == "rank-zero-tag"

    def gather_matching(output: list[Any], local: Any) -> None:
        output[:] = [local, local]

    monkeypatch.setattr(cli.dist, "all_gather_object", gather_matching)
    assert (
        cli._synchronize_ddp_run_tag(
            cfg,
            rank=0,
            world_size=2,
            generated=False,
        )
        is cfg
    )

    def gather_mismatch(output: list[Any], _local: Any) -> None:
        output[:] = ["rank-zero", "rank-one"]

    monkeypatch.setattr(cli.dist, "all_gather_object", gather_mismatch)
    with pytest.raises(AxBenchBaseError, match="different explicit run_tag"):
        cli._synchronize_ddp_run_tag(
            cfg,
            rank=0,
            world_size=2,
            generated=False,
        )

    monkeypatch.setattr(
        cli.dist,
        "broadcast_object_list",
        lambda payload, *, src: payload.__setitem__(0, None),
    )
    with pytest.raises(AxBenchBaseError, match="invalid autogenerated run_tag"):
        cli._synchronize_ddp_run_tag(
            cfg,
            rank=1,
            world_size=2,
            generated=True,
        )


@pytest.mark.parametrize(
    ("preflight_error", "expected_error"),
    [
        (None, None),
        (AxBenchBaseError("typed preflight failure"), "typed preflight failure"),
        (
            OSError("raw preflight failure"),
            "AxBench-Base legacy-output shadow preflight failed on rank 0: "
            "OSError: raw preflight failure",
        ),
    ],
)
def test_ddp_legacy_shadow_preflight_broadcasts_rank_zero_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight_error: Exception | None,
    expected_error: str | None,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path)))
    calls: list[dict[str, Any]] = []

    def preflight(_cfg: Any) -> None:
        if preflight_error is not None:
            raise preflight_error

    monkeypatch.setattr(cli.paths, "assert_no_legacy_output_shadow", preflight)
    monkeypatch.setattr(
        cli,
        "_broadcast_resume_decision",
        lambda decision, **kwargs: calls.append(
            {"decision": decision, **kwargs}
        ),
    )

    cli._assert_no_legacy_output_shadow_ddp(cfg, rank=0, world_size=2)

    assert calls == [
        {
            "decision": None,
            "rank": 0,
            "world_size": 2,
            "resume_error": expected_error,
        }
    ]


def test_ddp_legacy_shadow_worker_only_receives_rank_zero_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = cli.cfg_to_dataclass(_build_minimal_cfg(str(tmp_path)))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli.paths,
        "assert_no_legacy_output_shadow",
        lambda _cfg: pytest.fail("worker rank ran filesystem preflight"),
    )
    monkeypatch.setattr(
        cli,
        "_broadcast_resume_decision",
        lambda decision, **kwargs: calls.append(
            {"decision": decision, **kwargs}
        ),
    )

    cli._assert_no_legacy_output_shadow_ddp(cfg, rank=1, world_size=2)

    assert calls == [
        {
            "decision": None,
            "rank": 1,
            "world_size": 2,
            "resume_error": None,
        }
    ]


@pytest.mark.parametrize(
    ("configured_run_tag", "clustered", "expected_tag", "expected_events"),
    [
        (
            "explicit-cluster-tag",
            True,
            "explicit-cluster-tag",
            ["root", "tag:explicit-cluster-tag:False", "preflight", "resolve"],
        ),
        (
            None,
            True,
            "generated-cluster-tag",
            ["root", "resolve", "tag:generated-cluster-tag:True", "preflight"],
        ),
        (
            None,
            False,
            "generated-unclustered-tag",
            ["root", "tag:generated-unclustered-tag:True", "preflight"],
        ),
    ],
)
def test_run_synchronizes_identity_around_optional_contract_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_run_tag: str | None,
    clustered: bool,
    expected_tag: str,
    expected_events: list[str],
) -> None:
    """Pin explicit and contract-derived tag ordering without process mocks."""
    cfg = _build_minimal_cfg(
        str(tmp_path),
        stage="0c",
        run_tag=configured_run_tag,
    )
    events: list[str] = []
    seen_state: dict[str, Any] = {}
    contract_request = object()
    contract = SimpleNamespace(checkpoint=SimpleNamespace(step=123))

    class FakeRuntimeState:
        def __init__(self, **kwargs: Any) -> None:
            seen_state.update(kwargs)

        def close(self) -> None:
            events.append("state.close")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(
        cli,
        "_cluster_contract_request",
        lambda _cfg: contract_request if clustered else None,
    )
    monkeypatch.setattr(cli, "_generate_run_tag", lambda _cfg: expected_tag)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda **_kwargs: (False, 0, 0, 1, "cpu"),
    )
    monkeypatch.setattr(cli, "compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(cli, "_resolve_ddp_coordinates", lambda: (0, 1))
    monkeypatch.setattr(
        cli,
        "_assert_ddp_output_root_agreement",
        lambda _cfg, **_kwargs: events.append("root"),
    )

    def synchronize(
        cfg_obj: Any,
        *,
        rank: int,
        world_size: int,
        generated: bool,
    ) -> Any:
        assert (rank, world_size) == (0, 1)
        events.append(f"tag:{cfg_obj.run_tag}:{generated}")
        return cfg_obj

    monkeypatch.setattr(cli, "_synchronize_ddp_run_tag", synchronize)
    monkeypatch.setattr(
        cli,
        "_assert_no_legacy_output_shadow_ddp",
        lambda _cfg, **_kwargs: events.append("preflight"),
    )

    def resolve(request: Any) -> Any:
        assert request is contract_request
        events.append("resolve")
        return contract

    monkeypatch.setattr(cli, "resolve_cluster_feature_contract_ddp", resolve)
    monkeypatch.setattr(cli, "validate_env_for_stages", _stub_validate_env)
    monkeypatch.setattr(cli, "_install_cluster_marker_ddp", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(cli, "_dispatch_stage", lambda _cfg, _state: None)

    cli._run(cfg)

    assert seen_state["cfg"].run_tag == expected_tag
    assert seen_state["cfg"].encdec["step"] == (123 if clustered else 10000)
    assert seen_state["cluster_contract"] is (contract if clustered else None)
    assert events == [*expected_events, "state.close", "cleanup"]


def test_run_single_gpu_does_not_call_init_process_group(tmp_path, monkeypatch):
    """Single-GPU mode: no RANK / WORLD_SIZE env vars set. The
    ``compute_init`` call inside ``_run`` should NOT call
    ``init_process_group`` (which would fail without MASTER_ADDR /
    MASTER_PORT) — it falls through to a plain device.

    We verify by spying on ``dist.init_process_group`` and asserting it
    was never invoked while ``_run`` ran to completion. The test also
    confirms ``_run`` does not raise.
    """
    # Ensure no torchrun env vars are set for this test.
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    init_calls: list[Any] = []

    def _spy_init_process_group(*args: Any, **kwargs: Any) -> None:
        init_calls.append((args, kwargs))

    monkeypatch.setattr(dist, "init_process_group", _spy_init_process_group)

    cfg = _build_minimal_cfg(str(tmp_path), stage="0a")
    monkeypatch.setattr(cli, "validate_env_for_stages", _stub_validate_env)
    monkeypatch.setattr(cli, "validate_config", lambda cfg_obj: None)
    monkeypatch.setattr(cli, "_run_single_stage", lambda cfg, stage, state: None)
    monkeypatch.setattr(cli, "check_stage_dependencies", lambda c, s: None)
    monkeypatch.setattr(rs_mod.RuntimeState, "close", _stub_runtime_state_close)

    cli._run(cfg)

    assert init_calls == [], (
        f"single-GPU run unexpectedly called init_process_group: {init_calls!r}"
    )


def test_run_calls_compute_cleanup_on_exception(tmp_path, monkeypatch):
    """compute_cleanup must run on EVERY exit path, including exceptions.

    We force ``_dispatch_stage`` to raise and assert ``compute_cleanup``
    fired anyway. Without it, the process group would leak on the next
    run (init_process_group("nccl") would error with "process group
    already initialized").
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)

    cleanup_calls: list[int] = []

    def _spy_cleanup() -> None:
        cleanup_calls.append(1)

    monkeypatch.setattr(cli, "compute_cleanup", _spy_cleanup)

    def _raising_dispatch(cfg: Any, state: Any) -> None:
        del cfg, state
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(cli, "_dispatch_stage", _raising_dispatch)
    monkeypatch.setattr(cli, "validate_env_for_stages", _stub_validate_env)
    monkeypatch.setattr(cli, "validate_config", lambda cfg_obj: None)

    cfg = _build_minimal_cfg(str(tmp_path), stage="0a")
    monkeypatch.setattr(rs_mod.RuntimeState, "close", _stub_runtime_state_close)

    with pytest.raises(RuntimeError, match="simulated stage failure"):
        cli._run(cfg)

    assert len(cleanup_calls) == 1, (
        f"compute_cleanup must run exactly once on exception path; got {cleanup_calls!r}"
    )


@pytest.mark.parametrize(
    ("failure_point", "expected_events"),
    [
        (
            "compute_init",
            ["compute_init", "compute_cleanup"],
        ),
        (
            "profile_device",
            [
                "compute_init",
                "state.init",
                "profile_device",
                "state.close",
                "compute_cleanup",
            ],
        ),
        (
            "profile_enter",
            [
                "compute_init",
                "state.init",
                "profile_device",
                "profile.init",
                "profile.enter",
                "state.close",
                "compute_cleanup",
            ],
        ),
        (
            "dispatch",
            [
                "compute_init",
                "state.init",
                "profile_device",
                "profile.init",
                "profile.enter",
                "dispatch",
                "state.close",
                "profile.exit:RuntimeError",
                "compute_cleanup",
            ],
        ),
        (
            "state_close",
            [
                "compute_init",
                "state.init",
                "profile_device",
                "profile.init",
                "profile.enter",
                "dispatch",
                "state.close",
                "profile.exit:RuntimeError",
                "compute_cleanup",
            ],
        ),
        (
            "profile_exit",
            [
                "compute_init",
                "state.init",
                "profile_device",
                "profile.init",
                "profile.enter",
                "dispatch",
                "state.close",
                "profile.exit:None",
                "compute_cleanup",
            ],
        ),
    ],
)
def test_run_tears_down_each_pre_dispatch_and_context_failure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_events: list[str],
) -> None:
    """Every resource acquired by ``_run`` is released exactly once.

    These are distinct Python control-flow boundaries: a failed context-manager
    ``__enter__`` does not invoke ``__exit__``, while an exception from the
    body (including ``RuntimeState.close``) does. Pinning the complete event
    order prevents a superficially plausible ``finally`` rearrangement from
    silently leaking either the runtime state or distributed process group.
    """
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    events: list[str] = []

    class FakeRuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("state.init")

        def close(self) -> None:
            events.append("state.close")
            if failure_point == "state_close":
                raise RuntimeError("injected state_close failure")

    class FakeProfiler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("profile.init")

        def __enter__(self) -> FakeProfiler:
            events.append("profile.enter")
            if failure_point == "profile_enter":
                raise RuntimeError("injected profile_enter failure")
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: Any,
        ) -> None:
            events.append(
                f"profile.exit:{exc_type.__name__ if exc_type is not None else 'None'}"
            )
            if failure_point == "profile_exit":
                raise RuntimeError("injected profile_exit failure")

    def fake_compute_init(**_kwargs: Any) -> tuple[bool, int, int, int, str]:
        events.append("compute_init")
        if failure_point == "compute_init":
            raise RuntimeError("injected compute_init failure")
        return False, 0, 0, 1, "cpu"

    def fake_profile_device(*_args: Any, **_kwargs: Any) -> int:
        events.append("profile_device")
        if failure_point == "profile_device":
            raise RuntimeError("injected profile_device failure")
        return 0

    def fake_dispatch(_cfg: Any, _state: Any) -> None:
        events.append("dispatch")
        if failure_point == "dispatch":
            raise RuntimeError("injected dispatch failure")

    cfg = _build_minimal_cfg(str(tmp_path), stage="0a")
    cfg.profile = OmegaConf.create(
        {
            "enabled": True,
            "output_dir": None,
            "synchronize_cuda": False,
            "gpu_utilization": False,
            "gpu_poll_interval_ms": 50,
            "gpu_idle_threshold": 5.0,
        }
    )
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(cli, "validate_env_for_stages", _stub_validate_env)
    monkeypatch.setattr(cli, "compute_init", fake_compute_init)
    monkeypatch.setattr(
        cli,
        "compute_cleanup",
        lambda: events.append("compute_cleanup"),
    )
    monkeypatch.setattr(cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(
        cli,
        "resolve_profile_gpu_device_index",
        fake_profile_device,
    )
    monkeypatch.setattr(cli, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(cli, "_install_cluster_marker_ddp", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        cli._run(cfg)

    assert events == expected_events
    assert events.count("compute_cleanup") == 1
    assert events.count("state.close") == (0 if failure_point == "compute_init" else 1)


def _rank_gate_target(
    rank: int, world_size: int, tmp_path_str: str,
) -> dict[str, list[str]]:
    """Worker target: drive ``_dispatch_stage`` with a non-rank-aware
    stage. Records every stage handler invocation so the parent can
    verify rank 0 ran it and rank 1 skipped."""
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    invocations: list[str] = []

    def _spy_run_single_stage(cfg: Any, stage: Any, state: Any) -> None:
        del cfg, state
        invocations.append(str(Stage.coerce(stage).value))

    setattr(cli, "_run_single_stage", _spy_run_single_stage)
    setattr(cli, "check_stage_dependencies", lambda c, s: None)

    cfg: Any = _MockCfg(
        stage="0a", run_negative_mode=False, run_tag="rank-gate-test",
        device="cpu",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=rank, ddp_world_size=world_size)
    try:
        cli._dispatch_stage(cfg, state)
    finally:
        try:
            state.close()
        except Exception:
            pass

    return {"rank": [str(rank)], "stages_invoked": invocations}


def test_dispatch_stage_rank_0_only_under_ddp(tmp_path, gloo_multiprocess):
    """Non-rank-aware stages run on rank 0 only under DDP.

    Stage 0a (PREPARE_CONCEPTS) is NOT in :data:`_RANK_AWARE_STAGES`, so
    the dispatcher must:
    - call the stage handler on rank 0
    - skip the handler on rank 1 (and barrier instead)

    Without this gate, every stage would race-write SQLite caches,
    multiply OpenRouter API calls by world_size, and corrupt shared
    on-disk artifacts.
    """
    results = gloo_multiprocess(2, _rank_gate_target, str(tmp_path))
    rank0 = results[0]
    rank1 = results[1]

    assert rank0["stages_invoked"] == ["0a"], (
        f"rank 0 must invoke stage 0a; got {rank0['stages_invoked']!r}"
    )
    assert rank1["stages_invoked"] == [], (
        f"rank 1 must skip non-rank-aware stage 0a; got {rank1['stages_invoked']!r}"
    )


def test_stage_runs_on_rank_classifies_correctly():
    """Direct unit check on the helper: rank-aware stages run on every
    rank; everything else gates to rank 0."""
    # Rank-aware stage: every rank.
    assert cli._stage_runs_on_rank(Stage.PREPARE_NEGATIVE_PREFIXES, rank=0)
    assert cli._stage_runs_on_rank(Stage.PREPARE_NEGATIVE_PREFIXES, rank=1)
    assert cli._stage_runs_on_rank(Stage.PREPARE_NEGATIVE_PREFIXES, rank=7)
    # Non-rank-aware stage: rank 0 only.
    for stage in (
        Stage.PREPARE_CONCEPTS,
        Stage.SYNTH_POSITIVE_MODE,
        Stage.GENERATE,
        Stage.JUDGE,
        Stage.AGGREGATE,
    ):
        assert cli._stage_runs_on_rank(stage, rank=0), stage
        assert not cli._stage_runs_on_rank(stage, rank=1), stage
        assert not cli._stage_runs_on_rank(stage, rank=3), stage


def test_device_type_from_cfg():
    """``_device_type_from_cfg`` extracts the bare device type so the
    CLI can pass through ``cuda:0`` etc. from the cfg."""
    assert cli._device_type_from_cfg("cuda") == "cuda"
    assert cli._device_type_from_cfg("cuda:0") == "cuda"
    assert cli._device_type_from_cfg("cuda:7") == "cuda"
    assert cli._device_type_from_cfg("cpu") == "cpu"
    assert cli._device_type_from_cfg("mps") == "mps"
    # CASE-INSENSITIVE for forwards-compat.
    assert cli._device_type_from_cfg("CUDA") == "cuda"
    # Unsupported types raise.
    with pytest.raises(AxBenchBaseError, match="Unsupported"):
        cli._device_type_from_cfg("xpu")


# ---------------------------------------------------------------------------
# PR-4 second-round review: helper-level coverage (Tier 2 tests 19, 20, 21)
# ---------------------------------------------------------------------------


def test_maybe_barrier_no_op_at_world_size_1(monkeypatch):
    """At world_size=1 the helper is a no-op — never touches dist."""
    fired: list[int] = []

    def _spy_barrier(*args, **kwargs):
        fired.append(1)

    monkeypatch.setattr(dist, "barrier", _spy_barrier)
    cli._maybe_barrier(1)
    assert fired == [], "barrier must not fire at world_size=1"


def test_maybe_barrier_invokes_dist_when_world_size_gt_1_and_initialized(
    monkeypatch,
):
    """At world_size > 1 AND when ``dist.is_initialized()`` is True, the
    helper calls ``dist.barrier``."""
    fired: list[int] = []

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda: fired.append(1))
    cli._maybe_barrier(2)
    assert fired == [1], "barrier must fire exactly once at ws>1+initialized"


def test_maybe_barrier_no_op_when_not_initialized(monkeypatch):
    """At world_size > 1 BUT ``dist.is_initialized()`` is False (e.g.,
    single-GPU run that nonetheless asks for ws>1), the helper must NOT
    invoke ``dist.barrier`` — that would error."""
    fired: list[int] = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "barrier", lambda: fired.append(1))
    cli._maybe_barrier(2)
    assert fired == [], (
        f"barrier must not fire when not initialized; got {fired!r}"
    )


def test_resolve_ddp_coordinates_uses_dist_when_initialized(monkeypatch):
    """When the process group is initialized, the resolver reads
    ``dist.get_rank`` / ``dist.get_world_size``.

    This test pins the contract: a regression that fell back to (0, 1)
    even when initialized would silently break multi-GPU dispatch.
    """
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    rank, world_size = cli._resolve_ddp_coordinates()
    assert rank == 3
    assert world_size == 8


def test_resolve_ddp_coordinates_returns_0_1_when_not_initialized(monkeypatch):
    """Single-GPU path: when the process group is NOT initialized the
    resolver returns the default (0, 1) tuple."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    rank, world_size = cli._resolve_ddp_coordinates()
    assert rank == 0
    assert world_size == 1


# ---------------------------------------------------------------------------
# Bugs C-1/C-2 regression: rank-divergent resume short-circuit must not
# deadlock. The fix broadcasts rank-0's decision to every rank.
# ---------------------------------------------------------------------------


def _resume_short_circuit_target(
    rank: int, world_size: int, tmp_path_str: str,
) -> dict[str, Any]:
    """Worker target: rank 0 sees the artifacts as complete; rank 1
    sees them as missing. The broadcast-resume helper must coalesce
    these views to a single decision for both ranks.

    Without the fix: rank 0 returns early; rank 1 enters
    ``run_single_rank`` and blocks on ``all_gather_object`` forever.

    With the fix: rank 1 also sees rank 0's "resume" path and returns
    immediately. We assert both ranks return WITHIN a 10-second
    deadline by virtue of the gloo_multiprocess timeout (the helper
    itself has a 60s default; the test passes when both ranks
    actually return).
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    canonical = Path(tmp_path_str) / "canonical_artifact.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"placeholder")

    # Rank 0: pretend artifacts are complete and return the canonical
    # path as the local resume decision.
    # Rank 1: pretend artifacts are missing and return None.
    if rank == 0:
        local_resume: Path | None = canonical
    else:
        local_resume = None
    final = _broadcast_resume_decision(
        local_resume, rank=rank, world_size=world_size,
    )
    return {"rank": rank, "final_resume": str(final) if final else None}


def test_resume_short_circuit_divergence_resolved_by_broadcast(
    tmp_path, gloo_multiprocess,
):
    """Bugs C-1/C-2 regression. With 2 ranks where rank 0 sees the
    artifacts complete and rank 1 sees them missing, the broadcast
    helper must coalesce to rank 0's decision so neither rank blocks.

    The test passes if BOTH ranks return the same canonical path
    (within the gloo_multiprocess timeout). Without the fix one rank
    would block on a downstream all_gather and the helper would time
    out instead of returning.
    """
    results = gloo_multiprocess(
        2, _resume_short_circuit_target, str(tmp_path),
    )
    rank0 = results[0]
    rank1 = results[1]
    assert rank0["final_resume"] is not None, (
        f"rank 0 must adopt its own resume decision; got {rank0!r}"
    )
    assert rank1["final_resume"] is not None, (
        f"rank 1 MUST adopt rank 0's resume decision via broadcast; "
        f"got {rank1!r}"
    )
    assert rank0["final_resume"] == rank1["final_resume"], (
        f"both ranks must converge on the SAME path; got "
        f"rank0={rank0['final_resume']!r} rank1={rank1['final_resume']!r}"
    )


def _resume_error_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
) -> dict[str, Any]:
    """Worker target: rank 0's provenance error must reach every rank."""
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    local_error = "stale completed extractor identity" if rank == 0 else None
    try:
        _broadcast_resume_decision(
            None,
            rank=rank,
            world_size=world_size,
            resume_error=local_error,
        )
    except AxBenchBaseError as exc:
        return {"rank": rank, "error": str(exc)}
    return {"rank": rank, "error": None}


def test_resume_error_is_broadcast_to_every_rank(
    tmp_path,
    gloo_multiprocess,
):
    """No worker may enter Stage 0c after rank 0 rejects provenance."""
    results = gloo_multiprocess(2, _resume_error_target, str(tmp_path))
    assert [result["error"] for result in results] == [
        "stale completed extractor identity",
        "stale completed extractor identity",
    ]


# ---------------------------------------------------------------------------
# Issue #939 regression: a rank-0-only profiler failure must be caught before
# the stage-outcome broadcast so every rank observes the same loud failure.
# ---------------------------------------------------------------------------


def _raise_injected_profiler_sync_failure() -> None:
    """Fail at the real profile-region synchronization seam."""
    raise RuntimeError("injected rank-0 profiler synchronization failure")


def _profiled_stage_collective_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    inject_profiler_failure: bool,
) -> dict[str, Any]:
    """Run one rank-0-only stage with profiling active only on rank 0."""
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    broadcast_calls: list[int] = []
    invocations: list[str] = []
    real_broadcast_outcome = cli._broadcast_rank0_stage_outcome

    def _counting_broadcast_outcome(*args: Any, **kwargs: Any) -> None:
        broadcast_calls.append(1)
        real_broadcast_outcome(*args, **kwargs)

    def _spy_run_single_stage(cfg: Any, stage: Any, state: Any) -> None:
        del cfg, state
        invocations.append(Stage.coerce(stage).value)

    setattr(cli, "_broadcast_rank0_stage_outcome", _counting_broadcast_outcome)
    setattr(cli, "check_stage_dependencies", lambda c, s: None)
    setattr(cli, "_run_single_stage", _spy_run_single_stage)
    if rank == 0 and inject_profiler_failure:
        setattr(
            profiler_mod,
            "_synchronize_cuda_if_available",
            _raise_injected_profiler_sync_failure,
        )

    cfg: Any = _MockCfg(
        stage="0a",
        run_negative_mode=False,
        run_tag="profiled-stage-collective-test",
        device="cpu",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=rank, ddp_world_size=world_size)
    scenario = "failure" if inject_profiler_failure else "success"
    profile_dir = Path(tmp_path_str) / f"profile-{scenario}-rank{rank}"
    profile_cm = (
        AxBenchProfiler(
            profile_dir,
            synchronize_cuda=inject_profiler_failure,
            gpu_utilization=False,
        )
        if rank == 0
        else nullcontext()
    )
    error_type: str | None = None
    error: str | None = None
    try:
        with profile_cm:
            cli._dispatch_stage(cfg, state)
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)
    finally:
        try:
            state.close()
        except Exception:
            pass

    summary_path = profile_dir / "summary.yaml"
    summary_text = (
        summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
    )
    return {
        "rank": rank,
        "error_type": error_type,
        "error": error,
        "broadcasts": len(broadcast_calls),
        "stages_invoked": invocations,
        "profile_exists": summary_path.exists(),
        "profile_summary": summary_text,
    }


def test_rank0_profiler_failure_reaches_stage_outcome_boundary(
    tmp_path,
    monkeypatch,
):
    """Non-socket oracle for the exact failure boundary fixed by issue #939."""
    seen_resume_errors: list[str | None] = []

    def _raise_broadcast_error(
        local_resume: Any,
        *,
        rank: int,
        world_size: int,
        resume_error: str | None = None,
    ) -> None:
        del local_resume, rank, world_size
        seen_resume_errors.append(resume_error)
        if resume_error is not None:
            raise AxBenchBaseError(resume_error)

    monkeypatch.setattr(
        profiler_mod,
        "_synchronize_cuda_if_available",
        _raise_injected_profiler_sync_failure,
    )
    monkeypatch.setattr(cli, "_broadcast_resume_decision", _raise_broadcast_error)
    monkeypatch.setattr(cli, "check_stage_dependencies", lambda c, s: None)
    monkeypatch.setattr(cli, "_run_single_stage", lambda c, s, state: None)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    cfg: Any = _MockCfg(
        stage="0a",
        run_negative_mode=False,
        run_tag="profile-failure-boundary-test",
        device="cpu",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=2)
    with AxBenchProfiler(
        tmp_path / "profile-failure-boundary",
        synchronize_cuda=True,
        gpu_utilization=False,
    ):
        with pytest.raises(
            AxBenchBaseError,
            match=(
                "Rank-0-only stage '0a' failed collectively: RuntimeError: "
                "injected rank-0 profiler synchronization failure"
            ),
        ):
            cli._dispatch_stage(cfg, state)

    assert seen_resume_errors == [
        "Rank-0-only stage '0a' failed collectively: RuntimeError: "
        "injected rank-0 profiler synchronization failure"
    ]


def test_rank0_profiler_failure_raises_on_every_gloo_rank(
    tmp_path,
    monkeypatch,
    gloo_multiprocess,
):
    """A rank-0 profiler failure must not strand peers in the broadcast."""
    monkeypatch.setattr(mp_helpers, "_RESULT_TIMEOUT_S", 10)
    results = gloo_multiprocess(
        2,
        _profiled_stage_collective_target,
        str(tmp_path),
        True,
    )
    expected_error = (
        "Rank-0-only stage '0a' failed collectively: RuntimeError: "
        "injected rank-0 profiler synchronization failure"
    )
    assert [result["error_type"] for result in results] == [
        "AxBenchBaseError",
        "AxBenchBaseError",
    ]
    assert [result["error"] for result in results] == [
        expected_error,
        expected_error,
    ]
    assert [result["broadcasts"] for result in results] == [1, 1]


def test_successful_rank0_profiler_preserves_one_collective_and_stage_regions(
    tmp_path,
    gloo_multiprocess,
):
    """Profiling success keeps one collective per rank and the stage regions."""
    results = gloo_multiprocess(
        2,
        _profiled_stage_collective_target,
        str(tmp_path),
        False,
    )
    assert [result["broadcasts"] for result in results] == [1, 1]
    assert results[0]["stages_invoked"] == ["0a"]
    assert results[1]["stages_invoked"] == []
    assert results[0]["profile_exists"] is True
    assert results[1]["profile_exists"] is False
    assert "name: stage.0a" in results[0]["profile_summary"]
    assert "name: stage.dependency_check" in results[0]["profile_summary"]


# ---------------------------------------------------------------------------
# Multi-stage rank-gate dispatch deadlock prevention.
#
# The dispatch loop at scripts/eval/axbench_base_eval.py:780-803 must call
# ``_maybe_barrier`` exactly once per stage on EVERY rank — once on the
# rank-0-skip path and once on the rank-aware-or-rank-0 happy path. A
# regression that drops one of those barriers would deadlock when stages
# are mixed (e.g., 0a non-rank-aware + 0c rank-aware), because the rank
# that took the missing branch would walk past the next stage's barrier
# while peers were still waiting on the current one.
#
# The existing ``test_dispatch_stage_rank_0_only_under_ddp`` only exercises
# a single stage; these tests dispatch a SEQUENCE of mixed stages and
# verify per-rank barrier counts match across ranks, surfacing the
# regression even at world_size=4 where one missing barrier on a single
# rank would block all four.
# ---------------------------------------------------------------------------


_MULTI_STAGE_SEQUENCE = (
    Stage.PREPARE_CONCEPTS,           # 0a — non-rank-aware (rank-0 only)
    Stage.PREPARE_NEGATIVE_PREFIXES,  # 0c — RANK-AWARE (every rank)
    Stage.SYNTH_POSITIVE_MODE,        # 1b — non-rank-aware
    Stage.JUDGE,                      # 5  — non-rank-aware
)


def _multi_stage_rank_gate_target(
    rank: int, world_size: int, tmp_path_str: str,
) -> dict[str, Any]:
    """Worker target for multi-stage rank-gate verification.

    Dispatches the mixed rank-aware + rank-0-only stage sequence and
    records:

    1. Every stage handler invocation (so we can verify rank 0 ran
       every stage and rank > 0 ran only the rank-aware one).
    2. Every ``_maybe_barrier`` call (so we can verify per-rank
       barrier counts are EQUAL across ranks — the deadlock-free
       invariant).

    Critical: the barrier spy WRAPS the real ``dist.barrier`` rather
    than replacing it. A pure no-op spy would pass even if the
    dispatcher dropped a barrier — but a real barrier-with-counter
    surfaces the regression as a real timeout because the rank that
    skipped the barrier walks ahead and the other ranks block.

    Stubs go through :func:`setattr` so ty doesn't flag the
    type-narrowing function-pointer reassignments — the test here
    intentionally swaps strongly-typed module functions for ``Any``-
    typed test doubles, and ``# type: ignore[assignment]`` is not
    sufficient to suppress these in ty.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    invocations: list[str] = []
    barrier_calls: list[int] = []
    broadcast_calls: list[int] = []

    def _spy_run_single_stage(cfg: Any, stage: Any, state: Any) -> None:
        del cfg, state
        invocations.append(str(Stage.coerce(stage).value))

    real_maybe_barrier = cli._maybe_barrier
    real_broadcast_outcome = cli._broadcast_rank0_stage_outcome

    def _counting_maybe_barrier(world_size_arg: int) -> None:
        # Wrap-and-count: still calls the real barrier helper so the
        # test exercises real distributed semantics, not just
        # bookkeeping. A regression that drops the rank-0 barrier
        # surfaces as a real timeout (caught by the
        # ``gloo_multiprocess`` 60s deadline).
        barrier_calls.append(1)
        real_maybe_barrier(world_size_arg)

    def _counting_broadcast_outcome(*args: Any, **kwargs: Any) -> None:
        # A rank-0-only stage under DDP synchronizes through this
        # broadcast INSTEAD of ``_maybe_barrier`` (the dispatcher's
        # ``rank0_collective`` branch ``continue``s past the barrier).
        # It is still a collective every rank must enter, so it counts
        # toward the same per-stage budget. Wrapped, not replaced, for
        # the same reason as the barrier spy.
        broadcast_calls.append(1)
        real_broadcast_outcome(*args, **kwargs)

    setattr(cli, "_run_single_stage", _spy_run_single_stage)
    setattr(cli, "_maybe_barrier", _counting_maybe_barrier)
    setattr(cli, "_broadcast_rank0_stage_outcome", _counting_broadcast_outcome)
    setattr(cli, "check_stage_dependencies", lambda c, s: None)
    # Pin the stage sequence — the dispatcher consumes
    # ``_resolve_stage_list``'s output, so stub it to return our
    # multi-stage list regardless of what cfg.stage says.
    setattr(
        cli, "_resolve_stage_list",
        lambda stage, run_neg: list(_MULTI_STAGE_SEQUENCE),
    )

    cfg: Any = _MockCfg(
        stage="all", run_negative_mode=True, run_tag="multi-stage-gate-test",
        device="cpu",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=rank, ddp_world_size=world_size)
    try:
        cli._dispatch_stage(cfg, state)
    finally:
        try:
            state.close()
        except Exception:
            pass

    return {
        "rank": rank,
        "stages_invoked": invocations,
        "barriers": len(barrier_calls),
        "broadcasts": len(broadcast_calls),
        # The deadlock-free invariant is one COLLECTIVE per stage per
        # rank; which primitive provides it depends on the branch the
        # dispatcher took. Counting only barriers made this assertion
        # mechanism-specific, and it went red when rank-0-only stages
        # moved to broadcast-based synchronization.
        "syncs": len(barrier_calls) + len(broadcast_calls),
    }


def test_dispatch_stage_multi_stage_no_deadlock_at_world_size_2(
    tmp_path, gloo_multiprocess,
):
    """Multi-stage dispatch with mixed rank-aware + rank-0-only stages
    must NOT deadlock. Every rank enters exactly one COLLECTIVE per
    stage, regardless of which branch it took.

    Which primitive provides it depends on the branch: rank-aware stages
    synchronize through ``_maybe_barrier``; rank-0-only stages under DDP
    ``continue`` past that barrier and synchronize through
    ``_broadcast_rank0_stage_outcome`` instead, which also propagates a
    rank-0 failure to every rank (a bare barrier cannot). Asserting on
    barriers alone made this test mechanism-specific and it went red
    when that substitution landed.

    A regression that drops one of those barriers would surface here as
    a 60s timeout in :func:`gloo_multiprocess`'s deadline.
    """
    results = gloo_multiprocess(
        2, _multi_stage_rank_gate_target, str(tmp_path),
    )
    rank0 = results[0]
    rank1 = results[1]

    # Rank 0 ran every stage; rank 1 ran only the rank-aware Stage 0c.
    assert rank0["stages_invoked"] == [s.value for s in _MULTI_STAGE_SEQUENCE], (
        f"rank 0 must invoke every stage in the sequence; "
        f"got {rank0['stages_invoked']!r}"
    )
    assert rank1["stages_invoked"] == [Stage.PREPARE_NEGATIVE_PREFIXES.value], (
        f"rank 1 must invoke ONLY the rank-aware stage 0c; "
        f"got {rank1['stages_invoked']!r}"
    )

    # Per-rank COLLECTIVE counts must be EQUAL across ranks (one
    # collective per stage on EACH rank — the deadlock-free invariant).
    # A rank-0-only stage under DDP synchronizes via the outcome
    # broadcast rather than a barrier, so both primitives count. What
    # must never happen is a stage where one rank enters a collective
    # and another does not.
    expected_sync_count = len(_MULTI_STAGE_SEQUENCE)
    assert rank0["syncs"] == expected_sync_count, (
        f"rank 0 should enter exactly {expected_sync_count} collectives "
        f"(one per stage); got {rank0['syncs']} "
        f"(barriers={rank0['barriers']}, broadcasts={rank0['broadcasts']})"
    )
    assert rank1["syncs"] == expected_sync_count, (
        f"rank 1 should enter exactly {expected_sync_count} collectives "
        f"(one per stage); got {rank1['syncs']} "
        f"(barriers={rank1['barriers']}, broadcasts={rank1['broadcasts']})"
    )
    # Ranks must also agree on the AGGREGATE split, not just the total:
    # equal totals with different barrier/broadcast composition deadlock.
    #
    # Precision note (cross-review, LOW): this compares aggregate counts,
    # NOT per-stage sequences. Equal splits in a different per-stage
    # ORDER -- rank 0 (broadcast, barrier, ...) vs rank 1 (barrier,
    # broadcast, ...) -- would satisfy it. That is not a false-green
    # path, because the spies wrap the REAL collectives, so a mismatched
    # pairing hangs or errors before execution reaches this assertion.
    # Recording (stage_name, primitive) sequences would make the
    # diagnostic sharper; the hang oracle does not depend on it.
    assert (rank0["barriers"], rank0["broadcasts"]) == (
        rank1["barriers"], rank1["broadcasts"]
    ), (
        "ranks disagree on collective composition: "
        f"rank0 barriers={rank0['barriers']} broadcasts={rank0['broadcasts']}; "
        f"rank1 barriers={rank1['barriers']} broadcasts={rank1['broadcasts']}"
    )


def test_dispatch_stage_multi_stage_no_deadlock_at_world_size_4(
    tmp_path, gloo_multiprocess,
):
    """Same invariant at world_size=4: every rank enters every collective.

    Surfaces fan-out regressions where (e.g.) the dispatcher dropped a
    barrier on ranks > 1 only — at world_size=2 that test would still
    pass; at world_size=4 it deadlocks.
    """
    results = gloo_multiprocess(
        4, _multi_stage_rank_gate_target, str(tmp_path),
    )
    expected_sync_count = len(_MULTI_STAGE_SEQUENCE)
    for rank, result in enumerate(results):
        assert result["syncs"] == expected_sync_count, (
            f"rank {rank}/4 should enter exactly {expected_sync_count} "
            f"collectives; got {result['syncs']} "
            f"(barriers={result['barriers']}, "
            f"broadcasts={result['broadcasts']})"
        )
    # Ranks must agree on the aggregate barrier/broadcast split, not just
    # the total — a fan-out regression that swaps one primitive for
    # another on ranks > 1 keeps the totals equal and still hangs. As at
    # world_size=2, this compares aggregate counts rather than per-stage
    # sequences; the real wrapped collectives are the hang oracle.
    splits = {(r["barriers"], r["broadcasts"]) for r in results}
    assert len(splits) == 1, (
        f"ranks disagree on collective composition across world_size=4: {splits}"
    )

    # Rank 0 ran every stage; ranks 1..3 ran only the rank-aware one.
    assert results[0]["stages_invoked"] == [
        s.value for s in _MULTI_STAGE_SEQUENCE
    ]
    for rank in (1, 2, 3):
        assert results[rank]["stages_invoked"] == [
            Stage.PREPARE_NEGATIVE_PREFIXES.value
        ], (
            f"rank {rank}/4 must invoke ONLY the rank-aware stage 0c; "
            f"got {results[rank]['stages_invoked']!r}"
        )
