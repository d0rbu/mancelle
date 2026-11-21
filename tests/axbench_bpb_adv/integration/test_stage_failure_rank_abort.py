"""Real multi-process tests for #942: a stage failure must abort every rank.

These run an actual 2-rank gloo process group. That matters here specifically:
the defect is a *cross-rank* one, and every same-process test of it is a
simulation of the collective rather than the collective itself. The behaviour
being fixed only exists when two real processes disagree about whether to enter
a collective.

What the fix changes, precisely (measured, not assumed):

* Before, a rank that failed simply propagated its exception. The surviving
  ranks were already committed to ``_maybe_barrier``. They blocked until the
  failing process finished unwinding and its gloo transport dropped, then
  surfaced ``Connection closed by peer`` -- an error naming neither the stage
  nor the real cause. If the failing rank did not exit promptly, they blocked
  indefinitely (reproduced: a live-but-stalled rank 0 left rank 1 blocked past
  a 30s timeout, having never passed the barrier).
* After, every rank captures its own exception, all ranks meet at one exchange,
  and all raise together: the failing rank re-raises its original exception with
  its traceback, the others raise a summary naming the stage.

``run_gloo_multiprocess`` imposes its own timeout, so a regression that
reintroduces the block fails these tests loudly instead of hanging the suite.
"""

from __future__ import annotations

import io
from typing import Any, cast

import pytest
import torch.distributed as dist
from loguru import logger

from scripts.eval import axbench_bpb_adv_eval as cli
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, Stage
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from tests.axbench_base import _multiprocess_helpers as mp_helpers


def _dispatch_cfg() -> AxBenchBpbAdvConfig:
    """A cfg complete enough that ``_run_resolved`` runs to completion.

    Deliberately not a bare namespace: a fake that crashes on a missing attribute
    makes a test pass for the wrong reason, so these assertions are reached
    on their own merits rather than via an AttributeError.
    """
    return AxBenchBpbAdvConfig(
        stage=Stage.SCORE.value,
        run_tag="run",
        output_root="/tmp/does-not-need-to-exist",
        source_run_dir="/tmp/does-not-need-to-exist/source",
    )


def _worker_one_rank_fails(rank: int, world_size: int, failing_rank: int) -> str:
    """Every rank calls the exchange; ``failing_rank`` arrives carrying an error."""
    error: BaseException | None = None
    if rank == failing_rank:
        error = RuntimeError(f"stage blew up on rank {rank}")
    try:
        cli._abort_all_ranks_on_stage_failure(
            error,
            stage=Stage.SCORE,
            world_size=world_size,
        )
    except RuntimeError as exc:
        # Only the rank that actually failed re-raises the original object.
        return f"original:{exc}"
    except AxBenchBpbAdvError as exc:
        return f"summary:{exc}"
    return "no-raise"


def _worker_no_failure(rank: int, world_size: int) -> str:
    cli._abort_all_ranks_on_stage_failure(
        None,
        stage=Stage.SCORE,
        world_size=world_size,
    )
    return "completed"


def _worker_uninitialized_dist(rank: int, world_size: int) -> str:
    """Claim a multi-rank world while torch.distributed is torn down."""
    dist.destroy_process_group()
    try:
        cli._abort_all_ranks_on_stage_failure(None, stage=Stage.SCORE, world_size=2)
    except AxBenchBpbAdvError as exc:
        return f"guarded:{exc}"
    return "no-raise"


def _worker_rank_zero_marker_guard_logs(
    rank: int,
    world_size: int,
) -> tuple[str, str]:
    """Exercise a rank-zero guard and return its result plus captured log."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="ERROR", format="{message}")
    try:
        if rank == 0:
            def _raise_marker(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("rank-zero marker sentinel")

            cli.ensure_feature_contract_run_marker = cast(Any, _raise_marker)

        try:
            cli._install_feature_contract_marker_ddp(
                _dispatch_cfg(),
                cast(ClusterFeatureContract, object()),
                rank=rank,
                world_size=world_size,
            )
        except AxBenchBpbAdvError as exc:
            result = f"{type(exc).__name__}: {exc}"
        else:
            result = "no-raise"
        return result, sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_a_stage_failure_on_any_rank_aborts_every_rank(failing_rank: int) -> None:
    """Both directions: whichever rank fails, neither rank is left blocked.

    ``failing_rank=1`` is the case a broadcast-from-rank-0 exchange would miss
    entirely, and ``PREPARE_CONTEXT_CACHE`` runs on every rank -- so a non-zero
    rank really can be the one that fails.
    """
    results = mp_helpers.run_gloo_multiprocess(
        2, _worker_one_rank_fails, failing_rank
    )

    assert results[failing_rank].startswith("original:"), (
        f"the failing rank must re-raise its own exception, got {results[failing_rank]!r}"
    )
    assert f"blew up on rank {failing_rank}" in results[failing_rank]

    survivor = 1 - failing_rank
    assert results[survivor].startswith("summary:"), (
        f"the surviving rank must abort with a summary, got {results[survivor]!r}"
    )
    assert "all ranks aborting" in results[survivor]
    assert Stage.SCORE.value in results[survivor], (
        "the summary must name the stage so an operator knows where to look"
    )
    # The pre-fix failure mode surfaced a gloo transport error on the survivor.
    assert "Connection closed by peer" not in results[survivor]


def test_no_failure_lets_every_rank_continue() -> None:
    """The exchange must not turn a clean stage into an abort."""
    assert mp_helpers.run_gloo_multiprocess(2, _worker_no_failure) == [
        "completed",
        "completed",
    ]


def test_multi_rank_world_without_initialized_dist_fails_loud() -> None:
    """A multi-rank world with no process group is a config error, not a silent skip."""
    results = mp_helpers.run_gloo_multiprocess(1, _worker_uninitialized_dist)

    assert results[0].startswith("guarded:")
    assert "requires initialized" in results[0]


def test_rank_zero_guard_logs_traceback_and_completes_gloo_collective() -> None:
    results = mp_helpers.run_gloo_multiprocess(
        2,
        _worker_rank_zero_marker_guard_logs,
    )

    rank_zero_result, rank_zero_log = results[0]
    rank_one_result, rank_one_log = results[1]
    assert "feature-contract run-marker validation failed" in rank_zero_result
    assert "RuntimeError: rank-zero marker sentinel" in rank_zero_result
    assert "feature-contract run-marker validation failed" in rank_one_result
    assert "RuntimeError: rank-zero marker sentinel" in rank_one_result
    assert (
        "AxBench-BPB-Adv rank-0 feature-contract marker installation failed"
        in rank_zero_log
    )
    assert "RuntimeError: rank-zero marker sentinel" in rank_zero_log
    assert "_raise_marker" in rank_zero_log
    assert rank_one_log == ""


def test_single_rank_reraises_without_touching_a_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``world_size == 1`` must not call a collective at all.

    Single-process runs are the common case and often have no process group;
    reaching for ``all_reduce`` there would break every non-DDP invocation.
    """

    def _explode(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("single-rank path must not perform a collective")

    monkeypatch.setattr(cli.dist, "all_reduce", _explode)
    original = RuntimeError("single rank failure")

    with pytest.raises(RuntimeError) as excinfo:
        cli._abort_all_ranks_on_stage_failure(
            original, stage=Stage.SCORE, world_size=1
        )

    assert excinfo.value is original, "the original exception object must survive"


def test_single_rank_success_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("single-rank path must not perform a collective")

    monkeypatch.setattr(cli.dist, "all_reduce", _explode)
    cli._abort_all_ranks_on_stage_failure(None, stage=Stage.SCORE, world_size=1)


# ─────────────────────────────────────────────────────────────────────────────
# Call-site coverage: the tests above prove the exchange behaves correctly, but
# say nothing about the dispatcher using it. Stripping the call from the stage
# loop leaves every one of them green -- so pin the call site itself.
# ─────────────────────────────────────────────────────────────────────────────


def test_stage_loop_routes_every_stage_through_the_abort_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_resolved`` must hand each stage's outcome to the exchange.

    A helper that is correct but never called is worth nothing here: the whole
    defect is that the dispatcher raised directly instead of exchanging first.
    """
    seen: list[tuple[object, object, int]] = []

    monkeypatch.setattr(cli, "_run_stage", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli,
        "_abort_all_ranks_on_stage_failure",
        lambda error, *, stage, world_size: seen.append((error, stage, world_size)),
    )
    monkeypatch.setattr(cli, "_maybe_barrier", lambda _ws: None)

    cfg = _dispatch_cfg()
    cli._run_resolved(cfg, ddp_rank=0, ddp_world_size=2, cluster_contract=None)

    assert seen == [(None, Stage.SCORE, 2)], (
        "each stage must pass its outcome, the stage, and the world size to the exchange"
    )


def test_stage_loop_hands_a_stage_exception_to_the_exchange_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing stage must reach the exchange, not propagate straight out.

    This is the exact regression: raising directly is what leaves peer ranks
    committed to a collective the failing rank never joins.
    """
    boom = RuntimeError("stage exploded")
    seen: list[tuple[object, object, int]] = []

    def _raise(*_a: Any, **_k: Any) -> None:
        raise boom

    monkeypatch.setattr(cli, "_run_stage", _raise)
    monkeypatch.setattr(
        cli,
        "_abort_all_ranks_on_stage_failure",
        lambda error, *, stage, world_size: seen.append((error, stage, world_size)),
    )
    monkeypatch.setattr(cli, "_maybe_barrier", lambda _ws: None)

    cfg = _dispatch_cfg()
    cli._run_resolved(cfg, ddp_rank=0, ddp_world_size=2, cluster_contract=None)

    assert seen and seen[0][0] is boom, (
        f"the stage's exception object must reach the exchange, got {seen!r}"
    )
    assert seen[0][1] is Stage.SCORE
