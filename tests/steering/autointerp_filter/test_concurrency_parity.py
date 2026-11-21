from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

import steering.eval.autointerp_filter.runner as runner_module
from steering.eval.autointerp_filter.cache import IntruderTrialCache
from steering.eval.autointerp_filter.config import IntruderJudgeConfig
from steering.eval.autointerp_filter.intruder import IntruderDetectionJudge, judge_cache_key
from steering.eval.autointerp_filter.runner import (
    AutoInterpFilterRunner,
    _TrialJudgeOutcome,
    _TrialJudgeWorkItem,
)
from steering.eval.autointerp_filter.token_examples import TokenExampleSampler
from steering.eval.autointerp_filter.types import (
    AutoInterpCacheError,
    AutoInterpCostCapExceeded,
    IntruderResponseParseError,
    IntruderTrial,
    IntruderTrialResult,
    ScoringTarget,
)
from steering.refined_types import PositiveInt
from tests.steering.autointerp_filter.test_autointerp_filter import (
    _artifact_snapshot,
    _cfg,
    _example_set,
    _FakeTokenProvider,
    _MarkerRenderer,
    _RecordingIntruderJudge,
    _target,
    _write_feature_shard,
)

# Artifacts whose bytes carry no run-specific data (no timestamps, no output_dir
# paths), so serial and concurrent runs in distinct output dirs must produce
# byte-for-byte identical files.
_BYTE_IDENTICAL_ARTIFACTS = (
    "target_scores_path",
    "ranked_yaml_path",
    "accepted_features_path",
    "trial_audit_path",
)

# Per-trial-index judge behavior shared by every run below so results depend only
# on the trial identity, never on wall-clock completion order.  The delays force
# out-of-order completions on the concurrent path; the "wrong" set makes some
# targets partially correct so accept/reject and score fields are exercised.
_JUDGE_DELAYS = {0: 0.03, 1: 0.0, 2: 0.02, 3: 0.0}
_JUDGE_WRONG_TRIAL_INDICES = {1}


class _SyntheticRunnerCrash(RuntimeError):
    """Deterministic main-thread failure injected after a durable cache write."""


def _correct_result(trial: IntruderTrial) -> IntruderTrialResult:
    return IntruderTrialResult(
        trial_key=trial.trial_key,
        prediction=trial.intruder_index,
        correct_index=trial.intruder_index,
        correct=True,
        response=f"[RESPONSE]: {trial.intruder_index}",
        cached=False,
    )


class _BlockingJudge:
    """Complete one trial while another keeps the global wave open."""

    def __init__(self) -> None:
        self.fast_returned = threading.Event()
        self.release_slow = threading.Event()

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _MarkerRenderer,
    ) -> IntruderTrialResult:
        del renderer
        if trial.trial_index == 0:
            if not self.release_slow.wait(timeout=5.0):
                raise TimeoutError("test did not release the blocked trial")
        else:
            self.fast_returned.set()
        return _correct_result(trial)


class _ThreadProbeJudge:
    """Record the worker thread that evaluates each uncached trial."""

    def __init__(self, *, expected_parallel: int) -> None:
        self.thread_ids: list[int] = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(expected_parallel)

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _MarkerRenderer,
    ) -> IntruderTrialResult:
        del renderer
        with self._lock:
            self.thread_ids.append(threading.get_ident())
        self._barrier.wait(timeout=5.0)
        return _correct_result(trial)


class _SoftErrorResultJudge:
    """Return legacy soft-error results to prove they remain non-cacheable."""

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _MarkerRenderer,
    ) -> IntruderTrialResult:
        del renderer
        return IntruderTrialResult(
            trial_key=trial.trial_key,
            prediction=None,
            correct_index=trial.intruder_index,
            correct=False,
            response="not parseable",
            cached=False,
            error_message="synthetic soft error",
        )


def test_work_item_carries_global_and_local_coordinates() -> None:
    item = _TrialJudgeWorkItem(
        global_index=7,
        target_index=2,
        trial_index=1,
        trial=cast(IntruderTrial, None),
    )
    assert (item.global_index, item.target_index, item.trial_index) == (7, 2, 1)
    outcome = _TrialJudgeOutcome(global_index=7)
    assert outcome.global_index == 7 and outcome.result is None and outcome.error is None


def test_default_max_concurrent_is_concurrent() -> None:
    """Runs are concurrent by default: Task 2 proved serial and concurrent
    judge dispatch produce byte-identical artifacts, so there is no reason
    for the out-of-the-box config to leave judge throughput single-threaded."""
    assert IntruderJudgeConfig().max_concurrent == 8


def _score_config(
    tmp_path: Path,
    feature_dir: Path,
    targets: list[ScoringTarget],
    *,
    max_concurrent: int,
    label: str,
    **overrides: Any,
) -> Any:
    return _cfg(
        tmp_path / label,
        feature_dir,
        feature_ids=[target.feature_ids[0] for target in targets],
        n_trials_per_target=4,
        n_positive_examples=2,
        n_quantiles=2,
        score_threshold=0.5,
        audit_sample_limit=50,
        judge=IntruderJudgeConfig(
            model="test-judge",
            max_concurrent=PositiveInt.parse(max_concurrent),
        ),
        **overrides,
    )


def _run_scoring(
    tmp_path: Path,
    feature_dir: Path,
    targets: list[ScoringTarget],
    provider: _FakeTokenProvider,
    *,
    max_concurrent: int,
    label: str,
) -> Any:
    cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=max_concurrent,
        label=label,
    )
    judge = _RecordingIntruderJudge(
        delays=_JUDGE_DELAYS,
        wrong_trial_indices=_JUDGE_WRONG_TRIAL_INDICES,
    )
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    return cfg


def _cached_predictions(
    cfg: Any,
    targets: list[ScoringTarget],
    provider: _FakeTokenProvider,
) -> list[tuple[str, str, int | None]]:
    provider.prepare(targets)
    sampler = TokenExampleSampler(cfg, provider=provider)
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    rows: list[tuple[str, str, int | None]] = []
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        for target in targets:
            for trial in sampler.build_trials(target):
                cached = cache.get(
                    trial_key=trial.trial_key,
                    target_key=target.target_key,
                    judge_key=cache_judge_key,
                )
                if cached is not None:
                    rows.append((target.target_key, trial.trial_key, cached.prediction))
    return rows


def _trial_cache_rows(path: str | Path) -> list[tuple]:
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT trial_key, target_key, judge_key, prediction, correct_index, "
            "correct, response, error_message FROM trial_results "
            "ORDER BY trial_key, target_key, judge_key"
        ).fetchall()


def _artifact_snapshot_without_cache_provenance(cfg: Any) -> dict[str, object]:
    snapshot = _artifact_snapshot(cfg)
    audit_rows = cast(list[dict[str, object]], snapshot["trial_audit_jsonl"])
    for row in audit_rows:
        row["cached"] = False
    return snapshot


def test_serial_and_concurrent_artifacts_are_byte_identical(tmp_path: Path) -> None:
    """The whole point of the refactor: serial (max_concurrent=1) and concurrent
    (max_concurrent=8) runs over the same targets must produce byte-identical
    on-disk artifacts.  Multiple targets exercise the global cross-concept pass;
    per-trial delays force out-of-order completion on the concurrent path."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,)), _target((2,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})

    serial_cfg = _run_scoring(
        tmp_path, feature_dir, targets, provider, max_concurrent=1, label="serial"
    )
    concurrent_cfg = _run_scoring(
        tmp_path, feature_dir, targets, provider, max_concurrent=8, label="concurrent"
    )

    for attr in _BYTE_IDENTICAL_ARTIFACTS:
        serial_bytes = Path(getattr(serial_cfg, attr)).read_bytes()
        concurrent_bytes = Path(getattr(concurrent_cfg, attr)).read_bytes()
        assert serial_bytes == concurrent_bytes, f"{attr} diverged between serial and concurrent"

    # Also compare the normalized snapshot (covers report.json + manual-filter
    # metadata with generated_at removed and output_dir path-normalized).
    assert _artifact_snapshot(serial_cfg) == _artifact_snapshot(concurrent_cfg)


def test_serial_and_concurrent_identical_under_mid_run_budget_cap(
    tmp_path: Path,
) -> None:
    """Byte-identity must survive a judge-call budget that caps mid-run: because
    admission is deterministic and single-threaded, the *same* trials are admitted
    before the cap regardless of max_concurrent, so both runs raise the same error
    after caching the same admitted prefix (target0's 4 trials + target1 trial 0)."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,)), _target((2,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    serial_cfg = _score_config(
        tmp_path, feature_dir, targets, max_concurrent=1, label="serial_cap", max_judge_calls=5
    )
    concurrent_cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=8,
        label="concurrent_cap",
        max_judge_calls=5,
    )
    for cfg in (serial_cfg, concurrent_cfg):
        judge = _RecordingIntruderJudge(
            delays=_JUDGE_DELAYS,
            wrong_trial_indices=_JUDGE_WRONG_TRIAL_INDICES,
        )
        with pytest.raises(AutoInterpCostCapExceeded):
            AutoInterpFilterRunner(
                cfg,
                judge=cast(IntruderDetectionJudge, judge),
                token_example_provider=provider,
                renderer=_MarkerRenderer(),
            ).run()

    serial_cached = _cached_predictions(serial_cfg, targets, provider)
    concurrent_cached = _cached_predictions(concurrent_cfg, targets, provider)
    assert serial_cached == concurrent_cached
    assert len(serial_cached) == 5
    assert not Path(serial_cfg.target_scores_path).exists()
    assert not Path(concurrent_cfg.target_scores_path).exists()


def test_resolve_all_trials_dispatches_cache_hits_and_judged_globally(
    tmp_path: Path,
) -> None:
    """Directly exercise the Task-2 global resolution pass: cache hits are filled
    in place, misses are judged, and every slot of every sufficient target is
    resolved in canonical (target, trial) order."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    cfg = _score_config(tmp_path, feature_dir, targets, max_concurrent=4, label="resolve")
    provider.prepare(targets)
    sampler = TokenExampleSampler(cfg, provider=provider)
    trials_by_target = [sampler.build_trials(target) for target in targets]
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())

    judge = _RecordingIntruderJudge(wrong_trial_indices=_JUDGE_WRONG_TRIAL_INDICES)
    runner = AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    )

    total_trials = sum(len(trials) for trials in trials_by_target)
    warm_trial = trials_by_target[0][0]
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        cache.set(
            target_key=targets[0].target_key,
            judge_key=cache_judge_key,
            result=IntruderTrialResult(
                trial_key=warm_trial.trial_key,
                prediction=warm_trial.intruder_index,
                correct_index=warm_trial.intruder_index,
                correct=True,
                response=f"[RESPONSE]: {warm_trial.intruder_index}",
                cached=False,
            ),
        )
        resolved = runner._resolve_all_trials(
            targets,
            trials_by_target,
            cache,
            cache_judge_key=cache_judge_key,
        )

    assert [len(row) for row in resolved] == [len(trials) for trials in trials_by_target]
    assert all(result is not None for row in resolved for result in row)
    # The pre-warmed slot is served from cache; every other slot was judged once.
    assert resolved[0][0] is not None and resolved[0][0].cached is True
    judged = [
        result for row in resolved for result in row if result is not None and not result.cached
    ]
    assert len(judged) == total_trials - 1
    assert len(judge.calls) == total_trials - 1


def test_completed_trial_is_durable_while_later_trial_is_still_running(
    tmp_path: Path,
) -> None:
    """A committed cache hit must be externally visible before the wave returns."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    targets = [target]
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=2,
        label="mid_wave_visibility",
    )
    provider.prepare(targets)
    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)
    assert [trial.trial_index for trial in trials] == [0, 1, 2, 3]
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    judge = _BlockingJudge()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            AutoInterpFilterRunner(
                cfg,
                judge=cast(IntruderDetectionJudge, judge),
                token_example_provider=provider,
                renderer=_MarkerRenderer(),
            ).run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert judge.fast_returned.wait(timeout=5.0), "fast judge trial did not return"
        deadline = time.monotonic() + 5.0
        cached = None
        while time.monotonic() < deadline:
            with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
                cached = cache.get(
                    trial_key=trials[1].trial_key,
                    target_key=target.target_key,
                    judge_key=cache_judge_key,
                )
            if cached is not None:
                break
            time.sleep(0.01)
        assert cached == IntruderTrialResult(
            trial_key=trials[1].trial_key,
            prediction=trials[1].intruder_index,
            correct_index=trials[1].intruder_index,
            correct=True,
            response=f"[RESPONSE]: {trials[1].intruder_index}",
            cached=True,
        )
        assert thread.is_alive(), "runner returned before the blocked trial was released"
        assert not Path(cfg.target_scores_path).exists()
    finally:
        judge.release_slow.set()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert errors == []
    assert len(_trial_cache_rows(cfg.judge_cache_path)) == 4


@pytest.mark.parametrize("max_concurrent", [1, 4])
def test_crash_after_first_observed_completion_preserves_row_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_concurrent: int,
) -> None:
    """A main-thread failure after progress handling cannot erase the prior commit."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    targets = [target]
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=max_concurrent,
        label=f"crash_mc{max_concurrent}",
    )

    def crash_after_commit(completed: int, total: int) -> None:
        assert completed == 1
        assert total == 4
        raise _SyntheticRunnerCrash("synthetic crash after first committed completion")

    monkeypatch.setattr(runner_module, "_log_trial_wave_progress", crash_after_commit)
    with pytest.raises(_SyntheticRunnerCrash, match="first committed completion"):
        AutoInterpFilterRunner(
            cfg,
            judge=cast(IntruderDetectionJudge, _RecordingIntruderJudge()),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()

    rows_after_crash = _trial_cache_rows(cfg.judge_cache_path)
    assert len(rows_after_crash) == 1
    assert rows_after_crash[0][3] == rows_after_crash[0][4]
    assert rows_after_crash[0][5] == 1
    assert rows_after_crash[0][6] == f"[RESPONSE]: {rows_after_crash[0][3]}"
    assert rows_after_crash[0][7] is None
    with sqlite3.connect(cfg.judge_cache_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert not Path(cfg.target_scores_path).exists()
    assert not Path(cfg.runner_manifest_path).exists()

    monkeypatch.setattr(runner_module, "_log_trial_wave_progress", lambda _done, _total: None)
    resumed_judge = _RecordingIntruderJudge()
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, resumed_judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    assert len(resumed_judge.calls) == 3
    assert len(_trial_cache_rows(cfg.judge_cache_path)) == 4

    control_cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=max_concurrent,
        label=f"control_mc{max_concurrent}",
    )
    AutoInterpFilterRunner(
        control_cfg,
        judge=cast(IntruderDetectionJudge, _RecordingIntruderJudge()),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    assert _artifact_snapshot_without_cache_provenance(
        cfg
    ) == _artifact_snapshot_without_cache_provenance(control_cfg)
    assert _trial_cache_rows(cfg.judge_cache_path) == _trial_cache_rows(
        control_cfg.judge_cache_path
    )


def test_concurrent_judges_write_cache_only_from_runner_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=8,
        label="thread_ownership",
    )
    judge = _ThreadProbeJudge(expected_parallel=8)
    runner_thread_id = threading.get_ident()
    cache_set_thread_ids: list[int] = []
    original_set = IntruderTrialCache.set

    def recording_set(
        self: IntruderTrialCache,
        *,
        target_key: str,
        judge_key: str,
        result: IntruderTrialResult,
    ) -> None:
        cache_set_thread_ids.append(threading.get_ident())
        original_set(
            self,
            target_key=target_key,
            judge_key=judge_key,
            result=result,
        )

    monkeypatch.setattr(IntruderTrialCache, "set", recording_set)
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert len(cache_set_thread_ids) == 8
    assert set(cache_set_thread_ids) == {runner_thread_id}
    assert len(judge.thread_ids) == 8
    assert runner_thread_id not in judge.thread_ids
    assert len(set(judge.thread_ids)) > 1


def test_completion_callback_does_not_cache_soft_error_results(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    targets = [target]
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=4,
        label="soft_errors",
    )

    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, _SoftErrorResultJudge()),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert _trial_cache_rows(cfg.judge_cache_path) == []
    assert Path(cfg.target_scores_path).exists()


class _FirstErrorProbeJudge:
    """Raises a distinguishable ``IntruderResponseParseError`` for chosen
    (target_key, trial_index) pairs, sleeping the given number of seconds
    before raising.  Every other trial succeeds immediately.

    Unlike ``_RecordingIntruderJudge``, failures are keyed by full
    (target, trial) identity rather than a bare local ``trial_index`` alone,
    so a test can pin an exact *cross-target* global_index instead of relying
    on a single-target coincidence where trial_index == global_index.
    """

    def __init__(self, *, error_points: dict[tuple[str, int], float]) -> None:
        self._error_points = error_points

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _MarkerRenderer,
    ) -> IntruderTrialResult:
        key = (trial.target.target_key, trial.trial_index)
        delay = self._error_points.get(key)
        if delay is not None:
            if delay > 0.0:
                time.sleep(delay)
            raise IntruderResponseParseError(
                f"synthetic first-error-precedence failure: target_key={key[0]!r} "
                f"trial_index={key[1]}"
            )
        return IntruderTrialResult(
            trial_key=trial.trial_key,
            prediction=trial.intruder_index,
            correct_index=trial.intruder_index,
            correct=True,
            response=f"[RESPONSE]: {trial.intruder_index}",
            cached=False,
        )


def test_concurrent_run_raises_lowest_global_index_error_not_first_to_finish(
    tmp_path: Path,
) -> None:
    """Pins the guarantee that `_evaluate_trial_wave`'s trailing
    ``outcomes.sort(key=lambda outcome: outcome.global_index)`` (runner.py)
    exists for: when multiple trials raise, `_resolve_all_trials` must
    re-raise the LOWEST-global_index trial's error -- exactly what a serial
    (max_concurrent=1) run raises -- regardless of which erroring trial
    actually finishes first in wall-clock on the concurrent path.

    The other tests in this module scatter results by index (via
    zip/enumerate over the resolved rows), so they are insensitive to the
    *order* outcomes come back in and would pass even if the trailing sort
    were deleted.  This test instead asserts on the identity of the raised
    exception, which is order-sensitive.

    Two targets of 4 trials each, with a fresh judge cache, produce exactly 8
    work items with global_index 0..7 in canonical (target, trial) order:
    target0 -> global_index 0..3, target1 -> global_index 4..7.  The
    LOWER-global_index failure (target0, trial_index=1 -> global_index=1)
    sleeps 0.2s before raising; the HIGHER-global_index failure (target1,
    trial_index=2 -> global_index=6) raises immediately.  On the concurrent
    path (max_concurrent=8), ``as_completed`` therefore yields the
    global_index=6 failure long before the global_index=1 failure, so without
    the trailing sort `_resolve_all_trials` would raise the global_index=6
    error instead of the global_index=1 error a serial run raises.
    """
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    error_points: dict[tuple[str, int], float] = {
        (str(targets[0].target_key), 1): 0.2,
        (str(targets[1].target_key), 2): 0.0,
    }

    def _run_and_capture_error(max_concurrent: int, label: str) -> IntruderResponseParseError:
        cfg = _score_config(
            tmp_path, feature_dir, targets, max_concurrent=max_concurrent, label=label
        )
        judge = _FirstErrorProbeJudge(error_points=error_points)
        with pytest.raises(IntruderResponseParseError) as excinfo:
            AutoInterpFilterRunner(
                cfg,
                judge=cast(IntruderDetectionJudge, judge),
                token_example_provider=provider,
                renderer=_MarkerRenderer(),
            ).run()
        return excinfo.value

    serial_error = _run_and_capture_error(max_concurrent=1, label="serial_first_error")
    concurrent_error = _run_and_capture_error(max_concurrent=8, label="concurrent_first_error")

    lower_index_message = f"target_key={targets[0].target_key!r} trial_index=1"
    higher_index_message = f"target_key={targets[1].target_key!r} trial_index=2"

    # A serial run has no completion-order ambiguity: it always raises the
    # lowest-global_index error. Pin that as ground truth first.
    assert lower_index_message in str(serial_error)

    # The concurrent run must match the serial run byte-for-byte, even though
    # the higher-global_index trial finished first in wall-clock.
    assert str(concurrent_error) == str(serial_error)
    assert lower_index_message in str(concurrent_error)
    assert higher_index_message not in str(concurrent_error)


@pytest.mark.parametrize("lower_failure_kind", ["judge", "cache"])
def test_judge_and_cache_write_failures_share_canonical_error_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lower_failure_kind: str,
) -> None:
    """Completion order cannot decide whether a judge or cache error wins."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    reference_cfg = _score_config(
        tmp_path,
        feature_dir,
        targets,
        max_concurrent=8,
        label=f"{lower_failure_kind}_reference",
    )
    provider.prepare(targets)
    trials_by_target = [
        TokenExampleSampler(reference_cfg, provider=provider).build_trials(target)
        for target in targets
    ]
    lower_point = (targets[0].target_key, 1)
    higher_point = (targets[1].target_key, 2)
    lower_trial_key = trials_by_target[0][1].trial_key
    higher_trial_key = trials_by_target[1][2].trial_key
    if lower_failure_kind == "judge":
        judge_error_points: dict[tuple[str, int], float] = {
            (str(lower_point[0]), lower_point[1]): 0.2
        }
        cache_failure_key = higher_trial_key
        expected_error = f"target_key={targets[0].target_key!r} trial_index=1"
    else:
        judge_error_points = {(str(higher_point[0]), higher_point[1]): 0.0}
        cache_failure_key = lower_trial_key
        expected_error = f"synthetic cache failure for {lower_trial_key}"

    original_set = IntruderTrialCache.set

    def failing_set(
        self: IntruderTrialCache,
        *,
        target_key: str,
        judge_key: str,
        result: IntruderTrialResult,
    ) -> None:
        if result.trial_key == cache_failure_key:
            raise AutoInterpCacheError(f"synthetic cache failure for {result.trial_key}")
        original_set(
            self,
            target_key=target_key,
            judge_key=judge_key,
            result=result,
        )

    monkeypatch.setattr(IntruderTrialCache, "set", failing_set)
    errors: list[BaseException] = []
    cache_rows: list[list[tuple]] = []
    for max_concurrent, label in ((1, "serial"), (8, "concurrent")):
        cfg = _score_config(
            tmp_path,
            feature_dir,
            targets,
            max_concurrent=max_concurrent,
            label=f"{lower_failure_kind}_{label}",
        )
        judge = _FirstErrorProbeJudge(error_points=judge_error_points)
        with pytest.raises((IntruderResponseParseError, AutoInterpCacheError)) as excinfo:
            AutoInterpFilterRunner(
                cfg,
                judge=cast(IntruderDetectionJudge, judge),
                token_example_provider=provider,
                renderer=_MarkerRenderer(),
            ).run()
        errors.append(excinfo.value)
        cache_rows.append(_trial_cache_rows(cfg.judge_cache_path))

    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])
    assert expected_error in str(errors[0])
    assert cache_rows[0] == cache_rows[1]
    assert len(cache_rows[0]) == 6
    persisted_trial_keys = {str(row[0]) for row in cache_rows[0]}
    assert lower_trial_key not in persisted_trial_keys
    assert higher_trial_key not in persisted_trial_keys


def test_saturated_concurrency_preserves_serial_artifact_identity(tmp_path: Path) -> None:
    """Twenty completion-time commits cannot perturb canonical output bytes."""
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((idx,)) for idx in range(5)]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    total_trials = len(targets) * 4  # n_trials_per_target from _score_config
    assert total_trials >= 16, "must genuinely saturate a 16-worker pool"

    serial_cfg = _run_scoring(
        tmp_path,
        feature_dir,
        targets,
        provider,
        max_concurrent=1,
        label="cache_stress_serial",
    )
    hot_cfg = _run_scoring(
        tmp_path,
        feature_dir,
        targets,
        provider,
        max_concurrent=16,
        label="cache_stress_hot",
    )

    for attr in _BYTE_IDENTICAL_ARTIFACTS:
        serial_bytes = Path(getattr(serial_cfg, attr)).read_bytes()
        hot_bytes = Path(getattr(hot_cfg, attr)).read_bytes()
        assert serial_bytes == hot_bytes, f"{attr} diverged between serial and max_concurrent=16"

    assert _artifact_snapshot(serial_cfg) == _artifact_snapshot(hot_cfg)
