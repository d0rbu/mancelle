"""Threading failure domain: lockfile contention, concurrent dedup, exception propagation."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from errors import FeatureFilterError
from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.runner import FeatureFilterRunner
from steering.eval.feature_filter.types import (
    ConcurrentRunnerError,
)
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _cfg(tmp: Path, *, feature_ids: list[int], max_concurrent: int = 2) -> FeatureFilterConfig:
    d = str(tmp)
    return FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=feature_ids,
        n_sae_features=max(feature_ids) + 1 if feature_ids else 1,
        samples_per_bin=3,
        interp_min_score=1,
        consistency_min_score=1,
        max_concurrent_fetches=max_concurrent,
        max_concurrent_judges=max_concurrent,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )


def test_second_runner_on_same_lockfile_raises_concurrent_error(tmp_path: Path) -> None:
    """Two runners in the same process on the same output_dir: second raises ConcurrentRunnerError."""
    payloads = {i: build_payload(feature_id=i) for i in range(4)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    # Slow judge so runner 1 is still holding the lock when runner 2 starts.
    gate = threading.Event()
    release = threading.Event()

    def slow_judge(desc, samples):
        gate.set()
        release.wait(timeout=5.0)
        return InterpJudgeRating(3, 3)

    cfg1 = _cfg(tmp_path, feature_ids=[0, 1, 2, 3])
    cfg2 = _cfg(tmp_path, feature_ids=[0, 1, 2, 3])
    runner1 = FeatureFilterRunner(cfg1, np_cache=cache, judge=slow_judge, tokenizer=_Tok())
    runner2 = FeatureFilterRunner(cfg2, np_cache=cache, judge=slow_judge, tokenizer=_Tok())

    t1_exc: list[BaseException] = []

    def _run1() -> None:
        try:
            runner1.run()
        except BaseException as exc:  # pragma: no cover
            t1_exc.append(exc)

    t1 = threading.Thread(target=_run1)
    t1.start()
    assert gate.wait(timeout=5.0), "runner1 never reached judge"

    with pytest.raises(ConcurrentRunnerError) as excinfo:
        runner2.run()
    assert ".lock" in str(excinfo.value)

    release.set()
    t1.join(timeout=5.0)
    assert not t1_exc


def test_concurrent_feature_writes_are_deduplicated(tmp_path: Path) -> None:
    """High-concurrency run: every feature_id appears exactly once in the JSONL."""
    n = 40
    payloads = {i: build_payload(feature_id=i) for i in range(n)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    def _judge(desc, samples):
        return InterpJudgeRating(3, 3)

    cfg = _cfg(tmp_path, feature_ids=list(range(n)), max_concurrent=8)
    FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_Tok()).run()

    import json
    rows = [
        json.loads(line)
        for line in Path(cfg.features_output_path).read_text().splitlines()
        if line.strip()
    ]
    ids = [r["feature_id"] for r in rows]
    assert len(ids) == n
    assert len(set(ids)) == n  # deduped


def test_feature_filter_error_preserves_cause_chain() -> None:
    """Runner-style error wrapping keeps __cause__ and feature_id in the message."""
    root = ValueError("network timeout")
    try:
        try:
            raise root
        except ValueError as e:
            raise FeatureFilterError("fetch failed for feature_id=13") from e
    except FeatureFilterError as outer:
        assert "feature_id=13" in str(outer)
        assert outer.__cause__ is root


def test_blocking_worker_does_not_deadlock_shutdown(tmp_path: Path) -> None:
    """A judge that blocks past a sentinel must not prevent a fast runner on a
    different lockfile from finishing — verifies locks are per-path."""
    payloads = {i: build_payload(feature_id=i) for i in range(2)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    fast_dir.mkdir()
    slow_dir.mkdir()

    release = threading.Event()

    def slow_judge(desc, samples):
        release.wait(timeout=5.0)
        return InterpJudgeRating(3, 3)

    def fast_judge(desc, samples):
        return InterpJudgeRating(3, 3)

    cfg_slow = _cfg(slow_dir, feature_ids=[0, 1])
    cfg_fast = _cfg(fast_dir, feature_ids=[0, 1])

    slow_thread = threading.Thread(
        target=FeatureFilterRunner(
            cfg_slow, np_cache=cache, judge=slow_judge, tokenizer=_Tok()
        ).run
    )
    slow_thread.start()
    time.sleep(0.05)

    # Different lockfile → must complete independently.
    FeatureFilterRunner(
        cfg_fast, np_cache=cache, judge=fast_judge, tokenizer=_Tok()
    ).run()
    assert Path(cfg_fast.report_output_path).is_file()

    release.set()
    slow_thread.join(timeout=5.0)
    assert not slow_thread.is_alive()
