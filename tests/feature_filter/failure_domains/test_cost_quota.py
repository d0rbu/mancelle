"""Cost-quota boundary: runner halts cleanly at max_judge_calls, counter is race-safe."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.runner import FeatureFilterRunner
from steering.eval.feature_filter.types import CostCapExceeded
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _cfg(tmp: Path, *, feature_ids: list[int], max_judge_calls: int | None,
         max_concurrent: int = 1) -> FeatureFilterConfig:
    d = str(tmp)
    return FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=feature_ids,
        n_sae_features=max(feature_ids) + 1,
        samples_per_bin=3,
        interp_min_score=1,
        consistency_min_score=1,
        max_concurrent_fetches=max_concurrent,
        max_concurrent_judges=max_concurrent,
        max_judge_calls=max_judge_calls,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )


def test_max_judge_calls_one_with_many_features_short_circuits(tmp_path: Path) -> None:
    """max_judge_calls=1 and 10 features: second feature's judge must raise CostCapExceeded."""
    n_features = 10
    payloads = {i: build_payload(feature_id=i) for i in range(n_features)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    judge_calls: list[str] = []

    def _judge(desc, samples):
        judge_calls.append(desc or "")
        return InterpJudgeRating(3, 3)

    cfg = _cfg(tmp_path, feature_ids=list(range(n_features)),
               max_judge_calls=1, max_concurrent=1)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_Tok())
    with pytest.raises(CostCapExceeded):
        runner.run()
    # Exactly one judge call must have gone through.
    assert len(judge_calls) == 1


def test_cost_cap_exceeded_message_names_feature_id_and_cap(tmp_path: Path) -> None:
    """CostCapExceeded carries both the cap value and the feature_id that tripped it."""
    payloads = {i: build_payload(feature_id=i) for i in range(3)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    cfg = _cfg(tmp_path, feature_ids=[0, 1, 2], max_judge_calls=1, max_concurrent=1)
    runner = FeatureFilterRunner(
        cfg,
        np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(CostCapExceeded) as excinfo:
        runner.run()
    msg = str(excinfo.value)
    assert "max_judge_calls=1" in msg
    assert "feature_id=" in msg


def test_concurrent_cost_cap_never_overshoots(tmp_path: Path) -> None:
    """Under heavy concurrency the judge counter is mutex-guarded: number of
    judge invocations never exceeds max_judge_calls."""
    n_features = 20
    cap = 5
    payloads = {i: build_payload(feature_id=i) for i in range(n_features)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    counter_lock = threading.Lock()
    actual_calls = {"n": 0}

    def _judge(desc, samples):
        with counter_lock:
            actual_calls["n"] += 1
        return InterpJudgeRating(3, 3)

    cfg = _cfg(tmp_path, feature_ids=list(range(n_features)),
               max_judge_calls=cap, max_concurrent=8)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_Tok())
    with pytest.raises(CostCapExceeded):
        runner.run()
    assert actual_calls["n"] <= cap
