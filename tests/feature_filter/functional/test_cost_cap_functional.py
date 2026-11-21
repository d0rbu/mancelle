"""Cost cap halts the run after N judge calls, preserving partial artifacts."""
import json
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
    def convert_tokens_to_string(toks):
        return "".join(toks)


def test_cost_cap_halts_and_preserves_partial(tmp_path) -> None:
    feature_ids = list(range(10))
    payloads = {i: build_payload(feature_id=i) for i in feature_ids}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    judge_call_count = 0
    def _judge(desc, samples):
        nonlocal judge_call_count
        judge_call_count += 1
        return InterpJudgeRating(3, 3)

    d = str(tmp_path)
    cfg = FeatureFilterConfig(
        model_id="gemma-2-2b", sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d, feature_ids=feature_ids, n_sae_features=100,
        samples_per_bin=3, interp_min_score=1, consistency_min_score=3,
        max_concurrent_fetches=1, max_concurrent_judges=1,
        max_judge_calls=3,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_Tok())
    with pytest.raises(CostCapExceeded):
        runner.run()
    # First 3 rows should have been appended before the halt
    assert judge_call_count == 3
    rows = [json.loads(line) for line in Path(cfg.features_output_path).read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert {row["feature_id"] for row in rows}.issubset(set(feature_ids))
    assert not list(Path(tmp_path).rglob("*.tmp"))
