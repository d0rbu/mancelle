"""Concurrent execution: many features, no race on JSONL, no duplicate writes."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.runner import FeatureFilterRunner
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


def test_many_features_no_duplicate_writes(tmp_path) -> None:
    payloads = {i: build_payload(feature_id=i) for i in range(50)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    def _judge(desc, samples):
        return InterpJudgeRating(3, 3)

    d = str(tmp_path)
    cfg = FeatureFilterConfig(
        model_id="gemma-2-2b", sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d, feature_ids=list(range(50)), n_sae_features=100,
        samples_per_bin=3, interp_min_score=1, consistency_min_score=3,
        max_concurrent_fetches=8, max_concurrent_judges=4,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )
    FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_Tok()).run()
    rows = [json.loads(line) for line in Path(cfg.features_output_path).read_text().splitlines() if line.strip()]
    assert len(rows) == 50
    assert len({r["feature_id"] for r in rows}) == 50
    assert not list(Path(tmp_path).rglob("*.tmp"))
