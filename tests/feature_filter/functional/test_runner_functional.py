"""End-to-end functional run with mocked Neuronpedia + judge."""
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


def test_functional_mixed_outcomes(tmp_path) -> None:
    # 3 features: 1 accepted, 1 rejected, 1 fetch_error
    payloads = {
        1: build_payload(feature_id=1, description="good"),
        2: build_payload(feature_id=2, description="bad"),
    }
    cache = MagicMock()
    def _get(fid):
        if fid == 3:
            raise RuntimeError("boom")
        return payloads[fid]
    cache.get.side_effect = _get
    cache.get_cached.return_value = None

    ratings = {1: InterpJudgeRating(4, 5), 2: InterpJudgeRating(0, 0)}
    def _judge(desc, samples):
        # dispatch by description
        if desc == "good":
            return ratings[1]
        return ratings[2]

    d = str(tmp_path)
    cfg = FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=[1, 2, 3],
        n_sae_features=16384,
        samples_per_bin=3,
        interp_min_score=1,
        consistency_min_score=3,
        max_concurrent_fetches=2,
        max_concurrent_judges=1,
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
    report = runner.run()
    assert report.n_accepted == 1
    assert report.n_rejected == 1
    assert report.n_fetch_errors == 1
    assert Path(cfg.accepted_features_path).read_text() == "1\n"
    rep = json.loads(Path(cfg.report_output_path).read_text())
    assert rep["n_accepted"] == 1
