"""Resume: a second run skips already-scored features and appends the rest."""
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


def _cfg(tmp_path, feature_ids):
    d = str(tmp_path)
    return FeatureFilterConfig(
        model_id="gemma-2-2b", sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d, feature_ids=feature_ids, n_sae_features=100,
        samples_per_bin=3, interp_min_score=1, consistency_min_score=3,
        max_concurrent_fetches=1, max_concurrent_judges=1,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )


def test_resumes_skipping_prior_features(tmp_path) -> None:
    payloads = {i: build_payload(feature_id=i, description=f"d{i}") for i in range(3)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]

    call_descriptions: list[str] = []
    def _judge(desc, samples):
        call_descriptions.append(desc)
        return InterpJudgeRating(3, 3)

    # Run 1: process [0, 1]
    cfg1 = _cfg(tmp_path, [0, 1])
    FeatureFilterRunner(cfg1, np_cache=cache, judge=_judge, tokenizer=_Tok()).run()
    assert len(call_descriptions) == 2

    # Run 2: [0, 1, 2] — only feature 2 should hit the judge
    cfg2 = _cfg(tmp_path, [0, 1, 2])
    FeatureFilterRunner(cfg2, np_cache=cache, judge=_judge, tokenizer=_Tok()).run()
    assert len(call_descriptions) == 3  # +1 only
    assert call_descriptions[-1] == "d2"
    assert not list(Path(tmp_path).rglob("*.tmp"))
