"""Mandatory pre-release live-smoke: hits real Neuronpedia + OpenRouter.

Gated on RUN_LIVE_TESTS=1 plus NEURONPEDIA_API_KEY and OPENROUTER_API_KEY.
Skipped in CI by default; runnable manually for sign-off.
"""
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import (
    InterpJudgeConfig,
    make_interp_judge,
)
from steering.eval.feature_filter.neuronpedia_activations import (
    NeuronpediaActivationCache,
)
from steering.eval.feature_filter.runner import FeatureFilterRunner

RUN_LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"
HAS_KEYS = bool(os.environ.get("NEURONPEDIA_API_KEY")) and bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.slow
@pytest.mark.skipif(not (RUN_LIVE and HAS_KEYS), reason="Live-smoke disabled; set RUN_LIVE_TESTS=1 and API keys.")
def test_live_smoke_single_feature(tmp_path) -> None:
    d = str(tmp_path)
    cfg = FeatureFilterConfig(
        model_id="gemma-2-2b", sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d, feature_ids=[42], n_sae_features=16384,
        samples_per_bin=3, interp_min_score=0, consistency_min_score=0,
        max_concurrent_fetches=1, max_concurrent_judges=1,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(parse_max_retries=1),
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    cache = NeuronpediaActivationCache(cfg.model_id, cfg.sae_source, cfg.neuronpedia_cache_dir)
    llm = OpenRouterLLMClient(
        LLMClientConfig(provider="openrouter", model=cfg.judge.model, requests_per_minute=cfg.judge.requests_per_minute),
        response_cache=SynthResponseCache(Path(cfg.judge_cache_path)),
    )
    judge = make_interp_judge(llm, InterpJudgeConfig(parse_max_retries=1))
    report = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=tokenizer).run()
    assert report.n_total == 1
    assert report.n_fetch_errors == 0
