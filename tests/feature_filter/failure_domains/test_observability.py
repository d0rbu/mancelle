"""Observability: error messages include feature_id, loguru emits warnings on soft failures."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from loguru import logger

from errors import FeatureFilterError
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeConfig, make_interp_judge
from steering.eval.feature_filter.runner import FeatureFilterRunner
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _capture_loguru() -> tuple[io.StringIO, int]:
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{level}|{message}")
    return sink, handler_id


def test_fetch_error_rows_include_feature_id_and_message(tmp_path: Path) -> None:
    """Runner attaches feature_id and error text to fetch_error outcome rows."""
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()

    def _get(fid: int):
        if fid == 1:
            raise RuntimeError("neuronpedia returned 500")
        return payloads[fid]

    cache.get.side_effect = _get

    d = str(tmp_path)
    cfg = FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=[0, 1],
        n_sae_features=2,
        samples_per_bin=3,
        interp_min_score=1,
        consistency_min_score=1,
        max_concurrent_fetches=1,
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
    from steering.eval.feature_filter.interp_judge import InterpJudgeRating

    FeatureFilterRunner(
        cfg,
        np_cache=cache,
        judge=lambda desc, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    ).run()
    rows = [
        json.loads(line)
        for line in Path(cfg.features_output_path).read_text().splitlines()
        if line.strip()
    ]
    err_rows = [r for r in rows if r["outcome"] == "fetch_error"]
    assert len(err_rows) == 1
    assert err_rows[0]["feature_id"] == 1
    assert "500" in err_rows[0]["error_message"]


def test_interp_judge_logs_warning_on_llm_transport_failure() -> None:
    """A LLM transport failure is swallowed into None + a loguru WARNING."""
    sink, handler_id = _capture_loguru()
    try:
        class _BrokenClient:
            _cfg = type("C", (), {"model": "m"})()

            def complete(self, messages, cache_key):
                raise AxBenchBaseError("transport broke")

        judge = make_interp_judge(
            _BrokenClient(), InterpJudgeConfig(example_window_chars=0, parse_max_retries=0)
        )
        result = judge("desc", [])
        assert result is None
        out = sink.getvalue()
        assert "WARNING" in out
        assert "transport" in out.lower() or "llm" in out.lower()
    finally:
        logger.remove(handler_id)


def test_interp_judge_logs_warning_on_parse_failure() -> None:
    """When the LLM returns garbage, judge retries and logs WARNING for each bad parse."""
    sink, handler_id = _capture_loguru()
    try:
        class _NoiseClient:
            _cfg = type("C", (), {"model": "m"})()

            def complete(self, messages, cache_key):
                return "this is not a valid interpretability/consistency response"

        judge = make_interp_judge(
            _NoiseClient(), InterpJudgeConfig(example_window_chars=0, parse_max_retries=1)
        )
        result = judge("desc", [])
        assert result is None
        # Both attempts should log a parse warning.
        assert sink.getvalue().count("Could not parse") == 2
    finally:
        logger.remove(handler_id)


def test_feature_filter_error_str_includes_feature_id_context() -> None:
    """Manual wrapping pattern used by the runner preserves feature_id in the text."""
    err = FeatureFilterError("judging feature_id=99: some inner fail")
    assert "feature_id=99" in str(err)
