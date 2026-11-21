"""Schema migration: manifest + JSONL rows must carry SCHEMA_VERSION, off-by-one is rejected."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.runner import FeatureFilterRunner
from steering.eval.feature_filter.types import (
    SCHEMA_VERSION,
    FeatureFilterReport,
    FeatureInterpScore,
    SchemaVersionError,
)
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _cfg(tmp: Path, *, feature_ids: list[int]) -> FeatureFilterConfig:
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


def test_schema_version_constant_matches_dataclass_defaults() -> None:
    """SCHEMA_VERSION is 3 and appears as a field on both artifact dataclasses."""
    assert SCHEMA_VERSION == 3
    assert "schema_version" in {f.name for f in fields(FeatureInterpScore)}
    assert "schema_version" in {f.name for f in fields(FeatureFilterReport)}


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_v1_jsonl_row_rejected_on_resume(tmp_path: Path) -> None:
    """A persisted v1 JSONL row must be rejected on resume."""
    cfg = _cfg(tmp_path, feature_ids=[1])
    jsonl = Path(cfg.features_output_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(json.dumps({
        "feature_id": 1, "description": "x", "outcome": "accepted",
        "interpretability": 3, "consistency": 3, "samples": [],
        "error_message": None, "schema_version": 1,
    }) + "\n")
    runner = FeatureFilterRunner(
        cfg, np_cache=MagicMock(),
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(SchemaVersionError):
        runner._load_resume_set()


def test_v2_jsonl_row_accepted(tmp_path: Path) -> None:
    """A v2 JSONL row is accepted and its feature_id enters the resume set."""
    cfg = _cfg(tmp_path, feature_ids=[7])
    jsonl = Path(cfg.features_output_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(json.dumps({
        "feature_id": 7, "description": "x", "outcome": "accepted",
        "interpretability": 3, "consistency": 3, "samples": [],
        "error_message": None, "schema_version": SCHEMA_VERSION,
    }) + "\n")
    runner = FeatureFilterRunner(
        cfg, np_cache=MagicMock(),
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    assert runner._load_resume_set() == {7}


def test_manifest_with_older_schema_version_is_rejected(tmp_path: Path) -> None:
    """A manifest from a previous schema (version-1) must raise SchemaVersionError."""
    cfg = _cfg(tmp_path, feature_ids=[0])
    Path(cfg.runner_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.runner_manifest_path).write_text(json.dumps({
        "schema_version": SCHEMA_VERSION - 1,
        "model_id": cfg.model_id,
        "sae_source": cfg.sae_source,
    }))
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    runner = FeatureFilterRunner(
        cfg, np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(SchemaVersionError):
        runner.run()


def test_schema2_manifest_rejected_after_2to3_bump(tmp_path: Path) -> None:
    """Pin the specific gemma-scope-2 follow-up migration boundary.

    The `not_published` outcome bumped SCHEMA_VERSION 2->3. A resume
    manifest written under the old schema 2 (literal, not
    ``SCHEMA_VERSION - 1`` which would drift on a future bump) must be
    rejected cleanly with the typed ``SchemaVersionError`` rather than
    silently resuming with stale classification semantics.
    """
    assert SCHEMA_VERSION >= 3, "this pin is only meaningful post 2->3 bump"
    cfg = _cfg(tmp_path, feature_ids=[0])
    Path(cfg.runner_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.runner_manifest_path).write_text(json.dumps({
        "schema_version": 2,
        "model_id": cfg.model_id,
        "sae_source": cfg.sae_source,
    }))
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    runner = FeatureFilterRunner(
        cfg, np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(SchemaVersionError):
        runner.run()


def test_manifest_with_future_schema_version_is_rejected(tmp_path: Path) -> None:
    """Forward-compat: a future schema version must also be rejected (strict equality)."""
    cfg = _cfg(tmp_path, feature_ids=[0])
    Path(cfg.runner_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.runner_manifest_path).write_text(json.dumps({
        "schema_version": SCHEMA_VERSION + 1,
        "model_id": cfg.model_id,
        "sae_source": cfg.sae_source,
    }))
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    runner = FeatureFilterRunner(
        cfg, np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(SchemaVersionError):
        runner.run()


def test_jsonl_row_with_mismatched_schema_version_is_rejected(tmp_path: Path) -> None:
    """Resume-load rejects a JSONL row whose schema_version != current."""
    cfg = _cfg(tmp_path, feature_ids=[0])
    Path(cfg.features_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.features_output_path).write_text(
        json.dumps({"feature_id": 0, "schema_version": SCHEMA_VERSION + 1}) + "\n"
    )
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    runner = FeatureFilterRunner(
        cfg, np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(SchemaVersionError):
        runner.run()
