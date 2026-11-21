from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import yaml
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import steering.eval.autointerp_filter.config as config_module
import steering.eval.autointerp_filter.token_examples as token_examples_module
from errors import RemovedFeatureError
from steering.eval.autointerp_filter.artifacts import score_to_ranked_yaml
from steering.eval.autointerp_filter.cache import IntruderTrialCache, TokenExampleCache
from steering.eval.autointerp_filter.cli import build_config, run_from_config
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    IntruderJudgeConfig,
    NeuronpediaDescriptionConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
    validate_config,
)
from steering.eval.autointerp_filter.document_blacklist import (
    EMPTY_DOCUMENT_BLACKLIST_SHA256,
    resolve_document_blacklist,
)
from steering.eval.autointerp_filter.intruder import (
    IntruderDetectionJudge,
    build_intruder_messages,
    build_intruder_messages_from_texts,
    judge_cache_key,
    parse_intruder_response,
    render_trial_examples,
)
from steering.eval.autointerp_filter.manual_coverage import (
    compute_manual_filter_coverage,
    dump_manual_coverage_report,
    load_manual_feature_ids,
)
from steering.eval.autointerp_filter.manual_filter import render_manual_feature_filter_yaml
from steering.eval.autointerp_filter.prompt_calibration import (
    PromptCalibrationConfig,
    run_prompt_calibration,
)
from steering.eval.autointerp_filter.prompts import (
    IntruderFewShotSet,
    IntruderPromptVariant,
    available_few_shot_sets,
    available_prompt_variants,
    resolve_intruder_prompt_spec,
)
from steering.eval.autointerp_filter.renderers import TokenWindowRenderer
from steering.eval.autointerp_filter.runner import AutoInterpFilterRunner
from steering.eval.autointerp_filter.source_index import (
    SourceIndexData,
    SourceIndexRequest,
    fingerprint_paths,
    load_or_build_source_index,
)
from steering.eval.autointerp_filter.targets import resolve_targets
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    SourceDocRef,
    TokenExampleSampler,
    _selected_sae_activations,
    deterministic_intruder_highlight_indices,
    positive_highlight_indices,
)
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCacheError,
    AutoInterpConfigError,
    AutoInterpCostCapExceeded,
    AutoInterpDataError,
    IntruderPromptError,
    IntruderResponseParseError,
    IntruderTrial,
    IntruderTrialResult,
    RenderedExample,
    ScoringTarget,
    TargetKind,
    TargetOutcome,
    TargetScore,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient
from steering.eval.neuronpedia_client import NeuronpediaFetchError
from steering.feature_io import TensorKey, save_features
from steering.refined_types import (
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
    UnitInterval,
)
from steering.sae_resolver import ActivationType, ResolvedSAE, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)
from tests.steering.autointerp_filter._token_example_reference import load_shard_text


def _positive(value: int) -> PositiveInt:
    return PositiveInt.parse(value)


def _non_negative(value: int) -> NonNegativeInt:
    return NonNegativeInt.parse(value)


def _unit(value: int | float) -> UnitInterval:
    return UnitInterval.parse(value)


def _positive_finite(value: int | float) -> PositiveFiniteNumber:
    return PositiveFiniteNumber.parse(value)


def _write_feature_shard(
    feature_dir: Path,
    *,
    data: dict | None = None,
    ignore_bos_features: bool = True,
) -> Path:
    feature_dir.mkdir(parents=True, exist_ok=True)
    data = data or {
        TensorKey.OFFSETS: t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0, 0, 1, 2], dtype=t.int64),
        TensorKey.FEATURE_MAGS: t.tensor([10.0, 8.0, 3.0, 2.0]),
        TensorKey.DOC_IDS: t.tensor([0, 1, 2, 3], dtype=t.int64),
        TensorKey.SEGMENT_INDICES: t.tensor([0, 0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_POSITIONS: t.tensor([0, 0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_CHAR_STARTS: t.tensor([0, 0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_CHAR_ENDS: t.tensor([10, 10, 10, 10], dtype=t.int64),
        "n_docs": 4,
        "n_segments": 4,
        "layer": 0,
        "context_window": 64,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 16,
        "context_independent_features": "false",
        "ignore_bos_features": "true" if ignore_bos_features else "false",
    }
    return save_features(data, feature_dir / "shard_00000_features.safetensors.zst")


def _write_parquet(data_dir: Path, texts: list[str]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "shard_00000.parquet"
    pq.write_table(pa.table({"text": texts}), path, row_group_size=1)
    return path


def _write_stats(feature_dir: Path, *, n_segments: int = 4, n_sae_features: int = 6) -> Path:
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=16,
            context_window=64,
            layer=0,
            n_sae_features=n_sae_features,
            generated_at="test",
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=n_segments,
            n_tokens_gpt2=n_segments * 16,
            mean_tokens_per_doc=16.0,
            median_tokens_per_doc=16,
        ),
        features=SegmentFeatureStats(
            n_segments=n_segments,
            n_features_seen=n_sae_features,
            mean_active_per_segment=1.0,
            median_active_per_segment=1,
            p90_active_per_segment=1,
            max_active_per_segment=1,
            min_active_per_segment=1,
            n_total_sae_features=n_sae_features,
            pct_features_active=1.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=0,
            mean_bytes_per_doc=0.0,
            mean_bytes_per_token=0.0,
        ),
        per_feature_stats={},
        extrapolation_10b=Extrapolation(
            target_tokens=0,
            estimated_docs=0,
            estimated_storage_gb=0.0,
            estimated_wall_time_hours=0.0,
        ),
    )
    return save_statistics(stats, feature_dir / "feature_statistics.safetensors")


def _cfg(
    tmp_path: Path,
    feature_dir: Path,
    **overrides: Any,
) -> AutoInterpFilterConfig:
    output_dir = tmp_path / "out"
    override_keys = set(overrides)
    base: dict[str, Any] = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "data_dir": None,
        "n_sae_features": 6,
        "feature_ids": [0],
        "n_trials_per_target": 2,
        "min_trials_for_score": 10,
        "n_positive_examples": 2,
        "n_quantiles": 2,
        "score_threshold": 1.0,
        "top_k": None,
        "fail_on_empty": False,
        "audit_sample_limit": 20,
        "token_example_cache_path": str(output_dir / "token_example_cache.sqlite"),
        "judge_cache_path": str(output_dir / "intruder_trial_cache.sqlite"),
        "target_scores_path": str(output_dir / "target_scores.jsonl"),
        "ranked_yaml_path": str(output_dir / "autointerp_ranked.yaml"),
        "accepted_features_path": str(output_dir / "accepted_feature_ids.txt"),
        "manual_feature_filter_path": str(output_dir / "manual_feature_filter.yaml"),
        "manual_feature_filter_meta_path": str(output_dir / "manual_feature_filter.meta.json"),
        "report_output_path": str(output_dir / "report.json"),
        "trial_audit_path": str(output_dir / "trial_audit.jsonl"),
        "trial_audit_yaml_path": str(output_dir / "trial_audit.yaml"),
        "runner_manifest_path": str(output_dir / "runner_manifest.json"),
        "lockfile_path": str(output_dir / ".lockfile"),
        "judge": IntruderJudgeConfig(model="test-judge"),
        "token_examples": TokenExamplesConfig(
            example_ctx_len=_positive(4),
            max_positive_source_docs_per_target=_positive(8),
            max_intruder_source_docs=_positive(8),
            max_windows_per_source_doc=_positive(4),
            max_positive_windows_per_target=_positive(16),
            max_non_activating_windows_per_target=_positive(16),
        ),
        "source_index": SourceIndexConfig(
            path=str(output_dir / "source_doc_index.sqlite"),
            manifest_path=str(output_dir / "source_doc_index.manifest.json"),
            chunk_size=_positive(8),
        ),
    }
    base.update(overrides)
    final_output_dir = Path(str(base["output_dir"]))
    for key, filename in (
        ("token_example_cache_path", "token_example_cache.sqlite"),
        ("judge_cache_path", "intruder_trial_cache.sqlite"),
        ("target_scores_path", "target_scores.jsonl"),
        ("ranked_yaml_path", "autointerp_ranked.yaml"),
        ("accepted_features_path", "accepted_feature_ids.txt"),
        ("manual_feature_filter_path", "manual_feature_filter.yaml"),
        ("manual_feature_filter_meta_path", "manual_feature_filter.meta.json"),
        ("report_output_path", "report.json"),
        ("trial_audit_path", "trial_audit.jsonl"),
        ("trial_audit_yaml_path", "trial_audit.yaml"),
        ("runner_manifest_path", "runner_manifest.json"),
        ("lockfile_path", ".lockfile"),
    ):
        if key not in override_keys:
            base[key] = str(final_output_dir / filename)
    return AutoInterpFilterConfig(**base)


def _target(ids: tuple[int] = (0,)) -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=ids,
        target_key=f"feature:{ids[0]}",
        display_name=f"feature_{ids[0]}",
    )


def _ref(
    name: str,
    activations: list[float],
    *,
    active: bool = True,
    doc_id: int = 0,
    token_start: int = 0,
    target_ids: tuple[int, ...] = (0,),
) -> TokenWindowRef:
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"hash-{doc_id}",
        token_start=token_start,
        token_end=token_start + len(activations),
        token_ids=tuple(range(token_start, token_start + len(activations))),
        token_strings=tuple(f"{name}{idx}" for idx in range(len(activations))),
        target_feature_ids=target_ids,
        activations=tuple(activations),
        max_activation=max(activations, default=0.0),
        quantile=None,
        active=active,
    )


class _FakeTokenProvider:
    def __init__(self, examples: dict[str, TokenExampleSet]) -> None:
        self.examples = examples
        self.prepared_targets: list[ScoringTarget] = []

    def prepare(self, targets: list[ScoringTarget]) -> None:
        self.prepared_targets = list(targets)

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        if target.target_key not in self.examples and len(self.examples) == 1:
            return next(iter(self.examples.values()))
        return self.examples[target.target_key]

    def cache_context(self) -> dict:
        return {"fake_provider": sorted(self.examples)}


class _MarkerRenderer:
    def render(
        self,
        ref: TokenWindowRef,
        *,
        activating: bool,
        highlighted_token_indices: tuple[int, ...],
    ) -> RenderedExample:
        label = "POSITIVE" if activating else "INTRUDER"
        return RenderedExample(
            text=f"<<{label}:{ref.stable_key()}>>",
            ref=ref,
            activating=activating,
            highlighted_token_indices=highlighted_token_indices,
        )


class _IntruderFindingClient(LLMJudgeClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        self.calls += 1
        for line in messages[-1]["content"].splitlines():
            if line.startswith("Example ") and "INTRUDER" in line:
                idx = int(line.split(":", 1)[0].split()[1])
                return f"[RESPONSE]: {idx}"
        raise AssertionError("fake prompt did not contain an intruder marker")


class _FixedResponseClient(LLMJudgeClient):
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        return self.response


class _ExplodingClient(LLMJudgeClient):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        raise AssertionError("judge should have been served from cache")


class _RecordingIntruderJudge:
    def __init__(
        self,
        *,
        expected_parallel: int | None = None,
        delays: dict[int, float] | None = None,
        fail_trial_indices: set[int] | None = None,
        wrong_trial_indices: set[int] | None = None,
    ) -> None:
        self.calls: list[int] = []
        self.peak_concurrent = 0
        self._expected_parallel = expected_parallel
        self._delays = delays or {}
        self._fail_trial_indices = fail_trial_indices or set()
        self._wrong_trial_indices = wrong_trial_indices or set()
        self._in_flight = 0
        self._lock = threading.Lock()
        self._parallel_event = threading.Event()

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _MarkerRenderer,
    ) -> IntruderTrialResult:
        with self._lock:
            self.calls.append(trial.trial_index)
            self._in_flight += 1
            self.peak_concurrent = max(self.peak_concurrent, self._in_flight)
            if self._expected_parallel is not None and self._in_flight >= self._expected_parallel:
                self._parallel_event.set()
        try:
            if self._expected_parallel is not None:
                self._parallel_event.wait(timeout=2.0)
            delay = self._delays.get(trial.trial_index, 0.0)
            if delay > 0.0:
                time.sleep(delay)
            if trial.trial_index in self._fail_trial_indices:
                raise IntruderResponseParseError(
                    f"synthetic parse failure for trial {trial.trial_index}"
                )
            examples = render_trial_examples(trial, renderer)
            prediction = trial.intruder_index
            if trial.trial_index in self._wrong_trial_indices:
                prediction = (trial.intruder_index + 1) % len(examples)
            return IntruderTrialResult(
                trial_key=trial.trial_key,
                prediction=prediction,
                correct_index=trial.intruder_index,
                correct=prediction == trial.intruder_index,
                response=f"[RESPONSE]: {prediction}",
                cached=False,
                error_message=None,
                rendered_examples=tuple(examples),
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def _example_set(target: ScoringTarget) -> TokenExampleSet:
    positives = tuple(
        _ref(
            f"p{idx}-",
            [0.0, float(10 - idx), 0.0, 0.0],
            doc_id=idx,
            target_ids=target.feature_ids,
        )
        for idx in range(4)
    )
    non_activating = tuple(
        _ref(
            f"n{idx}-",
            [0.0, 0.0, 0.0, 0.0],
            active=False,
            doc_id=10 + idx,
            target_ids=target.feature_ids,
        )
        for idx in range(4)
    )
    assert all(ref.target_feature_ids == target.feature_ids for ref in positives)
    return TokenExampleSet(positives=positives, non_activating=non_activating)


def _artifact_snapshot(cfg: AutoInterpFilterConfig) -> dict[str, object]:
    return {
        "target_scores": [
            json.loads(line) for line in Path(cfg.target_scores_path).read_text().splitlines()
        ],
        "ranked_yaml": yaml.safe_load(Path(cfg.ranked_yaml_path).read_text()),
        "accepted_features": Path(cfg.accepted_features_path).read_text(),
        "manual_feature_filter": Path(cfg.manual_feature_filter_path).read_text(),
        "manual_feature_filter_meta": _normalize_output_dir(
            _without_generated_at(
                json.loads(Path(cfg.manual_feature_filter_meta_path).read_text())
            ),
            output_dir=Path(cfg.output_dir),
        ),
        "report": _normalize_output_dir(
            json.loads(Path(cfg.report_output_path).read_text()),
            output_dir=Path(cfg.output_dir),
        ),
        "trial_audit_jsonl": [
            json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
        ],
        "trial_audit_yaml": yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()),
    }


def _normalize_output_dir(value: object, *, output_dir: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _normalize_output_dir(item, output_dir=output_dir) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_output_dir(item, output_dir=output_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(output_dir), "<OUTPUT_DIR>")
    return value


def _without_generated_at(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("generated_at", None)
    return normalized


def test_resolve_targets_sorted_singleton_features_after_mask(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    cfg = _cfg(
        tmp_path,
        feature_dir,
        feature_ids=[2, 0, 1],
        feature_mask={"mode": "whitelist", "ids": [0, 2]},
    )

    targets = resolve_targets(
        cfg,
        n_sae_features=6,
        feature_dir=feature_dir,
    )

    assert [(target.kind, target.feature_ids) for target in targets] == [
        (TargetKind.feature, (0,)),
        (TargetKind.feature, (2,)),
    ]


def test_token_window_renderer_highlights_and_escapes_markers() -> None:
    ref = TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=0,
        source_text_hash="hash",
        token_start=0,
        token_end=4,
        token_ids=(1, 2, 3, 4),
        token_strings=("aa", "<<bb", "cc>>", "dd"),
        target_feature_ids=(0,),
        activations=(0.0, 1.0, 0.5, 0.0),
        max_activation=1.0,
        quantile=None,
        active=True,
    )

    rendered = TokenWindowRenderer().render(
        ref,
        activating=True,
        highlighted_token_indices=(1, 2),
    )

    assert rendered.text == "aa<<< <bbcc> >>>dd"
    assert rendered.text.count("<<") == 1
    assert rendered.text.count(">>") == 1
    assert rendered.highlighted_token_indices == (1, 2)
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="highlighted_token_indices",
    ):
        TokenWindowRenderer().render(ref, activating=True, highlighted_token_indices=())
    with pytest.raises(IntruderPromptError, match="out of range"):
        TokenWindowRenderer().render(ref, activating=True, highlighted_token_indices=(9,))


def test_positive_highlight_indices_use_relative_token_threshold() -> None:
    ref = _ref("x", [0.0, 2.0, 0.5, 4.0])

    assert positive_highlight_indices(ref, activation_threshold_ratio=0.5) == (1, 3)
    assert positive_highlight_indices(ref, activation_threshold_ratio=1.0) == (3,)
    assert positive_highlight_indices(ref, activation_threshold_ratio=0.0) == (0, 1, 2, 3)
    assert (
        positive_highlight_indices(
            _ref("z", [0.0, 0.0], active=False),
            activation_threshold_ratio=0.3,
        )
        == ()
    )

    inactive_positive_ref = _ref("z", [0.0, 0.0], active=True)
    assert (
        positive_highlight_indices(
            inactive_positive_ref,
            activation_threshold_ratio=0.3,
        )
        == ()
    )
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="highlighted_token_indices",
    ):
        TokenWindowRenderer().render(
            inactive_positive_ref,
            activating=True,
            highlighted_token_indices=(),
        )


def test_manual_feature_filter_yaml_is_comment_only_sorted() -> None:
    scores = [
        TargetScore(
            target_key="feature:5",
            display_name="feature_5",
            kind=TargetKind.feature,
            feature_ids=(5,),
            score=None,
            n_correct=0,
            n_trials=0,
            outcome=TargetOutcome.insufficient_examples,
        ),
        TargetScore(
            target_key="feature:12",
            display_name="feature_12",
            kind=TargetKind.feature,
            feature_ids=(12,),
            score=0.75,
            n_correct=3,
            n_trials=4,
            outcome=TargetOutcome.accepted,
        ),
        TargetScore(
            target_key="feature:2",
            display_name="feature_2",
            kind=TargetKind.feature,
            feature_ids=(2,),
            score=0.75,
            n_correct=3,
            n_trials=4,
            outcome=TargetOutcome.accepted,
        ),
        TargetScore(
            target_key="feature:1",
            display_name="feature_1",
            kind=TargetKind.feature,
            feature_ids=(1,),
            score=0.2,
            n_correct=1,
            n_trials=5,
            outcome=TargetOutcome.rejected,
        ),
    ]

    rendered = render_manual_feature_filter_yaml(
        scores,
        neuronpedia_model_id="gemma-3-4b",
        neuronpedia_sae_source="22-gemmascope-2-res-65k",
    )

    assert rendered.splitlines() == [
        '# - "02"  # score=0.750 https://www.neuronpedia.org/gemma-3-4b/22-gemmascope-2-res-65k/2',
        '# - "12"  # score=0.750 https://www.neuronpedia.org/gemma-3-4b/22-gemmascope-2-res-65k/12',
        '# - "01"  # score=0.200 https://www.neuronpedia.org/gemma-3-4b/22-gemmascope-2-res-65k/1',
        '# - "05"  # score=null https://www.neuronpedia.org/gemma-3-4b/22-gemmascope-2-res-65k/5',
    ]
    assert yaml.safe_load(rendered) is None


def test_manual_feature_filter_yaml_labels_partial_scores() -> None:
    rendered = render_manual_feature_filter_yaml(
        [
            TargetScore(
                target_key="feature:4",
                display_name="feature_4",
                kind=TargetKind.feature,
                feature_ids=(4,),
                score=0.5,
                n_correct=1,
                n_trials=1,
                outcome=TargetOutcome.accepted,
            ),
        ],
        neuronpedia_model_id="gemma-3-4b",
        neuronpedia_sae_source="22-gemmascope-2-res-65k",
        n_requested_trials=2,
    )

    assert rendered.splitlines() == [
        '# - "4"  # score=0.500 '
        "https://www.neuronpedia.org/gemma-3-4b/22-gemmascope-2-res-65k/4 "
        "trials=1/2 partial"
    ]
    assert yaml.safe_load(rendered) is None


def test_manual_filter_coverage_validates_yaml_ids(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("# - 1  # score=1.000\n")
    assert load_manual_feature_ids(path) == []
    path.write_text("- 1  # score=1.000\n- 3\n")
    assert load_manual_feature_ids(path) == [1, 3]
    path.write_text('- "00001"  # score=1.000\n- "00003"\n')
    assert load_manual_feature_ids(path) == [1, 3]

    bad_cases = [
        ("mapping", "feature_ids: [1]\n", "flat list"),
        ("bool", "- true\n", "strict int or quoted digits"),
        ("float", "- 1.2\n", "strict int or quoted digits"),
        ("string", "- abc\n", "strict int or quoted digits"),
        ("decimal-string", "- '1.2'\n", "strict int or quoted digits"),
        ("unquoted-padded", "- 064625\n", "zero-padded IDs must be quoted"),
        ("duplicate", "- 1\n- 1\n", "duplicate"),
        ("duplicate-padded", "- 1\n- '00001'\n", "duplicate"),
    ]
    for _name, text, match in bad_cases:
        path.write_text(text)
        with pytest.raises(AutoInterpConfigError, match=match):
            load_manual_feature_ids(path)
    with pytest.raises(AutoInterpConfigError, match="not found"):
        load_manual_feature_ids(tmp_path / "missing.yaml")


def test_load_manual_feature_ids_still_raises_autointerp_config_error(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("- 1\n- 1\n")
    with pytest.raises(AutoInterpConfigError, match="duplicate"):
        load_manual_feature_ids(path)


def test_manual_filter_coverage_empty_fast_path_and_streamed_counts(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    save_features(
        {
            TensorKey.OFFSETS: t.tensor([0, 2, 3, 5], dtype=t.int64),
            TensorKey.FEATURE_IDS: t.tensor([1, 3, 2, 1, 4], dtype=t.int64),
            TensorKey.FEATURE_MAGS: t.ones(5),
            TensorKey.DOC_IDS: t.tensor([0, 1, 2], dtype=t.int64),
            TensorKey.SEGMENT_INDICES: t.tensor([0, 0, 0], dtype=t.int64),
            TensorKey.SEGMENT_POSITIONS: t.tensor([0, 0, 0], dtype=t.int64),
            "n_docs": 3,
            "n_segments": 3,
            "layer": 0,
            "context_window": 64,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 16,
        },
        feature_dir / "shard_00000_features.safetensors.zst",
    )
    _write_stats(feature_dir, n_segments=3, n_sae_features=6)
    manual_path = tmp_path / "manual.yaml"
    manual_path.write_text("# - 1  # score=1.000\n")

    empty_report = compute_manual_filter_coverage(
        feature_dir=feature_dir,
        selection_yaml_path=manual_path,
    )

    assert empty_report["histogram"] == {0: 3}
    assert empty_report["percentiles"]["p100"] == 0

    manual_path.write_text("- 1\n")
    report = compute_manual_filter_coverage(
        feature_dir=feature_dir,
        selection_yaml_path=manual_path,
    )

    assert report["histogram"] == {0: 1, 1: 2}
    assert report["percentiles"] == {
        "p0": 0,
        "p1": 0,
        "p5": 0,
        "p10": 0,
        "p50": 1,
        "p90": 1,
        "p95": 1,
        "p99": 1,
        "p100": 1,
    }
    rendered = dump_manual_coverage_report(report)
    assert "0: 1" in rendered
    assert yaml.safe_load(rendered)["histogram"] == {0: 1, 1: 2}

    manual_path.write_text("- 1\n- 2\n- 3\n- 4\n")
    all_selected = compute_manual_filter_coverage(
        feature_dir=feature_dir,
        selection_yaml_path=manual_path,
    )
    assert all_selected["histogram"] == {1: 1, 2: 2}

    manual_path.write_text("- 6\n")
    with pytest.raises(AutoInterpConfigError, match="out of range"):
        compute_manual_filter_coverage(feature_dir=feature_dir, selection_yaml_path=manual_path)


def test_token_sampler_builds_stable_quantile_trials(tmp_path: Path) -> None:
    target = _target()
    cfg = _cfg(tmp_path, tmp_path / "features")
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    provider.prepare([target])
    sampler = TokenExampleSampler(cfg, provider=provider)

    trials = sampler.build_trials(target)
    trials2 = TokenExampleSampler(cfg, provider=provider).build_trials(target)

    assert len(trials) == cfg.n_trials_per_target
    assert [trial.trial_key for trial in trials] == [trial.trial_key for trial in trials2]
    for trial in trials:
        assert len(trial.positive_refs) == cfg.n_positive_examples
        assert trial.intruder_ref.max_activation == 0.0
        assert trial.intruder_highlight_indices == (3,)
        assert 0 <= trial.intruder_index <= cfg.n_positive_examples
        assert trial.quantile in {0, 1}


def test_token_sampler_returns_no_trials_without_enough_examples(tmp_path: Path) -> None:
    target = _target()
    cfg = _cfg(tmp_path, tmp_path / "features")
    provider = _FakeTokenProvider(
        {
            target.target_key: TokenExampleSet(
                positives=(_ref("p", [1.0, 0.0], doc_id=0),),
                non_activating=(),
            )
        }
    )
    provider.prepare([target])

    assert TokenExampleSampler(cfg, provider=provider).build_trials(target) == []


def test_token_sampler_dedupes_duplicate_source_windows(tmp_path: Path) -> None:
    target = _target()
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        n_trials_per_target=1,
        n_positive_examples=3,
        n_quantiles=1,
    )
    duplicate_low = _ref("dup-", [0.0, 1.0, 0.0, 0.0], doc_id=7)
    duplicate_high = _ref("dup-", [0.0, 3.0, 0.0, 0.0], doc_id=7)
    examples = TokenExampleSet(
        positives=(
            duplicate_low,
            duplicate_high,
            _ref("u1-", [0.0, 2.0, 0.0, 0.0], doc_id=8),
            _ref("u2-", [0.0, 4.0, 0.0, 0.0], doc_id=9),
        ),
        non_activating=tuple(
            _ref(f"n{idx}-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=20 + idx)
            for idx in range(3)
        ),
    )
    provider = _FakeTokenProvider({target.target_key: examples})
    provider.prepare([target])

    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)

    assert len(trials) == 1
    identities = [
        (ref.shard_stem, ref.shard_index, ref.doc_id, ref.token_start, ref.token_end)
        for ref in trials[0].positive_refs
    ]
    assert len(set(identities)) == len(identities)
    duplicate_ref = next(ref for ref in trials[0].positive_refs if ref.doc_id == 7)
    assert duplicate_ref.max_activation == 3.0


def test_token_sampler_rejects_overlapping_prompt_options(tmp_path: Path) -> None:
    target = _target()
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        n_trials_per_target=1,
        n_positive_examples=3,
        n_quantiles=1,
    )
    overlapping_a = _ref("doc-", [0.0, 5.0, 0.0, 0.0], doc_id=7, token_start=0)
    overlapping_b = _ref("doc-", [0.0, 4.0, 0.0, 0.0], doc_id=7, token_start=2)
    examples = TokenExampleSet(
        positives=(
            overlapping_a,
            overlapping_b,
            _ref("u1-", [0.0, 3.0, 0.0, 0.0], doc_id=8),
            _ref("u2-", [0.0, 2.0, 0.0, 0.0], doc_id=9),
            _ref("u3-", [0.0, 1.0, 0.0, 0.0], doc_id=10),
        ),
        non_activating=(
            _ref("doc-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=7, token_start=1),
            _ref("n1-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=20),
        ),
    )
    provider = _FakeTokenProvider({target.target_key: examples})
    provider.prepare([target])

    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)

    assert len(trials) == 1
    prompt_refs = (*trials[0].positive_refs, trials[0].intruder_ref)
    for idx, left in enumerate(prompt_refs):
        for right in prompt_refs[idx + 1 :]:
            if left.doc_id != right.doc_id:
                continue
            overlap = max(
                0,
                min(left.token_end, right.token_end) - max(left.token_start, right.token_start),
            )
            assert overlap == 0


def test_source_window_overlap_raises_on_token_mismatch() -> None:
    left = _ref("left-", [1.0, 0.0, 0.0], token_start=0)
    right = TokenWindowRef(
        shard_stem=left.shard_stem,
        shard_index=left.shard_index,
        doc_id=left.doc_id,
        source_text_hash=left.source_text_hash,
        token_start=1,
        token_end=3,
        token_ids=(999, 1000),
        token_strings=("bad0", "bad1"),
        target_feature_ids=left.target_feature_ids,
        activations=(0.0, 0.0),
        max_activation=0.0,
        quantile=None,
        active=False,
    )

    with pytest.raises(AutoInterpDataError, match="disagree on token IDs"):
        token_examples_module._source_windows_overlap(left, right)


def test_token_sampler_returns_no_trials_when_positive_refs_cannot_be_disjoint(
    tmp_path: Path,
) -> None:
    target = _target()
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        n_trials_per_target=1,
        n_positive_examples=2,
        n_quantiles=1,
    )
    examples = TokenExampleSet(
        positives=(
            _ref("doc-", [0.0, 5.0, 0.0, 0.0], doc_id=7, token_start=0),
            _ref("doc-", [0.0, 4.0, 0.0, 0.0], doc_id=7, token_start=1),
        ),
        non_activating=(_ref("n-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=20),),
    )
    provider = _FakeTokenProvider({target.target_key: examples})
    provider.prepare([target])

    assert TokenExampleSampler(cfg, provider=provider).build_trials(target) == []


def test_token_sampler_returns_no_trials_when_intruders_are_invalid(
    tmp_path: Path,
) -> None:
    target = _target()
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        n_trials_per_target=1,
        n_positive_examples=2,
        n_quantiles=1,
    )
    positive_a = _ref("p0-", [0.0, 5.0, 0.0, 0.0], doc_id=7, token_start=0)
    positive_b = _ref("p1-", [0.0, 4.0, 0.0, 0.0], doc_id=8, token_start=0)
    examples = TokenExampleSet(
        positives=(positive_a, positive_b),
        non_activating=(
            _ref("p0-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=7, token_start=0),
            _ref("active-", [0.0, 1.0, 0.0, 0.0], active=False, doc_id=20),
        ),
    )
    provider = _FakeTokenProvider({target.target_key: examples})
    provider.prepare([target])

    assert TokenExampleSampler(cfg, provider=provider).build_trials(target) == []


def test_document_blacklist_excludes_positive_and_intruder_docs(tmp_path: Path) -> None:
    target = _target()
    blacklist_path = tmp_path / "document_blacklist.txt"
    blacklist_path.write_text(
        "# comment line, blank lines below are ignored\n"
        "\n"
        "shard_00000_features:1\n"
        "shard_00000_features:11\n"
    )
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        document_blacklist_path=str(blacklist_path),
    )
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    provider.prepare([target])

    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)

    assert trials
    for trial in trials:
        assert all(ref.doc_id != 1 for ref in trial.positive_refs)
        assert trial.intruder_ref.doc_id != 11


def test_document_blacklist_none_is_byte_identical_to_disabled(tmp_path: Path) -> None:
    target = _target()
    cfg_disabled = _cfg(tmp_path / "disabled", tmp_path / "features")
    cfg_none = _cfg(
        tmp_path / "none",
        tmp_path / "features",
        document_blacklist_path=None,
    )
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    provider.prepare([target])

    trials_disabled = TokenExampleSampler(cfg_disabled, provider=provider).build_trials(target)
    trials_none = TokenExampleSampler(cfg_none, provider=provider).build_trials(target)

    assert [t.trial_key for t in trials_disabled] == [t.trial_key for t in trials_none]


def test_document_blacklist_starves_target_gracefully_when_all_intruders_excluded(
    tmp_path: Path,
) -> None:
    target = _target()
    blacklist_path = tmp_path / "document_blacklist.txt"
    blacklist_path.write_text(
        "shard_00000_features:10\n"
        "shard_00000_features:11\n"
        "shard_00000_features:12\n"
        "shard_00000_features:13\n"
    )
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        document_blacklist_path=str(blacklist_path),
    )
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    provider.prepare([target])

    assert TokenExampleSampler(cfg, provider=provider).build_trials(target) == []


def test_document_blacklist_matches_by_shard_stem_and_doc_id_not_stable_key(
    tmp_path: Path,
) -> None:
    # The same (shard_stem, doc_id) recurs across different token windows
    # (different token_start/token_end / target_feature_ids), each of which
    # has a distinct TokenWindowRef.stable_key(). The blacklist must match on
    # the doc-level identity, not the window-level stable_key.
    target = _target()
    blacklist_path = tmp_path / "document_blacklist.txt"
    blacklist_path.write_text("shard_00000_features:5\n")
    cfg = _cfg(
        tmp_path,
        tmp_path / "features",
        document_blacklist_path=str(blacklist_path),
        n_positive_examples=2,
        n_quantiles=1,
        n_trials_per_target=1,
    )
    same_doc_window_a = _ref("a-", [0.0, 5.0, 0.0, 0.0], doc_id=5, token_start=0)
    same_doc_window_b = _ref("b-", [0.0, 6.0, 0.0, 0.0], doc_id=5, token_start=100)
    other_doc = _ref("c-", [0.0, 4.0, 0.0, 0.0], doc_id=6, token_start=0)
    assert same_doc_window_a.stable_key() != same_doc_window_b.stable_key()
    examples = TokenExampleSet(
        positives=(same_doc_window_a, same_doc_window_b, other_doc),
        non_activating=tuple(
            _ref(f"n{idx}-", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=20 + idx)
            for idx in range(3)
        ),
    )
    provider = _FakeTokenProvider({target.target_key: examples})
    provider.prepare([target])

    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)

    for trial in trials:
        assert all(ref.doc_id != 5 for ref in trial.positive_refs)


def test_runner_manifest_binds_blacklist_entries_and_rejects_edited_file(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    blacklist_path = tmp_path / "document_blacklist.txt"
    blacklist_path.write_text("shard_00000_features:1\n", encoding="utf-8")
    cfg = _cfg(
        tmp_path,
        feature_dir,
        document_blacklist_path=str(blacklist_path),
    )

    runner = AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, _RecordingIntruderJudge()),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    )
    first_digest = resolve_document_blacklist(str(blacklist_path)).sha256
    blacklist_path.write_text("shard_00000_features:2\n", encoding="utf-8")
    runner.run()

    manifest = json.loads(Path(cfg.runner_manifest_path).read_text(encoding="utf-8"))
    assert manifest["document_blacklist_path"] == str(blacklist_path)
    assert manifest["document_blacklist_sha256"] == first_digest

    assert resolve_document_blacklist(str(blacklist_path)).sha256 != first_digest

    with pytest.raises(
        RuntimeError,
        match="Existing autointerp manifest .* does not match",
    ):
        AutoInterpFilterRunner(
            cfg,
            judge=cast(IntruderDetectionJudge, _RecordingIntruderJudge()),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()


def test_token_example_cache_roundtrip_and_rejects_stale_schema(tmp_path: Path) -> None:
    target = _target()
    examples = _example_set(target)
    cache_path = tmp_path / "token_cache.sqlite"

    with TokenExampleCache(cache_path) as cache:
        assert cache.get(namespace="ns", target_key=target.target_key) is None
        cache.set(namespace="ns", target_key=target.target_key, examples=examples)
        assert cache.get(namespace="ns", target_key=target.target_key) == examples
        assert cache.get(namespace="other", target_key=target.target_key) is None

    stale_path = tmp_path / "stale.sqlite"
    conn = sqlite3.connect(stale_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()
    with pytest.raises(AutoInterpCacheError):
        TokenExampleCache(stale_path)


def test_trial_cache_separates_judge_keys_and_rejects_errors(tmp_path: Path) -> None:
    result_a = IntruderTrialResult(
        trial_key="trial",
        prediction=0,
        correct_index=0,
        correct=True,
        response="[RESPONSE]: 0",
        cached=False,
    )
    result_b = IntruderTrialResult(
        trial_key="trial",
        prediction=1,
        correct_index=1,
        correct=True,
        response="[RESPONSE]: 1",
        cached=False,
    )
    cache_path = tmp_path / "cache.sqlite"
    with IntruderTrialCache(cache_path) as cache:
        cache.set(target_key="feature:0", judge_key="judge-a", result=result_a)
        cache.set(target_key="feature:0", judge_key="judge-b", result=result_b)
        assert cache.get(
            trial_key="trial",
            target_key="feature:0",
            judge_key="judge-a",
        ) == IntruderTrialResult(
            trial_key="trial",
            prediction=0,
            correct_index=0,
            correct=True,
            response="[RESPONSE]: 0",
            cached=True,
        )
        assert cache.get(
            trial_key="trial",
            target_key="feature:0",
            judge_key="judge-b",
        ) == IntruderTrialResult(
            trial_key="trial",
            prediction=1,
            correct_index=1,
            correct=True,
            response="[RESPONSE]: 1",
            cached=True,
        )
        poisoned = IntruderTrialResult(
            trial_key="bad",
            prediction=None,
            correct_index=0,
            correct=False,
            response=None,
            cached=False,
            error_message="boom",
        )
        cache.set(target_key="feature:0", judge_key="judge-a", result=poisoned)
        with pytest.raises(AutoInterpCacheError, match="error_message"):
            cache.get(trial_key="bad", target_key="feature:0", judge_key="judge-a")


def test_cached_provider_builds_token_windows_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    cfg = _cfg(
        tmp_path,
        feature_dir,
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
    )
    target = ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(0,),
        target_key="feature:0",
        display_name="feature_0",
    )
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=feature_dir,
        n_sae_features=3,
    )
    provider.prepare([target])

    def fake_doc_activations(doc_ref, feature_ids=None):
        token_ids = tuple(range(8))
        acts = t.zeros(8, 3)
        if doc_ref.doc_id == 0:
            acts[2, 0] = 5.0
            acts[3, 0] = 2.0
        elif doc_ref.doc_id == 1:
            acts[5, 0] = 4.0
        else:
            acts[1, 1] = 3.0
        return type(
            "Doc",
            (),
            {
                "source_text_hash": f"hash-{doc_ref.doc_id}",
                "token_offset": 0,
                "token_ids": token_ids,
                "token_strings": tuple(f"d{doc_ref.doc_id}t{i}" for i in range(8)),
                "sae_feature_ids": (0, 1, 2),
                "sae_acts": acts,
            },
        )()

    monkeypatch.setattr(provider, "_doc_activations", fake_doc_activations)

    examples = provider.examples_for_target(target)

    assert len(examples.positives) >= 2
    assert len(examples.non_activating) >= 1
    assert all(ref.target_feature_ids == (0,) for ref in examples.positives)
    assert all(ref.max_activation > 0 for ref in examples.positives)
    assert all(ref.max_activation == 0 for ref in examples.non_activating)

    monkeypatch.setattr(
        provider,
        "_build_examples_for_target",
        lambda target: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert provider.examples_for_target(target) == examples


def test_source_index_provider_serves_token_cache_hit_without_loading_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    target = _target((0,))
    cfg = _cfg(
        tmp_path,
        feature_dir,
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)

    def _unexpected_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("token cache hit should not load the full source index")

    monkeypatch.setattr(token_examples_module, "load_or_build_source_index", _unexpected_load)
    provider.prepare([target])
    examples = _example_set(target)
    with TokenExampleCache(Path(cfg.token_example_cache_path)) as cache:
        cache.set(
            namespace=provider._cache_namespace(target),
            target_key=target.target_key,
            examples=examples,
        )

    assert provider.examples_for_target(target) == examples
    assert not Path(cfg.source_index.path).exists()
    assert not Path(cfg.source_index.manifest_path).exists()


def test_source_index_provider_loads_index_on_token_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    target = _target((0,))
    cfg = _cfg(
        tmp_path,
        feature_dir,
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider.prepare([target])
    assert not provider._source_index_loaded

    load_calls = 0
    real_load = token_examples_module.load_or_build_source_index

    def _counting_load(*args: Any, **kwargs: Any):
        nonlocal load_calls
        load_calls += 1
        return real_load(*args, **kwargs)

    def fake_doc_activations(doc_ref, feature_ids=None):
        acts = t.zeros(8, 3)
        if doc_ref.doc_id in {0, 1}:
            acts[doc_ref.doc_id + 1, 0] = 3.0
        return type(
            "Doc",
            (),
            {
                "source_text_hash": f"hash-{doc_ref.doc_id}",
                "token_offset": 0,
                "token_ids": tuple(range(8)),
                "token_strings": tuple(f"d{doc_ref.doc_id}t{i}" for i in range(8)),
                "sae_feature_ids": (0, 1, 2),
                "sae_acts": acts,
            },
        )()

    monkeypatch.setattr(token_examples_module, "load_or_build_source_index", _counting_load)
    monkeypatch.setattr(provider, "_doc_activations", fake_doc_activations)

    examples = provider.examples_for_target(target)

    assert load_calls == 1
    assert provider._source_index_loaded
    assert len(examples.positives) == 2
    assert len(examples.non_activating) >= 2
    assert all(ref.max_activation == 0.0 for ref in examples.non_activating)
    assert Path(cfg.source_index.path).is_file()
    assert Path(cfg.source_index.manifest_path).is_file()


def test_source_index_provider_validates_stale_manifest_before_token_cache_hit(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    target = _target((0,))
    cfg = _cfg(
        tmp_path,
        feature_dir,
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider.prepare([target])
    provider._ensure_source_index_loaded()
    examples = _example_set(target)
    with TokenExampleCache(Path(cfg.token_example_cache_path)) as cache:
        cache.set(
            namespace=provider._cache_namespace(target),
            target_key=target.target_key,
            examples=examples,
        )
    stale_manifest = json.loads(Path(cfg.source_index.manifest_path).read_text())
    stale_manifest["seed"] = stale_manifest["seed"] + 1
    Path(cfg.source_index.manifest_path).write_text(
        json.dumps(stale_manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(AutoInterpCacheError, match="does not match"):
        CachedTokenExampleProvider(
            cfg,
            feature_dir=feature_dir,
            n_sae_features=3,
        ).prepare([target])


def test_source_index_provider_validates_stale_index_metadata_before_token_cache_hit(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    target = _target((0,))
    cfg = _cfg(
        tmp_path,
        feature_dir,
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider.prepare([target])
    provider._ensure_source_index_loaded()
    examples = _example_set(target)
    with TokenExampleCache(Path(cfg.token_example_cache_path)) as cache:
        cache.set(
            namespace=provider._cache_namespace(target),
            target_key=target.target_key,
            examples=examples,
        )
    conn = sqlite3.connect(cfg.source_index.path)
    try:
        conn.execute("UPDATE metadata SET value = '0' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AutoInterpCacheError, match="schema_version=0"):
        CachedTokenExampleProvider(
            cfg,
            feature_dir=feature_dir,
            n_sae_features=3,
        ).prepare([target])


def test_doc_activation_cache_is_segment_focused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    cfg = _cfg(tmp_path, feature_dir)
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider_any = cast(Any, provider)
    provider_any._sae = t.nn.Linear(1, 1)
    provider_any._model = object()
    provider._tokenizer = type(
        "Tokenizer",
        (),
        {"decode": lambda self, ids: f"tok{int(ids[0])}"},
    )()
    provider._context_window = 8
    provider._sae_max_context = 8
    provider._activation_feature_ids = (0,)
    provider._layer = 0
    provider._hook_name = "blocks.0.hook_resid_post"
    monkeypatch.setattr(provider, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(provider, "_full_text", lambda doc_ref: "same document")

    def fake_truncate(text, tokenizer, context_window, sae_max_context, focus_token_end):
        assert text == "same document"
        assert context_window == 8
        assert sae_max_context == 8
        end = int(focus_token_end)
        return ([end - 2, end - 1], 16, end - 2)

    calls: list[tuple[int, ...]] = []

    def fake_target_sae_activations(**kwargs):
        token_ids = tuple(int(x) for x in kwargs["tokens"].tolist())
        calls.append(token_ids)
        return t.tensor([[float(token_ids[0])], [float(token_ids[1])]])

    monkeypatch.setattr(
        token_examples_module,
        "_truncate_to_focused_window",
        fake_truncate,
    )
    monkeypatch.setattr(
        token_examples_module,
        "_target_sae_activations",
        fake_target_sae_activations,
    )

    ref_a = SourceDocRef("shard_00000", 0, 0, 0, 4)
    ref_b = SourceDocRef("shard_00000", 0, 0, 4, 8)

    doc_a = provider._doc_activations(ref_a)
    doc_b = provider._doc_activations(ref_b)
    doc_a_again = provider._doc_activations(ref_a)

    assert doc_a.token_ids == (2, 3)
    assert doc_b.token_ids == (6, 7)
    assert doc_a.token_offset == 2
    assert doc_b.token_offset == 6
    assert doc_a_again is doc_a
    assert calls == [(2, 3), (6, 7)]


def test_token_refs_use_absolute_source_token_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    cfg = _cfg(tmp_path, feature_dir)
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    doc = type(
        "Doc",
        (),
        {
            "source_text_hash": "hash-0",
            "token_offset": 128,
            "token_ids": tuple(range(32)),
            "token_strings": tuple(f"t{i}" for i in range(32)),
            "sae_feature_ids": (0, 1, 2),
            "sae_acts": t.zeros(32, 3),
        },
    )()
    doc.sae_acts[5, 0] = 9.0
    monkeypatch.setattr(provider, "_doc_activations", lambda doc_ref, feature_ids=None: doc)
    target = ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(0,),
        target_key="feature:0",
        display_name="feature_0",
    )

    refs = provider._positive_windows_for_doc(
        SourceDocRef("shard_00000", 0, 0, 128, 160),
        target,
    )

    assert len(refs) == 1
    assert refs[0].token_start == 130
    assert refs[0].token_end == 134
    assert refs[0].token_strings == ("t2", "t3", "t4", "t5")


def test_ci_doc_activations_injects_and_drops_bos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    cfg = _cfg(tmp_path, feature_dir)
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider_any = cast(Any, provider)
    provider_any._sae = t.nn.Linear(1, 1)
    provider_any._model = object()
    provider._tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_token_id": 0,
            "decode": lambda self, ids: f"tok{int(ids[0])}",
        },
    )()
    provider._context_window = 8
    provider._sae_max_context = 8
    provider._layer = 0
    provider._hook_name = "blocks.0.hook_resid_post"
    provider._context_independent_features = True
    provider._ignore_bos_features = True
    provider._ignore_bos_token_id = 0
    monkeypatch.setattr(provider, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(provider, "_full_text", lambda doc_ref: "same document")
    monkeypatch.setattr(
        token_examples_module,
        "_truncate_to_training_window",
        lambda text, tokenizer, context_window, sae_max_context: (
            [0, 10, 11, 12, 0, 13],
            6,
        ),
    )
    calls: list[tuple[int, ...]] = []

    def fake_target_sae_activations(**kwargs):
        token_ids = tuple(int(x) for x in kwargs["tokens"].tolist())
        calls.append(token_ids)
        if token_ids == (0, 10, 11, 12):
            return t.tensor([[99.0], [1.0], [2.0], [0.0]])
        if token_ids == (0, 10, 11):
            return t.tensor([[5.0], [6.0], [7.0]])
        raise AssertionError(token_ids)

    monkeypatch.setattr(
        token_examples_module,
        "_target_sae_activations",
        fake_target_sae_activations,
    )

    injected = provider._doc_activations(
        SourceDocRef("shard_00000", 0, 0, 1, 4),
        (0,),
    )
    natural = provider._doc_activations(
        SourceDocRef("shard_00000", 0, 0, 0, 3),
        (0,),
    )

    assert calls == [(0, 10, 11, 12), (0, 10, 11)]
    assert injected.token_offset == 1
    assert injected.token_ids == (10, 11, 12)
    assert injected.sae_acts[:, 0].tolist() == [1.0, 2.0, 0.0]
    assert natural.token_offset == 0
    assert natural.token_ids == (0, 10, 11)
    assert natural.sae_acts[:, 0].tolist() == [0.0, 6.0, 7.0]


def test_ci_reforward_surface_clamps_synthesized_tail_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    cfg = _cfg(tmp_path, feature_dir)
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
    provider._tokenizer = type("Tokenizer", (), {"bos_token_id": 0})()
    provider._context_window = 8
    provider._sae_max_context = 8
    provider._context_independent_features = True
    provider._ignore_bos_token_id = 0
    monkeypatch.setattr(
        token_examples_module,
        "_truncate_to_training_window",
        lambda text, tokenizer, context_window, sae_max_context: (
            [0, 10, 11, 12, 13],
            5,
        ),
    )

    model_input, real_tokens, token_offset = provider._reforward_surface(
        "doc",
        SourceDocRef("shard_00000", 0, 0, 4, 8),
    )

    assert model_input == [0, 13]
    assert real_tokens == [13]
    assert token_offset == 4


def test_token_example_helpers_fail_loudly_on_invalid_source_identity() -> None:
    with pytest.raises(AutoInterpDataError, match="does not contain a shard index"):
        token_examples_module._shard_index_from_stem("features_without_number")

    def tokenizer(text, padding=False):
        return {"input_ids": [1, 2, 3]}

    with pytest.raises(AutoInterpDataError, match="requires a focused segment token end"):
        token_examples_module._truncate_to_focused_window(
            "doc",
            tokenizer,
            context_window=4,
            sae_max_context=4,
            focus_token_end=cast(Any, None),
        )
    with pytest.raises(AutoInterpDataError, match="empty source text"):
        token_examples_module._truncate_to_focused_window(
            "",
            tokenizer,
            context_window=4,
            sae_max_context=4,
            focus_token_end=1,
        )


def test_token_example_scan_derives_focus_from_legacy_segment_indices(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=8,
    )
    data = {
        TensorKey.OFFSETS: t.tensor([0, 1, 2], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0, 1], dtype=t.int32),
        TensorKey.FEATURE_MAGS: t.tensor([1.0, 2.0], dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor([0, 0], dtype=t.int32),
        TensorKey.SEGMENT_INDICES: t.tensor([0, 2], dtype=t.int32),
        "segment_size": 4,
        "segment_mode": "fixed_tokens",
    }

    provider._scan_shard(Path("shard_00000.safetensors"), data)

    assert provider._intruder_docs == [
        SourceDocRef("shard_00000", 0, 0, 0, 4),
        SourceDocRef("shard_00000", 0, 0, 8, 12),
    ]


def test_token_example_scan_prefers_explicit_segment_token_bounds(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=8,
    )
    data = {
        TensorKey.OFFSETS: t.tensor([0, 1], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0], dtype=t.int32),
        TensorKey.FEATURE_MAGS: t.tensor([1.0], dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor([0], dtype=t.int32),
        TensorKey.SEGMENT_INDICES: t.tensor([99], dtype=t.int32),
        TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([5], dtype=t.int32),
        TensorKey.SEGMENT_TOKEN_ENDS: t.tensor([9], dtype=t.int32),
        "segment_size": 4,
        "segment_mode": "fixed_tokens",
    }

    provider._scan_shard(Path("shard_00000.safetensors"), data)

    assert provider._intruder_docs == [SourceDocRef("shard_00000", 0, 0, 5, 9)]


def test_token_example_scan_requires_derivable_segment_tokens(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=8,
    )
    data = {
        TensorKey.OFFSETS: t.tensor([0, 1], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0], dtype=t.int32),
        TensorKey.FEATURE_MAGS: t.tensor([1.0], dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor([0], dtype=t.int32),
        "segment_size": 4,
    }

    with pytest.raises(AutoInterpDataError, match="missing segment token provenance"):
        provider._scan_shard(Path("shard_00000.safetensors"), data)


def test_source_index_matches_small_target_scanner(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    shard_path = _write_feature_shard(
        feature_dir,
        data={
            TensorKey.OFFSETS: t.tensor([0, 3, 6, 9, 12, 15], dtype=t.int64),
            TensorKey.FEATURE_IDS: t.tensor(
                [0, 1, 2] * 5,
                dtype=t.int64,
            ),
            TensorKey.FEATURE_MAGS: t.tensor(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],
                dtype=t.float32,
            ),
            TensorKey.DOC_IDS: t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
            TensorKey.SEGMENT_INDICES: t.tensor([0, 0, 0, 0, 0], dtype=t.int64),
            TensorKey.SEGMENT_POSITIONS: t.tensor([0, 1, 2, 3, 4], dtype=t.int64),
            TensorKey.SEGMENT_CHAR_STARTS: t.tensor([0, 4, 8, 12, 16], dtype=t.int64),
            TensorKey.SEGMENT_CHAR_ENDS: t.tensor([4, 8, 12, 16, 20], dtype=t.int64),
            "n_docs": 5,
            "n_segments": 5,
            "layer": 0,
            "context_window": 64,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 16,
            "context_independent_features": "false",
            "ignore_bos_features": "true",
        },
    )
    data_path = _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3", "doc4"])
    targets = [_target((0,)), _target((1,)), _target((2,))]
    scan_cfg = _cfg(
        tmp_path / "scan",
        feature_dir,
        output_dir=str(tmp_path / "scan" / "out"),
        data_dir=str(data_dir),
        feature_ids=[0, 1, 2],
        n_sae_features=3,
        token_examples=TokenExamplesConfig(
            example_ctx_len=_positive(4),
            max_positive_source_docs_per_target=_positive(2),
            max_intruder_source_docs=_positive(8),
            max_windows_per_source_doc=_positive(4),
            max_positive_windows_per_target=_positive(16),
            max_non_activating_windows_per_target=_positive(16),
        ),
    )
    index_cfg = _cfg(
        tmp_path / "index",
        feature_dir,
        output_dir=str(tmp_path / "index" / "out"),
        data_dir=str(data_dir),
        feature_ids=[0, 1, 2],
        n_sae_features=3,
        token_examples=TokenExamplesConfig(
            example_ctx_len=_positive(4),
            max_positive_source_docs_per_target=_positive(2),
            max_intruder_source_docs=_positive(8),
            max_windows_per_source_doc=_positive(4),
            max_positive_windows_per_target=_positive(16),
            max_non_activating_windows_per_target=_positive(16),
        ),
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "index" / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "index" / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    scan_provider = CachedTokenExampleProvider(
        scan_cfg,
        feature_dir=feature_dir,
        n_sae_features=3,
    )
    indexed_provider = CachedTokenExampleProvider(
        index_cfg,
        feature_dir=feature_dir,
        n_sae_features=3,
    )

    scan_provider.prepare(targets)
    indexed_provider.prepare(targets)
    indexed_provider._ensure_source_index_loaded()

    request = SourceIndexRequest(
        feature_dir=feature_dir,
        data_dir=str(data_path.parent),
        index_path=Path(index_cfg.source_index.path),
        manifest_path=Path(index_cfg.source_index.manifest_path),
        shard_paths=(shard_path,),
        parquet_paths=(data_path,),
        target_feature_ids=(0, 1, 2),
        n_sae_features=3,
        max_positive_refs_per_feature=index_cfg.token_examples.max_positive_source_docs_per_target,
        max_intruder_refs=index_cfg.token_examples.max_intruder_source_docs,
        seed=index_cfg.seed,
        feature_fingerprint=fingerprint_paths((shard_path,)),
        data_fingerprint=fingerprint_paths((data_path,)),
        context_independent_features=False,
        ignore_bos_features=True,
        chunk_size=index_cfg.source_index.chunk_size,
        force_rebuild=False,
    )
    loaded_index = load_or_build_source_index(request)

    assert indexed_provider._intruder_docs == scan_provider._intruder_docs
    assert dict(indexed_provider._positive_docs) == dict(scan_provider._positive_docs)
    assert dict(indexed_provider._positive_doc_scores) == dict(scan_provider._positive_doc_scores)
    assert loaded_index.intruder_refs == indexed_provider._intruder_docs
    assert loaded_index.positives_by_feature == {
        fid: [
            (ref, indexed_provider._positive_doc_scores[f"feature:{fid}"][ref.stable_key()])
            for ref in sorted(
                indexed_provider._positive_docs[f"feature:{fid}"],
                key=lambda ref: (
                    -indexed_provider._positive_doc_scores[f"feature:{fid}"][ref.stable_key()],
                    ref.stable_key(),
                ),
            )
        ]
        for fid in (0, 1, 2)
    }
    conn = sqlite3.connect(str(request.index_path))
    try:
        rows = conn.execute(
            "SELECT feature_id, rank, magnitude, stable_key, shard_stem, shard_index, doc_id, segment_token_start, segment_token_end "
            "FROM positive_refs ORDER BY feature_id, rank"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (
            fid,
            rank,
            magnitude,
            ref.stable_key(),
            ref.shard_stem,
            ref.shard_index,
            ref.doc_id,
            ref.segment_token_start,
            ref.segment_token_end,
        )
        for fid in (0, 1, 2)
        for rank, (ref, magnitude) in enumerate(loaded_index.positives_by_feature[fid])
    ]
    manifest = json.loads(Path(index_cfg.source_index.manifest_path).read_text())
    assert manifest["context_independent_features"] is False
    assert manifest["ignore_bos_features"] is True
    with pytest.raises(AutoInterpCacheError, match="does not match"):
        CachedTokenExampleProvider(
            _cfg(
                tmp_path / "mismatch",
                feature_dir,
                output_dir=str(tmp_path / "mismatch" / "out"),
                data_dir=str(data_dir),
                feature_ids=[0],
                n_sae_features=3,
                source_index=SourceIndexConfig(
                    enabled=True,
                    path=index_cfg.source_index.path,
                    manifest_path=index_cfg.source_index.manifest_path,
                    chunk_size=_positive(2),
                ),
            ),
            feature_dir=feature_dir,
            n_sae_features=3,
        ).prepare([_target((0,))])


def test_cached_provider_shared_source_index_matches_local_index_and_scan(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    targets = [_target((0,)), _target((1,)), _target((2,))]
    token_examples_cfg = TokenExamplesConfig(
        example_ctx_len=_positive(4),
        max_positive_source_docs_per_target=_positive(2),
        max_intruder_source_docs=_positive(4),
        max_windows_per_source_doc=_positive(4),
        max_positive_windows_per_target=_positive(16),
        max_non_activating_windows_per_target=_positive(16),
    )
    scan_cfg = _cfg(
        tmp_path / "scan",
        feature_dir,
        output_dir=str(tmp_path / "scan" / "out"),
        data_dir=str(data_dir),
        feature_ids=[0, 1, 2],
        n_sae_features=3,
        token_examples=token_examples_cfg,
    )
    local_cfg = _cfg(
        tmp_path / "local",
        feature_dir,
        output_dir=str(tmp_path / "local" / "out"),
        data_dir=str(data_dir),
        feature_ids=[0, 1, 2],
        n_sae_features=3,
        token_examples=token_examples_cfg,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "local" / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "local" / "out" / "source_doc_index.manifest.json"),
            chunk_size=_positive(2),
        ),
    )
    shared_cfg = _cfg(
        tmp_path / "shared",
        feature_dir,
        output_dir=str(tmp_path / "shared" / "out"),
        data_dir=str(data_dir),
        feature_ids=[0, 1, 2],
        n_sae_features=3,
        token_examples=token_examples_cfg,
        source_index=SourceIndexConfig(
            enabled=True,
            path=str(tmp_path / "shared" / "out" / "source_doc_index.sqlite"),
            manifest_path=str(tmp_path / "shared" / "out" / "source_doc_index.manifest.json"),
            shared_cache_dir=str(tmp_path / "shared_cache"),
            chunk_size=_positive(2),
        ),
    )
    providers = [
        CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=3)
        for cfg in (scan_cfg, local_cfg, shared_cfg)
    ]
    for provider in providers:
        provider.prepare(targets)

    scan_provider, local_provider, shared_provider = providers
    for provider in (local_provider, shared_provider):
        provider._ensure_source_index_loaded()
    assert shared_provider._intruder_docs == local_provider._intruder_docs
    assert shared_provider._intruder_docs == scan_provider._intruder_docs
    assert dict(shared_provider._positive_docs) == dict(local_provider._positive_docs)
    assert dict(shared_provider._positive_docs) == dict(scan_provider._positive_docs)
    assert dict(shared_provider._positive_doc_scores) == dict(local_provider._positive_doc_scores)
    assert dict(shared_provider._positive_doc_scores) == dict(scan_provider._positive_doc_scores)
    assert not Path(shared_cfg.source_index.path).exists()
    assert not Path(shared_cfg.source_index.manifest_path).exists()
    assert len(list((tmp_path / "shared_cache").glob("source_doc_index_*.sqlite"))) == 1
    assert len(list((tmp_path / "shared_cache").glob("source_doc_index_*.manifest.json"))) == 1


def test_cached_provider_context_ignores_shared_source_index_cache_dir(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc zero text", "doc one text", "doc two text", "doc three text"])
    target = _target((0,))
    output_dir = tmp_path / "out"
    base_source_index = SourceIndexConfig(
        enabled=True,
        path=str(output_dir / "source_doc_index.sqlite"),
        manifest_path=str(output_dir / "source_doc_index.manifest.json"),
        shared_cache_dir=str(tmp_path / "shared_a"),
        chunk_size=_positive(2),
    )
    cfg_a = _cfg(
        tmp_path,
        feature_dir,
        output_dir=str(output_dir),
        data_dir=str(data_dir),
        feature_ids=[0],
        n_sae_features=3,
        source_index=base_source_index,
    )
    provider_a = CachedTokenExampleProvider(cfg_a, feature_dir=feature_dir, n_sae_features=3)
    provider_a.prepare([target])

    cfg_b = replace(
        cfg_a,
        source_index=replace(
            base_source_index,
            shared_cache_dir=str(tmp_path / "shared_b"),
        ),
    )
    provider_b = CachedTokenExampleProvider(cfg_b, feature_dir=feature_dir, n_sae_features=3)
    provider_b.prepare([target])

    assert provider_b.cache_context() == provider_a.cache_context()
    provider_a._ensure_source_index_loaded()
    provider_b._ensure_source_index_loaded()
    assert not Path(cfg_a.source_index.path).exists()
    assert list((tmp_path / "shared_a").glob("source_doc_index_*.sqlite"))
    assert list((tmp_path / "shared_b").glob("source_doc_index_*.sqlite"))


def test_cached_provider_rejects_shards_without_ignore_bos_features(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir, ignore_bos_features=False)
    _write_parquet(data_dir, ["doc0", "doc1", "doc2", "doc3"])
    provider = CachedTokenExampleProvider(
        _cfg(
            tmp_path,
            feature_dir,
            data_dir=str(data_dir),
            feature_ids=[0],
            n_sae_features=3,
        ),
        feature_dir=feature_dir,
        n_sae_features=3,
    )

    with pytest.raises(AutoInterpDataError, match="ignore_bos_features=true"):
        provider.prepare([_target((0,))])


def test_token_example_scan_rejects_unbounded_non_fixed_segments(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=8,
    )
    data = {
        TensorKey.OFFSETS: t.tensor([0, 1], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0], dtype=t.int32),
        TensorKey.FEATURE_MAGS: t.tensor([1.0], dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor([0], dtype=t.int32),
        TensorKey.SEGMENT_INDICES: t.tensor([0], dtype=t.int32),
        "segment_size": 4,
        "segment_mode": "turn_delimited_fixed_tokens",
    }

    with pytest.raises(AutoInterpDataError, match="has no explicit segment token provenance"):
        provider._scan_shard(Path("shard_00000.safetensors"), data)


def test_truncate_to_focused_window_keeps_focus_token_in_context() -> None:
    class Tokenizer:
        def __init__(self, ids):
            self.ids = list(ids)

        def __call__(self, text, padding=False):
            assert text == "doc"
            assert padding is False
            return {"input_ids": self.ids}

    assert token_examples_module._truncate_to_focused_window(
        "doc",
        Tokenizer(range(4)),
        context_window=8,
        sae_max_context=8,
        focus_token_end=3,
    ) == ([0, 1, 2, 3], 4, 0)
    assert token_examples_module._truncate_to_focused_window(
        "doc",
        Tokenizer(range(10)),
        context_window=4,
        sae_max_context=8,
        focus_token_end=7,
    ) == ([3, 4, 5, 6], 10, 3)
    assert token_examples_module._truncate_to_focused_window(
        "doc",
        Tokenizer(range(10)),
        context_window=8,
        sae_max_context=4,
        focus_token_end=99,
    ) == ([6, 7, 8, 9], 10, 6)


def test_intruder_highlight_sampler_fails_when_no_nonactivating_tokens(
    tmp_path: Path,
) -> None:
    target = _target()
    sampler = TokenExampleSampler(
        _cfg(tmp_path, tmp_path / "features"),
        provider=_FakeTokenProvider({target.target_key: _example_set(target)}),
    )
    active_ref = _ref("active-", [1.0, 2.0])

    with pytest.raises(AutoInterpDataError, match="no non-activating tokens"):
        sampler._sample_intruder_highlights(active_ref, n_highlights=1)


def test_token_provider_native_backend_selection_is_loud(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=3,
    )

    assert (
        provider._should_use_native_backend(
            backend="auto",
            model_name="gpt2",
            hook_name="blocks.0.hook_resid_pre",
            device="cuda",
        )
        is False
    )
    assert (
        provider._should_use_native_backend(
            backend="auto",
            model_name="gemma-3-4b-pt",
            hook_name="blocks.22.hook_resid_post",
            device="cuda",
        )
        is False
    )
    assert (
        provider._should_use_native_backend(
            backend="native",
            model_name="gemma-3-4b-pt",
            hook_name="blocks.22.hook_resid_post",
            device="cuda",
        )
        is True
    )
    with pytest.raises(AutoInterpDataError, match="requires a CUDA Gemma residual-post SAE"):
        provider._should_use_native_backend(
            backend="native",
            model_name="gpt2",
            hook_name="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_release_model_clears_model_tokenizer_and_batch_execution_state(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=3,
    )
    cast(Any, provider)._model = object()
    cast(Any, provider)._native_model = object()
    cast(Any, provider)._sae = object()
    provider._tokenizer = object()
    provider._ignore_bos_token_id = 2
    provider._activation_backend = "native"
    provider._batch_verified = True
    provider._batch_shadow_counter = 17
    provider._batch_companion_token_strings = {0: "zero", 2: "bos"}
    previous_thread_local = provider._batch_gpu_thread_local
    provider._batch_gpu_streams_ready = True

    provider.release_model()

    assert provider._model is None
    assert provider._native_model is None
    assert provider._sae is None
    assert provider._tokenizer is None
    assert provider._ignore_bos_token_id is None
    assert provider._activation_backend == "unloaded"
    assert provider._batch_verified is False
    assert provider._batch_shadow_counter == 0
    assert provider._batch_companion_token_strings == {}
    assert provider._batch_gpu_streams_ready is False
    assert provider._batch_gpu_thread_local is not previous_thread_local


def test_release_model_is_a_safe_no_op_before_any_load(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, feature_dir),
        feature_dir=feature_dir,
        n_sae_features=3,
    )

    provider.release_model()

    assert provider._model is None
    assert provider._native_model is None
    assert provider._sae is None
    assert provider._tokenizer is None
    assert provider._ignore_bos_token_id is None
    assert provider._activation_backend == "unloaded"
    assert provider._batch_verified is False
    assert provider._batch_shadow_counter == 0
    assert provider._batch_companion_token_strings == {}


def test_score_targets_releases_model_before_judge_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    _write_feature_shard(feature_dir)
    _write_parquet(data_dir, ["doc0"])
    cfg = _cfg(tmp_path, feature_dir, data_dir=str(data_dir))
    assert cfg.n_sae_features is not None
    provider = CachedTokenExampleProvider(
        cfg, feature_dir=feature_dir, n_sae_features=cfg.n_sae_features
    )
    provider.prepare([_target()])
    released: list[bool] = []
    monkeypatch.setattr(provider, "release_model", lambda: released.append(True))
    runner = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    )

    runner._score_targets(provider, [])

    assert released == [True]


def test_score_targets_does_not_require_release_model_on_fake_providers(
    tmp_path: Path,
) -> None:
    # _FakeTokenProvider has no release_model method; the runner must not
    # require one from arbitrary TokenExampleProvider implementations.
    target = _target()
    cfg = _cfg(tmp_path, tmp_path / "features")
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    provider.prepare([target])
    runner = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    )

    scores, _audit_rows = runner._score_targets(provider, [target])

    assert len(scores) == 1


def test_token_provider_truncates_transformerlens_load_to_target_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    cfg = _cfg(
        tmp_path,
        feature_dir,
        token_examples=TokenExamplesConfig(
            model="gemma-2-2b",
            sae_release="gemmascope",
            activation_backend="transformerlens",
        ),
    )
    provider = CachedTokenExampleProvider(cfg, feature_dir=feature_dir, n_sae_features=16)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        token_examples_module,
        "parse_feature_dir",
        lambda _feature_dir: type(
            "Parsed",
            (),
            {
                "model": "gemma-2-2b",
                "layer": 3,
                "activation_type": ActivationType.residual,
                "width": SAEWidth.W_16K,
            },
        )(),
    )
    monkeypatch.setattr(token_examples_module, "find_statistics_path", lambda _path: None)
    monkeypatch.setattr(
        token_examples_module,
        "resolve_sae",
        lambda **_kwargs: type(
            "Resolved",
            (),
            {
                "release": "release",
                "sae_id": "sae-id",
                "hook_name": "blocks.3.hook_resid_post",
            },
        )(),
    )

    def fake_load_transformerlens_token_example_model(**kwargs: Any) -> tuple[Any, Any, Any]:
        captured.update(kwargs)
        model = type(
            "Model",
            (),
            {
                "tokenizer": object(),
                "cfg": type("Cfg", (), {"n_ctx": 64})(),
            },
        )()
        return model, object(), model.tokenizer

    monkeypatch.setattr(
        token_examples_module,
        "_load_transformerlens_token_example_model",
        fake_load_transformerlens_token_example_model,
    )

    provider._ensure_model_loaded()

    assert captured["target_layer"] == 3
    assert provider._activation_backend == "transformerlens"

    missing_bos_provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=feature_dir,
        n_sae_features=16,
    )
    missing_bos_provider._context_independent_features = True
    missing_bos_provider._ignore_bos_features = True
    monkeypatch.setattr(token_examples_module, "resolve_bos_token_id", lambda _tokenizer: None)
    with pytest.raises(AutoInterpDataError, match="require tokenizer to define a BOS token id"):
        missing_bos_provider._ensure_model_loaded()


def test_transformerlens_token_example_loader_truncates_to_target_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    model = type(
        "Model",
        (),
        {
            "tokenizer": object(),
        },
    )()

    class _FakeHookedTransformer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured.update(kwargs)
            return model

    class _FakeSae:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> object:
            captured["sae"] = kwargs
            return object()

    resolved = ResolvedSAE(
        release="release",
        sae_id="sae-id",
        hook_name="blocks.3.hook_resid_post",
    )
    monkeypatch.setattr(token_examples_module, "HookedTransformer", _FakeHookedTransformer)
    monkeypatch.setattr(token_examples_module, "SAE", _FakeSae)

    loaded_model, _sae, tokenizer = token_examples_module._load_transformerlens_token_example_model(
        model_name="gemma-2-2b",
        resolved=resolved,
        device="cpu",
        target_layer=3,
    )

    assert loaded_model is model
    assert tokenizer is model.tokenizer
    assert captured["args"] == ("gemma-2-2b",)
    assert captured["device"] == "cpu"
    assert captured["first_n_layers"] == 4
    assert captured["sae"] == {
        "release": "release",
        "sae_id": "sae-id",
        "device": "cpu",
    }


def test_build_intruder_messages_uses_delphi_prompt_shape() -> None:
    target = _target()
    trial = _trial(target)
    examples = render_trial_examples(trial, TokenWindowRenderer())

    messages = build_intruder_messages(
        examples,
        cot=True,
        prompt_variant=IntruderPromptVariant.delphi.value,
        few_shot_set=IntruderFewShotSet.delphi.value,
    )
    final_prompt = messages[-1]["content"]
    example_lines = [line for line in final_prompt.splitlines() if line.startswith("Example ")]

    assert messages[0]["content"].startswith(
        'You are an intelligent and meticulous linguistics researcher doing a "intruder detection"'
    )
    assert "Getty Images" in messages[1]["content"]
    assert all("<<" not in messages[idx]["content"] for idx in (1, 3, 5))
    assert "because it is about videos and ads.[RESPONSE]: 2" in messages[2]["content"]
    assert len(example_lines) == 5
    assert all("<<" in line and ">>" in line for line in example_lines)
    assert final_prompt.startswith("Examples:\n\nExample 0:")
    assert "You may think briefly" not in final_prompt


def test_prompt_library_builds_calibrated_highlighted_messages() -> None:
    assert "delphi" in available_prompt_variants()
    assert "calibrated_v1" in available_prompt_variants()
    assert "highlight_context_v1" in available_prompt_variants()
    assert "highlight_context_token_first_v1" in available_prompt_variants()
    assert "delphi" in available_few_shot_sets()
    assert "sae_highlighted_v1" in available_few_shot_sets()
    assert "sae_balanced_v1" in available_few_shot_sets()
    assert "sae_realistic_v1" in available_few_shot_sets()

    spec = resolve_intruder_prompt_spec(
        prompt_variant=IntruderPromptVariant.calibrated_v1.value,
        few_shot_set=IntruderFewShotSet.sae_highlighted_v1.value,
    )
    cache_payload = spec.cache_payload()
    assert cache_payload["variant"] == "calibrated_v1"
    assert cache_payload["few_shot_set"] == "sae_highlighted_v1"
    assert cache_payload["version"].startswith("calibrated_v1:sae_highlighted_v1:v1:")
    assert len(cache_payload["version"].rsplit(":", maxsplit=1)[-1]) == 16

    messages = build_intruder_messages_from_texts(
        [
            "alpha <<one>>",
            "beta <<two>>",
            "gamma <<three>>",
            "delta <<four>>",
        ],
        prompt_variant="calibrated_v1",
        few_shot_set="sae_highlighted_v1",
    )

    assert "sparse-autoencoder feature examples" in messages[0]["content"]
    assert any("<<bytecode>>" in message["content"] for message in messages)
    assert any("<<oper>>" in message["content"] for message in messages)
    assert messages[-1]["content"].startswith("Examples:\n\nExample 0:")


def test_balanced_few_shot_set_uses_varied_answer_positions() -> None:
    spec = resolve_intruder_prompt_spec(
        prompt_variant=IntruderPromptVariant.highlight_context_v1.value,
        few_shot_set=IntruderFewShotSet.sae_balanced_v1.value,
    )
    assistant_answers = [
        message["content"] for message in spec.few_shot_messages if message["role"] == "assistant"
    ]

    assert assistant_answers == [
        "[RESPONSE]: 1",
        "[RESPONSE]: 3",
        "[RESPONSE]: 0",
        "[RESPONSE]: 2",
    ]
    assert "highlighted span and its nearby context first" in spec.system_prompt


def test_realistic_few_shot_set_uses_varied_answer_positions() -> None:
    spec = resolve_intruder_prompt_spec(
        prompt_variant=IntruderPromptVariant.highlight_context_v1.value,
        few_shot_set=IntruderFewShotSet.sae_realistic_v1.value,
    )
    assistant_answers = [
        message["content"] for message in spec.few_shot_messages if message["role"] == "assistant"
    ]

    assert assistant_answers == [
        "[RESPONSE]: 1",
        "[RESPONSE]: 3",
        "[RESPONSE]: 0",
        "[RESPONSE]: 2",
    ]
    assert any("<<to>>" in message["content"] for message in spec.few_shot_messages)


def test_token_first_prompt_prioritizes_short_highlights() -> None:
    spec = resolve_intruder_prompt_spec(
        prompt_variant=IntruderPromptVariant.highlight_context_token_first_v1.value,
        few_shot_set=IntruderFewShotSet.sae_realistic_v1.value,
    )

    assert "short words, punctuation, whitespace, or token fragments" in spec.system_prompt
    assert "broader document topic" in spec.system_prompt
    assert "Return only [RESPONSE]" in spec.system_prompt
    assert any("<<,>>" in message["content"] for message in spec.few_shot_messages)


def test_intruder_judge_defaults_use_calibrated_prompt() -> None:
    cfg = IntruderJudgeConfig(model="test-judge")

    assert cfg.prompt_variant == IntruderPromptVariant.highlight_context_token_first_v1.value
    assert cfg.few_shot_set == IntruderFewShotSet.sae_realistic_v1.value

    messages = build_intruder_messages_from_texts(
        ["alpha <<one>>", "beta <<two>>", "gamma <<three>>", "delta <<four>>"]
    )

    assert "short words, punctuation, whitespace, or token fragments" in messages[0]["content"]
    assert any("<<to>>" in message["content"] for message in messages)
    assert any("<<,>>" in message["content"] for message in messages)


def test_parse_intruder_response() -> None:
    assert parse_intruder_response("reason\n[RESPONSE]: 4", n_examples=5) == 4
    with pytest.raises(IntruderResponseParseError):
        parse_intruder_response("[RESPONSE]: 5", n_examples=5)
    with pytest.raises(IntruderResponseParseError):
        parse_intruder_response("no bracketed answer", n_examples=5)


def test_judge_records_final_prompt_and_retries_parse_failures() -> None:
    target = _target()
    trial = _trial(target, intruder_index=3)
    judge = IntruderDetectionJudge(
        _FixedResponseClient("[RESPONSE]: 3"),
        IntruderJudgeConfig(model="test-judge"),
    )

    result = judge.evaluate(trial, renderer=TokenWindowRenderer())

    assert result.correct is True
    assert result.prediction == 3
    assert result.final_user_prompt is not None
    assert "Example 3" in result.final_user_prompt

    bad_judge = IntruderDetectionJudge(
        _FixedResponseClient("malformed full judge response"),
        IntruderJudgeConfig(model="test-judge", parse_max_retries=_non_negative(1)),
    )
    with pytest.raises(IntruderResponseParseError) as exc_info:
        bad_judge.evaluate(trial, renderer=TokenWindowRenderer())
    assert "attempt 1" in str(exc_info.value)
    assert "attempt 2" in str(exc_info.value)
    assert "malformed full judge response" in str(exc_info.value)


def test_judge_cache_key_changes_with_provider_context() -> None:
    cfg = IntruderJudgeConfig(model="test-judge")

    key_a = judge_cache_key(cfg, render_context={"target": [0, 1]})
    key_b = judge_cache_key(cfg, render_context={"target": [0, 2]})
    key_c = judge_cache_key(
        IntruderJudgeConfig(model="other-judge"),
        render_context={"target": [0, 1]},
    )

    assert key_a != key_b
    assert key_a != key_c


def test_judge_cache_key_ignores_runtime_concurrency() -> None:
    context = {"target": [0, 1]}

    assert judge_cache_key(
        IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(1)),
        render_context=context,
    ) == judge_cache_key(
        IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(8)),
        render_context=context,
    )


def test_judge_cache_key_changes_with_prompt_selection() -> None:
    context = {"target": [0, 1]}
    key_a = judge_cache_key(
        IntruderJudgeConfig(
            model="test-judge",
            prompt_variant="delphi",
            few_shot_set="delphi",
        ),
        render_context=context,
    )
    key_b = judge_cache_key(
        IntruderJudgeConfig(
            model="test-judge",
            prompt_variant="calibrated_v1",
            few_shot_set="delphi",
        ),
        render_context=context,
    )
    key_c = judge_cache_key(
        IntruderJudgeConfig(
            model="test-judge",
            prompt_variant="delphi",
            few_shot_set="sae_highlighted_v1",
        ),
        render_context=context,
    )

    assert key_a != key_b
    assert key_a != key_c
    assert key_b != key_c


def test_prompt_calibration_replays_trial_audit_yaml(tmp_path: Path) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    output_path = tmp_path / "calibration.yaml"
    json_output_path = tmp_path / "calibration.json"
    trial_audit_path.write_text(
        yaml.safe_dump(
            [
                {
                    "display_name": "feature_1",
                    "feature_ids": [1],
                    "quantile": 0,
                    "prediction": 0,
                    "correct_index": 1,
                    "response": "[RESPONSE]: 0",
                    "examples": [
                        {"text": "alpha <<one>>", "max_activation": 10.0},
                        {"text": "intruder <<two>>", "max_activation": 0.0},
                        {"text": "gamma <<three>>", "max_activation": 12.0},
                    ],
                }
            ],
            sort_keys=False,
        )
    )
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(output_path),
        json_output_path=str(json_output_path),
        judge=IntruderJudgeConfig(
            model="test-judge",
            prompt_variant="calibrated_v1",
            few_shot_set="sae_highlighted_v1",
        ),
    )

    report = run_prompt_calibration(cfg, _FixedResponseClient("[RESPONSE]: 1"))

    assert report.n_trials == 1
    assert report.n_correct == 1
    assert report.accuracy == 1.0
    yaml_payload = yaml.safe_load(output_path.read_text())
    json_payload = json.loads(json_output_path.read_text())
    assert yaml_payload["prompt_variant"] == "calibrated_v1"
    assert yaml_payload["few_shot_set"] == "sae_highlighted_v1"
    assert yaml_payload["rows"][0]["baseline_correct"] is False
    assert yaml_payload["rows"][0]["correct"] is True
    assert json_payload["accuracy"] == 1.0


def test_prompt_calibration_rejects_empty_trial_audit_yaml(tmp_path: Path) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    trial_audit_path.write_text("[]\n")
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(tmp_path / "calibration.yaml"),
        judge=IntruderJudgeConfig(model="test-judge"),
    )

    with pytest.raises(AutoInterpConfigError, match="at least one trial row"):
        run_prompt_calibration(cfg, _FixedResponseClient("[RESPONSE]: 0"))


def test_prompt_calibration_rejects_invalid_baseline_prediction(tmp_path: Path) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    trial_audit_path.write_text(
        yaml.safe_dump(
            [
                {
                    "prediction": 9,
                    "correct_index": 0,
                    "examples": [
                        {"text": "alpha <<one>>"},
                        {"text": "beta <<two>>"},
                    ],
                }
            ],
            sort_keys=False,
        )
    )
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(tmp_path / "calibration.yaml"),
        judge=IntruderJudgeConfig(model="test-judge"),
    )

    with pytest.raises(AutoInterpConfigError, match="prediction=9 out of range"):
        run_prompt_calibration(cfg, _FixedResponseClient("[RESPONSE]: 0"))


def test_prompt_calibration_rejects_boolean_correct_index(tmp_path: Path) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    trial_audit_path.write_text(
        yaml.safe_dump(
            [
                {
                    "prediction": 0,
                    "correct_index": True,
                    "examples": [
                        {"text": "alpha <<one>>"},
                        {"text": "beta <<two>>"},
                    ],
                }
            ],
            sort_keys=False,
        )
    )
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(tmp_path / "calibration.yaml"),
        judge=IntruderJudgeConfig(model="test-judge"),
    )

    with pytest.raises(AutoInterpConfigError, match="correct_index must be an int"):
        run_prompt_calibration(cfg, _FixedResponseClient("[RESPONSE]: 0"))


def test_prompt_calibration_fails_loudly_after_parse_exhaustion(tmp_path: Path) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    trial_audit_path.write_text(
        yaml.safe_dump(
            [
                {
                    "prediction": None,
                    "correct_index": 0,
                    "examples": [
                        {"text": "alpha <<one>>"},
                        {"text": "beta <<two>>"},
                    ],
                }
            ],
            sort_keys=False,
        )
    )
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(tmp_path / "calibration.yaml"),
        judge=IntruderJudgeConfig(model="test-judge", parse_max_retries=_non_negative(0)),
    )

    with pytest.raises(IntruderResponseParseError, match="full_response='not parseable'"):
        run_prompt_calibration(cfg, _FixedResponseClient("not parseable"))


def test_prompt_calibration_can_disable_progress_and_checkpoint_writes(
    tmp_path: Path,
) -> None:
    trial_audit_path = tmp_path / "trial_audit.yaml"
    rows = [
        {
            "prediction": None,
            "correct_index": 0,
            "examples": [
                {"text": f"alpha-{index} <<one>>"},
                {"text": f"beta-{index} <<two>>"},
            ],
        }
        for index in range(2)
    ]
    trial_audit_path.write_text(yaml.safe_dump(rows, sort_keys=False))
    output_path = tmp_path / "calibration.yaml"
    cfg = PromptCalibrationConfig(
        trial_audit_path=str(trial_audit_path),
        output_path=str(output_path),
        progress_interval=0,
        checkpoint_interval=0,
        judge=IntruderJudgeConfig(model="test-judge"),
    )

    report = run_prompt_calibration(cfg, _FixedResponseClient("[RESPONSE]: 0"))

    assert report.n_trials == 2
    assert report.n_correct == 2
    assert report.n_parse_errors == 0
    assert yaml.safe_load(output_path.read_text())["complete"] is True


def test_runner_writes_outputs_audit_yaml_and_reuses_cache(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(tmp_path, feature_dir)
    client = _IntruderFindingClient()

    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(client, cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_targets_total == 1
    assert report.n_targets_accepted == 1
    assert report.n_features_accepted == 1
    assert client.calls == cfg.n_trials_per_target
    assert Path(cfg.accepted_features_path).read_text().splitlines() == ["0"]

    row = json.loads(Path(cfg.target_scores_path).read_text().strip())
    assert row["score"] == 1.0
    assert row["schema_version"] == SCHEMA_VERSION
    ranked = yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())
    assert ranked["accepted_feature_ids"] == [0]
    assert "schema_version" not in ranked
    assert len(ranked["targets"]) == 1
    ranked_target = ranked["targets"][0]
    assert set(ranked_target) == {"description", "feature_ids", "score"}
    assert ranked_target["description"] is None
    assert ranked_target["feature_ids"] == [0]
    assert ranked_target["score"] == 1.0
    manual_lines = Path(cfg.manual_feature_filter_path).read_text().splitlines()
    assert manual_lines == ['# - "0"  # score=1.000']
    assert yaml.safe_load(Path(cfg.manual_feature_filter_path).read_text()) is None
    manual_meta = json.loads(Path(cfg.manual_feature_filter_meta_path).read_text())
    assert manual_meta["target_scores_path"] == cfg.target_scores_path
    assert manual_meta["n_targets"] == 1
    report_payload = json.loads(Path(cfg.report_output_path).read_text())
    assert report_payload["trial_audit_yaml_path"] == cfg.trial_audit_yaml_path
    assert report_payload["manual_feature_filter_path"] == cfg.manual_feature_filter_path
    manifest = json.loads(Path(cfg.runner_manifest_path).read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert "clustering_enabled" not in manifest
    assert all(target["kind"] == "feature" for target in manifest["targets"])
    assert all(len(target["feature_ids"]) == 1 for target in manifest["targets"])

    audit_json_rows = [
        json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
    ]
    audit_yaml_rows = yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text())
    first = audit_json_rows[0]
    assert first["final_user_prompt_sent_to_judge"].startswith("Examples:")
    assert len(first["examples"]) == cfg.n_positive_examples + 1
    assert [example["role"] for example in first["examples"]].count("intruder") == 1
    assert all(example["token_strings"] for example in first["examples"])
    assert all(example["highlighted_token_indices"] for example in first["examples"])
    first_yaml = audit_yaml_rows[0]
    assert first_yaml["final_user_prompt_sent_to_judge"] == first["final_user_prompt_sent_to_judge"]
    assert first_yaml["examples"] == [
        {
            "text": example["text"],
            "max_activation": example["max_activation"],
        }
        for example in first["examples"]
    ]
    assert set(first_yaml).isdisjoint(
        {
            "target_key",
            "kind",
            "trial_key",
            "trial_index",
            "intruder_index",
            "correct",
            "cached",
        }
    )
    assert all(set(example) == {"text", "max_activation"} for example in first_yaml["examples"])

    cached_report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    assert cached_report.n_features_accepted == 1
    cached_audit_rows = [
        json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
    ]
    assert len(cached_audit_rows) == cfg.n_trials_per_target
    assert all(row["cached"] is True for row in cached_audit_rows)
    assert all(
        row["final_user_prompt_sent_to_judge"].startswith("Examples:") for row in cached_audit_rows
    )
    assert yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text())


def test_runner_records_missing_neuronpedia_description_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        neuronpedia=NeuronpediaDescriptionConfig(
            enabled=True,
            model_id="gemma-3-4b",
            sae_source="22-gemmascope-2-res-65k",
        ),
    )
    calls: list[int] = []

    def fake_fetch_feature_description(
        model_id: str,
        sae_source: str,
        feature_id: int,
        *,
        config: object | None = None,
    ) -> str:
        assert model_id == "gemma-3-4b"
        assert sae_source == "22-gemmascope-2-res-65k"
        assert config is not None
        calls.append(feature_id)
        raise NeuronpediaFetchError("No explanations in response payload")

    monkeypatch.setattr(
        "steering.eval.autointerp_filter.runner.fetch_feature_description",
        fake_fetch_feature_description,
    )
    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert calls == [0]
    assert report.n_targets_scored == 1
    row = json.loads(Path(cfg.target_scores_path).read_text().strip())
    assert row["score"] == 1.0
    assert row["descriptions"] == {"0": None}
    ranked = yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())
    assert ranked["targets"][0]["description"] is None


def test_runner_async_judge_artifacts_match_serial_when_completions_reorder(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    targets = [_target((0,)), _target((1,))]
    provider = _FakeTokenProvider({target.target_key: _example_set(target) for target in targets})
    base_kwargs: dict[str, Any] = {
        "feature_ids": [0, 1],
        "n_trials_per_target": 4,
        "n_positive_examples": 2,
        "n_quantiles": 2,
        "audit_sample_limit": 20,
        "score_threshold": 0.5,
    }
    serial_cfg = _cfg(
        tmp_path / "serial",
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(1)),
        **base_kwargs,
    )
    async_cfg = _cfg(
        tmp_path / "async",
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(3)),
        **base_kwargs,
    )
    delays = {0: 0.04, 1: 0.0, 2: 0.02, 3: 0.0}
    wrong = {1}

    serial_judge = _RecordingIntruderJudge(delays=delays, wrong_trial_indices=wrong)
    AutoInterpFilterRunner(
        serial_cfg,
        judge=cast(IntruderDetectionJudge, serial_judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    async_judge = _RecordingIntruderJudge(
        expected_parallel=3,
        delays=delays,
        wrong_trial_indices=wrong,
    )
    AutoInterpFilterRunner(
        async_cfg,
        judge=cast(IntruderDetectionJudge, async_judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert async_judge.peak_concurrent == 3
    assert _artifact_snapshot(async_cfg) == _artifact_snapshot(serial_cfg)
    audit_rows = [
        json.loads(line) for line in Path(async_cfg.trial_audit_path).read_text().splitlines()
    ]
    assert [row["trial_index"] for row in audit_rows] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_runner_async_judge_reuses_warm_trial_cache_without_calls(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(4)),
        n_trials_per_target=4,
    )

    first_judge = _RecordingIntruderJudge(expected_parallel=4)
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, first_judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    assert first_judge.peak_concurrent == 4

    second_judge = _RecordingIntruderJudge(expected_parallel=2)
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, second_judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert second_judge.calls == []
    cached_audit_rows = [
        json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
    ]
    assert len(cached_audit_rows) == cfg.n_trials_per_target
    assert all(row["cached"] is True for row in cached_audit_rows)


def test_runner_async_mixed_cache_hits_only_spends_budget_on_misses(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(4)),
        n_trials_per_target=4,
        max_judge_calls=2,
    )
    provider.prepare([target])
    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        for index in (0, 2):
            trial = trials[index]
            cache.set(
                target_key=target.target_key,
                judge_key=cache_judge_key,
                result=IntruderTrialResult(
                    trial_key=trial.trial_key,
                    prediction=trial.intruder_index,
                    correct_index=trial.intruder_index,
                    correct=True,
                    response=f"[RESPONSE]: {trial.intruder_index}",
                    cached=False,
                ),
            )

    judge = _RecordingIntruderJudge(expected_parallel=2)
    report = AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_targets_scored == 1
    assert sorted(judge.calls) == [1, 3]
    audit_rows = [json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()]
    assert [row["trial_index"] for row in audit_rows] == [0, 1, 2, 3]
    assert [row["cached"] for row in audit_rows] == [True, False, True, False]


def test_runner_async_budget_failure_caches_only_admitted_prefix(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(4)),
        n_trials_per_target=4,
        max_judge_calls=2,
    )

    with pytest.raises(AutoInterpCostCapExceeded):
        AutoInterpFilterRunner(
            cfg,
            judge=cast(IntruderDetectionJudge, _RecordingIntruderJudge()),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()

    provider.prepare([target])
    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        cached = [
            cache.get(
                trial_key=trial.trial_key,
                target_key=target.target_key,
                judge_key=cache_judge_key,
            )
            for trial in trials
        ]
    assert [result is not None for result in cached] == [True, True, False, False]
    assert not Path(cfg.target_scores_path).exists()
    assert not Path(cfg.report_output_path).exists()


def test_runner_async_cache_error_after_admitted_miss_preserves_prefix_cache(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(4)),
        n_trials_per_target=3,
    )
    provider.prepare([target])
    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        cache.set(
            target_key=target.target_key,
            judge_key=cache_judge_key,
            result=IntruderTrialResult(
                trial_key=trials[1].trial_key,
                prediction=None,
                correct_index=trials[1].intruder_index,
                correct=False,
                response="not parseable",
                cached=False,
                error_message="poisoned cached error",
            ),
        )

    judge = _RecordingIntruderJudge()
    with pytest.raises(AutoInterpCacheError, match="poisoned cached error"):
        AutoInterpFilterRunner(
            cfg,
            judge=cast(IntruderDetectionJudge, judge),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()

    assert judge.calls == [0]
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        assert (
            cache.get(
                trial_key=trials[0].trial_key,
                target_key=target.target_key,
                judge_key=cache_judge_key,
            )
            is not None
        )
        with pytest.raises(AutoInterpCacheError, match="poisoned cached error"):
            cache.get(
                trial_key=trials[1].trial_key,
                target_key=target.target_key,
                judge_key=cache_judge_key,
            )
        assert (
            cache.get(
                trial_key=trials[2].trial_key,
                target_key=target.target_key,
                judge_key=cache_judge_key,
            )
            is None
        )
    assert not Path(cfg.target_scores_path).exists()
    assert not Path(cfg.report_output_path).exists()


def test_runner_async_parse_failure_caches_every_other_completed_trial(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(3)),
        n_trials_per_target=3,
    )
    judge = _RecordingIntruderJudge(
        expected_parallel=3,
        delays={0: 0.04, 1: 0.0, 2: 0.0},
        fail_trial_indices={1},
    )

    with pytest.raises(IntruderResponseParseError, match="trial 1"):
        AutoInterpFilterRunner(
            cfg,
            judge=cast(IntruderDetectionJudge, judge),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()

    assert judge.peak_concurrent == 3
    provider.prepare([target])
    trials = TokenExampleSampler(cfg, provider=provider).build_trials(target)
    cache_judge_key = judge_cache_key(cfg.judge, render_context=provider.cache_context())
    with IntruderTrialCache(Path(cfg.judge_cache_path)) as cache:
        cached = [
            cache.get(
                trial_key=trial.trial_key,
                target_key=target.target_key,
                judge_key=cache_judge_key,
            )
            for trial in trials
        ]
    assert [result is not None for result in cached] == [True, False, True]
    assert not Path(cfg.target_scores_path).exists()
    assert not Path(cfg.report_output_path).exists()


def test_judge_config_rejects_invalid_max_concurrent_at_construction() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_concurrent"):
        IntruderJudgeConfig(
            model="test-judge",
            max_concurrent=cast(Any, 0),
        )


def test_runner_accepts_legacy_unclustered_manifest(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(tmp_path, feature_dir)

    AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    manifest_path = Path(cfg.runner_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest["clustering_enabled"] = False
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1


@pytest.mark.parametrize(
    "legacy_shape",
    ["missing_both", "path_only", "null_digest"],
)
def test_runner_accepts_legacy_disabled_blacklist_manifest_without_rewriting(
    tmp_path: Path,
    legacy_shape: str,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(tmp_path, feature_dir)

    AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    manifest_path = Path(cfg.runner_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["document_blacklist_path"] is None
    assert manifest["document_blacklist_sha256"] == EMPTY_DOCUMENT_BLACKLIST_SHA256
    if legacy_shape == "missing_both":
        manifest.pop("document_blacklist_path")
        manifest.pop("document_blacklist_sha256")
    elif legacy_shape == "path_only":
        manifest.pop("document_blacklist_sha256")
    elif legacy_shape == "null_digest":
        manifest["document_blacklist_sha256"] = None
    else:  # pragma: no cover - parameter table is exhaustive
        raise AssertionError(legacy_shape)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    legacy_bytes = manifest_path.read_bytes()

    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1
    assert manifest_path.read_bytes() == legacy_bytes


@pytest.mark.parametrize(
    "manifest_shape",
    ["pre_blacklist", "enabled_path_without_digest"],
)
def test_runner_rejects_unbound_manifest_when_blacklist_is_enabled(
    tmp_path: Path,
    manifest_shape: str,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(tmp_path, feature_dir)

    AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    manifest_path = Path(cfg.runner_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blacklist_path = tmp_path / "document_blacklist.txt"
    blacklist_path.write_text("shard_00000_features:99\n", encoding="utf-8")
    if manifest_shape == "pre_blacklist":
        manifest.pop("document_blacklist_path")
        manifest.pop("document_blacklist_sha256")
    elif manifest_shape == "enabled_path_without_digest":
        manifest["document_blacklist_path"] = str(blacklist_path)
        manifest.pop("document_blacklist_sha256")
    else:  # pragma: no cover - parameter table is exhaustive
        raise AssertionError(manifest_shape)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    enabled_cfg = replace(cfg, document_blacklist_path=str(blacklist_path))

    with pytest.raises(
        RuntimeError,
        match="Existing autointerp manifest .* does not match",
    ):
        AutoInterpFilterRunner(
            enabled_cfg,
            judge=IntruderDetectionJudge(_ExplodingClient(), enabled_cfg.judge),
            token_example_provider=provider,
            renderer=_MarkerRenderer(),
        ).run()


def test_runner_accepts_legacy_manifest_missing_judge_max_concurrent(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    cfg = _cfg(tmp_path, feature_dir)

    AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    manifest_path = Path(cfg.runner_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    del manifest["judge"]["max_concurrent"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1


def test_runner_allows_manifest_rerun_with_different_judge_max_concurrent(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    serial_cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(1)),
    )

    AutoInterpFilterRunner(
        serial_cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), serial_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    concurrent_cfg = _cfg(
        tmp_path,
        feature_dir,
        judge=IntruderJudgeConfig(model="test-judge", max_concurrent=_positive(8)),
    )
    report = AutoInterpFilterRunner(
        concurrent_cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), concurrent_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1


def test_runner_allows_manifest_rerun_with_different_scan_workers(
    tmp_path: Path,
) -> None:
    # scan_workers is a scheduling knob; bumping it between runs must not trip
    # the runner-manifest compatibility check (it cannot change any output byte).
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    serial_cfg = _cfg(
        tmp_path,
        feature_dir,
        source_index=SourceIndexConfig(scan_workers=_positive(1)),
    )
    AutoInterpFilterRunner(
        serial_cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), serial_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    parallel_cfg = _cfg(
        tmp_path,
        feature_dir,
        source_index=SourceIndexConfig(scan_workers=_positive(4)),
    )
    report = AutoInterpFilterRunner(
        parallel_cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), parallel_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1


def test_runner_allows_manifest_rerun_with_different_shared_source_index_cache(
    tmp_path: Path,
) -> None:
    # shared_cache_dir is a storage-only source-index knob; bumping it between
    # otherwise identical reruns must not trip runner-manifest compatibility.
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    provider = _FakeTokenProvider({target.target_key: _example_set(target)})
    first_cfg = _cfg(
        tmp_path,
        feature_dir,
        source_index=SourceIndexConfig(shared_cache_dir=str(tmp_path / "cache_a")),
    )
    AutoInterpFilterRunner(
        first_cfg,
        judge=IntruderDetectionJudge(_IntruderFindingClient(), first_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()
    manifest_before = json.loads(Path(first_cfg.runner_manifest_path).read_text())
    assert manifest_before["source_index"]["shared_cache_dir"] == str(tmp_path / "cache_a")

    second_cfg = _cfg(
        tmp_path,
        feature_dir,
        source_index=SourceIndexConfig(shared_cache_dir=str(tmp_path / "cache_b")),
    )
    report = AutoInterpFilterRunner(
        second_cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), second_cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_features_accepted == 1
    manifest_after = json.loads(Path(second_cfg.runner_manifest_path).read_text())
    assert manifest_after == manifest_before


@pytest.mark.parametrize(
    "stale_manifest",
    [
        {"clustering_enabled": True},
        {"targets": [{"kind": "cluster", "feature_ids": [0, 1], "target_key": "cluster:0"}]},
    ],
)
def test_runner_rejects_removed_cluster_manifest(
    tmp_path: Path,
    stale_manifest: dict,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    cfg = _cfg(tmp_path, feature_dir)
    Path(cfg.runner_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.runner_manifest_path).write_text(json.dumps(stale_manifest))

    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        AutoInterpFilterRunner(
            cfg,
            judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
            token_example_provider=_FakeTokenProvider({target.target_key: _example_set(target)}),
            renderer=_MarkerRenderer(),
        ).run()


def test_ranked_yaml_targets_use_descriptions_instead_of_display_names() -> None:
    feature_score = TargetScore(
        target_key="feature:7",
        display_name="feature_7",
        kind=TargetKind.feature,
        feature_ids=(7,),
        score=1.0,
        n_correct=1,
        n_trials=1,
        outcome=TargetOutcome.accepted,
        descriptions={7: "calendar and date references"},
    )
    assert score_to_ranked_yaml(feature_score) == {
        "description": "calendar and date references",
        "feature_ids": [7],
        "score": 1.0,
    }


def test_runner_marks_insufficient_examples(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    cfg = _cfg(tmp_path, feature_dir)
    provider = _FakeTokenProvider(
        {
            target.target_key: TokenExampleSet(
                positives=(_ref("p", [1.0, 0.0], doc_id=0),),
                non_activating=(),
            )
        }
    )

    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(_ExplodingClient(), cfg.judge),
        token_example_provider=provider,
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_targets_scored == 0
    row = json.loads(Path(cfg.target_scores_path).read_text().strip())
    assert row["outcome"] == "insufficient_examples"
    assert "full token trials" in row["error_message"]
    ranked = yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())
    assert len(ranked["targets"]) == 1
    ranked_target = ranked["targets"][0]
    assert set(ranked_target) == {
        "description",
        "feature_ids",
        "score",
        "error_message",
    }
    assert ranked_target["description"] is None
    assert ranked_target["feature_ids"] == [0]
    assert ranked_target["score"] is None
    assert ranked_target["error_message"] == row["error_message"]


def test_runner_scores_partial_trials_above_fraction_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    cfg = _cfg(
        tmp_path,
        feature_dir,
        n_trials_per_target=4,
        min_trials_for_score=2,
        score_threshold=0.0,
    )
    trials = [
        replace(_trial(target, intruder_index=2), trial_index=idx, trial_key=f"partial-{idx}")
        for idx in range(2)
    ]

    class _PartialSampler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def build_trials(self, _target: ScoringTarget) -> list[IntruderTrial]:
            return list(trials)

    monkeypatch.setattr(
        "steering.eval.autointerp_filter.runner.TokenExampleSampler",
        _PartialSampler,
    )
    client = _IntruderFindingClient()
    report = AutoInterpFilterRunner(
        cfg,
        judge=IntruderDetectionJudge(client, cfg.judge),
        token_example_provider=_FakeTokenProvider({target.target_key: _example_set(target)}),
        renderer=_MarkerRenderer(),
    ).run()

    assert report.n_targets_scored == 1
    assert client.calls == 2
    row = json.loads(Path(cfg.target_scores_path).read_text().strip())
    assert row["score"] == 1.0
    assert row["n_trials"] == 2
    assert row["outcome"] == "accepted"
    assert row["error_message"] == "partial score from 2/4 requested token trials"
    ranked_target = yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())["targets"][0]
    assert ranked_target["error_message"] == row["error_message"]
    assert Path(cfg.manual_feature_filter_path).read_text().splitlines() == [
        '# - "0"  # score=1.000 trials=2/4 partial'
    ]


def test_runner_respects_uncached_judge_budget(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    target = _target()
    cfg = _cfg(tmp_path, feature_dir, max_judge_calls=1)

    with pytest.raises(AutoInterpCostCapExceeded):
        AutoInterpFilterRunner(
            cfg,
            judge=IntruderDetectionJudge(_IntruderFindingClient(), cfg.judge),
            token_example_provider=_FakeTokenProvider({target.target_key: _example_set(target)}),
            renderer=_MarkerRenderer(),
        ).run()


def test_config_validation_rejects_invalid_values(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    invalids = [
        (
            {"judge": IntruderJudgeConfig(prompt_variant="missing")},
            "prompt_variant",
        ),
        (
            {"judge": IntruderJudgeConfig(few_shot_set="missing")},
            "few_shot_set",
        ),
        (
            {"token_examples": TokenExamplesConfig(activation_backend="bad")},
            "activation_backend",
        ),
        (
            {"token_examples": TokenExamplesConfig(model_dtype="float16")},
            "model_dtype",
        ),
        (
            {"document_blacklist_path": "${output_dir}/blacklist.txt"},
            r"document_blacklist_path contains unresolved interpolation token",
        ),
    ]
    for overrides, pattern in invalids:
        with pytest.raises(AutoInterpConfigError, match=pattern):
            validate_config(_cfg(tmp_path, feature_dir, **overrides))


def test_build_config_rejects_old_renderer_field(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "output_dir": str(tmp_path / "out"),
            "renderer": {"kind": "segment"},
        }
    )

    with pytest.raises(AutoInterpConfigError, match="invalid fields"):
        build_config(cfg)


def test_build_config_rejects_removed_feature_clustering(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "output_dir": str(tmp_path / "out"),
            "feature_clustering": {"enabled": False},
        }
    )

    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        build_config(cfg)


def test_gemma3_4bpt_f65k_preset_uses_matching_neuronpedia_coords() -> None:
    config_dir = str((Path.cwd() / "configs" / "extraction").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = build_config(
            compose(config_name="precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos")
        )

    assert cfg.n_sae_features == 65536
    assert cfg.token_examples.model == "gemma-3-4b-pt"
    assert cfg.token_examples.device == "cuda"
    assert cfg.token_examples.activation_backend == "native"
    assert cfg.token_examples.model_dtype == "bfloat16"
    assert cfg.neuronpedia.enabled is True
    assert cfg.neuronpedia.model_id == "gemma-3-4b"
    assert cfg.neuronpedia.sae_source == "22-gemmascope-2-res-65k"


def test_autointerp_filter_base_config_keeps_reference_token_example_defaults() -> None:
    config_dir = str((Path.cwd() / "configs" / "extraction").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = build_config(compose(config_name="precompute_autointerp_filter"))

    assert cfg.token_examples.device == "cpu"
    assert cfg.token_examples.activation_backend == "auto"


def test_autointerp_filter_hydra_config_exposes_judge_max_concurrent() -> None:
    config_dir = str((Path.cwd() / "configs" / "extraction").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = build_config(
            compose(
                config_name="precompute_autointerp_filter",
                overrides=["judge.max_concurrent=4"],
            )
        )

    assert cfg.judge.max_concurrent == 4


def test_autointerp_filter_hydra_config_exposes_shared_source_index_cache() -> None:
    config_dir = str((Path.cwd() / "configs" / "extraction").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        default_cfg = build_config(compose(config_name="precompute_autointerp_filter"))
        override_cfg = build_config(
            compose(
                config_name="precompute_autointerp_filter",
                overrides=[
                    "output_dir=out/autointerp_filter/shared-cache-test",
                    "source_index.shared_cache_dir=${output_dir}/source_index_cache",
                ],
            )
        )

    assert default_cfg.source_index.shared_cache_dir is None
    assert (
        override_cfg.source_index.shared_cache_dir
        == "out/autointerp_filter/shared-cache-test/source_index_cache"
    )


def test_run_from_config_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "output_dir": str(tmp_path / "out"),
            "n_sae_features": 3,
            "feature_ids": [0],
            "n_trials_per_target": 1,
            "n_positive_examples": 1,
            "n_quantiles": 1,
            "fail_on_empty": False,
            "token_examples": {
                "example_ctx_len": 4,
                "max_positive_source_docs_per_target": 1,
                "max_intruder_source_docs": 1,
            },
        }
    )

    class FakeClient(LLMJudgeClient):
        def complete(self, messages, *, cache_key=None):
            return "[RESPONSE]: 0"

    def fake_run(self):
        return type(
            "Report",
            (),
            {
                "n_targets_total": 1,
                "n_targets_accepted": 0,
                "n_features_accepted": 0,
            },
        )()

    monkeypatch.setattr(
        "steering.eval.autointerp_filter.cli.OpenRouterLLMClient",
        lambda cfg: FakeClient(),
    )
    monkeypatch.setattr(
        "steering.eval.autointerp_filter.cli.AutoInterpFilterRunner.run",
        fake_run,
    )
    assert run_from_config(cfg) == 0

    bad_cfg = OmegaConf.create({"feature_dir": "", "output_dir": str(tmp_path / "out2")})
    assert run_from_config(bad_cfg) == 2


def test_module_help_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.preprocess.precompute_autointerp_filter",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "precompute_autointerp_filter" in result.stdout


def _trial(target: ScoringTarget, *, intruder_index: int = 2) -> IntruderTrial:
    positives = tuple(_ref(f"p{idx}", [0.0, 1.0 + idx, 0.0, 0.0], doc_id=idx) for idx in range(4))
    intruder = _ref("n", [0.0, 0.0, 0.0, 0.0], active=False, doc_id=99)
    return IntruderTrial(
        target=target,
        trial_index=0,
        positive_refs=positives,
        intruder_ref=intruder,
        intruder_index=intruder_index,
        quantile=0,
        intruder_highlight_indices=(1,),
        activation_threshold_ratio=0.3,
        trial_key="trial-key",
    )


def _write_parquet_rg(path: Path, texts: list[str], row_group_size: int) -> None:
    pq.write_table(pa.table({"text": texts}), path, row_group_size=row_group_size)


def _row_group_provider(tmp_path: Path, parquet: Path) -> CachedTokenExampleProvider:
    provider = CachedTokenExampleProvider(
        _cfg(tmp_path, tmp_path / "features"),
        feature_dir=tmp_path / "features",
        n_sae_features=6,
    )
    provider._parquets = [parquet]
    return provider


def _text_doc_ref(doc_id: int) -> SourceDocRef:
    return SourceDocRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=doc_id,
        segment_token_start=0,
        segment_token_end=1,
    )


@pytest.mark.parametrize(
    "row_group_size,n_docs",
    [(1, 6), (4, 12), (4, 13), (7, 50), (1024, 20)],
)
def test_full_text_row_group_reads_byte_identical_to_full_shard(
    tmp_path: Path, row_group_size: int, n_docs: int
) -> None:
    # The optimized per-row-group read must resolve every doc to the exact same
    # string as loading the whole shard column (uniform + non-uniform final
    # group, unicode, and varied lengths).
    texts = [f"doc{i}-éü" + "z" * (i % 5) for i in range(n_docs)]
    parquet = tmp_path / "shard_00000.parquet"
    _write_parquet_rg(parquet, texts, row_group_size)
    oracle = load_shard_text(parquet)
    assert oracle == texts  # oracle round-trips the written texts
    offsets = token_examples_module._shard_row_group_offsets(parquet)
    assert offsets[0] == 0
    assert offsets[-1] == n_docs

    provider = _row_group_provider(tmp_path, parquet)
    for doc_id in range(n_docs):
        assert provider._full_text(_text_doc_ref(doc_id)) == oracle[doc_id]
    with pytest.raises(AutoInterpDataError, match="out of range"):
        provider._full_text(_text_doc_ref(n_docs))
    with pytest.raises(BeartypeCallHintParamViolation, match="doc_id"):
        _text_doc_ref(-1)


def test_full_text_eviction_preserves_correctness_and_bounds_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tiny budget forces eviction; results stay byte-identical and the char
    # accounting stays exact (no leak between the cache and its size ledger).
    texts = [f"document-number-{i:03d}" for i in range(30)]
    parquet = tmp_path / "shard_00000.parquet"
    _write_parquet_rg(parquet, texts, row_group_size=1)  # 30 single-doc row groups
    monkeypatch.setattr(token_examples_module, "_ROW_GROUP_CACHE_CHAR_BUDGET", 40)
    oracle = load_shard_text(parquet)

    provider = _row_group_provider(tmp_path, parquet)
    for doc_id in [0, 15, 3, 29, 3, 0, 20, 7, 15, 1]:  # shard-interleaved access
        assert provider._full_text(_text_doc_ref(doc_id)) == oracle[doc_id]

    assert len(provider._row_group_cache) < 30  # eviction happened
    # ledger cannot drift: total == sum of per-entry char counts, and each stored
    # count matches its own texts.
    assert provider._row_group_cache_total_chars == sum(
        chars for _texts, chars in provider._row_group_cache.values()
    )
    for texts, chars in provider._row_group_cache.values():
        assert chars == token_examples_module._row_group_text_chars(texts)


def test_full_text_handles_empty_intermediate_row_group(tmp_path: Path) -> None:
    # An empty (0-row) middle row group makes offsets have a duplicate value;
    # bisect_right must skip it so docs after the gap still resolve correctly.
    parquet = tmp_path / "shard_00000.parquet"
    schema = pa.schema([("text", pa.string())])
    with pq.ParquetWriter(parquet, schema) as writer:
        writer.write_table(pa.table({"text": ["a", "b"]}, schema=schema))
        writer.write_table(pa.table({"text": []}, schema=schema))  # empty row group
        writer.write_table(pa.table({"text": ["c", "d", "e"]}, schema=schema))
    oracle = load_shard_text(parquet)
    assert oracle == ["a", "b", "c", "d", "e"]
    assert token_examples_module._shard_row_group_offsets(parquet) == [0, 2, 2, 5]

    provider = _row_group_provider(tmp_path, parquet)
    for doc_id in range(5):
        assert provider._full_text(_text_doc_ref(doc_id)) == oracle[doc_id]


@pytest.mark.requires_feature_data
def test_full_text_row_group_matches_full_shard_on_real_parquet(tmp_path: Path) -> None:
    # Opt-in (env-gated) so the default suite never does a full ~240 MiB shard
    # load; set AUTOINTERP_REAL_PARQUET to a real corpus shard to run it.
    override = os.environ.get("AUTOINTERP_REAL_PARQUET")
    if not override:
        pytest.skip("set AUTOINTERP_REAL_PARQUET to a real corpus shard to run this")
    parquet = Path(override)
    if not parquet.exists():
        pytest.skip(f"AUTOINTERP_REAL_PARQUET shard not present: {parquet}")
    oracle = load_shard_text(parquet)
    offsets = token_examples_module._shard_row_group_offsets(parquet)
    assert offsets[-1] == len(oracle)

    provider = _row_group_provider(tmp_path, parquet)
    n = len(oracle)
    # sample across the shard incl. row-group boundaries and the tail
    sample = sorted({0, 1, 1023, 1024, 1025, n // 2, n - 2, n - 1})
    for doc_id in sample:
        assert provider._full_text(_text_doc_ref(doc_id)) == oracle[doc_id]


# --- migrated single-process survivors from removed target-sharding test files ---
# (config-bound validation, token-example helpers, and serial provider prepare
# behavior that is exercised by ordinary single-process runs)


def _mig_target(feature_id: int) -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=f"feature:{feature_id}",
        display_name=f"feature_{feature_id}",
    )


def _mig_window(*, activations: tuple[float, ...]) -> TokenWindowRef:
    n_tokens = len(activations)
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=7,
        source_text_hash="a" * 64,
        token_start=0,
        token_end=n_tokens,
        token_ids=tuple(range(n_tokens)),
        token_strings=tuple(f"token-{index}" for index in range(n_tokens)),
        target_feature_ids=(3,),
        activations=activations,
        max_activation=max(activations, default=0.0),
        quantile=None,
        active=False,
    )


class _MigThresholdSae(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(t.tensor([[1.0]], dtype=t.float32), requires_grad=False)
        self.b_enc = t.nn.Parameter(t.tensor([0.0], dtype=t.float32), requires_grad=False)
        self.register_buffer("threshold", t.tensor([0.5], dtype=t.float32))

    def process_sae_in(self, acts: t.Tensor) -> t.Tensor:
        return acts

    def activation_fn(self, hidden_pre: t.Tensor) -> t.Tensor:
        return t.relu(hidden_pre)


def _mig_provider_with_digest_guard_state(tmp_path: Path) -> CachedTokenExampleProvider:
    cfg = AutoInterpFilterConfig(
        feature_dir=str(tmp_path / "features"),
        output_dir=str(tmp_path / "output"),
        source_index=SourceIndexConfig(enabled=True),
    )
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=8,
    )
    provider._source_index_request = cast(SourceIndexRequest, object())
    provider._source_index_loaded = True
    return provider


def _mig_tokprov_cfg(tmp_path: Path, *, source_enabled: bool = True) -> AutoInterpFilterConfig:
    output = tmp_path / "out"
    return AutoInterpFilterConfig(
        feature_dir=str(tmp_path / "features"),
        output_dir=str(output),
        data_dir=str(tmp_path / "data"),
        n_sae_features=_positive(8),
        feature_ids=[0, 1, 2, 3],
        token_example_cache_path=str(output / "token.sqlite"),
        judge_cache_path=str(output / "judge.sqlite"),
        token_examples=TokenExamplesConfig(
            model="test-model",
            max_positive_source_docs_per_target=_positive(3),
            max_intruder_source_docs=_positive(5),
            max_windows_per_source_doc=_positive(2),
            max_positive_windows_per_target=_positive(6),
            max_non_activating_windows_per_target=_positive(6),
        ),
        source_index=SourceIndexConfig(
            enabled=source_enabled,
            path=str(output / "source.sqlite"),
            manifest_path=str(output / "source.manifest.json"),
            shared_cache_dir=str(tmp_path / "shared"),
            chunk_size=_positive(17),
            scan_workers=_positive(2),
        ),
    )


def _mig_patch_prepare_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    parquet_path = tmp_path / "data" / "shard_00000.parquet"
    feature_path.parent.mkdir(parents=True)
    parquet_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"feature-fingerprint")
    parquet_path.write_bytes(b"parquet-fingerprint")
    monkeypatch.setattr(
        token_examples_module,
        "glob_features",
        lambda _feature_dir: [str(feature_path)],
    )
    monkeypatch.setattr(
        token_examples_module,
        "list_parquet_files",
        lambda _data_dir: [str(parquet_path)],
    )
    monkeypatch.setattr(
        CachedTokenExampleProvider,
        "_feature_flags_from_path_or_shard",
        lambda self, path: (True, True),
    )
    return feature_path, parquet_path


def _mig_cap_config(output_dir: Path) -> AutoInterpFilterConfig:
    return AutoInterpFilterConfig(
        feature_dir="logical/features",
        output_dir=str(output_dir),
        n_sae_features=_positive(4),
        feature_ids=[0, 1, 2, 3],
        n_trials_per_target=_positive(1),
        min_trials_for_score=_positive(1),
        n_positive_examples=_positive(1),
        n_quantiles=_positive(1),
        seed=19,
        max_judge_calls=_positive(1),
        score_threshold=_unit(0.5),
        top_k=_positive(2),
        fail_on_empty=True,
        audit_sample_limit=_non_negative(4),
        token_example_cache_path=str(output_dir / "token_example_cache.sqlite"),
        judge_cache_path=str(output_dir / "intruder_trial_cache.sqlite"),
        target_scores_path=str(output_dir / "target_scores.jsonl"),
        ranked_yaml_path=str(output_dir / "autointerp_ranked.yaml"),
        accepted_features_path=str(output_dir / "accepted_feature_ids.txt"),
        manual_feature_filter_path=str(output_dir / "manual_feature_filter.yaml"),
        manual_feature_filter_meta_path=str(output_dir / "manual_feature_filter.meta.json"),
        report_output_path=str(output_dir / "report.json"),
        trial_audit_path=str(output_dir / "trial_audit.jsonl"),
        trial_audit_yaml_path=str(output_dir / "trial_audit.yaml"),
        runner_manifest_path=str(output_dir / "runner_manifest.json"),
        lockfile_path=str(output_dir / ".lockfile"),
        judge=IntruderJudgeConfig(
            model="deterministic-fixture",
            requests_per_minute=_non_negative(0),
            max_concurrent=_positive(1),
            parse_max_retries=_non_negative(0),
        ),
        token_examples=TokenExamplesConfig(
            model="fixture-model",
            example_ctx_len=_positive(2),
            max_positive_source_docs_per_target=_positive(1),
            max_intruder_source_docs=_positive(4),
            max_windows_per_source_doc=_positive(1),
            max_positive_windows_per_target=_positive(1),
            max_non_activating_windows_per_target=_positive(1),
        ),
        source_index=SourceIndexConfig(
            enabled=False,
            path="logical/unused_source.sqlite",
            manifest_path="logical/unused_source.manifest.json",
        ),
    )


def test_source_index_path_identity_oserror_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AutoInterpFilterConfig(
        feature_dir="logical/features",
        output_dir=str(tmp_path),
        source_index=SourceIndexConfig(
            enabled=True,
            path="logical/source.sqlite",
            manifest_path="logical/source.manifest.json",
        ),
    )
    identity_error = OSError("identity lookup denied")

    def fail_identity(_left: Path, _right: Path) -> bool:
        raise identity_error

    monkeypatch.setattr(config_module, "paths_alias", fail_identity)

    with pytest.raises(
        AutoInterpConfigError,
        match="cannot verify source_index.path and source_index.manifest_path identity",
    ) as caught:
        validate_config(cfg)

    assert caught.value.__cause__ is identity_error


def test_zero_intruder_highlight_count_is_an_exact_empty_noop() -> None:
    ref = _mig_window(activations=(1.0, 2.0, 3.0))

    assert deterministic_intruder_highlight_indices(ref, n_highlights=0) == ()


@pytest.mark.parametrize("n_highlights", [-1, -100, True])
def test_invalid_intruder_highlight_count_fails_at_typed_boundary(
    n_highlights: int,
) -> None:
    ref = _mig_window(activations=(1.0, 2.0, 3.0))

    with pytest.raises(BeartypeCallHintParamViolation, match="n_highlights"):
        deterministic_intruder_highlight_indices(ref, n_highlights=n_highlights)


def test_selected_sae_projection_applies_feature_threshold_exactly() -> None:
    """The projected path reproduces JumpReLU's strict threshold boundary."""
    sae = _MigThresholdSae()
    acts = t.tensor(
        [[0.49], [0.50], [0.51], [-2.0]],
        dtype=t.float32,
        requires_grad=True,
    )

    projected = _selected_sae_activations(sae=sae, acts=acts, feature_ids=(0,))

    expected = t.tensor([[0.0], [0.0], [0.51], [0.0]], dtype=t.float32)
    t.testing.assert_close(projected, expected, rtol=0.0, atol=0.0)
    assert projected.requires_grad is False


def test_empty_loaded_source_index_is_not_marked_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _mig_provider_with_digest_guard_state(tmp_path)
    provider._source_index_loaded = False
    digest = "d" * 64
    load_calls: list[SourceIndexRequest] = []

    def load_empty(request: SourceIndexRequest) -> SourceIndexData:
        load_calls.append(request)
        return SourceIndexData(
            positives_by_feature={},
            intruder_refs=[],
            manifest_sha256=digest,
            artifact_sha256=digest,
        )

    monkeypatch.setattr(token_examples_module, "load_or_build_source_index", load_empty)

    with pytest.raises(AutoInterpDataError, match="No source documents were discovered"):
        provider._ensure_source_index_loaded()

    assert load_calls == [provider._source_index_request]
    assert provider._source_index_loaded is False
    assert provider._intruder_docs == []


def test_legacy_prepare_preserves_none_projection_and_full_activation_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mig_patch_prepare_environment(monkeypatch, tmp_path)
    captured: list[SourceIndexRequest] = []
    monkeypatch.setattr(
        token_examples_module,
        "validate_source_index_cache",
        lambda request: captured.append(request) or request,
    )
    cfg = _mig_tokprov_cfg(tmp_path)
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=8,
    )
    full = [_mig_target(index) for index in range(4)]

    provider.prepare(full)

    assert captured[0].target_feature_ids == (0, 1, 2, 3)
    assert captured[0].load_target_feature_ids is None
    assert provider._activation_feature_ids == (0, 1, 2, 3)
    assert provider._targets == full


def test_prepare_rejects_duplicate_full_target_universe_before_io(
    tmp_path: Path,
) -> None:
    cfg = _mig_tokprov_cfg(tmp_path)
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=8,
    )

    with pytest.raises(AutoInterpDataError, match="full target universe contains duplicates"):
        provider.prepare([_mig_target(1), _mig_target(1)])


@pytest.mark.parametrize("value", [None, "4", 4.0, False, True])
def test_audit_sample_limit_requires_a_strict_integer(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="audit_sample_limit"):
        replace(_mig_cap_config(tmp_path), audit_sample_limit=cast(Any, value))


def test_audit_sample_limit_rejects_negative_and_accepts_zero(tmp_path: Path) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="audit_sample_limit"):
        replace(_mig_cap_config(tmp_path), audit_sample_limit=cast(Any, -1))

    validate_config(replace(_mig_cap_config(tmp_path), audit_sample_limit=cast(Any, 0)))


@pytest.mark.parametrize("value", [None, "true", 0, 1, 1.0, []])
def test_fail_on_empty_requires_a_strict_boolean(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="fail_on_empty"):
        replace(_mig_cap_config(tmp_path), fail_on_empty=cast(Any, value))


def test_neuronpedia_backoff_max_must_cover_base_and_equal_boundary_is_valid(
    tmp_path: Path,
) -> None:
    cfg = _mig_cap_config(tmp_path)
    invalid = NeuronpediaDescriptionConfig(
        backoff_base_seconds=_positive_finite(2.0),
        backoff_max_seconds=_positive_finite(1.0),
    )
    with pytest.raises(
        AutoInterpConfigError,
        match=("neuronpedia.backoff_max_seconds must be >= neuronpedia.backoff_base_seconds"),
    ):
        validate_config(replace(cfg, neuronpedia=invalid))

    validate_config(
        replace(
            cfg,
            neuronpedia=NeuronpediaDescriptionConfig(
                timeout_seconds=_positive_finite(1),
                max_retries=_non_negative(0),
                backoff_base_seconds=_positive_finite(0.25),
                backoff_max_seconds=_positive_finite(0.25),
            ),
        )
    )


@pytest.mark.parametrize(
    ("model_id", "sae_source", "error"),
    [
        (None, "fixture-sae", "neuronpedia.model_id is required"),
        ("fixture-model", None, "neuronpedia.sae_source is required"),
        ("fixture-model", "fixture-sae", None),
    ],
)
def test_enabled_neuronpedia_requires_and_accepts_complete_identity(
    tmp_path: Path,
    model_id: str | None,
    sae_source: str | None,
    error: str | None,
) -> None:
    cfg = _mig_cap_config(tmp_path)
    cfg = replace(
        cfg,
        neuronpedia=NeuronpediaDescriptionConfig(
            enabled=True,
            model_id=model_id,
            sae_source=sae_source,
        ),
    )

    if error is None:
        validate_config(cfg)
    else:
        with pytest.raises(AutoInterpConfigError, match=error):
            validate_config(cfg)
