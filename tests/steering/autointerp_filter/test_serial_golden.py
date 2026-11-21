"""Golden characterization of the legacy serial autointerp runner.

These tests deliberately exercise the public runner without the real model or
network.  They pin the unsharded ordering/finalization contract before target
sharding is introduced.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    IntruderJudgeConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.document_blacklist import (
    EMPTY_DOCUMENT_BLACKLIST_SHA256,
)
from steering.eval.autointerp_filter.intruder import (
    IntruderDetectionJudge,
    judge_cache_key,
    render_trial_examples,
)
from steering.eval.autointerp_filter.runner import AutoInterpFilterRunner
from steering.eval.autointerp_filter.targets import target_payload_hash
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    IntruderTrial,
    IntruderTrialResult,
    RenderedExample,
    ScoringTarget,
    TokenExampleProvider,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import NonNegativeInt, PositiveInt, UnitInterval

_PROVIDER_CONTEXT = {
    "fixture": "serial_runner_golden_v1",
    "source_contract": "fixed",
}
_CORRECT_TRIALS_BY_FEATURE = {
    0: frozenset({0, 2}),
    1: frozenset({1, 2}),
    2: frozenset({0, 1, 2}),
    3: frozenset({1}),
}


class _GoldenProvider:
    def __init__(self) -> None:
        self.prepared_targets: list[ScoringTarget] = []

    def prepare(self, targets: list[ScoringTarget]) -> None:
        self.prepared_targets = list(targets)

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        feature_id = target.feature_ids[0]
        positive = _ref(
            target,
            doc_id=feature_id * 100,
            active=True,
            activations=(0.0, float(feature_id + 2), 0.0, 0.0),
        )
        intruders = tuple(
            _ref(
                target,
                doc_id=feature_id * 100 + 10 + offset,
                active=False,
                activations=(0.0, 0.0, 0.0, 0.0),
            )
            for offset in range(4)
        )
        return TokenExampleSet(positives=(positive,), non_activating=intruders)

    def cache_context(self) -> dict:
        return dict(_PROVIDER_CONTEXT)


class _GoldenRenderer:
    def render(
        self,
        ref: TokenWindowRef,
        *,
        activating: bool,
        highlighted_token_indices: tuple[int, ...],
    ) -> RenderedExample:
        role = "POS" if activating else "INTRUDER"
        return RenderedExample(
            text=f"<{role}:{ref.target_feature_ids[0]}:{ref.doc_id}>",
            ref=ref,
            activating=activating,
            highlighted_token_indices=highlighted_token_indices,
        )


class _GoldenJudge:
    def __init__(self) -> None:
        self.trials: list[IntruderTrial] = []

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: _GoldenRenderer,
    ) -> IntruderTrialResult:
        self.trials.append(trial)
        feature_id = trial.target.feature_ids[0]
        correct = trial.trial_index in _CORRECT_TRIALS_BY_FEATURE[feature_id]
        prediction = trial.intruder_index
        if not correct:
            prediction = (prediction + 1) % 2
        examples = tuple(render_trial_examples(trial, renderer))
        return IntruderTrialResult(
            trial_key=trial.trial_key,
            prediction=prediction,
            correct_index=trial.intruder_index,
            correct=correct,
            response=f"[RESPONSE]: {prediction}",
            cached=False,
            rendered_examples=examples,
            final_user_prompt=(
                f"fixture prompt {trial.target.target_key} trial {trial.trial_index}"
            ),
        )


def _ref(
    target: ScoringTarget,
    *,
    doc_id: int,
    active: bool,
    activations: tuple[float, ...],
) -> TokenWindowRef:
    token_ids = tuple(doc_id * 10 + offset for offset in range(len(activations)))
    return TokenWindowRef(
        shard_stem="golden_shard",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"source-{doc_id}",
        token_start=0,
        token_end=len(activations),
        token_ids=token_ids,
        token_strings=tuple(f"tok-{token_id}" for token_id in token_ids),
        target_feature_ids=target.feature_ids,
        activations=activations,
        max_activation=max(activations, default=0.0),
        quantile=None,
        active=active,
    )


def _config(tmp_path: Path) -> AutoInterpFilterConfig:
    output_dir = tmp_path / "serial-golden"
    return AutoInterpFilterConfig(
        feature_dir="fixture/features",
        output_dir=str(output_dir),
        n_sae_features=PositiveInt.parse(4),
        feature_ids=[2, 0, 3, 1],
        n_trials_per_target=PositiveInt.parse(3),
        min_trials_for_score=PositiveInt.parse(1),
        n_positive_examples=PositiveInt.parse(1),
        n_quantiles=PositiveInt.parse(1),
        seed=17,
        score_threshold=UnitInterval.parse(0.5),
        top_k=PositiveInt.parse(2),
        fail_on_empty=True,
        audit_sample_limit=NonNegativeInt.parse(5),
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
            model="golden-judge",
            requests_per_minute=NonNegativeInt.parse(0),
            max_concurrent=PositiveInt.parse(1),
            max_prompt_chars=PositiveInt.parse(1234),
            parse_max_retries=NonNegativeInt.parse(0),
            cot=False,
        ),
        token_examples=TokenExamplesConfig(
            device="cpu",
            model="golden-model",
            sae_release="golden-sae",
            sae_l0=7,
            activation_backend="transformerlens",
            model_dtype="float32",
            context_window=PositiveInt.parse(64),
            example_ctx_len=PositiveInt.parse(4),
            activation_threshold_ratio=UnitInterval.parse(0.25),
            max_positive_source_docs_per_target=PositiveInt.parse(8),
            max_intruder_source_docs=PositiveInt.parse(16),
            max_windows_per_source_doc=PositiveInt.parse(2),
            max_positive_windows_per_target=PositiveInt.parse(8),
            max_non_activating_windows_per_target=PositiveInt.parse(8),
        ),
        source_index=SourceIndexConfig(
            enabled=False,
            path=str(output_dir / "source_doc_index.sqlite"),
            manifest_path=str(output_dir / "source_doc_index.manifest.json"),
            shared_cache_dir=None,
            force_rebuild=False,
            chunk_size=PositiveInt.parse(99),
            scan_workers=PositiveInt.parse(1),
        ),
    )


def _run(tmp_path: Path) -> tuple[AutoInterpFilterConfig, _GoldenProvider, _GoldenJudge]:
    cfg = _config(tmp_path)
    provider = _GoldenProvider()
    judge = _GoldenJudge()
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=cast(TokenExampleProvider, provider),
        renderer=_GoldenRenderer(),
    ).run()
    return cfg, provider, judge


def _jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_serial_runner_manifest_still_rejects_requests_per_minute_drift(
    tmp_path: Path,
) -> None:
    cfg, provider, _judge = _run(tmp_path)
    manifest_path = Path(cfg.runner_manifest_path)
    original = manifest_path.read_bytes()
    changed = replace(
        cfg,
        judge=replace(
            cfg.judge,
            requests_per_minute=cfg.judge.requests_per_minute + 1,
        ),
    )
    runner = AutoInterpFilterRunner(
        changed,
        judge=cast(IntruderDetectionJudge, _GoldenJudge()),
        token_example_provider=cast(TokenExampleProvider, _GoldenProvider()),
        renderer=_GoldenRenderer(),
    )

    with pytest.raises(
        RuntimeError,
        match="does not match the current run configuration",
    ):
        runner._ensure_manifest(
            n_sae_features=4,
            targets=provider.prepared_targets,
            provider_context=dict(_PROVIDER_CONTEXT),
            write_if_missing=False,
        )

    assert manifest_path.read_bytes() == original


def test_serial_runner_final_artifacts_are_golden(tmp_path: Path) -> None:
    cfg, provider, _judge = _run(tmp_path)

    assert [target.target_key for target in provider.prepared_targets] == [
        "feature:0",
        "feature:1",
        "feature:2",
        "feature:3",
    ]
    expected_scores = [
        {
            "descriptions": {"2": None},
            "display_name": "feature_2",
            "error_message": None,
            "feature_ids": [2],
            "kind": "feature",
            "n_correct": 3,
            "n_trials": 3,
            "outcome": "accepted",
            "schema_version": SCHEMA_VERSION,
            "score": 1.0,
            "target_key": "feature:2",
        },
        {
            "descriptions": {"0": None},
            "display_name": "feature_0",
            "error_message": None,
            "feature_ids": [0],
            "kind": "feature",
            "n_correct": 2,
            "n_trials": 3,
            "outcome": "accepted",
            "schema_version": SCHEMA_VERSION,
            "score": 2 / 3,
            "target_key": "feature:0",
        },
        {
            "descriptions": {"1": None},
            "display_name": "feature_1",
            "error_message": None,
            "feature_ids": [1],
            "kind": "feature",
            "n_correct": 2,
            "n_trials": 3,
            "outcome": "rejected",
            "schema_version": SCHEMA_VERSION,
            "score": 2 / 3,
            "target_key": "feature:1",
        },
        {
            "descriptions": {"3": None},
            "display_name": "feature_3",
            "error_message": None,
            "feature_ids": [3],
            "kind": "feature",
            "n_correct": 1,
            "n_trials": 3,
            "outcome": "rejected",
            "schema_version": SCHEMA_VERSION,
            "score": 1 / 3,
            "target_key": "feature:3",
        },
    ]
    assert _jsonl(cfg.target_scores_path) == expected_scores
    assert Path(cfg.target_scores_path).read_text() == "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in expected_scores
    )
    assert Path(cfg.accepted_features_path).read_text() == "0\n2\n"
    assert yaml.safe_load(Path(cfg.ranked_yaml_path).read_text()) == {
        "feature_dir": "fixture/features",
        "score_threshold": 0.5,
        "top_k": 2,
        "accepted_feature_ids": [0, 2],
        "targets": [
            {"description": None, "feature_ids": [2], "score": 1.0},
            {"description": None, "feature_ids": [0], "score": 2 / 3},
            {"description": None, "feature_ids": [1], "score": 2 / 3},
            {"description": None, "feature_ids": [3], "score": 1 / 3},
        ],
    }
    assert Path(cfg.ranked_yaml_path).read_text() == (
        "feature_dir: fixture/features\n"
        "score_threshold: 0.5\n"
        "top_k: 2\n"
        "accepted_feature_ids:\n"
        "- 0\n"
        "- 2\n"
        "targets:\n"
        "- description: null\n"
        "  feature_ids:\n"
        "  - 2\n"
        "  score: 1.0\n"
        "- description: null\n"
        "  feature_ids:\n"
        "  - 0\n"
        "  score: 0.6666666666666666\n"
        "- description: null\n"
        "  feature_ids:\n"
        "  - 1\n"
        "  score: 0.6666666666666666\n"
        "- description: null\n"
        "  feature_ids:\n"
        "  - 3\n"
        "  score: 0.3333333333333333\n"
    )
    assert Path(cfg.manual_feature_filter_path).read_text() == (
        '# - "2"  # score=1.000\n'
        '# - "0"  # score=0.667\n'
        '# - "1"  # score=0.667\n'
        '# - "3"  # score=0.333\n'
    )

    manual_meta = json.loads(Path(cfg.manual_feature_filter_meta_path).read_text())
    assert isinstance(manual_meta.pop("generated_at"), str)
    assert manual_meta == {
        "feature_dir": "fixture/features",
        "n_targets": 4,
        "neuronpedia_model_id": None,
        "neuronpedia_sae_source": None,
        "ranked_yaml_path": cfg.ranked_yaml_path,
        "schema_version": SCHEMA_VERSION,
        "score_threshold": 0.5,
        "target_scores_path": cfg.target_scores_path,
        "top_k": 2,
    }
    assert json.loads(Path(cfg.report_output_path).read_text()) == {
        "accepted_features_path": cfg.accepted_features_path,
        "manual_feature_filter_meta_path": cfg.manual_feature_filter_meta_path,
        "manual_feature_filter_path": cfg.manual_feature_filter_path,
        "n_features_accepted": 2,
        "n_targets_accepted": 2,
        "n_targets_scored": 4,
        "n_targets_total": 4,
        "ranked_yaml_path": cfg.ranked_yaml_path,
        "schema_version": SCHEMA_VERSION,
        "score_threshold": 0.5,
        "target_scores_path": cfg.target_scores_path,
        "top_k": 2,
        "trial_audit_path": cfg.trial_audit_path,
        "trial_audit_yaml_path": cfg.trial_audit_yaml_path,
    }


def test_serial_runner_audit_manifest_and_cache_identity_are_golden(
    tmp_path: Path,
) -> None:
    cfg, _provider, judge = _run(tmp_path)

    audit_rows = _jsonl(cfg.trial_audit_path)
    assert [(row["target_key"], row["trial_index"], row["correct"]) for row in audit_rows] == [
        ("feature:0", 0, True),
        ("feature:0", 1, False),
        ("feature:0", 2, True),
        ("feature:1", 0, False),
        ("feature:1", 1, True),
    ]
    assert [row["final_user_prompt_sent_to_judge"] for row in audit_rows] == [
        "fixture prompt feature:0 trial 0",
        "fixture prompt feature:0 trial 1",
        "fixture prompt feature:0 trial 2",
        "fixture prompt feature:1 trial 0",
        "fixture prompt feature:1 trial 1",
    ]
    assert all(row["cached"] is False for row in audit_rows)
    assert [[example["role"] for example in row["examples"]] for row in audit_rows] == [
        ["intruder", "positive"],
        ["intruder", "positive"],
        ["intruder", "positive"],
        ["positive", "intruder"],
        ["intruder", "positive"],
    ]
    assert all(
        set(row)
        == {
            "cached",
            "correct",
            "correct_index",
            "display_name",
            "examples",
            "feature_ids",
            "few_shot_set",
            "final_user_prompt_sent_to_judge",
            "intruder_index",
            "kind",
            "prediction",
            "prompt_variant",
            "quantile",
            "response",
            "target_key",
            "trial_index",
            "trial_key",
        }
        for row in audit_rows
    )
    assert all(
        set(example)
        == {
            "activating",
            "activations",
            "highlighted_token_indices",
            "index",
            "max_activation",
            "positive_max_token_indices",
            "ref",
            "role",
            "text",
            "token_ids",
            "token_strings",
        }
        for row in audit_rows
        for example in row["examples"]
    )
    assert all(
        set(example["ref"])
        == {
            "active",
            "doc_id",
            "quantile",
            "shard_index",
            "shard_stem",
            "source_text_hash",
            "stable_key",
            "target_feature_ids",
            "token_end",
            "token_start",
        }
        for row in audit_rows
        for example in row["examples"]
    )
    assert yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()) == [
        {
            "display_name": row["display_name"],
            "feature_ids": row["feature_ids"],
            "quantile": row["quantile"],
            "prediction": row["prediction"],
            "correct_index": row["correct_index"],
            "response": row["response"],
            "final_user_prompt_sent_to_judge": row["final_user_prompt_sent_to_judge"],
            "examples": [
                {"text": example["text"], "max_activation": example["max_activation"]}
                for example in row["examples"]
            ],
        }
        for row in audit_rows
    ]

    manifest = json.loads(Path(cfg.runner_manifest_path).read_text())
    assert isinstance(manifest.pop("created_at"), str)
    assert manifest == {
        "data_dir": None,
        "feature_dir": "fixture/features",
        "judge": {
            "cot": False,
            "few_shot_set": "sae_realistic_v1",
            "max_concurrent": 1,
            "max_prompt_chars": 1234,
            "model": "golden-judge",
            "parse_max_retries": 0,
            "prompt_variant": "highlight_context_token_first_v1",
            "requests_per_minute": 0,
        },
        "document_blacklist_path": None,
        "document_blacklist_sha256": EMPTY_DOCUMENT_BLACKLIST_SHA256,
        "min_trials_for_score": 1,
        "n_positive_examples": 1,
        "n_quantiles": 1,
        "n_sae_features": 4,
        "n_trials_per_target": 3,
        "provider_context": _PROVIDER_CONTEXT,
        "schema_version": SCHEMA_VERSION,
        "score_threshold": 0.5,
        "seed": 17,
        "source_index": {
            "chunk_size": 99,
            "enabled": False,
            "force_rebuild": False,
            "manifest_path": cfg.source_index.manifest_path,
            "path": cfg.source_index.path,
            "scan_workers": 1,
            "shared_cache_dir": None,
        },
        "targets": [
            {"feature_ids": [feature_id], "kind": "feature", "target_key": f"feature:{feature_id}"}
            for feature_id in range(4)
        ],
        "token_example_cache_path": cfg.token_example_cache_path,
        "token_examples": {
            "activation_backend": "transformerlens",
            "activation_threshold_ratio": 0.25,
            "context_window": 64,
            "device": "cpu",
            "example_ctx_len": 4,
            "max_intruder_source_docs": 16,
            "max_non_activating_windows_per_target": 8,
            "max_positive_source_docs_per_target": 8,
            "max_positive_windows_per_target": 8,
            "max_windows_per_source_doc": 2,
            "model": "golden-model",
            "model_dtype": "float32",
            "sae_l0": 7,
            "sae_release": "golden-sae",
        },
        "top_k": 2,
    }

    assert [target_payload_hash(trial.target) for trial in judge.trials[::3]] == [
        "e7ca55f4a9c2cc34e49912beedadaaf4658948f9715e8bb2c15ca0f498a584e3",
        "38d4e348f8a42cc48af2adf370c2b229493ad0a5f17e6398da96b76ec988a8ef",
        "c4901ef49220a725d92dedd06412821987c543588aa32250649bcdf5d891c2e9",
        "dd97d76a1b8810e7ab4cb3c0bb1033d04c5f47b6ce5e87fbd1e73e8a54fb0e03",
    ]
    assert [trial.trial_key for trial in judge.trials] == [
        "f9154491dbc68176edcd4b94f3ad89fed256d2f2e0c7832c65db3693f092e7f2",
        "ba96298aa74c78cbf3290b5dbb9b782d094cba2faa7f91251a02a09fa066b5a3",
        "2b974305293eaf11d7ca0a17e585f0ffef7d6d132865aeba7e203775bfe87206",
        "2a6ebcf4741f4bcee8b069349dd6be326a3f78057cc5a2e61cb5f563b731123d",
        "9d065cad6052a78f2e641ea9bcf3dfa7a5613ac6be2fcddc5c446b9f9ec5720b",
        "18614fd29c3f697157b3fc681e061c6599895c97738433575ff867ffaaef806a",
        "b3e551b5b9515aed18db6c2a4c9251b4fa897c2e2cf3923ca407a91fccac0418",
        "82faa488fba92143f4992781a9627502aab920f25e6717b71d93b1dbfccacbc1",
        "57e81f728a111721a3910f06669d65ebf6adcaed11e159fcb1b675302d63ab06",
        "46e8fb96f610c68583198b595d8002554551af0d09975e21f855372ec684bd7a",
        "d3be3be4f0413dad406df627137ae2a12d877f38fea3927a04ebac3fe3e8d325",
        "1782600aabd3dfbfeebad09b8a486e5dd95a33ba1fd0e4b2cc94960cf0a6fead",
    ]
    expected_judge_key = "48db6b630cf4dabb7451a53eecfd9fdeda038c12fe5962248380803b98279eeb"
    assert judge_cache_key(cfg.judge, render_context=_PROVIDER_CONTEXT) == expected_judge_key

    conn = sqlite3.connect(cfg.judge_cache_path)
    try:
        cached_keys = conn.execute(
            "SELECT target_key, trial_key, judge_key FROM trial_results "
            "ORDER BY target_key, trial_key"
        ).fetchall()
    finally:
        conn.close()
    assert cached_keys == sorted(
        (trial.target.target_key, trial.trial_key, expected_judge_key) for trial in judge.trials
    )
