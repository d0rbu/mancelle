from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from steering.eval.autointerp_filter.artifacts import (
    _resolve_output_path,
    dump_human_yaml,
    finalize_scores,
    passes_threshold,
    ranked_description,
    score_to_json,
    score_to_ranked_yaml,
    trial_audit_yaml_row,
    write_final_outputs,
)
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    NeuronpediaDescriptionConfig,
)
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    TargetKind,
    TargetOutcome,
    TargetScore,
)


def _score(
    feature_id: int,
    score: float | None,
    *,
    outcome: TargetOutcome | None = None,
    n_correct: int | None = None,
    n_trials: int = 10,
    description: str | None = None,
    error_message: str | None = None,
) -> TargetScore:
    if n_correct is None:
        n_correct = 0 if score is None else int(round(score * n_trials))
    if outcome is None:
        outcome = TargetOutcome.insufficient_examples if score is None else TargetOutcome.rejected
    return TargetScore(
        target_key=f"feature:{feature_id}",
        display_name=f"feature_{feature_id}",
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        score=score,
        n_correct=n_correct,
        n_trials=n_trials if score is not None else 0,
        outcome=outcome,
        descriptions={feature_id: description},
        error_message=error_message,
    )


def _cfg(tmp_path: Path, **overrides: Any) -> AutoInterpFilterConfig:
    output = tmp_path / "out"
    values: dict[str, Any] = {
        "feature_dir": str(tmp_path / "features"),
        "output_dir": str(output),
        "n_trials_per_target": 10,
        "score_threshold": 0.6,
        "top_k": None,
        "audit_sample_limit": 20,
        "target_scores_path": str(output / "target_scores.jsonl"),
        "ranked_yaml_path": str(output / "autointerp_ranked.yaml"),
        "accepted_features_path": str(output / "accepted_feature_ids.txt"),
        "manual_feature_filter_path": str(output / "manual_feature_filter.yaml"),
        "manual_feature_filter_meta_path": str(output / "manual_feature_filter.meta.json"),
        "report_output_path": str(output / "report.json"),
        "trial_audit_path": str(output / "trial_audit.jsonl"),
        "trial_audit_yaml_path": str(output / "trial_audit.yaml"),
        "neuronpedia": NeuronpediaDescriptionConfig(
            enabled=False,
            model_id="model/id",
            sae_source="layer/sae",
        ),
    }
    values.update(overrides)
    return AutoInterpFilterConfig(**values)


def _audit_row(index: int) -> dict:
    return {
        "display_name": f"feature_{index}",
        "feature_ids": [index],
        "quantile": index % 5,
        "prediction": index % 4,
        "correct_index": (index + 1) % 4,
        "response": f"response-{index}",
        "final_user_prompt_sent_to_judge": f"prompt-{index}\nsecond line",
        "examples": [
            {
                "text": f"example-{index}-{example_index}",
                "max_activation": float(example_index),
                "private_detail": "kept-only-in-jsonl",
            }
            for example_index in range(2)
        ],
        "machine_only": {"trial_index": index},
    }


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.0, None, True),
        (1.0, None, True),
        (0.0, 0.0, True),
        (0.599999999999, 0.6, False),
        (0.6, 0.6, True),
        (1.0, 1.0, True),
    ],
)
def test_passes_threshold_boundaries(
    score: float,
    threshold: float | None,
    expected: bool,
) -> None:
    assert passes_threshold(score, threshold) is expected


def test_finalize_scores_applies_global_top_k_after_threshold() -> None:
    scores = [
        _score(9, 0.9),
        _score(1, 0.7),
        _score(2, 0.8),
        _score(3, 0.5, outcome=TargetOutcome.accepted),
        _score(4, None, outcome=TargetOutcome.insufficient_examples),
    ]

    finalized = finalize_scores(scores, score_threshold=0.6, top_k=2)

    assert [score.target_key for score in finalized.accepted_scores] == [
        "feature:9",
        "feature:2",
    ]
    assert finalized.accepted_feature_ids == (2, 9)
    by_key: dict[str, TargetScore] = {
        str(score.target_key): score for score in finalized.ordered_scores
    }
    assert by_key["feature:9"].outcome is TargetOutcome.accepted
    assert by_key["feature:2"].outcome is TargetOutcome.accepted
    assert by_key["feature:1"].outcome is TargetOutcome.rejected
    assert by_key["feature:3"].outcome is TargetOutcome.rejected
    assert by_key["feature:4"].outcome is TargetOutcome.insufficient_examples


def test_finalize_scores_breaks_cross_shard_style_ties_by_feature_id() -> None:
    scores = [_score(17, 0.8), _score(2, 0.8), _score(11, 0.8)]

    finalized = finalize_scores(scores, score_threshold=0.8, top_k=2)

    assert [score.feature_ids for score in finalized.accepted_scores] == [(2,), (11,)]
    assert finalized.accepted_feature_ids == (2, 11)


def test_finalize_scores_threshold_none_accepts_all_concrete_scores() -> None:
    finalized = finalize_scores(
        [_score(0, 0.0), _score(1, 1.0), _score(2, None)],
        score_threshold=None,
        top_k=None,
    )

    assert finalized.accepted_feature_ids == (0, 1)
    assert [score.score for score in finalized.ordered_scores] == [1.0, 0.0, None]


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.integers(min_value=0, max_value=10),
        ),
        min_size=1,
        max_size=50,
        unique_by=lambda item: item[0],
    )
)
def test_finalize_scores_is_permutation_invariant_for_unique_targets(
    rows: list[tuple[int, int]],
) -> None:
    scores = [
        _score(feature_id, numerator / 10, n_correct=numerator) for feature_id, numerator in rows
    ]
    reversed_scores = list(reversed(scores))

    first = finalize_scores(scores, score_threshold=0.4, top_k=7)
    second = finalize_scores(reversed_scores, score_threshold=0.4, top_k=7)

    assert first == second


def test_score_to_json_uses_stable_json_types() -> None:
    score = _score(
        7,
        0.7,
        outcome=TargetOutcome.accepted,
        description="a description",
        error_message="partial",
    )

    payload = score_to_json(score)

    assert payload == {
        "target_key": "feature:7",
        "display_name": "feature_7",
        "kind": "feature",
        "feature_ids": [7],
        "score": 0.7,
        "n_correct": 7,
        "n_trials": 10,
        "outcome": "accepted",
        "descriptions": {"7": "a description"},
        "error_message": "partial",
        "schema_version": SCHEMA_VERSION,
    }


def test_ranked_description_singleton_and_multi_feature() -> None:
    singleton = _score(3, 1.0, description="singleton")
    multi = TargetScore(
        target_key="legacy:1,2,3",
        display_name="legacy",
        kind=TargetKind.feature,
        feature_ids=(1, 2, 3),
        score=1.0,
        n_correct=10,
        n_trials=10,
        outcome=TargetOutcome.accepted,
        descriptions={1: "one", 2: None, 3: "three"},
    )

    assert ranked_description(singleton) == "singleton"
    assert ranked_description(multi) == [
        {"feature_id": 1, "description": "one"},
        {"feature_id": 3, "description": "three"},
    ]


def test_score_to_ranked_yaml_includes_only_non_null_error() -> None:
    assert score_to_ranked_yaml(_score(0, 1.0)) == {
        "description": None,
        "feature_ids": [0],
        "score": 1.0,
    }
    assert score_to_ranked_yaml(_score(1, None, error_message="insufficient")) == {
        "description": None,
        "feature_ids": [1],
        "score": None,
        "error_message": "insufficient",
    }


def test_trial_audit_yaml_row_is_exact_human_projection() -> None:
    row = _audit_row(2)

    projected = trial_audit_yaml_row(row)

    assert projected == {
        "display_name": "feature_2",
        "feature_ids": [2],
        "quantile": 2,
        "prediction": 2,
        "correct_index": 3,
        "response": "response-2",
        "final_user_prompt_sent_to_judge": "prompt-2\nsecond line",
        "examples": [
            {"text": "example-2-0", "max_activation": 0.0},
            {"text": "example-2-1", "max_activation": 1.0},
        ],
    }
    assert "machine_only" not in projected
    assert all("private_detail" not in example for example in projected["examples"])


def test_dump_human_yaml_uses_literal_style_for_multiline_strings() -> None:
    rendered = dump_human_yaml({"prompt": "line one\nline two", "plain": "one line"})

    assert "prompt: |-\n  line one\n  line two" in rendered
    assert "plain: one line" in rendered
    assert yaml.safe_load(rendered) == {
        "prompt": "line one\nline two",
        "plain": "one line",
    }


def test_write_final_outputs_writes_canonical_global_artifact_set(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, top_k=2, audit_sample_limit=2)
    scores = [
        _score(99, 0.6, description="ninety nine"),
        _score(7, 0.9, description="seven"),
        _score(42, 0.9, description="forty two"),
        _score(3, None, error_message="needed trials"),
    ]
    audit_rows = [_audit_row(index) for index in range(4)]

    report = write_final_outputs(
        cfg,
        scores=scores,
        audit_rows=audit_rows,
        n_targets_total=4,
    )

    persisted_scores = [
        json.loads(line) for line in Path(cfg.target_scores_path).read_text().splitlines()
    ]
    assert [row["feature_ids"] for row in persisted_scores] == [[7], [42], [99], [3]]
    assert [row["outcome"] for row in persisted_scores] == [
        "accepted",
        "accepted",
        "rejected",
        "insufficient_examples",
    ]
    assert Path(cfg.accepted_features_path).read_text() == "7\n42\n"

    ranked = yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())
    assert ranked["accepted_feature_ids"] == [7, 42]
    assert ranked["targets"] == [
        {"description": "seven", "feature_ids": [7], "score": 0.9},
        {"description": "forty two", "feature_ids": [42], "score": 0.9},
        {"description": "ninety nine", "feature_ids": [99], "score": 0.6},
        {
            "description": None,
            "feature_ids": [3],
            "score": None,
            "error_message": "needed trials",
        },
    ]

    assert [
        json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
    ] == audit_rows[:2]
    assert yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()) == [
        trial_audit_yaml_row(row) for row in audit_rows[:2]
    ]

    manual_lines = Path(cfg.manual_feature_filter_path).read_text().splitlines()
    assert manual_lines == [
        '# - "07"  # score=0.900 https://www.neuronpedia.org/model/id/layer/sae/7',
        '# - "42"  # score=0.900 https://www.neuronpedia.org/model/id/layer/sae/42',
        '# - "99"  # score=0.600 https://www.neuronpedia.org/model/id/layer/sae/99',
        '# - "03"  # score=null https://www.neuronpedia.org/model/id/layer/sae/3',
    ]
    manual_meta = json.loads(Path(cfg.manual_feature_filter_meta_path).read_text())
    generated_at = manual_meta.pop("generated_at")
    assert generated_at.endswith("+00:00")
    assert manual_meta == {
        "feature_dir": cfg.feature_dir,
        "n_targets": 4,
        "neuronpedia_model_id": "model/id",
        "neuronpedia_sae_source": "layer/sae",
        "ranked_yaml_path": cfg.ranked_yaml_path,
        "schema_version": SCHEMA_VERSION,
        "score_threshold": 0.6,
        "target_scores_path": cfg.target_scores_path,
        "top_k": 2,
    }

    persisted_report = json.loads(Path(cfg.report_output_path).read_text())
    assert persisted_report == {
        "accepted_features_path": cfg.accepted_features_path,
        "manual_feature_filter_meta_path": cfg.manual_feature_filter_meta_path,
        "manual_feature_filter_path": cfg.manual_feature_filter_path,
        "n_features_accepted": 2,
        "n_targets_accepted": 2,
        "n_targets_scored": 3,
        "n_targets_total": 4,
        "ranked_yaml_path": cfg.ranked_yaml_path,
        "schema_version": SCHEMA_VERSION,
        "score_threshold": 0.6,
        "target_scores_path": cfg.target_scores_path,
        "top_k": 2,
        "trial_audit_path": cfg.trial_audit_path,
        "trial_audit_yaml_path": cfg.trial_audit_yaml_path,
    }
    assert report.n_targets_total == 4
    assert report.n_targets_scored == 3
    assert report.n_targets_accepted == 2
    assert report.n_features_accepted == 2


def test_write_final_outputs_rejects_logical_path_outside_anchored_output_root(
    tmp_path: Path,
) -> None:
    escaped_path = tmp_path / "escaped-target-scores.jsonl"
    physical_output_root = tmp_path / "anchored-output"
    physical_output_root.mkdir()
    cfg = _cfg(tmp_path, target_scores_path=str(escaped_path))

    with pytest.raises(
        ValueError,
        match=r"configured output path .*escaped-target-scores\.jsonl is outside output_dir",
    ):
        write_final_outputs(
            cfg,
            scores=[_score(1, 1.0)],
            audit_rows=[],
            physical_output_root=physical_output_root,
        )

    assert not escaped_path.exists()
    assert list(physical_output_root.iterdir()) == []


def test_write_final_outputs_zero_audit_limit_writes_empty_valid_files(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, audit_sample_limit=0)

    write_final_outputs(cfg, scores=[_score(1, 1.0)], audit_rows=[_audit_row(0)])

    assert Path(cfg.trial_audit_path).read_bytes() == b""
    assert Path(cfg.trial_audit_yaml_path).read_text() == "[]\n"
    assert yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()) == []


def test_write_final_outputs_empty_scores_is_well_formed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    report = write_final_outputs(cfg, scores=[], audit_rows=[])

    assert Path(cfg.target_scores_path).read_bytes() == b""
    assert Path(cfg.accepted_features_path).read_bytes() == b""
    assert Path(cfg.manual_feature_filter_path).read_bytes() == b""
    assert yaml.safe_load(Path(cfg.ranked_yaml_path).read_text())["targets"] == []
    assert report.n_targets_total == 0
    assert report.n_targets_scored == 0
    assert report.n_targets_accepted == 0
    assert report.n_features_accepted == 0


def test_resolve_output_path_is_identity_without_physical_root(tmp_path: Path) -> None:
    configured = str(tmp_path / "out" / "target_scores.jsonl")

    resolved = _resolve_output_path(
        configured,
        output_dir=str(tmp_path / "out"),
        physical_output_root=None,
    )

    assert resolved == Path(configured)


def test_resolve_output_path_remaps_into_physical_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    physical_root = tmp_path / "anchored"
    configured = str(output_dir / "nested" / "target_scores.jsonl")

    resolved = _resolve_output_path(
        configured,
        output_dir=str(output_dir),
        physical_output_root=physical_root,
    )

    assert resolved == physical_root / "nested" / "target_scores.jsonl"


def test_resolve_output_path_rejects_paths_outside_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    escaped = str(tmp_path / "elsewhere" / "target_scores.jsonl")

    with pytest.raises(ValueError, match="outside output_dir"):
        _resolve_output_path(
            escaped,
            output_dir=str(output_dir),
            physical_output_root=tmp_path / "anchored",
        )
