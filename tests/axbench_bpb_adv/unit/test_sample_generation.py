"""Unit tests for the pure (model-free) parts of ``sample_generation``.

Covers Task 2.2 (selection + prompt retrieval) and Task 2.4 (YAML doc
assembly). The model-backed generation path is exercised separately in
``tests/axbench_bpb_adv/functional/test_generate_samples_flow.py``.

The fixtures here mirror the real ``MACRO_AGG_SCHEMA`` / ``CANDIDATES_SCHEMA`` /
``PER_CANDIDATE_AGG_SCHEMA`` column sets so the selection helpers run against the
exact row shapes the production aggregation emits. The representative-coefficient
assertions cross-check against ``aggregation._select_representative_coefficient``
so the generated YAML's representative coefficient equals the plot's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    _representative_family_key,
    _select_representative_coefficient,
)
from steering.eval.axbench_bpb_adv.candidate_metadata import (
    CandidateMetadata,
    FasterGcgFinalMetadata,
    GepaFinalMetadata,
    NoAdversaryControlMetadata,
    candidate_metadata_to_dict,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.sample_generation import (
    FamilyStrengthAxis,
    GenConcept,
    GenerationConfig,
    GenFamily,
    GenPrefix,
    RepresentativeAttack,
    _family_strength_axis,
    _load_sampling_metrics,
    build_family_yaml_doc,
    families_with_representative_coefficient,
    representative_attack_for_datapoint,
    select_concepts_and_prefixes,
)
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
)
from steering.refined_types import NonNegativeInt


def _strength_axis(
    by_prefix: dict[tuple[int, str], dict[int, float]],
    *,
    strength_unit: str | None = "counter_direction_magnitude",
) -> FamilyStrengthAxis:
    """Build a ``FamilyStrengthAxis`` for the pure ``build_family_yaml_doc`` tests.

    The orchestrator derives this from the family's mode-correct candidates; these
    unit tests pass it explicitly so the strength set (which (concept, prefix,
    strength_index) the YAML emits) and the per_candidate metrics join can be
    exercised independently.
    """

    return FamilyStrengthAxis(by_prefix=by_prefix, strength_unit=strength_unit)


def _macro_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str = "4.0",
    adversarial_method: str = "additive_direction",
    adversarial_strength: float = 0.0,
    strength_index: int = 0,
    context_signal_mode: str = "not_applicable",
    baseline_primary_margin: float | None = None,
    is_control: bool = False,
    primary_margin_y: float = 1.0,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": primary_margin_y,
        "baseline_primary_margin": baseline_primary_margin,
        "general_val_bpb": None,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": is_control,
    }


def _two_coefficient_macro_rows() -> list[dict[str, Any]]:
    """One family, two candidate coefficients, control rows at strength 0.

    coefficient 4.0 is closest to the helper's default fair Feature-DiffMean
    target score, so it wins representative selection. Each coefficient carries a
    control (strength 0) row and a non-control (strength 1) row.
    """

    rows: list[dict[str, Any]] = []
    for coefficient_str, baseline in (("2.0", 0.5), ("4.0", 0.9)):
        rows.append(
            _macro_row(
                coefficient_str=coefficient_str,
                adversarial_strength=0.0,
                strength_index=0,
                is_control=True,
                baseline_primary_margin=baseline,
            )
        )
        rows.append(
            _macro_row(
                coefficient_str=coefficient_str,
                adversarial_strength=0.4,
                strength_index=1,
                is_control=False,
                baseline_primary_margin=baseline,
            )
        )
    return rows


def _feature_diffmean_target_row(
    *,
    score: float = 0.8,
    mode: str = "positive",
    roundtrip_score: float | None = None,
) -> dict[str, Any]:
    method_name = "feature_diffmean_encdec" if mode == "positive" else "neg_feature_diffmean_encdec"
    row = _macro_row(
        mode=mode,
        method_name=method_name,
        target_model="encdec",
        coefficient_str="None",
        is_control=True,
        baseline_primary_margin=score,
        primary_margin_y=score,
    )
    if roundtrip_score is not None:
        row["primary_roundtrip_y"] = roundtrip_score
    return row


def _with_feature_diffmean_targets(
    rows: list[dict[str, Any]],
    *,
    positive_score: float = 0.8,
    negative_score: float = 0.8,
    positive_roundtrip_score: float | None = None,
    negative_roundtrip_score: float | None = None,
) -> list[dict[str, Any]]:
    modes = {str(row["mode"]) for row in rows}
    targets: list[dict[str, Any]] = []
    if "positive" in modes:
        targets.append(
            _feature_diffmean_target_row(
                score=positive_score,
                mode="positive",
                roundtrip_score=positive_roundtrip_score,
            )
        )
    if "negative" in modes:
        targets.append(
            _feature_diffmean_target_row(
                score=negative_score,
                mode="negative",
                roundtrip_score=negative_roundtrip_score,
            )
        )
    return [*rows, *targets]


def _legacy_per_candidate_row() -> dict[str, Any]:
    return {
        "concept_id": 1,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "4.0",
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0,
        "baseline_primary_margin": 0.9,
        "general_val_bpb": None,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 1,
        "n_avoid_output_tokens": 1,
        "n_rich_output_bytes": 1,
        "n_avoid_output_bytes": 1,
        "is_control": True,
    }


def _roundtrip_per_candidate_row() -> dict[str, Any]:
    return {
        "concept_id": 1,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "4.0",
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "mean_roundtrip_pre_act": 8.0,
        "mean_roundtrip_post_act": 4.0,
        "mean_roundtrip_percentile": 80.0,
        "primary_roundtrip_y": 80.0,
        "mean_control_roundtrip_pre_act": 8.0,
        "mean_control_roundtrip_post_act": 4.0,
        "mean_control_roundtrip_percentile": 80.0,
        "control_primary_roundtrip_y": 80.0,
        "general_val_bpb": None,
        "n_samples": 1,
        "n_generated_tokens": 3,
        "n_control_generated_tokens": 3,
        "max_tokens_rate": 0.0,
        "control_max_tokens_rate": 0.0,
        "is_control": True,
    }


def _roundtrip_macro_row() -> dict[str, Any]:
    row = _roundtrip_per_candidate_row()
    row.pop("concept_id")
    row["n_concepts"] = 1
    return row


def _write_legacy_aggregates(
    paths: AxBenchBpbAdvPaths,
    *,
    macro: bool = True,
    per_candidate: bool = True,
) -> None:
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    if macro:
        pq.write_table(
            pa.Table.from_pylist([_macro_row()], schema=MACRO_AGG_SCHEMA),
            paths.aggregates_dir() / "macro.parquet",
        )
    if per_candidate:
        pq.write_table(
            pa.Table.from_pylist(
                [_legacy_per_candidate_row()],
                schema=PER_CANDIDATE_AGG_SCHEMA,
            ),
            paths.aggregates_dir() / "per_candidate.parquet",
        )


def _write_roundtrip_aggregates(
    paths: AxBenchBpbAdvPaths,
    *,
    macro: bool = True,
    per_candidate: bool = True,
) -> None:
    paths.roundtrip_aggregates_dir().mkdir(parents=True, exist_ok=True)
    if macro:
        pq.write_table(
            pa.Table.from_pylist(
                [_roundtrip_macro_row()],
                schema=ROUNDTRIP_MACRO_AGG_SCHEMA,
            ),
            paths.roundtrip_aggregates_dir() / "macro.parquet",
        )
    if per_candidate:
        pq.write_table(
            pa.Table.from_pylist(
                [_roundtrip_per_candidate_row()],
                schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
            ),
            paths.roundtrip_aggregates_dir() / "per_candidate.parquet",
        )


def _sampling_metric_paths(tmp_path: Path) -> AxBenchBpbAdvPaths:
    return AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")


# --- aggregate loader metric selection ----------------------------------------


def test_load_sampling_metrics_roundtrip_ignores_complete_legacy_tree(
    tmp_path: Path,
    caplog: Any,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_legacy_aggregates(paths)
    _write_roundtrip_aggregates(paths)

    with caplog.at_level(logging.WARNING):
        metrics = _load_sampling_metrics(paths, steering_metric="roundtrip")

    assert metrics.representative_baseline_metric == "primary_roundtrip_y"
    assert metrics.primary_metric_field == "primary_roundtrip_y"
    assert metrics.primary_metric_output_key == "primary_roundtrip_y"
    assert metrics.macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
    assert "ignoring" in caplog.text
    assert "class-margin" in caplog.text


def test_load_sampling_metrics_class_margin_ignores_complete_roundtrip_tree(
    tmp_path: Path,
    caplog: Any,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_legacy_aggregates(paths)
    _write_roundtrip_aggregates(paths)

    with caplog.at_level(logging.WARNING):
        metrics = _load_sampling_metrics(paths, steering_metric="bpb_class_margin")

    assert metrics.representative_baseline_metric == "baseline_primary_margin"
    assert metrics.primary_metric_field == "primary_margin_y"
    assert metrics.primary_metric_output_key == "primary_margin_y"
    assert metrics.macro.schema.equals(MACRO_AGG_SCHEMA)
    assert "ignoring" in caplog.text
    assert "roundtrip" in caplog.text


def test_load_sampling_metrics_roundtrip_rejects_legacy_only_run(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_legacy_aggregates(paths)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _load_sampling_metrics(paths, steering_metric="roundtrip")
    message = str(exc_info.value)
    assert "metric mismatch" in message
    assert "steering_metric='roundtrip'" in message
    assert "only class-margin aggregates" in message
    assert "roundtrip_aggregates/macro.parquet" in message
    assert "roundtrip_aggregates/per_candidate.parquet" in message


def test_load_sampling_metrics_class_margin_rejects_roundtrip_only_run(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_roundtrip_aggregates(paths)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _load_sampling_metrics(paths, steering_metric="bpb_class_margin")
    message = str(exc_info.value)
    assert "metric mismatch" in message
    assert "steering_metric='bpb_class_margin'" in message
    assert "only roundtrip aggregates" in message
    assert "aggregates/macro.parquet" in message
    assert "aggregates/per_candidate.parquet" in message


def test_load_sampling_metrics_roundtrip_rejects_partial_selected_run(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_legacy_aggregates(paths)
    _write_roundtrip_aggregates(paths, macro=True, per_candidate=False)

    with pytest.raises(AxBenchBpbAdvError, match="incomplete roundtrip aggregates"):
        _load_sampling_metrics(paths, steering_metric="roundtrip")


def test_load_sampling_metrics_class_margin_rejects_partial_selected_run(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_roundtrip_aggregates(paths)
    _write_legacy_aggregates(paths, macro=False, per_candidate=True)

    with pytest.raises(AxBenchBpbAdvError, match="incomplete class-margin aggregates"):
        _load_sampling_metrics(paths, steering_metric="bpb_class_margin")


def test_load_sampling_metrics_roundtrip_ignores_partial_legacy_tree(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_roundtrip_aggregates(paths)
    _write_legacy_aggregates(paths, macro=True, per_candidate=False)

    metrics = _load_sampling_metrics(paths, steering_metric="roundtrip")

    assert metrics.primary_metric_field == "primary_roundtrip_y"
    assert metrics.macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)


def test_load_sampling_metrics_class_margin_ignores_partial_roundtrip_tree(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)
    _write_legacy_aggregates(paths)
    _write_roundtrip_aggregates(paths, macro=False, per_candidate=True)

    metrics = _load_sampling_metrics(paths, steering_metric="bpb_class_margin")

    assert metrics.primary_metric_field == "primary_margin_y"
    assert metrics.macro.schema.equals(MACRO_AGG_SCHEMA)


def test_load_sampling_metrics_roundtrip_requires_selected_artifacts_when_empty(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _load_sampling_metrics(paths, steering_metric="roundtrip")
    message = str(exc_info.value)
    assert "requires finished aggregate artifacts" in message
    assert "steering_metric='roundtrip'" in message
    assert "roundtrip_aggregates/macro.parquet" in message
    assert "roundtrip_aggregates/per_candidate.parquet" in message


def test_load_sampling_metrics_class_margin_requires_selected_artifacts_when_empty(
    tmp_path: Path,
) -> None:
    paths = _sampling_metric_paths(tmp_path)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _load_sampling_metrics(paths, steering_metric="bpb_class_margin")
    message = str(exc_info.value)
    assert "requires finished aggregate artifacts" in message
    assert "steering_metric='bpb_class_margin'" in message
    assert "aggregates/macro.parquet" in message
    assert "aggregates/per_candidate.parquet" in message


def test_load_sampling_metrics_rejects_invalid_metric(tmp_path: Path) -> None:
    paths = _sampling_metric_paths(tmp_path)

    with pytest.raises(AxBenchBpbAdvError, match="steering_metric"):
        _load_sampling_metrics(paths, steering_metric="loss_diff")


# --- Task 2.2: families_with_representative_coefficient ------------------------


def test_families_representative_coefficient_matches_plot_selection() -> None:
    rows = _with_feature_diffmean_targets(_two_coefficient_macro_rows())
    families = families_with_representative_coefficient(rows, target_model="decoder_only")
    assert len(families) == 1
    family = families[0]
    assert isinstance(family, GenFamily)
    # The representative coefficient equals what the plot's selector chooses.
    selected = _select_representative_coefficient(rows)
    key = _representative_family_key(rows[0])
    assert family.representative_coefficient == selected[key] == "4.0"
    assert family.mode == "positive"
    assert family.method_name == "diffmean_decoder_only"
    assert family.target_model == "decoder_only"
    assert family.adversarial_method == "additive_direction"
    assert family.context_signal_mode == "not_applicable"


def test_families_filters_to_target_model() -> None:
    rows = _with_feature_diffmean_targets(_two_coefficient_macro_rows())
    rows.append(
        _macro_row(
            method_name="diffmean_encdec",
            target_model="encdec",
            coefficient_str="4.0",
            is_control=True,
            baseline_primary_margin=0.9,
        )
    )
    families = families_with_representative_coefficient(rows, target_model="decoder_only")
    assert {family.target_model for family in families} == {"decoder_only"}


def test_families_distinct_per_family_key_and_sorted_deterministically() -> None:
    rows = _two_coefficient_macro_rows()
    # A second family (negative mode) with its own representative coefficient.
    rows.append(
        _macro_row(
            mode="negative",
            method_name="neg_diffmean_decoder_only",
            coefficient_str="4.0",
            is_control=True,
            baseline_primary_margin=0.7,
        )
    )
    rows = _with_feature_diffmean_targets(rows, negative_score=0.7)
    families = families_with_representative_coefficient(rows, target_model="decoder_only")
    keys = [
        (
            f.mode,
            f.method_name,
            f.target_model,
            f.adversarial_method,
            f.context_signal_mode,
        )
        for f in families
    ]
    # One GenFamily per distinct representative-family key.
    assert len(keys) == len(set(keys)) == 2
    # Deterministic sort (negative < positive lexicographically on mode first).
    assert keys == sorted(keys)


def test_families_encdec_target_model_fails_loud() -> None:
    # GAP 3: the SELECTION path rejects encoder-decoder up front, mirroring the
    # generation path's raise (encdec sampling is a deliberate follow-up). Only
    # the generation-path raise was covered before; this pins the selection-path
    # one so a regression that silently accepted target_model="encdec" here
    # (and emitted decoder-only-shaped GenFamily rows for an encdec run) is
    # caught. The raise is unconditional on target_model, so an empty row list
    # still trips it -- no encdec rows need to be present.
    with pytest.raises(AxBenchBpbAdvError, match="encoder-decoder"):
        families_with_representative_coefficient([], target_model="encdec")


def test_families_encdec_requires_explicit_validated_cluster_gate() -> None:
    rows = [
        _macro_row(
            target_model="encdec",
            method_name="feature_diffmean_encdec",
            coefficient_str="1.0",
            is_control=True,
            primary_margin_y=0.5,
            baseline_primary_margin=0.5,
        )
    ]
    families = families_with_representative_coefficient(
        rows,
        target_model="encdec",
        allow_encdec=True,
    )
    assert [family.method_name for family in families] == [
        "feature_diffmean_encdec"
    ]


def test_families_none_representative_coefficient_preserved() -> None:
    # A family with no usable control margin selects ``None`` (kept, not dropped).
    rows = [
        _macro_row(
            coefficient_str="2.0",
            is_control=False,
            baseline_primary_margin=None,
        )
    ]
    rows = _with_feature_diffmean_targets(rows)
    families = families_with_representative_coefficient(rows, target_model="decoder_only")
    assert len(families) == 1
    assert families[0].representative_coefficient is None


def test_families_roundtrip_representative_uses_primary_roundtrip_y() -> None:
    rows = _with_feature_diffmean_targets(
        [
            {
                **_macro_row(
                    coefficient_str="2.0",
                    is_control=True,
                    baseline_primary_margin=None,
                ),
                "primary_roundtrip_y": 70.0,
            },
            {
                **_macro_row(
                    coefficient_str="4.0",
                    is_control=True,
                    baseline_primary_margin=None,
                ),
                "primary_roundtrip_y": 85.0,
            },
        ],
        positive_roundtrip_score=82.0,
    )

    families = families_with_representative_coefficient(
        rows,
        target_model="decoder_only",
        baseline_metric="primary_roundtrip_y",
    )

    assert len(families) == 1
    assert families[0].representative_coefficient == "4.0"


def test_families_feature_diffmean_selector_uses_hidden_encdec_target() -> None:
    rows = _with_feature_diffmean_targets(
        _two_coefficient_macro_rows(),
        positive_score=0.6,
    )

    families = families_with_representative_coefficient(
        rows,
        target_model="decoder_only",
    )

    assert len(families) == 1
    assert families[0].representative_coefficient == "2.0"


# --- Task 2.2: select_concepts_and_prefixes -----------------------------------


def _candidate_row(
    *,
    concept_id: int,
    core_prefix_id: str,
    prefix_text: str,
    candidate_id: str,
    method_name: str = "diffmean_decoder_only",
    adversarial_method: str = "additive_direction",
    coefficient_str: str = "4.0",
    strength_index: int = 0,
    mode: str = "positive",
    attacked_prompt_tokens: list[int] | None = None,
    source_prompt_tokens: list[int] | None = None,
    objective_value: float | None = None,
    attacked_prompt_text: str | None = None,
    replacement_text: str = "",
    candidate_metadata_json: str | None = '{"candidate":"no_adversary_control"}',
) -> dict[str, Any]:
    tokens = attacked_prompt_tokens if attacked_prompt_tokens is not None else [0, 1, 2]
    source = source_prompt_tokens if source_prompt_tokens is not None else list(tokens)
    text = (
        attacked_prompt_text
        if attacked_prompt_text is not None
        else "".join(chr(int(token_id)) for token_id in tokens if int(token_id) != 0)
    )
    return {
        "candidate_id": candidate_id,
        "concept_id": concept_id,
        "core_prefix_id": core_prefix_id,
        "prefix_text": prefix_text,
        "method_name": method_name,
        "adversarial_method": adversarial_method,
        "coefficient_str": coefficient_str,
        "strength_index": strength_index,
        "mode": mode,
        "attacked_prompt_tokens": list(tokens),
        "source_prompt_tokens": list(source),
        "objective_value": objective_value,
        "attacked_prompt_text": text,
        "replacement_text": replacement_text,
        "candidate_metadata_json": candidate_metadata_json,
    }


def test_select_concepts_and_prefixes_first_n_sorted() -> None:
    rows = [
        _candidate_row(concept_id=2, core_prefix_id="cp_b", prefix_text="B", candidate_id="c1"),
        _candidate_row(concept_id=2, core_prefix_id="cp_a", prefix_text="A", candidate_id="c2"),
        _candidate_row(concept_id=1, core_prefix_id="cp_z", prefix_text="Z", candidate_id="c3"),
        _candidate_row(concept_id=0, core_prefix_id="cp_y", prefix_text="Y", candidate_id="c4"),
    ]
    concepts = select_concepts_and_prefixes(
        rows,
        n_concepts=2,
        n_prefixes_per_concept=1,
        concept_labels={0: "cats", 1: "dogs", 2: "cars"},
    )
    assert [c.concept_id for c in concepts] == [0, 1]
    assert isinstance(concepts[0], GenConcept)
    assert concepts[0].concept_label == "cats"
    assert concepts[1].concept_label == "dogs"
    # One prefix each (n_prefixes_per_concept=1), sorted by core_prefix_id.
    assert [p.core_prefix_id for p in concepts[0].prefixes] == ["cp_y"]
    assert isinstance(concepts[0].prefixes[0], GenPrefix)
    assert concepts[0].prefixes[0].prefix_text == "Y"


def test_select_concepts_and_prefixes_distinct_prefixes_sorted() -> None:
    # concept 5 has 3 distinct prefixes across many candidate rows (dups present).
    rows = [
        _candidate_row(concept_id=5, core_prefix_id="cp_3", prefix_text="P3", candidate_id="c1"),
        _candidate_row(concept_id=5, core_prefix_id="cp_1", prefix_text="P1", candidate_id="c2"),
        _candidate_row(
            concept_id=5,
            core_prefix_id="cp_1",
            prefix_text="P1",
            candidate_id="c3",
            strength_index=1,
        ),
        _candidate_row(concept_id=5, core_prefix_id="cp_2", prefix_text="P2", candidate_id="c4"),
    ]
    concepts = select_concepts_and_prefixes(
        rows,
        n_concepts=1,
        n_prefixes_per_concept=2,
        concept_labels={5: "music"},
    )
    assert len(concepts) == 1
    # First 2 distinct (core_prefix_id, prefix_text) sorted by core_prefix_id.
    assert [p.core_prefix_id for p in concepts[0].prefixes] == ["cp_1", "cp_2"]
    assert [p.prefix_text for p in concepts[0].prefixes] == ["P1", "P2"]


def test_select_concepts_and_prefixes_logs_skips(caplog: Any) -> None:
    rows = [
        _candidate_row(concept_id=0, core_prefix_id="cp_a", prefix_text="A", candidate_id="c1"),
        _candidate_row(concept_id=1, core_prefix_id="cp_b", prefix_text="B", candidate_id="c2"),
        _candidate_row(concept_id=2, core_prefix_id="cp_c", prefix_text="C", candidate_id="c3"),
    ]
    with caplog.at_level(logging.INFO):
        concepts = select_concepts_and_prefixes(
            rows,
            n_concepts=1,
            n_prefixes_per_concept=1,
            concept_labels={0: "cats", 1: "dogs", 2: "cars"},
        )
    assert len(concepts) == 1
    # 3 concepts available, 1 chosen -> the skip must be visible (loguru routes
    # through the standard logging propagate handler caplog captures). Match the
    # REAL message substring (not bare "3"/"1", which could match any number in
    # any log line) so the assertion has teeth: it pins that the available/chosen
    # truncation counts are logged by THIS helper, not coincidentally present.
    text = caplog.text
    assert "3 concepts available, 1 chosen (n_concepts=1)" in text


def test_select_concepts_missing_label_falls_back_or_raises() -> None:
    # A concept_id absent from concept_labels must not silently produce a bogus
    # label; the helper raises a clear AxBenchBpbAdvError.
    rows = [
        _candidate_row(concept_id=7, core_prefix_id="cp_a", prefix_text="A", candidate_id="c1"),
    ]
    with pytest.raises(AxBenchBpbAdvError):
        select_concepts_and_prefixes(
            rows,
            n_concepts=1,
            n_prefixes_per_concept=1,
            concept_labels={},
        )


# --- Task 2.2: representative_attack_for_datapoint (issue #543) ----------------


def _call_repr(rows: list[dict[str, Any]], **overrides: Any) -> RepresentativeAttack:
    kwargs: dict[str, Any] = {
        "concept_id": 1,
        "core_prefix_id": "cp_a",
        "method_name": "diffmean_decoder_only",
        "adversarial_method": "faster_gcg",
        "coefficient_str": "4.0",
        "strength_index": 1,
        "mode": "positive",
    }
    kwargs.update(overrides)
    return representative_attack_for_datapoint(rows, **kwargs)


def test_representative_attack_additive_returns_attacked_equals_source() -> None:
    # On an additive datapoint the attacked prompt equals the source prompt.
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="additive_direction",
            strength_index=0,
            attacked_prompt_tokens=[0, 5, 6],
            source_prompt_tokens=[0, 5, 6],
            objective_value=None,
        ),
    ]
    attack = _call_repr(
        rows,
        adversarial_method="additive_direction",
        strength_index=0,
    )
    assert attack.prompt_tokens == [0, 5, 6]
    assert attack.objective_value is None
    assert rows[0]["source_prompt_tokens"] == attack.prompt_tokens


def test_representative_attack_picks_min_objective_value() -> None:
    # The SELECTOR's general contract: three candidates of ONE datapoint with
    # distinct objective_value and shuffled candidate_ids -> the lowest g wins.
    # (Under universal triggers a datapoint's candidates actually SHARE one g --
    # pinned by test_representative_attack_shared_objective_reduces_to_candidate_id
    # below -- but the selector must still pick the minimum when fed distinct values,
    # so this contract is exercised independently of how the optimizer populates it.)
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c3",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 71, 72],
            objective_value=-0.1,
            replacement_text="GH",
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 88, 89],
            objective_value=-0.9,
            replacement_text="XY",
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c2",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 50, 51],
            objective_value=-0.5,
            replacement_text="MN",
        ),
    ]
    attack = _call_repr(rows)
    # TEETH: the strongest (lowest g = -0.9) candidate wins.
    assert attack.objective_value == -0.9
    assert attack.prompt_tokens == [0, 88, 89]
    assert attack.replacement_text == "XY"
    assert attack.candidate_id == "c1"


def test_representative_attack_objective_strength_beats_candidate_id() -> None:
    # The strongest attack has the LARGEST candidate_id -> proves objective_value,
    # not candidate_id, drives the choice (a min-candidate_id selector would fail).
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="aaa",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 9, 9],
            objective_value=-0.2,
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="zzz",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 3, 4],
            objective_value=-0.8,
        ),
    ]
    attack = _call_repr(rows)
    assert attack.candidate_id == "zzz"
    assert attack.prompt_tokens == [0, 3, 4]


def test_representative_attack_objective_tie_breaks_on_candidate_id() -> None:
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c2",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 9, 9],
            objective_value=-0.5,
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 1, 2],
            objective_value=-0.5,
        ),
    ]
    attack = _call_repr(rows)
    assert attack.candidate_id == "c1"
    assert attack.prompt_tokens == [0, 1, 2]


def test_representative_attack_shared_objective_reduces_to_candidate_id() -> None:
    # The ACTUAL universal-trigger production regime: a concept's prefixes share ONE
    # trigger, so the per-pair candidates of a datapoint carry IDENTICAL
    # attacked_prompt_tokens / replacement_text / objective_value (the concept-level
    # group-aggregated g) and differ ONLY in candidate_id. The representative is then
    # well-defined: the objectives tie, so the candidate_id tie-break decides (min),
    # and -- because every candidate yields the same prompt/objective -- the surfaced
    # attacked prompt is independent of the tie-break anyway. TEETH: the shared g is
    # non-None, so this exercises the (False, g, candidate_id) sort path (NOT the
    # all-None reduction below) and confirms it stays total/crash-free under ties.
    shared_objective = -0.5
    shared_tokens = [0, 77, 78]
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c2",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=list(shared_tokens),
            objective_value=shared_objective,
            replacement_text="GG",
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=list(shared_tokens),
            objective_value=shared_objective,
            replacement_text="GG",
        ),
    ]
    attack = _call_repr(rows)
    # candidate_id breaks the (shared-objective) tie deterministically.
    assert attack.candidate_id == "c1"
    assert attack.objective_value == shared_objective
    # The surfaced prompt/replacement are the shared trigger's, identical to either
    # candidate -- the choice is well-defined regardless of which tied row wins.
    assert attack.prompt_tokens == shared_tokens
    assert attack.replacement_text == "GG"


def test_representative_attack_all_none_falls_back_to_candidate_id() -> None:
    # Additive/control: every objective_value is None -> reduces to min(candidate_id)
    # (the #538 behavior that keeps additive byte-identical).
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c2",
            adversarial_method="additive_direction",
            strength_index=0,
            attacked_prompt_tokens=[0, 9, 9],
            objective_value=None,
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="additive_direction",
            strength_index=0,
            attacked_prompt_tokens=[0, 1, 2],
            objective_value=None,
        ),
    ]
    attack = _call_repr(
        rows,
        adversarial_method="additive_direction",
        strength_index=0,
    )
    assert attack.candidate_id == "c1"
    assert attack.prompt_tokens == [0, 1, 2]


def test_representative_attack_mixed_none_is_none_safe_and_prefers_real() -> None:
    # Mixed None/real is impossible in production (k fixed -> uniform), but the sort
    # key must never raise on None < float; it prefers the real attack.
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 9, 9],
            objective_value=None,
        ),
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c2",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 3, 4],
            objective_value=0.7,
        ),
    ]
    attack = _call_repr(rows)
    assert attack.objective_value == 0.7
    assert attack.prompt_tokens == [0, 3, 4]


def test_representative_attack_carries_text_and_objective_fields() -> None:
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=2,
            attacked_prompt_tokens=[0, 65, 66],
            objective_value=-0.33,
            attacked_prompt_text="ABC",
            replacement_text="BC",
        ),
    ]
    attack = _call_repr(rows, strength_index=2)
    assert isinstance(attack, RepresentativeAttack)
    assert attack.attacked_prompt_text == "ABC"
    assert attack.replacement_text == "BC"
    assert attack.objective_value == -0.33
    assert attack.candidate_id == "c1"


def test_representative_attack_parses_candidate_metadata_into_optimization() -> None:
    # The chosen candidate's candidate_metadata_json (GCG step counts) is parsed
    # into typed metadata so the YAML can surface how hard the search worked.
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            objective_value=-0.5,
            candidate_metadata_json=(
                '{"candidate":"faster_gcg_final","max_steps":4,'
                '"accepted_steps":2,"visited_sequences":9}'
            ),
        ),
    ]
    attack = _call_repr(rows)
    assert attack.optimization == FasterGcgFinalMetadata(
        max_steps=NonNegativeInt.parse(4),
        accepted_steps=NonNegativeInt.parse(2),
        visited_sequences=NonNegativeInt.parse(9),
    )
    assert candidate_metadata_to_dict(attack.optimization) == {
        "candidate": "faster_gcg_final",
        "max_steps": 4,
        "accepted_steps": 2,
        "visited_sequences": 9,
    }


def test_representative_attack_missing_or_corrupt_metadata_raises() -> None:
    for raw in (None, "", "{}"):
        rows = [
            _candidate_row(
                concept_id=1,
                core_prefix_id="cp_a",
                prefix_text="A",
                candidate_id="c1",
                adversarial_method="faster_gcg",
                strength_index=1,
                objective_value=-0.5,
                candidate_metadata_json=raw,
            ),
        ]
        with pytest.raises(AxBenchBpbAdvError, match="candidate_metadata_json"):
            _call_repr(rows)


def test_representative_attack_corrupt_metadata_raises() -> None:
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            objective_value=-0.5,
            candidate_metadata_json=(
                '{"candidate":"faster_gcg_final","max_steps":"4",'
                '"accepted_steps":2,"visited_sequences":9}'
            ),
        ),
    ]

    with pytest.raises(AxBenchBpbAdvError, match="'max_steps'"):
        _call_repr(rows)


def test_representative_attack_nonfinite_objective_passthrough() -> None:
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
            attacked_prompt_tokens=[0, 1, 2],
            objective_value=float("-inf"),
        ),
    ]
    attack = _call_repr(rows)
    assert attack.objective_value == float("-inf")


def test_representative_attack_missing_datapoint_raises() -> None:
    rows = [
        _candidate_row(
            concept_id=1,
            core_prefix_id="cp_a",
            prefix_text="A",
            candidate_id="c1",
            adversarial_method="faster_gcg",
            strength_index=1,
        ),
    ]
    with pytest.raises(AxBenchBpbAdvError):
        _call_repr(rows, concept_id=999)


# --- Task 2.4: build_family_yaml_doc ------------------------------------------


def _per_candidate_row(
    *,
    concept_id: int,
    coefficient_str: str = "4.0",
    adversarial_method: str = "additive_direction",
    strength_index: int,
    adversarial_strength: float,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    primary_margin_y: float = 1.0,
    general_val_bpb: float | None = None,
    is_control: bool = False,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": primary_margin_y,
        "baseline_primary_margin": None,
        "general_val_bpb": general_val_bpb,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 1,
        "n_avoid_output_tokens": 1,
        "n_rich_output_bytes": 1,
        "n_avoid_output_bytes": 1,
        "is_control": is_control,
    }


def _family() -> GenFamily:
    return GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="additive_direction",
        context_signal_mode="not_applicable",
        representative_coefficient="4.0",
    )


def test_build_family_yaml_doc_structure_and_join() -> None:
    family = _family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            strength_index=0,
            adversarial_strength=0.0,
            primary_margin_y=1.5,
            general_val_bpb=0.8,
            is_control=True,
        ),
        _per_candidate_row(
            concept_id=1,
            strength_index=1,
            adversarial_strength=0.4,
            primary_margin_y=0.9,
            general_val_bpb=None,
            is_control=False,
        ),
    ]
    completions_by_key = {
        (1, "cp_a", 0): ["c-control-0", "c-control-1"],
        (1, "cp_a", 1): ["c-strength-0", "c-strength-1"],
    }
    # The strength axis (from the candidates) supplies the strength set + the
    # reported adversarial_strength per (concept, prefix).
    strength_axis = _strength_axis({(1, "cp_a"): {0: 0.0, 1: 0.4}})
    sampling = {"n_completions": 2, "temperature": 0.7, "top_k": None, "max_new_tokens": 8}
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key=completions_by_key,
        attack_by_key={},
        sampling=sampling,
        scatter_baseline_x="baseline_loss",
    )
    # Top-level family metadata.
    assert doc["family"] == _representative_family_key_dict(family)
    assert doc["mode"] == "positive"
    assert doc["method_name"] == "diffmean_decoder_only"
    assert doc["target_model"] == "decoder_only"
    assert doc["adversarial_method"] == "additive_direction"
    assert doc["representative_coefficient"] == "4.0"
    assert doc["scatter_baseline_x"] == "baseline_loss"
    # strength_unit is read from the per_candidate rows, not hardcoded.
    assert doc["strength_unit"] == "counter_direction_magnitude"
    assert doc["sampling"] == sampling
    # Nested concept -> prefix -> datapoints.
    assert len(doc["concepts"]) == 1
    concept_doc = doc["concepts"][0]
    assert concept_doc["concept_id"] == 1
    assert concept_doc["concept_label"] == "dogs"
    prefix_doc = concept_doc["prefixes"][0]
    assert prefix_doc["core_prefix_id"] == "cp_a"
    assert prefix_doc["prefix_text"] == "A"
    datapoints = prefix_doc["datapoints"]
    # Sorted distinct strength_index for the family+coefficient.
    assert [dp["strength_index"] for dp in datapoints] == [0, 1]
    # Metrics joined per datapoint.
    assert datapoints[0]["adversarial_strength"] == 0.0
    assert datapoints[0]["is_control"] is True
    assert datapoints[0]["primary_margin_y"] == 1.5
    assert datapoints[0]["general_val_bpb"] == 0.8
    assert datapoints[0]["completions"] == ["c-control-0", "c-control-1"]
    # Null general_val_bpb tolerated.
    assert datapoints[1]["general_val_bpb"] is None
    assert datapoints[1]["primary_margin_y"] == 0.9
    assert datapoints[1]["completions"] == ["c-strength-0", "c-strength-1"]


def test_build_family_yaml_doc_can_emit_roundtrip_primary_metric() -> None:
    family = _family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    row = _per_candidate_row(
        concept_id=1,
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    row.pop("primary_margin_y")
    row["primary_roundtrip_y"] = 77.0

    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=_strength_axis({(1, "cp_a"): {0: 0.0}}),
        per_candidate_rows=[row],
        completions_by_key={(1, "cp_a", 0): ["completion"]},
        attack_by_key={},
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
        primary_metric_field="primary_roundtrip_y",
        primary_metric_output_key="primary_roundtrip_y",
    )

    datapoint = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    assert datapoint["primary_roundtrip_y"] == 77.0
    assert "primary_margin_y" not in datapoint


def _representative_family_key_dict(family: GenFamily) -> str:
    # The YAML's top-level ``family`` is a human-readable identity string; the
    # test pins it via the same fields used to key the family so it stays stable.
    return (
        f"{family.mode}|{family.method_name}|{family.target_model}|"
        f"{family.adversarial_method}|{family.context_signal_mode}"
    )


def test_build_family_yaml_doc_only_representative_coefficient_rows() -> None:
    # per_candidate has rows at TWO coefficients; only the representative (4.0)
    # contributes datapoints (the join is filtered to representative_coefficient).
    family = _family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            coefficient_str="4.0",
            strength_index=0,
            adversarial_strength=0.0,
            is_control=True,
        ),
        # A different coefficient that must NOT leak into the datapoints.
        _per_candidate_row(
            concept_id=1,
            coefficient_str="2.0",
            strength_index=7,
            adversarial_strength=0.0,
            is_control=True,
        ),
    ]
    completions_by_key = {(1, "cp_a", 0): ["x"]}
    # The candidate strength axis carries ONLY strength 0 (the representative
    # coefficient's candidate). The per_candidate "2.0" row must not leak into the
    # metrics join (filtered to the representative coefficient).
    strength_axis = _strength_axis({(1, "cp_a"): {0: 0.0}})
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key=completions_by_key,
        attack_by_key={},
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert [dp["strength_index"] for dp in datapoints] == [0]


def test_build_family_yaml_doc_missing_completion_raises() -> None:
    # A datapoint with no completions in completions_by_key is a programming
    # error (the orchestrator must have generated them) -> raise loudly.
    family = _family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            strength_index=0,
            adversarial_strength=0.0,
            is_control=True,
        ),
    ]
    with pytest.raises(AxBenchBpbAdvError):
        build_family_yaml_doc(
            family=family,
            concepts=concepts,
            strength_axis=_strength_axis({(1, "cp_a"): {0: 0.0}}),
            per_candidate_rows=per_candidate_rows,
            completions_by_key={},  # missing the (1, "cp_a", 0) key
            attack_by_key={},
            sampling={"n_completions": 1},
            scatter_baseline_x="baseline_loss",
        )


def test_build_family_yaml_doc_missing_metrics_raises() -> None:
    # Review item (A.2): an emitted (concept, strength_index) datapoint whose
    # core metrics row is ABSENT from per_candidate must FAIL LOUD -- symmetric
    # with the missing-completion raise -- instead of silently emitting null
    # is_control / primary_margin_y beside a non-null adversarial_strength.
    #
    # Setup: strength_to_meta captures strength_index 0 from the per_candidate
    # rows of THIS concept, but the selected concept under generation is a
    # DIFFERENT concept (concept_id=2) that has no metrics row -> the join misses
    # and the raise must fire. The completion IS present, so this isolates the
    # missing-metrics path from the missing-completion one.
    family = _family()
    concepts = [
        GenConcept(
            concept_id=2,
            concept_label="cars",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        # concept 1 has a metrics row, but concept 2 (the one being emitted) does
        # NOT -> the per_candidate metrics join misses and must raise.
        _per_candidate_row(
            concept_id=1,
            strength_index=0,
            adversarial_strength=0.0,
            is_control=True,
        ),
    ]
    completions_by_key = {(2, "cp_a", 0): ["only-completion"]}
    # The candidate strength axis emits concept 2 at strength 0 (its completion is
    # present), isolating the missing-metrics path.
    strength_axis = _strength_axis({(2, "cp_a"): {0: 0.0}})
    with pytest.raises(AxBenchBpbAdvError, match="metrics"):
        build_family_yaml_doc(
            family=family,
            concepts=concepts,
            strength_axis=strength_axis,
            per_candidate_rows=per_candidate_rows,
            completions_by_key=completions_by_key,
            attack_by_key={},
            sampling={"n_completions": 1},
            scatter_baseline_x="baseline_loss",
        )


def test_build_family_yaml_doc_null_metrics_emit_none() -> None:
    # F10: the core metrics columns are nullable (pyarrow default). A null
    # primary_margin_y / is_control cell must emit None (symmetric with the
    # existing general_val_bpb null-guard), NOT crash on float(None)/bool(None) and
    # NOT silently coerce a null is_control to False.
    family = _family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    metrics_row = _per_candidate_row(
        concept_id=1,
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    # Null out the core metrics cells (a degenerate/future producer could).
    metrics_row["primary_margin_y"] = None
    metrics_row["is_control"] = None
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=_strength_axis({(1, "cp_a"): {0: 0.0}}),
        per_candidate_rows=[metrics_row],
        completions_by_key={(1, "cp_a", 0): ["x"]},
        attack_by_key={},
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    datapoint = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    # Null cells -> None (not a crash, not a silent False).
    assert datapoint["primary_margin_y"] is None
    assert datapoint["is_control"] is None
    # The (non-null) adversarial_strength still comes through from the axis.
    assert datapoint["adversarial_strength"] == 0.0


# --- Task 2.4: build_family_yaml_doc metrics-join filter clauses --------------


def test_build_family_yaml_doc_filters_metrics_by_adversarial_method() -> None:
    # GAP A.4: a per_candidate row that matches the family on everything EXCEPT
    # adversarial_method must NOT leak into the emitted metrics. adversarial_method
    # participates in the metrics-join identity (the pre-key filter clause
    # sample_generation ~:474 AND _datapoint_metrics_key together), so the wrong
    # row neither passes the filter nor shares the right row's lookup key. TEETH:
    # this fails RED if adversarial_method is dropped from the join identity (e.g.
    # removed from _datapoint_metrics_key) -- the wrong-method sentinel row then
    # collides and overwrites the real margin (verified by mutation).
    family = _family()  # adversarial_method == "additive_direction"
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    # The matching row (additive_direction) carries the real metric; a wrong-method
    # row (faster_gcg) shares the datapoint key but carries a sentinel margin.
    matching_row = _per_candidate_row(
        concept_id=1,
        adversarial_method="additive_direction",
        strength_index=0,
        adversarial_strength=0.0,
        primary_margin_y=1.5,
        is_control=True,
    )
    wrong_method_row = _per_candidate_row(
        concept_id=1,
        adversarial_method="faster_gcg",
        strength_index=0,
        adversarial_strength=0.0,
        primary_margin_y=-99.0,
        is_control=False,
    )
    # Order the leaking row LAST so, absent the filter, it would win the dict
    # assignment for the shared metrics key (teeth: the assertion would flip).
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=_strength_axis({(1, "cp_a"): {0: 0.0}}),
        per_candidate_rows=[matching_row, wrong_method_row],
        completions_by_key={(1, "cp_a", 0): ["x"]},
        attack_by_key={},
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    datapoint = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    # The additive_direction row won; the faster_gcg sentinel never leaked in.
    assert datapoint["primary_margin_y"] == 1.5
    assert datapoint["is_control"] is True


# --- _family_strength_axis: single-unit return + multi-unit raise -------------


def _axis_candidate_row(
    *,
    concept_id: int = 1,
    core_prefix_id: str = "cp_a",
    strength_index: int,
    adversarial_strength: float,
    strength_unit: str,
) -> dict[str, Any]:
    """A candidate row carrying just the fields ``_family_strength_axis`` reads."""

    return {
        "concept_id": concept_id,
        "core_prefix_id": core_prefix_id,
        "strength_index": strength_index,
        "adversarial_strength": adversarial_strength,
        "strength_unit": strength_unit,
    }


def test_family_strength_axis_single_unit_returns_that_unit() -> None:
    # GAP A.1 (positive): candidates that all carry ONE strength_unit -> the axis
    # reports exactly that unit (and the per-prefix strength map round-trips).
    family = _family()
    rows = [
        _axis_candidate_row(
            strength_index=0,
            adversarial_strength=0.0,
            strength_unit="counter_direction_magnitude",
        ),
        _axis_candidate_row(
            strength_index=1,
            adversarial_strength=0.4,
            strength_unit="counter_direction_magnitude",
        ),
    ]
    axis = _family_strength_axis(rows, family=family)
    assert axis.strength_unit == "counter_direction_magnitude"
    assert axis.points_for(1, "cp_a") == {0: 0.0, 1: 0.4}


def test_family_strength_axis_multiple_units_raises() -> None:
    # GAP A.1 (raise): two candidate rows of the SAME family carrying DIFFERENT
    # strength_units must trip a loud raise -- a family must sweep ONE axis. The
    # match pins the "multiple strength_units" message so an inverted/deleted
    # guard (silently picking one unit) turns this RED.
    family = _family()
    rows = [
        _axis_candidate_row(
            strength_index=0,
            adversarial_strength=0.0,
            strength_unit="counter_direction_magnitude",
        ),
        _axis_candidate_row(
            strength_index=1,
            adversarial_strength=1.0,
            strength_unit="optimized_tokens",
        ),
    ]
    with pytest.raises(AxBenchBpbAdvError, match="multiple strength_units"):
        _family_strength_axis(rows, family=family)


# --- GenerationConfig.sampling_dict + families_with_representative_coefficient -


def test_generation_sampling_dict_top_k_none_vs_int_echoed() -> None:
    # GAP A.8: sampling_dict() echoes top_k faithfully -- None stays None (NOT
    # coerced to int(None) crash, NOT silently defaulted), and an int is echoed
    # as an int.
    none_dict = GenerationConfig(top_k=None, n_completions=3, base_seed=7).sampling_dict()
    assert none_dict["top_k"] is None
    assert none_dict["n_completions"] == 3
    assert none_dict["base_seed"] == 7

    int_dict = GenerationConfig(top_k=42).sampling_dict()
    assert int_dict["top_k"] == 42
    assert isinstance(int_dict["top_k"], int)


def test_families_with_representative_coefficient_empty_returns_empty() -> None:
    # GAP A.8: an empty macro list for a supported (decoder_only) target yields an
    # empty family list -- it must NOT raise (only the encdec target raises) and
    # must NOT fabricate families.
    assert families_with_representative_coefficient([], target_model="decoder_only") == []


# --- Task 2.4 (issue #543): build_family_yaml_doc prompt-token field emission ---


def _gcg_family() -> GenFamily:
    return GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="faster_gcg",
        context_signal_mode="fixed_context",
        representative_coefficient="4.0",
    )


def _attack(
    *,
    tokens: list[int],
    text: str,
    replacement: str,
    objective: float | None,
    optimization: CandidateMetadata | None = None,
) -> RepresentativeAttack:
    return RepresentativeAttack(
        prompt_tokens=list(tokens),
        attacked_prompt_text=text,
        replacement_text=replacement,
        objective_value=objective,
        candidate_id="cid",
        optimization=(optimization if optimization is not None else NoAdversaryControlMetadata()),
    )


def test_build_family_yaml_doc_additive_omits_prompt_fields() -> None:
    family = _family()  # additive_direction
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            strength_index=0,
            adversarial_strength=0.0,
            primary_margin_y=1.5,
            is_control=True,
        ),
    ]
    strength_axis = FamilyStrengthAxis(
        by_prefix={(1, "cp_a"): {0: 0.0}},
        strength_unit="counter_direction_magnitude",
    )
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key={(1, "cp_a", 0): ["x"]},
        attack_by_key={
            (1, "cp_a", 0): _attack(tokens=[0, 5], text="A", replacement="", objective=None)
        },
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    dp = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    # EXACTLY the six pre-#543 keys, in order -> additive byte-identity.
    assert list(dp.keys()) == [
        "strength_index",
        "adversarial_strength",
        "is_control",
        "primary_margin_y",
        "general_val_bpb",
        "completions",
    ]


def test_build_family_yaml_doc_prompt_token_emits_three_fields() -> None:
    family = _gcg_family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            coefficient_str="4.0",
            adversarial_method="faster_gcg",
            strength_index=2,
            adversarial_strength=2.0,
            primary_margin_y=0.4,
            is_control=False,
        ),
    ]
    strength_axis = FamilyStrengthAxis(
        by_prefix={(1, "cp_a"): {2: 2.0}}, strength_unit="optimized_tokens"
    )
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key={(1, "cp_a", 2): ["x"]},
        attack_by_key={
            (1, "cp_a", 2): _attack(
                tokens=[0, 9],
                text="ATTACKED",
                replacement="ED",
                objective=-0.7,
                optimization=FasterGcgFinalMetadata(
                    max_steps=NonNegativeInt.parse(8),
                    accepted_steps=NonNegativeInt.parse(3),
                    visited_sequences=NonNegativeInt.parse(17),
                ),
            )
        },
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    dp = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    assert list(dp.keys()) == [
        "strength_index",
        "adversarial_strength",
        "attacked_prompt_text",
        "replacement_text",
        "objective_value",
        "optimization",
        "is_control",
        "primary_margin_y",
        "general_val_bpb",
        "completions",
    ]
    assert dp["attacked_prompt_text"] == "ATTACKED"
    assert dp["replacement_text"] == "ED"
    assert dp["objective_value"] == -0.7
    # The representative candidate's GCG optimization metadata is surfaced verbatim.
    assert dp["optimization"]["max_steps"] == 8
    assert dp["optimization"]["accepted_steps"] == 3


def test_build_family_yaml_doc_k0_control_emits_null_objective() -> None:
    family = _gcg_family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            coefficient_str="4.0",
            adversarial_method="faster_gcg",
            strength_index=0,
            adversarial_strength=0.0,
            primary_margin_y=0.9,
            is_control=True,
        ),
    ]
    strength_axis = FamilyStrengthAxis(
        by_prefix={(1, "cp_a"): {0: 0.0}}, strength_unit="optimized_tokens"
    )
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key={(1, "cp_a", 0): ["x"]},
        attack_by_key={
            (1, "cp_a", 0): _attack(
                tokens=[0, 5, 6],
                text="A",
                replacement="",
                objective=None,
                optimization=NoAdversaryControlMetadata(),
            )
        },
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    dp = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    assert dp["objective_value"] is None
    assert dp["replacement_text"] == ""
    assert dp["optimization"] == {"candidate": "no_adversary_control"}


def test_build_family_yaml_doc_prompt_token_missing_attack_raises() -> None:
    family = _gcg_family()
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            coefficient_str="4.0",
            adversarial_method="faster_gcg",
            strength_index=2,
            adversarial_strength=2.0,
            primary_margin_y=0.4,
            is_control=False,
        ),
    ]
    strength_axis = FamilyStrengthAxis(
        by_prefix={(1, "cp_a"): {2: 2.0}}, strength_unit="optimized_tokens"
    )
    with pytest.raises(AxBenchBpbAdvError):
        build_family_yaml_doc(
            family=family,
            concepts=concepts,
            strength_axis=strength_axis,
            per_candidate_rows=per_candidate_rows,
            completions_by_key={(1, "cp_a", 2): ["x"]},
            attack_by_key={},  # missing the (1, "cp_a", 2) entry
            sampling={"n_completions": 1},
            scatter_baseline_x="baseline_loss",
        )


def test_prompt_token_adversarial_methods_set_contents() -> None:
    # TEETH on the membership the emission branches on: both prompt-token attacks
    # (faster_gcg, gepa) MUST be present and additive_direction MUST NOT -- a drop
    # of "gepa" (which records no objective) would silently stop surfacing its
    # attacked prompts; including additive would break additive byte-identity.
    from steering.eval.axbench_bpb_adv.sample_generation import (
        PROMPT_TOKEN_ADVERSARIAL_METHODS,
    )

    assert PROMPT_TOKEN_ADVERSARIAL_METHODS == frozenset({"faster_gcg", "gepa"})


def test_build_family_yaml_doc_gepa_emits_fields_with_null_objective() -> None:
    # GEPA is a prompt-token attack but records NO objective_value (optimize.py),
    # so its representative falls back to candidate_id and objective_value is null --
    # yet it STILL surfaces attacked_prompt_text/replacement_text via the SAME
    # emission branch as faster_gcg, in the SAME field order. This is the only test
    # that drives a `gepa` family through build_family_yaml_doc.
    family = GenFamily(
        mode="positive",
        method_name="diffmean_decoder_only",
        target_model="decoder_only",
        adversarial_method="gepa",
        context_signal_mode="fixed_context",
        representative_coefficient="4.0",
    )
    concepts = [
        GenConcept(
            concept_id=1,
            concept_label="dogs",
            prefixes=[GenPrefix(core_prefix_id="cp_a", prefix_text="A")],
        )
    ]
    per_candidate_rows = [
        _per_candidate_row(
            concept_id=1,
            coefficient_str="4.0",
            adversarial_method="gepa",
            strength_index=1,
            adversarial_strength=8.0,
            primary_margin_y=0.4,
            is_control=False,
        ),
    ]
    strength_axis = FamilyStrengthAxis(
        by_prefix={(1, "cp_a"): {1: 8.0}}, strength_unit="metric_calls"
    )
    doc = build_family_yaml_doc(
        family=family,
        concepts=concepts,
        strength_axis=strength_axis,
        per_candidate_rows=per_candidate_rows,
        completions_by_key={(1, "cp_a", 1): ["x"]},
        attack_by_key={
            (1, "cp_a", 1): _attack(
                tokens=[0, 9],
                text="GEPA-ATTACKED",
                replacement="ED",
                objective=None,
                optimization=GepaFinalMetadata(),
            )
        },
        sampling={"n_completions": 1},
        scatter_baseline_x="baseline_loss",
    )
    dp = doc["concepts"][0]["prefixes"][0]["datapoints"][0]
    assert list(dp.keys()) == [
        "strength_index",
        "adversarial_strength",
        "attacked_prompt_text",
        "replacement_text",
        "objective_value",
        "optimization",
        "is_control",
        "primary_margin_y",
        "general_val_bpb",
        "completions",
    ]
    assert dp["attacked_prompt_text"] == "GEPA-ATTACKED"
    assert dp["replacement_text"] == "ED"
    assert dp["objective_value"] is None  # gepa records no objective
    assert dp["optimization"] == {"candidate": "gepa_final"}
