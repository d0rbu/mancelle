from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb_adv.aggregation import (
    ROUNDTRIP_SCATTER_PLOT_NAME,
    _filter_rows_to_representative_coefficient,
    _join_roundtrip_general_val_bpb,
    _scatter_baseline_x,
    aggregate_roundtrip_adv_scores,
    compute_roundtrip_macro_aggregation,
    compute_roundtrip_per_candidate_aggregation,
    plot_roundtrip_adv_summaries,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    GENERAL_VAL_BPB_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)


def _score_row(
    *,
    concept_id: int = 1,
    mode: str = "positive",
    coefficient_str: str = "1.0",
    strength: float = 0.0,
    strength_index: int = 0,
    percentile: float = 80.0,
    control_percentile: float | None = None,
    loss_rich: float = 1.0,
    loss_avoid: float = 3.0,
    sample_index: int = 0,
    is_control: bool = True,
    stop_reason: str = "max_tokens",
    control_stop_reason: str = "eos",
) -> dict[str, Any]:
    prompt_tokens = [97, 98]
    control_percentile = percentile if control_percentile is None else control_percentile
    return {
        "candidate_id": f"c-{concept_id}-{mode}-{coefficient_str}-{strength_index}-{sample_index}",
        "core_prefix_id": f"core-{concept_id}",
        "concept_id": concept_id,
        "mode": mode,
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": coefficient_str,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": strength,
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": "fixed_context",
        "source_prompt_tokens": prompt_tokens,
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "attacked_prompt_tokens": prompt_tokens if is_control else [97, 120],
        "attacked_prompt_token_digest": token_sequence_digest(
            prompt_tokens if is_control else [97, 120]
        ),
        "roundtrip_prompt_tokens": prompt_tokens if is_control else [97, 120],
        "roundtrip_prompt_token_digest": token_sequence_digest(
            prompt_tokens if is_control else [97, 120]
        ),
        "roundtrip_prompt_text": "ab" if is_control else "ax",
        "control_roundtrip_prompt_tokens": prompt_tokens if is_control else [97, 120],
        "control_roundtrip_prompt_token_digest": token_sequence_digest(
            prompt_tokens if is_control else [97, 120]
        ),
        "control_roundtrip_prompt_text": "ab" if is_control else "ax",
        "editable_token_start": len(prompt_tokens) if is_control else 1,
        "editable_token_count": 0 if is_control else 1,
        "replacement_tokens": [] if is_control else [120],
        "loss_concept_rich": loss_rich,
        "loss_concept_avoiding": loss_avoid,
        "sample_index": sample_index,
        "reference_segment_token_count": 3,
        "generation_seed": 100 + sample_index,
        "generated_text": " generated",
        "n_generated_tokens": 3,
        "stop_reason": stop_reason,
        "control_generation_seed": 200 + sample_index,
        "control_generated_text": " control",
        "control_n_generated_tokens": 2,
        "control_stop_reason": control_stop_reason,
        "is_control": is_control,
        "target_feature_id": 42,
        "n_roundtrip_features": 1,
        "roundtrip_pre_act": percentile / 10.0,
        "roundtrip_post_act": percentile / 20.0,
        "roundtrip_percentile": percentile,
        "control_n_roundtrip_features": 1,
        "control_roundtrip_pre_act": control_percentile / 10.0,
        "control_roundtrip_post_act": control_percentile / 20.0,
        "control_roundtrip_percentile": control_percentile,
    }


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA)


def _general_val_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=GENERAL_VAL_BPB_SCHEMA)


def _general_val_row(
    *,
    strength: float,
    bpb: float,
    general_val_corpus: str | None = "nd=2;nb=32;h=0000000000000000",
) -> dict[str, Any]:
    return {
        "target_model": "decoder_only",
        "method_name": "diffmean_decoder_only",
        "mode": "positive",
        "concept_id": 1,
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": strength,
        "general_val_bpb": bpb,
        "n_documents": 2,
        "n_val_bytes": 32,
        "intervention_signature": f"sig-{strength}",
        "general_val_corpus": general_val_corpus,
    }


def _with_feature_diffmean_roundtrip_target(
    macro: pa.Table,
    *,
    score: float = 78.0,
) -> pa.Table:
    rows = macro.to_pylist()
    target = {
        **rows[0],
        "method_name": "feature_diffmean_encdec",
        "target_model": "encdec",
        "coefficient_str": "None",
        "primary_roundtrip_y": score,
        "control_primary_roundtrip_y": score,
    }
    return pa.Table.from_pylist([*rows, target], schema=ROUNDTRIP_MACRO_AGG_SCHEMA)


def test_roundtrip_per_candidate_normalizes_positive_and_negative_modes() -> None:
    scores = _table(
        [
            _score_row(mode="positive", percentile=80.0),
            _score_row(
                mode="positive",
                strength=1.0,
                strength_index=1,
                percentile=60.0,
                control_percentile=50.0,
                is_control=False,
            ),
            _score_row(mode="negative", percentile=20.0),
            _score_row(
                mode="negative",
                strength=1.0,
                strength_index=1,
                percentile=40.0,
                control_percentile=70.0,
                is_control=False,
            ),
        ]
    )

    per_candidate = compute_roundtrip_per_candidate_aggregation(scores)

    assert per_candidate.schema.equals(ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
    by_key = {(row["mode"], row["strength_index"]): row for row in per_candidate.to_pylist()}
    assert by_key[("positive", 0)]["primary_roundtrip_y"] == pytest.approx(80.0)
    assert by_key[("positive", 1)]["primary_roundtrip_y"] == pytest.approx(60.0)
    assert by_key[("positive", 1)]["control_primary_roundtrip_y"] == pytest.approx(50.0)
    assert by_key[("negative", 0)]["primary_roundtrip_y"] == pytest.approx(80.0)
    assert by_key[("negative", 1)]["primary_roundtrip_y"] == pytest.approx(60.0)
    assert by_key[("negative", 1)]["control_primary_roundtrip_y"] == pytest.approx(30.0)


def test_roundtrip_per_candidate_averages_multiple_samples_and_control_rates() -> None:
    scores = _table(
        [
            _score_row(
                percentile=80.0,
                control_percentile=60.0,
                sample_index=0,
                stop_reason="max_tokens",
                control_stop_reason="eos",
            ),
            _score_row(
                percentile=40.0,
                control_percentile=20.0,
                sample_index=1,
                stop_reason="eos",
                control_stop_reason="max_tokens",
            ),
        ]
    )

    [row] = compute_roundtrip_per_candidate_aggregation(scores).to_pylist()

    assert row["n_samples"] == 2
    assert row["mean_roundtrip_percentile"] == pytest.approx(60.0)
    assert row["primary_roundtrip_y"] == pytest.approx(60.0)
    assert row["mean_control_roundtrip_percentile"] == pytest.approx(40.0)
    assert row["control_primary_roundtrip_y"] == pytest.approx(40.0)
    assert row["n_generated_tokens"] == 6
    assert row["n_control_generated_tokens"] == 4
    assert row["max_tokens_rate"] == pytest.approx(0.5)
    assert row["control_max_tokens_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("roundtrip_percentile", 101.0, "outside \\[0, 100\\]"),
        ("control_roundtrip_percentile", -0.1, "outside \\[0, 100\\]"),
        ("roundtrip_post_act", float("nan"), "non-finite"),
    ],
)
def test_roundtrip_aggregation_rejects_malformed_numeric_values(
    field: str,
    value: float,
    match: str,
) -> None:
    row = _score_row()
    row[field] = value

    with pytest.raises(AxBenchBpbAdvError, match=match):
        compute_roundtrip_per_candidate_aggregation(_table([row]))


def test_roundtrip_aggregate_raises_loudly_when_all_scores_empty(
    tmp_path: Path,
) -> None:
    # When EVERY roundtrip completion was empty, scoring now skips them all and
    # writes a 0-row table (instead of the old per-candidate raise at score
    # time). Aggregation must then fail LOUDLY rather than silently emit an
    # empty/misleading plot — closing the deferred-failure path.
    scores_path = tmp_path / "roundtrip_scores.parquet"
    pq.write_table(_table([]), scores_path)
    with pytest.raises(AxBenchBpbAdvError, match="no axbench-bpb-adv roundtrip scores"):
        aggregate_roundtrip_adv_scores(scores_path, tmp_path / "agg")


def test_roundtrip_macro_aggregation_preserves_schema_and_counts() -> None:
    per_candidate = compute_roundtrip_per_candidate_aggregation(
        _table(
            [
                _score_row(concept_id=1, percentile=80.0),
                _score_row(concept_id=2, percentile=70.0),
            ]
        )
    )

    macro = compute_roundtrip_macro_aggregation(per_candidate)

    assert macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
    row = macro.to_pylist()[0]
    assert row["n_concepts"] == 2
    assert row["n_samples"] == 2
    assert row["mean_loss_concept_rich"] == pytest.approx(1.0)
    assert row["mean_loss_concept_avoiding"] == pytest.approx(3.0)
    assert row["primary_roundtrip_y"] == pytest.approx(75.0)
    assert row["control_primary_roundtrip_y"] == pytest.approx(75.0)
    assert row["n_generated_tokens"] == 6
    assert row["n_control_generated_tokens"] == 4
    assert row["max_tokens_rate"] == pytest.approx(1.0)
    assert row["control_max_tokens_rate"] == pytest.approx(0.0)


# --- Issue #635 (D2): roundtrip macro rollup mirrors the BPB macro's single-corpus
# guarantee / mix-raise / value-without-corpus raise --------------------------


def _roundtrip_per_candidate_row(
    *,
    concept_id: int,
    general_val_bpb: float | None = None,
    general_val_corpus: str | None = None,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "fixed_context",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 3.0,
        "mean_roundtrip_pre_act": 8.0,
        "mean_roundtrip_post_act": 4.0,
        "mean_roundtrip_percentile": 80.0,
        "primary_roundtrip_y": 80.0,
        "mean_control_roundtrip_pre_act": 8.0,
        "mean_control_roundtrip_post_act": 4.0,
        "mean_control_roundtrip_percentile": 80.0,
        "control_primary_roundtrip_y": 80.0,
        "general_val_bpb": general_val_bpb,
        "general_val_corpus": general_val_corpus,
        "n_samples": 1,
        "n_generated_tokens": 3,
        "n_control_generated_tokens": 2,
        "max_tokens_rate": 1.0,
        "control_max_tokens_rate": 0.0,
        "is_control": True,
    }


def test_roundtrip_macro_carries_single_corpus_and_raises_on_mix() -> None:
    a = _roundtrip_per_candidate_row(
        concept_id=1, general_val_bpb=3.0, general_val_corpus="nd=2;nb=10;h=aa"
    )
    b = _roundtrip_per_candidate_row(
        concept_id=2, general_val_bpb=4.0, general_val_corpus="nd=2;nb=10;h=aa"
    )
    macro = compute_roundtrip_macro_aggregation(
        pa.Table.from_pylist([a, b], schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
    )
    assert macro.to_pylist()[0]["general_val_corpus"] == "nd=2;nb=10;h=aa"

    c = _roundtrip_per_candidate_row(
        concept_id=2, general_val_bpb=4.0, general_val_corpus="nd=9;nb=99;h=bb"
    )
    with pytest.raises(AxBenchBpbAdvError, match="mixes general-val corpus signatures"):
        compute_roundtrip_macro_aggregation(
            pa.Table.from_pylist([a, c], schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
        )


def test_roundtrip_macro_rejects_value_without_corpus() -> None:
    row = _roundtrip_per_candidate_row(concept_id=1, general_val_bpb=3.0, general_val_corpus=None)
    with pytest.raises(AxBenchBpbAdvError, match="general_val_bpb without a corpus signature"):
        compute_roundtrip_macro_aggregation(
            pa.Table.from_pylist([row], schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
        )

    # An EMPTY-STRING corpus is just as unusable as a null one: the macro check
    # mirrors the joins' falsy (`not corpus`) predicate, per the schema comment's
    # "null iff general_val_bpb is null" invariant (issue #635 fail-loud).
    empty = _roundtrip_per_candidate_row(concept_id=1, general_val_bpb=3.0, general_val_corpus="")
    with pytest.raises(AxBenchBpbAdvError, match="general_val_bpb without a corpus signature"):
        compute_roundtrip_macro_aggregation(
            pa.Table.from_pylist([empty], schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
        )


def test_roundtrip_join_rejects_general_val_value_without_corpus() -> None:
    # Mirror of the BPB join's value-without-corpus raise for the roundtrip twin.
    per_candidate = pa.Table.from_pylist(
        [_roundtrip_per_candidate_row(concept_id=1)],
        schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    )
    gv_row = _general_val_row(strength=0.0, bpb=3.0, general_val_corpus=None)
    with pytest.raises(AxBenchBpbAdvError, match="without a corpus signature"):
        _join_roundtrip_general_val_bpb(per_candidate, _general_val_table([gv_row]))


def test_roundtrip_representative_coefficient_uses_roundtrip_baseline() -> None:
    rows = compute_roundtrip_macro_aggregation(
        compute_roundtrip_per_candidate_aggregation(
            _table(
                [
                    _score_row(coefficient_str="1.0", percentile=80.0),
                    _score_row(coefficient_str="2.0", percentile=70.0),
                    _score_row(
                        coefficient_str="1.0",
                        strength=1.0,
                        strength_index=1,
                        percentile=50.0,
                        is_control=False,
                    ),
                    _score_row(
                        coefficient_str="2.0",
                        strength=1.0,
                        strength_index=1,
                        percentile=65.0,
                        is_control=False,
                    ),
                ]
            )
        )
    ).to_pylist()
    target = {
        **rows[0],
        "method_name": "feature_diffmean_encdec",
        "target_model": "encdec",
        "coefficient_str": "None",
        "primary_roundtrip_y": 78.0,
    }

    kept = _filter_rows_to_representative_coefficient(
        rows,
        baseline_metric="primary_roundtrip_y",
        baseline_rows=[*rows, target],
    )

    assert {row["coefficient_str"] for row in kept} == {"1.0"}
    assert {row["strength_index"] for row in kept} == {0, 1}


def test_aggregate_roundtrip_writes_parquet_csv_and_report(tmp_path: Path) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"
    out_dir = tmp_path / "roundtrip_aggregates"
    pq.write_table(_table([_score_row(), _score_row(sample_index=1)]), scores_path)

    macro = aggregate_roundtrip_adv_scores(scores_path, out_dir)

    assert macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
    assert (out_dir / "per_candidate.parquet").exists()
    assert (out_dir / "macro.parquet").exists()
    assert (out_dir / "per_candidate.csv").exists()
    assert (out_dir / "macro.csv").exists()
    report = (out_dir / "aggregation_report.yaml").read_text()
    assert "n_score_rows: 2" in report


def test_aggregate_roundtrip_joins_general_val_for_scatter(tmp_path: Path) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"
    general_val_path = tmp_path / "general_val_bpb.parquet"
    out_dir = tmp_path / "roundtrip_aggregates"
    pq.write_table(
        _table(
            [
                _score_row(strength=0.0, strength_index=0, is_control=True),
                _score_row(
                    strength=1.0,
                    strength_index=1,
                    is_control=False,
                    percentile=65.0,
                ),
            ]
        ),
        scores_path,
    )
    pq.write_table(
        _general_val_table(
            [
                _general_val_row(strength=0.0, bpb=2.25),
                _general_val_row(strength=1.0, bpb=3.5),
            ]
        ),
        general_val_path,
    )

    macro = aggregate_roundtrip_adv_scores(
        scores_path,
        out_dir,
        general_val_path=general_val_path,
    )

    assert macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
    by_strength = {row["adversarial_strength"]: row for row in macro.to_pylist()}
    assert by_strength[0.0]["general_val_bpb"] == pytest.approx(2.25)
    assert by_strength[1.0]["general_val_bpb"] == pytest.approx(3.5)
    assert _scatter_baseline_x(
        by_strength[1.0],
        scatter_baseline_x="general_val_bpb",
    ) == pytest.approx(3.5)
    written = plot_roundtrip_adv_summaries(
        _with_feature_diffmean_roundtrip_target(macro),
        tmp_path / "plots",
        scatter_baseline_x="general_val_bpb",
        plot_methods=["diffmean_decoder_only"],
    )
    assert ROUNDTRIP_SCATTER_PLOT_NAME in {path.name for path in written}


def test_roundtrip_aggregate_uses_supplied_scores_after_path_is_replaced(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"
    validated_scores = _table([_score_row(percentile=80.0)])
    pq.write_table(validated_scores, scores_path)
    pq.write_table(_table([_score_row(percentile=15.0)]), scores_path)

    macro = aggregate_roundtrip_adv_scores(
        scores_path,
        tmp_path / "roundtrip_aggregates",
        scores_table=validated_scores,
    )

    assert macro.to_pylist()[0]["primary_roundtrip_y"] == pytest.approx(80.0), (
        "roundtrip aggregation must use its caller-supplied score table instead "
        "of replacement content at scores_path"
    )


def test_roundtrip_aggregate_uses_supplied_general_val_after_path_is_replaced(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"
    general_val_path = tmp_path / "general_val_bpb.parquet"
    pq.write_table(_table([_score_row()]), scores_path)
    validated_general_val = _general_val_table(
        [_general_val_row(strength=0.0, bpb=2.25)]
    )
    pq.write_table(validated_general_val, general_val_path)
    pq.write_table(
        _general_val_table([_general_val_row(strength=0.0, bpb=9.75)]),
        general_val_path,
    )

    macro = aggregate_roundtrip_adv_scores(
        scores_path,
        tmp_path / "roundtrip_aggregates",
        general_val_path=general_val_path,
        general_val_table=validated_general_val,
    )

    assert macro.to_pylist()[0]["general_val_bpb"] == pytest.approx(2.25), (
        "roundtrip aggregation must use its caller-supplied general-val table "
        "instead of replacement content at general_val_path"
    )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            pa.Table.from_pydict({"wrong": [1]}),
            "schema mismatch in .*roundtrip_scores.parquet: expected "
            "ROUNDTRIP_SCORE_SCHEMA",
        ),
        (
            pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA),
            "no axbench-bpb-adv roundtrip scores in .*roundtrip_scores.parquet",
        ),
    ],
)
def test_roundtrip_none_scores_table_preserves_legacy_validation_messages(
    tmp_path: Path,
    table: pa.Table,
    message: str,
) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"
    pq.write_table(table, scores_path)

    with pytest.raises(AxBenchBpbAdvError, match=message):
        aggregate_roundtrip_adv_scores(
            scores_path,
            tmp_path / "roundtrip_aggregates",
            scores_table=None,
        )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            pa.Table.from_pydict({"wrong": [1]}),
            "schema mismatch in .*roundtrip_scores.parquet: expected "
            "ROUNDTRIP_SCORE_SCHEMA",
        ),
        (
            pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA),
            "no axbench-bpb-adv roundtrip scores in .*roundtrip_scores.parquet",
        ),
    ],
)
def test_roundtrip_supplied_scores_table_keeps_schema_and_empty_validation(
    tmp_path: Path,
    table: pa.Table,
    message: str,
) -> None:
    scores_path = tmp_path / "roundtrip_scores.parquet"

    with pytest.raises(AxBenchBpbAdvError, match=message):
        aggregate_roundtrip_adv_scores(
            scores_path,
            tmp_path / "roundtrip_aggregates",
            scores_table=table,
        )


def test_plot_roundtrip_outputs_index_legend_and_metric_pngs(tmp_path: Path) -> None:
    macro = compute_roundtrip_macro_aggregation(
        compute_roundtrip_per_candidate_aggregation(
            _table(
                [
                    _score_row(percentile=80.0),
                    _score_row(
                        strength=1.0,
                        strength_index=1,
                        percentile=60.0,
                        control_percentile=52.0,
                        is_control=False,
                    ),
                ]
            )
        )
    )

    written = plot_roundtrip_adv_summaries(
        _with_feature_diffmean_roundtrip_target(macro),
        tmp_path,
        plot_methods=["diffmean_decoder_only"],
    )

    names = {path.name for path in written}
    assert "plot_index.csv" in names
    assert "plot_series_legend.csv" in names
    assert "primary_roundtrip_y_vs_strength.png" in names
    assert ROUNDTRIP_SCATTER_PLOT_NAME in names
    index_text = (tmp_path / "plot_index.csv").read_text()
    assert "primary_roundtrip_y_vs_strength.png" in index_text
    assert ROUNDTRIP_SCATTER_PLOT_NAME in index_text
    assert "control_primary_roundtrip_y_vs_strength.png" in index_text
    index_rows = list(csv.DictReader(index_text.splitlines()))
    scatter_rows = [row for row in index_rows if row["plot_name"] == ROUNDTRIP_SCATTER_PLOT_NAME]
    assert len(scatter_rows) == 1
    assert int(scatter_rows[0]["n_rows"]) == 2


def test_plot_roundtrip_missing_target_preserves_previous_plots(tmp_path: Path) -> None:
    macro = compute_roundtrip_macro_aggregation(
        compute_roundtrip_per_candidate_aggregation(
            _table(
                [
                    _score_row(percentile=80.0),
                    _score_row(
                        strength=1.0,
                        strength_index=1,
                        percentile=60.0,
                        control_percentile=52.0,
                        is_control=False,
                    ),
                ]
            )
        )
    )
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    sentinel = plots_dir / "old-plot.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="missing fair feature-DiffMean"):
        plot_roundtrip_adv_summaries(
            macro,
            plots_dir,
            plot_methods=["diffmean_decoder_only"],
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_roundtrip_scatter_x_axis_is_mode_dependent_and_min_symmetric() -> None:
    rows = compute_roundtrip_macro_aggregation(
        compute_roundtrip_per_candidate_aggregation(
            _table(
                [
                    _score_row(
                        mode="positive",
                        loss_rich=1.25,
                        loss_avoid=3.75,
                    ),
                    _score_row(
                        mode="negative",
                        concept_id=2,
                        loss_rich=2.5,
                        loss_avoid=4.5,
                    ),
                ]
            )
        )
    ).to_pylist()
    by_mode = {row["mode"]: row for row in rows}

    assert _scatter_baseline_x(
        by_mode["positive"],
        scatter_baseline_x="baseline_loss",
    ) == pytest.approx(3.75)
    assert _scatter_baseline_x(
        by_mode["negative"],
        scatter_baseline_x="baseline_loss",
    ) == pytest.approx(2.5)
    assert _scatter_baseline_x(
        by_mode["positive"],
        scatter_baseline_x="min",
    ) == pytest.approx(1.25)
    assert _scatter_baseline_x(
        by_mode["negative"],
        scatter_baseline_x="min",
    ) == pytest.approx(2.5)


def test_plot_roundtrip_rejects_bpb_macro_schema(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="ROUNDTRIP_MACRO_AGG_SCHEMA"):
        plot_roundtrip_adv_summaries(pa.table({"x": [1]}), tmp_path)
