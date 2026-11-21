"""Unit tests for authoritative trainable/scorable concept validation."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.trainable_concepts import (
    concept_ids_present_in_rows,
    require_complete_concept_population_from_corpus,
    require_concept_population_coverage,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
)


def _row(concept_id: int, content_class: str) -> dict:
    return {"concept_id": concept_id, "content_class": content_class}


def test_both_present_concept_is_present():
    rows = [_row(0, "concept_rich"), _row(0, "concept_avoiding")]
    assert concept_ids_present_in_rows(rows) == {0}


def test_one_sided_concept_is_still_present():
    # One-sided is "present"; the trainer's empty-side guard handles it.
    assert concept_ids_present_in_rows([_row(3, "concept_rich")]) == {3}
    assert concept_ids_present_in_rows([_row(4, "concept_avoiding")]) == {4}


def test_absent_concept_excluded():
    present = concept_ids_present_in_rows([_row(0, "concept_rich"), _row(0, "concept_avoiding")])
    assert 1 not in present


def test_unrecognized_content_class_ignored():
    assert concept_ids_present_in_rows([_row(9, "garbage")]) == set()


def test_empty_rows_gives_empty_set():
    assert concept_ids_present_in_rows([]) == set()
def _write_corpus(path, present_ids):
    rows = []
    for cid in present_ids:
        rows.append({"concept_id": cid, "content_class": "concept_rich"})
        rows.append({"concept_id": cid, "content_class": "concept_avoiding"})
    schema = pa.schema([("concept_id", pa.int32()), ("content_class", pa.string())])
    if path.name == "synth_training_positive.parquet":
        schema = schema_with_positive_qc_contract(
            schema,
            accept_threshold=0.3,
        )
    schema = schema_with_stage1_continuation_contract(
        schema,
        role=(
            "stage1b_positive"
            if path.name == "synth_training_positive.parquet"
            else "stage1c_negative"
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_population_coverage_accepts_exact_or_superset_presence():
    require_concept_population_coverage(
        [0, 2],
        {0, 1, 2},
        mode=SteeringMode.POSITIVE,
        consumer="test consumer",
    )


def test_population_coverage_rejects_missing_authoritative_id():
    with pytest.raises(
        AxBenchBaseError,
        match=r"test consumer.*concepts \[2\].*negative synth corpus",
    ):
        require_concept_population_coverage(
            [0, 2],
            {0, 1},
            mode=SteeringMode.NEGATIVE,
            consumer="test consumer",
        )


def test_complete_population_requires_both_classes(tmp_path):
    path = tmp_path / "synth_training_negative.parquet"
    schema = schema_with_stage1_continuation_contract(
        pa.schema(
            [("concept_id", pa.int32()), ("content_class", pa.string())]
        ),
        role="stage1c_negative",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"concept_id": 3, "content_class": "concept_rich"}],
            schema=schema,
        ),
        path,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"concept_avoiding.*rerun Stage 1",
    ):
        require_complete_concept_population_from_corpus(
            path,
            [3],
            mode=SteeringMode.NEGATIVE,
            consumer="test consumer",
        )


def test_complete_population_allows_extra_non_authoritative_concept(tmp_path):
    path = tmp_path / "synth_training_negative.parquet"
    _write_corpus(path, [1, 2])

    require_complete_concept_population_from_corpus(
        path,
        [2],
        mode=SteeringMode.NEGATIVE,
        consumer="test consumer",
    )
