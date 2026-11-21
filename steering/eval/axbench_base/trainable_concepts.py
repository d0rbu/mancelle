"""Validate synth-corpus coverage of the Stage 1 concept population.

Issue #1057 replaces issue #508's absence-derived population with an explicit
filtered concept-list authority. Corpus presence is now validation evidence:
every authoritative concept must have both recognized classes
(``concept_rich`` and ``concept_avoiding``), while extra positive rows are
allowed for concepts Stage 1c later removed.

The content_class label constants are the canonical pair in
``steering.eval.axbench_base.types`` (single source of truth, #559); the names
are re-exported here and aliased as ``_POS_LABEL`` / ``_NEG_LABEL`` in
``steering.eval.axbench_base.baselines._capture``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
    SteeringMode,
)

_RECOGNIZED_LABELS: frozenset[str] = frozenset({CONCEPT_RICH_LABEL, CONCEPT_AVOIDING_LABEL})


def concept_ids_present_in_rows(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Return concept_ids with >=1 recognized (rich|avoid) row.

    A concept that is COMPLETELY ABSENT (0 rich AND 0 avoid) is excluded — the
    QC-drop signature. Rows whose ``content_class`` is neither label are ignored.
    """
    present: set[int] = set()
    for row in rows:
        if row.get("content_class") in _RECOGNIZED_LABELS:
            present.add(int(row["concept_id"]))
    return present


def require_concept_population_coverage(
    expected_concept_ids: Iterable[int],
    present: set[int],
    *,
    mode: SteeringMode,
    consumer: str,
) -> None:
    """Require every authoritative concept to have recognized corpus rows.

    Stage 1's filtered concept list, not corpus presence, defines the
    downstream population. Corpus presence is therefore an integrity check:
    missing rows indicate a broken Stage 1 handoff and must not silently turn
    into a second population filter. Extra rows are allowed because Stage 1c
    can further filter concepts after the positive corpus has been committed.
    """

    expected = set(expected_concept_ids)
    missing = sorted(expected - present)
    if missing:
        raise AxBenchBaseError(
            f"{consumer}: authoritative filtered concepts {missing} are absent "
            f"from the {mode.value} synth corpus; rerun Stage 1 with a fresh "
            "run_tag"
        )


def require_complete_concept_population_from_corpus(
    corpus_path: Path,
    expected_concept_ids: Iterable[int],
    *,
    mode: SteeringMode,
    consumer: str,
    positive_accept_threshold: object | None = None,
    filter_unaccepted: bool = False,
) -> None:
    """Require both canonical classes for every authoritative concept.

    ``filter_unaccepted`` mirrors downstream BPB's configurable source-row
    predicate. Base's own Stage 1 publishers remove failed-QC logical items
    before writing, so production Base callers leave it false.
    """

    try:
        columns = ["concept_id", "content_class"]
        if filter_unaccepted:
            columns.append("qc_accepted")
        table = pq.read_table(
            corpus_path,
            columns=columns,
        )
        validate_stage1_continuation_contract(
            corpus_path,
            consumer=consumer,
            expected_role=(
                "stage1b_positive"
                if mode == SteeringMode.POSITIVE
                else "stage1c_negative"
            ),
            schema=table.schema,
        )
        if positive_accept_threshold is not None:
            validate_positive_qc_contract(
                corpus_path,
                consumer=consumer,
                schema=table.schema,
                expected_accept_threshold=positive_accept_threshold,
            )
            validate_positive_selection_contract(
                corpus_path,
                consumer=consumer,
                schema=table.schema,
            )
    except AxBenchBaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve the corpus path in error
        raise AxBenchBaseError(
            f"{consumer}: failed to read synth corpus {corpus_path}: {exc}"
        ) from exc

    classes_by_concept: dict[int, set[str]] = {}
    for row in table.to_pylist():
        if filter_unaccepted and row["qc_accepted"] is not True:
            continue
        content_class = row["content_class"]
        if content_class not in _RECOGNIZED_LABELS:
            continue
        classes_by_concept.setdefault(int(row["concept_id"]), set()).add(str(content_class))
    required = set(_RECOGNIZED_LABELS)
    incomplete = {
        concept_id: sorted(required - classes_by_concept.get(concept_id, set()))
        for concept_id in expected_concept_ids
        if required - classes_by_concept.get(concept_id, set())
    }
    if incomplete:
        raise AxBenchBaseError(
            f"{consumer}: authoritative filtered concepts lack required "
            f"{mode.value} synth classes: {incomplete}; rerun Stage 1 with a "
            "fresh run_tag"
        )
