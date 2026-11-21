"""Unit and artifact-contract tests for issue #1057 concept-yield guards."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from steering.eval.axbench_base.synth.concept_yield import (
    ConceptYieldRecord,
    _make_record,
    concept_yield_report_is_committed,
    filter_negative_concept_rows,
    filter_positive_concept_rows,
    load_concept_yield_surviving_ids,
    load_filtered_concepts,
    prepare_concept_yield_report,
    write_concept_yield_report,
    write_filtered_concepts,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
    Concept,
)


def _concept(concept_id: int = 13, feature_index: int = 91) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=22,
        sae_source="gemmascope",
        description=f"concept {concept_id}",
        corpus_count=1000 + concept_id,
        description_source="test",
    )


def _positive_rich(
    concept: Concept,
    item: int,
    *,
    accepted: bool = True,
) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "qc_feature_id": concept.feature_index,
        "content_class": CONCEPT_RICH_LABEL,
        "qc_accepted": accepted,
        "qc_zero_activation_verified": False,
        "prefix_text": f"prefix-{item}",
        "output_concept": concept.description,
    }


def _positive_avoid(concept: Concept, item: int) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "qc_feature_id": concept.feature_index,
        "content_class": CONCEPT_AVOIDING_LABEL,
        "qc_accepted": True,
        "qc_zero_activation_verified": True,
        "prefix_text": f"prefix-{item}",
        "output_concept": concept.description,
    }


def _positive_pair(
    concept: Concept,
    item: int,
    *,
    accepted: bool = True,
) -> list[dict[str, Any]]:
    return [
        _positive_rich(concept, item, accepted=accepted),
        _positive_avoid(concept, item),
    ]


def _negative_prefix(
    concept: Concept,
    item: int,
) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "prefix_source_shard": 2,
        "prefix_source_doc": 100 + item,
        "prefix_activating_seg_idx": item,
        "prefix_text": f"prefix-{item}",
    }


def _negative_row(
    concept: Concept,
    item: int,
    content_class: str,
) -> dict[str, Any]:
    is_avoiding = content_class == CONCEPT_AVOIDING_LABEL
    return {
        "concept_id": concept.concept_id,
        "qc_feature_id": concept.feature_index,
        "content_class": content_class,
        "qc_accepted": True,
        "qc_zero_activation_verified": is_avoiding,
        "prefix_text": f"prefix-{item}",
        "prefix_source_shard": 2,
        "prefix_source_doc": 100 + item,
        "prefix_source_seg_idx": item,
    }


def _record(
    concept: Concept,
    *,
    excluded: bool,
) -> ConceptYieldRecord:
    return ConceptYieldRecord(
        concept_id=concept.concept_id,
        feature_index=concept.feature_index,
        attempted=4,
        accepted=1 if excluded else 4,
        dropped=3 if excluded else 0,
        drop_fraction=0.75 if excluded else 0.0,
        excluded=excluded,
        exclusion_reason="drop_fraction_exceeded" if excluded else None,
        reason_counts={"missing_rich": 3} if excluded else {},
        emitted_rows=0 if excluded else 8,
    )


def load_stage1b_surviving_concept_ids(
    cfg: Any,
    concepts: list[Concept],
    path: Path,
) -> frozenset[int]:
    """Test shorthand for the production stage-generic strict loader."""

    return load_concept_yield_surviving_ids(
        cfg,
        stage="stage1b",
        input_concepts=concepts,
        path=path,
    )


def test_positive_above_threshold_is_excluded_with_complete_accounting() -> None:
    concept = _concept()
    rows = [
        *_positive_pair(concept, 0),
        *_positive_pair(concept, 1, accepted=False),
        _positive_rich(concept, 2),
    ]

    kept, record = filter_positive_concept_rows(
        concept,
        rows,
        attempted=4,
        max_drop_fraction=0.5,
    )

    assert kept == []
    assert record == ConceptYieldRecord(
        concept_id=13,
        feature_index=91,
        attempted=4,
        accepted=1,
        dropped=3,
        drop_fraction=0.75,
        excluded=True,
        exclusion_reason="drop_fraction_exceeded",
        reason_counts={
            "missing_avoiding": 1,
            "missing_rich": 1,
            "rich_qc_failed": 1,
        },
        emitted_rows=0,
    )


def test_positive_below_threshold_retains_only_complete_accepted_pairs() -> None:
    concept = _concept()
    rows = [
        *_positive_pair(concept, 0),
        *_positive_pair(concept, 1),
        *_positive_pair(concept, 2),
    ]

    kept, record = filter_positive_concept_rows(
        concept,
        rows,
        attempted=4,
        max_drop_fraction=0.5,
    )

    assert kept == rows
    assert record.accepted == 3
    assert record.dropped == 1
    assert record.drop_fraction == 0.25
    assert record.excluded is False
    assert record.reason_counts == {"missing_rich": 1}
    assert record.emitted_rows == 6


def test_positive_no_drop_path_preserves_exact_row_objects_and_order() -> None:
    concept = _concept()
    rows = _positive_pair(concept, 0)

    kept, record = filter_positive_concept_rows(
        concept,
        rows,
        attempted=1,
        max_drop_fraction=0.5,
    )

    assert kept == rows
    assert kept[0] is rows[0]
    assert kept[1] is rows[1]
    assert record.accepted == 1
    assert record.dropped == 0
    assert record.reason_counts == {}


def test_positive_exact_half_boundary_survives_and_preserves_concept_id() -> None:
    concept = _concept(concept_id=27, feature_index=404)
    rows = [
        *_positive_pair(concept, 0),
        *_positive_pair(concept, 1),
    ]

    kept, record = filter_positive_concept_rows(
        concept,
        rows,
        attempted=4,
        max_drop_fraction=0.5,
    )

    assert kept == rows
    assert record.drop_fraction == 0.5
    assert record.excluded is False
    assert record.concept_id == 27
    assert {row["concept_id"] for row in kept} == {27}


def test_best_fallback_failed_qc_row_counts_as_dropped_and_is_not_emitted() -> None:
    concept = _concept()
    failed_best_pair = _positive_pair(concept, 0, accepted=False)

    kept, record = filter_positive_concept_rows(
        concept,
        failed_best_pair,
        attempted=2,
        max_drop_fraction=0.75,
    )

    assert kept == []
    assert record.accepted == 0
    assert record.dropped == 2
    assert record.reason_counts == {
        "missing_rich": 1,
        "rich_qc_failed": 1,
    }
    assert record.excluded is True


@pytest.mark.parametrize("fallback", ["error", "warn", "best"])
def test_yield_threshold_is_independent_of_fallback_label(
    make_valid_axbench_cfg,
    tmp_path: Path,
    fallback: str,
) -> None:
    """Error short-circuits in synthesis; completed reports use one threshold."""

    cfg = make_valid_axbench_cfg(
        qc_fallback_on_exhaust=fallback,
        max_dropped_concept_fraction=0.75,
    )
    concept = _concept()
    report_path = tmp_path / f"{fallback}.json"

    survivors = write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[_record(concept, excluded=False)],
        path=report_path,
    )

    assert survivors == frozenset({concept.concept_id})
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["qc_fallback_on_exhaust"] == fallback


def test_negative_mode_counts_attempted_prefixes_and_removes_unpaired_group() -> None:
    concept = _concept()
    prefixes = [_negative_prefix(concept, item) for item in range(3)]
    rows = [
        _negative_row(concept, 0, CONCEPT_RICH_LABEL),
        _negative_row(concept, 0, CONCEPT_RICH_LABEL),
        _negative_row(concept, 0, CONCEPT_AVOIDING_LABEL),
        _negative_row(concept, 1, CONCEPT_RICH_LABEL),
        _negative_row(concept, 1, CONCEPT_AVOIDING_LABEL),
        _negative_row(concept, 2, CONCEPT_RICH_LABEL),
    ]

    kept, record = filter_negative_concept_rows(
        concept,
        rows,
        prefix_rows=prefixes,
        max_drop_fraction=0.5,
    )

    assert kept == rows[:5]
    assert record.attempted == 3
    assert record.accepted == 2
    assert record.dropped == 1
    assert record.drop_fraction == pytest.approx(1 / 3)
    assert record.reason_counts == {"missing_avoiding": 1}
    assert record.excluded is False
    assert record.emitted_rows == 5


def test_negative_mode_above_threshold_drops_entire_concept() -> None:
    concept = _concept()
    prefixes = [_negative_prefix(concept, item) for item in range(4)]
    rows = [
        _negative_row(concept, 0, CONCEPT_RICH_LABEL),
        _negative_row(concept, 0, CONCEPT_AVOIDING_LABEL),
    ]

    kept, record = filter_negative_concept_rows(
        concept,
        rows,
        prefix_rows=prefixes,
        max_drop_fraction=0.5,
    )

    assert kept == []
    assert record.accepted == 1
    assert record.dropped == 3
    assert record.reason_counts == {"missing_pair_rows": 3}
    assert record.excluded is True


def test_population_guard_exact_half_survives_without_reindexing(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(max_dropped_concept_fraction=0.5)
    first = _concept(concept_id=4, feature_index=40)
    second = _concept(concept_id=9, feature_index=90)
    report_path = tmp_path / "stage1b.json"

    survivors = write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[
            _record(first, excluded=True),
            _record(second, excluded=False),
        ],
        path=report_path,
    )

    assert survivors == frozenset({9})
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["surviving_concept_ids"] == [9]
    assert payload["concepts"][1]["concept_id"] == 9
    assert concept_yield_report_is_committed(report_path)


def test_prepared_report_is_not_committed_until_final_publish(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg()
    concept = _concept()
    records = [_record(concept, excluded=False)]
    report_path = tmp_path / "stage1b.json"

    survivors = prepare_concept_yield_report(
        cfg,
        stage="stage1b",
        records=records,
        path=report_path,
    )

    assert survivors == frozenset({concept.concept_id})
    prepared = json.loads(report_path.read_text(encoding="utf-8"))
    assert prepared["status"] == "preparing"
    assert not concept_yield_report_is_committed(report_path)
    with pytest.raises(AxBenchBaseError, match="status='preparing'"):
        load_concept_yield_surviving_ids(
            cfg,
            stage="stage1b",
            input_concepts=[concept],
            path=report_path,
        )

    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=records,
        path=report_path,
    )

    committed = json.loads(report_path.read_text(encoding="utf-8"))
    assert committed["status"] == "ok"
    assert concept_yield_report_is_committed(report_path)
    assert (
        load_concept_yield_surviving_ids(
            cfg,
            stage="stage1b",
            input_concepts=[concept],
            path=report_path,
        )
        == survivors
    )


def test_population_guard_above_threshold_raises_after_durable_failed_report(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(max_dropped_concept_fraction=0.5)
    concepts = [_concept(concept_id=i, feature_index=100 + i) for i in range(3)]
    report_path = tmp_path / "stage1b.json"

    with pytest.raises(AxBenchBaseError, match="population guard failed"):
        write_concept_yield_report(
            cfg,
            stage="stage1b",
            records=[
                _record(concepts[0], excluded=True),
                _record(concepts[1], excluded=True),
                _record(concepts[2], excluded=False),
            ],
            path=report_path,
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_population_guard"
    assert payload["dropped_concept_fraction"] == pytest.approx(2 / 3)
    assert payload["population_guard_failures"]
    assert not concept_yield_report_is_committed(report_path)


def test_stage1b_survivor_loader_pins_config_and_ordered_concept_identity(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(
        qc_fallback_on_exhaust="warn",
        max_dropped_concept_fraction=0.75,
    )
    concepts = [
        _concept(concept_id=3, feature_index=30),
        _concept(concept_id=8, feature_index=80),
    ]
    report_path = tmp_path / "stage1b.json"
    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[
            _record(concepts[0], excluded=True),
            _record(concepts[1], excluded=False),
        ],
        path=report_path,
    )

    assert load_stage1b_surviving_concept_ids(
        cfg,
        concepts,
        report_path,
    ) == frozenset({8})

    changed_cfg = replace(cfg, max_concept_drop_fraction=0.25)
    with pytest.raises(AxBenchBaseError, match="max_concept_drop_fraction"):
        load_stage1b_surviving_concept_ids(
            changed_cfg,
            concepts,
            report_path,
        )
    with pytest.raises(AxBenchBaseError, match="ordered input concept identity"):
        load_stage1b_surviving_concept_ids(
            cfg,
            list(reversed(concepts)),
            report_path,
        )


def test_filtered_concepts_are_an_ordered_nonrenumbered_parent_subset(
    tmp_path: Path,
) -> None:
    concepts = [
        _concept(concept_id=0, feature_index=10),
        _concept(concept_id=1, feature_index=11),
        _concept(concept_id=2, feature_index=12),
    ]
    path = tmp_path / "concepts_stage1b.json"

    filtered = write_filtered_concepts(
        concepts,
        frozenset({0, 2}),
        path,
    )

    assert [concept.concept_id for concept in filtered] == [0, 2]
    assert [item["concept_id"] for item in json.loads(path.read_text(encoding="utf-8"))] == [0, 2]
    assert (
        load_filtered_concepts(
            path,
            parent_concepts=concepts,
        )
        == filtered
    )


def test_filtered_concepts_reject_unknown_survivor_id(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="absent from the input population"):
        write_filtered_concepts(
            [_concept(concept_id=4, feature_index=40)],
            frozenset({4, 7}),
            tmp_path / "concepts_stage1b.json",
        )


def test_filtered_concepts_reject_empty_population(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="empty filtered concept"):
        write_filtered_concepts(
            [_concept()],
            frozenset(),
            tmp_path / "concepts_stage1b.json",
        )


@pytest.mark.parametrize("corruption", ["reorder", "mutate", "duplicate", "unknown"])
def test_filtered_concepts_reject_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    concepts = [
        _concept(concept_id=0, feature_index=10),
        _concept(concept_id=1, feature_index=11),
        _concept(concept_id=2, feature_index=12),
    ]
    payload = [
        concepts[0].__dict__,
        concepts[2].__dict__,
    ]
    if corruption == "reorder":
        payload.reverse()
    elif corruption == "mutate":
        payload[1] = {**payload[1], "description": "mutated"}
    elif corruption == "duplicate":
        payload[1] = dict(payload[0])
    else:
        payload[1] = {**payload[1], "concept_id": 99}
    path = tmp_path / "concepts_stage1b.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError):
        load_filtered_concepts(path, parent_concepts=concepts)


def test_stage1b_report_loader_rejects_tampered_exclusion_decision(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(max_dropped_concept_fraction=0.75)
    concepts = [
        _concept(concept_id=0, feature_index=10),
        _concept(concept_id=1, feature_index=11),
    ]
    path = tmp_path / "stage1b.json"
    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[
            _record(concepts[0], excluded=True),
            _record(concepts[1], excluded=False),
        ],
        path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["concepts"][0]["excluded"] = False
    payload["concepts"][0]["exclusion_reason"] = None
    payload["surviving_concept_ids"] = [0, 1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="excluded=False, expected True"):
        load_stage1b_surviving_concept_ids(cfg, concepts, path)


@pytest.mark.parametrize(
    ("field_name", "corrupt_value"),
    [
        ("schema_version", True),
        ("n_concepts_attempted", True),
        ("dropped_concept_fraction", False),
    ],
)
def test_stage1b_report_loader_rejects_bool_numeric_impersonation(
    make_valid_axbench_cfg,
    tmp_path: Path,
    field_name: str,
    corrupt_value: bool,
) -> None:
    """JSON booleans must not satisfy integer/float artifact contracts."""

    cfg = make_valid_axbench_cfg()
    concept = _concept()
    path = tmp_path / "stage1b.json"
    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[_record(concept, excluded=False)],
        path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field_name] = corrupt_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match=field_name):
        load_stage1b_surviving_concept_ids(cfg, [concept], path)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("wrong_concept", "received row for concept_id"),
        ("wrong_feature", "received qc_feature_id"),
        ("rich_qc_not_bool", "non-boolean qc_accepted"),
        ("rich_zero_verified", "qc_zero_activation_verified=False"),
        ("avoid_rejected", "concept_avoiding row must have qc_accepted=True"),
        ("avoid_not_zero", "qc_zero_activation_verified=True"),
        ("avoid_first", "not a concept_rich row"),
        ("too_many_rich", "more concept_rich rows"),
        ("pair_prefix_mismatch", "do not share their prefix"),
        ("pair_description_mismatch", "do not share their prefix"),
    ],
)
def test_positive_pair_contract_rejects_every_structural_corruption(
    corruption: str,
    match: str,
) -> None:
    concept = _concept()
    rows = _positive_pair(concept, 0)
    attempted = 1
    if corruption == "wrong_concept":
        rows[0]["concept_id"] = concept.concept_id + 1
    elif corruption == "wrong_feature":
        rows[0]["qc_feature_id"] = concept.feature_index + 1
    elif corruption == "rich_qc_not_bool":
        rows[0]["qc_accepted"] = 1
    elif corruption == "rich_zero_verified":
        rows[0]["qc_zero_activation_verified"] = True
    elif corruption == "avoid_rejected":
        rows[1]["qc_accepted"] = False
    elif corruption == "avoid_not_zero":
        rows[1]["qc_zero_activation_verified"] = False
    elif corruption == "avoid_first":
        rows = [rows[1]]
    elif corruption == "too_many_rich":
        rows.extend(_positive_pair(concept, 1))
    elif corruption == "pair_prefix_mismatch":
        rows[1]["prefix_text"] = "different"
    else:
        rows[1]["output_concept"] = "different"

    with pytest.raises(AxBenchBaseError, match=match):
        filter_positive_concept_rows(
            concept,
            rows,
            attempted=attempted,
            max_drop_fraction=0.5,
        )


@pytest.mark.parametrize(
    ("attempted", "accepted", "reason_counts", "match"),
    [
        (0, 0, Counter(), "at least one attempted"),
        (1, 2, Counter(), "outside"),
        (2, 1, Counter(), "reasons sum"),
    ],
)
def test_record_construction_rejects_nonconserving_counts(
    attempted: int,
    accepted: int,
    reason_counts: Counter[str],
    match: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=match):
        _make_record(
            _concept(),
            attempted=attempted,
            accepted=accepted,
            reason_counts=reason_counts,
            emitted_rows=0,
            max_drop_fraction=0.5,
        )


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("invalid_source_provenance", "invalid source provenance"),
        ("invalid_output_provenance", "invalid source provenance"),
        ("duplicate_prefix", "duplicate prefix source provenance"),
        ("no_prefixes", "no verified prefixes"),
        ("unknown_prefix", "unknown prefix provenance"),
        ("unknown_label", "unknown content_class"),
        ("avoid_only", "no concept_rich"),
        ("multiple_avoid", "expected at most one"),
        ("wrong_order", "not ordered"),
        ("failed_rich_qc", "failed-QC concept_rich"),
        ("wrong_concept", "received row for concept_id"),
    ],
)
def test_negative_group_contract_rejects_every_structural_corruption(
    corruption: str,
    match: str,
) -> None:
    concept = _concept()
    prefixes = [_negative_prefix(concept, 0)]
    rows = [
        _negative_row(concept, 0, CONCEPT_RICH_LABEL),
        _negative_row(concept, 0, CONCEPT_AVOIDING_LABEL),
    ]
    if corruption == "invalid_source_provenance":
        prefixes[0]["prefix_source_doc"] = None
    elif corruption == "invalid_output_provenance":
        rows[0]["prefix_source_doc"] = None
    elif corruption == "duplicate_prefix":
        prefixes.append(dict(prefixes[0]))
    elif corruption == "no_prefixes":
        prefixes = []
        rows = []
    elif corruption == "unknown_prefix":
        rows[0]["prefix_source_doc"] = 999
    elif corruption == "unknown_label":
        rows[0]["content_class"] = "unknown"
    elif corruption == "avoid_only":
        rows = [rows[1]]
    elif corruption == "multiple_avoid":
        rows.append(dict(rows[1]))
    elif corruption == "wrong_order":
        rows.reverse()
    elif corruption == "failed_rich_qc":
        rows[0]["qc_accepted"] = False
    else:
        rows[0]["concept_id"] = concept.concept_id + 1

    with pytest.raises(AxBenchBaseError, match=match):
        filter_negative_concept_rows(
            concept,
            rows,
            prefix_rows=prefixes,
            max_drop_fraction=0.5,
        )


@pytest.mark.parametrize(
    ("base_excluded", "changes", "match"),
    [
        (False, {"concept_id": True}, "non-integer concept_id"),
        (False, {"feature_index": -1}, "negative concept/feature"),
        (
            False,
            {
                "attempted": 0,
                "accepted": 0,
                "dropped": 0,
                "drop_fraction": 0.0,
                "emitted_rows": 0,
            },
            "expected at least one",
        ),
        (False, {"accepted": -1}, "inconsistent attempted/accepted/dropped"),
        (False, {"drop_fraction": float("nan")}, "inconsistent drop_fraction"),
        (False, {"excluded": True}, "excluded=True, expected False"),
        (False, {"exclusion_reason": "bad"}, "exclusion_reason"),
        (False, {"emitted_rows": -1}, "invalid emitted_rows"),
        (True, {"emitted_rows": 1}, "invalid emitted_rows"),
        (False, {"reason_counts": {"": 1}}, "invalid reason_counts"),
        (False, {"reason_counts": {"unexpected": 1}}, "do not sum to dropped"),
    ],
)
def test_report_writer_rejects_invalid_record_contract(
    make_valid_axbench_cfg,
    tmp_path: Path,
    base_excluded: bool,
    changes: dict[str, Any],
    match: str,
) -> None:
    record = replace(_record(_concept(), excluded=base_excluded), **changes)

    with pytest.raises(AxBenchBaseError, match=match):
        write_concept_yield_report(
            make_valid_axbench_cfg(max_dropped_concept_fraction=0.9),
            stage="stage1b",
            records=[record],
            path=tmp_path / "report.json",
        )


def test_report_writer_rejects_invalid_stage_empty_records_and_duplicate_ids(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg()
    record = _record(_concept(), excluded=False)
    with pytest.raises(AxBenchBaseError, match="unknown concept-yield stage"):
        write_concept_yield_report(
            cfg,
            stage=cast(Any, "stage9"),
            records=[record],
            path=tmp_path / "unknown.json",
        )
    with pytest.raises(AxBenchBaseError, match="requires records"):
        write_concept_yield_report(
            cfg,
            stage="stage1b",
            records=[],
            path=tmp_path / "empty.json",
        )
    with pytest.raises(AxBenchBaseError, match="duplicate concept IDs"):
        write_concept_yield_report(
            cfg,
            stage="stage1b",
            records=[record, record],
            path=tmp_path / "duplicate.json",
        )


def test_population_guard_always_rejects_zero_survivors(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(max_dropped_concept_fraction=0.99)
    path = tmp_path / "zero-survivors.json"

    with pytest.raises(AxBenchBaseError, match="no concepts survived"):
        write_concept_yield_report(
            cfg,
            stage="stage1b",
            records=[_record(_concept(), excluded=True)],
            path=path,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_population_guard"
    assert "no concepts survived" in payload["population_guard_failures"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        [17],
        [{"concept_id": 3}],
    ],
)
def test_filtered_concept_loader_rejects_malformed_artifacts(
    tmp_path: Path,
    payload: object,
) -> None:
    path = tmp_path / "concepts_stage1b.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError):
        load_filtered_concepts(path, parent_concepts=[_concept()])


def test_filtered_concept_loader_wraps_missing_and_invalid_json(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(AxBenchBaseError, match="cannot load filtered"):
        load_filtered_concepts(missing, parent_concepts=[_concept()])

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="cannot load filtered"):
        load_filtered_concepts(invalid, parent_concepts=[_concept()])


def _write_valid_stage1b_report(
    cfg: Any,
    path: Path,
    concept: Concept,
) -> dict[str, Any]:
    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[_record(concept, excluded=False)],
        path=path,
    )
    return cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8")),
    )


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("top_level", "must be a JSON object"),
        ("concepts_not_list", "invalid concepts list"),
        ("record_not_object", "non-object concept record"),
        ("record_fields", "concept record fields"),
        ("survivors_not_list", "invalid survivors"),
        ("survivor_bool", "invalid survivors"),
        ("survivor_mismatch", "inconsistent surviving_concept_ids"),
        ("feature_identity", "ordered input concept identity"),
    ],
)
def test_report_loader_rejects_malformed_artifacts(
    make_valid_axbench_cfg,
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    cfg = make_valid_axbench_cfg()
    concept = _concept()
    path = tmp_path / f"{corruption}.json"
    payload: Any = _write_valid_stage1b_report(cfg, path, concept)
    if corruption == "top_level":
        payload = []
    elif corruption == "concepts_not_list":
        payload["concepts"] = {}
    elif corruption == "record_not_object":
        payload["concepts"][0] = []
    elif corruption == "record_fields":
        del payload["concepts"][0]["emitted_rows"]
    elif corruption == "survivors_not_list":
        payload["surviving_concept_ids"] = {}
    elif corruption == "survivor_bool":
        payload["surviving_concept_ids"] = [True]
    elif corruption == "survivor_mismatch":
        payload["surviving_concept_ids"] = [999]
    else:
        payload["concepts"][0]["feature_index"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match=match):
        load_stage1b_surviving_concept_ids(cfg, [concept], path)


def test_report_loader_rejects_missing_invalid_json_and_derived_empty_population(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    cfg = make_valid_axbench_cfg(max_dropped_concept_fraction=0.9)
    concept = _concept()
    missing = tmp_path / "missing.json"
    with pytest.raises(AxBenchBaseError, match="cannot load"):
        load_stage1b_surviving_concept_ids(cfg, [concept], missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="cannot load"):
        load_stage1b_surviving_concept_ids(cfg, [concept], invalid)

    empty = tmp_path / "empty.json"
    payload = _write_valid_stage1b_report(cfg, empty, concept)
    payload["concepts"][0] = _record(concept, excluded=True).__dict__
    payload["surviving_concept_ids"] = []
    empty.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="no survivors"):
        load_stage1b_surviving_concept_ids(cfg, [concept], empty)


def test_report_commit_marker_rejects_missing_invalid_and_nonfinite_artifacts(
    make_valid_axbench_cfg,
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    assert not concept_yield_report_is_committed(path)
    path.write_text("{", encoding="utf-8")
    assert not concept_yield_report_is_committed(path)

    payload = _write_valid_stage1b_report(
        make_valid_axbench_cfg(),
        path,
        _concept(),
    )
    payload["dropped_concept_fraction"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not concept_yield_report_is_committed(path)
