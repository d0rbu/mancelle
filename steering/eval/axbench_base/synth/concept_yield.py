"""Per-concept Stage 1 yield filtering and durable reports.

Stage 1 logical items are usable only when both sides of their training pair
survive QC:

- Stage 1b: one selected positive seed, with one accepted ``concept_rich`` row
  immediately followed by its strict-zero ``concept_avoiding`` row.
- Stage 1c: one verified negative-mode prefix, with one or more copied
  ``concept_rich`` rows and exactly one strict-zero ``concept_avoiding`` row.

Rows from incomplete or failed-QC items never enter the synth corpora. A
concept is excluded when its logical-item drop fraction is strictly greater
than ``cfg.max_concept_drop_fraction``. Reports preserve the original sparse
``concept_id`` values; ``concepts.json`` remains the immutable Stage 0c
population/provenance artifact.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from beartype.roar import BeartypeCallHintParamViolation

from nanochat.atomic_io import atomic_json_dump
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
    Concept,
)

_REPORT_SCHEMA_VERSION = 1
_REPORT_STAGES = ("stage1b", "stage1c")
ConceptYieldStage = Literal["stage1b", "stage1c"]
_NegativePrefixKey = tuple[int, int, int, str]


@dataclass(frozen=True)
class ConceptYieldRecord:
    """One concept's complete Stage 1 logical-item accounting."""

    concept_id: int
    feature_index: int
    attempted: int
    accepted: int
    dropped: int
    drop_fraction: float
    excluded: bool
    exclusion_reason: str | None
    reason_counts: dict[str, int]
    emitted_rows: int


def _validate_row_concept(row: dict[str, Any], concept: Concept, *, stage: str) -> None:
    if row.get("concept_id") != concept.concept_id:
        raise AxBenchBaseError(
            f"{stage} yield accounting for concept_id={concept.concept_id} "
            f"received row for concept_id={row.get('concept_id')!r}"
        )
    if row.get("qc_feature_id") != concept.feature_index:
        raise AxBenchBaseError(
            f"{stage} yield accounting for concept_id={concept.concept_id} "
            f"received qc_feature_id={row.get('qc_feature_id')!r}, expected "
            f"{concept.feature_index}"
        )


def _validate_rich_row(
    row: dict[str, Any],
    concept: Concept,
    *,
    stage: str,
    allow_failed_qc: bool,
) -> bool:
    _validate_row_concept(row, concept, stage=stage)
    accepted = row.get("qc_accepted")
    if not isinstance(accepted, bool):
        raise AxBenchBaseError(
            f"{stage} concept_id={concept.concept_id} concept_rich row has "
            f"non-boolean qc_accepted={accepted!r}"
        )
    if not accepted and not allow_failed_qc:
        raise AxBenchBaseError(
            f"{stage} concept_id={concept.concept_id} unexpectedly contains a "
            "failed-QC concept_rich row"
        )
    if row.get("qc_zero_activation_verified") is not False:
        raise AxBenchBaseError(
            f"{stage} concept_id={concept.concept_id} concept_rich row must have "
            "qc_zero_activation_verified=False"
        )
    return accepted


def _validate_avoiding_row(
    row: dict[str, Any],
    concept: Concept,
    *,
    stage: str,
) -> None:
    _validate_row_concept(row, concept, stage=stage)
    if row.get("qc_accepted") is not True:
        raise AxBenchBaseError(
            f"{stage} concept_id={concept.concept_id} concept_avoiding row must "
            "have qc_accepted=True"
        )
    if row.get("qc_zero_activation_verified") is not True:
        raise AxBenchBaseError(
            f"{stage} concept_id={concept.concept_id} concept_avoiding row must "
            "have qc_zero_activation_verified=True"
        )


def _make_record(
    concept: Concept,
    *,
    attempted: int,
    accepted: int,
    reason_counts: Counter[str],
    emitted_rows: int,
    max_drop_fraction: float,
) -> ConceptYieldRecord:
    if attempted < 1:
        raise AxBenchBaseError(
            f"concept_id={concept.concept_id} yield accounting requires at "
            f"least one attempted logical item, got {attempted}"
        )
    if not 0 <= accepted <= attempted:
        raise AxBenchBaseError(
            f"concept_id={concept.concept_id} yield accounting has "
            f"accepted={accepted} outside [0, attempted={attempted}]"
        )
    dropped = attempted - accepted
    if sum(reason_counts.values()) != dropped:
        raise AxBenchBaseError(
            f"concept_id={concept.concept_id} yield reasons sum to "
            f"{sum(reason_counts.values())}, expected dropped={dropped}"
        )
    drop_fraction = dropped / attempted
    excluded = drop_fraction > max_drop_fraction
    return ConceptYieldRecord(
        concept_id=concept.concept_id,
        feature_index=concept.feature_index,
        attempted=attempted,
        accepted=accepted,
        dropped=dropped,
        drop_fraction=drop_fraction,
        excluded=excluded,
        exclusion_reason=("drop_fraction_exceeded" if excluded else None),
        reason_counts=dict(sorted(reason_counts.items())),
        emitted_rows=0 if excluded else emitted_rows,
    )


def filter_positive_concept_rows(
    concept: Concept,
    rows: list[dict[str, Any]],
    *,
    attempted: int,
    max_drop_fraction: float,
) -> tuple[list[dict[str, Any]], ConceptYieldRecord]:
    """Keep only complete, accepted Stage 1b pairs and account for all seeds."""

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    observed_rich = 0
    row_idx = 0
    while row_idx < len(rows):
        rich = rows[row_idx]
        if rich.get("content_class") != CONCEPT_RICH_LABEL:
            raise AxBenchBaseError(
                f"Stage 1b concept_id={concept.concept_id} row {row_idx} is "
                "not a concept_rich row at the start of a logical pair"
            )
        observed_rich += 1
        if observed_rich > attempted:
            raise AxBenchBaseError(
                f"Stage 1b concept_id={concept.concept_id} emitted more "
                f"concept_rich rows ({observed_rich}) than attempted seeds "
                f"({attempted})"
            )
        rich_accepted = _validate_rich_row(
            rich,
            concept,
            stage="Stage 1b",
            allow_failed_qc=True,
        )
        avoid: dict[str, Any] | None = None
        if (
            row_idx + 1 < len(rows)
            and rows[row_idx + 1].get("content_class") == CONCEPT_AVOIDING_LABEL
        ):
            avoid = rows[row_idx + 1]
            _validate_avoiding_row(avoid, concept, stage="Stage 1b")
            if avoid.get("prefix_text") != rich.get("prefix_text") or avoid.get(
                "output_concept"
            ) != rich.get("output_concept"):
                raise AxBenchBaseError(
                    f"Stage 1b concept_id={concept.concept_id} rich/avoiding "
                    f"rows at indices {row_idx}/{row_idx + 1} do not share "
                    "their prefix and concept description"
                )
            row_idx += 1

        if not rich_accepted:
            reasons["rich_qc_failed"] += 1
        elif avoid is None:
            reasons["missing_avoiding"] += 1
        else:
            kept.extend((rich, avoid))
        row_idx += 1

    missing_rich = attempted - observed_rich
    if missing_rich:
        reasons["missing_rich"] += missing_rich
    accepted = len(kept) // 2
    record = _make_record(
        concept,
        attempted=attempted,
        accepted=accepted,
        reason_counts=reasons,
        emitted_rows=len(kept),
        max_drop_fraction=max_drop_fraction,
    )
    return ([] if record.excluded else kept), record


def _negative_prefix_key_from_source(row: dict[str, Any]) -> _NegativePrefixKey:
    try:
        return (
            int(row["prefix_source_shard"]),
            int(row["prefix_source_doc"]),
            int(row["prefix_activating_seg_idx"]),
            str(row["prefix_text"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AxBenchBaseError("Stage 1c prefix row has invalid source provenance") from exc


def _negative_prefix_key_from_output(row: dict[str, Any]) -> _NegativePrefixKey:
    try:
        return (
            int(row["prefix_source_shard"]),
            int(row["prefix_source_doc"]),
            int(row["prefix_source_seg_idx"]),
            str(row["prefix_text"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AxBenchBaseError("Stage 1c output row has invalid source provenance") from exc


def filter_negative_concept_rows(
    concept: Concept,
    rows: list[dict[str, Any]],
    *,
    prefix_rows: list[dict[str, Any]],
    max_drop_fraction: float,
) -> tuple[list[dict[str, Any]], ConceptYieldRecord]:
    """Keep only complete Stage 1c prefix groups and account for every prefix."""

    expected_keys = [_negative_prefix_key_from_source(row) for row in prefix_rows]
    if len(expected_keys) != len(set(expected_keys)):
        raise AxBenchBaseError(
            f"Stage 1c concept_id={concept.concept_id} has duplicate prefix source provenance"
        )
    if not expected_keys:
        raise AxBenchBaseError(f"Stage 1c concept_id={concept.concept_id} has no verified prefixes")
    groups: dict[_NegativePrefixKey, list[dict[str, Any]]] = {key: [] for key in expected_keys}
    for row in rows:
        _validate_row_concept(row, concept, stage="Stage 1c")
        key = _negative_prefix_key_from_output(row)
        if key not in groups:
            raise AxBenchBaseError(
                f"Stage 1c concept_id={concept.concept_id} emitted row for "
                f"unknown prefix provenance {key}"
            )
        groups[key].append(row)

    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    accepted = 0
    for key in expected_keys:
        group = groups[key]
        if not group:
            reasons["missing_pair_rows"] += 1
            continue
        labels = [row.get("content_class") for row in group]
        n_rich = labels.count(CONCEPT_RICH_LABEL)
        n_avoid = labels.count(CONCEPT_AVOIDING_LABEL)
        if any(label not in (CONCEPT_RICH_LABEL, CONCEPT_AVOIDING_LABEL) for label in labels):
            raise AxBenchBaseError(
                f"Stage 1c concept_id={concept.concept_id} prefix {key} has "
                f"unknown content_class sequence {labels!r}"
            )
        if n_rich < 1:
            raise AxBenchBaseError(
                f"Stage 1c concept_id={concept.concept_id} prefix {key} has "
                "an avoiding row but no concept_rich row"
            )
        if n_avoid > 1:
            raise AxBenchBaseError(
                f"Stage 1c concept_id={concept.concept_id} prefix {key} has "
                f"{n_avoid} concept_avoiding rows; expected at most one"
            )
        rich_rows = group[:n_rich]
        if any(row.get("content_class") != CONCEPT_RICH_LABEL for row in rich_rows) or any(
            row.get("content_class") != CONCEPT_AVOIDING_LABEL for row in group[n_rich:]
        ):
            raise AxBenchBaseError(
                f"Stage 1c concept_id={concept.concept_id} prefix {key} rows "
                "are not ordered rich row(s) then avoiding row"
            )
        for rich in rich_rows:
            _validate_rich_row(
                rich,
                concept,
                stage="Stage 1c",
                allow_failed_qc=False,
            )
        if n_avoid == 0:
            reasons["missing_avoiding"] += 1
            continue
        avoid = group[-1]
        _validate_avoiding_row(avoid, concept, stage="Stage 1c")
        kept.extend(group)
        accepted += 1

    record = _make_record(
        concept,
        attempted=len(expected_keys),
        accepted=accepted,
        reason_counts=reasons,
        emitted_rows=len(kept),
        max_drop_fraction=max_drop_fraction,
    )
    return ([] if record.excluded else kept), record


def _write_concept_yield_report(
    cfg: AxBenchBaseConfig,
    *,
    stage: ConceptYieldStage,
    records: list[ConceptYieldRecord],
    path: Path,
    success_status: Literal["preparing", "ok"],
) -> frozenset[int]:
    """Write one validated report state, then enforce the population guard."""

    if stage not in _REPORT_STAGES:
        raise AxBenchBaseError(f"unknown concept-yield stage {stage!r}")
    if not records:
        raise AxBenchBaseError(f"{stage} concept-yield report requires records")
    concept_ids = [record.concept_id for record in records]
    if len(concept_ids) != len(set(concept_ids)):
        raise AxBenchBaseError(f"{stage} concept-yield report has duplicate concept IDs")
    for record in records:
        _validate_report_record(
            record,
            max_drop_fraction=cfg.max_concept_drop_fraction,
            source=f"{stage} concept-yield producer",
        )
    excluded = [record for record in records if record.excluded]
    surviving_ids = [record.concept_id for record in records if not record.excluded]
    dropped_fraction = len(excluded) / len(records)
    failures: list[str] = []
    if dropped_fraction > cfg.max_dropped_concept_fraction:
        failures.append(
            "dropped_concept_fraction="
            f"{dropped_fraction:.6f} exceeds max_dropped_concept_fraction="
            f"{cfg.max_dropped_concept_fraction:.6f}"
        )
    if not surviving_ids:
        failures.append("no concepts survived")
    payload: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "stage": stage,
        "status": ("failed_population_guard" if failures else success_status),
        "qc_fallback_on_exhaust": cfg.qc_fallback_on_exhaust,
        "max_concept_drop_fraction": cfg.max_concept_drop_fraction,
        "max_dropped_concept_fraction": cfg.max_dropped_concept_fraction,
        "n_concepts_attempted": len(records),
        "n_concepts_surviving": len(surviving_ids),
        "n_concepts_dropped": len(excluded),
        "dropped_concept_fraction": dropped_fraction,
        "surviving_concept_ids": surviving_ids,
        "population_guard_failures": failures,
        "concepts": [asdict(record) for record in records],
    }
    atomic_json_dump(payload, path)
    if failures:
        raise AxBenchBaseError(
            f"{stage} concept population guard failed: {'; '.join(failures)}. "
            f"Diagnostic report written to {path}"
        )
    return frozenset(surviving_ids)


def prepare_concept_yield_report(
    cfg: AxBenchBaseConfig,
    *,
    stage: ConceptYieldStage,
    records: list[ConceptYieldRecord],
    path: Path,
) -> frozenset[int]:
    """Invalidate the commit marker before publishing dependent artifacts.

    A successful population guard is recorded with ``status="preparing"``.
    Downstream stages reject that state. The producer must publish its
    filtered concepts and corpus before replacing it with the final ``"ok"``
    report via :func:`write_concept_yield_report`.
    """

    return _write_concept_yield_report(
        cfg,
        stage=stage,
        records=records,
        path=path,
        success_status="preparing",
    )


def write_concept_yield_report(
    cfg: AxBenchBaseConfig,
    *,
    stage: ConceptYieldStage,
    records: list[ConceptYieldRecord],
    path: Path,
) -> frozenset[int]:
    """Atomically publish the final successful report commit marker."""

    return _write_concept_yield_report(
        cfg,
        stage=stage,
        records=records,
        path=path,
        success_status="ok",
    )


def write_filtered_concepts(
    concepts: list[Concept],
    surviving_concept_ids: frozenset[int],
    path: Path,
) -> list[Concept]:
    """Write one ordered, non-renumbered subset as a Stage 1 commit artifact."""

    known_ids = {concept.concept_id for concept in concepts}
    unknown_ids = sorted(surviving_concept_ids - known_ids)
    if unknown_ids:
        raise AxBenchBaseError(
            f"cannot write filtered concepts at {path}: survivor IDs are absent "
            f"from the input population: {unknown_ids}"
        )
    filtered = [concept for concept in concepts if concept.concept_id in surviving_concept_ids]
    if not filtered:
        raise AxBenchBaseError(f"cannot write empty filtered concept population at {path}")
    atomic_json_dump([asdict(concept) for concept in filtered], path)
    return filtered


def load_filtered_concepts(
    path: Path,
    *,
    parent_concepts: list[Concept],
) -> list[Concept]:
    """Load a Stage 1 subset and prove it is an exact parent subsequence."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(f"cannot load filtered concept population at {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise AxBenchBaseError(
            f"filtered concept population at {path} must be a non-empty JSON array"
        )
    parent_by_id = {concept.concept_id: concept for concept in parent_concepts}
    loaded: list[Concept] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AxBenchBaseError(f"filtered concepts[{index}] at {path} must be a JSON object")
        item_dict = cast(dict[str, Any], item)
        try:
            concept = Concept(**item_dict)
        except (TypeError, BeartypeCallHintParamViolation) as exc:
            raise AxBenchBaseError(
                f"filtered concepts[{index}] at {path} has invalid fields: {exc}"
            ) from exc
        parent = parent_by_id.get(concept.concept_id)
        if parent is None:
            raise AxBenchBaseError(
                f"filtered concepts[{index}] at {path} has concept_id="
                f"{concept.concept_id}, absent from its parent population"
            )
        parent_payload = asdict(parent)
        if item_dict != parent_payload or any(
            type(item_dict[field_name]) is not type(expected_value)
            for field_name, expected_value in parent_payload.items()
        ):
            raise AxBenchBaseError(
                f"filtered concepts[{index}] at {path} mutates parent "
                f"concept_id={concept.concept_id}"
            )
        loaded.append(concept)
    loaded_id_set = {item.concept_id for item in loaded}
    expected_order = [
        concept.concept_id for concept in parent_concepts if concept.concept_id in loaded_id_set
    ]
    loaded_ids = [concept.concept_id for concept in loaded]
    if loaded_ids != expected_order or len(loaded_ids) != len(set(loaded_ids)):
        raise AxBenchBaseError(
            f"filtered concept population at {path} is not an ordered, unique "
            "subsequence of its parent population"
        )
    return loaded


def _validate_report_record(
    record: ConceptYieldRecord,
    *,
    max_drop_fraction: float,
    source: str,
) -> None:
    integer_fields = {
        "concept_id": record.concept_id,
        "feature_index": record.feature_index,
        "attempted": record.attempted,
        "accepted": record.accepted,
        "dropped": record.dropped,
        "emitted_rows": record.emitted_rows,
    }
    for field_name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise AxBenchBaseError(f"{source} has non-integer {field_name}={value!r}")
    if record.concept_id < 0 or record.feature_index < 0:
        raise AxBenchBaseError(f"{source} has negative concept/feature identity")
    if record.attempted < 1:
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has "
            f"attempted={record.attempted}; expected at least one"
        )
    if (
        record.accepted < 0
        or record.dropped < 0
        or record.accepted + record.dropped != record.attempted
    ):
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has inconsistent "
            "attempted/accepted/dropped counts"
        )
    expected_fraction = record.dropped / record.attempted
    if (
        isinstance(record.drop_fraction, bool)
        or not isinstance(record.drop_fraction, (int, float))
        or not math.isfinite(float(record.drop_fraction))
        or float(record.drop_fraction) != expected_fraction
    ):
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has inconsistent "
            f"drop_fraction={record.drop_fraction!r}, expected "
            f"{expected_fraction!r}"
        )
    expected_excluded = expected_fraction > max_drop_fraction
    if record.excluded is not expected_excluded:
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has excluded="
            f"{record.excluded!r}, expected {expected_excluded!r}"
        )
    expected_reason = "drop_fraction_exceeded" if expected_excluded else None
    if record.exclusion_reason != expected_reason:
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has exclusion_reason="
            f"{record.exclusion_reason!r}, expected {expected_reason!r}"
        )
    if record.emitted_rows < 0 or (record.excluded and record.emitted_rows != 0):
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} has invalid "
            f"emitted_rows={record.emitted_rows}"
        )
    if not isinstance(record.reason_counts, dict) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for reason, count in record.reason_counts.items()
    ):
        raise AxBenchBaseError(f"{source} concept_id={record.concept_id} has invalid reason_counts")
    if sum(record.reason_counts.values()) != record.dropped:
        raise AxBenchBaseError(
            f"{source} concept_id={record.concept_id} reason_counts do not sum to dropped"
        )


def _record_from_json(
    raw_record: object,
    *,
    max_drop_fraction: float,
    source: str,
) -> ConceptYieldRecord:
    if not isinstance(raw_record, dict):
        raise AxBenchBaseError(f"{source} has a non-object concept record")
    record_dict = cast(dict[str, Any], raw_record)
    expected_fields = {
        "concept_id",
        "feature_index",
        "attempted",
        "accepted",
        "dropped",
        "drop_fraction",
        "excluded",
        "exclusion_reason",
        "reason_counts",
        "emitted_rows",
    }
    if set(record_dict) != expected_fields:
        raise AxBenchBaseError(
            f"{source} concept record fields are "
            f"{sorted(str(key) for key in record_dict)}, expected "
            f"{sorted(expected_fields)}"
        )
    record = ConceptYieldRecord(
        concept_id=record_dict["concept_id"],
        feature_index=record_dict["feature_index"],
        attempted=record_dict["attempted"],
        accepted=record_dict["accepted"],
        dropped=record_dict["dropped"],
        drop_fraction=record_dict["drop_fraction"],
        excluded=record_dict["excluded"],
        exclusion_reason=record_dict["exclusion_reason"],
        reason_counts=record_dict["reason_counts"],
        emitted_rows=record_dict["emitted_rows"],
    )
    _validate_report_record(
        record,
        max_drop_fraction=max_drop_fraction,
        source=source,
    )
    return record


def load_concept_yield_surviving_ids(
    cfg: AxBenchBaseConfig,
    *,
    stage: ConceptYieldStage,
    input_concepts: list[Concept],
    path: Path,
) -> frozenset[int]:
    """Load and strictly validate one Stage 1 yield diagnostic report."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            f"cannot load {stage} concept-yield report at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBaseError(f"{stage} concept-yield report at {path} must be a JSON object")
    expected_scalars: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "stage": stage,
        "status": "ok",
        "qc_fallback_on_exhaust": cfg.qc_fallback_on_exhaust,
        "max_concept_drop_fraction": cfg.max_concept_drop_fraction,
        "max_dropped_concept_fraction": cfg.max_dropped_concept_fraction,
    }
    for field_name, expected in expected_scalars.items():
        actual = payload.get(field_name)
        if type(actual) is not type(expected) or actual != expected:
            raise AxBenchBaseError(
                f"{stage} concept-yield report at {path} has "
                f"{field_name}={actual!r}, expected "
                f"{expected!r}; use a fresh run_tag or rerun {stage}"
            )
    raw_records = payload.get("concepts")
    if not isinstance(raw_records, list):
        raise AxBenchBaseError(f"{stage} concept-yield report at {path} has invalid concepts list")
    records = [
        _record_from_json(
            raw_record,
            max_drop_fraction=cfg.max_concept_drop_fraction,
            source=f"{stage} concept-yield report at {path}",
        )
        for raw_record in raw_records
    ]
    expected_ids = [concept.concept_id for concept in input_concepts]
    expected_features = [concept.feature_index for concept in input_concepts]
    report_ids = [record.concept_id for record in records]
    report_features = [record.feature_index for record in records]
    if report_ids != expected_ids or report_features != expected_features:
        raise AxBenchBaseError(
            f"{stage} concept-yield report at {path} does not match its "
            f"ordered input concept identity; use a fresh run_tag or rerun {stage}"
        )
    raw_survivors = payload.get("surviving_concept_ids")
    if not isinstance(raw_survivors, list) or any(
        isinstance(concept_id, bool) or not isinstance(concept_id, int)
        for concept_id in raw_survivors
    ):
        raise AxBenchBaseError(f"{stage} concept-yield report at {path} has invalid survivors")
    derived_survivors = [record.concept_id for record in records if not record.excluded]
    if raw_survivors != derived_survivors:
        raise AxBenchBaseError(
            f"{stage} concept-yield report at {path} has inconsistent surviving_concept_ids"
        )
    if not raw_survivors:
        raise AxBenchBaseError(f"{stage} concept-yield report at {path} has no survivors")
    expected_aggregates: dict[str, Any] = {
        "n_concepts_attempted": len(records),
        "n_concepts_surviving": len(derived_survivors),
        "n_concepts_dropped": len(records) - len(derived_survivors),
        "dropped_concept_fraction": ((len(records) - len(derived_survivors)) / len(records)),
        "population_guard_failures": [],
    }
    for field_name, expected in expected_aggregates.items():
        actual = payload.get(field_name)
        if type(actual) is not type(expected) or actual != expected:
            raise AxBenchBaseError(
                f"{stage} concept-yield report at {path} has inconsistent "
                f"{field_name}={actual!r}, expected {expected!r}"
            )
    survivor_ids = [int(concept_id) for concept_id in raw_survivors]
    return frozenset(survivor_ids)


def concept_yield_report_is_committed(path: Path) -> bool:
    """Return whether ``path`` is a parseable, successful report commit marker."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == _REPORT_SCHEMA_VERSION
        and payload.get("status") == "ok"
        and payload.get("stage") in _REPORT_STAGES
        and isinstance(payload.get("dropped_concept_fraction"), (int, float))
        and not isinstance(payload.get("dropped_concept_fraction"), bool)
        and math.isfinite(float(payload["dropped_concept_fraction"]))
    )
