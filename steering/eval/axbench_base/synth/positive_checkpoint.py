"""Crash-safe storage for Stage 1b positive-synthesis checkpoints.

The Stage 1b state machine lives in :mod:`positive_mode`.  This module owns
only the persisted transaction: strict JSON, the internal pair-row parquet
schema, parquet-first/JSON-second publication, resume reconciliation, and
cleanup.  Keeping the dependency one-way prevents checkpoint IO from growing
its own copy of the candidate scheduler.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.atomic_io import atomic_write_bytes, atomic_write_parquet
from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.eval.axbench_base.positive_selection_contract import (
    POSITIVE_INITIAL_COMPAT_TARGET,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    DOC_CONTINUATION_GEN_SOURCE_PREFIX,
    MAX_RAW_PERCENTILE,
    MIN_RAW_PERCENTILE,
    AxBenchBaseError,
)

CHECKPOINT_SCHEMA_VERSION_V1 = 1
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ALGORITHM_VERSION_V1 = "stage1b_same_target_resume_v1"
CHECKPOINT_ALGORITHM_VERSION_V2 = "stage1b_growing_target_resume_v2"
CHECKPOINT_ALGORITHM_VERSION_V3 = (
    "stage1b_growing_target_post_provider_qc_resume_v3"
)
CHECKPOINT_ALGORITHM_VERSION = (
    "stage1b_growing_target_bulk_seed_rows_identity_resume_v4"
)
_CHECKPOINT_ALGORITHM_VERSIONS = frozenset(
    {
        CHECKPOINT_ALGORITHM_VERSION_V1,
        CHECKPOINT_ALGORITHM_VERSION_V2,
        CHECKPOINT_ALGORITHM_VERSION_V3,
        CHECKPOINT_ALGORITHM_VERSION,
    }
)

_INTERNAL_FIELDS = (
    pa.field("checkpoint_seed_idx", pa.int64(), nullable=True),
    pa.field("checkpoint_candidate_rank", pa.int64(), nullable=True),
    pa.field("checkpoint_acceptance_sequence", pa.int64(), nullable=True),
)
INTERNAL_COLUMN_NAMES = frozenset(field.name for field in _INTERNAL_FIELDS)
_FULL_PRECISION_QC_FIELDS = frozenset(
    {"qc_seed_percentile", "qc_output_percentile", "qc_delta"}
)

_TOP_LEVEL_KEYS_V1 = frozenset(
    {
        "schema_version",
        "fingerprint",
        "accepted_pair_count",
        "next_chain_index",
        "next_acceptance_sequence",
        "concept_states",
    }
)
_TOP_LEVEL_KEYS_V2 = _TOP_LEVEL_KEYS_V1 | {"highest_requested_target"}
_CONCEPT_STATE_KEYS = frozenset(
    {"concept_id", "feature_id", "source", "tracker"}
)
_SOURCE_STATE_KEYS = frozenset(
    {
        "clean_cursor",
        "dirty_scored_once",
        "clean_scored",
        "dirty_scored",
        "headroom_rejected",
        "next_candidate_rank",
        "exhausted",
        "buffered_candidates",
    }
)
_TRACKER_STATE_KEYS = frozenset(
    {
        "seeds_drawn",
        "discarded_rich",
        "discarded_avoid",
        "semantic_rich_attempts",
        "semantic_avoid_attempts",
        "natural_avoid_acceptances",
        "termination_reason",
        "accepted",
        "discarded",
        "outstanding",
    }
)
_SELECTED_KEYS = frozenset(
    {
        "seed_idx",
        "candidate_rank",
        "prefix_text",
        "seed_percentile",
        "stored_clean",
    }
)
_ACCEPTED_KEYS = _SELECTED_KEYS | {"acceptance_sequence", "natural_avoid"}
_DISCARDED_KEYS = _SELECTED_KEYS | {
    "side",
    "rich_semantic_attempts",
    "avoid_semantic_attempts",
}
_OUTSTANDING_KEYS = _SELECTED_KEYS | {"chain_order"}
_TERMINATION_REASONS = frozenset({"pool_exhausted", "zero_hit_guard"})


def _fsync_parent_directory(path: Path) -> None:
    """Make the preceding atomic rename durable across host power loss."""

    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint_schema(training_schema: pa.Schema) -> pa.Schema:
    """Return the lossless Stage 1b checkpoint schema.

    Final training artifacts retain their historical float32 QC columns. The
    checkpoint keeps the three arithmetic-identity fields as float64 so a
    parquet round trip cannot invalidate exact ``delta == output - seed`` QC
    checks or flip an acceptance decision during resume.
    """

    collisions = INTERNAL_COLUMN_NAMES & set(training_schema.names)
    if collisions:
        raise AxBenchBaseError(
            "Stage 1b training schema collides with checkpoint columns: "
            f"{sorted(collisions)}"
        )
    schema = pa.schema(
        [
            (
                pa.field(
                    field.name,
                    pa.float64(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                if field.name in _FULL_PRECISION_QC_FIELDS
                else field
            )
            for field in training_schema
        ],
        metadata=training_schema.metadata,
    )
    for field in _INTERNAL_FIELDS:
        schema = schema.append(field)
    return schema


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_domain(value, path="$")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint is not canonical-JSON encodable: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _validate_json_domain(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint contains non-finite float at {path}: "
                f"{value!r}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_domain(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AxBenchBaseError(
                    "Stage 1b checkpoint JSON object keys must be strings: "
                    f"{path} contains {key!r}"
                )
            _validate_json_domain(item, path=f"{path}.{key}")
        return
    raise AxBenchBaseError(
        "Stage 1b checkpoint contains a non-JSON value at "
        f"{path}: {type(value).__name__}"
    )


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_json_constant,
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Malformed Stage 1b checkpoint JSON at {path}: {exc}. Retain "
            "the file for diagnosis; restore the matching configuration or "
            "explicitly remove both Stage 1b partial files."
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBaseError(
            f"Stage 1b checkpoint JSON at {path} must contain an object"
        )
    typed = cast(dict[str, Any], payload)
    _validate_json_domain(typed, path="$")
    return typed


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise AxBenchBaseError(
            f"{context} has fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _strict_nonnegative_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise AxBenchBaseError(
            f"{context} must be a non-negative strict int, got {value!r}"
        )
    return value


def _strict_positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 1:
        raise AxBenchBaseError(
            f"{context} must be a positive strict int, got {value!r}"
        )
    return value


def _strict_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise AxBenchBaseError(
            f"{context} must be a strict bool, got {value!r}"
        )
    return value


def _strict_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise AxBenchBaseError(f"{context} must be a string, got {value!r}")
    return value


def _strict_percentile(value: object, *, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise AxBenchBaseError(
            f"{context} must be a finite strict float, got {value!r}"
        )
    if not (
        value == 0.0
        or MIN_RAW_PERCENTILE <= value <= MAX_RAW_PERCENTILE
    ):
        raise AxBenchBaseError(
            f"{context} must be 0.0 or lie in "
            f"[{MIN_RAW_PERCENTILE}, {MAX_RAW_PERCENTILE}], got {value!r}"
        )
    return value


def _validate_selected_descriptor(
    descriptor: object,
    *,
    expected_keys: frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise AxBenchBaseError(f"{context} must be an object")
    typed = cast(dict[str, Any], descriptor)
    _require_exact_keys(typed, expected_keys, context=context)
    _strict_nonnegative_int(typed["seed_idx"], context=f"{context}.seed_idx")
    _strict_nonnegative_int(
        typed["candidate_rank"],
        context=f"{context}.candidate_rank",
    )
    _strict_string(typed["prefix_text"], context=f"{context}.prefix_text")
    _strict_percentile(
        typed["seed_percentile"],
        context=f"{context}.seed_percentile",
    )
    _strict_bool(typed["stored_clean"], context=f"{context}.stored_clean")
    return typed


def validate_checkpoint_document(
    document: Mapping[str, Any],
    *,
    json_path: Path,
) -> dict[str, Any]:
    """Validate an exact schema-v1/v2 document and return a plain dict."""

    payload = dict(document)
    context = f"Stage 1b checkpoint at {json_path}"
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        CHECKPOINT_SCHEMA_VERSION_V1,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise AxBenchBaseError(
            f"{context} has schema_version={schema_version!r}; expected "
            f"{CHECKPOINT_SCHEMA_VERSION_V1} or {CHECKPOINT_SCHEMA_VERSION}. "
            "Retain the partial files "
            "for diagnosis and explicitly restart under matching code."
        )
    expected_top_keys = (
        _TOP_LEVEL_KEYS_V1
        if schema_version == CHECKPOINT_SCHEMA_VERSION_V1
        else _TOP_LEVEL_KEYS_V2
    )
    _require_exact_keys(payload, expected_top_keys, context=context)
    highest_requested_target = (
        None
        if schema_version == CHECKPOINT_SCHEMA_VERSION_V1
        else _strict_positive_int(
            payload["highest_requested_target"],
            context=f"{context}.highest_requested_target",
        )
    )
    fingerprint = payload["fingerprint"]
    if not isinstance(fingerprint, dict):
        raise AxBenchBaseError(f"{context}.fingerprint must be an object")
    _validate_json_domain(fingerprint, path="$.fingerprint")
    accepted_pair_count = _strict_nonnegative_int(
        payload["accepted_pair_count"],
        context=f"{context}.accepted_pair_count",
    )
    _strict_nonnegative_int(
        payload["next_chain_index"],
        context=f"{context}.next_chain_index",
    )
    next_sequence = _strict_nonnegative_int(
        payload["next_acceptance_sequence"],
        context=f"{context}.next_acceptance_sequence",
    )
    if next_sequence != accepted_pair_count:
        raise AxBenchBaseError(
            f"{context} has next_acceptance_sequence={next_sequence} but "
            f"accepted_pair_count={accepted_pair_count}"
        )
    concept_states = payload["concept_states"]
    if not isinstance(concept_states, list):
        raise AxBenchBaseError(f"{context}.concept_states must be a list")

    concept_ids: set[int] = set()
    accepted_descriptors = 0
    acceptance_sequences: set[int] = set()
    for concept_index, raw_state in enumerate(concept_states):
        state_context = f"{context}.concept_states[{concept_index}]"
        if not isinstance(raw_state, dict):
            raise AxBenchBaseError(f"{state_context} must be an object")
        state = cast(dict[str, Any], raw_state)
        _require_exact_keys(state, _CONCEPT_STATE_KEYS, context=state_context)
        concept_id = _strict_nonnegative_int(
            state["concept_id"], context=f"{state_context}.concept_id"
        )
        if concept_id in concept_ids:
            raise AxBenchBaseError(
                f"{context} contains duplicate concept_id={concept_id}"
            )
        concept_ids.add(concept_id)
        _strict_nonnegative_int(
            state["feature_id"], context=f"{state_context}.feature_id"
        )

        source = state["source"]
        if not isinstance(source, dict):
            raise AxBenchBaseError(f"{state_context}.source must be an object")
        source = cast(dict[str, Any], source)
        _require_exact_keys(
            source,
            _SOURCE_STATE_KEYS,
            context=f"{state_context}.source",
        )
        for field in (
            "clean_cursor",
            "clean_scored",
            "dirty_scored",
            "headroom_rejected",
            "next_candidate_rank",
        ):
            _strict_nonnegative_int(
                source[field], context=f"{state_context}.source.{field}"
            )
        _strict_bool(
            source["dirty_scored_once"],
            context=f"{state_context}.source.dirty_scored_once",
        )
        _strict_bool(
            source["exhausted"],
            context=f"{state_context}.source.exhausted",
        )
        buffered = source["buffered_candidates"]
        if not isinstance(buffered, list):
            raise AxBenchBaseError(
                f"{state_context}.source.buffered_candidates must be a list"
            )
        for descriptor_index, descriptor in enumerate(buffered):
            _validate_selected_descriptor(
                descriptor,
                expected_keys=_SELECTED_KEYS,
                context=(
                    f"{state_context}.source.buffered_candidates"
                    f"[{descriptor_index}]"
                ),
            )

        tracker = state["tracker"]
        if not isinstance(tracker, dict):
            raise AxBenchBaseError(f"{state_context}.tracker must be an object")
        tracker = cast(dict[str, Any], tracker)
        _require_exact_keys(
            tracker,
            _TRACKER_STATE_KEYS,
            context=f"{state_context}.tracker",
        )
        for field in (
            "seeds_drawn",
            "discarded_rich",
            "discarded_avoid",
            "semantic_rich_attempts",
            "semantic_avoid_attempts",
            "natural_avoid_acceptances",
        ):
            _strict_nonnegative_int(
                tracker[field], context=f"{state_context}.tracker.{field}"
            )
        termination_reason = tracker["termination_reason"]
        if termination_reason is not None:
            if not isinstance(termination_reason, str) or (
                termination_reason not in _TERMINATION_REASONS
            ):
                raise AxBenchBaseError(
                    f"{state_context}.tracker.termination_reason has invalid "
                    f"value {termination_reason!r}"
                )
        for field, expected_keys in (
            ("accepted", _ACCEPTED_KEYS),
            ("discarded", _DISCARDED_KEYS),
            ("outstanding", _OUTSTANDING_KEYS),
        ):
            descriptors = tracker[field]
            if not isinstance(descriptors, list):
                raise AxBenchBaseError(
                    f"{state_context}.tracker.{field} must be a list"
                )
            for descriptor_index, descriptor in enumerate(descriptors):
                descriptor_context = (
                    f"{state_context}.tracker.{field}[{descriptor_index}]"
                )
                typed_descriptor = _validate_selected_descriptor(
                    descriptor,
                    expected_keys=expected_keys,
                    context=descriptor_context,
                )
                if field == "accepted":
                    sequence = _strict_nonnegative_int(
                        typed_descriptor["acceptance_sequence"],
                        context=f"{descriptor_context}.acceptance_sequence",
                    )
                    if sequence in acceptance_sequences:
                        raise AxBenchBaseError(
                            f"{context} contains duplicate acceptance_sequence="
                            f"{sequence}"
                        )
                    acceptance_sequences.add(sequence)
                    _strict_bool(
                        typed_descriptor["natural_avoid"],
                        context=f"{descriptor_context}.natural_avoid",
                    )
                    accepted_descriptors += 1
                elif field == "discarded":
                    side = typed_descriptor["side"]
                    if not isinstance(side, str) or side not in {"rich", "avoid"}:
                        raise AxBenchBaseError(
                            f"{descriptor_context}.side must be 'rich' or "
                            f"'avoid', got {side!r}"
                        )
                    _strict_nonnegative_int(
                        typed_descriptor["rich_semantic_attempts"],
                        context=(
                            f"{descriptor_context}.rich_semantic_attempts"
                        ),
                    )
                    _strict_nonnegative_int(
                        typed_descriptor["avoid_semantic_attempts"],
                        context=(
                            f"{descriptor_context}.avoid_semantic_attempts"
                        ),
                    )
                else:
                    _strict_nonnegative_int(
                        typed_descriptor["chain_order"],
                        context=f"{descriptor_context}.chain_order",
                    )
        if highest_requested_target is not None:
            accepted_count = len(cast(list[Any], tracker["accepted"]))
            outstanding_count = len(cast(list[Any], tracker["outstanding"]))
            if accepted_count + outstanding_count > highest_requested_target:
                raise AxBenchBaseError(
                    f"{state_context} commits {accepted_count} accepted and "
                    f"{outstanding_count} outstanding pairs, exceeding "
                    "highest_requested_target="
                    f"{highest_requested_target}"
                )

    if accepted_descriptors != accepted_pair_count:
        raise AxBenchBaseError(
            f"{context} records accepted_pair_count={accepted_pair_count} but "
            f"contains {accepted_descriptors} accepted descriptors"
        )
    expected_sequences = set(range(accepted_pair_count))
    if acceptance_sequences != expected_sequences:
        raise AxBenchBaseError(
            f"{context} acceptance sequences are not contiguous from zero: "
            f"missing={sorted(expected_sequences - acceptance_sequences)}, "
            f"unexpected={sorted(acceptance_sequences - expected_sequences)}"
        )
    return payload


def _first_mismatch(
    stored: Any,
    expected: Any,
    *,
    path: str,
) -> tuple[str, Any, Any] | None:
    if type(stored) is not type(expected):
        return path, stored, expected
    if isinstance(expected, dict):
        stored_keys = set(stored)
        expected_keys = set(expected)
        if stored_keys != expected_keys:
            missing = sorted(expected_keys - stored_keys, key=str)
            extra = sorted(stored_keys - expected_keys, key=str)
            return f"{path}.<keys>", {"missing": missing, "extra": extra}, {
                "missing": [],
                "extra": [],
            }
        for key in sorted(expected):
            mismatch = _first_mismatch(
                stored[key], expected[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(stored) != len(expected):
            return f"{path}.<length>", len(stored), len(expected)
        for index, (stored_item, expected_item) in enumerate(
            zip(stored, expected, strict=True)
        ):
            mismatch = _first_mismatch(
                stored_item,
                expected_item,
                path=f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if stored != expected:
        return path, stored, expected
    return None


def validate_fingerprint(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    json_path: Path,
) -> None:
    """Compare fingerprints recursively with exact keys and exact types."""

    stored_dict = dict(stored)
    expected_dict = dict(expected)
    _validate_json_domain(stored_dict, path="$.fingerprint")
    _validate_json_domain(expected_dict, path="$.expected_fingerprint")
    mismatch = _first_mismatch(
        stored_dict,
        expected_dict,
        path="fingerprint",
    )
    if mismatch is None:
        return
    field, stored_value, expected_value = mismatch
    raise AxBenchBaseError(
        f"Stage 1b checkpoint fingerprint mismatch at {field}: "
        f"stored={stored_value!r}, expected={expected_value!r}. Retain the "
        f"checkpoint at {json_path} for diagnosis; restore the matching "
        "configuration or explicitly remove both stage1b_partial.json and "
        "stage1b_partial.parquet."
    )


def _accepted_descriptor_by_sequence(
    document: Mapping[str, Any],
) -> dict[int, tuple[int, int, dict[str, Any]]]:
    result: dict[int, tuple[int, int, dict[str, Any]]] = {}
    for state in cast(list[dict[str, Any]], document["concept_states"]):
        concept_id = int(state["concept_id"])
        feature_id = int(state["feature_id"])
        tracker = cast(dict[str, Any], state["tracker"])
        for descriptor in cast(list[dict[str, Any]], tracker["accepted"]):
            result[int(descriptor["acceptance_sequence"])] = (
                concept_id,
                feature_id,
                descriptor,
            )
    return result


def _validate_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any],
    *,
    parquet_path: Path,
    training_schema: pa.Schema,
) -> None:
    expected_keys = set(checkpoint_schema(training_schema).names)
    expected_pairs = int(document["accepted_pair_count"])
    if len(rows) != 2 * expected_pairs:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint parquet at {parquet_path} has {len(rows)} "
            f"committed-prefix rows; expected {2 * expected_pairs}"
        )
    descriptors = _accepted_descriptor_by_sequence(document)
    seen_seed_keys: set[tuple[int, int]] = set()
    seen_rank_keys: set[tuple[int, int]] = set()
    for sequence in range(expected_pairs):
        rich = dict(rows[2 * sequence])
        avoid = dict(rows[2 * sequence + 1])
        for row_index, row in (
            (2 * sequence, rich),
            (2 * sequence + 1, avoid),
        ):
            if set(row) != expected_keys:
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint row {row_index} at {parquet_path} "
                    "has fields mismatch"
                )
        if rich["content_class"] != CONCEPT_RICH_LABEL or (
            avoid["content_class"] != CONCEPT_AVOIDING_LABEL
        ):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint pair {sequence} at {parquet_path} is "
                "not adjacent concept_rich/concept_avoiding rows"
            )
        repeated_fields = (
            "concept_id",
            "prefix_text",
            "checkpoint_seed_idx",
            "checkpoint_candidate_rank",
            "checkpoint_acceptance_sequence",
        )
        for field in repeated_fields:
            if rich[field] != avoid[field]:
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint pair {sequence} at {parquet_path} "
                    f"disagrees on {field}"
                )
        row_sequence = _strict_nonnegative_int(
            rich["checkpoint_acceptance_sequence"],
            context=(
                f"Stage 1b checkpoint pair {sequence} acceptance sequence"
            ),
        )
        if row_sequence != sequence:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint pair {sequence} at {parquet_path} "
                f"declares acceptance_sequence={row_sequence}"
            )
        seed_idx = _strict_nonnegative_int(
            rich["checkpoint_seed_idx"],
            context=f"Stage 1b checkpoint pair {sequence} seed index",
        )
        candidate_rank = _strict_nonnegative_int(
            rich["checkpoint_candidate_rank"],
            context=f"Stage 1b checkpoint pair {sequence} candidate rank",
        )
        concept_id, feature_id, descriptor = descriptors[sequence]
        expected_values = {
            "concept_id": concept_id,
            "checkpoint_seed_idx": int(descriptor["seed_idx"]),
            "checkpoint_candidate_rank": int(descriptor["candidate_rank"]),
            "prefix_text": descriptor["prefix_text"],
            "qc_feature_id": feature_id,
        }
        for field, expected_value in expected_values.items():
            if rich[field] != expected_value or (
                field in {"concept_id", "prefix_text", "qc_feature_id"}
                and avoid[field] != expected_value
            ):
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint pair {sequence} at {parquet_path} "
                    f"disagrees with JSON descriptor on {field}: "
                    f"row={rich[field]!r}, JSON={expected_value!r}"
                )
        natural_avoid = str(avoid["gen_source"]).startswith(
            DOC_CONTINUATION_GEN_SOURCE_PREFIX
        )
        if natural_avoid is not descriptor["natural_avoid"]:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint pair {sequence} at {parquet_path} "
                "disagrees with JSON descriptor on natural_avoid"
            )
        seed_key = (concept_id, seed_idx)
        rank_key = (concept_id, candidate_rank)
        if seed_key in seen_seed_keys:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint at {parquet_path} contains duplicate "
                f"(concept_id, seed_idx)={seed_key}"
            )
        if rank_key in seen_rank_keys:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint at {parquet_path} contains duplicate "
                f"(concept_id, candidate_rank)={rank_key}"
            )
        seen_seed_keys.add(seed_key)
        seen_rank_keys.add(rank_key)


def write_checkpoint(
    *,
    json_path: Path,
    parquet_path: Path,
    document: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    training_schema: pa.Schema,
) -> None:
    """Validate and atomically publish parquet first, then the JSON marker."""

    payload = validate_checkpoint_document(document, json_path=json_path)
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise AxBenchBaseError(
            "Stage 1b checkpoint writes require schema_version="
            f"{CHECKPOINT_SCHEMA_VERSION}; schema-v1 is read-only migration "
            "input"
        )
    _validate_checkpoint_rows(
        rows,
        payload,
        parquet_path=parquet_path,
        training_schema=training_schema,
    )
    schema = checkpoint_schema(training_schema)
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Could not build Stage 1b checkpoint parquet for {parquet_path}: "
            f"{exc}"
        ) from exc
    encoded = _canonical_json_bytes(payload)
    atomic_write_parquet(table, parquet_path)
    _fsync_parent_directory(parquet_path)
    atomic_write_bytes(encoded, json_path)
    _fsync_parent_directory(json_path)


def _stored_v1_target(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
) -> int:
    fingerprint = cast(Mapping[str, Any], payload["fingerprint"])
    selection = fingerprint.get("positive_selection_contract")
    if not isinstance(selection, dict):
        raise AxBenchBaseError(
            f"Stage 1b schema-v1 checkpoint at {json_path} has no complete "
            "positive_selection_contract fingerprint; retain it for "
            "diagnosis and explicitly restart"
        )
    target = selection.get("n_positives_per_concept")
    if type(target) is not int or target < 1:
        raise AxBenchBaseError(
            f"Stage 1b schema-v1 checkpoint at {json_path} has invalid stored "
            f"target {target!r}"
        )
    return target


def _normalize_v1_checkpoint(
    payload: dict[str, Any],
    *,
    json_path: Path,
    expected_fingerprint: Mapping[str, Any],
    expected_v1_fingerprint: Mapping[str, Any],
    current_target: int,
) -> dict[str, Any]:
    """Validate the narrow target-72 migration and normalize it in memory."""

    stored_target = _stored_v1_target(payload, json_path=json_path)
    if stored_target != POSITIVE_INITIAL_COMPAT_TARGET:
        raise AxBenchBaseError(
            f"Stage 1b schema-v1 checkpoint at {json_path} stores target "
            f"{stored_target}; only target {POSITIVE_INITIAL_COMPAT_TARGET} "
            "can migrate because Phase 1 prefix scoring used target-shaped "
            "BF16 batches. Finish this run under the Phase 1 revision or "
            "explicitly restart Stage 1b."
        )
    if current_target < stored_target:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint target decrease is unsupported: stored "
            f"highest target={stored_target}, current target={current_target}. "
            f"Restore target {POSITIVE_INITIAL_COMPAT_TARGET} or higher, or "
            "use a fresh explicit run_tag."
        )
    validate_fingerprint(
        cast(dict[str, Any], payload["fingerprint"]),
        expected_v1_fingerprint,
        json_path=json_path,
    )
    normalized = dict(payload)
    normalized["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    normalized["fingerprint"] = dict(expected_fingerprint)
    normalized["highest_requested_target"] = stored_target
    return normalized


def _normalize_v2_checkpoint(
    payload: dict[str, Any],
    *,
    json_path: Path,
    expected_fingerprint: Mapping[str, Any],
    expected_v2_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate the exact pre-batching schema-v2 fingerprint in memory.

    The persisted state shape is unchanged. A v2 checkpoint predates the
    effective post-provider QC batch identity, so its outstanding work adopts
    the current configured shape exactly once. The next checkpoint commit
    writes v4; subsequent attempts to change that shape fail exact fingerprint
    validation.
    """

    validate_fingerprint(
        cast(dict[str, Any], payload["fingerprint"]),
        expected_v2_fingerprint,
        json_path=json_path,
    )
    normalized = dict(payload)
    normalized["fingerprint"] = dict(expected_fingerprint)
    return normalized


def _checkpoint_algorithm_version(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
) -> str:
    fingerprint = cast(Mapping[str, Any], payload["fingerprint"])
    version = fingerprint.get("checkpoint_algorithm_version")
    if not isinstance(version, str) or version not in _CHECKPOINT_ALGORITHM_VERSIONS:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint at {json_path} has unsupported "
            f"checkpoint_algorithm_version={version!r}"
        )
    schema_version = int(payload["schema_version"])
    if (
        schema_version == CHECKPOINT_SCHEMA_VERSION_V1
        and version != CHECKPOINT_ALGORITHM_VERSION_V1
    ) or (
        schema_version == CHECKPOINT_SCHEMA_VERSION
        and version == CHECKPOINT_ALGORITHM_VERSION_V1
    ):
        raise AxBenchBaseError(
            f"Stage 1b checkpoint at {json_path} has schema_version="
            f"{schema_version} with incompatible checkpoint_algorithm_version="
            f"{version!r}"
        )
    return version


def load_checkpoint(
    *,
    json_path: Path,
    parquet_path: Path,
    expected_fingerprint: Mapping[str, Any],
    training_schema: pa.Schema,
    current_target: int,
    legacy_fingerprint_factory: Callable[[str], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load the last JSON-committed row prefix, or return ``None`` fresh.

    A parquet without JSON is an uncommitted orphan and is deliberately
    ignored.  JSON without parquet, a parquet shorter than the JSON commit, or
    any mismatch inside the committed prefix is irrecoverable data loss.
    """

    if not json_path.exists():
        return None
    payload = validate_checkpoint_document(
        _read_strict_json(json_path),
        json_path=json_path,
    )
    _strict_positive_int(
        current_target,
        context="Stage 1b current n_positives_per_concept",
    )
    checkpoint_algorithm_version = _checkpoint_algorithm_version(
        payload,
        json_path=json_path,
    )
    if payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION_V1:
        expected_legacy_fingerprint = legacy_fingerprint_factory(
            checkpoint_algorithm_version
        )
        payload = _normalize_v1_checkpoint(
            payload,
            json_path=json_path,
            expected_fingerprint=expected_fingerprint,
            expected_v1_fingerprint=expected_legacy_fingerprint,
            current_target=current_target,
        )
        payload = validate_checkpoint_document(payload, json_path=json_path)
    else:
        stored_fingerprint = cast(dict[str, Any], payload["fingerprint"])
        if checkpoint_algorithm_version == CHECKPOINT_ALGORITHM_VERSION:
            validate_fingerprint(
                stored_fingerprint,
                expected_fingerprint,
                json_path=json_path,
            )
        elif checkpoint_algorithm_version == CHECKPOINT_ALGORITHM_VERSION_V2:
            payload = _normalize_v2_checkpoint(
                payload,
                json_path=json_path,
                expected_fingerprint=expected_fingerprint,
                expected_v2_fingerprint=legacy_fingerprint_factory(
                    checkpoint_algorithm_version
                ),
            )
        else:
            validate_fingerprint(
                stored_fingerprint,
                legacy_fingerprint_factory(checkpoint_algorithm_version),
                json_path=json_path,
            )
            normalized = dict(payload)
            normalized["fingerprint"] = dict(expected_fingerprint)
            payload = normalized
        highest_requested_target = int(payload["highest_requested_target"])
        if current_target < highest_requested_target:
            raise AxBenchBaseError(
                "Stage 1b checkpoint target decrease is unsupported: stored "
                f"highest target={highest_requested_target}, current target="
                f"{current_target}. Restore the same or a higher target, or "
                "use a fresh explicit run_tag."
            )
    if not parquet_path.is_file():
        raise AxBenchBaseError(
            f"Stage 1b checkpoint JSON commit marker {json_path} has no "
            f"matching parquet at {parquet_path}. This is committed data "
            "loss; retain the JSON for diagnosis and explicitly restart."
        )
    expected_schema = checkpoint_schema(training_schema)
    try:
        actual_schema = pq.read_schema(parquet_path)
        if not actual_schema.equals(expected_schema, check_metadata=True):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint parquet at {parquet_path} has schema "
                f"{actual_schema}, expected {expected_schema}"
            )
        rows = cast(list[dict[str, Any]], pq.read_table(parquet_path).to_pylist())
    except AxBenchBaseError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise AxBenchBaseError(
            f"Could not read Stage 1b checkpoint parquet at {parquet_path}: "
            f"{exc}"
        ) from exc
    committed_rows = 2 * int(payload["accepted_pair_count"])
    if len(rows) < committed_rows:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint parquet at {parquet_path} has {len(rows)} "
            f"rows but JSON at {json_path} commits {committed_rows}. This is "
            "data loss; retain both files for diagnosis and explicitly "
            "restart Stage 1b."
        )
    if len(rows) > committed_rows and len(rows) % 2:
        raise AxBenchBaseError(
            f"Stage 1b checkpoint parquet at {parquet_path} is ahead of its "
            f"JSON marker by a half-pair: row_count={len(rows)}"
        )
    committed = rows[:committed_rows]
    _validate_checkpoint_rows(
        committed,
        payload,
        parquet_path=parquet_path,
        training_schema=training_schema,
    )
    return payload, committed


def ordinary_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Strip checkpoint-only columns from an in-memory committed row prefix."""

    return [
        {
            key: value
            for key, value in row.items()
            if key not in INTERNAL_COLUMN_NAMES
        }
        for row in checkpoint_rows
    ]


def delete_checkpoint(*, json_path: Path, parquet_path: Path) -> None:
    """Delete the commit marker first, then its now-uncommitted row file."""

    json_path.unlink(missing_ok=True)
    _fsync_parent_directory(json_path)
    parquet_path.unlink(missing_ok=True)
    _fsync_parent_directory(parquet_path)


__all__ = [
    "CHECKPOINT_ALGORITHM_VERSION",
    "CHECKPOINT_ALGORITHM_VERSION_V1",
    "CHECKPOINT_ALGORITHM_VERSION_V2",
    "CHECKPOINT_ALGORITHM_VERSION_V3",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION_V1",
    "INTERNAL_COLUMN_NAMES",
    "checkpoint_schema",
    "delete_checkpoint",
    "load_checkpoint",
    "ordinary_rows",
    "validate_checkpoint_document",
    "validate_fingerprint",
    "write_checkpoint",
]
