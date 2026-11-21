"""Stage 0c partition primitives for multi-GPU coordination (PR-4).

Provides:
- ``is_my_candidate(idx, *, rank, world_size)``: static modulo partition.
- ``my_candidate_block(block_idx, B, *, rank, world_size, n_candidates=None)``:
  the candidates a rank owns within a given block.
- ``write_per_rank_checkpoint`` / ``read_per_rank_checkpoint``: per-rank
  checkpoint file I/O.
- ``validate_per_rank_fingerprint``: enforce fingerprint equality on
  resume (excluding world_size which is intentionally not validated so
  resume across world_size changes works).
- ``merge_per_rank_checkpoints``: rank-0 final merge; sort accepts by
  candidate_index, truncate to max_concepts, assign canonical dense
  concept_ids in candidate-index sorted order.

See the design at the original (deleted) PR-4 plan; the byte-identity
invariant is "single-GPU output equals multi-GPU output for same cfg",
gated by tests/axbench_base/integration/test_stage0c_multi_gpu_byte_identity.py
(added in PR-4 Task 9).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from beartype import beartype

from nanochat.atomic_io import atomic_write_bytes, atomic_write_parquet
from steering.eval.axbench_base._stage0c_internals import (
    _NEG_PREFIXES_SCHEMA,
    _NEG_PREFIXES_TOKENIZER_METADATA_KEY,
    _schema_with_negative_prefixes_metadata,
    _semantic_json_identity,
)
from steering.eval.axbench_base.types import (
    STAGE0C_API_CHECKPOINT_SCHEMA_VERSION,
    STAGE0C_LOCAL_CHECKPOINT_SCHEMA_VERSION,
    AxBenchBaseError,
    Concept,
    QcOutcome,
    RejectedConceptReason,
)
from steering.refined_types import NonNegativeInt, PositiveInt

# Bumped 3->4 alongside ``_STAGE0C_CHECKPOINT_SCHEMA_VERSION`` 4->5 for
# extractor identity: the fingerprint now includes ``neuronpedia_model_id``,
# the effective ``feature_extraction_model_id``, and ``neuronpedia_source``.
# Stale per-rank checkpoints must be rejected before their partial rows are
# reused under a different extractor.
#
# Bumped 4->5 for candidate-pool identity (concept-whitelist PR-2): the live
# per-rank checkpoint fingerprint now embeds ``candidate_pool_identity`` (a
# content hash of the ordered candidate pool) so a regenerated pool under the
# same pinned ``run_tag`` refuses to resume against stale accepted concepts.
#
# Bumped 5->6 with the local Stage 0c checkpoint schema: enabled concept-QC
# configuration is now fingerprinted as one canonical nested object. This
# prevents rank checkpoints produced with different judge/fetch semantics from
# being resumed or merged together.
#
# Bumped 6->7 when provisional verification identity changed from a rank-local
# QC-survivor ordinal to stable candidate_index. Schema-6 rows were generated
# with topology-dependent seeds/shuffle order and cannot be resumed safely.
#
# Bumped 7->8 because Stage 0c generation semantics now depend on shared versus
# repeated verification prefill. Schema-7 rows have no mode identity and cannot
# be resumed or merged safely under the versioned schema-8 algorithm.
_RANK_CHECKPOINT_SCHEMA_VERSION = STAGE0C_LOCAL_CHECKPOINT_SCHEMA_VERSION


def _rank_checkpoint_schema(extraction_tokenizer_id: str) -> pa.Schema:
    """Return the typed per-rank parquet schema and tokenizer provenance."""
    rank_schema = pa.schema(
        list(_NEG_PREFIXES_SCHEMA)
        + [pa.field("candidate_index", pa.int64())]
    )
    inherited = _schema_with_negative_prefixes_metadata(
        extraction_tokenizer_id
    )
    return rank_schema.with_metadata(inherited.metadata)


@beartype
def is_my_candidate(
    idx: NonNegativeInt,
    *,
    rank: NonNegativeInt,
    world_size: PositiveInt,
) -> bool:
    """Return True iff candidate ``idx`` is owned by ``rank`` in a world
    of ``world_size`` ranks.

    Uses static modulo partition: rank R owns ``{i : i % W == R}``. This
    matches the existing AxBench-BPB ``is_mine`` partition idiom for
    consistency.
    """
    if rank >= world_size:
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    return idx % world_size == rank


@beartype
def my_candidate_block(
    block_idx: NonNegativeInt,
    B: PositiveInt,
    *,
    rank: NonNegativeInt,
    world_size: PositiveInt,
    n_candidates: NonNegativeInt | None = None,
) -> Iterator[int]:
    """Yield the candidate indices owned by ``rank`` in block ``block_idx``.

    A "block" of size ``B*world_size`` candidates spans indices
    ``[block_idx*B*W, (block_idx+1)*B*W)``. Within that range, rank R
    takes ``B`` indices: ``block_idx*B*W + R, +R+W, +R+2W, ..., +R+(B-1)*W``.
    Indices >= ``n_candidates`` (when provided) are clipped.
    """
    if rank >= world_size:
        raise ValueError(f"invalid (rank={rank}, world_size={world_size})")
    block_start = block_idx * B * world_size
    for k in range(B):
        idx = block_start + rank + k * world_size
        if n_candidates is not None and idx >= n_candidates:
            return
        yield idx


def write_per_rank_checkpoint(
    json_path: Path,
    parquet_path: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    extraction_tokenizer_id: str,
    schema_version: int = _RANK_CHECKPOINT_SCHEMA_VERSION,
) -> None:
    """Atomically write the per-rank Stage 0c checkpoint pair.

    Order: parquet first (durable row data) then JSON (commit marker).
    The payload schema separates ``fingerprint`` (validated on resume) from
    ``metadata`` (informational; rank, world_size).

    Uses the canonical atomic helpers from :mod:`nanochat.atomic_io`
    (consolidated in PR-4 review) so the tmp-file naming, fsync timing,
    and crash-safe rename behaviour matches every other writer in the
    repo.
    """
    if schema_version not in {
        _RANK_CHECKPOINT_SCHEMA_VERSION,
        STAGE0C_API_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise AxBenchBaseError(
            "Per-rank Stage 0c checkpoint schema_version must be 8 (local) "
            f"or 9 (API), got {schema_version!r}"
        )
    full_payload = {
        "schema_version": schema_version,
        **payload,
    }
    encoded = json.dumps(full_payload, indent=2, ensure_ascii=False).encode("utf-8")
    # Per-rank parquet has the legacy schema PLUS a candidate_index column.
    # Always replace it, including when rows is empty. A parquet-first crash
    # can otherwise leave a JSON-less orphan from an earlier attempt; skipping
    # this write on a fresh all-reject run would pair those stale rows with the
    # new JSON commit marker and leak them into merge.
    rank_schema = _rank_checkpoint_schema(extraction_tokenizer_id)
    table = pa.Table.from_pylist(rows, schema=rank_schema)
    atomic_write_parquet(table, parquet_path)
    atomic_write_bytes(encoded, json_path)


def read_per_rank_checkpoint(
    json_path: Path,
    parquet_path: Path,
    *,
    expected_schema_version: int = _RANK_CHECKPOINT_SCHEMA_VERSION,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the JSON + parquet pair. Returns (payload, rows).

    The schema version is validated against the caller-selected algorithm
    before the field is removed from the returned payload. Merge calls this
    reader directly, so validation cannot rely only on the separate
    partial-resume fingerprint gate.
    """
    raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path} must be a JSON object, got "
            f"{type(raw_payload).__name__}"
        )
    schema_version = raw_payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != expected_schema_version
    ):
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path}: stale schema_version "
            f"{schema_version!r} (expected {expected_schema_version!r}). "
            "Delete the per-rank stage0c_partial_rank_*.json/parquet files "
            "to restart Stage 0c from scratch."
        )
    payload = dict(raw_payload)
    payload.pop("schema_version", None)
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path} must contain a "
            "fingerprint object"
        )
    extraction_tokenizer_id = fingerprint.get("extraction_tokenizer_id")
    if not isinstance(extraction_tokenizer_id, str) or not extraction_tokenizer_id:
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path} fingerprint must contain a "
            "non-empty extraction_tokenizer_id"
        )
    if not parquet_path.is_file():
        raise AxBenchBaseError(
            f"Per-rank checkpoint JSON commit marker {json_path} has no "
            f"matching parquet file at {parquet_path}. Refusing to treat "
            "missing durable row state as an empty rank."
        )

    try:
        actual_schema = pq.read_schema(parquet_path)
    except (OSError, pa.ArrowException) as exc:
        raise AxBenchBaseError(
            f"Could not read per-rank Stage 0c parquet schema at "
            f"{parquet_path}: {exc}"
        ) from exc
    expected_schema = _rank_checkpoint_schema(extraction_tokenizer_id)
    if actual_schema.remove_metadata() != expected_schema.remove_metadata():
        raise AxBenchBaseError(
            f"Per-rank Stage 0c parquet at {parquet_path} has schema "
            f"{actual_schema.remove_metadata()}, expected "
            f"{expected_schema.remove_metadata()}"
        )

    metadata = actual_schema.metadata or {}
    metadata_key = _NEG_PREFIXES_TOKENIZER_METADATA_KEY.encode("utf-8")
    encoded_tokenizer_id = metadata.get(metadata_key)
    if encoded_tokenizer_id is None:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c parquet at {parquet_path} is missing "
            f"required schema metadata {_NEG_PREFIXES_TOKENIZER_METADATA_KEY!r}"
        )
    try:
        actual_tokenizer_id = encoded_tokenizer_id.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c parquet at {parquet_path} has non-UTF-8 "
            f"{_NEG_PREFIXES_TOKENIZER_METADATA_KEY!r} metadata"
        ) from exc
    if actual_tokenizer_id != extraction_tokenizer_id:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c parquet at {parquet_path} was written for "
            f"extraction_tokenizer_id={actual_tokenizer_id!r}, but its JSON "
            f"fingerprint at {json_path} requires {extraction_tokenizer_id!r}"
        )

    try:
        table = pq.read_table(parquet_path, schema=actual_schema)
    except (OSError, pa.ArrowException) as exc:
        raise AxBenchBaseError(
            f"Could not read per-rank Stage 0c parquet at {parquet_path}: "
            f"{exc}"
        ) from exc
    return payload, table.to_pylist()


def validate_per_rank_fingerprint(
    json_path: Path,
    expected_fingerprint: dict[str, Any],
    *,
    expected_schema_version: int = _RANK_CHECKPOINT_SCHEMA_VERSION,
) -> None:
    """Validate the stored fingerprint matches ``expected_fingerprint``.

    Also checks the per-rank checkpoint ``schema_version`` matches the
    caller-selected algorithm; a mismatch indicates a stale checkpoint
    written by a different code revision/config and must be rejected with a
    clear "delete partial files" message before fingerprint comparison.

    Raises AxBenchBaseError on mismatch (with actionable opt-out message).
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != expected_schema_version
    ):
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path}: stale schema_version "
            f"{schema_version!r} (expected {expected_schema_version!r}). "
            "Delete the per-rank stage0c_partial_rank_*.json/parquet files "
            "to restart Stage 0c from scratch."
        )
    actual = payload.get("fingerprint")
    if actual is None:
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path} is missing a fingerprint."
        )
    if not isinstance(actual, dict):
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path} has non-object fingerprint "
            f"{actual!r}."
        )
    if set(actual.keys()) != set(expected_fingerprint.keys()):
        missing = set(expected_fingerprint) - set(actual)
        extra = set(actual) - set(expected_fingerprint)
        raise AxBenchBaseError(
            f"Per-rank checkpoint at {json_path}: fingerprint field set "
            f"differs (missing={sorted(missing)}, extra={sorted(extra, key=str)}). "
            "Delete partial files to restart."
        )
    for k, expected_value in expected_fingerprint.items():
        if _semantic_json_identity(actual[k]) != _semantic_json_identity(
            expected_value
        ):
            raise AxBenchBaseError(
                f"Per-rank checkpoint at {json_path}: fingerprint mismatch "
                f"on field {k!r}: stored={actual[k]!r}, "
                f"expected={expected_value!r}. Delete partial "
                "files to restart."
            )


def _validate_candidate_index(
    value: object,
    *,
    context: str,
) -> NonNegativeInt:
    """Return one strict, non-negative checkpoint candidate index."""
    try:
        return NonNegativeInt.parse(value)
    except TypeError as exc:
        raise AxBenchBaseError(
            f"{context} must be a non-negative int, got {value!r}"
        ) from exc


def _validate_rank_checkpoint_state_for_merge(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    json_path: Path,
    parquet_path: Path,
) -> None:
    """Fail closed on rank ownership and JSON/parquet agreement before merge."""
    if not isinstance(payload.get("fingerprint"), dict):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} must contain a "
            "fingerprint object"
        )
    declared_rank = payload.get("rank")
    if (
        not isinstance(declared_rank, int)
        or isinstance(declared_rank, bool)
        or declared_rank != rank
    ):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} declares "
            f"rank={declared_rank!r}, expected the strict int rank={rank}"
        )
    declared_world_size = payload.get("world_size")
    if (
        not isinstance(declared_world_size, int)
        or isinstance(declared_world_size, bool)
        or declared_world_size != world_size
    ):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} declares "
            f"world_size={declared_world_size!r}, expected the strict current "
            f"world_size={world_size}. Each rank must commit reconciled state "
            "under the current topology before merge."
        )
    next_block_index = payload.get("next_block_index")
    if (
        not isinstance(next_block_index, int)
        or isinstance(next_block_index, bool)
        or next_block_index < 0
    ):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has invalid "
            f"next_block_index={next_block_index!r}; expected a non-negative "
            "int"
        )
    stopped_by_qc_cap = payload.get("stopped_by_qc_survivor_cap")
    if not isinstance(stopped_by_qc_cap, bool):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has non-boolean "
            "stopped_by_qc_survivor_cap"
        )
    if "next_local_qc_concept_id" in payload:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} contains removed "
            "next_local_qc_concept_id state. Schema-7 verification identity "
            "is the stable candidate_index; delete stale partial files."
        )
    n_candidates = payload.get("n_candidates")
    if (
        not isinstance(n_candidates, int)
        or isinstance(n_candidates, bool)
        or n_candidates < 0
    ):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has invalid "
            f"n_candidates={n_candidates!r}; expected a non-negative int"
        )

    def _require_in_candidate_pool(candidate_idx: int, *, context: str) -> None:
        if candidate_idx >= n_candidates:
            raise AxBenchBaseError(
                f"{context}={candidate_idx} is outside candidate pool "
                f"[0, {n_candidates})"
            )

    raw_visited = payload.get("visited_candidate_indices", [])
    if not isinstance(raw_visited, list):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has non-list "
            "visited_candidate_indices"
        )
    visited = [
        _validate_candidate_index(
            value,
            context=(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"visited_candidate_indices[{idx}]"
            ),
        )
        for idx, value in enumerate(raw_visited)
    ]
    for visited_idx, candidate_idx in enumerate(visited):
        _require_in_candidate_pool(
            candidate_idx,
            context=(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"visited_candidate_indices[{visited_idx}]"
            ),
        )
    if len(set(visited)) != len(visited):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} contains duplicate "
            "visited_candidate_indices"
        )
    typed_rank = NonNegativeInt.parse(rank)
    typed_world_size = PositiveInt.parse(world_size)
    non_owned_visited = [
        idx
        for idx in visited
        if not is_my_candidate(
            idx,
            rank=typed_rank,
            world_size=typed_world_size,
        )
    ]
    if non_owned_visited:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} retains candidate "
            f"indices not owned by rank {rank}/{world_size}: "
            f"{non_owned_visited!r}"
        )
    visited_set = set(visited)

    def _record_indices(field: str) -> tuple[list[dict[str, Any]], list[int]]:
        raw_records = payload.get(field, [])
        if not isinstance(raw_records, list):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has non-list "
                f"{field}"
            )
        records: list[dict[str, Any]] = []
        indices: list[int] = []
        for record_idx, record in enumerate(raw_records):
            if not isinstance(record, dict):
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"{field}[{record_idx}] must be an object"
                )
            record = cast(dict[str, Any], record)
            candidate_idx = _validate_candidate_index(
                record.get("candidate_index"),
                context=(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"{field}[{record_idx}].candidate_index"
                ),
            )
            _require_in_candidate_pool(
                candidate_idx,
                context=(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"{field}[{record_idx}].candidate_index"
                ),
            )
            records.append(record)
            indices.append(candidate_idx)
        if len(set(indices)) != len(indices):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has duplicate "
                f"candidate_index values in {field}: {indices!r}"
            )
        outside_visited = sorted(set(indices) - visited_set)
        if outside_visited:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has {field} "
                f"entries not present in visited_candidate_indices: "
                f"{outside_visited!r}"
            )
        return records, indices

    accept_records, accept_indices = _record_indices("rank_accepts")
    reject_records, reject_indices = _record_indices("rank_rejects")
    overlap = sorted(set(accept_indices) & set(reject_indices))
    if overlap:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} records candidate "
            f"indices as both accepted and rejected: {overlap!r}"
        )
    outcome_indices = set(accept_indices) | set(reject_indices)
    if outcome_indices != visited_set:
        missing_outcomes = sorted(visited_set - outcome_indices)
        outcomes_not_visited = sorted(outcome_indices - visited_set)
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} must record exactly "
            "one terminal accept/reject outcome for every visited candidate; "
            f"missing outcomes={missing_outcomes!r}, outcomes not visited="
            f"{outcomes_not_visited!r}"
        )

    def _validate_feature_and_description(
        record: dict[str, Any],
        *,
        field: str,
        record_idx: int,
    ) -> tuple[int, str]:
        feature_index = _validate_candidate_index(
            record.get("feature_index"),
            context=(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"{field}[{record_idx}].feature_index"
            ),
        )
        description = record.get("description")
        if not isinstance(description, str):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"{field}[{record_idx}].description must be a string, got "
                f"{description!r}"
            )
        return feature_index, description

    outcome_by_feature: dict[int, tuple[int, str, dict[str, Any]]] = {}
    for field, records, indices in (
        ("rank_accepts", accept_records, accept_indices),
        ("rank_rejects", reject_records, reject_indices),
    ):
        for record_idx, (record, candidate_idx) in enumerate(
            zip(records, indices, strict=True)
        ):
            feature_index, description = _validate_feature_and_description(
                record,
                field=field,
                record_idx=record_idx,
            )
            if feature_index in outcome_by_feature:
                prior_candidate_idx = outcome_by_feature[feature_index][0]
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path}: "
                    f"feature_index={feature_index} appears more than once, "
                    f"mapped to candidates {prior_candidate_idx} and "
                    f"{candidate_idx}"
                )
            outcome_by_feature[feature_index] = (
                candidate_idx,
                description,
                record,
            )

    for record_idx, record in enumerate(accept_records):
        for field in ("layer", "corpus_count"):
            _validate_candidate_index(
                record.get(field),
                context=(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_accepts[{record_idx}].{field}"
                ),
            )
        for field in ("sae_source", "description_source"):
            value = record.get(field)
            if not isinstance(value, str):
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_accepts[{record_idx}].{field} must be a string, "
                    f"got {value!r}"
                )
        qc_score = record.get("qc_score")
        if qc_score is not None and (
            not isinstance(qc_score, int)
            or isinstance(qc_score, bool)
            or not 0 <= qc_score <= 2
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_accepts[{record_idx}].qc_score must be None or an int "
                f"in [0, 2], got {qc_score!r}"
            )

    fingerprint = payload["fingerprint"]
    qc_enabled = fingerprint.get("enable_concept_examples_qc")
    if not isinstance(qc_enabled, bool):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} fingerprint must "
            "contain boolean enable_concept_examples_qc"
        )
    verification_target = fingerprint.get(
        "n_verified_prefixes_per_concept_target"
    )
    if (
        not isinstance(verification_target, int)
        or isinstance(verification_target, bool)
        or verification_target <= 0
    ):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} fingerprint must "
            "contain positive-int n_verified_prefixes_per_concept_target"
        )
    qc_config = fingerprint.get("concept_examples_qc_config")
    qc_judge_min_score: int | None = None
    qc_survivor_cap: int | None = None
    if qc_enabled:
        if not isinstance(qc_config, dict):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} enables "
                "examples QC without a fingerprinted examples-QC config"
            )
        raw_qc_judge_min_score = qc_config.get(
            "activation_example_judge_min_score"
        )
        if (
            not isinstance(raw_qc_judge_min_score, int)
            or isinstance(raw_qc_judge_min_score, bool)
            or not 0 <= raw_qc_judge_min_score <= 2
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has invalid "
                "fingerprinted activation_example_judge_min_score="
                f"{raw_qc_judge_min_score!r}"
            )
        qc_judge_min_score = raw_qc_judge_min_score
        raw_qc_survivor_cap = qc_config.get(
            "concept_examples_qc_target_surviving"
        )
        if raw_qc_survivor_cap is not None and (
            not isinstance(raw_qc_survivor_cap, int)
            or isinstance(raw_qc_survivor_cap, bool)
            or raw_qc_survivor_cap < 0
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has invalid "
                "fingerprinted concept_examples_qc_target_surviving="
                f"{raw_qc_survivor_cap!r}"
            )
        qc_survivor_cap = raw_qc_survivor_cap
    elif qc_config is not None:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} disables examples "
            "QC but stores a non-null concept_examples_qc_config"
        )

    if stopped_by_qc_cap:
        if not qc_enabled:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} sets "
                "stopped_by_qc_survivor_cap=true while examples QC is "
                "disabled"
            )
        if world_size != 1:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} sets "
                "stopped_by_qc_survivor_cap=true under world_size="
                f"{world_size}; the survivor cap is supported only at "
                "world_size=1"
            )
        if qc_survivor_cap is None:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} sets "
                "stopped_by_qc_survivor_cap=true but its fingerprint has "
                "no non-negative integer "
                "concept_examples_qc_target_surviving"
            )

    for record_idx, record in enumerate(reject_records):
        expected_reject_fields = {
            "candidate_index",
            "feature_index",
            "description",
            "reason",
            "n_verified_found",
            "llm_judge_score",
        }
        if set(record) != expected_reject_fields:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_rejects[{record_idx}] has fields {sorted(record)!r}, "
                f"expected {sorted(expected_reject_fields)!r}"
            )
        try:
            rejection_reason = RejectedConceptReason(record.get("reason"))
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_rejects[{record_idx}].reason="
                f"{record.get('reason')!r} is invalid"
            ) from exc
        n_verified_found = record.get("n_verified_found")
        if n_verified_found is not None and (
            not isinstance(n_verified_found, int)
            or isinstance(n_verified_found, bool)
            or n_verified_found < 0
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_rejects[{record_idx}].n_verified_found must be None "
                f"or a non-negative int, got {n_verified_found!r}"
            )
        llm_judge_score = record.get("llm_judge_score")
        if llm_judge_score is not None and (
            not isinstance(llm_judge_score, int)
            or isinstance(llm_judge_score, bool)
            or not 0 <= llm_judge_score <= 2
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_rejects[{record_idx}].llm_judge_score must be None "
                f"or an int in [0, 2], got {llm_judge_score!r}"
            )
        if rejection_reason == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES:
            if (
                not isinstance(n_verified_found, int)
                or isinstance(n_verified_found, bool)
                or not 0 <= n_verified_found < verification_target
            ):
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_rejects[{record_idx}] with reason="
                    "not_enough_verified_prefixes requires "
                    f"0 <= n_verified_found < {verification_target}, got "
                    f"{n_verified_found!r}"
                )
        elif n_verified_found is not None:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_rejects[{record_idx}] is a QC rejection and must have "
                f"n_verified_found=None, got {n_verified_found!r}"
            )
        if not qc_enabled:
            if rejection_reason != RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_rejects[{record_idx}] has QC-only reason="
                    f"{rejection_reason.value!r} while QC is disabled"
                )
            if llm_judge_score is not None:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_rejects[{record_idx}] stores an LLM score while QC "
                    "is disabled"
                )

    if stopped_by_qc_cap:
        assert qc_survivor_cap is not None
        terminal_qc_survivors = len(accept_records) + sum(
            record["reason"]
            == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value
            for record in reject_records
        )
        if terminal_qc_survivors != qc_survivor_cap:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} sets "
                "stopped_by_qc_survivor_cap=true, but terminal QC survivors "
                f"number {terminal_qc_survivors}; fingerprinted "
                "concept_examples_qc_target_surviving="
                f"{qc_survivor_cap}"
            )

    raw_qc_records = payload.get("rank_qc_per_feature", [])
    if not isinstance(raw_qc_records, list):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has non-list "
            "rank_qc_per_feature"
        )
    if not qc_enabled and raw_qc_records:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} contains QC records "
            "while enable_concept_examples_qc is false"
        )
    if not qc_enabled:
        stored_qc_scores = [
            record.get("qc_score") for record in accept_records
        ] + [record.get("llm_judge_score") for record in reject_records]
        if any(score is not None for score in stored_qc_scores):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} stores LLM QC "
                "scores while enable_concept_examples_qc is false"
            )

    expected_qc_fields = {
        "feature_index",
        "description",
        "outcome",
        "score",
        "n_examples",
        "error_message",
    }
    qc_by_feature: dict[int, dict[str, Any]] = {}
    for qc_idx, qc_record in enumerate(raw_qc_records):
        if not isinstance(qc_record, dict):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}] must be an object"
            )
        qc_record = cast(dict[str, Any], qc_record)
        if set(qc_record) != expected_qc_fields:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}] has fields "
                f"{sorted(qc_record)!r}, expected "
                f"{sorted(expected_qc_fields)!r}"
            )
        feature_index = _validate_candidate_index(
            qc_record.get("feature_index"),
            context=(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].feature_index"
            ),
        )
        if feature_index in qc_by_feature:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has duplicate "
                f"QC feature_index={feature_index}"
            )
        description = qc_record.get("description")
        if not isinstance(description, str):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].description must be a string"
            )
        raw_outcome = qc_record.get("outcome")
        try:
            outcome = QcOutcome(raw_outcome)
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].outcome={raw_outcome!r} is "
                "not a valid QcOutcome"
            ) from exc
        score = qc_record.get("score")
        if score is not None and (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 2
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].score must be None or an int "
                f"in [0, 2], got {score!r}"
            )
        n_examples = qc_record.get("n_examples")
        if (
            not isinstance(n_examples, int)
            or isinstance(n_examples, bool)
            or n_examples < 0
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].n_examples must be a "
                f"non-negative int, got {n_examples!r}"
            )
        error_message = qc_record.get("error_message")
        if error_message is not None and not isinstance(error_message, str):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}].error_message must be None "
                f"or a string, got {error_message!r}"
            )
        if outcome in (QcOutcome.ACCEPTED, QcOutcome.REJECTED):
            assert qc_judge_min_score is not None
            if score is None or n_examples == 0 or error_message is not None:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_qc_per_feature[{qc_idx}] outcome={outcome.value!r} "
                    "requires an integer score, at least one example, and "
                    "no error_message"
                )
            if (
                outcome is QcOutcome.ACCEPTED
                and score < qc_judge_min_score
            ) or (
                outcome is QcOutcome.REJECTED
                and score >= qc_judge_min_score
            ):
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_qc_per_feature[{qc_idx}] score={score} is "
                    f"inconsistent with outcome={outcome.value!r} at "
                    f"judge_min_score={qc_judge_min_score}"
                )
        else:
            if score is not None:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_qc_per_feature[{qc_idx}] outcome={outcome.value!r} "
                    "must not have a judge score"
                )
            if outcome in (QcOutcome.FETCH_ERROR, QcOutcome.EMPTY_EXAMPLES):
                if n_examples != 0:
                    raise AxBenchBaseError(
                        f"Per-rank Stage 0c checkpoint at {json_path} "
                        f"rank_qc_per_feature[{qc_idx}] outcome="
                        f"{outcome.value!r} requires n_examples=0"
                    )
            elif n_examples == 0:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_qc_per_feature[{qc_idx}] judge_error requires at "
                    "least one fetched example"
                )
            has_error = outcome in (
                QcOutcome.FETCH_ERROR,
                QcOutcome.JUDGE_ERROR,
            )
            if has_error != isinstance(error_message, str):
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} "
                    f"rank_qc_per_feature[{qc_idx}] outcome="
                    f"{outcome.value!r} has inconsistent error_message"
                )
        qc_by_feature[feature_index] = qc_record

    if qc_enabled and set(qc_by_feature) != set(outcome_by_feature):
        missing_qc = sorted(set(outcome_by_feature) - set(qc_by_feature))
        qc_without_outcome = sorted(set(qc_by_feature) - set(outcome_by_feature))
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} must have exactly "
            "one QC record for each visited candidate feature; missing QC="
            f"{missing_qc!r}, QC without terminal outcome="
            f"{qc_without_outcome!r}"
        )

    if qc_enabled:
        accepted_set = set(accept_indices)
        for feature_index, (
            candidate_idx,
            description,
            outcome_record,
        ) in outcome_by_feature.items():
            qc_record = qc_by_feature[feature_index]
            if qc_record["description"] != description:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} has a QC "
                    f"description mismatch for feature_index={feature_index}"
                )
            qc_outcome = QcOutcome(qc_record["outcome"])
            if candidate_idx in accepted_set:
                expected_qc_outcome = QcOutcome.ACCEPTED
                stored_score = outcome_record.get("qc_score")
            else:
                try:
                    rejection_reason = RejectedConceptReason(
                        outcome_record.get("reason")
                    )
                except (TypeError, ValueError) as exc:
                    raise AxBenchBaseError(
                        f"Per-rank Stage 0c checkpoint at {json_path} has "
                        f"invalid rejection reason for candidate "
                        f"{candidate_idx}: {outcome_record.get('reason')!r}"
                    ) from exc
                if (
                    rejection_reason
                    == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES
                ):
                    expected_qc_outcome = QcOutcome.ACCEPTED
                else:
                    expected_qc_outcome = qc_outcome
                    try:
                        mapped_reason = RejectedConceptReason.from_qc_outcome(
                            qc_outcome
                        )
                    except AxBenchBaseError as exc:
                        raise AxBenchBaseError(
                            f"Per-rank Stage 0c checkpoint at {json_path} has "
                            f"QC outcome={qc_outcome.value!r} incompatible "
                            f"with rejection reason={rejection_reason.value!r}"
                        ) from exc
                    if mapped_reason != rejection_reason:
                        raise AxBenchBaseError(
                            f"Per-rank Stage 0c checkpoint at {json_path} has "
                            f"QC outcome={qc_outcome.value!r} but rejection "
                            f"reason={rejection_reason.value!r} for candidate "
                            f"{candidate_idx}"
                        )
                stored_score = outcome_record.get("llm_judge_score")
            if qc_outcome != expected_qc_outcome:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} has QC "
                    f"outcome={qc_outcome.value!r} incompatible with terminal "
                    f"outcome for candidate {candidate_idx}"
                )
            if stored_score != qc_record["score"]:
                raise AxBenchBaseError(
                    f"Per-rank Stage 0c checkpoint at {json_path} has QC "
                    f"score disagreement for candidate {candidate_idx}: "
                    f"terminal outcome stores {stored_score!r}, QC record "
                    f"stores {qc_record['score']!r}"
                )

    expected_rows_by_candidate: dict[int, int] = {}
    for record_idx, (record, candidate_idx) in enumerate(
        zip(accept_records, accept_indices, strict=True)
    ):
        raw_count = record.get("verified_rows_count")
        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_accepts[{record_idx}].verified_rows_count must be a "
                f"strict int, got {raw_count!r}"
            )
        if raw_count != verification_target:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_accepts[{record_idx}].verified_rows_count="
                f"{raw_count} must equal fingerprinted "
                "n_verified_prefixes_per_concept_target="
                f"{verification_target}"
            )
        expected_rows_by_candidate[candidate_idx] = raw_count

    actual_rows_by_candidate = dict.fromkeys(accept_indices, 0)
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c parquet at {parquet_path} row {row_idx} "
                "must be an object"
            )
        candidate_idx = _validate_candidate_index(
            row.get("candidate_index"),
            context=(
                f"Per-rank Stage 0c parquet at {parquet_path} "
                f"row {row_idx}.candidate_index"
            ),
        )
        _require_in_candidate_pool(
            candidate_idx,
            context=(
                f"Per-rank Stage 0c parquet at {parquet_path} "
                f"row {row_idx}.candidate_index"
            ),
        )
        if candidate_idx not in actual_rows_by_candidate:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c parquet at {parquet_path} row {row_idx} "
                f"references candidate_index={candidate_idx}, which is not "
                "recorded exactly once in rank_accepts"
            )
        actual_rows_by_candidate[candidate_idx] += 1

    if actual_rows_by_candidate != expected_rows_by_candidate:
        raise AxBenchBaseError(
            "Per-rank Stage 0c checkpoint row-count mismatch between "
            f"{json_path} and {parquet_path}: expected per accepted candidate "
            f"{expected_rows_by_candidate!r}, found "
            f"{actual_rows_by_candidate!r}"
        )


def merge_per_rank_checkpoints(
    *,
    run_dir: Path,
    world_size: int,
    max_concepts: int,
    checkpoint_schema_version: int = _RANK_CHECKPOINT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Rank-0 final merge for the caller-selected checkpoint schema.

    Read all per-rank checkpoints, sort accepts by candidate_index, truncate
    to max_concepts, and assign canonical dense concept_ids in
    sorted-by-candidate-index order.

    Returns dict with:
        - 'accepted_concepts': list[Concept] with concept_id 0..N-1
        - 'accepted_records': validated accepted checkpoint records in the
          same candidate-index order (for merge-time live-pool comparison)
        - 'rejected_concepts': list[dict] (sorted by candidate_index)
        - 'rows': list[dict] for the final negative_mode_prefixes parquet
                  (concept_id remapped from candidate_index)
        - 'qc_per_feature': list[dict]
        - 'fingerprint': dict (verified consistent across ranks)
    """
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
    ):
        raise AxBenchBaseError(
            f"merge_per_rank_checkpoints requires world_size > 0, got "
            f"{world_size!r}"
        )
    if (
        not isinstance(max_concepts, int)
        or isinstance(max_concepts, bool)
        or max_concepts <= 0
    ):
        raise AxBenchBaseError(
            f"merge_per_rank_checkpoints requires max_concepts > 0, got "
            f"{max_concepts!r}"
        )

    rank_payloads: list[dict] = []
    rank_rows: list[list[dict]] = []
    missing_json_paths: list[Path] = []
    for rank in range(world_size):
        json_path = run_dir / f"stage0c_partial_rank_{rank}.json"
        parquet_path = run_dir / f"stage0c_partial_rank_{rank}.parquet"
        if not json_path.exists():
            missing_json_paths.append(json_path)
            continue
        payload, rows = read_per_rank_checkpoint(
            json_path,
            parquet_path,
            expected_schema_version=checkpoint_schema_version,
        )
        _validate_rank_checkpoint_state_for_merge(
            payload,
            rows,
            rank=rank,
            world_size=world_size,
            json_path=json_path,
            parquet_path=parquet_path,
        )
        rank_payloads.append(payload)
        rank_rows.append(rows)

    if missing_json_paths:
        if len(missing_json_paths) == world_size:
            prefix = "no per-rank checkpoint files found"
        else:
            prefix = "missing required per-rank checkpoint JSON file(s)"
        raise AxBenchBaseError(
            f"merge_per_rank_checkpoints({run_dir}, world_size={world_size}): "
            f"{prefix}: {[str(path) for path in missing_json_paths]!r}. Every "
            "current rank writes a JSON commit marker even when its stripe is "
            "empty; refusing to merge an incomplete rank set."
        )

    canonical_fp = rank_payloads[0]["fingerprint"]
    canonical_n_candidates = rank_payloads[0]["n_candidates"]
    for payload in rank_payloads[1:]:
        if _semantic_json_identity(payload["fingerprint"]) != (
            _semantic_json_identity(canonical_fp)
        ):
            raise AxBenchBaseError(
                "merge_per_rank_checkpoints: rank fingerprints differ — "
                "indicates corruption or concurrent runs with different "
                "configs. Delete per-rank partial files and restart."
            )
        if payload["n_candidates"] != canonical_n_candidates:
            raise AxBenchBaseError(
                "merge_per_rank_checkpoints: ranks declare different "
                "n_candidates values despite agreeing on a fingerprint — "
                "indicates corrupt or mixed checkpoint state. Delete per-rank "
                "partial files and restart."
            )

    all_accepts = [
        accept
        for payload in rank_payloads
        for accept in payload.get("rank_accepts", [])
    ]
    all_accepts.sort(key=lambda a: a["candidate_index"])
    checkpoint_accept_records = list(all_accepts)
    truncated_accepts = all_accepts[:max_concepts]
    candidate_index_to_concept_id: dict[int, int] = {
        a["candidate_index"]: cid
        for cid, a in enumerate(truncated_accepts)
    }

    accepted_concepts = [
        Concept(
            concept_id=cid,
            feature_index=a["feature_index"],
            layer=a["layer"],
            sae_source=a["sae_source"],
            description=a["description"],
            corpus_count=a["corpus_count"],
            description_source=a["description_source"],
        )
        for cid, a in enumerate(truncated_accepts)
    ]

    merged_rows: list[dict] = []
    for rows in rank_rows:
        for row in rows:
            cidx = row.get("candidate_index")
            if cidx in candidate_index_to_concept_id:
                new_row = dict(row)
                new_row["concept_id"] = candidate_index_to_concept_id[cidx]
                new_row.pop("candidate_index", None)
                merged_rows.append(new_row)
    # Deterministic row order: sort by canonical concept_id, then by source.
    merged_rows.sort(key=lambda r: (
        r["concept_id"],
        r.get("prefix_source_shard", 0),
        r.get("prefix_source_doc", 0),
        r.get("prefix_activating_seg_idx", 0),
    ))

    all_rejects = [
        reject
        for payload in rank_payloads
        for reject in payload.get("rank_rejects", [])
    ]
    all_rejects.sort(key=lambda r: r["candidate_index"])
    checkpoint_reject_records = list(all_rejects)
    # When the multi-GPU loop saturates ``max_concepts``, every rank that
    # has not yet rendezvoused at the all_gather barrier may have visited
    # owned candidates past the canonical "last accept" candidate index.
    # Those over-visit rejects are not in the single-GPU counterfactual
    # (ws=1 short-circuits at max_concepts), so truncate them out of the
    # merged ``rejected_concepts.json`` to preserve the byte-identity
    # contract with ws=1.
    #
    # When ``len(truncated_accepts) < max_concepts`` we instead kept every
    # candidate the ranks visited (pool was exhausted; the ws=1 loop also
    # would have visited the entire pool), so no filtering is required.
    if len(truncated_accepts) >= max_concepts and truncated_accepts:
        max_accept_candidate_index = truncated_accepts[-1]["candidate_index"]
        all_rejects = [
            r for r in all_rejects
            if r["candidate_index"] <= max_accept_candidate_index
        ]

    feature_to_candidate_index = {
        record["feature_index"]: record["candidate_index"]
        for record in (*checkpoint_accept_records, *checkpoint_reject_records)
    }
    indexed_qc: list[tuple[int, dict[str, Any]]] = []
    for payload in rank_payloads:
        for qc in payload.get("rank_qc_per_feature", []):
            feature_index = qc["feature_index"]
            candidate_idx = feature_to_candidate_index.get(feature_index)
            if candidate_idx is None:
                raise AxBenchBaseError(
                    "merge_per_rank_checkpoints: QC record for "
                    f"feature_index={feature_index} has no terminal checkpoint "
                    "outcome"
                )
            indexed_qc.append((candidate_idx, qc))
    if len(truncated_accepts) >= max_concepts and truncated_accepts:
        max_accept_candidate_index = truncated_accepts[-1]["candidate_index"]
        indexed_qc = [
            item for item in indexed_qc
            if item[0] <= max_accept_candidate_index
        ]
    indexed_qc.sort(key=lambda item: item[0])
    all_qc = [qc for _, qc in indexed_qc]

    return {
        "accepted_concepts": accepted_concepts,
        "accepted_records": truncated_accepts,
        "checkpoint_accept_records": checkpoint_accept_records,
        "checkpoint_reject_records": checkpoint_reject_records,
        "rejected_concepts": all_rejects,
        "rows": merged_rows,
        "qc_per_feature": all_qc,
        "fingerprint": canonical_fp,
        "n_candidates": canonical_n_candidates,
    }


__all__ = [
    "is_my_candidate",
    "merge_per_rank_checkpoints",
    "my_candidate_block",
    "read_per_rank_checkpoint",
    "validate_per_rank_fingerprint",
    "write_per_rank_checkpoint",
]
