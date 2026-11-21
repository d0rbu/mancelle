"""Versioned provenance for authenticated Stage 1 synth continuations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.eval.axbench_base.synth.continuation_guards import (
    MIN_PREFIX_SUFFIX_ECHO_CHARS,
    PREFIX_ECHO_CHARS,
    PREFIX_ECHO_SEPARATORS,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    STAGE1_CONTINUATION_PROTOCOL_VERSION,
)
from steering.eval.axbench_base.types import AxBenchBaseError

STAGE1_CONTINUATION_CONTRACT_METADATA_KEY = (
    "axbench_stage1_continuation_contract"
)
STAGE1_CONTINUATION_CONTRACT_VERSION = (
    "authenticated_stage1_continuation_v1"
)
type Stage1ContinuationArtifactRole = Literal[
    "stage1b_positive",
    "stage1c_negative",
]
_ROLES = frozenset({"stage1b_positive", "stage1c_negative"})


def stage1_continuation_contract_payload(
    role: Stage1ContinuationArtifactRole,
) -> dict[str, Any]:
    """Return the complete semantic identity of accepted Stage 1 text."""
    if role not in _ROLES:
        raise AxBenchBaseError(
            f"unknown Stage 1 continuation artifact role {role!r}"
        )
    return {
        "contract_version": STAGE1_CONTINUATION_CONTRACT_VERSION,
        "artifact_role": role,
        "protocol_version": STAGE1_CONTINUATION_PROTOCOL_VERSION,
        "request_identity": "canonical_json_sha256_attempt_scoped_v1",
        "marker_derivation": (
            "collision_checked_semantic_sha256_24hex_v1"
        ),
        "response_envelope": (
            "single_begin_first_end_truncated_final_bracket_tolerated_v1"
        ),
        "prefix_echo_chars": PREFIX_ECHO_CHARS,
        "minimum_prefix_echo_chars": MIN_PREFIX_SUFFIX_ECHO_CHARS,
        "allowed_left_echo_truncation_chars": 1,
        "allowed_echo_separators": list(PREFIX_ECHO_SEPARATORS),
        "continuation_boundary": "immediate_after_echo_v1",
        "terminal_trim_characters": "space_tab_cr_lf_v1",
        "copy_guard": (
            "full_prefix_suffix_overlap_source_phrase_repeated_phrase_v2"
        ),
        "special_token_guard": (
            "full_response_before_truncation_then_stored_window_v1"
        ),
        "stored_window_guard": (
            "decode_output_length_tokens_then_revalidate_v1"
        ),
    }


def _contract_json(role: Stage1ContinuationArtifactRole) -> str:
    return json.dumps(
        stage1_continuation_contract_payload(role),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def schema_with_stage1_continuation_contract(
    schema: pa.Schema,
    *,
    role: Stage1ContinuationArtifactRole,
) -> pa.Schema:
    """Stamp a synth-corpus schema with its exact continuation semantics."""
    metadata = dict(schema.metadata or {})
    metadata[STAGE1_CONTINUATION_CONTRACT_METADATA_KEY.encode("utf-8")] = (
        _contract_json(role).encode("utf-8")
    )
    return schema.with_metadata(metadata)


def stage1_continuation_contract_from_schema(
    schema: pa.Schema,
) -> dict[str, Any] | None:
    """Decode continuation metadata, rejecting malformed or ambiguous JSON."""
    raw = (schema.metadata or {}).get(
        STAGE1_CONTINUATION_CONTRACT_METADATA_KEY.encode("utf-8")
    )
    if raw is None:
        return None
    try:
        encoded = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        payload = json.loads(
            encoded,
            parse_constant=reject_nonstandard_json_constant,
            object_pairs_hook=strict_json_object,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"invalid {STAGE1_CONTINUATION_CONTRACT_METADATA_KEY!r} "
            "parquet metadata"
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBaseError(
            f"invalid {STAGE1_CONTINUATION_CONTRACT_METADATA_KEY!r} parquet "
            "metadata: expected a JSON object"
        )
    return payload


def validate_stage1_continuation_contract(
    path: Path,
    *,
    consumer: str,
    expected_role: Stage1ContinuationArtifactRole,
    schema: pa.Schema | None = None,
) -> dict[str, Any]:
    """Fail loudly unless a synth corpus has the exact current contract."""
    parquet_schema = schema if schema is not None else pq.read_schema(path)
    actual = stage1_continuation_contract_from_schema(parquet_schema)
    if actual is None:
        raise AxBenchBaseError(
            f"{consumer}: synth corpus at {path} is missing "
            f"{STAGE1_CONTINUATION_CONTRACT_METADATA_KEY!r} metadata. "
            "Regenerate Stage 1 under a fresh run_tag with the current code."
        )
    expected = stage1_continuation_contract_payload(expected_role)
    mismatches: dict[str, dict[str, Any]] = {}
    for field in sorted(set(expected) | set(actual)):
        expected_present = field in expected
        actual_present = field in actual
        if (
            expected_present != actual_present
            or (
                expected_present
                and actual_present
                and type(expected[field]) is not type(actual[field])
            )
            or expected.get(field) != actual.get(field)
        ):
            mismatches[field] = {
                "expected": (
                    expected[field] if expected_present else "<absent>"
                ),
                "actual": actual[field] if actual_present else "<absent>",
            }
    if mismatches:
        raise AxBenchBaseError(
            f"{consumer}: synth corpus at {path} has a Stage 1 continuation "
            f"contract mismatch: {mismatches}. Regenerate Stage 1 under a "
            "fresh run_tag with the current code."
        )
    return actual


__all__ = [
    "STAGE1_CONTINUATION_CONTRACT_METADATA_KEY",
    "STAGE1_CONTINUATION_CONTRACT_VERSION",
    "Stage1ContinuationArtifactRole",
    "schema_with_stage1_continuation_contract",
    "stage1_continuation_contract_from_schema",
    "stage1_continuation_contract_payload",
    "validate_stage1_continuation_contract",
]
