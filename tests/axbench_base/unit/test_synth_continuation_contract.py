import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest

from steering.eval.axbench_base.synth_continuation_contract import (
    STAGE1_CONTINUATION_CONTRACT_METADATA_KEY,
    STAGE1_CONTINUATION_CONTRACT_VERSION,
    Stage1ContinuationArtifactRole,
    schema_with_stage1_continuation_contract,
    stage1_continuation_contract_from_schema,
    stage1_continuation_contract_payload,
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def _schema(
    role: Stage1ContinuationArtifactRole = "stage1b_positive",
) -> pa.Schema:
    return schema_with_stage1_continuation_contract(
        pa.schema([("value", pa.int32())]),
        role=role,
    )


def _replace_payload(schema: pa.Schema, transform) -> pa.Schema:
    metadata = dict(schema.metadata or {})
    key = STAGE1_CONTINUATION_CONTRACT_METADATA_KEY.encode("utf-8")
    payload = json.loads(metadata[key])
    transform(payload)
    metadata[key] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return schema.with_metadata(metadata)


def test_payload_records_every_continuation_semantic_exactly() -> None:
    assert stage1_continuation_contract_payload("stage1b_positive") == {
        "contract_version": STAGE1_CONTINUATION_CONTRACT_VERSION,
        "artifact_role": "stage1b_positive",
        "protocol_version": 1,
        "request_identity": "canonical_json_sha256_attempt_scoped_v1",
        "marker_derivation": "collision_checked_semantic_sha256_24hex_v1",
        "response_envelope": (
            "single_begin_first_end_truncated_final_bracket_tolerated_v1"
        ),
        "prefix_echo_chars": 96,
        "minimum_prefix_echo_chars": 16,
        "allowed_left_echo_truncation_chars": 1,
        "allowed_echo_separators": [
            "\r\n\r\n",
            "\n\n",
            "\r\n",
            "\n",
            " ",
            "",
        ],
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


def test_schema_round_trip_preserves_unrelated_metadata() -> None:
    base = pa.schema([("value", pa.int32())], metadata={b"other": b"kept"})
    schema = schema_with_stage1_continuation_contract(
        base,
        role="stage1c_negative",
    )

    assert schema.metadata is not None
    assert schema.metadata[b"other"] == b"kept"
    assert stage1_continuation_contract_from_schema(schema) == (
        stage1_continuation_contract_payload("stage1c_negative")
    )


def test_validator_accepts_exact_role_and_rejects_role_swap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positive.parquet"
    schema = _schema()

    assert validate_stage1_continuation_contract(
        path,
        consumer="test",
        expected_role="stage1b_positive",
        schema=schema,
    ) == stage1_continuation_contract_payload("stage1b_positive")
    with pytest.raises(AxBenchBaseError, match="artifact_role"):
        validate_stage1_continuation_contract(
            path,
            consumer="test",
            expected_role="stage1c_negative",
            schema=schema,
        )


def test_validator_rejects_missing_contract(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="is missing.*continuation_contract"):
        validate_stage1_continuation_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            expected_role="stage1b_positive",
            schema=pa.schema([("value", pa.int32())]),
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"[]",
        b"{",
        b'{"contract_version":"a","contract_version":"b"}',
        b'{"value":NaN}',
        b"\xff",
    ],
)
def test_decoder_rejects_malformed_or_ambiguous_metadata(
    encoded: bytes,
) -> None:
    key = STAGE1_CONTINUATION_CONTRACT_METADATA_KEY.encode("utf-8")
    schema = pa.schema([("value", pa.int32())], metadata={key: encoded})
    with pytest.raises(AxBenchBaseError, match="invalid.*continuation_contract"):
        stage1_continuation_contract_from_schema(schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "legacy"),
        ("protocol_version", True),
        ("prefix_echo_chars", 95),
        ("allowed_left_echo_truncation_chars", 2),
        ("allowed_echo_separators", ["\n"]),
        ("copy_guard", "none"),
    ],
)
def test_validator_rejects_any_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    schema = _replace_payload(
        _schema(),
        lambda payload: payload.__setitem__(field, value),
    )
    with pytest.raises(AxBenchBaseError, match=field):
        validate_stage1_continuation_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            expected_role="stage1b_positive",
            schema=schema,
        )


def test_unknown_role_is_rejected_before_publication() -> None:
    with pytest.raises(AxBenchBaseError, match="unknown.*role"):
        schema_with_stage1_continuation_contract(
            pa.schema([("value", pa.int32())]),
            role=cast(Stage1ContinuationArtifactRole, "other"),
        )
