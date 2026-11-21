import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from steering.eval.axbench_base.config import AxBenchBaseConfig, SynthConfig
from steering.eval.axbench_base.positive_selection_contract import (
    POSITIVE_CHECKPOINT_SELECTION_VERSION,
    POSITIVE_INITIAL_COMPAT_TARGET,
    POSITIVE_POST_COMPATIBILITY_BATCHING,
    POSITIVE_POST_PROVIDER_QC_BATCHING,
    POSITIVE_SELECTION_CONTRACT_METADATA_KEY,
    positive_artifact_selection_contract_payload,
    positive_checkpoint_selection_payload,
    positive_selection_contract_from_schema,
    positive_selection_contract_payload,
    schema_with_default_positive_selection_contract,
    schema_with_positive_selection_contract,
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def _cfg(**changes: Any) -> AxBenchBaseConfig:
    values: dict[str, Any] = {
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/features/feature_statistics.safetensors",
        "decoder_only": {"source": "base", "model_tag": "base", "step": 1},
        "encdec": {"source": "encdec", "model_tag": "encdec", "step": 1},
        "device": "cpu",
        "output_root": "/tmp/output",
        "accept_threshold": 0.3,
        "positive_prefix_headroom_ceiling": 0.9,
        "min_prefix_chars": 100,
        "max_prefix_chars": 2_048,
        "qc_batch_size": 16,
    }
    values.update(changes)
    return AxBenchBaseConfig(**values)


def _schema(cfg: AxBenchBaseConfig | None = None) -> pa.Schema:
    return schema_with_positive_selection_contract(
        pa.schema([("value", pa.int32())]),
        _cfg() if cfg is None else cfg,
    )


def _replace_payload(
    schema: pa.Schema,
    transform,
) -> pa.Schema:
    metadata = dict(schema.metadata or {})
    key = POSITIVE_SELECTION_CONTRACT_METADATA_KEY.encode("utf-8")
    payload = json.loads(metadata[key])
    transform(payload)
    metadata[key] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return schema.with_metadata(metadata)


def test_payload_records_every_selection_semantic_exactly() -> None:
    payload = positive_selection_contract_payload(_cfg())

    assert payload == {
        "contract_version": POSITIVE_CHECKPOINT_SELECTION_VERSION,
        "percentile_currency": "raw_unquantized_percentile_points_v1",
        "accept_fraction": 0.3,
        "required_delta": 30.0,
        "headroom_ceiling_fraction": 0.9,
        "completion_ceiling": 90.0,
        "maximum_seed_percentile": 60.0,
        "min_prefix_chars": 100,
        "max_prefix_chars": 2_048,
        "prefix_window": "recent_character_suffix_v1",
        "prefix_baseline": (
            "ragged_right_padded_batch_prefix_alone_single_bos_excluded_v1"
        ),
        "prefix_score_batch_size": 16,
        "post_provider_qc_batch_size": 16,
        "post_provider_qc_batching": POSITIVE_POST_PROVIDER_QC_BATCHING,
        "continuation_span": (
            "joint_tokenization_completion_single_bos_excluded_v1"
        ),
        "continuation_copy_guard": (
            "authenticated_stage1_continuation_and_copy_guards_v2"
        ),
        "initial_compatibility_target": POSITIVE_INITIAL_COMPAT_TARGET,
        "post_compatibility_batching": POSITIVE_POST_COMPATIBILITY_BATCHING,
        "positive_max_attempts_per_seed": 8,
        "positive_zero_hit_bail_after": 2000,
        "candidate_ordering": (
            "seed_index_full_shuffle_clean_then_dirty_percentile_stable_v1"
        ),
        "replacement_semantics": (
            "lazy_atomic_complete_pair_target_refill_v1"
        ),
    }


@pytest.mark.parametrize("target", [1, 15, 16, 17, 71, 73, 100])
def test_selection_payload_is_target_independent(target: int) -> None:
    payload = positive_selection_contract_payload(
        _cfg(n_positives_per_concept=target)
    )

    assert payload == positive_selection_contract_payload(_cfg())
    assert "n_positives_per_concept" not in payload
    assert payload["post_provider_qc_batch_size"] == 16
    assert payload["post_provider_qc_batching"] == POSITIVE_POST_PROVIDER_QC_BATCHING
    assert payload["contract_version"] == POSITIVE_CHECKPOINT_SELECTION_VERSION


def test_checkpoint_selection_payload_is_target_independent() -> None:
    payload = positive_checkpoint_selection_payload(_cfg())

    assert payload == positive_checkpoint_selection_payload(
        _cfg(n_positives_per_concept=100)
    )
    assert "n_positives_per_concept" not in payload
    assert payload["contract_version"] == POSITIVE_CHECKPOINT_SELECTION_VERSION
    assert payload["initial_compatibility_target"] == 72
    assert payload["post_provider_qc_batch_size"] == 16
    assert payload["post_provider_qc_batching"] == POSITIVE_POST_PROVIDER_QC_BATCHING


def test_post_provider_batch_size_is_artifact_provenance_and_identity() -> None:
    singleton = positive_artifact_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=1))
    )
    capped = positive_artifact_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=8))
    )

    assert singleton["post_provider_qc_batch_size"] == 1
    assert capped["post_provider_qc_batch_size"] == 8
    assert singleton != capped
    assert positive_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=1))
    ) != positive_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=8))
    )
    assert positive_selection_contract_payload(_cfg())[
        "post_provider_qc_batch_size"
    ] == 16
    assert positive_artifact_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=16))
    ) == positive_artifact_selection_contract_payload(
        _cfg(synth=SynthConfig(max_concurrent=32))
    )


def test_schema_round_trip_preserves_unrelated_metadata() -> None:
    base = pa.schema([("value", pa.int32())], metadata={b"other": b"kept"})
    schema = schema_with_positive_selection_contract(base, _cfg())

    assert schema.metadata is not None
    assert schema.metadata[b"other"] == b"kept"
    assert positive_selection_contract_from_schema(schema) == (
        positive_artifact_selection_contract_payload(_cfg())
    )


def test_default_schema_helper_emits_the_exact_current_contract() -> None:
    schema = schema_with_default_positive_selection_contract(
        pa.schema([("value", pa.int32())]),
        accept_threshold=0.3,
    )

    assert positive_selection_contract_from_schema(schema) == (
        positive_artifact_selection_contract_payload(
            _cfg(
                feature_dir="",
                feature_statistics_path="",
                decoder_only={},
                encdec={},
                output_root=None,
            )
        )
    )


def test_validator_accepts_exact_contract_with_and_without_expected_cfg(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positive.parquet"
    schema = _schema()

    assert validate_positive_selection_contract(
        path,
        consumer="test",
        schema=schema,
    ) == positive_artifact_selection_contract_payload(_cfg())
    assert validate_positive_selection_contract(
        path,
        consumer="test",
        schema=schema,
        expected_cfg=_cfg(),
    ) == positive_artifact_selection_contract_payload(_cfg())


def test_validator_accepts_exact_v6_contract(tmp_path: Path) -> None:
    cfg = _cfg(n_positives_per_concept=73)

    assert validate_positive_selection_contract(
        tmp_path / "positive.parquet",
        consumer="test",
        schema=_schema(cfg),
        expected_cfg=cfg,
    ) == positive_artifact_selection_contract_payload(cfg)


def test_validator_rejects_missing_contract(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="is missing.*selection_contract"):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=pa.schema([("value", pa.int32())]),
        )


def test_validator_rejects_v2_with_fresh_run_tag_guidance(
    tmp_path: Path,
) -> None:
    v3_only_fields = {
        "n_positives_per_concept",
        "positive_max_attempts_per_seed",
        "positive_zero_hit_bail_after",
        "candidate_ordering",
        "replacement_semantics",
    }

    def restore_exact_v2_shape(payload: dict[str, Any]) -> None:
        payload["contract_version"] = "positive_prefix_headroom_v2"
        for field in v3_only_fields:
            payload.pop(field)

    schema = _replace_payload(
        _schema(),
        restore_exact_v2_shape,
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"positive_prefix_headroom_v2.*fresh run_tag",
    ):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=schema,
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
def test_decoder_rejects_malformed_or_ambiguous_metadata(encoded: bytes) -> None:
    key = POSITIVE_SELECTION_CONTRACT_METADATA_KEY.encode("utf-8")
    schema = pa.schema([("value", pa.int32())], metadata={key: encoded})
    with pytest.raises(AxBenchBaseError, match="invalid.*selection_contract"):
        positive_selection_contract_from_schema(schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "legacy"),
        ("percentile_currency", "uint8"),
        ("prefix_window", "first_chars"),
        ("prefix_baseline", "scalar"),
        ("continuation_span", "whole_pair"),
        ("continuation_copy_guard", "none"),
        ("candidate_ordering", "random"),
        ("replacement_semantics", "fixed_roster"),
        ("post_provider_qc_batching", "timing_dependent"),
        ("required_delta", "30.0"),
        ("required_delta", 29.0),
        ("completion_ceiling", 89.0),
        ("maximum_seed_percentile", 59.0),
        ("prefix_score_batch_size", 0),
        ("prefix_score_batch_size", True),
        ("post_provider_qc_batch_size", 0),
        ("post_provider_qc_batch_size", 17),
        ("post_provider_qc_batch_size", True),
        ("n_positives_per_concept", 0),
        ("positive_max_attempts_per_seed", 0),
        ("positive_zero_hit_bail_after", 0),
        ("positive_zero_hit_bail_after", True),
        ("min_prefix_chars", -1),
        ("max_prefix_chars", 0),
        ("max_prefix_chars", 99),
    ],
)
def test_validator_rejects_noncanonical_or_inconsistent_payload_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    schema = _replace_payload(_schema(), lambda payload: payload.__setitem__(field, value))
    with pytest.raises(AxBenchBaseError):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=schema,
        )


def test_validator_rejects_finite_json_syntax_that_overflows_to_infinity(
    tmp_path: Path,
) -> None:
    metadata = dict(_schema().metadata or {})
    key = POSITIVE_SELECTION_CONTRACT_METADATA_KEY.encode("utf-8")
    original = metadata[key]
    metadata[key] = original.replace(
        b'"required_delta":30.0',
        b'"required_delta":1e999',
    )
    assert metadata[key] != original

    with pytest.raises(AxBenchBaseError, match="non-finite.*required_delta"):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=pa.schema([("value", pa.int32())], metadata=metadata),
        )


def test_validator_rejects_missing_and_unexpected_fields(tmp_path: Path) -> None:
    missing = _replace_payload(_schema(), lambda payload: payload.pop("prefix_window"))
    extra = _replace_payload(_schema(), lambda payload: payload.__setitem__("extra", 1))
    for schema in (missing, extra):
        with pytest.raises(AxBenchBaseError, match="fields mismatch"):
            validate_positive_selection_contract(
                tmp_path / "positive.parquet",
                consumer="test",
                schema=schema,
            )


@pytest.mark.parametrize(
    ("cfg", "transform", "match"),
    [
        (
            _cfg(),
            lambda payload: payload.__setitem__(
                "n_positives_per_concept", 71
            ),
            "valid only at 72",
        ),
        (
            _cfg(n_positives_per_concept=73),
            lambda payload: payload.__setitem__(
                "n_positives_per_concept", 72
            ),
            "invalid at",
        ),
        (
            _cfg(n_positives_per_concept=73),
            lambda payload: payload.__setitem__(
                "n_positives_per_concept", 0
            ),
            "non-positive.*n_positives_per_concept",
        ),
        (
            _cfg(n_positives_per_concept=73),
            lambda payload: payload.__setitem__(
                "initial_compatibility_target", 71
            ),
            "initial_compatibility_target",
        ),
        (
            _cfg(n_positives_per_concept=73),
            lambda payload: payload.__setitem__(
                "post_compatibility_batching", "partial"
            ),
            "post_compatibility_batching",
        ),
    ],
)
def test_validator_rejects_inconsistent_current_contracts(
    tmp_path: Path,
    cfg: AxBenchBaseConfig,
    transform,
    match: str,
) -> None:
    schema = _replace_payload(_schema(cfg), transform)
    with pytest.raises(AxBenchBaseError, match=match):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=schema,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accept_threshold", 0.2),
        ("positive_prefix_headroom_ceiling", 0.95),
        ("min_prefix_chars", 101),
        ("max_prefix_chars", 2_047),
        ("qc_batch_size", 8),
        ("n_positives_per_concept", 71),
        ("positive_max_attempts_per_seed", 7),
        ("positive_zero_hit_bail_after", 1999),
    ],
)
def test_expected_config_comparison_rejects_every_configurable_semantic(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(AxBenchBaseError, match="selection contract mismatch"):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=_schema(),
            expected_cfg=_cfg(**{field: value}),
        )


def test_expected_config_comparison_is_type_strict(tmp_path: Path) -> None:
    schema = _replace_payload(
        _schema(),
        lambda payload: payload.__setitem__("prefix_score_batch_size", 16.0),
    )
    with pytest.raises(AxBenchBaseError, match="non-integer"):
        validate_positive_selection_contract(
            tmp_path / "positive.parquet",
            consumer="test",
            schema=schema,
            expected_cfg=_cfg(),
        )
