"""Independent property coverage for parsed autointerp runtime domains."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, assert_type, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.autointerp_filter.batch_validation import (
    BatchCalibrationContract,
    BatchDifferenceMetrics,
    BatchToleranceProfile,
    NumericTolerance,
)
from steering.eval.autointerp_filter.batching import (
    MaterializationPhase,
    PreparedBatch,
    PreparedDoc,
    TargetBlock,
    build_target_blocks,
)
from steering.eval.autointerp_filter.source_index import (
    SourceIndexRequest,
    _scan_source_index_shard_file_topk_vectorized,
)
from steering.eval.autointerp_filter.types import (
    AutoInterpBatchValidationError,
    AutoInterpDataError,
    ScoringTarget,
    SourceDocRef,
    TargetKind,
)
from steering.refined_types import (
    AtLeastTwoInt,
    BinaryInt,
    NonEmptyNonNegativeIntTuple,
    NonEmptyStr,
    NonNegativeFiniteNumber,
    NonNegativeInt,
    NonNegativeIntTuple,
    PositiveInt,
    ZeroInt,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)

_VALID_NON_NEGATIVE_INT = st.integers(min_value=0, max_value=10**9)
_VALID_POSITIVE_INT = st.integers(min_value=1, max_value=10**9)
_VALID_FINITE_NON_NEGATIVE = st.one_of(
    st.integers(min_value=0, max_value=10**100),
    st.floats(
        min_value=0.0,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_POSITIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=0),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_NON_NEGATIVE_FINITE = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.just(10**1000),
    st.floats(max_value=-1e-300, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.just(float("inf")),
    st.none(),
    st.text(),
)


def _target(feature_id: int) -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=f"feature:{feature_id}",
        display_name=f"feature_{feature_id}",
    )


def _doc_ref() -> SourceDocRef:
    return SourceDocRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=0,
        segment_token_start=0,
        segment_token_end=4,
    )


def _prepared_doc(
    *,
    block_ordinal: Any = 0,
    slot: Any = 0,
    target_key: Any = "feature:0",
    feature_id: Any = 0,
    source_text_hash: Any = "source-hash",
    model_token_ids: Any = (2, 11, 12),
    real_token_ids: Any = (11, 12),
    token_strings: Any = ("a", "b"),
    token_offset: Any = 0,
    activation_slice_start: Any = 1,
) -> PreparedDoc:
    return PreparedDoc(
        block_ordinal=block_ordinal,
        slot=slot,
        target_key=target_key,
        feature_id=feature_id,
        phase=MaterializationPhase.positive,
        doc_ref=_doc_ref(),
        source_text_hash=source_text_hash,
        model_token_ids=model_token_ids,
        real_token_ids=real_token_ids,
        token_strings=token_strings,
        token_offset=token_offset,
        activation_slice_start=activation_slice_start,
    )


def _contract() -> BatchCalibrationContract:
    return BatchCalibrationContract(
        model_name="model",
        model_dtype="bfloat16",
        activation_type="residual",
        layer=0,
        segment_size=1,
        sae_release="release",
        sae_id="sae",
        hook_name="hook",
    )


def _tolerance() -> NumericTolerance:
    return NumericTolerance(max_abs=1.0, p99_abs=0.5, mean_abs=0.25)


def _request(tmp_path: Path, **updates: Any) -> SourceIndexRequest:
    values: dict[str, Any] = {
        "feature_dir": tmp_path / "features",
        "data_dir": None,
        "index_path": tmp_path / "source.sqlite",
        "manifest_path": tmp_path / "source.manifest.json",
        "shard_paths": (),
        "parquet_paths": (),
        "target_feature_ids": (),
        "n_sae_features": 1,
        "max_positive_refs_per_feature": 1,
        "max_intruder_refs": 1,
        "seed": 0,
        "feature_fingerprint": "feature",
        "data_fingerprint": "data",
        "context_independent_features": True,
        "ignore_bos_features": True,
        "chunk_size": 1,
        "scan_workers": 1,
    }
    values.update(updates)
    return SourceIndexRequest(**values)


@PROPERTY_SETTINGS
@given(value=st.integers(min_value=-10, max_value=10))
def test_binary_and_zero_phantoms_exactly_match_their_integer_domains(value: int) -> None:
    assert isinstance(value, BinaryInt) is (value in (0, 1))
    assert isinstance(value, ZeroInt) is (value == 0)
    assert not isinstance(cast(Any, value == 0), BinaryInt)
    assert not isinstance(cast(Any, value == 0), ZeroInt)


@PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.binary(max_size=100),
        st.none(),
    )
)
def test_non_empty_string_phantom_preserves_whitespace_and_rejects_empty_values(
    value: object,
) -> None:
    assert isinstance(value, NonEmptyStr) is (isinstance(value, str) and bool(value))


@PROPERTY_SETTINGS
@given(
    feature_id=_VALID_NON_NEGATIVE_INT,
    batch_size=st.integers(min_value=1, max_value=256),
    ordinal=_VALID_NON_NEGATIVE_INT,
)
def test_target_blocks_accept_the_complete_scalar_domain_and_store_nominal_fields(
    feature_id: int,
    batch_size: int,
    ordinal: int,
) -> None:
    block_id, slot = divmod(feature_id, batch_size)
    slots: list[ScoringTarget | None] = [None] * batch_size
    slots[slot] = _target(feature_id)
    block = TargetBlock(
        ordinal=ordinal,
        block_id=block_id,
        batch_size=batch_size,
        targets_by_slot=tuple(slots),
    )
    assert isinstance(block.ordinal, NonNegativeInt)
    assert isinstance(block.block_id, NonNegativeInt)
    assert isinstance(block.batch_size, PositiveInt)
    assert block.targets == (_target(feature_id),)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("ordinal", "block_id")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_target_blocks_reject_every_invalid_non_negative_identity(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "ordinal": 0,
        "block_id": 0,
        "batch_size": 1,
        "targets_by_slot": (_target(0),),
    }
    values[field_name] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        TargetBlock(**values)


@PROPERTY_SETTINGS
@given(value=_INVALID_POSITIVE_INT)
def test_target_blocks_reject_every_invalid_positive_batch_size(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="batch_size"):
        build_target_blocks([_target(0)], batch_size=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    block_ordinal=_VALID_NON_NEGATIVE_INT,
    slot=_VALID_NON_NEGATIVE_INT,
    feature_id=_VALID_NON_NEGATIVE_INT,
    token_offset=_VALID_NON_NEGATIVE_INT,
    activation_slice_start=st.integers(min_value=0, max_value=1),
    real_token_ids=st.lists(
        _VALID_NON_NEGATIVE_INT,
        min_size=1,
        max_size=12,
    ),
)
def test_prepared_docs_accept_valid_domains_and_store_nominal_fields(
    block_ordinal: int,
    slot: int,
    feature_id: int,
    token_offset: int,
    activation_slice_start: int,
    real_token_ids: list[int],
) -> None:
    real = tuple(real_token_ids)
    model = real if activation_slice_start == 0 else (2, *real)
    doc = _prepared_doc(
        block_ordinal=block_ordinal,
        slot=slot,
        target_key=f"feature:{feature_id}",
        feature_id=feature_id,
        model_token_ids=model,
        real_token_ids=real,
        token_strings=tuple("token" for _ in real),
        token_offset=token_offset,
        activation_slice_start=activation_slice_start,
    )
    assert isinstance(doc.block_ordinal, NonNegativeInt)
    assert isinstance(doc.target_key, NonEmptyStr)
    assert isinstance(doc.model_token_ids, NonEmptyNonNegativeIntTuple)
    assert isinstance(doc.activation_slice_start, BinaryInt)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("block_ordinal", "slot", "feature_id", "token_offset")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_prepared_docs_reject_every_invalid_non_negative_scalar(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _prepared_doc(**{field_name: value})


@PROPERTY_SETTINGS
@given(
    prefix=st.lists(_VALID_NON_NEGATIVE_INT, max_size=8),
    invalid=_INVALID_NON_NEGATIVE_INT,
    suffix=st.lists(_VALID_NON_NEGATIVE_INT, max_size=8),
)
def test_prepared_docs_check_every_token_id_without_sampling(
    prefix: list[int],
    invalid: object,
    suffix: list[int],
) -> None:
    token_ids = (*prefix, invalid, *suffix)
    with pytest.raises(BeartypeCallHintParamViolation, match="real_token_ids"):
        _prepared_doc(
            model_token_ids=(2,),
            real_token_ids=token_ids,
            token_strings=tuple("token" for _ in token_ids),
        )


@PROPERTY_SETTINGS
@given(
    real_token_ids=st.lists(
        st.integers(min_value=0, max_value=10**6),
        min_size=1,
        max_size=12,
    )
)
def test_prepared_docs_retain_explicit_model_slice_relationships(
    real_token_ids: list[int],
) -> None:
    real = tuple(real_token_ids)
    mismatched_model = (2, real[0] + 1, *real[1:])
    with pytest.raises(AutoInterpDataError, match="does not contain"):
        _prepared_doc(
            model_token_ids=mismatched_model,
            real_token_ids=real,
            token_strings=tuple("token" for _ in real),
        )


@PROPERTY_SETTINGS
@given(
    model_width=st.integers(min_value=3, max_value=10**6),
    invalid_width=_INVALID_POSITIVE_INT,
    invalid_pad=st.one_of(
        st.booleans(),
        st.integers(max_value=-1),
        st.integers(min_value=1),
        st.floats(),
        st.none(),
    ),
)
def test_prepared_batches_store_valid_domains_and_reject_invalid_pinned_scalars(
    model_width: int,
    invalid_width: object,
    invalid_pad: object,
) -> None:
    block = TargetBlock(ordinal=0, block_id=0, batch_size=1, targets_by_slot=(_target(0),))
    valid = PreparedBatch(block=block, rows=(_prepared_doc(),), model_width=model_width)
    assert isinstance(valid.model_width, PositiveInt)
    assert isinstance(valid.pad_token_id, ZeroInt)
    with pytest.raises(BeartypeCallHintParamViolation, match="model_width"):
        PreparedBatch(block=block, rows=(_prepared_doc(),), model_width=cast(Any, invalid_width))
    with pytest.raises(BeartypeCallHintParamViolation, match="pad_token_id"):
        PreparedBatch(
            block=block,
            rows=(_prepared_doc(),),
            model_width=model_width,
            pad_token_id=cast(Any, invalid_pad),
        )


@PROPERTY_SETTINGS
@given(a=_VALID_FINITE_NON_NEGATIVE, b=_VALID_FINITE_NON_NEGATIVE, c=_VALID_FINITE_NON_NEGATIVE)
def test_numeric_tolerances_accept_complete_finite_domain_and_relationships(
    a: int | float,
    b: int | float,
    c: int | float,
) -> None:
    maximum = max(a, b, c)
    tolerance = NumericTolerance(max_abs=maximum, p99_abs=a, mean_abs=b)
    assert isinstance(tolerance.max_abs, NonNegativeFiniteNumber)
    assert tolerance.p99_abs <= tolerance.max_abs
    assert tolerance.mean_abs <= tolerance.max_abs


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("max_abs", "p99_abs", "mean_abs")),
    value=_INVALID_NON_NEGATIVE_FINITE,
)
def test_numeric_tolerances_reject_every_invalid_finite_bound(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {"max_abs": 1.0, "p99_abs": 0.5, "mean_abs": 0.25}
    values[field_name] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        NumericTolerance(**values)


@PROPERTY_SETTINGS
@given(maximum=st.floats(min_value=0.0, max_value=1e100, allow_nan=False, allow_infinity=False))
def test_numeric_tolerances_retain_cross_field_ordering(maximum: float) -> None:
    exceeded = maximum + 1.0 if maximum < 1e100 else float("inf")
    if exceeded == maximum or not isinstance(exceeded, NonNegativeFiniteNumber):
        exceeded = 1.0
        maximum = 0.0
    with pytest.raises(AutoInterpBatchValidationError, match="must not exceed"):
        NumericTolerance(max_abs=maximum, p99_abs=exceeded, mean_abs=0.0)


@PROPERTY_SETTINGS
@given(
    layer=_VALID_NON_NEGATIVE_INT,
    segment_size=_VALID_POSITIVE_INT,
    batch_size=st.integers(min_value=2, max_value=10**9),
    digest=st.one_of(st.none(), st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)),
)
def test_calibration_profiles_accept_valid_domains_and_store_nominal_fields(
    layer: int,
    segment_size: int,
    batch_size: int,
    digest: str | None,
) -> None:
    contract = BatchCalibrationContract(
        model_name=" ",
        model_dtype="dtype",
        activation_type="activation",
        layer=layer,
        segment_size=segment_size,
        sae_release="release",
        sae_id="sae",
        hook_name="hook",
    )
    profile = BatchToleranceProfile(
        name="profile",
        batch_size=batch_size,
        selected_sae=_tolerance(),
        raw_residual=_tolerance(),
        calibration_contract=contract,
        runtime_fingerprint_sha256=digest,
    )
    assert isinstance(contract.layer, NonNegativeInt)
    assert isinstance(contract.segment_size, PositiveInt)
    assert isinstance(profile.batch_size, AtLeastTwoInt)
    assert profile.runtime_fingerprint_sha256 == digest


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("layer", "segment_size", "batch_size")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_calibration_profiles_reject_invalid_integer_domains(
    field_name: str,
    value: object,
) -> None:
    if field_name in {"layer", "segment_size"}:
        values: dict[str, Any] = {
            "model_name": "model",
            "model_dtype": "dtype",
            "activation_type": "activation",
            "layer": 0,
            "segment_size": 1,
            "sae_release": "release",
            "sae_id": "sae",
            "hook_name": "hook",
        }
        values[field_name] = value
        with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
            BatchCalibrationContract(**values)
        return
    with pytest.raises(BeartypeCallHintParamViolation, match="batch_size"):
        BatchToleranceProfile(
            name="profile",
            batch_size=cast(Any, value),
            selected_sae=_tolerance(),
            raw_residual=_tolerance(),
            calibration_contract=_contract(),
        )


@PROPERTY_SETTINGS
@given(
    count=_VALID_POSITIVE_INT,
    support=st.integers(min_value=0, max_value=10**9),
    max_abs=_VALID_FINITE_NON_NEGATIVE,
)
def test_difference_metrics_separate_scalar_domains_from_count_relationships(
    count: int,
    support: int,
    max_abs: int | float,
) -> None:
    if support > count:
        with pytest.raises(AutoInterpBatchValidationError, match="between zero and count"):
            BatchDifferenceMetrics(
                count=count,
                support_mismatches=support,
                max_abs=max_abs,
                p99_abs=0.0,
                mean_abs=0.0,
            )
        return
    metrics = BatchDifferenceMetrics(
        count=count,
        support_mismatches=support,
        max_abs=max_abs,
        p99_abs=0.0,
        mean_abs=0.0,
    )
    assert isinstance(metrics.count, PositiveInt)
    assert isinstance(metrics.support_mismatches, NonNegativeInt)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        (
            "n_sae_features",
            "max_positive_refs_per_feature",
            "max_intruder_refs",
            "chunk_size",
            "scan_workers",
        )
    ),
    value=_INVALID_POSITIVE_INT,
)
def test_source_index_requests_reject_every_invalid_positive_scalar(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _request(Path("property-request"), **{field_name: value})


@PROPERTY_SETTINGS
@given(
    width=st.integers(min_value=1, max_value=128),
    target_ids=st.lists(st.integers(min_value=0, max_value=127), max_size=32),
    positive_cap=_VALID_POSITIVE_INT,
    intruder_cap=_VALID_POSITIVE_INT,
    chunk_size=_VALID_POSITIVE_INT,
    scan_workers=_VALID_POSITIVE_INT,
)
def test_source_index_requests_store_nominal_runtime_domains(
    width: int,
    target_ids: list[int],
    positive_cap: int,
    intruder_cap: int,
    chunk_size: int,
    scan_workers: int,
) -> None:
    in_range = tuple(feature_id % width for feature_id in target_ids)
    request = _request(
        Path("property-request"),
        target_feature_ids=in_range,
        n_sae_features=width,
        max_positive_refs_per_feature=positive_cap,
        max_intruder_refs=intruder_cap,
        chunk_size=chunk_size,
        scan_workers=scan_workers,
    )
    assert isinstance(request.target_feature_ids, NonNegativeIntTuple)
    assert isinstance(request.n_sae_features, PositiveInt)
    assert isinstance(request.scan_workers, PositiveInt)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(("n_sae_features", "max_positive_refs", "chunk_size")),
    value=_INVALID_POSITIVE_INT,
)
def test_source_index_scanners_reject_invalid_domains_before_io(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, Any] = {
        "shard_path": Path("never-opened"),
        "target_feature_ids": (),
        "n_sae_features": 1,
        "max_positive_refs": 1,
        "chunk_size": 1,
    }
    values[field_name] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _scan_source_index_shard_file_topk_vectorized(**values)


@PROPERTY_SETTINGS
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_runtime_domain_checks_do_not_consume_global_rng(seed: int) -> None:
    random.seed(seed)
    before = random.getstate()
    block = build_target_blocks([_target(0)], batch_size=1)[0]
    PreparedBatch(block=block, rows=(_prepared_doc(),), model_width=3)
    BatchToleranceProfile(
        name="profile",
        batch_size=2,
        selected_sae=_tolerance(),
        raw_residual=_tolerance(),
        calibration_contract=_contract(),
    )
    _request(Path("property-request"))
    assert random.getstate() == before


def test_runtime_records_expose_nominal_static_field_types(tmp_path: Path) -> None:
    block = build_target_blocks([_target(0)], batch_size=1)[0]
    doc = _prepared_doc()
    batch = PreparedBatch(block=block, rows=(doc,), model_width=3)
    tolerance = _tolerance()
    contract = _contract()
    profile = BatchToleranceProfile(
        name="profile",
        batch_size=2,
        selected_sae=tolerance,
        raw_residual=tolerance,
        calibration_contract=contract,
    )
    metrics = BatchDifferenceMetrics(
        count=1,
        support_mismatches=0,
        max_abs=0.0,
        p99_abs=0.0,
        mean_abs=0.0,
    )
    request = _request(tmp_path)

    assert_type(block.ordinal, NonNegativeInt)
    assert_type(block.batch_size, PositiveInt)
    assert_type(doc.target_key, NonEmptyStr)
    assert_type(doc.model_token_ids, NonEmptyNonNegativeIntTuple)
    assert_type(doc.activation_slice_start, BinaryInt)
    assert_type(batch.model_width, PositiveInt)
    assert_type(batch.pad_token_id, ZeroInt)
    assert_type(tolerance.max_abs, NonNegativeFiniteNumber)
    assert_type(contract.layer, NonNegativeInt)
    assert_type(profile.batch_size, AtLeastTwoInt)
    assert_type(metrics.count, PositiveInt)
    assert_type(request.target_feature_ids, NonNegativeIntTuple)
    assert_type(request.n_sae_features, PositiveInt)
