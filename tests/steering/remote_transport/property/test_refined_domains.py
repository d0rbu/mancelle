"""Independent property oracles for remote-transport refined domains."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, assert_type, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.refined_types import (
    AtLeastTwoInt,
    LowercaseSha256,
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
)
from steering.remote_body_prefetch import (
    RemoteBodyEntry,
    RemoteBodyFamily,
    calculate_remote_prefetch_disk_admission,
    owner_for_inventory_index,
)
from steering.remote_inline_eval import (
    _strict_nonnegative_int as _inline_strict_nonnegative_int,
)
from steering.remote_inline_eval import _strict_positive_int as _inline_strict_positive_int
from steering.remote_inline_eval import _strict_sha256 as _inline_strict_sha256
from steering.remote_training import (
    HfBucketBackend,
    RemoteBodyPrefetchConfig,
    RemoteCudaStagingConfig,
    RemoteTrainingConfig,
    RemoteTrainingError,
    RemoteTrainPackedConfig,
    _checkpoint_names,
    _strict_nonnegative_int,
    build_remote_loader_certificate,
    build_remote_run_fingerprint,
    build_remote_semantic_fingerprint_v2,
    build_remote_semantic_identity,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=100),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=220),
)
ASCII_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/ \\",
    max_size=220,
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: object) -> bool:
    return _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]


def _positive_int(value: object) -> bool:
    return _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]


def _at_least_two_int(value: object) -> bool:
    return _strict_int(value) and value >= 2  # ty: ignore[unsupported-operator]


def _positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bucket_component(value: str) -> bool:
    ascii_alnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    allowed = ascii_alnum + "._-"
    return (
        1 <= len(value) <= 96
        and value[0] in ascii_alnum
        and value[-1] in ascii_alnum
        and all(character in allowed for character in value)
    )


def _hf_bucket_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(_bucket_component(part) for part in parts)


def _normalized_remote_prefix(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.startswith("/")
        or value.endswith("/")
    ):
        return False
    if value == ".":
        return True
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _remote_config_arguments() -> dict[str, Any]:
    return {
        "bucket_id": "owner/bucket",
        "body_prefetch": RemoteBodyPrefetchConfig(2),
        "selection_prefix": "features/selection",
        "checkpoint_prefix": "checkpoints",
        "selection_wait_timeout_seconds": 1,
        "checkpoint_retry_timeout_seconds": 2.5,
        "retry_interval_seconds": 0.1,
        "resume_latest": False,
    }


def _raw_remote_config() -> dict[str, Any]:
    return {
        "bucket_id": "owner/bucket",
        "body_prefetch": {"lookahead_shards_per_rank": 2},
        "selection_prefix": "features/selection",
        "checkpoint_prefix": "checkpoints",
        "selection_wait_timeout_seconds": 1,
        "checkpoint_retry_timeout_seconds": 2.5,
        "retry_interval_seconds": 0.1,
        "resume_latest": False,
    }


def _contracts() -> dict[str, dict[str, str]]:
    return {
        label: {"selection_checksum": character * 64}
        for label, character in zip(
            ("train", "derived_val", "base_val"),
            "abc",
            strict=True,
        )
    }


def _family(root: Path) -> RemoteBodyFamily:
    return RemoteBodyFamily(
        family_id="1" * 64,
        family_aggregate="2" * 64,
        kind="selection",
        local_dir=str(root / "selection"),
        remote_dir="features/selection",
        entries=(RemoteBodyEntry("shard.zst", 7, "3" * 64),),
        preexisting_descriptors=(),
        schedule_id="4" * 64,
        train_schedule=True,
    )


def test_remote_config_objects_expose_nominal_refined_fields() -> None:
    prefetch = RemoteBodyPrefetchConfig(2)
    staging = RemoteCudaStagingConfig(1024, 2)
    packed = RemoteTrainPackedConfig(cuda_staging=staging)
    remote = RemoteTrainingConfig(**_remote_config_arguments(), train_packed=packed)

    assert_type(prefetch.lookahead_shards_per_rank, PositiveInt)
    assert_type(prefetch.cache_shards, AtLeastTwoInt)
    assert_type(staging.host_queue_bytes, PositiveInt)
    assert_type(staging.pinned_slots, PositiveInt)
    assert_type(remote.bucket_id, HfBucketId)
    assert_type(remote.selection_prefix, NormalizedRemotePrefix)
    assert_type(remote.checkpoint_prefix, NormalizedRemotePrefix)
    assert_type(remote.selection_wait_timeout_seconds, PositiveFiniteNumber)
    assert_type(_strict_nonnegative_int(0, surface="static"), NonNegativeInt)
    assert_type(_inline_strict_sha256("a" * 64, surface="static"), LowercaseSha256)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_hf_backend_bucket_domain_matches_independent_oracle(value: object) -> None:
    if _hf_bucket_id(value):
        backend = HfBucketBackend(cast(Any, value), api=object())
        assert backend.bucket_id == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="bucket_id"):
            HfBucketBackend(cast(Any, value), api=object())


@PROPERTY_SETTINGS
@given(value=ASCII_TEXT)
def test_bucket_id_phantom_matches_independent_oracle(value: str) -> None:
    assert isinstance(value, HfBucketId) is _hf_bucket_id(value)


@PROPERTY_SETTINGS
@given(value=ASCII_TEXT)
def test_remote_prefix_phantom_matches_independent_oracle(value: str) -> None:
    assert isinstance(value, NormalizedRemotePrefix) is _normalized_remote_prefix(value)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_body_prefetch_constructor_matches_positive_int_oracle(value: object) -> None:
    if _positive_int(value):
        assert RemoteBodyPrefetchConfig(cast(Any, value)).lookahead_shards_per_rank == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="lookahead_shards_per_rank"):
            RemoteBodyPrefetchConfig(cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("host_queue_bytes", "pinned_slots")), value=SCALARS)
def test_cuda_staging_constructor_matches_positive_int_oracle(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, Any] = {"host_queue_bytes": 1024, "pinned_slots": 2}
    arguments[field] = value
    if _positive_int(value):
        RemoteCudaStagingConfig(**arguments)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            RemoteCudaStagingConfig(**arguments)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "bucket_id",
            "selection_prefix",
            "checkpoint_prefix",
            "selection_wait_timeout_seconds",
            "checkpoint_retry_timeout_seconds",
            "retry_interval_seconds",
        )
    ),
    value=SCALARS,
)
def test_remote_config_constructor_domains_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    arguments = _remote_config_arguments()
    arguments[field] = value
    if field == "bucket_id":
        valid = _hf_bucket_id(value)
    elif field.endswith("_prefix"):
        valid = _normalized_remote_prefix(value)
    else:
        valid = _positive_finite_number(value)
    if valid:
        RemoteTrainingConfig(**arguments)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            RemoteTrainingConfig(**arguments)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "bucket_id",
            "selection_prefix",
            "checkpoint_prefix",
            "selection_wait_timeout_seconds",
            "checkpoint_retry_timeout_seconds",
            "retry_interval_seconds",
        )
    ),
    value=SCALARS,
)
def test_raw_remote_config_preserves_boundary_error_translation(
    field: str,
    value: object,
) -> None:
    raw = _raw_remote_config()
    raw[field] = value
    if field == "bucket_id":
        valid = _hf_bucket_id(value)
    elif field.endswith("_prefix"):
        valid = _normalized_remote_prefix(value)
    else:
        valid = _positive_finite_number(value)
    if valid:
        assert RemoteTrainingConfig.parse(raw) is not None
    else:
        with pytest.raises(RemoteTrainingError):
            RemoteTrainingConfig.parse(raw)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "selection_checksum",
            "selection_inventory_aggregate",
            "packed_row_fingerprint",
        )
    ),
    value=SCALARS,
)
def test_loader_digest_domains_match_independent_oracle(field: str, value: object) -> None:
    arguments: dict[str, Any] = {
        "selection_checksum": "1" * 64,
        "selection_inventory_aggregate": "2" * 64,
        "packed_row_fingerprint": "3" * 64,
        "max_features_per_segment": 0,
    }
    arguments[field] = value
    if _lowercase_sha256(value):
        build_remote_loader_certificate(**arguments)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            build_remote_loader_certificate(**arguments)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_loader_feature_limit_matches_non_negative_int_oracle(value: object) -> None:
    arguments: dict[str, Any] = {
        "selection_checksum": "1" * 64,
        "selection_inventory_aggregate": "2" * 64,
        "packed_row_fingerprint": "3" * 64,
        "max_features_per_segment": value,
    }
    if _non_negative_int(value):
        assert build_remote_loader_certificate(**arguments)["max_features_per_segment"] == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="max_features_per_segment"):
            build_remote_loader_certificate(**arguments)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("step", "world_size")), value=SCALARS)
def test_checkpoint_name_domains_match_independent_oracles(field: str, value: object) -> None:
    arguments: dict[str, Any] = {"step": 0, "world_size": 1}
    arguments[field] = value
    valid = _non_negative_int(value) if field == "step" else _positive_int(value)
    if valid:
        assert _checkpoint_names(**arguments)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _checkpoint_names(**arguments)


@PROPERTY_SETTINGS
@given(builder=st.sampled_from(("legacy", "semantic")), value=SCALARS)
def test_fingerprint_world_size_matches_positive_int_oracle(
    builder: str,
    value: object,
) -> None:
    function = cast(
        Any,
        build_remote_run_fingerprint
        if builder == "legacy"
        else build_remote_semantic_fingerprint_v2,
    )
    arguments = {
        "resolved_training_config": {},
        "world_size": value,
        "selection_contracts": _contracts(),
    }
    if _positive_int(value):
        assert _lowercase_sha256(function(**arguments))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="world_size"):
            function(**arguments)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_semantic_identity_digest_matches_independent_oracle(value: object) -> None:
    if _lowercase_sha256(value):
        assert build_remote_semantic_identity(cast(Any, value))["fingerprint"] == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="fingerprint"):
            build_remote_semantic_identity(cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("inventory_index", "world_size")), value=SCALARS)
def test_owner_domains_match_independent_oracles(field: str, value: object) -> None:
    arguments: dict[str, Any] = {"inventory_index": 0, "world_size": 1}
    arguments[field] = value
    valid = _non_negative_int(value) if field == "inventory_index" else _positive_int(value)
    if valid:
        assert (
            owner_for_inventory_index(**arguments)
            == arguments["inventory_index"] % arguments["world_size"]
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            owner_for_inventory_index(**arguments)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_family_entry_separates_scalar_domain_from_upper_bound(
    value: object,
) -> None:
    family = _family(Path("remote-property-cache"))
    if not _non_negative_int(value):
        with pytest.raises(BeartypeCallHintParamViolation, match="inventory_index"):
            family.entry(cast(Any, value))
    elif value == 0:
        assert family.entry(cast(int, value)) is family.entries[0]
    else:
        with pytest.raises(RemoteTrainingError, match="inventory index"):
            family.entry(cast(int, value))


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "world_size",
            "cache_shards",
            "lookahead_shards_per_rank",
            "safety_reserve_bytes",
        )
    ),
    value=SCALARS,
)
def test_disk_admission_scalar_domains_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    cache_root = Path(".")
    arguments: dict[str, Any] = {
        "cache_root": cache_root,
        "families": (_family(Path("remote-property-cache")),),
        "world_size": 1,
        "cache_shards": 1,
        "lookahead_shards_per_rank": 1,
        "safety_reserve_bytes": 0,
    }
    arguments[field] = value
    valid = _non_negative_int(value) if field == "safety_reserve_bytes" else _positive_int(value)
    if valid:
        assert calculate_remote_prefetch_disk_admission(**arguments).safety_reserve_bytes == (
            value if field == "safety_reserve_bytes" else 0
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            calculate_remote_prefetch_disk_admission(**arguments)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_raw_nonnegative_helpers_share_one_oracle_and_error_boundary(value: object) -> None:
    helpers = (_strict_nonnegative_int, _inline_strict_nonnegative_int)
    for helper in helpers:
        if _non_negative_int(value):
            assert helper(value, surface="property") == value
        else:
            with pytest.raises(RemoteTrainingError, match="property"):
                helper(value, surface="property")


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_inline_positive_helper_matches_independent_oracle(value: object) -> None:
    if _positive_int(value):
        assert _inline_strict_positive_int(value, surface="property") == value
    else:
        with pytest.raises(RemoteTrainingError, match="property"):
            _inline_strict_positive_int(value, surface="property")


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_inline_digest_helper_matches_independent_oracle(value: object) -> None:
    if _lowercase_sha256(value):
        assert _inline_strict_sha256(value, surface="property") == value
    else:
        with pytest.raises(RemoteTrainingError, match="property"):
            _inline_strict_sha256(value, surface="property")


@PROPERTY_SETTINGS
@given(seed=st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_remote_type_checks_do_not_perturb_python_global_rng(seed: int) -> None:
    random.seed(seed)
    before = random.getstate()
    remote = RemoteTrainingConfig(**_remote_config_arguments())
    build_remote_loader_certificate(
        selection_checksum="1" * 64,
        selection_inventory_aggregate="2" * 64,
        packed_row_fingerprint="3" * 64,
        max_features_per_segment=0,
    )
    build_remote_run_fingerprint(
        resolved_training_config={},
        world_size=1,
        selection_contracts=_contracts(),
    )
    build_remote_semantic_fingerprint_v2(
        resolved_training_config={},
        world_size=1,
        selection_contracts=_contracts(),
    )
    build_remote_semantic_identity("a" * 64)
    owner_for_inventory_index(0, 1)
    _family(Path("cache")).entry(0)
    assert remote.body_prefetch.lookahead_shards_per_rank == 2
    assert random.getstate() == before
