"""Versioned provenance for Stage 1b positive-prefix selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    derive_positive_qc_thresholds,
)
from steering.eval.axbench_base.percentile_units import (
    PERCENTILE_POINTS_PER_FRACTION,
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.prefix_text import validate_prefix_char_bounds
from steering.eval.axbench_base.types import AxBenchBaseError

POSITIVE_SELECTION_CONTRACT_METADATA_KEY = (
    "axbench_stage1b_positive_selection_contract"
)
POSITIVE_INITIAL_COMPAT_TARGET = 72
POSITIVE_SELECTION_CONTRACT_VERSION_V3 = "positive_target_refill_v3"
POSITIVE_SELECTION_CONTRACT_VERSION_V4 = (
    "positive_target_compatibility_v4"
)
POSITIVE_SELECTION_CONTRACT_VERSION_V5 = (
    "positive_target_refill_post_provider_qc_v5"
)
POSITIVE_SELECTION_CONTRACT_VERSION_V6 = (
    "positive_target_compatibility_post_provider_qc_v6"
)
# Backward-compatible import for callers that specifically name the current
# target-72 contract. Non-72 producers select v6 dynamically.
POSITIVE_SELECTION_CONTRACT_VERSION = POSITIVE_SELECTION_CONTRACT_VERSION_V5
POSITIVE_CHECKPOINT_SELECTION_VERSION_V1 = (
    "positive_target_compatibility_stream_v1"
)
POSITIVE_CHECKPOINT_SELECTION_VERSION = (
    "positive_target_compatibility_stream_post_provider_qc_v2"
)
POSITIVE_POST_COMPATIBILITY_BATCHING = (
    "full_prefix_score_batch_chunks_except_pool_exhaustion_v1"
)
POSITIVE_POST_PROVIDER_QC_BATCHING = "stable_chain_frontier_ragged_v1"
_STRING_FIELDS = {
    "percentile_currency": "raw_unquantized_percentile_points_v1",
    "prefix_window": "recent_character_suffix_v1",
    "prefix_baseline": (
        "ragged_right_padded_batch_prefix_alone_single_bos_excluded_v1"
    ),
    "continuation_span": (
        "joint_tokenization_completion_single_bos_excluded_v1"
    ),
    "continuation_copy_guard": (
        "authenticated_stage1_continuation_and_copy_guards_v2"
    ),
    "candidate_ordering": (
        "seed_index_full_shuffle_clean_then_dirty_percentile_stable_v1"
    ),
    "replacement_semantics": (
        "lazy_atomic_complete_pair_target_refill_v1"
    ),
}
_SHARED_CONTRACT_KEYS = frozenset(
    {
        "accept_fraction",
        "required_delta",
        "headroom_ceiling_fraction",
        "completion_ceiling",
        "maximum_seed_percentile",
        "min_prefix_chars",
        "max_prefix_chars",
        "prefix_score_batch_size",
        "positive_max_attempts_per_seed",
        "positive_zero_hit_bail_after",
        *_STRING_FIELDS,
    }
)
_V3_CONTRACT_KEYS = _SHARED_CONTRACT_KEYS | {
    "contract_version",
    "n_positives_per_concept",
}
_V4_CONTRACT_KEYS = _V3_CONTRACT_KEYS | {
    "initial_compatibility_target",
    "post_compatibility_batching",
}
_POST_PROVIDER_QC_KEYS = {
    "post_provider_qc_batch_size",
    "post_provider_qc_batching",
}
_V5_CONTRACT_KEYS = _V3_CONTRACT_KEYS | _POST_PROVIDER_QC_KEYS
_V6_CONTRACT_KEYS = _V4_CONTRACT_KEYS | _POST_PROVIDER_QC_KEYS


def effective_post_provider_qc_batch_size(cfg: AxBenchBaseConfig) -> int:
    """Return the deterministic maximum Stage 1b post-provider QC shape."""

    return min(cfg.qc_batch_size, max(1, cfg.synth.max_concurrent))


def _post_provider_qc_fields(cfg: AxBenchBaseConfig) -> dict[str, Any]:
    return {
        "post_provider_qc_batch_size": (
            effective_post_provider_qc_batch_size(cfg)
        ),
        "post_provider_qc_batching": POSITIVE_POST_PROVIDER_QC_BATCHING,
    }


def _shared_selection_fields(cfg: AxBenchBaseConfig) -> dict[str, Any]:
    """Return target-independent Stage 1b selection semantics."""

    thresholds = derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    )
    return {
        "percentile_currency": _STRING_FIELDS["percentile_currency"],
        "accept_fraction": thresholds.accept_fraction,
        "required_delta": thresholds.required_delta,
        "headroom_ceiling_fraction": thresholds.headroom_ceiling_fraction,
        "completion_ceiling": thresholds.completion_ceiling,
        "maximum_seed_percentile": thresholds.maximum_seed_percentile,
        "min_prefix_chars": cfg.min_prefix_chars,
        "max_prefix_chars": cfg.max_prefix_chars,
        "prefix_window": _STRING_FIELDS["prefix_window"],
        "prefix_baseline": _STRING_FIELDS["prefix_baseline"],
        "prefix_score_batch_size": cfg.qc_batch_size,
        "continuation_span": _STRING_FIELDS["continuation_span"],
        "continuation_copy_guard": _STRING_FIELDS[
            "continuation_copy_guard"
        ],
        "positive_max_attempts_per_seed": (
            cfg.positive_max_attempts_per_seed
        ),
        "positive_zero_hit_bail_after": cfg.positive_zero_hit_bail_after,
        "candidate_ordering": _STRING_FIELDS["candidate_ordering"],
        "replacement_semantics": _STRING_FIELDS["replacement_semantics"],
    }


def _artifact_selection_contract_payload(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any]:
    """Return the target-bearing provenance written into final parquets.

    The final artifact still records the requested target and the effective
    post-provider QC shape. The target is stopping-condition provenance and is
    excluded from resumable checkpoint identity. The post-provider shape
    remains part of that identity because BF16 scoring is not proven invariant
    to ragged batch shape.
    """

    payload = {
        "contract_version": (
            POSITIVE_SELECTION_CONTRACT_VERSION_V5
            if cfg.n_positives_per_concept
            == POSITIVE_INITIAL_COMPAT_TARGET
            else POSITIVE_SELECTION_CONTRACT_VERSION_V6
        ),
        **_shared_selection_fields(cfg),
        **_post_provider_qc_fields(cfg),
        "n_positives_per_concept": cfg.n_positives_per_concept,
    }
    if cfg.n_positives_per_concept != POSITIVE_INITIAL_COMPAT_TARGET:
        payload.update(
            {
                "initial_compatibility_target": (
                    POSITIVE_INITIAL_COMPAT_TARGET
                ),
                "post_compatibility_batching": (
                    POSITIVE_POST_COMPATIBILITY_BATCHING
                ),
            }
        )
    return payload


def positive_artifact_selection_contract_payload(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any]:
    """Return the target-bearing provenance written into final parquets."""

    return _artifact_selection_contract_payload(cfg)


def _target_independent_selection_payload(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any]:
    """Return the target-independent identity shared by resumable checkpoints.

    ``n_positives_per_concept`` is a stopping condition. Post-provider batch
    shape remains semantic until the production scorer has a proven
    shape-invariant numerical contract.
    """

    return {
        "contract_version": POSITIVE_CHECKPOINT_SELECTION_VERSION,
        **_shared_selection_fields(cfg),
        **_post_provider_qc_fields(cfg),
        "initial_compatibility_target": POSITIVE_INITIAL_COMPAT_TARGET,
        "post_compatibility_batching": (
            POSITIVE_POST_COMPATIBILITY_BATCHING
        ),
    }


def positive_selection_contract_payload(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any]:
    """Return target-independent resumable selection identity."""

    return _target_independent_selection_payload(cfg)


def positive_checkpoint_selection_payload(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any]:
    """Backward-compatible name for resumable selection identity.

    The effective post-provider batch shape remains explicit because the BF16
    production scorer is not proven bit-for-bit invariant across ragged batch
    shapes.
    """

    return positive_selection_contract_payload(cfg)


def _contract_json(cfg: AxBenchBaseConfig) -> str:
    return json.dumps(
        _artifact_selection_contract_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def schema_with_positive_selection_contract(
    schema: pa.Schema,
    cfg: AxBenchBaseConfig,
) -> pa.Schema:
    """Tag a positive synth schema with its exact prefix-selection contract."""

    metadata = dict(schema.metadata or {})
    metadata[POSITIVE_SELECTION_CONTRACT_METADATA_KEY.encode("utf-8")] = (
        _contract_json(cfg).encode("utf-8")
    )
    return schema.with_metadata(metadata)


def schema_with_default_positive_selection_contract(
    schema: pa.Schema,
    *,
    accept_threshold: object,
) -> pa.Schema:
    """Stamp default selection semantics for synthetic/test positive corpora.

    Production Stage 1b immediately replaces this record with its complete
    runtime config. Other positive-corpus builders in fixtures use the shipped
    selection defaults and need a complete, consumer-valid provenance record.
    """

    accept_points = percentile_fraction_to_points(
        accept_threshold,
        field_name="accept_threshold",
    )
    cfg = AxBenchBaseConfig(
        feature_dir="",
        feature_statistics_path="",
        decoder_only={},
        encdec={},
        device="cpu",
        output_root=None,
        accept_threshold=(
            accept_points / PERCENTILE_POINTS_PER_FRACTION
        ),
    )
    return schema_with_positive_selection_contract(schema, cfg)


def positive_selection_contract_from_schema(
    schema: pa.Schema,
) -> dict[str, Any] | None:
    """Decode selection metadata, rejecting malformed or ambiguous JSON."""

    metadata = schema.metadata or {}
    raw = metadata.get(
        POSITIVE_SELECTION_CONTRACT_METADATA_KEY.encode("utf-8")
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
            f"invalid {POSITIVE_SELECTION_CONTRACT_METADATA_KEY!r} "
            "parquet metadata"
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBaseError(
            f"invalid {POSITIVE_SELECTION_CONTRACT_METADATA_KEY!r} parquet "
            "metadata: expected a JSON object"
        )
    return payload


def _validate_payload(
    payload: dict[str, Any],
    *,
    consumer: str,
    path: Path,
) -> None:
    keys = set(payload)
    # Check a recognizable older version before exact-key validation so a
    # genuine v2 corpus (which necessarily lacks the v3 refill fields) receives
    # the established migration guidance instead of a generic shape error.
    version = payload.get("contract_version")
    if version not in {
        POSITIVE_SELECTION_CONTRACT_VERSION_V3,
        POSITIVE_SELECTION_CONTRACT_VERSION_V4,
        POSITIVE_SELECTION_CONTRACT_VERSION_V5,
        POSITIVE_SELECTION_CONTRACT_VERSION_V6,
    }:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has unsupported "
            f"selection contract version {version!r}; expected "
            f"{POSITIVE_SELECTION_CONTRACT_VERSION_V3!r}, "
            f"{POSITIVE_SELECTION_CONTRACT_VERSION_V4!r}, "
            f"{POSITIVE_SELECTION_CONTRACT_VERSION_V5!r}, or "
            f"{POSITIVE_SELECTION_CONTRACT_VERSION_V6!r}. Regenerate "
            "Stage 1b and its downstream artifacts under a fresh run_tag."
        )
    expected_keys = {
        POSITIVE_SELECTION_CONTRACT_VERSION_V3: _V3_CONTRACT_KEYS,
        POSITIVE_SELECTION_CONTRACT_VERSION_V4: _V4_CONTRACT_KEYS,
        POSITIVE_SELECTION_CONTRACT_VERSION_V5: _V5_CONTRACT_KEYS,
        POSITIVE_SELECTION_CONTRACT_VERSION_V6: _V6_CONTRACT_KEYS,
    }[version]
    if keys != expected_keys:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has selection "
            "contract fields mismatch: "
            f"missing={sorted(expected_keys - keys)}, "
            f"unexpected={sorted(keys - expected_keys)}"
        )
    for field, expected in _STRING_FIELDS.items():
        if payload[field] != expected:
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has selection "
                f"contract {field}={payload[field]!r}; expected {expected!r}"
            )
    if version in {
        POSITIVE_SELECTION_CONTRACT_VERSION_V5,
        POSITIVE_SELECTION_CONTRACT_VERSION_V6,
    } and payload["post_provider_qc_batching"] != (
        POSITIVE_POST_PROVIDER_QC_BATCHING
    ):
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has selection "
            "contract post_provider_qc_batching="
            f"{payload['post_provider_qc_batching']!r}; expected "
            f"{POSITIVE_POST_PROVIDER_QC_BATCHING!r}"
        )
    target = payload["n_positives_per_concept"]
    if version in {
        POSITIVE_SELECTION_CONTRACT_VERSION_V3,
        POSITIVE_SELECTION_CONTRACT_VERSION_V5,
    } and (
        type(target) is not int or target != POSITIVE_INITIAL_COMPAT_TARGET
    ):
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has {version!r} "
            f"selection target {target!r}; this version is valid only at "
            f"{POSITIVE_INITIAL_COMPAT_TARGET}"
        )
    if version in {
        POSITIVE_SELECTION_CONTRACT_VERSION_V4,
        POSITIVE_SELECTION_CONTRACT_VERSION_V6,
    }:
        if type(target) is not int or (
            target == POSITIVE_INITIAL_COMPAT_TARGET
        ):
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has {version!r} "
                f"selection target {target!r}; this version is invalid at the "
                f"compatibility target {POSITIVE_INITIAL_COMPAT_TARGET}"
            )
        compatibility_target = payload["initial_compatibility_target"]
        if type(compatibility_target) is not int or (
            compatibility_target != POSITIVE_INITIAL_COMPAT_TARGET
        ):
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has "
                "inconsistent initial_compatibility_target="
                f"{compatibility_target!r}; expected "
                f"{POSITIVE_INITIAL_COMPAT_TARGET}"
            )
        batching = payload["post_compatibility_batching"]
        if batching != POSITIVE_POST_COMPATIBILITY_BATCHING:
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has "
                f"post_compatibility_batching={batching!r}; expected "
                f"{POSITIVE_POST_COMPATIBILITY_BATCHING!r}"
            )
    numeric: dict[str, float] = {}
    for field in (
        "accept_fraction",
        "required_delta",
        "headroom_ceiling_fraction",
        "completion_ceiling",
        "maximum_seed_percentile",
    ):
        value = payload[field]
        if type(value) is not float:
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has non-float "
                f"selection contract {field}={value!r}"
            )
        numeric[field] = value
        if not math.isfinite(numeric[field]):
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has non-finite "
                f"selection contract {field}={value!r}"
            )
    thresholds = derive_positive_qc_thresholds(
        numeric["accept_fraction"],
        numeric["headroom_ceiling_fraction"],
    )
    expected_numeric = {
        "required_delta": thresholds.required_delta,
        "completion_ceiling": thresholds.completion_ceiling,
        "maximum_seed_percentile": thresholds.maximum_seed_percentile,
    }
    for field, expected in expected_numeric.items():
        if numeric[field] != expected:
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has inconsistent "
                f"selection contract {field}={numeric[field]!r}; expected "
                f"{expected!r} from its threshold fractions"
            )
    for field in (
        "min_prefix_chars",
        "max_prefix_chars",
        "prefix_score_batch_size",
        "n_positives_per_concept",
        "positive_max_attempts_per_seed",
        *(
            ("post_provider_qc_batch_size",)
            if version
            in {
                POSITIVE_SELECTION_CONTRACT_VERSION_V5,
                POSITIVE_SELECTION_CONTRACT_VERSION_V6,
            }
            else ()
        ),
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has non-integer "
                f"selection contract {field}={value!r}"
            )
    if payload["min_prefix_chars"] < 0:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has negative "
            "selection contract min_prefix_chars="
            f"{payload['min_prefix_chars']!r}"
        )
    if payload["max_prefix_chars"] < 1:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has non-positive "
            "selection contract max_prefix_chars="
            f"{payload['max_prefix_chars']!r}"
        )
    validate_prefix_char_bounds(
        payload["min_prefix_chars"],
        payload["max_prefix_chars"],
    )
    if payload["prefix_score_batch_size"] < 1:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has non-positive "
            "selection contract prefix_score_batch_size="
            f"{payload['prefix_score_batch_size']!r}"
        )
    if version in {
        POSITIVE_SELECTION_CONTRACT_VERSION_V5,
        POSITIVE_SELECTION_CONTRACT_VERSION_V6,
    }:
        post_provider_batch_size = payload["post_provider_qc_batch_size"]
        if post_provider_batch_size < 1 or (
            post_provider_batch_size > payload["prefix_score_batch_size"]
        ):
            raise AxBenchBaseError(
                f"{consumer}: positive synth corpus at {path} has invalid "
                "selection contract post_provider_qc_batch_size="
                f"{post_provider_batch_size!r}; expected 1 <= value <= "
                f"prefix_score_batch_size={payload['prefix_score_batch_size']!r}"
            )
    if payload["n_positives_per_concept"] < 1:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has non-positive "
            "selection contract n_positives_per_concept="
            f"{payload['n_positives_per_concept']!r}"
        )
    if payload["positive_max_attempts_per_seed"] < 1:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has non-positive "
            "selection contract positive_max_attempts_per_seed="
            f"{payload['positive_max_attempts_per_seed']!r}"
        )
    bail_after = payload["positive_zero_hit_bail_after"]
    if bail_after is not None and (
        isinstance(bail_after, bool)
        or not isinstance(bail_after, int)
        or bail_after < 1
    ):
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has invalid "
            "selection contract positive_zero_hit_bail_after="
            f"{bail_after!r}"
        )


def validate_positive_selection_contract(
    path: Path,
    *,
    consumer: str,
    schema: pa.Schema | None = None,
    expected_cfg: AxBenchBaseConfig | None = None,
) -> dict[str, Any]:
    """Fail loudly unless a positive corpus has valid selection provenance."""

    parquet_schema = schema if schema is not None else pq.read_schema(path)
    actual = positive_selection_contract_from_schema(parquet_schema)
    if actual is None:
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} is missing "
            f"{POSITIVE_SELECTION_CONTRACT_METADATA_KEY!r} metadata. "
            "Regenerate Stage 1b under a fresh run_tag with the current code."
        )
    _validate_payload(actual, consumer=consumer, path=path)
    if expected_cfg is None:
        return actual

    expected = _artifact_selection_contract_payload(expected_cfg)
    expected_json = _contract_json(expected_cfg)
    actual_json = json.dumps(
        actual,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if actual_json != expected_json:
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
        raise AxBenchBaseError(
            f"{consumer}: positive synth corpus at {path} has a selection "
            f"contract mismatch: {mismatches}. Regenerate Stage 1b with the "
            "current config."
        )
    return actual


__all__ = [
    "POSITIVE_CHECKPOINT_SELECTION_VERSION",
    "POSITIVE_CHECKPOINT_SELECTION_VERSION_V1",
    "POSITIVE_INITIAL_COMPAT_TARGET",
    "POSITIVE_POST_COMPATIBILITY_BATCHING",
    "POSITIVE_POST_PROVIDER_QC_BATCHING",
    "POSITIVE_SELECTION_CONTRACT_METADATA_KEY",
    "POSITIVE_SELECTION_CONTRACT_VERSION",
    "POSITIVE_SELECTION_CONTRACT_VERSION_V3",
    "POSITIVE_SELECTION_CONTRACT_VERSION_V4",
    "POSITIVE_SELECTION_CONTRACT_VERSION_V5",
    "POSITIVE_SELECTION_CONTRACT_VERSION_V6",
    "effective_post_provider_qc_batch_size",
    "positive_artifact_selection_contract_payload",
    "positive_checkpoint_selection_payload",
    "positive_selection_contract_from_schema",
    "positive_selection_contract_payload",
    "schema_with_positive_selection_contract",
    "schema_with_default_positive_selection_contract",
    "validate_positive_selection_contract",
]
