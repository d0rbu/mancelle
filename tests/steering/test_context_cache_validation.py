from __future__ import annotations

import pytest
import torch as t

import steering.eval.context_cache_validation as validation
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.context_cache_validation import (
    compare_context_record_sets,
    infer_final_segment_start,
    validate_context_cache_records,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.feature_config import FeatureConfig
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
    MockTokenizer,
)


def _record(tokens=(0, 99, 100, 101, 102), metadata=None):
    return EvalContextRecord(
        suite="core",
        task="toy",
        seed=1,
        example_index=2,
        row_index=3,
        token_ids=tuple(tokens),
        context_text="abcd",
        metadata=metadata if metadata is not None else {"target_start": 3},
    )


def _segments():
    return (
        ContextFeatureSegment((1,), (1,), 0, 2),
        ContextFeatureSegment((2,), (2,), 2, 4),
    )


def _provider(
    tmp_path,
    entries,
    *,
    n_sae_features=TEST_CACHE_N_SAE_FEATURES,
    num_buckets=TEST_CACHE_NUM_BUCKETS,
):
    save_context_cache(
        tmp_path,
        entries,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        identity=TEST_CACHE_IDENTITY,
    )
    return SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_validate_context_cache_records_checks_routing_and_features(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
    )

    summary = validate_context_cache_records(
        records=[record, record],
        provider=provider,
        tokenizer=tokenizer,
        target_feature_configs=[FeatureConfig(name="target", features={3: 1})],
    )

    assert summary.expected_contexts == 2
    assert summary.unique_contexts == 1
    assert summary.cached_contexts == 1
    assert summary.total_context_segments == 2
    assert summary.total_context_features == 2
    assert summary.target_feature_configs == 1
    check = summary.routing_checks[0]
    assert check.context_assigned_token_count == 3
    assert check.final_assigned_token_count == 2
    assert check.final_start == 3
    assert check.target_conditioning_mode == "current"
    assert check.encoder_slot_count == 3
    assert check.target_slot_index == 2
    assert check.shifted_context_slot_count == 0
    # kills: emitting an empty turn_target_checks key (fixed summary bytes must not change).
    assert "turn_target_checks" not in summary.to_dict()
    routing_dict = summary.to_dict()["routing_checks"][0]
    assert routing_dict["key"] == record.key
    assert routing_dict["target_conditioning_mode"] == "current"
    assert routing_dict["encoder_slot_count"] == 3
    assert routing_dict["target_slot_index"] == 2


def test_validate_context_cache_records_supports_next_segment_routing(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
    )

    summary = validate_context_cache_records(
        records=[record],
        provider=provider,
        tokenizer=tokenizer,
        target_feature_configs=[FeatureConfig(name="nextseg", features={3: 1})],
        target_conditioning_mode="next",
    )

    check = summary.routing_checks[0]
    assert check.context_segment_count == 2
    assert check.context_assigned_token_count == 3
    assert check.final_assigned_token_count == 2
    assert check.target_conditioning_mode == "next"
    assert check.encoder_slot_count == 2
    assert check.target_slot_index == 1
    assert check.shifted_context_slot_count == 1


def test_validate_context_cache_records_rejects_bad_target_conditioning_mode(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
    )

    with pytest.raises(ContextCacheValidationError, match="target_conditioning_mode"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
            target_conditioning_mode="previous",
        )


def test_validator_routing_matches_target_aware_wrapper_assignment():
    tokenizer = MockTokenizer()
    tokens = list(_record().token_ids)
    segments = _segments()
    wrapper = TargetAwareEncoderDecoderEvalWrapper.__new__(
        TargetAwareEncoderDecoderEvalWrapper
    )
    wrapper.tokenizer = tokenizer
    # The fixed-mode assignment rule is what this validator mirrors.
    wrapper.self_cond_segment_mode = None
    wrapper_seg_map = t.full((len(tokens),), -1, dtype=t.long)

    wrapper._fill_context_assignments(wrapper_seg_map, tokens, segments)

    validator_seg_map = validation._assign_context_tokens_to_segments(
        tokenizer,
        tokens,
        segments,
    )
    assert validator_seg_map.tolist() == wrapper_seg_map.tolist()


def test_validate_context_cache_records_rejects_missing_cache_row(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences([0, 1], _segments())],
    )

    with pytest.raises(ContextCacheValidationError, match="missing"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
        )


def test_validate_context_cache_records_rejects_empty_context_features(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [
            ContextCacheEntry.from_sequences(
                record.token_ids,
                [ContextFeatureSegment((), (), 0, 4)],
            )
        ],
    )

    with pytest.raises(ContextCacheValidationError, match="no real context features"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
        )


def test_validate_context_cache_records_rejects_empty_segments_when_strict(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [
            ContextCacheEntry.from_sequences(
                record.token_ids,
                [
                    ContextFeatureSegment((1,), (1,), 0, 2),
                    ContextFeatureSegment((), (), 2, 4),
                ],
            )
        ],
    )

    with pytest.raises(ContextCacheValidationError, match="context segment"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
            require_nonempty_context_segments=True,
        )


def test_validate_context_cache_records_rejects_misaligned_char_bounds(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [
            ContextCacheEntry.from_sequences(
                record.token_ids,
                [ContextFeatureSegment((1,), (1,), 2, 4)],
            )
        ],
    )

    with pytest.raises(ContextCacheValidationError, match="character alignment failed"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
            require_nonempty_context_features=True,
        )


def test_validate_context_cache_records_rejects_empty_cache_rows(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, [])],
    )

    with pytest.raises(ContextCacheValidationError, match="no real context features"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
        )


def test_validate_context_cache_records_rejects_bad_target_feature_config(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
    )

    with pytest.raises(ValueError, match="feature id"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
            target_feature_configs=[FeatureConfig("bad", {99: 1})],
        )


def test_validate_context_cache_records_rejects_extra_cache_rows_when_strict(tmp_path):
    tokenizer = MockTokenizer()
    record = _record()
    provider = _provider(
        tmp_path,
        [
            ContextCacheEntry.from_sequences(record.token_ids, _segments()),
            ContextCacheEntry.from_sequences([0, 1, 2], _segments()),
        ],
    )

    with pytest.raises(ContextCacheValidationError, match="extra context"):
        validate_context_cache_records(
            records=[record],
            provider=provider,
            tokenizer=tokenizer,
            fail_on_extra_cached_contexts=True,
        )


def test_validate_context_cache_records_rejects_empty_record_list(tmp_path):
    tokenizer = MockTokenizer()
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences([0, 1, 2], _segments())],
    )

    with pytest.raises(ContextCacheValidationError, match="No eval context"):
        validate_context_cache_records(
            records=[],
            provider=provider,
            tokenizer=tokenizer,
        )


def test_infer_final_segment_start_rejects_invalid_metadata():
    assert infer_final_segment_start(_record(metadata={"answer_position": 2})) == 2
    assert infer_final_segment_start(_record(metadata={})) == 4

    with pytest.raises(ContextCacheValidationError, match="zero tokens"):
        infer_final_segment_start(_record(tokens=(), metadata={}))
    with pytest.raises(ContextCacheValidationError, match="outside token range"):
        infer_final_segment_start(_record(metadata={"target_start": 99}))


def test_compare_context_record_sets_rejects_missing_and_extra_records():
    expected = [_record(tokens=(0, 1)), _record(tokens=(0, 2))]
    observed = [_record(tokens=(0, 1)), _record(tokens=(0, 3))]

    with pytest.raises(ContextCacheValidationError, match="missing"):
        compare_context_record_sets(expected, observed)
