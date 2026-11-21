"""Parity and override tests for the shared target-real artifact builder."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.context_cache_io import ContextCacheEntry, ContextFeatureSegment
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import PooledTargetRealConfig, SingleTargetSegment
from steering.eval.target_real_dispatch import build_target_for_real_config
from steering.eval.target_real_runner import (
    build_target_real_artifacts_for_example,
    context_tokens_from_target_mask,
)
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec


class _RowsProvider:
    """In-memory provider with the same exact-key contract as production."""

    def __init__(self, rows: dict[tuple[int, ...], tuple[ContextFeatureSegment, ...]]):
        self.rows = rows
        self.manifest: dict[str, object] = {"num_buckets": 4}

    def get(self, token_ids):
        return self.rows.get(tuple(int(token) for token in token_ids))


def _fixture_payload() -> tuple[
    MockTokenizer,
    list[int],
    list[int],
    list[int],
    _RowsProvider,
    PooledTargetRealConfig,
]:
    tokenizer = MockTokenizer()
    full_token_ids = cast(
        list[int],
        tokenizer.encode("abcd", prepend=tokenizer.get_bos_token_id()),
    )
    context_token_ids = full_token_ids[:2]
    char_end = int(token_to_char_offsets(tokenizer, full_token_ids)[-1])
    context_char_end = int(token_to_char_offsets(tokenizer, context_token_ids)[-1])
    full_segments = (
        ContextFeatureSegment(
            feature_ids=(3, 4),
            buckets=(1, 3),
            char_start=0,
            char_end=char_end,
        ),
    )
    context_segments = (
        ContextFeatureSegment(
            feature_ids=(9,),
            buckets=(2,),
            char_start=0,
            char_end=context_char_end,
        ),
    )
    provider = _RowsProvider(
        {
            tuple(full_token_ids): full_segments,
            tuple(context_token_ids): context_segments,
        }
    )
    trc = PooledTargetRealConfig(
        name="pooled-test",
        cache_dir=Path("/tmp/target-real-test"),
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    mask = [0, 0, 1, 1, 1]
    return tokenizer, full_token_ids, mask, context_token_ids, provider, trc


def test_default_builder_matches_the_old_explicit_dispatch_contract() -> None:
    tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    model = _make_encdec(device="cpu")

    artifacts = build_target_real_artifacts_for_example(
        model=model,
        tokenizer=tokenizer,
        trc=trc,
        provider=provider,
        full_token_ids=full_ids,
        mask=mask,
        context_token_ids=context_ids,
    )

    full_entry = provider.get(full_ids)
    assert full_entry is not None
    expected_target = build_target_for_real_config(
        cfg=trc,
        entry=ContextCacheEntry.from_sequences(full_ids, full_entry),
        target_char_start=artifacts.target_char_start,
        target_char_end=artifacts.target_char_end,
        full_token_ids=full_ids,
        target_token_loss_mask=mask,
        tokenizer=tokenizer,
    )
    assert artifacts.target == expected_target
    assert artifacts.context_token_ids == context_ids
    assert artifacts.target_char_start < artifacts.target_char_end
    assert isinstance(artifacts.wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert isinstance(artifacts.target, SingleTargetSegment)

    # Rebuild the old explicit wrapper construction and compare the
    # externally observable conditioning plan, not just the target dataclass.
    legacy_context_provider = artifacts.wrapper.context_provider
    legacy_wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        target=expected_target,
        context_provider=legacy_context_provider,
        target_mode="current",
    )
    assert artifacts.wrapper._target_segments == legacy_wrapper._target_segments
    assert artifacts.wrapper.target_conditioning_mode == legacy_wrapper.target_conditioning_mode
    assert legacy_context_provider.get(context_ids) == provider.get(context_ids)


def test_builder_explicit_range_and_loss_mask_are_forwarded_without_changing_context() -> None:
    tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    model = _make_encdec(device="cpu")
    explicit_range = (1, 4)
    explicit_loss_mask = [1, 0, 1, 0, 1]

    artifacts = build_target_real_artifacts_for_example(
        model=model,
        tokenizer=tokenizer,
        trc=trc,
        provider=provider,
        full_token_ids=full_ids,
        mask=mask,
        target_char_range=explicit_range,
        target_token_loss_mask=explicit_loss_mask,
        context_token_ids=context_ids,
    )

    assert (artifacts.target_char_start, artifacts.target_char_end) == explicit_range
    assert artifacts.context_token_ids == context_ids
    # The pooled target is independent of routing assignments, but its exact
    # feature result proves the explicit target range reached dispatch.
    assert isinstance(artifacts.target, SingleTargetSegment)
    assert artifacts.target.features == {3: 1, 4: 3}

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_char_range": (3, 3)}, "positive range"),
        ({"target_char_range": (-1, 2)}, "positive range"),
        ({"target_char_range": (1, 2, 3)}, "exactly two"),
        ({"target_token_loss_mask": [1, 0]}, "length mismatch"),
    ],
)
def test_builder_rejects_invalid_explicit_overrides(kwargs, match: str) -> None:
    tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    with pytest.raises(ValueError, match=match):
        build_target_real_artifacts_for_example(
            model=_make_encdec(device="cpu"),
            tokenizer=tokenizer,
            trc=trc,
            provider=provider,
            full_token_ids=full_ids,
            mask=mask,
            context_token_ids=context_ids,
            **kwargs,
        )


def test_builder_default_context_prefix_is_derived_from_original_mask() -> None:
    tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    artifacts = build_target_real_artifacts_for_example(
        model=_make_encdec(device="cpu"),
        tokenizer=tokenizer,
        trc=trc,
        provider=provider,
        full_token_ids=full_ids,
        mask=mask,
    )
    assert artifacts.context_token_ids == context_tokens_from_target_mask(full_ids, mask)
    assert artifacts.context_token_ids == context_ids


def test_builder_reports_exact_full_and_context_cache_misses() -> None:
    tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    provider.rows.pop(tuple(full_ids))
    with pytest.raises(KeyError, match="no cache row found"):
        build_target_real_artifacts_for_example(
            model=_make_encdec(device="cpu"),
            tokenizer=tokenizer,
            trc=trc,
            provider=provider,
            full_token_ids=full_ids,
            mask=mask,
            context_token_ids=context_ids,
        )

    _tokenizer, full_ids, mask, context_ids, provider, trc = _fixture_payload()
    provider.rows.pop(tuple(context_ids))
    with pytest.raises(KeyError, match="no prompt/context cache row"):
        build_target_real_artifacts_for_example(
            model=_make_encdec(device="cpu"),
            tokenizer=tokenizer,
            trc=trc,
            provider=provider,
            full_token_ids=full_ids,
            mask=mask,
            context_token_ids=context_ids,
        )
