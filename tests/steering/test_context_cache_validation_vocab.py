"""Vocabulary-aware validation tests for cached evaluation contexts."""

from __future__ import annotations

from pathlib import Path

import pytest

from errors import FeatureVocabError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.context_cache_io import (
    CLUSTER_SELECTION_SEMANTICS,
    CLUSTER_SELECTION_SEMANTICS_KEY,
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.context_cache_validation import validate_context_cache_records
from steering.eval.feature_config import FeatureConfig
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
    MockTokenizer,
)
from tests.steering.test_context_cache_validation import _record, _segments


def _vocab(ids: tuple[int, ...]) -> FeatureVocab:
    mask = FeatureMask(FeatureMaskMode.whitelist, ids)
    vocab = effective_feature_vocab(mask, TEST_CACHE_N_SAE_FEATURES)
    assert vocab is not None
    return vocab


def _provider(
    cache_dir: Path,
    *,
    segments: tuple[ContextFeatureSegment, ...] | None = None,
    expected_feature_vocab: FeatureVocab | None = None,
) -> SafetensorsContextFeatureProvider:
    record = _record()
    metadata = None
    if expected_feature_vocab is not None:
        metadata = {
            FEATURE_VOCAB_META_KEY: expected_feature_vocab.metadata_dict(),
            CLUSTER_SELECTION_SEMANTICS_KEY: CLUSTER_SELECTION_SEMANTICS,
        }
    save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                record.token_ids,
                _segments() if segments is None else segments,
            )
        ],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        identity=TEST_CACHE_IDENTITY,
        metadata=metadata,
    )
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=TEST_CACHE_IDENTITY,
        expected_feature_vocab=expected_feature_vocab,
    )


def _validate(
    provider: SafetensorsContextFeatureProvider,
    *,
    feature_vocab: FeatureVocab | None,
    target_feature_configs: list[FeatureConfig] | None = None,
):
    return validate_context_cache_records(
        records=[_record()],
        provider=provider,
        tokenizer=MockTokenizer(),
        target_feature_configs=target_feature_configs,
        feature_vocab=feature_vocab,
    )


def test_explicit_vocabulary_must_equal_authoritative_provider_vocabulary(
    tmp_path: Path,
) -> None:
    provider_vocab = _vocab((1, 2, 3, 4))
    explicit_vocab = _vocab((1, 2, 3, 5))
    provider = _provider(
        tmp_path,
        expected_feature_vocab=provider_vocab,
    )

    with pytest.raises(FeatureVocabError, match="authoritative"):
        _validate(provider, feature_vocab=explicit_vocab)


def test_explicit_vocabulary_is_used_when_provider_has_none(tmp_path: Path) -> None:
    explicit_vocab = _vocab((1, 2, 3, 4))
    provider = _provider(tmp_path)

    summary = _validate(provider, feature_vocab=explicit_vocab)

    assert summary.total_context_features == sum(
        len(segment.feature_ids) for segment in _segments()
    )


def test_provider_vocabulary_is_used_when_explicit_vocabulary_is_none(
    tmp_path: Path,
) -> None:
    provider_vocab = _vocab((1, 2, 3, 4))
    outside_id = next(
        feature_id
        for feature_id in range(TEST_CACHE_N_SAE_FEATURES)
        if feature_id not in provider_vocab.original_ids
    )
    provider = _provider(
        tmp_path,
        expected_feature_vocab=provider_vocab,
    )

    with pytest.raises(FeatureVocabError, match=rf"\b{outside_id}\b"):
        _validate(
            provider,
            feature_vocab=None,
            target_feature_configs=[FeatureConfig("outside", {outside_id: 0})],
        )


def test_cached_segment_id_outside_vocabulary_is_rejected(tmp_path: Path) -> None:
    explicit_vocab = _vocab((1, 2, 3, 4))
    outside_id = next(
        feature_id
        for feature_id in range(TEST_CACHE_N_SAE_FEATURES)
        if feature_id not in explicit_vocab.original_ids
    )
    segments = (
        ContextFeatureSegment((1,), (1,), 0, 2),
        ContextFeatureSegment((outside_id,), (2,), 2, 4),
    )
    provider = _provider(tmp_path, segments=segments)

    with pytest.raises(FeatureVocabError, match=rf"\b{outside_id}\b"):
        _validate(provider, feature_vocab=explicit_vocab)


def test_target_feature_config_id_outside_vocabulary_is_rejected(
    tmp_path: Path,
) -> None:
    explicit_vocab = _vocab((1, 2, 3, 4))
    outside_id = next(
        feature_id
        for feature_id in range(TEST_CACHE_N_SAE_FEATURES)
        if feature_id not in explicit_vocab.original_ids
    )
    provider = _provider(tmp_path)

    with pytest.raises(FeatureVocabError, match=rf"\b{outside_id}\b"):
        _validate(
            provider,
            feature_vocab=explicit_vocab,
            target_feature_configs=[FeatureConfig("outside", {outside_id: 0})],
        )


def test_none_vocabulary_everywhere_preserves_existing_summary(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    legacy = validate_context_cache_records(
        records=[_record()],
        provider=provider,
        tokenizer=MockTokenizer(),
    )
    explicit_none = validate_context_cache_records(
        records=[_record()],
        provider=provider,
        tokenizer=MockTokenizer(),
        feature_vocab=None,
    )

    assert explicit_none == legacy
