"""Vocabulary-aware validation tests for static target conditioning."""

from __future__ import annotations

import pytest

from errors import FeatureVocabError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.target_conditioning import validate_feature_dict
from steering.refined_types import PositiveInt

N_SAE = PositiveInt.parse(64)
NUM_BUCKETS = PositiveInt.parse(4)


def _vocab(*, n_sae_features: PositiveInt = N_SAE) -> FeatureVocab:
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        tuple(range(12, n_sae_features)),
    )
    vocab = effective_feature_vocab(mask, n_sae_features)
    assert vocab is not None
    return vocab


def _legacy_outcome(
    features: dict[int, int] | None,
    *,
    explicit_none: bool,
) -> tuple[type[ValueError], str] | None:
    try:
        if explicit_none:
            validate_feature_dict(
                features,
                N_SAE,
                NUM_BUCKETS,
                name="unit target",
                feature_vocab=None,
            )
        else:
            validate_feature_dict(
                features,
                N_SAE,
                NUM_BUCKETS,
                name="unit target",
            )
    except ValueError as exc:
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize(
    "features",
    [
        None,
        {},
        {0: 0, N_SAE - 1: NUM_BUCKETS - 1},
        {N_SAE: 0},
        {0: NUM_BUCKETS},
    ],
)
def test_none_vocabulary_preserves_existing_validation_behavior(
    features: dict[int, int] | None,
) -> None:
    assert _legacy_outcome(features, explicit_none=True) == _legacy_outcome(
        features,
        explicit_none=False,
    )


def test_all_requested_ids_inside_vocabulary_are_accepted() -> None:
    vocab = _vocab()
    features = {
        feature_id: index % NUM_BUCKETS
        for index, feature_id in enumerate(vocab.original_ids[:4])
    }

    validate_feature_dict(
        features,
        N_SAE,
        NUM_BUCKETS,
        name="unit target",
        feature_vocab=vocab,
    )


def test_in_range_id_outside_vocabulary_is_rejected_by_id() -> None:
    vocab = _vocab()
    outside_id = next(
        feature_id
        for feature_id in range(N_SAE)
        if feature_id not in vocab.original_ids
    )

    with pytest.raises(FeatureVocabError, match=rf"\b{outside_id}\b"):
        validate_feature_dict(
            {outside_id: 0},
            N_SAE,
            NUM_BUCKETS,
            name="unit target",
            feature_vocab=vocab,
        )


def test_many_outside_ids_report_full_count_and_truncate_after_eight() -> None:
    vocab = _vocab()
    outside_ids = [
        feature_id
        for feature_id in reversed(range(N_SAE))
        if feature_id not in vocab.original_ids
    ]
    assert len(outside_ids) == 12
    reported = ", ".join(str(feature_id) for feature_id in sorted(outside_ids)[:8])

    with pytest.raises(
        FeatureVocabError,
        match=rf"12.*\[{reported}\].*\(\+4 more\)",
    ):
        validate_feature_dict(
            dict.fromkeys(outside_ids, 0),
            N_SAE,
            NUM_BUCKETS,
            name="unit target",
            feature_vocab=vocab,
        )


def test_vocabulary_width_must_match_validation_width() -> None:
    other_width = PositiveInt.parse(N_SAE + 1)

    with pytest.raises(FeatureVocabError, match="n_sae_features"):
        validate_feature_dict(
            {N_SAE - 1: 0},
            N_SAE,
            NUM_BUCKETS,
            name="unit target",
            feature_vocab=_vocab(n_sae_features=other_width),
        )


def test_empty_features_still_validate_vocabulary_width_first() -> None:
    other_width = PositiveInt.parse(N_SAE + 1)

    with pytest.raises(FeatureVocabError, match="n_sae_features"):
        validate_feature_dict(
            {},
            N_SAE,
            NUM_BUCKETS,
            name="unit target",
            feature_vocab=_vocab(n_sae_features=other_width),
        )
