from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from errors import FeatureMaskError
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    feature_mask_from_checkpoint_meta,
)


def test_absent_meta_returns_none():
    assert feature_mask_from_checkpoint_meta(None) is None


def test_absent_key_returns_none():
    assert feature_mask_from_checkpoint_meta({"user_config": {}}) is None


def test_none_value_returns_none():
    assert feature_mask_from_checkpoint_meta({"feature_mask_canonical": None}) is None


@pytest.mark.parametrize("mode", list(FeatureMaskMode))
def test_round_trip_from_canonical_dict(mode: FeatureMaskMode):
    original = FeatureMask(mode=mode, ids=(1, 4, 9))
    assert (
        feature_mask_from_checkpoint_meta({"feature_mask_canonical": original.canonical_dict()})
        == original
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"mode": "blacklist"},
        {"ids": [1]},
        {"mode": "blacklist", "ids": [1], "extra": True},
    ],
)
def test_wrong_keys_raise(raw: object):
    with pytest.raises(FeatureMaskError, match="exactly"):
        feature_mask_from_checkpoint_meta({"feature_mask_canonical": raw})


def test_non_mapping_raises():
    with pytest.raises(FeatureMaskError, match="mapping"):
        feature_mask_from_checkpoint_meta({"feature_mask_canonical": ["blacklist", [1]]})


def test_ids_not_a_list_raises():
    with pytest.raises(
        FeatureMaskError,
        match="^feature_mask_canonical.ids must be a list$",
    ):
        feature_mask_from_checkpoint_meta(
            {
                "feature_mask_canonical": {
                    "mode": "blacklist",
                    "ids": (1, 2),
                }
            }
        )


def test_empty_ids_raise():
    with pytest.raises(
        FeatureMaskError,
        match="^feature_mask resolved to an empty set$",
    ):
        feature_mask_from_checkpoint_meta(
            {"feature_mask_canonical": {"mode": "whitelist", "ids": []}}
        )


def test_unsorted_ids_raise():
    with pytest.raises(FeatureMaskError, match="strictly increasing"):
        feature_mask_from_checkpoint_meta(
            {"feature_mask_canonical": {"mode": "blacklist", "ids": [2, 1]}}
        )


def test_duplicate_ids_raise():
    with pytest.raises(FeatureMaskError, match="strictly increasing"):
        feature_mask_from_checkpoint_meta(
            {"feature_mask_canonical": {"mode": "whitelist", "ids": [1, 1]}}
        )


def test_bad_mode_raises():
    with pytest.raises(FeatureMaskError, match="mode"):
        feature_mask_from_checkpoint_meta({"feature_mask_canonical": {"mode": "allow", "ids": [1]}})


def test_bool_id_raises():
    with pytest.raises(FeatureMaskError, match="strict ints"):
        feature_mask_from_checkpoint_meta(
            {"feature_mask_canonical": {"mode": "blacklist", "ids": [True]}}
        )


def test_negative_id_raises():
    with pytest.raises(FeatureMaskError, match="non-negative"):
        feature_mask_from_checkpoint_meta(
            {"feature_mask_canonical": {"mode": "blacklist", "ids": [-1]}}
        )


@given(
    mode=st.sampled_from(list(FeatureMaskMode)),
    ids=st.sets(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=50),
)
def test_hypothesis_round_trip(mode: FeatureMaskMode, ids: set[int]):
    original = FeatureMask(mode=mode, ids=tuple(ids))
    restored = feature_mask_from_checkpoint_meta(
        {"feature_mask_canonical": original.canonical_dict()}
    )
    assert restored == original
