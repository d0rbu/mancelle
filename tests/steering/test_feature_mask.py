# tests/steering/test_feature_mask.py
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureMaskError
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
    feature_mask_dirname,
)


def test_featuremask_sorts_and_dedups():
    m = FeatureMask(mode="blacklist", ids=(5, 1, 5, 3))
    assert m.ids == (1, 3, 5)


def test_featuremask_rejects_bad_mode():
    with pytest.raises(FeatureMaskError):
        FeatureMask(mode="nope", ids=(1,))


def test_featuremask_rejects_empty():
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode="whitelist", ids=())


def test_canonical_dict_none():
    assert canonical_dict(None) is None


def test_canonical_dict_shape():
    assert canonical_dict(FeatureMask("blacklist", (3, 1))) == {
        "mode": "blacklist",
        "ids": [1, 3],
    }


# ---------------------------------------------------------------------------
# H0a new tests
# ---------------------------------------------------------------------------


def test_mode_is_strenum():
    assert issubclass(FeatureMaskMode, str)
    assert FeatureMaskMode("blacklist") is FeatureMaskMode.blacklist
    assert set(FeatureMaskMode) == {FeatureMaskMode.blacklist, FeatureMaskMode.whitelist}


def test_featuremask_mode_coerced_to_enum():
    assert FeatureMask(mode="blacklist", ids=(1,)).mode is FeatureMaskMode.blacklist
    assert FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1,)).mode is FeatureMaskMode.whitelist


def test_featuremask_invalid_mode_raises_FeatureMaskError():
    with pytest.raises(FeatureMaskError):
        FeatureMask("nope", (1,))
    with pytest.raises(BeartypeCallHintParamViolation, match="mode"):
        FeatureMask(123, (1,))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_canonical_dict_mode_is_plain_str_for_hash_stability():
    d = canonical_dict(FeatureMask("blacklist", (1,)))
    assert d is not None
    assert d["mode"] == "blacklist"
    assert type(d["mode"]) is str


# ---------------------------------------------------------------------------
# H0b new tests — method API exists and matches module wrappers
# ---------------------------------------------------------------------------


def test_method_canonical_dict_matches_wrapper():
    """m.canonical_dict() == canonical_dict(m); order-invariant; wrapper None-safe."""
    m = FeatureMask("blacklist", (2, 1))
    assert m.canonical_dict() == canonical_dict(m)
    assert m.canonical_dict() == canonical_dict(FeatureMask("blacklist", (1, 2)))
    assert canonical_dict(None) is None


def test_method_dirname_matches_wrapper():
    """m.dirname() == feature_mask_dirname(m); wrapper returns '' for None."""
    m = FeatureMask("blacklist", (2, 1))
    assert m.dirname() == feature_mask_dirname(m)
    assert feature_mask_dirname(None) == ""


def test_method_apply_to_positive_blacklist():
    """m.apply_to_positive drops blacklisted ids."""
    m = FeatureMask("blacklist", (1, 2))
    fids = t.tensor([1, 2, 3])
    result, = m.apply_to_positive(fids)[:1]
    assert result.tolist() == [3]


def test_method_assert_no_masked_ok_and_raises():
    """m.assert_no_masked passes on clean tensor; raises FeatureMaskError on violation."""
    m = FeatureMask("blacklist", (1, 2))
    m.assert_no_masked(t.tensor([3]))  # ok
    with pytest.raises(FeatureMaskError):
        m.assert_no_masked(t.tensor([1]))
