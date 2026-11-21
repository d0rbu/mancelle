# tests/steering/test_feature_mask_apply.py
import pytest
import torch as t

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask


def _seg():
    fids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    mags = t.tensor([1.0, 2.0, 3.0, 4.0])
    pcts = t.tensor([10, 20, 30, 40], dtype=t.uint8)
    return fids, mags, pcts


def test_blacklist_drops():
    fids, mags, pcts = _seg()
    m = FeatureMask("blacklist", (2, 4))
    of, om, op = m.apply_to_positive(fids, mags, pcts)
    assert of.tolist() == [1, 3]
    assert om.tolist() == [1.0, 3.0]
    assert op.tolist() == [10, 30]


def test_whitelist_keeps_only():
    fids, mags, pcts = _seg()
    m = FeatureMask("whitelist", (2, 4, 99))
    of, om, _ = m.apply_to_positive(fids, mags, pcts)
    assert of.tolist() == [2, 4]
    assert om.tolist() == [2.0, 4.0]


def test_empty_in_empty_out():
    e = t.tensor([], dtype=t.int64)
    of, om = FeatureMask("blacklist", (1,)).apply_to_positive(e, e.float())
    assert of.numel() == 0 and om.numel() == 0


def test_allowed_universe():
    bl = FeatureMask("blacklist", (1, 3)).allowed_universe_mask(5)
    assert bl.tolist() == [True, False, True, False, True]
    wl = FeatureMask("whitelist", (1, 3)).allowed_universe_mask(5)
    assert wl.tolist() == [False, True, False, True, False]


def test_assert_no_masked():
    FeatureMask("blacklist", (9,)).assert_no_masked(t.tensor([1, 2]))  # ok
    with pytest.raises(FeatureMaskError):
        FeatureMask("blacklist", (2,)).assert_no_masked(t.tensor([1, 2]))
    with pytest.raises(FeatureMaskError):
        FeatureMask("whitelist", (1,)).assert_no_masked(t.tensor([1, 2]))
