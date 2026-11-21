# tests/steering/test_feature_mask_append_leaf.py
"""Unit tests for append_mask_leaf (Step 0 helper)."""

import os

import pytest

from errors import FeatureMaskError
from steering.conditional.feature_mask import (
    FeatureMask,
    append_mask_leaf,
    feature_mask_dirname,
    feature_mask_leaf_from_cfg,
    resolve_feature_mask,
)


def test_no_mask_returns_path_unchanged():
    """mask=None → returns path byte-identical (I2)."""
    assert append_mask_leaf("/a/b", None) == "/a/b"


def test_no_mask_trailing_slash_unchanged():
    assert append_mask_leaf("/a/b/", None) == "/a/b/"


def test_blacklist_mask_appends_bl_leaf():
    mask = FeatureMask("blacklist", (3, 1, 2))
    result = append_mask_leaf("/a/b", mask)
    expected = os.path.join("/a/b", feature_mask_dirname(mask))
    assert result == expected
    assert "/bl_" in result


def test_whitelist_mask_appends_wl_leaf():
    mask = FeatureMask("whitelist", (7, 99))
    result = append_mask_leaf("/a/b", mask)
    assert "/wl_" in result
    assert result == os.path.join("/a/b", feature_mask_dirname(mask))


def test_mask_leaf_exactly_one_bl_component():
    mask = FeatureMask("blacklist", (1,))
    result = append_mask_leaf("/a/b", mask)
    parts = result.split("/")
    bl_parts = [p for p in parts if p.startswith("bl_")]
    assert len(bl_parts) == 1


def test_mask_leaf_exactly_one_wl_component():
    mask = FeatureMask("whitelist", (1,))
    result = append_mask_leaf("/a/b", mask)
    parts = result.split("/")
    wl_parts = [p for p in parts if p.startswith("wl_")]
    assert len(wl_parts) == 1


def test_mask_equals_join_with_feature_mask_dirname():
    mask = FeatureMask("blacklist", (5, 10, 15))
    assert append_mask_leaf("/a/b", mask) == os.path.join("/a/b", feature_mask_dirname(mask))


def test_no_mask_empty_path():
    assert append_mask_leaf("", None) == ""


def test_mask_on_empty_path():
    mask = FeatureMask("blacklist", (1,))
    result = append_mask_leaf("", mask)
    assert result == feature_mask_dirname(mask)


# ---------------------------------------------------------------------------
# feature_mask_leaf_from_cfg
# ---------------------------------------------------------------------------


def test_cfg_block_none_returns_empty():
    assert feature_mask_leaf_from_cfg(None) == ""


def test_cfg_block_no_feature_mask_returns_empty():
    assert feature_mask_leaf_from_cfg({}) == ""


def test_cfg_block_with_blacklist():
    leaf = feature_mask_leaf_from_cfg(
        {"feature_mask": {"mode": "blacklist", "ids": [3, 1, 2]}}
    )
    assert leaf.startswith("bl_")
    # Must match the fully-resolved dirname for same ids
    expected = feature_mask_dirname(FeatureMask("blacklist", (1, 2, 3)))
    assert leaf == expected


def test_cfg_block_with_whitelist():
    leaf = feature_mask_leaf_from_cfg(
        {"feature_mask": {"mode": "whitelist", "ids": [7, 99]}}
    )
    assert leaf.startswith("wl_")


# ---------------------------------------------------------------------------
# RISK-A anti-drift: feature_mask_leaf_from_cfg == feature_mask_dirname(resolve_feature_mask(cfg, n))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,ids", [
    ("blacklist", [3, 7, 42]),
    ("whitelist", [1, 2]),
    ("blacklist", [0]),
    ("whitelist", [9, 1, 4]),     # unsorted but unique+in-range → canonical-order must agree
])
def test_leaf_from_cfg_matches_resolve_then_dirname(mode, ids):
    """RISK-A anti-drift: for VALID configs both leaf paths must agree
    (orchestrator dir == the dir the validated trainer writes)."""
    cfg = {"feature_mask": {"mode": mode, "ids": list(ids)}}   # use the cfg shape the existing tests/_coerce_block use
    n_sae = max(ids) + 1
    assert feature_mask_leaf_from_cfg(cfg) == feature_mask_dirname(
        resolve_feature_mask(cfg, n_sae)
    )


def test_leaf_from_cfg_disabled_is_empty():
    assert feature_mask_leaf_from_cfg({}) == ""
    assert feature_mask_leaf_from_cfg({"feature_mask": None}) == ""


# ---------------------------------------------------------------------------
# H2: strict-int validation for feature_mask_leaf_from_cfg (A2 bug fix)
# ---------------------------------------------------------------------------


def test_leaf_from_cfg_rejects_duplicate_ids():
    """feature_mask_leaf_from_cfg must reject duplicate ids fast (A2 fix)."""
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "ids": [3, 3]}})


def test_leaf_from_cfg_rejects_float_and_bool_ids():
    """feature_mask_leaf_from_cfg must reject float and bool ids (A2 fix)."""
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "ids": [3.9]}})
    # Integral float also rejected — strict-int
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "ids": [3.0]}})
    # bool
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "ids": [True]}})
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "ids": [False]}})


def test_leaf_from_cfg_rejects_empty_after_parse(tmp_path):
    """feature_mask_leaf_from_cfg must reject empty id lists (A2 fix)."""
    # comment-only .txt file → no ids after parse → empty error
    p = tmp_path / "ids.txt"
    p.write_text("# only a comment\n\n")
    with pytest.raises(FeatureMaskError, match="empty"):
        feature_mask_leaf_from_cfg({"feature_mask": {"mode": "blacklist", "path": str(p)}})


# ---------------------------------------------------------------------------
# H2: symmetry contract — resolve and leaf-from-cfg validate symmetrically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg,n", [
    # duplicate ids
    ({"feature_mask": {"mode": "blacklist", "ids": [3, 3]}}, 100),
    # float id
    ({"feature_mask": {"mode": "blacklist", "ids": [3.7]}}, 100),
    # bool id
    ({"feature_mask": {"mode": "blacklist", "ids": [True]}}, 100),
    # neither ids nor path
    ({"feature_mask": {"mode": "blacklist"}}, 100),
    # both ids and path
    ({"feature_mask": {"mode": "blacklist", "ids": [1], "path": "/x"}}, 100),
    # bad mode
    ({"feature_mask": {"mode": "sideways", "ids": [1]}}, 100),
])
def test_resolve_and_leaf_validation_symmetric(cfg, n):
    """BOTH resolve_feature_mask AND feature_mask_leaf_from_cfg must reject the same
    invalid cfgs (symmetry contract). The only acceptable asymmetry is the
    n_sae_features range check — which is resolve-only by design."""
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask(cfg, n)
    with pytest.raises(FeatureMaskError):
        feature_mask_leaf_from_cfg(cfg)


@pytest.mark.parametrize("mode,ids", [
    ("blacklist", [3, 7, 42]),
    ("whitelist", [1, 2]),
    ("blacklist", [0]),
    ("whitelist", [9, 1, 4]),
])
def test_resolve_and_leaf_valid_cfg_risk_a(mode, ids):
    """RISK-A anti-drift: for VALID configs, leaf == dirname(resolve(cfg)).
    Extends test_leaf_from_cfg_matches_resolve_then_dirname to explicitly assert
    the H2 symmetry contract for valid inputs."""
    cfg = {"feature_mask": {"mode": mode, "ids": list(ids)}}
    n_sae = max(ids) + 1
    leaf = feature_mask_leaf_from_cfg(cfg)
    resolved = resolve_feature_mask(cfg, n_sae)
    assert resolved is not None
    assert leaf == feature_mask_dirname(resolved)
