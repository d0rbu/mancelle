# tests/steering/test_feature_mask_dirname.py
from steering.conditional.feature_mask import FeatureMask, feature_mask_dirname


def test_dirname_disabled_is_empty():
    """I2: no mask → empty leaf (byte-identical legacy checkpoint path)."""
    assert feature_mask_dirname(None) == ""


def test_dirname_prefix_and_stability():
    """Deterministic, order-independent, 16-hex format."""
    a = feature_mask_dirname(FeatureMask("blacklist", (3, 1)))
    b = feature_mask_dirname(FeatureMask("blacklist", (1, 3)))
    # Canonical order: ids sorted, so (3,1) == (1,3)
    assert a == b
    assert a.startswith("bl_")
    # 16 hex digits after "bl_" prefix (64-bit hash)
    assert len(a) == 3 + 16, f"expected 16-hex leaf, got {a!r} (len={len(a)})"


def test_dirname_mode_distinguishes():
    """Blacklist vs whitelist produce different leaves (distinct mode prefix AND hash)."""
    bl = feature_mask_dirname(FeatureMask("blacklist", (1,)))
    wl = feature_mask_dirname(FeatureMask("whitelist", (1,)))
    assert bl.startswith("bl_") and wl.startswith("wl_")
    assert bl[3:] != wl[3:]


def test_dirname_known_value_blacklist():
    """Pin the exact 16-hex leaf for a fixed blacklist mask (regression pin)."""
    leaf = feature_mask_dirname(FeatureMask("blacklist", (3, 1)))
    assert leaf == "bl_1a3a22d167c1f6bc", f"unexpected leaf: {leaf!r}"


def test_dirname_known_value_whitelist():
    """Pin the exact 16-hex leaf for a fixed whitelist mask (regression pin)."""
    leaf = feature_mask_dirname(FeatureMask("whitelist", (1,)))
    assert leaf == "wl_517ed1ce9ebaa003", f"unexpected leaf: {leaf!r}"


def test_dirname_distinct_masks_distinct_leaves():
    """Different id sets → different leaves."""
    a = feature_mask_dirname(FeatureMask("blacklist", (1,)))
    b = feature_mask_dirname(FeatureMask("blacklist", (2,)))
    assert a != b, "different masks must not produce the same leaf"


def test_dirname_leaf_hash_is_64bit():
    """Structural regression pin: the hex portion is exactly 16 chars (64 bits),
    not 8 (32 bits). Protects against a future truncation narrowing it back."""
    for mode, ids in [("blacklist", (5,)), ("whitelist", (10, 20))]:
        leaf = feature_mask_dirname(FeatureMask(mode, ids))
        prefix_len = 3  # "bl_" or "wl_"
        hex_part = leaf[prefix_len:]
        assert len(hex_part) == 16, (
            f"leaf hash must be 16 hex chars (64 bits), got {len(hex_part)} in {leaf!r}"
        )
        assert all(c in "0123456789abcdef" for c in hex_part), (
            f"leaf hash must be lowercase hex, got {hex_part!r}"
        )
