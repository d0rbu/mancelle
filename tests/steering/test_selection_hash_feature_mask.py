# tests/steering/test_selection_hash_feature_mask.py
from steering.packed_batch_io import compute_selection_hash

POS = {"threshold_type": "segment_topk", "threshold": 2048}


def _legacy():
    return compute_selection_hash(POS, None, None, 10, 42, 0)


def test_disabled_is_byte_identical_legacy():
    # Frozen golden: the legacy 6-key payload hash must NOT change.
    golden = _legacy()
    assert compute_selection_hash(POS, None, None, 10, 42, 0, feature_mask=None) == golden
    assert compute_selection_hash(POS, None, None, 10, 42, 0) == golden


def test_mask_changes_hash_and_is_distinct_by_mode():
    base = _legacy()
    bl = compute_selection_hash(POS, None, None, 10, 42, 0,
                                feature_mask={"mode": "blacklist", "ids": [1, 2]})
    wl = compute_selection_hash(POS, None, None, 10, 42, 0,
                                feature_mask={"mode": "whitelist", "ids": [1, 2]})
    assert len({base, bl, wl}) == 3


def test_canonical_invariance():
    a = compute_selection_hash(POS, None, None, 10, 42, 0,
                               feature_mask={"mode": "blacklist", "ids": [1, 2]})
    b = compute_selection_hash(POS, None, None, 10, 42, 0,
                               feature_mask={"mode": "blacklist", "ids": [2, 1]})
    assert a == b
