# tests/steering/staging/test_verify_staging_identity.py
import torch as t

from scripts.benchmark.verify_staging_identity import diff_dirs
from steering.feature_io import save_features
from tests.steering.test_feature_io import _make_shard_dict


def test_identical_dirs_have_no_diff(tmp_path):
    d = _make_shard_dict()
    a, b = tmp_path / "a", tmp_path / "b"
    save_features(dict(d), str(a / "shard_00000_features"))
    save_features(dict(d), str(b / "shard_00000_features"))
    assert diff_dirs(str(a), str(b)) == []


def test_tampered_dir_is_flagged(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    save_features(_make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=1), str(a / "shard_00000_features"))
    save_features(_make_shard_dict(n_segments=2, feats_per_seg=5, n_docs=1), str(b / "shard_00000_features"))
    assert diff_dirs(str(a), str(b)) != []


def test_same_structure_different_payload_is_flagged(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    da = _make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=1)
    db = _make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=1)
    db["feature_mags"] = da["feature_mags"] + 1.0  # SAME offsets, different payload
    db["feature_ids"] = da["feature_ids"].clone()  # keep ids identical so only mags differ
    save_features(dict(da), str(a / "shard_00000_features"))
    save_features(dict(db), str(b / "shard_00000_features"))
    errs = diff_dirs(str(a), str(b))
    assert any("feature_mags" in e for e in errs), errs  # payload branch flags it


def test_filename_set_mismatch_is_flagged(tmp_path):
    """Dirs holding different shard filenames must be flagged (the filename-set branch)."""
    a, b = tmp_path / "a", tmp_path / "b"
    save_features(_make_shard_dict(), str(a / "shard_00000_features"))
    save_features(_make_shard_dict(), str(b / "shard_00001_features"))  # different name
    errs = diff_dirs(str(a), str(b))
    assert any("filename set" in e for e in errs), errs


def test_metadata_tensor_mismatch_is_flagged(tmp_path):
    """Same offsets but a differing metadata tensor (doc_token_counts) must be flagged
    (the structural metadata-compare branch)."""
    a, b = tmp_path / "a", tmp_path / "b"
    da = _make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=2)
    db = dict(da)
    db["doc_token_counts"] = da["doc_token_counts"].clone() + 1  # same offsets, different metadata
    save_features(dict(da), str(a / "shard_00000_features"))
    save_features(dict(db), str(b / "shard_00000_features"))
    errs = diff_dirs(str(a), str(b))
    assert any("doc_token_counts" in e for e in errs), errs


def test_feature_ids_mismatch_within_segment_is_flagged(tmp_path):
    """Same offsets and magnitudes but a differing feature_id must be flagged (the
    per-segment feature_ids branch — distinct from the feature_mags branch)."""
    a, b = tmp_path / "a", tmp_path / "b"
    da = _make_shard_dict(n_segments=1, feats_per_seg=2, n_docs=1)
    da["feature_ids"] = t.tensor([3, 5], dtype=t.int32)
    da["feature_mags"] = t.tensor([0.5, 0.9], dtype=t.float32)
    db = dict(da)
    db["feature_ids"] = t.tensor([3, 7], dtype=t.int32)  # same offsets + mags, one id changed
    save_features(dict(da), str(a / "shard_00000_features"))
    save_features(dict(db), str(b / "shard_00000_features"))
    errs = diff_dirs(str(a), str(b))
    assert any("feature_ids" in e for e in errs), errs


def test_mispaired_id_mag_within_segment_is_flagged(tmp_path):
    """Same feature_ids and the SAME magnitude multiset per segment, but with the
    id<->mag pairing swapped, MUST be flagged. Independent per-segment sorting of
    ids and mags would hide this (both sides magnitude-sort to the same sequence),
    so the oracle must compare in stored (feature_id-ascending) order."""
    a, b = tmp_path / "a", tmp_path / "b"
    da = _make_shard_dict(n_segments=1, feats_per_seg=2, n_docs=1)
    da["feature_ids"] = t.tensor([3, 5], dtype=t.int32)   # distinct, ascending
    da["feature_mags"] = t.tensor([0.5, 0.9], dtype=t.float32)
    db = dict(da)
    db["feature_ids"] = t.tensor([3, 5], dtype=t.int32)   # identical ids
    db["feature_mags"] = t.tensor([0.9, 0.5], dtype=t.float32)  # mag<->id pairing swapped
    save_features(dict(da), str(a / "shard_00000_features"))
    save_features(dict(db), str(b / "shard_00000_features"))
    errs = diff_dirs(str(a), str(b))
    assert any("feature_mags" in e for e in errs), errs
