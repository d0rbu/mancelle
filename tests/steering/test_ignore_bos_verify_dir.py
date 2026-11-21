"""verify_feature_dir cross-checks ignore_bos_features (shard + stats levels)."""

import json
import struct

import pytest
import torch as t

from steering.feature_io import FeatureDirectoryMismatchError, save_features
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import _read_metadata, save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)


def _dir(tmp_path, ignore_bos_leaf):
    d = build_feature_dir(
        root=str(tmp_path), segment_size=128, layer=8,
        activation_type=ActivationType.residual, pooling=PoolingMode.max_pool,
        width=SAEWidth("16k"), model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE, ignore_bos_features=ignore_bos_leaf,
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_shard(d, name, ignore_bos_meta):
    save_features({
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "n_docs": 1, "n_segments": 1, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_mode": "fixed_tokens",
        "segment_size": 128, "ignore_bos_features": ignore_bos_meta,
    }, d / name)


def _write_stats(d, ignore_bos: bool):
    save_statistics(
        ExtractionStatistics(
            metadata=MetadataStats("max_pool", SegmentMode.fixed_tokens, 128,
                                    2048, 8, 16384, "2026-05-17T00:00:00+00:00"),
            corpus=CorpusStats(1, 1, 1, 1.0, 1),
            features=SegmentFeatureStats(
                1, 1, 1.0, 1, 1, 1, 1, 16384, 1.0, {}, {}, {}, {}),
            storage=StorageStats(1, 1.0, 1.0),
            per_feature_stats={},
            extrapolation_10b=Extrapolation(1, 1, 1.0, 1.0),
            ignore_bos_features=ignore_bos,
        ),
        d / "feature_statistics.safetensors",
    )


def _strip_stats_ignore_bos_key(d):
    """Simulate a stats file written before the field existed: remove the
    ``ignore_bos_features`` key from a plain-safetensors stats header."""
    p = d / "feature_statistics.safetensors"
    raw = p.read_bytes()
    n = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + n])
    header.get("__metadata__", {}).pop("ignore_bos_features", None)
    nh = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(nh)) + nh + raw[8 + n :])


def test_consistent_no_bos_dir_passes(tmp_path):
    d = _dir(tmp_path, True)
    _write_shard(d, "shard_00000_features", True)
    _write_stats(d, True)
    verify_feature_dir(d)  # no raise


def test_shard_meta_mismatch_raises(tmp_path):
    d = _dir(tmp_path, True)  # leaf says no_bos
    _write_shard(d, "shard_00000_features", False)  # shard says false
    _write_stats(d, True)
    with pytest.raises(FeatureDirectoryMismatchError, match="ignore_bos_features"):
        verify_feature_dir(d)


def test_stats_meta_mismatch_raises(tmp_path):
    d = _dir(tmp_path, True)
    _write_shard(d, "shard_00000_features", True)
    _write_stats(d, False)  # stats disagrees with leaf
    with pytest.raises(FeatureDirectoryMismatchError, match="ignore_bos_features"):
        verify_feature_dir(d)


def test_legacy_stats_without_key_not_flagged(tmp_path):
    d = _dir(tmp_path, False)  # legacy layout, no /no_bos leaf
    _write_shard(d, "shard_00000_features", False)
    _write_stats(d, False)
    _strip_stats_ignore_bos_key(d)  # genuinely a pre-feature stats file
    assert "ignore_bos_features" not in _read_metadata(
        d / "feature_statistics.safetensors"
    )
    verify_feature_dir(d)  # no raise (guarded back-compat: key absent → skip)


def test_legacy_stats_without_key_still_flags_shard_leaf_mismatch(tmp_path):
    # Even with a pre-feature (keyless) stats file, a SHARD whose
    # ignore_bos_features disagrees with the /no_bos leaf must still be caught
    # (the guarded stats check being skipped must not mask the shard check).
    d = _dir(tmp_path, True)  # /no_bos leaf
    _write_shard(d, "shard_00000_features", False)  # shard says false ≠ leaf
    _write_stats(d, True)
    _strip_stats_ignore_bos_key(d)
    with pytest.raises(FeatureDirectoryMismatchError, match="ignore_bos_features"):
        verify_feature_dir(d)


# ---------------------------------------------------------------------------
# Post-#466: corrupt-shard surface — new tests
# ---------------------------------------------------------------------------


def test_no_bos_dir_with_corrupt_shard_surfaces_in_report(tmp_path):
    """verify_feature_dir accumulates the corrupt-shard error for the empty
    shard, does NOT crash on the ignore_bos block (the corrupt shard continues
    before reaching it), and does NOT falsely flag the valid shard."""
    d = _dir(tmp_path, True)  # /no_bos leaf → expected.ignore_bos_features=True
    # Valid shard: shard_00001 (written after the corrupt one so glob order is predictable)
    _write_shard(d, "shard_00001_features", True)
    # Corrupt shard: shard_00000 (sorts before shard_00001 in glob)
    corrupt = d / "shard_00000_features.safetensors.zst"
    corrupt.write_bytes(b"")
    _write_stats(d, True)

    with pytest.raises(FeatureDirectoryMismatchError) as exc_info:
        verify_feature_dir(d)
    msg = str(exc_info.value)
    # Corruption surfaced by name
    assert "shard_00000_features.safetensors.zst" in msg
    # Valid shard not falsely flagged
    assert "shard_00001" not in msg


def test_shard_malformed_ignore_bos_metadata_accumulates(tmp_path, monkeypatch):
    """A shard whose ignore_bos_features is an unrecognized string ("yes")
    must produce a 'malformed' error accumulated in the FeatureDirectoryMismatchError
    report.  The monkeypatched meta must include context_independent_features as
    "false" (so that block does not raise ValueError and skip our block)."""
    import steering.feature_paths as fp_mod

    d = _dir(tmp_path, True)
    _write_shard(d, "shard_00000_features", True)
    _write_stats(d, True)

    # Provide a meta dict that is valid for layer/segment_size/pooling/
    # context_independent_features, but has a garbage ignore_bos_features value.
    # This ensures we reach the ignore_bos block and trigger the malformed branch.
    monkeypatch.setattr(
        fp_mod,
        "read_shard_metadata",
        lambda _path: {
            "layer": "8",
            "segment_size": "128",
            "pooling": "max_pool",
            "context_independent_features": "false",
            "ignore_bos_features": "yes",
        },
    )

    with pytest.raises(FeatureDirectoryMismatchError, match="ignore_bos_features.*malformed"):
        verify_feature_dir(d)


def test_stats_malformed_ignore_bos_raises(tmp_path, monkeypatch):
    """Stats-level ignore_bos_features with a garbage value ("bad") must
    produce a 'ignore_bos_features malformed' error in the accumulated report."""
    import steering.feature_paths as fp_mod

    d = _dir(tmp_path, True)
    _write_shard(d, "shard_00000_features", True)
    _write_stats(d, True)

    # Capture the real stats metadata, then inject a bad ignore_bos_features value.
    real_read_stats_metadata = fp_mod._read_stats_metadata

    def _patched_read_stats_metadata(path):
        meta = dict(real_read_stats_metadata(path))
        meta["ignore_bos_features"] = "bad"
        return meta

    monkeypatch.setattr(fp_mod, "_read_stats_metadata", _patched_read_stats_metadata)

    with pytest.raises(FeatureDirectoryMismatchError, match="ignore_bos_features malformed"):
        verify_feature_dir(d)
