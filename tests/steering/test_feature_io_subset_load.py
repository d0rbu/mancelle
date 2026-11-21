"""Field-subset feature loading equivalence (issue #529, Phase 3).

``load_features(path, keys=...)`` must return tensors byte-identical to the full load for the
requested fields, skip the unrequested (large) feature tensors, and still parse scalar metadata.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
import zstandard
from safetensors.torch import save as safetensors_save

from steering.feature_io import (
    TensorKey,
    load_features,
    save_features,
    spooled_feature_shard,
)
from tests.scripts._token_boundary_fixtures import _shard_data

_SEG_KEYS = {
    "offsets",
    "doc_ids",
    "segment_indices",
    "segment_char_starts",
    "segment_char_ends",
}


def _save(tmp_path):
    return save_features(_shard_data(0), tmp_path / "shard_00000_features")


def _write_raw_safetensors_zst(
    path: Path,
    tensors: dict[str, t.Tensor],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor().compress(safetensors_save(tensors)))
    return path


def test_subset_load_equals_full(tmp_path):
    p = _save(tmp_path)
    full = load_features(p)
    sub = load_features(p, keys=_SEG_KEYS)

    for k in _SEG_KEYS:
        assert t.equal(sub[k], full[k]), f"{k}: subset != full"
    # The large feature tensors are skipped entirely.
    assert "feature_ids" not in sub
    assert "feature_mags" not in sub
    # Scalar metadata is always parsed (cheap).
    assert int(sub["segment_size"]) == int(full["segment_size"])
    assert int(sub["n_segments"]) == int(full["n_segments"])


def test_subset_load_none_is_full(tmp_path):
    p = _save(tmp_path)
    full = load_features(p)
    same = load_features(p, keys=None)
    assert set(same) == set(full)
    for k, v in full.items():
        if isinstance(v, t.Tensor):
            assert t.equal(same[k], v), f"{k} changed under keys=None"


def test_subset_load_unknown_key_raises(tmp_path):
    p = _save(tmp_path)
    # ValueError (not a bare assert) so the loud-fail contract survives `python -O`.
    with pytest.raises(ValueError, match="not present"):
        load_features(p, keys={"offsets", "not_a_real_tensor"})


def test_subset_load_feature_ids_without_offsets_raises(tmp_path):
    # feature_ids are delta-encoded against offsets; requesting them without offsets would
    # silently return raw deltas — must raise instead of corrupt-and-return.
    p = _save(tmp_path)
    with pytest.raises(ValueError, match="offsets"):
        load_features(p, keys={"feature_ids"})


def test_spooled_feature_shard_full_tensors_match_eager_load(tmp_path):
    p = _save(tmp_path)
    full = load_features(p)
    spool_dir = tmp_path / "spool"

    with spooled_feature_shard(p, temp_dir=spool_dir) as shard:
        assert shard.tensor_keys.issuperset(_SEG_KEYS | {"feature_ids", "feature_mags"})
        assert any(spool_dir.iterdir())
        for key, expected in full.items():
            if isinstance(expected, t.Tensor):
                assert t.equal(shard.get_tensor(key), expected), f"{key} changed"
        assert shard.parsed_metadata()["segment_size"] == full["segment_size"]
        assert shard.parsed_metadata()["n_segments"] == full["n_segments"]

    assert not list(spool_dir.iterdir())


def test_spooled_feature_shard_slice_reads_match_eager_for_plain_tensors(tmp_path):
    p = _save(tmp_path)
    full = load_features(p)

    with spooled_feature_shard(p, temp_dir=tmp_path / "spool") as shard:
        assert t.equal(
            shard.get_tensor_slice(TensorKey.FEATURE_MAGS, 1, 4),
            full[TensorKey.FEATURE_MAGS][1:4],
        )
        assert t.equal(
            shard.get_tensor_slice(TensorKey.OFFSETS, 0, 3),
            full[TensorKey.OFFSETS][0:3],
        )
        assert shard.tensor_shape(TensorKey.FEATURE_IDS) == [
            int(full[TensorKey.FEATURE_IDS].numel())
        ]


@pytest.mark.parametrize(
    ("start", "end", "pattern"),
    [
        (-1, 1, "invalid slice"),
        (2, 1, "invalid slice"),
        (0, 10_000, "out of bounds"),
    ],
)
def test_spooled_feature_shard_slice_rejects_invalid_bounds(
    tmp_path,
    start,
    end,
    pattern,
):
    p = _save(tmp_path)

    with spooled_feature_shard(p, temp_dir=tmp_path / "spool") as shard:
        with pytest.raises(ValueError, match=pattern):
            shard.get_tensor_slice(TensorKey.FEATURE_MAGS, start, end)


def test_spooled_feature_shard_unknown_tensor_raises(tmp_path):
    p = _save(tmp_path)

    with spooled_feature_shard(p, temp_dir=tmp_path / "spool") as shard:
        with pytest.raises(ValueError, match="not present"):
            shard.get_tensor("not_a_real_tensor")


def test_spooled_feature_shard_rejects_invalid_feature_id_decode_metadata(tmp_path):
    p = _write_raw_safetensors_zst(
        tmp_path / "shard_00000_features.safetensors.zst",
        {
            str(TensorKey.OFFSETS): t.tensor([0, 2], dtype=t.int64),
            str(TensorKey.FEATURE_IDS): t.tensor([1], dtype=t.int64),
        },
    )

    with spooled_feature_shard(p, temp_dir=tmp_path / "spool") as shard:
        with pytest.raises(ValueError, match=r"offsets\[-1\]"):
            shard.get_tensor(TensorKey.FEATURE_IDS)


def test_spooled_feature_shard_cleans_temp_file_after_exception(tmp_path):
    p = _save(tmp_path)
    spool_dir = tmp_path / "spool"

    with pytest.raises(RuntimeError, match="boom"):
        with spooled_feature_shard(p, temp_dir=spool_dir):
            assert any(spool_dir.iterdir())
            raise RuntimeError("boom")

    assert not list(spool_dir.iterdir())
