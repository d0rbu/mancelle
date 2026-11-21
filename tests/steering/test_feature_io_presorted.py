from __future__ import annotations

import pytest
import torch as t
import zstandard

import steering.feature_io as feature_io
from errors import EmptyExtractionSegmentError
from scripts.preprocess._splice_profile import SpliceShardProfile
from steering.feature_io import load_features, save_features, save_features_presorted


def _shard(feature_ids: t.Tensor | None = None, offsets: t.Tensor | None = None) -> dict:
    feature_ids = (
        t.tensor([1, 5, 0, 3, 7], dtype=t.int32)
        if feature_ids is None
        else feature_ids
    )
    offsets = t.tensor([0, 2, 5], dtype=t.int64) if offsets is None else offsets
    n_segments = int(offsets.numel() - 1)
    return {
        "offsets": offsets,
        "feature_ids": feature_ids,
        "feature_mags": t.ones(feature_ids.numel(), dtype=t.float32),
        "doc_ids": t.arange(n_segments, dtype=t.int32),
        "segment_indices": t.zeros(n_segments, dtype=t.int32),
        "segment_positions": t.zeros(n_segments, dtype=t.int32),
        "segment_char_starts": t.zeros(n_segments, dtype=t.int32),
        "segment_char_ends": t.ones(n_segments, dtype=t.int32),
        "doc_token_counts": t.ones(n_segments, dtype=t.int32),
        "n_docs": n_segments,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": 2048,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "context_independent_features": False,
        "ignore_bos_features": True,
        "l0_tier": "medium",
    }


def test_presorted_save_roundtrips_like_generic_save(tmp_path):
    data = _shard()
    generic_path = save_features(data, tmp_path / "generic")
    presorted_path = save_features_presorted(data, tmp_path / "presorted")

    generic = load_features(generic_path)
    presorted = load_features(presorted_path)

    assert set(generic) == set(presorted)
    for key, value in generic.items():
        other = presorted[key]
        if isinstance(value, t.Tensor):
            assert value.dtype == other.dtype
            assert t.equal(value, other)
        else:
            assert value == other


def test_presorted_save_rejects_unsorted_segment(tmp_path):
    data = _shard(feature_ids=t.tensor([5, 1], dtype=t.int32), offsets=t.tensor([0, 2]))
    with pytest.raises(ValueError, match="not sorted ascending"):
        save_features_presorted(data, tmp_path / "bad")


def test_presorted_save_rejects_empty_segment(tmp_path):
    data = _shard(feature_ids=t.zeros(0, dtype=t.int32), offsets=t.tensor([0, 0]))
    with pytest.raises(EmptyExtractionSegmentError, match="zero features"):
        save_features_presorted(data, tmp_path / "empty")


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("zstd_threads", [0, 2])
def test_deterministic_presorted_save_pins_bytes_and_preserves_decoded_values(
    tmp_path, monkeypatch, profiled, zstd_threads
):
    data = _shard()
    profile = SpliceShardProfile("shard_00000", 0) if profiled else None
    compressor = zstandard.ZstdCompressor
    calls = []

    def recording_compressor(*args, **kwargs):
        calls.append(kwargs.get("threads", 0))
        return compressor(*args, **kwargs)

    monkeypatch.setattr(feature_io.zstandard, "ZstdCompressor", recording_compressor)
    first = save_features_presorted(
        data, tmp_path / "first", deterministic=True, profile=profile, zstd_threads=zstd_threads
    )
    second = save_features_presorted(
        dict(reversed(list(data.items()))), tmp_path / "second", deterministic=True,
        zstd_threads=zstd_threads,
    )
    assert first.read_bytes() == second.read_bytes()
    assert calls == [zstd_threads, zstd_threads]
    generic = load_features(save_features(data, tmp_path / "generic"))
    deterministic = load_features(first)
    assert set(generic) == set(deterministic)
    for key, value in generic.items():
        other = deterministic[key]
        if isinstance(value, t.Tensor):
            assert value.dtype == other.dtype
            assert t.equal(value, other)
        else:
            assert value == other
    if profile is not None:
        assert profile.detail_seconds["save.safetensors_serialize"] > 0.0


@pytest.mark.parametrize("profiled", [False, True])
def test_default_writers_keep_legacy_serialization(tmp_path, monkeypatch, profiled):
    def unexpected_canonicalization(*_args, **_kwargs):
        pytest.fail("ordinary/default feature writers must not opt into canonical serialization")

    monkeypatch.setattr(feature_io, "deterministic_safetensors_bytes", unexpected_canonicalization)
    data = _shard()
    profile = SpliceShardProfile("shard_00000", 0) if profiled else None
    generic = save_features(data, tmp_path / "generic")
    presorted = save_features_presorted(data, tmp_path / "presorted", profile=profile)
    assert t.equal(load_features(generic)["feature_ids"], load_features(presorted)["feature_ids"])
    for path in (generic, presorted):
        raw = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        assert path.read_bytes() == zstandard.ZstdCompressor(level=feature_io.ZSTD_LEVEL).compress(raw)


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "2", None])
def test_presorted_save_rejects_invalid_threads_before_writing(tmp_path, value):
    with pytest.raises(ValueError, match="zstd_threads must be an integer >= 0"):
        save_features_presorted(_shard(), tmp_path / "bad", zstd_threads=value)
    assert not list(tmp_path.iterdir())
