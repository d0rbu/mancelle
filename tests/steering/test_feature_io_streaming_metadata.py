"""Streaming metadata read — robustification of PR #466.

Covers _read_exact, hardened read_shard_header (corruption matrix +
short-read reassembly), public read_shard_metadata, read_l0_tier_from_shard
corruption behavior, and end-to-end l0-tier guard integration.

Real on-disk artifacts via tmp_path + real zstandard; the only fakes are
chunk-limited / counting readers used to prove the read-loop and early-stop.
"""

import json
import struct

import pytest
import torch as t
import zstandard

from steering.feature_io import (
    MAX_SAFETENSORS_HEADER_SIZE,
    ZSTD_LEVEL,
    CorruptFeatureShardError,
    FeatureDirectoryMismatchError,
    TensorKey,
    _read_exact,
    _read_safetensors_metadata_from_bytes,
    assert_l0_tier_compatible,
    assert_uniform_l0_tier,
    glob_features,
    load_features,
    read_feature_value_count,
    read_l0_tier_from_shard,
    read_shard_header,
    read_shard_metadata,
    save_features,
)
from steering.feature_paths import (
    ActivationType,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
    verify_segment_size,
)
from steering.sae_resolver import L0Tier, SAEWidth


class _ChunkReader:
    """A reader whose .read(n) returns at most ``chunk`` bytes per call,
    simulating a zstandard version that short-reads."""

    def __init__(self, data: bytes, chunk: int):
        self._data = data
        self._pos = 0
        self._chunk = chunk

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data) - self._pos
        take = min(n, self._chunk, len(self._data) - self._pos)
        out = self._data[self._pos : self._pos + take]
        self._pos += take
        return out


class TestReadExact:
    """Unit tests for the ``_read_exact`` read-loop helper."""

    def test_returns_exactly_n_when_available(self):
        r = _ChunkReader(b"abcdefghij", chunk=10)
        assert _read_exact(r, 4) == b"abcd"

    def test_returns_short_at_true_eof(self):
        r = _ChunkReader(b"abc", chunk=10)
        assert _read_exact(r, 8) == b"abc"

    def test_reassembles_across_chunk_limited_reader(self):
        # chunk=1 forces 6 read() calls; _read_exact must stitch them.
        r = _ChunkReader(b"HEADER", chunk=1)
        assert _read_exact(r, 6) == b"HEADER"

    def test_zero_length_request_returns_empty(self):
        r = _ChunkReader(b"abc", chunk=2)
        assert _read_exact(r, 0) == b""

    def test_partial_reassembly_then_eof(self):
        # Accumulate across >1 one-byte chunks, then hit EOF before n:
        # exercises the loop-back arc AND the falsy-chunk break together.
        r = _ChunkReader(b"abcde", chunk=1)
        assert _read_exact(r, 64) == b"abcde"


def _make_real_shard(tmp_path, **extra):
    """Write a real shard via save_features; return its .safetensors.zst path."""
    n = 8
    data = {
        "offsets": t.tensor([0, n], dtype=t.int64),
        "feature_ids": t.randint(0, 100, (n,), dtype=t.int32),
        "feature_mags": t.rand(n) + 0.1,
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([10], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "segment_size": 64,
        "pooling": "mean_pool",
    }
    data.update(extra)
    return save_features(data, tmp_path / "shard_00000")


def _write_compressed(path, raw: bytes):
    path.write_bytes(zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(raw))


class TestReadFeatureValueCount:
    def test_reads_real_feature_id_length_from_streamed_header(self, tmp_path):
        path = _make_real_shard(tmp_path)
        assert read_feature_value_count(path) == 8

    @pytest.mark.parametrize(
        "descriptor",
        [None, [], {}, {"shape": []}, {"shape": [True]}, {"shape": [-1]}, {"shape": [1, 2]}],
    )
    def test_rejects_missing_or_malformed_feature_id_shape(self, tmp_path, descriptor):
        path = tmp_path / "bad_features.safetensors.zst"
        header: dict[str, object] = {}
        if descriptor is not None:
            header[TensorKey.FEATURE_IDS.value] = descriptor
        encoded = json.dumps(header, separators=(",", ":")).encode()
        _write_compressed(path, struct.pack("<Q", len(encoded)) + encoded)
        with pytest.raises(CorruptFeatureShardError, match="feature_ids"):
            read_feature_value_count(path)


class TestReadShardHeaderCorruptionMatrix:
    def test_zero_byte_file(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        p.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_frame_decompresses_to_zero_bytes(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, b"")
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_frame_decompresses_to_fewer_than_8_bytes(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, b"abc")
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_non_zstd_garbage(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        p.write_bytes(b"this is definitely not a zstd frame at all")
        # Genuine zstd-decode failure → the `except zstandard.ZstdError`
        # boundary specifically (pinned by message so a future refactor
        # that drops that clause is caught here).
        with pytest.raises(
            CorruptFeatureShardError, match="not a readable zstd stream"
        ):
            read_shard_header(p)

    def test_truncated_zstd_frame(self, tmp_path):
        # NOTE: on the pinned zstandard 0.25.0, truncating a frame to its
        # first half makes read(8) return 0 bytes via premature EOF (no
        # ZstdError), so this lands on the `< 8 bytes` branch. The assertion
        # is intentionally branch-agnostic: any corruption mode must raise
        # CorruptFeatureShardError naming the shard. The genuine zstd-decode
        # boundary is pinned by test_non_zstd_garbage above.
        p = tmp_path / "z_features.safetensors.zst"
        sf = struct.pack("<Q", 64) + b'{"__metadata__": {"layer": "8"}}' + b"\x00" * 8192
        comp = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(sf)
        p.write_bytes(comp[: len(comp) // 2])
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_header_size_zero(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, struct.pack("<Q", 0))
        # Pin the zero-length guard's distinguishing message: deleting the
        # `header_size == 0` guard still raises (via JSONDecodeError on empty
        # bytes) but with a different message — match="zero-length" fails then.
        with pytest.raises(CorruptFeatureShardError, match="zero-length"):
            read_shard_header(p)

    def test_frame_shorter_than_claimed_header_size(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, struct.pack("<Q", 50) + b"{}")  # claims 50, has 2
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_header_not_valid_json(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = b"NOTJSON!"
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_header_size_over_cap_is_valueerror_mentioning_header(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, struct.pack("<Q", 2**62) + b"{}")
        # Pin the cap message specifically (not any message containing
        # "header"): deleting the cap guard would let the over-large
        # allocation fail elsewhere with a different "header" message.
        with pytest.raises(
            CorruptFeatureShardError,
            match=r"refusing to allocate \d+ bytes for safetensors header",
        ):
            read_shard_header(p)
        # back-compat: still a ValueError (existing cap test relies on this)
        with pytest.raises(ValueError, match="header"):
            read_shard_header(p)

    def test_truncation_before_full_header_raises(self, tmp_path):
        # A frame that decodes the 8-byte length but ZstdError's mid-header.
        sf = struct.pack("<Q", 4096) + b'{"__metadata__": {}}' + b"\x00" * 65536
        comp = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(sf)
        p = tmp_path / "z_features.safetensors.zst"
        # Intent: keep enough to emit the 8-byte length but not the full
        # 4096-byte header, exercising a mid-header read failure. Empirically
        # on the pinned zstandard 0.25.0 this truncation makes the
        # stream_reader return 0 decoded bytes at premature EOF (no
        # ZstdError), so read_shard_header lands on the `len(size_bytes) < 8`
        # ("empty or truncated") branch — not the ZstdError arm nor "ended
        # mid-header". Still a loud CorruptFeatureShardError naming the shard;
        # this test pins the invariant "any truncation point →
        # CorruptFeatureShardError naming the shard". The genuine ZstdError
        # boundary is pinned by test_non_zstd_garbage's dedicated message
        # assertion.
        p.write_bytes(comp[: len(comp) // 3])
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)

    def test_header_size_exactly_at_cap_is_not_rejected_by_cap_guard(self, tmp_path):
        # header_size == MAX (guard is strict '>'): must NOT raise the cap
        # message; with a tiny body it raises the truncation message instead,
        # proving '==cap' passes the cap guard (off-by-one pin).
        p = tmp_path / "z_features.safetensors.zst"
        _write_compressed(p, struct.pack("<Q", MAX_SAFETENSORS_HEADER_SIZE) + b"{}")
        with pytest.raises(CorruptFeatureShardError) as ei:
            read_shard_header(p)
        assert "ended mid-header" in str(ei.value)
        assert "refusing to allocate" not in str(ei.value)  # NOT the cap message


class TestReadShardHeaderShortReadReassembly:
    """A zstandard that short-reads (future version) must not corrupt parsing."""

    def test_header_split_across_many_reads(self, tmp_path, monkeypatch):
        real = _make_real_shard(tmp_path, layer=12, segment_size=128)
        # Decompress the real file once (real decompressor) to get the exact
        # safetensors bytes — BEFORE patching.
        sf_bytes = zstandard.ZstdDecompressor().decompress(real.read_bytes())

        class _FakeDecompressor:
            def stream_reader(self, f):
                return _ChunkReader(sf_bytes, chunk=1)  # 1 byte per read()

        # feature_io references the module-global ``zstandard``; patching the
        # module object (same singleton) redirects read_shard_header's
        # ``zstandard.ZstdDecompressor()`` to the chunk-limited fake.
        monkeypatch.setattr(zstandard, "ZstdDecompressor", _FakeDecompressor)
        header = read_shard_header(real)
        assert header["__metadata__"]["layer"] == "12"
        assert header["__metadata__"]["segment_size"] == "128"
        assert TensorKey.FEATURE_IDS.value in header["__tensor_keys__"]
        assert "__metadata__" not in header["__tensor_keys__"]


class TestReadShardHeaderShape:
    def test_returns_metadata_and_tensor_keys(self, tmp_path):
        path = _make_real_shard(tmp_path, layer=8)
        header = read_shard_header(path)
        assert set(header) == {"__metadata__", "__tensor_keys__"}
        assert isinstance(header["__metadata__"], dict)
        assert isinstance(header["__tensor_keys__"], set)
        assert "__metadata__" not in header["__tensor_keys__"]
        assert header["__metadata__"]["layer"] == "8"

    def test_metadataless_header_yields_empty_metadata(self, tmp_path):
        # Valid safetensors header with tensor entries but no __metadata__:
        # read_shard_header must return {} metadata while still populating
        # __tensor_keys__ (the header.pop default arm).
        p = tmp_path / "z_features.safetensors.zst"
        header = {"feature_ids": {"dtype": "I32", "shape": [0],
                                  "data_offsets": [0, 0]}}
        hdr_bytes = json.dumps(header).encode()
        _write_compressed(p, struct.pack("<Q", len(hdr_bytes)) + hdr_bytes)
        out = read_shard_header(p)
        assert out["__metadata__"] == {}
        assert "feature_ids" in out["__tensor_keys__"]
        assert "__metadata__" not in out["__tensor_keys__"]


class TestReadShardMetadata:
    def test_valid_shard_returns_metadata_dict(self, tmp_path):
        path = _make_real_shard(tmp_path, layer=8, segment_size=64)
        meta = read_shard_metadata(path)
        assert meta["layer"] == "8"
        assert meta["segment_size"] == "64"

    def test_parity_with_full_decompress_path(self, tmp_path):
        """Streaming path == old full-decompress path for valid data."""
        path = _make_real_shard(tmp_path, layer=8, segment_size=64)
        sf_bytes = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        from_bytes = _read_safetensors_metadata_from_bytes(sf_bytes)
        assert read_shard_metadata(path) == from_bytes
        loaded = load_features(path)
        assert read_shard_metadata(path)["layer"] == str(loaded["layer"])

    def test_missing_metadata_returns_empty_dict(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = json.dumps({"feature_ids": {"dtype": "I32", "shape": [0],
                                          "data_offsets": [0, 0]}}).encode()
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        assert read_shard_metadata(p) == {}

    def test_corrupt_shard_raises(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        p.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_metadata(p)


class TestReadL0TierFromShardCorruption:
    def test_present_absent_unknown_unchanged(self, tmp_path):
        p_big = _make_real_shard(tmp_path / "a", l0_tier="big")
        p_none = _make_real_shard(tmp_path / "b")
        p_bad = _make_real_shard(tmp_path / "c", l0_tier="totally_bogus")
        assert read_l0_tier_from_shard(p_big) is L0Tier.big
        assert read_l0_tier_from_shard(p_none) is None
        with pytest.raises(ValueError, match="Unknown L0Tier"):
            read_l0_tier_from_shard(p_bad)

    def test_corrupt_shard_raises_not_silent_v1(self, tmp_path):
        """The integrity-guard regression: a corrupt shard must NOT read
        back as None (v1) — it must raise."""
        p = tmp_path / "shard_00000_features.safetensors.zst"
        p.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError):
            read_l0_tier_from_shard(p)


class TestStreamingDoesNotDecompressBody:
    def test_only_header_bytes_are_pulled(self, tmp_path, monkeypatch):
        """Deterministic early-stop proof: bytes pulled through the reader
        stay ~header-sized even when the decompressed body is many MB."""
        # ~8M int32 ids + fp16 mags -> tens of MB decompressed body.
        big_n = 4_000_000
        data = {
            "offsets": t.tensor([0, big_n], dtype=t.int64),
            "feature_ids": t.randint(0, 60000, (big_n,), dtype=t.int32),
            "feature_mags": (t.rand(big_n) + 0.1),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([10], dtype=t.int32),
            "layer": 8,
            "segment_size": 64,
        }
        path = save_features(data, tmp_path / "big")
        decompressed_size = len(
            zstandard.ZstdDecompressor().decompress(path.read_bytes())
        )
        assert decompressed_size > 4 * 1024 * 1024  # body really is large

        real_cls = zstandard.ZstdDecompressor

        class _CountingReader:
            def __init__(self, real):
                self._real = real
                self.total = 0

            def read(self, n: int = -1) -> bytes:
                b = self._real.read(n)
                self.total += len(b)
                return b

        spy: dict[str, _CountingReader | None] = {"reader": None}

        class _SpyDecompressor:
            def stream_reader(self, f):
                r = _CountingReader(real_cls().stream_reader(f))
                spy["reader"] = r
                return r

        # feature_io references the module-global ``zstandard`` (same module
        # object); patching it redirects read_shard_header's decompressor.
        monkeypatch.setattr(zstandard, "ZstdDecompressor", _SpyDecompressor)
        meta = read_shard_metadata(path)
        assert meta["layer"] == "8"
        reader = spy["reader"]
        assert reader is not None  # spy was actually installed + used
        assert reader.total < 64 * 1024  # << multi-MB body


class TestVerifyFeatureDirAccumulatesCorruptShard:
    def test_corrupt_shard_surfaced_in_single_aggregated_report(
        self, tmp_path
    ):
        """A truncated shard must appear in verify_feature_dir's single
        aggregated FeatureDirectoryMismatchError (not short-circuit it),
        alongside the missing-statistics error."""
        feature_dir = build_feature_dir(
            root=tmp_path, segment_size=64, layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool, width=SAEWidth.W_16K,
            model="gemma-2-2b", dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        corrupt = feature_dir / "shard_00000_features.safetensors.zst"
        corrupt.write_bytes(b"")  # truncated/empty -> CorruptFeatureShardError

        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        assert "shard_00000_features.safetensors.zst" in msg
        assert "Statistics file not found" in msg  # other errors still reported

    def test_loop_continues_past_corrupt_to_accumulate_later_shard(
        self, tmp_path
    ):
        """The per-shard loop must CONTINUE past a corrupt shard and still
        accumulate a *later* shard's (layer-mismatch) error — the literal
        all-problems-in-one-report contract for coexisting problems. The
        single-shard test only proves this indirectly (missing-stats is
        appended before the loop); this pins loop continuation itself."""
        feature_dir = build_feature_dir(
            root=tmp_path, segment_size=64, layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool, width=SAEWidth.W_16K,
            model="gemma-2-2b", dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        # shard_00000: corrupt (empty) -> caught + accumulated, loop continues.
        (feature_dir / "shard_00000_features.safetensors.zst").write_bytes(b"")
        # shard_00001: VALID but layer=12 mismatches the dir's expected layer=8
        # -> only reachable if the loop did NOT short-circuit on shard_00000.
        save_features(
            {
                "offsets": t.tensor([0, 2], dtype=t.int64),
                "feature_ids": t.tensor([0, 1], dtype=t.int32),
                "feature_mags": t.tensor([1.0, 2.0], dtype=t.float32),
                "doc_ids": t.tensor([0], dtype=t.int32),
                "segment_indices": t.tensor([0], dtype=t.int32),
                "segment_positions": t.tensor([0], dtype=t.int32),
                "segment_char_starts": t.tensor([0], dtype=t.int32),
                "segment_char_ends": t.tensor([10], dtype=t.int32),
                "layer": 12,
                "segment_size": 64,
            },
            feature_dir / "shard_00001",
        )
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        assert "shard_00000_features.safetensors.zst" in msg  # corrupt accumulated
        assert "shard_00001" in msg and "layer=12" in msg  # later shard still checked


class TestL0TierGuardIntegration:
    def test_same_tier_passes_cross_tier_raises(self, tmp_path):
        _make_real_shard(tmp_path, l0_tier="medium")
        # Same tier into a populated dir: must not raise.
        assert_l0_tier_compatible(tmp_path, L0Tier.medium)
        with pytest.raises(FeatureDirectoryMismatchError, match="l0_tier"):
            assert_l0_tier_compatible(tmp_path, L0Tier.big)

    def test_corrupt_shard_halts_assert_l0_tier_compatible(self, tmp_path):
        _make_real_shard(tmp_path, l0_tier="medium")
        corrupt = tmp_path / "shard_99999_features.safetensors.zst"
        corrupt.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError, match=str(corrupt)):
            assert_l0_tier_compatible(tmp_path, L0Tier.medium)

    def test_corrupt_shard_halts_assert_uniform_l0_tier(self, tmp_path):
        _make_real_shard(tmp_path, l0_tier="medium")
        corrupt = tmp_path / "shard_99999_features.safetensors.zst"
        corrupt.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError, match=str(corrupt)):
            assert_uniform_l0_tier(glob_features(tmp_path))

    def test_uniform_tier_dir_returns_tier(self, tmp_path):
        _make_real_shard(tmp_path / "a", l0_tier="big")
        _make_real_shard(tmp_path / "b", l0_tier="big")
        shards = glob_features(tmp_path / "a") + glob_features(tmp_path / "b")
        assert assert_uniform_l0_tier(shards) is L0Tier.big

    def test_verify_segment_size_corrupt_shard_raises_typed(self, tmp_path):
        seg_dir = tmp_path / "s64" / "l8" / "a_res" / "p_mean_pool" / "f16k" \
            / "m_gemma-2-2b" / "base"
        seg_dir.mkdir(parents=True, exist_ok=True)
        corrupt = seg_dir / "shard_00000_features.safetensors.zst"
        corrupt.write_bytes(b"")
        with pytest.raises(CorruptFeatureShardError, match=str(corrupt)):
            verify_segment_size(corrupt, expected_segment_size=64)


class TestNonDictMetadataIsCorrupt:
    """A valid-JSON safetensors header whose __metadata__ is not a dict
    must raise CorruptFeatureShardError (the ShardHeader contract says
    __metadata__ is a dict), not yield a bare AttributeError downstream."""

    @pytest.mark.parametrize(
        "bad", ['"a string"', "[1, 2]", "null", "42", "3.14", "true", "false"]
    )
    def test_non_dict_metadata_raises(self, tmp_path, bad):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = (
            '{"__metadata__": ' + bad + ', '
            '"feature_ids": {"dtype": "I32", "shape": [0], '
            '"data_offsets": [0, 0]}}'
        ).encode()
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_metadata(p)

    def test_missing_metadata_still_ok(self, tmp_path):
        # Regression guard: absent __metadata__ -> {} (a dict) must still pass.
        p = tmp_path / "z_features.safetensors.zst"
        hdr = b'{"feature_ids": {"dtype": "I32", "shape": [0], "data_offsets": [0, 0]}}'
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        assert read_shard_metadata(p) == {}

    def test_dict_metadata_still_ok(self, tmp_path):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = b'{"__metadata__": {"layer": "8"}, "feature_ids": {"dtype": "I32", "shape": [0], "data_offsets": [0, 0]}}'
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        assert read_shard_metadata(p) == {"layer": "8"}


class TestNonStrMetadataValueIsCorrupt:
    """A safetensors __metadata__ that is a dict but has a non-str VALUE
    (writer-impossible — safetensors.save enforces str→str) must raise
    CorruptFeatureShardError naming the shard, NOT be silently coerced to
    str. Consistent with the module's loud-failure-on-corruption contract."""

    @pytest.mark.parametrize(
        "meta_json",
        [
            '{"layer": 8}',                       # int value
            '{"layer": "8", "ctx": true}',        # bool value
            '{"layer": [1, 2]}',                  # list value
            '{"layer": null}',                    # null value
            '{"layer": 3.5}',                     # float value
            '{"nested": {"a": "b"}}',             # dict value
        ],
    )
    def test_non_str_metadata_value_raises(self, tmp_path, meta_json):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = (
            '{"__metadata__": ' + meta_json + ', '
            '"feature_ids": {"dtype": "I32", "shape": [0], '
            '"data_offsets": [0, 0]}}'
        ).encode()
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_metadata(p)

    def test_non_str_metadata_value_halts_l0_guard(self, tmp_path):
        p = tmp_path / "shard_00000_features.safetensors.zst"
        hdr = (
            b'{"__metadata__": {"layer": 8}, '
            b'"feature_ids": {"dtype": "I32", "shape": [0], '
            b'"data_offsets": [0, 0]}}'
        )
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_l0_tier_from_shard(p)

    def test_valid_str_metadata_unchanged(self, tmp_path):
        # True no-op guard: a normal str→str shard still round-trips.
        path = _make_real_shard(tmp_path, layer=8, segment_size=64)
        meta = read_shard_metadata(path)
        assert meta["layer"] == "8"
        assert meta["segment_size"] == "64"


class TestNonObjectTopLevelHeaderIsCorrupt:
    """A valid-JSON but non-OBJECT top-level safetensors header (list/int/
    null/str/bool/float) must raise CorruptFeatureShardError naming the
    shard — NOT a bare TypeError/AttributeError from header.pop(...)."""

    @pytest.mark.parametrize(
        "whole_header", ["[]", "[1, 2, 3]", "5", "3.14", "null", '"a string"', "true", "false"]
    )
    def test_non_object_header_raises(self, tmp_path, whole_header):
        p = tmp_path / "z_features.safetensors.zst"
        hdr = whole_header.encode()
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_header(p)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_shard_metadata(p)

    def test_non_object_header_halts_l0_guard(self, tmp_path):
        # End-to-end: must not escape as a bare TypeError past read_l0_tier_from_shard.
        p = tmp_path / "shard_00000_features.safetensors.zst"
        hdr = b"[1, 2, 3]"
        _write_compressed(p, struct.pack("<Q", len(hdr)) + hdr)
        with pytest.raises(CorruptFeatureShardError, match=str(p)):
            read_l0_tier_from_shard(p)


class TestVerifyFeatureDirDoubleSurface:
    def test_corrupt_shard_surfaces_twice(self, tmp_path):
        """feature_paths.py documents that a corrupt shard surfaces TWICE
        (per-shard loop + assert_uniform_l0_tier re-raise) and 'do NOT
        de-dupe'. Pin it so a future de-dupe refactor fails loudly."""
        feature_dir = build_feature_dir(
            root=tmp_path, segment_size=64, layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool, width=SAEWidth.W_16K,
            model="gemma-2-2b", dataset_phase=DatasetPhase.BASE,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "shard_00000_features.safetensors.zst").write_bytes(b"")
        with pytest.raises(FeatureDirectoryMismatchError) as ei:
            verify_feature_dir(feature_dir)
        msg = str(ei.value)
        # The empty shard surfaces from TWO distinct, independent checks:
        # (1) verify_feature_dir's per-shard loop (`Shard <name>: ...`) and
        # (2) the shared assert_uniform_l0_tier re-raise. Its distinctive
        # "is empty or truncated" phrase therefore appears EXACTLY twice —
        # once per surface. A future de-dupe / collapse of either surface
        # drops one occurrence, so this exact-count pins the documented
        # "loud and doubled; do NOT de-dupe" contract (mutation-killing).
        assert msg.count("is empty or truncated") == 2
        assert "shard_00000_features.safetensors.zst" in msg


class TestAssertL0TierCompatibleDirectionalKinds:
    def test_v1_dir_v2_write_kind(self, tmp_path):
        _make_real_shard(tmp_path)  # no l0_tier -> v1
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v1 \(no l0_tier\) → v2"):
            assert_l0_tier_compatible(tmp_path, L0Tier.medium)

    def test_v2_dir_v1_write_kind(self, tmp_path):
        _make_real_shard(tmp_path, l0_tier="medium")
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v2 → v1 \(no l0_tier\)"):
            assert_l0_tier_compatible(tmp_path, None)
