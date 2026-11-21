"""Tests for steering.feature_io -- compressed feature shard I/O."""

import pytest
import torch as t
from safetensors.torch import save as safetensors_save_bytes

import steering.feature_io as feature_io
from errors import EmptyExtractionSegmentError
from steering.feature_io import (
    COMPRESSED_EXT,
    MetadataKey,
    _delta_decode,
    _delta_encode,
    _optimize_dtype,
    _sort_within_segments,
    decode_feature_id_deltas,
    decode_feature_safetensors_bytes,
    decode_percentile_feature_safetensors_bytes,
    feature_path,
    feature_path_exists,
    glob_features,
    load_features,
    parse_percentile_feature_safetensors_bytes,
    save_features,
    shard_stem_from_path,
)
from steering.shard_validation import assert_csr_invariants, validate_shard


def _make_shard_dict(
    n_segments: int = 3,
    feats_per_seg: int = 5,
    n_docs: int = 2,
    max_feature_id: int = 16383,
) -> dict:
    """Build a minimal valid shard dict."""
    n_features = n_segments * feats_per_seg
    offsets = [i * feats_per_seg for i in range(n_segments + 1)]

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        segs_for_doc = n_segments // n_docs + (1 if d < n_segments % n_docs else 0)
        for s in range(segs_for_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 512)
            seg_char_ends.append((s + 1) * 512)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.randint(0, max_feature_id + 1, (n_features,), dtype=t.int32),
        "feature_mags": t.rand(n_features, dtype=t.float32).abs() + 0.01,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def _reference_delta_decode(delta_ids: t.Tensor, offsets: t.Tensor) -> t.Tensor:
    """Original scalar-boundary decoder retained as a parity oracle."""
    ids = delta_ids.clone()
    for segment_index in range(offsets.numel() - 1):
        start = offsets[segment_index].item()
        end = offsets[segment_index + 1].item()
        if end - start >= 2:
            ids[start:end] = delta_ids[start:end].cumsum(0)
    return ids


@pytest.mark.parametrize(
    ("stored_id_dtype", "restored_id_dtype"),
    [
        (t.uint16, t.int32),
        (t.int16, t.int32),
        (t.int32, t.int32),
        (t.int64, t.int64),
    ],
)
def test_percentile_decoder_matches_generic_values_without_widening_magnitudes(
    stored_id_dtype: t.dtype,
    restored_id_dtype: t.dtype,
) -> None:
    large_delta = 30_000 if stored_id_dtype == t.int16 else 65_000
    payload = safetensors_save_bytes(
        {
            "offsets": t.tensor([0, 3, 6], dtype=t.int32),
            "feature_ids": t.tensor(
                [7, 4, 9, large_delta, 100, 200],
                dtype=stored_id_dtype,
            ),
            "feature_mags": t.tensor(
                [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
                dtype=t.float16,
            ),
            "unused": t.tensor([123], dtype=t.int32),
        },
        metadata={"segment_size": "128", "n_segments": "2"},
    )

    encoded = parse_percentile_feature_safetensors_bytes(payload)
    narrow = decode_percentile_feature_safetensors_bytes(payload)
    generic = decode_feature_safetensors_bytes(payload)

    assert encoded.feature_id_deltas.dtype == stored_id_dtype
    assert encoded.feature_id_deltas.tolist() == [
        7,
        4,
        9,
        large_delta,
        100,
        200,
    ]
    assert encoded.feature_magnitudes.dtype == t.float16
    assert encoded.offsets.dtype == t.int64
    assert encoded.metadata["segment_size"] == 128
    assert encoded.metadata["n_segments"] == 2
    assert encoded.n_values == 6
    assert narrow["feature_ids"].dtype == restored_id_dtype
    assert t.equal(narrow["feature_ids"], generic["feature_ids"])
    assert narrow["feature_mags"].dtype == t.float16
    assert t.equal(narrow["feature_mags"].float(), generic["feature_mags"])
    assert narrow["offsets"].dtype == t.int64
    assert t.equal(narrow["offsets"], generic["offsets"])
    assert narrow["segment_size"] == 128
    assert narrow["n_segments"] == 2
    assert "unused" not in narrow


@pytest.mark.parametrize("stored_id_dtype", [t.uint16, t.int16, t.int32, t.int64])
def test_generic_decoder_owns_delta_decode_buffer(
    stored_id_dtype: t.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_id = 30_000 if stored_id_dtype == t.int16 else 65_000
    deltas = t.tensor([7, 4, 9, first_id, 2000, 2000], dtype=stored_id_dtype)
    payload = safetensors_save_bytes(
        {
            "feature_ids": deltas,
            "offsets": t.tensor([0, 3, 6], dtype=t.int32),
            "feature_mags": t.ones(6, dtype=t.float16),
        },
        metadata={"segment_size": "128", "n_segments": "2"},
    )
    payload_before = bytearray(payload)
    parsed_ids: list[t.Tensor] = []
    upcast_ids: list[t.Tensor] = []
    original_load, original_upcast = feature_io.safetensors_load, feature_io._upcast

    def load(data: bytes) -> dict[str, t.Tensor]:
        tensors = original_load(data)
        parsed_ids.append(tensors["feature_ids"])
        return tensors

    def upcast(key: str, tensor: t.Tensor) -> t.Tensor:
        restored = original_upcast(key, tensor)
        if key == "feature_ids":
            upcast_ids.append(restored)
        return restored

    monkeypatch.setattr(feature_io, "safetensors_load", load)
    monkeypatch.setattr(feature_io, "_upcast", upcast)
    first = decode_feature_safetensors_bytes(payload)
    second = decode_feature_safetensors_bytes(payload)
    widened = stored_id_dtype in (t.uint16, t.int16)
    restored_dtype = t.int32 if widened else stored_id_dtype
    expected = t.tensor(
        [7, 11, 20, first_id, first_id + 2000, first_id + 4000], dtype=restored_dtype
    )
    for result, raw, restored in zip(
        (first, second), parsed_ids, upcast_ids, strict=True
    ):
        assert result["feature_ids"].dtype == restored_dtype
        assert t.equal(result["feature_ids"], expected)
        assert result["feature_ids"].data_ptr() != raw.data_ptr()
        assert (result["feature_ids"] is restored) == widened
        assert t.equal(raw, deltas)
        assert result["segment_size"] == 128
    assert first["feature_ids"].data_ptr() != second["feature_ids"].data_ptr()
    first["feature_ids"][0] = 999
    assert t.equal(second["feature_ids"], expected)
    assert all(t.equal(raw, deltas) for raw in parsed_ids)
    assert payload == payload_before


@pytest.mark.parametrize("stored_id_dtype", [t.uint16, t.int32, t.int64])
def test_chunk_delta_decode_does_not_mutate_storage_tensor(
    stored_id_dtype: t.dtype,
) -> None:
    deltas = t.tensor([7, 4, 9, 2, 3], dtype=stored_id_dtype)
    original = deltas.clone()

    observed = decode_feature_id_deltas(
        deltas,
        t.tensor([0, 3, 5], dtype=t.int64),
    )

    expected_dtype = t.int32 if stored_id_dtype == t.uint16 else stored_id_dtype
    assert observed.dtype == expected_dtype
    assert observed.tolist() == [7, 11, 20, 2, 5]
    assert t.equal(deltas, original)


@pytest.mark.parametrize(
    "decoder",
    [
        decode_percentile_feature_safetensors_bytes,
        parse_percentile_feature_safetensors_bytes,
    ],
)
@pytest.mark.parametrize("missing", ["offsets", "feature_ids", "feature_mags"])
def test_percentile_decoder_requires_complete_lookup_inputs(
    missing: str,
    decoder,
) -> None:
    tensors = {
        "offsets": t.tensor([0, 1], dtype=t.int32),
        "feature_ids": t.tensor([3], dtype=t.uint16),
        "feature_mags": t.tensor([0.5], dtype=t.float16),
    }
    del tensors[missing]

    with pytest.raises(ValueError, match=rf"required tensor.*{missing}"):
        decoder(safetensors_save_bytes(tensors))


class TestCompressedRoundtrip:
    """save_features -> load_features should preserve all data through compression."""

    def test_roundtrip_tensors(self, tmp_path):
        original = _make_shard_dict()
        out = save_features(original, tmp_path / "shard_00000_features")
        loaded = load_features(out)

        for key in (
            "offsets",
            "feature_ids",
            "feature_mags",
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "doc_token_counts",
        ):
            assert key in loaded, f"Missing key: {key}"
            assert loaded[key].dtype == original[key].dtype, (
                f"{key}: expected {original[key].dtype}, got {loaded[key].dtype}"
            )
            if key in ("feature_ids", "feature_mags"):
                # IDs are re-sorted ascending within segments, mags co-sorted.
                # Compare per-segment sorted multisets.
                offsets = original["offsets"]
                n_segs = offsets.numel() - 1
                for seg in range(n_segs):
                    s, e = offsets[seg].item(), offsets[seg + 1].item()
                    orig_sorted = original[key][s:e].sort().values
                    load_sorted = loaded[key][s:e].sort().values
                    if key == "feature_mags":
                        assert t.allclose(orig_sorted, load_sorted, atol=1e-3), (
                            f"{key} segment {seg}: values differ beyond tolerance"
                        )
                    else:
                        assert t.equal(orig_sorted, load_sorted), (
                            f"{key} segment {seg}: values differ"
                        )
            else:
                assert t.equal(loaded[key], original[key]), f"{key}: values differ"

    def test_roundtrip_metadata(self, tmp_path):
        original = _make_shard_dict()
        out = save_features(original, tmp_path / "shard_00000_features")
        loaded = load_features(out)

        for key in (
            "n_docs",
            "n_segments",
            "layer",
            "context_window",
            "pooling",
            "segment_mode",
            "segment_size",
        ):
            assert key in loaded, f"Missing metadata key: {key}"
            assert loaded[key] == original[key], (
                f"{key}: expected {original[key]}, got {loaded[key]}"
            )

    def test_output_extension_is_zst(self, tmp_path):
        original = _make_shard_dict()
        out = save_features(original, tmp_path / "shard_00000_features.pt")
        assert out.name.endswith(COMPRESSED_EXT)
        assert out.exists()

    def test_roundtrip_empty_features(self, tmp_path):
        """A whole-shard-empty shard (n_segments==0, no segments emitted) round-trips.

        Under the issue #522 invariant a shard with n_segments>0 cannot have an
        empty segment; the legitimate 'zero features' case is n_segments==0.
        """
        data = _make_shard_dict(n_segments=0, feats_per_seg=0, n_docs=1)
        data["feature_ids"] = t.tensor([], dtype=t.int32)
        data["feature_mags"] = t.tensor([], dtype=t.float32)
        data["offsets"] = t.tensor([0], dtype=t.int64)

        out = save_features(data, tmp_path / "empty")
        loaded = load_features(out)
        assert loaded["feature_ids"].numel() == 0
        assert loaded["feature_mags"].numel() == 0

    def test_roundtrip_single_feature_per_segment(self, tmp_path):
        """Single feature per segment means no delta encoding occurs."""
        data = _make_shard_dict(n_segments=3, feats_per_seg=1, n_docs=1)
        out = save_features(data, tmp_path / "single_feat")
        loaded = load_features(out)

        # Single features are already trivially sorted
        assert t.equal(loaded["feature_ids"], data["feature_ids"])

    def test_roundtrip_large_shard(self, tmp_path):
        """A larger shard with many segments and features."""
        data = _make_shard_dict(n_segments=100, feats_per_seg=50, n_docs=10)
        out = save_features(data, tmp_path / "large")
        loaded = load_features(out)

        assert loaded["feature_ids"].shape == data["feature_ids"].shape
        assert loaded["feature_mags"].shape == data["feature_mags"].shape
        assert loaded["offsets"].shape == data["offsets"].shape

        # Verify all feature IDs match (as sorted sets per segment)
        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            orig_sorted = data["feature_ids"][s:e].sort().values
            load_sorted = loaded["feature_ids"][s:e].sort().values
            assert t.equal(orig_sorted, load_sorted)

    def test_roundtrip_max_feature_id_65535(self, tmp_path):
        """Feature IDs at the uint16 boundary should roundtrip."""
        data = _make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=1)
        data["feature_ids"] = t.tensor(
            [0, 32767, 65535, 100, 200, 65535], dtype=t.int32
        )

        out = save_features(data, tmp_path / "max_id")
        loaded = load_features(out)

        # Check sets per segment match
        offsets = data["offsets"]
        for seg in range(2):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            orig_sorted = data["feature_ids"][s:e].sort().values
            load_sorted = loaded["feature_ids"][s:e].sort().values
            assert t.equal(orig_sorted, load_sorted)

    def test_roundtrip_char_boundary_fields(self, tmp_path):
        """segment_char_starts and segment_char_ends should roundtrip."""
        data = _make_shard_dict()
        out = save_features(data, tmp_path / "char_bounds")
        loaded = load_features(out)

        assert "segment_char_starts" in loaded
        assert "segment_char_ends" in loaded
        assert t.equal(loaded["segment_char_starts"], data["segment_char_starts"])
        assert t.equal(loaded["segment_char_ends"], data["segment_char_ends"])

    def test_compressed_smaller_than_uncompressed(self, tmp_path):
        """Compressed file should be smaller than raw safetensors bytes."""
        data = _make_shard_dict(n_segments=50, feats_per_seg=100, n_docs=5)

        # Get uncompressed safetensors size
        tensors = {}
        for key, value in data.items():
            if isinstance(value, t.Tensor):
                tensors[key] = value
        uncompressed_bytes = safetensors_save_bytes(tensors)
        uncompressed_size = len(uncompressed_bytes)

        # Save compressed
        out = save_features(data, tmp_path / "compare")
        compressed_size = out.stat().st_size

        assert compressed_size < uncompressed_size, (
            f"Compressed ({compressed_size}) should be smaller than "
            f"uncompressed ({uncompressed_size})"
        )

    def test_ids_sorted_ascending_after_load(self, tmp_path):
        """After roundtrip, feature_ids should be sorted ascending within segments."""
        data = _make_shard_dict()
        # Ensure IDs are NOT sorted (descending)
        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            if e - s > 1:
                data["feature_ids"][s:e] = (
                    data["feature_ids"][s:e].sort(descending=True).values
                )

        out = save_features(data, tmp_path / "sorted")
        loaded = load_features(out)

        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            if e - s > 1:
                seg_ids = loaded["feature_ids"][s:e]
                assert t.all(seg_ids[1:] >= seg_ids[:-1]), (
                    f"Segment {seg}: IDs not sorted ascending"
                )

    def test_roundtrip_already_sorted_feature_ids(self, tmp_path):
        """Sort+delta should be identity for already-sorted input."""
        data = _make_shard_dict(n_segments=3, feats_per_seg=4, n_docs=1)
        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        # Make feature_ids already sorted ascending within each segment.
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            data["feature_ids"][s:e] = t.arange(
                seg * 100, seg * 100 + (e - s), dtype=t.int32
            )

        out = save_features(data, tmp_path / "presorted")
        loaded = load_features(out)

        # Exact equality, not just set equality -- sort+delta must be identity.
        assert t.equal(loaded["feature_ids"], data["feature_ids"])

    def test_roundtrip_duplicate_feature_ids(self, tmp_path):
        """Duplicate IDs within a segment (delta of 0) should roundtrip correctly."""
        data = _make_shard_dict(n_segments=2, feats_per_seg=4, n_docs=1)
        # Segment 0: duplicates
        data["feature_ids"][:4] = t.tensor([10, 10, 20, 30], dtype=t.int32)
        # Segment 1: all identical
        data["feature_ids"][4:8] = t.tensor([50, 50, 50, 50], dtype=t.int32)

        out = save_features(data, tmp_path / "dupes")
        loaded = load_features(out)

        offsets = data["offsets"]
        for seg in range(2):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            # Already sorted, so exact equality should hold.
            assert t.equal(loaded["feature_ids"][s:e], data["feature_ids"][s:e]), (
                f"Segment {seg}: duplicate feature IDs not preserved"
            )


class TestDeltaEncoding:
    """Tests for _delta_encode and _delta_decode."""

    def test_delta_encode_decode_roundtrip(self):
        offsets = t.tensor([0, 4, 7], dtype=t.int64)
        sorted_ids = t.tensor([10, 20, 30, 40, 100, 200, 300], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_delta_encode_single_element(self):
        offsets = t.tensor([0, 1], dtype=t.int64)
        sorted_ids = t.tensor([42], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        assert t.equal(encoded, sorted_ids)  # No change for single element

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_delta_encode_empty_segment(self):
        offsets = t.tensor([0, 0, 3], dtype=t.int64)
        sorted_ids = t.tensor([10, 20, 30], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_delta_encode_consecutive_ids(self):
        offsets = t.tensor([0, 5], dtype=t.int64)
        sorted_ids = t.tensor([100, 101, 102, 103, 104], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected = t.tensor([100, 1, 1, 1, 1], dtype=t.int32)
        assert t.equal(encoded, expected)

    def test_delta_encode_with_gaps(self):
        offsets = t.tensor([0, 4], dtype=t.int64)
        sorted_ids = t.tensor([10, 50, 100, 60000], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected = t.tensor([10, 40, 50, 59900], dtype=t.int32)
        assert t.equal(encoded, expected)

    def test_delta_decode_cumsum_correctness(self):
        offsets = t.tensor([0, 3, 6], dtype=t.int64)
        deltas = t.tensor([5, 10, 15, 100, 50, 25], dtype=t.int32)

        decoded = _delta_decode(deltas, offsets)
        expected = t.tensor([5, 15, 30, 100, 150, 175], dtype=t.int32)
        assert t.equal(decoded, expected)

    @pytest.mark.parametrize("offset_dtype", [t.int32, t.int64])
    @pytest.mark.parametrize("id_dtype", [t.int16, t.int32, t.int64])
    def test_delta_decode_matches_reference_for_supported_integer_dtypes(
        self,
        offset_dtype,
        id_dtype,
    ):
        offsets = t.tensor([0, 0, 1, 2, 7, 7, 12], dtype=offset_dtype)
        deltas = t.tensor(
            [31, 7, 100, 9, 11, 0, 13, 6, 4, 3, 2, 1],
            dtype=id_dtype,
        )

        expected = _reference_delta_decode(deltas, offsets)
        actual = _delta_decode(deltas, offsets)

        assert actual.dtype == deltas.dtype
        assert t.equal(actual, expected)

    def test_delta_decode_matches_reference_on_random_segment_layouts(self):
        generator = t.Generator().manual_seed(20260812)

        for _ in range(100):
            lengths = t.randint(0, 65, (128,), generator=generator, dtype=t.int64)
            offsets = t.cat((t.zeros(1, dtype=t.int64), lengths.cumsum(0)))
            deltas = t.randint(
                0,
                256,
                (int(offsets[-1].item()),),
                generator=generator,
                dtype=t.int32,
            )

            expected = _reference_delta_decode(deltas, offsets)
            actual = _delta_decode(deltas, offsets)

            assert t.equal(actual, expected)

    @pytest.mark.parametrize(
        ("deltas", "offsets"),
        [
            (t.empty(0, dtype=t.int32), t.empty(0, dtype=t.int64)),
            (t.empty(0, dtype=t.int32), t.tensor([0], dtype=t.int64)),
            (t.empty(0, dtype=t.int32), t.tensor([0, 0, 0], dtype=t.int64)),
        ],
    )
    def test_delta_decode_preserves_empty_inputs(self, deltas, offsets):
        actual = _delta_decode(deltas, offsets)

        assert actual.dtype == deltas.dtype
        assert actual.shape == deltas.shape
        assert actual is not deltas

    def test_delta_decode_does_not_mutate_inputs(self):
        offsets = t.tensor([0, 3, 4, 8], dtype=t.int64)
        deltas = t.tensor([5, 2, 3, 17, 9, 0, 4, 1], dtype=t.int32)
        offsets_before = offsets.clone()
        deltas_before = deltas.clone()

        decoded = _delta_decode(deltas, offsets)

        assert t.equal(offsets, offsets_before)
        assert t.equal(deltas, deltas_before)
        assert decoded.data_ptr() != deltas.data_ptr()


class TestSortWithinSegments:
    """Tests for _sort_within_segments."""

    def test_sort_reorders_by_id(self):
        offsets = t.tensor([0, 4], dtype=t.int64)
        ids = t.tensor([30, 10, 40, 20], dtype=t.int32)
        mags = t.tensor([3.0, 1.0, 4.0, 2.0], dtype=t.float32)

        sorted_ids, _ = _sort_within_segments(ids, mags, offsets)
        expected = t.tensor([10, 20, 30, 40], dtype=t.int32)
        assert t.equal(sorted_ids, expected)

    def test_sort_cosorts_magnitudes(self):
        offsets = t.tensor([0, 4], dtype=t.int64)
        ids = t.tensor([30, 10, 40, 20], dtype=t.int32)
        mags = t.tensor([3.0, 1.0, 4.0, 2.0], dtype=t.float32)

        _, sorted_mags = _sort_within_segments(ids, mags, offsets)
        expected = t.tensor([1.0, 2.0, 3.0, 4.0], dtype=t.float32)
        assert t.equal(sorted_mags, expected)

    def test_sort_preserves_segment_boundaries(self):
        offsets = t.tensor([0, 3, 6], dtype=t.int64)
        ids = t.tensor([30, 10, 20, 60, 40, 50], dtype=t.int32)
        mags = t.tensor([3.0, 1.0, 2.0, 6.0, 4.0, 5.0], dtype=t.float32)

        sorted_ids, sorted_mags = _sort_within_segments(ids, mags, offsets)

        # Segment 0: [10, 20, 30]
        assert t.equal(sorted_ids[0:3], t.tensor([10, 20, 30], dtype=t.int32))
        assert t.equal(sorted_mags[0:3], t.tensor([1.0, 2.0, 3.0]))

        # Segment 1: [40, 50, 60]
        assert t.equal(sorted_ids[3:6], t.tensor([40, 50, 60], dtype=t.int32))
        assert t.equal(sorted_mags[3:6], t.tensor([4.0, 5.0, 6.0]))

    def test_sort_handles_empty_segments(self):
        offsets = t.tensor([0, 0, 3], dtype=t.int64)
        ids = t.tensor([30, 10, 20], dtype=t.int32)
        mags = t.tensor([3.0, 1.0, 2.0], dtype=t.float32)

        sorted_ids, sorted_mags = _sort_within_segments(ids, mags, offsets)
        assert t.equal(sorted_ids, t.tensor([10, 20, 30], dtype=t.int32))
        assert t.equal(sorted_mags, t.tensor([1.0, 2.0, 3.0]))

    def test_sort_already_sorted(self):
        offsets = t.tensor([0, 3], dtype=t.int64)
        ids = t.tensor([10, 20, 30], dtype=t.int32)
        mags = t.tensor([1.0, 2.0, 3.0], dtype=t.float32)

        sorted_ids, sorted_mags = _sort_within_segments(ids, mags, offsets)
        assert t.equal(sorted_ids, ids)
        assert t.equal(sorted_mags, mags)


class TestDtypeOptimization:
    """Verify that save_features downcasts correctly inside compressed format."""

    def test_feature_ids_stored_as_uint16(self, tmp_path):
        data = _make_shard_dict(max_feature_id=16383)
        result = _optimize_dtype("feature_ids", data["feature_ids"])
        assert result.dtype == t.uint16

    def test_feature_ids_stays_int32_for_large_ids(self, tmp_path):
        ids = t.tensor([0, 100000], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.int32

    def test_feature_mags_stored_as_float16(self, tmp_path):
        mags = t.rand(10, dtype=t.float32)
        result = _optimize_dtype("feature_mags", mags)
        assert result.dtype == t.float16

    def test_offsets_stored_as_int32(self, tmp_path):
        offsets = t.tensor([0, 5, 10], dtype=t.int64)
        result = _optimize_dtype("offsets", offsets)
        assert result.dtype == t.int32

    def test_offsets_stays_int64_for_large_values(self, tmp_path):
        offsets = t.tensor([0, 3_000_000_000], dtype=t.int64)
        result = _optimize_dtype("offsets", offsets)
        assert result.dtype == t.int64


class TestGlobFeatures:
    """glob_features should find .safetensors.zst files."""

    def test_finds_compressed_files(self, tmp_path):
        data = _make_shard_dict()
        save_features(data, tmp_path / "shard_00000_features")
        save_features(data, tmp_path / "shard_00001_features")
        found = glob_features(tmp_path)
        assert len(found) == 2
        assert all(p.name.endswith(COMPRESSED_EXT) for p in found)

    def test_ignores_plain_safetensors(self, tmp_path):
        """Plain .safetensors files are not found by glob_features."""
        fake = tmp_path / "shard_00000_features.safetensors"
        fake.write_bytes(b"fake")
        found = glob_features(tmp_path)
        assert len(found) == 0

    def test_ignores_pt_files(self, tmp_path):
        """Plain .pt files are not found by glob_features."""
        fake = tmp_path / "shard_00000_features.pt"
        fake.write_bytes(b"fake")
        found = glob_features(tmp_path)
        assert len(found) == 0

    def test_empty_dir(self, tmp_path):
        assert glob_features(tmp_path) == []

    def test_sorted_output(self, tmp_path):
        data = _make_shard_dict()
        save_features(data, tmp_path / "shard_00002_features")
        save_features(data, tmp_path / "shard_00000_features")
        save_features(data, tmp_path / "shard_00001_features")
        found = glob_features(tmp_path)
        assert len(found) == 3
        names = [p.name for p in found]
        assert names == sorted(names)


class TestHelpers:
    def test_feature_path(self, tmp_path):
        p = feature_path(tmp_path, "shard_00005")
        assert p.name == "shard_00005_features.safetensors.zst"

    def test_feature_path_exists_compressed(self, tmp_path):
        data = _make_shard_dict()
        save_features(data, tmp_path / "shard_00000_features")
        assert feature_path_exists(tmp_path, "shard_00000")
        assert not feature_path_exists(tmp_path, "shard_00001")

    def test_feature_path_exists_ignores_pt(self, tmp_path):
        """Plain .pt files are not detected by feature_path_exists."""
        fake = tmp_path / "shard_00000_features.pt"
        fake.write_bytes(b"fake")
        assert not feature_path_exists(tmp_path, "shard_00000")

    def test_shard_stem_from_path_compressed(self):
        assert (
            shard_stem_from_path("shard_00001_features.safetensors.zst")
            == "shard_00001"
        )

    def test_shard_stem_from_path_with_directory(self):
        assert (
            shard_stem_from_path("/a/b/shard_00001_features.safetensors.zst")
            == "shard_00001"
        )

    def test_load_unknown_extension_raises(self, tmp_path):
        fake = tmp_path / "shard.npz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unknown feature file format"):
            load_features(fake)

    def test_load_plain_safetensors_raises(self, tmp_path):
        """Plain .safetensors files are no longer supported for loading."""
        fake = tmp_path / "shard_00000_features.safetensors"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unknown feature file format"):
            load_features(fake)


class TestAssertCsrInvariants:
    """Tests for assert_csr_invariants with ID-ascending sort order."""

    def test_accepts_ids_sorted_ascending(self):
        """IDs sorted ascending within each segment should pass."""
        ids = [
            t.tensor([10, 20, 30], dtype=t.int32),
            t.tensor([5, 15, 25], dtype=t.int32),
        ]
        mags = [
            t.tensor([0.3, 0.1, 0.2], dtype=t.float32),
            t.tensor([0.5, 0.4, 0.6], dtype=t.float32),
        ]
        assert_csr_invariants(ids, mags)

    def test_rejects_ids_sorted_descending(self):
        """IDs sorted descending should fail the ascending check."""
        ids = [t.tensor([30, 20, 10], dtype=t.int32)]
        mags = [t.tensor([0.3, 0.2, 0.1], dtype=t.float32)]
        with pytest.raises(AssertionError, match="not sorted ascending"):
            assert_csr_invariants(ids, mags)

    def test_rejects_unsorted_ids(self):
        """Unsorted IDs should fail."""
        ids = [t.tensor([10, 30, 20], dtype=t.int32)]
        mags = [t.tensor([0.1, 0.3, 0.2], dtype=t.float32)]
        with pytest.raises(AssertionError, match="not sorted ascending"):
            assert_csr_invariants(ids, mags)

    def test_accepts_equal_adjacent_ids(self):
        """Equal adjacent IDs (duplicates) are allowed (<=)."""
        ids = [t.tensor([10, 10, 20, 20], dtype=t.int32)]
        mags = [t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32)]
        assert_csr_invariants(ids, mags)

    def test_accepts_single_element(self):
        """Single-element segments are trivially sorted."""
        ids = [t.tensor([42], dtype=t.int32)]
        mags = [t.tensor([1.0], dtype=t.float32)]
        assert_csr_invariants(ids, mags)

    def test_rejects_empty_segment(self):
        """Empty base segments must be rejected (issue #522)."""
        ids = [t.tensor([], dtype=t.int32)]
        mags = [t.tensor([], dtype=t.float32)]
        with pytest.raises(AssertionError, match="empty|zero features"):
            assert_csr_invariants(ids, mags)

    def test_rejects_non_positive_magnitudes(self):
        """Magnitudes must be > 0."""
        ids = [t.tensor([10, 20], dtype=t.int32)]
        mags = [t.tensor([0.5, 0.0], dtype=t.float32)]
        with pytest.raises(AssertionError, match="non-positive magnitudes"):
            assert_csr_invariants(ids, mags)

    def test_rejects_length_mismatch(self):
        """ID and magnitude lists must have the same length."""
        ids = [t.tensor([10, 20], dtype=t.int32)]
        mags = [t.tensor([0.5], dtype=t.float32)]
        with pytest.raises(AssertionError, match="len mismatch"):
            assert_csr_invariants(ids, mags)

    def test_rejects_wrong_dtype(self):
        """IDs must be int32, mags must be float32."""
        ids = [t.tensor([10, 20], dtype=t.int64)]
        mags = [t.tensor([0.5, 0.6], dtype=t.float32)]
        with pytest.raises(AssertionError, match="ids dtype"):
            assert_csr_invariants(ids, mags)

    def test_feature_range_check(self):
        """Feature IDs exceeding n_features should fail."""
        ids = [t.tensor([10, 200], dtype=t.int32)]
        mags = [t.tensor([0.5, 0.6], dtype=t.float32)]
        with pytest.raises(AssertionError, match="feature ID"):
            assert_csr_invariants(ids, mags, n_features=100)

    def test_no_longer_requires_mag_descending(self):
        """Magnitudes NOT sorted descending should still pass (old check removed)."""
        ids = [t.tensor([10, 20, 30], dtype=t.int32)]
        mags = [t.tensor([0.1, 0.5, 0.2], dtype=t.float32)]
        # This would have failed with the old mag-descending check
        assert_csr_invariants(ids, mags)


class TestValidateShardCompressed:
    """Tests for validate_shard with compressed .safetensors.zst format."""

    def test_valid_shard_passes(self, tmp_path):
        """A well-formed compressed shard should validate successfully."""
        data = _make_shard_dict(n_segments=3, feats_per_seg=5, n_docs=2)
        out = save_features(data, tmp_path / "shard_00000_features")
        result = validate_shard(str(out))
        assert result.valid, f"Violations: {result.violations}"

    def test_detects_negative_magnitudes(self, tmp_path):
        """Shard with negative magnitudes should fail validation."""
        data = _make_shard_dict(n_segments=2, feats_per_seg=3, n_docs=1)
        data["feature_mags"][0] = -1.0
        out = save_features(data, tmp_path / "shard_00000_features")
        result = validate_shard(str(out))
        assert not result.valid
        assert any("magnitudes" in v for v in result.violations)


class TestContextIndependentMetadataKey:
    def test_enum_value(self):
        assert MetadataKey.CONTEXT_INDEPENDENT_FEATURES.value == "context_independent_features"

    def test_enum_member_exists(self):
        names = {m.name for m in MetadataKey}
        assert "CONTEXT_INDEPENDENT_FEATURES" in names


def _make_shard_dict_with_ci(flag: bool) -> dict:
    """Minimal valid shard dict carrying context_independent_features metadata."""
    return {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([42], dtype=t.int32),
        "feature_mags": t.tensor([0.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([4], dtype=t.int32),
        "doc_token_counts": t.tensor([4], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "context_independent_features": flag,
    }


class TestContextIndependentFeaturesRoundTrip:
    def test_save_load_true(self, tmp_path):
        data = _make_shard_dict_with_ci(True)
        save_features(data, str(tmp_path / "shard_test_features"))
        loaded = load_features(str(tmp_path / "shard_test_features.safetensors.zst"))
        assert loaded[MetadataKey.CONTEXT_INDEPENDENT_FEATURES] is True
        assert isinstance(loaded[MetadataKey.CONTEXT_INDEPENDENT_FEATURES], bool)

    def test_save_load_false(self, tmp_path):
        data = _make_shard_dict_with_ci(False)
        save_features(data, str(tmp_path / "shard_test_features"))
        loaded = load_features(str(tmp_path / "shard_test_features.safetensors.zst"))
        assert loaded[MetadataKey.CONTEXT_INDEPENDENT_FEATURES] is False
        assert isinstance(loaded[MetadataKey.CONTEXT_INDEPENDENT_FEATURES], bool)

    def test_legacy_shard_defaults_false(self, tmp_path):
        """Legacy shards without the key load as False (backward compat)."""
        data = _make_shard_dict_with_ci(True)
        del data["context_independent_features"]
        save_features(data, str(tmp_path / "shard_test_features"))
        loaded = load_features(str(tmp_path / "shard_test_features.safetensors.zst"))
        # Key must be present (unconditional exposure) AND equal False.
        assert MetadataKey.CONTEXT_INDEPENDENT_FEATURES in loaded
        assert loaded[MetadataKey.CONTEXT_INDEPENDENT_FEATURES] is False


def _min_shard(offsets, n_feats):
    n_seg = len(offsets) - 1
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.arange(n_feats, dtype=t.int32),
        "feature_mags": t.ones(n_feats, dtype=t.float32),
        "doc_ids": t.zeros(n_seg, dtype=t.int32),
        "segment_indices": t.arange(n_seg, dtype=t.int32),
        "segment_positions": t.arange(n_seg, dtype=t.int32),
        "segment_char_starts": t.arange(n_seg, dtype=t.int32),
        "segment_char_ends": t.arange(1, n_seg + 1, dtype=t.int32),
        "doc_token_counts": t.tensor([10], dtype=t.int32),
        "n_docs": 1, "n_segments": n_seg, "layer": 0, "context_window": 128,
        "pooling": "max_pool", "segment_mode": "fixed_tokens", "segment_size": 8,
    }


class TestSaveFeaturesEmptySegmentGate:
    """Base-extraction writer must refuse to persist an empty segment (issue #522)."""

    def test_save_features_raises_on_empty_segment(self, tmp_path):
        with pytest.raises(EmptyExtractionSegmentError, match="segment 1"):
            save_features(_min_shard([0, 2, 2, 4], 4), tmp_path / "shard_00000")

    def test_save_features_accepts_nonempty(self, tmp_path):
        p = save_features(_min_shard([0, 2, 4], 4), tmp_path / "shard_00001")
        assert p.exists()
