"""Exhaustive contract tests for compressed safetensors layout inspection."""

from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path

import pytest
import torch as t
import zstandard
from safetensors.torch import save as safetensors_save

import steering.safetensors_layout as layout_module
from steering.feature_io import MAX_SAFETENSORS_HEADER_SIZE
from steering.safetensors_layout import (
    FileIdentity,
    SafetensorsLayoutError,
    parse_semantic_layout,
    read_compressed_safetensors_layout,
    semantic_layout_from_compressed,
    semantic_layout_to_json,
)


def _padded_json(value: object) -> bytes:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return raw + b" " * ((-len(raw)) % 8)


def _write_payload(
    path: Path,
    header: object,
    body: bytes = b"",
    *,
    pad: bool = True,
) -> bytes:
    header_bytes = _padded_json(header) if pad else json.dumps(header).encode()
    raw = struct.pack("<Q", len(header_bytes)) + header_bytes + body
    compressed = zstandard.ZstdCompressor(level=1).compress(raw)
    path.write_bytes(compressed)
    return raw


def _write_raw_json_header(path: Path, raw_json: bytes) -> None:
    header = raw_json + b" " * ((-len(raw_json)) % 8)
    raw = struct.pack("<Q", len(header)) + header
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw))


def _write_real(path: Path, tensors: dict[str, t.Tensor]) -> bytes:
    raw = safetensors_save(tensors, metadata={"z": "last", "a": "first"})
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw))
    return raw


class TestValidLayouts:
    def test_semantic_summary_round_trip_excludes_filesystem_identity(self, tmp_path: Path) -> None:
        path = tmp_path / "semantic.safetensors.zst"
        _write_real(
            path,
            {
                "x": t.arange(6, dtype=t.int32).reshape(2, 3),
                "y": t.empty(0, dtype=t.uint8),
            },
        )
        local = read_compressed_safetensors_layout(path)

        summary = semantic_layout_to_json(local)
        parsed = parse_semantic_layout(summary)

        assert parsed == semantic_layout_from_compressed(local)
        encoded = json.dumps(summary)
        assert str(path) not in encoded
        assert "identity" not in encoded
        assert "mtime" not in encoded

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            (lambda raw: raw.__setitem__("path", "/tmp/leak"), "contain exactly"),
            (
                lambda raw: raw["tensors"][0].__setitem__("numel", True),
                "non-negative strict integer",
            ),
            (
                lambda raw: raw["tensors"][0].__setitem__(
                    "data_offsets", [1, raw["tensors"][0]["data_offsets"][1] + 1]
                ),
                "expected 0",
            ),
            (
                lambda raw: raw.__setitem__("serialized_bytes", raw["serialized_bytes"] + 1),
                "serialized_bytes",
            ),
        ],
    )
    def test_semantic_summary_rejects_noncanonical_mutations(
        self, tmp_path: Path, mutation, match: str
    ) -> None:
        path = tmp_path / "mutated.safetensors.zst"
        _write_real(path, {"x": t.arange(4, dtype=t.int32)})
        summary = semantic_layout_to_json(read_compressed_safetensors_layout(path))
        raw = copy.deepcopy(summary)
        mutation(raw)

        with pytest.raises(SafetensorsLayoutError, match=match):
            parse_semantic_layout(raw)

    def test_real_safetensors_exact_counts_and_order(self, tmp_path: Path) -> None:
        path = tmp_path / "real.safetensors.zst"
        raw = _write_real(
            path,
            {
                "z_tensor": t.arange(12, dtype=t.int32).reshape(3, 4),
                "a_tensor": t.arange(5, dtype=t.int32).to(t.uint16),
                "empty": t.empty((2, 0, 7), dtype=t.float32),
            },
        )
        result = read_compressed_safetensors_layout(path)

        assert result.path == str(path.absolute())
        assert result.metadata == (("a", "first"), ("z", "last"))
        assert [item.name for item in result.tensors] == [
            "a_tensor",
            "empty",
            "z_tensor",
        ]
        assert result.tensor_names == frozenset({"a_tensor", "empty", "z_tensor"})
        assert result.tensor("a_tensor").dtype == "U16"
        assert result.tensor("a_tensor").shape == (5,)
        assert result.tensor("a_tensor").numel == 5
        assert result.tensor("a_tensor").nbytes == 10
        assert result.tensor("empty").shape == (2, 0, 7)
        assert result.tensor("empty").numel == 0
        assert result.tensor("empty").nbytes == 0
        assert result.tensor("z_tensor").shape == (3, 4)
        assert result.tensor("z_tensor").nbytes == 48
        assert result.tensor_bytes == 58
        assert result.serialized_bytes == len(raw)
        assert result.compressed_bytes == path.stat().st_size
        assert result.serialized_bytes == 8 + result.header_bytes + result.tensor_bytes
        assert result.identity.path == str(path.absolute())
        assert result.identity.stat_fields[3] == path.stat().st_size

    def test_symlink_tracks_link_and_target_identity(self, tmp_path: Path) -> None:
        target = tmp_path / "target.safetensors.zst"
        _write_real(target, {"x": t.arange(3, dtype=t.uint8)})
        link = tmp_path / "link.safetensors.zst"
        link.symlink_to(target.name)

        result = read_compressed_safetensors_layout(link)

        assert result.path == str(link.absolute())
        assert result.identity.lstat_fields[1:3] != result.identity.stat_fields[1:3]
        assert result.compressed_bytes == target.stat().st_size
        assert result.tensor("x").shape == (3,)

    @pytest.mark.parametrize(
        ("dtype", "expected_bytes"),
        [
            ("BOOL", 1),
            ("U8", 1),
            ("I8", 1),
            ("F8_E4M3", 1),
            ("F8_E5M2", 1),
            ("U16", 2),
            ("I16", 2),
            ("F16", 2),
            ("BF16", 2),
            ("U32", 4),
            ("I32", 4),
            ("F32", 4),
            ("U64", 8),
            ("I64", 8),
            ("F64", 8),
            ("C64", 8),
        ],
    )
    def test_every_supported_dtype_has_exact_size(
        self, tmp_path: Path, dtype: str, expected_bytes: int
    ) -> None:
        path = tmp_path / f"{dtype}.safetensors.zst"
        raw = _write_payload(
            path,
            {
                "x": {
                    "dtype": dtype,
                    "shape": [2, 3],
                    "data_offsets": [0, 6 * expected_bytes],
                }
            },
            b"\0" * (6 * expected_bytes),
        )
        result = read_compressed_safetensors_layout(path)
        assert result.tensor("x").numel == 6
        assert result.tensor("x").nbytes == 6 * expected_bytes
        assert result.serialized_bytes == len(raw)

    def test_scalar_shape_has_one_element(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "I64", "shape": [], "data_offsets": [0, 8]}},
            b"\0" * 8,
        )
        tensor = read_compressed_safetensors_layout(path).tensor("x")
        assert tensor.shape == ()
        assert tensor.numel == 1
        assert tensor.nbytes == 8

    def test_empty_tensor_set_is_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.safetensors.zst"
        raw = _write_payload(path, {"__metadata__": {"kind": "empty"}})
        result = read_compressed_safetensors_layout(path)
        assert result.tensor_names == frozenset()
        assert result.tensor_bytes == 0
        assert result.serialized_bytes == len(raw)

    def test_zero_length_tensors_can_share_current_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "zeros.safetensors.zst"
        _write_payload(
            path,
            {
                "empty_a": {
                    "dtype": "F32",
                    "shape": [0],
                    "data_offsets": [0, 0],
                },
                "full": {
                    "dtype": "U8",
                    "shape": [3],
                    "data_offsets": [0, 3],
                },
                "empty_b": {
                    "dtype": "I64",
                    "shape": [0],
                    "data_offsets": [3, 3],
                },
            },
            b"abc",
        )
        result = read_compressed_safetensors_layout(path)
        assert result.tensor_bytes == 3
        assert result.tensor("empty_a").data_offsets == (0, 0)
        assert result.tensor("empty_b").data_offsets == (3, 3)

    def test_declared_body_need_not_be_read(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "bounded.safetensors.zst"
        header = _padded_json(
            {
                "x": {
                    "dtype": "U8",
                    "shape": [10_000_000],
                    "data_offsets": [0, 10_000_000],
                }
            }
        )
        decompressed = struct.pack("<Q", len(header)) + header + b"x" * 10_000_000
        path.write_bytes(b"placeholder")
        state = {"bytes": 0}

        class Reader:
            def __init__(self) -> None:
                self.position = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, count: int) -> bytes:
                chunk = decompressed[self.position : self.position + count]
                self.position += len(chunk)
                state["bytes"] += len(chunk)
                return chunk

        class Decompressor:
            def stream_reader(self, _source):
                return Reader()

        monkeypatch.setattr(layout_module.zstandard, "ZstdDecompressor", Decompressor)
        result = read_compressed_safetensors_layout(path)
        assert result.tensor_bytes == 10_000_000
        assert state["bytes"] == 8 + len(header)
        assert state["bytes"] < len(decompressed) // 1000

    def test_short_reads_are_reassembled(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "short.safetensors.zst"
        header = _padded_json({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}})
        decompressed = struct.pack("<Q", len(header)) + header + b"z"
        path.write_bytes(b"placeholder")

        class Reader:
            def __init__(self) -> None:
                self.position = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _count: int) -> bytes:
                chunk = decompressed[self.position : self.position + 1]
                self.position += len(chunk)
                return chunk

        class Decompressor:
            def stream_reader(self, _source):
                return Reader()

        monkeypatch.setattr(layout_module.zstandard, "ZstdDecompressor", Decompressor)
        assert read_compressed_safetensors_layout(path).tensor_bytes == 1

    def test_header_only_contract_accepts_absent_declared_body(self, tmp_path: Path) -> None:
        path = tmp_path / "header-only.safetensors.zst"
        _write_payload(
            path,
            {
                "x": {
                    "dtype": "U8",
                    "shape": [1_000_000],
                    "data_offsets": [0, 1_000_000],
                }
            },
        )
        result = read_compressed_safetensors_layout(path)
        assert result.tensor_bytes == 1_000_000
        assert result.serialized_bytes > result.compressed_bytes

    def test_result_objects_are_frozen(self, tmp_path: Path) -> None:
        path = tmp_path / "frozen.safetensors.zst"
        _write_real(path, {"x": t.arange(2)})
        result = read_compressed_safetensors_layout(path)
        with pytest.raises(AttributeError):
            result.tensor_bytes = 0  # ty: ignore[invalid-assignment] -- frozen contract
        with pytest.raises(AttributeError):
            result.tensor("x").dtype = (  # ty: ignore[invalid-assignment] -- frozen contract
                "U8"
            )

    def test_missing_tensor_lookup_is_keyerror(self, tmp_path: Path) -> None:
        path = tmp_path / "missing-key.safetensors.zst"
        _write_real(path, {"x": t.arange(2)})
        with pytest.raises(KeyError, match="absent"):
            read_compressed_safetensors_layout(path).tensor("absent")


class TestPathAndStreamFailures:
    def test_wrong_extension_fails_before_io(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.bin"
        path.write_bytes(b"anything")
        with pytest.raises(SafetensorsLayoutError, match="expected a .safetensors.zst"):
            read_compressed_safetensors_layout(path)

    def test_missing_path_names_path(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.safetensors.zst"
        with pytest.raises(SafetensorsLayoutError, match=str(path)):
            read_compressed_safetensors_layout(path)

    def test_directory_target_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "directory.safetensors.zst"
        path.mkdir()
        with pytest.raises(SafetensorsLayoutError, match="not a regular file"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("payload", [b"", b"not-zstd", b"\x28\xb5\x2f\xfd"])
    def test_unreadable_stream_is_normalized(self, tmp_path: Path, payload: bytes) -> None:
        path = tmp_path / "broken.safetensors.zst"
        path.write_bytes(payload)
        with pytest.raises(SafetensorsLayoutError, match=str(path)):
            read_compressed_safetensors_layout(path)

    def test_short_header_length(self, tmp_path: Path) -> None:
        path = tmp_path / "short-length.safetensors.zst"
        raw = b"\x01\x02\x03"
        path.write_bytes(zstandard.ZstdCompressor().compress(raw))
        with pytest.raises(SafetensorsLayoutError, match="eight-byte"):
            read_compressed_safetensors_layout(path)

    def test_zero_header_length(self, tmp_path: Path) -> None:
        path = tmp_path / "zero.safetensors.zst"
        path.write_bytes(zstandard.ZstdCompressor().compress(struct.pack("<Q", 0)))
        with pytest.raises(SafetensorsLayoutError, match="length is zero"):
            read_compressed_safetensors_layout(path)

    def test_unaligned_header_length(self, tmp_path: Path) -> None:
        path = tmp_path / "unaligned.safetensors.zst"
        _write_payload(path, {}, pad=False)
        with pytest.raises(SafetensorsLayoutError, match="not aligned"):
            read_compressed_safetensors_layout(path)

    def test_header_over_cap(self, tmp_path: Path) -> None:
        path = tmp_path / "huge.safetensors.zst"
        raw = struct.pack("<Q", MAX_SAFETENSORS_HEADER_SIZE + 8)
        path.write_bytes(zstandard.ZstdCompressor().compress(raw))
        with pytest.raises(SafetensorsLayoutError, match="exceeds"):
            read_compressed_safetensors_layout(path)

    def test_stream_ends_mid_header(self, tmp_path: Path) -> None:
        path = tmp_path / "mid-header.safetensors.zst"
        raw = struct.pack("<Q", 16) + b"{}      "
        path.write_bytes(zstandard.ZstdCompressor().compress(raw))
        with pytest.raises(SafetensorsLayoutError, match="8 of 16"):
            read_compressed_safetensors_layout(path)

    def test_non_json_header(self, tmp_path: Path) -> None:
        path = tmp_path / "non-json.safetensors.zst"
        header = b"not json"
        assert len(header) == 8
        raw = struct.pack("<Q", 8) + header
        path.write_bytes(zstandard.ZstdCompressor().compress(raw))
        with pytest.raises(SafetensorsLayoutError, match="not valid JSON"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("header", [[], 1, "text", None, True])
    def test_top_level_must_be_object(self, tmp_path: Path, header: object) -> None:
        path = tmp_path / "top.safetensors.zst"
        _write_payload(path, header)
        with pytest.raises(SafetensorsLayoutError, match="JSON object"):
            read_compressed_safetensors_layout(path)

    def test_replacement_during_read_is_detected(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "raced.safetensors.zst"
        _write_real(path, {"x": t.arange(2, dtype=t.uint8)})
        replacement = tmp_path / "replacement.safetensors.zst"
        _write_real(replacement, {"x": t.arange(2, dtype=t.uint8)})
        original_capture = FileIdentity.capture.__func__
        calls = 0

        def raced_capture(cls, captured_path: Path):
            nonlocal calls
            calls += 1
            if calls == 2:
                os.replace(replacement, path)
            return original_capture(cls, captured_path)

        monkeypatch.setattr(FileIdentity, "capture", classmethod(raced_capture))
        with pytest.raises(SafetensorsLayoutError, match="changed during"):
            read_compressed_safetensors_layout(path)


class TestDescriptorValidation:
    @pytest.mark.parametrize(
        "raw_json",
        [
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
            b'"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            b'{"x":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
            b'{"__metadata__":{"key":"first","key":"second"}}',
        ],
    )
    def test_duplicate_json_keys_are_rejected_at_every_object_level(
        self, tmp_path: Path, raw_json: bytes
    ) -> None:
        path = tmp_path / "duplicate.safetensors.zst"
        _write_raw_json_header(path, raw_json)
        with pytest.raises(SafetensorsLayoutError, match="duplicate JSON key"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("metadata", [[], "x", 1, None, True])
    def test_metadata_must_be_object(self, tmp_path: Path, metadata: object) -> None:
        path = tmp_path / "metadata.safetensors.zst"
        _write_payload(path, {"__metadata__": metadata})
        with pytest.raises(SafetensorsLayoutError, match="metadata__ must be an object"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("value", [1, None, [], {}, True])
    def test_metadata_values_must_be_strings(self, tmp_path: Path, value: object) -> None:
        path = tmp_path / "metadata-value.safetensors.zst"
        _write_payload(path, {"__metadata__": {"key": value}})
        with pytest.raises(SafetensorsLayoutError, match="string-to-string"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("descriptor", [None, [], 1, "tensor", True])
    def test_tensor_descriptor_must_be_object(self, tmp_path: Path, descriptor: object) -> None:
        path = tmp_path / "descriptor.safetensors.zst"
        _write_payload(path, {"x": descriptor})
        with pytest.raises(SafetensorsLayoutError, match="descriptor must be an object"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize(
        "descriptor",
        [
            {},
            {"dtype": "U8"},
            {"dtype": "U8", "shape": [1]},
            {
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [0, 1],
                "extra": 1,
            },
        ],
    )
    def test_descriptor_keys_are_exact(self, tmp_path: Path, descriptor: dict) -> None:
        path = tmp_path / "keys.safetensors.zst"
        _write_payload(path, {"x": descriptor})
        with pytest.raises(SafetensorsLayoutError, match="descriptor keys"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("dtype", [None, 1, True, "", "C128", "FLOAT32"])
    def test_dtype_must_be_supported(self, tmp_path: Path, dtype: object) -> None:
        path = tmp_path / "dtype.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": dtype, "shape": [0], "data_offsets": [0, 0]}},
        )
        with pytest.raises(SafetensorsLayoutError, match="unsupported dtype"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("shape", [None, {}, "1", 1, True])
    def test_shape_must_be_list(self, tmp_path: Path, shape: object) -> None:
        path = tmp_path / "shape.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "U8", "shape": shape, "data_offsets": [0, 0]}},
        )
        with pytest.raises(SafetensorsLayoutError, match="shape must be a list"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("dimension", [True, False, 1.0, "1", None, [], {}])
    def test_shape_dimensions_must_be_non_boolean_integers(
        self, tmp_path: Path, dimension: object
    ) -> None:
        path = tmp_path / "dimension.safetensors.zst"
        _write_payload(
            path,
            {
                "x": {
                    "dtype": "U8",
                    "shape": [dimension],
                    "data_offsets": [0, 0],
                }
            },
        )
        with pytest.raises(SafetensorsLayoutError, match=r"shape\[0\].*integer"):
            read_compressed_safetensors_layout(path)

    def test_negative_dimension_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "negative-shape.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 0]}},
        )
        with pytest.raises(SafetensorsLayoutError, match="negative"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("offsets", [None, {}, 1, [], [0], [0, 1, 2]])
    def test_offsets_must_be_two_element_list(self, tmp_path: Path, offsets: object) -> None:
        path = tmp_path / "offset-shape.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": offsets}},
        )
        with pytest.raises(SafetensorsLayoutError, match="two-element list"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize(
        "offsets", [[False, 1], [0, True], [0.0, 1], [0, 1.0], ["0", 1], [0, None]]
    )
    def test_offsets_must_be_non_boolean_integers(
        self, tmp_path: Path, offsets: list[object]
    ) -> None:
        path = tmp_path / "offset-types.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": offsets}},
        )
        with pytest.raises(SafetensorsLayoutError, match="offsets must be integers"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize("offsets", [[-1, 0], [2, 1]])
    def test_offsets_must_be_ordered_and_nonnegative(
        self, tmp_path: Path, offsets: list[int]
    ) -> None:
        path = tmp_path / "offset-order.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": offsets}},
        )
        with pytest.raises(SafetensorsLayoutError, match="invalid data_offsets"):
            read_compressed_safetensors_layout(path)

    @pytest.mark.parametrize(
        ("dtype", "shape", "offsets"),
        [
            ("U8", [2], [0, 1]),
            ("I64", [2], [0, 8]),
            ("F32", [0], [0, 4]),
            ("U16", [2, 3], [0, 11]),
        ],
    )
    def test_span_must_equal_shape_times_dtype(
        self,
        tmp_path: Path,
        dtype: str,
        shape: list[int],
        offsets: list[int],
    ) -> None:
        path = tmp_path / "span.safetensors.zst"
        _write_payload(
            path,
            {"x": {"dtype": dtype, "shape": shape, "data_offsets": offsets}},
        )
        with pytest.raises(SafetensorsLayoutError, match="span is"):
            read_compressed_safetensors_layout(path)

    def test_gap_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "gap.safetensors.zst"
        _write_payload(
            path,
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [3, 5]},
            },
        )
        with pytest.raises(SafetensorsLayoutError, match="leaves a data gap"):
            read_compressed_safetensors_layout(path)

    def test_overlap_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "overlap.safetensors.zst"
        _write_payload(
            path,
            {
                "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]},
            },
        )
        with pytest.raises(SafetensorsLayoutError, match="overlaps prior data"):
            read_compressed_safetensors_layout(path)
