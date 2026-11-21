"""Bounded, header-only layout inspection for compressed safetensors files.

The ordinary readers intentionally decompress complete shards.  Resource
admission needs a different primitive: exact declared tensor sizes without
allocating a multi-gigabyte body.  This module reads only the zstd prefix that
contains the safetensors header, validates the complete descriptor layout, and
detects path/target replacement during inspection.

This is an accounting primitive, not a replacement for the existing full-body
validators.  A header can describe bytes that a truncated zstd frame does not
actually contain; callers must retain their normal eager validation contract.
"""

from __future__ import annotations

import json
import os
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import zstandard

from steering.feature_io import COMPRESSED_EXT, MAX_SAFETENSORS_HEADER_SIZE


class SafetensorsLayoutError(ValueError):
    """A compressed safetensors header is unreadable or structurally invalid."""


_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
}


@dataclass(frozen=True)
class FileIdentity:
    """Link and target identity used to detect inspection-time replacement."""

    path: str
    lstat_fields: tuple[int, int, int, int, int, int]
    stat_fields: tuple[int, int, int, int, int, int]

    @staticmethod
    def _fields(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            int(result.st_mode),
            int(result.st_dev),
            int(result.st_ino),
            int(result.st_size),
            int(result.st_mtime_ns),
            int(result.st_ctime_ns),
        )

    @classmethod
    def capture(cls, path: Path) -> FileIdentity:
        absolute = path.absolute()
        try:
            link_result = absolute.lstat()
            target_result = absolute.stat()
        except OSError as exc:
            raise SafetensorsLayoutError(
                f"cannot inspect compressed safetensors path {absolute}: {exc}"
            ) from exc
        if not stat.S_ISREG(target_result.st_mode):
            raise SafetensorsLayoutError(
                f"compressed safetensors target is not a regular file: {absolute}"
            )
        return cls(
            path=str(absolute),
            lstat_fields=cls._fields(link_result),
            stat_fields=cls._fields(target_result),
        )


@dataclass(frozen=True)
class TensorLayout:
    """One validated safetensors tensor descriptor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    numel: int
    nbytes: int


@dataclass(frozen=True)
class CompressedSafetensorsLayout:
    """Validated header layout and exact declared/compressed byte counts."""

    path: str
    identity: FileIdentity
    metadata: tuple[tuple[str, str], ...]
    tensors: tuple[TensorLayout, ...]
    header_bytes: int
    tensor_bytes: int
    serialized_bytes: int
    compressed_bytes: int

    def tensor(self, name: str) -> TensorLayout:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(tensor.name for tensor in self.tensors)


@dataclass(frozen=True)
class SemanticSafetensorsLayout:
    """Filesystem-independent safetensors layout authenticated by an inventory."""

    metadata: tuple[tuple[str, str], ...]
    tensors: tuple[TensorLayout, ...]
    header_bytes: int
    tensor_bytes: int
    serialized_bytes: int
    compressed_bytes: int

    @property
    def path(self) -> str:
        """Diagnostic label; never serialized as semantic evidence."""
        return "<authenticated semantic layout>"

    def tensor(self, name: str) -> TensorLayout:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(tensor.name for tensor in self.tensors)


_SEMANTIC_LAYOUT_KEYS = frozenset(
    {
        "metadata",
        "tensors",
        "header_bytes",
        "tensor_bytes",
        "serialized_bytes",
        "compressed_bytes",
    }
)
_SEMANTIC_TENSOR_KEYS = frozenset({"name", "dtype", "shape", "data_offsets", "numel", "nbytes"})


def semantic_layout_from_compressed(
    layout: CompressedSafetensorsLayout,
) -> SemanticSafetensorsLayout:
    """Drop machine-specific path/inode evidence from a validated local layout."""
    return SemanticSafetensorsLayout(
        metadata=layout.metadata,
        tensors=layout.tensors,
        header_bytes=layout.header_bytes,
        tensor_bytes=layout.tensor_bytes,
        serialized_bytes=layout.serialized_bytes,
        compressed_bytes=layout.compressed_bytes,
    )


def semantic_layout_to_json(
    layout: SemanticSafetensorsLayout | CompressedSafetensorsLayout,
) -> dict[str, Any]:
    """Return the canonical, filesystem-independent JSON layout summary."""
    semantic = (
        semantic_layout_from_compressed(layout)
        if isinstance(layout, CompressedSafetensorsLayout)
        else layout
    )
    return {
        "metadata": dict(semantic.metadata),
        "tensors": [
            {
                "name": tensor.name,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "data_offsets": list(tensor.data_offsets),
                "numel": tensor.numel,
                "nbytes": tensor.nbytes,
            }
            for tensor in semantic.tensors
        ],
        "header_bytes": semantic.header_bytes,
        "tensor_bytes": semantic.tensor_bytes,
        "serialized_bytes": semantic.serialized_bytes,
        "compressed_bytes": semantic.compressed_bytes,
    }


def _strict_nonnegative_int(value: object, *, surface: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SafetensorsLayoutError(
            f"{surface} must be a non-negative strict integer, got {value!r}"
        )
    return value


def parse_semantic_layout(
    raw: object,
    *,
    surface: str = "semantic safetensors layout",
) -> SemanticSafetensorsLayout:
    """Strictly parse and revalidate an inventory-carried layout summary."""
    if not isinstance(raw, Mapping) or set(raw) != _SEMANTIC_LAYOUT_KEYS:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise SafetensorsLayoutError(
            f"{surface} must contain exactly {sorted(_SEMANTIC_LAYOUT_KEYS)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    mapping = cast(Mapping[str, object], raw)
    metadata_raw = mapping["metadata"]
    if not isinstance(metadata_raw, Mapping):
        raise SafetensorsLayoutError(f"{surface}.metadata must be an object")
    metadata: list[tuple[str, str]] = []
    for key, value in metadata_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SafetensorsLayoutError(f"{surface}.metadata entries must be string-to-string")
        metadata.append((key, value))
    metadata_tuple = tuple(sorted(metadata))

    tensors_raw = mapping["tensors"]
    if not isinstance(tensors_raw, list):
        raise SafetensorsLayoutError(f"{surface}.tensors must be a list")
    tensors: list[TensorLayout] = []
    previous_name: str | None = None
    for index, item in enumerate(tensors_raw):
        item_surface = f"{surface}.tensors[{index}]"
        if not isinstance(item, Mapping) or set(item) != _SEMANTIC_TENSOR_KEYS:
            keys = set(item) if isinstance(item, Mapping) else set()
            raise SafetensorsLayoutError(
                f"{item_surface} must contain exactly {sorted(_SEMANTIC_TENSOR_KEYS)}; "
                f"got keys={sorted(keys, key=str)}"
            )
        descriptor = cast(Mapping[str, object], item)
        name = descriptor["name"]
        if not isinstance(name, str) or not name:
            raise SafetensorsLayoutError(f"{item_surface}.name must be non-empty")
        if previous_name is not None and name <= previous_name:
            raise SafetensorsLayoutError(
                f"{surface}.tensors must have unique strictly name-sorted entries"
            )
        previous_name = name
        parsed = _parse_tensor(
            Path(f"<{surface}>"),
            name,
            {
                "dtype": descriptor["dtype"],
                "shape": descriptor["shape"],
                "data_offsets": descriptor["data_offsets"],
            },
        )
        declared_numel = _strict_nonnegative_int(
            descriptor["numel"], surface=f"{item_surface}.numel"
        )
        declared_nbytes = _strict_nonnegative_int(
            descriptor["nbytes"], surface=f"{item_surface}.nbytes"
        )
        if declared_numel != parsed.numel or declared_nbytes != parsed.nbytes:
            raise SafetensorsLayoutError(
                f"{item_surface} declared numel/nbytes disagree with dtype, shape, and offsets"
            )
        tensors.append(parsed)

    cursor = 0
    by_offset = sorted(
        tensors,
        key=lambda tensor: (
            tensor.data_offsets[0],
            tensor.data_offsets[1],
            tensor.name,
        ),
    )
    for tensor in by_offset:
        start, end = tensor.data_offsets
        if start != cursor:
            relation = "overlaps prior data" if start < cursor else "leaves a data gap"
            raise SafetensorsLayoutError(
                f"{surface} tensor {tensor.name!r} starts at {start}, expected "
                f"{cursor} ({relation})"
            )
        cursor = end

    header_bytes = _strict_nonnegative_int(
        mapping["header_bytes"], surface=f"{surface}.header_bytes"
    )
    tensor_bytes = _strict_nonnegative_int(
        mapping["tensor_bytes"], surface=f"{surface}.tensor_bytes"
    )
    serialized_bytes = _strict_nonnegative_int(
        mapping["serialized_bytes"], surface=f"{surface}.serialized_bytes"
    )
    compressed_bytes = _strict_nonnegative_int(
        mapping["compressed_bytes"], surface=f"{surface}.compressed_bytes"
    )
    if header_bytes == 0 or header_bytes % 8 != 0:
        raise SafetensorsLayoutError(f"{surface}.header_bytes must be a positive multiple of eight")
    if header_bytes > MAX_SAFETENSORS_HEADER_SIZE:
        raise SafetensorsLayoutError(
            f"{surface}.header_bytes exceeds the {MAX_SAFETENSORS_HEADER_SIZE}-byte limit"
        )
    if tensor_bytes != cursor:
        raise SafetensorsLayoutError(
            f"{surface}.tensor_bytes={tensor_bytes} but descriptors end at {cursor}"
        )
    expected_serialized = 8 + header_bytes + tensor_bytes
    if serialized_bytes != expected_serialized:
        raise SafetensorsLayoutError(
            f"{surface}.serialized_bytes={serialized_bytes}, expected {expected_serialized}"
        )
    if compressed_bytes == 0:
        raise SafetensorsLayoutError(f"{surface}.compressed_bytes must be positive")
    parsed_layout = SemanticSafetensorsLayout(
        metadata=metadata_tuple,
        tensors=tuple(tensors),
        header_bytes=header_bytes,
        tensor_bytes=tensor_bytes,
        serialized_bytes=serialized_bytes,
        compressed_bytes=compressed_bytes,
    )
    if semantic_layout_to_json(parsed_layout) != dict(mapping):
        raise SafetensorsLayoutError(f"{surface} is not in canonical semantic form")
    return parsed_layout


def _read_exact(reader, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        chunk = reader.read(count - len(result))
        if not chunk:
            break
        result.extend(chunk)
    return bytes(result)


def _invalid(path: Path, detail: str) -> SafetensorsLayoutError:
    return SafetensorsLayoutError(f"compressed safetensors layout is invalid for {path}: {detail}")


def _parse_metadata(path: Path, value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise _invalid(path, "__metadata__ must be an object")
    result: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise _invalid(path, "__metadata__ entries must be string-to-string")
        result.append((key, item))
    return tuple(sorted(result))


def _parse_tensor(path: Path, name: str, value: object) -> TensorLayout:
    if not isinstance(value, dict):
        raise _invalid(path, f"tensor {name!r} descriptor must be an object")
    descriptor = cast(dict[str, object], value)
    expected_keys = {"dtype", "shape", "data_offsets"}
    if set(descriptor) != expected_keys:
        raise _invalid(
            path,
            f"tensor {name!r} descriptor keys are {sorted(descriptor)}, "
            f"expected {sorted(expected_keys)}",
        )

    dtype = descriptor["dtype"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise _invalid(path, f"tensor {name!r} has unsupported dtype {dtype!r}")

    raw_shape = descriptor["shape"]
    if not isinstance(raw_shape, list):
        raise _invalid(path, f"tensor {name!r} shape must be a list")
    shape: list[int] = []
    numel = 1
    for axis, dimension in enumerate(raw_shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise _invalid(
                path,
                f"tensor {name!r} shape[{axis}] must be an integer, got {dimension!r}",
            )
        if dimension < 0:
            raise _invalid(path, f"tensor {name!r} shape[{axis}] is negative: {dimension}")
        shape.append(dimension)
        numel *= dimension

    raw_offsets = descriptor["data_offsets"]
    if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
        raise _invalid(path, f"tensor {name!r} data_offsets must be a two-element list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_offsets):
        raise _invalid(path, f"tensor {name!r} data_offsets must be integers")
    start = cast(int, raw_offsets[0])
    end = cast(int, raw_offsets[1])
    if start < 0 or end < start:
        raise _invalid(path, f"tensor {name!r} has invalid data_offsets [{start}, {end}]")
    nbytes = numel * _DTYPE_BYTES[dtype]
    if end - start != nbytes:
        raise _invalid(
            path,
            f"tensor {name!r} span is {end - start} bytes but "
            f"dtype={dtype}, shape={shape} require {nbytes}",
        )
    return TensorLayout(
        name=name,
        dtype=dtype,
        shape=tuple(shape),
        data_offsets=(start, end),
        numel=numel,
        nbytes=nbytes,
    )


def _unique_json_object(
    path: Path,
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid(path, f"header contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_compressed_safetensors_layout(
    path: str | Path,
) -> CompressedSafetensorsLayout:
    """Read and validate only a ``.safetensors.zst`` header.

    The returned ``serialized_bytes`` is exact for the bytes declared by the
    validated header: eight length bytes, the padded JSON header, and the
    contiguous tensor body.  The function intentionally does not read that
    body.  Link and target identities must remain unchanged for the duration
    of inspection.
    """
    path = Path(path).absolute()
    if not path.name.endswith(COMPRESSED_EXT):
        raise SafetensorsLayoutError(f"expected a {COMPRESSED_EXT} file, got {path.name!r}")
    before = FileIdentity.capture(path)
    try:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                raw_header_size = _read_exact(reader, 8)
                if len(raw_header_size) != 8:
                    raise _invalid(
                        path,
                        "stream ended before the eight-byte safetensors header length",
                    )
                header_size = struct.unpack("<Q", raw_header_size)[0]
                if header_size == 0:
                    raise _invalid(path, "header length is zero")
                if header_size % 8 != 0:
                    raise _invalid(
                        path,
                        f"header length {header_size} is not aligned to eight bytes",
                    )
                if header_size > MAX_SAFETENSORS_HEADER_SIZE:
                    raise _invalid(
                        path,
                        f"header length {header_size} exceeds the "
                        f"{MAX_SAFETENSORS_HEADER_SIZE}-byte limit",
                    )
                header = _read_exact(reader, header_size)
    except SafetensorsLayoutError:
        raise
    except (OSError, zstandard.ZstdError) as exc:
        raise SafetensorsLayoutError(
            f"cannot read compressed safetensors header {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if len(header) != header_size:
        raise _invalid(
            path,
            f"stream ended after {len(header)} of {header_size} header bytes",
        )
    try:
        parsed = json.loads(
            header,
            object_pairs_hook=lambda pairs: _unique_json_object(path, pairs),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid(path, f"header is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _invalid(path, "header must be a JSON object")

    metadata = _parse_metadata(path, parsed.pop("__metadata__", {}))
    tensors = tuple(
        sorted(
            (_parse_tensor(path, name, value) for name, value in parsed.items()),
            key=lambda item: item.name,
        )
    )
    by_offset = sorted(
        tensors,
        key=lambda item: (item.data_offsets[0], item.data_offsets[1], item.name),
    )
    cursor = 0
    for tensor in by_offset:
        start, end = tensor.data_offsets
        if start != cursor:
            relation = "overlaps prior data" if start < cursor else "leaves a data gap"
            raise _invalid(
                path,
                f"tensor {tensor.name!r} starts at {start}, expected {cursor} ({relation})",
            )
        cursor = end

    after = FileIdentity.capture(path)
    if after != before:
        raise SafetensorsLayoutError(
            f"compressed safetensors path changed during header inspection: {path}"
        )
    return CompressedSafetensorsLayout(
        path=str(path),
        identity=before,
        metadata=metadata,
        tensors=tensors,
        header_bytes=header_size,
        tensor_bytes=cursor,
        serialized_bytes=8 + header_size + cursor,
        compressed_bytes=before.stat_fields[3],
    )
