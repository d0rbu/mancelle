"""Dedicated suite for canonical-header safetensors serialization."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch as t
from safetensors.torch import load, save

import steering.safetensors_io as safetensors_io
from steering.immutable_io import ImmutableArtifactError
from steering.safetensors_io import (
    canonicalize_safetensors_bytes,
    deterministic_safetensors_bytes,
    load_safetensors_file_snapshot,
    read_safetensors_metadata_from_bytes,
)


def _tensors() -> dict[str, t.Tensor]:
    return {
        "b_second": t.arange(6, dtype=t.int64).reshape(2, 3),
        "a_first": t.ones(4, dtype=t.float32),
        "c_empty": t.empty(0, dtype=t.uint8),
    }


def _header_parts(payload: bytes) -> tuple[dict[str, object], bytes]:
    header_length = int.from_bytes(payload[:8], "little")
    return json.loads(payload[8 : 8 + header_length]), payload[8 + header_length :]


def _with_header(header: bytes, data: bytes = b"") -> bytes:
    padded = header + b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(padded)) + padded + data


def test_round_trip_preserves_tensors_and_metadata():
    payload = deterministic_safetensors_bytes(_tensors(), metadata={"k1": "v1", "k2": "v2"})
    loaded = load(payload)
    assert set(loaded) == {"a_first", "b_second", "c_empty"}
    assert t.equal(loaded["b_second"], _tensors()["b_second"])
    assert t.equal(loaded["a_first"], _tensors()["a_first"])
    assert loaded["c_empty"].numel() == 0


def test_bytes_identical_across_metadata_insertion_orders():
    m1 = {"zeta": "1", "alpha": "2", "mid": "3"}
    m2 = {"mid": "3", "zeta": "1", "alpha": "2"}
    assert deterministic_safetensors_bytes(
        _tensors(), metadata=m1
    ) == deterministic_safetensors_bytes(_tensors(), metadata=m2)


def test_bytes_identical_across_tensor_dict_insertion_orders():
    d1 = _tensors()
    d2 = dict(reversed(list(_tensors().items())))
    assert deterministic_safetensors_bytes(d1) == deterministic_safetensors_bytes(d2)


def test_canonicalize_legacy_header_is_idempotent_and_preserves_data_section():
    raw = bytes(save(_tensors(), metadata={"zeta": "1", "alpha": "2", "mid": "3"}))
    header, data = _header_parts(raw)
    metadata = header["__metadata__"]
    assert isinstance(metadata, dict)
    header["__metadata__"] = dict(reversed(list(metadata.items())))
    legacy = _with_header(
        json.dumps(dict(reversed(list(header.items()))), separators=(",", ":")).encode(),
        data,
    )

    canonical = canonicalize_safetensors_bytes(legacy)
    _canonical_header, canonical_data = _header_parts(canonical)

    assert canonical == deterministic_safetensors_bytes(
        _tensors(), metadata={"zeta": "1", "alpha": "2", "mid": "3"}
    )
    assert canonicalize_safetensors_bytes(canonical) == canonical
    assert canonical_data == data


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "too short"),
        (struct.pack("<Q", 8) + b"{}", "invalid header length"),
        (_with_header(b"[]"), "non-object header"),
        (_with_header(b'{"x":{},"x":{}}'), "duplicate JSON object key"),
    ],
)
def test_canonicalize_rejects_malformed_or_ambiguous_headers(payload: bytes, message: str):
    with pytest.raises(ValueError, match=message):
        canonicalize_safetensors_bytes(payload)


def test_header_is_canonical_sorted_json_padded_to_8():
    payload = deterministic_safetensors_bytes(_tensors(), metadata={"z": "1", "a": "2"})
    header_len = int.from_bytes(payload[:8], "little")
    assert header_len % 8 == 0
    header_bytes = payload[8 : 8 + header_len]
    stripped = header_bytes.rstrip(b" ")
    header = json.loads(stripped)
    assert list(header) == sorted(header)
    assert (
        stripped
        == json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def test_non_ascii_metadata_round_trips():
    payload = deterministic_safetensors_bytes(_tensors(), metadata={"desc": "représentant→amax"})
    loaded_header = json.loads(payload[8 : 8 + int.from_bytes(payload[:8], "little")].rstrip(b" "))
    assert loaded_header["__metadata__"]["desc"] == "représentant→amax"


def test_no_metadata_serializes_to_bytes():
    assert isinstance(deterministic_safetensors_bytes(_tensors()), bytes)


def test_empty_tensor_dict_serializes_canonically():
    payload = deterministic_safetensors_bytes({})
    assert isinstance(payload, bytes)
    header_len = int.from_bytes(payload[:8], "little")
    header_bytes = payload[8 : 8 + header_len]
    stripped = header_bytes.rstrip(b" ")
    assert stripped == b"{}"
    assert json.loads(stripped) == {}
    assert load(payload) == {}


def test_descriptor_bound_snapshot_loads_tensors_and_metadata_from_one_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.safetensors"
    path.write_bytes(deterministic_safetensors_bytes(_tensors(), metadata={"format": "v1"}))

    tensors, metadata = load_safetensors_file_snapshot(path)

    assert metadata == {"format": "v1"}
    assert set(tensors) == set(_tensors())
    for key, expected in _tensors().items():
        assert t.equal(tensors[key], expected)


def test_descriptor_bound_snapshot_rejects_top_level_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.safetensors"
    link = tmp_path / "link.safetensors"
    target.write_bytes(deterministic_safetensors_bytes(_tensors()))
    link.symlink_to(target)

    with pytest.raises(ImmutableArtifactError, match="not a regular file"):
        load_safetensors_file_snapshot(link)


@pytest.mark.parametrize(
    "header",
    [
        b'{"__metadata__":{"format":1}}',
        b'{"__metadata__":{"format":null}}',
        b'{"__metadata__":{"format":{}}}',
    ],
)
def test_metadata_parser_rejects_non_string_values(header: bytes) -> None:
    with pytest.raises(ValueError, match="metadata keys and values must be strings"):
        read_safetensors_metadata_from_bytes(_with_header(header))


def test_metadata_parser_distinguishes_absent_empty_and_nonobject_metadata() -> None:
    assert read_safetensors_metadata_from_bytes(_with_header(b"{}")) is None
    assert read_safetensors_metadata_from_bytes(_with_header(b'{"__metadata__":{}}')) == {}
    with pytest.raises(ValueError, match="non-object __metadata__"):
        read_safetensors_metadata_from_bytes(_with_header(b'{"__metadata__":[]}'))


def test_strict_header_rejects_nonfinite_json_constant() -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        canonicalize_safetensors_bytes(_with_header(b'{"value":NaN}'))


def test_serializer_wraps_impossible_invalid_internal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safetensors_io, "save", lambda *_args, **_kwargs: b"invalid")
    with pytest.raises(ValueError, match="serializer returned invalid bytes"):
        deterministic_safetensors_bytes({})
