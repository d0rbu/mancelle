"""Unit tests for the strict cross-interpreter JSON framing."""

from __future__ import annotations

import io
import json
import struct
from typing import Any

import pytest

from scripts.eval.axbench_adv.vendored import protocol


class ChunkReader:
    def __init__(self, value: bytes, chunk_size: int = 1) -> None:
        self.value = value
        self.chunk_size = chunk_size

    def read(self, size: int) -> bytes:
        take = min(size, self.chunk_size, len(self.value))
        result, self.value = self.value[:take], self.value[take:]
        return result


class FlushWriter(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushed = False

    def flush(self) -> None:
        self.flushed = True
        super().flush()


def _raw_frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def test_frame_encoding_is_canonical_utf8_and_roundtrips_partial_reads() -> None:
    value = {"version": 1, "text": "café", "nested": {"z": True}, "items": [1, None]}
    expected_payload = '{"items":[1,null],"nested":{"z":true},"text":"café","version":1}'.encode()
    encoded = protocol.encode_frame(value)
    assert encoded == _raw_frame(expected_payload)
    assert protocol.decode_frame(
        ChunkReader(encoded)  # ty: ignore[invalid-argument-type]
    ) == value


@pytest.mark.parametrize(
    "value",
    [
        {1: "not-a-string-key"},
        {"bad": float("nan")},
        {"bad": object()},
        {"bad": "\ud800"},
        ["not", "a", "mapping"],
    ],
)
def test_encode_rejects_every_non_strict_json_shape(value: Any) -> None:
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_frame(value)


def test_encode_and_decode_enforce_frame_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 2)
    with pytest.raises(protocol.ProtocolError, match="too large"):
        protocol.encode_frame({"x": 1})
    with pytest.raises(protocol.ProtocolError, match="exceeds"):
        protocol.decode_frame(io.BytesIO(struct.pack(">I", 3)))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"\xff", "UTF-8 JSON"),
        (b"{", "UTF-8 JSON"),
        (json.dumps([1, 2]).encode(), "JSON object"),
    ],
)
def test_decode_rejects_invalid_utf8_json_and_nonobject(payload: bytes, match: str) -> None:
    with pytest.raises(protocol.ProtocolError, match=match):
        protocol.decode_frame(io.BytesIO(_raw_frame(payload)))


def test_read_exact_reports_clean_and_partial_eof() -> None:
    with pytest.raises(EOFError, match="4 bytes"):
        protocol.read_exact(io.BytesIO(), 4)
    with pytest.raises(EOFError, match="2 bytes"):
        protocol.read_exact(io.BytesIO(b"ab"), 4)
    assert protocol.read_exact(io.BytesIO(), 0) == b""


@pytest.mark.parametrize("bad_call", [1, 2])
def test_decode_rejects_broken_exact_read_callback(bad_call: int) -> None:
    calls = 0

    def broken(reader: io.BytesIO, size: int) -> bytes:
        nonlocal calls
        calls += 1
        value = protocol.read_exact(reader, size)
        return value[:-1] if calls == bad_call else value

    with pytest.raises(protocol.ProtocolError, match="wrong"):
        protocol.decode_frame(
            io.BytesIO(protocol.encode_frame({"ok": True})),
            read_exact_fn=broken,  # ty: ignore[invalid-argument-type]
        )


def test_write_frame_flushes_exact_encoded_bytes() -> None:
    writer = FlushWriter()
    protocol.write_frame(writer, {"ok": True})
    assert writer.flushed is True
    assert writer.getvalue() == protocol.encode_frame({"ok": True})
