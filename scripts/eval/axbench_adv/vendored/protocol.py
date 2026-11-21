"""Length-prefixed JSON frames shared by the root and vendored interpreters."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping
from typing import Any, BinaryIO

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024
_LENGTH = struct.Struct(">I")


class ProtocolError(Exception):
    """A frame is malformed or violates the transport contract."""


def read_exact(reader: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise ``EOFError``."""

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = reader.read(remaining)
        if not chunk:
            raise EOFError(f"framed pipe closed with {remaining} bytes left to read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_frame(value: Mapping[str, Any]) -> bytes:
    """Encode one canonical JSON-object frame."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProtocolError("frame must be a mapping with string keys")
    try:
        payload = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError("frame is not strict UTF-8 JSON") from exc
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is too large: {len(payload)} bytes")
    return _LENGTH.pack(len(payload)) + payload


def decode_frame(
    reader: BinaryIO,
    *,
    read_exact_fn: Callable[[BinaryIO, int], bytes] = read_exact,
) -> dict[str, Any]:
    """Decode one JSON-object frame using an injectable exact-read primitive."""

    length_bytes = read_exact_fn(reader, _LENGTH.size)
    if len(length_bytes) != _LENGTH.size:
        raise ProtocolError("exact-read callback returned the wrong length")
    payload_length = _LENGTH.unpack(length_bytes)[0]
    if payload_length > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame length {payload_length} exceeds the protocol limit")
    payload = read_exact_fn(reader, payload_length)
    if len(payload) != payload_length:
        raise ProtocolError("exact-read callback returned the wrong payload length")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProtocolError("frame payload must be a JSON object with string keys")
    return value


def write_frame(writer: BinaryIO, value: Mapping[str, Any]) -> None:
    """Write and flush one frame."""

    writer.write(encode_frame(value))
    writer.flush()


__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "decode_frame",
    "encode_frame",
    "read_exact",
    "write_frame",
]
