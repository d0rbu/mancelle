"""Stream one zstd-compressed packed shard to stdout as decoded bytes.

A stalled non-erroring local tee write can consume the bounded worker deadline;
best-effort covers I/O errors, while local-ext4 hangs are an accepted risk.
"""

import sys
from pathlib import Path
from typing import IO, cast

import zstandard


class _TeeReader:
    """Mirror compressed reads to a best-effort local file."""

    def __init__(self, source: IO[bytes], tee: IO[bytes] | None):
        self.source = source
        self.tee = tee

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        if chunk and self.tee is not None:
            try:
                written = self.tee.write(chunk)
                if written != len(chunk):
                    raise OSError(f"short tee write: {written} != {len(chunk)}")
            except OSError:
                try:
                    self.tee.close()
                except OSError:
                    pass
                self.tee = None
        return chunk


if len(sys.argv) not in (2, 3):
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} SHARD_PATH [TEE_PATH]")

with Path(sys.argv[1]).open("rb") as source:
    tee_handle: IO[bytes] | None = None
    try:
        if len(sys.argv) == 3:
            try:
                tee_handle = Path(sys.argv[2]).open("wb")  # noqa: SIM115
            except OSError:
                pass
        reader = _TeeReader(source, tee_handle)
        zstandard.ZstdDecompressor().copy_stream(
            cast(IO[bytes], reader),
            sys.stdout.buffer,
        )
    finally:
        if tee_handle is not None:
            try:
                tee_handle.close()
            except OSError:
                pass
sys.stdout.buffer.flush()
