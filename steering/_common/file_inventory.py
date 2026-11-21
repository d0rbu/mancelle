"""Uncached content inventories with a file-state fence around every read."""

import hashlib
from pathlib import Path
from typing import Any


def file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def file_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        before = file_signature(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if before != file_signature(path):
            raise ValueError(f"input changed while hashing {path}")
        result.append({"path": str(path.resolve()), "size": before[2], "sha256": digest})
    return result
