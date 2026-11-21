"""Private soft-prompt tensor transport shared by root and vendored Python."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import torch as t
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file


def validate_soft_embeddings(embeddings: t.Tensor, shape: tuple[int, int]) -> None:
    if (
        not isinstance(embeddings, t.Tensor)
        or embeddings.dtype is not t.float32
        or tuple(embeddings.shape) != shape
        or not bool(t.isfinite(embeddings).all())
    ):
        raise ValueError(f"soft embeddings must be finite float32 with shape {shape}")


def write_soft_embeddings(path: Path, embeddings: t.Tensor) -> None:
    """Publish a fresh inode; prior mapped tensors remain valid after replacement."""
    tensor = embeddings.detach().cpu().contiguous()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        save_file({"embeddings": tensor}, str(temporary))
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_soft_embeddings(*, path: Path, shape: tuple[int, int]) -> t.Tensor:
    """Read the fixed private input file with the prepared cell's tensor schema."""
    try:
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("soft tensor sidecar must be a regular file")
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("soft tensor sidecar must be private to the current user")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embeddings"}:
                raise ValueError("soft tensor sidecar key is invalid")
            embeddings = handle.get_tensor("embeddings")
    except (OSError, SafetensorError) as exc:
        raise ValueError("soft tensor sidecar is unavailable or invalid safetensors") from exc
    validate_soft_embeddings(embeddings, shape)
    return embeddings
