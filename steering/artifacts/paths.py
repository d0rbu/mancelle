"""HF storage-bucket URI construction."""

from __future__ import annotations


def BucketURI(*, owner: str, bucket: str, path: str = "") -> str:  # noqa: N802
    """Build an ``hf://buckets`` URI without a trailing slash at bucket root."""
    root = f"hf://buckets/{owner}/{bucket}"
    return f"{root}/{path}" if path else root
