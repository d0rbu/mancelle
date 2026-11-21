"""Reusable phantom domains for Hugging Face bucket transport."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated

from beartype.vale import Is
from phantom import Phantom

__all__ = [
    "HfBucketId",
    "HfBucketIdLike",
    "NormalizedRemotePrefix",
    "NormalizedRemotePrefixLike",
]

_HF_BUCKET_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"
)


def _is_hf_bucket_id(value: object) -> bool:
    return isinstance(value, str) and _HF_BUCKET_ID_RE.fullmatch(value) is not None


def _is_normalized_remote_prefix(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and not value.endswith("/")
        and not any(part in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == value
    )


class HfBucketId(str, Phantom, predicate=_is_hf_bucket_id):
    """A Hugging Face bucket identifier in canonical ``owner/name`` form."""


class NormalizedRemotePrefix(str, Phantom, predicate=_is_normalized_remote_prefix):
    """A trimmed, normalized, relative POSIX bucket prefix."""


HfBucketIdLike = Annotated[
    str,
    Is[lambda value: isinstance(value, HfBucketId)],
]
NormalizedRemotePrefixLike = Annotated[
    str,
    Is[lambda value: isinstance(value, NormalizedRemotePrefix)],
]
