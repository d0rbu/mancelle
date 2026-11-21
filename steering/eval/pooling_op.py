"""PoolingOp enum — isolated to avoid circular imports between
target_conditioning and target_segments."""

from __future__ import annotations

from enum import StrEnum


class PoolingOp(StrEnum):
    BUCKET_MAX = "bucket_max"
    BUCKET_MEAN = "bucket_mean"
