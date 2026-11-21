"""Shared CacheBuilderStage enum used by build_eval_context_cache.py
(and any future cache builder)."""

from __future__ import annotations

from enum import StrEnum


class CacheBuilderStage(StrEnum):
    ENUMERATE = "enumerate"
    CONVERT = "convert"
    VALIDATE = "validate"
