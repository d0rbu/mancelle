"""Shared pytest fixtures for steering tests."""

from __future__ import annotations

import pytest

import steering.segmentation as _seg
from steering.models import resolve_tokenizer


@pytest.fixture(scope="session")
def gemma_tok():
    """Real gemma-2-2b tokenizer — session-scoped to amortise load cost.

    Tokenizer objects are read-only so session scope is safe across all tests
    that use this fixture.  Previously duplicated with a function-local import
    in each of the roundtrip-seam and feature-shard-splice-build test modules.
    """
    return resolve_tokenizer("gemma-2-2b")


@pytest.fixture
def patch_counter(monkeypatch):
    """Capture record_counter calls in steering.segmentation.

    Used by tests for align_segments_to_decoded_prefix and
    align_char_ranges_to_decoded_prefix to assert on emitted counters.
    """
    captured: list[tuple[str, float, dict]] = []

    if hasattr(_seg, "record_counter"):
        def fake(name, value=1, **attrs):
            captured.append((name, float(value), attrs))

        monkeypatch.setattr(_seg, "record_counter", fake)
    return captured
