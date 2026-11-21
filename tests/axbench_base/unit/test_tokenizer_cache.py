"""Tests for the process-local AxBench nanochat tokenizer cache."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from steering.eval.axbench_base import tokenizer_cache
from steering.eval.axbench_base.synth import negative_mode, positive_mode
from steering.eval.axbench_base.synth.negative_mode import _exact_output_start as negative_boundary
from steering.eval.axbench_base.synth.positive_mode import _exact_output_start as positive_boundary
from steering.segmentation import compute_tokenizer_checksum
from tests.axbench_base.tokenizer_cache import clear_tokenizer_cache


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    clear_tokenizer_cache()
    yield
    clear_tokenizer_cache()


class _CharTokenizer:
    """Small tokenizer with the same boundary-facing API as nanochat."""

    def __init__(self, bos_id: int = 1) -> None:
        self.bos_id = bos_id

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_get_nanochat_tokenizer_loads_once_and_preserves_identity(monkeypatch):
    calls: list[None] = []
    tokenizer = _CharTokenizer()

    def _loader() -> _CharTokenizer:
        calls.append(None)
        return tokenizer

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)

    first = tokenizer_cache.get_nanochat_tokenizer()
    second = tokenizer_cache.get_nanochat_tokenizer()

    assert first is tokenizer
    assert second is first
    assert len(calls) == 1


def test_cache_clear_forces_one_new_load(monkeypatch):
    calls = 0

    def _loader() -> _CharTokenizer:
        nonlocal calls
        calls += 1
        return _CharTokenizer(bos_id=calls)

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)

    first = tokenizer_cache.get_nanochat_tokenizer()
    clear_tokenizer_cache()
    second = tokenizer_cache.get_nanochat_tokenizer()

    assert first is not second
    assert calls == 2


def test_loader_exception_is_not_cached(monkeypatch):
    calls = 0

    def _loader() -> _CharTokenizer:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("tokenizer artifact unavailable")
        return _CharTokenizer()

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)

    with pytest.raises(RuntimeError, match="artifact unavailable"):
        tokenizer_cache.get_nanochat_tokenizer()
    loaded = tokenizer_cache.get_nanochat_tokenizer()

    assert isinstance(loaded, _CharTokenizer)
    assert calls == 2


def test_concurrent_cold_load_is_single_flight(monkeypatch):
    calls: list[None] = []
    entered_loader = threading.Event()
    release_loader = threading.Event()
    tokenizer = _CharTokenizer()

    def _loader() -> _CharTokenizer:
        calls.append(None)
        entered_loader.set()
        assert release_loader.wait(timeout=2.0)
        return tokenizer

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)
    with ThreadPoolExecutor(max_workers=5) as executor:
        first = executor.submit(tokenizer_cache.get_nanochat_tokenizer)
        assert entered_loader.wait(timeout=2.0)
        rest = [
            executor.submit(tokenizer_cache.get_nanochat_tokenizer)
            for _ in range(4)
        ]
        time.sleep(0.05)
        release_loader.set()
        results = [first.result(timeout=2.0)] + [
            future.result(timeout=2.0) for future in rest
        ]

    assert len(calls) == 1
    assert all(result is tokenizer for result in results)


def test_stage1_modules_share_the_cached_loader(monkeypatch):
    calls = 0

    def _loader() -> _CharTokenizer:
        nonlocal calls
        calls += 1
        return _CharTokenizer()

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)

    assert positive_mode.get_tokenizer() is negative_mode.get_tokenizer()
    assert positive_boundary("prefix", " output") == 7
    assert negative_boundary("prefix", " output") == 7
    assert calls == 1


def test_cached_and_fresh_tokenizers_have_identical_checksum(monkeypatch):
    def _loader() -> _CharTokenizer:
        return _CharTokenizer()

    monkeypatch.setattr(tokenizer_cache, "_nanochat_get_tokenizer", _loader)
    cached = tokenizer_cache.get_nanochat_tokenizer()
    cached_checksum = compute_tokenizer_checksum(cached)

    clear_tokenizer_cache()
    fresh = tokenizer_cache.get_nanochat_tokenizer()

    assert compute_tokenizer_checksum(fresh) == cached_checksum


def test_cached_boundaries_match_fresh_boundaries(monkeypatch):
    pairs = [
        ("prefix", " output"),
        ("café ", "日本語"),
        ("emoji: ", "🎉"),
    ]

    monkeypatch.setattr(
        tokenizer_cache,
        "_nanochat_get_tokenizer",
        lambda: _CharTokenizer(),
    )
    cached_boundaries = [positive_boundary(prefix, output) for prefix, output in pairs]

    clear_tokenizer_cache()
    fresh_boundaries = [positive_boundary(prefix, output) for prefix, output in pairs]

    assert fresh_boundaries == cached_boundaries
