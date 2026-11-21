import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from steering.eval.axbench_base.synth import response_parser
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
    get_extraction_tokenizer,
)


def _clear_extraction_tokenizer_cache() -> None:
    with response_parser._TOKENIZER_CACHE_LOCK:
        response_parser._get_extraction_tokenizer_cached.cache_clear()


def test_get_extraction_tokenizer_loads(monkeypatch):
    """The tokenizer accessor loads and returns the requested tokenizer."""
    fake_tok = MagicMock()
    fake_tok.encode.return_value = [1, 2, 3]
    _clear_extraction_tokenizer_cache()
    monkeypatch.setattr(
        response_parser.AutoTokenizer, "from_pretrained", lambda tokenizer_id: fake_tok,
    )
    tok = get_extraction_tokenizer("google/gemma-2-2b")
    assert tok is fake_tok
    # Confirm the encode path works on a plain string.
    ids = tok.encode("hello world", add_special_tokens=False)
    assert len(ids) >= 2
    _clear_extraction_tokenizer_cache()


def test_get_extraction_tokenizer_cached_per_id(monkeypatch):
    """The LRU cache is keyed per tokenizer_id: same ID returns the same object,
    different IDs return different objects, and the loader is called exactly once
    per unique ID."""
    call_count = {"n": 0}

    def _fake_loader(tokenizer_id):
        call_count["n"] += 1
        return MagicMock(name=f"tok_{tokenizer_id}_{call_count['n']}")

    _clear_extraction_tokenizer_cache()
    monkeypatch.setattr(
        response_parser.AutoTokenizer, "from_pretrained", _fake_loader,
    )
    a1 = get_extraction_tokenizer("model-A")
    a2 = get_extraction_tokenizer("model-A")
    b = get_extraction_tokenizer("model-B")
    assert a1 is a2  # same ID → same cached instance
    assert a1 is not b  # different ID → different instance
    assert call_count["n"] == 2  # loader called once per unique ID
    _clear_extraction_tokenizer_cache()


def test_get_extraction_tokenizer_default_and_explicit_default_share_cache(
    monkeypatch,
):
    call_count = {"n": 0}

    def _fake_loader(tokenizer_id):
        call_count["n"] += 1
        return MagicMock(name=f"tok_{tokenizer_id}_{call_count['n']}")

    _clear_extraction_tokenizer_cache()
    monkeypatch.setattr(
        response_parser.AutoTokenizer, "from_pretrained", _fake_loader,
    )
    implicit = get_extraction_tokenizer()
    explicit = get_extraction_tokenizer(DEFAULT_EXTRACTION_TOKENIZER_ID)

    assert implicit is explicit
    assert call_count["n"] == 1
    _clear_extraction_tokenizer_cache()


def test_get_extraction_tokenizer_concurrent_first_use_single_flight(
    monkeypatch,
):
    calls: list[str] = []
    entered_loader = threading.Event()
    release_loader = threading.Event()
    lock = threading.Lock()

    def _fake_loader(tokenizer_id):
        with lock:
            calls.append(tokenizer_id)
        entered_loader.set()
        assert release_loader.wait(timeout=2.0)
        return MagicMock(name=f"tok_{tokenizer_id}")

    _clear_extraction_tokenizer_cache()
    monkeypatch.setattr(
        response_parser.AutoTokenizer, "from_pretrained", _fake_loader,
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        first = executor.submit(get_extraction_tokenizer, "same-tokenizer")
        assert entered_loader.wait(timeout=2.0)
        rest = [
            executor.submit(get_extraction_tokenizer, "same-tokenizer")
            for _ in range(4)
        ]
        # Give the remaining workers a chance to contend on the same cold key.
        time.sleep(0.05)
        release_loader.set()
        results = [first.result(timeout=2.0)] + [
            future.result(timeout=2.0) for future in rest
        ]

    assert calls == ["same-tokenizer"]
    assert all(result is results[0] for result in results)
    _clear_extraction_tokenizer_cache()
