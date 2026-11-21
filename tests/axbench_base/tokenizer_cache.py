"""Test-only lifecycle controls for the process-local tokenizer cache."""

from steering.eval.axbench_base import tokenizer_cache


def clear_tokenizer_cache() -> None:
    """Reset the cache while holding the same lock as production reads."""
    with tokenizer_cache._TOKENIZER_CACHE_LOCK:
        tokenizer_cache._get_nanochat_tokenizer_cached.cache_clear()
