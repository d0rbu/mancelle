"""Extraction tokenizer loader and Stage 1 refusal type.

Reference: spec 3.5.2.

The *extraction* tokenizer is used to count continuation length in the
synth corpus; it must match the tokenizer that was used to train the
SAE that our feature extraction pipeline runs against. By default this
is ``google/gemma-2-2b`` (to match the gemmascope-res-16k SAE), but
any HuggingFace ``AutoTokenizer``-loadable model ID is accepted.
"""

from __future__ import annotations

import functools
import threading

from transformers import AutoTokenizer

from steering.eval.axbench_base.types import AxBenchBaseError

DEFAULT_EXTRACTION_TOKENIZER_ID = "google/gemma-2-2b"


class LlmSpecialTokenRefusal(AxBenchBaseError):
    """An LLM continuation response encoded a special-token id.

    The authenticated continuation protocol raises this distinct subtype so
    retry loops can reject one polluted generation without swallowing unrelated
    :class:`AxBenchBaseError` failures.
    """


@functools.lru_cache(maxsize=8)
def _get_extraction_tokenizer_cached(tokenizer_id: str):
    return AutoTokenizer.from_pretrained(tokenizer_id)


_TOKENIZER_CACHE_LOCK = threading.RLock()


def get_extraction_tokenizer(
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
):
    """Return the cached extraction tokenizer for ``tokenizer_id``.

    The returned tokenizer is used to count continuation length because the
    synth corpus is measured in extraction-tokenizer tokens (which match the SAE
    vocabulary), not nanochat tokens.

    The LRU cache is keyed on the normalized string tokenizer ID so the default
    and an explicitly supplied default ID share one entry. The lock makes
    first-use cache misses single-flight, avoiding duplicate concurrent
    HuggingFace loads for the same ID.
    """
    normalized_id = str(tokenizer_id)
    with _TOKENIZER_CACHE_LOCK:
        return _get_extraction_tokenizer_cached(normalized_id)
