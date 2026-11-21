"""End-to-end: build a cache for a samples-suite cfg, then load via provider
and assert the keys match what the training script will look up at runtime.

Stubs out feature-extraction; uses synthetic feature shards so the test
runs in seconds. The point is the keys/identity, not the SAE forward.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    ContextFeatureSegment,
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.context_enumeration import enumerate_sample_prompt_contexts
from steering.refined_types import PositiveInt
from tests.conftest import build_tiny_tokenizer

_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(16)
_NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)
_SHARD_SIZE: PositiveInt = PositiveInt.parse(4)


def _make_synthetic_cache(out_dir: Path, records, identity: CacheIdentity) -> Path:
    """Produce a one-segment-per-record cache so provider.get works."""
    entries = [
        ContextCacheEntry(
            token_ids=r.token_ids,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(0,), buckets=(1,),
                    char_start=0, char_end=max(1, len(r.context_text)),
                ),
            ),
        )
        for r in records
    ]
    return save_context_cache(
        out_dir, entries,
        identity=identity,
        n_sae_features=_N_SAE_FEATURES, num_buckets=_NUM_BUCKETS,
        metadata={"model_max_seq_len": 64},
        shard_size=_SHARD_SIZE, overwrite=True,
    )


def test_samples_suite_keys_match_training_script_lookup_keys():
    """The training script will tokenize each prompt and look up by
    token_context_key(tokens). Cache built from the same prompts must
    contain those keys."""
    tokenizer = build_tiny_tokenizer()
    prompts = ["alpha", "beta"]
    records = enumerate_sample_prompt_contexts(
        prompts=prompts, tokenizer=tokenizer, max_seq_len=64,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        cache_dir = context_length_cache_dir(cache_root, 64, hash_leaf="b" * 16)
        identity = CacheIdentity(
            model_id="test-gemma", sae_id="test-sae",
            layer=8, pooling="max_pool",
            feature_selection_hash="b" * 64,
        )
        _make_synthetic_cache(cache_dir, records, identity)

        provider = SafetensorsContextFeatureProvider(
            cache_dir, identity=identity,
            expected_n_sae_features=_N_SAE_FEATURES, expected_num_buckets=_NUM_BUCKETS,
            expected_metadata={"model_max_seq_len": 64},
        )

        for prompt in prompts:
            tokens = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
            segments = provider.get(list(tokens))  # ty: ignore[invalid-argument-type]
            assert segments is not None, f"prompt {prompt!r} missing from cache"
            assert len(segments) == 1


def test_cache_miss_when_prompt_edited_after_cache_build():
    """Edit-the-prompt -> KeyError -> operator rebuilds. Pin the
    no-silent-staleness invariant."""
    tokenizer = build_tiny_tokenizer()
    records_old = enumerate_sample_prompt_contexts(
        prompts=["alpha"], tokenizer=tokenizer, max_seq_len=64,
    )
    records_new = enumerate_sample_prompt_contexts(
        prompts=["alpha modified"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert records_old[0].key != records_new[0].key
