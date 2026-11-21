from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import cast

import pytest
from omegaconf import OmegaConf

import scripts.eval.build_eval_context_cache as builder
from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    context_length_cache_dir,
    feature_selection_hash_leaf,
    save_context_cache,
)
from steering.eval.eval_cache_paths import default_target_cache_dir
from steering.refined_types import PositiveInt
from tests.conftest import MockTokenizer
from tests.statistics_doubles import fixed_statistics_double

HASH_A = "a" * 64
HASH_B = "b" * 64
_CONTEXT_LEN = PositiveInt.parse(1_024)
_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(8)
_NUM_BUCKETS: PositiveInt = PositiveInt.parse(2)


def _identity(feature_hash: str = HASH_A) -> CacheIdentity:
    return CacheIdentity(
        model_id="gemma-2-2b",
        sae_id="gemmascope-residual-l8-16k",
        layer=8,
        pooling="max_pool",
        feature_selection_hash=feature_hash,
    )


def test_context_length_cache_dir_accepts_root_cseq_and_resolved_leaf(tmp_path):
    leaf = "a" * 16
    assert context_length_cache_dir(tmp_path / "cache", 1024, hash_leaf=leaf) == (
        tmp_path / "cache" / "c1024" / leaf
    )
    assert context_length_cache_dir(tmp_path / "cache" / "c1024", 1024, hash_leaf=leaf) == (
        tmp_path / "cache" / "c1024" / leaf
    )
    assert context_length_cache_dir(
        tmp_path / "cache" / "c1024" / leaf,
        1024,
        hash_leaf=leaf,
    ) == (tmp_path / "cache" / "c1024" / leaf)

    with pytest.raises(TypeError, match="hash_leaf"):
        context_length_cache_dir(tmp_path / "cache", 1024, hash_leaf=cast(str, 123))
    with pytest.raises(ValueError, match="layout mismatch"):
        context_length_cache_dir(
            tmp_path / "cache" / "c1024" / ("b" * 16),
            1024,
            hash_leaf=leaf,
        )


@pytest.mark.parametrize("bad_hash", [None, "h", "g" * 64, "a" * 63, "a" * 65])
def test_cache_identity_rejects_missing_or_invalid_hash(bad_hash):
    with pytest.raises((TypeError, ValueError), match="feature_selection_hash"):
        CacheIdentity(
            model_id="gemma-2-2b",
            sae_id="gemmascope-residual-l8-16k",
            layer=8,
            pooling="max_pool",
            feature_selection_hash=bad_hash,  # type: ignore[arg-type]
        )


def test_default_target_cache_dir_requires_hash_leaf_layout(tmp_path):
    path = default_target_cache_dir(
        suite="core_target",
        sae_identity=_identity("0123456789abcdef" + "0" * 48),
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    assert path.parts[-2:] == ("c1024", "0123456789abcdef")
    assert "no_hash_pinned" not in path.parts


def test_builder_writes_where_default_target_cache_dir_resolves(tmp_path, monkeypatch):
    identity = _identity(HASH_A)
    target_dir = default_target_cache_dir(
        suite="core_target",
        sae_identity=identity,
        model_max_seq_len=_CONTEXT_LEN,
        base_dir=tmp_path,
    )
    output_root = target_dir.parent.parent
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_text(
        json.dumps(
            {
                "suite": "core_target",
                "token_ids": [0, 1, 2],
                "context_text": "abc",
                "metadata": {},
            }
        )
        + "\n"
    )
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        lambda *args, **kwargs: SimpleNamespace(feature_mask=None),
    )
    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        lambda **kwargs: [ContextCacheEntry.from_sequences([0, 1, 2], [])],
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())

    cfg = OmegaConf.create(
        {
            "contexts_file": str(contexts),
            "feature_dir": str(feature_path.parent),
            "stats_path": str(tmp_path / "stats.safetensors"),
            "output_dir": str(output_root),
            "model_max_seq_len": 1024,
            "n_sae_features": 8,
            "num_buckets": 2,
            "source_bos_char_len": 0,
            "feature_selection": {"threshold_type": "segment_topk", "threshold": 1},
            "negative_selection": None,
            "identity": identity.as_dict(),
            "metadata": {},
            "shard_size": 16,
            "overwrite": False,
            "zstd_level": None,
        }
    )

    assert builder.build_cache_from_config(cfg) == target_dir / "manifest.json"


def test_two_hashes_coexist_under_one_context_parent(tmp_path):
    for feature_hash in (HASH_A, HASH_B):
        identity = _identity(feature_hash)
        cache_dir = context_length_cache_dir(
            tmp_path / "cache",
            1024,
            hash_leaf=feature_selection_hash_leaf(feature_hash),
        )
        save_context_cache(
            cache_dir,
            [ContextCacheEntry.from_sequences([0, 1, int(feature_hash[0], 16)], [])],
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            metadata={"model_max_seq_len": 1024},
            identity=identity,
        )
        assert (cache_dir / "manifest.json").is_file()


def test_legacy_cseq_manifest_fails_layout_validation(tmp_path):
    identity = _identity(HASH_A)
    legacy_dir = tmp_path / "cache" / "c1024"

    with pytest.raises(ValueError, match="layout mismatch"):
        save_context_cache(
            legacy_dir,
            [ContextCacheEntry.from_sequences([0, 1, 2], [])],
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            identity=identity,
        )


def test_context_dir_property_bijection_fixed_seed(tmp_path):
    rng = random.Random(1234)
    for _ in range(200):
        feature_hash = "".join(rng.choice("0123456789abcdef") for _ in range(64))
        leaf = feature_hash[:16]
        identity = _identity(feature_hash)
        assert feature_selection_hash_leaf(feature_hash) == leaf
        assert context_length_cache_dir(tmp_path / "cache", 1024, hash_leaf=leaf) == (
            tmp_path / "cache" / "c1024" / leaf
        )
        assert (
            default_target_cache_dir(
                suite="chat_target",
                sae_identity=identity,
                model_max_seq_len=_CONTEXT_LEN,
                base_dir=tmp_path,
            ).name
            == leaf
        )
