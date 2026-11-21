from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.eval.context_cache_io as cache_io
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.feature_vocab import META_KEY as FEATURE_VOCAB_META_KEY
from steering.eval.clustered_cache_seed_contract import CheckpointSeedProvenance
from steering.eval.context_cache_build import (
    build_cache_entries_from_extracted_feature_shards,
)
from steering.eval.context_cache_io import (
    CLUSTER_SELECTION_SEMANTICS,
    REMOVED_FEATURES_MANIFEST_KEY,
    REMOVED_FEATURES_MARKER,
    SOURCE_SEGMENT_MODE_KEY,
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    load_context_cache_manifest,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from steering.refined_types import NonNegativeInt, PositiveInt
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_SHARD_SIZE_ONE: PositiveInt = PositiveInt.parse(1)


def _entry(tokens=(0, 2, 3)) -> ContextCacheEntry:
    return ContextCacheEntry.from_sequences(
        tokens,
        [
            ContextFeatureSegment((1, 2), (0, 1), 0, 2),
            ContextFeatureSegment((3,), (2,), 2, 3),
        ],
    )


def test_context_length_cache_dir_requires_root_and_positive_context_len(tmp_path):
    assert context_length_cache_dir(tmp_path / "cache", 1024) == (tmp_path / "cache" / "c1024")
    assert context_length_cache_dir(tmp_path / "cache", "2048") == (tmp_path / "cache" / "c2048")

    with pytest.raises(ValueError, match="positive integer"):
        context_length_cache_dir(tmp_path / "cache", 0)
    with pytest.raises(ValueError, match="positive integer"):
        context_length_cache_dir(tmp_path / "cache", "bad")

    assert context_length_cache_dir(tmp_path / "cache" / "c1024", 1024) == (
        tmp_path / "cache" / "c1024"
    )
    assert context_length_cache_dir(
        tmp_path / "cache",
        1024,
        hash_leaf="a" * 16,
    ) == (tmp_path / "cache" / "c1024" / ("a" * 16))
    assert context_length_cache_dir(
        tmp_path / "cache" / "c1024",
        1024,
        hash_leaf="a" * 16,
    ) == (tmp_path / "cache" / "c1024" / ("a" * 16))
    assert context_length_cache_dir(
        tmp_path / "cache" / "c1024" / ("a" * 16),
        1024,
        hash_leaf="a" * 16,
    ) == (tmp_path / "cache" / "c1024" / ("a" * 16))
    with pytest.raises(TypeError, match="hash_leaf"):
        context_length_cache_dir(tmp_path / "cache", 1024, hash_leaf=123)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="layout mismatch"):
        context_length_cache_dir(tmp_path / "cache" / "c2048", 1024)
    with pytest.raises(ValueError, match="layout mismatch"):
        context_length_cache_dir(
            tmp_path / "cache" / "c1024" / ("b" * 16),
            1024,
            hash_leaf="a" * 16,
        )


def test_safetensors_context_cache_roundtrip(tmp_path):
    entry = _entry()
    save_context_cache(
        tmp_path,
        [entry],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"tokenizer_checksum": "tok"},
        identity=TEST_CACHE_IDENTITY,
    )

    provider = SafetensorsContextFeatureProvider(
        tmp_path,
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=TEST_CACHE_NUM_BUCKETS,
        expected_metadata={"tokenizer_checksum": "tok"},
        identity=TEST_CACHE_IDENTITY,
    )

    assert len(provider) == 1
    assert provider.get(entry.token_ids) == entry.segments
    assert provider.get([999]) is None


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("n_sae_features", "num_buckets", "shard_size")),
    value=st.one_of(st.integers(max_value=0), st.booleans()),
)
def test_save_context_cache_rejects_complete_invalid_integer_domains(
    field: str,
    value: int,
) -> None:
    kwargs: dict[str, Any] = {
        "n_sae_features": TEST_CACHE_N_SAE_FEATURES,
        "num_buckets": TEST_CACHE_NUM_BUCKETS,
        "shard_size": PositiveInt.parse(1),
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        save_context_cache(
            "/unused-invalid-context-cache",
            [],
            identity=TEST_CACHE_IDENTITY,
            **kwargs,
        )


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        ("n_sae_features", "num_buckets", "source_bos_char_len")
    ),
    value=st.one_of(st.integers(max_value=-1), st.booleans()),
)
def test_build_cache_entries_rejects_complete_invalid_integer_domains(
    field: str,
    value: int,
) -> None:
    kwargs: dict[str, Any] = {
        "n_sae_features": TEST_CACHE_N_SAE_FEATURES,
        "num_buckets": TEST_CACHE_NUM_BUCKETS,
        "source_bos_char_len": NonNegativeInt.parse(0),
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        build_cache_entries_from_extracted_feature_shards(
            context_records=[],
            feature_shard_paths=[],
            pipeline=object.__new__(FeaturePipelineConfig),
            tokenizer=None,
            **kwargs,
        )


def test_save_context_cache_requires_overwrite_for_existing_manifest(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_context_cache(
            tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
        )

    save_context_cache(
        tmp_path,
        [_entry((0, 4, 5))],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        overwrite=True,
        identity=TEST_CACHE_IDENTITY,
    )
    provider = SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)
    assert provider.get((0, 4, 5)) is not None


def test_save_context_cache_detects_digest_collision(tmp_path):
    entries = [_entry((0, 1)), _entry((0, 2))]

    with (
        patch(
            "steering.eval.context_cache_io._scoped_row_digest",
            return_value=b"x" * 32,
        ),
        pytest.raises(ContextCacheValidationError, match="ambiguous cache"),
    ):
        save_context_cache(
            tmp_path, entries, n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
        )


def test_save_context_cache_rejects_duplicate_token_sequences(tmp_path):
    with pytest.raises(ContextCacheValidationError, match="Duplicate context token"):
        save_context_cache(
            tmp_path,
            [_entry((0, 1)), _entry((0, 1))],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )


def test_provider_detects_internal_hash_token_mismatch(tmp_path):
    entry = _entry()
    save_context_cache(
        tmp_path, [entry], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    provider = SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)
    digest = next(iter(provider._rows))
    provider._rows[digest] = ((999,), entry.segments)

    with pytest.raises(ContextCacheValidationError, match="token IDs differed"):
        provider.get(entry.token_ids)


def test_context_cache_rejects_invalid_feature_and_bucket(tmp_path):
    bad_feature = ContextCacheEntry.from_sequences(
        [0],
        [ContextFeatureSegment((9,), (0,), 0, 1)],
    )
    with pytest.raises(ContextCacheValidationError, match="feature id"):
        save_context_cache(
            tmp_path / "bad_feature",
            [bad_feature],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )

    bad_bucket = ContextCacheEntry.from_sequences(
        [0],
        [ContextFeatureSegment((1,), (4,), 0, 1)],
    )
    with pytest.raises(ContextCacheValidationError, match="bucket"):
        save_context_cache(
            tmp_path / "bad_bucket",
            [bad_bucket],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )


def test_context_cache_rejects_duplicate_features_and_bad_spans(tmp_path):
    duplicate = ContextCacheEntry.from_sequences(
        [0],
        [ContextFeatureSegment((1, 1), (0, 1), 0, 1)],
    )
    with pytest.raises(ContextCacheValidationError, match="duplicate"):
        save_context_cache(
            tmp_path / "duplicate",
            [duplicate],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )

    bad_span = ContextCacheEntry.from_sequences(
        [0],
        [ContextFeatureSegment((1,), (0,), 2, 1)],
    )
    with pytest.raises(ContextCacheValidationError, match="char_end"):
        save_context_cache(
            tmp_path / "bad_span",
            [bad_span],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )


def test_context_cache_manifest_validation(tmp_path):
    save_context_cache(
        tmp_path,
        [_entry()],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"eval_config_hash": "abc"},
        identity=TEST_CACHE_IDENTITY,
    )

    with pytest.raises(ContextCacheValidationError, match="n_sae_features"):
        SafetensorsContextFeatureProvider(
            tmp_path, expected_n_sae_features=9, identity=TEST_CACHE_IDENTITY
        )
    with pytest.raises(ContextCacheValidationError, match="num_buckets"):
        SafetensorsContextFeatureProvider(
            tmp_path, expected_num_buckets=5, identity=TEST_CACHE_IDENTITY
        )
    with pytest.raises(ContextCacheValidationError, match="metadata"):
        SafetensorsContextFeatureProvider(
            tmp_path,
            expected_metadata={"eval_config_hash": "def"},
            identity=TEST_CACHE_IDENTITY,
        )


def test_context_cache_manifest_stamps_removed_features_marker(tmp_path):
    save_context_cache(
        tmp_path,
        [_entry()],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        identity=TEST_CACHE_IDENTITY,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest[REMOVED_FEATURES_MANIFEST_KEY] == REMOVED_FEATURES_MARKER
    assert manifest[SOURCE_SEGMENT_MODE_KEY] == "fixed_tokens"
    load_context_cache_manifest(tmp_path, identity=TEST_CACHE_IDENTITY)

    manifest[REMOVED_FEATURES_MANIFEST_KEY] = {"feature_clustering": "unknown"}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContextCacheValidationError, match="removed-features marker"):
        load_context_cache_manifest(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_manifest_rejects_bad_json_and_shape(tmp_path):
    manifest = tmp_path / "manifest.json"
    tmp_path.mkdir(exist_ok=True)
    manifest.write_text("{")
    with pytest.raises(ContextCacheValidationError, match="not valid JSON"):
        load_context_cache_manifest(tmp_path)

    manifest.write_text("[]")
    with pytest.raises(ContextCacheValidationError, match="must be a JSON object"):
        load_context_cache_manifest(tmp_path)


def test_context_cache_manifest_rejects_bad_fields(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    manifest_path = tmp_path / "manifest.json"
    base = json.loads(manifest_path.read_text())

    cases = [
        ("schema_version", 999, "schema_version"),
        (SOURCE_SEGMENT_MODE_KEY, 1, "must be str"),
        ("format", "wrong", "format"),
        ("n_contexts", -1, "non-negative"),
        ("n_sae_features", 0, "n_sae_features"),
        ("num_buckets", 0, "num_buckets"),
        ("metadata", [], "metadata"),
    ]
    for key, value, match in cases:
        bad = dict(base)
        bad[key] = value
        manifest_path.write_text(json.dumps(bad))
        with pytest.raises(ContextCacheValidationError, match=match):
            load_context_cache_manifest(tmp_path)

    bad = dict(base)
    del bad["format"]
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ContextCacheValidationError, match="missing required"):
        load_context_cache_manifest(tmp_path)

    bad = dict(base)
    del bad[SOURCE_SEGMENT_MODE_KEY]
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ContextCacheValidationError, match="missing required"):
        load_context_cache_manifest(tmp_path)

    bad = dict(base)
    bad["metadata"] = None
    manifest_path.write_text(json.dumps(bad))
    loaded = load_context_cache_manifest(tmp_path)
    assert loaded["metadata"] == {}


def test_context_cache_manifest_rejects_unparseable_source_segment_mode(
    tmp_path: Path,
) -> None:
    # kills: letting an unrecognised source_segment_mode string escape as a
    # bare ValueError instead of the cache-validation contract error.
    save_context_cache(
        tmp_path,
        [_entry()],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        identity=TEST_CACHE_IDENTITY,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[SOURCE_SEGMENT_MODE_KEY] = "sliding_window"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        ContextCacheValidationError,
        match=r"source_segment_mode is invalid: 'sliding_window'",
    ):
        load_context_cache_manifest(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_turn_context_cache_manifest_rejects_before_any_shard_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_context_cache(
        tmp_path,
        [_entry()],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        identity=TEST_CACHE_IDENTITY,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[SOURCE_SEGMENT_MODE_KEY] = "turn_delimited_fixed_tokens"
    manifest_path.write_text(json.dumps(manifest))
    shard_reads: list[Path] = []

    def forbidden_shard_read(path: Path) -> bytes:
        shard_reads.append(path)
        raise AssertionError("turn cache reached shard loading")

    monkeypatch.setattr(cache_io, "load_overlay_shard_bytes", forbidden_shard_read)
    with pytest.raises(
        ContextCacheValidationError,
        match=r"source_segment_mode='turn_delimited_fixed_tokens', expected 'fixed_tokens'",
    ):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)
    assert shard_reads == []


def test_context_cache_manifest_rejects_bad_shards(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    manifest_path = tmp_path / "manifest.json"
    base = json.loads(manifest_path.read_text())

    bad = dict(base)
    bad["shards"] = [1]
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ContextCacheValidationError, match="shard 0"):
        load_context_cache_manifest(tmp_path)

    bad = dict(base)
    bad["shards"] = [dict(base["shards"][0], file="nested/file")]
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ContextCacheValidationError, match="basename"):
        load_context_cache_manifest(tmp_path)

    bad = dict(base)
    shard = dict(base["shards"][0])
    shard["n_contexts"] = "1"
    bad["shards"] = [shard]
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ContextCacheValidationError, match="must be int"):
        load_context_cache_manifest(tmp_path)


def test_context_cache_detects_shard_checksum_mismatch(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError, match="checksum"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_hashes_the_same_shard_bytes_it_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    replacement_dir = tmp_path / "replacement"
    save_context_cache(
        cache_dir,
        [_entry((0, 2, 3)), _entry((4, 5, 6))],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        shard_size=_SHARD_SIZE_ONE,
        identity=TEST_CACHE_IDENTITY,
    )
    save_context_cache(
        replacement_dir,
        [_entry((7, 8, 9))],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        shard_size=_SHARD_SIZE_ONE,
        identity=TEST_CACHE_IDENTITY,
    )
    target_path = sorted(cache_dir.glob("contexts_*.safetensors.zst"))[1]
    replacement_path = next(replacement_dir.glob("contexts_*.safetensors.zst"))
    original_read_bytes = Path.read_bytes
    swapped = False

    def replace_before_decode(path: Path) -> bytes:
        nonlocal swapped
        if path == target_path and not swapped:
            replacement_path.replace(target_path)
            swapped = True
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", replace_before_decode)

    with pytest.raises(ContextCacheValidationError, match="Shard checksum mismatch"):
        SafetensorsContextFeatureProvider(
            cache_dir,
            identity=TEST_CACHE_IDENTITY,
        )
    assert swapped


def test_context_cache_detects_missing_shard_file(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    shard_path = next(tmp_path.glob("contexts_*.safetensors.zst"))
    shard_path.unlink()

    with pytest.raises(FileNotFoundError, match="Context cache shard"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_detects_corrupt_offsets(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    shard_path = next(tmp_path.glob("contexts_*.safetensors.zst"))
    tensors = load_overlay_shard(shard_path)
    tensors["feature_offsets"] = t.tensor([0, 1, 99], dtype=t.int64)
    save_overlay_shard(tensors, shard_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"][0]["sha256"] = _sha256_file(shard_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError, match="terminal offset"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_detects_corrupt_tensors(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    shard_path = next(tmp_path.glob("contexts_*.safetensors.zst"))
    tensors = load_overlay_shard(shard_path)

    corrupt_cases = [
        ("missing", lambda d: d.pop("buckets"), "missing tensor"),
        (
            "bad_hash_shape",
            lambda d: d.__setitem__("context_hash", t.zeros(32, dtype=t.uint8)),
            "context_hash",
        ),
        (
            "char_len",
            lambda d: d.__setitem__("segment_char_ends", t.tensor([1])),
            "lengths differ",
        ),
        (
            "bucket_len",
            lambda d: d.__setitem__("buckets", t.tensor([1])),
            "feature_ids and buckets",
        ),
        (
            "first_offset",
            lambda d: d.__setitem__("context_token_offsets", t.tensor([1, 3], dtype=t.int64)),
            "first offset",
        ),
        (
            "monotonic",
            lambda d: d.__setitem__("feature_offsets", t.tensor([0, 3, 2], dtype=t.int64)),
            "monotonic",
        ),
        (
            "negative_feature",
            lambda d: d.__setitem__("feature_ids", t.tensor([-1, 2, 3], dtype=t.int64)),
            "strict non-negative integers",
        ),
        (
            "negative_bucket",
            lambda d: d.__setitem__("buckets", t.tensor([-1, 1, 2], dtype=t.int64)),
            "strict non-negative integers",
        ),
        (
            "negative_char_start",
            lambda d: d.__setitem__(
                "segment_char_starts", t.tensor([-1, 2], dtype=t.int32)
            ),
            "strict non-negative integers",
        ),
    ]

    for name, mutate, match in corrupt_cases:
        case_dir = tmp_path / name
        save_context_cache(
            case_dir, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
        )
        case_shard = next(case_dir.glob("contexts_*.safetensors.zst"))
        case_tensors = load_overlay_shard(case_shard)
        mutate(case_tensors)
        save_overlay_shard(case_tensors, case_shard)
        manifest = json.loads((case_dir / "manifest.json").read_text())
        manifest["shards"][0]["sha256"] = _sha256_file(case_shard)
        (case_dir / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ContextCacheValidationError, match=match):
            SafetensorsContextFeatureProvider(case_dir, identity=TEST_CACHE_IDENTITY)

    tensors["context_hash"][0, 0] ^= 1
    save_overlay_shard(tensors, shard_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"][0]["sha256"] = _sha256_file(shard_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContextCacheValidationError, match="hash does not match"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_detects_shard_count_and_duplicate_mismatch(tmp_path):
    entries = [_entry((0, 1)), _entry((0, 2))]
    save_context_cache(
        tmp_path,
        entries,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        shard_size=_SHARD_SIZE_ONE,
        identity=TEST_CACHE_IDENTITY,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    bad_count = json.loads(json.dumps(manifest))
    bad_count["shards"][0]["n_contexts"] = 2
    manifest_path.write_text(json.dumps(bad_count))
    with pytest.raises(ContextCacheValidationError, match="manifest says"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)

    manifest_path.write_text(json.dumps(manifest))
    second_shard = tmp_path / manifest["shards"][1]["file"]
    first_shard = tmp_path / manifest["shards"][0]["file"]
    second_shard.write_bytes(first_shard.read_bytes())
    manifest["shards"][1]["sha256"] = _sha256_file(second_shard)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContextCacheValidationError, match="Duplicate"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_detects_manifest_total_mismatch(tmp_path):
    save_context_cache(
        tmp_path, [_entry()], n_sae_features=TEST_CACHE_N_SAE_FEATURES, num_buckets=TEST_CACHE_NUM_BUCKETS, identity=TEST_CACHE_IDENTITY
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_contexts"] = 2
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError, match="Manifest says"):
        SafetensorsContextFeatureProvider(tmp_path, identity=TEST_CACHE_IDENTITY)


def test_context_cache_entry_type_and_token_range_validation(tmp_path):
    bad_entry = object.__new__(ContextCacheEntry)
    object.__setattr__(bad_entry, "token_ids", [1])
    object.__setattr__(bad_entry, "segments", ())

    with pytest.raises(TypeError, match="token_ids"):
        save_context_cache(
            tmp_path / "bad_type",
            [bad_entry],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )

    with pytest.raises(OverflowError):
        save_context_cache(
            tmp_path / "bad_token",
            [ContextCacheEntry.from_sequences([2**80], [])],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            identity=TEST_CACHE_IDENTITY,
        )


def test_context_cache_private_validators_cover_empty_and_offset_edges():
    empty_tensors = cache_io._entries_to_tensors([], identity=TEST_CACHE_IDENTITY)
    assert empty_tensors["context_hash"].shape == (0, 32)

    with pytest.raises(ContextCacheValidationError, match="expected length"):
        cache_io._validate_offsets(
            t.tensor([[0]], dtype=t.int64),
            expected_len=1,
            terminal=0,
            source="unit",
        )

    cache_io._validate_offsets(
        t.tensor([], dtype=t.int64),
        expected_len=0,
        terminal=0,
        source="unit",
    )

    with pytest.raises(ContextCacheValidationError, match="monotonic"):
        cache_io._validate_offsets(
            t.tensor([0, 2, 1, 3], dtype=t.int64),
            expected_len=4,
            terminal=3,
            source="unit",
        )

    with pytest.raises(ContextCacheValidationError, match="lengths differ"):
        cache_io._validate_segment(
            ContextFeatureSegment((1, 2), (0,), 0, 1),
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            source="unit",
        )


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ============================================================================
# D-contract (PR #739 Track D, Option B) Tier 4: clustered_context_cache_
# metadata's additive, optional checkpoint_seed_provenance kwarg (a
# CheckpointSeedProvenance, not Track B's ClusterFeatureContract -- see
# steering/eval/clustered_cache_seed_contract.py). Two properties: omitting
# it (or passing None explicitly) is byte-identical to this function's
# pre-Track-D behavior (U-invariant); passing a resolved provenance adds
# exactly the 5 documented checkpoint_-prefixed keys, colliding with none of
# the existing ones.
# ============================================================================


def _tiny_cluster_consolidation(tmp_path: Path) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((1, 3),))
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(json.dumps(definition.canonical_dict()), encoding="utf-8")
    resolved = ResolvedFeatureClusters(
        source_path=cluster_path,
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=TEST_CACHE_N_SAE_FEATURES),
    )
    return FeatureClusterConsolidation.build(resolved, None)


def test_clustered_context_cache_metadata_omitting_checkpoint_contract_matches_pinned_pre_track_d_shape(
    tmp_path,
):
    """U-invariant: omitting checkpoint_seed_provenance (its default) and
    passing checkpoint_seed_provenance=None explicitly must both reproduce
    this function's exact pre-Track-D metadata shape, byte for byte --
    proving the new kwarg is purely additive and inert when unused."""

    clusters = _tiny_cluster_consolidation(tmp_path)
    vocab = effective_feature_vocab(clusters, clusters.n_sae_features)
    kwargs = {
        "feature_mask": None,
        "feature_clusters": clusters,
        "feature_vocab": vocab,
        "stats_source_sha256": "a" * 64,
    }

    omitted = clustered_context_cache_metadata(**kwargs)  # ty: ignore[invalid-argument-type]
    explicit_none = clustered_context_cache_metadata(**kwargs, checkpoint_seed_provenance=None)

    # Pinned literal: this function's own documented pre-Track-D shape
    # (steering/eval/context_cache_io.py, built from the same lower-level
    # helpers the function itself delegates to -- feature_mask_canonical,
    # feature_clusters_checkpoint_metadata's two keys, the feature-vocab
    # metadata key, the cluster-selection-semantics tag, feature_delta, and
    # the caller-supplied stats_source_sha256; nothing checkpoint_-prefixed).
    pinned_pre_track_d_literal = {
        "feature_mask_canonical": None,
        **feature_clusters_checkpoint_metadata(clusters.resolved),
        FEATURE_VOCAB_META_KEY: vocab.metadata_dict() if vocab is not None else None,
        "cluster_selection_semantics": CLUSTER_SELECTION_SEMANTICS,
        "feature_delta": None,
        "stats_source_sha256": "a" * 64,
    }

    assert omitted == pinned_pre_track_d_literal
    assert explicit_none == pinned_pre_track_d_literal
    assert json.dumps(omitted, sort_keys=True) == json.dumps(
        pinned_pre_track_d_literal, sort_keys=True
    )
    assert not any(key.startswith("checkpoint_") for key in omitted)


def test_clustered_context_cache_metadata_with_contract_adds_exactly_five_keys_no_collision(
    tmp_path,
):
    clusters = _tiny_cluster_consolidation(tmp_path)
    vocab = effective_feature_vocab(clusters, clusters.n_sae_features)
    kwargs = {
        "feature_mask": None,
        "feature_clusters": clusters,
        "feature_vocab": vocab,
        "stats_source_sha256": "a" * 64,
    }
    baseline = clustered_context_cache_metadata(**kwargs)  # ty: ignore[invalid-argument-type]

    provenance = CheckpointSeedProvenance(
        resolved_tag="faketag",
        step=7,
        model_sha256="b" * 64,
        metadata_sha256="c" * 64,
        preselection_seed=99,
    )
    with_contract = clustered_context_cache_metadata(
        **kwargs, checkpoint_seed_provenance=provenance
    )

    new_keys = set(with_contract) - set(baseline)
    assert new_keys == {
        "checkpoint_resolved_tag",
        "checkpoint_model_sha256",
        "checkpoint_metadata_sha256",
        "checkpoint_step",
        "checkpoint_preselection_seed",
    }
    assert with_contract["checkpoint_resolved_tag"] == "faketag"
    assert with_contract["checkpoint_model_sha256"] == "b" * 64
    assert with_contract["checkpoint_metadata_sha256"] == "c" * 64
    assert with_contract["checkpoint_step"] == 7
    assert with_contract["checkpoint_preselection_seed"] == 99
    # No collision: every pre-existing key keeps its pre-Track-D value.
    for key, value in baseline.items():
        assert with_contract[key] == value


def test_fixed_cache_writer_cannot_discard_turn_entry_provenance(tmp_path: Path) -> None:
    entry = replace(
        _entry(), segment_mode=cache_io.SegmentMode.turn_delimited_fixed_tokens,
        token_assignments=(0, 0, 1),
    )
    with pytest.raises(ValueError,
                       match="persistent fixed_tokens context caches support only fixed_tokens"):
        save_context_cache(
            tmp_path, [entry], identity=TEST_CACHE_IDENTITY,
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
        )
    assert not list(tmp_path.glob("contexts_*"))
