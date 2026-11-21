"""Regression tests: context cache key MUST include full SAE identity.

The full identity is model_id, sae_id, layer, pooling, and
feature_selection_hash.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from steering.eval.context_cache_io import (
    SCHEMA_VERSION,
    SOURCE_SEGMENT_MODE_KEY,
    CacheIdentity,
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    load_context_cache_manifest,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import TEST_CACHE_N_SAE_FEATURES, TEST_CACHE_NUM_BUCKETS

BASE_IDENTITY = CacheIdentity(
    model_id="gemma-2-9b",
    sae_id="16k-layer12",
    layer=12,
    pooling="mean",
    feature_selection_hash="d" * 64,
)


def _entry(tokens=(0, 2, 3)) -> ContextCacheEntry:
    return ContextCacheEntry.from_sequences(
        tokens,
        [ContextFeatureSegment((1, 2), (0, 1), 0, 2)],
    )


def _mutated_identity(field: str):
    if field == "model_id":
        return replace(BASE_IDENTITY, model_id="gemma-2-2b")
    if field == "sae_id":
        return replace(BASE_IDENTITY, sae_id="65k-layer12")
    if field == "layer":
        return replace(BASE_IDENTITY, layer=13)
    if field == "pooling":
        return replace(BASE_IDENTITY, pooling="max")
    if field == "feature_selection_hash":
        return replace(BASE_IDENTITY, feature_selection_hash="c" * 64)
    raise AssertionError(field)


@pytest.mark.parametrize(
    "field",
    ["model_id", "sae_id", "layer", "pooling", "feature_selection_hash"],
)
def test_provider_miss_when_identity_field_differs(tmp_path, field):
    """Build cache with identity A, then try to load with mutated identity B;
    every field change must produce an identity-mismatch error (no silent
    stale-cache read)."""

    save_context_cache(
        tmp_path,
        [_entry()],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )

    # Same identity: round-trip succeeds.
    provider = SafetensorsContextFeatureProvider(tmp_path, identity=BASE_IDENTITY)
    assert provider.get((0, 2, 3)) is not None

    # Mutated identity: manifest identity check fires.
    with pytest.raises(ContextCacheValidationError, match="identity mismatch"):
        SafetensorsContextFeatureProvider(tmp_path, identity=_mutated_identity(field))


def test_provider_hit_and_content_equality_for_same_identity(tmp_path):
    entry = _entry((5, 6, 7))
    save_context_cache(
        tmp_path,
        [entry],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    provider = SafetensorsContextFeatureProvider(tmp_path, identity=BASE_IDENTITY)
    segments = provider.get((5, 6, 7))
    assert segments == entry.segments


@pytest.mark.parametrize("old_schema", [1, 2, 3])
def test_old_schema_cache_rejected_with_migration_message(tmp_path, old_schema):
    """Old caches are rejected because pre-removal rows are not trustworthy."""

    save_context_cache(
        tmp_path,
        [_entry()],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = old_schema
    if old_schema == 1:
        manifest.pop("identity", None)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError) as exc_info:
        SafetensorsContextFeatureProvider(tmp_path, identity=BASE_IDENTITY)
    msg = str(exc_info.value)
    assert "cache schema" in msg.lower()
    assert "re-run build" in msg.lower()
    if old_schema == 2:
        assert "required identity and source segmentation schema" in msg.lower()
    if old_schema == 3:
        assert "segmentation" in msg.lower()


def test_manifest_missing_identity_block_rejected(tmp_path):
    save_context_cache(
        tmp_path,
        [_entry()],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("identity", None)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError, match="identity"):
        SafetensorsContextFeatureProvider(tmp_path, identity=BASE_IDENTITY)


def test_save_context_cache_requires_identity_kwarg():
    """Compile-time safety: omitting identity must raise TypeError."""

    with pytest.raises(TypeError, match="identity"):
        save_context_cache(  # type: ignore[call-arg]
            "/tmp/irrelevant",
            [],
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
        )  # ty: ignore[missing-argument]


def test_provider_requires_identity_kwarg(tmp_path):
    save_context_cache(
        tmp_path,
        [_entry()],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    with pytest.raises(TypeError, match="identity"):
        SafetensorsContextFeatureProvider(tmp_path)  # type: ignore[call-arg]  # ty: ignore[missing-argument]


def test_callers_cannot_forget_sae_identity():
    """AST check: production callers of ``save_context_cache`` and
    ``SafetensorsContextFeatureProvider`` must always pass ``identity=`` as
    a keyword argument. A caller who omits it or passes only positional args
    is a bug."""

    targets = [
        Path(__file__).resolve().parent.parent.parent / "steering" / "eval" / "context_cache_io.py",
        Path(__file__).resolve().parent.parent.parent
        / "steering"
        / "eval"
        / "context_cache_build.py",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "build_eval_context_cache.py",
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "validate_eval_context_cache.py",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "training_progression.py",
    ]
    offenders: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name not in {"save_context_cache", "SafetensorsContextFeatureProvider"}:
                continue
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            # Allow calls that are clearly re-exports (no args) — real calls
            # must pass identity by keyword.
            if "identity" not in kwarg_names:
                offenders.append(f"{path.name}:{node.lineno}:{name}")
    assert not offenders, (
        "Every call to save_context_cache / SafetensorsContextFeatureProvider "
        "in production code MUST pass identity= explicitly. Offenders: " + ", ".join(offenders)
    )


def test_schema_v4_is_pinned_to_the_source_segment_mode_witness(tmp_path):
    """v4 exists to make the source-geometry witness mandatory."""

    # kills: reverting SCHEMA_VERSION, and accepting a pre-witness (v3)
    # manifest whose source_segment_mode key is absent.
    assert SCHEMA_VERSION == 4

    save_context_cache(
        tmp_path,
        [_entry()],
        identity=BASE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 3
    del manifest[SOURCE_SEGMENT_MODE_KEY]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        ContextCacheValidationError,
        match=(
            r"manifest schema_version=3, expected 4\..*"
            r"required identity and source segmentation schema"
        ),
    ):
        load_context_cache_manifest(tmp_path, identity=BASE_IDENTITY)
