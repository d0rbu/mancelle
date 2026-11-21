"""Strict parent-scan catalog tests for spawn-worker FeatureIndex hydration."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
import torch as t

import steering.conditional.feature_index as feature_index_module
from steering.conditional.feature_index import (
    ArtifactSignature,
    FeatureIndex,
    FeatureIndexCatalog,
    FeatureIndexCatalogError,
    SegmentFeatures,
)
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import TensorKey, load_features, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

PERCENTILE_CHECKSUM = "a" * 64
TOKENIZER_CHECKSUM = "b" * 64
CONTENT_FINGERPRINT = "c" * 64
TEXTS_BY_SHARD = (
    ("alpha beta gamma", "delta epsilon zeta"),
    ("eta theta iota", "kappa lambda mu", "nu xi omicron"),
    ("pi rho sigma", "tau upsilon phi"),
)


def _build_feature_tree(
    root: Path,
    *,
    overlays: bool = True,
) -> tuple[Path, dict[str, int]]:
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True)
    tokenizer = CharMergeTokenizer(bos_id=500)
    docs_per_stem: dict[str, int] = {}
    for shard_idx, texts in enumerate(TEXTS_BY_SHARD):
        stem = f"shard_{shard_idx:05d}"
        data = make_base_shard(list(texts), tokenizer, 2, n_features=65_536)
        save_features(data, feature_dir / f"{stem}_features")
        docs_per_stem[stem] = len(texts)

    if not overlays:
        return feature_dir, docs_per_stem

    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, PERCENTILE_CHECKSUM)
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, TOKENIZER_CHECKSUM)
    write_manifest(pct_dir, PERCENTILE_CHECKSUM, OverlayKind.PERCENTILE)
    write_manifest(tok_dir, TOKENIZER_CHECKSUM, OverlayKind.SEG_BOUNDARY)
    for stem in docs_per_stem:
        raw = load_features(feature_dir / f"{stem}_features.safetensors.zst")
        feature_ids = raw["feature_ids"]
        n_segments = int(raw["n_segments"])
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: (feature_ids.to(t.int64).remainder(256).to(t.uint8))},
            overlay_shard_path(pct_dir, stem),
        )
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.arange(n_segments, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.arange(1, n_segments + 1, dtype=t.int32),
            },
            overlay_shard_path(tok_dir, stem),
        )
    return feature_dir, docs_per_stem


def _new_parent(feature_dir: Path, *, overlays: bool = True) -> FeatureIndex:
    return FeatureIndex(
        str(feature_dir),
        percentile_checksum=PERCENTILE_CHECKSUM if overlays else None,
        tokenizer_checksum=TOKENIZER_CHECKSUM if overlays else None,
        max_resident_shards=2,
    )


def _segment_snapshot(segment: SegmentFeatures) -> tuple[Any, ...]:
    return (
        segment.segment_idx,
        segment.position,
        segment.feature_ids.dtype,
        segment.feature_ids.tolist(),
        segment.feature_mags.dtype,
        segment.feature_mags.tolist(),
        segment.char_start,
        segment.char_end,
        None
        if segment.feature_percentiles is None
        else (
            segment.feature_percentiles.dtype,
            segment.feature_percentiles.tolist(),
        ),
        segment.token_start,
        segment.token_end,
    )


def _index_snapshot(
    index: FeatureIndex,
    docs_per_stem: dict[str, int],
) -> dict[tuple[str, int], list[tuple[Any, ...]]]:
    return {
        (stem, doc_idx): [_segment_snapshot(segment) for segment in index.lookup(stem, doc_idx)]
        for stem, n_docs in docs_per_stem.items()
        for doc_idx in range(n_docs)
    }


def _spawn_hydrate_catalog(
    catalog: FeatureIndexCatalog,
    result_queue: multiprocessing.Queue,
) -> None:
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("FeatureIndex.from_catalog called load_features")

    feature_index_module.load_features = (  # ty: ignore[invalid-assignment] -- child-process sentinel
        forbidden_load
    )
    try:
        index = FeatureIndex.from_catalog(
            catalog,
            prefetch=True,
            skip_feature_mags=True,
            compact_feature_ids=True,
            max_resident_shards=3,
        )
        result_queue.put(
            (
                "ok",
                index.shard_stems,
                index.segment_size,
                index.n_shards,
                index.has_precomputed_percentiles,
                index.has_precomputed_token_boundaries,
                index.max_resident_shards,
                index.cache_stats.snapshot(),
            )
        )
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


def _signature_for(catalog: FeatureIndexCatalog, path: Path) -> ArtifactSignature:
    absolute = str(path.absolute())
    return next(signature for signature in catalog.signatures if signature.path == absolute)


def test_catalog_round_trip_preserves_all_index_metadata(tmp_path: Path) -> None:
    feature_dir, docs_per_stem = _build_feature_tree(tmp_path)
    parent = _new_parent(feature_dir)
    catalog = parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    hydrated = FeatureIndex.from_catalog(catalog, max_resident_shards=2)

    assert hydrated.feature_dir == feature_dir
    assert hydrated.shard_stems == sorted(docs_per_stem)
    assert hydrated.segment_size == parent.segment_size == 2
    assert hydrated.n_shards == parent.n_shards == len(docs_per_stem)
    assert hydrated.has_precomputed_percentiles is True
    assert hydrated.has_precomputed_token_boundaries is True
    assert hydrated.max_resident_shards == parent.max_resident_shards == 2
    assert hydrated.cache_stats.snapshot() == parent.cache_stats.snapshot()
    assert hydrated.export_catalog() == replace(catalog, validated_content_fingerprint=None)


@pytest.mark.parametrize("skip_mags", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_catalog_lookup_is_value_and_order_identical_to_fresh_scan(
    tmp_path: Path,
    skip_mags: bool,
    compact: bool,
    prefetch: bool,
) -> None:
    feature_dir, docs_per_stem = _build_feature_tree(tmp_path)
    scanned = FeatureIndex(
        str(feature_dir),
        percentile_checksum=PERCENTILE_CHECKSUM,
        tokenizer_checksum=TOKENIZER_CHECKSUM,
        skip_feature_mags=skip_mags,
        compact_feature_ids=compact,
        prefetch=prefetch,
        max_resident_shards=2,
    )
    catalog = scanned.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    hydrated = FeatureIndex.from_catalog(
        catalog,
        skip_feature_mags=skip_mags,
        compact_feature_ids=compact,
        prefetch=prefetch,
        max_resident_shards=2,
    )

    assert _index_snapshot(hydrated, docs_per_stem) == _index_snapshot(scanned, docs_per_stem)
    hydrated_stats = hydrated.cache_stats.snapshot()
    scanned_stats = scanned.cache_stats.snapshot()
    hydrated_load_seconds = hydrated_stats.pop("load_seconds")
    scanned_load_seconds = scanned_stats.pop("load_seconds")
    assert hydrated_stats == scanned_stats
    assert hydrated_load_seconds > 0
    assert scanned_load_seconds > 0


@pytest.mark.parametrize("skip_mags", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_catalog_shared_store_lookup_matches_private_scan(
    tmp_path: Path,
    skip_mags: bool,
    compact: bool,
) -> None:
    feature_dir, docs_per_stem = _build_feature_tree(tmp_path)
    private = _new_parent(feature_dir)
    catalog = private.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    shared = FeatureIndex.from_catalog(
        catalog,
        skip_feature_mags=skip_mags,
        compact_feature_ids=compact,
        max_resident_shards=1,
        shared_store=SharedStemStore(tmp_path / "store", capacity=1),
    )
    reference = FeatureIndex.from_catalog(
        catalog,
        skip_feature_mags=skip_mags,
        compact_feature_ids=compact,
        max_resident_shards=1,
    )

    assert _index_snapshot(shared, docs_per_stem) == _index_snapshot(reference, docs_per_stem)
    assert shared.cache_stats.shared_misses >= 1
    assert shared.cache_stats.materializations >= 1


def test_parent_loads_first_body_once_and_catalog_hydration_loads_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    real_load = feature_index_module.load_features
    loaded_paths: list[Path] = []

    def recording_load(path, *, keys=None):
        loaded_paths.append(Path(path))
        return real_load(path, keys=keys)

    monkeypatch.setattr(feature_index_module, "load_features", recording_load)
    parent = _new_parent(feature_dir)
    catalog = parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    for _ in range(16):
        hydrated = FeatureIndex.from_catalog(catalog, max_resident_shards=1)
        assert hydrated.n_shards == 3

    assert loaded_paths == [feature_dir / "shard_00000_features.safetensors.zst"]


def test_spawn_hydration_never_calls_full_shard_loader(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_hydrate_catalog,
        args=(catalog, result_queue),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        pytest.fail("spawn catalog hydration did not terminate within 30 seconds")

    assert process.exitcode == 0
    result = result_queue.get(timeout=5)
    assert result == (
        "ok",
        ["shard_00000", "shard_00001", "shard_00002"],
        2,
        3,
        True,
        True,
        3,
        {
            "loads": 0,
            "hits": 0,
            "evictions": 0,
            "retain_calls": 0,
            "load_seconds": 0.0,
            "bytes_read": 0,
            "shared_hits": 0,
            "shared_misses": 0,
            "shared_get_races": 0,
            "materializations": 0,
            "materialize_seconds": 0.0,
            "materialize_lock_wait_seconds": 0.0,
            "publications": 0,
            "publish_declines": 0,
            "private_fallbacks": 0,
            "shared_mapped_bytes": 0,
        },
    )


def test_catalog_is_pickle_stable_and_frozen(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    restored = pickle.loads(pickle.dumps(catalog, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored == catalog
    restored.validate()
    with pytest.raises(FrozenInstanceError):
        restored.segment_size = 99
    with pytest.raises(FrozenInstanceError):
        restored.signatures[0].path = "changed"


def test_catalog_without_overlays_preserves_absence(tmp_path: Path) -> None:
    feature_dir, docs_per_stem = _build_feature_tree(tmp_path, overlays=False)
    parent = _new_parent(feature_dir, overlays=False)
    catalog = parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    hydrated = FeatureIndex.from_catalog(catalog)

    assert catalog.percentile_overlay_dir is None
    assert catalog.percentile_overlay_entries == ()
    assert catalog.token_overlay_dir is None
    assert catalog.token_overlay_entries == ()
    assert hydrated.has_precomputed_percentiles is False
    assert hydrated.has_precomputed_token_boundaries is False
    assert _index_snapshot(hydrated, docs_per_stem) == _index_snapshot(parent, docs_per_stem)


def test_empty_directory_catalog_round_trip(tmp_path: Path) -> None:
    feature_dir = tmp_path / "empty"
    feature_dir.mkdir()
    parent = FeatureIndex(str(feature_dir))
    catalog = parent.export_catalog()
    hydrated = FeatureIndex.from_catalog(catalog)

    assert catalog.shard_entries == ()
    assert catalog.segment_size is None
    assert [signature.path for signature in catalog.signatures] == [str(feature_dir)]
    assert hydrated.n_shards == 0
    assert hydrated.shard_stems == []
    with pytest.raises(KeyError, match="Unknown shard stem"):
        hydrated.lookup("missing", 0)


def test_catalog_inventory_covers_every_consumed_artifact_exactly(
    tmp_path: Path,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    expected_paths = {str(feature_dir)}
    expected_paths.update(path for _stem, path in catalog.shard_entries)
    for directory, entries in (
        (catalog.percentile_overlay_dir, catalog.percentile_overlay_entries),
        (catalog.token_overlay_dir, catalog.token_overlay_entries),
    ):
        assert directory is not None
        expected_paths.add(directory)
        expected_paths.add(str(Path(directory) / "manifest.json"))
        expected_paths.update(path for _stem, path in entries)

    assert {signature.path for signature in catalog.signatures} == expected_paths
    assert len(catalog.percentile_overlay_entries) == 3
    assert len(catalog.token_overlay_entries) == 3
    assert len(catalog.signatures) == 14
    assert all(Path(path).is_absolute() for path in expected_paths)


def test_scan_time_signature_is_not_rebased_by_export(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    parent = _new_parent(feature_dir)
    first_path = feature_dir / "shard_00000_features.safetensors.zst"
    original = first_path.read_bytes()
    replacement = tmp_path / "replacement.zst"
    replacement.write_bytes(original)
    replacement.replace(first_path)

    with pytest.raises(
        FeatureIndexCatalogError,
        match="changed after parent validation",
    ):
        parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)


def test_parent_first_body_validation_error_is_not_caught_or_reordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path, overlays=False)
    sentinel = RuntimeError("sentinel full-body validation failure")

    def fail_load(_path, *, keys=None):
        del keys
        raise sentinel

    monkeypatch.setattr(feature_index_module, "load_features", fail_load)
    with pytest.raises(RuntimeError, match="sentinel full-body") as raised:
        FeatureIndex(str(feature_dir))
    assert raised.value is sentinel


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "schema_version"),
        ("feature_dir", "relative/features", "feature_dir must be absolute"),
        ("segment_size", 0, "segment_size int >= 1"),
        ("segment_size", True, "segment_size int >= 1"),
        (
            "validated_content_fingerprint",
            "ABC",
            "lowercase SHA-256 hex digest",
        ),
        (
            "validated_content_fingerprint",
            "g" * 64,
            "lowercase SHA-256 hex digest",
        ),
    ],
)
def test_catalog_rejects_invalid_scalar_structure(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )

    with pytest.raises(FeatureIndexCatalogError, match=message):
        replace(catalog, **{field: value}).validate()


def test_empty_catalog_rejects_non_null_segment_size(tmp_path: Path) -> None:
    feature_dir = tmp_path / "empty"
    feature_dir.mkdir()
    catalog = FeatureIndex(str(feature_dir)).export_catalog()

    with pytest.raises(FeatureIndexCatalogError, match="requires segment_size=None"):
        replace(catalog, segment_size=1).validate()


def test_catalog_rejects_unsorted_and_duplicate_base_stems(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    reversed_entries = tuple(reversed(catalog.shard_entries))
    duplicate_entries = catalog.shard_entries + (catalog.shard_entries[-1],)

    with pytest.raises(FeatureIndexCatalogError, match="unique sorted stems"):
        replace(catalog, shard_entries=reversed_entries).validate()
    with pytest.raises(FeatureIndexCatalogError, match="unique sorted stems"):
        replace(catalog, shard_entries=duplicate_entries).validate()


def test_catalog_rejects_noncanonical_base_entry(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    stem, path = catalog.shard_entries[0]

    with pytest.raises(FeatureIndexCatalogError, match="direct-child path"):
        replace(
            catalog,
            shard_entries=(("shard_00000_wrong", path), *catalog.shard_entries[1:]),
        ).validate()
    with pytest.raises(FeatureIndexCatalogError, match="must be absolute"):
        replace(
            catalog,
            shard_entries=((stem, Path(path).name), *catalog.shard_entries[1:]),
        ).validate()


@pytest.mark.parametrize(
    ("checksum_field", "directory_field", "message"),
    [
        (
            "percentile_checksum",
            "percentile_overlay_dir",
            "percentile checksum/overlay presence mismatch",
        ),
        (
            "tokenizer_checksum",
            "token_overlay_dir",
            "tokenizer checksum/overlay presence mismatch",
        ),
    ],
)
def test_catalog_rejects_checksum_directory_mismatch(
    tmp_path: Path,
    checksum_field: str,
    directory_field: str,
    message: str,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )

    with pytest.raises(FeatureIndexCatalogError, match=message):
        replace(catalog, **{checksum_field: None}).validate()
    with pytest.raises(FeatureIndexCatalogError, match=message):
        replace(catalog, **{directory_field: None}).validate()


def test_catalog_rejects_malformed_percentile_coverage(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )

    with pytest.raises(FeatureIndexCatalogError, match="cover every base shard"):
        replace(
            catalog,
            percentile_overlay_entries=catalog.percentile_overlay_entries[:-1],
        ).validate()
    with pytest.raises(FeatureIndexCatalogError, match="unique sorted stems"):
        replace(
            catalog,
            percentile_overlay_entries=tuple(reversed(catalog.percentile_overlay_entries)),
        ).validate()


def test_catalog_rejects_malformed_token_coverage(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    token_dir = Path(catalog.token_overlay_dir or "")
    foreign = ("shard_99999", str(overlay_shard_path(token_dir, "shard_99999")))

    with pytest.raises(FeatureIndexCatalogError, match="non-base shard stem"):
        replace(
            catalog,
            token_overlay_entries=(*catalog.token_overlay_entries, foreign),
        ).validate()


def test_catalog_rejects_noncanonical_overlay_payload_path(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    stem, _path = catalog.token_overlay_entries[0]
    wrong = str(feature_dir / "wrong.safetensors.zst")

    with pytest.raises(FeatureIndexCatalogError, match="noncanonical path"):
        replace(
            catalog,
            token_overlay_entries=((stem, wrong), *catalog.token_overlay_entries[1:]),
        ).validate()


def test_catalog_rejects_unsorted_duplicate_missing_and_extra_signatures(
    tmp_path: Path,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    reversed_signatures = tuple(reversed(catalog.signatures))
    duplicate_signatures = tuple(
        sorted((*catalog.signatures, catalog.signatures[0]), key=lambda item: item.path)
    )
    extra_path = tmp_path / "extra"
    extra_path.write_text("not consumed")
    extra_signatures = tuple(
        sorted(
            (*catalog.signatures, ArtifactSignature.capture(extra_path)),
            key=lambda item: item.path,
        )
    )

    with pytest.raises(FeatureIndexCatalogError, match="path-sorted"):
        replace(catalog, signatures=reversed_signatures).validate()
    with pytest.raises(FeatureIndexCatalogError, match="duplicate"):
        replace(catalog, signatures=duplicate_signatures).validate()
    with pytest.raises(FeatureIndexCatalogError, match="missing required signatures"):
        replace(catalog, signatures=catalog.signatures[1:]).validate()
    with pytest.raises(FeatureIndexCatalogError, match="unexpected signatures"):
        replace(catalog, signatures=extra_signatures).validate()


@pytest.mark.parametrize("target_kind", ["base", "manifest", "percentile", "token"])
def test_catalog_rejects_consumed_file_content_mutation(
    tmp_path: Path,
    target_kind: str,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    targets = {
        "base": Path(catalog.shard_entries[1][1]),
        "manifest": Path(catalog.percentile_overlay_dir or "") / "manifest.json",
        "percentile": Path(catalog.percentile_overlay_entries[1][1]),
        "token": Path(catalog.token_overlay_entries[1][1]),
    }
    target = targets[target_kind]
    original_stat = target.stat()
    original = target.read_bytes()
    replacement_byte = b"x" if original[:1] != b"x" else b"y"
    target.write_bytes(replacement_byte + original[1:])
    os.utime(
        target,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(
        FeatureIndexCatalogError,
        match="changed after parent validation",
    ):
        FeatureIndex.from_catalog(catalog)


@pytest.mark.parametrize("target_kind", ["base", "manifest", "percentile", "token"])
def test_catalog_rejects_consumed_file_deletion(
    tmp_path: Path,
    target_kind: str,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    targets = {
        "base": Path(catalog.shard_entries[0][1]),
        "manifest": Path(catalog.token_overlay_dir or "") / "manifest.json",
        "percentile": Path(catalog.percentile_overlay_entries[0][1]),
        "token": Path(catalog.token_overlay_entries[0][1]),
    }
    targets[target_kind].unlink()

    with pytest.raises(
        FeatureIndexCatalogError,
        match="disappeared or became unreadable",
    ):
        catalog.validate()


def test_catalog_rejects_same_byte_atomic_replacement(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    target = Path(catalog.shard_entries[2][1])
    before = _signature_for(catalog, target)
    replacement_path = tmp_path / target.name
    shutil.copyfile(target, replacement_path)
    replacement_path.replace(target)

    assert target.read_bytes()
    assert ArtifactSignature.capture(target).stat_fields != before.stat_fields
    with pytest.raises(FeatureIndexCatalogError, match="changed after parent validation"):
        catalog.validate()


def test_catalog_rejects_mode_change_even_when_bytes_are_unchanged(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    target = Path(catalog.shard_entries[0][1])
    target.chmod(0o600 if target.stat().st_mode & 0o077 else 0o644)

    with pytest.raises(FeatureIndexCatalogError, match="changed after parent validation"):
        catalog.validate()


def test_catalog_rejects_new_direct_child_after_scan(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    (feature_dir / "unexpected-direct-child").write_text("drift")

    with pytest.raises(FeatureIndexCatalogError, match="changed after parent validation"):
        catalog.validate()


def test_catalog_allows_output_write_below_unsigned_existing_subtree(
    tmp_path: Path,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, PERCENTILE_CHECKSUM)
    output_dir = pct_dir / "packed" / "sel" / "selection-hash"
    output_dir.mkdir(parents=True)
    parent = _new_parent(feature_dir)
    catalog = parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    (output_dir / "shard_00000.safetensors.zst").write_bytes(b"selection")

    catalog.validate()
    assert FeatureIndex.from_catalog(catalog).n_shards == 3


def test_catalog_rejects_overlay_payload_symlink_retarget(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    payload = overlay_shard_path(
        overlay_dir(feature_dir, OverlayKind.PERCENTILE, PERCENTILE_CHECKSUM),
        "shard_00000",
    )
    target_a = tmp_path / "target-a.zst"
    target_b = tmp_path / "target-b.zst"
    shutil.copyfile(payload, target_a)
    shutil.copyfile(payload, target_b)
    payload.unlink()
    payload.symlink_to(target_a)
    parent = _new_parent(feature_dir)
    catalog = parent.export_catalog(validated_content_fingerprint=CONTENT_FINGERPRINT)
    signed = _signature_for(catalog, payload)
    assert signed.lstat_fields != signed.stat_fields
    payload.unlink()
    payload.symlink_to(target_b)

    with pytest.raises(FeatureIndexCatalogError, match="changed after parent validation"):
        FeatureIndex.from_catalog(catalog)


def test_from_catalog_preserves_max_resident_shards_validation(tmp_path: Path) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )

    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="int >= 1"):
            FeatureIndex.from_catalog(
                catalog,
                max_resident_shards=invalid,  # ty: ignore[invalid-argument-type] -- runtime boundary
            )


def test_catalog_drift_failure_occurs_before_cache_state_is_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _docs_per_stem = _build_feature_tree(tmp_path)
    catalog = _new_parent(feature_dir).export_catalog(
        validated_content_fingerprint=CONTENT_FINGERPRINT
    )
    Path(catalog.shard_entries[0][1]).unlink()
    initialized = False
    real_initialize = FeatureIndex._initialize_empty

    def recording_initialize(self, **kwargs):
        nonlocal initialized
        initialized = True
        return real_initialize(self, **kwargs)

    monkeypatch.setattr(FeatureIndex, "_initialize_empty", recording_initialize)
    with pytest.raises(FeatureIndexCatalogError, match="disappeared"):
        FeatureIndex.from_catalog(catalog)
    assert initialized is False
