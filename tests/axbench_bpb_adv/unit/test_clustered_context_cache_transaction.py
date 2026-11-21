"""DDP transaction and committed-manifest contracts for clustered BPB-Adv."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.eval.axbench_bpb_adv.clustered_context_cache as cache_module
import tests.axbench_base._multiprocess_helpers as multiprocess_helpers
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.sample_loader import (
    collect_unique_prefixes,
    make_prefix_hash,
)
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    CanonicalPrefixPopulation,
    ClusteredReplayState,
    RawClusteredFeatureSegment,
    RawClusteredPrefix,
    build_clustered_context_cache,
    canonicalize_prefix_population,
    load_clustered_context_segments_by_prefix_hash,
    validate_clustered_context_cache_manifest,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContract,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    build_cluster_feature_contract,
)
from steering.eval.target_conditioning import ContextFeatureSegment

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _contract(tmp_path: Path) -> ClusterFeatureContract:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    definition = FeatureClusterDefinition(clusters=((1, 2), (4, 5)))
    clusters = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=8,
        ),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    return build_cluster_feature_contract(
        feature_mask=mask,
        feature_clusters=clusters,
        statistics=FeatureStatisticsIdentity(
            file_sha256=_SHA_A,
            percentile_table_checksum=_SHA_B,
            n_sae_features=8,
            feature_id_domain=(1, 2, 3, 4, 5, 6),
        ),
        feature_directory=FeatureDirectoryIdentity(
            segment_size=4,
            layer=8,
            activation_type="residual",
            pooling="max_pool",
            width="16k",
            model="gemma-2-2b",
            dataset_phase="base",
            context_independent_features=False,
            ignore_bos_features=False,
        ),
        sae=SaeIdentity(
            release="release",
            sae_id="sae-id",
            hook_name="blocks.8.hook_resid_post",
            model="gemma-2-2b",
            layer=8,
            activation_type="residual",
            width="16k",
            actual_loaded_width=8,
        ),
        checkpoint=CheckpointPin(
            checkpoint_dir=checkpoint_dir,
            resolved_tag="clustered",
            step=7,
            model_file=StableFileDigest(
                path=checkpoint_dir / "model_000007.pt",
                sha256=_SHA_C,
                byte_length=1,
                stat_signature=(1, 2, 1, 3, 4, 5),
            ),
            metadata_file=StableFileDigest(
                path=checkpoint_dir / "meta_000007.json",
                sha256=_SHA_D,
                byte_length=1,
                stat_signature=(1, 3, 1, 3, 4, 5),
            ),
            n_sae_features=8,
            feature_vocab=FeatureVocab.from_mask(mask, 8),
            load_bearing_metadata_json="{}",
        ),
        selection=CheckpointSelectionContract(
            positive_selection_json=json.dumps(
                {
                    "threshold_type": "feature_percentile",
                    "threshold": 0.5,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            negative_selection_json=json.dumps(
                {
                    "distribution": "normal",
                    "mean": 1.0,
                    "std": 0.01,
                    "min_k": 1,
                    "max_k": 2,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            num_buckets=2,
            effective_positive_buckets=1,
            preselection_seed=19,
            selection_hash=_SHA_A,
            selection_manifest_sha256=_SHA_B,
            selection_shards_sha256=_SHA_C,
            packed_row_fingerprint=_SHA_D,
        ),
        concept_source_mode="statistics_domain",
        concept_source_domain=(1, 2, 3, 4, 5, 6),
    )


def _replay_state(contract: ClusterFeatureContract) -> ClusteredReplayState:
    rng = t.Generator().manual_seed(contract.selection.preselection_seed)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                ThresholdType.feature_percentile,
                0.5,
            ),
            precomputed_thresholds=t.zeros(8, dtype=t.float32),
        ),
        boundaries=t.zeros(8, 0, dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=1.0,
                std=0.01,
                min_k=1,
                max_k=2,
            ),
            n_sae_features=8,
            rng=rng,
            feature_mask=contract.feature_mask,
        ),
        effective_buckets=1,
        feature_mask=contract.feature_mask,
    )
    percentile_table = t.arange(1, 100, dtype=t.float64).unsqueeze(0).repeat(8, 1)
    return ClusteredReplayState(
        pipeline=pipeline,
        percentile_table=percentile_table,
    )


def _raw_extract(
    *,
    prefix_text: str,
    base_model: Any,
    base_sae: Any,
    contract: ClusterFeatureContract,
    segment_size: int,
    extraction_max_tokens: int,
) -> RawClusteredPrefix:
    del base_model, base_sae, contract, segment_size, extraction_max_tokens
    magnitude = float(60 + (sum(prefix_text.encode("utf-8")) % 30))
    return RawClusteredPrefix(
        segments=(
            RawClusteredFeatureSegment(
                feature_ids=t.tensor([2, 3], dtype=t.int64),
                feature_magnitudes=t.tensor(
                    [magnitude, magnitude / 2],
                    dtype=t.float32,
                ),
                char_start=0,
                char_end=len(prefix_text),
            ),
        ),
        n_tokens=2,
    )


def _cfg(
    *,
    force_rebuild: bool = False,
    segment_size: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        context_cache=SimpleNamespace(
            segment_size=segment_size,
            num_buckets=2,
            force_rebuild=force_rebuild,
        ),
        extraction_max_tokens=32,
        extraction_threshold=0.0,
        filter_unaccepted_samples=True,
    )


def _write_sources(root: Path) -> tuple[Path, Path]:
    positive = root / "positive.parquet"
    negative = root / "negative.parquet"
    positive_table = pa.Table.from_pylist(
        [
            {"prefix_text": "delta", "qc_accepted": True},
            {"prefix_text": "alpha", "qc_accepted": True},
            {"prefix_text": "charlie", "qc_accepted": True},
        ]
    )
    positive_schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            positive_table.schema,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(
        positive_table.cast(positive_schema),
        positive,
    )
    negative_table = pa.Table.from_pylist(
        [
            {"prefix_text": "bravo", "qc_accepted": True},
            {"prefix_text": "alpha", "qc_accepted": True},
        ]
    )
    negative_schema = schema_with_stage1_continuation_contract(
        negative_table.schema,
        role="stage1c_negative",
    )
    pq.write_table(
        negative_table.cast(negative_schema),
        negative,
    )
    return positive, negative


def _canonical_population(
    positive: Path,
    negative: Path,
) -> CanonicalPrefixPopulation:
    return canonicalize_prefix_population(
        collect_unique_prefixes(
            positive,
            negative,
            filter_unaccepted=True,
        )
    )


def test_canonical_prefix_population_is_sorted_validated_and_immutable() -> None:
    source = {
        "zeta": make_prefix_hash("zeta"),
        "alpha": make_prefix_hash("alpha"),
    }

    population = canonicalize_prefix_population(source)
    source["replacement"] = make_prefix_hash("replacement")
    mutable_view = population.as_dict()
    mutable_view["another"] = make_prefix_hash("another")

    assert population.entries == (
        ("alpha", make_prefix_hash("alpha")),
        ("zeta", make_prefix_hash("zeta")),
    )
    assert population.as_dict() == {
        "alpha": make_prefix_hash("alpha"),
        "zeta": make_prefix_hash("zeta"),
    }


def test_canonical_prefix_population_rejects_every_invalid_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
        canonicalize_prefix_population([])  # ty: ignore[invalid-argument-type]
    with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
        canonicalize_prefix_population({1: "digest"})  # ty: ignore[invalid-argument-type]
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        CanonicalPrefixPopulation(entries=[])  # ty: ignore[invalid-argument-type]
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        CanonicalPrefixPopulation(
            entries=(["alpha", make_prefix_hash("alpha")],)  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(AxBenchBpbAdvError, match="strictly increasing"):
        CanonicalPrefixPopulation(
            entries=(
                ("zeta", make_prefix_hash("zeta")),
                ("alpha", make_prefix_hash("alpha")),
            )
        )
    with pytest.raises(AxBenchBpbAdvError, match="does not match"):
        CanonicalPrefixPopulation(entries=(("alpha", "0" * 16),))

    monkeypatch.setattr(cache_module, "make_prefix_hash", lambda _prefix: "same")
    with pytest.raises(AxBenchBpbAdvError, match="distinct prefix texts collide"):
        CanonicalPrefixPopulation(entries=(("alpha", "same"), ("zeta", "same")))
    with pytest.raises(BeartypeCallHintParamViolation, match="prefix_population"):
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=_paths(tmp_path, "invalid-population"),
            prefix_population={},  # ty: ignore[invalid-argument-type]
            base_model=object(),
            base_sae=object(),
            cluster_contract=_contract(tmp_path),
        )


def _paths(root: Path, run_tag: str) -> AxBenchBpbAdvPaths:
    return AxBenchBpbAdvPaths(output_root=root / "out", run_tag=run_tag)


def _install_build_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "extract_raw_clustered_prefix", _raw_extract)
    monkeypatch.setattr(
        cache_module,
        "reconstruct_clustered_replay_state",
        _replay_state,
    )


def _build_single(
    root: Path,
    *,
    run_tag: str,
    cfg: SimpleNamespace | None = None,
) -> tuple[AxBenchBpbAdvPaths, ClusterFeatureContract]:
    positive, negative = _write_sources(root)
    contract = _contract(root)
    paths = _paths(root, run_tag)
    original_extract = cache_module.extract_raw_clustered_prefix
    original_replay = cache_module.reconstruct_clustered_replay_state
    cache_module.__dict__["extract_raw_clustered_prefix"] = _raw_extract
    cache_module.__dict__["reconstruct_clustered_replay_state"] = _replay_state
    try:
        build_clustered_context_cache(
            cfg=_cfg() if cfg is None else cfg,
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    finally:
        cache_module.__dict__["extract_raw_clustered_prefix"] = original_extract
        cache_module.__dict__["reconstruct_clustered_replay_state"] = original_replay
    return paths, contract


def _expected_identity(
    contract: ClusterFeatureContract,
) -> dict[str, Any]:
    return cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )


def _selected_serialization_digest() -> str:
    """Subprocess entry point for the selected-file byte determinism gate."""

    contract = _contract(Path("/tmp/g3c-selected-prefix-determinism"))
    segment = cache_module._MaterializedSegment(
        context=ContextFeatureSegment(
            feature_ids=(1, 3),
            buckets=(0, 0),
            char_start=0,
            char_end=5,
        ),
        positive_percentiles_uint8=t.tensor([204, 153], dtype=t.uint8),
        n_positive=2,
        n_negative=0,
    )
    metadata = cache_module._selected_metadata(
        contract=contract,
        prefix_hash="0123456789abcdef",
        prefix_text_sha256=_SHA_A,
        extraction_identity_sha256=_SHA_B,
        raw_sha256=_SHA_C,
        generator_state_after_sha256=_SHA_D,
    )
    payload = cache_module._serialize_selected_prefix(
        [segment],
        metadata=metadata,
        contract=contract,
    )
    return hashlib.sha256(payload).hexdigest()


def test_single_rank_build_validate_load_and_completed_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, "single")

    manifest_path = build_clustered_context_cache(
        cfg=_cfg(),
        output_paths=paths,
        prefix_population=_canonical_population(positive, negative),
        base_model=object(),
        base_sae=object(),
        cluster_contract=contract,
    )

    manifest = validate_clustered_context_cache_manifest(
        paths=paths,
        contract=contract,
    )
    assert manifest_path == paths.context_cache_manifest()
    assert "ddp_world_size" not in manifest
    assert not paths.clustered_context_cache_partial_marker().exists()
    assert set(manifest["prefixes"]) == {
        make_prefix_hash(prefix) for prefix in ("alpha", "bravo", "charlie", "delta")
    }
    loaded = load_clustered_context_segments_by_prefix_hash(
        paths=paths,
        contract=contract,
    )
    assert set(loaded) == set(manifest["prefixes"])
    assert all(len(segments) == 1 for segments in loaded.values())
    selected_before = {
        prefix_hash: paths.context_cache_file(prefix_hash).read_bytes()
        for prefix_hash in manifest["prefixes"]
    }

    monkeypatch.setattr(
        cache_module,
        "extract_raw_clustered_prefix",
        lambda **_kwargs: pytest.fail("completed resume must not re-extract"),
    )
    assert (
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
        == manifest_path
    )
    assert selected_before == {
        prefix_hash: paths.context_cache_file(prefix_hash).read_bytes()
        for prefix_hash in manifest["prefixes"]
    }


def test_build_rejects_force_rebuild_for_completed_and_partial_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    completed = _paths(tmp_path, "completed")
    build_clustered_context_cache(
        cfg=_cfg(),
        output_paths=completed,
        prefix_population=_canonical_population(positive, negative),
        base_model=object(),
        base_sae=object(),
        cluster_contract=contract,
    )
    with pytest.raises(AxBenchBpbAdvError, match="force_rebuild=true is forbidden"):
        build_clustered_context_cache(
            cfg=_cfg(force_rebuild=True),
            output_paths=completed,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )

    partial = _paths(tmp_path, "partial")
    partial.context_cache_dir().mkdir(parents=True)
    evidence = partial.context_cache_dir() / "existing-evidence"
    evidence.write_bytes(b"immutable")
    with pytest.raises(AxBenchBpbAdvError, match="force_rebuild=true is forbidden"):
        build_clustered_context_cache(
            cfg=_cfg(force_rebuild=True),
            output_paths=partial,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    assert evidence.read_bytes() == b"immutable"
    assert not partial.clustered_context_cache_partial_marker().exists()


def test_partial_marker_install_readback_world_pin_and_writer_validation(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, "marker")
    expected = cache_module._partial_marker_document(
        contract=contract,
        extraction_identity=_expected_identity(contract),
        prefix_population_sha256=_SHA_A,
        world_size=1,
    )
    cache_module._ensure_partial_marker(
        paths=paths,
        expected=expected,
        rank=0,
        world_size=1,
        contract=contract,
    )
    marker_path = paths.clustered_context_cache_partial_marker()
    assert json.loads(marker_path.read_bytes()) == expected

    changed_world = {**expected, "ddp_world_size": 2}
    with pytest.raises(AxBenchBpbAdvError, match="pinned DDP world size"):
        cache_module._ensure_partial_marker(
            paths=paths,
            expected=changed_world,
            rank=0,
            world_size=1,
            contract=contract,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="world_size"):
        cache_module._partial_marker_document(
            contract=contract,
            extraction_identity=_expected_identity(contract),
            prefix_population_sha256=_SHA_A,
            world_size=0,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="prefix_population_sha256"):
        cache_module._partial_marker_document(
            contract=contract,
            extraction_identity=_expected_identity(contract),
            prefix_population_sha256="bad",
            world_size=1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field",
        "extra_field",
        "schema_version",
        "kind",
        "status",
        "feature_contract_fingerprint",
        "extraction_identity_sha256",
        "prefix_population_sha256",
        "ddp_world_size",
    ],
)
def test_partial_marker_validation_failure_map_raises(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract = _contract(tmp_path)
    marker = cache_module._partial_marker_document(
        contract=contract,
        extraction_identity=_expected_identity(contract),
        prefix_population_sha256=_SHA_A,
        world_size=1,
    )
    if mutation == "missing_field":
        marker.pop("kind")
    elif mutation == "extra_field":
        marker["unexpected"] = True
    elif mutation in {"schema_version", "ddp_world_size"}:
        marker[mutation] = False
    elif mutation in {"kind", "status"}:
        marker[mutation] = "wrong"
    elif mutation == "feature_contract_fingerprint":
        marker[mutation] = _SHA_D
    elif mutation in {
        "extraction_identity_sha256",
        "prefix_population_sha256",
    }:
        marker[mutation] = "invalid"
    else:
        raise AssertionError(f"unknown marker mutation {mutation!r}")
    with pytest.raises(AxBenchBpbAdvError):
        cache_module._validate_partial_marker_document(
            marker,
            contract=contract,
        )


@pytest.mark.parametrize("payload", [b"[1, 2, 3]", b"{"])
def test_read_json_mapping_rejects_non_mapping_or_malformed_snapshot(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(AxBenchBpbAdvError):
        cache_module._read_json_mapping(path, surface="test JSON")


def test_partial_marker_is_unlinked_only_after_manifest_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, "ordering")
    original_validate = cache_module.validate_clustered_context_cache_manifest
    calls = 0

    def _fail_post_install_validation(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-install validation failure")
        return original_validate(**kwargs)

    monkeypatch.setattr(
        cache_module,
        "validate_clustered_context_cache_manifest",
        _fail_post_install_validation,
    )
    with pytest.raises(AxBenchBpbAdvError, match="manifest commit failed"):
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    assert paths.context_cache_manifest().is_file()
    assert paths.clustered_context_cache_partial_marker().is_file()

    # Asserting the two coexist proves the ORDERING, but not that the state is
    # RECOVERABLE -- and this is the state a crash between manifest install and
    # marker unlink leaves behind. Resume must succeed on it: the manifest is
    # the sole completeness marker, so every rank validates it and returns
    # before ever consulting the leftover marker.
    monkeypatch.setattr(
        cache_module,
        "validate_clustered_context_cache_manifest",
        original_validate,
    )
    resumed = build_clustered_context_cache(
        cfg=_cfg(),
        output_paths=paths,
        prefix_population=_canonical_population(positive, negative),
        base_model=object(),
        base_sae=object(),
        cluster_contract=contract,
    )
    assert resumed == paths.context_cache_manifest()


@pytest.mark.parametrize("artifact_kind", ["raw", "selected"])
def test_missing_partial_marker_rejects_orphaned_cache_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, f"orphan-{artifact_kind}")
    prefix_hash = make_prefix_hash("alpha")
    artifact_path = (
        paths.clustered_context_cache_raw_file(prefix_hash)
        if artifact_kind == "raw"
        else paths.context_cache_file(prefix_hash)
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"orphaned-immutable-evidence")

    with pytest.raises(
        AxBenchBpbAdvError,
        match="artifacts exist without the exact incomplete marker",
    ):
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    assert artifact_path.read_bytes() == b"orphaned-immutable-evidence"
    assert not paths.clustered_context_cache_partial_marker().exists()
    assert not paths.context_cache_manifest().exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0123456789abcdef", "0123456789abcdef"),
        ("../0123456789abcd", None),
        ("0123456789abcde", None),
        ("0123456789abcdeg", None),
        ("0123456789ABCDEF", None),
        ("", None),
    ],
)
def test_valid_prefix_hash_slug(value: str, expected: str | None) -> None:
    assert cache_module._valid_prefix_hash_slug(value) == expected


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_clustered_loader_rejects_missing_or_corrupt_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag=f"loader-{failure}")
    manifest = json.loads(paths.context_cache_manifest().read_bytes())
    prefix_hash = next(iter(manifest["prefixes"]))
    selected_path = paths.context_cache_file(prefix_hash)
    if failure == "missing":
        selected_path.unlink()
    else:
        selected_path.write_bytes(selected_path.read_bytes() + b"corrupt")
    with pytest.raises(AxBenchBpbAdvError):
        load_clustered_context_segments_by_prefix_hash(
            paths=paths,
            contract=contract,
        )


def test_manifest_loader_rechecks_selected_snapshot_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="loader-toctou")
    manifest = json.loads(paths.context_cache_manifest().read_bytes())
    prefix_hash = next(iter(manifest["prefixes"]))
    selected_path = paths.context_cache_file(prefix_hash)
    original_validate = cache_module.validate_clustered_context_cache_manifest

    def _validate_then_replace(**kwargs: Any) -> dict[str, Any]:
        result = original_validate(**kwargs)
        selected_path.write_bytes(selected_path.read_bytes() + b"replaced")
        return result

    monkeypatch.setattr(
        cache_module,
        "validate_clustered_context_cache_manifest",
        _validate_then_replace,
    )
    with pytest.raises(AxBenchBpbAdvError, match="content digest mismatch"):
        load_clustered_context_segments_by_prefix_hash(
            paths=paths,
            contract=contract,
        )


def test_manifest_validator_hashes_and_parses_each_prefix_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="snapshot-count")
    manifest = json.loads(paths.context_cache_manifest().read_bytes())
    raw_paths = {
        paths.clustered_context_cache_raw_file(prefix_hash) for prefix_hash in manifest["prefixes"]
    }
    selected_paths = {paths.context_cache_file(prefix_hash) for prefix_hash in manifest["prefixes"]}
    counts = dict.fromkeys(raw_paths | selected_paths, 0)
    original_read_bytes = Path.read_bytes

    def _counted_read_bytes(path: Path) -> bytes:
        if path in counts:
            counts[path] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _counted_read_bytes)
    validate_clustered_context_cache_manifest(paths=paths, contract=contract)
    assert set(counts.values()) == {1}


# Each case is pinned to the SPECIFIC guard it targets. A bare
# pytest.raises(AxBenchBpbAdvError) only proves "the malformed manifest was
# rejected", which this validator's defense-in-depth satisfies through some
# other check -- so 9 of 25 guard mutations survived while every case still
# passed, hiding two real holes (a self-consistent forged slug, and an extra
# entry field). Attribution is what makes a survivor visible.
_MANIFEST_MUTATIONS = (
    ("missing_top_field", "fields mismatch"),
    ("extra_top_field", "fields mismatch"),
    ("schema_version", "unsupported clustered context-cache schema"),
    ("kind", "not a clustered BPB-Adv manifest"),
    ("feature_contract_fingerprint", "feature contract does not match"),
    ("extraction_identity_type", "extraction_identity must be an object"),
    ("extraction_identity_contract_mismatch", "extraction identity disagrees with contract"),
    ("prefix_population_sha256", "is not a SHA-256 value"),
    ("generator_state_sha256", "is not a SHA-256 value"),
    ("selected_content_sha256", "is not a SHA-256 value"),
    ("prefixes_type", "prefixes must be an object"),
    ("prefixes_empty", "prefixes must not be empty|final generator state"),
    ("prefix_slug_traversal", "invalid prefix hash|prefix-population digest"),
    ("prefix_slug_odd_length", "invalid prefix hash|prefix-population digest"),
    ("prefix_slug_non_hex", "invalid prefix hash|prefix-population digest"),
    ("entry_type", "must be an object"),
    ("entry_fields", "fields mismatch|is invalid"),
    ("entry_prefix_text_sha256", "is invalid"),
    ("entry_raw_sha256", "is invalid"),
    ("entry_selected_sha256", "is invalid"),
    ("entry_generator_state_after_sha256", "is invalid"),
    ("entry_n_segments", "is invalid"),
    ("entry_n_features_total", "is invalid"),
    ("entry_n_positive_total", "is invalid"),
    ("entry_n_negative_total", "is invalid"),
    ("entry_boolean_count", "is invalid"),
    ("entry_inconsistent_counts", "feature counts are inconsistent"),
    ("selected_aggregate", "selected-content aggregate digest mismatch"),
    ("population_aggregate", "prefix-population digest mismatch"),
    ("final_generator_state", "final generator state does not match"),
    ("structural_count", "structural count mismatch"),
    ("missing_raw_file", "artifact is missing|could not read"),
    ("missing_selected_file", "artifact is missing|could not read"),
    ("corrupt_raw_file", "content digest mismatch|could not read"),
    ("corrupt_selected_file", "content digest mismatch|could not load selected prefix"),
)


def _replace_prefix_slug(
    manifest: dict[str, Any],
    replacement: str,
) -> None:
    prefixes = manifest["prefixes"]
    prefix_hash = next(iter(prefixes))
    prefixes[replacement] = prefixes.pop(prefix_hash)


def _mutate_manifest_or_files(
    mutation: str,
    *,
    manifest: dict[str, Any],
    paths: AxBenchBpbAdvPaths,
) -> bool:
    """Apply one corruption; return whether the manifest bytes changed."""

    prefixes = manifest["prefixes"]
    prefix_hash = next(iter(prefixes))
    entry = prefixes[prefix_hash]
    if mutation == "missing_top_field":
        manifest.pop("kind")
    elif mutation == "extra_top_field":
        manifest["unexpected"] = True
    elif mutation == "schema_version":
        manifest["schema_version"] = 999
    elif mutation == "kind":
        manifest["kind"] = "wrong"
    elif mutation == "feature_contract_fingerprint":
        manifest["feature_contract_fingerprint"] = _SHA_D
    elif mutation == "extraction_identity_type":
        manifest["extraction_identity"] = []
    elif mutation == "extraction_identity_contract_mismatch":
        manifest["extraction_identity"]["segment_size"] = 99
    elif mutation in {
        "prefix_population_sha256",
        "generator_state_sha256",
        "selected_content_sha256",
    }:
        manifest[mutation] = "not-a-sha256"
    elif mutation == "prefixes_type":
        manifest["prefixes"] = []
    elif mutation == "prefixes_empty":
        manifest["prefixes"] = {}
    elif mutation == "prefix_slug_traversal":
        _replace_prefix_slug(manifest, "../0123456789abc")
    elif mutation == "prefix_slug_odd_length":
        _replace_prefix_slug(manifest, "0123456789abcde")
    elif mutation == "prefix_slug_non_hex":
        _replace_prefix_slug(manifest, "g" * 16)
    elif mutation == "entry_type":
        prefixes[prefix_hash] = []
    elif mutation == "entry_fields":
        entry.pop("raw_sha256")
    elif mutation.startswith("entry_") and mutation.removeprefix("entry_") in {
        "prefix_text_sha256",
        "raw_sha256",
        "selected_sha256",
        "generator_state_after_sha256",
    }:
        entry[mutation.removeprefix("entry_")] = "invalid"
    elif mutation.startswith("entry_") and mutation.removeprefix("entry_") in {
        "n_segments",
        "n_features_total",
        "n_positive_total",
        "n_negative_total",
    }:
        entry[mutation.removeprefix("entry_")] = -1
    elif mutation == "entry_boolean_count":
        entry["n_segments"] = False
    elif mutation == "entry_inconsistent_counts":
        entry["n_features_total"] += 1
    elif mutation == "selected_aggregate":
        manifest["selected_content_sha256"] = _SHA_D
    elif mutation == "population_aggregate":
        manifest["prefix_population_sha256"] = _SHA_D
    elif mutation == "final_generator_state":
        manifest["generator_state_sha256"] = _SHA_D
    elif mutation == "structural_count":
        entry["n_segments"] += 1
    elif mutation == "missing_raw_file":
        paths.clustered_context_cache_raw_file(prefix_hash).unlink()
        return False
    elif mutation == "missing_selected_file":
        paths.context_cache_file(prefix_hash).unlink()
        return False
    elif mutation == "corrupt_raw_file":
        raw_path = paths.clustered_context_cache_raw_file(prefix_hash)
        raw_path.write_bytes(raw_path.read_bytes() + b"corrupt")
        return False
    elif mutation == "corrupt_selected_file":
        selected_path = paths.context_cache_file(prefix_hash)
        selected_path.write_bytes(selected_path.read_bytes() + b"corrupt")
        return False
    else:
        raise AssertionError(f"unknown manifest mutation {mutation!r}")
    return True


@pytest.mark.parametrize(("mutation", "match"), _MANIFEST_MUTATIONS)
def test_manifest_validation_failure_map_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag=f"manifest-{mutation}")
    manifest_path = paths.context_cache_manifest()
    manifest = json.loads(manifest_path.read_bytes())
    if _mutate_manifest_or_files(
        mutation,
        manifest=manifest,
        paths=paths,
    ):
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_clustered_context_cache_manifest(
            paths=paths,
            contract=contract,
        )


def test_manifest_expected_identity_and_population_mismatches_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="manifest-expected")
    expected_identity = _expected_identity(contract)
    changed_identity = {**expected_identity, "extraction_max_tokens": 31}
    with pytest.raises(AxBenchBpbAdvError, match="extraction identity mismatch"):
        validate_clustered_context_cache_manifest(
            paths=paths,
            contract=contract,
            expected_extraction_identity=changed_identity,
        )
    with pytest.raises(AxBenchBpbAdvError, match="prefix population mismatch"):
        validate_clustered_context_cache_manifest(
            paths=paths,
            contract=contract,
            expected_prefix_population_sha256=_SHA_D,
        )


def test_manifest_digest_shape_checks_use_canonical_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="manifest-shape-helper")
    manifest_path = paths.context_cache_manifest()
    manifest = json.loads(manifest_path.read_bytes())
    manifest["generator_state_sha256"] = "invalid-shape"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[Any] = []
    original_is_sha256_hex = cache_module.is_sha256_hex

    def _record_is_sha256_hex(value: Any) -> bool:
        observed.append(value)
        return original_is_sha256_hex(value)

    monkeypatch.setattr(
        cache_module,
        "is_sha256_hex",
        _record_is_sha256_hex,
    )
    with pytest.raises(AxBenchBpbAdvError, match="not a SHA-256"):
        validate_clustered_context_cache_manifest(
            paths=paths,
            contract=contract,
        )
    assert "invalid-shape" in observed
    assert "_is_sha256" not in cache_module.__dict__


def test_manifest_writer_rejects_reader_invalid_evidence_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="writer-symmetry")
    manifest = json.loads(paths.context_cache_manifest().read_bytes())
    prefixes = manifest["prefixes"]
    prefix_hash = next(iter(prefixes))
    prefixes[prefix_hash]["raw_sha256"] = "invalid"
    with pytest.raises(AxBenchBpbAdvError, match="raw_sha256 is invalid"):
        cache_module._manifest_document(
            contract=contract,
            extraction_identity=manifest["extraction_identity"],
            prefix_population_sha256=manifest["prefix_population_sha256"],
            generator_state_sha256=manifest["generator_state_sha256"],
            prefix_entries=prefixes,
        )


def test_manifest_commit_prevalidates_referenced_files_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, "precommit-validation")

    def _fail_precommit(**_kwargs: Any) -> None:
        raise RuntimeError("injected precommit referenced-file failure")

    monkeypatch.setattr(
        cache_module,
        "_validate_manifest_prefix_files",
        _fail_precommit,
    )
    with pytest.raises(AxBenchBpbAdvError, match="manifest commit failed"):
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    assert not paths.context_cache_manifest().exists()
    assert paths.clustered_context_cache_partial_marker().is_file()


def _transaction_worker(
    rank: int,
    world_size: int,
    root_text: str,
    run_tag: str,
    failure_phase: str | None,
) -> dict[str, Any]:
    root = Path(root_text)
    contract = _contract(root)
    cache_module.__dict__["extract_raw_clustered_prefix"] = _raw_extract
    cache_module.__dict__["reconstruct_clustered_replay_state"] = _replay_state
    cfg = _cfg(
        force_rebuild=failure_phase == "force_rebuild_skew" and rank == 0,
        segment_size=(5 if failure_phase == "preflight" and rank == 0 else 4),
    )
    if failure_phase == "preflight_identity_skew" and rank == 0:
        cfg.extraction_max_tokens = 31

    if failure_phase == "prefix_population":
        original_as_dict = cache_module.CanonicalPrefixPopulation.as_dict

        def _fail_population(self: Any) -> dict[str, str]:
            if rank == 0:
                raise RuntimeError("injected prefix population failure")
            return original_as_dict(self)

        cache_module.CanonicalPrefixPopulation.as_dict = _fail_population
    elif failure_phase == "completed_manifest_validation":
        original_validate = cache_module.validate_clustered_context_cache_manifest

        def _fail_completed_validation(**kwargs: Any) -> dict[str, Any]:
            if rank == 0:
                raise RuntimeError("injected completed-manifest validation failure")
            return original_validate(**kwargs)

        cache_module.__dict__["validate_clustered_context_cache_manifest"] = (
            _fail_completed_validation
        )
    elif failure_phase == "partial_marker_install":
        original_install = cache_module._install_once

        def _fail_partial_install(
            payload: bytes,
            path: Path,
            *,
            surface: str,
        ) -> None:
            if rank == 0 and surface.endswith("partial marker"):
                raise RuntimeError("injected partial-marker install failure")
            original_install(payload, path, surface=surface)

        cache_module.__dict__["_install_once"] = _fail_partial_install
    elif failure_phase == "partial_marker_read_back":
        original_read_mapping = cache_module._read_json_mapping

        def _fail_partial_read(path: Path, *, surface: str) -> dict[str, Any]:
            if rank == 1 and surface == "clustered partial-cache marker":
                raise RuntimeError("injected partial-marker read-back failure")
            return original_read_mapping(path, surface=surface)

        cache_module.__dict__["_read_json_mapping"] = _fail_partial_read
    elif failure_phase == "raw_extraction":
        original_extract = cache_module.extract_raw_clustered_prefix

        def _fail_raw(**kwargs: Any) -> RawClusteredPrefix:
            if rank == 0:
                raise RuntimeError("injected raw extraction failure")
            return original_extract(**kwargs)

        cache_module.__dict__["extract_raw_clustered_prefix"] = _fail_raw
    elif failure_phase == "replay":
        original_replay = cache_module.reconstruct_clustered_replay_state

        def _fail_replay(
            resolved: ClusterFeatureContract,
        ) -> ClusteredReplayState:
            if rank == 1:
                raise RuntimeError("injected replay failure")
            return original_replay(resolved)

        cache_module.__dict__["reconstruct_clustered_replay_state"] = _fail_replay
    elif failure_phase == "manifest_commit":
        original_install = cache_module._install_once

        def _fail_manifest_install(
            payload: bytes,
            path: Path,
            *,
            surface: str,
        ) -> None:
            if rank == 0 and surface.endswith("manifest"):
                raise RuntimeError("injected manifest commit failure")
            original_install(payload, path, surface=surface)

        cache_module.__dict__["_install_once"] = _fail_manifest_install
    elif failure_phase == "final_visibility":
        original_validate = cache_module.validate_clustered_context_cache_manifest

        def _fail_final_visibility(**kwargs: Any) -> dict[str, Any]:
            if rank == 1:
                raise RuntimeError("injected final visibility failure")
            return original_validate(**kwargs)

        cache_module.__dict__["validate_clustered_context_cache_manifest"] = _fail_final_visibility

    try:
        build_clustered_context_cache(
            cfg=cfg,
            output_paths=_paths(root, run_tag),
            prefix_population=_canonical_population(
                root / "positive.parquet",
                root / "negative.parquet",
            ),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )
    except BaseException as exc:
        return {
            "raised": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"raised": False, "error": None}


_FAILURE_PHASES = (
    (
        "preflight",
        "preflight and prefix population resolution failed",
    ),
    (
        "preflight_identity_skew",
        "DDP ranks resolved different extraction identities",
    ),
    (
        "prefix_population",
        "preflight and prefix population resolution failed",
    ),
    (
        "completed_manifest_validation",
        "completed-manifest validation failed",
    ),
    (
        "force_rebuild_skew",
        "completed-manifest validation failed",
    ),
    (
        "partial_marker_install",
        "partial-marker setup failed",
    ),
    (
        "partial_marker_read_back",
        "partial-marker validation failed",
    ),
    (
        "raw_extraction",
        "raw extraction failed",
    ),
    (
        "replay",
        "clustered selection replay failed",
    ),
    (
        "manifest_commit",
        "manifest commit failed",
    ),
    (
        "final_visibility",
        "final manifest visibility failed",
    ),
)


@pytest.mark.slow
@pytest.mark.parametrize(("failure_phase", "expected_error"), _FAILURE_PHASES)
def test_one_rank_failure_synchronously_raises_on_every_rank_without_hang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    expected_error: str,
) -> None:
    root = tmp_path / failure_phase
    root.mkdir()
    _write_sources(root)
    run_tag = "transaction"
    if failure_phase == "completed_manifest_validation":
        _build_single(root, run_tag=run_tag)
    monkeypatch.setattr(multiprocess_helpers, "_RESULT_TIMEOUT_S", 20)

    results = multiprocess_helpers.run_gloo_multiprocess(
        2,
        _transaction_worker,
        str(root),
        run_tag,
        failure_phase,
    )

    assert all(result["raised"] for result in results), results
    assert all(expected_error in result["error"] for result in results), results


@pytest.mark.slow
def test_resume_replays_existing_prefixes_and_matches_different_world_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    resumed_paths = _paths(tmp_path, "resumed-world-one")
    original_install = cache_module._install_once

    def _interrupt_manifest_commit(
        payload: bytes,
        path: Path,
        *,
        surface: str,
    ) -> None:
        if surface.endswith("manifest"):
            raise RuntimeError("interrupt after every selected owner write")
        original_install(payload, path, surface=surface)

    monkeypatch.setattr(
        cache_module,
        "_install_once",
        _interrupt_manifest_commit,
    )
    with pytest.raises(AxBenchBpbAdvError, match="manifest commit failed"):
        build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=resumed_paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )
    prefix_hashes = {make_prefix_hash(prefix) for prefix in ("alpha", "bravo", "charlie", "delta")}
    selected_before_resume = {
        prefix_hash: resumed_paths.context_cache_file(prefix_hash).read_bytes()
        for prefix_hash in prefix_hashes
    }
    assert resumed_paths.clustered_context_cache_partial_marker().is_file()

    monkeypatch.setattr(cache_module, "_install_once", original_install)
    build_clustered_context_cache(
        cfg=_cfg(),
        output_paths=resumed_paths,
        prefix_population=_canonical_population(positive, negative),
        base_model=object(),
        base_sae=object(),
        cluster_contract=contract,
    )
    selected_after_resume = {
        prefix_hash: resumed_paths.context_cache_file(prefix_hash).read_bytes()
        for prefix_hash in prefix_hashes
    }
    assert selected_after_resume == selected_before_resume
    assert not resumed_paths.clustered_context_cache_partial_marker().exists()

    monkeypatch.setattr(multiprocess_helpers, "_RESULT_TIMEOUT_S", 20)
    results = multiprocess_helpers.run_gloo_multiprocess(
        2,
        _transaction_worker,
        str(tmp_path),
        "fresh-world-two",
        None,
    )
    assert all(not result["raised"] for result in results), results
    world_two_paths = _paths(tmp_path, "fresh-world-two")
    assert {
        prefix_hash: world_two_paths.context_cache_file(prefix_hash).read_bytes()
        for prefix_hash in prefix_hashes
    } == selected_after_resume
    assert (
        world_two_paths.context_cache_manifest().read_bytes()
        == resumed_paths.context_cache_manifest().read_bytes()
    )


@pytest.mark.slow
def test_selected_prefix_serialization_is_stable_across_process_hash_seeds() -> None:
    script = (
        "from tests.axbench_bpb_adv.unit."
        "test_clustered_context_cache_transaction "
        "import _selected_serialization_digest; "
        "print(_selected_serialization_digest())"
    )
    digests: set[str] = set()
    for seed in ("0", "1", "42", "1337", "random"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, sorted(digests)


def test_collective_shape_has_eight_call_sites_and_no_barrier() -> None:
    """Pin the collective shape by counting CALLS, not source substrings.

    The substring count said "ten sites" because it also counted the two helper
    DEFINITIONS -- `_all_gather_object(` appears 7 times (1 def + 6 calls) and
    `_broadcast_rank_zero_status(` 3 times (1 def + 2 calls). There are eight
    transaction-level call sites. Counting text would also have been satisfied
    by a mention in a comment or docstring, so this walks the AST instead: only
    a real `Call` node counts.
    """
    tree = ast.parse(inspect.getsource(cache_module))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("_all_gather_object") == 6
    assert calls.count("_broadcast_rank_zero_status") == 2

    # The two primitive torch.distributed sites live inside those helpers.
    attribute_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert attribute_calls.count("all_gather_object") == 1
    assert attribute_calls.count("broadcast_object_list") == 1
    assert attribute_calls.count("barrier") == 0


def test_collective_helpers_short_circuit_and_fail_on_missing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cache_module.dist,
        "all_gather_object",
        lambda *_args, **_kwargs: pytest.fail("world_size=1 must not enter a collective"),
    )
    value = {"rank": 0, "error": None}
    assert cache_module._all_gather_object(value, world_size=1) == [value]

    def _leave_missing(
        gathered: list[dict[str, Any] | None],
        _value: dict[str, Any],
    ) -> None:
        gathered[0] = {"rank": 0, "error": None}

    monkeypatch.setattr(
        cache_module.dist,
        "all_gather_object",
        _leave_missing,
    )
    with pytest.raises(AxBenchBpbAdvError, match="missing rank payload"):
        cache_module._all_gather_object(value, world_size=2)

    with pytest.raises(AxBenchBpbAdvError, match="unexpectedly absent"):
        cache_module._broadcast_rank_zero_status(
            None,
            rank=0,
            world_size=1,
        )

    def _malform_broadcast(payload: list[Any], *, src: int) -> None:
        assert src == 0
        payload[0] = "malformed"

    monkeypatch.setattr(
        cache_module.dist,
        "broadcast_object_list",
        _malform_broadcast,
    )
    with pytest.raises(AxBenchBpbAdvError, match="malformed payload"):
        cache_module._broadcast_rank_zero_status(
            {"error": None},
            rank=0,
            world_size=2,
        )


def test_validate_ddp_documents_irreducible_precollective_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_module._validate_ddp(rank=0, world_size=1)
    with pytest.raises(BeartypeCallHintParamViolation, match="rank"):
        cache_module._validate_ddp(rank=True, world_size=1)
    with pytest.raises(AxBenchBpbAdvError, match="invalid DDP coordinates"):
        cache_module._validate_ddp(rank=1, world_size=1)
    monkeypatch.setattr(cache_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(cache_module.dist, "is_initialized", lambda: False)
    with pytest.raises(AxBenchBpbAdvError, match="initialized"):
        cache_module._validate_ddp(rank=0, world_size=2)
    monkeypatch.setattr(cache_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cache_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(cache_module.dist, "get_rank", lambda: 1)
    with pytest.raises(AxBenchBpbAdvError, match="arguments disagree"):
        cache_module._validate_ddp(rank=0, world_size=2)


@pytest.mark.parametrize(
    ("claimed_rank", "claimed_world", "actual_rank", "actual_world", "expect_raise"),
    [
        # G3c-D2: the defect. A rank inside a real 2-rank group claims to be a
        # lone single-rank builder. Previously accepted, so it skipped every
        # collective, committed the cache and returned SUCCESS while its peer
        # blocked in the first gather.
        (0, 1, 0, 2, True),
        # The same lie told by the non-zero rank: it must still claim rank=0 to
        # pass the basic coordinate check, so this is the shape the defect
        # actually takes on rank 1.
        (0, 1, 1, 2, True),
        # Honest coordinates inside a group still pass.
        (0, 2, 0, 2, False),
        (1, 2, 1, 2, False),
        # A genuine single-rank group.
        (0, 1, 0, 1, False),
        # Self-consistent but globally wrong: rank 1 told it is rank 0.
        (0, 2, 1, 2, True),
    ],
)
def test_validate_ddp_consults_the_live_group_whenever_one_exists(
    claimed_rank: int,
    claimed_world: int,
    actual_rank: int,
    actual_world: int,
    expect_raise: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(cache_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cache_module.dist, "get_rank", lambda: actual_rank)
    monkeypatch.setattr(cache_module.dist, "get_world_size", lambda: actual_world)

    if expect_raise:
        with pytest.raises(AxBenchBpbAdvError, match="disagree with the initialized"):
            cache_module._validate_ddp(rank=claimed_rank, world_size=claimed_world)
    else:
        cache_module._validate_ddp(rank=claimed_rank, world_size=claimed_world)


def test_validate_ddp_still_allows_single_rank_without_a_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one genuinely irreducible case: no group means no channel to agree on.

    Guards against over-correcting G3c-D2 into rejecting ordinary single-process
    runs, which never initialize a process group at all.
    """
    monkeypatch.setattr(cache_module.dist, "is_initialized", lambda: False)
    cache_module._validate_ddp(rank=0, world_size=1)
    with pytest.raises(AxBenchBpbAdvError, match="requires an initialized"):
        cache_module._validate_ddp(rank=0, world_size=2)


def test_status_error_text_never_raises_on_a_hostile_exception() -> None:
    """G3c-D3: this runs inside every gathered `except` before its collective.

    `__str__` is arbitrary user code. If it raises, the old inline f-string let
    the new exception escape the handler, so that rank never joined the gather
    its peers were already blocked in -- turning a synchronized application
    failure into an opaque transport error or a hang.
    """

    class HostileStr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("stringification failed")

    class HostileType(Exception):
        def __str__(self) -> str:
            raise RuntimeError("stringification failed")

        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

    assert cache_module._status_error_text(ValueError("ordinary")) == ("ValueError: ordinary")
    for hostile in (HostileStr(), HostileType()):
        text = cache_module._status_error_text(hostile)
        assert "rendering failed" in text
        assert type(hostile).__name__ in text

    # A BaseException raised from __str__ must not escape either.
    class Interrupting(Exception):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    assert "rendering failed" in cache_module._status_error_text(Interrupting())


def test_every_gathered_handler_routes_through_the_guarded_renderer() -> None:
    """No gathered phase may re-introduce a bare f-string stringification.

    Testing the helper alone would not catch a NEW phase that formats inline;
    the defect is per-call-site, so the pin has to be per-call-site too.
    """
    source = inspect.getsource(cache_module)
    body = source.split("def _status_error_text", 1)[1].split("\ndef ", 1)[1]
    assert 'f"{type(exc).__name__}: {exc}"' not in body
    assert body.count("_status_error_text(exc)") == 8


def test_manifest_rejects_a_self_consistent_forged_traversal_slug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slug guard must stand alone against an internally consistent forgery.

    The `prefix_slug_traversal` case in the failure map renames one key and
    thereby breaks the population aggregate, so it proves the AGGREGATE fires --
    not the slug check. Removing the slug check entirely still left that case
    passing. Here every aggregate is recomputed for the forged key, so the slug
    charset check is the only remaining defense; `"../0123456789abc"` is exactly
    16 characters, so the length check passes too.

    This matters because the key is interpolated into a filesystem path, and a
    traversal-shaped key escapes the run's raw/selected directories.
    """
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="forged-slug")
    manifest_path = paths.context_cache_manifest()
    manifest = json.loads(manifest_path.read_bytes())

    original_slug = next(iter(manifest["prefixes"]))
    forged_slug = "../0123456789abc"
    assert len(forged_slug) == len(original_slug) == 16
    entry = manifest["prefixes"].pop(original_slug)
    manifest["prefixes"][forged_slug] = entry

    # Make the forgery internally consistent: recompute every aggregate the
    # validator cross-checks, exactly as _manifest_document would have.
    manifest["selected_content_sha256"] = cache_module._sha256_json(
        [[forged_slug, entry["selected_sha256"]]]
    )
    manifest["prefix_population_sha256"] = cache_module._sha256_json(
        [
            {
                "prefix_hash": forged_slug,
                "prefix_text_sha256": entry["prefix_text_sha256"],
            }
        ]
    )
    manifest["generator_state_sha256"] = entry["generator_state_after_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="invalid prefix hash"):
        validate_clustered_context_cache_manifest(paths=paths, contract=contract)


def test_manifest_rejects_an_extra_field_on_a_prefix_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry field-set must be exact in BOTH directions, like the top level.

    The failure map's `entry_fields` case only removes a key, which the
    per-field validation catches anyway via `.get()` returning None. Nothing
    downstream inspects for UNEXPECTED keys -- every other check reads only the
    keys it expects -- so the field-set guard is the sole defense against a
    smuggled extra field, and no test exercised that direction. The top-level
    manifest already tests both (`missing_top_field` / `extra_top_field`).
    """
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="entry-extra")
    manifest_path = paths.context_cache_manifest()
    manifest = json.loads(manifest_path.read_bytes())

    entry = next(iter(manifest["prefixes"].values()))
    entry["unexpected_extra_key"] = "smuggled"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="fields mismatch"):
        validate_clustered_context_cache_manifest(paths=paths, contract=contract)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        # Each corruption below is a shape NOT redundantly caught by a
        # downstream check. The failure map's equivalents use `[]`, which the
        # adjacent emptiness/field-set checks catch anyway -- so removing these
        # type guards left every existing case passing while these shapes
        # escaped as a raw TypeError, breaking the "always AxBenchBpbAdvError"
        # contract the module's callers rely on.
        ("identity_none", "extraction_identity must be an object"),
        ("identity_int", "extraction_identity must be an object"),
        ("prefixes_nonempty_list", "prefixes must be an object"),
        ("entry_none", "must be an object"),
    ],
)
def test_manifest_type_guards_raise_the_contract_error_not_a_raw_typeerror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
) -> None:
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag=f"typeguard-{case}")
    manifest_path = paths.context_cache_manifest()
    manifest = json.loads(manifest_path.read_bytes())

    if case == "identity_none":
        manifest["extraction_identity"] = None
    elif case == "identity_int":
        manifest["extraction_identity"] = 42
    elif case == "prefixes_nonempty_list":
        manifest["prefixes"] = ["0123456789abcdef"]
    elif case == "entry_none":
        slug = next(iter(manifest["prefixes"]))
        manifest["prefixes"][slug] = None
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_clustered_context_cache_manifest(paths=paths, contract=contract)


def test_downstream_loader_reads_each_selected_prefix_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consumer must parse the bytes it hashed, same as the validator.

    The existing one-read counter only instruments
    `validate_clustered_context_cache_manifest`'s internal reads, so the
    DOWNSTREAM loader -- the last code to touch these artifacts before they
    reach the eval -- had no reopen detection at all. Reintroducing a pathname
    reopen there would restore the campaign's hash-then-reopen defect at the
    one site nothing else re-validates afterwards.
    """
    _install_build_overrides(monkeypatch)
    paths, contract = _build_single(tmp_path, run_tag="loader-one-read")
    manifest = json.loads(paths.context_cache_manifest().read_bytes())
    prefix_hash = next(iter(manifest["prefixes"]))
    selected_path = paths.context_cache_file(prefix_hash)

    reads: list[str] = []
    real_read_bytes = Path.read_bytes

    def _counting(self: Path) -> bytes:
        if self == selected_path:
            reads.append(str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _counting)
    segments = load_clustered_context_segments_by_prefix_hash(
        paths=paths,
        contract=contract,
    )

    assert segments[prefix_hash]
    # Two reads, deliberately: manifest validation takes one snapshot and the
    # loader takes its own afterwards, each internally binding its digest to
    # the bytes it parses. That second, independent snapshot is the point --
    # it is what detects a replacement between validation and consumption. A
    # third read would mean the loader had gone back to the pathname after
    # hashing, which is the defect this pins.
    assert len(reads) == 2, reads


@pytest.mark.parametrize(
    ("statuses", "expect_raise"),
    [
        # Ranks disagree: one validated the completed cache, one did not.
        (
            [
                {"rank": 0, "error": None, "valid": True, "exists": True},
                {"rank": 1, "error": None, "valid": False, "exists": False},
            ],
            True,
        ),
        # The dissenter saw the file but did not find it valid -- the shape a
        # partially-visible network filesystem produces.
        (
            [
                {"rank": 0, "error": None, "valid": True, "exists": True},
                {"rank": 1, "error": None, "valid": False, "exists": True},
            ],
            True,
        ),
        # Unanimous agreement in either direction must NOT raise.
        (
            [
                {"rank": 0, "error": None, "valid": True, "exists": True},
                {"rank": 1, "error": None, "valid": True, "exists": True},
            ],
            False,
        ),
        (
            [
                {"rank": 0, "error": None, "valid": False, "exists": False},
                {"rank": 1, "error": None, "valid": False, "exists": False},
            ],
            False,
        ),
    ],
)
def test_completed_cache_visibility_disagreement_raises_on_every_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[dict[str, Any]],
    expect_raise: bool,
) -> None:
    """The visibility guard is a pure function of the GATHERED status list.

    No multiprocess phase is needed to reach it -- substituting the gather is
    enough, and it exercises shapes a two-rank local run cannot produce (both
    ranks seeing the file, only one finding it valid). Nothing exercised this
    guard before: deleting it failed no test in the suite.

    The unanimous cases are the over-correction check: a guard that fires when
    ranks AGREE would break every ordinary build and resume.
    """
    _install_build_overrides(monkeypatch)
    positive, negative = _write_sources(tmp_path)
    contract = _contract(tmp_path)
    paths = _paths(tmp_path, run_tag="visibility-disagreement")

    real_gather = cache_module._all_gather_object

    def _substituted(value: dict[str, Any], *, world_size: int):
        if {"valid", "exists"} <= set(value):
            return statuses
        return real_gather(value, world_size=world_size)

    monkeypatch.setattr(cache_module, "_all_gather_object", _substituted)

    def _run() -> Path:
        return build_clustered_context_cache(
            cfg=_cfg(),
            output_paths=paths,
            prefix_population=_canonical_population(positive, negative),
            base_model=object(),
            base_sae=object(),
            cluster_contract=contract,
        )

    if expect_raise:
        with pytest.raises(AxBenchBpbAdvError, match="disagree about completed"):
            _run()
    else:
        _run()
