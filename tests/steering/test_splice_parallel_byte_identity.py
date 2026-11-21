"""Byte-identity of parallel (multi-worker) vs sequential splice preselection.

Issue #740 PR-D: `selection_num_workers > 1` processes packed shards across a
spawn process pool. A shard's selection output is a pure function of (shard data,
RNG seeded from shard_name, config) written to its own file, so the parallel run
must be BYTE-IDENTICAL to the single-process run. Following the parity-test
convention, both runs use the SAME build (run -> snapshot -> wipe sel/ -> rerun)
so any difference is attributable solely to parallelism, not to two independent
builds' manifest-fingerprint / absolute-path differences.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest
from omegaconf import OmegaConf

import steering.conditional.sampling as sampling
import steering.conditional.splice_selection_certificate as splice_certificate
from errors import FeatureClustersError
from scripts.preprocess import precompute_selected_features as psf
from steering.conditional.clustered_selection_validation import (
    parse_completed_splice_validation,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    parse_feature_delta,
)
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.stats_io import find_statistics_path, load_statistics
from tests.steering._splice_cf_cache_fixtures import (
    WHITELIST_MASK_BELOW_150,
    build_interleaved_env,
    build_multisplice_env,
    run_selection_main,
    write_cluster_yaml,
)

_TEST_CLUSTER_DESCRIPTOR = {
    "schema_version": 1,
    "semantics": "post_mask_post_delta_max_percentile_v1",
    "n_sae_features": 8,
    "content_sha256": "d" * 64,
}


class _SelectionCommonKwargs(TypedDict, total=False):
    """Precisely typed shared keyword subsets for ``run_selection_main``."""

    cache_shards: int
    sidecar_shards: int
    mask: dict
    clusters_path: str


def _resume_metadata_fixture(
    *, status: str = "complete"
) -> tuple[dict, dict, dict[str, psf.SpliceSelectionStats]]:
    """Small already-byte-certified domain for strict resume-schema tests."""
    stats = {
        "shard_00000": psf.SpliceSelectionStats(11, 3, 3),
        "shard_00001": psf.SpliceSelectionStats(13, 5, 5),
    }
    integrity = {
        "schema_version": 1,
        "shards": [
            {
                "name": f"{stem}.safetensors.zst",
                "byte_length": 100 + index,
                "content_sha256": str(index + 1) * 64,
            }
            for index, stem in enumerate(stats)
        ],
        "aggregate_sha256": "a" * 64,
    }
    storage_validated: dict[str, str] = {
        stem: str(integrity["shards"][index]["content_sha256"])
        for index, stem in enumerate(stats)
    }
    resume = psf._build_splice_resume_metadata(
        stats_by_stem=stats,
        selection_integrity=integrity,
        storage_validated_by_stem=storage_validated,
        validated_max_features_per_segment=9,
        packed_rows_hash="b" * 64,
        effective_buckets=7,
        status=status,
    )
    return resume, integrity, stats


def _rerun_on_same_build(env, *, first_kwargs, second_kwargs, monkeypatch):
    """Run selection twice on ONE build, wiping sel/ between runs."""
    first = run_selection_main(env, monkeypatch=monkeypatch, **first_kwargs)
    shutil.rmtree(env["packed_row_dir"] / "sel")
    second = run_selection_main(env, monkeypatch=monkeypatch, **second_kwargs)
    return first, second


class TestSpliceResumeMetadataSchema:
    @pytest.mark.parametrize(
        ("status", "expected_complete"),
        [("in_progress", False), ("complete", True)],
    )
    def test_exact_roundtrip(self, status, expected_complete):
        resume, integrity, expected_stats = _resume_metadata_fixture(status=status)
        (
            actual_stats,
            outputs_validated,
            storage_validated,
            validated_max_features,
        ) = psf._parse_splice_resume_metadata(
            resume,
            selection_integrity=integrity,
            packed_rows_hash="b" * 64,
            effective_buckets=7,
            manifest_status=status,
            surface="test manifest",
        )
        assert actual_stats == expected_stats
        assert outputs_validated is expected_complete
        assert storage_validated == {
            stem: integrity["shards"][index]["content_sha256"]
            for index, stem in enumerate(expected_stats)
        }
        assert validated_max_features == 9

    def test_completed_consumer_certificate_skips_redundant_body_scan(self):
        resume, integrity, stats = _resume_metadata_fixture()
        metadata = {
            psf.SPLICE_RESUME_METADATA_KEY: resume,
            SELECTION_SHARD_INTEGRITY_KEY: integrity,
            "feature_clusters": {
                "status": "complete",
                "descriptor": _TEST_CLUSTER_DESCRIPTOR,
            },
            "splice_counterfactual": {
                "schema_version": (
                    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION
                ),
                "status": "complete",
                CLUSTER_DELTA_SEMANTICS_KEY: (
                    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
                ),
                "total_packed_segments_selected": sum(
                    item.total_packed_segments for item in stats.values()
                ),
                "total_sidecar_splice_positions_referenced": sum(
                    item.total_sidecar_splice_positions_referenced
                    for item in stats.values()
                ),
                "total_delta_positions_selected": sum(
                    item.total_delta_positions_selected for item in stats.values()
                ),
            },
        }

        completed = parse_completed_splice_validation(
            metadata,
            selection_integrity=integrity,
            packed_row_fingerprint="b" * 64,
            effective_buckets=7,
            feature_clusters_descriptor=_TEST_CLUSTER_DESCRIPTOR,
            expected_stems=set(stats),
            surface="test manifest",
        )
        assert completed.validated_max_features_per_segment == 9

    def test_consumer_certificate_rejects_summary_count_drift(self):
        resume, integrity, stats = _resume_metadata_fixture()
        metadata = {
            psf.SPLICE_RESUME_METADATA_KEY: resume,
            SELECTION_SHARD_INTEGRITY_KEY: integrity,
            "feature_clusters": {
                "status": "complete",
                "descriptor": _TEST_CLUSTER_DESCRIPTOR,
            },
            "splice_counterfactual": {
                "schema_version": (
                    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION
                ),
                "status": "complete",
                CLUSTER_DELTA_SEMANTICS_KEY: (
                    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
                ),
                "total_packed_segments_selected": 25,
                "total_sidecar_splice_positions_referenced": 8,
                "total_delta_positions_selected": 8,
            },
        }

        with pytest.raises(
            FeatureClustersError,
            match="total_packed_segments_selected.*shard total 24",
        ):
            parse_completed_splice_validation(
                metadata,
                selection_integrity=integrity,
                packed_row_fingerprint="b" * 64,
                effective_buckets=7,
                feature_clusters_descriptor=_TEST_CLUSTER_DESCRIPTOR,
                expected_stems=set(stats),
                surface="test manifest",
            )

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ("extra_top_key", "must contain exactly"),
            ("boolean_schema", "schema_version must be strict integer"),
            ("stale_cluster_delta_semantics", "current canonical"),
            ("empty_shards", "shards must be a non-empty list"),
            ("reverse_shards", "unique, strictly sorted"),
            ("path_shard_name", "unique, strictly sorted"),
            ("malformed_content_digest", "digest .* malformed"),
            ("wrong_content_digest", "not bound to the certified"),
            ("malformed_storage_digest", "storage validation .* must equal"),
            ("wrong_storage_digest", "storage validation .* must equal"),
            ("missing_complete_storage_validation", "unvalidated serialized"),
            ("boolean_stat", "non-negative strict integer"),
            ("negative_stat", "non-negative strict integer"),
            ("missing_shard", "names differ"),
            ("extra_output_key", "output_validation keys are malformed"),
            ("status_mismatch", "does not match manifest"),
            ("selection_aggregate_mismatch", "not bound to the current selection"),
            ("packed_fingerprint_mismatch", "stale for current packed rows"),
            ("boolean_buckets", "effective_buckets"),
            ("boolean_validated_max", "non-negative strict integer"),
            ("negative_validated_max", "non-negative strict integer"),
            ("missing_validated_max", "does not match its post-serialization"),
            ("signed_stat_tamper", "aggregate mismatch"),
            ("aggregate_mismatch", "aggregate mismatch"),
        ],
    )
    def test_every_corruption_class_fails_loud(self, mutation, match):
        resume, integrity, _stats = _resume_metadata_fixture()
        raw = copy.deepcopy(resume)
        if mutation == "extra_top_key":
            raw["extra"] = None
        elif mutation == "boolean_schema":
            raw["schema_version"] = True
        elif mutation == "stale_cluster_delta_semantics":
            raw[psf.CLUSTER_DELTA_SEMANTICS_KEY] = "post_mask_post_delta_max_percentile_v1"
        elif mutation == "empty_shards":
            raw["shards"] = []
        elif mutation == "reverse_shards":
            raw["shards"].reverse()
        elif mutation == "path_shard_name":
            raw["shards"][0]["name"] = "../shard_00000.safetensors.zst"
        elif mutation == "malformed_content_digest":
            raw["shards"][0]["selection_content_sha256"] = "A" * 64
        elif mutation == "wrong_content_digest":
            raw["shards"][0]["selection_content_sha256"] = "f" * 64
        elif mutation == "malformed_storage_digest":
            raw["shards"][0]["storage_validation_content_sha256"] = "A" * 64
        elif mutation == "wrong_storage_digest":
            raw["shards"][0]["storage_validation_content_sha256"] = "f" * 64
        elif mutation == "missing_complete_storage_validation":
            raw["shards"][0]["storage_validation_content_sha256"] = None
        elif mutation == "boolean_stat":
            raw["shards"][0]["total_packed_segments"] = True
        elif mutation == "negative_stat":
            raw["shards"][0]["total_delta_positions_selected"] = -1
        elif mutation == "missing_shard":
            raw["shards"].pop()
        elif mutation == "extra_output_key":
            raw["output_validation"]["extra"] = None
        elif mutation == "status_mismatch":
            raw["output_validation"]["status"] = "in_progress"
        elif mutation == "selection_aggregate_mismatch":
            raw["output_validation"]["selection_aggregate_sha256"] = "c" * 64
        elif mutation == "packed_fingerprint_mismatch":
            raw["output_validation"]["packed_row_fingerprint"] = "c" * 64
        elif mutation == "boolean_buckets":
            raw["output_validation"]["effective_buckets"] = True
        elif mutation == "boolean_validated_max":
            raw["output_validation"]["validated_max_features_per_segment"] = True
        elif mutation == "negative_validated_max":
            raw["output_validation"]["validated_max_features_per_segment"] = -1
        elif mutation == "missing_validated_max":
            raw["output_validation"]["validated_max_features_per_segment"] = None
        elif mutation == "signed_stat_tamper":
            raw["shards"][0]["total_packed_segments"] += 1
        elif mutation == "aggregate_mismatch":
            raw["aggregate_sha256"] = "0" * 64
        else:
            raise AssertionError(f"unhandled mutation {mutation}")

        with pytest.raises(FeatureClustersError, match=match):
            psf._parse_splice_resume_metadata(
                raw,
                selection_integrity=integrity,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                manifest_status="complete",
                surface="test manifest",
            )

    def test_builder_rejects_stats_domain_different_from_integrity(self):
        _resume, integrity, stats = _resume_metadata_fixture()
        stats = dict(stats)
        stats["shard_extra"] = psf.SpliceSelectionStats()
        with pytest.raises(FeatureClustersError, match="shard names differ"):
            psf._build_splice_resume_metadata(
                stats_by_stem=stats,
                selection_integrity=integrity,
                storage_validated_by_stem={
                    stem: integrity["shards"][index]["content_sha256"]
                    for index, stem in enumerate(stats)
                    if stem != "shard_extra"
                },
                validated_max_features_per_segment=9,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                status="complete",
            )

    def test_builder_rejects_complete_certificate_with_unvalidated_shard(self):
        _resume, integrity, stats = _resume_metadata_fixture()
        first_stem = next(iter(stats))
        with pytest.raises(FeatureClustersError, match="before every serialized"):
            psf._build_splice_resume_metadata(
                stats_by_stem=stats,
                selection_integrity=integrity,
                storage_validated_by_stem={
                    first_stem: integrity["shards"][0]["content_sha256"]
                },
                validated_max_features_per_segment=9,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                status="complete",
            )

    def test_in_progress_v3_certificate_tracks_partial_storage_validation(self):
        _resume, integrity, stats = _resume_metadata_fixture()
        first_stem = next(iter(stats))
        first_digest = integrity["shards"][0]["content_sha256"]
        resume = psf._build_splice_resume_metadata(
            stats_by_stem=stats,
            selection_integrity=integrity,
            storage_validated_by_stem={first_stem: first_digest},
            validated_max_features_per_segment=5,
            packed_rows_hash="b" * 64,
            effective_buckets=7,
            status="in_progress",
        )
        parsed_stats, outputs_validated, storage_validated, validated_max = (
            psf._parse_splice_resume_metadata(
                resume,
                selection_integrity=integrity,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                manifest_status="in_progress",
                surface="partial test manifest",
            )
        )
        assert parsed_stats == stats
        assert outputs_validated is False
        assert storage_validated == {first_stem: first_digest}
        assert validated_max == 5
        assert [
            entry["storage_validation_content_sha256"]
            for entry in resume["shards"]
        ] == [first_digest, None]

    def test_builder_rejects_storage_validation_outside_integrity_domain(self):
        _resume, integrity, stats = _resume_metadata_fixture()
        with pytest.raises(FeatureClustersError, match="outside the selection integrity"):
            psf._build_splice_resume_metadata(
                stats_by_stem=stats,
                selection_integrity=integrity,
                storage_validated_by_stem={"not_a_shard": "f" * 64},
                validated_max_features_per_segment=5,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                status="in_progress",
            )

    @pytest.mark.parametrize("legacy_schema", [1, 2])
    def test_pre_cluster_delta_semantic_certificate_is_rejected(
        self,
        legacy_schema,
    ):
        resume, integrity, _stats = _resume_metadata_fixture()
        legacy = copy.deepcopy(resume)
        legacy["schema_version"] = legacy_schema
        signed = {
            "schema_version": legacy["schema_version"],
            psf.CLUSTER_DELTA_SEMANTICS_KEY: legacy[
                psf.CLUSTER_DELTA_SEMANTICS_KEY
            ],
            "shards": legacy["shards"],
            "output_validation": legacy["output_validation"],
        }
        legacy["aggregate_sha256"] = splice_certificate._canonical_json_sha256(
            signed
        )

        with pytest.raises(
            FeatureClustersError,
            match="schema_version must be strict integer 3",
        ):
            psf._parse_splice_resume_metadata(
                legacy,
                selection_integrity=integrity,
                packed_rows_hash="b" * 64,
                effective_buckets=7,
                manifest_status="complete",
                surface="legacy test manifest",
            )


@pytest.mark.parametrize("failure", ["aggregate", "domain"])
def test_clustered_producer_rejects_startup_packed_proof_disagreement(
    tmp_path,
    monkeypatch,
    failure,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    real_resolve = psf.resolve_packed_row_fingerprint_strict

    def forged_resolution(*args, **kwargs):
        resolved = real_resolve(*args, **kwargs)
        if failure == "aggregate":
            return replace(resolved, aggregate_sha256="0" * 64)
        return replace(resolved, files=resolved.files[:-1])

    monkeypatch.setattr(
        psf,
        "resolve_packed_row_fingerprint_strict",
        forged_resolution,
    )
    with pytest.raises(
        FeatureClustersError,
        match=(
            "disagrees with the selection manifest identity"
            if failure == "aggregate"
            else "does not cover the exact packed shard domain"
        ),
    ):
        run_selection_main(
            env,
            num_workers=1,
            cache_shards=2,
            sidecar_shards=4,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=write_cluster_yaml(tmp_path),
            monkeypatch=monkeypatch,
        )


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_parallel_matches_sequential_with_clusters(tmp_path, monkeypatch, workers):
    """Production-like path (mask + clusters): N workers == 1 worker, byte-for-byte."""
    env = build_interleaved_env(tmp_path)
    clusters = write_cluster_yaml(tmp_path)
    common: _SelectionCommonKwargs = {
        "cache_shards": 5,
        "sidecar_shards": 16,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": clusters,
    }

    def forbidden_final_replay(*_args, **_kwargs):
        raise AssertionError("fresh splice run performed final packed-body replay")

    monkeypatch.setattr(
        psf,
        "_validate_clustered_selection_outputs",
        forbidden_final_replay,
    )
    seq, par = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={**common, "num_workers": workers},
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(par), f"file set differs (workers={workers})"
    for name in sorted(seq):
        assert seq[name] == par[name], (
            f"{name} differs between sequential and {workers}-worker parallel"
        )


def test_sparse_inactive_sampler_matches_dense_full_pipeline_bytes(
    tmp_path,
    monkeypatch,
):
    """Every selection shard and manifest matches the retained dense path."""
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    sparse = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    monkeypatch.setattr(sampling, "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT", -1)
    dense = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    assert sparse.keys() == dense.keys()
    for name in sparse:
        assert sparse[name] == dense[name], name


def test_post_serialization_validation_is_race_bound_to_hashed_bytes(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    real_extend = psf.extend_clustered_selection_shard_integrity

    def mutate_after_hash(prior, shard_path, *, surface):
        integrity = real_extend(prior, shard_path, surface=surface)
        shard_path.write_bytes(shard_path.read_bytes() + b"post-hash-mutation")
        return integrity

    monkeypatch.setattr(
        psf,
        "extend_clustered_selection_shard_integrity",
        mutate_after_hash,
    )
    with pytest.raises(
        FeatureClustersError,
        match="changed between validation and fingerprinting",
    ):
        run_selection_main(
            env,
            num_workers=1,
            cache_shards=3,
            sidecar_shards=8,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=write_cluster_yaml(tmp_path),
            monkeypatch=monkeypatch,
        )


def test_corrupt_reloaded_selection_fails_before_progress_certification(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    real_load = psf.load_selection_shard

    def load_with_corrupt_offsets(path):
        shard = real_load(path)
        shard.sel_offsets = shard.sel_offsets[:-1]
        return shard

    monkeypatch.setattr(psf, "load_selection_shard", load_with_corrupt_offsets)
    with pytest.raises(FeatureClustersError, match="offset"):
        run_selection_main(
            env,
            num_workers=1,
            cache_shards=3,
            sidecar_shards=8,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=write_cluster_yaml(tmp_path),
            monkeypatch=monkeypatch,
        )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = json.loads((sel_dir / "manifest.json").read_text(encoding="utf-8"))
    assert SELECTION_SHARD_INTEGRITY_KEY not in manifest["metadata"]
    assert psf.SPLICE_RESUME_METADATA_KEY not in manifest["metadata"]


def test_clustered_splice_loader_resume_state_binds_delta_semantics(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    cluster_path = write_cluster_yaml(tmp_path)
    run_selection_main(
        env,
        num_workers=1,
        cache_shards=3,
        sidecar_shards=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )

    feature_dir = Path(env["feature_dir"])
    stats_path = find_statistics_path(feature_dir)
    assert stats_path is not None
    statistics = load_statistics(stats_path)
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(150)))
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": cluster_path}},
        statistics.metadata.n_sae_features,
    )
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "min_k": 0,
                "max_k": 49,
            },
        }
    )
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=statistics.per_feature_stats,
        n_sae_features=statistics.metadata.n_sae_features,
        num_buckets=2,
        seed=42,
        feature_delta=parse_feature_delta(
            {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
        ),
        feature_mask=mask,
    )
    selection_dir = next((env["packed_row_dir"] / "sel").iterdir())

    def loader(resume_state_dict=None):
        return precomputed_data_loader(
            packed_row_dir=env["packed_row_dir"],
            B=1,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=selection_dir,
            effective_buckets=pipeline.effective_buckets,
            n_sae_features=statistics.metadata.n_sae_features,
            splice_feature_dir=feature_dir,
            resume_state_dict=resume_state_dict,
            feature_clusters=clusters,
        )

    first_state = next(loader())[-1]
    assert first_state[CLUSTER_DELTA_SEMANTICS_KEY] == (CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS)
    resumed_state = next(loader(first_state))[-1]
    assert resumed_state[CLUSTER_DELTA_SEMANTICS_KEY] == (CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS)

    stale_state = dict(first_state)
    stale_state[CLUSTER_DELTA_SEMANTICS_KEY] = "post_mask_post_delta_max_percentile_v1"
    with pytest.raises(ValueError, match="cluster-delta semantics mismatch"):
        next(loader(stale_state))


def test_parallel_matches_sequential_no_clusters(tmp_path, monkeypatch):
    """Mask-only (no clusters) path is also byte-identical under parallelism."""
    env = build_interleaved_env(tmp_path)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    seq, par = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={**common, "num_workers": 3},
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(par)
    for name in sorted(seq):
        assert seq[name] == par[name], f"{name} differs (no-clusters)"


def test_parallel_matches_sequential_multisplice(tmp_path, monkeypatch):
    """The multi-splice-per-doc corpus (heaviest per-shard work) stays identical."""
    env = build_multisplice_env(tmp_path)
    clusters = write_cluster_yaml(tmp_path)
    common: _SelectionCommonKwargs = {
        "cache_shards": 5,
        "sidecar_shards": 16,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": clusters,
    }
    seq, par = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={**common, "num_workers": 4},
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(par)
    for name in sorted(seq):
        assert seq[name] == par[name], f"{name} differs (multisplice)"


@pytest.mark.parametrize("workers", [2, 4])
def test_shared_store_matches_no_store_with_clusters(tmp_path, monkeypatch, workers):
    """PR-F: N workers WITH the shared /dev/shm store == N workers WITHOUT it, byte-for-byte —
    the store is a read-only cache (one physical copy per stem) in front of the same loader."""
    env = build_interleaved_env(tmp_path)
    clusters = write_cluster_yaml(tmp_path)
    shm = tmp_path / "shm"
    shm.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 5, "sidecar_shards": 16, "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": clusters,
    }
    nostore, store = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": workers},
        second_kwargs={
            **common, "num_workers": workers, "shared_store": True, "shm_dir": str(shm),
        },
        monkeypatch=monkeypatch,
    )
    assert set(nostore) == set(store), f"file set differs (workers={workers})"
    for name in sorted(nostore):
        assert nostore[name] == store[name], (
            f"{name} differs: shared store vs none (workers={workers})"
        )
    # the /dev/shm dir is cleaned up on exit
    assert not any(shm.iterdir()), "shared store dir left behind"


def test_shared_store_matches_sequential_no_clusters(tmp_path, monkeypatch):
    """Shared store (mask-only, no clusters) == 1-worker sequential, byte-for-byte."""
    env = build_interleaved_env(tmp_path)
    shm = tmp_path / "shm"
    shm.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    seq, store = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={
            **common, "num_workers": 3, "shared_store": True, "shm_dir": str(shm),
        },
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(store)
    for name in sorted(seq):
        assert seq[name] == store[name], f"{name} differs (shared store vs sequential)"


def test_disk_spooled_sequential_matches_eager_byte_for_byte(tmp_path, monkeypatch):
    """The disk-backed packed reader is an exact resource-mode substitution."""
    env = build_interleaved_env(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 5,
        "sidecar_shards": 16,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    eager, spooled = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={
            **common,
            "num_workers": 1,
            "packed_spool_dir": str(spool),
        },
        monkeypatch=monkeypatch,
    )

    assert eager == spooled
    assert list(spool.iterdir()) == []


@pytest.mark.parametrize("workers", [2, 4])
def test_spool_compact_sidecars_and_shared_store_match_eager_reference(
    tmp_path,
    monkeypatch,
    workers,
):
    """Exercise all issue-758 memory modes together against the W=1 oracle."""
    env = build_interleaved_env(tmp_path)
    spool = tmp_path / "spool"
    shm = tmp_path / "shm"
    spool.mkdir()
    shm.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 5,
        "sidecar_shards": 16,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    eager, bounded = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={
            **common,
            "num_workers": workers,
            "shared_store": True,
            "shared_store_capacity": 2,
            "shm_dir": str(shm),
            "packed_spool_dir": str(spool),
        },
        monkeypatch=monkeypatch,
    )

    assert eager == bounded
    assert list(spool.iterdir()) == []
    assert list(shm.iterdir()) == []


def test_spooled_parallel_resume_matches_complete_eager_reference(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    complete = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shard_files = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    for path in shard_files[::2]:
        path.unlink()

    resumed = run_selection_main(
        env,
        num_workers=3,
        packed_spool_dir=str(spool),
        monkeypatch=monkeypatch,
        **common,
    )

    assert complete == resumed
    assert list(spool.iterdir()) == []


@pytest.mark.parametrize("bad", ["", "   ", "/definitely/absent/spool"])
def test_invalid_spool_directory_fails_before_output_creation(
    tmp_path,
    monkeypatch,
    bad,
):
    env = build_interleaved_env(tmp_path, n_shards=2)

    with pytest.raises(ValueError, match="selection_packed_spool_dir"):
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=4,
            num_workers=2,
            packed_spool_dir=bad,
            mask=WHITELIST_MASK_BELOW_150,
            monkeypatch=monkeypatch,
        )

    assert not (env["packed_row_dir"] / "sel").exists()


def test_shared_store_tiny_capacity_still_identical(tmp_path, monkeypatch):
    """A shared-store capacity below the working set (constant eviction/churn) stays byte-
    identical — the eviction race + private re-materialize fallback are correctness-preserving."""
    env = build_interleaved_env(tmp_path)
    clusters = write_cluster_yaml(tmp_path)
    shm = tmp_path / "shm"
    shm.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 5, "sidecar_shards": 16, "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": clusters,
    }
    seq, store = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={
            **common, "num_workers": 4, "shared_store": True,
            "shared_store_capacity": 1, "shm_dir": str(shm),
        },
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(store)
    for name in sorted(seq):
        assert seq[name] == store[name], f"{name} differs (tiny-capacity shared store)"


def test_more_workers_than_shards_is_safe(tmp_path, monkeypatch):
    """num_workers far exceeding the shard count must not deadlock or change output."""
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 2,
        "sidecar_shards": 4,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    seq, par = _rerun_on_same_build(
        env,
        first_kwargs={**common, "num_workers": 1},
        second_kwargs={**common, "num_workers": 16},
        monkeypatch=monkeypatch,
    )
    assert set(seq) == set(par)
    for name in sorted(seq):
        assert seq[name] == par[name]


def test_invalid_num_workers_raises(tmp_path, monkeypatch):
    env = build_interleaved_env(tmp_path, n_shards=2)
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="selection_num_workers"):
            run_selection_main(
                env,
                cache_shards=2,
                sidecar_shards=4,
                num_workers=bad,
                mask=WHITELIST_MASK_BELOW_150,
                monkeypatch=monkeypatch,
            )
        shutil.rmtree(env["packed_row_dir"] / "sel", ignore_errors=True)


@pytest.mark.parametrize("bad", [0, 1, None, "false", "true"])
def test_invalid_shared_store_boolean_fails_before_output_creation(
    tmp_path,
    monkeypatch,
    bad,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    with pytest.raises(ValueError, match="selection_shared_store must be a bool"):
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=4,
            num_workers=2,
            shared_store=bad,
            mask=WHITELIST_MASK_BELOW_150,
            monkeypatch=monkeypatch,
        )
    assert not (env["packed_row_dir"] / "sel").exists()


@pytest.mark.parametrize("bad", [0, 1, None, "false", "true", [], {}])
def test_invalid_selection_resume_boolean_fails_before_output_creation(
    tmp_path,
    monkeypatch,
    bad,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    with pytest.raises(ValueError, match="selection_resume must be a bool"):
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=4,
            num_workers=2,
            selection_resume=bad,
            mask=WHITELIST_MASK_BELOW_150,
            monkeypatch=monkeypatch,
        )
    assert not (env["packed_row_dir"] / "sel").exists()


def test_rejected_num_workers_preserves_existing_complete_manifest(tmp_path, monkeypatch):
    """selection_num_workers is validated BEFORE any output-dir/manifest write, so
    a rejected value must leave an existing COMPLETE sel/ manifest byte-unchanged
    (regression for validating after the in-progress manifest was already written)."""
    env = build_interleaved_env(tmp_path)
    common: _SelectionCommonKwargs = {
        "cache_shards": 2,
        "sidecar_shards": 4,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    run_selection_main(env, num_workers=1, monkeypatch=monkeypatch, **common)
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = sel_dir / "manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="selection_num_workers"):
        run_selection_main(env, num_workers=0, monkeypatch=monkeypatch, **common)
    assert manifest.read_bytes() == before, "rejected value mutated the complete manifest"


def test_parallel_admission_runs_before_output_or_shared_store_creation(
    tmp_path,
    monkeypatch,
):
    """The fail-closed boundary precedes every output/store mutation."""
    env = build_interleaved_env(tmp_path, n_shards=2)
    spool = tmp_path / "spool"
    shm = tmp_path / "shm"
    spool.mkdir()
    shm.mkdir()
    captured = {}

    def reject_before_side_effects(**kwargs):
        captured.update(kwargs)
        assert not (env["packed_row_dir"] / "sel").exists()
        assert list(spool.iterdir()) == []
        assert list(shm.iterdir()) == []
        raise RuntimeError("admission-order-sentinel")

    monkeypatch.setattr(psf, "admit_splice_resources", reject_before_side_effects)
    with pytest.raises(RuntimeError, match="admission-order-sentinel"):
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=4,
            num_workers=2,
            shared_store=True,
            shared_store_capacity=None,
            shm_dir=str(shm),
            packed_spool_dir=str(spool),
            mask=WHITELIST_MASK_BELOW_150,
            monkeypatch=monkeypatch,
        )

    assert captured["num_workers"] == 2
    assert captured["has_selection_work"] is True
    assert captured["shared_store_enabled"] is True
    assert captured["shared_store_capacity"] == 6  # auto = workers + 4
    assert captured["shared_store_root"] == str(shm)
    assert captured["packed_spool_dir"] == str(spool.absolute())
    assert not (env["packed_row_dir"] / "sel").exists()
    assert list(spool.iterdir()) == []
    assert list(shm.iterdir()) == []


def test_single_worker_preserves_legacy_path_without_admission(tmp_path, monkeypatch):
    env = build_interleaved_env(tmp_path, n_shards=2)

    def forbidden_admission(**_kwargs):
        raise AssertionError("W=1 must not invoke parallel admission")

    monkeypatch.setattr(psf, "admit_splice_resources", forbidden_admission)
    result = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=4,
        num_workers=1,
        mask=WHITELIST_MASK_BELOW_150,
        monkeypatch=monkeypatch,
    )
    assert "manifest.json" in result
    assert any(name.endswith(".safetensors.zst") for name in result)


def test_rejected_parallel_resume_preserves_every_complete_output_byte(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    common: _SelectionCommonKwargs = {
        "cache_shards": 2,
        "sidecar_shards": 4,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    complete = run_selection_main(
        env, num_workers=1, monkeypatch=monkeypatch, **common
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    before = {path.name: path.read_bytes() for path in sorted(sel_dir.iterdir())}
    captured = {}

    def reject_resume(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("resume-admission-sentinel")

    monkeypatch.setattr(psf, "admit_splice_resources", reject_resume)
    with pytest.raises(RuntimeError, match="resume-admission-sentinel"):
        run_selection_main(
            env, num_workers=2, monkeypatch=monkeypatch, **common
        )

    after = {path.name: path.read_bytes() for path in sorted(sel_dir.iterdir())}
    assert captured["has_selection_work"] is False
    assert after == before == complete


def test_parallel_resume_reprocesses_missing_only(tmp_path, monkeypatch):
    """Parallel resume: workers skip existing shards (stat-only) and reprocess only
    the missing ones; the final output equals a from-scratch run."""
    env = build_interleaved_env(tmp_path)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    full = run_selection_main(env, num_workers=1, monkeypatch=monkeypatch, **common)
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shard_files = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shard_files) >= 3, "need multiple shards to exercise mixed resume"
    for p in shard_files[::2]:  # delete every other shard -> partial
        p.unlink()
    resumed = run_selection_main(env, num_workers=3, monkeypatch=monkeypatch, **common)
    assert set(resumed) == set(full)
    for name in sorted(full):
        assert resumed[name] == full[name], f"{name} differs after parallel resume"


def test_selection_resume_false_recomputes_and_repairs_unclustered_shards(
    tmp_path,
    monkeypatch,
):
    """The explicit opt-out means full recompute on the legacy unclustered path too."""
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
    }
    expected = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shard_paths = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert shard_paths
    shard_paths[-1].write_bytes(b"corrupt-unclustered-selection-shard")

    real_select = psf.select_splice_counterfactual_features_for_shard
    calls: list[str] = []

    def recording_select(*args, **kwargs):
        calls.append(args[3])
        return real_select(*args, **kwargs)

    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        recording_select,
    )
    repaired = run_selection_main(
        env,
        num_workers=1,
        selection_resume=False,
        monkeypatch=monkeypatch,
        **common,
    )

    assert calls == [path.name.removesuffix(".safetensors.zst") for path in shard_paths]
    assert repaired == expected


def test_selection_resume_false_repairs_corrupt_clustered_certificate(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=2)
    common: _SelectionCommonKwargs = {
        "cache_shards": 2,
        "sidecar_shards": 4,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    expected = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"][psf.CLUSTERED_SELECTION_VALIDATION_KEY][
        "aggregate_sha256"
    ] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    repaired = run_selection_main(
        env,
        num_workers=1,
        selection_resume=False,
        monkeypatch=monkeypatch,
        **common,
    )

    assert repaired == expected


def test_parallel_clustered_resume_reuses_only_byte_certified_shards(
    tmp_path,
    monkeypatch,
):
    """An in-progress clustered run resumes its signed prefix exactly.

    A file present after the last manifest checkpoint models the crash window:
    it is not trusted merely because it looks like a shard and is overwritten.
    """
    env = build_interleaved_env(tmp_path, n_shards=5)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    complete = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    shard_paths = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shard_paths) >= 2
    certified = shard_paths[:1]
    uncommitted = shard_paths[1]
    certified_inodes = {path.name: path.stat().st_ino for path in certified}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
    manifest["metadata"]["splice_counterfactual"]["status"] = "in_progress"
    # Model an interruption before the producer's first current-semantic
    # resume certificate. A certificate bound to the former complete
    # integrity domain would be malformed after truncation, so remove it.
    manifest["metadata"].pop(psf.SPLICE_RESUME_METADATA_KEY, None)
    manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY] = (
        build_clustered_selection_shard_integrity(certified)
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    uncommitted.write_bytes(b"whole-but-uncertified-crash-window-output")

    resumed = run_selection_main(
        env,
        num_workers=3,
        monkeypatch=monkeypatch,
        **common,
    )

    assert resumed == complete
    for path in certified:
        assert path.stat().st_ino == certified_inodes[path.name]


def test_interrupted_clustered_producer_checkpoints_each_written_shard(
    tmp_path,
    monkeypatch,
):
    """A real mid-loop failure leaves a useful, exact restart boundary."""
    env = build_interleaved_env(tmp_path, n_shards=4)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    complete = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")

    real_select = psf.select_splice_counterfactual_features_for_shard
    calls: list[str] = []

    def fail_before_second(*args, **kwargs):
        shard_name = args[3]
        calls.append(shard_name)
        if len(calls) == 2:
            raise RuntimeError("injected interruption after first atomic shard")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        fail_before_second,
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_selection_main(
            env,
            num_workers=1,
            monkeypatch=monkeypatch,
            **common,
        )

    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = json.loads((sel_dir / "manifest.json").read_text(encoding="utf-8"))
    progress = manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY]
    resume_progress = manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    assert manifest["metadata"]["feature_clusters"]["status"] == "in_progress"
    assert [entry["name"] for entry in progress["shards"]] == [
        "shard_00000.safetensors.zst"
    ]
    certified_path = sel_dir / progress["shards"][0]["name"]
    certified_inode = certified_path.stat().st_ino
    assert resume_progress["output_validation"]["status"] == "in_progress"
    assert [entry["name"] for entry in resume_progress["shards"]] == [
        certified_path.name
    ]
    assert (
        resume_progress["shards"][0]["storage_validation_content_sha256"]
        == progress["shards"][0]["content_sha256"]
    )

    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        real_select,
    )

    def forbidden_stats_replay(*_args, **_kwargs):
        raise AssertionError("certified partial resume replayed existing shard stats")

    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        forbidden_stats_replay,
    )
    resumed = run_selection_main(
        env,
        num_workers=3,
        monkeypatch=monkeypatch,
        **common,
    )

    assert resumed == complete
    assert certified_path.stat().st_ino == certified_inode


def test_complete_clustered_resume_skips_all_shard_body_work(
    tmp_path,
    monkeypatch,
):
    """A complete v2 certificate skips tensor/stat replay after byte integrity."""
    env = build_interleaved_env(tmp_path, n_shards=4)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    first = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = json.loads((sel_dir / "manifest.json").read_text(encoding="utf-8"))
    resume = manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    assert resume["output_validation"]["status"] == "complete"
    assert sum(entry["total_packed_segments"] for entry in resume["shards"]) == (
        manifest["metadata"]["splice_counterfactual"][
            "total_packed_segments_selected"
        ]
    )

    real_validate = psf.validate_packed_provenance
    real_sidecar_fingerprint = psf.splice_sidecar_fingerprint
    real_derived_fingerprint = psf.derived_feature_fingerprint
    cache_wiring_calls = {"validation": 0, "sidecar": 0, "derived": 0}

    def recording_validate(path, *, cache_dir=None):
        cache_wiring_calls["validation"] += 1
        assert cache_dir == env["packed_row_dir"]
        return real_validate(path, cache_dir=cache_dir)

    def recording_sidecar(path, *, cache_dir=None):
        cache_wiring_calls["sidecar"] += 1
        assert cache_dir == env["packed_row_dir"]
        return real_sidecar_fingerprint(path, cache_dir=cache_dir)

    def recording_derived(path, *, expected_stems=None, cache_dir=None):
        cache_wiring_calls["derived"] += 1
        assert cache_dir == env["packed_row_dir"]
        return real_derived_fingerprint(
            path,
            expected_stems=expected_stems,
            cache_dir=cache_dir,
        )

    monkeypatch.setattr(psf, "validate_packed_provenance", recording_validate)
    monkeypatch.setattr(psf, "splice_sidecar_fingerprint", recording_sidecar)
    monkeypatch.setattr(psf, "derived_feature_fingerprint", recording_derived)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete certified resume touched a shard body")

    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        forbidden,
    )
    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        forbidden,
    )
    monkeypatch.setattr(psf, "_validate_clustered_selection_outputs", forbidden)
    monkeypatch.setattr(psf, "_run_splice_shards_parallel", forbidden)
    second = run_selection_main(
        env,
        num_workers=3,
        monkeypatch=monkeypatch,
        **common,
    )
    assert cache_wiring_calls == {"validation": 1, "sidecar": 1, "derived": 1}
    assert second == first


def test_clustered_splice_finalization_rejects_corruption_before_complete_manifest(
    tmp_path, monkeypatch
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=2)
    real_build = psf._build_completed_clustered_selection_validation

    def corrupt_before_validation(**kwargs):
        path = next(kwargs["output_dir"].glob("*.safetensors.zst"))
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        return real_build(**kwargs)

    monkeypatch.setattr(
        psf, "_build_completed_clustered_selection_validation", corrupt_before_validation
    )
    with pytest.raises(FeatureClustersError, match="rebuilt a selection inventory"):
        run_selection_main(
            env,
            num_workers=1,
            monkeypatch=monkeypatch,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=write_cluster_yaml(tmp_path),
        )

    selection_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = json.loads((selection_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["feature_clusters"]["status"] == "in_progress"
    assert psf.CLUSTERED_SELECTION_VALIDATION_KEY not in manifest["metadata"]


def test_splice_resume_and_remote_loader_certify_the_exact_same_global_width(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    run_selection_main(
        env,
        num_workers=1,
        cache_shards=3,
        sidecar_shards=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=write_cluster_yaml(tmp_path),
        monkeypatch=monkeypatch,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest = json.loads((sel_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    resume_max = metadata[psf.SPLICE_RESUME_METADATA_KEY]["output_validation"][
        "validated_max_features_per_segment"
    ]
    remote_max = metadata[psf.REMOTE_LOADER_KEY]["max_features_per_segment"]
    actual_max = max(
        psf._selection_max_features_per_segment(psf.load_selection_shard(path))
        for path in sorted(sel_dir.glob("shard_*.safetensors.zst"))
    )
    assert resume_max == remote_max == actual_max


def test_complete_resume_rejects_signed_remote_loader_width_disagreement(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(env, num_workers=1, monkeypatch=monkeypatch, **common)
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    remote = manifest["metadata"][psf.REMOTE_LOADER_KEY]
    manifest["metadata"][psf.REMOTE_LOADER_KEY] = (
        psf.build_remote_loader_certificate(
            selection_checksum=remote["selection_checksum"],
            selection_inventory_aggregate=remote[
                "selection_inventory_aggregate"
            ],
            packed_row_fingerprint=remote["packed_row_fingerprint"],
            max_features_per_segment=remote["max_features_per_segment"] + 1,
            cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
        )
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    before = manifest_path.read_bytes()

    with pytest.raises(
        FeatureClustersError,
        match="certificates disagree on max_features_per_segment",
    ):
        run_selection_main(env, num_workers=1, monkeypatch=monkeypatch, **common)
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("legacy_schema", [1, 2])
def test_pre_semantic_resume_certificate_fails_before_body_or_mutation(
    tmp_path,
    monkeypatch,
    legacy_schema,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resume = manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    resume["schema_version"] = legacy_schema
    resume.pop(psf.CLUSTER_DELTA_SEMANTICS_KEY)
    if legacy_schema == 1:
        for entry in resume["shards"]:
            entry.pop("storage_validation_content_sha256")
        resume["output_validation"].pop("validated_max_features_per_segment")
    resume["aggregate_sha256"] = splice_certificate._canonical_json_sha256(
        {
            "schema_version": resume["schema_version"],
            "shards": resume["shards"],
            "output_validation": resume["output_validation"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    before = manifest_path.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("stale semantic certificate opened a shard body")

    monkeypatch.setattr(psf, "load_selection_shard", forbidden)
    monkeypatch.setattr(psf, "load_packed_row_shard", forbidden)
    with pytest.raises(FeatureClustersError, match="must contain exactly"):
        run_selection_main(
            env,
            num_workers=1,
            monkeypatch=monkeypatch,
            **common,
        )
    assert manifest_path.read_bytes() == before


def test_complete_v3_without_remote_loader_reuses_signed_width_without_body_reads(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    expected = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resume = manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    certified_width = resume["output_validation"][
        "validated_max_features_per_segment"
    ]
    manifest["metadata"].pop(psf.REMOTE_LOADER_KEY)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    def forbidden_body_replay(*_args, **_kwargs):
        raise AssertionError("v3 signed-width recovery opened a shard body")

    monkeypatch.setattr(psf, "load_selection_shard", forbidden_body_replay)
    monkeypatch.setattr(psf, "load_packed_row_shard", forbidden_body_replay)
    monkeypatch.setattr(psf, "_validate_clustered_selection_outputs", forbidden_body_replay)
    actual = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    assert actual == expected
    upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert upgraded["metadata"][psf.REMOTE_LOADER_KEY][
        "max_features_per_segment"
    ] == certified_width


def test_legacy_clustered_manifest_replays_once_then_upgrades(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=4)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    expected = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"].pop(psf.SPLICE_RESUME_METADATA_KEY)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    real_stats = psf.splice_counterfactual_stats_for_packed_shard
    real_validation = psf._validate_clustered_selection_outputs
    real_storage_validation = psf._validate_and_fingerprint_clustered_selection_shard
    calls = {"stats": 0, "validation": 0, "storage_validation": 0}

    def recording_stats(*args, **kwargs):
        calls["stats"] += 1
        return real_stats(*args, **kwargs)

    def recording_validation(*args, **kwargs):
        calls["validation"] += 1
        return real_validation(*args, **kwargs)

    def recording_storage_validation(*args, **kwargs):
        calls["storage_validation"] += 1
        return real_storage_validation(*args, **kwargs)

    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        recording_stats,
    )
    monkeypatch.setattr(
        psf,
        "_validate_clustered_selection_outputs",
        recording_validation,
    )
    monkeypatch.setattr(
        psf,
        "_validate_and_fingerprint_clustered_selection_shard",
        recording_storage_validation,
    )
    upgraded = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    assert calls == {
        "stats": len(list(env["packed_row_dir"].glob("*.safetensors.zst"))),
        "validation": 0,
        "storage_validation": len(
            list(env["packed_row_dir"].glob("*.safetensors.zst"))
        ),
    }
    assert upgraded == expected


def test_v3_in_progress_resume_completes_without_selection_body_replay(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=4)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    expected = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
    manifest["metadata"]["splice_counterfactual"]["status"] = "in_progress"
    resume = manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    resume["output_validation"]["status"] = "in_progress"
    resume["aggregate_sha256"] = splice_certificate._canonical_json_sha256(
        {
            "schema_version": resume["schema_version"],
            psf.CLUSTER_DELTA_SEMANTICS_KEY: resume[
                psf.CLUSTER_DELTA_SEMANTICS_KEY
            ],
            "shards": resume["shards"],
            "output_validation": resume["output_validation"],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v3 in-progress completion replayed a shard body")

    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        forbidden,
    )
    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        forbidden,
    )
    monkeypatch.setattr(psf, "_run_splice_shards_parallel", forbidden)
    monkeypatch.setattr(psf, "_validate_clustered_selection_outputs", forbidden)
    monkeypatch.setattr(psf, "load_selection_shard", forbidden)
    monkeypatch.setattr(psf, "load_packed_row_shard", forbidden)
    upgraded = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    assert upgraded == expected
    upgraded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upgraded_resume = upgraded_manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]
    assert (
        upgraded_resume["schema_version"]
        == splice_certificate.SPLICE_RESUME_SCHEMA_VERSION
    )
    assert all(
        entry["storage_validation_content_sha256"]
        == entry["selection_content_sha256"]
        for entry in upgraded_resume["shards"]
    )


def test_resumed_splice_cannot_publish_cross_inconsistent_complete_evidence(
    tmp_path,
    monkeypatch,
):
    """A signed in-progress global width cannot contradict fresh shard proof."""

    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    integrity = metadata[SELECTION_SHARD_INTEGRITY_KEY]
    resume = metadata[psf.SPLICE_RESUME_METADATA_KEY]
    packed_rows_hash = metadata["packed_rows"]["packed_row_fingerprint"]
    effective_buckets = resume["output_validation"]["effective_buckets"]
    stats, _complete, storage_validated, actual_width = (
        psf._parse_splice_resume_metadata(
            resume,
            selection_integrity=integrity,
            packed_rows_hash=packed_rows_hash,
            effective_buckets=effective_buckets,
            manifest_status="complete",
            surface="completed fixture",
        )
    )
    assert actual_width is not None
    forged_width = actual_width + 1
    metadata["feature_clusters"]["status"] = "in_progress"
    metadata["splice_counterfactual"]["status"] = "in_progress"
    metadata[psf.SPLICE_RESUME_METADATA_KEY] = psf._build_splice_resume_metadata(
        stats_by_stem=stats,
        selection_integrity=integrity,
        storage_validated_by_stem=storage_validated,
        validated_max_features_per_segment=forged_width,
        packed_rows_hash=packed_rows_hash,
        effective_buckets=effective_buckets,
        status="in_progress",
    )
    metadata.pop(psf.CLUSTERED_SELECTION_VALIDATION_KEY)
    metadata.pop(psf.REMOTE_LOADER_KEY)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    shard_bytes = {
        path.name: path.read_bytes()
        for path in sorted(sel_dir.glob("shard_*.safetensors.zst"))
    }

    with pytest.raises(
        FeatureClustersError,
        match="certificates disagree on max_features_per_segment",
    ):
        run_selection_main(
            env,
            num_workers=1,
            monkeypatch=monkeypatch,
            **common,
        )

    failed = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_metadata = failed["metadata"]
    assert failed_metadata["feature_clusters"]["status"] == "in_progress"
    assert failed_metadata["splice_counterfactual"]["status"] == "in_progress"
    assert psf.CLUSTERED_SELECTION_VALIDATION_KEY not in failed_metadata
    assert psf.REMOTE_LOADER_KEY not in failed_metadata
    assert {
        path.name: path.read_bytes()
        for path in sorted(sel_dir.glob("shard_*.safetensors.zst"))
    } == shard_bytes


def test_malformed_splice_resume_certificate_fails_before_output_mutation(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=4)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"][psf.SPLICE_RESUME_METADATA_KEY]["shards"][0][
        "total_packed_segments"
    ] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(FeatureClustersError, match="non-negative strict integer"):
        run_selection_main(
            env,
            num_workers=3,
            monkeypatch=monkeypatch,
            **common,
        )
    assert manifest_path.read_bytes() == before


def test_old_cluster_delta_manifest_fails_before_resume_or_output_mutation(
    tmp_path,
    monkeypatch,
):
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splice = manifest["metadata"]["splice_counterfactual"]
    splice["schema_version"] = 1
    splice.pop(psf.CLUSTER_DELTA_SEMANTICS_KEY)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(psf.SpliceProvenanceError, match="schema_version"):
        run_selection_main(
            env,
            num_workers=3,
            monkeypatch=monkeypatch,
            **common,
        )
    assert manifest_path.read_bytes() == before


def test_resume_metadata_without_byte_certificate_fails_before_output_mutation(
    tmp_path,
    monkeypatch,
):
    """An advertised stats certificate can never outlive byte certification."""
    env = build_interleaved_env(tmp_path, n_shards=3)
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
    manifest["metadata"]["splice_counterfactual"]["status"] = "in_progress"
    manifest["metadata"].pop(SELECTION_SHARD_INTEGRITY_KEY)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    before_manifest = manifest_path.read_bytes()
    shard_paths = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    before_shards = {
        path.name: (path.stat().st_ino, path.read_bytes()) for path in shard_paths
    }

    with pytest.raises(
        FeatureClustersError,
        match="splice_resume without any byte-certified reusable selection shards",
    ):
        run_selection_main(
            env,
            num_workers=3,
            monkeypatch=monkeypatch,
            **common,
        )

    assert manifest_path.read_bytes() == before_manifest
    assert {
        path.name: (path.stat().st_ino, path.read_bytes()) for path in shard_paths
    } == before_shards


def test_shared_store_worker_write_failure_cleans_and_resumes_exactly(
    tmp_path,
    monkeypatch,
):
    """A real spawn-task failure cannot strand the store or bless peer output.

    A directory at the first shard's exact destination makes the worker's
    atomic ``os.replace`` fail after all parent validation and pool startup.
    Later workers may already have published files, but ordered result
    consumption means none can be certified past the failed first shard. The
    next default run therefore overwrites every uncommitted file and must
    reproduce the clean W=1 tree exactly.
    """
    env = build_interleaved_env(tmp_path, n_shards=4)
    shm = tmp_path / "shm"
    shm.mkdir()
    common: _SelectionCommonKwargs = {
        "cache_shards": 3,
        "sidecar_shards": 8,
        "mask": WHITELIST_MASK_BELOW_150,
        "clusters_path": write_cluster_yaml(tmp_path),
    }
    complete = run_selection_main(
        env,
        num_workers=1,
        monkeypatch=monkeypatch,
        **common,
    )
    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    manifest_path = sel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
    manifest["metadata"]["splice_counterfactual"]["status"] = "in_progress"
    manifest["metadata"].pop(SELECTION_SHARD_INTEGRITY_KEY, None)
    manifest["metadata"].pop(psf.SPLICE_RESUME_METADATA_KEY, None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in sel_dir.glob("shard_*.safetensors.zst"):
        path.unlink()
    blocker = sel_dir / "shard_00000.safetensors.zst"
    blocker.mkdir()

    with pytest.raises(OSError):
        run_selection_main(
            env,
            num_workers=2,
            shared_store=True,
            shared_store_capacity=2,
            shm_dir=str(shm),
            monkeypatch=monkeypatch,
            **common,
        )

    assert list(shm.iterdir()) == []
    assert list(sel_dir.glob("*.safetensors.zst.tmp.*")) == []
    failed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failed_manifest["metadata"]["feature_clusters"]["status"] == "in_progress"
    assert SELECTION_SHARD_INTEGRITY_KEY not in failed_manifest["metadata"]

    blocker.rmdir()
    resumed = run_selection_main(
        env,
        num_workers=2,
        shared_store=True,
        shared_store_capacity=2,
        shm_dir=str(shm),
        monkeypatch=monkeypatch,
        **common,
    )

    assert resumed == complete
    assert list(shm.iterdir()) == []


def test_dead_atomic_selection_temps_are_removed_only_in_expected_domain(
    tmp_path,
):
    output_dir = tmp_path / "sel"
    output_dir.mkdir()
    stale = output_dir / "shard_00000.safetensors.zst.tmp.2000000000.1234abcd"
    stale.write_bytes(b"incomplete atomic output")
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("preserve me", encoding="utf-8")

    removed = psf._sweep_dead_atomic_selection_temps(
        output_dir,
        {"shard_00000"},
    )

    assert removed == (stale.name,)
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_live_atomic_selection_temp_is_a_concurrent_writer_error(tmp_path):
    output_dir = tmp_path / "sel"
    output_dir.mkdir()
    temporary = (
        output_dir
        / f"shard_00000.safetensors.zst.tmp.{os.getpid()}.1234abcd"
    )
    temporary.write_bytes(b"possibly live")

    with pytest.raises(RuntimeError, match="live PID.*refusing concurrent"):
        psf._sweep_dead_atomic_selection_temps(output_dir, {"shard_00000"})

    assert temporary.read_bytes() == b"possibly live"


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        (
            "shard_00000.safetensors.zst.tmp.not-a-pid.bad",
            "file",
            "malformed atomic temporary",
        ),
        (
            "shard_99999.safetensors.zst.tmp.2000000000.1234abcd",
            "file",
            "outside the current packed-row domain",
        ),
        (
            "shard_00000.safetensors.zst.tmp.2000000000.1234abcd",
            "symlink",
            "must be a real regular file",
        ),
    ],
)
def test_atomic_selection_temp_sweeper_never_guesses_at_unsafe_entries(
    tmp_path,
    name: str,
    kind: str,
    message: str,
):
    output_dir = tmp_path / "sel"
    output_dir.mkdir()
    path = output_dir / name
    if kind == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"must not unlink through this")
        path.symlink_to(target)
    else:
        path.write_bytes(b"preserve ambiguous entry")

    with pytest.raises(RuntimeError, match=message):
        psf._sweep_dead_atomic_selection_temps(output_dir, {"shard_00000"})

    assert path.exists() or path.is_symlink()
