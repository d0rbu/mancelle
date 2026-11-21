"""Unit/functional tests for fail-closed parallel resource admission."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from phantom.sized import NonEmptyStr

import steering.conditional.splice_resource_admission as admission_module
from steering.conditional.feature_index import (
    FeatureIndex,
    FeatureIndexCatalogError,
)
from steering.conditional.splice_resource_admission import (
    AdmissionPolicy,
    CapacitySnapshot,
    FilesystemCapacity,
    SpliceResourceAdmissionError,
    _read_cgroup_available_bytes,
    admit_splice_resources,
    capture_capacity_snapshot,
    evaluate_splice_resource_admission,
    parse_mountinfo,
)
from steering.feature_io import TensorKey, save_features
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard
from steering.refined_types import (
    AtLeastTwoInt,
    NonNegativeInt,
    PartsPerMillion,
    PositiveInt,
)
from steering.safetensors_layout import (
    CompressedSafetensorsLayout,
    SafetensorsLayoutError,
    read_compressed_safetensors_layout,
)

PCT_CHECKSUM = "a" * 64
HUGE = 1 << 50


def _source_data(shard_idx: int, *, feature_count: int) -> dict:
    n_segments = 2 + shard_idx
    counts = [feature_count // n_segments] * n_segments
    for i in range(feature_count % n_segments):
        counts[i] += 1
    assert all(count > 0 for count in counts)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    feature_ids = [
        value
        for count in counts
        for value in range(shard_idx * 100, shard_idx * 100 + count)
    ]
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.linspace(0.1, 1.0, feature_count, dtype=t.float32),
        "doc_ids": t.arange(n_segments, dtype=t.int32),
        "segment_indices": t.zeros(n_segments, dtype=t.int32),
        "segment_positions": t.arange(n_segments, dtype=t.int32) * 2,
        "doc_token_counts": t.full((n_segments,), 2, dtype=t.int32),
        "segment_char_starts": t.arange(n_segments, dtype=t.int32) * 4,
        "segment_char_ends": t.arange(1, n_segments + 1, dtype=t.int32) * 4,
        "n_docs": n_segments,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": 16,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 2,
    }


def _packed_shard(*, n_segments: int, n_features: int) -> PackedRowShard:
    assert n_segments >= 1
    counts = [n_features // n_segments] * n_segments
    for i in range(n_features % n_segments):
        counts[i] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return PackedRowShard(
        tokens=t.arange(9, dtype=t.int32).reshape(1, 9).to(t.uint16),
        row_seg_offsets=t.tensor([0, n_segments], dtype=t.int32),
        seg_token_start=t.arange(n_segments, dtype=t.int32).to(t.uint16),
        seg_token_end=(t.arange(n_segments, dtype=t.int32) + 1).to(t.uint16),
        seg_doc_id=t.arange(n_segments, dtype=t.int32).remainder(2).to(t.uint8),
        seg_feat_offsets=t.tensor(offsets, dtype=t.int32),
        feature_ids=t.arange(n_features, dtype=t.int32).to(t.uint16),
        feature_pcts=t.arange(n_features, dtype=t.int32).remainder(256).to(t.uint8),
    )


def _build_artifacts(root: Path, *, source_counts: tuple[int, ...] = (7, 13, 19)) -> dict:
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True)
    stems: list[str] = []
    source_paths: list[Path] = []
    for shard_idx, feature_count in enumerate(source_counts):
        stem = f"shard_{shard_idx:05d}"
        stems.append(stem)
        source_paths.append(
            save_features(
                _source_data(shard_idx, feature_count=feature_count),
                feature_dir / f"{stem}_features",
                zstd_level=1,
            )
        )

    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, PCT_CHECKSUM)
    write_manifest(pct_dir, PCT_CHECKSUM, OverlayKind.PERCENTILE)
    for stem, feature_count in zip(stems, source_counts, strict=True):
        save_overlay_shard(
            {
                TensorKey.FEATURE_PERCENTILES: t.arange(
                    feature_count, dtype=t.int32
                )
                .remainder(256)
                .to(t.uint8)
            },
            overlay_shard_path(pct_dir, stem),
            zstd_level=1,
        )

    sidecar_dir = feature_dir / PROVENANCE_DIRNAME
    sidecar_dir.mkdir()
    for shard_idx, stem in enumerate(stems):
        # Admission uses the exact file size as the explicit compact-sidecar
        # estimate; the production parser remains the schema validator.
        (sidecar_dir / f"{stem}.json").write_text(
            json.dumps({"schema_version": 2, "padding": "x" * (10 + shard_idx)})
        )

    catalog = FeatureIndex(
        str(feature_dir), percentile_checksum=PCT_CHECKSUM
    ).export_catalog()

    packed_dir = root / "packed"
    provenance_dir = packed_dir / "provenance"
    packed_dir.mkdir()
    provenance_dir.mkdir()
    packed_paths: list[Path] = []
    provenance_paths: list[Path] = []
    for shard_idx, (n_segments, n_features) in enumerate(((2, 5), (3, 11), (4, 23))):
        name = f"packed_{shard_idx:05d}.safetensors.zst"
        packed_path = save_packed_row_shard(
            _packed_shard(n_segments=n_segments, n_features=n_features),
            packed_dir / name,
            zstd_level=1,
        )
        provenance_path = save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.arange(
                    n_segments, dtype=t.int32
                ).remainder(len(stems)),
                PROVENANCE_DOC_IDX_KEY: t.arange(n_segments, dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.zeros(n_segments, dtype=t.int32),
            },
            provenance_dir / name,
            zstd_level=1,
        )
        packed_paths.append(packed_path)
        provenance_paths.append(provenance_path)
    return {
        "feature_dir": feature_dir,
        "stems": stems,
        "source_paths": source_paths,
        "catalog": catalog,
        "packed_paths": packed_paths,
        "provenance_paths": provenance_paths,
        "spool_dir": root / "spool",
        "store_dir": root / "store",
    }


@pytest.fixture
def artifacts(tmp_path: Path) -> dict:
    result = _build_artifacts(tmp_path)
    result["spool_dir"].mkdir()
    result["store_dir"].mkdir()
    return result


def _filesystem(
    path: Path, *, available: int = HUGE, fs_type: str = "ext4", block: int = 4096
) -> FilesystemCapacity:
    return FilesystemCapacity(
        path=NonEmptyStr.parse(str(path.resolve())),
        mount_point=NonEmptyStr.parse(str(path.resolve())),
        fs_type=NonEmptyStr.parse(fs_type),
        available_bytes=NonNegativeInt.parse(available),
        block_size=PositiveInt.parse(block),
    )


def _capacities(
    artifacts: dict,
    *,
    memory: int = HUGE,
    cgroup: int | None = None,
    store_available: int = HUGE,
    store_fs: str = "tmpfs",
    spool_available: int = HUGE,
    spool_fs: str = "ext4",
) -> CapacitySnapshot:
    effective = min(memory, cgroup) if cgroup is not None else memory
    return CapacitySnapshot(
        host_mem_available_bytes=NonNegativeInt.parse(memory),
        cgroup_memory_available_bytes=(
            NonNegativeInt.parse(cgroup) if cgroup is not None else None
        ),
        effective_memory_available_bytes=NonNegativeInt.parse(effective),
        shared_store=_filesystem(
            artifacts["store_dir"], available=store_available, fs_type=store_fs
        ),
        packed_spool=_filesystem(
            artifacts["spool_dir"], available=spool_available, fs_type=spool_fs
        ),
    )


def _kwargs(artifacts: dict, **updates) -> dict:
    values = {
        "packed_shard_paths": artifacts["packed_paths"],
        "provenance_shard_paths": artifacts["provenance_paths"],
        "base_shard_stems": artifacts["stems"],
        "feature_index_catalog": artifacts["catalog"],
        "n_sae_features": 65_536,
        "skip_feature_mags": True,
        "num_workers": 2,
        "source_cache_shards": 2,
        "sidecar_cache_shards": 2,
        "has_selection_work": True,
        "shared_store_enabled": True,
        "shared_store_capacity": 2,
        "packed_spool_dir": artifacts["spool_dir"],
        "shared_store_root": artifacts["store_dir"],
        "capacities": _capacities(artifacts),
    }
    values.update(updates)
    return values


class TestAdmissionAccounting:
    def test_supported_shared_spooled_configuration_reports_every_component(
        self, artifacts: dict
    ) -> None:
        report = evaluate_splice_resource_admission(**_kwargs(artifacts))

        assert report.decision == "admit"
        assert report.violations == ()
        assert report.schema_version == 1
        assert report.requested_workers == 2
        assert report.active_workers == 2
        assert report.has_selection_work is True
        assert report.shared_store_enabled is True
        assert report.shared_store_capacity == 2
        assert report.packed_storage_mode == "disk_spool"
        assert len(report.source_stems) == len(artifacts["stems"])
        assert report.exact_bytes["packed_serialized_resident"] > 0
        assert report.exact_bytes["packed_spool_required"] == report.exact_bytes[
            "packed_serialized_resident"
        ]
        assert report.exact_bytes["shared_store_union_stems"] == 3
        assert report.estimated_bytes["shared_store_union_filesystem"] == sum(
            item.shared_allocated_bytes for item in report.source_stems
        )
        assert report.exact_bytes["shared_store_union_tensor_payload"] == sum(
            item.shared_tensor_bytes for item in report.source_stems
        )
        assert report.estimated_bytes["private_source_cache"] == 0
        assert report.estimated_bytes["source_materialization_peak"] == max(
            item.materialization_private_peak_bytes for item in report.source_stems
        )
        assert report.estimated_bytes["required_with_host_reserve"] < HUGE
        assert "not a guarantee" in report.basis

    def test_json_is_stable_roundtrippable_and_complete(self, artifacts: dict) -> None:
        report = evaluate_splice_resource_admission(**_kwargs(artifacts))
        raw = report.to_json()
        assert raw == report.to_json()
        decoded = json.loads(raw)
        assert decoded == json.loads(json.dumps(report.to_dict()))
        assert decoded["decision"] == "admit"
        assert decoded["capacities"]["shared_store"]["fs_type"] == "tmpfs"
        assert decoded["source_stems"][0]["stem"] == artifacts["stems"][0]

    def test_shared_lazy_document_locator_is_exactly_header_accounted(
        self, artifacts: dict
    ) -> None:
        shared = evaluate_splice_resource_admission(**_kwargs(artifacts))
        private = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                shared_store_enabled=False,
                shared_store_capacity=0,
                shared_store_root=None,
            )
        )
        assert shared.capacities.shared_store is not None
        block_size = shared.capacities.shared_store.block_size
        assert block_size > 0
        for shared_stem, private_stem, source_path in zip(
            shared.source_stems,
            private.source_stems,
            artifacts["source_paths"],
            strict=True,
        ):
            layout = read_compressed_safetensors_layout(source_path)
            n_segments = int(dict(layout.metadata)["n_segments"])
            locator_bytes = 2 * n_segments * 8
            locator_build_peak = 40 * n_segments
            locator_allocated = 2 * (
                (n_segments * 8 + block_size - 1) // block_size * block_size
            )

            assert shared_stem.shared_doc_index_tensor_bytes == locator_bytes
            assert shared_stem.shared_doc_index_build_peak_bytes == locator_build_peak
            assert private_stem.shared_doc_index_tensor_bytes == 0
            assert private_stem.shared_doc_index_build_peak_bytes == 0
            assert (
                shared_stem.shared_tensor_bytes - private_stem.shared_tensor_bytes
                == locator_bytes
            )
            assert (
                shared_stem.shared_allocated_bytes
                - private_stem.shared_allocated_bytes
                == locator_allocated
            )
            assert shared_stem.materialization_private_peak_bytes >= (
                locator_build_peak
            )

    def test_active_workers_are_bounded_by_packed_shards(self, artifacts: dict) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(artifacts, num_workers=16)
        )
        assert report.requested_workers == 16
        assert report.active_workers == len(artifacts["packed_paths"])

    def test_eager_counts_compressed_transient_but_spool_does_not(
        self, artifacts: dict
    ) -> None:
        spooled = evaluate_splice_resource_admission(**_kwargs(artifacts))
        eager = evaluate_splice_resource_admission(
            **_kwargs(artifacts, packed_spool_dir=None)
        )
        assert eager.packed_storage_mode == "eager"
        assert spooled.packed_storage_mode == "disk_spool"
        assert eager.exact_bytes["packed_spool_required"] == 0
        assert (
            eager.estimated_bytes["packed_memory_peak"]
            == eager.exact_bytes["packed_eager_compressed_plus_serialized"]
        )
        assert (
            spooled.estimated_bytes["packed_memory_peak"]
            == spooled.exact_bytes["packed_serialized_resident"]
        )
        assert eager.estimated_bytes["packed_memory_peak"] > spooled.estimated_bytes[
            "packed_memory_peak"
        ]

    def test_complete_resume_omits_all_source_costs(self, artifacts: dict) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(artifacts, has_selection_work=False)
        )
        assert report.source_stems == ()
        assert report.exact_bytes["shared_store_union_stems"] == 0
        assert report.exact_bytes["shared_store_union_tensor_payload"] == 0
        assert report.estimated_bytes["shared_store_union_filesystem"] == 0
        assert report.estimated_bytes["private_source_cache"] == 0
        assert report.estimated_bytes["source_materialization_peak"] == 0
        # Stats-only resume still opens packed rows, packed provenance, and sidecars.
        assert report.estimated_bytes["packed_memory_peak"] > 0
        assert report.estimated_bytes["provenance_memory_peak"] > 0
        assert report.estimated_bytes["compact_sidecar_cache"] > 0

    @pytest.mark.parametrize(
        ("capacity", "workers", "expected_union"),
        [(1, 2, 3), (2, 2, 3), (100, 2, 3), (1, 16, 3)],
    )
    def test_unlink_aware_store_union_is_c_plus_two_w_plus_publication_capped_by_sources(
        self,
        artifacts: dict,
        capacity: int,
        workers: int,
        expected_union: int,
    ) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                shared_store_capacity=capacity,
                num_workers=workers,
            )
        )
        assert report.exact_bytes["shared_store_union_stems"] == expected_union

    def test_store_union_counts_old_and_inflight_mapping_for_every_worker(
        self, tmp_path: Path
    ) -> None:
        artifacts = _build_artifacts(
            tmp_path,
            source_counts=tuple(range(5, 17)),
        )
        artifacts["spool_dir"].mkdir()
        artifacts["store_dir"].mkdir()
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                num_workers=3,
                shared_store_capacity=2,
            )
        )
        # C visible + W old resident mappings + W in-flight next mappings
        # + one publication temporary, capped by the 12 source stems.
        assert report.exact_bytes["shared_store_union_stems"] == 2 + 3 + 3 + 1

    def test_store_union_selects_largest_allocations(self, artifacts: dict) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(artifacts, shared_store_capacity=1, num_workers=2)
        )
        count = report.exact_bytes["shared_store_union_stems"]
        expected = sum(
            sorted(
                (item.shared_allocated_bytes for item in report.source_stems),
                reverse=True,
            )[:count]
        )
        assert report.estimated_bytes["shared_store_union_filesystem"] == expected
        expected_tensor = sum(
            sorted(
                (item.shared_tensor_bytes for item in report.source_stems),
                reverse=True,
            )[:count]
        )
        assert report.exact_bytes["shared_store_union_tensor_payload"] == expected_tensor

    def test_private_mode_multiplies_cache_and_build_by_active_workers(
        self, artifacts: dict
    ) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                shared_store_enabled=False,
                shared_store_capacity=0,
                shared_store_root=None,
            )
        )
        per_worker_cache = sum(
            sorted(
                (
                    item.private_resident_tensor_bytes
                    for item in report.source_stems
                ),
                reverse=True,
            )[:2]
        )
        assert report.exact_bytes["shared_store_union_tensor_payload"] == 0
        assert report.estimated_bytes["shared_store_union_filesystem"] == 0
        assert report.estimated_bytes["private_source_cache"] == (
            report.active_workers * per_worker_cache
        )
        assert report.estimated_bytes["source_materialization_peak"] == (
            report.active_workers
            * max(item.private_build_peak_bytes for item in report.source_stems)
        )

    def test_uint16_is_used_only_at_vocabulary_width_at_most_65536(
        self, artifacts: dict
    ) -> None:
        narrow = evaluate_splice_resource_admission(
            **_kwargs(artifacts, n_sae_features=65_536)
        )
        wide = evaluate_splice_resource_admission(
            **_kwargs(artifacts, n_sae_features=65_537)
        )
        feature_counts = [
            read_compressed_safetensors_layout(path).tensor("feature_ids").numel
            for path in artifacts["source_paths"]
        ]
        for narrow_stem, wide_stem, count in zip(
            narrow.source_stems, wide.source_stems, feature_counts, strict=True
        ):
            assert wide_stem.shared_tensor_bytes - narrow_stem.shared_tensor_bytes == (
                2 * count
            )
            assert (
                wide_stem.private_resident_tensor_bytes
                - narrow_stem.private_resident_tensor_bytes
                == 2 * count
            )

    def test_magnitude_skip_removes_exact_float32_runtime_bytes(
        self, artifacts: dict
    ) -> None:
        skipped = evaluate_splice_resource_admission(
            **_kwargs(artifacts, skip_feature_mags=True)
        )
        retained = evaluate_splice_resource_admission(
            **_kwargs(artifacts, skip_feature_mags=False)
        )
        for skipped_stem, retained_stem, path in zip(
            skipped.source_stems,
            retained.source_stems,
            artifacts["source_paths"],
            strict=True,
        ):
            count = read_compressed_safetensors_layout(path).tensor(
                "feature_mags"
            ).numel
            assert retained_stem.shared_tensor_bytes - skipped_stem.shared_tensor_bytes == (
                4 * count
            )
            assert (
                retained_stem.private_resident_tensor_bytes
                - skipped_stem.private_resident_tensor_bytes
                == 4 * count
            )

    def test_sidecar_estimate_uses_largest_cache_entries_per_worker(
        self, artifacts: dict
    ) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(artifacts, sidecar_cache_shards=2)
        )
        sizes = [
            (artifacts["feature_dir"] / PROVENANCE_DIRNAME / f"{stem}.json").stat().st_size
            for stem in artifacts["stems"]
        ]
        assert report.estimated_bytes["compact_sidecar_cache"] == (
            report.active_workers * sum(sorted(sizes, reverse=True)[:2])
        )

    def test_worker_private_floor_applies_to_tiny_artifacts(self, artifacts: dict) -> None:
        policy = AdmissionPolicy(
            worker_private_floor_bytes=NonNegativeInt.parse(123_456)
        )
        report = evaluate_splice_resource_admission(
            **_kwargs(artifacts, policy=policy)
        )
        assert report.estimated_bytes["worker_private_residual"] >= (
            report.active_workers * 123_456
        )


class TestAdmissionBoundaries:
    def test_store_exact_boundary_passes_and_one_byte_short_rejects(
        self, artifacts: dict
    ) -> None:
        policy = AdmissionPolicy(store_free_reserve_bytes=NonNegativeInt.parse(101))
        baseline = evaluate_splice_resource_admission(
            **_kwargs(artifacts, policy=policy)
        )
        needed = baseline.estimated_bytes["shared_store_union_filesystem"] + 101

        exact = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, store_available=needed),
            )
        )
        short = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, store_available=needed - 1),
            )
        )
        assert exact.decision == "admit"
        assert not any("shared-store filesystem" in item for item in exact.violations)
        assert short.decision == "reject"
        assert any("shared-store filesystem" in item for item in short.violations)

    def test_spool_exact_boundary_passes_and_one_byte_short_rejects(
        self, artifacts: dict
    ) -> None:
        policy = AdmissionPolicy(
            spool_min_free_reserve_bytes=NonNegativeInt.parse(100),
            spool_max_free_reserve_bytes=NonNegativeInt.parse(100),
            spool_free_reserve_fraction_ppm=PartsPerMillion.parse(0),
        )
        baseline = evaluate_splice_resource_admission(
            **_kwargs(artifacts, policy=policy)
        )
        needed = baseline.exact_bytes["packed_spool_required"] + 100
        exact = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, spool_available=needed),
            )
        )
        short = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, spool_available=needed - 1),
            )
        )
        assert exact.decision == "admit"
        assert not any("spool filesystem" in item for item in exact.violations)
        assert short.decision == "reject"
        assert any("spool filesystem" in item for item in short.violations)

    def test_memory_exact_boundary_passes_and_one_byte_short_rejects(
        self, artifacts: dict
    ) -> None:
        policy = AdmissionPolicy(
            host_min_reserve_bytes=NonNegativeInt.parse(0),
            host_max_reserve_bytes=NonNegativeInt.parse(0),
            host_reserve_fraction_ppm=PartsPerMillion.parse(0),
        )
        baseline = evaluate_splice_resource_admission(
            **_kwargs(artifacts, policy=policy)
        )
        needed = baseline.estimated_bytes["incremental_peak_before_host_reserve"]
        exact = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, memory=needed),
            )
        )
        short = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(artifacts, memory=needed - 1),
            )
        )
        assert exact.decision == "admit"
        assert not any("memory headroom" in item for item in exact.violations)
        assert short.decision == "reject"
        assert any("memory headroom" in item for item in short.violations)

    def test_cgroup_headroom_is_the_effective_memory_limit(self, artifacts: dict) -> None:
        policy = AdmissionPolicy(
            host_min_reserve_bytes=NonNegativeInt.parse(0),
            host_max_reserve_bytes=NonNegativeInt.parse(0),
            host_reserve_fraction_ppm=PartsPerMillion.parse(0),
        )
        baseline = evaluate_splice_resource_admission(
            **_kwargs(artifacts, policy=policy)
        )
        needed = baseline.estimated_bytes["incremental_peak_before_host_reserve"]
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                policy=policy,
                capacities=_capacities(
                    artifacts, memory=HUGE, cgroup=needed - 1
                ),
            )
        )
        assert report.capacities.effective_memory_available_bytes == needed - 1
        assert any("memory headroom" in item for item in report.violations)

    @pytest.mark.parametrize("fs_type", ["tmpfs", "ramfs"])
    def test_memory_backed_spool_is_always_rejected(
        self, artifacts: dict, fs_type: str
    ) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                capacities=_capacities(artifacts, spool_fs=fs_type),
            )
        )
        assert report.decision == "reject"
        assert any(f"memory-backed ({fs_type})" in item for item in report.violations)

    def test_eager_mode_does_not_apply_spool_capacity_or_fstype(
        self, artifacts: dict
    ) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                packed_spool_dir=None,
                capacities=_capacities(
                    artifacts, spool_available=0, spool_fs="tmpfs"
                ),
            )
        )
        assert not any("spool" in item for item in report.violations)

    def test_multiple_independent_violations_are_all_reported(self, artifacts: dict) -> None:
        report = evaluate_splice_resource_admission(
            **_kwargs(
                artifacts,
                capacities=_capacities(
                    artifacts,
                    memory=1,
                    store_available=1,
                    spool_available=1,
                    spool_fs="tmpfs",
                ),
            )
        )
        assert report.decision == "reject"
        assert len(report.violations) == 4
        assert any("shared-store" in item for item in report.violations)
        assert any("memory-backed" in item for item in report.violations)
        assert any("spool filesystem" in item for item in report.violations)
        assert any("memory headroom" in item for item in report.violations)

    def test_admit_wrapper_raises_with_the_exact_report(self, artifacts: dict) -> None:
        kwargs = _kwargs(
            artifacts,
            capacities=_capacities(artifacts, memory=1, store_available=1),
        )
        expected = evaluate_splice_resource_admission(**kwargs)
        with pytest.raises(SpliceResourceAdmissionError) as caught:
            admit_splice_resources(**kwargs)
        assert caught.value.report == expected
        assert expected.to_json() in str(caught.value)
        assert "rejected" in str(caught.value)

    def test_admit_wrapper_returns_pass_report(self, artifacts: dict) -> None:
        report = admit_splice_resources(**_kwargs(artifacts))
        assert report.decision == "admit"
        assert report.violations == ()


class TestArtifactValidation:
    def test_catalog_drift_is_detected_before_accounting(self, artifacts: dict) -> None:
        source = artifacts["source_paths"][1]
        source.touch()
        with pytest.raises(FeatureIndexCatalogError, match="changed after parent"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_missing_sidecar_is_loud(self, artifacts: dict) -> None:
        path = (
            artifacts["feature_dir"]
            / PROVENANCE_DIRNAME
            / f"{artifacts['stems'][0]}.json"
        )
        path.unlink()
        with pytest.raises(RuntimeError, match="cannot stat splice sidecar"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_missing_catalog_stem_is_loud(self, artifacts: dict) -> None:
        with pytest.raises(SafetensorsLayoutError, match="no complete source"):
            evaluate_splice_resource_admission(
                **_kwargs(
                    artifacts,
                    base_shard_stems=[*artifacts["stems"], "shard_99999"],
                )
            )

    def test_absent_percentile_catalog_is_loud(self, artifacts: dict) -> None:
        catalog = replace(
            artifacts["catalog"],
            percentile_checksum=None,
            percentile_overlay_dir=None,
            percentile_overlay_entries=(),
            signatures=tuple(
                signature
                for signature in artifacts["catalog"].signatures
                if "/pct/" not in signature.path
            ),
        )
        # The structural catalog itself is valid after removing every percentile
        # field/signature; admission then rejects the missing required overlay.
        catalog.validate()
        with pytest.raises(SafetensorsLayoutError, match="requires a percentile"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, feature_index_catalog=catalog)
            )

    def test_packed_and_provenance_counts_must_match(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="counts must match"):
            evaluate_splice_resource_admission(
                **_kwargs(
                    artifacts,
                    provenance_shard_paths=artifacts["provenance_paths"][:-1],
                )
            )

    def test_provenance_must_match_packed_filename(self, artifacts: dict) -> None:
        renamed = artifacts["provenance_paths"][0].with_name(
            "unmatched.safetensors.zst"
        )
        artifacts["provenance_paths"][0].rename(renamed)
        paths = [renamed, *artifacts["provenance_paths"][1:]]
        with pytest.raises(ValueError, match="basenames must match exactly"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, provenance_shard_paths=paths)
            )

    def test_provenance_length_mismatch_is_loud(self, artifacts: dict) -> None:
        path = artifacts["provenance_paths"][0]
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.zeros(1, dtype=t.int32),
                PROVENANCE_DOC_IDX_KEY: t.zeros(1, dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.zeros(1, dtype=t.int32),
            },
            path,
            zstd_level=1,
        )
        with pytest.raises(SafetensorsLayoutError, match=r"expected \(2,\)"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_provenance_extra_tensor_is_loud(self, artifacts: dict) -> None:
        path = artifacts["provenance_paths"][0]
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.zeros(2, dtype=t.int32),
                PROVENANCE_DOC_IDX_KEY: t.zeros(2, dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.zeros(2, dtype=t.int32),
                "extra": t.zeros(2, dtype=t.int32),
            },
            path,
            zstd_level=1,
        )
        with pytest.raises(SafetensorsLayoutError, match="provenance tensor keys"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_packed_extra_tensor_is_loud(self, artifacts: dict) -> None:
        path = artifacts["packed_paths"][0]
        shard = _packed_shard(n_segments=2, n_features=5)
        save_overlay_shard(
            {
                **{
                    name: getattr(shard, name)
                    for name in (
                        "tokens",
                        "row_seg_offsets",
                        "seg_token_start",
                        "seg_token_end",
                        "seg_doc_id",
                        "seg_feat_offsets",
                        "feature_ids",
                        "feature_pcts",
                    )
                },
                "extra": t.zeros(1, dtype=t.uint8),
            },
            path,
            zstd_level=1,
        )
        with pytest.raises(SafetensorsLayoutError, match="packed tensor keys"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_packed_relational_shape_mismatch_is_loud(self, artifacts: dict) -> None:
        path = artifacts["packed_paths"][0]
        shard = _packed_shard(n_segments=2, n_features=5)
        shard.feature_pcts = t.zeros(4, dtype=t.uint8)
        save_packed_row_shard(shard, path, zstd_level=1)
        with pytest.raises(SafetensorsLayoutError, match="feature_pcts.*expected"):
            evaluate_splice_resource_admission(**_kwargs(artifacts))

    def test_no_source_layout_reads_for_complete_resume(self, artifacts: dict, monkeypatch) -> None:
        source_paths = {str(path.absolute()) for path in artifacts["source_paths"]}
        original = admission_module.read_compressed_safetensors_layout
        seen: list[str] = []

        def tracked(path):
            seen.append(str(Path(path).absolute()))
            return original(path)

        monkeypatch.setattr(
            admission_module, "read_compressed_safetensors_layout", tracked
        )
        evaluate_splice_resource_admission(
            **_kwargs(artifacts, has_selection_work=False)
        )
        assert source_paths.isdisjoint(seen)


class TestInputContracts:
    @pytest.mark.parametrize(
        ("name", "value", "type_name"),
        [
            ("num_workers", 1, "AtLeastTwoInt"),
            ("num_workers", True, "AtLeastTwoInt"),
            ("source_cache_shards", 0, "PositiveInt"),
            ("sidecar_cache_shards", False, "PositiveInt"),
            ("shared_store_capacity", -1, "NonNegativeInt"),
            ("n_sae_features", 0, "PositiveInt"),
        ],
    )
    def test_integer_inputs_are_strict(
        self, artifacts: dict, name: str, value: object, type_name: str
    ) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=rf"parameter {name}=.*{type_name}",
        ):
            evaluate_splice_resource_admission(**_kwargs(artifacts, **{name: value}))

    def test_empty_packed_set_is_rejected(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="at least one packed"):
            evaluate_splice_resource_admission(
                **_kwargs(
                    artifacts,
                    packed_shard_paths=[],
                    provenance_shard_paths=[],
                )
            )

    def test_enabled_store_requires_root(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="shared_store_root is required"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, shared_store_root=None)
            )

    def test_disabled_store_requires_zero_capacity(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="capacity must be 0"):
            evaluate_splice_resource_admission(
                **_kwargs(
                    artifacts,
                    shared_store_enabled=False,
                    shared_store_root=None,
                    shared_store_capacity=1,
                )
            )

    def test_enabled_store_requires_positive_capacity(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, shared_store_capacity=0)
            )

    @pytest.mark.parametrize(
        "name",
        ["skip_feature_mags", "has_selection_work", "shared_store_enabled"],
    )
    @pytest.mark.parametrize("value", [0, 1, None, "true"])
    def test_boolean_inputs_are_strict(
        self, artifacts: dict, name: str, value: object
    ) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=rf"parameter {name}=.*not instance of bool",
        ):
            evaluate_splice_resource_admission(**_kwargs(artifacts, **{name: value}))

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("packed_shard_paths", ["not-a-path"], "pathlib.Path"),
            ("provenance_shard_paths", ["not-a-path"], "pathlib.Path"),
            ("base_shard_stems", [""], "non-empty strings"),
            ("base_shard_stems", [1], "non-empty strings"),
        ],
    )
    def test_path_and_stem_element_types_are_strict(
        self, artifacts: dict, name: str, value: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            evaluate_splice_resource_admission(**_kwargs(artifacts, **{name: value}))

    def test_manual_container_validation_checks_every_element(
        self, artifacts: dict
    ) -> None:
        paths: list[object] = list(artifacts["packed_paths"])
        paths[-1] = "invalid-final-element"
        with pytest.raises(ValueError, match="pathlib.Path"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, packed_shard_paths=paths)
            )

    def test_duplicate_base_stems_are_rejected(self, artifacts: dict) -> None:
        with pytest.raises(ValueError, match="base_shard_stems.*duplicates"):
            evaluate_splice_resource_admission(
                **_kwargs(
                    artifacts,
                    base_shard_stems=[artifacts["stems"][0]] * 2,
                )
            )

    @pytest.mark.parametrize("kind", ["packed", "provenance"])
    def test_duplicate_shard_basenames_are_rejected(
        self, artifacts: dict, kind: str
    ) -> None:
        paths = list(artifacts[f"{kind}_paths"])
        paths[1] = paths[0]
        updates = {
            f"{kind}_shard_paths": paths,
        }
        with pytest.raises(ValueError, match=rf"{kind}_shard_paths.*unique basenames"):
            evaluate_splice_resource_admission(**_kwargs(artifacts, **updates))

    def test_shard_basename_sets_must_match_exactly(self, artifacts: dict) -> None:
        paths = list(artifacts["provenance_paths"])
        replacement = paths[0].with_name("unmatched.safetensors.zst")
        replacement.write_bytes(paths[0].read_bytes())
        paths[0] = replacement
        with pytest.raises(ValueError, match="basenames must match exactly"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, provenance_shard_paths=paths)
            )

    def test_snapshot_must_include_enabled_store(self, artifacts: dict) -> None:
        capacities = replace(_capacities(artifacts), shared_store=None)
        with pytest.raises(ValueError, match="missing shared-store"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, capacities=capacities)
            )

    def test_snapshot_must_include_configured_spool(self, artifacts: dict) -> None:
        capacities = replace(_capacities(artifacts), packed_spool=None)
        with pytest.raises(ValueError, match="missing packed-spool"):
            evaluate_splice_resource_admission(
                **_kwargs(artifacts, capacities=capacities)
            )

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"host_mem_available_bytes": -1}, "host_mem_available_bytes"),
            ({"host_mem_available_bytes": True}, "host_mem_available_bytes"),
            ({"cgroup_memory_available_bytes": -1}, "cgroup_memory_available_bytes"),
            ({"effective_memory_available_bytes": -1}, "effective_memory_available_bytes"),
            ({"effective_memory_available_bytes": 123}, "must equal the minimum"),
        ],
    )
    def test_capacity_scalar_contracts_are_strict(
        self, artifacts: dict, updates: dict, message: str
    ) -> None:
        if updates == {"effective_memory_available_bytes": 123}:
            capacities = replace(_capacities(artifacts), **updates)
            with pytest.raises(ValueError, match=message):
                evaluate_splice_resource_admission(
                    **_kwargs(artifacts, capacities=capacities)
                )
            return
        with pytest.raises(BeartypeCallHintParamViolation, match=message):
            replace(_capacities(artifacts), **updates)

    @pytest.mark.parametrize(
        ("field_name", "value", "message"),
        [
            ("path", "", "parameter path"),
            ("mount_point", "", "parameter mount_point"),
            ("fs_type", "", "parameter fs_type"),
            ("available_bytes", -1, "available_bytes"),
            ("available_bytes", True, "available_bytes"),
            ("block_size", 0, "block_size"),
        ],
    )
    def test_filesystem_capacity_contracts_are_strict(
        self,
        artifacts: dict,
        field_name: str,
        value: object,
        message: str,
    ) -> None:
        capacities = _capacities(artifacts)
        assert capacities.shared_store is not None
        with pytest.raises(BeartypeCallHintParamViolation, match=message):
            replace(capacities.shared_store, **{field_name: value})

    @pytest.mark.parametrize(
        "updates",
        [
            {"worker_packed_ratio_denominator": 0},
            {"host_reserve_fraction_ppm": 1_000_001},
            {"spool_free_reserve_fraction_ppm": 1_000_001},
            {
                "spool_min_free_reserve_bytes": 2,
                "spool_max_free_reserve_bytes": 1,
            },
            {"host_min_reserve_bytes": 2, "host_max_reserve_bytes": 1},
            {"parent_private_reserve_bytes": -1},
            {"worker_private_floor_bytes": True},
        ],
    )
    def test_policy_validation_is_fail_loud(self, artifacts: dict, updates: dict) -> None:
        relational_fields = {
            "spool_min_free_reserve_bytes",
            "spool_max_free_reserve_bytes",
            "host_min_reserve_bytes",
            "host_max_reserve_bytes",
        }
        if relational_fields.intersection(updates):
            policy = replace(AdmissionPolicy(), **updates)
            with pytest.raises(ValueError, match="reserve"):
                evaluate_splice_resource_admission(
                    **_kwargs(artifacts, policy=policy)
                )
            return
        field_name = next(iter(updates))
        with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
            replace(AdmissionPolicy(), **updates)


class TestLinuxCapacityDiscovery:
    def test_mountinfo_chooses_longest_matching_mount_and_decodes_escapes(
        self, tmp_path: Path
    ) -> None:
        nested = tmp_path / "space dir" / "nested"
        nested.mkdir(parents=True)
        root = str(tmp_path).replace(" ", "\\040")
        spaced = str(tmp_path / "space dir").replace(" ", "\\040")
        text = (
            f"1 0 0:1 / {root} rw - ext4 /dev/a rw\n"
            f"2 1 0:2 / {spaced} rw - xfs /dev/b rw\n"
        )
        result = parse_mountinfo(text, nested)
        assert result.mount_point == str(tmp_path / "space dir")
        assert result.fs_type == "xfs"

    def test_mountinfo_rejects_malformed_line(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no separator"):
            parse_mountinfo("malformed", tmp_path)

    def test_mountinfo_rejects_no_matching_entry(self, tmp_path: Path) -> None:
        text = "1 0 0:1 / /definitely/not/the/tmp/path rw - ext4 /dev/a rw\n"
        with pytest.raises(ValueError, match="no mountinfo entry"):
            parse_mountinfo(text, tmp_path)

    def test_cgroup_finite_limit_returns_remaining_headroom(self, tmp_path: Path) -> None:
        proc = tmp_path / "self.cgroup"
        proc.write_text("0::/jobs/abc\n")
        group = tmp_path / "cgroups" / "jobs" / "abc"
        group.mkdir(parents=True)
        (group / "memory.max").write_text("1000\n")
        (group / "memory.current").write_text("400\n")
        assert (
            _read_cgroup_available_bytes(
                proc_cgroup_path=proc, cgroup_root=tmp_path / "cgroups"
            )
            == 600
        )

    def test_cgroup_over_limit_clamps_to_zero(self, tmp_path: Path) -> None:
        proc = tmp_path / "self.cgroup"
        proc.write_text("0::/\n")
        group = tmp_path / "cgroups"
        group.mkdir()
        (group / "memory.max").write_text("100\n")
        (group / "memory.current").write_text("101\n")
        assert (
            _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=group)
            == 0
        )

    @pytest.mark.parametrize(
        ("cgroup_text", "maximum", "current"),
        [
            ("1:memory:/legacy\n", None, None),
            ("0::/\n", "max", "100"),
            ("0::/\n", "invalid", "100"),
            ("0::/\n", "100", "invalid"),
        ],
    )
    def test_unlimited_or_unreadable_cgroup_returns_none(
        self,
        tmp_path: Path,
        cgroup_text: str,
        maximum: str | None,
        current: str | None,
    ) -> None:
        proc = tmp_path / "self.cgroup"
        proc.write_text(cgroup_text)
        group = tmp_path / "cgroups"
        group.mkdir()
        if maximum is not None:
            (group / "memory.max").write_text(maximum)
        if current is not None:
            (group / "memory.current").write_text(current)
        assert (
            _read_cgroup_available_bytes(proc_cgroup_path=proc, cgroup_root=group)
            is None
        )

    def test_live_capacity_capture_is_read_only_and_self_consistent(
        self, artifacts: dict
    ) -> None:
        before_spool = list(artifacts["spool_dir"].iterdir())
        before_store = list(artifacts["store_dir"].iterdir())
        snapshot = capture_capacity_snapshot(
            shared_store_root=artifacts["store_dir"],
            packed_spool_dir=artifacts["spool_dir"],
        )
        assert snapshot.host_mem_available_bytes > 0
        assert snapshot.effective_memory_available_bytes <= snapshot.host_mem_available_bytes
        assert snapshot.shared_store is not None
        assert snapshot.shared_store.available_bytes >= 0
        assert snapshot.shared_store.block_size > 0
        assert snapshot.packed_spool is not None
        assert snapshot.packed_spool.available_bytes >= 0
        assert list(artifacts["spool_dir"].iterdir()) == before_spool
        assert list(artifacts["store_dir"].iterdir()) == before_store


@pytest.mark.requires_feature_data
def test_real_l22_full_corpus_reports_required_matrix_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in full production-artifact admission gate for issue 758.

    This intentionally scans all real source artifacts and all packed/provenance
    headers. It loads no model and launches no workers: splice preselection is a
    CPU/artifact pipeline, so a GPU-model test would exercise unrelated code.
    """
    raw_feature_dir = os.environ.get("SPLICE758_REAL_FEATURE_DIR")
    raw_packed_dir = os.environ.get("SPLICE758_REAL_PACKED_DIR")
    if not raw_feature_dir or not raw_packed_dir:
        pytest.skip(
            "set SPLICE758_REAL_FEATURE_DIR and SPLICE758_REAL_PACKED_DIR "
            "to run the full-corpus admission gate"
        )
    feature_dir = Path(raw_feature_dir)
    packed_dir = Path(raw_packed_dir)
    if not feature_dir.is_dir() or not packed_dir.is_dir():
        pytest.fail(
            f"real artifact directories are unavailable: {feature_dir}, {packed_dir}"
        )
    manifest = read_manifest(packed_dir)
    if manifest is None:
        pytest.fail(f"real packed directory has no manifest: {packed_dir}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        pytest.fail("real packed manifest metadata is not an object")
    pct_checksum = metadata.get("pct_checksum")
    raw_base_stems = metadata.get("base_shard_stems")
    if not isinstance(pct_checksum, str) or not pct_checksum:
        pytest.fail("real packed manifest has no pct_checksum")
    if not isinstance(raw_base_stems, list) or not all(
        isinstance(stem, str) and stem for stem in raw_base_stems
    ):
        pytest.fail("real packed manifest has invalid base_shard_stems")
    # Rebuild a precisely typed list after the runtime element check. This is
    # deliberately not a cast: the gated test should validate the production
    # manifest instead of merely promising its shape to the type checker.
    base_stems = [stem for stem in raw_base_stems if isinstance(stem, str)]

    packed_paths = glob_overlay_shards(packed_dir)
    provenance_paths = [provenance_dir(packed_dir) / path.name for path in packed_paths]
    shared_root = Path(os.environ.get("NANOCHAT_SHM_DIR", "/dev/shm"))
    before_owned = set(shared_root.glob(f"nanochat-sel-{os.getpid()}-*"))
    catalog = FeatureIndex(
        str(feature_dir), percentile_checksum=pct_checksum
    ).export_catalog()

    # The seven matrix points share exactly the same immutable artifact inputs.
    # Read each compressed layout once, but prove the file identity is stable
    # both around the real read and at every subsequent reuse. Without this
    # scoped cache the gated test would stream the full corpus five times while
    # adding no coverage or evidence.
    real_layout_reader = admission_module.read_compressed_safetensors_layout
    layout_cache: dict[Path, CompressedSafetensorsLayout] = {}
    layout_identities: dict[Path, tuple[int, int, int, int, int, int]] = {}

    def identity(path: Path) -> tuple[int, int, int, int, int, int]:
        result = path.stat()
        return (
            result.st_dev,
            result.st_ino,
            result.st_mode,
            result.st_size,
            result.st_mtime_ns,
            result.st_ctime_ns,
        )

    def stable_cached_layout(path: Path) -> CompressedSafetensorsLayout:
        path = path.absolute()
        before = identity(path)
        if path in layout_cache:
            assert before == layout_identities[path], f"artifact changed during matrix: {path}"
            return layout_cache[path]
        layout = real_layout_reader(path)
        after = identity(path)
        assert after == before, f"artifact changed while reading compressed layout: {path}"
        layout_cache[path] = layout
        layout_identities[path] = after
        return layout

    monkeypatch.setattr(
        admission_module,
        "read_compressed_safetensors_layout",
        stable_cached_layout,
    )

    def evaluate(workers: int, capacity: int):
        return evaluate_splice_resource_admission(
            packed_shard_paths=packed_paths,
            provenance_shard_paths=provenance_paths,
            base_shard_stems=base_stems,
            feature_index_catalog=catalog,
            n_sae_features=PositiveInt.parse(65_536),
            skip_feature_mags=True,
            num_workers=AtLeastTwoInt.parse(workers),
            source_cache_shards=PositiveInt.parse(5),
            sidecar_cache_shards=PositiveInt.parse(16),
            has_selection_work=True,
            shared_store_enabled=True,
            shared_store_capacity=NonNegativeInt.parse(capacity),
            packed_spool_dir=None,
            shared_store_root=shared_root,
        )

    shapes = {
        "w8c12": (8, 12),
        "w12c16": (12, 16),
        "w16c20": (16, 20),
        # Evidence-gathering candidates only: their host-memory decision is a
        # point-in-time value and is deliberately not asserted by this gate.
        "w2c8": (2, 8),
        "w3c6": (3, 6),
        "w3c12": (3, 12),
        "w4c8": (4, 8),
    }
    reports = {
        name: evaluate(workers, capacity)
        for name, (workers, capacity) in shapes.items()
    }
    for path, expected_identity in layout_identities.items():
        assert identity(path) == expected_identity, f"artifact changed during matrix: {path}"

    print(
        "ISSUE758_REAL_ADMISSION_MATRIX="
        + json.dumps(
            {
                name: {
                    "decision": report.decision,
                    "exact_bytes": report.exact_bytes,
                    "estimated_bytes": report.estimated_bytes,
                    "capacities": report.to_dict()["capacities"],
                    "violations": report.violations,
                    "source_stem_count": len(report.source_stems),
                }
                for name, report in reports.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert len(packed_paths) == 423
    assert len(base_stems) == len(set(base_stems)) == 370
    for name in ("w8c12", "w12c16", "w16c20"):
        report = reports[name]
        workers, capacity = shapes[name]
        assert report.active_workers == workers
        assert report.exact_bytes["shared_store_union_stems"] == (
            capacity + 2 * workers + 1
        )
        assert report.capacities.shared_store is not None
        assert report.estimated_bytes["shared_store_union_filesystem"] > (
            report.capacities.shared_store.available_bytes
        )
        assert report.decision == "reject"
        assert any("shared-store filesystem" in item for item in report.violations)
    assert set(shared_root.glob(f"nanochat-sel-{os.getpid()}-*")) == before_owned
