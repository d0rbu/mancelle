"""Functional/integration coverage for versioned splice segment batching."""

from __future__ import annotations

import inspect
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

import scripts.preprocess.precompute_selected_features as psf
from errors import FeatureClustersError
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.overlay_io import glob_overlay_shards, read_manifest
from steering.packed_batch_io import load_packed_row_shard, load_selection_shard
from tests.steering._splice_cf_cache_fixtures import (
    WHITELIST_MASK_BELOW_150,
    build_interleaved_env,
    build_multisplice_env,
    run_selection_main,
    write_cluster_yaml,
)


def _selection_dirs(packed_row_dir: Path) -> list[Path]:
    return sorted(path for path in (packed_row_dir / "sel").iterdir() if path.is_dir())


def _batching_descriptor(selection_dir: Path) -> dict[str, Any] | None:
    manifest = read_manifest(selection_dir)
    assert manifest is not None
    metadata = manifest.get("metadata")
    assert isinstance(metadata, dict)
    splice = metadata.get("splice_counterfactual")
    assert isinstance(splice, dict)
    descriptor = splice.get("selection_batching")
    assert descriptor is None or isinstance(descriptor, dict)
    return descriptor


def _dir_for_batch_size(packed_row_dir: Path, batch_size: int) -> Path:
    matches = []
    for path in _selection_dirs(packed_row_dir):
        descriptor = _batching_descriptor(path)
        if (
            (batch_size == 1 and descriptor is None)
            or (
                descriptor is not None
                and descriptor.get("batch_size") == batch_size
            )
        ):
            matches.append(path)
    assert len(matches) == 1, (batch_size, matches)
    return matches[0]


def _assert_selection_shard_semantics(
    *,
    selection_path: Path,
    packed_path: Path,
    allowed: t.Tensor | None = None,
    clusters: FeatureClusterConsolidation | None = None,
) -> None:
    selection = load_selection_shard(selection_path)
    packed = load_packed_row_shard(packed_path)
    assert selection.sel_offsets.dtype == t.int32
    assert selection.sel_fids.dtype == t.uint16
    assert selection.sel_buckets.dtype == t.uint8
    assert selection.sel_pcts.dtype == t.uint8
    assert selection.neg_offsets.dtype == t.int32
    assert selection.neg_fids.dtype == t.uint16
    assert selection.total_segments == packed.total_segments
    assert selection.sel_offsets.numel() == packed.total_segments + 1
    assert selection.neg_offsets.numel() == packed.total_segments + 1
    assert int(selection.sel_offsets[0].item()) == 0
    assert int(selection.neg_offsets[0].item()) == 0
    assert int(selection.sel_offsets[-1].item()) == selection.sel_fids.numel()
    assert int(selection.neg_offsets[-1].item()) == selection.neg_fids.numel()
    assert bool((selection.sel_offsets[1:] >= selection.sel_offsets[:-1]).all())
    assert bool((selection.neg_offsets[1:] >= selection.neg_offsets[:-1]).all())
    assert selection.sel_fids.numel() == selection.sel_buckets.numel()
    assert selection.sel_fids.numel() == selection.sel_pcts.numel()
    assert bool((selection.sel_buckets < 1).all())

    for segment_index in range(packed.total_segments):
        sel_start = int(selection.sel_offsets[segment_index].item())
        sel_end = int(selection.sel_offsets[segment_index + 1].item())
        neg_start = int(selection.neg_offsets[segment_index].item())
        neg_end = int(selection.neg_offsets[segment_index + 1].item())
        selected = selection.sel_fids[sel_start:sel_end].to(t.int64)
        negative = selection.neg_fids[neg_start:neg_end].to(t.int64)
        assert selected.unique().numel() == selected.numel()
        assert negative.unique().numel() == negative.numel()
        assert not bool(t.isin(selected, negative).any())
        if allowed is not None:
            assert bool(allowed[selected].all()) if selected.numel() else True
            assert bool(allowed[negative].all()) if negative.numel() else True
        if clusters is not None:
            clusters.assert_representatives_only(
                selected,
                surface=f"integration positive segment {segment_index}",
            )
            clusters.assert_representatives_only(
                negative,
                surface=f"integration negative segment {segment_index}",
            )


def test_null_and_explicit_one_preserve_complete_legacy_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path)
    default = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    explicit = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=1,
        monkeypatch=monkeypatch,
    )
    assert default == explicit
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 1)
    assert _batching_descriptor(selection_dir) is None


def test_enabled_mode_is_separately_addressed_and_manifest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path)
    legacy = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        monkeypatch=monkeypatch,
    )
    batched = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    dirs = _selection_dirs(env["packed_row_dir"])
    assert len(dirs) == 2
    legacy_dir = _dir_for_batch_size(env["packed_row_dir"], 1)
    batched_dir = _dir_for_batch_size(env["packed_row_dir"], 8)
    assert legacy_dir.name != batched_dir.name
    assert json.loads(legacy["manifest.json"])["checksum"] == legacy_dir.name
    assert json.loads(batched["manifest.json"])["checksum"] == batched_dir.name
    assert _batching_descriptor(batched_dir) == {
        "schema_version": 1,
        "algorithm": "ordinary_execution_runs_v1",
        "batch_size": 8,
    }


def test_enabled_mode_is_exactly_deterministic_across_fresh_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_multisplice_env(tmp_path)
    first = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    second = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def test_enabled_mode_is_byte_exact_across_worker_and_cache_topologies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=5)
    reference = run_selection_main(
        env,
        cache_shards=1,
        sidecar_shards=1,
        num_workers=1,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    parallel = run_selection_main(
        env,
        cache_shards=5,
        sidecar_shards=16,
        num_workers=2,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    assert parallel.keys() == reference.keys()
    for name in reference:
        assert parallel[name] == reference[name], name


def test_enabled_mode_is_byte_exact_through_shared_store_and_spooling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=4)
    cluster_path = write_cluster_yaml(tmp_path)
    spool = tmp_path / "spool"
    shared = tmp_path / "shared"
    spool.mkdir()
    shared.mkdir()
    reference = run_selection_main(
        env,
        cache_shards=3,
        sidecar_shards=8,
        num_workers=1,
        splice_batch_size=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    candidate = run_selection_main(
        env,
        cache_shards=3,
        sidecar_shards=8,
        num_workers=2,
        shared_store=True,
        shared_store_capacity=3,
        shm_dir=str(shared),
        packed_spool_dir=str(spool),
        splice_batch_size=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    assert candidate.keys() == reference.keys()
    for name in reference:
        assert candidate[name] == reference[name], name
    assert list(shared.iterdir()) == []
    assert list(spool.iterdir()) == []


def test_enabled_clustered_resume_reconstructs_exact_full_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=4)
    cluster_path = write_cluster_yaml(tmp_path)
    full = run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")

    real_select = psf.select_splice_counterfactual_features_for_shard
    calls = 0

    def fail_before_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected batching resume interruption")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        fail_before_second,
    )
    with pytest.raises(RuntimeError, match="batching resume interruption"):
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=5,
            splice_batch_size=8,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=cluster_path,
            monkeypatch=monkeypatch,
        )
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 8)
    interrupted = read_manifest(selection_dir)
    assert interrupted is not None
    assert interrupted["metadata"]["feature_clusters"]["status"] == "in_progress"
    certified = sorted(selection_dir.glob("shard_*.safetensors.zst"))
    assert len(certified) == 1
    certified_bytes = certified[0].read_bytes()

    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        real_select,
    )
    resumed = run_selection_main(
        env,
        cache_shards=5,
        sidecar_shards=16,
        splice_batch_size=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    assert resumed.keys() == full.keys()
    for name in full:
        assert resumed[name] == full[name], name
    assert certified[0].read_bytes() == certified_bytes


def test_batch_sizes_never_share_an_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path)
    run_selection_main(
        env,
        splice_batch_size=4,
        monkeypatch=monkeypatch,
    )
    run_selection_main(
        env,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    four = _dir_for_batch_size(env["packed_row_dir"], 4)
    eight = _dir_for_batch_size(env["packed_row_dir"], 8)
    assert four.name != eight.name
    four_manifest = read_manifest(four)
    eight_manifest = read_manifest(eight)
    assert four_manifest is not None
    assert eight_manifest is not None
    assert four_manifest["checksum"] == four.name
    assert eight_manifest["checksum"] == eight.name


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        {
            "schema_version": 1,
            "algorithm": "ordinary_execution_runs_v1",
            "batch_size": 4,
        },
        {
            "schema_version": 1,
            "algorithm": "future_unreviewed_algorithm",
            "batch_size": 8,
        },
    ],
    ids=["missing", "wrong-size", "wrong-algorithm"],
)
def test_manifest_batching_tamper_fails_before_reusing_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any] | None,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=3)
    run_selection_main(
        env,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 8)
    shard_bytes = {
        path.name: path.read_bytes()
        for path in selection_dir.glob("shard_*.safetensors.zst")
    }
    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splice = manifest["metadata"]["splice_counterfactual"]
    if mutation is None:
        splice.pop("selection_batching")
    else:
        splice["selection_batching"] = mutation
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(psf.SpliceProvenanceError, match="selection batching"):
        run_selection_main(
            env,
            splice_batch_size=8,
            monkeypatch=monkeypatch,
        )
    assert {
        path.name: path.read_bytes()
        for path in selection_dir.glob("shard_*.safetensors.zst")
    } == shard_bytes


def test_legacy_manifest_rejects_an_explicit_null_batching_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=3)
    run_selection_main(env, monkeypatch=monkeypatch)
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 1)
    shard_bytes = {
        path.name: path.read_bytes()
        for path in selection_dir.glob("shard_*.safetensors.zst")
    }
    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["splice_counterfactual"]["selection_batching"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(psf.SpliceProvenanceError, match="selection batching"):
        run_selection_main(env, monkeypatch=monkeypatch)
    assert {
        path.name: path.read_bytes()
        for path in selection_dir.glob("shard_*.safetensors.zst")
    } == shard_bytes


def test_real_producer_routes_only_ordinary_rows_through_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_multisplice_env(tmp_path)
    ordinary_batches: list[tuple[int, ...]] = []
    delta_segments = 0
    original_batch = psf._materialize_ordinary_splice_selection_batch
    original_single = psf._materialize_splice_segment_selection
    log_messages: list[str] = []

    def observed_batch(inputs, **kwargs):
        ordinary_batches.append(tuple(item.segment_index for item in inputs))
        return original_batch(inputs, **kwargs)

    def observed_single(*args, **kwargs):
        nonlocal delta_segments
        if kwargs.get("tensor_output") is True:
            assert kwargs.get("delta_result") is not None
            delta_segments += 1
        return original_single(*args, **kwargs)

    monkeypatch.setattr(psf, "_materialize_ordinary_splice_selection_batch", observed_batch)
    monkeypatch.setattr(psf, "_materialize_splice_segment_selection", observed_single)
    sink_id = logger.add(lambda message: log_messages.append(str(message)))
    try:
        run_selection_main(
            env,
            cache_shards=2,
            sidecar_shards=5,
            splice_batch_size=8,
            monkeypatch=monkeypatch,
        )
    finally:
        logger.remove(sink_id)
    packed_segments = sum(
        load_packed_row_shard(path).total_segments
        for path in glob_overlay_shards(env["packed_row_dir"])
    )
    assert ordinary_batches
    assert any(len(indices) > 1 for indices in ordinary_batches)
    assert all(1 <= len(indices) <= 8 for indices in ordinary_batches)
    assert delta_segments >= 2
    assert sum(map(len, ordinary_batches)) + delta_segments == packed_segments

    shard_profiles = []
    for message in log_messages:
        marker = "SPLICE_RESOURCE_PROFILE "
        if marker not in message:
            continue
        profile = json.loads(message.split(marker, 1)[1])
        if profile.get("event") == "shard":
            shard_profiles.append(profile)
    assert shard_profiles
    assert sum(
        profile["selection_batching"]["ordinary_segments"]
        for profile in shard_profiles
    ) == sum(map(len, ordinary_batches))
    assert sum(
        profile["selection_batching"]["serial_delta_segments"]
        for profile in shard_profiles
    ) == delta_segments
    assert all(
        profile["selection_batching"]["configured_batch_size"] == 8
        and profile["selection_batching"]["algorithm"]
        == "ordinary_execution_runs_v1"
        and profile["selection_batching"]["tensor_native_csr"] is True
        and profile["selection_batching"]["maximum_ordinary_batch_size"] <= 8
        for profile in shard_profiles
    )


def test_real_producer_flushes_before_percentile_representation_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=4)
    real_validate = psf._validate_packed_segment_matches_feature_index
    real_batch = psf._materialize_ordinary_splice_selection_batch
    validated = 0
    observed_batches: list[tuple[bool, int]] = []

    def alternating_percentile_representation(*args, **kwargs):
        nonlocal validated
        fids, mags, pcts = real_validate(*args, **kwargs)
        assert pcts is not None
        keep_overlay = validated % 2 == 0
        validated += 1
        return fids, mags, pcts if keep_overlay else None

    def observed_batch(inputs, **kwargs):
        representations = {item.raw_pcts is not None for item in inputs}
        assert len(representations) == 1
        observed_batches.append((representations.pop(), len(inputs)))
        return real_batch(inputs, **kwargs)

    monkeypatch.setattr(
        psf,
        "_validate_packed_segment_matches_feature_index",
        alternating_percentile_representation,
    )
    monkeypatch.setattr(
        psf,
        "_materialize_ordinary_splice_selection_batch",
        observed_batch,
    )
    run_selection_main(
        env,
        cache_shards=2,
        sidecar_shards=5,
        splice_batch_size=8,
        monkeypatch=monkeypatch,
    )
    assert validated > 2
    assert {representation for representation, _size in observed_batches} == {
        False,
        True,
    }
    assert any(
        left_representation != right_representation
        for (left_representation, _left_size), (
            right_representation,
            _right_size,
        ) in itertools.pairwise(observed_batches)
    )


def test_tensor_native_hot_path_contains_no_payload_tolist_roundtrip() -> None:
    for function in (
        psf._materialize_ordinary_splice_selection_batch,
        psf._tensor_selection_batch,
        psf._TensorSelectionShardBuilder.build,
    ):
        source = inspect.getsource(function)
        assert ".tolist(" not in source, function.__qualname__


def test_clustered_batched_output_satisfies_every_serialized_invariant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_interleaved_env(tmp_path)
    cluster_path = write_cluster_yaml(tmp_path)
    run_selection_main(
        env,
        cache_shards=5,
        sidecar_shards=16,
        splice_batch_size=8,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 8)
    mask = FeatureMask(
        FeatureMaskMode.whitelist,
        tuple(int(value) for value in WHITELIST_MASK_BELOW_150["ids"]),
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": cluster_path}},
        4096,
    )
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    negative_mask = clusters.effective_negative_mask()
    assert negative_mask is not None
    allowed = negative_mask.allowed_universe_mask(4096)
    packed_by_name = {
        path.name: path for path in glob_overlay_shards(env["packed_row_dir"])
    }
    selection_paths = sorted(selection_dir.glob("shard_*.safetensors.zst"))
    assert selection_paths
    for selection_path in selection_paths:
        _assert_selection_shard_semantics(
            selection_path=selection_path,
            packed_path=packed_by_name[selection_path.name],
            allowed=allowed,
            clusters=clusters,
        )

    manifest = read_manifest(selection_dir)
    assert manifest is not None
    splice = manifest["metadata"]["splice_counterfactual"]
    assert splice["status"] == "complete"
    assert splice["total_packed_segments_selected"] == sum(
        load_packed_row_shard(path).total_segments for path in packed_by_name.values()
    )


@pytest.mark.parametrize("channel", ["positive", "negative"])
@pytest.mark.parametrize(
    ("feature_id", "violation"),
    [(150, "mask-disallowed"), (42, "consolidated-away")],
)
def test_clustered_ordinary_output_is_rejected_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    feature_id: int,
    violation: str,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=3)
    cluster_path = write_cluster_yaml(tmp_path)
    poisoned = False

    def poison(ids: t.Tensor) -> None:
        nonlocal poisoned
        if not poisoned and ids.numel() > 0:
            ids[0] = feature_id
            poisoned = True

    if channel == "negative":
        real_sample = psf.sample_inactive_features_batched

        def poison_negative_output(*args, **kwargs):
            rows = real_sample(*args, **kwargs)
            for ids in rows:
                poison(ids)
            return rows

        # A bad internal result reaches the formerly repeated row guards.
        monkeypatch.setattr(psf, "sample_inactive_features_batched", poison_negative_output)
    else:
        real_materialize = psf._materialize_ordinary_splice_selection_batch

        def poison_positive_output(*args, **kwargs):
            batch = real_materialize(*args, **kwargs)
            poison(batch.sel_fids)
            return batch

        # Poison after candidate-percentile lookup, which would otherwise
        # reject an ID that cannot occur among the valid positive candidates.
        monkeypatch.setattr(
            psf,
            "_materialize_ordinary_splice_selection_batch",
            poison_positive_output,
        )

    with pytest.raises(
        FeatureClustersError,
        match=rf"serialized clustered selection shard .* {channel} selection "
        rf"contains {violation} feature ID {feature_id}",
    ):
        run_selection_main(
            env,
            splice_batch_size=8,
            mask=WHITELIST_MASK_BELOW_150,
            clusters_path=cluster_path,
            monkeypatch=monkeypatch,
        )

    assert poisoned
    selection_dir = _dir_for_batch_size(env["packed_row_dir"], 8)
    assert list(selection_dir.glob("shard_*.safetensors.zst"))
    manifest = read_manifest(selection_dir)
    assert manifest is not None
    metadata = manifest["metadata"]
    assert metadata["feature_clusters"]["status"] == "in_progress"
    assert metadata["splice_counterfactual"]["status"] == "in_progress"
    assert psf.CLUSTERED_SELECTION_VALIDATION_KEY not in metadata
    assert psf.REMOTE_LOADER_KEY not in metadata
