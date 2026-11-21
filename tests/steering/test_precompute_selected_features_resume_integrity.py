"""Resume-integrity regressions for precompute_selected_features."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.preprocess.precompute_selected_features as psf
import steering.overlay_io as overlay_io
from errors import FeatureClustersError, GradientAttributionError
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
)
from steering.conditional.feature_clusters import (
    build_clustered_selection_shard_integrity,
)
from steering.conditional.splice_counterfactual import (
    SpliceProvenanceError,
    resolve_packed_row_fingerprint_strict,
)
from steering.feature_io import save_features
from steering.gradient_overlay import (
    GRAD_DOC_SEG_OFFSETS_KEY,
    GRAD_OFFSETS_KEY,
    GRAD_SCORES_KEY,
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    canonical_gradient_descriptor,
    compute_gradient_overlay_checksum,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.stats_io import save_statistics
from tests.steering.test_selection_manifest_feature_mask import (
    _make_production_extraction_stats,
)

_FROZEN_CREATED_AT = "2026-01-01T00:00:00Z"
_N_SAE_FEATURES = 64
_N_SEGMENTS = 6
_FEATURES_PER_SEGMENT = 8
_BASE_STEM = "shard_00000"


def _sha256_tree(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _bytes_tree(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _selection_dir(packed_dir: Path) -> Path:
    directories = sorted(path for path in (packed_dir / "sel").iterdir() if path.is_dir())
    assert len(directories) == 1
    return directories[0]


def _interrupt_after_first_shard(
    env: dict[str, object], monkeypatch
) -> tuple[Path, int]:
    real_select = psf.select_features_for_shard
    calls = 0

    def _interrupting_select(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected selection crash")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(psf, "select_features_for_shard", _interrupting_select)
    with pytest.raises(RuntimeError, match="injected selection crash"):
        psf.main.__wrapped__(env["cfg"])
    monkeypatch.setattr(psf, "select_features_for_shard", real_select)
    return _selection_dir(env["packed_dir"]), calls  # ty: ignore[invalid-argument-type]


def _gradient_cfg(feature_dir: Path, packed_dir: Path):
    return OmegaConf.create(
        {
            "packed_row_dir": str(packed_dir),
            "feature_dir": str(feature_dir),
            "segment_size": 128,
            "feature_selection": {
                "distribution": "geometric",
                "mean": 4.0,
                "std": 4.0,
                "min_k": 1,
                "max_k": 7,
                "mode": "top_k_percentile",
                "gradient": {
                    "mode": "future_half_split",
                    "model_dtype": "float32",
                    "min_final_segment_tokens": 4,
                },
            },
            "negative_selection": None,
            "feature_delta": None,
            "feature_mask_freq_min": None,
            "feature_mask_freq_max": None,
            "selection_resume": True,
            "num_buckets": 4,
            "seed": 17,
            "sae": {
                "release": "gemmascope",
                "model": "gemma-2-2b",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
                "l0": None,
            },
        }
    )


def _ordinary_cfg(
    feature_dir: Path,
    packed_dir: Path,
    *,
    feature_selection: dict[str, object],
    negative_selection: dict[str, object] | None,
):
    return OmegaConf.create(
        {
            "packed_row_dir": str(packed_dir),
            "feature_dir": str(feature_dir),
            "segment_size": 128,
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
            "feature_delta": None,
            "feature_mask_freq_min": None,
            "feature_mask_freq_max": None,
            "selection_resume": True,
            "num_buckets": 4,
            "seed": 29,
        }
    )


@pytest.fixture()
def gradient_resume_env(tmp_path: Path, monkeypatch) -> dict[str, object]:
    """Three packed shards plus an already-existing, descriptor-valid grad overlay.

    The fixture materializes the overlay directly in tests because PR3 deliberately
    forbids producing new gradient shards without a token-stream witness.
    """
    feature_dir = tmp_path / "s128" / "features"
    feature_dir.mkdir(parents=True)
    save_statistics(
        _make_production_extraction_stats(_N_SAE_FEATURES),
        feature_dir / "feature_statistics",
    )

    offsets = t.arange(
        0,
        (_N_SEGMENTS + 1) * _FEATURES_PER_SEGMENT,
        _FEATURES_PER_SEGMENT,
        dtype=t.int32,
    )
    feature_ids = t.arange(
        _N_SEGMENTS * _FEATURES_PER_SEGMENT, dtype=t.int64
    )
    base_path = save_features(
        {
            "offsets": offsets,
            "feature_ids": feature_ids,
            "feature_mags": t.linspace(0.5, 2.0, len(feature_ids)),
            "n_docs": 1,
            "n_segments": _N_SEGMENTS,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "ignore_bos_features": False,
        },
        feature_dir / _BASE_STEM,
    )

    percentile_checksum = hashlib.sha256(b"resume-integrity-pct").hexdigest()
    percentile_dir = overlay_dir(
        feature_dir, OverlayKind.PERCENTILE, percentile_checksum
    )
    save_overlay_shard(
        {"feature_percentiles": t.arange(1, len(feature_ids) + 1, dtype=t.uint8)},
        percentile_dir / f"{_BASE_STEM}.safetensors.zst",
    )
    write_manifest(
        percentile_dir,
        percentile_checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=[base_path],
    )

    descriptor = canonical_gradient_descriptor(
        attribution_mode="future_half_split",
        model="gemma-2-2b",
        model_dtype="float32",
        layer=8,
        sae_release="gemmascope",
        sae_width="16k",
        sae_l0=None,
        activation_type="residual",
        ignore_bos_features=False,
        min_final_segment_tokens=4,
        percentile_checksum=percentile_checksum,
    )
    gradient_checksum = compute_gradient_overlay_checksum(descriptor)
    gradient_dir = overlay_dir(feature_dir, OverlayKind.GRADIENT, gradient_checksum)
    save_overlay_shard(
        {
            GRAD_SCORES_KEY: (
                (t.arange(len(feature_ids), dtype=t.float32) % 11) + 1
            ).to(t.float16),
            GRAD_OFFSETS_KEY: offsets.to(t.int64),
            GRAD_DOC_SEG_OFFSETS_KEY: t.tensor([0, _N_SEGMENTS], dtype=t.int32),
        },
        gradient_dir / f"{_BASE_STEM}.safetensors.zst",
    )
    write_manifest(
        gradient_dir,
        gradient_checksum,
        OverlayKind.GRADIENT,
        metadata=descriptor,
        base_shard_paths=[base_path],
    )

    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    packed = PackedRowShard(
        tokens=t.arange(34, dtype=t.int64).to(t.uint16).reshape(2, 17),
        row_seg_offsets=t.tensor([0, 3, 6], dtype=t.int32),
        seg_token_start=t.tensor([0, 5, 10, 0, 5, 10], dtype=t.uint16),
        seg_token_end=t.tensor([5, 10, 16, 5, 10, 16], dtype=t.uint16),
        seg_doc_id=t.tensor([0, 0, 0, 1, 1, 1], dtype=t.uint8),
        seg_feat_offsets=offsets,
        feature_ids=feature_ids.to(t.uint16),
        feature_pcts=(
            (t.arange(len(feature_ids), dtype=t.int64) * 29 + 31) % 255
        ).add(1).to(t.uint8),
    )
    prov_dir = provenance_dir(packed_dir)
    for shard_idx in range(3):
        stem = f"shard_{shard_idx:05d}"
        save_packed_row_shard(packed, packed_dir / stem)
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.zeros(_N_SEGMENTS, dtype=t.int32),
                PROVENANCE_DOC_IDX_KEY: t.zeros(_N_SEGMENTS, dtype=t.int32),
                PROVENANCE_SEG_ORDINAL_KEY: t.arange(_N_SEGMENTS, dtype=t.int32),
            },
            prov_dir / f"{stem}.safetensors.zst",
        )
    write_manifest(
        packed_dir,
        "packed-resume-integrity",
        OverlayKind.PACKED_ROW,
        metadata={
            "base_shard_stems": [_BASE_STEM],
            "provenance_shards": len(glob_overlay_shards(prov_dir)),
        },
        base_shard_paths=[base_path],
    )

    monkeypatch.setattr(
        overlay_io.time,
        "strftime",
        lambda *_args, **_kwargs: _FROZEN_CREATED_AT,
    )
    return {
        "cfg": _gradient_cfg(feature_dir, packed_dir),
        "feature_dir": feature_dir,
        "gradient_checksum": gradient_checksum,
        "gradient_dir": gradient_dir,
        "packed_dir": packed_dir,
    }


def test_gradient_fresh_run_matches_pre_pr4_sha256(gradient_resume_env) -> None:
    """Healthy fresh output stays byte-identical to unmodified origin/main."""
    psf.main.__wrapped__(gradient_resume_env["cfg"])
    actual = _sha256_tree(_selection_dir(gradient_resume_env["packed_dir"]))
    assert actual == {
        "manifest.json": (
            "a57da78f7b5f8ad88f2dc48657687c55fefc2c690a56dee9c97ce7bbeb5eb3da"
        ),
        "shard_00000.safetensors.zst": (
            "4bd2ac635cc79cb49a54f4a7837aab3bc2c1a5055f2b21078fb7ed35afde3f2c"
        ),
        "shard_00001.safetensors.zst": (
            "5eb6cd3befe5ff97a9a82fcdc214e8db14656ce4ed7da7e8cff9a0d8d095a29c"
        ),
        "shard_00002.safetensors.zst": (
            "202820d39076d1bfdcff86cc95a15fd9f895ef54fa3d3f00dc9975749ab43ce6"
        ),
    }


def test_gradient_partial_run_publishes_overlay_identity_before_first_shard(
    gradient_resume_env, monkeypatch
) -> None:
    sel_dir, calls = _interrupt_after_first_shard(gradient_resume_env, monkeypatch)

    assert calls == 2
    assert [path.name for path in glob_overlay_shards(sel_dir)] == [
        "shard_00000.safetensors.zst"
    ]
    manifest = read_manifest(sel_dir)
    assert manifest is not None
    assert manifest["metadata"]["gradient_attribution"]["overlay_checksum"] == (
        gradient_resume_env["gradient_checksum"]
    )


def test_gradient_partial_resume_rejects_changed_overlay_before_mutation(
    gradient_resume_env, monkeypatch
) -> None:
    sel_dir, _ = _interrupt_after_first_shard(gradient_resume_env, monkeypatch)
    before = _bytes_tree(sel_dir)
    calls = {"select": 0, "sweep": 0, "write_manifest": 0}

    real_resolve = psf.resolve_gradient_artifacts
    real_select = psf.select_features_for_shard
    real_sweep = psf._sweep_dead_atomic_selection_temps
    real_write_manifest = psf.write_manifest

    def _resolve_overlay_b(*args: Any, **kwargs: Any):
        joiner, prov_dir, _ = real_resolve(*args, **kwargs)
        return joiner, prov_dir, "B" * 64

    def _count_select(*args: Any, **kwargs: Any) -> bool:
        calls["select"] += 1
        return real_select(*args, **kwargs)

    def _count_sweep(*args: Any, **kwargs: Any):
        calls["sweep"] += 1
        return real_sweep(*args, **kwargs)

    def _count_write(*args: Any, **kwargs: Any):
        calls["write_manifest"] += 1
        return real_write_manifest(*args, **kwargs)

    monkeypatch.setattr(psf, "resolve_gradient_artifacts", _resolve_overlay_b)
    monkeypatch.setattr(psf, "select_features_for_shard", _count_select)
    monkeypatch.setattr(psf, "_sweep_dead_atomic_selection_temps", _count_sweep)
    monkeypatch.setattr(psf, "write_manifest", _count_write)

    with pytest.raises(GradientAttributionError, match="cannot be authenticated|stale"):
        psf.main.__wrapped__(gradient_resume_env["cfg"])

    assert calls == {"select": 0, "sweep": 0, "write_manifest": 0}
    assert _bytes_tree(sel_dir) == before


@pytest.mark.parametrize("manifest_state", ["missing", "invalid"])
def test_gradient_shards_without_valid_manifest_are_rejected_before_mutation(
    gradient_resume_env, monkeypatch, manifest_state: str
) -> None:
    psf.main.__wrapped__(gradient_resume_env["cfg"])
    sel_dir = _selection_dir(gradient_resume_env["packed_dir"])
    manifest_path = sel_dir / "manifest.json"
    if manifest_state == "missing":
        manifest_path.unlink()
    else:
        manifest_path.write_text("not-json\n")
    before = _bytes_tree(sel_dir)
    calls = {"select": 0, "sweep": 0, "write_manifest": 0}

    real_select = psf.select_features_for_shard
    real_sweep = psf._sweep_dead_atomic_selection_temps
    real_write_manifest = psf.write_manifest

    def _count_select(*args: Any, **kwargs: Any) -> bool:
        calls["select"] += 1
        return real_select(*args, **kwargs)

    def _count_sweep(*args: Any, **kwargs: Any):
        calls["sweep"] += 1
        return real_sweep(*args, **kwargs)

    def _count_write(*args: Any, **kwargs: Any):
        calls["write_manifest"] += 1
        return real_write_manifest(*args, **kwargs)

    monkeypatch.setattr(psf, "select_features_for_shard", _count_select)
    monkeypatch.setattr(psf, "_sweep_dead_atomic_selection_temps", _count_sweep)
    monkeypatch.setattr(psf, "write_manifest", _count_write)

    with pytest.raises(GradientAttributionError, match="cannot be authenticated"):
        psf.main.__wrapped__(gradient_resume_env["cfg"])

    assert calls == {"select": 0, "sweep": 0, "write_manifest": 0}
    assert _bytes_tree(sel_dir) == before


def test_complete_gradient_resume_skips_all_payload_io_and_preserves_exact_tree(
    gradient_resume_env, monkeypatch
) -> None:
    psf.main.__wrapped__(gradient_resume_env["cfg"])
    sel_dir = _selection_dir(gradient_resume_env["packed_dir"])
    before = _bytes_tree(sel_dir)
    calls = {"overlay_load": 0, "gradient_load": 0, "select": 0}

    real_overlay_load = psf.load_overlay_shard
    real_gradient_load = psf.GradScoreJoiner._load
    real_select = psf.select_features_for_shard

    def _count_overlay_load(*args: Any, **kwargs: Any):
        calls["overlay_load"] += 1
        return real_overlay_load(*args, **kwargs)

    def _count_gradient_load(self, base_idx: int):
        calls["gradient_load"] += 1
        return real_gradient_load(self, base_idx)

    def _count_select(*args: Any, **kwargs: Any) -> bool:
        calls["select"] += 1
        return real_select(*args, **kwargs)

    monkeypatch.setattr(psf, "load_overlay_shard", _count_overlay_load)
    monkeypatch.setattr(psf.GradScoreJoiner, "_load", _count_gradient_load)
    monkeypatch.setattr(psf, "select_features_for_shard", _count_select)

    psf.main.__wrapped__(gradient_resume_env["cfg"])

    assert calls == {"overlay_load": 0, "gradient_load": 0, "select": 0}
    assert _bytes_tree(sel_dir) == before


def test_gradient_partial_resume_builds_joins_for_rng_replay_and_restores_bytes(
    gradient_resume_env, monkeypatch
) -> None:
    psf.main.__wrapped__(gradient_resume_env["cfg"])
    sel_dir = _selection_dir(gradient_resume_env["packed_dir"])
    fresh = _bytes_tree(sel_dir)
    (sel_dir / "shard_00002.safetensors.zst").unlink()
    calls = {"join": 0, "overlay_load": 0, "gradient_load": 0}

    real_join = psf.ShardGradientJoin
    real_overlay_load = psf.load_overlay_shard
    real_gradient_load = psf.GradScoreJoiner._load

    def _count_join(*args: Any, **kwargs: Any):
        calls["join"] += 1
        return real_join(*args, **kwargs)

    def _count_overlay_load(*args: Any, **kwargs: Any):
        calls["overlay_load"] += 1
        return real_overlay_load(*args, **kwargs)

    def _count_gradient_load(self, base_idx: int):
        calls["gradient_load"] += 1
        return real_gradient_load(self, base_idx)

    monkeypatch.setattr(psf, "ShardGradientJoin", _count_join)
    monkeypatch.setattr(psf, "load_overlay_shard", _count_overlay_load)
    monkeypatch.setattr(psf.GradScoreJoiner, "_load", _count_gradient_load)

    psf.main.__wrapped__(gradient_resume_env["cfg"])

    assert calls == {"join": 3, "overlay_load": 4, "gradient_load": 1}
    assert _bytes_tree(sel_dir) == fresh


@pytest.mark.parametrize(
    ("feature_selection", "negative_selection", "consumes_rng"),
    [
        pytest.param(
            {
                "distribution": "geometric",
                "mean": 4.0,
                "std": 4.0,
                "min_k": 1,
                "max_k": 7,
                "mode": "top_k_percentile",
            },
            None,
            True,
            id="positive-sampling-only",
        ),
        pytest.param(
            {"threshold_type": "feature_percentile", "threshold": 0.5},
            {
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
                "min_k": 0,
                "max_k": 6,
            },
            True,
            id="threshold-plus-negative-sampling",
        ),
        pytest.param(
            {
                "distribution": "geometric",
                "mean": 4.0,
                "std": 4.0,
                "min_k": 1,
                "max_k": 7,
                "mode": "top_k_percentile",
            },
            {
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
                "min_k": 0,
                "max_k": 6,
            },
            True,
            id="shared-rng-positive-and-negative",
        ),
        pytest.param(
            {"threshold_type": "feature_percentile", "threshold": 0.5},
            None,
            False,
            id="threshold-only-rng-free-control",
        ),
    ],
)
def test_ordinary_partial_resume_replays_rng_and_restores_exact_bytes(
    gradient_resume_env,
    monkeypatch,
    feature_selection: dict[str, object],
    negative_selection: dict[str, object] | None,
    consumes_rng: bool,
) -> None:
    packed_dir = gradient_resume_env["packed_dir"]
    cfg = _ordinary_cfg(
        gradient_resume_env["feature_dir"],
        packed_dir,
        feature_selection=feature_selection,
        negative_selection=negative_selection,
    )
    psf.main.__wrapped__(cfg)
    sel_dir = _selection_dir(packed_dir)
    fresh = _bytes_tree(sel_dir)
    tail_name = "shard_00002.safetensors.zst"

    real_select = psf.select_features_for_shard
    real_consumes_rng = psf._pipeline_consumes_selection_rng
    calls: list[tuple[str, bool]] = []

    def _record_select(
        base_shard_path: Path,
        pipeline_config,
        output_dir: Path,
        shard_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        calls.append((shard_name, Path(output_dir) == sel_dir))
        return real_select(
            base_shard_path,
            pipeline_config,
            output_dir,
            shard_name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(psf, "select_features_for_shard", _record_select)

    if consumes_rng:
        (sel_dir / tail_name).unlink()
        (sel_dir / "manifest.json").unlink()
        monkeypatch.setattr(psf, "_pipeline_consumes_selection_rng", lambda _cfg: False)
        psf.main.__wrapped__(cfg)
        assert calls == [("shard_00002", True)]
        assert (sel_dir / tail_name).read_bytes() != fresh[tail_name]

        calls.clear()
        (sel_dir / tail_name).unlink()
        (sel_dir / "manifest.json").unlink()
        monkeypatch.setattr(
            psf, "_pipeline_consumes_selection_rng", real_consumes_rng
        )
    else:
        (sel_dir / tail_name).unlink()
        (sel_dir / "manifest.json").unlink()

    psf.main.__wrapped__(cfg)

    expected_calls = (
        [
            ("shard_00000", False),
            ("shard_00001", False),
            ("shard_00002", True),
        ]
        if consumes_rng
        else [("shard_00002", True)]
    )
    assert calls == expected_calls
    assert _bytes_tree(sel_dir) == fresh


def _delta_free_clustered_cfg(
    env: dict[str, object],
    tmp_path: Path,
):
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": [[0, 1], [8, 9]],
            }
        ),
        encoding="utf-8",
    )
    cfg = _ordinary_cfg(
        env["feature_dir"],  # ty: ignore[invalid-argument-type]
        env["packed_dir"],  # ty: ignore[invalid-argument-type]
        feature_selection={
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
            "min_k": 1,
            "max_k": 7,
            "mode": "top_k_percentile",
        },
        negative_selection={
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
            "min_k": 0,
            "max_k": 6,
        },
    )
    cfg.feature_clusters = {"path": str(cluster_path)}
    return cfg


def test_delta_free_clustered_resume_reissues_certificate_without_shard_rewrite(
    gradient_resume_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _delta_free_clustered_cfg(gradient_resume_env, tmp_path)

    real_select = psf.select_features_for_shard
    calls = 0

    def _interrupting_select(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected clustered selection crash")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(psf, "select_features_for_shard", _interrupting_select)
    with pytest.raises(RuntimeError, match="injected clustered selection crash"):
        psf.main.__wrapped__(cfg)
    monkeypatch.setattr(psf, "select_features_for_shard", real_select)

    selection_dir = _selection_dir(gradient_resume_env["packed_dir"])
    partial = read_manifest(selection_dir)
    assert partial is not None
    assert partial["metadata"]["feature_clusters"]["status"] == "in_progress"
    assert (
        CLUSTERED_SELECTION_VALIDATION_KEY
        not in partial["metadata"]
    )

    psf.main.__wrapped__(cfg)
    completed = read_manifest(selection_dir)
    assert completed is not None
    assert completed["metadata"]["feature_clusters"]["status"] == "complete"
    assert completed["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "status"
    ] == "complete"
    shard_bytes = {
        path.name: path.read_bytes()
        for path in glob_overlay_shards(selection_dir)
    }

    manifest_path = selection_dir / "manifest.json"
    uncertified = json.loads(manifest_path.read_text(encoding="utf-8"))
    uncertified["metadata"].pop(CLUSTERED_SELECTION_VALIDATION_KEY)
    manifest_path.write_text(
        json.dumps(uncertified, sort_keys=True),
        encoding="utf-8",
    )

    psf.main.__wrapped__(cfg)
    reissued = read_manifest(selection_dir)
    assert reissued is not None
    assert CLUSTERED_SELECTION_VALIDATION_KEY in reissued["metadata"]
    assert {
        path.name: path.read_bytes()
        for path in glob_overlay_shards(selection_dir)
    } == shard_bytes

    corrupt = json.loads(manifest_path.read_text(encoding="utf-8"))
    corrupt["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "aggregate_sha256"
    ] = "0" * 64
    manifest_path.write_text(
        json.dumps(corrupt, sort_keys=True),
        encoding="utf-8",
    )
    cfg.selection_resume = False

    psf.main.__wrapped__(cfg)

    repaired = read_manifest(selection_dir)
    assert repaired is not None
    assert repaired["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "aggregate_sha256"
    ] != "0" * 64
    assert {
        path.name: path.read_bytes()
        for path in glob_overlay_shards(selection_dir)
    } == shard_bytes


def test_delta_free_clustered_producer_populates_cold_cache(
    gradient_resume_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _delta_free_clustered_cfg(gradient_resume_env, tmp_path)
    packed_dir = gradient_resume_env["packed_dir"]
    cache_path = packed_dir / ".packed-row-fingerprint-cache.json"
    real_fingerprint = psf.packed_row_fingerprint

    def fingerprint_then_remove_cache(*args: Any, **kwargs: Any) -> str:
        value = real_fingerprint(*args, **kwargs)
        cache_path.unlink(missing_ok=True)
        return value

    monkeypatch.setattr(
        psf,
        "packed_row_fingerprint",
        fingerprint_then_remove_cache,
    )

    psf.main.__wrapped__(cfg)

    manifest = read_manifest(_selection_dir(packed_dir))
    assert cache_path.is_file()
    assert manifest is not None
    assert manifest["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "status"
    ] == "complete"


def test_delta_free_clustered_producer_rejects_incomplete_validation_domain(
    gradient_resume_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _delta_free_clustered_cfg(gradient_resume_env, tmp_path)
    packed_dir = gradient_resume_env["packed_dir"]
    real_validate = psf._validate_clustered_selection_outputs

    def omit_one_validated_stem(*args: Any, **kwargs: Any) -> int:
        result = real_validate(*args, **kwargs)
        segment_counts = kwargs["segment_counts_by_stem"]
        segment_counts.pop(next(iter(segment_counts)))
        return result

    monkeypatch.setattr(
        psf,
        "_validate_clustered_selection_outputs",
        omit_one_validated_stem,
    )

    with pytest.raises(
        FeatureClustersError,
        match="semantic-validation segment counts",
    ):
        psf.main.__wrapped__(cfg)

    manifest = read_manifest(_selection_dir(packed_dir))
    assert manifest is not None
    assert manifest["metadata"]["feature_clusters"]["status"] == "in_progress"
    assert CLUSTERED_SELECTION_VALIDATION_KEY not in manifest["metadata"]


class _AcceptingClusterConsolidation:
    class _Resolved:
        @staticmethod
        def descriptor_dict() -> dict[str, Any]:
            return {
                "schema_version": 1,
                "n_sae_features": 8,
                "content_sha256": "d" * 64,
                "semantics": "post_mask_post_delta_max_percentile_v1",
            }

    resolved = _Resolved()

    def assert_selection_shard(self, **_kwargs: Any) -> None:
        pass


def _write_clustered_finalization_pair(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    packed_dir = tmp_path / "packed_finalization"
    selection_dir = tmp_path / "selection_finalization"
    packed_dir.mkdir()
    selection_dir.mkdir()
    packed_path = save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([2], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
            feature_ids=t.tensor([0], dtype=t.uint16),
            feature_pcts=t.tensor([200], dtype=t.uint8),
        ),
        packed_dir / "shard_00000",
    )
    selection_path = save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 1], dtype=t.int32),
            sel_fids=t.tensor([0], dtype=t.uint16),
            sel_buckets=t.tensor([0], dtype=t.uint8),
            sel_pcts=t.tensor([200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 0], dtype=t.int32),
            neg_fids=t.empty(0, dtype=t.uint16),
        ),
        selection_dir / "shard_00000",
    )
    integrity = build_clustered_selection_shard_integrity([selection_path])
    return packed_path, selection_path, integrity


def test_delta_free_finalization_rejects_selection_swap_after_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    invalid_source = tmp_path / "replacement-selection"
    invalid_source.write_bytes(selection_path.read_bytes()[:16])
    real_load = psf.load_selection_shard

    def load_then_swap(path: Path):
        loaded = real_load(path)
        os.replace(invalid_source, path)
        return loaded

    monkeypatch.setattr(psf, "load_selection_shard", load_then_swap)
    with pytest.raises(FeatureClustersError, match="changed while it was loaded"):
        psf._validate_clustered_selection_outputs(
            output_dir=selection_path.parent,
            base_shard_paths=[packed_path],
            packed_resolution=proof,
            expected_selection_integrity=integrity,
            expected_packed_rows_hash=proof.aggregate_sha256,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            effective_buckets=1,
        )


def test_delta_free_finalization_rejects_packed_swap_after_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    replacement = tmp_path / "replacement-packed"
    replacement.write_bytes(packed_path.read_bytes())
    real_load = psf.load_packed_row_shard

    def load_then_swap(path: Path):
        loaded = real_load(path)
        os.replace(replacement, path)
        return loaded

    monkeypatch.setattr(psf, "load_packed_row_shard", load_then_swap)
    with pytest.raises(
        SpliceProvenanceError,
        match="changed after its strict fingerprint",
    ):
        psf._validate_clustered_selection_outputs(
            output_dir=selection_path.parent,
            base_shard_paths=[packed_path],
            packed_resolution=proof,
            expected_selection_integrity=integrity,
            expected_packed_rows_hash=proof.aggregate_sha256,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            effective_buckets=1,
        )


@pytest.mark.parametrize(
    "failure",
    ["packed_aggregate", "packed_domain", "selection_integrity"],
)
def test_delta_free_finalization_rejects_cross_proof_disagreement(
    tmp_path: Path,
    failure: str,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    expected_packed = proof.aggregate_sha256
    expected_integrity = integrity
    if failure == "packed_aggregate":
        expected_packed = "0" * 64
    elif failure == "packed_domain":
        proof = replace(proof, files=())
    else:
        expected_integrity = {}

    with pytest.raises(FeatureClustersError):
        psf._validate_clustered_selection_outputs(
            output_dir=selection_path.parent,
            base_shard_paths=[packed_path],
            packed_resolution=proof,
            expected_selection_integrity=expected_integrity,
            expected_packed_rows_hash=expected_packed,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            effective_buckets=1,
        )


@pytest.mark.parametrize(
    "mutation",
    ["selection_inventory", "packed_inventory", "selection_signature"],
)
def test_completed_clustered_evidence_final_fence_rejects_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    packed_path, selection_path, _integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    evidence = psf._CompletedClusteredValidationEvidence(
        certificate={},
        selection_signatures=(
            (
                selection_path,
                psf._selection_shard_file_identity(
                    selection_path,
                    surface="test completed evidence",
                ),
            ),
        ),
        packed_resolution=proof,
    )
    if mutation == "selection_inventory":
        (selection_path.parent / "shard_99999.safetensors.zst").write_bytes(
            selection_path.read_bytes()
        )
    elif mutation == "packed_inventory":
        (packed_path.parent / "shard_99999.safetensors.zst").write_bytes(
            packed_path.read_bytes()
        )
    else:
        selection_path.write_bytes(selection_path.read_bytes() + b"changed")

    with pytest.raises(FeatureClustersError):
        psf._assert_completed_clustered_validation_evidence_current(
            evidence,
            output_dir=selection_path.parent,
            packed_row_dir=packed_path.parent,
        )


def _build_completed_validation_evidence(
    *,
    packed_path: Path,
    selection_path: Path,
    integrity: dict[str, Any],
    expected_segments: int = 1,
    prior_metadata: dict[str, Any] | None = None,
) -> psf._CompletedClusteredValidationEvidence:
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    return psf._build_completed_clustered_selection_validation(
        output_dir=selection_path.parent,
        packed_proof=proof,
        selection_hash="c" * 64,
        selection_integrity=integrity,
        packed_rows_hash=proof.aggregate_sha256,
        feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
        feature_delta=None,
        effective_buckets=1,
        expected_segments_by_stem={"shard_00000": expected_segments},
        validated_max_features_by_stem={"shard_00000": 1},
        prior_metadata=prior_metadata,
    )


@pytest.mark.parametrize("has_resume_metadata", [False, True])
def test_splice_resume_rechecks_inventory_from_its_manifest_read(
    tmp_path: Path,
    has_resume_metadata: bool,
) -> None:
    _packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    output_dir = selection_path.parent
    metadata = {psf.SELECTION_SHARD_INTEGRITY_KEY: integrity}
    certified_names = psf.validated_clustered_selection_shard_names(
        output_dir, metadata, surface="initial resume validation", allow_untracked=True
    )
    assert certified_names == frozenset({selection_path.name})
    # Replace the inventory after the earlier byte-validation boundary.
    corrupted = {**integrity, "aggregate_sha256": "0" * 64}
    metadata[psf.SELECTION_SHARD_INTEGRITY_KEY] = corrupted
    if has_resume_metadata:
        metadata[psf.SPLICE_RESUME_METADATA_KEY] = {}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"checksum": "c" * 64, "metadata": metadata}))
    manifest_bytes = manifest_path.read_bytes()
    selection_bytes = selection_path.read_bytes()

    with pytest.raises(FeatureClustersError, match="aggregate mismatch"):
        psf._load_splice_resume_state(
            output_dir,
            certified_existing_stems={
                selection_path.name.removesuffix(".safetensors.zst")
            },
            packed_rows_hash="a" * 64,
            effective_buckets=1,
        )

    assert manifest_path.read_bytes() == manifest_bytes
    assert selection_path.read_bytes() == selection_bytes


def test_completed_clustered_validation_requires_exact_segment_stem_domain(
    tmp_path: Path,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    with pytest.raises(FeatureClustersError, match="exact selection/segment"):
        psf._build_completed_clustered_selection_validation(
            output_dir=selection_path.parent,
            packed_proof=proof,
            selection_hash="c" * 64,
            selection_integrity=integrity,
            packed_rows_hash=proof.aggregate_sha256,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            feature_delta=None,
            effective_buckets=1,
            expected_segments_by_stem={"different": 1},
            validated_max_features_by_stem={},
            prior_metadata=None,
        )


def test_completed_clustered_validation_can_build_from_fresh_semantic_scan(
    tmp_path: Path,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    evidence = psf._build_completed_clustered_selection_validation(
        output_dir=selection_path.parent,
        packed_proof=proof,
        selection_hash="c" * 64,
        selection_integrity=integrity,
        packed_rows_hash=proof.aggregate_sha256,
        feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
        feature_delta=None,
        effective_buckets=1,
        expected_segments_by_stem={"shard_00000": 1},
        validated_max_features_by_stem={},
        prior_metadata=None,
    )
    assert evidence.certificate["shards"][0][
        "max_features_per_segment"
    ] == 1


def test_completed_clustered_validation_rejects_rebuilt_inventory_disagreement(
    tmp_path: Path,
) -> None:
    packed_path, selection_path, _integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    with pytest.raises(FeatureClustersError, match="rebuilt a selection inventory"):
        psf._build_completed_clustered_selection_validation(
            output_dir=selection_path.parent,
            packed_proof=proof,
            selection_hash="c" * 64,
            selection_integrity={},
            packed_rows_hash=proof.aggregate_sha256,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            feature_delta=None,
            effective_buckets=1,
            expected_segments_by_stem={"shard_00000": 1},
            validated_max_features_by_stem={},
            prior_metadata=None,
        )


def test_completed_clustered_validation_rejects_packed_aggregate_disagreement(
    tmp_path: Path,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    with pytest.raises(FeatureClustersError, match="packed cache disagrees"):
        psf._build_completed_clustered_selection_validation(
            output_dir=selection_path.parent,
            packed_proof=proof,
            selection_hash="c" * 64,
            selection_integrity=integrity,
            packed_rows_hash="0" * 64,
            feature_clusters=_AcceptingClusterConsolidation(),  # ty: ignore[invalid-argument-type]
            feature_delta=None,
            effective_buckets=1,
            expected_segments_by_stem={"shard_00000": 1},
            validated_max_features_by_stem={"shard_00000": 1},
            prior_metadata=None,
        )


@pytest.mark.parametrize(
    ("mutation_call", "match"),
    [
        (3, "certificate was reissued"),
        (4, "certificate was assembled"),
    ],
)
def test_completed_clustered_validation_rejects_selection_signature_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_call: int,
    match: str,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    real_identity = psf._selection_shard_file_identity
    calls = 0

    def mutate_observed_signature(path: Path, *, surface: str):
        nonlocal calls
        calls += 1
        signature = real_identity(path, surface=surface)
        if calls == mutation_call:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(
        psf,
        "_selection_shard_file_identity",
        mutate_observed_signature,
    )
    with pytest.raises(FeatureClustersError, match=match):
        _build_completed_validation_evidence(
            packed_path=packed_path,
            selection_path=selection_path,
            integrity=integrity,
        )


@pytest.mark.parametrize("prior_cluster_status", ["complete", "in_progress"])
def test_completed_clustered_validation_reissues_prior_schema_v1_authority(
    tmp_path: Path,
    prior_cluster_status: str,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    first = _build_completed_validation_evidence(
        packed_path=packed_path,
        selection_path=selection_path,
        integrity=integrity,
    )
    prior_metadata = {
        CLUSTERED_SELECTION_VALIDATION_KEY: first.certificate,
        "feature_clusters": {"status": prior_cluster_status},
    }
    reissued = _build_completed_validation_evidence(
        packed_path=packed_path,
        selection_path=selection_path,
        integrity=integrity,
        prior_metadata=prior_metadata,
    )
    assert reissued.certificate == first.certificate


def test_completed_clustered_validation_rejects_prior_segment_count_disagreement(
    tmp_path: Path,
) -> None:
    packed_path, selection_path, integrity = _write_clustered_finalization_pair(
        tmp_path
    )
    first = _build_completed_validation_evidence(
        packed_path=packed_path,
        selection_path=selection_path,
        integrity=integrity,
    )
    prior_metadata = {
        CLUSTERED_SELECTION_VALIDATION_KEY: first.certificate,
        "feature_clusters": {"status": "complete"},
    }
    with pytest.raises(
        FeatureClustersError,
        match="per-shard segment-count domain",
    ):
        _build_completed_validation_evidence(
            packed_path=packed_path,
            selection_path=selection_path,
            integrity=integrity,
            expected_segments=2,
            prior_metadata=prior_metadata,
        )


@pytest.mark.parametrize("existing", [False, True])
def test_spawn_worker_brackets_every_clustered_packed_read_with_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    packed_path, _selection_path, _integrity = (
        _write_clustered_finalization_pair(tmp_path)
    )
    proof = resolve_packed_row_fingerprint_strict(packed_path.parent)
    checks: list[Path] = []
    stats = psf.SpliceSelectionStats(total_packed_segments=1)
    state = {
        "provenance_dir": tmp_path,
        "packed_fingerprint_files": {packed_path.name: proof.files[0]},
        "feature_clusters": object(),
        "existing_stems": {"shard_00000"} if existing else set(),
        "packed_info": object(),
        "splice_index": object(),
        "packed_spool_dir": None,
        "pipeline": object(),
        "output_dir": tmp_path,
        "feature_index": object(),
        "seed": 0,
        "splice_batch_size": None,
    }
    monkeypatch.setattr(psf, "_SPLICE_WORKER_STATE", state)
    monkeypatch.setattr(
        psf,
        "assert_packed_row_fingerprint_file_current",
        lambda _entry, path: checks.append(path),
    )
    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        lambda *_args, **_kwargs: stats,
    )
    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        lambda *_args, **_kwargs: stats,
    )

    shard_name, was_written, result = psf._process_splice_shard_in_worker(
        str(packed_path)
    )
    assert (shard_name, was_written, result) == (
        "shard_00000",
        not existing,
        stats,
    )
    assert checks == [packed_path, packed_path]


@pytest.mark.parametrize("existing", [False, True])
def test_spawn_worker_preserves_unclustered_path_without_packed_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    packed_path, _selection_path, _integrity = (
        _write_clustered_finalization_pair(tmp_path)
    )
    stats = psf.SpliceSelectionStats(total_packed_segments=1)
    monkeypatch.setattr(
        psf,
        "_SPLICE_WORKER_STATE",
        {
            "provenance_dir": tmp_path,
            "packed_fingerprint_files": {},
            "feature_clusters": None,
            "existing_stems": {"shard_00000"} if existing else set(),
            "packed_info": object(),
            "splice_index": object(),
            "packed_spool_dir": None,
            "pipeline": object(),
            "output_dir": tmp_path,
            "feature_index": object(),
            "seed": 0,
            "splice_batch_size": None,
        },
    )
    monkeypatch.setattr(
        psf,
        "assert_packed_row_fingerprint_file_current",
        lambda *_args, **_kwargs: pytest.fail(
            "unclustered worker unexpectedly required packed proof"
        ),
    )
    monkeypatch.setattr(
        psf,
        "splice_counterfactual_stats_for_packed_shard",
        lambda *_args, **_kwargs: stats,
    )
    monkeypatch.setattr(
        psf,
        "select_splice_counterfactual_features_for_shard",
        lambda *_args, **_kwargs: stats,
    )

    shard_name, was_written, result = psf._process_splice_shard_in_worker(
        str(packed_path)
    )

    assert (shard_name, was_written, result) == (
        "shard_00000",
        not existing,
        stats,
    )


def test_spawn_worker_requires_clustered_packed_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_path, _selection_path, _integrity = (
        _write_clustered_finalization_pair(tmp_path)
    )
    monkeypatch.setattr(
        psf,
        "_SPLICE_WORKER_STATE",
        {
            "provenance_dir": tmp_path,
            "packed_fingerprint_files": {},
            "feature_clusters": object(),
            "existing_stems": set(),
        },
    )
    with pytest.raises(FeatureClustersError, match="no strict packed-row proof"):
        psf._process_splice_shard_in_worker(str(packed_path))
