"""Real l22 subset byte-identity for splice preselection (issue #740, gated).

Opt-in: set SPLICE740_* env vars to real l22 artifact paths. Builds a scratch
symlink-farm subset of a few REAL packed shards, runs the candidate cache path,
and byte-compares each output shard against the production sel/ shard the
pre-change live job produced (the real-scale reference). Manifest excluded
(production created_at is not frozen; shard bytes are the identity gate).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.preprocess.precompute_selected_features as psf
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, resolve_feature_mask
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import read_manifest
from steering.packed_batch_io import load_packed_row_shard, load_selection_shard

pytestmark = pytest.mark.requires_feature_data

_ENV = (
    "SPLICE740_FEATURE_DIR",
    "SPLICE740_PACKED_DIR",
    "SPLICE740_REF_SEL_DIR",
    "SPLICE740_WHITELIST",
    "SPLICE740_CLUSTERS",
)


def _require_env() -> dict[str, str]:
    missing = [k for k in _ENV if not os.environ.get(k)]
    if missing:
        pytest.skip(f"real-l22 test needs env vars: {missing}")
    return {k: os.environ[k] for k in _ENV}


def _build_symlink_subset(packed_dir: Path, scratch: Path, stems: list[str]) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    out_prov = provenance_dir(scratch)
    out_prov.mkdir(parents=True, exist_ok=True)
    src_prov = provenance_dir(packed_dir)
    for stem in stems:
        fn = f"{stem}.safetensors.zst"
        (scratch / fn).symlink_to((packed_dir / fn).resolve())
        (out_prov / fn).symlink_to((src_prov / fn).resolve())
    manifest = read_manifest(packed_dir)
    assert manifest is not None, f"missing packed-row manifest: {packed_dir}"
    manifest["metadata"]["provenance_shards"] = len(stems)
    (scratch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scratch


def _real_l22_config(
    env: dict[str, str],
    subset: Path,
    *,
    cache_shards: int,
    splice_batch_size: int | None = None,
):
    cfg = OmegaConf.create(
        {
            "packed_row_dir": str(subset),
            "feature_dir": env["SPLICE740_FEATURE_DIR"],
            "segment_size": 128,
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            # NOT the live CLI's arm: the live command never overrides
            # negative_selection, so this is configs/training/conditional_encdec_
            # train.yaml's plain top-level default (mean=3.0/max_k=15), NOT the
            # feature_selection_variant/sampling.yaml positive-arm numbers
            # (mean=12.0/max_k=49) reused by the shared synthetic test fixture
            # (tests/steering/_splice_cf_cache_fixtures.py). Verified by composing
            # the real Hydra config with the live job's exact CLI overrides
            # (`hydra.compose` against configs/training/conditional_encdec_train.
            # yaml) and by reading the identical file from the live job's own
            # checkout at /tmp/nano-gpt-pretrain-steer-latest2 (PID 1284396,
            # `git rev-parse HEAD` == 3f125ac2, byte-identical negative_selection
            # block). compute_selection_hash hashes this dict verbatim
            # (steering/packed_batch_io.py), so the wrong arm silently produces a
            # different sel/<hash>/ dir instead of failing loudly.
            "negative_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "min_k": 0,
                "max_k": 15,
            },
            "feature_delta": {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
            "feature_mask": {"mode": "whitelist", "path": env["SPLICE740_WHITELIST"]},
            "feature_clusters": {"path": env["SPLICE740_CLUSTERS"]},
            "feature_mask_freq_min": None,
            "feature_mask_freq_max": None,
            "num_buckets": 2,
            "seed": 42,
            "sae": {
                "release": "gemmascope_2",
                "model": "gemma-3-4b-pt",
                "activation_type": "residual",
                "layer": 22,
                "width": "65k",
                "l0": "medium",
            },
            "feature_root": None,
            "pooling": "max_pool",
            "dataset_phase": None,
            "selection_source_cache_shards": cache_shards,
            "selection_sidecar_cache_shards": 16,
            "selection_prefetch_stage_shards": 1,
            "selection_splice_batch_size": splice_batch_size,
        }
    )
    return cfg


def test_real_l22_subset_matches_production_reference(tmp_path: Path, monkeypatch) -> None:
    env = _require_env()
    cap = int(os.environ.get("SPLICE740_CAP", "3"))
    stems = os.environ.get("SPLICE740_SHARDS", "shard_00000,shard_00001,shard_00002").split(",")
    subset = _build_symlink_subset(Path(env["SPLICE740_PACKED_DIR"]), tmp_path / "subset", stems)
    cfg = _real_l22_config(env, subset, cache_shards=cap)
    psf.main.__wrapped__(cfg)

    cand_sel = next((subset / "sel").iterdir())
    ref_sel = Path(env["SPLICE740_REF_SEL_DIR"])
    # The config identity must reproduce the production selection hash.
    assert cand_sel.name == ref_sel.name, (
        f"candidate sel hash {cand_sel.name} != production {ref_sel.name}; config drift"
    )
    for stem in stems:
        fn = f"{stem}.safetensors.zst"
        cand = (cand_sel / fn).read_bytes()
        ref = (ref_sel / fn).read_bytes()
        assert cand == ref, f"{fn}: candidate bytes differ from production reference"


@pytest.mark.very_slow
def test_real_l22_multisegment_batching_satisfies_full_shard_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run the opt-in producer on the same real first-three-shard fixture.

    There is intentionally no byte comparison to legacy mode: the versioned
    batching algorithm changes RNG draw assignment. Instead, this gate checks
    its independently-addressed manifest plus every serialized row invariant,
    resolved whitelist constraint, and cluster-representative constraint.
    """
    env = _require_env()
    cap = int(os.environ.get("SPLICE740_CAP", "3"))
    stems = os.environ.get(
        "SPLICE740_SHARDS",
        "shard_00000,shard_00001,shard_00002",
    ).split(",")
    subset = _build_symlink_subset(
        Path(env["SPLICE740_PACKED_DIR"]),
        tmp_path / "batched_subset",
        stems,
    )
    cfg = _real_l22_config(
        env,
        subset,
        cache_shards=cap,
        splice_batch_size=64,
    )
    psf.main.__wrapped__(cfg)

    selection_dirs = list((subset / "sel").iterdir())
    assert len(selection_dirs) == 1
    selection_dir = selection_dirs[0]
    assert selection_dir.name != Path(env["SPLICE740_REF_SEL_DIR"]).name
    manifest = read_manifest(selection_dir)
    assert manifest is not None
    metadata = manifest["metadata"]
    splice = metadata["splice_counterfactual"]
    assert splice["selection_batching"] == {
        "schema_version": 1,
        "algorithm": "ordinary_execution_runs_v1",
        "batch_size": 64,
    }
    assert splice["status"] == "complete"

    mask = resolve_feature_mask(cfg, 65_536)
    assert isinstance(mask, FeatureMask)
    allowed = mask.allowed_universe_mask(65_536)
    resolved_clusters = resolve_feature_clusters(cfg, 65_536)
    assert resolved_clusters is not None
    clusters = FeatureClusterConsolidation.build(resolved_clusters, mask)

    total_segments = 0
    for stem in stems:
        filename = f"{stem}.safetensors.zst"
        packed = load_packed_row_shard(subset / filename)
        selected = load_selection_shard(selection_dir / filename)
        total_segments += packed.total_segments
        assert selected.total_segments == packed.total_segments
        assert selected.sel_offsets.dtype == t.int32
        assert selected.sel_fids.dtype == t.uint16
        assert selected.sel_buckets.dtype == t.uint8
        assert selected.sel_pcts.dtype == t.uint8
        assert selected.neg_offsets.dtype == t.int32
        assert selected.neg_fids.dtype == t.uint16
        assert int(selected.sel_offsets[0].item()) == 0
        assert int(selected.neg_offsets[0].item()) == 0
        assert int(selected.sel_offsets[-1].item()) == selected.sel_fids.numel()
        assert int(selected.neg_offsets[-1].item()) == selected.neg_fids.numel()
        assert bool((selected.sel_offsets[1:] >= selected.sel_offsets[:-1]).all())
        assert bool((selected.neg_offsets[1:] >= selected.neg_offsets[:-1]).all())
        assert selected.sel_fids.numel() == selected.sel_buckets.numel()
        assert selected.sel_fids.numel() == selected.sel_pcts.numel()
        assert bool((selected.sel_buckets < 2).all())

        for segment_index in range(packed.total_segments):
            sel_start = int(selected.sel_offsets[segment_index].item())
            sel_end = int(selected.sel_offsets[segment_index + 1].item())
            neg_start = int(selected.neg_offsets[segment_index].item())
            neg_end = int(selected.neg_offsets[segment_index + 1].item())
            positive_ids = selected.sel_fids[sel_start:sel_end].to(t.int64)
            negative_ids = selected.neg_fids[neg_start:neg_end].to(t.int64)
            assert positive_ids.unique().numel() == positive_ids.numel()
            assert negative_ids.unique().numel() == negative_ids.numel()
            assert not bool(t.isin(positive_ids, negative_ids).any())
            if positive_ids.numel():
                assert bool(allowed[positive_ids].all())
            if negative_ids.numel():
                assert bool(allowed[negative_ids].all())
            clusters.assert_representatives_only(
                positive_ids,
                surface=f"real l22 positive {stem}:{segment_index}",
            )
            clusters.assert_representatives_only(
                negative_ids,
                surface=f"real l22 negative {stem}:{segment_index}",
            )

    assert splice["total_packed_segments_selected"] == total_segments
