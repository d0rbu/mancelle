# tests/axbench_bpb/unit/test_context_cache_sharding.py
"""Unit tests for build_context_cache's round-robin sharding.

Commit 24680f51 added ``ddp_rank`` / ``ddp_world_size`` kwargs to
``build_context_cache``. The function sorts unique prefixes by
``prefix_hash`` and slices ``items[ddp_rank::ddp_world_size]`` so each
rank handles a deterministic, disjoint subset. These tests pin every
sharding invariant at the unit layer (no real models, no real
distributed group).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from steering.eval.axbench_bpb import context_cache as cc_mod
from steering.eval.axbench_bpb.config import (
    AxBenchBpbConfig,
    ContextCacheConfig,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.target_conditioning import ContextFeatureSegment


def _prefix_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_cfg(**overrides: Any) -> AxBenchBpbConfig:
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/src",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        filter_unaccepted_samples=True,
        context_cache=ContextCacheConfig(force_rebuild=False),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _stub_collect(monkeypatch, prefixes: list[str]) -> dict[str, str]:
    """Stub collect_unique_prefixes to return a deterministic
    {prefix_text: prefix_hash} dict for the given list."""
    unique = {p: _prefix_hash(p) for p in prefixes}

    def _fake_collect(*_a, **_kw):
        return unique

    monkeypatch.setattr(cc_mod, "collect_unique_prefixes", _fake_collect)
    return unique


def _stub_extract(monkeypatch) -> list[str]:
    """Stub extract_prefix_features + serialize_context_segments to
    record which prefixes a rank processed. Returns a list mutated
    in-place by the stubs."""
    processed: list[str] = []

    def _fake_extract(*, prefix_text, **_kw):
        processed.append(prefix_text)
        # Return a single dummy segment so serialize has something to write.
        return (ContextFeatureSegment(
            feature_ids=(0,), buckets=(0,), char_start=0, char_end=1,
        ),)

    def _fake_serialize(_segs, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # marker so .exists() returns True

    monkeypatch.setattr(cc_mod, "extract_prefix_features", _fake_extract)
    monkeypatch.setattr(cc_mod, "serialize_context_segments", _fake_serialize)
    return processed


def _make_output_paths(tmp_path: Path) -> OutputPaths:
    return OutputPaths(output_root=tmp_path, run_tag="test")


# ---------- single-process behaviour preserved ----------


def test_single_process_processes_all_prefixes(tmp_path, monkeypatch):
    """ws=1: every prefix is processed; manifest is written; the
    return value is the manifest path."""
    prefixes = [f"p{i}" for i in range(10)]
    _stub_collect(monkeypatch, prefixes)
    processed = _stub_extract(monkeypatch)
    # Stub the manifest writer too — we test it separately.
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest",
        lambda **kwargs: kwargs["output_paths"].context_cache_manifest(),
    )

    result = cc_mod.build_context_cache(
        cfg=_make_cfg(),
        output_paths=_make_output_paths(tmp_path),
        source_positive=tmp_path / "pos.parquet",
        source_negative=tmp_path / "neg.parquet",
        base_model=MagicMock(),
        base_sae=MagicMock(),
        boundaries=MagicMock(),
        segment_size=128,
        sae_layer=8,
    )
    assert sorted(processed) == sorted(prefixes)
    # Manifest path returned (stubbed to context_cache_manifest()).
    assert result is not None
    assert result.name == "manifest.json"


# ---------- sharding correctness: complete + disjoint partition ----------


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_sharding_partitions_prefixes_completely_and_disjointly(
    tmp_path, monkeypatch, world_size,
):
    """For any world_size, the union of rank slices covers every prefix
    exactly once."""
    n_prefixes = 17  # non-multiple of 8/4/3/2 to surface off-by-ones
    prefixes = [f"p{i:03d}" for i in range(n_prefixes)]
    _stub_collect(monkeypatch, prefixes)
    # Each rank gets its own processed-list stub.
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest", lambda **kw: None,
    )

    seen_by_rank: dict[int, list[str]] = {}
    for rank in range(world_size):
        processed = _stub_extract(monkeypatch)
        cc_mod.build_context_cache(
            cfg=_make_cfg(),
            output_paths=_make_output_paths(tmp_path),
            source_positive=tmp_path / "pos.parquet",
            source_negative=tmp_path / "neg.parquet",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            boundaries=MagicMock(),
            segment_size=128,
            sae_layer=8,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )
        seen_by_rank[rank] = list(processed)

    # Coverage: union over ranks == set of all prefixes.
    union = [p for rank_list in seen_by_rank.values() for p in rank_list]
    assert sorted(union) == sorted(prefixes), (
        f"world_size={world_size}: rank slices did not cover all prefixes. "
        f"Missing: {set(prefixes) - set(union)}. "
        f"Extra: {set(union) - set(prefixes)}"
    )
    # Disjoint: no prefix is processed by more than one rank.
    rank_for_prefix: dict[str, int] = {}
    for r, items in seen_by_rank.items():
        for p in items:
            assert p not in rank_for_prefix, (
                f"world_size={world_size}: prefix {p!r} processed by both "
                f"rank {rank_for_prefix[p]} and rank {r}"
            )
            rank_for_prefix[p] = r


def test_sharding_is_deterministic_across_ranks(tmp_path, monkeypatch):
    """Two callers that compute their own slice independently produce
    disjoint, deterministic results: rank N always processes the same
    prefixes regardless of call order or other ranks' progress."""
    prefixes = [f"prefix_{i}" for i in range(20)]
    _stub_collect(monkeypatch, prefixes)
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest", lambda **kw: None,
    )

    def _run(rank: int, call_id: int) -> list[str]:
        # Use a fresh output dir per call so idempotency (which skips
        # already-on-disk safetensors) doesn't mask the rank-slicing
        # behaviour we're trying to test.
        processed = _stub_extract(monkeypatch)
        cc_mod.build_context_cache(
            cfg=_make_cfg(),
            output_paths=_make_output_paths(tmp_path / f"r{rank}_c{call_id}"),
            source_positive=tmp_path / "pos.parquet",
            source_negative=tmp_path / "neg.parquet",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            boundaries=MagicMock(),
            segment_size=128,
            sae_layer=8,
            ddp_rank=rank,
            ddp_world_size=4,
        )
        return list(processed)

    first_call_rank_1 = _run(1, 0)
    second_call_rank_1 = _run(1, 1)
    assert first_call_rank_1 == second_call_rank_1, (
        f"rank 1's slice is non-deterministic across calls: "
        f"{first_call_rank_1!r} != {second_call_rank_1!r}"
    )


def test_sharding_with_more_ranks_than_prefixes(tmp_path, monkeypatch):
    """If world_size > len(prefixes), some ranks get an empty slice but
    the loop still completes cleanly (no division-by-zero, no crash)."""
    prefixes = [f"p{i}" for i in range(3)]  # 3 prefixes, 8 ranks
    _stub_collect(monkeypatch, prefixes)
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest", lambda **kw: None,
    )

    for rank in range(8):
        processed = _stub_extract(monkeypatch)
        cc_mod.build_context_cache(
            cfg=_make_cfg(),
            output_paths=_make_output_paths(tmp_path / f"r{rank}"),
            source_positive=tmp_path / "pos.parquet",
            source_negative=tmp_path / "neg.parquet",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            boundaries=MagicMock(),
            segment_size=128,
            sae_layer=8,
            ddp_rank=rank,
            ddp_world_size=8,
        )
        if rank < 3:
            assert len(processed) == 1, (
                f"rank {rank} of 8 with 3 prefixes should process 1, got "
                f"{len(processed)}"
            )
        else:
            assert processed == [], (
                f"rank {rank} of 8 with 3 prefixes should be empty, got "
                f"{processed!r}"
            )


# ---------- DDP-vs-single-process return contract ----------


def test_returns_none_under_ddp(tmp_path, monkeypatch):
    """Under ws>1, build_context_cache returns None — the caller is
    responsible for the barrier + manifest write. This pins the
    contract so a refactor that re-introduces a per-rank manifest
    write is caught."""
    _stub_collect(monkeypatch, ["p0"])
    _stub_extract(monkeypatch)
    # Spy on the manifest writer so we can assert it is NOT called.
    manifest_call_count = [0]
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest",
        lambda **_kw: manifest_call_count.__setitem__(0, manifest_call_count[0] + 1),
    )

    result = cc_mod.build_context_cache(
        cfg=_make_cfg(),
        output_paths=_make_output_paths(tmp_path),
        source_positive=tmp_path / "pos.parquet",
        source_negative=tmp_path / "neg.parquet",
        base_model=MagicMock(),
        base_sae=MagicMock(),
        boundaries=MagicMock(),
        segment_size=128,
        sae_layer=8,
        ddp_rank=0,
        ddp_world_size=4,
    )
    assert result is None, (
        f"build_context_cache under DDP must return None; got {result!r}"
    )
    assert manifest_call_count[0] == 0, (
        f"build_context_cache under DDP must NOT call write_context_cache_manifest; "
        f"got {manifest_call_count[0]} calls"
    )


def test_single_process_writes_manifest_inline(tmp_path, monkeypatch):
    """ws=1: the legacy path writes the manifest from inside
    build_context_cache."""
    _stub_collect(monkeypatch, ["p0"])
    _stub_extract(monkeypatch)
    manifest_call_count = [0]
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest",
        lambda **_kw: (
            manifest_call_count.__setitem__(0, manifest_call_count[0] + 1)
            or _kw["output_paths"].context_cache_manifest()
        ),
    )

    result = cc_mod.build_context_cache(
        cfg=_make_cfg(),
        output_paths=_make_output_paths(tmp_path),
        source_positive=tmp_path / "pos.parquet",
        source_negative=tmp_path / "neg.parquet",
        base_model=MagicMock(),
        base_sae=MagicMock(),
        boundaries=MagicMock(),
        segment_size=128,
        sae_layer=8,
        ddp_rank=0,
        ddp_world_size=1,
    )
    assert manifest_call_count[0] == 1
    assert result is not None


# ---------- idempotency / force_rebuild ----------


def test_idempotency_skips_existing_files(tmp_path, monkeypatch):
    """If a prefix's safetensors already exists and force_rebuild=False,
    extract_prefix_features is NOT called for that prefix."""
    prefixes = ["p0", "p1"]
    unique = _stub_collect(monkeypatch, prefixes)
    output_paths = _make_output_paths(tmp_path)
    # Pre-create one of the two safetensors files.
    existing_hash = unique["p0"]
    existing_path = output_paths.context_cache_file(existing_hash)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("")

    processed = _stub_extract(monkeypatch)
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest", lambda **_kw: None,
    )

    cc_mod.build_context_cache(
        cfg=_make_cfg(),
        output_paths=output_paths,
        source_positive=tmp_path / "pos.parquet",
        source_negative=tmp_path / "neg.parquet",
        base_model=MagicMock(),
        base_sae=MagicMock(),
        boundaries=MagicMock(),
        segment_size=128,
        sae_layer=8,
        ddp_rank=0,
        ddp_world_size=1,
    )
    # p0 (already on disk) should be skipped; p1 should be processed.
    assert processed == ["p1"], (
        f"Idempotency violated: expected only ['p1'] processed; got {processed!r}"
    )


def test_force_rebuild_overrides_idempotency(tmp_path, monkeypatch):
    """force_rebuild=True re-extracts every prefix regardless of
    existing files."""
    prefixes = ["p0", "p1"]
    unique = _stub_collect(monkeypatch, prefixes)
    output_paths = _make_output_paths(tmp_path)
    existing_hash = unique["p0"]
    existing_path = output_paths.context_cache_file(existing_hash)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("")

    processed = _stub_extract(monkeypatch)
    monkeypatch.setattr(
        cc_mod, "write_context_cache_manifest", lambda **_kw: None,
    )

    cfg = _make_cfg()
    cfg.context_cache.force_rebuild = True
    cc_mod.build_context_cache(
        cfg=cfg,
        output_paths=output_paths,
        source_positive=tmp_path / "pos.parquet",
        source_negative=tmp_path / "neg.parquet",
        base_model=MagicMock(),
        base_sae=MagicMock(),
        boundaries=MagicMock(),
        segment_size=128,
        sae_layer=8,
        ddp_rank=0,
        ddp_world_size=1,
    )
    assert sorted(processed) == ["p0", "p1"], (
        f"force_rebuild=True must re-extract every prefix; got {processed!r}"
    )
