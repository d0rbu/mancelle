"""Byte-identity, profiling, and cache tests for precompute_token_boundaries (issue #529).

The golden anchor (``tests/scripts/golden/token_boundaries/``) is captured from the
*unmodified* script and is the reference every optimization phase must reproduce exactly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file

from scripts.preprocess.precompute_token_boundaries import run as run_tok
from steering.feature_io import TensorKey, shard_stem_from_path
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
)
from steering.segmentation import compute_tokenizer_checksum
from tests.scripts._token_boundary_fixtures import make_token_boundary_env

_GOLDEN = Path(__file__).parent / "golden" / "token_boundaries"
_BOUNDARY_KEYS = (TensorKey.SEGMENT_TOKEN_STARTS, TensorKey.SEGMENT_TOKEN_ENDS)


def _cfg(env: dict, **overrides) -> DictConfig:
    base = {
        "feature_dir": env["feature_dir"],
        "data_dir": env["data_dir"],
        "segment_size": env["segment_size"],
        "dry_run": False,
        "resume": False,
        "force": False,
        "workers": 1,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _run(env: dict, **overrides) -> int:
    with patch(
        "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
        return_value=env["tokenizer"],
    ):
        return run_tok(_cfg(env, **overrides))


def _overlay_dir_for(env: dict) -> Path:
    checksum = compute_tokenizer_checksum(env["tokenizer"])
    return overlay_dir(Path(env["feature_dir"]), OverlayKind.SEG_BOUNDARY, checksum)


def test_fixture_runs_end_to_end(tmp_path):
    env = make_token_boundary_env(tmp_path, n_shards=1)
    assert _run(env) == 0
    assert len(glob_overlay_shards(_overlay_dir_for(env))) == 1


def test_golden_byte_identity(tmp_path):
    """Every optimization phase must reproduce this golden (captured from main) exactly.

    The anchor is the OVERLAY output, which is fully byte-reproducible: ``save_overlay_shard``
    writes empty safetensors metadata, so its compressed bytes are deterministic across
    processes, and the token-boundary tensors depend only on the (constant) segmentation
    metadata. We deliberately do NOT anchor ``base_shard_content_hash`` here: ``save_features``
    serializes its rich metadata via the safetensors Rust ``HashMap`` whose key order is
    randomized per call, so a base shard rebuilt by the fixture has non-reproducible bytes
    (and thus hash) even for identical tensors. The cache's hash byte-identity invariant
    ("cached == streamed over the SAME on-disk files") is deterministic and is proven in
    tests/steering/test_base_shard_hash_cache.py — see issue #529 notes.
    """
    env = make_token_boundary_env(tmp_path, n_shards=1)
    assert _run(env) == 0
    d = _overlay_dir_for(env)
    shard_zst = overlay_shard_path(d, "shard_00000")
    tensors = load_overlay_shard(shard_zst)

    if os.environ.get("TOKBOUND_CAPTURE_GOLDEN") == "1":
        _GOLDEN.mkdir(parents=True, exist_ok=True)
        st_save_file(
            {str(k): tensors[k].contiguous() for k in _BOUNDARY_KEYS},
            str(_GOLDEN / "shard_00000_seg_boundaries.safetensors"),
        )
        (_GOLDEN / "shard_00000_seg_boundaries.safetensors.zst").write_bytes(
            shard_zst.read_bytes()
        )
        pytest.skip(f"captured golden -> {_GOLDEN}")

    assert (_GOLDEN / "shard_00000_seg_boundaries.safetensors").is_file(), (
        "golden missing; run with TOKBOUND_CAPTURE_GOLDEN=1 first"
    )
    golden = st_load_file(str(_GOLDEN / "shard_00000_seg_boundaries.safetensors"))
    for k in _BOUNDARY_KEYS:
        assert t.equal(tensors[k], golden[str(k)]), f"{k} diverged from golden"
    # save_overlay_shard is deterministic (empty metadata); the compressed bytes are a
    # strong same-env anchor (regenerate the .zst golden if the system zstd version changes).
    assert (
        shard_zst.read_bytes()
        == (_GOLDEN / "shard_00000_seg_boundaries.safetensors.zst").read_bytes()
    ), "compressed overlay bytes diverged (zstd version drift? re-capture golden)"


def _assert_shard0_matches_golden(env: dict) -> None:
    golden = st_load_file(str(_GOLDEN / "shard_00000_seg_boundaries.safetensors"))
    tensors = load_overlay_shard(overlay_shard_path(_overlay_dir_for(env), "shard_00000"))
    for k in _BOUNDARY_KEYS:
        assert t.equal(tensors[k], golden[str(k)]), f"{k} changed vs golden"


def test_profile_sidecar_written_and_outputs_unchanged(tmp_path):
    env = make_token_boundary_env(tmp_path, n_shards=2)
    summary = tmp_path / "prof.json"
    assert _run(env, profile=True, profile_summary_output=str(summary)) == 0

    data = json.loads(summary.read_text())
    assert set(data) == {"config", "manifest_pre_pass", "worker_pool", "per_shard", "totals"}
    assert data["totals"]["n_processed"] == 2
    assert len(data["per_shard"]) == 2
    assert data["manifest_pre_pass"]["seconds"] >= 0.0
    assert data["manifest_pre_pass"]["bytes_read"] > 0  # P0: full pre-pass measured
    assert data["manifest_pre_pass"]["n_shards_hashed"] == 2
    for sh in data["per_shard"]:
        assert sh["n_segments"] > 0
        assert sh["load_bytes"] > 0
        assert sh["overlay_bytes"] > 0
        assert set(sh["offset_paths"]) == {"fast", "slow", "small_n"}
        assert all(
            set(sh["offset_paths"][path]) == {"count", "total_ms"}
            for path in sh["offset_paths"]
        )
        assert sum(stats["count"] for stats in sh["offset_paths"].values()) == 2
        assert all(stats["total_ms"] >= 0.0 for stats in sh["offset_paths"].values())
        assert sh["worker_compute_seconds"] > 0.0
        assert sh["map_wait_seconds"] == 0.0
        assert sh["scatter_seconds"] >= 0.0

    # Instrumentation must not change the overlay output.
    _assert_shard0_matches_golden(env)


def test_profile_offset_path_counts_match_between_worker_modes(tmp_path):
    env_seq = make_token_boundary_env(tmp_path / "seq", n_shards=2)
    seq_summary = tmp_path / "seq-profile.json"
    assert _run(env_seq, workers=1, profile=True, profile_summary_output=str(seq_summary)) == 0
    seq_data = json.loads(seq_summary.read_text())
    seq_counts = [
        {path: stats["count"] for path, stats in sh["offset_paths"].items()}
        for sh in seq_data["per_shard"]
    ]

    env_pool = make_token_boundary_env(tmp_path / "pool", n_shards=2)
    pool_summary = tmp_path / "pool-profile.json"
    assert _run(env_pool, workers=2, profile=True, profile_summary_output=str(pool_summary)) == 0
    pool_data = json.loads(pool_summary.read_text())
    pool_counts = [
        {path: stats["count"] for path, stats in sh["offset_paths"].items()}
        for sh in pool_data["per_shard"]
    ]
    assert pool_counts == seq_counts
    _assert_shard0_matches_golden(env_pool)


def test_profile_false_writes_no_sidecar(tmp_path):
    env = make_token_boundary_env(tmp_path, n_shards=1)
    summary = tmp_path / "prof.json"
    assert _run(env, profile=False, profile_summary_output=str(summary)) == 0
    assert not summary.exists()
    _assert_shard0_matches_golden(env)


def test_run_uses_cache_second_pass(tmp_path):
    env = make_token_boundary_env(tmp_path, n_shards=2)
    cache = Path(env["feature_dir"]) / ".base_shard_hash_cache.json"
    s1, s2 = tmp_path / "p1.json", tmp_path / "p2.json"

    assert _run(env, profile=True, profile_summary_output=str(s1)) == 0
    assert cache.is_file(), "first run must write the hash cache"
    pre1 = json.loads(s1.read_text())["manifest_pre_pass"]
    assert pre1["cache_hit"] is False and pre1["bytes_read"] > 0
    manifest1 = read_manifest(_overlay_dir_for(env))
    assert manifest1 is not None
    bh1 = manifest1["base_shard_content_hash"]

    # Second run (force=true) rebuilds overlays but must HIT the hash cache (base shards unchanged).
    assert _run(env, profile=True, profile_summary_output=str(s2), force=True) == 0
    pre2 = json.loads(s2.read_text())["manifest_pre_pass"]
    assert pre2["cache_hit"] is True, "second run must hit the cache"
    assert pre2["bytes_read"] == 0, "cache hit must read zero shard bytes"
    manifest2 = read_manifest(_overlay_dir_for(env))
    assert manifest2 is not None
    bh2 = manifest2["base_shard_content_hash"]
    assert bh1 == bh2, "cached hash value must equal the streamed value across runs"
    _assert_shard0_matches_golden(env)


# ---------------------------------------------------------------------------
# Phase 2 — pool workers across shards (#2, now unconditional)
# ---------------------------------------------------------------------------


def _count_pools(monkeypatch):
    import scripts.preprocess.precompute_token_boundaries as mod

    n = {"count": 0}
    real = mod.ProcessPoolExecutor

    def counting(*args, **kwargs):
        n["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "ProcessPoolExecutor", counting)
    return n


@pytest.mark.slow
def test_pooled_creates_one_pool(tmp_path, monkeypatch):
    # Worker pooling is unconditional: a multi-shard, multi-worker run creates exactly ONE pool
    # (reused across every shard), never one per shard — and still reproduces the golden output.
    n = _count_pools(monkeypatch)
    env = make_token_boundary_env(tmp_path, n_shards=3)
    assert _run(env, workers=2) == 0
    assert n["count"] == 1, f"pooled run must create exactly one pool, got {n['count']}"
    _assert_shard0_matches_golden(env)


def test_sequential_run_creates_no_pool(tmp_path, monkeypatch):
    # workers=1 ⇒ sequential: run() builds no pool and passes worker_pool=None to precompute_shard
    # (n_workers and the old multi-worker guard are gone; None is the sequential signal). The
    # overlay still matches the golden.
    n = _count_pools(monkeypatch)
    env = make_token_boundary_env(tmp_path, n_shards=2)
    assert _run(env, workers=1) == 0
    assert n["count"] == 0, f"sequential run must construct no pool, got {n['count']}"
    _assert_shard0_matches_golden(env)


def test_resume_detects_bos_from_first_pending_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed shard needs no parquet and must not trigger a BOS guess."""
    import scripts.preprocess.precompute_token_boundaries as mod

    env = make_token_boundary_env(tmp_path, n_shards=2)
    assert _run(env, shard_end=1) == 0
    (Path(env["data_dir"]) / "shard_00000.parquet").unlink()

    seen: list[str] = []
    real_detect = mod._detect_source_bos_char_len

    def recording_detect(shard_path, parquet_path, *, load_keys=None):
        seen.append(shard_stem_from_path(shard_path))
        return real_detect(shard_path, parquet_path, load_keys=load_keys)

    monkeypatch.setattr(mod, "_detect_source_bos_char_len", recording_detect)

    assert _run(env, resume=True) == 0
    assert seen == ["shard_00001"]
    assert overlay_shard_path(_overlay_dir_for(env), "shard_00001").is_file()


def test_completed_resume_needs_no_parquet_or_worker_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.preprocess.precompute_token_boundaries as mod

    env = make_token_boundary_env(tmp_path, n_shards=2)
    assert _run(env) == 0
    for parquet_path in Path(env["data_dir"]).glob("*.parquet"):
        parquet_path.unlink()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("completed resume must not detect BOS or start workers")

    monkeypatch.setattr(mod, "_detect_source_bos_char_len", unexpected)
    monkeypatch.setattr(mod, "ProcessPoolExecutor", unexpected)

    assert _run(env, resume=True, workers=2) == 0


def test_empty_shard_range_is_a_write_free_noop(tmp_path: Path) -> None:
    env = make_token_boundary_env(tmp_path, n_shards=1)

    assert _run(env, shard_start=1, shard_end=1) == 0
    assert not _overlay_dir_for(env).exists()


def test_pending_shard_requires_recorded_segment_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.preprocess.precompute_token_boundaries as mod

    env = make_token_boundary_env(tmp_path, n_shards=1)
    real_read_metadata = mod.read_shard_metadata

    def metadata_without_segment_size(path):
        metadata = real_read_metadata(path)
        metadata.pop("segment_size", None)
        return metadata

    monkeypatch.setattr(mod, "read_shard_metadata", metadata_without_segment_size)

    with pytest.raises(KeyError, match="segment_size"):
        _run(env)


@pytest.mark.slow
def test_pooled_matches_sequential_all_shards(tmp_path):
    # The pooled path (workers>1) must produce overlays byte-identical to the sequential path
    # (workers=1) for EVERY shard via run() — not just shard0's golden. Closes the gap left by
    # removing the old pooled-vs-per-shard / subset-on-vs-off equivalence tests.
    env_pooled = make_token_boundary_env(tmp_path / "pooled", n_shards=3)
    env_seq = make_token_boundary_env(tmp_path / "seq", n_shards=3)
    assert _run(env_pooled, workers=2) == 0
    assert _run(env_seq, workers=1) == 0
    d_pooled, d_seq = _overlay_dir_for(env_pooled), _overlay_dir_for(env_seq)
    for stem in ("shard_00000", "shard_00001", "shard_00002"):
        tp = load_overlay_shard(overlay_shard_path(d_pooled, stem))
        ts = load_overlay_shard(overlay_shard_path(d_seq, stem))
        for k in _BOUNDARY_KEYS:
            assert t.equal(tp[k], ts[k]), f"{stem}/{k}: pooled != sequential"


# ---------------------------------------------------------------------------
# Phase 3 — subset / header-only loading (#3, now unconditional)
# ---------------------------------------------------------------------------


def test_segment_size_read_is_header_only(tmp_path, monkeypatch):
    import scripts.preprocess.precompute_token_boundaries as mod

    calls = {"meta": 0}
    real = mod.read_shard_metadata

    def spy(p):
        calls["meta"] += 1
        return real(p)

    monkeypatch.setattr(mod, "read_shard_metadata", spy)
    env = make_token_boundary_env(tmp_path, n_shards=1)
    assert _run(env) == 0
    assert calls["meta"] >= 1, "segment_size must be read via header-only read_shard_metadata"
    _assert_shard0_matches_golden(env)


def test_subset_load_passes_keys_and_skips_full_load(tmp_path, monkeypatch):
    import scripts.preprocess.precompute_token_boundaries as mod

    seen = {"with_keys": 0, "without_keys": 0}
    real = mod.load_features

    def spy(path, *, keys=None):
        if keys is None:
            seen["without_keys"] += 1
        else:
            seen["with_keys"] += 1
        return real(path, keys=keys)

    monkeypatch.setattr(mod, "load_features", spy)
    env = make_token_boundary_env(tmp_path, n_shards=1)
    assert _run(env) == 0
    assert seen["with_keys"] >= 1, "subset loading must pass keys to load_features"
    assert seen["without_keys"] == 0, "subset loading must do no full feature load"


# ---------------------------------------------------------------------------
# Regression tests for code-review findings
# ---------------------------------------------------------------------------


def test_profile_output_null_does_not_crash(tmp_path):
    # profile=true with profile_output=None (the shipped YAML default) must not crash on
    # Path(None) in the chrome-trace branch.
    env = make_token_boundary_env(tmp_path, n_shards=1)
    assert (
        _run(
            env,
            profile=True,
            profile_output=None,
            profile_summary_output=str(tmp_path / "p.json"),
        )
        == 0
    )
    _assert_shard0_matches_golden(env)


def test_profile_trace_false_skips_trace_keeps_sidecar(tmp_path):
    """profile_trace=false must skip the chrome trace but keep the sidecar.

    Review/benchmark finding (#1269): the first-shard torch.profiler trace
    OOMs on real production shards (with_stack event buffers reach tens of
    GB) and forces the first shard sequential, distorting worker sweeps —
    the benchmark harness therefore runs with profile_trace=false and needs
    the sidecar alone, with byte-identical outputs.
    """
    env = make_token_boundary_env(tmp_path, n_shards=1)
    trace_path = tmp_path / "trace.json"
    summary = tmp_path / "p.json"
    assert (
        _run(
            env,
            profile=True,
            profile_trace=False,
            profile_output=str(trace_path),
            profile_summary_output=str(summary),
        )
        == 0
    )
    assert not trace_path.exists()
    assert summary.is_file()
    assert json.loads(summary.read_text())["totals"]["n_processed"] == 1
    _assert_shard0_matches_golden(env)


def test_sidecar_flushed_when_shard_raises(tmp_path, monkeypatch):
    # A shard raising mid-loop must still flush the profile sidecar via the finally (and shut
    # down the pool) — the timings collected before the raise must not be lost.
    import scripts.preprocess.precompute_token_boundaries as mod

    env = make_token_boundary_env(tmp_path, n_shards=2)
    summary = tmp_path / "p.json"
    real = mod.precompute_shard
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("simulated shard failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "precompute_shard", boom)
    with pytest.raises(ValueError, match="simulated"):
        _run(env, profile=True, profile_summary_output=str(summary))

    assert summary.is_file(), "sidecar must be flushed in finally even on a mid-loop raise"
    data = json.loads(summary.read_text())
    assert data["totals"]["n_processed"] == 1
    assert len(data["per_shard"]) == 1
