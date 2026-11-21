"""End-to-end cross-writer base-shard-hash-cache interop + byte-identity (issue #530).

Per-writer byte-identity (every manifest's ``base_shard_content_hash`` equals the
``cache_dir=None`` streaming reference over the same on-disk shards) is already proven in
``tests/steering/test_base_shard_hash_cache.py`` and the per-producer suites. This slow test
closes the remaining gap: two DIFFERENT feature_dir-caching producers, run in sequence on ONE
shared base-shard fixture, must INTEROP through the single ``feature_dir`` cache, and the cached
value the second writer consumes must be byte-identical to the streaming reference.

Pair: ``precompute_percentiles.run()`` (WRITER A — populates the cache via ``write_manifest(...,
cache_dir=feature_dir)``) then ``precompute_token_boundaries.run()`` (WRITER B — computes the base
hash unconditionally via ``_compute_base_hash(shard_paths, cache_dir=feature_dir)`` before any
per-shard work). Both hash ``glob_features(feature_dir)`` and cache in ``feature_dir``; neither
depends on the other's overlay, so this is a clean interop probe.

NON-VACUITY of the spy-count==0 assertion: ``_compute_base_hash`` calls the canonical streamer
``_stream_base_hash`` ONLY on a cache miss; on a hit it returns the cached digest with
``bytes_read=0`` and never touches ``_stream_base_hash``. WRITER B calls ``_compute_base_hash``
unconditionally at the top of ``run()`` (the manifest pre-pass in precompute_token_boundaries.py),
so a spy count of exactly 0 genuinely means "WRITER B hit the cache WRITER A populated", not "the
hash was never needed". The companion ``test_writer_b_streams_when_cache_absent`` proves the spy
fires (count >= 1) when the cache is removed, so a 0 here is a real, earned cache hit.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

import steering.overlay_io as overlay_io
from scripts.preprocess.precompute_percentiles import run as run_pct
from scripts.preprocess.precompute_token_boundaries import run as run_tok
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import glob_features
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    OverlayKind,
    compute_base_shard_content_hash,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_dir,
    read_manifest,
)
from steering.segmentation import (
    SegmentMode,
    compute_tokenizer_checksum,
)
from steering.stats_io import load_statistics, save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)
from tests.scripts._token_boundary_fixtures import (
    _N_SAE,
    make_token_boundary_env,
)

_BACKEND_PATCH = "scripts.preprocess.precompute_token_boundaries.get_tokenizer"


# ---------------------------------------------------------------------------
# Stats builder (a REAL ExtractionStatistics; precompute_percentiles.run() calls
# load_statistics + build_percentile_table, so a stub would not survive).
# ---------------------------------------------------------------------------


def _linear_percentiles(lo: float, hi: float) -> dict[str, float]:
    """Strictly-monotone p1..p99 (build_percentile_table rejects non-monotone rows)."""
    return {f"p{k}": lo + (hi - lo) * (k - 1) / 98.0 for k in range(1, 100)}


def _make_stat(fid: int) -> PerFeatureStat:
    # Distinct, strictly-increasing range per feature so the table is well-formed
    # and the magnitude→percentile lookup is exercised for every feature id.
    lo = 0.01 * (fid + 1)
    hi = lo + 10.0
    pcts = _linear_percentiles(lo, hi)
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=pcts["p50"],
        mean_mag_incl_zeros=pcts["p50"] * 0.5,
        std_mag=1.0,
        min_mag=pcts["p1"],
        max_mag=pcts["p99"],
        median_mag=pcts["p50"],
        percentiles=pcts,
    )


def _write_real_stats(feature_dir: Path, n_sae: int, segment_size: int) -> Path:
    """Build + persist a minimal-but-REAL stats file at feature_dir/feature_statistics.safetensors.

    Every feature id 0..n_sae-1 gets a monotone stat so build_percentile_table fills a real row for
    each id used by the fixture shards. ``n_sae_features`` matches the fixture's _N_SAE so the
    percentile table row count covers every feature id the base shards reference.
    """
    per_feature_stats = {str(fid): _make_stat(fid) for fid in range(n_sae)}
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="mean_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=segment_size,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae,
            generated_at="2026-06-15T00:00:00Z",
            activation_percentile=95.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=2,
            n_tokens_gpt2=100,
            mean_tokens_per_doc=50.0,
            median_tokens_per_doc=50,
        ),
        features=SegmentFeatureStats(
            n_segments=4,
            n_features_seen=n_sae,
            mean_active_per_segment=3.0,
            median_active_per_segment=3,
            p90_active_per_segment=3,
            max_active_per_segment=3,
            min_active_per_segment=3,
            n_total_sae_features=n_sae,
            pct_features_active=1.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=100,
            mean_bytes_per_doc=10.0,
            mean_bytes_per_token=0.1,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=int(1e10),
            estimated_docs=int(1e7),
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )
    return save_statistics(stats, feature_dir / "feature_statistics")


# ---------------------------------------------------------------------------
# Spy on the canonical streamer (the teeth).
# ---------------------------------------------------------------------------


def _install_stream_spy(monkeypatch) -> dict:
    """Wrap overlay_io._stream_base_hash so we can count real streaming passes.

    Patches the symbol the token-boundaries module dereferences. The producer imports
    ``_compute_base_hash`` (not ``_stream_base_hash``) and ``_compute_base_hash`` calls
    ``_stream_base_hash`` via the module global, so patching it on ``overlay_io`` is what the
    miss-path actually invokes. count==0 ⇒ no streaming pass ⇒ cache hit.
    """
    state = {"count": 0}
    real = overlay_io._stream_base_hash

    def spy(sorted_paths):
        state["count"] += 1
        return real(sorted_paths)

    monkeypatch.setattr(overlay_io, "_stream_base_hash", spy)
    return state


def _cfg_pct(env: dict, **overrides) -> DictConfig:
    base = {
        "feature_dir": env["feature_dir"],
        "segment_size": env["segment_size"],
        "device": "cpu",
        "dry_run": False,
        "resume": False,
        "force": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _cfg_tok(env: dict, **overrides) -> DictConfig:
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


def _run_tok(env: dict, **overrides) -> int:
    with patch(_BACKEND_PATCH, return_value=env["tokenizer"]):
        return run_tok(_cfg_tok(env, **overrides))


def _pct_overlay_dir(env: dict, stats_path: Path) -> Path:
    stats = load_statistics(stats_path)
    table = build_percentile_table(stats.per_feature_stats, stats.metadata.n_sae_features)
    checksum = compute_percentile_checksum(table)
    return overlay_dir(Path(env["feature_dir"]), OverlayKind.PERCENTILE, checksum)


def _tok_overlay_dir(env: dict) -> Path:
    checksum = compute_tokenizer_checksum(env["tokenizer"])
    return overlay_dir(Path(env["feature_dir"]), OverlayKind.SEG_BOUNDARY, checksum)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cross_writer_cache_interop_and_byte_identity(tmp_path, monkeypatch):
    """percentiles (WRITER A) populates the feature_dir cache; token_boundaries (WRITER B) HITS it.

    Asserts, in order:
      1. WRITER A creates the cache dotfile.
      2. WRITER B does ZERO streaming passes (spy==0) — it consumed A's cache.
      3. Each producer's manifest base_shard_content_hash == the cache_dir=None streaming
         reference over the SAME on-disk shards (cross-writer byte-identity, both manifests).
      4. The two manifests agree on the base-shard hash (same fixture ⇒ same provenance).
      5. A written overlay shard from each producer re-loads to a non-empty tensor dict.
    """
    # Clear DDP env so percentiles' rank-0-only manifest write (which populates the cache) runs.
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)

    env = make_token_boundary_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    cache_path = feature_dir / BASE_SHARD_HASH_CACHE_FILENAME
    assert not cache_path.exists(), "fixture must start with no base-shard hash cache"

    stats_path = _write_real_stats(feature_dir, _N_SAE, env["segment_size"])
    assert stats_path.is_file()

    # --- WRITER A: percentiles. Populates feature_dir/.base_shard_hash_cache.json. ---
    assert run_pct(_cfg_pct(env)) == 0
    assert cache_path.is_file(), (
        "WRITER A (percentiles) must populate the feature_dir base-shard hash cache"
    )

    # Reference hash, streamed over the SAME on-disk base shards (no cache).
    base_paths = glob_features(feature_dir)
    assert len(base_paths) == 2
    ref_hash = compute_base_shard_content_hash(base_paths, cache_dir=None)

    # --- WRITER B: token_boundaries, under the streaming spy. Must HIT A's cache. ---
    spy = _install_stream_spy(monkeypatch)
    assert _run_tok(env) == 0
    assert spy["count"] == 0, (
        "WRITER B (token_boundaries) computed the base hash but DID NOT hit the cache "
        f"WRITER A populated: {spy['count']} streaming pass(es). Cross-writer interop is broken."
    )

    # --- Byte-identity: both manifests' base_shard_content_hash == streaming reference. ---
    pct_manifest = read_manifest(_pct_overlay_dir(env, stats_path))
    tok_manifest = read_manifest(_tok_overlay_dir(env))
    assert pct_manifest is not None and tok_manifest is not None
    assert pct_manifest["base_shard_content_hash"] == ref_hash, (
        "percentiles manifest base hash diverged from the cache_dir=None streaming reference"
    )
    assert tok_manifest["base_shard_content_hash"] == ref_hash, (
        "token_boundaries manifest base hash (consumed from the cache) diverged from the "
        "cache_dir=None streaming reference"
    )
    assert pct_manifest["base_shard_content_hash"] == tok_manifest["base_shard_content_hash"], (
        "the two producers disagree on the base-shard hash for one shared fixture"
    )
    assert pct_manifest.get("schema_version") == 2
    assert tok_manifest.get("schema_version") == 2

    # --- Overlay shards re-load to non-empty tensor dicts (both producers wrote real data). ---
    pct_shards = glob_overlay_shards(_pct_overlay_dir(env, stats_path))
    tok_shards = glob_overlay_shards(_tok_overlay_dir(env))
    assert len(pct_shards) == 2, f"expected 2 percentile overlay shards, got {len(pct_shards)}"
    assert len(tok_shards) == 2, f"expected 2 token-boundary overlay shards, got {len(tok_shards)}"
    pct_tensors = load_overlay_shard(pct_shards[0])
    tok_tensors = load_overlay_shard(tok_shards[0])
    assert pct_tensors and all(v.numel() > 0 for v in pct_tensors.values()), (
        "percentile overlay shard re-loaded empty"
    )
    assert tok_tensors and all(v.numel() > 0 for v in tok_tensors.values()), (
        "token-boundary overlay shard re-loaded empty"
    )


@pytest.mark.slow
def test_writer_b_streams_when_cache_absent(tmp_path, monkeypatch):
    """Non-vacuity guard: with NO pre-populated cache, WRITER B's spy fires (count >= 1).

    Proves the spy-count==0 in the interop test is a real, earned cache hit — not a spy that can
    never fire. Here we run only token_boundaries (no percentiles to seed the cache) and delete any
    cache it would self-write before the measured run, so its unconditional ``_compute_base_hash``
    must stream the corpus at least once.
    """
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)

    env = make_token_boundary_env(tmp_path, n_shards=2)
    cache_path = Path(env["feature_dir"]) / BASE_SHARD_HASH_CACHE_FILENAME
    cache_path.unlink(missing_ok=True)
    assert not cache_path.exists()

    spy = _install_stream_spy(monkeypatch)
    assert _run_tok(env) == 0
    assert spy["count"] >= 1, (
        "with no pre-populated cache, token_boundaries MUST stream the base shards at least once; "
        "a spy that cannot fire would make the interop test's count==0 vacuous"
    )
    # And it self-wrote the cache on that miss.
    assert cache_path.is_file()
