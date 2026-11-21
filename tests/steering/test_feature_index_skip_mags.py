"""O4 (issue #531): FeatureIndex(skip_feature_mags=True) — subset-load that skips the large
feature_mags tensor for the packed-rows path. Proves: (1) every SegmentFeatures field except
feature_mags is identical to a full-load index, with mags empty; (2) the default/live index
still loads real mags (byte-identical, unchanged); (3) packed-row output is byte-identical
whether the index skips mags or not (run() uses the skip index in production)."""
from __future__ import annotations

import torch as t

from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from steering.conditional.feature_index import FeatureIndex
from tests.scripts._packed_rows_fixtures import (
    SimpleTokenizer,
    _make_precompute_env,
    _shard_bytes,
)


def _index(env, *, skip_feature_mags, prefetch=False):
    return FeatureIndex(
        env["feature_dir"],
        prefetch=prefetch,
        percentile_checksum=env["pct_checksum"],
        tokenizer_checksum=env["tok_checksum"],
        skip_feature_mags=skip_feature_mags,
    )


def test_skip_mags_matches_full_except_mags(tmp_path):
    env = _make_precompute_env(tmp_path, n_shards=2)
    full = _index(env, skip_feature_mags=False)
    skip = _index(env, skip_feature_mags=True)

    stem = "shard_00000"
    full_docs = full.doc_ids_for_shard(stem)
    skip_docs = skip.doc_ids_for_shard(stem)
    assert full_docs == skip_docs and full_docs

    compared = 0
    for doc_id in full_docs:
        fsegs = full.lookup(stem, doc_id)
        ssegs = skip.lookup(stem, doc_id)
        assert len(fsegs) == len(ssegs)
        for fs, ss in zip(fsegs, ssegs, strict=True):
            # Everything except mags is identical.
            assert t.equal(fs.feature_ids, ss.feature_ids)
            assert fs.char_start == ss.char_start and fs.char_end == ss.char_end
            assert fs.token_start == ss.token_start and fs.token_end == ss.token_end
            assert fs.segment_idx == ss.segment_idx and fs.position == ss.position
            if fs.feature_percentiles is None:
                assert ss.feature_percentiles is None
            else:
                assert t.equal(fs.feature_percentiles, ss.feature_percentiles)
            # Full index has real mags; skip index has an empty tensor.
            assert fs.feature_mags.numel() > 0
            assert ss.feature_mags.numel() == 0
            assert ss.feature_mags.dtype == t.float32
            compared += 1
    assert compared > 0


def test_load_profiling_is_additive_and_preserves_exact_segments(tmp_path):
    env = _make_precompute_env(tmp_path, n_shards=2)
    baseline = _index(env, skip_feature_mags=True)
    profiled = FeatureIndex(
        env["feature_dir"],
        percentile_checksum=env["pct_checksum"],
        tokenizer_checksum=env["tok_checksum"],
        skip_feature_mags=True,
        profile_loads=True,
    )

    stem = "shard_00000"
    doc_id = baseline.doc_ids_for_shard(stem)[0]
    expected = baseline.lookup(stem, doc_id)
    actual = profiled.lookup(stem, doc_id)
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual, strict=True):
        assert left.segment_idx == right.segment_idx
        assert left.position == right.position
        assert left.char_start == right.char_start
        assert left.char_end == right.char_end
        assert left.token_start == right.token_start
        assert left.token_end == right.token_end
        assert t.equal(left.feature_ids, right.feature_ids)
        assert left.feature_percentiles is not None
        assert right.feature_percentiles is not None
        assert t.equal(left.feature_percentiles, right.feature_percentiles)

    records = profiled.load_profiles
    assert [record["phase"] for record in records] == [
        "scan_validation",
        "resident_load",
    ]
    resident = records[1]
    assert resident["shard_stem"] == stem
    assert resident["n_segments"] > 0
    assert resident["n_features"] > 0
    assert resident["n_docs"] > 0
    assert resident["total_seconds"] > 0
    assert resident["compressed_bytes"] > 0
    assert resident["decompressed_bytes"] > resident["compressed_bytes"]
    assert resident["detail_seconds"]["doc_index"] > 0

    # The public accessor returns a copy: benchmark/report code cannot mutate
    # later producer evidence by editing a previously returned record.
    records[0]["detail_seconds"]["injected"] = 1.0
    assert "injected" not in profiled.load_profiles[0]["detail_seconds"]


def test_skip_mags_shares_readonly_empty_singleton(tmp_path):
    """O4 (issue #531 B-A3/D-D1): on a skip_feature_mags index every segment's feature_mags is the
    SAME shared module-level empty tensor (_EMPTY_MAGS), not a fresh per-segment empty. Pin both the
    singleton identity (so a regression allocating n_segments empties — pure waste — is caught) and
    the read-only contract (numel 0, float32). The aliasing is only SAFE because the lone production
    skip-mags consumer (_seg_features_to_packed) never reads mags; this test documents that contract so
    a future in-place mutation of the shared tensor cannot slip in unnoticed."""
    from steering.conditional.feature_index import _EMPTY_MAGS

    env = _make_precompute_env(tmp_path, n_shards=2)
    skip = _index(env, skip_feature_mags=True)
    stem = "shard_00000"
    seen = 0
    for doc_id in skip.doc_ids_for_shard(stem):
        for seg in skip.lookup(stem, doc_id):
            assert seg.feature_mags is _EMPTY_MAGS, "skip-mags segs must share the _EMPTY_MAGS singleton"
            assert seg.feature_mags.numel() == 0
            assert seg.feature_mags.dtype == t.float32
            seen += 1
    assert seen > 0


def test_live_default_index_loads_real_mags(tmp_path):
    """The default (live-dataloader) path is unchanged: real, non-empty mags."""
    env = _make_precompute_env(tmp_path, n_shards=2)
    live = _index(env, skip_feature_mags=False)  # default
    assert live._skip_feature_mags is False
    segs = live.lookup("shard_00000", live.doc_ids_for_shard("shard_00000")[0])
    assert all(s.feature_mags.numel() == s.feature_ids.numel() for s in segs)
    assert any(s.feature_mags.numel() > 0 for s in segs)


def test_packed_rows_byte_identical_skip_vs_full(tmp_path):
    """run() uses skip_feature_mags=True; its packed-row shards must be byte-identical to the
    full-load path the golden was captured with."""
    env = _make_precompute_env(tmp_path, n_shards=4)

    def run(skip):
        out = tmp_path / ("skip" if skip else "full")
        out.mkdir()
        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=SimpleTokenizer(),
            feature_index=_index(env, skip_feature_mags=skip),
            context_len=32,
            output_dir=out,
            buffer_size=20,
            rows_per_shard=3,
        )
        return _shard_bytes(out)

    assert run(skip=True) == run(skip=False)


def test_packed_rows_byte_identical_production_index_config(tmp_path):
    """O2+O4: the production index (prefetch=True, skip_feature_mags=True) must produce
    byte-identical packed-row shards to the baseline (prefetch=False, skip=False)."""
    env = _make_precompute_env(tmp_path, n_shards=4)

    def run(name, **idx_kwargs):
        out = tmp_path / name
        out.mkdir()
        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=SimpleTokenizer(),
            feature_index=_index(env, **idx_kwargs),
            context_len=32,
            output_dir=out,
            buffer_size=20,
            rows_per_shard=3,
        )
        return _shard_bytes(out)

    baseline = run("baseline", skip_feature_mags=False, prefetch=False)
    production = run("production", skip_feature_mags=True, prefetch=True)
    assert production == baseline


def test_prefetch_is_invoked_when_enabled(tmp_path, monkeypatch):
    """O2 wiring: a prefetch index started a background prefetch as the doc stream crosses
    base shards (so the next shard's load overlaps the current shard's packing)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    idx = _index(env, skip_feature_mags=True, prefetch=True)

    calls: list[str] = []
    real_schedule = idx._schedule_prefetch

    def spy(stem):
        calls.append(stem)
        return real_schedule(stem)

    monkeypatch.setattr(idx, "_schedule_prefetch", spy)

    out = tmp_path / "out"
    out.mkdir()
    precompute_packed_rows(
        parquet_paths=env["parquet_paths"],
        tokenizer=SimpleTokenizer(),
        feature_index=idx,
        context_len=32,
        output_dir=out,
        buffer_size=20,
        rows_per_shard=3,
    )
    assert calls, "prefetch was never started despite prefetch=True crossing base shards"
