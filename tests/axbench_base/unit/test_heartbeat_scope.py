"""Tests for the per-index (cross-call) heartbeat in
``_CandidatePrefixIndex._materialize_shard_candidates``.

The original PR-1 implementation reset ``n_considered`` to 0 on every
call, so for features with fewer than 1000 segments per shard the
heartbeat never fired. This test file pins the corrected behavior:

* heartbeats accumulate ACROSS calls (so many small features still
  produce a heartbeat once aggregate scan crosses 1000),
* a time-based fallback emits a heartbeat after a long stretch even
  if 1000 has not been crossed,
* the count-based path fires at 999/1000/1001 boundaries deterministically,
* rank > 0 is silenced via :func:`_log_helpers.is_log_rank`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch as t

from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base._stage0c_internals import (
    _CandidatePrefixIndex,
    _LoadedCandidateShard,
)
from tests.axbench_base._loguru_capture import capture_loguru_info


def _make_index_and_shard(n_segments: int) -> tuple[_CandidatePrefixIndex, _LoadedCandidateShard]:
    cfg = MagicMock()
    cfg.validation_shards = [0]
    cfg.feature_dir = "/tmp"
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_length = 64
    cfg.max_prefix_chars = 200
    cfg.min_prefix_chars = 5
    cfg.verification_candidate_scan_limit = None
    doc_ids = t.zeros(n_segments, dtype=t.int64)
    seg_ends = t.full((n_segments,), 100, dtype=t.int64)
    shard = _LoadedCandidateShard(
        shard_idx=0,
        feature_shard_path=MagicMock(),
        parquet_path=MagicMock(),
        features={
            "doc_ids": doc_ids,
            "segment_char_ends": seg_ends,
            "offsets": t.arange(n_segments + 1, dtype=t.int64),
            "feature_ids": t.zeros(n_segments, dtype=t.int32),
        },
        texts=["a" * 200],
    )
    index = _CandidatePrefixIndex(
        cfg, [], [], candidate_feature_ids={0}, base_context_window=2048,
    )
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = lambda text, add_special_tokens: [1, 2, 3, 4]
    index._tokenizer = fake_tokenizer
    return index, shard


def test_heartbeat_accumulates_across_multiple_small_features(monkeypatch):
    """Many calls, each scanning only 200 segments (< 1000), still emit
    a heartbeat once total segments scanned crosses 1000."""
    monkeypatch.delenv("RANK", raising=False)
    index, shard = _make_index_and_shard(200)
    with capture_loguru_info() as captured:
        # 6 calls × 200 segments = 1200 total
        for _ in range(6):
            index._materialize_shard_candidates(shard, 0)
    heartbeats = [line for line in captured if "heartbeat" in line.lower()]
    assert len(heartbeats) >= 1, (
        f"expected ≥1 heartbeat after 1200 cumulative segments across 6 "
        f"small-feature calls; got 0: {captured!r}"
    )


def test_heartbeat_fires_at_1000_segments_boundary(monkeypatch):
    """A single call scanning exactly 1000 segments emits exactly one
    heartbeat (at the 1000 boundary). 999 emits zero, 1001 still emits
    one (no double-count)."""
    monkeypatch.delenv("RANK", raising=False)
    for n_segments, expected_min in ((999, 0), (1000, 1), (1001, 1)):
        index, shard = _make_index_and_shard(n_segments)
        with capture_loguru_info() as captured:
            index._materialize_shard_candidates(shard, 0)
        heartbeats = [line for line in captured if "heartbeat" in line.lower()]
        assert len(heartbeats) == expected_min, (
            f"n={n_segments}: expected {expected_min} heartbeat(s), "
            f"got {len(heartbeats)}: {heartbeats!r}"
        )


def test_heartbeat_silenced_on_non_zero_rank(monkeypatch):
    """A worker on rank>0 must not emit heartbeat lines, even when the
    1000-segment boundary is crossed."""
    monkeypatch.setenv("RANK", "1")
    index, shard = _make_index_and_shard(2500)
    with capture_loguru_info() as captured:
        index._materialize_shard_candidates(shard, 0)
    heartbeats = [line for line in captured if "heartbeat" in line.lower()]
    assert heartbeats == [], (
        f"non-zero rank must not emit heartbeats; got {heartbeats!r}"
    )


def test_heartbeat_content_includes_running_counts(monkeypatch):
    """The heartbeat message must include the running ``considered=``,
    ``matched=``, and ``dropped=`` numbers so the log line is actionable
    rather than a content-free pulse."""
    monkeypatch.delenv("RANK", raising=False)
    index, shard = _make_index_and_shard(2500)
    with capture_loguru_info() as captured:
        index._materialize_shard_candidates(shard, 0)
    heartbeats = [line for line in captured if "heartbeat" in line.lower()]
    assert heartbeats, f"no heartbeats emitted: {captured!r}"
    first = heartbeats[0]
    assert "considered=1000" in first or "considered=2000" in first, (
        f"heartbeat should report exact running considered count; got {first!r}"
    )
    assert "matched=" in first
    assert "dropped=" in first


def test_heartbeat_time_fallback_fires_for_long_slow_scan(monkeypatch):
    """Even when total segments stay under 1000, a heartbeat must fire
    once enough wall time has elapsed (so a slow scan is not silent
    forever). Tested by monkey-patching the time threshold to a tiny
    value and inserting a sleep into the tokenizer hook."""
    import time as _time
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(_si, "_HEARTBEAT_SECONDS", 0.001)
    index, shard = _make_index_and_shard(50)

    real_encode_called = {"n": 0}

    def slow_encode(text, add_special_tokens):
        real_encode_called["n"] += 1
        if real_encode_called["n"] == 5:
            _time.sleep(0.01)
        return [1, 2, 3]

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = slow_encode
    index._tokenizer = fake_tokenizer

    with capture_loguru_info() as captured:
        index._materialize_shard_candidates(shard, 0)
    heartbeats = [line for line in captured if "heartbeat" in line.lower()]
    assert len(heartbeats) >= 1, (
        f"time-fallback should emit ≥1 heartbeat after the wall-time "
        f"threshold even with <1000 segments scanned; got {captured!r}"
    )
