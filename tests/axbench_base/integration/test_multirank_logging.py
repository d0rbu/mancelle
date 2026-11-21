"""Multi-rank logging integration tests for PR-1 instrumentation.

The PR-1 instrumentation gates all new ``logger.info`` calls behind
:func:`steering.eval.axbench_base._log_helpers.is_log_rank`, which
checks the ``RANK`` env var. This test runs the new logging code under
a real 2-rank ``gloo_multiprocess`` spawn and asserts:

* Rank 0 captures many log lines from
  ``_load_feature_extraction_pair`` and the heartbeat path.
* Rank 1 captures zero log lines (silenced by the rank gate).

Without these tests the rank-gating could silently regress (e.g., a
future ``logger.info(...)`` call inside the same function bypassing
the gate) and only surface as torchrun stderr spam in production.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch as t

from steering.eval.axbench_base import runtime_state as rs_mod
from steering.eval.axbench_base._stage0c_internals import (
    _CandidatePrefixIndex,
    _LoadedCandidateShard,
)
from steering.eval.axbench_base.runtime_state import (
    _load_feature_extraction_pair,
)
from tests.axbench_base._loguru_capture import capture_loguru_info


def _run_load_feature_extraction(rank: int, world_size: int) -> list[str]:
    """Worker target: stub heavy loaders and capture loguru output."""
    cfg = MagicMock()
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    cfg.neuronpedia_model_id = "google/gemma-2-2b"
    cfg.feature_extraction_model_id = "gemma-2-2b"
    cfg.device = "cpu"

    with patch.object(rs_mod, "load_sae_and_model",
                      lambda **kw: (
                          MagicMock(),
                          SimpleNamespace(to=MagicMock()),
                          MagicMock(),
                      )), \
         patch.object(rs_mod, "build_nnterp_model",
                      lambda *a, **kw: MagicMock()):
        with capture_loguru_info() as captured:
            _load_feature_extraction_pair(cfg)
    return list(captured)


def _run_heartbeat_path(rank: int, world_size: int) -> list[str]:
    """Worker target: drive _materialize_shard_candidates past the
    1000-segment heartbeat threshold and capture loguru output."""
    n = 2500
    cfg = MagicMock()
    cfg.validation_shards = [0]
    cfg.feature_dir = "/tmp"
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_length = 64
    cfg.max_prefix_chars = 200
    cfg.min_prefix_chars = 5
    cfg.verification_candidate_scan_limit = None
    doc_ids = t.zeros(n, dtype=t.int64)
    seg_ends = t.full((n,), 100, dtype=t.int64)
    shard = _LoadedCandidateShard(
        shard_idx=0,
        feature_shard_path=MagicMock(),
        parquet_path=MagicMock(),
        features={
            "doc_ids": doc_ids,
            "segment_char_ends": seg_ends,
            "offsets": t.arange(n + 1, dtype=t.int64),
            "feature_ids": t.zeros(n, dtype=t.int32),
        },
        texts=["a" * 200],
    )
    index = _CandidatePrefixIndex(
        cfg, [], [], candidate_feature_ids={0}, base_context_window=2048,
    )
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = lambda text, add_special_tokens: [1, 2, 3, 4]
    index._tokenizer = fake_tokenizer
    with capture_loguru_info() as captured:
        index._materialize_shard_candidates(shard, 0)
    return list(captured)


def test_load_feature_extraction_pair_rank_gating(gloo_multiprocess):
    """Under gloo 2-rank spawn: rank 0 emits all begin/end logs; rank 1
    emits none. Without rank-gating both ranks would spam stderr."""
    results = gloo_multiprocess(2, _run_load_feature_extraction)
    rank0_lines, rank1_lines = results[0], results[1]
    rank0_text = "\n".join(rank0_lines)
    assert "[runtime_state] resolving Neuronpedia source" in rank0_text
    assert "[runtime_state] feature-extraction pair ready" in rank0_text
    assert rank1_lines == [], (
        f"rank 1 must be silent for new instrumentation; got "
        f"{rank1_lines!r}"
    )


def test_materialize_heartbeat_rank_gating(gloo_multiprocess):
    """Under gloo 2-rank spawn: rank 0 emits heartbeat lines; rank 1
    emits none. The heartbeat gating is the most-triggered new log
    site so a regression here would dominate production stderr."""
    results = gloo_multiprocess(2, _run_heartbeat_path)
    rank0_lines, rank1_lines = results[0], results[1]
    rank0_heartbeats = [line for line in rank0_lines if "heartbeat" in line.lower()]
    rank1_heartbeats = [line for line in rank1_lines if "heartbeat" in line.lower()]
    assert len(rank0_heartbeats) >= 1, (
        f"rank 0 must emit ≥1 heartbeat for 2500 segments; got "
        f"{len(rank0_heartbeats)}: {rank0_heartbeats!r}"
    )
    assert rank1_heartbeats == [], (
        f"rank 1 must emit zero heartbeats; got {rank1_heartbeats!r}"
    )
