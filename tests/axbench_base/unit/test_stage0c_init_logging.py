"""Tests for the diagnostic logging added in Stage 0c init (PR-1)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base import extraction
from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _CandidatePrefixIndex,
    _LoadedCandidateShard,
)
from steering.eval.axbench_base.types import AxBenchBaseError, Concept
from tests.axbench_base._loguru_capture import capture_loguru_info


def _raise_axbench_base_error(*args: object, **kwargs: object) -> None:
    raise AxBenchBaseError("stub: halting after prefix-index init")


@pytest.fixture
def tiny_init_setup(tmp_path, monkeypatch):
    """Stub heavy bits so the dispatcher reaches the setup-phase logs."""
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text(json.dumps([
        {"feature_index": 0, "layer": 8, "sae_source": "x",
         "description": "fake", "corpus_count": 10,
         "description_source": "neuronpedia"},
    ]), encoding="utf-8")

    fake_concept = Concept(
        concept_id=0,
        feature_index=0,
        layer=8,
        sae_source="x",
        description="fake",
        corpus_count=10,
        description_source="neuronpedia",
    )
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda path, n_sae_features: [fake_concept])
    monkeypatch.setattr(_si, "_list_parquets_for_feature_dir",
                        lambda cfg: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda feature_dir: [])
    monkeypatch.setattr(extraction, "build_percentile_table_from_stats",
                        lambda path, n_sae_features: MagicMock())
    monkeypatch.setattr(_si, "_CandidatePrefixIndex", _raise_axbench_base_error)

    cfg = MagicMock()
    cfg.run_tag = "fake-run"
    cfg.feature_statistics_path = tmp_path / "stats.safetensors"
    cfg.verification_length = 64
    cfg.feature_dir = tmp_path
    cfg.neuronpedia_source = "x"
    return cfg, concepts_path


def test_init_setup_emits_progress_logs(tiny_init_setup, monkeypatch):
    """Each setup-phase boundary emits an INFO log line."""
    cfg, concepts_path = tiny_init_setup
    base_model = MagicMock()
    base_model.cfg.n_ctx = 2048

    target_meta = MagicMock()
    target_meta.sae_layer = 8

    with capture_loguru_info() as captured_lines:
        with pytest.raises(AxBenchBaseError):
            runner_mod.dispatch_prepare_negative_prefixes(
                cfg, concepts_path,
                base_model=base_model,
                base_sae=MagicMock(),
                target_model_meta=target_meta,
                n_sae_features=16384,
                response_cache=None,
            )

    log_text = "\n".join(captured_lines)
    assert "[stage0c] loading candidate concepts" in log_text
    assert "[stage0c] listing parquets" in log_text
    assert "[stage0c] globbing feature shards" in log_text
    assert "[stage0c] building percentile table" in log_text
    assert "[stage0c] initialising candidate prefix index" in log_text


def test_load_shard_emits_first_load_progress(tmp_path, monkeypatch):
    """First lazy load of a shard logs parquet read + feature shard read sizes."""
    # Build a minimal parquet shard
    parquet_path = tmp_path / "shard_00000.parquet"
    table = pa.table({"text": ["hello world", "second doc"]})
    pq.write_table(table, parquet_path)

    feature_shard_path = tmp_path / "shard_00000_features.safetensors"
    feature_shard_path.write_bytes(b"\x00" * 16)

    fake_features = {
        "doc_ids": MagicMock(shape=(2,)),
        "offsets": MagicMock(),
        "feature_ids": MagicMock(numel=lambda: 0),
        "segment_char_ends": MagicMock(),
    }
    monkeypatch.setattr(_si, "load_features", lambda p: fake_features)
    monkeypatch.setattr(_si, "_resolve_parquet",
                        lambda parquets, idx: parquet_path)
    monkeypatch.setattr(_si, "_resolve_feature_shard",
                        lambda fs, idx, fd: feature_shard_path)

    cfg = MagicMock()
    cfg.validation_shards = [0]
    cfg.feature_dir = str(tmp_path)
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_length = 64
    cfg.max_prefix_chars = 200
    cfg.min_prefix_chars = 5
    cfg.verification_candidate_scan_limit = None

    index = _CandidatePrefixIndex(
        cfg, [parquet_path], [feature_shard_path],
        candidate_feature_ids={0},
        base_context_window=2048,
    )

    with capture_loguru_info() as captured_lines:
        index._load_shard(0)

    log_text = "\n".join(captured_lines)
    assert "[stage0c] lazy first load of shard 0" in log_text
    # Assert the resolved file basenames land in the log line — not
    # just the tokens "parquet=" / "feature_shard=", which would
    # silently pass if the values were ``None`` / ``""``. The
    # implementation logs basenames (``.name``) for compactness.
    assert f"parquet={parquet_path.name}" in log_text, (
        f"expected ``parquet={parquet_path.name}`` in log line; "
        f"got {log_text!r}"
    )
    assert f"feature_shard={feature_shard_path.name}" in log_text, (
        f"expected ``feature_shard={feature_shard_path.name}`` in log line; "
        f"got {log_text!r}"
    )


def test_materialize_emits_heartbeat_every_1000(tmp_path, monkeypatch):
    """A long per-feature scan emits a heartbeat log every 1000 segments."""
    cfg = MagicMock()
    cfg.validation_shards = [0]
    cfg.feature_dir = str(tmp_path)
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_length = 64
    cfg.max_prefix_chars = 200
    cfg.min_prefix_chars = 5
    cfg.verification_candidate_scan_limit = None

    n = 2500
    doc_ids_t = t.zeros(n, dtype=t.int64)
    seg_ends_t = t.full((n,), 100, dtype=t.int64)
    shard = _LoadedCandidateShard(
        shard_idx=0,
        feature_shard_path=tmp_path / "fake.safetensors",
        parquet_path=tmp_path / "fake.parquet",
        features={
            "doc_ids": doc_ids_t,
            "segment_char_ends": seg_ends_t,
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

    with capture_loguru_info() as captured_lines:
        index._materialize_shard_candidates(shard, 0)

    heartbeats = [line for line in captured_lines
                  if "heartbeat" in line.lower()]
    assert len(heartbeats) >= 2, (
        f"Expected ≥2 heartbeat lines for 2500 segments; got {len(heartbeats)}: "
        f"{heartbeats}"
    )
