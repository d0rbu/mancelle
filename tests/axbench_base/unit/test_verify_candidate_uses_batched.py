"""Tests that _verify_candidate uses the batched generation kernel."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base._stage0c_internals import (
    _CONCEPT_STRIDE,
    _VERIFY_SEED_OFFSET,
    _VSEED_INCREMENT,
    _PrefixCandidate,
    _verify_candidate,
)


def test_verify_candidate_calls_batched_generate_seeds_once(monkeypatch):
    """One batched call replaces the five sequential calls."""
    cfg = MagicMock()
    cfg.n_verification_seeds = 5
    cfg.generation = MagicMock(master_seed=1, temperature=0.7, top_k_sampling=50)
    cfg.verification_length = 8
    cfg.verification_percentile_threshold = 0.5
    cfg.min_consistency_passes = 1
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_prefill_mode = "repeated"

    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=3, seg_idx=0,
        prefix_text="hello", prefix_gemma_tokens=[1, 2, 3],
    )

    batched_calls = []
    def fake_batched(base_model, token_ids, *, seeds, max_new_tokens,
                     temperature, top_k, stop_token_ids):
        batched_calls.append({
            "token_ids": token_ids,
            "seeds": list(seeds),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "stop_token_ids": stop_token_ids,
        })
        return [[10, 11, 12] for _ in seeds]

    monkeypatch.setattr(_si, "batched_generate_seeds", fake_batched)
    monkeypatch.setattr(_si, "get_extraction_tokenizer",
                        lambda tok_id: MagicMock(decode=lambda toks: "decoded"))
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda tok: set())
    monkeypatch.setattr(_si, "compute_fc_percentiles",
                        lambda *a, **kw: (50.0, 80.0))

    result = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=MagicMock(), base_sae=MagicMock(),
        percentile_table=MagicMock(),
        sae_layer=8,
    )

    assert len(batched_calls) == 1
    assert len(batched_calls[0]["seeds"]) == 5
    assert batched_calls[0]["max_new_tokens"] == 8
    assert batched_calls[0]["temperature"] == 0.7
    assert batched_calls[0]["top_k"] == 50
    assert batched_calls[0]["token_ids"] == [1, 2, 3]
    assert result is not None
    assert len(result.verification_continuations) == 5


def test_verify_candidate_seeds_match_legacy_formula(monkeypatch):
    """The seeds passed to the batched kernel match the documented
    deterministic formula: master + offset + concept_id*stride + vseed*incr + doc_idx."""
    cfg = MagicMock()
    cfg.n_verification_seeds = 3
    cfg.generation = MagicMock(master_seed=1000, temperature=0.7,
                                top_k_sampling=50)
    cfg.verification_length = 4
    cfg.verification_percentile_threshold = 0.5
    cfg.min_consistency_passes = 1
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_prefill_mode = "repeated"

    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=42, seg_idx=0,
        prefix_text="hello", prefix_gemma_tokens=[1, 2, 3],
    )

    captured_seeds: list[int] = []
    def fake_batched(base_model, token_ids, *, seeds, **kwargs):
        captured_seeds.extend(seeds)
        return [[10, 11] for _ in seeds]

    monkeypatch.setattr(_si, "batched_generate_seeds", fake_batched)
    monkeypatch.setattr(_si, "get_extraction_tokenizer",
                        lambda tok_id: MagicMock(decode=lambda toks: "decoded"))
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda tok: set())
    monkeypatch.setattr(_si, "compute_fc_percentiles",
                        lambda *a, **kw: (50.0, 80.0))

    _verify_candidate(
        cfg, candidate, f_c=0, concept_id=7,
        base_model=MagicMock(), base_sae=MagicMock(),
        percentile_table=MagicMock(), sae_layer=8,
    )

    expected = [
        1000 + _VERIFY_SEED_OFFSET + 7 * _CONCEPT_STRIDE
        + i * _VSEED_INCREMENT + 42
        for i in range(3)
    ]
    assert captured_seeds == expected


def test_verify_candidate_with_single_seed(monkeypatch):
    """n_verification_seeds=1 routes correctly: 1 seed in batched call, 1 percentile."""
    cfg = MagicMock()
    cfg.n_verification_seeds = 1
    cfg.generation = MagicMock(master_seed=1, temperature=0.7, top_k_sampling=50)
    cfg.verification_length = 4
    cfg.verification_percentile_threshold = 0.5
    cfg.min_consistency_passes = 1
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_prefill_mode = "repeated"

    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=3, seg_idx=0,
        prefix_text="hello", prefix_gemma_tokens=[1, 2, 3],
    )

    captured: dict = {}
    def fake_batched(base_model, token_ids, *, seeds, **kw):
        captured["seeds"] = list(seeds)
        return [[10, 11] for _ in seeds]

    monkeypatch.setattr(_si, "batched_generate_seeds", fake_batched)
    monkeypatch.setattr(_si, "get_extraction_tokenizer",
                        lambda tok_id: MagicMock(decode=lambda toks: "decoded"))
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda tok: set())
    monkeypatch.setattr(_si, "compute_fc_percentiles",
                        lambda *a, **kw: (50.0, 80.0))

    result = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=MagicMock(), base_sae=MagicMock(),
        percentile_table=MagicMock(), sae_layer=8,
    )

    assert len(captured["seeds"]) == 1
    assert result is not None
    assert len(result.verification_continuations) == 1


def test_verify_candidate_returns_none_when_all_seeds_fail_threshold(monkeypatch):
    """When all percentiles are below threshold, _verify_candidate returns None."""
    cfg = MagicMock()
    cfg.n_verification_seeds = 5
    cfg.generation = MagicMock(master_seed=1, temperature=0.7, top_k_sampling=50)
    cfg.verification_length = 4
    cfg.verification_percentile_threshold = 0.5  # raw threshold = 50.0
    cfg.min_consistency_passes = 3
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_prefill_mode = "repeated"

    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=3, seg_idx=0,
        prefix_text="hello", prefix_gemma_tokens=[1, 2, 3],
    )

    monkeypatch.setattr(_si, "batched_generate_seeds",
                        lambda *a, **kw: [[10, 11] for _ in kw["seeds"]])
    monkeypatch.setattr(_si, "get_extraction_tokenizer",
                        lambda tok_id: MagicMock(decode=lambda toks: "decoded"))
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda tok: set())
    # All percentiles below threshold
    monkeypatch.setattr(_si, "compute_fc_percentiles",
                        lambda *a, **kw: (10.0, 20.0))  # raw 20 < 50

    result = _verify_candidate(
        cfg, candidate, f_c=0, concept_id=0,
        base_model=MagicMock(), base_sae=MagicMock(),
        percentile_table=MagicMock(), sae_layer=8,
    )
    assert result is None


def test_verify_candidate_propagates_compute_fc_percentiles_error(monkeypatch):
    """If compute_fc_percentiles raises mid-loop, the error propagates."""
    cfg = MagicMock()
    cfg.n_verification_seeds = 5
    cfg.generation = MagicMock(master_seed=1, temperature=0.7, top_k_sampling=50)
    cfg.verification_length = 4
    cfg.verification_percentile_threshold = 0.5
    cfg.min_consistency_passes = 1
    cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    cfg.verification_prefill_mode = "repeated"

    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=3, seg_idx=0,
        prefix_text="hello", prefix_gemma_tokens=[1, 2, 3],
    )

    call_count = [0]
    def raising_compute(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 3:
            raise RuntimeError("simulated percentile failure for seed 3")
        return (50.0, 80.0)

    monkeypatch.setattr(_si, "batched_generate_seeds",
                        lambda *a, **kw: [[10, 11] for _ in kw["seeds"]])
    monkeypatch.setattr(_si, "get_extraction_tokenizer",
                        lambda tok_id: MagicMock(decode=lambda toks: "decoded"))
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda tok: set())
    monkeypatch.setattr(_si, "compute_fc_percentiles", raising_compute)

    with pytest.raises(RuntimeError, match="seed 3"):
        _verify_candidate(
            cfg, candidate, f_c=0, concept_id=0,
            base_model=MagicMock(), base_sae=MagicMock(),
            percentile_table=MagicMock(), sae_layer=8,
        )
