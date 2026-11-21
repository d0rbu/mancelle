"""End-to-end: cfg.evals='samples' resolves through enumerate_expected_eval_contexts."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.eval.validate_eval_context_cache import (
    enumerate_expected_eval_contexts,
)
from tests.conftest import build_tiny_tokenizer


def _make_cfg(**overrides):
    base = {
        "evals": "samples",
        "sample_prompts": ["alpha", "beta", "gamma"],
        "model_max_seq_len": 64,
        "base_seed": 1234,
        "num_seeds": 1,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_samples_suite_dispatches_through_enumerate_expected_eval_contexts():
    tokenizer = build_tiny_tokenizer()
    cfg = _make_cfg()
    records = enumerate_expected_eval_contexts(cfg, tokenizer)
    assert len(records) == 3
    assert all(r.suite == "samples" for r in records)
    assert [r.metadata["prompt"] for r in records] == ["alpha", "beta", "gamma"]


def test_samples_suite_combined_with_other_suites_produces_concatenated_records():
    """evals='samples,harmfulness' -> records from both suites.
    Tests the comma-split path; harmfulness contexts use the existing
    harmful_prompts/benign_prompts cfg knobs."""
    tokenizer = build_tiny_tokenizer()
    cfg = _make_cfg(
        evals="samples,harmfulness",
        harmful_prompts=[],
        benign_prompts=[],
        harm_num_harmful=0,
        harm_num_benign=0,
    )
    records = enumerate_expected_eval_contexts(cfg, tokenizer)
    samples_records = [r for r in records if r.suite == "samples"]
    assert len(samples_records) == 3


def test_unknown_suite_still_raises():
    """Regression: invalid suite names continue to raise ValueError."""
    tokenizer = build_tiny_tokenizer()
    cfg = _make_cfg(evals="unknown_suite_zzz")
    with pytest.raises(ValueError, match="Unsupported eval suite"):
        enumerate_expected_eval_contexts(cfg, tokenizer)


def test_samples_suite_requires_sample_prompts_key():
    """Missing sample_prompts -> clear error."""
    tokenizer = build_tiny_tokenizer()
    cfg = OmegaConf.create({
        "evals": "samples",
        "model_max_seq_len": 64,
        "base_seed": 1234,
        "num_seeds": 1,
    })
    with pytest.raises((ValueError, Exception), match=r"sample_prompts"):
        enumerate_expected_eval_contexts(cfg, tokenizer)


def test_samples_suite_requires_non_empty_sample_prompts():
    """Missing OR empty sample_prompts -> clear error."""
    tokenizer = build_tiny_tokenizer()
    cfg = OmegaConf.create({
        "evals": "samples",
        "sample_prompts": [],  # empty list
        "model_max_seq_len": 64,
        "base_seed": 1234,
        "num_seeds": 1,
    })
    with pytest.raises(ValueError, match="sample_prompts"):
        enumerate_expected_eval_contexts(cfg, tokenizer)
