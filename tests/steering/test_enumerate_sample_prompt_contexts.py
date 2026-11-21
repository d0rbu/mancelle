"""Tests for enumerate_sample_prompt_contexts (raw-text completion-eval suite)."""

from __future__ import annotations

import pytest

from steering.eval.context_enumeration import (
    enumerate_sample_prompt_contexts,
)
from tests.conftest import build_tiny_tokenizer


def test_enumerate_returns_one_record_per_prompt():
    tokenizer = build_tiny_tokenizer()
    prompts = ["alpha", "beta", "gamma"]
    records = enumerate_sample_prompt_contexts(
        prompts=prompts, tokenizer=tokenizer, max_seq_len=64,
    )
    assert len(records) == 3


def test_records_use_samples_suite():
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=["hello"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert all(r.suite == "samples" for r in records)


def test_records_carry_example_index():
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=["a", "b", "c"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert [r.example_index for r in records] == [0, 1, 2]


def test_token_ids_include_bos():
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=["abc"], tokenizer=tokenizer, max_seq_len=64,
    )
    token_ids = list(records[0].token_ids)
    expected = list(tokenizer("abc", prepend=tokenizer.get_bos_token_id()))
    assert token_ids == expected


def test_metadata_carries_prompt_text():
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=["alpha"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert records[0].metadata == {"prompt": "alpha"}


def test_token_context_key_stable_across_reruns():
    tokenizer = build_tiny_tokenizer()
    a = enumerate_sample_prompt_contexts(
        prompts=["alpha", "beta"], tokenizer=tokenizer, max_seq_len=64,
    )
    b = enumerate_sample_prompt_contexts(
        prompts=["alpha", "beta"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert [r.key for r in a] == [r.key for r in b]


def test_max_seq_len_rejection_names_offending_prompt_index():
    tokenizer = build_tiny_tokenizer()
    long_prompt = "x" * 1024
    with pytest.raises(ValueError, match=r"prompt index 1"):
        enumerate_sample_prompt_contexts(
            prompts=["a", long_prompt], tokenizer=tokenizer, max_seq_len=4,
        )


def test_empty_prompts_list_returns_empty_records():
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=[], tokenizer=tokenizer, max_seq_len=64,
    )
    assert records == []


def test_context_text_is_decoded_token_sequence():
    """context_text is what the extractor sees; must round-trip cleanly
    through the existing context_text_for_extraction convention."""
    tokenizer = build_tiny_tokenizer()
    records = enumerate_sample_prompt_contexts(
        prompts=["abc"], tokenizer=tokenizer, max_seq_len=64,
    )
    assert isinstance(records[0].context_text, str)
    assert len(records[0].context_text) > 0


def test_non_string_prompt_raises_with_index_and_type():
    tokenizer = build_tiny_tokenizer()
    with pytest.raises(ValueError, match=r"sample_prompts\[1\].*int"):
        enumerate_sample_prompt_contexts(
            prompts=["valid string", 42],  # int at index 1  # ty: ignore[invalid-argument-type]
            tokenizer=tokenizer, max_seq_len=64,
        )


def test_none_prompt_raises_with_index_and_type():
    tokenizer = build_tiny_tokenizer()
    with pytest.raises(ValueError, match=r"sample_prompts\[0\].*NoneType"):
        enumerate_sample_prompt_contexts(
            prompts=[None],  # ty: ignore[invalid-argument-type]
            tokenizer=tokenizer, max_seq_len=64,
        )
