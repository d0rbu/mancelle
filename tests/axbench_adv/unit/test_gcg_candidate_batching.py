"""Batch tokenization preserves candidate results and scalar failure ordering."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from tests.axbench_adv.gcg_filter_reference import scalar_candidate
from tests.axbench_adv.unit.test_gcg_vendor_worker import (
    GCGSession,
    _prepare,
    _runtime,
    _Tokenizer,
    _write_manifest,
)


class _BatchTokenizer(_Tokenizer):
    is_fast = True

    def __init__(self) -> None:
        self.batch_fault: str | None = None
        self.fail_negative = False
        self.fail_roundtrip = False
        self.batches: list[list[str]] = []
        self.scalar_calls: list[str] = []

    def __call__(self, text: Any, **kwargs: Any) -> Any:
        if isinstance(text, list):
            assert kwargs == {
                "add_special_tokens": all("Question:" in value for value in text),
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            self.batches.append(text.copy())
            if self.batch_fault == "raise":
                raise ValueError("batch-only failure")
            if self.batch_fault == "missing":
                return {}
            rows = [_Tokenizer.__call__(self, value, **kwargs)["input_ids"] for value in text]
            if self.batch_fault == "short":
                rows.pop()
            if self.batch_fault == "extra":
                rows.append(rows[-1])
            if self.batch_fault == "nonlist":
                return {"input_ids": tuple(rows)}
            if self.batch_fault == "bad_ids":
                rows[-1] = [True]
            return {"input_ids": rows}
        self.scalar_calls.append(text)
        if self.fail_roundtrip and text == "?":
            return {"input_ids": [ord("!")]}
        if self.fail_negative and "Question: N" in text:
            raise ValueError("late scalar failure")
        return super().__call__(text, **kwargs)


def _session(tmp_path: Path, *, k: int = 1) -> tuple[Any, _BatchTokenizer]:
    _write_manifest(tmp_path)
    tokenizer = _BatchTokenizer()
    session = GCGSession(_runtime(tmp_path, tokenizer=tokenizer))
    _prepare(session, suffix_token_count=k)
    tokenizer.batches.clear()
    tokenizer.scalar_calls.clear()
    return session, tokenizer


@pytest.mark.parametrize("k", [1, 2])
def test_batched_candidate_filter_matches_frozen_scalar_results(tmp_path: Path, k: int) -> None:
    session, tokenizer = _session(tmp_path, k=k)
    alternative = [ord("?"), *session.seed_token_ids[1:]]
    payload = {
        "candidate_token_ids": [
            list(session.seed_token_ids),
            alternative,
            [],
            [True],
            [-1],
            [128],
            [0],
            [32],
            "?",
        ]
    }
    actual = session.filter(payload)
    assert len(tokenizer.batches) == 2
    assert all(len(batch) == 2 for batch in tokenizer.batches)
    # Only standalone suffix canonicalization remains scalar for valid candidates.
    assert all("Question:" not in text for text in tokenizer.scalar_calls)
    with patch.object(GCGSession, "_candidate", scalar_candidate):
        expected = session.filter(payload)
    assert actual == expected
    assert [row["valid"] for row in actual["candidates"]] == [True, True, *([False] * 7)]
    session.close()


def test_non_roundtripping_suffix_is_rejected_before_batch_tokenization(tmp_path: Path) -> None:
    session, tokenizer = _session(tmp_path)
    tokenizer.fail_roundtrip = True
    tokens = (ord("?"),)
    actual = session._candidate(tokens)
    assert actual == ("?", "candidate decode/re-encode changes optimized token ids")
    assert tokenizer.batches == []
    assert actual == scalar_candidate(session, tokens)
    session.close()


@pytest.mark.parametrize(
    "batch_fault", ["raise", "missing", "short", "extra", "nonlist", "bad_ids"]
)
@pytest.mark.parametrize("outcome", ["valid", "first_framing", "late_exception"])
def test_batch_fallback_preserves_first_scalar_failure(
    tmp_path: Path, batch_fault: str, outcome: str
) -> None:
    session, tokenizer = _session(tmp_path)
    tokenizer.batch_fault = batch_fault
    tokenizer.fail_negative = outcome != "valid"
    if outcome == "first_framing":
        pair = session.layouts[0]
        session.layouts = (
            replace(
                pair, positive=replace(pair.positive, fixed_after=(*pair.positive.fixed_after, 1))
            ),
        )
    payload = {"candidate_token_ids": [[ord("?")]]}
    actual = session.filter(payload)
    assert len(tokenizer.batches) == 1
    if outcome == "first_framing":
        assert all("Question: N" not in text for text in tokenizer.scalar_calls)
    with patch.object(GCGSession, "_candidate", scalar_candidate):
        expected = session.filter(payload)
    assert actual == expected
    [row] = actual["candidates"]
    assert (
        row["error"]
        == {
            "valid": None,
            "first_framing": "candidate changes a fixed rendered wrapper or tail",
            "late_exception": "candidate canonicalization failed: late scalar failure",
        }[outcome]
    )
    session.close()


@pytest.mark.parametrize("side_name", ["positive", "negative"])
def test_successful_batch_preserves_framing_rejection(tmp_path: Path, side_name: str) -> None:
    session, tokenizer = _session(tmp_path)
    pair = session.layouts[0]
    side = getattr(pair, side_name)
    session.layouts = (
        replace(pair, **{side_name: replace(side, fixed_after=(*side.fixed_after, 1))}),
    )
    tokens = (ord("?"),)
    actual = session._candidate(tokens)
    assert actual == ("?", "candidate changes a fixed rendered wrapper or tail")
    assert len(tokenizer.batches) == 1
    assert actual == scalar_candidate(session, tokens)
    session.close()


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("texts", [[], [""], ["short", "longer input", "\u00e9 \U0001f600\nnext"]])
def test_real_fast_tokenizer_batch_is_exact(texts: list[str], add_special_tokens: bool) -> None:
    # Exercise the real fast-tokenizer API without a model download: UTF-8 byte
    # pieces, unequal lengths, and an actual BOS/EOS postprocessor.
    vocabulary = {token: index for index, token in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update({"[BOS]": 256, "[EOS]": 257})
    backend = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 256), ("[EOS]", 257)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]"
    )
    module = sys.modules[GCGSession.__module__]
    expected = [
        module._encode(tokenizer, text, add_special_tokens=add_special_tokens) for text in texts
    ]
    actual = list(module._encode_many(tokenizer, texts, add_special_tokens=add_special_tokens))
    assert actual == expected
    if texts and add_special_tokens:
        assert all(row[0] == 256 and row[-1] == 257 for row in actual)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_wrapper_preserves_scalar_result(tmp_path: Path, empty: bool) -> None:
    session, _tokenizer = _session(tmp_path)
    session._wrapper = None
    if empty:
        session.layouts = ()
    tokens = (ord("?"),)
    assert session._candidate(tokens) == scalar_candidate(session, tokens)
    session.close()
