"""Public project preparation preserves the original canonical token pool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from nanochat.tokenizer import SPECIAL_TOKENS
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell
from tests.axbench_adv.gcg_pool_reference import original_project_allowed_token_ids
from tests.axbench_adv.unit.test_gcg_token_pool import (
    _TEXT_POLICY_CASES,
    _ascii_strings,
    _TextPolicyTokenizer,
)


def _prepare_pool(monkeypatch: pytest.MonkeyPatch, tokenizer: _TextPolicyTokenizer) -> Any:
    cell = object.__new__(_BoundSelfConditioningCell)
    cell._runtime = cast(
        Any,
        SimpleNamespace(
            _target_model=object(),
            _target_tokenizer=tokenizer,
            _gcg_activation_directions={},
        ),
    )
    cell._concept_id = 0
    cell._method = "SelfConditioning"
    cell._factor = 1.0
    row = SimpleNamespace(
        prompt_prefix=(1,), clean_prompt=(1,), response=(2,), source_slot=0, target_slot=1
    )
    monkeypatch.setattr(
        cell, "_prepare_gcg_row", lambda **_kwargs: (row, tuple(tokenizer.encode("!")))
    )
    monkeypatch.setattr(
        self_conditioning, "_project_embedding_weight", lambda _model: t.zeros(1, 2)
    )
    sample = {"sample_id": 1, "prompt": "prompt", "response": "response"}
    prepared = cell.prepare_gcg(
        objective="response_margin",
        alpha=0.01,
        rows=[{"pair_id": 0, "positive": sample, "negative": sample}],
        suffix_token_count=1,
        device_batch_size=1,
        separator="\n\n",
    )
    assert cell._gcg_allowed == frozenset(prepared.allowed_token_ids)
    assert prepared.seed_token_ids == tuple(tokenizer.encode("!"))
    return prepared.allowed_token_ids


def test_project_pool_preserves_every_ascii_string_up_to_two_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = _ascii_strings()
    tokenizer = _TextPolicyTokenizer(texts)
    expected = tuple(
        index
        for index, text in enumerate(texts)
        if text and all(32 <= ord(character) <= 126 for character in text)
    )
    assert len(expected) == 9120
    assert _prepare_pool(monkeypatch, tokenizer) == expected
    assert original_project_allowed_token_ids(tokenizer, SPECIAL_TOKENS) == expected


@pytest.mark.parametrize("text,baseline_allowed,project_allowed", _TEXT_POLICY_CASES)
def test_project_pool_preserves_unicode_and_string_subclass_policy(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    baseline_allowed: bool,
    project_allowed: bool,
) -> None:
    tokenizer = _TextPolicyTokenizer(("!", text))
    expected = (0, 1) if project_allowed else (0,)
    assert _prepare_pool(monkeypatch, tokenizer) == expected
    assert original_project_allowed_token_ids(tokenizer, SPECIAL_TOKENS) == expected
