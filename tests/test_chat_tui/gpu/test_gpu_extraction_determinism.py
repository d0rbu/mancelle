"""Real Gemma+SAE prompt-extraction determinism."""

from __future__ import annotations

import pytest
import torch as t

from scripts.chat_tui.setup import load_chat_setup
from steering.eval.chat_generation import render_prompt_for_chat_completion
from tests.test_chat_tui.gpu.conftest import LEAF_META

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")


def _contents(prepared) -> tuple:
    return (
        prepared.token_ids,
        prepared.assistant_end_id,
        tuple(
            (
                segment.feature_ids,
                segment.buckets,
                segment.char_start,
                segment.char_end,
            )
            for segment in prepared.segments
        ),
    )


def test_real_extraction_is_cache_stable_and_fresh_service_pure(
    leaf_model_tokenizer,
    extraction_service_factory,
    stats_path,
) -> None:
    _model, tokenizer = leaf_model_tokenizer
    prompt = tuple(
        render_prompt_for_chat_completion(
            tokenizer,
            "Summarize the role of replication in experimental science.",
        ).token_ids
    )
    unrelated = tuple(
        render_prompt_for_chat_completion(
            tokenizer,
            "Describe a quiet harbor at sunrise.",
        ).token_ids
    )

    first_setup = load_chat_setup(LEAF_META, stats_path)
    service = extraction_service_factory(first_setup, tokenizer)
    first = service.prepare(prompt)
    service.prepare(unrelated)
    repeated = service.prepare(prompt)

    assert repeated is first, "repeated exact tokens were not a service cache hit"
    assert _contents(repeated) == _contents(first)

    fresh_setup = load_chat_setup(LEAF_META, stats_path)
    fresh_service = extraction_service_factory(fresh_setup, tokenizer)
    # Discriminating order: the unrelated prompt goes FIRST on the fresh
    # service. Without the per-miss reseed rule its pipeline RNG would arrive
    # at the target prompt in an advanced state and the features would differ.
    fresh_service.prepare(unrelated)
    fresh = fresh_service.prepare(prompt)
    assert fresh is not first
    assert _contents(fresh) == _contents(first)
