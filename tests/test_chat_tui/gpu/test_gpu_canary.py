"""Full-size leaf dtype and one-turn smoke canary."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch as t

from scripts.chat_tui.session import (
    ChatSession,
    ChatSessionConfig,
    GenerationEventKind,
    SegmentKind,
)

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")


def test_leaf_one_turn_dtype_canary(
    leaf_setup,
    leaf_model_tokenizer,
    extraction_stack,
) -> None:
    model, tokenizer = leaf_model_tokenizer
    session = ChatSession(
        setup=leaf_setup,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        config=ChatSessionConfig(
            self_cond_segment_size=32,
            temperature=1.0,
            top_k=50,
            seed=42,
            max_new_tokens_per_turn=8,
            deterministic_rewind=False,
        ),
        extraction=extraction_stack,
    )
    events = []

    session.send_user_turn("Hello there", events.append)

    conversation = session.conversation_tokens()
    ledger = session.ledger()
    assert len(conversation) > 1
    assert any(event.kind is GenerationEventKind.TURN_CLOSED for event in events)
    assert ledger
    selfcond = [entry for entry in ledger if entry.kind is SegmentKind.SELFCOND]
    assert selfcond and selfcond[0].features
    used, capacity = session.token_budget()
    assert used == len(conversation)
    assert capacity == model.dec_config.sequence_len
    text, spans = session.transcript_spans()
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(text)
    assert all(
        left.char_end == right.char_start for left, right in pairwise(spans)
    )
    assert "".join(text[span.char_start : span.char_end] for span in spans) == text
