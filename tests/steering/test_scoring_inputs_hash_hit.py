"""Hash-HIT branch coverage for ``parquet_row_to_scoring_input``
(steering/eval/axbench_bpb/scoring_inputs.py).

The existing regression tests in
``tests/axbench_bpb/unit/test_max_seq_len_per_target.py`` always pass
``context_segments_by_hash={}``, so ``context_segments_by_hash.get(prefix_hash, ())``
always returns the empty-tuple default — the populated-lookup branch was never
exercised. These tests populate the dict with a key equal to
``make_prefix_hash(prefix_text)`` and assert the looked-up tuple flows through
unchanged onto ``ScoringInputs.prefix_context_segments``.

They also pin the two other fields a caller relies on (``sample_id`` and
``output_byte_weights``) so a refactor of the lookup can't silently regress
identity or byte-weight wiring.
"""

from __future__ import annotations

import torch

from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.scoring_inputs import parquet_row_to_scoring_input
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import MockTokenizer


def _row(prefix_text: str, output_text: str, *, concept_id: int, content_class: str):
    return {
        "prefix_text": prefix_text,
        "output_text": output_text,
        "content_class": content_class,
        "concept_id": concept_id,
    }


def test_hash_hit_returns_looked_up_segments_object_identical() -> None:
    """When the prefix hash is present, its tuple flows through unchanged."""
    tok = MockTokenizer()
    prefix_text = "the quick brown fox"
    output_text = "jumps"
    row = _row(prefix_text, output_text, concept_id=7, content_class="concept_rich")

    looked_up = (
        ContextFeatureSegment(
            feature_ids=(3, 9),
            buckets=(1, 2),
            char_start=0,
            char_end=len(prefix_text),
        ),
        ContextFeatureSegment(
            feature_ids=(0,),
            buckets=(1,),
            char_start=4,
            char_end=9,
        ),
    )
    by_hash = {make_prefix_hash(prefix_text): looked_up}

    token_bytes = torch.arange(MockTokenizer.vocab_size, dtype=torch.int64)
    si = parquet_row_to_scoring_input(
        row,
        mode=SteeringMode.POSITIVE,
        context_segments_by_hash=by_hash,
        tokenizer=tok,
        max_seq_len=128,
        token_bytes=token_bytes,
    )

    # The HIT branch must return the exact tuple object from the map.
    assert si.prefix_context_segments is looked_up
    assert len(si.prefix_context_segments) == 2
    assert si.prefix_context_segments[0].feature_ids == (3, 9)
    # The prefix_hash field must equal the lookup key.
    assert si.prefix_hash == make_prefix_hash(prefix_text)


def test_hash_hit_sample_id_and_byte_weights_are_correct() -> None:
    """The HIT branch must not perturb sample_id or output_byte_weights."""
    tok = MockTokenizer()
    prefix_text = "context prefix here"
    output_text = "abc"
    concept_id = 42
    content_class = "concept_avoiding"
    mode = SteeringMode.NEGATIVE
    row = _row(prefix_text, output_text, concept_id=concept_id, content_class=content_class)

    looked_up = (
        ContextFeatureSegment(
            feature_ids=(5,),
            buckets=(3,),
            char_start=0,
            char_end=len(prefix_text),
        ),
    )
    by_hash = {make_prefix_hash(prefix_text): looked_up}

    # Distinct per-id byte weights so the gather is observable.
    token_bytes = torch.arange(MockTokenizer.vocab_size, dtype=torch.int64) * 2
    si = parquet_row_to_scoring_input(
        row,
        mode=mode,
        context_segments_by_hash=by_hash,
        tokenizer=tok,
        max_seq_len=256,
        token_bytes=token_bytes,
    )

    # sample_id is the deterministic SHA-derivative of the identity bundle.
    expected_sample_id = make_sample_id(
        concept_id=concept_id,
        mode=mode,
        content_class=content_class,
        prefix_text=prefix_text,
        output_text=output_text,
    )
    assert si.sample_id == expected_sample_id

    # output_byte_weights gathers token_bytes at each output token id.
    output_tokens = list(tok.encode(output_text))
    expected_weights = token_bytes[torch.tensor(output_tokens, dtype=torch.long)]
    assert torch.equal(si.output_byte_weights, expected_weights)
    assert si.output_byte_weights.tolist() == [tid * 2 for tid in output_tokens]
    assert si.n_output == len(output_tokens)

    # And the HIT segments still flow through object-identically.
    assert si.prefix_context_segments is looked_up


def test_hash_miss_still_returns_empty_tuple() -> None:
    """A populated map that lacks this prefix still yields the () default —
    proving the HIT path is keyed on the prefix hash, not mere non-emptiness."""
    tok = MockTokenizer()
    present_text = "some other prefix"
    target_text = "the prefix we score"
    row = _row(target_text, "out", concept_id=1, content_class="concept_rich")

    by_hash = {
        make_prefix_hash(present_text): (
            ContextFeatureSegment(
                feature_ids=(1,), buckets=(1,), char_start=0, char_end=3,
            ),
        )
    }
    assert make_prefix_hash(target_text) not in by_hash

    token_bytes = torch.zeros(MockTokenizer.vocab_size, dtype=torch.int64)
    si = parquet_row_to_scoring_input(
        row,
        mode=SteeringMode.POSITIVE,
        context_segments_by_hash=by_hash,
        tokenizer=tok,
        max_seq_len=128,
        token_bytes=token_bytes,
    )

    assert si.prefix_context_segments == ()
