"""Unit teeth for ``gcg_bags._prefixed_completions`` (the DiffMean text seam).

This function decides WHAT TEXT every sign-critical DiffMean warm-start SAE
forward sees, so its two contracts get direct pins here:

1. **Raw prefix text, never a decode.**  ``prefix_text`` must be the raw
   ``BpbAttackPrefix.prefix_text`` field.  Building it from
   ``tokenizer.decode(prefix.prefix_tokens)`` (the pre-fix behavior) prepends
   the nanochat BOS literal — under the REAL nanochat tokenizer that renders as
   ``'<bos>'``, which the Gemma extraction tokenizer re-encodes into a REAL BOS
   id, silently double-BOSing every warm-start forward.  The round-trip check
   (``encode_prefix(prefix_text) == prefix_tokens``) makes text/tokens drift a
   loud error instead of a silent divergence.

2. **Per-prefix pairing.**  Every completion must be paired with ITS OWN
   prefix's text (rich with rich, avoid with avoid, in stable order).  A
   pairing bug — e.g. hoisting ``prefixes[0]``'s text over a multi-prefix
   optimize subset, or swapping the rich/avoid sides — changes every DiffMean
   forward while all count-only assertions stay green; the exact-equality
   assertions here go RED instead.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.feature_diffmean import PrefixedCompletion
from steering.eval.axbench_bpb_adv.gcg_bags import _prefixed_completions
from tests.axbench_bpb_adv.tokenizer_stubs import StubNanochatTokenizer

_BOS_ID = 1  # StubNanochatTokenizer.get_bos_token_id()

# Reuse the shared digit-word round-trip nanochat stub (ONE home).
_RoundTripTokenizer = StubNanochatTokenizer


def _make_prefix(
    *,
    core_prefix_id: str,
    prefix_text: str,
    rich: list[list[int]],
    avoid: list[list[int]],
    tokenizer: _RoundTripTokenizer,
) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id=core_prefix_id,
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        prefix_text=prefix_text,
        prefix_hash=f"hash-{core_prefix_id}",
        prefix_tokens=encode_prefix(prefix_text, tokenizer),
        concept_rich_sample_ids=[f"r{i}" for i in range(len(rich))],
        concept_avoiding_sample_ids=[f"a{i}" for i in range(len(avoid))],
        concept_rich_output_tokens=rich,
        concept_avoiding_output_tokens=avoid,
    )


def test_multi_prefix_pairing_is_exact() -> None:
    """Each completion pairs with ITS OWN prefix's RAW text, in stable order.

    Two prefixes with distinct texts and distinct completions; the expected
    lists are written out in full so ANY pairing mutation (hoisted first-prefix
    text, swapped rich/avoid, reordered extends) diverges from at least one
    exact tuple.
    """
    tokenizer = _RoundTripTokenizer()
    prefix_a = _make_prefix(
        core_prefix_id="pA",
        prefix_text="10 11 13",
        rich=[[20, 21], [22]],
        avoid=[[30]],
        tokenizer=tokenizer,
    )
    prefix_b = _make_prefix(
        core_prefix_id="pB",
        prefix_text="10 12 13",
        rich=[[40]],
        avoid=[[50, 51]],
        tokenizer=tokenizer,
    )

    rich_items, avoid_items = _prefixed_completions(
        [prefix_a, prefix_b], tokenizer=tokenizer
    )

    assert rich_items == [
        PrefixedCompletion(prefix_text="10 11 13", completion_text="20 21"),
        PrefixedCompletion(prefix_text="10 11 13", completion_text="22"),
        PrefixedCompletion(prefix_text="10 12 13", completion_text="40"),
    ]
    assert avoid_items == [
        PrefixedCompletion(prefix_text="10 11 13", completion_text="30"),
        PrefixedCompletion(prefix_text="10 12 13", completion_text="50 51"),
    ]


def test_prefix_text_is_the_raw_field_not_a_decode() -> None:
    """The paired prefix text is the RAW field — no BOS literal, no decode.

    ``prefix_tokens`` is ``[BOS, *content]``, so a decode-based implementation
    (the pre-fix behavior) would produce ``"1 10 11 13"`` — the BOS literal
    leading the text, which the Gemma extraction tokenizer would re-encode
    into a real BOS id (the double-BOS trap).  The raw field has no BOS.
    """
    tokenizer = _RoundTripTokenizer()
    prefix = _make_prefix(
        core_prefix_id="p0",
        prefix_text="10 11 13",
        rich=[[20]],
        avoid=[[30]],
        tokenizer=tokenizer,
    )
    assert prefix.prefix_tokens[0] == _BOS_ID  # the trap is armed
    decoded = tokenizer.decode(prefix.prefix_tokens)
    assert decoded != prefix.prefix_text  # decode WOULD pollute

    rich_items, avoid_items = _prefixed_completions([prefix], tokenizer=tokenizer)

    assert rich_items[0].prefix_text == "10 11 13"
    assert avoid_items[0].prefix_text == "10 11 13"
    assert rich_items[0].prefix_text != decoded  # never the BOS-carrying decode


def test_tokens_text_drift_raises() -> None:
    """A prefix whose tokens do not round-trip from its text raises loudly.

    Text/tokens drift means the DiffMean warm-start (text) and the GCG members
    (tokens) would silently condition on DIFFERENT prefixes — a corrupt-parquet
    class of bug that must never pass quietly.
    """
    tokenizer = _RoundTripTokenizer()
    prefix = _make_prefix(
        core_prefix_id="pDrift",
        prefix_text="10 11 13",
        rich=[[20]],
        avoid=[[30]],
        tokenizer=tokenizer,
    )
    drifted = BpbAttackPrefix(
        core_prefix_id=prefix.core_prefix_id,
        concept_id=prefix.concept_id,
        mode=prefix.mode,
        prefix_text="99 98 97",  # text no longer matches the stored tokens
        prefix_hash=prefix.prefix_hash,
        prefix_tokens=prefix.prefix_tokens,
        concept_rich_sample_ids=prefix.concept_rich_sample_ids,
        concept_avoiding_sample_ids=prefix.concept_avoiding_sample_ids,
        concept_rich_output_tokens=prefix.concept_rich_output_tokens,
        concept_avoiding_output_tokens=prefix.concept_avoiding_output_tokens,
    )
    with pytest.raises(AxBenchBpbAdvError, match="round-trip"):
        _prefixed_completions([drifted], tokenizer=tokenizer)


def test_drift_error_names_the_prefix() -> None:
    """The drift error carries the core_prefix_id (operators need the culprit)."""
    tokenizer = _RoundTripTokenizer()
    prefix = _make_prefix(
        core_prefix_id="the-culprit",
        prefix_text="10 11",
        rich=[[20]],
        avoid=[[30]],
        tokenizer=tokenizer,
    )
    drifted = BpbAttackPrefix(
        core_prefix_id=prefix.core_prefix_id,
        concept_id=prefix.concept_id,
        mode=prefix.mode,
        prefix_text="10 11 12",  # one extra token after re-encode
        prefix_hash=prefix.prefix_hash,
        prefix_tokens=prefix.prefix_tokens,
        concept_rich_sample_ids=prefix.concept_rich_sample_ids,
        concept_avoiding_sample_ids=prefix.concept_avoiding_sample_ids,
        concept_rich_output_tokens=prefix.concept_rich_output_tokens,
        concept_avoiding_output_tokens=prefix.concept_avoiding_output_tokens,
    )
    with pytest.raises(AxBenchBpbAdvError, match="the-culprit"):
        _prefixed_completions([drifted], tokenizer=tokenizer)


def test_completions_are_decoded_from_nanochat_tokens() -> None:
    """Completions (which exist only as token lists) ARE decoded via the tokenizer."""
    tokenizer = _RoundTripTokenizer()
    prefix = _make_prefix(
        core_prefix_id="p0",
        prefix_text="10 11",
        rich=[[20, 21, 22]],
        avoid=[[30, 31]],
        tokenizer=tokenizer,
    )
    rich_items, avoid_items = _prefixed_completions([prefix], tokenizer=tokenizer)
    assert rich_items[0].completion_text == tokenizer.decode([20, 21, 22])
    assert avoid_items[0].completion_text == tokenizer.decode([30, 31])
