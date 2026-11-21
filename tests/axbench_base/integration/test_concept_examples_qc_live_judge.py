"""Live-judge integration test for the examples-QC path.

This test is the end-to-end sanity check that would have caught the
original regression: run ``ConceptExamplesQC.filter`` against the real
OpenRouter judge (``openai/gpt-5-nano`` by default), with realistic
concept-example triples, and assert that at least one passes QC. If
the judge-response parser is broken, every concept gets ``score=None``
-> ``judge_error`` and the assertion fails loud.

Gated behind ``@pytest.mark.very_slow`` and a runtime check for
``OPENROUTER_API_KEY`` so default CI (no creds) stays green.

Run manually::

    OPENROUTER_API_KEY=sk-or-... uv run pytest \\
        tests/axbench_base/integration/test_concept_examples_qc_live_judge.py \\
        -v -m very_slow

The Neuronpedia fetcher is MOCKED -- we inject three hand-picked
(description, examples) triples directly. The judge call is REAL.
That isolates this test from Neuronpedia's availability while still
exercising the full parser -> retry -> QC-outcome path that silently
rejected 16k features in production.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from steering.eval.axbench_base.concept_examples_qc import (
    ConceptExamplesQC,
    make_llm_concept_examples_judge,
)
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.feature_filter.types import ActivatingExample

# Hand-picked triples designed to elicit a range of scores from a
# well-behaved judge. Exact scores will vary across runs (temperature,
# model updates) but the BATCH should produce at least one pass --
# that's what the assertion requires.
_LIVE_JUDGE_CASES: list[tuple[str, list[str]]] = [
    # Should score 2 (strong match).
    (
        "cooking preparation instructions",
        [
            "preheat the oven to 375°F and grease a 9x13 baking dish",
            "sauté the diced onions over medium heat until translucent",
            "bring a large pot of salted water to a rolling boil for pasta",
        ],
    ),
    # Should score 2 (strong match).
    (
        "legal contract language about breach of warranty",
        [
            "the Seller hereby disclaims any warranty of merchantability",
            "in the event of any material breach, Buyer's sole remedy shall be limited to repair",
            "no warranty shall apply if the product has been subjected to misuse",
        ],
    ),
    # Should score 0 (unrelated).
    (
        "medical symptoms of appendicitis",
        [
            "the quarterback threw a perfect spiral to the wide receiver",
            "Lake Superior is the largest of the Great Lakes by surface area",
            "Python's list comprehensions offer a concise syntax",
        ],
    ),
]


def _make_example(text: str) -> ActivatingExample:
    """Wrap a plain string in an ActivatingExample."""
    tokens = text.split()
    return ActivatingExample(
        tokens=tokens,
        text=text,
        max_value=3.14,
        max_value_token_index=0,
        peak_char_offset=0,
    )


def _fake_fetch(
    model_id: str,
    sae_source: str,
    feature_id: int,
    n_examples: int,
    tokenizer: Any,
) -> list[ActivatingExample]:
    """Inject the hand-picked examples for this feature_id."""
    del model_id, sae_source, tokenizer, n_examples
    if feature_id < 0 or feature_id >= len(_LIVE_JUDGE_CASES):
        raise IndexError(f"feature_id {feature_id} out of range")
    _, examples = _LIVE_JUDGE_CASES[feature_id]
    return [_make_example(ex) for ex in examples]


@pytest.mark.very_slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set; skipping live judge test",
)
def test_concept_examples_qc_against_real_judge() -> None:
    """End-to-end live judge run on 3 hand-picked concept-example triples.

    Passes iff at least one concept scores >= 1 AND zero concepts end
    up in ``outcome='judge_error'`` from a parse failure. Both halves
    matter:

    - "at least one >= 1" catches the wholesale-rejection regression
      (the original bug: 0/16379 features passed).
    - "zero judge_errors" catches silent parse failures that would
      otherwise hide in the ``rejected`` bucket if the judge happens
      to score 0 for unrelated reasons.
    """
    llm_cfg = LLMClientConfig(requests_per_minute=60, max_concurrent=1)
    llm_client = OpenRouterLLMClient(
        llm_cfg,
        max_retries=3,
    )
    judge = make_llm_concept_examples_judge(
        llm_client,
        parse_max_retries=3,
        model=llm_cfg.model,
    )
    # ``_fake_fetch`` discards the tokenizer argument, and no other
    # code path inside ``ConceptExamplesQC.filter`` calls it, so
    # ``None`` is sufficient. A real tokenizer would only add a Hub
    # download without exercising anything new.
    qc = ConceptExamplesQC(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        n_examples=5,
        judge=judge,
        tokenizer=None,
        min_score=1,
        fetch_fn=_fake_fetch,
    )

    candidates = [
        (i, desc) for i, (desc, _) in enumerate(_LIVE_JUDGE_CASES)
    ]
    survivors, report = qc.filter(candidates)

    n_parse_errors = sum(
        1
        for e in report.per_feature
        if e.outcome == "judge_error"
        and e.error_message is not None
        and "could not parse" in e.error_message.lower()
    )
    assert n_parse_errors == 0, (
        f"{n_parse_errors}/{len(candidates)} concepts had parse-level "
        f"judge errors. This is the regression this test exists to "
        f"catch. Per-feature outcomes: "
        f"{[(e.feature_index, e.outcome, e.score) for e in report.per_feature]}"
    )
    assert len(survivors) >= 1, (
        f"0/{len(candidates)} concepts passed QC with min_score=1. "
        f"If this test consistently fails with 0 parse-errors, the "
        f"judge may have become stricter; review the report. "
        f"Per-feature outcomes: "
        f"{[(e.feature_index, e.outcome, e.score) for e in report.per_feature]}"
    )

