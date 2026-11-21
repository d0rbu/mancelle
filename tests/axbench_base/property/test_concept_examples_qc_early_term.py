"""Property-based tests for ordered ``ConceptExamplesQC.filter`` early termination.

These invariants must hold for every (n_candidates, judge_pattern,
target_surviving, caller-owned order, max_workers) tuple Hypothesis can
generate. They complement the per-path unit tests in
``tests/axbench_base/unit/test_concept_examples_qc.py`` and guard
against whole classes of regressions (wrong-subset, duplicate-entries,
over-processing).
"""

from __future__ import annotations

import random
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.concept_examples_qc import (
    ActivatingExample,
    ConceptExamplesQC,
)


class _NoopTokenizer:
    """Minimal tokenizer stub; ``filter`` never actually tokenizes under a stub fetch_fn."""

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


def _ex() -> ActivatingExample:
    return ActivatingExample(
        tokens=["t"], text="t", max_value=1.0,
        max_value_token_index=0, peak_char_offset=0,
    )


def _qc(
    judge, *, max_workers: int = 4, fetch_fn: Any | None = None,
) -> ConceptExamplesQC:
    return ConceptExamplesQC(
        model_id="m", sae_source="s", n_examples=1, judge=judge,
        tokenizer=_NoopTokenizer(), min_score=1,
        fetch_fn=fetch_fn, max_workers=max_workers,
    )


def _fetch_all_pass(model_id, sae_source, fid, n_examples, tokenizer):
    return [_ex()]


def _shuffled_candidates(
    candidates: list[tuple[int, str]],
    seed: int | None,
) -> list[tuple[int, str]]:
    if seed is None:
        return candidates
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled


@given(
    n_candidates=st.integers(min_value=0, max_value=30),
    target=st.integers(min_value=0, max_value=30),
    seed=st.one_of(st.none(), st.integers(min_value=0, max_value=999)),
    max_workers=st.integers(min_value=1, max_value=8),
)
def test_survivors_never_exceed_candidates(
    n_candidates, target, seed, max_workers,
):
    """Invariant: ``len(survivors) <= len(candidates)``.

    Trivial-but-catches-off-by-one. A shuffle bug that duplicates
    entries, or an early-term bug that double-counts, would violate
    this.
    """
    qc = _qc(
        judge=lambda d, e: 2, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    candidates = [(i, f"d{i}") for i in range(n_candidates)]
    survivors, _report = qc.filter(
        _shuffled_candidates(candidates, seed),
        target_surviving=target,
    )
    assert len(survivors) <= len(candidates)


@given(
    n_candidates=st.integers(min_value=1, max_value=30),
    target=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=999),
    max_workers=st.integers(min_value=1, max_value=8),
)
def test_all_pass_survivors_is_prefix_of_shuffled_order(
    n_candidates, target, seed, max_workers,
):
    """Invariant: when every concept passes, the survivors are exactly
    the first min(target, n) elements of the seeded permutation.

    This pins the sampling semantics: early termination is equivalent
    to "take the first target elements in shuffled order", which is
    statistically the same as "shuffle-then-truncate" but without
    the wasted work on the truncated tail.
    """
    qc = _qc(
        judge=lambda d, e: 2, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    candidates = [(i, f"d{i}") for i in range(n_candidates)]
    survivors, _r = qc.filter(
        _shuffled_candidates(candidates, seed),
        target_surviving=target,
    )
    expected_count = min(target, n_candidates)
    # In the all-pass case, we're allowed to overshoot by up to
    # (batch_size - 1) because the final batch's fetches all complete
    # and judges run serially — all results within the final batch
    # are appended before the outer loop checks the target.
    # That is: target <= len(survivors) <= target + max_workers - 1.
    # BUT: survivors MUST be a prefix of the shuffled permutation.
    assert len(survivors) >= expected_count
    assert len(survivors) <= min(n_candidates, expected_count + max_workers - 1)

    # Prefix property: compare against the caller-provided permutation.
    shuffled = _shuffled_candidates(candidates, seed)
    assert survivors == shuffled[: len(survivors)]


@given(
    n_candidates=st.integers(min_value=0, max_value=30),
    seed=st.integers(min_value=0, max_value=999),
    max_workers=st.integers(min_value=1, max_value=8),
)
def test_shuffle_is_a_permutation(n_candidates, seed, max_workers):
    """With no target and judge always passes, every candidate must
    appear in the survivors exactly once (no lost or duplicated rows)."""
    qc = _qc(
        judge=lambda d, e: 2, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    candidates = [(i, f"d{i}") for i in range(n_candidates)]
    survivors, _r = qc.filter(_shuffled_candidates(candidates, seed))
    assert sorted(survivors) == sorted(candidates)


@given(
    n_candidates=st.integers(min_value=1, max_value=30),
    target=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=999),
    max_workers=st.integers(min_value=1, max_value=8),
)
def test_determinism_same_inputs_same_outputs(
    n_candidates, target, seed, max_workers,
):
    """Hash consistency: two runs with identical config produce identical
    survivors. Parallel fetch must not introduce non-determinism in
    the returned list.
    """
    candidates = [(i, f"d{i}") for i in range(n_candidates)]
    qc1 = _qc(
        judge=lambda d, e: 2, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    qc2 = _qc(
        judge=lambda d, e: 2, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    ordered = _shuffled_candidates(candidates, seed)
    a, _ra = qc1.filter(ordered, target_surviving=target)
    b, _rb = qc2.filter(ordered, target_surviving=target)
    assert a == b


@given(
    n_candidates=st.integers(min_value=0, max_value=30),
    reject_bitmask=st.integers(min_value=0, max_value=(1 << 30) - 1),
    target=st.integers(min_value=0, max_value=30),
    seed=st.one_of(st.none(), st.integers(min_value=0, max_value=999)),
    max_workers=st.integers(min_value=1, max_value=5),
)
def test_n_processed_accounting(
    n_candidates, reject_bitmask, target, seed, max_workers,
):
    """Accounting invariant (post-fix):
        n_processed == len(per_feature)
        n_processed ==
            n_accepted + n_rejected + n_fetch_errors + n_judge_errors + n_empty_examples
        n_processed <= n_total (== n_candidates)
    """

    def _judge(description, examples):
        # Per-fid pass/fail pattern encoded in the bitmask.
        fid = int(description[1:])
        return 2 if (reject_bitmask >> fid) & 1 else 0

    qc = _qc(
        judge=_judge, max_workers=max_workers, fetch_fn=_fetch_all_pass,
    )
    candidates = [(i, f"d{i}") for i in range(n_candidates)]
    _survivors, report = qc.filter(
        _shuffled_candidates(candidates, seed),
        target_surviving=target,
    )
    assert report.n_processed == len(report.per_feature)
    assert report.n_processed == (
        report.n_accepted
        + report.n_rejected
        + report.n_fetch_errors
        + report.n_judge_errors
        + report.n_empty_examples
    )
    assert report.n_processed <= report.n_total
    assert report.n_total == n_candidates
