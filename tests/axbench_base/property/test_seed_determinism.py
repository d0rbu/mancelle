"""Property tests for seed derivation helpers.

``intervention_dispatch._sampling_seed`` controls "same input -> same
seed" behavior across the AxBench-Base pipeline: it is the Stage 4
per-call sampling seed derivation used by ``run_intervention``. The same
``(method_name, concept_id, eval_seed, master_seed)`` tuple must always
produce the same int, and DIFFERENT tuples must produce different ints
(modulo the tiny 2**-31 collision probability of a SHA256-mod-int31).

This feeds into the Stage 4 reproducibility story: a given
``(method, concept, eval_seed)`` must produce identical generations
across re-runs.
"""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from steering.eval.axbench_base.intervention_dispatch import _sampling_seed

_method_name_strategy = st.text(
    alphabet=st.characters(
        min_codepoint=0x21, max_codepoint=0x7E,
        blacklist_characters="|",
    ),
    min_size=1,
    max_size=32,
)

_concept_id_strategy = st.integers(min_value=0, max_value=10_000)
_eval_seed_strategy = st.integers(min_value=0, max_value=10_000)
_master_seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# _sampling_seed determinism
# ---------------------------------------------------------------------------


@given(
    method_name=_method_name_strategy,
    concept_id=_concept_id_strategy,
    eval_seed=_eval_seed_strategy,
    master_seed=_master_seed_strategy,
)
def test_sampling_seed_deterministic(
    method_name: str,
    concept_id: int,
    eval_seed: int,
    master_seed: int,
) -> None:
    """Same inputs -> same output every time."""
    first = _sampling_seed(
        method_name,
        concept_id=concept_id,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    second = _sampling_seed(
        method_name,
        concept_id=concept_id,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    third = _sampling_seed(
        method_name,
        concept_id=concept_id,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    assert first == second == third
    assert 0 <= first < (1 << 31)


@given(
    method_a=_method_name_strategy,
    method_b=_method_name_strategy,
    concept_id=_concept_id_strategy,
    eval_seed=_eval_seed_strategy,
    master_seed=_master_seed_strategy,
)
def test_sampling_seed_differs_on_method_name(
    method_a: str,
    method_b: str,
    concept_id: int,
    eval_seed: int,
    master_seed: int,
) -> None:
    """Changing only ``method_name`` flips the seed."""
    assume(method_a != method_b)
    seed_a = _sampling_seed(
        method_a,
        concept_id=concept_id,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    seed_b = _sampling_seed(
        method_b,
        concept_id=concept_id,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    assert seed_a != seed_b


@given(
    method_name=_method_name_strategy,
    concept_a=_concept_id_strategy,
    concept_b=_concept_id_strategy,
    eval_seed=_eval_seed_strategy,
    master_seed=_master_seed_strategy,
)
def test_sampling_seed_differs_on_concept_id(
    method_name: str,
    concept_a: int,
    concept_b: int,
    eval_seed: int,
    master_seed: int,
) -> None:
    """Changing only ``concept_id`` flips the seed."""
    assume(concept_a != concept_b)
    seed_a = _sampling_seed(
        method_name,
        concept_id=concept_a,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    seed_b = _sampling_seed(
        method_name,
        concept_id=concept_b,
        eval_seed=eval_seed,
        master_seed=master_seed,
    )
    assert seed_a != seed_b


@given(
    method_name=_method_name_strategy,
    concept_id=_concept_id_strategy,
    eval_seed_a=_eval_seed_strategy,
    eval_seed_b=_eval_seed_strategy,
    master_seed=_master_seed_strategy,
)
def test_sampling_seed_differs_on_eval_seed(
    method_name: str,
    concept_id: int,
    eval_seed_a: int,
    eval_seed_b: int,
    master_seed: int,
) -> None:
    """Changing only ``eval_seed`` flips the seed."""
    assume(eval_seed_a != eval_seed_b)
    seed_a = _sampling_seed(
        method_name,
        concept_id=concept_id,
        eval_seed=eval_seed_a,
        master_seed=master_seed,
    )
    seed_b = _sampling_seed(
        method_name,
        concept_id=concept_id,
        eval_seed=eval_seed_b,
        master_seed=master_seed,
    )
    assert seed_a != seed_b
