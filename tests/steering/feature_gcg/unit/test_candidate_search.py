from __future__ import annotations

import copy
import random

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_gcg.candidate_search import (
    deterministic_step_rng,
    rank_replacements_from_gradients,
    replacement_pool_exhausted,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.edit_primitives import (
    TokenReplacement,
    replace_single_token,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError


def test_rank_rejects_bad_top_k_at_typed_boundary() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="top_k"):
        rank_replacements_from_gradients(
            embedding_gradient=t.zeros(3, 4),
            embedding_weight=t.zeros(5, 4),
            editable_positions=[1],
            allowed_token_ids=[0, 1],
            current_tokens=[0, 1, 2],
            top_k=0,
            distance_weight=0.0,
        )


def test_deterministic_step_rng_reproducible() -> None:
    a = deterministic_step_rng(candidate_seed="g", step=2, master_seed=0)
    b = deterministic_step_rng(candidate_seed="g", step=2, master_seed=0)
    assert [a.random() for _ in range(3)] == [b.random() for _ in range(3)]


def _replacement(position: int, token_id: int, score: float = 0.0) -> TokenReplacement:
    return TokenReplacement(position=position, token_id=token_id, score=score)


def _derived_digest(
    current: list[int], replacement: TokenReplacement
) -> str:
    return token_sequence_digest(
        replace_single_token(
            current,
            position=replacement.position,
            token_id=replacement.token_id,
        )
    )


def test_replacement_pool_exhausted_only_when_every_derived_sequence_was_visited() -> None:
    current = [0, 10, 20]
    first = _replacement(1, 11)
    second = _replacement(2, 21)
    proposals = {1: (first,), 2: (second,)}
    visited = {_derived_digest(current, first), _derived_digest(current, second)}

    assert replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )

    visited.remove(_derived_digest(current, second))
    assert not replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )


def test_replacement_pool_exhausted_uses_resulting_sequence_digests() -> None:
    current = [0, 10, 20]
    duplicate = _replacement(1, 11)
    no_op = _replacement(2, 20)
    at_first_position = _replacement(1, 7)
    at_second_position = _replacement(2, 7)
    proposals = {
        1: (duplicate, duplicate, at_first_position),
        2: (no_op, at_second_position),
    }
    visited = {
        token_sequence_digest(current),
        _derived_digest(current, duplicate),
        _derived_digest(current, at_first_position),
    }

    # The duplicate and no-op are exhausted through their resulting digests, but
    # replacing position 2 with token 7 is a distinct, still-unvisited sequence.
    assert not replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )
    visited.add(_derived_digest(current, at_second_position))
    assert replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )


def test_replacement_pool_exhausted_is_read_only() -> None:
    current = [0, 10, 20]
    proposals = {1: (_replacement(1, 11),), 2: (_replacement(2, 21),)}
    visited = {_derived_digest(current, proposals[1][0])}
    before_current = copy.deepcopy(current)
    before_proposals = copy.deepcopy(proposals)
    before_visited = set(visited)

    replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )

    assert current == before_current
    assert proposals == before_proposals
    assert visited == before_visited


@pytest.mark.parametrize(
    ("position", "message"),
    [(0, "would edit BOS"), (2, "out of range")],
)
def test_replacement_pool_exhausted_preserves_invalid_coordinate_failure(
    position: int,
    message: str,
) -> None:
    with pytest.raises(FeatureGcgError, match=message):
        replacement_pool_exhausted(
            current_tokens=[0, 10],
            replacements_by_position={position: (_replacement(position, 11),)},
            visited_digests=set(),
        )


class _AlwaysFirstRandom(random.Random):
    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        return [population[0]] * k


def test_bounded_rejection_empty_is_not_exact_exhaustion() -> None:
    current = [0, 10, 20]
    visited_high_weight = _replacement(1, 11, score=0.0)
    unvisited_tiny_weight = _replacement(2, 21, score=1_000.0)
    proposals = {
        1: (visited_high_weight,),
        2: (unvisited_tiny_weight,),
    }
    visited = {_derived_digest(current, visited_high_weight)}

    sampled = sample_candidate_sequences(
        current_tokens=current,
        replacements_by_position=proposals,
        batch_size=1,
        temperature=0.1,
        rng=_AlwaysFirstRandom(0),
        visited_digests=visited,
    )

    assert sampled == []
    assert not replacement_pool_exhausted(
        current_tokens=current,
        replacements_by_position=proposals,
        visited_digests=visited,
    )


def test_empty_replacement_mapping_still_fails_in_sampler() -> None:
    with pytest.raises(FeatureGcgError, match="replacement pool is empty"):
        sample_candidate_sequences(
            current_tokens=[0, 10],
            replacements_by_position={},
            batch_size=1,
            temperature=1.0,
            rng=random.Random(0),
            visited_digests=set(),
        )
