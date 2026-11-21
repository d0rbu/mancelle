"""Pure aggregation primitives shared by the GCG core and the AxBench-BPB-Adv eval.

These functions are stdlib-only (no torch, no model) and live here so
``steering/conditional/feature_gcg/core.py`` can import them without pulling
in the eval package (avoiding a steering→eval back-edge).

Callers import these primitives from this canonical module directly.
"""

from __future__ import annotations

import hashlib
import math
import random

from beartype import beartype

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.refined_types import NonNegativeInt, PositiveInt


@beartype
def minibatch_for_step(
    n_pairs: NonNegativeInt,
    *,
    step: NonNegativeInt,
    minibatch_size: PositiveInt,
    group_seed: str,
) -> list[int]:
    """Deal ``minibatch_size`` pair indices for ``step`` without replacement.

    Within an epoch (``epoch_len = ceil(n_pairs / minibatch_size)`` consecutive
    steps) every index in ``range(n_pairs)`` is dealt exactly once: the epoch
    uses a single shuffled permutation and each step takes the next contiguous
    slice. The permutation is reseeded per epoch from ``group_seed`` so epochs
    reshuffle deterministically. ``minibatch_size >= n_pairs`` yields all indices
    (in the epoch's shuffled order) every step. ``n_pairs == 0`` returns ``[]``;
    refined scalar constraints are enforced at the function boundary.
    """

    if n_pairs == 0:
        return []

    epoch_len = math.ceil(n_pairs / minibatch_size)
    epoch = step // epoch_len
    offset = (step % epoch_len) * minibatch_size

    seed_hex = hashlib.sha256(f"{group_seed}|epoch{epoch}".encode()).hexdigest()
    permutation = list(range(n_pairs))
    random.Random(seed_hex).shuffle(permutation)
    return permutation[offset : offset + minibatch_size]


def aggregate_objective(pair_g: list[float]) -> float:
    """Mean the per-pair objective values into the group's accept-decision scalar.

    Lower is better; the mean keeps the aggregate on the same scale as a single
    pair's objective, so the 1-pair
    case reduces exactly to the per-prefix accept rule. Finiteness guard (design
    review m10): if ANY pair objective is non-finite (NaN / +inf / -inf) the
    aggregate is ``+inf``. This is well-defined under the strict ``best < current``
    accept compare -- ``+inf`` is never accepted -- whereas a raw mean containing
    a NaN would be NaN, which compares ``False`` in BOTH directions and would
    silently neither accept nor reject in a way that depends on iteration order.
    Propagating ``+inf`` makes a poisoned minibatch a clean reject. Empty input
    raises.
    """

    if not pair_g:
        raise FeatureGcgError(
            "aggregate_objective requires at least one pair objective"
        )
    if any(not math.isfinite(value) for value in pair_g):
        return math.inf
    return math.fsum(pair_g) / len(pair_g)
