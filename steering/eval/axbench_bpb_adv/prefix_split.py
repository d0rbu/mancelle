"""Deterministic per-concept optimize / held-out prefix sub-split.

Splits a list of :class:`~steering.eval.axbench_bpb_adv.attack_rows.BpbAttackPrefix`
objects for one ``(concept_id, mode)`` group into an *optimize* partition and a
*held-out* partition.  The shuffle is seeded from the stable triple
``(concept_id, mode, seed)`` via SHA-256 so:

- The same group and caller seed always produce the exact same partition.
- Different ``concept_id`` or ``mode`` values produce independent shuffles even
  when the caller passes the same ``seed``.
- No global RNG state is mutated; each call uses its own :class:`random.Random`
  instance.

Python's built-in :func:`hash` is intentionally avoided because it is
per-process randomised (``PYTHONHASHSEED``).  Instead we derive the integer
seed with :func:`hashlib.sha256`.
"""

from __future__ import annotations

import hashlib
import random

from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def _derive_rng_seed(concept_id: int, mode: object, seed: int) -> int:
    """Produce a stable 64-bit integer seed from ``(concept_id, mode, seed)``.

    Uses SHA-256 over the UTF-8 encoded pipe-separated triple so the result is
    identical across Python processes regardless of ``PYTHONHASHSEED``.
    """
    payload = f"{int(concept_id)}|{mode}|{int(seed)}".encode()
    digest = hashlib.sha256(payload).digest()
    # Take the first 8 bytes as a little-endian unsigned integer — this gives
    # a 64-bit seed with uniform coverage of :class:`random.Random`'s state.
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def order_prefixes_for_concept(
    prefixes: list[BpbAttackPrefix],
    *,
    seed: int,
) -> list[BpbAttackPrefix]:
    """Deterministically order one ``(concept_id, mode)`` group's prefixes.

    Returns the group's prefixes shuffled by the same stable, per-group
    ``(concept_id, mode, seed)`` SHA-256 RNG :func:`split_prefixes_for_concept`
    uses, so the two share a single ordering primitive.  Unlike the split, an
    empty or single-element group is a valid input and passes through unchanged
    (a 0/1-element sequence has exactly one ordering) — this is what the roofline
    derive needs to pick a reproducible subsample without requiring a held-out
    split.

    Parameters
    ----------
    prefixes:
        All prefixes belonging to a single ``(concept_id, mode)`` group.
    seed:
        Caller-provided reproducibility seed, combined with ``concept_id`` and
        ``mode`` to derive the per-group RNG seed.
    """
    n = len(prefixes)
    if n < 2:
        return list(prefixes)

    # Derive a stable, per-group RNG seed — concept_id and mode are taken from
    # the first element; the caller guarantees all elements share the same group.
    concept_id = prefixes[0].concept_id
    mode = prefixes[0].mode
    rng_seed = _derive_rng_seed(concept_id, mode, seed)
    rng = random.Random(rng_seed)

    shuffled = list(prefixes)
    rng.shuffle(shuffled)
    return shuffled


def split_prefixes_for_concept(
    prefixes: list[BpbAttackPrefix],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[BpbAttackPrefix], list[BpbAttackPrefix]]:
    """Shuffle and split one ``(concept_id, mode)`` group's prefixes.

    Parameters
    ----------
    prefixes:
        All prefixes belonging to a single ``(concept_id, mode)`` group.
    ratio:
        Fraction of prefixes to place in the *optimize* partition.
        ``n_optimize = int(len(prefixes) * ratio)``, mirroring the split
        mechanics used in :mod:`steering.eval.axbench_base.seed_pools`.
    seed:
        Caller-provided reproducibility seed.  Combined with ``concept_id``
        and ``mode`` to derive the per-group RNG seed, so two groups with
        different identities always shuffle independently.

    Returns
    -------
    (optimize, held_out):
        Two disjoint lists whose union equals ``prefixes``.
        ``len(optimize) == int(len(prefixes) * ratio)``.

    Degenerate case
    ---------------
    When ``len(prefixes) < 2`` the function **raises**
    :class:`~steering.eval.axbench_bpb_adv.errors.AxBenchBpbAdvError`: a single (or
    zero) prefix cannot be split into disjoint optimize / held-out subsets, so a
    caller reaching this state has a real data/config problem rather than a
    routine empty case.  A caller that only needs a deterministic ordering (no
    held-out split — e.g. the roofline derive) must use
    :func:`order_prefixes_for_concept`, which tolerates the 0/1-element case.
    """
    n = len(prefixes)

    if n < 2:
        concept_ctx = (
            f" (concept_id={prefixes[0].concept_id}, mode={prefixes[0].mode.value})"
            if n == 1
            else ""
        )
        raise AxBenchBpbAdvError(
            f"The AxBench-BPB-Adv held-out evaluation split needs >= 2 prefixes "
            f"per (concept, mode) "
            f"to form an optimize / held-out split, but got {n} prefix(es){concept_ctx}. "
            "Ensure the source corpus has >= 2 accepted prefixes for that concept "
            "and mode, or exclude the concept from the run."
        )

    shuffled = order_prefixes_for_concept(prefixes, seed=seed)
    n_optimize = int(n * ratio)
    return (shuffled[:n_optimize], shuffled[n_optimize:])
