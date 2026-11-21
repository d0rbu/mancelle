"""Test-only helpers for windowed seam checker tests.

``swap_is_stable`` was previously a method on :class:`WindowedSeamChecker`
but was removed as production-dead (the production pipeline drives
``stitch_is_stable`` directly per seam slot). It lives here as a free
function so the differential/oracle equivalence test suite can keep using
it unchanged without polluting the production API.

Production location: ``steering/_roundtrip_seam.py``
Used by: tests/steering/test_roundtrip_seam_windowed.py,
         tests/steering/test_roundtrip_seam_edge.py
"""

from __future__ import annotations

from steering._roundtrip_seam import WindowedSeamChecker


def swap_is_stable(
    checker: WindowedSeamChecker,
    chain_a: list,
    chain_b: list,
    k: int,
    seg_tokens: list,
) -> bool:
    """A tail swap at boundary ``k`` is committable iff BOTH new seams are
    round-trip-stable: ``A[:k+1]·B[k+1:]`` and ``B[:k+1]·A[k+1:]``.

    Port of the former ``WindowedSeamChecker.swap_is_stable`` method.
    Delegates to ``checker.stitch_is_stable`` for each of the two new
    junctions, passing the full chain-side ids so the primitive can window
    internally.  Context-assembly semantics are identical to the original
    method body — the flatten/junction construction drives the differential
    proof in the equivalence test suite.
    """
    prefix_a = tuple(tid for s, j in chain_a[:k + 1] for tid in seg_tokens[s][j])
    suffix_b = tuple(tid for s, j in chain_b[k + 1:] for tid in seg_tokens[s][j])
    if not checker.stitch_is_stable(prefix_a, suffix_b):
        return False
    prefix_b = tuple(tid for s, j in chain_b[:k + 1] for tid in seg_tokens[s][j])
    suffix_a = tuple(tid for s, j in chain_a[k + 1:] for tid in seg_tokens[s][j])
    return checker.stitch_is_stable(prefix_b, suffix_a)
