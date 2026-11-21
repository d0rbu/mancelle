"""Canonical signed class-margin sign rule, shared across the BPB pipelines.

There is ONE rule for turning a (concept_rich, concept_avoiding) loss pair into a
"higher = stronger steering" scalar:

* positive mode: ``avoid - rich``
* negative mode: ``rich - avoid`` (sign flip)

equivalently ``margin_sign(mode) * (avoid - rich)`` with ``+1`` for positive and
``-1`` for negative. The axbench_bpb_adv aggregation calls ``signed_class_margin``
here and the sibling axbench_bpb aggregation calls ``margin_sign`` here, so the
sign rule has a single canonical definition shared by both. The GCG objective's
desired/contrast ordering (``model_optimize._desired_contrast``) mirrors the same
rule independently -- kept equivalent by a cross-site identity test rather than a
call into this module -- so a divergent edit cannot silently break the design's
load-bearing identity:

    g = loss_contrast - loss_desired

equals ``signed_class_margin`` (positive mode:
``contrast - desired == avoid - rich``).

This is a LEAF module: it depends only on :class:`SteeringMode` from
``axbench_base.types`` (and the stdlib), so it is import-cycle-free for every
consumer in ``axbench_base``, ``axbench_bpb``, and ``axbench_bpb_adv``. It raises
plain :class:`ValueError` on an unknown mode rather than any package-specific
error type, to avoid pulling a downstream error class into ``axbench_base``.
"""

from __future__ import annotations

from steering.eval.axbench_base.types import SteeringMode


def margin_sign(mode: SteeringMode | str) -> float:
    """Return ``+1.0`` for positive mode, ``-1.0`` for negative mode.

    Accepts either a :class:`SteeringMode` member or its bare-string value
    (``SteeringMode`` is a :class:`~enum.StrEnum`, so both compare equal).
    Raises :class:`ValueError` on any unknown mode.
    """

    if mode == SteeringMode.POSITIVE:
        return 1.0
    if mode == SteeringMode.NEGATIVE:
        return -1.0
    raise ValueError(f"unknown steering mode {mode!r}")


def signed_class_margin(
    *,
    mode: SteeringMode | str,
    rich: float,
    avoid: float,
) -> float:
    """Signed class margin: ``margin_sign(mode) * (avoid - rich)``.

    ``rich`` is the concept_rich loss and ``avoid`` the concept_avoiding loss.
    Higher is stronger steering in both modes. Raises :class:`ValueError` on an
    unknown mode (via :func:`margin_sign`).
    """

    return margin_sign(mode) * (float(avoid) - float(rich))
