"""Restrict feature-id tensors to a compacted checkpoint's allowed vocabulary V (issue #695 follow-up).

A compacted encoder-decoder model's ``FeatureEncoder`` holds embedding rows only for a subset ``V``
of the full ``n_sae`` SAE features (see ``steering.conditional.feature_vocab.FeatureVocab``);
conditioning on an id outside ``V`` trips ``FeatureEncoder._to_compact``'s OOV assert. AxBench-BPB's
feature-selection sites (feature-DiffMean, general-val, context-cache features — see
``EvalConditioning.conditions_within_vocab``) must restrict their OWN candidate ids to ``V`` *before*
conditioning, rather than relying on that assert as the only backstop. This module is the shared
helper both directions of that restriction need: reading ``V`` off a loaded model, and splitting an
id tensor into the ids that survive it vs. the ids it drops.

Legacy (uncompacted, ``feature_vocab is None``) checkpoints: every restriction here is a no-op —
``feature_vocab_allowed_ids`` returns ``None``, and ``restrict_to_vocab`` with ``allowed=None`` keeps
every id unchanged. This is what makes the byte-identical legacy behavior the design requires hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from steering.eval.axbench_bpb.types import AxBenchBpbError

#: Sentinel :func:`checkpoint_vocab_fingerprint` returns for a legacy (uncompacted,
#: ``feature_vocab is None``) checkpoint -- it has no vocabulary to fingerprint, but
#: still needs a stable, comparable value so callers never special-case ``None``.
UNCOMPACTED_VOCAB_FINGERPRINT = "uncompacted"


def feature_vocab_allowed_ids(model: Any) -> torch.Tensor | None:
    """The loaded model's allowed original SAE feature ids ``V``, or ``None`` if uncompacted.

    Returns the encoder's OWN persistent ``feature_vocab_ids`` buffer (``int64[V]``, sorted
    ascending, already on the model's device) rather than re-deriving a tensor from
    ``FeatureVocab.original_ids`` — the buffer IS the checkpoint's loaded,
    ``assert_vocab_consistent``-verified record (see ``EncoderDecoderGPT.assert_vocab_consistent``),
    so there is nothing to keep in sync by recomputing it.

    ``model`` is duck-typed (mirrors ``TargetModelBundle.model: Any`` elsewhere in this eval
    layer): it must expose ``.feature_vocab`` (``FeatureVocab | None``, e.g. the property
    ``EncoderDecoderGPT.feature_vocab``) and, only when compacted, ``.encoder.feature_vocab_ids``.
    """
    if model.feature_vocab is None:
        return None
    return model.encoder.feature_vocab_ids


def checkpoint_vocab_fingerprint(model: Any) -> str:
    """The loaded model's checkpoint ``V``-fingerprint, or the uncompacted sentinel.

    A compacted encoder-decoder checkpoint's allowed feature-id set ``V`` is a
    property of ITS ``FeatureVocab`` alone (n_sae_features + the id list -- see
    ``FeatureVocab.fingerprint()``, the same sha256 already mirrored into every
    ``meta_*.json`` and cross-checked by ``build_encdec_model``). This wraps that
    EXISTING fingerprint for eval-side consumers that need to bind a derived
    artifact (e.g. a precomputed feature-conditioning bag) to the specific
    checkpoint vocabulary it was derived against, so a later read against a
    DIFFERENT checkpoint's ``V`` can be detected rather than silently served.

    ``model`` is duck-typed (mirrors :func:`feature_vocab_allowed_ids`): it must
    expose ``.feature_vocab`` (``FeatureVocab | None``). A legacy (uncompacted)
    checkpoint's ``feature_vocab`` is ``None`` -- there is no ``V`` to fingerprint,
    so this returns :data:`UNCOMPACTED_VOCAB_FINGERPRINT` rather than a sha256,
    giving callers ONE stable string to compare in every case (no ``is None``
    branch of their own).
    """
    vocab = model.feature_vocab
    if vocab is None:
        return UNCOMPACTED_VOCAB_FINGERPRINT
    return vocab.fingerprint()


def restrict_to_vocab(
    ids: torch.Tensor, allowed: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``ids`` into ``(kept, dropped)`` by membership in ``allowed``.

    Both outputs preserve ``ids``' original order and dtype, and stay on ``ids``' device.
    ``allowed=None`` means "no vocabulary restriction" (an uncompacted checkpoint): ``kept``
    is every id unchanged and ``dropped`` is empty — so callers can thread this through
    unconditionally, with no ``if compacted`` branch of their own.

    ``allowed`` is moved onto ``ids``' device before the membership test: callers routinely
    pass a CPU-built ``ids`` (e.g. a freshly ``t.tensor(...)``-constructed candidate list)
    against ``allowed`` read off a CUDA-resident model's ``feature_vocab_ids`` buffer (see
    :func:`feature_vocab_allowed_ids`), and ``torch.isin`` requires both operands on the same
    device.
    """
    if allowed is None:
        return ids, ids.new_empty((0,))
    mask = torch.isin(ids, allowed.to(ids.device))
    return ids[mask], ids[~mask]


def restrict_segment_conditions_to_vocab(
    segment_conditions: Sequence[dict[int, int]],
    allowed_ids: torch.Tensor | None,
) -> tuple[Sequence[dict[int, int]], int]:
    """Restrict each segment's ``{feature_id: bucket}`` dict to ``allowed`` ``V``.

    The shared drop/raise primitive for FIXED per-segment context conditioning
    (context-cache features, extracted BEFORE any per-method selection against
    ``V`` -- unlike a method's own conditioning bag, which is already restricted
    upstream of wherever it is threaded in).  Mirrors
    ``steering.eval.axbench_bpb.teacher_force._build_encdec_seg_map_and_encoder``'s
    context-cache filter (``teacher_force.py:306-340``) EXACTLY: for each segment,
    ids outside ``allowed_ids`` are dropped via :func:`restrict_to_vocab`; a segment
    that HAD feature ids but is left with NONE after the filter (every one of its
    ids fell outside ``V``) is a zero-conditionable-feature segment -- an error, not
    a silently-empty one -- and raises :class:`AxBenchBpbError`.  A segment that was
    ALREADY empty before filtering (no feature ids at all) is unaffected: that is
    pre-existing, unrelated behaviour, not something "emptied by the filter".

    ``allowed_ids=None`` (the default; an uncompacted checkpoint) is a STRICT
    no-op: returns *segment_conditions* UNCHANGED (the identical object, not a
    copy) and ``total_dropped=0`` -- callers can thread this through
    unconditionally with no ``if compacted`` branch of their own, and the
    uncompacted path stays byte-identical to code that never called this.

    Unlike ``_build_encdec_seg_map_and_encoder``, this does NOT itself emit a
    ``logger.warning`` -- it only returns the aggregated dropped count.  Callers
    own emitting exactly ONE warning naming that count (in their own wording),
    which keeps this helper reusable across call sites without hard-coding a
    message.

    Args:
        segment_conditions: One ``{feature_id: bucket}`` dict per context
            segment, in order.
        allowed_ids: The checkpoint's compacted vocabulary ``V`` (see
            :func:`feature_vocab_allowed_ids`), or ``None`` for an uncompacted
            checkpoint.

    Returns:
        A ``(restricted_segment_conditions, total_dropped)`` pair: the restricted
        per-segment dicts (same length/order as the input; a dict only ever loses
        keys, never gains or reorders them) and the total number of ids dropped
        across every segment.

    Raises:
        AxBenchBpbError: a segment had >= 1 feature id and every one of them fell
            outside ``allowed_ids``.
    """
    if allowed_ids is None:
        return segment_conditions, 0

    restricted: list[dict[int, int]] = []
    total_dropped = 0
    for seg_idx, segment in enumerate(segment_conditions):
        ids_tensor = torch.tensor(list(segment.keys()), dtype=torch.int64)
        kept, dropped = restrict_to_vocab(ids_tensor, allowed_ids)
        if dropped.numel() > 0:
            total_dropped += int(dropped.numel())
            if kept.numel() == 0:
                raise AxBenchBpbError(
                    "restrict_segment_conditions_to_vocab: segment "
                    f"{seg_idx} has {len(segment)} feature id(s), ALL outside "
                    "the checkpoint's compacted vocabulary V (dropped ids: "
                    f"{sorted(int(x) for x in dropped.tolist())}). A context "
                    "segment with zero conditionable features is an error; "
                    "rebuild/validate the context cache against this "
                    "checkpoint's feature_vocab."
                )
        kept_ids = set(kept.tolist())
        restricted.append(
            {fid: bucket for fid, bucket in segment.items() if fid in kept_ids}
        )
    return restricted, total_dropped


__all__ = [
    "UNCOMPACTED_VOCAB_FINGERPRINT",
    "checkpoint_vocab_fingerprint",
    "feature_vocab_allowed_ids",
    "restrict_segment_conditions_to_vocab",
    "restrict_to_vocab",
]
