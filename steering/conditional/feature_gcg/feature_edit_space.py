"""``FeatureConditioningEditSpace`` — the feature-space GCG edit space.

This is the central integration of the feature_gcg Phase 1 optimizer: it
implements the merged 12-method :class:`~steering.conditional.feature_gcg.edit_space.EditSpace`
protocol over :class:`~steering.conditional.feature_gcg.feature_conditioning.FeatureConditioning`
bags (``E = FeatureConditioning``), wiring together the aggregated gradient
(:class:`~steering.conditional.feature_gcg.gradient.FeatureGrad`), the candidate
primitives (:func:`~steering.conditional.feature_gcg.candidate_search.rank_replacements_from_gradients`),
the config (:class:`~steering.conditional.feature_gcg.config.FeatureGcgConfig`), and
the partition bounds (:class:`~steering.conditional.feature_gcg.partition.PartitionBounds`).

Scoping
-------
The edit space **enforces partition caps** (via ``bounds``) during candidate
generation; it does NOT apply ``count_penalty`` — that is added to the *objective*
passed to ``run_gcg`` (Task 7), so ``choose_best`` + the strict-``<`` accept gate
see it but the gradient proposal does not (the penalty is count-independent).

The four ``FeatureGrad`` tensors (``id_grad``, ``bucket_grad``, ``id_weight``,
``bucket_weight``) are treated as the only source of the SAE vocabulary size
(``id_weight.shape[0]``) and the bucket count is read from ``bounds.num_buckets``,
so the EditSpace needs no deep ``model`` introspection — the constructor's
``model`` handle is held only for the caller's wiring (Task 7). ``id_weight`` and
``bucket_weight`` are SHARED read-only tensors and are never mutated.

Candidate generation (§6.3)
---------------------------
ADD, REMOVE, and REPLACE-id are always generated; the magnitude (bucket) axis is
generated iff there is more than one bucket. Per editable bag slot with current
``(fid, bucket)``:

* **REPLACE-id** — :func:`rank_replacements_from_gradients` over
  ``allowed_feature_ids`` EXCLUDING ids already in the bag (dup guard), scored by
  ``id_grad``/``id_weight``; keeps the slot's bucket.
* **REPLACE-bucket** (iff magnitude active) — top-k of ``bucket_grad·bucket_weight``
  delta vs the current bucket; keeps the slot's fid.
* **REPLACE-joint** (iff magnitude active) — per-slot Cartesian of the id and
  bucket top-k (separable sum), globally truncated to ``top_k``.
* **ADD** — segment-pooled gradient proposes ``(fid, bucket)``; rejected at the
  target partition's cap.
* **REMOVE** — one per slot, first-order removal estimate; rejected at the
  partition's ``min_k`` floor.

"Magnitude active" = ``bounds.num_buckets > 1`` (``num_buckets == 1`` has no
second bucket to move to, so the magnitude axis is OFF).

ADD/REMOVE proposal *signs* are best-effort (a flipped sign only slows
convergence — the exact objective always selects the accepted candidate), but a
universal ``_within_bounds`` gate in :meth:`sample_candidates` is authoritative:
every produced candidate satisfies both partitions' ``[min_k, max_k - 1]`` hard
bounds (also catching REPLACE-bucket partition crossings).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype

from steering.conditional.feature_gcg.candidate_search import (
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.edit_primitives import TokenReplacement
from steering.conditional.feature_gcg.edit_space import GroupResult, ValidatedGroup
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.partition import PartitionBounds, is_active
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import (
    NonEmptyNonNegativeIntTuple,
    PositiveFiniteNumberLike,
    PositiveIntLike,
)

if TYPE_CHECKING:
    _FeatureConditioningList = list[FeatureConditioning]
    _StringSet = set[str]
else:
    _FeatureConditioningList = list
    _StringSet = set


@dataclass(frozen=True)
class FeatureRemoval:
    """Remove the entry at ``slot``. Applied as ``bag.with_removed(slot)``."""

    slot: int
    score: float


@dataclass(frozen=True)
class FeatureReplacement:
    """Replace ``slot`` with ``(fid, bucket)``.

    Applied as ``bag.with_replaced(slot, fid, bucket)``.
    """

    slot: int
    fid: int
    bucket: int
    score: float


@dataclass(frozen=True)
class FeatureAddition:
    """Add ``(fid, bucket)``. Applied as ``bag.with_added(fid, bucket)``."""

    fid: int
    bucket: int
    score: float


# One proposed single-edit. Lower ``score`` is better (a larger predicted
# objective decrease), matching the soft-greedy weighting in
# :meth:`FeatureConditioningEditSpace.sample_candidates`. The three variants make
# illegal states unrepresentable — no sentinel ``slot`` / ``fid`` / ``bucket``.
FeatureEdit = FeatureRemoval | FeatureReplacement | FeatureAddition


@dataclass(frozen=True)
class FeatureProposals:
    """Opaque proposal bundle returned by :meth:`rank_candidates`."""

    edits: tuple[FeatureEdit, ...]


class FeatureConditioningEditSpace:
    """``EditSpace[FeatureConditioning]`` for the feature-space GCG optimizer."""

    @beartype
    def __init__(
        self,
        *,
        bounds: PartitionBounds,
        allowed_feature_ids: NonEmptyNonNegativeIntTuple,
        warm_start: FeatureConditioning,
        feature_vocab: FeatureVocab | None = None,
    ) -> None:
        self.bounds = bounds
        self.allowed_feature_ids = allowed_feature_ids
        self.warm_start = warm_start
        # #720: feature ids stay ORIGINAL everywhere in this optimizer; ``feature_vocab``
        # is the SOLE compacted-only input. On a COMPACTED encoder the ``id_weight`` table
        # has ``V`` rows indexed by COMPACT id, so scoring an ORIGINAL candidate id needs
        # its compact row — that translation is ``_id_weight_rows``. ``None`` ⇒ uncompacted
        # ⇒ the translation is a literal identity (byte-identical to the pre-#720 path).
        self.feature_vocab = feature_vocab
        self._id_weight_row_map: t.Tensor | None = (
            None if feature_vocab is None else feature_vocab.old_to_new_tensor()
        )
        # #720 (perf): the ``int64[n_sae]`` row map is CPU-resident after construction, but
        # ``_id_weight_rows`` runs on EVERY ranking step. Moving the whole map to the gather
        # device on each call is a per-step host→device transfer; cache one resident copy per
        # device (moved once, reused). Stays empty on the uncompacted path — which returns
        # before ``_row_map_on`` — so it is a no-op there.
        self._id_weight_row_map_by_device: dict[t.device, t.Tensor] = {}

    # ──────────────────────────────────────────────────────────────────────
    # id_weight row translation (#720 — the one compacted-only branch)
    # ──────────────────────────────────────────────────────────────────────

    def _row_map_on(self, device: t.device) -> t.Tensor:
        """The ``old_to_new`` row map resident on ``device``, moved ONCE per device and reused.

        Only reached on the COMPACTED path (``_id_weight_row_map is not None``). The map is
        constant for the optimizer's lifetime, so caching the moved copy turns the per-call
        transfer in :meth:`_id_weight_rows` into a one-time move per device (#720 perf).
        """
        assert self._id_weight_row_map is not None
        cached = self._id_weight_row_map_by_device.get(device)
        if cached is None:
            cached = self._id_weight_row_map.to(device)
            self._id_weight_row_map_by_device[device] = cached
        return cached

    def _id_weight_rows(self, allowed_t: t.Tensor) -> t.Tensor:
        """Translate ORIGINAL feature ids to the rows that index the encoder's ``id_weight``.

        The SOLE compacted-only branch in the optimizer (#720). On an UNCOMPACTED model
        (``feature_vocab is None``) this is a literal early-return — ``allowed_t`` already
        IS the ``id_weight`` row — so every gather routed through it stays byte-identical to
        the pre-#720 optimizer. On a COMPACTED model the ``id_weight`` table has ``V`` rows
        indexed by COMPACT id, so an original id ``o`` is mapped to ``old_to_new[o]`` before
        it can gather the correct row (via the device-cached map). Returns a ``LongTensor`` on
        ``allowed_t``'s device.

        ``allowed_t`` is range-checked against ``[0, n_sae)`` BEFORE the gather (#720 T5).
        ``self._id_weight_row_map`` is ``old_to_new_tensor()``, shaped ``(n_sae,)``: fancy-
        indexing it with an id ``>= n_sae`` raises a raw, uninformative ``IndexError`` (on
        CUDA, plausibly a context-poisoning device assert); a NEGATIVE id does not raise at
        all — Python/torch fancy indexing WRAPS (``-1`` -> the last row), silently returning
        a PLAUSIBLE in-range translated row and dodging :meth:`_require_rows_in_vocab`'s
        ``-1`` check, which runs AFTER this gather and inspects only the *translated row*,
        never the input id. This guard is independent of, and a superset of, that one: it
        catches an id that is not a valid original SAE id AT ALL, which is a different
        failure from being a valid SAE id that is merely OOV of the compacted vocabulary V.
        """
        if self._id_weight_row_map is None:
            return allowed_t
        n_sae = self._id_weight_row_map.shape[0]
        out_of_range = (allowed_t < 0) | (allowed_t >= n_sae)
        bad_idx = t.nonzero(out_of_range, as_tuple=False).flatten()
        if bad_idx.numel() > 0:
            offending_id = int(allowed_t[int(bad_idx[0].item())].item())
            raise FeatureGcgError(
                f"feature id {offending_id} is outside the valid original SAE id range "
                f"[0, {n_sae}), so it cannot be translated to a compacted id_weight row. "
                "It is not a valid SAE feature id at all (a different failure from being "
                "a valid SAE id that is merely outside the compacted vocabulary V)."
            )
        return self._row_map_on(allowed_t.device)[allowed_t]

    def _require_rows_in_vocab(
        self,
        rows: t.Tensor,
        original_ids: t.Tensor,
        vocab_size: int,
        *,
        uncompacted_message: str,
    ) -> None:
        """Fail loud if any translated ``id_weight`` row is outside ``[0, vocab_size)``.

        Guards every ORIGINAL-id → ``id_weight``-row gather routed through
        :meth:`_id_weight_rows`, so a ``-1`` sentinel can never negative-index the last row.
        On an UNCOMPACTED table (``feature_vocab is None``) ``rows`` IS the original id, so an
        out-of-range row means the id is genuinely outside the SAE vocab — ``uncompacted_message``
        is raised verbatim, preserving the pre-#720 behavior. On a COMPACTED table an
        out-of-range row is the ``-1`` that :meth:`FeatureVocab.old_to_new_tensor` emits for an
        original id NOT in the compacted vocabulary ``V``; the message names the first offending
        ORIGINAL id and reports it as OOV of ``V`` (an in-range SAE id with no compact embedding
        row), NOT as an out-of-range SAE id.
        """
        bad = (rows < 0) | (rows >= vocab_size)
        bad_idx = t.nonzero(bad, as_tuple=False).flatten()
        if bad_idx.numel() == 0:
            return
        if self._id_weight_row_map is None:
            raise FeatureGcgError(uncompacted_message)
        first = int(bad_idx[0].item())
        offending_id = int(original_ids[first].item())
        offending_row = int(rows[first].item())
        raise FeatureGcgError(
            f"feature id {offending_id} is a valid SAE id but is NOT in this compacted "
            f"model's vocabulary V: it translates to id_weight row {offending_row}, outside "
            f"[0, {vocab_size}). It has no compact embedding row."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Partition helpers
    # ──────────────────────────────────────────────────────────────────────

    def _is_pos(self, bucket: int) -> bool:
        """True iff *bucket* counts toward the positive (active) partition."""
        if not self.bounds.has_negative:
            return True
        return is_active(bucket, self.bounds.num_buckets)

    def _counts(self, bag: FeatureConditioning) -> tuple[int, int]:
        """``(n_pos, n_neg)`` for *bag* under the partition convention."""
        if not self.bounds.has_negative:
            return len(bag), 0
        n_pos = sum(1 for _fid, bucket in bag.entries if is_active(bucket, self.bounds.num_buckets))
        return n_pos, len(bag) - n_pos

    def _within_bounds(self, bag: FeatureConditioning) -> bool:
        """True iff every partition count is within its ``[min_k, max_k - 1]``."""
        n_pos, n_neg = self._counts(bag)
        if not (self.bounds.min_k_pos <= n_pos <= self.bounds.max_k_pos - 1):
            return False
        return not self.bounds.has_negative or (
            self.bounds.min_k_neg <= n_neg <= self.bounds.max_k_neg - 1
        )

    def _add_allowed(self, bucket: int, n_pos: int, n_neg: int) -> bool:
        """True iff adding an entry in *bucket*'s partition keeps it under the cap."""
        if self._is_pos(bucket):
            return n_pos + 1 <= self.bounds.max_k_pos - 1
        return n_neg + 1 <= self.bounds.max_k_neg - 1

    def _remove_allowed(self, bucket: int, n_pos: int, n_neg: int) -> bool:
        """True iff removing an entry in *bucket*'s partition stays at/above ``min_k``."""
        if self._is_pos(bucket):
            return n_pos - 1 >= self.bounds.min_k_pos
        return n_neg - 1 >= self.bounds.min_k_neg

    def magnitude_active(self) -> bool:
        """The magnitude axis is searched iff there is more than one bucket.

        ``num_buckets == 1`` forces the magnitude search OFF (there is no second
        bucket to move to).
        """
        return self.bounds.num_buckets > 1

    # ──────────────────────────────────────────────────────────────────────
    # Seam 1: validate
    # ──────────────────────────────────────────────────────────────────────

    def validate(self, group: Any) -> ValidatedGroup:
        key_str = getattr(group, "key_str", None)
        if not isinstance(key_str, str) or not key_str:
            raise FeatureGcgError(
                f"group.key_str must be a non-empty str, got {key_str!r}"
            )
        members = getattr(group, "members", None)
        if members is None or len(members) < 1:
            raise FeatureGcgError("group.members must be a non-empty sequence")
        for i, member in enumerate(members):
            if member is None:
                raise FeatureGcgError(f"group member {i} is None")

        # #720 T5 (FIX 1 front door): on a COMPACTED model, ``self._id_weight_row_map``
        # (shape ``(n_sae,)``) is the valid input range for every ``_id_weight_rows`` gather
        # this edit space will ever perform. Catching an out-of-range id HERE — before the
        # search even starts — is strictly EARLIER than the same guard inside
        # ``_id_weight_rows``, which still applies on its own to ids that never pass through
        # ``allowed_feature_ids`` at all (e.g. a REMOVE candidate's current bag id). There is
        # no equivalent bound on an UNCOMPACTED model: the valid id range there is
        # ``id_weight.shape[0]``, known only once a real gradient arrives.
        n_sae = None if self._id_weight_row_map is None else self._id_weight_row_map.shape[0]
        for fid in self.allowed_feature_ids:
            if n_sae is not None and fid >= n_sae:
                raise FeatureGcgError(
                    f"allowed feature id {fid} is outside the valid original SAE id range "
                    f"[0, {n_sae}) — this model's feature_vocab is compacted, so allowed "
                    "ids must be original SAE ids, not compact ids or arbitrary integers."
                )

        allowed_set = set(self.allowed_feature_ids)
        for fid, bucket in self.warm_start.entries:
            if fid not in allowed_set:
                raise FeatureGcgError(
                    f"warm_start feature id {fid} is not in allowed_feature_ids "
                    f"({len(allowed_set)} allowed id(s), a sparse set — not necessarily "
                    "a contiguous range)"
                )
            if not 0 <= bucket < self.bounds.num_buckets:
                raise FeatureGcgError(
                    f"warm_start bucket {bucket} outside [0, {self.bounds.num_buckets})"
                )
        if not self._within_bounds(self.warm_start):
            n_pos, n_neg = self._counts(self.warm_start)
            raise FeatureGcgError(
                f"warm_start violates partition bounds: n_pos={n_pos} "
                f"(allowed [{self.bounds.min_k_pos}, {self.bounds.max_k_pos - 1}]), "
                f"n_neg={n_neg} "
                f"(allowed [{self.bounds.min_k_neg}, {self.bounds.max_k_neg - 1}], "
                f"has_negative={self.bounds.has_negative})"
            )

        return ValidatedGroup(key_str=key_str, members=list(members), payload=group)

    # ──────────────────────────────────────────────────────────────────────
    # Seams 2-4: triviality / init
    # ──────────────────────────────────────────────────────────────────────

    def is_trivial(self, validated: ValidatedGroup) -> bool:
        return False

    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult:
        raise FeatureGcgError(
            "FeatureConditioningEditSpace has no trivial/control path; "
            "is_trivial is always False, so finalize_trivial must never be called."
        )

    def init_target(self, validated: ValidatedGroup) -> FeatureConditioning:
        return self.warm_start

    # ──────────────────────────────────────────────────────────────────────
    # Seam 5: digest (pure)
    # ──────────────────────────────────────────────────────────────────────

    def digest(self, target: FeatureConditioning) -> str:
        return target.digest()

    # ──────────────────────────────────────────────────────────────────────
    # Seam 6: aggregate_gradient (slot-aligned mean, consumes the iterable)
    # ──────────────────────────────────────────────────────────────────────

    def aggregate_gradient(self, per_member: Iterable[tuple[Any, Any]]) -> FeatureGrad:
        # Materialise (and thereby fully consume the strict zip from run_gcg).
        grads: list[FeatureGrad] = [grad for grad, _member in per_member]
        if not grads:
            raise FeatureGcgError("aggregate_gradient requires at least one member gradient")
        id_grad = t.stack([g.id_grad for g in grads], dim=0).mean(dim=0)
        bucket_grad = t.stack([g.bucket_grad for g in grads], dim=0).mean(dim=0)
        # id_weight / bucket_weight are shared read-only tensors (identical across
        # members) — reuse the first member's, never clone or mutate.
        return FeatureGrad(
            id_grad=id_grad,
            bucket_grad=bucket_grad,
            id_weight=grads[0].id_weight,
            bucket_weight=grads[0].bucket_weight,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Seam 7: rank_candidates (§6.3)
    # ──────────────────────────────────────────────────────────────────────

    def rank_candidates(
        self, agg_grad: Any, current: FeatureConditioning, cfg: FeatureGcgConfig
    ) -> FeatureProposals:
        grad: FeatureGrad = agg_grad
        n_real = len(current)
        n_pos, n_neg = self._counts(current)
        top_k = int(cfg.top_k)
        magnitude = self.magnitude_active()
        current_fids = set(current.feature_ids())
        allowed_excluded = [fid for fid in self.allowed_feature_ids if fid not in current_fids]

        edits: list[FeatureEdit] = []

        # REPLACE-id (and capture per-slot id replacements for the joint product).
        id_repls: dict[int, tuple[TokenReplacement, ...]] = {}
        if n_real > 0 and allowed_excluded:
            id_repls = self._rank_replace_id(grad, current, top_k, allowed_excluded)
            for slot, repls in id_repls.items():
                cur_bucket = current.entries[slot][1]
                for repl in repls:
                    edits.append(
                        FeatureReplacement(slot=slot, fid=repl.token_id, bucket=cur_bucket, score=repl.score)
                    )

        # REPLACE-bucket (and capture per-slot bucket rankings for the joint product).
        bucket_ranks: dict[int, list[tuple[int, float]]] = {}
        if magnitude and n_real > 0:
            bucket_ranks = self._rank_buckets(grad, current, top_k)
            for slot, ranked in bucket_ranks.items():
                cur_fid = current.entries[slot][0]
                for bucket, delta in ranked:
                    edits.append(FeatureReplacement(slot=slot, fid=cur_fid, bucket=bucket, score=delta))

        # REPLACE-joint: per-slot Cartesian of the two axes, globally top-k.
        if magnitude and id_repls and bucket_ranks:
            joint: list[FeatureReplacement] = []
            for slot, repls in id_repls.items():
                ranked = bucket_ranks.get(slot, [])
                for repl in repls:
                    for bucket, delta in ranked:
                        joint.append(
                            FeatureReplacement(slot=slot, fid=repl.token_id, bucket=bucket, score=repl.score + delta)
                        )
            joint.sort(key=lambda e: e.score)
            edits.extend(joint[:top_k])

        # ADD (always generated).
        edits.extend(
            self._add_edits(grad, current, top_k, magnitude, allowed_excluded, n_pos, n_neg)
        )

        # REMOVE (always generated).
        edits.extend(self._remove_edits(grad, current, n_pos, n_neg))

        return FeatureProposals(edits=tuple(edits))

    def _rank_replace_id(
        self,
        grad: FeatureGrad,
        current: FeatureConditioning,
        top_k: int,
        allowed_excluded: Sequence[int],
    ) -> dict[int, tuple[TokenReplacement, ...]]:
        """REPLACE-id via the shared primitive, re-keyed slot → replacements.

        The primitive reserves position 0 for an (un-edited) BOS token, so bag
        slot ``s`` is mapped to position ``s + 1`` with a dummy zero row/token at
        position 0; the returned positions are mapped back to slots.

        ``distance_weight=0.0``: the primitive's no-op-replacement penalty is inert
        here — ``allowed_excluded`` already drops every current feature-id, so the
        replace-with-self candidate it targets never exists.

        ``id_to_row`` (#720 T5 FIX 2) is passed ONLY on a COMPACTED model. The shared
        primitive's OOV error message is picked by ``id_to_row is None`` — legacy
        "outside embedding vocab" wording when ``None`` (uncompacted), the translated
        "...vocabulary V..." wording otherwise — mirroring how :meth:`_require_rows_in_vocab`
        already discriminates on ``self._id_weight_row_map is None``. Passing the
        (identity-valued but non-``None``) ``self._id_weight_rows`` unconditionally would
        make an UNCOMPACTED model's OOV always read as "compacted", which is false; the
        VALUES are unaffected either way, since ``_id_weight_rows`` is a literal identity
        when uncompacted (same tensor, same numbers) — only the failure MESSAGE changes.
        """
        n_real = len(current)
        id_dim = grad.id_grad.shape[1]
        emb_grad = t.zeros((n_real + 1, id_dim), dtype=grad.id_grad.dtype, device=grad.id_grad.device)
        emb_grad[1:] = grad.id_grad
        current_tokens = [0, *current.feature_ids()]
        by_pos = rank_replacements_from_gradients(
            embedding_gradient=emb_grad,
            embedding_weight=grad.id_weight,
            editable_positions=range(1, n_real + 1),
            allowed_token_ids=allowed_excluded,
            current_tokens=current_tokens,
            top_k=top_k,
            distance_weight=0.0,
            id_to_row=self._id_weight_rows if self._id_weight_row_map is not None else None,
        )
        return {pos - 1: repls for pos, repls in by_pos.items()}

    def _rank_buckets(
        self, grad: FeatureGrad, current: FeatureConditioning, top_k: int
    ) -> dict[int, list[tuple[int, float]]]:
        """Per-slot bucket replacement ranking (delta vs the slot's current bucket).

        Scores are first-order deltas ``(bucket_weight[b] - bucket_weight[b_cur]) ·
        bucket_grad[slot]`` so they share scale with the REPLACE-id deltas; the
        current bucket is excluded (it is a no-op).
        """
        out: dict[int, list[tuple[int, float]]] = {}
        for slot, (_fid, cur_bucket) in enumerate(current.entries):
            scores = grad.bucket_grad[slot] @ grad.bucket_weight.t()  # (num_buckets,)
            deltas = scores - scores[cur_bucket]
            ranked: list[tuple[int, float]] = []
            for bucket in t.argsort(deltas).tolist():
                if bucket == cur_bucket:
                    continue
                ranked.append((int(bucket), float(deltas[bucket])))
                if len(ranked) >= top_k:
                    break
            out[slot] = ranked
        return out

    def _add_edits(
        self,
        grad: FeatureGrad,
        current: FeatureConditioning,
        top_k: int,
        magnitude: bool,
        allowed_excluded: Sequence[int],
        n_pos: int,
        n_neg: int,
    ) -> list[FeatureEdit]:
        """ADD proposals from the segment-pooled gradient (proposal-only).

        Masked (would-be-new) slots have zero gradient, so the pooled gradient
        over the existing real slots is the only first-order signal. Each
        ``(fid, bucket)`` is rejected when its partition is at the cap.
        """
        if not allowed_excluded:
            return []
        id_dim = grad.id_grad.shape[1]
        d_bucket = grad.bucket_weight.shape[1]
        if len(current) > 0:
            pooled_id = grad.id_grad.mean(dim=0)
            pooled_bucket = grad.bucket_grad.mean(dim=0)
        else:
            pooled_id = t.zeros(id_dim, dtype=grad.id_grad.dtype, device=grad.id_grad.device)
            pooled_bucket = t.zeros(d_bucket, dtype=grad.bucket_grad.dtype, device=grad.bucket_grad.device)

        allowed_t = t.tensor(list(allowed_excluded), dtype=t.long, device=grad.id_weight.device)
        # #720: translate ORIGINAL ids to their id_weight ROWS before gathering/range-checking
        # (identity on an uncompacted table). The returned candidate fid stays ORIGINAL.
        rows = self._id_weight_rows(allowed_t)
        self._require_rows_in_vocab(
            rows,
            allowed_t,
            grad.id_weight.shape[0],
            uncompacted_message="allowed_feature_ids contain ids outside the SAE vocab",
        )
        f_scores = grad.id_weight.index_select(0, rows) @ pooled_id  # (n_allowed,)
        kf = min(top_k, len(allowed_excluded))
        fvals, fidx = t.topk(f_scores, kf, largest=False)
        fid_candidates = [(int(allowed_t[int(i)].item()), float(v.item())) for v, i in zip(fvals, fidx, strict=True)]

        if magnitude:
            b_scores = grad.bucket_weight @ pooled_bucket  # (num_buckets,)
            kb = min(top_k, self.bounds.num_buckets)
            bvals, bidx = t.topk(b_scores, kb, largest=False)
            bucket_candidates = [(int(i.item()), float(v.item())) for v, i in zip(bvals, bidx, strict=True)]
        else:
            bucket_candidates = [(0, 0.0)]

        edits: list[FeatureEdit] = []
        for fid, fscore in fid_candidates:
            for bucket, bscore in bucket_candidates:
                if not self._add_allowed(bucket, n_pos, n_neg):
                    continue
                edits.append(FeatureAddition(fid=fid, bucket=bucket, score=fscore + bscore))
        edits.sort(key=lambda e: e.score)
        return edits[:top_k]

    def _remove_edits(
        self, grad: FeatureGrad, current: FeatureConditioning, n_pos: int, n_neg: int
    ) -> list[FeatureEdit]:
        """REMOVE proposals: one per slot, scored by the first-order removal estimate.

        Removing slot ``s`` zeroes its contribution, so the predicted objective
        change is ``-grad·embedding`` (lower ⇒ better). Rejected when removing the
        entry would drop its partition below ``min_k``.
        """
        edits: list[FeatureEdit] = []
        if len(current) == 0:
            return edits
        # #720: the current bag's fids are ORIGINAL ids; translate them to their id_weight
        # ROWS before gathering (identity on an uncompacted table). Sorted-canonical
        # ``feature_ids()`` is slot-aligned with ``entries``, so row[slot] holds fid[slot].
        feature_ids_t = t.tensor(
            list(current.feature_ids()), dtype=t.long, device=grad.id_weight.device
        )
        cur_rows = self._id_weight_rows(feature_ids_t)
        # #720 (fail-loud): mirror the ADD guard. A COMPACTED bag id ∉ V translates to a ``-1``
        # row that would silently negative-index ``id_weight[-1]`` (the LAST compact row) — the
        # cross-space mis-index this optimizer exists to eliminate. RAISE, naming the id.
        self._require_rows_in_vocab(
            cur_rows,
            feature_ids_t,
            grad.id_weight.shape[0],
            uncompacted_message="current bag feature ids contain ids outside the SAE vocab",
        )
        for slot, (_fid, bucket) in enumerate(current.entries):
            if not self._remove_allowed(bucket, n_pos, n_neg):
                continue
            row = int(cur_rows[slot].item())
            contrib = float((grad.id_grad[slot] @ grad.id_weight[row]).item()) + float(
                (grad.bucket_grad[slot] @ grad.bucket_weight[bucket]).item()
            )
            edits.append(FeatureRemoval(slot=slot, score=-contrib))
        return edits

    # ──────────────────────────────────────────────────────────────────────
    # Seam 8: sample_candidates
    # ──────────────────────────────────────────────────────────────────────

    @beartype
    def sample_candidates(
        self,
        proposals: Any,
        current: FeatureConditioning,
        *,
        batch_size: PositiveIntLike,
        temperature: PositiveFiniteNumberLike,
        rng: random.Random,
        visited: _StringSet,
    ) -> _FeatureConditioningList:
        pool: tuple[FeatureEdit, ...] = proposals.edits
        if not pool:
            return []

        # Soft-greedy weighting (Faster-GCG): weight ∝ exp(-(score - min_score)/T).
        # Subtracting min_score keeps the best edit at weight 1.0 (overflow-safe);
        # the score deltas are <= 0 after subtraction so exp never overflows.
        min_score = min(edit.score for edit in pool)
        weights = [math.exp(-(edit.score - min_score) / temperature) for edit in pool]

        out: list[FeatureConditioning] = []
        max_attempts = max(batch_size * 20, 100)
        attempts = 0
        while len(out) < batch_size and attempts < max_attempts:
            attempts += 1
            edit = rng.choices(pool, weights=weights, k=1)[0]
            candidate = self._apply_edit(edit, current)
            if candidate is None:
                continue
            digest = candidate.digest()
            if digest in visited:
                continue
            visited.add(digest)
            out.append(candidate)
        return out

    def _apply_edit(
        self, edit: FeatureEdit, bag: FeatureConditioning
    ) -> FeatureConditioning | None:
        """Apply one edit; return the new bag, or ``None`` if it violates a cap.

        The proposals are structurally valid by construction (in-range slots,
        non-duplicate fids), so the bag mutation never raises; the only soft
        rejection is the authoritative ``_within_bounds`` partition-cap gate
        (which also catches REPLACE-bucket partition crossings).
        """
        if isinstance(edit, FeatureAddition):
            candidate = bag.with_added(edit.fid, edit.bucket)
        elif isinstance(edit, FeatureRemoval):
            candidate = bag.with_removed(edit.slot)
        else:
            candidate = bag.with_replaced(edit.slot, edit.fid, edit.bucket)
        if not self._within_bounds(candidate):
            return None
        return candidate

    # ──────────────────────────────────────────────────────────────────────
    # Seam 9: model_item (unified)
    # ──────────────────────────────────────────────────────────────────────

    def model_item(self, member: Any, target: FeatureConditioning) -> Any:
        return (member, target)

    # ──────────────────────────────────────────────────────────────────────
    # Seams 10-11: event fields (concept identity only)
    # ──────────────────────────────────────────────────────────────────────

    def _identity_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        """Concept-identity fields; keys are disjoint from ``_RESERVED_EVENT_KEYS``."""
        fields: dict[str, Any] = {
            "group_key": validated.key_str,
            "n_members": len(validated.members),
        }
        group = validated.payload
        concept_id = getattr(group, "concept_id", None)
        if concept_id is not None:
            fields["concept_id"] = concept_id
        mode = getattr(group, "mode", None)
        if mode is not None:
            fields["mode"] = mode
        return fields

    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return self._identity_fields(validated)

    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return self._identity_fields(validated)

    # ──────────────────────────────────────────────────────────────────────
    # Seam 12: finalize
    # ──────────────────────────────────────────────────────────────────────

    def finalize(
        self,
        validated: ValidatedGroup,
        target: FeatureConditioning,
        *,
        objective: float,
        accepted_steps: int,
        visited_count: int,
    ) -> GroupResult:
        n_pos, n_neg = self._counts(target)
        return GroupResult(
            payload={
                "bag": target,
                "objective": objective,
                "accepted_steps": accepted_steps,
                "visited": visited_count,
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        )
