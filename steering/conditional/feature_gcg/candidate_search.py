"""Model-agnostic Faster-GCG candidate bookkeeping."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch as t
from beartype import beartype

from steering.conditional.feature_gcg.edit_primitives import (
    TokenReplacement,
    replace_single_token,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.refined_types import (
    NonNegativeFiniteNumberLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
)

_TokenIdLike = int | np.integer[Any]

if TYPE_CHECKING:
    _IntSequence = Sequence[int]
    _TokenIdSequence = Sequence[_TokenIdLike]
    _ReplacementMapping = Mapping[int, Sequence[TokenReplacement]]
    _ReplacementRanking = dict[int, tuple[TokenReplacement, ...]]
    _StringAbstractSet = AbstractSet[str]
    _StringSet = set[str]
else:
    # Avoid beartype's global-RNG-backed sampling of parameterized containers.
    # The algorithm retains its precise static shapes and explicit structural
    # checks; beartype owns the refined scalar parameters introduced here.
    _IntSequence = Sequence
    _TokenIdSequence = Sequence
    _ReplacementMapping = Mapping
    _ReplacementRanking = dict
    _StringAbstractSet = AbstractSet
    _StringSet = set


@dataclass(frozen=True)
class CandidateSequence:
    prompt_tokens: tuple[int, ...]
    position: int
    token_id: int
    proposal_score: float


if TYPE_CHECKING:
    _CandidateSequenceList = list[CandidateSequence]
else:
    _CandidateSequenceList = list


def _canonical_device(device: t.device | str) -> t.device:
    """Resolve a device spec to the form a materialized tensor will report.

    ``t.device("cuda")`` and ``t.device("cuda:0")`` are NOT equal, but a tensor
    moved with ``.to("cuda")`` reports the indexed form. The cache re-validates
    its pool's device on every hit, so an unindexed declared device would fail
    that check against its own correct output on the very first cache hit.
    """

    resolved = t.device(device)
    if resolved.type == "cuda" and resolved.index is None and t.cuda.is_available():
        return t.device("cuda", t.cuda.current_device())
    return resolved


def _embedding_weight_signature(weight: t.Tensor) -> tuple[object, ...]:
    """Return a constant-time structural identity for a cached source tensor.

    This deliberately does not read tensor contents: a full content checksum on
    every ranking step would traverse the same ``(V, D)`` matrix that the cache
    exists to avoid gathering. The universal token-trigger owner supplies one
    detached, immutable surrogate tensor for the lifetime of a search.
    """

    return (
        id(weight),
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.dtype,
        weight.device,
        bool(weight.requires_grad),
    )


def _materialize_allowed_embedding_pool(
    *,
    embedding_weight: t.Tensor,
    allowed_token_ids: _TokenIdSequence,
    id_to_row: Callable[[t.Tensor], t.Tensor] | None = None,
    device: t.device | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Build the allowed-id tensor and its gathered embedding rows.

    The operation order is the pre-#712 ranking implementation verbatim. Keeping
    one helper for both the raw reference path and the cache miss prevents the
    two materialization semantics from drifting.

    ``device`` relocates the finished pool. ``None`` keeps it on
    ``embedding_weight.device``, which is the pre-existing behavior verbatim.
    The BPB-Adv owner supplies a CPU surrogate snapshot (the gradient capture
    path forces CPU), but the ranking matmul it feeds is
    ``(P, D) @ (D, V)`` over a 0.66 GiB pool -- 55ms on CPU, 0.9ms on GPU at
    the production shapes -- so the pool is placed on the model's device while
    the source snapshot stays where the gradient path needs it.

    ``id_to_row`` (#720) translates an ORIGINAL feature id to the ``embedding_weight``
    ROW that holds its embedding: the IDENTITY on an uncompacted table (it returns its
    input unchanged, so ``rows is allowed`` and this path is byte-identical), and
    ``old_to_new[allowed]`` on a COMPACTED table (``V`` rows indexed by compact id). The
    gather AND the range check both operate on the ROW; the RETURNED ``allowed`` stays the
    ORIGINAL id tensor, so callers keep naming candidates in original-id space. ``None`` is
    the pre-#720 behavior verbatim.
    """

    with profile_region(
        "feature_gcg.rank.materialize_allowed_embedding_pool",
        allowed_count=len(allowed_token_ids),
        embedding_dim=int(embedding_weight.shape[1]),
        dtype=str(embedding_weight.dtype),
        device=str(embedding_weight.device),
    ):
        allowed = t.tensor(
            list(allowed_token_ids),
            dtype=t.long,
            device=embedding_weight.device,
        )
        rows = id_to_row(allowed) if id_to_row is not None else allowed
        if (
            int(rows.min().item()) < 0
            or int(rows.max().item()) >= embedding_weight.shape[0]
        ):
            if id_to_row is None:
                # Uncompacted (row == id): the id is genuinely out of range. Message preserved
                # VERBATIM (pinned by test_cache_preserves_legacy_error_text_and_precedence).
                raise FeatureGcgError(
                    "allowed_token_ids contain ids outside embedding vocab"
                )
            # Compacted (#720): a ``-1`` translated row is an in-range ORIGINAL id that is OOV
            # of the compacted vocabulary V (no compact embedding row), NOT an out-of-range id.
            bad = (rows < 0) | (rows >= embedding_weight.shape[0])
            first = int(t.nonzero(bad, as_tuple=False).flatten()[0].item())
            offending_id = int(allowed[first].item())
            offending_row = int(rows[first].item())
            raise FeatureGcgError(
                f"allowed_token_ids contain original id {offending_id} that is NOT in this "
                f"compacted model's vocabulary V: it translates to embedding row "
                f"{offending_row}, outside [0, {embedding_weight.shape[0]}). It has no compact "
                "embedding row."
            )
        allowed_embeddings = embedding_weight.index_select(0, rows)
        if device is not None and device != embedding_weight.device:
            # Relocate the pool AFTER the gather and AFTER the range checks, so
            # the validation semantics and error text above are untouched -- the
            # move is the last thing that happens and cannot change which ids
            # are rejected. Both tensors move together; the ranking derives its
            # working device from ``allowed_embeddings`` and indexes ``allowed``
            # with topk output, so a split placement would be a device mismatch.
            allowed = allowed.to(device)
            allowed_embeddings = allowed_embeddings.to(device)

    materialized_bytes = (
        allowed.numel() * allowed.element_size()
        + allowed_embeddings.numel() * allowed_embeddings.element_size()
    )
    record_counter(
        "feature_gcg.rank.allowed_embedding_pool_materializations",
        1,
        allowed_count=len(allowed_token_ids),
        embedding_dim=int(embedding_weight.shape[1]),
        dtype=str(embedding_weight.dtype),
        device=str(embedding_weight.device),
    )
    record_counter(
        "feature_gcg.rank.allowed_embedding_pool_bytes",
        materialized_bytes,
        allowed_count=len(allowed_token_ids),
        embedding_dim=int(embedding_weight.shape[1]),
        dtype=str(embedding_weight.dtype),
        device=str(embedding_weight.device),
    )
    return allowed, allowed_embeddings


class AllowedEmbeddingCache:
    """Lazy, search-scoped cache of allowed embedding rows.

    The cache is bound to one exact embedding tensor and one immutable token-id
    tuple. It is intentionally not global: the owner controls the source tensor's
    lifetime and releases the roughly vocab-sized pool when that search returns.

    Same-storage value mutation after materialization is unsupported for both
    the source and the returned cached tensors. Detecting it would require
    reading the whole pool on every cache hit, defeating the optimization. The
    production owner passes a detached surrogate snapshot, and the ranking path
    treats every resolved tensor as immutable for the search lifetime.
    """

    def __init__(
        self,
        *,
        embedding_weight: t.Tensor,
        allowed_token_ids: _TokenIdSequence,
        device: t.device | str | None = None,
    ) -> None:
        self._embedding_weight = embedding_weight
        self.allowed_token_ids = cast(
            tuple[int, ...],
            tuple(allowed_token_ids),
        )
        # Where the POOL lives, which is not necessarily where the SOURCE lives.
        # ``None`` means "same as the source", i.e. the pre-existing behavior.
        # The pool is materialized once per search and reused across every group
        # (the owner builds one edit space outside the group loop), so a
        # cross-device placement costs a single transfer for the whole run.
        self.device = (
            embedding_weight.device
            if device is None
            else _canonical_device(device)
        )
        self._allowed: t.Tensor | None = None
        self._allowed_embeddings: t.Tensor | None = None
        self._source_signature: tuple[object, ...] | None = None

    @property
    def is_materialized(self) -> bool:
        """Whether both cached tensors have been committed successfully."""

        states = (
            self._allowed is not None,
            self._allowed_embeddings is not None,
            self._source_signature is not None,
        )
        if any(states) and not all(states):
            raise FeatureGcgError(
                "allowed embedding cache is partially initialized"
            )
        return all(states)

    def resolve(
        self,
        *,
        embedding_weight: t.Tensor,
        allowed_token_ids: _TokenIdSequence,
    ) -> tuple[t.Tensor, t.Tensor]:
        """Return the cached pool, materializing it on the first valid use."""

        if embedding_weight is not self._embedding_weight:
            raise FeatureGcgError(
                "allowed embedding cache was created for a different "
                "embedding_weight tensor"
            )
        if allowed_token_ids is not self.allowed_token_ids:
            raise FeatureGcgError(
                "allowed embedding cache was created for a different "
                "allowed_token_ids sequence"
            )
        if embedding_weight.requires_grad:
            raise FeatureGcgError(
                "allowed embedding cache requires embedding_weight.requires_grad=False"
            )

        if not self.is_materialized:
            signature_before = _embedding_weight_signature(embedding_weight)
            allowed, allowed_embeddings = _materialize_allowed_embedding_pool(
                embedding_weight=embedding_weight,
                allowed_token_ids=allowed_token_ids,
                device=self.device,
            )
            signature_after = _embedding_weight_signature(embedding_weight)
            if signature_after != signature_before:
                raise FeatureGcgError(
                    "embedding_weight changed while materializing allowed embedding cache"
                )
            # Publish cache references only after every validation and the gather
            # succeed. Normal execution cannot expose partially validated tensors;
            # is_materialized still fails loudly on externally corrupted state.
            self._allowed = allowed
            self._allowed_embeddings = allowed_embeddings
            self._source_signature = signature_after
            return allowed, allowed_embeddings

        if _embedding_weight_signature(embedding_weight) != self._source_signature:
            raise FeatureGcgError(
                "embedding_weight changed after allowed embedding cache materialization"
            )
        allowed = self._allowed
        allowed_embeddings = self._allowed_embeddings
        if allowed is None or allowed_embeddings is None:
            raise FeatureGcgError(
                "allowed embedding cache is partially initialized"
            )
        expected_count = len(self.allowed_token_ids)
        if (
            allowed.ndim != 1
            or allowed.shape != (expected_count,)
            or allowed.dtype != t.long
            or allowed.device != self.device
        ):
            raise FeatureGcgError(
                "allowed embedding cache id tensor has inconsistent structure"
            )
        if (
            allowed_embeddings.shape
            != (expected_count, embedding_weight.shape[1])
            or allowed_embeddings.dtype != embedding_weight.dtype
            or allowed_embeddings.device != self.device
        ):
            raise FeatureGcgError(
                "allowed embedding cache embedding tensor has inconsistent structure"
            )
        return allowed, allowed_embeddings


@beartype
def rank_replacements_from_gradients(
    *,
    embedding_gradient: t.Tensor,
    embedding_weight: t.Tensor,
    editable_positions: _IntSequence,
    allowed_token_ids: _TokenIdSequence,
    current_tokens: _IntSequence,
    top_k: PositiveIntLike,
    distance_weight: NonNegativeFiniteNumberLike,
    allowed_embedding_cache: AllowedEmbeddingCache | None = None,
    id_to_row: Callable[[t.Tensor], t.Tensor] | None = None,
) -> _ReplacementRanking:
    """Rank token replacements using the exact objective gradient.

    Lower proposal scores are better. The score is the first-order GCG
    replacement delta dot product plus Faster-GCG's regularizer that
    discourages no-op replacements.

    ``id_to_row`` (#720) maps an ORIGINAL feature id to the ``embedding_weight`` ROW that
    holds its embedding — the identity on an uncompacted table (byte-identical to the
    pre-#720 path), ``old_to_new`` on a COMPACTED one whose ``V`` rows are indexed by
    compact id. It is applied to BOTH the ``allowed`` candidate ids (via
    :func:`_materialize_allowed_embedding_pool`) and the ``current_tokens`` embedded per
    position, so a compacted ``embedding_weight`` is gathered by the right rows while the
    RETURNED ``token_id`` stays the ORIGINAL id. ``None`` is the pre-#720 behavior
    verbatim (row == id); it is incompatible with ``allowed_embedding_cache`` (the cache
    path is uncompacted-only).
    """

    if embedding_gradient.ndim != 2:
        raise FeatureGcgError(
            f"embedding_gradient must be 2-D (T, D), got {embedding_gradient.shape}"
        )
    if embedding_weight.ndim != 2:
        raise FeatureGcgError(
            f"embedding_weight must be 2-D (V, D), got {embedding_weight.shape}"
        )
    if embedding_gradient.shape[1] != embedding_weight.shape[1]:
        raise FeatureGcgError(
            "gradient/embedding dim mismatch: "
            f"{embedding_gradient.shape[1]} vs {embedding_weight.shape[1]}"
        )
    if not allowed_token_ids:
        raise FeatureGcgError("allowed_token_ids must be non-empty")
    if id_to_row is not None and allowed_embedding_cache is not None:
        raise FeatureGcgError(
            "id_to_row (compacted id→row translation) and allowed_embedding_cache cannot "
            "be combined; the cache path is uncompacted-only"
        )

    if allowed_embedding_cache is None:
        allowed, allowed_embeddings = _materialize_allowed_embedding_pool(
            embedding_weight=embedding_weight,
            allowed_token_ids=allowed_token_ids,
            id_to_row=id_to_row,
        )
    else:
        allowed, allowed_embeddings = allowed_embedding_cache.resolve(
            embedding_weight=embedding_weight,
            allowed_token_ids=allowed_token_ids,
        )
    # Translate the CURRENT token ids to their embedding ROWS too (identity on an
    # uncompacted table). ``current_tokens[0]`` is the reserved BOS/dummy slot — never an
    # editable position — so a possible ``-1`` translation there is never gathered.
    current_rows = (
        id_to_row(t.tensor([int(c) for c in current_tokens], dtype=t.long, device=embedding_weight.device))
        if id_to_row is not None
        else None
    )
    k_eff = min(int(top_k), len(allowed_token_ids))

    # Validate every position FIRST, in iteration order, so the raised error is
    # the same one (and for the same position) as the pre-batching loop.
    positions: list[int] = []
    rows: list[int] = []
    for position in editable_positions:
        if position <= 0 or position >= len(current_tokens):
            raise FeatureGcgError(
                f"editable position {position} outside non-BOS prompt range"
            )
        current_token = int(current_tokens[int(position)])
        if current_rows is None:
            current_row = current_token
            if current_row < 0 or current_row >= embedding_weight.shape[0]:
                raise FeatureGcgError(
                    f"current token id {current_token} at position {position} "
                    "is outside embedding vocab"
                )
        else:
            current_row = int(current_rows[int(position)].item())
            if current_row < 0 or current_row >= embedding_weight.shape[0]:
                raise FeatureGcgError(
                    f"current token id {current_token} at position {position} maps to "
                    f"embedding row {current_row}, outside the embedding table "
                    f"[0, {embedding_weight.shape[0]})"
                )
        positions.append(int(position))
        rows.append(int(current_row))

    if not positions:
        return {}

    # Score every editable position in ONE matmul.
    #
    # The per-position form was ``(allowed_embeddings - current_embedding) @ grad``.
    # That materializes a full (V, D) copy of the allowed-embedding pool for EVERY
    # position of EVERY step -- 100 MB per position at V=32768, D=768 -- and then
    # reads it again for the matvec. A py-spy profile of a live BPB-Adv
    # build_candidates run measured that single line at **60.3% of in-loop
    # main-thread time**, against 0.7% for the actual model backward: the search
    # was bound on allocator traffic, not on the model.
    #
    # Distributing the dot product removes the temporary entirely:
    #
    #     (A - c) . g  ==  A . g - (c . g)        with (c . g) a SCALAR
    #
    # so the pool is read once for all positions instead of once per position.
    # Measured at the production shapes: 9.08 ms -> 0.14 ms per step (~65x).
    device = allowed_embeddings.device
    # Phase-level instrumentation (no-op unless an AxBenchProfiler is bound).
    # The phases are split so a profile distinguishes DEVICE-SIDE cost (gather,
    # scores, topk) from the HOST transfer and the Python object construction --
    # a flat timing for this function cannot tell those apart, and py-spy
    # attributes GPU time to whichever line happens to synchronize.
    common_attrs = {
        "positions": len(positions),
        "allowed_count": int(allowed.shape[0]),
        "embedding_dim": int(allowed_embeddings.shape[1]),
        "top_k": int(k_eff),
        "dtype": str(allowed_embeddings.dtype),
        "device": str(device),
    }

    with profile_region("feature_gcg.rank.gather_operands", **common_attrs):
        grads = embedding_gradient.index_select(
            0, t.tensor(positions, dtype=t.long, device=embedding_gradient.device)
        ).to(device)  # (P, D)
        current_embeddings = embedding_weight.index_select(
            0, t.tensor(rows, dtype=t.long, device=embedding_weight.device)
        ).to(device)  # (P, D)

    with profile_region("feature_gcg.rank.scores", **common_attrs):
        # (P, V) -- row p is the score vector the old loop computed for position p.
        scores = grads @ allowed_embeddings.T
        scores = scores - (current_embeddings * grads).sum(dim=-1, keepdim=True)

    if distance_weight:
        with profile_region("feature_gcg.rank.distance_weight", **common_attrs):
            current_token_ids = t.tensor(
                [int(current_tokens[p]) for p in positions],
                dtype=allowed.dtype,
                device=allowed.device,
            ).unsqueeze(1)
            same = allowed.unsqueeze(0) == current_token_ids
            scores = scores + same.to(scores.dtype) * float(distance_weight)

    with profile_region("feature_gcg.rank.topk", **common_attrs):
        values, indices = t.topk(scores, k=k_eff, dim=-1, largest=False)
        gathered_ids = allowed[indices]

    with profile_region("feature_gcg.rank.host_transfer", **common_attrs):
        # One host transfer for the whole ranking. The old code called .item()
        # twice per returned replacement, i.e. 2 * k_eff * P device syncs per
        # step. This is also the first sync after the device work above, so it
        # is where any queued GPU time will be billed.
        token_ids = gathered_ids.tolist()
        scores_out = values.tolist()

    with profile_region("feature_gcg.rank.build_replacements", **common_attrs):
        ranking = {
            position: tuple(
                TokenReplacement(
                    position=position,
                    token_id=int(token_id),
                    score=float(score),
                )
                for score, token_id in zip(
                    scores_out[i], token_ids[i], strict=True
                )
            )
            for i, position in enumerate(positions)
        }

    record_counter(
        "feature_gcg.rank.replacements_returned",
        len(positions) * int(k_eff),
        **common_attrs,
    )
    return ranking


@beartype
def sample_candidate_sequences(
    *,
    current_tokens: _IntSequence,
    replacements_by_position: _ReplacementMapping,
    batch_size: PositiveIntLike,
    temperature: PositiveFiniteNumberLike,
    rng: random.Random,
    visited_digests: _StringSet | None = None,
) -> _CandidateSequenceList:
    pool = [
        replacement
        for replacements in replacements_by_position.values()
        for replacement in replacements
    ]
    if not pool:
        raise FeatureGcgError("replacement pool is empty")

    # Soft-greedy candidate sampling (Faster-GCG): weight each replacement by
    # exp(-(score - min_score) / temperature). temperature interpolates between
    # uniform sampling (temp -> inf, vanilla GCG) and deterministic argmin
    # (temp -> 0); the default 0.1 sits near the greedy end BY DESIGN, so a
    # peaked softmax here is intended, not a bug. NOTE: `score` is the raw
    # first-order estimate (e_cand - e_cur)·grad in UNNORMALISED units, so this
    # temperature is not scale-portable across models with different gradient
    # magnitudes. Subtracting min_score keeps the best replacement at weight 1.0
    # (overflow-safe normalisation).
    min_score = min(replacement.score for replacement in pool)
    weights = [
        math.exp(-(replacement.score - min_score) / temperature)
        for replacement in pool
    ]
    visited = visited_digests if visited_digests is not None else set()
    out: list[CandidateSequence] = []
    max_attempts = max(batch_size * 20, 100)
    attempts = 0
    while len(out) < batch_size and attempts < max_attempts:
        attempts += 1
        replacement = rng.choices(pool, weights=weights, k=1)[0]
        tokens = replace_single_token(
            current_tokens,
            position=replacement.position,
            token_id=replacement.token_id,
        )
        digest = token_sequence_digest(tokens)
        if digest in visited:
            continue
        visited.add(digest)
        out.append(
            CandidateSequence(
                prompt_tokens=tuple(tokens),
                position=replacement.position,
                token_id=replacement.token_id,
                proposal_score=replacement.score,
            )
        )
    return out


@beartype
def replacement_pool_exhausted(
    *,
    current_tokens: _IntSequence,
    replacements_by_position: _ReplacementMapping,
    visited_digests: _StringAbstractSet,
) -> bool:
    """Return whether every sequence derivable from ``replacements`` was visited.

    Exhaustion is defined in the sampler's complete-sequence digest space, not
    by proposal entries. This makes duplicate and no-op replacements collapse
    exactly as they do in :func:`sample_candidate_sequences`. The caller owns
    ``visited_digests``; this scan is strictly read-only.
    """

    for replacements in replacements_by_position.values():
        for replacement in replacements:
            tokens = replace_single_token(
                current_tokens,
                position=replacement.position,
                token_id=replacement.token_id,
            )
            if token_sequence_digest(tokens) not in visited_digests:
                return False
    return True


def deterministic_step_rng(*, candidate_seed: str, step: int, master_seed: int) -> random.Random:
    payload = f"{candidate_seed}|{int(step)}|{int(master_seed)}"
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)
