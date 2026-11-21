"""Model-free bookkeeping core for the UNIVERSAL per-concept GCG trigger.

The legacy Faster-GCG eval optimised a separate adversarial suffix for every
(prefix, rich, avoid) attack row. This module replaces that with a UNIVERSAL,
per-concept trigger: all attack rows that share the same steering+adversarial
configuration (the 7-tuple ``key`` below) are optimised jointly against one
shared editable-token trigger.

Everything here is pure and deterministic over attack-row dicts -- no model, no
torch forward. Group membership, the per-step minibatch deal, and the trigger
seed are all functions of the row data plus seeded ``random.Random`` / hashing,
so a run is byte-reproducible. Failures are loud via ``AxBenchBpbAdvError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import torch as t
from beartype import beartype
from beartype.vale import Is

from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    TokenReplacement,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_bpb_adv.bos_utils import bos_padded_tokens
from steering.eval.axbench_bpb_adv.editable_tokens import (
    select_final_k_editable_span,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.refined_types import (
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
)

# Attack-row columns that define a concept group. Every row carrying the same
# 7-tuple is optimised against ONE shared universal trigger. The order here is
# the canonical ``key`` order and drives both ``key_str`` (the seed string) and
# the deterministic group sort.
# The canonical concept-group identity: the column names AND their strict runtime
# domains, defined ONCE here. Every consumer -- the grouping below, the
# universal-trigger artifact (optimize._trigger_row), and general_val's trigger
# lookup -- derives its key via group_key(), so a new identity field is added in
# exactly one place. (The aggregation join key is a DIFFERENT, coarser key and is
# intentionally not derived from this.)
GROUP_KEY_COLUMNS: tuple[str, ...] = (
    "concept_id",
    "mode",
    "method_name",
    "target_model",
    "coefficient_str",
    "adversarial_method",
    "strength_index",
)
# Only Faster-GCG rows carry an editable trigger; GEPA / additive rows are not
# universal-GCG members.
_FASTER_GCG = "faster_gcg"

# Canonical within-group row order for determinism.
_ROW_SORT_COLUMNS: tuple[str, ...] = (
    "core_prefix_id",
    "concept_rich_sample_id",
    "concept_avoiding_sample_id",
)


def _is_concept_group_key(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == len(GROUP_KEY_COLUMNS)
        and isinstance(value[0], NonNegativeInt)
        and all(isinstance(part, NonEmptyStr) for part in value[1:6])
        and isinstance(value[6], NonNegativeInt)
    )


def _is_non_empty_attack_row_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, dict) and all(isinstance(key, str) for key in row) for row in value)
    )


if TYPE_CHECKING:
    ConceptGroupKey = tuple[
        NonNegativeInt,
        NonEmptyStr,
        NonEmptyStr,
        NonEmptyStr,
        NonEmptyStr,
        NonEmptyStr,
        NonNegativeInt,
    ]
    _NonEmptyAttackRowListLike = list[dict[str, Any]]
else:
    ConceptGroupKey = Annotated[
        tuple,
        Is[_is_concept_group_key],
    ]
    _NonEmptyAttackRowListLike = Annotated[
        list,
        Is[_is_non_empty_attack_row_list],
    ]


@dataclass(frozen=True, init=False)
class ConceptGroup:
    """All Faster-GCG attack rows sharing one steering+adversarial config.

    A concept group is the unit a single universal trigger is optimised over.
    ``editable_token_count`` (``k``) is the constant editable-span width shared
    by every member row; ``k == 0`` marks a no-adversary control group (which
    has no trigger). ``key_str`` is a deterministic seed string for the group's
    sampler RNG.
    """

    key: ConceptGroupKey
    key_str: NonEmptyStr
    editable_token_count: NonNegativeInt
    rows: list[dict[str, Any]]

    @beartype
    def __init__(
        self,
        key: ConceptGroupKey,
        key_str: NonEmptyStrLike,
        editable_token_count: NonNegativeIntLike,
        rows: _NonEmptyAttackRowListLike,
    ) -> None:
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "key_str", NonEmptyStr.parse(key_str))
        object.__setattr__(
            self,
            "editable_token_count",
            NonNegativeInt.parse(editable_token_count),
        )
        object.__setattr__(self, "rows", rows)
        if self.key_str != _key_str(self.key):
            raise AxBenchBpbAdvError("concept group key_str must be the canonical pipe-joined key")

    @property
    def is_control(self) -> bool:
        return self.editable_token_count == 0


def _group_non_negative_int(values: Mapping[str, Any], column: str) -> NonNegativeInt:
    try:
        return NonNegativeInt.parse(values[column])
    except (KeyError, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"concept group field {column!r} must be a non-negative integer"
        ) from exc


def _group_non_empty_str(values: Mapping[str, Any], column: str) -> NonEmptyStr:
    try:
        return NonEmptyStr.parse(values[column])
    except (KeyError, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"concept group field {column!r} must be a non-empty string"
        ) from exc


def group_key(values: Mapping[str, Any]) -> ConceptGroupKey:
    """The canonical group-identity tuple from a mapping of GROUP_KEY_COLUMNS.

    The SINGLE source of the concept-group key, shared by grouping, the trigger
    artifact, and general_val's trigger lookup. Field types must already match
    the schema: concept/strength indices are strict non-negative integers and
    identity labels are non-empty strings. ``values`` may carry extra keys (e.g.
    a row dict / an ``asdict`` combo); only GROUP_KEY_COLUMNS are read.
    """

    return (
        _group_non_negative_int(values, "concept_id"),
        _group_non_empty_str(values, "mode"),
        _group_non_empty_str(values, "method_name"),
        _group_non_empty_str(values, "target_model"),
        _group_non_empty_str(values, "coefficient_str"),
        _group_non_empty_str(values, "adversarial_method"),
        _group_non_negative_int(values, "strength_index"),
    )


@beartype
def _key_str(key: ConceptGroupKey) -> NonEmptyStr:
    return NonEmptyStr.parse("|".join(str(part) for part in key))


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _group_non_empty_str(row, "core_prefix_id"),
        _group_non_empty_str(row, "concept_rich_sample_id"),
        _group_non_empty_str(row, "concept_avoiding_sample_id"),
    )


def concept_groups(attack_rows: list[dict[str, Any]]) -> list[ConceptGroup]:
    """Partition Faster-GCG attack rows into universal-trigger concept groups.

    Non-``faster_gcg`` rows are dropped (GEPA / additive carry no editable
    trigger). Rows are grouped by the 7-tuple ``key``; within each group the
    editable span width MUST be constant -- the group ``k`` is anchored on the
    first member row and every member row is cross-checked to equal it (the
    invariant the design review flagged as load-bearing for the shared trigger).
    A violation raises ``AxBenchBpbAdvError`` naming the offending group and row.
    Groups are returned sorted by ``key`` and rows within a group sorted by
    ``(core_prefix_id, concept_rich_sample_id, concept_avoiding_sample_id)`` so
    the output is byte-deterministic.
    """

    buckets: dict[ConceptGroupKey, list[dict[str, Any]]] = {}
    for row in attack_rows:
        if _group_non_empty_str(row, "adversarial_method") != _FASTER_GCG:
            continue
        buckets.setdefault(group_key(row), []).append(row)

    groups: list[ConceptGroup] = []
    for key in sorted(buckets):
        member_rows = buckets[key]
        # The group k is the constant editable-span width. Anchor it on the first
        # member row, then VALIDATE (the cross-check the design review flagged as
        # the load-bearing invariant for the shared trigger) that every member
        # row equals it -- a non-constant set fails here, naming the offender.
        k = _group_non_negative_int(member_rows[0], "editable_token_count")
        for row in member_rows:
            row_k = _group_non_negative_int(row, "editable_token_count")
            if row_k != k:
                raise AxBenchBpbAdvError(
                    "concept group "
                    f"{_key_str(key)} has non-constant editable_token_count: row "
                    f"(core_prefix_id={row['core_prefix_id']!r}, "
                    f"concept_rich_sample_id={row['concept_rich_sample_id']!r}, "
                    f"concept_avoiding_sample_id="
                    f"{row['concept_avoiding_sample_id']!r}) has "
                    f"editable_token_count {row_k} != group k={k}; a universal "
                    "trigger requires one fixed span width across all member rows"
                )
        groups.append(
            ConceptGroup(
                key=key,
                key_str=_key_str(key),
                editable_token_count=k,
                rows=sorted(member_rows, key=_row_sort_key),
            )
        )
    return groups


def init_trigger(group: ConceptGroup) -> list[int]:
    """Seed the universal trigger from the smallest-prefix row's last-k span.

    The seed is the final ``group.editable_token_count`` source tokens of the
    row with the lexicographically smallest ``core_prefix_id``. With a single
    member row this is exactly that row's last-k editable span, which anchors
    the future 1-pair equivalence with per-prefix Faster-GCG. Control groups
    (``k == 0``) have no trigger and raise.
    """

    if group.is_control:
        raise AxBenchBpbAdvError(
            f"concept group {group.key_str} is a control group (k=0) and has no "
            "trigger to initialise"
        )

    seed_row = min(group.rows, key=lambda row: str(row["core_prefix_id"]))
    source_tokens: Sequence[int] = seed_row["source_prompt_tokens"]
    span = select_final_k_editable_span(source_tokens, k=group.editable_token_count)
    return [int(token_id) for token_id in source_tokens[span.start : span.stop]]


def aggregate_trigger_gradient(
    per_pair: list[tuple[t.Tensor, EditableSpan]],
) -> t.Tensor:
    """Mean the per-pair last-k gradient slices into one shared ``(k, D)`` block.

    Each pair contributes a full ``(prompt_len, D)`` objective gradient plus its
    OWN last-k editable span (prefixes of different ``prompt_len`` align by their
    own ``[span.start, span.stop)``, never an absolute index). We slice each pair
    to its ``(k, D)`` trigger block and mean over the pairs, giving the shared
    trigger gradient used for trigger-local ranking.

    With a single pair this is exactly that pair's last-k slice -- the identity
    that anchors the 1-pair equivalence with per-prefix Faster-GCG. A SUM would
    scale the gradient (hence the proposal-score softmax) by the minibatch size;
    MEAN keeps the score scale comparable across minibatch sizes for a fixed pair
    set, so the sampler ``temperature`` stays meaningful. Empty input, a non-2-D
    gradient, or slices of differing width all raise ``AxBenchBpbAdvError``.
    """

    if not per_pair:
        raise AxBenchBpbAdvError(
            "aggregate_trigger_gradient requires at least one (gradient, span) pair"
        )

    slices: list[t.Tensor] = []
    expected_k: int | None = None
    for index, (gradient, span) in enumerate(per_pair):
        if gradient.ndim != 2:
            raise AxBenchBpbAdvError(
                "per-pair gradient must be 2-D (prompt_len, D), got "
                f"{tuple(gradient.shape)} for pair index {index}"
            )
        if span.count < 0:
            raise AxBenchBpbAdvError(
                f"editable span count must be >= 0, got {span.count} for pair "
                f"index {index}"
            )
        if span.stop > gradient.shape[0]:
            raise AxBenchBpbAdvError(
                f"editable span [{span.start}, {span.stop}) exceeds gradient "
                f"prompt length {gradient.shape[0]} for pair index {index}"
            )
        block = gradient[span.start : span.stop]
        if expected_k is None:
            expected_k = block.shape[0]
        elif block.shape[0] != expected_k:
            raise AxBenchBpbAdvError(
                "per-pair last-k slices disagree in width: "
                f"pair index {index} has k={block.shape[0]} != {expected_k}; "
                "every member prefix must share one trigger width"
            )
        slices.append(block)

    stacked = t.stack(slices, dim=0)  # (n_pairs, k, D)
    return stacked.mean(dim=0)  # (k, D)


def rank_trigger_replacements(
    *,
    trigger_gradient: t.Tensor,
    embedding_weight: t.Tensor,
    current_trigger: Sequence[int],
    allowed_token_ids: Sequence[int],
    top_k: int,
    distance_weight: float,
    bos_token_id: int,
    allowed_embedding_cache: AllowedEmbeddingCache | None = None,
) -> dict[int, tuple[TokenReplacement, ...]]:
    """Rank trigger-token replacements in trigger-local coordinates.

    The shared primitive ``rank_replacements_from_gradients`` guards
    ``position <= 0`` as a BOS slot (never editable). The trigger has no BOS, so
    we rank it through a BOS-PADDED view: prepend a zero gradient row and the
    real ``bos_token_id`` token, rank the padded positions ``1..k`` (the original
    trigger positions), then RE-INDEX the returned dict back to trigger-local
    ``0..k-1``. The BOS pad row carries NO signal -- it is zero and its position 0
    is never ranked -- so the result is identical to ranking the same trigger as
    the last-k of a full BOS-prefixed prompt (the math behind the 1-pair
    equivalence). ``bos_token_id`` MUST be a real in-vocab id because the
    primitive validates the current token at each ranked position is in
    ``[0, vocab)``; the BOS row itself is never ranked.
    """

    if trigger_gradient.ndim != 2:
        raise AxBenchBpbAdvError(
            "trigger_gradient must be 2-D (k, D), got "
            f"{tuple(trigger_gradient.shape)}"
        )
    k = trigger_gradient.shape[0]
    if len(current_trigger) != k:
        raise AxBenchBpbAdvError(
            f"current_trigger length {len(current_trigger)} must equal trigger "
            f"gradient k={k}"
        )
    if bos_token_id < 0 or bos_token_id >= embedding_weight.shape[0]:
        raise AxBenchBpbAdvError(
            f"bos_token_id {bos_token_id} is outside embedding vocab "
            f"[0, {embedding_weight.shape[0]})"
        )

    d = trigger_gradient.shape[1]
    padded_grad = t.cat(
        [
            t.zeros(1, d, dtype=trigger_gradient.dtype, device=trigger_gradient.device),
            trigger_gradient,
        ],
        dim=0,
    )
    padded_tokens = bos_padded_tokens(
        current_trigger,
        bos_id=int(bos_token_id),
        context="rank_trigger_replacements BOS-padded trigger view",
    )

    try:
        padded_ranked = rank_replacements_from_gradients(
            embedding_gradient=padded_grad,
            embedding_weight=embedding_weight,
            editable_positions=range(1, k + 1),
            allowed_token_ids=allowed_token_ids,
            current_tokens=padded_tokens,
            top_k=top_k,
            distance_weight=distance_weight,
            allowed_embedding_cache=allowed_embedding_cache,
        )
    except FeatureGcgError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc

    # Re-index padded positions 1..k back to trigger-local 0..k-1 (and re-stamp
    # each TokenReplacement.position to its trigger-local coordinate). The BOS
    # pad slot (padded position 0) is never present in `padded_ranked`.
    local: dict[int, tuple[TokenReplacement, ...]] = {}
    for padded_position, replacements in padded_ranked.items():
        local_position = padded_position - 1
        local[local_position] = tuple(
            TokenReplacement(
                position=local_position,
                token_id=replacement.token_id,
                score=replacement.score,
            )
            for replacement in replacements
        )
    return local
