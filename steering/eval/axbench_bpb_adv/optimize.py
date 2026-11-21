"""Candidate construction for AxBench-BPB-Adv.

Sign convention (pinned). The worst-case adversary DEGRADES steering, i.e. it
MINIMIZES the signed class margin ``primary_margin_y``. The optimizer retains a
candidate only when ``best_value < current_value``, so the search objective is
defined to be MINIMIZED:

    g      = loss_contrast - loss_desired
    grad g = grad_contrast - grad_desired

where ``desired`` is the class steering favors (positive mode -> rich,
negative mode -> avoid) and ``contrast`` is the other side. ``g`` is exactly the
signed class margin ``primary_margin_y`` (positive mode: ``BPB(avoid) -
BPB(rich)``), so the adversary minimizes the margin directly. ``objective_value``
returns ``g``; ``objective_gradient`` returns ``grad g``. The PLOTTED y-axis is
always ``primary_margin_y`` -- which is exactly ``g``.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import pyarrow as pa
import torch as t
from pyarrow import ArrowInvalid, ArrowTypeError

from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    replacement_pool_exhausted,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.core import _emit, run_gcg
from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    TokenReplacement,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.edit_space import GroupResult, ValidatedGroup
from steering.eval.axbench_bpb_adv.bos_utils import (
    bos_padded_tokens,
    strip_bos_padded_tokens,
)
from steering.eval.axbench_bpb_adv.candidate_metadata import (
    CandidateMetadata,
    FasterGcgFinalMetadata,
    GepaFinalMetadata,
    NoAdversaryControlMetadata,
    candidate_metadata_to_json,
)
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.editable_tokens import (
    replace_editable_tokens,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.universal_gcg import (
    GROUP_KEY_COLUMNS,
    ConceptGroup,
    aggregate_trigger_gradient,
    concept_groups,
    init_trigger,
    rank_trigger_replacements,
)
from steering.refined_types import NonNegativeInt
from steering.segmentation import resolve_bos_token_id

# Batched objective: many (row, tokens) -> one objective each, evaluated in a
# single padded forward (the universal candidate-ranking speedup). Each value
# matches a separate single-prompt evaluation up to bf16 batch-shape rounding.
ObjectiveBatchFn = Callable[
    [Sequence[tuple[dict[str, Any], Sequence[int]]]], list[float]
]
# Batched gradient: the universal step's whole minibatch -> one per-pair objective
# gradient each, computed in a single padded fwd+bwd (the per-step speedup). Each
# tensor matches a separate single-prompt gradient up to bf16 batch-shape
# rounding; aggregate_trigger_gradient consumes them exactly as it did the
# per-pair loop.
GradientBatchProvider = Callable[
    [Sequence[tuple[dict[str, Any], Sequence[int]]], int], list[t.Tensor]
]
GepaOptimizer = Callable[[dict[str, Any], Sequence[int], int], Sequence[int]]
ProgressCallback = Callable[[dict[str, Any]], None]


def objective_value(
    *,
    loss_desired: float,
    loss_contrast: float,
) -> float:
    """Search objective to MINIMIZE: ``g = loss_contrast - loss_desired``.

    This equals ``primary_margin_y`` (see module docstring).
    """

    return float(loss_contrast) - float(loss_desired)


def objective_gradient(
    *,
    grad_loss_desired: t.Tensor,
    grad_loss_contrast: t.Tensor,
) -> t.Tensor:
    """Gradient of ``g``: ``grad g = grad_contrast - grad_desired``."""

    return grad_loss_contrast - grad_loss_desired


def _decode_tokens(tokenizer: Any, tokens: Sequence[int]) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if decoder is None:
        return ""
    return str(decoder([int(token_id) for token_id in tokens]))


def _strip_single_bos(tokenizer: Any, tokens: Sequence[int]) -> list[int]:
    bos_id = resolve_bos_token_id(tokenizer)
    out = [int(token_id) for token_id in tokens]
    if bos_id is not None and out and out[0] == bos_id:
        return out[1:]
    return out


def _decode_prompt_tokens(tokenizer: Any, tokens: Sequence[int]) -> str:
    return _decode_tokens(tokenizer, _strip_single_bos(tokenizer, tokens))


def _candidate_id(row: dict[str, Any], attacked_digest: str) -> str:
    payload = {
        "core_prefix_id": row["core_prefix_id"],
        "concept_rich_output_digest": row["concept_rich_output_digest"],
        "concept_avoiding_output_digest": row["concept_avoiding_output_digest"],
        "method_name": row["method_name"],
        "target_model": row["target_model"],
        "coefficient_str": row["coefficient_str"],
        "adversarial_method": row["adversarial_method"],
        "adversarial_strength": row["adversarial_strength"],
        "strength_unit": row["strength_unit"],
        "strength_index": row["strength_index"],
        "context_signal_mode": row["context_signal_mode"],
        "source_prompt_token_digest": row["source_prompt_token_digest"],
        "attacked_prompt_token_digest": attacked_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_attack_rows(attack_rows: pa.Table) -> None:
    if not attack_rows.schema.equals(ATTACK_ROWS_SCHEMA):
        raise AxBenchBpbAdvError("expected attack rows with ATTACK_ROWS_SCHEMA")


def _span_from_row(row: dict[str, Any]) -> EditableSpan:
    return EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )


def _validate_attack_row(row: dict[str, Any]) -> tuple[list[int], EditableSpan]:
    source_tokens = [int(token_id) for token_id in row["source_prompt_tokens"]]
    expected_digest = str(row["source_prompt_token_digest"])
    actual_digest = token_sequence_digest(source_tokens)
    if actual_digest != expected_digest:
        raise AxBenchBpbAdvError(
            f"source prompt digest mismatch for attack_row_id={row['attack_row_id']!r}: "
            f"expected {expected_digest}, computed {actual_digest}"
        )
    span = _span_from_row(row)
    expected_editable = [int(token_id) for token_id in row["original_editable_tokens"]]
    if source_tokens[span.start:span.stop] != expected_editable:
        raise AxBenchBpbAdvError(
            f"original editable tokens do not match prompt span for "
            f"attack_row_id={row['attack_row_id']!r}"
        )
    return source_tokens, span


def _candidate_row(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    source_tokens: Sequence[int],
    attacked_tokens: Sequence[int],
    objective: float | None,
    metadata: CandidateMetadata,
) -> dict[str, Any]:
    span = _span_from_row(row)
    replacement_tokens = [int(token_id) for token_id in attacked_tokens[span.start:span.stop]]
    attacked_digest = token_sequence_digest(attacked_tokens)
    return {
        "candidate_id": _candidate_id(row, attacked_digest),
        "attack_row_id": str(row["attack_row_id"]),
        "core_prefix_id": str(row["core_prefix_id"]),
        "concept_rich_sample_id": str(row["concept_rich_sample_id"]),
        "concept_avoiding_sample_id": str(row["concept_avoiding_sample_id"]),
        "concept_id": int(row["concept_id"]),
        "mode": str(row["mode"]),
        "method_name": str(row["method_name"]),
        "target_model": str(row["target_model"]),
        "coefficient_str": str(row["coefficient_str"]),
        "adversarial_method": str(row["adversarial_method"]),
        "adversarial_strength": float(row["adversarial_strength"]),
        "strength_unit": str(row["strength_unit"]),
        "strength_index": int(row["strength_index"]),
        "context_signal_mode": str(row["context_signal_mode"]),
        "prefix_text": str(row["prefix_text"]),
        "prefix_hash": str(row["prefix_hash"]),
        "source_prompt_tokens": [int(token_id) for token_id in source_tokens],
        "source_prompt_token_digest": str(row["source_prompt_token_digest"]),
        "attacked_prompt_text": _decode_prompt_tokens(tokenizer, attacked_tokens),
        "attacked_prompt_tokens": [int(token_id) for token_id in attacked_tokens],
        "attacked_prompt_token_digest": attacked_digest,
        "concept_rich_output_tokens": [
            int(token_id) for token_id in row["concept_rich_output_tokens"]
        ],
        "concept_avoiding_output_tokens": [
            int(token_id) for token_id in row["concept_avoiding_output_tokens"]
        ],
        "concept_rich_output_digest": str(row["concept_rich_output_digest"]),
        "concept_avoiding_output_digest": str(row["concept_avoiding_output_digest"]),
        "editable_token_start": span.start,
        "editable_token_count": span.count,
        "replacement_tokens": replacement_tokens,
        "replacement_text": _decode_tokens(tokenizer, replacement_tokens),
        "objective_value": None if objective is None else float(objective),
        "candidate_metadata_json": candidate_metadata_to_json(metadata),
    }


def build_control_candidate(*, row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    source_tokens, _span = _validate_attack_row(row)
    return _candidate_row(
        row=row,
        tokenizer=tokenizer,
        source_tokens=source_tokens,
        attacked_tokens=source_tokens,
        objective=None,
        metadata=NoAdversaryControlMetadata(),
    )


def _bos_token_id(tokenizer: Any) -> int:
    """Resolve the tokenizer's BOS id (used only as the rank/sample pad slot).

    Delegates to the shared :func:`steering.segmentation.resolve_bos_token_id`
    (probes both the ``.bos_token_id`` attribute and ``get_bos_token_id()``), so
    every BOS-resolution site in the pipeline agrees on the same contract."""

    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise AxBenchBpbAdvError(
            "tokenizer exposes no BOS token id; the universal trigger ranking "
            "requires a BOS pad slot (see rank_trigger_replacements)"
        )
    return bos_id


def _padded_trigger_tokens(
    *,
    trigger: Sequence[int],
    bos_token_id: int,
    context: str,
) -> list[int]:
    """Return ``[BOS, *trigger]`` and reject accidental double-BOS views."""

    return bos_padded_tokens(
        trigger,
        bos_id=int(bos_token_id),
        context=context,
    )


def _strip_padded_trigger_tokens(
    padded: Sequence[int],
    *,
    bos_token_id: int,
    context: str,
) -> list[int]:
    """Validate and strip the BOS-padded trigger view used by candidate sampling."""

    return strip_bos_padded_tokens(
        padded,
        bos_id=int(bos_token_id),
        context=context,
    )


def _validate_group_rows(
    group: ConceptGroup,
) -> list[tuple[dict[str, Any], list[int], EditableSpan]]:
    """Validate every member row and re-assert its span width equals group k.

    v2 finding C1: ``concept_groups`` already pins a constant
    ``editable_token_count``, but the substitution path reads
    ``editable_token_start``/``count`` straight from the row, so we re-assert the
    cross-check here (the load-bearing invariant for a SHARED trigger) before any
    substitution, naming the offending row.
    """

    out: list[tuple[dict[str, Any], list[int], EditableSpan]] = []
    for row in group.rows:
        source_tokens, span = _validate_attack_row(row)
        if span.count != group.editable_token_count:
            raise AxBenchBpbAdvError(
                f"concept group {group.key_str} row "
                f"(core_prefix_id={row['core_prefix_id']!r}, "
                f"concept_rich_sample_id={row['concept_rich_sample_id']!r}, "
                f"concept_avoiding_sample_id="
                f"{row['concept_avoiding_sample_id']!r}) has editable span width "
                f"{span.count} != group k={group.editable_token_count}; a "
                "universal trigger requires one fixed span width across all rows"
            )
        out.append((row, source_tokens, span))
    return out


def _sub_into_row(
    source_tokens: Sequence[int],
    span: EditableSpan,
    trigger: Sequence[int],
) -> list[int]:
    """Substitute ``trigger`` into a row's last-k editable span (length-preserving).

    The named ``_sub(row, trigger)`` helper from the plan/spec: every prefix in a
    concept group shares ONE trigger, substituted at its OWN last-k span.
    """

    return replace_editable_tokens(source_tokens, span, trigger)


def _group_event_fields(group: ConceptGroup) -> dict[str, Any]:
    """Group-level identity for progress events (v2 M6).

    A concept group spans MANY prefixes, so it has no single ``attack_row_id``;
    the progress vocabulary is group-scoped (``group_key_str``, ``n_pairs``,
    ``editable_token_count``) plus the shared steering identity. ``method_name`` /
    ``target_model`` / ``coefficient_str`` / ``adversarial_strength`` are constant
    within a group, so they are read off the first row.
    """

    first = group.rows[0]
    return {
        "group_key_str": group.key_str,
        "n_pairs": len(group.rows),
        "method_name": str(first["method_name"]),
        "target_model": str(first["target_model"]),
        "coefficient_str": str(first["coefficient_str"]),
        "adversarial_strength": float(first["adversarial_strength"]),
        "context_signal_mode": str(first["context_signal_mode"]),
        "editable_token_count": group.editable_token_count,
    }


def _trigger_row(
    *,
    group: ConceptGroup,
    tokenizer: Any,
    trigger: Sequence[int],
    objective_value: float,
    max_steps: int,
    accepted_steps: int,
    visited_sequences: int,
) -> dict[str, Any]:
    """One UNIVERSAL_TRIGGERS_SCHEMA row per non-control concept group.

    Carries the group's 7-tuple identity, the optimized trigger (ids +
    ``trigger_text == tokenizer.decode(trigger_tokens)``), the final
    group-aggregated objective ``g``, the optimization metadata, and ``n_pairs``
    (the minibatch population). The emitted dict's columns are EXACTLY
    ``schemas.UNIVERSAL_TRIGGERS_SCHEMA`` (so
    ``pa.Table.from_pylist([...], schema=UNIVERSAL_TRIGGERS_SCHEMA)`` round-trips
    without narrowing). ``model_optimize`` concatenates these per-group rows
    across targets and writes ``paths.universal_triggers_path()``.
    """

    first = group.rows[0]
    return {
        # The 7 identity columns come straight from the canonical group.key (built
        # by universal_gcg.group_key), so the trigger artifact cannot drift from
        # the grouping/lookup key. adversarial_strength is NOT part of the key (k
        # is carried by strength_index there) but IS a schema column, so it is
        # read from the group's rows.
        **dict(zip(GROUP_KEY_COLUMNS, group.key, strict=True)),
        "adversarial_strength": float(first["adversarial_strength"]),
        "trigger_tokens": [int(token_id) for token_id in trigger],
        "trigger_text": _decode_tokens(tokenizer, trigger),
        "objective_value": float(objective_value),
        "max_steps": int(max_steps),
        "accepted_steps": int(accepted_steps),
        "visited_sequences": int(visited_sequences),
        "n_pairs": len(group.rows),
    }


class TokenTriggerEditSpace:
    """EditSpace[list[int]] adapter for the universal token-trigger GCG attack.

    Delegates every method to the existing token helpers in optimize.py and
    universal_gcg.py so that the generic ``run_gcg`` loop can drive the same
    attack that ``build_universal_gcg_candidates`` implements inline.

    The byte-identity-critical contract: ``digest(trigger)`` operates in the
    **BOS-padded** space (``[BOS, *trigger]``), matching the ``visited`` set
    seeded in ``build_universal_gcg_candidates`` and the digests computed by
    ``sample_candidate_sequences``.  The bare-trigger digest is a DIFFERENT
    value and must NOT be used as the dedup key.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        cfg: FasterGcgConfig,
        embedding_weight: t.Tensor,
        allowed_token_ids: Sequence[int],
        pool_device: t.device | str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.embedding_weight = embedding_weight
        # ``pool_device`` places only the allowed-embedding POOL; the source
        # snapshot stays where the gradient path put it (CPU). ``None`` keeps
        # the pool with the source, i.e. the pre-existing behavior.
        self._allowed_embedding_cache: AllowedEmbeddingCache | None = (
            AllowedEmbeddingCache(
                embedding_weight=embedding_weight,
                allowed_token_ids=allowed_token_ids,
                device=pool_device,
            )
        )
        self.allowed_token_ids = self._allowed_embedding_cache.allowed_token_ids
        self.bos_token_id = _bos_token_id(tokenizer)

    def validate(self, group: ConceptGroup) -> ValidatedGroup:
        """Validate every member row and wrap into a ValidatedGroup."""
        rows = _validate_group_rows(group)
        return ValidatedGroup(key_str=group.key_str, members=rows, payload=group)

    def is_trivial(self, validated: ValidatedGroup) -> bool:
        """A k=0 group is a no-adversary control (no trigger to optimise)."""
        return validated.payload.is_control

    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult:
        """Emit control candidates (source == attacked) for a k=0 group."""
        rows = [
            build_control_candidate(row=row, tokenizer=self.tokenizer)
            for (row, _source_tokens, _span) in validated.members
        ]
        return GroupResult(
            payload={"candidates": rows, "trigger": None},
        )

    def init_target(self, validated: ValidatedGroup) -> list[int]:
        """Seed the trigger from the smallest core prefix's last-k span."""
        return init_trigger(validated.payload)

    def digest(self, target: list[int]) -> str:
        """Return the BOS-padded trigger digest (the dedup/visited key).

        Uses ``[BOS, *target]`` so the digest lives in the SAME space as the
        seeds computed by ``build_universal_gcg_candidates`` and the dedup
        digests computed by ``sample_candidate_sequences``.

        The parameter is named ``target`` to match the ``EditSpace`` protocol
        (``EditSpace.digest(self, target: E) -> str``).
        """
        padded = _padded_trigger_tokens(
            trigger=target,
            bos_token_id=self.bos_token_id,
            context="universal GCG trigger view",
        )
        return token_sequence_digest(padded)

    def aggregate_gradient(
        self,
        per_member: Iterable[tuple[t.Tensor, tuple[dict[str, Any], list[int], EditableSpan]]],
    ) -> t.Tensor:
        """Mean the per-pair (k, D) trigger gradient slices."""
        return aggregate_trigger_gradient([(grad, member[2]) for (grad, member) in per_member])

    def rank_candidates(
        self,
        agg_grad: t.Tensor,
        current: list[int],
        cfg: FasterGcgConfig,
    ) -> dict[int, tuple[TokenReplacement, ...]]:
        """Rank token replacements and shift to BOS-padded coordinates (1..k)."""
        local = rank_trigger_replacements(
            trigger_gradient=agg_grad,
            embedding_weight=self.embedding_weight,
            current_trigger=current,
            allowed_token_ids=self.allowed_token_ids,
            top_k=cfg.top_k,
            distance_weight=cfg.distance_weight,
            bos_token_id=self.bos_token_id,
            allowed_embedding_cache=self._allowed_embedding_cache,
        )
        # rank_trigger_replacements returns trigger-local coords 0..k-1; shift to
        # BOS-padded frame 1..k so sample_candidate_sequences can guard position 0.
        return {
            local_pos + 1: tuple(
                TokenReplacement(
                    position=local_pos + 1,
                    token_id=replacement.token_id,
                    score=replacement.score,
                )
                for replacement in replacements_at
            )
            for local_pos, replacements_at in local.items()
        }

    def sample_candidates(
        self,
        proposals: dict[int, tuple[TokenReplacement, ...]],
        current: list[int],
        *,
        batch_size: int,
        temperature: float,
        rng: Any,
        visited: set[str],
    ) -> list[list[int]]:
        """Sample candidate triggers from the BOS-padded view; return bare triggers."""
        padded = _padded_trigger_tokens(
            trigger=current,
            bos_token_id=self.bos_token_id,
            context="universal GCG trigger view",
        )
        raw = sample_candidate_sequences(
            current_tokens=padded,
            replacements_by_position=proposals,
            batch_size=batch_size,
            temperature=temperature,
            rng=rng,
            visited_digests=visited,
        )
        out: list[list[int]] = []
        for candidate_seq in raw:
            bare = _strip_padded_trigger_tokens(
                candidate_seq.prompt_tokens,
                bos_token_id=self.bos_token_id,
                context="strip candidate trigger",
            )
            out.append(bare)
        return out

    def full_batch_pool_exhausted(
        self,
        proposals: dict[int, tuple[TokenReplacement, ...]],
        current: list[int],
        visited: AbstractSet[str],
    ) -> bool:
        """Check exact exhaustion in the token sampler's BOS-padded digest space."""

        padded = _padded_trigger_tokens(
            trigger=current,
            bos_token_id=self.bos_token_id,
            context="universal GCG trigger view",
        )
        return replacement_pool_exhausted(
            current_tokens=padded,
            replacements_by_position=proposals,
            visited_digests=visited,
        )

    def model_item(
        self,
        member: tuple[dict[str, Any], list[int], EditableSpan],
        target: list[int],
    ) -> tuple[dict[str, Any], list[int]]:
        """Substitute the trigger into this member's editable span."""
        row, source_tokens, span = member
        return (row, _sub_into_row(source_tokens, span, target))

    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        """Return group-level identity fields for progress events."""
        return _group_event_fields(validated.payload)

    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        """Return the narrow field set emitted on step_done events.

        Restores the original inline-loop field set (group_key_str + n_pairs only),
        so step_done events carry only the group identity, not the full 8-field
        group identity block. This keeps step_done events byte-identical to the
        pre-refactor inline loop.
        """
        return {
            "group_key_str": validated.key_str,
            "n_pairs": len(validated.payload.rows),
        }

    def finalize(
        self,
        validated: ValidatedGroup,
        target: list[int],
        *,
        objective: float,
        accepted_steps: int,
        visited_count: int,
    ) -> GroupResult:
        """Build final candidate rows and the trigger artifact row."""
        meta = FasterGcgFinalMetadata(
            max_steps=NonNegativeInt.parse(int(self.cfg.max_steps)),
            accepted_steps=NonNegativeInt.parse(accepted_steps),
            visited_sequences=NonNegativeInt.parse(visited_count),
        )
        candidate_rows = [
            _candidate_row(
                row=row,
                tokenizer=self.tokenizer,
                source_tokens=src,
                attacked_tokens=_sub_into_row(src, sp, target),
                objective=objective,
                metadata=meta,
            )
            for (row, src, sp) in validated.members
        ]
        trigger = _trigger_row(
            group=validated.payload,
            tokenizer=self.tokenizer,
            trigger=target,
            objective_value=objective,
            max_steps=int(self.cfg.max_steps),
            accepted_steps=accepted_steps,
            visited_sequences=visited_count,
        )
        return GroupResult(
            payload={"candidates": candidate_rows, "trigger": trigger},
        )


def _require_trigger_row(row: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    """Return a schema-normalized universal-trigger row or fail loudly."""

    expected_columns = UNIVERSAL_TRIGGERS_SCHEMA.names
    if set(row) != set(expected_columns):
        raise AxBenchBpbAdvError(
            f"{context} must contain exactly UNIVERSAL_TRIGGERS_SCHEMA columns: "
            f"expected {expected_columns}, observed {sorted(row)}"
        )
    try:
        table = pa.Table.from_pylist([dict(row)], schema=UNIVERSAL_TRIGGERS_SCHEMA)
    except (ArrowInvalid, ArrowTypeError, OverflowError, TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            f"{context} does not conform to UNIVERSAL_TRIGGERS_SCHEMA: {exc}"
        ) from exc
    if any(column.null_count for column in table.columns):
        raise AxBenchBpbAdvError(f"{context} must not contain null values")
    if table.column("trigger_tokens").combine_chunks().values.null_count:
        raise AxBenchBpbAdvError(f"{context} trigger_tokens must not contain null values")
    return table.to_pylist()[0]


def _trigger_rows_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Field-for-field equality, including IEEE-754 bits for float columns."""

    if left.keys() != right.keys():
        return False
    for name in left:
        lhs = left[name]
        rhs = right[name]
        if isinstance(lhs, float) and isinstance(rhs, float):
            if struct.pack(">d", lhs) != struct.pack(">d", rhs):
                return False
        elif lhs != rhs:
            return False
    return True


def build_universal_gcg_candidates(
    *,
    attack_rows: pa.Table,
    tokenizer: Any,
    cfg: FasterGcgConfig,
    embedding_weight: t.Tensor,
    allowed_token_ids: Sequence[int],
    objective: ObjectiveBatchFn,
    gradient_provider_batch: GradientBatchProvider,
    progress_callback: ProgressCallback | None = None,
    group_boundary_callback: Callable[[], None] | None = None,
    pool_device: t.device | str | None = None,
    restored_triggers: Mapping[str, Mapping[str, Any]] | None = None,
    on_trigger_complete: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[pa.Table, list[dict[str, Any]]]:
    """Build candidates from ONE universal trigger per concept group.

    Replaces the per-prefix ``build_faster_gcg_candidates`` on the production
    path. All Faster-GCG attack rows sharing a concept group's 7-tuple are
    optimised jointly against a single shared k-token trigger, which is then
    substituted into EVERY member prefix's last-k span (length-preserving) to
    yield one candidate per pair. k=0 groups are no-adversary controls
    (byte-identical to ``build_control_candidate``).

    Returns ``(candidates, trigger_rows)`` where ``trigger_rows`` is one plain
    dict per non-control group, each conforming to
    ``schemas.UNIVERSAL_TRIGGERS_SCHEMA`` (``model_optimize`` concatenates them
    across targets and writes ``paths.universal_triggers_path()``). The candidate
    rows carry the SAME ``faster_gcg_final``
    metadata as the per-prefix path (so the 1-pair group is byte-identical to the
    Phase-0 golden); the trigger identity lives in ``trigger_rows``.

    Determinism: candidate sampling is seeded on the group key (not an
    ``attack_row_id``); every RNG derives from ``(group.key_str, step,
    cfg.master_seed)``. The trigger is ranked/sampled through a BOS-padded view
    (the trigger itself has no BOS); ``visited`` tracks BOS-padded trigger digests
    (a 1:1 stand-in for the trigger). Full-batch groups retain the first ranked
    proposal pool for each current trigger. An empty bounded-rejection sample is
    a clean no-op unless every complete sequence derivable from that retained
    pool is already visited, in which case the group terminates after emitting
    the empty step. Partial-minibatch groups continue to rank on every step.

    ``group_boundary_callback`` (if given) is invoked once at the START of every
    group (control and non-control alike). The model-backed wiring passes
    ``BpbObjectiveCallbacks.clear_eval_cache`` so the per-(attack_row_id,digest)
    eval cache is evicted at each group boundary (its bound; see v2 C3).
    """

    _validate_attack_rows(attack_rows)
    groups = concept_groups(attack_rows.to_pylist())
    non_control_by_key = {group.key_str: group for group in groups if not group.is_control}
    restored = dict(restored_triggers or {})
    unknown_restored = sorted(set(restored) - set(non_control_by_key))
    if unknown_restored:
        raise AxBenchBpbAdvError(
            "restored Faster-GCG trigger keys do not belong to this build: "
            f"{unknown_restored}"
        )
    consumed_restored: set[str] = set()
    space = TokenTriggerEditSpace(
        tokenizer=tokenizer,
        cfg=cfg,
        embedding_weight=embedding_weight,
        allowed_token_ids=allowed_token_ids,
        pool_device=pool_device,
    )

    def _restore(validated: ValidatedGroup) -> GroupResult | None:
        stored = restored.get(validated.key_str)
        if stored is None:
            return None
        if validated.key_str in consumed_restored:
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} was consumed more than once"
            )
        normalized = _require_trigger_row(
            stored,
            context=f"restored Faster-GCG trigger {validated.key_str!r}",
        )
        accepted_steps = int(normalized["accepted_steps"])
        max_steps = int(normalized["max_steps"])
        visited_sequences = int(normalized["visited_sequences"])
        objective = float(normalized["objective_value"])
        if accepted_steps < 0 or accepted_steps > max_steps:
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} accepted_steps "
                f"must be in [0, {max_steps}], observed {accepted_steps}"
            )
        if visited_sequences < 1:
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} "
                f"visited_sequences must be >= 1, observed {visited_sequences}"
            )
        if accepted_steps >= visited_sequences:
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} accepted_steps "
                "must be smaller than visited_sequences, observed "
                f"accepted_steps={accepted_steps}, visited_sequences={visited_sequences}"
            )
        if math.isnan(objective) or objective == -math.inf:
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} has impossible "
                f"objective_value {objective!r}"
            )
        result = space.finalize(
            validated,
            [int(token_id) for token_id in normalized["trigger_tokens"]],
            objective=objective,
            accepted_steps=accepted_steps,
            visited_count=visited_sequences,
        )
        reconstructed = _require_trigger_row(
            result.payload["trigger"],
            context=f"reconstructed Faster-GCG trigger {validated.key_str!r}",
        )
        if not _trigger_rows_equal(normalized, reconstructed):
            differing = [
                name
                for name in UNIVERSAL_TRIGGERS_SCHEMA.names
                if not _trigger_rows_equal(
                    {name: normalized[name]},
                    {name: reconstructed[name]},
                )
            ]
            raise AxBenchBpbAdvError(
                f"restored Faster-GCG trigger {validated.key_str!r} does not "
                f"reconstruct exactly; differing fields: {differing}"
            )
        consumed_restored.add(validated.key_str)
        return result

    def _commit(validated: ValidatedGroup, result: GroupResult) -> None:
        if on_trigger_complete is None:
            return
        if not isinstance(result.payload, Mapping) or set(result.payload) != {
            "candidates",
            "trigger",
        }:
            raise AxBenchBpbAdvError(
                f"completed Faster-GCG group {validated.key_str!r} has an invalid payload"
            )
        trigger = result.payload["trigger"]
        if not isinstance(trigger, Mapping):
            raise AxBenchBpbAdvError(
                f"completed Faster-GCG group {validated.key_str!r} must contain one trigger"
            )
        normalized = _require_trigger_row(
            trigger,
            context=f"completed Faster-GCG trigger {validated.key_str!r}",
        )
        on_trigger_complete(validated.key_str, normalized)

    results = run_gcg(
        edit_space=space,
        groups=groups,
        objective_batch=objective,
        gradient_provider_batch=gradient_provider_batch,
        cfg=cfg,
        progress_callback=progress_callback,
        group_boundary_callback=group_boundary_callback,
        full_batch_pool_exhausted=space.full_batch_pool_exhausted,
        restore_group_result=_restore if restored else None,
        commit_group_result=_commit if on_trigger_complete is not None else None,
    )
    if consumed_restored != set(restored):
        missing = sorted(set(restored) - consumed_restored)
        raise AxBenchBpbAdvError(
            f"restored Faster-GCG trigger keys were not consumed exactly once: {missing}"
        )
    candidate_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    for result in results:
        candidate_rows.extend(result.payload["candidates"])
        if result.payload["trigger"] is not None:
            trigger_rows.append(result.payload["trigger"])
    return (pa.Table.from_pylist(candidate_rows, schema=CANDIDATES_SCHEMA), trigger_rows)


def parse_gepa_token_candidate(
    value: str,
    *,
    expected_len: int,
    vocab_size: int,
) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AxBenchBpbAdvError(f"GEPA candidate is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise AxBenchBpbAdvError("GEPA candidate must be a JSON list")
    tokens: list[int] = []
    for item in parsed:
        if not isinstance(item, int):
            raise AxBenchBpbAdvError("GEPA candidate token ids must be integers")
        if item < 0 or item >= vocab_size:
            raise AxBenchBpbAdvError(
                f"GEPA candidate token id {item} is outside vocab size {vocab_size}"
            )
        tokens.append(int(item))
    if len(tokens) != expected_len:
        raise AxBenchBpbAdvError(
            f"GEPA candidate length {len(tokens)} does not match expected {expected_len}"
        )
    return tokens


def build_gepa_candidates(
    *,
    attack_rows: pa.Table,
    tokenizer: Any,
    optimizer: GepaOptimizer,
    progress_callback: ProgressCallback | None = None,
) -> pa.Table:
    """Build GEPA candidates from fixed-length optimized token-id artifacts."""

    _validate_attack_rows(attack_rows)
    rows: list[dict[str, Any]] = []
    source_rows = [
        row
        for row in attack_rows.to_pylist()
        if str(row["adversarial_method"]) == AdversarialMethod.GEPA.value
    ]
    total_rows = len(source_rows)
    for row_index, row in enumerate(source_rows, start=1):
        source_tokens, span = _validate_attack_row(row)
        seed_tokens = source_tokens[span.start:span.stop]
        _emit(
            progress_callback,
            event="row_start",
            row_index=row_index,
            total_rows=total_rows,
            attack_row_id=str(row["attack_row_id"]),
            target_model=str(row["target_model"]),
            method_name=str(row["method_name"]),
            adversarial_strength=float(row["adversarial_strength"]),
            editable_token_count=span.count,
        )
        replacement = [
            int(token_id)
            for token_id in optimizer(
                row,
                seed_tokens,
                int(row["adversarial_strength"]),
            )
        ]
        attacked = replace_editable_tokens(source_tokens, span, replacement)
        rows.append(
            _candidate_row(
                row=row,
                tokenizer=tokenizer,
                source_tokens=source_tokens,
                attacked_tokens=attacked,
                objective=None,
                metadata=GepaFinalMetadata(),
            )
        )
        _emit(
            progress_callback,
            event="row_done",
            row_index=row_index,
            total_rows=total_rows,
            attack_row_id=str(row["attack_row_id"]),
            target_model=str(row["target_model"]),
            method_name=str(row["method_name"]),
            adversarial_strength=float(row["adversarial_strength"]),
            editable_token_count=span.count,
        )
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def build_additive_direction_candidates(
    *,
    attack_rows: pa.Table,
    tokenizer: Any,
) -> pa.Table:
    _validate_attack_rows(attack_rows)
    rows = [
        build_control_candidate(row=row, tokenizer=tokenizer)
        for row in attack_rows.to_pylist()
        if str(row["adversarial_method"]) == AdversarialMethod.ADDITIVE_DIRECTION.value
    ]
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def concat_candidate_tables(tables: Sequence[pa.Table]) -> pa.Table:
    non_empty = [table for table in tables if table.num_rows > 0]
    if not non_empty:
        return pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA)
    return pa.concat_tables(non_empty, promote_options="default")
