"""Deterministic fixed-block primitives for token-example materialization.

This module intentionally owns no model, tokenizer, parquet, cache, or runner
objects. Keeping block identity and diagnostic tensor assembly pure makes the
scheduling contract independently testable. Production model forwards consume
each ``PreparedDoc.model_token_ids`` tuple at its true length; the padded tensor
is only a structural/probe surface and is never a model input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import torch as t
from beartype import beartype
from beartype.vale import Is

from steering.eval.autointerp_filter.types import (
    AutoInterpDataError,
    ScoringTarget,
    SourceDocRef,
)
from steering.refined_types import (
    BinaryInt,
    BinaryIntLike,
    NonEmptyNonNegativeIntTuple,
    NonEmptyNonNegativeIntTupleLike,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
    StrTuple,
    StrTupleLike,
    ZeroInt,
    ZeroIntLike,
)

if TYPE_CHECKING:
    _ScoringTargetListLike = list[ScoringTarget]
    _TargetSlotsLike = tuple[ScoringTarget | None, ...]
else:
    _ScoringTargetListLike = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list) and all(isinstance(item, ScoringTarget) for item in value)
            )
        ],
    ]
    _TargetSlotsLike = Annotated[
        tuple,
        Is[
            lambda value: (
                isinstance(value, tuple)
                and all(item is None or isinstance(item, ScoringTarget) for item in value)
            )
        ],
    ]


class MaterializationPhase(StrEnum):
    """The serial builder phase that requested a prepared document."""

    positive = "positive"
    intruder = "intruder"


@dataclass(frozen=True, init=False)
class TargetBlock:
    """One feature-ID-addressed fixed-slot materialization block."""

    ordinal: NonNegativeInt
    block_id: NonNegativeInt
    batch_size: PositiveInt
    targets_by_slot: tuple[ScoringTarget | None, ...]

    @beartype
    def __init__(
        self,
        ordinal: NonNegativeIntLike,
        block_id: NonNegativeIntLike,
        batch_size: PositiveIntLike,
        targets_by_slot: _TargetSlotsLike,
    ) -> None:
        object.__setattr__(self, "ordinal", NonNegativeInt.parse(ordinal))
        object.__setattr__(self, "block_id", NonNegativeInt.parse(block_id))
        object.__setattr__(self, "batch_size", PositiveInt.parse(batch_size))
        object.__setattr__(self, "targets_by_slot", targets_by_slot)
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.targets_by_slot) != self.batch_size:
            raise AutoInterpDataError(
                "target block slot count does not match batch size: "
                f"{len(self.targets_by_slot)} != {self.batch_size}"
            )
        if not any(target is not None for target in self.targets_by_slot):
            raise AutoInterpDataError("target block cannot be entirely empty")
        target_keys: set[str] = set()
        for slot, target in enumerate(self.targets_by_slot):
            if target is None:
                continue
            feature_id = target.feature_ids[0]
            if target.target_key in target_keys:
                raise AutoInterpDataError(
                    f"target block target keys must be unique; got {target.target_key!r}"
                )
            target_keys.add(target.target_key)
            expected_block, expected_slot = divmod(feature_id, self.batch_size)
            if (expected_block, expected_slot) != (self.block_id, slot):
                raise AutoInterpDataError(
                    "target is in the wrong deterministic materialization slot: "
                    f"feature={feature_id} expected=({expected_block}, {expected_slot}) "
                    f"actual=({self.block_id}, {slot})"
                )

    @property
    def targets(self) -> tuple[ScoringTarget, ...]:
        return tuple(target for target in self.targets_by_slot if target is not None)


@dataclass(frozen=True, init=False)
class PreparedDoc:
    """CPU-prepared source row whose model surface is fully explicit."""

    block_ordinal: NonNegativeInt
    slot: NonNegativeInt
    target_key: NonEmptyStr
    feature_id: NonNegativeInt
    phase: MaterializationPhase
    doc_ref: SourceDocRef
    source_text_hash: NonEmptyStr
    model_token_ids: NonEmptyNonNegativeIntTuple
    real_token_ids: NonEmptyNonNegativeIntTuple
    token_strings: StrTuple
    token_offset: NonNegativeInt
    activation_slice_start: BinaryInt

    @beartype
    def __init__(
        self,
        block_ordinal: NonNegativeIntLike,
        slot: NonNegativeIntLike,
        target_key: NonEmptyStrLike,
        feature_id: NonNegativeIntLike,
        phase: MaterializationPhase,
        doc_ref: SourceDocRef,
        source_text_hash: NonEmptyStrLike,
        model_token_ids: NonEmptyNonNegativeIntTupleLike,
        real_token_ids: NonEmptyNonNegativeIntTupleLike,
        token_strings: StrTupleLike,
        token_offset: NonNegativeIntLike,
        activation_slice_start: BinaryIntLike,
    ) -> None:
        object.__setattr__(self, "block_ordinal", NonNegativeInt.parse(block_ordinal))
        object.__setattr__(self, "slot", NonNegativeInt.parse(slot))
        object.__setattr__(self, "target_key", NonEmptyStr.parse(target_key))
        object.__setattr__(self, "feature_id", NonNegativeInt.parse(feature_id))
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "doc_ref", doc_ref)
        object.__setattr__(self, "source_text_hash", NonEmptyStr.parse(source_text_hash))
        object.__setattr__(
            self,
            "model_token_ids",
            NonEmptyNonNegativeIntTuple.parse(model_token_ids),
        )
        object.__setattr__(
            self,
            "real_token_ids",
            NonEmptyNonNegativeIntTuple.parse(real_token_ids),
        )
        object.__setattr__(self, "token_strings", StrTuple.parse(token_strings))
        object.__setattr__(self, "token_offset", NonNegativeInt.parse(token_offset))
        object.__setattr__(
            self,
            "activation_slice_start",
            BinaryInt.parse(activation_slice_start),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.real_token_ids) != len(self.token_strings):
            raise AutoInterpDataError(
                "prepared document real-token/token-string length mismatch: "
                f"{len(self.real_token_ids)} != {len(self.token_strings)}"
            )
        expected_model_len = len(self.real_token_ids) + self.activation_slice_start
        if len(self.model_token_ids) != expected_model_len:
            raise AutoInterpDataError(
                "prepared document model surface length does not match explicit "
                f"injection slice: {len(self.model_token_ids)} != {expected_model_len}"
            )
        real_slice = self.model_token_ids[
            self.activation_slice_start : self.activation_slice_start + len(self.real_token_ids)
        ]
        if real_slice != self.real_token_ids:
            raise AutoInterpDataError(
                "prepared document model surface does not contain the declared real "
                "token sequence at activation_slice_start"
            )


if TYPE_CHECKING:
    _PreparedRowsLike = tuple[PreparedDoc | None, ...]
else:
    _PreparedRowsLike = Annotated[
        tuple,
        Is[
            lambda value: (
                isinstance(value, tuple)
                and all(item is None or isinstance(item, PreparedDoc) for item in value)
            )
        ],
    ]


@dataclass(frozen=True, init=False)
class PreparedBatch:
    """One bounded-width diagnostic row per deterministic feature slot."""

    block: TargetBlock
    rows: tuple[PreparedDoc | None, ...]
    model_width: PositiveInt
    pad_token_id: ZeroInt = ZeroInt.parse(0)
    _cpu_token_tensor: t.Tensor = field(init=False, repr=False, compare=False)

    @beartype
    def __init__(
        self,
        block: TargetBlock,
        rows: _PreparedRowsLike,
        model_width: PositiveIntLike,
        pad_token_id: ZeroIntLike = 0,
    ) -> None:
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "model_width", PositiveInt.parse(model_width))
        object.__setattr__(self, "pad_token_id", ZeroInt.parse(pad_token_id))
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.rows) != self.block.batch_size:
            raise AutoInterpDataError(
                "prepared batch row count does not match target block batch size: "
                f"{len(self.rows)} != {self.block.batch_size}"
            )
        for slot, (target, row) in enumerate(
            zip(self.block.targets_by_slot, self.rows, strict=True)
        ):
            if row is None:
                continue
            if target is None:
                raise AutoInterpDataError(
                    f"prepared batch slot {slot} has work but no block target"
                )
            if row.slot != slot or row.block_ordinal != self.block.ordinal:
                raise AutoInterpDataError(
                    f"prepared batch row is assigned to the wrong slot/block at slot {slot}"
                )
            if row.target_key != target.target_key or row.feature_id != target.feature_ids[0]:
                raise AutoInterpDataError(f"prepared batch row target mismatch at slot {slot}")
            if len(row.model_token_ids) > self.model_width:
                raise AutoInterpDataError(
                    f"prepared row length {len(row.model_token_ids)} exceeds fixed "
                    f"model width {self.model_width} at slot {slot}"
                )
        object.__setattr__(self, "_cpu_token_tensor", self._assemble_token_tensor())

    @property
    def has_work(self) -> bool:
        return any(row is not None for row in self.rows)

    def token_tensor(self) -> t.Tensor:
        """Return a defensive copy of the diagnostic padded CPU surface."""
        return self._cpu_token_tensor.clone(memory_format=t.contiguous_format)

    def _assemble_token_tensor(self) -> t.Tensor:
        tokens = t.full(
            (self.block.batch_size, self.model_width),
            self.pad_token_id,
            dtype=t.long,
            device="cpu",
        )
        for slot, row in enumerate(self.rows):
            if row is None:
                continue
            row_tokens = t.tensor(row.model_token_ids, dtype=t.long, device="cpu")
            tokens[slot, : row_tokens.numel()] = row_tokens
        return tokens.contiguous()


def target_plan_hash(targets: list[ScoringTarget]) -> str:
    """Hash the exact ordered target plan consumed by bulk materialization."""
    payload = [
        {
            "target_key": target.target_key,
            "kind": target.kind.value,
            "feature_ids": list(target.feature_ids),
            "display_name": target.display_name,
        }
        for target in targets
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


if TYPE_CHECKING:
    _TargetBlockListLike = list[TargetBlock]
else:
    _TargetBlockListLike = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list) and all(isinstance(item, TargetBlock) for item in value)
            )
        ],
    ]


@beartype
def build_target_blocks(
    targets: _ScoringTargetListLike,
    *,
    batch_size: PositiveIntLike,
) -> _TargetBlockListLike:
    """Map singleton targets to immutable ``feature_id // batch_size`` blocks."""
    keys = [target.target_key for target in targets]
    if len(keys) != len(set(keys)):
        raise AutoInterpDataError("target plan contains duplicate target keys")
    slots_by_block: dict[int, list[ScoringTarget | None]] = {}
    for target in targets:
        feature_id = target.feature_ids[0]
        block_id, slot = divmod(feature_id, batch_size)
        slots = slots_by_block.setdefault(block_id, [None] * batch_size)
        previous = slots[slot]
        if previous is not None:
            raise AutoInterpDataError(
                "two targets map to the same deterministic materialization slot: "
                f"{previous.target_key!r} and {target.target_key!r}"
            )
        slots[slot] = target
    return [
        TargetBlock(
            ordinal=ordinal,
            block_id=block_id,
            batch_size=batch_size,
            targets_by_slot=tuple(slots_by_block[block_id]),
        )
        for ordinal, block_id in enumerate(sorted(slots_by_block))
    ]
