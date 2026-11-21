"""Immutable bag of (feature_id, bucket) conditioning entries.

``FeatureConditioning`` is the value-object that represents a single segment's
SAE feature conditioning state in the feature-space GCG optimizer.  It is
designed to be constructed, mutated (returning a new instance), and passed to
the encoder, which handles FP8 ×16-padding internally.

Invariants
----------
* Entries are stored in **canonical order** (ascending ``feature_id``).
* No duplicate ``feature_id`` is allowed; ``with_added`` / ``with_replaced``
  and direct construction all raise :class:`FeatureGcgError` on violation.
* Both ``feature_id`` and ``bucket`` must be non-negative integers.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from beartype import beartype
from beartype.vale import Is
from phantom import Phantom

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.refined_types import NonNegativeIntLike


def _is_feature_conditioning_entries(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    seen_fids: set[int] = set()
    for entry in value:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return False
        fid, bucket = entry
        if (
            not isinstance(fid, int)
            or isinstance(fid, bool)
            or fid < 0
            or not isinstance(bucket, int)
            or isinstance(bucket, bool)
            or bucket < 0
            or fid in seen_fids
        ):
            return False
        seen_fids.add(fid)
    return True


class FeatureConditioningEntries(
    tuple[tuple[int, int], ...],
    Phantom,
    predicate=_is_feature_conditioning_entries,
):
    """A tuple of unique ``(feature_id, bucket)`` non-negative integer pairs."""


# Beartype's O(1) checking of parameterized containers consumes Python's global
# RNG to select an element. Feature-GCG promises not to perturb that state, so
# runtime annotations use the unsubscripted container plus a deterministic
# phantom predicate. Static checkers still see the exact builtin shape.
if TYPE_CHECKING:
    FeatureConditioningEntriesLike = tuple[tuple[int, int], ...]
    _FeatureConditioningIdTuple = tuple[int, ...]
    _FeatureConditioningDict = dict[int, int]
else:
    FeatureConditioningEntriesLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, FeatureConditioningEntries)],
    ]
    _FeatureConditioningIdTuple = tuple
    _FeatureConditioningDict = dict


@beartype
@dataclass(frozen=True)
class FeatureConditioning:
    """Immutable bag of ``(feature_id, bucket)`` conditioning entries.

    Entries are kept in canonical order (sorted ascending by
    ``feature_id``).  Duplicate ``feature_id`` values are rejected at
    construction and mutation.  Range constraint: ``feature_id >= 0`` and
    ``bucket >= 0``.
    """

    entries: FeatureConditioningEntriesLike  # sorted by __post_init__

    def __post_init__(self) -> None:
        sorted_entries = tuple(sorted(self.entries, key=lambda e: e[0]))
        object.__setattr__(self, "entries", sorted_entries)

    # ──────────────────────────────────────────────────────────────────────
    # Mutation helpers (return new instances)
    # ──────────────────────────────────────────────────────────────────────

    def with_added(
        self,
        fid: NonNegativeIntLike,
        bucket: NonNegativeIntLike,
    ) -> FeatureConditioning:
        """Return a new bag with ``(fid, bucket)`` added.

        Unary argument domains are enforced by :mod:`beartype`; duplicate
        feature IDs raise :class:`FeatureGcgError`.
        """
        for existing_fid, _ in self.entries:
            if existing_fid == fid:
                raise FeatureGcgError(f"feature_id {fid} already present")
        new_entries = tuple(sorted(self.entries + ((fid, bucket),), key=lambda e: e[0]))
        return FeatureConditioning(entries=new_entries)

    def with_replaced(
        self,
        slot: NonNegativeIntLike,
        fid: NonNegativeIntLike,
        bucket: NonNegativeIntLike,
    ) -> FeatureConditioning:
        """Return a new bag with ``entries[slot]`` replaced by ``(fid, bucket)``.

        *slot* is an index into the current (sorted) ``entries`` tuple.
        Unary argument domains are enforced by :mod:`beartype`. An out-of-range
        *slot* or an *fid* already present in a **different** slot raises
        :class:`FeatureGcgError`.
        """
        if slot >= len(self.entries):
            raise FeatureGcgError(
                f"slot {slot} out of range [0, {len(self.entries)})"
            )
        for i, (existing_fid, _) in enumerate(self.entries):
            if i != slot and existing_fid == fid:
                raise FeatureGcgError(
                    f"feature_id {fid} already present in slot {i}"
                )
        entries_list = list(self.entries)
        entries_list[slot] = (fid, bucket)
        new_entries = tuple(sorted(entries_list, key=lambda e: e[0]))
        return FeatureConditioning(entries=new_entries)

    def with_removed(self, slot: NonNegativeIntLike) -> FeatureConditioning:
        """Return a new bag with ``entries[slot]`` dropped.

        A negative *slot* is rejected by :mod:`beartype`; an upper-bound
        violation raises :class:`FeatureGcgError`.
        """
        if slot >= len(self.entries):
            raise FeatureGcgError(
                f"slot {slot} out of range [0, {len(self.entries)})"
            )
        new_entries = self.entries[:slot] + self.entries[slot + 1:]
        return FeatureConditioning(entries=new_entries)

    # ──────────────────────────────────────────────────────────────────────
    # Content digest
    # ──────────────────────────────────────────────────────────────────────

    def digest(self) -> str:
        """SHA-256 hex digest over canonical sorted ``(fid, bucket)`` pairs.

        Deterministic across processes (uses ``struct``-packed bytes, not
        Python's built-in ``hash``).  Permutation-invariant because entries
        are always sorted before hashing.
        """
        h = hashlib.sha256()
        for fid, bucket in self.entries:
            h.update(struct.pack("<QQ", fid, bucket))
        return h.hexdigest()

    # ──────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────

    def feature_ids(self) -> _FeatureConditioningIdTuple:
        """Return feature IDs in canonical (sorted) order."""
        return tuple(fid for fid, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    # ──────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, bag: _FeatureConditioningDict) -> FeatureConditioning:
        """Construct from a ``{feature_id: bucket}`` mapping.

        Negative keys or values are rejected by the typed constructor boundary.
        """
        entries = tuple(sorted(bag.items(), key=lambda e: e[0]))
        return cls(entries=entries)
