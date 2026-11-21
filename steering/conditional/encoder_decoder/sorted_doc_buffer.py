"""Sorted document buffer for O(log n) best-fit packing.

This module is shared between the live dataloader and the precompute script.
It lives in its own file to avoid circular imports between those two modules.
"""

from __future__ import annotations

from typing import Any, cast

from sortedcontainers import SortedList

# Type alias: (text_tokens, seg_features_or_None) — callers may append extra
# trailing payload elements (the packer carries (text_tokens, seg_features,
# provenance) 3-tuples); the buffer keys ONLY on len(entry[0]) and never
# inspects the rest. Using Any to avoid importing SegFeatures and creating
# circular dependencies.
DocEntry = tuple[list[int], Any] | tuple[list[int], Any, Any]


class SortedDocBuffer:
    """Document buffer sorted by token length for O(log n) best-fit lookup.

    Wraps ``SortedList`` keyed by ``len(doc_tokens)`` to replace the O(n)
    linear scan in packing loops with O(log n) bisect.

    Supports the same operations as a plain list buffer: append, len, bool,
    plus fast best-fit and shortest-doc queries.
    """

    __slots__ = ("_sorted",)

    def __init__(self) -> None:
        self._sorted: SortedList = SortedList(
            key=lambda entry: len(entry[0]),
        )

    def append(self, entry: DocEntry) -> None:
        self._sorted.add(entry)

    def __len__(self) -> int:
        return len(self._sorted)

    def __bool__(self) -> bool:
        return len(self._sorted) > 0

    def snapshot(self) -> list[DocEntry]:
        """Return entries in the exact internal iteration order.

        Equal-length insertion order is observable through both pop methods,
        so durable rebuilds must append this sequence without re-sorting it.
        """
        return list(self._sorted)

    def pop_best_fit(self, remaining: int) -> DocEntry | None:
        """Remove and return the largest doc with len <= remaining, or None."""
        idx = self._sorted.bisect_key_right(remaining)
        if idx == 0:
            return None
        return cast(DocEntry, self._sorted.pop(idx - 1))

    def pop_shortest(self) -> DocEntry:
        """Remove and return the shortest doc."""
        return cast(DocEntry, self._sorted.pop(0))
