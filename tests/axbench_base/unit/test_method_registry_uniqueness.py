"""Slice 16 audit — pin method-registry uniqueness invariant.

Map P3-3 / C27: ``method_registry._build_all_methods`` does NOT
explicitly assert name uniqueness. A typo could silently produce two
methods with the same name; ``get_method_by_name`` would return the
first. This test pins the invariant so a future PR that adds a
duplicate name fails immediately.

Also pins:

- ``ALL_METHODS`` is non-empty (registry never collapses to empty).
- ``get_method_by_name`` raises ``AxBenchBaseError`` for unknown names
  (defensive contract documented at ``method_registry.py``).
- The 35-entry total per map / contract C26.
- The 17-vs-18 positive/negative split documented in C26.
- The 3 ``requires_negative_sampling=True`` entries per C28.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    get_method_by_name,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
)


def test_method_registry_is_non_empty() -> None:
    assert len(ALL_METHODS) > 0


def test_method_registry_total_count_pins_c26() -> None:
    """Map contract C26: 35 methods total.

    Original 25 (incl. the 4 no-steering controls) + 4 feature-GCG Phase-2
    methods + 2 class-margin-alpha methods + 4 feature-DiffMean conditioning-bag
    methods = 35 total.
    """
    assert len(ALL_METHODS) == 35, (
        f"Slice 16 audit: ALL_METHODS expected 35 entries (C26+Task1+#593+alpha); got "
        f"{len(ALL_METHODS)}. If the registry deliberately changed, "
        f"update this test AND the map's contract table."
    )


def test_method_registry_names_unique() -> None:
    """Pin P3-3 / C27: no duplicate method names.

    Programmatic construction means a typo could silently produce two
    methods with the same name; ``get_method_by_name`` returns the
    first. This test catches it at registry-build time.
    """
    names = [m.name for m in ALL_METHODS]
    seen: set[str] = set()
    duplicates: list[str] = []
    for n in names:
        if n in seen:
            duplicates.append(n)
        seen.add(n)
    assert not duplicates, (
        f"Slice 16 audit: duplicate method names detected: {duplicates}. "
        f"Each entry in ALL_METHODS must have a unique ``name``."
    )


def test_positive_vs_negative_mode_split_pins_c26() -> None:
    """Map contract C26: 16 positive + 17 negative methods.

    Original 12 positive + 2 feature-GCG + 2 feature-DiffMean = 16.
    Original 13 negative + 2 feature-GCG + 2 feature-DiffMean = 17.
    """
    n_positive = sum(1 for m in ALL_METHODS if m.mode == SteeringMode.POSITIVE)
    n_negative = sum(1 for m in ALL_METHODS if m.mode == SteeringMode.NEGATIVE)
    assert n_positive == 17, f"Expected 17 positive methods, got {n_positive}"
    assert n_negative == 18, f"Expected 18 negative methods, got {n_negative}"


def test_requires_negative_sampling_count_pins_c28() -> None:
    """Map contract C28: exactly 3 methods set ``requires_negative_sampling=True``."""
    count = sum(1 for m in ALL_METHODS if m.requires_negative_sampling)
    assert count == 3, (
        f"Slice 16 audit (C28): expected 3 methods with "
        f"requires_negative_sampling=True, got {count}"
    )


def test_get_method_by_name_unknown_raises() -> None:
    """Defensive contract: unknown name raises ``AxBenchBaseError``."""
    with pytest.raises(AxBenchBaseError):
        get_method_by_name("definitely-not-a-real-method-xyz")


def test_get_method_by_name_round_trip() -> None:
    """Every entry in ``ALL_METHODS`` is fetchable by its ``name``."""
    for m in ALL_METHODS:
        out = get_method_by_name(m.name)
        assert out is m, f"get_method_by_name({m.name!r}) returned a different object"
