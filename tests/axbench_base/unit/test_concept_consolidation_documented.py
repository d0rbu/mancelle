"""Pin the ``Concept`` fields consumed by ``axbench_bpb``.

The canonical type lives in ``axbench_base``. These tests keep the shared field
subset explicit so a future refactor cannot silently break its consumer.

The fields slice 17 needs (per slice-17 mapper §1.19):

- ``concept_id`` (int)
- ``feature_index`` (int)
- ``description`` (str)

These fields define the consumer-facing surface of ``Concept``.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, get_args, get_origin, get_type_hints

from steering.eval.axbench_base.types import Concept
from tests.runtime_attribute_helpers import set_runtime_attribute

# The field subset slice 17 imports from this Concept. If a future
# axbench_base refactor renames or removes any of these, slice 17's
# import will fail. This list is the canonical contract.
_SLICE_17_REQUIRED_FIELDS = frozenset(
    {"concept_id", "feature_index", "description"}
)


def _static_base_type(field_name: str) -> object:
    hint = get_type_hints(Concept, include_extras=True)[field_name]
    return get_args(hint)[0] if get_origin(hint) is Annotated else hint


def test_concept_is_frozen_dataclass() -> None:
    """Pin the dataclass identity (slice 17 calls ``dataclasses.asdict`` on it)."""
    assert dataclasses.is_dataclass(Concept)
    fields = dataclasses.fields(Concept)
    assert len(fields) > 0


def test_slice_17_required_fields_present() -> None:
    """Pin the §1.19 cross-slice contract."""
    field_names = {f.name for f in dataclasses.fields(Concept)}
    missing = _SLICE_17_REQUIRED_FIELDS - field_names
    assert not missing, (
        f"§1.19 violation: Concept missing fields slice 17 needs: "
        f"{missing}. axbench_bpb relies on these field names for the "
        f"_Concept consolidation."
    )


def test_concept_id_is_int() -> None:
    """``concept_id`` retains ``int`` as its static base type."""
    assert _static_base_type("concept_id") is int


def test_feature_index_is_int() -> None:
    """``feature_index`` retains ``int`` as its static base type."""
    assert _static_base_type("feature_index") is int


def test_description_is_str() -> None:
    """``description`` retains ``str`` as its static base type."""
    assert _static_base_type("description") is str


def test_concept_is_frozen() -> None:
    """``Concept`` is frozen (so ``dataclasses.asdict`` is safe in
    multi-thread judge code).
    """
    c = Concept(
        concept_id=0,
        feature_index=1,
        layer=8,
        sae_source="src",
        description="d",
        corpus_count=1,
        description_source="manual",
    )
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(c, "concept_id", 99)
