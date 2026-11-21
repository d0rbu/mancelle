"""Strict validation for stage-boundary concept JSON artifacts.

``validate_concepts`` is the pure in-memory invariant check over an
already-constructed ``list[Concept]``.

``load_and_validate_concepts`` and
``load_and_validate_candidate_concepts`` are the public stage-boundary
helpers that read JSON files from disk, wrap each entry into the
appropriate dataclass (catching missing / unexpected fields as a
``TypeError`` and re-raising as ``AxBenchBaseError``), and then run the
in-memory checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
)


def validate_concepts(concepts: list[Concept], n_sae_features: int) -> None:
    """Validate an in-memory list of ``Concept`` entries.

    Required-field presence is already guaranteed by the dataclass
    constructor, so this function only enforces cross-entry and value-range
    invariants:

    - empty list
    - non-contiguous ``concept_id`` (must be 0, 1, 2, ... in order)
    - duplicate ``feature_index``
    - ``feature_index`` outside ``[0, n_sae_features)``

    Unary field domains are enforced when :class:`Concept` is constructed.
    """
    if not concepts:
        raise AxBenchBaseError("concepts.json is empty")
    seen_fids: set[int] = set()
    for i, c in enumerate(concepts):
        if c.concept_id != i:
            raise AxBenchBaseError(
                f"concepts[{i}].concept_id = {c.concept_id}, expected {i} (contiguous)"
            )
        fid = c.feature_index
        if fid in seen_fids:
            raise AxBenchBaseError(
                f"concepts[{i}].feature_index = {fid} appears more than once"
            )
        if not (0 <= fid < n_sae_features):
            raise AxBenchBaseError(
                f"concepts[{i}].feature_index = {fid} out of [0, {n_sae_features}) for the SAE"
            )
        seen_fids.add(fid)


def validate_candidate_concepts(
    concepts: list[CandidateConcept], n_sae_features: int,
) -> None:
    """Validate the Stage 0a ``candidate_concepts.json`` pre-QC schema.

    Candidate rows deliberately do not have ``concept_id``. Stage 0c
    assigns IDs after lazy examples-QC, preserving the old eager-QC RNG
    semantics while keeping Stage 0a cheap.
    """
    if not concepts:
        raise AxBenchBaseError("candidate_concepts.json is empty")
    seen_fids: set[int] = set()
    for i, c in enumerate(concepts):
        fid = c.feature_index
        if fid in seen_fids:
            raise AxBenchBaseError(
                f"candidate_concepts[{i}].feature_index = {fid} appears more than once"
            )
        if not (0 <= fid < n_sae_features):
            raise AxBenchBaseError(
                f"candidate_concepts[{i}].feature_index = {fid} out of "
                f"[0, {n_sae_features}) for the SAE"
            )
        seen_fids.add(fid)


def load_and_validate_concepts(path: Path, n_sae_features: int) -> list[Concept]:
    """Read ``concepts.json`` from ``path``, wrap as ``Concept``, and validate.

    This is the canonical stage-boundary reader: every downstream stage
    should call this instead of parsing the JSON themselves.

    Raises ``AxBenchBaseError`` on:

    - top-level JSON not being an array
    - any array element not being a JSON object
    - an element missing a required field or containing an unexpected field
      (both surface as ``TypeError`` from the ``Concept.__init__`` call and
      are re-raised as ``AxBenchBaseError``)
    - any invariant violation detected by ``validate_concepts``
    """
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AxBenchBaseError(
            f"concepts.json at {path} must be a JSON array, got {type(raw).__name__}"
        )
    concepts: list[Concept] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AxBenchBaseError(
                f"concepts[{i}] at {path} must be a JSON object, got {type(item).__name__}"
            )
        item_dict = cast(dict[str, Any], item)
        try:
            concepts.append(Concept(**item_dict))
        except (TypeError, BeartypeCallHintParamViolation) as exc:
            raise AxBenchBaseError(
                f"concepts[{i}] at {path} has invalid fields: {exc}"
            ) from exc
    validate_concepts(concepts, n_sae_features)
    return concepts


def load_and_validate_candidate_concepts(
    path: Path, n_sae_features: int,
) -> list[CandidateConcept]:
    """Read and validate the Stage 0a pre-QC candidate pool.

    This intentionally does not accept the pre-refactor ``Concept``
    schema. Runs with old ``candidate_concepts.json`` artifacts must
    re-run Stage 0a.
    """
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AxBenchBaseError(
            f"candidate_concepts.json at {path} must be a JSON array, "
            f"got {type(raw).__name__}"
        )
    concepts: list[CandidateConcept] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AxBenchBaseError(
                f"candidate_concepts[{i}] at {path} must be a JSON object, "
                f"got {type(item).__name__}"
            )
        item_dict = cast(dict[str, Any], item)
        try:
            concepts.append(CandidateConcept(**item_dict))
        except (TypeError, BeartypeCallHintParamViolation) as exc:
            raise AxBenchBaseError(
                f"candidate_concepts[{i}] at {path} has invalid fields: {exc}. Runs with "
                "pre-refactor candidate_concepts.json artifacts must re-run "
                "Stage 0a."
            ) from exc
    validate_candidate_concepts(concepts, n_sae_features)
    return concepts
