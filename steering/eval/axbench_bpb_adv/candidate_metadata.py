"""Typed metadata payloads stored in ``candidate_metadata_json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from beartype import beartype

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.refined_types import NonNegativeInt


@dataclass(frozen=True)
class NoAdversaryControlMetadata:
    candidate: Literal["no_adversary_control"] = "no_adversary_control"


@dataclass(frozen=True)
class GepaFinalMetadata:
    candidate: Literal["gepa_final"] = "gepa_final"


@beartype
@dataclass(frozen=True)
class FasterGcgFinalMetadata:
    max_steps: NonNegativeInt
    accepted_steps: NonNegativeInt
    visited_sequences: NonNegativeInt
    candidate: Literal["faster_gcg_final"] = "faster_gcg_final"


CandidateMetadata = (
    NoAdversaryControlMetadata | GepaFinalMetadata | FasterGcgFinalMetadata
)


def candidate_metadata_to_dict(metadata: CandidateMetadata) -> dict[str, Any]:
    """Serialize typed metadata to the stable YAML/JSON mapping shape."""

    if isinstance(metadata, (NoAdversaryControlMetadata, GepaFinalMetadata)):
        return {"candidate": metadata.candidate}
    return {
        "candidate": metadata.candidate,
        "max_steps": int(metadata.max_steps),
        "accepted_steps": int(metadata.accepted_steps),
        "visited_sequences": int(metadata.visited_sequences),
    }


def candidate_metadata_to_json(metadata: CandidateMetadata) -> str:
    return json.dumps(
        candidate_metadata_to_dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _int_field(payload: dict[str, Any], key: str) -> NonNegativeInt:
    value = payload.get(key)
    try:
        return NonNegativeInt.parse(value)
    except TypeError as exc:
        raise AxBenchBpbAdvError(
            "candidate_metadata_json faster_gcg_final field "
            f"{key!r} must be a non-negative integer, got {value!r}"
        ) from exc


def parse_candidate_metadata_json(raw: Any) -> CandidateMetadata:
    """Parse known candidate metadata payloads.

    Returns one of the dataclass-backed metadata variants, or fails loud. The
    candidate stage writes this field for every candidate, so missing or malformed
    metadata means the artifact is corrupt rather than optional.
    """

    if raw is None:
        raise AxBenchBpbAdvError("candidate_metadata_json is missing")
    if raw == "":
        raise AxBenchBpbAdvError("candidate_metadata_json is empty")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            f"candidate_metadata_json is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise AxBenchBpbAdvError(
            "candidate_metadata_json must decode to an object, got "
            f"{type(parsed).__name__}"
        )
    if "candidate" not in parsed:
        raise AxBenchBpbAdvError(
            "candidate_metadata_json is missing required 'candidate' field"
        )
    candidate = parsed.get("candidate")
    if candidate == "no_adversary_control":
        return NoAdversaryControlMetadata()
    if candidate == "gepa_final":
        return GepaFinalMetadata()
    if candidate == "faster_gcg_final":
        max_steps = _int_field(parsed, "max_steps")
        accepted_steps = _int_field(parsed, "accepted_steps")
        visited_sequences = _int_field(parsed, "visited_sequences")
        if accepted_steps > max_steps:
            raise AxBenchBpbAdvError(
                "candidate_metadata_json faster_gcg_final has "
                f"accepted_steps={accepted_steps} > max_steps={max_steps}"
            )
        if visited_sequences < accepted_steps + 1:
            raise AxBenchBpbAdvError(
                "candidate_metadata_json faster_gcg_final has "
                f"visited_sequences={visited_sequences}, expected at least "
                f"accepted_steps + 1 ({accepted_steps + 1})"
            )
        return FasterGcgFinalMetadata(
            max_steps=max_steps,
            accepted_steps=accepted_steps,
            visited_sequences=visited_sequences,
        )
    raise AxBenchBpbAdvError(
        "candidate_metadata_json has unknown candidate tag "
        f"{candidate!r}; expected no_adversary_control, gepa_final, or "
        "faster_gcg_final"
    )
