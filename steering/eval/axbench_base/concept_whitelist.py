"""Stage 0a concept-candidate whitelist resolution.

Resolves ``cfg.concept_feature_whitelist_path``
(``accepted_feature_ids.txt`` or ``manual_feature_filter.yaml``) into a
validated, sorted, unique tuple of SAE feature ids, or ``None`` when no
whitelist is configured. Applied before Neuronpedia description fetching so a
curated feature list bounds the fetch.

This is concept-candidate filtering, distinct from the downstream
``feature_mask`` (a training/eval selection mask).
"""

from __future__ import annotations

from pathlib import Path

from errors import ManualFeatureSelectionError
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.manual_feature_ids import (
    parse_feature_ids_txt,
    parse_manual_feature_yaml_ids,
    validate_feature_ids_in_range,
)

_SUPPORTED_WHITELIST_SUFFIXES = (".yaml", ".yml", ".txt")


def resolve_concept_feature_whitelist(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
) -> tuple[int, ...] | None:
    path = cfg.concept_feature_whitelist_path
    if path is None:
        return None
    if not isinstance(path, str):
        raise AxBenchBaseError(
            "concept_feature_whitelist_path must be a string path or null, "
            f"got {type(path).__name__}"
        )
    try:
        ids = _parse_whitelist_path(path)
    except ManualFeatureSelectionError as exc:
        raise AxBenchBaseError(f"concept_feature_whitelist_path: {exc}") from exc

    if len(ids) != len(set(ids)):
        raise AxBenchBaseError("concept_feature_whitelist_path contains duplicate ids")
    if not ids:
        raise AxBenchBaseError(
            f"concept_feature_whitelist_path {path!r} resolved to an empty "
            "selection (no IDs uncommented?)"
        )
    try:
        validate_feature_ids_in_range(ids, n_sae_features=n_sae_features)
    except ManualFeatureSelectionError as exc:
        raise AxBenchBaseError(f"concept_feature_whitelist_path: {exc}") from exc
    return tuple(sorted(set(ids)))


def _parse_whitelist_path(path: str) -> list[int]:
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return parse_manual_feature_yaml_ids(path)
    if suffix == ".txt":
        return parse_feature_ids_txt(path)
    raise ManualFeatureSelectionError(
        f"concept_feature_whitelist_path {path!r} has unsupported extension "
        f"{Path(path).suffix!r}; expected one of {_SUPPORTED_WHITELIST_SUFFIXES} "
        "(.yaml/.yml manual filter or .txt accepted_feature_ids)"
    )
