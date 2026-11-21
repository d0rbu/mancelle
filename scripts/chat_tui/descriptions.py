"""Offline feature descriptions: the local JSON mapping and its generator.

The reader and the writer live together so the on-disk format has a single
definition. Every id here is an ORIGINAL SAE feature id, which is the id space
the session ledger records: the self-conditioning provider translates compact
prediction-head rows through ``FeatureVocab.new_to_old_tensor()`` before
recording a segment, so a compacted checkpoint needs no extra translation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from nanochat.atomic_io import atomic_json_dump

# Ranked-artifact schema keys written by
# ``steering.eval.autointerp_filter.artifacts.score_to_ranked_yaml``.
_TARGETS_KEY: Final = "targets"
_FEATURE_IDS_KEY: Final = "feature_ids"
_DESCRIPTION_KEY: Final = "description"
_FEATURE_ID_KEY: Final = "feature_id"


class OfflineDescriptionService:
    """Synchronous ``{feature_id: text}`` lookups from a validated JSON file.

    Drop-in for the Neuronpedia-backed service at the steer menu's protocol
    boundary: callbacks fire immediately on the caller's thread, and features
    absent from the file resolve to an empty description (rendered as none).
    """

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"offline descriptions must be a JSON object mapping feature ids "
                f"to text, got {type(payload).__name__}: {path}"
            )
        descriptions: dict[int, str] = {}
        for key, value in payload.items():
            try:
                feature_id = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"offline description key {key!r} is not a feature id: {path}"
                ) from None
            if not isinstance(value, str):
                raise ValueError(
                    f"offline description for feature {feature_id} must be a "
                    f"string, got {type(value).__name__}: {path}"
                )
            descriptions[feature_id] = value
        self._descriptions = descriptions

    def __len__(self) -> int:
        return len(self._descriptions)

    def request_description(
        self,
        feature_id: int,
        callback: Callable[[int, str], None],
        /,
        *,
        on_start: Callable[[int], None] | None = None,
    ) -> None:
        if on_start is not None:
            on_start(feature_id)
        callback(feature_id, self._descriptions.get(int(feature_id), ""))

    def shutdown(self) -> None:
        """Protocol parity with the network-backed service; nothing to stop."""


def _ranked_feature_ids(target: Mapping[str, Any], *, index: int, path: Path) -> list[int]:
    raw_ids = target.get(_FEATURE_IDS_KEY)
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(
            f"autointerp target {index} has no non-empty {_FEATURE_IDS_KEY!r} list: {path}"
        )
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"autointerp target {index} has non-integer feature id {value!r}: {path}"
            )
    return [int(value) for value in raw_ids]


def _target_descriptions(target: Any, *, index: int, path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(feature id, text)`` pairs for one ranked target row.

    Mirrors ``artifacts.ranked_description``: a singleton target carries a bare
    string, a multi-feature target carries a per-feature list, and either shape
    is absent when no description was recorded for the target.
    """

    if not isinstance(target, Mapping):
        raise ValueError(
            f"autointerp target {index} is not a mapping, got {type(target).__name__}: {path}"
        )
    feature_ids = _ranked_feature_ids(target, index=index, path=path)
    description = target.get(_DESCRIPTION_KEY)
    if description is None:
        return
    if isinstance(description, str):
        if len(feature_ids) != 1:
            raise ValueError(
                f"autointerp target {index} pairs one {_DESCRIPTION_KEY!r} string with "
                f"{len(feature_ids)} feature ids {feature_ids}: {path}"
            )
        yield feature_ids[0], description
        return
    if not isinstance(description, list):
        raise ValueError(
            f"autointerp target {index} has a {_DESCRIPTION_KEY!r} of unsupported type "
            f"{type(description).__name__}: {path}"
        )
    declared = set(feature_ids)
    for entry in description:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"autointerp target {index} has a non-mapping {_DESCRIPTION_KEY!r} entry "
                f"{entry!r}: {path}"
            )
        feature_id = entry.get(_FEATURE_ID_KEY)
        text = entry.get(_DESCRIPTION_KEY)
        if isinstance(feature_id, bool) or not isinstance(feature_id, int):
            raise ValueError(
                f"autointerp target {index} has a {_DESCRIPTION_KEY!r} entry whose "
                f"{_FEATURE_ID_KEY!r} is not a feature id: {feature_id!r}: {path}"
            )
        if feature_id not in declared:
            raise ValueError(
                f"autointerp target {index} describes feature {feature_id}, which is absent "
                f"from its {_FEATURE_IDS_KEY!r} {feature_ids}: {path}"
            )
        if not isinstance(text, str):
            raise ValueError(
                f"autointerp target {index} has a non-string {_DESCRIPTION_KEY!r} for "
                f"feature {feature_id}, got {type(text).__name__}: {path}"
            )
        yield int(feature_id), text


def descriptions_from_autointerp_ranked(path: Path) -> dict[int, str]:
    """Extract ``{original SAE feature id: text}`` from one ranked YAML artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"autointerp ranked artifact does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"autointerp ranked artifact must be a YAML mapping, got "
            f"{type(payload).__name__}: {path}"
        )
    targets = payload.get(_TARGETS_KEY)
    if not isinstance(targets, list):
        raise ValueError(f"autointerp ranked artifact has no {_TARGETS_KEY!r} list: {path}")
    descriptions: dict[int, str] = {}
    for index, target in enumerate(targets):
        for feature_id, text in _target_descriptions(target, index=index, path=path):
            previous = descriptions.get(feature_id)
            if previous is not None and previous != text:
                raise ValueError(
                    f"autointerp ranked artifact gives feature {feature_id} conflicting "
                    f"descriptions {previous!r} and {text!r}: {path}"
                )
            descriptions[feature_id] = text
    return descriptions


def write_offline_descriptions(descriptions: Mapping[int, str], path: Path) -> None:
    """Write the ``{feature_id: text}`` JSON that ``OfflineDescriptionService`` loads."""

    for feature_id, text in descriptions.items():
        if isinstance(feature_id, bool) or not isinstance(feature_id, int):
            raise ValueError(f"offline description key {feature_id!r} is not a feature id: {path}")
        if not isinstance(text, str):
            raise ValueError(
                f"offline description for feature {feature_id} must be a string, "
                f"got {type(text).__name__}: {path}"
            )
    atomic_json_dump(
        {str(feature_id): descriptions[feature_id] for feature_id in sorted(descriptions)},
        path,
    )
