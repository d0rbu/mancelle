"""Pure parsers/validators for manual feature-id selection files.

Shared by manual-coverage tooling, feature-mask and frequency-blacklist
resolution, AxBench-Base concept whitelisting, and analysis/benchmark tools.
Deliberately dependency-light (``re`` + ``yaml`` + the project-root ``errors``
module only) so it imports from any layer without dragging heavy packages or
risking an import cycle.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from errors import ManualFeatureSelectionError

_UNQUOTED_ZERO_PADDED_ID = re.compile(r"^[+-]?0[0-9]+$")
_QUOTED_DIGIT_ID = re.compile(r"^[0-9]+$")


def _read_utf8_text(path: Path, *, artifact_kind: str) -> str:
    """Read a feature-ID artifact with one stable public error contract."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManualFeatureSelectionError(
            f"{artifact_kind} {path} is not valid UTF-8: {exc}"
        ) from exc
    except OSError as exc:
        raise ManualFeatureSelectionError(
            f"could not read {artifact_kind} {path}: {exc}"
        ) from exc


def parse_manual_feature_yaml_ids(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.is_file():
        raise ManualFeatureSelectionError(f"manual feature YAML not found: {path}")
    text = _read_utf8_text(path, artifact_kind="manual feature YAML")
    plain_ids = re.fullmatch(r"(?:- (?:0|[1-9][0-9]*)\n)+", text) is not None
    try:
        if plain_ids:
            payload = [int(line[2:]) for line in text.splitlines()]
        else:
            loader = yaml.SafeLoader(text)
            try:
                root = loader.get_single_node()
                # Construction can mutate aliased nodes; retain only the original line.
                zero_padded_line = _unquoted_zero_padded_line(root)
                payload = loader.construct_document(root) if root is not None else None
            finally:
                loader.dispose()
    except yaml.YAMLError as exc:
        raise ManualFeatureSelectionError(
            f"manual feature YAML {path} is malformed: {exc}"
        ) from exc
    # Construction errors take precedence over zero-padding validation.
    if not plain_ids and zero_padded_line is not None:
        raise ManualFeatureSelectionError(
            "manual feature YAML zero-padded IDs must be quoted to avoid YAML "
            f"octal parsing (line {zero_padded_line})"
        )
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ManualFeatureSelectionError(
            "manual feature YAML must be a flat list of strict ints or quoted digits, "
            f"got {type(payload).__name__}"
        )
    out: list[int] = []
    seen: set[int] = set()
    for idx, raw in enumerate(payload):
        feature_id = coerce_feature_id(raw, idx)
        if feature_id in seen:
            raise ManualFeatureSelectionError(
                f"manual feature YAML contains duplicate id {feature_id}"
            )
        seen.add(feature_id)
        out.append(feature_id)
    return out


def _unquoted_zero_padded_line(root: yaml.Node | None) -> int | None:
    """Find the first octal-prone direct scalar before construction mutates nodes."""
    if not isinstance(root, yaml.SequenceNode):
        return None
    for item in root.value:
        if (
            isinstance(item, yaml.ScalarNode)
            and _UNQUOTED_ZERO_PADDED_ID.fullmatch(item.value.replace("_", ""))
            and (
                item.style is None
                or item.tag != "tag:yaml.org,2002:str"
            )
        ):
            return item.start_mark.line + 1
    return None


def coerce_feature_id(raw: Any, idx: int, *, context: str = "manual feature YAML") -> int:
    if isinstance(raw, bool):
        raise ManualFeatureSelectionError(
            f"{context} item {idx} must be a strict int or quoted digits, got {raw!r}"
        )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and _QUOTED_DIGIT_ID.fullmatch(raw):
        return int(raw)
    raise ManualFeatureSelectionError(
        f"{context} item {idx} must be a strict int or quoted digits, got {raw!r}"
    )


def validate_feature_ids_in_range(ids: list[int], *, n_sae_features: int) -> None:
    for feature_id in ids:
        if feature_id < 0 or feature_id >= n_sae_features:
            raise ManualFeatureSelectionError(
                f"manual feature id {feature_id} out of range [0, {n_sae_features})"
            )


def parse_feature_ids_txt(path: str | Path) -> list[int]:
    """Parse a text feature-ID file containing one integer per line.

    This is the canonical parser for text ID artifacts throughout the repo.
    Blank lines and comments (including trailing ``#`` comments) are skipped.
    Non-integer tokens and duplicate IDs are rejected.
    """
    path = Path(path)
    if not path.is_file():
        raise ManualFeatureSelectionError(f"feature id txt file not found: {path}")
    text = _read_utf8_text(path, artifact_kind="feature id txt file")
    out: list[int] = []
    seen: set[int] = set()
    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        try:
            feature_id = int(token)
        except ValueError as exc:
            raise ManualFeatureSelectionError(
                f"feature id txt file {path} line {line_number}: "
                f"non-integer entry {token!r}"
            ) from exc
        if feature_id in seen:
            raise ManualFeatureSelectionError(
                f"feature id txt file {path} contains duplicate id {feature_id}"
            )
        seen.add(feature_id)
        out.append(feature_id)
    return out
