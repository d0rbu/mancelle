"""Fail-closed segment-mode guards for fixed-geometry eval consumers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beartype import beartype

from steering.feature_paths import TURN_DELIMITED_LEAF, parse_feature_dir
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import resolve_checkpoint_sft_segment_mode


def require_fixed_segment_mode(
    mode: SegmentMode,
    *,
    surface: str,
    source: str,
) -> None:
    """Reject segment modes unsupported by a fixed-only consumer."""

    if mode is SegmentMode.fixed_tokens:
        return
    raise ValueError(
        f"{surface} does not support {source} with segment mode "
        f"{mode.value!r}; only {SegmentMode.fixed_tokens.value!r} is supported. "
        "Use a fixed-token checkpoint/artifact or a consumer with explicit "
        "turn-delimited inference support."
    )


def require_fixed_checkpoint_segment_mode(
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    surface: str,
    checkpoint_dir: str | Path | None = None,
) -> None:
    """Resolve checkpoint lineage and reject turn-delimited SFT checkpoints."""

    resolved_dir = checkpoint_dir
    if resolved_dir is None and checkpoint_meta is not None:
        resolved_dir = checkpoint_meta.get("checkpoint_dir")
    checkpoint_label = "<unknown checkpoint>" if resolved_dir is None else str(resolved_dir)
    mode = resolve_checkpoint_sft_segment_mode(
        checkpoint_meta,
        checkpoint_dir=checkpoint_label,
    )
    require_fixed_segment_mode(
        mode,
        surface=surface,
        source=f"checkpoint {checkpoint_label}",
    )


def _path_segment_mode(candidate: Path) -> SegmentMode | None:
    """Segment mode witnessed by a feature path, or ``None`` when unparseable.

    Historical explicit feature directories were not required to use the
    canonical layout, so an unparseable path is not itself an error. The
    canonical turn leaf is honoured before parsing so a malformed or
    copy-relocated turn path cannot fall through to the legacy fixed default
    just because an earlier canonical component fails to parse.
    """

    if candidate.name == TURN_DELIMITED_LEAF:
        return SegmentMode.turn_delimited_fixed_tokens
    try:
        return parse_feature_dir(candidate).segment_mode
    except ValueError:
        return None


@beartype
def require_fixed_feature_dir_segment_mode(
    feature_dir: str | Path,
    *,
    surface: str,
    statistics_segment_mode: SegmentMode | None = None,
) -> None:
    """Resolve feature path/statistics witnesses and reject turn geometry.

    Raises:
        ValueError: If witnesses disagree or the resolved mode is not
            ``fixed_tokens``.
    """

    path = Path(feature_dir)
    candidates = [("feature path", path)]
    resolved_path = path.resolve(strict=False)
    if resolved_path != path.absolute():
        candidates.append(("resolved feature path", resolved_path))
    witnesses = [
        (label, mode)
        for label, candidate in candidates
        if (mode := _path_segment_mode(candidate)) is not None
    ]
    if statistics_segment_mode is not None:
        witnesses.append(("feature statistics", statistics_segment_mode))

    distinct = {mode for _, mode in witnesses}
    if len(distinct) > 1:
        rendered = ", ".join(f"{name}={mode.value!r}" for name, mode in witnesses)
        raise ValueError(
            f"{surface} feature artifact segment-mode mismatch for {path}: {rendered}"
        )
    mode = witnesses[0][1] if witnesses else SegmentMode.fixed_tokens
    require_fixed_segment_mode(
        mode,
        surface=surface,
        source=f"feature directory {path}",
    )


__all__ = [
    "require_fixed_checkpoint_segment_mode",
    "require_fixed_feature_dir_segment_mode",
    "require_fixed_segment_mode",
]
