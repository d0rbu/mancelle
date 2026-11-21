"""Prompting baseline descriptions writer (Task 26, axbench_base Stage 3c).

Writes ``{mode}_descriptions.json`` files mapping ``concept_id`` (as a
string key) to the concept's human-readable ``description``.  This is the
entire "training" step for the prompting baseline — no model is loaded, no
gradients are computed.  The resulting JSON files are consumed downstream by
the Stage 4 generation dispatch to build system-prompt or user-turn strings
that condition the model on concept presence or absence.

Output paths are resolved via
``AxBenchBasePaths.prompting_descriptions(mode)`` which returns:

    ``{output_root}/baselines/prompting/{mode}_descriptions.json``

JSON format
-----------
Keys are ``str(concept_id)`` (string representation of the integer concept
ID), values are the ``description`` field of the corresponding
:class:`~steering.eval.axbench_base.types.Concept`.  Written with
``indent=2, ensure_ascii=False`` for readability and unicode fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)

_VALID_MODES: tuple[SteeringMode, ...] = (SteeringMode.POSITIVE, SteeringMode.NEGATIVE)


def _write_single_mode(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    mode: SteeringMode,
) -> Path:
    """Serialise ``concept_id → description`` JSON for one mode.

    Args:
        output_paths: Explicit output-path contract for this run.
        concepts: All concepts for this run.
        mode: ``"positive"`` or ``"negative"``.

    Returns:
        The :class:`~pathlib.Path` of the written file.
    """
    out_path = output_paths.prompting_descriptions(mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {str(concept.concept_id): concept.description for concept in concepts}
    out_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def write_prompting_descriptions(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
) -> list[Path]:
    """Write both positive and (optionally) negative descriptions files.

    The "training" for the prompting baseline is trivial: write
    ``concept_id → description`` mappings to JSON files so that the
    Stage 4 generation step can build conditioned prompts without any
    model loading.

    Args:
        cfg: Pipeline configuration.  ``cfg.run_tag`` must be set (not
            ``None``).  ``cfg.run_negative_mode`` controls whether the
            negative descriptions file is written.
        concepts: Concepts for this run.  May be empty, in which case an
            empty JSON object is written.

    Returns:
        List of :class:`~pathlib.Path` objects for the files written, in
        the order ``[positive_path]`` or ``[positive_path, negative_path]``.

    Raises:
        AxBenchBaseError: if ``cfg.run_tag is None``.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling write_prompting_descriptions"
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    out_paths: list[Path] = []

    # Positive mode always runs (run_positive_mode was removed in Batch H).
    out_paths.append(_write_single_mode(output_paths, concepts, SteeringMode.POSITIVE))

    if cfg.run_negative_mode:
        out_paths.append(_write_single_mode(output_paths, concepts, SteeringMode.NEGATIVE))

    return out_paths
