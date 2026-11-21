"""Exact model-ready prompt construction shared by AxBench Stages 3e and 4."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.tokenizer_utils import prepend_owned_bos
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
)
from steering.eval.target_conditioning import token_context_key


def prompting_prefix_text(mode: SteeringMode, description: str) -> str:
    """Render the raw prompting-baseline prefix for one steering mode."""

    if mode is SteeringMode.POSITIVE:
        return f"Related to {description}:\n\n"
    if mode is SteeringMode.NEGATIVE:
        return f"Not related to {description}:\n\n"
    raise AxBenchBaseError(f"Unsupported prompting mode: {mode!r}")


def load_prompting_descriptions(
    cfg: AxBenchBaseConfig,
    mode: SteeringMode,
) -> dict[int, str]:
    """Load and type-check the Stage 3c description mapping for ``mode``."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before loading prompts")
    path = paths.AxBenchBasePaths.from_config(cfg).prompting_descriptions(mode)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            f"Prompting descriptions are missing or invalid at {path}; rerun Stage 3c"
        ) from exc
    if not isinstance(raw, dict):
        raise AxBenchBaseError(f"Prompting descriptions must be a JSON object: {path}")
    result: dict[int, str] = {}
    for raw_key, raw_description in raw.items():
        try:
            concept_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError(
                f"Invalid concept id {raw_key!r} in {path}"
            ) from exc
        if concept_id < 0 or not isinstance(raw_description, str) or not raw_description:
            raise AxBenchBaseError(
                f"Invalid prompting description for concept {raw_key!r} in {path}"
            )
        if concept_id in result:
            raise AxBenchBaseError(f"Duplicate concept id {concept_id} in {path}")
        result[concept_id] = raw_description
    return result


def build_stage4_prompt_tokens(
    *,
    tokenizer: Any,
    seed_content_tokens: Sequence[int],
    prompting_prefix: str | None = None,
) -> list[int]:
    """Build one exact model-ready Stage 4 prompt.

    Ordinary methods receive ``[owned BOS] + seed content``. Prompting
    methods receive ``[owned BOS] + raw prompt-prefix tokens + seed content``.
    Payload token IDs are never inspected for BOS-like values.
    """

    seed_tokens = [int(token) for token in seed_content_tokens]
    if not seed_tokens:
        raise AxBenchBaseError("Stage 4 seed content tokens must be non-empty")
    prompt_content: list[int] = []
    if prompting_prefix is not None:
        prompt_content = [int(token) for token in tokenizer.encode(prompting_prefix)]
        if not prompt_content:
            raise AxBenchBaseError(
                "Stage 4 prompting prefix encoded to an empty token sequence"
            )
    return prepend_owned_bos(tokenizer, [*prompt_content, *seed_tokens])


def prompt_token_key(tokens: Sequence[int]) -> str:
    """Return the shared exact-token cache/resume key."""

    return token_context_key(tokens)


def prompting_description_hashes(
    cfg: AxBenchBaseConfig,
    modes: Sequence[SteeringMode],
) -> dict[str, str]:
    """Hash the exact Stage 3c files required by enabled prompting methods."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before hashing prompts")
    hashes: dict[str, str] = {}
    for mode in modes:
        path = paths.AxBenchBasePaths.from_config(cfg).prompting_descriptions(mode)
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise AxBenchBaseError(f"Cannot hash prompting descriptions: {path}") from exc
        hashes[mode.value] = digest.hexdigest()
    return hashes


__all__ = [
    "build_stage4_prompt_tokens",
    "load_prompting_descriptions",
    "prompt_token_key",
    "prompting_description_hashes",
    "prompting_prefix_text",
]
