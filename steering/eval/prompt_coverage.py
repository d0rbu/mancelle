"""Pre-generation coverage checks for exact eval prompt contexts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from steering.eval.context_cache_io import (
    ContextCacheValidationError as _ContextCacheValidationError,
)
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.context_enumeration import EvalContextRecord


@dataclass(frozen=True)
class PromptCoverageSummary:
    """What one exact prompt-coverage check found."""

    surface: str
    expected_prompt_count: int
    expected_unique_keys: int
    cached_row_count: int
    extra_cached_rows: int


def assert_exact_prompt_coverage(
    provider: SafetensorsContextFeatureProvider,
    records: Sequence[EvalContextRecord],
    *,
    surface: str,
) -> PromptCoverageSummary:
    """Require the cache to cover every unique prompt token sequence requested."""

    if not records:
        raise _ContextCacheValidationError(
            f"{surface}: no expected prompt context records were provided"
        )

    records_by_key: dict[str, EvalContextRecord] = {}
    for record in records:
        records_by_key.setdefault(record.key, record)

    missing: list[EvalContextRecord] = []
    zero_segment: list[EvalContextRecord] = []
    for record in records_by_key.values():
        segments = provider.get(record.token_ids)
        if segments is None:
            missing.append(record)
        elif len(segments) == 0:
            zero_segment.append(record)

    unavailable = [*missing, *zero_segment]
    if unavailable:
        examples = "; ".join(
            _format_record_example(record) for record in unavailable[:8]
        )
        remaining = len(unavailable) - 8
        if remaining > 0:
            examples = f"{examples}; … (+{remaining} more)"
        raise _ContextCacheValidationError(
            f"{surface}: {len(unavailable)} of {len(records_by_key)} required "
            "unique prompt context row(s) are missing or unusable "
            f"({len(missing)} absent, {len(zero_segment)} with zero segments). "
            f"Examples: {examples}. Cache manifest: {provider.manifest_path}. "
            "Run scripts/eval/build_eval_context_cache.py for the evaluation "
            "contexts that produce these rows, then retry."
        )

    summary = PromptCoverageSummary(
        surface=surface,
        expected_prompt_count=len(records),
        expected_unique_keys=len(records_by_key),
        cached_row_count=len(provider),
        extra_cached_rows=len(provider) - len(records_by_key),
    )
    logger.info(
        f"{surface}: exact prompt coverage passed: "
        f"expected_prompts={summary.expected_prompt_count}, "
        f"expected_unique_keys={summary.expected_unique_keys}, "
        f"cached_rows={summary.cached_row_count}, "
        f"extra_cached_rows={summary.extra_cached_rows}"
    )
    return summary


def _format_record_example(record: EvalContextRecord) -> str:
    preview = record.context_text[:60]
    if len(record.context_text) > 60:
        preview = f"{preview}…"
    return f"example_index={record.example_index!r} context_text={preview!r}"
