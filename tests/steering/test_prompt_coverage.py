from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.prompt_coverage import assert_exact_prompt_coverage
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
)


def _record(
    index: int,
    *,
    token_ids: tuple[int, ...] | None = None,
    context_text: str | None = None,
) -> EvalContextRecord:
    return EvalContextRecord(
        suite="harmfulness",
        task="harmfulness",
        seed=17,
        example_index=index,
        token_ids=token_ids if token_ids is not None else (0, index + 10),
        context_text=context_text if context_text is not None else f"prompt {index}",
    )


def _segments() -> tuple[ContextFeatureSegment, ...]:
    return (ContextFeatureSegment((1,), (1,), 0, 1),)


def _provider(
    cache_dir: Path,
    entries: list[ContextCacheEntry],
) -> SafetensorsContextFeatureProvider:
    save_context_cache(
        cache_dir,
        entries,
        identity=TEST_CACHE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=TEST_CACHE_IDENTITY,
    )


def _entry(record: EvalContextRecord) -> ContextCacheEntry:
    return ContextCacheEntry.from_sequences(record.token_ids, _segments())


def test_full_prompt_coverage_returns_summary(tmp_path: Path) -> None:
    records = [_record(0), _record(1)]
    provider = _provider(tmp_path, [_entry(record) for record in records])

    summary = assert_exact_prompt_coverage(
        provider,
        records,
        surface="harmfulness_eval",
    )

    assert summary.surface == "harmfulness_eval"
    assert summary.expected_prompt_count == 2
    assert summary.expected_unique_keys == 2
    assert summary.cached_row_count == 2
    assert summary.extra_cached_rows == 0


def test_one_missing_prompt_reports_missing_fraction(tmp_path: Path) -> None:
    records = [_record(0), _record(1)]
    provider = _provider(tmp_path, [_entry(records[0])])

    with pytest.raises(ContextCacheValidationError, match="1 of 2"):
        assert_exact_prompt_coverage(
            provider,
            records,
            surface="harmfulness_eval",
        )


def test_many_missing_prompts_report_eight_examples_and_full_count(
    tmp_path: Path,
) -> None:
    records = [_record(index) for index in range(12)]
    provider = _provider(tmp_path, [])

    with pytest.raises(ContextCacheValidationError, match="12 of 12") as exc_info:
        assert_exact_prompt_coverage(
            provider,
            records,
            surface="harmfulness_eval",
        )

    message = str(exc_info.value)
    assert message.count("example_index=") == 8
    assert "… (+4 more)" in message


def test_zero_segment_cache_row_counts_as_missing(tmp_path: Path) -> None:
    record = _record(0)
    provider = _provider(
        tmp_path,
        [ContextCacheEntry.from_sequences(record.token_ids, [])],
    )

    with pytest.raises(ContextCacheValidationError, match="zero segments"):
        assert_exact_prompt_coverage(
            provider,
            [record],
            surface="harmfulness_eval",
        )


def test_duplicate_token_sequences_need_only_one_cache_row(tmp_path: Path) -> None:
    records = [
        _record(0, token_ids=(0, 20), context_text="first prompt"),
        _record(1, token_ids=(0, 20), context_text="second prompt"),
    ]
    provider = _provider(tmp_path, [_entry(records[0])])

    summary = assert_exact_prompt_coverage(
        provider,
        records,
        surface="harmfulness_eval",
    )

    assert summary.expected_prompt_count == 2
    assert summary.expected_unique_keys == 1
    assert summary.cached_row_count == 1
    assert summary.extra_cached_rows == 0


def test_extra_cache_rows_are_informational(tmp_path: Path) -> None:
    expected = _record(0)
    cached = [expected, _record(1), _record(2)]
    provider = _provider(tmp_path, [_entry(record) for record in cached])

    summary = assert_exact_prompt_coverage(
        provider,
        [expected],
        surface="harmfulness_eval",
    )

    assert summary.expected_unique_keys == 1
    assert summary.cached_row_count == 3
    assert summary.extra_cached_rows == 2


def test_empty_expected_records_name_surface(tmp_path: Path) -> None:
    provider = _provider(tmp_path, [])

    with pytest.raises(ContextCacheValidationError, match="training_progression"):
        assert_exact_prompt_coverage(
            provider,
            [],
            surface="training_progression",
        )


def test_missing_prompt_error_names_manifest_path(tmp_path: Path) -> None:
    expected = _record(0)
    provider = _provider(tmp_path, [_entry(_record(1))])

    with pytest.raises(ContextCacheValidationError, match="1 of 1") as exc_info:
        assert_exact_prompt_coverage(
            provider,
            [expected],
            surface="harmfulness_eval",
        )

    assert str(provider.manifest_path) in str(exc_info.value)
