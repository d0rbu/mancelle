"""Assertion and fixture-loading helpers for token-example parity tests."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from steering.eval.autointerp_filter.parity import (
    FP32_NATIVE_VS_TL_ATOL,
    TensorToleranceReport,
    TokenExampleSurfaceReport,
    TokenGroupSurfaceReport,
    compare_token_example_surface,
    sqlite_ro_uri,
)
from steering.eval.autointerp_filter.types import AutoInterpCacheError, TokenExampleSet

LATENT_NATIVE_VS_TL_ATOL = 0.02
SAE_NATIVE_VS_TL_ATOL = 0.08

# Native-CUDA bf16 structural envelope measured for issue #678 PR B.
BF16_MIN_MATCHED_FRACTION = 0.80
BF16_POSITIVE_MIN_ORDER_FRACTION = 0.05
BF16_NON_ACTIVATING_MIN_ORDER_FRACTION = 0.75
BF16_MIN_HIGHLIGHT_JACCARD = 0.40


def assert_tensor_report_within(report: TensorToleranceReport, *, atol: float) -> None:
    assert report.reference_values == report.candidate_values
    assert report.within_abs_tolerance, f"max_abs={report.max_abs} > atol={atol}"


def assert_group_surface_within_tolerance(
    report: TokenGroupSurfaceReport,
    *,
    atol: float = FP32_NATIVE_VS_TL_ATOL,
    min_order_fraction: float = 1.0,
    min_highlight_jaccard: float = 1.0,
    require_exact_highlights: bool = True,
    require_exact_quantiles: bool = True,
    require_identity_set_equal: bool = True,
    min_matched_fraction: float = 1.0,
    require_activation_tolerance: bool = True,
) -> None:
    """Assert one token-window group against the selected test envelope."""
    assert report.reference_count == report.candidate_count
    if require_identity_set_equal:
        assert report.identity_set_equal
    assert report.matched_window_count >= min_matched_fraction * report.reference_count
    assert report.same_identity_order_fraction >= min_order_fraction
    if require_activation_tolerance:
        assert_tensor_report_within(report.activation, atol=atol)
        assert_tensor_report_within(report.max_activation, atol=atol)
    if require_exact_quantiles:
        assert report.quantile_exact_count == report.matched_window_count
    assert report.min_highlight_jaccard >= min_highlight_jaccard
    if require_exact_highlights:
        assert report.highlight_exact_count == report.matched_window_count


def assert_token_example_surface_within_tolerance(
    report: TokenExampleSurfaceReport,
    *,
    positive_min_order_fraction: float = 1.0,
    non_activating_min_order_fraction: float = 1.0,
) -> None:
    assert_group_surface_within_tolerance(
        report.positives,
        min_order_fraction=positive_min_order_fraction,
    )
    assert_group_surface_within_tolerance(
        report.non_activating,
        min_order_fraction=non_activating_min_order_fraction,
    )


def assert_bf16_surface_within_tolerance(
    reference: TokenExampleSet,
    candidate: TokenExampleSet,
) -> None:
    """Assert a CPU-reference vs native-bf16 surface is within its test envelope."""
    report = compare_token_example_surface(reference, candidate)
    assert report.non_activating.activation.max_abs == 0.0
    assert report.non_activating.max_activation.max_abs == 0.0
    assert_group_surface_within_tolerance(
        report.non_activating,
        min_order_fraction=BF16_NON_ACTIVATING_MIN_ORDER_FRACTION,
        min_highlight_jaccard=BF16_MIN_HIGHLIGHT_JACCARD,
        require_exact_highlights=False,
        require_exact_quantiles=False,
        require_identity_set_equal=False,
        min_matched_fraction=BF16_MIN_MATCHED_FRACTION,
        require_activation_tolerance=False,
    )
    assert_group_surface_within_tolerance(
        report.positives,
        min_order_fraction=BF16_POSITIVE_MIN_ORDER_FRACTION,
        min_highlight_jaccard=BF16_MIN_HIGHLIGHT_JACCARD,
        require_exact_highlights=False,
        require_exact_quantiles=False,
        require_identity_set_equal=False,
        min_matched_fraction=BF16_MIN_MATCHED_FRACTION,
        require_activation_tolerance=False,
    )


def load_single_token_example_set(path: Path) -> TokenExampleSet:
    """Load the first cached set for fixtures that intentionally contain one row."""
    with closing(sqlite3.connect(sqlite_ro_uri(path), uri=True)) as conn:
        row = conn.execute(
            "SELECT payload FROM token_example_sets ORDER BY target_key LIMIT 1"
        ).fetchone()
    if row is None:
        raise AutoInterpCacheError(f"{path} has no token_example_sets row")
    return TokenExampleSet.from_json(json.loads(row[0]))
