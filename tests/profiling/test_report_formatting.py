"""Golden-output test for scripts.profile_progression.report."""

from __future__ import annotations

from pathlib import Path

from scripts.profile_progression import render_macro_markdown

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "profile_progression"


def test_render_macro_markdown_matches_golden():
    csv_path = FIXTURES / "sample_macro_table.csv"
    rendered = render_macro_markdown(csv_path)
    assert rendered.strip() == """# Macro wall-time summary

## Suite Passes

| Path | Suite | Variant | Conditioned Config | Median (s) | 95% CI (s) | N iters | Warmup (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | chat | - | - | 10.9 | [10.8, 11.0] | 3 | 12.5 |
| A | core | - | - | 38.1 | [37.9, 38.5] | 3 | 45.2 |
| A | harmfulness | - | - | 88.0 | [87.5, 89.2] | 3 | 90.4 |

## Stage Rows

| Path | Suite | Variant | Conditioned Config | Stage | Iter | Warmup | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | - | - | - | checkpoint_load | 0 | true | 3.20 |
| A | - | - | - | build_wrapper | 0 | true | 0.05 |"""
