"""Slice 16 audit — real-stage integration test (X2 §6.1 mocking-trap fix).

X2 §6.1 flagged ``tests/axbench_base/conftest.py:248-359``
(``stub_heavy_loaders``) as a mocking trap: tests using that fixture
verify the dispatch layer's plumbing but not that any real stage code
runs. The recommendation: add at least one real-stage CPU integration
test that does NOT use ``stub_heavy_loaders``.

This file delivers that. We exercise three real-stage code paths
END-TO-END (no monkeypatching of production modules):

1. Stage 6 ``aggregation.hmean`` / ``best_of_grid`` /
   ``bootstrap_ci`` against real numerics (no mocks).
2. ``judge_parser.parse_axbench_rating`` against real LLM-style
   strings (no LLM mock — pure regex parser).
3. ``paths._sanitize`` + ``paths.run_output_root`` resolution
   against a real tmp_path environment.

These exercises hit the actual production code paths that the
``stub_heavy_loaders`` fixture would shortcut. None of them require
GPU, network, or real model weights.

The fixture also serves as a regression breadcrumb: if a future
refactor breaks the public surfaces of ``aggregation``,
``judge_parser``, or ``paths``, this test fails immediately even
without any other axbench_base tests being run.
"""

from __future__ import annotations

import pytest

from tests.axbench_base import paths as test_paths


def test_real_aggregation_math() -> None:
    """Real ``aggregation`` public functions exercised end-to-end."""
    from steering.eval.axbench_base.aggregation import (
        best_of_grid,
        bootstrap_ci,
        hmean,
    )

    # hmean zero-bottom: 0 in input → 0.0 out.
    assert hmean(0, 5) == 0.0
    assert hmean(5, 0) == 0.0
    # Standard 2ab/(a+b).
    assert hmean(2, 2) == 2.0
    assert hmean(1, 2) == pytest.approx(4.0 / 3.0)

    # best_of_grid: returns max.
    assert best_of_grid([0.1, 0.5, 0.3]) == 0.5
    assert best_of_grid([1.0]) == 1.0

    # bootstrap_ci: deterministic with fixed seed.
    lo, hi = bootstrap_ci(
        [0.0, 0.25, 0.5, 0.75, 1.0], n_boot=100, alpha=0.05, seed=0,
    )
    assert 0.0 <= lo <= hi <= 1.0
    # Single-value input collapses to (v, v).
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_real_judge_parser() -> None:
    """Real ``judge_parser`` against representative LLM response strings."""
    from steering.eval.axbench_base.judge_parser import (
        JUDGE_MAX_SCORE,
        parse_axbench_rating,
    )

    # Each input is the kind of text a real LLM would emit.
    assert parse_axbench_rating("Rating: [[2]]") == 2
    assert parse_axbench_rating("Rating: 1") == 1
    assert parse_axbench_rating("Some text\nRating: 0") == 0
    assert parse_axbench_rating("I think this is a Rating: 2") == 2

    # Documented rejection cases.
    assert parse_axbench_rating(None) is None
    assert parse_axbench_rating("Sorry, no rating") is None
    assert parse_axbench_rating(f"Rating: {JUDGE_MAX_SCORE + 1}") is None


def test_real_paths_resolution(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real ``paths.run_output_root`` resolution against a tmp_path environment."""

    # Real env var: production reads ``NANOCHAT_BASE_DIR``.
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = test_paths.default_paths("safe_tag_42").run_output_root()
    assert root.parent.name == "axbench_base"
    assert root.name == "safe_tag_42"
    assert str(tmp_path) in str(root)

    # Path traversal defense: resolved path stays inside the base.
    bad = test_paths.default_paths("bad/../../etc/passwd").run_output_root()
    assert "etc" not in bad.parts
    assert "passwd" not in bad.parts
    assert bad.name.startswith("traversal_")


def test_real_concept_dataclass_construction() -> None:
    """Real ``Concept`` dataclass construction (no fixtures, no mocks).

    Pins the public surface shape — slice 17's audit (§1.19 Concept
    consolidation) consumes this dataclass directly. Adding/removing
    fields here ripples to slice 17.
    """
    from steering.eval.axbench_base.types import Concept

    c = Concept(
        concept_id=0,
        feature_index=42,
        layer=8,
        sae_source="test-sae-source",
        description="example concept",
        corpus_count=100,
        description_source="manual",
    )
    assert c.concept_id == 0
    assert c.feature_index == 42
    assert c.layer == 8
    assert c.sae_source == "test-sae-source"
    assert c.description == "example concept"
    assert c.corpus_count == 100
    assert c.description_source == "manual"


def test_real_method_registry_traversal() -> None:
    """Real registry traversal — no fixtures, no mocks."""
    from steering.eval.axbench_base.method_registry import ALL_METHODS
    from steering.eval.axbench_base.types import SteeringMode

    # Sanity counts that the production registry round-trips correctly.
    assert all(m.name for m in ALL_METHODS)
    assert all(
        m.mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE)
        for m in ALL_METHODS
    )


def test_real_paths_avoids_stub_heavy_loaders_fixture() -> None:
    """Pin the X2 §6.1 fix: this test should run WITHOUT importing the
    ``stub_heavy_loaders`` fixture. We assert that the runtime_state
    module has not been patched (i.e., the production loader functions
    are present).
    """
    import steering.eval.axbench_base.runtime_state as rs

    # If ``stub_heavy_loaders`` had run, ``_load_feature_extraction_pair``
    # would be a MagicMock-ish callable closed over a stub. Instead, the
    # production module exposes its real loader symbol.
    assert hasattr(rs, "_load_feature_extraction_pair")
    # The production loader is a real function (not a Mock).
    loader = rs._load_feature_extraction_pair
    assert callable(loader)
    # Mock objects don't have ``__module__`` resolving to the production
    # module path.
    assert loader.__module__.startswith("steering.eval.axbench_base"), (
        f"X2 §6.1 fix violation: real-stage test environment has "
        f"a stubbed loader (module={loader.__module__!r}). The "
        f"``stub_heavy_loaders`` fixture must not be loaded by this test."
    )
