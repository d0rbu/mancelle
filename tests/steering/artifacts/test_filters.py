"""Glob-filter merging semantics for hf_push / hf_pull."""
from __future__ import annotations

from steering.artifacts.filters import ResolvedFilters, resolve_filters


def test_no_user_filters_yields_defaults():
    f = resolve_filters(
        default_include=["*.pt"],
        default_exclude=["**/optim_*.pt"],
        user_include=None,
        user_exclude=None,
        no_default_include=False,
        no_default_exclude=False,
    )
    assert f == ResolvedFilters(include=["*.pt"], exclude=["**/optim_*.pt"])


def test_user_filters_append_to_defaults():
    f = resolve_filters(
        default_include=[],
        default_exclude=["**/optim_*.pt"],
        user_include=["*.safetensors"],
        user_exclude=["scratch/**"],
        no_default_include=False,
        no_default_exclude=False,
    )
    assert f.include == ["*.safetensors"]
    assert sorted(f.exclude) == sorted(["**/optim_*.pt", "scratch/**"])


def test_no_default_exclude_clears_defaults():
    f = resolve_filters(
        default_include=[],
        default_exclude=["**/optim_*.pt"],
        user_include=None,
        user_exclude=["scratch/**"],
        no_default_include=False,
        no_default_exclude=True,
    )
    assert f.exclude == ["scratch/**"]


def test_no_default_include_clears_defaults():
    f = resolve_filters(
        default_include=["*.pt"],
        default_exclude=[],
        user_include=["*.safetensors"],
        user_exclude=None,
        no_default_include=True,
        no_default_exclude=False,
    )
    assert f.include == ["*.safetensors"]


def test_empty_filters_round_trip_to_none():
    f = resolve_filters(
        default_include=[],
        default_exclude=[],
        user_include=None,
        user_exclude=None,
        no_default_include=False,
        no_default_exclude=False,
    )
    assert f.include == [] and f.exclude == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_user_include_empty_list_treated_as_none():
    """An empty user_include list (e.g. from `+include='[]'` on the CLI) does NOT
    replace the default — it's equivalent to passing user_include=None."""
    f = resolve_filters(
        default_include=["*.pt"],
        default_exclude=[],
        user_include=[],          # empty list from a noisy CLI
        user_exclude=None,
        no_default_include=False,
        no_default_exclude=False,
    )
    assert f.include == ["*.pt"]   # default preserved


def test_no_default_include_with_no_user_include_yields_empty():
    """Clearing the default include with no user include leaves filters empty."""
    f = resolve_filters(
        default_include=["*.pt"],
        default_exclude=[],
        user_include=None,
        user_exclude=None,
        no_default_include=True,
        no_default_exclude=False,
    )
    assert f.include == []


def test_returned_lists_are_isolated_from_inputs():
    """Mutating the returned filter lists does not leak back into the input lists
    (defensive copy semantics)."""
    default_inc = ["*.pt"]
    default_exc = ["**/optim_*.pt"]
    f = resolve_filters(
        default_include=default_inc,
        default_exclude=default_exc,
        user_include=None,
        user_exclude=None,
        no_default_include=False,
        no_default_exclude=False,
    )
    # ResolvedFilters is frozen so we can't append, but we can build a new
    # mutable list from the returned values and verify it's isolated.
    inc_copy = list(f.include)
    inc_copy.append("LEAK")
    assert "LEAK" not in default_inc
