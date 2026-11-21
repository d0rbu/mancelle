"""Resolve user-supplied filters against per-category defaults."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResolvedFilters:
    """Final include/exclude lists after merging defaults + user overrides."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


def resolve_filters(
    *,
    default_include: list[str],
    default_exclude: list[str],
    user_include: list[str] | None,
    user_exclude: list[str] | None,
    no_default_include: bool,
    no_default_exclude: bool,
) -> ResolvedFilters:
    """Merge category defaults with user-supplied include/exclude.

    Semantics:
        - User include REPLACES the default include if any user includes are
          supplied (since adding to an "all" include list would be no-op);
          ``no_default_include`` is therefore mainly cosmetic.
        - User exclude APPENDS to the default exclude. Pass
          ``no_default_exclude=True`` to drop the defaults — useful when a
          contributor explicitly wants the optimizer state.
    """
    inc_base = [] if no_default_include else list(default_include)
    if user_include:
        include = list(user_include)
    else:
        include = inc_base

    exc_base = [] if no_default_exclude else list(default_exclude)
    exclude = exc_base + list(user_exclude or [])

    return ResolvedFilters(include=include, exclude=exclude)
