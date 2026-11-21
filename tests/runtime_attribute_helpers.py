"""Helpers for tests that deliberately cross a static attribute boundary."""

from __future__ import annotations


def set_runtime_attribute(target: object, name: str, value: object) -> None:
    """Set an attribute through Python's runtime protocol.

    Use this only when the mutation itself is part of the test contract: building
    a deliberately structural fixture, installing a process-local module stub, or
    asserting that a frozen/validated object rejects an invalid assignment.
    """

    setattr(target, name, value)


def call_runtime_attribute(
    target: object,
    name: str,
    *args: object,
) -> object:
    """Call an attribute whose deliberate absence is part of a negative test."""

    method = getattr(target, name)
    if not callable(method):
        raise TypeError(f"runtime attribute {name!r} is not callable")
    return method(*args)
