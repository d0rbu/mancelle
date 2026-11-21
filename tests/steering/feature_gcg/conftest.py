"""Collection hook: auto-mark tests requesting slow-gated fixtures as ``slow``.

The ``real_model`` fixture (``tests/steering/feature_gcg/_equivalence.py``) loads a real
d12 encdec checkpoint and must only run in the slow suite (it skips-not-fails
when the checkpoint is absent).  The Task-2 brief mandated that the *fixture* be
the single slow-gating choke point so individual call sites do not each need a
``@pytest.mark.slow``.

pytest 9 forbids applying a mark directly to a fixture — ``@pytest.mark.slow``
on a fixture raises ``PytestRemovedIn9Warning: Marks applied to fixtures have no
effect`` as a hard collection error (verified empirically on pytest 9.0.2).  The
canonical, pytest-9-compatible way to make a fixture a slow-gating choke point
is therefore this ``pytest_collection_modifyitems`` hook: any test whose fixture
closure includes a name in :data:`SLOW_GATED_FIXTURES` is marked ``slow`` here.
"""

from __future__ import annotations

import pytest

# Fixture names whose use auto-gates a test into the ``slow`` suite.
SLOW_GATED_FIXTURES: frozenset[str] = frozenset({"real_model"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark every collected test that requests a slow-gated fixture as ``slow``.

    ``item.fixturenames`` is the test's full fixture closure (direct +
    transitive), so this fires whether ``real_model`` is requested directly or
    pulled in by another fixture.
    """
    for item in items:
        fixturenames = frozenset(getattr(item, "fixturenames", ()))
        if fixturenames & SLOW_GATED_FIXTURES:
            item.add_marker(pytest.mark.slow)
