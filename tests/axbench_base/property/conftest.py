"""Hypothesis profile for AxBench-Base property-based tests.

Registers and loads the ``axbench`` profile so every property test in
this directory runs 100 random examples with the default deadline
removed (property tests touch no I/O and run comfortably under 100 ms
per example, but Hypothesis occasionally raises false-positive
deadline warnings on cold caches). The function-scoped fixture health
check is suppressed because some tests in this directory reuse a
module-scoped RNG via a function-scoped fixture for isolation.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "axbench",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("axbench")
