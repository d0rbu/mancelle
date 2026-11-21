"""Guard test: coverage.py `concurrency` must be a list, not a bare string.

A bare string (e.g. `concurrency = "multiprocessing"`) makes coverage.py raise
`ConfigError: Option [tool.coverage.run]concurrency is not a list`, which breaks
`pytest --cov` for the entire repo. This test pins the invariant so it cannot
silently regress.
"""

import tomllib
from pathlib import Path

from coverage import Coverage

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_pyproject_concurrency_is_a_list() -> None:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)

    concurrency = data["tool"]["coverage"]["run"]["concurrency"]
    assert isinstance(concurrency, list), (
        f"[tool.coverage.run].concurrency must be a list, got "
        f"{type(concurrency).__name__!r}: {concurrency!r}"
    )
    assert all(isinstance(item, str) for item in concurrency)
    assert "multiprocessing" in concurrency


def test_coverage_can_load_pyproject_config() -> None:
    """coverage.py must be able to construct from pyproject.toml without error."""
    cov = Coverage(config_file=str(PYPROJECT))
    # Accessing the parsed concurrency list proves the config parsed cleanly.
    assert isinstance(cov.config.concurrency, list)
    assert "multiprocessing" in cov.config.concurrency
