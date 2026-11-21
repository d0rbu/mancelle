"""Test-only conveniences layered on the production AxBench path contract."""

from steering.eval.axbench_base.paths import AxBenchBasePaths, default_output_root


def default_paths(run_tag: str) -> AxBenchBasePaths:
    """Construct the historical default-root contract used by tests."""
    return AxBenchBasePaths(default_output_root(), run_tag)
