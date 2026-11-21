"""Independent adversarial review tests for the issue-1673 slice merger."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval.axbench_adv import merge_slices as merger
from tests.axbench_adv.unit.test_merge_slices import _write_run


def test_merge_plot_failure_leaves_no_completion_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    _write_run(roots[0], methods=("DiffMean",), concept_ids=(0,))
    _write_run(roots[1], methods=("DiffMean",), concept_ids=(1,))

    def fail_plot(*_args, **_kwargs):
        raise RuntimeError("injected plot failure")

    monkeypatch.setattr(merger, "plot_concept_vs_fluency", fail_plot)
    output = tmp_path / "merged"
    with pytest.raises(RuntimeError, match="injected plot failure"):
        merger.merge_slices(roots, output)
    assert not (output / "manifest.json").exists()
