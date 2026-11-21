"""Regression tests for script-level users of the canonical text-ID parser."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.analysis.gradient_vs_magnitude_examples import (
    _resolve_frequency_blacklist,
)
from scripts.benchmark.benchmark_autointerp_source_index_build import (
    _read_ids,
)
from scripts.benchmark.benchmark_autointerp_source_index_memory import (
    _read_id_file,
)


@pytest.mark.parametrize("reader", [_read_ids, _read_id_file])
def test_benchmark_id_file_adapters_share_comment_semantics(
    tmp_path: Path,
    reader: Callable[[Path], set[int]],
) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("3  # trailing comment\n\n# whole-line comment\n1\n")

    assert reader(path) == {1, 3}


@pytest.mark.parametrize("reader", [_read_ids, _read_id_file])
def test_benchmark_id_file_adapters_fail_loudly_on_duplicate_ids(
    tmp_path: Path,
    reader: Callable[[Path], set[int]],
) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("1\n1\n")

    with pytest.raises(SystemExit, match="duplicate id 1"):
        reader(path)


def test_gradient_blacklist_adapter_shares_comment_semantics(tmp_path: Path) -> None:
    blacklist_dir = tmp_path / "blacklist"
    blacklist_dir.mkdir()
    path = blacklist_dir / "freq.txt"
    path.write_text("3  # trailing comment\n\n# whole-line comment\n1\n")

    ids, source = _resolve_frequency_blacklist(tmp_path)

    assert ids == (1, 3)
    assert source == "blacklist/freq.txt"


def test_gradient_blacklist_adapter_fails_loudly_on_duplicate_ids(
    tmp_path: Path,
) -> None:
    blacklist_dir = tmp_path / "blacklist"
    blacklist_dir.mkdir()
    (blacklist_dir / "freq.txt").write_text("1\n1\n")

    with pytest.raises(SystemExit, match="duplicate id 1"):
        _resolve_frequency_blacklist(tmp_path)
