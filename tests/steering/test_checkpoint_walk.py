"""Tests for ``steering._common.checkpoint_walk.find_last_step_or_none``.

This is the soft variant of ``nanochat.checkpoint_manager.find_last_step``
that fixes the empty-directory crash in ``_recover_model_config`` /
``_recover_parent_user_config`` (slice 11 audit P1 #4 / consolidation
manifest section 1.10).

Test coverage:

* missing directory -> ``None``
* path is a file (not a dir) -> ``None``
* empty directory -> ``None`` (not ``FileNotFoundError``)
* directory with non-checkpoint files only -> ``None``
* directory with one ``model_*.pt`` -> the parsed step
* directory with multiple ``model_*.pt`` -> the largest parsed step
* mixed-content tree -> ignores subdirs, returns the largest leaf step
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering._common.checkpoint_walk import find_last_step_or_none


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_missing_directory_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"
    assert not missing.exists()
    assert find_last_step_or_none(missing) is None


def test_path_is_file_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "afile.pt"
    _touch(f)
    assert find_last_step_or_none(f) is None


def test_empty_directory_returns_none(tmp_path: Path) -> None:
    """Existing-but-empty directory is the bug shape — must NOT raise."""
    d = tmp_path / "empty"
    d.mkdir()
    assert find_last_step_or_none(d) is None


def test_directory_with_non_checkpoint_files_returns_none(tmp_path: Path) -> None:
    d = tmp_path / "junk"
    _touch(d / "meta_000100.json")
    _touch(d / "optim_000100_rank0.pt")
    _touch(d / "README.md")
    assert find_last_step_or_none(d) is None


def test_directory_with_single_checkpoint_returns_step(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    _touch(d / "model_000100.pt")
    assert find_last_step_or_none(d) == 100


def test_directory_with_multiple_checkpoints_returns_largest_step(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    _touch(d / "model_000100.pt")
    _touch(d / "model_000500.pt")
    _touch(d / "model_002500.pt")
    assert find_last_step_or_none(d) == 2500


def test_returns_none_when_only_subdirs_no_checkpoints(tmp_path: Path) -> None:
    """A directory with subdirectories but no top-level model_*.pt
    must not return a step (find_last_step is shallow)."""
    d = tmp_path / "ckpt"
    nested = d / "subdir"
    _touch(nested / "model_001000.pt")
    # Top-level dir has no model_*.pt — the helper should ignore subdirs.
    assert find_last_step_or_none(d) is None


def test_pathlib_path_argument_is_accepted(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    _touch(d / "model_000050.pt")
    assert find_last_step_or_none(Path(d)) == 50


def test_string_argument_is_accepted(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    _touch(d / "model_000050.pt")
    assert find_last_step_or_none(str(d)) == 50


def test_does_not_swallow_unrelated_errors(tmp_path: Path, monkeypatch) -> None:
    """Only ``FileNotFoundError`` should be caught; other errors propagate.

    This protects against silent corruption — e.g. a permissions error
    must surface, not return ``None`` (which the caller would treat as
    "no checkpoint here, skip recovery").

    After the circular-import fix (PR-A), ``checkpoint_walk`` is self-contained
    and uses ``glob.glob`` directly rather than delegating to
    ``nanochat.checkpoint_manager.find_last_step``.  We therefore simulate the
    OS-level error by monkeypatching ``glob.glob`` inside the module under test.
    """
    from steering._common import checkpoint_walk

    def boom(_pattern: str) -> list:
        raise OSError("simulated permissions error")

    monkeypatch.setattr(checkpoint_walk.glob, "glob", boom)
    d = tmp_path / "ckpt"
    d.mkdir()
    with pytest.raises(OSError, match="simulated permissions error"):
        find_last_step_or_none(d)


def test_recover_pattern_matches_existing_callsite_shape(tmp_path: Path) -> None:
    """The helper is designed to drop in for the
    ``parent_step = find_last_step(parent_dir) if os.path.isdir(parent_dir) else None``
    line in ``_recover_model_config`` / ``_recover_parent_user_config``.
    Pin the contract: an empty parent dir yields ``None``, allowing the
    caller to fall through to Strategy 2.
    """
    parent = tmp_path / "parent_empty"
    parent.mkdir()
    parent_step = find_last_step_or_none(parent)
    assert parent_step is None
    # Strategy 2 fallback would be activated by ``if parent_step is not None:``
    # being False — the test simulates that decision point.
    assert (parent_step is not None) is False
