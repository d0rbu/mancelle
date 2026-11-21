"""Tests confirming that project `load_dotenv()` calls do NOT walk ancestor
directories.

Threat model:
    `dotenv.load_dotenv()` with no path argument calls `find_dotenv()` which
    walks up from the current working directory looking for a `.env` file.
    An attacker who can plant `.env` in any ancestor of the directory a user
    runs the script from (e.g. `/tmp/.env`) can inject arbitrary environment
    variables — including API keys — into the Python process.

These tests plant a real attacker `.env` in an ancestor and verify that the
project's centralized dotenv loader does not load it. The tests deliberately
do NOT mock `dotenv.load_dotenv`; they verify real behavior end-to-end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanochat.common import load_project_dotenv


# The marker env var used across all tests. `monkeypatch.delenv` is applied
# per-test via an autouse fixture below so tests remain hermetic.
_MARKER: str = "NANO_GPT_DOTENV_TEST_KEY"


@pytest.fixture(autouse=True)
def _clear_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the marker env var is clear at the start of every test."""
    monkeypatch.delenv(_MARKER, raising=False)


def _build_attack_layout(
    tmp_path: Path,
    *,
    plant_project_env: bool,
    project_value: str = "correct",
    attacker_value: str = "attack",
) -> tuple[Path, Path]:
    """Build a fake filesystem that simulates the attack scenario.

    Layout::

        tmp_path/
            project/                  <- fake project root
                pyproject.toml
                (.env)                <- only if plant_project_env is True
            ancestor_attack/
                .env                  <- attacker-controlled
                deeper_dir/           <- user's cwd when running the script

    Returns (project_root, cwd).
    """
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("# fake\n")
    if plant_project_env:
        (project_root / ".env").write_text(f"{_MARKER}={project_value}\n")

    ancestor_attack = tmp_path / "ancestor_attack"
    ancestor_attack.mkdir()
    (ancestor_attack / ".env").write_text(f"{_MARKER}={attacker_value}\n")

    cwd = ancestor_attack / "deeper_dir"
    cwd.mkdir()

    return project_root, cwd


def test_load_dotenv_does_not_walk_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attacker-planted `.env` in an ancestor of cwd MUST NOT be loaded.

    This is the core security invariant: even if the user runs the script
    from a directory whose parent contains a malicious `.env`, the loader
    must only honor the project-root `.env`.
    """
    project_root, cwd = _build_attack_layout(tmp_path, plant_project_env=True)
    monkeypatch.chdir(cwd)

    load_project_dotenv(project_root=project_root)

    # The attacker value must NOT be in the environment.
    assert os.environ.get(_MARKER) != "attack", (
        "Attacker-planted .env in ancestor directory was loaded — "
        "load_project_dotenv() walked ancestors instead of pinning to "
        "the project root."
    )


def test_load_dotenv_finds_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project-root `.env` MUST be loaded when present."""
    project_root, cwd = _build_attack_layout(tmp_path, plant_project_env=True)
    monkeypatch.chdir(cwd)

    load_project_dotenv(project_root=project_root)

    assert os.environ.get(_MARKER) == "correct", (
        "Project-root .env was not loaded."
    )


def test_load_dotenv_ignores_missing_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no `.env` exists at the project root, loader MUST NOT fall back
    to walking ancestors even though an attacker `.env` exists there."""
    project_root, cwd = _build_attack_layout(tmp_path, plant_project_env=False)
    monkeypatch.chdir(cwd)

    load_project_dotenv(project_root=project_root)

    assert os.environ.get(_MARKER) is None, (
        "Loader fell back to ancestor .env when project-root .env was missing."
    )


def test_load_dotenv_default_project_root_uses_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit ``project_root`` argument, the loader MUST use the
    project root of this repository (computed via ``get_project_root()``),
    not the current working directory."""
    # Plant an attacker `.env` in tmp_path and chdir there. If the loader
    # walked ancestors or used cwd to find the file, it would load the
    # attacker value. It must not.
    attack_dir = tmp_path / "rogue_cwd"
    attack_dir.mkdir()
    (tmp_path / ".env").write_text(f"{_MARKER}=attack\n")
    monkeypatch.chdir(attack_dir)

    load_project_dotenv()

    assert os.environ.get(_MARKER) != "attack", (
        "Loader used cwd/ancestor .env when no project_root arg was given; "
        "it should have anchored to the repository root."
    )
