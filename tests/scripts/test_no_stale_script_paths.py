"""Regression test for stale ``scripts/X.py`` and ``-m scripts.X.Y`` references.

PR #394 reorganized 43 flat scripts under
``scripts/{train,eval,preprocess,utils,validation}/``. This test keeps
documentation, sbatch templates, ``runs/*.sh`` scripts, and config docstrings
aligned with the current layout.
This test scans every operator-facing surface and asserts that every
``python -m scripts.X.Y`` invocation resolves to a real importable module, and
every ``scripts/X.py`` path reference resolves to a file on disk.

This test ALSO checks ``runs/`` for a few patterns that indicate Hydra-syntax
drift (``--key=value`` argparse-style flags after ``-- ``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Sets of directories to walk. Each tuple: (root, glob).
SCAN_TARGETS: list[tuple[Path, str]] = [
    (REPO_ROOT / "runs", "*.sh"),
    (REPO_ROOT / "scripts" / "slurm", "*.sbatch"),
    (REPO_ROOT / "docs", "**/*.md"),
    (REPO_ROOT / "configs", "**/*.yaml"),
    (REPO_ROOT, "README.md"),
]

# `python -m scripts.X.Y` and `-m scripts.X.Y` references.
MODULE_RE = re.compile(r"(?:python\s+)?-m\s+(scripts(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)")

# `scripts/X/Y.py` path references (NOT inside .venv, .git, etc.). Match on
# the captured group; we do NOT match `scripts/X.py` *as documented anchors*
# inside config-key blocks (those are rare and most have been fixed).
SCRIPT_PATH_RE = re.compile(r"(?<!tests/)\bscripts/(?:[a-zA-Z_][a-zA-Z0-9_]*/)*([a-zA-Z_][a-zA-Z0-9_]*)\.py\b")


def _collect_scan_files() -> list[Path]:
    files: list[Path] = []
    for root, glob in SCAN_TARGETS:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
        else:
            for fp in root.glob(glob):
                if not fp.is_file():
                    continue
                files.append(fp)
    return sorted(files)


SCAN_FILES = _collect_scan_files()


@pytest.mark.parametrize("path", SCAN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_module_references_resolve(path: Path) -> None:
    """Every ``-m scripts.X.Y`` reference must resolve to a real file."""
    text = path.read_text()
    for m in MODULE_RE.finditer(text):
        module_path = m.group(1)
        # Resolve "scripts.train.base_train" → "scripts/train/base_train.py"
        rel = module_path.replace(".", "/") + ".py"
        candidate = REPO_ROOT / rel
        # Some modules are packages (e.g. scripts.completion_tui). Allow a
        # `<name>/__init__.py` or `<name>/__main__.py` form as well.
        package_init = REPO_ROOT / module_path.replace(".", "/") / "__init__.py"
        package_main = REPO_ROOT / module_path.replace(".", "/") / "__main__.py"
        assert candidate.is_file() or package_init.is_file() or package_main.is_file(), (
            f"{path.relative_to(REPO_ROOT)} references ``-m {module_path}`` which "
            f"does not resolve to {candidate}, {package_init}, or {package_main}. "
            f"PR #394 may have moved or renamed the script."
        )


@pytest.mark.parametrize("path", SCAN_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_script_path_references_resolve(path: Path) -> None:
    """Every ``scripts/X.py`` path reference must resolve to a real file.

    NOTE: This is intentionally less strict than module references. Some
    docs/configs use ``scripts/X.py`` shorthand even when ``-m scripts.X``
    is canonical. The test only fires on the longest match found, so a
    correct ``scripts/train/base_train.py`` won't trip a false-positive on
    ``scripts/base_train.py``.
    """
    text = path.read_text()
    for m in SCRIPT_PATH_RE.finditer(text):
        full_match = m.group(0)  # e.g. ``scripts/train/base_train.py``
        candidate = REPO_ROOT / full_match
        assert candidate.is_file(), (
            f"{path.relative_to(REPO_ROOT)} references ``{full_match}`` which "
            f"does not exist on disk. PR #394 may have moved or removed the script. "
            f"Note: prefer ``-m scripts.X.Y`` invocations over file paths."
        )


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "runs").glob("*.sh")),
    ids=lambda p: p.name,
)
def test_no_argparse_style_after_torchrun_dash_dash(path: Path) -> None:
    """torchrun ... -m scripts.X -- depth=12 (Hydra) NOT --depth=12 (argparse).

    Slice 18 P0 #1 — ``runs/speedrun.sh:73`` shipped with broken
    ``--depth=26 --target-param-data-ratio=...`` argparse-style flags after the
    Hydra ``--`` separator. Hydra rejects such flags.
    """
    text = path.read_text()
    # Find every line containing ``torchrun`` and ``-m scripts.``.
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if "torchrun" not in line:
            continue
        if "-m scripts" not in line and "-m steering" not in line and "-m nanochat" not in line:
            continue
        # Look for the multi-line continuation: include trailing ``\\``
        # extension lines.
        full_line = line
        j = i
        while full_line.rstrip().endswith("\\") and j < len(lines):
            full_line = full_line.rstrip().rstrip("\\") + " " + lines[j]
            j += 1
        # Split on ``-- ``: everything to the right is the Hydra override
        # block. Inside that block, any ``--<key>=<value>`` is broken.
        parts = full_line.split(" -- ", 1)
        if len(parts) != 2:
            continue
        override_block = parts[1]
        offending = re.findall(r"--[a-zA-Z][a-zA-Z0-9_-]*=\S+", override_block)
        # Filter out ``--config-name`` and ``--config-dir`` which Hydra accepts.
        offending = [o for o in offending if not o.startswith(("--config-name", "--config-dir"))]
        assert not offending, (
            f"{path.relative_to(REPO_ROOT)}:{i} uses argparse-style flags "
            f"after the ``-- `` separator: {offending}. Hydra requires "
            f"``key=value`` form (slice 18 P0 #1)."
        )


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "runs").glob("*.sh")),
    ids=lambda p: p.name,
)
def test_runs_no_missing_source_env(path: Path) -> None:
    """``source env.sh`` was the legacy form that never existed at repo root.

    Slice 18 P0 #3 — ``runs/sft_d12_10k.sh:7`` shipped with a hard-fail
    ``source env.sh``. Replaced by activated-venv detection.
    """
    text = path.read_text()
    # Match a top-level ``source env.sh`` (not commented out).
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"^\s*source\s+env\.sh\b", stripped):
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)}:{i} sources ``env.sh`` which "
                f"does not exist at repo root (slice 18 P0 #3). Use the "
                f"activated-venv pattern from sibling scripts."
            )


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "runs").glob("*.sh")),
    ids=lambda p: p.name,
)
def test_runs_use_pipefail(path: Path) -> None:
    """Runner scripts must not mask failed commands before/after ``tee`` pipes."""
    text = path.read_text()
    assert re.search(r"^\s*set\s+-[^\n]*pipefail\b", text, flags=re.MULTILINE), (
        f"{path.relative_to(REPO_ROOT)} must enable pipefail so piped training "
        "commands fail closed."
    )


@pytest.mark.parametrize(
    "path",
    [REPO_ROOT / "runs" / "miniseries.sh", REPO_ROOT / "runs" / "scaling_laws.sh"],
    ids=lambda p: p.name,
)
def test_optional_core_score_grep_is_guarded_under_pipefail(path: Path) -> None:
    """Missing CORE metric is an explicit optional fallback in these runners."""
    text = path.read_text()
    matches = re.findall(
        r"CORE_SCORE=\$\([^)]*grep \"CORE metric:\"[^)]*\)",
        text,
        flags=re.DOTALL,
    )
    assert matches, f"{path.relative_to(REPO_ROOT)} should parse CORE_SCORE"
    assert all("|| true" in match for match in matches), (
        f"{path.relative_to(REPO_ROOT)} parses optional CORE_SCORE without "
        "guarding the grep pipeline under pipefail."
    )
