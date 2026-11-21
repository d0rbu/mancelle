"""Structural invariants for the test tree.

These tests enforce the test-tree organization. They fail loudly when a
contributor adds a flat test file at `tests/` root, or when a test ends up
in the wrong subdirectory relative to the source module it tests. Keeping
these invariants in CI (via the default fast-test run) prevents the tree
from drifting back to a flat layout.

Conventions enforced:

- Test modules live under a subdirectory that mirrors the source tree:
  `tests/steering/...` for tests of `steering/...`, `tests/nanochat/...`
  for tests of `nanochat/...`, etc. No `test_*.py` files directly at
  `tests/` root (this file is the documented exception).
- Uncategorized subdirectories should not become a new dumping ground for
  tests whose primary import clearly belongs to a known source package
  (`steering`, `nanochat`, `scripts`, `tasks`). Existing source-package
  subdirectories may contain intentional cross-boundary tests.

Curated subdirectories (`tests/axbench_adv/`, `tests/axbench_base/`,
`tests/axbench_bpb/`, `tests/feature_filter/`, `tests/profiling/`,
`tests/code_quality/`, `tests/test_completion_tui/`) are exempt from the
mirror rule because they organize tests around the eval / TUI surface area
rather than a single source package.
"""

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).parent
REPO_ROOT = TESTS_ROOT.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"

# This file is the structural-invariants test itself; it intentionally
# lives at the tests/ root.
ROOT_TEST_EXEMPTIONS = {"test_structure.py"}

# Pre-existing curated subdirs that organize tests around an eval / TUI
# surface rather than mirroring a single source package. Tests inside
# these dirs are exempt from the per-source mirror rule.
CURATED_SUBDIRS = {
    "axbench_adv",
    "axbench_base",
    "axbench_bpb",
    "axbench_bpb_adv",
    "code_quality",
    "feature_filter",
    "harmfulness_progression",
    "profiling",
    "test_completion_tui",
    "fixtures",
    # By-discipline subdirs (orthogonal to source-mirror layout): cover
    # property-style fuzz and gated-slow tier-3 smoke. Per the Phase 1/2
    # plan, these intentionally cross source-package boundaries.
    "properties",
    "slow",
}

SOURCE_PACKAGES = ("steering", "nanochat", "scripts", "tasks")
SOURCE_TEST_SUBDIRS = set(SOURCE_PACKAGES)

IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+(\w+)(?:\.\w+)*\s+import|import\s+(\w+))",
    re.MULTILINE,
)


def _primary_source_package(test_file: Path) -> str | None:
    """Return the source package this test most plausibly tests, or None.

    Counts top-level imports from each known source package and returns
    the one with the most imports. Ties (or no relevant imports) → None,
    meaning the test doesn't have a clear single-package home and the
    mirror rule doesn't apply.
    """
    text = test_file.read_text(encoding="utf-8", errors="ignore")
    counts = dict.fromkeys(SOURCE_PACKAGES, 0)
    for match in IMPORT_PATTERN.finditer(text):
        module = match.group(1) or match.group(2)
        if module in counts:
            counts[module] += 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top, top_count = ranked[0]
    if top_count == 0:
        return None
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return None
    return top


def test_no_flat_test_files_at_tests_root():
    """All test_*.py files must live under a subdirectory."""
    flat = sorted(
        p.name for p in TESTS_ROOT.glob("test_*.py")
        if p.name not in ROOT_TEST_EXEMPTIONS
    )
    assert flat == [], (
        f"Found {len(flat)} test files at tests/ root that should live "
        f"under a subdirectory mirroring their source package "
        f"(tests/steering/, tests/nanochat/, tests/scripts/, tests/tasks/). "
        f"First 10: {flat[:10]}"
    )


def test_steering_tests_live_under_tests_steering():
    """Tests whose primary import is from `steering.*` live under tests/steering/."""
    misplaced: list[str] = []
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        rel = test_file.relative_to(TESTS_ROOT)
        if rel.name in ROOT_TEST_EXEMPTIONS:
            continue
        if rel.parts and rel.parts[0] in CURATED_SUBDIRS | SOURCE_TEST_SUBDIRS:
            continue
        if _primary_source_package(test_file) != "steering":
            continue
        if rel.parts[0] != "steering":
            misplaced.append(str(rel))
    assert misplaced == [], (
        f"{len(misplaced)} test files import primarily from steering.* but "
        f"live outside tests/steering/. First 10: {misplaced[:10]}"
    )


def test_nanochat_tests_live_under_tests_nanochat():
    """Tests whose primary import is from `nanochat.*` live under tests/nanochat/."""
    misplaced: list[str] = []
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        rel = test_file.relative_to(TESTS_ROOT)
        if rel.name in ROOT_TEST_EXEMPTIONS:
            continue
        if rel.parts and rel.parts[0] in CURATED_SUBDIRS | SOURCE_TEST_SUBDIRS:
            continue
        if _primary_source_package(test_file) != "nanochat":
            continue
        if rel.parts[0] != "nanochat":
            misplaced.append(str(rel))
    assert misplaced == [], (
        f"{len(misplaced)} test files import primarily from nanochat.* but "
        f"live outside tests/nanochat/. First 10: {misplaced[:10]}"
    )


def test_scripts_tests_live_under_tests_scripts():
    """Tests whose primary import is from `scripts.*` live under tests/scripts/."""
    misplaced: list[str] = []
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        rel = test_file.relative_to(TESTS_ROOT)
        if rel.name in ROOT_TEST_EXEMPTIONS:
            continue
        if rel.parts and rel.parts[0] in CURATED_SUBDIRS | SOURCE_TEST_SUBDIRS:
            continue
        if _primary_source_package(test_file) != "scripts":
            continue
        if rel.parts[0] != "scripts":
            misplaced.append(str(rel))
    assert misplaced == [], (
        f"{len(misplaced)} test files import primarily from scripts.* but "
        f"live outside tests/scripts/. First 10: {misplaced[:10]}"
    )


def test_tasks_tests_live_under_tests_tasks():
    """Tests whose primary import is from `tasks.*` live under tests/tasks/."""
    misplaced: list[str] = []
    for test_file in TESTS_ROOT.rglob("test_*.py"):
        rel = test_file.relative_to(TESTS_ROOT)
        if rel.name in ROOT_TEST_EXEMPTIONS:
            continue
        if rel.parts and rel.parts[0] in CURATED_SUBDIRS | SOURCE_TEST_SUBDIRS:
            continue
        if _primary_source_package(test_file) != "tasks":
            continue
        if rel.parts[0] != "tasks":
            misplaced.append(str(rel))
    assert misplaced == [], (
        f"{len(misplaced)} test files import primarily from tasks.* but "
        f"live outside tests/tasks/. First 10: {misplaced[:10]}"
    )


# ---------------------------------------------------------------------------
# scripts/ structure invariants
# ---------------------------------------------------------------------------

# Files allowed at scripts/ root. __init__.py is required for package
# import; everything else (entry-point scripts) lives under a
# purpose-based subdirectory.
SCRIPTS_ROOT_PY_EXEMPTIONS = {"__init__.py"}

# Purpose-based subdirs that should exist so contributors know where to
# put new scripts. Pre-existing curated subdirs (completion_tui/,
# profile_progression/, profile_progression_microbench/, slurm/, test/,
# validation/) are not enforced here; the rule only requires the new
# purpose subdirs exist.
SCRIPTS_PURPOSE_SUBDIRS = {
    "train",        # Model training (pretrain, SFT, RL, sweeps)
    "eval",         # Checkpoint evaluation
    "chat",         # Interactive chat CLIs / web UI
    "preprocess",   # Data prep, precomputation, tokenizer training, conversion
    "benchmark",    # Performance microbenchmarks + ad-hoc profiling
    "utils",        # Misc validation / verification / weight-transform tools
}


def test_no_flat_python_scripts_at_scripts_root():
    """All Python entry points must live under a subdirectory."""
    flat = sorted(
        p.name for p in SCRIPTS_ROOT.glob("*.py")
        if p.name not in SCRIPTS_ROOT_PY_EXEMPTIONS
    )
    assert flat == [], (
        f"Found {len(flat)} Python files at scripts/ root that should live "
        f"under a purpose-based subdirectory ({sorted(SCRIPTS_PURPOSE_SUBDIRS)}). "
        f"First 10: {flat[:10]}"
    )


def test_scripts_purpose_subdirs_exist():
    """The purpose-based subdirs exist (so new scripts have a home)."""
    actual = {p.name for p in SCRIPTS_ROOT.iterdir() if p.is_dir()}
    missing = SCRIPTS_PURPOSE_SUBDIRS - actual
    assert missing == set(), (
        f"Missing purpose-based subdirs under scripts/: {sorted(missing)}. "
        f"Create them so contributors know where to put new scripts."
    )
