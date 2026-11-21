"""Resolution tests for ``scripts/slurm/*.sbatch`` Python module references.

Slice 11 audit P0 #2: ``scripts/slurm/train.sbatch`` referenced
``scripts/pretrain_steered.py`` which never existed in this repo.
PR #394 reorganized scripts under ``scripts/{train,eval,...}/``, so
the modern entry point is ``scripts.train.base_train`` driven by
Hydra.

This test sweeps all ``scripts/slurm/*.sbatch`` files and asserts that
every ``-m scripts.X.Y`` reference resolves to a file on disk. It also
asserts the legacy ``scripts/pretrain_steered.py`` reference is gone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SLURM_DIR = REPO_ROOT / "scripts" / "slurm"


def test_slurm_directory_exists() -> None:
    assert SLURM_DIR.is_dir(), f"missing {SLURM_DIR}"


@pytest.mark.parametrize(
    "sbatch_path",
    sorted(SLURM_DIR.glob("*.sbatch")),
    ids=lambda p: p.name,
)
def test_module_references_resolve(sbatch_path: Path) -> None:
    """Each ``-m scripts.X.Y`` reference in a sbatch file must
    resolve to a file in the repo."""
    text = sbatch_path.read_text()
    # Match ``-m scripts.X.Y...`` invocations. Only the dotted module
    # part (without the ``-m`` prefix) is captured.
    pattern = re.compile(r"-m\s+(scripts(?:\.\w+)+)")
    for m in pattern.finditer(text):
        module_path = m.group(1)
        rel = module_path.replace(".", "/") + ".py"
        candidate = REPO_ROOT / rel
        assert candidate.is_file(), (
            f"{sbatch_path.name} references ``-m {module_path}`` which "
            f"does not resolve to a file (expected at {candidate}). "
            f"If the script was moved, update the sbatch."
        )


@pytest.mark.parametrize(
    "sbatch_path",
    sorted(SLURM_DIR.glob("*.sbatch")),
    ids=lambda p: p.name,
)
def test_no_legacy_pretrain_steered_reference(sbatch_path: Path) -> None:
    """``scripts/pretrain_steered.py`` was the legacy reference that
    never existed. Slice 11 audit P0 #2 removed it."""
    text = sbatch_path.read_text()
    assert "pretrain_steered" not in text, (
        f"{sbatch_path.name} still references the (never-existed) "
        f"legacy ``scripts/pretrain_steered.py`` script. The modern "
        f"entry point is ``scripts.train.base_train`` (slice 11 audit P0 #2)."
    )
