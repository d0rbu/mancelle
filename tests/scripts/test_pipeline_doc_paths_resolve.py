"""Resolution tests for ``docs/pipelines/*-training.md`` script paths.

Slice 11 audit P0 #3: the pipeline docs and architecture reference
listed pre-reorg paths like ``scripts/base_train.py`` which were
moved to ``scripts/train/base_train.py`` in PR #394. New users
following these docs hit dead links.

This test greps the relevant docs for ``scripts/<name>.py`` references
and asserts each resolves to a file in the repo. It is intentionally
**permissive** for path forms that are NOT load-bearing references
(e.g. inside a fenced code block illustrating a stale snippet) — but
the audit fixes have brought everything into a state where every
``scripts/...`` reference should resolve.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_TO_CHECK = [
    REPO_ROOT / "docs" / "pipelines" / "base-training.md",
    REPO_ROOT / "docs" / "pipelines" / "chat-training.md",
    REPO_ROOT / "docs" / "reference" / "architecture.md",
]

# Pattern matches ``scripts/<dotted_path>/<filename>.py`` references
# inside backticks (the canonical path-quoting form in our docs).
_SCRIPT_REF_PATTERN = re.compile(r"`(scripts/[\w/]+\.py)`")


@pytest.mark.parametrize(
    "doc_path",
    DOCS_TO_CHECK,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_all_referenced_scripts_exist(doc_path: Path) -> None:
    assert doc_path.is_file(), f"missing {doc_path}"
    text = doc_path.read_text()
    refs = _SCRIPT_REF_PATTERN.findall(text)
    assert refs, (
        f"{doc_path.name} contains no ``scripts/.../*.py`` references — "
        f"the regex must be wrong, or the doc was emptied (audit it)."
    )
    missing: list[str] = []
    for rel in refs:
        candidate = REPO_ROOT / rel
        if not candidate.is_file():
            missing.append(rel)
    assert not missing, (
        f"{doc_path.name} references the following non-existent script "
        f"paths: {missing}. Slice 11 audit P0 #3 should have fixed all "
        f"stale references; this list is a regression."
    )
