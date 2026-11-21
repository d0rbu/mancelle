"""Verify audited files keep imports at top of file.

Pins map suspected-bug #34 (P2): manifest §3.3 local-import violation.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDITED_PATHS = [
    "tests/reference_extraction_pipeline.py",
    "scripts/benchmark/verify_overlay_cache_byte_identity.py",
    "scripts/eval/build_eval_context_cache.py",
    "scripts/eval/validate_eval_context_cache.py",
    "scripts/train/training_progression.py",
    "scripts/utils/nnterp_weight_transform.py",
    "tests/conftest.py",
]


@pytest.mark.parametrize("rel_path", AUDITED_PATHS)
def test_audited_files_imports_at_top(rel_path: str):
    """No `import` / `from ... import` statements inside function bodies.

    Per project memory: imports MUST be at top of file — no exceptions.
    """
    source = (REPO_ROOT / rel_path).read_text()
    tree = ast.parse(source)

    # Walk every FunctionDef / AsyncFunctionDef / ClassDef and check that no
    # import statements appear in their bodies.
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    violations.append(
                        f"{rel_path}:{child.lineno}: import inside function "
                        f"{node.name!r}"
                    )

    assert not violations, "\n".join(violations)
