from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("scripts", "steering", "nanochat")
LOADERS = frozenset({
    "build_encdec_model", "load_model", "load_model_from_dir", "load_encdec_model_from_dir",
})


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _load_call_sites() -> list[tuple[str, int, ast.Call]]:
    sites = []
    for root in ROOTS:
        for p in (REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            # Skip the definitions themselves (they name the function but don't *call* it).
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) in LOADERS:
                    sites.append((p.relative_to(REPO_ROOT).as_posix(), node.lineno, node))
    return sites


def test_scanner_finds_the_known_load_sites():
    sites = _load_call_sites()
    assert len(sites) >= 20, f"expected the ~25 known load sites, found {len(sites)} — scanner broke"


def test_every_load_call_declares_eval_conditioning():
    offenders = []
    for rel, lineno, node in _load_call_sites():
        # The definitions live in checkpoint_manager.py; a def is not a Call, so only calls reach here.
        kwds = {k.arg for k in node.keywords if k.arg is not None}
        if "eval_conditioning" not in kwds:
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "these model-load call sites do not declare an eval_conditioning stance — a load must "
        "declare how it conditions the encoder (#695). Offenders:\n  " + "\n  ".join(offenders)
    )
