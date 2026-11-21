"""Cross-evaluator unification tests for the "specificity" metric.

Canonical definition: ``specificity = real_mean - random_doc_random_seg_mean``
where both baselines are drawn from two fully-independent random samples
(a different document AND a different segment position). Implemented in
``steering.eval.inline_steering_eval._aggregate_roundtrip_metrics``.

This test file locks in four invariants:

1. Only ``inline_steering_eval`` emits a metric key containing ``specificity``.
   The other three evaluators (``steering_eval``, ``shuffled_conditioning_eval``,
   ``completion_steering_eval``) must not use that substring in string
   literals (metric keys) so W&B dashboards cannot conflate them.
2. The rename-target keys for the 3 non-survivors are distinct and contain
   no ``specificity`` substring.
3. The survivor's module + aggregator docstring pins the canonical definition
   with the literal tokens ``real``, ``random_doc``, ``random_seg``.
4. The aggregator returns ``real_mean - random_mean`` bit-exactly (no tolerance).
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers: locate the 4 evaluator source files.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STEERING_EVAL = _REPO_ROOT / "steering" / "eval" / "steering_eval.py"
_INLINE_STEERING_EVAL = _REPO_ROOT / "steering" / "eval" / "inline_steering_eval.py"
_SHUFFLED_CONDITIONING_EVAL = _REPO_ROOT / "steering" / "eval" / "shuffled_conditioning_eval.py"
_COMPLETION_STEERING_EVAL = _REPO_ROOT / "steering" / "eval" / "completion_steering_eval.py"

_NON_SURVIVORS = {
    "steering_eval": _STEERING_EVAL,
    "shuffled_conditioning_eval": _SHUFFLED_CONDITIONING_EVAL,
    "completion_steering_eval": _COMPLETION_STEERING_EVAL,
}


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _non_docstring_string_literals(tree: ast.AST) -> list[str]:
    """All string Constants EXCEPT those used as a module/class/function
    docstring (the first statement's expression value)."""
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_ids:
                continue
            out.append(node.value)
    return out


# ---------------------------------------------------------------------------
# Test 1 — Only the canonical evaluator emits any metric key containing
# the substring ``specificity``.
# ---------------------------------------------------------------------------


def test_only_inline_steering_eval_emits_specificity_key():
    inline_src = _read(_INLINE_STEERING_EVAL)
    inline_tree = ast.parse(inline_src)
    inline_keys = [s for s in _non_docstring_string_literals(inline_tree) if "specificity" in s]
    assert inline_keys, (
        "Expected inline_steering_eval to emit at least one metric key containing "
        "'specificity' (the canonical survivor). Found none."
    )

    for name, path in _NON_SURVIVORS.items():
        sub_tree = ast.parse(_read(path))
        offending = [s for s in _non_docstring_string_literals(sub_tree) if "specificity" in s]
        assert not offending, (
            f"{name} emits string literal(s) containing 'specificity' outside of "
            f"docstrings: {offending!r}. Rename these so the canonical 'specificity' "
            "metric name (from inline_steering_eval) is unambiguous across W&B runs."
        )


# ---------------------------------------------------------------------------
# Test 2 — the 3 renamed keys are distinct and none contain 'specificity'.
# ---------------------------------------------------------------------------

# Canonical rename map per Phase 3 Wave 3D task 1.
_RENAME_MAP = {
    "steering_eval": "standard_win_rate_at_middle_magnitude",
    "shuffled_conditioning_eval": "shuffled_same_seg_delta",
    "completion_steering_eval": "completion_scan_select_delta",
}


def test_renamed_keys_are_unambiguous():
    names = list(_RENAME_MAP.values())
    assert len(set(names)) == len(names), f"Rename-target keys must be distinct, got {names!r}"
    for n in names:
        assert "specificity" not in n, (
            f"Renamed key {n!r} must not contain 'specificity' — that name is "
            "reserved for the canonical inline_steering_eval metric."
        )


# ---------------------------------------------------------------------------
# Test 3 — Inline eval's docstring pins the canonical definition using
# the literal tokens ``real``, ``random_doc``, ``random_seg``.
# ---------------------------------------------------------------------------


def test_docstring_explains_canonical_specificity():
    src = _read(_INLINE_STEERING_EVAL)
    tree = ast.parse(src)
    mod_doc = ast.get_docstring(tree) or ""

    # Collect all function + method docstrings (the aggregator is where the
    # canonical contract is pinned, but the module/entry-function docstrings
    # are also acceptable).
    func_docs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node)
            if d:
                func_docs.append(d)

    combined = mod_doc + "\n" + "\n".join(func_docs)
    for token in ("real", "random_doc", "random_seg"):
        assert token in combined, (
            f"Canonical specificity docstring must contain token {token!r}. "
            f"Not found anywhere in inline_steering_eval.py module/function docstrings."
        )


# ---------------------------------------------------------------------------
# Test 4 — bit-exact: the aggregator's "specificity" key equals
# real_mean - random_mean, no tolerance.
# ---------------------------------------------------------------------------


def test_canonical_specificity_is_real_minus_random():
    """With a stub set of probe scores with known means, assert the
    aggregator returns ``real_mean - random_mean`` bit-exactly (``==``,
    not ``pytest.approx``).

    Inputs are chosen so means are exactly representable in IEEE-754
    binary64, ensuring subtraction has no rounding error.
    """
    from steering.eval.inline_steering_eval import _aggregate_roundtrip_metrics

    # Real: pre=[4.0, 2.0] → mean 3.0. Random: pre=[1.0, 0.5] → mean 0.75.
    # real - random = 2.25 (exactly representable).
    raw = [
        ("real", 4.0, 8.0, 90.0),
        ("real", 2.0, 4.0, 70.0),
        ("random", 1.0, 2.0, 30.0),
        ("random", 0.5, 1.0, 10.0),
    ]
    m = _aggregate_roundtrip_metrics(raw, total_gt_features=0, n_docs=0)

    import statistics

    expected_spec = statistics.mean([4.0, 2.0]) - statistics.mean([1.0, 0.5])

    assert m["steering/roundtrip_specificity"] == expected_spec, (
        f"Expected {expected_spec!r}, got {m['steering/roundtrip_specificity']!r}"
    )
    # Sanity check the known numeric value too.
    assert m["steering/roundtrip_specificity"] == 2.25


# ---------------------------------------------------------------------------
# Test 5 — legacy mis-labelled key must not appear in steering_eval.py's
# distractor report output (regression against reintroducing "median_win_rate").
# ---------------------------------------------------------------------------


def test_steering_eval_no_longer_emits_median_win_rate_literal():
    tree = ast.parse(_read(_STEERING_EVAL))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "median_win_rate":
                    raise AssertionError(
                        "steering_eval.py still uses \"median_win_rate\" as a dict "
                        "key. Rename to \"standard_win_rate_at_middle_magnitude\"."
                    )
    # And the renamed key must be present somewhere.
    src = _read(_STEERING_EVAL)
    assert '"standard_win_rate_at_middle_magnitude"' in src, (
        "Expected steering_eval.py to emit the renamed key "
        "\"standard_win_rate_at_middle_magnitude\" in its distractor report."
    )
