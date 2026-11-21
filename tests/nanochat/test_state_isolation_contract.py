"""Contract test: tests/ and scripts/ files must not mutate global PyTorch
or nanochat state at module import time.

Module-top side effects (e.g., torch.set_float32_matmul_precision('high'),
nanochat.flash_attention._override_impl = SDPA, nanochat.common.compute_init())
fire during pytest collection and persist across test boundaries, breaking
order-independence. This contract test scans every .py file under tests/ and
scripts/ via AST and fails if any forbidden mutation appears at module top.

Wrap such calls in a function (main(), an autouse fixture, etc.) instead.

KNOWN_LATENT_OFFENDERS lists files with module-top compute_init() that are
not currently imported by any test, so the leak is dormant. Each entry is
tracked for follow-up; do not add new ones — refactor the file instead.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Functions whose call at module top-level is forbidden because they mutate
# global PyTorch / nanochat state. Matched by full attribute chain
# (e.g., "torch.set_default_dtype") OR by last-segment short name (e.g.,
# "compute_init"). The short-name match accepts aliases (`import torch as t`
# → `t.set_default_dtype(...)`).
FORBIDDEN_CALLS: frozenset[str] = frozenset({
    # nanochat
    "compute_init",
    "nanochat.common.compute_init",
    # torch precision
    "torch.set_float32_matmul_precision",
    "set_float32_matmul_precision",
    # torch defaults
    "torch.set_default_dtype",
    "set_default_dtype",
    "torch.set_default_device",
    "set_default_device",
    # torch determinism
    "torch.use_deterministic_algorithms",
    "use_deterministic_algorithms",
    "torch.set_grad_enabled",
    "set_grad_enabled",
    # torch seeds
    "torch.manual_seed",
    "manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
    "manual_seed_all",
    # python stdlib globals
    "logging.basicConfig",  # mutates root logger; test capture interactions
    "basicConfig",
})

# Last-segment names of attributes whose assignment at module top-level is
# forbidden (e.g., `<x>._override_impl = ...`).
FORBIDDEN_ATTR_ASSIGNS: frozenset[str] = frozenset({
    "_override_impl",   # nanochat.flash_attention._override_impl
    "allow_tf32",       # torch.backends.cuda.matmul.allow_tf32 / cudnn
    "deterministic",    # torch.backends.cudnn.deterministic
    "benchmark",        # torch.backends.cudnn.benchmark
})

# Files exempted from this check. Each entry must have a comment explaining
# why. These represent latent bugs — fix the file rather than adding new
# entries here. The contract test fails if an allowlisted file no longer has
# violations (signalling the entry should be removed).
KNOWN_LATENT_OFFENDERS: frozenset[str] = frozenset({
    # Module-top compute_init() — not currently imported by any test, so the
    # leak is dormant. Tracked for follow-up (one PR per script).
    "scripts/train/base_train.py",
    "scripts/train/chat_rl.py",
})


def _attr_chain(node: ast.AST) -> str:
    """Return 'a.b.c' for `Attribute(Name(a).b).c` chains, '' otherwise."""
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _is_main_guard(node: ast.If) -> bool:
    """True iff `node` is `if __name__ == '__main__':`."""
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left = test.left
    right = test.comparators[0]
    is_dunder_name = isinstance(left, ast.Name) and left.id == "__name__"
    is_main_str = isinstance(right, ast.Constant) and right.value == "__main__"
    return is_dunder_name and is_main_str


def _walk_module_top_level(tree: ast.Module) -> Iterable[ast.AST]:
    """Yield every node reachable from the module body without crossing a
    FunctionDef / AsyncFunctionDef / ClassDef / Lambda boundary, or an
    `if __name__ == '__main__':` guard."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            stack.extend(node.orelse)
            continue
        for child in ast.iter_child_nodes(node):
            stack.append(child)


def _iter_attribute_targets(node: ast.AST) -> Iterable[ast.Attribute]:
    """Yield every `Attribute` reachable through `Tuple` / `List` / `Starred`
    target wrappers. Handles tuple/list unpacking with attribute targets like
    `fa._override_impl, x = SDPA, 1`."""
    if isinstance(node, ast.Attribute):
        yield node
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _iter_attribute_targets(elt)
    elif isinstance(node, ast.Starred):
        yield from _iter_attribute_targets(node.value)


def _check_file(path: Path) -> list[str]:
    """Return a list of human-readable violations found in `path`, sorted by
    line number."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    # Collect (lineno, message) tuples so we can sort deterministically.
    raw: list[tuple[int, str]] = []
    for node in _walk_module_top_level(tree):
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            short = chain.rsplit(".", 1)[-1] if chain else ""
            if chain in FORBIDDEN_CALLS or short in FORBIDDEN_CALLS:
                raw.append((
                    node.lineno,
                    f"line {node.lineno}: forbidden module-top call to "
                    f"{chain or short!r}",
                ))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for attr_target in _iter_attribute_targets(target):
                    if attr_target.attr in FORBIDDEN_ATTR_ASSIGNS:
                        chain = _attr_chain(attr_target)
                        raw.append((
                            attr_target.lineno,
                            f"line {attr_target.lineno}: forbidden module-top "
                            f"assignment to {chain or attr_target.attr}",
                        ))
        if isinstance(node, ast.AnnAssign):
            for attr_target in _iter_attribute_targets(node.target):
                if attr_target.attr in FORBIDDEN_ATTR_ASSIGNS:
                    chain = _attr_chain(attr_target)
                    raw.append((
                        attr_target.lineno,
                        f"line {attr_target.lineno}: forbidden module-top "
                        f"annotated assignment to {chain or attr_target.attr}",
                    ))
        if isinstance(node, ast.AugAssign):
            for attr_target in _iter_attribute_targets(node.target):
                if attr_target.attr in FORBIDDEN_ATTR_ASSIGNS:
                    chain = _attr_chain(attr_target)
                    raw.append((
                        attr_target.lineno,
                        f"line {attr_target.lineno}: forbidden module-top "
                        f"aug-assignment to {chain or attr_target.attr}",
                    ))
    raw.sort(key=lambda x: x[0])
    return [msg for _, msg in raw]


def _python_files_in(*subdirs: str) -> list[Path]:
    files: list[Path] = []
    for subdir in subdirs:
        for path in (REPO_ROOT / subdir).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


SELF_PATH = Path(__file__).resolve()
_PY_FILES = _python_files_in("tests", "scripts")


# ---------------------------------------------------------------------------
# Main contract scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "py_file",
    _PY_FILES,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_no_module_top_state_mutation(py_file: Path) -> None:
    """File must not mutate global PyTorch/nanochat state at module import."""
    if py_file == SELF_PATH:
        pytest.skip("contract test does not lint itself")
    rel = str(py_file.relative_to(REPO_ROOT))
    violations = _check_file(py_file)

    if rel in KNOWN_LATENT_OFFENDERS:
        if not violations:
            pytest.fail(
                f"{rel} is in KNOWN_LATENT_OFFENDERS but has no module-top "
                f"violations any more. Please remove it from the allowlist "
                f"in tests/nanochat/test_state_isolation_contract.py."
            )
        pytest.skip(f"{rel} is on KNOWN_LATENT_OFFENDERS allowlist")

    assert not violations, (
        f"{rel} mutates global state at module import time:\n  "
        + "\n  ".join(violations)
        + "\n\nWrap these in a function (e.g., main()) or pytest fixture. "
        "See docs/superpowers/specs/2026-04-28-test-order-state-leaks-design.md."
    )


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestContractCheckerHelpers:
    """Unit tests for _check_file, _walk_module_top_level, etc."""

    def _write(self, tmp_path: Path, source: str) -> Path:
        target = tmp_path / "candidate.py"
        target.write_text(source, encoding="utf-8")
        return target

    def test_compute_init_at_module_top_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\ncompute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_compute_init_inside_main_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "def main():\n    compute_init('cuda')\n",
        )
        assert _check_file(path) == []

    def test_compute_init_inside_main_guard_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "if __name__ == '__main__':\n    compute_init('cuda')\n",
        )
        assert _check_file(path) == []

    def test_compute_init_inside_top_level_if_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch\n"
            "from nanochat.common import compute_init\n"
            "if torch.cuda.is_available():\n    compute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_override_impl_assign_at_module_top_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import nanochat.flash_attention as fa\n"
            "fa._override_impl = fa.FlashAttnImpl.SDPA\n",
        )
        violations = _check_file(path)
        assert any("_override_impl" in v for v in violations)

    def test_override_impl_assign_inside_fixture_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import pytest\n"
            "import nanochat.flash_attention as fa\n"
            "@pytest.fixture(autouse=True)\n"
            "def _force_sdpa():\n"
            "    fa._override_impl = fa.FlashAttnImpl.SDPA\n"
            "    yield\n"
            "    fa._override_impl = None\n",
        )
        assert _check_file(path) == []

    def test_torch_set_default_dtype_at_module_top_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch\ntorch.set_default_dtype(torch.float64)\n",
        )
        violations = _check_file(path)
        assert any("set_default_dtype" in v for v in violations)

    def test_aliased_set_default_dtype_at_module_top_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch as t\nt.set_default_dtype(t.float64)\n",
        )
        violations = _check_file(path)
        assert any("set_default_dtype" in v for v in violations)

    def test_set_default_dtype_inside_method_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch\n"
            "class Foo:\n"
            "    def x(self):\n"
            "        torch.set_default_dtype(torch.float64)\n",
        )
        assert _check_file(path) == []

    def test_empty_file_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "")
        assert _check_file(path) == []

    def test_syntax_error_file_is_silently_skipped(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "def broken(:\n")
        assert _check_file(path) == []

    def test_main_guard_recognised(self) -> None:
        tree = ast.parse("if __name__ == '__main__':\n    pass\n")
        if_node = tree.body[0]
        assert isinstance(if_node, ast.If)
        assert _is_main_guard(if_node) is True

    def test_other_if_not_recognised_as_main_guard(self) -> None:
        tree = ast.parse("if __name__ == 'other':\n    pass\n")
        if_node = tree.body[0]
        assert isinstance(if_node, ast.If)
        assert _is_main_guard(if_node) is False

    def test_attr_chain_extracts_dotted_name(self) -> None:
        tree = ast.parse("torch.backends.cuda.matmul")
        expr_node = tree.body[0]
        assert isinstance(expr_node, ast.Expr)
        assert _attr_chain(expr_node.value) == "torch.backends.cuda.matmul"

    def test_attr_chain_handles_non_attribute(self) -> None:
        tree = ast.parse("'a string'")
        expr_node = tree.body[0]
        assert isinstance(expr_node, ast.Expr)
        assert _attr_chain(expr_node.value) == ""

    # -----------------------------------------------------------------------
    # AnnAssign and tuple-unpack coverage (regressions: prior versions of
    # _check_file silently missed these idioms)
    # -----------------------------------------------------------------------

    def test_annotated_assignment_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import nanochat.flash_attention as fa\n"
            "fa._override_impl: object = fa.FlashAttnImpl.SDPA\n",
        )
        violations = _check_file(path)
        assert any("_override_impl" in v for v in violations)

    def test_tuple_unpack_with_attribute_target_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import nanochat.flash_attention as fa\n"
            "fa._override_impl, x = fa.FlashAttnImpl.SDPA, 1\n",
        )
        violations = _check_file(path)
        assert any("_override_impl" in v for v in violations)

    def test_list_unpack_with_attribute_target_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import nanochat.flash_attention as fa\n"
            "[fa._override_impl, x] = [fa.FlashAttnImpl.SDPA, 1]\n",
        )
        violations = _check_file(path)
        assert any("_override_impl" in v for v in violations)

    def test_starred_unpack_with_attribute_target_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import nanochat.flash_attention as fa\n"
            "*rest, fa._override_impl = [1, 2, fa.FlashAttnImpl.SDPA]\n",
        )
        violations = _check_file(path)
        assert any("_override_impl" in v for v in violations)

    def test_aug_assign_to_forbidden_attr_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch\n"
            "torch.backends.cudnn.benchmark |= True\n",
        )
        violations = _check_file(path)
        assert any("benchmark" in v for v in violations)

    # -----------------------------------------------------------------------
    # Parametrized coverage for every entry in FORBIDDEN_CALLS / _ATTR_ASSIGNS
    # so that no entry can be added without a matching test.
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("name", "source_line"),
        [
            ("compute_init", "compute_init('cuda')"),
            ("set_float32_matmul_precision", "torch.set_float32_matmul_precision('high')"),
            ("set_default_dtype", "torch.set_default_dtype(torch.float64)"),
            ("set_default_device", "torch.set_default_device('cpu')"),
            ("use_deterministic_algorithms", "torch.use_deterministic_algorithms(True)"),
            ("set_grad_enabled", "torch.set_grad_enabled(False)"),
            ("manual_seed", "torch.manual_seed(0)"),
            ("manual_seed_all", "torch.cuda.manual_seed_all(0)"),
            ("basicConfig", "logging.basicConfig(level=logging.INFO)"),
        ],
        ids=lambda v: v[0] if isinstance(v, tuple) else v,
    )
    def test_each_forbidden_call_is_flagged(
        self, tmp_path: Path, name: str, source_line: str
    ) -> None:
        path = self._write(
            tmp_path,
            "import logging\nimport torch\nfrom nanochat.common import compute_init\n"
            + source_line
            + "\n",
        )
        violations = _check_file(path)
        assert any(name in v for v in violations), (
            f"expected {name!r} flagged in {source_line!r}; got {violations}"
        )

    @pytest.mark.parametrize(
        ("name", "source_line"),
        [
            ("_override_impl", "fa._override_impl = 0"),
            ("allow_tf32", "torch.backends.cuda.matmul.allow_tf32 = True"),
            ("deterministic", "torch.backends.cudnn.deterministic = True"),
            ("benchmark", "torch.backends.cudnn.benchmark = False"),
        ],
        ids=lambda v: v[0] if isinstance(v, tuple) else v,
    )
    def test_each_forbidden_attr_assign_is_flagged(
        self, tmp_path: Path, name: str, source_line: str
    ) -> None:
        path = self._write(
            tmp_path,
            "import torch\nimport nanochat.flash_attention as fa\n" + source_line + "\n",
        )
        violations = _check_file(path)
        assert any(name in v for v in violations), (
            f"expected {name!r} flagged in {source_line!r}; got {violations}"
        )

    # -----------------------------------------------------------------------
    # Output ordering: violations must be sorted by line number so a long
    # report is readable and stable across runs.
    # -----------------------------------------------------------------------

    def test_violations_sorted_by_line_number(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "import torch\n"
            "torch.set_default_dtype(torch.float64)\n"
            "torch.set_default_device('cpu')\n"
            "torch.use_deterministic_algorithms(True)\n"
            "torch.set_grad_enabled(False)\n"
            "torch.manual_seed(0)\n",
        )
        violations = _check_file(path)
        line_numbers = [int(v.split(":", 1)[0].split()[-1]) for v in violations]
        assert line_numbers == sorted(line_numbers), (
            f"violations not sorted by line: {violations}"
        )

    # -----------------------------------------------------------------------
    # Boundary nodes: ensure the walker descends into module-top control flow
    # (which executes on import) and skips function/class/main-guard bodies.
    # -----------------------------------------------------------------------

    def test_compute_init_inside_top_level_try_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "try:\n    pass\nexcept Exception:\n    compute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_compute_init_inside_top_level_with_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from contextlib import nullcontext\n"
            "from nanochat.common import compute_init\n"
            "with nullcontext():\n    compute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_compute_init_inside_top_level_for_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "for _ in range(1):\n    compute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_compute_init_inside_top_level_match_is_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "x = 'cuda'\n"
            "match x:\n    case 'cuda':\n        compute_init('cuda')\n",
        )
        violations = _check_file(path)
        assert any("compute_init" in v for v in violations)

    def test_compute_init_inside_async_function_is_not_flagged(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "async def main_async():\n    compute_init('cuda')\n",
        )
        assert _check_file(path) == []

    def test_compute_init_inside_lambda_is_not_flagged(self, tmp_path: Path) -> None:
        # Pathological but valid: lambda body is not run on import.
        path = self._write(
            tmp_path,
            "from nanochat.common import compute_init\n"
            "f = lambda: compute_init('cuda')\n",
        )
        assert _check_file(path) == []
