"""Ratchet module-level production symbols that are called only by tests.

The analysis is deliberately narrower than a claim of whole-program
reachability: it resolves normal imports, qualified module access, local
references, and ``__all__`` exports. Dynamic framework dispatch and external
library APIs require an exact, reasoned exception below.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_ROOTS = ("steering", "scripts", "tasks", "viz")
_PRODUCTION_CALLER_ROOTS = (*_PRODUCTION_ROOTS, "nanochat")
_TEST_ROOTS = ("tests",)
_NON_PRODUCTION_DEFINITION_PREFIXES = ("scripts/benchmark/", "scripts/profile_progression/")


@dataclass(frozen=True, order=True)
class SymbolKey:
    module: str
    name: str
    kind: str


@dataclass(frozen=True, order=True)
class Symbol:
    key: SymbolKey
    path: str
    line: int


# Every exception is an intentional public, dynamic, diagnostic, or migration
# surface. The enforcement test also rejects stale entries, so moving/removing
# one of these symbols requires deleting its rationale.
_ALLOWED_TEST_ONLY_PRODUCTION_SYMBOLS: dict[SymbolKey, str] = {
    SymbolKey(
        "scripts.eval.axbench_adv.pilot_budget",
        "PilotBudgetAdapter",
        "class",
    ): (
        "Dynamically imported by steering.eval.axbench_base.openrouter only when "
        "the optional pilot ledger is enabled."
    ),
    SymbolKey(
        "scripts.eval.axbench_adv.pilot_budget",
        "async_budget_hooks",
        "function",
    ): (
        "Dynamically imported by the isolated vendored OpenRouter client only "
        "when the optional pilot ledger is enabled."
    ),
    SymbolKey(
        "scripts.eval.axbench_adv.pilot_budget",
        "budget_summary",
        "function",
    ): "Documented operator inspection API for the opt-in pilot ledger.",
    SymbolKey(
        "scripts.eval.axbench_adv.pilot_budget",
        "create_pilot_budget",
        "function",
    ): "Documented operator creation API for a fresh opt-in pilot ledger.",
    SymbolKey(
        "scripts.completion_tui.transparent",
        "disable_transparent_background",
        "function",
    ): "Public inverse lifecycle operation for the transparent-background patch.",
    SymbolKey(
        "steering.eval.multiseg_generator",
        "generate_completion",
        "function",
    ): "Public single-completion facade used by downstream evaluation clients.",
    SymbolKey(
        "steering.eval.harmfulness_judge",
        "write_results",
        "function",
    ): (
        "Legacy schema-v1 harmfulness writer retained as an external compatibility "
        "surface; production issue-1338 artifacts use write_harmfulness_payload."
    ),
    SymbolKey("steering.shard_validation", "assert_csr_invariants", "function"): (
        "Documented public assertion API for in-memory CSR validation."
    ),
    SymbolKey("steering.feature_io", "read_ignore_bos_from_shard", "function"): (
        "Public single-shard metadata reader, parallel to read_l0_tier_from_shard."
    ),
    SymbolKey("steering.feature_paths", "sft_phase_split", "function"): (
        "Public inverse lookup for the typed SFT phase mapping."
    ),
    SymbolKey("tasks.alpaca", "Alpaca", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
    SymbolKey("tasks.anthropic_hh", "AnthropicHH", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
    SymbolKey("tasks.code_alpaca", "CodeAlpaca", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
    SymbolKey("tasks.no_robots", "NoRobots", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
    SymbolKey("tasks.pku_saferlhf", "PKUSafeRLHF", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
    SymbolKey("tasks.ultrachat", "UltraChat", "class"): (
        "Documented standalone task loader available to custom task mixtures."
    ),
}


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _python_files(root: Path, roots: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for relative_root in roots
        if (directory := root / relative_root).exists()
        for path in directory.rglob("*.py")
    )


def _resolve_from_module(
    *,
    current_module: str,
    is_package: bool,
    imported_module: str | None,
    level: int,
) -> str:
    if level == 0:
        assert imported_module is not None
        return imported_module

    package = current_module.split(".") if is_package else current_module.split(".")[:-1]
    keep = len(package) - (level - 1)
    if keep < 0:
        return ""
    base = package[:keep]
    if imported_module:
        base.extend(imported_module.split("."))
    return ".".join(base)


def _attribute_chain(node: ast.Attribute) -> tuple[str, tuple[str, ...]] | None:
    attributes: list[str] = []
    cursor: ast.expr = node
    while isinstance(cursor, ast.Attribute):
        attributes.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    return cursor.id, tuple(reversed(attributes))


def _assigned_string_exports(tree: ast.Module) -> set[str]:
    exports: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            exports.update(
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return exports


def _references_from_file(
    *,
    path: Path,
    tree: ast.Module,
    root: Path,
    definitions: dict[tuple[str, str], Symbol],
    modules: set[str],
) -> set[tuple[str, str]]:
    current_module = _module_name(path, root)
    loads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    exported_names = _assigned_string_exports(tree)
    used_bindings = loads | exported_names
    references = {
        (current_module, name) for name in used_bindings if (current_module, name) in definitions
    }

    direct_aliases: dict[str, tuple[str, str]] = {}
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                binding = imported.asname or imported.name.split(".")[0]
                module_aliases[binding] = (
                    imported.name if imported.asname else imported.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_module(
                current_module=current_module,
                is_package=path.name == "__init__.py",
                imported_module=node.module,
                level=node.level,
            )
            for imported in node.names:
                if imported.name == "*":
                    references.update(key for key in definitions if key[0] == base)
                    continue
                binding = imported.asname or imported.name
                submodule = f"{base}.{imported.name}" if base else imported.name
                if submodule in modules:
                    module_aliases[binding] = submodule
                else:
                    direct_aliases[binding] = (base, imported.name)

    references.update(
        target
        for binding, target in direct_aliases.items()
        if binding in used_bindings and target in definitions
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain = _attribute_chain(node)
            if chain is None:
                continue
            binding, attributes = chain
            if binding not in module_aliases:
                continue
            parts = [*module_aliases[binding].split("."), *attributes]
            for index in range(len(parts) - 1, 0, -1):
                target = (".".join(parts[:index]), parts[index])
                if target in definitions:
                    references.add(target)
                    break
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in module_aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            target = (module_aliases[node.args[0].id], node.args[1].value)
            if target in definitions:
                references.add(target)
    return references


def find_test_only_production_symbols(
    root: Path,
    *,
    production_roots: tuple[str, ...] = _PRODUCTION_ROOTS,
    production_caller_roots: tuple[str, ...] = _PRODUCTION_CALLER_ROOTS,
    test_roots: tuple[str, ...] = _TEST_ROOTS,
) -> tuple[Symbol, ...]:
    """Return module-level functions/classes referenced by tests but not production."""
    all_roots = tuple(dict.fromkeys((*production_caller_roots, *test_roots)))
    files = _python_files(root, all_roots)
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in files
    }
    modules = {_module_name(path, root) for path in files}

    definitions: dict[tuple[str, str], Symbol] = {}
    for path, tree in trees.items():
        module = _module_name(path, root)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                key = SymbolKey(module=module, name=node.name, kind=kind)
                definitions[(module, node.name)] = Symbol(
                    key=key,
                    path=path.relative_to(root).as_posix(),
                    line=node.lineno,
                )

    production_references: set[tuple[str, str]] = set()
    test_references: set[tuple[str, str]] = set()
    for path, tree in trees.items():
        references = _references_from_file(
            path=path,
            tree=tree,
            root=root,
            definitions=definitions,
            modules=modules,
        )
        top_level = path.relative_to(root).parts[0]
        if top_level in test_roots:
            test_references.update(references)
        else:
            production_references.update(references)

    production_root_set = set(production_roots)
    return tuple(
        sorted(
            symbol
            for identity, symbol in definitions.items()
            if symbol.path.split("/", 1)[0] in production_root_set
            and not symbol.path.startswith(_NON_PRODUCTION_DEFINITION_PREFIXES)
            and identity in test_references
            and identity not in production_references
        )
    )


def _format_symbols(symbols: set[SymbolKey], by_key: dict[SymbolKey, Symbol]) -> str:
    return "\n".join(
        f"  {by_key[key].path}:{by_key[key].line}: {key.kind} {key.name}" for key in sorted(symbols)
    )


def test_no_unapproved_test_only_production_symbols() -> None:
    findings = find_test_only_production_symbols(_REPO_ROOT)
    by_key = {finding.key: finding for finding in findings}
    actual = set(by_key)
    allowed = set(_ALLOWED_TEST_ONLY_PRODUCTION_SYMBOLS)
    unexpected = actual - allowed
    stale = allowed - actual

    messages: list[str] = []
    if unexpected:
        messages.append(
            "Production symbols are called only by tests. Move test machinery "
            "under tests/, add a production caller, or document a dynamic/external "
            f"contract in the exact exception map:\n{_format_symbols(unexpected, by_key)}"
        )
    if stale:
        messages.append(
            "Stale production-caller exceptions must be removed:\n"
            + "\n".join(f"  {key.module}.{key.name} ({key.kind})" for key in sorted(stale))
        )
    assert not messages, "\n\n".join(messages)


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")


def test_finder_resolves_direct_and_qualified_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "app/api.py",
        """
        def direct():
            return 1

        def qualified():
            return 2
        """,
    )
    _write(
        tmp_path / "tests/test_api.py",
        """
        import app.api as api
        from app.api import direct

        assert direct() == api.qualified()
        """,
    )

    findings = find_test_only_production_symbols(
        tmp_path,
        production_roots=("app",),
        production_caller_roots=("app",),
        test_roots=("tests",),
    )

    assert {finding.key.name for finding in findings} == {"direct", "qualified"}


def test_finder_ignores_symbols_with_production_callers(tmp_path: Path) -> None:
    _write(tmp_path / "app/api.py", "def live():\n    return 1\n")
    _write(tmp_path / "app/caller.py", "from app.api import live\n\nVALUE = live()\n")
    _write(tmp_path / "tests/test_api.py", "from app.api import live\n\nassert live() == 1\n")

    findings = find_test_only_production_symbols(
        tmp_path,
        production_roots=("app",),
        production_caller_roots=("app",),
        test_roots=("tests",),
    )

    assert findings == ()


def test_finder_treats_production_reexports_as_callers(tmp_path: Path) -> None:
    _write(tmp_path / "app/api.py", "def public_helper():\n    return 1\n")
    _write(
        tmp_path / "app/__init__.py",
        """
        from app.api import public_helper

        __all__ = ["public_helper"]
        """,
    )
    _write(
        tmp_path / "tests/test_api.py",
        "from app.api import public_helper\n\nassert public_helper() == 1\n",
    )

    findings = find_test_only_production_symbols(
        tmp_path,
        production_roots=("app",),
        production_caller_roots=("app",),
        test_roots=("tests",),
    )

    assert findings == ()


def test_finder_does_not_confuse_same_named_functions(tmp_path: Path) -> None:
    _write(tmp_path / "app/left.py", "def helper():\n    return 'left'\n")
    _write(tmp_path / "app/right.py", "def helper():\n    return 'right'\n")
    _write(tmp_path / "app/caller.py", "from app.left import helper\n\nVALUE = helper()\n")
    _write(
        tmp_path / "tests/test_right.py",
        "from app.right import helper\n\nassert helper() == 'right'\n",
    )

    findings = find_test_only_production_symbols(
        tmp_path,
        production_roots=("app",),
        production_caller_roots=("app",),
        test_roots=("tests",),
    )

    assert [(finding.key.module, finding.key.name) for finding in findings] == [
        ("app.right", "helper")
    ]
