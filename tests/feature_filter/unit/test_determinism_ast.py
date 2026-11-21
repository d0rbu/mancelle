"""AST-level guard: feature_filter modules must not use non-deterministic APIs.

Forbids imports of ``random``/``secrets``/``uuid`` and numpy/torch random
submodules. Forbids attribute accesses/calls on: ``os.urandom``,
``datetime.now``/``datetime.utcnow``, ``time.time``/``time.monotonic``/
``time.perf_counter``/``time.time_ns``.

A ``# allow-nondeterministic: <reason>`` comment on the same line as the
offending call whitelists it. A file-level allow comment (any line)
exempts the entire file.
"""
import ast
from pathlib import Path

PKG_DIR = Path("steering/eval/feature_filter")
FORBIDDEN_IMPORTS = {"random", "secrets", "uuid"}
FORBIDDEN_ATTR_IMPORTS = {
    "numpy.random", "np.random", "torch.random",
}
FORBIDDEN_CALLS: dict[str, set[str]] = {
    "os": {"urandom"},
    "datetime": {"now", "utcnow"},
    "time": {"time", "monotonic", "perf_counter", "time_ns"},
}
ALLOW_COMMENT = "# allow-nondeterministic:"


def _module_files() -> list[Path]:
    return sorted(p for p in PKG_DIR.rglob("*.py") if p.name != "__pycache__")


def _file_has_allow_comment(source: str) -> bool:
    # File-level whitelist: any line containing the marker exempts file from
    # import-level checks. Call-site whitelisting is per-line below.
    return any(
        line.strip().startswith(ALLOW_COMMENT)
        for line in source.splitlines()
    )


def _line_allows(source_lines: list[str], lineno: int) -> bool:
    if 1 <= lineno <= len(source_lines):
        return ALLOW_COMMENT in source_lines[lineno - 1]
    return False


def _check_imports(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    errors.append(
                        f"{path}:{node.lineno} imports forbidden module {alias.name}"
                    )
                for bad in FORBIDDEN_ATTR_IMPORTS:
                    if alias.name == bad or alias.name.startswith(bad + "."):
                        errors.append(
                            f"{path}:{node.lineno} imports forbidden module {alias.name}"
                        )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in FORBIDDEN_IMPORTS:
                errors.append(
                    f"{path}:{node.lineno} imports from forbidden module {mod}"
                )
            for bad in FORBIDDEN_ATTR_IMPORTS:
                if mod == bad or mod.startswith(bad + "."):
                    errors.append(
                        f"{path}:{node.lineno} imports from forbidden submodule {mod}"
                    )
    return errors


def _check_calls(path: Path, tree: ast.AST, source_lines: list[str]) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        # Attribute access like datetime.now, time.time, os.urandom
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            attr = node.attr
            if base in FORBIDDEN_CALLS and attr in FORBIDDEN_CALLS[base]:
                if not _line_allows(source_lines, node.lineno):
                    errors.append(
                        f"{path}:{node.lineno} uses forbidden "
                        f"{base}.{attr}; add '{ALLOW_COMMENT} <reason>' "
                        f"on that line to allow"
                    )
    return errors


def test_bin_sampler_is_scanned() -> None:
    """Guard: bin_sampler.py must be present and covered by the AST scan."""
    files = {p.name for p in _module_files()}
    assert "bin_sampler.py" in files
    assert "percentile_sampler.py" not in files


def test_no_forbidden_imports_or_calls() -> None:
    all_errors: list[str] = []
    for path in _module_files():
        source = path.read_text()
        source_lines = source.splitlines()
        tree = ast.parse(source)
        # Import-level: file-level allow comment exempts.
        if not _file_has_allow_comment(source):
            all_errors.extend(_check_imports(path, tree))
        # Call-sites: per-line allow comment.
        all_errors.extend(_check_calls(path, tree, source_lines))
    assert not all_errors, "Non-determinism guard failed:\n" + "\n".join(all_errors)


def test_guard_catches_uuid_and_urandom_in_synthetic_source(tmp_path) -> None:
    """Verify the guard would catch a new violation in bin_sampler.py."""
    bad = tmp_path / "bin_sampler.py"
    bad.write_text(
        "import uuid\n"
        "import os\n"
        "def f():\n"
        "    return uuid.uuid4(), os.urandom(16)\n"
    )
    tree = ast.parse(bad.read_text())
    source_lines = bad.read_text().splitlines()
    import_errors = _check_imports(bad, tree)
    call_errors = _check_calls(bad, tree, source_lines)
    joined = "\n".join(import_errors + call_errors)
    assert "uuid" in joined
    assert "os.urandom" in joined


def test_guard_respects_per_line_allow_comment(tmp_path) -> None:
    good = tmp_path / "m.py"
    good.write_text(
        "import datetime\n"
        "def f():\n"
        "    return datetime.now()  # allow-nondeterministic: test\n"
    )
    tree = ast.parse(good.read_text())
    source_lines = good.read_text().splitlines()
    errors = _check_calls(good, tree, source_lines)
    assert errors == []


def test_all_dataclasses_frozen() -> None:
    import importlib
    import pkgutil

    import steering.eval.feature_filter as pkg

    for info in pkgutil.walk_packages(pkg.__path__, prefix="steering.eval.feature_filter."):
        mod = importlib.import_module(info.name)
        for name in dir(mod):
            obj = getattr(mod, name)
            if hasattr(obj, "__dataclass_fields__"):
                params = getattr(obj, "__dataclass_params__", None)
                if params is not None and not params.frozen:
                    raise AssertionError(f"{info.name}.{name} is a mutable dataclass")
