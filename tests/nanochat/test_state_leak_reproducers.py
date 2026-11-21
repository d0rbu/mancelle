"""Runtime reproducers for two test-order state leaks.

Each reproducer captures a specific way the leak manifested. Tests 1-2 use
fresh subprocesses to measure import-time state mutations cleanly (in-process
testing would conflict with whatever pytest already loaded). Tests 3-4 use
nested pytest invocations to reproduce the user-reported batch failures
exactly. Test 5 is a direct unit test of the autouse fixture used in
tests/nanochat/test_encdec_stress.py.

See docs/superpowers/specs/2026-04-28-test-order-state-leaks-design.md.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import nanochat.flash_attention as _fa
from nanochat.flash_attention import FlashAttnImpl


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Subprocess-based import reproducers
# ---------------------------------------------------------------------------


def test_chat_web_import_does_not_mutate_tf32() -> None:
    """Importing scripts.chat.chat_web at module level must not enable TF32.

    Pre-fix: chat_web ran compute_init() at module top, which called
    torch.set_float32_matmul_precision('high') (= TF32). Any test importing
    the module would see TF32 globally enabled, breaking
    test_cross_attention's CPU/CUDA SDPA comparison (1e-4 threshold vs
    1e-3 TF32 precision).
    """
    code = textwrap.dedent('''
        import torch
        before_precision = torch.get_float32_matmul_precision()
        before_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        import scripts.chat.chat_web  # noqa: F401
        after_precision = torch.get_float32_matmul_precision()
        after_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if before_precision != after_precision:
            raise SystemExit(
                f"matmul precision mutated: {before_precision!r} -> {after_precision!r}"
            )
        if before_allow_tf32 != after_allow_tf32:
            raise SystemExit(
                f"allow_tf32 mutated: {before_allow_tf32} -> {after_allow_tf32}"
            )
    ''')
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"chat_web import mutated TF32 state.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_encdec_stress_import_does_not_mutate_override_impl() -> None:
    """Importing tests.nanochat.test_encdec_stress must not mutate
    nanochat.flash_attention._override_impl at module load time.

    Pre-fix: the file's line 16 set _fa._override_impl = SDPA at module top.
    """
    code = textwrap.dedent('''
        import nanochat.flash_attention as fa
        before = fa._override_impl
        import tests.nanochat.test_encdec_stress  # noqa: F401
        after = fa._override_impl
        if before != after:
            raise SystemExit(
                f"_override_impl mutated by import: {before!r} -> {after!r}"
            )
    ''')
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"test_encdec_stress import mutated _override_impl.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Batch-composition reproducers (nested pytest)
# ---------------------------------------------------------------------------


def test_chat_web_does_not_break_cross_attention_in_batch() -> None:
    """Running test_chat_web_security alongside cross_attention's CPU/CUDA
    test must not break cross_attention.

    This is the minimal nested-pytest reproducer of the user-reported failure.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/scripts/test_chat_web_security.py",
        "tests/nanochat/test_cross_attention.py::TestCPUAndCUDAConsistency::test_cpu_and_cuda_produce_close_results",
        "--no-header",
        "-q",
        "--tb=short",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"Batch run failed; chat_web TF32 leak likely reintroduced.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_encdec_stress_passes_after_other_files_run() -> None:
    """Running test_encdec_stress.py's stress test alongside earlier files
    must not break it.

    Pre-fix: test_encdec_stress's module-top `_override_impl = SDPA` is set
    at collection time. Earlier files (test_attention_fallback's autouse
    reset; test_encdec_lr_scaling_integration's autouse save/restore landing
    on the prior None) leave `_override_impl = None` before the stress test
    runs. `_use_flex()` then returns True and flex_attention fails to
    compile on the tiny stress tensors.

    Empirically the smallest reproducer is the 3-file combination of
    test_attention_fallback.py + test_encdec_lr_scaling_integration.py +
    the targeted stress test. The 2-file combination
    (test_attention_fallback + stress) does NOT reproduce; an additional
    file that runs flex_attention through torch.compile is needed.
    """
    # Precondition: this reproducer empirically depends on
    # test_encdec_lr_scaling_integration.py existing and exercising
    # flex_attention through torch.compile. If that file is renamed or
    # removed, find another file with the same property and update this
    # reproducer; otherwise the test could pass for the wrong reason.
    lr_scaling_path = REPO_ROOT / "tests/nanochat/test_encdec_lr_scaling_integration.py"
    assert lr_scaling_path.exists(), (
        f"This reproducer depends on {lr_scaling_path} existing. If you "
        f"renamed/removed it, replace it here with another file that runs "
        f"flex_attention through torch.compile during pytest collection."
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/nanochat/test_attention_fallback.py",
        "tests/nanochat/test_encdec_lr_scaling_integration.py",
        "tests/nanochat/test_encdec_stress.py::TestExtremeInputValues::test_large_feature_ids_near_boundary",
        "--no-header",
        "-q",
        "--tb=short",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"Batch run failed; test_encdec_stress _override_impl leak likely "
        f"reintroduced.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Positive control: each fixed file must still pass when run alone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_file",
    [
        "tests/nanochat/test_encdec_stress.py",
        "tests/nanochat/test_torch_compile_correctness.py",
    ],
    ids=lambda p: Path(p).name,
)
def test_fixed_files_pass_in_isolation(test_file: str) -> None:
    """Positive control: each fixed test file passes when run alone.

    The original bug only manifested in batch — these files passed in
    isolation. After the fix they must still pass in isolation. This pins
    the standalone-pass invariant so a future refactor that breaks the
    file's solo behavior can't slip past CI.

    test_autocast_canonical.py is omitted because it has unrelated
    stale-script-path failures (out of scope for this PR).
    """
    cmd = [sys.executable, "-m", "pytest", test_file, "--no-header", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"{test_file} fails in isolation — fix may have broken its "
        f"standalone behavior.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Direct unit test of the encdec_stress autouse fixture's contract
# ---------------------------------------------------------------------------


class TestForceSdpaFixtureSemantics:
    """Logic-level test of the save → set → yield → restore pattern.

    Tests the LOGIC of save/restore, not the WIRING. See
    `TestForceSdpaFixtureWiring` below for tests that verify each fixed
    file's `_force_sdpa` is actually decorated as an autouse,
    function-scoped pytest fixture and applies to its tests.
    """

    @pytest.mark.parametrize(
        "prior",
        [None, FlashAttnImpl.SDPA, FlashAttnImpl.FA4],
        ids=["prior_none", "prior_sdpa", "prior_fa4"],
    )
    def test_save_set_yield_restore_under_each_prior(self, prior, monkeypatch) -> None:
        monkeypatch.setattr(_fa, "_override_impl", prior, raising=False)
        # Inline simulation of the fixture body
        saved = _fa._override_impl
        _fa._override_impl = FlashAttnImpl.SDPA
        # Test body would run here; check the override was applied
        assert _fa._override_impl is FlashAttnImpl.SDPA
        # Restore (simulating the post-yield half of the fixture)
        _fa._override_impl = saved
        assert _fa._override_impl == prior, (
            f"restore failed: expected {prior!r}, got {_fa._override_impl!r}"
        )


# ---------------------------------------------------------------------------
# Fixture WIRING tests — verify each fixed file's _force_sdpa is correctly
# registered as an autouse, function-scoped fixture and is actually applied
# to a real (class-method) test in the file.
# ---------------------------------------------------------------------------


_FIXED_FILES = [
    "tests/nanochat/test_encdec_stress.py",
    "tests/nanochat/test_autocast_canonical.py",
    "tests/nanochat/test_torch_compile_correctness.py",
]


@pytest.mark.parametrize("module_path", _FIXED_FILES, ids=lambda p: Path(p).name)
def test_force_sdpa_is_autouse_function_scoped(module_path: str) -> None:
    """Each fixed file must have a `_force_sdpa` fixture decorated with
    `@pytest.fixture(autouse=True)` at default (function) scope.

    Spawns a fresh Python interpreter, imports the file as a module, and
    inspects the fixture's `_pytestfixturefunction` marker. Catches
    regressions where someone changes `autouse=True` to False, switches
    scope to module/session, or removes the fixture decorator entirely —
    the inline `TestForceSdpaFixtureSemantics` test would not catch any
    of these because it never runs the real fixture.
    """
    # `module_path` like "tests/nanochat/test_encdec_stress.py" -> dotted
    # module name "tests.nanochat.test_encdec_stress".
    module_name = module_path.replace(".py", "").replace("/", ".")
    code = textwrap.dedent(f'''
        import importlib
        import inspect
        mod = importlib.import_module({module_name!r})
        fixture = getattr(mod, "_force_sdpa", None)
        if fixture is None:
            raise SystemExit("module has no _force_sdpa attribute")
        # pytest 8.x stores the marker on `_fixture_function_marker`. The
        # underlying function (for source inspection) is on `_fixture_function`.
        marker = getattr(fixture, "_fixture_function_marker", None)
        if marker is None:
            raise SystemExit("_force_sdpa is not decorated with @pytest.fixture")
        if not marker.autouse:
            raise SystemExit("_force_sdpa is not autouse=True")
        if marker.scope != "function":
            raise SystemExit(
                f"_force_sdpa scope is {{marker.scope!r}}, expected 'function'"
            )
        # Verify the body really does save/restore (not reset to None).
        inner = getattr(fixture, "_fixture_function", None)
        if inner is None:
            raise SystemExit("could not retrieve fixture wrapped function")
        src = inspect.getsource(inner)
        if "saved" not in src or "yield" not in src:
            raise SystemExit(
                "_force_sdpa is missing the save/yield/restore pattern; "
                "found source: " + src
            )
    ''')
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Fixture wiring check failed for {module_path}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "test_id",
    [
        "tests/nanochat/test_encdec_stress.py::TestExtremeInputValues::test_large_feature_ids_near_boundary",
        "tests/nanochat/test_torch_compile_correctness.py::TestCompileCorrectnessGPT::test_compiled_backward_no_nan",
    ],
    ids=lambda t: t.split("::", 1)[1],
)
def test_force_sdpa_actually_applies_to_class_method_test(test_id: str) -> None:
    """Verify `_force_sdpa` actually fires for class-method tests.

    Most of the fixed tests are class methods (TestExtremeInputValues etc.).
    If the fixture were declared in a way that only applied to module-level
    tests, all class-method tests would silently skip the fixture and
    re-introduce the leak.

    Uses pytest's `--setup-only` mode, which prints the fixture chain for
    each test without running test bodies. We assert `_force_sdpa` appears
    in the chain.
    """
    cmd = [
        sys.executable, "-m", "pytest", test_id,
        "--setup-only", "--no-header", "-q",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"--setup-only inspection failed for {test_id}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "_force_sdpa" in result.stdout, (
        f"_force_sdpa autouse fixture did not apply to {test_id}.\n"
        f"Setup output:\n{result.stdout}"
    )
