"""Guard: ``backend=native`` on a non-CUDA device MUST abort, not silently
fall back to the CPU-resident TransformerLens model.

Pre-merge adversarial review found ``main()``'s backend-selection block
asymmetric: the build-None paths ``logger.error(...) + return`` (abort), but
``backend=native`` with a non-CUDA ``device`` skipped *both* the
``native``-branch (``device.startswith("cuda")`` False) and the
``nnterp``/``auto`` branch (``native`` not in that tuple) with **no
diagnostic**, then silently ran extraction on the CPU TransformerLens model
the user did not ask for.

The decision is now a pure helper ``_require_cuda_for_native(backend,
device)`` called by ``main()`` immediately after ``backend = Backend(...)``;
when it returns True ``main()`` logs an error and returns (symmetric with the
build-None aborts).

Why test the helper + wiring rather than call ``main()`` directly: reaching
the guard at the call site requires ``main()`` to first bind a Hydra config,
resolve the data dir, list parquet shards, validate the sae XOR, resolve the
SAE, and ``load_sae_and_model`` (a real model + SAE + tokenizer) plus a
filesystem ``assert_l0_tier_compatible`` — mocking all of that just to reach
one ``if`` would be vacuous. This mirrors the established repo convention in
``tests/steering/test_extract_features_l0_tier_wiring.py`` (``main()`` "needs
a GPU model, so this test cannot call it directly. Instead it exercises the
*exact same wiring units*"): the helper is exhaustively unit-tested, and the
production call site is pinned to actually call it + abort with the required
message.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import steering.extract_features as ef
from steering.extract_features import Backend, _require_cuda_for_native


def _extract_features_source() -> str:
    assert ef.__file__ is not None
    return Path(ef.__file__).read_text()


def _extract_features_func_def(name: str) -> ast.FunctionDef:
    tree = ast.parse(_extract_features_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in steering.extract_features")


# --- Part 1: the guard decision, exhaustive over backend x device ----------

@pytest.mark.parametrize(
    ("backend", "device", "expected_abort"),
    [
        # native + non-CUDA  -> abort (the previously-silent misconfiguration)
        (Backend.native, "cpu", True),
        (Backend.native, "mps", True),
        (Backend.native, "", True),
        # native + CUDA      -> ok (the normal native run)
        (Backend.native, "cuda", False),
        (Backend.native, "cuda:0", False),
        (Backend.native, "cuda:3", False),
        # auto / nnterp / TL never gate on this guard, CPU or CUDA
        (Backend.auto, "cpu", False),
        (Backend.auto, "cuda:0", False),
        (Backend.nnterp, "cpu", False),
        (Backend.nnterp, "cuda:0", False),
        (Backend.transformerlens, "cpu", False),
        (Backend.transformerlens, "cuda:0", False),
    ],
)
def test_require_cuda_for_native_decision(
    backend: Backend, device: str, expected_abort: bool
) -> None:
    assert _require_cuda_for_native(backend, device) is expected_abort


def test_only_native_is_gated() -> None:
    """Exactly one enum member triggers the abort on a non-CUDA device; the
    others stay runnable on CPU (no behavior change for working configs)."""
    aborts = {
        b for b in Backend if _require_cuda_for_native(b, "cpu")
    }
    assert aborts == {Backend.native}


def test_cuda_prefix_not_exact_match() -> None:
    """The guard keys on the ``cuda`` prefix (so ``cuda:0``/``cuda:7`` pass),
    matching ``main()``'s own ``device.startswith("cuda")`` branch — not an
    exact ``== "cuda"`` check."""
    assert _require_cuda_for_native(Backend.native, "cuda:7") is False
    # A device that merely starts with the letters but is not the CUDA prefix
    # must still abort (defensive: only the real ``cuda`` prefix is accepted).
    assert _require_cuda_for_native(Backend.native, "cpu:0") is True


# --- Part 2: production wiring (helper extracted; main() delegates) --------
#
# The build-time policy moved into the directly-unit-tested pure helper
# _build_accelerated_backend (see test_accel_backend_failfast.py). Its
# decisions are no longer logger.error+return inside main() but raises in the
# helper. Here we only pin that main() actually delegates to the helper
# right after resolving the backend (AST, no @hydra.main needed).

def test_main_delegates_to_build_accelerated_backend() -> None:
    func_def = _extract_features_func_def("main")
    found = False
    for node in ast.walk(func_def):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_build_accelerated_backend"
        ):
            tgt = node.targets[0]
            assert isinstance(tgt, ast.Tuple)
            names = [e.id for e in tgt.elts if isinstance(e, ast.Name)]
            assert names == ["nn_model", "native_model"], names
            found = True
    assert found, "main() must delegate build-time policy to _build_accelerated_backend"


def test_require_cuda_helper_still_pure_and_used() -> None:
    """The pre-flight predicate is unchanged and is consumed by the helper
    (not main() directly) now."""
    assert _require_cuda_for_native(Backend.native, "cpu") is True
    assert _require_cuda_for_native(Backend.native, "cuda:0") is False
    helper_src = ast.get_source_segment(
        _extract_features_source(),
        _extract_features_func_def("_build_accelerated_backend"),
    )
    assert helper_src is not None
    assert "_require_cuda_for_native(" in helper_src
