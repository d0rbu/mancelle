"""Wiring pin: the production call site must thread ``layer=cfg.layer`` into
``build_native_model(...)``.

``build_native_model`` gained a REQUIRED positional ``layer: int`` (T3 — the
structural-truncation build slices the decoder stack to ``[:layer+1]``). The
production call site did NOT pass it, so a real ``backend=native`` run raised
``TypeError: build_native_model() missing 1 required positional argument:
'layer'`` before any extraction. This pin guards that regression.

Contract migration (sub-project #4): the build-time backend policy was
extracted out of ``main()`` into the pure helper
``steering.extract_features._build_accelerated_backend``; the
``build_native_model(...)`` call now lives there (``main()`` only does
``nn_model, native_model = _build_accelerated_backend(backend, model, device,
model_dtype, cfg)``). The layer-threading guarantee is UNCHANGED and just as
critical — this test now pins it at the new site (the helper), preserving the
original protective intent (the truncated native model MUST be built at
exactly ``cfg.layer``).

``cfg.layer`` is the extraction layer: ``main()`` uses it for SAE resolution
(``resolve_sae(..., layer=cfg.layer, ...)``), FLOPs estimation, and
auto-batch — and it is the value forwarded into ``_build_accelerated_backend``
as ``cfg``. So the truncated native model MUST be built at exactly
``cfg.layer``.

Two complementary pins (statically AND behaviorally non-vacuous):

* AST/inspect on ``_build_accelerated_backend`` source: the
  ``build_native_model(...)`` call passes ``cfg.layer`` as ``layer`` (keyword
  or 4th positional). Mirrors the established repo convention in
  ``test_require_cuda_for_native_unit.py`` Part 2 /
  ``test_extract_features_l0_tier_wiring.py`` — pin the production call site
  to actually thread the argument, statically, without the heavy model
  prelude.
* Behavioral: monkeypatch ``ef.build_native_model`` to capture kwargs and
  drive ``_build_accelerated_backend(Backend.native, model, "cuda:0", dtype,
  _Cfg(layer=N))``; assert the captured ``layer == cfg.layer`` (proves the
  AST pin is not vacuous — the value really flows through at runtime).
"""

from __future__ import annotations

import ast
import inspect

import torch as t

import steering.extract_features as ef
from steering.extract_features import Backend


def _build_native_call() -> ast.Call:
    """The ``build_native_model(...)`` call expression inside
    ``_build_accelerated_backend()`` (the new home of the build-time policy).

    ``inspect.getsource(ef._build_accelerated_backend)`` -> ``ast.parse`` ->
    walk the function body for the ``ast.Call`` whose func is the
    ``build_native_model`` name. Uses the same ``isinstance(node,
    (ast.FunctionDef, ast.AsyncFunctionDef))`` narrowing as the R6 guard test
    so the AST access stays ty-clean.
    """
    src = inspect.getsource(ef._build_accelerated_backend)
    tree = ast.parse(src)
    func_def = tree.body[0]
    assert isinstance(
        func_def, (ast.FunctionDef, ast.AsyncFunctionDef)
    ), "ef._build_accelerated_backend is expected to be a function def"
    for node in ast.walk(func_def):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_native_model"
        ):
            return node
    raise AssertionError(
        "`build_native_model(...)` call not found in "
        "_build_accelerated_backend()"
    )


def _is_cfg_layer(value: ast.expr) -> bool:
    """True iff ``value`` is the attribute access ``cfg.layer``."""
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "layer"
        and isinstance(value.value, ast.Name)
        and value.value.id == "cfg"
    )


def test_build_accelerated_backend_threads_cfg_layer_into_build_native_model() -> None:
    """``_build_accelerated_backend()``'s ``build_native_model(...)`` call
    passes ``cfg.layer`` as the ``layer`` argument — either ``layer=cfg.layer``
    (keyword) or the 4th positional ``cfg.layer`` (matching
    ``build_native_model``'s signature ``(model_name, device, model_dtype,
    layer, *, ...)``). Guards the prior "layer not threaded" regression at the
    new (post-extraction) call site."""
    call = _build_native_call()

    kw_layer = next((kw for kw in call.keywords if kw.arg == "layer"), None)
    if kw_layer is not None:
        assert _is_cfg_layer(kw_layer.value), (
            "build_native_model(...) must be called with layer=cfg.layer "
            f"(the extraction layer), got layer={ast.dump(kw_layer.value)}"
        )
        return

    # No keyword `layer=` -> it must be supplied positionally as the 4th
    # argument (index 3), per the signature.
    assert len(call.args) >= 4, (
        "build_native_model(...) must thread the extraction layer; found "
        f"neither layer=... keyword nor a 4th positional (args={len(call.args)})"
    )
    assert _is_cfg_layer(call.args[3]), (
        "the 4th positional argument to build_native_model(...) must be "
        f"cfg.layer (the extraction layer), got {ast.dump(call.args[3])}"
    )


class _Cfg:
    def __init__(self, model: str = "gemma-2-2b", layer: int = 6) -> None:
        self.model = model
        self.layer = layer


def test_build_accelerated_backend_passes_runtime_cfg_layer_to_build_native_model(
    monkeypatch,
) -> None:
    """Behavioral (non-vacuity) complement to the AST pin: driving
    ``_build_accelerated_backend(Backend.native, model, "cuda:0", dtype,
    _Cfg(layer=N))`` actually forwards ``layer == cfg.layer`` into
    ``build_native_model`` at runtime (not just textually). Uses a distinctive
    non-default layer so a stray hard-coded constant would be caught."""
    captured: dict[str, object] = {}
    cfg = _Cfg(model="gemma-2-2b", layer=11)

    def fake_build_native_model(model_name, *, device, model_dtype, layer):
        captured["model_name"] = model_name
        captured["device"] = device
        captured["model_dtype"] = model_dtype
        captured["layer"] = layer
        return object()  # non-None -> helper returns (None, native_model)

    monkeypatch.setattr(ef, "build_native_model", fake_build_native_model)
    nn_model, native_model = ef._build_accelerated_backend(
        Backend.native, t.nn.Linear(2, 2), "cuda:0", t.float32, cfg
    )

    assert captured["layer"] == cfg.layer == 11, (
        "build_native_model must receive layer=cfg.layer (the extraction "
        f"layer); captured {captured.get('layer')!r}, expected {cfg.layer!r}"
    )
    assert captured["model_name"] == cfg.model
    # Sanity: native success path returns (None, <native>) — mutual exclusion
    # with nnterp (also pinned in test_process_shard_param.py).
    assert nn_model is None and native_model is not None
