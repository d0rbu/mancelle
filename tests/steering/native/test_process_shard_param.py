"""``process_shard``/``main`` native-dispatch structure pins.

``test_process_shard_accepts_native_model`` pins the threaded ``native_model``
parameter (unchanged).

``test_*_builds_native_only_under_backend_native`` originally pinned that
``main()``'s *inline* backend-selection block built the native model ONLY
under ``backend == Backend.native`` (mutual exclusivity: ``auto``/``nnterp``
build nnterp, not native; gated by the ``if backend == Backend.native`` arm
vs. the ``elif backend in (Backend.nnterp, Backend.auto)`` arm).

Contract migration (sub-project #4): that build-time policy was extracted out
of ``main()`` into the pure helper
``steering.extract_features._build_accelerated_backend``; ``main()`` now only
does ``nn_model, native_model = _build_accelerated_backend(...)``. The
mutual-exclusivity guarantee is UNCHANGED — preserved by the helper's early
returns (native built only for ``Backend.native`` + cuda; nnterp only for
``nnterp``/``auto`` + cuda). This test re-pins that SAME guarantee at the new
site, behaviorally.

Non-redundancy with ``test_accel_backend_failfast.py``: that file already
covers ``Backend.auto`` + cuda success ``-> (nnterp, None)`` and
``transformerlens -> (None, None)`` (native/nnterp not built). It does NOT
cover the *native side* of mutual exclusion — that ``Backend.native`` + cuda
success returns ``(None, <native>)`` while NOT building nnterp, and that
explicit ``Backend.nnterp`` builds nnterp while NOT building native. Those are
the precise guarantees the original ``main()``-structure test protected;
re-expressed here at the helper without duplicating the auto/TL cases.
"""

from __future__ import annotations

import inspect

import torch as t

import steering.extract_features as ef
from steering.extract_features import Backend


def test_process_shard_accepts_native_model():
    sig = inspect.signature(ef.process_shard)
    assert "native_model" in sig.parameters
    assert sig.parameters["native_model"].default is None


class _Cfg:
    def __init__(self, model: str = "gemma-2-2b", layer: int = 6) -> None:
        self.model = model
        self.layer = layer


def test_build_accelerated_backend_builds_native_only_under_backend_native(
    monkeypatch,
):
    """``Backend.native`` + cuda: the helper builds the NATIVE model (not
    nnterp) and returns ``(None, <native>)`` — the native side of the
    build-time mutual-exclusivity guarantee that ``main()``'s inline
    ``if backend == Backend.native`` arm used to enforce."""
    built: list[str] = []
    native_sentinel = object()

    def fake_build_native_model(*a, **k):
        built.append("native")
        return native_sentinel

    def fake_build_nnterp_model(*a, **k):
        built.append("nnterp")
        return object()

    monkeypatch.setattr(ef, "build_native_model", fake_build_native_model)
    monkeypatch.setattr(ef, "build_nnterp_model", fake_build_nnterp_model)

    nn_model, native_model = ef._build_accelerated_backend(
        Backend.native, t.nn.Linear(2, 2), "cuda:0", t.float32, _Cfg()
    )

    assert built == ["native"], (
        f"backend=native must build ONLY the native model, got {built}"
    )
    assert native_model is native_sentinel
    assert nn_model is None, "native run must NOT also build/return nnterp"


def test_build_accelerated_backend_offloads_tl_before_native_build(monkeypatch):
    """backend=native must free the already-loaded TL model before loading the
    separate native HF collector."""
    events: list[str] = []
    native_sentinel = object()

    class _Model:
        def to(self, device):
            events.append(f"model.to({device})")
            return self

    def fake_build_native_model(*a, **k):
        events.append("build_native")
        return native_sentinel

    monkeypatch.setattr(ef, "build_native_model", fake_build_native_model)
    monkeypatch.setattr(ef.t.cuda, "empty_cache", lambda: events.append("empty_cache"))

    nn_model, native_model = ef._build_accelerated_backend(
        Backend.native, _Model(), "cuda:0", t.float32, _Cfg()
    )

    assert events == ["model.to(cpu)", "empty_cache", "build_native"]
    assert native_model is native_sentinel
    assert nn_model is None


def test_build_accelerated_backend_explicit_nnterp_does_not_build_native(
    monkeypatch,
):
    """The mutual-exclusivity converse: explicit ``Backend.nnterp`` + cuda
    builds nnterp and returns ``(<nnterp>, None)`` and NEVER builds the native
    model (the ``elif backend in (Backend.nnterp, Backend.auto)`` arm in the
    old inline structure was a separate branch from the native ``if``)."""
    built: list[str] = []
    nnterp_sentinel = object()

    def fake_build_native_model(*a, **k):
        built.append("native")
        return object()

    def fake_build_nnterp_model(*a, **k):
        built.append("nnterp")
        return nnterp_sentinel

    monkeypatch.setattr(ef, "build_native_model", fake_build_native_model)
    monkeypatch.setattr(ef, "build_nnterp_model", fake_build_nnterp_model)

    nn_model, native_model = ef._build_accelerated_backend(
        Backend.nnterp, t.nn.Linear(2, 2), "cuda:0", t.float32, _Cfg()
    )

    assert built == ["nnterp"], (
        f"backend=nnterp must build ONLY nnterp, never native, got {built}"
    )
    assert nn_model is nnterp_sentinel
    assert native_model is None, "nnterp run must NOT build/return native"
