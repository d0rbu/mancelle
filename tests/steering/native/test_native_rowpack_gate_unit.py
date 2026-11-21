"""Unit tests for the ideal-math-relative gated row-packing path.

Two groups:

* ``test_rowpack_gate_error_hierarchy`` — the typed-error contract.
* ``_native_rowpack_gate`` behaviour — the runtime self-verifying row-packing
  gate.  These use the conftest tiny gemma twins (CPU, no download), a tiny
  linear+ReLU ``_SAE`` (same shape contract as
  ``tests/steering/native/test_native_compile_gate_unit.py``'s), and control
  the packed-vs-unpacked divergence by monkeypatching
  ``native_model.capture_packed`` to a behaviour-preserving or CORRUPTED
  wrapper — so the resulting SAE-feature delta can be made to land
  deterministically INSIDE or OUTSIDE the ideal-math-relative gate (real-model
  correctness is the slow gate in test_native_integration.py).

The gate's pass/fail decision MUST be the IDENTICAL predicate
``compiled_passes_ideal_math_gate`` (the production gate calls this and
nothing else).  The unit tests pin both the predicate semantics directly AND
the gate's three-feature-set wiring (fp32 ref UNPACKED, model_dtype UNPACKED
baseline, model_dtype PACKED candidate).

Row-packing is the S2 twin of S1 ``_native_compile_gate``; this suite is its
exact analogue.
"""
from __future__ import annotations

import loguru
import pytest
import torch as t

import steering.extract_features as ef
from errors import BosOnlySegmentError, ExtractionError, NanoGPTError, RowPackGateError
from steering.extract_features import (
    RowpackGateProbe,
    _build_rowpack_gate_probe,
    _native_rowpack_gate,
    _rowpack_gate_sparse_from_sae_sequences,
)
from steering.feature_paths import PoolingMode
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import _build_tiny_gemma2

# 0 <= 2 < TINY_LAYERS (4): a real interior truncation.
_LAYER = 2
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

# Ragged probe batch (lengths [5, 2, 6]) matching the A1/A2 packing tests;
# ids < TINY_VOCAB=64.
_PROBE_LISTS: list[list[int]] = [
    [3, 9, 1, 7, 2],     # L=5
    [5, 8],              # L=2
    [1, 4, 6, 0, 5, 2],  # L=6
]


def _gate_probe() -> RowpackGateProbe:
    """Small fixed-token row surface over _PROBE_LISTS."""
    return RowpackGateProbe(
        token_id_lists=_PROBE_LISTS,
        segments_by_sequence=[
            [(0, 2), (2, len(ids))] if len(ids) > 2 else [(0, len(ids))]
            for ids in _PROBE_LISTS
        ],
        keep_masks_by_sequence=[None for _ in _PROBE_LISTS],
        context_independent_features=False,
    )

# SAE feature width (must match _SAE below)
_D_MODEL = 32
_F_FEATS = 48


# --------------------------------------------------------------------------- #
# Shared tiny SAE harness (mirrors test_native_compile_gate_unit.py exactly)  #
# --------------------------------------------------------------------------- #


class _SAE:
    """Tiny deterministic linear+ReLU SAE (frozen, fp32 or bf16).

    ``encode`` maps ``[*, d] -> [*, f]`` so the gate's packed/unpacked paths
    produce real, non-vacuous per-segment (ids, mags).  Deterministic seeded
    weights ⇒ a fixed activation perturbation maps to a fixed, reproducible
    SAE-feature delta.
    """

    def __init__(self, d: int = _D_MODEL, f: int = _F_FEATS) -> None:
        with t.random.fork_rng():
            t.manual_seed(1234)
            self._w = t.nn.Parameter(t.randn(d, f), requires_grad=False)
            self._b = t.nn.Parameter(t.randn(f), requires_grad=False)
        self.d, self.f = d, f

    def parameters(self):
        yield self._w
        yield self._b

    def to(self, dtype):
        self._w = t.nn.Parameter(self._w.to(dtype), requires_grad=False)
        self._b = t.nn.Parameter(self._b.to(dtype), requires_grad=False)
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x.float().matmul(self._w.float()) + self._b.float())


class _CastRecordingSAE(_SAE):
    """An ``_SAE`` that records every dtype passed to ``.to`` (so a test can
    pin the gate's fp32-then-back cast contract without monkeypatching the
    method — keeps ty clean)."""

    def __init__(self, d: int = _D_MODEL, f: int = _F_FEATS) -> None:
        super().__init__(d, f)
        self.cast_history: list[t.dtype] = []

    def to(self, dtype):
        self.cast_history.append(dtype)
        return super().to(dtype)


class _RaiseOnFp32SAE(_SAE):
    """An ``_SAE`` whose ``.to(torch.float32)`` raises (every other dtype
    works) — pins the gate's "SAE cannot fp32-cast" failure mode."""

    def to(self, dtype):
        if dtype == t.float32:
            raise AttributeError("test: SAE class does not support .to()")
        return super().to(dtype)


class _EmptyRefSAE:
    """An SAE that emits a zero post-ReLU feature vector ⇒
    ``_gate_sparse_features`` yields EMPTY (ids, mags) per segment ⇒ the
    gate's non-vacuity guard rejects.

    Used to certify the "reference produced no features → cannot decide" path
    raises a clear ``RowPackGateError`` instead of falling through to a garbage
    predicate decision.
    """
    d, f = _D_MODEL, _F_FEATS

    def parameters(self):
        return iter([])

    def to(self, dtype):
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.zeros((x.shape[0], self.f))


# --------------------------------------------------------------------------- #
# Fresh collector + fp32-ref stub helpers                                      #
# --------------------------------------------------------------------------- #


def _fresh_collector() -> ef.NativeCollector:
    """A fresh truncated gemma-2 collector (mutable build ⇒ fresh twin)."""
    nc = ef.NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER
    )
    assert nc is not None
    return nc


def _patch_build_ref(monkeypatch) -> list[dict]:
    """Patch ``ef.build_native_model`` (which the gate calls for the fp32
    reference build) to return a fresh tiny gemma-2 truncated collector — no
    HF download, deterministic, byte-identical weights to the test's
    ``_fresh_collector`` baseline.  Records each call's args so tests can pin
    the contract.

    Returns the per-call recorder list.
    """
    calls: list[dict] = []

    def _fake_build(model_name, device, model_dtype, layer):
        calls.append({
            "model_name": model_name,
            "device": device,
            "model_dtype": model_dtype,
            "layer": layer,
        })
        return ef.NativeCollector.truncated(
            _build_tiny_gemma2(), model_name=model_name, layer=layer
        )

    monkeypatch.setattr(ef, "build_native_model", _fake_build)
    return calls


class _FakeTokenizer:
    bos_token_id = 0

    def decode(self, token_ids) -> str:
        return "".join(chr(65 + int(tok) % 26) for tok in token_ids)


def _patch_probe_tokenizer(monkeypatch) -> None:
    fake_tokenizer = _FakeTokenizer()
    docs = [
        [0, 10, 11, 12, 13],
        [20, 21, 0, 22, 23],
    ]
    monkeypatch.setattr(ef, "resolve_tokenizer", lambda _model_name: fake_tokenizer)
    monkeypatch.setattr(
        ef, "_tokenize_native_gate_probe_docs", lambda _tokenizer: docs
    )


def _assert_sparse_equal(
    a_ids: list[t.Tensor],
    a_mags: list[t.Tensor],
    b_ids: list[t.Tensor],
    b_mags: list[t.Tensor],
) -> None:
    assert len(a_ids) == len(b_ids) == len(a_mags) == len(b_mags)
    for row, (aid, amg, bid, bmg) in enumerate(
        zip(a_ids, a_mags, b_ids, b_mags, strict=True)
    ):
        assert t.equal(aid, bid), f"row {row}: ids {aid} != {bid}"
        assert t.allclose(amg, bmg, atol=0, rtol=0), (
            f"row {row}: mags {amg} != {bmg}"
        )


# --------------------------------------------------------------------------- #
# Test 0: error hierarchy                                                      #
# --------------------------------------------------------------------------- #


def test_rowpack_gate_error_hierarchy():
    """``RowPackGateError`` inherits from both ``ExtractionError`` and
    ``NanoGPTError``, and ``ExtractionError`` is NOT a subclass of
    ``RowPackGateError``."""
    assert issubclass(RowPackGateError, ExtractionError)
    assert issubclass(RowPackGateError, NanoGPTError)
    assert not issubclass(ExtractionError, RowPackGateError)


def test_rowpack_gate_probe_context_independent_bos_injection_and_masks(
    monkeypatch,
) -> None:
    _patch_probe_tokenizer(monkeypatch)

    probe = _build_rowpack_gate_probe(
        "fake-model",
        segment_size=2,
        context_independent_features=True,
        ignore_bos_features=True,
    )

    assert probe.context_independent_features is True
    assert probe.token_id_lists == [
        [0, 10],
        [0, 11, 12],
        [0, 13],
        [0, 20, 21],
        [0, 22],
        [0, 23],
    ]
    assert probe.segments_by_sequence == [
        [(0, 2)],  # natural BOS: pooled span includes it, mask drops it
        [(1, 3)],  # injected BOS: outside pooled span
        [(1, 2)],
        [(1, 3)],
        [(0, 2)],  # natural mid-doc BOS: no injection, mask drops it
        [(1, 2)],
    ]
    assert probe.keep_masks_by_sequence == [
        (False, True),
        (False, True, True),
        (False, True),
        (False, True, True),
        (False, True),
        (False, True),
    ]
    assert [
        (len(ids), segments[0], len(mask) if mask is not None else None)
        for ids, segments, mask in zip(
            probe.token_id_lists,
            probe.segments_by_sequence,
            probe.keep_masks_by_sequence,
            strict=True,
        )
    ] == [
        (2, (0, 2), 2),
        (3, (1, 3), 3),
        (2, (1, 2), 2),
        (3, (1, 3), 3),
        (2, (0, 2), 2),
        (2, (1, 2), 2),
    ]


def test_rowpack_gate_probe_context_dependent_fixed_segments_and_masks(
    monkeypatch,
) -> None:
    _patch_probe_tokenizer(monkeypatch)

    probe = _build_rowpack_gate_probe(
        "fake-model",
        segment_size=2,
        context_independent_features=False,
        ignore_bos_features=True,
    )

    assert probe.context_independent_features is False
    assert probe.token_id_lists == [
        [0, 10, 11, 12, 13],
        [20, 21, 0, 22, 23],
    ]
    assert probe.segments_by_sequence == [
        [(0, 2), (2, 4), (4, 5)],
        [(0, 2), (2, 4), (4, 5)],
    ]
    assert probe.keep_masks_by_sequence == [
        (False, True, True, True, True),
        (True, True, False, True, True),
    ]


def test_rowpack_projection_context_independent_masks_bos_feature() -> None:
    acts = [
        t.tensor([
            [1000.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
        ])
    ]
    probe = RowpackGateProbe(
        token_id_lists=[[0, 5, 6]],
        segments_by_sequence=[[(0, 3)]],
        keep_masks_by_sequence=[(False, True, True)],
        context_independent_features=True,
    )

    ids, mags = _rowpack_gate_sparse_from_sae_sequences(
        acts, probe, PoolingMode.max_pool
    )

    assert ids[0].tolist() == [2, 1]
    assert mags[0].tolist() == [4.0, 3.0]


@pytest.mark.parametrize("pooling", list(PoolingMode))
def test_rowpack_projection_context_dependent_matches_production_pooling(
    pooling: PoolingMode,
) -> None:
    acts = t.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [1000.0, 0.0, 8.0],
        [4.0, 5.0, 6.0],
        [7.0, 0.0, 9.0],
    ])
    segments = [(0, 2), (2, 5)]
    keep_mask = t.tensor([True, True, False, True, True])
    probe = RowpackGateProbe(
        token_id_lists=[[1, 2, 0, 3, 4]],
        segments_by_sequence=[segments],
        keep_masks_by_sequence=[tuple(keep_mask.tolist())],
        context_independent_features=False,
    )

    ids, mags = _rowpack_gate_sparse_from_sae_sequences(
        [acts], probe, pooling
    )
    expected = ef.extract_sparse_gpu(
        ef.pool_segments_batched_gpu(
            acts,
            segments,
            pooling,
            ef.SegmentMode.fixed_tokens,
            keep_mask=keep_mask,
        )
    )
    _assert_sparse_equal(ids, mags, expected[0], expected[1])


def test_rowpack_projection_all_true_mask_is_noop_byte_equivalent() -> None:
    acts = [t.tensor([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
        [5.0, 6.0, 7.0],
    ])]
    no_mask = RowpackGateProbe(
        token_id_lists=[[1, 2, 3]],
        segments_by_sequence=[[(0, 3)]],
        keep_masks_by_sequence=[None],
        context_independent_features=False,
    )
    all_true = RowpackGateProbe(
        token_id_lists=[[1, 2, 3]],
        segments_by_sequence=[[(0, 3)]],
        keep_masks_by_sequence=[(True, True, True)],
        context_independent_features=False,
    )

    ids_a, mags_a = _rowpack_gate_sparse_from_sae_sequences(
        acts, no_mask, PoolingMode.mean_pool
    )
    ids_b, mags_b = _rowpack_gate_sparse_from_sae_sequences(
        acts, all_true, PoolingMode.mean_pool
    )

    _assert_sparse_equal(ids_a, mags_a, ids_b, mags_b)


def test_rowpack_projection_all_false_segment_mask_raises_bos_only() -> None:
    probe = RowpackGateProbe(
        token_id_lists=[[0, 0]],
        segments_by_sequence=[[(0, 2)]],
        keep_masks_by_sequence=[(False, False)],
        context_independent_features=True,
    )

    with pytest.raises(BosOnlySegmentError):
        _rowpack_gate_sparse_from_sae_sequences(
            [t.ones(2, 3)], probe, PoolingMode.max_pool
        )


# --------------------------------------------------------------------------- #
# Test 1: PASS — real tiny model, real capture_packed, gate PASSES             #
# --------------------------------------------------------------------------- #


def test_gate_pass_verifies_rowpack_and_logs(monkeypatch, caplog):
    """Real ``capture_packed`` on a tiny gemma-2 is byte-identical to the
    padded path (A1/A2 proved ~2e-6 allclose) ⇒ the ideal-math-relative gate
    PASSES ⇒ ``native_model._rowpack_verified is True`` and a ``[rowpack-gate]``
    INFO line is emitted.

    Mutation-probe target: inverting the pass branch (set_rowpack_verified on
    FAIL) flips ``_rowpack_verified`` to False here ⇒ this test fails.
    """
    sae = _SAE()
    nc = _fresh_collector()
    _patch_build_ref(monkeypatch)

    sink: list[str] = []
    handler_id = loguru.logger.add(sink.append, level="INFO")
    try:
        with caplog.at_level("INFO"):
            _native_rowpack_gate(
                nc, sae,  # ty: ignore[invalid-argument-type]
                probe=_gate_probe(),
                pooling=PoolingMode.max_pool,
                model_dtype=t.float32,
            )
    finally:
        loguru.logger.remove(handler_id)

    assert nc._rowpack_verified is True, (
        "real capture_packed ≈ padded path ⇒ gate must PASS and set "
        "_rowpack_verified True"
    )
    text = "".join(sink)
    assert "[rowpack-gate]" in text, (
        f"pass must emit a tagged [rowpack-gate] line; got: {text!r}"
    )
    # All four predicate measurements plus row-pack diagnostics must appear.
    for key in (
        "j_baseline_vs_ref",
        "j_compiled_vs_ref",
        "slack",
        "margin",
        "j_packed_vs_unpacked",
        "count_rdiff_packed_vs_unpacked",
        "n_probe_sequences",
        "n_probe_rows",
        "max_probe_len",
        "pooling=max_pool",
        "context_independent_features=False",
        "ignore_bos_features=False",
    ):
        assert key in text, (
            f"the [rowpack-gate] line must carry {key!r}; got {text!r}"
        )


# --------------------------------------------------------------------------- #
# Test 2: FAIL — corrupted capture_packed ⇒ RowPackGateError                  #
# --------------------------------------------------------------------------- #


def test_gate_fail_raises_rowpack_gate_error(monkeypatch):
    """Monkeypatching ``capture_packed`` to return random activations makes the
    packed features diverge from the fp32-eager reference far more than the
    unpacked baseline does (Jaccard collapses ≪ baseline − slack) ⇒ the gate
    raises ``RowPackGateError`` carrying the four measurements, and
    ``_rowpack_verified is False`` after the raise.

    Mutation-probe verification: with the corruption REMOVED (i.e. capture_packed
    returns the real correct output), the gate PASSES and
    ``pytest.raises(RowPackGateError)`` would itself fail — confirmed manually
    by running the test body without the monkeypatch and observing no raise.
    """
    sae = _SAE()
    nc = _fresh_collector()
    _patch_build_ref(monkeypatch)

    # Override capture_packed with random activations that share nothing with
    # the real unpacked forward ⇒ Jaccard(packed, ref) ≈ 0, far below the
    # unpacked baseline ⇒ gate fails.
    def _corrupt_packed(
        docs, layer, *, hook_name=None, activation_type=ActivationType.residual
    ):
        return [t.randn(len(d), _D_MODEL) for d in docs]

    monkeypatch.setattr(nc, "capture_packed", _corrupt_packed)

    with pytest.raises(RowPackGateError) as excinfo:
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.float32,
        )

    msg = str(excinfo.value)
    for key in (
        "j_baseline_vs_ref",
        "j_compiled_vs_ref",
        "slack",
        "margin",
        "j_packed_vs_unpacked",
        "count_rdiff_packed_vs_unpacked",
        "n_probe_sequences",
        "n_probe_rows",
    ):
        assert key in msg, (
            f"RowPackGateError must carry {key!r}; got {msg!r}"
        )
    assert "cross-document leakage" not in msg
    assert nc._rowpack_verified is False, (
        "after a gate-fail raise the collector must NOT be left "
        "rowpack-verified (no verified-but-wrong state)"
    )


def test_gate_failure_clears_stale_rowpack_verified_state(monkeypatch):
    """A rerun after a previous pass must not preserve stale verified state.

    Production builds a fresh collector and runs the row-pack gate once, but the
    gate itself should still be structurally safe if called on a collector whose
    ``_rowpack_verified`` flag was already true.
    """
    sae = _SAE()
    nc = _fresh_collector()
    nc.set_rowpack_verified(True)
    _patch_build_ref(monkeypatch)

    def _corrupt_packed(
        docs, layer, *, hook_name=None, activation_type=ActivationType.residual
    ):
        return [t.randn(len(d), _D_MODEL) for d in docs]

    monkeypatch.setattr(nc, "capture_packed", _corrupt_packed)

    with pytest.raises(RowPackGateError):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.float32,
        )

    assert nc._rowpack_verified is False


# --------------------------------------------------------------------------- #
# Test 3: dtype-restore-on-throw — finally runs even when ref capture throws  #
# --------------------------------------------------------------------------- #


def test_gate_restores_sae_dtype_when_ref_capture_throws(monkeypatch):
    """If the fp32-reference ``capture()`` raises (OOM is the docstring-
    anticipated failure mode for the gate's most memory-heavy op), the
    production SAE — cast IN-PLACE to fp32 at gate start (``sae.to`` is
    in-place; ``ref_sae`` IS ``sae``) — MUST be restored to ``model_dtype``
    before the exception propagates.

    Mutation-probe: deleting the ``finally: sae.to(model_dtype)`` in
    ``_native_rowpack_gate``'s try/finally leaves ``cast_history == [float32]``
    (no restore) ⇒ this test FAILS on the last-cast assertion.
    """
    class _RefCaptureThrows:
        """Real tiny collector for everything EXCEPT ``capture()``, which
        raises — simulates the fp32-ref OOM after the SAE was cast to fp32."""

        def __init__(self, inner) -> None:
            object.__setattr__(self, "_inner", inner)

        def capture(self, *a, **k):
            raise RuntimeError(
                "CUDA out of memory (simulated fp32 ref capture OOM)"
            )

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

    def _fake_build(model_name, device, model_dtype, layer):
        inner = ef.NativeCollector.truncated(
            _build_tiny_gemma2(), model_name=model_name, layer=layer
        )
        return _RefCaptureThrows(inner)

    monkeypatch.setattr(ef, "build_native_model", _fake_build)
    sae = _CastRecordingSAE()
    nc = _fresh_collector()

    # The raw OOM propagates out of the gate (the capture failure is NOT
    # wrapped by the gate — only the build/SAE-cast failures are); the point
    # of THIS test is the dtype restoration on that propagation.
    with pytest.raises(RuntimeError, match="out of memory"):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.bfloat16,
        )

    assert sae.cast_history and sae.cast_history[-1] == t.bfloat16, (
        f"production SAE dtype was NOT restored after the fp32-ref capture "
        f"threw: cast_history={sae.cast_history} (the last cast MUST be "
        f"model_dtype=bfloat16 — the in-place fp32 cast leaked otherwise)"
    )


# --------------------------------------------------------------------------- #
# Test 4: fp32 reference build raises ⇒ RowPackGateError                      #
# --------------------------------------------------------------------------- #


def test_gate_fp32_ref_build_failure_raises_rowpack_gate_error(monkeypatch):
    """If ``build_native_model`` raises (e.g. OOM at load time), the gate
    must convert it to ``RowPackGateError`` with a clear message."""
    def _raise(*a, **k):
        raise RuntimeError("CUDA out of memory: tried to allocate 4 GB")

    monkeypatch.setattr(ef, "build_native_model", _raise)
    sae = _SAE()
    nc = _fresh_collector()

    with pytest.raises(
        RowPackGateError,
        match="rowpack gate FAILED to build fp32 reference",
    ):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.bfloat16,
        )


# --------------------------------------------------------------------------- #
# Test 5: fp32 reference returns None ⇒ RowPackGateError                      #
# --------------------------------------------------------------------------- #


def test_gate_fp32_ref_returns_none_raises_rowpack_gate_error(monkeypatch):
    """If ``build_native_model`` returns None (out-of-scope architecture),
    the gate must raise ``RowPackGateError`` with a clear message."""
    monkeypatch.setattr(ef, "build_native_model", lambda *a, **k: None)
    sae = _SAE()
    nc = _fresh_collector()

    with pytest.raises(
        RowPackGateError,
        match="build_native_model returned None",
    ):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.bfloat16,
        )


# --------------------------------------------------------------------------- #
# Test 6: SAE .to(fp32) failure ⇒ RowPackGateError                            #
# --------------------------------------------------------------------------- #


def test_gate_sae_fp32_cast_failure_raises_rowpack_gate_error(monkeypatch):
    """If the SAE class does not support ``.to(torch.float32)`` (e.g. its
    ``.to`` raises), the gate must raise ``RowPackGateError`` with a clear
    message."""
    _patch_build_ref(monkeypatch)
    sae = _RaiseOnFp32SAE()
    nc = _fresh_collector()

    with pytest.raises(
        RowPackGateError,
        match="SAE class .* does not support .to.torch.float32.",
    ):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.bfloat16,
        )


# --------------------------------------------------------------------------- #
# Test 7: fp32 ref produces no features ⇒ RowPackGateError (non-vacuity)      #
# --------------------------------------------------------------------------- #


def test_gate_empty_ref_features_raises_rowpack_gate_error(monkeypatch):
    """If the fp32-eager reference produces NO features (every segment empty),
    the gate must raise ``RowPackGateError`` (cannot decide without a
    non-empty reference) — never silently fall through to a vacuous predicate
    decision."""
    _patch_build_ref(monkeypatch)
    sae = _EmptyRefSAE()
    nc = _fresh_collector()

    with pytest.raises(
        RowPackGateError,
        match="fp32-eager reference produced no features",
    ):
        _native_rowpack_gate(
            nc, sae,  # ty: ignore[invalid-argument-type]
            probe=_gate_probe(),
            pooling=PoolingMode.max_pool,
            model_dtype=t.float32,
        )


# --------------------------------------------------------------------------- #
# Test 8: build_native_model called exactly once for the fp32 ref             #
# --------------------------------------------------------------------------- #


def test_gate_builds_fp32_reference_exactly_once_with_correct_args(monkeypatch):
    """The gate calls ``build_native_model`` EXACTLY once for the ideal-math
    reference with ``model_dtype=t.float32`` (regardless of the production
    dtype passed to the gate).  Pins the dtype-awareness of the gate: the
    reference is ALWAYS fp32 (ideal-math) even when the production cell is
    bf16.

    Mutation-probe target: in ``_native_rowpack_gate``, skipping the fp32
    reference build (using baseline as ref) makes ``calls`` empty here ⇒
    this test fails.
    """
    sae = _SAE()
    nc = _fresh_collector()
    calls = _patch_build_ref(monkeypatch)

    _native_rowpack_gate(
        nc, sae,  # ty: ignore[invalid-argument-type]
        probe=_gate_probe(),
        pooling=PoolingMode.max_pool,
        model_dtype=t.bfloat16,  # production dtype is bf16 but ref must be fp32
    )

    assert len(calls) == 1, (
        f"the gate must build the fp32 reference EXACTLY once, got "
        f"{len(calls)} calls: {calls}"
    )
    call = calls[0]
    assert call["model_dtype"] is t.float32, (
        f"the ideal-math reference MUST be fp32 regardless of the production "
        f"dtype; got model_dtype={call['model_dtype']}"
    )
    assert call["model_name"] == nc.model_name, (
        f"reference build must use the same model_name as the collector "
        f"({nc.model_name}); got {call['model_name']}"
    )
    assert call["layer"] == nc.truncated_layer, (
        f"reference build must use the same truncated_layer; "
        f"got layer={call['layer']} vs nc.truncated_layer={nc.truncated_layer}"
    )
