"""Shared equivalence-test infrastructure for feature-GCG performance tasks (T3-T6).

Provides:

``differential(reference_fn, candidate_fn, *, tensors_eq)``
    Generic oracle: calls both callables, asserts every extracted tensor
    satisfies ``tensors_eq``.  On failure raises ``AssertionError`` with the
    first differing flat index and the ref/cand values at that position.

``feature_grads_equal(a, b, *, exact, atol)``
    Field-by-field ``FeatureGrad`` comparison.  ``exact=True`` (default) uses
    ``torch.equal`` — raises on a 1-ULP difference.  ``exact=False`` uses
    ``torch.allclose(atol=atol, rtol=0.0)`` for the batching optimisation.

Hypothesis strategies
    ``bags()``         — random ``FeatureConditioning`` (incl. empty).
    ``allowed_sets(n_sae)`` — subsets of ``[0, n_sae)``.
    ``members(model, device)`` — real-shaped ``FeatureGradMember`` lists with
                                  varied completion lengths.

pytest fixtures
    ``tiny_model``   — function-scoped tiny ``EncoderDecoderGPT`` (CPU, float32).
    ``real_model``   — session-scoped d12/l8 checkpoint (skip-not-fail if absent).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn
from torch import Tensor

import nanochat.flash_attention as fa_module
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGrad, FeatureGradMember
from tests.steering.feature_gcg.functional.test_feature_gradient import (
    N_SAE,
    NUM_BUCKETS,
    _build_randomized_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Documented tolerance for the batching optimisation (Task 6)
# ─────────────────────────────────────────────────────────────────────────────

# Conservative near-exact default atol for the approximate (``exact=False``)
# branch of ``feature_grads_equal``.  1e-5 is intentionally TIGHTER than typical
# kernel reassociation drift, so it is a strict near-exact default — it is NOT
# the batching tolerance and NOT a safety margin over cuBLAS rounding.  The
# batching optimisation (Task 6) does NOT rely on this default: it derives its
# own, dtype-appropriate atol (e.g. ~5e-2 for GPU bf16 cuBLAS batch-shape
# rounding) and passes it explicitly.
_DEFAULT_APPROX_ATOL: float = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_tensors(obj: Any) -> list[tuple[str, Tensor]]:
    """Recursively extract ``(name, tensor)`` pairs from a result object.

    Handles:
    * A bare ``Tensor`` → one pair named ``"result"``.
    * A ``FeatureGrad`` → four pairs, one per field.
    * A Python scalar (``bool`` / ``int`` / ``float``) → one pair named
      ``"result"``, wrapped in a 0-d tensor.  This is what makes the **loss path**
      ACTUALLY compared: ``losses`` / ``reference_masked_loss`` / ``reference_g_steer``
      return ``float`` / ``list[float]``, so without this branch ``differential``
      on the loss path would extract ZERO tensors and pass VACUOUSLY (a 1.0-vs-999.0
      mismatch would slip through).  ``bool`` is matched before ``int`` (``bool`` is
      an ``int`` subclass) so it keeps its own dtype.
    * A ``list`` or ``tuple`` → recursed, with index-prefixed names (so a
      ``list[float]`` becomes one wrapped tensor per element).
    * Anything else → empty list.  A non-None payload that reduces to zero tensors
      is rejected loudly by :func:`differential` (it would otherwise be vacuous).
    """
    if isinstance(obj, Tensor):
        return [("result", obj)]
    if isinstance(obj, FeatureGrad):
        return [
            ("id_grad", obj.id_grad),
            ("bucket_grad", obj.bucket_grad),
            ("id_weight", obj.id_weight),
            ("bucket_weight", obj.bucket_weight),
        ]
    if isinstance(obj, bool):
        return [("result", torch.tensor(obj))]
    if isinstance(obj, (int, float)):
        return [("result", torch.tensor(obj))]
    if isinstance(obj, (list, tuple)):
        pairs: list[tuple[str, Tensor]] = []
        for i, item in enumerate(obj):
            for name, t in _extract_tensors(item):
                pairs.append((f"[{i}].{name}", t))
        return pairs
    return []


def _first_diff_index(ref: Tensor, cand: Tensor) -> int:
    """Return the flat index of the first element where ``ref`` and ``cand`` differ.

    Uses the **same NaN-as-unequal convention as ``torch.equal``**: a NaN
    compares unequal to everything, including another NaN at the same position,
    so a same-position NaN registers as a real differing index (it does NOT
    return -1).  This keeps the mismatch message coherent for the case where the
    callers' inequality predicate (e.g. ``torch.equal``) already fired *because*
    of a same-position NaN.  Returns -1 only when the tensors are genuinely
    element-wise identical (no NaNs).

    Both tensors are cast to float32 for comparison so element-wise ``!=`` works
    uniformly across all dtypes (bfloat16, int, etc.).
    """
    ref_flat = ref.detach().float().flatten()
    cand_flat = cand.detach().float().flatten()
    differs = ref_flat != cand_flat
    nonzero = differs.nonzero(as_tuple=False)
    if len(nonzero) == 0:
        return -1
    return int(nonzero[0].item())


# ─────────────────────────────────────────────────────────────────────────────
# Public API — differential & feature_grads_equal
# ─────────────────────────────────────────────────────────────────────────────


def differential(
    reference_fn: Callable[[], Any],
    candidate_fn: Callable[[], Any],
    *,
    tensors_eq: Callable[[Tensor, Tensor], bool] = torch.equal,
) -> None:
    """Assert every tensor returned by both callables satisfies ``tensors_eq``.

    Both functions are called with no arguments.  Tensors are extracted
    recursively from the return value (see :func:`_extract_tensors`).

    On failure raises ``AssertionError`` with:
    * the name of the first differing tensor (e.g. ``"id_grad"``),
    * the first differing **flat** index,
    * the reference and candidate values at that index.

    Parameters
    ----------
    reference_fn, candidate_fn:
        Zero-argument callables returning the same structure of tensors.
    tensors_eq:
        Predicate used to compare each tensor pair.  Default: ``torch.equal``
        (exact / byte-identical).  For the batching optimisation supply
        ``lambda a, b: torch.allclose(a, b, atol=..., rtol=0.0)``.
    """
    ref_out = reference_fn()
    cand_out = candidate_fn()

    ref_pairs = _extract_tensors(ref_out)
    cand_pairs = _extract_tensors(cand_out)

    # Vacuity guard: a NON-None result that reduces to ZERO tensors (a dict, a
    # string, an empty list, …) would make the comparison loop below run over
    # nothing and pass silently.  Scalars and lists thereof are now extractable
    # (see :func:`_extract_tensors`), so a non-None-but-empty extraction is an
    # unsupported / vacuous payload and must fail loudly.  ``None`` is exempt (an
    # explicit "no result" choice).
    for label, out, pairs in (
        ("reference", ref_out, ref_pairs),
        ("candidate", cand_out, cand_pairs),
    ):
        if out is not None and not pairs:
            raise AssertionError(
                f"differential: {label}_fn returned a {type(out).__name__} from "
                "which zero tensors were extracted — the comparison would be "
                "vacuous. Return a Tensor / FeatureGrad / float / int (or nested "
                "lists thereof), or None."
            )

    assert len(ref_pairs) == len(cand_pairs), (
        f"differential: reference returned {len(ref_pairs)} tensors, "
        f"candidate returned {len(cand_pairs)}"
    )

    for (ref_name, ref_t), (_, cand_t) in zip(ref_pairs, cand_pairs, strict=True):
        # Shape guard BEFORE the predicate: an ``allclose`` predicate BROADCASTS,
        # so a (10,) vs (10, 1) divergence would otherwise pass vacuously.  Checking
        # shape first makes the oracle reject a collapsed / transposed / broadcast
        # divergence under ANY predicate (not just ``torch.equal``).
        if ref_t.shape != cand_t.shape:
            raise AssertionError(
                f"differential: tensor {ref_name!r} shape mismatch: "
                f"ref={tuple(ref_t.shape)}, cand={tuple(cand_t.shape)}"
            )
        if not tensors_eq(ref_t, cand_t):
            idx = _first_diff_index(ref_t, cand_t)
            ref_flat = ref_t.detach().flatten()
            cand_flat = cand_t.detach().flatten()
            ref_val = ref_flat[idx].item() if idx >= 0 else float("nan")
            cand_val = cand_flat[idx].item() if idx >= 0 else float("nan")
            raise AssertionError(
                f"differential: tensor {ref_name!r} differs: "
                f"first differing flat index={idx}, "
                f"ref={ref_val!r}, cand={cand_val!r}; "
                f"shapes: ref={tuple(ref_t.shape)}, cand={tuple(cand_t.shape)}"
            )


def feature_grads_equal(
    a: FeatureGrad,
    b: FeatureGrad,
    *,
    exact: bool = True,
    atol: float = _DEFAULT_APPROX_ATOL,
) -> None:
    """Assert two ``FeatureGrad`` objects are equal in all four tensor fields.

    Parameters
    ----------
    exact:
        ``True`` (default) — ``torch.equal`` for byte-identical equality.
        Raises on a 1-ULP bf16 difference.
        ``False`` — ``torch.allclose(atol=atol, rtol=0.0)`` for approximate
        equality (used by the batching optimisation in Task 6).
    atol:
        Absolute tolerance used only when ``exact=False``.  Defaults to
        ``_DEFAULT_APPROX_ATOL = 1e-5`` — a conservative *near-exact* tolerance
        for approximate comparisons, intentionally TIGHTER than typical kernel
        reassociation drift.  This default is NOT the batching tolerance: the
        batching optimisation (Task 6) supplies its own derived,
        dtype-appropriate atol (e.g. ~5e-2 for GPU bf16 cuBLAS batch-shape
        rounding) rather than relying on this default.

    Raises
    ------
    AssertionError
        On a per-field SHAPE mismatch (both modes — checked before any value
        comparison so an ``allclose`` broadcast cannot mask a collapsed/transposed
        slot) or DTYPE mismatch (both modes — approx mode still compares values in
        float32, but the storage dtype must match), or on the first differing
        field, with the field name, first differing flat index, and the values at
        that index (exact mode) or the max absolute difference (allclose mode).
    """
    fields: list[tuple[str, Tensor, Tensor]] = [
        ("id_grad", a.id_grad, b.id_grad),
        ("bucket_grad", a.bucket_grad, b.bucket_grad),
        ("id_weight", a.id_weight, b.id_weight),
        ("bucket_weight", a.bucket_weight, b.bucket_weight),
    ]
    for field_name, ta, tb in fields:
        # Shape guard FIRST, in BOTH modes: ``torch.allclose`` (approx mode) and
        # even ``torch.equal`` over a ``.float()`` cast would BROADCAST a collapsed
        # / duplicated / transposed slot (e.g. (n,D) vs (1,D) or (D,n)) and pass
        # vacuously.  A FeatureGrad field whose shape changed is ALWAYS a real
        # divergence, so reject it before any value comparison.
        if ta.shape != tb.shape:
            raise AssertionError(
                f"feature_grads_equal: FeatureGrad.{field_name} shape mismatch: "
                f"a={tuple(ta.shape)}, b={tuple(tb.shape)}"
            )
        # Dtype guard in BOTH modes.  Exact mode must compare same-dtype tensors
        # (``torch.equal`` already requires it, but an explicit check gives a clear
        # message instead of an opaque "first differing index").  Approx mode
        # COMPARES values in float32 (the ``.float()`` cast below) so reduced-
        # precision (bf16) reassociation rounding is tolerated — but the STORAGE
        # dtype must still match, so a silent float32-vs-bf16 comparison is rejected
        # rather than masked by the cast.
        if ta.dtype != tb.dtype:
            raise AssertionError(
                f"feature_grads_equal: FeatureGrad.{field_name} dtype mismatch: "
                f"a={ta.dtype}, b={tb.dtype}"
            )
        if exact:
            if not torch.equal(ta, tb):
                idx = _first_diff_index(ta, tb)
                ta_flat = ta.detach().flatten()
                tb_flat = tb.detach().flatten()
                a_val = ta_flat[idx].item() if idx >= 0 else float("nan")
                b_val = tb_flat[idx].item() if idx >= 0 else float("nan")
                raise AssertionError(
                    f"feature_grads_equal: FeatureGrad.{field_name} differs "
                    f"(exact mode): first differing flat index={idx}, "
                    f"a={a_val!r}, b={b_val!r}; shape={tuple(ta.shape)}"
                )
        else:
            if not torch.allclose(ta.float(), tb.float(), atol=atol, rtol=0.0):
                max_diff = float((ta.float() - tb.float()).abs().max().item())
                raise AssertionError(
                    f"feature_grads_equal: FeatureGrad.{field_name} differs "
                    f"(allclose mode, atol={atol}): max_diff={max_diff:.3e}; "
                    f"shape={tuple(ta.shape)}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis strategies
# ─────────────────────────────────────────────────────────────────────────────


@st.composite
def bags(
    draw: DrawFn,
    n_sae: int = N_SAE,
    num_buckets: int = NUM_BUCKETS,
    max_size: int = 8,
) -> FeatureConditioning:
    """Strategy drawing random ``FeatureConditioning`` bags.

    The empty bag (size 0) is always reachable (``min_size=0``).  Non-empty
    bags contain unique feature-ids drawn from ``[0, n_sae)`` paired with
    buckets from ``[0, num_buckets)``.  Entries are canonically sorted by
    ``FeatureConditioning.__post_init__``.

    ``max_size`` is clamped to ``n_sae`` so a caller passing ``max_size > n_sae``
    gets a valid draw (at most ``n_sae`` unique fids exist) rather than an
    internal Hypothesis "cannot satisfy unique constraint" error.
    """
    max_size = min(max_size, n_sae)
    size = draw(st.integers(min_value=0, max_value=max_size))
    if size == 0:
        return FeatureConditioning(entries=())
    fids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae - 1),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    fids_sorted = sorted(fids)
    buckets = [
        draw(st.integers(min_value=0, max_value=num_buckets - 1))
        for _ in fids_sorted
    ]
    return FeatureConditioning(entries=tuple(zip(fids_sorted, buckets, strict=True)))


def allowed_sets(n_sae: int = N_SAE) -> st.SearchStrategy[frozenset[int]]:
    """Strategy drawing arbitrary subsets of ``[0, n_sae)``."""
    return st.frozensets(st.integers(min_value=0, max_value=n_sae - 1))


@st.composite
def members(
    draw: DrawFn,
    model: Any,
    device: torch.device | str,
    min_n: int = 1,
    max_n: int = 4,
    min_t: int = 4,
    max_t: int = 12,
) -> list[FeatureGradMember]:
    """Strategy drawing real-shaped ``FeatureGradMember`` lists.

    Follows ``_make_member``'s shape conventions:

    * ``dec_input`` / ``dec_target``: ``(1, T)`` token tensors, T varied.
    * ``seg_map``: ``(1, T)`` — first half attends segment 0, second half
      attends segment 1 (the generation slot).
    * ``generation_slot = 1`` — the optimised segment.
    * ``seg0`` uses ``(3+i, 1)`` and ``(9+i, 2)`` per-member to keep feature-ids
      globally distinct across members of one draw.

    The empty bag at ``generation_slot`` is a placeholder; callers (e.g.
    ``feature_conditioning_gradient_batched``) substitute the shared target bag.

    ``model`` is accepted as a parameter to allow future strategies that
    compute reference gradients per-member (e.g. for T3-T6 oracles); it is not
    used in the member construction itself.

    Each member draws its OWN seed (a fresh ``torch.Generator`` per member), so
    Hypothesis can shrink members independently and isolate the single failing
    member rather than shrinking all members in lockstep off one shared seed.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))

    result: list[FeatureGradMember] = []
    for i in range(n):
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        gen = torch.Generator().manual_seed(seed)
        t_len = draw(st.integers(min_value=min_t, max_value=max_t))
        dec_input = torch.randint(1, 100, (1, t_len), generator=gen, device=device)
        dec_target = torch.randint(0, 100, (1, t_len), generator=gen, device=device)
        # First half attends segment 0 (context), second half attends segment 1
        # (generation slot) — same convention as _make_member.
        seg_map = torch.zeros((1, t_len), dtype=torch.long, device=device)
        seg_map[0, t_len // 2 :] = 1
        # Each member gets distinct feature IDs (offset by member index).
        seg0 = FeatureConditioning(entries=((3 + i, 1), (9 + i, 2)))
        result.append(
            FeatureGradMember(
                dec_input=dec_input,
                dec_target=dec_target,
                segment_conditions=[seg0, FeatureConditioning(entries=())],
                seg_map=seg_map,
                generation_slot=1,
            )
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Tiny randomised ``EncoderDecoderGPT`` (CPU, float32, non-zero-init encoder).

    Identical construction to the FD oracle in
    ``tests/steering/feature_gcg/functional/test_feature_gradient.py``:

    * ``COMPUTE_DTYPE`` patched to ``float32`` in all forward-path modules
      (via ``_build_randomized_model``).
    * FlexAttention overridden to SDPA (required for CPU backward).
    * Encoder weights randomised to avoid the inert-toy trap.
    * Model is returned in ``eval()`` mode.

    Import this fixture into a test module to use it::

        from tests.steering.feature_gcg._equivalence import tiny_model  # noqa: F401
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    return _build_randomized_model(monkeypatch, seed=1234)


@pytest.fixture(scope="session")
def real_model() -> Any:
    """Real d12/l8 ``EncoderDecoderGPT`` loaded once per test session.

    Slow-gating (single choke point)
    --------------------------------
    Any test that requests this fixture is auto-marked ``slow`` by the
    ``pytest_collection_modifyitems`` hook in ``tests/steering/feature_gcg/conftest.py`` —
    so **no call site needs its own** ``@pytest.mark.slow``.  The mark cannot be
    placed on the fixture itself: pytest 9 makes "marks applied to fixtures" a
    hard collection error (``Marks applied to fixtures have no effect``), so the
    collection hook is the canonical, pytest-9-compatible realisation of the
    "fixture is the slow-gating choke point" mandate.

    Skip-not-fail
    -------------
    When the checkpoint is absent the fixture calls ``pytest.skip`` (not
    ``pytest.fail``), so the test is counted as skipped rather than failed.
    Enable Layer B tests on a GPU box via:

    .. code-block:: bash

        export FEATURE_GCG_D12_CKPT=/path/to/d12_encdec_checkpoint
        export FEATURE_GCG_D12_STEP=54600       # default
        pytest -m slow ...

    Usage — the slow mark is applied automatically::

        from tests.steering.feature_gcg._equivalence import real_model  # noqa: F401

        def test_something_real(real_model):  # auto-marked slow by the hook
            ...
    """
    ckpt_env = os.environ.get("FEATURE_GCG_D12_CKPT", "")
    if not ckpt_env:
        pytest.skip(
            "FEATURE_GCG_D12_CKPT env var is not set; "
            "point it to the d12 encdec checkpoint directory "
            "(looked for env var FEATURE_GCG_D12_CKPT) to enable real-model tests."
        )
    ckpt_dir = Path(ckpt_env)
    if not ckpt_dir.exists():
        pytest.skip(
            f"FEATURE_GCG_D12_CKPT={ckpt_dir!r} does not exist; "
            "skipping real-model tests."
        )
    step = int(os.environ.get("FEATURE_GCG_D12_STEP", "54600"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _tok, meta_data = build_encdec_model(
        checkpoint_dir=str(ckpt_dir),
        step=step,
        device=str(device),
        phase=ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.deferred("feature_gcg_equivalence"),
    )
    model._loaded_meta_data = meta_data  # ty: ignore[invalid-assignment]
    model.eval()
    return model
