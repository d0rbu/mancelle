"""S2 Task A1 — `NativeCollector.capture_packed` unit gates (tiny CPU twins).

Pins the packing core (A1 in the S2 design spec §3-A1):
  1. Packed ≈ unpacked per-doc equivalence (gemma-2 AND gemma-3).
  2. use_cache=False enforced + non-vacuous teeth-guard (force True → cross-doc
     leakage → at least one doc ≥1 diverges from the padded reference).
  3. Length-splitting shape contract (B=1, single-token doc, ragged mix).
  4. Typed-error contract (wrong activation_type / wrong layer / bad hook_name).
  5. Output detached, requires_grad=False.

CPU only — no CUDA required (tiny seeded models in conftest.py, deterministic).

`capture_packed` is ADDITIVE (does not touch `capture` / `_truncated_forward` /
the S1 compiled path) — all existing tests in this directory remain green.

The CORRECT equivalence for the packed-vs-unpacked test is:
  capture_packed(docs)[i]  ≈  padded_capture(all_docs)[i, :Lᵢ]
i.e. the padded BATCH reference (right-padded to max length), then stripped per
true length.  This is what `_get_sae_activations_batched` does in production.
Comparing against per-doc individual `capture` runs would be wrong because
`capture` runs without `position_ids`, whereas the packed path passes
`position_ids`, introducing a different (but valid) HF code path that may
produce sub-ULP fp32 differences for the same semantic computation.
"""

from __future__ import annotations

import pytest
import torch as t
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from errors import ExtractionError
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import (
    _build_tiny_gemma2,
    _build_tiny_gemma3,
)

# Layer to truncate at.  0 <= 2 < TINY_LAYERS (4).
_LAYER = 2
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"

# Ragged document batch (lengths [5, 2, 6] — distinct so a wrong split offset
# mis-slices) of valid tiny-vocab (< TINY_VOCAB=64) token ids.
_DOCS: list[list[int]] = [
    [3, 9, 1, 7, 2],     # L=5
    [5, 8],              # L=2
    [1, 4, 6, 0, 5, 2],  # L=6
]

# Tight fp32 tolerance: packed vs padded is a same-kernel (eager) comparison,
# the only differences are mask construction (block-diag vs right-pad) and the
# per-doc-reset position_ids — both exact in fp32.  1e-5 leaves headroom for
# fp32 float reduction ordering noise.
_ATOL = 1e-5
_RTOL = 1e-5


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _build_truncated(model_name: str, fresh_factory) -> NativeCollector:
    nc = NativeCollector.truncated(
        fresh_factory(), model_name=model_name, layer=_LAYER
    )
    assert nc is not None
    return nc


def _padded_reference_per_doc(
    nc: NativeCollector,
    docs: list[list[int]],
) -> list[t.Tensor]:
    """Build the padded reference exactly as `_get_sae_activations_batched`.

    Right-pad token-id 0 to max length, run the production `capture` on the
    whole batch, then strip each row back to its true length.  This is the
    authoritative thing `capture_packed[i]` must match per doc.
    """
    lengths = [len(ids) for ids in docs]
    max_seq = max(lengths)
    device = nc.device
    padded = t.zeros(len(docs), max_seq, dtype=t.long, device=device)
    for i, ids in enumerate(docs):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    acts = nc.capture(
        padded,
        nc.truncated_layer,
        hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )
    return [acts[i, : lengths[i]] for i in range(len(docs))]


def _capture_packed(
    nc: NativeCollector, token_id_lists: list[list[int]]
) -> list[t.Tensor]:
    return nc.capture_packed(
        token_id_lists,
        _LAYER,
        hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )


# --------------------------------------------------------------------------- #
# Test 1 — packed ≈ padded per-doc equivalence (gemma-2 AND gemma-3)           #
# --------------------------------------------------------------------------- #


def _assert_packed_matches_padded(model_name: str, fresh_factory) -> None:
    """Per-doc packed outputs must allclose the per-doc padded-then-stripped
    reference (the production `_get_sae_activations_batched` recipe).

    Build TWO truncated collectors with byte-identical weights — one for packed,
    one for the padded reference — to isolate the packing math.
    """
    nc_packed = _build_truncated(model_name, fresh_factory)
    nc_padded = _build_truncated(model_name, fresh_factory)

    ref = _padded_reference_per_doc(nc_padded, _DOCS)
    packed_slices = _capture_packed(nc_packed, _DOCS)

    assert len(packed_slices) == len(_DOCS)
    for i, (packed, r) in enumerate(zip(packed_slices, ref, strict=True)):
        assert packed.shape == r.shape, (
            f"{model_name} doc {i}: packed shape {packed.shape} != "
            f"ref shape {r.shape}"
        )
        max_abs = (packed - r).abs().max().item()
        assert t.allclose(packed, r, atol=_ATOL, rtol=_RTOL), (
            f"{model_name} doc {i}: packed != padded reference "
            f"(max_abs={max_abs!r}, atol={_ATOL})"
        )


def test_packed_matches_padded_gemma2() -> None:
    """gemma-2 tiny twin: packed per-doc slice ≈ padded-batch reference."""
    _assert_packed_matches_padded("gemma-2-2b", _build_tiny_gemma2)


def test_packed_matches_padded_gemma3() -> None:
    """gemma-3 tiny twin: packed per-doc slice ≈ padded-batch reference.

    Gemma-3 has dual local/global RoPE.  Packing MUST reset both per-doc via
    per-doc-reset `position_ids`; any failure here means cross-doc RoPE
    contamination.
    """
    _assert_packed_matches_padded("gemma-3-1b-pt", _build_tiny_gemma3)


def test_gemma3_packed_sliding_mask_is_doc_local() -> None:
    """Gemma-3 packed position_ids reset document-local attention.

    Regression for the issue-599 root-cause hypothesis: a packed
    ``position_ids=[0..1099, 0..7]`` sequence must not let the short second doc
    attend back into the long first doc, and token 1099 in doc0 must see only
    its own 1024-token sliding window.
    """
    sliding_window = 1024
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=sliding_window,
    )
    config._attn_implementation = "eager"
    position_ids = t.cat([t.arange(1100), t.arange(8)]).unsqueeze(0)
    inputs_embeds = t.zeros(1, position_ids.shape[1], 8)

    mask = create_sliding_window_causal_mask(
        config,
        inputs_embeds,
        None,
        past_key_values=None,
        position_ids=position_ids,
    )

    assert isinstance(mask, t.Tensor)
    allowed = mask == 0
    doc1_rows = slice(1100, 1108)
    doc0_cols = slice(0, 1100)
    assert not bool(allowed[0, 0, doc1_rows, doc0_cols].any())

    doc0_token_1099_allowed = (
        allowed[0, 0, 1099, :1100].nonzero(as_tuple=True)[0].tolist()
    )
    assert doc0_token_1099_allowed == list(range(76, 1100))
    assert len(doc0_token_1099_allowed) == sliding_window

    assert allowed[0, 0, 1100, 1100]
    assert not allowed[0, 0, 1100, 1101]
    assert (
        allowed[0, 0, 1107].nonzero(as_tuple=True)[0].tolist()
        == list(range(1100, 1108))
    )


# --------------------------------------------------------------------------- #
# Test 2 — use_cache=False teeth-guard (NON-VACUOUS mutation probe)            #
# --------------------------------------------------------------------------- #


class _UseCacheRecorder:
    """Wraps the real truncated `_base`, recording the `use_cache` kwarg and
    OPTIONALLY forcing it to a different value (to prove True corrupts)."""

    def __init__(self, real_base, *, force_use_cache=None):
        self._real_base = real_base
        self.force_use_cache = force_use_cache
        self.seen_use_cache: list = []
        self.config = real_base.config

    def parameters(self):
        # capture_packed reads `device` via `next(self._base.parameters())`.
        return self._real_base.parameters()

    def __call__(self, *args, **kwargs):
        self.seen_use_cache.append(kwargs.get("use_cache", "ABSENT"))
        if self.force_use_cache is not None:
            kwargs["use_cache"] = self.force_use_cache
        return self._real_base(*args, **kwargs)


def test_use_cache_false_kwarg_enforced_gemma2() -> None:
    """capture_packed passes use_cache=False (hard-wired literal, not a knob).

    Monkeypatches `_base` with a recorder to observe the exact kwarg value
    that `capture_packed` passes; asserts it is exactly `False`.
    """
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    rec = _UseCacheRecorder(nc._base)
    nc._base = rec
    _capture_packed(nc, _DOCS)
    assert rec.seen_use_cache, "recorder was never called"
    assert rec.seen_use_cache == [False] * len(rec.seen_use_cache), (
        f"capture_packed must pass use_cache=False (got {rec.seen_use_cache!r})"
    )


def test_use_cache_false_teeth_guard_non_vacuous_gemma2() -> None:
    """Prove that IF use_cache=True were used, at least one doc ≥1 diverges
    from the padded reference — pinning invariant I3 non-vacuously.

    Strategy:
      (a) Compute the padded reference.
      (b) Compute capture_packed (use_cache=False) — must allclose ref.
      (c) Force use_cache=True via the recorder → at least one doc ≥1 must
          DIVERGE from ref (cross-doc attention leakage signature).
    """
    nc_ref = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    nc_ok = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    nc_bad = _build_truncated("gemma-2-2b", _build_tiny_gemma2)

    ref = _padded_reference_per_doc(nc_ref, _DOCS)

    # (b) real capture_packed — must allclose ref
    good_slices = _capture_packed(nc_ok, _DOCS)
    for i, (good, r) in enumerate(zip(good_slices, ref, strict=True)):
        assert t.allclose(good, r, atol=_ATOL, rtol=_RTOL), (
            f"gemma-2 doc {i}: capture_packed (use_cache=False) != padded ref "
            f"(max_abs={(good - r).abs().max().item()!r})"
        )

    # (c) force use_cache=True
    rec_bad = _UseCacheRecorder(nc_bad._base, force_use_cache=True)
    nc_bad._base = rec_bad
    bad_slices = _capture_packed(nc_bad, _DOCS)

    # At least one doc ≥1 must diverge (cross-doc leakage)
    diverged = any(
        not t.allclose(bad_slices[i], ref[i], atol=_ATOL, rtol=_RTOL)
        for i in range(1, len(_DOCS))
    )
    assert diverged, (
        "use_cache=True produced outputs ALLCLOSE to per-doc reference for all "
        "docs ≥1 — the teeth-guard is VACUOUS (either HF packing-autodetect "
        "now works with use_cache=True, or the test needs a longer sequence). "
        "The I3 non-vacuousness check FAILED."
    )


# --------------------------------------------------------------------------- #
# Test 3 — length splitting / shape contract                                   #
# --------------------------------------------------------------------------- #


def test_length_splitting_empty() -> None:
    """Empty token_id_lists → empty list, no crash."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    result = nc.capture_packed(
        [],
        _LAYER,
        hook_name=_RESID_HOOK,
        activation_type=ActivationType.residual,
    )
    assert result == []


def test_length_splitting_single_token_doc() -> None:
    """B=1 single-token doc → [1, d] slice."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    slices = _capture_packed(nc, [[7]])
    assert len(slices) == 1
    assert slices[0].ndim == 2
    assert slices[0].shape[0] == 1  # L=1


def test_length_splitting_ragged() -> None:
    """Ragged batch → each output slice has the correct per-doc length."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    docs: list[list[int]] = [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]]
    slices = _capture_packed(nc, docs)
    assert len(slices) == len(docs)
    for i, (doc, sl) in enumerate(zip(docs, slices, strict=True)):
        assert sl.shape[0] == len(doc), (
            f"doc {i}: slice length {sl.shape[0]} != expected {len(doc)}"
        )
        assert sl.ndim == 2, f"doc {i}: expected 2-D slice, got shape {sl.shape}"


def test_length_splitting_single_doc() -> None:
    """Single multi-token doc → one slice of the right length."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    doc = [3, 1, 4, 1, 5, 9]
    slices = _capture_packed(nc, [doc])
    assert len(slices) == 1
    assert slices[0].shape[0] == len(doc)
    assert slices[0].ndim == 2


# --------------------------------------------------------------------------- #
# Test 4 — typed-error contract                                                 #
# --------------------------------------------------------------------------- #


def test_error_wrong_activation_type() -> None:
    """Non-residual activation_type → ExtractionError."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    with pytest.raises(ExtractionError, match="residual-only"):
        nc.capture_packed(
            [[1, 2, 3]],
            _LAYER,
            hook_name=_RESID_HOOK,
            activation_type=ActivationType.mlp,
        )


def test_error_wrong_layer() -> None:
    """layer != truncated_layer → ExtractionError."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    wrong_layer = _LAYER + 1
    with pytest.raises(ExtractionError, match="truncated for layer"):
        nc.capture_packed(
            [[1, 2, 3]],
            wrong_layer,
            hook_name=_RESID_HOOK,
            activation_type=ActivationType.residual,
        )


def test_error_bad_hook_name() -> None:
    """hook_name not containing 'hook_resid_post' → ExtractionError."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    with pytest.raises(ExtractionError, match="resid_post"):
        nc.capture_packed(
            [[1, 2, 3]],
            _LAYER,
            hook_name=f"blocks.{_LAYER}.hook_mlp_out",
            activation_type=ActivationType.residual,
        )


def test_error_none_hook_name() -> None:
    """hook_name=None → ExtractionError (same resid_post guard)."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    with pytest.raises(ExtractionError, match="resid_post"):
        nc.capture_packed(
            [[1, 2, 3]],
            _LAYER,
            hook_name=None,
            activation_type=ActivationType.residual,
        )


# --------------------------------------------------------------------------- #
# Test 5 — output tensors detached, requires_grad=False                        #
# --------------------------------------------------------------------------- #


def test_output_detached_no_grad() -> None:
    """Each per-doc slice must be detached and requires_grad=False."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    docs: list[list[int]] = [[1, 2], [3, 4, 5]]
    slices = _capture_packed(nc, docs)
    for i, sl in enumerate(slices):
        assert not sl.requires_grad, (
            f"slice {i}: requires_grad=True (must be detached)"
        )


# --------------------------------------------------------------------------- #
# Test 6 — `_rowpack_verified` defaults False                                  #
# --------------------------------------------------------------------------- #


def test_rowpack_verified_defaults_false() -> None:
    """A freshly built collector has `_rowpack_verified = False` (inert until
    the S2 runtime gate calls `set_rowpack_verified(True)`)."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    assert nc._rowpack_verified is False


def test_set_rowpack_verified_flips() -> None:
    """`set_rowpack_verified(True)` flips the flag; `False` flips it back."""
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    nc.set_rowpack_verified(True)
    assert nc._rowpack_verified is True
    nc.set_rowpack_verified(False)
    assert nc._rowpack_verified is False


# --------------------------------------------------------------------------- #
# Test 7 — forward failure is wrapped as a typed ExtractionError               #
# --------------------------------------------------------------------------- #


def test_capture_packed_forward_failure_wrapped_as_extraction_error() -> None:
    """A packed forward that throws — here a token id >= TINY_VOCAB (64), which
    makes the embedding lookup raise an index-out-of-range error mid-forward —
    is caught and re-raised as a typed `ExtractionError` (carrying the cause),
    not leaked raw. Covers `capture_packed`'s defensive `except Exception`
    branch. Non-vacuous: the equivalence tests above exercise the same call
    with in-vocab ids and do NOT raise, so the raise here is caused by the
    forward failure, not the call shape.
    """
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    with pytest.raises(ExtractionError, match="packed truncated forward failed"):
        nc.capture_packed(
            [[100, 5]],  # 100 >= TINY_VOCAB (64) ⇒ embedding raises in the forward
            _LAYER,
            hook_name=_RESID_HOOK,
            activation_type=ActivationType.residual,
        )


# --------------------------------------------------------------------------- #
# Test 8 — capture_packed routes through the S1 compiled base when active      #
# --------------------------------------------------------------------------- #


def test_capture_packed_routes_through_compiled_base_when_active(monkeypatch) -> None:
    """When the S1 compiled forward is armed + active, `capture_packed` runs
    the packed forward through `_compiled_base` (so packing COMPOSES with S1's
    torch.compile), not the eager `_base`. Spy: a recording stand-in for
    `_compiled_base` (delegating to the real eager base for a valid output)
    MUST be the callable used. Non-vacuous: the equivalence tests above run
    with compile inactive (the default) and never install the spy, so a
    regression that ignored `_compiled_base` would fail THIS test only.
    """
    nc = _build_truncated("gemma-2-2b", _build_tiny_gemma2)
    real_base = nc._base
    calls: list[int] = []

    def _spy_compiled(**kwargs):
        calls.append(1)
        return real_base(**kwargs)

    # Stand in for torch.compile(_base) without paying a real CPU compile.
    monkeypatch.setattr(nc, "_compiled_base", _spy_compiled)
    nc.set_compiled_active(True)
    out = _capture_packed(nc, _DOCS)

    assert calls == [1], (
        "capture_packed must route through _compiled_base exactly once when "
        "compile is active (packing must compose with S1 compile)"
    )
    assert len(out) == len(_DOCS)
