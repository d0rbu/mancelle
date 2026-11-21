"""THE EQUIVALENCE PROOF: universal feature-GCG on a COMPACTED model over V is
BYTE-IDENTICAL to the same GCG on the bitwise-equivalent UNCOMPACTED model restricted to
the same V (#720 T1 headline).

This is the load-bearing claim of PR-2's Task 1 — the id-space translation — and it is
proven by RUNNING BOTH OPTIMIZERS, not by reasoning about them.

    Build the matched pair ``ref = _build(None)`` / ``cmp = _build(VOCAB)`` at the SAME
    seed. Slice-init makes ``cmp``'s ``V`` compact embedding rows bitwise equal to the
    corresponding rows of ``ref``'s full table (asserted at step 0 for non-vacuity — the
    same premise ``test_encdec_compaction_byte_exact_e2e.py`` proves for training). Run the
    universal optimizer on ``cmp`` (proposal universe from ``feature_vocab_allowed_ids`` →
    V) and on ``ref`` (universe explicitly set to the SAME sorted V), everything seeded
    identically.

Because the compacted forward is bitwise identical to the uncompacted one and the SOLE
compacted-only op — the ``id_weight`` row gather — is an EXACT integer translation
(``old_to_new``), every gradient, every proposal score, every candidate bag, and every
accept/reject decision matches bit for bit. So:

* the accepted bag is byte-identical (``dict(bag_cmp) == dict(bag_ref)``), and
* the full candidate/loss TRAJECTORY is byte-identical (a seeded objective spy records
  every ``(bag, loss)`` the optimizer evaluates; the two logs are ``==``).

fp32/SDPA is the byte-identical regime. A bf16 variant (Step 7 below) casts every
parameter/buffer to bf16 EXCEPT the rotary ``cos``/``sin``, which the model's own
build path always keeps in ``COMPUTE_DTYPE`` (fp32 on CPU) — see
``_bf16_cast_respecting_compute_dtype``. It asserts closeness within a tight, empirically
calibrated tolerance rather than exact equality, in case a different machine's BLAS/thread
config perturbs bf16 rounding order; on THIS environment the measured divergence is
exactly 0.0 (the same slice-init relationship that makes fp32 byte-identical survives
elementwise bf16 rounding too), so the tolerance functions as a bug-DETECTION threshold,
not a "close enough" fudge factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

import nanochat.flash_attention as fa_module
import steering.conditional.feature_gcg.optimize as opt_module
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember
from steering.conditional.feature_gcg.objective import FeatureLmItem, lm_loss_on_masked_tokens
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from steering.conditional.feature_vocab import FeatureVocab

# ─────────────────────────────────────────────────────────────────────────────
# Matched-pair fixtures (the byte-exact compaction test's `_build` pattern)
# ─────────────────────────────────────────────────────────────────────────────

N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)  # V = 12
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
NUM_BUCKETS = 4
VOCAB_SIZE = 128
SEQ_LEN = 32
T_LEN = 12
SEED = 42
CONTEXT = FeatureConditioning(entries=((10, 1), (20, 2)))  # ids ⊆ V
WARM = FeatureConditioning(entries=((3, 1), (7, 2)))  # ids ⊆ V
CPU = torch.device("cpu")

# bf16 forwards round; the id translation does NOT (exact integer gather). Empirically,
# on this CPU/SDPA build the compacted-vs-uncompacted divergence measures EXACTLY 0.0
# (compaction is purely representational: the gathered rows are bitwise related, so
# deterministic bf16 rounding applied to them stays related too) — so this atol is not a
# "close enough" fudge factor, it is a DETECTION THRESHOLD. Calibrated against two
# simulated row-translation bugs on this fixture (corrupt one gathered row: diff ~9.8e-4;
# swap two gathered rows, approximating a wrong id->row map: diff ~2.1e-3) — tight enough
# to catch either (>= ~5x headroom below both) while leaving >= 500x headroom above the
# true 0.0 for incidental cross-platform float noise.
BF16_ATOL = 5e-4


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _build(vocab: FeatureVocab | None, *, seed: int = SEED) -> EncoderDecoderGPT:
    """Meta → ``to_empty`` → seed → ``init_weights`` — the production/byte-exact path.

    At the SAME seed, ``_build(None)`` and ``_build(VOCAB)`` are slice-init related: the
    compacted table's ``V`` rows are bitwise the corresponding rows of the full table
    (asserted at step 0 below).
    """
    dec_config = GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=2, n_head=None, n_kv_head=None, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None)
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    model._loaded_meta_data = {"user_config": {}}  # ty: ignore[invalid-assignment]
    return model


def _restrict_both_to_V(monkeypatch: pytest.MonkeyPatch, ref: Any) -> None:
    """``ref`` (uncompacted) is explicitly restricted to sorted V; every other model keeps
    the REAL ``feature_vocab_allowed_ids`` path (so ``cmp`` derives V from its own buffer)."""
    real = opt_module.feature_vocab_allowed_ids

    def _patched(model: Any) -> Any:
        if model is ref:
            return torch.tensor(sorted(ALLOWED), dtype=torch.int64)
        return real(model)

    monkeypatch.setattr(opt_module, "feature_vocab_allowed_ids", _patched)


def _config() -> FeatureGcgConfig:
    return FeatureGcgConfig(
        max_steps=6,
        candidate_batch_size=4,
        top_k=4,
        temperature=1.0,
        master_seed=0,
        reg_lambda=0.0,
        min_k_pos=0,
        max_k_pos=6,
    )


def _member(seed: int) -> FeatureGradMember:
    gen = torch.Generator().manual_seed(seed)
    half = T_LEN // 2
    seg_map = torch.tensor([[0] * half + [1] * (T_LEN - half)], dtype=torch.long)
    dec_input = torch.randint(1, 100, (1, T_LEN), generator=gen)
    dec_target = torch.full((1, T_LEN), -1, dtype=torch.long)
    for pos in range(T_LEN - 3, T_LEN):
        dec_target[0, pos] = int(torch.randint(1, 100, (1,), generator=gen).item())
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[CONTEXT, FeatureConditioning(entries=())],
        seg_map=seg_map,
        generation_slot=1,
    )


def _members() -> list[FeatureGradMember]:
    return [_member(1), _member(2), _member(3)]


def _assert_slice_init_related(ref: EncoderDecoderGPT, cmp: EncoderDecoderGPT) -> None:
    """Non-vacuity: cmp's V compact rows must be bitwise ref's corresponding rows."""
    sel = torch.tensor(ALLOWED)
    assert torch.equal(
        cmp.encoder.feature_embed.weight, ref.encoder.feature_embed.weight[sel]
    ), "matched pair is not slice-init related — the equivalence would be vacuous"
    for key in cmp.encoder.value_embeds:
        assert torch.equal(
            cmp.encoder.value_embeds[key].weight, ref.encoder.value_embeds[key].weight[sel]  # ty: ignore[invalid-argument-type, not-subscriptable]
        )


@dataclass
class _SpyObjective:
    """Wraps an objective, recording every ``(bag, loss)`` the optimizer evaluates."""

    inner: Any

    def __post_init__(self) -> None:
        self.log: list[tuple[str, tuple[Any, ...], tuple[float, ...] | None]] = []

    def losses(self, items: Any) -> list[float]:
        out = self.inner.losses(items)
        self.log.append(
            ("losses", tuple(tuple(i.bag.entries) for i in items), tuple(out))
        )
        return out

    def gradients(self, items: Any, step: int) -> list[Any]:
        out = self.inner.gradients(items, step)
        self.log.append(
            ("gradients", tuple(tuple(i.bag.entries) for i in items), None)
        )
        return out


def _run_with_spy(
    model: EncoderDecoderGPT,
) -> tuple[FeatureConditioning, _SpyObjective]:
    """Run the universal optimizer on ``model``, whatever dtype it is ALREADY in.

    Takes no ``dtype`` argument: the caller casts ``model`` (see
    ``_bf16_cast_respecting_compute_dtype``) BEFORE calling this, exactly like
    ``_aggregate_objective`` below — casting here too would be a second, redundant knob
    that could silently disagree with the model the caller already prepared.
    """
    spy_holder: dict[str, _SpyObjective] = {}

    def _factory(m: Any) -> Any:
        spy = _SpyObjective(lm_loss_on_masked_tokens(m, device=CPU))
        spy_holder["spy"] = spy
        return spy

    bag = optimize_feature_conditioning_universal(
        model,
        _members(),
        WARM,
        objective_factory=_factory,
        config=_config(),
    )
    return bag, spy_holder["spy"]


def _aggregate_objective(model: EncoderDecoderGPT, bag: FeatureConditioning) -> float:
    """Mean masked-CE of ``bag`` over the members (reg_lambda=0 ⇒ this IS the objective)."""
    obj = lm_loss_on_masked_tokens(model, device=CPU)
    losses = obj.losses([FeatureLmItem(member=m, bag=bag) for m in _members()])
    return sum(losses) / len(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — the headline: fp32 byte-identity
# ─────────────────────────────────────────────────────────────────────────────


def test_compacted_optimizer_is_byte_identical_to_uncompacted_over_V(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted bag AND the full trajectory are byte-identical at fp32/SDPA."""
    ref = _build(None)
    cmp = _build(VOCAB)
    _assert_slice_init_related(ref, cmp)  # non-vacuity
    _restrict_both_to_V(monkeypatch, ref)

    bag_ref, spy_ref = _run_with_spy(ref)
    bag_cmp, spy_cmp = _run_with_spy(cmp)

    # (1) The accepted bag is byte-identical, in ORIGINAL-id space, and ⊆ V.
    assert dict(bag_cmp.entries) == dict(bag_ref.entries)
    assert set(bag_cmp.feature_ids()) <= set(ALLOWED)

    # (2) The optimizer actually did work (non-vacuous trajectory).
    assert len(spy_ref.log) > 1
    assert any(kind == "losses" for kind, _b, _l in spy_ref.log)

    # (3) The FULL candidate/loss trajectory matches bit for bit — same proposed bags,
    #     same loss values at every step. A wrong id→row translation would perturb the
    #     proposal scores and diverge this immediately.
    assert spy_cmp.log == spy_ref.log


def test_the_matched_pair_is_actually_compacted_vs_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against a vacuous proof: the two models genuinely differ in table height, and
    the compacted one derives V from its OWN buffer (the real helper), not the patch."""
    ref = _build(None)
    cmp = _build(VOCAB)
    assert cmp.encoder.feature_embed.weight.shape[0] == len(ALLOWED)
    assert ref.encoder.feature_embed.weight.shape[0] == N_SAE
    assert cmp.feature_vocab is not None and ref.feature_vocab is None
    # The real helper reads cmp's persistent buffer and returns exactly sorted V.
    assert opt_module.feature_vocab_allowed_ids(cmp).tolist() == sorted(ALLOWED)  # ty: ignore[unresolved-attribute]
    assert opt_module.feature_vocab_allowed_ids(ref) is None


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — bf16 tolerance variant (translation exact; only the forward rounds)
# ─────────────────────────────────────────────────────────────────────────────


def _bf16_cast_respecting_compute_dtype(model: EncoderDecoderGPT) -> EncoderDecoderGPT:
    """Cast ``model`` to bf16 the way the model's OWN build path treats dtype.

    A blanket ``model.to(torch.bfloat16)`` casts EVERY floating-point parameter/buffer,
    including the rotary ``cos``/``sin`` buffers — but ``EncoderDecoderGPT.forward``
    (``steering/conditional/encoder_decoder/model.py`` ~:568, mirrored by
    ``nanochat/gpt.py`` ~:556 for the plain decoder) asserts ``self.cos.dtype ==
    COMPUTE_DTYPE``. On this CPU-only box ``COMPUTE_DTYPE`` auto-detects to
    ``torch.float32`` (``nanochat.common._detect_compute_dtype``: no CUDA ⇒ fp32), so a
    blanket cast trips that assert immediately — the model's own ``init_weights`` /
    ``init_buffers`` never do this: ``cos``/``sin`` are always built directly in
    ``COMPUTE_DTYPE`` (``_precompute_rotary_embeddings``'s own ``.to(COMPUTE_DTYPE)``),
    and only ``wte``/``value_embeds`` ever get an explicit blanket dtype cast, gated to
    CUDA. This mirrors that: cast everything, then restore the two buffers the model
    itself never lets drift from ``COMPUTE_DTYPE``.
    """
    model.to(torch.bfloat16)
    model.cos = model.cos.to(COMPUTE_DTYPE)
    model.sin = model.sin.to(COMPUTE_DTYPE)
    return model


def _bf16_cpu_forward_supported() -> bool:
    """Probe whether a bf16 encoder-decoder forward runs on CPU under SDPA."""
    old = fa_module._override_impl
    fa_module._override_impl = FlashAttnImpl.SDPA
    try:
        m = _bf16_cast_respecting_compute_dtype(_build(None))
        obj = lm_loss_on_masked_tokens(m, device=CPU)
        obj.losses([FeatureLmItem(member=_member(1), bag=WARM)])
        return True
    except Exception:
        return False
    finally:
        fa_module._override_impl = old


_BF16_CPU_OK = _bf16_cpu_forward_supported()


@pytest.mark.skipif(
    not _BF16_CPU_OK,
    reason=(
        "bf16 encoder-decoder forward unsupported on CPU in this environment (missing "
        "bf16 CPU kernel for some op the forward needs) — this needs a CUDA box to run; "
        "see this module's _bf16_cpu_forward_supported probe for the exact failure"
    ),
)
def test_compacted_optimizer_matches_uncompacted_in_bf16_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bf16: the id translation stays EXACT, so compacted ≈ uncompacted within a documented
    ``atol``. ``cos``/``sin`` stay in ``COMPUTE_DTYPE`` (fp32 on CPU) exactly as the real
    model always keeps them (:func:`_bf16_cast_respecting_compute_dtype`); everything else
    — including every ``Linear`` weight — is genuinely bf16, so ``nanochat.gpt.Linear``'s
    cast-on-use (``self.weight.to(dtype=x.dtype)``) and the encoder's row-gather both run
    against real bf16-precision values, exactly as a saved bf16 checkpoint would."""
    ref = _bf16_cast_respecting_compute_dtype(_build(None))
    cmp = _bf16_cast_respecting_compute_dtype(_build(VOCAB))
    _restrict_both_to_V(monkeypatch, ref)

    # (a) Forward-equivalence anchor: scoring the SAME bag on both models agrees within
    #     atol — the id→row gather is exact, so this bounds the pure bf16-forward drift.
    for bag in (WARM, FeatureConditioning(entries=((3, 1), (21, 2), (40, 3)))):
        assert abs(_aggregate_objective(ref, bag) - _aggregate_objective(cmp, bag)) <= BF16_ATOL

    # (b) The two optimizers' achieved objectives agree within the same tolerance.
    bag_ref, _ = _run_with_spy(ref)
    bag_cmp, _ = _run_with_spy(cmp)
    assert set(bag_cmp.feature_ids()) <= set(ALLOWED)
    obj_ref = _aggregate_objective(ref, bag_ref)
    obj_cmp = _aggregate_objective(cmp, bag_cmp)
    assert abs(obj_ref - obj_cmp) <= BF16_ATOL, (
        f"bf16 objectives diverged beyond {BF16_ATOL}: ref={obj_ref} cmp={obj_cmp}"
    )
