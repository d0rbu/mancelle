"""Real-model (Layer B, @slow) gate + measurement for the Task-6 batched prototype.

Two responsibilities:

1. **fp64-equivalence gate** (``real_model`` fixture → auto-marked ``slow``).  On
   the real d12 encdec checkpoint (GPU bf16 compute), the batched objective (now
   the SOLE production path) must equal the per-item REFERENCE
   (``tests.steering.feature_gcg._reference``: ``reference_masked_loss`` /
   ``reference_g_steer`` / ``reference_g_steer_grad``) within a *derived*
   tolerance.  The exactness of the batching MATH is proven on the deterministic
   CPU/float32 tiny model in ``tests/steering/feature_gcg/unit/test_perf_batching.py``
   (there production and reference converge to a float32-ULP ~1e-5 — i.e. they are
   the same computation); here we only have to bound the residual the real model's
   **bf16** matmuls add when the batch·T kernel shape changes.

   Deriving the atol.  bf16 carries 8 mantissa bits ⇒ unit roundoff
   ``u = 2**-8 ≈ 3.9e-3``.  A masked-mean CE loss on this model is ``O(2-6)`` and
   accumulates through ~12 transformer layers, so the batch-shape reassociation of
   the *same* math is bounded by a small multiple of ``u * loss`` ≈ a few ×1e-2.
   We set ``_BF16_LOSS_ATOL = 5e-2`` and RECORD the measured max |Δ| (printed) to
   confirm the margin.  The comparison subtraction is done in float64 (Python
   ``float``) so it adds no error — the bounded quantity is purely the bf16
   forward delta.  A genuine batching bug shifts a loss by ``O(0.1-1)``, well above
   this, so the gate keeps teeth.  The class-margin gradient additionally crosses a
   ``index_add`` (atomic, non-deterministic on GPU) backward, so it carries a
   slightly looser ``_BF16_GRAD_ATOL``; its EXACT ``desired − contrast`` sign is
   pinned bit-for-bit on CPU.

2. **Step-4 speedup measurement** (``@pytest.mark.slow``, env-gated like the Task-1
   baseline).  Re-runs ``profile_optimize`` on the real d12_l12 workload with
   batching ON, writes ``artifacts/profile_batched.json``, and prints the
   end-to-end + per-region speedup versus ``artifacts/profile_baseline.json``.

Enable via::

    export FEATURE_GCG_D12_CKPT=/path/to/d12_encdec_checkpoint
    export FEATURE_GCG_L12_RUN=/path/to/d12_l12_source_run   # measurement only
    export FEATURE_GCG_D12_STEP=54600                        # default
    uv run pytest tests/steering/feature_gcg/integration/test_perf_batching.py -m slow -v -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember, substitute_bag
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    ClassMarginMember,
    FeatureLmItem,
    _masked_mean_scalar,
    class_margin_objective,
    lm_loss_on_masked_tokens,
)
from steering.eval.axbench_base.types import SteeringMode
from tests.steering.feature_gcg._equivalence import feature_grads_equal
from tests.steering.feature_gcg._reference import (
    reference_g_steer,
    reference_g_steer_grad,
    reference_masked_loss,
)
from tests.steering.feature_gcg.perf.gpu_wait import wait_for_idle_local_gpu
from tests.steering.feature_gcg.perf.profile_harness import profile_optimize
from tests.steering.feature_gcg.perf.test_profile_optimize import (
    ARTIFACTS,
    _assert_breakdown,
    _load_real_l12_workload,
    _skip_unless_env,
)

_MODE = SteeringMode.POSITIVE

# Derived bf16 batch-shape tolerances.  MEASURED on the real d12 model: the
# batched-vs-per-item loss delta is RELATIVE to the loss magnitude — ~3-4e-3 * loss
# (≈ bf16 unit roundoff u = 2**-8 ≈ 3.9e-3, compounded over the ~12 layers): e.g.
# 7.6e-2 on a loss of ~22.  So we bound each loss by ``rel*|loss| + floor``, and a
# class-margin ``g_steer`` (a DIFFERENCE of two losses) by ``rel*(|L_rich|+
# |L_avoid|) + floor`` — the bf16 deltas of the two large underlying losses (NOT of
# the small g_steer).  ``rel = 1e-2`` is ~3x the measured floor: tight enough that a
# real batching bug (an O(loss) shift) REDs, loose enough to pass true bf16
# reassociation.  The comparison subtraction is float64-exact (Python ``float``), so
# the bounded quantity is purely the bf16 forward delta.  Gradients reassociate FAR
# less (measured max |Δ| ~4e-5 — the backward cancels most forward drift), so the
# grad atol is tight at 1e-3 (~20x margin).  B=1 is byte-IDENTICAL even in bf16
# (same kernel shape) and is asserted exactly.  All tests PRINT the measured |Δ| so
# the margin is auditable.
_BF16_REL = 1e-2
_BF16_ABS_FLOOR = 1e-2
_BF16_GRAD_ATOL = 1e-3


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SDPA (the Task-1 baseline ran under SDPA — match it for the gradient
    backward and apples-to-apples timing)."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Real-shaped member builders (valid buckets DERIVED from the model num_buckets)
# ─────────────────────────────────────────────────────────────────────────────


def _valid_hi_bucket(model: Any) -> int:
    """Top in-range bucket index.  d12_l12 has num_buckets == 2 (the minimum); an
    out-of-range bucket triggers a CUDA device-side assert in the magnitude gather."""
    num_buckets = int(model.encoder.magnitude_embed.weight.shape[0])
    hi = min(1, num_buckets - 1)
    assert 0 <= hi < num_buckets, f"derived bucket {hi} out of range for {num_buckets}"
    return hi


def _real_member(model: Any, device: torch.device, *, seed: int, t_len: int, n_scored: int = 3) -> FeatureGradMember:
    """One real-shaped teacher-forced member (placeholder bag at the gen slot 1)."""
    hi = _valid_hi_bucket(model)
    gen = torch.Generator(device=device).manual_seed(seed)
    dec_input = torch.randint(1, 1000, (1, t_len), generator=gen, device=device)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=device)
    for pos in range(t_len - n_scored, t_len):
        dec_target[0, pos] = int(torch.randint(1, 1000, (1,), generator=gen, device=device).item())
    seg_map = torch.zeros((1, t_len), dtype=torch.long, device=device)
    seg_map[0, t_len // 2 :] = 1  # second half cross-attends the generation slot
    context = FeatureConditioning(entries=((5, 0), (11, hi)))
    placeholder = FeatureConditioning(entries=())
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[context, placeholder],
        seg_map=seg_map,
        generation_slot=1,
    )


def _candidate_bag(model: Any) -> FeatureConditioning:
    """A valid candidate conditioning bag (buckets in range, feature-ids < n_sae)."""
    hi = _valid_hi_bucket(model)
    return FeatureConditioning(entries=((17, hi), (23, 0), (29, hi)))


def _ref_lm_loss(
    model: Any, member: FeatureGradMember, bag: FeatureConditioning, device: Any
) -> float:
    """Per-item LM REFERENCE loss for one (member, bag) (``reference_masked_loss``)."""
    return reference_masked_loss(
        model,
        dec_input=member.dec_input,
        dec_target=member.dec_target,
        segment_conditions=substitute_bag(member, bag),
        seg_map=member.seg_map,
        scalar_fn=_masked_mean_scalar(member.dec_target != -1),
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. fp64-equivalence gate (auto-@slow via the real_model fixture)
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_losses_batched_matches_reference_real(real_model: Any) -> None:
    """Real model: production (batched) lm ``losses`` == per-item reference within the derived bf16 atol.

    Members of DIFFERING length carrying DIFFERENT bags (the candidate-eval regime).
    """
    device = next(real_model.parameters()).device
    members = [
        _real_member(real_model, device, seed=10 + i, t_len=t)
        for i, t in enumerate([16, 24, 20, 18])
    ]
    hi = _valid_hi_bucket(real_model)
    bags = [
        _candidate_bag(real_model),
        FeatureConditioning(entries=((8, 0),)),
        FeatureConditioning(entries=()),
        FeatureConditioning(entries=((6, hi), (14, 0))),
    ]
    items = [FeatureLmItem(member=m, bag=b) for m, b in zip(members, bags, strict=True)]

    reference = [_ref_lm_loss(real_model, m, b, device) for m, b in zip(members, bags, strict=True)]
    batched = lm_loss_on_masked_tokens(real_model, device=device).losses(items)

    assert len(batched) == len(reference) == len(members)
    deltas = [abs(float(b) - float(p)) for b, p in zip(batched, reference, strict=True)]
    rels = [d / abs(p) if p else 0.0 for d, p in zip(deltas, reference, strict=True)]
    print(
        f"\n[lm losses bf16 equivalence] reference={[round(p, 4) for p in reference]} "
        f"max|Δ|={max(deltas):.3e} max_rel={max(rels):.2e}"
    )
    for b, (d, p) in enumerate(zip(deltas, reference, strict=True)):
        tol = _BF16_REL * abs(p) + _BF16_ABS_FLOOR
        assert d <= tol, f"item {b}: |Δ|={d:.3e} > rel*|loss|+floor={tol:.3e} (loss={p:.3f})"


def test_lm_losses_batched_b1_matches_reference_real(real_model: Any) -> None:
    """Real model: a single-item (B=1) production loss matches its per-item reference byte-for-byte.

    B=1 builds the identical (1, T) decoder + (1, n_seg, F) encoder tensors as the
    single-item reference, so even in bf16 the kernel shape is identical.
    """
    device = next(real_model.parameters()).device
    member = _real_member(real_model, device, seed=99, t_len=20)
    bag = _candidate_bag(real_model)
    item = [FeatureLmItem(member=member, bag=bag)]
    reference = _ref_lm_loss(real_model, member, bag, device)
    batched = lm_loss_on_masked_tokens(real_model, device=device).losses(item)[0]
    print(f"\n[lm losses B=1] reference={reference:.6f} batched={batched:.6f} |Δ|={abs(batched - reference):.3e}")
    # B=1 builds the identical (1, T) kernel shape ⇒ byte-identical even in bf16.
    assert batched == reference


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_class_margin_losses_batched_matches_reference_real(real_model: Any, mode: SteeringMode) -> None:
    """Real model: production (batched) class-margin ``losses`` (g_steer) == per-item reference, both modes."""
    device = next(real_model.parameters()).device
    members = [
        ClassMarginMember(
            rich=_real_member(real_model, device, seed=200 + i, t_len=t),
            avoid=_real_member(real_model, device, seed=300 + i, t_len=t + 2),
        )
        for i, t in enumerate([16, 22, 18])
    ]
    bag = _candidate_bag(real_model)
    items = [ClassMarginItem(member=m, bag=bag) for m in members]

    reference = [reference_g_steer(real_model, m, bag, mode=mode, device=device) for m in members]
    batched = class_margin_objective(real_model, mode=mode, device=device).losses(items)

    # The g_steer bf16 delta scales with the UNDERLYING loss magnitudes (g_steer is
    # their difference), so bound it by rel*(|L_rich|+|L_avoid|) — measured via the
    # (proven) per-item lm reference on the same bag.
    scales = [
        abs(_ref_lm_loss(real_model, m.rich, bag, device))
        + abs(_ref_lm_loss(real_model, m.avoid, bag, device))
        for m in members
    ]

    assert len(batched) == len(reference) == len(members)
    deltas = [abs(float(b) - float(p)) for b, p in zip(batched, reference, strict=True)]
    print(
        f"\n[class_margin losses bf16 equivalence, {mode.value}] "
        f"g_steer reference={[round(p, 4) for p in reference]} max|Δ|={max(deltas):.3e} "
        f"loss_scales={[round(s, 1) for s in scales]}"
    )
    for b, (d, scale) in enumerate(zip(deltas, scales, strict=True)):
        tol = _BF16_REL * scale + _BF16_ABS_FLOOR
        assert d <= tol, f"item {b}: |Δ|={d:.3e} > rel*(L_rich+L_avoid)+floor={tol:.3e}"


def test_class_margin_gradients_batched_matches_reference_real(real_model: Any) -> None:
    """Real model: production (batched) class-margin ``gradients`` == per-item reference within the derived
    bf16+atomic-backward grad atol (``feature_grads_equal(exact=False)``)."""
    device = next(real_model.parameters()).device
    members = [
        ClassMarginMember(
            rich=_real_member(real_model, device, seed=400 + i, t_len=t, n_scored=2),
            avoid=_real_member(real_model, device, seed=500 + i, t_len=t + 1, n_scored=2),
        )
        for i, t in enumerate([16, 20])
    ]
    bag = _candidate_bag(real_model)
    items = [ClassMarginItem(member=m, bag=bag) for m in members]

    reference = [reference_g_steer_grad(real_model, m, bag, mode=_MODE, device=device) for m in members]
    batched = class_margin_objective(real_model, mode=_MODE, device=device).gradients(items, 0)

    assert len(batched) == len(reference) == len(members)
    for b, (gb, gp) in enumerate(zip(batched, reference, strict=True)):
        max_id = float((gb.id_grad - gp.id_grad).abs().max().item())
        max_bucket = float((gb.bucket_grad - gp.bucket_grad).abs().max().item())
        print(
            f"\n[class_margin grad bf16 equivalence] item {b}: "
            f"max|Δ id_grad|={max_id:.3e} max|Δ bucket_grad|={max_bucket:.3e} (atol={_BF16_GRAD_ATOL})"
        )
        # id_weight / bucket_weight are the SHARED static cache ⇒ exactly equal.
        assert torch.equal(gb.id_weight, gp.id_weight)
        assert torch.equal(gb.bucket_weight, gp.bucket_weight)
        feature_grads_equal(gb, gp, exact=False, atol=_BF16_GRAD_ATOL)
        # Non-triviality floor: a vacuous rich≈avoid member (grad_desired−grad_contrast
        # ≈ 0) would pass the equivalence on 0≈0 within the bf16 atol.
        assert gb.id_grad.abs().sum().item() > 0.0, f"item {b}: vacuous (zero) id_grad"
        assert gb.bucket_grad.abs().sum().item() > 0.0, f"item {b}: vacuous (zero) bucket_grad"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Step-4 speedup measurement (@slow, env-gated like the Task-1 baseline)
# ─────────────────────────────────────────────────────────────────────────────


def _lm_factory_batched(device: Any) -> Any:
    """Production (batched) lm objective factory for the speedup measurement."""
    def _factory(m: Any) -> Any:
        return lm_loss_on_masked_tokens(m, device=device)
    return _factory


def _cm_factory_batched(device: Any) -> Any:
    """Production (batched) class-margin objective factory for the speedup measurement."""
    def _factory(m: Any) -> Any:
        return class_margin_objective(m, mode=_MODE, device=device)
    return _factory


def _speedup_table(name: str, baseline: dict[str, Any], batched: dict[str, Any]) -> dict[str, Any]:
    """Print + return the end-to-end and per-region speedup (baseline_ms / batched_ms)."""
    out: dict[str, Any] = {}
    base_total = baseline["total_ms"]
    new_total = batched["total_ms"]
    out["total_ms"] = {"baseline": base_total, "batched": new_total, "speedup": base_total / new_total}
    print(f"\n[{name}] end-to-end: baseline={base_total:.1f}ms batched={new_total:.1f}ms "
          f"=> {base_total / new_total:.2f}x")
    for region in sorted(baseline["regions"]):
        b_ms = baseline["regions"][region]["total_ms"]
        n_ms = batched["regions"].get(region, {}).get("total_ms", float("nan"))
        speedup = (b_ms / n_ms) if n_ms else float("nan")
        out[region] = {"baseline": b_ms, "batched": n_ms, "speedup": speedup}
        print(f"  {region:24s}: baseline={b_ms:8.1f}ms batched={n_ms:8.1f}ms => {speedup:5.2f}x")
    return out


@pytest.mark.slow
def test_profile_batched_on_real_model() -> None:
    """[Task 6 Step 4] Profile both objectives with batching ON; write profile_batched.json
    and print the speedup vs profile_baseline.json.

    Skip-not-fail when the checkpoint / source-run env vars are unset.  This is a
    MEASUREMENT (the keep/drop decision is the human's, Step 5) — it asserts only
    the breakdown sanity invariants, not a specific speedup.
    """
    ckpt_dir = _skip_unless_env(
        "FEATURE_GCG_D12_CKPT",
        "the d12 encdec checkpoint directory (containing model_054600.pt + meta_054600.json)",
    )
    run_dir = _skip_unless_env(
        "FEATURE_GCG_L12_RUN",
        "the d12_l12 source-run directory (containing generations/, concept_features.safetensors)",
    )

    gpu = wait_for_idle_local_gpu()
    device = f"cuda:{gpu}"
    minibatch_size = int(os.environ.get("FEATURE_GCG_D12_L12_MINIBATCH_SIZE", "4"))

    model, lm_members, cm_members, warm_start, cfg, artifact_label = _load_real_l12_workload(
        ckpt_dir, run_dir, minibatch_size
    )

    breakdown_lm = profile_optimize(
        model, lm_members, warm_start, cfg,
        objective_factory=_lm_factory_batched(device), device=device, repeats=3,
    )
    breakdown_cm = profile_optimize(
        model, cm_members, warm_start, cfg,
        objective_factory=_cm_factory_batched(device), device=device, repeats=3,
    )

    artifact = {
        "model": "d12_l12",
        "checkpoint_step": int(os.environ.get("FEATURE_GCG_D12_STEP", "54600")),
        "device": device,
        "n_members": minibatch_size,
        "member_artifact_label": artifact_label,
        "batched": True,
        "note": "Task-6 batched objective (sole production path); compare vs profile_baseline.json (pre-batching per-item baseline)",
        "lm_loss": breakdown_lm,
        "class_margin": breakdown_cm,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    batched_path = ARTIFACTS / "profile_batched.json"
    batched_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n[Task 6] Batched profile artifact written to: {batched_path}")

    _assert_breakdown(breakdown_lm)
    _assert_breakdown(breakdown_cm)

    baseline_path = ARTIFACTS / "profile_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        speedups = {
            "lm_loss": _speedup_table("lm_loss", baseline["lm_loss"], breakdown_lm),
            "class_margin": _speedup_table("class_margin", baseline["class_margin"], breakdown_cm),
        }
        (ARTIFACTS / "profile_batched_speedup.json").write_text(json.dumps(speedups, indent=2))
    else:
        print(f"\n[Task 6] {baseline_path} absent — run the Task-1 baseline first to compute speedups.")
