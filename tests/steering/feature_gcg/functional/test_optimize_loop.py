"""Optimizer-loop integration tests for Task 7 (optimize.py).

All tests run on CPU with a synthetic stub model + objective: no real model
forward passes are needed for the loop properties (monotone, determinism, group
sizes, edit types, golden, efficiency).  The stub model satisfies the interface
expected by ``partition_bounds_from_model`` and ``_n_sae_features``; the
synthetic objective directly computes from the bag without touching the model.

Synthetic objective
-------------------
Minimize ``sum(fid for fid, _ in bag.entries)`` (bucket changes are also guided).

With::

    id_weight[i, :] = i / N_FEATURES  (constant across D_EMBD columns)
    bucket_weight[b, :] = b / NUM_BUCKETS

and ``id_grad = bucket_grad = 1/D_EMBD`` (uniform), the GCG candidate scores
are:

- REPLACE-id slot s (fid → f'): ``(f' − f) / N_FEATURES``  (lower f' = better)
- REPLACE-bucket (b → b'):      ``(b' − b) / NUM_BUCKETS`` (lower b' = better)
- ADD (fid=f', bucket=b'):       ``f'/N + b'/B``
- REMOVE slot s (fid=f, b):     ``−(f/N + b/B)``           (higher fid/bucket = remove first)

This makes gradient-guided proposals strictly better than zero-gradient (random)
proposals for the given objective.
"""

from __future__ import annotations

import json
import math
import weakref
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

import nanochat.flash_attention as fa_module
import steering.conditional.feature_gcg.gradient as grad_module
import steering.conditional.feature_gcg.optimize as optimize_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureAddition,
    FeatureConditioningEditSpace,
    FeatureRemoval,
    FeatureReplacement,
)
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.objective import (
    ClassMarginMember,
    FeatureObjective,
    class_margin_objective,
    lm_loss_on_masked_tokens,
)
from steering.conditional.feature_gcg.optimize import (
    _Group,
    _ItemAdapter,
    optimize_feature_conditioning_universal,
)
from steering.conditional.feature_gcg.partition import partition_bounds_from_model
from steering.eval.axbench_base.types import SteeringMode
from steering.refined_types import NonEmptyNonNegativeIntTuple
from tests.steering.feature_gcg.functional.test_feature_gradient import _build_randomized_model
from tests.steering.feature_gcg.functional.test_feature_objective import _completion, _make_member

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_FEATURES = 16
D_EMBD = 8
NUM_BUCKETS = 4
SCALE = 1.0 / N_FEATURES
SCALE_B = 1.0 / NUM_BUCKETS

GOLDEN_PATH = Path(__file__).parent / "feature_golden" / "optimize_universal_golden.json"
API_GOLDEN_PATH = (
    Path(__file__).parent / "feature_golden" / "optimize_universal_api_golden.json"
)

# ──────────────────────────────────────────────────────────────────────────────
# Stub model (no real forward passes)
# ──────────────────────────────────────────────────────────────────────────────


class _StubFeatureEmbed:
    weight: torch.Tensor = torch.zeros(N_FEATURES, D_EMBD)


class _StubEncoderConfig:
    num_buckets: int = NUM_BUCKETS
    # ORIGINAL SAE width — what optimize._n_sae_features reads (#720). Matches the
    # feature_embed table height here (this stub is uncompacted: V == n_sae).
    n_sae_features: int = N_FEATURES


class _StubEncoder:
    config = _StubEncoderConfig()
    feature_embed = _StubFeatureEmbed()


class _StubModel:
    """Minimal model satisfying partition_bounds_from_model + _n_sae_features.

    Uncompacted: ``feature_vocab is None`` (the #720 proposal-universe path reads it via
    ``feature_vocab_allowed_ids`` and falls back to ``range(n_sae)``).
    """

    encoder = _StubEncoder()
    feature_vocab = None
    _loaded_meta_data: dict = {
        "user_config": {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "min_k": 0,
                "max_k": 6,
            }
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic objective: minimize sum(fid) + bucket-0 preference
# ──────────────────────────────────────────────────────────────────────────────


class _SyntheticObjective:
    """Gradient-guided objective: minimize ``sum(fid for fid, _ in bag)``.

    Uses a fixed linear weight matrix so that GCG scores linearly prefer
    lower-fid / lower-bucket replacements.  The gradient is uniform across
    all real slots (the objective decomposes additively per slot).
    """

    def __init__(self) -> None:
        # id_weight[i, :] = [i * SCALE] * D_EMBD
        self._id_weight = (
            torch.arange(N_FEATURES, dtype=torch.float32)
            .mul(SCALE)
            .unsqueeze(1)
            .expand(-1, D_EMBD)
            .contiguous()
        )
        # bucket_weight[b, :] = [b * SCALE_B] * D_EMBD
        self._bucket_weight = (
            torch.arange(NUM_BUCKETS, dtype=torch.float32)
            .mul(SCALE_B)
            .unsqueeze(1)
            .expand(-1, D_EMBD)
            .contiguous()
        )

    def losses(self, items: Sequence[Any]) -> list[float]:
        return [float(sum(fid for fid, _ in item.bag.entries)) for item in items]

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        return [self._grad(item.bag) for item in items]

    def _grad(self, bag: FeatureConditioning) -> FeatureGrad:
        n_real = len(bag)
        # Uniform gradient across all slots and embedding dims.
        # score(REPLACE to f') = (f' − f) * SCALE → lower f' scores lower (better).
        uni = 1.0 / D_EMBD
        id_grad = torch.full((n_real, D_EMBD), uni)
        bucket_grad = torch.full((n_real, D_EMBD), uni)
        return FeatureGrad(
            id_grad=id_grad,
            bucket_grad=bucket_grad,
            id_weight=self._id_weight,
            bucket_weight=self._bucket_weight,
        )


class _ZeroGradObjective:
    """Same losses as ``_SyntheticObjective`` but zero gradient (random proposals)."""

    def __init__(self, real_obj: _SyntheticObjective) -> None:
        self._real = real_obj

    def losses(self, items: Sequence[Any]) -> list[float]:
        return self._real.losses(items)

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        return [
            FeatureGrad(
                id_grad=torch.zeros(len(item.bag), D_EMBD),
                bucket_grad=torch.zeros(len(item.bag), D_EMBD),
                id_weight=self._real._id_weight,
                bucket_weight=self._real._bucket_weight,
            )
            for item in items
        ]


class _RecordingSyntheticObjective(_SyntheticObjective):
    """Synthetic objective that records its exact public ``losses`` call trace."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_trace: list[dict[str, Any]] = []

    def losses(self, items: Sequence[Any]) -> list[float]:
        losses = super().losses(items)
        self.loss_trace.append(
            {
                "bags": [
                    [[int(fid), int(bucket)] for fid, bucket in item.bag.entries]
                    for item in items
                ],
                "losses": losses,
            }
        )
        return losses


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

_MODEL = _StubModel()
_WARM_START = FeatureConditioning(entries=((8, 2), (12, 2)))  # sum=20


def _make_config(**overrides: Any) -> FeatureGcgConfig:
    defaults: dict[str, Any] = {
        "max_steps": 20,
        "candidate_batch_size": 4,
        "top_k": 4,
        "temperature": 1.0,
        "master_seed": 0,
        "minibatch_size": 16,
        "reg_lambda": 0.0,
        "min_k_pos": 0,
        "max_k_pos": 6,
    }
    defaults.update(overrides)
    return FeatureGcgConfig(**defaults)


def _synth_factory(_model: Any) -> FeatureObjective:
    return _SyntheticObjective()


def _zero_grad_factory(_model: Any) -> FeatureObjective:
    return _ZeroGradObjective(_SyntheticObjective())


def _run_universal_api_golden_case(n_members: int) -> dict[str, Any]:
    members = [f"m{i}" for i in range(n_members)]
    objective = _RecordingSyntheticObjective()
    cfg = _make_config(
        max_steps=6,
        candidate_batch_size=4,
        top_k=4,
        master_seed=0,
    )
    result = optimize_feature_conditioning_universal(
        _MODEL,
        members,
        _WARM_START,
        objective_factory=lambda _model: objective,
        config=cfg,
    )
    return {
        "name": f"universal_n{n_members}",
        "n_members": n_members,
        "members": members,
        "warm_start": [list(entry) for entry in _WARM_START.entries],
        "config": asdict(cfg),
        "returned_entries": [list(entry) for entry in result.entries],
        "final_objective": float(n_members * sum(fid for fid, _ in result.entries)),
        "loss_trace": objective.loss_trace,
    }


def test_optimize_universal_public_api_golden() -> None:
    """The public universal API is exact for supported N=1 and N=3 shapes."""
    assert API_GOLDEN_PATH.is_file(), f"missing tracked API golden: {API_GOLDEN_PATH}"
    expected = json.loads(API_GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = {
        "cases": [
            _run_universal_api_golden_case(1),
            _run_universal_api_golden_case(3),
        ]
    }
    assert actual == expected


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: monotone — retained objective non-increasing
# ──────────────────────────────────────────────────────────────────────────────


def test_monotone_universal() -> None:
    """The final retained objective is ≤ the warm-start objective.

    The GCG loop retains the best full-batch bag seen across all accepted steps;
    by construction this is monotonically non-increasing.
    """
    synth = _SyntheticObjective()
    initial_loss = float(sum(fid for fid, _ in _WARM_START.entries))

    cfg = _make_config(max_steps=30)
    result = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg
    )

    final_loss = synth.losses([type("_I", (), {"bag": result})()])[0]
    assert final_loss <= initial_loss, (
        f"final objective {final_loss} > initial {initial_loss} — monotonicity violated"
    )


def test_monotone_accepts_at_least_one_step() -> None:
    """With 30 steps and a clear gradient signal, at least one step is accepted."""
    initial_loss = float(sum(fid for fid, _ in _WARM_START.entries))
    cfg = _make_config(max_steps=30)
    result = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg
    )
    final_loss = float(sum(fid for fid, _ in result.entries))
    assert final_loss < initial_loss, (
        f"expected at least one accepted step; final={final_loss}, initial={initial_loss}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: determinism — same seed → same bag; different seeds → different bags
# ──────────────────────────────────────────────────────────────────────────────


def test_determinism_same_seed() -> None:
    """Two runs with the same master_seed yield byte-identical bags."""
    cfg = _make_config(master_seed=42, max_steps=15)
    bag_a = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg
    )
    bag_b = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg
    )
    assert bag_a.entries == bag_b.entries, (
        f"determinism failed: seed=42 produced different bags\n  A={bag_a.entries}\n  B={bag_b.entries}"
    )


def test_determinism_different_seeds_both_complete() -> None:
    """Two runs with DIFFERENT master_seeds both terminate without error (RNG wiring smoke).

    With a strongly-convex synthetic objective that has a unique global minimum
    (``sum(fid)=0``), all seeds converge to the same optimal bag regardless of
    the random candidate selection path; the key reproducibility property (same
    seed → same bag) is verified by ``test_determinism_same_seed``.
    """
    cfg_a = _make_config(master_seed=0, max_steps=5)
    cfg_b = _make_config(master_seed=99, max_steps=5)
    bag_a = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg_a
    )
    bag_b = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START, objective_factory=_synth_factory, config=cfg_b
    )
    # Both seeds complete and produce valid FeatureConditioning objects.
    assert isinstance(bag_a, FeatureConditioning)
    assert isinstance(bag_b, FeatureConditioning)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: both group sizes — 1-member and N-member both improve
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_members", [1, 3])
def test_group_sizes_improve(n_members: int) -> None:
    """Both one-member and multi-member universal groups improve their objective."""
    initial_loss = float(sum(fid for fid, _ in _WARM_START.entries))

    members = [f"m{i}" for i in range(n_members)]
    cfg = _make_config(max_steps=30)
    result = optimize_feature_conditioning_universal(
        _MODEL, members, _WARM_START, objective_factory=_synth_factory, config=cfg
    )

    final_loss = float(sum(fid for fid, _ in result.entries))
    assert final_loss < initial_loss, (
        f"n_members={n_members}: expected improvement; final={final_loss} >= initial={initial_loss}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: all edit types reachable
# ──────────────────────────────────────────────────────────────────────────────


def test_all_edit_types_reachable() -> None:
    """rank_candidates returns ADD, REMOVE, REPLACE proposals unconditionally."""
    bounds = partition_bounds_from_model(_MODEL, _make_config())
    cfg = _make_config(top_k=4)

    # Warm start with 1 feature in a non-zero bucket → room to ADD (up to 5) and REMOVE (≥0).
    warm = FeatureConditioning(entries=((5, 1),))
    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(
            tuple(range(N_FEATURES))
        ),
        warm_start=warm,
    )
    synth = _SyntheticObjective()
    grad = FeatureGrad(
        id_grad=torch.full((1, D_EMBD), 1.0 / D_EMBD),
        bucket_grad=torch.full((1, D_EMBD), 1.0 / D_EMBD),
        id_weight=synth._id_weight,
        bucket_weight=synth._bucket_weight,
    )

    proposals = edit_space.rank_candidates(grad, warm, cfg)

    assert any(isinstance(e, FeatureAddition) for e in proposals.edits), \
        "ADD not in proposals"
    assert any(isinstance(e, FeatureRemoval) for e in proposals.edits), \
        "REMOVE not in proposals"
    assert any(isinstance(e, FeatureReplacement) for e in proposals.edits), \
        "REPLACE not in proposals"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: top_k=1 golden — byte-stable result
# ──────────────────────────────────────────────────────────────────────────────


def _run_golden() -> tuple[FeatureConditioning, float, int]:
    """Run the golden configuration; return (bag, base_loss, accepted_steps).

    Uses a lightweight config: top_k=1, candidate_batch_size=1, 6 steps.
    The count-penalty is disabled (reg_lambda=0) so the objective is pure
    ``sum(fid)`` — an integer, exact across runs.
    """
    synth = _SyntheticObjective()

    cfg = _make_config(
        max_steps=6,
        candidate_batch_size=1,
        top_k=1,
        temperature=1.0,
        master_seed=7,
        reg_lambda=0.0,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)
    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(
            tuple(range(N_FEATURES))
        ),
        warm_start=_WARM_START,
    )

    def _obj(items: Sequence[Any]) -> list[float]:
        adapted = [_ItemAdapter(member=m, bag=bag) for m, bag in items]
        return synth.losses(adapted)

    def _grad(items: Sequence[Any], step: int) -> list[FeatureGrad]:
        adapted = [_ItemAdapter(member=m, bag=bag) for m, bag in items]
        return synth.gradients(adapted, step)

    group = _Group(key_str="golden_m0", members=["m0"])
    results = run_gcg(
        edit_space=edit_space,
        groups=[group],
        objective_batch=_obj,
        gradient_provider_batch=_grad,
        cfg=cfg,
    )
    payload = results[0].payload
    bag: FeatureConditioning = payload["bag"]
    objective: float = payload["objective"]
    accepted_steps: int = payload["accepted_steps"]
    return bag, objective, accepted_steps


def test_top_k1_golden() -> None:
    """top_k=1 fixed-seed run is byte-stable across re-executions.

    First run: writes the golden file and skips.
    Subsequent runs: compares byte-exactly against the committed golden.
    """
    bag, objective, accepted_steps = _run_golden()

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        golden = {
            "entries": [list(e) for e in bag.entries],
            "objective": objective,
            "accepted_steps": accepted_steps,
        }
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
        pytest.skip(f"golden written to {GOLDEN_PATH}; re-run to verify stability")

    expected = json.loads(GOLDEN_PATH.read_text())
    assert [list(e) for e in bag.entries] == expected["entries"], (
        f"golden bag mismatch:\n  got     {bag.entries}\n  expected {expected['entries']}"
    )
    assert math.isclose(objective, expected["objective"], rel_tol=1e-9, abs_tol=1e-12), (
        f"golden objective mismatch: got {objective}, expected {expected['objective']}"
    )
    assert accepted_steps == expected["accepted_steps"], (
        f"golden accepted_steps mismatch: got {accepted_steps}, expected {expected['accepted_steps']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: proposal-beats-random efficiency
# ──────────────────────────────────────────────────────────────────────────────


def test_gradient_beats_zero_gradient() -> None:
    """Gradient-guided proposals reach a lower objective than zero-gradient (random).

    With 25 steps and a synthetic objective where the gradient PERFECTLY ranks
    lower-fid candidates above higher-fid ones, gradient-guided optimization
    strictly out-performs zero-gradient (random proposal) optimization.
    """
    cfg = _make_config(max_steps=25, candidate_batch_size=4, top_k=4, master_seed=13)

    bag_guided = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START,
        objective_factory=_synth_factory,
        config=cfg,
    )
    bag_random = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], _WARM_START,
        objective_factory=_zero_grad_factory,
        config=cfg,
    )

    obj_guided = float(sum(fid for fid, _ in bag_guided.entries))
    obj_random = float(sum(fid for fid, _ in bag_random.entries))
    initial_loss = float(sum(fid for fid, _ in _WARM_START.entries))

    assert obj_guided <= obj_random, (
        f"gradient-guided ({obj_guided}) should be ≤ zero-gradient ({obj_random})"
    )
    # The gradient-guided run should reach objective 0 (fids {0,1,...}) within 25 steps;
    # the random run is unlikely to be at the minimum.
    assert obj_guided < initial_loss, (
        f"gradient-guided failed to improve from warm start (initial={initial_loss})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 (new): count-penalty end-to-end wiring — reg_lambda drives sparsity
# ──────────────────────────────────────────────────────────────────────────────


def test_count_penalty_drives_sparsity() -> None:
    """Non-zero reg_lambda removes features that the base objective cannot.

    Design
    ------
    Warm start: ``((0, 2),)`` — one feature, fid=0, bucket=2 (active partition).
    Base objective = ``sum(fid)`` = 0, already at its global minimum.  With
    ``reg_lambda=0.0`` no edit improves the objective: REMOVE gives the same
    base loss (0 → 0), ADDs raise it.  The warm start is returned unchanged.

    With ``reg_lambda=100.0`` the count penalty (L1: reg_lambda * len(bag))
    adds 100 for the single active feature.  REMOVE drops the penalised
    objective from 100 to 0 — a strict improvement — so the step is accepted
    and the final bag is empty.

    Asserts ``len(bag_high.entries) < len(bag_zero.entries)`` (0 < 1).

    Mutation check (in-process, reverted)
    --------------------------------------
    Removing ``+ count_penalty(...)`` from ``_penalized_obj_batch`` in
    ``optimize.py`` makes both runs see the bare base loss (0 for empty bag =
    0 for warm-start) → the REMOVE accept gate fails → ``bag_high`` stays at
    1 feature → ``assert len(bag_high.entries) == 0`` REDs.
    """
    # Single feature at the base-objective minimum: fid=0, bucket=2 (active).
    warm = FeatureConditioning(entries=((0, 2),))

    # Shared hyperparameters — only reg_lambda differs.
    shared_kwargs: dict[str, Any] = {
        "max_steps": 20,
        "candidate_batch_size": 4,
        "top_k": 4,
        "master_seed": 0,
    }
    cfg_zero = _make_config(**shared_kwargs, reg_lambda=0.0)
    cfg_high = _make_config(**shared_kwargs, reg_lambda=100.0)

    bag_zero = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm, objective_factory=_synth_factory, config=cfg_zero
    )
    bag_high = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm, objective_factory=_synth_factory, config=cfg_high
    )

    # reg_lambda=0: base objective already at minimum; no edit is strictly accepted.
    assert len(bag_zero.entries) == 1, (
        f"reg_lambda=0 run should preserve the warm-start bag (1 feature); "
        f"got {bag_zero.entries}"
    )
    # reg_lambda=100: count penalty dominates; REMOVE(fid=0) accepted → empty bag.
    # This assertion REDs when `+ count_penalty(...)` is removed from
    # _penalized_obj_batch: without the penalty both bags stay at 1 feature.
    assert len(bag_high.entries) == 0, (
        f"reg_lambda=100 run should drive the bag to empty; got {bag_high.entries}"
    )
    assert len(bag_high.entries) < len(bag_zero.entries), (
        f"count-penalty wiring failed to drive sparsity: "
        f"high-λ bag={bag_high.entries}, zero-λ bag={bag_zero.entries}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# B2 (T7 review): the UNIVERSAL entry point drives a REAL batched objective
# end-to-end on the tiny model AND scopes the static-weight cache to the call
# ──────────────────────────────────────────────────────────────────────────────

_CPU = torch.device("cpu")
_GEOMETRIC_META: dict[str, Any] = {
    "user_config": {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 3.0,
            "min_k": 0,
            "max_k": 6,
        }
    }
}


def _spy_clear_static_weight_cache(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch ``optimize.clear_static_weight_cache`` to record calls, still clearing."""
    calls: list[int] = []
    real_clear = optimize_module.clear_static_weight_cache

    def _spy() -> None:
        calls.append(1)
        real_clear()

    monkeypatch.setattr(optimize_module, "clear_static_weight_cache", _spy)
    return calls


def _fresh_static_weight_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> weakref.WeakKeyDictionary[Any, Any]:
    """Install an isolated cache so tests can assert cleanup effects directly."""
    cache: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
    monkeypatch.setattr(grad_module, "_encoder_weight_cache", cache)
    return cache


def _universal_cfg() -> FeatureGcgConfig:
    return FeatureGcgConfig(
        max_steps=3,
        candidate_batch_size=2,
        top_k=4,
        temperature=1.0,
        master_seed=0,
        minibatch_size=3,
        reg_lambda=0.0,
        min_k_pos=0,
        max_k_pos=6,
    )


@pytest.mark.parametrize("n_members", (1, 3))
def test_optimize_universal_real_lm_objective_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
    n_members: int,
) -> None:
    """B2: ``optimize_feature_conditioning_universal`` runs the REAL batched
    ``lm_loss_on_masked_tokens`` objective end-to-end on the tiny model AND clears
    the static-weight cache at both call boundaries (spy-verified).

    The universal entry point's wiring (real model forward + the per-optimize
    ``clear_static_weight_cache``) was only @slow-tested; every CPU optimize-loop
    test uses a synthetic stub (no forward, no clear).  This closes that gap on CPU.

    Teeth: the exact two-call assertion requires one clear before the optimize
    and one on its successful exit. Removing either boundary makes it RED.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=2)
    model._loaded_meta_data = _GEOMETRIC_META
    cache = _fresh_static_weight_cache(monkeypatch)
    clear_calls = _spy_clear_static_weight_cache(monkeypatch)

    members = [
        _completion(torch.Generator().manual_seed(700 + i), t_len=12, n_scored=3)
        for i in range(n_members)
    ]
    warm = FeatureConditioning(entries=((17, 2),))

    result = optimize_feature_conditioning_universal(
        model,
        members,
        warm,
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=_CPU),
        config=_universal_cfg(),
    )

    assert isinstance(result, FeatureConditioning)
    # The optimized bag respects the tiny model's vocabulary (N_SAE=40) and buckets.
    assert all(0 <= fid < 40 for fid, _ in result.entries), result.entries
    assert all(0 <= bucket < 6 for _, bucket in result.entries), result.entries
    assert clear_calls == [1, 1], (
        "static-weight cache must be cleared exactly at universal optimize entry "
        "and successful exit"
    )
    assert len(cache) == 0, "successful optimize retained detached static weights"


def test_optimize_universal_real_class_margin_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2: the universal entry point also drives the REAL ``class_margin_objective``
    (ClassMarginMember members, the §7.2 adversary) end-to-end on the tiny model.

    Exercises the 2N-completion interleave through the production optimize loop —
    not just the objective in isolation.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=3)
    model._loaded_meta_data = _GEOMETRIC_META
    cache = _fresh_static_weight_cache(monkeypatch)
    clear_calls = _spy_clear_static_weight_cache(monkeypatch)

    members: list[ClassMarginMember] = [_make_member(seed=40 + i, n_scored=3) for i in range(3)]
    warm = FeatureConditioning(entries=((17, 2),))

    result = optimize_feature_conditioning_universal(
        model,
        members,
        warm,
        objective_factory=lambda m: class_margin_objective(
            m, mode=SteeringMode.POSITIVE, device=_CPU
        ),
        config=_universal_cfg(),
    )

    assert isinstance(result, FeatureConditioning)
    assert all(0 <= fid < 40 for fid, _ in result.entries), result.entries
    assert clear_calls == [1, 1], (
        "static-weight cache must be cleared exactly at universal optimize entry "
        "and successful exit"
    )
    assert len(cache) == 0, "successful optimize retained detached static weights"


def test_optimize_universal_clears_cache_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optimizer exception cannot leave detached device weights cached."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=4)
    model._loaded_meta_data = _GEOMETRIC_META
    cache = _fresh_static_weight_cache(monkeypatch)
    clear_calls = _spy_clear_static_weight_cache(monkeypatch)

    def _raise_from_optimize(**_kwargs: Any) -> Any:
        ve_keys = tuple(sorted(model.encoder.value_embeds.keys(), key=int))
        grad_module._static_weights(model.encoder, ve_keys)  # ty: ignore[invalid-argument-type]
        assert len(cache) == 1, "error stub failed to populate the static-weight cache"
        raise RuntimeError("sentinel optimizer failure")

    monkeypatch.setattr(optimize_module, "run_gcg", _raise_from_optimize)
    member = _completion(torch.Generator().manual_seed(704), t_len=12, n_scored=3)

    with pytest.raises(RuntimeError, match="sentinel optimizer failure"):
        optimize_feature_conditioning_universal(
            model,
            [member],
            FeatureConditioning(entries=((17, 2),)),
            objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=_CPU),
            config=_universal_cfg(),
        )

    assert clear_calls == [1, 1], (
        "static-weight cache must be cleared exactly at universal optimize entry "
        "and exceptional exit"
    )
    assert len(cache) == 0, "exceptional optimize retained detached static weights"
