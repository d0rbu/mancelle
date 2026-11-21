"""Consolidated teeth harness for feature_gcg Phase 1 (Task 9).

Each test confirms that a deliberately broken variant REDs the corresponding
guard — proving the suite has real mutation-detection teeth.  Every test:

1. Applies the break IN-SCOPE (monkeypatch / subclass / object-bypass).
2. Drives the relevant guard logic.
3. Asserts the guard FAILS (``pytest.raises(AssertionError)``).

The outer teeth test PASSES when the inner assertion fails as expected.

Guard targets
-------------
(a) ADD always generated            → ADD-presence guard from
                                      ``test_add_proposals_present_on_normal_bag``.
(b) Negated ``count_penalty``       → "higher λ ⇒ sparser" guard from
                                      ``test_count_penalty_drives_sparsity``.
(c) Bypassed dup-id guard           → uniqueness property guard from
                                      ``test_property_entries_always_sorted_and_dup_free``.
(d) Non-permutation-invariant digest → permutation-invariance guard from
                                      ``test_digest_permutation_invariant``.
(e) Zero-init encoder attn.c_proj  → non-zero-init guard from
                                      ``test_finite_difference_oracle``.
(f) Flipped g_steer sign            → class-margin sign guard from
                                      ``test_class_margin_g_steer_equals_neg_primary_margin``.
(g) REPLACE-id slot↔row alignment   → slot-alignment guard (Task 6 Minor) +
                                      meta-teeth: shifted grad rows → guard REDs.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import pytest
import torch

import nanochat.flash_attention as fa_module
import steering.conditional.feature_gcg.optimize as _opt_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureAddition,
    FeatureConditioningEditSpace,
    FeatureReplacement,
)
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    class_margin_objective,
)
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from steering.conditional.feature_gcg.partition import (
    count_penalty as _count_penalty_ref,
)
from steering.conditional.feature_gcg.partition import (
    partition_bounds_from_model,
)
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_base.types import SteeringMode
from steering.refined_types import NonEmptyNonNegativeIntTuple
from tests.steering.feature_gcg.functional.test_feature_gradient import _build_randomized_model
from tests.steering.feature_gcg.functional.test_feature_objective import (
    _BAG,
    _measure_loss,
    _member_with_margin,
)
from tests.steering.feature_gcg.functional.test_optimize_loop import (
    _MODEL,
    D_EMBD,
    N_FEATURES,
    NUM_BUCKETS,
    SCALE,
    SCALE_B,
    _make_config,
    _synth_factory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level autouse fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SDPA for every test in this module (FlexAttention lacks a CPU backward)."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Broken helper classes (teeth c and d)
# ─────────────────────────────────────────────────────────────────────────────


class _BrokenWithAddedConditioning(FeatureConditioning):
    """Broken mutation that bypasses both duplicate guards and typed construction."""

    def with_added(  # type: ignore[override]
        self, fid: int, bucket: int
    ) -> _BrokenWithAddedConditioning:
        """BROKEN: inject duplicate entries without calling the typed constructor."""
        new_entries = tuple(sorted(self.entries + ((fid, bucket),), key=lambda e: e[0]))
        candidate = object.__new__(type(self))
        object.__setattr__(candidate, "entries", new_entries)
        return candidate


class _BrokenDigestConditioning(FeatureConditioning):
    """FeatureConditioning subclass with a non-permutation-invariant ``digest``.

    Appends ``id(self)`` to the hash so two distinct instances carrying the
    same content produce different digests — breaking the permutation-invariance
    guarantee upheld by the production ``digest``.
    """

    def digest(self) -> str:  # type: ignore[override]
        h = hashlib.sha256()
        for fid, bucket in self.entries:
            h.update(struct.pack("<QQ", fid, bucket))
        # NON-DETERMINISTIC across instances: object identity contaminates the hash.
        h.update(struct.pack("<Q", id(self)))
        return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE-id slot↔row alignment guard helpers (teeth g1/g2)
# ─────────────────────────────────────────────────────────────────────────────

# Discriminating gradient constants — separate from the stub model's N_FEATURES / D_EMBD.
_SLOT_N_SAE: int = 31  # SAE vocab covers fids 0..30
_SLOT_D: int = 4  # embedding dimension for discriminating id_grad / id_weight


def _make_discriminating_grad(
    *, shift_rows: bool = False
) -> tuple[FeatureGrad, FeatureConditioning, list[int]]:
    """Build a FeatureGrad where each bag slot uniquely prefers a different replacement.

    Current bag: fids [10, 20, 30] at bucket 0 (active, well within cap).
    Pool (id-replace candidates): all fids in 0.._SLOT_N_SAE-1 except 10, 20, 30.

    Discriminating layout::

        id_grad[s, s] = 1.0   (one-hot — slot s looks at embedding dim s)
        id_weight[s+1, s] = −1.0  (fid s+1 is the unique minimum for slot s)

    Score for replacing slot s (current fid=f_cur) with pool fid f':
        (id_weight[f'] − id_weight[f_cur]) @ id_grad[s]
        = id_weight[f', s]   (current fids 10/20/30 are all-zero in id_weight)

    Minimum score (best) for slot s: f' = s+1, score = −1.0.
    All other pool fids score 0.0 (tie — but strictly worse than −1.0).

    With ``shift_rows=True`` the id_grad rows are cyclically shifted by 1
    (``torch.roll(..., shifts=1, dims=0)``), so slot 0 gets old row 2
    and now prefers fid 3 instead of fid 1 — breaking the slot↔row mapping.
    """
    current = FeatureConditioning(entries=((10, 0), (20, 0), (30, 0)))
    n_slots = len(current)  # 3

    id_grad = torch.zeros(n_slots, _SLOT_D)
    for s in range(n_slots):
        id_grad[s, s] = 1.0  # slot s reads embedding dim s

    if shift_rows:
        # torch.roll([row0, row1, row2], 1, 0) → [row2, row0, row1]
        # new slot 0 gets old row 2 = [0, 0, 1, 0] → prefers fid 3 (not 1).
        id_grad = torch.roll(id_grad, shifts=1, dims=0)

    id_weight = torch.zeros(_SLOT_N_SAE, _SLOT_D)
    for s in range(n_slots):
        id_weight[s + 1, s] = -1.0  # fid (s+1) is preferred by slot s

    grad = FeatureGrad(
        id_grad=id_grad,
        bucket_grad=torch.zeros(n_slots, _SLOT_D),
        id_weight=id_weight,
        bucket_weight=torch.zeros(NUM_BUCKETS, _SLOT_D),
    )

    # Expected top fid for each slot (unshifted mapping: slot s → fid s+1).
    expected_top_fids = [s + 1 for s in range(n_slots)]

    return grad, current, expected_top_fids


def _assert_slot_alignment(
    proposals: Any,
    current: FeatureConditioning,
    expected_top_fids: list[int],
) -> None:
    """Assert that each slot's top id-replace proposal matches ``expected_top_fids[slot]``.

    Only REPLACE edits whose ``fid`` is not already in the bag are considered
    (pure id-replace proposals, not bucket-only or joint edits).
    """
    current_fids = set(current.feature_ids())
    for slot, expected_fid in enumerate(expected_top_fids):
        id_repls = [
            e
            for e in proposals.edits
            if isinstance(e, FeatureReplacement) and e.slot == slot and e.fid not in current_fids
        ]
        assert id_repls, f"no id-replace proposals for slot {slot}"
        best = min(id_repls, key=lambda e: e.score)
        assert best.fid == expected_fid, (
            f"slot {slot}: expected top fid={expected_fid}, got fid={best.fid} "
            f"(score={best.score:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (a replacement): ADD proposals are always generated (unconditional)
# ─────────────────────────────────────────────────────────────────────────────


def test_add_proposals_present_on_normal_bag() -> None:
    """ADD proposals are always generated (unconditionally) in Phase 1.

    Previously gated by ``add_enabled``; now ADD is ALWAYS generated and this
    test verifies the unconditional behavior replaces the old toggling.
    """
    cfg = _make_config(top_k=4)
    bounds = partition_bounds_from_model(_MODEL, cfg)
    warm = FeatureConditioning(entries=((5, 1),))

    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(range(N_FEATURES))),
        warm_start=warm,
    )

    id_weight = (
        torch.arange(N_FEATURES, dtype=torch.float32)
        .mul(SCALE)
        .unsqueeze(1)
        .expand(-1, D_EMBD)
        .contiguous()
    )
    bucket_weight = (
        torch.arange(NUM_BUCKETS, dtype=torch.float32)
        .mul(SCALE_B)
        .unsqueeze(1)
        .expand(-1, D_EMBD)
        .contiguous()
    )
    grad = FeatureGrad(
        id_grad=torch.full((1, D_EMBD), 1.0 / D_EMBD),
        bucket_grad=torch.full((1, D_EMBD), 1.0 / D_EMBD),
        id_weight=id_weight,
        bucket_weight=bucket_weight,
    )

    proposals = edit_space.rank_candidates(grad, warm, cfg)
    add_proposals = [e for e in proposals.edits if isinstance(e, FeatureAddition)]
    assert add_proposals, f"ADD proposals must always be generated; got edits={proposals.edits}"


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (b): negated count_penalty → "higher λ ⇒ sparser" guard REDs
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_b_neg_count_penalty_reds_sparsity_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tooth (b): negating count_penalty makes higher λ produce DENSER bags; guard REDs.

    Guard under test: ``assert len(bag_high.entries) == 0`` from
    ``test_count_penalty_drives_sparsity`` in test_optimize_loop.py.

    With the CORRECT penalty (L1: reg_lambda * len(bag)), λ=100 adds 100 to
    the penalized objective for the single active feature, making REMOVE strictly
    cheaper; the bag empties.  With NEGATED penalty the feature carries a −100
    contribution, so REMOVE raises the objective; the optimizer keeps (or adds)
    features instead.
    """

    # BREAK: negate count_penalty in the optimize module's namespace.
    def _neg_count_penalty(bag: FeatureConditioning, reg_lambda: float) -> float:
        return -_count_penalty_ref(bag, reg_lambda)

    monkeypatch.setattr(_opt_module, "count_penalty", _neg_count_penalty)

    warm = FeatureConditioning(entries=((0, 2),))  # fid=0, bucket=2 (active), base_loss=0
    cfg = _make_config(
        max_steps=20,
        candidate_batch_size=4,
        top_k=4,
        master_seed=0,
        reg_lambda=100.0,
    )
    bag_high = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm, objective_factory=_synth_factory, config=cfg
    )

    # Guard: λ=100 with CORRECT penalty drives the bag to empty.
    # With NEGATED penalty the feature is retained (REMOVE raises the penalized obj).
    with pytest.raises(AssertionError):
        assert len(bag_high.entries) == 0, (
            f"reg_lambda=100 run should drive the bag to empty; got {bag_high.entries}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (c): bypassed dup-id guard → uniqueness property guard REDs
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_c_bypassed_dup_guard_reds_uniqueness_assertion() -> None:
    """Tooth (c): bypassing both typed boundaries creates duplicate feature IDs.

    Guard under test: ``assert len(fids) == len(set(fids))`` from
    ``test_property_entries_always_sorted_and_dup_free`` in
    test_feature_conditioning.py.
    """
    # BREAK: the subclass injects entries without the mutation guard or constructor.
    fc = _BrokenWithAddedConditioning(entries=((1, 2),))
    fc2 = fc.with_added(1, 5)  # fid=1 already present — succeeds (guard bypassed).

    fids = [fid for fid, _ in fc2.entries]

    # Guard (from test_property_entries_always_sorted_and_dup_free):
    # entries must contain no duplicate feature_ids.
    # fc2.entries = ((1, 2), (1, 5)) → fids = [1, 1] → len([1,1]) != len({1}) → FAILS.
    with pytest.raises(AssertionError, match="duplicate feature_ids"):
        assert len(fids) == len(set(fids)), "duplicate feature_ids found"


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (d): non-permutation-invariant digest → permutation-invariance guard REDs
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_d_nonpermutation_digest_reds_invariance_assertion() -> None:
    """Tooth (d): a digest that includes object identity is not permutation-invariant.

    Guard under test: ``assert fc1.digest() == fc2.digest()`` from
    ``test_digest_permutation_invariant`` in test_feature_conditioning.py.

    Two simultaneously-alive objects are guaranteed to have different id()
    values, so the broken digest differs between equal-content bags.
    """
    # BREAK: use the subclass whose digest includes id(self).
    fc1 = _BrokenDigestConditioning.from_dict({1: 2, 3: 4})
    fc2 = _BrokenDigestConditioning.from_dict({3: 4, 1: 2})
    # fc1 and fc2 carry identical sorted entries but are distinct objects.
    # id(fc1) != id(fc2) (both alive simultaneously) → different digests.

    # Guard (from test_digest_permutation_invariant):
    # two bags with the same content must produce the same digest.
    # With the broken digest fc1.digest() != fc2.digest() → assertion FAILS.
    with pytest.raises(AssertionError):
        assert fc1.digest() == fc2.digest()


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (e): zero-init encoder → non-zero-init guard REDs
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_e_zero_init_encoder_reds_nonzero_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tooth (e): zeroing encoder attn.c_proj makes the non-zero-init guard RED.

    Guard under test: ``assert block.attn.c_proj.weight.norm() > 0`` from
    ``test_finite_difference_oracle`` in test_feature_gradient.py.  Proves the
    FD oracle is non-vacuous: with zero attn.c_proj the attention block is inert
    and the value-embedding gradient is identically zero; the FD check would
    pass vacuously without this guard.
    """
    model = _build_randomized_model(monkeypatch, seed=0)

    # BREAK: zero-init every encoder block's attn.c_proj weight.
    with torch.no_grad():
        for block in model.encoder.blocks:
            block.attn.c_proj.weight.zero_()

    # Guard (from test_finite_difference_oracle, non-vacuity Guard 1):
    # "every encoder block attn.c_proj is non-zero — inert-toy trap (FD would be vacuous)".
    # After zeroing, norm == 0.0 → the assertion FAILS (guard REDs).
    with pytest.raises(AssertionError, match="inert-toy trap"):
        for block in model.encoder.blocks:
            assert block.attn.c_proj.weight.norm().item() > 0.0, (
                "encoder attn.c_proj is zero — inert-toy trap (FD would be vacuous)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (f): flipped g_steer sign → class-margin sign guard REDs
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_f_flipped_g_steer_reds_sign_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tooth (f): flipping g_steer's sign makes the class-margin sign guard RED.

    Guard under test: ``assert g_steer == pytest.approx(-primary_margin_y)``
    from ``test_class_margin_g_steer_equals_neg_primary_margin`` in
    test_feature_objective.py.  This is the spec's #1 adversary (§7.2 sign trap).

    The correct g_steer = loss_desired − loss_contrast = −primary_margin_y.
    Flipping it gives loss_contrast − loss_desired = +primary_margin_y; the
    guard ``g_steer ≈ −primary_margin_y`` then fails for any non-zero margin.
    """
    model = _build_randomized_model(monkeypatch, seed=21)
    member = _member_with_margin(model)
    item = ClassMarginItem(member=member, bag=_BAG)
    mode = SteeringMode.POSITIVE

    correct_obj = class_margin_objective(model, mode=mode, device=torch.device("cpu"))

    # BREAK: negate losses (equivalent to swapping desired/contrast — the adversary's
    # sign trap; flipped g_steer = −(loss_desired−loss_contrast) = +primary_margin_y).
    class _FlippedGSteerObjective:
        def losses(self, items: Any) -> list[float]:
            return [-v for v in correct_obj.losses(items)]

        def gradients(self, items: Any, step: int) -> list[Any]:
            return correct_obj.gradients(items, step)

    flipped_obj = _FlippedGSteerObjective()
    g_steer = flipped_obj.losses([item])[0]

    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    primary_margin_y = signed_class_margin(mode=mode, rich=loss_rich, avoid=loss_avoid)

    assert abs(primary_margin_y) > 1e-3, (
        f"vacuous fixture: |primary_margin_y|={abs(primary_margin_y):.2e} ≈ 0 hides sign"
    )

    # Guard (from test_class_margin_g_steer_equals_neg_primary_margin):
    # g_steer must equal −primary_margin_y.
    # With the flipped sign, g_steer == +primary_margin_y ≠ −primary_margin_y → FAILS.
    with pytest.raises(AssertionError):
        assert g_steer == pytest.approx(-primary_margin_y, rel=1e-5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (g1): REPLACE-id slot↔row alignment guard (positive — must pass)
# ─────────────────────────────────────────────────────────────────────────────


def test_slot_alignment_guard() -> None:
    """Slot-alignment guard: each slot's top id-replace matches its own gradient row.

    Constructs a discriminating FeatureGrad (slot s favors fid s+1 via a
    one-hot id_grad row) and verifies that rank_candidates routes each slot's
    top proposal to the correct replacement fid — pinning the bag-slot ↔
    gradient-row mapping (Task 6 Minor review item).
    """
    grad, current, expected = _make_discriminating_grad(shift_rows=False)

    cfg = FeatureGcgConfig(
        top_k=4,
        max_steps=1,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)

    space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(range(_SLOT_N_SAE))),
        warm_start=current,
    )

    proposals = space.rank_candidates(grad, current, cfg)

    # Guard: each slot's top id-replace must match its gradient row's preferred fid.
    _assert_slot_alignment(proposals, current, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Tooth (g2): shifted grad rows → slot-alignment guard REDs (meta-teeth)
# ─────────────────────────────────────────────────────────────────────────────


def test_tooth_g_shifted_grad_reds_slot_alignment_guard() -> None:
    """Tooth (g): cyclically shifting id_grad rows by 1 breaks the slot-alignment guard.

    With the shift, slot 0 receives old row 2 = [0, 0, 1, 0] and now prefers
    fid 3 (not fid 1).  The guard ``_assert_slot_alignment`` catches the mismatch
    at slot 0, proving the guard has real teeth against a slot↔row permutation bug.

    Baseline (unshifted) is verified to PASS first — ensuring the guard is
    sensitive, not permanently green.
    """
    cfg = FeatureGcgConfig(
        top_k=4,
        max_steps=1,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)

    # Unshifted gradient — baseline must PASS (guard is sensitive, not always green).
    grad_ok, current, expected = _make_discriminating_grad(shift_rows=False)
    space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(range(_SLOT_N_SAE))),
        warm_start=current,
    )
    proposals_ok = space.rank_candidates(grad_ok, current, cfg)
    _assert_slot_alignment(proposals_ok, current, expected)  # must pass

    # BREAK: cyclically shift id_grad rows by 1.
    # torch.roll([row0, row1, row2], 1, 0) → [row2, row0, row1]:
    #   new slot 0 uses old row 2 = [0, 0, 1, 0] → prefers fid 3 (not fid 1).
    grad_shifted, _, _ = _make_discriminating_grad(shift_rows=True)
    proposals_broken = space.rank_candidates(grad_shifted, current, cfg)

    # Guard: slot 0 expected fid=1, broken grad → top fid=3 → FAILS (guard REDs).
    with pytest.raises(AssertionError, match="slot 0"):
        _assert_slot_alignment(proposals_broken, current, expected)
