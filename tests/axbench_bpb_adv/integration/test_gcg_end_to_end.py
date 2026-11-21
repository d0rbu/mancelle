"""End-to-end capstone integration test for the Phase-2 feature-GCG BPB-Adv methods (Task 9).

This is the strongest correctness proof of the whole feature-GCG Phase-2 stack: it
drives the REAL pipeline end to end on a tiny encoder-decoder (CPU/float32, SDPA),
tying Tasks 1-8 together::

    BUILD_CANDIDATES  (the real fair + roofline derive -- Tasks 5/6)
        -> writes the per-(concept, mode, lambda[, prefix]) conditioning bags
    APPLY             (the real build_intervention -> _build_ours_intervention -- Task 7)
        -> loads the cached bag from disk and returns
           Intervention.ours(feature_conditions=bag)
    SCORE             (the unchanged encoder-generation-slot scoring -- class_margin_objective)
        -> the encoder conditions on the loaded bag and teacher-forces the
           generation slot; ``g_steer = loss_desired - loss_contrast`` (lower =
           stronger steering) is the signed class margin

REAL vs mocked
--------------
REAL (driven, not stubbed):
  * the derive -- ``optimize_feature_conditioning_universal`` inside
    ``build_ours_gcg_bags_fair`` / ``build_ours_gcg_bags_roofline``;
  * the per-concept optimize/held-out sub-split -- ``split_prefixes_for_concept``;
  * the apply -- ``build_intervention`` -> ``_build_ours_intervention`` ->
    ``read_ours_gcg_bag`` -> ``Intervention.ours`` (the Task-7 loader, keyed by
    ``(concept, mode, reg_lambda, family[, prefix_id])``);
  * the score -- ``class_margin_objective`` (via ``mean_g_steer``): the encoder
    conditions on the loaded bag and teacher-forces the generation slot.
MOCKED (only the model-independent, GPU/SAE-bound seam):
  * ``feature_diffmean_bag`` -> a fixed, deterministic, SUB-OPTIMAL warm-start so
    the GCG optimizer has room to refine and the test is CPU-fast + deterministic.
    The full real-SAE/GPU smoke is deferred to a CPU box; this is the CPU
    end-to-end.

What is pinned (the capstone proof), for ``ours_gcg_encdec`` AND
``ours_gcg_roofline_encdec`` in BOTH positive and negative mode:
  (a) FAIR is scored on the HELD-OUT sub-split -- the optimize-subset prefixes are
      NOT in the fair method's scored (generalization) set; a leak would invalidate
      the generalization measure
      (``test_end_to_end_fair_scored_on_held_out_split``);
  (b) ROOFLINE is scored on the SAME HELD-OUT sub-split, per prefix -- each held-out
      prefix loads its OWN oracle bag while optimize-subset prefixes have no bag
      (``test_end_to_end_roofline_scored_in_sample_per_prefix``);
  (c) the steered GCG margin beats the no-steering (k=0) control on the optimize
      set (``test_end_to_end_steered_gcg_beats_control_on_optimize``);
  (d) the held-out margin is REPORTED and equals an independent re-score of the
      bag loaded through the REAL apply path (folded into (a));
  (e) one scored point per ``reg_lambda`` (folded into (a)).
  THE END-TO-END SIGN TRAP: flipping ONE sign-chain link (the ``mode`` passed to
  the derive's ``class_margin_objective``) makes the held-out margin -- measured
  end to end through the REAL apply path -- move the WRONG direction. Pinned in
  BOTH directions by the ``flip_sign`` parametrization of
  ``test_end_to_end_sign_trap_held_out_margin_direction`` (RED-proof in
  ``.superpowers/sdd/task-9-report.md``).
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as fa_module
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_GRAD_BATCH,
    DEFAULT_OBJECTIVE_ALPHA,
)
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.objective import class_margin_objective
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept, SteeringMode
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb.types import InterventionKind
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.gcg_bags import (
    build_ours_gcg_bags_fair,
    build_ours_gcg_bags_roofline,
    mean_g_steer,
)
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept

# Reuse the proven Task-5 tiny-encdec builder, the deterministic sub-optimal
# warm-start stub, and the trivial nanochat tokenizer (the warm-start stub ignores
# its decoded text).  Prefixes/members are built locally so each test can choose
# SHARED vs DISTINCT completions (the leak teeth need distinct; the sign trap needs
# shared -- see the per-test fixtures below).
from tests.axbench_bpb_adv.integration.test_gcg_bags_fair import (
    _MAX_SEQ_LEN,
    _N_SAE,
    _NUM_BUCKETS,
    _WARM_BAG,
    _build_model,
    _context_map,
    _segments_for,
    _stub_feature_diffmean_bag,
    _StubBaseModel,
    _StubTokenizer,
)

_CPU = t.device("cpu")
_CONCEPT_ID = 7

# Method names by steering mode (positive / negative variants of each family).
_FAIR_METHOD = {
    SteeringMode.POSITIVE: "ours_gcg_encdec",
    SteeringMode.NEGATIVE: "neg_ours_gcg_encdec",
}
_ROOFLINE_METHOD = {
    SteeringMode.POSITIVE: "ours_gcg_roofline_encdec",
    SteeringMode.NEGATIVE: "neg_ours_gcg_roofline_encdec",
}
_CONTROL_METHOD = {
    SteeringMode.POSITIVE: "control_encdec",
    SteeringMode.NEGATIVE: "neg_control_encdec",
}

_MODES = [SteeringMode.POSITIVE, SteeringMode.NEGATIVE]


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Data builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_prefix(
    concept_id: int,
    mode: SteeringMode,
    idx: int,
    *,
    rich: tuple[int, ...] = (20, 21, 22, 23),
    avoid: tuple[int, ...] = (30, 31, 32, 33),
) -> BpbAttackPrefix:
    """A prefix with distinct prefix tokens and the given rich/avoid completions.

    SHARED completions (the defaults) make a bag's effect on ``g_steer`` near
    identical across the optimize/held-out split, so the held-out generalization
    DIRECTION (the sign-trap teeth) is reliable rather than statistical.  DISTINCT
    completions (``rich=(20+i, ...)``) make each prefix's member ``g_steer`` differ
    so the held-out re-score differs from an optimize/all-members one -- the
    held-out NON-LEAK teeth.
    """
    prefix_text = f"10 11 {12 + idx} 13"
    return BpbAttackPrefix(
        core_prefix_id=f"c{concept_id}_{mode.value}_p{idx}",
        concept_id=concept_id,
        mode=mode,
        prefix_text=prefix_text,
        prefix_hash=f"h{concept_id}{mode.value}{idx}",
        prefix_tokens=encode_prefix(prefix_text, _StubTokenizer()),
        concept_rich_sample_ids=["r0"],
        concept_avoiding_sample_ids=["a0"],
        concept_rich_output_tokens=[list(rich)],
        concept_avoiding_output_tokens=[list(avoid)],
    )


def _shared_prefixes(mode: SteeringMode, n: int = 5) -> list[BpbAttackPrefix]:
    """``n`` prefixes sharing rich/avoid completions (reliable generalization)."""
    return [_make_prefix(_CONCEPT_ID, mode, i) for i in range(n)]


def _distinct_prefixes(mode: SteeringMode, n: int = 5) -> list[BpbAttackPrefix]:
    """``n`` prefixes with per-prefix-distinct completions (leak / per-prefix teeth)."""
    return [
        _make_prefix(
            _CONCEPT_ID, mode, i, rich=(20 + i, 21, 22, 23), avoid=(30 + i, 31, 32, 33)
        )
        for i in range(n)
    ]


def _members(prefixes: list[BpbAttackPrefix]) -> list[Any]:
    """Class-margin members for *prefixes* under the derive's exact layout.

    Uses the SAME single empty-feature context segment (per-prefix, spanning the
    decoded prefix) that ``_context_map`` feeds the derive, so the members scored
    here ARE the members the derive optimized / measured against (single context
    slot + generation slot).
    """
    tok = _StubTokenizer()
    out: list[Any] = []
    for prefix in prefixes:
        out.extend(
            build_class_margin_members(
                prefix,
                prefix_segments=_segments_for(prefix, tok),
                tokenizer=tok,
                max_seq_len=_MAX_SEQ_LEN,
                device=_CPU,
            )
        )
    return out


def _make_cfg(
    *,
    reg_lambda_grid: tuple[float, ...] = (0.0,),
    objective_alpha_grid: tuple[float, ...] = (DEFAULT_OBJECTIVE_ALPHA,),
    split_ratio: float = 0.6,
) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="gemma-2-2b",
        base_methods=None,
        feature_dir="",
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=reg_lambda_grid,
            objective_alpha_grid=objective_alpha_grid,
            split_ratio=split_ratio,
            split_seed=0,
            max_steps=8,
            minibatch_size=8,
            master_seed=0,
        ),
    )


def _make_concept(concept_id: int = _CONCEPT_ID) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=0,
        layer=8,
        sae_source="dummy",
        description="capstone concept",
        corpus_count=10,
        description_source="manual",
    )


# ─────────────────────────────────────────────────────────────────────────────
# The real derive (Tasks 5/6), with an optional flipped sign-chain link
# ─────────────────────────────────────────────────────────────────────────────


def _mode_flipped_objective(
    model: Any,
    *,
    mode: Any,
    device: Any,
    max_forward_batch: int = 8,
    max_grad_batch: int = DEFAULT_MAX_GRAD_BATCH,
    objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
) -> Any:
    """``class_margin_objective`` with the steering mode NEGATED (a flipped link).

    Built from the directly-imported real ``class_margin_objective`` (never the
    module-global the derive references), so installing this as
    ``gcg_bags.class_margin_objective`` flips ONLY the derive's optimizer objective
    while this wrapper itself still computes the correctly-signed margin under the
    opposite mode.  Mirrors the real factory's signature (incl. the threaded
    ``max_forward_batch``, since #1245 ``max_grad_batch``, and the class-margin
    ``objective_alpha`` -- all of which the derive's objective_factory forwards).
    """
    flipped = (
        SteeringMode.NEGATIVE
        if SteeringMode(mode) == SteeringMode.POSITIVE
        else SteeringMode.POSITIVE
    )
    return class_margin_objective(
        model,
        mode=flipped,
        device=device,
        max_forward_batch=max_forward_batch,
        max_grad_batch=max_grad_batch,
        objective_alpha=objective_alpha,
    )


def _drive(
    derive: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    flip_sign: bool,
) -> None:
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", _stub_feature_diffmean_bag)
    tok = _StubTokenizer()
    kwargs = {
        "prefixes": prefixes,
        "model": model,
        "base_model": _StubBaseModel(),  # cfg.n_ctx read; DiffMean forward stubbed
        "sae": object(),
        "percentile_table": t.zeros((_N_SAE, 99), dtype=t.float64),
        "cfg": cfg,
        "paths": paths,
        "device": _CPU,
        "tokenizer": tok,
        "context_segments_by_prefix_hash": _context_map(prefixes, tok),
    }
    if flip_sign:
        # Scope the flipped objective to the derive ONLY: the re-scoring afterwards
        # uses the restored (correct) gcg_bags.class_margin_objective.
        with monkeypatch.context() as m:
            m.setattr(gcg_bags, "class_margin_objective", _mode_flipped_objective)
            derive(**kwargs)
    else:
        derive(**kwargs)


def _run_fair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    flip_sign: bool = False,
) -> None:
    _drive(
        build_ours_gcg_bags_fair,
        monkeypatch,
        model=model,
        prefixes=prefixes,
        cfg=cfg,
        paths=paths,
        flip_sign=flip_sign,
    )


def _run_roofline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
) -> None:
    _drive(
        build_ours_gcg_bags_roofline,
        monkeypatch,
        model=model,
        prefixes=prefixes,
        cfg=cfg,
        paths=paths,
        flip_sign=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The real apply path (Task 7) -- build_intervention -> Intervention.ours(bag)
# ─────────────────────────────────────────────────────────────────────────────


def _apply_bag(
    *,
    method_name: str,
    reg_lambda: float | None,
    paths: AxBenchBpbAdvPaths,
    model: Any,
    prefix_id: str | None = None,
) -> FeatureConditioning:
    """Load the cached bag through the REAL Task-7 apply path.

    Drives ``build_intervention`` -> ``_build_ours_intervention`` ->
    ``read_ours_gcg_bag`` -> ``Intervention.ours`` exactly as the BPB-Adv scorer
    does at score time (``coefficient`` IS the ``reg_lambda`` slug; ``core_prefix_id``
    threads the per-prefix roofline bag and is ignored for the fair universal bag),
    and returns the loaded conditioning as a :class:`FeatureConditioning`.

    ``model`` is the SAME model the derive step wrote the bag against (checkpoint-V
    stamping/verify -- issue #695 PR-1 follow-up): passing a DIFFERENT model here
    would raise, since this apply path now verifies the loaded bag's derive-time
    ``V``-fingerprint against the model applying it.
    """
    intervention = build_intervention(
        method=get_method_by_name(method_name),
        concept=_make_concept(),
        coefficient=reg_lambda,
        source_paths=SourcePaths(source_run_dir=paths.root),  # unused on the gcg path
        diffmean_layer=8,
        target_model_meta=SimpleNamespace(
            negative_selection=None, num_buckets=_NUM_BUCKETS
        ),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        adv_paths=paths,
        core_prefix_id=prefix_id,
        model=model,
        apply_max_act_scale=False,
    )
    assert intervention.kind == InterventionKind.OURS, intervention.kind
    return FeatureConditioning.from_dict(dict(intervention.feature_conditions or {}))


# ─────────────────────────────────────────────────────────────────────────────
# (a)/(d)/(e) — FAIR is scored on the HELD-OUT sub-split (optimize excluded)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
def test_end_to_end_fair_scored_on_held_out_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: SteeringMode
) -> None:
    """The fair method's reported margin is the HELD-OUT generalization measure.

    Drives the real fair derive, then for each ``reg_lambda`` loads the universal
    bag through the REAL apply path and re-scores it via the unchanged
    encoder-generation-slot scoring. Pins:

    (a) NON-LEAK -- the optimize prefixes are disjoint from the scored (held-out)
        set, and the recorded ``held_out_g_steer`` equals the held-out re-score of
        the loaded bag over EXACTLY the held-out members (distinct completions make
        this differ from an optimize/all-members computation, so a leak REDs);
    (d) the held-out margin is REPORTED (finite) for every lambda;
    (e) exactly one bag + one scored margin per ``reg_lambda``.
    """
    model = _build_model(monkeypatch)
    # Distinct completions => held-out members' g_steer differs from the optimize
    # members', so matching the recorded held_out_g_steer to the held-out-only
    # re-score genuinely pins the SCORED SET to held-out (the leak teeth).
    prefixes = _distinct_prefixes(mode)
    reg_lambda_grid = (0.0, 0.5)
    cfg = _make_cfg(reg_lambda_grid=reg_lambda_grid)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair")
    _run_fair(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=0.6, seed=0
    )
    optimize_ids = {p.core_prefix_id for p in optimize_prefixes}
    held_out_ids = {p.core_prefix_id for p in held_out_prefixes}
    # The Task-3 split is disjoint and the held-out set is non-empty: no optimize
    # prefix can leak into the fair method's scored (generalization) set.
    assert held_out_ids, "fixture must produce a non-empty held-out split"
    assert optimize_ids.isdisjoint(held_out_ids)

    optimize_members = _members(optimize_prefixes)
    held_out_members = _members(held_out_prefixes)
    method_name = _FAIR_METHOD[mode]

    scored: dict[float, float] = {}
    for reg_lambda in reg_lambda_grid:
        bag_path = paths.ours_gcg_bag(_CONCEPT_ID, mode, reg_lambda, "fair")
        assert bag_path.exists(), f"missing fair bag for lambda={reg_lambda}: {bag_path}"
        _bag_dict, meta = read_ours_gcg_bag(bag_path)

        # APPLY through the REAL Task-7 path (fair => prefix_id is ignored: ONE
        # universal bag shared across prefixes).
        loaded = _apply_bag(
            method_name=method_name, reg_lambda=reg_lambda, paths=paths, model=model
        )

        # The scored set is EXACTLY the held-out members (disjoint from optimize).
        assert int(meta["n_held_out_members"]) == len(held_out_members) >= 1
        assert int(meta["n_optimize_members"]) == len(optimize_members)

        # (d) the held-out margin is reported AND equals an independent held-out
        # re-score of the bag the REAL apply path returned. Distinct completions
        # make this differ from the optimize re-score, so a derive that secretly
        # scored optimize (or all) members would mismatch here.
        g_held = mean_g_steer(model, held_out_members, loaded, mode=mode, device=_CPU)
        g_opt = mean_g_steer(model, optimize_members, loaded, mode=mode, device=_CPU)
        recorded = float(meta["held_out_g_steer"])
        assert recorded == recorded, "held_out_g_steer must be finite"
        assert recorded == pytest.approx(g_held, abs=1e-5)
        assert g_held != pytest.approx(g_opt, abs=1e-5), (
            "fixture too weak: held-out and optimize margins coincide, so the "
            "non-leak match cannot distinguish the scored set"
        )
        scored[reg_lambda] = g_held

    # (e) exactly one scored point per reg_lambda.
    assert set(scored) == set(reg_lambda_grid)


# ─────────────────────────────────────────────────────────────────────────────
# (b) — ROOFLINE scores held-out prefixes, per prefix (each with its OWN bag)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
def test_end_to_end_roofline_scored_in_sample_per_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: SteeringMode
) -> None:
    """Each held-out roofline prefix loads its OWN bag and is scored in-sample.

    Drives the real roofline derive, then for each held-out prefix loads THAT
    prefix's bag through the REAL apply path (``prefix_id=core_prefix_id``) and
    scores it on the prefix's own members. Pins: per-prefix in-sample scoring (the
    bag never worsens its own ``g_steer`` vs the warm-start), the recorded
    ``in_sample_g_steer`` matches the re-score, distinct prefixes load DISTINCT bags
    (the per-prefix optimization is real and correctly prefix-keyed), optimize
    prefixes have NO roofline bag, and the fair universal bag is NEVER loaded by a
    roofline method.
    """
    model = _build_model(monkeypatch)
    # Distinct completions => per-prefix warm-starts/optimizations differ, so the
    # prefix-keyed apply genuinely returns a DIFFERENT bag per prefix_id.
    prefixes = _distinct_prefixes(mode)
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof")
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )
    assert optimize_prefixes and held_out_prefixes

    method_name = _ROOFLINE_METHOD[mode]
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    loaded_by_prefix: dict[str, FeatureConditioning] = {}
    for prefix in optimize_prefixes:
        bag_path = paths.ours_gcg_bag(
            _CONCEPT_ID, mode, 0.0, "roofline", prefix_id=prefix.core_prefix_id
        )
        assert not bag_path.exists(), (
            "roofline must score the same held-out population as fair; optimize "
            f"prefix {prefix.core_prefix_id} should not have a per-prefix bag"
        )

    for prefix in held_out_prefixes:
        bag_path = paths.ours_gcg_bag(
            _CONCEPT_ID, mode, 0.0, "roofline", prefix_id=prefix.core_prefix_id
        )
        assert bag_path.exists(), f"missing roofline bag for {prefix.core_prefix_id}"
        _bag_dict, meta = read_ours_gcg_bag(bag_path)
        assert meta["prefix_id"] == prefix.core_prefix_id
        assert meta["scored_split"] == "held_out"

        # APPLY through the REAL Task-7 path: prefix_id threads the per-prefix bag.
        loaded = _apply_bag(
            method_name=method_name,
            reg_lambda=0.0,
            paths=paths,
            model=model,
            prefix_id=prefix.core_prefix_id,
        )
        loaded_by_prefix[prefix.core_prefix_id] = loaded

        # In-sample scoring: the prefix's OWN bag on the prefix's OWN members.
        members = _members([prefix])
        g_refined = mean_g_steer(model, members, loaded, mode=mode, device=_CPU)
        g_warm = mean_g_steer(model, members, warm, mode=mode, device=_CPU)
        assert g_refined <= g_warm + 1e-9, (
            f"roofline must not worsen its OWN in-sample g_steer for "
            f"{prefix.core_prefix_id}: refined={g_refined} > warm={g_warm}"
        )
        assert float(meta["in_sample_g_steer"]) == pytest.approx(g_refined, abs=1e-5)

    # A roofline method NEVER loads the fair universal bag (only the roofline derive
    # ran, so the fair artifact does not even exist).
    assert not paths.ours_gcg_bag(_CONCEPT_ID, mode, 0.0, "fair").exists()

    # The apply path is genuinely prefix-keyed: distinct prefixes -> distinct bags.
    distinct_bags = {tuple(b.entries) for b in loaded_by_prefix.values()}
    assert len(distinct_bags) >= 2, (
        "per-prefix roofline bags collapsed to one; the prefix-keyed apply is not "
        "distinguishing prefixes (or the fixture's completions are not distinct)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) — the steered GCG margin beats the no-steering (k=0) control on the optimize set
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
def test_end_to_end_steered_gcg_beats_control_on_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: SteeringMode
) -> None:
    """The steered GCG margin beats the no-steering (k=0) control on the optimize set.

    Both interventions are built through the REAL apply path: the steered fair bag
    via ``build_intervention(ours_gcg_*)``, and the empty no-steering control via
    ``build_intervention(control_encdec)`` (which yields ``Intervention.ours({})``).
    The steered ``g_steer`` (lower = stronger steering) must be strictly below the
    control's on the optimize members -- the GCG conditioning genuinely steers the
    tiny encdec.
    """
    model = _build_model(monkeypatch)
    prefixes = _shared_prefixes(mode)
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="ctrl")
    _run_fair(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    optimize_prefixes, _held = split_prefixes_for_concept(prefixes, ratio=0.6, seed=0)
    optimize_members = _members(optimize_prefixes)

    steered = _apply_bag(
        method_name=_FAIR_METHOD[mode], reg_lambda=0.0, paths=paths, model=model
    )
    control = _apply_bag(
        method_name=_CONTROL_METHOD[mode], reg_lambda=None, paths=paths, model=model
    )
    assert len(control) == 0, "the k=0 control must be empty (no-steering) conditioning"
    assert len(steered) > 0, "the steered GCG bag must be non-empty"

    g_steered = mean_g_steer(model, optimize_members, steered, mode=mode, device=_CPU)
    g_control = mean_g_steer(model, optimize_members, control, mode=mode, device=_CPU)
    assert g_steered < g_control, (
        "steered GCG margin must beat the no-steering control on the optimize set: "
        f"g_steered={g_steered} >= g_control={g_control}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# THE END-TO-END SIGN TRAP — a flipped derive link moves the held-out margin WRONG
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("flip_sign", [False, True])
def test_end_to_end_sign_trap_held_out_margin_direction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: SteeringMode,
    flip_sign: bool,
) -> None:
    """A flipped derive sign moves the held-out margin the WRONG way, end to end.

    Drives the real fair derive (optionally with the ``mode`` passed to the derive's
    ``class_margin_objective`` NEGATED -- a single flipped sign-chain link), loads
    the resulting bag through the REAL apply path, and re-scores it on the held-out
    members with the CORRECT mode:

      flip_sign=False (correct chain): the optimized bag STRENGTHENS steering, so
        the held-out ``g_steer`` drops below the warm-start's (margin moves RIGHT);
      flip_sign=True  (one link flipped): the optimizer MAXIMIZES the
        correctly-signed margin, so the held-out ``g_steer`` RISES above the
        warm-start's (margin moves WRONG) -- the flip=False assertion would RED here.

    Pinning BOTH directions (and both modes) proves the whole positive/negative +
    DiffMean + GCG + apply chain is sign-consistent end to end.
    """
    model = _build_model(monkeypatch)
    # Shared completions => held-out generalization tracks the optimize direction,
    # so the held-out margin direction is a reliable (not statistical) trap.
    prefixes = _shared_prefixes(mode)
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag=f"sign_{int(flip_sign)}")
    _run_fair(
        monkeypatch,
        model=model,
        prefixes=prefixes,
        cfg=cfg,
        paths=paths,
        flip_sign=flip_sign,
    )

    _opt, held_out_prefixes = split_prefixes_for_concept(prefixes, ratio=0.6, seed=0)
    held_out_members = _members(held_out_prefixes)
    assert held_out_members, "fixture must produce a non-empty held-out split"

    # APPLY through the REAL Task-7 path, then score on held-out with the CORRECT mode.
    loaded = _apply_bag(
        method_name=_FAIR_METHOD[mode], reg_lambda=0.0, paths=paths, model=model
    )
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    g_warm = mean_g_steer(model, held_out_members, warm, mode=mode, device=_CPU)
    g_refined = mean_g_steer(model, held_out_members, loaded, mode=mode, device=_CPU)

    if flip_sign:
        # One flipped link => the end-to-end held-out margin moves the WRONG way.
        assert g_refined > g_warm, (
            "flipped derive sign should DEGRADE held-out steering, but "
            f"g_refined={g_refined} <= g_warm={g_warm} (the trap failed to bite)"
        )
    else:
        # Correct chain => the end-to-end held-out margin moves the RIGHT way.
        assert g_refined < g_warm, (
            "correct derive sign should STRENGTHEN held-out steering, but "
            f"g_refined={g_refined} >= g_warm={g_warm}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class-margin alpha through the REAL apply path
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_alpha_method_coefficient_loads_the_alpha_bag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``ours_gcg_alpha_encdec``'s coefficient resolves on the ALPHA axis, live.

    The score-side counterpart to the derive-side sweep test. It drives the real
    ``build_intervention`` -> ``_build_ours_intervention`` -> ``read_ours_gcg_bag``
    chain, so it pins the axis routing as BEHAVIOUR rather than as a source-text
    assertion.

    The trap this closes: ``ours_gcg_alpha_encdec`` and ``ours_gcg_encdec`` share
    the FAIR bag family, so the ONLY thing distinguishing their coefficient is the
    method name. Read on the wrong axis, coefficient 0.5 would resolve to the
    reg_lambda=0.5 bag and be scored + plotted as if it were alpha=0.5.

    Teeth: the two methods are asked for the SAME scalar 0.5 and must return
    DIFFERENT bags -- and the alpha method's bag must equal the one the derive
    wrote at alpha=0.5.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = _CONCEPT_ID, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(4)]
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="alpha-apply")

    # One derive covering BOTH axes: reg_lambda in {0.0, 0.5} x alpha in {1.0, 0.5}.
    cfg = _make_cfg(reg_lambda_grid=(0.0, 0.5), objective_alpha_grid=(1.0, 0.5))
    _drive(
        build_ours_gcg_bags_fair,
        monkeypatch,
        model=model,
        prefixes=prefixes,
        cfg=cfg,
        paths=paths,
        flip_sign=False,
    )

    # The alpha method reads coefficient 0.5 as an ALPHA (reg_lambda pinned to 0).
    via_alpha_method = _apply_bag(
        method_name="ours_gcg_alpha_encdec", reg_lambda=0.5, paths=paths, model=model
    )
    on_disk_alpha_bag, _meta = read_ours_gcg_bag(
        paths.ours_gcg_bag(concept_id, mode, 0.0, "fair", objective_alpha=0.5)
    )
    assert dict(via_alpha_method.entries) == dict(on_disk_alpha_bag), (
        "alpha method did not load the bag the derive wrote at alpha=0.5"
    )

    # The lambda method reads the SAME scalar as a REG_LAMBDA -> a different bag.
    via_lambda_method = _apply_bag(
        method_name="ours_gcg_encdec", reg_lambda=0.5, paths=paths, model=model
    )
    on_disk_lambda_bag, _meta2 = read_ours_gcg_bag(
        paths.ours_gcg_bag(concept_id, mode, 0.5, "fair")
    )
    assert dict(via_lambda_method.entries) == dict(on_disk_lambda_bag)

    # NON-VACUITY: if the two axes had collapsed, both loads would be the same
    # file and this test would pass for the wrong reason.
    assert (
        paths.ours_gcg_bag(concept_id, mode, 0.0, "fair", objective_alpha=0.5)
        != paths.ours_gcg_bag(concept_id, mode, 0.5, "fair")
    )


def test_end_to_end_alpha_one_reuses_the_plain_fair_bag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """At alpha=1 the alpha method loads the SAME artifact as the plain method.

    This is the designed free-reuse property: a sweep that includes alpha=1 must
    not re-derive the plain-margin bag, and the two methods must agree bit for
    bit at that point (their curves have to meet there to be comparable).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = _CONCEPT_ID, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(4)]
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="alpha-reuse")
    cfg = _make_cfg(reg_lambda_grid=(0.0,), objective_alpha_grid=(1.0,))
    _drive(
        build_ours_gcg_bags_fair,
        monkeypatch,
        model=model,
        prefixes=prefixes,
        cfg=cfg,
        paths=paths,
        flip_sign=False,
    )

    via_alpha = _apply_bag(
        method_name="ours_gcg_alpha_encdec", reg_lambda=1.0, paths=paths, model=model
    )
    via_lambda = _apply_bag(
        method_name="ours_gcg_encdec", reg_lambda=0.0, paths=paths, model=model
    )
    assert dict(via_alpha.entries) == dict(via_lambda.entries)
    # ... and it really is ONE file on disk, not two identical ones.
    assert (
        paths.ours_gcg_bag(concept_id, mode, 0.0, "fair", objective_alpha=1.0)
        == paths.ours_gcg_bag(concept_id, mode, 0.0, "fair")
    )
