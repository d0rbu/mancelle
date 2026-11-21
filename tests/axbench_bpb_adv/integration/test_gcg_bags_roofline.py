"""Integration tests for the roofline feature-GCG derive (Task 6, ``gcg_bags.py``).

The roofline derive is the in-sample upper bound and the sibling of the Task-5
fair derive: for each held-out prefix (the eval unit) it optimizes a bag on THAT
prefix's OWN rich/avoid members, warm-started from the in-sample Feature-DiffMean
over the SAME prefix's completions (the #575 oracle), and caches it keyed by
``prefix_id``. The scored prefix set is the same held-out split as fair.

These drive the REAL ``optimize_feature_conditioning_universal`` on the tiny
randomized encoder-decoder reused from ``test_gcg_bags_fair`` (CPU/float32, SDPA),
stubbing only ``feature_diffmean_bag`` to a deterministic, sub-optimal warm-start
so the optimizer has room to improve and the test stays CPU-fast.

What is pinned
--------------
(a) one bag artifact per PROCESSED prefix, keyed by ``prefix_id``;
(b) THE LADDER INVARIANT — the roofline bag's in-sample ``g_steer`` on a prefix's
    own members is <= the fair (universal) bag's ``g_steer`` on the same members
    (in-sample dominates the universal bag);
(c) the warm-start is the IN-SAMPLE DiffMean — ``feature_diffmean_bag`` is called
    once per prefix with exactly THAT prefix's own (non-empty) completions;
(d) the SIGN guard — the optimized in-sample bag lowers ``g_steer`` below the
    warm-start (steering strengthens); a flipped derive mode REDs this
    (mutation-proved in ``.superpowers/sdd/task-6-report.md``);
(e) determinism — same ``master_seed`` ⇒ byte-identical bag content + metadata;
(f) the scored-prefix artifact is exactly the fair held-out split;
(g) the method-activation predicates.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

import nanochat.flash_attention as fa_module
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import SteeringMode

# Reuse the proven Task-5 tiny-encdec builder + the deterministic sub-optimal
# warm-start + the trivial tokenizer (its decode is the warm-start text source).
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_bags import (
    active_roofline_gcg_modes,
    build_ours_gcg_bags_fair,
    build_ours_gcg_bags_roofline,
    mean_g_steer,
    roofline_gcg_methods_active,
)
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from tests.axbench_bpb_adv.integration.test_gcg_bags_fair import (
    _MAX_SEQ_LEN,
    _N_SAE,
    _NUM_BUCKETS,
    _WARM_BAG,
    _build_model,
    _context_map,
    _segments_for,
    _StubBaseModel,
    _StubTokenizer,
)

_CPU = t.device("cpu")


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
    """A prefix with one rich + one avoid completion (distinct prefix tokens).

    Shared completions (the defaults) keep every prefix's member near-identical so
    the fair/roofline ladder comparison isolates the optimize-target difference;
    callers pass per-prefix ``rich``/``avoid`` when the test must distinguish each
    prefix's in-sample warm-start by content.  ``prefix_tokens`` round-trips from
    ``prefix_text`` under ``_StubTokenizer``, mirroring real attack rows.
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


def _members(prefix: BpbAttackPrefix) -> list[Any]:
    """The class-margin members of a SINGLE prefix (the roofline eval unit).

    Uses the same single empty-feature context segment the fair test uses, so the
    members match the derive's exactly (single context slot + generation slot).
    """
    tok = _StubTokenizer()
    return list(
        build_class_margin_members(
            prefix,
            prefix_segments=_segments_for(prefix, tok),
            tokenizer=tok,
            max_seq_len=_MAX_SEQ_LEN,
            device=_CPU,
        )
    )


def _make_cfg(
    *,
    reg_lambda_grid: tuple[float, ...] = (0.0,),
    split_ratio: float = 0.8,
    base_methods: list[str] | None = None,
    mean_g_steer_batch: int = 16,
    max_forward_batch: int = 8,
    max_grad_batch: int = 4,
) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="gemma-2-2b",
        base_methods=base_methods,
        feature_dir="",
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=reg_lambda_grid,
            split_ratio=split_ratio,
            split_seed=0,
            max_steps=8,
            minibatch_size=8,
            master_seed=0,
            mean_g_steer_batch=mean_g_steer_batch,
            max_forward_batch=max_forward_batch,
            max_grad_batch=max_grad_batch,
        ),
    )


def _decoded(tokens: tuple[int, ...]) -> str:
    return _StubTokenizer().decode(list(tokens))


_WarmCall = tuple[
    tuple[str, ...],  # rich prefix texts (the RAW field — the pairing teeth)
    tuple[str, ...],  # rich completion texts
    tuple[str, ...],  # avoid prefix texts
    tuple[str, ...],  # avoid completion texts
]


def _recording_warm_stub() -> tuple[Callable[..., FeatureConditioning], list[_WarmCall]]:
    """A deterministic, sub-optimal warm-start that RECORDS its item texts.

    The recording proves the derive feeds each prefix its OWN in-sample
    completions (one call per prefix, that prefix's completions only) AND that
    every item carries ITS OWN prefix's RAW text — a hoisted/decoded/mispaired
    prefix_text diverges from the expected 4-tuples.

    The reused ``_build_model`` (from ``test_gcg_bags_fair``) never constructs
    a compacted model, so the derive must always thread ``allowed_ids=None``
    here (issue #720 PR-2 Task 2's "uncompacted unchanged" contract) -- every
    test in this module that runs a derive to completion exercises this
    assertion for free.
    """
    calls: list[_WarmCall] = []

    def _stub(
        rich_items: Any,
        avoid_items: Any,
        *,
        sae: Any,
        model: Any,
        sae_layer: Any,
        percentile_table: Any,
        mode: Any,
        k_pos: int,
        k_neg: int,
        positive_bucket: int,
        negative_bucket: int,
        device: Any,
        extraction_batch_size: int | None = None,
        allowed_ids: Any = None,
    ) -> FeatureConditioning:
        del extraction_batch_size
        assert allowed_ids is None, (
            f"uncompacted derive must thread allowed_ids=None, got {allowed_ids!r}"
        )
        calls.append(
            (
                tuple(item.prefix_text for item in rich_items),
                tuple(item.completion_text for item in rich_items),
                tuple(item.prefix_text for item in avoid_items),
                tuple(item.completion_text for item in avoid_items),
            )
        )
        assert rich_items and avoid_items, "derive must pass prefixed completions"
        return FeatureConditioning.from_dict(dict(_WARM_BAG))

    return _stub, calls


def _run_roofline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
) -> list[_WarmCall]:
    stub, calls = _recording_warm_stub()
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", stub)
    tok = _StubTokenizer()
    build_ours_gcg_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=_StubBaseModel(),  # cfg.n_ctx read; DiffMean forward stubbed
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=cfg,
        paths=paths,
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )
    return calls


def _run_fair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
) -> None:
    stub, _calls = _recording_warm_stub()
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", stub)
    tok = _StubTokenizer()
    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=_StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=cfg,
        paths=paths,
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (a)/(c) — per-prefix artifacts + in-sample warm-start provenance
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_writes_per_prefix_bags_and_uses_in_sample_warm_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    # DISTINCT completions per prefix so the warm-start provenance is by content.
    prefixes = [
        _make_prefix(
            concept_id, mode, i, rich=(20 + i, 21, 22, 23), avoid=(30 + i, 31, 32, 33)
        )
        for i in range(3)
    ]
    cfg = _make_cfg(reg_lambda_grid=(0.0, 0.5), split_ratio=0.5)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof")
    calls = _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    _optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )
    held_out_ids = {prefix.core_prefix_id for prefix in held_out_prefixes}

    # (a) one bag per (held-out prefix, lambda), keyed by prefix_id.
    for prefix in prefixes:
        for reg_lambda in (0.0, 0.5):
            path = paths.ours_gcg_bag(
                concept_id, mode, reg_lambda, "roofline", prefix_id=prefix.core_prefix_id
            )
            assert path.exists() is (prefix.core_prefix_id in held_out_ids)

    # The fair (universal, no prefix_id) artifact is NOT written by the roofline derive.
    assert not paths.ours_gcg_bag(concept_id, mode, 0.0, "fair").exists()

    # (c) warm-start called ONCE PER HELD-OUT PREFIX with that prefix's OWN completions
    # AND its own RAW prefix text (one rich + one avoid item each, never the
    # group aggregate, never a decoded/hoisted/mispaired prefix). Lambdas reuse
    # the per-prefix warm-start, so #calls == #held_out (not #held_out * #lambdas).
    assert len(calls) == len(held_out_prefixes), (
        f"expected one in-sample warm-start per held-out prefix, got {len(calls)} calls"
    )
    expected = {
        (
            (prefix.prefix_text,),
            (_decoded(tuple(prefix.concept_rich_output_tokens[0])),),
            (prefix.prefix_text,),
            (_decoded(tuple(prefix.concept_avoiding_output_tokens[0])),),
        )
        for prefix in held_out_prefixes
    }
    assert set(calls) == expected, (
        "warm-start must be the per-prefix in-sample DiffMean (one prefix's "
        f"completions, paired with that prefix's RAW text, per call); got {calls}"
    )

    # metadata records the per-prefix identity + the in-sample g_steer.
    first_held_out = held_out_prefixes[0]
    bag_dict, metadata = read_ours_gcg_bag(
        paths.ours_gcg_bag(
            concept_id, mode, 0.0, "roofline", prefix_id=first_held_out.core_prefix_id
        )
    )
    assert bag_dict, "roofline bag is unexpectedly empty"
    assert all(0 <= bucket < _NUM_BUCKETS for bucket in bag_dict.values())
    assert metadata["mode"] == mode.value
    assert metadata["mode_tag"] == "roofline"
    assert metadata["prefix_id"] == first_held_out.core_prefix_id
    assert metadata["scored_split"] == "held_out"

    p0_members = _members(first_held_out)
    refined = FeatureConditioning.from_dict(bag_dict)
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    g_refined = mean_g_steer(model, p0_members, refined, mode=mode, device=_CPU)
    g_warm = mean_g_steer(model, p0_members, warm, mode=mode, device=_CPU)
    # in-sample bag never worsens its OWN g_steer vs the warm-start.
    assert g_refined <= g_warm + 1e-9
    assert float(metadata["in_sample_g_steer"]) == pytest.approx(g_refined, abs=1e-5)
    assert float(metadata["warm_start_g_steer"]) == pytest.approx(g_warm, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# (b) — THE LADDER INVARIANT (roofline dominates the fair universal bag in-sample)
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_ladder_invariant_dominates_fair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """roofline g_steer(prefix) <= fair g_steer(prefix) + tol on the prefix's members.

    Roofline optimizes the prefix ALONE (in-sample); fair optimizes a universal
    bag over its OPTIMIZE subset. The ladder is compared on the exact HELD-OUT
    scored set shared by fair and roofline, so the comparison isolates oracle
    per-prefix conditioning from fair universal conditioning.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(4)]
    reg_lambda = 0.0
    split_ratio, split_seed = 0.8, 0  # int(4 * 0.8) == 3 optimize + 1 held-out

    fair_cfg = _make_cfg(reg_lambda_grid=(reg_lambda,), split_ratio=split_ratio)
    roof_cfg = _make_cfg(reg_lambda_grid=(reg_lambda,), split_ratio=split_ratio)
    fair_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair")
    roof_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof")
    _run_fair(monkeypatch, model=model, prefixes=prefixes, cfg=fair_cfg, paths=fair_paths)
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=roof_cfg, paths=roof_paths)

    fair_dict, _ = read_ours_gcg_bag(
        fair_paths.ours_gcg_bag(concept_id, mode, reg_lambda, "fair")
    )
    fair_bag = FeatureConditioning.from_dict(fair_dict)
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))

    # The SAME seeded split the fair derive used — its held-out subset is the
    # scored fair-vs-roofline comparison set.
    _optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=split_ratio, seed=split_seed
    )
    assert len(held_out_prefixes) == 1  # non-vacuous: exact fair scored set compared

    tol = 1e-3
    for prefix in held_out_prefixes:
        members = _members(prefix)
        roof_dict, _ = read_ours_gcg_bag(
            roof_paths.ours_gcg_bag(
                concept_id, mode, reg_lambda, "roofline", prefix_id=prefix.core_prefix_id
            )
        )
        roof_bag = FeatureConditioning.from_dict(roof_dict)
        g_roof = mean_g_steer(model, members, roof_bag, mode=mode, device=_CPU)
        g_fair = mean_g_steer(model, members, fair_bag, mode=mode, device=_CPU)
        g_warm = mean_g_steer(model, members, warm, mode=mode, device=_CPU)
        assert g_roof <= g_fair + tol, (
            f"ladder violated for {prefix.core_prefix_id}: roofline g_steer "
            f"{g_roof} > fair g_steer {g_fair} + {tol}"
        )
        # roofline also never worsens its own warm-start (the in-sample floor).
        assert g_roof <= g_warm + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# (d) — SIGN guard: the in-sample bag strengthens steering
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_sign_trap_in_sample_margin_strengthens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optimized in-sample bag lowers g_steer below the warm-start.

    ``build_ours_gcg_bags_roofline`` builds the objective via
    ``class_margin_objective(mode=group.mode)`` and MINIMIZES
    ``g_steer = loss_desired - loss_contrast`` (the single §7.2 negation), which
    STRENGTHENS steering. Passing the WRONG mode (or negating g_steer) makes the
    optimizer MAXIMIZE the correctly-signed margin, so the in-sample g_steer rises
    ABOVE the warm-start and this assertion REDs (RED-proved by mutation in the
    task report).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(3)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="sign")
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    _optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )

    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    for prefix in held_out_prefixes:
        members = _members(prefix)
        bag_dict, metadata = read_ours_gcg_bag(
            paths.ours_gcg_bag(
                concept_id, mode, 0.0, "roofline", prefix_id=prefix.core_prefix_id
            )
        )
        refined = FeatureConditioning.from_dict(bag_dict)
        g_warm = mean_g_steer(model, members, warm, mode=mode, device=_CPU)
        g_refined = mean_g_steer(model, members, refined, mode=mode, device=_CPU)
        # CORRECT mode → the in-sample margin moves the RIGHT way (g_steer drops).
        assert g_refined < g_warm, (
            f"in-sample margin moved the wrong way for {prefix.core_prefix_id}: "
            f"refined={g_refined} >= warm={g_warm} (a flipped derive mode would do this)"
        )
        assert float(metadata["in_sample_g_steer"]) == pytest.approx(g_refined, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# (e) — determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_determinism_same_seed_identical_bag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same ``master_seed`` ⇒ identical bag content + metadata across runs.

    Identity is on the DECODED bag (``{fid: bucket}``) and the metadata dict, NOT
    raw bytes (safetensors serializes ``__metadata__`` through a Rust HashMap whose
    key order is unstable — the repo-wide "byte-identity == decompressed tensors"
    convention).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(3)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))

    paths_a = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run_a")
    paths_b = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run_b")
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths_a)
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths_b)
    _optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )

    for prefix in held_out_prefixes:
        bag_a, meta_a = read_ours_gcg_bag(
            paths_a.ours_gcg_bag(
                concept_id, mode, 0.0, "roofline", prefix_id=prefix.core_prefix_id
            )
        )
        bag_b, meta_b = read_ours_gcg_bag(
            paths_b.ours_gcg_bag(
                concept_id, mode, 0.0, "roofline", prefix_id=prefix.core_prefix_id
            )
        )
        assert bag_a == bag_b, f"non-deterministic roofline bag content for {prefix.core_prefix_id}"
        assert meta_a == meta_b, "non-deterministic roofline bag metadata"


# ─────────────────────────────────────────────────────────────────────────────
# (f) — scored prefix set equals the fair held-out split
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_scored_prefixes_match_fair_held_out_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    n_prefixes = 5
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(n_prefixes)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,), split_ratio=0.6)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="cap")

    messages: list[str] = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO", format="{message}")
    try:
        _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    finally:
        logger.remove(handler_id)

    optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )
    held_out_ids = [p.core_prefix_id for p in held_out_prefixes]
    optimize_ids = [p.core_prefix_id for p in optimize_prefixes]
    assert held_out_ids and optimize_ids
    for pid in held_out_ids:
        assert paths.ours_gcg_bag(concept_id, mode, 0.0, "roofline", prefix_id=pid).exists()
    for pid in optimize_ids:
        assert not paths.ours_gcg_bag(concept_id, mode, 0.0, "roofline", prefix_id=pid).exists()

    # No silent population choice: the split counts are logged.
    joined = "\n".join(messages)
    assert "optimize=3" in joined and "held_out=2" in joined, (
        f"roofline held-out split must log optimize/held_out counts; got:\n{joined}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (g) — method-activation predicates (model-free)
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_gcg_methods_active_predicates() -> None:
    # None => all methods active => both roofline modes derive.
    assert active_roofline_gcg_modes(_make_cfg(base_methods=None)) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert roofline_gcg_methods_active(_make_cfg(base_methods=None))
    assert active_roofline_gcg_modes(
        _make_cfg(base_methods=["ours_gcg_roofline_encdec"])
    ) == {SteeringMode.POSITIVE}
    assert active_roofline_gcg_modes(
        _make_cfg(base_methods=["neg_ours_gcg_roofline_encdec"])
    ) == {SteeringMode.NEGATIVE}
    # The FAIR methods do NOT activate the roofline derive.
    assert active_roofline_gcg_modes(_make_cfg(base_methods=["ours_gcg_encdec"])) == set()
    assert not roofline_gcg_methods_active(_make_cfg(base_methods=["diffmean"]))


# ─────────────────────────────────────────────────────────────────────────────
# #14 — a prefix with no class-margin members RAISES (was warn + continue)
# ─────────────────────────────────────────────────────────────────────────────


def test_roofline_empty_members_raises_not_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prefix yielding no class-margin members RAISES (never a silent skip).

    ``build_class_margin_members`` returns >= 1 member or raises, so empty members
    is a real inconsistency, not a routine skip. Forced here by stubbing
    ``_members_for_prefixes -> []``; the roofline derive must raise
    ``AxBenchBpbAdvError``, symmetric with the fair derive's empty-optimize-members
    guard — replacing the old warn + continue that silently dropped the prefix from
    the scored set.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(2)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof")
    monkeypatch.setattr(gcg_bags, "_members_for_prefixes", lambda *a, **k: [])
    with pytest.raises(AxBenchBpbAdvError, match="no class-margin members"):
        _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)


def test_roofline_backstop_raises_on_empty_held_out_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty roofline held-out selection raises instead of persisting an empty scored set.

    OursGcgConfig bounds split_ratio < 1 and split_prefixes_for_concept rejects
    <2-prefix groups, so this is the defense-in-depth backstop against a future
    split change silently zeroing the roofline scored set.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(2)]
    cfg = _make_cfg()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="empty_sel")
    monkeypatch.setattr(
        gcg_bags, "split_prefixes_for_concept", lambda group, *, ratio, seed: (list(group), [])
    )
    with pytest.raises(AxBenchBpbAdvError, match="selected no HELD-OUT prefixes"):
        _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    assert not paths.ours_gcg_scored_prefixes(7, SteeringMode.POSITIVE, "roofline").exists()


def test_roofline_derive_threads_nondefault_batch_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NON-DEFAULT knob values reach EVERY roofline g_steer call + objective build.

    The roofline derive binds the batching knobs in its own per-group
    ``_group_g_steer`` / ``_objective_factory``; dropping either binding silently
    reverts roofline measurements to the module defaults (16 / 8) while fair
    keeps the configured values — a fair-vs-roofline drift with zero test signal
    (max-review finding #4). With defaults distinct from the configured values
    (3 / 5) any such drop diverges here.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(3)]
    cfg = _make_cfg(mean_g_steer_batch=3, max_forward_batch=5, max_grad_batch=7)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof_knobs")

    real_mean_g_steer = gcg_bags.mean_g_steer
    g_steer_knobs: list[tuple[int, int]] = []

    def _spy_mean_g_steer(*args: Any, **kwargs: Any) -> float:
        g_steer_knobs.append((kwargs["batch_size"], kwargs["max_forward_batch"]))
        return real_mean_g_steer(*args, **kwargs)

    real_objective = gcg_bags.class_margin_objective
    objective_mfbs: list[int] = []
    objective_mgbs: list[int | None] = []
    objective_alphas: list[float | None] = []

    def _spy_objective(
        model_arg: Any,
        *,
        mode: Any,
        device: Any,
        max_forward_batch: int,
        max_grad_batch: int | None = None,
        objective_alpha: float | None = None,
    ):
        # None distinguishes "this site did not pass the knob" from "passed the
        # value that happens to equal the default" -- the two a plain int would
        # conflate, and the second is exactly the regression being guarded.
        objective_mfbs.append(max_forward_batch)
        objective_mgbs.append(max_grad_batch)
        objective_alphas.append(objective_alpha)
        extra: dict[str, Any] = (
            {} if max_grad_batch is None else {"max_grad_batch": max_grad_batch}
        )
        if objective_alpha is not None:
            extra["objective_alpha"] = objective_alpha
        return real_objective(
            model_arg,
            mode=mode,
            device=device,
            max_forward_batch=max_forward_batch,
            **extra,
        )

    monkeypatch.setattr(gcg_bags, "mean_g_steer", _spy_mean_g_steer)
    monkeypatch.setattr(gcg_bags, "class_margin_objective", _spy_objective)
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    # Every roofline g_steer measurement (warm + in-sample, per prefix) under (3, 5).
    assert g_steer_knobs, "no g_steer measurement was taken"
    assert set(g_steer_knobs) == {(3, 5)}
    # Every objective construction (measurements + the optimizer factory) got 5.
    assert objective_mfbs, "no objective was ever constructed"
    assert set(objective_mfbs) == {5}
    # #1245: the GRADIENT-path bound must be threaded to the OPTIMIZER's
    # objective. A dropped binding there silently reverts that site to
    # DEFAULT_MAX_GRAD_BATCH (4), which is why the configured value is 7.
    passed = [v for v in objective_mgbs if v is not None]
    assert passed, "no objective was built with the gradient bound at all"
    assert set(passed) == {7}
    # mean_g_steer is forward-only (.losses() only), so it legitimately omits
    # the gradient bound -- but it must still construct SOME objective, or the
    # assertion above could hold vacuously on a single site.
    assert any(v is None for v in objective_mgbs), (
        "expected the forward-only mean_g_steer objective to omit the bound"
    )
