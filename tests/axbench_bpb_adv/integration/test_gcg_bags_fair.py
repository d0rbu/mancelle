"""Integration tests for the fair feature-GCG derive (Task 5, ``gcg_bags.py``).

These drive the REAL ``optimize_feature_conditioning_universal`` on a tiny
randomized encoder-decoder (CPU/float32, SDPA) so the GCG bag genuinely refines
the warm-start.  Only the model-dependent seams are stubbed:

* ``feature_diffmean_bag`` (Task 4) → a fixed, deterministic, *sub-optimal*
  warm-start so the optimizer has room to improve and the test stays CPU-fast.
* the tokenizer → a trivial ``decode`` (the warm-start stub ignores the text).

What is pinned
--------------
(a) one bag artifact per ``(concept, mode, reg_lambda)``;
(b) the GCG-refined ``g_steer`` on the optimize members ≤ (and strictly below)
    the warm-start's — GCG improves on DiffMean;
(c) determinism — same ``master_seed`` ⇒ byte-identical bag file;
(d) the held-out ``g_steer`` is computed + recorded in the bag metadata;
(e) the SIGN TRAP — the held-out margin moves the RIGHT way under the correct
    mode (steering strengthens).  Passing the wrong ``mode`` to
    ``class_margin_objective`` inside the derive flips this (RED-proved by
    mutation, see ``.superpowers/sdd/task-5-report.md``);
(f) the SAE-identity guard raises on a mismatched / missing SAE block.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import nanochat.common
import nanochat.flash_attention as fa_module
import nanochat.gpt
import steering.conditional.encoder_decoder.encoder as enc_module
import steering.conditional.encoder_decoder.model as model_module
import steering.eval.axbench_bpb_adv.feature_diffmean as feature_diffmean
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.percentile_table import compute_percentile_checksum
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_bags import (
    active_fair_gcg_modes,
    assert_sae_matches_encdec,
    build_clustered_warm_start_cache,
    build_ours_gcg_bags_fair,
    build_ours_gcg_bags_roofline,
    fair_gcg_methods_active,
    mean_g_steer,
)
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    read_ours_gcg_bag,
    write_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from tests.axbench_bpb_adv.tokenizer_stubs import (
    StubBaseModel,
    StubNanochatTokenizer,
)
from tests.encdec_test_helpers import build_tiny_model

_CPU = t.device("cpu")
_N_SAE = 40
_NUM_BUCKETS = 2
# Any value >= every fixture's prefix+completion length disables the score-time
# truncation guard; members are then byte-identical regardless of the exact
# value (the derive itself uses build_tiny_model's dec_config.sequence_len,
# which is also comfortably above the ~9-token fixtures — so _members() here and
# the derive build the SAME untruncated members).
_MAX_SEQ_LEN = 1024
# Canonical training-SAE block matching the default ``neuronpedia_source``
# "8-gemmascope-res-16k"; layer 8, gemmascope, residual, 16k, gemma-2-2b.
_SAE_META = {
    "release": "gemmascope",
    "model": "gemma-2-2b",
    "activation_type": "residual",
    "layer": 8,
    "width": "16k",
    "l0": None,
}
_USER_CONFIG = {
    "feature_selection": {"distribution": "geometric", "mean": 5.0, "min_k": 1, "max_k": 25},
    "negative_selection": {"distribution": "geometric", "mean": 3.0, "min_k": 0, "max_k": 15},
    "sae": _SAE_META,
}
# A deterministic, sub-optimal warm-start the stubbed DiffMean returns.
_WARM_BAG = {5: 0, 12: 0, 23: 1, 31: 1}


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny-model + data builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_model(monkeypatch: pytest.MonkeyPatch, seed: int = 1234) -> Any:
    """A tiny num_buckets=2 encdec with a non-zero-init encoder (healthy gradient)."""
    for mod in (nanochat.common, nanochat.gpt, enc_module, model_module):
        monkeypatch.setattr(mod, "COMPUTE_DTYPE", t.float32)
    t.manual_seed(seed)
    model = build_tiny_model(n_sae_features=_N_SAE, num_buckets=_NUM_BUCKETS, n_layer=4)
    g = t.Generator().manual_seed(seed)
    enc = model.encoder
    with t.no_grad():
        enc.feature_embed.weight.normal_(0.0, 1.0, generator=g)
        enc.magnitude_embed.weight.normal_(0.0, 1.0, generator=g)
        for ve in enc.value_embeds.values():
            ve.weight.normal_(0.0, 3.0, generator=g)
        for block in enc.blocks:
            block.attn.c_proj.weight.normal_(0.0, 0.5, generator=g)
            block.mlp.c_proj.weight.normal_(0.0, 0.5, generator=g)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.normal_(0.0, 1.0, generator=g)
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0.0, 0.3, generator=g)
            block.mlp.c_proj.weight.normal_(0.0, 0.3, generator=g)
        for xattn in model.cross_attn_layers.values():
            xattn.c_proj.weight.normal_(0.0, 2.0, generator=g)
    model.eval()
    model._loaded_meta_data = {"user_config": _USER_CONFIG}
    return model


# Shared tokenizer/model stubs (ONE home — tests/axbench_bpb_adv/tokenizer_stubs).
# Aliased to the historical private names the sibling test modules import.
_StubTokenizer = StubNanochatTokenizer
_StubBaseModel = StubBaseModel


def _make_prefix(concept_id: int, mode: SteeringMode, idx: int) -> BpbAttackPrefix:
    """A prefix with SHARED rich/avoid completions (distinct prefix tokens).

    Shared completions make the bag's effect on ``g_steer`` near-identical across
    the optimize / held-out split, so held-out generalization (the sign-trap
    teeth) is reliable rather than statistical.  ``prefix_tokens`` round-trips
    from ``prefix_text`` under ``_StubTokenizer`` (BOS 1 + one id per digit
    word), mirroring how attack_rows builds real prefixes.
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
        concept_rich_output_tokens=[[20, 21, 22, 23]],
        concept_avoiding_output_tokens=[[30, 31, 32, 33]],
    )


def _stub_feature_diffmean_bag(
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
    """Deterministic, sub-optimal warm-start (the seam the real DiffMean fills).

    ``_build_model`` (below) never constructs a compacted model, so the derive
    must always thread ``allowed_ids=None`` here (issue #720 PR-2 Task 2's
    "uncompacted unchanged" contract) -- every test in this module that runs
    the derive to completion exercises this assertion for free.
    """
    del extraction_batch_size
    assert allowed_ids is None, (
        f"uncompacted derive must thread allowed_ids=None, got {allowed_ids!r}"
    )
    assert rich_items and avoid_items, "derive must pass prefixed completions"
    # The derive must thread the RAW prefix text (never a BOS-carrying decode:
    # the digit scheme would render the nanochat BOS as a leading "1 " word).
    for item in (*rich_items, *avoid_items):
        assert not item.prefix_text.startswith("1 10 "), (
            f"prefix_text carries the decoded nanochat BOS: {item.prefix_text!r}"
        )
    return FeatureConditioning.from_dict(dict(_WARM_BAG))


def _patch_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", _stub_feature_diffmean_bag)


def _feature_dir(layer: int, width: SAEWidth) -> str:
    """A canonical feature_dir string for the SAE-identity guard's feature_dir leg."""
    return str(
        build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=layer,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=width,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
    )


def _make_cfg(
    *,
    reg_lambda_grid: tuple[float, ...] = (0.0,),
    objective_alpha_grid: tuple[float, ...] = (DEFAULT_OBJECTIVE_ALPHA,),
    base_methods: list[str] | None = None,
    neuronpedia_source: str = "8-gemmascope-res-16k",
    neuronpedia_model_id: str = "gemma-2-2b",
    feature_extraction_model_id: str | None = None,
    feature_dir: str = "",
    split_ratio: float = 0.6,
    mean_g_steer_batch: int = 16,
    max_forward_batch: int = 8,
    max_grad_batch: int = 4,
    extraction_tokenizer_id: str = "google/gemma-2-2b",
) -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        neuronpedia_source=neuronpedia_source,
        neuronpedia_model_id=neuronpedia_model_id,
        base_methods=base_methods,
        feature_dir=feature_dir,
        extraction_tokenizer_id=extraction_tokenizer_id,
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=reg_lambda_grid,
            objective_alpha_grid=objective_alpha_grid,
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
    cfg_with_local_model: Any = cfg
    cfg_with_local_model.feature_extraction_model_id = feature_extraction_model_id
    return cfg


def _segments_for(
    prefix: BpbAttackPrefix, tokenizer: Any
) -> tuple[ContextFeatureSegment, ...]:
    """One EMPTY-feature context segment spanning the whole decoded prefix.

    These derive-orchestration tests exercise the split / warm-start / optimize /
    held-out machinery, NOT per-segment context conditioning (unit-tested in
    ``test_gcg_members.py``).  An empty-feature single segment reproduces the
    prior single-context + generation 2-slot layout EXACTLY — two empty
    conditioning bags, every prefix token → slot 0, output → slot 1 — so the
    sign-trap and g_steer teeth stay calibrated against the same layout.  The
    char span covers the tokenizer's decode of ``prefix_tokens`` so
    ``build_multi_segment_seg_map`` assigns every prefix token to slot 0.
    """
    char_end = len(tokenizer.decode(list(prefix.prefix_tokens)))
    return (
        ContextFeatureSegment(
            feature_ids=(), buckets=(), char_start=0, char_end=char_end
        ),
    )


def _context_map(
    prefixes: list[BpbAttackPrefix], tokenizer: Any
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    """The prefix-hash → segments map the derive looks each prefix up in."""
    return {prefix.prefix_hash: _segments_for(prefix, tokenizer) for prefix in prefixes}


def _two_real_feature_segments(
    prefix: BpbAttackPrefix, tokenizer: Any
) -> tuple[ContextFeatureSegment, ...]:
    """Split the decoded prefix into TWO segments (real features), at a token boundary.

    Exercises the multi-slot encoder-forward + non-empty fixed context conditioning
    (feature ids < n_sae_features == 40, buckets < num_buckets == 2). The char
    boundary is the decoded length of the first half of the tokens, so
    build_multi_segment_seg_map routes the first tokens to slot 0 and the rest to
    slot 1 (generation slot becomes 2).
    """
    toks = list(prefix.prefix_tokens)
    k = max(1, len(toks) // 2)
    boundary = len(tokenizer.decode(toks[:k]))
    full = len(tokenizer.decode(toks))
    return (
        ContextFeatureSegment(
            feature_ids=(3,), buckets=(1,), char_start=0, char_end=boundary
        ),
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(1,), char_start=boundary, char_end=full
        ),
    )


def _members(prefixes: list[BpbAttackPrefix]) -> list[Any]:
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


def _run_derive(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    prefixes: list[BpbAttackPrefix],
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
) -> None:
    _patch_warm(monkeypatch)
    # The derive looks each prefix's context segments up by prefix_hash in the map
    # the score stage also loads (load_context_segments_by_prefix_hash), and aligns
    # them to prefix_tokens with the nanochat tokenizer for the per-member seg_map.
    tok = _StubTokenizer()
    build_ours_gcg_bags_fair(
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


def test_clustered_phase_reorder_matches_direct_real_diffmean_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """D1: one cache phase preserves direct clustered fair/roofline warm bags.

    Both sides use one shared fixed-seed prefix population and the production
    ``feature_diffmean_bag`` percentile/ranking implementation. Only the costly
    base-model/SAE activation extraction seam is replaced with deterministic
    per-text activations. The direct side models the pre-reorder clustered
    sequence; the reordered side must build one cache and pass that same cache
    to both consumers.
    """
    assert gcg_bags.feature_diffmean_bag is feature_diffmean.feature_diffmean_bag
    n_features = _N_SAE
    percentile_table = t.arange(1, 100, dtype=t.float64).repeat(n_features, 1)
    cluster_definition = FeatureClusterDefinition(clusters=((38, 39),))
    resolved_clusters = ResolvedFeatureClusters(
        source_path=Path("d1-clusters.yaml"),
        definition=cluster_definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            cluster_definition,
            n_sae_features=n_features,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(
        resolved_clusters,
        FeatureMask(FeatureMaskMode.blacklist, (39,)),
    )
    contract: Any = SimpleNamespace(
        feature_contract_fingerprint="a" * 64,
        representative_universe=tuple(
            int(feature_id)
            for feature_id in t.nonzero(
                consolidation.cluster_allowed_universe,
                as_tuple=False,
            )
            .squeeze(1)
            .tolist()
        ),
        statistics=SimpleNamespace(
            percentile_table_checksum=compute_percentile_checksum(percentile_table),
        ),
        consolidation=consolidation,
        selection=SimpleNamespace(num_buckets=_NUM_BUCKETS),
    )

    extraction_phases: list[str] = []
    phase = "direct"

    def deterministic_activations(items: Any, **_kwargs: Any) -> t.Tensor:
        extraction_phases.append(phase)
        rows = []
        for item in items:
            seed = (
                sum(item.prefix_text.encode()) * 257
                + sum(item.completion_text.encode())
            )
            rows.append(
                t.tensor(
                    [((seed + 17 * fid) % 101) for fid in range(n_features)],
                    dtype=t.float32,
                )
            )
        return t.stack(rows)

    monkeypatch.setattr(
        feature_diffmean,
        "_max_pooled_sae_activations",
        deterministic_activations,
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (5, 3))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 8)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 128)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [
            f"member:{prefix.core_prefix_id}" for prefix in values
        ],
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        gcg_bags,
        "_group_g_steer",
        lambda *_args, **_kwargs: lambda _members, _bag: 0.0,
    )
    monkeypatch.setattr(
        gcg_bags,
        "_objective_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        gcg_bags,
        "checkpoint_vocab_fingerprint",
        lambda _model: "uncompacted",
    )

    direct_warms: list[FeatureConditioning] = []
    cached_warms: list[FeatureConditioning] = []
    captured_warms = direct_warms

    def capture_warm_start(
        _model: Any,
        _members: Any,
        *,
        warm_start: FeatureConditioning,
        **_kwargs: Any,
    ) -> FeatureConditioning:
        captured_warms.append(warm_start)
        return warm_start

    monkeypatch.setattr(
        gcg_bags,
        "optimize_feature_conditioning_universal",
        capture_warm_start,
    )

    concept_id = 7
    mode = SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,), split_ratio=0.6)
    tokenizer = _StubTokenizer()
    context_map = _context_map(prefixes, tokenizer)
    model = object()
    base_model = _StubBaseModel()
    sae = object()

    direct_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="direct")
    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        cfg=cfg,
        paths=direct_paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=context_map,
        cluster_contract=contract,
    )
    build_ours_gcg_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        cfg=cfg,
        paths=direct_paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=context_map,
        cluster_contract=contract,
    )
    direct_extraction_calls = len(extraction_phases)

    phase = "cache"
    cache = build_clustered_warm_start_cache(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        cfg=cfg,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=context_map,
        fair_modes={mode},
        roofline_modes={mode},
        cluster_contract=contract,
    )
    cache_extraction_calls = len(extraction_phases) - direct_extraction_calls

    phase = "cached consumers"
    captured_warms = cached_warms
    cached_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="cached")
    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        cfg=cfg,
        paths=cached_paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=context_map,
        cluster_contract=contract,
        clustered_warm_starts=cache,
    )
    build_ours_gcg_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=base_model,
        sae=sae,
        percentile_table=percentile_table,
        cfg=cfg,
        paths=cached_paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=context_map,
        cluster_contract=contract,
        clustered_warm_starts=cache,
    )

    expected_entries = [
        ((0, 1), (1, 0), (2, 1), (7, 0), (13, 0), (24, 1), (30, 0), (36, 0)),
        ((0, 1), (1, 1), (2, 0), (3, 0), (31, 1), (37, 0)),
        ((0, 1), (1, 1), (4, 1), (10, 0), (16, 0), (22, 0), (28, 0), (34, 0)),
    ]
    assert [warm.entries for warm in direct_warms] == expected_entries
    assert cached_warms == direct_warms
    assert direct_extraction_calls == 6
    assert cache_extraction_calls == 6
    assert "cached consumers" not in extraction_phases

    _optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes,
        ratio=cfg.ours_gcg.split_ratio,
        seed=cfg.ours_gcg.split_seed,
    )
    artifact_keys: list[tuple[str, str | None]] = [("fair", None)]
    artifact_keys.extend(
        ("roofline", prefix.core_prefix_id) for prefix in held_out_prefixes
    )
    for mode_tag, prefix_id in artifact_keys:
        direct_path = direct_paths.ours_gcg_bag(
            concept_id,
            mode,
            0.0,
            mode_tag,
            prefix_id=prefix_id,
        )
        cached_path = cached_paths.ours_gcg_bag(
            concept_id,
            mode,
            0.0,
            mode_tag,
            prefix_id=prefix_id,
        )
        direct_bag, direct_metadata = read_ours_gcg_bag(
            direct_path,
            expected_feature_contract_fingerprint=(
                contract.feature_contract_fingerprint
            ),
        )
        cached_bag, cached_metadata = read_ours_gcg_bag(
            cached_path,
            expected_feature_contract_fingerprint=(
                contract.feature_contract_fingerprint
            ),
        )
        assert cached_bag == direct_bag
        assert cached_metadata == direct_metadata
        assert FeatureConditioning.from_dict(cached_bag).digest() == (
            FeatureConditioning.from_dict(direct_bag).digest()
        )


# ─────────────────────────────────────────────────────────────────────────────
# (a)/(b)/(d) — artifacts, g_steer improvement, held-out recording
# ─────────────────────────────────────────────────────────────────────────────


def test_fair_derive_writes_bags_and_refines_g_steer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0, 0.5))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="derive")
    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    # (a) one bag per (concept, mode, lambda).
    for reg_lambda in (0.0, 0.5):
        path = paths.ours_gcg_bag(concept_id, mode, reg_lambda, "fair")
        assert path.exists(), f"missing bag for lambda={reg_lambda}: {path}"

    bag_dict, metadata = read_ours_gcg_bag(
        paths.ours_gcg_bag(concept_id, mode, 0.0, "fair")
    )
    assert bag_dict, "refined bag is unexpectedly empty"
    assert all(0 <= bucket < _NUM_BUCKETS for bucket in bag_dict.values())

    # (b) GCG (lambda=0) refines g_steer on the optimize members vs the warm-start.
    optimize_prefixes, _held = split_prefixes_for_concept(prefixes, ratio=0.6, seed=0)
    optimize_members = _members(optimize_prefixes)
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    refined = FeatureConditioning.from_dict(bag_dict)
    g_warm = mean_g_steer(model, optimize_members, warm, mode=mode, device=_CPU)
    g_refined = mean_g_steer(model, optimize_members, refined, mode=mode, device=_CPU)
    assert g_refined <= g_warm, f"GCG must not worsen g_steer: {g_refined} > {g_warm}"
    assert g_warm - g_refined > 1e-4, (
        f"vacuous fixture: GCG did not move g_steer (warm={g_warm}, refined={g_refined})"
    )
    # the recorded metadata agrees with a fresh measurement.
    assert float(metadata["optimize_g_steer"]) == pytest.approx(g_refined, abs=1e-5)
    assert float(metadata["warm_start_g_steer"]) == pytest.approx(g_warm, abs=1e-5)

    # (d) held-out g_steer computed + recorded (finite — held-out split is non-empty).
    assert metadata["mode"] == mode.value
    assert metadata["mode_tag"] == "fair"
    assert int(metadata["n_held_out_members"]) >= 1
    held_out_recorded = float(metadata["held_out_g_steer"])
    assert held_out_recorded == held_out_recorded, "held_out_g_steer must be finite"


def test_fair_derive_runs_optimizer_on_multi_segment_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL optimizer completes on prefixes with TWO real-feature context segments.

    The other derive tests use a single empty-feature context segment (to keep the
    sign-trap teeth calibrated on the 2-slot layout).  This drives
    ``optimize_feature_conditioning_universal`` through the tiny encdec on a 3-slot
    layout — two FIXED context bags (real feature ids/buckets) + one optimized
    generation bag, with each prefix token routed to its own context slot by the
    multi-segment seg_map — closing the multi-segment encoder-forward path on CPU
    (the real d12 model uses the same encoder/cross-attention mechanism the score
    stage already exercises on multi-segment prefixes).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="multiseg")
    _patch_warm(monkeypatch)
    tok = _StubTokenizer()
    ctx: dict[str, tuple[ContextFeatureSegment, ...]] = {
        p.prefix_hash: _two_real_feature_segments(p, tok) for p in prefixes
    }

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
        context_segments_by_prefix_hash=ctx,
    )

    # The derive completed the multi-slot forward + optimize and wrote a valid bag.
    path = paths.ours_gcg_bag(concept_id, mode, 0.0, "fair")
    assert path.exists(), f"multi-segment derive wrote no bag: {path}"
    bag_dict, _metadata = read_ours_gcg_bag(path)
    assert bag_dict, "multi-segment refined bag is unexpectedly empty"
    assert all(0 <= bucket < _NUM_BUCKETS for bucket in bag_dict.values())


def test_fair_derive_raises_on_empty_optimize_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A split_ratio too small to yield any optimize prefix RAISES (no silent skip).

    With 2 prefixes and split_ratio=0.4, ``int(2 * 0.4) == 0`` — the optimize
    partition is empty, so there is nothing to GCG-optimize. The derive must raise
    loudly rather than persist an empty scored set and continue.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(2)]
    cfg = _make_cfg(split_ratio=0.4)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="empty_opt")
    with pytest.raises(AxBenchBpbAdvError, match="no optimize members"):
        _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)


def test_fair_derive_raises_on_single_prefix_concept_names_concept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A (concept, mode) group with only one prefix makes the FAIR derive raise.

    The error message must name the offending concept_id and mode so an operator
    can act on it (ensure >= 2 accepted prefixes in the source corpus or exclude
    the concept). This behavior was previously untested.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 99, SteeringMode.NEGATIVE
    # Only ONE prefix for this concept — the fair derive needs >= 2 to hold out.
    prefixes = [_make_prefix(concept_id, mode, 0)]
    cfg = _make_cfg()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="single_pfx")
    with pytest.raises(AxBenchBpbAdvError, match=">= 2 prefixes") as exc_info:
        _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    msg = str(exc_info.value)
    assert str(concept_id) in msg, f"error message must name the concept_id; got: {msg}"
    assert mode.value in msg, f"error message must name the mode; got: {msg}"


# ─────────────────────────────────────────────────────────────────────────────
# (e) — sign trap on the held-out margin
# ─────────────────────────────────────────────────────────────────────────────


def test_sign_trap_held_out_margin_moves_the_right_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optimized bag lowers held-out g_steer (strengthens steering).

    This is the permanent sign guard: ``build_ours_gcg_bags_fair`` builds the
    objective via ``class_margin_objective(mode=group.mode)``.  Passing the WRONG
    mode (or negating g_steer) makes the optimizer MAXIMIZE the correctly-signed
    margin, so the held-out g_steer rises above the warm-start's and this
    assertion REDs (RED-proved by mutation in the task report).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="signtrap")
    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    _opt, held_out_prefixes = split_prefixes_for_concept(prefixes, ratio=0.6, seed=0)
    held_out_members = _members(held_out_prefixes)
    assert held_out_members, "fixture must produce a non-empty held-out split"

    bag_dict, metadata = read_ours_gcg_bag(
        paths.ours_gcg_bag(concept_id, mode, 0.0, "fair")
    )
    refined = FeatureConditioning.from_dict(bag_dict)
    warm = FeatureConditioning.from_dict(dict(_WARM_BAG))
    g_held_warm = mean_g_steer(model, held_out_members, warm, mode=mode, device=_CPU)
    g_held_refined = mean_g_steer(model, held_out_members, refined, mode=mode, device=_CPU)

    # CORRECT mode → the held-out margin moves the RIGHT way (g_steer drops).
    assert g_held_refined < g_held_warm, (
        f"held-out margin moved the wrong way: refined={g_held_refined} "
        f">= warm={g_held_warm} (a flipped derive mode would do this)"
    )
    # the recorded held-out g_steer is the refined-bag measurement.
    assert float(metadata["held_out_g_steer"]) == pytest.approx(
        g_held_refined, abs=1e-5
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) — determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_determinism_same_seed_identical_bag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same ``master_seed`` ⇒ identical bag content + metadata.

    Identity is asserted on the DECODED bag (``{fid: bucket}``) and the metadata
    dict, NOT the raw file bytes: ``safetensors`` serializes the ``__metadata__``
    block through a Rust ``HashMap`` whose key order is not stable across runs, so
    two semantically-identical bags can differ byte-for-byte in the header (the
    repo-wide "byte-identity == decompressed tensors" convention).
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,))

    paths_a = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run_a")
    paths_b = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run_b")
    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths_a)
    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths_b)

    bag_a, meta_a = read_ours_gcg_bag(paths_a.ours_gcg_bag(concept_id, mode, 0.0, "fair"))
    bag_b, meta_b = read_ours_gcg_bag(paths_b.ours_gcg_bag(concept_id, mode, 0.0, "fair"))
    assert bag_a == bag_b, f"non-deterministic bag content: {bag_a} != {bag_b}"
    assert meta_a == meta_b, "non-deterministic bag metadata across identical-seed runs"


# ─────────────────────────────────────────────────────────────────────────────
# Grouping + method-activation predicates (model-free)
# ─────────────────────────────────────────────────────────────────────────────


def test_group_prefixes_by_concept_mode_order_and_split() -> None:
    prefixes = [
        _make_prefix(7, SteeringMode.POSITIVE, 0),
        _make_prefix(7, SteeringMode.NEGATIVE, 0),
        _make_prefix(7, SteeringMode.POSITIVE, 1),
        _make_prefix(9, SteeringMode.POSITIVE, 0),
    ]
    groups = gcg_bags._group_prefixes_by_concept_mode(prefixes)
    keys = [(cid, mode.value, len(members)) for cid, mode, members in groups]
    assert keys == [
        (7, "positive", 2),
        (7, "negative", 1),
        (9, "positive", 1),
    ]


def test_fair_gcg_methods_active_predicates() -> None:
    # None => all methods active => both fair modes derive.
    assert active_fair_gcg_modes(_make_cfg(base_methods=None)) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert fair_gcg_methods_active(_make_cfg(base_methods=None))
    assert active_fair_gcg_modes(_make_cfg(base_methods=["ours_gcg_encdec"])) == {
        SteeringMode.POSITIVE
    }
    assert active_fair_gcg_modes(_make_cfg(base_methods=["neg_ours_gcg_encdec"])) == {
        SteeringMode.NEGATIVE
    }
    assert active_fair_gcg_modes(_make_cfg(base_methods=["diffmean"])) == set()
    assert not fair_gcg_methods_active(_make_cfg(base_methods=["diffmean"]))


# ─────────────────────────────────────────────────────────────────────────────
# Bag I/O round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_bag_io_roundtrip(tmp_path: Path) -> None:
    bag = FeatureConditioning.from_dict({3: 0, 17: 1, 8: 1})
    path = tmp_path / "bag.safetensors"
    write_ours_gcg_bag(
        path,
        bag,
        metadata={"mode": "positive", "reg_lambda": "0.0"},
        vocab_fingerprint="uncompacted",
    )
    loaded, metadata = read_ours_gcg_bag(path)
    assert loaded == {3: 0, 8: 1, 17: 1}
    assert metadata["mode"] == "positive"
    assert metadata["reg_lambda"] == "0.0"
    assert metadata["vocab_fingerprint"] == "uncompacted"


# ─────────────────────────────────────────────────────────────────────────────
# (f) — SAE-identity guard
# ─────────────────────────────────────────────────────────────────────────────


class _StubEncoderConfig:
    n_sae_features = 16384  # == SAEWidth("16k")


class _StubEncoder:
    config = _StubEncoderConfig()


class _StubModel:
    encoder = _StubEncoder()

    def __init__(self, sae_block: dict[str, Any] | None, *, include_sae: bool = True) -> None:
        user_config: dict[str, Any] = {}
        if include_sae:
            user_config["sae"] = sae_block
        self._loaded_meta_data = {"user_config": user_config}


def test_sae_guard_matching_ok() -> None:
    # No feature_dir (the feature_dir leg is skipped) — still ok.
    assert_sae_matches_encdec(_make_cfg(), _StubModel(dict(_SAE_META)))
    # A MATCHING feature_dir (layer 8, 16k) — the feature_dir leg passes too.
    cfg = _make_cfg(feature_dir=_feature_dir(8, SAEWidth.W_16K))
    assert_sae_matches_encdec(cfg, _StubModel(dict(_SAE_META)))


def test_unclustered_sae_guard_ignores_clustered_local_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-signed guard preserves the exact legacy model-id source at None."""
    cfg = _make_cfg(
        neuronpedia_model_id="gemma-2-2b",
        feature_extraction_model_id="same-shape-different-local-model",
    )
    monkeypatch.setattr(
        gcg_bags,
        "resolve_sae",
        lambda **_kwargs: pytest.fail(
            "unclustered guard entered the clustered exact-resolution branch"
        ),
    )

    assert_sae_matches_encdec(
        cfg,
        _StubModel(dict(_SAE_META)),
        cluster_contract=None,
    )


def test_clustered_sae_guard_preserves_tier_and_checks_exact_resolved_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(
        neuronpedia_source="22-gemmascope-2-res-16k-big",
        neuronpedia_model_id="gemma-3-4b-pt",
    )
    checkpoint_sae = {
        "release": "gemmascope_2",
        "model": "gemma-3-4b-pt",
        "activation_type": "residual",
        "layer": 22,
        "width": "16k",
        "l0": "big",
    }
    resolved = type(
        "Resolved",
        (),
        {
            "release": "release-big",
            "sae_id": "layer_22_width_16k_l0_big",
            "hook_name": "blocks.22.hook_resid_post",
        },
    )()
    contract: Any = SimpleNamespace(
        sae=SimpleNamespace(
            model="gemma-3-4b-pt",
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        )
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        gcg_bags,
        "assert_cluster_contract_loaded_state",
        lambda *_args, **_kwargs: None,
    )

    def _resolve(**kwargs: Any) -> Any:
        observed.update(kwargs)
        return resolved

    monkeypatch.setattr(gcg_bags, "resolve_sae", _resolve)

    assert_sae_matches_encdec(
        cfg,
        _StubModel(checkpoint_sae),
        cluster_contract=contract,
    )
    assert observed["l0"] is not None

    contract.sae.sae_id = "same-shape-different-sae"
    with pytest.raises(AxBenchBpbAdvError, match="does not match the feature contract"):
        assert_sae_matches_encdec(
            cfg,
            _StubModel(checkpoint_sae),
            cluster_contract=contract,
        )


def test_clustered_sae_guard_uses_local_model_not_neuronpedia_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(
        neuronpedia_source="22-gemmascope-2-res-16k-big",
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
    )
    checkpoint_sae = {
        "release": "gemmascope_2",
        "model": "gemma-3-4b-pt",
        "activation_type": "residual",
        "layer": 22,
        "width": "16k",
        "l0": "big",
    }
    resolved = SimpleNamespace(
        release="release-big",
        sae_id="layer_22_width_16k_l0_big",
        hook_name="blocks.22.hook_resid_post",
    )
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(
            sae=SimpleNamespace(
                model="gemma-3-4b-pt",
                activation_type="residual",
                layer=22,
                width="16k",
                release=resolved.release,
                sae_id=resolved.sae_id,
                hook_name=resolved.hook_name,
            )
        ),
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        gcg_bags,
        "assert_cluster_contract_loaded_state",
        lambda *_args, **_kwargs: None,
    )

    def _resolve(**kwargs: object) -> object:
        seen.update(kwargs)
        return resolved

    monkeypatch.setattr(gcg_bags, "resolve_sae", _resolve)

    assert_sae_matches_encdec(
        cfg,
        _StubModel(checkpoint_sae),
        cluster_contract=contract,
    )
    assert seen["model"] == "gemma-3-4b-pt"


def test_clustered_sae_mismatch_names_feature_extraction_model_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(
        neuronpedia_source="22-gemmascope-2-res-16k-big",
        neuronpedia_model_id="neuronpedia-namespace-model",
        feature_extraction_model_id="local-extraction-model",
    )
    checkpoint_sae = {
        "release": "gemmascope_2",
        "model": "different-checkpoint-model",
        "activation_type": "residual",
        "layer": 22,
        "width": "16k",
        "l0": "big",
    }
    resolved = SimpleNamespace(
        release="release-big",
        sae_id="layer_22_width_16k_l0_big",
        hook_name="blocks.22.hook_resid_post",
    )
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(
            sae=SimpleNamespace(
                model="local-extraction-model",
                activation_type="residual",
                layer=22,
                width="16k",
                release=resolved.release,
                sae_id=resolved.sae_id,
                hook_name=resolved.hook_name,
            )
        ),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_cluster_contract_loaded_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(gcg_bags, "resolve_sae", lambda **_kwargs: resolved)

    with pytest.raises(AxBenchBpbAdvError, match="SAE-identity mismatch") as exc_info:
        assert_sae_matches_encdec(
            cfg,
            _StubModel(checkpoint_sae),
            cluster_contract=contract,
        )

    message = str(exc_info.value)
    assert "feature_extraction_model_id='local-extraction-model'" in message
    assert "neuronpedia_model_id='local-extraction-model'" not in message


def test_clustered_sae_guard_wraps_loaded_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gcg_bags,
        "assert_cluster_contract_loaded_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            gcg_bags.ClusterFeatureContractError("loaded contract mismatch")
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="loaded contract mismatch") as exc_info:
        assert_sae_matches_encdec(
            _make_cfg(),
            _StubModel(dict(_SAE_META)),
            cluster_contract=cast(ClusterFeatureContract, SimpleNamespace()),
        )
    assert isinstance(exc_info.value.__cause__, gcg_bags.ClusterFeatureContractError)


def test_clustered_sae_guard_reports_exact_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gcg_bags,
        "assert_cluster_contract_loaded_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        gcg_bags,
        "resolve_sae",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("no exact SAE")),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="cannot resolve the clustered BPB-Adv SAE exactly: no exact SAE",
    ) as exc_info:
        assert_sae_matches_encdec(
            _make_cfg(),
            _StubModel(dict(_SAE_META)),
            cluster_contract=cast(ClusterFeatureContract, SimpleNamespace()),
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_sae_guard_mismatched_layer_raises() -> None:
    # Same width (16k) but a DIFFERENT layer (8 -> 12): the silent-corruption case.
    cfg = _make_cfg()
    mismatched = dict(_SAE_META, layer=12)
    model = _StubModel(mismatched)
    with pytest.raises(AxBenchBpbAdvError, match="SAE-identity mismatch"):
        assert_sae_matches_encdec(cfg, model)


def test_sae_guard_mismatched_source_string_raises() -> None:
    # The cfg points at a different SAE (layer 12) than the encdec recorded (8).
    cfg = _make_cfg(neuronpedia_source="12-gemmascope-res-16k")
    model = _StubModel(dict(_SAE_META))
    with pytest.raises(AxBenchBpbAdvError, match="SAE-identity mismatch"):
        assert_sae_matches_encdec(cfg, model)


def test_sae_guard_missing_sae_block_raises() -> None:
    cfg = _make_cfg()
    model = _StubModel(None, include_sae=False)
    with pytest.raises(AxBenchBpbAdvError, match="no 'sae' block"):
        assert_sae_matches_encdec(cfg, model)


def test_sae_guard_mismatched_feature_dir_raises() -> None:
    # neuronpedia_source + encdec SAE agree (layer 8), but feature_dir points at a
    # layer-12 SAE — the percentile_table would come from a different SAE.
    cfg = _make_cfg(feature_dir=_feature_dir(12, SAEWidth.W_16K))
    model = _StubModel(dict(_SAE_META))
    with pytest.raises(AxBenchBpbAdvError, match="feature_dir SAE mismatch"):
        assert_sae_matches_encdec(cfg, model)


def test_sae_guard_is_a_feature_gcg_error() -> None:
    # AxBenchBpbAdvError subclasses FeatureGcgError, so callers catching the base win.
    cfg = _make_cfg(neuronpedia_source="12-gemmascope-res-16k")
    model = _StubModel(dict(_SAE_META))
    with pytest.raises(FeatureGcgError):
        assert_sae_matches_encdec(cfg, model)


# ─────────────────────────────────────────────────────────────────────────────
# Batch-knob threading (review #2/#12 teeth) + empty-held-out backstop
# ─────────────────────────────────────────────────────────────────────────────


def test_fair_derive_threads_nondefault_batch_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NON-DEFAULT knob values reach EVERY g_steer call and objective build.

    A refactor that drops one of the hand-threaded ``batch_size=…`` /
    ``max_forward_batch=…`` bindings silently reverts that site to the module
    default; with defaults distinct from the configured values (3 / 5 vs
    16 / 8) any such drop diverges here.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(5)]
    cfg = _make_cfg(mean_g_steer_batch=3, max_forward_batch=5, max_grad_batch=7)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="knobs")

    real_mean_g_steer = gcg_bags.mean_g_steer
    g_steer_knobs: list[tuple[int, int]] = []

    def _spy_mean_g_steer(*args: Any, **kwargs: Any) -> float:
        g_steer_knobs.append(
            (kwargs["batch_size"], kwargs["max_forward_batch"])
        )
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
    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    # warm + optimize + held-out => three measurements, ALL under (3, 5).
    assert len(g_steer_knobs) == 3
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


def test_fair_derive_backstop_raises_on_empty_held_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty held-out split raises BEFORE any GCG compute (defense in depth).

    OursGcgConfig already bounds split_ratio < 1, so this can only fire if the
    split function's arithmetic changes — the backstop keeps 'burn the budget,
    persist an EMPTY fair scored set, silently drop every fair score' impossible.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(3)]
    cfg = _make_cfg()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="empty_held")
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group), []),
    )
    with pytest.raises(AxBenchBpbAdvError, match="no HELD-OUT prefixes"):
        _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)
    # Nothing was persisted for the aborted group.
    assert not paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, "fair").exists()


def test_fair_derive_raises_on_held_out_prefix_tokens_text_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A HELD-OUT prefix whose tokens drift from its text raises (max-review #5).

    The round-trip check was only on optimize/roofline prefixes; the fair
    held-out prefixes — which feed held_out_g_steer (the persisted generalization
    metric) and the score stage — were not checked. A held-out prefix whose
    stored tokens disagree with its text must raise, symmetric with the
    optimize-subset drift raise.
    """
    model = _build_model(monkeypatch)
    clean = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(2)]
    drifted = BpbAttackPrefix(
        core_prefix_id="held-out-drift",
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        prefix_text="99 98 97",  # text no longer matches the stored tokens
        prefix_hash="h-drift",
        prefix_tokens=encode_prefix("10 11 13", _StubTokenizer()),
        concept_rich_sample_ids=["r0"],
        concept_avoiding_sample_ids=["a0"],
        concept_rich_output_tokens=[[20, 21, 22, 23]],
        concept_avoiding_output_tokens=[[30, 31, 32, 33]],
    )
    cfg = _make_cfg()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="held_drift")
    # Force the drifted prefix into the HELD-OUT split (clean ones optimize).
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(clean), [drifted]),
    )
    with pytest.raises(AxBenchBpbAdvError, match="round-trip"):
        _run_derive(
            monkeypatch,
            model=model,
            prefixes=[*clean, drifted],
            cfg=cfg,
            paths=paths,
        )


def test_fair_derive_raises_on_missing_context_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prefix absent from ``context_segments_by_prefix_hash`` → loud raise.

    The context cache MUST carry an entry for every optimize/held-out prefix (the
    score stage conditions each one on its own segments).  A missing entry is a
    real pipeline inconsistency — the derive cannot build that prefix's members —
    so it raises rather than silently skipping the prefix from the scored set.
    """
    model = _build_model(monkeypatch)
    prefixes = [_make_prefix(7, SteeringMode.POSITIVE, i) for i in range(5)]
    cfg = _make_cfg()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="missing_ctx")
    _patch_warm(monkeypatch)

    tok = _StubTokenizer()
    ctx = _context_map(prefixes, tok)
    # Simulate a context cache missing ONE prefix's entry.
    del ctx[prefixes[0].prefix_hash]

    with pytest.raises(AxBenchBpbAdvError, match="no context-cache segments"):
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
            context_segments_by_prefix_hash=ctx,
        )


# --- org-prefix tolerance on the checkpoint SAE-identity guard -----------------
#
# SAELens force-prefixes Gemma-3 model ids, so a config naming the canonical
# ``google/gemma-3-4b-pt`` resolves prefixed while an encdec checkpoint records
# the bare feature-path token ``gemma-3-4b-pt``. The guard compared the raw
# strings and rejected that correctly-matched pair. A real clustered run hit it:
#   "resolves to ('google/gemma-3-4b-pt', ...), which does NOT match the encdec
#    checkpoint's recorded training SAE ('gemma-3-4b-pt', ...)"
# Only the prefix differed. This is not clustered-specific — the same false
# rejection reaches an unclustered run whose ``neuronpedia_model_id`` carries an
# org prefix, which is why these pin the guard at ``cluster_contract=None``
# (that branch never calls ``resolve_sae``, so they stay hermetic).


def test_sae_guard_accepts_org_prefixed_config_against_bare_checkpoint() -> None:
    """Config prefixed, checkpoint bare: the same SAE, so no raise."""
    cfg = _make_cfg(neuronpedia_model_id="google/gemma-2-2b")
    assert_sae_matches_encdec(cfg, _StubModel(dict(_SAE_META)))


def test_sae_guard_accepts_bare_config_against_org_prefixed_checkpoint() -> None:
    """The tolerance is symmetric — a checkpoint may record the prefixed id."""
    prefixed_meta = dict(_SAE_META) | {"model": "google/gemma-2-2b"}
    cfg = _make_cfg(neuronpedia_model_id="gemma-2-2b")
    assert_sae_matches_encdec(cfg, _StubModel(prefixed_meta))


def test_sae_guard_accepts_prefixed_on_both_sides() -> None:
    """Normalization is idempotent, so a prefixed/prefixed pair still agrees."""
    prefixed_meta = dict(_SAE_META) | {"model": "google/gemma-2-2b"}
    cfg = _make_cfg(neuronpedia_model_id="google/gemma-2-2b")
    assert_sae_matches_encdec(cfg, _StubModel(prefixed_meta))


@pytest.mark.parametrize(
    "configured",
    [
        "google/gemma-2-9b",  # prefixed, genuinely different model
        "gemma-2-9b",  # bare, genuinely different model
    ],
)
def test_sae_guard_still_rejects_a_different_model(configured: str) -> None:
    """Anti-vacuity: stripping the org must not stop the guard from guarding.

    If normalization were implemented as "ignore the model id", every test above
    would pass while a real same-shape-different-model misconfig — the exact
    silent bag corruption this guard exists to catch — would sail through.
    """
    cfg = _make_cfg(neuronpedia_model_id=configured)
    with pytest.raises(AxBenchBpbAdvError, match="SAE-identity mismatch"):
        assert_sae_matches_encdec(cfg, _StubModel(dict(_SAE_META)))


def test_sae_guard_rejects_same_suffix_under_a_different_org() -> None:
    """Pin the documented limit rather than leave it to be rediscovered.

    A checkpoint records no org, so ``other-org/gemma-2-2b`` is indistinguishable
    from ``google/gemma-2-2b`` here and is ACCEPTED. ``_model_names_agree`` in
    cluster_checkpoint_pin documents the same gap. If a future change makes an org
    verifiable, this test should flip to expecting a raise.
    """
    cfg = _make_cfg(neuronpedia_model_id="other-org/gemma-2-2b")
    assert_sae_matches_encdec(cfg, _StubModel(dict(_SAE_META)))


# ─────────────────────────────────────────────────────────────────────────────
# Class-margin alpha: the REAL derive at a non-default alpha
# ─────────────────────────────────────────────────────────────────────────────


def test_fair_derive_sweeps_alpha_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real derive over an alpha grid writes ONE bag per alpha, correctly keyed.

    Every other alpha test is a pure-function unit test (grid construction, path
    formatting, identity stamping). This is the only place the WHOLE chain runs
    at alpha != 1: config -> `_lambda_alpha_grid` -> a per-alpha
    `_objective_factory` -> `optimize_feature_conditioning_universal` ->
    `_write_ours_gcg_bag` -> the on-disk path and stamped identity.

    Pins, in order of what would break silently without them:
      (a) alpha = 1.0 lands at the LEGACY path (no suffix) -- the strip-default
          rule that lets every pre-alpha bag on disk still resolve;
      (b) each non-unit alpha lands at its OWN sibling path, so a sweep never
          overwrites the plain-margin bag it reuses;
      (c) only the non-unit bags carry the `objective_alpha` identity key;
      (d) the bags actually DIFFER -- if alpha never reached the objective, the
          optimizer would return the same bag under three different names and
          every path/identity assertion above would still pass.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    alphas = (1.0, 0.5, 0.0)
    cfg = _make_cfg(reg_lambda_grid=(0.0,), objective_alpha_grid=alphas)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="alpha-derive")

    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    written: dict[float, Any] = {}
    for alpha in alphas:
        path = paths.ours_gcg_bag(concept_id, mode, 0.0, "fair", objective_alpha=alpha)
        assert path.exists(), f"missing bag for alpha={alpha}: {path}"
        bag, metadata = read_ours_gcg_bag(path)
        written[alpha] = (path, bag, metadata)

    # (a) alpha = 1.0 IS the legacy path a pre-alpha derive would have written.
    assert written[1.0][0].name == "lambda_0.0.safetensors"
    # (b) distinct siblings, same parent directory.
    paths_by_alpha = {alpha: entry[0] for alpha, entry in written.items()}
    assert len({str(p) for p in paths_by_alpha.values()}) == len(alphas)
    assert len({p.parent for p in paths_by_alpha.values()}) == 1

    # (c) strip-default in the STAMPED metadata, not just in the helper.
    assert "objective_alpha" not in written[1.0][2]
    assert written[0.5][2]["objective_alpha"] == "0.5"
    assert written[0.0][2]["objective_alpha"] == "0.0"

    # (d) NON-VACUITY: alpha must have reached the objective. Without this the
    # test passes even if `_objective_factory` dropped alpha entirely.
    bags = {alpha: entry[1] for alpha, entry in written.items()}
    assert any(
        bags[1.0] != bags[other] for other in (0.5, 0.0)
    ), f"alpha never changed the derived bag -- all identical: {bags[1.0]}"


def test_fair_derive_alpha_bags_do_not_disturb_the_encdec_fit_glob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An alpha sweep must not break #1249's `load_encdec_fit_conditioning`.

    That loader globs ``feature_diffmean/<mode>/c<id>/lambda_*.safetensors`` and
    requires EXACTLY ONE match. The alpha filenames (``lambda_0.0_alpha_0.5``)
    match that glob pattern, so the invariant holds only because alpha bags land
    in the ``fair`` family directory while the fit store reads
    ``feature_diffmean``. That separation is load-bearing and nothing else pins
    it -- if a future change gave the feature-DiffMean derive an alpha axis, the
    loader would start seeing N bags and #1249's conditioned fits would break.
    """
    model = _build_model(monkeypatch)
    concept_id, mode = 7, SteeringMode.POSITIVE
    prefixes = [_make_prefix(concept_id, mode, i) for i in range(5)]
    cfg = _make_cfg(reg_lambda_grid=(0.0,), objective_alpha_grid=(1.0, 0.5, 0.0))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="alpha-glob")

    _run_derive(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    fair_dir = paths.ours_gcg_bag(
        concept_id, mode, 0.0, "fair", objective_alpha=0.5
    ).parent
    diffmean_dir = paths.ours_gcg_bag(
        concept_id, mode, 0.0, "feature_diffmean"
    ).parent
    assert fair_dir != diffmean_dir, "alpha bags share the fit store's directory"
    # The sweep really did produce multiple glob-matching siblings...
    assert len(sorted(fair_dir.glob("lambda_*.safetensors"))) == 3
    # ...none of which landed where the encdec fit loader looks.
    assert sorted(diffmean_dir.glob("lambda_*.safetensors")) == []
