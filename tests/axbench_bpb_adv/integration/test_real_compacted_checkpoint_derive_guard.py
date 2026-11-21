"""REAL minted compacted checkpoint: the feature-DiffMean AND feature-GCG fair derives RUN
V-correct (#695/#720 PR-2).

The REAL-DATA complement to the synthetic ``test_gcg_bags_compacted_guard.py``. Task 6a drove the
real ``axbench_bpb_adv`` feature-DiffMean *fair* derive against the REAL minted compacted checkpoint
(``smoke_align_compact_0715``, ``|V|=15458`` of ``n_sae=16384``) and hit a raw device-side CUDA OOV
assert inside the encoder forward — because ``mean_g_steer`` conditioned the compacted encoder on
UN-restricted, original-SAE-id features. #695 PR-1 turned that into a clean ``NotImplementedError``;
#720 PR-2 then made the path actually V-correct (T1: the universal optimizer proposes/scores only ids
in ``V``; T2: every derive restricts its warm bags + contexts to ``V``), so the blanket guard is now
LIFTED (T3). T4-Step-4 (below, §4) adds the same real-checkpoint regression for T1's own consumer —
the feature-GCG *fair* derive (``build_ours_gcg_bags_fair`` → ``optimize_feature_conditioning_universal``)
— which Task 6a's DiffMean crash never reached but sits on the exact same OOV hazard.

This test proves, against the SAME real checkpoint loaded by the SAME production loader
(``load_model(source="encdec", ..., eval_conditioning=conditions_within_vocab("axbench_bpb"))``, mirroring
``RuntimeState.encdec``), that ``mean_g_steer``, the feature-DiffMean *fair* derive, and the
feature-GCG *fair* derive all now RUN on the compacted checkpoint with in-``V`` conditioning —
returning a finite g_steer and writing bags whose feature ids all lie in ``V`` — instead of raising.
Only THIS file exercises the real loaded ``FeatureVocab`` (real fingerprint, real ``|V|``, real
clean11 ⊆ ``V``), the surface the mock-based unit tests structurally cannot cover.

Approach: the **targeted derive** (not the full CPU pipeline), for both derives. The feature-DiffMean
fair derive is forward-only (DiffMean warm-start, stubbed here, plus g_steer measurements — no GCG
optimize). The feature-GCG fair derive reuses the same stubbed DiffMean warm-start, then runs a REAL
(budget-minimized: 1-2 steps, minibatch/forward-batch of 1) ``optimize_feature_conditioning_universal``
loop against the real compacted encoder — the exact function T1 restricted to propose/score only
ORIGINAL ids in ``V``. Both run on CPU directly. ``pytest.skip`` cleanly when the local, gitignored
checkpoint is absent, so CI never hard-fails; marked ``slow`` because it loads a ~900 MB real
checkpoint.

Known Tier-3 limitation: ``_real_compacted_cluster_overlay`` fabricates the
``CheckpointPin`` digest fields after the real checkpoint has been loaded.
That is consistent with the derive guard's deliberate no-I/O loaded-state
contract; checkpoint-byte binding is established separately by
``RuntimeState.encdec()``. These tests therefore prove real-vocabulary
execution, not that the fabricated pin is bound to the checkpoint's bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as fa_module
import steering.eval.axbench_bpb_adv.feature_diffmean_bags as feature_diffmean_bags_module
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags_module
from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb.vocab_restriction import (
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
)
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.feature_diffmean import encdec_dec_sequence_len
from steering.eval.axbench_bpb_adv.feature_diffmean_bags import (
    build_feature_diffmean_bags_fair,
    build_feature_diffmean_bags_roofline,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    build_ours_gcg_bags_fair,
    mean_g_steer,
)
from steering.eval.axbench_bpb_adv.gcg_family import GcgFamily
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    VOCAB_FINGERPRINT_META_KEY,
    read_ours_gcg_bag,
    read_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import PrecomputedBagFamily
from steering.eval.cluster_contract_identity import (
    CONCEPT_PROJECTION_VERSION,
    FEATURE_CONTRACT_SCHEMA_VERSION,
    FEATURE_DIFFMEAN_PROJECTION_VERSION,
    GENERIC_ROOFLINE_SIGNAL_VERSION,
    PACKED_PERCENTILE_SELECTION_VERSION,
    QC_TARGET_REDUCTION_VERSION,
    _canonical_json,
    _sha256_json,
)
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContract,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_bpb_adv.tokenizer_stubs import StubBaseModel, StubNanochatTokenizer

pytestmark = pytest.mark.slow

# ─────────────────────────────────────────────────────────────────────────────
# The REAL minted compacted checkpoint (Task 6a §1; local-only, gitignored)
# ─────────────────────────────────────────────────────────────────────────────
_CHECKPOINT_TAG = "smoke_align_compact_0715"
_CHECKPOINT_STEP = 4
_MODEL_FILE = f"model_{_CHECKPOINT_STEP:06d}.pt"
#: ``FeatureVocab.fingerprint()`` of the compacted vocabulary ``V`` — the SAME sha256 mirrored into
#: the checkpoint's ``meta_000004.json`` ``feature_vocab.sha256`` and cross-checked by ``build_encdec_model``.
_EXPECTED_VOCAB_FINGERPRINT = "2d9ddc6ed6176b44942a2f567c2c6d9890fd84cbc26346d94c1806b0abc24e48"
_EXPECTED_VOCAB_SIZE = 15458
_N_SAE = 16384
#: clean11_l8 concept ``feature_index`` values — every one must survive in ``V`` (Task 6a §1: the
#: 11 concepts' feature ids ARE exactly ``V ∩ concepts``), else the derive's bag/context go empty.
_CLEAN11_FEATURE_INDEX = (
    1970, 4190, 5409, 5668, 9314, 10482, 10544, 10874, 11837, 13429, 15672,
)
_CPU = t.device("cpu")


def _checkpoint_tag_dir() -> Path:
    """The tag root the production encdec loader depth-first searches for ``model_*.pt``.

    ``get_base_dir()`` resolves to the package's own ``nanochat_cache`` (i.e. this worktree's),
    exactly as ``load_model(source="encdec", ...)`` resolves it — so the skip check and the load
    agree on where the checkpoint must live.
    """
    return Path(get_base_dir()) / "conditional_encdec_checkpoints" / _CHECKPOINT_TAG


def _checkpoint_present() -> bool:
    tag_dir = _checkpoint_tag_dir()
    return tag_dir.is_dir() and any(tag_dir.rglob(_MODEL_FILE))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU lowering; force SDPA for CPU safety.

    The compacted guard fires BEFORE any forward, so no attention actually runs in these tests — this
    is defensive only (keeps the module CPU-safe if a future edit ever moves a forward ahead of the guard).
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


@pytest.fixture(scope="module")
def real_compacted_model() -> Any:
    """Load the REAL minted compacted checkpoint on CPU via the PRODUCTION loader, once per module.

    Mirrors ``RuntimeState.encdec`` (``runtime_state.py:136-144``): ``load_model(source="encdec", ...,
    eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"))`` then attach
    ``_loaded_meta_data`` (the loader returns it separately; the fair derive's
    ``assert_single_positive_bucket`` reads it). Skips the whole module when the gitignored local
    checkpoint is absent so CI without it never hard-fails.
    """
    if not _checkpoint_present():
        pytest.skip(
            f"real compacted checkpoint '{_CHECKPOINT_TAG}' absent under {_checkpoint_tag_dir()} "
            "(local, gitignored artifact — see Task 6a §1)"
        )
    loaded, _tokenizer, meta = load_model(
        source="encdec",
        device="cpu",
        phase="eval",
        model_tag=_CHECKPOINT_TAG,
        step=_CHECKPOINT_STEP,
        eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
    )
    # Duck-typed like TargetModelBundle.model (Any) across this eval layer, matching the sibling
    # synthetic guard test's `_build_model -> Any`.
    model: Any = loaded
    # RuntimeState.encdec attaches this post-load (runtime_state.py:144); the fair derive needs it.
    model._loaded_meta_data = meta
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Real-member / real-bag builders (member CONTENT is inert before the guard — the guard is
# mean_g_steer's first statement after the empty/batch checks — but we build genuine members so
# the derive really reaches that statement with a non-empty set).
# ─────────────────────────────────────────────────────────────────────────────


def _make_prefix(idx: int = 0) -> BpbAttackPrefix:
    """A prefix whose tokens round-trip from its text under the stub nanochat tokenizer."""
    tokenizer = StubNanochatTokenizer()
    prefix_text = f"10 11 {12 + idx} 13"
    return BpbAttackPrefix(
        core_prefix_id=f"c7_positive_p{idx}",
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        prefix_text=prefix_text,
        prefix_hash=f"h7positive{idx}",
        prefix_tokens=encode_prefix(prefix_text, tokenizer),
        concept_rich_sample_ids=["r0"],
        concept_avoiding_sample_ids=["a0"],
        concept_rich_output_tokens=[[20, 21, 22, 23]],
        concept_avoiding_output_tokens=[[30, 31, 32, 33]],
    )


def _segments_for(prefix: BpbAttackPrefix, tokenizer: Any) -> tuple[ContextFeatureSegment, ...]:
    """One EMPTY-feature context segment spanning the decoded prefix (2-slot layout)."""
    char_end = len(tokenizer.decode(list(prefix.prefix_tokens)))
    return (
        ContextFeatureSegment(feature_ids=(), buckets=(), char_start=0, char_end=char_end),
    )


def _members(prefixes: list[BpbAttackPrefix], model: Any) -> list[Any]:
    tokenizer = StubNanochatTokenizer()
    max_seq_len = encdec_dec_sequence_len(model)
    out: list[Any] = []
    for prefix in prefixes:
        out.extend(
            build_class_margin_members(
                prefix,
                prefix_segments=_segments_for(prefix, tokenizer),
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                device=_CPU,
            )
        )
    return out


def _context_map(
    prefixes: list[BpbAttackPrefix], tokenizer: Any
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    return {p.prefix_hash: _segments_for(p, tokenizer) for p in prefixes}


def _stub_feature_diffmean_bag(*_args: Any, **_kwargs: Any) -> FeatureConditioning:
    """Warm-start seam: the real DiffMean SAE forward (base gemma-2-2b + 16k SAE) is not under test
    here and would need the GPU. Its ids are clean11, all ∈ V, so the forward-only ``group_g_steer``
    measurements the derive runs on them condition the compacted encoder without OOV."""
    return FeatureConditioning.from_dict(
        {_CLEAN11_FEATURE_INDEX[0]: 0, _CLEAN11_FEATURE_INDEX[1]: 1}
    )


def _make_cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="gemma-2-2b",
        base_methods=None,
        feature_dir="",
        extraction_tokenizer_id="google/gemma-2-2b",
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=(0.0,),
            split_ratio=0.6,
            split_seed=0,
            max_steps=8,
            minibatch_size=8,
            master_seed=0,
            mean_g_steer_batch=16,
            max_forward_batch=8,
        ),
    )


def _make_gcg_cfg() -> AxBenchBpbAdvConfig:
    """Same run identity as ``_make_cfg`` but a CPU-tractable feature-GCG step budget.

    Unlike the DiffMean derive, ``build_ours_gcg_bags_fair`` runs
    ``optimize_feature_conditioning_universal``'s real GCG loop (a gradient pass plus a batch
    of candidate forwards per step), so ``max_steps`` / ``minibatch_size`` / ``max_forward_batch``
    are pinned to the smallest values ``OursGcgConfig`` accepts (``>= 1``) — enough to exercise
    the real optimizer against the real compacted encoder without turning this into a slow test.
    """
    return AxBenchBpbAdvConfig(
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="gemma-2-2b",
        base_methods=None,
        feature_dir="",
        extraction_tokenizer_id="google/gemma-2-2b",
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=(0.0,),
            split_ratio=0.6,
            split_seed=0,
            max_steps=2,
            minibatch_size=1,
            master_seed=0,
            mean_g_steer_batch=16,
            max_forward_batch=1,
        ),
    )


def _real_compacted_cluster_overlay(
    model: Any,
) -> tuple[ClusterFeatureContract, dict[str, Any]]:
    """Build an active cluster contract that preserves this real checkpoint's V.

    The local minted checkpoint predates static-cluster training. The overlay
    clusters one allowed representative with one already-masked member, so its
    effective vocabulary remains exactly the checkpoint's real persistent V
    while the derive still exercises every active cluster-contract boundary.
    """
    metadata = dict(model._loaded_meta_data)
    raw_mask = metadata["feature_mask_canonical"]
    mask = FeatureMask(
        FeatureMaskMode(raw_mask["mode"]),
        tuple(int(value) for value in raw_mask["ids"]),
    )
    representative_id = _CLEAN11_FEATURE_INDEX[0]
    masked_member_id = mask.ids[0]
    assert representative_id not in mask.ids
    definition = FeatureClusterDefinition.from_mapping(
        {
            "schema_version": 1,
            "clusters": [[representative_id, masked_member_id]],
        },
        n_sae_features=_N_SAE,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/real-compacted-cluster-overlay.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=_N_SAE,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    expected_vocab = effective_feature_vocab(consolidation, _N_SAE)
    assert expected_vocab == model.feature_vocab
    checkpoint_dir = _checkpoint_tag_dir()
    checkpoint = CheckpointPin(
        checkpoint_dir=checkpoint_dir,
        resolved_tag=_CHECKPOINT_TAG,
        step=_CHECKPOINT_STEP,
        model_file=StableFileDigest(
            path=checkpoint_dir / _MODEL_FILE,
            sha256="1" * 64,
            byte_length=1,
            stat_signature=(0o100644, 1, 2, 1, 4, 5),
        ),
        metadata_file=StableFileDigest(
            path=checkpoint_dir / f"meta_{_CHECKPOINT_STEP:06d}.json",
            sha256="2" * 64,
            byte_length=1,
            stat_signature=(0o100644, 1, 3, 1, 4, 5),
        ),
        n_sae_features=_N_SAE,
        feature_vocab=model.feature_vocab,
        load_bearing_metadata_json="{}",
    )
    user_config = metadata["user_config"]
    selection = CheckpointSelectionContract(
        positive_selection_json=json.dumps(
            user_config["feature_selection"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        negative_selection_json=json.dumps(
            user_config["negative_selection"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        num_buckets=2,
        effective_positive_buckets=1,
        preselection_seed=0,
        selection_hash="3" * 64,
        selection_manifest_sha256="4" * 64,
        selection_shards_sha256="5" * 64,
        packed_row_fingerprint="6" * 64,
    )
    statistics = FeatureStatisticsIdentity(
        file_sha256="7" * 64,
        percentile_table_checksum="8" * 64,
        n_sae_features=_N_SAE,
        feature_id_domain=_CLEAN11_FEATURE_INDEX,
    )
    feature_directory = FeatureDirectoryIdentity(
        segment_size=128,
        layer=8,
        activation_type="residual",
        pooling="max_pool",
        width="16k",
        model="gemma-2-2b",
        dataset_phase="base",
        context_independent_features=False,
        ignore_bos_features=True,
    )
    sae = SaeIdentity(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_8/width_16k/average_l0_71",
        hook_name="blocks.8.hook_resid_post",
        model="gemma-2-2b",
        layer=8,
        activation_type="residual",
        width="16k",
        actual_loaded_width=_N_SAE,
    )
    allowed_ids = tuple(model.feature_vocab.original_ids)
    representatives = tuple(
        int(value)
        for value in t.nonzero(
            consolidation.cluster_allowed_universe,
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    members_by_representative: dict[int, list[int]] = {
        representative: [] for representative in representatives
    }
    for feature_id in allowed_ids:
        representative = int(consolidation.rep_of[feature_id].item())
        members_by_representative[representative].append(feature_id)
    members = tuple(
        (representative, tuple(members_by_representative[representative]))
        for representative in representatives
    )
    source_projection = tuple(
        (
            source_id,
            int(consolidation.rep_of[source_id].item()),
        )
        for source_id in _CLEAN11_FEATURE_INDEX
    )
    projecting: dict[int, list[int]] = {}
    for source_id, representative in source_projection:
        projecting.setdefault(representative, []).append(source_id)
    representative_to_sources = tuple(
        (representative, tuple(projecting.get(representative, [])))
        for representative in representatives
    )
    semantic = {
        "schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "algorithms": {
            "concept_projection": CONCEPT_PROJECTION_VERSION,
            "qc_target_reduction": QC_TARGET_REDUCTION_VERSION,
            "packed_percentile_selection": PACKED_PERCENTILE_SELECTION_VERSION,
            "feature_diffmean_projection": FEATURE_DIFFMEAN_PROJECTION_VERSION,
            "generic_roofline_signal": GENERIC_ROOFLINE_SIGNAL_VERSION,
        },
        "n_sae_features": _N_SAE,
        "feature_mask": {
            "canonical": mask.canonical_dict(),
            "allowed_ids_sha256": _sha256_json(list(allowed_ids)),
            "allowed_count": len(allowed_ids),
        },
        "feature_clusters": {
            "descriptor": resolved.descriptor_dict(),
            "canonical": resolved.canonical_dict(),
            "representative_to_allowed_members": [
                [representative, list(member_ids)]
                for representative, member_ids in members
            ],
            "consolidated_away": [
                int(value)
                for value in t.nonzero(
                    consolidation.consolidated_away,
                    as_tuple=False,
                )
                .squeeze(1)
                .tolist()
            ],
            "representative_universe": list(representatives),
        },
        "concept_source": {
            "mode": "statistics_domain",
            "domain": list(_CLEAN11_FEATURE_INDEX),
            "source_to_representative": [
                list(pair) for pair in source_projection
            ],
        },
        "statistics": statistics.canonical_dict(),
        "feature_directory": feature_directory.canonical_dict(),
        "sae": sae.canonical_dict(),
        "checkpoint": checkpoint.semantic_dict(),
        "selection": selection.canonical_dict(),
    }
    provenance = {
        **checkpoint.provenance_dict(),
        **selection.provenance_dict(),
        "feature_clusters_path": str(resolved.source_path),
    }
    contract = ClusterFeatureContract(
        n_sae_features=_N_SAE,
        feature_mask=mask,
        feature_clusters=resolved,
        consolidation=consolidation,
        representative_universe=representatives,
        representative_to_allowed_members=members,
        source_to_representative=source_projection,
        representative_to_projecting_sources=representative_to_sources,
        statistics=statistics,
        feature_directory=feature_directory,
        sae=sae,
        checkpoint=checkpoint,
        selection=selection,
        concept_source_mode="statistics_domain",
        concept_source_domain=_CLEAN11_FEATURE_INDEX,
        semantic_json=_canonical_json(semantic),
        provenance_json=_canonical_json(provenance),
        feature_contract_fingerprint=_sha256_json(semantic),
    )
    metadata.update(feature_clusters_checkpoint_metadata(resolved))
    return contract, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 1) Load-side V-restriction facts on the REAL checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def test_real_compacted_checkpoint_load_side_facts(real_compacted_model: Any) -> None:
    """The production loader yields a genuinely COMPACTED encoder with the recorded ``V``.

    These are the deterministic, GPU-free facts Task 6a proved at load time and that the derive-raise
    tests then build on: the encoder IS compacted, ``|V|=15458 < 16384``, the fingerprint is the exact
    checkpoint sha256, and every clean11 concept feature id survives in ``V``.
    """
    model = real_compacted_model
    assert model.encoder.feature_vocab is not None, "the real checkpoint's encoder must be compacted"
    vocab = model.feature_vocab
    assert vocab is not None
    assert vocab.size == _EXPECTED_VOCAB_SIZE, f"|V| expected {_EXPECTED_VOCAB_SIZE}, got {vocab.size}"
    assert vocab.size < _N_SAE, "a compacted vocab must be strictly smaller than n_sae"
    assert vocab.n_sae_features == _N_SAE
    assert checkpoint_vocab_fingerprint(model) == _EXPECTED_VOCAB_FINGERPRINT

    allowed = feature_vocab_allowed_ids(model)
    assert allowed is not None, "a compacted model must expose its allowed original SAE ids V"
    assert allowed.dtype == t.int64
    assert tuple(allowed.shape) == (_EXPECTED_VOCAB_SIZE,)
    allowed_set = set(allowed.tolist())
    missing = [fid for fid in _CLEAN11_FEATURE_INDEX if fid not in allowed_set]
    assert not missing, f"clean11 concept feature_index values not in V: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# 2) Direct guard site — mean_g_steer RUNS on the real compacted model (in-V bag)
# ─────────────────────────────────────────────────────────────────────────────


def test_real_compacted_mean_g_steer_runs_with_in_vocab_bag(real_compacted_model: Any) -> None:
    """``mean_g_steer`` on the REAL compacted model RUNS with an in-V bag (#720).

    This is the exact ``gcg_bags.py`` frame Task 6a's traceback passed THROUGH on its way to the
    encoder forward + device-side CUDA assert. It now completes: the clean11 bag ids are all ∈ V, so
    the compacted encoder's ``_to_compact`` maps them to real rows and the forward-only measurement
    returns a finite g_steer — the direct proof the lifted guard was safe on real data.
    """
    model = real_compacted_model
    members = _members([_make_prefix()], model)
    # Non-vacuity: a NON-EMPTY real member set is required to pass mean_g_steer's own empty-check and
    # actually REACH the forward (an empty set raises AxBenchBpbAdvError first — asserted below).
    assert members, "fixture must produce >= 1 real class-margin member so the forward is truly reached"
    bag = FeatureConditioning.from_dict(
        {_CLEAN11_FEATURE_INDEX[0]: 0, _CLEAN11_FEATURE_INDEX[1]: 1}
    )

    g = mean_g_steer(model, members, bag, mode=SteeringMode.POSITIVE, device=_CPU)

    assert isinstance(g, float)
    assert g == g, "real compacted g_steer must be finite (the forward really ran)"

    # Non-vacuity backstop: with an EMPTY member set the DIFFERENT (earlier) empty-members guard fires,
    # proving the finite g_steer above came from real members reaching the forward, not a short-circuit.
    with pytest.raises(AxBenchBpbAdvError, match="no members"):
        mean_g_steer(model, [], bag, mode=SteeringMode.POSITIVE, device=_CPU)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Task 6a's exact crash entry point — the feature-DiffMean FAIR derive — now runs
# ─────────────────────────────────────────────────────────────────────────────


def test_real_compacted_fair_derive_runs_and_writes_bag_within_V(
    real_compacted_model: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clustered fair derive completes on the REAL compacted checkpoint.

    Reproduces Task 6a's §5 entry point (``feature_diffmean_bags.py``) against the REAL compacted
    checkpoint. The DiffMean warm-start select is stubbed (its real SAE forward needs the busy GPU and
    is not what's under test); the forward-only ``group_g_steer`` measurements condition the encoder on
    clean11 ids (all ∈ V), so ``_to_compact`` sees only in-V ids and the derive runs to the end,
    persisting a bag whose feature ids all lie in the real checkpoint's V.
    """
    monkeypatch.setattr(
        feature_diffmean_bags_module, "feature_diffmean_bag", _stub_feature_diffmean_bag
    )
    model = real_compacted_model
    cluster_contract, clustered_metadata = _real_compacted_cluster_overlay(model)
    monkeypatch.setattr(model, "_loaded_meta_data", clustered_metadata)
    prefixes = [_make_prefix(i) for i in range(5)]
    tokenizer = StubNanochatTokenizer()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="real_fair_compact")

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=_context_map(prefixes, tokenizer),
        cluster_contract=cluster_contract,
    )

    bag_path = paths.ours_gcg_bag(
        7, SteeringMode.POSITIVE, 0.0, PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    )
    assert bag_path.exists(), f"fair DiffMean derive must write a bag: {bag_path}"
    bag_dict, metadata = read_ours_gcg_bag(
        bag_path,
        expected_feature_contract_fingerprint=(
            cluster_contract.feature_contract_fingerprint
        ),
    )
    assert bag_dict, "fair DiffMean bag is unexpectedly empty"
    allowed = feature_vocab_allowed_ids(model)
    assert allowed is not None
    allowed_set = {int(x) for x in allowed.tolist()}
    assert set(bag_dict) <= allowed_set, (
        f"every fair DiffMean bag id must lie in the real checkpoint's V; got {sorted(bag_dict)}"
    )
    assert metadata["feature_contract_fingerprint"] == (
        cluster_contract.feature_contract_fingerprint
    )


def test_real_compacted_roofline_derive_runs_and_writes_bags_within_V(
    real_compacted_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The clustered roofline derive completes on the REAL compacted checkpoint."""
    monkeypatch.setattr(
        feature_diffmean_bags_module,
        "feature_diffmean_bag",
        _stub_feature_diffmean_bag,
    )
    model = real_compacted_model
    cluster_contract, clustered_metadata = _real_compacted_cluster_overlay(model)
    monkeypatch.setattr(model, "_loaded_meta_data", clustered_metadata)
    prefixes = [_make_prefix(i) for i in range(5)]
    tokenizer = StubNanochatTokenizer()
    paths = AxBenchBpbAdvPaths(
        output_root=tmp_path,
        run_tag="real_roofline_compact",
    )

    build_feature_diffmean_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=_context_map(prefixes, tokenizer),
        cluster_contract=cluster_contract,
    )

    scored_path = paths.ours_gcg_scored_prefixes(
        7,
        SteeringMode.POSITIVE,
        PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
    )
    scored_prefix_ids = read_ours_gcg_scored_prefixes(
        scored_path,
        expected_feature_contract_fingerprint=(
            cluster_contract.feature_contract_fingerprint
        ),
    )
    assert scored_prefix_ids
    allowed = feature_vocab_allowed_ids(model)
    assert allowed is not None
    allowed_set = {int(value) for value in allowed.tolist()}
    assert len(allowed_set) < _N_SAE
    for prefix_id in scored_prefix_ids:
        bag_path = paths.ours_gcg_bag(
            7,
            SteeringMode.POSITIVE,
            0.0,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            prefix_id=prefix_id,
        )
        bag_dict, metadata = read_ours_gcg_bag(
            bag_path,
            expected_feature_contract_fingerprint=(
                cluster_contract.feature_contract_fingerprint
            ),
        )
        assert bag_dict, f"roofline DiffMean bag is unexpectedly empty: {prefix_id}"
        assert set(bag_dict) <= allowed_set
        assert metadata["feature_contract_fingerprint"] == (
            cluster_contract.feature_contract_fingerprint
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4) T1's own consumer — the feature-GCG FAIR derive — now runs V-correct too
# ─────────────────────────────────────────────────────────────────────────────


def test_real_compacted_fair_gcg_derive_runs_and_writes_bag_within_V(
    real_compacted_model: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``build_ours_gcg_bags_fair`` — the feature-GCG derive whose optimizer T1 restricted to
    propose/score only ORIGINAL ids in ``V`` — now COMPLETES on the REAL compacted checkpoint
    and writes a bag ⊆ V (T4-Step-4, PR-2 #720).

    ``build_ours_gcg_bags_fair`` (``gcg_bags.py``) calls
    ``optimize_feature_conditioning_universal`` (``steering/conditional/feature_gcg/optimize.py``)
    — the EXACT code PR-2 changed to restrict its candidate-proposal universe to the checkpoint's
    ``V`` and translate accepted ids to compacted encoder rows via ``_id_weight_rows``. Until this
    test, that function was exercised ONLY against stub models
    (``test_gcg_bags_fair.py``/``test_gcg_bags_roofline.py``/``test_gcg_end_to_end.py``/
    ``test_gcg_bags_compacted_guard.py``); the real compacted checkpoint was exercised only for
    ``mean_g_steer`` and the feature-DiffMean derive (§2/§3 above) — never for the GCG optimizer
    itself. A real GPU e2e already proved this property (11/11 concept bags written, 93/93 feature
    ids ∈ V, every bag's ``vocab_fingerprint`` matching the checkpoint's, and
    ``optimize_g_steer <= warm_start_g_steer`` in 11/11); this is that property's CPU, skip-safe
    regression pin.

    The DiffMean warm-start is stubbed exactly as in §3 (its real SAE forward needs the busy GPU
    and is not what's under test) — monkeypatched on ``gcg_bags`` (not ``feature_diffmean_bags``):
    ``build_ours_gcg_bags_fair`` imports ``feature_diffmean_bag`` directly into ITS OWN module
    namespace, so patching the diffmean module (as §3 does for the DiffMean derive) would leave
    this derive calling the real, GPU-hungry implementation. The GCG step budget
    (``_make_gcg_cfg``) is pinned to the smallest values ``OursGcgConfig``/``FeatureGcgConfig``
    accept (``max_steps=2``, ``minibatch_size=1``, ``max_forward_batch=1``) so the real optimizer
    still runs its real gradient + candidate-proposal + accept-gate loop against the real
    compacted encoder, on CPU, without becoming a slow test.
    """
    monkeypatch.setattr(gcg_bags_module, "feature_diffmean_bag", _stub_feature_diffmean_bag)
    model = real_compacted_model
    prefixes = [_make_prefix(i) for i in range(5)]
    tokenizer = StubNanochatTokenizer()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="real_fair_gcg_compact")

    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_gcg_cfg(),
        paths=paths,
        device=_CPU,
        tokenizer=tokenizer,
        context_segments_by_prefix_hash=_context_map(prefixes, tokenizer),
    )

    bag_path = paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, GcgFamily.FAIR)
    assert bag_path.exists(), f"fair feature-GCG derive must write a bag: {bag_path}"
    bag_dict, metadata = read_ours_gcg_bag(bag_path)
    # Non-vacuity: an EMPTY bag would make the "every id in V" check below pass trivially
    # (the empty set is a subset of anything) while proving nothing about the real optimizer.
    assert bag_dict, (
        "fair feature-GCG bag is unexpectedly EMPTY -- an empty bag would make the ids-in-V "
        "check below pass vacuously, proving nothing about the real optimizer"
    )

    allowed = feature_vocab_allowed_ids(model)
    assert allowed is not None, "a compacted model must expose its allowed original SAE ids V"
    # Non-vacuity: V must be a STRICT subset of the SAE width, else "id in V" is trivially true
    # for every possible feature id and the check below would prove nothing either.
    assert len(allowed) < _N_SAE, (
        f"V must be strictly smaller than n_sae, else every id is trivially in V; "
        f"got |V|={len(allowed)} n_sae={_N_SAE}"
    )
    allowed_set = set(allowed.tolist())
    assert set(bag_dict) <= allowed_set, (
        f"every fair feature-GCG bag id must lie in the real checkpoint's V; got {sorted(bag_dict)}"
    )

    assert metadata[VOCAB_FINGERPRINT_META_KEY] == _EXPECTED_VOCAB_FINGERPRINT, (
        f"fair feature-GCG bag must be stamped with the derive-time checkpoint's V-fingerprint; "
        f"got {metadata.get(VOCAB_FINGERPRINT_META_KEY)!r}"
    )

    # Accept-gate: the GCG optimizer must never accept a bag whose optimize-split g_steer is
    # WORSE than the warm start it began from (values are `repr(float(...))`-encoded and can be
    # negative -- g_steer is a signed margin difference, not a magnitude).
    warm_start_g_steer = float(metadata["warm_start_g_steer"])
    optimize_g_steer = float(metadata["optimize_g_steer"])
    assert optimize_g_steer <= warm_start_g_steer, (
        "the GCG optimizer must never accept a bag worse than its warm start: "
        f"optimize_g_steer={optimize_g_steer} > warm_start_g_steer={warm_start_g_steer}"
    )
