"""Conditioned SFT emits negative (inactive) features, reusing pretraining's code.

A checkpoint pretrained with ``negative_selection`` saw, on every segment, a
sampled set of INACTIVE feature ids encoded into the reserved bucket
``effective_buckets = num_buckets - 1``.  Conditioned SFT used to emit none, so
its encoder input was structurally unlike anything pretraining produced.

SFT now samples them through the SAME production code pretraining uses:

* ``steering.conditional.sampling.sample_inactive_features_batched`` — the
  sampler the live pretrain dataloader and the packed-row precompute writer both
  call;
* ``steering.conditional.feature_pipeline.append_negative_features`` — the merge
  extracted out of the live pretrain dataloader so exactly one function owns the
  "negatives ride in bucket ``effective_buckets`` with percentile 0.0" encoding;
* ``steering.conditional.feature_pipeline.negative_sampling_mask`` — the pool
  resolver, now cluster-aware.

The tests below pin, in order: the extraction's byte-identity for the pretrain
path, the emission itself, the cluster interaction, FP-target purity, geometry
invariance, both generators, and an end-to-end probe driven from the real l22
checkpoint metadata.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch as t

from errors import FeatureClustersError
from nanochat.common import get_base_dir
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    append_negative_features,
    build_feature_pipeline,
    negative_sampling_mask,
)
from steering.conditional.sampling import (
    NegativeBundle,
    SamplingBundle,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.sft_schema import SFTSplit
from steering.stats_types import PerFeatureStat
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _full_sequence_char_count,
    patch_load_sft_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_SOURCE = REPO_ROOT / "scripts" / "train" / "conditional_encdec_sft.py"
SFT_SELECTION_SOURCE = (
    REPO_ROOT / "steering" / "conditional" / "encoder_decoder" / "sft_selection.py"
)
DATALOADER_SOURCE = (
    REPO_ROOT / "steering" / "conditional" / "encoder_decoder" / "dataloader.py"
)
MERGE_GOLDEN_PATH = (
    REPO_ROOT / "tests" / "steering" / "goldens" / "pretrain_negative_merge_golden.json"
)
L22_META_RELATIVE = Path(
    "conditional_encdec_checkpoints/l22_main_h100_8rank_db16_v1/s128/"
    "dist_geometric/mean_12.0/std_12.0/wl_9172c540310ba842/"
    "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039/"
    "meta_054600.json"
)
# The checkpoint inventory lives in the nanochat base dir, which is the shared
# checkout's ``nanochat_cache`` (``~/.cache/nanochat`` symlinks to it). A git
# worktree gets its own empty ``nanochat_cache``, so both roots are tried.
L22_META_CANDIDATES = (
    Path(get_base_dir()) / L22_META_RELATIVE,
    Path.home() / ".cache" / "nanochat" / L22_META_RELATIVE,
)

N_SAE = 16


# ---------------------------------------------------------------------------
# Source-bound harness: everything below drives the PRODUCTION expressions
# ---------------------------------------------------------------------------


def _sft_call_source(call_name: str, contains: str | None = None) -> str:
    tree = ast.parse(SFT_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != call_name:
            continue
        source = ast.unparse(node)
        if contains is None or contains in source:
            return source
    raise AssertionError(
        f"no call to {call_name} containing {contains!r} in {SFT_SOURCE}"
    )


def _sft_pipeline(**namespace: Any) -> FeaturePipelineConfig:
    """Evaluate SFT's literal ``build_feature_pipeline(...)`` call."""
    return eval(  # noqa: S307 - source is this repo's own training script
        _sft_call_source("build_feature_pipeline"),
        {"build_feature_pipeline": build_feature_pipeline},
        namespace,
    )


def _sft_negative_bundle(
    pipeline: FeaturePipelineConfig,
    consolidation: FeatureClusterConsolidation | None,
) -> NegativeBundle | None:
    """Evaluate SFT's literal negative-bundle construction."""
    if pipeline.negative_bundle is None:
        return None
    bundle = eval(  # noqa: S307 - source is this repo's own training script
        _sft_call_source("replace", "feature_pipeline.negative_bundle"),
        {"replace": replace, "negative_sampling_mask": negative_sampling_mask},
        {"feature_pipeline": pipeline, "consolidation": consolidation},
    )
    assert isinstance(bundle, NegativeBundle)
    return bundle


def _stats(fids: list[int]) -> dict[str, PerFeatureStat]:
    return {
        str(fid): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=10.0,
            min_mag=1.0,
            max_mag=99.0,
            median_mag=50.0,
            percentiles={f"p{p}": float(p) for p in range(1, 100)},
        )
        for fid in fids
    }


NEGATIVE_SELECTION_CFG = {
    "distribution": "geometric",
    "mean": 3.0,
    "std": 3.0,
    "min_k": 2,
    "max_k": 4,
}
SAMPLING_SELECTION_CFG = {
    "distribution": "geometric",
    "mean": 3.0,
    "std": 3.0,
    "min_k": 1,
    "max_k": 4,
    "mode": "top_k_percentile",
}


def _build_pipeline(
    *,
    n_sae_features: int = N_SAE,
    num_buckets: int = 2,
    with_negatives: bool = True,
    feature_mask: FeatureMask | None = None,
    active_fids: list[int] | None = None,
) -> FeaturePipelineConfig:
    active = list(range(n_sae_features)) if active_fids is None else active_fids
    return _sft_pipeline(
        feature_selection=parse_feature_selection(dict(SAMPLING_SELECTION_CFG)),
        pretrain_negative_selection=(
            parse_negative_selection(dict(NEGATIVE_SELECTION_CFG))
            if with_negatives
            else None
        ),
        per_feature_stats=_stats(active),
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        ddp_rank=0,
        _sft_feature_mask=feature_mask,
    )


def _generator_batch(
    *,
    pipeline: FeaturePipelineConfig,
    negative_bundle: NegativeBundle | None,
    active_fids: list[int],
    magnitudes: list[float],
    consolidation: FeatureClusterConsolidation | None = None,
    feature_mask: FeatureMask | None = None,
    n_sae_features: int | None = None,
    fp_percentile_table: t.Tensor | None = None,
    build_val_loader: bool = False,
) -> tuple:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=4, user_chars=3, asst_chars=7, prefix="c")
    total_chars = _full_sequence_char_count(tokenizer, task[0])
    raw_segment = SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor(active_fids, dtype=t.int32),
        feature_mags=t.tensor(magnitudes, dtype=t.float32),
        char_start=0,
        char_end=total_chars,
    )
    features = {index: [raw_segment] for index in range(len(task))}
    rendered, _ = tokenizer.render_conversation(task[0])
    sequence_len = len(rendered) - 1
    sampling_bundle = pipeline.selection_config
    assert isinstance(sampling_bundle, SamplingBundle)

    with patch_load_sft_features(
        features,
        split=SFTSplit.VAL if build_val_loader else None,
    ):
        if build_val_loader:
            generator = sft._build_sft_val_loader(
                val_task=task,
                tokenizer=tokenizer,
                B=2,
                T=sequence_len,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/dummy/features",
                expected_feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=None,
                boundaries=pipeline.boundaries,
                precomputed_thresholds=None,
                sampling_bundle=sampling_bundle,
                loss_mask_mode=sft.LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=1,
                n_sae_features=n_sae_features,
                fp_percentile_table=fp_percentile_table,
                negative_bundle=negative_bundle,
                feature_mask=feature_mask,
                consolidation=consolidation,
                alignment_probe_samples=0,
            )
        else:
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=sequence_len,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                boundaries=pipeline.boundaries,
                sampling_bundle=sampling_bundle,
                min_loss_tokens=1,
                n_sae_features=n_sae_features,
                fp_percentile_table=fp_percentile_table,
                negative_bundle=negative_bundle,
                feature_mask=feature_mask,
                consolidation=consolidation,
                alignment_probe_samples=0,
            )
        return next(generator)


# ---------------------------------------------------------------------------
# 1. The extraction is byte-identical for the PRETRAIN path
# ---------------------------------------------------------------------------


def _reference_merge(
    fids: t.Tensor,
    buckets: t.Tensor,
    pcts: t.Tensor,
    neg_fids: t.Tensor,
    effective_buckets: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """The live pretrain dataloader's merge, verbatim, as it stood BEFORE the
    extraction (``dataloader.py`` lines 1311-1322 of PR #1305's tree).

    Frozen here on purpose: it is the oracle the extracted helper must match.
    """
    if neg_fids is not None and len(neg_fids) > 0:
        neg_buckets = t.full((len(neg_fids),), effective_buckets, dtype=t.int64)
        fids = t.cat([fids.to(t.int64), neg_fids])
        buckets = t.cat([buckets, neg_buckets])
        pcts = t.cat([pcts, t.zeros(len(neg_fids), dtype=t.float32)])
    return fids, buckets, pcts


def _merge_cases() -> list[dict[str, Any]]:
    rng = t.Generator().manual_seed(20260808)
    cases: list[dict[str, Any]] = []
    for n_pos, n_neg, effective_buckets in [
        (0, 0, 1),
        (0, 3, 1),
        (5, 0, 1),
        (5, 4, 1),
        (7, 2, 3),
        (3, 9, 15),
    ]:
        cases.append(
            {
                "fids": t.randint(0, 64, (n_pos,), generator=rng, dtype=t.int32),
                "buckets": t.randint(
                    0, max(effective_buckets, 1), (n_pos,), generator=rng, dtype=t.int64
                ),
                "pcts": t.rand(n_pos, generator=rng, dtype=t.float32),
                "neg_fids": t.randint(
                    64, 128, (n_neg,), generator=rng, dtype=t.int64
                ),
                "effective_buckets": effective_buckets,
            }
        )
    return cases


class TestPretrainMergeIsByteIdentical:
    @pytest.mark.parametrize("case", _merge_cases())
    def test_extracted_helper_matches_the_frozen_inline_reference(
        self, case: dict[str, Any]
    ) -> None:
        expected = _reference_merge(
            case["fids"].clone(),
            case["buckets"].clone(),
            case["pcts"].clone(),
            case["neg_fids"].clone(),
            case["effective_buckets"],
        )
        actual = append_negative_features(
            case["fids"].clone(),
            case["buckets"].clone(),
            case["pcts"].clone(),
            case["neg_fids"].clone(),
            effective_buckets=case["effective_buckets"],
        )
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype
            assert t.equal(got, want)

    def test_output_matches_the_frozen_golden(self) -> None:
        """Golden pin: silently perturbing pretraining would be far worse than
        the bug this PR fixes, so the merge's bytes are frozen on disk."""
        golden = json.loads(MERGE_GOLDEN_PATH.read_text(encoding="utf-8"))
        cases = _merge_cases()
        assert len(cases) == len(golden["cases"])
        for case, entry in zip(cases, golden["cases"], strict=True):
            fids, buckets, pcts = append_negative_features(
                case["fids"],
                case["buckets"],
                case["pcts"],
                case["neg_fids"],
                effective_buckets=case["effective_buckets"],
            )
            assert fids.tolist() == entry["fids"]
            assert buckets.tolist() == entry["buckets"]
            assert pcts.tolist() == pytest.approx(entry["pcts"])

    def test_empty_negatives_returns_the_inputs_untouched(self) -> None:
        """The no-negatives control at the helper level: identical objects, so a
        checkpoint without ``negative_selection`` cannot be perturbed."""
        fids = t.tensor([1, 2, 3], dtype=t.int32)
        buckets = t.tensor([0, 0, 1], dtype=t.int64)
        pcts = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
        out = append_negative_features(
            fids,
            buckets,
            pcts,
            t.tensor([], dtype=t.int64),
            effective_buckets=1,
        )
        assert out[0] is fids
        assert out[1] is buckets
        assert out[2] is pcts

    def test_both_owners_call_the_shared_helper(self) -> None:
        """DRY, as requested: exactly one implementation, two callers."""
        dataloader = DATALOADER_SOURCE.read_text(encoding="utf-8")
        sft_source = SFT_SELECTION_SOURCE.read_text(encoding="utf-8")
        assert "append_negative_features(" in dataloader
        assert "append_negative_features(" in sft_source
        for source in (dataloader, sft_source):
            assert "neg_buckets = t.full(" not in source, (
                "the merge must not be re-implemented at a call site"
            )


# ---------------------------------------------------------------------------
# 2. SFT emits negatives — and does not when the lineage is unconditioned
# ---------------------------------------------------------------------------


ACTIVE_FIDS = [0, 1, 2, 3, 4, 5]
ACTIVE_MAGS = [10.0, 25.0, 40.0, 60.0, 80.0, 95.0]


class TestSftEmitsNegatives:
    def test_negatives_occupy_the_reserved_inactive_bucket(self) -> None:
        pipeline = _build_pipeline()
        bundle = _sft_negative_bundle(pipeline, None)
        assert bundle is not None
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp = batch
        buckets = enc_mags[enc_mask]
        ids = enc_fids[enc_mask]

        assert pipeline.effective_buckets == 1
        assert set(buckets.tolist()) == {0, 1}, (
            "binary mode: positives in bucket 0, negatives in the reserved "
            "inactive bucket 1"
        )
        n_neg = int((buckets == pipeline.effective_buckets).sum())
        assert n_neg >= 2, f"negative_selection min_k=2 must emit negatives, got {n_neg}"
        negatives = set(ids[buckets == pipeline.effective_buckets].tolist())
        assert negatives.isdisjoint(ACTIVE_FIDS), (
            "a sampled negative must be INACTIVE in the segment it is emitted for"
        )

    def test_no_negative_bundle_emits_no_negatives(self) -> None:
        """Mutation control: with emission disabled, the reserved bucket is
        never written, so the assertion above is not vacuous."""
        pipeline = _build_pipeline()
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        _inputs, _targets, enc_fids, enc_mags, enc_mask, _seg_map, _fp = batch
        buckets = enc_mags[enc_mask]
        assert set(buckets.tolist()) == {0}
        assert set(enc_fids[enc_mask].tolist()) <= set(ACTIVE_FIDS)

    def test_unconditioned_lineage_produces_no_bundle_at_all(self) -> None:
        """The no-negatives control at the source: a checkpoint without
        ``negative_selection`` yields ``negative_bundle=None``, so SFT's output
        is bit-identical to the pre-negatives behaviour."""
        pipeline = _build_pipeline(with_negatives=False)
        assert pipeline.negative_bundle is None
        assert _sft_negative_bundle(pipeline, None) is None

        with_none = _generator_batch(
            pipeline=pipeline,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pipeline_again = _build_pipeline(with_negatives=False)
        again = _generator_batch(
            pipeline=pipeline_again,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        for lhs, rhs in zip(with_none, again, strict=True):
            if lhs is None:
                assert rhs is None
            else:
                assert t.equal(lhs, rhs)

    def test_three_bucket_checkpoint_reserves_bucket_two(self) -> None:
        """Not binary-mode specific: the inactive row is always
        ``effective_buckets``."""
        pipeline = _build_pipeline(num_buckets=3)
        bundle = _sft_negative_bundle(pipeline, None)
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        _i, _t, enc_fids, enc_mags, enc_mask, _s, _fp = batch
        buckets = enc_mags[enc_mask]
        assert pipeline.effective_buckets == 2
        assert int((buckets == 2).sum()) >= 2
        positives = enc_fids[enc_mask][buckets < 2]
        assert set(positives.tolist()) <= set(ACTIVE_FIDS)
        assert max(buckets.tolist()) == 2

    def test_val_loader_emits_negatives_identically(self) -> None:
        """Both generators: ``_build_sft_val_loader`` must condition the same
        way, or validation measures a different distribution than training."""
        train_pipeline = _build_pipeline()
        train_batch = _generator_batch(
            pipeline=train_pipeline,
            negative_bundle=_sft_negative_bundle(train_pipeline, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        val_pipeline = _build_pipeline()
        val_batch = _generator_batch(
            pipeline=val_pipeline,
            negative_bundle=_sft_negative_bundle(val_pipeline, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
            build_val_loader=True,
        )
        for lhs, rhs in zip(train_batch, val_batch, strict=True):
            if lhs is None:
                assert rhs is None
            else:
                assert t.equal(lhs, rhs)

    def test_both_generator_call_sites_pass_the_bundle(self) -> None:
        source = SFT_SOURCE.read_text(encoding="utf-8")
        assert source.count("negative_bundle=sft_negative_bundle") == 2
        assert source.count("negative_bundle=negative_bundle") == 3

    def test_masked_ids_never_reach_the_encoder(self) -> None:
        mask = FeatureMask(FeatureMaskMode.blacklist, (7, 8, 9, 10, 11))
        pipeline = _build_pipeline(feature_mask=mask)
        bundle = _sft_negative_bundle(pipeline, None)
        assert bundle is not None
        assert bundle.feature_mask == mask
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
            feature_mask=mask,
        )
        _i, _t, enc_fids, enc_mags, enc_mask, _s, _fp = batch
        ids = enc_fids[enc_mask]
        assert set(ids.tolist()).isdisjoint({7, 8, 9, 10, 11})
        mask.assert_no_masked(ids.to(t.int64))
        assert int((enc_mags[enc_mask] == pipeline.effective_buckets).sum()) >= 2


# ---------------------------------------------------------------------------
# 3. Clusters: negatives must be representatives the encoder has a row for
# ---------------------------------------------------------------------------


def _consolidation(
    *,
    n_sae_features: int = N_SAE,
    groups: tuple[tuple[int, ...], ...] = ((1, 3), (5, 7, 9)),
    feature_mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=groups)
    resolved = ResolvedFeatureClusters(
        source_path=Path("/dummy/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, feature_mask)


class TestClusteredNegatives:
    def test_negative_pool_is_the_representative_only_universe(self) -> None:
        """``negative_sampling_mask(pipeline)`` alone is cluster-blind; with a
        consolidation it must resolve to ``effective_negative_mask()``, whose
        allowed universe is exactly ``cluster_allowed_universe``."""
        consolidation = _consolidation()
        pipeline = _build_pipeline()
        resolved = negative_sampling_mask(pipeline, consolidation)
        assert resolved is not None
        assert resolved == consolidation.effective_negative_mask()
        assert t.equal(
            resolved.allowed_universe_mask(N_SAE),
            consolidation.cluster_allowed_universe,
        )
        assert not bool(resolved.allowed_universe_mask(N_SAE)[3])
        assert not bool(resolved.allowed_universe_mask(N_SAE)[7])
        assert not bool(resolved.allowed_universe_mask(N_SAE)[9])

    def test_unclustered_resolution_is_the_plain_mask_byte_identically(self) -> None:
        mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
        pipeline = _build_pipeline(feature_mask=mask)
        assert negative_sampling_mask(pipeline) is pipeline.feature_mask
        assert negative_sampling_mask(pipeline, None) is pipeline.feature_mask

    def test_emitted_negatives_are_all_representatives(self) -> None:
        consolidation = _consolidation()
        pipeline = _build_pipeline()
        bundle = _sft_negative_bundle(pipeline, consolidation)
        assert bundle is not None
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=[0, 1, 2, 4],
            magnitudes=[10.0, 40.0, 60.0, 90.0],
            consolidation=consolidation,
        )
        _i, _t, enc_fids, enc_mags, enc_mask, _s, _fp = batch
        ids = enc_fids[enc_mask].to(t.int64)
        consolidation.assert_representatives_only(ids, surface="test")
        assert set(ids.tolist()).isdisjoint({3, 7, 9})
        assert int((enc_mags[enc_mask] == pipeline.effective_buckets).sum()) >= 2

    def test_cluster_blind_pool_is_rejected_at_the_public_boundary(self) -> None:
        """Non-vacuity for the fix: handing the generator a bundle built with
        the plain (cluster-blind) mask must fail loudly, because that is exactly
        the configuration that leaks a consolidated-away sibling into a
        compacted encoder with no row for it."""
        consolidation = _consolidation()
        pipeline = _build_pipeline()
        assert pipeline.negative_bundle is not None
        blind = replace(
            pipeline.negative_bundle,
            feature_mask=negative_sampling_mask(pipeline),
        )
        with pytest.raises(
            FeatureClustersError,
            match=r"(?s)must sample negatives from the representative-only pool",
        ):
            _generator_batch(
                pipeline=pipeline,
                negative_bundle=blind,
                active_fids=[0, 1, 2, 4],
                magnitudes=[10.0, 40.0, 60.0, 90.0],
                consolidation=consolidation,
            )

    def test_cluster_blind_pool_really_would_have_leaked(self) -> None:
        """Domain proof that the guard above is not theatre: the plain mask's
        allowed universe still contains consolidated-away ids, so an unguarded
        sampler could draw one."""
        consolidation = _consolidation()
        pipeline = _build_pipeline()
        blind = negative_sampling_mask(pipeline)
        assert blind is None, "no user mask configured here"
        # With no user mask the blind pool is the FULL SAE width, which
        # trivially includes the consolidated-away siblings.
        away = t.nonzero(consolidation.consolidated_away).squeeze(1).tolist()
        assert away == [3, 7, 9]

    def test_whitelist_and_clusters_compose(self) -> None:
        """The l22 shape in miniature: whitelist-masked AND consolidated."""
        mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 3, 4, 5, 7, 9, 12))
        consolidation = _consolidation(feature_mask=mask)
        pipeline = _build_pipeline(feature_mask=mask)
        bundle = _sft_negative_bundle(pipeline, consolidation)
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=[0, 1, 2, 4],
            magnitudes=[10.0, 40.0, 60.0, 90.0],
            consolidation=consolidation,
            feature_mask=mask,
        )
        _i, _t, enc_fids, enc_mags, enc_mask, _s, _fp = batch
        ids = enc_fids[enc_mask].to(t.int64)
        consolidation.assert_representatives_only(ids, surface="test")
        mask.assert_no_masked(ids)
        assert set(ids.tolist()) <= {0, 1, 2, 4, 5, 12}
        assert int((enc_mags[enc_mask] == pipeline.effective_buckets).sum()) >= 2


# ---------------------------------------------------------------------------
# 4. FP targets must carry the pre-selection ACTIVE set only
# ---------------------------------------------------------------------------


class TestFeaturePredictionTargetsUncontaminated:
    def _fp_batch(self, *, negatives: bool) -> tuple:
        pipeline = _build_pipeline()
        assert isinstance(pipeline.selection_config, SamplingBundle)
        table = pipeline.selection_config.percentile_table
        return _generator_batch(
            pipeline=pipeline,
            negative_bundle=(
                _sft_negative_bundle(pipeline, None) if negatives else None
            ),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
            n_sae_features=N_SAE,
            fp_percentile_table=table,
        )

    def test_fp_targets_are_identical_with_and_without_negatives(self) -> None:
        without = self._fp_batch(negatives=False)
        with_neg = self._fp_batch(negatives=True)
        fp_without = without[6]
        fp_with = with_neg[6]
        assert fp_without is not None
        assert t.equal(fp_without, fp_with), (
            "fp_targets predict what is ACTIVE; a sampled inactive feature must "
            "never enter all_feature_ids / all_feature_pcts"
        )

    def test_fp_targets_are_nonzero_only_on_the_active_columns(self) -> None:
        with_neg = self._fp_batch(negatives=True)
        fp = with_neg[6]
        nonzero_cols = set(t.nonzero(fp.sum(dim=(0, 1))).squeeze(1).tolist())
        assert nonzero_cols <= set(ACTIVE_FIDS)
        # Non-vacuity: negatives really were emitted in this same batch.
        _i, _t, _fids, enc_mags, enc_mask, _s, _fp, _sd = with_neg
        assert int((enc_mags[enc_mask] == 1).sum()) >= 2


# ---------------------------------------------------------------------------
# 5. Routing / geometry must be bit-identical with and without negatives
# ---------------------------------------------------------------------------


class TestGeometryUnchanged:
    def test_tokens_targets_and_seg_map_are_bit_identical(self) -> None:
        """Negatives have no decoder tokens: they ride in the encoder input
        only, so nothing about the sequence or its segment routing may move."""
        pipeline_a = _build_pipeline()
        without = _generator_batch(
            pipeline=pipeline_a,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pipeline_b = _build_pipeline()
        with_neg = _generator_batch(
            pipeline=pipeline_b,
            negative_bundle=_sft_negative_bundle(pipeline_b, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        inputs_a, targets_a, _fa, _ma, _ka, seg_map_a, _fpa = without
        inputs_b, targets_b, _fb, _mb, _kb, seg_map_b, _fpb = with_neg

        assert t.equal(inputs_a, inputs_b)
        assert t.equal(targets_a, targets_b)
        assert t.equal(seg_map_a, seg_map_b)
        assert seg_map_a.shape == seg_map_b.shape

    def test_only_the_encoder_feature_axis_grows(self) -> None:
        pipeline_a = _build_pipeline()
        without = _generator_batch(
            pipeline=pipeline_a,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pipeline_b = _build_pipeline()
        with_neg = _generator_batch(
            pipeline=pipeline_b,
            negative_bundle=_sft_negative_bundle(pipeline_b, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        assert without[2].shape[:2] == with_neg[2].shape[:2], "B and n_seg unchanged"
        # The (B, n_seg) grid is fixed; only the feature axis carries the extra
        # rows. Counted on the reserved bucket rather than on F_max, which the
        # shared-RNG positive shift also moves.
        without_buckets = without[3][without[4]]
        with_buckets = with_neg[3][with_neg[4]]
        assert int((without_buckets == pipeline_a.effective_buckets).sum()) == 0
        assert int((with_buckets == pipeline_b.effective_buckets).sum()) >= 2
        assert with_neg[2].shape[2] >= without[2].shape[2]

    def test_truncation_helper_is_untouched_by_negatives(self) -> None:
        """``_truncate_seg_features`` slices on char coordinates, which
        negatives do not carry; pin that it never inspects the feature axis."""
        segments = [
            sft.SegFeatures(
                feature_ids=t.tensor([1, 2, 40, 41], dtype=t.int64),
                buckets=t.tensor([0, 0, 1, 1], dtype=t.int64),
                percentiles=t.tensor([0.4, 0.6, 0.0, 0.0], dtype=t.float32),
                char_start=0,
                char_end=10,
            ),
            sft.SegFeatures(
                feature_ids=t.tensor([3, 42], dtype=t.int64),
                buckets=t.tensor([0, 1], dtype=t.int64),
                percentiles=t.tensor([0.5, 0.0], dtype=t.float32),
                char_start=10,
                char_end=20,
            ),
        ]
        stripped = [
            replace(
                seg,
                feature_ids=seg.feature_ids[seg.buckets == 0],
                buckets=seg.buckets[seg.buckets == 0],
                percentiles=seg.percentiles[seg.buckets == 0],
            )
            for seg in segments
        ]
        with_neg = sft._truncate_seg_features(segments, 8)
        without = sft._truncate_seg_features(stripped, 8)
        assert with_neg is not None and without is not None
        assert len(with_neg) == len(without)
        for lhs, rhs in zip(with_neg, without, strict=True):
            assert (lhs.char_start, lhs.char_end) == (rhs.char_start, rhs.char_end)
            assert lhs.prediction_target_only == rhs.prediction_target_only


# ---------------------------------------------------------------------------
# 6. RNG: shared stream, and an unperturbed no-negatives path
# ---------------------------------------------------------------------------


class TestRngContract:
    def test_bundle_shares_the_pipelines_generator_object(self) -> None:
        """``build_feature_pipeline``'s sampling path deliberately shares one
        RNG between SamplingBundle and NegativeBundle. Rebuilding the bundle for
        the cluster-aware mask must not clone or reseed it."""
        pipeline = _build_pipeline()
        bundle = _sft_negative_bundle(pipeline, None)
        assert bundle is not None
        assert isinstance(pipeline.selection_config, SamplingBundle)
        assert pipeline.negative_bundle is not None
        assert bundle.rng is pipeline.selection_config.rng
        assert bundle.rng is pipeline.negative_bundle.rng
        assert bundle.selection == pipeline.negative_bundle.selection

    def test_negative_draws_shift_the_positive_stream_as_pretraining_does(self) -> None:
        """Expected and matching pretraining: drawing negatives advances the
        SHARED generator, so the positives a negatives-trained checkpoint
        selects differ from the positives-only run. Pinned so the coupling is a
        documented property, not a surprise."""
        pipeline_a = _build_pipeline()
        without = _generator_batch(
            pipeline=pipeline_a,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pipeline_b = _build_pipeline()
        with_neg = _generator_batch(
            pipeline=pipeline_b,
            negative_bundle=_sft_negative_bundle(pipeline_b, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pos_without = sorted(without[2][without[4]].tolist())
        buckets = with_neg[3][with_neg[4]]
        pos_with = sorted(with_neg[2][with_neg[4]][buckets == 0].tolist())
        assert pos_without != pos_with

    def test_the_no_negatives_path_is_unperturbed(self) -> None:
        """The control that matters: for an unconditioned checkpoint the
        positive stream is exactly what it was, because no negative is ever
        drawn from the shared generator."""
        pipeline_a = _build_pipeline(with_negatives=False)
        first = _generator_batch(
            pipeline=pipeline_a,
            negative_bundle=None,
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        pipeline_b = _build_pipeline(with_negatives=False)
        second = _generator_batch(
            pipeline=pipeline_b,
            negative_bundle=_sft_negative_bundle(pipeline_b, None),
            active_fids=ACTIVE_FIDS,
            magnitudes=ACTIVE_MAGS,
        )
        for lhs, rhs in zip(first, second, strict=True):
            if lhs is None:
                assert rhs is None
            else:
                assert t.equal(lhs, rhs)


# ---------------------------------------------------------------------------
# 7. End-to-end probe driven from the real l22 checkpoint metadata
# ---------------------------------------------------------------------------


def _l22_meta() -> dict:
    for candidate in L22_META_CANDIDATES:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    pytest.skip(
        "l22 checkpoint metadata not present at any of "
        f"{[str(path) for path in L22_META_CANDIDATES]}"
    )


class TestL22RealMetadataProbe:
    """The checkpoint this work targets: whitelist-masked AND cluster-
    consolidated, 2074 encoder rows out of 65536, num_buckets=2."""

    @staticmethod
    def _setup() -> tuple[
        dict, FeatureMask, FeatureClusterConsolidation, FeaturePipelineConfig, list[int]
    ]:
        meta = _l22_meta()
        n_sae = 65536
        raw_mask = meta["feature_mask_canonical"]
        feature_mask = FeatureMask(
            FeatureMaskMode(raw_mask["mode"]),
            tuple(int(fid) for fid in raw_mask["ids"]),
        )
        definition = FeatureClusterDefinition(
            clusters=tuple(
                tuple(int(fid) for fid in cluster)
                for cluster in meta["feature_clusters_canonical"]["clusters"]
            )
        )
        consolidation = FeatureClusterConsolidation.build(
            ResolvedFeatureClusters(
                source_path=Path("/l22/clusters.yaml"),
                definition=definition,
                descriptor=FeatureClusterDescriptor.from_definition(
                    definition,
                    n_sae_features=n_sae,
                ),
            ),
            feature_mask,
        )
        conditioning = read_checkpoint_conditioning(meta)
        active = (
            t.nonzero(consolidation.cluster_allowed_universe).squeeze(1)[:6].tolist()
        )
        # ``feature_selection`` comes from the SFT config, not the checkpoint —
        # SFT never inherits the pretraining ranking, which is exactly the delta
        # gap the hardcoded policy warns about.
        pipeline = _sft_pipeline(
            feature_selection=parse_feature_selection(dict(SAMPLING_SELECTION_CFG)),
            pretrain_negative_selection=parse_negative_selection(
                dict(conditioning.negative_selection)
            ),
            per_feature_stats=_stats(active),
            n_sae_features=n_sae,
            num_buckets=int(meta["user_config"]["num_buckets"]),
            ddp_rank=0,
            _sft_feature_mask=feature_mask,
        )
        return meta, feature_mask, consolidation, pipeline, active

    def test_metadata_is_the_configuration_this_work_targets(self) -> None:
        meta, feature_mask, consolidation, pipeline, _active = self._setup()
        conditioning = read_checkpoint_conditioning(meta)
        assert conditioning.negative_selection == {
            "distribution": "geometric",
            "mean": 12.0,
            "min_k": 0,
            "max_k": 49,
        }
        assert conditioning.delta_weights == "percentile_delta"
        assert meta["user_config"]["num_buckets"] == 2
        assert meta["user_config"]["use_precomputed_batches"] is True
        assert feature_mask.mode is FeatureMaskMode.whitelist
        assert pipeline.effective_buckets == 1
        assert tuple(pipeline.boundaries.shape) == (65536, 0)
        assert int(consolidation.cluster_allowed_universe.sum()) == (
            meta["feature_vocab"]["vocab_size"]
        )
        assert meta["feature_vocab"]["vocab_size"] == 2074

    def test_positives_in_bucket_zero_negatives_in_the_inactive_bucket(self) -> None:
        _meta, feature_mask, consolidation, pipeline, active = self._setup()
        bundle = _sft_negative_bundle(pipeline, consolidation)
        assert bundle is not None
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=bundle,
            active_fids=active,
            magnitudes=[10.0, 25.0, 40.0, 60.0, 80.0, 95.0],
            consolidation=consolidation,
            feature_mask=feature_mask,
        )
        _i, _t, enc_fids, enc_mags, enc_mask, _s, _fp = batch
        ids = enc_fids[enc_mask].to(t.int64)
        buckets = enc_mags[enc_mask]

        assert set(buckets.tolist()) == {0, 1}
        positives = ids[buckets == 0]
        negatives = ids[buckets == 1]
        assert len(positives) > 0
        assert len(negatives) > 0
        assert int((buckets == pipeline.effective_buckets).sum()) == len(negatives)
        assert set(positives.tolist()) <= set(active), (
            "positives come from the segment's active set"
        )
        assert set(negatives.tolist()).isdisjoint(active)
        # Zero positives in the inactive bucket — the whole point of the fix.
        assert not bool((buckets[: len(positives)] == pipeline.effective_buckets).any())

    def test_every_emitted_id_is_a_representative_inside_the_whitelist(self) -> None:
        _meta, feature_mask, consolidation, pipeline, active = self._setup()
        batch = _generator_batch(
            pipeline=pipeline,
            negative_bundle=_sft_negative_bundle(pipeline, consolidation),
            active_fids=active,
            magnitudes=[10.0, 25.0, 40.0, 60.0, 80.0, 95.0],
            consolidation=consolidation,
            feature_mask=feature_mask,
        )
        ids = batch[2][batch[4]].to(t.int64)
        consolidation.assert_representatives_only(ids, surface="l22 probe")
        feature_mask.assert_no_masked(ids)
        assert bool(consolidation.cluster_allowed_universe[ids].all())
        assert bool(feature_mask.allowed_universe_mask(65536)[ids].all())

    def test_the_clustered_negative_pool_matches_what_precompute_used(self) -> None:
        """The clustered pretraining path (``use_precomputed_batches: true``)
        resolved its negative pool as ``feature_clusters.effective_negative_mask()``
        in ``scripts/preprocess/precompute_selected_features.py``.  SFT resolves
        to the same object, so it draws from the same universe l22's negatives
        actually came from."""
        _meta, _mask, consolidation, pipeline, _active = self._setup()
        bundle = _sft_negative_bundle(pipeline, consolidation)
        assert bundle is not None
        pool = bundle.feature_mask
        assert pool is not None
        assert pool == consolidation.effective_negative_mask()
        assert int(pool.allowed_universe_mask(65536).sum()) == 2074
