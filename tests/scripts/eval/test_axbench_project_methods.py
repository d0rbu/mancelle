"""Focused contracts for corrected AxBench SelfConditioning."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import pytest
import torch as t
import yaml
from safetensors.torch import save_file

from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS
from scripts.preprocess.precompute_token_boundaries import (
    compute_doc_token_boundaries_from_tokens,
)
from scripts.train.conditional_encdec_selfcond import build_selfcond_checkpoint_meta
from steering.conditional import missing_percentile
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.encoder_decoder import sft_selection
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
)
from steering.conditional.encoder_decoder.selfcond_training import SelfCondDataSource
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    ONLINE_SAE_LINEAGE_KEY,
    OnlineSAELineage,
    build_online_sae_lineage_metadata,
)
from steering.conditional.encoder_decoder.sft_selection import SegmentSelectionContext
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import (
    LOWEST_PERCENTILE_RANK,
    MissingPercentileError,
    MissingPercentilePolicy,
    MissingPercentileResolver,
    percentile_table_coverage,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.eval import turn_context_contract
from steering.eval.axbench_upstream import methods
from steering.eval.inference_engine import InferenceEngine
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_paths import (
    ActivationType,
    DatasetPhase,
    PoolingMode,
    SAEWidth,
    build_feature_dir,
)
from steering.refined_types import NonNegativeInt
from steering.segmentation import token_to_char_offsets
from steering.sft.turn_spans import ChatSpecialTokens
from steering.stats_io import content_sha256
from tests.encdec_test_helpers import build_tiny_model
from tests.encdec_test_helpers import method_setup_double as _fake_setup
from tests.nanochat.test_structured_conversation_renderer import _CharacterTokenizer
from tests.scripts.test_conditional_encdec_selfcond_startup import (
    _config as _selfcond_config,
)
from tests.steering.selfcond.helpers import tiny_head_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SWEEPS = (
    _REPO_ROOT / "vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/qwen2_5_3b_main.yaml",
    _REPO_ROOT / "vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/qwen2_5_3b_aux.yaml",
)


def _consolidation() -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((0, 1, 3),))
    descriptor = FeatureClusterDescriptor.from_definition(definition, n_sae_features=5)
    resolved = ResolvedFeatureClusters(
        source_path=Path("unused-clusters.yaml"),
        definition=definition,
        descriptor=descriptor,
    )
    return FeatureClusterConsolidation.build(
        resolved,
        FeatureMask("whitelist", (1, 3, 4)),
    )


def _selection() -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        delta_weights=DeltaSamplingWeights.percentile_delta,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )


def _negative_selection() -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=0,
        max_k=4,
    )


def _online_sae_lineage() -> OnlineSAELineage:
    return OnlineSAELineage(
        model="gemma-3-4b-pt",
        release="gemmascope_2",
        activation_type="residual",
        layer=22,
        width="65k",
        l0="medium",
        source_segment_size=17,
        feature_dir="/canonical/turn_delimited",
        statistics_path="/canonical/feature_statistics.safetensors",
        statistics_sha256="a" * 64,
        n_sae_features=65_536,
        num_buckets=2,
        feature_selection={
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "delta_weights": None,
            "min_k": 1,
            "max_k": 4,
            "mode": "top_k_percentile",
        },
        negative_selection={
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 0,
            "max_k": 4,
        },
        missing_percentile_policy="lowest",
        selection_seed=42,
        source_segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens.value,
    )


def test_method_setup_double_states_every_turn_field() -> None:
    # kills: dropping any key from TURN_FIELD_DEFAULTS (tests/encdec_test_helpers.py).
    # ``MethodSetup`` declares these five without defaults precisely so a
    # partially built setup cannot masquerade as a fixed-token checkpoint;
    # SimpleNamespace would silently omit a dropped one instead.
    turn_fields = {
        "segment_mode",
        "prompt_segment_size",
        "prompt_context_window",
        "turn_self_cond_segment_size_rustbpe",
        "prompt_selection_context",
    }

    assert turn_fields <= {field.name for field in fields(methods.MethodSetup)}
    assert turn_fields <= set(vars(_fake_setup()))


def _turn_prompt_pipeline(*, seed: int) -> FeaturePipelineConfig:
    percentile_table = t.linspace(0.1, 9.9, 99, dtype=t.float64).unsqueeze(0).expand(5, -1).clone()
    rng = t.Generator(device="cpu").manual_seed(seed)
    sampling_bundle = SamplingBundle(
        selection=replace(_selection(), delta_weights=None),
        percentile_table=percentile_table,
        rng=rng,
    )
    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=t.zeros((5, 0), dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=_negative_selection(),
            n_sae_features=5,
            rng=rng,
            feature_mask=None,
        ),
        effective_buckets=1,
        feature_mask=None,
    )


def _setup(
    *,
    n_sae_features: int = 4,
    allowed_ids: t.Tensor | None = None,
    feature_vocab: FeatureVocab | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
) -> Any:
    return _fake_setup(
        sae_layer=22,
        sae_model="gemma-3-4b-pt",
        n_sae_features=n_sae_features,
        percentile_table=t.zeros((n_sae_features, 99), dtype=t.float64),
        allowed_ids=allowed_ids,
        feature_vocab=feature_vocab,
        consolidation=consolidation,
        output_selection=_selection(),
        output_negative_selection=_negative_selection(),
        feature_mask=None,
    )


@pytest.mark.parametrize(
    ("total_concepts", "expected_release_indices"),
    [
        (0, []),
        (1, [0]),
        (4, [3]),
        (5, [4]),
        (6, [4, 5]),
        (49, [4, 9, 14, 19, 24, 29, 34, 39, 44, 48]),
        (50, [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]),
    ],
)
def test_masked_memory_release_fires_every_interval_and_always_on_last_concept(
    total_concepts: int,
    expected_release_indices: list[int],
) -> None:
    """fit_concept_dense_diffmeans's CUDA memory release must fire every
    _MASKED_MEMORY_RELEASE_INTERVAL_CONCEPTS concepts AND unconditionally on
    the last concept (so a run's final allocator state is always released,
    regardless of how total_concepts divides by the interval) -- without
    this, a run whose concept count isn't a multiple of the interval would
    leave the last few concepts' cycle-trapped tensors unreleased forever.
    """
    actual = [
        index
        for index in range(total_concepts)
        if methods._should_release_masked_memory(index, total_concepts)
    ]
    assert actual == expected_release_indices


def test_fit_concept_dense_diffmeans_applies_the_release_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = {
        concept_id: SimpleNamespace(
            positive=(f"positive-{concept_id}",),
            negative=(f"negative-{concept_id}",),
        )
        for concept_id in range(4)
    }
    fit_calls: list[tuple[int, tuple[str, ...], tuple[str, ...], int]] = []
    cadence_calls: list[tuple[int, int]] = []
    release_calls: list[str] = []

    def fit_dense_diffmean(
        _setup: Any,
        _base_model: Any,
        _sae: Any,
        positive: tuple[str, ...],
        negative: tuple[str, ...],
        *,
        concept_id: int,
        extraction_batch_size: int,
        **_kwargs: object,
    ) -> t.Tensor:
        fit_calls.append((concept_id, positive, negative, extraction_batch_size))
        return t.tensor([float(concept_id)])

    def should_release(concept_index: int, total_concepts: int) -> bool:
        cadence_calls.append((concept_index, total_concepts))
        return concept_index in {1, 3}

    monkeypatch.setattr(methods, "fit_dense_diffmean", fit_dense_diffmean)
    monkeypatch.setattr(methods, "_should_release_masked_memory", should_release)
    monkeypatch.setattr(methods.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(methods.t.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(methods.t.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(methods.gc, "collect", lambda: release_calls.append("gc"))
    monkeypatch.setattr(
        methods.t.cuda,
        "empty_cache",
        lambda: release_calls.append("empty_cache"),
    )

    result = methods.fit_concept_dense_diffmeans(
        object(),
        object(),
        object(),
        training,
        extraction_batch_size=7,
    )

    assert fit_calls == [
        (concept_id, (f"positive-{concept_id}",), (f"negative-{concept_id}",), 7)
        for concept_id in range(4)
    ]
    assert cadence_calls == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert release_calls == ["gc", "empty_cache", "gc", "empty_cache"]
    assert list(result) == list(training)
    for concept_id, value in result.items():
        assert t.equal(value, t.tensor([float(concept_id)]))


def test_fit_concept_dense_diffmeans_reuses_only_repeated_batch_one_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = {
        0: SimpleNamespace(positive=("a", "shared", "a"), negative=("b", "shared")),
        1: SimpleNamespace(positive=("shared", "c"), negative=("b", "d")),
    }
    activations = {
        text: t.tensor([value, value + 1, value + 2, value + 3], dtype=t.float32)
        for value, text in enumerate(("a", "shared", "b", "c", "d"), start=10)
    }
    calls: list[list[str]] = []

    def pool(texts: list[str], *_args: object, **_kwargs: object) -> t.Tensor:
        calls.append(texts)
        return t.stack([activations[text] for text in texts])

    monkeypatch.setattr(methods, "_batched_dense_sae_max_pool", pool)
    monkeypatch.setattr(methods.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        missing_percentile, "magnitude_to_percentile",
        lambda _ids, magnitudes, _table: magnitudes,
    )
    setup = _setup()
    baseline = methods.fit_concept_dense_diffmeans(
        setup, object(), object(), training, extraction_batch_size=1,
    )
    assert sum(map(len, calls)) == 9
    calls.clear()
    reused = methods.fit_concept_dense_diffmeans(
        setup, object(), object(), training,
        extraction_batch_size=1, reuse_duplicate_texts=True,
    )
    assert sum(map(len, calls)) == 5
    assert all(t.equal(reused[cid], baseline[cid]) for cid in baseline)
    with pytest.raises(ValueError, match="extraction_batch_size=1"):
        methods.fit_concept_dense_diffmeans(
            setup, object(), object(), training,
            extraction_batch_size=2, reuse_duplicate_texts=True,
        )


def test_resolve_raw_negative_selection_prefers_top_level() -> None:
    top_level = {"distribution": "geometric", "mean": 2.0, "min_k": 0, "max_k": 4}
    meta = {
        "pretrain_conditioning": {
            "negative_selection": {"distribution": "geometric", "mean": 9.0, "min_k": 0, "max_k": 9}
        }
    }
    user_config = {"negative_selection": top_level}
    assert methods.resolve_raw_negative_selection(meta, user_config) is top_level


def test_resolve_raw_negative_selection_falls_back_to_pretrain_conditioning() -> None:
    nested = {"distribution": "geometric", "mean": 12.0, "min_k": 0, "max_k": 49}
    meta = {"pretrain_conditioning": {"negative_selection": nested}}
    user_config = {"negative_selection": None}
    assert methods.resolve_raw_negative_selection(meta, user_config) is nested


def test_resolve_raw_negative_selection_absent_everywhere_is_none() -> None:
    assert methods.resolve_raw_negative_selection({}, {}) is None
    assert methods.resolve_raw_negative_selection({"pretrain_conditioning": {}}, {}) is None
    assert (
        methods.resolve_raw_negative_selection({"pretrain_conditioning": "not-a-dict"}, {}) is None
    )


def test_resolve_raw_feature_delta_prefers_top_level() -> None:
    top_level = {"mode": "splice_counterfactual"}
    meta = {"pretrain_conditioning": {"feature_delta": {"mode": "prev_segment"}}}
    user_config = {"feature_delta": top_level}
    assert methods.resolve_raw_feature_delta(meta, user_config) is top_level


def test_resolve_raw_feature_delta_falls_back_to_pretrain_conditioning() -> None:
    nested = {"mode": "splice_counterfactual"}
    meta = {"pretrain_conditioning": {"feature_delta": nested}}
    user_config = {"feature_delta": None}
    assert methods.resolve_raw_feature_delta(meta, user_config) is nested


def test_resolve_raw_feature_delta_absent_everywhere_is_none() -> None:
    assert methods.resolve_raw_feature_delta({}, {}) is None
    assert methods.resolve_raw_feature_delta({"pretrain_conditioning": {}}, {}) is None


def test_resolve_raw_feature_selection_keeps_own_delta_weights() -> None:
    user_config = {
        "feature_selection": {"distribution": "geometric", "delta_weights": "percentile_delta"}
    }
    meta = {"pretrain_conditioning": {"delta_weights": "should-not-be-used"}}

    result = methods.resolve_raw_feature_selection(meta, user_config)

    assert result["delta_weights"] == "percentile_delta"


def test_resolve_raw_feature_selection_patches_null_delta_weights_from_pretrain() -> None:
    user_config = {
        "feature_selection": {"distribution": "geometric", "delta_weights": None, "mean": 12.0}
    }
    meta = {"pretrain_conditioning": {"delta_weights": "percentile_delta"}}

    result = methods.resolve_raw_feature_selection(meta, user_config)

    assert result == {
        "distribution": "geometric",
        "delta_weights": "percentile_delta",
        "mean": 12.0,
    }


def test_resolve_raw_feature_selection_leaves_null_delta_weights_when_pretrain_missing() -> None:
    user_config = {"feature_selection": {"distribution": "geometric", "delta_weights": None}}

    result = methods.resolve_raw_feature_selection({}, user_config)

    assert result["delta_weights"] is None


def test_resolve_raw_feature_selection_passthrough_when_not_a_mapping() -> None:
    assert methods.resolve_raw_feature_selection({}, {"feature_selection": None}) is None
    assert methods.resolve_raw_feature_selection({}, {}) is None


def test_conditioning_block_returns_own_block_when_present() -> None:
    own = {"stats_source_sha256": "a" * 64}
    meta = {"user_config": {"model_tag": "m", "conditioning": own}}
    result = methods._conditioning_block(
        meta, target_meta_path=Path("target.json"), pretrain_meta_path=None
    )
    assert result is own


def test_conditioning_block_requires_pretrain_meta_path_when_absent() -> None:
    meta = {"user_config": {"model_tag": "m"}}
    with pytest.raises(ValueError, match="SFT-phase checkpoint"):
        methods._conditioning_block(
            meta, target_meta_path=Path("target.json"), pretrain_meta_path=None
        )


def test_conditioning_block_falls_back_to_pretrain_meta_path(tmp_path: Path) -> None:
    pretrain_conditioning = {"stats_source_sha256": "b" * 64}
    pretrain_path = tmp_path / "pretrain_meta.json"
    pretrain_path.write_text(
        json.dumps({"user_config": {"model_tag": "m", "conditioning": pretrain_conditioning}}),
        encoding="utf-8",
    )
    meta = {"user_config": {"model_tag": "m"}}

    result = methods._conditioning_block(
        meta, target_meta_path=Path("target.json"), pretrain_meta_path=pretrain_path
    )

    assert result == pretrain_conditioning


def test_conditioning_block_rejects_mismatched_pretrain_model_tag(tmp_path: Path) -> None:
    pretrain_path = tmp_path / "pretrain_meta.json"
    pretrain_path.write_text(
        json.dumps({"user_config": {"model_tag": "other-model", "conditioning": {"seed": 1}}}),
        encoding="utf-8",
    )
    meta = {"user_config": {"model_tag": "m"}}

    with pytest.raises(ValueError, match="model_tag does not match"):
        methods._conditioning_block(
            meta, target_meta_path=Path("target.json"), pretrain_meta_path=pretrain_path
        )


def test_resolve_raw_sae_config_returns_own_when_no_pretrain_path() -> None:
    own = {"release": "gemmascope_2", "model": "gemma-3-4b-pt", "width": "65k"}
    user_config = {"sae": own}

    assert methods.resolve_raw_sae_config(user_config, pretrain_meta_path=None) is own


def test_resolve_raw_sae_config_always_prefers_pretrain_when_supplied(tmp_path: Path) -> None:
    pretrain_sae = {"release": "gemmascope_2", "model": "gemma-3-4b-pt", "width": "65k"}
    pretrain_path = tmp_path / "pretrain_meta.json"
    pretrain_path.write_text(
        json.dumps({"user_config": {"model_tag": "m", "sae": pretrain_sae}}),
        encoding="utf-8",
    )
    stale_sae = {"release": "gemmascope", "model": "gemma-2-2b", "width": "16k"}
    user_config = {"model_tag": "m", "sae": stale_sae}

    result = methods.resolve_raw_sae_config(user_config, pretrain_meta_path=pretrain_path)

    assert result == pretrain_sae


def test_resolve_raw_sae_config_rejects_mismatched_pretrain_model_tag(tmp_path: Path) -> None:
    pretrain_path = tmp_path / "pretrain_meta.json"
    pretrain_path.write_text(
        json.dumps({"user_config": {"model_tag": "other-model", "sae": {}}}),
        encoding="utf-8",
    )
    user_config = {"model_tag": "m", "sae": {}}

    with pytest.raises(ValueError, match="model_tag does not match"):
        methods.resolve_raw_sae_config(user_config, pretrain_meta_path=pretrain_path)


def test_project_sampling_regime_and_closed_policy() -> None:
    temperatures = []
    for path in _VENDOR_SWEEPS:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        temperatures.append(loaded["inference"]["temperature"])

    assert temperatures == [1.0, 1.0]
    assert methods.TEMPERATURE == 1.0
    assert methods.TOP_K == 50
    assert methods.SELF_CONDITIONING_POLICY == (
        "prompt_last_segment_max_pool_logit_rank_diffmean_lerp_v2"
    )
    assert methods.SELF_CONDITIONING_MASKED_POLICY == (
        "prompt_last_segment_max_pool_logit_rank_diffmean_sign_rank_v3"
    )
    assert methods.SELF_CONDITIONING_FACTORS == (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )


@pytest.mark.parametrize("field", ["feature_mask", "feature_clusters"])
def test_prompt_feature_contract_rejects_nested_top_level_drift(field: str) -> None:
    user_config: dict[str, object] = {
        "feature_mask": {"mode": "whitelist", "path": "mask.yaml"},
        "feature_clusters": {"path": "clusters.yaml"},
    }
    conditioning = dict(user_config)
    conditioning[field] = None

    with pytest.raises(ValueError, match=rf"conditioning\.{field} disagrees"):
        methods._prompt_feature_contract_metadata(
            {},
            user_config,
            conditioning,
        )


def test_prompt_feature_contract_tolerates_staging_path_prefix_drift() -> None:
    user_config = {
        "feature_mask": {
            "mode": "whitelist",
            "path": "/orcd/scratch/orcd/007/henryac/sft-standard-20260819/support/"
            "l22_gemma3pt_clustered/feature_mask/manual_feature_filter_with_descriptions.yaml",
        },
        "feature_clusters": {
            "path": "/orcd/scratch/orcd/007/henryac/sft-standard-20260819/support/"
            "l22_gemma3pt_clustered/feature_clusters/feature_clusters_two_stage_schema_v1.yaml"
        },
    }
    conditioning = {
        "feature_mask": {
            "mode": "whitelist",
            "path": "out/autointerp_filter/gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_"
            "20260710_145543/manual_feature_filter_with_descriptions.yaml",
        },
        "feature_clusters": {"path": "out/artifacts/feature_clusters_two_stage_schema_v1.yaml"},
    }

    contract_meta = methods._prompt_feature_contract_metadata({}, user_config, conditioning)

    assert contract_meta["user_config"] is conditioning


def test_prompt_feature_contract_rejects_different_basenames() -> None:
    user_config = {
        "feature_mask": {"mode": "whitelist", "path": "/staged/other_mask.yaml"},
        "feature_clusters": {"path": "clusters.yaml"},
    }
    conditioning = {
        "feature_mask": {"mode": "whitelist", "path": "mask.yaml"},
        "feature_clusters": {"path": "clusters.yaml"},
    }

    with pytest.raises(ValueError, match=r"conditioning\.feature_mask disagrees"):
        methods._prompt_feature_contract_metadata({}, user_config, conditioning)


def test_prompt_feature_contract_uses_nested_user_config_with_canonical_payload() -> None:
    conditioning = {
        "feature_mask": {"mode": "whitelist", "path": "mask.yaml"},
        "feature_clusters": {"path": "clusters.yaml"},
    }
    meta = {
        "feature_mask_canonical": {"mode": "whitelist", "ids": [1, 2]},
        "feature_clusters_descriptor": {"descriptor": True},
        "feature_clusters_canonical": {"clusters": [[1, 2]]},
    }

    contract_meta = methods._prompt_feature_contract_metadata(
        meta,
        dict(conditioning),
        conditioning,
    )

    assert contract_meta["user_config"] is conditioning
    assert contract_meta["feature_mask_canonical"] == meta["feature_mask_canonical"]
    assert contract_meta["feature_clusters_descriptor"] == meta["feature_clusters_descriptor"]
    assert contract_meta["feature_clusters_canonical"] == meta["feature_clusters_canonical"]


def test_statistics_identity_is_checkpoint_bound(tmp_path: Path) -> None:
    statistics_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(statistics_path), metadata={"k": "v"})
    expected = content_sha256(statistics_path)
    conditioning = {"stats_source_sha256": expected}

    assert methods._validate_statistics_identity(conditioning, statistics_path) == expected

    mismatched = {"stats_source_sha256": "0" * 64}
    with pytest.raises(ValueError, match="do not match"):
        methods._validate_statistics_identity(mismatched, statistics_path)
    # An explicit, operator-asserted override logs instead of raising.
    assert (
        methods._validate_statistics_identity(
            mismatched, statistics_path, require_exact_statistics_identity=False
        )
        == expected
    )

    with pytest.raises(ValueError, match="64-character lowercase"):
        methods._validate_statistics_identity({}, statistics_path)


def test_statistics_identity_accepts_exact_legacy_bytes_without_relaxation(tmp_path: Path) -> None:
    statistics_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0])}, str(statistics_path), metadata={"k": "v"})
    content_identity = content_sha256(statistics_path)
    raw_identity = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    assert raw_identity != content_identity
    conditioning = {"stats_source_sha256": raw_identity}
    assert methods._validate_statistics_identity(conditioning, statistics_path) == content_identity

    # A real content mutation must still fail despite valid legacy provenance.
    save_file({"weights": t.tensor([1.0, 3.0])}, str(statistics_path), metadata={"k": "v"})
    with pytest.raises(ValueError, match="do not match"):
        methods._validate_statistics_identity(conditioning, statistics_path)


def _statistics_fixture(**metadata_overrides: Any) -> Any:
    metadata = {
        "pooling_mode": "max_pool",
        "segment_mode": methods.SegmentMode.fixed_tokens,
        "segment_size": methods.PROMPT_SEGMENT_SIZE,
        "context_window": methods.PROMPT_CONTEXT_WINDOW,
        "layer": 22,
        "n_sae_features": 65_536,
    }
    metadata.update(metadata_overrides)
    return SimpleNamespace(metadata=SimpleNamespace(**metadata), ignore_bos_features=True)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"pooling_mode": "mean_pool"}, "pooling_mode"),
        ({"segment_mode": "wrong_mode"}, "segment_mode"),
        ({"segment_size": 256}, "segment_size"),
        ({"context_window": 1024}, "context_window"),
        ({"layer": 21}, "layer"),
        ({"n_sae_features": 16_384}, "n_sae_features"),
    ],
)
def test_prompt_statistics_metadata_contract_rejects_drift(
    tmp_path: Path,
    overrides: dict[str, Any],
    error: str,
) -> None:
    path = tmp_path / "context_independent" / "no_bos" / "feature_statistics.safetensors"
    with pytest.raises(ValueError, match=error):
        methods._validate_statistics_contract(
            _statistics_fixture(**overrides),
            path=path,
            sae_model="gemma-3-4b-pt",
            sae_layer=22,
            n_sae_features=65_536,
        )


def test_is_under_context_independent_no_bos_accepts_spliced_nesting(tmp_path: Path) -> None:
    simple = tmp_path / "context_independent" / "no_bos" / "feature_statistics.safetensors"
    spliced = (
        tmp_path
        / "context_independent"
        / "no_bos"
        / "splice"
        / "cut_frac0.15"
        / "seed42"
        / "feature_statistics.safetensors"
    )
    unrelated = tmp_path / "feature_statistics.safetensors"
    wrong_order = tmp_path / "no_bos" / "context_independent" / "feature_statistics.safetensors"

    assert methods._is_under_context_independent_no_bos(simple)
    assert methods._is_under_context_independent_no_bos(spliced)
    assert not methods._is_under_context_independent_no_bos(unrelated)
    assert not methods._is_under_context_independent_no_bos(wrong_order)


def test_prompt_statistics_contract_accepts_spliced_nesting(tmp_path: Path) -> None:
    spliced_path = (
        tmp_path
        / "context_independent"
        / "no_bos"
        / "splice"
        / "cut_frac0.15"
        / "seed42"
        / "feature_statistics.safetensors"
    )
    methods._validate_statistics_contract(
        _statistics_fixture(),
        path=spliced_path,
        sae_model="gemma-3-4b-pt",
        sae_layer=22,
        n_sae_features=65_536,
    )


def test_turn_statistics_contract_uses_lineage_geometry_not_staging_path(
    tmp_path: Path,
) -> None:
    staged_path = tmp_path / "staged-statistics.safetensors"
    turn_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    statistics = _statistics_fixture(
        segment_mode=turn_mode,
        segment_size=17,
        context_window=333,
    )

    methods._validate_statistics_contract(
        statistics,
        path=staged_path,
        sae_model="gemma-3-4b-pt",
        sae_layer=22,
        n_sae_features=65_536,
        segment_mode=turn_mode,
        segment_size=17,
    )

    # Fixed mode still binds the canonical family path.
    with pytest.raises(ValueError, match="context_independent/no_bos"):
        methods._validate_statistics_contract(
            _statistics_fixture(),
            path=staged_path,
            sae_model="gemma-3-4b-pt",
            sae_layer=22,
            n_sae_features=65_536,
        )


@pytest.mark.parametrize("window", [1, 333, 8192])
def test_turn_statistics_contract_derives_the_window_instead_of_asserting_it(
    tmp_path: Path,
    window: int,
) -> None:
    # kills: asserting the window in turn mode (the caller reads it off the
    # same statistics, so the conjunct could only compare the field to itself).
    # The fixed-token half states that dropping the conjunct outright is NOT
    # what happened.
    turn_mode = methods.SegmentMode.turn_delimited_fixed_tokens

    methods._validate_statistics_contract(
        _statistics_fixture(segment_mode=turn_mode, segment_size=17, context_window=window),
        path=tmp_path / "staged-statistics.safetensors",
        sae_model="gemma-3-4b-pt",
        sae_layer=22,
        n_sae_features=65_536,
        segment_mode=turn_mode,
        segment_size=17,
        context_window=window + 1,
    )

    with pytest.raises(
        ValueError,
        match=rf"feature statistics context_window={window}, expected "
        rf"{methods.PROMPT_CONTEXT_WINDOW}",
    ):
        methods._validate_statistics_contract(
            _statistics_fixture(context_window=window),
            path=tmp_path / "context_independent" / "no_bos" / "feature_statistics.safetensors",
            sae_model="gemma-3-4b-pt",
            sae_layer=22,
            n_sae_features=65_536,
        )


class _TurnSetupFixture(NamedTuple):
    """Paths plus recorded collaborator arguments for one turn-mode load."""

    meta_path: Path
    statistics_path: Path
    statistics: Any
    lineage: OnlineSAELineage
    prompt_pipeline: Any
    selection_context: Any
    observed: dict[str, Any]


def _post_trained_meta(sft_meta: dict[str, Any]) -> dict[str, Any]:
    """Reshape one turn SFT meta and mint the REAL self-conditioned successor.

    The SFT trainer records ``negative_selection`` / ``feature_delta`` under
    ``pretrain_conditioning`` instead of in its own ``user_config``, and nulls
    ``feature_selection.delta_weights`` there, so a self-conditioning
    checkpoint that fails to re-publish those blocks leaves AxBench with no
    delta policy and no step at all (GPU jobs 22039415 / 22039772).
    """

    base = deepcopy(sft_meta)
    user_config = base["user_config"]
    base["pretrain_conditioning"] = {
        "negative_selection": user_config.pop("negative_selection"),
        "feature_delta": user_config.pop("feature_delta"),
        "delta_weights": user_config["feature_selection"]["delta_weights"],
    }
    user_config["feature_selection"] = {
        **user_config["feature_selection"],
        "delta_weights": None,
    }
    base["step"] = 20
    base["dataset"] = "safety"
    base["sft_loss_mask"] = "full_sequence"
    base["missing_percentile_policy"] = "lowest"
    base["encoder_config"] = {"depth": 1}
    base["feature_prediction_config"] = {
        "block": "mlp",
        "head_layers": 0,
        "weight": 0.1,
        "layer": -1,
        "loss_mode": "max_pool",
        "loss_fn": "bce",
    }
    del base["selfcond"]
    del base["selfcond_data"]
    return build_selfcond_checkpoint_meta(
        model=tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, num_buckets=2, seed=801),
        base_meta=base,
        identity={},
        cfg=_selfcond_config(),
        dataloader_state_dict=None,
        loop_state={"step": 22},
        base_checkpoint_dir="/checkpoints/turn-sft-leaf",
        data_source=SelfCondDataSource.SFT,
        selfcond_data={"sft_segment_mode": methods.SegmentMode.turn_delimited_fixed_tokens.value},
        turn_self_cond_segment_size_rustbpe=37,
    )


def _real_lineage_block(tmp_path: Path) -> dict[str, Any]:
    """The 21-key lineage block the conditioned-SFT trainer actually publishes.

    Built by the production writer over a canonical turn feature directory, so
    the checkpoint under test carries a block that
    ``OnlineSAELineage.from_checkpoint_meta`` parses for real instead of the
    two-key stand-in every stubbed turn test hands it.
    """

    feature_dir = build_feature_dir(
        root=tmp_path / "features",
        segment_size=17,
        layer=22,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_65K,
        model="gemma-3-4b-pt",
        dataset_phase=DatasetPhase.STANDARD_SFT,
        context_independent_features=True,
        ignore_bos_features=True,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
    )
    feature_dir.mkdir(parents=True)
    statistics_path = feature_dir / "feature_statistics.safetensors"
    statistics_path.write_bytes(b"pinned-turn-statistics")
    return build_online_sae_lineage_metadata(
        feature_dir=str(feature_dir),
        statistics_path=statistics_path,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        sae_config={
            "model": "gemma-3-4b-pt",
            "release": "gemmascope_2",
            "activation_type": "residual",
            "layer": 22,
            "width": "65k",
            "l0": "medium",
        },
        feature_selection={
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "delta_weights": None,
            "min_k": 1,
            "max_k": 4,
            "mode": "top_k_percentile",
        },
        negative_selection={
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 0,
            "max_k": 4,
        },
        missing_percentile_policy="lowest",
        n_sae_features=65_536,
        num_buckets=2,
        selection_seed=42,
    )[ONLINE_SAE_LINEAGE_KEY]


def _install_turn_setup_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    lineage: OnlineSAELineage | None = None,
    statistics: Any = None,
    real_selection_context: bool = False,
    fixed_mode: bool = False,
    prompt_pipeline: Any = None,
    post_trained: bool = False,
    real_lineage_block: dict[str, Any] | None = None,
) -> _TurnSetupFixture:
    """Stage one turn checkpoint whose expensive collaborators are stubbed.

    Only the checkpoint-geometry collaborators are replaced (SAE resolution,
    feature contract, statistics snapshot, percentile table, pipeline build).
    ``load_method_setup``'s own mode/cadence/lineage resolution and every
    contract it enforces stay real.

    ``real_lineage_block`` (see :func:`_real_lineage_block`) writes the trainer's
    own 21-key lineage into the checkpoint and leaves
    ``OnlineSAELineage.from_checkpoint_meta`` UNPATCHED, so the production parse
    runs; the default two-key block keeps that parse stubbed.
    """

    turn_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    feature_selection = {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "delta_weights": "percentile_delta",
        "min_k": 1,
        "max_k": 4,
        "mode": "top_k_percentile",
    }
    negative_selection = {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "min_k": 0,
        "max_k": 4,
    }
    conditioning = {
        "feature_selection": feature_selection,
        "negative_selection": negative_selection,
        "n_sae_features": 65_536,
        "num_buckets": 2,
        "seed": 99,
        "stats_source_sha256": "0" * 64,
    }
    meta = {
        "sft_segment_mode": turn_mode.value,
        "turn_self_cond_segment_size_rustbpe": 37,
        "structured_sft_online_sae_lineage_v1": (
            {"schema_version": 1, "source_segment_mode": turn_mode.value}
            if real_lineage_block is None
            else real_lineage_block
        ),
        "selfcond": {"data_source": "sft"},
        "selfcond_data": {"sft_segment_mode": turn_mode.value},
        "feature_vocab": None,
        "user_config": {
            "model_tag": "d12",
            "sae": {
                "release": "gemmascope_2",
                "model": "gemma-3-4b-pt",
                "activation_type": "residual",
                "layer": 22,
                "width": "65k",
                "l0": "medium",
            },
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
            "feature_delta": {"mode": "splice_counterfactual"},
            "num_buckets": 2,
            "conditioning": conditioning,
        },
    }
    if fixed_mode:
        for key in (
            "sft_segment_mode",
            "turn_self_cond_segment_size_rustbpe",
            "structured_sft_online_sae_lineage_v1",
            "selfcond",
            "selfcond_data",
        ):
            del meta[key]
    if post_trained:
        meta = _post_trained_meta(meta)
    meta_path = tmp_path / ("meta_000022.json" if post_trained else "meta_000007.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    statistics_path = (
        tmp_path / "context_independent" / "no_bos" / "feature_statistics.safetensors"
        if fixed_mode
        else tmp_path / "staged-statistics.safetensors"
    )
    if statistics is None:
        statistics = (
            _statistics_fixture()
            if fixed_mode
            else _statistics_fixture(
                segment_mode=turn_mode,
                segment_size=17,
                context_window=333,
            )
        )
        statistics.per_feature_stats = {}
    resolved_lineage = _online_sae_lineage() if lineage is None else lineage
    contract = CheckpointFeatureContract(None, None)
    percentile_table = t.zeros((1, 1), dtype=t.float64)
    prompt_rng = t.Generator(device="cpu").manual_seed(42)
    if prompt_pipeline is None:
        prompt_pipeline = SimpleNamespace(
            selection_config=methods.SamplingBundle(
                selection=_selection(),
                percentile_table=percentile_table,
                rng=prompt_rng,
            ),
            negative_bundle=object(),
            effective_buckets=1,
        )
    selection_context = object()
    observed: dict[str, Any] = {}

    if real_lineage_block is None:
        monkeypatch.setattr(
            methods.OnlineSAELineage,
            "from_checkpoint_meta",
            classmethod(lambda _cls, _meta, **_kwargs: resolved_lineage),
        )
    monkeypatch.setattr(
        methods,
        "resolve_sae",
        lambda **_kwargs: SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
    )
    monkeypatch.setattr(
        methods,
        "resolve_checkpoint_feature_contract",
        lambda *_args, **_kwargs: contract,
    )

    def fake_load_turn_statistics(
        path: Path,
        *,
        expected_raw_sha256: str,
    ) -> Any:
        observed["statistics_snapshot"] = (path, expected_raw_sha256)
        return statistics

    def fake_build_pipeline(cfg_block: Any, **kwargs: Any) -> Any:
        observed["pipeline"] = (cfg_block, kwargs)
        return prompt_pipeline

    def fake_selection_context(**kwargs: Any) -> Any:
        observed["selection_context"] = kwargs
        return selection_context

    monkeypatch.setattr(methods, "_load_turn_statistics_snapshot", fake_load_turn_statistics)
    monkeypatch.setattr(
        methods,
        "_validate_statistics_identity",
        lambda *_args, **_kwargs: observed.setdefault("legacy_identity", True),
    )
    monkeypatch.setattr(methods, "load_statistics", lambda path: statistics)
    monkeypatch.setattr(methods, "build_percentile_table", lambda *_args: percentile_table)
    monkeypatch.setattr(methods, "build_feature_pipeline_from_cfg_block", fake_build_pipeline)
    if not real_selection_context:
        monkeypatch.setattr(methods, "_build_turn_prompt_selection_context", fake_selection_context)
    return _TurnSetupFixture(
        meta_path=meta_path,
        statistics_path=statistics_path,
        statistics=statistics,
        lineage=resolved_lineage,
        prompt_pipeline=prompt_pipeline,
        selection_context=selection_context,
        observed=observed,
    )


def test_load_turn_setup_binds_lineage_statistics_selection_and_cadence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    turn_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path)
    observed = fixture.observed
    lineage_selection = {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "delta_weights": None,
        "min_k": 1,
        "max_k": 4,
        "mode": "top_k_percentile",
    }
    negative_selection = {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "min_k": 0,
        "max_k": 4,
    }

    setup = methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert setup.segment_mode is turn_mode
    assert setup.prompt_segment_size == 17
    assert setup.prompt_context_window == 333
    assert setup.turn_self_cond_segment_size_rustbpe == 37
    assert setup.prompt_selection_context is fixture.selection_context
    assert observed["statistics_snapshot"] == (
        fixture.statistics_path,
        "a" * 64,
    )
    pipeline_source, pipeline_kwargs = observed["pipeline"]
    assert pipeline_source == {
        "feature_selection": lineage_selection,
        "negative_selection": negative_selection,
    }
    assert pipeline_kwargs["seed"] == 42
    assert pipeline_kwargs["per_feature_stats"] == {}
    assert observed["selection_context"] == {
        "pipeline": fixture.prompt_pipeline,
        "missing_percentile_policy": "lowest",
        "consolidation": None,
    }


# kills: dropping the base's top-level identity blocks from
# build_selfcond_checkpoint_meta. Without `pretrain_conditioning` the SFT
# base's nulled delta_weights stays null and _validate_target_policy rejects the
# checkpoint; without the top-level `step` written from loop_state AxBench
# cannot locate the weights at all. Post-training's OWN builder produces the
# metadata here, so this is the surface the GPU acceptance run failed on.
def test_load_setup_reads_a_real_post_trained_turn_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path, post_trained=True)

    setup = methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert setup.prompt_statistics_raw_sha256 == "a" * 64
    assert setup.meta["step"] == 22
    assert setup.meta["base_checkpoint_dir"] == "/checkpoints/turn-sft-leaf"
    assert setup.segment_mode is methods.SegmentMode.turn_delimited_fixed_tokens
    assert setup.turn_self_cond_segment_size_rustbpe == 37
    assert setup.prompt_segment_size == 17
    # Resolved only through the re-published pretrain_conditioning block: the
    # SFT base nulls delta_weights in its own user_config.
    assert setup.meta["user_config"]["feature_selection"]["delta_weights"] is None
    assert setup.output_selection.delta_weights is DeltaSamplingWeights.percentile_delta
    assert setup.output_negative_selection is not None
    assert setup.meta["missing_percentile_policy"] == "lowest"


def test_load_turn_setup_parses_the_trainers_own_lineage_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: every turn assertion in this suite depending on the two-key lineage
    # stand-in. Here the checkpoint carries the block
    # ``build_online_sae_lineage_metadata`` actually writes and
    # ``OnlineSAELineage.from_checkpoint_meta`` is NOT patched, so the real
    # 21-field parse -- and the geometry it hands the setup -- runs end to end.
    block = _real_lineage_block(tmp_path)
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path, real_lineage_block=block)

    setup = methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert setup.segment_mode is methods.SegmentMode.turn_delimited_fixed_tokens
    assert setup.prompt_segment_size == 17
    assert setup.turn_self_cond_segment_size_rustbpe == 37
    # The digest the snapshot loader is asked for is the lineage's own, hashed
    # by the production writer over the real statistics file it points at.
    assert fixture.observed["statistics_snapshot"] == (
        fixture.statistics_path,
        hashlib.sha256(b"pinned-turn-statistics").hexdigest(),
    )
    assert fixture.observed["pipeline"][1]["seed"] == 42


def test_load_turn_setup_rejects_a_lineage_block_missing_a_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: stubbing OnlineSAELineage.from_checkpoint_meta on this path, which
    # would accept a checkpoint whose published lineage is incomplete.
    block = _real_lineage_block(tmp_path)
    del block["selection_seed"]
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path, real_lineage_block=block)

    with pytest.raises(
        SFTGRPOError,
        match=r"online-SAE lineage fields are incomplete or unknown: expected=\[",
    ) as excinfo:
        methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert "'selection_seed'" not in str(excinfo.value).split("got=", 1)[1]


def test_load_turn_setup_refuses_the_relaxed_statistics_identity_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: silently ignoring require_exact_statistics_identity in turn mode.
    # The lineage pins a RAW file sha256 that no re-save reproduces, so the
    # operator escape hatch cannot mean anything here and must say so.
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        methods,
        "_load_turn_statistics_snapshot",
        lambda *_args, **_kwargs: pytest.fail("the relaxed flag reached the snapshot loader"),
    )

    with pytest.raises(
        ValueError,
        match=r"turn-delimited checkpoints have no relaxed statistics identity mode",
    ):
        methods.load_method_setup(
            fixture.meta_path,
            fixture.statistics_path,
            require_exact_statistics_identity=False,
        )


_TURN_LINEAGE_CONTRACT_MUTATIONS: dict[str, Any] = {
    "model": "gemma-2-2b",
    "release": "gemmascope_1",
    "activation_type": "mlp",
    "layer": 12,
    "width": "16k",
    "l0": "big",
    "n_sae_features": 16_384,
    "num_buckets": 3,
}


@pytest.mark.parametrize(
    ("field", "value"),
    sorted(_TURN_LINEAGE_CONTRACT_MUTATIONS.items()),
)
def test_load_turn_setup_rejects_every_conflicting_lineage_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    # kills: dropping the _validate_turn_lineage_contract call from
    # load_method_setup, or weakening any single conjunct of its `expected`
    # dict, which would let AxBench condition prompts on a feature source the
    # conditioned-SFT run never used.
    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        lineage=replace(_online_sae_lineage(), **{field: value}),
    )

    with pytest.raises(
        ValueError,
        match=r"lineage conflicts with the target checkpoint.*pass pretrain_meta_path",
    ):
        methods.load_method_setup(fixture.meta_path, fixture.statistics_path)


def test_turn_lineage_contract_mutation_matrix_covers_every_compared_field() -> None:
    # kills: adding a conjunct to _validate_turn_lineage_contract's `expected`
    # dict without a rejection parametrization for it.
    source = inspect.getsource(methods._validate_turn_lineage_contract)
    expected_block = source.split("expected = {", 1)[1].split("}", 1)[0]
    compared = set(re.findall(r'"([a-z0-9_]+)":', expected_block))

    assert compared == set(_TURN_LINEAGE_CONTRACT_MUTATIONS)


def test_turn_setup_accepts_the_unmutated_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Non-vacuity partner for the rejection matrix: the same fixture with an
    # unmutated lineage must load, so the parametrized raises above are caused
    # by the mutation and not by the fixture itself.
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path)

    setup = methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert setup.prompt_segment_size == fixture.lineage.source_segment_size


def test_load_fixed_setup_pins_the_legacy_prompt_geometry_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fixed branch keeps the constants and the legacy identity check."""

    # kills: dropping the legacy _validate_statistics_identity preflight that
    # binds the statistics bytes to the checkpoint, or routing the fixed arm
    # through the turn statistics snapshot. (The geometry constants themselves
    # are pinned by the drift matrix below -- this fixture's statistics record
    # exactly the constants, so it cannot see where they came from.)
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path, fixed_mode=True)

    setup = methods.load_method_setup(fixture.meta_path, fixture.statistics_path)

    assert setup.segment_mode is methods.SegmentMode.fixed_tokens
    assert setup.prompt_segment_size == methods.PROMPT_SEGMENT_SIZE
    assert setup.prompt_context_window == methods.PROMPT_CONTEXT_WINDOW
    assert setup.turn_self_cond_segment_size_rustbpe is None
    assert setup.prompt_selection_context is None
    assert setup.statistics is fixture.statistics
    assert fixture.observed["legacy_identity"] is True
    assert "statistics_snapshot" not in fixture.observed


@pytest.mark.parametrize(
    ("override", "field"),
    [({"segment_size": 256}, "segment_size"), ({"context_window": 1024}, "context_window")],
)
def test_load_fixed_setup_rejects_statistics_geometry_off_the_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: dict[str, Any],
    field: str,
) -> None:
    """The fixed arm's geometry is the constants, never the statistics file."""

    # kills: sourcing the fixed-mode prompt geometry from statistics.metadata
    # (as turn mode does) instead of PROMPT_SEGMENT_SIZE/PROMPT_CONTEXT_WINDOW.
    # Under that mutation _validate_statistics_contract compares the statistics
    # against itself and accepts; only a statistics file whose geometry differs
    # from the constants can tell the two sources apart.
    statistics = _statistics_fixture(**override)
    statistics.per_feature_stats = {}
    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        fixed_mode=True,
        statistics=statistics,
    )

    with pytest.raises(ValueError, match=field):
        methods.load_method_setup(fixture.meta_path, fixture.statistics_path)


@pytest.mark.parametrize("context_window", [0, -1, True, 2048.0, "2048", None])
def test_load_turn_setup_rejects_a_non_positive_int_statistics_context_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context_window: Any,
) -> None:
    # kills: dropping the positive-int check on the turn statistics'
    # context_window, which is the only constraint on that value (the contract
    # conjunct that consumes it compares it against itself).
    turn_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    statistics = _statistics_fixture(
        segment_mode=turn_mode,
        segment_size=17,
        context_window=context_window,
    )
    statistics.per_feature_stats = {}
    fixture = _install_turn_setup_fixture(monkeypatch, tmp_path, statistics=statistics)

    with pytest.raises(
        ValueError,
        match="turn-delimited feature statistics context_window must be a positive integer",
    ):
        methods.load_method_setup(fixture.meta_path, fixture.statistics_path)


def test_turn_prompt_selection_context_rejects_a_non_sampling_pipeline() -> None:
    # kills: dropping the SamplingBundle guard in
    # _build_turn_prompt_selection_context, which would hand a threshold-mode
    # pipeline's selection_config straight to the live conditioned-SFT selector
    # (load_method_setup's identically worded check fires first on its own
    # path, so only a direct call reaches this one).
    pipeline = replace(_turn_prompt_pipeline(seed=7), selection_config=object())

    with pytest.raises(
        ValueError,
        match="checkpoint prompt conditioning must use sampling selection",
    ):
        methods._build_turn_prompt_selection_context(
            pipeline=pipeline,
            missing_percentile_policy="error",
            consolidation=None,
        )


def test_turn_statistics_snapshot_rejects_a_non_safetensors_extension(tmp_path: Path) -> None:
    # kills: dropping the extension guard, which would hand an arbitrary file's
    # bytes to the safetensors statistics parser after digest-checking them.
    path = tmp_path / "feature_statistics.pt"
    path.write_bytes(b"not safetensors")

    with pytest.raises(ValueError, match=r"Unknown statistics file extension: '\.pt'"):
        methods._load_turn_statistics_snapshot(path, expected_raw_sha256="0" * 64)


def test_turn_statistics_snapshot_hashes_and_parses_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    statistics_path = tmp_path / "feature_statistics.safetensors"
    save_file(
        {"weights": t.tensor([1.0, 2.0, 3.0])},
        str(statistics_path),
        metadata={"identity": "original"},
    )
    original_payload = statistics_path.read_bytes()
    expected_raw_sha256 = hashlib.sha256(original_payload).hexdigest()
    sentinel = _statistics_fixture()
    observed_payloads: list[bytes] = []

    def fake_parse(payload: bytes) -> tuple[Any, None, None]:
        observed_payloads.append(payload)
        # Replacing the live path here simulates an adversarial mutation in the
        # old verify(path) -> load(path) gap. The parser must still receive the
        # already-verified original snapshot.
        statistics_path.write_bytes(b"replacement after snapshot")
        return sentinel, None, None

    monkeypatch.setattr(turn_context_contract, "load_statistics_and_checksum_from_bytes", fake_parse)

    loaded = methods._load_turn_statistics_snapshot(
        statistics_path,
        expected_raw_sha256=expected_raw_sha256,
    )

    assert loaded is sentinel
    assert observed_payloads == [original_payload]
    assert statistics_path.read_bytes() != original_payload


def test_turn_statistics_snapshot_rejects_raw_digest_mismatch_before_parse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    statistics_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0])}, str(statistics_path))
    monkeypatch.setattr(
        turn_context_contract,
        "load_statistics_and_checksum_from_bytes",
        lambda _payload: pytest.fail("digest-mismatched statistics reached parsing"),
    )

    with pytest.raises(ValueError, match="raw SHA-256 does not match"):
        methods._load_turn_statistics_snapshot(
            statistics_path,
            expected_raw_sha256="0" * 64,
        )


def test_turn_prompt_selection_context_reuses_pipeline_percentile_table_identity() -> None:
    consolidation = _consolidation()
    pipeline = _turn_prompt_pipeline(seed=19)
    assert pipeline.negative_bundle is not None
    pipeline = replace(
        pipeline,
        feature_mask=consolidation.feature_mask,
        negative_bundle=replace(
            pipeline.negative_bundle,
            feature_mask=consolidation.feature_mask,
        ),
    )

    context = methods._build_turn_prompt_selection_context(
        pipeline=pipeline,
        missing_percentile_policy="lowest",
        consolidation=consolidation,
    )

    assert isinstance(pipeline.selection_config, SamplingBundle)
    assert context.fp_percentile_table is pipeline.selection_config.percentile_table
    assert context.cluster_threshold_percentile_table is pipeline.selection_config.percentile_table
    assert context.sampling_bundle is pipeline.selection_config
    assert context.negative_bundle is not None
    assert context.negative_bundle.rng is pipeline.selection_config.rng
    assert context.negative_bundle.feature_mask == consolidation.effective_negative_mask()
    # kills: hard-coding MissingPercentileResolver.error() instead of resolving
    # the checkpoint's recorded missing_percentile_policy, which would make an
    # out-of-coverage prompt feature abort a run that trained fine.
    assert context.missing_percentile.policy is MissingPercentilePolicy.LOWEST
    assert context.missing_percentile.covered is not None
    assert t.equal(
        context.missing_percentile.covered,
        percentile_table_coverage(pipeline.selection_config.percentile_table),
    )
    assert context.missing_percentile.substitutes


def test_turn_prompt_selection_matches_sft_producer_outputs_rng_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_pipeline = _turn_prompt_pipeline(seed=23)
    producer_pipeline = _turn_prompt_pipeline(seed=23)
    actual_context = methods._build_turn_prompt_selection_context(
        pipeline=actual_pipeline,
        missing_percentile_policy="error",
        consolidation=None,
    )
    # kills: hard-coding the coverage-backed resolver, which would silently
    # diverge an ERROR-policy checkpoint from its training-time selector.
    assert actual_context.missing_percentile.policy is MissingPercentilePolicy.ERROR
    assert actual_context.missing_percentile.covered is None
    assert isinstance(producer_pipeline.selection_config, SamplingBundle)
    producer_context = SegmentSelectionContext(
        threshold_cfg=None,
        precomputed_thresholds=None,
        sampling_bundle=producer_pipeline.selection_config,
        negative_bundle=producer_pipeline.negative_bundle,
        boundaries=producer_pipeline.boundaries,
        fp_percentile_table=producer_pipeline.selection_config.percentile_table,
        cluster_threshold_percentile_table=None,
        feature_mask=producer_pipeline.feature_mask,
        consolidation=None,
        missing_percentile=MissingPercentileResolver.error(),
    )
    raw_segments = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([0.2, 4.0, 8.0], dtype=t.float32),
            char_start=0,
            char_end=8,
        ),
        SegmentFeatures(
            segment_idx=1,
            position=3,
            feature_ids=t.tensor([1, 3, 4], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0, 9.0], dtype=t.float32),
            char_start=8,
            char_end=16,
        ),
    ]
    original_sample_features = sft_selection.sample_features
    observed_overrides: list[t.Tensor | None] = []

    def recording_sample_features(*args: Any, **kwargs: Any) -> Any:
        override = kwargs.get("override_weights")
        observed_overrides.append(None if override is None else override.clone())
        return original_sample_features(*args, **kwargs)

    monkeypatch.setattr(sft_selection, "sample_features", recording_sample_features)

    actual = sft_selection.select_conversation_segments(
        raw_segments,
        ctx=actual_context,
        example_idx=0,
    )
    expected = sft_selection.select_conversation_segments(
        raw_segments,
        ctx=producer_context,
        example_idx=0,
    )

    assert len(actual) == len(expected)
    for actual_segment, expected_segment in zip(actual, expected, strict=True):
        assert t.equal(actual_segment.feature_ids, expected_segment.feature_ids)
        assert t.equal(actual_segment.buckets, expected_segment.buckets)
        assert t.equal(actual_segment.percentiles, expected_segment.percentiles)
        assert actual_segment.char_start == expected_segment.char_start
        assert actual_segment.char_end == expected_segment.char_end
        assert actual_segment.all_feature_ids is not None
        assert expected_segment.all_feature_ids is not None
        assert t.equal(actual_segment.all_feature_ids, expected_segment.all_feature_ids)
        assert actual_segment.all_feature_pcts is not None
        assert expected_segment.all_feature_pcts is not None
        assert t.equal(actual_segment.all_feature_pcts, expected_segment.all_feature_pcts)
    assert isinstance(actual_pipeline.selection_config, SamplingBundle)
    assert t.equal(
        actual_pipeline.selection_config.rng.get_state(),
        producer_pipeline.selection_config.rng.get_state(),
    )
    assert len(observed_overrides) == 2 * len(raw_segments)
    assert all(override is not None for override in observed_overrides)
    for actual_override, expected_override in zip(
        observed_overrides[: len(raw_segments)],
        observed_overrides[len(raw_segments) :],
        strict=True,
    ):
        assert actual_override is not None
        assert expected_override is not None
        assert t.equal(actual_override, expected_override)


def test_prompt_statistics_contract_binds_bos_path_and_sae_identity(tmp_path: Path) -> None:
    valid_path = tmp_path / "context_independent" / "no_bos" / "feature_statistics.safetensors"
    statistics = _statistics_fixture()
    methods._validate_statistics_contract(
        statistics,
        path=valid_path,
        sae_model="gemma-3-4b-pt",
        sae_layer=22,
        n_sae_features=65_536,
    )

    statistics.ignore_bos_features = False
    with pytest.raises(ValueError, match="ignore_bos_features=true"):
        methods._validate_statistics_contract(
            statistics,
            path=valid_path,
            sae_model="gemma-3-4b-pt",
            sae_layer=22,
            n_sae_features=65_536,
        )
    statistics.ignore_bos_features = True
    with pytest.raises(ValueError, match="context_independent/no_bos"):
        methods._validate_statistics_contract(
            statistics,
            path=tmp_path / "feature_statistics.safetensors",
            sae_model="gemma-3-4b-pt",
            sae_layer=22,
            n_sae_features=65_536,
        )
    with pytest.raises(ValueError, match="Gemma-3-4B-PT"):
        methods._validate_statistics_contract(
            statistics,
            path=valid_path,
            sae_model="gemma-2-2b",
            sae_layer=22,
            n_sae_features=65_536,
        )


@pytest.mark.parametrize(
    ("feature_delta", "delta_weights", "error"),
    [
        (
            {"mode": "prev_segment"},
            DeltaSamplingWeights.percentile_delta,
            "feature_delta.mode=splice_counterfactual",
        ),
        (
            {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
            None,
            "feature_selection.delta_weights=percentile_delta",
        ),
    ],
)
def test_target_training_policy_is_closed(
    feature_delta: dict[str, object],
    delta_weights: DeltaSamplingWeights | None,
    error: str,
) -> None:
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        delta_weights=delta_weights,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    with pytest.raises(ValueError, match=error):
        methods._validate_target_policy(feature_delta, selection)


def test_dense_diffmean_is_full_coordinate_float32_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = iter(
        (
            t.tensor([[80.0, 10.0, 50.0, 20.0], [60.0, 30.0, 70.0, 40.0]]),
            t.tensor([[20.0, 30.0, 50.0, 10.0], [40.0, 10.0, 30.0, 50.0]]),
        )
    )
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )
    # The percentile lookup is reached through MissingPercentileResolver.ranks,
    # which under the `error` policy every fixed-mode setup resolves to is
    # exactly magnitude_to_percentile; patch it where the resolver reads it.
    monkeypatch.setattr(
        missing_percentile,
        "magnitude_to_percentile",
        lambda feature_ids, magnitudes, _table: magnitudes,
    )

    result = methods.fit_dense_diffmean(
        _setup(),
        object(),
        object(),
        ("positive-a", "positive-b"),
        ("negative-a", "negative-b"),
        concept_id=7,
    )

    assert t.equal(result, t.tensor([0.4, 0.0, 0.2, 0.0], dtype=t.float32))
    assert result.device.type == "cpu"
    assert result.dtype == t.float32
    assert result.is_contiguous()


def test_dense_diffmean_middle_coordinate_and_compact_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = iter((t.tensor([[10.0, 80.0, 70.0, 60.0]]), t.tensor([[10.0, 20.0, 10.0, 40.0]])))
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )
    monkeypatch.setattr(
        missing_percentile,
        "magnitude_to_percentile",
        lambda feature_ids, magnitudes, _table: magnitudes,
    )
    vocab = FeatureVocab(n_sae_features=4, original_ids=(1, 2, 3))

    result = methods.fit_dense_diffmean(
        _setup(
            allowed_ids=t.tensor([1, 2, 3]),
            feature_vocab=vocab,
        ),
        object(),
        object(),
        ("positive",),
        ("negative",),
        concept_id=0,
    )

    assert t.equal(result, t.tensor([0.6, 0.6, 0.2]))
    assert vocab.original_ids == (1, 2, 3)


def test_dense_diffmean_masks_before_percentile_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = iter((t.tensor([[99.0, 80.0, 99.0, 60.0]]), t.tensor([[99.0, 20.0, 99.0, 40.0]])))
    observed_ids: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )

    def fake_percentiles(feature_ids: t.Tensor, magnitudes: t.Tensor, _table: t.Tensor) -> t.Tensor:
        observed_ids.append(tuple(int(value) for value in feature_ids.tolist()))
        return magnitudes

    monkeypatch.setattr(missing_percentile, "magnitude_to_percentile", fake_percentiles)
    vocab = FeatureVocab(n_sae_features=4, original_ids=(1, 3))
    result = methods.fit_dense_diffmean(
        _setup(allowed_ids=t.tensor([1, 3]), feature_vocab=vocab),
        object(),
        object(),
        ("positive",),
        ("negative",),
        concept_id=0,
    )

    assert observed_ids == [(1, 3), (1, 3)]
    assert t.equal(result, t.tensor([0.6, 0.2]))


def test_clustered_percentiles_max_fold_each_document_before_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        missing_percentile,
        "magnitude_to_percentile",
        lambda feature_ids, magnitudes, _table: magnitudes,
    )
    activations = t.tensor(
        [
            [99.0, 10.0, 99.0, 90.0, 40.0],
            [99.0, 80.0, 99.0, 20.0, 60.0],
        ]
    )

    actual = methods.dense_mean_percentiles(
        activations, percentile_table=t.zeros((5, 99), dtype=t.float64),
        allowed_ids=t.tensor([1, 3, 4]), consolidation=_consolidation(),
        missing=MissingPercentileResolver.error(),
    )

    assert t.equal(actual, t.tensor([0.0, 85.0, 0.0, 0.0, 50.0], dtype=t.float64))


def _uncovered_percentile_table() -> t.Tensor:
    """A four-feature table whose feature 2 row is the all-zero 'never seen' row.

    ``build_percentile_table`` leaves exactly this row for a feature the
    statistics artifact never saw, and it is what a turn-delimited checkpoint's
    own SFT-domain statistics leave for a feature the AxBench concept corpus
    activates but the SFT mixture never did.
    """
    table = t.linspace(1.0, 99.0, 99, dtype=t.float64).unsqueeze(0).repeat(4, 1)
    table[2] = 0.0
    return table


def _turn_diffmean_setup(
    policy: str,
    table: t.Tensor,
    *,
    consolidation: FeatureClusterConsolidation | None = None,
) -> Any:
    """A turn setup whose prompt selector carries the lineage's recorded policy."""
    rng = t.Generator(device="cpu").manual_seed(11)
    feature_mask = None if consolidation is None else consolidation.feature_mask
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=replace(_selection(), delta_weights=None),
            percentile_table=table,
            rng=rng,
        ),
        boundaries=t.zeros((4, 0), dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=_negative_selection(),
            n_sae_features=4,
            rng=rng,
            feature_mask=feature_mask,
        ),
        effective_buckets=1,
        feature_mask=feature_mask,
    )
    return _fake_setup(
        sae_layer=22,
        sae_model="gemma-3-4b-pt",
        n_sae_features=4,
        percentile_table=table,
        allowed_ids=None,
        feature_vocab=None,
        consolidation=consolidation,
        output_selection=_selection(),
        output_negative_selection=_negative_selection(),
        feature_mask=feature_mask,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        prompt_selection_context=methods._build_turn_prompt_selection_context(
            pipeline=pipeline,
            missing_percentile_policy=policy,
            consolidation=consolidation,
        ),
    )


@pytest.mark.parametrize("mode", ["turn_lowest", "turn_error", "fixed"])
def test_clustered_dense_diffmean_composes_policy_mask_and_compact_row_order(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    # Exercise the complete fit, not only the document mean. Kills:
    # bypassing the checkpoint resolver in either polarity, checking cluster
    # representatives instead of all allowed siblings, mean-before-max folding,
    # and omitting compact FP-row projection.
    definition = FeatureClusterDefinition(clusters=((1, 2),))
    consolidation = FeatureClusterConsolidation.build(
        ResolvedFeatureClusters(
            source_path=Path("unused-clusters.yaml"),
            definition=definition,
            descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=4),
        ),
        FeatureMask("whitelist", (1, 2, 3)),
    )
    table = _uncovered_percentile_table()
    table[0] = 0.0  # A masked-out hole must never enter the required population.
    vocab = FeatureVocab(n_sae_features=4, original_ids=(1, 3))
    if mode == "fixed":
        table[2] = t.arange(1, 100, dtype=t.float64)
        setup = _setup(consolidation=consolidation, feature_vocab=vocab)
        setup.percentile_table = table
    else:
        setup = _turn_diffmean_setup(
            "lowest" if mode == "turn_lowest" else "error",
            table,
            consolidation=consolidation,
        )
        setup.feature_vocab = vocab
    setup.feature_mask = consolidation.feature_mask
    # Exactly what `effective_feature_vocab(...).new_to_old_tensor()` yields for
    # this contract: cluster REPRESENTATIVES only. The clustered arm must build
    # its coverage population from `consolidation.allowed_universe` (1, 2, 3)
    # instead, so swapping in `setup.allowed_ids` drops the uncovered sibling 2
    # from the required set and the `error` arm below stops raising.
    setup.allowed_ids = t.tensor([1, 3], dtype=t.int64)
    extracted = iter(
        (
            t.tensor([[99.0, 60.0, 10.0, 80.0], [99.0, 20.0, 80.0, 60.0]]),
            t.tensor([[99.0, 20.0, 30.0, 20.0], [99.0, 10.0, 30.0, 40.0]]),
        )
    )
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )
    if mode == "turn_error":
        with pytest.raises(
            MissingPercentileError,
            match=r"missing_percentile_policy='error'.*for 1 of the 3 distinct.*examples: \[2\]",
        ):
            methods.fit_dense_diffmean(
                setup, object(), object(), ("p0", "p1"), ("n0", "n1"), concept_id=3
            )
        return

    result = methods.fit_dense_diffmean(
        setup, object(), object(), ("p0", "p1"), ("n0", "n1"), concept_id=3
    )
    # Fixed: representative 1 pools [60, 80] versus [30, 30] => 0.4.
    # Turn lowest: sibling 2 is pinned to 0.5 in both classes, so representative
    # 1 pools [60, 20] versus [20, 10] => 0.25. Singleton 3 always gives 0.4.
    expected = t.tensor([0.4 if mode == "fixed" else 0.25, 0.4], dtype=t.float32)
    assert t.equal(result, expected)
    assert result.dtype == t.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()


# kills: calling magnitude_to_percentile directly in the turn-mode DiffMean fit
# instead of the lineage's MissingPercentileResolver -- feature 2's all-zero row
# would abort the fit under `lowest`, exactly as GPU job 22040044 did. Also
# kills dropping the uncovered id from the queried set instead of pinning it:
# an excluded id leaves the zero-initialised mean at 0.0, not the policy's 0.5.
def test_turn_dense_diffmean_pins_an_uncovered_feature_under_lowest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _uncovered_percentile_table()
    setup = _turn_diffmean_setup("lowest", table)
    positive = t.tensor([[60.0, 60.0, 60.0, 60.0]])
    negative = t.tensor([[20.0, 20.0, 20.0, 20.0]])
    extracted = iter((positive.clone(), negative.clone()))
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )

    result = methods.fit_dense_diffmean(
        setup,
        object(),
        object(),
        ("positive",),
        ("negative",),
        concept_id=3,
    )

    # Rows [1..99] make mag 60 -> rank 60.0 and mag 20 -> rank 20.0 exactly, so
    # every covered feature contrasts to 0.4 while the pinned one contrasts to 0.
    assert t.equal(result, t.tensor([0.4, 0.4, 0.0, 0.4], dtype=t.float32))
    assert setup.prompt_selection_context.missing_percentile.substitutes
    means = methods.dense_mean_percentiles(
        positive, percentile_table=table, allowed_ids=t.arange(4, dtype=t.int64),
        consolidation=None, missing=setup.prompt_selection_context.missing_percentile,
    )
    assert float(means[2]) == LOWEST_PERCENTILE_RANK
    assert float(means[0]) == 60.0


# kills: leaving the `error` policy to magnitude_to_percentile's own raise,
# which names neither the policy in force nor the complete uncovered
# population, and firing on whichever id happened to be looked up first.
def test_turn_dense_diffmean_names_the_policy_and_the_hole_under_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _turn_diffmean_setup("error", _uncovered_percentile_table())
    extracted = iter((t.tensor([[60.0, 60.0, 60.0, 60.0]]), t.tensor([[20.0, 20.0, 20.0, 20.0]])))
    monkeypatch.setattr(
        methods, "_batched_dense_sae_max_pool", lambda *_args, **_kwargs: next(extracted)
    )

    with pytest.raises(
        MissingPercentileError,
        match=(
            r"dense DiffMean cannot rank every mask-allowed feature under "
            r"missing_percentile_policy='error'.*missing percentile statistics "
            r"for 1 of the 4 distinct feature ids queried as dense DiffMean "
            r"mask-allowed feature ids \(examples: \[2\]\)"
        ),
    ):
        methods.fit_dense_diffmean(
            setup,
            object(),
            object(),
            ("positive",),
            ("negative",),
            concept_id=3,
        )


# kills: running the coverage pre-check in FIXED mode too. The fixed-mode
# doubles above carry an all-zero percentile table, so an unconditional check
# would report every feature uncovered and abort them under the default policy;
# this states the invariant those doubles rely on outright.
def test_fixed_mode_dense_diffmean_keeps_the_primitive_as_its_only_guard() -> None:
    setup = _setup()

    assert setup.prompt_selection_context is None
    resolver = methods._diffmean_missing_percentile(setup, t.arange(4, dtype=t.int64))

    assert resolver.policy is MissingPercentilePolicy.ERROR
    assert resolver.covered is None
    assert not resolver.substitutes


@pytest.mark.parametrize(
    ("positive", "negative", "error"),
    [
        ((), ("negative",), "nonempty positive and negative"),
        (("positive",), (), "nonempty positive and negative"),
    ],
)
def test_dense_diffmean_rejects_empty_classes(
    positive: tuple[str, ...], negative: tuple[str, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        methods.fit_dense_diffmean(_setup(), object(), object(), positive, negative, concept_id=0)


@pytest.mark.parametrize("wrong_statistics", [False, True])
def test_dense_diffmean_rejects_wrong_activation_width(monkeypatch: pytest.MonkeyPatch, wrong_statistics: bool) -> None:
    monkeypatch.setattr(
        methods,
        "_batched_dense_sae_max_pool",
        lambda *_args, **_kwargs: t.ones((1, 3)),
    )
    setup = _setup()
    if wrong_statistics:
        setup.percentile_table = t.ones(3, 99, dtype=t.float64)
    with pytest.raises(ValueError, match="statistics width disagrees" if wrong_statistics else "positive width 3 != 4"):
        methods.fit_dense_diffmean(
            setup, object(), object(), ("positive",), ("negative",), concept_id=0
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_dense_diffmean_rejects_nonfinite_activations(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
    polarity: str,
) -> None:
    monkeypatch.setattr(
        methods,
        "_batched_dense_sae_max_pool",
        lambda texts, *_args, **_kwargs: t.tensor([[bad_value if texts == [polarity] else 1.0, 1.0, 1.0, 1.0]]),
    )
    setup = _setup()
    setup.percentile_table.copy_(t.arange(1, 100, dtype=t.float64).repeat(4, 1))
    with pytest.raises(ValueError, match=f"{polarity} activations contain NaN or Inf"):
        methods.fit_dense_diffmean(
            setup, object(), object(), ("positive",), ("negative",), concept_id=0
        )


def test_dense_diffmean_rejects_nonpositive_extraction_batch_size() -> None:
    with pytest.raises(ValueError, match="extraction_batch_size must be positive"):
        methods.fit_dense_diffmean(
            _setup(),
            object(),
            object(),
            ("positive",),
            ("negative",),
            concept_id=0,
            extraction_batch_size=0,
        )


def test_prepare_prompt_context_passes_resolved_sae_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    resolved_hook = "blocks.22.hook_resid_post"
    setup = _fake_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name=resolved_hook),
        n_sae_features=4,
        prompt_pipeline=object(),
        prompt_consolidation=None,
    )
    gemma_tokenizer = SimpleNamespace(
        bos_token_id=1,
        decode=lambda _ids: "<bos>",
    )
    base_model = SimpleNamespace(
        tokenizer=gemma_tokenizer,
        cfg=SimpleNamespace(n_ctx=4096),
        nnterp_model=object(),
    )
    nanochat_tokenizer = object()

    monkeypatch.setattr(methods, "context_text_for_extraction", lambda *_args: "prompt")
    monkeypatch.setattr(
        methods,
        "truncate_to_training_window",
        lambda *_args: ([1, 2], 2),
    )

    def fake_extract_document_features(**kwargs: Any) -> tuple[Any, ...]:
        observed.update(kwargs)
        return ([t.tensor([1])], [t.tensor([0.5])], [0], [0], [6], 2)

    monkeypatch.setattr(methods, "extract_document_features", fake_extract_document_features)
    monkeypatch.setattr(methods, "resolve_bos_token_id", lambda _tokenizer: 1)
    monkeypatch.setattr(
        methods,
        "build_context_entry_from_extracted_segments",
        lambda **kwargs: SimpleNamespace(
            token_ids=tuple(kwargs["token_ids"]),
            segment_mode=methods.SegmentMode.fixed_tokens,
            token_assignments=None,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(1,),
                    buckets=(0,),
                    char_start=0,
                    char_end=6,
                ),
            ),
        ),
    )

    prepared = methods.prepare_prompt_context(
        setup,  # ty: ignore[invalid-argument-type]
        base_model,
        object(),
        nanochat_tokenizer,
        token_ids=(11, 12),
        assistant_end_id=13,
    )

    assert observed["model"] is base_model
    assert observed["context_independent_features"] is True
    assert observed["hook_name"] == resolved_hook
    assert prepared.token_ids == (11, 12)


def test_prepare_prompt_context_rejects_overlong_prompt_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _fake_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
    )
    base_model = SimpleNamespace(
        tokenizer=object(),
        cfg=SimpleNamespace(n_ctx=4096),
    )
    monkeypatch.setattr(methods, "context_text_for_extraction", lambda *_args: "prompt")
    monkeypatch.setattr(
        methods,
        "truncate_to_training_window",
        lambda *_args: ([1] * methods.PROMPT_CONTEXT_WINDOW, 2049),
    )

    def unexpected_extraction(**_kwargs: Any) -> None:
        pytest.fail("overlong prompt reached feature extraction")

    monkeypatch.setattr(methods, "extract_document_features", unexpected_extraction)

    with pytest.raises(ValueError, match="2049 Gemma tokens.*2048"):
        methods.prepare_prompt_context(
            setup,  # ty: ignore[invalid-argument-type]
            base_model,
            object(),
            object(),
            token_ids=(1, 2),
            assistant_end_id=9,
        )


@pytest.mark.parametrize(
    "segment_mode",
    [
        methods.SegmentMode.fixed_tokens,
        methods.SegmentMode.turn_delimited_fixed_tokens,
    ],
)
def test_prompt_context_window_exceeding_model_fails_before_tokenization_extraction_or_rng(
    monkeypatch: pytest.MonkeyPatch,
    segment_mode: methods.SegmentMode,
) -> None:
    pipeline = _turn_prompt_pipeline(seed=31)
    selection_context = methods._build_turn_prompt_selection_context(
        pipeline=pipeline,
        missing_percentile_policy="error",
        consolidation=None,
    )
    assert isinstance(pipeline.selection_config, SamplingBundle)
    rng_state = pipeline.selection_config.rng.get_state().clone()
    setup = _fake_setup(
        segment_mode=segment_mode,
        prompt_segment_size=17,
        prompt_context_window=257,
        prompt_pipeline=pipeline,
        prompt_selection_context=selection_context,
    )
    extraction_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=256),
        tokenizer=object(),
    )

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid extraction window reached tokenization or extraction")

    monkeypatch.setattr(methods, "context_text_for_extraction", unexpected)
    monkeypatch.setattr(methods, "rendered_text_and_turn_char_starts", unexpected)
    monkeypatch.setattr(methods, "truncate_to_training_window", unexpected)
    monkeypatch.setattr(methods, "extract_document_features", unexpected)
    monkeypatch.setattr(methods, "select_conversation_segments", unexpected)

    with pytest.raises(
        ValueError,
        match="extraction context window 257 exceeds the extraction model context length 256",
    ):
        methods.build_extracted_context_entry(
            setup,  # ty: ignore[invalid-argument-type]
            extraction_model,
            object(),
            object(),
            token_ids=(1, 2),
            source="window preflight",
        )

    assert t.equal(pipeline.selection_config.rng.get_state(), rng_state)


def test_prepare_turn_prompt_uses_exact_render_boundaries_and_live_sft_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _CharacterTokenizer()
    rendered_ids, _mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "Hello 世界!"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    assistant_end = tokenizer.encode_special("<assistant_end>")
    assert rendered_ids[-1] == assistant_end
    prompt_ids = tuple(rendered_ids[:-1])
    expected_text = tokenizer.decode(prompt_ids[1:])
    expected_turn_starts = (
        0,
        len("<user_start>Hello 世界!<user_end>"),
    )
    expected_segments = (
        ContextFeatureSegment((7,), (0,), 0, expected_turn_starts[1]),
        ContextFeatureSegment(
            (9,),
            (1,),
            expected_turn_starts[1],
            len(expected_text),
        ),
    )
    selection_context = object()
    setup = _fake_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
        n_sae_features=64,
        prompt_pipeline=object(),
        prompt_consolidation=None,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        prompt_segment_size=17,
        prompt_context_window=333,
        prompt_selection_context=selection_context,
    )

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, ids: list[int]) -> str:
            assert ids == [1]
            return "<bos>"

    extraction_model = SimpleNamespace(
        tokenizer=_GemmaTokenizer(),
        cfg=SimpleNamespace(n_ctx=8192),
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        methods,
        "truncate_to_training_window",
        lambda text, *_args: ([1, 2, 3], 3) if text == expected_text else pytest.fail(text),
    )

    def fake_extract(**kwargs: Any) -> tuple[Any, ...]:
        observed["extract"] = kwargs
        boundary = 5 + expected_turn_starts[1]
        return (
            [t.tensor([3]), t.tensor([4])],
            [t.tensor([0.5]), t.tensor([0.75])],
            [0, 2],
            [5, boundary],
            [boundary, 5 + len(expected_text)],
            3,
        )

    def fake_select(raw_segments: Any, *, ctx: Any, example_idx: int) -> Any:
        observed["raw_segments"] = raw_segments
        assert ctx is selection_context
        assert example_idx == 0
        return [
            SimpleNamespace(
                feature_ids=t.tensor([7]),
                buckets=t.tensor([0]),
                char_start=raw_segments[0].char_start,
                char_end=raw_segments[0].char_end,
            ),
            SimpleNamespace(
                feature_ids=t.tensor([9]),
                buckets=t.tensor([1]),
                char_start=raw_segments[1].char_start,
                char_end=raw_segments[1].char_end,
            ),
        ]

    def fake_build_selected(**kwargs: Any) -> Any:
        observed["build"] = kwargs
        return SimpleNamespace(
            token_ids=prompt_ids,
            segments=expected_segments,
            segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
            token_assignments=tuple([0] * (len(prompt_ids) - 1) + [1]),
        )

    monkeypatch.setattr(methods, "extract_document_features", fake_extract)
    monkeypatch.setattr(methods, "select_conversation_segments", fake_select)
    monkeypatch.setattr(
        methods,
        "build_context_entry_from_selected_segments",
        fake_build_selected,
    )

    sae = object()
    prepared = methods.prepare_prompt_context(
        setup,  # ty: ignore[invalid-argument-type]
        extraction_model,
        sae,
        tokenizer,
        token_ids=prompt_ids,
        assistant_end_id=assistant_end,
    )

    assert prepared == methods.PreparedPromptContext(
        token_ids=prompt_ids,
        assistant_end_id=assistant_end,
        segments=expected_segments,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        token_assignments=tuple([0] * (len(prompt_ids) - 1) + [1]),
    )
    assert observed["extract"] == {
        "model": extraction_model,
        "sae": sae,
        "text": expected_text,
        "layer": 22,
        "context_window": 333,
        "pooling": methods.PoolingMode.max_pool,
        "segment_mode": methods.SegmentMode.turn_delimited_fixed_tokens,
        "segment_size": 17,
        "hook_name": "blocks.22.hook_resid_post",
        "context_independent_features": True,
        "ignore_bos_features": True,
        "turn_char_starts": expected_turn_starts,
    }
    assert [segment.segment_idx for segment in observed["raw_segments"]] == [0, 1]
    assert [segment.position for segment in observed["raw_segments"]] == [0, 2]
    assert observed["build"]["token_ids"] == prompt_ids
    assert observed["build"]["source_bos_char_len"] == 5


def _turn_extraction_setup(**overrides: Any) -> Any:
    """A turn-mode setup double for ``build_extracted_context_entry`` guards."""

    return _fake_setup(
        **{
            "sae_layer": 22,
            "resolved_sae": SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
            "n_sae_features": 64,
            "prompt_pipeline": object(),
            "prompt_consolidation": None,
            "segment_mode": methods.SegmentMode.turn_delimited_fixed_tokens,
            "prompt_segment_size": 17,
            "prompt_context_window": 333,
            "prompt_selection_context": object(),
            **overrides,
        }
    )


def test_extracted_context_entry_rejects_an_empty_token_tuple() -> None:
    # kills: dropping the nonempty-token guard, which would send an empty
    # rendered row into the extractor and produce a zero-segment entry.
    with pytest.raises(ValueError, match=r"empty prompt: token tuple must be nonempty"):
        methods.build_extracted_context_entry(
            _turn_extraction_setup(),
            SimpleNamespace(cfg=SimpleNamespace(n_ctx=8192), tokenizer=object()),
            object(),
            object(),
            token_ids=(),
            source="empty prompt",
        )


_USABLE_GEMMA_TOKENIZER = SimpleNamespace(bos_token_id=1, decode=lambda _ids: "<bos>")
_ONE_SEGMENT_EXTRACTION: tuple[Any, ...] = ([t.tensor([1])], [t.tensor([1.0])], [0], [0], [1], 1)


def _build_turn_context_entry_over_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    setup: Any,
    source: str,
    gemma_tokenizer: Any = _USABLE_GEMMA_TOKENIZER,
    extraction: tuple[Any, ...] = _ONE_SEGMENT_EXTRACTION,
    turn_starts: tuple[int, ...] = (0,),
) -> Any:
    """Drive the turn branch of ``build_extracted_context_entry`` over stubs.

    Only the three expensive collaborators are replaced (turn rendering, the
    training-window truncation, and Gemma extraction); every guard under test,
    and the selector dispatch behind them, stay real.
    """

    monkeypatch.setattr(
        methods,
        "rendered_text_and_turn_char_starts",
        lambda *_a: ("t", turn_starts),
    )
    monkeypatch.setattr(methods, "truncate_to_training_window", lambda *_a: ([1], 1))
    monkeypatch.setattr(methods, "extract_document_features", lambda **_kwargs: extraction)
    return methods.build_extracted_context_entry(
        setup,
        SimpleNamespace(cfg=SimpleNamespace(n_ctx=8192), tokenizer=gemma_tokenizer),
        object(),
        object(),
        token_ids=(1, 2),
        source=source,
    )


@pytest.mark.parametrize(
    ("bos_token_id", "decoded", "message"),
    [
        (None, "<bos>", "Gemma tokenizer has no BOS token"),
        (1, "", "Gemma tokenizer BOS decodes to empty text"),
    ],
)
def test_extracted_context_entry_rejects_an_unusable_gemma_bos(
    monkeypatch: pytest.MonkeyPatch,
    bos_token_id: int | None,
    decoded: str,
    message: str,
) -> None:
    # kills: dropping either BOS guard. The BOS character width is the shift
    # that maps Gemma character ranges onto nanochat tokens, so a missing or
    # zero-width BOS would silently offset every converted segment.
    class _GemmaTokenizer:
        def decode(self, _ids: list[int]) -> str:
            return decoded

    gemma_tokenizer = _GemmaTokenizer()
    gemma_tokenizer.bos_token_id = bos_token_id  # ty: ignore[unresolved-attribute]

    with pytest.raises(ValueError, match=message):
        _build_turn_context_entry_over_stubs(
            monkeypatch,
            setup=_turn_extraction_setup(),
            source="bos prompt",
            gemma_tokenizer=gemma_tokenizer,
        )


def test_extracted_context_entry_requires_a_turn_prompt_selection_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping the missing-selection-context guard, which would reach
    # select_conversation_segments with ctx=None on a turn checkpoint.
    monkeypatch.setattr(
        methods,
        "select_conversation_segments",
        lambda *_a, **_k: pytest.fail("a context-less turn setup reached the selector"),
    )

    with pytest.raises(
        ValueError,
        match=r"turn prompt: turn-delimited prompt selection context is missing",
    ):
        _build_turn_context_entry_over_stubs(
            monkeypatch,
            setup=_turn_extraction_setup(prompt_selection_context=None),
            source="turn prompt",
        )


def test_extracted_context_entry_rejects_ragged_turn_extraction_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: dropping strict=True from the SegmentFeatures zip, which would let
    # a short positions array truncate the segment list instead of failing
    # loudly, and moving any selection work ahead of that zip.
    monkeypatch.setattr(
        methods,
        "select_conversation_segments",
        lambda *_a, **_k: pytest.fail("ragged extraction must fail before selection"),
    )

    with pytest.raises(ValueError, match=r"zip\(\) argument 3 is shorter"):
        _build_turn_context_entry_over_stubs(
            monkeypatch,
            setup=_turn_extraction_setup(),
            source="ragged prompt",
            extraction=(
                [t.tensor([1]), t.tensor([2])],
                [t.tensor([1.0]), t.tensor([2.0])],
                [0],
                [0, 1],
                [1, 2],
                1,
            ),
            turn_starts=(0, 1),
        )


def test_turn_prompt_context_composes_real_selection_and_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the turn branch with its real selector and real converter.

    Only ``extract_document_features`` and the Gemma tokenizer are stubbed;
    ``select_conversation_segments`` and
    ``build_context_entry_from_selected_segments`` run for real, so the
    ``SegmentFeatures`` glue in ``build_extracted_context_entry`` is exercised
    end to end against an independent hand computation of the token intervals.
    """

    # kills: swapping feature_ids/feature_mags in the SegmentFeatures loop
    # (silent at production SAE width), and any change that stops passing the
    # extractor's Gemma char ranges through to the selector unaltered.
    tokenizer = _CharacterTokenizer()
    rendered_ids, _mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "known"},
                {"role": "user", "content": "last"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    prompt_ids = tuple(rendered_ids[:-1])
    assistant_end = tokenizer.encode_special("<assistant_end>")
    assert rendered_ids[-1] == assistant_end
    rendered_text, turn_starts = methods.rendered_text_and_turn_char_starts(
        prompt_ids,
        tokenizer,
    )
    gemma_bos_text = "<bos>"
    bos_len = len(gemma_bos_text)
    source_starts = [0, *(bos_len + start for start in turn_starts[1:])]
    source_ends = [*source_starts[1:], bos_len + len(rendered_text)]
    n_segments = len(source_starts)

    extracted_ids = [t.tensor([index, index + 1], dtype=t.int64) for index in range(n_segments)]
    extracted_mags = [
        t.tensor([9.0 - index, 0.5 + index], dtype=t.float64) for index in range(n_segments)
    ]
    pipeline = _turn_prompt_pipeline(seed=31)
    selection_context = methods._build_turn_prompt_selection_context(
        pipeline=pipeline,
        missing_percentile_policy="error",
        consolidation=None,
    )
    setup = _fake_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
        n_sae_features=5,
        prompt_pipeline=pipeline,
        prompt_consolidation=None,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        prompt_segment_size=17,
        prompt_context_window=333,
        prompt_selection_context=selection_context,
    )

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, ids: list[int]) -> str:
            assert ids == [1]
            return gemma_bos_text

    extraction_model = SimpleNamespace(
        tokenizer=_GemmaTokenizer(),
        cfg=SimpleNamespace(n_ctx=8192),
    )
    monkeypatch.setattr(
        methods,
        "truncate_to_training_window",
        lambda text, *_args: ([1], 3) if text == rendered_text else pytest.fail(text),
    )
    raw_segments_seen: list[Any] = []
    real_select = methods.select_conversation_segments

    def recording_select(raw_segments: Any, **kwargs: Any) -> Any:
        raw_segments_seen.extend(raw_segments)
        return real_select(raw_segments, **kwargs)

    monkeypatch.setattr(methods, "select_conversation_segments", recording_select)
    monkeypatch.setattr(
        methods,
        "extract_document_features",
        lambda **_kwargs: (
            extracted_ids,
            extracted_mags,
            list(range(n_segments)),
            source_starts,
            source_ends,
            3,
        ),
    )

    entry = methods.build_extracted_context_entry(
        setup,  # ty: ignore[invalid-argument-type]
        extraction_model,
        object(),
        tokenizer,
        token_ids=prompt_ids,
        source="turn composition",
    )

    # (a) The raw SegmentFeatures carry the extractor's own tensors, cast but
    # never transposed: a magnitudes/ids swap changes both value and dtype.
    assert [segment.segment_idx for segment in raw_segments_seen] == list(range(n_segments))
    assert [segment.position for segment in raw_segments_seen] == list(range(n_segments))
    assert [segment.char_start for segment in raw_segments_seen] == source_starts
    assert [segment.char_end for segment in raw_segments_seen] == source_ends
    for index, segment in enumerate(raw_segments_seen):
        assert segment.feature_ids.dtype is t.int32
        assert segment.feature_mags.dtype is t.float32
        assert t.equal(segment.feature_ids, extracted_ids[index].to(t.int32))
        assert t.equal(segment.feature_mags, extracted_mags[index].to(t.float32))

    # (b) The converted nanochat segments land on the independently computed
    # token-offset boundaries for the same source character ranges.
    token_offsets = token_to_char_offsets(tokenizer, list(prompt_ids))
    expected_token_starts, expected_token_ends = compute_doc_token_boundaries_from_tokens(
        list(prompt_ids),
        source_starts,
        source_ends,
        tokenizer,
        source_bos_char_len=NonNegativeInt.parse(bos_len),
    )
    assert len(entry.segments) == n_segments
    for index, segment in enumerate(entry.segments):
        assert segment.char_start == token_offsets[int(expected_token_starts[index].item())]
        assert segment.char_end == token_offsets[int(expected_token_ends[index].item())]
    assert tuple(entry.token_ids) == prompt_ids

    # (c) The composed entry survives the runtime turn-prefill contract.
    prepared = TargetAwareEncoderDecoderEvalWrapper._prepare_turn_self_cond_prompt_context(
        cast(
            Any,
            SimpleNamespace(
                tokenizer=tokenizer,
                _turn_specials=ChatSpecialTokens.from_tokenizer(tokenizer),
                self_cond_segment_mode=entry.segment_mode,
                context_provider=entry,
            ),
        ),
        list(prompt_ids),
        entry.segments,
    )
    assert prepared is not None


def test_target_load_ignores_cluster_source_label_but_checks_canonical_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consolidation = _consolidation()
    setup_contract = CheckpointFeatureContract(
        feature_mask=consolidation.feature_mask,
        resolved_clusters=consolidation.resolved,
    )
    loaded_clusters = ResolvedFeatureClusters(
        source_path=Path("loaded-checkpoint-directory"),
        definition=consolidation.resolved.definition,
        descriptor=consolidation.resolved.descriptor,
    )
    loaded_contract = CheckpointFeatureContract(
        feature_mask=consolidation.feature_mask,
        resolved_clusters=loaded_clusters,
    )
    vocab = FeatureVocab(n_sae_features=5, original_ids=(1, 4))
    model = SimpleNamespace(feature_vocab=vocab)
    tokenizer = object()
    setup = _fake_setup(
        meta={"step": 7},
        target_meta_path=Path("checkpoint/meta_000007.json"),
        feature_contract=setup_contract,
        feature_vocab=vocab,
    )
    monkeypatch.setattr(
        methods,
        "build_encdec_model",
        lambda *_args, **_kwargs: (model, tokenizer, {"loaded": True}),
    )
    monkeypatch.setattr(
        methods,
        "resolve_and_assert_model_feature_contract",
        lambda *_args, **_kwargs: loaded_contract,
    )

    assert methods.load_target_model(
        setup,  # ty: ignore[invalid-argument-type]
        "cpu",
    ) == (model, tokenizer)


_TURN_MODE_VALUE = methods.SegmentMode.turn_delimited_fixed_tokens.value
_TURN_IDENTITY_META: dict[str, Any] = {
    "sft_segment_mode": _TURN_MODE_VALUE,
    "turn_self_cond_segment_size_rustbpe": 37,
    "structured_sft_online_sae_lineage_v1": {
        "schema_version": 1,
        "source_segment_mode": _TURN_MODE_VALUE,
        "source_segment_size": 17,
    },
    "selfcond_data": {"sft_segment_mode": _TURN_MODE_VALUE},
}


def _turn_target_load(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded_meta: dict[str, Any],
    feature_contract: Any,
) -> tuple[Any, Any, Any]:
    """Stage ``load_target_model`` over a turn setup with a stubbed model build.

    ``feature_contract`` is whatever
    ``resolve_and_assert_model_feature_contract`` should do -- a contract for
    the accept path, a ``pytest.fail`` sink for the rejection matrix.
    """

    model = SimpleNamespace(feature_vocab=None)
    tokenizer = object()
    setup = _fake_setup(
        meta={"step": 7, **_TURN_IDENTITY_META},
        target_meta_path=Path("checkpoint/meta_000007.json"),
        feature_contract=CheckpointFeatureContract(None, None),
        feature_vocab=None,
        segment_mode=methods.SegmentMode.turn_delimited_fixed_tokens,
        prompt_segment_size=17,
        turn_self_cond_segment_size_rustbpe=37,
    )
    monkeypatch.setattr(
        methods,
        "build_encdec_model",
        lambda *_args, **_kwargs: (model, tokenizer, loaded_meta),
    )
    monkeypatch.setattr(methods, "resolve_and_assert_model_feature_contract", feature_contract)
    return setup, model, tokenizer


@pytest.mark.parametrize(
    ("key", "drifted_value"),
    [
        ("sft_segment_mode", methods.SegmentMode.fixed_tokens.value),
        (
            "structured_sft_online_sae_lineage_v1",
            {
                "schema_version": 1,
                "source_segment_mode": _TURN_MODE_VALUE,
                "source_segment_size": 18,
            },
        ),
        ("turn_self_cond_segment_size_rustbpe", 64),
        ("selfcond_data", {"sft_segment_mode": methods.SegmentMode.fixed_tokens.value}),
    ],
)
def test_target_load_rejects_each_drifted_segmentation_identity_key(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    drifted_value: Any,
) -> None:
    # kills: dropping ANY single key from SEGMENTATION_IDENTITY_META_KEYS (each
    # case leaves the other three byte-equal, so only that key's conjunct can
    # raise), neutralizing the comparison outright, and moving it after the
    # feature-contract load (which is wired to fail).
    setup, _model, _tokenizer = _turn_target_load(
        monkeypatch,
        loaded_meta={"step": 7, **_TURN_IDENTITY_META, key: drifted_value},
        feature_contract=lambda *_args, **_kwargs: pytest.fail(
            f"drifted {key} reached feature-contract loading"
        ),
    )

    with pytest.raises(
        ValueError,
        match=rf"turn-segmentation metadata differs from the prevalidated "
        rf"AxBench setup: \['{key}'\]",
    ):
        methods.load_target_model(setup, "cpu")


def test_target_load_accepts_byte_equal_segmentation_identity_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Non-vacuity partner for the matrix above: with all four identity blocks
    # byte-equal the reload guard must not fire, so each rejection there is
    # caused by its own perturbation and not by the fixture.
    setup, model, tokenizer = _turn_target_load(
        monkeypatch,
        loaded_meta={"step": 7, **_TURN_IDENTITY_META},
        feature_contract=lambda *_args, **_kwargs: CheckpointFeatureContract(None, None),
    )

    assert methods.load_target_model(
        setup,
        "cpu",
    ) == (model, tokenizer)


def test_target_load_separates_an_absent_identity_key_from_a_null_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: comparing the two metadata reads with a bare ``dict.get``, whose
    # ``None`` default makes "the setup never carried this block" and "the
    # reload carries it, set to null" compare equal. The canonical readers still
    # reject the null block afterwards, so only the drift message -- naming the
    # key -- proves the presence sentinel is the check that fired.
    setup = _fake_setup(
        meta={"step": 7},
        target_meta_path=Path("checkpoint/meta_000007.json"),
        feature_contract=CheckpointFeatureContract(None, None),
        feature_vocab=None,
    )
    monkeypatch.setattr(
        methods,
        "build_encdec_model",
        lambda *_args, **_kwargs: (
            SimpleNamespace(feature_vocab=None),
            object(),
            {"step": 7, "selfcond_data": None},
        ),
    )
    monkeypatch.setattr(
        methods,
        "resolve_and_assert_model_feature_contract",
        lambda *_args, **_kwargs: pytest.fail("a null identity block reached contract loading"),
    )

    with pytest.raises(
        ValueError,
        match=r"turn-segmentation metadata differs from the prevalidated "
        r"AxBench setup: \['selfcond_data'\]",
    ):
        methods.load_target_model(setup, "cpu")


def test_target_load_rejects_a_float_cadence_that_compares_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: comparing the reloaded values with raw `!=`: 37.0 == 37, so the
    # drift list would stay empty; the canonical-JSON comparison keeps the
    # float distinct from the integer the setup was validated with.
    setup, _model, _tokenizer = _turn_target_load(
        monkeypatch,
        loaded_meta={
            "step": 7,
            **_TURN_IDENTITY_META,
            "turn_self_cond_segment_size_rustbpe": 37.0,
        },
        feature_contract=lambda *_args, **_kwargs: pytest.fail(
            "a float cadence reached feature-contract loading"
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"differs from the prevalidated AxBench setup: \['turn_self_cond_segment_size_rustbpe'\]",
    ):
        methods.load_target_model(setup, "cpu")


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(0.0, 0.0), (1.0, 1.0 / 6.0), (3.0, 0.5), (6.0, 1.0)],
)
def test_strength_maps_to_unit_interval(factor: float, expected: float) -> None:
    assert float(methods.self_conditioning_weight(factor)) == expected


@pytest.mark.parametrize("factor", [-0.1, 0.3, 6.1, float("nan"), float("inf")])
def test_strength_rejects_values_outside_declared_grid(factor: float) -> None:
    with pytest.raises(ValueError, match="declared.*factor grid"):
        methods.self_conditioning_weight(factor)


class _TargetModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = t.nn.Parameter(t.zeros(()))
        self.feature_pred_config = SimpleNamespace(loss_mode=FeaturePredLossMode.MAX_POOL)
        self.dec_config = SimpleNamespace(sequence_len=512)
        self.encoder = SimpleNamespace(config=SimpleNamespace(num_buckets=2, n_sae_features=4))


def test_self_conditioning_config_uses_sequence_length_and_dense_target() -> None:
    model = _TargetModel()
    setup = _setup()
    config = methods.build_self_conditioning_config(
        setup,
        model,
        SelfCondScoreInterpolation(
            target_scores=t.tensor([0.3, 0.2, 0.1, 0.0]),
            negative_selection=setup.output_negative_selection,
            weight=methods.self_conditioning_weight(3.0),
        ),
        selection_seed=123,
    )

    assert int(config.self_cond_segment_size) == 512
    assert config.pool_mode is FeaturePredLossMode.MAX_POOL
    assert config.fp_logit_bias is None
    assert isinstance(config.score_policy, SelfCondScoreInterpolation)
    assert float(config.score_policy.weight) == 0.5
    assert t.equal(
        config.score_policy.target_scores,
        t.tensor([0.3, 0.2, 0.1, 0.0]),
    )
    assert config.score_policy.negative_selection is setup.output_negative_selection
    assert isinstance(config.selection.selection, SamplingSelection)
    assert config.selection.selection.delta_weights is None


def test_turn_self_conditioning_config_uses_checkpoint_rustbpe_cadence() -> None:
    model = _TargetModel()
    setup = _setup()
    setup.segment_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    setup.turn_self_cond_segment_size_rustbpe = 37

    config = methods.build_self_conditioning_config(
        setup,
        model,
        None,
        selection_seed=123,
    )

    assert config.segment_mode is methods.SegmentMode.turn_delimited_fixed_tokens
    assert int(config.self_cond_segment_size) == 37
    assert int(config.self_cond_segment_size) != model.dec_config.sequence_len


def test_turn_self_conditioning_config_rejects_a_missing_cadence() -> None:
    # kills: dropping the ``cadence is None`` narrowing so turn mode silently
    # falls through to ``model.dec_config.sequence_len``.
    model = _TargetModel()
    setup = _setup()
    setup.segment_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    setup.turn_self_cond_segment_size_rustbpe = None

    with pytest.raises(ValueError, match="cadence resolved from checkpoint"):
        methods.build_self_conditioning_config(
            setup,
            model,
            None,
            selection_seed=123,
        )


@pytest.mark.parametrize("cadence", [True, 0, -1, 1.5, "37"])
def test_turn_self_conditioning_config_rejects_a_non_positive_int_cadence(
    cadence: object,
) -> None:
    # kills: widening ``PositiveInt.parse`` on the turn cadence to a plain int
    # cast, which would accept ``True``/``0``/``-1`` as a refresh window.
    model = _TargetModel()
    setup = _setup()
    setup.segment_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    setup.turn_self_cond_segment_size_rustbpe = cadence

    with pytest.raises(TypeError, match=r"Could not parse .*PositiveInt"):
        methods.build_self_conditioning_config(
            setup,
            model,
            None,
            selection_seed=123,
        )


def test_self_conditioning_config_accepts_pure_policy_with_identical_other_fields() -> None:
    model = _TargetModel()
    setup = _setup()
    interpolation = SelfCondScoreInterpolation(
        target_scores=t.tensor([0.3, 0.2, 0.1, 0.0]),
        negative_selection=setup.output_negative_selection,
        weight=methods.self_conditioning_weight(3.0),
    )

    pure = methods.build_self_conditioning_config(
        setup,
        model,
        None,
        selection_seed=123,
    )
    steered = methods.build_self_conditioning_config(
        setup,
        model,
        interpolation,
        selection_seed=123,
    )

    assert pure.score_policy is None
    for config_field in fields(pure):
        if config_field.name == "score_policy":
            continue
        pure_value = getattr(pure, config_field.name)
        steered_value = getattr(steered, config_field.name)
        if config_field.name != "selection":
            if isinstance(pure_value, t.Tensor):
                assert t.equal(pure_value, steered_value)
            else:
                assert pure_value == steered_value
            continue
        for selection_field in fields(pure_value):
            pure_selection_value = getattr(pure_value, selection_field.name)
            steered_selection_value = getattr(steered_value, selection_field.name)
            if selection_field.name == "rng":
                assert pure_selection_value.initial_seed() == steered_selection_value.initial_seed()
                assert bytes(pure_selection_value.get_state().tolist()) == bytes(
                    steered_selection_value.get_state().tolist()
                )
            elif isinstance(pure_selection_value, t.Tensor):
                assert t.equal(pure_selection_value, steered_selection_value)
            else:
                assert pure_selection_value == steered_selection_value


def test_prepared_prompt_rerender_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = SimpleNamespace(get_bos_token_id=lambda: 1)
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2, 3),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    monkeypatch.setattr(
        methods,
        "render_prompt_for_chat_completion",
        lambda *_args: SimpleNamespace(token_ids=[1, 2, 3], assistant_end_id=9),
    )

    methods.assert_prepared_prompt_matches(tokenizer, "prompt", prepared)


@pytest.mark.parametrize(
    ("token_ids", "assistant_end_id", "bos_id", "error"),
    [
        ([1, 2, 4], 9, 1, "token tuple"),
        ([1, 2, 3], 8, 1, "assistant-end"),
        ([1, 2, 3], 9, 7, "nanochat BOS"),
    ],
)
def test_prepared_prompt_rerender_contract_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    token_ids: list[int],
    assistant_end_id: int,
    bos_id: int,
    error: str,
) -> None:
    tokenizer = SimpleNamespace(get_bos_token_id=lambda: bos_id)
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2, 3),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    monkeypatch.setattr(
        methods,
        "render_prompt_for_chat_completion",
        lambda *_args: SimpleNamespace(
            token_ids=token_ids,
            assistant_end_id=assistant_end_id,
        ),
    )

    with pytest.raises(ValueError, match=error):
        methods.assert_prepared_prompt_matches(tokenizer, "prompt", prepared)


def _policy(kind: str = "interpolation") -> Any:
    setup = _setup()
    diffmean = t.tensor([0.3, 0.2, -0.1, -0.2])
    if kind == "interpolation":
        return SelfCondScoreInterpolation(
            target_scores=diffmean,
            negative_selection=setup.output_negative_selection,
            weight=methods.self_conditioning_weight(3.0),
        )
    return methods.build_self_conditioning_masked_policy(setup, diffmean, 3.0)


def _invoke_generation(
    prepared: methods.PreparedPromptContext,
    max_new_tokens: Any,
    *,
    batched: bool,
) -> Any:
    if batched:
        requests = (
            methods.SelfConditioningGenerationRequest(_policy(), 1, 2),
            methods.SelfConditioningGenerationRequest(_policy("masking"), 3, 4),
        )
        return methods.generate_self_conditioning_batch(
            _setup(),
            _TargetModel(),
            object(),
            prepared,
            requests,
            max_new_tokens=max_new_tokens,
        )
    return methods.generate_self_conditioning(
        _setup(),
        _TargetModel(),
        object(),
        prepared,
        _policy(),
        selection_seed=1,
        seed=2,
        max_new_tokens=max_new_tokens,
    )


@pytest.mark.parametrize("kind", ["interpolation", "masking"])
def test_generate_reuses_exact_prepared_tokens_and_stops(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    captured: dict[str, Any] = {}
    profile_regions: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def record_region(name: str, **attrs: Any):
        profile_regions.append((name, attrs))
        yield

    class Engine:
        def generate_stream(self, token_ids: list[int], **kwargs: Any):
            captured["tokens"] = token_ids
            captured.update(kwargs)
            return iter((7, 8))

    monkeypatch.setattr(
        methods, "build_self_conditioning_config", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        methods, "build_engine_self_conditioned", lambda *_args, **_kwargs: Engine()
    )
    monkeypatch.setattr(methods, "profile_region", record_region)
    tokenizer = SimpleNamespace(
        get_bos_token_id=lambda: 1, decode=lambda values: f"decoded:{values}"
    )
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2, 3),
        assistant_end_id=9,
        segments=(ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=1),),
    )
    model = _TargetModel()
    model.dec_config.sequence_len = 19

    result = methods.generate_self_conditioning(
        _setup(),
        model,
        tokenizer,
        prepared,
        _policy(kind),
        selection_seed=11,
        seed=12,
        max_new_tokens=16,
    )

    assert result == "decoded:[7, 8]"
    assert captured["tokens"] == [1, 2, 3]
    assert captured["stop_token_ids"] == {1, 9}
    assert captured["temperature"] == 1.0
    assert captured["top_k"] == 50
    assert profile_regions == [
        (
            "axbench_upstream.generation.engine_construction",
            {"driver": "project", "prompt_tokens": 3},
        ),
        (
            "axbench_upstream.generation.stream",
            {"driver": "project", "prompt_tokens": 3, "max_new_tokens": 16},
        ),
        (
            "axbench_upstream.generation.tokenizer_decode",
            {"driver": "project", "generated_tokens": 2},
        ),
    ]


def test_generate_batch_freezes_independent_policy_states_and_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TargetModel()
    model.dec_config.sequence_len = 1024
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2, 3),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    policies = (_policy("interpolation"), _policy("masking"))
    requests = tuple(
        methods.SelfConditioningGenerationRequest(policy, selection_seed, token_seed)
        for policy, selection_seed, token_seed in zip(
            policies,
            (11, 29),
            (101, 103),
            strict=True,
        )
    )
    configs: list[Any] = []
    prefills: list[tuple[int, int, int]] = []
    batch_kwargs: dict[str, Any] = {}

    class Engine:
        def __init__(self, row: int, config: Any) -> None:
            self.row = row
            self.config = config

        def prepare_self_conditioned_prefill(
            self,
            tokens: list[int],
            *,
            max_new_tokens: int,
        ) -> tuple[int, int, int]:
            assert tokens == list(prepared.token_ids)
            assert max_new_tokens == 128
            assert int(self.config.self_cond_segment_size) == 1024
            rng_value = int(t.randint(0, 1 << 20, (), generator=self.config.selection.rng))
            snapshot = (self.row, id(self.config.score_policy), rng_value)
            prefills.append(snapshot)
            return snapshot

        def generate_self_conditioned_batch(
            self,
            frozen: list[tuple[int, int, int]],
            **kwargs: Any,
        ) -> list[list[int]]:
            assert self.row == 1
            assert frozen == prefills
            batch_kwargs.update(kwargs)
            return [[], [7, 8]]

    def build_engine(
        builder_model: Any,
        _tokenizer: Any,
        device: t.device,
        config: Any,
        *,
        context_provider: Any,
    ) -> Engine:
        assert builder_model is model
        assert device == model.anchor.device
        assert context_provider.get(prepared.token_ids) == prepared.segments
        configs.append(config)
        return Engine(len(configs) - 1, config)

    monkeypatch.setattr(methods, "build_engine_self_conditioned", build_engine)
    tokenizer = SimpleNamespace(
        get_bos_token_id=lambda: 1,
        decode=lambda values: f"decoded:{values}",
    )

    actual = methods.generate_self_conditioning_batch(
        _setup(),
        model,
        tokenizer,
        prepared,
        requests,
        max_new_tokens=128,
    )

    expected_rng_values = []
    for seed in (11, 29):
        rng = t.Generator(device="cpu")
        rng.manual_seed(seed)
        expected_rng_values.append(int(t.randint(0, 1 << 20, (), generator=rng)))
    assert actual == ("decoded:[]", "decoded:[7, 8]")
    assert len({id(config) for config in configs}) == 2
    assert len({id(config.selection.rng) for config in configs}) == 2
    assert [config.score_policy for config in configs] == list(policies)
    assert [snapshot[2] for snapshot in prefills] == expected_rng_values
    assert batch_kwargs == {
        "seeds": [101, 103],
        "max_new_tokens": 128,
        "temperature": 1.0,
        "top_k": 50,
        "stop_token_ids": {1, 9},
    }


def test_generate_batch_rejects_turn_mode_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    setup.segment_mode = methods.SegmentMode.turn_delimited_fixed_tokens
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2, 3),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    requests = (
        methods.SelfConditioningGenerationRequest(_policy(), 11, 101),
        methods.SelfConditioningGenerationRequest(_policy("masking"), 29, 103),
    )
    monkeypatch.setattr(
        methods,
        "build_engine_self_conditioned",
        lambda *_args, **_kwargs: pytest.fail("turn mode constructed a batch engine"),
    )

    with pytest.raises(
        ValueError,
        match="supports only fixed_tokens.*requires the scalar cell dispatcher",
    ):
        methods.generate_self_conditioning_batch(
            setup,
            _TargetModel(),
            object(),
            prepared,
            requests,
            max_new_tokens=16,
        )


@pytest.mark.parametrize(
    "policy_order",
    [
        ("interpolation", "masking"),
        ("masking", "interpolation"),
    ],
)
def test_ranked_policy_batch_exactly_matches_scalar_and_preserves_prefills(
    policy_order: tuple[str, str],
    mock_tokenizer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production ranked policies through the real tiny model."""

    n_sae_features = 16
    t.manual_seed(20260830)
    model = build_tiny_model(
        n_sae_features=n_sae_features,
        num_buckets=2,
        seq_len=64,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
        ),
    )
    model.eval()
    setup = _setup(n_sae_features=n_sae_features)
    source_scores = t.linspace(-0.8, 0.8, n_sae_features)
    expected_scores = source_scores.clone()
    factors = (0.2, 1.5, 3.0, 6.0)
    kinds = policy_order * 2
    policies = tuple(
        (
            SelfCondScoreInterpolation(
                target_scores=source_scores,
                negative_selection=setup.output_negative_selection,
                weight=methods.self_conditioning_weight(factor),
            )
            if kind == "interpolation"
            else methods.build_self_conditioning_masked_policy(
                setup,
                source_scores,
                factor,
            )
        )
        for kind, factor in zip(kinds, factors, strict=True)
    )
    assert methods.SELF_CONDITIONING_MASKED_POLICY.endswith("sign_rank_v3")
    source_scores.zero_()
    for policy in policies:
        assert t.equal(policy.target_scores, expected_scores)

    prompt_tokens = (mock_tokenizer.get_bos_token_id(), 2, 3)
    char_end = token_to_char_offsets(mock_tokenizer, list(prompt_tokens))[-1]
    prepared = methods.PreparedPromptContext(
        token_ids=prompt_tokens,
        assistant_end_id=1,
        segments=(ContextFeatureSegment((1,), (0,), 0, int(char_end)),),
    )
    selection_seeds = (11, 11, 29, 29)
    token_seeds = (101, 103, 107, 109)
    requests = tuple(
        methods.SelfConditioningGenerationRequest(policy, selection_seed, token_seed)
        for policy, selection_seed, token_seed in zip(
            policies,
            selection_seeds,
            token_seeds,
            strict=True,
        )
    )

    original_batch = InferenceEngine.generate_self_conditioned_batch
    checked_prefill_counts: list[int] = []

    def snapshot(prefill: Any) -> tuple[t.Tensor, ...]:
        encoder_out = prefill.runtime.state.encoder_out
        return (
            prefill.logits.clone(),
            prefill.kv_cache.k_cache.clone(),
            prefill.kv_cache.v_cache.clone(),
            encoder_out.values.clone(),
            encoder_out.layout.flat_indices.clone(),
            encoder_out.layout.valid_mask.clone(),
            encoder_out.layout.batch_indices.clone(),
            encoder_out.layout.segment_indices.clone(),
        )

    def checked_batch(
        engine: InferenceEngine,
        prefills: Any,
        **kwargs: Any,
    ) -> list[list[int]]:
        frozen = tuple(prefills)
        pinned = tuple(snapshot(prefill) for prefill in frozen)
        assert all(prefill.runtime.tokens_in_seg == 0 for prefill in frozen)
        assert all(not prefill.runtime.fp_buffer for prefill in frozen)
        result = original_batch(engine, frozen, **kwargs)
        for prefill, expected in zip(frozen, pinned, strict=True):
            for actual_tensor, expected_tensor in zip(
                snapshot(prefill),
                expected,
                strict=True,
            ):
                assert t.equal(actual_tensor, expected_tensor)
            assert prefill.runtime.tokens_in_seg == 0
            assert not prefill.runtime.fp_buffer
        checked_prefill_counts.append(len(frozen))
        return result

    monkeypatch.setattr(
        InferenceEngine,
        "generate_self_conditioned_batch",
        checked_batch,
    )
    expected = tuple(
        methods.generate_self_conditioning(
            setup,
            model,
            mock_tokenizer,
            prepared,
            request.policy,
            selection_seed=request.selection_seed,
            seed=request.token_seed,
            max_new_tokens=4,
        )
        for request in requests
    )
    actual = methods.generate_self_conditioning_batch(
        setup,
        model,
        mock_tokenizer,
        prepared,
        requests,
        max_new_tokens=4,
    )

    assert actual == expected
    assert checked_prefill_counts == [len(requests)]


def test_generate_batch_rejects_singleton_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        methods,
        "build_engine_self_conditioned",
        lambda *_args, **_kwargs: pytest.fail("singleton reached engine construction"),
    )
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    request = methods.SelfConditioningGenerationRequest(_policy(), 1, 2)

    with pytest.raises(ValueError, match="requires at least two rows"):
        methods.generate_self_conditioning_batch(
            _setup(),
            _TargetModel(),
            object(),
            prepared,
            (request,),
            max_new_tokens=128,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_generate_rejects_prompt_plus_budget_over_sequence_length(batched: bool) -> None:
    prepared = methods.PreparedPromptContext(
        token_ids=tuple(range(500)),
        assistant_end_id=9,
        segments=(),
    )
    with pytest.raises(ValueError, match="generation budget exceeds"):
        _invoke_generation(prepared, 13, batched=batched)


@pytest.mark.parametrize(
    ("max_new_tokens", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
@pytest.mark.parametrize("batched", [False, True])
def test_generate_rejects_invalid_generation_budget_domain(
    max_new_tokens: Any,
    error: type[Exception],
    batched: bool,
) -> None:
    prepared = methods.PreparedPromptContext(
        token_ids=(1, 2),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    with pytest.raises(error, match="max_new_tokens"):
        _invoke_generation(prepared, max_new_tokens, batched=batched)


@pytest.mark.parametrize("missing", ["tokens", "segments"])
@pytest.mark.parametrize("batched", [False, True])
def test_generate_rejects_incomplete_prepared_prompt(missing: str, batched: bool) -> None:
    prepared = methods.PreparedPromptContext(
        token_ids=() if missing == "tokens" else (1, 2),
        assistant_end_id=9,
        segments=() if missing == "segments" else (ContextFeatureSegment((1,), (0,), 0, 1),),
    )
    with pytest.raises(ValueError, match="token tuple|real feature segment"):
        _invoke_generation(prepared, 1, batched=batched)


@pytest.mark.parametrize(
    ("strength", "n_candidates", "mean", "expected"),
    [
        (0.0, 7, 2.0, 7),
        (6.0, 7, 2.0, 4),
        (3.0, 4, 0.5, 2),  # 2.5 rounds to even 2
        (3.0, 6, 0.5, 4),  # 3.5 rounds to even 4
        (6.0, 9, 0.2, 1),
        (6.0, 4, 2.0, 4),
        (6.0, 4, 3.0, 4),
        (2.75, 1, 10.0, 1),
    ],
)
def test_masked_pool_limit_matches_independent_arithmetic_oracle(
    strength: float,
    n_candidates: int,
    mean: float,
    expected: int,
) -> None:
    assert (
        methods.self_conditioning_masked_pool_limit(
            strength,
            n_candidates=n_candidates,
            mean=mean,
        )
        == expected
    )


def test_masked_pool_limit_covers_grid_continuous_strengths_and_monotonicity() -> None:
    strengths = (*methods.SELF_CONDITIONING_FACTORS, 0.1, 2.75, 5.5)
    ordered = sorted(set(strengths))
    limits = [
        methods.self_conditioning_masked_pool_limit(
            strength,
            n_candidates=37,
            mean=3.25,
        )
        for strength in ordered
    ]
    expected = [
        min(37, max(1, round(37 * (1 - strength / 6.0) + 6.5 * strength / 6.0)))
        for strength in ordered
    ]
    assert limits == expected
    assert limits == sorted(limits, reverse=True)


@pytest.mark.parametrize(
    ("strength", "n_candidates", "mean", "error"),
    [
        (True, 4, 2.0, TypeError),
        ("1", 4, 2.0, TypeError),
        (float("nan"), 4, 2.0, ValueError),
        (float("inf"), 4, 2.0, ValueError),
        (-0.1, 4, 2.0, ValueError),
        (6.1, 4, 2.0, ValueError),
        (1.0, True, 2.0, TypeError),
        (1.0, 1.5, 2.0, TypeError),
        (1.0, 0, 2.0, ValueError),
        (1.0, -1, 2.0, ValueError),
        (1.0, 4, True, TypeError),
        (1.0, 4, 0.0, ValueError),
        (1.0, 4, -1.0, ValueError),
        (1.0, 4, float("nan"), ValueError),
        (1.0, 4, float("inf"), ValueError),
    ],
)
def test_masked_pool_limit_rejects_invalid_domains(
    strength: Any,
    n_candidates: Any,
    mean: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        methods.self_conditioning_masked_pool_limit(
            strength,
            n_candidates=n_candidates,
            mean=mean,
        )


def test_masked_config_uses_sequence_length_dense_target_and_rank_limits() -> None:
    model = _TargetModel()
    setup = _setup()
    config = methods.build_self_conditioning_config(
        setup,
        model,
        methods.build_self_conditioning_masked_policy(
            setup,
            t.tensor([0.3, 0.2, 0.1, 0.0]),
            3.0,
        ),
        selection_seed=123,
    )

    assert int(config.self_cond_segment_size) == 512
    assert config.pool_mode is FeaturePredLossMode.MAX_POOL
    assert config.fp_logit_bias is None
    assert isinstance(config.score_policy, SelfCondScoreMasking)
    assert int(config.score_policy.positive_pool_limit) == 4
    assert int(config.score_policy.negative_pool_limit) == 4
    assert t.equal(
        config.score_policy.target_scores,
        t.tensor([0.3, 0.2, 0.1, 0.0]),
    )
    assert config.score_policy.negative_selection is setup.output_negative_selection
    assert isinstance(config.selection.selection, SamplingSelection)
    assert config.selection.selection.delta_weights is None


def test_masked_policy_uses_compact_post_mask_n_and_independent_side_means() -> None:
    setup = _setup(n_sae_features=10, allowed_ids=t.tensor([1, 3, 5, 7]))
    setup.feature_mask = FeatureMask("blacklist", (1,))
    setup.output_selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=0.5,
        std=0.5,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    setup.output_negative_selection = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=1.0,
        std=1.0,
        min_k=0,
        max_k=4,
    )

    policy = methods.build_self_conditioning_masked_policy(
        setup,
        t.tensor([0.9, 0.5, -0.5, -0.9]),
        6.0,
    )

    assert int(policy.positive_pool_limit) == 1
    assert int(policy.negative_pool_limit) == 2


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (
            lambda model: methods.build_self_conditioning_config(
                _setup(),
                model,
                _policy("interpolation"),
                selection_seed=1,
            ),
            "requires checkpoint feature_prediction metadata",
        ),
        (
            lambda model: methods.build_self_conditioning_config(
                _setup(),
                model,
                _policy("masking"),
                selection_seed=1,
            ),
            "requires checkpoint feature_prediction metadata",
        ),
    ],
)
def test_builder_requires_feature_prediction_metadata(
    builder: Any,
    match: str,
) -> None:
    model = _TargetModel()
    model.feature_pred_config = None  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match=match):
        builder(model)


def test_builder_requires_a_max_pool_feature_prediction_head() -> None:
    model = _TargetModel()
    model.feature_pred_config = SimpleNamespace(loss_mode=FeaturePredLossMode.LAST_TOKEN)
    with pytest.raises(ValueError, match="requires a MAX_POOL feature-prediction head"):
        methods.build_self_conditioning_config(
            _setup(),
            model,
            _policy("interpolation"),
            selection_seed=1,
        )


def _preflight_setup(
    *,
    positive_min_k: int = 1,
    negative_min_k: int = 0,
    feature_mask: Any = None,
    allowed_ids: Any = None,
    n_sae_features: int = 8,
) -> Any:
    return SimpleNamespace(
        n_sae_features=n_sae_features,
        feature_mask=feature_mask,
        allowed_ids=allowed_ids,
        output_selection=SimpleNamespace(min_k=positive_min_k),
        output_negative_selection=SimpleNamespace(min_k=negative_min_k),
    )


def test_masked_preflight_accepts_a_two_sided_diffmean() -> None:
    methods.assert_masked_diffmean_feasible(
        _preflight_setup(),
        t.linspace(0.8, -0.8, 8),
        concept_id=0,
    )


def test_masked_preflight_rejects_an_all_negative_diffmean_naming_the_concept() -> None:
    with pytest.raises(
        ValueError,
        match=r"concept_id=3.*empty positive candidate pool",
    ):
        methods.assert_masked_diffmean_feasible(
            _preflight_setup(),
            t.linspace(-0.1, -0.8, 8),
            concept_id=3,
        )


def test_masked_preflight_accepts_weak_magnitude_two_sided_diffmean() -> None:
    diffmean = t.tensor([0.02, 0.01, -0.5, -0.4, -0.3, -0.2, -0.1, -0.6])
    methods.assert_masked_diffmean_feasible(_preflight_setup(), diffmean, concept_id=0)


def test_masked_preflight_applies_the_feature_mask_to_the_pools() -> None:
    diffmean = t.tensor([0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = FeatureMask("blacklist", (0,))
    with pytest.raises(ValueError, match="empty positive candidate pool"):
        methods.assert_masked_diffmean_feasible(
            _preflight_setup(feature_mask=mask),
            diffmean,
            concept_id=0,
        )


def test_masked_preflight_rejects_a_mask_with_no_eligible_head_rows() -> None:
    with pytest.raises(ValueError, match="no eligible head rows"):
        methods.assert_masked_diffmean_feasible(
            _preflight_setup(feature_mask=FeatureMask("whitelist", (99,))),
            t.linspace(0.8, -0.8, 8),
            concept_id=0,
        )


def test_masked_preflight_rejects_id_width_mismatch() -> None:
    mask = FeatureMask("blacklist", (0,))
    with pytest.raises(ValueError, match="widths disagree"):
        methods.assert_masked_diffmean_feasible(
            _preflight_setup(feature_mask=mask, allowed_ids=t.arange(4)),
            t.linspace(0.8, -0.8, 8),
            concept_id=0,
        )


def test_boolean_factor_is_rejected_by_grid_policy_builder() -> None:
    with pytest.raises(ValueError, match="declared.*factor grid"):
        methods.build_self_conditioning_masked_policy(
            _setup(),
            t.tensor([0.3, 0.2, -0.1, -0.2]),
            True,
        )


def test_declared_factor_grid_matches_the_frame_factor_grid() -> None:
    # run_methods requires every main-frame factor to be a member of
    # SELF_CONDITIONING_FACTORS while the frame itself is built on
    # merge_methods.FULL_FACTORS; the two hand-typed twin literals drifting
    # apart would only surface as a mid-run rejection.
    assert methods.SELF_CONDITIONING_FACTORS == FULL_FACTORS


@pytest.mark.parametrize("factor", ["3.0", None, [3.0]])
def test_non_numeric_masked_factors_raise_value_error_not_coerce(factor: Any) -> None:
    # float("3.0") would silently succeed and garbage would raise TypeError if
    # the coercion ran before the type guard; both mappers must reject with
    # the contract's ValueError instead.
    with pytest.raises(ValueError, match="declared.*factor grid"):
        methods.build_self_conditioning_masked_policy(
            _setup(),
            t.tensor([0.3, 0.2, -0.1, -0.2]),
            factor,
        )
