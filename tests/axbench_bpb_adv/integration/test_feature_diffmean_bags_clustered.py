"""Cluster-aware feature-DiffMean bag derives and warm-cache regressions.

Known roofline-cache limitation: ``core_prefix_id`` binds only concept, mode,
and prefix text, not the completion snapshot used to derive the bag. The
current G6b orchestration is safe because cache construction and consumption
share one in-memory prefix collection. A direct caller that regenerates
completion content under unchanged prefix text must not reuse the old cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.feature_diffmean_bags as diffmean_bags
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.feature_diffmean import PrefixedCompletion
from steering.eval.axbench_bpb_adv.feature_diffmean_bags import (
    build_feature_diffmean_bags_fair,
    build_feature_diffmean_bags_roofline,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    ClusteredWarmStartCache,
    build_clustered_warm_start_cache,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    read_ours_gcg_bag,
    read_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
)
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContract,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    build_cluster_feature_contract,
)

_N_SAE_FEATURES = 8
_MASKED_ID = 7


@dataclass(frozen=True)
class _ContractFixture:
    contract: ClusterFeatureContract
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _DiffMeanInputs:
    base_model: Any
    sae: Any
    percentile_table: t.Tensor
    cfg: AxBenchBpbAdvConfig


@dataclass(frozen=True)
class _DiffMeanCall:
    rich_items: tuple[PrefixedCompletion, ...]
    avoid_items: tuple[PrefixedCompletion, ...]
    kwargs: dict[str, Any]


@dataclass
class _StubObservations:
    member_contracts: list[ClusterFeatureContract | None]
    diffmean_calls: list[_DiffMeanCall]


class _ContractModel:
    def __init__(
        self,
        fixture: _ContractFixture,
        *,
        feature_vocab: FeatureVocab | None = None,
    ) -> None:
        pinned_vocab = fixture.contract.checkpoint.feature_vocab
        assert pinned_vocab is not None
        self.feature_vocab = pinned_vocab if feature_vocab is None else feature_vocab
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=fixture.contract.n_sae_features,
                num_buckets=fixture.contract.selection.num_buckets,
            ),
            feature_vocab_ids=self.feature_vocab.new_to_old_tensor(),
        )
        self._loaded_meta_data = fixture.metadata

    def assert_vocab_consistent(self) -> None:
        assert self.feature_vocab is not None
        assert tuple(self.encoder.feature_vocab_ids.tolist()) == (
            self.feature_vocab.original_ids
        )


class _UnclusteredModel:
    feature_vocab = None

    def __init__(self) -> None:
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=_N_SAE_FEATURES,
                num_buckets=2,
            )
        )
        self._loaded_meta_data = {"user_config": {}}


class _Tokenizer:
    @staticmethod
    def decode(tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


def _contract_fixture(*, preselection_seed: int = 17) -> _ContractFixture:
    mask = FeatureMask(FeatureMaskMode.blacklist, (_MASKED_ID,))
    definition = FeatureClusterDefinition.from_mapping(
        {"schema_version": 1, "clusters": [[0, 1]]},
        n_sae_features=_N_SAE_FEATURES,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/g4-bags-clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=_N_SAE_FEATURES,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    feature_vocab = effective_feature_vocab(
        consolidation,
        _N_SAE_FEATURES,
    )
    assert feature_vocab is not None
    checkpoint_dir = Path("/test/g4-bags-checkpoint")
    checkpoint = CheckpointPin(
        checkpoint_dir=checkpoint_dir,
        resolved_tag="g4-bags",
        step=7,
        model_file=StableFileDigest(
            path=checkpoint_dir / "model_000007.pt",
            sha256="1" * 64,
            byte_length=123,
            stat_signature=(0o100644, 1, 2, 123, 4, 5),
        ),
        metadata_file=StableFileDigest(
            path=checkpoint_dir / "meta_000007.json",
            sha256="2" * 64,
            byte_length=45,
            stat_signature=(0o100644, 1, 3, 45, 4, 5),
        ),
        n_sae_features=_N_SAE_FEATURES,
        feature_vocab=feature_vocab,
        load_bearing_metadata_json="{}",
    )
    selection = CheckpointSelectionContract(
        positive_selection_json=json.dumps(
            {"distribution": "geometric", "mean": 1.0},
            sort_keys=True,
            separators=(",", ":"),
        ),
        negative_selection_json=json.dumps(
            {"distribution": "geometric", "mean": 1.0},
            sort_keys=True,
            separators=(",", ":"),
        ),
        num_buckets=2,
        effective_positive_buckets=1,
        preselection_seed=preselection_seed,
        selection_hash="3" * 64,
        selection_manifest_sha256="4" * 64,
        selection_shards_sha256="5" * 64,
        packed_row_fingerprint="6" * 64,
    )
    contract = build_cluster_feature_contract(
        feature_mask=mask,
        feature_clusters=resolved,
        statistics=FeatureStatisticsIdentity(
            file_sha256="7" * 64,
            percentile_table_checksum="8" * 64,
            n_sae_features=_N_SAE_FEATURES,
            feature_id_domain=tuple(range(_N_SAE_FEATURES)),
        ),
        feature_directory=FeatureDirectoryIdentity(
            segment_size=128,
            layer=1,
            activation_type="residual",
            pooling="max_pool",
            width="unit",
            model="unit",
            dataset_phase="base",
            context_independent_features=False,
            ignore_bos_features=True,
        ),
        sae=SaeIdentity(
            release="unit",
            sae_id="unit",
            hook_name="blocks.1.hook_resid_post",
            model="unit",
            layer=1,
            activation_type="residual",
            width="unit",
            actual_loaded_width=_N_SAE_FEATURES,
        ),
        checkpoint=checkpoint,
        selection=selection,
        concept_source_mode="statistics_domain",
        concept_source_domain=tuple(range(_N_SAE_FEATURES)),
    )
    metadata = {
        "feature_mask_canonical": mask.canonical_dict(),
        **feature_clusters_checkpoint_metadata(resolved),
        "feature_vocab": feature_vocab.metadata_dict(),
        "user_config": {},
    }
    return _ContractFixture(contract=contract, metadata=metadata)


def _cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="src",
        output_root="out",
        feature_dir="features",
        feature_statistics_path="stats.safetensors",
        ours_gcg=OursGcgConfig(
            split_ratio=0.5,
            split_seed=11,
            top_k_pos=1,
            top_k_neg=2,
        ),
    )


def _diffmean_inputs() -> _DiffMeanInputs:
    return _DiffMeanInputs(
        base_model=object(),
        sae=object(),
        percentile_table=t.zeros(
            (_N_SAE_FEATURES, 99),
            dtype=t.float64,
        ),
        cfg=_cfg(),
    )


def _prefix(
    idx: int,
    *,
    concept_id: int,
    mode: SteeringMode,
) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id=f"c{concept_id}-{mode.value}-p{idx}",
        concept_id=concept_id,
        mode=mode,
        prefix_text=f"prefix {concept_id} {mode.value} {idx}",
        prefix_hash=f"h-{concept_id}-{mode.value}-{idx}",
        prefix_tokens=[idx],
        concept_rich_sample_ids=[f"r-{concept_id}-{mode.value}-{idx}"],
        concept_avoiding_sample_ids=[f"a-{concept_id}-{mode.value}-{idx}"],
        concept_rich_output_tokens=[[10 + idx]],
        concept_avoiding_output_tokens=[[20 + idx]],
    )


def _union_prefixes() -> list[BpbAttackPrefix]:
    return [
        *(
            _prefix(idx, concept_id=7, mode=SteeringMode.POSITIVE)
            for idx in range(4)
        ),
        *(
            _prefix(idx, concept_id=8, mode=SteeringMode.NEGATIVE)
            for idx in range(4)
        ),
    ]


def _fake_diffmean(
    rich_items: list[PrefixedCompletion],
    avoid_items: list[PrefixedCompletion],
    *,
    observations: _StubObservations,
    **kwargs: Any,
) -> FeatureConditioning:
    observations.diffmean_calls.append(
        _DiffMeanCall(
            rich_items=tuple(rich_items),
            avoid_items=tuple(avoid_items),
            kwargs=dict(kwargs),
        )
    )
    cluster_contract = kwargs["cluster_contract"]
    if cluster_contract is None:
        raise AssertionError("clustered test reached DiffMean without its contract")
    assert rich_items
    assert avoid_items
    mode = SteeringMode(kwargs["mode"])
    positive_bucket = int(kwargs["positive_bucket"])
    negative_bucket = int(kwargs["negative_bucket"])
    representatives = cluster_contract.representative_universe
    marker = sum(
        ord(char)
        for item in (*rich_items, *avoid_items)
        for char in f"{item.prefix_text}|{item.completion_text}"
    )
    first = representatives[marker % len(representatives)]
    second = representatives[(marker + 1) % len(representatives)]
    if second == first:
        second = representatives[(marker + 2) % len(representatives)]
    buckets = (
        (positive_bucket, negative_bucket)
        if mode is SteeringMode.POSITIVE
        else (negative_bucket, positive_bucket)
    )
    return FeatureConditioning(((first, buckets[0]), (second, buckets[1])))


def _install_clustered_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: ClusterFeatureContract,
) -> _StubObservations:
    observations = _StubObservations(member_contracts=[], diffmean_calls=[])

    def _members(
        prefixes: list[BpbAttackPrefix],
        *,
        cluster_contract: ClusterFeatureContract | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        observations.member_contracts.append(cluster_contract)
        assert cluster_contract is contract
        return [f"member:{prefix.core_prefix_id}" for prefix in prefixes]

    def _group_g_steer(*_args: Any, **_kwargs: Any) -> Any:
        def _measure(members: list[str], bag: FeatureConditioning) -> float:
            return float(len(members) + len(bag))

        return _measure

    for target in (gcg_bags, diffmean_bags):
        monkeypatch.setattr(target, "_sae_layer_for_cfg", lambda _cfg: 1)
        monkeypatch.setattr(target, "encdec_dec_sequence_len", lambda _model: 64)
        monkeypatch.setattr(target, "_members_for_prefixes", _members)
        monkeypatch.setattr(
            target,
            "feature_diffmean_bag",
            partial(_fake_diffmean, observations=observations),
        )
        monkeypatch.setattr(
            target,
            "_assert_prefix_tokens_roundtrip",
            lambda *_args, **_kwargs: None,
        )
    monkeypatch.setattr(diffmean_bags, "_group_g_steer", _group_g_steer)
    return observations


def _build_cache(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fixture: _ContractFixture,
    model: _ContractModel,
    prefixes: list[BpbAttackPrefix],
    inputs: _DiffMeanInputs | None = None,
) -> tuple[ClusteredWarmStartCache, _StubObservations]:
    resolved_inputs = _diffmean_inputs() if inputs is None else inputs
    observed = _install_clustered_stubs(
        monkeypatch,
        contract=fixture.contract,
    )
    cache = build_clustered_warm_start_cache(
        prefixes=prefixes,
        model=model,
        base_model=resolved_inputs.base_model,
        sae=resolved_inputs.sae,
        percentile_table=resolved_inputs.percentile_table,
        cfg=resolved_inputs.cfg,
        device="cpu",
        tokenizer=_Tokenizer(),
        context_segments_by_prefix_hash={},
        fair_modes={SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
        roofline_modes={SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
        cluster_contract=fixture.contract,
    )
    return cache, observed


def _derive_kwargs(
    *,
    prefixes: list[BpbAttackPrefix],
    model: Any,
    paths: AxBenchBpbAdvPaths,
    inputs: _DiffMeanInputs | None = None,
) -> dict[str, Any]:
    resolved_inputs = _diffmean_inputs() if inputs is None else inputs
    return {
        "prefixes": prefixes,
        "model": model,
        "base_model": resolved_inputs.base_model,
        "sae": resolved_inputs.sae,
        "percentile_table": resolved_inputs.percentile_table,
        "cfg": resolved_inputs.cfg,
        "paths": paths,
        "device": "cpu",
        "tokenizer": _Tokenizer(),
        "context_segments_by_prefix_hash": {},
    }


_REQUIRED_DIFFMEAN_KWARGS = {
    "sae",
    "model",
    "sae_layer",
    "percentile_table",
    "mode",
    "k_pos",
    "k_neg",
    "positive_bucket",
    "negative_bucket",
    "device",
    "extraction_batch_size",
    "allowed_ids",
    "cluster_contract",
}


def _assert_kwarg_value_equal(
    *,
    name: str,
    cache_value: Any,
    inline_value: Any,
) -> None:
    if isinstance(cache_value, t.Tensor) or isinstance(inline_value, t.Tensor):
        assert isinstance(cache_value, t.Tensor), name
        assert isinstance(inline_value, t.Tensor), name
        assert cache_value.dtype == inline_value.dtype, name
        assert cache_value.shape == inline_value.shape, name
        assert t.equal(cache_value.cpu(), inline_value.cpu()), name
        return
    assert cache_value == inline_value, name


def _assert_inline_cache_call_parity(
    *,
    cache_calls: tuple[_DiffMeanCall, ...],
    inline_calls: list[_DiffMeanCall],
) -> None:
    assert inline_calls
    for inline_call in inline_calls:
        matches = [
            cache_call
            for cache_call in cache_calls
            if cache_call.rich_items == inline_call.rich_items
            and cache_call.avoid_items == inline_call.avoid_items
            and cache_call.kwargs["mode"] == inline_call.kwargs["mode"]
        ]
        assert len(matches) == 1
        cache_call = matches[0]
        assert set(cache_call.kwargs) == _REQUIRED_DIFFMEAN_KWARGS
        assert set(inline_call.kwargs) == _REQUIRED_DIFFMEAN_KWARGS
        for name in sorted(_REQUIRED_DIFFMEAN_KWARGS):
            _assert_kwarg_value_equal(
                name=name,
                cache_value=cache_call.kwargs[name],
                inline_value=inline_call.kwargs[name],
            )


def test_clustered_cache_equals_inline_for_filtered_modes_and_stamps_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """T2.1: union-cache bags exactly match mode-filtered inline derives."""
    fixture = _contract_fixture()
    model = _ContractModel(fixture)
    prefixes = _union_prefixes()
    inputs = _diffmean_inputs()
    cache, observations = _build_cache(
        monkeypatch,
        fixture=fixture,
        model=model,
        prefixes=prefixes,
        inputs=inputs,
    )
    cache_calls = tuple(observations.diffmean_calls)
    assert cache_calls
    positive = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.POSITIVE
    ]
    negative = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.NEGATIVE
    ]

    # Non-vacuity: the cache sees two modes/two groups, while each derive sees a
    # proper mode-filtered subset of that union.
    assert len(prefixes) == 8
    assert len(positive) == len(negative) == 4
    assert {prefix.mode for prefix in prefixes} == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert {
        (prefix.concept_id, prefix.mode) for prefix in prefixes
    } == {
        (7, SteeringMode.POSITIVE),
        (8, SteeringMode.NEGATIVE),
    }
    union_ids = {prefix.core_prefix_id for prefix in prefixes}
    assert {prefix.core_prefix_id for prefix in positive} < union_ids
    assert {prefix.core_prefix_id for prefix in negative} < union_ids

    fair_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="inline-fair")
    n_calls_before_fair = len(observations.diffmean_calls)
    build_feature_diffmean_bags_fair(
        **_derive_kwargs(
            prefixes=positive,
            model=model,
            paths=fair_paths,
            inputs=inputs,
        ),
        cluster_contract=fixture.contract,
    )
    _assert_inline_cache_call_parity(
        cache_calls=cache_calls,
        inline_calls=observations.diffmean_calls[n_calls_before_fair:],
    )
    fair_path = fair_paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
    )
    fair_dict, fair_metadata = read_ours_gcg_bag(
        fair_path,
        expected_feature_contract_fingerprint=(
            fixture.contract.feature_contract_fingerprint
        ),
    )
    inline_fair = FeatureConditioning.from_dict(fair_dict)
    cache_fair = cache.fair_bag(
        7,
        SteeringMode.POSITIVE,
        contract=fixture.contract,
        surface="T2.1 fair",
    )
    assert cache_fair.entries == inline_fair.entries
    assert cache_fair.digest() == inline_fair.digest()
    assert fair_metadata["feature_contract_fingerprint"] == (
        fixture.contract.feature_contract_fingerprint
    )
    fair_scored_path = fair_paths.ours_gcg_scored_prefixes(
        7,
        SteeringMode.POSITIVE,
        PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
    )
    assert read_ours_gcg_scored_prefixes(
        fair_scored_path,
        expected_feature_contract_fingerprint=(
            fixture.contract.feature_contract_fingerprint
        ),
    )

    roofline_paths = AxBenchBpbAdvPaths(
        output_root=tmp_path,
        run_tag="inline-roofline",
    )
    n_calls_before_roofline = len(observations.diffmean_calls)
    build_feature_diffmean_bags_roofline(
        **_derive_kwargs(
            prefixes=negative,
            model=model,
            paths=roofline_paths,
            inputs=inputs,
        ),
        cluster_contract=fixture.contract,
    )
    _assert_inline_cache_call_parity(
        cache_calls=cache_calls,
        inline_calls=observations.diffmean_calls[n_calls_before_roofline:],
    )
    _optimize, held_out = split_prefixes_for_concept(
        negative,
        ratio=_cfg().ours_gcg.split_ratio,
        seed=_cfg().ours_gcg.split_seed,
    )
    assert len(held_out) == 2
    for prefix in held_out:
        bag_path = roofline_paths.ours_gcg_bag(
            8,
            SteeringMode.NEGATIVE,
            0.0,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            prefix_id=prefix.core_prefix_id,
        )
        bag_dict, metadata = read_ours_gcg_bag(
            bag_path,
            expected_feature_contract_fingerprint=(
                fixture.contract.feature_contract_fingerprint
            ),
        )
        inline_bag = FeatureConditioning.from_dict(bag_dict)
        cached_bag = cache.roofline_bag(
            8,
            SteeringMode.NEGATIVE,
            str(prefix.core_prefix_id),
            contract=fixture.contract,
            surface="T2.1 roofline",
        )
        assert cached_bag.entries == inline_bag.entries
        assert cached_bag.digest() == inline_bag.digest()
        assert metadata["feature_contract_fingerprint"] == (
            fixture.contract.feature_contract_fingerprint
        )
    roofline_scored_path = roofline_paths.ours_gcg_scored_prefixes(
        8,
        SteeringMode.NEGATIVE,
        PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
    )
    assert read_ours_gcg_scored_prefixes(
        roofline_scored_path,
        expected_feature_contract_fingerprint=(
            fixture.contract.feature_contract_fingerprint
        ),
    ) == sorted(prefix.core_prefix_id for prefix in held_out)
    assert observations.member_contracts
    assert all(value is fixture.contract for value in observations.member_contracts)


def test_clustered_derives_consume_cache_without_rerunning_diffmean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _contract_fixture()
    model = _ContractModel(fixture)
    prefixes = _union_prefixes()
    cache, _observed = _build_cache(
        monkeypatch,
        fixture=fixture,
        model=model,
        prefixes=prefixes,
    )
    monkeypatch.setattr(
        diffmean_bags,
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: pytest.fail(
            "cached feature-DiffMean derive reran inline extraction"
        ),
    )
    positive = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.POSITIVE
    ]
    negative = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.NEGATIVE
    ]

    fair_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="cached-fair")
    build_feature_diffmean_bags_fair(
        **_derive_kwargs(prefixes=positive, model=model, paths=fair_paths),
        cluster_contract=fixture.contract,
        clustered_warm_starts=cache,
    )
    fair_dict, _metadata = read_ours_gcg_bag(
        fair_paths.ours_gcg_bag(
            7,
            SteeringMode.POSITIVE,
            0.0,
            PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
        ),
        expected_feature_contract_fingerprint=(
            fixture.contract.feature_contract_fingerprint
        ),
    )
    cached_fair = cache.fair_bag(
        7,
        SteeringMode.POSITIVE,
        contract=fixture.contract,
        surface="cached fair proof",
    )
    assert FeatureConditioning.from_dict(fair_dict).entries == cached_fair.entries

    roofline_paths = AxBenchBpbAdvPaths(
        output_root=tmp_path,
        run_tag="cached-roofline",
    )
    build_feature_diffmean_bags_roofline(
        **_derive_kwargs(
            prefixes=negative,
            model=model,
            paths=roofline_paths,
        ),
        cluster_contract=fixture.contract,
        clustered_warm_starts=cache,
    )
    _optimize, held_out = split_prefixes_for_concept(
        negative,
        ratio=_cfg().ours_gcg.split_ratio,
        seed=_cfg().ours_gcg.split_seed,
    )
    for prefix in held_out:
        bag_dict, _metadata = read_ours_gcg_bag(
            roofline_paths.ours_gcg_bag(
                8,
                SteeringMode.NEGATIVE,
                0.0,
                PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
                prefix_id=prefix.core_prefix_id,
            ),
            expected_feature_contract_fingerprint=(
                fixture.contract.feature_contract_fingerprint
            ),
        )
        cached_bag = cache.roofline_bag(
            8,
            SteeringMode.NEGATIVE,
            prefix.core_prefix_id,
            contract=fixture.contract,
            surface="cached roofline proof",
        )
        assert FeatureConditioning.from_dict(bag_dict).entries == cached_bag.entries


def test_fair_cache_derive_uses_validated_public_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fair derive must consume the cache through ``fair_bag`` validation."""
    fixture = _contract_fixture()
    model = _ContractModel(fixture)
    prefixes = _union_prefixes()
    cache, _observed = _build_cache(
        monkeypatch,
        fixture=fixture,
        model=model,
        prefixes=prefixes,
    )
    positive = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.POSITIVE
    ]
    observed_lookups: list[
        tuple[int, SteeringMode, ClusterFeatureContract | None, str]
    ] = []
    real_lookup = ClusteredWarmStartCache.fair_bag

    def _spy_lookup(
        self: ClusteredWarmStartCache,
        concept_id: int,
        mode: SteeringMode,
        *,
        contract: ClusterFeatureContract | None,
        surface: str,
    ) -> FeatureConditioning:
        observed_lookups.append((concept_id, mode, contract, surface))
        return real_lookup(
            self,
            concept_id,
            mode,
            contract=contract,
            surface=surface,
        )

    monkeypatch.setattr(ClusteredWarmStartCache, "fair_bag", _spy_lookup)
    build_feature_diffmean_bags_fair(
        **_derive_kwargs(
            prefixes=positive,
            model=model,
            paths=AxBenchBpbAdvPaths(
                output_root=tmp_path,
                run_tag="fair-public-lookup-spy",
            ),
        ),
        cluster_contract=fixture.contract,
        clustered_warm_starts=cache,
    )

    assert len(observed_lookups) == 1
    concept_id, mode, contract, surface = observed_lookups[0]
    assert concept_id == 7
    assert mode is SteeringMode.POSITIVE
    assert contract is fixture.contract
    assert surface == "feature-DiffMean derive"


@pytest.mark.parametrize(
    ("builder", "surface"),
    [
        (build_feature_diffmean_bags_fair, "feature-DiffMean derive"),
        (
            build_feature_diffmean_bags_roofline,
            "roofline feature-DiffMean derive",
        ),
    ],
)
def test_feature_diffmean_derive_cluster_cache_cross_product(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    builder: Any,
    surface: str,
) -> None:
    """T2.2: all four contract/cache cells for each derive family."""
    fixture_a = _contract_fixture(preselection_seed=17)
    fixture_b = _contract_fixture(preselection_seed=18)
    model = _ContractModel(fixture_a)
    cache, _observed = _build_cache(
        monkeypatch,
        fixture=fixture_a,
        model=model,
        prefixes=_union_prefixes(),
    )

    # (None, None): exact legacy entry path.
    builder(
        **_derive_kwargs(
            prefixes=[],
            model=_UnclusteredModel(),
            paths=AxBenchBpbAdvPaths(
                output_root=tmp_path,
                run_tag=f"{surface}-legacy",
            ),
        )
    )
    # (set, None): clustered inline path.
    builder(
        **_derive_kwargs(
            prefixes=[],
            model=model,
            paths=AxBenchBpbAdvPaths(
                output_root=tmp_path,
                run_tag=f"{surface}-inline",
            ),
        ),
        cluster_contract=fixture_a.contract,
    )
    # (None, set): a clustered cache cannot be consumed without its contract.
    with pytest.raises(
        diffmean_bags.AxBenchBpbAdvError,
        match=f"{surface}: clustered warm-start cache requires",
    ):
        builder(
            **_derive_kwargs(
                prefixes=[],
                model=model,
                paths=AxBenchBpbAdvPaths(
                    output_root=tmp_path,
                    run_tag=f"{surface}-missing-contract",
                ),
            ),
            clustered_warm_starts=cache,
        )
    # (set, set with different fingerprint): reject before any lookup/write.
    with pytest.raises(
        diffmean_bags.AxBenchBpbAdvError,
        match=f"{surface}: clustered warm-start cache feature-contract fingerprint mismatch",
    ):
        builder(
            **_derive_kwargs(
                prefixes=[],
                model=model,
                paths=AxBenchBpbAdvPaths(
                    output_root=tmp_path,
                    run_tag=f"{surface}-wrong-contract",
                ),
            ),
            cluster_contract=fixture_b.contract,
            clustered_warm_starts=cache,
        )


def test_roofline_cache_key_uses_the_prefix_being_derived(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pin prefix routing, not completion-content freshness.

    ``core_prefix_id`` is content-blind with respect to regenerated completion
    tokens. This test covers the current shared-prefix-collection orchestration,
    not reuse of a cache across independently regenerated completions.
    """
    fixture = _contract_fixture()
    model = _ContractModel(fixture)
    prefixes = _union_prefixes()
    cache, _observed = _build_cache(
        monkeypatch,
        fixture=fixture,
        model=model,
        prefixes=prefixes,
    )
    negative = [
        prefix for prefix in prefixes if prefix.mode is SteeringMode.NEGATIVE
    ]
    _optimize, held_out = split_prefixes_for_concept(
        negative,
        ratio=_cfg().ours_gcg.split_ratio,
        seed=_cfg().ours_gcg.split_seed,
    )
    expected_keys = [str(prefix.core_prefix_id) for prefix in held_out]
    observed_keys: list[str] = []
    # The production lookup is keyed only by concept, mode, and this prefix id.
    # It cannot distinguish completion snapshot A from regenerated snapshot B
    # when the prefix text (and therefore core_prefix_id) is unchanged.
    real_lookup = ClusteredWarmStartCache.roofline_bag

    def _spy_lookup(
        self: ClusteredWarmStartCache,
        concept_id: int,
        mode: SteeringMode,
        prefix_id: str,
        *,
        contract: ClusterFeatureContract | None,
        surface: str,
    ) -> FeatureConditioning:
        observed_keys.append(prefix_id)
        return real_lookup(
            self,
            concept_id,
            mode,
            prefix_id,
            contract=contract,
            surface=surface,
        )

    monkeypatch.setattr(ClusteredWarmStartCache, "roofline_bag", _spy_lookup)
    build_feature_diffmean_bags_roofline(
        **_derive_kwargs(
            prefixes=negative,
            model=model,
            paths=AxBenchBpbAdvPaths(
                output_root=tmp_path,
                run_tag="roofline-key-spy",
            ),
        ),
        cluster_contract=fixture.contract,
        clustered_warm_starts=cache,
    )
    assert observed_keys == expected_keys

    changed_prefix = held_out[0]
    missing_id = "missing-roofline-prefix"
    changed = [
        (
            replace(prefix, core_prefix_id=missing_id)
            if prefix is changed_prefix
            else prefix
        )
        for prefix in negative
    ]
    with pytest.raises(
        diffmean_bags.AxBenchBpbAdvError,
        match=(
            "roofline feature-DiffMean derive: clustered warm-start cache is "
            "missing roofline key"
        ),
    ):
        build_feature_diffmean_bags_roofline(
            **_derive_kwargs(
                prefixes=changed,
                model=model,
                paths=AxBenchBpbAdvPaths(
                    output_root=tmp_path,
                    run_tag="roofline-key-miss",
                ),
            ),
            cluster_contract=fixture.contract,
            clustered_warm_starts=cache,
        )
    assert missing_id in observed_keys


@pytest.mark.parametrize(
    "builder",
    [
        build_feature_diffmean_bags_fair,
        build_feature_diffmean_bags_roofline,
    ],
)
def test_mismatched_model_contract_raises_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    builder: Any,
) -> None:
    """T2.4: pin the existing in-memory FeatureVocab guard; add no new guard."""
    fixture = _contract_fixture()
    mismatched_vocab = FeatureVocab(
        n_sae_features=_N_SAE_FEATURES,
        original_ids=(0, 2, 3, 4, 5),
    )
    model = _ContractModel(fixture, feature_vocab=mismatched_vocab)
    _install_clustered_stubs(monkeypatch, contract=fixture.contract)
    output_root = tmp_path / builder.__name__
    output_root.mkdir()

    with pytest.raises(
        FeatureGcgError,
        match="in-memory FeatureVocab does not match pinned checkpoint state",
    ):
        builder(
            **_derive_kwargs(
                prefixes=_union_prefixes(),
                model=model,
                paths=AxBenchBpbAdvPaths(
                    output_root=output_root,
                    run_tag="mismatched-contract",
                ),
            ),
            cluster_contract=fixture.contract,
        )

    assert list(output_root.iterdir()) == []
