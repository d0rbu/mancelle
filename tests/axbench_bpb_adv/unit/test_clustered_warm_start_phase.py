"""Clustered precomputed-bag CUDA ownership and warm-start reuse regressions."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.feature_diffmean_bags as fd_bags
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from errors import FeatureClustersError, FeatureMaskError
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_bags import (
    ClusteredWarmStartCache,
    build_clustered_warm_start_cache,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

_N_SAE_FEATURES = 64
_REPRESENTATIVE_ID = 10
_CONSOLIDATED_AWAY_ID = 11
_MASKED_ID = 63


def _consolidation() -> FeatureClusterConsolidation:
    """Build the real cluster/mask invariant exercised by production guards."""
    definition = FeatureClusterDefinition(
        clusters=((_REPRESENTATIVE_ID, _CONSOLIDATED_AWAY_ID),)
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("unit-clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=_N_SAE_FEATURES,
        ),
    )
    return FeatureClusterConsolidation.build(
        resolved,
        FeatureMask(FeatureMaskMode.blacklist, (_MASKED_ID,)),
    )


def _contract(fingerprint: str = "a" * 64) -> Any:
    return SimpleNamespace(
        feature_contract_fingerprint=fingerprint,
        consolidation=_consolidation(),
        selection=SimpleNamespace(num_buckets=2),
    )


def _prefix(prefix_id: str) -> Any:
    return SimpleNamespace(core_prefix_id=prefix_id)


def _member_prefix(prefix_hash: str = "context-hash") -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id="member-prefix",
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        prefix_text="ab",
        prefix_hash=prefix_hash,
        prefix_tokens=[ord("a"), ord("b")],
        concept_rich_sample_ids=["rich"],
        concept_avoiding_sample_ids=["avoid"],
        concept_rich_output_tokens=[[ord("c")]],
        concept_avoiding_output_tokens=[[ord("d")]],
    )


def _segment(
    feature_ids: tuple[int, ...],
    buckets: tuple[int, ...] | None = None,
) -> ContextFeatureSegment:
    if buckets is None:
        buckets = tuple(0 for _ in feature_ids)
    return ContextFeatureSegment(
        feature_ids=feature_ids,
        buckets=buckets,
        char_start=0,
        char_end=2,
    )


def _cfg() -> Any:
    return SimpleNamespace(
        ours_gcg=SimpleNamespace(
            split_ratio=0.5,
            split_seed=7,
            reg_lambda_grid=(0.0,),
            objective_alpha_grid=(DEFAULT_OBJECTIVE_ALPHA,),
            max_steps=1,
            minibatch_size=1,
            master_seed=11,
            max_forward_batch=1,
            max_grad_batch=1,
            mean_g_steer_batch=1,
        ),
        feature_diffmean_extraction_batch_size=1,
    )


def _patch_cache_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    groups: list[tuple[int, SteeringMode, list[Any]]],
) -> None:
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: groups,
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, *, tokenizer: (
            [f"rich:{values[0].core_prefix_id}"],
            [f"avoid:{values[0].core_prefix_id}"],
        ),
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
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: FeatureConditioning(((10, 0),)),
    )


def _build_cache(
    *,
    fair_modes: set[SteeringMode],
    roofline_modes: set[SteeringMode],
    contract: Any | None = None,
) -> ClusteredWarmStartCache:
    return build_clustered_warm_start_cache(
        prefixes=[],
        model=object(),
        base_model=object(),
        sae=object(),
        percentile_table=t.zeros((2, 99), dtype=t.float64),
        cfg=_cfg(),
        device="cpu",
        tokenizer=object(),
        context_segments_by_prefix_hash={},
        fair_modes=fair_modes,
        roofline_modes=roofline_modes,
        cluster_contract=_contract() if contract is None else contract,
    )


def _cache_with_bags(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract: Any,
    concept_id: int = 1,
    fair_warm: FeatureConditioning | None = None,
    roofline_warm: FeatureConditioning | None = None,
) -> ClusteredWarmStartCache:
    """Build a focused cache through the only supported construction boundary."""
    if fair_warm is None and roofline_warm is None:
        raise ValueError("at least one test warm bag is required")
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(concept_id, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{value.core_prefix_id}" for value in values],
    )

    def fake_diffmean(rich: list[str], _avoid: list[str], **_kwargs: Any) -> Any:
        marker = rich[0]
        if marker == "rich:p0" and fair_warm is not None:
            return fair_warm
        if marker == "rich:p1" and roofline_warm is not None:
            return roofline_warm
        raise AssertionError(f"unexpected warm-start derivation marker {marker!r}")

    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", fake_diffmean)
    return _build_cache(
        fair_modes=(
            {SteeringMode.POSITIVE} if fair_warm is not None else set()
        ),
        roofline_modes=(
            {SteeringMode.POSITIVE} if roofline_warm is not None else set()
        ),
        contract=contract,
    )


def test_clustered_warm_start_phase_extracts_all_before_single_target_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive_group = [_prefix("p0"), _prefix("p1")]
    negative_group = [_prefix("n0"), _prefix("n1")]
    events: list[str] = []
    feature_ids = iter((10, 20, 30, 40))

    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [
            (1, SteeringMode.POSITIVE, positive_group),
            (2, SteeringMode.NEGATIVE, negative_group),
        ],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda group, *, tokenizer: (
            [f"rich:{group[0].core_prefix_id}"],
            [f"avoid:{group[0].core_prefix_id}"],
        ),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{value.core_prefix_id}" for value in values],
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_vocab_allowed_ids",
        lambda _model: t.tensor([1, 2], dtype=t.int64),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: events.append("extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: events.append("target"),
    )

    def fake_diffmean(rich: list[str], _avoid: list[str], **kwargs: Any) -> Any:
        events.append(f"bag:{rich[0]}")
        assert kwargs["cluster_contract"] is contract
        return FeatureConditioning(((next(feature_ids), 0),))

    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", fake_diffmean)
    contract = _contract()
    cache = build_clustered_warm_start_cache(
        prefixes=[],
        model=object(),
        base_model=object(),
        sae=object(),
        percentile_table=t.zeros((2, 99), dtype=t.float64),
        cfg=_cfg(),
        device="cpu",
        tokenizer=object(),
        context_segments_by_prefix_hash={},
        fair_modes={SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
        roofline_modes={SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
        cluster_contract=contract,
    )

    assert events == [
        "extractor",
        "bag:rich:p0",
        "bag:rich:n0",
        "bag:rich:p1",
        "bag:rich:n1",
        "target",
    ]
    assert cache.fair_bag(
        1,
        SteeringMode.POSITIVE,
        contract=contract,
        surface="test",
    ) == FeatureConditioning(((10, 0),))
    assert cache.roofline_bag(
        2,
        SteeringMode.NEGATIVE,
        "n1",
        contract=contract,
        surface="test",
    ) == FeatureConditioning(((40, 0),))
    assert not hasattr(cache, "fair")
    assert not hasattr(cache, "roofline")
    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        cache.assert_matches(_contract("b" * 64), surface="test")
    with pytest.raises(AxBenchBpbAdvError, match="validated cluster feature contract"):
        cache.assert_matches(None, surface="test")
    with pytest.raises(AxBenchBpbAdvError, match="missing fair key"):
        cache.fair_bag(
            999,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="test",
        )
    invalid_bucket_cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((10, 2),)),
    )
    with pytest.raises(AxBenchBpbAdvError, match="out-of-range bucket"):
        invalid_bucket_cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="test",
        )


def test_members_guard_rejects_consolidated_away_id_at_fixed_context_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = _member_prefix()
    monkeypatch.setattr(
        gcg_bags,
        "build_class_margin_members",
        lambda *_args, **_kwargs: pytest.fail(
            "consolidated-away context reached member construction"
        ),
    )

    with pytest.raises(
        FeatureClustersError,
        match=(
            r"feature-GCG fixed context context-hash segment 0 contains "
            r"consolidated-away feature ID 11"
        ),
    ):
        gcg_bags._members_for_prefixes(
            [prefix],
            device=t.device("cpu"),
            context_segments_by_prefix_hash={
                prefix.prefix_hash: (_segment((_CONSOLIDATED_AWAY_ID,)),)
            },
            tokenizer=object(),
            max_seq_len=16,
            cluster_contract=_contract(),
        )


def test_members_guard_passes_representative_context_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = _member_prefix()
    segment = _segment((_REPRESENTATIVE_ID,), (1,))
    sentinel = cast(Any, object())
    observed: dict[str, Any] = {}

    def _build(
        _prefix_value: Any,
        *,
        prefix_segments: tuple[ContextFeatureSegment, ...],
        allowed_ids: t.Tensor | None,
        **_kwargs: Any,
    ) -> list[Any]:
        observed["segments"] = prefix_segments
        observed["allowed_ids"] = allowed_ids
        return [sentinel]

    monkeypatch.setattr(gcg_bags, "build_class_margin_members", _build)
    members = gcg_bags._members_for_prefixes(
        [prefix],
        device=t.device("cpu"),
        context_segments_by_prefix_hash={prefix.prefix_hash: (segment,)},
        tokenizer=object(),
        max_seq_len=16,
        allowed_ids=t.tensor([_REPRESENTATIVE_ID], dtype=t.int64),
        cluster_contract=_contract(),
    )

    assert members == [sentinel]
    assert observed["segments"] == (segment,)
    assert observed["segments"][0] is segment
    assert observed["allowed_ids"] is None


def test_members_mask_guard_rejects_masked_id_when_rep_guard_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the mask backstop independently of the stricter cluster guard."""
    prefix = _member_prefix()
    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "assert_representatives_only",
        lambda _self, _ids, *, surface: None,
    )
    monkeypatch.setattr(
        gcg_bags,
        "build_class_margin_members",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(FeatureMaskError, match="feature_mask violated"):
        gcg_bags._members_for_prefixes(
            [prefix],
            device=t.device("cpu"),
            context_segments_by_prefix_hash={
                prefix.prefix_hash: (_segment((_MASKED_ID,)),)
            },
            tokenizer=object(),
            max_seq_len=16,
            cluster_contract=_contract(),
        )


def test_clustered_members_keep_contract_valid_ids_despite_stale_allowed_ids(
    caplog: Any,
) -> None:
    """The clustered override prevents warning-only loss before member creation."""
    prefix = _member_prefix()
    segment = _segment((_REPRESENTATIVE_ID, 20), (0, 1))

    with caplog.at_level(logging.WARNING):
        members = gcg_bags._members_for_prefixes(
            [prefix],
            device=t.device("cpu"),
            context_segments_by_prefix_hash={prefix.prefix_hash: (segment,)},
            tokenizer=SimpleTokenizer(),
            max_seq_len=16,
            # Deliberately stale versus the validated contract. Passing this
            # through would silently drop representative 10 with one warning.
            allowed_ids=t.tensor([20], dtype=t.int64),
            cluster_contract=_contract(),
        )

    assert len(members) == 1
    assert members[0].rich.segment_conditions[0].entries == (
        (_REPRESENTATIVE_ID, 0),
        (20, 1),
    )
    assert members[0].avoid.segment_conditions[0].entries == (
        (_REPRESENTATIVE_ID, 0),
        (20, 1),
    )
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_cache_guard_rejects_consolidated_away_warm_start_at_validation_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((_CONSOLIDATED_AWAY_ID, 0),)),
    )

    with pytest.raises(
        FeatureClustersError,
        match=(
            r"cache representative guard cached warm start contains "
            r"consolidated-away feature ID 11"
        ),
    ):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="cache representative guard",
        )


def test_cache_guard_returns_representative_warm_start_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    source_bag = FeatureConditioning(((_REPRESENTATIVE_ID, 0), (20, 1)))
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=source_bag,
    )

    returned = cache.fair_bag(
        1,
        SteeringMode.POSITIVE,
        contract=contract,
        surface="cache representative pass-through",
    )
    assert returned is source_bag
    assert returned.entries == ((_REPRESENTATIVE_ID, 0), (20, 1))


def test_cache_mask_guard_rejects_masked_id_when_rep_guard_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the cached-mask backstop independently of the cluster guard."""
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((_MASKED_ID, 0),)),
    )
    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "assert_representatives_only",
        lambda _self, _ids, *, surface: None,
    )

    with pytest.raises(FeatureMaskError, match="feature_mask violated"):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="cache mask guard",
        )


def test_cache_refuses_cross_contract_before_bag_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a = _contract("a" * 64)
    contract_b = _contract("b" * 64)
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract_a,
        fair_warm=FeatureConditioning(((10, 0),)),
    )
    monkeypatch.setattr(
        ClusteredWarmStartCache,
        "_validated_bag",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched contract reached bag validation"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract_b,
            surface="cross-contract lookup",
        )


def test_validated_bag_independently_refuses_cross_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a = _contract("a" * 64)
    contract_b = _contract("b" * 64)
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract_a,
        fair_warm=FeatureConditioning(((10, 0),)),
    )
    monkeypatch.setattr(
        ClusteredWarmStartCache,
        "assert_matches",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="bag/cache/request feature-contract fingerprints disagree",
    ):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract_b,
            surface="independent second gate",
        )


def test_validated_bag_requires_contract_with_focused_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache_with_bags(
        monkeypatch,
        contract=_contract(),
        fair_warm=FeatureConditioning(((10, 0),)),
    )
    monkeypatch.setattr(
        ClusteredWarmStartCache,
        "assert_matches",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(AxBenchBpbAdvError, match="has no validated contract"):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=None,
            surface="missing second-gate contract",
        )


def test_cache_missing_roofline_key_has_focused_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((10, 0),)),
    )

    with pytest.raises(AxBenchBpbAdvError, match="missing roofline key"):
        cache.roofline_bag(
            1,
            SteeringMode.POSITIVE,
            "p0",
            contract=contract,
            surface="missing roofline",
        )


def test_cache_public_lookup_requires_validated_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache_with_bags(
        monkeypatch,
        contract=_contract(),
        fair_warm=FeatureConditioning(((10, 0),)),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires a validated cluster feature contract",
    ):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=None,
            surface="missing public contract",
        )


def test_cache_missing_fair_key_has_focused_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        roofline_warm=FeatureConditioning(((10, 0),)),
    )

    with pytest.raises(AxBenchBpbAdvError, match="missing fair key"):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="missing fair",
        )


def test_cache_out_of_range_bucket_has_focused_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((10, 2),)),
    )

    with pytest.raises(AxBenchBpbAdvError, match="out-of-range bucket"):
        cache.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract,
            surface="invalid bucket",
        )


def test_cache_relabel_attack_is_stopped_by_bag_bound_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a = _contract("a" * 64)
    contract_b = _contract("b" * 64)
    source_bag = FeatureConditioning(((17, 0),))
    cache_a = _cache_with_bags(
        monkeypatch,
        contract=contract_a,
        fair_warm=source_bag,
    )

    relabeled_b = replace(
        cache_a,
        feature_contract_fingerprint=contract_b.feature_contract_fingerprint,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="bag/cache/request feature-contract fingerprints disagree",
    ):
        relabeled_b.fair_bag(
            1,
            SteeringMode.POSITIVE,
            contract=contract_b,
            surface="relabel attack",
        )


def test_cache_rejects_hand_assembled_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        fair_warm=FeatureConditioning(((10, 0),)),
    )
    hand_assembled = {
        (1, SteeringMode.POSITIVE.value): FeatureConditioning(((99, 0),))
    }

    with pytest.raises(TypeError, match="factory-bound warm bags"):
        replace(cache, _fair=hand_assembled)
    with pytest.raises(TypeError, match="factory-only"):
        ClusteredWarmStartCache(
            feature_contract_fingerprint=contract.feature_contract_fingerprint,
            _fair={},
            _roofline={},
            _factory_token=object(),
        )


def test_clustered_warm_start_phase_restores_target_when_extraction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    group = [_prefix("p0"), _prefix("p1")]
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, *, tokenizer: (["rich"], ["avoid"]),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{value.core_prefix_id}" for value in values],
    )
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: events.append("extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: events.append("target"),
    )

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        events.append("boom")
        raise RuntimeError("extract failed")

    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", boom)
    with pytest.raises(RuntimeError, match="extract failed"):
        build_clustered_warm_start_cache(
            prefixes=[],
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((2, 99), dtype=t.float64),
            cfg=_cfg(),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
            cluster_contract=_contract(),
        )
    assert events == ["extractor", "boom", "target"]


def test_clustered_warm_start_phase_restores_target_when_extractor_upload_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    group = [_prefix("p0"), _prefix("p1")]
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, *, tokenizer: (["rich"], ["avoid"]),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{values[0].core_prefix_id}"],
    )

    def fail_upload(**_kwargs: Any) -> None:
        events.append("extractor-upload")
        raise RuntimeError("SAE upload failed")

    monkeypatch.setattr(gcg_bags, "_prepare_for_diffmean_forward", fail_upload)
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: events.append("target-restore"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: pytest.fail(
            "warm-start extraction ran after extractor upload failed"
        ),
    )

    with pytest.raises(RuntimeError, match="SAE upload failed"):
        build_clustered_warm_start_cache(
            prefixes=[],
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((2, 99), dtype=t.float64),
            cfg=_cfg(),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
            cluster_contract=_contract(),
        )

    assert events == ["extractor-upload", "target-restore"]


def test_clustered_warm_start_phase_validates_all_groups_before_extractor_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale later group must fail before any expensive extractor work."""

    groups = [
        (1, SteeringMode.POSITIVE, [_prefix("p0"), _prefix("p1")]),
        (2, SteeringMode.NEGATIVE, [_prefix("n0"), _prefix("n1")]),
    ]
    events: list[str] = []
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: groups,
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, *, tokenizer: (["rich"], ["avoid"]),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)

    def validate(values: list[Any], **_kwargs: Any) -> list[str]:
        prefix_id = values[0].core_prefix_id
        events.append(f"validate:{prefix_id}")
        if prefix_id == "n1":
            raise AxBenchBpbAdvError("stale later context")
        return [f"member:{prefix_id}"]

    monkeypatch.setattr(gcg_bags, "_members_for_prefixes", validate)
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: pytest.fail(
            "extractor forward phase started before every group validated"
        ),
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: pytest.fail(
            "feature extraction started before every group validated"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="stale later context"):
        build_clustered_warm_start_cache(
            prefixes=[],
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((2, 99), dtype=t.float64),
            cfg=_cfg(),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            fair_modes={SteeringMode.POSITIVE, SteeringMode.NEGATIVE},
            roofline_modes=set(),
            cluster_contract=_contract(),
        )

    assert events == ["validate:p1", "validate:p0", "validate:n1"]


def test_cache_builder_raises_on_no_held_out_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values), []),
    )

    with pytest.raises(AxBenchBpbAdvError, match="selected no held-out prefixes"):
        _build_cache(
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_on_no_held_out_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: (
            [] if values[0].core_prefix_id == "p1" else ["member"]
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="produced no held-out members"):
        _build_cache(
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_on_no_optimize_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: ([], list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: ["held-member"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="selected no optimize prefixes"):
        _build_cache(
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_on_no_optimize_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(1, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: (
            [] if values[0].core_prefix_id == "p0" else ["held-member"]
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="produced no optimize members"):
        _build_cache(
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_when_no_active_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[(1, SteeringMode.POSITIVE, group)],
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="received no active fair or roofline tasks",
    ):
        _build_cache(
            fair_modes={SteeringMode.NEGATIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_on_duplicate_fair_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[
            (1, SteeringMode.POSITIVE, group),
            (1, SteeringMode.POSITIVE, group),
        ],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: ["member"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="duplicate fair key"):
        _build_cache(
            fair_modes={SteeringMode.POSITIVE},
            roofline_modes=set(),
        )


def test_cache_builder_raises_on_duplicate_roofline_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [_prefix("p0"), _prefix("p1")]
    _patch_cache_builder(
        monkeypatch,
        groups=[
            (1, SteeringMode.POSITIVE, group),
            (1, SteeringMode.POSITIVE, group),
        ],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: ["member"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="duplicate roofline key"):
        _build_cache(
            fair_modes=set(),
            roofline_modes={SteeringMode.POSITIVE},
        )


def test_reordered_fair_consumer_rejects_cache_from_other_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a = _contract("a" * 64)
    contract_b = _contract("b" * 64)
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract_a,
        concept_id=7,
        fair_warm=FeatureConditioning(((17, 0),)),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [],
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        gcg_bags.build_ours_gcg_bags_fair(
            prefixes=[],
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=_cfg(),
            paths=cast(
                AxBenchBpbAdvPaths,
                SimpleNamespace(),
            ),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract_b,
            clustered_warm_starts=cache,
        )


def test_reordered_roofline_consumer_rejects_cache_from_other_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_a = _contract("a" * 64)
    contract_b = _contract("b" * 64)
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract_a,
        concept_id=7,
        roofline_warm=FeatureConditioning(((17, 0),)),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [],
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        gcg_bags.build_ours_gcg_bags_roofline(
            prefixes=[],
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=_cfg(),
            paths=cast(
                AxBenchBpbAdvPaths,
                SimpleNamespace(),
            ),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract_b,
            clustered_warm_starts=cache,
        )


def test_fair_gcg_consumer_uses_same_cached_warm_start_without_model_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        concept_id=7,
        fair_warm=cached_bag,
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7
        prefix.mode = SteeringMode.POSITIVE
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_assert_prefix_tokens_roundtrip",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, **_kwargs: (["rich"], ["avoid"]),
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: pytest.fail("cached consumer reran the extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: pytest.fail("cached consumer restored the extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: pytest.fail("cached consumer repeated the target swap"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
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
    optimized_from: list[FeatureConditioning] = []

    def fake_optimize(
        _model: Any,
        _members: Any,
        *,
        warm_start: FeatureConditioning,
        **_kwargs: Any,
    ) -> FeatureConditioning:
        optimized_from.append(warm_start)
        return warm_start

    monkeypatch.setattr(
        gcg_bags,
        "optimize_feature_conditioning_universal",
        fake_optimize,
    )
    monkeypatch.setattr(gcg_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")
    written_bag_metadata: list[dict[str, str]] = []
    monkeypatch.setattr(
        gcg_bags,
        "_write_ours_gcg_bag",
        lambda *_args, **kwargs: written_bag_metadata.append(kwargs["metadata"]),
    )
    scored_prefix_kwargs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        gcg_bags,
        "_write_ours_gcg_scored_prefixes",
        lambda *_args, **kwargs: scored_prefix_kwargs.append(kwargs),
    )
    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_args, **_kwargs: "bag",
        ours_gcg_scored_prefixes=lambda *_args, **_kwargs: "prefixes",
    )

    gcg_bags.build_ours_gcg_bags_fair(
        prefixes=group,
        model=object(),
        base_model=object(),
        sae=object(),
        percentile_table=t.zeros((1, 1)),
        cfg=_cfg(),
        paths=cast(AxBenchBpbAdvPaths, paths),
        device="cpu",
        tokenizer=object(),
        context_segments_by_prefix_hash={},
        cluster_contract=contract,
        clustered_warm_starts=cache,
    )
    assert optimized_from == [cached_bag]
    assert written_bag_metadata == [
        {
            "concept_id": "7",
            "mode": "positive",
            "mode_tag": "fair",
            "reg_lambda": "0.0",
            "n_features": "2",
            "warm_start_g_steer": "0.0",
            "optimize_g_steer": "0.0",
            "held_out_g_steer": "0.0",
            "n_optimize_prefixes": "1",
            "n_held_out_prefixes": "1",
            "n_optimize_members": "1",
            "n_held_out_members": "1",
            "n_rich_completions": "1",
            "n_avoid_completions": "1",
            "k_pos": "1",
            "k_neg": "1",
            "positive_bucket": "0",
            "negative_bucket": "1",
            "extraction_batch_size": "1",
            "master_seed": "11",
            "split_ratio": "0.5",
            "split_seed": "7",
            "max_steps": "1",
            "minibatch_size": "1",
            # A derive input, not just a memory knob: bound into the GCG objective via
            # _objective_factory, and gradient.py documents batch-shape-dependent bf16
            # rounding. Added to the identity by the #1054 review (see
            # test_..._reuse_raises_when_identity_changed[max_forward_batch]).
            "max_forward_batch": "1",
            "max_grad_batch": "1",
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
            # Clustered-only stamp (#1050): lets a resumed run detect a change in
            # derive SEMANTICS, which identity-based bag reuse cannot infer from
            # config alone. Deliberately absent on the unclustered path so those
            # bag bytes stay byte-identical.
            "bag_derive_schema_version": "1",
        }
    ]
    assert scored_prefix_kwargs[0]["feature_contract_fingerprint"] == (
        contract.feature_contract_fingerprint
    )


def test_roofline_gcg_consumer_uses_cached_warm_start_and_writes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        concept_id=7,
        roofline_warm=cached_bag,
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7
        prefix.mode = SteeringMode.POSITIVE
    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_kwargs: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prefixed_completions",
        lambda values, **_kwargs: (["rich"], ["avoid"]),
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_args, **_kwargs: pytest.fail("cached consumer reran the extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_diffmean_forward",
        lambda **_kwargs: pytest.fail("cached consumer restored the extractor"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "_prepare_for_gcg_forward",
        lambda **_kwargs: pytest.fail("cached consumer repeated the target swap"),
    )
    monkeypatch.setattr(
        gcg_bags,
        "assert_single_positive_bucket",
        lambda *_args, **_kwargs: (0, 1),
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
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
    optimized_from: list[FeatureConditioning] = []

    def fake_optimize(
        _model: Any,
        _members: Any,
        *,
        warm_start: FeatureConditioning,
        **_kwargs: Any,
    ) -> FeatureConditioning:
        optimized_from.append(warm_start)
        return warm_start

    monkeypatch.setattr(
        gcg_bags,
        "optimize_feature_conditioning_universal",
        fake_optimize,
    )
    monkeypatch.setattr(gcg_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")
    bag_fingerprints: list[str] = []
    monkeypatch.setattr(
        gcg_bags,
        "_write_ours_gcg_bag",
        lambda *_args, **kwargs: bag_fingerprints.append(
            kwargs["metadata"]["feature_contract_fingerprint"]
        ),
    )
    scored_fingerprints: list[str] = []
    monkeypatch.setattr(
        gcg_bags,
        "_write_ours_gcg_scored_prefixes",
        lambda *_args, **kwargs: scored_fingerprints.append(
            kwargs["feature_contract_fingerprint"]
        ),
    )
    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_args, **_kwargs: "bag",
        ours_gcg_scored_prefixes=lambda *_args, **_kwargs: "prefixes",
    )

    gcg_bags.build_ours_gcg_bags_roofline(
        prefixes=group,
        model=object(),
        base_model=object(),
        sae=object(),
        percentile_table=t.zeros((1, 1)),
        cfg=_cfg(),
        paths=cast(AxBenchBpbAdvPaths, paths),
        device="cpu",
        tokenizer=object(),
        context_segments_by_prefix_hash={},
        cluster_contract=contract,
        clustered_warm_starts=cache,
    )

    assert optimized_from == [cached_bag]
    assert bag_fingerprints == [contract.feature_contract_fingerprint]
    assert scored_fingerprints == [contract.feature_contract_fingerprint]


def test_clustered_fair_bag_is_reused_on_a_second_derive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1050: a committed bag whose identity matches is reused, not re-derived.

    This is the ONLY test that exercises the reuse branch through a real derive.
    ``test_fair_gcg_consumer_uses_same_cached_warm_start_without_model_swap`` above
    cannot: it stubs ``write_ours_gcg_bag`` and points ``ours_gcg_bag`` at the literal
    string ``"bag"``, which never exists, so ``committed_bag_for_reuse`` always returns
    None there. Without this test a mutation that disables reuse entirely -- turning
    every resume back into the #1050 byte-compare failure -- passes the whole suite.

    Uses the REAL writer so the artifact actually lands on disk, and the real
    scored-prefixes writer so the bag/marker pair is exercised together.
    """
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch, contract=contract, concept_id=7, fair_warm=cached_bag
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7

    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_k: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        gcg_bags, "_prefixed_completions", lambda values, **_k: (["rich"], ["avoid"])
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_a, **_k: pytest.fail("reuse test must consume the cached warm start"),
    )
    monkeypatch.setattr(
        gcg_bags, "_prepare_for_diffmean_forward", lambda **_k: None
    )
    monkeypatch.setattr(gcg_bags, "_prepare_for_gcg_forward", lambda **_k: None)
    monkeypatch.setattr(
        gcg_bags, "assert_single_positive_bucket", lambda *_a, **_k: (0, 1)
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_a, **_k: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags, "_group_g_steer", lambda *_a, **_k: lambda _members, _bag: 0.0
    )
    monkeypatch.setattr(gcg_bags, "_objective_factory", lambda *_a, **_k: object())
    monkeypatch.setattr(gcg_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")

    optimized: list[FeatureConditioning] = []

    def fake_optimize(
        _model: Any, _members: Any, *, warm_start: FeatureConditioning, **_k: Any
    ) -> FeatureConditioning:
        optimized.append(warm_start)
        return warm_start

    monkeypatch.setattr(
        gcg_bags, "optimize_feature_conditioning_universal", fake_optimize
    )

    bag_path = tmp_path / "bag.safetensors"
    marker_path = tmp_path / "scored_prefixes.json"
    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_a, **_k: bag_path,
        ours_gcg_scored_prefixes=lambda *_a, **_k: marker_path,
    )

    def run() -> None:
        gcg_bags.build_ours_gcg_bags_fair(
            prefixes=group,
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=_cfg(),
            paths=cast(AxBenchBpbAdvPaths, paths),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract,
            clustered_warm_starts=cache,
        )

    run()
    assert bag_path.exists(), "first derive must commit the bag"
    assert len(optimized) == 1
    first_bytes = bag_path.read_bytes()

    # Second derive over the same run_tag: the committed bag's identity matches, so
    # the optimizer must not run again and the artifact must be untouched.
    run()
    assert len(optimized) == 1, (
        "the optimizer ran a second time -- reuse did not fire, so a resumed run "
        "would re-derive and trip install_bytes_once (#1050)"
    )
    assert bag_path.read_bytes() == first_bytes, "reuse must not rewrite the artifact"


@pytest.mark.parametrize(
    "changed_field", ["master_seed", "max_forward_batch", "max_grad_batch"]
)
def test_clustered_fair_bag_reuse_raises_when_identity_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed_field: str
) -> None:
    """A genuine input change must stop the run rather than be reused over.

    Changes a PLAIN identity key rather than the contract fingerprint. An earlier
    revision of this test changed the fingerprint, which trips the reader's own
    two-way fingerprint rule *before* the identity comparison is ever reached;
    mutation testing showed it therefore SURVIVED a mutant that disabled the identity
    comparison entirely. Asserting a rejection that comes from a different guard than
    the one under test is the exact hole this now closes.

    ``max_grad_batch`` (#1245) is covered for the same reason as its forward-path
    sibling: it slices the fwd+bwd, and under real GPU bf16 the batch shape carries
    kernel rounding that can flip a GCG argmax, hence the derived bag.

    ``max_forward_batch`` is covered because review found it absent from the GCG
    identities while being a real derive input: it is bound into
    ``class_margin_objective`` through ``_objective_factory``, and ``gradient.py``
    documents batch-shape-dependent bf16 rounding, so two derives at different values
    explore different trajectories. Without it in the identity a resumed run silently
    reused a bag built under a different batch shape.
    """
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch, contract=contract, concept_id=7, fair_warm=cached_bag
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7

    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_k: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        gcg_bags, "_prefixed_completions", lambda values, **_k: (["rich"], ["avoid"])
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_a, **_k: pytest.fail("reuse test must consume the cached warm start"),
    )
    monkeypatch.setattr(
        gcg_bags, "_prepare_for_diffmean_forward", lambda **_k: None
    )
    monkeypatch.setattr(gcg_bags, "_prepare_for_gcg_forward", lambda **_k: None)
    monkeypatch.setattr(
        gcg_bags, "assert_single_positive_bucket", lambda *_a, **_k: (0, 1)
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_a, **_k: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags, "_group_g_steer", lambda *_a, **_k: lambda _members, _bag: 0.0
    )
    monkeypatch.setattr(gcg_bags, "_objective_factory", lambda *_a, **_k: object())
    monkeypatch.setattr(gcg_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")
    monkeypatch.setattr(
        gcg_bags,
        "optimize_feature_conditioning_universal",
        lambda _m, _mem, *, warm_start, **_k: warm_start,
    )

    bag_path = tmp_path / "bag.safetensors"
    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_a, **_k: bag_path,
        ours_gcg_scored_prefixes=lambda *_a, **_k: tmp_path / "scored_prefixes.json",
    )

    def run(cfg: Any) -> None:
        gcg_bags.build_ours_gcg_bags_fair(
            prefixes=group,
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=cfg,
            paths=cast(AxBenchBpbAdvPaths, paths),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract,
            clustered_warm_starts=cache,
        )

    run(_cfg())
    assert bag_path.exists()
    changed = _cfg()
    setattr(
        changed.ours_gcg,
        changed_field,
        {"master_seed": 12345, "max_forward_batch": 8, "max_grad_batch": 8}[
            changed_field
        ],
    )
    with pytest.raises(AxBenchBpbAdvError, match="does not match this run's identity"):
        run(changed)


def test_clustered_roofline_bag_is_reused_on_a_second_derive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1050 reuse through the ROOFLINE derive, which review found untested.

    Review disabled roofline-GCG reuse (and both Feature-DiffMean reuse blocks) and the
    entire default suite still passed, so this branch had no deterministic proof. The
    GPU acceptance gives it only probabilistic cover: a nondeterministic re-derive
    usually diverges, but "usually" is not a branch assertion.

    Roofline is keyed per (concept, mode, prefix, reg_lambda), so the path lambda
    varies by ``prefix_id`` -- a single shared path would make prefix two collide with
    prefix one's artifact and raise on ``prefix_id`` instead of exercising reuse.
    """
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch, contract=contract, concept_id=7, roofline_warm=cached_bag
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7
        prefix.mode = SteeringMode.POSITIVE

    monkeypatch.setattr(
        gcg_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        gcg_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(
        gcg_bags, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        gcg_bags,
        "_members_for_prefixes",
        lambda values, **_k: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        gcg_bags, "_prefixed_completions", lambda values, **_k: (["rich"], ["avoid"])
    )
    monkeypatch.setattr(
        gcg_bags,
        "feature_diffmean_bag",
        lambda *_a, **_k: pytest.fail("reuse test must consume the cached warm start"),
    )
    monkeypatch.setattr(gcg_bags, "_prepare_for_diffmean_forward", lambda **_k: None)
    monkeypatch.setattr(gcg_bags, "_prepare_for_gcg_forward", lambda **_k: None)
    monkeypatch.setattr(
        gcg_bags, "assert_single_positive_bucket", lambda *_a, **_k: (0, 1)
    )
    monkeypatch.setattr(gcg_bags, "resolve_k", lambda *_a, **_k: (1, 1))
    monkeypatch.setattr(gcg_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(gcg_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(gcg_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(
        gcg_bags, "_group_g_steer", lambda *_a, **_k: lambda _members, _bag: 0.0
    )
    monkeypatch.setattr(gcg_bags, "_objective_factory", lambda *_a, **_k: object())
    monkeypatch.setattr(gcg_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")

    optimized: list[FeatureConditioning] = []

    def fake_optimize(
        _model: Any, _members: Any, *, warm_start: FeatureConditioning, **_k: Any
    ) -> FeatureConditioning:
        optimized.append(warm_start)
        return warm_start

    monkeypatch.setattr(
        gcg_bags, "optimize_feature_conditioning_universal", fake_optimize
    )

    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_a, **k: tmp_path / f"bag_{k.get('prefix_id', 'x')}.st",
        ours_gcg_scored_prefixes=lambda *_a, **_k: tmp_path / "scored.json",
    )

    def run() -> None:
        gcg_bags.build_ours_gcg_bags_roofline(
            prefixes=group,
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=_cfg(),
            paths=cast(AxBenchBpbAdvPaths, paths),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract,
            clustered_warm_starts=cache,
        )

    run()
    written = sorted(tmp_path.glob("bag_*.st"))
    assert written, "first derive must commit at least one roofline bag"
    first = {p.name: p.read_bytes() for p in written}
    derives_after_first = len(optimized)
    assert derives_after_first > 0

    run()
    assert len(optimized) == derives_after_first, (
        "the optimizer ran again on the second derive -- roofline reuse did not fire"
    )
    assert {p.name: p.read_bytes() for p in sorted(tmp_path.glob("bag_*.st"))} == first


def _patch_fd_derive(monkeypatch: pytest.MonkeyPatch, group: list[Any]) -> list[Any]:
    """Stub the Feature-DiffMean derive's outward calls; return the g_steer spy log.

    The DiffMean derive has no optimizer to spy on -- its bag comes straight from the
    clustered warm-start cache -- so the observable boundary for "did work happen" is
    ``_group_g_steer``, which the derive calls only when it is about to write.
    """
    measured: list[str] = []
    monkeypatch.setattr(
        fd_bags,
        "_group_prefixes_by_concept_mode",
        lambda _prefixes: [(7, SteeringMode.POSITIVE, group)],
    )
    monkeypatch.setattr(
        fd_bags,
        "split_prefixes_for_concept",
        lambda values, *, ratio, seed: (list(values[:1]), list(values[1:])),
    )
    monkeypatch.setattr(fd_bags, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None)
    monkeypatch.setattr(
        fd_bags,
        "_members_for_prefixes",
        lambda values, **_k: [f"member:{values[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        fd_bags, "_prefixed_completions", lambda values, **_k: (["rich"], ["avoid"])
    )
    monkeypatch.setattr(fd_bags, "assert_single_positive_bucket", lambda *_a, **_k: (0, 1))
    monkeypatch.setattr(fd_bags, "resolve_k", lambda *_a, **_k: (1, 1))
    monkeypatch.setattr(fd_bags, "_sae_layer_for_cfg", lambda _cfg: 22)
    monkeypatch.setattr(fd_bags, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(fd_bags, "feature_vocab_allowed_ids", lambda _model: None)
    monkeypatch.setattr(fd_bags, "checkpoint_vocab_fingerprint", lambda _model: "v")

    def spy_group_g_steer(*_a: Any, **_k: Any) -> Any:
        def _measure(_members: Any, _bag: Any) -> float:
            measured.append("g_steer")
            return 0.0

        return _measure

    monkeypatch.setattr(fd_bags, "_group_g_steer", spy_group_g_steer)
    return measured


@pytest.mark.parametrize("family", ["fair", "roofline"])
def test_clustered_feature_diffmean_bag_is_reused_on_a_second_derive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str
) -> None:
    """#1050 reuse through BOTH Feature-DiffMean derives, which review found untested.

    Review disabled both Feature-DiffMean reuse blocks and not only did the default
    suite pass, so did the real 14-method GPU acceptance -- because DiffMean is
    deterministic, a re-derive produces the same bytes, so the acceptance cannot
    distinguish reuse from recomputation. Only a boundary assertion can.

    The spy is ``_group_g_steer``: the derive measures only when it is about to write,
    so a second invocation that measures again has re-derived rather than reused.
    """
    contract = _contract()
    cached_bag = FeatureConditioning(((17, 0), (31, 1)))
    cache = _cache_with_bags(
        monkeypatch,
        contract=contract,
        concept_id=7,
        fair_warm=cached_bag,
        roofline_warm=cached_bag,
    )
    group = [_prefix("p0"), _prefix("p1")]
    for prefix in group:
        prefix.concept_id = 7
        prefix.mode = SteeringMode.POSITIVE
    measured = _patch_fd_derive(monkeypatch, group)

    paths = SimpleNamespace(
        ours_gcg_bag=lambda *_a, **k: tmp_path / f"fd_{family}_{k.get('prefix_id','x')}.st",
        ours_gcg_scored_prefixes=lambda *_a, **_k: tmp_path / f"fd_{family}_scored.json",
    )
    derive = (
        fd_bags.build_feature_diffmean_bags_fair
        if family == "fair"
        else fd_bags.build_feature_diffmean_bags_roofline
    )

    def run() -> None:
        derive(
            prefixes=group,
            model=object(),
            base_model=object(),
            sae=object(),
            percentile_table=t.zeros((1, 1)),
            cfg=_cfg(),
            paths=cast(AxBenchBpbAdvPaths, paths),
            device="cpu",
            tokenizer=object(),
            context_segments_by_prefix_hash={},
            cluster_contract=contract,
            clustered_warm_starts=cache,
        )

    run()
    written = sorted(tmp_path.glob(f"fd_{family}_*.st"))
    assert written, "first derive must commit at least one bag"
    first = {p.name: p.read_bytes() for p in written}
    measurements_after_first = len(measured)
    assert measurements_after_first > 0

    run()
    assert len(measured) == measurements_after_first, (
        f"{family} Feature-DiffMean measured again on the second derive -- reuse did "
        "not fire, so a resumed run would re-derive and re-write"
    )
    assert {
        p.name: p.read_bytes() for p in sorted(tmp_path.glob(f"fd_{family}_*.st"))
    } == first
