"""Unit tests for roofline general-val per-document conditioning helpers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from errors import FeatureClustersError, FeatureMaskError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.vocab_restriction import UNCOMPACTED_VOCAB_FINGERPRINT
from steering.eval.axbench_bpb_adv import general_val_roofline
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val import (
    _InterventionCombo,
    _load_persisted_roofline_bag,
    _resolve_roofline_signal_budget,
    _roofline_reg_lambda,
    _write_roofline_bag,
)
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    GeneralValSignalBudget,
    GeneralValWarmStartCaps,
    build_general_val_doc_member,
    derive_real_output_signal,
    general_val_bag_identity_metadata,
    general_val_doc_gcg_config,
    general_val_warm_start_partition_caps,
    optimize_general_val_doc_bag,
    penalized_lm_objective_value,
    validate_general_val_bag_metadata,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)
from steering.eval.cluster_feature_contract import (
    GENERIC_ROOFLINE_SIGNAL_VERSION,
    resolve_cluster_feature_contract,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.stats_io import load_statistics
from tests.steering.test_cluster_feature_contract import _write_full_preflight_fixture


class _FakeTokenizer:
    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(chr(65 + int(token_id) % 26) for token_id in token_ids)


def _encdec_model_with_negative_selection() -> Any:
    return SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)),
        _loaded_meta_data={"user_config": {"negative_selection": {"method": "stub"}}},
    )


def _encdec_model_with_selection_means(*, pos: float = 3.0, neg: float = 2.0) -> Any:
    return SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)),
        _loaded_meta_data={
            "user_config": {
                "feature_selection": {
                    "distribution": "geometric",
                    "mean": pos,
                    "std": pos,
                },
                "negative_selection": {
                    "distribution": "geometric",
                    "mean": neg,
                    "std": neg,
                },
            }
        },
    )


def _target_bundle(model: Any) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=_FakeTokenizer(),
        max_seq_len=64,
        target_name="encdec",
    )


def _doc() -> Any:
    return SimpleNamespace(
        sample_id="general_val_7",
        prefix_tokens=[0, 1, 2],
        output_tokens=[3, 4],
        prefix_text="ABC",
    )


def _percentile_table(n_features: int, *, scales: dict[int, float] | None = None) -> t.Tensor:
    table = t.zeros(n_features, 99, dtype=t.float64)
    for fid in range(n_features):
        scale = float((scales or {}).get(fid, 1.0))
        table[fid] = t.linspace(scale / 99.0, scale, 99, dtype=t.float64)
    return table


def _linear_cluster_percentile_table(n_features: int = 8) -> t.Tensor:
    return t.arange(1, 100, dtype=t.float64).unsqueeze(0).repeat(n_features, 1)


def _cluster_contract(
    percentile_table: t.Tensor,
    *,
    mask: FeatureMask | None = None,
    fingerprint: str = "a" * 64,
    preselection_seed: int = 19,
    clusters: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3)),
    negative_selection: object | None = None,
    num_buckets: int = 2,
    effective_positive_buckets: int = 1,
) -> Any:
    n_features = int(percentile_table.shape[0])
    resolved_mask = mask or FeatureMask(
        FeatureMaskMode.whitelist,
        tuple(range(n_features - 1)),
    )
    definition = FeatureClusterDefinition.from_mapping(
        {
            "schema_version": 1,
            "clusters": [list(cluster) for cluster in clusters],
        },
        n_sae_features=n_features,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_features,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, resolved_mask)
    representatives = tuple(
        int(value)
        for value in t.nonzero(
            consolidation.cluster_allowed_universe,
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    vocab = effective_feature_vocab(consolidation, n_features)
    assert vocab is not None
    selection = (
        {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 0,
            "max_k": 8,
        }
        if negative_selection is None
        else negative_selection
    )
    return SimpleNamespace(
        n_sae_features=n_features,
        feature_mask=resolved_mask,
        consolidation=consolidation,
        representative_universe=representatives,
        feature_contract_fingerprint=fingerprint,
        statistics=SimpleNamespace(
            percentile_table_checksum=compute_percentile_checksum(percentile_table)
        ),
        checkpoint=SimpleNamespace(feature_vocab=vocab),
        selection=SimpleNamespace(
            preselection_seed=preselection_seed,
            negative_selection_json=json.dumps(
                selection,
                sort_keys=True,
                separators=(",", ":"),
            ),
            num_buckets=num_buckets,
            effective_positive_buckets=effective_positive_buckets,
        ),
    )


def _derive_clustered_signal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    activations: list[float],
    contract: Any,
    percentile_table: t.Tensor,
    k_pos: int,
    k_neg: int,
    corpus_signature: str = "nd=1;nb=5;h=clustered",
    doc: Any | None = None,
) -> FeatureConditioning:
    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        lambda *_args, **_kwargs: t.tensor([activations], dtype=t.float32),
    )
    return derive_real_output_signal(
        _doc() if doc is None else doc,
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=t.zeros(len(activations), 1, dtype=t.float64),
        percentile_table=percentile_table,
        extraction_threshold=0.0,
        signal_budget=GeneralValSignalBudget(
            k_pos=k_pos,
            k_neg=k_neg,
            positive_bucket=0,
            negative_bucket=1,
        ),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        allowed_ids=t.tensor(
            contract.checkpoint.feature_vocab.original_ids,
            dtype=t.int64,
        ),
        cluster_contract=contract,
        corpus_signature=corpus_signature,
    )


def _derive_identity_pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    activations: t.Tensor,
    contract: Any,
    percentile_table: t.Tensor,
    boundaries: t.Tensor,
    signal_budget: GeneralValSignalBudget,
) -> tuple[FeatureConditioning, FeatureConditioning]:
    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        lambda *_args, **_kwargs: activations.unsqueeze(0),
    )
    allowed_ids = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    legacy = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=boundaries,
        percentile_table=percentile_table,
        extraction_threshold=0.0,
        signal_budget=signal_budget,
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        allowed_ids=allowed_ids,
    )
    clustered = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=boundaries,
        percentile_table=percentile_table,
        extraction_threshold=0.0,
        signal_budget=signal_budget,
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        allowed_ids=allowed_ids,
        cluster_contract=contract,
        corpus_signature="nd=1;nb=5;h=identity-differential",
    )
    return legacy, clustered


def _plain_cluster_seed(
    contract: Any,
    corpus_signature: str,
    sample_id: str,
) -> int:
    payload = [
        int(contract.selection.preselection_seed),
        contract.feature_contract_fingerprint,
        corpus_signature,
        sample_id,
        GENERIC_ROOFLINE_SIGNAL_VERSION,
    ]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _plain_cluster_signal_reference(
    *,
    mags: t.Tensor,
    table: t.Tensor,
    contract: Any,
    k_pos: int,
    k_neg: int,
    corpus_signature: str,
    sample_id: str,
) -> tuple[tuple[int, int], ...] | None:
    """Independent mask -> rep-amax -> ranked positives -> sampled negatives."""

    allowed = contract.consolidation.allowed_universe.tolist()
    rep_of = contract.consolidation.rep_of.tolist()
    active_ids = [
        fid for fid, mag in enumerate(mags.tolist()) if mag > 0.0 and allowed[fid]
    ]
    active_tensor = t.tensor(active_ids, dtype=t.int64)
    active_mags = mags[active_tensor]
    scores = uint8_to_percentile(
        percentile_to_uint8(
            magnitude_to_percentile(active_tensor, active_mags, table)
        )
    ).tolist()
    max_by_rep: dict[int, float] = {}
    for fid, score in zip(active_ids, scores, strict=True):
        rep = int(rep_of[fid])
        max_by_rep[rep] = max(max_by_rep.get(rep, -1.0), float(score))
    positives = sorted(max_by_rep, key=lambda rep: (-max_by_rep[rep], rep))[:k_pos]
    positive_buckets = {
        rep: min(
            int(max_by_rep[rep] * contract.selection.effective_positive_buckets),
            contract.selection.effective_positive_buckets - 1,
        )
        for rep in positives
    }
    negative_pool = [
        rep
        for rep in contract.representative_universe
        if rep not in max_by_rep
    ]
    rng = random.Random(_plain_cluster_seed(contract, corpus_signature, sample_id))
    negatives = rng.sample(negative_pool, min(k_neg, len(negative_pool)))
    entries = {
        **positive_buckets,
        **dict.fromkeys(negatives, contract.selection.num_buckets - 1),
    }
    return tuple(sorted(entries.items())) if entries else None


def _namespace_replace(value: Any, **changes: Any) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(value), **changes})


def test_top_signal_indices_for_bucket_ranks_by_percentile_then_feature_id() -> None:
    selected = general_val_roofline._top_signal_indices_for_bucket(
        t.tensor([30, 10, 20, 40], dtype=t.int64),
        t.tensor([0, 0, 0, 1], dtype=t.int64),
        t.tensor([0.7, 0.7, 0.8, 0.9], dtype=t.float64),
        bucket=0,
        limit=2,
    )

    # Highest percentile first; tied percentiles break toward lower feature id.
    assert selected == [2, 1]


def test_top_signal_indices_for_bucket_validates_aligned_vectors() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="one-dimensional"):
        general_val_roofline._top_signal_indices_for_bucket(
            t.tensor([[1, 2]], dtype=t.int64),
            t.tensor([0, 0], dtype=t.int64),
            t.tensor([0.1, 0.2], dtype=t.float64),
            bucket=0,
            limit=1,
        )

    with pytest.raises(AxBenchBpbAdvError, match="equal lengths"):
        general_val_roofline._top_signal_indices_for_bucket(
            t.tensor([1, 2], dtype=t.int64),
            t.tensor([0], dtype=t.int64),
            t.tensor([0.1, 0.2], dtype=t.float64),
            bucket=0,
            limit=1,
        )

    assert (
        general_val_roofline._top_signal_indices_for_bucket(
            t.tensor([1, 2], dtype=t.int64),
            t.tensor([0, 0], dtype=t.int64),
            t.tensor([0.1, 0.2], dtype=t.float64),
            bucket=0,
            limit=0,
        )
        == []
    )


def test_select_signal_indices_combines_partitions_then_restores_input_order() -> None:
    selected = general_val_roofline._select_signal_indices(
        t.tensor([30, 10, 20, 40, 50], dtype=t.int64),
        t.tensor([0, 0, 0, 1, 1], dtype=t.int64),
        t.tensor([0.7, 0.7, 0.8, 0.6, 0.9], dtype=t.float64),
        signal_budget=GeneralValSignalBudget(
            k_pos=2,
            k_neg=1,
            positive_bucket=0,
            negative_bucket=1,
        ),
    )

    # Bucket 0 contributes indices 2 and 1; bucket 1 contributes index 4. The
    # returned selection is sorted so the final bag preserves extraction order.
    assert selected == [1, 2, 4]


def test_derive_real_output_signal_thresholds_and_discretizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.0, 0.2, 0.9, 1.2, 0.7, 1.5]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    bag = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=t.tensor([[0.5], [0.5], [1.0], [1.0], [1.0], [1.0]], dtype=t.float64),
        percentile_table=_percentile_table(
            6,
            scales={
                1: 10.0,  # low percentile despite bucket 0
                2: 1.0,
                3: 2.0,  # lower percentile than fid 5 despite bucket 1
                4: 1.0,
                5: 1.5,
            },
        ),
        extraction_threshold=0.1,
        signal_budget=GeneralValSignalBudget(
            k_pos=2,
            k_neg=1,
            positive_bucket=0,
            negative_bucket=1,
        ),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
    )

    assert dict(bag.entries) == {2: 0, 4: 0, 5: 1}


def test_derive_real_output_signal_preserves_sparse_available_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.0, 0.3, 0.4, 1.4]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    bag = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=t.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=t.float64),
        percentile_table=_percentile_table(4),
        extraction_threshold=0.1,
        signal_budget=GeneralValSignalBudget(
            k_pos=10,
            k_neg=10,
            positive_bucket=0,
            negative_bucket=1,
        ),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
    )

    assert dict(bag.entries) == {1: 0, 2: 0, 3: 1}


def test_derive_real_output_signal_raises_on_empty_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.zeros((1, 4), dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    with pytest.raises(AxBenchBpbAdvError, match="zero features"):
        derive_real_output_signal(
            _doc(),
            base_model=object(),
            base_sae=object(),
            sae_layer=8,
            boundaries=t.tensor([[0.5], [0.5], [1.0], [1.0]], dtype=t.float64),
            percentile_table=_percentile_table(4),
            extraction_threshold=0.0,
            signal_budget=GeneralValSignalBudget(
                k_pos=2,
                k_neg=2,
                positive_bucket=0,
                negative_bucket=1,
            ),
            tokenizer=_FakeTokenizer(),
            device=t.device("cpu"),
        )


def test_derive_real_output_signal_raises_when_sparse_budget_selects_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _acts(*_args: Any, **_kwargs: Any) -> t.Tensor:
        return t.tensor([[0.0, 0.3, 1.3]], dtype=t.float32)

    monkeypatch.setattr(general_val_roofline, "_max_pooled_sae_activations", _acts)
    with pytest.raises(AxBenchBpbAdvError, match="selected zero features"):
        derive_real_output_signal(
            _doc(),
            base_model=object(),
            base_sae=object(),
            sae_layer=8,
            boundaries=t.tensor([[1.0], [1.0], [1.0]], dtype=t.float64),
            percentile_table=_percentile_table(3),
            extraction_threshold=0.1,
            signal_budget=GeneralValSignalBudget(
                k_pos=0,
                k_neg=0,
                positive_bucket=0,
                negative_bucket=1,
            ),
            tokenizer=_FakeTokenizer(),
            device=t.device("cpu"),
        )


def test_clustered_signal_projects_siblings_ranks_representatives_and_samples_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    captured: dict[str, Any] = {}

    def _negative_sampler(
        active_ids_list: list[t.Tensor],
        n_sae_features: int,
        config: Any,
        rng: t.Generator,
        *,
        feature_mask: FeatureMask | None,
    ) -> list[t.Tensor]:
        captured.update(
            active_ids=active_ids_list[0].clone(),
            n_sae_features=n_sae_features,
            config=config,
            initial_seed=rng.initial_seed(),
            feature_mask=feature_mask,
        )
        return [t.tensor([6], dtype=t.int64)]

    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        _negative_sampler,
    )
    bag = _derive_clustered_signal(
        monkeypatch,
        activations=[10.0, 90.0, 60.0, 80.0, 50.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=2,
        k_neg=1,
    )

    assert dict(bag.entries) == {0: 0, 2: 0, 6: 1}
    assert captured["active_ids"].tolist() == [0, 2, 4]
    assert captured["n_sae_features"] == 8
    assert captured["config"].min_k == 1
    assert captured["config"].max_k == 2
    assert captured["feature_mask"] == contract.consolidation.effective_negative_mask()
    assert captured["initial_seed"] == general_val_roofline._clustered_signal_seed(
        cluster_contract=contract,
        corpus_signature="nd=1;nb=5;h=clustered",
        sample_id="general_val_7",
    )


def test_clustered_signal_tie_breaks_by_ascending_representative_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )

    bag = _derive_clustered_signal(
        monkeypatch,
        activations=[0.0, 50.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=0,
    )

    assert dict(bag.entries) == {0: 0}


def test_clustered_signal_applies_mask_before_member_amax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(1, 7)))
    contract = _cluster_contract(table, mask=mask)
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )

    bag = _derive_clustered_signal(
        monkeypatch,
        activations=[99.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=0,
    )

    assert dict(bag.entries) == {2: 0}


def test_clustered_negative_selection_is_private_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    activations = [90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    first = _derive_clustered_signal(
        monkeypatch,
        activations=activations,
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=3,
    )
    t.manual_seed(987654321)
    _ = t.rand(100)
    second = _derive_clustered_signal(
        monkeypatch,
        activations=activations,
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=3,
    )

    assert first == second
    assert len(first) == 4
    assert dict(first.entries)[0] == 0
    assert all(bucket == 1 for fid, bucket in first.entries if fid != 0)


def test_identity_after_mask_signal_is_exact_legacy_and_skips_cluster_currency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
    )
    assert contract.consolidation.is_identity_after_mask
    monkeypatch.setattr(
        general_val_roofline,
        "percentile_to_uint8",
        lambda *_args, **_kwargs: pytest.fail(
            "quantized cluster currency entered identity-equivalent branch"
        ),
    )
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: pytest.fail(
            "cluster negative sampler entered identity-equivalent branch"
        ),
    )

    legacy, clustered = _derive_identity_pair(
        monkeypatch,
        activations=t.tensor(
            [10.0, 99.0, 70.0, 98.0, 30.0, 80.0, 20.0, 60.0],
            dtype=t.float32,
        ),
        contract=contract,
        percentile_table=table,
        boundaries=t.full((8, 1), 50.0, dtype=t.float64),
        signal_budget=_signal_budget(k_pos=2, k_neg=2),
    )

    assert clustered == legacy
    assert clustered.entries == ((2, 1), (4, 0), (5, 1), (6, 0))


def test_identity_after_mask_drops_masked_active_statless_without_changing_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[1].zero_()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
    )
    assert contract.consolidation.is_identity_after_mask
    mags = t.tensor(
        [0.0, 80.0, 40.0, 20.0, 0.0, 0.0, 0.0, 0.0],
        dtype=t.float32,
    )
    original_mags = mags.clone()
    budget = _signal_budget(k_pos=1, k_neg=1)
    allowed_ids = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    no_drop_table = table.clone()
    no_drop_table[1] = t.arange(1, 100, dtype=t.float64)
    expected_without_drop = general_val_roofline._derive_legacy_real_output_signal(
        mags=mags,
        boundaries=t.zeros(8, 1, dtype=t.float64),
        percentile_table=no_drop_table,
        extraction_threshold=0.0,
        signal_budget=budget,
        allowed_ids=allowed_ids,
        sample_id="identity-statless",
    )
    messages: list[str] = []
    sink = general_val_roofline.logger.add(lambda message: messages.append(str(message)))
    try:
        actual_with_drop = general_val_roofline._derive_clustered_real_output_signal(
            mags=mags,
            boundaries=t.zeros(8, 1, dtype=t.float64),
            percentile_table=table,
            extraction_threshold=0.0,
            signal_budget=budget,
            allowed_ids=allowed_ids,
            cluster_contract=contract,
            corpus_signature="identity-statless-corpus",
            sample_id="identity-statless",
        )
    finally:
        general_val_roofline.logger.remove(sink)

    assert actual_with_drop == expected_without_drop
    assert t.equal(mags, original_mags)
    assert any("masked feature(s) [1]" in message for message in messages)


def test_identity_after_mask_wraps_allowed_active_statless_error() -> None:
    table = _linear_cluster_percentile_table()
    table[2].zero_()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
    )
    assert contract.consolidation.is_identity_after_mask

    with pytest.raises(
        AxBenchBpbAdvError,
        match="clustered active output feature lacks percentile statistics",
    ) as exc_info:
        general_val_roofline._derive_clustered_real_output_signal(
            mags=t.tensor(
                [0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=t.float32,
            ),
            boundaries=t.zeros(8, 1, dtype=t.float64),
            percentile_table=table,
            extraction_threshold=0.0,
            signal_budget=_signal_budget(k_pos=1, k_neg=1),
            allowed_ids=t.tensor(
                contract.checkpoint.feature_vocab.original_ids,
                dtype=t.int64,
            ),
            cluster_contract=contract,
            corpus_signature="identity-statless-corpus",
            sample_id="identity-statless",
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_identity_after_mask_signal_differential_corpus_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
    )
    assert contract.consolidation.is_identity_after_mask
    generator = t.Generator().manual_seed(737)

    for case_index in range(128):
        activations = t.rand(8, generator=generator, dtype=t.float32) * 120.0
        boundaries = t.rand((8, 1), generator=generator, dtype=t.float64) * 120.0
        budget = _signal_budget(
            k_pos=1 + case_index % 4,
            k_neg=1 + (case_index * 3) % 4,
        )
        legacy, clustered = _derive_identity_pair(
            monkeypatch,
            activations=activations,
            contract=contract,
            percentile_table=table,
            boundaries=boundaries,
            signal_budget=budget,
        )

        assert clustered == legacy, f"identity differential failed at case {case_index}"


def test_clustered_seed_and_metadata_bind_same_representatives_to_fingerprint() -> None:
    table = _linear_cluster_percentile_table()
    contract_a = _cluster_contract(table, fingerprint="a" * 64)
    contract_b = _cluster_contract(table, fingerprint="b" * 64)
    assert contract_a.representative_universe == contract_b.representative_universe

    seed_a = general_val_roofline._clustered_signal_seed(
        cluster_contract=contract_a,
        corpus_signature="nd=1;nb=5;h=clustered",
        sample_id="general_val_7",
    )
    seed_b = general_val_roofline._clustered_signal_seed(
        cluster_contract=contract_b,
        corpus_signature="nd=1;nb=5;h=clustered",
        sample_id="general_val_7",
    )
    assert seed_a != seed_b

    metadata_a = _signal_budget(k_pos=2, k_neg=1).metadata(
        cluster_contract=contract_a
    )
    metadata_b = _signal_budget(k_pos=2, k_neg=1).metadata(
        cluster_contract=contract_b
    )
    assert metadata_a["signal_policy"] == GENERIC_ROOFLINE_SIGNAL_VERSION
    assert metadata_a["feature_contract_fingerprint"] == "a" * 64
    assert metadata_b["feature_contract_fingerprint"] == "b" * 64


def test_identity_and_consolidating_contracts_share_policy_but_not_fingerprint() -> None:
    table = _linear_cluster_percentile_table()
    identity = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
        fingerprint="c" * 64,
    )
    consolidating = _cluster_contract(
        table,
        fingerprint="d" * 64,
    )
    assert identity.consolidation.is_identity_after_mask
    assert not consolidating.consolidation.is_identity_after_mask

    identity_metadata = _signal_budget().metadata(cluster_contract=identity)
    consolidating_metadata = _signal_budget().metadata(
        cluster_contract=consolidating
    )
    assert (
        identity_metadata["signal_policy"]
        == consolidating_metadata["signal_policy"]
        == GENERIC_ROOFLINE_SIGNAL_VERSION
    )
    assert (
        identity_metadata["feature_contract_fingerprint"]
        != consolidating_metadata["feature_contract_fingerprint"]
    )


def test_clustered_signal_fails_loud_on_missing_corpus_and_checksum_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        lambda *_args, **_kwargs: t.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=t.float32,
        ),
    )
    with pytest.raises(AxBenchBpbAdvError, match="corpus_signature"):
        derive_real_output_signal(
            _doc(),
            base_model=object(),
            base_sae=object(),
            sae_layer=8,
            boundaries=t.zeros(8, 1, dtype=t.float64),
            percentile_table=table,
            extraction_threshold=0.0,
            signal_budget=_signal_budget(k_pos=1, k_neg=0),
            tokenizer=_FakeTokenizer(),
            device=t.device("cpu"),
            cluster_contract=contract,
        )

    mutated = table.clone()
    mutated[0, 0] += 1.0
    with pytest.raises(AxBenchBpbAdvError, match="percentile table does not match"):
        derive_real_output_signal(
            _doc(),
            base_model=object(),
            base_sae=object(),
            sae_layer=8,
            boundaries=t.zeros(8, 1, dtype=t.float64),
            percentile_table=mutated,
            extraction_threshold=0.0,
            signal_budget=_signal_budget(k_pos=1, k_neg=0),
            tokenizer=_FakeTokenizer(),
            device=t.device("cpu"),
            cluster_contract=contract,
            corpus_signature="nd=1;nb=5;h=clustered",
        )


def test_no_cluster_signal_branch_does_not_enter_cluster_currency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        lambda *_args, **_kwargs: t.tensor(
            [[0.0, 0.2, 0.9, 1.2, 0.7, 1.5]],
            dtype=t.float32,
        ),
    )
    monkeypatch.setattr(
        general_val_roofline,
        "compute_percentile_checksum",
        lambda *_args, **_kwargs: pytest.fail("cluster checksum entered legacy branch"),
    )
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: pytest.fail("cluster sampler entered legacy branch"),
    )

    bag = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=object(),
        sae_layer=8,
        boundaries=t.tensor(
            [[0.5], [0.5], [1.0], [1.0], [1.0], [1.0]],
            dtype=t.float64,
        ),
        percentile_table=_percentile_table(
            6,
            scales={1: 10.0, 2: 1.0, 3: 2.0, 4: 1.0, 5: 1.5},
        ),
        extraction_threshold=0.1,
        signal_budget=GeneralValSignalBudget(
            k_pos=2,
            k_neg=1,
            positive_bucket=0,
            negative_bucket=1,
        ),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
    )

    assert dict(bag.entries) == {2: 0, 4: 0, 5: 1}


# Frozen pre-G8 copy of ``general_val_roofline.py`` (the state of ``origin/main``
# immediately before PR #909 merged, blob taken from ``50e7d1c2^1``).
#
# WHY A COMMITTED GOLDEN AND NOT ``git merge-base origin/main HEAD``: the original
# form of these two differentials resolved the "old" implementation from git
# topology.  That works on a PR branch, but the instant the PR merges,
# ``merge-base(origin/main, HEAD)`` becomes HEAD itself -- so the AST test failed
# loudly on main and, worse, the 512-draw differential silently compared the
# implementation against ITSELF and passed vacuously.  A U-invariant proof must not
# depend on the repository state it is proving something about; pin the artifact.
_PRE_G8_GOLDEN_PATH = (
    Path(__file__).parent / "_goldens" / "general_val_roofline_pre_g8.py.txt"
)
_PRE_G8_GOLDEN_SHA256 = (
    "df98694629077bd1c5b67d0854bd603a5c90a29320695ae4099369f2a46c527d"
)


def _pre_g8_golden_source() -> str:
    """Return the frozen pre-G8 module source, digest-pinned against tampering."""

    payload = _PRE_G8_GOLDEN_PATH.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != _PRE_G8_GOLDEN_SHA256:
        raise AssertionError(
            "pre-G8 legacy golden has drifted: expected "
            f"{_PRE_G8_GOLDEN_SHA256}, got {actual}. This golden is the frozen "
            "reference the unclustered U-invariant differentials compare against; "
            "it must never be regenerated from the current tree."
        )
    return payload.decode("utf-8")


def test_unclustered_derive_matches_pre_g8_golden_over_512_randomized_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pre_g8_golden_source()
    module_path = tmp_path / "pre_g8_general_val_roofline.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pre_g8_general_val_roofline", module_path)
    assert spec is not None and spec.loader is not None
    pre_g8_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pre_g8_module
    spec.loader.exec_module(pre_g8_module)

    generator = t.Generator().manual_seed(900_739)
    for case_index in range(512):
        n_features = 1 + case_index % 17
        mags = t.rand(n_features, generator=generator, dtype=t.float32) * 4.0
        mags[t.rand(n_features, generator=generator) < 0.3] = 0.0
        if case_index % 29 == 0:
            mags.zero_()
        table = _percentile_table(
            n_features,
            scales={
                fid: 0.25 + float(t.rand((), generator=generator).item()) * 4.0
                for fid in range(n_features)
            },
        )
        boundaries = t.sort(
            t.rand((n_features, 3), generator=generator, dtype=t.float64) * 4.0,
            dim=1,
        ).values
        allowed_ids: t.Tensor | None
        if case_index % 5 == 0:
            allowed_ids = None
        else:
            allowed_ids = t.tensor(
                [
                    fid
                    for fid in range(n_features)
                    if bool(t.rand((), generator=generator) > 0.35)
                ],
                dtype=t.int64,
            )
        budget = GeneralValSignalBudget(
            k_pos=case_index % 7,
            k_neg=(case_index * 3) % 7,
            positive_bucket=case_index % 3,
            negative_bucket=3,
        )
        threshold = (case_index % 6) * 0.1
        doc = _doc()
        if case_index == 0:
            doc.sample_id = 7
        elif case_index == 29:

            class _CustomSampleId:
                def __repr__(self) -> str:
                    return "<custom-sample-id>"

            doc.sample_id = _CustomSampleId()
        else:
            doc.sample_id = f"pre-g8-differential-{case_index}"

        def _invoke(
            module: Any,
            *,
            actual_mags: t.Tensor = mags,
            actual_doc: Any = doc,
            actual_boundaries: t.Tensor = boundaries,
            actual_table: t.Tensor = table,
            actual_threshold: float = threshold,
            actual_budget: GeneralValSignalBudget = budget,
            actual_allowed_ids: t.Tensor | None = allowed_ids,
        ) -> tuple[str, object]:
            monkeypatch.setattr(
                module,
                "_max_pooled_sae_activations",
                lambda *_args, **_kwargs: actual_mags.unsqueeze(0),
            )
            try:
                result = module.derive_real_output_signal(
                    actual_doc,
                    base_model=object(),
                    base_sae=object(),
                    sae_layer=8,
                    boundaries=actual_boundaries,
                    percentile_table=actual_table,
                    extraction_threshold=actual_threshold,
                    signal_budget=module.GeneralValSignalBudget(
                        k_pos=actual_budget.k_pos,
                        k_neg=actual_budget.k_neg,
                        positive_bucket=actual_budget.positive_bucket,
                        negative_bucket=actual_budget.negative_bucket,
                    ),
                    tokenizer=_FakeTokenizer(),
                    device=t.device("cpu"),
                    allowed_ids=actual_allowed_ids,
                )
            except BaseException as exc:
                return "error", (type(exc), str(exc))
            return "ok", result.entries

        assert _invoke(general_val_roofline) == _invoke(pre_g8_module), (
            f"legacy differential failed at randomized case {case_index}"
        )


def test_extracted_legacy_body_is_pre_g8_ast_exact_modulo_sample_id() -> None:
    old_source = _pre_g8_golden_source()
    new_source = Path(general_val_roofline.__file__).read_text(encoding="utf-8")

    def _function(source: str, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    old = _function(old_source, "derive_real_output_signal")
    new = _function(new_source, "_derive_legacy_real_output_signal")
    old_start = next(
        index
        for index, statement in enumerate(old.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "all_fids"
    )
    new_body = new.body[1:]  # skip helper docstring

    class _SampleIdNormalizer(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if (
                node.attr == "sample_id"
                and isinstance(node.value, ast.Name)
                and node.value.id == "doc"
            ):
                return ast.copy_location(ast.Name(id="sample_id", ctx=node.ctx), node)
            return self.generic_visit(node)

    normalized_old = [
        _SampleIdNormalizer().visit(statement) for statement in old.body[old_start:]
    ]
    assert ast.dump(ast.Module(body=normalized_old, type_ignores=[])) == ast.dump(
        ast.Module(body=new_body, type_ignores=[])
    )


def test_unclustered_signal_budget_metadata_key_order_is_legacy_exact() -> None:
    metadata = _signal_budget(k_pos=3, k_neg=4).metadata()
    assert list(metadata) == [
        "signal_policy",
        "signal_k_pos",
        "signal_k_neg",
        "positive_bucket",
        "negative_bucket",
    ]


def test_clustered_signal_budget_metadata_key_order_is_contract_exact() -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    metadata = _signal_budget(k_pos=3, k_neg=4).metadata(cluster_contract=contract)
    assert list(metadata) == [
        "signal_policy",
        "signal_k_pos",
        "signal_k_neg",
        "positive_bucket",
        "negative_bucket",
        "feature_contract_fingerprint",
    ]


def test_all_cluster_parameters_are_keyword_only_none_defaults() -> None:
    functions_and_parameters = [
        (GeneralValSignalBudget.metadata, ("cluster_contract",)),
        (derive_real_output_signal, ("cluster_contract", "corpus_signature")),
        (build_general_val_doc_member, ("cluster_contract",)),
        (general_val_warm_start_partition_caps, ("cluster_contract",)),
        (general_val_doc_gcg_config, ("cluster_contract",)),
        (penalized_lm_objective_value, ("cluster_contract",)),
        (optimize_general_val_doc_bag, ("cluster_contract",)),
        (validate_general_val_bag_metadata, ("cluster_contract",)),
        (general_val_bag_identity_metadata, ("cluster_contract",)),
    ]
    for function, parameter_names in functions_and_parameters:
        signature = inspect.signature(function)
        for parameter_name in parameter_names:
            parameter = signature.parameters[parameter_name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is None


def test_d1a_active_subset_percentiling_matches_full_width_when_fully_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = t.Generator().manual_seed(841)
    calls: list[tuple[t.Tensor, t.Tensor]] = []
    real_magnitude_to_percentile = magnitude_to_percentile

    def _spy(
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        calls.append((feature_ids.clone(), feature_mags.clone()))
        return real_magnitude_to_percentile(feature_ids, feature_mags, percentile_table)

    monkeypatch.setattr(general_val_roofline, "magnitude_to_percentile", _spy)
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )
    for case_index in range(64):
        table = t.sort(
            t.rand((8, 99), generator=generator, dtype=t.float64) * 10.0 + 0.01,
            dim=1,
        ).values
        contract = _cluster_contract(table)
        mags = t.rand(8, generator=generator, dtype=t.float32) * 10.0
        mags[(t.arange(8) + case_index) % 3 == 0] = 0.0
        calls.clear()
        _derive_clustered_signal(
            monkeypatch,
            activations=mags.tolist(),
            contract=contract,
            percentile_table=table,
            k_pos=2,
            k_neg=0,
        )
        active_ids = t.nonzero(mags > 0, as_tuple=False).squeeze(1)
        assert len(calls) == 1
        assert t.equal(calls[0][0], active_ids)
        full = real_magnitude_to_percentile(t.arange(8), mags, table)
        subset = real_magnitude_to_percentile(active_ids, mags[active_ids], table)
        assert t.equal(
            uint8_to_percentile(percentile_to_uint8(full))[active_ids],
            uint8_to_percentile(percentile_to_uint8(subset)),
        )


def test_d1b_masked_active_statless_feature_is_logged_and_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[1].zero_()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 7)),
    )
    assert not contract.consolidation.is_identity_after_mask
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )
    messages: list[str] = []
    sink = general_val_roofline.logger.add(lambda message: messages.append(str(message)))
    try:
        bag = _derive_clustered_signal(
            monkeypatch,
            activations=[0.0, 80.0, 40.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            contract=contract,
            percentile_table=table,
            k_pos=1,
            k_neg=0,
        )
    finally:
        general_val_roofline.logger.remove(sink)

    assert dict(bag.entries) == {2: 0}
    assert any("masked feature(s) [1]" in message for message in messages)


def test_d1c_nonmasked_active_statless_feature_uses_adv_error_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[2].zero_()
    contract = _cluster_contract(table)
    with pytest.raises(
        AxBenchBpbAdvError,
        match="clustered active output feature lacks percentile statistics",
    ) as exc_info:
        _derive_clustered_signal(
            monkeypatch,
            activations=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            contract=contract,
            percentile_table=table,
            k_pos=1,
            k_neg=0,
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_d1b_drops_only_masked_statless_and_nonmasked_statless_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[1].zero_()
    table[2].zero_()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 7)),
    )
    assert not contract.consolidation.is_identity_after_mask
    messages: list[str] = []
    sink = general_val_roofline.logger.add(lambda message: messages.append(str(message)))
    try:
        with pytest.raises(
            AxBenchBpbAdvError,
            match="clustered active output feature lacks percentile statistics",
        ) as exc_info:
            _derive_clustered_signal(
                monkeypatch,
                # Feature 3 is a covered, active sibling of allowed stat-less
                # feature 2. It keeps representative 2 present if mutation #10
                # wrongly drops feature 2, so the direct fatal-error assertion
                # fails instead of an unrelated presence invariant.
                activations=[0.0, 80.0, 40.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                contract=contract,
                percentile_table=table,
                k_pos=1,
                k_neg=1,
            )
    finally:
        general_val_roofline.logger.remove(sink)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert any("masked feature(s) [1]" in message for message in messages)


def test_d1a_d1b_entire_active_set_dropped_before_empty_percentile_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[1].zero_()
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1, 7)),
    )
    calls: list[tuple[t.Tensor, t.Tensor]] = []
    real_lookup = magnitude_to_percentile

    def _spy(
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        calls.append((feature_ids.clone(), feature_mags.clone()))
        return real_lookup(feature_ids, feature_mags, percentile_table)

    monkeypatch.setattr(general_val_roofline, "magnitude_to_percentile", _spy)
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.tensor([0], dtype=t.int64)],
    )
    bag = _derive_clustered_signal(
        monkeypatch,
        activations=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=1,
    )

    assert len(calls) == 1
    assert calls[0][0].numel() == calls[0][1].numel() == 0
    assert bag.entries == ((0, 1),)


def test_clustered_seed_binds_all_five_payload_components() -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table, fingerprint="a" * 64, preselection_seed=19)

    def _seed(
        actual_contract: Any = contract,
        corpus_signature: str = "corpus-a",
        sample_id: str = "doc-a",
    ) -> int:
        return general_val_roofline._clustered_signal_seed(
            cluster_contract=actual_contract,
            corpus_signature=corpus_signature,
            sample_id=sample_id,
        )

    baseline = _seed()
    assert _seed(_cluster_contract(table, preselection_seed=20)) != baseline
    assert _seed(_cluster_contract(table, fingerprint="b" * 64)) != baseline
    assert _seed(corpus_signature="corpus-b") != baseline
    assert _seed(sample_id="doc-b") != baseline
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            general_val_roofline,
            "GENERIC_ROOFLINE_SIGNAL_VERSION",
            GENERIC_ROOFLINE_SIGNAL_VERSION + "-changed",
        )
        assert _seed() != baseline
    finally:
        monkeypatch.undo()


def test_clustered_seed_is_process_stable_across_python_hash_seeds() -> None:
    script = """
from types import SimpleNamespace
from steering.eval.axbench_bpb_adv.general_val_roofline import _clustered_signal_seed
contract = SimpleNamespace(
    selection=SimpleNamespace(preselection_seed=19),
    feature_contract_fingerprint="a" * 64,
)
print(_clustered_signal_seed(
    cluster_contract=contract,
    corpus_signature="corpus",
    sample_id="doc",
))
"""
    outputs = []
    for hash_seed in ("0", "1", "937"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
        )
    assert len(set(outputs)) == 1


def test_clustered_derivation_preserves_global_python_and_torch_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    random.seed(12345)
    t.manual_seed(67890)
    python_state = random.getstate()
    torch_state = t.random.get_rng_state().clone()

    first = _derive_clustered_signal(
        monkeypatch,
        activations=[80.0, 0.0, 40.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=2,
        k_neg=2,
    )
    second = _derive_clustered_signal(
        monkeypatch,
        activations=[80.0, 0.0, 40.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=2,
        k_neg=2,
    )

    assert first == second
    assert random.getstate() == python_state
    assert t.equal(t.random.get_rng_state(), torch_state)


def test_clustered_documents_have_independent_negative_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table(32)
    contract = _cluster_contract(
        table,
        mask=FeatureMask(FeatureMaskMode.whitelist, tuple(range(31))),
        clusters=((0, 1), (2, 3)),
    )
    doc_a = _doc()
    doc_a.sample_id = "doc-a"
    doc_b = _doc()
    doc_b.sample_id = "doc-b"
    activations = [80.0, *([0.0] * 31)]
    bag_a = _derive_clustered_signal(
        monkeypatch,
        activations=activations,
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=6,
        doc=doc_a,
    )
    bag_b = _derive_clustered_signal(
        monkeypatch,
        activations=activations,
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=6,
        doc=doc_b,
    )
    assert bag_a != bag_b


@pytest.mark.parametrize(
    "case,match",
    [
        ("corpus", "corpus_signature"),
        ("corpus_empty", "corpus_signature"),
        ("corpus_nonstring", "corpus_signature"),
        ("sample_id", "sample_id"),
        ("mags_shape", "activation width"),
        ("mags_width", "activation width"),
        ("mags_nonfinite", "finite and non-negative"),
        ("mags_negative", "finite and non-negative"),
        ("threshold", "extraction_threshold=0.0"),
        ("table_ndim", "must have shape"),
        ("table_shape", "must have shape"),
        ("checksum", "percentile table does not match"),
        ("consolidation_width", "consolidation original SAE width"),
        ("consolidation_mask", "consolidation mask"),
        ("positive_bucket_count", "reserve exactly the final bucket"),
        ("positive_bucket_mismatch", "reserve exactly the final bucket"),
        ("signal_buckets", "signal-budget buckets disagree"),
        ("signal_positive_bucket", "signal-budget buckets disagree"),
        ("missing_vocab", "persistent compacted FeatureVocab"),
        ("allowed_bool", "one-dimensional integer tensor"),
        ("allowed_float", "one-dimensional integer tensor"),
        ("allowed_complex", "one-dimensional integer tensor"),
        ("allowed_2d", "one-dimensional integer tensor"),
        ("allowed_mismatch", "exactly match"),
    ],
)
def test_clustered_input_validator_fail_loud_matrix_is_nonvacuous(
    case: str,
    match: str,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    valid_allowed = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    kwargs: dict[str, Any] = {
        "mags": t.ones(8, dtype=t.float32),
        "percentile_table": table,
        "extraction_threshold": 0.0,
        "signal_budget": _signal_budget(k_pos=1, k_neg=1),
        "allowed_ids": valid_allowed,
        "cluster_contract": contract,
        "corpus_signature": "validator-corpus",
        "sample_id": "validator-doc",
    }
    general_val_roofline._validate_clustered_signal_inputs(**kwargs)
    mutated = dict(kwargs)
    if case == "corpus":
        mutated["corpus_signature"] = None
    elif case == "corpus_empty":
        mutated["corpus_signature"] = ""
    elif case == "corpus_nonstring":
        mutated["corpus_signature"] = 1
    elif case == "sample_id":
        mutated["sample_id"] = ""
    elif case == "mags_shape":
        mutated["mags"] = t.ones((2, 4), dtype=t.float32)
    elif case == "mags_width":
        mutated["mags"] = t.ones(7, dtype=t.float32)
    elif case == "mags_nonfinite":
        mutated["mags"] = t.tensor([float("nan"), *([1.0] * 7)])
    elif case == "mags_negative":
        mutated["mags"] = t.tensor([-1.0, *([1.0] * 7)])
    elif case == "threshold":
        mutated["extraction_threshold"] = 0.1
    elif case == "table_ndim":
        mutated["percentile_table"] = t.ones(8, dtype=t.float64)
    elif case == "table_shape":
        mutated["percentile_table"] = table[:, :-1]
    elif case == "checksum":
        changed = table.clone()
        changed[0, 0] += 1.0
        mutated["percentile_table"] = changed
    elif case == "consolidation_width":
        bad_consolidation = replace(contract.consolidation, rep_of=t.arange(8))
        object.__setattr__(bad_consolidation, "resolved", SimpleNamespace(n_sae_features=7))
        mutated["cluster_contract"] = _namespace_replace(
            contract,
            consolidation=bad_consolidation,
        )
    elif case == "consolidation_mask":
        bad_mask = FeatureMask(FeatureMaskMode.blacklist, (6,))
        mutated["cluster_contract"] = _namespace_replace(
            contract,
            consolidation=replace(contract.consolidation, feature_mask=bad_mask),
        )
    elif case == "positive_bucket_count":
        mutated["cluster_contract"] = _namespace_replace(
            contract,
            selection=_namespace_replace(
                contract.selection,
                effective_positive_buckets=0,
            ),
        )
    elif case == "positive_bucket_mismatch":
        mutated["cluster_contract"] = _namespace_replace(
            contract,
            selection=_namespace_replace(
                contract.selection,
                num_buckets=3,
                effective_positive_buckets=1,
            ),
        )
    elif case == "signal_buckets":
        mutated["signal_budget"] = GeneralValSignalBudget(
            k_pos=1,
            k_neg=1,
            positive_bucket=1,
            negative_bucket=0,
        )
    elif case == "signal_positive_bucket":
        mutated["signal_budget"] = GeneralValSignalBudget(
            k_pos=1,
            k_neg=1,
            positive_bucket=2,
            negative_bucket=1,
        )
    elif case == "missing_vocab":
        mutated["cluster_contract"] = _namespace_replace(
            contract,
            checkpoint=SimpleNamespace(feature_vocab=None),
        )
    elif case == "allowed_bool":
        mutated["allowed_ids"] = t.ones(4, dtype=t.bool)
    elif case == "allowed_float":
        mutated["allowed_ids"] = valid_allowed.float()
    elif case == "allowed_complex":
        mutated["allowed_ids"] = valid_allowed.to(t.complex64)
    elif case == "allowed_2d":
        mutated["allowed_ids"] = valid_allowed.unsqueeze(0)
    elif case == "allowed_mismatch":
        mutated["allowed_ids"] = valid_allowed[:-1]
    else:
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        general_val_roofline._validate_clustered_signal_inputs(**mutated)
    # Restoring the one mutated field is the passing fixture above.
    general_val_roofline._validate_clustered_signal_inputs(**kwargs)


def test_clustered_input_validator_accepts_implicit_contract_vocab() -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    validated_table, signature = (
        general_val_roofline._validate_clustered_signal_inputs(
            mags=t.ones(8),
            percentile_table=table,
            extraction_threshold=0.0,
            signal_budget=_signal_budget(k_pos=1, k_neg=1),
            allowed_ids=None,
            cluster_contract=contract,
            corpus_signature="implicit-vocab",
            sample_id="doc",
        )
    )
    assert t.equal(validated_table, table)
    assert signature == "implicit-vocab"


def test_clustered_signal_matches_independent_plain_python_oracle_over_320_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reference_sampler(
        active_ids_list: list[t.Tensor],
        n_sae_features: int,
        config: Any,
        rng: t.Generator,
        *,
        feature_mask: FeatureMask | None,
    ) -> list[t.Tensor]:
        assert len(active_ids_list) == 1
        allowed = (
            list(range(n_sae_features))
            if feature_mask is None
            else t.nonzero(
                feature_mask.allowed_universe_mask(n_sae_features),
                as_tuple=False,
            )
            .squeeze(1)
            .tolist()
        )
        active = {int(value) for value in active_ids_list[0].tolist()}
        pool = [int(value) for value in allowed if int(value) not in active]
        sampled = random.Random(rng.initial_seed()).sample(
            pool,
            min(int(config.min_k), len(pool)),
        )
        return [t.tensor(sampled, dtype=t.int64)]

    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        _reference_sampler,
    )
    generator = t.Generator().manual_seed(900_320)
    table = t.arange(1, 100, dtype=t.float64).repeat(8, 1)
    corpus_signature = "independent-semantic-oracle"
    clusters = ((0, 1), (2, 3, 4), (5, 6))

    saw_whitelist = saw_blacklist = saw_tie = saw_zero_pos = saw_zero_neg = False
    saw_single_representative = saw_pool_clamp = False
    for case_index in range(320):
        if case_index % 19 == 0:
            mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1))
        elif case_index % 2 == 0:
            ids = tuple(
                fid
                for fid in range(8)
                if fid not in {0, 1}
                and bool(t.rand((), generator=generator) < 0.25)
            )
            if not ids:
                ids = (7,)
            mask = FeatureMask(FeatureMaskMode.blacklist, ids)
            saw_blacklist = True
        else:
            allowed = {0, 1}
            allowed.update(
                fid
                for fid in range(2, 8)
                if bool(t.rand((), generator=generator) > 0.35)
            )
            mask = FeatureMask(FeatureMaskMode.whitelist, tuple(sorted(allowed)))
            saw_whitelist = True
        num_buckets = 2 + case_index % 4
        contract = _cluster_contract(
            table,
            mask=mask,
            fingerprint=hashlib.sha256(f"case-{case_index}".encode()).hexdigest(),
            preselection_seed=case_index,
            clusters=clusters,
            num_buckets=num_buckets,
            effective_positive_buckets=num_buckets - 1,
        )
        mags = t.rand(8, generator=generator, dtype=t.float32) * 120.0
        mags[t.rand(8, generator=generator) < 0.4] = 0.0
        if case_index % 23 == 0:
            mags[0] = mags[1] = 50.0
            saw_tie = True
        k_pos = case_index % 5
        k_neg = (case_index * 3) % 6
        saw_zero_pos |= k_pos == 0
        saw_zero_neg |= k_neg == 0
        saw_single_representative |= len(contract.representative_universe) == 1
        saw_pool_clamp |= k_neg > len(contract.representative_universe)
        sample_id = f"semantic-{case_index}"
        expected = _plain_cluster_signal_reference(
            mags=mags,
            table=table,
            contract=contract,
            k_pos=k_pos,
            k_neg=k_neg,
            corpus_signature=corpus_signature,
            sample_id=sample_id,
        )
        budget = GeneralValSignalBudget(
            k_pos=k_pos,
            k_neg=k_neg,
            positive_bucket=0,
            negative_bucket=num_buckets - 1,
        )
        try:
            actual = general_val_roofline._derive_clustered_real_output_signal(
                mags=mags,
                boundaries=t.zeros(8, 1, dtype=t.float64),
                percentile_table=table,
                extraction_threshold=0.0,
                signal_budget=budget,
                allowed_ids=t.tensor(
                    contract.checkpoint.feature_vocab.original_ids,
                    dtype=t.int64,
                ),
                cluster_contract=contract,
                corpus_signature=corpus_signature,
                sample_id=sample_id,
            )
        except AxBenchBpbAdvError as exc:
            assert expected is None
            assert "selected zero features" in str(exc)
        else:
            assert expected is not None
            assert actual.entries == expected

    assert all(
        (
            saw_whitelist,
            saw_blacklist,
            saw_tie,
            saw_zero_pos,
            saw_zero_neg,
            saw_single_representative,
            saw_pool_clamp,
        )
    )


def test_semantic_counterexamples_reject_wrong_reduction_and_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    table[0] *= 10.0
    table[1] *= 100.0
    contract = _cluster_contract(table)
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )
    correct = _derive_clustered_signal(
        monkeypatch,
        activations=[80.0, 90.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=0,
    )
    # Raw-max would choose cluster rep 0 from member 1's magnitude 90; the
    # member-percentile amax correctly prefers rep 2.
    assert correct.entries == ((2, 0),)
    assert correct.entries != ((0, 0),)

    tie_table = _linear_cluster_percentile_table()
    tie_contract = _cluster_contract(tie_table)
    tie = _derive_clustered_signal(
        monkeypatch,
        activations=[50.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        contract=tie_contract,
        percentile_table=tie_table,
        k_pos=1,
        k_neg=0,
    )
    assert tie.entries == ((0, 0),)
    assert tie.entries != ((2, 0),)


def test_uint8_quantization_tie_can_flip_unquantized_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    contract = _cluster_contract(table)
    real_lookup = magnitude_to_percentile

    def _near_tied_lookup(
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        if feature_ids.tolist() == [0, 2]:
            return t.tensor([10.00, 10.01], dtype=t.float64)
        return real_lookup(feature_ids, feature_mags, percentile_table)

    monkeypatch.setattr(
        general_val_roofline,
        "magnitude_to_percentile",
        _near_tied_lookup,
    )
    monkeypatch.setattr(
        general_val_roofline,
        "sample_inactive_features_batched",
        lambda *_args, **_kwargs: [t.empty(0, dtype=t.int64)],
    )
    bag = _derive_clustered_signal(
        monkeypatch,
        activations=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        contract=contract,
        percentile_table=table,
        k_pos=1,
        k_neg=0,
    )

    assert percentile_to_uint8(t.tensor([10.00, 10.01])).tolist() == [26, 26]
    assert bag.entries == ((0, 0),)
    assert bag.entries != ((2, 0),)


@pytest.mark.parametrize(
    "case,match",
    [
        ("identity_missing_vocab", "unexpectedly lacks FeatureVocab"),
        ("identity_invariant", "identity-equivalent clustered output-signal invariant"),
        ("projection_error", "projection failed"),
        ("presence_disagreement", "projected raw presence disagree"),
        ("negative_invalid", "negative-selection contract is invalid"),
        ("negative_disabled", "negative selection is disabled"),
        ("sampler_rows", "returned 0 rows"),
        ("final_invariant", "output-signal invariant failed"),
        ("duplicates", "duplicate representatives"),
        ("negative_exceeded", "exceeded k_neg"),
        ("negative_active", "active raw member"),
        ("positive_bucket", "positive bucket is outside"),
        ("negative_bucket", "reserved final bucket"),
        ("empty", "selected zero features"),
    ],
)
def test_clustered_postcondition_fail_loud_matrix_is_nonvacuous(
    case: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_cluster_percentile_table()
    base_contract = _cluster_contract(table)
    allowed = t.tensor(
        base_contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    base_kwargs: dict[str, Any] = {
        "mags": t.tensor([80.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "boundaries": t.zeros(8, 1, dtype=t.float64),
        "percentile_table": table,
        "extraction_threshold": 0.0,
        "signal_budget": _signal_budget(k_pos=1, k_neg=1),
        "allowed_ids": allowed,
        "cluster_contract": base_contract,
        "corpus_signature": "postcondition-corpus",
        "sample_id": "postcondition-doc",
    }
    general_val_roofline._derive_clustered_real_output_signal(**base_kwargs)
    mutated = dict(base_kwargs)

    if case == "identity_missing_vocab":
        identity = _cluster_contract(
            table,
            mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
        )
        mutated["allowed_ids"] = t.tensor(
            identity.checkpoint.feature_vocab.original_ids,
            dtype=t.int64,
        )
        mutated["cluster_contract"] = _namespace_replace(
            identity,
            checkpoint=SimpleNamespace(feature_vocab=None),
        )
        monkeypatch.setattr(
            general_val_roofline,
            "_validate_clustered_signal_inputs",
            lambda **_kwargs: (table, "postcondition-corpus"),
        )
    elif case == "identity_invariant":
        identity = _cluster_contract(
            table,
            mask=FeatureMask(FeatureMaskMode.blacklist, (1, 3)),
        )
        mutated["cluster_contract"] = identity
        mutated["allowed_ids"] = t.tensor(
            identity.checkpoint.feature_vocab.original_ids,
            dtype=t.int64,
        )

        def _bad_identity_assert(
            _self: Any,
            _feature_ids: t.Tensor,
            *,
            surface: str = "",
        ) -> None:
            raise FeatureClustersError(f"{surface}: sentinel")

        monkeypatch.setattr(
            FeatureClusterConsolidation,
            "assert_representatives_only",
            _bad_identity_assert,
        )
    elif case == "projection_error":
        monkeypatch.setattr(
            FeatureMask,
            "apply_to_positive",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                FeatureMaskError("projection sentinel")
            ),
        )
    elif case == "presence_disagreement":
        monkeypatch.setattr(
            FeatureClusterConsolidation,
            "project_allowed_presence",
            lambda self, _presence: t.zeros(self.n_sae_features, dtype=t.bool),
        )
    elif case == "negative_invalid":
        mutated["cluster_contract"] = _namespace_replace(
            base_contract,
            selection=_namespace_replace(
                base_contract.selection,
                negative_selection_json="[]",
            ),
        )
    elif case == "negative_disabled":
        monkeypatch.setattr(
            general_val_roofline,
            "parse_negative_selection",
            lambda _config: None,
        )
    elif case == "sampler_rows":
        monkeypatch.setattr(
            general_val_roofline,
            "sample_inactive_features_batched",
            lambda *_args, **_kwargs: [],
        )
    elif case == "final_invariant":

        def _bad_final_assert(
            _self: Any,
            _feature_ids: t.Tensor,
            *,
            surface: str = "",
        ) -> None:
            if surface == "clustered roofline general-val signal":
                raise FeatureClustersError("final sentinel")

        monkeypatch.setattr(
            FeatureClusterConsolidation,
            "assert_representatives_only",
            _bad_final_assert,
        )
    elif case == "duplicates":
        monkeypatch.setattr(
            general_val_roofline,
            "sample_inactive_features_batched",
            lambda *_args, **_kwargs: [t.tensor([0], dtype=t.int64)],
        )
    elif case == "negative_exceeded":
        monkeypatch.setattr(
            general_val_roofline,
            "sample_inactive_features_batched",
            lambda *_args, **_kwargs: [t.tensor([4, 5], dtype=t.int64)],
        )
    elif case == "negative_active":
        monkeypatch.setattr(
            general_val_roofline,
            "sample_inactive_features_batched",
            lambda *_args, **_kwargs: [t.tensor([2], dtype=t.int64)],
        )
    elif case == "positive_bucket":
        monkeypatch.setattr(
            general_val_roofline,
            "percentile_to_bucket",
            lambda values, _buckets: t.ones_like(values, dtype=t.int64),
        )
    elif case == "negative_bucket":
        real_full_like = t.full_like
        monkeypatch.setattr(
            general_val_roofline.t,
            "full_like",
            lambda values, _fill: real_full_like(values, 0),
        )
    elif case == "empty":
        mutated["mags"] = t.zeros(8, dtype=t.float32)
        mutated["signal_budget"] = _signal_budget(k_pos=0, k_neg=0)
    else:
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        general_val_roofline._derive_clustered_real_output_signal(**mutated)


def test_clustered_derive_uses_real_disk_backed_resolved_contract_and_stub_sae_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, preflight_sae = _write_full_preflight_fixture(tmp_path)
    contract = resolve_cluster_feature_contract(
        request,
        sae_loader=lambda _resolved: preflight_sae,
    )
    statistics = load_statistics(Path(request.feature_statistics_path))
    table = build_percentile_table(
        statistics.per_feature_stats,
        contract.n_sae_features,
    )
    assert compute_percentile_checksum(table) == (
        contract.statistics.percentile_table_checksum
    )
    activations = t.zeros(contract.n_sae_features, dtype=t.float32)
    activations[1] = 5.0
    activations[6] = 7.0  # masked and stat-less: legitimate corpus mismatch

    class _StubSae:
        calls = 0

        def forward(self) -> t.Tensor:
            self.calls += 1
            return activations.unsqueeze(0)

    stub_sae = _StubSae()

    def _stub_extraction(
        _completions: Any,
        *,
        model: Any,
        sae: Any,
        sae_layer: int,
        device: t.device,
    ) -> t.Tensor:
        assert model is not None
        assert sae is stub_sae
        assert sae_layer == 8
        assert device.type == "cpu"
        return cast(_StubSae, sae).forward()

    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        _stub_extraction,
    )
    feature_vocab = contract.checkpoint.feature_vocab
    assert feature_vocab is not None
    bag = derive_real_output_signal(
        _doc(),
        base_model=object(),
        base_sae=stub_sae,
        sae_layer=8,
        boundaries=t.zeros(contract.n_sae_features, 1, dtype=t.float64),
        percentile_table=table,
        extraction_threshold=0.0,
        signal_budget=_signal_budget(k_pos=1, k_neg=1),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        allowed_ids=t.tensor(
            feature_vocab.original_ids,
            dtype=t.int64,
        ),
        cluster_contract=contract,
        corpus_signature="real-disk-backed-contract",
    )

    assert stub_sae.calls == 1
    assert dict(bag.entries)[0] == 0
    contract.consolidation.assert_representatives_only(
        t.tensor(list(bag.feature_ids()), dtype=t.int64),
        surface="disk-backed roofline integration result",
    )
    assert contract.checkpoint.model_file.path.is_file()
    assert contract.checkpoint.metadata_file.path.is_file()


def test_resolve_roofline_signal_budget_uses_checkpoint_means_and_overrides() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/s",
        output_root="/tmp/o",
        feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats",
    )
    cfg.ours_gcg = OursGcgConfig()

    budget = _resolve_roofline_signal_budget(
        target_bundle=_target_bundle(_encdec_model_with_selection_means(pos=3.0, neg=2.0)),
        cfg=cfg,
    )

    assert budget == GeneralValSignalBudget(
        k_pos=3,
        k_neg=2,
        positive_bucket=0,
        negative_bucket=1,
    )

    cfg.ours_gcg = OursGcgConfig(top_k_pos=7, top_k_neg=4)
    overridden = _resolve_roofline_signal_budget(
        target_bundle=_target_bundle(_encdec_model_with_selection_means(pos=3.0, neg=2.0)),
        cfg=cfg,
    )

    assert overridden == GeneralValSignalBudget(
        k_pos=7,
        k_neg=4,
        positive_bucket=0,
        negative_bucket=1,
    )


def test_resolve_roofline_signal_budget_rejects_context_bucket_mismatch() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/s",
        output_root="/tmp/o",
        feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats",
    )
    cfg.context_cache.num_buckets = 3

    with pytest.raises(AxBenchBpbAdvError, match="context_cache.num_buckets"):
        _resolve_roofline_signal_budget(
            target_bundle=_target_bundle(_encdec_model_with_selection_means()),
            cfg=cfg,
        )


def test_build_general_val_doc_member_matches_shift_and_mask() -> None:
    output_bag = FeatureConditioning.from_dict({9: 0})
    member = build_general_val_doc_member(
        _doc(),
        segments=(ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),),
        tokenizer=_FakeTokenizer(),
        output_bag=output_bag,
        max_seq_len=16,
        device=t.device("cpu"),
    )

    assert member.dec_input.tolist() == [[0, 1, 2, 3]]
    assert member.dec_target.tolist() == [[-1, -1, 3, 4]]
    assert member.seg_map.tolist() == [[0, 0, 0, 1]]
    assert member.generation_slot == 1
    assert member.segment_conditions == [
        FeatureConditioning.from_dict({1: 0}),
        output_bag,
    ]


def test_build_general_val_doc_member_accepts_exact_cluster_vocab() -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    allowed_ids = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    output_bag = FeatureConditioning.from_dict({2: 0})

    member = build_general_val_doc_member(
        _doc(),
        segments=(
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(0,),
                char_start=0,
                char_end=3,
            ),
        ),
        tokenizer=_FakeTokenizer(),
        output_bag=output_bag,
        max_seq_len=16,
        device=t.device("cpu"),
        allowed_ids=allowed_ids,
        cluster_contract=contract,
    )

    assert member.segment_conditions == [
        FeatureConditioning.from_dict({0: 0}),
        output_bag,
    ]


def test_build_general_val_doc_member_rejects_cluster_vocab_mismatch() -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    exact_ids = contract.checkpoint.feature_vocab.original_ids

    with pytest.raises(AxBenchBpbAdvError, match="vocabulary.*does not match"):
        build_general_val_doc_member(
            _doc(),
            segments=(
                ContextFeatureSegment(
                    feature_ids=(0,),
                    buckets=(0,),
                    char_start=0,
                    char_end=3,
                ),
            ),
            tokenizer=_FakeTokenizer(),
            output_bag=FeatureConditioning.from_dict({2: 0}),
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=t.tensor(exact_ids[:-1], dtype=t.int64),
            cluster_contract=contract,
        )


def test_build_general_val_doc_member_rejects_missing_contract_vocab_cleanly() -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    allowed_ids = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    missing_vocab_contract = _namespace_replace(
        contract,
        checkpoint=SimpleNamespace(feature_vocab=None),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="checkpoint vocabulary does not match",
    ):
        build_general_val_doc_member(
            _doc(),
            segments=(
                ContextFeatureSegment(
                    feature_ids=(0,),
                    buckets=(0,),
                    char_start=0,
                    char_end=3,
                ),
            ),
            tokenizer=_FakeTokenizer(),
            output_bag=FeatureConditioning.from_dict({2: 0}),
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=allowed_ids,
            cluster_contract=cast(Any, missing_vocab_contract),
        )


def test_build_general_val_doc_member_cluster_fail_loud_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    allowed = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    valid_segments = (
        ContextFeatureSegment(
            feature_ids=(0,),
            buckets=(0,),
            char_start=0,
            char_end=3,
        ),
    )
    valid_output = FeatureConditioning.from_dict({2: 0})

    def _build(
        *,
        segments: tuple[ContextFeatureSegment, ...] = valid_segments,
        output_bag: FeatureConditioning = valid_output,
        actual_allowed: t.Tensor | None = allowed,
    ) -> Any:
        return build_general_val_doc_member(
            _doc(),
            segments=segments,
            tokenizer=_FakeTokenizer(),
            output_bag=output_bag,
            max_seq_len=16,
            device=t.device("cpu"),
            allowed_ids=actual_allowed,
            cluster_contract=contract,
        )

    _build()

    with pytest.raises(AxBenchBpbAdvError, match="exact checkpoint feature vocabulary"):
        _build(actual_allowed=None)
    with pytest.raises(AxBenchBpbAdvError, match="fixed context must already"):
        _build(
            segments=(
                ContextFeatureSegment(
                    feature_ids=(1,),
                    buckets=(0,),
                    char_start=0,
                    char_end=3,
                ),
            )
        )
    with pytest.raises(AxBenchBpbAdvError, match="output bag violates"):
        _build(output_bag=FeatureConditioning.from_dict({1: 0}))

    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "assert_representatives_only",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        FeatureMask,
        "assert_no_masked",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(AxBenchBpbAdvError, match="required read-time filtering"):
        _build(
            segments=(
                ContextFeatureSegment(
                    feature_ids=(0, 1),
                    buckets=(0, 0),
                    char_start=0,
                    char_end=3,
                ),
            )
        )


def test_build_general_val_doc_member_raises_on_overlength() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="exceeding encdec max_seq_len"):
        build_general_val_doc_member(
            _doc(),
            segments=(ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),),
            tokenizer=_FakeTokenizer(),
            output_bag=FeatureConditioning.from_dict({9: 0}),
            max_seq_len=4,
            device=t.device("cpu"),
        )


def test_optimize_general_val_doc_bag_threads_ours_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    warm = FeatureConditioning.from_dict({1: 0})
    expected = FeatureConditioning.from_dict({2: 1})

    def _fake_optimize(
        model: Any,
        members: list[Any],
        warm_start: FeatureConditioning,
        *,
        objective_factory: Any,
        config: Any,
    ) -> FeatureConditioning:
        captured.update(
            {
                "model": model,
                "members": members,
                "warm_start": warm_start,
                "objective_factory": objective_factory,
                "config": config,
            }
        )
        return expected

    monkeypatch.setattr(
        general_val_roofline,
        "optimize_feature_conditioning_universal",
        _fake_optimize,
    )
    member = cast(Any, SimpleNamespace())
    model = _encdec_model_with_negative_selection()
    device = t.device("cpu")
    ours = OursGcgConfig(max_steps=7, minibatch_size=3, master_seed=11, max_forward_batch=2)

    result = optimize_general_val_doc_bag(
        model,
        member,
        warm=warm,
        reg_lambda=0.25,
        ours=ours,
        device=device,
    )

    assert result == expected
    assert captured["model"] is model
    assert captured["members"] == [member]
    assert captured["warm_start"] == warm
    objective = captured["objective_factory"](model)
    assert objective._device == device
    assert objective._max_forward_batch == 2
    assert captured["config"].reg_lambda == pytest.approx(0.25)
    assert captured["config"].max_steps == 7
    assert captured["config"].minibatch_size == 3
    assert captured["config"].master_seed == 11
    assert captured["config"].max_forward_batch == 2
    assert captured["config"].max_k_pos == 100
    assert captured["config"].max_k_neg == 100


def test_optimize_general_val_doc_bag_threads_cluster_contract_only_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster_contract = cast(Any, object())
    warm = FeatureConditioning.from_dict({1: 0})
    expected = FeatureConditioning.from_dict({2: 0})
    captured: dict[str, Any] = {}
    gcg_cfg = FeatureGcgConfig()

    def _fake_config(
        model: Any,
        actual_warm: FeatureConditioning,
        *,
        reg_lambda: float,
        ours: OursGcgConfig,
        cluster_contract: Any,
    ) -> FeatureGcgConfig:
        captured["config_contract"] = cluster_contract
        assert model is not None
        assert actual_warm == warm
        assert reg_lambda == 0.25
        assert isinstance(ours, OursGcgConfig)
        return gcg_cfg

    def _fake_optimize(
        model: Any,
        members: list[Any],
        warm_start: FeatureConditioning,
        *,
        objective_factory: Any,
        config: FeatureGcgConfig,
        cluster_contract: Any,
    ) -> FeatureConditioning:
        del model, members, objective_factory
        assert warm_start == warm
        assert config is gcg_cfg
        captured["optimizer_contract"] = cluster_contract
        return expected

    monkeypatch.setattr(
        general_val_roofline,
        "general_val_doc_gcg_config",
        _fake_config,
    )
    monkeypatch.setattr(
        general_val_roofline,
        "optimize_feature_conditioning_universal",
        _fake_optimize,
    )

    result = optimize_general_val_doc_bag(
        object(),
        cast(Any, object()),
        warm=warm,
        reg_lambda=0.25,
        ours=OursGcgConfig(),
        device=t.device("cpu"),
        cluster_contract=cluster_contract,
    )

    assert result == expected
    assert captured == {
        "config_contract": cluster_contract,
        "optimizer_contract": cluster_contract,
    }


def test_general_val_doc_gcg_config_expands_caps_for_dense_real_signal() -> None:
    warm = FeatureConditioning.from_dict(
        {
            **dict.fromkeys(range(120), 0),
            **{1000 + fid: 1 for fid in range(115)},
        }
    )
    cfg = general_val_doc_gcg_config(
        _encdec_model_with_negative_selection(),
        warm,
        reg_lambda=0.0,
        ours=OursGcgConfig(max_steps=8, minibatch_size=1, master_seed=0, max_forward_batch=1),
    )

    assert cfg.max_k_pos == 121
    assert cfg.max_k_neg == 116


def test_general_val_warm_start_partition_caps_returns_dataclass() -> None:
    warm = FeatureConditioning.from_dict(
        {
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
        }
    )
    caps = general_val_warm_start_partition_caps(
        _encdec_model_with_negative_selection(),
        warm,
        FeatureGcgConfig(max_k_pos=2, max_k_neg=2),
    )

    assert caps == GeneralValWarmStartCaps(max_k_pos=4, max_k_neg=3)


def test_clustered_partition_caps_thread_contract_and_no_negative_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = cast(Any, object())
    captured: list[Any] = []

    def _bounds(
        _model: Any,
        cfg: FeatureGcgConfig,
        *,
        cluster_contract: Any,
    ) -> Any:
        captured.append(cluster_contract)
        return general_val_roofline.PartitionBounds(
            num_buckets=1,
            has_negative=False,
            min_k_pos=cfg.min_k_pos,
            max_k_pos=cfg.max_k_pos,
            min_k_neg=cfg.min_k_neg,
            max_k_neg=cfg.max_k_neg,
        )

    monkeypatch.setattr(general_val_roofline, "partition_bounds_from_model", _bounds)
    warm = FeatureConditioning.from_dict({0: 0, 2: 0})
    caps = general_val_warm_start_partition_caps(
        object(),
        warm,
        FeatureGcgConfig(max_k_pos=2, max_k_neg=2),
        cluster_contract=contract,
    )
    assert caps == GeneralValWarmStartCaps(max_k_pos=3, max_k_neg=2)
    assert captured == [contract]


def test_penalized_objective_validates_clustered_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    member = cast(
        Any,
        SimpleNamespace(
            segment_conditions=[
                FeatureConditioning.from_dict({0: 0}),
                FeatureConditioning.from_dict({2: 0}),
            ]
        ),
    )
    bag = FeatureConditioning.from_dict({4: 0})

    class _Objective:
        @staticmethod
        def losses(_items: Any) -> list[float]:
            return [2.5]

    monkeypatch.setattr(
        general_val_roofline,
        "lm_loss_on_masked_tokens",
        lambda *_args, **_kwargs: _Objective(),
    )
    value = penalized_lm_objective_value(
        object(),
        member,
        bag,
        reg_lambda=0.25,
        ours=OursGcgConfig(),
        device=t.device("cpu"),
        cluster_contract=contract,
    )
    assert value == pytest.approx(2.75)
    legacy_value = penalized_lm_objective_value(
        object(),
        member,
        bag,
        reg_lambda=0.5,
        ours=OursGcgConfig(),
        device=t.device("cpu"),
    )
    assert legacy_value == pytest.approx(3.0)

    invalid_member = cast(
        Any,
        SimpleNamespace(
            segment_conditions=[FeatureConditioning.from_dict({1: 0})]
        ),
    )
    with pytest.raises(AxBenchBpbAdvError, match="invalid clustered conditioning"):
        penalized_lm_objective_value(
            object(),
            invalid_member,
            bag,
            reg_lambda=0.0,
            ours=OursGcgConfig(),
            device=t.device("cpu"),
            cluster_contract=contract,
        )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="invalid clustered conditioning",
    ) as exc_info:
        penalized_lm_objective_value(
            object(),
            member,
            FeatureConditioning.from_dict({1: 0}),
            reg_lambda=0.0,
            ours=OursGcgConfig(),
            device=t.device("cpu"),
            cluster_contract=contract,
        )
    assert isinstance(exc_info.value.__cause__, FeatureClustersError)


def test_penalized_objective_wraps_masked_bag_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _cluster_contract(_linear_cluster_percentile_table())
    member = cast(
        Any,
        SimpleNamespace(
            segment_conditions=[FeatureConditioning.from_dict({0: 0})]
        ),
    )
    monkeypatch.setattr(
        FeatureClusterConsolidation,
        "assert_representatives_only",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        general_val_roofline,
        "lm_loss_on_masked_tokens",
        lambda *_args, **_kwargs: SimpleNamespace(losses=lambda _items: [0.0]),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="invalid clustered conditioning",
    ) as exc_info:
        penalized_lm_objective_value(
            object(),
            member,
            FeatureConditioning.from_dict({7: 0}),
            reg_lambda=0.0,
            ours=OursGcgConfig(),
            device=t.device("cpu"),
            cluster_contract=contract,
        )
    assert isinstance(exc_info.value.__cause__, FeatureMaskError)


_OURS = OursGcgConfig(max_steps=8, minibatch_size=16, master_seed=0, max_forward_batch=8)
_CORPUS = "nd=2;nb=10;h=aa"
_CONFIG_METADATA = {
    "extraction_threshold": "0.0",
    "neuronpedia_model_id": "gemma-2-2b",
    "context_cache_num_buckets": "2",
    "bucket_boundaries_sha256": "abc123",
}


def _signal_budget(
    *, k_pos: int = 10, k_neg: int = 10
) -> GeneralValSignalBudget:
    return GeneralValSignalBudget(
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=0,
        negative_bucket=1,
    )


def _full_bag_metadata(**overrides: str) -> dict[str, str]:
    md = {
        "sample_id": "general_val_7",
        "mode_tag": "feature_diffmean_roofline",
        "reg_lambda": "0.0",
        "general_val_corpus": _CORPUS,
        "signal_policy": "top_k_by_partition_full_percentile_v1",
        "signal_k_pos": "10",
        "signal_k_neg": "10",
        "positive_bucket": "0",
        "negative_bucket": "1",
        "max_steps": "8",
        "minibatch_size": "16",
        "master_seed": "0",
        "max_forward_batch": "8",
        **_CONFIG_METADATA,
    }
    md.update(overrides)
    return md


def test_general_val_bag_identity_metadata_matches_persisted_surface() -> None:
    metadata = general_val_bag_identity_metadata(
        sample_id="general_val_7",
        family="feature_diffmean_roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
    )

    assert metadata == _full_bag_metadata()
    without_config_metadata = general_val_bag_identity_metadata(
        sample_id="general_val_7",
        family="feature_diffmean_roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
    )
    assert not (set(_CONFIG_METADATA) & set(without_config_metadata))


def test_clustered_general_val_bag_identity_and_validator_bind_policy_and_fingerprint() -> None:
    table = _linear_cluster_percentile_table()
    contract_a = _cluster_contract(table, fingerprint="a" * 64)
    contract_b = _cluster_contract(table, fingerprint="b" * 64)
    metadata = general_val_bag_identity_metadata(
        sample_id="general_val_7",
        family="feature_diffmean_roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        cluster_contract=contract_a,
    )

    assert metadata["signal_policy"] == GENERIC_ROOFLINE_SIGNAL_VERSION
    assert metadata["feature_contract_fingerprint"] == "a" * 64
    validate_general_val_bag_metadata(
        metadata=metadata,
        sample_id="general_val_7",
        family="feature_diffmean_roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        cluster_contract=contract_a,
    )
    with pytest.raises(AxBenchBpbAdvError, match="feature_contract_fingerprint"):
        validate_general_val_bag_metadata(
            metadata=metadata,
            sample_id="general_val_7",
            family="feature_diffmean_roofline",
            reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
            cluster_contract=contract_b,
        )


def test_validate_general_val_bag_metadata_accepts_matching() -> None:
    # A bag written under the SAME identity (path components + corpus + all GCG
    # knobs) validates without raising.
    validate_general_val_bag_metadata(
        metadata=_full_bag_metadata(),
        sample_id="general_val_7",
        family="feature_diffmean_roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
    )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("sample_id", "general_val_8"),
        ("mode_tag", "ours_gcg_roofline"),
        ("reg_lambda", "0.5"),
        # issue #635 review [0]: corpus + every GCG knob must be validated so a
        # same-run_tag resume under a changed config cannot silently reuse a bag
        # optimized for a different problem (wrong frontier x, no error).
        ("general_val_corpus", "nd=9;nb=99;h=bb"),
        ("signal_policy", "dense_v0"),
        ("signal_k_pos", "999"),
        ("signal_k_neg", "999"),
        ("positive_bucket", "1"),
        ("negative_bucket", "0"),
        ("max_steps", "16"),
        ("minibatch_size", "4"),
        ("master_seed", "1"),
        ("max_forward_batch", "4"),
        ("extraction_threshold", "0.2"),
        ("neuronpedia_model_id", "gemma-3-4b-it"),
        ("context_cache_num_buckets", "4"),
        ("bucket_boundaries_sha256", "def456"),
    ],
)
def test_validate_general_val_bag_metadata_raises_on_each_drift(
    field: str, bad_value: str
) -> None:
    # The persisted metadata carries the DRIFTED value; the caller passes the
    # current run's (matching-except-for-`field`) identity, so validation catches
    # exactly that field.
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        validate_general_val_bag_metadata(
            metadata=_full_bag_metadata(**{field: bad_value}),
            sample_id="general_val_7",
            family="feature_diffmean_roofline",
            reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
        )


def test_validate_general_val_bag_metadata_rejects_missing_signal_budget() -> None:
    stale_dense_metadata = {
        key: value
        for key, value in _full_bag_metadata().items()
        if not key.startswith("signal_") and key not in {"positive_bucket", "negative_bucket"}
    }
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        validate_general_val_bag_metadata(
            metadata=stale_dense_metadata,
            sample_id="general_val_7",
            family="feature_diffmean_roofline",
            reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
        )


def test_validate_general_val_bag_metadata_checks_gcg_warm_caps_when_expected() -> None:
    metadata = _full_bag_metadata(mode_tag="roofline", max_k_pos="121", max_k_neg="116")
    validate_general_val_bag_metadata(
        metadata=metadata,
        sample_id="general_val_7",
        family="roofline",
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        warm_start_caps=GeneralValWarmStartCaps(max_k_pos=121, max_k_neg=116),
    )

    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        validate_general_val_bag_metadata(
            metadata=metadata,
            sample_id="general_val_7",
            family="roofline",
            reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
            warm_start_caps=GeneralValWarmStartCaps(max_k_pos=122, max_k_neg=116),
        )

    missing_cap_metadata = _full_bag_metadata(mode_tag="roofline", max_k_neg="116")
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        validate_general_val_bag_metadata(
            metadata=missing_cap_metadata,
            sample_id="general_val_7",
            family="roofline",
            reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
            warm_start_caps=GeneralValWarmStartCaps(max_k_pos=121, max_k_neg=116),
        )


# --------------------------------------------------------------------------- #
# roofline orchestration helpers (issue #635 review [7]) + mode-agnostic ([4]) #
# --------------------------------------------------------------------------- #
def _roofline_combo(*, method_name: str, coefficient_str: str) -> _InterventionCombo:
    return _InterventionCombo(
        target_model="encdec",
        method_name=method_name,
        mode="positive",
        concept_id=1,
        coefficient_str=coefficient_str,
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )


def test_roofline_reg_lambda_diffmean_zero_gcg_coefficient() -> None:
    # feature-DiffMean roofline has no optimizer -> reg_lambda 0.0; GCG roofline
    # uses its coefficient_str as reg_lambda.
    dm = _roofline_combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")
    assert _roofline_reg_lambda(dm, PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN) == 0.0
    gcg = _roofline_combo(method_name="ours_gcg_roofline_encdec", coefficient_str="0.5")
    assert _roofline_reg_lambda(gcg, PrecomputedBagFamily.ROOFLINE_GCG) == 0.5


def test_roofline_reg_lambda_gcg_requires_numeric_coefficient() -> None:
    gcg = _roofline_combo(method_name="ours_gcg_roofline_encdec", coefficient_str="None")
    with pytest.raises(AxBenchBpbAdvError, match="numeric reg_lambda"):
        _roofline_reg_lambda(gcg, PrecomputedBagFamily.ROOFLINE_GCG)


def test_persisted_roofline_bag_roundtrip_hit_miss_and_drift(tmp_path: Any) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    doc = _doc()  # sample_id="general_val_7"
    family = PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
    bag = FeatureConditioning.from_dict({3: 1, 7: 2})
    _write_roofline_bag(
        paths=paths,
        doc=doc,
        family=family,
        reg_lambda=0.0,
        bag=bag,
        metadata=_full_bag_metadata(mode_tag=family.value),
        vocab_fingerprint="uncompacted",
    )

    # HIT: same corpus + same knobs -> returns the persisted bag (no re-derive). The
    # uncompacted stub model matches the "uncompacted" stamp, so the Fix-2 checkpoint-V
    # check is a strict no-op here (byte-identity on the legacy path).
    loaded = _load_persisted_roofline_bag(
        paths=paths, doc=doc, family=family, reg_lambda=0.0,
        corpus_signature=_CORPUS, ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        model=_uncompacted_model(),
    )
    assert loaded is not None and dict(loaded.entries) == {3: 1, 7: 2}

    # DRIFT on corpus -> raises (review [0]).
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        _load_persisted_roofline_bag(
            paths=paths, doc=doc, family=family, reg_lambda=0.0,
            corpus_signature="nd=9;nb=99;h=bb", ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
            model=_uncompacted_model(),
        )
    # DRIFT on the current sparse signal budget -> raises even if the persisted
    # metadata itself is internally consistent.
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        _load_persisted_roofline_bag(
            paths=paths, doc=doc, family=family, reg_lambda=0.0,
            corpus_signature=_CORPUS, ours=_OURS,
            signal_budget=_signal_budget(k_pos=11),
            config_metadata=_CONFIG_METADATA,
            model=_uncompacted_model(),
        )
    # DRIFT on current extraction/boundary metadata -> raises.
    drifted_config = {**_CONFIG_METADATA, "bucket_boundaries_sha256": "def456"}
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        _load_persisted_roofline_bag(
            paths=paths, doc=doc, family=family, reg_lambda=0.0,
            corpus_signature=_CORPUS, ours=_OURS,
            signal_budget=_signal_budget(),
            config_metadata=drifted_config,
            model=_uncompacted_model(),
        )
    # DRIFT on a GCG knob -> raises.
    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        _load_persisted_roofline_bag(
            paths=paths, doc=doc, family=family, reg_lambda=0.0,
            corpus_signature=_CORPUS,
            ours=OursGcgConfig(max_steps=999, minibatch_size=16, master_seed=0, max_forward_batch=8),
            signal_budget=_signal_budget(),
            config_metadata=_CONFIG_METADATA,
            model=_uncompacted_model(),
        )

    # MISS: an unwritten doc -> None (re-derive path).
    other = _doc()
    other.sample_id = "general_val_99"
    assert _load_persisted_roofline_bag(
        paths=paths, doc=other, family=family, reg_lambda=0.0,
        corpus_signature=_CORPUS, ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        model=_uncompacted_model(),
    ) is None


def test_roofline_bag_is_mode_agnostic_shared_artifact() -> None:
    # [4] (owner-confirmed intended for feature-DiffMean + feature-GCG): the
    # roofline conditioning is the doc's own output signal, which has no
    # concept-steering direction, so positive and negative roofline methods map
    # to the SAME family + reg_lambda + bag path -> one persisted artifact and
    # one general_val_bpb (they collapse to one capability x on the frontier).
    for pos, neg, fam in [
        (
            "feature_diffmean_roofline_encdec",
            "neg_feature_diffmean_roofline_encdec",
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        ),
        (
            "ours_gcg_roofline_encdec",
            "neg_ours_gcg_roofline_encdec",
            PrecomputedBagFamily.ROOFLINE_GCG,
        ),
    ]:
        assert precomputed_bag_method_family(pos) is fam
        assert precomputed_bag_method_family(neg) is fam
        coeff = "None" if fam is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN else "0.0"
        pos_lambda = _roofline_reg_lambda(_roofline_combo(method_name=pos, coefficient_str=coeff), fam)
        neg_lambda = _roofline_reg_lambda(_roofline_combo(method_name=neg, coefficient_str=coeff), fam)
        assert pos_lambda == neg_lambda
        paths = AxBenchBpbAdvPaths(output_root="/tmp/unused", run_tag="r")
        pos_path = paths.general_val_bag("general_val_7", fam, pos_lambda)
        neg_path = paths.general_val_bag("general_val_7", fam, neg_lambda)
        assert pos_path == neg_path  # deliberate shared artifact (mode-agnostic)


# --------------------------------------------------------------------------- #
# FIX 2 (#695 PR-1 review): the roofline bag cache VERIFIES the checkpoint      #
# V-fingerprint on read — the read-side match to the stamp _write_roofline_bag  #
# records, mirroring the fair path's intervention._assert_bag_matches_          #
# checkpoint_vocab. validate_general_val_bag_metadata OMITS the fingerprint, so #
# without this a run_tag reused across checkpoints with a different compacted    #
# vocabulary V would silently serve a WRONG bag (feature ids this checkpoint     #
# never saw). Matrix: match / mismatch-raise / missing-when-compacted-raise /   #
# uncompacted-passes — the same one Agent C verified for the fair path.         #
# --------------------------------------------------------------------------- #
_VOCAB_A = FeatureVocab(n_sae_features=64, original_ids=(0, 3, 7, 11))
_VOCAB_B = FeatureVocab(n_sae_features=64, original_ids=(1, 2, 5, 9))
_ROOFLINE_DM = PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN


def _uncompacted_model() -> Any:
    """Duck-typed legacy checkpoint stub: checkpoint_vocab_fingerprint reads .feature_vocab."""
    return SimpleNamespace(feature_vocab=None)


def _compacted_model(vocab: FeatureVocab) -> Any:
    """Duck-typed compacted checkpoint stub carrying a real FeatureVocab (V-fingerprint)."""
    return SimpleNamespace(feature_vocab=vocab)


def _write_stamped_roofline_bag(
    paths: AxBenchBpbAdvPaths,
    doc: Any,
    *,
    bag: dict[int, int],
    vocab_fingerprint: str,
) -> None:
    """Write a roofline bag with valid identity metadata + a stamped V-fingerprint."""
    _write_roofline_bag(
        paths=paths,
        doc=doc,
        family=_ROOFLINE_DM,
        reg_lambda=0.0,
        bag=FeatureConditioning.from_dict(bag),
        metadata=_full_bag_metadata(mode_tag=_ROOFLINE_DM.value),
        vocab_fingerprint=vocab_fingerprint,
    )


def _write_unstamped_roofline_bag(
    paths: AxBenchBpbAdvPaths, doc: Any, *, bag: dict[int, int]
) -> None:
    """Write a roofline bag with valid identity metadata but NO vocab_fingerprint key.

    Bypasses write_ours_gcg_bag (which now REQUIRES the stamp) via save_file to reproduce
    the exact artifact shape a pre-#695-PR1 on-disk cache has: the identity metadata is
    still valid (so validate_general_val_bag_metadata passes and the read reaches the
    checkpoint-V check), only the fingerprint key is absent.
    """
    path = paths.general_val_bag(doc.sample_id, _ROOFLINE_DM, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_fids = sorted(bag)
    tensors = {
        "feature_ids": t.tensor(sorted_fids, dtype=t.int64),
        "buckets": t.tensor([bag[k] for k in sorted_fids], dtype=t.int64),
    }
    save_file(tensors, str(path), metadata=_full_bag_metadata(mode_tag=_ROOFLINE_DM.value))


def _load(paths: AxBenchBpbAdvPaths, doc: Any, *, model: Any) -> FeatureConditioning | None:
    return _load_persisted_roofline_bag(
        paths=paths,
        doc=doc,
        family=_ROOFLINE_DM,
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_signal_budget(),
        config_metadata=_CONFIG_METADATA,
        model=model,
    )


def test_fix2_fixture_vocabs_fingerprint_differently() -> None:
    # Non-vacuous: the two fixture vocabs must actually disagree, else the mismatch test
    # below would be vacuous.
    assert _VOCAB_A.fingerprint() != _VOCAB_B.fingerprint()


def test_load_persisted_roofline_bag_matching_fingerprint_loads(tmp_path: Any) -> None:
    """A V_A-stamped bag read against the SAME V_A checkpoint loads unchanged."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="vmatch")
    doc = _doc()
    _write_stamped_roofline_bag(paths, doc, bag={3: 1, 7: 0}, vocab_fingerprint=_VOCAB_A.fingerprint())

    loaded = _load(paths, doc, model=_compacted_model(_VOCAB_A))

    assert loaded is not None and dict(loaded.entries) == {3: 1, 7: 0}


def test_load_persisted_roofline_bag_mismatched_fingerprint_raises(tmp_path: Any) -> None:
    """A V_A-stamped bag read against a DIFFERENT V_B checkpoint raises, naming both Vs.

    The mutation trap for Fix 2: WITHOUT the checkpoint-V check, this cross-V read passes
    (returns the wrong bag) instead of raising.
    """
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="vmismatch")
    doc = _doc()
    _write_stamped_roofline_bag(paths, doc, bag={3: 1}, vocab_fingerprint=_VOCAB_A.fingerprint())

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _load(paths, doc, model=_compacted_model(_VOCAB_B))

    message = str(exc_info.value)
    assert _VOCAB_A.fingerprint() in message
    assert _VOCAB_B.fingerprint() in message


def test_load_persisted_roofline_bag_missing_stamp_compacted_raises(tmp_path: Any) -> None:
    """An un-stamped (pre-feature) bag against a COMPACTED checkpoint cannot prove it was
    derived for this V, so it must RAISE — the same back-compat decision as the fair path."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="vmissingc")
    doc = _doc()
    _write_unstamped_roofline_bag(paths, doc, bag={5: 0})

    with pytest.raises(AxBenchBpbAdvError, match="no.*vocab_fingerprint"):
        _load(paths, doc, model=_compacted_model(_VOCAB_A))


def test_load_persisted_roofline_bag_missing_stamp_uncompacted_loads(tmp_path: Any) -> None:
    """The SAME un-stamped bag against a LEGACY (uncompacted) checkpoint loads unchanged —
    every pre-#695 bag was implicitly derived against the full vocabulary a legacy
    checkpoint has (byte-identity preserved)."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="vmissingu")
    doc = _doc()
    _write_unstamped_roofline_bag(paths, doc, bag={6: 1})

    loaded = _load(paths, doc, model=_uncompacted_model())

    assert loaded is not None and dict(loaded.entries) == {6: 1}


def test_load_persisted_roofline_bag_uncompacted_stamp_and_model_loads(tmp_path: Any) -> None:
    """Byte-identity no-op: an ``"uncompacted"``-stamped bag against a legacy checkpoint
    loads unchanged — Fix 2 changes nothing on the uncompacted path (feature_vocab is None
    both at derive and at read)."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="vuncompacted")
    doc = _doc()
    _write_stamped_roofline_bag(
        paths, doc, bag={9: 0}, vocab_fingerprint=UNCOMPACTED_VOCAB_FINGERPRINT
    )

    loaded = _load(paths, doc, model=_uncompacted_model())

    assert loaded is not None and dict(loaded.entries) == {9: 0}
