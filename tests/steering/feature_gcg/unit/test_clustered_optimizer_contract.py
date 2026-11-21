"""Cluster-aware universal Feature-GCG contract boundaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

import steering.conditional.feature_gcg.optimize as optimize_module
import steering.eval.cluster_feature_contract as contract_module
from errors import FeatureClustersError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FEATURE_EMBED_KEY
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContract,
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    build_cluster_feature_contract,
    resolve_cluster_feature_contract,
)
from steering.refined_types import NonEmptyNonNegativeIntTuple
from tests.steering.test_cluster_feature_contract import (
    _write_full_preflight_fixture,
)

_A_SHA = "a" * 64
_B_SHA = "b" * 64
_C_SHA = "c" * 64
_D_SHA = "d" * 64


def _selection() -> CheckpointSelectionContract:
    return CheckpointSelectionContract(
        positive_selection_json=json.dumps(
            {"distribution": "geometric", "mean": 2.0},
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
        preselection_seed=42,
        selection_hash=_A_SHA,
        selection_manifest_sha256=_B_SHA,
        selection_shards_sha256=_C_SHA,
        packed_row_fingerprint=_D_SHA,
    )


def _contract_model_pair(
    tmp_path: Path,
    *,
    identity_after_mask: bool = False,
) -> tuple[ClusterFeatureContract, SimpleNamespace, dict[str, Any]]:
    """Build an I/O-free real contract and matching loaded-state double."""

    n_sae_features = 4
    mask_ids = (0, 2, 3) if identity_after_mask else (0, 1, 2, 3)
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
    definition = FeatureClusterDefinition(clusters=((0, 1),))
    feature_clusters = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.json",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(
        feature_clusters,
        feature_mask,
    )
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert feature_vocab is not None

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint = CheckpointPin(
        checkpoint_dir=checkpoint_dir,
        resolved_tag="unit",
        step=7,
        model_file=StableFileDigest(
            path=checkpoint_dir / "model_000007.pt",
            sha256=_C_SHA,
            byte_length=123,
            stat_signature=(0o100644, 1, 2, 123, 4, 5),
        ),
        metadata_file=StableFileDigest(
            path=checkpoint_dir / "meta_000007.json",
            sha256=_D_SHA,
            byte_length=45,
            stat_signature=(0o100644, 1, 3, 45, 4, 5),
        ),
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
        load_bearing_metadata_json="{}",
    )
    contract = build_cluster_feature_contract(
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        statistics=FeatureStatisticsIdentity(
            file_sha256=_A_SHA,
            percentile_table_checksum=_B_SHA,
            n_sae_features=n_sae_features,
            feature_id_domain=(0, 1, 2, 3),
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
            actual_loaded_width=n_sae_features,
        ),
        checkpoint=checkpoint,
        selection=_selection(),
        concept_source_mode="statistics_domain",
        concept_source_domain=(0, 1, 2, 3),
    )
    metadata: dict[str, Any] = {
        "feature_mask_canonical": feature_mask.canonical_dict(),
        "feature_vocab": feature_vocab.metadata_dict(),
        **feature_clusters_checkpoint_metadata(feature_clusters),
        "user_config": {
            "feature_clusters": {"path": str(feature_clusters.source_path)},
            "feature_selection": {
                "distribution": "geometric",
                "mean": 2.0,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 1.0,
            },
        },
    }
    model = SimpleNamespace(
        _loaded_meta_data=metadata,
        feature_vocab=feature_vocab,
        encoder=SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=n_sae_features,
                num_buckets=2,
            ),
            feature_vocab_ids=t.tensor(feature_vocab.original_ids, dtype=t.int64),
        ),
        assert_vocab_consistent=lambda: None,
    )
    return contract, model, metadata


def _skip_disk_rehash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch only checkpoint-file rehashing; loaded-state validation stays real."""

    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        lambda _contract: None,
    )


def _stub_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: FeatureConditioning,
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def _edit_space(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(optimize_module, "FeatureConditioningEditSpace", _edit_space)
    monkeypatch.setattr(
        optimize_module,
        "run_gcg",
        lambda **_kwargs: [SimpleNamespace(payload={"bag": result})],
    )
    return captured


def _optimize(
    model: Any,
    contract: ClusterFeatureContract,
    *,
    warm_start: FeatureConditioning,
) -> FeatureConditioning:
    return optimize_module.optimize_feature_conditioning_universal(
        model,
        [object()],
        warm_start,
        objective_factory=lambda _model: cast(Any, object()),
        config=FeatureGcgConfig(),
        cluster_contract=contract,
    )


class _RecordingIdentityObjective:
    def __init__(self) -> None:
        self.loss_trace: list[tuple[tuple[tuple[int, int], ...], ...]] = []
        self.id_weight = t.tensor(
            [[0.0] * 4, [2.0] * 4, [3.0] * 4],
            dtype=t.float32,
        )
        self.bucket_weight = t.tensor(
            [[0.0] * 4, [1.0] * 4],
            dtype=t.float32,
        )

    def losses(self, items: Sequence[Any]) -> list[float]:
        self.loss_trace.append(tuple(item.bag.entries for item in items))
        return [
            float(
                sum(
                    feature_id + bucket * 0.25
                    for feature_id, bucket in item.bag.entries
                )
            )
            for item in items
        ]

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        del step
        return [
            FeatureGrad(
                id_grad=t.full((len(item.bag), 4), 0.25, dtype=t.float32),
                bucket_grad=t.full((len(item.bag), 4), 0.25, dtype=t.float32),
                id_weight=self.id_weight,
                bucket_weight=self.bucket_weight,
            )
            for item in items
        ]


def test_clustered_optimizer_uses_representatives_under_forged_wide_helper_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is patched; a forged wide helper output tests defense-in-depth."""

    contract, model, _ = _contract_model_pair(tmp_path)
    _skip_disk_rehash(monkeypatch)
    monkeypatch.setattr(
        optimize_module,
        "feature_vocab_allowed_ids",
        lambda _model: t.tensor((0, 1, 2, 3), dtype=t.int64),
    )
    expected = FeatureConditioning(entries=((0, 0), (3, 1)))
    captured = _stub_search(monkeypatch, result=expected)

    bag = _optimize(
        model,
        contract,
        warm_start=FeatureConditioning(entries=((0, 0),)),
    )

    assert bag is expected
    assert isinstance(
        captured[0]["allowed_feature_ids"], NonEmptyNonNegativeIntTuple
    )
    assert captured[0]["allowed_feature_ids"] == (0, 2, 3)
    assert 1 not in captured[0]["allowed_feature_ids"]


def test_clustered_optimizer_rejects_nonrepresentative_warm_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is patched; loaded-state validation remains real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    _skip_disk_rehash(monkeypatch)
    _stub_search(
        monkeypatch,
        result=FeatureConditioning(entries=((0, 0),)),
    )

    with pytest.raises(
        FeatureGcgError,
        match="feature_gcg warm start contains consolidated-away feature ID 1",
    ) as exc_info:
        _optimize(
            model,
            contract,
            warm_start=FeatureConditioning(entries=((1, 0),)),
        )
    assert isinstance(exc_info.value.__cause__, FeatureClustersError)


def test_clustered_optimizer_rejects_nonrepresentative_final_bag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is patched; loaded-state validation remains real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    _skip_disk_rehash(monkeypatch)
    _stub_search(
        monkeypatch,
        result=FeatureConditioning(entries=((1, 0),)),
    )

    with pytest.raises(
        FeatureGcgError,
        match="feature_gcg final bag contains consolidated-away feature ID 1",
    ) as exc_info:
        _optimize(
            model,
            contract,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )
    assert isinstance(exc_info.value.__cause__, FeatureClustersError)


def test_corrupted_uncompacted_state_requires_persistent_feature_vocab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: validation is bypassed to forge an impossible checkpoint."""

    contract, model, _ = _contract_model_pair(tmp_path)
    forged = replace(
        contract,
        checkpoint=replace(contract.checkpoint, feature_vocab=None),
    )
    model.feature_vocab = None
    monkeypatch.setattr(
        optimize_module,
        "assert_loaded_cluster_target",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        optimize_module,
        "partition_bounds_from_model",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(FeatureGcgError, match="persistent FeatureVocab"):
        _optimize(
            model,
            forged,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )


def test_identity_after_mask_optimizer_matches_legacy_trace_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is patched; the real loaded-state contract stays active."""

    contract, clustered_model, _ = _contract_model_pair(
        tmp_path,
        identity_after_mask=True,
    )
    assert contract.consolidation.is_identity_after_mask
    legacy_model = SimpleNamespace(
        _loaded_meta_data={
            "user_config": {
                "negative_selection": {
                    "distribution": "geometric",
                    "mean": 1.0,
                }
            }
        },
        feature_vocab=clustered_model.feature_vocab,
        encoder=clustered_model.encoder,
        assert_vocab_consistent=lambda: None,
    )
    _skip_disk_rehash(monkeypatch)
    captured_payloads: list[dict[str, Any]] = []
    production_run_gcg = optimize_module.run_gcg

    def _capture_run(**kwargs: Any):
        results = production_run_gcg(**kwargs)
        captured_payloads.append(results[0].payload)
        return results

    monkeypatch.setattr(optimize_module, "run_gcg", _capture_run)
    config = FeatureGcgConfig(
        max_steps=12,
        candidate_batch_size=4,
        top_k=3,
        temperature=0.8,
        master_seed=737,
        minibatch_size=2,
        min_k_pos=1,
        max_k_pos=3,
        min_k_neg=1,
        max_k_neg=3,
    )
    warm = FeatureConditioning(entries=((2, 0), (3, 1)))
    members = ["member-a", "member-b", "member-c"]

    legacy_objective = _RecordingIdentityObjective()
    global_rng_before = t.random.get_rng_state().clone()
    legacy = optimize_module.optimize_feature_conditioning_universal(
        legacy_model,
        members,
        warm,
        objective_factory=lambda _model: legacy_objective,
        config=config,
    )
    global_rng_after_legacy = t.random.get_rng_state().clone()

    clustered_objective = _RecordingIdentityObjective()
    clustered = optimize_module.optimize_feature_conditioning_universal(
        clustered_model,
        members,
        warm,
        objective_factory=lambda _model: clustered_objective,
        config=config,
        cluster_contract=contract,
    )
    global_rng_after_clustered = t.random.get_rng_state().clone()

    assert clustered == legacy
    assert clustered_objective.loss_trace == legacy_objective.loss_trace
    assert captured_payloads[1] == captured_payloads[0]
    assert t.equal(global_rng_after_legacy, global_rng_before)
    assert t.equal(global_rng_after_clustered, global_rng_before)


def test_corrupted_contract_empty_representative_universe_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: only rehash is patched; loaded-state validation is real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    empty_consolidation = replace(
        contract.consolidation,
        cluster_allowed_universe=t.zeros(contract.n_sae_features, dtype=t.bool),
    )
    forged = replace(
        contract,
        consolidation=empty_consolidation,
        representative_universe=(),
    )
    _skip_disk_rehash(monkeypatch)

    with pytest.raises(FeatureGcgError, match="empty representative universe"):
        _optimize(
            model,
            forged,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )


def test_desynced_contract_representative_outside_vocab_names_missing_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: only rehash is patched; loaded-state validation is real."""

    contract, model, _ = _contract_model_pair(tmp_path)
    forged_universe = contract.consolidation.cluster_allowed_universe.clone()
    forged_universe[1] = True
    forged_consolidation = replace(
        contract.consolidation,
        cluster_allowed_universe=forged_universe,
    )
    forged = replace(
        contract,
        consolidation=forged_consolidation,
        representative_universe=(0, 1, 2, 3),
    )
    _skip_disk_rehash(monkeypatch)

    with pytest.raises(
        FeatureGcgError,
        match=r"IDs outside the checkpoint FeatureVocab: \[1\]",
    ):
        _optimize(
            model,
            forged,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )


def test_optimize_contract_rejects_non_mapping_metadata(
    tmp_path: Path,
) -> None:
    """No rehash patch is needed: metadata shape fails before checkpoint I/O."""

    contract, model, _ = _contract_model_pair(tmp_path)
    model._loaded_meta_data = []

    with pytest.raises(
        FeatureGcgError,
        match="optimize checkpoint is missing mapping metadata",
    ):
        _optimize(
            model,
            contract,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )


def test_optimize_wraps_contract_error_and_preserves_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is replaced by a sentinel contract failure."""

    contract, model, _ = _contract_model_pair(tmp_path)
    cause = ClusterFeatureContractError("sentinel contract failure")

    def _fail(_contract: ClusterFeatureContract) -> None:
        raise cause

    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        _fail,
    )

    with pytest.raises(FeatureGcgError, match="sentinel contract failure") as exc_info:
        _optimize(
            model,
            contract,
            warm_start=FeatureConditioning(entries=((0, 0),)),
        )
    assert exc_info.value.__cause__ is cause


def test_contract_validation_precedes_partition_objective_and_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rehash sentinel proves ordering before loaded-state or downstream work."""

    contract, model, _ = _contract_model_pair(tmp_path)
    calls = {"rehash": 0, "partition": 0, "objective": 0, "forward": 0}

    def _fail_rehash(_contract: ClusterFeatureContract) -> None:
        calls["rehash"] += 1
        raise ClusterFeatureContractError("ordering sentinel")

    def _unexpected_partition(*_args: Any, **_kwargs: Any) -> Any:
        calls["partition"] += 1
        raise AssertionError("contract failure reached partition parsing")

    def _unexpected_objective(_model: Any) -> Any:
        calls["objective"] += 1
        raise AssertionError("contract failure reached objective construction")

    def _unexpected_forward(*_args: Any, **_kwargs: Any) -> Any:
        calls["forward"] += 1
        raise AssertionError("contract failure reached a model forward")

    model.forward = _unexpected_forward
    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        _fail_rehash,
    )
    monkeypatch.setattr(
        optimize_module,
        "partition_bounds_from_model",
        _unexpected_partition,
    )

    with pytest.raises(FeatureGcgError, match="ordering sentinel"):
        optimize_module.optimize_feature_conditioning_universal(
            model,
            [object()],
            FeatureConditioning(entries=((0, 0),)),
            objective_factory=_unexpected_objective,
            config=FeatureGcgConfig(),
            cluster_contract=contract,
        )
    assert calls == {"rehash": 1, "partition": 0, "objective": 0, "forward": 0}


def test_valid_clustered_and_unclustered_paths_derive_the_same_vocab_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only disk rehash is patched; sanctioned V equals representatives exactly."""

    contract, clustered_model, _ = _contract_model_pair(tmp_path)
    _skip_disk_rehash(monkeypatch)
    result = FeatureConditioning(entries=((0, 0),))
    captured = _stub_search(monkeypatch, result=result)
    legacy_model = SimpleNamespace(
        _loaded_meta_data={
            "user_config": {
                "negative_selection": {
                    "distribution": "geometric",
                    "mean": 1.0,
                }
            }
        },
        feature_vocab=clustered_model.feature_vocab,
        encoder=clustered_model.encoder,
        assert_vocab_consistent=lambda: None,
    )

    optimize_module.optimize_feature_conditioning_universal(
        legacy_model,
        [object()],
        result,
        objective_factory=lambda _model: cast(Any, object()),
        config=FeatureGcgConfig(),
    )
    _optimize(clustered_model, contract, warm_start=result)

    expected = tuple(clustered_model.encoder.feature_vocab_ids.tolist())
    assert expected == contract.representative_universe
    assert isinstance(
        captured[0]["allowed_feature_ids"], NonEmptyNonNegativeIntTuple
    )
    assert isinstance(
        captured[1]["allowed_feature_ids"], NonEmptyNonNegativeIntTuple
    )
    assert captured[0]["allowed_feature_ids"] == expected
    assert captured[1]["allowed_feature_ids"] == expected


def test_none_contract_keeps_optimize_rejection_type_and_message_exact() -> None:
    model = SimpleNamespace(_loaded_meta_data={"feature_clusters_canonical": {}})

    def _call(*, explicit_none: bool) -> BaseException:
        kwargs = {"cluster_contract": None} if explicit_none else {}
        with pytest.raises(BaseException) as exc_info:
            optimize_module.optimize_feature_conditioning_universal(
                model,
                [],
                FeatureConditioning(entries=()),
                objective_factory=lambda _model: cast(Any, object()),
                config=FeatureGcgConfig(),
                **kwargs,
            )
        return exc_info.value

    omitted = _call(explicit_none=False)
    explicit = _call(explicit_none=True)
    expected_message = (
        "feature_gcg optimize checkpoint cannot consume a checkpoint/artifact "
        "trained with static feature clusters. Cluster semantics are currently "
        "supported only by precomputed encoder-decoder pretraining selections."
    )
    for actual in (omitted, explicit):
        assert type(actual) is FeatureClustersError
        assert str(actual) == expected_message


def _disk_backed_loaded_model(
    contract: ClusterFeatureContract,
    metadata: dict[str, Any],
) -> SimpleNamespace:
    feature_vocab = contract.checkpoint.feature_vocab
    assert feature_vocab is not None
    return SimpleNamespace(
        _loaded_meta_data=metadata,
        feature_vocab=feature_vocab,
        encoder=SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=contract.n_sae_features,
                num_buckets=contract.selection.num_buckets,
            ),
            feature_vocab_ids=t.tensor(feature_vocab.original_ids, dtype=t.int64),
        ),
        assert_vocab_consistent=lambda: None,
    )


def test_optimizer_executes_real_checkpoint_rehash_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 3: no contract-validation monkeypatch; checkpoint files are rehashed."""

    request, fake_sae = _write_full_preflight_fixture(tmp_path)
    contract = resolve_cluster_feature_contract(
        request,
        sae_loader=lambda _resolved: fake_sae,
    )
    metadata = json.loads(contract.checkpoint.metadata_file.path.read_text())
    model = _disk_backed_loaded_model(contract, metadata)
    result = FeatureConditioning(entries=((0, 0),))
    real_rehash = contract_module.assert_cluster_checkpoint_pin_unchanged
    rehash_calls = 0

    def _rehash_spy(actual_contract: ClusterFeatureContract) -> None:
        nonlocal rehash_calls
        rehash_calls += 1
        real_rehash(actual_contract)

    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        _rehash_spy,
    )
    monkeypatch.setattr(
        optimize_module,
        "run_gcg",
        lambda **_kwargs: [SimpleNamespace(payload={"bag": result})],
    )

    actual = optimize_module.optimize_feature_conditioning_universal(
        model,
        [object()],
        result,
        objective_factory=lambda _model: cast(Any, object()),
        config=FeatureGcgConfig(),
        cluster_contract=contract,
    )

    assert actual is result
    assert rehash_calls == 1


def test_optimizer_real_rehash_rejects_checkpoint_mutation(
    tmp_path: Path,
) -> None:
    """Tier 3 negative: no monkeypatch; changed checkpoint bytes fail before work."""

    request, fake_sae = _write_full_preflight_fixture(tmp_path)
    contract = resolve_cluster_feature_contract(
        request,
        sae_loader=lambda _resolved: fake_sae,
    )
    metadata = json.loads(contract.checkpoint.metadata_file.path.read_text())
    model = _disk_backed_loaded_model(contract, metadata)
    checkpoint_path = contract.checkpoint.model_file.path
    state = t.load(checkpoint_path, map_location="cpu", weights_only=True)
    state[FEATURE_EMBED_KEY][0, 0] = 1.0
    t.save(state, checkpoint_path)

    with pytest.raises(
        FeatureGcgError,
        match="clustered target checkpoint changed after preflight",
    ) as exc_info:
        optimize_module.optimize_feature_conditioning_universal(
            model,
            [object()],
            FeatureConditioning(entries=((0, 0),)),
            objective_factory=lambda _model: cast(Any, object()),
            config=FeatureGcgConfig(),
            cluster_contract=contract,
        )
    assert isinstance(exc_info.value.__cause__, ClusterFeatureContractError)
