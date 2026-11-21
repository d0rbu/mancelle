"""Tests for the shared checkpoint feature-universe contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from errors import FeatureClustersError, FeatureVocabError
from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    assert_model_matches_checkpoint_feature_contract,
    resolve_and_assert_model_feature_contract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt

N_SAE = PositiveInt.parse(64)
CHECKPOINT_DIR = "checkpoint"
MISSING_METADATA_LABEL = "feature-universe metadata"
MISSING_METADATA_GUIDANCE = "Re-export the checkpoint with canonical metadata."


def _mask() -> FeatureMask:
    return FeatureMask(FeatureMaskMode.whitelist, tuple(range(8, 24)))


def _resolved_clusters(*, n_sae_features: PositiveInt = N_SAE) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(
        clusters=((8, 9, 10), (12, 13), (20, 21)),
    )
    return ResolvedFeatureClusters(
        source_path=Path("cluster-source.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )


def _meta(*, mask: FeatureMask | None, clusters: ResolvedFeatureClusters | None) -> dict:
    meta = {}
    if mask is not None:
        meta["feature_mask_canonical"] = mask.canonical_dict()
    meta.update(feature_clusters_checkpoint_metadata(clusters))
    return meta


def _resolve(
    *,
    mask: FeatureMask | None = None,
    clusters: ResolvedFeatureClusters | None = None,
) -> CheckpointFeatureContract:
    return resolve_checkpoint_feature_contract(
        _meta(mask=mask, clusters=clusters),
        n_sae_features=N_SAE,
        surface="unit checkpoint",
        source_path="checkpoint/meta.json",
    )


def _assert_matches(
    feature_vocab: FeatureVocab | None,
    contract: CheckpointFeatureContract,
) -> None:
    assert_model_matches_checkpoint_feature_contract(
        feature_vocab,
        contract,
        n_sae_features=N_SAE,
        checkpoint_dir=CHECKPOINT_DIR,
        surface="unit",
        missing_metadata_label=MISSING_METADATA_LABEL,
        missing_metadata_guidance=MISSING_METADATA_GUIDANCE,
    )


class _LoadedModel:
    def __init__(
        self,
        feature_vocab: FeatureVocab | None,
        *,
        n_sae_features: object = N_SAE,
    ) -> None:
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(n_sae_features=n_sae_features),
        )
        self.feature_vocab = feature_vocab
        self.consistency_calls = 0

    def assert_vocab_consistent(self) -> None:
        self.consistency_calls += 1


def _resolve_and_assert(model: object) -> CheckpointFeatureContract:
    return resolve_and_assert_model_feature_contract(
        model,
        _meta(mask=_mask(), clusters=None),
        checkpoint_dir=CHECKPOINT_DIR,
        surface="unit",
        missing_metadata_label=MISSING_METADATA_LABEL,
        missing_metadata_guidance=MISSING_METADATA_GUIDANCE,
    )


def test_legacy_metadata_accepts_full_width_model() -> None:
    contract = _resolve()

    assert contract == CheckpointFeatureContract(None, None)
    _assert_matches(None, contract)


def test_mask_only_metadata_accepts_matching_derived_vocabulary() -> None:
    contract = _resolve(mask=_mask())
    expected = effective_feature_vocab(contract.universe, N_SAE)

    assert expected is not None
    _assert_matches(expected, contract)


def test_mask_only_metadata_rejects_vocabulary_with_one_extra_id() -> None:
    contract = _resolve(mask=_mask())
    expected = effective_feature_vocab(contract.universe, N_SAE)
    assert expected is not None
    extra_id = next(feature_id for feature_id in range(N_SAE) if feature_id not in expected.original_ids)
    observed = FeatureVocab(N_SAE, (*expected.original_ids, extra_id))

    with pytest.raises(FeatureVocabError, match="does not match"):
        _assert_matches(observed, contract)


def test_clustered_mask_metadata_accepts_cluster_effective_vocabulary() -> None:
    contract = _resolve(mask=_mask(), clusters=_resolved_clusters())
    expected = effective_feature_vocab(contract.universe, N_SAE)

    assert expected is not None
    _assert_matches(expected, contract)


def test_clustered_mask_metadata_rejects_pre_cluster_mask_only_vocabulary() -> None:
    mask = _mask()
    contract = _resolve(mask=mask, clusters=_resolved_clusters())
    mask_only = effective_feature_vocab(mask, N_SAE)
    assert mask_only is not None

    with pytest.raises(FeatureVocabError, match="retrained"):
        _assert_matches(mask_only, contract)


def test_compacted_vocabulary_requires_metadata_that_explains_compaction() -> None:
    compacted = effective_feature_vocab(_mask(), N_SAE)
    assert compacted is not None

    with pytest.raises(FeatureVocabError, match=MISSING_METADATA_LABEL):
        _assert_matches(compacted, _resolve())


def test_vocabulary_original_width_must_match_model_width() -> None:
    other_width = PositiveInt.parse(N_SAE + 1)
    observed = effective_feature_vocab(_mask(), other_width)
    assert observed is not None

    with pytest.raises(FeatureVocabError, match="n_sae"):
        _assert_matches(observed, _resolve(mask=_mask()))


def test_cluster_descriptor_width_must_match_model_width() -> None:
    other_width = PositiveInt.parse(N_SAE + 1)

    with pytest.raises(FeatureClustersError, match="n_sae_features"):
        _resolve(mask=_mask(), clusters=_resolved_clusters(n_sae_features=other_width))


def test_consolidation_accesses_are_equal_but_not_identical() -> None:
    contract = _resolve(mask=_mask(), clusters=_resolved_clusters())

    first = contract.consolidation
    second = contract.consolidation
    assert isinstance(first, FeatureClusterConsolidation)
    assert first == second
    assert first is not second


def test_model_contract_resolution_rejects_missing_feature_vocab_attribute() -> None:
    model = SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=N_SAE)),
        assert_vocab_consistent=lambda: None,
    )

    with pytest.raises(FeatureVocabError, match="model.feature_vocab"):
        _resolve_and_assert(model)


def test_model_contract_resolution_rejects_missing_consistency_assertion() -> None:
    feature_vocab = effective_feature_vocab(_mask(), N_SAE)
    model = SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=N_SAE)),
        feature_vocab=feature_vocab,
    )

    with pytest.raises(FeatureVocabError, match="assert_vocab_consistent"):
        _resolve_and_assert(model)


def test_model_contract_resolution_rejects_non_positive_sae_width() -> None:
    model = _LoadedModel(None, n_sae_features=0)

    with pytest.raises(FeatureVocabError, match="positive original SAE width"):
        _resolve_and_assert(model)


def test_model_contract_resolution_checks_consistency_once_on_success() -> None:
    feature_vocab = effective_feature_vocab(_mask(), N_SAE)
    model = _LoadedModel(feature_vocab)

    contract = _resolve_and_assert(model)

    assert contract.feature_mask == _mask()
    assert model.consistency_calls == 1
