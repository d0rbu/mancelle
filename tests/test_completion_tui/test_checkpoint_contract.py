from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import scripts.completion_tui.checkpoint_contract as checkpoint_contract
from errors import (
    FeatureClustersError,
    FeatureMaskError,
    FeatureVocabError,
    GradientRankingUnsupportedError,
    RemovedFeatureError,
)
from scripts.completion_tui.checkpoint_contract import (
    CheckpointFeatureContract,
    assert_contract_supports_selection,
    assert_model_matches_contract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
N_SAE: PositiveInt = PositiveInt.parse(16)
_EMPTY_CONTRACT = CheckpointFeatureContract(
    feature_mask=None,
    resolved_clusters=None,
    consolidation=None,
    cluster_view=None,
)


def _resolved(n_sae_features: int = N_SAE) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=((1, 3, 5), (8, 9)))
    return ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=n_sae_features
        ),
    )


def _meta(
    *,
    mask: FeatureMask | None = None,
    clustered: bool = False,
) -> dict:
    metadata: dict = {}
    if mask is not None:
        metadata["feature_mask_canonical"] = mask.canonical_dict()
    if clustered:
        metadata.update(feature_clusters_checkpoint_metadata(_resolved()))
    return metadata


def _contract(
    *,
    mask: FeatureMask | None = None,
    clustered: bool = False,
):
    return resolve_checkpoint_feature_contract(
        _meta(mask=mask, clustered=clustered),
        n_sae_features=N_SAE,
        meta_path="checkpoint/meta_000001.json",
    )


def test_no_mask_no_clusters_resolves_identity_contract():
    contract = _contract()
    assert contract.feature_mask is None
    assert contract.resolved_clusters is None
    assert contract.consolidation is None
    assert contract.cluster_view is None
    assert not contract.clustered


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_checkpoint_width_consumers_reject_every_invalid_value(bad_width: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        resolve_checkpoint_feature_contract(
            {},
            n_sae_features=cast(Any, bad_width),
            meta_path="meta.json",
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        assert_model_matches_contract(
            None,
            _EMPTY_CONTRACT,
            n_sae_features=cast(Any, bad_width),
            checkpoint_dir="checkpoint",
        )


def test_invalid_checkpoint_width_fails_before_metadata_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_reads = 0

    def forbidden_metadata_read(_meta: object) -> None:
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("invalid width must fail before metadata access")

    monkeypatch.setattr(
        checkpoint_contract,
        "resolve_shared_checkpoint_feature_contract",
        forbidden_metadata_read,
    )

    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        resolve_checkpoint_feature_contract(
            {},
            n_sae_features=cast(Any, 0),
            meta_path="meta.json",
        )

    assert metadata_reads == 0


def test_mask_only_resolves_without_clusters():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    contract = _contract(mask=mask)
    assert contract.feature_mask == mask
    assert contract.consolidation is None


def test_clustered_contract_builds_consolidation_and_view():
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    contract = _contract(mask=mask, clustered=True)
    assert contract.clustered
    assert contract.consolidation is not None
    assert contract.cluster_view is not None
    assert contract.cluster_view.rep_to_members == {1: (1, 5), 8: (8, 9)}


def test_clustered_without_mask_is_supported():
    contract = _contract(clustered=True)
    assert contract.feature_mask is None
    assert contract.cluster_view is not None


def test_partial_descriptor_metadata_raises():
    metadata = feature_clusters_checkpoint_metadata(_resolved())
    metadata.pop("feature_clusters_canonical")
    with pytest.raises(FeatureClustersError, match="incomplete"):
        resolve_checkpoint_feature_contract(metadata, n_sae_features=N_SAE, meta_path="meta.json")


def test_user_config_only_metadata_raises():
    with pytest.raises(FeatureClustersError, match="incomplete"):
        resolve_checkpoint_feature_contract(
            {"user_config": {"feature_clusters": {"path": "x"}}},
            n_sae_features=N_SAE,
            meta_path="meta.json",
        )


def test_legacy_metadata_raises_removed_feature_error():
    with pytest.raises(RemovedFeatureError):
        resolve_checkpoint_feature_contract(
            {"feature_clustering_canonical": {"clusters": []}},
            n_sae_features=N_SAE,
            meta_path="meta.json",
        )


def test_mask_id_out_of_stats_range_raises():
    mask = FeatureMask(FeatureMaskMode.blacklist, (N_SAE,))
    with pytest.raises(FeatureMaskError, match="n_sae_features"):
        resolve_checkpoint_feature_contract(
            _meta(mask=mask), n_sae_features=N_SAE, meta_path="meta.json"
        )


def test_cluster_descriptor_stats_width_mismatch_raises():
    metadata = feature_clusters_checkpoint_metadata(_resolved(N_SAE + 1))
    with pytest.raises(FeatureClustersError, match="17.*16"):
        resolve_checkpoint_feature_contract(metadata, n_sae_features=N_SAE, meta_path="meta.json")


def test_clustered_selection_missing_raises():
    with pytest.raises(ValueError, match="require.*feature_selection"):
        assert_contract_supports_selection(_contract(clustered=True), {})


def test_clustered_splice_counterfactual_feature_delta_passes():
    assert_contract_supports_selection(
        _contract(clustered=True),
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 5.0,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            "feature_delta": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
        },
    )


@pytest.mark.parametrize("mode", ["prev_segment", "cum_max"])
def test_clustered_always_delta_modes_remain_rejected(mode: str):
    with pytest.raises(ValueError, match="produces a delta for every segment"):
        assert_contract_supports_selection(
            _contract(clustered=True),
            {
                "feature_selection": {
                    "distribution": "geometric",
                    "mean": 5.0,
                    "mode": "top_k_percentile",
                    "delta_weights": "percentile_delta",
                },
                "feature_delta": {"mode": mode},
            },
        )


@pytest.mark.parametrize("ratio", [0.0, 0.5, 0.999])
def test_clustered_partial_splice_delta_ratio_remains_rejected(ratio: float):
    with pytest.raises(ValueError, match="neg_delta_ratio=1.0"):
        assert_contract_supports_selection(
            _contract(clustered=True),
            {
                "feature_selection": {
                    "distribution": "geometric",
                    "mean": 5.0,
                    "mode": "top_k_percentile",
                    "delta_weights": "percentile_delta",
                },
                "feature_delta": {
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": ratio,
                },
            },
        )


def test_clustered_gradient_ranking_rejected_by_contract_gate():
    with pytest.raises(
        GradientRankingUnsupportedError,
        match="completion TUI live selection.*gradient-attribution ranking.*offline-only",
    ):
        assert_contract_supports_selection(
            _contract(clustered=True),
            {
                "feature_selection": {
                    "distribution": "geometric",
                    "mean": 5.0,
                    "mode": "top_k_percentile",
                    "gradient": {"mode": "future_half_split"},
                }
            },
        )


def test_clustered_delta_weights_without_feature_delta_rejected_as_forged():
    with pytest.raises(
        ValueError,
        match="delta_weights.*feature_delta.*forged",
    ):
        assert_contract_supports_selection(
            _contract(clustered=True),
            {
                "feature_selection": {
                    "distribution": "geometric",
                    "mean": 5.0,
                    "mode": "weighted",
                    "delta_weights": "absolute_percentile",
                }
            },
        )


@pytest.mark.parametrize("threshold_type", ["segment_topk", "segment_percentile"])
def test_clustered_raw_magnitude_threshold_raises(threshold_type: str):
    with pytest.raises(ValueError) as exc_info:
        assert_contract_supports_selection(
            _contract(clustered=True),
            {
                "feature_selection": {
                    "threshold_type": threshold_type,
                    "threshold": 1,
                }
            },
        )
    assert str(exc_info.value) == (
        "clustered completion TUI checkpoints do not support threshold type "
        f"{threshold_type!r}; supported percentile-only types are "
        "['feature_percentile', 'feature_percentile_segment_topk']"
    )
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("threshold_type", "threshold"),
    [
        ("feature_percentile", 0.9),
        ("feature_percentile_segment_topk", 2.9),
    ],
)
def test_clustered_percentile_thresholds_pass(threshold_type: str, threshold: float):
    assert_contract_supports_selection(
        _contract(clustered=True),
        {
            "feature_selection": {
                "threshold_type": threshold_type,
                "threshold": threshold,
            }
        },
    )


def test_clustered_sampling_selection_passes():
    assert_contract_supports_selection(
        _contract(clustered=True),
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "mode": "weighted",
            }
        },
    )


def test_unclustered_selection_gate_is_noop():
    assert_contract_supports_selection(_contract(), {})


def test_model_vocab_matches_mask_contract():
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    assert vocab is not None
    assert_model_matches_contract(
        vocab,
        _contract(mask=mask),
        n_sae_features=N_SAE,
        checkpoint_dir="checkpoint",
    )


def test_model_vocab_none_matches_unmasked_contract():
    assert_model_matches_contract(
        None,
        _contract(),
        n_sae_features=N_SAE,
        checkpoint_dir="checkpoint",
    )


def test_loaded_model_width_must_match_feature_statistics_width():
    with pytest.raises(
        FeatureVocabError,
        match="loaded model has original n_sae=17.*feature statistics have n_sae=16",
    ):
        assert_model_matches_contract(
            None,
            _contract(),
            n_sae_features=N_SAE,
            checkpoint_dir="checkpoint",
            loaded_model_n_sae_features=PositiveInt.parse(N_SAE + 1),
        )


def test_model_vocab_mismatch_names_observed_and_expected_sizes():
    expected_mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    observed_mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4, 6))
    observed = FeatureVocab.from_mask(observed_mask, N_SAE)
    assert observed is not None
    with pytest.raises(
        FeatureVocabError,
        match="observed V=13 of n_sae=16, expected V=14 of n_sae=16",
    ):
        assert_model_matches_contract(
            observed,
            _contract(mask=expected_mask),
            n_sae_features=N_SAE,
            checkpoint_dir="checkpoint",
        )


def test_clustered_model_vocab_matches_effective_vocabulary():
    """Clustered checkpoints carry the effective (cluster-compacted) vocab.

    Clusters ((1,3,5),(8,9)) under blacklist {2,4}: representatives 1 and 8
    keep rows; consolidated-away 3, 5, 9 do not — V_eff = 16 - 2 - 3 = 11.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    contract = _contract(mask=mask, clustered=True)
    assert contract.consolidation is not None
    effective = effective_feature_vocab(contract.consolidation, N_SAE)
    assert effective is not None
    assert effective.size == 11
    assert_model_matches_contract(
        effective,
        contract,
        n_sae_features=N_SAE,
        checkpoint_dir="checkpoint",
    )


def test_clustered_no_mask_model_vocab_matches_effective_vocabulary():
    """Clustered-without-mask skips the dump-remedy arm and matches effectively.

    Clusters ((1,3,5),(8,9)) with no mask: away members 3, 5, 9 drop —
    V_eff = 16 - 3 = 13. The compacted-model early raise is gated on
    `not contract.clustered`, so this legitimate state flows through the
    expected-vocab comparison.
    """
    contract = _contract(clustered=True)
    assert contract.feature_mask is None
    assert contract.consolidation is not None
    effective = effective_feature_vocab(contract.consolidation, N_SAE)
    assert effective is not None
    assert effective.size == 13
    assert_model_matches_contract(
        effective,
        contract,
        n_sae_features=N_SAE,
        checkpoint_dir="checkpoint",
    )


def test_clustered_no_mask_vocab_mismatch_uses_general_error_not_dump_remedy():
    contract = _contract(clustered=True)
    wrong = FeatureVocab.from_mask(FeatureMask(FeatureMaskMode.blacklist, (2,)), N_SAE)
    assert wrong is not None
    with pytest.raises(FeatureVocabError) as exc_info:
        assert_model_matches_contract(
            wrong,
            contract,
            n_sae_features=N_SAE,
            checkpoint_dir="checkpoint",
        )
    message = str(exc_info.value)
    assert "observed V=15 of n_sae=16, expected V=13 of n_sae=16" in message
    assert "dump_feature_mask" not in message


def test_clustered_model_with_stale_mask_only_vocab_rejected():
    """A pre-effective-vocab clustered checkpoint (mask-only rows) fails exactly."""
    mask = FeatureMask(FeatureMaskMode.blacklist, (2, 4))
    stale = FeatureVocab.from_mask(mask, N_SAE)
    assert stale is not None
    with pytest.raises(
        FeatureVocabError,
        match="observed V=14 of n_sae=16, expected V=11 of n_sae=16.*"
        "before cluster-aware compaction",
    ):
        assert_model_matches_contract(
            stale,
            _contract(mask=mask, clustered=True),
            n_sae_features=N_SAE,
            checkpoint_dir="checkpoint",
        )


def test_compacted_vocab_without_mask_canonical_names_dump_remedy():
    vocab = FeatureVocab.from_mask(FeatureMask(FeatureMaskMode.blacklist, (2,)), N_SAE)
    assert vocab is not None
    with pytest.raises(FeatureVocabError) as exc_info:
        assert_model_matches_contract(
            vocab,
            _contract(),
            n_sae_features=N_SAE,
            checkpoint_dir="checkpoint",
        )
    assert str(exc_info.value) == (
        "completion TUI checkpoint checkpoint has a compacted feature vocabulary "
        "(observed V=15 of n_sae=16) but lacks feature_mask_canonical metadata. "
        "Use scripts.analysis.dump_feature_mask to diagnose/recover the mask. "
        "The TUI requires canonical mask metadata in a newly published checkpoint; "
        "do not hand-edit the existing artifact."
    )
    assert exc_info.value.__cause__ is None
