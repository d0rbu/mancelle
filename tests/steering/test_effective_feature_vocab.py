"""Unit and property contract for the cluster-aware effective feature vocabulary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from errors import FeatureClustersError, FeatureVocabError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab


def _resolved(n_sae_features: int, clusters: list[list[int]]) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=tuple(tuple(cluster) for cluster in clusters))
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=n_sae_features,
    )
    return ResolvedFeatureClusters(
        source_path=Path("<unit-test-clusters>"),
        definition=definition,
        descriptor=descriptor,
    )


def _state(
    n_sae_features: int,
    clusters: list[list[int]],
    mask: FeatureMask | None,
) -> FeatureClusterConsolidation:
    return FeatureClusterConsolidation.build(
        _resolved(n_sae_features, clusters),
        mask,
    )


def _ids(vocab: FeatureVocab | None, n_sae_features: int) -> tuple[int, ...]:
    return tuple(range(n_sae_features)) if vocab is None else vocab.original_ids


@pytest.mark.parametrize(
    "mask,n_sae_features",
    [
        (None, 6),
        (FeatureMask(FeatureMaskMode.blacklist, (1, 4)), 6),
        (FeatureMask(FeatureMaskMode.whitelist, (1, 4)), 6),
        (FeatureMask(FeatureMaskMode.whitelist, tuple(range(6))), 6),
    ],
)
def test_no_clusters_matches_mask_only_constructor(
    mask: FeatureMask | None,
    n_sae_features: int,
):
    assert effective_feature_vocab(mask, n_sae_features) == FeatureVocab.from_mask(
        mask,
        n_sae_features,
    )


def test_no_clusters_is_literal_delegation_and_returns_the_same_object():
    sentinel = object()
    mask = FeatureMask(FeatureMaskMode.whitelist, (2,))
    with patch.object(FeatureVocab, "from_mask", return_value=sentinel) as delegated:
        result = effective_feature_vocab(mask, 5)
    assert result is sentinel
    delegated.assert_called_once_with(mask, 5)


def test_no_clusters_preserves_empty_allowed_error_text_exactly():
    mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(5)))
    with pytest.raises(FeatureVocabError) as legacy:
        FeatureVocab.from_mask(mask, 5)
    with pytest.raises(FeatureVocabError) as effective:
        effective_feature_vocab(mask, 5)
    assert str(effective.value) == str(legacy.value)


@pytest.mark.parametrize(
    "mask,clusters,expected",
    [
        (None, [[1, 3, 5]], (0, 1, 2, 4, 6, 7)),
        (
            FeatureMask(FeatureMaskMode.whitelist, (1, 3, 5, 6)),
            [[1, 3, 5]],
            (1, 6),
        ),
        (
            FeatureMask(FeatureMaskMode.blacklist, (1, 7)),
            [[1, 3, 5]],
            (0, 2, 3, 4, 6),
        ),
        (
            FeatureMask(FeatureMaskMode.whitelist, (0, 2, 7)),
            [[1, 3, 5]],
            (0, 2, 7),
        ),
        (
            FeatureMask(FeatureMaskMode.whitelist, (0, 3, 6)),
            [[1, 3, 5]],
            (0, 3, 6),
        ),
        (
            FeatureMask(FeatureMaskMode.blacklist, (0,)),
            [[1, 3], [2, 5, 7]],
            (1, 2, 4, 6),
        ),
    ],
)
def test_hand_derived_effective_universes_cover_mask_cluster_boundaries(
    mask: FeatureMask | None,
    clusters: list[list[int]],
    expected: tuple[int, ...],
):
    state = _state(8, clusters, mask)
    vocab = effective_feature_vocab(state, 8)
    assert _ids(vocab, 8) == expected


def test_user_example_ten_whitelisted_five_clustered_means_six_rows():
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(10)))
    state = _state(20, [[1, 2, 3, 4, 5]], mask)
    vocab = effective_feature_vocab(state, 20)
    assert vocab is not None
    assert vocab.size == 6
    assert vocab.original_ids == (0, 1, 6, 7, 8, 9)


def test_a_consolidation_carries_its_own_mask_so_drift_is_unrepresentable():
    """The union signature's point: no separate mask argument exists to disagree.

    A consolidation built against one mask derives the vocabulary of THAT mask —
    there is nothing a caller could pass alongside it to contradict it, which is
    what replaced the old two-argument drift guard.
    """
    built_mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3))
    state = _state(6, [[1, 2]], built_mask)
    vocab = effective_feature_vocab(state, 6)
    assert vocab is not None
    assert vocab.original_ids == (1, 3)  # built_mask's universe minus away member 2


def test_wrong_input_type_raises_type_error():
    with pytest.raises(TypeError, match="FeatureClusterConsolidation, a FeatureMask, or None"):
        effective_feature_vocab((1, 2, 3), 6)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_sae_width_consistency_guard_names_both_sides():
    state = _state(6, [[1, 2]], None)
    with pytest.raises(
        FeatureClustersError,
        match=r"feature_clusters\.n_sae_features=6, n_sae_features=7",
    ):
        effective_feature_vocab(state, 7)


@pytest.mark.parametrize(
    "mask",
    [
        None,
        FeatureMask(FeatureMaskMode.whitelist, tuple(range(6))),
    ],
)
def test_identity_cluster_universe_normalizes_to_none(mask: FeatureMask | None):
    """Pin normalization even for a defensively supplied identity consolidation.

    The strict schema requires every source cluster to have at least two members, so
    ordinary no-mask/all-allowing builds necessarily consolidate.  The constructor is
    public, however, and normalization must follow the universe rather than trust the
    redundant ``is_identity_after_mask`` flag.
    """
    ordinary = _state(6, [[1, 2]], mask)
    identity = replace(
        ordinary,
        rep_of=t.arange(6, dtype=t.int64),
        consolidated_away=t.zeros(6, dtype=t.bool),
        allowed_universe=t.ones(6, dtype=t.bool),
        cluster_allowed_universe=t.ones(6, dtype=t.bool),
        is_identity_after_mask=True,
    )
    assert effective_feature_vocab(identity, 6) is None


@settings(max_examples=150, deadline=None)
@given(data=st.data())
def test_effective_vocab_properties(data):
    n = data.draw(st.integers(min_value=2, max_value=96), label="n")
    assignments = [0, 0] + data.draw(
        st.lists(
            st.integers(min_value=-1, max_value=5),
            min_size=n - 2,
            max_size=n - 2,
        ),
        label="cluster_assignments",
    )
    clusters = [
        [feature_id for feature_id, assignment in enumerate(assignments) if assignment == group]
        for group in range(6)
    ]
    clusters = [cluster for cluster in clusters if len(cluster) >= 2]

    mode = data.draw(st.sampled_from(("none", "blacklist", "whitelist")), label="mode")
    if mode == "none":
        mask = None
    elif mode == "blacklist":
        mask_ids = data.draw(
            st.sets(st.integers(0, n - 1), min_size=1, max_size=n - 1),
            label="mask_ids",
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(mask_ids))
    else:
        mask_ids = data.draw(
            st.sets(st.integers(0, n - 1), min_size=1),
            label="mask_ids",
        )
        mask = FeatureMask(FeatureMaskMode.whitelist, tuple(mask_ids))

    state = _state(n, clusters, mask)
    vocab = effective_feature_vocab(state, n)
    effective_ids = _ids(vocab, n)
    allowed_ids = tuple(int(i) for i in state.allowed_universe.nonzero().flatten().tolist())
    away_ids = tuple(int(i) for i in state.consolidated_away.nonzero().flatten().tolist())

    assert effective_ids == tuple(sorted(set(effective_ids)))
    assert effective_ids == tuple(
        int(i) for i in state.cluster_allowed_universe.nonzero().flatten().tolist()
    )
    assert len(effective_ids) == len(allowed_ids) - len(away_ids)
    assert effective_ids
    assert min(effective_ids) == min(allowed_ids)

    effective_set = set(effective_ids)
    for cluster in state.resolved.definition.clusters:
        survivors = [feature_id for feature_id in cluster if feature_id in set(allowed_ids)]
        if survivors:
            assert min(survivors) in effective_set
    assert effective_set.isdisjoint(away_ids)


def test_maximal_compaction_builds_and_runs_a_one_row_encoder():
    n_sae_features = 8
    mask = FeatureMask(FeatureMaskMode.whitelist, (2, 3, 4, 5))
    state = _state(n_sae_features, [[2, 3, 4, 5]], mask)
    vocab = effective_feature_vocab(state, n_sae_features)
    assert vocab is not None
    assert vocab.size == 1
    assert vocab.original_ids == (2,)

    config = FeatureEncoderConfig(
        n_sae_features=n_sae_features,
        num_buckets=3,
        n_embd=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )
    with t.device("meta"):
        encoder = FeatureEncoder(config, feature_vocab=vocab)
    encoder.to_empty(device="cpu")
    t.manual_seed(7)
    encoder.init_weights()

    feature_ids = t.tensor([[[2, 0]]], dtype=t.int64)
    magnitudes = t.tensor([[[1, 0]]], dtype=t.int64)
    real_mask = t.tensor([[[True, False]]])
    encoded = encoder(feature_ids, magnitudes, real_mask)
    assert encoder.feature_embed.weight.shape == (1, 32)
    assert encoded.values.shape == (1, 32, 32)
    assert bool(encoded.layout.valid_mask[0])
    assert not encoded.layout.valid_mask[1:].any()


def test_e4_defensive_empty_universe_raise_fires_on_an_impossible_state():
    """E4's raise is unreachable through valid inputs — execute it via a forged state.

    `dataclasses.replace` keeps the frozen mask (so the wiring guards pass) while
    zeroing the universe, the one corruption the branch exists to catch. A mutant
    deleting or inverting the guard fails here and nowhere else.
    """
    mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, 1, 2))
    state = _state(8, [[0, 1]], mask)
    forged = replace(state, cluster_allowed_universe=t.zeros(8, dtype=t.bool))
    with pytest.raises(FeatureClustersError, match="is empty after applying"):
        effective_feature_vocab(forged, 8)


def test_e5_min_id_preservation_named_case():
    """E5 as a dedicated example: the smallest allowed id is clustered, and survives.

    Whitelist {3,4,5,9}; cluster [3,4,9] consolidates 4 and 9 away but keeps 3 —
    its own smallest allowed member — so compact row 0 denotes the same original id
    (3) in both the mask-only and effective vocabularies. Padding semantics hinge on
    exactly this.
    """
    mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(3, 4, 5, 9))
    state = _state(16, [[3, 4, 9]], mask)
    effective = effective_feature_vocab(state, 16)
    mask_only = FeatureVocab.from_mask(mask, 16)
    assert effective is not None and mask_only is not None
    assert effective.original_ids == (3, 5)
    assert effective.original_ids[0] == mask_only.original_ids[0] == 3


def test_maximal_compaction_with_no_mask_at_all_reaches_v_eff_one():
    """Edge-case 16, spec-literal variant: one cluster spans EVERY id, mask=None.

    The whole dictionary collapses onto representative 0: V_eff == 1 < n_sae, so the
    result is a genuine (non-identity) compaction with no mask involved anywhere.
    """
    n_sae_features = 8
    state = _state(n_sae_features, [list(range(n_sae_features))], None)
    vocab = effective_feature_vocab(state, n_sae_features)
    assert vocab is not None
    assert vocab.size == 1
    assert vocab.original_ids == (0,)
    assert FeatureVocab.from_mask(None, n_sae_features) is None
