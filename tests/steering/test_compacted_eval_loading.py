"""Tests for compacted-checkpoint loading on conditioned eval surfaces."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t

from errors import FeatureClustersError, FeatureVocabError
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.compacted_eval_loading import (
    CHAT_EVAL_SELF_CONDITIONED_SURFACE,
    CHAT_EVAL_TURN_SURFACE,
    CONDITIONED_EVAL_SURFACES,
    assert_conditioned_eval_model,
    conditioned_eval_conditioning,
    conditioned_eval_feature_vocab,
)
from steering.refined_types import PositiveInt

N_SAE = PositiveInt.parse(64)
CHECKPOINT_DIR = "checkpoint"


def _mask() -> FeatureMask:
    return FeatureMask(FeatureMaskMode.whitelist, tuple(range(8, 24)))


def _resolved_clusters() -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(
        clusters=((8, 9, 10), (12, 13), (20, 21)),
    )
    return ResolvedFeatureClusters(
        source_path=Path("cluster-source.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=N_SAE,
        ),
    )


def _meta(
    *,
    mask: FeatureMask | None,
    clusters: ResolvedFeatureClusters | None,
) -> dict:
    meta = {}
    if mask is not None:
        meta["feature_mask_canonical"] = mask.canonical_dict()
    meta.update(feature_clusters_checkpoint_metadata(clusters))
    return meta


class _EncDecStub(EncoderDecoderGPT):
    """Minimal real-type stub for the helper's intentional isinstance boundary."""

    def __init__(self, feature_vocab: FeatureVocab | None) -> None:
        t.nn.Module.__init__(self)
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(n_sae_features=N_SAE),
            feature_vocab=feature_vocab,
        )
        self.consistency_calls = 0

    def assert_vocab_consistent(self) -> None:
        self.consistency_calls += 1


def test_harmfulness_conditioning_allows_compacted_vocabulary() -> None:
    feature_vocab = effective_feature_vocab(_mask(), N_SAE)
    assert feature_vocab is not None

    conditioned_eval_conditioning("harmfulness_eval").assert_loadable(
        feature_vocab,
        checkpoint_dir=CHECKPOINT_DIR,
    )


def test_harmfulness_conditioning_allows_full_width_vocabulary() -> None:
    conditioned_eval_conditioning("harmfulness_eval").assert_loadable(
        None,
        checkpoint_dir=CHECKPOINT_DIR,
    )


@pytest.mark.parametrize("eval_name", ["base_eval", "chat_eval", "chat_eval_fixed", "chat_eval_turn_typo", ""])
def test_conditioning_rejects_surface_without_within_vocab_enforcement(eval_name) -> None:
    with pytest.raises(
        ValueError,
        match="harmfulness_eval.*training_progression",
    ):
        conditioned_eval_conditioning(eval_name)
    with pytest.raises(ValueError, match="within-vocabulary eval surfaces"):
        assert_conditioned_eval_model(object(), {}, checkpoint_dir=CHECKPOINT_DIR, eval_name=eval_name)


def test_chat_surfaces_are_narrowly_allowlisted():
    # kills: globally admitting static fixed ChatCORE or omitting the self-conditioned surface.
    assert CONDITIONED_EVAL_SURFACES == {"harmfulness_eval", "training_progression", CHAT_EVAL_TURN_SURFACE,
                                       CHAT_EVAL_SELF_CONDITIONED_SURFACE}
    for surface in (CHAT_EVAL_TURN_SURFACE, CHAT_EVAL_SELF_CONDITIONED_SURFACE):
        conditioned_eval_conditioning(surface).assert_loadable(
            effective_feature_vocab(_mask(), N_SAE), checkpoint_dir=CHECKPOINT_DIR)


def test_assert_conditioned_eval_model_ignores_decoder_only_model() -> None:
    assert (
        assert_conditioned_eval_model(
            object(),
            None,
            checkpoint_dir=CHECKPOINT_DIR,
            eval_name="harmfulness_eval",
        )
        is None
    )


@pytest.mark.parametrize("eval_name", sorted(CONDITIONED_EVAL_SURFACES))
def test_decoder_only_model_still_rejects_clustered_metadata(eval_name: str) -> None:
    """A conditioned checkpoint routed to a decoder-only build must fail loudly.

    Making these surfaces cluster-aware must not quietly drop the guard on the
    decoder branch: if branch selection ever mis-routes a conditioned
    checkpoint, generation would run UNCONDITIONED while the report still
    labels it conditioned.
    """
    clusters = _resolved_clusters()

    with pytest.raises(FeatureClustersError, match="static feature clusters"):
        assert_conditioned_eval_model(
            object(),
            _meta(mask=_mask(), clusters=clusters),
            checkpoint_dir=CHECKPOINT_DIR,
            eval_name=eval_name,
        )


def test_assert_conditioned_eval_model_returns_clustered_contract() -> None:
    mask = _mask()
    clusters = _resolved_clusters()
    contract_meta = _meta(mask=mask, clusters=clusters)
    expected_contract_vocab = effective_feature_vocab(
        FeatureClusterConsolidation.build(clusters, mask),
        N_SAE,
    )
    assert expected_contract_vocab is not None
    model = _EncDecStub(expected_contract_vocab)

    contract = assert_conditioned_eval_model(
        model,
        contract_meta,
        checkpoint_dir=CHECKPOINT_DIR,
        eval_name="harmfulness_eval",
    )

    assert isinstance(contract, CheckpointFeatureContract)
    assert contract.feature_mask == mask
    assert contract.resolved_clusters is not None
    assert contract.resolved_clusters.definition == clusters.definition
    assert contract.resolved_clusters.descriptor == clusters.descriptor
    assert contract.resolved_clusters.source_path == Path(CHECKPOINT_DIR)
    assert model.consistency_calls == 1


def test_assert_conditioned_eval_model_requires_compaction_metadata() -> None:
    feature_vocab = effective_feature_vocab(_mask(), N_SAE)
    assert feature_vocab is not None

    with pytest.raises(FeatureVocabError, match="feature-universe metadata"):
        assert_conditioned_eval_model(
            _EncDecStub(feature_vocab),
            {},
            checkpoint_dir=CHECKPOINT_DIR,
            eval_name="harmfulness_eval",
        )


def test_assert_conditioned_eval_model_rejects_cluster_vocab_mismatch() -> None:
    mask = _mask()
    mask_only_vocab = effective_feature_vocab(mask, N_SAE)
    assert mask_only_vocab is not None

    with pytest.raises(FeatureVocabError, match="retrained"):
        assert_conditioned_eval_model(
            _EncDecStub(mask_only_vocab),
            _meta(mask=mask, clusters=_resolved_clusters()),
            checkpoint_dir=CHECKPOINT_DIR,
            eval_name="training_progression",
        )


def test_conditioned_eval_feature_vocab_reflects_model_type() -> None:
    feature_vocab = effective_feature_vocab(_mask(), N_SAE)
    assert feature_vocab is not None

    assert conditioned_eval_feature_vocab(_EncDecStub(feature_vocab)) == feature_vocab
    assert conditioned_eval_feature_vocab(object()) is None
