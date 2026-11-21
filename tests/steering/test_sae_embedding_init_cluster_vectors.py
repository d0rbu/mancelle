"""Cluster-vector semantics for SAE-backed encoder embedding initialization."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.sae_embedding_init as init_mod
from errors import FeatureClustersError
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    _resolved_user_config,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingClusterVectors,
    EncoderEmbeddingInit,
    _apply_cluster_means,
    initialize_encoder_feature_embeddings_from_matrix,
    maybe_initialize_encoder_feature_embeddings_from_sae,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

REPO_ROOT = Path(__file__).parents[2]
TRAINING_CONFIG_DIR = str(REPO_ROOT / "configs" / "training")


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _cluster_state(
    n_sae_features: int,
    clusters: tuple[tuple[int, ...], ...],
    feature_mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=clusters)
    resolved = ResolvedFeatureClusters(
        source_path=Path("<sae-init-cluster-vectors>"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, feature_mask)


def _embedding_model(
    n_sae_features: int,
    encoder_dim: int,
    vocab: FeatureVocab | None,
    *,
    full_avg_norm: float = 1.75,
):
    vocab_size = n_sae_features if vocab is None else vocab.size
    feature_embed = t.nn.Embedding(vocab_size, encoder_dim)
    values = t.arange(1, vocab_size * encoder_dim + 1, dtype=t.float32).reshape(
        vocab_size,
        encoder_dim,
    )
    with t.no_grad():
        feature_embed.weight.copy_(values / 17.0)
    return types.SimpleNamespace(
        encoder=types.SimpleNamespace(
            feature_embed=feature_embed,
            feature_vocab=vocab,
            feature_embed_full_avg_row_norm=t.tensor(full_avg_norm),
            config=types.SimpleNamespace(n_sae_features=n_sae_features),
        )
    )


def _normalized_random_reference(
    matrix: t.Tensor,
    encoder_dim: int,
    *,
    seed: int,
    target_avg_norm: float | t.Tensor,
) -> t.Tensor:
    """Independently spell out orient -> seeded projection -> full-table rescale."""
    matrix = matrix.detach().to(device="cpu", dtype=t.float32)
    generator = t.Generator(device="cpu").manual_seed(seed)
    projection = t.randn(
        matrix.shape[1],
        encoder_dim,
        generator=generator,
        dtype=t.float32,
    ) / (matrix.shape[1] ** 0.5)
    projected = matrix @ projection
    target = t.as_tensor(target_avg_norm, dtype=t.float32)
    return projected * (target / projected.norm(dim=1).mean())


def _full_model(
    n_sae_features: int,
    vocab: FeatureVocab | None,
    *,
    seed: int,
    device: str = "cpu",
    double_init: bool = False,
) -> EncoderDecoderGPT:
    with t.device("meta"):
        model = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=8,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                window_pattern="L",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=2,
                n_kv_head=2,
                head_dim=16,
            ),
            CrossAttentionConfig(frequency=1, n_head=2, n_kv_head=2),
            n_sae_features=n_sae_features,
            num_buckets=4,
            feature_vocab=vocab,
        )
    model.to_empty(device=device)
    for _ in range(2 if double_init else 1):
        t.manual_seed(seed)
        model.init_weights()
    model.eval()
    return model


def _compact_row(vocab: FeatureVocab | None, original_id: int) -> int:
    return original_id if vocab is None else vocab.original_ids.index(original_id)


def test_mean_matches_a_hand_computed_post_rescale_pre_gather_reference():
    n_sae = 7
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 4, 5, 6))
    state = _cluster_state(
        n_sae,
        ((0, 2), (1, 3, 4), (5, 6)),
        feature_mask,
    )
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    model = _embedding_model(n_sae, 2, vocab, full_avg_norm=1.75)
    matrix = t.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 1.0],
            [0.5, 1.5, 2.5],
            [90.0, 80.0, 70.0],
            [3.0, 5.0, 7.0],
            [2.0, 4.0, 8.0],
            [1.0, 3.0, 9.0],
        ],
        dtype=t.float32,
    )

    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=23,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    normalized = _normalized_random_reference(
        matrix,
        2,
        seed=23,
        target_avg_norm=1.75,
    )
    expected = normalized.clone()
    expected[0] = normalized[t.tensor([0, 2])].mean(dim=0)
    expected[1] = normalized[t.tensor([1, 4])].mean(dim=0)
    expected[5] = normalized[t.tensor([5, 6])].mean(dim=0)
    expected = expected[t.tensor(vocab.original_ids)]

    assert t.equal(model.encoder.feature_embed.weight, expected)
    assert result.cluster_vectors == "mean"
    assert result.clusters_averaged == 3
    assert result.members_averaged == 6


def test_svd_mean_matches_hand_computed_averaged_dictionary_basis_reference():
    """Independently pin orient -> mean -> select -> eigh -> project -> rescale."""
    n_sae = 8
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 4, 5, 6, 7))
    state = _cluster_state(
        n_sae,
        ((0, 2), (1, 3, 4), (5, 6)),
        feature_mask,
    )
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    assert vocab.original_ids == (0, 1, 5, 7)
    model = _embedding_model(n_sae, 3, vocab, full_avg_norm=1.75)
    canonical = t.tensor(
        [
            [1.0, 2.0, 3.0, 5.0, 7.0],
            [2.0, 1.0, 4.0, 3.0, 6.0],
            [3.0, 5.0, 2.0, 1.0, 4.0],
            [101.0, 103.0, 107.0, 109.0, 113.0],
            [7.0, 2.0, 6.0, 4.0, 1.0],
            [4.0, 8.0, 1.0, 2.0, 5.0],
            [6.0, 3.0, 7.0, 9.0, 2.0],
            [5.0, 7.0, 8.0, 6.0, 3.0],
        ],
        dtype=t.float32,
    )
    transposed_input = canonical.T.contiguous()

    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        transposed_input,
        EncoderEmbeddingInit.SVD,
        seed=999,  # svd ignores the seed
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    # Independent reference. Do not call orient_sae_decoder_matrix,
    # _apply_cluster_means, project_sae_decoder_matrix, or the norm helpers.
    oriented = transposed_input.T.contiguous().to(device="cpu", dtype=t.float32)
    averaged = oriented.clone()
    averaged[0] = oriented[t.tensor([0, 2])].mean(dim=0)
    averaged[1] = oriented[t.tensor([1, 4])].mean(dim=0)
    averaged[5] = oriented[t.tensor([5, 6])].mean(dim=0)
    selected = averaged[t.tensor(vocab.original_ids)]
    covariance = selected.T @ selected
    _eigvals, eigvecs = t.linalg.eigh(covariance)
    basis = eigvecs[:, -3:].flip(dims=(1,)).contiguous()
    projected = selected @ basis
    projected_avg_norm = projected.norm(dim=1).mean()
    expected = projected * (t.tensor(1.75) / projected_avg_norm)

    assert t.equal(model.encoder.feature_embed.weight, expected)
    assert result.cluster_vectors is EncoderEmbeddingClusterVectors.MEAN
    assert result.clusters_averaged == 3
    assert result.members_averaged == 6
    assert result.projected_avg_norm == float(projected_avg_norm.item())
    assert result.final_avg_norm == float(expected.norm(dim=1).mean().item())


@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize(
    "mode",
    [EncoderEmbeddingInit.RANDOM, EncoderEmbeddingInit.SVD],
    ids=["random", "svd"],
)
def test_representative_is_bitwise_the_pre_option_path(
    compacted: bool,
    mode: EncoderEmbeddingInit,
):
    n_sae = 8
    state = _cluster_state(n_sae, ((1, 2, 3),))
    vocab = effective_feature_vocab(state, n_sae) if compacted else None
    baseline = _embedding_model(n_sae, 3, vocab)
    explicit = _embedding_model(n_sae, 3, vocab)
    matrix = t.randn(n_sae, 5, generator=t.Generator().manual_seed(912))

    baseline_result = initialize_encoder_feature_embeddings_from_matrix(
        baseline,
        matrix,
        mode,
        seed=7,
    )
    explicit_result = initialize_encoder_feature_embeddings_from_matrix(
        explicit,
        matrix,
        mode,
        seed=7,
        cluster_vectors=EncoderEmbeddingClusterVectors.REPRESENTATIVE,
        feature_clusters=state if compacted else None,
    )

    # Bitwise-identical tables: the default (mean, no clusters threaded) degenerates
    # to the representative pipeline exactly. Only the DECLARED policy fields differ.
    assert t.equal(baseline.encoder.feature_embed.weight, explicit.encoder.feature_embed.weight)
    assert baseline_result.cluster_vectors is EncoderEmbeddingClusterVectors.MEAN
    assert explicit_result.cluster_vectors is EncoderEmbeddingClusterVectors.REPRESENTATIVE
    for field in (
        "mode",
        "n_sae_features",
        "sae_dim",
        "encoder_dim",
        "target_avg_norm",
        "projected_avg_norm",
        "final_avg_norm",
        "vocab_size",
        "clusters_averaged",
        "members_averaged",
    ):
        assert getattr(baseline_result, field) == getattr(explicit_result, field)
    assert explicit_result.clusters_averaged == 0
    assert explicit_result.members_averaged == 0


def test_mean_changes_only_surviving_representative_rows():
    n_sae = 8
    state = _cluster_state(n_sae, ((1, 2, 3), (5, 6)))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    representative = _embedding_model(n_sae, 3, vocab)
    mean = _embedding_model(n_sae, 3, vocab)
    matrix = t.arange(1, n_sae * 5 + 1, dtype=t.float32).reshape(n_sae, 5)

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=19,
    )
    initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=19,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    for original_id in (0, 4, 7):
        row = _compact_row(vocab, original_id)
        assert t.equal(
            mean.encoder.feature_embed.weight[row],
            representative.encoder.feature_embed.weight[row],
        )
    # And the representative rows genuinely CHANGED — without this half, a mutant
    # deleting the averaging block entirely would pass the "only" claim vacuously.
    for original_id in (1, 5):
        row = _compact_row(vocab, original_id)
        assert not t.equal(
            mean.encoder.feature_embed.weight[row],
            representative.encoder.feature_embed.weight[row],
        )


def test_singleton_survivor_cluster_is_a_bitwise_noop():
    n_sae = 6
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 4, 5))
    state = _cluster_state(n_sae, ((1, 2, 3),), feature_mask)
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    representative = _embedding_model(n_sae, 3, vocab)
    mean = _embedding_model(n_sae, 3, vocab)
    matrix = t.arange(1, n_sae * 4 + 1, dtype=t.float32).reshape(n_sae, 4)

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=31,
    )
    result = initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=31,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    assert t.equal(mean.encoder.feature_embed.weight, representative.encoder.feature_embed.weight)
    assert result.clusters_averaged == 1
    assert result.members_averaged == 1


def test_svd_mean_with_only_singleton_survivors_is_bitwise_representative():
    n_sae = 8
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 4, 6, 7))
    state = _cluster_state(n_sae, ((1, 2, 3), (4, 5)), feature_mask)
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    assert state.is_identity_after_mask
    representative = _embedding_model(n_sae, 3, vocab)
    mean = _embedding_model(n_sae, 3, vocab)
    matrix = t.randn(n_sae, 5, generator=t.Generator().manual_seed(913))

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.SVD,
        seed=31,
        cluster_vectors=EncoderEmbeddingClusterVectors.REPRESENTATIVE,
        feature_clusters=state,
    )
    result = initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.SVD,
        seed=31,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    assert t.equal(mean.encoder.feature_embed.weight, representative.encoder.feature_embed.weight)
    assert result.clusters_averaged == 2
    assert result.members_averaged == 2


def test_active_svd_means_change_the_covariance_basis_and_initialized_table():
    n_sae = 8
    state = _cluster_state(n_sae, ((0, 1, 2), (4, 5)))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    representative = _embedding_model(n_sae, 3, vocab)
    mean = _embedding_model(n_sae, 3, vocab)
    matrix = t.randn(n_sae, 5, generator=t.Generator().manual_seed(914))

    representative_dictionary = matrix[t.tensor(vocab.original_ids)]
    averaged = matrix.clone()
    averaged[0] = matrix[t.tensor([0, 1, 2])].mean(dim=0)
    averaged[4] = matrix[t.tensor([4, 5])].mean(dim=0)
    mean_dictionary = averaged[t.tensor(vocab.original_ids)]
    _rep_eigvals, representative_eigvecs = t.linalg.eigh(
        representative_dictionary.T @ representative_dictionary
    )
    _mean_eigvals, mean_eigvecs = t.linalg.eigh(mean_dictionary.T @ mean_dictionary)
    representative_basis = representative_eigvecs[:, -3:].flip(dims=(1,)).contiguous()
    mean_basis = mean_eigvecs[:, -3:].flip(dims=(1,)).contiguous()
    assert not t.equal(mean_basis, representative_basis), "negative control: basis did not change"

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.SVD,
        seed=37,
        cluster_vectors=EncoderEmbeddingClusterVectors.REPRESENTATIVE,
        feature_clusters=state,
    )
    result = initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.SVD,
        seed=37,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    assert not t.equal(
        mean.encoder.feature_embed.weight, representative.encoder.feature_embed.weight
    )
    assert result.clusters_averaged == 2
    assert result.members_averaged == 5


def test_mean_uses_only_mask_surviving_members():
    n_sae = 5
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 3, 4))
    state = _cluster_state(n_sae, ((1, 2, 3),), feature_mask)
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    model = _embedding_model(n_sae, 2, vocab, full_avg_norm=2.0)
    matrix = t.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1000.0, 2000.0, 3000.0],
            [5.0, 7.0, 11.0],
            [13.0, 17.0, 19.0],
        ]
    )

    initialize_encoder_feature_embeddings_from_matrix(
        model,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=13,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    normalized = _normalized_random_reference(
        matrix,
        2,
        seed=13,
        target_avg_norm=2.0,
    )
    row = _compact_row(vocab, 1)
    expected_survivors = normalized[t.tensor([1, 3])].mean(dim=0)
    excluded_member_mean = normalized[t.tensor([1, 2, 3])].mean(dim=0)
    assert t.equal(model.encoder.feature_embed.weight[row], expected_survivors)
    assert not t.equal(model.encoder.feature_embed.weight[row], excluded_member_mean)


@pytest.mark.parametrize(
    "mode",
    [EncoderEmbeddingInit.RANDOM, EncoderEmbeddingInit.SVD],
    ids=["random", "svd"],
)
def test_mean_rejects_a_model_built_with_the_mask_only_vocab(mode: EncoderEmbeddingInit):
    n_sae = 8
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 3, 4, 5))
    state = _cluster_state(n_sae, ((1, 2, 3),), feature_mask)
    mask_only_vocab = FeatureVocab.from_mask(feature_mask, n_sae)
    assert mask_only_vocab is not None
    model = _embedding_model(n_sae, 3, mask_only_vocab)
    before = model.encoder.feature_embed.weight.detach().clone()

    with pytest.raises(
        FeatureClustersError,
        match="model feature vocabulary.*effective_feature_vocab.*expected V=4, got V=6",
    ):
        initialize_encoder_feature_embeddings_from_matrix(
            model,
            t.ones(1),  # the vocabulary guard must win before W_dec orientation
            mode,
            seed=3,
            cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
            feature_clusters=state,
        )
    assert t.equal(model.encoder.feature_embed.weight, before)


@pytest.mark.parametrize(
    "mode",
    [EncoderEmbeddingInit.RANDOM, EncoderEmbeddingInit.SVD],
    ids=["random", "svd"],
)
def test_mean_rejects_cluster_state_for_a_different_sae_width(mode: EncoderEmbeddingInit):
    state = _cluster_state(8, ((1, 2),))
    model = _embedding_model(9, 3, None)
    with pytest.raises(FeatureClustersError, match="SAE-width mismatch.*8.*9"):
        initialize_encoder_feature_embeddings_from_matrix(
            model,
            t.ones(1),  # the SAE-width guard must win before W_dec orientation
            mode,
            seed=3,
            cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
            feature_clusters=state,
        )


def test_svd_mean_rank_guard_uses_the_effective_vocabulary_row_count():
    n_sae = 8
    state = _cluster_state(n_sae, (tuple(range(7)),))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    assert vocab.original_ids == (0, 7)
    model = _embedding_model(n_sae, 4, vocab)
    before = model.encoder.feature_embed.weight.detach().clone()

    with pytest.raises(ValueError) as exc:
        initialize_encoder_feature_embeddings_from_matrix(
            model,
            t.randn(n_sae, 6, generator=t.Generator().manual_seed(915)),
            EncoderEmbeddingInit.SVD,
            seed=3,
            cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
            feature_clusters=state,
        )

    message = str(exc.value)
    assert "admits only 2 features" in message
    assert "encoder_dim=4" in message
    # This fixture has NO feature_mask — the narrowing is pure cluster
    # consolidation — so the diagnosis must speak of the EFFECTIVE vocabulary
    # and name BOTH levers, never a bare "widen the feature_mask" directive
    # (the operator here has no mask to widen; that wording was the exact trap
    # TestTheRankGuardDiagnosesTheRIGHTModel eliminated for the mask/no-mask
    # split).
    assert "effective feature vocabulary" in message
    assert "feature_clusters" in message
    assert "feature_mask" in message  # named as ONE of the two possible levers
    assert "Widen whichever input narrowed the vocabulary" in message
    assert t.equal(model.encoder.feature_embed.weight, before)


def test_mean_is_deterministic_and_does_not_advance_global_rng():
    n_sae = 8
    state = _cluster_state(n_sae, ((0, 2, 4), (3, 6)))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    first = _embedding_model(n_sae, 3, vocab)
    second = _embedding_model(n_sae, 3, vocab)
    matrix = t.arange(1, n_sae * 5 + 1, dtype=t.float32).reshape(n_sae, 5)
    t.manual_seed(8821)
    before = t.random.get_rng_state()

    initialize_encoder_feature_embeddings_from_matrix(
        first,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=123,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )
    initialize_encoder_feature_embeddings_from_matrix(
        second,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=123,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    assert t.equal(first.encoder.feature_embed.weight, second.encoder.feature_embed.weight)
    assert t.equal(t.random.get_rng_state(), before)


def test_mean_result_fields_and_log_report_averaged_support():
    n_sae = 8
    state = _cluster_state(n_sae, ((0, 2, 4), (1, 3)))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    model = _embedding_model(n_sae, 3, vocab)
    matrix = t.arange(1, n_sae * 5 + 1, dtype=t.float32).reshape(n_sae, 5)
    messages: list[str] = []

    def spy(template, *args, **kwargs):
        messages.append(str(template).format(*args, **kwargs))

    cfg = {
        "initialize_encoder_embeddings_from_sae": "random",
        "initialize_encoder_embeddings_cluster_vectors": "mean",
        "sae": {"model": "stub"},
    }
    with (
        patch.object(init_mod.logger, "info", spy),
        patch.object(init_mod, "load_sae_decoder_matrix_from_cfg", lambda _cfg: matrix),
    ):
        result = maybe_initialize_encoder_feature_embeddings_from_sae(
            model,
            cfg,
            seed=17,
            feature_clusters=state,
        )

    assert result is not None
    assert result.cluster_vectors == "mean"
    assert result.clusters_averaged == 2
    assert result.members_averaged == 5
    rendered = "\n".join(messages)
    assert "cluster_vectors=mean" in rendered
    assert "clusters_averaged=2" in rendered
    assert "members_averaged=5" in rendered


@st.composite
def _partition_cases(draw):
    n_sae = draw(st.integers(min_value=2, max_value=12))
    permutation = draw(st.permutations(tuple(range(n_sae))))
    first_width = draw(st.integers(min_value=2, max_value=n_sae))
    clusters: list[tuple[int, ...]] = [tuple(sorted(permutation[:first_width]))]
    offset = first_width
    while n_sae - offset >= 2 and draw(st.booleans()):
        width = draw(st.integers(min_value=2, max_value=n_sae - offset))
        clusters.append(tuple(sorted(permutation[offset : offset + width])))
        offset += width
    allowed = draw(
        st.sets(
            st.integers(min_value=0, max_value=n_sae - 1),
            min_size=1,
            max_size=n_sae,
        )
    )
    return n_sae, tuple(clusters), tuple(sorted(allowed))


@settings(max_examples=40, deadline=None)
@given(case=_partition_cases())
def test_mean_partition_property(case):
    n_sae, clusters, allowed_ids = case
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, allowed_ids)
    state = _cluster_state(n_sae, clusters, feature_mask)
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    model = _embedding_model(n_sae, 3, vocab, full_avg_norm=1.25)
    matrix = t.arange(1, n_sae * 5 + 1, dtype=t.float32).reshape(n_sae, 5) / 11.0

    initialize_encoder_feature_embeddings_from_matrix(
        model,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=29,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    normalized = _normalized_random_reference(
        matrix,
        3,
        seed=29,
        target_avg_norm=1.25,
    )
    expected = normalized.clone()
    allowed = set(allowed_ids)
    representatives: set[int] = set()
    for cluster in clusters:
        survivors = [feature_id for feature_id in cluster if feature_id in allowed]
        if not survivors:
            continue
        representative = survivors[0]
        representatives.add(representative)
        expected[representative] = normalized[t.tensor(survivors)].mean(dim=0)
        if len(survivors) == 1:
            assert t.equal(expected[representative], normalized[representative])

    actual = model.encoder.feature_embed.weight
    selected_expected = expected[t.tensor(vocab.original_ids)]
    assert t.equal(actual, selected_expected)
    for row, original_id in enumerate(vocab.original_ids):
        if original_id not in representatives:
            assert t.equal(actual[row], normalized[original_id])


def test_id_zero_representative_changes_content_but_not_padding_heavy_forward():
    n_sae = 8
    state = _cluster_state(n_sae, ((0, 1, 2),))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    assert vocab.original_ids[0] == 0
    representative = _full_model(n_sae, vocab, seed=701)
    mean = _full_model(n_sae, vocab, seed=701)
    matrix = t.arange(1, n_sae * 6 + 1, dtype=t.float32).reshape(n_sae, 6)

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=41,
    )
    initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=41,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )
    assert not t.equal(
        representative.encoder.feature_embed.weight[0],
        mean.encoder.feature_embed.weight[0],
    )

    token_ids = t.arange(8, dtype=t.int64).unsqueeze(0)
    feature_ids = t.zeros(1, 2, 8, dtype=t.int64)
    magnitudes = t.zeros_like(feature_ids)
    feature_mask = t.zeros_like(feature_ids, dtype=t.bool)
    feature_ids[0, 0, 0] = 4
    feature_ids[0, 1, 0] = 5
    magnitudes[:, :, 0] = 2
    feature_mask[:, :, 0] = True
    segment_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=t.int64)
    with t.no_grad():
        representative_logits = representative(
            token_ids,
            None,
            feature_ids,
            magnitudes,
            feature_mask,
            segment_map,
        )
        mean_logits = mean(
            token_ids,
            None,
            feature_ids,
            magnitudes,
            feature_mask,
            segment_map,
        )
    assert t.equal(representative_logits, mean_logits)


def test_default_checkpoint_user_config_bytes_match_the_pre_key_payload(
    hydra_compose,
):
    current = hydra_compose(TRAINING_CONFIG_DIR, "conditional_encdec_train")
    previous = OmegaConf.create(OmegaConf.to_container(current, resolve=True))
    del previous["initialize_encoder_embeddings_cluster_vectors"]  # ty: ignore[invalid-argument-type]

    current_bytes = json.dumps(
        _checkpoint_user_config(current),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    previous_bytes = json.dumps(
        _checkpoint_user_config(previous),  # ty: ignore[invalid-argument-type]
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert current_bytes == previous_bytes
    assert _resolved_user_config(current)["initialize_encoder_embeddings_cluster_vectors"] == "mean"
    assert "initialize_encoder_embeddings_cluster_vectors" not in _checkpoint_user_config(current)


@pytest.mark.parametrize(
    "raw",
    [None, "mean", "MEAN", "Mean", "null", "none"],
)
def test_checkpoint_user_config_omits_every_semantically_default_cluster_vector_spelling(raw):
    """The pop is judged by the PARSER, not a literal-value set.

    Every spelling the initializer treats as the default (now ``mean``) must be
    popped — a non-canonical casing that parses to mean but leaked into the
    checkpoint payload would change the metadata schema for a behaviorally
    vanilla run.
    """
    cfg = OmegaConf.create(
        {
            "depth": 12,
            "initialize_encoder_embeddings_cluster_vectors": raw,
        }
    )
    assert _checkpoint_user_config(cfg) == {"depth": 12}


def test_checkpoint_user_config_retains_representative_in_any_casing():
    for raw in ("representative", "REPRESENTATIVE", "Representative"):
        cfg = OmegaConf.create(
            {
                "depth": 12,
                "initialize_encoder_embeddings_cluster_vectors": raw,
            }
        )
        assert _checkpoint_user_config(cfg)["initialize_encoder_embeddings_cluster_vectors"] == raw


def test_checkpoint_user_config_raises_on_unparseable_cluster_vector_value():
    cfg = OmegaConf.create(
        {
            "depth": 12,
            "initialize_encoder_embeddings_cluster_vectors": "centroid",
        }
    )
    with pytest.raises(ValueError, match="representative or mean"):
        _checkpoint_user_config(cfg)


def test_checkpoint_user_config_retains_representative_cluster_vector_policy():
    cfg = OmegaConf.create(
        {
            "depth": 12,
            "initialize_encoder_embeddings_cluster_vectors": "representative",
        }
    )
    assert _checkpoint_user_config(cfg) == {
        "depth": 12,
        "initialize_encoder_embeddings_cluster_vectors": "representative",
    }


def test_maximal_compaction_mean_init_builds_and_forwards_with_one_row():
    n_sae = 8
    state = _cluster_state(n_sae, (tuple(range(n_sae)),))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    assert vocab.size == 1
    model = _full_model(n_sae, vocab, seed=303)
    matrix = t.arange(1, n_sae * 6 + 1, dtype=t.float32).reshape(n_sae, 6)

    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=11,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )
    assert model.encoder.feature_embed.weight.shape == (1, 32)
    assert result.clusters_averaged == 1
    assert result.members_averaged == n_sae

    token_ids = t.arange(8, dtype=t.int64).unsqueeze(0)
    feature_ids = t.zeros(1, 2, 4, dtype=t.int64)
    magnitudes = t.zeros_like(feature_ids)
    feature_mask = t.zeros_like(feature_ids, dtype=t.bool)
    feature_mask[:, :, 0] = True
    segment_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=t.int64)
    with t.no_grad():
        logits = model(
            token_ids,
            None,
            feature_ids,
            magnitudes,
            feature_mask,
            segment_map,
        )
    assert bool(t.isfinite(logits).all())


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_mean_under_production_bf16_cuda_double_init():
    """Exercise the CUDA-only dtype transition hidden by CPU single-init tests.

    Production initializes once while materializing, casts feature tables to
    ``COMPUTE_DTYPE`` on CUDA, then reinitializes the already-cast tables after the
    trainer reseed. The full-table norm captured by that second pass is therefore a
    bf16/CUDA value that CPU tests cannot reproduce.
    """
    n_sae = 64
    state = _cluster_state(n_sae, ((0, 1, 2, 3), (9, 10, 11)))
    vocab = effective_feature_vocab(state, n_sae)
    assert vocab is not None
    representative = _full_model(
        n_sae,
        vocab,
        seed=4242,
        device="cuda",
        double_init=True,
    )
    mean = _full_model(
        n_sae,
        vocab,
        seed=4242,
        device="cuda",
        double_init=True,
    )
    assert mean.encoder.feature_embed.weight.dtype != t.float32 or COMPUTE_DTYPE == t.float32
    captured_target = mean.encoder.feature_embed_full_avg_row_norm.detach().clone()
    matrix = t.arange(1, n_sae * 24 + 1, dtype=t.float32).reshape(n_sae, 24) / 97.0

    initialize_encoder_feature_embeddings_from_matrix(
        representative,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=37,
    )
    result = initialize_encoder_feature_embeddings_from_matrix(
        mean,
        matrix,
        EncoderEmbeddingInit.RANDOM,
        seed=37,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    normalized = _normalized_random_reference(
        matrix,
        32,
        seed=37,
        target_avg_norm=captured_target.cpu(),
    )
    representative_ids = {0: (0, 1, 2, 3), 9: (9, 10, 11)}
    for original_id in vocab.original_ids:
        row = _compact_row(vocab, original_id)
        if original_id in representative_ids:
            expected = normalized[t.tensor(representative_ids[original_id])].mean(dim=0)
            expected = expected.to(
                device="cuda",
                dtype=mean.encoder.feature_embed.weight.dtype,
            )
            assert t.equal(mean.encoder.feature_embed.weight[row], expected)
        else:
            assert t.equal(
                mean.encoder.feature_embed.weight[row],
                representative.encoder.feature_embed.weight[row],
            )
    assert result.target_avg_norm == float(captured_target.item())


class TestApplyClusterMeansPreGatherContract:
    """Direct pins on the PRE-GATHER matrix — the rows the vocabulary gather drops.

    The e2e tests can only observe post-gather rows, so a mutant that also wrote
    the mean into away rows (or every member row) was previously indistinguishable
    from the spec'd behavior. The helper makes `M[j] = N[j] for every
    non-representative row` a directly-asserted contract instead of an
    unobservability argument.
    """

    def test_away_and_nonmember_rows_are_verbatim_and_reps_are_means(self):
        n_sae = 8
        mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 5, 6, 7))
        state = _cluster_state(n_sae, ((1, 2, 4), (5, 6)), mask)
        source = t.arange(1, n_sae * 3 + 1, dtype=t.float32).reshape(n_sae, 3)
        pristine = source.clone()

        out, clusters_averaged, members_averaged = _apply_cluster_means(source, state)

        # Input never mutated (reads-from-pristine contract).
        assert t.equal(source, pristine)
        # Representatives: mean over mask-SURVIVING members only (4 is mask-excluded).
        assert t.equal(out[1], (pristine[1] + pristine[2]) / 2)
        assert t.equal(out[5], (pristine[5] + pristine[6]) / 2)
        # Away members and non-members: VERBATIM rows, pre-gather.
        for row in (0, 2, 3, 4, 6, 7):
            assert t.equal(out[row], pristine[row])
        assert clusters_averaged == 2
        assert members_averaged == 4

    def test_zero_survivor_cluster_is_skipped_and_uncounted(self):
        n_sae = 6
        mask = FeatureMask(FeatureMaskMode.whitelist, (0, 5))
        state = _cluster_state(n_sae, ((1, 2), (3, 4)), mask)
        source = t.randn(n_sae, 4)

        out, clusters_averaged, members_averaged = _apply_cluster_means(source, state)

        assert t.equal(out, source)
        assert clusters_averaged == 0
        assert members_averaged == 0


@settings(max_examples=25, deadline=None)
@given(case=_partition_cases())
def test_svd_mean_partition_property(case):
    """The svd staging contract, fuzzed: means land on oriented W_dec BEFORE the
    effective-vocab subset and covariance, and every step downstream matches an
    independently re-derived reference.

    The shared `_apply_cluster_means` is already fuzzed via the random-mode
    property; this fuzzes what is svd-SPECIFIC — the ordering (mean -> select ->
    covariance -> eigh -> project -> rescale) and the rank-guard interplay —
    which previously rested on hand-picked fixtures alone. Draws whose effective
    vocabulary is rank-deficient for the encoder dim assert the loud guard
    instead of skipping, so the guard's reachability is fuzzed too.
    """
    n_sae, clusters, allowed_ids = case
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, allowed_ids)
    state = _cluster_state(n_sae, clusters, feature_mask)
    vocab = effective_feature_vocab(state, n_sae)
    encoder_dim = 2
    sae_dim = 4
    matrix = t.arange(1, n_sae * sae_dim + 1, dtype=t.float32).reshape(n_sae, sae_dim) / 7.0

    v_eff = n_sae if vocab is None else vocab.size
    model = _embedding_model(n_sae, encoder_dim, vocab, full_avg_norm=1.5)

    if min(v_eff, sae_dim) < encoder_dim:
        with pytest.raises(ValueError, match="effective feature vocabulary|SAE decoder"):
            initialize_encoder_feature_embeddings_from_matrix(
                model,
                matrix,
                EncoderEmbeddingInit.SVD,
                seed=41,
                cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
                feature_clusters=state,
            )
        return

    initialize_encoder_feature_embeddings_from_matrix(
        model,
        matrix,
        EncoderEmbeddingInit.SVD,
        seed=41,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        feature_clusters=state,
    )

    # Independent reference: raw tensor ops only, no production helpers.
    averaged = matrix.clone()
    allowed = set(allowed_ids)
    for cluster in clusters:
        survivors = [feature_id for feature_id in cluster if feature_id in allowed]
        if not survivors:
            continue
        averaged[survivors[0]] = matrix[t.tensor(survivors)].mean(dim=0)
    rows = averaged if vocab is None else averaged[t.tensor(vocab.original_ids, dtype=t.int64)]
    covariance = rows.T @ rows
    eigvals, eigvecs = t.linalg.eigh(covariance)
    del eigvals
    basis = eigvecs[:, -encoder_dim:].flip(dims=(1,)).contiguous()
    projected = rows @ basis
    expected = projected * (t.tensor(1.5, dtype=t.float32) / projected.norm(dim=1).mean())

    assert t.equal(model.encoder.feature_embed.weight, expected)
