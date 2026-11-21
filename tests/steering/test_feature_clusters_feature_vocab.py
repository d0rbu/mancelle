"""Static clusters size the encoder from the effective original-ID universe.

Selection artifacts continue to carry ORIGINAL SAE ids and the encoder remains the
only original-to-compact translation boundary.  Representatives retain rows;
consolidated-away members do not.  Slice initialization still draws the full SAE table,
so every original id shared by two effective vocabularies receives bitwise-identical row
values even when their table heights differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from errors import FeatureVocabError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

N_SAE = 16
MASK = FeatureMask(FeatureMaskMode.blacklist, (0, 2, 6, 15))


def _cluster_state(
    tmp_path: Path,
    groups: list[list[int]],
    *,
    name: str,
    mask: FeatureMask | None = MASK,
    n_sae_features: int = N_SAE,
) -> FeatureClusterConsolidation:
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": groups}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        n_sae_features,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _effective(state: FeatureClusterConsolidation) -> FeatureVocab:
    vocab = effective_feature_vocab(state, state.n_sae_features)
    assert vocab is not None
    return vocab


def _encoder(vocab: FeatureVocab, *, seed: int = 123) -> FeatureEncoder:
    config = FeatureEncoderConfig(
        n_sae_features=N_SAE,
        num_buckets=4,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )
    with t.device("meta"):
        encoder = FeatureEncoder(config, feature_vocab=vocab)
    encoder.to_empty(device="cpu")
    t.manual_seed(seed)
    encoder.init_weights()
    return encoder


def test_effective_vocab_drops_away_rows_and_keeps_representatives(tmp_path: Path):
    clusters = _cluster_state(
        tmp_path,
        [[1, 3, 4], [5, 7], [9, 10, 11]],
        name="clusters.json",
    )
    vocab = _effective(clusters)

    assert clusters.consolidated_away.nonzero().squeeze(1).tolist() == [3, 4, 7, 10, 11]
    assert vocab.original_ids == (1, 5, 8, 9, 12, 13, 14)
    assert vocab.to_compact(t.tensor([1, 5, 9])).tolist() == [0, 1, 3]
    with pytest.raises(FeatureVocabError, match="NOT in this encoder"):
        vocab.to_compact(t.tensor([3, 4, 7, 10, 11]))


def test_representatives_remap_once_at_encoder_boundary(tmp_path: Path):
    clusters = _cluster_state(
        tmp_path,
        [[1, 3, 4], [5, 7], [9, 10, 11]],
        name="clusters.json",
    )
    vocab = _effective(clusters)
    encoder = _encoder(vocab)
    original_ids = t.tensor([1, 5, 8, 9, 12, 0], dtype=t.int64)
    real_mask = t.tensor([True, True, True, True, True, False])

    compact = encoder._to_compact(original_ids, real_mask)

    assert compact.tolist() == [0, 1, 2, 3, 4, 0]
    assert t.equal(compact[:-1], vocab.to_compact(original_ids[:-1]))
    clusters.assert_representatives_only(
        original_ids[:-1],
        surface="pre-encoder clustered IDs",
    )


def test_cluster_content_changes_shape_but_not_shared_original_row_values(
    tmp_path: Path,
):
    clusters_a = _cluster_state(
        tmp_path,
        [[1, 3, 4], [5, 7]],
        name="a.json",
    )
    clusters_b = _cluster_state(
        tmp_path,
        [[1, 5]],
        name="b.json",
    )
    assert clusters_a.resolved.descriptor != clusters_b.resolved.descriptor
    vocab_a = _effective(clusters_a)
    vocab_b = _effective(clusters_b)
    assert vocab_a.size == 9
    assert vocab_b.size == 11

    encoder_a = _encoder(vocab_a, seed=999)
    encoder_b = _encoder(vocab_b, seed=999)
    shared = sorted(set(vocab_a.original_ids).intersection(vocab_b.original_ids))
    for original_id in shared:
        row_a = vocab_a.original_ids.index(original_id)
        row_b = vocab_b.original_ids.index(original_id)
        assert t.equal(
            encoder_a.feature_embed.weight[row_a],
            encoder_b.feature_embed.weight[row_b],
        )
        for key in encoder_a.value_embeds:
            assert t.equal(
                encoder_a.value_embeds[key].weight[row_a],  # ty: ignore[not-subscriptable]
                encoder_b.value_embeds[key].weight[row_b],  # ty: ignore[not-subscriptable]
            )


def test_mask_only_constructor_remains_explicitly_available(tmp_path: Path):
    """The migration/reference shape remains constructible, but is not production."""
    clusters = _cluster_state(
        tmp_path,
        [[1, 3, 4], [5, 7]],
        name="clusters.json",
    )
    mask_only = FeatureVocab.from_mask(MASK, N_SAE)
    effective = _effective(clusters)
    assert mask_only is not None
    assert mask_only.size == 12
    assert all(feature_id in mask_only.original_ids for feature_id in (3, 4, 7))
    assert all(feature_id not in effective.original_ids for feature_id in (3, 4, 7))


def test_mask_changes_cluster_representative_and_effective_compact_rank(
    tmp_path: Path,
):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3, 4]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        N_SAE,
    )
    assert resolved is not None
    unmasked = FeatureClusterConsolidation.build(resolved, None)
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 2))
    masked = FeatureClusterConsolidation.build(resolved, mask)
    unmasked_ids, _ = unmasked.consolidate_segment(
        t.tensor([3, 4]),
        t.tensor([0.8, 0.9]),
    )
    masked_ids, _ = masked.consolidate_segment(
        t.tensor([3, 4]),
        t.tensor([0.8, 0.9]),
    )
    assert unmasked_ids.tolist() == [1]
    assert masked_ids.tolist() == [3]

    masked_vocab = _effective(masked)
    assert masked_vocab.to_compact(masked_ids).tolist() == [1]
    assert masked_ids.item() != masked_vocab.to_compact(masked_ids).item()
    with pytest.raises(FeatureVocabError, match="NOT in this encoder"):
        masked_vocab.to_compact(t.tensor([4]))


def test_real_sae_width_effective_vocab_preserves_original_uint16_ceiling(
    tmp_path: Path,
):
    n_sae = 65_536
    mask = FeatureMask(FeatureMaskMode.blacklist, (0, 65_530))
    clusters = _cluster_state(
        tmp_path,
        [[65_530, 65_534, 65_535], [100, 20_000]],
        name="real_width_clusters.json",
        mask=mask,
        n_sae_features=n_sae,
    )
    vocab = _effective(clusters)
    assert vocab.size == 65_532
    assert vocab.original_ids[-1] == 65_534

    original_ids, pcts = clusters.consolidate_segment(
        t.tensor([65_534, 65_535, 20_000], dtype=t.int64),
        t.tensor([0.7, 0.99, 0.5]),
    )
    assert original_ids.tolist() == [100, 65_534]
    assert t.equal(pcts, t.tensor([0.5, 0.99]))
    compact_ids = vocab.to_compact(original_ids.to(t.uint16))
    assert compact_ids.tolist() == [99, 65_531]
    assert vocab.to_original(compact_ids).tolist() == original_ids.tolist()
    with pytest.raises(FeatureVocabError, match="NOT in this encoder"):
        vocab.to_compact(t.tensor([65_535], dtype=t.int64))
