"""Focused clustered-source semantics for AxBench Stage 1."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from steering.conditional.percentile_table import magnitude_to_percentile
from steering.eval.axbench_base.synth import positive_mode, qc
from steering.eval.axbench_base.types import Concept, SeedRow
from steering.eval.cluster_feature_contract import ClusterFeatureTarget
from tests.axbench_base.unit.test_config import make_valid_config


def _seed(doc_idx: int, prefix: str, feature_ids: list[int]) -> SeedRow:
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=6,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text=prefix,
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=feature_ids,
        sae_feature_mags=[1.0] * len(feature_ids),
        continuation_text="continuation",
        continuation_content_nanochat_tokens=[4, 5, 6],
    )


def _target() -> ClusterFeatureTarget:
    return ClusterFeatureTarget(
        representative_id=2,
        allowed_member_ids=(2, 5),
        projecting_source_ids=(2, 5),
        feature_contract_fingerprint="a" * 64,
    )


def _valid_stage1b_prefix(label: str) -> str:
    prefix = (
        f"{label}. This is realistic neutral source context used to exercise "
        "cluster-aware seed selection without violating Stage 1b bounds."
    )
    assert len(prefix) >= 100
    return prefix


def test_stage1_cluster_seed_cleanliness_and_dirty_ranking_use_all_members():
    concept = Concept(
        concept_id=0,
        feature_index=2,
        layer=8,
        sae_source="source",
        description="cluster target",
        corpus_count=100,
        description_source="neuronpedia",
    )
    clean = _valid_stage1b_prefix("clean")
    dirty_sibling_mid = _valid_stage1b_prefix("dirty-sibling-mid")
    dirty_representative_high = _valid_stage1b_prefix(
        "dirty-representative-high"
    )
    dirty_representative_low = _valid_stage1b_prefix(
        "dirty-representative-low"
    )
    seeds = [
        _seed(0, clean, [1, 4]),
        _seed(1, dirty_sibling_mid, [1, 5]),
        _seed(2, dirty_representative_high, [2, 7]),
        _seed(3, dirty_representative_low, [2, 8]),
    ]
    target = _target()

    class _PrefixCache:
        batch_size = 8

        def __init__(self) -> None:
            self.batch_size = 8
            self.calls: list[list[str]] = []

        def get_many(self, prefixes, feature_id, **kwargs):
            assert feature_id == 2
            assert kwargs == {"cluster_target": target}
            self.calls.append(list(prefixes))
            return [
                {
                    clean: 0.0,
                    dirty_sibling_mid: 50.0,
                    dirty_representative_high: 91.0,
                    dirty_representative_low: 12.0,
                }[prefix]
                for prefix in prefixes
            ]

    prefix_cache = _PrefixCache()
    cfg = make_valid_config(
        n_positives_per_concept=3,
        min_prefix_chars=1,
        accept_threshold=0.001,
        positive_prefix_headroom_ceiling=1.0,
    )
    tracker = positive_mode._prepare_positive_target_tracker(
        cfg,
        concept,
        seeds,
        base_model=object(),
        base_sae=object(),
        percentile_table=object(),
        sae_layer=8,
        prefix_cache=prefix_cache,
        cluster_target=target,
        effective_prefixes=positive_mode._resolve_stage1b_prefixes(
            cfg,
            seeds,
        ),
    )
    works, _next_order = positive_mode._admit_positive_work_items(
        tracker,
        start_order=0,
    )
    selected = [work.pending.selected for work in works]

    # The fixed RNG order is [0, 2, 1, 3]. Correct all-member partitioning
    # leaves only seed 0 clean, then ranks all three dirty seeds. A
    # representative-only regression instead misclassifies sibling-dirty
    # seed 1 as clean and returns [0, 1, 3].
    assert [candidate.seed_idx for candidate in selected] == [0, 3, 1]
    assert [candidate.seed_percentile for candidate in selected] == [
        0.0,
        12.0,
        50.0,
    ]
    assert [candidate.stored_clean for candidate in selected] == [
        True,
        False,
        False,
    ]
    assert prefix_cache.calls == [
        [clean],
        [
            dirty_representative_high,
            dirty_sibling_mid,
            dirty_representative_low,
        ],
    ]


def test_stage1_cluster_qc_uses_member_own_percentiles_before_amax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real clustered ``compute_fc_percentiles`` implementation."""

    percentile_table = t.zeros((16, 99), dtype=t.float64)
    percentile_grid = t.arange(1, 100, dtype=t.float64) / 100.0
    percentile_table[:] = percentile_grid
    # Sibling 5 has a ten-times-larger activation scale. Its raw activation is
    # larger, but its own-statistic percentile is much smaller.
    percentile_table[5] *= 10.0

    activations = t.zeros((3, 16), dtype=t.float32)
    activations[1, 2] = 0.3
    activations[1, 5] = 0.95
    activations[2, 2] = 0.8
    activations[2, 5] = 0.9
    monkeypatch.setattr(
        qc,
        "_tokenize_prefix_and_full",
        lambda *_args, **_kwargs: ([11], [1, 11, 12]),
    )
    monkeypatch.setattr(
        qc,
        "_run_sae_on_full",
        lambda *_args, **_kwargs: activations.clone(),
    )

    seed_percentile, output_percentile = qc.compute_fc_percentiles(
        "prefix",
        "continuation",
        2,
        SimpleNamespace(cfg=SimpleNamespace(n_ctx=16)),
        object(),
        percentile_table,
        sae_layer=8,
        release_memory=False,
        cluster_target=_target(),
    )

    raw_output_max = activations[2, [2, 5]].max()
    raw_max_then_representative_percentile = float(
        magnitude_to_percentile(
            t.tensor([2], dtype=t.int64),
            raw_output_max.reshape(1),
            percentile_table,
        ).item()
    )
    assert int(activations[2, [2, 5]].argmax().item()) == 1
    assert seed_percentile == pytest.approx(30.0, abs=1e-4)
    assert output_percentile == pytest.approx(80.0, abs=1e-4)
    assert raw_max_then_representative_percentile == pytest.approx(90.0, abs=1e-4)
    assert output_percentile != pytest.approx(raw_max_then_representative_percentile)
