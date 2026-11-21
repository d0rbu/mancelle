"""Sign-critical unit tests for ``feature_diffmean_bag`` (Task 4).

The whole point of this module is to pin EVERY link of the warm-start sign
chain with a mutation trap that goes RED when the link is broken:

    diff[fid] = mean_rich_pct[fid] - mean_avoid_pct[fid]   (PERCENTILE space)

  * POSITIVE mode: top-k_pos by +diff -> (fid, positive_bucket=0);
                   bottom-k_neg by -diff -> (fid, negative_bucket=1).
  * NEGATIVE mode: the MIRROR -- top-k_pos -> negative_bucket;
                   bottom-k_neg -> positive_bucket.

Synthetic activations are injected through the ``_max_pooled_sae_activations``
seam so the answer is known by construction (feature A always higher-percentile
on rich, feature B always higher-percentile on avoid).  The REAL percentile
mapping (``magnitude_to_percentile``) and the REAL selection logic run under
test; only the model forward is stubbed.

Each ``test_sign_trap_*`` asserts the real bag equals the SIGN-CORRECT bag AND
differs from the bag a mutated implementation would produce (raw magnitudes /
flipped diff / dropped bottom-k / un-mirrored negative).  Mutating the
production link makes the real bag collapse onto the mutated bag and the
``== correct`` assertion REDs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

from errors import FeatureClustersError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
)
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv import feature_diffmean
from steering.eval.axbench_bpb_adv.feature_diffmean import (
    PrefixedCompletion,
    feature_diffmean_bag,
)
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContract,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    build_cluster_feature_contract,
)
from tests.axbench_base._loguru_capture import capture_loguru_info

POSITIVE_BUCKET = 0
NEGATIVE_BUCKET = 1


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _linear_percentile_table(n_features: int) -> t.Tensor:
    """``[n_features, 99]`` table whose every row is ``[1, 2, ..., 99]``.

    With this table a magnitude of 0 maps below p1 -> percentile 0.5, and a
    magnitude >= 99 maps above p99 -> percentile 99.5; an interior magnitude m
    maps to percentile ~= m.  Identical across features, so the percentile
    ordering equals the raw ordering -- exactly what the canonical tests want
    (the raw-vs-percentile divergence test uses a *non-uniform* table instead).
    """
    row = t.arange(1, 100, dtype=t.float64)
    return row.unsqueeze(0).repeat(n_features, 1)


def _patch_activations(
    monkeypatch: pytest.MonkeyPatch,
    acts_by_text: Mapping[str, Sequence[float]],
) -> None:
    """Stub ``_max_pooled_sae_activations`` to return injected raw magnitudes.

    Keyed by each item's COMPLETION text so a test can hand distinct rich/avoid
    completions and pin each completion's per-feature magnitude vector. This is
    the seam below the prefix-context + completion-span pooling, so the sign
    chain (diff / selection / mirror) is exercised for real while the model
    forward is bypassed.
    """

    def _stub(
        items: Sequence[PrefixedCompletion],
        *,
        model: object,
        sae: object,
        sae_layer: object,
        device: object,
        extraction_batch_size: object = None,
    ) -> t.Tensor:
        del extraction_batch_size
        return t.tensor(
            [list(acts_by_text[item.completion_text]) for item in items],
            dtype=t.float32,
        )

    monkeypatch.setattr(feature_diffmean, "_max_pooled_sae_activations", _stub)


@dataclass(frozen=True)
class _TestClusterStatistics:
    percentile_table_checksum: str


@dataclass(frozen=True)
class _TestClusterContract:
    consolidation: FeatureClusterConsolidation
    representative_universe: tuple[int, ...]
    statistics: _TestClusterStatistics


def _cluster_contract(
    *,
    percentile_table: t.Tensor,
    mask: FeatureMask | None = None,
) -> _TestClusterContract:
    definition = FeatureClusterDefinition.from_mapping(
        {"schema_version": 1, "clusters": [[0, 1]]},
        n_sae_features=4,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=4,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    representatives = tuple(
        int(value)
        for value in t.nonzero(
            consolidation.cluster_allowed_universe,
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    return _TestClusterContract(
        consolidation=consolidation,
        representative_universe=representatives,
        statistics=_TestClusterStatistics(
            percentile_table_checksum=compute_percentile_checksum(percentile_table)
        ),
    )


def _real_cluster_contract(
    *,
    percentile_table: t.Tensor,
    mask: FeatureMask,
) -> ClusterFeatureContract:
    """Build the public, validated contract without filesystem I/O."""

    n_sae_features = int(percentile_table.shape[0])
    definition = FeatureClusterDefinition.from_mapping(
        {"schema_version": 1, "clusters": [[0, 1]]},
        n_sae_features=n_sae_features,
    )
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=n_sae_features,
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved, mask)
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert feature_vocab is not None
    statistics_domain = tuple(
        int(value)
        for value in t.nonzero(
            ~(percentile_table == 0).all(dim=1),
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    checkpoint_dir = Path("/test/checkpoint")
    checkpoint = CheckpointPin(
        checkpoint_dir=checkpoint_dir,
        resolved_tag="unit",
        step=7,
        model_file=StableFileDigest(
            path=checkpoint_dir / "model_000007.pt",
            sha256="c" * 64,
            byte_length=123,
            stat_signature=(0o100644, 1, 2, 123, 4, 5),
        ),
        metadata_file=StableFileDigest(
            path=checkpoint_dir / "meta_000007.json",
            sha256="d" * 64,
            byte_length=45,
            stat_signature=(0o100644, 1, 3, 45, 4, 5),
        ),
        n_sae_features=n_sae_features,
        feature_vocab=feature_vocab,
        load_bearing_metadata_json="{}",
    )
    selection = CheckpointSelectionContract(
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
        selection_hash="a" * 64,
        selection_manifest_sha256="b" * 64,
        selection_shards_sha256="c" * 64,
        packed_row_fingerprint="d" * 64,
    )
    return build_cluster_feature_contract(
        feature_mask=mask,
        feature_clusters=resolved,
        statistics=FeatureStatisticsIdentity(
            file_sha256="a" * 64,
            percentile_table_checksum=compute_percentile_checksum(percentile_table),
            n_sae_features=n_sae_features,
            feature_id_domain=statistics_domain,
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
        selection=selection,
        concept_source_mode="statistics_domain",
        concept_source_domain=statistics_domain,
    )


def _call(
    *,
    rich_texts: list[str],
    avoid_texts: list[str],
    percentile_table: t.Tensor,
    mode: SteeringMode,
    k_pos: int,
    k_neg: int,
) -> FeatureConditioning:
    # The stubbed pool keys off completion_text only; an empty prefix keeps the
    # sign-trap inputs identical to the pre-#8 whole-completion pool.
    return feature_diffmean_bag(
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in rich_texts],
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in avoid_texts],
        sae=None,
        model=None,
        sae_layer=0,
        percentile_table=percentile_table,
        mode=mode,
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        device="cpu",
    )


def _call_clustered(
    *,
    rich_texts: list[str],
    avoid_texts: list[str],
    percentile_table: t.Tensor,
    contract: ClusterFeatureContract | _TestClusterContract,
    mode: SteeringMode = SteeringMode.POSITIVE,
    k_pos: int = 1,
    k_neg: int = 1,
    allowed_ids: t.Tensor | None = None,
) -> FeatureConditioning:
    if allowed_ids is None:
        allowed_ids = t.tensor(contract.representative_universe, dtype=t.int64)
    return feature_diffmean_bag(
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in rich_texts],
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in avoid_texts],
        sae=None,
        model=None,
        sae_layer=0,
        percentile_table=percentile_table,
        mode=mode,
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        device="cpu",
        allowed_ids=allowed_ids,
        cluster_contract=cast(ClusterFeatureContract, contract),
    )


def _bag_dict(bag: FeatureConditioning) -> dict[int, int]:
    return dict(bag.entries)


# ─────────────────────────────────────────────────────────────────────────────
# Clustered raw-percent-rank projection and fail-loud contract checks
# ─────────────────────────────────────────────────────────────────────────────


def _full_width_cluster_projection_reference(
    acts: t.Tensor,
    percentile_table: t.Tensor,
    *,
    contract: _TestClusterContract,
) -> t.Tensor:
    """Oracle-shaped full-width lookup, valid only when no statistics row is dead."""

    consolidation = contract.consolidation
    all_ids = t.arange(acts.shape[1], dtype=t.int64)
    if consolidation.feature_mask is None:
        allowed_ids = all_ids
    else:
        allowed_ids = all_ids[consolidation.feature_mask.keep_mask(all_ids)]
    total = t.zeros(acts.shape[1], dtype=t.float64)
    for row_acts in acts:
        ranks = magnitude_to_percentile(
            all_ids,
            row_acts.detach().to(dtype=t.float32, device="cpu"),
            percentile_table.to("cpu"),
        )
        representatives, _, projected = consolidation.consolidate_segment(
            allowed_ids,
            ranks[allowed_ids] / 100.0,
            ranks[allowed_ids],
        )
        row = t.zeros(acts.shape[1], dtype=t.float64)
        row[representatives] = projected.to(t.float64)
        total += row
    return total / float(acts.shape[0])


def _mean_before_cluster_projection(
    acts: t.Tensor,
    percentile_table: t.Tensor,
    *,
    contract: _TestClusterContract,
) -> t.Tensor:
    """Mutation reference: mean member percentiles before representative amax."""

    consolidation = contract.consolidation
    all_ids = t.arange(acts.shape[1], dtype=t.int64)
    if consolidation.feature_mask is None:
        allowed_ids = all_ids
    else:
        allowed_ids = all_ids[consolidation.feature_mask.keep_mask(all_ids)]
    rank_rows = [
        magnitude_to_percentile(
            allowed_ids,
            row_acts.detach().to(dtype=t.float32, device="cpu")[allowed_ids],
            percentile_table.to("cpu"),
        )
        for row_acts in acts
    ]
    member_means = t.stack(rank_rows).mean(dim=0)
    representatives, _, projected = consolidation.consolidate_segment(
        allowed_ids,
        member_means / 100.0,
        member_means,
    )
    result = t.zeros(acts.shape[1], dtype=t.float64)
    result[representatives] = projected.to(t.float64)
    return result


def test_clustered_diffmean_projects_each_completion_before_class_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-completion member amax differs from the mean-before-amax mutation."""

    table = _linear_percentile_table(4)
    rich_acts = t.tensor(
        [
            [100.0, 0.0, 0.0, 70.0],
            [0.0, 100.0, 0.0, 70.0],
        ],
        dtype=t.float32,
    )
    avoid_acts = t.tensor(
        [
            [50.0, 50.0, 100.0, 50.0],
            [50.0, 50.0, 100.0, 50.0],
        ],
        dtype=t.float32,
    )
    acts = {
        "rich_a": rich_acts[0].tolist(),
        "rich_b": rich_acts[1].tolist(),
        "avoid_a": avoid_acts[0].tolist(),
        "avoid_b": avoid_acts[1].tolist(),
    }
    _patch_activations(monkeypatch, acts)
    contract = _cluster_contract(percentile_table=table)

    bag = _call_clustered(
        rich_texts=["rich_a", "rich_b"],
        avoid_texts=["avoid_a", "avoid_b"],
        percentile_table=table,
        contract=contract,
    )
    mutated_diff = _mean_before_cluster_projection(
        rich_acts,
        table,
        contract=contract,
    ) - _mean_before_cluster_projection(
        avoid_acts,
        table,
        contract=contract,
    )
    mutated_bag = feature_diffmean._select_bag(
        mutated_diff,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=t.tensor(contract.representative_universe, dtype=t.int64),
    )

    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert _bag_dict(mutated_bag) == {2: NEGATIVE_BUCKET, 3: POSITIVE_BUCKET}
    assert bag != mutated_bag


def test_clustered_diffmean_excludes_masked_sibling_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    acts = {
        "rich": [0.0, 100.0, 0.0, 70.0],
        "avoid": [0.0, 0.0, 100.0, 50.0],
    }
    _patch_activations(monkeypatch, acts)
    contract = _cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (1,)),
    )

    bag = _call_clustered(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        contract=contract,
    )

    assert _bag_dict(bag) == {2: NEGATIVE_BUCKET, 3: POSITIVE_BUCKET}
    assert 1 not in bag.feature_ids()


def test_cluster_projection_ignores_masked_dead_statistics_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-fix deviation: masked dead rows are excluded before percentile lookup."""

    table = _linear_percentile_table(4)
    table[3].zero_()
    acts = {
        "rich": [100.0, 0.0, 0.0, 100.0],
        "avoid": [0.0, 0.0, 100.0, 0.0],
    }
    _patch_activations(monkeypatch, acts)
    contract = _cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (3,)),
    )

    bag = _call_clustered(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        contract=contract,
        allowed_ids=t.tensor([0, 2], dtype=t.int64),
    )

    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert 3 not in bag.feature_ids()


def test_cluster_projection_nonmasked_dead_statistics_row_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    table[3].zero_()
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 0.0]},
    )
    contract = _cluster_contract(percentile_table=table)

    with pytest.raises(
        FeatureGcgError,
        match=(
            r"allowed cluster feature percentile statistics are invalid: "
            r"Percentile statistics are missing for 1 distinct feature ID\(s\) "
            r"\(first 1: \[3\]\)"
        ),
    ) as exc_info:
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=contract,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__).startswith(
        "Percentile statistics are missing for 1 distinct feature ID(s) "
        "(first 1: [3]); all-zero percentile rows cannot be queried."
    )


def test_cluster_projection_allowed_only_matches_full_width_without_dead_rows() -> None:
    table = _linear_percentile_table(4)
    acts = t.tensor(
        [
            [100.0, 25.0, 75.0, 10.0],
            [40.0, 90.0, 15.0, 80.0],
        ],
        dtype=t.float32,
    )
    contract = _cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.blacklist, (3,)),
    )

    allowed_only = feature_diffmean._cluster_projected_mean_percentiles(
        acts,
        table,
        cluster_contract=cast(ClusterFeatureContract, contract),
    )
    full_width = _full_width_cluster_projection_reference(
        acts,
        table,
        contract=contract,
    )

    assert t.equal(allowed_only, full_width)


def _identity_legacy_call(
    *,
    rich_texts: list[str],
    avoid_texts: list[str],
    percentile_table: t.Tensor,
    mode: SteeringMode,
    k_pos: int,
    k_neg: int,
) -> FeatureConditioning:
    return feature_diffmean_bag(
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in rich_texts],
        [PrefixedCompletion(prefix_text="", completion_text=text) for text in avoid_texts],
        sae=None,
        model=None,
        sae_layer=0,
        percentile_table=percentile_table,
        mode=mode,
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        device="cpu",
        allowed_ids=t.tensor([0, 2, 3], dtype=t.int64),
    )


def test_identity_after_mask_cluster_is_byte_exact_legacy_at_former_quantization_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw-rank agreement preserves ordering at a former quantization tie."""

    table = _linear_percentile_table(4)
    acts = {
        "rich_a": [99.1976089, 47.9139214, 79.4518661, 93.2277298],
        "rich_b": [11.4441280, 80.3945770, 6.5061212, 36.4963951],
        "avoid_a": [29.8390217, 3.2362998, 2.9042125, 1.7940879],
        "avoid_b": [11.3233566, 22.0614605, 33.5195618, 77.9739532],
    }
    _patch_activations(monkeypatch, acts)
    contract = _real_cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 2, 3)),
    )
    assert contract.consolidation.is_identity_after_mask

    legacy = _identity_legacy_call(
        rich_texts=["rich_a", "rich_b"],
        avoid_texts=["avoid_a", "avoid_b"],
        percentile_table=table,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    clustered = _call_clustered(
        rich_texts=["rich_a", "rich_b"],
        avoid_texts=["avoid_a", "avoid_b"],
        percentile_table=table,
        contract=contract,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        allowed_ids=t.tensor([0, 2, 3], dtype=t.int64),
    )

    assert clustered == legacy
    assert clustered.digest() == legacy.digest()
    assert clustered.entries == ((0, POSITIVE_BUCKET), (2, NEGATIVE_BUCKET))


def test_identity_after_mask_projection_matches_short_circuit_randomized() -> None:
    """The raw-rank consolidation path is exactly the identity short circuit."""

    table = _linear_percentile_table(4)
    contract = _real_cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 2, 3)),
    )
    assert contract.consolidation.is_identity_after_mask
    allowed_ids = t.tensor(contract.representative_universe, dtype=t.int64)
    generator = t.Generator().manual_seed(906)

    for case_index in range(128):
        acts = (
            t.rand(
                (1 + case_index % 7, 4),
                generator=generator,
                dtype=t.float32,
            )
            * 120.0
        )
        short_circuit = feature_diffmean._mean_percentiles_for_feature_ids(
            acts,
            table,
            feature_ids=allowed_ids,
        )
        consolidated = feature_diffmean._cluster_projected_mean_percentiles(
            acts,
            table,
            cluster_contract=contract,
        )

        assert t.equal(consolidated, short_circuit), (
            f"identity projection differed at randomized case {case_index}"
        )


def test_identity_after_mask_diffmean_differential_corpus_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity-contract and no-cluster bags agree on all 128 corpus cases.

    This retains the former tie-trap corpus, whose sub-bin jitter concentrates
    values inside the old uint8 bins, as a positive agreement test now that both
    branches use raw float64 percent ranks.
    """

    table = _linear_percentile_table(4)
    acts: dict[str, list[float]] = {}
    _patch_activations(monkeypatch, acts)
    contract = _real_cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 2, 3)),
    )
    allowed_ids = t.tensor([0, 2, 3], dtype=t.int64)
    generator = t.Generator().manual_seed(737)
    divergent_cases: list[int] = []

    for case_index in range(128):
        row_codes = t.randint(16, 240, (4, 1), generator=generator)
        row_centers = row_codes.to(t.float32) * (100.0 / 255.0)
        sub_bin_jitter = (
            t.rand((4, 4), generator=generator, dtype=t.float32) - 0.5
        ) * 0.25
        matrix = row_centers.repeat(1, 4) + sub_bin_jitter
        acts.update(
            {
                "rich_a": matrix[0].tolist(),
                "rich_b": matrix[1].tolist(),
                "avoid_a": matrix[2].tolist(),
                "avoid_b": matrix[3].tolist(),
            }
        )
        mode = (
            SteeringMode.POSITIVE
            if case_index % 2 == 0
            else SteeringMode.NEGATIVE
        )
        k_pos = case_index % 5
        k_neg = (case_index * 3) % 5
        call_kwargs = {
            "rich_texts": ["rich_a", "rich_b"],
            "avoid_texts": ["avoid_a", "avoid_b"],
            "percentile_table": table,
            "mode": mode,
            "k_pos": k_pos,
            "k_neg": k_neg,
        }

        legacy = _identity_legacy_call(**call_kwargs)
        clustered = _call_clustered(
            contract=contract,
            allowed_ids=allowed_ids,
            **call_kwargs,
        )

        if clustered != legacy:
            divergent_cases.append(case_index)

    assert not divergent_cases, (
        "identity differential failed in "
        f"{len(divergent_cases)} of 128 cases: {divergent_cases}"
    )


def test_identity_after_mask_ignores_masked_dead_statistics_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    table[1].zero_()
    table[3].zero_()
    _patch_activations(
        monkeypatch,
        {
            "rich": [100.0, 0.0, 0.0, 0.0],
            "avoid": [0.0, 0.0, 100.0, 0.0],
        },
    )
    contract = _real_cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 2)),
    )
    assert contract.consolidation.is_identity_after_mask
    assert contract.representative_universe == (0, 2)
    assert contract.statistics.feature_id_domain == (0, 2)
    assert contract.checkpoint.feature_vocab is not None
    assert contract.checkpoint.feature_vocab.original_ids == (0, 2)

    bag = _call_clustered(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        contract=contract,
        allowed_ids=t.tensor([0, 2], dtype=t.int64),
    )

    assert bag.entries == ((0, POSITIVE_BUCKET), (2, NEGATIVE_BUCKET))


def test_identity_after_mask_allowed_dead_statistics_row_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    table[2].zero_()
    _patch_activations(
        monkeypatch,
        {
            "rich": [100.0, 0.0, 0.0, 0.0],
            "avoid": [0.0, 0.0, 100.0, 0.0],
        },
    )
    contract = _real_cluster_contract(
        percentile_table=table,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 2)),
    )
    assert contract.consolidation.is_identity_after_mask

    with pytest.raises(
        FeatureGcgError,
        match=(
            r"identity-after-mask allowed feature percentile statistics are invalid: "
            r"Percentile statistics are missing for 1 distinct feature ID\(s\) "
            r"\(first 1: \[2\]\)"
        ),
    ) as exc_info:
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=contract,
            allowed_ids=t.tensor([0, 2], dtype=t.int64),
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__).startswith(
        "Percentile statistics are missing for 1 distinct feature ID(s) "
        "(first 1: [2]); all-zero percentile rows cannot be queried."
    )


def test_clustered_diffmean_rejects_original_sae_width_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(5)
    _patch_activations(
        monkeypatch,
        {
            "rich": [100.0, 0.0, 0.0, 0.0, 0.0],
            "avoid": [0.0, 0.0, 100.0, 0.0, 0.0],
        },
    )
    contract = _cluster_contract(percentile_table=table)

    with pytest.raises(
        FeatureGcgError,
        match=r"cluster contract original SAE width 4 disagrees with extracted width 5",
    ):
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=contract,
        )


def test_clustered_diffmean_rejects_same_width_different_percentile_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 0.0]},
    )
    contract = _cluster_contract(percentile_table=table)
    changed = table.clone()
    changed[0, 0] += 0.25

    with pytest.raises(
        FeatureGcgError,
        match=(
            r"percentile table does not match the cluster feature contract "
            r"statistics: expected=[0-9a-f]{64}, actual=[0-9a-f]{64}"
        ),
    ):
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=changed,
            contract=contract,
        )


def test_forged_empty_representative_universe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 0.0]},
    )
    contract = _cluster_contract(percentile_table=table)
    empty_consolidation = replace(
        contract.consolidation,
        cluster_allowed_universe=t.zeros(4, dtype=t.bool),
    )
    forged = replace(
        contract,
        consolidation=empty_consolidation,
        representative_universe=(),
    )

    with pytest.raises(
        FeatureGcgError,
        match="cluster contract has an empty representative universe",
    ):
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=forged,
        )


def test_forged_representative_outside_vocab_names_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 0.0]},
    )
    contract = _cluster_contract(percentile_table=table)
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

    with pytest.raises(
        FeatureGcgError,
        match=r"representative universe contains IDs outside.*FeatureVocab: \[1\]",
    ):
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=forged,
            allowed_ids=t.tensor([0, 2, 3], dtype=t.int64),
        )


def test_forged_wide_vocab_still_ranks_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: a desynced full-width V still cannot admit away ID 1."""

    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 20.0, 10.0], "avoid": [0.0, 0.0, 80.0, 90.0]},
    )
    contract = _cluster_contract(percentile_table=table)

    bag = _call_clustered(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        contract=contract,
        k_pos=3,
        k_neg=0,
        allowed_ids=t.arange(4, dtype=t.int64),
    )

    assert bag.feature_ids() == (0, 2, 3)


def test_wide_allowed_ids_return_direct_representative_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking narrows to representatives before the final safety assertion."""

    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {
            "rich": [0.0, 0.0, 0.0, 0.0],
            "avoid": [100.0, 100.0, 80.0, 60.0],
        },
    )
    contract = _cluster_contract(percentile_table=table)
    wide_allowed_ids = t.arange(4, dtype=t.int64)
    assert set(contract.representative_universe) < set(wide_allowed_ids.tolist())

    bag = _call_clustered(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        contract=contract,
        k_pos=1,
        k_neg=0,
        allowed_ids=wide_allowed_ids,
    )

    assert bag.feature_ids() == (3,)


def test_clustered_final_bag_assert_rejects_away_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping the final artifact assertion makes this forged output pass."""

    table = _linear_percentile_table(4)
    _patch_activations(
        monkeypatch,
        {"rich": [100.0, 0.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 0.0]},
    )
    contract = _cluster_contract(percentile_table=table)
    monkeypatch.setattr(
        feature_diffmean,
        "_select_bag",
        lambda *args, **kwargs: FeatureConditioning(entries=((1, POSITIVE_BUCKET),)),
    )

    with pytest.raises(
        FeatureClustersError,
        match="clustered feature-DiffMean bag contains consolidated-away feature ID 1",
    ):
        _call_clustered(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=table,
            contract=contract,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Canonical known-by-construction scenarios
# ─────────────────────────────────────────────────────────────────────────────
#
# feature 0 fires high on rich (mag 100 -> p99.5) and ~0 on avoid (p0.5)
# feature 1 fires ~0 on rich (p0.5) and high on avoid (mag 100 -> p99.5)
# features 2, 3 fire identically on both sides (mag 50 -> p~50) -> diff 0.
#   => diff = [+99, -99, 0, 0]


_CANON_RICH = {"rich": [100.0, 0.0, 50.0, 50.0]}
_CANON_AVOID = {"avoid": [0.0, 100.0, 50.0, 50.0]}


def test_positive_mode_canonical_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSITIVE: A(rich-firing)->positive bucket 0, B(avoid-firing)->negative 1."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET}


def test_negative_mode_is_the_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGATIVE: the MIRROR -- A->negative bucket 1, B->positive bucket 0."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.NEGATIVE,
        k_pos=1,
        k_neg=1,
    )
    assert _bag_dict(bag) == {0: NEGATIVE_BUCKET, 1: POSITIVE_BUCKET}


def test_positive_and_negative_modes_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive bag and negative bag must NOT be identical (mirror, not copy)."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    table = _linear_percentile_table(4)
    pos = _call(
        rich_texts=["rich"], avoid_texts=["avoid"], percentile_table=table,
        mode=SteeringMode.POSITIVE, k_pos=1, k_neg=1,
    )
    neg = _call(
        rich_texts=["rich"], avoid_texts=["avoid"], percentile_table=table,
        mode=SteeringMode.NEGATIVE, k_pos=1, k_neg=1,
    )
    assert _bag_dict(pos) != _bag_dict(neg)


def test_mean_over_multiple_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diff is over the MEAN percentile across each side's completions."""
    # rich completions: feature 0 fires on average higher than avoid.
    acts = {
        "r0": [100.0, 0.0, 0.0],
        "r1": [100.0, 0.0, 0.0],
        "a0": [0.0, 100.0, 0.0],
        "a1": [0.0, 100.0, 0.0],
        "a2": [0.0, 100.0, 0.0],
    }
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["r0", "r1"],
        avoid_texts=["a0", "a1", "a2"],
        percentile_table=_linear_percentile_table(3),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET}


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-TRAP (i): PERCENTILE space, not RAW magnitudes
# ─────────────────────────────────────────────────────────────────────────────


def test_sign_trap_percentile_not_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-positive is chosen in PERCENTILE space, not raw-magnitude space.

    Construction: feature 0 has tiny magnitudes but a tiny-valued percentile
    table, so its rich/avoid percentile gap is LARGE.  Feature 1 has huge
    magnitudes but a huge-valued percentile table, so its percentile gap is
    SMALLER even though its RAW gap is enormous.  Feature 2 fires on avoid only
    (clear negative both ways).

      percentile diff:  f0 (+~80) > f1 (+~40) > f2 (-~99)   => top-1 = f0
      raw        diff:  f1 (+400) > f0 (+0.8) > f2 (-100)   => top-1 = f1

    Real (percentile) top-1 must be f0, NOT the raw winner f1.
    """
    # Non-uniform percentile table: per-feature monotonic rows on different scales.
    table = t.zeros(3, 99, dtype=t.float64)
    table[0] = t.linspace(0.01, 0.99, 99, dtype=t.float64)     # tiny scale
    table[1] = t.linspace(10.0, 990.0, 99, dtype=t.float64)    # huge scale
    table[2] = t.arange(1, 100, dtype=t.float64)               # mid scale
    acts = {
        # f0 small mags / f1 huge mags / f2 avoid-only
        "rich": [0.9, 500.0, 0.0],
        "avoid": [0.1, 100.0, 100.0],
    }
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=table,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    percentile_correct = {0: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    raw_mutated = {1: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert percentile_correct != raw_mutated  # the input genuinely diverges
    assert _bag_dict(bag) == percentile_correct
    assert _bag_dict(bag) != raw_mutated


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-TRAP (ii): diff is rich - avoid (NOT avoid - rich)
# ─────────────────────────────────────────────────────────────────────────────


def test_sign_trap_diff_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    """diff = rich - avoid.  Flipping to avoid - rich swaps every bucket."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    correct = {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET}
    flipped = {0: NEGATIVE_BUCKET, 1: POSITIVE_BUCKET}  # avoid - rich
    assert _bag_dict(bag) == correct
    assert _bag_dict(bag) != flipped


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-TRAP (iii): bottom-k is kept (NOT dropped)
# ─────────────────────────────────────────────────────────────────────────────


def test_sign_trap_bottom_k_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The negative (bottom-k) features are part of the bag, not dropped."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    # The negative entry (feature 1, avoid-firing) MUST be present in bucket 1.
    assert (1, NEGATIVE_BUCKET) in bag.entries
    assert len(bag) == 2  # top-k (1) + bottom-k (1); dropping bottom-k -> len 1


def test_bottom_k_count_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """k_neg controls how many negative features are taken."""
    acts = {
        "rich": [100.0, 0.0, 0.0, 0.0],
        "avoid": [0.0, 100.0, 90.0, 80.0],  # f1>f2>f3 fire on avoid
    }
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=2,
    )
    negatives = sorted(fid for fid, bucket in bag.entries if bucket == NEGATIVE_BUCKET)
    assert negatives == [1, 2]  # the two most avoid-firing features


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-TRAP (iv): NEGATIVE mode mirrors (is NOT a copy of POSITIVE)
# ─────────────────────────────────────────────────────────────────────────────


def test_sign_trap_negative_is_mirrored_not_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE-mode assignment is the mirror of POSITIVE, not the same bag."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.NEGATIVE,
        k_pos=1,
        k_neg=1,
    )
    mirrored_correct = {0: NEGATIVE_BUCKET, 1: POSITIVE_BUCKET}
    unmirrored_mutated = {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET}  # == positive bag
    assert _bag_dict(bag) == mirrored_correct
    assert _bag_dict(bag) != unmirrored_mutated


# ─────────────────────────────────────────────────────────────────────────────
# Dedup / overlap (no duplicate fids; keep larger |diff| side = sign of diff)
# ─────────────────────────────────────────────────────────────────────────────


def test_overlap_positive_diff_goes_to_top_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fid in BOTH top-k and bottom-k with diff>0 stays on the top side."""
    # diff = [+99 (f0), +20 (f1, the overlap), -99 (f2)]; k_pos=k_neg=2 -> f1 in both.
    acts = {"rich": [100.0, 60.0, 0.0], "avoid": [0.0, 40.0, 100.0]}
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(3),
        mode=SteeringMode.POSITIVE,
        k_pos=2,
        k_neg=2,
    )
    # no duplicate fid (FeatureConditioning would raise) and f1 (diff>0) -> positive.
    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 1: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert len(bag.feature_ids()) == len(set(bag.feature_ids()))


def test_overlap_negative_diff_goes_to_bottom_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fid in BOTH top-k and bottom-k with diff<0 moves to the bottom side."""
    # diff = [+99 (f0), -20 (f1, overlap), -99 (f2)]; k_pos=k_neg=2 -> f1 in both.
    acts = {"rich": [100.0, 40.0, 0.0], "avoid": [0.0, 60.0, 100.0]}
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(3),
        mode=SteeringMode.POSITIVE,
        k_pos=2,
        k_neg=2,
    )
    assert _bag_dict(bag) == {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert len(bag.feature_ids()) == len(set(bag.feature_ids()))


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_deterministic_same_inputs_same_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure / deterministic: identical inputs -> byte-identical bag digest."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    table = _linear_percentile_table(4)
    first = _call(
        rich_texts=["rich"], avoid_texts=["avoid"], percentile_table=table,
        mode=SteeringMode.POSITIVE, k_pos=1, k_neg=1,
    )
    second = _call(
        rich_texts=["rich"], avoid_texts=["avoid"], percentile_table=table,
        mode=SteeringMode.POSITIVE, k_pos=1, k_neg=1,
    )
    assert first.digest() == second.digest()


def test_tie_break_is_deterministic_by_feature_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal diffs break ties by ascending feature id (stable selection)."""
    # f0 and f1 both fire identically high on rich (tie); f2/f3 fire on avoid.
    acts = {"rich": [100.0, 100.0, 0.0, 0.0], "avoid": [0.0, 0.0, 100.0, 100.0]}
    _patch_activations(monkeypatch, acts)
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
    )
    # tie between f0 and f1 for the single positive slot -> smaller fid (0) wins.
    positives = [fid for fid, bucket in bag.entries if bucket == POSITIVE_BUCKET]
    assert positives == [0]


def test_k_larger_than_features_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """k_pos/k_neg larger than the feature count must not error or duplicate."""
    _patch_activations(monkeypatch, {**_CANON_RICH, **_CANON_AVOID})
    bag = _call(
        rich_texts=["rich"],
        avoid_texts=["avoid"],
        percentile_table=_linear_percentile_table(4),
        mode=SteeringMode.POSITIVE,
        k_pos=99,
        k_neg=99,
    )
    assert len(bag.feature_ids()) == len(set(bag.feature_ids()))
    assert len(bag) <= 4  # at most one entry per feature


def test_empty_rich_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_activations(monkeypatch, {**_CANON_AVOID})
    with pytest.raises(FeatureGcgError, match="rich"):
        _call(
            rich_texts=[],
            avoid_texts=["avoid"],
            percentile_table=_linear_percentile_table(4),
            mode=SteeringMode.POSITIVE,
            k_pos=1,
            k_neg=1,
        )


def test_empty_avoid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_activations(monkeypatch, {**_CANON_RICH})
    with pytest.raises(FeatureGcgError, match="avoid"):
        _call(
            rich_texts=["rich"],
            avoid_texts=[],
            percentile_table=_linear_percentile_table(4),
            mode=SteeringMode.POSITIVE,
            k_pos=1,
            k_neg=1,
        )


def test_percentile_table_width_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activation feature count must match the percentile table row count."""
    _patch_activations(monkeypatch, {"rich": [1.0, 2.0, 3.0], "avoid": [3.0, 2.0, 1.0]})
    with pytest.raises(FeatureGcgError):
        _call(
            rich_texts=["rich"],
            avoid_texts=["avoid"],
            percentile_table=_linear_percentile_table(4),  # 4 rows vs 3 features
            mode=SteeringMode.POSITIVE,
            k_pos=1,
            k_neg=1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Real-extraction-path smoke test (content-dependent stub model + SAE flow
# through the genuine _get_sae_activations_batched / percentile / selection).
# ─────────────────────────────────────────────────────────────────────────────


_FAKE_BOS_ID = 1000


class _FakeExtractionTokenizer:
    """Maps each whitespace token to its integer feature id (content-dependent).

    e.g. ``"0 0 2"`` -> ``[0, 0, 2]`` so feature 0 fires twice, feature 2 once.
    Exposes ``bos_token_id`` so ``resolve_bos_token_id`` finds one --
    ``_max_pooled_sae_activations`` prepends it at position 0 and pools it OUT
    with the prefix.  The BOS id (1000) sits OUTSIDE the content vocabulary,
    mirroring real tokenizers where no ordinary text encodes to the BOS id --
    which is exactly the invariant the special-id pollution guard enforces
    (a content id colliding with the BOS id would trip it spuriously).
    Because concatenation is in CHARACTER space, fixtures put a boundary space
    between prefix and completion so the whitespace split does not merge the
    last prefix token with the first completion token (a genuine boundary merge
    is the splice-mismatch trap, tested separately).
    """

    bos_token_id = _FAKE_BOS_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(tok) for tok in text.split()]


class _OffsetExtractionTokenizer(_FakeExtractionTokenizer):
    """Whitespace tokenizer stub that reports character offsets."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        offsets: list[tuple[int, int]] = []
        pos = 0
        for piece in text.split():
            start = text.index(piece, pos)
            end = start + len(piece)
            offsets.append((start, end))
            pos = end
        if return_offsets_mapping:
            return {"input_ids": input_ids, "offset_mapping": offsets}
        return {"input_ids": input_ids}


class _OneHotModel:
    """Minimal HookedTransformer stand-in: residual encodes the token id.

    ``run_with_cache`` returns ``(None, {hook_name: residual})`` with residual
    ``[B, T, 1]`` carrying the (float) token id, so the SAE downstream can fire
    a content-dependent feature.  Implements only the surface the genuine
    ``_get_sae_activations_batched`` TransformerLens path touches.

    Carries its OWN ``.tokenizer`` (like a real HookedTransformer): the DiffMean
    forward tokenizes with ``model.tokenizer``, so the model and the tokenization
    are the same object — the drift-proof contract the production code enforces.
    """

    def __init__(
        self, hook_name: str, n_ctx: int = 64, tokenizer: Any = None
    ) -> None:
        self._hook_name = hook_name
        self.cfg = type("Cfg", (), {"n_ctx": n_ctx})()
        self.tokenizer = (
            tokenizer if tokenizer is not None else _FakeExtractionTokenizer()
        )

    def parameters(self):  # noqa: ANN201 - test stub
        yield t.zeros(1)

    def run_with_cache(
        self,
        tokens: t.Tensor,
        names_filter: list[str] | None = None,
        attention_mask: t.Tensor | None = None,
        stop_at_layer: int | None = None,
    ) -> tuple[None, dict[str, t.Tensor]]:
        residual = tokens.unsqueeze(-1).to(t.float32)  # [B, T, 1]
        return None, {self._hook_name: residual}


class _OneHotSAE:
    """Content-dependent SAE: token id -> that feature fires with magnitude 100.

    ``encode`` receives flattened ``[N, 1]`` residuals (token ids) and returns
    ``[N, n_features]`` with a single hot feature per row, magnitude 100 (which
    the linear percentile table maps to p99.5).  Mirrors the surface the genuine
    extractor reads (``encode``, ``parameters``, ``cfg.d_sae``/``hook_name``).
    """

    def __init__(self, n_features: int) -> None:
        self.feature_dim = n_features
        self._cfg = type(
            "Cfg",
            (),
            {"d_sae": n_features, "hook_name": "blocks.0.hook_resid_post"},
        )()

    @property
    def cfg(self):  # noqa: ANN201 - test stub
        return self._cfg

    @property
    def d_sae(self) -> int:
        return self.feature_dim

    def parameters(self):  # noqa: ANN201 - test stub
        yield t.zeros(1)

    def encode(self, residual: t.Tensor) -> t.Tensor:
        flat = residual.reshape(-1, residual.shape[-1])
        token_ids = flat[:, 0].round().long()
        out = t.zeros(flat.shape[0], self.feature_dim, dtype=t.float32)
        out[t.arange(flat.shape[0]), token_ids % self.feature_dim] = 100.0
        return out


def test_real_extraction_path_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end through the genuine extractor: rich-firing feature -> positive.

    Drives the REAL ``_max_pooled_sae_activations`` (real tokenization seam,
    real ``_run_batched_extractor`` -> ``_get_sae_activations_batched`` ->
    ``sae.encode``, real max-pool, real ``magnitude_to_percentile``, real
    selection).  Content-dependent stubs (no multi-GB checkpoint) make
    "fires more on rich" true by construction: feature 1 appears only in rich
    completions, feature 2 only in avoid completions.
    """
    n_features = 8
    sae = _OneHotSAE(n_features)
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    table = _linear_percentile_table(n_features)
    # rich completions repeatedly contain feature-1 tokens; avoid contain feature-2.
    # Empty prefix => the pool covers the whole completion (span == completion).
    rich_items = [
        PrefixedCompletion(prefix_text="", completion_text="1 1 1"),
        PrefixedCompletion(prefix_text="", completion_text="1 1"),
    ]
    avoid_items = [
        PrefixedCompletion(prefix_text="", completion_text="2 2 2"),
        PrefixedCompletion(prefix_text="", completion_text="2 2"),
    ]
    bag = feature_diffmean_bag(
        rich_items,
        avoid_items,
        sae=sae,
        model=model,
        sae_layer=0,
        percentile_table=table,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        device="cpu",
    )
    assert len(bag) >= 1  # non-empty
    # Feature 1 genuinely fires more on rich -> positive bucket.
    assert (1, POSITIVE_BUCKET) in bag.entries
    # Feature 2 genuinely fires more on avoid -> negative bucket.
    assert (2, NEGATIVE_BUCKET) in bag.entries


# ─────────────────────────────────────────────────────────────────────────────
# #8 — prefix context conditions the forward; the pool covers ONLY the completion
# ─────────────────────────────────────────────────────────────────────────────


def test_completion_span_pooling_excludes_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool covers ONLY the completion tokens; prefix-context tokens are excluded.

    The prefix token fires feature 7 and the completion tokens fire feature 1
    (content-dependent one-hot SAE, run through the REAL extractor + max-pool).
    Pooling over the completion span must leave feature 7 COLD -- if the pool
    covered the whole prefix+completion sequence (the pre-#8 whole-sequence pool)
    feature 7 would be hot at magnitude 100. This is the completion-span teeth.
    """
    n_features = 8
    sae = _OneHotSAE(n_features)
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    acts = feature_diffmean._max_pooled_sae_activations(
        [PrefixedCompletion(prefix_text="7 7", completion_text=" 1 1 1")],
        model=model,
        sae=sae,
        sae_layer=0,
        device="cpu",
    )
    assert acts.shape == (1, n_features)
    # Completion feature fires; prefix feature AND the prepended BOS are excluded
    # from the pooled span.  The one-hot SAE fires feature (token_id % n_features)
    # so the BOS row (1000 -> feature 0) going hot would mean the pool leaked
    # position 0.
    assert acts[0, 1].item() == pytest.approx(100.0)
    assert acts[0, 7].item() == pytest.approx(0.0)
    assert acts[0, _FAKE_BOS_ID % n_features].item() == pytest.approx(0.0)


def test_prefix_context_present_but_not_pooled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix feature that is ABSENT from the completion stays out of the pool.

    Two items share the same completion ("1 1") but carry different prefixes
    (feature 3 vs feature 5). The pooled completion vector must be identical for
    both -- the prefix conditions the forward but never enters the pooled span.
    """
    sae = _OneHotSAE(8)
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    acts = feature_diffmean._max_pooled_sae_activations(
        [
            PrefixedCompletion(prefix_text="3 3", completion_text=" 1 1"),
            PrefixedCompletion(prefix_text="5 5", completion_text=" 1 1"),
        ],
        model=model,
        sae=sae,
        sae_layer=0,
        device="cpu",
    )
    assert t.equal(acts[0], acts[1])  # identical completion -> identical pool
    assert acts[0, 1].item() == pytest.approx(100.0)
    assert acts[0, 3].item() == pytest.approx(0.0)  # prefix feature 3 excluded
    assert acts[1, 5].item() == pytest.approx(0.0)  # prefix feature 5 excluded


def test_prefix_filling_context_window_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix that (with the BOS) fills n_ctx leaves no completion span -> RAISE.

    n_ctx=2 and a 3-token prefix: after the leading BOS the front-clamp keeps only
    ``[BOS, prefix0]``, so no completion position survives.  The boundary space in
    the completion keeps the splice check happy so the n_ctx guard is what fires.
    """
    sae = _OneHotSAE(8)
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post", n_ctx=2)
    with pytest.raises(FeatureGcgError, match="no completion tokens remain"):
        feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 1")],
            model=model,
            sae=sae,
            sae_layer=0,
                device="cpu",
        )


def test_empty_completion_after_tokenization_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion contributing zero tokens beyond the prefix raises.

    With the combined tokenization this is ``len(full_ids) <= len(prefix_ids)``
    (the empty completion adds no tokens), the splice-world spelling of the
    original zero-token-completion raise.
    """
    sae = _OneHotSAE(8)
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    with pytest.raises(FeatureGcgError, match="zero completion tokens"):
        feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="7", completion_text="")],
            model=model,
            sae=sae,
            sae_layer=0,
                device="cpu",
        )


# ─────────────────────────────────────────────────────────────────────────────
# #1/#2 — the leading BOS + the completion-span OFF-BY-ONE (teeth)
# ─────────────────────────────────────────────────────────────────────────────


def _capturing_onehot_extractor(
    captured: dict[str, list[list[int]]], n_features: int
):
    """A ``_run_batched_extractor`` stub: record the exact token id lists it is
    handed and return per-token one-hot acts (token ``tid`` -> feature
    ``tid % n_features`` at magnitude 100), 1:1 with the input, so the pooled span
    is a pure function of the ``model_input`` the SUT built."""

    def _extractor(
        *,
        model: object,
        sae: object,
        token_id_lists: list[list[int]],
        layer: object,
        hook_name: object,
        activation_type: object = None,
        batch_size: object = None,
        per_item_reduce: Callable[[t.Tensor, int], t.Tensor] | None = None,
    ) -> list[t.Tensor]:
        del batch_size
        captured["inputs"] = [list(ids) for ids in token_id_lists]
        out: list[t.Tensor] = []
        for ids in token_id_lists:
            rows = t.zeros(len(ids), n_features, dtype=t.float32)
            for pos, tid in enumerate(ids):
                rows[pos, tid % n_features] = 100.0
            out.append(rows)
        if per_item_reduce is None:
            return out
        # Mirror the real extractor's #1240 contract: collapse each item as it is
        # produced, indexed GLOBALLY. A stub that ignored this would silently
        # hand the SUT raw activations where it expects pooled rows.
        return [per_item_reduce(acts, index) for index, acts in enumerate(out)]

    return _extractor


def test_bos_prepended_and_pool_covers_only_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_input[0] is the BOS and the pool covers ONLY the completion span.

    Sequence the SUT builds for prefix ``"7 7"`` + completion ``" 4 1 1"``::

        model_input = [BOS(1000), 7, 7, 4, 1, 1]
                          pos 0    1  2  3  4  5
        BOS      = position 0     (feature 1000%8=0)  -> context, pooled OUT
        prefix   = positions 1..2 (feature 7)         -> context, pooled OUT
        completion = positions 3..5 (features 4,1,1)  -> the pooled span

    This is the BOS off-by-one teeth; it REDs under every span mutation:
      * pooling the WHOLE sequence (incl. BOS+prefix): feature 7 and feature 0
        become hot -> the ``== 0`` asserts fail.
      * off-by-one that INCLUDES the BOS row / last prefix row (completion_start
        too small): feature 7 (or 0) leaks in -> ``== 0`` fails.
      * off-by-one that DROPS the first completion row (completion_start too
        large): feature 4 (fires ONLY at the first completion position) goes
        cold -> ``== 100`` fails.

    Tokenization uses the model's OWN tokenizer (``model.tokenizer``): the model
    carries a ``_FakeExtractionTokenizer`` and the forward tokenizes with it, so
    the model and its tokenizer are the same object (the drift-proof contract).
    """
    n_features = 8
    captured: dict[str, list[list[int]]] = {}
    monkeypatch.setattr(
        feature_diffmean,
        "_run_batched_extractor",
        _capturing_onehot_extractor(captured, n_features),
    )
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    acts = feature_diffmean._max_pooled_sae_activations(
        [PrefixedCompletion(prefix_text="7 7", completion_text=" 4 1 1")],
        model=model,
        sae=_OneHotSAE(n_features),
        sae_layer=0,
        device="cpu",
    )
    # EXACTLY ONE BOS at position 0, then the combined prefix+completion tokens.
    assert captured["inputs"] == [[_FAKE_BOS_ID, 7, 7, 4, 1, 1]]
    assert captured["inputs"][0][0] == _FAKE_BOS_ID
    # The pooled span is ONLY the completion [4, 1, 1].
    assert acts[0, 4].item() == pytest.approx(100.0)  # first completion token
    assert acts[0, 1].item() == pytest.approx(100.0)  # rest of completion
    assert acts[0, 7].item() == pytest.approx(0.0)  # prefix excluded
    # BOS row (1000 -> feature 1000 % 8 == 0) excluded from the pool.
    assert acts[0, _FAKE_BOS_ID % n_features].item() == pytest.approx(0.0)


def test_splice_mismatch_without_offsets_raises() -> None:
    """An offsetless tokenizer that merges across the boundary raises loudly.

    With the whitespace tokenizer, ``"3 4 5" + "1"`` (no boundary space) tokenizes
    to ``[3, 4, 51]`` whose head ``[3, 4, 51]`` != the prefix's own ``[3, 4, 5]``;
    if the tokenizer cannot report character offsets, the completion span is
    unrecoverable and the SUT must raise rather than pool a guessed span.
    """
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    with pytest.raises(FeatureGcgError, match="does not expose character offsets"):
        feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="3 4 5", completion_text="1")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )


def test_boundary_merge_with_offsets_pools_boundary_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offset-capable tokenizers include a boundary-crossing completion token.

    ``"7 8 9" + "9"`` tokenizes jointly as ``[7, 8, 99]``.  Token ``99`` spans
    the final prefix character and the first completion character, so it is the
    first completion-bearing token and must be pooled.  The pure-prefix tokens
    ``7`` and ``8`` must remain excluded.
    """
    n_features = 20
    captured: dict[str, list[list[int]]] = {}
    monkeypatch.setattr(
        feature_diffmean,
        "_run_batched_extractor",
        _capturing_onehot_extractor(captured, n_features),
    )
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post",
        tokenizer=_OffsetExtractionTokenizer(),
    )
    acts = feature_diffmean._max_pooled_sae_activations(
        [PrefixedCompletion(prefix_text="7 8 9", completion_text="9")],
        model=model,
        sae=_OneHotSAE(n_features),
        sae_layer=0,
        device="cpu",
    )
    assert captured["inputs"] == [[_FAKE_BOS_ID, 7, 8, 99]]
    assert acts[0, 99 % n_features].item() == pytest.approx(100.0)
    assert acts[0, 7].item() == pytest.approx(0.0)
    assert acts[0, 8].item() == pytest.approx(0.0)
    assert acts[0, _FAKE_BOS_ID % n_features].item() == pytest.approx(0.0)


class _NoBosTokenizer:
    """Whitespace tokenizer with NO discoverable BOS id (``bos_token_id`` None)."""

    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(tok) for tok in text.split()]


def test_missing_bos_raises() -> None:
    """No discoverable BOS id -> raise loudly (never silently run without a BOS)."""
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post", tokenizer=_NoBosTokenizer()
    )
    with pytest.raises(FeatureGcgError, match="no BOS id"):
        feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="7 7", completion_text=" 1 1")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )


def test_missing_model_tokenizer_raises() -> None:
    """A base model with no ``.tokenizer`` raises loudly (no silent fallback).

    The DiffMean forward MUST tokenize with the base model's own tokenizer (the
    one it runs the forward under); a model without one is a real loader bug.
    """
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    model.tokenizer = None
    with pytest.raises(FeatureGcgError, match="no .tokenizer"):
        feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="7", completion_text=" 1")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )


def test_uses_the_models_own_tokenizer() -> None:
    """The pool tokenizes with ``model.tokenizer`` — NOT a config-id registry load.

    Two models carrying DIFFERENT tokenizers (different content→id maps) produce
    DIFFERENT pooled features for the SAME text, proving the tokenization comes
    from the model object and can never drift from the model doing the forward
    (max-review finding #3 — the config-id seam is gone).
    """

    class _ShiftedTokenizer:
        """Like _FakeExtractionTokenizer but maps each digit word to id+3."""

        bos_token_id = _FAKE_BOS_ID

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [int(tok) + 3 for tok in text.split()]

    default_model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    shifted_model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post", tokenizer=_ShiftedTokenizer()
    )
    item = [PrefixedCompletion(prefix_text="0", completion_text=" 1")]
    acts_default = feature_diffmean._max_pooled_sae_activations(
        item, model=default_model, sae=_OneHotSAE(8), sae_layer=0, device="cpu"
    )
    acts_shifted = feature_diffmean._max_pooled_sae_activations(
        item, model=shifted_model, sae=_OneHotSAE(8), sae_layer=0, device="cpu"
    )
    # default: completion "1" -> feature 1 hot; shifted: "1"->id 4 -> feature 4 hot.
    assert acts_default[0, 1].item() == pytest.approx(100.0)
    assert acts_shifted[0, 4].item() == pytest.approx(100.0)
    assert not t.equal(acts_default, acts_shifted)


# ──────────────────────────────────────────────────────────────────
# In-data special-id warn+preserve (the double-BOS) + the special-id UNION
# ──────────────────────────────────────────────────────────────────


class _BosLiteralTokenizer:
    """Mirrors real Gemma behavior: the literal ``'<bos>'`` substring re-encodes
    to the REAL BOS id even with ``add_special_tokens=False`` (verified on
    google/gemma-2-2b: ``encode('<bos>Hello', add_special_tokens=False)`` starts
    with id 2).  Whitespace words that are digits map to their int id."""

    bos_token_id = _FAKE_BOS_ID
    eos_token_id = _FAKE_BOS_ID + 1
    all_special_ids = [_FAKE_BOS_ID, _FAKE_BOS_ID + 1]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word == "<bos>":
                ids.append(self.bos_token_id)
            elif word == "<eos>":
                ids.append(self.eos_token_id)
            else:
                ids.append(int(word))
        return ids


@pytest.mark.parametrize(
    ("prefix_text", "completion_text", "hot_features", "cold_features", "polluting_id"),
    [
        # in-data BOS in the PREFIX -> pooled OUT (feature 0 stays cold); only the
        # completion ('4') is pooled. Offset teeth: feature 3 (prefix content)
        # must not leak into the completion span.
        pytest.param("<bos> 3", " 4", (4,), (0, 3), 1000, id="bos-literal-in-prefix"),
        # in-data BOS in the COMPLETION -> PRESERVED and pooled IN (1000 % 8 == 0,
        # so feature 0 goes hot). Preservation teeth: feature 0 hot proves the
        # in-data special survives in the pooled completion.
        pytest.param("3", " 4 <bos>", (4, 0), (3,), 1000, id="bos-literal-in-completion"),
        # in-data EOS in the COMPLETION -> PRESERVED (1001 % 8 == 1, feature 1 hot).
        pytest.param("3", " 4 <eos>", (4, 1), (3,), 1001, id="eos-literal-in-completion"),
    ],
)
def test_special_token_literal_in_text_warns_and_preserves(
    prefix_text: str,
    completion_text: str,
    hot_features: tuple[int, ...],
    cold_features: tuple[int, ...],
    polluting_id: int,
) -> None:
    """TEXT that encodes an in-data special id WARNS (not raises) and preserves.

    This is the class of input where an upstream caller builds ``prefix_text``
    from a specials-preserving decode of BOS-carrying tokens: the literal
    ``'<bos>'`` re-encodes to a REAL BOS id.  That in-data special is genuine
    document DATA, not corruption: only the COMPLETION span is pooled, so a
    special in the PREFIX is pooled OUT (with our injected BOS at position 0) and
    a special in the COMPLETION is PRESERVED and pooled IN.  So instead of
    refusing, the code logs a warning and runs the DiffMean forward as-is.

    Teeth: the per-case ``hot_features``/``cold_features`` pin (a) the in-data
    special being PRESERVED in the completion (bos/eos-in-completion cases) and
    (b) the prefix -- including any in-data special -- being pooled OUT at the
    correct completion offset (bos-in-prefix case). The warning must fire naming
    the polluting id.
    """
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post", tokenizer=_BosLiteralTokenizer()
    )
    with capture_loguru_info() as captured:
        acts = feature_diffmean._max_pooled_sae_activations(
            [
                PrefixedCompletion(
                    prefix_text=prefix_text, completion_text=completion_text
                )
            ],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )
    assert acts.shape == (1, 8)
    for f in hot_features:
        assert acts[0, f].item() == pytest.approx(100.0), (
            f"feature {f} must be hot (pooled from the completion span)"
        )
    for f in cold_features:
        assert acts[0, f].item() == pytest.approx(0.0), (
            f"feature {f} must be cold (pooled OUT with the prefix/BOS)"
        )
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected exactly one pollution warning: {captured!r}"
    assert f"[{polluting_id}]" in warnings[0], (
        f"warning must name the polluting id {polluting_id}: {warnings[0]!r}"
    )


def test_clean_text_fires_no_pollution_warning() -> None:
    """Clean text pools normally AND emits NO in-data-special warning (no false
    positives): the detection set must not spuriously match ordinary content."""
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post", tokenizer=_BosLiteralTokenizer()
    )
    with capture_loguru_info() as captured:
        acts = feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="7 7", completion_text=" 4 1")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )
    assert acts.shape == (1, 8)
    assert acts[0, 4].item() == pytest.approx(100.0)
    warnings = [line for line in captured if "in-data special token id" in line]
    assert warnings == [], f"clean text must not warn: {warnings!r}"


def test_pollution_detection_without_all_special_ids_still_catches_bos() -> None:
    """Without ``all_special_ids`` the detection SET still includes a BOS id.

    ``_FakeExtractionTokenizer`` exposes no ``all_special_ids``; the union helper
    still includes the resolved BOS id (and the named bos attr), so a text
    encoding to the BOS id fires the in-data-special WARNING — the double-BOS
    case is still DETECTED on minimal tokenizer surfaces (max-review finding #7),
    now warn+preserve instead of raise. Teeth: if the union narrowed to exclude
    the BOS on a no-``all_special_ids`` tokenizer, no warning would fire.
    """
    model = _OneHotModel(hook_name="blocks.0.hook_resid_post")
    with capture_loguru_info() as captured:
        acts = feature_diffmean._max_pooled_sae_activations(
            [
                PrefixedCompletion(
                    prefix_text=f"{_FAKE_BOS_ID} 3", completion_text=" 4"
                )
            ],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )
    # The in-data BOS sits in the PREFIX, so it is pooled OUT (feature 0 cold);
    # only the completion ('4') is pooled (feature 4 hot). Detection still fires:
    assert acts[0, 4].item() == pytest.approx(100.0)
    assert acts[0, 0].item() == pytest.approx(0.0)  # in-data BOS pooled out
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected one detection warning: {captured!r}"
    assert f"[{_FAKE_BOS_ID}]" in warnings[0], (
        f"warning must name the detected BOS id {_FAKE_BOS_ID}: {warnings[0]!r}"
    )


class _EosNoAllSpecialsTokenizer:
    """A minimal tokenizer WITHOUT ``all_special_ids`` but WITH a named
    ``eos_token_id`` — the exact surface the bare ``all_special_ids or (bos,)``
    fallback narrowed to BOS-only, letting a literal '<eos>' slip through."""

    bos_token_id = _FAKE_BOS_ID
    eos_token_id = _FAKE_BOS_ID + 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [
            self.eos_token_id if word == "<eos>" else int(word)
            for word in text.split()
        ]


def test_pollution_detection_unions_named_eos_without_all_special_ids() -> None:
    """A named ``eos_token_id`` is in the detection set even without ``all_special_ids``.

    The union includes the named bos/eos/pad attrs, so a literal '<eos>' that
    re-encodes to the EOS id fires the warning — closing the gap the BOS-only
    fallback left (max-review finding #7). Here the '<eos>' sits in the
    COMPLETION, so it is PRESERVED and pooled IN (1001 % 8 == 1 -> feature 1 hot):
    preservation + detection teeth in one.
    """
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post",
        tokenizer=_EosNoAllSpecialsTokenizer(),
    )
    with capture_loguru_info() as captured:
        acts = feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text="3", completion_text=" 4 <eos>")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )
    assert acts[0, 4].item() == pytest.approx(100.0)  # completion '4'
    assert acts[0, 1].item() == pytest.approx(100.0)  # in-data <eos> preserved
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected one detection warning: {captured!r}"
    assert f"[{_FAKE_BOS_ID + 1}]" in warnings[0], (
        f"warning must name the detected EOS id {_FAKE_BOS_ID + 1}: {warnings[0]!r}"
    )


class _BosOmittedFromAllSpecialsTokenizer:
    """``all_special_ids`` is PRESENT but OMITS the BOS (BOS discovered via a
    method probe / named attr, not registered as an HF special) — the exact
    surface where a bare ``all_special_ids`` set (no bos union) would miss the
    double-BOS this guard exists to catch."""

    bos_token_id = _FAKE_BOS_ID
    all_special_ids = [_FAKE_BOS_ID + 5]  # some OTHER special, NOT the BOS

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(tok) for tok in text.split()]


def test_pollution_detection_unions_bos_even_when_all_special_ids_omits_it() -> None:
    """The BOS id is always in the detection set, even if all_special_ids omits it.

    A literal BOS-id token in the text still fires the warning — the resolved
    bos_id is unioned unconditionally (the refuted-but-valid twin of finding #7).
    The in-data BOS is in the PREFIX here, so it is pooled OUT (feature 0 cold);
    only the completion ('4') is pooled. Teeth: no warning would fire if the
    unconditional bos union regressed."""
    model = _OneHotModel(
        hook_name="blocks.0.hook_resid_post",
        tokenizer=_BosOmittedFromAllSpecialsTokenizer(),
    )
    with capture_loguru_info() as captured:
        acts = feature_diffmean._max_pooled_sae_activations(
            [PrefixedCompletion(prefix_text=f"{_FAKE_BOS_ID} 3", completion_text=" 4")],
            model=model,
            sae=_OneHotSAE(8),
            sae_layer=0,
            device="cpu",
        )
    assert acts[0, 4].item() == pytest.approx(100.0)  # completion '4'
    assert acts[0, 0].item() == pytest.approx(0.0)  # in-data BOS pooled out
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected one detection warning: {captured!r}"
    assert f"[{_FAKE_BOS_ID}]" in warnings[0], (
        f"warning must name the detected BOS id {_FAKE_BOS_ID}: {warnings[0]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vocab restriction (issue #695 PR-1 task 2): `_select_bag` / `feature_diffmean_bag`
# restricted to a compacted checkpoint's allowed feature vocabulary `V`.
# ─────────────────────────────────────────────────────────────────────────────


def test_select_bag_restricts_ranking_to_allowed_vocabulary() -> None:
    """The non-vacuous core: a global-top feature OUTSIDE `V` is EXCLUDED and a
    lower-ranked in-`V` feature is chosen instead -- not merely "every returned
    id happens to be in V".

    Construction: feature 3 has the LARGEST +diff (90) and would win the
    unrestricted top-1 slot, but sits OUTSIDE `V`. Feature 1 has the largest
    +diff WITHIN `V` (40). Feature 2 has the most negative diff (-50) and is
    inside `V`, so it wins the bottom-1 slot either way.
    """
    diff = t.tensor([10.0, 40.0, -50.0, 90.0], dtype=t.float64)
    allowed = t.tensor([0, 1, 2], dtype=t.int64)  # V excludes feature 3

    # Sanity: prove the construction genuinely diverges -- an UNRESTRICTED call
    # picks feature 3 (global top, diff=90), not feature 1.
    unrestricted = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=None,
    )
    assert _bag_dict(unrestricted) == {3: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}

    restricted = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=allowed,
    )
    restricted_dict = _bag_dict(restricted)

    # (a) every returned id is in V.
    assert set(restricted_dict) <= {0, 1, 2}
    # (b) the V-internal top-1 (feature 1, diff=40) wins the positive slot --
    # NOT the global top-1 (feature 3, diff=90, excluded because it is outside V).
    assert 3 not in restricted_dict
    assert restricted_dict == {1: POSITIVE_BUCKET, 2: NEGATIVE_BUCKET}
    assert restricted_dict != _bag_dict(unrestricted)


def test_select_bag_negative_mode_also_restricts_to_vocabulary() -> None:
    """The V-restriction applies under NEGATIVE (mirrored-bucket) mode too."""
    diff = t.tensor([10.0, 40.0, -50.0, 90.0], dtype=t.float64)
    allowed = t.tensor([0, 1, 2], dtype=t.int64)  # V excludes feature 3

    bag = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.NEGATIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=allowed,
    )
    bag_dict = _bag_dict(bag)
    assert set(bag_dict) <= {0, 1, 2}
    assert 3 not in bag_dict
    # NEGATIVE mirrors: the V-internal top-diff feature (1) -> negative_bucket;
    # the V-internal bottom-diff feature (2) -> positive_bucket.
    assert bag_dict == {1: NEGATIVE_BUCKET, 2: POSITIVE_BUCKET}


def test_select_bag_v_smaller_than_k_returns_all_of_v_no_crash() -> None:
    """|V| < k_pos/k_neg on a side -> return all of V, no pad/crash."""
    diff = t.tensor([5.0, -5.0, 100.0, -100.0, 3.0], dtype=t.float64)
    allowed = t.tensor([0, 1], dtype=t.int64)  # only 2 features in V, k asks for 10

    bag = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=10,
        k_neg=10,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=allowed,
    )
    bag_dict = _bag_dict(bag)
    assert set(bag_dict) <= {0, 1}  # never selects outside V
    # feature 0 (diff=+5, positive) and feature 1 (diff=-5, negative) are ALL of
    # V, and both fit comfortably under k_pos=k_neg=10 -- no crash, no padding
    # with an out-of-V id to fill the unused budget.
    assert bag_dict == {0: POSITIVE_BUCKET, 1: NEGATIVE_BUCKET}


def test_select_bag_empty_vocabulary_returns_empty_bag_no_crash() -> None:
    """|V| == 0 -> an empty bag, not a crash."""
    diff = t.tensor([5.0, -5.0, 100.0], dtype=t.float64)
    allowed = t.tensor([], dtype=t.int64)

    bag = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=2,
        k_neg=2,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=allowed,
    )
    assert _bag_dict(bag) == {}


def test_select_bag_allowed_ids_none_matches_pre_change_bag_snapshot() -> None:
    """``allowed_ids=None`` reproduces the pre-change (unrestricted) bag exactly.

    Fixed ``diff`` with a hand-verifiable, tie-free ranking: top-2 by +diff are
    indices 2 (3.0) and 1 (2.0); bottom-2 by -diff are indices 3 (-3.0) and
    4 (-2.0). This IS the selection ``_select_bag`` has always produced --
    unchanged by this PR's `V`-restriction -- so the snapshot pins the legacy
    contract for the (default) unrestricted call.
    """
    diff = t.tensor([1.0, 2.0, 3.0, -3.0, -2.0, -1.0], dtype=t.float64)
    bag = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=2,
        k_neg=2,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        allowed_ids=None,
    )
    assert _bag_dict(bag) == {
        2: POSITIVE_BUCKET,
        1: POSITIVE_BUCKET,
        3: NEGATIVE_BUCKET,
        4: NEGATIVE_BUCKET,
    }
    # Omitting allowed_ids entirely must default to the same None behavior.
    again = feature_diffmean._select_bag(
        diff,
        mode=SteeringMode.POSITIVE,
        k_pos=2,
        k_neg=2,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
    )
    assert bag.digest() == again.digest()


def test_feature_diffmean_bag_threads_allowed_ids_into_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``feature_diffmean_bag``'s ``allowed_ids`` reaches ``_select_bag`` for real
    (not just accepted and dropped) -- an out-of-V feature that would otherwise
    win the top slot is excluded end-to-end through the public entry point."""
    acts = {
        "rich": [10.0, 40.0, 0.0, 90.0],
        "avoid": [0.0, 0.0, 50.0, 0.0],
    }
    _patch_activations(monkeypatch, acts)
    table = _linear_percentile_table(4)
    allowed = t.tensor([0, 1, 2], dtype=t.int64)  # excludes feature 3

    unrestricted = _call(
        rich_texts=["rich"], avoid_texts=["avoid"], percentile_table=table,
        mode=SteeringMode.POSITIVE, k_pos=1, k_neg=1,
    )
    assert 3 in _bag_dict(unrestricted)  # feature 3 wins unrestricted (sanity)

    restricted = feature_diffmean_bag(
        [PrefixedCompletion(prefix_text="", completion_text="rich")],
        [PrefixedCompletion(prefix_text="", completion_text="avoid")],
        sae=None,
        model=None,
        sae_layer=0,
        percentile_table=table,
        mode=SteeringMode.POSITIVE,
        k_pos=1,
        k_neg=1,
        positive_bucket=POSITIVE_BUCKET,
        negative_bucket=NEGATIVE_BUCKET,
        device="cpu",
        allowed_ids=allowed,
    )
    restricted_dict = _bag_dict(restricted)
    assert set(restricted_dict) <= {0, 1, 2}
    assert 3 not in restricted_dict
    assert restricted_dict[1] == POSITIVE_BUCKET  # V-internal top-diff feature
