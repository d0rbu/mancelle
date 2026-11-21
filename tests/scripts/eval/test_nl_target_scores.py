"""Focused retrieval-target alignment and validation tests."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from zipfile import ZipFile

import numpy as np
import pytest
import torch as t

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_upstream.methods import MethodSetup
from steering.eval.axbench_upstream.nl_target_scores import (
    NL_RETRIEVAL_ALGORITHM,
    NL_STEERING_POLICY,
    build_retrieval_target_scores,
    retrieval_feature_identity,
)


def _setup(
    tmp_path: Path,
    *,
    n_sae_features: int,
    mask: FeatureMask | None = None,
    clusters: tuple[tuple[int, ...], ...] | None = None,
) -> MethodSetup:
    resolved = None
    if clusters is not None:
        definition = FeatureClusterDefinition(clusters=clusters)
        resolved = ResolvedFeatureClusters(
            source_path=tmp_path / "clusters.yaml",
            definition=definition,
            descriptor=FeatureClusterDescriptor.from_definition(
                definition,
                n_sae_features=n_sae_features,
            ),
        )
    contract = CheckpointFeatureContract(
        feature_mask=mask,
        resolved_clusters=resolved,
    )
    consolidation = contract.consolidation
    vocab = effective_feature_vocab(contract.universe, n_sae_features)
    return cast(
        MethodSetup,
        SimpleNamespace(
            n_sae_features=n_sae_features,
            feature_contract=contract,
            consolidation=consolidation,
            feature_vocab=vocab,
        ),
    )


def _write_descriptions(
    path: Path,
    fids: object,
    emb: object,
    **extra: np.ndarray,
) -> Path:
    np.savez(path, fids=np.asarray(fids), emb=np.asarray(emb), **extra)
    return path


def _write_concepts(
    path: Path,
    texts: object,
    emb: object,
    **extra: np.ndarray,
) -> Path:
    np.savez(path, texts=np.asarray(texts), emb=np.asarray(emb), **extra)
    return path


def _npy_bytes(value: object) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_duplicate_npz(
    path: Path,
    *,
    duplicate_name: str,
    duplicate_values: tuple[object, object],
    other_name: str,
    other_value: object,
) -> Path:
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(path, "w") as archive:
            archive.writestr(f"{duplicate_name}.npy", _npy_bytes(duplicate_values[0]))
            archive.writestr(f"{duplicate_name}.npy", _npy_bytes(duplicate_values[1]))
            archive.writestr(f"{other_name}.npy", _npy_bytes(other_value))
    return path


def _score(
    tmp_path: Path,
    setup: MethodSetup,
    *,
    fids: object,
    description_emb: object,
    texts: object = ("target",),
    concept_emb: object = ((1.0, 0.0),),
    concepts: dict[int, str] | None = None,
) -> dict[int, t.Tensor]:
    return build_retrieval_target_scores(
        setup=setup,
        description_embeddings_path=_write_descriptions(
            tmp_path / "descriptions.npz", fids, description_emb
        ),
        concept_embeddings_path=_write_concepts(
            tmp_path / "concepts.npz", texts, concept_emb
        ),
        concepts={7: "target"} if concepts is None else concepts,
    )


def _frozen_legacy_retrieval(
    setup: MethodSetup,
    fids: np.ndarray,
    description_emb: np.ndarray,
    query: np.ndarray,
) -> t.Tensor:
    """Frozen retrieval-only math from the pre-integration NL branch."""
    assert setup.feature_vocab is not None
    assert setup.consolidation is not None
    vectors = description_emb.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    vector_of = {int(fid): vectors[row] for row, fid in enumerate(fids.tolist())}
    original_ids = t.as_tensor(list(setup.feature_vocab.original_ids), dtype=t.int64)
    rep_of = setup.consolidation.rep_of
    allowed = setup.consolidation.allowed_universe
    row_of_rep = {int(fid): row for row, fid in enumerate(original_ids.tolist())}
    members_by_row: list[list[int]] = [[] for _ in range(len(row_of_rep))]
    for fid in t.nonzero(allowed, as_tuple=False).squeeze(1).tolist():
        row = row_of_rep.get(int(rep_of[fid]))
        if row is not None:
            members_by_row[row].append(int(fid))
    contributions = [
        (row, vector_of[fid])
        for row, members in enumerate(members_by_row)
        for fid in members
        if fid in vector_of
    ]
    sums = np.zeros((len(members_by_row), vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(members_by_row), dtype=np.int64)
    for row, vector in contributions:
        sums[row] += vector
        counts[row] += 1
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    covered_np = (counts > 0) & (norms.squeeze(1) > 1e-8)
    sums[covered_np] /= norms[covered_np]
    sums[~covered_np] = 0.0
    table = t.from_numpy(sums)
    covered = t.from_numpy(covered_np)
    similarities = table @ t.as_tensor(query.astype(np.float32)).reshape(-1)
    centered = similarities - similarities[covered].median()
    scale = centered[covered].abs().max()
    scores = centered / scale if scale > 0 else t.zeros_like(centered)
    scores[~covered] = 0.0
    return scores.to(t.float32).clamp_(-1.0, 1.0)


def test_hand_computed_lower_median_golden_and_extra_concept(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=4)
    scores = _score(
        tmp_path,
        setup,
        fids=[3, 1, 0, 2],
        description_emb=np.eye(4, dtype=np.float32)[[3, 1, 0, 2]],
        texts=["unused extra", "target"],
        concept_emb=[[9.0, 9.0, 9.0, 9.0], [0.8, 0.2, -0.4, -0.9]],
    )[7]

    t.testing.assert_close(
        scores,
        t.tensor([1.0, 0.5, 0.0, -5.0 / 12.0], dtype=t.float32),
        atol=1e-7,
        rtol=0.0,
    )
    assert scores.dtype is t.float32
    assert scores.device.type == "cpu"
    assert scores.is_contiguous()


def test_compacted_mask_preserves_canonical_head_row_order(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 4))
    setup = _setup(tmp_path, n_sae_features=6, mask=mask)

    scores = _score(
        tmp_path,
        setup,
        fids=[4, 1, 3, 0],
        description_emb=[[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )[7]

    assert setup.feature_vocab is not None
    assert tuple(setup.feature_vocab.original_ids) == (1, 3, 4)
    assert t.equal(scores, t.tensor([1.0, 0.0, 0.0]))


def test_clustered_members_sum_in_ascending_original_id_order_exactly(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 3, 4, 5))
    setup = _setup(
        tmp_path,
        n_sae_features=6,
        mask=mask,
        clusters=((1, 3), (2, 4, 5)),
    )
    # File order puts the large canceling pair before the tiny vector. Canonical
    # member order is 2, 4, 5, for which float32 loses the tiny x contribution
    # before adding feature 5. A file-order aggregation does not.
    fids = np.array([2, 5, 4, 0, 3], dtype=np.int64)
    descriptions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 1e-4, 0.0],
            [1e-16, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    query = np.array([0.2, -0.7, 0.9], dtype=np.float64)

    actual = _score(
        tmp_path,
        setup,
        fids=fids,
        description_emb=descriptions,
        concept_emb=[query],
    )[7]
    expected = _frozen_legacy_retrieval(setup, fids, descriptions, query)

    assert setup.feature_vocab is not None
    assert tuple(setup.feature_vocab.original_ids) == (0, 2, 3)
    assert t.equal(actual, expected)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_canceling_cluster_is_an_uncovered_neutral_target(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=3, clusters=((0, 1),))
    scores = _score(
        tmp_path,
        setup,
        fids=[0, 1, 2],
        description_emb=[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        concept_emb=[[0.25, 1.0]],
    )[7]

    assert setup.feature_vocab is not None
    assert tuple(setup.feature_vocab.original_ids) == (0, 2)
    assert t.equal(scores, t.zeros(2, dtype=t.float32))


def test_equal_covered_cosines_are_a_valid_all_zero_target(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    scores = _score(
        tmp_path,
        setup,
        fids=[0, 1],
        description_emb=[[1.0, 0.0], [0.0, 1.0]],
        concept_emb=[[1.0, 1.0]],
    )[7]
    assert t.equal(scores, t.zeros(2, dtype=t.float32))


@pytest.mark.parametrize("seed", range(10))
def test_random_clustered_inputs_are_bit_exact_to_frozen_legacy_retrieval(
    tmp_path: Path,
    seed: int,
) -> None:
    setup = _setup(
        tmp_path,
        n_sae_features=8,
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2, 3, 4, 5, 6)),
        clusters=((0, 3, 5), (1, 4)),
    )
    rng = np.random.default_rng(seed)
    fids = rng.permutation(8).astype(np.int64)
    descriptions = rng.normal(size=(8, 7)).astype(np.float64)
    query = rng.normal(size=7).astype(np.float64)

    actual = _score(
        tmp_path,
        setup,
        fids=fids,
        description_emb=descriptions,
        concept_emb=[query],
    )[7]
    expected = _frozen_legacy_retrieval(setup, fids, descriptions, query)
    assert t.equal(actual, expected)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_feature_identity_binds_full_clusters_mask_and_rows(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 2, 3, 4, 5))
    setup = _setup(
        tmp_path,
        n_sae_features=6,
        mask=mask,
        clusters=((1, 3), (2, 4)),
    )
    assert NL_RETRIEVAL_ALGORITHM == "unit_description_cosine_lower_median_maxabs_v1"
    assert NL_STEERING_POLICY.endswith("nl_retrieval_lerp_v1")
    assert retrieval_feature_identity(setup) == {
        "n_sae_features": 6,
        "original_ids": [0, 2, 3, 5],
        "feature_mask": {"mode": "whitelist", "ids": [0, 2, 3, 4, 5]},
        "feature_clusters": {
            "schema_version": 1,
            "clusters": [[1, 3], [2, 4]],
        },
    }


@pytest.mark.parametrize(
    ("fids", "emb", "match"),
    [
        ([0, 0], [[1.0, 0.0], [0.0, 1.0]], "globally unique"),
        ([0, 3], [[1.0, 0.0], [0.0, 1.0]], "out of range"),
        ([False], [[1.0, 0.0]], "integer dtype excluding bool"),
        ([0], [[0.0, 0.0]], "zero vector"),
        ([0], [[np.nan, 0.0]], "finite values"),
    ],
)
def test_invalid_description_rows_fail_loudly(
    tmp_path: Path,
    fids: object,
    emb: object,
    match: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=3)
    with pytest.raises(ValueError, match=match):
        _score(tmp_path, setup, fids=fids, description_emb=emb)


@pytest.mark.parametrize(
    ("fids", "emb", "match"),
    [
        ([[0]], [[1.0, 0.0]], "fids must be rank 1"),
        ([0.0], [[1.0, 0.0]], "integer dtype excluding bool"),
        ([0], [1.0, 0.0], "embeddings must be rank 2"),
        ([0], [[1, 0]], "embeddings must use a floating dtype"),
        ([0, 1], [[1.0, 0.0]], "row count disagrees"),
        ([0], np.empty((1, 0), dtype=np.float32), "positive dimension"),
        (
            [0],
            [[np.finfo(np.float32).max, np.finfo(np.float32).max]],
            "norms must be finite",
        ),
    ],
)
def test_description_snapshot_shapes_and_dtypes_fail_loudly(
    tmp_path: Path,
    fids: object,
    emb: object,
    match: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    with pytest.raises(ValueError, match=match):
        _score(tmp_path, setup, fids=fids, description_emb=emb)


@pytest.mark.parametrize(
    ("texts", "emb", "concepts", "match"),
    [
        (["target", "target"], [[1.0, 0.0], [0.0, 1.0]], {7: "target"}, "unique"),
        (["target"], [[0.0, 0.0]], {7: "target"}, "zero query"),
        (["target"], [[np.inf, 0.0]], {7: "target"}, "finite values"),
        (["target"], [[1.0, 0.0]], {7: "Target"}, "missing exact"),
    ],
)
def test_invalid_concept_snapshots_or_requests_fail_loudly(
    tmp_path: Path,
    texts: object,
    emb: object,
    concepts: dict[int, str],
    match: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    with pytest.raises(ValueError, match=match):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0, 0.0], [0.0, 1.0]],
            texts=texts,
            concept_emb=emb,
            concepts=concepts,
        )


@pytest.mark.parametrize(
    ("texts", "emb", "match"),
    [
        ([["target"]], [[1.0, 0.0]], "texts must be rank 1"),
        (np.array([b"target"]), [[1.0, 0.0]], "Unicode string dtype"),
        (["target"], [1.0, 0.0], "embeddings must be rank 2"),
        (["target"], [[1, 0]], "embeddings must use a floating dtype"),
        (["target", "other"], [[1.0, 0.0]], "row count disagrees"),
        (["target"], [[1.0, 0.0, 0.0]], "dimension 3 disagrees"),
        ([" "], [[1.0, 0.0]], "texts must be nonblank"),
    ],
)
def test_concept_snapshot_shapes_and_dtypes_fail_loudly(
    tmp_path: Path,
    texts: object,
    emb: object,
    match: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    with pytest.raises(ValueError, match=match):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0, 0.0], [0.0, 1.0]],
            texts=texts,
            concept_emb=emb,
        )


def test_distinct_concept_ids_may_share_text_without_aliasing_targets(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    scores = _score(
        tmp_path,
        setup,
        fids=[0, 1],
        description_emb=[[1.0, 0.0], [0.0, 1.0]],
        concepts={3: "target", 7: "target"},
    )
    assert t.equal(scores[3], scores[7])
    assert scores[3].data_ptr() != scores[7].data_ptr()


@pytest.mark.parametrize(
    ("concepts", "match"),
    [
        ({}, "at least one"),
        ({-1: "target"}, "non-negative"),
        ({False: "target"}, "non-negative"),
        ({7: " "}, "nonblank string"),
    ],
)
def test_empty_negative_or_bool_concept_ids_fail_loudly(
    tmp_path: Path,
    concepts: dict[int, str],
    match: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    with pytest.raises(ValueError, match=match):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0, 0.0], [0.0, 1.0]],
            concepts=concepts,
        )


def test_large_finite_query_overflow_fails_instead_of_becoming_zero(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    maximum = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="non-finite cosine similarities"):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0, 1.0], [-1.0, 0.0]],
            concept_emb=[[maximum, maximum]],
        )


def test_finite_opposite_cosines_that_overflow_when_centered_fail_loudly(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    maximum = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="non-finite centered similarities"):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0], [-1.0]],
            concept_emb=[[maximum]],
        )


def test_wholly_empty_or_canceled_coverage_fails_loudly(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=3, clusters=((0, 1),))
    with pytest.raises(ValueError, match="no conditioning unit"):
        _score(
            tmp_path,
            setup,
            fids=[0, 1],
            description_emb=[[1.0, 0.0], [-1.0, 0.0]],
        )


def test_npz_keys_and_pickle_free_text_contract_are_strict(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz",
        [0],
        [[1.0, 0.0]],
        labels=np.array(["unused"]),
    )
    concepts = _write_concepts(tmp_path / "concepts.npz", ["target"], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="exactly"):
        build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=descriptions,
            concept_embeddings_path=concepts,
            concepts={7: "target"},
        )


def test_object_concept_texts_are_rejected_without_pickle(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz", [0], [[1.0, 0.0]]
    )
    concepts = _write_concepts(
        tmp_path / "concepts-object.npz",
        np.array(["target"], dtype=object),
        [[1.0, 0.0]],
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=descriptions,
            concept_embeddings_path=concepts,
            concepts={7: "target"},
        )


@pytest.mark.parametrize("artifact", ["description", "concept"])
def test_duplicate_npz_archive_members_are_rejected(
    tmp_path: Path,
    artifact: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz", [0, 1], [[1.0, 0.0], [0.0, 1.0]]
    )
    concepts = _write_concepts(tmp_path / "concepts.npz", ["target"], [[1.0, 0.0]])
    if artifact == "description":
        descriptions = _write_duplicate_npz(
            tmp_path / "duplicate-descriptions.npz",
            duplicate_name="fids",
            duplicate_values=([0], [1]),
            other_name="emb",
            other_value=[[1.0, 0.0]],
        )
        match = "description embedding NPZ must contain exactly"
    else:
        concepts = _write_duplicate_npz(
            tmp_path / "duplicate-concepts.npz",
            duplicate_name="texts",
            duplicate_values=(["target"], ["other"]),
            other_name="emb",
            other_value=[[1.0, 0.0]],
        )
        match = "concept embedding NPZ must contain exactly"

    with pytest.raises(ValueError, match=match):
        build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=descriptions,
            concept_embeddings_path=concepts,
            concepts={7: "target"},
        )


@pytest.mark.parametrize("clustered", [False, True])
def test_retrieval_and_identity_preserve_all_global_rng_states(
    tmp_path: Path,
    clustered: bool,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2))
    setup = _setup(
        tmp_path,
        n_sae_features=4,
        mask=mask,
        clusters=((0, 1),) if clustered else None,
    )
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz",
        [0, 1, 2, 3],
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]],
    )
    concepts = _write_concepts(
        tmp_path / "concepts.npz", ["target"], [[0.25, 0.75]]
    )
    random.seed(1729)
    np.random.seed(1729)
    t.manual_seed(1729)
    python_before = random.getstate()
    numpy_before = cast(
        tuple[str, np.ndarray, int, int, float],
        np.random.get_state(),
    )
    torch_before = t.random.get_rng_state().clone()

    retrieval_feature_identity(setup)
    build_retrieval_target_scores(
        setup=setup,
        description_embeddings_path=descriptions,
        concept_embeddings_path=concepts,
        concepts={7: "target"},
    )

    numpy_after = cast(
        tuple[str, np.ndarray, int, int, float],
        np.random.get_state(),
    )
    assert random.getstate() == python_before
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert t.equal(t.random.get_rng_state(), torch_before)


def test_feature_vocab_or_consolidation_drift_fails_before_scoring(tmp_path: Path) -> None:
    setup = _setup(tmp_path, n_sae_features=4, clusters=((0, 1),))
    bad_vocab = cast(
        MethodSetup,
        SimpleNamespace(
            n_sae_features=4,
            feature_contract=setup.feature_contract,
            consolidation=setup.consolidation,
            feature_vocab=FeatureVocab(n_sae_features=4, original_ids=(0, 2)),
        ),
    )
    with pytest.raises(ValueError, match="feature vocabulary disagrees"):
        retrieval_feature_identity(bad_vocab)

    assert setup.consolidation is not None
    bad_consolidation = replace(
        setup.consolidation,
        rep_of=setup.consolidation.rep_of.roll(1),
    )
    drifted = cast(
        MethodSetup,
        SimpleNamespace(
            n_sae_features=4,
            feature_contract=setup.feature_contract,
            consolidation=bad_consolidation,
            feature_vocab=setup.feature_vocab,
        ),
    )
    with pytest.raises(ValueError, match="consolidation disagrees"):
        retrieval_feature_identity(drifted)


def test_invalid_width_and_one_sided_consolidation_fail_loudly(tmp_path: Path) -> None:
    valid = _setup(tmp_path, n_sae_features=3)
    invalid_width = cast(
        MethodSetup,
        SimpleNamespace(
            n_sae_features=0,
            feature_contract=valid.feature_contract,
            consolidation=None,
            feature_vocab=None,
        ),
    )
    with pytest.raises(ValueError, match="positive strict integer"):
        retrieval_feature_identity(invalid_width)

    clustered = _setup(tmp_path, n_sae_features=3, clusters=((0, 1),))
    mismatched = cast(
        MethodSetup,
        SimpleNamespace(
            n_sae_features=3,
            feature_contract=valid.feature_contract,
            consolidation=clustered.consolidation,
            feature_vocab=valid.feature_vocab,
        ),
    )
    with pytest.raises(ValueError, match="consolidation disagrees"):
        retrieval_feature_identity(mismatched)


@pytest.mark.parametrize(
    ("artifact_key", "message"),
    [
        ("description_embeddings", "description embeddings SHA-256 mismatch"),
        ("concept_embeddings", "concept embeddings SHA-256 mismatch"),
    ],
)
def test_expected_hashes_authenticate_the_exact_parsed_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_key: str,
    message: str,
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz", [0, 1], [[1.0, 0.0], [0.0, 1.0]]
    )
    concepts = _write_concepts(tmp_path / "concepts.npz", ["target"], [[1.0, 0.0]])
    expected_hashes = {
        "description_embeddings": hashlib.sha256(descriptions.read_bytes()).hexdigest(),
        "concept_embeddings": hashlib.sha256(concepts.read_bytes()).hexdigest(),
    }
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def counted_read_bytes(path: Path) -> bytes:
        read_counts[path] = read_counts.get(path, 0) + 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    scores = build_retrieval_target_scores(
        setup=setup,
        description_embeddings_path=descriptions,
        concept_embeddings_path=concepts,
        concepts={7: "target"},
        expected_hashes=expected_hashes,
    )
    assert t.equal(scores[7], t.tensor([1.0, 0.0]))
    assert read_counts == {descriptions: 1, concepts: 1}

    with pytest.raises(ValueError, match=message):
        build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=descriptions,
            concept_embeddings_path=concepts,
            concepts={7: "target"},
            expected_hashes={
                **expected_hashes,
                artifact_key: "0" * 64,
            },
        )


@pytest.mark.parametrize(
    "expected_hashes",
    [
        {},
        {"description_embeddings": "0" * 64},
        {
            "description_embeddings": "0" * 64,
            "concept_embeddings": "0" * 64,
            "extra": "0" * 64,
        },
        {
            "description_embeddings": "0" * 63,
            "concept_embeddings": "0" * 64,
        },
        {
            "description_embeddings": "A" * 64,
            "concept_embeddings": "0" * 64,
        },
    ],
)
def test_expected_hash_mapping_shape_and_digest_format_are_strict(
    tmp_path: Path,
    expected_hashes: dict[str, str],
) -> None:
    setup = _setup(tmp_path, n_sae_features=2)
    descriptions = _write_descriptions(
        tmp_path / "descriptions.npz", [0], [[1.0, 0.0]]
    )
    concepts = _write_concepts(tmp_path / "concepts.npz", ["target"], [[1.0, 0.0]])
    match = "contain exactly" if set(expected_hashes) != {
        "description_embeddings",
        "concept_embeddings",
    } else "64 lowercase hexadecimal"
    with pytest.raises(ValueError, match=match):
        build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=descriptions,
            concept_embeddings_path=concepts,
            concepts={7: "target"},
            expected_hashes=expected_hashes,
        )
