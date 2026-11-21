"""Pure retrieval targets for natural-language AxBench steering."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch as t

from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.eval.axbench_upstream.methods import MethodSetup
from steering.immutable_io import is_sha256_hex

NL_RETRIEVAL_ALGORITHM = "unit_description_cosine_lower_median_maxabs_v1"
NL_STEERING_POLICY = "prompt_last_segment_max_pool_logit_rank_nl_retrieval_lerp_v1"
_EXPECTED_HASH_KEYS = frozenset({"description_embeddings", "concept_embeddings"})


def _canonical_original_ids(setup: MethodSetup) -> tuple[int, ...]:
    """Validate the checkpoint feature contract and return head-row IDs."""
    n_sae_features = setup.n_sae_features
    if type(n_sae_features) is not int or n_sae_features <= 0:
        raise ValueError(
            "NL retrieval n_sae_features must be a positive strict integer, "
            f"got {n_sae_features!r}"
        )

    authoritative = setup.feature_contract.consolidation
    supplied = setup.consolidation
    if (authoritative is None) != (supplied is None):
        raise ValueError(
            "NL retrieval setup consolidation disagrees with the checkpoint "
            "feature contract"
        )
    if authoritative is not None and supplied is not None:
        tensor_fields = (
            "rep_of",
            "consolidated_away",
            "allowed_universe",
            "cluster_allowed_universe",
        )
        if (
            supplied.resolved.canonical_dict()
            != authoritative.resolved.canonical_dict()
            or supplied.feature_mask != authoritative.feature_mask
            or supplied.n_sae_features != authoritative.n_sae_features
            or any(
                not t.equal(getattr(supplied, field), getattr(authoritative, field))
                for field in tensor_fields
            )
        ):
            raise ValueError(
                "NL retrieval setup consolidation disagrees with the checkpoint "
                "feature contract"
            )

    feature_universe = (
        authoritative if authoritative is not None else setup.feature_contract.feature_mask
    )
    expected_vocab = effective_feature_vocab(feature_universe, n_sae_features)
    if setup.feature_vocab != expected_vocab:
        raise ValueError(
            "NL retrieval setup feature vocabulary disagrees with the checkpoint "
            "feature contract"
        )

    if setup.feature_vocab is None:
        return tuple(range(n_sae_features))
    return tuple(int(feature_id) for feature_id in setup.feature_vocab.original_ids)


def retrieval_feature_identity(setup: MethodSetup) -> dict[str, Any]:
    """Return the canonical checkpoint feature identity used by retrieval."""
    original_ids = _canonical_original_ids(setup)
    feature_mask = setup.feature_contract.feature_mask
    resolved_clusters = setup.feature_contract.resolved_clusters
    return {
        "n_sae_features": setup.n_sae_features,
        "original_ids": list(original_ids),
        "feature_mask": None if feature_mask is None else feature_mask.canonical_dict(),
        "feature_clusters": (
            None if resolved_clusters is None else resolved_clusters.canonical_dict()
        ),
    }


def _snapshot_npz(path: Path, *, expected_sha256: str | None, label: str) -> BytesIO:
    snapshot = path.read_bytes()
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(snapshot).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, "
                f"read {actual_sha256}"
            )
    return BytesIO(snapshot)


def _load_description_vectors(
    path: Path,
    *,
    n_sae_features: int,
    expected_sha256: str | None,
) -> tuple[dict[int, np.ndarray], int]:
    snapshot = _snapshot_npz(
        path,
        expected_sha256=expected_sha256,
        label="description embeddings",
    )
    with np.load(snapshot, allow_pickle=False) as payload:
        if len(payload.files) != 2 or set(payload.files) != {"fids", "emb"}:
            raise ValueError(
                "description embedding NPZ must contain exactly ['emb', 'fids']; "
                f"got {sorted(payload.files)}"
            )
        fids = payload["fids"]
        embeddings = payload["emb"]

    if fids.ndim != 1:
        raise ValueError(
            f"description fids must be rank 1, got shape {tuple(fids.shape)}"
        )
    if fids.dtype.kind not in {"i", "u"} or fids.dtype == np.dtype(np.bool_):
        raise ValueError(
            f"description fids must use an integer dtype excluding bool, got {fids.dtype}"
        )
    if embeddings.ndim != 2:
        raise ValueError(
            "description embeddings must be rank 2, got shape "
            f"{tuple(embeddings.shape)}"
        )
    if embeddings.dtype.kind != "f":
        raise ValueError(
            f"description embeddings must use a floating dtype, got {embeddings.dtype}"
        )
    if embeddings.shape[0] != fids.shape[0]:
        raise ValueError(
            "description embedding row count disagrees with fids: "
            f"{embeddings.shape[0]} != {fids.shape[0]}"
        )
    if embeddings.shape[1] == 0:
        raise ValueError("description embeddings must have a positive dimension")

    ids = [int(feature_id) for feature_id in fids.tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError("description fids must be globally unique")
    invalid = next(
        (feature_id for feature_id in ids if feature_id < 0 or feature_id >= n_sae_features),
        None,
    )
    if invalid is not None:
        raise ValueError(
            f"description feature ID {invalid} is out of range [0, {n_sae_features})"
        )

    vectors = embeddings.astype(np.float32, copy=True)
    if not bool(np.isfinite(vectors).all()):
        raise ValueError("description embeddings must contain only finite values")
    if bool((~np.any(vectors != 0, axis=1)).any()):
        raise ValueError("description embeddings must not contain a zero vector")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not bool(np.isfinite(norms).all()):
        raise ValueError("description embedding norms must be finite")
    vectors /= np.maximum(norms, np.float32(1e-8))
    return ({feature_id: vectors[row] for row, feature_id in enumerate(ids)}, vectors.shape[1])


def _load_concept_vectors(
    path: Path,
    *,
    expected_dim: int,
    expected_sha256: str | None,
) -> dict[str, np.ndarray]:
    snapshot = _snapshot_npz(
        path,
        expected_sha256=expected_sha256,
        label="concept embeddings",
    )
    with np.load(snapshot, allow_pickle=False) as payload:
        if len(payload.files) != 2 or set(payload.files) != {"texts", "emb"}:
            raise ValueError(
                "concept embedding NPZ must contain exactly ['emb', 'texts']; "
                f"got {sorted(payload.files)}"
            )
        texts = payload["texts"]
        embeddings = payload["emb"]

    if texts.ndim != 1:
        raise ValueError(f"concept texts must be rank 1, got shape {tuple(texts.shape)}")
    if texts.dtype.kind != "U":
        raise ValueError(f"concept texts must use a Unicode string dtype, got {texts.dtype}")
    if embeddings.ndim != 2:
        raise ValueError(
            f"concept embeddings must be rank 2, got shape {tuple(embeddings.shape)}"
        )
    if embeddings.dtype.kind != "f":
        raise ValueError(
            f"concept embeddings must use a floating dtype, got {embeddings.dtype}"
        )
    if embeddings.shape[0] != texts.shape[0]:
        raise ValueError(
            "concept embedding row count disagrees with texts: "
            f"{embeddings.shape[0]} != {texts.shape[0]}"
        )
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"concept embedding dimension {embeddings.shape[1]} disagrees with "
            f"description embedding dimension {expected_dim}"
        )

    names = [str(text) for text in texts.tolist()]
    if any(not text.strip() for text in names):
        raise ValueError("concept embedding texts must be nonblank")
    if len(names) != len(set(names)):
        raise ValueError("concept embedding texts must be globally unique")

    vectors = embeddings.astype(np.float32, copy=True)
    if not bool(np.isfinite(vectors).all()):
        raise ValueError("concept embeddings must contain only finite values")
    if bool((~np.any(vectors != 0, axis=1)).any()):
        raise ValueError("concept embeddings must not contain a zero query")
    return {text: vectors[row] for row, text in enumerate(names)}


def _unit_description_table(
    setup: MethodSetup,
    original_ids: tuple[int, ...],
    vector_of: Mapping[int, np.ndarray],
    *,
    dimension: int,
) -> tuple[t.Tensor, t.Tensor]:
    row_of = {feature_id: row for row, feature_id in enumerate(original_ids)}
    consolidation = setup.consolidation
    feature_ids = (
        original_ids if consolidation is None
        else t.nonzero(consolidation.allowed_universe, as_tuple=False).flatten().tolist()
    )
    sums = np.zeros((len(original_ids), dimension), dtype=np.float32)
    # Ascending feature IDs retain the legacy addition order within each row.
    for feature_id in feature_ids:
        representative = (
            feature_id if consolidation is None else int(consolidation.rep_of[feature_id].item())
        )
        row = row_of.get(representative)
        if row is None:
            raise ValueError(
                "NL retrieval consolidation maps allowed feature "
                f"{feature_id} to representative {representative}, which has no "
                "feature-vocabulary row"
            )
        vector = vector_of.get(feature_id)
        if vector is not None:
            sums[row] += vector

    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    if not bool(np.isfinite(sums).all()) or not bool(np.isfinite(norms).all()):
        raise ValueError("unit description aggregation produced non-finite values")
    covered = norms[:, 0] > np.float32(1e-8)
    if not bool(covered.any()):
        raise ValueError("no conditioning unit has a usable description embedding")
    sums[covered] /= norms[covered]
    sums[~covered] = np.float32(0.0)
    return t.from_numpy(sums), t.from_numpy(covered)


def build_retrieval_target_scores(
    *,
    setup: MethodSetup,
    description_embeddings_path: Path,
    concept_embeddings_path: Path,
    concepts: Mapping[int, str],
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[int, t.Tensor]:
    """Build canonical FP-head-row retrieval targets for requested concepts.

    Extra concept embeddings are accepted so one all-concept snapshot can serve
    filtered evaluations. Requested descriptions still match by exact text.
    Rows without a usable unit description remain literal neutral targets; they
    are not removed from downstream selection.
    """
    if not concepts:
        raise ValueError("NL retrieval requires at least one requested concept")
    hashes = {} if expected_hashes is None else dict(expected_hashes)
    if expected_hashes is not None:
        if set(hashes) != _EXPECTED_HASH_KEYS:
            raise ValueError(
                "expected_hashes must contain exactly "
                f"{sorted(_EXPECTED_HASH_KEYS)}; got {sorted(hashes)}"
            )
        for label in sorted(_EXPECTED_HASH_KEYS):
            digest = hashes[label]
            if not is_sha256_hex(digest):
                raise ValueError(
                    f"expected_hashes[{label!r}] must be 64 lowercase hexadecimal "
                    f"characters, got {digest!r}"
                )

    original_ids = _canonical_original_ids(setup)
    vector_of, dimension = _load_description_vectors(
        Path(description_embeddings_path),
        n_sae_features=setup.n_sae_features,
        expected_sha256=hashes.get("description_embeddings"),
    )
    concept_vector_of = _load_concept_vectors(
        Path(concept_embeddings_path),
        expected_dim=dimension,
        expected_sha256=hashes.get("concept_embeddings"),
    )

    requested: dict[int, str] = {}
    for concept_id, text in concepts.items():
        if type(concept_id) is not int or concept_id < 0:
            raise ValueError(
                f"concept IDs must be non-negative strict integers, got {concept_id!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"concept {concept_id} description must be a nonblank string, got {text!r}"
            )
        if text not in concept_vector_of:
            raise ValueError(
                f"concept embedding snapshot is missing exact requested text {text!r}"
            )
        requested[concept_id] = text

    table, covered = _unit_description_table(
        setup,
        original_ids,
        vector_of,
        dimension=dimension,
    )
    scores_by_id: dict[int, t.Tensor] = {}
    for concept_id in sorted(requested):
        query = t.from_numpy(concept_vector_of[requested[concept_id]])
        similarities = table @ query
        if not bool(t.isfinite(similarities).all()):
            raise ValueError("NL retrieval produced non-finite cosine similarities")
        centered = similarities - similarities[covered].median()
        if not bool(t.isfinite(centered).all()):
            raise ValueError("NL retrieval produced non-finite centered similarities")
        scale = centered[covered].abs().max()
        scores = centered / scale if bool(scale > 0) else t.zeros_like(centered)
        scores[~covered] = 0.0
        scores = scores.to(dtype=t.float32).clamp_(-1.0, 1.0).contiguous()
        if scores.device.type != "cpu" or not bool(t.isfinite(scores).all()):
            raise ValueError("NL retrieval produced a non-finite or non-CPU target")
        scores_by_id[concept_id] = scores
    return scores_by_id
