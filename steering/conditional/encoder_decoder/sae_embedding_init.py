"""SAE-backed initialization for encoder feature embeddings.

The encoder consumes discrete SAE feature IDs through ``feature_embed``.  This
module optionally replaces the random feature-ID embedding table with a
projection of the SAE decoder directions, while preserving the average row norm
of the random table it overwrites.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch as t
from beartype import beartype
from loguru import logger
from sae_lens import SAE

from errors import FeatureClustersError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)


class EncoderEmbeddingInit(StrEnum):
    """Supported SAE-based encoder feature embedding initialization modes."""

    RANDOM = "random"
    SVD = "svd"


class EncoderEmbeddingClusterVectors(StrEnum):
    """Cluster-vector policies for SAE-backed feature embedding initialization."""

    REPRESENTATIVE = "representative"
    MEAN = "mean"


@dataclass(frozen=True)
class EncoderEmbeddingInitResult:
    """Summary of an applied encoder feature embedding initialization."""

    mode: EncoderEmbeddingInit
    n_sae_features: int  # ORIGINAL dictionary width
    sae_dim: int
    encoder_dim: int
    target_avg_norm: float
    projected_avg_norm: float
    final_avg_norm: float
    vocab_size: int = 0  # V — rows actually written. Equals n_sae_features when uncompacted.
    cluster_vectors: EncoderEmbeddingClusterVectors = EncoderEmbeddingClusterVectors.MEAN
    clusters_averaged: int = 0
    members_averaged: int = 0


def _mapping_get(mapping: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    get = getattr(mapping, "get", None)
    if callable(get):
        return get(key, default)
    return getattr(mapping, key, default)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def parse_encoder_embedding_init(raw: object) -> EncoderEmbeddingInit | None:
    """Parse the user-facing initialization mode.

    ``None``/Hydra ``null`` is the no-op default.  String parsing is
    case-insensitive so CLI values like ``SVD`` behave as expected.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"null", "none"}:
            return None
        try:
            return EncoderEmbeddingInit(normalized)
        except ValueError as exc:
            raise ValueError(
                "initialize_encoder_embeddings_from_sae must be one of "
                "null, random, or svd; got "
                f"{raw!r}."
            ) from exc
    raise ValueError(
        "initialize_encoder_embeddings_from_sae must be one of null, random, "
        f"or svd; got {raw!r} ({type(raw).__name__})."
    )


def parse_encoder_embedding_cluster_vectors(raw: object) -> EncoderEmbeddingClusterVectors:
    """Parse the cluster-vector policy. ``mean`` is the DEFAULT (owner decision).

    An absent key and Hydra ``null`` resolve to ``mean``: when clusters are
    configured, averaging their surviving members is the intended treatment, and
    when they are not, ``mean`` has nothing to average and degenerates to the
    representative path bitwise — so the default is safe for every unclustered
    config. ``representative`` is the explicit opt-out. String parsing is
    case-insensitive, matching :func:`parse_encoder_embedding_init`.
    """
    if raw is None:
        return EncoderEmbeddingClusterVectors.MEAN
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"null", "none"}:
            return EncoderEmbeddingClusterVectors.MEAN
        try:
            return EncoderEmbeddingClusterVectors(normalized)
        except ValueError as exc:
            raise ValueError(
                "initialize_encoder_embeddings_cluster_vectors must be one of "
                f"representative or mean; got {raw!r}."
            ) from exc
    raise ValueError(
        "initialize_encoder_embeddings_cluster_vectors must be one of representative "
        f"or mean; got {raw!r} ({type(raw).__name__})."
    )


def _validate_cluster_vector_configuration(
    cluster_vectors: EncoderEmbeddingClusterVectors,
) -> None:
    """Reject a policy value outside the declared enum, before any SAE is loaded.

    This validates ONLY the policy value itself. Every mode/cluster combination of
    the two declared policies is legal: ``representative`` leaves either projection
    path unchanged, and ``mean`` — the config DEFAULT — degenerates to an inert
    bitwise no-op wherever there is nothing to average (``null`` init, no clusters).
    Where the averaging actually lands per mode is
    ``initialize_encoder_feature_embeddings_from_matrix``'s contract, not this
    function's. The raise guards direct API callers handing the matrix initializer
    a non-enum value; the config path cannot reach it
    (``parse_encoder_embedding_cluster_vectors`` returns a member or raises first).
    """
    if cluster_vectors is EncoderEmbeddingClusterVectors.REPRESENTATIVE:
        return
    if cluster_vectors is EncoderEmbeddingClusterVectors.MEAN:
        return
    raise ValueError(f"Unsupported encoder embedding cluster-vector policy: {cluster_vectors!r}")


def _apply_cluster_means(
    normalized: t.Tensor,
    feature_clusters: FeatureClusterConsolidation,
) -> tuple[t.Tensor, int, int]:
    """Replace each representative row with the mean of its mask-surviving members.

    Operates on a full fp32 ``(n_sae, d)`` matrix at the caller's mode-specific fixed
    stage: ``random`` passes its post-rescale projection before the final vocabulary
    gather, while ``svd`` passes oriented ``W_dec`` before the vocabulary subset and
    covariance. ONLY the representative row (the cluster's smallest surviving member,
    first of the canonically ascending tuple) is written; away rows keep their raw
    values verbatim. That is a contract, not an accident of a later gather dropping
    them: the spec defines ``M[j] = N[j]`` for every non-representative row, and this
    helper exists as a separable unit precisely so a test can pin the full matrix
    directly instead of arguing from unobservability.

    Reads always come from the pristine input (writes land in a clone), so no write
    can feed a later mean even hypothetically — clusters are structurally disjoint,
    but the isolation must not depend on that.

    Returns ``(matrix, clusters_averaged, members_averaged)``; singleton-survivor
    clusters are counted but are bitwise no-ops.
    """
    source = normalized
    out = source.clone()
    clusters_averaged = 0
    members_averaged = 0
    for cluster in feature_clusters.resolved.definition.clusters:
        survivors = [
            feature_id
            for feature_id in cluster
            if bool(feature_clusters.allowed_universe[feature_id])
        ]
        if not survivors:
            continue
        survivor_indices = t.tensor(survivors, dtype=t.int64, device=source.device)
        out[survivors[0]] = source[survivor_indices].mean(dim=0)
        clusters_averaged += 1
        members_averaged += len(survivors)
    return out, clusters_averaged, members_averaged


def coerce_sae_l0(release: SAERelease, raw_l0: object) -> int | L0Tier | None:
    """Coerce a config ``sae.l0`` value to ``resolve_sae``'s contract."""
    if release == SAERelease.gemmascope_2:
        if isinstance(raw_l0, str):
            return L0Tier(raw_l0)
        if raw_l0 is None or isinstance(raw_l0, L0Tier):
            return raw_l0
        raise ValueError(
            f"sae.l0={raw_l0!r} is not valid for release {release.value!r}: "
            "expected null or a gemma-scope-2 tier name "
            "(small/medium/medium_seed_1/big)."
        )
    if raw_l0 is None or (isinstance(raw_l0, int) and not isinstance(raw_l0, bool)):
        return raw_l0
    raise ValueError(
        f"sae.l0={raw_l0!r} is not valid for release {release.value!r}: "
        "expected null for the canonical v1 SAE or an int average-L0."
    )


def load_sae_decoder_matrix_from_cfg(sae_cfg: Mapping[str, Any] | Any) -> t.Tensor:
    """Load the configured SAE and return its decoder matrix on CPU as fp32."""
    release = SAERelease(_enum_value(_mapping_get(sae_cfg, "release")))
    resolved = resolve_sae(
        model=str(_mapping_get(sae_cfg, "model")),
        release=release,
        activation_type=ActivationType(_enum_value(_mapping_get(sae_cfg, "activation_type"))),
        layer=int(_mapping_get(sae_cfg, "layer")),
        width=SAEWidth(_enum_value(_mapping_get(sae_cfg, "width"))),
        l0=coerce_sae_l0(release, _mapping_get(sae_cfg, "l0", None)),
    )
    logger.info(
        "Loading SAE for encoder embedding initialization: release={} sae_id={}",
        resolved.release,
        resolved.sae_id,
    )
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cpu",
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    matrix = getattr(sae, "W_dec", None)
    if matrix is None:
        raise ValueError(
            f"SAE {resolved.release}/{resolved.sae_id} does not expose W_dec; "
            "cannot initialize encoder feature embeddings from SAE decoder directions."
        )
    return matrix.detach().to(device="cpu", dtype=t.float32)


def orient_sae_decoder_matrix(matrix: t.Tensor, n_sae_features: int) -> t.Tensor:
    """Return matrix as ``(n_sae_features, sae_dim)``.

    SAELens stores decoder directions as ``W_dec`` with shape
    ``(n_sae_features, sae_dim)`` because decode is ``feature_acts @ W_dec``.
    This helper also accepts the transpose for callers/tests that provide raw
    arrays in ``(sae_dim, n_sae_features)`` form.
    """
    if matrix.ndim != 2:
        raise ValueError(f"SAE decoder matrix must be rank-2, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] == n_sae_features:
        return matrix
    if matrix.shape[1] == n_sae_features:
        return matrix.T.contiguous()
    raise ValueError(
        "SAE decoder matrix shape must include n_sae_features="
        f"{n_sae_features}; got shape {tuple(matrix.shape)}."
    )


@beartype
def project_sae_decoder_matrix(
    matrix: t.Tensor,
    encoder_dim: PositiveInt,
    mode: EncoderEmbeddingInit,
    *,
    seed: int,
    rows_are_feature_vocab: bool = False,
) -> t.Tensor:
    """Project SAE decoder directions to the encoder residual dimension.

    ``@beartype`` owns the unary positive ``encoder_dim`` contract; the body
    retains the matrix-rank relationship required by SVD projection.

    ``rows_are_feature_vocab`` tells the svd rank guard WHAT it is looking at, which it
    cannot otherwise know: a matrix is a matrix. ``True`` ⇒ the rows are a compacted
    model's feature vocabulary (the mask's allowed universe, subset in *before* the
    projection — requirement 6, and the only caller that does so). The default ``False``
    ⇒ the rows are the whole SAE dictionary, which is what every other caller hands over,
    including the ``random`` path on a compacted model (it subsets LAST, to stay bitwise).

    It exists only to make the guard's *diagnosis* right. It changes no math, and it is
    read on no path but the raise.
    """
    matrix = matrix.to(device="cpu", dtype=t.float32)
    sae_dim = matrix.shape[1]
    if mode == EncoderEmbeddingInit.RANDOM:
        generator = t.Generator(device="cpu").manual_seed(int(seed))
        projection = t.randn(
            sae_dim,
            encoder_dim,
            generator=generator,
            device="cpu",
            dtype=t.float32,
        )
        projection = projection / (sae_dim**0.5)
        return matrix @ projection

    if mode == EncoderEmbeddingInit.SVD:
        n_rows = matrix.shape[0]
        # The eigenbasis is drawn from `matrix.T @ matrix`, whose rank is min(rows, sae_dim)
        # — so BOTH bounds matter. Only `sae_dim` was ever checked, because before #695 the
        # row count was always n_sae (16,384 / 65,536) and the rank was therefore always
        # sae_dim. Since #695 the svd path is fed the V-ROW SUBSET (requirement 6: the
        # principal directions must be the sub-dictionary's), so a whitelist with
        # V < encoder_dim makes `encoder_dim - V` of the selected eigenvectors NULL-SPACE
        # directions with ~zero eigenvalues, and `matrix @ top_vectors` emits whole columns
        # that are identically ~0.
        #
        # Nothing downstream catches that: average_row_norm is finite, the
        # `current_avg_norm <= 0` check does not trip, and the degenerate table is rescaled
        # to the full table's norm and reported as a normal init. MEASURED at sae_dim=48,
        # encoder_dim=32: V=16 -> 16 of 32 columns dead; V=4 -> 28 of 32. Silently.
        rank_bound = min(n_rows, sae_dim)
        if encoder_dim > rank_bound:
            # The rows and the sae_dim are two DIFFERENT failures with two different
            # remedies, and `rows` is only a VOCABULARY size on the compacted svd path.
            # Advising "widen the mask" unconditionally sent an operator who has no mask
            # off to widen one, and — where sae_dim is what actually binds — sent even a
            # masked operator off to do a no-op.
            use_random = (
                "or use initialize_encoder_embeddings_from_sae: random, whose projection "
                "is seeded and rank-independent."
            )
            if n_rows >= sae_dim:
                diagnosis = (
                    f"The SAE decoder dimension is the binding constraint: sae_dim="
                    f"{sae_dim} < encoder_dim={encoder_dim}."
                )
                if rows_are_feature_vocab:
                    diagnosis += (
                        f" Widening the effective vocabulary CANNOT help — its {n_rows} "
                        "features already exceed sae_dim."
                    )
                remedy = (
                    f"Lower the encoder dim (encoder.n_embd) to at most sae_dim={sae_dim}, "
                    f"{use_random}"
                )
            elif rows_are_feature_vocab:
                # The rows here are the EFFECTIVE vocabulary: mask-allowed ids minus
                # cluster-consolidated members. The guard cannot know WHICH of the two
                # narrowed it — a run may have a mask, clusters, or both — so the message
                # names both levers rather than sending an operator to widen a
                # feature_mask they may never have configured (the exact trap
                # TestTheRankGuardDiagnosesTheRIGHTModel exists to prevent for the
                # mask/no-mask split).
                diagnosis = (
                    "The model's effective feature vocabulary is the binding constraint: "
                    f"it admits only {n_rows} features, fewer than encoder_dim="
                    f"{encoder_dim}. The svd basis is computed over the effective "
                    "vocabulary's rows only — mask-allowed ids minus cluster-consolidated "
                    "members (#695/#714) — so a narrow feature_mask, aggressive "
                    "feature_clusters, or their combination triggers this where the full "
                    "dictionary would not."
                )
                remedy = (
                    "Widen whichever input narrowed the vocabulary — the feature_mask's "
                    "allowed universe and/or the feature_clusters definition "
                    "(consolidation removes one row per extra surviving member) — until "
                    f"at least encoder_dim={encoder_dim} effective features remain, "
                    f"{use_random}"
                )
            else:
                diagnosis = (
                    f"The SAE dictionary itself is the binding constraint: it has only "
                    f"{n_rows} features, fewer than encoder_dim={encoder_dim}. This model "
                    "is NOT compacted — these rows are the WHOLE dictionary, not a mask's "
                    "allowed universe — so no feature_mask is involved here and none can "
                    "be adjusted to fix it."
                )
                remedy = (
                    f"Lower the encoder dim (encoder.n_embd) to at most {rank_bound}, "
                    f"point sae: at a wider dictionary, {use_random}"
                )
            raise ValueError(
                "SVD encoder embedding initialization needs at least encoder_dim "
                f"independent directions, but the SAE decoder matrix it was given has rank "
                f"at most {rank_bound} (rows={n_rows}, sae_dim={sae_dim}) while "
                f"encoder_dim={encoder_dim}.\n"
                f"{diagnosis}\n"
                f"{encoder_dim - rank_bound} of the {encoder_dim} selected eigenvectors "
                "would be null-space directions, so the initialized table would have that "
                "many identically-zero columns — a structurally degenerate init that no "
                "later check would catch.\n"
                f"{remedy}"
            )
        covariance = matrix.T @ matrix
        eigvals, eigvecs = t.linalg.eigh(covariance)
        del eigvals
        top_vectors = eigvecs[:, -encoder_dim:].flip(dims=(1,)).contiguous()
        return matrix @ top_vectors

    raise ValueError(f"Unsupported encoder embedding initialization mode: {mode!r}")


def average_row_norm(matrix: t.Tensor) -> t.Tensor:
    """Return the mean L2 norm across rows, computed in fp32."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a rank-2 embedding matrix, got shape {tuple(matrix.shape)}")
    avg = matrix.detach().float().norm(dim=1).mean()
    if not bool(t.isfinite(avg)):
        raise ValueError("Average row norm is not finite.")
    return avg


def normalize_to_average_row_norm(matrix: t.Tensor, target_avg_norm: t.Tensor) -> t.Tensor:
    """Scale ``matrix`` so its average row norm matches ``target_avg_norm``."""
    current_avg_norm = average_row_norm(matrix)
    target_avg_norm = target_avg_norm.detach().to(device=matrix.device, dtype=t.float32)
    if current_avg_norm <= 0:
        raise ValueError("Projected SAE embedding matrix has zero average row norm.")
    if not bool(t.isfinite(target_avg_norm)):
        raise ValueError("Target encoder feature embedding average row norm is not finite.")
    if target_avg_norm <= 0:
        raise ValueError("Target encoder feature embedding average row norm must be positive.")
    return matrix * (target_avg_norm.float() / current_avg_norm)


@t.no_grad()
def initialize_encoder_feature_embeddings_from_matrix(
    model: Any,
    sae_decoder_matrix: t.Tensor,
    mode: EncoderEmbeddingInit,
    *,
    seed: int,
    cluster_vectors: EncoderEmbeddingClusterVectors = EncoderEmbeddingClusterVectors.MEAN,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> EncoderEmbeddingInitResult:
    """Overwrite ``model.encoder.feature_embed.weight`` from SAE directions.

    Compaction-aware (issue #695, requirement 6). ``n_sae_features`` MUST come from the
    encoder config, not from the table's row count: once compacted that row count is
    ``V``, and orienting ``W_dec`` (which has ``n_sae`` rows) against it would fail.

    Compact row ``j`` receives the SAE direction of ORIGINAL id
    ``vocab.original_ids[j]``. Getting this wrong would silently give every feature
    another feature's direction.

    The rescale TARGET is always the FULL ``(n_sae, d)`` table's average row norm — the
    number the uncompacted reference would have used. A V-row subset's mean row-norm is
    only a *sample estimate* of it (2.1e-04 relative at V=35,275 of 65,536), so using it
    would let compaction silently perturb the init. Compaction is purely representational.

    The two modes then diverge, deliberately:

    * ``random`` projects through a seeded, row-count-independent matrix. We project the
      **full** ``(n_sae, sae_dim)`` matrix and subset **LAST**, so the compacted and the
      reference runs execute the *identical* GEMM and the identical rescale. The result is
      **bitwise** the reference's rows — not merely close. (Subsetting first would be
      mathematically equal but would run a different-shaped GEMM, leaving byte-exactness
      hostage to how BLAS reduces at M=V vs M=n_sae.)
    * ``svd`` computes its covariance over the SELECTED rows, so the principal directions
      are those of the *sub-dictionary*. That is the literal reading of requirement 6 and
      a deliberate owner decision, and it means a compacted ``svd`` init CANNOT equal the
      row-subset of an uncompacted one. Pinned by test, so the choice cannot drift.

    With ``cluster_vectors=mean`` — the DEFAULT — averaging is a third, deliberately
    mode-specific stage. ``random`` averages the fp32 normalized full-dictionary
    projection after the global rescale and before its final vocabulary gather. ``svd``
    averages the fp32 oriented full dictionary before its initial vocabulary subset, so
    the covariance and basis are computed over the averaged effective dictionary. This
    asymmetry mirrors where each mode already places its row subset; moving either mean
    across that mode's subset/rescale boundary is a different algorithm. Each non-empty
    source cluster writes its representative from the ascending mask-surviving members.
    Singleton means are still counted but are bitwise no-ops. Without clusters, ``mean``
    has nothing to average and the pipeline is bitwise the pre-option path.
    """
    feature_weight = model.encoder.feature_embed.weight  # (V, encoder_dim)
    vocab = model.encoder.feature_vocab
    n_sae_features = model.encoder.config.n_sae_features  # ORIGINAL width
    vocab_size, encoder_dim = feature_weight.shape

    _validate_cluster_vector_configuration(cluster_vectors)
    averaging = (
        cluster_vectors is EncoderEmbeddingClusterVectors.MEAN and feature_clusters is not None
    )
    if averaging:
        assert feature_clusters is not None  # narrowed by `averaging`
        if feature_clusters.n_sae_features != n_sae_features:
            raise FeatureClustersError(
                "cluster-vector mean SAE-width mismatch: "
                f"feature_clusters.n_sae_features={feature_clusters.n_sae_features}, "
                f"model n_sae_features={n_sae_features}"
            )
        expected_vocab = effective_feature_vocab(feature_clusters, n_sae_features)
        if vocab != expected_vocab:
            expected_size = n_sae_features if expected_vocab is None else expected_vocab.size
            actual_size = n_sae_features if vocab is None else vocab.size
            raise FeatureClustersError(
                "cluster-vector mean requires the model feature vocabulary to equal "
                "effective_feature_vocab(feature_clusters, n_sae_features); "
                f"expected V={expected_size}, "
                f"got V={actual_size}. Build the model with the resolved effective "
                "feature vocabulary."
            )

    # The FULL table's norm. When uncompacted the table IS the full table; when compacted
    # the full table only ever existed as a transient inside init_weights(), which
    # captured this for us (and raises if it never ran, rather than guessing).
    target_avg_norm = (
        average_row_norm(feature_weight)
        if vocab is None
        else model.encoder.feature_embed_full_avg_row_norm
    )

    clusters_averaged = 0
    members_averaged = 0
    oriented = orient_sae_decoder_matrix(sae_decoder_matrix, n_sae_features)
    sae_dim = oriented.shape[1]

    # svd + mean: the model's dictionary IS the averaged one, so its selected-row
    # covariance and eigenbasis must see the means. Apply on the full oriented matrix
    # before svd's existing subset-first step. `random` deliberately averages later.
    if averaging and mode == EncoderEmbeddingInit.SVD:
        assert feature_clusters is not None  # narrowed by `averaging`
        oriented, clusters_averaged, members_averaged = _apply_cluster_means(
            oriented,
            feature_clusters,
        )

    # svd ONLY: the basis must come from the sub-dictionary, so subset before projecting.
    subset_before_projection = vocab is not None and mode == EncoderEmbeddingInit.SVD
    if subset_before_projection:
        oriented = vocab.select_rows(oriented)  # (V, sae_dim)

    # `subset_before_projection` is exactly "the rows below are a vocabulary, not the
    # dictionary" — the one fact the rank guard needs to diagnose itself correctly.
    projected = project_sae_decoder_matrix(
        oriented,
        PositiveInt.parse(encoder_dim),
        mode,
        seed=seed,
        rows_are_feature_vocab=subset_before_projection,
    )

    # Validate BEFORE the rescale, so a bad projection can never partially overwrite the
    # table (and so it reports "shape mismatch" rather than tripping the rescale's
    # zero-norm guard first). The expected row count is the PROJECTION's, which is V only
    # when we subset ahead of it — otherwise the full n_sae.
    expected_rows = vocab_size if subset_before_projection else n_sae_features
    if projected.shape != (expected_rows, encoder_dim):
        raise ValueError(
            "Projected SAE embedding matrix shape mismatch: expected "
            f"{(expected_rows, encoder_dim)}, got {tuple(projected.shape)}."
        )

    projected_avg_norm = average_row_norm(projected)
    normalized = normalize_to_average_row_norm(projected, target_avg_norm)

    if averaging and mode == EncoderEmbeddingInit.RANDOM:
        assert feature_clusters is not None  # narrowed by `averaging`
        normalized, clusters_averaged, members_averaged = _apply_cluster_means(
            normalized,
            feature_clusters,
        )

    # random: subset LAST, off the fully-normalized (n_sae, d) matrix ⇒ bitwise-exact.
    if vocab is not None and not subset_before_projection:
        normalized = vocab.select_rows(normalized)

    if normalized.shape != feature_weight.shape:  # unreachable; guards a select_rows bug
        raise ValueError(
            "Normalized SAE embedding matrix shape mismatch: expected "
            f"{tuple(feature_weight.shape)}, got {tuple(normalized.shape)}."
        )
    feature_weight.copy_(normalized.to(device=feature_weight.device, dtype=feature_weight.dtype))
    final_avg_norm = average_row_norm(feature_weight)
    return EncoderEmbeddingInitResult(
        mode=mode,
        n_sae_features=n_sae_features,
        sae_dim=sae_dim,
        encoder_dim=encoder_dim,
        target_avg_norm=float(target_avg_norm.item()),
        projected_avg_norm=float(projected_avg_norm.item()),
        final_avg_norm=float(final_avg_norm.item()),
        vocab_size=int(vocab_size),
        cluster_vectors=cluster_vectors,
        clusters_averaged=clusters_averaged,
        members_averaged=members_averaged,
    )


def maybe_initialize_encoder_feature_embeddings_from_sae(
    model: Any,
    cfg: Mapping[str, Any] | Any,
    *,
    seed: int,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> EncoderEmbeddingInitResult | None:
    """Apply configured SAE initialization, or return ``None`` for no-op mode."""
    mode = parse_encoder_embedding_init(
        _mapping_get(cfg, "initialize_encoder_embeddings_from_sae", None)
    )
    cluster_vectors = parse_encoder_embedding_cluster_vectors(
        _mapping_get(cfg, "initialize_encoder_embeddings_cluster_vectors", None)
    )
    _validate_cluster_vector_configuration(cluster_vectors)
    if mode is None:
        return None
    sae_cfg = _mapping_get(cfg, "sae")
    if sae_cfg is None:
        raise ValueError("initialize_encoder_embeddings_from_sae requires a configured sae: block.")
    sae_decoder_matrix = load_sae_decoder_matrix_from_cfg(sae_cfg)
    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        sae_decoder_matrix,
        mode,
        seed=seed,
        cluster_vectors=cluster_vectors,
        feature_clusters=feature_clusters,
    )
    # These three norms exist to answer one question — "did the rescale work?" — and on a
    # compacted run they could not be read. The line said n_sae_features=65536 while only V
    # rows were written, and `final_avg_norm` (measured over the V WRITTEN rows) no longer
    # equals `target_avg_norm` (deliberately the FULL table's norm), where pre-#695 they
    # agreed by construction. Without the explanation, a correct init looks broken.
    compacted = result.vocab_size != result.n_sae_features
    logger.info(
        "Initialized encoder.feature_embed.weight from SAE using mode={} "
        "(cluster_vectors={}, clusters_averaged={}, members_averaged={}, "
        "n_sae_features={}, vocab_size={}, sae_dim={}, encoder_dim={}, "
        "target_avg_norm={:.6f}, projected_avg_norm={:.6f}, final_avg_norm={:.6f}){}",
        result.mode.value,
        result.cluster_vectors.value,
        result.clusters_averaged,
        result.members_averaged,
        result.n_sae_features,
        result.vocab_size,
        result.sae_dim,
        result.encoder_dim,
        result.target_avg_norm,
        result.projected_avg_norm,
        result.final_avg_norm,
        (
            f"\n  encoder is COMPACTED: {result.vocab_size} of {result.n_sae_features} "
            "rows written. target_avg_norm is the FULL table's norm — the number an "
            "UNCOMPACTED run would have used, which is what keeps `random` bitwise — "
            "while final_avg_norm is measured over the V written rows only. They are NOT "
            "expected to be equal: under `representative`, final is a sample of the "
            "reference's rows, not the whole table; under `mean`, retained representative "
            "rows additionally use mode-specific member means (post-rescale projection "
            "for `random`, pre-basis W_dec for `svd`). A gap therefore does not mean that "
            "the full-table rescale failed."
            if compacted
            else ""
        ),
    )
    return result
