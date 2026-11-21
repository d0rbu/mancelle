"""Public feature-contract facade for clustered evaluation.

Assemble resolved identity and checkpoint snapshots into contracts and
representative targets; resolve filesystem-backed preflight locally or
across DDP ranks; check loaded state; and read/write sidecars and run markers.
Identity primitives and checkpoint helpers are re-exported through
``__all__`` so callers can keep using this public import path.

The fingerprint hashes canonical semantic content: algorithm versions,
mask/cluster mappings, concept-source projection, statistics, feature-directory
and SAE identity, checkpoint identity, and selection identity. The separately
recorded provenance fields, including input paths, do not enter that
fingerprint. ``n_sae_features`` denotes the original SAE width; a
checkpoint's persistent feature vocabulary may be smaller after compaction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.distributed as dist
from beartype import beartype
from loguru import logger

from errors import FeatureClustersError, FeatureMaskError, FeatureVocabError
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, resolve_feature_mask

# Re-exported / reused from the checkpoint layer (B2a+B2b). The five
# underscore-prefixed names are private to that module too (not in its
# `__all__`); `resolve_cluster_feature_contract` below is their only caller
# outside `cluster_checkpoint_pin` itself, mirroring the same "public facade
# imports the layer's private orchestration helpers" shape B2 already uses
# for B1's primitives.
from steering.eval.cluster_checkpoint_pin import (
    CheckpointPin,
    CheckpointSelectionContract,
    _assert_checkpoint_agreement,
    _default_sae_loader,
    _load_json_mapping,
    _resolve_and_validate_sae,
    _selection_contract,
    _statistics_identity,
    resolve_checkpoint_pin,
    resolve_eval_checkpoint_directory,
    resolve_selection_manifest_path,
)
from steering.eval.cluster_config import parse_cluster_eval_paths

# Reuse the identity layer's public types and private canonicalization,
# hashing, and feature-ID validation helpers rather than duplicating them.
from steering.eval.cluster_contract_identity import (
    CONCEPT_PROJECTION_VERSION,
    FEATURE_CONTRACT_SCHEMA_VERSION,
    FEATURE_DIFFMEAN_PROJECTION_VERSION,
    GENERIC_ROOFLINE_SIGNAL_VERSION,
    MAX_SAFE_SELECTED_FEATURE_WIDTH,
    PACKED_PERCENTILE_SELECTION_VERSION,
    QC_TARGET_REDUCTION_VERSION,
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    _canonical_json,
    _sha256_json,
    _validate_feature_id_tuple,
    stat_signature_now,
    stream_hash_file_stable,
)
from steering.feature_paths import parse_feature_dir
from steering.immutable_io import ImmutableArtifactError, install_bytes_once
from steering.manual_feature_ids import (
    parse_feature_ids_txt,
    parse_manual_feature_yaml_ids,
)
from steering.refined_types import (
    LowercaseSha256Like,
    NonNegativeIntLike,
    NonNegativeIntTupleLike,
    UniqueNonEmptyNonNegativeIntTupleLike,
)
from steering.sae_resolver import ResolvedSAE
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY


@beartype
@dataclass(frozen=True)
class ClusterFeatureTarget:
    """Representative target threaded through clustered AxBench QC."""

    representative_id: NonNegativeIntLike
    allowed_member_ids: UniqueNonEmptyNonNegativeIntTupleLike
    projecting_source_ids: NonNegativeIntTupleLike
    feature_contract_fingerprint: LowercaseSha256Like

    def __post_init__(self) -> None:
        for name, ids in (
            ("allowed_member_ids", self.allowed_member_ids),
            ("projecting_source_ids", self.projecting_source_ids),
        ):
            if not isinstance(ids, tuple) or tuple(sorted(set(ids))) != ids:
                raise ClusterFeatureContractError(
                    f"{name} must be a sorted unique tuple of strict ints"
                )
        if self.allowed_member_ids[0] != self.representative_id:
            raise ClusterFeatureContractError(
                "allowed_member_ids must be non-empty and start with representative_id"
            )


@dataclass(frozen=True)
class ClusterFeatureContract:
    """Immutable resolved clustered-evaluation contract."""

    n_sae_features: int
    feature_mask: FeatureMask
    feature_clusters: ResolvedFeatureClusters
    consolidation: FeatureClusterConsolidation = field(compare=False, repr=False)
    representative_universe: tuple[int, ...]
    representative_to_allowed_members: tuple[tuple[int, tuple[int, ...]], ...]
    source_to_representative: tuple[tuple[int, int], ...]
    representative_to_projecting_sources: tuple[tuple[int, tuple[int, ...]], ...]
    statistics: FeatureStatisticsIdentity
    feature_directory: FeatureDirectoryIdentity
    sae: SaeIdentity
    checkpoint: CheckpointPin
    selection: CheckpointSelectionContract
    concept_source_mode: str
    concept_source_domain: tuple[int, ...]
    semantic_json: str
    provenance_json: str
    feature_contract_fingerprint: str
    schema_version: int = FEATURE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_CONTRACT_SCHEMA_VERSION:
            raise ClusterFeatureContractError(
                f"unsupported feature contract schema {self.schema_version}"
            )
        if _sha256_json(json.loads(self.semantic_json)) != self.feature_contract_fingerprint:
            raise ClusterFeatureContractError(
                "feature contract fingerprint does not match semantic content"
            )
        expected_reps = tuple(
            int(value)
            for value in t.nonzero(
                self.consolidation.cluster_allowed_universe,
                as_tuple=False,
            )
            .squeeze(1)
            .tolist()
        )
        if self.representative_universe != expected_reps:
            raise ClusterFeatureContractError(
                "representative universe does not match cluster consolidation"
            )

    def semantic_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.semantic_json))

    def provenance_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.provenance_json))

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_contract_fingerprint": self.feature_contract_fingerprint,
            "semantic": self.semantic_dict(),
            "provenance": self.provenance_dict(),
        }

    def target_for_representative(self, representative_id: int) -> ClusterFeatureTarget:
        """Resolve one representative without projecting or accepting a sibling ID."""

        if type(representative_id) is not int:
            raise ClusterFeatureContractError("cluster representative target must be a strict int")
        members_by_rep = dict(self.representative_to_allowed_members)
        if representative_id not in members_by_rep:
            raise ClusterFeatureContractError(
                f"feature ID {representative_id} is not in the representative universe"
            )
        projecting_by_rep = dict(self.representative_to_projecting_sources)
        return ClusterFeatureTarget(
            representative_id=representative_id,
            allowed_member_ids=members_by_rep[representative_id],
            projecting_source_ids=projecting_by_rep.get(representative_id, ()),
            feature_contract_fingerprint=self.feature_contract_fingerprint,
        )

@dataclass(frozen=True)
class ClusterFeatureContractRequest:
    """Filesystem-backed inputs required by production preflight."""

    feature_dir: str | Path
    feature_statistics_path: str | Path
    feature_mask: Mapping[str, Any]
    feature_clusters: Mapping[str, Any]
    checkpoint_dir: str | Path | None = None
    checkpoint_source: str | None = None
    checkpoint_model_tag: str | None = None
    checkpoint_step: int | None = None
    checkpoint_tag: str | None = None
    selection_manifest_path: str | Path | None = None
    concept_feature_whitelist_path: str | Path | None = None
    gcg_top_k_pos: int | None = None
    gcg_top_k_neg: int | None = None
    #: Producer-side ``selection_splice_batch_size`` used by
    #: ``precompute_selected_features`` when it wrote this checkpoint's
    #: ``sel/<hash>/`` overlay. It participates in the selection hash but is
    #: NOT recorded in checkpoint metadata, so a splice-trained checkpoint
    #: whose producer batched (> 1) must state it here or manifest discovery
    #: finds nothing. ``None`` reproduces the unbatched producer identity.
    #: Same explicit-override shape as ``gcg_top_k_pos``/``gcg_top_k_neg``.
    selection_splice_batch_size: int | None = None
    #: Methods this run will execute. Required only when the checkpoint carries
    #: an auxiliary ``feature_prediction`` head, which is supported solely for
    #: methods that never read it
    #: (``CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS``). Left None for a
    #: checkpoint without a head, which every existing caller is.
    enabled_methods: tuple[str, ...] | None = None
    #: Deliberately bypass even a valid packed fingerprint cache and execute
    #: every canonical packed/selection body validator. Rank 0 owns the one
    #: shared-cache refresh under DDP.
    force_cluster_shard_revalidation: bool = False
    expected_feature_contract_document: Mapping[str, Any] | None = None
    expected_feature_contract_path: str | Path | None = None


#: Trailing overlay written by ``scripts.preprocess.splice_feature_shards``
#: onto an otherwise-canonical feature directory (issue #590). Its documented
#: default layout is ``<feature_dir>/splice/cut_frac{cut_fraction}/seed{seed}``
#: (see ``configs/extraction/splice_feature_shards.yaml``).
_SPLICE_OVERLAY_RE = re.compile(r"^cut_frac\d+(?:\.\d+)?$")


def _strip_splice_overlay(feature_dir: str) -> tuple[str, str | None]:
    """Split a checkpoint-recorded feature dir into (canonical stack, overlay).

    A checkpoint trained on spliced counterfactual features records the
    *derived* directory in ``user_config.feature_dir``. The splice overlay
    describes how the training corpus was recombined; it does not change the
    SAE feature stack (segment / layer / activation / pooling / width / model /
    phase), which is what this module's identity check is about. Strip it so
    the canonical prefix can be parsed, and return it so the caller can log
    what was ignored rather than dropping it silently.

    Returns the input unchanged (with ``None``) when no overlay is present.
    """

    parts = Path(feature_dir).parts
    if (
        len(parts) >= 4
        and parts[-3] == "splice"
        and _SPLICE_OVERLAY_RE.match(parts[-2])
        and parts[-1].startswith("seed")
    ):
        return str(Path(*parts[:-3])), "/".join(parts[-3:])
    return feature_dir, None


def _assert_checkpoint_feature_directory_identity(
    metadata: Mapping[str, Any],
    configured: FeatureDirectoryIdentity,
) -> None:
    """Cross-check the configured feature directory against checkpoint training
    provenance. Kept in B3 (not B2a/B2b, whose ``_statistics_identity``
    produces the ``FeatureDirectoryIdentity`` this consumes): its only caller
    is this module's own ``resolve_cluster_feature_contract`` orchestration,
    so there is no B2-internal caller to justify placing it there instead.
    """

    user_config = metadata.get("user_config")
    if not isinstance(user_config, Mapping):
        raise ClusterFeatureContractError("checkpoint user_config is missing")
    # A conditioned-SFT checkpoint records the directory it was *launched* with under
    # ``user_config.feature_dir`` (node-local staging can make that a flat copy such as
    # ``/scratch/.../features/s32/train``) but publishes the canonical training directory in its
    # online-SAE lineage; that lineage is the authoritative provenance when present.
    if ONLINE_SAE_LINEAGE_META_KEY in metadata:
        lineage = metadata[ONLINE_SAE_LINEAGE_META_KEY]
        if not isinstance(lineage, Mapping):
            raise ClusterFeatureContractError("checkpoint online-SAE lineage must be a mapping")
        checkpoint_feature_dir = lineage.get("feature_dir")
        if not isinstance(checkpoint_feature_dir, str) or not checkpoint_feature_dir.strip():
            raise ClusterFeatureContractError("checkpoint online-SAE lineage feature_dir must be nonblank")
        if isinstance(user_config.get("feature_dir"), str) and user_config["feature_dir"] != checkpoint_feature_dir:
            logger.info(
                "checkpoint user_config.feature_dir {} differs from its online-SAE lineage feature_dir {}; "
                "using the lineage",
                user_config["feature_dir"],
                checkpoint_feature_dir,
            )
    else:
        checkpoint_feature_dir = user_config.get("feature_dir")
    if not isinstance(checkpoint_feature_dir, str) or not checkpoint_feature_dir:
        raise ClusterFeatureContractError(
            "clustered checkpoint lacks historical user_config.feature_dir provenance"
        )
    canonical_feature_dir, splice_overlay = _strip_splice_overlay(checkpoint_feature_dir)
    if splice_overlay is not None:
        logger.info(
            "clustered checkpoint was trained on spliced features; comparing the "
            "canonical feature stack {} and ignoring the splice overlay {} "
            "(corpus recombination does not change SAE stack identity)",
            canonical_feature_dir,
            splice_overlay,
        )
    try:
        checkpoint_identity = FeatureDirectoryIdentity.from_components(
            parse_feature_dir(canonical_feature_dir)
        )
    except (TypeError, ValueError) as exc:
        raise ClusterFeatureContractError(
            f"checkpoint user_config.feature_dir is not a canonical feature directory: {exc}"
        ) from exc
    if checkpoint_identity != configured:
        raise ClusterFeatureContractError(
            "configured feature-directory identity does not match checkpoint training "
            f"metadata: configured={configured.canonical_dict()!r}, "
            f"checkpoint={checkpoint_identity.canonical_dict()!r}"
        )


def _parse_concept_source_ids(
    path: str | Path | None,
    *,
    stats_domain: tuple[int, ...],
    n_sae_features: int,
    feature_mask: FeatureMask,
) -> tuple[str, tuple[int, ...]]:
    if path is None:
        return "statistics_domain", stats_domain
    source_path = Path(path).expanduser().resolve(strict=True)
    try:
        if source_path.suffix.lower() in {".yaml", ".yml"}:
            raw_ids = parse_manual_feature_yaml_ids(source_path)
        elif source_path.suffix.lower() == ".txt":
            raw_ids = parse_feature_ids_txt(source_path)
        else:
            raise ClusterFeatureContractError(
                "concept feature whitelist must use .txt, .yaml, or .yml"
            )
    except Exception as exc:
        if isinstance(exc, ClusterFeatureContractError):
            raise
        raise ClusterFeatureContractError(
            f"invalid concept feature whitelist {source_path}: {exc}"
        ) from exc
    if not raw_ids or len(raw_ids) != len(set(raw_ids)):
        raise ClusterFeatureContractError(
            "concept feature whitelist must be non-empty and contain no duplicates"
        )
    source_ids = tuple(sorted(raw_ids))
    _validate_feature_id_tuple(
        source_ids,
        n_sae_features=n_sae_features,
        field_name="concept feature whitelist",
    )
    stats_set = set(stats_domain)
    allowed = feature_mask.allowed_universe_mask(n_sae_features)
    missing_stats = [feature_id for feature_id in source_ids if feature_id not in stats_set]
    disallowed = [feature_id for feature_id in source_ids if not bool(allowed[feature_id])]
    if missing_stats or disallowed:
        raise ClusterFeatureContractError(
            "concept feature whitelist is incompatible with clustered evaluation: "
            f"missing_statistics={missing_stats}, mask_disallowed={disallowed}"
        )
    return "explicit_whitelist", source_ids


def _concept_source_identity(
    request: ClusterFeatureContractRequest,
    *,
    expected_document: Mapping[str, Any] | None,
    stats_domain: tuple[int, ...],
    n_sae_features: int,
    feature_mask: FeatureMask,
) -> tuple[str, tuple[int, ...]]:
    """Resolve source identity from AxBench config or a BPB source marker.

    BPB does not repeat AxBench's independent concept whitelist in its public
    config.  Instead it supplies the source run's validated feature-contract
    document.  Only the concept-source coordinate is borrowed before the
    target contract is rebuilt; the complete rebuilt fingerprint is compared
    below, so no other target/config mismatch can be hidden by the source.
    """

    expected_raw = expected_document
    if expected_raw is None:
        return _parse_concept_source_ids(
            request.concept_feature_whitelist_path,
            stats_domain=stats_domain,
            n_sae_features=n_sae_features,
            feature_mask=feature_mask,
        )
    if request.concept_feature_whitelist_path is not None:
        raise ClusterFeatureContractError(
            "concept_feature_whitelist_path and expected_feature_contract_document "
            "are mutually exclusive"
        )
    expected = validate_feature_contract_document(expected_raw)
    semantic = expected["semantic"]
    concept_source = semantic.get("concept_source")
    if not isinstance(concept_source, Mapping):
        raise ClusterFeatureContractError(
            "expected source feature contract is missing concept_source semantics"
        )
    mode = concept_source.get("mode")
    raw_domain = concept_source.get("domain")
    if mode not in {"explicit_whitelist", "statistics_domain"}:
        raise ClusterFeatureContractError(
            f"expected source feature contract has invalid concept source mode {mode!r}"
        )
    if not isinstance(raw_domain, list) or any(
        type(feature_id) is not int for feature_id in raw_domain
    ):
        raise ClusterFeatureContractError(
            "expected source feature contract concept domain must be a list of strict ints"
        )
    domain = tuple(raw_domain)
    _validate_feature_id_tuple(
        domain,
        n_sae_features=n_sae_features,
        field_name="expected source feature-contract concept domain",
    )
    return str(mode), domain


def _load_expected_feature_contract(
    request: ClusterFeatureContractRequest,
) -> dict[str, Any] | None:
    if (
        request.expected_feature_contract_document is not None
        and request.expected_feature_contract_path is not None
    ):
        raise ClusterFeatureContractError(
            "expected_feature_contract_document and expected_feature_contract_path "
            "are mutually exclusive"
        )
    if request.expected_feature_contract_path is not None:
        return load_feature_contract_sidecar(request.expected_feature_contract_path)
    if request.expected_feature_contract_document is not None:
        return validate_feature_contract_document(request.expected_feature_contract_document)
    return None


def _group_allowed_ids_by_representative(
    allowed_ids: Sequence[int],
    representative_of_feature_id: Sequence[int],
) -> dict[int, list[int]]:
    """Group ``allowed_ids`` by their representative in a single pass (#931).

    Performs **exactly** ``len(allowed_ids)`` lookups into
    ``representative_of_feature_id`` and preserves ``allowed_ids`` order within each
    group, which is what makes the grouping byte-identical to the
    ``representatives x allowed_ids`` filter it replaced.

    Extracted so that "one pass" is directly assertable: a test can pass an
    access-counting sequence and require the lookup count to equal
    ``len(allowed_ids)``. Counting tensor reads alone cannot catch a regression that
    reintroduces the quadratic scan over the already-materialized Python list.
    """
    grouped: dict[int, list[int]] = {}
    for feature_id in allowed_ids:
        grouped.setdefault(representative_of_feature_id[feature_id], []).append(feature_id)
    return grouped


def build_cluster_feature_contract(
    *,
    feature_mask: FeatureMask,
    feature_clusters: ResolvedFeatureClusters,
    statistics: FeatureStatisticsIdentity,
    feature_directory: FeatureDirectoryIdentity,
    sae: SaeIdentity,
    checkpoint: CheckpointPin,
    selection: CheckpointSelectionContract,
    concept_source_mode: str,
    concept_source_domain: tuple[int, ...],
) -> ClusterFeatureContract:
    """Pure constructor from validated provenance snapshots.

    No filesystem I/O, no SAE loading: every input is an already-resolved
    identity/pin snapshot. This is deliberately the boundary the fingerprint-
    differential gate exercises (see ``tests/steering/test_cluster_feature_
    contract_fingerprint_differential.py``) — it is the ONE function whose
    output the oracle and the B1/B2a/B2b/B3 stack can be driven through with
    identical, hand-built inputs and no real checkpoint/SAE-download
    dependency.
    """

    n_sae_features = statistics.n_sae_features
    if not (
        checkpoint.n_sae_features
        == feature_clusters.n_sae_features
        == sae.actual_loaded_width
        == n_sae_features
    ):
        raise ClusterFeatureContractError(
            "original SAE width disagrees across statistics, checkpoint, clusters, and SAE"
        )
    if concept_source_mode not in {"explicit_whitelist", "statistics_domain"}:
        raise ClusterFeatureContractError(f"unknown concept source mode {concept_source_mode!r}")
    _validate_feature_id_tuple(
        concept_source_domain,
        n_sae_features=n_sae_features,
        field_name="concept source domain",
    )
    consolidation = FeatureClusterConsolidation.build(feature_clusters, feature_mask)
    allowed = consolidation.allowed_universe
    representatives = tuple(
        int(value)
        for value in t.nonzero(
            consolidation.cluster_allowed_universe,
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    allowed_ids = tuple(
        int(value) for value in t.nonzero(allowed, as_tuple=False).squeeze(1).tolist()
    )
    # Materialize `rep_of` once. Reading it element-by-element off the tensor
    # costs a device round-trip per access; the previous form did that inside a
    # `representatives x allowed_ids` double scan, i.e. O(|R|*|A|) syncs. At the
    # SAE widths this pipeline actually runs that is ~268M accesses (~190s) at
    # 16k and ~4.3B (~51min) at 65k, paid once per clustered run before any real
    # work. Grouping in ONE pass over `allowed_ids` is O(|A|) with a single sync.
    #
    # Output is unchanged, which the fingerprint differential's byte-equal TIER 1
    # cells enforce (`members` feeds `representative_to_allowed_members`, hence
    # the contract fingerprint): `representatives` still drives the outer order,
    # and appending in `allowed_ids` order reproduces the old filter's inner
    # order exactly. `int(...)` is kept so a non-int64 `rep_of` truncates exactly
    # as the old `int(tensor.item())` did.
    representative_of_feature_id = [int(value) for value in consolidation.rep_of.tolist()]
    allowed_members_by_representative = _group_allowed_ids_by_representative(
        allowed_ids,
        representative_of_feature_id,
    )
    members: list[tuple[int, tuple[int, ...]]] = [
        (representative, tuple(allowed_members_by_representative.get(representative, ())))
        for representative in representatives
    ]
    stats_set = set(statistics.feature_id_domain)
    if concept_source_mode == "explicit_whitelist":
        missing_stats = [value for value in concept_source_domain if value not in stats_set]
        disallowed = [value for value in concept_source_domain if not bool(allowed[value])]
        if missing_stats or disallowed:
            raise ClusterFeatureContractError(
                "explicit concept source domain contains unavailable IDs: "
                f"missing_statistics={missing_stats}, mask_disallowed={disallowed}"
            )
        surviving_sources = concept_source_domain
    else:
        if concept_source_domain != statistics.feature_id_domain:
            raise ClusterFeatureContractError(
                "statistics-domain concept source must exactly equal the current "
                "per_feature_stats ID domain"
            )
        surviving_sources = tuple(
            feature_id for feature_id in concept_source_domain if bool(allowed[feature_id])
        )
    # Same reason as above: reuse the already-materialized list instead of one
    # device sync per surviving source. `surviving_sources` is validated to lie
    # within `n_sae_features` (`_validate_feature_id_tuple` above, and the
    # `allowed` filter in the statistics-domain branch), so every id is a valid
    # index into `representative_of_feature_id`.
    source_projection = tuple(
        (
            source_id,
            representative_of_feature_id[source_id],
        )
        for source_id in surviving_sources
    )
    # Codex fix batch finding 4 (MEDIUM), NAMED BUG-FIX DEVIATION from the
    # oracle (see the differential's own "empty_concept_projection_after_
    # masking" cell): mask filtering above can remove every statistics-
    # domain source (e.g. the configured statistics/concept domain and the
    # feature mask's allowed universe are disjoint), leaving
    # `surviving_sources`/`source_projection` empty. The oracle happily
    # returns a "successful" contract in that case, with
    # `source_to_representative == ()` and an empty `projecting_source_ids`
    # for every representative -- a contract that can never usefully target
    # anything downstream. This is the same E4 non-emptiness invariant
    # `effective_feature_vocab` already guards at the encoder-vocabulary
    # layer (`steering/conditional/effective_feature_vocab.py`: "A valid
    # mask/consolidation leaves at least one effective feature"), applied
    # here at the concept-source-projection layer: a valid configuration
    # must leave at least one projecting source.
    if not source_projection:
        raise ClusterFeatureContractError(
            "concept source projection is empty after mask filtering: no "
            f"{concept_source_mode!r} source id survives the configured "
            "feature mask, so no representative would ever receive a "
            "projecting source (E4 non-emptiness; see "
            "effective_feature_vocab's own E4 invariant)"
        )
    projecting: dict[int, list[int]] = {}
    for source_id, representative in source_projection:
        projecting.setdefault(representative, []).append(source_id)
    representative_to_sources = tuple(
        (representative, tuple(projecting.get(representative, [])))
        for representative in representatives
    )
    semantic = {
        "schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "algorithms": {
            "concept_projection": CONCEPT_PROJECTION_VERSION,
            "qc_target_reduction": QC_TARGET_REDUCTION_VERSION,
            "packed_percentile_selection": PACKED_PERCENTILE_SELECTION_VERSION,
            "feature_diffmean_projection": FEATURE_DIFFMEAN_PROJECTION_VERSION,
            "generic_roofline_signal": GENERIC_ROOFLINE_SIGNAL_VERSION,
        },
        "n_sae_features": n_sae_features,
        "feature_mask": {
            "canonical": feature_mask.canonical_dict(),
            "allowed_ids_sha256": _sha256_json(list(allowed_ids)),
            "allowed_count": len(allowed_ids),
        },
        "feature_clusters": {
            "descriptor": feature_clusters.descriptor_dict(),
            "canonical": feature_clusters.canonical_dict(),
            "representative_to_allowed_members": [
                [representative, list(member_ids)] for representative, member_ids in members
            ],
            "consolidated_away": [
                int(value)
                for value in t.nonzero(
                    consolidation.consolidated_away,
                    as_tuple=False,
                )
                .squeeze(1)
                .tolist()
            ],
            "representative_universe": list(representatives),
        },
        "concept_source": {
            "mode": concept_source_mode,
            "domain": list(concept_source_domain),
            "source_to_representative": [list(pair) for pair in source_projection],
        },
        "statistics": statistics.canonical_dict(),
        "feature_directory": feature_directory.canonical_dict(),
        "sae": sae.canonical_dict(),
        "checkpoint": checkpoint.semantic_dict(),
        "selection": selection.canonical_dict(),
    }
    fingerprint = _sha256_json(semantic)
    provenance = {
        **checkpoint.provenance_dict(),
        **selection.provenance_dict(),
        "feature_clusters_path": str(feature_clusters.source_path),
    }
    return ClusterFeatureContract(
        n_sae_features=n_sae_features,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        consolidation=consolidation,
        representative_universe=representatives,
        representative_to_allowed_members=tuple(members),
        source_to_representative=source_projection,
        representative_to_projecting_sources=representative_to_sources,
        statistics=statistics,
        feature_directory=feature_directory,
        sae=sae,
        checkpoint=checkpoint,
        selection=selection,
        concept_source_mode=concept_source_mode,
        concept_source_domain=concept_source_domain,
        semantic_json=_canonical_json(semantic),
        provenance_json=_canonical_json(provenance),
        feature_contract_fingerprint=fingerprint,
    )


def resolve_cluster_feature_contract(
    request: ClusterFeatureContractRequest,
    *,
    sae_loader: Callable[[ResolvedSAE], Any] | None = None,
    allow_packed_cache_write: bool = True,
) -> ClusterFeatureContract:
    """Perform filesystem-backed clustered preflight and freeze its identity."""

    expected_feature_contract = _load_expected_feature_contract(request)
    parsed_paths = parse_cluster_eval_paths(
        feature_mask=request.feature_mask,
        feature_clusters=request.feature_clusters,
    )
    if parsed_paths is None:
        raise ClusterFeatureContractError(
            "resolve_cluster_feature_contract requires active mask and cluster paths"
        )
    checkpoint_dir, resolved_tag = resolve_eval_checkpoint_directory(
        checkpoint_dir=request.checkpoint_dir,
        checkpoint_source=request.checkpoint_source,
        checkpoint_model_tag=request.checkpoint_model_tag,
    )
    checkpoint, metadata, _ = resolve_checkpoint_pin(
        checkpoint_dir,
        step=request.checkpoint_step,
        resolved_tag=request.checkpoint_tag or resolved_tag,
    )
    mask_cfg, cluster_cfg = parsed_paths
    config_block = {
        "feature_mask": mask_cfg.canonical_dict(),
        "feature_clusters": cluster_cfg.canonical_dict(),
    }
    try:
        feature_mask = resolve_feature_mask(config_block, checkpoint.n_sae_features)
        feature_clusters = resolve_feature_clusters(
            config_block,
            checkpoint.n_sae_features,
        )
    except (FeatureMaskError, FeatureClustersError) as exc:
        raise ClusterFeatureContractError(str(exc)) from exc
    if feature_mask is None or feature_clusters is None:
        raise ClusterFeatureContractError("cluster preflight resolved a disabled contract")
    _assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
    )
    feature_dir = Path(request.feature_dir).expanduser().resolve(strict=True)
    statistics_path = Path(request.feature_statistics_path).expanduser().resolve(strict=True)
    statistics, feature_directory, _ = _statistics_identity(
        feature_dir=feature_dir,
        statistics_path=statistics_path,
        n_sae_features=checkpoint.n_sae_features,
    )
    _assert_checkpoint_feature_directory_identity(metadata, feature_directory)
    sae = _resolve_and_validate_sae(
        metadata=metadata,
        feature_directory=feature_directory,
        n_sae_features=checkpoint.n_sae_features,
        sae_loader=sae_loader or _default_sae_loader,
    )
    consolidation = FeatureClusterConsolidation.build(feature_clusters, feature_mask)
    selection_manifest_path = (
        resolve_selection_manifest_path(
            feature_dir=feature_dir,
            metadata=metadata,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            gcg_top_k_pos=request.gcg_top_k_pos,
            gcg_top_k_neg=request.gcg_top_k_neg,
            selection_splice_batch_size=request.selection_splice_batch_size,
            enabled_methods=request.enabled_methods,
        )
        if request.selection_manifest_path is None
        else Path(request.selection_manifest_path).expanduser().resolve(strict=True)
    )
    selection = _selection_contract(
        metadata=metadata,
        manifest_path=selection_manifest_path,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        consolidation=consolidation,
        percentile_checksum=statistics.percentile_table_checksum,
        gcg_top_k_pos=request.gcg_top_k_pos,
        gcg_top_k_neg=request.gcg_top_k_neg,
        selection_splice_batch_size=request.selection_splice_batch_size,
        enabled_methods=request.enabled_methods,
        force_cluster_shard_revalidation=(
            request.force_cluster_shard_revalidation
        ),
        allow_packed_cache_write=allow_packed_cache_write,
    )
    source_mode, source_domain = _concept_source_identity(
        request,
        expected_document=expected_feature_contract,
        stats_domain=statistics.feature_id_domain,
        n_sae_features=checkpoint.n_sae_features,
        feature_mask=feature_mask,
    )
    contract = build_cluster_feature_contract(
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        statistics=statistics,
        feature_directory=feature_directory,
        sae=sae,
        checkpoint=checkpoint,
        selection=selection,
        concept_source_mode=source_mode,
        concept_source_domain=source_domain,
    )
    if expected_feature_contract is not None:
        expected = expected_feature_contract
        if expected["feature_contract_fingerprint"] != contract.feature_contract_fingerprint:
            raise ClusterFeatureContractError(
                "configured mask/clusters and target checkpoint do not match the "
                "clustered AxBench source feature contract: "
                f"source={expected['feature_contract_fingerprint']}, "
                f"target={contract.feature_contract_fingerprint}"
            )
    # The paths are provenance only, but record every file-backed source.
    provenance = contract.provenance_dict()
    provenance.update(
        {
            "feature_dir": str(feature_dir),
            "feature_statistics_path": str(statistics_path),
            "feature_mask_path": mask_cfg.path,
            "feature_clusters_path": cluster_cfg.path,
            "selection_manifest_path": str(selection_manifest_path),
            "concept_feature_whitelist_path": (
                None
                if request.concept_feature_whitelist_path is None
                else str(Path(request.concept_feature_whitelist_path).expanduser().resolve())
            ),
        }
    )
    return replace(
        contract,
        provenance_json=_canonical_json(provenance),
    )


def resolve_cluster_feature_contract_ddp(
    request: ClusterFeatureContractRequest,
    *,
    sae_loader: Callable[[ResolvedSAE], Any] | None = None,
) -> ClusterFeatureContract:
    """Resolve on every rank with rank-0 canonical error/fingerprint broadcast.

    All ranks participate in the same collectives even after a local failure,
    preventing one corrupt/missing mount from stranding peers at a barrier.
    Rank 0 resolves first and broadcasts its canonical result or error. On
    success every other rank independently reconstructs the contract, then an
    unconditional gather proves identical fingerprints before returning.
    """

    if not dist.is_available() or not dist.is_initialized():
        return resolve_cluster_feature_contract(request, sae_loader=sae_loader)

    rank = dist.get_rank()
    rank_zero_contract: ClusterFeatureContract | None = None
    rank_zero_status: dict[str, str | None]
    if rank == 0:
        try:
            rank_zero_contract = resolve_cluster_feature_contract(
                request,
                sae_loader=sae_loader,
            )
            rank_zero_status = {
                "error": None,
                "fingerprint": rank_zero_contract.feature_contract_fingerprint,
            }
        except Exception as exc:
            rank_zero_status = {
                "error": f"{type(exc).__name__}: {exc}",
                "fingerprint": None,
            }
    else:
        rank_zero_status = {"error": None, "fingerprint": None}
    broadcast: list[dict[str, str | None]] = [rank_zero_status]
    dist.broadcast_object_list(broadcast, src=0)
    canonical_status = broadcast[0]
    local_contract = rank_zero_contract
    local_error: str | None = (
        rank_zero_status["error"] if rank == 0 else None
    )
    if rank != 0 and canonical_status["error"] is None:
        try:
            local_contract = resolve_cluster_feature_contract(
                replace(
                    request,
                    force_cluster_shard_revalidation=False,
                ),
                sae_loader=sae_loader,
                allow_packed_cache_write=False,
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    if (
        canonical_status["error"] is None
        and local_contract is None
        and local_error is None
    ):
        local_error = (
            "ClusterFeatureContractError: cluster preflight resolver returned "
            "without a local contract"
        )
    local_fingerprint = (
        None if local_contract is None else local_contract.feature_contract_fingerprint
    )
    local_status = {
        "rank": rank,
        "error": local_error,
        "fingerprint": local_fingerprint,
    }
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_status)
    failures = [status for status in gathered if status and status["error"] is not None]
    fingerprints = {
        status["fingerprint"] for status in gathered if status and status["fingerprint"] is not None
    }
    if failures:
        raise ClusterFeatureContractError(
            f"cluster preflight failed on one or more ranks: {failures}"
        )
    if fingerprints != {canonical_status["fingerprint"]}:
        raise ClusterFeatureContractError(
            "cluster preflight ranks reconstructed different feature-contract "
            f"fingerprints: {sorted(fingerprints)}"
        )
    if local_contract is None:
        raise ClusterFeatureContractError("cluster preflight completed without a local contract")
    return local_contract


def assert_cluster_checkpoint_pin_unchanged(contract: ClusterFeatureContract) -> None:
    """Rehash immediately before first target forward and recheck vocabulary/meta."""

    current, metadata, _ = resolve_checkpoint_pin(
        contract.checkpoint.checkpoint_dir,
        step=contract.checkpoint.step,
        resolved_tag=contract.checkpoint.resolved_tag,
    )
    if current.semantic_dict() != contract.checkpoint.semantic_dict():
        raise ClusterFeatureContractError("clustered target checkpoint changed after preflight")
    _assert_checkpoint_agreement(
        checkpoint=current,
        metadata=metadata,
        feature_mask=contract.feature_mask,
        feature_clusters=contract.feature_clusters,
    )


def assert_cluster_contract_loaded_state(
    contract: ClusterFeatureContract,
    model: Any,
    metadata: Mapping[str, Any],
    *,
    surface: str,
) -> None:
    """Assert metadata and actual in-memory encoder vocabulary without I/O.

    This is the lightweight exact-contract boundary for consumers that already
    hold a loaded target and have separately established the checkpoint byte
    pin. It is intentionally stronger than a cluster-descriptor check: the
    mask, authoritative persistent vocabulary, original width, and loaded
    buffers must all agree bidirectionally.
    """

    _assert_checkpoint_agreement(
        checkpoint=contract.checkpoint,
        metadata=metadata,
        feature_mask=contract.feature_mask,
        feature_clusters=contract.feature_clusters,
    )
    encoder = getattr(model, "encoder", None)
    config = getattr(encoder, "config", None)
    actual_width = getattr(config, "n_sae_features", None)
    if type(actual_width) is not int or actual_width != contract.n_sae_features:
        raise ClusterFeatureContractError(
            f"{surface} original SAE width does not match the feature contract: "
            f"{actual_width!r} != {contract.n_sae_features}"
        )
    actual_num_buckets = getattr(config, "num_buckets", None)
    if type(actual_num_buckets) is not int or actual_num_buckets != contract.selection.num_buckets:
        raise ClusterFeatureContractError(
            f"{surface} bucket count does not match the feature contract: "
            f"{actual_num_buckets!r} != {contract.selection.num_buckets}"
        )
    actual_vocab = getattr(model, "feature_vocab", None)
    if actual_vocab != contract.checkpoint.feature_vocab:
        raise ClusterFeatureContractError(
            f"{surface} in-memory FeatureVocab does not match pinned checkpoint state"
        )
    assert_consistent = getattr(model, "assert_vocab_consistent", None)
    if not callable(assert_consistent):
        raise ClusterFeatureContractError(f"{surface} does not expose assert_vocab_consistent()")
    try:
        assert_consistent()
    except FeatureVocabError as exc:
        raise ClusterFeatureContractError(
            f"{surface} has inconsistent persistent FeatureVocab buffers: {exc}"
        ) from exc


def assert_loaded_cluster_target(
    contract: ClusterFeatureContract,
    model: Any,
    metadata: Mapping[str, Any],
    *,
    surface: str,
) -> None:
    """Rehash the checkpoint, then assert its actual loaded encoder state."""

    assert_cluster_checkpoint_pin_unchanged(contract)
    assert_cluster_contract_loaded_state(
        contract,
        model,
        metadata,
        surface=surface,
    )


def validate_feature_contract_document(raw: Any) -> dict[str, Any]:
    """Validate the strict serialized sidecar shape and semantic fingerprint."""

    expected = {
        "schema_version",
        "feature_contract_fingerprint",
        "semantic",
        "provenance",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise ClusterFeatureContractError(
            "feature contract sidecar must contain exactly "
            f"{sorted(expected)}; got keys={sorted(keys, key=str)}"
        )
    if raw["schema_version"] != FEATURE_CONTRACT_SCHEMA_VERSION:
        raise ClusterFeatureContractError(
            f"unsupported feature contract schema {raw['schema_version']!r}"
        )
    if not isinstance(raw["semantic"], Mapping) or not isinstance(raw["provenance"], Mapping):
        raise ClusterFeatureContractError(
            "feature contract semantic and provenance fields must be mappings"
        )
    expected_fingerprint = _sha256_json(raw["semantic"])
    if raw["feature_contract_fingerprint"] != expected_fingerprint:
        raise ClusterFeatureContractError(
            "feature contract sidecar fingerprint does not match semantic content"
        )
    # Codex pass-2 residual finding 2 (MEDIUM), THIRD path to the same E4
    # condition (see build_cluster_feature_contract's own "empty_concept_
    # projection_after_masking" comment above for the first two -- direct
    # build, and the resolver's live call into it): a document that reaches
    # this validator can be schema-valid and correctly hashed against its
    # OWN (empty-projection) semantic content while never having gone
    # through the live builder at all -- e.g. a hand-authored
    # ``expected_feature_contract_document``, or an on-disk sidecar written
    # by an older stack build that predates this invariant. Neither the
    # builder's own guard nor the fingerprint check above can catch that:
    # the builder never runs on an already-serialized document, and a
    # self-consistent (hash matches content) empty-projection document is
    # not internally contradictory, just semantically useless downstream.
    # NAMED DEVIATION from the oracle (mirroring the builder's own comment):
    # the oracle's equivalent document validator has no such check and
    # accepts this shape; the stack's now rejects it on every path that can
    # hand a "successful" contract document to a caller, not only the ones
    # that go through a live build.
    concept_source = raw["semantic"].get("concept_source")
    source_to_representative = (
        concept_source.get("source_to_representative")
        if isinstance(concept_source, Mapping)
        else None
    )
    if not source_to_representative:
        concept_source_mode = (
            concept_source.get("mode") if isinstance(concept_source, Mapping) else None
        )
        raise ClusterFeatureContractError(
            "concept source projection is empty after mask filtering: no "
            f"{concept_source_mode!r} source id survives the configured "
            "feature mask, so no representative would ever receive a "
            "projecting source (E4 non-emptiness; see "
            "effective_feature_vocab's own E4 invariant)"
        )
    return {
        "schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "feature_contract_fingerprint": expected_fingerprint,
        "semantic": dict(raw["semantic"]),
        "provenance": dict(raw["provenance"]),
    }


def write_feature_contract_sidecar(
    path: str | Path,
    contract: ClusterFeatureContract,
) -> Path:
    """Atomically write the clustered contract sidecar/commit marker."""

    output = Path(path)
    payload = json.dumps(contract.document(), indent=2).encode("utf-8")
    try:
        install_bytes_once(payload, output, surface="cluster feature contract")
    except ImmutableArtifactError as exc:
        # Provenance paths are intentionally excluded from the semantic
        # fingerprint. Concurrent preflights that resolved equivalent inputs
        # at different locations may therefore render different sidecar bytes;
        # preserve the winner and accept it only after full schema validation
        # proves the same semantic fingerprint.
        try:
            load_feature_contract_sidecar(
                output,
                expected_fingerprint=contract.feature_contract_fingerprint,
            )
        except Exception as validation_exc:
            raise ClusterFeatureContractError(str(exc)) from validation_exc
    return output


def ensure_feature_contract_run_marker(
    run_dir: str | Path,
    contract: ClusterFeatureContract,
    *,
    filename: str = "feature_contract.json",
) -> Path:
    """Commit a clustered run root only after successful preflight.

    A non-empty directory without the marker, or a marker whose full semantic
    document differs, is never repaired in place. The operator must choose a
    fresh run tag so stale per-artifact ownership cannot be legitimized by a
    newly written root marker.
    """

    root = Path(run_dir)
    marker = root / filename
    if root.exists() and not root.is_dir():
        raise ClusterFeatureContractError(f"clustered run root is not a directory: {root}")
    existing_entries = [] if not root.exists() else list(root.iterdir())
    if existing_entries and not marker.is_file():
        # The BPB-Adv preflight commits the exact source snapshot before this
        # root marker.  Permit that one known, matching entry so a successful
        # first run can publish its marker; every other pre-existing artifact
        # remains a stale/torn-root failure.
        source_contract = root / "source_contract.json"
        unexpected = [entry for entry in existing_entries if entry != source_contract]
        if unexpected or not source_contract.is_file():
            raise ClusterFeatureContractError(
                f"existing clustered run directory {root} is non-empty but lacks "
                f"{filename}; choose a fresh run tag"
            )
        try:
            source_payload = json.loads(source_contract.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ClusterFeatureContractError(
                f"existing source contract at {source_contract} is unreadable"
            ) from exc
        if not isinstance(source_payload, Mapping) or (
            source_payload.get("feature_contract_fingerprint")
            != contract.feature_contract_fingerprint
        ):
            raise ClusterFeatureContractError(
                f"existing source contract at {source_contract} does not match "
                "the clustered feature contract"
            )
    if marker.is_file():
        load_feature_contract_sidecar(
            marker,
            expected_fingerprint=contract.feature_contract_fingerprint,
        )
        return marker
    root.mkdir(parents=True, exist_ok=True)
    return write_feature_contract_sidecar(marker, contract)


def load_feature_contract_sidecar(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    input_path = Path(path)
    # Return the validated document without retaining the loader's file digest.
    raw, _digest = _load_json_mapping(input_path, surface="feature contract sidecar")
    parsed = validate_feature_contract_document(raw)
    if expected_fingerprint is not None and (
        parsed["feature_contract_fingerprint"] != expected_fingerprint
    ):
        raise ClusterFeatureContractError(
            "feature contract sidecar does not match the expected fingerprint"
        )
    return parsed


__all__ = [
    "CONCEPT_PROJECTION_VERSION",
    "FEATURE_CONTRACT_SCHEMA_VERSION",
    "FEATURE_DIFFMEAN_PROJECTION_VERSION",
    "GENERIC_ROOFLINE_SIGNAL_VERSION",
    "MAX_SAFE_SELECTED_FEATURE_WIDTH",
    "PACKED_PERCENTILE_SELECTION_VERSION",
    "QC_TARGET_REDUCTION_VERSION",
    "CheckpointPin",
    "CheckpointSelectionContract",
    "ClusterFeatureContract",
    "ClusterFeatureContractError",
    "ClusterFeatureContractRequest",
    "ClusterFeatureTarget",
    "FeatureDirectoryIdentity",
    "FeatureStatisticsIdentity",
    "SaeIdentity",
    "StableFileDigest",
    "assert_cluster_checkpoint_pin_unchanged",
    "assert_cluster_contract_loaded_state",
    "assert_loaded_cluster_target",
    "build_cluster_feature_contract",
    "ensure_feature_contract_run_marker",
    "load_feature_contract_sidecar",
    "resolve_checkpoint_pin",
    "resolve_cluster_feature_contract",
    "resolve_cluster_feature_contract_ddp",
    "resolve_eval_checkpoint_directory",
    "resolve_selection_manifest_path",
    "stat_signature_now",
    "stream_hash_file_stable",
    "validate_feature_contract_document",
    "write_feature_contract_sidecar",
]
