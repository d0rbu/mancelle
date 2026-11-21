"""Bounded clustered inline-steering cache built from validated ``sel/`` rows.

Clustered training performs selection offline, including splice-counterfactual
delta ranking.  Inline steering must therefore consume the same validation
selection overlay instead of re-thresholding raw feature magnitudes.  This
module performs a two-pass provenance join and materializes only the selected
target prefixes and donor segments needed by the periodic metric.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch as t
import torch.distributed as dist

from errors import FeatureClustersError, FeatureVocabError
from nanochat.dataset import list_parquet_files
from steering.conditional.delta import DeltaMode, parse_feature_delta
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    canonical_feature_clusters_descriptor,
)
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, canonical_dict
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.selection_artifact_validation import (
    validate_complete_selection_dir,
)
from steering.conditional.splice_counterfactual import (
    PackedProvenanceInfo,
    validate_packed_provenance,
)
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
)
from steering.overlay_io import load_overlay_shard, read_manifest
from steering.packed_batch_io import SelectionShard, load_selection_shard

CLUSTERED_INLINE_CACHE_SEMANTICS = "delta_selection_provenance_join_v1"


@dataclass(frozen=True)
class ClusteredInlineTarget:
    """One deterministic target and its already-selected conditioning rows."""

    doc_idx: int
    target_segment_ordinal: int
    prefix_segments: tuple[SegmentFeatures, ...]
    positive_feature_ids: tuple[int, ...]
    negative_feature_ids: tuple[int, ...]
    positive_member_groups: tuple[tuple[int, ...], ...]
    donor_doc_idx: int
    donor_segment_ordinal: int
    donor_segment: SegmentFeatures

    @property
    def key(self) -> tuple[int, int]:
        return self.doc_idx, self.target_segment_ordinal


@dataclass(frozen=True)
class ClusteredInlineEvalPayload:
    """Validated clustered payload attached conditionally to ``InlineEvalCache``."""

    source_identity: dict[str, Any]
    targets: tuple[ClusteredInlineTarget, ...]
    digest: str

    def target(self, doc_idx: int, segment_ordinal: int) -> ClusteredInlineTarget:
        key = (doc_idx, segment_ordinal)
        matches = [target for target in self.targets if target.key == key]
        if len(matches) != 1:
            raise ValueError(
                "clustered inline cache expected exactly one target for "
                f"{key}, found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class ClusteredInlineCacheBuild:
    """Complete result used to extend the legacy ``InlineEvalCache``."""

    selected_docs: list[tuple[int, int]]
    selection_metrics: dict[str, int | float]
    doc_texts: dict[int, str]
    payload: ClusteredInlineEvalPayload


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_validation_doc_texts(
    data_dir: str | None,
    *,
    required_stem: str | None = None,
) -> tuple[str, dict[int, str]]:
    parquet_paths = [Path(path) for path in list_parquet_files(data_dir=data_dir)]
    if required_stem is None:
        val_parquet = parquet_paths[-1]
    else:
        matches = [path for path in parquet_paths if path.stem == required_stem]
        if len(matches) != 1:
            raise ValueError(
                "clustered inline steering requires exactly one certified "
                f"validation parquet stem {required_stem!r}; matches="
                f"{[str(path) for path in matches]}"
            )
        val_parquet = matches[0]
    texts = pq.read_table(val_parquet, columns=["text"])["text"].to_pylist()
    return val_parquet.stem, {idx: str(text) for idx, text in enumerate(texts)}


def _aligned_paths(
    packed_info: PackedProvenanceInfo,
    selection_dir: Path,
) -> list[tuple[Path, Path, Path]]:
    provenance_by_name = {
        path.name: path for path in packed_info.provenance_shard_paths
    }
    triples: list[tuple[Path, Path, Path]] = []
    for packed_path in packed_info.packed_shard_paths:
        provenance_path = provenance_by_name.get(packed_path.name)
        selection_path = selection_dir / packed_path.name
        if provenance_path is None or not selection_path.is_file():
            raise ValueError(
                "clustered inline steering requires exact packed/provenance/selection "
                f"shard alignment for {packed_path.name!r}"
            )
        triples.append((packed_path, provenance_path, selection_path))
    return triples


def _load_aligned_selection_provenance(
    *,
    provenance_path: Path,
    selection_path: Path,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
) -> tuple[SelectionShard, t.Tensor, t.Tensor, t.Tensor]:
    selection = load_selection_shard(selection_path)
    provenance = load_overlay_shard(provenance_path)
    try:
        base_idx = provenance[PROVENANCE_BASE_SHARD_IDX_KEY].to(t.int64)
        doc_idx = provenance[PROVENANCE_DOC_IDX_KEY].to(t.int64)
        seg_ordinal = provenance[PROVENANCE_SEG_ORDINAL_KEY].to(t.int64)
    except KeyError as exc:
        raise ValueError(
            f"{provenance_path}: missing packed provenance key {exc.args[0]!r}"
        ) from exc
    lengths = {
        selection.total_segments,
        int(base_idx.numel()),
        int(doc_idx.numel()),
        int(seg_ordinal.numel()),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"{selection_path.name}: selection/provenance segment-count mismatch "
            f"{sorted(lengths)}"
        )
    feature_clusters.assert_selection_shard(
        sel_offsets=selection.sel_offsets,
        sel_fids=selection.sel_fids,
        sel_buckets=selection.sel_buckets,
        sel_pcts=selection.sel_pcts,
        neg_offsets=selection.neg_offsets,
        neg_fids=selection.neg_fids,
        expected_segments=selection.total_segments,
        effective_buckets=effective_buckets,
        surface=f"clustered inline steering {selection_path.name}",
    )
    return selection, base_idx, doc_idx, seg_ordinal


def _selected_targets_and_donors(
    *,
    aligned_paths: list[tuple[Path, Path, Path]],
    val_base_idx: int,
    val_shard_stem: str,
    feature_index: FeatureIndex,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
    n_eval_docs: int,
    seed: int,
    model_max_seq_len: int,
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    int,
    int,
]:
    seen: set[tuple[int, int]] = set()
    positive_by_doc: dict[int, list[int]] = {}
    for _packed_path, provenance_path, selection_path in aligned_paths:
        selection, base_idx, doc_idx, seg_ordinal = (
            _load_aligned_selection_provenance(
                provenance_path=provenance_path,
                selection_path=selection_path,
                feature_clusters=feature_clusters,
                effective_buckets=effective_buckets,
            )
        )
        offsets = selection.sel_offsets.to(t.int64)
        for segment_idx in t.nonzero(
            base_idx == val_base_idx,
            as_tuple=False,
        ).flatten().tolist():
            key = (int(doc_idx[segment_idx]), int(seg_ordinal[segment_idx]))
            if key in seen:
                raise ValueError(
                    "clustered inline steering found duplicate validation packed "
                    f"provenance key {key}"
                )
            seen.add(key)
            if int(offsets[segment_idx + 1]) > int(offsets[segment_idx]):
                positive_by_doc.setdefault(key[0], []).append(key[1])

    if model_max_seq_len < 1:
        raise ValueError(
            "model_max_seq_len must be positive for clustered inline steering, "
            f"got {model_max_seq_len}"
        )

    eligible_positive_by_doc: dict[int, list[int]] = {}
    n_context_filtered = 0
    for source_doc_idx in sorted(positive_by_doc):
        try:
            source_segments = feature_index.lookup(val_shard_stem, source_doc_idx)
        except KeyError as exc:
            raise ValueError(
                "clustered inline steering positive selection points at missing "
                f"derived document {source_doc_idx} in shard {val_shard_stem!r}"
            ) from exc
        for ordinal in sorted(positive_by_doc[source_doc_idx]):
            # Segment zero has no non-empty document prefix and therefore cannot
            # satisfy generate_samples_for_condition's strict prefill contract.
            if ordinal == 0:
                continue
            if ordinal < 0 or ordinal >= len(source_segments):
                raise ValueError(
                    "clustered inline steering positive provenance segment ordinal "
                    f"{ordinal} is out of range for derived document "
                    f"{source_doc_idx} with {len(source_segments)} segments"
                )
            source_segment = source_segments[ordinal]
            if source_segment.segment_idx != ordinal:
                raise ValueError(
                    "clustered inline steering positive provenance ordinal does "
                    "not match the derived FeatureIndex segment_idx: "
                    f"key={(source_doc_idx, ordinal)}, "
                    f"segment_idx={source_segment.segment_idx}"
                )
            if source_segment.token_start is None or source_segment.token_end is None:
                raise ValueError(
                    "clustered inline steering target eligibility requires "
                    "authoritative token boundaries for derived segment "
                    f"{(source_doc_idx, ordinal)}"
                )
            token_start = int(source_segment.token_start)
            token_end = int(source_segment.token_end)
            if token_start <= 0 or token_end <= token_start:
                raise ValueError(
                    "clustered inline steering target has invalid authoritative "
                    f"token bounds for segment {(source_doc_idx, ordinal)}: "
                    f"start={token_start}, end={token_end}"
                )
            if token_end > model_max_seq_len:
                n_context_filtered += 1
                continue
            eligible_positive_by_doc.setdefault(source_doc_idx, []).append(ordinal)

    eligible_docs = sorted(
        doc_idx for doc_idx, ordinals in eligible_positive_by_doc.items() if ordinals
    )
    rng = random.Random(seed)
    rng.shuffle(eligible_docs)
    chosen_docs = eligible_docs[:n_eval_docs]
    selected_docs = [
        (
            doc_idx,
            rng.choice(eligible_positive_by_doc[doc_idx]),
        )
        for doc_idx in chosen_docs
    ]

    positive_candidates = sorted(
        (doc_idx, ordinal)
        for doc_idx, ordinals in positive_by_doc.items()
        for ordinal in ordinals
    )
    donors: dict[tuple[int, int], tuple[int, int]] = {}
    for target in selected_docs:
        donor_candidates = [
            candidate
            for candidate in positive_candidates
            if candidate[0] != target[0]
        ]
        if not donor_candidates:
            raise ValueError(
                "clustered inline steering needs a positive-selected donor from "
                f"a different document for target {target}"
            )
        donors[target] = rng.choice(donor_candidates)
    return selected_docs, donors, len(eligible_docs), n_context_filtered


def _selected_segment(
    *,
    selection: SelectionShard,
    segment_idx: int,
    source_segment: SegmentFeatures,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
) -> tuple[SegmentFeatures, tuple[int, ...], tuple[int, ...]]:
    pos_lo = int(selection.sel_offsets[segment_idx])
    pos_hi = int(selection.sel_offsets[segment_idx + 1])
    neg_lo = int(selection.neg_offsets[segment_idx])
    neg_hi = int(selection.neg_offsets[segment_idx + 1])
    positive_ids = tuple(int(value) for value in selection.sel_fids[pos_lo:pos_hi])
    positive_buckets = tuple(
        int(value) for value in selection.sel_buckets[pos_lo:pos_hi]
    )
    negative_ids = tuple(int(value) for value in selection.neg_fids[neg_lo:neg_hi])

    if len(positive_ids) != len(set(positive_ids)):
        raise FeatureClustersError(
            "clustered inline steering selection contains duplicate positive IDs"
        )
    if len(negative_ids) != len(set(negative_ids)):
        raise FeatureClustersError(
            "clustered inline steering selection contains duplicate negative IDs"
        )
    overlap = set(positive_ids).intersection(negative_ids)
    if overlap:
        raise FeatureClustersError(
            "clustered inline steering selection contains feature ID "
            f"{min(overlap)} as both positive and negative"
        )
    feature_clusters.assert_representatives_only(
        t.tensor(positive_ids, dtype=t.int64),
        surface="clustered inline steering positives",
    )
    feature_clusters.assert_representatives_only(
        t.tensor(negative_ids, dtype=t.int64),
        surface="clustered inline steering negatives",
    )
    feature_ids = t.tensor((*positive_ids, *negative_ids), dtype=t.int64)
    feature_mags = t.tensor(
        (*positive_buckets, *((effective_buckets,) * len(negative_ids))),
        dtype=t.int64,
    )
    return (
        SegmentFeatures(
            segment_idx=source_segment.segment_idx,
            position=source_segment.position,
            feature_ids=feature_ids,
            feature_mags=feature_mags,
            char_start=source_segment.char_start,
            char_end=source_segment.char_end,
            token_start=source_segment.token_start,
            token_end=source_segment.token_end,
        ),
        positive_ids,
        negative_ids,
    )


def _materialize_required_segments(
    *,
    aligned_paths: list[tuple[Path, Path, Path]],
    val_base_idx: int,
    val_shard_stem: str,
    selected_docs: list[tuple[int, int]],
    donors: dict[tuple[int, int], tuple[int, int]],
    feature_index: FeatureIndex,
    feature_clusters: FeatureClusterConsolidation,
    effective_buckets: int,
) -> dict[
    tuple[int, int],
    tuple[SegmentFeatures, tuple[int, ...], tuple[int, ...]],
]:
    required: set[tuple[int, int]] = set(donors.values())
    for doc_idx, target_ordinal in selected_docs:
        required.update((doc_idx, ordinal) for ordinal in range(target_ordinal + 1))

    materialized: dict[
        tuple[int, int],
        tuple[SegmentFeatures, tuple[int, ...], tuple[int, ...]],
    ] = {}
    for _packed_path, provenance_path, selection_path in aligned_paths:
        selection, base_idx, doc_idx, seg_ordinal = (
            _load_aligned_selection_provenance(
                provenance_path=provenance_path,
                selection_path=selection_path,
                feature_clusters=feature_clusters,
                effective_buckets=effective_buckets,
            )
        )
        for segment_idx in t.nonzero(
            base_idx == val_base_idx,
            as_tuple=False,
        ).flatten().tolist():
            key = (int(doc_idx[segment_idx]), int(seg_ordinal[segment_idx]))
            if key not in required:
                continue
            if key in materialized:
                raise ValueError(
                    "clustered inline steering joined required provenance key "
                    f"{key} more than once"
                )
            try:
                source_segments = feature_index.lookup(val_shard_stem, key[0])
            except KeyError as exc:
                raise ValueError(
                    "clustered inline steering provenance points at missing "
                    f"derived document {key[0]} in shard {val_shard_stem!r}"
                ) from exc
            if key[1] >= len(source_segments):
                raise ValueError(
                    "clustered inline steering provenance segment ordinal "
                    f"{key[1]} is out of range for derived document {key[0]} "
                    f"with {len(source_segments)} segments"
                )
            source_segment = source_segments[key[1]]
            if source_segment.segment_idx != key[1]:
                raise ValueError(
                    "clustered inline steering provenance ordinal does not match "
                    f"derived FeatureIndex segment_idx: key={key}, "
                    f"segment_idx={source_segment.segment_idx}"
                )
            if source_segment.token_start is None or source_segment.token_end is None:
                raise ValueError(
                    "clustered inline steering requires authoritative token "
                    f"boundaries for derived segment {key}"
                )
            materialized[key] = _selected_segment(
                selection=selection,
                segment_idx=segment_idx,
                source_segment=source_segment,
                feature_clusters=feature_clusters,
                effective_buckets=effective_buckets,
            )

    missing = sorted(required - set(materialized))
    if missing:
        raise ValueError(
            "clustered inline steering could not join every required packed "
            f"provenance key; missing={missing[:10]}"
        )
    return materialized


def _member_groups(
    feature_clusters: FeatureClusterConsolidation,
    representatives: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    rep_of = feature_clusters.rep_of.to(t.int64)
    allowed = feature_clusters.allowed_universe
    groups: list[tuple[int, ...]] = []
    for representative in representatives:
        members = tuple(
            int(value)
            for value in t.nonzero(
                allowed & (rep_of == representative),
                as_tuple=False,
            ).flatten()
        )
        if not members or members[0] != representative:
            raise FeatureClustersError(
                "clustered inline steering cannot derive the surviving member "
                f"group for representative {representative}"
            )
        groups.append(members)
    return tuple(groups)


def prepare_clustered_inline_eval_cache(
    *,
    feature_dir: Path | str,
    packed_row_dir: Path | str,
    selection_dir: Path | str,
    selection_hash: str,
    feature_index: FeatureIndex,
    data_dir: str | None,
    n_eval_docs: int,
    seed: int,
    model_max_seq_len: int,
    effective_buckets: int,
    feature_mask: FeatureMask | None,
    feature_clusters: FeatureClusterConsolidation,
    feature_vocab: FeatureVocab | None,
    delta_config: dict[str, Any] | None,
    boundary_fingerprint: str | None,
    splice_batch_size: object = None,
    validation_shard_stem: str | None = None,
    remote_derived_feature_fingerprint: str | None = None,
    source_certificate_fingerprint: str | None = None,
) -> ClusteredInlineCacheBuild:
    """Build and validate the clustered steering cache before training starts."""
    feature_dir = Path(feature_dir)
    packed_row_dir = Path(packed_row_dir)
    selection_dir = Path(selection_dir)
    if n_eval_docs < 0:
        raise ValueError(f"n_eval_docs must be non-negative, got {n_eval_docs}")
    if effective_buckets < 1:
        raise ValueError(
            f"effective_buckets must be positive, got {effective_buckets}"
        )
    parsed_delta = parse_feature_delta(delta_config)
    if parsed_delta is None or parsed_delta.mode is not DeltaMode.splice_counterfactual:
        raise FeatureClustersError(
            "clustered inline steering currently requires "
            "feature_delta.mode=splice_counterfactual so the offline validation "
            "selection can be joined through its certified packed provenance"
        )
    if feature_clusters.feature_mask != feature_mask:
        raise FeatureClustersError(
            "clustered inline steering clusters were derived from a different mask"
        )
    expected_vocab = effective_feature_vocab(
        feature_clusters,
        feature_clusters.n_sae_features,
    )
    if expected_vocab != feature_vocab:
        raise FeatureVocabError(
            "clustered inline steering vocabulary differs from the model vocabulary"
        )

    validated_dir = validate_complete_selection_dir(
        packed_row_dir=packed_row_dir,
        selection_hash=selection_hash,
        configured_mask=canonical_dict(feature_mask),
        label="clustered inline steering validation",
        feature_clusters=feature_clusters,
        delta_config=delta_config,
        splice_feature_dir=feature_dir,
        boundary_fingerprint=boundary_fingerprint,
        splice_batch_size=splice_batch_size,
        remote_packed_streaming=remote_derived_feature_fingerprint is not None,
        remote_derived_feature_fingerprint=remote_derived_feature_fingerprint,
    )
    if validated_dir.resolve() != selection_dir.resolve():
        raise ValueError(
            "clustered inline steering selection_dir differs from the canonical "
            f"validated path: configured={selection_dir}, derived={validated_dir}"
        )
    initial_manifest = read_manifest(selection_dir)
    if not isinstance(initial_manifest, dict):
        raise ValueError(
            "clustered inline steering selection manifest disappeared after "
            f"startup validation: {selection_dir}"
        )
    packed_info = validate_packed_provenance(
        packed_row_dir,
        cache_dir=packed_row_dir,
    )
    aligned_paths = _aligned_paths(packed_info, selection_dir)
    val_shard_stem, doc_texts = _load_validation_doc_texts(
        data_dir,
        required_stem=validation_shard_stem,
    )
    matching_base_indices = [
        index
        for index, stem in enumerate(packed_info.base_shard_stems)
        if stem == val_shard_stem
    ]
    if len(matching_base_indices) != 1:
        raise ValueError(
            "clustered inline steering validation parquet must match exactly one "
            "packed provenance base shard: "
            f"stem={val_shard_stem!r}, matches={matching_base_indices}"
        )
    val_base_idx = matching_base_indices[0]
    if val_shard_stem not in feature_index.shard_stems:
        raise ValueError(
            "clustered inline steering validation shard is absent from the "
            f"derived FeatureIndex: {val_shard_stem!r}"
        )

    selected_docs, donors, n_available, n_context_filtered = (
        _selected_targets_and_donors(
            aligned_paths=aligned_paths,
            val_base_idx=val_base_idx,
            val_shard_stem=val_shard_stem,
            feature_index=feature_index,
            feature_clusters=feature_clusters,
            effective_buckets=effective_buckets,
            n_eval_docs=n_eval_docs,
            seed=seed,
            model_max_seq_len=model_max_seq_len,
        )
    )
    materialized = _materialize_required_segments(
        aligned_paths=aligned_paths,
        val_base_idx=val_base_idx,
        val_shard_stem=val_shard_stem,
        selected_docs=selected_docs,
        donors=donors,
        feature_index=feature_index,
        feature_clusters=feature_clusters,
        effective_buckets=effective_buckets,
    )

    targets: list[ClusteredInlineTarget] = []
    for target_key in selected_docs:
        doc_idx, target_ordinal = target_key
        if doc_idx not in doc_texts:
            raise ValueError(
                "clustered inline steering selected a document absent from the "
                f"derived validation parquet: doc_idx={doc_idx}"
            )
        prefix_segments = tuple(
            materialized[(doc_idx, ordinal)][0]
            for ordinal in range(target_ordinal + 1)
        )
        positive_ids = materialized[target_key][1]
        negative_ids = materialized[target_key][2]
        if not positive_ids:
            raise ValueError(
                f"clustered inline steering selected target {target_key} without positives"
            )
        donor_key = donors[target_key]
        targets.append(
            ClusteredInlineTarget(
                doc_idx=doc_idx,
                target_segment_ordinal=target_ordinal,
                prefix_segments=prefix_segments,
                positive_feature_ids=positive_ids,
                negative_feature_ids=negative_ids,
                positive_member_groups=_member_groups(
                    feature_clusters,
                    positive_ids,
                ),
                donor_doc_idx=donor_key[0],
                donor_segment_ordinal=donor_key[1],
                donor_segment=materialized[donor_key][0],
            )
        )

    metadata = initial_manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"clustered inline steering selection manifest is malformed: {selection_dir}"
        )
    integrity = metadata.get(SELECTION_SHARD_INTEGRITY_KEY)
    if not isinstance(integrity, dict):
        raise ValueError(
            "clustered inline steering selection manifest lacks its content aggregate"
        )
    splice_metadata = metadata.get("splice_counterfactual")
    source_identity: dict[str, Any] = {
        "semantics": CLUSTERED_INLINE_CACHE_SEMANTICS,
        "selection_hash": selection_hash,
        "selection_content_aggregate": integrity.get("aggregate_sha256"),
        "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
        "packed_row_fingerprint": packed_info.packed_row_fingerprint,
        "provenance_fingerprint": packed_info.provenance_fingerprint,
        "splice_counterfactual": splice_metadata,
        "feature_mask_canonical": canonical_dict(feature_mask),
        "feature_clusters": canonical_feature_clusters_descriptor(
            feature_clusters.resolved
        ),
        "feature_vocab": (
            None if feature_vocab is None else feature_vocab.metadata_dict()
        ),
        "validation_shard_stem": val_shard_stem,
        "model_max_seq_len": model_max_seq_len,
    }
    if source_certificate_fingerprint is not None:
        source_identity["remote_inline_eval_source"] = source_certificate_fingerprint
    digest_payload = {
        "source_identity": source_identity,
        "targets": [
            {
                "doc_idx": target.doc_idx,
                "target_segment_ordinal": target.target_segment_ordinal,
                "donor_doc_idx": target.donor_doc_idx,
                "donor_segment_ordinal": target.donor_segment_ordinal,
            }
            for target in targets
        ],
    }

    # Bind the two-pass materialization to immutable bytes once more.  A file
    # replaced during either scan must fail instead of yielding a mixed cache.
    validate_complete_selection_dir(
        packed_row_dir=packed_row_dir,
        selection_hash=selection_hash,
        configured_mask=canonical_dict(feature_mask),
        label="clustered inline steering post-scan validation",
        feature_clusters=feature_clusters,
        delta_config=delta_config,
        splice_feature_dir=feature_dir,
        boundary_fingerprint=boundary_fingerprint,
        splice_batch_size=splice_batch_size,
        remote_packed_streaming=remote_derived_feature_fingerprint is not None,
        remote_derived_feature_fingerprint=remote_derived_feature_fingerprint,
    )
    final_manifest = read_manifest(selection_dir)
    if final_manifest != initial_manifest:
        raise ValueError(
            "clustered inline steering selection manifest identity changed "
            "during cache construction"
        )
    final_packed_info = validate_packed_provenance(
        packed_row_dir,
        cache_dir=packed_row_dir,
    )
    if final_packed_info != packed_info:
        raise ValueError(
            "clustered inline steering packed/provenance identity changed during "
            "cache construction"
        )

    return ClusteredInlineCacheBuild(
        selected_docs=selected_docs,
        selection_metrics={
            "steering/n_docs_available": n_available,
            "steering/n_docs_selected": len(selected_docs),
            "steering/n_targets_context_filtered": n_context_filtered,
        },
        doc_texts=doc_texts if selected_docs else {},
        payload=ClusteredInlineEvalPayload(
            source_identity=source_identity,
            targets=tuple(targets),
            digest=_sha256_json(digest_payload),
        ),
    )


def assert_clustered_inline_cache_ddp_consistent(
    payload: ClusteredInlineEvalPayload,
    *,
    world_size: int,
) -> None:
    """All-gather the startup cache digest and reject rank disagreement."""
    if world_size <= 1:
        return
    if not dist.is_initialized():
        raise RuntimeError(
            "clustered inline steering requested multi-rank cache validation "
            "before torch.distributed initialization"
        )
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, payload.digest)
    if any(digest != payload.digest for digest in gathered):
        raise RuntimeError(
            "clustered inline steering cache digest disagreement across ranks: "
            f"{gathered}"
        )
