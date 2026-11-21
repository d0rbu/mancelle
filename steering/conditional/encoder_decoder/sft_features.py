"""Lazy SFT feature lookup keyed by original SFT mixture index."""

from __future__ import annotations

import json
import weakref
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch as t
from loguru import logger

from errors import DataLoadError
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import (
    TensorKey,
    glob_features,
    load_features,
    read_shard_header,
    shard_stem_from_path,
)
from steering.parquet_io import detect_source_bos_char_len_from_data
from steering.segmentation import SegmentMode, segment_mode_from_metadata
from steering.sft.mixture_fingerprint import (
    MIXTURE_FINGERPRINT_KEY,
    SFT_MIXTURE_FINGERPRINT_METADATA_KEY,
    MixtureFingerprint,
    _uniform_sft_mixture_fingerprint_metadata_json,
    format_mixture_fingerprint_mismatch,
    mixture_fingerprint_json,
    normalize_mixture_fingerprint,
    parse_mixture_fingerprint_json,
)
from steering.sft_schema import SFTConversionMeta, SFTSplit
from steering.stats_io import find_statistics_path, load_statistics_metadata

_SOURCE_BOS_LOAD_KEYS: set[str] = {
    TensorKey.DOC_IDS,
    TensorKey.SEGMENT_CHAR_ENDS,
    TensorKey.DOC_TOKEN_COUNTS,
}


def assert_sft_source_bos_char_len_matches(
    feature_dir: str,
    parquet_dir: str,
    *,
    configured: int,
    source: str,
) -> int:
    """Fail unless an SFT artifact family agrees with its configured BOS origin."""
    feature_shards = glob_features(feature_dir)
    if not feature_shards:
        raise DataLoadError(
            f"Cannot verify conditioned SFT {source} source BOS character length: "
            f"no feature shard files found in {feature_dir!r}"
        )
    feature_shard = feature_shards[0]
    parquet_path = Path(parquet_dir) / f"{shard_stem_from_path(feature_shard)}.parquet"
    try:
        detected = detect_source_bos_char_len_from_data(
            load_features(feature_shard, keys=_SOURCE_BOS_LOAD_KEYS),
            parquet_path,
        )
    except Exception as exc:  # noqa: BLE001 - preserve cause under artifact context
        raise DataLoadError(
            f"Could not verify conditioned SFT {source} source BOS character length "
            f"from feature shard {feature_shard} and parquet {parquet_path}: {exc}"
        ) from exc

    expected = int(configured)
    if detected != expected:
        raise DataLoadError(
            f"Conditioned SFT {source} feature_source_bos_char_len mismatch: "
            f"detected={detected} from feature shard {feature_shard} and parquet "
            f"{parquet_path}, configured={expected}. Set "
            "feature_source_bos_char_len to the detected artifact value, or use "
            "features extracted with the intended source tokenizer."
        )
    logger.info(
        f"Conditioned SFT {source} source BOS character length verified: "
        f"detected={detected}, configured={expected}"
    )
    return detected


@dataclass
class SFTFeatureLoadReport:
    """Optional details for one :func:`load_sft_features` call.

    The report is deliberately an output object rather than part of the return
    value: the feature store mapping and its cache identity remain unchanged for
    every existing caller.  The conditioned-SFT startup tracer uses this to
    distinguish the real index construction from a weak-cache reuse without
    performing a second directory or parquet scan.
    """

    cache_hit: bool | None = None
    feature_shards: int | None = None
    parquet_shards: int | None = None
    documents: int | None = None


class _IdScanAnomaly(Exception):
    """Internal signal: the vectorized scan met input it must not vouch for.

    Never propagated to callers — ``scan_queried_feature_id_bitmap`` catches
    it (and every other Exception) and re-runs the reference walk, whose
    outcome is authoritative.
    """


# Integer dtypes the vectorized scan widens to int64 without changing the
# reference walk's dict-key semantics. Anything else (float, bool, complex,
# multi-dim) is handed to the reference walk, which defines their behavior.
_ID_SCAN_INT_DTYPES = (t.int8, t.int16, t.int32, t.int64, t.uint8, t.uint16)


def _require_id_scan_canonical(name: str, tensor: t.Tensor, stem: str) -> None:
    if tensor.dim() != 1 or tensor.dtype not in _ID_SCAN_INT_DTYPES:
        raise _IdScanAnomaly(
            f"shard {stem!r} tensor {name!r} is not a 1-D integer tensor "
            f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
        )


class SFTFeatureStore(Mapping[int, list[SegmentFeatures]]):
    """Lazy mapping from SFT mixture index to pre-extracted segment features."""

    def __init__(
        self,
        feature_dir: str,
        locations: dict[int, tuple[str, int]],
        *,
        segment_mode: SegmentMode,
        parquet_dir: str | None = None,
        parquet_original_indices: frozenset[int],
        conversion_mixture: str | None = None,
        conversion_split: SFTSplit | None = None,
        conversion_total_conversations: int | None = None,
        parquet_mixture_fingerprint: MixtureFingerprint | None = None,
        feature_mixture_fingerprint: MixtureFingerprint | None = None,
        feature_shard_count: int | None = None,
        parquet_shard_count: int | None = None,
    ) -> None:
        # Issue #1504: the background next-shard load hides the steady-state
        # boundary crossing, and lazy per-document materialization keeps the load
        # itself short enough for it to win that race.
        self._feature_index = FeatureIndex(feature_dir, prefetch=True, lazy_docs=True)
        self._locations = locations
        self.segment_mode = segment_mode
        self.feature_shard_count = feature_shard_count
        self.parquet_shard_count = parquet_shard_count
        self._parquet_dir = Path(parquet_dir) if parquet_dir is not None else None
        # The reference document set, read from the parquet's own
        # ``original_index`` column independently of the feature shards.
        self.parquet_original_indices = parquet_original_indices
        self.conversion_mixture = conversion_mixture
        self.conversion_split = conversion_split
        self.conversion_total_conversations = conversion_total_conversations
        self.parquet_mixture_fingerprint = parquet_mixture_fingerprint
        self.feature_mixture_fingerprint = feature_mixture_fingerprint
        self.mixture_fingerprint = parquet_mixture_fingerprint

    def __getitem__(self, key: int) -> list[SegmentFeatures]:
        shard_stem, local_doc_id = self._locations[key]
        # lookup() raises KeyError (naming shard+doc) if the row is absent — a
        # corrupt/stale feature index. We let that propagate rather than masking
        # it as None (issue #521). `self._locations[key]` above already raises a
        # plain KeyError(key) for a key outside the mapping (Mapping protocol).
        return self._feature_index.lookup(shard_stem, local_doc_id)

    def __iter__(self) -> Iterator[int]:
        return iter(self._locations)

    def __len__(self) -> int:
        return len(self._locations)

    def __contains__(self, key: object) -> bool:
        return key in self._locations

    def load_texts(self, original_indices: list[int]) -> dict[int, str]:
        """Read only parquet shards containing the requested original indices."""
        if self._parquet_dir is None:
            raise DataLoadError(
                "SFTFeatureStore has no parquet_dir, so the independent content "
                "alignment probe cannot run"
            )
        by_stem: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for original_index in original_indices:
            try:
                stem, local_doc_id = self._locations[original_index]
            except KeyError as exc:
                raise DataLoadError(
                    "SFT content alignment probe requested missing "
                    f"original_index={original_index}"
                ) from exc
            by_stem[stem].append((original_index, local_doc_id))

        texts: dict[int, str] = {}
        for stem, requested_rows in by_stem.items():
            parquet_path = self._parquet_dir / f"{stem}.parquet"
            schema = pq.read_schema(parquet_path)
            if "text" not in schema.names:
                raise DataLoadError(
                    f"SFT content alignment probe cannot read {parquet_path}: "
                    "missing 'text' column"
                )
            text_values = pq.read_table(parquet_path, columns=["text"]).column(
                "text"
            )
            for original_index, local_doc_id in requested_rows:
                if local_doc_id < 0 or local_doc_id >= len(text_values):
                    raise DataLoadError(
                        f"SFT content alignment probe local row {local_doc_id} is "
                        f"outside {parquet_path}'s {len(text_values)} rows"
                    )
                value = text_values[local_doc_id].as_py()
                if not isinstance(value, str):
                    raise DataLoadError(
                        f"SFT content alignment probe expected string text at "
                        f"{parquet_path} row {local_doc_id}, got {value!r}"
                    )
                texts[original_index] = value
        return texts

    def scan_queried_feature_id_bitmap(
        self,
        dataset_indices: Sequence[int],
        *,
        n_sae_features: int,
        source: str,
    ) -> t.Tensor:
        """ID-only vectorized union of raw pre-selection ids (issue #1495).

        Byte-identical to :func:`scan_queried_feature_ids_reference` — bitmap
        and exception — for every input whose corruption (if any) manifests in
        the fields this scan reads: on any detected anomaly or raised
        exception it logs a WARNING and re-runs the reference walk, returning
        or raising exactly what the reference does. Corruption confined to
        fields only the FULL loader reads may scan cleanly here and fail on
        that shard's first generator access instead — the accepted deviation
        documented in docs/pipelines/encoder-decoder-training.md.
        """
        try:
            return self._vectorized_id_scan(
                dataset_indices, n_sae_features=n_sae_features, source=source
            )
        except Exception as exc:  # noqa: BLE001 - reference walk is the arbiter
            try:
                logger.warning(
                    f"conditioned SFT {source} ID-only percentile scan fell "
                    f"back to the reference document walk: "
                    f"{type(exc).__name__}: {exc}"
                )
            except Exception:  # noqa: BLE001, S110 - best-effort telemetry
                # The warning (including str(exc) itself) must never outrank
                # the reference outcome.
                pass
        return scan_queried_feature_ids_reference(
            self, dataset_indices, n_sae_features=n_sae_features, source=source
        )

    def _vectorized_id_scan(
        self,
        dataset_indices: Sequence[int],
        *,
        n_sae_features: int,
        source: str,
    ) -> t.Tensor:
        by_stem: dict[str, list[int]] = defaultdict(list)
        for index in dataset_indices:
            location = self._locations.get(index)
            if location is None:
                raise _IdScanAnomaly(f"dataset index {index} has no feature location")
            stem, local_doc_id = location
            if not isinstance(local_doc_id, int):
                # Non-int rows change dict-key semantics; the reference decides.
                raise _IdScanAnomaly(
                    f"dataset index {index} maps to non-integer local document row {local_doc_id!r}"
                )
            by_stem[stem].append(local_doc_id)

        seen = t.zeros(n_sae_features, dtype=t.bool)
        documents = 0
        segments_selected = 0
        compressed_bytes = 0
        for stem in sorted(by_stem):
            profile: dict[str, float | int] = {}
            arrays = self._feature_index.load_id_scan_arrays(stem, profile=profile)
            compressed_bytes += int(profile.get("compressed_bytes", 0))
            offsets = arrays["offsets"]
            feature_ids = arrays["feature_ids"]
            # feature_ids need no canonicality check: both paths apply the same
            # elementwise ``.long()`` before bounds-checking, so any dtype the
            # reference tolerates lands on the identical id set here. The
            # offsets check IS load-bearing: the ``segment_mask.all()``
            # shortcut below skips ``repeat_interleave``, which is otherwise
            # the only operation that rejects non-integer offsets.
            for name in ("offsets", "doc_ids"):
                _require_id_scan_canonical(name, arrays[name], stem)
            # Widened before use: torch.isin does not support uint16 promotion.
            doc_ids = arrays["doc_ids"].to(t.int64)
            # The reference's split_with_sizes requires the segment sizes to
            # sum EXACTLY to len(feature_ids); with monotonicity already
            # validated, that is offsets[0] == 0 and offsets[-1] == len.
            csr_start, csr_end = int(offsets[0].item()), int(offsets[-1].item())
            if csr_start != 0 or csr_end != len(feature_ids):
                raise _IdScanAnomaly(
                    f"shard {stem!r} CSR span [{csr_start}, {csr_end}] does not cover "
                    f"its {len(feature_ids)} feature ids"
                )
            requested = t.tensor(sorted(set(by_stem[stem])), dtype=t.int64)
            present = t.isin(requested, doc_ids)
            if not bool(present.all()):
                absent = requested[~present]
                raise _IdScanAnomaly(
                    f"shard {stem!r} is missing document rows {absent[:10].tolist()}"
                )
            documents += requested.numel()
            segment_mask = t.isin(doc_ids, requested)
            if bool(segment_mask.all()):
                selected = feature_ids
            else:
                counts = offsets[1:] - offsets[:-1]
                selected = feature_ids[segment_mask.repeat_interleave(counts)]
            segments_selected += int(segment_mask.sum().item())
            # Non-empty by construction: every requested doc is present and no
            # validated segment is empty. ``.long()`` before min/max — those
            # reductions are unimplemented for uint16, and the reference makes
            # the identical elementwise int64 conversion.
            selected_long = selected.long()
            min_id = int(selected_long.min().item())
            max_id = int(selected_long.max().item())
            if min_id < 0 or max_id >= n_sae_features:
                raise _IdScanAnomaly(
                    f"shard {stem!r} carries feature ids outside "
                    f"[0, {n_sae_features}) (min={min_id}, max={max_id})"
                )
            seen[selected_long] = True
        logger.info(
            f"conditioned SFT {source} ID-only percentile scan: "
            f"{len(by_stem)} shards, {documents} documents, "
            f"{segments_selected} segments, {compressed_bytes} compressed bytes"
        )
        return seen


def scan_queried_feature_ids_reference(
    sft_features: Mapping[int, Any],
    dataset_indices: Sequence[int],
    *,
    n_sae_features: int,
    source: str,
) -> t.Tensor:
    """``bool[n_sae_features]`` — every RAW id the percentile lookups will see.

    Walks the covered documents once and unions their pre-selection segment
    feature ids.  This is the only authoritative answer to "which ids will be
    queried": the FP lookup runs on ``seg.feature_ids`` verbatim, before
    feature-mask filtering and before cluster consolidation, so neither the
    allowed universe nor the cluster representatives bound it (in the l12 pilot
    the whitelist allows 16,254 ids and the representatives are 2,074, while
    the base corpus activates 65,471 features).

    It costs one pass over the feature index — the same reads the first epoch
    performs — and it is paid ONCE, up front, so the operator learns the
    complete picture before the run rather than dying on whichever id happened
    to be looked up first.
    """
    seen = t.zeros(n_sae_features, dtype=t.bool)
    for index in dataset_indices:
        for seg in sft_features[index]:
            fids = seg.feature_ids
            if fids.numel() == 0:
                continue
            fids_long = fids.reshape(-1).long()
            if bool((fids_long < 0).any()) or bool((fids_long >= n_sae_features).any()):
                bad = fids_long[(fids_long < 0) | (fids_long >= n_sae_features)]
                raise DataLoadError(
                    f"conditioned SFT {source} document {index} carries feature "
                    f"id(s) {bad[:10].tolist()} outside [0, {n_sae_features}); "
                    "the feature shards were extracted for a different SAE width "
                    "than the checkpoint's encoder."
                )
            seen[fids_long] = True
    return seen


def scan_required_feature_id_bitmap(
    sft_features: Mapping[int, Any],
    dataset_indices: Sequence[int],
    *,
    n_sae_features: int,
    source: str,
) -> t.Tensor:
    """Route a real store to the ID-only scan, anything else to the reference.

    Test doubles are plain mappings without shard-level structure; they take
    the reference walk — which is also the semantic definition the fast path
    is pinned against (equality + error-parity tests in
    tests/steering/test_sft_feature_id_scan.py).
    """
    if isinstance(sft_features, SFTFeatureStore):
        return sft_features.scan_queried_feature_id_bitmap(
            dataset_indices, n_sae_features=n_sae_features, source=source
        )
    return scan_queried_feature_ids_reference(
        sft_features, dataset_indices, n_sae_features=n_sae_features, source=source
    )


_SFT_FEATURE_CACHE = weakref.WeakValueDictionary()


def load_sft_features(
    feature_dir: str,
    parquet_dir: str,
    *,
    report: SFTFeatureLoadReport | None = None,
) -> SFTFeatureStore:
    """Build a lazy feature mapping keyed by original SFT mixture index.

    Each feature shard corresponds to a full-sequence SFT parquet shard.  The
    parquet's ``original_index`` column maps each row back to the original
    TaskMixture index; the returned mapping uses those indices as keys and
    loads feature shard tensors only on lookup.
    """
    if report is not None:
        # A caller may reuse an observer object across attempts. Never leave a
        # successful prior attempt looking like the current call after a later
        # validation error.
        report.cache_hit = None
        report.feature_shards = None
        report.parquet_shards = None
        report.documents = None
    parquet_path = Path(parquet_dir)
    all_files = glob_features(feature_dir)
    if not all_files:
        raise DataLoadError(f"No feature shard files found in {feature_dir}")

    # A wrong or unmounted ``sft_parquet_dir`` is a one-line path fix, not a
    # multi-hour regeneration. Diagnose it BEFORE reading feature headers or
    # the symmetric stem set-difference below, whose "orphan feature shards"
    # wording would send the operator off to re-run conversion (review
    # finding, PR #1311).
    if not parquet_path.is_dir():
        raise DataLoadError(
            f"SFT parquet directory {parquet_dir} does not exist (the feature "
            f"directory {feature_dir} has {len(all_files)} shards). This is a "
            "path problem, not a stale artifact: check sft_parquet_dir / the "
            "mount before regenerating anything."
        )

    metadata_by_path = [
        (path, read_shard_header(path)["__metadata__"])
        for path in all_files
    ]

    conversion_mixture: str | None = None
    conversion_split: SFTSplit | None = None
    conversion_total_conversations: int | None = None
    parquet_mixture_fingerprint: MixtureFingerprint | None = None
    conversion_meta_path = parquet_path / "conversion_meta.json"
    if conversion_meta_path.exists():
        conversion_meta = SFTConversionMeta.parse(json.loads(conversion_meta_path.read_text()))
        if conversion_meta.format != "full_nanochat_text":
            raise DataLoadError(
                "Conditioned SFT feature loading requires format="
                f"'full_nanochat_text', got {conversion_meta.format!r} at "
                f"{conversion_meta_path}. Structured rollout artifacts are not "
                "legacy feature-extraction inputs."
            )
        conversion_mixture = conversion_meta.mixture
        conversion_split = conversion_meta.split
        conversion_total_conversations = conversion_meta.total_conversations
        # ``SFTConversionMeta`` types both shard counts as required ints, so
        # this is the ONE reachable incomplete-conversion guard: every
        # downstream consumer inherits it (review finding 13, PR #1311).
        if conversion_meta.completed_shards != conversion_meta.total_shards:
            raise DataLoadError(
                f"SFT conversion at {parquet_dir} is incomplete: "
                f"completed_shards={conversion_meta.completed_shards}, "
                f"total_shards={conversion_meta.total_shards}. Finish conversion "
                "before extracting features or training."
            )
        parquet_fingerprint_raw = conversion_meta.raw.get(MIXTURE_FINGERPRINT_KEY)
        if parquet_fingerprint_raw is not None:
            try:
                parquet_mixture_fingerprint = normalize_mixture_fingerprint(
                    parquet_fingerprint_raw,
                    source=f"{conversion_meta_path} mixture_fingerprint",
                )
            except ValueError as exc:
                raise DataLoadError(str(exc)) from exc

    try:
        feature_fingerprint_json = _uniform_sft_mixture_fingerprint_metadata_json(
            metadata_by_path,
            operation="load_sft_features",
        )
        feature_mixture_fingerprint = (
            parse_mixture_fingerprint_json(
                feature_fingerprint_json,
                source="uniform feature shard SFT mixture fingerprint",
            )
            if feature_fingerprint_json is not None
            else None
        )
    except ValueError as exc:
        raise DataLoadError(str(exc)) from exc

    statistics_mixture_fingerprint: MixtureFingerprint | None = None
    stats_path = find_statistics_path(feature_dir)
    statistics_metadata: dict[str, str] = {}
    if stats_path is not None:
        statistics_metadata = load_statistics_metadata(stats_path)
        statistics_fingerprint_json = statistics_metadata.get(
            SFT_MIXTURE_FINGERPRINT_METADATA_KEY
        )
        if statistics_fingerprint_json is not None:
            try:
                statistics_mixture_fingerprint = parse_mixture_fingerprint_json(
                    statistics_fingerprint_json,
                    source=(
                        f"{stats_path} metadata "
                        f"{SFT_MIXTURE_FINGERPRINT_METADATA_KEY}"
                    ),
                )
            except ValueError as exc:
                raise DataLoadError(str(exc)) from exc

    if statistics_mixture_fingerprint is not None and (
        feature_mixture_fingerprint is None
        or statistics_mixture_fingerprint != feature_mixture_fingerprint
    ):
        if feature_mixture_fingerprint is None:
            detail = (
                "feature shard headers carry no SFT mixture fingerprint, while "
                "feature statistics carry one"
            )
        else:
            detail = format_mixture_fingerprint_mismatch(
                "feature_shard_fingerprint",
                feature_mixture_fingerprint,
                "feature_statistics_fingerprint",
                statistics_mixture_fingerprint,
            )
        raise DataLoadError(
            "conditioned SFT feature shard-vs-statistics mixture fingerprint "
            f"mismatch: {detail}. Re-extract the complete feature directory and "
            "recompute train statistics from the same conversion."
        )

    if (parquet_mixture_fingerprint is None) != (
        feature_mixture_fingerprint is None
    ):
        raise DataLoadError(
            "conditioned SFT mixture fingerprint present on exactly one side "
            "(half-migrated state is not legacy): "
            f"parquet_fingerprint={parquet_mixture_fingerprint!r}, "
            f"feature_shard_fingerprint={feature_mixture_fingerprint!r}. "
            "Regenerate the SFT parquet and re-extract features as one matched "
            "artifact family."
        )
    if (
        parquet_mixture_fingerprint is not None
        and feature_mixture_fingerprint is not None
        and parquet_mixture_fingerprint != feature_mixture_fingerprint
    ):
        raise DataLoadError(
            "conditioned SFT parquet-vs-feature mixture fingerprint mismatch: "
            + format_mixture_fingerprint_mismatch(
                "parquet_fingerprint",
                parquet_mixture_fingerprint,
                "feature_shard_fingerprint",
                feature_mixture_fingerprint,
            )
        )

    shard_modes: dict[SegmentMode, list[str]] = defaultdict(list)
    for feat_path, feature_metadata in metadata_by_path:
        try:
            shard_mode = segment_mode_from_metadata(
                feature_metadata,
                source=f"Feature shard {feat_path}",
            )
        except ValueError as exc:
            raise DataLoadError(str(exc)) from exc
        shard_modes[shard_mode].append(str(feat_path))

    if len(shard_modes) != 1:
        details = ", ".join(
            f"{mode.value}={paths}"
            for mode, paths in sorted(
                shard_modes.items(), key=lambda item: item[0].value
            )
        )
        raise DataLoadError(
            "conditioned SFT segment_mode differs across feature shards "
            f"(half-migrated state is not legacy): {details}. Re-extract the "
            "complete feature directory in one segmentation mode."
        )
    segment_mode = next(iter(shard_modes))
    if stats_path is not None:
        try:
            statistics_mode = segment_mode_from_metadata(
                statistics_metadata,
                source=f"Feature directory {feature_dir} statistics metadata",
            )
        except ValueError as exc:
            raise DataLoadError(str(exc)) from exc
        if statistics_mode is not segment_mode:
            raise DataLoadError(
                f"Feature directory {feature_dir} has segment_mode disagreement: "
                f"feature shards record {segment_mode.value!r}, but statistics record "
                f"{statistics_mode.value!r}. Recompute statistics from the same "
                "segmentation mode as the feature shards."
            )

    parquet_fingerprint_cache = (
        mixture_fingerprint_json(parquet_mixture_fingerprint)
        if parquet_mixture_fingerprint is not None
        else None
    )
    feature_fingerprint_cache = (
        mixture_fingerprint_json(feature_mixture_fingerprint)
        if feature_mixture_fingerprint is not None
        else None
    )
    cache_key = (
        str(Path(feature_dir).resolve()),
        str(parquet_path.resolve()),
        parquet_fingerprint_cache,
        feature_fingerprint_cache,
        conversion_total_conversations,
        segment_mode,
    )
    cached = _SFT_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        if report is not None:
            report.cache_hit = True
            report.feature_shards = cached.feature_shard_count
            report.parquet_shards = cached.parquet_shard_count
            report.documents = len(cached)
        return cached

    parquet_paths = sorted(parquet_path.glob("*.parquet"))
    if not parquet_paths:
        raise DataLoadError(
            f"SFT parquet directory {parquet_dir} contains no *.parquet shards, "
            f"while {len(all_files)} feature shards were found in "
            f"{feature_dir}. This is a path problem, not a stale artifact: "
            "check sft_parquet_dir / the mount first. Only if the path is "
            "correct does this mean the conversion output was lost and must be "
            "regenerated with uv run python -m "
            "scripts.preprocess.convert_sft_to_parquet mixture=<name>."
        )
    feature_stems = {shard_stem_from_path(path) for path in all_files}
    parquet_stems = {path.stem for path in parquet_paths}
    missing_feature_stems = sorted(parquet_stems - feature_stems)
    orphan_feature_stems = sorted(feature_stems - parquet_stems)
    if missing_feature_stems or orphan_feature_stems:
        raise DataLoadError(
            "Feature directory and SFT parquet directory have diverged: "
            f"missing feature shards for parquet={missing_feature_stems[:10]}, "
            f"orphan feature shards={orphan_feature_stems[:10]}. Regenerate the "
            "matched parquet and feature artifact family."
        )

    # The parquet's own ``original_index`` column is the reference document
    # set. It is read straight from the parquet shards so that the feature
    # index built below is validated AGAINST it rather than defining it — the
    # coverage comparison downstream would otherwise be tautological.
    parquet_indices_by_stem: dict[str, list[int]] = {}
    parquet_original_indices: dict[int, str] = {}
    for pq_file in parquet_paths:
        schema = pq.read_schema(pq_file)
        if "original_index" not in schema.names:
            raise DataLoadError(
                f"Parquet {pq_file} is not a full_nanochat SFT parquet: missing "
                "'original_index' column. Regenerate it with uv run python -m "
                "scripts.preprocess.convert_sft_to_parquet mixture=<name>."
            )
        shard_indices = [
            int(value)
            for value in pq.read_table(pq_file, columns=["original_index"])
            .column("original_index")
            .to_pylist()
        ]
        parquet_indices_by_stem[pq_file.stem] = shard_indices
        for mixture_idx in shard_indices:
            if mixture_idx in parquet_original_indices:
                raise ValueError(
                    f"Duplicate original_index={mixture_idx} across SFT parquet shards; "
                    "cannot build a one-to-one feature index."
                )
            parquet_original_indices[mixture_idx] = pq_file.stem

    locations: dict[int, tuple[str, int]] = {}
    feature_metadata_by_path = dict(metadata_by_path)
    for feat_path in all_files:
        stem = shard_stem_from_path(feat_path)
        pq_path = parquet_path / f"{stem}.parquet"
        orig_indices = parquet_indices_by_stem[stem]
        feature_n_docs_raw = feature_metadata_by_path[feat_path].get("n_docs")
        if feature_n_docs_raw is None:
            raise DataLoadError(
                f"Feature shard {feat_path} is missing required n_docs metadata; "
                "the parquet↔feature document correspondence cannot be verified. "
                "Re-extract the complete feature directory."
            )
        try:
            feature_n_docs = int(feature_n_docs_raw)
        except (TypeError, ValueError) as exc:
            raise DataLoadError(
                f"Feature shard {feat_path} has malformed n_docs metadata: "
                f"{feature_n_docs_raw!r}"
            ) from exc
        if feature_n_docs != len(orig_indices):
            raise DataLoadError(
                f"Feature shard {feat_path} declares n_docs={feature_n_docs}, "
                f"but parquet shard {pq_path} contains {len(orig_indices)} "
                "original_index rows. Regenerate the matched feature/parquet "
                "artifact family."
            )
        for local_doc_id, mixture_idx in enumerate(orig_indices):
            locations[mixture_idx] = (stem, local_doc_id)

    store = SFTFeatureStore(
        feature_dir,
        locations,
        segment_mode=segment_mode,
        parquet_dir=parquet_dir,
        parquet_original_indices=frozenset(parquet_original_indices),
        conversion_mixture=conversion_mixture,
        conversion_split=conversion_split,
        conversion_total_conversations=conversion_total_conversations,
        parquet_mixture_fingerprint=parquet_mixture_fingerprint,
        feature_mixture_fingerprint=feature_mixture_fingerprint,
        feature_shard_count=len(all_files),
        parquet_shard_count=len(parquet_paths),
    )
    _SFT_FEATURE_CACHE[cache_key] = store
    if report is not None:
        report.cache_hit = False
        report.feature_shards = len(all_files)
        report.parquet_shards = len(parquet_paths)
        report.documents = len(store)
    logger.info(
        f"Indexed SFT features: {len(store)} documents from "
        f"{len(all_files)} shards in {feature_dir}"
    )
    return store
