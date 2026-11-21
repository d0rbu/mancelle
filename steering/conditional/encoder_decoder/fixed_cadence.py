"""Measure fixed-window cadence from authenticated packed TRAIN inputs.

Like turn_cadence, midpoint even-spacing bounds the measured population to 4096
rows; it does not guarantee the full corpus's rounded mean. Rows follow sorted
packed shard filenames, then physical row index, once (not loader shuffle or
rank order). Every retained window occurrence has equal weight. Inventory
hashing/validation is not bounded by the sample size. A scalar RustBPE cadence
is an approximation, not a causal Gemma boundary scheduler.

Native manifests bind current source bytes and tokenizer identity. Historical
checkpoints bind their recorded paths,
packing configuration and loader shard names; this cannot prove that a file was
never replaced before measurement. The receipt preserves that limitation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch as t
import zstandard

from nanochat.dataset import DatasetSplit, list_parquet_files
from scripts.preprocess.precompute_packed_rows import _row_to_csr
from scripts.preprocess.precompute_token_boundaries import compute_doc_token_boundaries_from_tokens
from scripts.train.conditional_encdec_train import _packed_row_train_dir
from steering._common.file_inventory import file_inventory, file_signature
from steering.conditional.delta import DeltaMode, parse_feature_delta
from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    truncate_to_boundary_coverage,
)
from steering.conditional.splice_provenance import SpliceProvenanceIndex
from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    CONTENT_BOUND_MANIFEST_VERSION,
    OverlayKind,
    compute_base_shard_content_hash,
    discover_overlay,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_shard_path,
    read_manifest,
    validate_token_boundary_overlay_shard,
)
from steering.packed_batch_io import PackedRowShard, load_packed_row_shard
from steering.parquet_io import detect_source_bos_char_len_from_data, read_parquet_texts
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    compute_tokenizer_checksum,
    segment_mode_from_metadata,
)
from steering.sft.segment_mode_identity import (
    FIXED_CADENCE_MAX_ROWS,
    FIXED_CADENCE_SOURCE_WIDTH,
    FIXED_CADENCE_VERSION,
    resolve_checkpoint_sft_segment_mode,
)
from steering.stats_io import (
    find_statistics_path,
    load_percentile_checksum,
    load_tokenizer_checksum,
)


class FixedCadenceArtifactError(ValueError):
    """Measurement requires authenticated feature/parquet/packed artifacts."""


class WindowExclusion(StrEnum):
    CLIPPED = "clipped_windows"
    INCOMPLETE_TAIL = "incomplete_source_tails"
    AMBIGUOUS_LENGTH = "ambiguous_source_lengths"
    NONSTANDARD_LENGTH = "oversize_source_windows"
    PREDICTION_ONLY = "prediction_only_windows"


@dataclass(frozen=True)
class FixedCadenceMeasurement:
    cadence: PositiveInt
    windows: PositiveInt
    rows: PositiveInt
    documents: PositiveInt
    sampled_rows: PositiveInt
    sampled_documents: PositiveInt
    document_occurrences: PositiveInt
    mean: Fraction
    minimum: PositiveInt
    maximum: PositiveInt
    quantiles: dict[str, int]
    exclusions: dict[str, int]
    provenance: dict[str, Any]
    artifact_signatures: tuple[tuple[Path, tuple[int, int, int, int, int]], ...]

    def assert_artifacts_current(self) -> None:
        """Recheck the measured snapshot immediately before receipt publication."""
        _require(all(file_signature(path) == signature for path, signature in self.artifact_signatures),
                 "artifacts changed during measurement/publication")

    def metadata_dict(self) -> dict[str, Any]:
        metadata = {
            "cadence": int(self.cadence), "windows": int(self.windows),
            "rows": int(self.rows), "documents": int(self.documents),
            "sampled_rows": int(self.sampled_rows), "sampled_documents": int(self.sampled_documents),
            "document_occurrences": int(self.document_occurrences),
            "mean": {"numerator": self.mean.numerator, "denominator": self.mean.denominator},
            "minimum": int(self.minimum), "maximum": int(self.maximum),
            "quantiles": self.quantiles, "exclusions": self.exclusions,
        }
        return metadata


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedCadenceArtifactError(
            f"fixed cadence artifact requirement: {message}. Restore the matching "
            "training source, paired parquets, token-boundary overlays and native packed provenance; "
            "no inferred cadence fallback is available"
        )


def _positive(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer, got {value!r}")
    return value


def evenly_spaced_rows(total: int, max_rows: int) -> tuple[int, ...]:
    """Midpoints of equal intervals in the canonical physical-row traversal."""
    _positive(total, "total rows")
    _positive(max_rows, "max_rows")
    _require(max_rows <= FIXED_CADENCE_MAX_ROWS, "max_rows must be <=4096")
    count = min(total, max_rows)
    return tuple(((2 * k + 1) * total) // (2 * count) for k in range(count))


def _packed_shape(path: Path) -> tuple[int, int]:
    # Read only the compressed safetensors header for the canonical row census.
    # The sampled shard is later decoded normally; inventory hashes bind both.
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        size = int.from_bytes(stream.read(8), "little")
        _require(0 < size <= 16 * 1024 * 1024, f"invalid packed header length in {path}")
        header = json.loads(stream.read(size))
    spec = header.get("tokens", {})
    shape = spec.get("shape", [])
    _require(spec.get("dtype") == "U16" and len(shape) == 2, f"invalid packed tokens header in {path}")
    return _positive(shape[0], "packed rows"), _positive(shape[1], "packed capacity") - 1


@dataclass(frozen=True)
class _Document:
    tokens: list[int]
    segments: list[SegFeatures]
    source_lengths: list[int | None]


class _SourceDocuments:
    """One decoded source shard resident at a time; sampled documents cached separately."""

    def __init__(self, features: list[Path], parquets: dict[str, Path], overlay: Path, tokenizer: Any,
                 context_len: int, splice: SpliceProvenanceIndex | None):
        self.features = {shard_stem_from_path(p): p for p in features}
        self.parquets = parquets
        self.overlay = overlay
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.splice = splice
        self.cache: dict[tuple[str, int], _Document] = {}
        self.stem: str | None = None
        self.data: dict[str, Any] = {}
        self.texts: list[str] = []
        self.doc_rows: dict[int, list[int]] = {}
        self.bos_lengths: dict[str, int] = {}
        self.starts = t.empty(0, dtype=t.int32)
        self.ends = t.empty(0, dtype=t.int32)

    def get(self, stem: str, doc: int) -> _Document:
        key = (stem, doc)
        if key in self.cache:
            return self.cache[key]
        if self.stem != stem:
            data = load_features(self.features[stem], keys={
                "doc_ids", "segment_indices", "segment_positions", "segment_char_starts",
                "segment_char_ends", "doc_token_counts", "offsets", "feature_ids",
            })
            _require(segment_mode_from_metadata(data, source=stem) is SegmentMode.fixed_tokens,
                     f"source {stem} is not fixed_tokens")
            _require(data.get("segment_size") == FIXED_CADENCE_SOURCE_WIDTH, f"source {stem} width is not 128")
            self.texts = read_parquet_texts(self.parquets[stem])
            _require(data.get("n_docs") == len(self.texts), f"feature/parquet document count mismatch in {stem}")
            _require(len(data["doc_token_counts"]) == len(self.texts), f"source token counts missing in {stem}")
            self.starts, self.ends = validate_token_boundary_overlay_shard(
                overlay_shard_path(self.overlay, stem), expected_n_segments=int(data["n_segments"]),
            )
            self.doc_rows = {}
            for row, document in enumerate(data["doc_ids"].tolist()):
                _require(0 <= document < len(self.texts), f"invalid document mapping in {stem}")
                self.doc_rows.setdefault(document, []).append(row)
            _require(set(self.doc_rows) == set(range(len(self.texts))), f"missing source documents in {stem}")
            self.bos_lengths[stem] = detect_source_bos_char_len_from_data(data, self.parquets[stem])
            if self.splice is not None:
                _require(self.splice.shard_n_docs(stem) == len(self.texts), f"splice document count mismatch in {stem}")
            self.data, self.stem = data, stem
        rows = self.doc_rows.get(doc)
        _require(rows is not None, f"source document missing: {stem}:{doc}")
        assert rows is not None
        data = self.data
        _require(data["segment_indices"][rows].tolist() == list(range(len(rows))),
                 f"source ordinals are not a contiguous prefix in {stem}:{doc}")
        if self.splice is not None:
            _require(self.splice.doc_n_segments(stem, doc) == len(rows), f"splice segment count mismatch in {stem}:{doc}")
        starts, ends = self.starts[rows], self.ends[rows]
        chars_start = data["segment_char_starts"][rows].tolist()
        chars_end = data["segment_char_ends"][rows].tolist()
        tokens = self.tokenizer.encode(self.texts[doc], prepend=self.tokenizer.get_bos_token_id())
        expected_starts, expected_ends = compute_doc_token_boundaries_from_tokens(
            tokens, chars_start, chars_end, self.tokenizer, source_bos_char_len=self.bos_lengths[stem],
        )
        _require(t.equal(starts, expected_starts) and t.equal(ends, expected_ends),
                 f"token-boundary overlay disagrees with paired parquet/tokenizer for {stem}:{doc}")
        _require(int(starts[0]) == 0 and bool((ends >= starts).all())
                 and t.equal(starts[1:], ends[:-1]), f"nonmonotone ownership in {stem}:{doc}")
        positions = data["segment_positions"][rows].tolist()
        count = int(data["doc_token_counts"][doc])
        context_window = _positive(data.get("context_window"), "source context_window")
        _require(positions[0] == 0 and all(a < b for a, b in pairwise(positions)),
                 f"invalid source positions in {stem}:{doc}")
        retained_source = min(count, context_window)
        # A successor proves an interior length. A terminal count proves a tail
        # only if it actually extends beyond the last start; otherwise exclude.
        lengths: list[int | None] = [b - a for a, b in pairwise(positions)]
        lengths.append(retained_source - positions[-1] if retained_source > positions[-1] else None)
        segments = []
        for index, cs, ce, lo, hi in zip(rows, chars_start, chars_end, starts, ends, strict=True):
            begin, end = (int(v) for v in data["offsets"][index:index + 2])
            ids = data["feature_ids"][begin:end].to(t.int64)
            segments.append(SegFeatures(ids, t.empty(0, dtype=t.int64), t.zeros(len(ids), dtype=t.uint8),
                                        cs, ce, token_start=int(lo), token_end=int(hi)))
        tokens = truncate_to_boundary_coverage(tokens, segments, self.context_len,
                                               PositiveInt.parse(FIXED_CADENCE_SOURCE_WIDTH), f"{stem}:{doc}")
        result = _Document(tokens, segments, lengths)
        self.cache[key] = result
        return result


def _measure_row(packed: PackedRowShard, row: int, prov: dict[str, t.Tensor], stems: list[str],
                 train_stems: set[str], source: _SourceDocuments,
                 exclusions: Counter[str]) -> tuple[list[int], set[tuple[str, int]], set[tuple[str, int]], int]:
    begin, end = (int(v) for v in packed.row_seg_offsets[row:row + 2])
    _require(0 <= begin < end <= packed.total_segments, "invalid packed row segment offsets")
    groups: list[tuple[int, int]] = []
    for index in range(begin, end):
        if index == begin or packed.seg_doc_id[index] != packed.seg_doc_id[index - 1]:
            groups.append((index, index + 1))
        else:
            groups[-1] = (groups[-1][0], index + 1)
    row_docs = []
    identities = []
    documents = []
    for ordinal, (a, b) in enumerate(groups):
        _require(int(packed.seg_doc_id[a]) == ordinal, "packed document IDs are not contiguous")
        base, doc = int(prov[PROVENANCE_BASE_SHARD_IDX_KEY][a]), int(prov[PROVENANCE_DOC_IDX_KEY][a])
        _require(0 <= base < len(stems) and doc >= 0, "packed source identity out of range")
        stem = stems[base]
        _require(stem in train_stems, f"held-out/non-TRAIN source {stem} appears in packed TRAIN rows")
        _require(bool((prov[PROVENANCE_BASE_SHARD_IDX_KEY][a:b] == base).all())
                 and bool((prov[PROVENANCE_DOC_IDX_KEY][a:b] == doc).all())
                 and prov[PROVENANCE_SEG_ORDINAL_KEY][a:b].tolist() == list(range(b - a)),
                 f"packed provenance is not a document prefix for {stem}:{doc}")
        document = source.get(stem, doc)
        _require(b - a <= len(document.segments), f"too many packed segments for {stem}:{doc}")
        offset = int(packed.seg_token_start[a])
        next_offset = int(packed.seg_token_start[groups[ordinal + 1][0]]) if ordinal + 1 < len(groups) else packed.context_len
        placed = min(len(document.tokens), next_offset - offset, packed.context_len - offset)
        _require(placed > 0, f"packed document has no inputs: {stem}:{doc}")
        _require(packed.tokens[row, offset:offset + placed].tolist() == document.tokens[:placed],
                 f"packed token bytes disagree with paired parquet for {stem}:{doc}")
        if ordinal + 1 < len(groups):
            _require(offset + placed == next_offset, "gap between packed documents")
        row_docs.append((offset, document.segments, document.tokens[:placed]))
        identities.append((stem, doc))
        documents.append((document, offset, placed))
    # Replay the production loader/packer range construction, including its
    # historical last-range extension over zero padding. Count only real input
    # intersections below. This authenticates starts/ends, not guessed widths.
    reconstructed = _row_to_csr(packed.tokens[row].to(t.int64), row_docs, packed.context_len, source.tokenizer)
    _require(t.equal(reconstructed.seg_token_start.to(t.int64), packed.seg_token_start[begin:end].to(t.int64))
             and t.equal(reconstructed.seg_token_end.to(t.int64), packed.seg_token_end[begin:end].to(t.int64)),
             "packed starts/ends disagree with training loader")
    used = sum(placed for _, _, placed in documents)
    _require(bool((packed.tokens[row, used:packed.context_len] == 0).all()),
             "unmapped packed input positions are not zero padding")
    exclusions["padding_input_positions"] += packed.context_len - used
    exclusions["prediction_only_positions"] += 1  # the T+1-th stored token, never an input
    widths = []
    contributing: set[tuple[str, int]] = set()
    for identity, (document, offset, placed), (a, b) in zip(identities, documents, groups, strict=True):
        for ordinal, (segment, length) in enumerate(zip(document.segments, document.source_lengths, strict=True)):
            lo, hi = segment.token_start, segment.token_end
            assert lo is not None and hi is not None
            if length is None or length <= 0:
                exclusions[WindowExclusion.AMBIGUOUS_LENGTH.value] += 1
            elif length > FIXED_CADENCE_SOURCE_WIDTH:
                exclusions[WindowExclusion.NONSTANDARD_LENGTH.value] += 1
            elif length < FIXED_CADENCE_SOURCE_WIDTH:
                exclusions[WindowExclusion.INCOMPLETE_TAIL.value] += 1
            elif lo == placed and offset + lo == packed.context_len:
                exclusions[WindowExclusion.PREDICTION_ONLY.value] += 1
            elif hi > placed or ordinal >= b - a:
                exclusions[WindowExclusion.CLIPPED.value] += 1
            else:
                index = a + ordinal
                width = min(int(packed.seg_token_end[index]), offset + placed) - int(packed.seg_token_start[index])
                _require(width == hi - lo and width > 0, "fully retained packed width disagrees with overlay")
                widths.append(width)
                contributing.add(identity)
    return widths, contributing, set(identities), len(identities)


def measure_fixed_self_cond_segment_size_rustbpe(
    *, checkpoint_meta: Mapping[str, Any], feature_dir: str | Path, parquet_dir: str | Path,
    tokenizer: Any, max_rows: int = FIXED_CADENCE_MAX_ROWS,
) -> FixedCadenceMeasurement:
    """Authenticate artifacts, then count full 128-source-token window occurrences.

    Uses the training discovery path and TRAIN split (limit shards, then hold
    out the last parquet). Spliced synthetic-BOS windows of width 129 are not
    128-token windows. Incomplete or unprovable terminal lengths are excluded.
    Final stored prediction targets and zero padding never contribute to K.
    """
    _positive(max_rows, "max_rows")
    _require(max_rows <= FIXED_CADENCE_MAX_ROWS, "max_rows must be <=4096")
    feature_dir, parquet_dir = Path(feature_dir).resolve(), Path(parquet_dir).resolve()
    _require(resolve_checkpoint_sft_segment_mode(checkpoint_meta, checkpoint_dir=str(feature_dir))
             is SegmentMode.fixed_tokens, "checkpoint is not fixed_tokens")
    config = checkpoint_meta.get("user_config")
    _require(isinstance(config, Mapping), "checkpoint lacks user_config")
    assert isinstance(config, Mapping)
    _require(isinstance(config.get("feature_dir"), str) and Path(config["feature_dir"]).resolve() == feature_dir,
             "checkpoint feature_dir differs from requested source inventory")
    _require(type(config.get("segment_size")) is int and config["segment_size"] == FIXED_CADENCE_SOURCE_WIDTH, "checkpoint source width is not 128")
    # Legacy checkpoints may omit this key; native packed manifests and loader
    # shard state must still authenticate them below. Explicit non-True values
    # are contradictions; the unrelated historical use_precomputed key is ignored.
    _require(config.get("use_precomputed_batches", True) is True,
             "checkpoint did not use packed rows (use_precomputed_batches)")
    _require(config.get("next_segment_conditioning_offset", False) is False, "next-segment conditioning is unsupported")
    context_len = _positive(config.get("max_seq_len"), "checkpoint max_seq_len")
    _require(checkpoint_meta.get("max_seq_len", context_len) == context_len, "checkpoint context widths disagree")
    max_shards = config.get("max_train_shards")
    if max_shards is not None:
        _positive(max_shards, "max_train_shards")
    raw_delta = config.get("feature_delta")
    _require(raw_delta is None or isinstance(raw_delta, dict), "invalid feature_delta metadata")
    delta = parse_feature_delta(raw_delta)
    _require(delta is None or delta.mode is DeltaMode.splice_counterfactual,
             "packed cadence does not support history-dependent feature_delta")
    spliced = delta is not None
    recorded_parquet = config.get("data_dir")
    if recorded_parquet is None and spliced:
        recorded_parquet = str(feature_dir / "_parquets")
    _require(isinstance(recorded_parquet, str) and Path(recorded_parquet).resolve() == parquet_dir,
             "checkpoint paired-parquet directory differs")
    splice = SpliceProvenanceIndex(feature_dir) if spliced else None
    features = glob_features(feature_dir)
    _require(bool(features), "source feature inventory is empty")
    stems = [shard_stem_from_path(path) for path in features]
    parquets = [Path(p) for p in list_parquet_files(str(parquet_dir))]
    _require([p.stem for p in parquets] == stems, "source/paired-parquet inventories disagree")
    selected_parquets = parquets if max_shards is None else parquets[:max_shards]
    _require(len(selected_parquets) >= 2, "TRAIN split requires at least two paired shards")
    train_stems = {p.stem for p in selected_parquets[:-1]}
    early_signatures = {p: file_signature(p) for p in features + parquets}
    checksum = compute_tokenizer_checksum(tokenizer)
    stats = find_statistics_path(feature_dir)
    _require(stats is not None, "training statistics artifact is missing")
    assert stats is not None
    early_signatures[Path(stats)] = file_signature(Path(stats))
    pct_checksum = load_percentile_checksum(stats)
    _require(isinstance(pct_checksum, str) and len(pct_checksum) == 64, "training percentile checksum is missing")
    assert isinstance(pct_checksum, str)
    recorded_tok = load_tokenizer_checksum(stats)
    _require(recorded_tok is None or recorded_tok == checksum, "wrong tokenizer checksum in training statistics")
    packed_dir = _packed_row_train_dir(feature_dir, percentile_checksum=pct_checksum,
                                       tokenizer_checksum=checksum, context_len=context_len, max_train_shards=max_shards)
    if (packed_dir / "manifest.json").is_file():
        early_signatures[packed_dir / "manifest.json"] = file_signature(packed_dir / "manifest.json")
    manifest = read_manifest(packed_dir)
    _require(isinstance(manifest, dict), f"packed TRAIN manifest missing at {packed_dir}")
    assert manifest is not None
    metadata = manifest.get("metadata", {})
    _require(isinstance(metadata, dict), "invalid packed manifest metadata")
    expected = {"type": "packed_rows", "context_len": context_len, "max_shards": max_shards,
                "pct_checksum": pct_checksum, "tok_checksum": checksum, "base_shard_stems": stems}
    for key, value in expected.items():
        _require(key in metadata and type(metadata[key]) is type(value) and metadata[key] == value,
                 f"packed provenance mismatch for {key}")
    for key in ("buffer_size", "rows_per_shard"):
        _positive(metadata.get(key), f"packed {key}")
    _require(manifest.get("checksum") == f"{pct_checksum}:{checksum}" and manifest.get("kind") == OverlayKind.PACKED_ROW.value,
             "packed manifest checksum/kind mismatch")
    packed_paths = glob_overlay_shards(packed_dir)
    prov_paths = glob_overlay_shards(provenance_dir(packed_dir))
    _require(bool(packed_paths) and [p.name for p in packed_paths] == [p.name for p in prov_paths]
             and metadata.get("provenance_shards") == len(packed_paths), "packed provenance shard inventory mismatch")
    state = checkpoint_meta.get("dataloader_state_dict", {})
    states = state.get("rank_states", [state]) if isinstance(state, Mapping) else []
    _require(isinstance(states, list) and bool(states) and all(
        isinstance(s, Mapping) and s.get("shard") in {p.name for p in packed_paths} for s in states
    ), "checkpoint loader state does not authenticate packed TRAIN shards")
    token_manifest_path = feature_dir / "tok" / checksum[:16] / "seg_boundaries" / "manifest.json"
    if token_manifest_path.is_file():
        early_signatures[token_manifest_path] = file_signature(token_manifest_path)
    base_hash = compute_base_shard_content_hash(features)
    _require(manifest.get("schema_version") == CONTENT_BOUND_MANIFEST_VERSION and manifest.get("base_shard_content_hash") == base_hash,
             "packed provenance source inventory hash mismatch (v2 manifest required)")
    overlay = discover_overlay(feature_dir, OverlayKind.SEG_BOUNDARY, checksum, precomputed_base_hash=base_hash)
    _require(overlay is not None, "validated token-boundary overlay is missing for decoder tokenizer")
    assert overlay is not None
    token_manifest = read_manifest(overlay)
    _require(token_manifest is not None and token_manifest.get("schema_version") == CONTENT_BOUND_MANIFEST_VERSION
             and token_manifest.get("base_shard_content_hash") == base_hash
             and token_manifest.get("kind") == OverlayKind.SEG_BOUNDARY.value, "token-boundary v2 provenance required")
    token_paths = [overlay_shard_path(overlay, stem) for stem in stems]
    sidecars = sorted((feature_dir / "_splice_provenance").glob("*.json")) if spliced else []
    if spliced:
        _require([p.stem for p in sidecars] == stems, "splice sidecar inventory mismatch")
    all_paths = features + parquets + packed_paths + prov_paths + token_paths + sidecars + [
        Path(stats), packed_dir / "manifest.json", overlay / "manifest.json",
    ]
    signatures = {p: file_signature(p) for p in all_paths} | early_signatures
    inventories = {label: file_inventory(paths) for label, paths in {
        "source": features, "paired_parquet": parquets, "packed": packed_paths,
        "packed_provenance": prov_paths, "token_boundaries": token_paths, "splice": sidecars,
        "manifests_and_statistics": all_paths[-3:],
    }.items()}
    shapes = [_packed_shape(p) for p in packed_paths]
    _require(all(width == context_len for _, width in shapes), "packed context shape mismatch")
    _require(all(rows == metadata["rows_per_shard"] for rows, _ in shapes[:-1])
             and shapes[-1][0] <= metadata["rows_per_shard"], "noncanonical packed shard row counts")
    total = sum(rows for rows, _ in shapes)
    selected = set(evenly_spaced_rows(total, max_rows))
    source = _SourceDocuments(features, {p.stem: p for p in parquets}, overlay, tokenizer, context_len, splice)
    widths: list[int] = []
    documents: set[tuple[str, int]] = set()
    sampled_documents: set[tuple[str, int]] = set()
    exclusions: Counter[str] = Counter({reason.value: 0 for reason in WindowExclusion})
    contributing_rows = occurrences = offset = 0
    row_ids = []
    for path, (n_rows, _) in zip(packed_paths, shapes, strict=True):
        local_rows = sorted(index - offset for index in selected if offset <= index < offset + n_rows)
        offset += n_rows
        if not local_rows:
            continue
        packed = load_packed_row_shard(path)
        _require((packed.n_rows, packed.context_len) == (n_rows, context_len), "packed header/body disagreement")
        prov = load_overlay_shard(provenance_dir(packed_dir) / path.name)
        _require(set(prov) == {PROVENANCE_BASE_SHARD_IDX_KEY, PROVENANCE_DOC_IDX_KEY, PROVENANCE_SEG_ORDINAL_KEY}
                 and all(v.dtype == t.int32 and v.ndim == 1 and len(v) == packed.total_segments for v in prov.values()),
                 "packed provenance tensor mismatch")
        for row in local_rows:
            measured, contributed, sampled, count = _measure_row(packed, row, prov, stems, train_stems, source, exclusions)
            row_ids.append({"shard": path.name, "row": row})
            widths.extend(measured)
            documents.update(contributed)
            sampled_documents.update(sampled)
            contributing_rows += bool(measured)
            occurrences += count
        del packed
    _require(bool(widths), "no eligible fully retained 128-source-token TRAIN windows")
    _require(all(file_signature(p) == sig for p, sig in signatures.items()), "artifacts changed during measurement")
    _require(glob_features(feature_dir) == features and [Path(p) for p in list_parquet_files(str(parquet_dir))] == parquets
             and glob_overlay_shards(packed_dir) == packed_paths and glob_overlay_shards(provenance_dir(packed_dir)) == prov_paths,
             "artifact inventory changed during measurement")
    mean = Fraction(sum(widths), len(widths))
    ordered = sorted(widths)
    return FixedCadenceMeasurement(
        cadence=PositiveInt.parse(round(mean)), windows=PositiveInt.parse(len(widths)),
        rows=PositiveInt.parse(contributing_rows), documents=PositiveInt.parse(len(documents)),
        sampled_rows=PositiveInt.parse(len(selected)), sampled_documents=PositiveInt.parse(len(sampled_documents)),
        document_occurrences=PositiveInt.parse(occurrences), mean=mean,
        minimum=PositiveInt.parse(min(widths)), maximum=PositiveInt.parse(max(widths)),
        quantiles={f"p{q:02d}": ordered[max(0, (len(ordered) * q + 99) // 100 - 1)] for q in (5, 25, 50, 75, 95)},
        artifact_signatures=tuple(signatures.items()),
        exclusions=dict(exclusions), provenance={
            "implementation_version": FIXED_CADENCE_VERSION,
            "cadence_fidelity": "measured_rustbpe_approximation",
            "boundary_protocol": "fixed_prompt_seed_then_generated_input_replay_v1",
            "source_width": FIXED_CADENCE_SOURCE_WIDTH, "tokenizer_checksum": checksum,
            "document_count_scope": "documents_with_at_least_one_packed_input_provenance_entry",
            "split": DatasetSplit.TRAIN.value, "held_out_parquet": str(selected_parquets[-1]),
            "packing": metadata, "packed_manifest": manifest, "token_boundary_manifest": token_manifest,
            "source_bos_char_lengths": source.bos_lengths, "inventories": inventories,
            "sample_policy": {"traversal": "sorted_packed_shard_filename_then_physical_row_index_v1",
                              "total_rows": total, "max_rows": max_rows, "selected_rows": row_ids,
                              "indices": "floor((2*k+1)*total/(2*min(total,max_rows)))",
                              "weighting": "equal_weight_per_retained_window_occurrence",
                              "quantiles": "nearest_rank", "rounding": "exact_fraction_ties_to_even"},
            "checkpoint_binding": "recorded_paths_loader_shards_and_native_manifests",
        },
    )
