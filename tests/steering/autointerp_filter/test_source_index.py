from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
import zstandard
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn
from safetensors.torch import save as safetensors_save

import steering.eval.autointerp_filter.source_index as si
from nanochat.atomic_io import atomic_write_text
from scripts.benchmark.benchmark_autointerp_source_index_build import (
    _summarize_progress_events,
)
from steering.eval.autointerp_filter.source_index import (
    SourceIndexData,
    SourceIndexProgressEvent,
    SourceIndexProgressEventKind,
    SourceIndexRequest,
    _assert_unique_shard_stems,
    _connect,
    _decode_span_for_entry_chunk,
    _DecodedEntryChunk,
    _finalize_vectorized_rows,
    _insert_intruder_refs,
    _iter_spooled_entry_chunks,
    _load_spooled_source_index_shard_data,
    _positive_row_from_tuple,
    _rank_row_stable_keys,
    _reduce_source_index_candidates,
    _scan_source_index_shard_file_topk_vectorized,
    _SpooledSourceIndexShardData,
    _vectorized_chunk_survivors,
    _worst_magnitudes_from_state,
    fingerprint_paths,
    load_or_build_source_index,
    segment_token_bounds,
    shard_index_from_stem,
    validate_source_index_cache,
)
from steering.eval.autointerp_filter.strict_json import load_json_object
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCacheError,
    AutoInterpDataError,
    SourceDocRef,
)
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    glob_features,
    load_features,
    save_features,
    spooled_feature_shard,
)
from tests.steering.autointerp_filter._source_index_workers import (
    completion_marker_path,
    fail_second_shard_worker,
    reverse_completion_order_worker,
)


def _positive_row_tuple(row: si._PositiveRow) -> tuple:
    return (
        row.feature_id,
        row.magnitude,
        row.stable_key,
        row.doc_ref.shard_stem,
        row.doc_ref.shard_index,
        row.doc_ref.doc_id,
        row.doc_ref.segment_token_start,
        row.doc_ref.segment_token_end,
    )


def _base_shard_data() -> dict:
    return {
        TensorKey.OFFSETS: t.tensor([0, 2, 3, 4, 5, 6, 7], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor([0, 1, 1, 1, 1, 2, 2], dtype=t.int64),
        TensorKey.FEATURE_MAGS: t.tensor(
            [1.0, 0.5, 3.0, 3.0, 2.5, 8.0, 1.0],
            dtype=t.float32,
        ),
        TensorKey.DOC_IDS: t.tensor([0, 1, 2, 3, 4, 5], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([0, 4, 8, 12, 16, 20], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_ENDS: t.tensor([4, 8, 12, 16, 20, 24], dtype=t.int64),
        "n_docs": 6,
        "n_segments": 6,
        "layer": 22,
        "context_window": 128,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "context_independent_features": "true",
        "ignore_bos_features": "true",
    }


def _write_feature_shard(feature_dir: Path, data: dict | None = None) -> Path:
    feature_dir.mkdir(parents=True, exist_ok=True)
    return save_features(
        _base_shard_data() if data is None else data,
        feature_dir / "shard_00000_features.safetensors.zst",
    )


def _write_raw_safetensors_zst(
    path: Path,
    *,
    tensors: dict[str, t.Tensor],
    metadata: dict[str, str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf_bytes = safetensors_save(tensors, metadata=metadata or {})
    compressed = zstandard.ZstdCompressor().compress(sf_bytes)
    path.write_bytes(compressed)
    return path


@st.composite
def _small_source_shard_specs(draw: DrawFn) -> dict:
    n_sae_features = draw(st.integers(min_value=1, max_value=8))
    n_rows = draw(st.integers(min_value=1, max_value=6))
    row_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    n_entries = sum(row_lengths)
    feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features + 2),
            min_size=n_entries,
            max_size=n_entries,
        )
    )
    magnitudes = draw(
        st.lists(
            st.integers(min_value=0, max_value=20),
            min_size=n_entries,
            max_size=n_entries,
        )
    )
    target_feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features - 1),
            unique=True,
            max_size=n_sae_features,
        )
    )
    return {
        "n_sae_features": n_sae_features,
        "row_lengths": row_lengths,
        "feature_ids": feature_ids,
        "magnitudes": magnitudes,
        "target_feature_ids": tuple(target_feature_ids),
        "max_positive_refs": draw(st.integers(min_value=1, max_value=4)),
        "chunk_size": draw(st.integers(min_value=1, max_value=n_entries + 2)),
        "merge_pending_min_rows": draw(st.sampled_from([1, 3, None, 10_000_000])),
    }


def _source_shard_data_from_spec(spec: dict) -> dict:
    # Legacy scanners are chunk-size-dependent on duplicate
    # (feature, stable_key) entries (documented defect); shared specs stay
    # duplicate-free so legacy-vs-oracle assertions remain well-defined.
    # Duplicate-key semantics are pinned by dedicated deterministic tests
    # and the vectorized-only property test below.
    row_lengths: list[int] = []
    feature_ids: list[int] = []
    magnitudes: list[float] = []
    cursor = 0
    for raw_length in spec["row_lengths"]:
        seen: set[int] = set()
        kept = 0
        for offset in range(raw_length):
            fid = spec["feature_ids"][cursor + offset]
            if fid in seen:
                continue
            seen.add(fid)
            feature_ids.append(fid)
            magnitudes.append(spec["magnitudes"][cursor + offset])
            kept += 1
        cursor += raw_length
        row_lengths.append(kept)
    offsets = [0]
    for row_length in row_lengths:
        offsets.append(offsets[-1] + row_length)
    n_rows = len(row_lengths)
    return {
        TensorKey.OFFSETS: t.tensor(offsets, dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor(feature_ids, dtype=t.int64),
        TensorKey.FEATURE_MAGS: t.tensor(magnitudes, dtype=t.float32),
        TensorKey.DOC_IDS: t.arange(n_rows, dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.arange(n_rows, dtype=t.int64) * 4,
        TensorKey.SEGMENT_TOKEN_ENDS: t.arange(1, n_rows + 1, dtype=t.int64) * 4,
        "n_docs": n_rows,
        "n_segments": n_rows,
        "layer": 22,
        "context_window": 128,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "context_independent_features": "true",
        "ignore_bos_features": "true",
    }


def _source_shard_data(
    row_lengths: list[int],
    feature_ids: list[int],
    magnitudes: list[float],
    *,
    doc_ids: list[int] | None = None,
    token_starts: list[int] | None = None,
    token_ends: list[int] | None = None,
    context_independent_features: str = "true",
    ignore_bos_features: str = "true",
) -> dict:
    offsets = [0]
    for row_length in row_lengths:
        offsets.append(offsets[-1] + row_length)
    n_rows = len(row_lengths)
    if doc_ids is None:
        doc_ids = list(range(n_rows))
    if token_starts is None:
        token_starts = [row_index * 4 for row_index in range(n_rows)]
    if token_ends is None:
        token_ends = [start + 4 for start in token_starts]
    return {
        TensorKey.OFFSETS: t.tensor(offsets, dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor(feature_ids, dtype=t.int64),
        TensorKey.FEATURE_MAGS: t.tensor(magnitudes, dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor(doc_ids, dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.tensor(token_starts, dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_ENDS: t.tensor(token_ends, dtype=t.int64),
        "n_docs": n_rows,
        "n_segments": n_rows,
        "layer": 22,
        "context_window": 128,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "context_independent_features": context_independent_features,
        "ignore_bos_features": ignore_bos_features,
    }


def _data_marker(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "shard_00000.parquet"
    if not path.exists():
        path.write_text("source-index fingerprint fixture\n")
    return path


def _request(
    root: Path,
    feature_dir: Path,
    shard_path: Path,
    *,
    target_feature_ids: tuple[int, ...] = (1, 2, 3),
    n_sae_features: int = 4,
    max_positive_refs_per_feature: int = 3,
    max_intruder_refs: int = 4,
    chunk_size: int = 2,
    seed: int = 17,
    context_independent_features: bool = True,
    ignore_bos_features: bool = True,
    scan_workers: int = 1,
    force_rebuild: bool = False,
    shared_cache_dir: Path | None = None,
) -> SourceIndexRequest:
    data_path = _data_marker(root)
    index_dir = root / "index"
    return SourceIndexRequest(
        feature_dir=feature_dir,
        data_dir=str(data_path.parent),
        index_path=index_dir / "source_doc_index.sqlite",
        manifest_path=index_dir / "source_doc_index.manifest.json",
        shard_paths=(shard_path,),
        parquet_paths=(data_path,),
        target_feature_ids=target_feature_ids,
        n_sae_features=n_sae_features,
        max_positive_refs_per_feature=max_positive_refs_per_feature,
        max_intruder_refs=max_intruder_refs,
        seed=seed,
        feature_fingerprint=fingerprint_paths((shard_path,)),
        data_fingerprint=fingerprint_paths((data_path,)),
        context_independent_features=context_independent_features,
        ignore_bos_features=ignore_bos_features,
        chunk_size=chunk_size,
        scan_workers=scan_workers,
        force_rebuild=force_rebuild,
        shared_cache_dir=shared_cache_dir,
    )


def _row_ref(row: tuple) -> SourceDocRef:
    return SourceDocRef(
        shard_stem=str(row[3]),
        shard_index=int(row[4]),
        doc_id=int(row[5]),
        segment_token_start=int(row[6]),
        segment_token_end=int(row[7]),
    )


def _brute_force_topk_rows(
    shard_path: Path,
    target_feature_ids: tuple[int, ...],
    n_sae_features: int,
    max_positive_refs: int,
) -> list[tuple]:
    """Chunk-independent true reduction over all raw entries of a shard."""
    data = load_features(shard_path)
    offsets = data[TensorKey.OFFSETS].to(t.int64)
    fids = data[TensorKey.FEATURE_IDS].to(t.int64)
    mags = data[TensorKey.FEATURE_MAGS].to(t.float32)
    doc_ids = data[TensorKey.DOC_IDS].to(t.int64)
    starts, ends = segment_token_bounds(
        data, shard_path=shard_path, n_rows=int(offsets.numel()) - 1
    )
    stem = shard_path.stem.removesuffix(".safetensors")
    shard_index = shard_index_from_stem(stem)
    refs = [
        SourceDocRef(
            shard_stem=stem,
            shard_index=shard_index,
            doc_id=int(doc_ids[row].item()),
            segment_token_start=int(starts[row].item()),
            segment_token_end=int(ends[row].item()),
        )
        for row in range(int(offsets.numel()) - 1)
    ]
    allowed = set(target_feature_ids)
    buckets: dict[int, dict[str, tuple[float, SourceDocRef]]] = {}
    row_index = 0
    for entry_index in range(int(fids.numel())):
        while entry_index >= int(offsets[row_index + 1].item()):
            row_index += 1
        fid = int(fids[entry_index].item())
        if fid < 0 or fid >= n_sae_features or fid not in allowed:
            continue
        mag = float(mags[entry_index].item())
        ref = refs[row_index]
        key = ref.stable_key()
        bucket = buckets.setdefault(fid, {})
        old = bucket.get(key)
        if old is None or mag > old[0]:
            bucket[key] = (mag, ref)
    rows: list[tuple] = []
    for fid in sorted(buckets):
        ordered = sorted(buckets[fid].items(), key=lambda kv: (-kv[1][0], kv[0]))
        for key, (mag, ref) in ordered[:max_positive_refs]:
            rows.append(
                (
                    fid,
                    mag,
                    key,
                    ref.shard_stem,
                    ref.shard_index,
                    ref.doc_id,
                    ref.segment_token_start,
                    ref.segment_token_end,
                )
            )
    return rows


def _oracle_reduce(
    fids: list[int],
    ranks: list[int],
    mags: list[float],
    max_positive_refs: int,
) -> list[tuple[int, int, float]]:
    """Dict-based reference mirroring bucket semantics on (fid, rank, mag)."""
    buckets: dict[int, dict[int, float]] = {}
    for fid, rank, mag in zip(fids, ranks, mags, strict=True):
        bucket = buckets.setdefault(fid, {})
        if rank not in bucket or mag > bucket[rank]:
            bucket[rank] = mag
    out: list[tuple[int, int, float]] = []
    for fid in sorted(buckets):
        ordered = sorted(buckets[fid].items(), key=lambda kv: (-kv[1], kv[0]))
        out.extend((fid, rank, mag) for rank, mag in ordered[:max_positive_refs])
    return out


@st.composite
def _reduce_specs(draw: DrawFn) -> dict:
    n = draw(st.integers(min_value=0, max_value=60))
    return {
        "fids": draw(st.lists(st.integers(min_value=0, max_value=6), min_size=n, max_size=n)),
        "ranks": draw(st.lists(st.integers(min_value=0, max_value=9), min_size=n, max_size=n)),
        # small int-valued floats -> heavy exact ties
        "mags": draw(st.lists(st.integers(min_value=0, max_value=5), min_size=n, max_size=n)),
        "max_positive_refs": draw(st.integers(min_value=1, max_value=4)),
    }


@given(spec=_reduce_specs())
@settings(max_examples=200, deadline=None)
def test_reduce_candidates_matches_oracle(spec: dict) -> None:
    # int32 fids/ranks/rows and float32 mags mirror the scanner's state dtypes.
    fids = t.tensor(spec["fids"], dtype=t.int32)
    ranks = t.tensor(spec["ranks"], dtype=t.int32)
    mags = t.tensor(spec["mags"], dtype=t.float32)
    rows = t.arange(int(fids.shape[0]), dtype=t.int32)

    out_f, out_r, out_m, out_w = _reduce_source_index_candidates(
        fids, ranks, mags, rows, max_positive_refs=spec["max_positive_refs"]
    )

    assert out_f.dtype == t.int32
    assert out_r.dtype == t.int32
    assert out_m.dtype == t.float32
    assert out_w.dtype == t.int32
    got = list(zip(out_f.tolist(), out_r.tolist(), out_m.tolist(), strict=True))
    expected = _oracle_reduce(
        spec["fids"],
        spec["ranks"],
        mags.tolist(),  # float32-rounded magnitudes, matching the kernel input
        spec["max_positive_refs"],
    )
    assert got == expected
    # returned rows must carry the winning magnitude for their (fid, rank)
    for f, r, m, w in zip(
        out_f.tolist(),
        out_r.tolist(),
        out_m.tolist(),
        out_w.tolist(),
        strict=True,
    ):
        assert int(ranks[w]) == r and int(fids[w]) == f
        assert float(mags[w]) == m


def test_decode_span_for_entry_chunk_expands_to_segment_boundaries() -> None:
    offsets = t.tensor([0, 3, 7, 10], dtype=t.int64)

    decode_start, decode_end, local_offsets = _decode_span_for_entry_chunk(
        offsets,
        offsets[1:],
        chunk_start=4,
        chunk_end=8,
    )

    assert (decode_start, decode_end) == (3, 10)
    assert local_offsets.tolist() == [0, 4, 7]


@pytest.mark.parametrize(
    ("chunk_start", "chunk_end", "pattern"),
    [
        (2, 2, "invalid source-index entry chunk bounds"),
        (9, 11, "exceeds CSR payload length"),
    ],
)
def test_decode_span_for_entry_chunk_rejects_invalid_bounds(
    chunk_start: int,
    chunk_end: int,
    pattern: str,
) -> None:
    offsets = t.tensor([0, 3, 7, 10], dtype=t.int64)

    with pytest.raises(AutoInterpDataError, match=pattern):
        _decode_span_for_entry_chunk(
            offsets,
            offsets[1:],
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )


def test_iter_spooled_entry_chunks_decodes_mid_segment_chunks(
    tmp_path: Path,
) -> None:
    data = {
        **_base_shard_data(),
        TensorKey.OFFSETS: t.tensor([0, 3, 7, 10], dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor(
            [5, 1, 3, 2, 2, 6, 1, 0, 4, 4],
            dtype=t.int64,
        ),
        TensorKey.FEATURE_MAGS: t.tensor(
            [5.0, 1.0, 3.0, 2.0, 2.5, 6.0, 1.5, 0.5, 4.0, 4.5],
            dtype=t.float32,
        ),
        TensorKey.DOC_IDS: t.tensor([0, 1, 2], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([0, 4, 8], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_ENDS: t.tensor([4, 8, 12], dtype=t.int64),
        "n_docs": 3,
        "n_segments": 3,
    }
    shard_path = _write_feature_shard(tmp_path / "features", data)
    full = load_features(shard_path)

    with spooled_feature_shard(shard_path, temp_dir=tmp_path / "spool") as shard:
        scan_data = _load_spooled_source_index_shard_data(
            shard,
            shard_path=shard_path,
        )
        chunks = list(
            _iter_spooled_entry_chunks(
                shard,
                scan_data,
                chunk_size=4,
            )
        )

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert t.equal(
        t.cat([chunk.feature_ids for chunk in chunks]),
        full[TensorKey.FEATURE_IDS],
    )
    assert t.equal(
        t.cat([chunk.magnitudes for chunk in chunks]),
        full[TensorKey.FEATURE_MAGS],
    )


def test_spooled_feature_shard_exposes_temp_path_during_context(
    tmp_path: Path,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    spool_dir = tmp_path / "spool"

    with spooled_feature_shard(shard_path, temp_dir=spool_dir) as shard:
        temp_path = shard.temp_path
        assert temp_path.is_file()
        assert temp_path.parent == spool_dir

    assert not temp_path.exists()


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(spec=_small_source_shard_specs())
def test_vectorized_scanner_property_matches_brute_force_oracle(
    tmp_path: Path,
    spec: dict,
) -> None:
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data_from_spec(spec),
    )

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=spec["target_feature_ids"],
        n_sae_features=spec["n_sae_features"],
        max_positive_refs=spec["max_positive_refs"],
        chunk_size=spec["chunk_size"],
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=spec["merge_pending_min_rows"],
    )

    assert rows == _brute_force_topk_rows(
        shard_path,
        spec["target_feature_ids"],
        spec["n_sae_features"],
        spec["max_positive_refs"],
    )
    assert len(intruders) == len(spec["row_lengths"])


@st.composite
def _dense_source_shard_specs(draw: DrawFn) -> dict:
    n_sae_features = draw(st.integers(min_value=1, max_value=8))
    n_rows = draw(st.integers(min_value=2, max_value=8))
    row_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    n_entries = sum(row_lengths)
    feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features + 1),
            min_size=n_entries,
            max_size=n_entries,
        )
    )
    magnitudes = draw(
        st.lists(
            st.integers(min_value=0, max_value=7),
            min_size=n_entries,
            max_size=n_entries,
        )
    )
    target_feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_sae_features - 1),
            unique=True,
            min_size=1,
            max_size=n_sae_features,
        )
    )
    return {
        "n_sae_features": n_sae_features,
        "row_lengths": row_lengths,
        "feature_ids": feature_ids,
        "magnitudes": magnitudes,
        "target_feature_ids": tuple(target_feature_ids),
        "max_positive_refs": draw(st.integers(min_value=1, max_value=3)),
        "chunk_size": draw(st.integers(min_value=1, max_value=n_entries)),
        "merge_pending_min_rows": draw(st.sampled_from([1, 3, None, 10_000_000])),
    }


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(spec=_dense_source_shard_specs())
def test_vectorized_scanner_dense_property_matches_brute_force_oracle(
    tmp_path: Path,
    spec: dict,
) -> None:
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data_from_spec(spec),
    )

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=spec["target_feature_ids"],
        n_sae_features=spec["n_sae_features"],
        max_positive_refs=spec["max_positive_refs"],
        chunk_size=spec["chunk_size"],
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=spec["merge_pending_min_rows"],
    )

    assert rows == _brute_force_topk_rows(
        shard_path,
        spec["target_feature_ids"],
        spec["n_sae_features"],
        spec["max_positive_refs"],
    )
    assert len(intruders) == len(spec["row_lengths"])


def test_load_source_index_prunes_chunk_local_candidates_globally(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)

    chunk_one = load_or_build_source_index(
        _request(tmp_path / "chunk_one", feature_dir, shard_path, chunk_size=1)
    )
    chunk_all = load_or_build_source_index(
        _request(tmp_path / "chunk_all", feature_dir, shard_path, chunk_size=100)
    )

    assert chunk_one == chunk_all
    assert chunk_one.positives_by_feature[1] == [
        (SourceDocRef("shard_00000_features", 0, 1, 4, 8), 3.0),
        (SourceDocRef("shard_00000_features", 0, 2, 8, 12), 3.0),
        (SourceDocRef("shard_00000_features", 0, 3, 12, 16), 2.5),
    ]
    assert chunk_one.positives_by_feature[3] == []
    assert len(chunk_one.intruder_refs) == 4
    assert len({ref.stable_key() for ref in chunk_one.intruder_refs}) == 4


def test_source_index_manifest_pins_flags_and_requires_both_cache_files(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)

    load_or_build_source_index(request)

    manifest = json.loads(request.manifest_path.read_text())
    assert manifest["context_independent_features"] is True
    assert manifest["ignore_bos_features"] is True
    assert manifest["target_feature_ids"] == [1, 2, 3]

    with pytest.raises(AutoInterpCacheError, match="does not match"):
        load_or_build_source_index(
            _request(
                tmp_path,
                feature_dir,
                shard_path,
                context_independent_features=True,
                seed=18,
            )
        )

    request.index_path.unlink()
    with pytest.raises(AutoInterpCacheError, match="requires both SQLite and manifest"):
        load_or_build_source_index(request)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("duplicate", "contains invalid JSON:.*duplicate JSON object key"),
        ("nonfinite", "contains invalid JSON:.*non-finite JSON number"),
        ("nonobject", "must contain a JSON object"),
        ("malformed", "contains invalid JSON"),
        ("bool_integer_confusion", "does not match"),
    ],
)
def test_source_index_manifest_json_is_unambiguous_and_type_exact(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)
    load_or_build_source_index(request)
    manifest = json.loads(request.manifest_path.read_text(encoding="utf-8"))
    encoded = json.dumps(manifest, sort_keys=True)
    if corruption == "duplicate":
        encoded = encoded[:-1] + f', "seed": {manifest["seed"]}' + "}"
    elif corruption == "nonfinite":
        encoded = encoded.replace(f'"seed": {manifest["seed"]}', '"seed": NaN')
    elif corruption == "nonobject":
        encoded = "[]"
    elif corruption == "malformed":
        encoded = "{"
    else:
        manifest["n_feature_shards"] = True
        encoded = json.dumps(manifest, sort_keys=True)
    request.manifest_path.write_text(encoded, encoding="utf-8")

    with pytest.raises(AutoInterpCacheError, match=message):
        load_or_build_source_index(request)


def test_source_index_shared_cache_reuses_across_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    cache_dir = tmp_path / "shared_source_index"
    base_request = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        shared_cache_dir=cache_dir,
    )
    first_request = replace(
        base_request,
        index_path=tmp_path / "run_a" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "run_a" / "source_doc_index.manifest.json",
    )
    second_request = replace(
        base_request,
        index_path=tmp_path / "run_b" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "run_b" / "source_doc_index.manifest.json",
    )

    first_index = load_or_build_source_index(first_request)
    manifest = si._manifest_payload(first_request)
    resolved = si._resolve_shared_cache_request(first_request, manifest)
    assert resolved.index_path.is_file()
    assert resolved.manifest_path.is_file()
    assert not first_request.index_path.exists()
    assert not second_request.index_path.exists()

    def _unexpected_rebuild(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("shared source index should have been reused")

    monkeypatch.setattr(si, "_build_index", _unexpected_rebuild)
    second_index = load_or_build_source_index(second_request)

    assert second_index == first_index
    assert (
        si._resolve_shared_cache_request(
            second_request, si._manifest_payload(second_request)
        ).index_path
        == resolved.index_path
    )


def test_source_index_shared_cache_tables_match_output_local_index(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    first_shard = _write_feature_shard(feature_dir)
    save_features(
        _source_shard_data(
            row_lengths=[2, 1, 3],
            feature_ids=[1, 2, 1, 3, 1, 2],
            magnitudes=[4.0, 8.0, 4.0, 1.0, 9.0, 8.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    shard_paths = tuple(glob_features(feature_dir))
    base = replace(
        _request(tmp_path / "contract", feature_dir, first_shard, chunk_size=2),
        shard_paths=shard_paths,
        feature_fingerprint=fingerprint_paths(shard_paths),
    )
    local_request = replace(
        base,
        index_path=tmp_path / "local" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "local" / "source_doc_index.manifest.json",
    )
    shared_request = replace(
        base,
        index_path=tmp_path / "run_tag" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "run_tag" / "source_doc_index.manifest.json",
        shared_cache_dir=tmp_path / "shared_source_index",
    )

    local_index = load_or_build_source_index(local_request)
    shared_index = load_or_build_source_index(shared_request)
    resolved_shared = si._resolve_shared_cache_request(
        shared_request, si._manifest_payload(shared_request)
    )

    assert shared_index == local_index
    assert _dump_index_tables(resolved_shared.index_path) == _dump_index_tables(
        local_request.index_path
    )
    assert json.loads(resolved_shared.manifest_path.read_text()) == json.loads(
        local_request.manifest_path.read_text()
    )
    assert not shared_request.index_path.exists()
    assert not shared_request.manifest_path.exists()


def test_source_index_shared_cache_ignores_output_local_stale_files(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    output_local_request = _request(
        tmp_path / "output_local",
        feature_dir,
        shard_path,
    )
    load_or_build_source_index(output_local_request)
    stale_local_manifest = json.loads(output_local_request.manifest_path.read_text())
    stale_local_manifest["seed"] = stale_local_manifest["seed"] + 1
    output_local_request.manifest_path.write_text(
        json.dumps(stale_local_manifest, indent=2, sort_keys=True) + "\n"
    )
    shared_request = replace(
        output_local_request,
        shared_cache_dir=tmp_path / "shared_source_index",
    )

    shared_index = load_or_build_source_index(shared_request)
    resolved_shared = si._resolve_shared_cache_request(
        shared_request, si._manifest_payload(shared_request)
    )

    assert resolved_shared.index_path.is_file()
    assert resolved_shared.manifest_path.is_file()
    assert json.loads(output_local_request.manifest_path.read_text()) == stale_local_manifest
    assert shared_index == load_or_build_source_index(shared_request)


def test_source_index_shared_cache_key_uses_semantic_manifest_fields(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    cache_dir = tmp_path / "shared_source_index"
    base = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        scan_workers=1,
        shared_cache_dir=cache_dir,
    )
    worker_variant = replace(base, scan_workers=4)
    semantic_variant = replace(
        base,
        chunk_size=base.chunk_size + 1,
    )

    base_resolved = si._resolve_shared_cache_request(base, si._manifest_payload(base))
    worker_resolved = si._resolve_shared_cache_request(
        worker_variant,
        si._manifest_payload(worker_variant),
    )
    semantic_resolved = si._resolve_shared_cache_request(
        semantic_variant,
        si._manifest_payload(semantic_variant),
    )

    assert worker_resolved.index_path == base_resolved.index_path
    assert worker_resolved.manifest_path == base_resolved.manifest_path
    assert semantic_resolved.index_path != base_resolved.index_path
    assert semantic_resolved.manifest_path != base_resolved.manifest_path


def test_source_index_shared_cache_key_pins_manifest_semantics(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        shared_cache_dir=tmp_path / "shared_source_index",
    )
    manifest = si._manifest_payload(request)
    assert set(manifest) == {
        "schema_version",
        "method",
        "feature_dir",
        "data_dir",
        "feature_fingerprint",
        "data_fingerprint",
        "target_feature_ids",
        "n_sae_features",
        "shard_domain",
        "n_feature_shards",
        "n_data_shards",
        "max_positive_refs_per_feature",
        "max_intruder_refs",
        "seed",
        "context_independent_features",
        "ignore_bos_features",
        "chunk_size",
    }
    assert "scan_workers" not in manifest
    assert "force_rebuild" not in manifest
    assert "index_path" not in manifest
    assert "manifest_path" not in manifest
    assert "shared_cache_dir" not in manifest

    base_key = si._source_index_manifest_cache_key(manifest)
    assert base_key == si._source_index_manifest_cache_key(dict(reversed(list(manifest.items()))))
    for field_name, replacement_value in {
        "schema_version": manifest["schema_version"] + 1,
        "method": f"{manifest['method']}_next",
        "feature_dir": f"{manifest['feature_dir']}/other",
        "data_dir": f"{manifest['data_dir']}/other",
        "feature_fingerprint": f"{manifest['feature_fingerprint']}x",
        "data_fingerprint": f"{manifest['data_fingerprint']}x",
        "target_feature_ids": [1, 2],
        "n_sae_features": manifest["n_sae_features"] + 1,
        "shard_domain": [
            *manifest["shard_domain"],
            "shard_99999_features.safetensors.zst",
        ],
        "n_feature_shards": manifest["n_feature_shards"] + 1,
        "n_data_shards": manifest["n_data_shards"] + 1,
        "max_positive_refs_per_feature": manifest["max_positive_refs_per_feature"] + 1,
        "max_intruder_refs": manifest["max_intruder_refs"] + 1,
        "seed": manifest["seed"] + 1,
        "context_independent_features": not manifest["context_independent_features"],
        "ignore_bos_features": not manifest["ignore_bos_features"],
        "chunk_size": manifest["chunk_size"] + 1,
    }.items():
        mutated = dict(manifest)
        mutated[field_name] = replacement_value
        assert si._source_index_manifest_cache_key(mutated) != base_key, field_name


@pytest.mark.parametrize("missing_file", ["sqlite", "manifest"])
def test_source_index_shared_cache_partial_entry_fails_closed(
    tmp_path: Path,
    missing_file: str,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        shared_cache_dir=tmp_path / "shared_source_index",
    )
    resolved = si._resolve_shared_cache_request(request, si._manifest_payload(request))
    resolved.index_path.parent.mkdir(parents=True, exist_ok=True)
    if missing_file == "sqlite":
        resolved.manifest_path.write_text(
            json.dumps(si._manifest_payload(request), indent=2, sort_keys=True) + "\n"
        )
    else:
        resolved.index_path.write_bytes(b"not a valid sqlite index")

    with pytest.raises(AutoInterpCacheError, match="requires both SQLite and manifest"):
        load_or_build_source_index(request)


def test_source_index_shared_cache_stale_manifest_rejects_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        shared_cache_dir=tmp_path / "shared_source_index",
    )
    load_or_build_source_index(request)
    resolved = si._resolve_shared_cache_request(request, si._manifest_payload(request))
    stale_manifest = json.loads(resolved.manifest_path.read_text())
    stale_manifest["seed"] = stale_manifest["seed"] + 1
    resolved.manifest_path.write_text(json.dumps(stale_manifest, indent=2, sort_keys=True) + "\n")

    def _unexpected_rebuild(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("stale shared source index must fail validation, not rebuild")

    monkeypatch.setattr(si, "_build_index", _unexpected_rebuild)
    with pytest.raises(AutoInterpCacheError, match="does not match"):
        load_or_build_source_index(request)


def test_source_index_shared_cache_force_rebuild_replaces_stale_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(
        tmp_path / "shared_contract",
        feature_dir,
        shard_path,
        shared_cache_dir=tmp_path / "shared_source_index",
    )
    first_index = load_or_build_source_index(request)
    resolved = si._resolve_shared_cache_request(request, si._manifest_payload(request))
    stale_manifest = json.loads(resolved.manifest_path.read_text())
    stale_manifest["method"] = "source_doc_index_v0"
    resolved.manifest_path.write_text(json.dumps(stale_manifest, indent=2, sort_keys=True) + "\n")
    original_build_index = si._build_index
    build_calls: list[tuple[Path, Path]] = []

    def _recording_build_index(
        build_request: SourceIndexRequest,
        manifest: dict,
        *,
        progress_callback: Any = None,
    ) -> None:
        build_calls.append((build_request.index_path, build_request.manifest_path))
        original_build_index(
            build_request,
            manifest,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(si, "_build_index", _recording_build_index)
    rebuilt_index = load_or_build_source_index(replace(request, force_rebuild=True))

    assert build_calls == [(resolved.index_path, resolved.manifest_path)]
    assert rebuilt_index == first_index
    assert json.loads(resolved.manifest_path.read_text()) == si._manifest_payload(request)


def test_source_index_rejects_incompatible_method_generation(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)

    load_or_build_source_index(request)
    manifest = json.loads(request.manifest_path.read_text())
    assert manifest["method"] == "source_doc_index_v2"

    # Simulate an index left by the legacy chunk-dependent scanner, whose
    # manifest tagged method=source_doc_index_v1. Everything else (fingerprints,
    # targets, flags) matches, so only the pinned generation distinguishes it.
    manifest["method"] = "source_doc_index_v1"
    request.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(AutoInterpCacheError, match="incompatible generation"):
        load_or_build_source_index(request)

    # force_rebuild recovers by rebuilding with the current scanner.
    rebuilt = load_or_build_source_index(
        _request(tmp_path, feature_dir, shard_path, force_rebuild=True)
    )
    assert json.loads(request.manifest_path.read_text())["method"] == "source_doc_index_v2"
    assert rebuilt.positives_by_feature[1]


def test_source_index_load_rejects_corrupt_positive_ref_stable_key(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)
    load_or_build_source_index(request)

    conn = sqlite3.connect(str(request.index_path))
    try:
        conn.execute(
            "UPDATE positive_refs SET stable_key = 'corrupt-key' "
            "WHERE rowid = (SELECT rowid FROM positive_refs LIMIT 1)"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AutoInterpCacheError, match="stable_key mismatch"):
        load_or_build_source_index(request)


def test_source_index_load_rejects_corrupt_intruder_stable_key(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)
    load_or_build_source_index(request)

    conn = sqlite3.connect(str(request.index_path))
    try:
        conn.execute(
            "UPDATE intruder_refs SET stable_key = 'corrupt-key' "
            "WHERE rowid = (SELECT rowid FROM intruder_refs LIMIT 1)"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(AutoInterpCacheError, match="stable_key mismatch"):
        load_or_build_source_index(request)


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"target_feature_ids": (-1,)}, "target feature IDs out of range"),
        ({"target_feature_ids": (4,)}, "target feature IDs out of range"),
        ({"n_sae_features": 0}, "n_sae_features"),
        ({"max_positive_refs_per_feature": 0}, "max_positive_refs_per_feature"),
        ({"max_intruder_refs": 0}, "max_intruder_refs"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"scan_workers": 0}, "scan_workers must be >= 1"),
        ({"ignore_bos_features": False}, "ignore_bos_features=true"),
    ],
)
def test_source_index_request_validation(
    tmp_path: Path,
    overrides: dict,
    pattern: str,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)

    field_name = next(iter(overrides))
    direct_domain_error = field_name in {
        "n_sae_features",
        "max_positive_refs_per_feature",
        "max_intruder_refs",
        "chunk_size",
        "scan_workers",
    } or overrides == {"target_feature_ids": (-1,)}
    error_type = BeartypeCallHintParamViolation if direct_domain_error else AutoInterpDataError
    match = field_name if direct_domain_error else pattern
    with pytest.raises(error_type, match=match):
        load_or_build_source_index(_request(tmp_path / "bad", feature_dir, shard_path, **overrides))


def test_manifest_payload_excludes_scan_workers(tmp_path: Path) -> None:
    # scan_workers is a scheduling knob: it must be absent from the manifest
    # (cache identity) so one index is reusable across any worker count. Two
    # requests differing only in scan_workers must yield an identical payload.
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    payload_1 = si._manifest_payload(_request(tmp_path, feature_dir, shard_path, scan_workers=1))
    payload_4 = si._manifest_payload(_request(tmp_path, feature_dir, shard_path, scan_workers=4))
    assert "scan_workers" not in payload_1
    assert payload_1 == payload_4


def test_segment_token_bounds_prefers_explicit_then_positions_then_indices() -> None:
    explicit = {
        TensorKey.OFFSETS: t.tensor([0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([2, 10], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_ENDS: t.tensor([5, 12], dtype=t.int64),
        TensorKey.SEGMENT_POSITIONS: t.tensor([99, 100], dtype=t.int64),
        TensorKey.SEGMENT_INDICES: t.tensor([7, 8], dtype=t.int64),
        "segment_size": 4,
        "segment_mode": "fixed_tokens",
    }
    starts, ends = segment_token_bounds(
        explicit,
        shard_path=Path("shard_00000_features.safetensors.zst"),
        n_rows=2,
    )
    assert starts.tolist() == [2, 10]
    assert ends.tolist() == [5, 12]

    positions = {
        TensorKey.OFFSETS: t.tensor([0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_POSITIONS: t.tensor([3, 9], dtype=t.int64),
        "segment_size": 4,
        "segment_mode": "fixed_tokens",
    }
    starts, ends = segment_token_bounds(
        positions,
        shard_path=Path("shard_00000_features.safetensors.zst"),
        n_rows=2,
    )
    assert starts.tolist() == [3, 9]
    assert ends.tolist() == [7, 13]

    indices = {
        TensorKey.OFFSETS: t.tensor([0, 0, 0], dtype=t.int64),
        TensorKey.SEGMENT_INDICES: t.tensor([1, 3], dtype=t.int64),
        "segment_size": 4,
        "segment_mode": "fixed_tokens",
    }
    starts, ends = segment_token_bounds(
        indices,
        shard_path=Path("shard_00000_features.safetensors.zst"),
        n_rows=2,
    )
    assert starts.tolist() == [4, 12]
    assert ends.tolist() == [8, 16]


@pytest.mark.parametrize(
    ("data", "pattern"),
    [
        (
            {
                TensorKey.OFFSETS: t.tensor([0, 0], dtype=t.int64),
                TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([0], dtype=t.int64),
                "segment_size": 4,
            },
            "must contain both segment_token_starts",
        ),
        (
            {
                TensorKey.OFFSETS: t.tensor([0, 0], dtype=t.int64),
                TensorKey.SEGMENT_TOKEN_STARTS: t.tensor([4], dtype=t.int64),
                TensorKey.SEGMENT_TOKEN_ENDS: t.tensor([4], dtype=t.int64),
                "segment_size": 4,
            },
            "invalid segment token bounds",
        ),
        (
            {
                TensorKey.OFFSETS: t.tensor([0, 0], dtype=t.int64),
                TensorKey.SEGMENT_INDICES: t.tensor([0], dtype=t.int64),
                "segment_size": 4,
                "segment_mode": "turn_delimited_fixed_tokens",
            },
            "has no explicit segment token provenance",
        ),
    ],
)
def test_segment_token_bounds_rejects_ambiguous_or_invalid_provenance(
    data: dict,
    pattern: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=pattern):
        segment_token_bounds(
            data,
            shard_path=Path("shard_00000_features.safetensors.zst"),
            n_rows=1,
        )


def test_rank_row_stable_keys_matches_python_string_order() -> None:
    keys = [
        "shard_00000_features:shard0:doc9:tok0-4",
        "shard_00000_features:shard0:doc10:tok0-4",
        "shard_00000_features:shard0:doc1:tok8-12",
        "shard_00000_features:shard0:doc1:tok8-12",  # duplicate
        "shard_00000_features:shard0:doc1:tok12-16",
    ]
    ranks = _rank_row_stable_keys(keys)

    assert ranks.dtype == t.int32
    assert ranks[2] == ranks[3]
    order_by_rank = sorted(range(len(keys)), key=lambda i: (int(ranks[i]), i))
    order_by_key = sorted(range(len(keys)), key=lambda i: (keys[i], i))
    assert order_by_rank == order_by_key
    # doc10 < doc9 lexicographically ("1" < "9")
    assert ranks[1] < ranks[0]
    assert _rank_row_stable_keys([]).shape == (0,)


@given(
    keys=st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=1000),
            max_size=12,
        ),
        max_size=24,
    )
)
@settings(max_examples=100, deadline=None)
def test_rank_row_stable_keys_property_matches_sorted_order(
    keys: list[str],
) -> None:
    ranks = _rank_row_stable_keys(keys)

    assert ranks.shape == (len(keys),)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if keys[i] < keys[j]:
                assert ranks[i] < ranks[j]
            elif keys[i] == keys[j]:
                assert ranks[i] == ranks[j]


def test_worst_magnitudes_from_state_full_and_partial_groups() -> None:
    # state grouped (fid asc, mag desc): fid 1 has 2 rows (full at k=2),
    # fid 3 has 1 row (partial), fid 5 absent.
    state_fids = t.tensor([1, 1, 3], dtype=t.int32)
    state_mags = t.tensor([9.0, 4.0, 7.0], dtype=t.float32)

    worst = _worst_magnitudes_from_state(
        state_fids,
        state_mags,
        n_sae_features=6,
        max_positive_refs=2,
    )

    assert worst.dtype == t.float32
    assert float(worst[1]) == 4.0
    assert float(worst[3]) == float("-inf")
    assert float(worst[5]) == float("-inf")
    empty = _worst_magnitudes_from_state(
        t.zeros(0, dtype=t.int32),
        t.zeros(0, dtype=t.float32),
        n_sae_features=3,
        max_positive_refs=2,
    )
    assert bool((empty == float("-inf")).all())


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 100])
def test_vectorized_scanner_matches_brute_force_oracle(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(0, 1, 2, 3),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=chunk_size,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == _brute_force_topk_rows(shard_path, (0, 1, 2, 3), 4, 2)
    assert len(intruders) == 6
    assert not list((tmp_path / "vec_spool").iterdir())


@pytest.mark.parametrize("merge_pending_min_rows", [1, None, 10_000_000])
def test_vectorized_scanner_merge_threshold_invariance(
    tmp_path: Path,
    merge_pending_min_rows: int | None,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(0, 1, 2, 3),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=merge_pending_min_rows,
    )

    assert rows == _brute_force_topk_rows(shard_path, (0, 1, 2, 3), 4, 2)
    assert len(intruders) == 6


def test_vectorized_scanner_empty_targets_returns_intruders_only(
    tmp_path: Path,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        temp_dir=tmp_path / "spool",
    )

    assert rows == []
    assert len(intruders) == 6


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 100])
@pytest.mark.parametrize("merge_pending_min_rows", [1, None])
def test_vectorized_scanner_duplicate_key_in_topk_window_matches_true_reduction(
    tmp_path: Path,
    chunk_size: int,
    merge_pending_min_rows: int | None,
) -> None:
    # Case A: one row holds the SAME feature twice; both occurrences rank in
    # the chunk-local top-2, which makes the legacy scanners drop doc 1 at
    # chunk_size >= 3 (known, documented defect — they are deliberately NOT
    # asserted here).
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[2, 1],
            feature_ids=[1, 1, 1],
            magnitudes=[10.0, 9.0, 8.0],
        ),
    )
    expected = _brute_force_topk_rows(shard_path, (1,), 4, 2)

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=chunk_size,
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=merge_pending_min_rows,
    )

    assert rows == expected
    assert [(float(r[1]), _row_ref(r).doc_id) for r in rows] == [
        (10.0, 0),
        (8.0, 1),
    ]
    assert len(intruders) == 2


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 100])
def test_vectorized_scanner_cross_row_duplicate_key_matches_true_reduction(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    # Case B: two DIFFERENT rows share a stable_key (same doc + token
    # bounds); a third distinct-key row must survive the reduction at every
    # chunk size.
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[1, 1, 1],
            feature_ids=[1, 1, 1],
            magnitudes=[8.0, 7.0, 2.0],
            doc_ids=[7, 7, 9],
            token_starts=[0, 0, 0],
            token_ends=[4, 4, 4],
        ),
    )
    expected = _brute_force_topk_rows(shard_path, (1,), 4, 2)

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=chunk_size,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == expected
    assert [(float(r[1]), _row_ref(r).doc_id) for r in rows] == [
        (8.0, 7),
        (2.0, 9),
    ]


@st.composite
def _duplicate_capable_shard_specs(draw: DrawFn) -> dict:
    n_sae_features = draw(st.integers(min_value=1, max_value=6))
    n_rows = draw(st.integers(min_value=1, max_value=5))
    row_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    n_entries = sum(row_lengths)
    return {
        "n_sae_features": n_sae_features,
        "row_lengths": row_lengths,
        # duplicates within a row are INTENTIONAL here
        "feature_ids": draw(
            st.lists(
                st.integers(min_value=0, max_value=n_sae_features + 1),
                min_size=n_entries,
                max_size=n_entries,
            )
        ),
        "magnitudes": draw(
            st.lists(
                st.integers(min_value=0, max_value=6),
                min_size=n_entries,
                max_size=n_entries,
            )
        ),
        # duplicate stable_keys across rows are INTENTIONAL here
        "doc_ids": draw(
            st.lists(
                st.integers(min_value=0, max_value=2),
                min_size=n_rows,
                max_size=n_rows,
            )
        ),
        "target_feature_ids": tuple(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=n_sae_features - 1),
                    unique=True,
                    min_size=1,
                    max_size=n_sae_features,
                )
            )
        ),
        "max_positive_refs": draw(st.integers(min_value=1, max_value=3)),
        "chunk_size": draw(st.integers(min_value=1, max_value=n_entries)),
        "merge_pending_min_rows": draw(st.sampled_from([1, 2, None])),
    }


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(spec=_duplicate_capable_shard_specs())
def test_vectorized_scanner_property_matches_true_reduction_with_duplicates(
    tmp_path: Path,
    spec: dict,
) -> None:
    offsets = [0]
    for row_length in spec["row_lengths"]:
        offsets.append(offsets[-1] + row_length)
    n_rows = len(spec["row_lengths"])
    data = {
        TensorKey.OFFSETS: t.tensor(offsets, dtype=t.int64),
        TensorKey.FEATURE_IDS: t.tensor(spec["feature_ids"], dtype=t.int64),
        TensorKey.FEATURE_MAGS: t.tensor(spec["magnitudes"], dtype=t.float32),
        TensorKey.DOC_IDS: t.tensor(spec["doc_ids"], dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_STARTS: t.zeros(n_rows, dtype=t.int64),
        TensorKey.SEGMENT_TOKEN_ENDS: t.full((n_rows,), 4, dtype=t.int64),
        "n_docs": n_rows,
        "n_segments": n_rows,
        "layer": 22,
        "context_window": 128,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "context_independent_features": "true",
        "ignore_bos_features": "true",
    }
    shard_path = _write_feature_shard(tmp_path / "features", data)

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=spec["target_feature_ids"],
        n_sae_features=spec["n_sae_features"],
        max_positive_refs=spec["max_positive_refs"],
        chunk_size=spec["chunk_size"],
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=spec["merge_pending_min_rows"],
    )

    assert rows == _brute_force_topk_rows(
        shard_path,
        spec["target_feature_ids"],
        spec["n_sae_features"],
        spec["max_positive_refs"],
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 100])
def test_vectorized_scanner_fp16_tie_heavy_boundary_across_chunks(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    # 12 rows, one feature; magnitudes collapse onto a tiny fp16 grid so the
    # k-boundary (k=3) is contested by exact ties spanning chunk borders.
    mags = [2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 1.5, 2.0, 1.5, 2.0, 1.5, 2.0]
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[1] * 12,
            feature_ids=[1] * 12,
            magnitudes=mags,
            doc_ids=[11, 2, 9, 4, 7, 6, 5, 8, 3, 10, 1, 0],
        ),
    )

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=3,
        chunk_size=chunk_size,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == _brute_force_topk_rows(shard_path, (1,), 4, 3)
    assert len(intruders) == 12
    assert [float(row[1]) for row in rows] == [2.0, 2.0, 2.0]


def test_vectorized_scanner_string_vs_numeric_doc_order(
    tmp_path: Path,
) -> None:
    # doc10 sorts before doc9 as a string; equal magnitudes force the
    # stable-key tiebreak to decide.
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[1, 1],
            feature_ids=[1, 1],
            magnitudes=[4.0, 4.0],
            doc_ids=[9, 10],
        ),
    )

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=1,
        chunk_size=1,
        temp_dir=tmp_path / "vec_spool",
    )

    assert _row_ref(rows[0]).doc_id == 10


def test_vectorized_scanner_nan_parity_matched_vs_unmatched(
    tmp_path: Path,
) -> None:
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[2],
            feature_ids=[1, 2],
            magnitudes=[float("nan"), 5.0],
        ),
    )

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(2,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        temp_dir=tmp_path / "ok_spool",
    )
    assert [float(row[1]) for row in rows] == [5.0]

    with pytest.raises(AutoInterpDataError, match="non-finite magnitude"):
        _scan_source_index_shard_file_topk_vectorized(
            shard_path=shard_path,
            target_feature_ids=(1,),
            n_sae_features=4,
            max_positive_refs=2,
            chunk_size=2,
            temp_dir=tmp_path / "bad_spool",
        )


def test_vectorized_scanner_out_of_range_fids_dropped(tmp_path: Path) -> None:
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[3],
            feature_ids=[1, 5, 7],  # 5 and 7 out of range for n_sae=4
            magnitudes=[2.0, 9.0, 9.0],
        ),
    )

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1, 2, 3),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=1,
        temp_dir=tmp_path / "spool",
    )

    assert [(int(row[0]), float(row[1])) for row in rows] == [(1, 2.0)]


def test_vectorized_scanner_segment_positions_provenance(
    tmp_path: Path,
) -> None:
    # Production shard-0 shape: no explicit token starts/ends; bounds derive
    # from segment_positions + segment_size.
    data = _source_shard_data(
        row_lengths=[1, 1, 1],
        feature_ids=[1, 1, 1],
        magnitudes=[3.0, 5.0, 4.0],
    )
    del data[TensorKey.SEGMENT_TOKEN_STARTS]
    del data[TensorKey.SEGMENT_TOKEN_ENDS]
    data[TensorKey.SEGMENT_POSITIONS] = t.tensor([0, 4, 8], dtype=t.int64)
    shard_path = _write_feature_shard(tmp_path / "features", data)

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == _brute_force_topk_rows(shard_path, (1,), 4, 2)
    assert len(intruders) == 3
    assert rows[0][6:8] == (4, 8)


def test_vectorized_scanner_full_target_set(tmp_path: Path) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=tuple(range(4)),
        n_sae_features=4,
        max_positive_refs=3,
        chunk_size=3,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == _brute_force_topk_rows(shard_path, tuple(range(4)), 4, 3)
    assert len(intruders) == 6


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_vectorized_scanner_non_target_chunk_before_match(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    # Rows 0-1 hold only in-range NON-target feature 2; the target (feature 1)
    # first appears in rows 2-3 (distinct segments -> distinct stable_keys).
    # With chunk_size=1 the leading chunks are in-range but match no target, so
    # the scanner returns no survivors for them (pending stays empty via the
    # allowed-mask None path) and still accumulates the later matches -- the
    # None-then-match transition.
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[1, 1, 1, 1],
            feature_ids=[2, 2, 1, 1],
            magnitudes=[5.0, 6.0, 3.0, 7.0],
        ),
    )
    expected = _brute_force_topk_rows(shard_path, (1,), 4, 2)

    rows, intruders = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=chunk_size,
        temp_dir=tmp_path / "vec_spool",
    )

    assert rows == expected
    assert [(float(r[1]), _row_ref(r).doc_id) for r in rows] == [(7.0, 3), (3.0, 2)]
    assert len(intruders) == 4


def _valid_vec_chunk_kwargs() -> dict:
    # A well-formed direct call to _vectorized_chunk_survivors; individual
    # guard tests override one field each. offsets_tail=[1,2] => 2 rows,
    # payload length 2.
    return {
        "chunk": _DecodedEntryChunk(
            start=0,
            end=2,
            feature_ids=t.tensor([1, 2], dtype=t.int64),
            magnitudes=t.tensor([1.0, 2.0], dtype=t.float32),
        ),
        "offsets_tail": t.tensor([1, 2], dtype=t.int64),
        "allowed": t.ones(4, dtype=t.bool),
        "n_sae_features": 4,
        "n_rows": 2,
        "worst": t.full((4,), float("-inf"), dtype=t.float32),
        "key_ranks": t.tensor([0, 1], dtype=t.int32),
    }


def test_vectorized_chunk_survivors_baseline_and_guard_rejections() -> None:
    # Baseline: the valid call returns survivors (feature 1 and 2 both allowed).
    survivors = _vectorized_chunk_survivors(**_valid_vec_chunk_kwargs())
    assert survivors is not None
    assert survivors[0].tolist() == [1, 2]

    # Defensive guards (unreachable via the chunk iterator, which always yields
    # valid chunks) must still fail loud with the same messages as the legacy
    # entry-chunk scanner when called directly with malformed inputs.
    ndim_kwargs = _valid_vec_chunk_kwargs()
    ndim_kwargs["chunk"] = _DecodedEntryChunk(
        start=0,
        end=2,
        feature_ids=t.tensor([[1, 2]], dtype=t.int64),
        magnitudes=t.tensor([1.0, 2.0], dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="must be 1-D tensors"):
        _vectorized_chunk_survivors(**ndim_kwargs)

    numel_kwargs = _valid_vec_chunk_kwargs()
    numel_kwargs["chunk"] = _DecodedEntryChunk(
        start=0,
        end=2,
        feature_ids=t.tensor([1, 2], dtype=t.int64),
        magnitudes=t.tensor([1.0], dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="entry chunk length mismatch"):
        _vectorized_chunk_survivors(**numel_kwargs)

    neg_kwargs = _valid_vec_chunk_kwargs()
    neg_kwargs["chunk"] = _DecodedEntryChunk(
        start=-1,
        end=1,
        feature_ids=t.tensor([1], dtype=t.int64),
        magnitudes=t.tensor([1.0], dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="negative start"):
        _vectorized_chunk_survivors(**neg_kwargs)

    exceeds_kwargs = _valid_vec_chunk_kwargs()
    exceeds_kwargs["chunk"] = _DecodedEntryChunk(
        start=0,
        end=3,
        feature_ids=t.tensor([1, 1, 1], dtype=t.int64),
        magnitudes=t.tensor([1.0, 1.0, 1.0], dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="exceeds CSR payload length"):
        _vectorized_chunk_survivors(**exceeds_kwargs)

    # Row-lookup overflow: feature_ids longer than the chunk span (end-start)
    # pushes a matched entry index onto offsets_tail[-1], so searchsorted maps
    # it to row == n_rows. Only reachable with such an internal inconsistency.
    row_kwargs = _valid_vec_chunk_kwargs()
    row_kwargs["offsets_tail"] = t.tensor([1], dtype=t.int64)
    row_kwargs["n_rows"] = 1
    row_kwargs["key_ranks"] = t.tensor([0], dtype=t.int32)
    row_kwargs["chunk"] = _DecodedEntryChunk(
        start=0,
        end=1,
        feature_ids=t.tensor([1, 1], dtype=t.int64),
        magnitudes=t.tensor([1.0, 1.0], dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="row lookup exceeded"):
        _vectorized_chunk_survivors(**row_kwargs)


def test_finalize_vectorized_rows_rejects_non_finite_state_magnitude() -> None:
    # Defensive double-check: matched-entry magnitudes are NaN-guarded in the
    # chunk scanner, so a non-finite magnitude cannot reach the finalizer via
    # the scanner. Called directly with a poisoned state it must still raise.
    ref = SourceDocRef("shard_00000_features", 0, 0, 0, 4)
    scan_data = _SpooledSourceIndexShardData(
        offsets=t.tensor([0, 1], dtype=t.int64),
        n_entries=1,
        doc_ids=t.tensor([0], dtype=t.int64),
        segment_token_starts=t.tensor([0], dtype=t.int64),
        segment_token_ends=t.tensor([4], dtype=t.int64),
        stem="shard_00000_features",
        shard_index=0,
        intruders=[ref],
    )
    state = (
        t.tensor([1], dtype=t.int32),
        t.tensor([0], dtype=t.int32),
        t.tensor([float("nan")], dtype=t.float32),
        t.tensor([0], dtype=t.int32),
    )

    with pytest.raises(AutoInterpDataError, match="non-finite magnitude"):
        _finalize_vectorized_rows(state, scan_data=scan_data, row_keys=[ref.stable_key()])


def test_vectorized_scanner_worst_filter_drops_later_low_candidate(
    tmp_path: Path,
) -> None:
    # Feature 1 fills its top-2 (mags 10, 9) in the first rows; a later row
    # carries a lower mag (2) for feature 1. With merge_pending_min_rows=1 the
    # worst-magnitude filter is populated mid-scan, so that later candidate is
    # dropped by `keep.any() == False` rather than carried into pending.
    shard_path = _write_feature_shard(
        tmp_path / "features",
        _source_shard_data(
            row_lengths=[1, 1, 1],
            feature_ids=[1, 1, 1],
            magnitudes=[10.0, 9.0, 2.0],
        ),
    )
    expected = _brute_force_topk_rows(shard_path, (1,), 4, 2)

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=1,
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=1,
    )

    assert rows == expected
    assert [(float(r[1]), _row_ref(r).doc_id) for r in rows] == [(10.0, 0), (9.0, 1)]


def test_scan_shard_worker_matches_direct_scan(tmp_path: Path) -> None:
    # The spawn-safe worker + compact pack/unpack must reproduce the direct
    # scanner's (positive_rows, intruders) EXACTLY and in order, so the parallel
    # path is byte-identical to serial.
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    save_features(
        _source_shard_data(
            row_lengths=[2, 1, 3],
            feature_ids=[1, 2, 1, 3, 1, 2],
            magnitudes=[4.0, 8.0, 4.0, 1.0, 9.0, 8.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    request = replace(
        _request(tmp_path, feature_dir, shard_paths[0]),
        shard_paths=tuple(shard_paths),
    )
    request.index_path.parent.mkdir(parents=True, exist_ok=True)
    target_ids = tuple(sorted(set(request.target_feature_ids)))
    for shard_index in range(len(shard_paths)):
        direct_rows, direct_intruders = si._scan_source_index_shard_file_topk_vectorized(
            shard_path=shard_paths[shard_index],
            target_feature_ids=target_ids,
            n_sae_features=request.n_sae_features,
            max_positive_refs=request.max_positive_refs_per_feature,
            chunk_size=request.chunk_size,
            temp_dir=request.index_path.parent,
            shard_index=shard_index,
            shard_count=len(shard_paths),
        )
        result = si._scan_shard_worker(si._scan_task(request, shard_index, target_ids))
        assert result.positive_rows == direct_rows
        assert si._unpack_intruders(result) == direct_intruders


def test_scan_result_is_picklable(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)
    request.index_path.parent.mkdir(parents=True, exist_ok=True)
    task = si._scan_task(request, 0, tuple(sorted(set(request.target_feature_ids))))
    assert pickle.loads(pickle.dumps(task)) == task
    result = si._scan_shard_worker(task)
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert si._unpack_intruders(restored) == si._unpack_intruders(result)


def _dump_index_tables(index_path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(str(index_path))
    try:
        return {
            "positive_refs": conn.execute(
                "SELECT * FROM positive_refs ORDER BY feature_id, rank"
            ).fetchall(),
            "intruder_refs": conn.execute("SELECT * FROM intruder_refs ORDER BY rank").fetchall(),
        }
    finally:
        conn.close()


def _reference_build_index_sql(
    request: SourceIndexRequest,
    manifest: dict,
    *,
    progress_callback: object = None,
) -> None:
    """Reference: main's per-shard-SQLite build with a GLOBAL intruder dedup
    set (metadata/positive_work/positive_refs/intruder_refs). The byte-equality
    oracle for the per-shard intruder-set fix: production keeps this exact
    SQLite positive build but scopes the intruder set per shard, and every
    _build_both_ways test asserts the two produce identical tables — proving
    the per-shard set is output-equivalent to the global one."""
    tmp_path = request.index_path.with_name(f"{request.index_path.name}.tmp.ref.{os.getpid()}")
    tmp_path.unlink(missing_ok=True)
    conn = _connect(tmp_path)
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            "CREATE TABLE positive_work ("
            " feature_id INTEGER NOT NULL, magnitude REAL NOT NULL,"
            " stable_key TEXT NOT NULL, shard_stem TEXT NOT NULL,"
            " shard_index INTEGER NOT NULL, doc_id INTEGER NOT NULL,"
            " segment_token_start INTEGER NOT NULL, segment_token_end INTEGER NOT NULL,"
            " PRIMARY KEY (feature_id, stable_key))"
        )
        conn.execute(
            "CREATE TABLE positive_refs ("
            " feature_id INTEGER NOT NULL, rank INTEGER NOT NULL,"
            " magnitude REAL NOT NULL, stable_key TEXT NOT NULL,"
            " shard_stem TEXT NOT NULL, shard_index INTEGER NOT NULL,"
            " doc_id INTEGER NOT NULL, segment_token_start INTEGER NOT NULL,"
            " segment_token_end INTEGER NOT NULL, PRIMARY KEY (feature_id, rank))"
        )
        conn.execute(
            "CREATE TABLE intruder_refs ("
            " rank INTEGER PRIMARY KEY, stable_key TEXT NOT NULL,"
            " shard_stem TEXT NOT NULL, shard_index INTEGER NOT NULL,"
            " doc_id INTEGER NOT NULL, segment_token_start INTEGER NOT NULL,"
            " segment_token_end INTEGER NOT NULL)"
        )
        conn.execute("CREATE INDEX positive_work_feature_idx ON positive_work(feature_id)")
        conn.execute("CREATE INDEX positive_refs_feature_idx ON positive_refs(feature_id)")
        conn.commit()

        intruder_refs: list[SourceDocRef] = []
        intruder_keys: set[str] = set()
        intruder_seen = 0
        target_ids = tuple(sorted(set(request.target_feature_ids)))
        for shard_idx, shard_path in enumerate(request.shard_paths):
            positive_rows, shard_intruders = si._scan_source_index_shard_file_topk_vectorized(
                shard_path=shard_path,
                target_feature_ids=target_ids,
                n_sae_features=request.n_sae_features,
                max_positive_refs=request.max_positive_refs_per_feature,
                chunk_size=request.chunk_size,
                temp_dir=request.index_path.parent,
                shard_index=shard_idx,
                shard_count=len(request.shard_paths),
            )
            if positive_rows:
                conn.executemany(
                    "INSERT INTO positive_work (feature_id, magnitude, stable_key, "
                    "shard_stem, shard_index, doc_id, segment_token_start, "
                    "segment_token_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(feature_id, stable_key) DO UPDATE SET "
                    "magnitude=max(magnitude, excluded.magnitude)",
                    (
                        _positive_row_tuple(
                            _positive_row_from_tuple(
                                row, context="reference SQL positive row insert"
                            )
                        )
                        for row in positive_rows
                    ),
                )
                conn.commit()
            conn.execute(
                "DELETE FROM positive_work WHERE rowid IN (SELECT rowid FROM ("
                " SELECT rowid, ROW_NUMBER() OVER (PARTITION BY feature_id "
                " ORDER BY magnitude DESC, stable_key ASC) AS rn FROM positive_work"
                ") WHERE rn > ?)",
                (request.max_positive_refs_per_feature,),
            )
            conn.commit()
            for doc_ref in shard_intruders:
                key = doc_ref.stable_key()
                if key in intruder_keys:
                    continue
                intruder_keys.add(key)
                intruder_seen += 1
                if len(intruder_refs) < request.max_intruder_refs:
                    intruder_refs.append(doc_ref)
                    continue
                replacement = random.Random(f"{request.seed}:intruder:{intruder_seen}").randrange(
                    intruder_seen
                )
                if replacement < request.max_intruder_refs:
                    intruder_refs[replacement] = doc_ref
        conn.execute(
            "INSERT INTO positive_refs (feature_id, rank, magnitude, stable_key, "
            "shard_stem, shard_index, doc_id, segment_token_start, segment_token_end) "
            "SELECT feature_id, rn - 1, magnitude, stable_key, shard_stem, shard_index, "
            "doc_id, segment_token_start, segment_token_end FROM ("
            " SELECT *, ROW_NUMBER() OVER (PARTITION BY feature_id "
            " ORDER BY magnitude DESC, stable_key ASC) AS rn FROM positive_work"
            ") WHERE rn <= ? ORDER BY feature_id, rn",
            (request.max_positive_refs_per_feature,),
        )
        conn.execute("DROP TABLE positive_work")
        _insert_intruder_refs(conn, intruder_refs)
        conn.commit()
    except Exception:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp_path, request.index_path)
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        request.manifest_path,
    )


def _build_both_ways(
    tmp_path: Path,
    feature_dir: Path,
    shard_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    **request_overrides: Any,
) -> tuple[dict, dict, Path, Path]:
    """Build the same request via the SQL reference and via production, return
    (ref_tables, prod_tables, ref_manifest_path, prod_manifest_path)."""
    base = _request(tmp_path, feature_dir, shard_paths[0], **request_overrides)
    base = replace(
        base,
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
    )
    ref_req = replace(
        base,
        index_path=tmp_path / "ref" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "ref" / "source_doc_index.manifest.json",
    )
    monkeypatch.setattr(si, "_build_index", _reference_build_index_sql)
    load_or_build_source_index(ref_req)
    monkeypatch.undo()

    prod_req = replace(
        base,
        index_path=tmp_path / "prod" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "prod" / "source_doc_index.manifest.json",
    )
    load_or_build_source_index(prod_req)
    return (
        _dump_index_tables(ref_req.index_path),
        _dump_index_tables(prod_req.index_path),
        ref_req.manifest_path,
        prod_req.manifest_path,
    )


def test_build_index_matches_sql_reference_two_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    save_features(
        _source_shard_data(
            row_lengths=[2, 1, 3],
            feature_ids=[1, 2, 1, 3, 1, 2],
            magnitudes=[4.0, 8.0, 4.0, 1.0, 9.0, 8.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    ref_tables, prod_tables, ref_manifest, prod_manifest = _build_both_ways(
        tmp_path, feature_dir, shard_paths, monkeypatch, chunk_size=2
    )
    assert prod_tables == ref_tables
    assert json.loads(prod_manifest.read_text()) == json.loads(ref_manifest.read_text())


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_build_index_matches_sql_reference_fixture_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
) -> None:
    feature_dir = tmp_path / "features"
    # shard 0: base (features 0,1,2). shard 1: cross-shard mag ties on feature 1
    # (mag 3 on a different key than shard 0's), feature 3 only here.
    # shard 2: zero matched positives for the target set (only feature 7,
    # out of range for n_sae=4).
    _write_feature_shard(feature_dir)
    save_features(
        _source_shard_data(
            row_lengths=[1, 1, 1],
            feature_ids=[1, 3, 1],
            magnitudes=[3.0, 5.0, 3.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    save_features(
        _source_shard_data(
            row_lengths=[1, 1],
            feature_ids=[7, 7],
            magnitudes=[9.0, 9.0],
        ),
        feature_dir / "shard_00002_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    assert len(shard_paths) == 3
    ref_tables, prod_tables, ref_manifest, prod_manifest = _build_both_ways(
        tmp_path, feature_dir, shard_paths, monkeypatch, chunk_size=chunk_size
    )
    assert prod_tables == ref_tables
    assert json.loads(prod_manifest.read_text()) == json.loads(ref_manifest.read_text())


def test_build_index_matches_sql_reference_prune_then_reenter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # k=1: shard 0 gives feature 1 keys A(mag5, doc0) and B(mag3, doc1); B is
    # pruned. shard 1 gives key B' (doc1 again, distinct shard stem so distinct
    # key) at mag 10. The winner must be shard 1's high-mag entry in both paths.
    feature_dir = tmp_path / "features"
    save_features(
        _source_shard_data(
            row_lengths=[1, 1],
            feature_ids=[1, 1],
            magnitudes=[5.0, 3.0],
        ),
        feature_dir / "shard_00000_features.safetensors.zst",
    )
    save_features(
        _source_shard_data(
            row_lengths=[1],
            feature_ids=[1],
            magnitudes=[10.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    ref_tables, prod_tables, _, _ = _build_both_ways(
        tmp_path,
        feature_dir,
        shard_paths,
        monkeypatch,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_positive_refs_per_feature=1,
        chunk_size=2,
    )
    assert prod_tables == ref_tables
    assert [r[2] for r in prod_tables["positive_refs"]] == [10.0]


def test_build_index_matches_sql_reference_permuted_shard_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir)
    save_features(
        _source_shard_data(
            row_lengths=[2, 1],
            feature_ids=[1, 2, 1],
            magnitudes=[6.0, 7.0, 2.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    positives_by_order: list[list] = []
    for order in (shard_paths, list(reversed(shard_paths))):
        ref_tables, prod_tables, _, _ = _build_both_ways(
            tmp_path / f"o{order[0].name}",
            feature_dir,
            order,
            monkeypatch,
            chunk_size=2,
        )
        # The reference and production must always agree on the SAME shard order.
        assert prod_tables["positive_refs"] == ref_tables["positive_refs"]
        assert prod_tables["intruder_refs"] == ref_tables["intruder_refs"]
        positives_by_order.append(prod_tables["positive_refs"])
    # positive_refs are the per-feature top-k across all shards, invariant to
    # shard processing order (intruders are NOT — the reservoir depends on order).
    assert positives_by_order[0] == positives_by_order[1]


def test_build_index_reservoir_parity_when_intruders_exceed_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    for shard_i in range(3):
        save_features(
            _source_shard_data(
                row_lengths=[1] * 6,
                feature_ids=[1, 1, 1, 1, 1, 1],
                magnitudes=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                doc_ids=[shard_i * 10 + j for j in range(6)],
                token_starts=[j * 4 for j in range(6)],
                token_ends=[j * 4 + 4 for j in range(6)],
            ),
            feature_dir / f"shard_{shard_i:05d}_features.safetensors.zst",
        )
    shard_paths = glob_features(feature_dir)
    # cap intruders below the 18 unique rows to force reservoir replacement.
    ref_tables, prod_tables, _, _ = _build_both_ways(
        tmp_path,
        feature_dir,
        shard_paths,
        monkeypatch,
        target_feature_ids=(1,),
        n_sae_features=4,
        max_intruder_refs=5,
        seed=42,
        chunk_size=2,
    )
    assert prod_tables["intruder_refs"] == ref_tables["intruder_refs"]
    assert len(prod_tables["intruder_refs"]) == 5


def test_build_index_rejects_duplicate_shard_stems(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path, feature_dir, shard_path)
    request = replace(request, shard_paths=(shard_path, shard_path))
    with pytest.raises(AutoInterpDataError, match="duplicate feature shard stem"):
        load_or_build_source_index(request)
    assert not request.index_path.exists()
    assert not request.manifest_path.exists()


def test_build_index_cache_identity_with_sql_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    # Build with the SQL reference (global-intruder, v2 method), then load under
    # production (per-shard-intruder) code: the manifest matches, so the load
    # path must NOT rebuild.
    monkeypatch.setattr(si, "_build_index", _reference_build_index_sql)
    request = _request(tmp_path, feature_dir, shard_path)
    built = load_or_build_source_index(request)
    monkeypatch.undo()

    # Spy: a second call must reuse the existing index, never invoking
    # _build_index (proves the v2<->v2 cache identity is a genuine no-rebuild).
    def _fail_if_rebuilt(*args: object, **kwargs: object) -> None:
        raise AssertionError("load path unexpectedly rebuilt the index")

    monkeypatch.setattr(si, "_build_index", _fail_if_rebuilt)
    loaded = load_or_build_source_index(request)
    monkeypatch.undo()
    assert loaded == built


def test_build_index_empty_targets_intruders_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    ref_tables, prod_tables, _, _ = _build_both_ways(
        tmp_path,
        feature_dir,
        [shard_path],
        monkeypatch,
        target_feature_ids=(),
    )
    assert prod_tables == ref_tables
    assert prod_tables["positive_refs"] == []
    assert prod_tables["intruder_refs"]


def test_build_index_empty_targets_still_intruders_only(tmp_path: Path) -> None:
    # target=() -> empty positive_refs, intruders unaffected, no crash
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir, n_shards=2, rows_per_shard=6)
    base = _request(tmp_path / "shared", feature_dir, shard_paths[0], target_feature_ids=())
    request = replace(
        base,
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
    )
    si._build_index(request, si._manifest_payload(request))
    tables = _dump_index_tables(request.index_path)
    assert tables["positive_refs"] == []
    assert tables["intruder_refs"]  # intruders still collected with no targets


def test_build_index_shard_done_rows_done_tracks_live_survivors(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    save_features(
        _source_shard_data(
            row_lengths=[1, 1, 1],
            feature_ids=[1, 1, 2],
            magnitudes=[5.0, 4.0, 1.0],
        ),
        feature_dir / "shard_00000_features.safetensors.zst",
    )
    save_features(
        _source_shard_data(
            row_lengths=[1, 1, 1],
            feature_ids=[1, 2, 2],
            magnitudes=[6.0, 2.0, 3.0],
        ),
        feature_dir / "shard_00001_features.safetensors.zst",
    )
    save_features(
        _source_shard_data(
            row_lengths=[1, 1],
            feature_ids=[3, 3],
            magnitudes=[9.0, 8.0],
        ),
        feature_dir / "shard_00002_features.safetensors.zst",
    )
    shard_paths = glob_features(feature_dir)
    base = _request(
        tmp_path / "shared",
        feature_dir,
        shard_paths[0],
        target_feature_ids=(1, 2),
        max_positive_refs_per_feature=2,
        chunk_size=2,
    )
    request = replace(
        base,
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
        index_path=tmp_path / "out" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "out" / "source_doc_index.manifest.json",
    )
    events: list[SourceIndexProgressEvent] = []

    si._build_index(
        request,
        si._manifest_payload(request),
        progress_callback=events.append,
    )

    shard_done = [
        event for event in events if event.kind == SourceIndexProgressEventKind.SHARD_DONE
    ]
    assert [event.rows_done for event in shard_done] == [3, 4, 4]
    assert [event.positive_rows for event in shard_done] == [3, 3, 0]


def test_merge_intruder_dedup_is_per_shard_call() -> None:
    per_shard_refs = [
        SourceDocRef("shard", 0, doc_id, doc_id * 4, doc_id * 4 + 4) for doc_id in range(6)
    ]
    duplicated_within_shard = [ref for ref in per_shard_refs for _duplicate in range(2)]
    state = si._TopKMergeState(1)
    intruder_refs: list[SourceDocRef] = []
    intruder_seen = 0

    for _shard_call in range(3):
        intruder_seen, live_positive_rows = si._merge_shard_into_index(
            state,
            max_intruder_refs=100,
            seed=17,
            positive_rows=[],
            shard_intruders=duplicated_within_shard,
            intruder_refs=intruder_refs,
            intruder_seen=intruder_seen,
        )
        assert live_positive_rows == 0

    # Each invocation suppresses its six within-shard duplicates, then drops
    # the local key set. A leaked/global set would keep intruder_seen at six.
    assert intruder_seen == 3 * len(per_shard_refs)
    assert intruder_refs == per_shard_refs * 3


def _write_multi_shard_fixture(
    feature_dir: Path, *, n_shards: int = 3, rows_per_shard: int = 6
) -> list[Path]:
    """N feature shards with distinct docs (unique stems) — real inputs the
    spawn workers scan, with enough rows to fill/overflow the intruder reservoir.
    """
    for shard in range(n_shards):
        save_features(
            _source_shard_data(
                row_lengths=[1] * rows_per_shard,
                feature_ids=[(j % 3) + 1 for j in range(rows_per_shard)],
                magnitudes=[float((shard * 7 + j) % 11 + 1) for j in range(rows_per_shard)],
                doc_ids=[shard * 100 + j for j in range(rows_per_shard)],
                token_starts=[j * 4 for j in range(rows_per_shard)],
                token_ends=[j * 4 + 4 for j in range(rows_per_shard)],
            ),
            feature_dir / f"shard_{shard:05d}_features.safetensors.zst",
        )
    return glob_features(feature_dir)


def _build_workers_index(
    tmp_path: Path,
    feature_dir: Path,
    shard_paths: list[Path],
    scan_workers: int,
    **overrides: Any,
) -> tuple[dict[str, list[tuple]], dict]:
    """Build the index at a given scan_workers and return (tables, manifest).
    Data lives under a SHARED root so data_dir/data_fingerprint match across
    worker counts; only the index/manifest OUTPUT path varies per worker count
    (to force a rebuild rather than a cache hit), and it is absent from the
    manifest, so the manifest is comparable across worker counts."""
    base = _request(
        tmp_path / "shared",
        feature_dir,
        shard_paths[0],
        scan_workers=scan_workers,
        **overrides,
    )
    out = tmp_path / f"w{scan_workers}"
    request = replace(
        base,
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
        index_path=out / "source_doc_index.sqlite",
        manifest_path=out / "source_doc_index.manifest.json",
    )
    load_or_build_source_index(request)
    return (
        _dump_index_tables(request.index_path),
        json.loads(request.manifest_path.read_text()),
    )


def test_build_index_workers_default_suite_byte_identical(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(
        feature_dir,
        n_shards=2,
        rows_per_shard=4,
    )

    ref_tables, ref_manifest = _build_workers_index(
        tmp_path,
        feature_dir,
        shard_paths,
        1,
    )
    worker_tables, worker_manifest = _build_workers_index(
        tmp_path,
        feature_dir,
        shard_paths,
        2,
    )

    assert worker_tables["positive_refs"] == ref_tables["positive_refs"]
    assert worker_tables["intruder_refs"] == ref_tables["intruder_refs"]
    assert worker_manifest == ref_manifest


def test_build_index_parallel_workers_forward_progress_events(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(
        feature_dir,
        n_shards=2,
        rows_per_shard=8,
    )
    ref_tables, ref_manifest = _build_workers_index(
        tmp_path,
        feature_dir,
        shard_paths,
        1,
        chunk_size=2,
    )
    base = _request(
        tmp_path / "shared",
        feature_dir,
        shard_paths[0],
        scan_workers=2,
        chunk_size=2,
    )
    out = tmp_path / "w2_progress"
    request = replace(
        base,
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
        index_path=out / "source_doc_index.sqlite",
        manifest_path=out / "source_doc_index.manifest.json",
    )
    events: list[SourceIndexProgressEvent] = []

    si._build_index(
        request,
        si._manifest_payload(request),
        progress_callback=events.append,
    )

    tables = _dump_index_tables(request.index_path)
    manifest = json.loads(request.manifest_path.read_text())
    assert tables == ref_tables
    assert manifest == ref_manifest

    worker_event_indices = [
        index
        for index, event in enumerate(events)
        if event.kind
        in {
            SourceIndexProgressEventKind.SHARD_SPOOLED,
            SourceIndexProgressEventKind.CHUNK_DONE,
        }
    ]
    first_done_index = next(
        index
        for index, event in enumerate(events)
        if event.kind == SourceIndexProgressEventKind.SHARD_DONE
    )
    assert worker_event_indices
    assert min(worker_event_indices) < first_done_index

    shard_done = [
        event for event in events if event.kind == SourceIndexProgressEventKind.SHARD_DONE
    ]
    assert shard_done
    for event in shard_done:
        wait_seconds = event.wait_seconds
        merge_seconds = event.merge_seconds
        assert wait_seconds is not None
        assert merge_seconds is not None
        assert wait_seconds >= 0.0
        assert merge_seconds >= 0.0


def test_load_or_build_source_index_forwards_progress_callback(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(
        feature_dir,
        n_shards=1,
        rows_per_shard=4,
    )
    request = replace(
        _request(tmp_path, feature_dir, shard_paths[0], chunk_size=2),
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
    )
    events: list[SourceIndexProgressEvent] = []

    load_or_build_source_index(request, progress_callback=events.append)

    assert [events[0].kind, events[-1].kind] == [
        SourceIndexProgressEventKind.BUILD_START,
        SourceIndexProgressEventKind.BUILD_DONE,
    ]
    shard_done = [
        event for event in events if event.kind == SourceIndexProgressEventKind.SHARD_DONE
    ]
    assert len(shard_done) == 1
    assert shard_done[0].wait_seconds is None
    assert shard_done[0].merge_seconds is not None
    assert shard_done[0].merge_seconds >= 0.0
    summary = _summarize_progress_events(events, scan_workers=1, n_shards=1)
    assert summary["merge_sum_s"] == shard_done[0].merge_seconds
    assert request.index_path.is_file()
    assert request.manifest_path.is_file()


def test_source_index_build_benchmark_progress_summary() -> None:
    events = [
        SourceIndexProgressEvent(kind=SourceIndexProgressEventKind.BUILD_START),
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_SPOOLED,
            spool_bytes=10,
        ),
        SourceIndexProgressEvent(kind=SourceIndexProgressEventKind.CHUNK_DONE),
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_DONE,
            wait_seconds=1.0,
            merge_seconds=0.25,
            elapsed_seconds=1.25,
        ),
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_SPOOLED,
            spool_bytes=20,
        ),
        SourceIndexProgressEvent(kind=SourceIndexProgressEventKind.CHUNK_DONE),
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.SHARD_DONE,
            wait_seconds=2.0,
            merge_seconds=0.5,
            elapsed_seconds=2.5,
        ),
        SourceIndexProgressEvent(
            kind=SourceIndexProgressEventKind.BUILD_DONE,
            elapsed_seconds=4.0,
        ),
    ]

    summary = _summarize_progress_events(events, scan_workers=4, n_shards=2)

    assert summary["progress_event_count"] == len(events)
    assert summary["shard_done_event_count"] == 2
    assert summary["shard_spooled_event_count"] == 2
    assert summary["chunk_done_event_count"] == 2
    assert summary["build_progress_elapsed_s"] == 4.0
    assert summary["wait_sum_s"] == 3.0
    assert summary["wait_max_s"] == 2.0
    assert summary["wait_mean_s"] == 1.5
    assert summary["merge_sum_s"] == 0.75
    assert summary["merge_max_s"] == 0.5
    assert summary["merge_mean_s"] == 0.375
    assert summary["shard_elapsed_sum_s"] == 3.75
    assert summary["max_spool_bytes"] == 20
    assert summary["estimated_peak_concurrent_spool_bytes"] == 40


@pytest.mark.slow
@pytest.mark.parametrize("scan_workers", [1, 2, 4])
def test_build_index_workers_byte_identical(tmp_path: Path, scan_workers: int) -> None:
    # positive_refs, intruder_refs, and manifest must be identical to the serial
    # (workers=1) build for every worker count — the knob changes no output byte.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir)
    ref_tables, ref_manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, 1)
    tables, manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, scan_workers)
    assert tables == ref_tables
    assert manifest == ref_manifest


@pytest.mark.slow
def test_build_index_out_of_order_completion_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force earlier shards to finish LATER (delay ∝ shard_count - shard_index) so
    # completion order != shard order; the in-order rendezvous must still yield
    # byte-identical output.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir)
    ref_tables, _ = _build_workers_index(tmp_path, feature_dir, shard_paths, 1)
    monkeypatch.setattr(si, "_scan_shard_worker", reverse_completion_order_worker)
    tables, _ = _build_workers_index(tmp_path, feature_dir, shard_paths, 3)
    assert tables == ref_tables
    completion_times = [
        int(completion_marker_path(tmp_path / "w3", shard_index).read_text())
        for shard_index in range(len(shard_paths))
    ]
    assert completion_times[0] > completion_times[-1]


@pytest.mark.slow
def test_build_index_workers1_constructs_no_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir)
    constructed: list[int] = []
    real_pool = si.ProcessPoolExecutor

    def _spy(*args: Any, **kwargs: Any) -> Any:
        constructed.append(1)
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(si, "ProcessPoolExecutor", _spy)
    _build_workers_index(tmp_path, feature_dir, shard_paths, 1)
    assert constructed == []  # serial path builds no pool
    _build_workers_index(tmp_path, feature_dir, shard_paths, 2)
    assert constructed == [1]  # parallel path builds exactly one


@pytest.mark.slow
def test_build_index_reservoir_identical_across_workers(tmp_path: Path) -> None:
    # Small cap so the reservoir replacement path is exercised; the sampled
    # intruder_refs must match serial for every worker count (fixed seed).
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir, rows_per_shard=8)
    ref_tables, _ = _build_workers_index(tmp_path, feature_dir, shard_paths, 1, max_intruder_refs=3)
    for workers in (2, 3):
        tables, _ = _build_workers_index(
            tmp_path, feature_dir, shard_paths, workers, max_intruder_refs=3
        )
        assert tables["intruder_refs"] == ref_tables["intruder_refs"]


@pytest.mark.slow
@pytest.mark.parametrize("n_shards", [16, 64])
@pytest.mark.parametrize("scan_workers", [2, 4])
def test_build_index_workers_byte_identical_at_scale(
    tmp_path: Path, n_shards: int, scan_workers: int
) -> None:
    # Same exact output across worker counts at 16- and 64-shard scale: many
    # in-order rendezvous merges, many prune cycles, and reservoir overflow
    # accumulated across all shards must not perturb positive_refs/intruder_refs
    # or the manifest.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir, n_shards=n_shards)
    ref_tables, ref_manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, 1)
    tables, manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, scan_workers)
    assert tables == ref_tables
    assert manifest == ref_manifest


@pytest.mark.slow
@pytest.mark.parametrize("scan_workers", [1, 2, 4])
def test_build_index_vectorized_byte_identical_across_workers(
    tmp_path: Path, scan_workers: int
) -> None:
    # positive_refs + intruder_refs + manifest identical to the serial build for
    # every worker count — the vectorized merge changes no output byte.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir, n_shards=6, rows_per_shard=24)
    ref_tables, ref_manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, 1)
    tables, manifest = _build_workers_index(tmp_path, feature_dir, shard_paths, scan_workers)
    assert tables == ref_tables
    assert manifest == ref_manifest


def test_cleanup_worker_spools_removes_orphans(tmp_path: Path) -> None:
    # A hard-crashed worker can leave a .{shard}.spool.* temp file; the parent
    # sweeps exactly this build's shard spools on abort, and nothing else.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir, n_shards=2)
    request = replace(
        _request(tmp_path / "shared", feature_dir, shard_paths[0], scan_workers=2),
        shard_paths=tuple(shard_paths),
        index_path=tmp_path / "out" / "source_doc_index.sqlite",
    )
    temp_dir = request.index_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    orphans = [temp_dir / f".{p.name}.spool.deadbeef.safetensors" for p in shard_paths]
    for orphan in orphans:
        orphan.write_text("leaked spool")
    keep = temp_dir / "unrelated.parquet"
    keep.write_text("keep me")
    si._cleanup_worker_spools(request)
    assert all(not orphan.exists() for orphan in orphans)
    assert keep.exists()


@pytest.mark.slow
def test_build_index_worker_crash_aborts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A worker failure must abort the whole build loudly, leaving no index,
    # no manifest, and no orphaned spool temp files.
    feature_dir = tmp_path / "features"
    shard_paths = _write_multi_shard_fixture(feature_dir)
    monkeypatch.setattr(si, "_scan_shard_worker", fail_second_shard_worker)
    out = tmp_path / "out"
    request = replace(
        _request(tmp_path / "shared", feature_dir, shard_paths[0], scan_workers=2),
        shard_paths=tuple(shard_paths),
        feature_fingerprint=fingerprint_paths(tuple(shard_paths)),
        index_path=out / "source_doc_index.sqlite",
        manifest_path=out / "source_doc_index.manifest.json",
    )
    with pytest.raises(RuntimeError, match="test-injected worker failure"):
        load_or_build_source_index(request)
    assert not request.index_path.exists()
    assert not request.manifest_path.exists()
    assert list(out.glob(".*.spool.*")) == []


def _disk_budget_request(tmp_path: Path, feature_dir: Path) -> SourceIndexRequest:
    shard_paths = _write_multi_shard_fixture(feature_dir, n_shards=2)
    request = replace(
        _request(tmp_path / "shared", feature_dir, shard_paths[0], scan_workers=4),
        shard_paths=tuple(shard_paths),
        index_path=tmp_path / "out" / "source_doc_index.sqlite",
    )
    request.index_path.parent.mkdir(parents=True, exist_ok=True)
    return request


def test_scan_worker_disk_budget_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _disk_budget_request(tmp_path, tmp_path / "features")
    monkeypatch.setattr(
        si.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=1 << 40, used=1 << 40, free=1),
    )
    with pytest.raises(AutoInterpDataError, match="insufficient free disk"):
        si._assert_scan_worker_disk_budget(request)


def test_scan_worker_disk_budget_ok_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _disk_budget_request(tmp_path, tmp_path / "features")
    monkeypatch.setattr(
        si.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=1 << 50, used=0, free=1 << 50),
    )
    si._assert_scan_worker_disk_budget(request)  # ample space -> no raise


@pytest.mark.parametrize(
    "overrides, pattern",
    [
        ({"target_feature_ids": (9,)}, "target feature IDs out of range"),
        ({"n_sae_features": 0}, "n_sae_features must be positive"),
        ({"max_positive_refs": 0}, "max_positive_refs must be positive"),
        ({"chunk_size": 0}, "chunk_size must be positive"),
    ],
)
def test_vectorized_scanner_validation_errors(
    tmp_path: Path,
    overrides: dict,
    pattern: str,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    kwargs: dict[str, Any] = {
        "shard_path": shard_path,
        "target_feature_ids": (1,),
        "n_sae_features": 4,
        "max_positive_refs": 2,
        "chunk_size": 2,
        "temp_dir": tmp_path / "spool",
    }
    kwargs.update(overrides)

    field_name = next(iter(overrides))
    error_type = (
        AutoInterpDataError
        if field_name == "target_feature_ids"
        else BeartypeCallHintParamViolation
    )
    match = pattern if error_type is AutoInterpDataError else field_name
    with pytest.raises(error_type, match=match):
        _scan_source_index_shard_file_topk_vectorized(**kwargs)


def test_vectorized_scanner_rejects_missing_required_tensor(
    tmp_path: Path,
) -> None:
    shard_path = _write_raw_safetensors_zst(
        tmp_path / "features" / "shard_00000_features.safetensors.zst",
        tensors={
            str(TensorKey.OFFSETS): t.tensor([0, 1], dtype=t.int64),
            str(TensorKey.FEATURE_IDS): t.tensor([1], dtype=t.int64),
            str(TensorKey.DOC_IDS): t.tensor([0], dtype=t.int64),
            str(TensorKey.SEGMENT_TOKEN_STARTS): t.tensor([0], dtype=t.int64),
            str(TensorKey.SEGMENT_TOKEN_ENDS): t.tensor([4], dtype=t.int64),
        },
        metadata={"segment_size": "4", "segment_mode": "fixed_tokens"},
    )

    with pytest.raises(AutoInterpDataError, match="missing required source-index"):
        _scan_source_index_shard_file_topk_vectorized(
            shard_path=shard_path,
            target_feature_ids=(1,),
            n_sae_features=4,
            max_positive_refs=2,
            chunk_size=1,
            temp_dir=tmp_path / "spool",
        )


def test_vectorized_scanner_progress_events_match_pinned_fields(
    tmp_path: Path,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    events: list[SourceIndexProgressEvent] = []
    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(0, 1, 2, 3),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        shard_index=0,
        shard_count=1,
        temp_dir=tmp_path / "vec_spool",
        progress_callback=events.append,
    )

    assert [event.kind for event in events] == [
        SourceIndexProgressEventKind.SHARD_SPOOLED,
        SourceIndexProgressEventKind.CHUNK_DONE,
        SourceIndexProgressEventKind.CHUNK_DONE,
        SourceIndexProgressEventKind.CHUNK_DONE,
        SourceIndexProgressEventKind.CHUNK_DONE,
    ]
    spooled = events[0]
    assert spooled.shard_index == 0
    assert spooled.shard_count == 1
    assert spooled.shard_name == shard_path.name
    assert spooled.entries_total == 7
    assert spooled.rows_total == 6

    chunks = events[1:]
    assert [event.chunk_index for event in chunks] == [1, 2, 3, 4]
    assert [event.chunk_count for event in chunks] == [4, 4, 4, 4]
    assert [event.entries_done for event in chunks] == [2, 4, 6, 7]
    assert all(event.entries_total == 7 for event in chunks)
    last_chunk = chunks[-1]
    assert last_chunk.bucket_rows is not None
    assert last_chunk.bucket_rows >= len(rows)


def test_vectorized_scanner_final_bucket_rows_with_forced_merges(
    tmp_path: Path,
) -> None:
    shard_path = _write_feature_shard(tmp_path / "features")
    events: list[SourceIndexProgressEvent] = []

    rows, _ = _scan_source_index_shard_file_topk_vectorized(
        shard_path=shard_path,
        target_feature_ids=(0, 1, 2, 3),
        n_sae_features=4,
        max_positive_refs=2,
        chunk_size=2,
        temp_dir=tmp_path / "vec_spool",
        merge_pending_min_rows=1,
        progress_callback=events.append,
    )

    chunk_events = [e for e in events if e.kind == SourceIndexProgressEventKind.CHUNK_DONE]
    assert chunk_events[-1].bucket_rows == len(rows)


def test_assert_unique_shard_stems_rejects_duplicates() -> None:
    _assert_unique_shard_stems(
        (
            Path("a/shard_00000_features.safetensors.zst"),
            Path("b/shard_00001_features.safetensors.zst"),
        )
    )
    with pytest.raises(AutoInterpDataError, match="duplicate feature shard stem"):
        _assert_unique_shard_stems(
            (
                Path("a/shard_00000_features.safetensors.zst"),
                Path("b/shard_00000_features.safetensors.zst"),
            )
        )


def _ground_truth_topk(shards: list[list[tuple]], k: int) -> list[tuple]:
    # per-feature max-dedup by stable_key, then top-k by
    # (magnitude DESC, stable_key ASC); rank 0-indexed; features in ascending id
    # order. Ground truth for byte-identity.
    by_feat: dict[int, dict[str, tuple]] = {}
    for shard in shards:
        for row in shard:
            bucket = by_feat.setdefault(row[0], {})
            existing = bucket.get(row[2])
            if existing is None or float(row[1]) > float(existing[1]):
                bucket[row[2]] = row
    out: list[tuple] = []
    for fid in sorted(by_feat):
        top = sorted(by_feat[fid].values(), key=lambda r: (-r[1], r[2]))[:k]
        for rank, r in enumerate(top):
            out.append((r[0], rank, r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    return out


def _mkrow(fid: int, mag: float, shard: int, doc: int, a: int = 0, b: int = 4) -> tuple:
    stem = f"shard_{shard:05d}_features"
    key = f"{stem}:shard{shard}:doc{doc}:tok{a}-{b}"
    return (fid, float(mag), key, stem, shard, doc, a, b)


def test_topk_merge_state_matches_ground_truth_with_ties() -> None:
    # 3 shards, features {1,2}, deliberate magnitude ties within a feature
    shards = [
        [_mkrow(1, 5, 0, 0), _mkrow(1, 5, 0, 1), _mkrow(2, 9, 0, 2)],
        [_mkrow(1, 7, 1, 0), _mkrow(1, 5, 1, 1), _mkrow(2, 9, 1, 2)],
        [_mkrow(1, 6, 2, 0), _mkrow(2, 8, 2, 1)],
    ]
    state = si._TopKMergeState(2)
    for shard in shards:
        state.add_shard(shard)
    assert state.materialize() == _ground_truth_topk(shards, 2)


def test_topk_merge_state_empty_and_short_features() -> None:
    state = si._TopKMergeState(5)
    assert state.add_shard([]) == 0
    assert state.materialize() == []
    # feature with fewer than k refs -> keep all
    rows = [_mkrow(1, 3, 0, 0), _mkrow(1, 9, 0, 1)]
    state = si._TopKMergeState(5)
    state.add_shard(rows)
    assert state.materialize() == _ground_truth_topk([rows], 5)


def test_topk_merge_state_many_ties_stress() -> None:
    # low-entropy magnitudes -> many ties -> stresses the stable_key tie-break
    shards = []
    for shard in range(4):
        shards.append(
            [_mkrow((fid % 3) + 1, (fid * 3 + shard) % 5, shard, fid) for fid in range(30)]
        )
    state = si._TopKMergeState(4)
    for shard in shards:
        state.add_shard(shard)
    assert state.materialize() == _ground_truth_topk(shards, 4)


def test_topk_merge_state_duplicate_stable_key_keeps_max_magnitude() -> None:
    duplicate_low = _mkrow(1, 2.0, 0, 0)
    duplicate_high = _mkrow(1, 5.0, 0, 0)
    distinct = _mkrow(1, 4.0, 0, 1)
    shards = [[duplicate_low, distinct], [duplicate_high]]

    state = si._TopKMergeState(3)
    for shard in shards:
        state.add_shard(shard)

    materialized = state.materialize()
    assert materialized == _sqlite_oracle_positive_refs(shards, 3)
    assert [(float(row[2]), row[3]) for row in materialized] == [
        (5.0, duplicate_high[2]),
        (4.0, distinct[2]),
    ]


def test_topk_merge_state_float32_boundary_and_ties() -> None:
    one = t.tensor(1.0, dtype=t.float32)
    above = float(t.nextafter(one, t.tensor(2.0, dtype=t.float32)).item())
    below = float(t.nextafter(one, t.tensor(0.0, dtype=t.float32)).item())
    rows = [
        _mkrow(1, 1.0, 0, 2),
        _mkrow(1, above, 0, 1),
        _mkrow(1, 1.0, 0, 0),
        _mkrow(1, below, 0, 3),
    ]

    state = si._TopKMergeState(3)
    state.add_shard(rows[:2])
    state.add_shard(rows[2:])

    assert state.materialize() == _sqlite_oracle_positive_refs([rows], 3)
    assert [(float(row[2]), row[6]) for row in state.materialize()] == [
        (above, 1),
        (1.0, 0),
        (1.0, 2),
    ]


def _ref_create_positive_schema(conn: sqlite3.Connection) -> None:
    """``positive_work`` + ``positive_refs`` ``CREATE TABLE``s, copied verbatim
    from the pre-#649 production ``_create_schema`` (production dropped
    ``positive_work`` when ``_build_index`` switched to the vectorized
    ``_TopKMergeState`` merge). Frozen reference schema for
    ``_sqlite_oracle_positive_refs`` below."""
    conn.execute(
        "CREATE TABLE positive_work ("
        " feature_id INTEGER NOT NULL,"
        " magnitude REAL NOT NULL,"
        " stable_key TEXT NOT NULL,"
        " shard_stem TEXT NOT NULL,"
        " shard_index INTEGER NOT NULL,"
        " doc_id INTEGER NOT NULL,"
        " segment_token_start INTEGER NOT NULL,"
        " segment_token_end INTEGER NOT NULL,"
        " PRIMARY KEY (feature_id, stable_key)"
        ")"
    )
    conn.execute(
        "CREATE TABLE positive_refs ("
        " feature_id INTEGER NOT NULL,"
        " rank INTEGER NOT NULL,"
        " magnitude REAL NOT NULL,"
        " stable_key TEXT NOT NULL,"
        " shard_stem TEXT NOT NULL,"
        " shard_index INTEGER NOT NULL,"
        " doc_id INTEGER NOT NULL,"
        " segment_token_start INTEGER NOT NULL,"
        " segment_token_end INTEGER NOT NULL,"
        " PRIMARY KEY (feature_id, rank)"
        ")"
    )
    conn.commit()


def _ref_insert_positive_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    # Frozen reference copy of the retired production `_insert_positive_rows`.
    if not rows:
        return
    conn.executemany(
        "INSERT INTO positive_work (feature_id, magnitude, stable_key, shard_stem, "
        "shard_index, doc_id, segment_token_start, segment_token_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(feature_id, stable_key) DO UPDATE SET "
        "magnitude=max(magnitude, excluded.magnitude)",
        (
            _positive_row_tuple(
                _positive_row_from_tuple(
                    row,
                    context="source-index SQLite positive row insert",
                )
            )
            for row in rows
        ),
    )
    conn.commit()


def _ref_prune_positive_work(conn: sqlite3.Connection, max_positive_refs: int) -> None:
    # Frozen reference copy of the retired production `_prune_positive_work`.
    conn.execute(
        "DELETE FROM positive_work WHERE rowid IN ("
        " SELECT rowid FROM ("
        "   SELECT rowid, ROW_NUMBER() OVER ("
        "     PARTITION BY feature_id ORDER BY magnitude DESC, stable_key ASC"
        "   ) AS rn"
        "   FROM positive_work"
        " ) WHERE rn > ?"
        ")",
        (max_positive_refs,),
    )
    conn.commit()


def _ref_materialize_positive_refs(
    conn: sqlite3.Connection,
    max_positive_refs: int,
) -> None:
    # Frozen reference copy of the retired production `_materialize_positive_refs`.
    conn.execute("DELETE FROM positive_refs")
    conn.execute(
        "INSERT INTO positive_refs (feature_id, rank, magnitude, stable_key, "
        "shard_stem, shard_index, doc_id, segment_token_start, segment_token_end) "
        "SELECT feature_id, rn - 1, magnitude, stable_key, shard_stem, shard_index, "
        "doc_id, segment_token_start, segment_token_end FROM ("
        " SELECT *, ROW_NUMBER() OVER ("
        "   PARTITION BY feature_id ORDER BY magnitude DESC, stable_key ASC"
        " ) AS rn FROM positive_work"
        ") WHERE rn <= ? ORDER BY feature_id, rn",
        (max_positive_refs,),
    )
    conn.execute("DROP TABLE positive_work")


def _sqlite_oracle_positive_refs(shards: list[list[tuple]], k: int) -> list[tuple]:
    # Faithful oracle: reference copies of the pre-#649 production SQLite merge
    # functions (moved here when _build_index switched to _TopKMergeState).
    conn = sqlite3.connect(":memory:")
    _ref_create_positive_schema(conn)
    for shard in shards:
        _ref_insert_positive_rows(conn, shard)
        _ref_prune_positive_work(conn, k)
    _ref_materialize_positive_refs(conn, k)
    rows = conn.execute(
        "SELECT feature_id, rank, magnitude, stable_key, shard_stem, shard_index,"
        " doc_id, segment_token_start, segment_token_end FROM positive_refs "
        "ORDER BY feature_id, rank"
    ).fetchall()
    conn.close()
    return [tuple(r) for r in rows]


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("n_shards", [1, 3, 7])
def test_topk_state_byte_identical_to_sqlite(k: int, n_shards: int) -> None:
    shards = []
    for shard in range(n_shards):
        shards.append(
            [
                _mkrow(
                    (fid % 5) + 1,
                    (fid * 7 + shard * 3) % 11,
                    shard,
                    fid,
                )
                for fid in range(40)
            ]
        )
    state = si._TopKMergeState(k)
    for shard in shards:
        state.add_shard(shard)
    assert state.materialize() == _sqlite_oracle_positive_refs(shards, k)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_topk_state_prefix_after_each_shard_matches_sqlite(k: int) -> None:
    shards = [
        [
            _mkrow(
                (fid % 4) + 1,
                (fid * 5 + shard * 2) % 13,
                shard,
                fid,
            )
            for fid in range(24)
        ]
        for shard in range(5)
    ]
    state = si._TopKMergeState(k)
    for shard_index, shard in enumerate(shards):
        state.add_shard(shard)
        assert state.materialize() == _sqlite_oracle_positive_refs(
            shards[: shard_index + 1],
            k,
        )


# --- migrated from test_source_index_defensive_contracts.py (source-index fault-injection proofs) ---


def _dfc_request(root: Path) -> si.SourceIndexRequest:
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    shard_path = feature_dir / "feature_shard_00000.safetensors"
    shard_path.touch()
    return si.SourceIndexRequest(
        feature_dir=feature_dir,
        data_dir=None,
        index_path=root / "cache" / "source_doc_index.sqlite",
        manifest_path=root / "cache" / "source_doc_index.manifest.json",
        shard_paths=(shard_path,),
        parquet_paths=(),
        target_feature_ids=(1, 3),
        n_sae_features=8,
        max_positive_refs_per_feature=2,
        max_intruder_refs=2,
        seed=17,
        feature_fingerprint="feature-fingerprint",
        data_fingerprint="data-fingerprint",
        context_independent_features=True,
        ignore_bos_features=True,
        chunk_size=8,
    )


def _dfc_empty_spooled_scan_data(*, n_rows: int = 0) -> si._SpooledSourceIndexShardData:
    return si._SpooledSourceIndexShardData(
        offsets=t.zeros(n_rows + 1, dtype=t.int64),
        n_entries=0,
        doc_ids=t.zeros(n_rows, dtype=t.int64),
        segment_token_starts=t.zeros(n_rows, dtype=t.int64),
        segment_token_ends=t.ones(n_rows, dtype=t.int64),
        stem="feature_shard_00000",
        shard_index=0,
        intruders=[],
    )


def test_request_path_identity_error_is_translated_with_exact_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _dfc_request(tmp_path)
    injected = OSError("synthetic identity probe failure")

    def fail_identity_probe(*_args: Any, **_kwargs: Any) -> bool:
        raise injected

    monkeypatch.setattr(si, "paths_alias", fail_identity_probe)

    with pytest.raises(
        AutoInterpDataError,
        match="cannot verify source index SQLite/manifest path identity",
    ) as caught:
        si._validate_request(request)

    assert caught.value.__cause__ is injected


def test_manifest_unicode_read_error_is_translated_with_exact_cause(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xff")

    with pytest.raises(AutoInterpCacheError, match="cannot read source index manifest") as caught:
        load_json_object(path, context="source index manifest")

    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_source_doc_ref_rejects_invalid_token_bounds() -> None:
    with pytest.raises(AutoInterpDataError, match="invalid token bounds: 4:4"):
        si._validated_source_doc_ref(
            shard_stem="feature_shard_00000",
            shard_index=0,
            doc_id=0,
            segment_token_start=4,
            segment_token_end=4,
            context="defensive fixture",
        )


def test_sqlite_publication_failure_removes_only_the_unpublished_temp_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject at the final publication boundary, not an earlier ``os.replace``.

    Source-shard spooling also performs atomic replacements, so an unqualified
    patch of ``os.replace`` can fail before ``_build_index`` reaches its own
    publication guard and produce a false-positive cleanup test.
    """
    request = _dfc_request(tmp_path)
    real_replace = si.os.replace
    publication_attempts: list[tuple[Path, Path]] = []
    injected = OSError("synthetic final SQLite publication failure")

    monkeypatch.setattr(
        si,
        "_scan_source_index_shard_file_topk_vectorized",
        lambda **_kwargs: ([], []),
    )

    def fail_only_final_publication(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == request.index_path:
            publication_attempts.append((source_path, destination_path))
            assert source_path.name == f"{request.index_path.name}.tmp.{os.getpid()}"
            assert source_path.is_file()
            raise injected
        real_replace(source, destination)

    monkeypatch.setattr(si.os, "replace", fail_only_final_publication)

    with pytest.raises(OSError, match="final SQLite publication failure") as caught:
        si.load_or_build_source_index(request)

    assert caught.value is injected
    assert len(publication_attempts) == 1
    unpublished, destination = publication_attempts[0]
    assert destination == request.index_path
    assert not unpublished.exists()
    assert not request.index_path.exists()
    assert not request.manifest_path.exists()

    monkeypatch.setattr(si.os, "replace", real_replace)
    recovered = si.load_or_build_source_index(request)
    assert recovered.positives_by_feature == {1: [], 3: []}
    assert request.index_path.is_file()
    assert request.manifest_path.is_file()


def test_sqlite_header_read_error_and_non_database_are_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-sqlite"
    path.write_bytes(b"not a sqlite database")
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(AutoInterpCacheError, match="not a SQLite 3 database"):
            si._validate_bound_sqlite_single_file_header(file_descriptor, path=path)
    finally:
        os.close(file_descriptor)

    with pytest.raises(
        AutoInterpCacheError,
        match="cannot read bound source-index SQLite header",
    ) as caught:
        si._validate_bound_sqlite_single_file_header(file_descriptor, path=path)
    assert isinstance(caught.value.__cause__, OSError)


def test_shard_stem_without_numeric_index_is_rejected() -> None:
    with pytest.raises(AutoInterpDataError, match="does not contain a shard index"):
        si.shard_index_from_stem("feature_shard_without_ordinal")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "invalid segment_size=0"),
        ({MetadataKey.SEGMENT_SIZE: 4}, "missing segment token provenance"),
        (
            {
                MetadataKey.SEGMENT_SIZE: 4,
                TensorKey.SEGMENT_POSITIONS: t.tensor([-1], dtype=t.int64),
            },
            "negative segment token starts",
        ),
    ],
)
def test_segment_token_bound_fallbacks_fail_closed(
    tmp_path: Path,
    data: dict[Any, Any],
    message: str,
) -> None:
    with pytest.raises(AutoInterpDataError, match=message):
        si.segment_token_bounds(
            data,
            shard_path=tmp_path / "feature_shard_00000.safetensors",
            n_rows=1,
        )


def test_stable_key_rank_rejects_int32_space_overflow_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(si.t, "iinfo", lambda _dtype: SimpleNamespace(max=0))

    with pytest.raises(AutoInterpDataError, match="row count exceeds int32 rank space"):
        si._rank_row_stable_keys(["one-row-is-over-the-injected-limit"])


def test_vectorized_scanner_rejects_int32_feature_space_before_row_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_path = tmp_path / "feature_shard_00000.safetensors"
    shard_path.touch()
    scan_data = _dfc_empty_spooled_scan_data()

    @contextmanager
    def fake_spooled_shard(*_args: Any, **_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(temp_path=None)

    monkeypatch.setattr(si, "spooled_feature_shard", fake_spooled_shard)
    monkeypatch.setattr(
        si,
        "_load_spooled_source_index_shard_data",
        lambda *_args, **_kwargs: scan_data,
    )
    monkeypatch.setattr(si.t, "iinfo", lambda _dtype: SimpleNamespace(max=0))

    with pytest.raises(AutoInterpDataError, match="exceeds int32 feature space"):
        si._scan_source_index_shard_file_topk_vectorized(
            shard_path=shard_path,
            target_feature_ids=(0,),
            n_sae_features=1,
            max_positive_refs=1,
            chunk_size=1,
        )


def test_decode_span_rejects_nonmonotonic_local_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = iter((t.tensor(0), t.tensor(1)))
    monkeypatch.setattr(si.t, "searchsorted", lambda *_args, **_kwargs: next(search_results))

    with pytest.raises(AutoInterpDataError, match="non-monotonic local offsets"):
        si._decode_span_for_entry_chunk(
            t.tensor([0, 2, 1], dtype=t.int64),
            t.tensor([2, 1], dtype=t.int64),
            chunk_start=0,
            chunk_end=1,
        )


def test_spooled_chunk_iterator_rejects_decoded_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneEntryShard:
        def get_tensor_slice(
            self,
            key: TensorKey,
            _start: int,
            _end: int,
        ) -> t.Tensor:
            if key == TensorKey.FEATURE_MAGS:
                return t.tensor([1.0], dtype=t.float32)
            return t.tensor([0], dtype=t.int64)

    scan_data = replace(
        _dfc_empty_spooled_scan_data(n_rows=1),
        offsets=t.tensor([0, 1], dtype=t.int64),
        n_entries=1,
    )
    monkeypatch.setattr(
        si,
        "_delta_decode",
        lambda *_args, **_kwargs: t.zeros(0, dtype=t.int64),
    )

    with pytest.raises(AutoInterpDataError, match="decoded source-index chunk length mismatch"):
        list(si._iter_spooled_entry_chunks(OneEntryShard(), scan_data, chunk_size=1))


# --- migrated from test_source_index_process_sharing.py (single-actor
# source-index cache/lock contracts: these exercise production code that
# survives single-process operation; the file's genuinely multi-process
# tests were correctly deleted alongside it) ---


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(full: SourceIndexData, feature_ids: tuple[int, ...]) -> SourceIndexData:
    return SourceIndexData(
        positives_by_feature={
            feature_id: full.positives_by_feature[feature_id] for feature_id in feature_ids
        },
        intruder_refs=full.intruder_refs,
    )


def test_request_rejects_same_resolved_sqlite_and_manifest_path(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)
    same_path = replace(request, manifest_path=request.index_path)

    with pytest.raises(
        AutoInterpDataError,
        match="SQLite and manifest paths must resolve to distinct files",
    ):
        load_or_build_source_index(same_path)

    assert not same_path.index_path.exists()

    request.index_path.parent.mkdir(parents=True)
    manifest_alias = request.index_path.parent / "manifest-alias.json"
    manifest_alias.symlink_to(request.index_path)
    symlink_alias = replace(request, manifest_path=manifest_alias)

    with pytest.raises(
        AutoInterpDataError,
        match="SQLite and manifest paths must resolve to distinct files",
    ):
        validate_source_index_cache(symlink_alias)

    assert manifest_alias.is_symlink()
    assert not request.index_path.exists()


def test_validate_cache_returns_canonical_resolved_storage_request(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(canonical_dir, target_is_directory=True)
    request = replace(
        _request(tmp_path / "request", feature_dir, shard_path),
        index_path=alias_dir / "source.sqlite",
        manifest_path=alias_dir / "source.manifest.json",
        load_target_feature_ids=(1,),
    )

    resolved = validate_source_index_cache(request)

    assert request.index_path.parent == alias_dir
    assert request.manifest_path.parent == alias_dir
    assert resolved.index_path == (canonical_dir / "source.sqlite").resolve()
    assert resolved.manifest_path == (canonical_dir / "source.manifest.json").resolve()
    assert resolved.load_target_feature_ids == (1,)


def test_read_only_connection_never_creates_and_rejects_writes(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing" / "source_doc_index.sqlite"
    missing.parent.mkdir(parents=True)

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        si._connect_read_only(missing)
    assert not missing.exists()

    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)
    load_or_build_source_index(request)
    sqlite_sha256 = _sha256(request.index_path)

    conn = si._connect_read_only(request.index_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM metadata").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO metadata (key, value) VALUES ('x', 'y')")
    finally:
        conn.close()
    assert _sha256(request.index_path) == sqlite_sha256


def test_validate_cache_uses_read_only_validation_and_force_shortcut(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)

    assert validate_source_index_cache(request) == request
    assert not request.index_path.exists()
    assert not request.manifest_path.exists()

    load_or_build_source_index(request)
    manifest_bytes = request.manifest_path.read_bytes()
    sqlite_sha256 = _sha256(request.index_path)
    assert validate_source_index_cache(request) == request
    assert request.manifest_path.read_bytes() == manifest_bytes
    assert _sha256(request.index_path) == sqlite_sha256

    fresh_force = replace(
        request,
        index_path=tmp_path / "force" / "source_doc_index.sqlite",
        manifest_path=tmp_path / "force" / "source_doc_index.manifest.json",
        force_rebuild=True,
    )
    assert validate_source_index_cache(fresh_force) == fresh_force

    conn = sqlite3.connect(request.index_path)
    try:
        conn.execute("UPDATE metadata SET value = '-1' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(AutoInterpCacheError, match="schema_version=-1"):
        validate_source_index_cache(request)


def test_keyboard_interrupt_during_scan_cleans_temp_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)

    def _interrupt_scan(**_kwargs: Any) -> tuple[list[tuple], list[SourceDocRef]]:
        raise KeyboardInterrupt("injected source-index interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(si, "_scan_source_index_shard_file_topk_vectorized", _interrupt_scan)
        with pytest.raises(KeyboardInterrupt, match="source-index interrupt"):
            load_or_build_source_index(request)

    assert not request.index_path.exists()
    assert not request.manifest_path.exists()
    assert not list(request.index_path.parent.glob(f"{request.index_path.name}.tmp.*"))
    assert not list(request.index_path.parent.glob(".*.spool.*"))

    recovered = load_or_build_source_index(request)
    assert recovered.positives_by_feature[1]
    assert request.index_path.is_file()
    assert request.manifest_path.is_file()


def test_completed_cache_remains_loadable_from_read_only_directory(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)
    full = load_or_build_source_index(request)

    request.index_path.chmod(0o444)
    request.manifest_path.chmod(0o444)
    request.index_path.parent.chmod(0o555)
    try:
        assert validate_source_index_cache(request) == request
        assert load_or_build_source_index(
            replace(request, load_target_feature_ids=(1, 3))
        ) == _project(full, (1, 3))
    finally:
        request.index_path.parent.chmod(0o755)
        request.index_path.chmod(0o644)
        request.manifest_path.chmod(0o644)


def test_manifest_publication_error_leaves_partial_pair_and_force_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    shard_path = _write_feature_shard(feature_dir)
    request = _request(tmp_path / "index", feature_dir, shard_path)

    def _fail_manifest_publication(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected manifest publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(si, "atomic_write_text", _fail_manifest_publication)
        with pytest.raises(RuntimeError, match="manifest publication failure"):
            load_or_build_source_index(request)

    assert request.index_path.is_file()
    assert not request.manifest_path.exists()
    assert not list(request.index_path.parent.glob(f"{request.index_path.name}.tmp.*"))
    with pytest.raises(AutoInterpCacheError, match="requires both SQLite and manifest"):
        load_or_build_source_index(request)

    recovered = load_or_build_source_index(replace(request, force_rebuild=True))
    assert recovered.positives_by_feature[1]
    assert request.index_path.is_file()
    assert request.manifest_path.is_file()
