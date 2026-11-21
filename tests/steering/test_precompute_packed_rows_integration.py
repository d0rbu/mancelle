"""Integration tests for precompute_packed_rows.

Tests the full precomputation pipeline: document iteration → best-fit packing →
CSR conversion → shard writing. Verifies data integrity at each step and
ensures the output is compatible with the precomputed_data_loader.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors.torch import save_file as safetensors_save_file

import scripts.preprocess.precompute_packed_rows as packed_rows_module
import steering.overlay_io as _oio
from errors import PackedRowInvariantError
from scripts.preprocess.precompute_packed_rows import (
    _seg_features_to_packed,
    precompute_packed_rows,
)
from scripts.preprocess.precompute_packed_rows import (
    run as run_packed_rows,
)
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import (
    CorruptFeatureShardError,
    TensorKey,
    glob_features,
    load_features,
    save_features,
)
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    OverlayKind,
    compute_base_shard_content_hash,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_packed_row_shard,
    reconstruct_batch,
)
from steering.segmentation import compute_tokenizer_checksum, token_to_char_offsets
from steering.stats_io import save_percentile_checksum
from tests.steering._helpers import save_raw_shard


def _make_identity_selection(shard: PackedRowShard) -> SelectionShard:
    """Create a SelectionShard that selects all features from the base shard."""
    return SelectionShard(
        sel_offsets=shard.seg_feat_offsets.to(t.int32),
        sel_fids=shard.feature_ids,
        sel_buckets=t.ones(len(shard.feature_ids), dtype=t.uint8),
        sel_pcts=shard.feature_pcts,
        neg_offsets=t.zeros(shard.total_segments + 1, dtype=t.int32),
        neg_fids=t.tensor([], dtype=t.uint16),
    )

# ---------------------------------------------------------------------------
# Test tokenizer
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Deterministic tokenizer: each word → one token."""

    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            parts.append(f"t{tid:03d} " if tid != 0 else "<bos>")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEGMENT_SIZE = 8
N_SAE = 200
N_DOCS = 15
N_FEATS_PER_SEG = 4


def _make_precompute_env(tmp_path: Path, n_shards: int = 2) -> dict:
    """Create a complete environment for testing precompute_packed_rows."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    tok = SimpleTokenizer()
    bos = tok.get_bos_token_id()

    all_parquet_paths = []
    for shard_idx in range(n_shards):
        stem = f"shard_{shard_idx:05d}"

        # Create varied-length documents
        texts = []
        for doc_idx in range(N_DOCS):
            n_words = 10 + doc_idx * 2 + shard_idx * 3
            texts.append(" ".join(f"word{w}" for w in range(n_words)))

        pq_path = data_dir / f"{stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))
        all_parquet_paths.append(str(pq_path))

        # Build feature shard
        all_fids, all_mags = [], []
        offsets = [0]
        doc_ids, seg_indices_list, seg_positions = [], [], []
        seg_char_starts, seg_char_ends = [], []
        seg_token_starts, seg_token_ends = [], []

        for doc_idx in range(N_DOCS):
            text = texts[doc_idx]
            text_tokens = tok.encode(text, prepend=bos)
            char_offsets = token_to_char_offsets(tok, text_tokens)
            n_tokens = len(text_tokens)
            n_segs = max(1, (n_tokens + SEGMENT_SIZE - 1) // SEGMENT_SIZE)

            for seg_idx in range(n_segs):
                doc_ids.append(doc_idx)
                seg_indices_list.append(seg_idx)
                seg_positions.append(seg_idx * SEGMENT_SIZE)
                tok_start = seg_idx * SEGMENT_SIZE
                tok_end = min((seg_idx + 1) * SEGMENT_SIZE, n_tokens)
                c_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
                c_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]
                seg_char_starts.append(c_start)
                seg_char_ends.append(c_end)
                seg_token_starts.append(tok_start)
                seg_token_ends.append(tok_end)

                for f in range(N_FEATS_PER_SEG):
                    fid = (doc_idx * 10 + seg_idx * 5 + f + shard_idx * 7) % N_SAE
                    all_fids.append(fid)
                    all_mags.append(0.1 * (f + 1) + 0.01 * doc_idx)
                offsets.append(len(all_fids))

        total_segs = len(doc_ids)
        shard_data = {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.tensor(all_fids, dtype=t.int32),
            "feature_mags": t.tensor(all_mags, dtype=t.float32),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(seg_indices_list, dtype=t.int32),
            "segment_positions": t.tensor(seg_positions, dtype=t.int32),
            "doc_token_counts": t.tensor(
                [len(tok.encode(texts[i], prepend=bos)) for i in range(N_DOCS)],
                dtype=t.int32,
            ),
            "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
            "n_docs": N_DOCS,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": SEGMENT_SIZE,
        }
        save_features(shard_data, feature_dir / f"{stem}_features")

    # Create percentile overlay
    pct_checksum = "pct_precompute_test" + "0" * 44
    for shard_idx in range(n_shards):
        stem = f"shard_{shard_idx:05d}"
        shard_data = load_features(
            feature_dir / f"{stem}_features.safetensors.zst"
        )
        n_feats = len(shard_data["feature_ids"])
        pct_data = t.randint(0, 255, (n_feats,), dtype=t.uint8)

        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, pct_checksum)
        if shard_idx == 0:
            write_manifest(pct_dir, pct_checksum, OverlayKind.PERCENTILE)
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: pct_data},
            overlay_shard_path(pct_dir, stem),
        )

    # Create token boundary overlay
    tok_checksum = "tok_precompute_test" + "0" * 45
    for shard_idx in range(n_shards):
        stem = f"shard_{shard_idx:05d}"
        shard_data = load_features(
            feature_dir / f"{stem}_features.safetensors.zst"
        )

        # Recompute token boundaries for this shard
        n_segs_shard = int(shard_data["n_segments"])
        tok_starts_shard = []
        tok_ends_shard = []
        doc_ids_shard = shard_data["doc_ids"]
        seg_pos_shard = shard_data["segment_positions"]

        texts_shard = []
        for doc_idx in range(N_DOCS):
            n_words = 10 + doc_idx * 2 + shard_idx * 3
            texts_shard.append(" ".join(f"word{w}" for w in range(n_words)))

        for seg_global in range(n_segs_shard):
            doc_idx = int(doc_ids_shard[seg_global].item())
            text = texts_shard[doc_idx]
            text_tokens = tok.encode(text, prepend=bos)
            n_tokens = len(text_tokens)
            seg_pos = int(seg_pos_shard[seg_global].item())
            tok_start = seg_pos
            tok_end = min(seg_pos + SEGMENT_SIZE, n_tokens)
            tok_starts_shard.append(tok_start)
            tok_ends_shard.append(tok_end)

        tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tok_checksum)
        if shard_idx == 0:
            write_manifest(tok_dir, tok_checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.tensor(tok_starts_shard, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.tensor(tok_ends_shard, dtype=t.int32),
            },
            overlay_shard_path(tok_dir, stem),
        )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "parquet_paths": all_parquet_paths,
        "pct_checksum": pct_checksum,
        "tok_checksum": tok_checksum,
        "n_shards": n_shards,
    }


def _make_feature_index(env: dict) -> FeatureIndex:
    return FeatureIndex(
        env["feature_dir"],
        percentile_checksum=env["pct_checksum"],
        tokenizer_checksum=env["tok_checksum"],
    )


def _run_precompute(
    env: dict,
    output_dir: Path,
    *,
    rows_per_shard: int = 3,
    resume: bool = False,
    resume_from_shard=None,
):
    output_dir.mkdir(exist_ok=True)
    return precompute_packed_rows(
        parquet_paths=env["parquet_paths"],
        tokenizer=SimpleTokenizer(),
        feature_index=_make_feature_index(env),
        context_len=32,
        output_dir=output_dir,
        buffer_size=20,
        rows_per_shard=rows_per_shard,
        resume=resume,
        resume_from_shard=resume_from_shard,
    )


def _shard_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in sorted(output_dir.glob("shard_*.safetensors.zst"))
    }


def _copy_shards(src_dir: Path, dst_dir: Path, shard_indices: list[int]) -> None:
    """Copy COMPLETE shards (packed + provenance sidecar + _dims companion,
    when present) for the given indices — i.e. simulate a partial output dir
    whose listed shards each finished their full flush. The merged packer writes
    the provenance sidecar in the SAME worker as the packed shard, so a
    faithfully-simulated crash/gap carries it; without it, resume would treat the
    prefix as incomplete (missing provenance) and re-pack from shard 0."""
    dst_dir.mkdir(exist_ok=True)
    for idx in shard_indices:
        name = f"shard_{idx:05d}.safetensors.zst"
        (dst_dir / name).write_bytes((src_dir / name).read_bytes())
        prov_src = src_dir / "provenance" / name
        if prov_src.exists():
            (dst_dir / "provenance").mkdir(exist_ok=True)
            (dst_dir / "provenance" / name).write_bytes(prov_src.read_bytes())
        dims_src = src_dir / "_dims" / f"shard_{idx:05d}.json"
        if dims_src.exists():
            (dst_dir / "_dims").mkdir(exist_ok=True)
            (dst_dir / "_dims" / f"shard_{idx:05d}.json").write_text(dims_src.read_text())


# ======================================================================
# 1. precompute_packed_rows function tests
# ======================================================================


class TestPrecomputePackedRows:
    """Test the core precompute_packed_rows function."""

    def test_produces_shard_files(self, tmp_path):
        """precompute_packed_rows writes at least one shard file."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        total_rows, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        assert total_rows > 0
        shard_files = list(output_dir.glob("shard_*.safetensors.zst"))
        assert len(shard_files) > 0

    @pytest.mark.parametrize(
        ("n_shards", "buffer_size"),
        [(1, 1000), (2, 8)],
    )
    def test_exhaustion_drains_every_loaded_document(
        self,
        tmp_path,
        monkeypatch,
        n_shards,
        buffer_size,
    ):
        """End-of-input drains both a never-filled buffer and a final tail."""
        env = _make_precompute_env(tmp_path, n_shards=n_shards)
        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        packed_documents: set[tuple[int, int]] = set()
        original_pack = packed_rows_module._pack_row_with_provenance

        def recording_pack(doc_buffer, row_capacity):
            result = original_pack(doc_buffer, row_capacity)
            packed_documents.update(result[2])
            return result

        monkeypatch.setattr(
            packed_rows_module,
            "_pack_row_with_provenance",
            recording_pack,
        )
        total_rows, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=SimpleTokenizer(),
            feature_index=_make_feature_index(env),
            context_len=32,
            output_dir=output_dir,
            buffer_size=buffer_size,
            rows_per_shard=3,
        )

        assert total_rows > 0
        assert packed_documents == {
            (shard_idx, doc_idx)
            for shard_idx in range(n_shards)
            for doc_idx in range(N_DOCS)
        }

    def test_empty_input_fails_before_writing_an_empty_artifact(self, tmp_path):
        env = _make_precompute_env(tmp_path, n_shards=1)
        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        with pytest.raises(
            PackedRowInvariantError,
            match="read zero input documents",
        ):
            precompute_packed_rows(
                parquet_paths=[],
                tokenizer=SimpleTokenizer(),
                feature_index=_make_feature_index(env),
                context_len=32,
                output_dir=output_dir,
                buffer_size=1000,
                rows_per_shard=3,
            )

        assert not glob_overlay_shards(output_dir)

    def test_shard_is_loadable(self, tmp_path):
        """Written shards can be loaded back as PackedRowShard."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard_files = sorted(output_dir.glob("shard_*.safetensors.zst"))
        assert len(shard_files) > 0

        shard = load_packed_row_shard(shard_files[0])
        assert shard.n_rows > 0
        assert shard.context_len == 32

    def test_token_content_preserved(self, tmp_path):
        """Tokens in packed rows are non-zero (contain actual token data)."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        # Token data should not be all zeros (BOS is 0 but other tokens > 0)
        assert shard.tokens.sum().item() > 0

    def test_csr_invariants_hold(self, tmp_path):
        """CSR format invariants: offsets monotonic, lengths consistent."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])

        # Row segment offsets: monotonically non-decreasing
        for i in range(len(shard.row_seg_offsets) - 1):
            assert shard.row_seg_offsets[i] <= shard.row_seg_offsets[i + 1]

        # First offset is 0, last matches total segments
        assert shard.row_seg_offsets[0].item() == 0
        assert shard.row_seg_offsets[-1].item() == shard.total_segments

        # Feature offsets: monotonically non-decreasing
        for i in range(len(shard.seg_feat_offsets) - 1):
            assert shard.seg_feat_offsets[i] <= shard.seg_feat_offsets[i + 1]

        assert shard.seg_feat_offsets[0].item() == 0
        assert shard.seg_feat_offsets[-1].item() == shard.total_features

        # Feature array lengths match
        assert len(shard.feature_ids) == shard.total_features
        assert len(shard.feature_pcts) == shard.total_features

    def test_features_present(self, tmp_path):
        """Packed rows contain actual feature data (not all empty)."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        assert shard.total_features > 0, "No features in packed shard"

    def test_segment_boundaries_valid(self, tmp_path):
        """Segment token start/end positions are within context length."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        T = 32
        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=T,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])

        # All segment boundaries within [0, T] (cast to int32 for comparison support)
        starts = shard.seg_token_start.to(t.int32)
        ends = shard.seg_token_end.to(t.int32)
        assert (starts >= 0).all()
        assert (ends <= T).all()
        # Start <= end
        assert (starts <= ends).all()

    def test_reconstruct_produces_valid_batch(self, tmp_path):
        """reconstruct_batch on precomputed shards produces correctly shaped tensors."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        T = 32
        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=T,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        B = min(4, shard.n_rows)
        rb = reconstruct_batch(shard, list(range(B)), _make_identity_selection(shard), 10)

        assert rb.inputs.shape == (B, T)
        assert rb.targets.shape == (B, T)
        assert rb.seg_map.shape == (B, T)
        assert rb.enc_fids.shape[0] == B
        assert rb.enc_pcts.shape[0] == B
        assert rb.enc_mask.shape[0] == B

        # inputs/targets are from T+1 token sequence
        assert rb.inputs.dtype == t.int64
        assert rb.targets.dtype == t.int64

        # Percentiles in [0, 1] for valid features
        valid_pcts = rb.enc_pcts[rb.enc_mask]
        if len(valid_pcts) > 0:
            assert (valid_pcts >= 0).all()
            assert (valid_pcts <= 1).all()


# ======================================================================
# 2. Resume logic
# ======================================================================


class TestPrecomputeResume:
    """Test the resume capability of precompute_packed_rows."""

    def test_resume_skips_existing_shards(self, tmp_path):
        """Running with resume=True skips shards that already exist."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        # First run
        total1, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=10,
        )

        n_shards1 = len(list(output_dir.glob("shard_*.safetensors.zst")))
        assert n_shards1 > 0

        # Second run with resume
        total2, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=10,
            resume=True,
        )

        n_shards2 = len(list(output_dir.glob("shard_*.safetensors.zst")))
        # Should not create additional shards
        assert n_shards2 == n_shards1

    def test_resume_from_shard_matches_full_run_bytes(self, tmp_path):
        """Explicit resume reproduces the same shard bytes as an uninterrupted run."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 3

        full_dir = tmp_path / "packed_full"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        full_bytes = _shard_bytes(full_dir)
        assert len(full_bytes) >= 6

        explicit_dir = tmp_path / "packed_explicit_resume"
        _run_precompute(
            env,
            explicit_dir,
            rows_per_shard=rows_per_shard,
            resume=True,
            resume_from_shard=3,
        )
        explicit_bytes = _shard_bytes(explicit_dir)

        expected_names = sorted(name for name in full_bytes if name >= "shard_00003")
        assert sorted(explicit_bytes) == expected_names
        for name in expected_names:
            assert explicit_bytes[name] == full_bytes[name]

    def test_resume_from_shard_does_not_require_resume_flag(self, tmp_path):
        """Setting resume_from_shard is sufficient to select explicit replay."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 3

        full_dir = tmp_path / "packed_full"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        full_bytes = _shard_bytes(full_dir)

        explicit_dir = tmp_path / "packed_explicit_without_resume_flag"
        _run_precompute(
            env,
            explicit_dir,
            rows_per_shard=rows_per_shard,
            resume=False,
            resume_from_shard=3,
        )

        explicit_bytes = _shard_bytes(explicit_dir)
        expected_names = sorted(name for name in full_bytes if name >= "shard_00003")
        assert sorted(explicit_bytes) == expected_names
        for name in expected_names:
            assert explicit_bytes[name] == full_bytes[name]

    def test_resume_from_shard_does_not_glob_output_shards(self, tmp_path, monkeypatch):
        """Explicit resume must not inspect local shard count."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        tok = SimpleTokenizer()
        idx = _make_feature_index(env)
        output_dir = tmp_path / "packed_no_glob"
        output_dir.mkdir()

        original_glob = Path.glob

        def fail_on_output_shard_count(path, pattern):
            if path == output_dir and pattern == "shard_*":
                raise AssertionError("explicit resume_from_shard must not glob shard count")
            return original_glob(path, pattern)

        monkeypatch.setattr(Path, "glob", fail_on_output_shard_count)

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=3,
            resume=True,
            resume_from_shard=0,
        )

    def test_existing_len_based_resume_still_matches_full_run_bytes(self, tmp_path):
        """The historical consecutive-shard resume path remains byte-identical."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 3

        full_dir = tmp_path / "packed_full"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        full_bytes = _shard_bytes(full_dir)
        assert len(full_bytes) >= 6

        resume_dir = tmp_path / "packed_len_resume"
        _copy_shards(full_dir, resume_dir, [0, 1, 2])
        _run_precompute(
            env,
            resume_dir,
            rows_per_shard=rows_per_shard,
            resume=True,
            resume_from_shard=None,
        )

        assert _shard_bytes(resume_dir) == full_bytes

    def test_resume_from_shard_none_is_fresh_start_regression(self, tmp_path):
        """Explicit None takes the original no-resume code path."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 3

        omitted_dir = tmp_path / "packed_omitted"
        explicit_none_dir = tmp_path / "packed_explicit_none"

        _run_precompute(env, omitted_dir, rows_per_shard=rows_per_shard)
        _run_precompute(
            env,
            explicit_none_dir,
            rows_per_shard=rows_per_shard,
            resume_from_shard=None,
        )

        assert _shard_bytes(explicit_none_dir) == _shard_bytes(omitted_dir)

    def test_resume_from_shard_ignores_present_file_count_with_gaps(self, tmp_path):
        """Sparse local files do not affect the explicit start shard."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 2

        full_dir = tmp_path / "packed_full"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        full_bytes = _shard_bytes(full_dir)
        assert len(full_bytes) >= 9

        sparse_dir = tmp_path / "packed_sparse"
        _copy_shards(full_dir, sparse_dir, [3, 4, 5])
        before_sparse_bytes = _shard_bytes(sparse_dir)

        _run_precompute(
            env,
            sparse_dir,
            rows_per_shard=rows_per_shard,
            resume=True,
            resume_from_shard=6,
        )
        sparse_bytes = _shard_bytes(sparse_dir)

        assert (
            sparse_bytes["shard_00003.safetensors.zst"]
            == before_sparse_bytes["shard_00003.safetensors.zst"]
        )
        assert (
            sparse_bytes["shard_00004.safetensors.zst"]
            == before_sparse_bytes["shard_00004.safetensors.zst"]
        )
        assert (
            sparse_bytes["shard_00005.safetensors.zst"]
            == before_sparse_bytes["shard_00005.safetensors.zst"]
        )

        expected_names = [name for name in sorted(full_bytes) if name >= "shard_00006"]
        for name in expected_names:
            assert sparse_bytes[name] == full_bytes[name]

    def test_resume_from_shard_overwrites_existing_shards_at_start(self, tmp_path):
        """Existing local shards at or after the explicit start are regenerated."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 2

        full_dir = tmp_path / "packed_full"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        full_bytes = _shard_bytes(full_dir)
        assert len(full_bytes) >= 9

        resume_dir = tmp_path / "packed_overwrite"
        _copy_shards(full_dir, resume_dir, [3, 4, 5])
        corrupt_path = resume_dir / "shard_00006.safetensors.zst"
        corrupt_path.write_bytes(b"corrupt pre-existing shard")

        _run_precompute(
            env,
            resume_dir,
            rows_per_shard=rows_per_shard,
            resume=True,
            resume_from_shard=6,
        )

        resume_bytes = _shard_bytes(resume_dir)
        assert (
            resume_bytes["shard_00006.safetensors.zst"]
            == full_bytes["shard_00006.safetensors.zst"]
        )
        assert resume_bytes["shard_00006.safetensors.zst"] != b"corrupt pre-existing shard"
        for name in sorted(full_bytes):
            if name >= "shard_00006":
                assert resume_bytes[name] == full_bytes[name]

    @pytest.mark.parametrize(
        "invalid_resume_from_shard",
        [-1, True, False, 1.0, "1", math.nan],
    )
    def test_resume_from_shard_validation_rejects_invalid_values(
        self,
        tmp_path,
        invalid_resume_from_shard,
    ):
        """The explicit resume point must be a real non-negative int."""
        env = _make_precompute_env(tmp_path, n_shards=2)

        with pytest.raises(ValueError, match="resume_from_shard"):
            _run_precompute(
                env,
                tmp_path / "packed_invalid",
                rows_per_shard=3,
                resume=True,
                resume_from_shard=invalid_resume_from_shard,
            )

    def test_resume_from_shard_zero_is_valid_fresh_start(self, tmp_path):
        """resume_from_shard=0 writes the same bytes as a fresh run."""
        env = _make_precompute_env(tmp_path, n_shards=4)
        rows_per_shard = 3

        full_dir = tmp_path / "packed_full"
        zero_dir = tmp_path / "packed_zero"
        _run_precompute(env, full_dir, rows_per_shard=rows_per_shard)
        _run_precompute(
            env,
            zero_dir,
            rows_per_shard=rows_per_shard,
            resume=True,
            resume_from_shard=0,
        )

        assert _shard_bytes(zero_dir) == _shard_bytes(full_dir)

    def test_resume_from_shard_beyond_data_returns_gracefully(self, tmp_path):
        """Exhausting data during explicit replay-skip warns and returns."""
        env = _make_precompute_env(tmp_path, n_shards=2)

        total_rows, max_n_seg, max_seg_size = _run_precompute(
            env,
            tmp_path / "packed_exhausted",
            rows_per_shard=3,
            resume=True,
            resume_from_shard=10_000,
        )

        assert total_rows > 0
        assert max_n_seg == 0
        assert max_seg_size == 0
        assert _shard_bytes(tmp_path / "packed_exhausted") == {}


# ======================================================================
# 3. Char-level mode
# ======================================================================


class TestPrecomputeCharLevel:
    """Test precomputation in char-level segment mode."""

    def test_produces_valid_shards(self, tmp_path):
        """Char-level segment mode produces valid shards."""
        env = _make_precompute_env(tmp_path)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        total, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=20,
            rows_per_shard=100,
        )

        assert total > 0
        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        assert shard.n_rows > 0
        assert shard.total_features > 0


# ======================================================================
# 4. Packed segment conversion
# ======================================================================


class TestSegFeaturesToPacked:
    """The packed-only carrier preserves overlay codes without changing live features."""

    def test_preserves_every_uint8_code_exactly(self):
        codes = t.arange(256, dtype=t.uint8)
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.arange(256, dtype=t.int64),
            feature_mags=t.ones(256),
            char_start=4,
            char_end=99,
            token_start=2,
            token_end=20,
            feature_percentiles=codes,
        )

        packed = _seg_features_to_packed(seg)

        assert packed.percentiles.data_ptr() == codes.data_ptr()
        assert t.equal(packed.percentiles, codes)
        assert packed.char_start == 4 and packed.char_end == 99
        assert packed.token_start == 2 and packed.token_end == 20

    def test_missing_overlay_codes_remain_exact_zero_bytes(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([3, 7, 9]),
            feature_mags=t.ones(3),
            char_start=0,
            char_end=10,
        )

        packed = _seg_features_to_packed(seg)

        assert t.equal(packed.percentiles, t.zeros(3, dtype=t.uint8))

    def test_omits_live_only_bucket_field(self):
        """The packed-only carrier must not expose an unused bucket contract."""
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.arange(100_000),
            feature_mags=t.ones(100_000),
            char_start=0,
            char_end=10,
            feature_percentiles=t.zeros(100_000, dtype=t.uint8),
        )

        packed = _seg_features_to_packed(seg)

        assert "buckets" not in packed.__dataclass_fields__
        assert not hasattr(packed, "buckets")

    @pytest.mark.parametrize(
        "codes",
        [
            t.tensor([1, 2], dtype=t.int64),
            t.tensor([[1, 2]], dtype=t.uint8),
            t.tensor([1], dtype=t.uint8),
        ],
    )
    def test_rejects_malformed_overlay_codes(self, codes):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([3, 7]),
            feature_mags=t.ones(2),
            char_start=0,
            char_end=10,
            feature_percentiles=codes,
        )

        with pytest.raises(PackedRowInvariantError, match="aligned rank-1 uint8"):
            _seg_features_to_packed(seg)


# ======================================================================
# 5. Train/val split precomputation
# ======================================================================


class TestTrainValSplit:
    """Test that precompute_packed_rows produces both train and val splits."""

    def _precompute_both_splits(self, tmp_path, n_shards=3):
        """Helper: precompute train and val splits (mirrors run() logic).

        Uses context_len=16 (short) and buffer_size=50 so that the short
        synthetic docs from _make_precompute_env can be packed into rows.
        """
        env = _make_precompute_env(tmp_path, n_shards=n_shards)
        tok = SimpleTokenizer()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"])

        train_dir = tmp_path / "packed" / f"s{n_shards}"
        train_dir.mkdir(parents=True)
        val_dir = tmp_path / "packed" / f"val_s{n_shards}"
        val_dir.mkdir(parents=True)

        all_parquet_paths = env["parquet_paths"]
        train_paths = all_parquet_paths[:-1]
        val_paths = all_parquet_paths[-1:]

        # buffer_size must be <= N_DOCS (15) per shard, otherwise the buffer
        # never fills and no rows are packed (StopIteration fires first).
        context_len = 32
        buffer_size = 10

        train_rows, *_ = precompute_packed_rows(
            parquet_paths=train_paths,
            tokenizer=tok,
            feature_index=idx,
            context_len=context_len,
            output_dir=train_dir,
            buffer_size=buffer_size,
            rows_per_shard=100,
        )
        val_rows, *_ = precompute_packed_rows(
            parquet_paths=val_paths,
            tokenizer=tok,
            feature_index=idx,
            context_len=context_len,
            output_dir=val_dir,
            buffer_size=buffer_size,
            rows_per_shard=100,
        )

        return {
            "train_dir": train_dir,
            "val_dir": val_dir,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "n_shards": n_shards,
        }

    def test_both_splits_produce_shards(self, tmp_path):
        """Both train and val splits produce at least one shard file."""
        result = self._precompute_both_splits(tmp_path)
        train_shards = list(result["train_dir"].glob("shard_*.safetensors.zst"))
        val_shards = list(result["val_dir"].glob("shard_*.safetensors.zst"))
        assert len(train_shards) > 0, "Train split produced no shards"
        assert len(val_shards) > 0, "Val split produced no shards"

    def test_both_splits_have_rows(self, tmp_path):
        """Both splits contain actual data rows."""
        result = self._precompute_both_splits(tmp_path)
        assert result["train_rows"] > 0, "Train split has 0 rows"
        assert result["val_rows"] > 0, "Val split has 0 rows"

    def test_train_has_more_rows_than_val(self, tmp_path):
        """Train split (n-1 shards) should have more rows than val (1 shard)."""
        result = self._precompute_both_splits(tmp_path)
        assert result["train_rows"] > result["val_rows"], (
            f"Train ({result['train_rows']}) should have more rows than "
            f"val ({result['val_rows']})"
        )

    def test_val_dir_naming_matches_training_discovery(self, tmp_path):
        """Val directory name matches the training script's discovery convention:
        train_dir.parent / f'val_{train_dir.name}'."""
        result = self._precompute_both_splits(tmp_path)
        train_dir = result["train_dir"]
        expected_val_dir = train_dir.parent / f"val_{train_dir.name}"
        assert result["val_dir"] == expected_val_dir, (
            f"Val dir {result['val_dir']} != expected {expected_val_dir}"
        )

    def test_val_shards_are_loadable(self, tmp_path):
        """Val shards can be loaded as PackedRowShard with valid data."""
        result = self._precompute_both_splits(tmp_path)
        val_shards = sorted(result["val_dir"].glob("shard_*.safetensors.zst"))
        assert len(val_shards) > 0
        shard = load_packed_row_shard(val_shards[0])
        assert shard.n_rows > 0
        assert shard.total_features > 0
        assert shard.total_segments > 0

    def test_train_and_val_use_disjoint_parquets(self, tmp_path):
        """Train uses parquets[:-1], val uses parquets[-1:]. No overlap."""
        result = self._precompute_both_splits(tmp_path)
        train_set = set(result["train_paths"])
        val_set = set(result["val_paths"])
        assert train_set.isdisjoint(val_set), (
            f"Overlap between train and val parquets: {train_set & val_set}"
        )
        # Together they should cover all shards
        assert len(train_set) + len(val_set) == result["n_shards"]

    def test_val_reconstruct_produces_valid_batch(self, tmp_path):
        """Batches reconstructed from val shards have correct shapes."""
        result = self._precompute_both_splits(tmp_path)
        val_shards = sorted(result["val_dir"].glob("shard_*.safetensors.zst"))
        shard = load_packed_row_shard(val_shards[0])
        rb = reconstruct_batch(shard, list(range(min(2, shard.n_rows))), _make_identity_selection(shard), 10, device="cpu")
        assert rb.inputs.shape[0] <= 2
        assert rb.inputs.shape[1] == 32  # context_len used in _precompute_both_splits
        assert rb.enc_fids.shape[0] <= 2

    def test_with_two_shards(self, tmp_path):
        """With n_shards=2: 1 train shard + 1 val shard. Minimum viable split."""
        result = self._precompute_both_splits(tmp_path, n_shards=2)
        assert result["train_rows"] > 0
        assert result["val_rows"] > 0
        assert len(result["train_paths"]) == 1
        assert len(result["val_paths"]) == 1


class TestPackedRowsEmptySegmentInheritance:
    """Issue #522 site-f: precompute_packed_rows reads base shards ONLY via
    FeatureIndex, so it INHERITS the Task-5 empty-segment load guard — no
    packed-rows-specific code change is required. This test documents the
    inheritance: an empty base segment surfaces as CorruptFeatureShardError
    when FeatureIndex loads the shard during a lookup."""

    def test_feature_index_lookup_raises_on_empty_base_segment(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        # Corrupt base shard with an empty middle segment (offsets[1]==offsets[2]).
        corrupt = {
            "offsets": t.tensor([0, 2, 2, 4], dtype=t.int64),  # segment 1 empty
            "feature_ids": t.tensor([1, 5, 2, 7], dtype=t.int32),
            "feature_mags": t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32),
            "doc_ids": t.zeros(3, dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 2], dtype=t.int32),
            "segment_positions": t.tensor([0, 8, 16], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 8, 16], dtype=t.int32),
            "segment_char_ends": t.tensor([8, 16, 24], dtype=t.int32),
            "doc_token_counts": t.tensor([24], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 3,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 8,
        }
        save_raw_shard(feature_dir, "shard_00000", corrupt)

        idx = FeatureIndex(str(feature_dir))
        # _scan reads segment_size from the first shard but the empty-segment
        # guard fires lazily inside _load_shard, triggered by the lookup.
        with pytest.raises(CorruptFeatureShardError, match="segment 1|non-empty"):
            idx.lookup("shard_00000", 0)


# ======================================================================
# 9. run() base-shard hash cache (issue #530 Task 3)
# ======================================================================


def _make_run_env(tmp_path: Path, n_shards: int = 2) -> dict:
    """Build a complete env for calling precompute_packed_rows.run().

    Key differences from _make_precompute_env:
    - feature_dir lives under tmp_path / "s{SEGMENT_SIZE}" so that
      verify_segment_size's _parse_segment_size_from_path finds the s{N}
      component.
    - A feature_statistics.safetensors with the pct_checksum is created in
      feature_dir (run() requires find_statistics_path to return a path).
    - Tok boundary overlays are created under the real tok checksum derived
      from SimpleTokenizer(), not the hardcoded test string that
      _make_precompute_env uses.  run() computes the tok checksum itself via
      get_tokenizer() + compute_tokenizer_checksum(), which we patch to
      return SimpleTokenizer(); both sides must agree.
    """
    # Provide a sub-tree that embeds the segment-size in the path.
    sub = tmp_path / f"s{SEGMENT_SIZE}"
    sub.mkdir()
    env = _make_precompute_env(sub, n_shards=n_shards)
    feature_dir = Path(env["feature_dir"])  # = sub / "features"

    pct_checksum = env["pct_checksum"]

    # ---- 1. Stats file with pct_checksum ----
    # find_statistics_path(feature_dir) needs feature_statistics.safetensors to
    # exist; load_percentile_checksum needs the "percentile_checksum" key in its
    # safetensors metadata.  We write a minimal stub and then use the library's
    # own save_percentile_checksum to embed the key.
    stats_path = feature_dir / "feature_statistics.safetensors"
    safetensors_save_file({"_stub": t.zeros(1)}, str(stats_path), metadata={})
    save_percentile_checksum(stats_path, pct_checksum)

    # ---- 2. Tok boundary overlays under the REAL tok checksum ----
    # run() calls get_tokenizer() (patched → SimpleTokenizer()) then
    # compute_tokenizer_checksum().  We compute the same value here and
    # re-save the overlay shards under the matching directory.
    tok = SimpleTokenizer()
    real_tok_checksum = compute_tokenizer_checksum(tok)

    old_tok_checksum = env["tok_checksum"]
    old_tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, old_tok_checksum)
    new_tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, real_tok_checksum)

    if new_tok_dir != old_tok_dir:
        new_tok_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(new_tok_dir, real_tok_checksum, OverlayKind.SEG_BOUNDARY)
        for shard_path in glob_overlay_shards(old_tok_dir):
            (new_tok_dir / shard_path.name).write_bytes(shard_path.read_bytes())

    return {
        **env,
        "feature_dir": str(feature_dir),
        "tok_checksum": real_tok_checksum,
    }


def test_packed_rows_writes_and_reuses_base_hash_cache(tmp_path, monkeypatch):
    """run() (train + val write_manifest calls) populates a base-shard hash
    cache in feature_dir; a subsequent re-run hits the cache with 0 streaming
    reads.

    Teeth: monkeypatch _stream_base_hash and assert call count == 0 on re-run.
    """
    env = _make_run_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])

    cfg = OmegaConf.create({
        "feature_dir": env["feature_dir"],
        "sae_width": "16k",
        "segment_size": SEGMENT_SIZE,
        "data_dir": env["data_dir"],
        "context_len": 16,
        "rows_per_shard": 8,
        "buffer_size": 64,
        "max_shards": None,
        "resume": False,
        "resume_from_shard": None,
        "zstd_level": 12,
        "compress_workers": 0,
        "profile": False,
        "profile_summary_output": None,
    })

    tok = SimpleTokenizer()
    with patch(
        "scripts.preprocess.precompute_packed_rows.get_tokenizer",
        return_value=tok,
    ):
        assert run_packed_rows(cfg) == 0

    # Cache file must be present after the first run.
    cache_file = feature_dir / BASE_SHARD_HASH_CACHE_FILENAME
    assert cache_file.is_file(), f"Cache file missing: {cache_file}"

    # Every packed_rows-kind manifest must carry the correct hash.
    expected_hash = compute_base_shard_content_hash(
        glob_features(feature_dir), cache_dir=None
    )
    packed_manifests = [
        p
        for p in feature_dir.rglob("manifest.json")
        if json.loads(p.read_text()).get("kind") == "packed_rows"
    ]
    assert len(packed_manifests) >= 2, (
        f"Expected train + val manifests, found {len(packed_manifests)}"
    )
    for p in packed_manifests:
        data = json.loads(p.read_text())
        assert data["base_shard_content_hash"] == expected_hash, (
            f"{p}: hash {data['base_shard_content_hash']!r} != {expected_hash!r}"
        )

    # Re-run must hit the cache — _stream_base_hash must not be called.
    calls = {"n": 0}
    orig_stream = _oio._stream_base_hash

    def _counting_stream(sorted_paths):
        calls["n"] += 1
        return orig_stream(sorted_paths)

    monkeypatch.setattr(_oio, "_stream_base_hash", _counting_stream)
    cfg.resume = True
    with patch(
        "scripts.preprocess.precompute_packed_rows.get_tokenizer",
        return_value=tok,
    ):
        assert run_packed_rows(cfg) == 0

    assert calls["n"] == 0, (
        f"Expected 0 _stream_base_hash calls on re-run (cache should be warm), "
        f"got {calls['n']}"
    )
