"""Shared deterministic fixtures for token-boundary optimization tests (issue #529).

Mirrors the *proven* full-``run`` fixture in
``tests/scripts/test_token_boundaries_manifest_v2.py`` (``_make_env``) — in particular
the ``segment_char_starts = seg_positions * 6`` convention that makes the script's
``_detect_source_bos_char_len`` resolve consistently — generalized to N shards.

Segmentation metadata (texts, char spans, ``doc_token_counts``) is **identical across
shards** so BOS detection from shard 0 stays valid for every shard; only the SAE feature
content (ids/mags) varies per shard, so shards have **distinct bytes** (distinct
``base_shard_content_hash``) while the token-boundary computation is uniformly valid.

This is test-only infrastructure; production paths never import it.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.feature_io import save_features

SEGMENT_SIZE = 16
_N_DOCS = 2
_N_SEGS_PER_DOC = 2
_N_FEATS_PER_SEG = 3
_N_SAE = 64


class CharWordTokenizer:
    """Deterministic char-level tokenizer: one token per whitespace word, +BOS.

    Token 0 = BOS = ``"<bos>"`` (5 chars); token n>0 = ``f"t{n:03d} "`` (5 chars).
    Byte-for-byte the same behavior as the ``_SimpleTokenizer`` used by the proven
    manifest-v2 fixture, so its tokenizer checksum and produced boundaries are stable.
    """

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(s, prepend=prepend) for s in text]
        tokens = [i + 1 for i in range(len(text.split()))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        out = []
        for tid in token_ids:
            out.append("<bos>" if tid == 0 else f"t{tid:03d} ")
        return "".join(out)


def _texts() -> list[str]:
    """Same per-doc texts for every shard (keeps BOS detection consistent)."""
    return [" ".join(f"word{w}" for w in range(20 + d * 5)) for d in range(_N_DOCS)]


def _shard_data(shard_idx: int) -> dict:
    """Build one shard's CSR + segmentation dict. Feature content varies by shard."""
    total_segs = _N_DOCS * _N_SEGS_PER_DOC
    total_feats = total_segs * _N_FEATS_PER_SEG
    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0
    for doc_idx in range(_N_DOCS):
        for seg_idx in range(_N_SEGS_PER_DOC):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset
            for f in range(_N_FEATS_PER_SEG):
                # Distinct, non-duplicate ids within a segment; varied per shard.
                fid = (shard_idx * 7 + doc_idx * 10 + seg_idx * 5 + f) % _N_SAE
                all_fids[feat_offset + f] = fid
                all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx + 0.001 * shard_idx
            feat_offset += _N_FEATS_PER_SEG
            seg_global += 1
    offsets[total_segs] = feat_offset

    segment_char_starts = seg_positions * 6
    segment_char_ends = (seg_positions + SEGMENT_SIZE) * 6

    return {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((_N_DOCS,), 41, dtype=t.int32),
        "segment_char_starts": segment_char_starts,
        "segment_char_ends": segment_char_ends,
        "n_docs": _N_DOCS,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }


def make_token_boundary_env(tmp_path: Path, n_shards: int = 1) -> dict:
    """Materialize parquet + N base shards under an ``s16`` feature dir.

    Returns a dict with ``data_dir``, ``feature_dir`` (both str), a deterministic
    ``tokenizer``, and ``segment_size``.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = tmp_path / "features" / f"s{SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    texts = _texts()
    for i in range(n_shards):
        stem = f"shard_{i:05d}"
        pq.write_table(pa.table({"text": texts}), str(data_dir / f"{stem}.parquet"))
        save_features(_shard_data(i), feature_dir / f"{stem}_features")
    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "tokenizer": CharWordTokenizer(),
        "segment_size": SEGMENT_SIZE,
    }
