"""Paired source/parquet/token-overlay fixtures packed by the production packer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from safetensors.torch import save_file

from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from scripts.preprocess.precompute_token_boundaries import compute_doc_token_boundaries_from_tokens
from scripts.train.conditional_encdec_train import _packed_row_train_dir
from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import glob_features, save_features
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.segmentation import compute_tokenizer_checksum


class PairTokenizer:
    """Two UTF-8 bytes per decoder token; incomplete UTF-8 prefixes have zero width."""

    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text: Any, prepend: Any = None) -> Any:
        if isinstance(text, list):
            return [self.encode(item, prepend) for item in text]
        raw = text.encode()
        tokens = [int.from_bytes(raw[i:i + 2], "big") + (257 if len(raw[i:i + 2]) == 2 else 1)
                  for i in range(0, len(raw), 2)]
        return ([0] if prepend is not None else []) + tokens

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join((b"<bos>" if token == 0 else
                        bytes([token - 1]) if token <= 256 else
                        (token - 257).to_bytes(2, "big")) for token in tokens)
        return raw.decode(errors="ignore")


@dataclass
class CadenceFixture:
    feature_dir: Path
    parquet_dir: Path
    packed_dir: Path
    token_dir: Path
    meta: dict[str, Any]
    tokenizer: PairTokenizer

    def arguments(self) -> dict[str, Any]:
        return {"checkpoint_meta": self.meta, "feature_dir": self.feature_dir,
                "parquet_dir": self.parquet_dir, "tokenizer": self.tokenizer}


def make_cadence_fixture(
    root: Path, *, context_len: int = 15,
    documents: tuple[tuple[int, ...], ...] = ((10, 14, 8), (8, 12), (6, 10)),
    source_lengths: tuple[tuple[int, ...], ...] | None = None,
    source_bos: int = 0, unicode_text: bool = False, spliced: bool = False,
    rows_per_shard: int = 2, n_shards: int = 3,
) -> CadenceFixture:
    feature_dir = root / "features"
    parquet_dir = feature_dir / "_parquets"
    parquet_dir.mkdir(parents=True)
    tokenizer = PairTokenizer()
    checksum = compute_tokenizer_checksum(tokenizer)
    pct = "a" * 64
    token_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, pct)
    source_lengths = source_lengths or tuple(tuple(128 for _ in doc) for doc in documents)
    for shard in range(n_shards):
        stem = f"shard_{shard:05d}"
        doc_ids, positions, ordinals, cs, ce, counts, starts, ends = [], [], [], [], [], [], [], []
        texts = []
        for doc, (char_widths, lengths) in enumerate(zip(documents, source_lengths, strict=True)):
            text = "".join(chr(97 + ordinal) * width for ordinal, width in enumerate(char_widths))
            if unicode_text:
                text = "€" + text[1:]
            texts.append(text)
            source_pos = char_pos = 0
            for ordinal, (width, length) in enumerate(zip(char_widths, lengths, strict=True)):
                doc_ids.append(doc)
                ordinals.append(ordinal)
                positions.append(source_pos)
                cs.append(0 if ordinal == 0 else source_bos + char_pos)
                char_pos += width
                ce.append(source_bos + char_pos)
                source_pos += length
            counts.append(source_pos)
            n = len(char_widths)
            lo, hi = compute_doc_token_boundaries_from_tokens(
                tokenizer.encode(text, prepend=0), cs[-n:], ce[-n:], tokenizer,
                source_bos_char_len=source_bos,
            )
            starts.extend(lo.tolist())
            ends.extend(hi.tolist())
        n = len(doc_ids)
        pq.write_table(pa.table({"text": texts}), parquet_dir / f"{stem}.parquet", row_group_size=1)
        save_features({
            "offsets": t.arange(n + 1, dtype=t.int64), "feature_ids": t.ones(n, dtype=t.int32),
            "feature_mags": t.ones(n), "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(ordinals, dtype=t.int32),
            "segment_positions": t.tensor(positions, dtype=t.int32),
            "segment_char_starts": t.tensor(cs, dtype=t.int32), "segment_char_ends": t.tensor(ce, dtype=t.int32),
            "doc_token_counts": t.tensor(counts, dtype=t.int32), "n_docs": len(texts), "n_segments": n,
            "segment_mode": "fixed_tokens", "segment_size": 128, "context_window": 4096,
            "layer": 22, "pooling": "max_pool", "ignore_bos_features": True,
        }, feature_dir / f"{stem}_features")
        save_overlay_shard({"segment_token_starts": t.tensor(starts, dtype=t.int32),
                            "segment_token_ends": t.tensor(ends, dtype=t.int32)}, overlay_shard_path(token_dir, stem))
        save_overlay_shard({"feature_percentiles": t.full((n,), 127, dtype=t.uint8)}, overlay_shard_path(pct_dir, stem))
        if spliced:
            sidecar = feature_dir / "_splice_provenance" / f"{stem}.json"
            sidecar.parent.mkdir(exist_ok=True)
            sidecar.write_text(json.dumps({"schema_version": 2, "n_docs": len(texts),
                "docs": [{"doc_id": i, "n_segments": len(widths), "splices": (
                    [{"pos": 1, "whf_base": [shard, i, 1], "whf_derived": [i, 1]}] if len(widths) > 1 else [])}
                    for i, widths in enumerate(documents)],
                "params": {"n_splice_positions": sum(len(widths) > 1 for widths in documents)}}))
    features = glob_features(feature_dir)
    write_manifest(token_dir, checksum, OverlayKind.SEG_BOUNDARY, base_shard_paths=features)
    write_manifest(pct_dir, pct, OverlayKind.PERCENTILE, base_shard_paths=features)
    save_file({}, feature_dir / "feature_statistics.safetensors", metadata={"percentile_checksum": pct, "tokenizer_checksum": checksum})
    index = FeatureIndex(str(feature_dir), percentile_checksum=pct, tokenizer_checksum=checksum)
    packed_dir = _packed_row_train_dir(feature_dir, percentile_checksum=pct, tokenizer_checksum=checksum,
                                       context_len=context_len, max_train_shards=None)
    precompute_packed_rows([str(p) for p in sorted(parquet_dir.glob("*.parquet"))[:-1]], tokenizer, index,
                           context_len, packed_dir, buffer_size=2, rows_per_shard=rows_per_shard)
    packed_paths = glob_overlay_shards(packed_dir)
    write_manifest(packed_dir, f"{pct}:{checksum}", OverlayKind.PACKED_ROW, metadata={
        "type": "packed_rows", "context_len": context_len, "max_shards": None,
        "pct_checksum": pct, "tok_checksum": checksum, "buffer_size": 2, "rows_per_shard": rows_per_shard,
        "base_shard_stems": [f"shard_{i:05d}" for i in range(n_shards)], "provenance_shards": len(packed_paths),
    }, base_shard_paths=features)
    meta = {"step": 250000, "max_seq_len": context_len, "user_config": {
        "feature_dir": str(feature_dir), "data_dir": str(parquet_dir), "segment_size": 128,
        "max_seq_len": context_len, "use_precomputed_batches": True,
        "feature_delta": {"mode": "splice_counterfactual"} if spliced else None,
    }, "dataloader_state_dict": {"shard": packed_paths[0].name}}
    return CadenceFixture(feature_dir, parquet_dir, packed_dir, token_dir, meta, tokenizer)
