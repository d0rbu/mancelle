"""Fixtures for splice_counterfactual live-dataloader tests (issue #596).

Builds a REAL derived directory by running the actual splicer per shard and
writing shard/parquet/sidecar exactly as scripts/preprocess/splice_feature_shards
does — the strongest realism available without GPU artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from steering.feature_io import save_features
from steering.feature_shard_splice import (
    PARQUET_DIRNAME,
    PROVENANCE_DIRNAME,
    splice_feature_shard,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

DEFAULT_TEXTS_BY_SHARD = [
    # train shard: 6 docs, several segments each (seg_size=2 chars ≈ 2 tokens)
    ["alpha beta gamma", "delta epsilon zeta", "eta theta iota",
     "kappa lambda mu", "nu xi omicron", "pi rho sigma"],
    # val shard (the loader's VAL split = last parquet)
    ["tau upsilon phi", "chi psi omega", "ab cd ef gh", "ij kl mn op"],
]


class SpliceCfTokenizer(CharMergeTokenizer):
    """CharMergeTokenizer + the nanochat batch-encode API the live loader uses.

    encode(text) without prepend stays byte-compatible with the splicer-side
    tokenization; encode(list, prepend=bos) mirrors nanochat's tokenizer so
    loader-side ids == [bos, *body] == the splicer's __call__ output.
    """

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(x, prepend=prepend) for x in text]
        ids = self._encode_body(text)
        if prepend is not None:
            ids = [prepend, *ids]
        return ids

    def get_vocab_size(self) -> int:  # nanochat API surface used by callers
        return 1000


def build_derived_env(tmp_path, *, cut_fraction, texts_by_shard=None, seg_size=2):
    """Splice a synthetic base dir into a derived dir with v2 sidecars.

    Returns {"feature_dir", "data_dir", "tokenizer", "seg_size"} where
    feature_dir is the derived dir (contains s{seg_size} via the base path so
    verify_segment_size passes) and data_dir is its _parquets.
    """
    tok = SpliceCfTokenizer(bos_id=500)
    texts_by_shard = texts_by_shard or DEFAULT_TEXTS_BY_SHARD
    base_dir = Path(tmp_path) / "features" / f"s{seg_size}" / "base"
    base_dir.mkdir(parents=True)
    derived = base_dir / "splice" / f"cut_frac{cut_fraction}" / "seed42"
    (derived / PARQUET_DIRNAME).mkdir(parents=True)
    (derived / PROVENANCE_DIRNAME).mkdir(parents=True)
    for idx, texts in enumerate(texts_by_shard):
        stem = f"shard_{idx:05d}"
        base_shard = make_base_shard(texts, tok, seg_size)
        save_features(base_shard, base_dir / stem)
        base_pq = base_dir / f"{stem}.parquet"
        atomic_write_parquet(pa.table({"text": texts}), base_pq)
        d_shard, d_texts, prov = splice_feature_shard(
            base_dir / f"{stem}_features.safetensors.zst", base_pq, tok,
            model="test-model", cut_fraction=cut_fraction, seed=42,
            shard_index=idx, shard_name=stem, segment_size=seg_size,
            context_window=2048, sae_max_context=None,
        )
        atomic_write_parquet(pa.table({"text": d_texts}), derived / PARQUET_DIRNAME / f"{stem}.parquet")
        atomic_json_dump(prov, derived / PROVENANCE_DIRNAME / f"{stem}.json")
        save_features(d_shard, derived / stem)
    return {
        "feature_dir": str(derived),
        "data_dir": str(derived / PARQUET_DIRNAME),
        "tokenizer": tok,
        "seg_size": seg_size,
    }
