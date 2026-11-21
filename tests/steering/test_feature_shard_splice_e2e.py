"""End-to-end tests for splice_feature_shard (issue #590, Task 9).

Covers: disk-level determinism (byte-identical parquet + t.equal tensors),
golden fixed-seed snapshot, skip_existing behaviour, and the seeding contract
(same params -> identical; different p -> independent selection).
"""

from __future__ import annotations

import time

import pyarrow as pa
import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import COMPRESSED_EXT, glob_features, load_features, save_features
from steering.feature_shard_splice import recombination_seed, splice_feature_shard
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)
SEG_SIZE = 2

# Corpus used for most e2e tests: enough docs + splice points that cuts actually happen.
E2E_CORPUS = [
    "alpha beta gamma delta",
    "epsilon zeta eta theta",
    "iota kappa lambda mu",
    "nu xi omicron pi rho",
    "sigma tau upsilon phi",
]

# For the golden snapshot: a stable, well-known corpus + seed combination.
GOLDEN_CORPUS = [
    "cats",
    "zebra",
    "dolphin",
    "penguin",
    "elephant",
    "kangaroos",
]
GOLDEN_SEED = 42
GOLDEN_CUT_FRACTION = 0.5
GOLDEN_SHARD_INDEX = 7


def _make_base(tmp_path, texts, *, name="shard_00007", tok=TOK, seg_size=SEG_SIZE):
    """Write a base shard + parquet; return (feature_path, parquet_path)."""
    shard = make_base_shard(texts, tok, seg_size)
    save_features(shard, tmp_path / name)
    parquet_path = tmp_path / f"{name}.parquet"
    atomic_write_parquet(pa.table({"text": texts}), parquet_path)
    paths = glob_features(tmp_path)
    assert paths, "save_features produced no files"
    return paths[0], parquet_path


def _splice_call(shard_path, parquet_path, *, cut_fraction, seed, shard_index=7, tok=TOK):
    return splice_feature_shard(
        shard_path,
        parquet_path,
        tok,
        model="test-model",
        cut_fraction=cut_fraction,
        seed=seed,
        shard_index=shard_index,
        shard_name="shard_00007",
        segment_size=SEG_SIZE,
        context_window=2048,
        sae_max_context=None,
    )


# ---------------------------------------------------------------------------
# Determinism through disk
# ---------------------------------------------------------------------------


def test_determinism_through_disk(tmp_path):
    """Two splice calls on the same inputs produce byte-identical parquet and
    t.equal tensors (feature_ids, feature_mags, offsets, doc_ids, …)."""
    shard_path, parquet_path = _make_base(tmp_path, E2E_CORPUS)

    derived1, texts1, prov1 = _splice_call(shard_path, parquet_path, cut_fraction=0.6, seed=42)
    derived2, texts2, prov2 = _splice_call(shard_path, parquet_path, cut_fraction=0.6, seed=42)

    # Provenance dicts must be identical.
    assert prov1 == prov2

    # Text lists must be identical.
    assert texts1 == texts2

    # Tensor fields must be bit-for-bit equal.
    tensor_keys = [
        "feature_ids",
        "feature_mags",
        "offsets",
        "doc_ids",
        "segment_indices",
        "segment_positions",
        "segment_char_starts",
        "segment_char_ends",
        "doc_token_counts",
    ]
    for key in tensor_keys:
        assert t.equal(derived1[key], derived2[key]), (
            f"derived['{key}'] differs between two identical calls"
        )

    # Write to disk and compare bytes (parquet) + decompressed tensors.
    out_dir = tmp_path / "derived"
    out_dir.mkdir()
    pq_path1 = out_dir / "out1.parquet"
    pq_path2 = out_dir / "out2.parquet"
    atomic_write_parquet(pa.table({"text": texts1}), pq_path1)
    atomic_write_parquet(pa.table({"text": texts2}), pq_path2)
    assert pq_path1.read_bytes() == pq_path2.read_bytes(), (
        "byte-identical parquet required (same row order, same encodings)"
    )

    shard_stem1 = out_dir / "derived1"
    shard_stem2 = out_dir / "derived2"
    save_features(derived1, shard_stem1)
    save_features(derived2, shard_stem2)
    # Load back and compare decompressed tensors.
    loaded1 = load_features(out_dir / f"derived1_features{COMPRESSED_EXT}")
    loaded2 = load_features(out_dir / f"derived2_features{COMPRESSED_EXT}")
    for key in tensor_keys:
        assert t.equal(loaded1[key], loaded2[key]), (
            f"decompressed '{key}' differs after disk round-trip"
        )


# ---------------------------------------------------------------------------
# Golden fixed-seed snapshot
# ---------------------------------------------------------------------------

# Goldens recorded after Task 9 seeding switch (recombination_seed wiring).
# To re-record: print(prov["n_docs"], prov["docs"][0]["provenance"], prov["docs"][-1]["provenance"])
# CharMergeTokenizer(bos_id=500), SEG_SIZE=2, GOLDEN_CORPUS, seed=42, p=0.5, shard_index=7
GOLDEN_N_DOCS: int = 6
GOLDEN_FIRST_PROV: list = [[7, 0, 0], [7, 5, 3], [7, 5, 4]]
GOLDEN_LAST_PROV: list = [[7, 5, 0], [7, 5, 1], [7, 5, 2], [7, 4, 1], [7, 4, 2], [7, 4, 4]]


def test_golden_snapshot(tmp_path):
    """Fixed seed + corpus -> deterministic doc count + first/last provenance records."""
    tmp = tmp_path / "g"
    tmp.mkdir()
    shard_path, parquet_path = _make_base(tmp, GOLDEN_CORPUS, name="shard_00007")
    derived, texts, prov = _splice_call(
        shard_path,
        parquet_path,
        cut_fraction=GOLDEN_CUT_FRACTION,
        seed=GOLDEN_SEED,
        shard_index=GOLDEN_SHARD_INDEX,
    )

    # Structural invariants always checked (independent of the exact golden).
    assert prov["n_docs"] == derived["n_docs"]
    assert len(texts) == derived["n_docs"]
    # shard_seed must equal recombination_seed.
    assert prov["shard_seed"] == recombination_seed(
        GOLDEN_SEED, GOLDEN_SHARD_INDEX, GOLDEN_CUT_FRACTION
    )
    # cut_fraction key present.
    assert "cut_fraction" in prov["params"]
    # All 7 counters present.
    for key in [
        "n_cuts_final",
        "n_redrawn",
        "n_loops",
        "n_splits",
        "n_trimmed_docs",
        "n_trimmed_tokens",
        "n_marker_normalized_docs",
    ]:
        assert key in prov["params"]

    # Concrete golden checks (hardcoded after first verified run).
    assert prov["n_docs"] == GOLDEN_N_DOCS, f"expected {GOLDEN_N_DOCS} docs, got {prov['n_docs']}"
    assert prov["docs"][0]["provenance"] == GOLDEN_FIRST_PROV
    assert prov["docs"][-1]["provenance"] == GOLDEN_LAST_PROV


# ---------------------------------------------------------------------------
# skip_existing behaviour
# ---------------------------------------------------------------------------


def test_skip_existing_does_not_rewrite(tmp_path):
    """Second run with skip_existing=True must not overwrite already-written files."""
    feature_dir = tmp_path / "base"
    parquet_dir = tmp_path / "parquets"
    out_dir = tmp_path / "out"
    feature_dir.mkdir()
    parquet_dir.mkdir()

    for idx, texts in {0: E2E_CORPUS}.items():
        save_features(make_base_shard(texts, TOK, SEG_SIZE), feature_dir / f"shard_{idx:05d}")
        atomic_write_parquet(pa.table({"text": texts}), parquet_dir / f"shard_{idx:05d}.parquet")

    def _cfg(**overrides):
        base = {
            "feature_dir": str(feature_dir),
            "data_dir": str(parquet_dir),
            "output_dir": str(out_dir),
            "model": "gemma-2-2b",
            "segment_size": SEG_SIZE,
            "context_window": 2048,
            "cut_fraction": 0.6,
            "seed": 42,
            "skip_existing": True,
            "write_provenance": True,
            "shard_parallelism": 1,
            "async_feature_save_depth": 0,
            "profile": False,
            "profile_summary_output": "out/traces/splice_feature_shards_profile.json",
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return TOK

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
        mp.setattr(sfs, "resolve_sae_max_context", lambda _model: 8192)
        sfs.run(_cfg())

    # Record mtime of all written files.
    written = list(out_dir.rglob("*"))
    assert written, "first run must have written some files"
    mtimes_before = {p: p.stat().st_mtime for p in written if p.is_file()}

    # Small delay to ensure any new writes would have a different mtime.
    time.sleep(0.05)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
        mp.setattr(sfs, "resolve_sae_max_context", lambda _model: 8192)
        sfs.run(_cfg(skip_existing=True))

    mtimes_after = {p: p.stat().st_mtime for p in written if p.is_file()}
    for path, mt in mtimes_before.items():
        assert mtimes_after.get(path) == mt, f"skip_existing=True must not rewrite {path.name}"


# ---------------------------------------------------------------------------
# Seeding contract
# ---------------------------------------------------------------------------


def test_seeding_same_params_identical(tmp_path):
    """Same (seed, shard_index, cut_fraction) must produce identical shard + texts + provenance."""
    shard_path, parquet_path = _make_base(tmp_path, E2E_CORPUS)

    d1, t1, p1 = _splice_call(shard_path, parquet_path, cut_fraction=0.6, seed=7, shard_index=3)
    d2, t2, p2 = _splice_call(shard_path, parquet_path, cut_fraction=0.6, seed=7, shard_index=3)

    assert t1 == t2
    assert p1 == p2
    for key in ["feature_ids", "feature_mags", "offsets"]:
        assert t.equal(d1[key], d2[key])


def test_seeding_different_p_gives_different_selection(tmp_path):
    """Different cut_fraction values fold into the shard_seed, yielding independent selections.

    For two cut_fraction values whose target_k values differ (0.3 vs 0.8 on a many-splice
    corpus), the derived doc structures must differ.  With old seeding (stable_shard_seed,
    no cut_fraction), the seeds were identical; with recombination_seed the shuffles diverge.
    The test asserts the outputs differ (which holds both for target_k reasons and
    for seed-independence after the change).
    """
    shard_path, parquet_path = _make_base(tmp_path, E2E_CORPUS)

    _, texts_lo, prov_lo = _splice_call(shard_path, parquet_path, cut_fraction=0.3, seed=42)
    _, texts_hi, prov_hi = _splice_call(shard_path, parquet_path, cut_fraction=0.8, seed=42)

    # Doc count or structure must differ (guaranteed because target_k differs).
    docs_lo = [(d["provenance"]) for d in prov_lo["docs"]]
    docs_hi = [(d["provenance"]) for d in prov_hi["docs"]]
    assert docs_lo != docs_hi, (
        "cut_fraction=0.3 and cut_fraction=0.8 must produce different derived shard structures"
    )

    # The shard_seeds recorded must also differ (recombination_seed folds cut_fraction in).
    seed_lo = recombination_seed(42, 7, 0.3)
    seed_hi = recombination_seed(42, 7, 0.8)
    assert prov_lo["shard_seed"] == seed_lo
    assert prov_hi["shard_seed"] == seed_hi
    assert seed_lo != seed_hi, "recombination_seed must be cut_fraction-sensitive"
