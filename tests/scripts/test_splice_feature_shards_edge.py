"""Orchestration validation edge cases for splice_feature_shards (issue #590).

The cut_fraction-independent wiring/validation that raises BEFORE recombination:
required feature_dir, empty feature dir, the truncation-limit precondition, and
invalid shard_parallelism. Also covers: cut_fraction out-of-range raises before
any shard work; _build_jobs selects ALL shards sorted (no range filter);
_default_output_dir produces the exact expected string.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import COMPRESSED_EXT, glob_features, save_features
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)
SHARD_TEXTS = {
    0: ["alpha beta", "gamma delta epsilon"],
    1: ["one two three", "four five six"],
    2: ["red green", "blue cyan magenta"],
}


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return TOK

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda model: 8192)


def _make_base_dir(tmp_path, texts=SHARD_TEXTS):
    feature_dir = tmp_path / "base"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    for idx, shard_texts in texts.items():
        save_features(make_base_shard(shard_texts, TOK, 2), feature_dir / f"shard_{idx:05d}")
        atomic_write_parquet(
            pa.table({"text": shard_texts}), parquet_dir / f"shard_{idx:05d}.parquet"
        )
    return feature_dir, parquet_dir


def _cfg(feature_dir, parquet_dir, out, **overrides):
    base = {
        "feature_dir": str(feature_dir) if feature_dir is not None else None,
        "data_dir": str(parquet_dir),
        "output_dir": str(out) if out is not None else None,
        "model": "gemma-2-2b",
        "segment_size": 2,
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


def test_feature_dir_required(tmp_path):
    with pytest.raises(ValueError, match="feature_dir is required"):
        sfs.run(_cfg(None, tmp_path, tmp_path / "out"))


def test_no_base_shards_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "pq").mkdir()
    with pytest.raises(ValueError, match="no base feature shards"):
        sfs.run(_cfg(empty, tmp_path / "pq", tmp_path / "out"))


def test_output_dir_equal_feature_dir_raises_before_any_write(tmp_path):
    """output_dir == feature_dir would os.replace the immutable base shards in place
    (derived shards share the base filenames). It MUST raise before touching anything."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    base_before = {p: p.read_bytes() for p in feature_dir.iterdir() if p.is_file()}
    n_files_before = len(list(feature_dir.iterdir()))

    with pytest.raises(ValueError, match="immutable"):
        sfs.run(_cfg(feature_dir, parquet_dir, feature_dir))  # output_dir == feature_dir

    # Message names both paths.
    with pytest.raises(ValueError, match=str(feature_dir)):
        sfs.run(_cfg(feature_dir, parquet_dir, feature_dir))

    # Not a single base shard byte changed, and nothing new was written.
    assert len(list(feature_dir.iterdir())) == n_files_before, "no new files in feature_dir"
    for p, data in base_before.items():
        assert p.read_bytes() == data, f"base shard {p.name} was mutated"


def test_output_dir_equal_feature_dir_resolves_symlinks(tmp_path):
    """Equality is by resolved path: a non-normalized alias of feature_dir also raises."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    alias = feature_dir / "." / ".." / feature_dir.name  # same dir, different string
    with pytest.raises(ValueError, match="immutable"):
        sfs.run(_cfg(feature_dir, parquet_dir, alias))


def test_default_output_subdir_under_feature_dir_is_legal(tmp_path):
    """The DEFAULT <feature_dir>/splice/... subdir is under feature_dir but is NOT
    exact-equal, so it must be allowed (output_dir=None -> default)."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    sfs.run(_cfg(feature_dir, parquet_dir, None))  # None -> default subdir; must not raise
    default_out = sfs._default_output_dir(feature_dir, 0.6, 42)
    assert default_out.exists(), "default output subdir should have been created + written"
    assert any(default_out.glob(f"*_features{COMPRESSED_EXT}")), (
        "default subdir must hold derived shards"
    )


def test_precondition_limit_not_multiple_of_segment_size(tmp_path):
    # An odd context_window (7) with segment_size=2 -> min(ctx, sae_max_context)=7
    # is not a multiple of 2, so no recombined doc length bound can be guaranteed.
    feature_dir2 = tmp_path / "base2"
    parquet_dir2 = tmp_path / "pq2"
    feature_dir2.mkdir()
    parquet_dir2.mkdir()
    texts = ["alpha beta gamma delta", "epsilon zeta eta"]
    save_features(make_base_shard(texts, TOK, 2, context_window=7), feature_dir2 / "shard_00000")
    atomic_write_parquet(pa.table({"text": texts}), parquet_dir2 / "shard_00000.parquet")
    with pytest.raises(ValueError, match="multiple of"):
        sfs.run(_cfg(feature_dir2, parquet_dir2, tmp_path / "out2", context_window=7))


def test_invalid_shard_parallelism_raises(tmp_path):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="shard_parallelism"):
        sfs.run(_cfg(feature_dir, parquet_dir, out, shard_parallelism=0))
    # Validation happens BEFORE any output-dir mkdir (mirrors cut_fraction ordering),
    # so no directories are created on the raise.
    assert not out.exists(), "no output dir must be created on shard_parallelism validation error"


def test_skip_existing_requires_write_provenance(tmp_path):
    """skip_existing=true + write_provenance=false is rejected at config validation:
    an unverified resume (skip without a sidecar to verify recorded params against)
    violates the no-unverified-anything contract. It must raise before any shard work
    or directory creation."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="skip_existing requires write_provenance"):
        sfs.run(_cfg(feature_dir, parquet_dir, out, skip_existing=True, write_provenance=False))
    assert not out.exists(), "no output dir must be created on this validation error"


def test_skip_existing_false_allows_no_provenance(tmp_path):
    """The complement: skip_existing=false + write_provenance=false is legal (always
    recompute, no sidecar) and must run to completion without raising."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(feature_dir, parquet_dir, out, skip_existing=False, write_provenance=False))
    assert any(out.glob(f"*_features{COMPRESSED_EXT}")), "outputs must have been written"
    assert not (out / sfs.PROVENANCE_DIRNAME).exists(), (
        "no provenance dir when write_provenance=false"
    )


def test_cut_fraction_out_of_range_low_raises(tmp_path):
    """cut_fraction < 0.0 must raise ValueError before any shard work."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="cut_fraction"):
        sfs.run(_cfg(feature_dir, parquet_dir, out, cut_fraction=-0.1))
    # No output must have been written.
    assert not out.exists(), "output dir must not be created on cut_fraction validation error"


def test_cut_fraction_out_of_range_high_raises(tmp_path):
    """cut_fraction > 1.0 must raise ValueError before any shard work."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="cut_fraction"):
        sfs.run(_cfg(feature_dir, parquet_dir, out, cut_fraction=1.5))
    assert not out.exists(), "output dir must not be created on cut_fraction validation error"


def test_build_jobs_selects_all_shards_sorted(tmp_path):
    """_build_jobs selects ALL shards in sorted shard-index order (no range filter)."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    base_shards = glob_features(feature_dir)
    parquet_by_stem = {Path(p).stem: Path(p) for p in parquet_dir.glob("*.parquet")}
    run_params = sfs.SpliceRunParams(
        model="gemma-2-2b",
        segment_size=2,
        context_window=2048,
        sae_max_context=8192,
        cut_fraction=0.6,
        seed=42,
        write_provenance=False,
        skip_existing=False,
    )
    jobs, n_skipped = sfs._build_jobs(
        base_shards,
        parquet_by_stem,
        out,
        rank=0,
        world_size=1,
        skip_existing=False,
        run_params=run_params,
    )
    assert n_skipped == 0
    assert len(jobs) == 3, f"expected 3 jobs (all shards), got {len(jobs)}"
    indices = [j.shard_index for j in jobs]
    assert indices == sorted(indices), "jobs must be in sorted shard-index order"
    assert set(indices) == {0, 1, 2}, f"expected shard indices {{0,1,2}}, got {set(indices)}"


def test_default_output_dir_exact_string(tmp_path):
    """_default_output_dir must produce the exact path segment cut_frac{f}/seed{s}."""
    feature_dir = tmp_path / "feat"
    result = sfs._default_output_dir(feature_dir, 0.15, 42)
    # Exact string pin: cut_fraction=0.15, seed=42 -> cut_frac0.15/seed42
    assert result == feature_dir / "splice" / "cut_frac0.15" / "seed42", (
        f"unexpected output dir: {result}"
    )


def test_default_output_dir_exact_string_half(tmp_path):
    """cut_fraction=0.5, seed=7 -> cut_frac0.5/seed7."""
    feature_dir = tmp_path / "feat"
    result = sfs._default_output_dir(feature_dir, 0.5, 7)
    assert result == feature_dir / "splice" / "cut_frac0.5" / "seed7"
