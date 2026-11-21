"""Unit tests for BpbCache SQLite + reconcile (Tasks 9.2-9.3)."""
from __future__ import annotations

from sqlite3 import Connection

import pytest

from steering.eval.axbench_bpb.bpb_cache import BpbCache, reconcile_cache
from steering.eval.axbench_bpb.parquet_writer import (
    write_bpb_score_rows,
)
from steering.eval.axbench_bpb.types import BpbScoreRow


def _connection(cache: BpbCache) -> Connection:
    connection = cache._conn
    assert connection is not None
    return connection


def test_cache_put_has_basic(tmp_path):
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        # Before put: has() returns False
        assert not cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )
        cache.put(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        # After put: has() returns True
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )


def test_cache_persistence_across_opens(tmp_path):
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
    # Re-open: entry still present
    with BpbCache(path) as cache:
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )


def test_cache_put_idempotent_upsert(tmp_path):
    """Double-put of same primary key updates in place (INSERT OR REPLACE)."""
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        cache.put(  # overwrite
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=2.5, n_output_tokens=20, n_output_bytes=80, truncated=True,
        )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )
        # Direct query to verify overwrite worked
        cur = _connection(cache).execute(
            "SELECT bpb, n_output_tokens, truncated FROM bpb_scores "
            "WHERE method_name=? AND concept_id=? AND sample_id=? "
            "AND coefficient_str=? AND target=?",
            ("m", 1, "s", "0", "decoder_only"),
        )
        row = cur.fetchone()
        assert row == (2.5, 20, 1)


def test_cache_distinguishes_by_target(tmp_path):
    """Same (method, concept, sample, coef) but different target → 2 separate entries."""
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        for target in ("decoder_only", "encdec"):
            cache.put(
                method_name="m", concept_id=1, sample_id="s",
                coefficient_str="0", target=target,
                bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
            )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="encdec",
        )


def test_cache_distinguishes_by_coefficient(tmp_path):
    """coefficient_str is part of the primary key."""
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )
        assert not cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="1.5", target="decoder_only",
        )


def test_cache_has_without_enter_asserts(tmp_path):
    """Using has() outside the context manager asserts (requires __enter__)."""
    path = tmp_path / "rank_000.sqlite"
    cache = BpbCache(path)
    with pytest.raises(AssertionError):
        cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )


def test_cache_creates_parent_dirs(tmp_path):
    """Nested cache path auto-creates parent directories."""
    path = tmp_path / "deep" / "nested" / "rank.sqlite"
    with BpbCache(path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
    assert path.exists()


def test_cache_many_entries_by_concept(tmp_path):
    """Index on concept_id works — insert 100 entries across 10 concepts."""
    path = tmp_path / "rank_000.sqlite"
    with BpbCache(path) as cache:
        for cid in range(10):
            for i in range(10):
                cache.put(
                    method_name=f"m{i}", concept_id=cid, sample_id=f"s{i}",
                    coefficient_str="0", target="decoder_only",
                    bpb=float(i), n_output_tokens=10,
                    n_output_bytes=40, truncated=False,
                )
        cur = _connection(cache).execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 100
        # Query with index
        cur = _connection(cache).execute(
            "SELECT COUNT(*) FROM bpb_scores WHERE concept_id=?", (5,),
        )
        assert cur.fetchone()[0] == 10


def _make_row(
    method_name="m", concept_id=1, sample_id="s",
    coefficient_str="0", target="decoder_only",
    bpb=1.5, n_tokens=10, n_bytes=40, truncated=False,
):
    return BpbScoreRow(
        method_name=method_name, concept_id=concept_id,
        mode="positive", content_class="concept_rich", target=target,
        coefficient_str=coefficient_str, sample_id=sample_id,
        bpb_intervened=bpb, n_output_tokens=n_tokens,
        n_output_bytes=n_bytes, truncated=truncated,
    )


def test_reconcile_reseeds_sqlite_from_parquet(tmp_path):
    """Parquet has rows but SQLite is empty → reconcile reseeds SQLite."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows = [_make_row()]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        assert cache.has(
            method_name="m", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )


def test_reconcile_drops_sqlite_entries_not_in_parquet(tmp_path):
    """SQLite has entries but parquet is absent → reconcile drops them."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()  # empty — no parquet files
    cache_path = tmp_path / "rank_000.sqlite"

    with BpbCache(cache_path) as cache:
        cache.put(
            method_name="orphan", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
            bpb=1.5, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        assert not cache.has(
            method_name="orphan", concept_id=1, sample_id="s",
            coefficient_str="0", target="decoder_only",
        )


def test_reconcile_idempotent_on_consistent_state(tmp_path):
    """Already-consistent SQLite+parquet: reconcile is a no-op, data preserved."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows = [_make_row(bpb=1.5)]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    # Second call should not change anything
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        cur = _connection(cache).execute(
            "SELECT bpb FROM bpb_scores WHERE sample_id=?", ("s",)
        )
        assert cur.fetchone() == (1.5,)


def test_reconcile_preserves_partial_overlap(tmp_path):
    """Parquet has rows A, B; SQLite has B, C. Reconcile ends with SQLite = {A, B}."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows_parquet = [
        _make_row(sample_id="A", bpb=1.0),
        _make_row(sample_id="B", bpb=2.0),
    ]
    write_bpb_score_rows(rows_parquet, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    with BpbCache(cache_path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="B",
            coefficient_str="0", target="decoder_only",
            bpb=2.0, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        cache.put(
            method_name="m", concept_id=1, sample_id="C",  # orphan
            coefficient_str="0", target="decoder_only",
            bpb=3.0, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        assert cache.has(method_name="m", concept_id=1, sample_id="A",
                         coefficient_str="0", target="decoder_only")
        assert cache.has(method_name="m", concept_id=1, sample_id="B",
                         coefficient_str="0", target="decoder_only")
        assert not cache.has(method_name="m", concept_id=1, sample_id="C",
                             coefficient_str="0", target="decoder_only")
        cur = _connection(cache).execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 2


def test_reconcile_multiple_parquet_shards(tmp_path):
    """Multiple batch_NNNNNN.parquet shards are all read."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    write_bpb_score_rows(
        [_make_row(sample_id="s1")], rank_dir / "batch_000000.parquet",
    )
    write_bpb_score_rows(
        [_make_row(sample_id="s2")], rank_dir / "batch_000001.parquet",
    )
    write_bpb_score_rows(
        [_make_row(sample_id="s3")], rank_dir / "batch_000002.parquet",
    )
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        for sid in ("s1", "s2", "s3"):
            assert cache.has(method_name="m", concept_id=1, sample_id=sid,
                             coefficient_str="0", target="decoder_only")


def test_reconcile_fresh_empty_dir_is_noop(tmp_path):
    """Non-existent rank dir + absent SQLite → creates empty cache cleanly."""
    rank_dir = tmp_path / "nonexistent_rank_000"
    cache_path = tmp_path / "fresh_cache.sqlite"

    # Should not raise even though rank_dir doesn't exist
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        cur = _connection(cache).execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 0


def test_reconcile_preserves_row_bpb_values(tmp_path):
    """Reseeded SQLite rows carry the original bpb/n_tokens/n_bytes/truncated."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows = [_make_row(
        sample_id="X", bpb=3.14, n_tokens=42, n_bytes=168, truncated=True,
    )]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        cur = _connection(cache).execute(
            "SELECT bpb, n_output_tokens, n_output_bytes, truncated "
            "FROM bpb_scores WHERE sample_id=?", ("X",),
        )
        row = cur.fetchone()
        assert row == (3.14, 42, 168, 1)
