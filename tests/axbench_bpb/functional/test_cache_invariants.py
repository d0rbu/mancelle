"""Functional tests covering BpbCache + reconcile_cache invariants.

Cache correctness is a famously hard problem; these tests pin the
trickier edge cases not covered by ``test_cache_reconciliation.py`` or
``test_context_cache_*``:

1. ``test_reconcile_idempotent`` — repeated reconciles preserve state
2. ``test_reconcile_drops_sqlite_entries_for_deleted_parquet`` — orphan
   removal after a parquet is manually deleted
3. ``test_reconcile_reseeds_sqlite_from_parquet_after_db_loss`` —
   recovery from SQLite file deletion
4. ``test_cache_has_check_is_correct_after_concurrent_writes`` — has()
   reflects put() immediately + empty-cache negative case
5. ``test_reconcile_handles_empty_rank_dir`` — empty (vs missing) dir
   reconciles to an empty cache
6. ``test_cache_put_overwrites_existing_key`` — second put with same
   key overwrites (no IntegrityError)
7. ``test_reconcile_preserves_data_integrity_under_partial_writes`` —
   multi-shard partial writes preserve all rows without duplicates
8. ``test_cache_keys_are_namespace_isolated_by_target`` — same logical
   key but different target coexists across decoder_only and encdec
"""
from __future__ import annotations

import os
from pathlib import Path

from steering.eval.axbench_bpb.bpb_cache import BpbCache, reconcile_cache
from steering.eval.axbench_bpb.parquet_writer import write_bpb_score_rows
from steering.eval.axbench_bpb.types import BpbScoreRow


def _make_row(
    method_name: str = "m",
    concept_id: int = 1,
    sample_id: str = "s",
    coefficient_str: str = "0",
    target: str = "decoder_only",
    bpb: float = 1.5,
    n_tokens: int = 10,
    n_bytes: int = 40,
    truncated: bool = False,
) -> BpbScoreRow:
    """Mirrors the helper from test_cache_reconciliation.py."""
    return BpbScoreRow(
        method_name=method_name,
        concept_id=concept_id,
        mode="positive",
        content_class="concept_rich",
        target=target,
        coefficient_str=coefficient_str,
        sample_id=sample_id,
        bpb_intervened=bpb,
        n_output_tokens=n_tokens,
        n_output_bytes=n_bytes,
        truncated=truncated,
    )


def _snapshot_sqlite_state(cache_path: Path) -> tuple[int, list[tuple]]:
    """Return (row_count, sorted_rows) of the SQLite cache for equality
    checks. Sorting drops insertion-time non-determinism."""
    with BpbCache(cache_path) as cache:
        assert cache._conn is not None
        cur = cache._conn.execute(
            "SELECT method_name, concept_id, sample_id, coefficient_str, target, "
            "bpb, n_output_tokens, n_output_bytes, truncated FROM bpb_scores"
        )
        rows = sorted(cur.fetchall())
    return len(rows), rows


def test_reconcile_idempotent(tmp_path: Path):
    """Two reconciles in a row produce identical SQLite state."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows = [
        _make_row(sample_id="a", bpb=1.0),
        _make_row(sample_id="b", bpb=2.0),
        _make_row(sample_id="c", bpb=3.0),
    ]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    state_after_first = _snapshot_sqlite_state(cache_path)

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    state_after_second = _snapshot_sqlite_state(cache_path)

    assert state_after_first == state_after_second, (
        "reconcile_cache is not idempotent: SQLite state diverged across calls"
    )
    assert state_after_first[0] == 3  # row count


def test_reconcile_drops_sqlite_entries_for_deleted_parquet(tmp_path: Path):
    """Populate cache, write parquet, reconcile, manually rm parquet,
    reconcile again — SQLite must drop the orphaned rows."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    parquet_path = rank_dir / "batch_000000.parquet"
    rows = [
        _make_row(sample_id="will_be_deleted", bpb=9.0),
        _make_row(sample_id="also_will_be_deleted", bpb=8.0),
    ]
    write_bpb_score_rows(rows, parquet_path)
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    # Sanity: cache has both entries
    with BpbCache(cache_path) as cache:
        assert cache.has(
            method_name="m", concept_id=1, sample_id="will_be_deleted",
            coefficient_str="0", target="decoder_only",
        )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="also_will_be_deleted",
            coefficient_str="0", target="decoder_only",
        )

    # Manually delete the parquet — simulates external cleanup
    os.remove(parquet_path)
    assert not parquet_path.exists()

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        assert not cache.has(
            method_name="m", concept_id=1, sample_id="will_be_deleted",
            coefficient_str="0", target="decoder_only",
        )
        assert not cache.has(
            method_name="m", concept_id=1, sample_id="also_will_be_deleted",
            coefficient_str="0", target="decoder_only",
        )
        assert cache._conn is not None
        cur = cache._conn.execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 0


def test_reconcile_reseeds_sqlite_from_parquet_after_db_loss(tmp_path: Path):
    """Write parquet + populate cache. Manually delete the SQLite file.
    Reconcile must rebuild SQLite from the surviving parquet rows."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    rows = [
        _make_row(sample_id="recoverable_a", bpb=1.1),
        _make_row(sample_id="recoverable_b", bpb=2.2),
        _make_row(sample_id="recoverable_c", bpb=3.3),
    ]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    assert cache_path.exists()

    # Simulate disk loss / corruption / external rm of the SQLite cache
    os.remove(cache_path)
    # SQLite WAL companion files may also exist — clean up to mimic full loss
    for ext in ("-wal", "-shm", "-journal"):
        side = Path(str(cache_path) + ext)
        if side.exists():
            os.remove(side)
    assert not cache_path.exists()

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        for sid in ("recoverable_a", "recoverable_b", "recoverable_c"):
            assert cache.has(
                method_name="m", concept_id=1, sample_id=sid,
                coefficient_str="0", target="decoder_only",
            ), f"Sample {sid} not reseeded from parquet after DB loss"
        assert cache._conn is not None
        cur = cache._conn.execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 3


def test_cache_has_check_is_correct_after_concurrent_writes(tmp_path: Path):
    """has() reflects put() immediately within the same connection +
    empty-cache returns False as the negative baseline."""
    cache_path = tmp_path / "rank_000.sqlite"

    # Negative baseline: empty cache
    with BpbCache(cache_path) as cache:
        assert not cache.has(
            method_name="m", concept_id=1, sample_id="never_inserted",
            coefficient_str="0", target="decoder_only",
        )

    # Within same connection: put() is followed by has() returning True
    with BpbCache(cache_path) as cache:
        cache.put(
            method_name="m", concept_id=42, sample_id="just_inserted",
            coefficient_str="1.5", target="encdec",
            bpb=4.4, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        assert cache.has(
            method_name="m", concept_id=42, sample_id="just_inserted",
            coefficient_str="1.5", target="encdec",
        )
        # Different key still returns False
        assert not cache.has(
            method_name="m", concept_id=42, sample_id="different_sample",
            coefficient_str="1.5", target="encdec",
        )


def test_reconcile_handles_empty_rank_dir(tmp_path: Path):
    """Existing rank_dir with zero parquets reconciles to an empty cache.
    Distinct from the missing-dir case (already covered) — this exercises
    the glob('*.parquet') zero-match path."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()  # exists but empty
    cache_path = tmp_path / "rank_000.sqlite"

    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        assert cache._conn is not None
        cur = cache._conn.execute("SELECT COUNT(*) FROM bpb_scores")
        assert cur.fetchone()[0] == 0


def test_cache_put_overwrites_existing_key(tmp_path: Path):
    """Second put() with the same primary key overwrites the row
    (most-recent wins) instead of raising IntegrityError. Verify by
    reading back the bpb value via the underlying connection."""
    cache_path = tmp_path / "rank_000.sqlite"
    with BpbCache(cache_path) as cache:
        cache.put(
            method_name="m", concept_id=7, sample_id="overwrite_me",
            coefficient_str="2.0", target="decoder_only",
            bpb=10.0, n_output_tokens=5, n_output_bytes=20, truncated=False,
        )
        cache.put(  # same key, different bpb
            method_name="m", concept_id=7, sample_id="overwrite_me",
            coefficient_str="2.0", target="decoder_only",
            bpb=99.0, n_output_tokens=15, n_output_bytes=60, truncated=True,
        )
        # has() still says True
        assert cache.has(
            method_name="m", concept_id=7, sample_id="overwrite_me",
            coefficient_str="2.0", target="decoder_only",
        )
        # And the stored value is the second write
        assert cache._conn is not None
        cur = cache._conn.execute(
            "SELECT bpb, n_output_tokens, n_output_bytes, truncated "
            "FROM bpb_scores WHERE sample_id=?", ("overwrite_me",),
        )
        row = cur.fetchone()
        assert row == (99.0, 15, 60, 1)
        # And only one row exists for this key (no duplicate insert)
        cur = cache._conn.execute(
            "SELECT COUNT(*) FROM bpb_scores WHERE sample_id=?", ("overwrite_me",),
        )
        assert cur.fetchone()[0] == 1


def test_reconcile_preserves_data_integrity_under_partial_writes(tmp_path: Path):
    """Two parquet shards with overlapping content_class / different
    sample_ids: reconcile picks up rows from BOTH shards with no
    duplicates and no missing keys."""
    rank_dir = tmp_path / "rank_000"
    rank_dir.mkdir()
    cache_path = tmp_path / "rank_000.sqlite"

    # First shard
    shard_a_rows = [
        _make_row(sample_id="A1", bpb=1.0),
        _make_row(sample_id="A2", bpb=2.0),
    ]
    write_bpb_score_rows(shard_a_rows, rank_dir / "batch_000000.parquet")
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    # Second shard adds new samples (B1, B2). A* unchanged — emulates
    # partial-progress resume after the first shard already flushed.
    shard_b_rows = [
        _make_row(sample_id="B1", bpb=3.0),
        _make_row(sample_id="B2", bpb=4.0),
    ]
    write_bpb_score_rows(shard_b_rows, rank_dir / "batch_000001.parquet")
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    with BpbCache(cache_path) as cache:
        for sid in ("A1", "A2", "B1", "B2"):
            assert cache.has(
                method_name="m", concept_id=1, sample_id=sid,
                coefficient_str="0", target="decoder_only",
            ), f"Sample {sid} missing after partial-write reconcile"
        assert cache._conn is not None
        cur = cache._conn.execute("SELECT COUNT(*) FROM bpb_scores")
        # Exactly 4 rows, no duplicates
        assert cur.fetchone()[0] == 4
        # And no key appears twice
        cur = cache._conn.execute(
            "SELECT sample_id, COUNT(*) FROM bpb_scores GROUP BY sample_id "
            "HAVING COUNT(*) > 1"
        )
        assert cur.fetchall() == []


def test_cache_keys_are_namespace_isolated_by_target(tmp_path: Path):
    """Same (method_name, concept_id, sample_id, coefficient_str) but
    different target → both rows coexist independently. This pins the
    target column's role in the composite primary key."""
    cache_path = tmp_path / "rank_000.sqlite"
    with BpbCache(cache_path) as cache:
        cache.put(
            method_name="m", concept_id=1, sample_id="shared_sample",
            coefficient_str="0", target="decoder_only",
            bpb=1.0, n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        cache.put(
            method_name="m", concept_id=1, sample_id="shared_sample",
            coefficient_str="0", target="encdec",
            bpb=2.0, n_output_tokens=20, n_output_bytes=80, truncated=True,
        )
        # Both present — same logical key, different target column
        assert cache.has(
            method_name="m", concept_id=1, sample_id="shared_sample",
            coefficient_str="0", target="decoder_only",
        )
        assert cache.has(
            method_name="m", concept_id=1, sample_id="shared_sample",
            coefficient_str="0", target="encdec",
        )
        # Total rows = 2
        assert cache._conn is not None
        cur = cache._conn.execute(
            "SELECT COUNT(*) FROM bpb_scores WHERE sample_id=?", ("shared_sample",),
        )
        assert cur.fetchone()[0] == 2
        # And each target's bpb is its own value (no cross-contamination)
        cur = cache._conn.execute(
            "SELECT target, bpb FROM bpb_scores WHERE sample_id=? ORDER BY target",
            ("shared_sample",),
        )
        rows = cur.fetchall()
        assert rows == [("decoder_only", 1.0), ("encdec", 2.0)]
