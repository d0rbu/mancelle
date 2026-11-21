"""When reconcile_cache encounters a corrupted parquet shard, the error
must clearly identify the offending file and suggest remediation, rather
than letting a generic pyarrow.lib.ArrowInvalid bubble up.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_bpb.bpb_cache import reconcile_cache
from steering.eval.axbench_bpb.parquet_writer import write_bpb_score_rows
from steering.eval.axbench_bpb.types import AxBenchBpbError, BpbScoreRow


def test_reconcile_cache_surfaces_corrupted_parquet(tmp_path: Path):
    """Plant a non-parquet file with a .parquet extension; assert
    reconcile raises AxBenchBpbError naming the file path.
    """
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    cache_path = tmp_path / "cache.sqlite"

    # Non-parquet bytes with a .parquet extension
    bad = rank_dir / "batch_000000.parquet"
    bad.write_bytes(b"this is not a parquet file at all")

    with pytest.raises(AxBenchBpbError, match=r"[Cc]orrupted.*parquet"):
        reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)


def test_reconcile_cache_includes_file_path_in_error(tmp_path: Path):
    """The error message must contain the offending file's basename so
    the operator can find it without reading a traceback.
    """
    rank_dir = tmp_path / "rank_1"
    rank_dir.mkdir()
    cache_path = tmp_path / "cache.sqlite"

    bad = rank_dir / "batch_000042.parquet"
    bad.write_bytes(b"\x00\x00\x00\x00")  # short / empty

    with pytest.raises(AxBenchBpbError) as exc_info:
        reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
    assert "batch_000042.parquet" in str(exc_info.value)


def test_reconcile_cache_skips_truly_missing_dir(tmp_path: Path):
    """Pre-existing behavior: missing rank_dir is a no-op (no error)."""
    cache_path = tmp_path / "cache.sqlite"
    # rank_dir doesn't exist — must be a no-op
    reconcile_cache(rank_dir=tmp_path / "nonexistent", cache_path=cache_path)
    # If it didn't raise, we're good.


def test_reconcile_cache_succeeds_on_well_formed_parquet(tmp_path: Path):
    """Sanity check: a real parquet shard reconciles without error."""
    rank_dir = tmp_path / "rank_0"
    rank_dir.mkdir()
    cache_path = tmp_path / "cache.sqlite"

    rows = [
        BpbScoreRow(
            method_name="m",
            concept_id=1,
            mode="positive",
            content_class="concept_rich",
            target="decoder_only",
            coefficient_str="0",
            sample_id="s",
            bpb_intervened=1.5,
            n_output_tokens=10,
            n_output_bytes=40,
            truncated=False,
        )
    ]
    write_bpb_score_rows(rows, rank_dir / "batch_000000.parquet")

    # Should not raise
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)
