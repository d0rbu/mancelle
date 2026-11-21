# tests/steering/staging/test_staged_prefetcher.py
from pathlib import Path
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.refined_types import PositiveInt
from steering.staging.prefetcher import StagedPrefetcher

_DEPTH_1 = PositiveInt.parse(1)
_DEPTH_2 = PositiveInt.parse(2)
_DEPTH_3 = PositiveInt.parse(3)


def _touch(p: Path) -> str:
    p.write_bytes(b"x")
    return str(p)


def test_fifo_order_and_stage_then_delete(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"shard_{i}.bin").write_bytes(bytes([i]) * 10)
    def load_fn(source_path, local_path):
        assert Path(local_path).exists()  # staged copy present during load
        return (source_path, Path(local_path).read_bytes())
    results = []
    with StagedPrefetcher(
        scratch_dir=tmp_path / "scratch", depth=_DEPTH_3, load_fn=load_fn
    ) as pf:
        for i in range(3):
            pf.submit(str(src / f"shard_{i}.bin"))
        for _ in range(3):
            results.append(pf.get())
    assert [r[0] for r in results] == [str(src / f"shard_{i}.bin") for i in range(3)]  # FIFO
    assert [r[1] for r in results] == [bytes([i]) * 10 for i in range(3)]  # identical bytes
    assert list((tmp_path / "scratch").glob("*.bin")) == []  # deleted after load


def test_submit_beyond_depth_raises(tmp_path):
    with StagedPrefetcher(
        scratch_dir=tmp_path / "s", depth=_DEPTH_1, load_fn=lambda s, lp: 1
    ) as pf:
        pf.submit(_touch(tmp_path / "a.bin"))
        with pytest.raises(RuntimeError, match="depth"):
            pf.submit(_touch(tmp_path / "b.bin"))


def test_get_with_nothing_pending_raises(tmp_path):
    with StagedPrefetcher(
        scratch_dir=tmp_path / "s", depth=_DEPTH_1, load_fn=lambda s, lp: 1
    ) as pf:
        with pytest.raises(RuntimeError, match="nothing pending"):
            pf.get()


def test_load_failure_propagates_on_get(tmp_path):
    scratch = tmp_path / "s"
    def bad(source_path, local_path):
        raise ValueError("load boom")
    with StagedPrefetcher(scratch_dir=scratch, depth=_DEPTH_1, load_fn=bad) as pf:
        pf.submit(_touch(tmp_path / "a.bin"))
        with pytest.raises(ValueError, match="load boom"):
            pf.get()
        assert list(scratch.glob("*")) == []  # finally deleted the staged copy even on load failure


def test_rejects_bad_depth(tmp_path):
    scratch = tmp_path / "s"
    for bad in (0, -1, True, 1.5):
        invalid_depth: Any = bad
        with pytest.raises(BeartypeCallHintParamViolation):
            StagedPrefetcher(
                scratch_dir=scratch,
                depth=invalid_depth,
                load_fn=lambda s, lp: 1,
            )
        assert not scratch.exists()


def test_startup_sweeps_stale_scratch(tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir(parents=True)
    (scratch / "stale.parquet").write_bytes(b"old")  # crash leftover from a prior run
    with StagedPrefetcher(
        scratch_dir=scratch, depth=_DEPTH_1, load_fn=lambda s, lp: 1
    ):
        pass
    assert not (scratch / "stale.parquet").exists()  # swept at startup


def test_exit_drains_failed_pending_without_masking(tmp_path):
    """A pending job that fails must NOT propagate out of __exit__ (it would mask the
    real reason teardown was triggered) — __exit__ swallows it and still sweeps."""
    scratch = tmp_path / "s"
    def bad(source_path, local_path):
        raise ValueError("prefetch boom")
    with StagedPrefetcher(scratch_dir=scratch, depth=_DEPTH_2, load_fn=bad) as pf:
        pf.submit(_touch(tmp_path / "a.bin"))  # failing job left pending (never get()'d)
    assert list(scratch.glob("*")) == []  # swept; no exception escaped __exit__


def test_exit_swallows_baseexception_pending(tmp_path):
    """A pending job raising a non-Exception BaseException during teardown must still be
    swallowed (not masking the original) — __exit__ catches BaseException. (Finding #5.)"""
    class _Boom(BaseException):
        pass
    scratch = tmp_path / "s"
    def boom(source_path, local_path):
        raise _Boom("base boom")
    with StagedPrefetcher(scratch_dir=scratch, depth=_DEPTH_2, load_fn=boom) as pf:
        pf.submit(_touch(tmp_path / "a.bin"))  # left pending; __exit__ drains it
    assert list(scratch.glob("*")) == []  # swallowed + swept; _Boom did not escape __exit__


def test_copy_failure_surfaces_and_leaves_no_orphan(tmp_path):
    """A failed copyfile (missing source) surfaces at get() and leaves no scratch orphan —
    copyfile is inside the try/finally so a partial/failed copy is cleaned. (Finding #6.)"""
    scratch = tmp_path / "s"
    with StagedPrefetcher(
        scratch_dir=scratch, depth=_DEPTH_1, load_fn=lambda s, lp: 1
    ) as pf:
        pf.submit(str(tmp_path / "does_not_exist.parquet"))  # copyfile fails
        with pytest.raises(OSError):
            pf.get()
        assert list(scratch.glob("*")) == []  # finally cleaned, no orphan
