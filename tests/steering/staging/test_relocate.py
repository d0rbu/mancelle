# tests/steering/staging/test_relocate.py
import errno
import shutil
from pathlib import Path
from typing import Any

import pytest

import steering.staging.relocate as rl
from steering.staging.relocate import (
    DestinationSpaceError,
    assert_startup_free_space,
    coordinate_free_space_check,
    prepare_scratch_dir,
    relocate,
    sweep_dir,
    wait_for_free_space,
)


def _coord(run_scratch: Path, local_rank: int, **over: Any) -> None:
    kw: dict[str, Any] = {
        "local_rank": local_rank,
        "run_id": "rid",
        "output_depth": 2,
        "est_shard_gb": 1.0,
        "local_world_size": 1,
        "min_free_gb": 0.0,
        "timeout_s": 0.1,
        "poll_s": 0.01,
    }
    kw.update(over)
    coordinate_free_space_check(run_scratch, **kw)


def _write(p: Path, b: bytes) -> Path:
    p.write_bytes(b)
    return p


def test_same_fs_uses_rename_no_copy(tmp_path, monkeypatch):
    src = _write(tmp_path / "a.bin", b"hello")
    final = tmp_path / "out" / "a.bin"
    called = {"copyfile": 0}
    monkeypatch.setattr(shutil, "copyfile", lambda *a, **k: called.__setitem__("copyfile", 1))
    relocate(src, final)
    assert final.read_bytes() == b"hello" and not src.exists() and called["copyfile"] == 0


def test_cross_fs_copies_fsyncs_renames(tmp_path, monkeypatch):
    src = _write(tmp_path / "a.bin", b"payload-bytes")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr("steering.staging.relocate._same_filesystem", lambda a, b: False)
    relocate(src, final)
    assert final.read_bytes() == b"payload-bytes"  # O2 bytes preserved
    assert not src.exists()
    assert not list(final.parent.glob("*.tmp.*"))  # no glob-visible temp


def test_cross_fs_failure_cleans_temp_and_retains_source(tmp_path, monkeypatch):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr("steering.staging.relocate._same_filesystem", lambda a, b: False)
    monkeypatch.setattr(rl.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("rename fail")))
    with pytest.raises(OSError):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"          # source RETAINED for resume
    assert not final.exists()                      # never a partial final
    assert not list(final.parent.glob("*"))        # temp cleaned


def test_cross_fs_non_oserror_still_cleans_temp_and_retains_source(tmp_path, monkeypatch):
    class StopRelocation(BaseException):
        pass

    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr(rl, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        rl.shutil,
        "copyfile",
        lambda *a, **k: (_ for _ in ()).throw(StopRelocation()),
    )
    with pytest.raises(StopRelocation):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"
    assert not final.exists() and not list(final.parent.glob("*.tmp.*"))


def test_never_cross_fs_os_replace(tmp_path, monkeypatch):
    src = _write(tmp_path / "a.bin", b"x")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr("steering.staging.relocate._same_filesystem", lambda a, b: False)
    seen = []
    real = rl.os.replace
    monkeypatch.setattr(rl.os, "replace", lambda a, b: (seen.append((a, b)), real(a, b))[1])
    relocate(src, final)
    a, b = seen[-1]
    assert Path(b) == final and Path(a).parent == final.parent  # rename within dest FS only


def test_same_fs_exdev_falls_back_to_copy(tmp_path, monkeypatch):
    """st_dev matched but the mount rejects a cross-directory rename (union/overlay/
    FUSE) -> os.replace raises EXDEV; relocate must fall back to the copy path, not
    abort. (Finding #13.)"""
    src = _write(tmp_path / "a.bin", b"exdev-payload")
    final = tmp_path / "out" / "a.bin"  # same FS as src -> takes the rename branch
    real_replace = rl.os.replace

    def fake_replace(a, b):
        if Path(a) == src:  # only the direct src->final rename hits EXDEV
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(a, b)  # the temp->final rename within dest still works

    monkeypatch.setattr(rl.os, "replace", fake_replace)
    relocate(src, final)
    assert final.read_bytes() == b"exdev-payload"
    assert not src.exists()
    assert not list(final.parent.glob("*.tmp.*"))  # temp cleaned, no orphan


def test_sweep_dir_fails_fast_by_default_but_swallows_on_request(tmp_path):
    """sweep_dir RE-RAISES a non-FileNotFoundError listing error by default (fail-fast at
    startup), but swallows it under swallow_errors=True (the teardown `finally` context,
    where masking the real exception is the worse outcome). iterdir on a FILE raises
    NotADirectoryError. (Findings #8 / #15.)"""
    not_a_dir = _write(tmp_path / "scratch_is_a_file", b"x")
    with pytest.raises(OSError):
        sweep_dir(not_a_dir)  # default: fail fast
    sweep_dir(not_a_dir, swallow_errors=True)  # teardown context: quiet, no raise


def test_prepare_scratch_dir_creates_sweeps_and_fails_fast(tmp_path):
    """prepare_scratch_dir mkdirs + sweeps stale files; fails fast on a misconfigured
    path (a regular file where a dir is expected). (Finding #9 / #8.)"""
    scratch = tmp_path / "scr"
    (scratch).mkdir()
    (scratch / "stale.zst").write_bytes(b"old")
    out = prepare_scratch_dir(scratch)
    assert out == scratch and scratch.is_dir()
    assert not (scratch / "stale.zst").exists()  # swept
    with pytest.raises(OSError):  # a file where a dir is expected -> FileExistsError from mkdir
        prepare_scratch_dir(_write(tmp_path / "a_file", b"x"))


def test_wait_for_free_space_warns_while_blocked(tmp_path, monkeypatch):
    """A long block must not be a silent freeze — wait_for_free_space logs while it
    waits. (Finding #12.)"""
    monkeypatch.setattr(rl, "free_gb", lambda p: 0.1)  # never recovers
    warnings: list = []
    monkeypatch.setattr(rl.logger, "warning", lambda *a, **k: warnings.append((a, k)))
    with pytest.raises(RuntimeError, match="below"):
        wait_for_free_space(tmp_path, min_free_gb=6.0, timeout_s=0.05, poll_s=0.01, warn_interval_s=0.0)
    assert warnings, "expected a 'blocking' warning while waiting for free space"


def test_wait_for_free_space_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 0.1)
    with pytest.raises(RuntimeError, match="below"):
        wait_for_free_space(tmp_path, min_free_gb=6.0, timeout_s=0.05, poll_s=0.01)


def test_wait_for_free_space_destination_message(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 0.1)
    with pytest.raises(RuntimeError, match="destination"):
        wait_for_free_space(
            tmp_path,
            min_free_gb=6.0,
            timeout_s=0.05,
            poll_s=0.01,
            resource="destination",
        )


def test_sweep_dir_removes_files(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"1")
    (tmp_path / "y.zst").write_bytes(b"2")
    sweep_dir(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_sweep_dir_preserves_subdirectories(tmp_path):
    """sweep_dir must remove only top-level FILES, never recurse/rmtree — the staging
    scratch holds rank{r}/{out,in} subdirs that a startup sweep must not destroy."""
    (tmp_path / "leftover.zst").write_bytes(b"1")
    sub = tmp_path / "rank0"
    sub.mkdir()
    (sub / "nested.zst").write_bytes(b"2")
    sweep_dir(tmp_path)
    assert not (tmp_path / "leftover.zst").exists()  # top-level file gone
    assert sub.is_dir() and (sub / "nested.zst").exists()  # subtree untouched


def test_sweep_dir_missing_directory_is_noop(tmp_path):
    sweep_dir(tmp_path / "does_not_exist")  # FileNotFoundError early-return, must not raise


def test_assert_startup_free_space_raises_below_need_passes_above(tmp_path, monkeypatch):
    # need = max(2,1)*1.0*max(3,1) + 0.5 = 6.5
    monkeypatch.setattr(rl, "free_gb", lambda p: 6.0)
    with pytest.raises(RuntimeError, match="needs"):
        assert_startup_free_space(tmp_path, output_depth=2, est_shard_gb=1.0, local_world_size=3, min_free_gb=0.5)
    monkeypatch.setattr(rl, "free_gb", lambda p: 7.0)
    assert_startup_free_space(tmp_path, output_depth=2, est_shard_gb=1.0, local_world_size=3, min_free_gb=0.5)


def test_assert_startup_free_space_multiplies_by_ranks(tmp_path, monkeypatch):
    """Teeth on the ×local_world_size term: 5 GB fits one rank's wave (2) but not 4 ranks' (8)."""
    monkeypatch.setattr(rl, "free_gb", lambda p: 5.0)
    assert_startup_free_space(tmp_path, output_depth=2, est_shard_gb=1.0, local_world_size=1, min_free_gb=0.0)
    with pytest.raises(RuntimeError):
        assert_startup_free_space(tmp_path, output_depth=2, est_shard_gb=1.0, local_world_size=4, min_free_gb=0.0)


def test_assert_startup_free_space_max_guards_floor_to_one(tmp_path, monkeypatch):
    """output_depth=0 / local_world_size=0 are floored to 1, so need = 1*1*1 + 0 = 1."""
    monkeypatch.setattr(rl, "free_gb", lambda p: 0.5)
    with pytest.raises(RuntimeError):
        assert_startup_free_space(tmp_path, output_depth=0, est_shard_gb=1.0, local_world_size=0, min_free_gb=0.0)


def test_same_fs_non_exdev_oserror_reraises_without_copy(tmp_path, monkeypatch):
    """A same-FS os.replace failure that is NOT EXDEV (e.g. EACCES) must propagate, NOT
    fall through to the copy path. Teeth on the `if e.errno != errno.EXDEV: raise` guard."""
    src = _write(tmp_path / "a.bin", b"x")
    final = tmp_path / "out" / "a.bin"
    copied = {"n": 0}
    monkeypatch.setattr(shutil, "copyfile", lambda *a, **k: copied.__setitem__("n", 1))
    monkeypatch.setattr(rl.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError(errno.EACCES, "denied")))
    with pytest.raises(OSError) as ei:
        relocate(src, final)
    assert ei.value.errno == errno.EACCES
    assert copied["n"] == 0  # never reached the copy fallback


@pytest.mark.parametrize("space_errno", [errno.ENOSPC, errno.EDQUOT])
def test_same_fs_destination_space_error_names_destination_and_retains_source(
    tmp_path, monkeypatch, space_errno
):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "out" / "a.bin"
    monkeypatch.setattr(
        rl.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError(space_errno, "destination full")),
    )
    with pytest.raises(DestinationSpaceError, match=f"destination {final.parent}") as ei:
        relocate(src, final)
    assert ei.value.errno == space_errno
    assert isinstance(ei.value.__cause__, OSError)
    assert ei.value.__cause__.errno == space_errno
    assert src.read_bytes() == b"keepme" and not final.exists()


@pytest.mark.parametrize("space_errno", [errno.ENOSPC, errno.EDQUOT])
def test_cross_fs_copy_space_error_cleans_temp_and_retains_source(
    tmp_path, monkeypatch, space_errno
):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr(rl, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        rl.shutil,
        "copyfile",
        lambda *a, **k: (_ for _ in ()).throw(OSError(space_errno, "destination full")),
    )
    with pytest.raises(DestinationSpaceError, match=f"destination {final.parent}"):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"
    assert not final.exists() and not list(final.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("space_errno", [errno.ENOSPC, errno.EDQUOT])
def test_cross_fs_sync_space_error_cleans_temp_and_retains_source(
    tmp_path, monkeypatch, space_errno
):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr(rl, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        rl.os,
        "fsync",
        lambda *a, **k: (_ for _ in ()).throw(OSError(space_errno, "destination full")),
    )
    with pytest.raises(DestinationSpaceError, match=f"destination {final.parent}"):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"
    assert not final.exists() and not list(final.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("space_errno", [errno.ENOSPC, errno.EDQUOT])
def test_cross_fs_publication_space_error_cleans_temp_and_retains_source(
    tmp_path, monkeypatch, space_errno
):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr(rl, "_same_filesystem", lambda a, b: False)
    monkeypatch.setattr(
        rl.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError(space_errno, "destination full")),
    )
    with pytest.raises(DestinationSpaceError, match=f"destination {final.parent}"):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"
    assert not final.exists() and not list(final.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("space_errno", [errno.ENOSPC, errno.EDQUOT])
def test_destination_mkdir_space_error_is_explicit_and_retains_source(
    tmp_path, monkeypatch, space_errno
):
    src = _write(tmp_path / "a.bin", b"keepme")
    final = tmp_path / "dest" / "a.bin"
    real_mkdir = Path.mkdir

    def fail_destination_mkdir(self, *args, **kwargs):
        if self == final.parent:
            raise OSError(space_errno, "destination full")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_destination_mkdir)
    with pytest.raises(DestinationSpaceError, match=f"destination {final.parent}"):
        relocate(src, final)
    assert src.read_bytes() == b"keepme"


def test_cross_fs_src_unlink_failure_warns_not_raises(tmp_path, monkeypatch):
    """After a successful cross-FS move, a failure to unlink the source must warn and
    NOT raise (the move already succeeded; the shard is at its destination)."""
    src = _write(tmp_path / "a.bin", b"payload")
    final = tmp_path / "dest" / "a.bin"
    monkeypatch.setattr("steering.staging.relocate._same_filesystem", lambda a, b: False)
    warns = []
    monkeypatch.setattr(rl.logger, "warning", lambda *a, **k: warns.append(1))
    real_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError(errno.EDQUOT, "unlink quota"))
                        if self == src else real_unlink(self, *a, **k))
    relocate(src, final)  # must NOT raise
    assert final.read_bytes() == b"payload" and warns


def test_wait_for_free_space_returns_immediately_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 100.0)
    wait_for_free_space(tmp_path, min_free_gb=6.0, timeout_s=0.01, poll_s=0.01)  # loop body skipped, no raise


def test_coordinate_rank0_passes_and_signals_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 100.0)  # plenty
    run_scratch = tmp_path / "run"
    destination = tmp_path / "destination"
    run_scratch.mkdir()
    destination.mkdir()
    _coord(run_scratch, 0, destination_dir=destination)  # checks both filesystems
    assert (run_scratch / ".startup_ok.rid").exists()


def test_coordinate_rank0_shortage_signals_fail_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 0.1)  # below need
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    with pytest.raises(RuntimeError):
        _coord(run_scratch, 0)  # rank 0: shortage -> writes fail sentinel + raises
    assert (run_scratch / ".startup_fail.rid").exists()  # so waiting ranks fail fast


def test_coordinate_rank0_checks_destination_and_signals_diagnosis(tmp_path, monkeypatch):
    run_scratch = tmp_path / "run"
    destination = tmp_path / "destination"
    run_scratch.mkdir()
    destination.mkdir()

    def fake_free_gb(path):
        return 0.1 if Path(path) == destination else 100.0

    monkeypatch.setattr(rl, "free_gb", fake_free_gb)
    with pytest.raises(RuntimeError, match="destination"):
        _coord(run_scratch, 0, destination_dir=destination)
    fail = run_scratch / ".startup_fail.rid"
    assert "destination" in fail.read_text()


def test_coordinate_nonzero_propagates_destination_diagnosis(tmp_path):
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    (run_scratch / ".startup_fail.rid").write_text(
        "staging: destination /shared/out has 0.1 GB free but needs ~1.0 GB"
    )
    with pytest.raises(RuntimeError, match="destination /shared/out"):
        _coord(run_scratch, 1)


def test_coordinate_rank0_clears_stale_sentinels(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "free_gb", lambda p: 100.0)
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    (run_scratch / ".startup_ok.old").write_text("stale")
    (run_scratch / ".startup_fail.older").write_text("stale")
    _coord(run_scratch, 0)
    assert not (run_scratch / ".startup_ok.old").exists()  # prior-launch sentinels cleared
    assert not (run_scratch / ".startup_fail.older").exists()
    assert (run_scratch / ".startup_ok.rid").exists()  # fresh verdict written


def test_coordinate_nonzero_proceeds_on_ok(tmp_path):
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    (run_scratch / ".startup_ok.rid").write_text("ok")  # rank 0 already signaled OK
    _coord(run_scratch, 1)  # returns, no raise


def test_coordinate_nonzero_fails_on_fail_signal(tmp_path):
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    (run_scratch / ".startup_fail.rid").write_text("nope")  # rank 0 signaled shortage
    with pytest.raises(RuntimeError, match="rank 0 reported insufficient"):
        _coord(run_scratch, 1)


def test_coordinate_nonzero_times_out_without_signal(tmp_path):
    run_scratch = tmp_path / "run"
    run_scratch.mkdir()
    with pytest.raises(RuntimeError, match="no free-space verdict"):
        _coord(run_scratch, 1, timeout_s=0.05)  # rank 0 never signaled


def test_tmp_path_for_is_unique_and_pid_tagged(tmp_path):
    """relocate's temp name carries the pid + a random uuid, so concurrent ranks
    moving to the same dir never collide on the temp, and it never glob-matches a
    real shard (the multi-rank-safety property the old AsyncSaver suite covered)."""
    final = tmp_path / "shard_00000_features.safetensors.zst"
    names = {str(rl._tmp_path_for(final)) for _ in range(50)}
    assert len(names) == 50  # all distinct (uuid component)
    sample = str(rl._tmp_path_for(final))
    assert f".{rl.os.getpid()}." in sample  # pid component -> cross-process uniqueness
    assert not sample.endswith("_features.safetensors.zst")  # never matches a real shard glob
