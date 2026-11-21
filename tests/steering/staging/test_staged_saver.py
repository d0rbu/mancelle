# tests/steering/staging/test_staged_saver.py
from pathlib import Path
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.staging.saver as sv
from steering.feature_io import load_features, save_features
from steering.refined_types import PositiveInt
from steering.staging import BackgroundSaver
from steering.staging.saver import StagedSaver
from tests.steering.test_feature_io import _make_shard_dict

_DEPTH_1 = PositiveInt.parse(1)
_DEPTH_2 = PositiveInt.parse(2)
_DEPTH_3 = PositiveInt.parse(3)


def _assert_tensor_metadata_equal(a: dict, b: dict):
    for k in ("offsets", "doc_ids", "segment_indices", "segment_positions",
              "segment_char_starts", "segment_char_ends", "doc_token_counts"):
        assert t.equal(a[k], b[k]), f"{k} differs"
    off = a["offsets"]
    for s in range(off.numel() - 1):
        lo, hi = int(off[s]), int(off[s + 1])
        # Stored order (feature_id-ascending, mags co-sorted) — compare directly,
        # NOT via independent sorts, so an id<->mag mispairing cannot pass.
        assert t.equal(a["feature_ids"][lo:hi], b["feature_ids"][lo:hi])
        assert t.equal(a["feature_mags"][lo:hi], b["feature_mags"][lo:hi])


def test_staged_output_is_tensor_identical_to_direct_save(tmp_path):
    data = _make_shard_dict()
    scratch = tmp_path / "scratch" / "rank0" / "out"
    final_dir = tmp_path / "out"
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=final_dir,
        depth=_DEPTH_2,
        zstd_level=3,
    ) as s:
        s.submit(dict(data), str(final_dir / "shard_00000_features"))
    staged = load_features(str(final_dir / "shard_00000_features.safetensors.zst"))
    ref_path = save_features(dict(data), str(tmp_path / "ref" / "shard_00000_features"), zstd_level=3)
    _assert_tensor_metadata_equal(staged, load_features(str(ref_path)))


def test_drain_completes_in_flight_without_closing(tmp_path):
    """saver.drain() waits for in-flight save+relocate (so the GPU-idle tail can be
    measured inside the profiler region) and leaves the saver usable. (Finding #3.)"""
    data = _make_shard_dict()
    scratch = tmp_path / "scratch" / "rank0" / "out"
    final_dir = tmp_path / "out"
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=final_dir,
        depth=_DEPTH_3,
        zstd_level=3,
    ) as s:
        s.submit(dict(data), str(final_dir / "shard_00000_features"))
        s.drain()
        assert s.in_flight == 0
        assert (final_dir / "shard_00000_features.safetensors.zst").exists()  # relocated
        s.submit(dict(data), str(final_dir / "shard_00001_features"))  # still usable
        s.drain()
        assert (final_dir / "shard_00001_features.safetensors.zst").exists()


def test_peak_local_bytes_bounded_by_depth(tmp_path, monkeypatch):
    data = _make_shard_dict()
    scratch = tmp_path / "scratch" / "rank0" / "out"
    final_dir = tmp_path / "out"
    peak = {"v": 0}
    real = sv.relocate
    def counting(src, dst):
        peak["v"] = max(peak["v"], len(list(Path(scratch).glob("*.zst"))))
        real(src, dst)
    monkeypatch.setattr(sv, "relocate", counting)
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=final_dir,
        depth=_DEPTH_2,
        zstd_level=3,
    ) as s:
        for i in range(6):
            s.submit(dict(data), str(final_dir / f"shard_{i:05d}_features"))
    assert peak["v"] <= 2


def test_startup_sweeps_stale_scratch(tmp_path):
    scratch = tmp_path / "scratch" / "rank0" / "out"
    scratch.mkdir(parents=True)
    (scratch / "leftover_features.safetensors.zst").write_bytes(b"stale")
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=tmp_path / "out",
        depth=_DEPTH_1,
        zstd_level=3,
    ):
        pass
    assert not (scratch / "leftover_features.safetensors.zst").exists()


def test_exception_in_save_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "save_features", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("save boom")))
    with pytest.raises(RuntimeError, match="save boom"):
        with StagedSaver(
            scratch_dir=tmp_path / "s",
            final_dir=tmp_path / "o",
            depth=_DEPTH_1,
        ) as s:
            s.submit({"x": 1}, str(tmp_path / "o" / "shard_00000_features"))


def test_min_free_gb_gates_save(tmp_path, monkeypatch):
    calls: list[tuple[Path, dict]] = []

    def record_guard(path, **kwargs):
        calls.append((Path(path), kwargs))

    monkeypatch.setattr(sv, "wait_for_free_space", record_guard)
    scratch = tmp_path / "s"
    final_dir = tmp_path / "o"
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=final_dir,
        depth=_DEPTH_1,
        zstd_level=3,
        min_free_gb=5.0,
    ) as s:
        s.submit(_make_shard_dict(), str(tmp_path / "o" / "shard_00000_features"))
    assert [path for path, _ in calls] == [scratch, final_dir]
    assert calls[0][1] == {"min_free_gb": 5.0}
    assert calls[1][1] == {"min_free_gb": 5.0, "resource": "destination"}


def test_destination_guard_failure_prevents_relocate(tmp_path, monkeypatch):
    scratch = tmp_path / "s"
    final_dir = tmp_path / "o"
    calls: list[Path] = []

    def fail_destination_guard(path, **kwargs):
        calls.append(Path(path))
        if kwargs.get("resource") == "destination":
            raise RuntimeError("destination guard failed")

    monkeypatch.setattr(sv, "wait_for_free_space", fail_destination_guard)
    monkeypatch.setattr(sv, "relocate", lambda *a, **k: pytest.fail("relocate must not run"))
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=final_dir,
        depth=_DEPTH_1,
        min_free_gb=1.0,
    ) as s:
        with pytest.raises(RuntimeError, match="destination guard failed"):
            s._save_and_move(_make_shard_dict(), "shard_00000_features")
        assert list(scratch.glob("*.zst")), "scratch output should remain for diagnosis"
    assert calls == [scratch, final_dir]
    assert not list(final_dir.glob("*.zst"))


def test_zero_min_free_disables_both_runtime_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sv,
        "wait_for_free_space",
        lambda *a, **k: pytest.fail("free-space guard must be disabled at zero floor"),
    )
    final_dir = tmp_path / "o"
    with StagedSaver(
        scratch_dir=tmp_path / "s",
        final_dir=final_dir,
        depth=_DEPTH_1,
        min_free_gb=0.0,
    ) as s:
        s._save_and_move(_make_shard_dict(), "shard_00000_features")
    assert list(final_dir.glob("*.zst"))


def test_exit_sweep_removes_leftover(tmp_path):
    scratch = tmp_path / "scratch" / "rank0" / "out"
    with StagedSaver(
        scratch_dir=scratch,
        final_dir=tmp_path / "out",
        depth=_DEPTH_1,
        zstd_level=3,
    ):
        (scratch / "stray.tmp").write_bytes(b"leftover")  # appears mid-run
    assert not (scratch / "stray.tmp").exists()  # swept on exit


def test_satisfies_background_saver_protocol(tmp_path):
    with StagedSaver(
        scratch_dir=tmp_path / "s",
        final_dir=tmp_path / "o",
        depth=_DEPTH_1,
    ) as s:
        assert isinstance(s, BackgroundSaver)


def test_exit_finally_sweeps_even_when_drain_raises(tmp_path, monkeypatch):
    """Non-vacuous teeth for __exit__'s `finally: sweep_dir`: a save that WRITES the
    scratch .zst then fails in relocate leaves a real leftover; the drain re-raises and
    the finally must still sweep it (removing the finally would leave the .zst behind)."""
    data = _make_shard_dict()
    scratch = tmp_path / "scratch" / "rank0" / "out"
    final_dir = tmp_path / "out"
    monkeypatch.setattr(sv, "relocate", lambda src, dst: (_ for _ in ()).throw(OSError("relocate boom")))
    with pytest.raises(OSError, match="relocate boom"):
        with StagedSaver(
            scratch_dir=scratch,
            final_dir=final_dir,
            depth=_DEPTH_2,
            zstd_level=3,
        ) as s:
            s.submit(dict(data), str(final_dir / "shard_00000_features"))
    assert list(scratch.glob("*.zst")) == []  # save_features' scratch file swept despite the drain raise


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_rejects_bad_depth_before_touching_scratch(tmp_path, bad: Any) -> None:
    scratch = tmp_path / "scratch"
    with pytest.raises(BeartypeCallHintParamViolation):
        StagedSaver(
            scratch_dir=scratch,
            final_dir=tmp_path / "out",
            depth=bad,
        )
    assert not scratch.exists()
