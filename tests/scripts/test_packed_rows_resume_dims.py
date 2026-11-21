"""Issue #539: resume-safe manifest max dims via per-shard _dims/ companions.

A resumed (or idempotent re-run) precompute_packed_rows must return global max_n_seg/max_seg_size
reflecting ALL shards (existing + new), never the post-resume tail only and never 0. The fix
persists each shard's dims in a _dims/shard_NNNNN.json companion; resume seeds the running maxima
from the kept prefix's companions, with a warn+reduce-over-shards fallback for legacy dirs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import scripts.preprocess.precompute_packed_rows as m
from scripts.preprocess._packed_rows_profile import PackedRowsProfile
from scripts.preprocess.precompute_packed_rows import (
    _reduce_max_dims_over_shards,
    precompute_packed_rows,
)
from steering.overlay_io import glob_overlay_shards
from tests.scripts._packed_rows_fixtures import (
    SimpleTokenizer,
    _copy_shards,
    _make_feature_index,
    _make_precompute_env,
)


def _run(env, out: Path, **kw):
    out.mkdir(parents=True, exist_ok=True)
    return precompute_packed_rows(
        parquet_paths=env["parquet_paths"], tokenizer=SimpleTokenizer(),
        feature_index=_make_feature_index(env), context_len=32, output_dir=out,
        buffer_size=20, rows_per_shard=3, **kw,
    )


def test_reduce_max_dims_over_shards_matches_return(tmp_path):
    """The shared reducer over the written shards equals the run's returned global maxima."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "full"
    _rows, ret_mns, ret_mss = _run(env, out, compress_workers=4)
    shard_paths = glob_overlay_shards(out)
    assert len(shard_paths) >= 3
    red_mns, red_mss = _reduce_max_dims_over_shards(shard_paths)
    assert (red_mns, red_mss) == (ret_mns, ret_mss)


def test_dims_companion_written_per_shard_and_invisible_to_glob(tmp_path):
    """Each shard gets a _dims/shard_NNNNN.json with its true dims; the _dims/ subdir is invisible
    to glob_overlay_shards (so no consumer regresses)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "full"
    _run(env, out, compress_workers=4)
    shard_paths = glob_overlay_shards(out)
    assert len(shard_paths) >= 3
    # glob_overlay_shards is non-recursive + .zst-only ⇒ it never returns the _dims companions.
    assert all(p.name.endswith(".safetensors.zst") for p in shard_paths)
    assert (out / "_dims").is_dir()
    for sp in shard_paths:
        idx = int(sp.name.split("_")[1].split(".")[0])
        comp = out / "_dims" / f"shard_{idx:05d}.json"
        assert comp.is_file(), f"missing companion for shard {idx}"
        d = json.loads(comp.read_text())
        mns, mss = m._compute_shard_max_dims(m.load_packed_row_shard(sp))
        assert d == {"max_n_seg": mns, "max_seg_size": mss}


def test_idempotent_rerun_preserves_max_dims(tmp_path):
    """Re-running a COMPLETE dir with resume=true writes 0 new shards; the returned dims must stay
    correct (seeded from companions), never reset to 0 (issue #539 symptom 1)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "full"
    _rows, mns, mss = _run(env, out, compress_workers=4)
    assert mns > 0 and mss > 0
    _rows2, mns2, mss2 = _run(env, out, resume=True, compress_workers=4)
    assert (mns2, mss2) == (mns, mss), "idempotent re-run must preserve max dims, not zero them"


def test_crash_resume_seeds_true_max_not_tail_only(tmp_path, monkeypatch):
    """If the global max lives in an early (pre-crash) shard, resume must seed it from that shard's
    companion so the returned max is the TRUE max — not the tail-only re-pack (issue #539 symptom
    2). Force shard 0's dims large (the env makes later shards larger, so this isolates the bug)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    real = m._assemble_and_save

    def force_shard0_large(shard_idx, rows, prov, out_path, prov_path, zstd, dims_path):
        idx, n, s, timing = real(shard_idx, rows, prov, out_path, prov_path, zstd, dims_path)
        if idx == 0:  # overwrite the companion + report a large max for shard 0
            from nanochat.atomic_io import atomic_json_dump
            atomic_json_dump({"max_n_seg": 100, "max_seg_size": 50}, dims_path)
            return idx, 100, 50, timing
        return idx, n, s, timing

    monkeypatch.setattr(m, "_assemble_and_save", force_shard0_large)
    full = tmp_path / "full"
    _rows, full_mns, full_mss = _run(env, full, compress_workers=4)
    assert (full_mns, full_mss) == (100, 50)  # shard 0 holds the (forced) global max

    # Crash sim: keep only the contiguous prefix [0, 1] as COMPLETE shards
    # (packed + provenance + _dims companion); drop the tail.
    shard_paths = sorted(full.glob("shard_*.safetensors.zst"))
    assert len(shard_paths) >= 4
    crashed = tmp_path / "crashed"
    _copy_shards(full, crashed, [0, 1])

    # Resume WITHOUT the forcing patch (real worker for the re-packed tail).
    monkeypatch.setattr(m, "_assemble_and_save", real)
    _r, res_mns, res_mss = _run(env, crashed, resume=True, compress_workers=4)
    assert (res_mns, res_mss) == (100, 50), (
        "resume must seed the early shard's max from its companion; a tail-only reduction would "
        "drop it (teeth)"
    )


def test_legacy_resume_without_companions_falls_back_and_warns(tmp_path, caplog):
    """A dir written before #539 has no _dims/; resume must WARN and reduce over the existing
    shards (correct, slow) rather than understate (issue #539 §3.3)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    full = tmp_path / "full"
    _rows, mns, mss = _run(env, full, compress_workers=4)
    shutil.rmtree(full / "_dims")  # simulate a legacy (pre-#539) dir
    with caplog.at_level("WARNING"):
        _r, res_mns, res_mss = _run(env, full, resume=True, compress_workers=4)
    assert (res_mns, res_mss) == (mns, mss)
    assert any("dims" in r.message.lower() for r in caplog.records), "expected a legacy-fallback warning"


def test_gap_resume_seeds_from_contiguous_prefix_companions(tmp_path):
    """An O1 out-of-order crash can leave a gap (shards 0,1,3.. but not 2). Resume seeds from the
    contiguous prefix [0,1]'s companions and re-packs from shard 2. Teeth: force shard 0's
    companion to a max the (uniform) re-packed tail can't reproduce, so this fails if prefix-
    companion seeding were dropped. (Byte-identity of the re-filled gap is covered by
    test_resume_fills_gap_from_parallel_crash in test_packed_rows_parallel_compress.py.)"""
    env = _make_precompute_env(tmp_path, n_shards=4)
    full = tmp_path / "full"
    _run(env, full, compress_workers=4)
    names = sorted(p.name for p in full.glob("shard_*.safetensors.zst"))
    assert len(names) >= 5, "need >=5 shards for a meaningful middle gap"
    gap = tmp_path / "gap"
    # Copy every COMPLETE shard (packed + provenance + _dims) EXCEPT the gap at
    # index 2 — the gap an out-of-order pool can leave.
    kept = [i for i in (int(n.split("_")[1].split(".")[0]) for n in names) if i != 2]
    _copy_shards(full, gap, kept)
    # Teeth: shard 0 (in the kept prefix) carries a max the re-packed tail can't reproduce.
    (gap / "_dims" / "shard_00000.json").write_text(json.dumps({"max_n_seg": 100, "max_seg_size": 50}))
    _r, gap_mns, gap_mss = _run(env, gap, resume=True, compress_workers=4)
    assert (gap_mns, gap_mss) == (100, 50), (
        "gap-resume must seed from the contiguous-prefix companions; without seeding the re-packed "
        "tail would understate (drop shard 0's forced max)"
    )


def test_resume_from_shard_seeds_from_present_prefix_companions(tmp_path):
    """The EXPLICIT resume_from_shard branch also seeds global_max from the kept prefix's
    companions (the new tests otherwise drive only the contiguous resume=true branch). Force shard
    0's companion large; resume_from_shard over the present shards must surface it (seeding runs
    before the replay loop, so the data-exhausted early-return carries it too)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "full"
    _run(env, out, compress_workers=4)
    n_shards = len(sorted(out.glob("shard_*.safetensors.zst")))
    assert n_shards >= 3
    (out / "_dims" / "shard_00000.json").write_text(json.dumps({"max_n_seg": 100, "max_seg_size": 50}))
    _r, mns, mss = _run(env, out, resume_from_shard=n_shards, compress_workers=4)
    assert (mns, mss) == (100, 50), "resume_from_shard must seed from the present prefix companions"


def test_corrupt_companion_falls_back_not_crashes(tmp_path, caplog):
    """A present-but-malformed _dims companion must NOT abort the resume — it degrades to the same
    reduce-over-shards fallback as a missing one (A-linescan/F1)."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "full"
    _rows, mns, mss = _run(env, out, compress_workers=4)
    (out / "_dims" / "shard_00000.json").write_text("{not json")            # JSONDecodeError
    (out / "_dims" / "shard_00001.json").write_text(json.dumps({"max_n_seg": None}))  # TypeError/KeyError
    with caplog.at_level("WARNING"):
        _r, r_mns, r_mss = _run(env, out, resume=True, compress_workers=4)
    assert (r_mns, r_mss) == (mns, mss), "corrupt companions must fall back to reduce-over-shards"
    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_resume_from_shard_absent_prefix_warns_about_understatement(tmp_path, caplog):
    """Major (Altitude/F1): an explicit resume_from_shard whose prefix shards are NOT local cannot
    recover their max dims (nothing local to read), so the manifest may understate — it must WARN
    loudly, turning a silent understatement into an actionable one."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "fresh"  # NO prefix shards on disk
    out.mkdir()
    with caplog.at_level("WARNING"):
        _run(env, out, resume_from_shard=5, compress_workers=4)
    assert any(
        ("not present locally" in r.message.lower()) or ("understate" in r.message.lower())
        for r in caplog.records
    ), "absent-prefix resume_from_shard must warn about possible manifest understatement"


def test_profile_resume_accumulates_across_calls(tmp_path):
    """E3 (#540): run() shares ONE profile across the train + val precompute calls, so
    _record_replay must ACCUMULATE the replay record — a plain assignment let val clobber train's.
    Resume the same complete dir twice with one shared profile and assert the record doubles."""
    env = _make_precompute_env(tmp_path, n_shards=4)
    full = tmp_path / "full"
    _run(env, full, compress_workers=4)

    p1 = PackedRowsProfile()
    _run(env, full, resume=True, compress_workers=4, profile=p1)
    single = p1.resume["rows_replayed"]
    assert single > 0 and p1.resume["mode"] == "replay"

    p2 = PackedRowsProfile()
    _run(env, full, resume=True, compress_workers=4, profile=p2)
    _run(env, full, resume=True, compress_workers=4, profile=p2)
    assert p2.resume["rows_replayed"] == 2 * single, "resume record must accumulate, not clobber"
