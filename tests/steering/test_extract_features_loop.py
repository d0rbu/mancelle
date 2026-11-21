import json
import threading
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.extract_features as ef
import steering.profiling.extraction as prof
from steering.extract_features import (
    ActivationType,
    ExtractionStagingConfig,
    PoolingMode,
    ProcessShardParams,
    SegmentMode,
    run_extraction_loop,
)
from steering.feature_io import glob_features, load_features, save_features
from steering.refined_types import (
    NonNegativeFiniteNumber,
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
    UnitInterval,
    ZstdCompressionLevel,
)
from steering.staging.saver import StagedSaver
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)
from tests.steering.test_feature_io import _make_shard_dict


def _assert_shards_decompressed_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for key, value in a.items():
        if not isinstance(value, t.Tensor):
            assert value == b[key], f"metadata {key} differs"
    for k in ("offsets", "doc_ids", "segment_indices", "segment_positions",
              "segment_char_starts", "segment_char_ends", "doc_token_counts"):
        assert t.equal(a[k], b[k]), f"{k} differs"
    off = a["offsets"]
    for s in range(off.numel() - 1):
        lo, hi = int(off[s]), int(off[s + 1])
        # Stored order (feature_id-ascending, mags co-sorted) — compare directly,
        # NOT via independent sorts, so an id<->mag mispairing cannot pass.
        assert t.equal(a["feature_ids"][lo:hi], b["feature_ids"][lo:hi]), f"feature_ids seg {s}"
        assert t.equal(a["feature_mags"][lo:hi], b["feature_mags"][lo:hi]), f"feature_mags seg {s}"


def make_parquet(path: Path, texts: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts}), str(path))
    return str(path)


def tiny_params(output_dir: Path, **over) -> ProcessShardParams:
    base: dict[str, Any] = {
        "model": TinyHookedTransformer(), "sae": TinySAE(),
        "layer": NonNegativeInt.parse(1), "output_dir": str(output_dir),
        "context_window": PositiveInt.parse(64), "pooling": PoolingMode.max_pool,
        "gpu_batch_size": PositiveInt.parse(2), "cpu_batch_size": PositiveInt.parse(2),
        "gc_mem_threshold": UnitInterval.parse(0.9), "nnterp_model": None,
        "native_model": None, "segment_mode": SegmentMode.fixed_tokens,
        "segment_size": PositiveInt.parse(16), "hook_name": None,
        "activation_type": ActivationType.residual, "dataset_phase": None,
        "zstd_level": ZstdCompressionLevel.parse(3),
        "context_independent_features": False, "ignore_bos_features": False, "run_l0_tier": None,
    }
    base.update(over)
    return ProcessShardParams(**base)


def test_run_loop_processes_all_and_profiles(tmp_path):
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"Hello {i}.", f"World {i}."]) for i in range(2)]
    out = tmp_path / "out"
    results = run_extraction_loop(
        tiny_params(out), paths, rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(2),
        staging_cfg=_staging(tmp_path, 0, 0),
        profile_enabled=True, profile_output=str(tmp_path / "p"),
        flops_per_token=1.0, gpu_peak_flops=1.0,
    )
    assert len(results) == 2
    assert len(list(glob_features(str(out)))) == 2
    summary = json.loads((Path(str(tmp_path / "p") + "_rank0.json")).read_text())
    assert summary["shards_completed"] == 2


def test_run_loop_accepts_integer_flops_per_token(tmp_path):
    """MFU input accepts the integer returned by the FLOPs estimator."""
    results = run_extraction_loop(
        tiny_params(tmp_path / "out"), [], rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(0),
        staging_cfg=_staging(tmp_path / "s", 0, 0), profile_enabled=False,
        profile_output=str(tmp_path / "p"), flops_per_token=5,
        gpu_peak_flops=float("inf"),
    )
    assert results == []


def test_loop_drains_saver_inside_profiler_region(tmp_path, monkeypatch):
    """With output staging + profiling, the loop must drain the final in-flight
    save+relocate BEFORE the profiler latches wall_seconds and stops the monitor —
    otherwise the GPU-idle save tail falls outside the measured window and biases
    gpu_idle_fraction optimistically. Assert saver.drain() runs before the sidecar
    is written (the profiler's LIFO teardown). (Finding #3.)"""
    events: list[str] = []
    real_drain = StagedSaver.drain
    monkeypatch.setattr(
        StagedSaver, "drain",
        lambda self: (events.append("drain"), real_drain(self))[1], raising=False,
    )
    real_sidecar = prof._write_sidecar
    monkeypatch.setattr(
        prof, "_write_sidecar",
        lambda *a, **k: (events.append("sidecar"), real_sidecar(*a, **k))[1],
    )
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["Hello.", "World."])]
    run_extraction_loop(
        tiny_params(tmp_path / "out"), paths, rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "s", 3, 0),
        profile_enabled=True, profile_output=str(tmp_path / "p"),
        flops_per_token=1.0, gpu_peak_flops=1.0,
    )
    assert "drain" in events, f"saver.drain() was not called inside the loop: {events}"
    assert events.index("drain") < events.index("sidecar"), events


def test_run_loop_skips_existing(tmp_path):
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"a {i}"]) for i in range(2)]
    out = tmp_path / "out"
    save_features(_make_shard_dict(n_segments=1, feats_per_seg=1, n_docs=1),
                  str(out / "shard_00000_features"))
    results = run_extraction_loop(
        tiny_params(out), paths, rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(2),
        staging_cfg=_staging(tmp_path, 0, 0),
        profile_enabled=False, profile_output=str(tmp_path / "p"),
        flops_per_token=1.0, gpu_peak_flops=1.0,
    )
    assert len(results) == 1  # shard 0 skipped, shard 1 processed


def _staging(tmp_path, output_depth, input_depth):
    return ExtractionStagingConfig(
        scratch_dir=str(tmp_path / "scratch"),
        output_depth=NonNegativeInt.parse(output_depth),
        input_depth=NonNegativeInt.parse(input_depth),
        min_free_gb=NonNegativeFiniteNumber.parse(0.0),
        est_shard_gb=PositiveFiniteNumber.parse(0.001),
    )


def _read_all(out_dir):
    return {p.name: load_features(str(p)) for p in glob_features(str(out_dir))}


@pytest.mark.parametrize("output_depth", [0, 3])
def test_staged_output_byte_identical_to_direct(tmp_path, output_depth):
    # O4 (output side): output_depth=0 (direct) vs >=1 (staged) -> identical shards.
    # Share ONE model+SAE across both runs so output_depth is the only variable
    # (the tiny fixtures init random weights on each construction).
    model, sae = TinyHookedTransformer(), TinySAE()
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"alpha {i}", f"beta {i}"]) for i in range(3)]
    ref_out = tmp_path / "ref"
    run_extraction_loop(tiny_params(ref_out, model=model, sae=sae), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(3),
                        staging_cfg=_staging(tmp_path / "r", 0, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    new_out = tmp_path / "new"
    run_extraction_loop(tiny_params(new_out, model=model, sae=sae), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(3),
                        staging_cfg=_staging(tmp_path / "n", output_depth, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    ref, new = _read_all(ref_out), _read_all(new_out)
    assert set(ref) == set(new)
    assert len(ref) > 0  # non-vacuous: shards actually written
    for name in ref:
        _assert_shards_decompressed_equal(ref[name], new[name])


@pytest.mark.parametrize("output_depth,input_depth", [(0, 0), (0, 3), (2, 0), (2, 3)])
def test_staging_grid_output_identical(tmp_path, output_depth, input_depth):
    # O4 full grid: staging on EITHER side (output_depth and/or input_depth), any
    # depth, is byte-identical to the direct/inline (0, 0) reference. Share ONE
    # model+SAE across both runs so depth is the only variable (the tiny fixtures
    # init random weights on each construction).
    model, sae = TinyHookedTransformer(), TinySAE()
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"a {i}", f"b {i}", f"c {i}"]) for i in range(4)]
    ref_out = tmp_path / "ref"
    run_extraction_loop(tiny_params(ref_out, model=model, sae=sae), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(4),
                        staging_cfg=_staging(tmp_path / "r", 0, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    new_out = tmp_path / f"new_{output_depth}_{input_depth}"
    run_extraction_loop(tiny_params(new_out, model=model, sae=sae), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(4),
                        staging_cfg=_staging(tmp_path / f"s_{output_depth}_{input_depth}", output_depth, input_depth),
                        profile_enabled=False, profile_output=str(tmp_path / "p"),
                        flops_per_token=1.0, gpu_peak_flops=1.0)
    ref, new = _read_all(ref_out), _read_all(new_out)
    assert set(ref) == set(new)
    assert len(ref) > 0  # non-vacuous: shards actually written
    for name in ref:
        _assert_shards_decompressed_equal(ref[name], new[name])


def test_input_depth_routes_through_staged_prefetcher(tmp_path, monkeypatch):
    # Wiring contract: input_depth>=1 builds exactly one StagedPrefetcher with a
    # PER-RANK scratch dir (rank{rank}/in) at the configured depth; input_depth=0
    # builds none (process_shard does the inline read). RED before the wiring
    # exists (run_extraction_loop never references StagedPrefetcher).
    calls: list[tuple[str, int]] = []
    real = ef.StagedPrefetcher

    def spy(*, scratch_dir, depth, load_fn):
        calls.append((str(scratch_dir), depth))
        return real(scratch_dir=scratch_dir, depth=depth, load_fn=load_fn)

    monkeypatch.setattr(ef, "StagedPrefetcher", spy)
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"a {i}"]) for i in range(3)]
    run_extraction_loop(tiny_params(tmp_path / "o0"), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(3),
                        staging_cfg=_staging(tmp_path / "s0", 0, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    assert calls == []  # input_depth=0 -> no prefetcher (inline read)
    run_extraction_loop(tiny_params(tmp_path / "o2"), paths, rank=2, world_size=4, local_rank=2,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(3),
                        staging_cfg=_staging(tmp_path / "s2", 0, 2), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    assert len(calls) == 1  # input_depth=2 -> built once
    scratch, depth = calls[0]
    assert depth == 2
    assert Path(scratch).parts[-3:] == ("range_0_3", "rank2", "in")  # range + rank isolation


def test_resume_skips_completed_shard(tmp_path):
    # O5: rerun with staging on -> all shards skipped, mtimes unchanged.
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"x {i}"]) for i in range(3)]
    out = tmp_path / "out"
    kw = {"rank": 0, "world_size": 1, "local_rank": 0,
          "shard_start": NonNegativeInt.parse(0), "shard_end": NonNegativeInt.parse(3),
          "staging_cfg": _staging(tmp_path / "s", 2, 0),
              "profile_enabled": False, "profile_output": str(tmp_path / "p"),
              "flops_per_token": 1.0, "gpu_peak_flops": 1.0}
    run_extraction_loop(tiny_params(out), paths, **kw)
    before = {p.name: p.stat().st_mtime_ns for p in glob_features(str(out))}
    assert len(before) == 3  # all 3 shards were written on the first pass
    run_extraction_loop(tiny_params(out), paths, **kw)
    after = {p.name: p.stat().st_mtime_ns for p in glob_features(str(out))}
    assert before == after


def test_scratch_is_isolated_per_output_dir(tmp_path, monkeypatch):
    """Scratch must be keyed by output_dir, so two concurrent runs to DIFFERENT
    output_dirs (same node, same rank, same scratch_dir) use disjoint subtrees and
    one run's startup sweep can't delete the other's in-flight shards. (Finding #1.)"""
    out_a, out_b = tmp_path / "outA", tmp_path / "outB"
    assert ef._scratch_run_key(str(out_a)) != ef._scratch_run_key(str(out_b))
    assert ef._scratch_run_key(str(out_a)) == ef._scratch_run_key(str(out_a))  # stable -> resume reuses
    captured: dict = {}
    real_init = StagedSaver.__init__
    def spy_init(self, **kw):
        captured["scratch_dir"] = str(kw["scratch_dir"])
        real_init(self, **kw)
    monkeypatch.setattr(StagedSaver, "__init__", spy_init)
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    run_extraction_loop(tiny_params(out_a), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
                        staging_cfg=_staging(tmp_path / "scr", 2, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    assert ef._scratch_run_key(str(out_a)) in captured["scratch_dir"]
    assert Path(captured["scratch_dir"]).parts[-2:] == ("rank0", "out")  # per-rank suffix preserved


def test_scratch_range_namespace_is_stable_and_shared_by_ddp_ranks(tmp_path, monkeypatch):
    """One configured window is shared by all ranks, while sibling windows differ."""
    scratch = tmp_path / "scratch"
    out = tmp_path / "out"
    start = NonNegativeInt.parse(17)
    end = NonNegativeInt.parse(42)
    sibling_end = NonNegativeInt.parse(43)
    assert ef._scratch_range_key(start, end) == "range_17_42"
    assert ef._scratch_range_key(start, end) == ef._scratch_range_key(start, end)
    assert ef._scratch_range_key(start, end) != ef._scratch_range_key(start, sibling_end)

    captured: list[str] = []
    real_init = StagedSaver.__init__

    def spy_init(self, **kw):
        captured.append(str(kw["scratch_dir"]))
        real_init(self, **kw)

    monkeypatch.setattr(StagedSaver, "__init__", spy_init)
    monkeypatch.setattr(ef, "coordinate_free_space_check", lambda *args, **kwargs: None)
    cfg = _staging(scratch, 1, 0)
    for rank in (0, 1):
        run_extraction_loop(
            tiny_params(out), [], rank=rank, world_size=2, local_rank=rank,
            shard_start=start, shard_end=end, staging_cfg=cfg,
            profile_enabled=False, profile_output=str(tmp_path / "p"),
            flops_per_token=1.0, gpu_peak_flops=1.0,
        )

    base = Path(cfg.scratch_dir) / ef._scratch_run_key(str(out)) / "range_17_42"
    assert captured == [str(base / "rank0" / "out"), str(base / "rank1" / "out")]
    assert not (scratch / ef._scratch_run_key(str(out)) / "range_17_43").exists()


def test_scratch_range_cleanup_cannot_delete_sibling_range(tmp_path):
    """A completed range removes only its own rank subtree."""
    cfg = _staging(tmp_path / "scratch", 1, 0)
    out = tmp_path / "out"
    sibling = (
        Path(cfg.scratch_dir)
        / ef._scratch_run_key(str(out))
        / ef._scratch_range_key(NonNegativeInt.parse(1), NonNegativeInt.parse(2))
        / "rank0"
        / "out"
    )
    sibling.mkdir(parents=True)
    sentinel = sibling / "in_flight.safetensors"
    sentinel.write_bytes(b"survivor")
    path = make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])
    run_extraction_loop(
        tiny_params(out), [path], rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
        staging_cfg=cfg, profile_enabled=False, profile_output=str(tmp_path / "p"),
        flops_per_token=1.0, gpu_peak_flops=1.0,
    )
    assert sentinel.read_bytes() == b"survivor"


def test_concurrent_disjoint_ranges_survive_each_others_cleanup(tmp_path, monkeypatch):
    """Regression: a finishing range must not delete another range's in-flight save."""
    scratch = tmp_path / "scratch"
    out = tmp_path / "out"
    path_a = make_parquet(tmp_path / "data" / "shard_00000.parquet", ["a"])
    path_b = make_parquet(tmp_path / "data" / "shard_00001.parquet", ["b"])
    cfg = _staging(scratch, 1, 0)
    b_started = threading.Event()
    release_b = threading.Event()
    errors: list[BaseException] = []
    real_save_and_move = StagedSaver._save_and_move

    def fake_process(parquet_path, *args, **kwargs):
        shard_dict = _make_shard_dict(n_segments=1, feats_per_seg=1, n_docs=1)
        out_stem = Path(kwargs["output_dir"]) / f"{Path(parquet_path).stem}_features"
        kwargs["background_saver"].submit(shard_dict, out_stem)
        return ef.ShardResult(
            n_docs=1,
            n_tokens_gpt2=1,
            n_total_features=1,
            elapsed_seconds=0.0,
            output_path=str(ef.feature_path(kwargs["output_dir"], Path(parquet_path).stem)),
            bytes_on_disk=0,
        )

    def gated_save_and_move(self, data, out_stem):
        if "range_1_2" in str(self._scratch_dir):
            b_started.set()
            assert release_b.wait(timeout=10), "timed out waiting to release range B"
            assert self._scratch_dir.exists(), "range B scratch was deleted by range A"
        return real_save_and_move(self, data, out_stem)

    monkeypatch.setattr(ef, "process_shard", fake_process)
    monkeypatch.setattr(StagedSaver, "_save_and_move", gated_save_and_move)
    monkeypatch.setattr(ef, "coordinate_free_space_check", lambda *args, **kwargs: None)

    def run_range(path, start, end):
        try:
            run_extraction_loop(
                tiny_params(out), [path], rank=0, world_size=1, local_rank=0,
                shard_start=NonNegativeInt.parse(start), shard_end=NonNegativeInt.parse(end),
                staging_cfg=cfg, profile_enabled=False, profile_output=str(tmp_path / "p"),
                flops_per_token=1.0, gpu_peak_flops=1.0,
            )
        except BaseException as exc:  # propagate thread failures to the assertion below
            errors.append(exc)

    thread_b = threading.Thread(target=run_range, args=(path_b, 1, 2))
    thread_b.start()
    assert b_started.wait(timeout=10), "range B did not reach its staged save"
    thread_a = threading.Thread(target=run_range, args=(path_a, 0, 1))
    thread_a.start()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive(), "range A did not finish"
    release_b.set()
    thread_b.join(timeout=10)
    assert not thread_b.is_alive(), "range B did not finish"
    assert errors == []
    assert len(list(glob_features(str(out)))) == 2


def test_negative_output_depth_raises():
    """A negative output depth cannot construct the loop's typed config."""
    with pytest.raises(BeartypeCallHintParamViolation):
        ExtractionStagingConfig(
            scratch_dir="/tmp/staging",
            output_depth=-1,  # ty: ignore[invalid-argument-type]
            input_depth=NonNegativeInt.parse(0),
            min_free_gb=NonNegativeFiniteNumber.parse(0.0),
            est_shard_gb=PositiveFiniteNumber.parse(0.001),
        )


def test_negative_input_depth_raises():
    """A negative input depth cannot construct the loop's typed config."""
    with pytest.raises(BeartypeCallHintParamViolation):
        ExtractionStagingConfig(
            scratch_dir="/tmp/staging",
            output_depth=NonNegativeInt.parse(0),
            input_depth=-1,  # ty: ignore[invalid-argument-type]
            min_free_gb=NonNegativeFiniteNumber.parse(0.0),
            est_shard_gb=PositiveFiniteNumber.parse(0.001),
        )


def test_invalid_local_world_size_falls_back(tmp_path, monkeypatch):
    """An empty / non-numeric LOCAL_WORLD_SIZE must not crash the run before any
    shard (int('') raises ValueError) — fall back to world_size. (Finding #14.)"""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "")
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    run_extraction_loop(tiny_params(tmp_path / "out"), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
                        staging_cfg=_staging(tmp_path / "s", 2, 0), profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    assert len(list(glob_features(str(tmp_path / "out")))) == 1


def test_free_space_check_delegates_to_coordinator(tmp_path, monkeypatch):
    """The loop delegates the startup free-space check to coordinate_free_space_check
    (rank-0 gate, others wait), called once per run with this rank's local_rank. The
    gate's own rank-0-vs-wait behavior is unit-tested in test_relocate.py. (Finding #11.)"""
    calls: list = []
    monkeypatch.setattr(ef, "coordinate_free_space_check", lambda run_scratch, **kw: calls.append(kw))
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    kw = {"shard_start": NonNegativeInt.parse(0), "shard_end": NonNegativeInt.parse(3),
          "staging_cfg": _staging(tmp_path / "s", 2, 0), "profile_enabled": False,
              "profile_output": str(tmp_path / "p"), "flops_per_token": 1.0, "gpu_peak_flops": 1.0}
    run_extraction_loop(tiny_params(tmp_path / "o2"), paths, rank=2, world_size=4, local_rank=2, **kw)
    assert len(calls) == 1
    assert calls[0]["local_rank"] == 2 and calls[0]["output_depth"] == 2
    assert calls[0]["destination_dir"] == tmp_path / "o2"
    assert (tmp_path / "o2").is_dir()


def test_direct_output_skips_staging_free_space_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ef,
        "coordinate_free_space_check",
        lambda *a, **k: pytest.fail("direct output must not run staging coordination"),
    )
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    run_extraction_loop(
        tiny_params(tmp_path / "out"),
        paths,
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "s", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "p"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )


def test_staging_startup_failure_precedes_shard_processing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ef,
        "coordinate_free_space_check",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("destination preflight failed")),
    )
    monkeypatch.setattr(
        ef,
        "process_shard",
        lambda *a, **k: pytest.fail("shard processing must not start after preflight failure"),
    )
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    with pytest.raises(RuntimeError, match="destination preflight failed"):
        run_extraction_loop(
            tiny_params(tmp_path / "out"),
            paths,
            rank=0,
            world_size=1,
            local_rank=0,
            shard_start=NonNegativeInt.parse(0),
            shard_end=NonNegativeInt.parse(1),
            staging_cfg=_staging(tmp_path / "s", 2, 0),
            profile_enabled=False,
            profile_output=str(tmp_path / "p"),
            flops_per_token=1.0,
            gpu_peak_flops=1.0,
        )


def test_staged_prefetch_load_errors_name_source_path(tmp_path):
    """A staged shard with a bad schema must raise naming the durable SOURCE path,
    not the ephemeral uuid scratch copy (already deleted by the time you read the
    error). (Finding #8.)"""
    source = str(tmp_path / "shard_00042.parquet")  # durable source (for diagnostics)
    local = tmp_path / "scratch" / "deadbeef_shard_00042.parquet"
    local.parent.mkdir(parents=True)
    pq.write_table(pa.table({"not_text": ["a"]}), str(local))  # staged copy, missing 'text'
    with pytest.raises(ValueError) as ei:
        ef._staged_prefetch_load(
            source,
            str(local),
            cpu_batch_size=2,
            dataset_phase=None,
            segment_mode=SegmentMode.fixed_tokens,
        )
    msg = str(ei.value)
    assert source in msg          # error names the source shard
    assert "deadbeef" not in msg  # NOT the deleted scratch copy


def test_staged_prefetch_load_repoints_identity_to_source(tmp_path):
    """For a valid staged shard, both parquet_path AND shard_name must reflect the
    source, not the uuid scratch copy. (Finding #8.)"""
    source = str(tmp_path / "shard_00007.parquet")
    local = tmp_path / "scratch" / "cafef00d_shard_00007.parquet"
    local.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["hello", "world"]}), str(local))
    shard = ef._staged_prefetch_load(
        source,
        str(local),
        cpu_batch_size=2,
        dataset_phase=None,
        segment_mode=SegmentMode.fixed_tokens,
    )
    assert shard.parquet_path == source
    assert shard.shard_name == "shard_00007"  # from source, not "cafef00d_shard_00007"


def test_safe_int_env_branches(monkeypatch):
    """_safe_int_env: valid int parses; unset/empty/whitespace/non-numeric all fall
    back to the default (int('') and int('abc') would otherwise crash before any shard)."""
    monkeypatch.setenv("LWS_PROBE", "4")
    assert ef._safe_int_env("LWS_PROBE", 7) == 4          # valid -> parsed
    monkeypatch.setenv("LWS_PROBE", "abc")
    assert ef._safe_int_env("LWS_PROBE", 7) == 7          # non-numeric -> default (except ValueError)
    monkeypatch.setenv("LWS_PROBE", "   ")
    assert ef._safe_int_env("LWS_PROBE", 7) == 7          # whitespace -> default
    monkeypatch.delenv("LWS_PROBE", raising=False)
    assert ef._safe_int_env("LWS_PROBE", 7) == 7          # unset -> default


def test_input_depth_exceeding_num_shards_is_clamped(tmp_path):
    """input_depth > num_shards must not over-submit during priming — the
    min(input_depth, len) clamp; without it, range(input_depth) would IndexError."""
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]  # 1 shard
    run_extraction_loop(tiny_params(tmp_path / "out"), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
                        staging_cfg=_staging(tmp_path / "s", 0, 2), profile_enabled=False,  # input_depth=2 > 1
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    assert len(list(glob_features(str(tmp_path / "out")))) == 1


def test_resume_skip_with_input_staging_stays_aligned(tmp_path):
    """Resume with input_depth>=1: each skipped shard is still get()'d before the
    skip check, so the prefetcher FIFO stays aligned and the second pass is a clean
    all-skip with unchanged outputs."""
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"x {i}", f"y {i}"]) for i in range(3)]
    out = tmp_path / "out"
    kw = {"rank": 0, "world_size": 1, "local_rank": 0,
          "shard_start": NonNegativeInt.parse(0), "shard_end": NonNegativeInt.parse(3),
          "profile_enabled": False,
              "profile_output": str(tmp_path / "p"), "flops_per_token": 1.0, "gpu_peak_flops": 1.0}
    run_extraction_loop(tiny_params(out), paths, staging_cfg=_staging(tmp_path / "s1", 0, 2), **kw)
    before = {p.name: p.stat().st_mtime_ns for p in glob_features(str(out))}
    assert len(before) == 3
    run_extraction_loop(tiny_params(out), paths, staging_cfg=_staging(tmp_path / "s2", 0, 2), **kw)
    after = {p.name: p.stat().st_mtime_ns for p in glob_features(str(out))}
    assert before == after


def test_staging_config_depths_are_strict():
    """The typed config rejects bool, non-int, and negative staging depths."""
    for bad in (True, False, 1.5, -1, "2"):
        with pytest.raises(BeartypeCallHintParamViolation):
            ExtractionStagingConfig(
                scratch_dir="/tmp/staging",
                output_depth=bad,  # ty: ignore[invalid-argument-type]
                input_depth=NonNegativeInt.parse(0),
                min_free_gb=NonNegativeFiniteNumber.parse(0.0),
                est_shard_gb=PositiveFiniteNumber.parse(0.001),
            )


def test_scratch_rank_subtree_removed_after_clean_run(tmp_path):
    """A rank's own scratch subtree is rmtree'd on clean exit, so per-output_dir keying
    doesn't accumulate empty rank{rank}/{out,in} trees. Per-rank (not the shared run_scratch)
    so a fast rank can't delete a still-running rank's scratch. (Finding #13.)"""
    cfg = _staging(tmp_path / "s", 2, 2)  # output + input staging
    out = tmp_path / "out"
    paths = [make_parquet(tmp_path / "data" / "shard_00000.parquet", ["x", "y"])]
    run_extraction_loop(tiny_params(out), paths, rank=0, world_size=1, local_rank=0,
                        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(1),
                        staging_cfg=cfg, profile_enabled=False,
                        profile_output=str(tmp_path / "p"), flops_per_token=1.0, gpu_peak_flops=1.0)
    run_scratch = (
        Path(cfg.scratch_dir)
        / ef._scratch_run_key(str(out))
        / ef._scratch_range_key(NonNegativeInt.parse(0), NonNegativeInt.parse(1))
    )
    assert not (run_scratch / "rank0").exists()  # this rank's {out,in} subtree removed


def test_resume_does_not_stage_already_done_shards(tmp_path, monkeypatch):
    """On resume, pre-filtering means the input prefetcher never stages (copies/reads)
    an already-extracted shard. (Finding #14.)"""
    data = tmp_path / "data"
    paths = [make_parquet(data / f"shard_{i:05d}.parquet", [f"x {i}"]) for i in range(3)]
    out = tmp_path / "out"
    kw = {"rank": 0, "world_size": 1, "local_rank": 0,
          "shard_start": NonNegativeInt.parse(0), "shard_end": NonNegativeInt.parse(3),
          "profile_enabled": False,
              "profile_output": str(tmp_path / "p"), "flops_per_token": 1.0, "gpu_peak_flops": 1.0}
    run_extraction_loop(tiny_params(out), paths, staging_cfg=_staging(tmp_path / "s1", 0, 2), **kw)
    assert len(list(glob_features(str(out)))) == 3
    submits: list = []
    real_submit = ef.StagedPrefetcher.submit
    monkeypatch.setattr(ef.StagedPrefetcher, "submit",
                        lambda self, p: (submits.append(p), real_submit(self, p))[1])
    run_extraction_loop(tiny_params(out), paths, staging_cfg=_staging(tmp_path / "s2", 0, 2), **kw)
    assert submits == []  # all 3 pre-filtered -> nothing staged on the resume
