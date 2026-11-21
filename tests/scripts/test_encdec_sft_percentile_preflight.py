"""_percentile_coverage_preflight: rank-once scan wiring + coverage verdicts.

Single-process tests drive the extracted helper directly with real feature
tables and dict/real stores; the two-rank gloo tests prove the scan runs
exactly once (rank 0), every rank receives identical raw/masked sets and the
same resolver, and a rank-0 scan failure aborts both ranks in lockstep.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch as t
import torch.distributed as dist

import scripts.train.conditional_encdec_sft as sft
from scripts.train.conditional_encdec_sft import _percentile_coverage_preflight
from steering._common.ddp import AlignedRankFailure, rank_synchronized_stage
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.missing_percentile import (
    MissingPercentileError,
    MissingPercentilePolicy,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from tests.gloo_two_rank_helpers import run_two_ranks
from tests.steering.test_sft_feature_id_scan import _write_custom_shard
from tests.steering.test_sft_features import _write_parquet

N_SAE = 32


def _seg(fids: list[int]) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor(fids, dtype=t.int32),
        feature_mags=t.tensor([1.0] * len(fids), dtype=t.float32),
        char_start=0,
        char_end=8,
    )


def _store(active: dict[int, list[int]]) -> dict[int, list[SegmentFeatures]]:
    return {index: [_seg(fids)] for index, fids in active.items()}


def _table(covered: list[int]) -> t.Tensor:
    table = t.zeros(N_SAE, 99, dtype=t.float64)
    for fid in covered:
        table[fid] = t.arange(1, 100, dtype=t.float64)
    return table


def _phase_extra(trace: sft._SFTStartupTrace, phase: str) -> dict[str, object]:
    """Test oracle over the trace's private records (the startup-timing suite
    reads them the same way; there is deliberately no public accessor)."""
    by_name = {record.name: record for record in trace._records}
    return dict(by_name[phase].extra)


def _capture_broadcasts() -> list[t.Tensor]:
    """Spy on the bitmap this rank receives from the rank-0 broadcast.

    Direct module-attribute patch: monkeypatch is unavailable inside a spawned
    gloo worker, and the child process isolates the mutation.
    """
    captured: list[t.Tensor] = []
    real_broadcast = sft.broadcast_bitmap_from_rank_zero

    def capturing_broadcast(*args, **kwargs):
        bitmap = real_broadcast(*args, **kwargs)
        captured.append(bitmap)
        return bitmap

    sft.broadcast_bitmap_from_rank_zero = capturing_broadcast  # ty: ignore[invalid-assignment]
    return captured


class TestSingleProcessPreflight:
    def test_fully_covered_run_returns_error_resolver(self) -> None:
        """Downstream byte-identity rationale: under ``error`` the resolver the
        generator receives is exactly ``MissingPercentileResolver.error()`` —
        policy ERROR, no bitmap, ``substitutes`` False — i.e. the identical
        object semantics the generator gets today, so every downstream tensor
        path is untouched by construction (and remains covered by the existing
        conditioned-integration and telemetry-parity suites)."""
        resolver = _percentile_coverage_preflight(
            coverage_sources=[("train", _store({0: [1, 5]}), [0])],
            n_sae_features=N_SAE,
            feature_mask=None,
            fp_percentile_table=_table([1, 5]),
            sft_sampling_bundle=None,
            cluster_threshold_percentile_table=None,
            missing_percentile_policy=MissingPercentilePolicy.ERROR,
            device="cpu",
            startup_trace=None,
        )
        assert resolver.policy is MissingPercentilePolicy.ERROR
        assert resolver.covered is None
        assert resolver.substitutes is False

    def test_missing_ids_abort_with_complete_count(self) -> None:
        with pytest.raises(
            MissingPercentileError,
            match=r"fp_percentile_table is missing percentile statistics for 2 of the 3",
        ):
            _percentile_coverage_preflight(
                coverage_sources=[("train", _store({0: [1, 5, 9]}), [0])],
                n_sae_features=N_SAE,
                feature_mask=None,
                fp_percentile_table=_table([1]),
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=None,
            )

    def test_mask_does_not_hide_raw_fp_holes(self) -> None:
        """Contract point 7: an id outside the whitelist still fails FP."""
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1,))
        with pytest.raises(MissingPercentileError, match="fp_percentile_table"):
            _percentile_coverage_preflight(
                coverage_sources=[("train", _store({0: [1, 5]}), [0])],
                n_sae_features=N_SAE,
                feature_mask=mask,
                fp_percentile_table=_table([1]),
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=None,
            )

    def test_union_across_train_and_val_sources(self) -> None:
        with pytest.raises(
            MissingPercentileError, match=r"missing percentile statistics for 1 of the 2"
        ):
            _percentile_coverage_preflight(
                coverage_sources=[
                    ("train", _store({0: [1]}), [0]),
                    ("val", _store({0: [9]}), [0]),
                ],
                n_sae_features=N_SAE,
                feature_mask=None,
                fp_percentile_table=_table([1]),
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=None,
            )

    def test_lowest_returns_combined_coverage_resolver(self) -> None:
        resolver = _percentile_coverage_preflight(
            coverage_sources=[("train", _store({0: [1, 9]}), [0])],
            n_sae_features=N_SAE,
            feature_mask=None,
            fp_percentile_table=_table([1]),
            sft_sampling_bundle=None,
            cluster_threshold_percentile_table=None,
            missing_percentile_policy=MissingPercentilePolicy.LOWEST,
            device="cpu",
            startup_trace=None,
        )
        assert resolver.policy is MissingPercentilePolicy.LOWEST
        assert resolver.covered is not None
        assert bool(resolver.covered[1]) and not bool(resolver.covered[9])

    def test_lowest_with_no_tables_rejects_the_override(self) -> None:
        with pytest.raises(ValueError, match="queries no percentile table at all"):
            _percentile_coverage_preflight(
                coverage_sources=[],
                n_sae_features=N_SAE,
                feature_mask=None,
                fp_percentile_table=None,
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.LOWEST,
                device="cpu",
                startup_trace=None,
            )

    def test_no_sources_records_zero_counts_and_error_resolver(self) -> None:
        trace = sft._SFTStartupTrace()
        resolver = _percentile_coverage_preflight(
            coverage_sources=[],
            n_sae_features=N_SAE,
            feature_mask=None,
            fp_percentile_table=None,
            sft_sampling_bundle=None,
            cluster_threshold_percentile_table=None,
            missing_percentile_policy=MissingPercentilePolicy.ERROR,
            device="cpu",
            startup_trace=trace,
        )
        assert resolver.policy is MissingPercentilePolicy.ERROR
        scan_extra = _phase_extra(trace, "sft_percentile_id_scan")
        assert scan_extra["sources"] == 0
        assert scan_extra["raw_ids"] == 0
        assert scan_extra["masked_ids"] == 0
        assert _phase_extra(trace, "sft_percentile_coverage")["tables"] == 0

    def test_trace_records_raw_and_masked_counts(self) -> None:
        trace = sft._SFTStartupTrace()
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 2))
        _percentile_coverage_preflight(
            coverage_sources=[("train", _store({0: [1, 2, 5]}), [0])],
            n_sae_features=N_SAE,
            feature_mask=mask,
            fp_percentile_table=_table([1, 2, 5]),
            sft_sampling_bundle=None,
            cluster_threshold_percentile_table=None,
            missing_percentile_policy=MissingPercentilePolicy.ERROR,
            device="cpu",
            startup_trace=trace,
        )
        assert _phase_extra(trace, "sft_percentile_id_scan") == {
            "sources": 1,
            "raw_ids": 3,
            "masked_ids": 2,
        }

    def test_sampling_table_is_checked_against_post_mask_candidates(self) -> None:
        """The sampling-bundle branch of the helper, exercised end to end.

        Active {1,5,9}, whitelist (1,5) → masked candidates {1,5}; the
        sampling table covers only {1}, so exactly one post-mask hole (5) —
        and 9 must NOT count (it is masked away for selection)."""
        bundle = SamplingBundle(
            percentile_table=_table([1]),
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=10.0,
                std=10.0,
                min_k=1,
                max_k=8,
                mode=SamplingMode.weighted,
            ),
            rng=t.Generator().manual_seed(0),
        )
        with pytest.raises(
            MissingPercentileError,
            match=r"sampling_bundle\.percentile_table is missing percentile "
            r"statistics for 1 of the 2",
        ):
            _percentile_coverage_preflight(
                coverage_sources=[("train", _store({0: [1, 5, 9]}), [0])],
                n_sae_features=N_SAE,
                feature_mask=FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 5)),
                fp_percentile_table=None,
                sft_sampling_bundle=bundle,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=None,
            )

    def test_cluster_table_is_checked_against_post_mask_candidates(self) -> None:
        with pytest.raises(
            MissingPercentileError,
            match=r"cluster_threshold_percentile_table is missing percentile "
            r"statistics for 1 of the 2",
        ):
            _percentile_coverage_preflight(
                coverage_sources=[("train", _store({0: [1, 5, 9]}), [0])],
                n_sae_features=N_SAE,
                feature_mask=FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 5)),
                fp_percentile_table=None,
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=_table([1]),
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=None,
            )

    def test_lowest_covered_bitmap_is_the_conjunction_of_all_tables(self) -> None:
        bundle = SamplingBundle(
            percentile_table=_table([1]),
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=10.0,
                std=10.0,
                min_k=1,
                max_k=8,
                mode=SamplingMode.weighted,
            ),
            rng=t.Generator().manual_seed(0),
        )
        resolver = _percentile_coverage_preflight(
            coverage_sources=[("train", _store({0: [1]}), [0])],
            n_sae_features=N_SAE,
            feature_mask=None,
            fp_percentile_table=_table([1, 2]),
            sft_sampling_bundle=bundle,
            cluster_threshold_percentile_table=None,
            missing_percentile_policy=MissingPercentilePolicy.LOWEST,
            device="cpu",
            startup_trace=None,
        )
        assert resolver.policy is MissingPercentilePolicy.LOWEST
        assert resolver.covered is not None
        assert bool(resolver.covered[1])  # covered by BOTH tables
        assert not bool(resolver.covered[2])  # fp-only ⇒ conjunction is False
        assert not bool(resolver.covered[3])  # covered by neither


def test_main_calls_the_preflight_helper() -> None:
    main_ast = ast.parse(inspect.getsource(sft.main))
    called = {
        node.func.id
        for node in ast.walk(main_ast)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_percentile_coverage_preflight" in called


def test_main_splits_source_loading_from_the_scan_stage() -> None:
    """The two-stage split is load-bearing: the scan stage opens with a
    collective (the rank-0 bitmap broadcast), so store loading must be aligned
    in its own EARLIER stage — merging them back would let a rank-local load
    failure mis-pair its stage-end all-reduce with a peer's scan collective."""
    source = inspect.getsource(sft.main)
    sources_stage = source.index('rank_synchronized_stage("percentile coverage sources"')
    scan_stage = source.index('rank_synchronized_stage("percentile coverage",')
    helper_call = source.index("_percentile_coverage_preflight(")
    assert sources_stage < scan_stage < helper_call


class _CountingStore(dict):
    """Dict store that counts reference-scan traversals via __getitem__."""

    def __init__(self, data, counter):
        super().__init__(data)
        self._counter = counter

    def __getitem__(self, key):
        self._counter.append(key)
        return super().__getitem__(key)


def _gloo_preflight_success_worker(rank, world_size, init_file, result_queue) -> None:
    """Two sources (train+val) so the DDP case covers the multi-source union.

    The broadcast CONTENT is verified exactly: a spy captures the bitmap the
    preflight received from ``broadcast_bitmap_from_rank_zero`` on each rank,
    and rank 1 performs zero store lookups, so its exact id set can only have
    arrived through the rank-0 broadcast.
    """
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        lookups: list[int] = []
        train_store = _CountingStore(_store({0: [1, 5], 1: [9]}), lookups)
        val_store = _CountingStore(_store({0: [2]}), lookups)
        captured = _capture_broadcasts()
        trace = sft._SFTStartupTrace()
        with rank_synchronized_stage("percentile coverage", device="cpu"):
            resolver = _percentile_coverage_preflight(
                coverage_sources=[
                    ("train", train_store, [0, 1]),
                    ("val", val_store, [0]),
                ],
                n_sae_features=N_SAE,
                feature_mask=FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 9)),
                fp_percentile_table=_table([1, 2, 5, 9]),
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=trace,
            )
        scan_extra = _phase_extra(trace, "sft_percentile_id_scan")
        assert len(captured) == 1
        result_queue.put(
            (
                rank,
                len(lookups),
                resolver.policy.value,
                captured[0].nonzero().squeeze(1).tolist(),
                scan_extra["masked_ids"],
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, -1, f"{type(exc).__name__}: {exc}", [], -1))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_preflight_rank0_failure_worker(rank, world_size, init_file, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        class _BrokenStore(dict):
            def __getitem__(self, key):
                raise OSError("cannot read shard shard_00001")

        store = _BrokenStore({0: None})
        try:
            with rank_synchronized_stage("percentile coverage", device="cpu"):
                _percentile_coverage_preflight(
                    coverage_sources=[("train", store, [0])],
                    n_sae_features=N_SAE,
                    feature_mask=None,
                    fp_percentile_table=_table([1]),
                    sft_sampling_bundle=None,
                    cluster_threshold_percentile_table=None,
                    missing_percentile_policy=MissingPercentilePolicy.ERROR,
                    device="cpu",
                    startup_trace=None,
                )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_scan_runs_only_on_rank_zero_and_bitmaps_match() -> None:
    results = run_two_ranks(_gloo_preflight_success_worker, "preflight_success")
    rank0_lookups, rank0_policy, rank0_raw, rank0_masked = results[0]
    rank1_lookups, rank1_policy, rank1_raw, rank1_masked = results[1]
    assert rank0_lookups == 3, results  # 2 train + 1 val documents, walked once
    assert rank1_lookups == 0, results  # rank 1 never touched either store
    assert rank0_policy == rank1_policy == "error"
    # Union {1,5,9} ∪ {2}; whitelist (1,9) keeps 2. Rank 1 can only know the
    # exact id set through the broadcast bitmap.
    assert (rank0_raw, rank0_masked) == ([1, 2, 5, 9], 2), results
    assert (rank1_raw, rank1_masked) == ([1, 2, 5, 9], 2), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_rank0_scan_failure_aborts_both() -> None:
    results = run_two_ranks(_gloo_preflight_rank0_failure_worker, "preflight_failure")
    for rank in (0, 1):
        message = results[rank][0]
        assert "sft_percentile_id_scan" in message, message
        assert "rank 0" in message, message
        assert "cannot read shard shard_00001" in message, message


def _gloo_preflight_empty_sources_worker(rank, world_size, init_file, result_queue) -> None:
    """Zero sources still runs the (unconditional) broadcast — no hang."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        trace = sft._SFTStartupTrace()
        with rank_synchronized_stage("percentile coverage", device="cpu"):
            resolver = _percentile_coverage_preflight(
                coverage_sources=[],
                n_sae_features=N_SAE,
                feature_mask=None,
                fp_percentile_table=None,
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=trace,
            )
        scan_extra = _phase_extra(trace, "sft_percentile_id_scan")
        result_queue.put((rank, resolver.policy.value, scan_extra["raw_ids"]))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}", -1))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_preflight_train_only_success_worker(rank, world_size, init_file, result_queue) -> None:
    """The TRAIN-ONLY source shape (the eval_every=0 case) through the real
    two-rank collective path, with the exact broadcast bitmap captured."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        lookups: list[int] = []
        store = _CountingStore(_store({0: [1, 5], 1: [9]}), lookups)
        captured = _capture_broadcasts()
        trace = sft._SFTStartupTrace()
        with rank_synchronized_stage("percentile coverage", device="cpu"):
            resolver = _percentile_coverage_preflight(
                coverage_sources=[("train", store, [0, 1])],
                n_sae_features=N_SAE,
                feature_mask=None,
                fp_percentile_table=_table([1, 5, 9]),
                sft_sampling_bundle=None,
                cluster_threshold_percentile_table=None,
                missing_percentile_policy=MissingPercentilePolicy.ERROR,
                device="cpu",
                startup_trace=trace,
            )
        assert len(captured) == 1
        result_queue.put(
            (
                rank,
                len(lookups),
                resolver.policy.value,
                captured[0].nonzero().squeeze(1).tolist(),
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, -1, f"{type(exc).__name__}: {exc}", []))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_preflight_real_store_failure_worker(
    rank, world_size, init_file, result_queue, family_dir
) -> None:
    """Compose Gloo alignment with the REAL store + ID-loader seam: both ranks
    build the store from clean artifacts, then rank 0 corrupts a non-first
    shard body and the scan must abort BOTH ranks aligned."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        feature_dir = str(Path(family_dir) / "features")
        parquet_dir = str(Path(family_dir) / "parquet")
        store = load_sft_features(feature_dir, parquet_dir)
        dist.barrier()
        if rank == 0:
            shard1 = next(Path(feature_dir).glob("shard_00001*"))
            shard1.write_bytes(b"not a zstd stream")
        dist.barrier()
        try:
            with rank_synchronized_stage("percentile coverage", device="cpu"):
                _percentile_coverage_preflight(
                    coverage_sources=[("train", store, sorted(store))],
                    n_sae_features=N_SAE,
                    feature_mask=None,
                    fp_percentile_table=_table(list(range(N_SAE))),
                    sft_sampling_bundle=None,
                    cluster_threshold_percentile_table=None,
                    missing_percentile_policy=MissingPercentilePolicy.ERROR,
                    device="cpu",
                    startup_trace=None,
                )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_empty_sources_broadcast_does_not_hang() -> None:
    results = run_two_ranks(_gloo_preflight_empty_sources_worker, "preflight_empty_sources")
    assert results[0] == ("error", 0), results
    assert results[1] == ("error", 0), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_train_only_source_succeeds() -> None:
    results = run_two_ranks(_gloo_preflight_train_only_success_worker, "preflight_train_only")
    rank0_lookups, rank0_policy, rank0_raw = results[0]
    rank1_lookups, rank1_policy, rank1_raw = results[1]
    assert rank0_lookups == 2, results
    assert rank1_lookups == 0, results
    assert rank0_policy == rank1_policy == "error"
    assert rank0_raw == rank1_raw == [1, 5, 9], results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_ranks_real_store_scan_failure_aborts_both(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    _write_custom_shard(
        feature_dir,
        "shard_00000",
        doc_ids=[0],
        segment_indices=[0],
        seg_feature_ids=[[1]],
        n_docs=1,
    )
    _write_parquet(parquet_dir, "shard_00000", original_indices=[0])
    _write_custom_shard(
        feature_dir,
        "shard_00001",
        doc_ids=[0],
        segment_indices=[0],
        seg_feature_ids=[[2]],
        n_docs=1,
    )
    _write_parquet(parquet_dir, "shard_00001", original_indices=[1])
    results = run_two_ranks(
        _gloo_preflight_real_store_failure_worker,
        "preflight_real_store",
        extra_args=(str(tmp_path),),
    )
    for rank in (0, 1):
        message = results[rank][0]
        assert "sft_percentile_id_scan" in message, message
        assert "rank 0" in message, message
        assert "ZstdError" in message, message
