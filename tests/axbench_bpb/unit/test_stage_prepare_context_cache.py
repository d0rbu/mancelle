"""In-process unit tests for _stage_prepare_context_cache.

These complement the real-multiprocess integration tests by exercising
the per-rank dispatch branches in a single-process context, providing
deterministic coverage independent of multiprocessing config.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.eval.axbench_bpb_eval import _stage_prepare_context_cache
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from tests.axbench_bpb._mini_runtime import build_mini_runtime_state
from tests.statistics_doubles import fixed_statistics_double


def _stub_all_rank_loaders(monkeypatch):
    """Stub the heavy loaders that fire on ALL ranks (load_statistics +
    build_boundaries_from_stats) plus the rank-0-only loaders (the
    decoder bundle + tokenizer guards + manifest helpers).

    Slice 17 audit (2026-05-05) P1.B added two rank-0 startup guards
    (assert_tokenizer_checksum_matches / assert_bos_lengths_match). The
    parallel-Stage-1 refactor moved load_statistics + build_boundaries
    to run on every rank (they're cheap; CPU-only) — so the test setup
    stubs both classes of loader uniformly.
    """
    import scripts.eval.axbench_bpb_eval as eval_mod
    from steering.eval.axbench_bpb.runtime_state import RuntimeState
    from steering.eval.axbench_bpb.teacher_force import TargetModelBundle

    fake_stats = fixed_statistics_double()
    monkeypatch.setattr(eval_mod, "load_statistics", lambda *_a, **_kw: fake_stats)
    monkeypatch.setattr(
        eval_mod, "build_boundaries_from_stats", lambda *_a, **_kw: MagicMock(),
    )
    # Decoder bundle: only rank 0 touches it, but stubbing here keeps the
    # setup branchless and matches the original test's posture.
    fake_bundle = TargetModelBundle(
        model=MagicMock(),
        tokenizer=MagicMock(),
        max_seq_len=2048,
        target_name="decoder_only",
    )
    monkeypatch.setattr(
        RuntimeState, "decoder_only", lambda self: fake_bundle,
    )
    monkeypatch.setattr(
        eval_mod, "assert_tokenizer_checksum_matches",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        eval_mod, "assert_bos_lengths_match", lambda *_a, **_kw: None,
    )
    # Rank-0-only manifest helpers — stubbed so the manifest-write
    # branch doesn't crash trying to read non-existent parquet shards.
    monkeypatch.setattr(
        eval_mod, "collect_unique_prefixes", lambda *_a, **_kw: {},
    )
    monkeypatch.setattr(
        eval_mod, "write_context_cache_manifest", lambda **_kw: None,
    )


# kills: dropping either witness stage 0 hands the feature-dir guard -- the
# statistics snapshot OR ``cfg.feature_dir``, whose canonical ``turn_delimited``
# leaf is the only witness in the second case.
@pytest.mark.parametrize(
    ("path_mode", "statistics_mode", "message"),
    [
        (
            None,
            SegmentMode.turn_delimited_fixed_tokens,
            r"context-cache extraction does not support.*turn_delimited",
        ),
        (
            SegmentMode.turn_delimited_fixed_tokens,
            SegmentMode.fixed_tokens,
            r"context-cache extraction feature artifact segment-mode mismatch"
            r".*feature path='turn_delimited_fixed_tokens'"
            r".*feature statistics='fixed_tokens'",
        ),
    ],
    ids=["statistics-witness", "feature-path-witness"],
)
def test_stage0_rejects_turn_statistics_before_loading_models_or_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    path_mode: SegmentMode | None,
    statistics_mode: SegmentMode,
    message: str,
) -> None:
    import scripts.eval.axbench_bpb_eval as eval_mod
    from steering.eval.axbench_bpb.runtime_state import RuntimeState

    monkeypatch.setattr(
        eval_mod,
        "load_statistics",
        lambda *_args, **_kwargs: fixed_statistics_double(segment_mode=statistics_mode),
    )
    monkeypatch.setattr(
        RuntimeState,
        "base_gemma_and_sae",
        lambda _self: pytest.fail("turn guard reached Gemma/SAE loading"),
    )
    monkeypatch.setattr(
        eval_mod,
        "build_context_cache",
        lambda **_kwargs: pytest.fail("turn guard reached cache construction"),
    )

    runtime_state = build_mini_runtime_state(
        run_root=tmp_path,
        rank=0,
        world_size=1,
    )
    if path_mode is not None:
        runtime_state.cfg.feature_dir = str(
            build_feature_dir(
                root=tmp_path / "features",
                segment_size=128,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.max_pool,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase=DatasetPhase.BASE,
                segment_mode=path_mode,
            )
        )
    with pytest.raises(ValueError, match=message):
        _stage_prepare_context_cache(runtime_state)


def test_stage0_rank_0_invokes_build(monkeypatch, tmp_path):
    """rank=0 invokes build_context_cache exactly once with its rank/world."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    invocations = []

    def fake_build(**kwargs):
        invocations.append(kwargs)

    monkeypatch.setattr(eval_mod, "build_context_cache", fake_build)

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    _stage_prepare_context_cache(rs)

    assert len(invocations) == 1, (
        f"rank=0 should invoke build_context_cache exactly once; "
        f"got {len(invocations)}"
    )
    assert invocations[0]["ddp_rank"] == 0
    assert invocations[0]["ddp_world_size"] == 1
    assert invocations[0]["concept_ids"] == frozenset({0})


def test_stage0_non_rank_0_invokes_build_with_rank(monkeypatch, tmp_path):
    """Parallel Stage 1: rank=1 ALSO invokes build_context_cache, with
    its own ddp_rank kwarg, so it processes its 1/world_size shard.

    This replaces the previous ``test_stage0_non_rank_0_skips_build`` —
    parallelizing Stage 1 eliminated the rank-0-only gate. Ranks 1+ now
    do real extraction work instead of idling on a barrier.
    """
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    invocations = []

    def fake_build(**kwargs):
        invocations.append(kwargs)

    monkeypatch.setattr(eval_mod, "build_context_cache", fake_build)

    rs = build_mini_runtime_state(run_root=tmp_path, rank=1, world_size=2)
    # _maybe_barrier is a no-op when dist.is_initialized() is False (no group),
    # so this test runs cleanly in-process without setting up gloo.
    _stage_prepare_context_cache(rs)

    assert len(invocations) == 1, (
        f"rank=1 must invoke build_context_cache once (sharded extraction); "
        f"got {len(invocations)}"
    )
    assert invocations[0]["ddp_rank"] == 1
    assert invocations[0]["ddp_world_size"] == 2


def test_stage0_world_size_1_invokes_build_no_barrier(monkeypatch, tmp_path):
    """Single-process invocation (world_size=1) invokes build with no
    barrier blocking (the barrier short-circuits for ws<=1)."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    invocations = []

    def fake_build(**kwargs):
        invocations.append(kwargs)

    monkeypatch.setattr(eval_mod, "build_context_cache", fake_build)

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    _stage_prepare_context_cache(rs)

    assert len(invocations) == 1


def test_stage0_only_rank_0_writes_manifest(monkeypatch, tmp_path):
    """Under DDP, only rank 0 calls write_context_cache_manifest after
    the barrier — guarantees the rank-0 manifest write happens exactly
    once across the world."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    manifest_calls = []
    monkeypatch.setattr(
        eval_mod, "build_context_cache", lambda **_kw: None,
    )
    monkeypatch.setattr(
        eval_mod, "write_context_cache_manifest",
        lambda **kw: manifest_calls.append(kw),
    )

    # Rank 0 (under DDP) writes manifest
    rs0 = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=4)
    _stage_prepare_context_cache(rs0)
    # Rank 2 (under DDP) does NOT write manifest
    rs2 = build_mini_runtime_state(run_root=tmp_path, rank=2, world_size=4)
    _stage_prepare_context_cache(rs2)

    assert len(manifest_calls) == 1, (
        f"Only rank 0 should write the manifest; got {len(manifest_calls)} "
        f"calls across the rank-0+rank-2 runs"
    )


def test_stage0_guards_run_after_manifest_write(monkeypatch, tmp_path):
    """Pin commit 32cc48db's deadlock-class fix: the tokenizer guards
    must run AFTER the manifest write on rank 0. A regression that
    re-orders to "guards-before-manifest" would re-introduce the
    pre-barrier-1 deadlock the commit fixed.

    We use ordered recorders for build → barrier → collect_unique →
    write_manifest → guards. The relative order build → manifest →
    guards is the contract."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    sequence: list[str] = []
    monkeypatch.setattr(
        eval_mod, "build_context_cache",
        lambda **_kw: sequence.append("build"),
    )
    monkeypatch.setattr(
        eval_mod, "collect_unique_prefixes",
        lambda *_a, **_kw: (sequence.append("collect_unique"), {})[1],
    )
    monkeypatch.setattr(
        eval_mod, "write_context_cache_manifest",
        lambda **_kw: sequence.append("write_manifest"),
    )
    monkeypatch.setattr(
        eval_mod, "assert_tokenizer_checksum_matches",
        lambda **_kw: sequence.append("checksum_guard"),
    )
    monkeypatch.setattr(
        eval_mod, "assert_bos_lengths_match",
        lambda **_kw: sequence.append("bos_guard"),
    )

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=4)
    _stage_prepare_context_cache(rs)

    # Required order (deadlock-fix invariant):
    #   build → collect_unique → write_manifest → guards
    # Both guards must come strictly AFTER write_manifest.
    assert sequence.index("write_manifest") > sequence.index("build"), (
        f"write_manifest must come after build; got {sequence!r}"
    )
    assert sequence.index("checksum_guard") > sequence.index("write_manifest"), (
        f"checksum_guard must come after write_manifest; got {sequence!r}"
    )
    assert sequence.index("bos_guard") > sequence.index("write_manifest"), (
        f"bos_guard must come after write_manifest; got {sequence!r}"
    )


def test_stage0_guards_only_run_on_rank_0(monkeypatch, tmp_path):
    """Pin: tokenizer guards execute on rank 0 ONLY. A regression that
    runs them on every rank would (a) load decoder_only on every rank
    (heavy GPU memory) and (b) re-introduce the original
    pre-barrier-1 deadlock for every rank that fails the guard."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    monkeypatch.setattr(
        eval_mod, "build_context_cache", lambda **_kw: None,
    )
    monkeypatch.setattr(
        eval_mod, "write_context_cache_manifest", lambda **_kw: None,
    )
    guard_calls: list[str] = []
    monkeypatch.setattr(
        eval_mod, "assert_tokenizer_checksum_matches",
        lambda **_kw: guard_calls.append("checksum"),
    )
    monkeypatch.setattr(
        eval_mod, "assert_bos_lengths_match",
        lambda **_kw: guard_calls.append("bos"),
    )

    # Rank 0: both guards fire
    rs0 = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=4)
    _stage_prepare_context_cache(rs0)
    assert guard_calls == ["checksum", "bos"], (
        f"Rank 0 must run both guards; got {guard_calls!r}"
    )

    # Rank 2: neither guard fires
    guard_calls.clear()
    rs2 = build_mini_runtime_state(run_root=tmp_path, rank=2, world_size=4)
    _stage_prepare_context_cache(rs2)
    assert guard_calls == [], (
        f"Non-rank-0 must skip both guards; got {guard_calls!r}"
    )


def test_stage0_calls_maybe_barrier_exactly_twice_on_rank_0(monkeypatch, tmp_path):
    """Defense-in-depth: pin that ``_maybe_barrier(ws)`` is invoked
    exactly twice on rank 0 (once between extraction and manifest
    write, once between guards and downstream stages). A regression
    that drops barrier 1 (the post-extraction barrier) would let
    rank 0 read partial safetensors written by other ranks before
    they finish — silent manifest corruption, NOT a deadlock. The
    real-gloo failure tests in
    ``tests/axbench_bpb/integration/test_bpb_real_multiprocess.py``
    do NOT catch this (they only verify the fixture's failure-relay
    path); this test does.
    """
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    monkeypatch.setattr(
        eval_mod, "build_context_cache", lambda **_kw: None,
    )
    monkeypatch.setattr(
        eval_mod, "write_context_cache_manifest", lambda **_kw: None,
    )

    barrier_calls: list[int] = []
    monkeypatch.setattr(
        eval_mod, "_maybe_barrier",
        lambda ws: barrier_calls.append(ws),
    )

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=4)
    _stage_prepare_context_cache(rs)

    assert barrier_calls == [4, 4], (
        f"rank 0 must invoke _maybe_barrier exactly twice (post-extraction "
        f"and post-manifest), each with the configured world_size; got "
        f"{barrier_calls!r}"
    )


def test_stage0_calls_maybe_barrier_exactly_twice_on_non_rank_0(monkeypatch, tmp_path):
    """Companion: non-rank-0 ranks must hit BOTH barriers too,
    otherwise the post-extraction barrier degenerates. Rank 2's
    sequence is build → barrier 1 → barrier 2 (no manifest, no
    guards in between)."""
    import scripts.eval.axbench_bpb_eval as eval_mod

    _stub_all_rank_loaders(monkeypatch)
    monkeypatch.setattr(
        eval_mod, "build_context_cache", lambda **_kw: None,
    )

    barrier_calls: list[int] = []
    monkeypatch.setattr(
        eval_mod, "_maybe_barrier",
        lambda ws: barrier_calls.append(ws),
    )

    rs = build_mini_runtime_state(run_root=tmp_path, rank=2, world_size=4)
    _stage_prepare_context_cache(rs)

    assert barrier_calls == [4, 4], (
        f"non-rank-0 rank must invoke _maybe_barrier exactly twice "
        f"(post-extraction, post-manifest-window); got {barrier_calls!r}"
    )


__all__: list[str] = []
