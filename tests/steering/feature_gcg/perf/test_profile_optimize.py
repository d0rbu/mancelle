"""Perf test: GPU-gated profiling harness + the measure-first profile run (Task 1).

GPU-independent self-tests (always runnable on CPU)
-----------------------------------------------------
test_wait_returns_idle_index
    Verifies wait_for_idle_local_gpu with a mocked nvidia-smi (two-poll fixture:
    first poll finds no idle GPU, second poll finds GPU 0 idle at 2% util / 80 GiB).

test_wait_timeout_zero_raises_without_sleeping
    timeout_s=0 with a busy GPU raises TimeoutError before any time.sleep call
    (the timeout is checked before the first sleep).

test_timer_cuda_branch_uses_cuda_events / test_timer_cpu_branch_uses_perf_counter
    The device-aware _Timer dispatches to torch.cuda.Event timing for a "cuda"
    device-type (verified via monkeypatched cuda Event + synchronize, no real GPU)
    and to perf_counter for "cpu".

test_profile_optimize_tiny_model_lm
    Runs profile_optimize with lm_loss_on_masked_tokens on the randomized tiny
    EncoderDecoderGPT (CPU, float32, SDPA).  Asserts the exact 5 region names are
    present, every region has a finite positive total_ms, sum(region_ms) <=
    total_ms * 1.05, and the id_weight_cat_only informational metric is present.

test_profile_optimize_tiny_model_class_margin
    Same check with class_margin_objective (POSITIVE mode) and ClassMarginMember
    pairs; validates the class-margin hot path (the headline adversary).

@slow GPU-gated test (written but NOT RUN — pending a free GPU)
---------------------------------------------------------------
test_profile_baseline_on_real_model
    Loads the real d12_l12 encdec checkpoint (FEATURE_GCG_D12_CKPT env var),
    builds real members from AxBench source-run data (FEATURE_GCG_L12_RUN),
    profiles both lm_loss and class_margin objectives, writes
    tests/steering/feature_gcg/perf/artifacts/profile_baseline.json, and asserts the
    sanity invariants.  Skip-not-fail if either env var is unset or path absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import safetensors.torch as st
import torch
from loguru import logger

import nanochat.common
import nanochat.flash_attention as fa_module
import nanochat.gpt
import steering.conditional.encoder_decoder.encoder as enc_module
import steering.conditional.encoder_decoder.model as model_module
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember
from steering.conditional.feature_gcg.objective import (
    ClassMarginMember,
    class_margin_objective,
    lm_loss_on_masked_tokens,
)
from steering.eval.axbench_base.types import SteeringMode
from tests.steering.feature_gcg.functional.test_feature_gradient import _build_randomized_model
from tests.steering.feature_gcg.perf import gpu_wait
from tests.steering.feature_gcg.perf.gpu_wait import wait_for_idle_local_gpu
from tests.steering.feature_gcg.perf.profile_harness import (
    INFO_NAMES,
    REGION_NAMES,
    _Timer,
    profile_optimize,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CPU = torch.device("cpu")
_MODE = SteeringMode.POSITIVE

# Artifact directory (relative to this file).
ARTIFACTS = Path(__file__).parent / "artifacts"

# Tiny-model meta-data: no negative_selection → has_negative=False.
# feature_selection.max_k=6 (exclusive) → max realised count = 5.
_TINY_META: dict[str, Any] = {
    "user_config": {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 3.0,
            "min_k": 0,
            "max_k": 6,
        }
    }
}

# Small config for the CPU self-tests (tiny model + 3 steps → fast).
_TINY_CFG = FeatureGcgConfig(
    max_steps=3,
    candidate_batch_size=2,
    top_k=4,
    temperature=1.0,
    master_seed=0,
    minibatch_size=2,
    reg_lambda=0.0,
    min_k_pos=0,
    max_k_pos=6,
)

# Warm start for the tiny model (buckets 1–3 are valid for num_buckets=6).
_TINY_WARM = FeatureConditioning(entries=((17, 2), (23, 3)))

# Context for the tiny model (slot 0).
_TINY_CONTEXT = FeatureConditioning(entries=((5, 1), (11, 2)))


# ---------------------------------------------------------------------------
# Autouse: force SDPA (FlexAttention has no CPU backward)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SDPA backend (FlexAttention has no CPU backward in tests)."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ---------------------------------------------------------------------------
# Helpers: tiny-model member builders
# ---------------------------------------------------------------------------


def _force_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch COMPUTE_DTYPE to float32 in every forward-path module."""
    for mod in (nanochat.common, nanochat.gpt, enc_module, model_module):
        monkeypatch.setattr(mod, "COMPUTE_DTYPE", torch.float32)


def _make_tiny_lm_member(seed: int, t_len: int = 12, n_scored: int = 4) -> FeatureGradMember:
    """Build a single FeatureGradMember for the LM-loss objective on the tiny model."""
    g = torch.Generator().manual_seed(seed)
    half = t_len // 2
    seg_map = torch.tensor([[0] * half + [1] * (t_len - half)], dtype=torch.long, device=_CPU)
    dec_input = torch.randint(1, 100, (1, t_len), generator=g, device=_CPU)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=_CPU)
    for pos in range(t_len - n_scored, t_len):
        dec_target[0, pos] = int(torch.randint(1, 100, (1,), generator=g).item())
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[_TINY_CONTEXT, FeatureConditioning(entries=())],
        seg_map=seg_map,
        generation_slot=1,
    )


def _make_tiny_cm_member(seed: int, t_len: int = 12, n_scored: int = 4) -> ClassMarginMember:
    """Build a ClassMarginMember (rich, avoid) pair for the tiny model."""
    g = torch.Generator().manual_seed(seed)

    def _one(gen: torch.Generator) -> FeatureGradMember:
        half = t_len // 2
        seg_map = torch.tensor([[0] * half + [1] * (t_len - half)], dtype=torch.long)
        dec_input = torch.randint(1, 100, (1, t_len), generator=gen)
        dec_target = torch.full((1, t_len), -1, dtype=torch.long)
        for pos in range(t_len - n_scored, t_len):
            dec_target[0, pos] = int(torch.randint(1, 100, (1,), generator=gen).item())
        return FeatureGradMember(
            dec_input=dec_input,
            dec_target=dec_target,
            segment_conditions=[_TINY_CONTEXT, FeatureConditioning(entries=())],
            seg_map=seg_map,
            generation_slot=1,
        )

    return ClassMarginMember(rich=_one(g), avoid=_one(g))


# ---------------------------------------------------------------------------
# GPU-independent self-test 1: wait_for_idle_local_gpu (mocked nvidia-smi)
# ---------------------------------------------------------------------------


def test_wait_returns_idle_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait_for_idle_local_gpu returns the first idle GPU index across two polls.

    First poll: GPU 0 at 90% util / 0.98 GiB free, GPU 1 at 80% / 1.95 GiB — both busy.
    Second poll: GPU 0 at 2% util / 80 GiB free — idle; should return 0.
    """
    outputs = iter([
        "0, 90, 1000\n1, 80, 2000\n",   # poll 1: both busy
        "0, 2, 81920\n1, 80, 2000\n",   # poll 2: GPU 0 idle (81920 MiB ≈ 80 GiB)
    ])
    monkeypatch.setattr(gpu_wait, "_query_nvidia_smi", lambda: next(outputs))
    result = wait_for_idle_local_gpu(max_util_pct=5, min_free_gib=20, poll_s=0)
    assert result == 0


def test_wait_timeout_zero_raises_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeout_s=0 with a busy GPU raises TimeoutError WITHOUT calling time.sleep.

    Locks the invariant that the deadline is checked before the first sleep.
    """
    monkeypatch.setattr(gpu_wait, "_query_nvidia_smi", lambda: "0, 99, 100\n1, 88, 200\n")
    slept: list[float] = []
    monkeypatch.setattr(gpu_wait.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(TimeoutError):
        wait_for_idle_local_gpu(max_util_pct=5, min_free_gib=20, poll_s=30, timeout_s=0)

    assert slept == [], "wait_for_idle_local_gpu slept before the timeout fired"


class _RecordingLogger:
    """Captures ``warning()`` calls; no-ops ``info()`` (stand-in for loguru logger)."""

    def __init__(self) -> None:
        self.warnings: list[tuple[Any, ...]] = []

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(args)


def test_wait_warns_and_skips_non_integer_utilization(monkeypatch: pytest.MonkeyPatch) -> None:
    """D5: an ``[N/A]`` utilization line is SKIPPED with a logged WARNING (not silent),
    and a real idle GPU on the SAME poll is still returned.

    Teeth: reverting the parse-failure branch to a silent ``continue`` (no warning)
    leaves ``rec.warnings`` empty → the warning assertion REDs.
    """
    rec = _RecordingLogger()
    monkeypatch.setattr(gpu_wait, "logger", rec)
    # GPU 0 has an unparseable utilization '[N/A]'; GPU 1 is idle (2% / 80 GiB).
    monkeypatch.setattr(
        gpu_wait, "_query_nvidia_smi", lambda: "0, [N/A], 80000\n1, 2, 81920\n"
    )
    result = wait_for_idle_local_gpu(max_util_pct=5, min_free_gib=20, poll_s=0)
    assert result == 1, "the idle GPU 1 must be found despite GPU 0's [N/A] utilization"
    assert rec.warnings, (
        "the non-integer utilization line must log a WARNING (not be skipped silently)"
    )


def test_wait_non_integer_utilization_honors_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """D5: a persistently-unparseable GPU does NOT hang or crash — a finite timeout fires.

    ``timeout_s=0`` with an all-``[N/A]`` poll raises TimeoutError before sleeping,
    proving the parse-failure path cannot defeat the deadline (the hang the
    controller hit came from this branch with ``timeout_s=None``).
    """
    rec = _RecordingLogger()
    monkeypatch.setattr(gpu_wait, "logger", rec)
    monkeypatch.setattr(gpu_wait, "_query_nvidia_smi", lambda: "0, [N/A], 80000\n")
    slept: list[float] = []
    monkeypatch.setattr(gpu_wait.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(TimeoutError):
        wait_for_idle_local_gpu(max_util_pct=5, min_free_gib=20, poll_s=30, timeout_s=0)
    assert slept == [], "must raise on the deadline without sleeping on a parse failure"
    assert rec.warnings, "the [N/A] line must still log a warning before the timeout"


# ---------------------------------------------------------------------------
# D3 / D4: synthetic-fallback robustness (loud log + process-stable seed)
# ---------------------------------------------------------------------------


def test_build_real_member_synthetic_seed_is_process_stable() -> None:
    """D4: synthetic-fallback tokens use a PROCESS-STABLE sha256 seed, not ``hash()``.

    Python ``str`` hashing is PYTHONHASHSEED-randomized per process, so the baseline
    and batched profile runs (separate processes) would otherwise get DIFFERENT
    synthetic tokens (non-apples-to-apples speedup).  Reproduces the expected tokens
    from the sha256 seed and asserts a match — a regression to
    ``hash(prefix_text) & 0xFFFFFFFF`` (a different seed even within THIS process)
    REDs.
    """
    prefix = "a representative concept prefix"
    token_count = 16
    member, label = _build_real_member(prefix, "gen ignored", token_count, tokenizer=None)
    assert label == "length-real"
    # Reproduce the seed independently: sha256(prefix) → first 4 bytes, big-endian.
    expected_seed = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:4], "big")
    g = torch.Generator().manual_seed(expected_seed)
    expected_input = torch.randint(1, 32768, (1, token_count), generator=g)
    assert torch.equal(member.dec_input, expected_input), (
        "synthetic dec_input does not match the sha256-seeded reference — the seed "
        "is not process-stable (regressed to hash()?)"
    )


def test_build_real_member_encode_failure_logs_loudly_and_falls_back() -> None:
    """D3: a ``tokenizer.encode`` failure logs an ERROR and falls back to synthetic.

    Teeth: reverting to the blanket ``except Exception: pass`` (no log) leaves the
    captured ERROR sink empty → the 'logged loudly' assertion REDs (the silent
    fallback to FAKE data is exactly what D3 prevents).
    """

    class _BadTokenizer:
        def encode(self, text: str) -> list[int]:
            raise ValueError("disallowed special token in the prefix")

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="ERROR")
    try:
        member, label = _build_real_member("prefix", "gen", 16, tokenizer=_BadTokenizer())
    finally:
        logger.remove(sink_id)

    assert label == "length-real", "must fall back to synthetic tokens on encode failure"
    assert any("encode failed" in r for r in records), (
        "the tokenizer.encode failure must be logged LOUDLY (not silently swallowed)"
    )


# ---------------------------------------------------------------------------
# GPU-independent self-test: device-aware _Timer branch selection (no real GPU)
# ---------------------------------------------------------------------------


def test_timer_cuda_branch_uses_cuda_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """_Timer('cuda') dispatches to torch.cuda.Event timing (no real GPU needed).

    Monkeypatches torch.cuda.Event + torch.cuda.synchronize to verify the CUDA
    branch (a) creates two enable_timing events, (b) runs the timed fn, (c)
    synchronizes once before reading elapsed_time.
    """
    events_created: list[bool] = []
    sync_calls: list[int] = []

    class _FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            events_created.append(enable_timing)

        def record(self) -> None:
            pass

        def elapsed_time(self, other: object) -> float:
            return 12.5

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(1))

    timer = _Timer("cuda")
    assert timer.is_cuda is True

    ran: list[str] = []

    def _work() -> int:
        ran.append("x")
        return 7

    result, ms = timer.time(_work)

    assert result == 7
    assert ran == ["x"]                       # the timed fn actually ran
    assert ms == 12.5                         # elapsed_time() was used
    assert events_created == [True, True]     # start + end, both enable_timing
    assert len(sync_calls) == 1               # synchronized once before reading


def test_timer_cpu_branch_uses_perf_counter() -> None:
    """_Timer('cpu') uses perf_counter (no CUDA), returning a finite ms >= 0."""
    timer = _Timer("cpu")
    assert timer.is_cuda is False
    result, ms = timer.time(lambda: 5)
    assert result == 5
    assert math.isfinite(ms) and ms >= 0.0


# ---------------------------------------------------------------------------
# GPU-independent self-test 2: profile_optimize on the tiny model (LM loss)
# ---------------------------------------------------------------------------


def test_profile_optimize_tiny_model_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    """profile_optimize with lm_loss_on_masked_tokens on the tiny randomized model.

    Uses CPU / float32 / SDPA (no GPU required).  Asserts (via _assert_breakdown):
    - The EXACT 5 region names {losses, gradients, extract_feature_grads,
      rank_candidates, _add_edits}.
    - Every region has finite, positive total_ms.
    - sum(region_ms) <= total_ms * 1.05 (exclusive-region invariant).
    - The id_weight_cat_only informational metric is present.
    """
    _force_float32(monkeypatch)
    torch.manual_seed(0)
    model = _build_randomized_model(monkeypatch, seed=42)
    model._loaded_meta_data = _TINY_META

    members = [_make_tiny_lm_member(seed=i) for i in range(2)]

    breakdown = profile_optimize(
        model,
        members,
        _TINY_WARM,
        _TINY_CFG,
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=_CPU),
        device=_CPU,
        repeats=1,
    )

    _assert_breakdown(breakdown)


# ---------------------------------------------------------------------------
# GPU-independent self-test 3: profile_optimize on the tiny model (class margin)
# ---------------------------------------------------------------------------


def test_profile_optimize_tiny_model_class_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """profile_optimize with class_margin_objective (POSITIVE mode) on the tiny model.

    The class-margin path is the headline adversary hot path in Phase 1.
    Same assertions as the LM-loss self-test.
    """
    _force_float32(monkeypatch)
    torch.manual_seed(0)
    model = _build_randomized_model(monkeypatch, seed=42)
    model._loaded_meta_data = _TINY_META

    members = [_make_tiny_cm_member(seed=i) for i in range(2)]

    breakdown = profile_optimize(
        model,
        members,
        _TINY_WARM,
        _TINY_CFG,
        objective_factory=lambda m: class_margin_objective(m, mode=_MODE, device=_CPU),
        device=_CPU,
        repeats=1,
    )

    _assert_breakdown(breakdown)


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------


def _assert_breakdown(breakdown: dict[str, Any]) -> None:
    """Common sanity checks on a profile_optimize return value.

    Asserts the EXACT 5-region name set (a mislabeled region would otherwise pass
    a bare ``len == 5`` check), every region's stats are finite/positive, the
    exclusive-region sum invariant holds, and the ``id_weight_cat_only``
    informational sub-metric is present.
    """
    assert "regions" in breakdown, "breakdown missing 'regions' key"
    assert "info" in breakdown, "breakdown missing 'info' key"
    assert "total_ms" in breakdown, "breakdown missing 'total_ms' key"

    regions = breakdown["regions"]
    info = breakdown["info"]
    total_ms = breakdown["total_ms"]

    # Exact name-set check (not just the count) so a renamed/mislabeled region fails.
    assert set(regions.keys()) == set(REGION_NAMES), (
        f"region names {sorted(regions)} != expected {sorted(REGION_NAMES)}"
    )

    for name, stats in regions.items():
        assert math.isfinite(stats["total_ms"]), f"region {name!r}: total_ms is not finite"
        assert stats["total_ms"] > 0, f"region {name!r}: total_ms={stats['total_ms']} is not positive"
        assert stats["calls"] > 0, f"region {name!r}: calls={stats['calls']} is not positive"
        assert math.isfinite(stats["ms_per_call"]), f"region {name!r}: ms_per_call is not finite"

    # Informational id_weight_cat_only must be present (the #12-opt target metric).
    assert set(INFO_NAMES) <= set(info.keys()), (
        f"info section missing {sorted(set(INFO_NAMES) - set(info))}; got {sorted(info)}"
    )
    cat_only = info["id_weight_cat_only"]
    assert math.isfinite(cat_only["total_ms"]) and cat_only["total_ms"] >= 0.0, (
        f"id_weight_cat_only.total_ms={cat_only['total_ms']} is not finite/non-negative"
    )
    assert cat_only["calls"] > 0, "id_weight_cat_only was never recorded (calls == 0)"

    # Exclusive-region sum invariant (id_weight_cat_only is NOT part of this sum).
    region_sum = sum(r["total_ms"] for r in regions.values())
    assert region_sum <= total_ms * 1.05, (
        f"sum of region times ({region_sum:.2f} ms) > total_ms ({total_ms:.2f} ms) * 1.05; "
        f"regions may be double-counting (non-exclusive overlap too large)"
    )

    assert math.isfinite(total_ms) and total_ms > 0, f"total_ms={total_ms} is not finite/positive"


# ---------------------------------------------------------------------------
# @slow GPU-gated profile test (written; NOT RUN — pending a free GPU)
# ---------------------------------------------------------------------------


def _skip_unless_env(var: str, desc: str) -> Path:
    """Return Path from env var or skip-not-fail with a clear message."""
    val = os.environ.get(var, "")
    if not val:
        pytest.skip(
            f"{var} is not set; set it to {desc} to run this test.\n"
            f"Example: {var}=/path/to/{var.lower().replace('_', '-')} uv run pytest "
            "tests/steering/feature_gcg/perf/ -m slow -q"
        )
    p = Path(val)
    if not p.exists():
        pytest.skip(
            f"{var}={p} does not exist; "
            "make sure the path is correct and accessible."
        )
    return p


def _load_tokenizer() -> Any | None:
    """Load the cached tiktoken tokenizer (nanochat_cache/tokenizer/tokenizer.pkl).

    Returns None if not found (caller falls back to synthetic tokens).
    """
    tok_path = Path(__file__).parent.parent.parent.parent / "nanochat_cache" / "tokenizer" / "tokenizer.pkl"
    if not tok_path.exists():
        logger.warning(
            "tokenizer pickle not found at {}; the @slow baseline will fall back to "
            "SYNTHETIC tokens (label 'length-real') — speedups are on fake data.",
            tok_path,
        )
        return None
    try:
        with tok_path.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ImportError, AttributeError, ValueError) as exc:
        # Narrow to the realistic unpickle failures (corruption / version / missing
        # class) and log LOUDLY (D3): a real tokenizer breakage must NOT be hidden
        # behind a silent synthetic fallback that yields speedups on fake data.
        logger.error(
            "failed to unpickle the tokenizer at {} ({!r}); falling back to "
            "SYNTHETIC tokens — the @slow speedup will be measured on FAKE data.",
            tok_path,
            exc,
        )
        return None


def _build_real_member(
    prefix_text: str,
    gen_text: str,
    token_count: int,
    tokenizer: Any | None,
) -> tuple[FeatureGradMember, str]:
    """Build a FeatureGradMember from a real (or synthetic) source.

    Returns (member, artifact_label) where artifact_label is 'tokenized'
    or 'length-real' (indicating synthetic tokens of realistic length).
    """
    if tokenizer is not None:
        try:
            prefix_ids = tokenizer.encode(prefix_text)
            gen_ids = tokenizer.encode(gen_text)
            # Truncate to avoid OOM on very long prefixes.
            max_prefix = 256
            prefix_ids = prefix_ids[-max_prefix:]
            total_ids = prefix_ids + gen_ids
            prefix_len = len(prefix_ids)
            total_len = len(total_ids)

            dec_input = torch.tensor([total_ids], dtype=torch.long)
            dec_target = torch.full((1, total_len), -1, dtype=torch.long)
            dec_target[0, prefix_len:] = torch.tensor(gen_ids, dtype=torch.long)
            seg_map = torch.zeros((1, total_len), dtype=torch.long)
            seg_map[0, prefix_len:] = 1

            member = FeatureGradMember(
                dec_input=dec_input,
                dec_target=dec_target,
                segment_conditions=[
                    FeatureConditioning(entries=()),   # context segment (slot 0)
                    FeatureConditioning(entries=()),   # generation placeholder (slot 1)
                ],
                seg_map=seg_map,
                generation_slot=1,
            )
            return member, "tokenized"
        except (ValueError, KeyError, RuntimeError, UnicodeEncodeError) as exc:
            # Narrow to the realistic encode failures (disallowed special token /
            # vocab / encoding) and log LOUDLY (D3): a real encode breakage must NOT
            # be hidden behind a silent synthetic fallback.
            logger.error(
                "tokenizer.encode failed ({!r}); falling back to SYNTHETIC tokens "
                "for this member — the @slow baseline will measure FAKE data.",
                exc,
            )

    # Fallback: synthetic tokens of the correct generation length.  Seed
    # DETERMINISTICALLY (D4) via sha256 — NOT Python ``hash()``, whose str hashing
    # is PYTHONHASHSEED-randomized PER PROCESS, so the baseline and batched profile
    # runs (separate processes) would otherwise get DIFFERENT synthetic tokens and
    # the measured speedup would not be apples-to-apples.
    seed = int.from_bytes(hashlib.sha256(prefix_text.encode()).digest()[:4], "big")
    g = torch.Generator().manual_seed(seed)
    t_len = token_count  # real generation token count
    half = t_len // 2
    dec_input = torch.randint(1, 32768, (1, t_len), generator=g)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long)
    dec_target[0, half:] = torch.randint(1, 32768, (t_len - half,), generator=g)
    seg_map = torch.zeros((1, t_len), dtype=torch.long)
    seg_map[0, half:] = 1

    member = FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[
            FeatureConditioning(entries=()),
            FeatureConditioning(entries=()),
        ],
        seg_map=seg_map,
        generation_slot=1,
    )
    return member, "length-real"


def _load_real_l12_workload(
    ckpt_dir: Path,
    run_dir: Path,
    minibatch_size: int,
) -> tuple[Any, list[FeatureGradMember], list[ClassMarginMember], FeatureConditioning, FeatureGcgConfig, str]:
    """Load d12_l12 model + real members from the source-run data.

    Returns (model, lm_members, cm_members, warm_start, config, artifact_label).

    lm_members: minibatch_size FeatureGradMember's for lm_loss_on_masked_tokens.
    cm_members: minibatch_size ClassMarginMember's for class_margin_objective.
                Paired arbitrarily from consecutive rows (perf-identical to semantic pairing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the d12_l12 checkpoint.
    model, _tok, meta_data = build_encdec_model(
        checkpoint_dir=str(ckpt_dir),
        step=int(os.environ.get("FEATURE_GCG_D12_STEP", "54600")),
        device=str(device),
        phase=ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.deferred("feature_gcg_perf_profile"),
    )
    model._loaded_meta_data = meta_data  # ty: ignore[invalid-assignment]
    model.eval()

    # Config: d12_l12 has num_buckets=2, has_negative=True (negative_selection present).
    # min_k_pos=1: warm_start must have at least 1 positive feature.
    cfg = FeatureGcgConfig(
        max_steps=10,
        candidate_batch_size=4,
        top_k=16,
        temperature=1.0,
        master_seed=0,
        minibatch_size=minibatch_size,
        reg_lambda=0.0,
        min_k_pos=1,
        max_k_pos=40,
        min_k_neg=0,
        max_k_neg=40,
    )

    # Load concept_features: feature_ids (10, 256) int32, -1-padded.
    feat = st.load_file(run_dir / "concept_features.safetensors")
    feature_ids_all = feat["feature_ids"]  # (10, 256) int32

    # Use concept 0: its non-(-1) features as the warm-start bag (bucket 0 = active).
    concept_fids = [int(f) for f in feature_ids_all[0].tolist() if f != -1]
    # One-feature warm start (min_k_pos=1 requires at least 1 positive entry).
    warm_start = FeatureConditioning(entries=((concept_fids[0], 0),))

    # Load generations for concept 0 (batch_000000.parquet).
    gen_dir = run_dir / "generations"
    batch_file = sorted(gen_dir.glob("batch_*.parquet"))[0]
    table = pq.read_table(batch_file)

    # Need 2 * minibatch_size rows to form LM members + CM pairs.
    n_rows = min(2 * minibatch_size, len(table))
    tokenizer = _load_tokenizer()
    artifact_labels: list[str] = []

    raw_members: list[FeatureGradMember] = []
    for i in range(n_rows):
        prefix_text = table.column("prefix_text")[i].as_py()
        gen_text = table.column("generation_text")[i].as_py()
        token_count = table.column("token_count")[i].as_py()
        member, label = _build_real_member(prefix_text, gen_text, token_count, tokenizer)
        raw_members.append(member)
        artifact_labels.append(label)

    # LM members: first minibatch_size rows.
    lm_members = raw_members[:minibatch_size]

    # CM members: paired as (raw_members[0], raw_members[1]), etc.
    # Semantically arbitrary pairing — perf-identical to a meaningful pairing.
    cm_members = [
        ClassMarginMember(rich=raw_members[2 * j], avoid=raw_members[2 * j + 1])
        for j in range(minibatch_size)
        if 2 * j + 1 < len(raw_members)
    ]
    # Pad if we got fewer pairs than minibatch_size.
    while len(cm_members) < minibatch_size and len(lm_members) >= 2:
        cm_members.append(ClassMarginMember(rich=lm_members[0], avoid=lm_members[-1]))

    artifact_label = artifact_labels[0] if artifact_labels else "length-real"
    return model, lm_members, cm_members, warm_start, cfg, artifact_label


def _cm_factory(device: Any) -> Any:
    """objective_factory for class_margin_objective (POSITIVE mode)."""
    def _factory(m: Any) -> Any:
        return class_margin_objective(m, mode=_MODE, device=device)
    return _factory


def _lm_factory(device: Any) -> Any:
    """objective_factory for lm_loss_on_masked_tokens."""
    def _factory(m: Any) -> Any:
        return lm_loss_on_masked_tokens(m, device=device)
    return _factory


@pytest.mark.slow
def test_profile_baseline_on_real_model() -> None:
    """[Task 1 @slow] Profile both objectives on the real d12_l12 encdec model.

    Loads the d12_l12 checkpoint and source-run data from env vars, profiles
    lm_loss_on_masked_tokens AND class_margin_objective, writes the baseline
    artifact to tests/steering/feature_gcg/perf/artifacts/profile_baseline.json, and
    asserts the sanity invariants.

    Env vars
    --------
    FEATURE_GCG_D12_CKPT   path to the d12 encdec checkpoint directory (required)
    FEATURE_GCG_D12_STEP   checkpoint step (default 54600)
    FEATURE_GCG_L12_RUN    path to the d12_l12 source-run directory (required)
                           Must contain: generations/batch_*.parquet,
                           concept_features.safetensors, concepts.json.
    FEATURE_GCG_D12_L12_MINIBATCH_SIZE  number of members per profile run (default 4)

    Skip-not-fail when either env var is unset or the path does not exist.

    Note: Tokenization uses nanochat_cache/tokenizer/tokenizer.pkl (tiktoken).
    Falls back to synthetic tokens of length token_count=64 if the tokenizer is
    absent, and labels the artifact 'length-real' instead of 'tokenized'.

    Note on class_margin pairing: rich/avoid are paired arbitrarily from
    consecutive rows in the parquet.  The perf profile is identical to a
    semantically-correct pairing since it is just 2 forwards/item.
    """
    ckpt_dir = _skip_unless_env(
        "FEATURE_GCG_D12_CKPT",
        "the d12 encdec checkpoint directory (containing model_054600.pt + meta_054600.json)",
    )
    run_dir = _skip_unless_env(
        "FEATURE_GCG_L12_RUN",
        "the d12_l12 source-run directory (containing generations/, concept_features.safetensors)",
    )

    gpu = wait_for_idle_local_gpu()
    device = f"cuda:{gpu}"

    minibatch_size = int(os.environ.get("FEATURE_GCG_D12_L12_MINIBATCH_SIZE", "4"))

    model, lm_members, cm_members, warm_start, cfg, artifact_label = _load_real_l12_workload(
        ckpt_dir, run_dir, minibatch_size
    )

    # Profile lm_loss_on_masked_tokens.
    breakdown_lm = profile_optimize(
        model,
        lm_members,
        warm_start,
        cfg,
        objective_factory=_lm_factory(device),
        device=device,
        repeats=3,
    )

    # Profile class_margin_objective (POSITIVE mode).
    # This is the headline adversary hot path — it is measured separately so
    # the report captures the per-objective breakdown.
    breakdown_cm = profile_optimize(
        model,
        cm_members,
        warm_start,
        cfg,
        objective_factory=_cm_factory(device),
        device=device,
        repeats=3,
    )

    # Combine into artifact.
    artifact = {
        "model": "d12_l12",
        "checkpoint_step": int(os.environ.get("FEATURE_GCG_D12_STEP", "54600")),
        "device": device,
        "n_members": minibatch_size,
        "member_artifact_label": artifact_label,
        "note_class_margin_pairing": (
            "rich/avoid paired arbitrarily from consecutive rows; "
            "perf-identical to a semantically-correct pairing (2 forwards/item)"
        ),
        "lm_loss": breakdown_lm,
        "class_margin": breakdown_cm,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS / "profile_baseline.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n[Task 1] Profile artifact written to: {artifact_path}")

    # Sanity invariants on both breakdowns (reuse the shared CPU-test helper so
    # the exact-name-set + info-presence checks apply on the real model too).
    _assert_breakdown(breakdown_lm)
    _assert_breakdown(breakdown_cm)

    # Print region summary (+ the id_weight_cat_only sub-metric) for inspection.
    # This ranking is the headline output that sets the priority order for the
    # later opt tasks (#12 targets the id_weight cat specifically).
    for obj_name, breakdown in [("lm_loss", breakdown_lm), ("class_margin", breakdown_cm)]:
        print(f"\n[{obj_name}] total_ms={breakdown['total_ms']:.1f}")
        for region, stats in sorted(breakdown["regions"].items(), key=lambda kv: -kv[1]["total_ms"]):
            pct = 100.0 * stats["total_ms"] / breakdown["total_ms"]
            print(f"  {region:24s}: {stats['total_ms']:8.1f} ms  ({pct:.1f}%)  {stats['calls']} calls")
        cat = breakdown["info"]["id_weight_cat_only"]
        print(
            f"  (info) id_weight_cat_only: {cat['total_ms']:8.1f} ms  {cat['calls']} calls "
            "[#12 opt target, NOT in region sum]"
        )
