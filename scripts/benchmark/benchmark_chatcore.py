"""Real-checkpoint ChatCORE timing, profiling, and boolean-grading comparison."""

from __future__ import annotations

import cProfile
import hashlib
import json
import os
import pstats
import resource
import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import torch as t
import triton
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.checkpoint_manager import build_model
from nanochat.common import compute_init, get_base_dir
from nanochat.execution import execute_code
from steering.eval import greedy_token_batch as greedy
from steering.eval.sanity_check import run_all
from tasks import humaneval

_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@hydra.main(version_base=None, config_path="../../configs/profiling", config_name="chatcore")
def main(cfg: DictConfig) -> None:
    if cfg.grading not in ("reference", "candidate"):
        raise ValueError("grading must be reference or candidate")
    if cfg.repeats < 1:
        raise ValueError("repeats must be positive")
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    compute_init("cuda")
    t.set_num_threads(4)
    model, tokenizer, metadata = build_model(cfg.checkpoint_dir, cfg.step, t.device("cuda"), "eval")
    source_paths = (
        "scripts/benchmark/benchmark_chatcore.py",
        "steering/eval/python_success.py",
        "steering/eval/greedy_token_batch.py",
        "steering/eval/triton_qk_rope_norm.py",
        "steering/eval/triton_ragged_attention.py",
        "steering/eval/triton_relu_squared.py",
        "steering/eval/triton_residual_blend.py",
        "steering/eval/sanity_check/run_all.py",
        "tasks/humaneval.py",
        "nanochat/execution.py",
    )
    metrics: dict[str, Any] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
        ).strip() if (_ROOT / ".git").exists() else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "gpu": t.cuda.get_device_name(),
        "gpu_uuid": str(t.cuda.get_device_properties(t.cuda.current_device()).uuid),
        "capability": t.cuda.get_device_capability(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch": t.__version__,
        "triton": triton.__version__,
        "model_config": metadata["model_config"],
        "instrumented_timing": bool(cfg.record_logits),
        "cache_environment": {key: os.environ.get(key) for key in (
            "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NANOCHAT_BASE_DIR",
        )},
        "source_sha256": {
            path: _sha256(_ROOT / path) if (_ROOT / path).exists() else None
            for path in source_paths
        },
        "input_sha256": {
            str(path): _sha256(path) for path in (
                Path(cfg.checkpoint_dir) / f"model_{cfg.step:06d}.pt",
                Path(cfg.checkpoint_dir) / f"meta_{cfg.step:06d}.json",
                Path(get_base_dir()) / "tokenizer/tokenizer.pkl",
            )
        },
        "runs": [],
    }
    original_generate = run_all.generate_greedy_token_batch
    original_grade = humaneval.python_program_passes
    original_batch_grade = getattr(humaneval, "python_programs_pass", None)
    original_decode = greedy._ragged_forward_one
    completions: list[Any] = []
    grades: list[tuple[str, bool]] = []
    grade_batches: list[dict[str, int]] = []
    scalar_grade_calls = 0
    logits: list[dict[str, Any]] = []
    decode_calls = 0

    def capture_generate(
        model: Any,
        tokenizer: Any,
        prompt_tokens: list[list[int]],
        *,
        max_new_tokens: int,
        enable_python_tools: bool = False,
        ragged_cache_initial_len: int | None = None,
    ) -> list[list[int]] | None:
        result = original_generate(
            model, tokenizer, prompt_tokens,
            max_new_tokens=max_new_tokens,
            enable_python_tools=enable_python_tools,
            ragged_cache_initial_len=ragged_cache_initial_len,
        )
        completions.append(result)
        return result

    def capture_grade(
        code: str,
        timeout: float = 5.0,
        maximum_memory_bytes: int | None = 256 * 1024 * 1024,
    ) -> bool:
        nonlocal scalar_grade_calls
        passed = (
            execute_code(code, timeout, maximum_memory_bytes).success
            if cfg.grading == "reference"
            else original_grade(code, timeout, maximum_memory_bytes)
        )
        grades.append((hashlib.sha256(code.encode()).hexdigest(), passed))
        scalar_grade_calls += 1
        return passed

    def capture_decode(
        model: Any, idx: t.Tensor, cache: Any, *, cache_end: int,
    ) -> t.Tensor:
        nonlocal decode_calls
        result = original_decode(model, idx, cache, cache_end=cache_end)
        if decode_calls % 64 == 0:
            logits.append({
                "call": decode_calls,
                "ids": idx.cpu().clone(),
                "logits": result[:4].float().cpu().clone(),
            })
        decode_calls += 1
        return result

    def capture_batch_grade(
        codes: Sequence[str],
        *,
        max_workers: int,
        timeout: float = 5.0,
        maximum_memory_bytes: int | None = 256 * 1024 * 1024,
    ) -> list[bool]:
        programs = list(codes)
        if cfg.grading == "reference":
            passed = [
                execute_code(code, timeout, maximum_memory_bytes).success
                for code in programs
            ]
        else:
            assert original_batch_grade is not None
            passed = original_batch_grade(
                programs, max_workers=max_workers, timeout=timeout,
                maximum_memory_bytes=maximum_memory_bytes,
            )
        grades.extend(
            (hashlib.sha256(code.encode()).hexdigest(), verdict)
            for code, verdict in zip(programs, passed, strict=True)
        )
        grade_batches.append({"programs": len(programs), "max_workers": max_workers})
        return passed

    with (
        ExitStack() as optional_patches,
        patch.object(run_all, "generate_greedy_token_batch", capture_generate),
        patch.object(humaneval, "python_program_passes", capture_grade),
        patch.object(
            greedy, "_ragged_forward_one", capture_decode if cfg.record_logits else original_decode,
        ),
    ):
        if original_batch_grade is not None:
            optional_patches.enter_context(
                patch.object(humaneval, "python_programs_pass", capture_batch_grade),
            )
        for repeat in range(cfg.repeats):
            completions.clear()
            grades.clear()
            grade_batches.clear()
            scalar_grade_calls = 0
            logits.clear()
            decode_calls = 0
            t.cuda.reset_peak_memory_stats()
            t.cuda.synchronize()
            started = time.perf_counter()
            with t.inference_mode():
                scores = run_all.test_chatcore(
                    model, tokenizer, t.autocast("cuda", dtype=t.bfloat16), max_problems=50,
                )
            t.cuda.synchronize()
            entry = {
                "repeat": repeat,
                "seconds": time.perf_counter() - started,
                "peak_allocated": t.cuda.max_memory_allocated(),
                "peak_reserved": t.cuda.max_memory_reserved(),
                "scores": scores,
                "completion_sha256": hashlib.sha256(json.dumps(completions).encode()).hexdigest(),
                "completion_tokens": sum(
                    len(row) for batch in completions if batch is not None for row in batch
                ),
                "grade_sha256": hashlib.sha256(json.dumps(grades).encode()).hexdigest(),
                "graded_programs": len(grades),
                "grade_batches": list(grade_batches),
                "scalar_graded_programs": scalar_grade_calls,
                "max_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "max_single_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            }
            metrics["runs"].append(entry)
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            (output / f"completions_{repeat}.json").write_text(json.dumps(completions))
            (output / f"grades_{repeat}.json").write_text(json.dumps(grades))
            if cfg.record_logits:
                t.save(logits, output / f"logits_{repeat}.pt")
            logger.info("ChatCORE run: {}", entry)

        if cfg.profile:
            run_all.generate_greedy_token_batch = original_generate
            greedy._ragged_forward_one = original_decode
            cpu = cProfile.Profile()
            with t.inference_mode(), t.autocast("cuda", dtype=t.bfloat16):
                with t.profiler.profile(activities=[
                    t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA,
                ]) as profile:
                    cpu.enable()
                    scores = run_all._run_chatcore_generative_tasks(
                        model, tokenizer, max_problems=50, max_new_tokens=64,
                    )
                    t.cuda.synchronize()
                    cpu.disable()
            profile.export_chrome_trace(str(output / "trace.json"))
            (output / "operators.txt").write_text(
                profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=65),
            )
            with (output / "host_profile.txt").open("w") as stream:
                pstats.Stats(cpu, stream=stream).sort_stats("cumulative").print_stats(90)
            (output / "profile_scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
