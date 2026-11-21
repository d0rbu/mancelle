"""Measure project GCG through normal preparation, without generation or judging."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import hydra
import numpy as np
import pandas as pd
import torch as t
import torch._functorch.config as functorch_config
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from scripts.benchmark.gcg_paired import measure_paired
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_project_gcg_activation_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.gcg_driver import GcgRuntime
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample
from steering.eval.axbench_adv.self_conditioning import SelfConditioningMethodRuntime
from steering.eval.axbench_upstream.data import select_training_texts
from steering.eval.axbench_upstream.methods import load_method_setup
from tests.axbench_adv.gcg_state_reference import (
    preserve_gcg_model_state as reference_model_state,
)


class _NoJudge:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def evaluate(self, **_kwargs: Any) -> Any:
        raise AssertionError("the GCG benchmark must not generate or judge completions")

    def close(self) -> None:
        pass


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _measure(operation: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, Any]]:
    samples = []
    result = None
    t.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        t.cuda.synchronize()
        start = time.perf_counter()
        result = operation()
        t.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return result, {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_allocated": t.cuda.max_memory_allocated(),
    }


@contextmanager
def _reference(model: t.nn.Module, enabled: bool, *, kind: str = "logits") -> Iterator[None]:
    if not enabled:
        yield
        return
    if kind == "state":
        with patch.object(self_conditioning, "preserve_gcg_model_state", reference_model_state):
            yield
        return
    original = model.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("logits_to_keep", None)
        return original(*args, **kwargs)

    with patch.object(model, "forward", forward):
        yield


def _numerics(value: Any) -> t.Tensor:
    return t.stack(value) if isinstance(value[0], t.Tensor) else t.tensor(value, dtype=t.float64)


def _measure_paired(
    operation: Callable[[], Any],
    repeats: int,
    *,
    model: t.nn.Module,
    warmups: int,
    reference_kind: str,
) -> tuple[Any, dict[str, Any]]:
    return measure_paired(
        operation,
        repeats,
        reference=partial(_reference, model, True, kind=reference_kind),
        numerics=_numerics,
        warmups=warmups,
    )


def _profile(operation: Callable[[], Any], path: Path) -> dict[str, Any]:
    with t.profiler.profile(
        activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        operation()
        t.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    digest = _sha256(path)
    compressed = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)
    with gzip.open(compressed, "rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
            raise RuntimeError("compressed profiler trace differs from its source")
    path.unlink()
    events = profiler.key_averages()
    path.with_suffix(".table.txt").write_text(
        events.table(sort_by="self_cuda_time_total", row_limit=40)
    )
    return {"trace": compressed.name, "uncompressed_sha256": digest}


@hydra.main(
    version_base=None,
    config_path="../../configs/eval",
    config_name="benchmark_axbench_adv_gcg_project",
)
def main(args: DictConfig) -> None:
    if not t.cuda.is_available():
        raise ValueError("this benchmark requires CUDA")
    if args.method not in {"SelfConditioning", "SelfConditioningMasked"}:
        raise ValueError(
            "project benchmark method must be SelfConditioning or SelfConditioningMasked"
        )
    for name in ("members", "batch_size", "warmups", "repeats"):
        if type(args[name]) is not int or args[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.paired and args.full_logits:
        raise ValueError("paired=true requires full_logits=false")
    if args.paired_reference not in {"logits", "state"}:
        raise ValueError("paired_reference must be logits or state")
    if args.paired_reference == "state" and not args.paired:
        raise ValueError("paired_reference=state requires paired=true")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    t.use_deterministic_algorithms(args.deterministic)
    t.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    source = data["source"]
    if args.members > len(data["rows"]):
        raise ValueError("members exceeds the prepared training population")
    meta_path = Path(source["target_meta_path"])
    meta = json.loads(meta_path.read_text())
    checkpoint_path = meta_path.with_name(meta_path.name.replace("meta_", "model_")).with_suffix(
        ".pt"
    )
    checkpoint_sha = _sha256(checkpoint_path)
    setup = load_method_setup(meta_path, Path(source["feature_statistics_path"]))
    fit = preflight_project_direction_fit(
        Path(source["project_direction_root"]),
        ProjectCounterExpectations(
            target_model_sha256=checkpoint_sha,
            target_metadata_sha256=_sha256(meta_path),
            concept_texts={0: data["concept"]},
            decoder_width=meta["model_config"]["n_embd"],
            layer=source["project_layer"],
            fit_seed=42,
        ),
    )
    direction = load_project_gcg_activation_direction(fit, concept_id=0)
    frame = pd.read_parquet(Path(source["main_dump_dir"]) / "inference/steering_data.parquet")
    frame = frame.loc[frame["concept_id"].eq(0)].copy()
    examples = [
        OptimizationExample(
            input_id=index,
            user_message=str(frame.loc[frame["input_id"].eq(index), "original_prompt"].iloc[0]),
        )
        for index in range(10)
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report: dict[str, Any] = {
        "git_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": platform.python_version(),
        "torch": str(t.__version__),
        "cuda": t.version.cuda,
        "gpu": t.cuda.get_device_name(),
        "input_sha256": _sha256(input_path),
        "checkpoint_sha256": checkpoint_sha,
        "metadata_sha256": _sha256(meta_path),
        "source_sha256": {
            name: _sha256(Path(name))
            for name in (
                "steering/conditional/encoder_decoder/model.py",
                "steering/eval/axbench_adv/self_conditioning.py",
                "scripts/benchmark/benchmark_axbench_adv_gcg_project.py",
                "scripts/benchmark/gcg_paired.py",
                "scripts/eval/axbench_adv/gcg_runtime_state.py",
                "tests/axbench_adv/gcg_state_reference.py",
            )
        },
        "config": OmegaConf.to_container(args, resolve=True),
        "measurements": [],
    }
    tensors: dict[str, t.Tensor] = {}
    with (
        functorch_config.patch(donated_buffer=args.donated_buffer),
        SelfConditioningMethodRuntime(
            setup=setup,
            training=select_training_texts(Path(source["prepared_dir"]), max_concepts=1),
            main_frame=frame,
            max_concepts=1,
            max_new_tokens=8,
            master_seed=42,
            device="cuda:0",
            extraction_batch_size=4,
            judge_factory=cast(Any, _NoJudge),
            judge_cache=output / "unused.sqlite3",
            judge_model="benchmark/no-provider",
            api_base_url="http://127.0.0.1:1",
            cache_only=True,
            timeout_seconds=120,
            stderr_dir=output / "unused-judge-logs",
            gcg_activation_directions={0: direction},
        ) as runtime,
    ):
        with (
            runtime.bind_cell(
                method=args.method,
                concept_id=0,
                concept=data["concept"],
                factor=args.factor,
                examples=examples,
            ) as cell,
            _reference(runtime._target_model, args.full_logits),
        ):
            gcg_cell = cast(GcgRuntime, cell)
            measure = (
                partial(
                    _measure_paired,
                    model=runtime._target_model,
                    warmups=args.warmups,
                    reference_kind=args.paired_reference,
                )
                if args.paired
                else _measure
            )
            for objective in args.objectives:
                for k in args.token_counts:
                    case = f"{args.method}.{objective}.k{k}"
                    prepared, prepare_time = _measure(
                        partial(
                            gcg_cell.prepare_gcg,
                            objective=objective,
                            alpha=0.01,
                            rows=data["rows"],
                            suffix_token_count=k,
                            device_batch_size=args.batch_size,
                            separator="\n\n",
                        ),
                        1,
                    )
                    tokens = prepared.seed_token_ids
                    alternative = tuple(runtime._target_tokenizer.encode(" ".join(["?"] * k)))
                    candidates = [tokens, alternative]
                    if not all(item.valid for item in gcg_cell.gcg_filter(candidates)):
                        raise ValueError("benchmark candidates failed round-trip filtering")
                    members = list(range(args.members))
                    gradient_operation = partial(gcg_cell.gcg_gradient, tokens, members)
                    score_operation = partial(
                        gcg_cell.gcg_score,
                        candidates,
                        members,
                        candidate_eval_device_batch_size=args.batch_size,
                    )
                    for _ in range(args.warmups):
                        gradient_operation()
                        score_operation()
                    gradients, gradient_time = measure(gradient_operation, args.repeats)
                    scores, score_time = measure(score_operation, args.repeats)
                    tensors[case + ".gradient"] = t.stack(gradients).contiguous()
                    tensors[case + ".scores"] = t.tensor(scores, dtype=t.float32)
                    record = {
                        "case": case,
                        "prepared_pairs": prepared.member_count,
                        "tokens": tokens,
                        "candidates": candidates,
                        "prepare": prepare_time,
                        "gradient": gradient_time,
                        "score": score_time,
                    }
                    if args.profile and k == args.token_counts[0]:
                        for label, operation in (
                            ("gradient", gradient_operation),
                            ("score", score_operation),
                        ):
                            record[label + "_profile"] = _profile(
                                operation, output / f"{case}.{label}.trace.json"
                            )
                    report["measurements"].append(record)
                    (output / "report.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False)
                    )
                    save_file(tensors, output / "numerics.safetensors")
                    logger.info(json.dumps(record))


if __name__ == "__main__":
    main()
