"""Dispatcher for progressive-eval profiling runs."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile

from scripts.profile_progression_microbench import get as get_microbench
from scripts.train.training_progression import (
    CheckpointSpec,
    conditioned_configs_for_phase,
    evaluate_single_checkpoint,
)
from steering.profiling.markers import TimingRecorder, use_recorder

_CONFIG_DIR = str(
    Path(__file__).resolve().parents[2] / "configs/eval/profile_progression"
)


@dataclass(frozen=True)
class RunDirs:
    run_root: Path
    run_dir: Path


def _iso_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _profile_path(cfg: DictConfig) -> Literal["A", "C"]:
    model_family = str(cfg.progression.model_family)
    if model_family == "decoder":
        return "A"
    if model_family == "encoder_decoder":
        return "C"
    raise ValueError(
        "progression.model_family must be decoder or encoder_decoder for profiling; "
        f"got {model_family!r}"
    )


def _resolve_run_dir(cfg: DictConfig) -> RunDirs:
    out_root = Path(cfg.out_root)
    run_root = out_root / "runs"
    run_dir = Path(cfg.run_dir) if cfg.get("run_dir") else run_root / (
        f"{_iso_timestamp()}_{_profile_path(cfg)}_{cfg.level}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "summary").mkdir(parents=True, exist_ok=True)
    return RunDirs(run_root=run_root, run_dir=run_dir)


def _write_stamps(cfg: DictConfig, run_dir: Path) -> None:
    (run_dir / "config.resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover
        sha = f"unknown ({exc})"
    (run_dir / "git_sha.txt").write_text(sha + "\n")
    env_lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"torch={t.__version__}",
        f"cuda_available={t.cuda.is_available()}",
    ]
    if t.cuda.is_available():
        env_lines.append(f"cuda_device={t.cuda.get_device_name(0)}")
        env_lines.append(f"cuda_driver={t.version.cuda}")
    (run_dir / "env.txt").write_text("\n".join(env_lines) + "\n")


def _gpu_preflight(cfg: DictConfig, run_dir: Path) -> None:
    if cfg.get("preflight_skip", False) or not t.cuda.is_available():
        (run_dir / "gpu_preflight.txt").write_text("skipped\n")
        return
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        (run_dir / "gpu_preflight.txt").write_text(f"unavailable: {exc}\n")
        return

    (run_dir / "gpu_preflight.txt").write_text(out + "\n")
    mem_used, _mem_free, util = [int(part.strip()) for part in out.splitlines()[0].split(",")]
    if (
        mem_used > int(cfg.preflight_max_external_mem_mb)
        or util > int(cfg.preflight_max_external_util_pct)
    ):
        raise RuntimeError(
            "GPU preflight failed: external process(es) detected. "
            f"mem_used={mem_used} MB, util={util}%."
        )


def _build_progression_dictconfig(cfg: DictConfig) -> DictConfig:
    raw = OmegaConf.to_container(cfg.progression, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("progression config must resolve to a mapping")
    return OmegaConf.create(raw)


def _make_spec_from_cfg(cfg: DictConfig) -> CheckpointSpec:
    return CheckpointSpec(
        step=int(cfg.progression.step),
        phase=str(cfg.progression.phase),
        subdir=str(cfg.progression.subdir),
    )


def _validate_path_c_conditioned_configs(progression_cfg: DictConfig) -> None:
    if str(progression_cfg.model_family) != "encoder_decoder":
        return
    conditioned_configs = conditioned_configs_for_phase(
        progression_cfg,
        str(progression_cfg.phase),
    )
    if conditioned_configs:
        return
    raise ValueError(
        "path=C profiling requires at least one non-empty "
        "progression.conditioning.conditioned_configs entry."
    )


def _resolve_device() -> t.device:
    return t.device("cuda:0" if t.cuda.is_available() else "cpu")


_SUITE_STAGE_MAP: dict[str, tuple[str, str]] = {
    "evaluate_single_checkpoint": ("", "run_total"),
    "load_progression_model": ("", "checkpoint_load"),
    "validate_context_coverage": ("", "context_coverage"),
    "build_eval_wrapper": ("", "build_wrapper"),
    "run_core_eval": ("core", "suite_total"),
    "run_chat_eval_suite": ("chat", "suite_total"),
    "run_harmfulness_eval": ("harmfulness", "suite_total"),
    "encdec_eval_wrapper_forward": ("", "encdec_eval_wrapper_forward"),
    "encdec_build_conditioning": ("", "encdec_build_conditioning"),
}
_COARSE_MARKERS: frozenset[str] = frozenset(_SUITE_STAGE_MAP)


def _write_macro_csv(recorder: TimingRecorder, csv_path: Path, *, path: str) -> None:
    """Write macro rows with append-only accounting columns."""
    fieldnames = [
        "path",
        "suite",
        "stage",
        "iter",
        "warmup",
        "wall_seconds",
        "variant",
        "conditioned_config",
        "seed",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in recorder.rows():
            name = row["name"]
            if name not in _SUITE_STAGE_MAP:
                continue
            default_suite, stage = _SUITE_STAGE_MAP[name]
            writer.writerow(
                {
                    "path": row.get("path", path),
                    "suite": row.get("suite", default_suite),
                    "stage": stage,
                    "iter": row.get("iter", ""),
                    "warmup": str(row.get("warmup", "")).lower(),
                    "wall_seconds": f"{float(row['elapsed_ns']) / 1e9:.6f}",
                    "variant": row.get("eval_variant", row.get("variant", "")),
                    "conditioned_config": row.get("conditioned_config", ""),
                    "seed": row.get("seed", ""),
                }
            )


def _tag_recent_rows(
    recorder: TimingRecorder,
    last_index: int,
    **tags: object,
) -> int:
    return recorder.tag_rows_since(last_index, tags)


def run_macro(cfg: DictConfig, run_dir: Path) -> None:
    progression_cfg = _build_progression_dictconfig(cfg)
    _validate_path_c_conditioned_configs(progression_cfg)
    path = _profile_path(cfg)
    spec = _make_spec_from_cfg(cfg)
    device = _resolve_device()
    ptdtype = t.float32 if progression_cfg.dtype == "float32" else t.bfloat16
    eval_suites = [s.strip() for s in str(progression_cfg.evals).split(",") if s.strip()]
    seeds = [
        int(progression_cfg.base_seed) + i * 1000
        for i in range(int(progression_cfg.num_seeds))
    ]

    recorder = TimingRecorder()
    total_iters = int(cfg.warmup_iterations) + int(cfg.num_iterations)
    for iter_idx in range(total_iters):
        warmup = iter_idx < int(cfg.warmup_iterations)
        logger.info(f"[macro] iter {iter_idx} warmup={warmup} path={path}")
        last_index = len(recorder)
        with use_recorder(recorder):
            evaluate_single_checkpoint(
                spec=spec,
                device=device,
                ptdtype=ptdtype,
                eval_suites=eval_suites,
                seeds=seeds,
                cfg=progression_cfg,
                existing_entry=None,
            )
        _tag_recent_rows(recorder, last_index, iter=iter_idx, warmup=warmup, path=path)
        if t.cuda.is_available():
            t.cuda.synchronize()
            t.cuda.empty_cache()

    csv_path = run_dir / "macro_wall_times.csv"
    _write_macro_csv(recorder, csv_path, path=path)
    logger.info(f"[macro] wrote {csv_path}")


def _apply_suite_scale_reduction(cfg: DictConfig) -> DictConfig:
    progression_cfg = _build_progression_dictconfig(cfg)
    progression_cfg.max_per_task = 5
    progression_cfg.chat_max_problems = 5
    progression_cfg.harm_num_harmful = 3
    progression_cfg.harm_num_benign = 3
    return progression_cfg


def _write_record_function_summary(prof, csv_path: Path) -> None:
    totals: dict[str, dict[str, int]] = {}
    for evt in prof.events():
        if evt.name not in _COARSE_MARKERS:
            continue
        row = totals.setdefault(
            evt.name,
            {"self_cpu_time_us": 0, "device_time_us": 0, "count": 0},
        )
        row["self_cpu_time_us"] += int(evt.self_cpu_time_total or 0)
        row["device_time_us"] += int(evt.device_time_total or 0)
        row["count"] += int(evt.count)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "self_cpu_time_us", "device_time_us", "count"],
        )
        writer.writeheader()
        for name in sorted(totals):
            writer.writerow({"name": name, **totals[name]})


def run_suite(cfg: DictConfig, run_dir: Path) -> None:
    progression_cfg = _apply_suite_scale_reduction(cfg)
    _validate_path_c_conditioned_configs(progression_cfg)
    device = _resolve_device()
    ptdtype = t.float32 if progression_cfg.dtype == "float32" else t.bfloat16
    eval_suites = [s.strip() for s in str(progression_cfg.evals).split(",") if s.strip()]
    seeds = [
        int(progression_cfg.base_seed) + i * 1000
        for i in range(int(progression_cfg.num_seeds))
    ]
    spec = _make_spec_from_cfg(cfg)

    evaluate_single_checkpoint(
        spec=spec,
        device=device,
        ptdtype=ptdtype,
        eval_suites=eval_suites,
        seeds=seeds,
        cfg=progression_cfg,
        existing_entry=None,
    )
    if t.cuda.is_available():
        t.cuda.synchronize()
        t.cuda.empty_cache()

    activities = [ProfilerActivity.CPU]
    if t.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        evaluate_single_checkpoint(
            spec=spec,
            device=device,
            ptdtype=ptdtype,
            eval_suites=eval_suites,
            seeds=seeds,
            cfg=progression_cfg,
            existing_entry=None,
        )

    trace_path = run_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    ka = prof.key_averages(group_by_input_shape=False)
    (run_dir / "key_averages.txt").write_text(
        ka.table(sort_by="self_cpu_time_total", row_limit=30)
    )
    _write_record_function_summary(prof, run_dir / "record_function_summary.csv")


def run_hotspot(cfg: DictConfig, run_dir: Path) -> None:
    name = cfg.get("hotspot_name")
    if not name:
        raise ValueError("level=hotspot requires hotspot_name")
    fixture_path = Path(
        cfg.get("hotspot_fixture") or f"tests/fixtures/profile_progression/{name}.pt"
    )
    device_str = "cuda:0" if t.cuda.is_available() else "cpu"
    result = get_microbench(str(name))(
        fixture_path,
        device_str,
        int(cfg.get("hotspot_n_iters", 10)),
    )
    rows = result if isinstance(result, list) else [result]

    jsonl_path = run_dir / "microbench_results.jsonl"
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    summary_path = Path(cfg.out_root) / "summary" / "baselines.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _dispatch(cfg: DictConfig, run_dir: Path) -> None:
    if cfg.level == "macro":
        run_macro(cfg, run_dir)
    elif cfg.level == "suite":
        run_suite(cfg, run_dir)
    elif cfg.level == "hotspot":
        run_hotspot(cfg, run_dir)
    else:
        raise ValueError(f"Unknown level={cfg.level!r}; expected macro|suite|hotspot")


@hydra.main(
    config_path=_CONFIG_DIR,
    config_name="profile_progression",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    dirs = _resolve_run_dir(cfg)
    logger.info(
        f"profile_progression: path={_profile_path(cfg)} run_dir={dirs.run_dir}"
    )
    _write_stamps(cfg, dirs.run_dir)
    _gpu_preflight(cfg, dirs.run_dir)
    _dispatch(cfg, dirs.run_dir)


if __name__ == "__main__":
    main()
