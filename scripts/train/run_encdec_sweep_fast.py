"""Fast encoder-decoder sweep runner that amortizes startup costs across configs.

Instead of launching a separate torchrun process per config (as run_encdec_sweep.py does),
this script builds shared state once and iterates over configs in-process — eliminating
repeated model/tokenizer/feature-index loading overhead.

Multi-GPU: use sweep_rank/sweep_world_size to stripe configs across
independent processes (one per GPU). A launcher script is provided for
convenience. Results are written to per-rank JSONL files, then merged.

Usage:
    # Dry-run: list all configs
    uv run python -m scripts.train.run_encdec_sweep_fast mode=dry_run

    # Run all on a single GPU
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all

    # Multi-GPU (2 GPUs):
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all gpu=0 sweep_rank=0 sweep_world_size=2 &
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all gpu=1 sweep_rank=1 sweep_world_size=2 &
    wait

    # Limit to first 20 configs
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all min_idx=0 max_idx=20

    # Resume: skip configs already in the results file
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all resume=true

    # Shuffle execution order (reproducible via shuffle_seed)
    uv run python -m scripts.train.run_encdec_sweep_fast mode=all shuffle=true shuffle_seed=99
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import hydra
import torch as t
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from nanochat.common import DummyWandb, compute_cleanup, print_banner
from nanochat.config_validation import validate_sweep_mode
from scripts.train.conditional_encdec_train import (
    SharedState,
    TrainResult,
    _resolve_splice_counterfactual_data_dir,
    _resolved_user_config,
    build_shared_state,
    train_one_config,
)
from scripts.train.run_encdec_sweep import SweepConfig, build_sweep_configs
from steering.conditional.feature_clusters import reject_feature_clusters_config

# ---------------------------------------------------------------------------
# Hydra helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")


def _compose_config(config_dir: str, config_name: str, overrides: list[str] | None = None):
    """Compose a Hydra config, clearing global state if needed."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


# ---------------------------------------------------------------------------
# Per-config execution
# ---------------------------------------------------------------------------


def _run_one(
    sweep_cfg: SweepConfig,
    index: int,
    total: int,
    shared: SharedState,
    config_name: str = "conditional_encdec_sweep",
    training_overrides: list[str] | None = None,
) -> TrainResult:
    """Run a single sweep config against pre-built shared state."""
    master_process = shared.ddp_rank == 0

    t0 = time.time()
    logger.info(f"[{index + 1}/{total}] Starting config: {sweep_cfg.model_tag}")
    logger.info(f"  Feature selection: {sweep_cfg.fs.distribution}(mean={sweep_cfg.fs.mean}, std={sweep_cfg.fs.std})")
    logger.info(f"  Negative sampling: geometric(mean={sweep_cfg.ns.mean})")
    logger.info(f"  Feature delta: {sweep_cfg.fd.short_name}")
    logger.info(f"  FP weight: {sweep_cfg.fp_weight}")

    # Filter out config_name= from overrides (we already compose with that name)
    overrides = [o for o in sweep_cfg.hydra_overrides(config_name=config_name) if not o.startswith("config_name=")]
    if training_overrides:
        overrides.extend(training_overrides)

    wandb_run: object = DummyWandb()  # default so finally block always works
    result: TrainResult
    try:
        per_cfg = _compose_config(
            config_dir=_CONFIG_DIR,
            config_name=config_name,
            overrides=overrides,
        )
        reject_feature_clusters_config(
            per_cfg,
            "run_encdec_sweep_fast per-config",
        )
        _resolve_splice_counterfactual_data_dir(per_cfg, shared.feature_dir)
        user_config = _resolved_user_config(per_cfg)

        if master_process:
            wandb_run = wandb.init(
                project="nanochat",
                name=sweep_cfg.model_tag,
                config=user_config,
                reinit=True,
            )

        result = train_one_config(per_cfg, shared, wandb_run)
    except Exception as exc:
        logger.error(f"Config {sweep_cfg.model_tag} failed: {exc}")
        result = TrainResult(
            model_tag=sweep_cfg.model_tag,
            val_bpb=None,
            min_val_bpb=float("inf"),
            total_training_time=time.time() - t0,
            peak_memory_mib=0.0,
            startup_time=0.0,
            error=str(exc),
        )
    finally:
        wandb_run.finish()
        # Free ephemeral per-config tensors / optimizer state
        gc.enable()
        if hasattr(gc, "unfreeze"):
            gc.unfreeze()
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()

    elapsed = time.time() - t0
    val_str = f"{result.val_bpb:.6f}" if result.val_bpb is not None else "N/A"
    status = "FAILED" if result.error else "OK"
    logger.info(f"  [{status}] val_bpb={val_str}  elapsed={elapsed:.1f}s")
    return result


# ---------------------------------------------------------------------------
# JSONL serialization helpers
# ---------------------------------------------------------------------------


def _serialize_result(
    result: TrainResult,
    sweep_cfg: SweepConfig,
    index: int,
) -> dict:
    """Convert a TrainResult + SweepConfig into a JSON-serializable dict.

    Replaces float('inf') and float('nan') with None for JSON compat.
    """

    def _sanitize(v: float | None) -> float | None:
        if v is None:
            return None
        if not math.isfinite(v):
            return None
        return v

    config_label = (
        f"{sweep_cfg.fs.short_name}/"
        f"ns_{sweep_cfg.ns.short_name}/"
        f"fd_{sweep_cfg.fd.short_name}/"
        f"fpw_{sweep_cfg.fp_weight:g}"
    )

    return {
        "model_tag": result.model_tag,
        "config_label": config_label,
        "val_bpb": _sanitize(result.val_bpb),
        "min_val_bpb": _sanitize(result.min_val_bpb),
        "total_training_time": _sanitize(round(result.total_training_time, 2)),
        "peak_memory_mib": _sanitize(round(result.peak_memory_mib, 1)),
        "startup_time": _sanitize(round(result.startup_time, 2)),
        "error": result.error,
        "fs_distribution": sweep_cfg.fs.distribution,
        "fs_mean": sweep_cfg.fs.mean,
        "fs_std": sweep_cfg.fs.std,
        "ns_mean": sweep_cfg.ns.mean,
        "fd_mode": sweep_cfg.fd.mode.value,
        "fd_neg_delta_ratio": sweep_cfg.fd.neg_delta_ratio,
        "fp_weight": sweep_cfg.fp_weight,
        "config_index": index,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _append_result(
    path: Path,
    result: TrainResult,
    sweep_cfg: SweepConfig,
    index: int,
) -> None:
    """Append one JSONL line for a completed config."""
    record = _serialize_result(result, sweep_cfg, index)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_csv_summary(jsonl_path: Path, csv_path: Path) -> None:
    """Convert a JSONL results file to CSV for spreadsheet analysis."""
    if not jsonl_path.exists():
        return

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))

    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_summary(results: list[TrainResult]) -> None:
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"  Total:      {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")

    if successful:
        best = min(successful, key=lambda r: r.val_bpb if r.val_bpb is not None else float("inf"))
        best_val = f"{best.val_bpb:.6f}" if best.val_bpb is not None else "N/A"
        print(f"  Best val_bpb: {best_val}  ({best.model_tag})")

    if successful:
        print("\nResults (sorted by val_bpb):")
        sorted_ok = sorted(successful, key=lambda r: r.val_bpb if r.val_bpb is not None else float("inf"))
        for r in sorted_ok:
            val_str = f"{r.val_bpb:.6f}" if r.val_bpb is not None else "N/A"
            print(f"  {val_str}  {r.model_tag}  (t={r.total_training_time:.1f}s)")

    if failed:
        print("\nFailed configs:")
        for r in failed:
            print(f"  {r.model_tag}  error={r.error}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def _load_completed_tags(output_dir: Path) -> set[str]:
    """Read all JSONL result files in output_dir and return completed model_tags.

    Looks at both sweep_results.jsonl and per-rank files (sweep_results_rank*.jsonl).
    Only non-errored configs count as completed.
    """
    completed: set[str] = set()
    for jsonl_file in output_dir.glob("sweep_results*.jsonl"):
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    record = json.loads(stripped_line)
                    if record.get("error") is None and record.get("model_tag"):
                        completed.add(record["model_tag"])
        except (OSError, json.JSONDecodeError):
            continue
    return completed


def _merge_rank_files(output_dir: Path) -> None:
    """Merge per-rank JSONL files into a single sweep_results.jsonl."""
    rank_files = sorted(output_dir.glob("sweep_results_rank*.jsonl"))
    if not rank_files:
        return
    merged_path = output_dir / "sweep_results.jsonl"
    all_records: list[dict] = []
    for rf in rank_files:
        with open(rf, encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    all_records.append(json.loads(stripped_line))
    # Sort by config_index for deterministic output
    all_records.sort(key=lambda r: r.get("config_index", 0))
    with open(merged_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Merged {len(all_records)} results from {len(rank_files)} rank files into {merged_path}")


@hydra.main(
    config_path="../../configs/training",
    config_name="run_encdec_sweep_fast",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    validate_sweep_mode(cfg)

    all_configs = build_sweep_configs()
    total_all = len(all_configs)
    print_banner()
    print(f"Total sweep configurations: {total_all}")

    # --- Resolve the subset to run ---
    mode = cfg.mode

    if mode == "dry_run":
        for i, c in enumerate(all_configs):
            print(f"[{i}] {c.model_tag}  overrides={c.hydra_overrides()}")
        return

    if isinstance(mode, int):
        if mode < 0 or mode >= total_all:
            print(
                f"Error: index {mode} out of range [0, {total_all})",
                file=sys.stderr,
            )
            sys.exit(1)
        selected = [(mode, all_configs[mode])]

    elif isinstance(mode, (list, ListConfig)):
        start, end = mode
        start = max(0, start)
        end = min(end, total_all)
        selected = list(enumerate(all_configs))[start:end]

    else:  # mode is None — run all
        selected = list(enumerate(all_configs))

    # --- Apply min/max index filtering ---
    if cfg.min_idx is not None:
        selected = [(i, c) for i, c in selected if i >= cfg.min_idx]
    if cfg.max_idx is not None:
        selected = [(i, c) for i, c in selected if i < cfg.max_idx]

    if cfg.shuffle:
        rng = random.Random(cfg.shuffle_seed)
        rng.shuffle(selected)
        logger.info(f"Shuffled {len(selected)} configs (seed={cfg.shuffle_seed})")

    # --- Multi-GPU: config parallelism via multigpu_config ---
    if cfg.multigpu_config is not None:
        rank = cfg.multigpu_config.sweep_rank
        world_size = cfg.multigpu_config.sweep_world_size
        gpu_id = cfg.multigpu_config.gpu if cfg.multigpu_config.gpu is not None else rank
    else:
        rank = 0
        world_size = 1
        gpu_id = 0
    master_process = rank == 0

    if world_size > 1:
        logger.info(f"Sweep rank {rank}/{world_size}: using GPU {gpu_id}")
        selected = [(i, c) for idx, (i, c) in enumerate(selected) if idx % world_size == rank]
        logger.info(f"Sweep rank {rank}/{world_size}: assigned {len(selected)} configs (striped)")

    # --- Signal handling: graceful interruption ---
    interrupted = False

    def _handle_signal(signum, _frame):  # noqa: ANN001
        nonlocal interrupted
        if interrupted:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise SystemExit(130)
        logger.warning(f"Received signal {signum}, will stop after current config. (Press Ctrl+C again to force-quit.)")
        interrupted = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- One-time shared state ---
    training_overrides = list(cfg.training_overrides) if cfg.training_overrides else []
    if training_overrides:
        logger.info(f"Training overrides: {training_overrides}")
    base_cfg = _compose_config(
        config_dir=_CONFIG_DIR,
        config_name=cfg.config_name,
        overrides=training_overrides,
    )
    reject_feature_clusters_config(base_cfg, "run_encdec_sweep_fast shared setup")
    shared = build_shared_state(base_cfg, gpu_index=gpu_id if world_size > 1 else None)

    # --- Resolve output directory ---
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(OmegaConf.select(base_cfg, "output_dir", default="out"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-rank JSONL to avoid write contention
    if world_size > 1:
        jsonl_path = output_dir / f"sweep_results_rank{rank}.jsonl"
    else:
        jsonl_path = output_dir / "sweep_results.jsonl"

    # --- Resume: skip completed configs ---
    if cfg.resume:
        completed_tags = _load_completed_tags(output_dir)
        before = len(selected)
        selected = [(i, c) for i, c in selected if c.model_tag not in completed_tags]
        skipped = before - len(selected)
        if skipped > 0:
            logger.info(f"Resume: skipping {skipped} already-completed configs ({len(selected)} remaining)")

    total = len(selected)
    if total == 0:
        logger.info("No configs to run (all completed or filtered out).")
        if master_process:
            _merge_rank_files(output_dir)
        compute_cleanup()
        return

    logger.info(f"Will run {total} config(s)")
    logger.info(f"Sweep results will be written to: {jsonl_path}")

    # --- Per-config loop ---
    results: list[TrainResult] = []
    sweep_t0 = time.time()

    for loop_i, (original_index, sweep_cfg) in enumerate(selected):
        if interrupted:
            logger.warning("Interrupted — stopping sweep early.")
            break

        result = _run_one(sweep_cfg, loop_i, total, shared, config_name=cfg.config_name, training_overrides=training_overrides)
        results.append(result)
        _append_result(jsonl_path, result, sweep_cfg, original_index)

    sweep_elapsed = time.time() - sweep_t0

    # --- Final cleanup ---
    compute_cleanup()

    # Each rank writes a done marker; rank 0 waits then merges
    if world_size > 1:
        done_marker = output_dir / f".done_rank{rank}"
        done_marker.touch()
        if master_process:
            # Wait for all ranks to finish (simple file-based sync)
            for r in range(world_size):
                marker = output_dir / f".done_rank{r}"
                for _ in range(600):  # 10 min max wait
                    if marker.exists():
                        break
                    time.sleep(1)
            _merge_rank_files(output_dir)
            # Clean up markers
            for r in range(world_size):
                (output_dir / f".done_rank{r}").unlink(missing_ok=True)

    if master_process:
        all_results_path = output_dir / "sweep_results.jsonl"
        if all_results_path.exists():
            csv_path = all_results_path.with_suffix(".csv")
            _write_csv_summary(all_results_path, csv_path)
            logger.info(f"Results saved: {all_results_path} and {csv_path}")

    _print_summary(results)
    print(f"\nTotal sweep wall time: {sweep_elapsed:.1f}s (rank {rank})")

    if interrupted:
        print("Sweep was interrupted before completion.", file=sys.stderr)
        sys.exit(130)

    failed_count = sum(1 for r in results if r.error is not None)
    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
