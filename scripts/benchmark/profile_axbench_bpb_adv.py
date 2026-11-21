"""Profile the AxBench-BPB-Adv evaluation pipeline.

Mirror of :mod:`scripts.benchmark.profile_axbench_base` for the BPB-Adv
pipeline. Composes the same Hydra config tree as
:mod:`scripts.eval.axbench_bpb_adv_eval`, activates
:class:`steering.eval.axbench_base.profiler.AxBenchProfiler`, and dispatches
the normal stage runner. Outputs are written under::

    <output_root>/<run_tag>/profile/events.jsonl
    <output_root>/<run_tag>/profile/summary.yaml
    <output_root>/<run_tag>/profile/summary.csv

Usage::

    uv run python -m scripts.benchmark.profile_axbench_bpb_adv \\
        source_run_dir=out/eval/axbench_base/<run_tag> \\
        output_root=out/eval/axbench_bpb_adv \\
        feature_dir=... feature_statistics_path=... \\
        stage=validate_source

The production eval script (``scripts.eval.axbench_bpb_adv_eval``) carries
the ``@_profiled`` decorators + ``record_counter`` calls; they are no-ops
unless this wrapper binds a profiler around the run. The production path is
therefore behaviourally unchanged (no syscalls, no allocations, no per-row
work) when invoked directly — the residual cost is a handful of constant-time
``get_current_profiler()`` lookups per stage.

**Process model:** this wrapper is **single-process by design** — it does
NOT call ``compute_init`` and therefore does NOT initialize a
``torch.distributed`` process group. Launching it under ``torchrun`` *without*
externally initializing dist will leave every rank's
``dist.is_initialized()`` False, and every rank would race on the shared
``events.jsonl`` — DON'T. If dist *is* externally initialized, the wrapper's
rank guard correctly binds the profiler on rank 0 only and lets non-zero
ranks run the dispatcher without a profiler (the in-stage ``profile_region``
calls then no-op on those ranks).

Because ``compute_init`` is skipped, this wrapper also does NOT set
``torch.set_float32_matmul_precision('high')`` (tf32) or the global seed that
the production CLI applies. CUDA wall-clock timings captured here are thus a
lower-floor approximation, not strictly production-representative, unless the
caller configures matmul precision first. (The sibling
``profile_axbench_base`` wrapper has the same property.)
"""

from __future__ import annotations

import dataclasses

import hydra
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict


def _run(cfg: DictConfig) -> None:
    """Testable profiling entry point.

    Builds the dataclass config the same way the production eval does, then
    resolves the profile output directory under the (synchronized) run tag and
    activates an :class:`AxBenchProfiler` around the production dispatcher.
    """
    cfg_obj = cfg_to_dataclass(cfg)
    bpb_adv_cli.validate_config(cfg_obj)
    bpb_adv_cli.validate_all_stage_representative_target_contract(cfg_obj)

    if dist.is_available() and dist.is_initialized():
        ddp_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
    else:
        ddp_rank, ddp_world_size = 0, 1
    bpb_adv_cli._synchronize_run_tag(
        cfg_obj,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    run_tag = cfg_obj.run_tag
    assert run_tag is not None

    out_paths = bpb_adv_cli._paths(cfg_obj)
    profile_dir = out_paths.profile_dir()

    metadata = {
        "run_tag": run_tag,
        "stage": cfg_obj.stage,
        "target_models": list(cfg_obj.target_models),
        "base_methods": cfg_obj.base_methods,
        "adversarial_methods": [
            dataclasses.asdict(method) for method in cfg_obj.adversarial_methods
        ],
        "ddp_world_size": ddp_world_size,
        # Whole-config serialization must go through the shared null-stripping
        # serializer, exactly like the production eval script's two sites. The
        # per-method asdict above is a DIFFERENT dataclass and stays raw.
        "config": axbench_bpb_adv_config_to_dict(cfg_obj),
    }

    if ddp_rank == 0:
        logger.info(
            f"Starting AxBench-BPB-Adv profiled run run_tag={run_tag!r}; "
            f"profile output: {profile_dir}"
        )
        with AxBenchProfiler(profile_dir, metadata=metadata):
            bpb_adv_cli._run_resolved(
                cfg_obj,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
            )
        logger.info(f"AxBench-BPB-Adv profile written to {profile_dir}")
    else:
        # Non-zero ranks: run the production dispatcher WITHOUT a profiler.
        # The `profile_region`/`record_counter` calls inside become no-ops on
        # these ranks, matching the production behavior. This is the same
        # contract `prepare_context_cache` already expects from the dispatcher.
        bpb_adv_cli._run_resolved(
            cfg_obj,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )


@hydra.main(
    version_base=None,
    config_path="../../configs/eval/axbench_bpb_adv",
    config_name="axbench_bpb_adv_eval",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point."""
    _run(cfg)


__all__ = ["AxBenchProfiler", "_run", "main"]


if __name__ == "__main__":
    main()
