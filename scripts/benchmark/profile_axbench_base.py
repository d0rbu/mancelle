"""Profile the AxBench-Base evaluation pipeline.

This CLI composes the same Hydra config tree as ``scripts.eval.axbench_base_eval``,
activates :mod:`steering.eval.axbench_base.profiler`, and dispatches the
normal stage runner.  Outputs are written under:

    <run_output_root>/profile/events.jsonl
    <run_output_root>/profile/summary.yaml
    <run_output_root>/profile/summary.csv

Dev live run example (5 concepts, no mocks)::

    uv run python -m scripts.benchmark.profile_axbench_base tiers=dev \
        concept_description_fetch_limit=50 \
        concept_examples_qc_target_surviving=5 \
        verification_candidate_scan_limit=1000 \
        verification_candidate_attempt_limit=300 \
        feature_dir=... feature_statistics_path=... \
        decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
        encdec.source=encdec encdec.model_tag=d13 encdec.step=10000
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

import scripts.eval.axbench_base_eval as axbench_cli
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import validate_config
from steering.eval.axbench_base.config_loader import cfg_to_dataclass
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.runtime_state import (
    RuntimeState,
    validate_env_for_stages,
)
from steering.eval.cluster_config import axbench_base_config_to_dict


def _run(cfg: DictConfig) -> None:
    """Testable profiling entry point."""
    cfg_obj = cfg_to_dataclass(cfg)
    validate_config(cfg_obj)
    if cfg_obj.run_tag is None:
        cfg_obj = cfg_obj.replace(run_tag=axbench_cli._generate_run_tag(cfg_obj))
    run_tag = cfg_obj.run_tag
    assert run_tag is not None
    output_paths = paths.AxBenchBasePaths.from_config(cfg_obj)
    paths.assert_no_legacy_output_shadow(cfg_obj)

    stages = axbench_cli._resolve_stage_list(
        cfg_obj.stage,
        cfg_obj.run_negative_mode,
    )
    validate_env_for_stages(stages, cfg=cfg_obj)

    profile_dir = output_paths.profile_dir()
    metadata = {
        "run_tag": run_tag,
        "stage": cfg_obj.stage,
        "stages": [stage.value for stage in stages],
        "configured_output_root": cfg_obj.output_root,
        "effective_output_root": str(output_paths.output_root),
        "run_output_root": str(output_paths.run_output_root()),
        "max_concepts": cfg_obj.max_concepts,
        "run_negative_mode": cfg_obj.run_negative_mode,
        "concept_feature_mode": cfg_obj.concept_feature_mode,
        "enabled_methods": cfg_obj.enabled_methods,
        "disabled_methods": cfg_obj.disabled_methods,
        "concept_examples_qc_target_surviving": (
            cfg_obj.concept_examples_qc_target_surviving
        ),
        "concept_description_fetch_limit": cfg_obj.concept_description_fetch_limit,
        "verification_candidate_scan_limit": (
            cfg_obj.verification_candidate_scan_limit
        ),
        "verification_candidate_attempt_limit": (
            cfg_obj.verification_candidate_attempt_limit
        ),
        "config": axbench_base_config_to_dict(cfg_obj),
    }

    logger.info(
        f"Starting AxBench-Base profiled run run_tag={run_tag!r}; "
        f"profile output: {profile_dir}"
    )
    state = RuntimeState(cfg=cfg_obj)
    try:
        with AxBenchProfiler(profile_dir, metadata=metadata):
            axbench_cli._dispatch_stage(cfg_obj, state)
    finally:
        state.close()
    logger.info(f"AxBench-Base profile written to {profile_dir}")


@hydra.main(
    version_base=None,
    config_path="../../configs/eval/axbench_base",
    config_name="axbench_base",
)
def main(cfg: DictConfig) -> None:
    """CLI entry point."""
    _run(cfg)


__all__ = ["AxBenchProfiler", "_run", "main"]


if __name__ == "__main__":
    main()
