"""Hyperparameter sweep launcher for encoder-decoder conditional pretraining.

Generates all combinations of:
- Feature selection: geometric(mean) and normal(mean, std)
- Negative sampling: geometric(mean)
- Feature delta: disabled, prev_segment, cum_max with neg_delta_ratio
- Feature prediction weight

Launches each config as a separate torchrun job.

Usage:
    # List all configs
    uv run python -m scripts.train.run_encdec_sweep mode=dry_run

    # Run config #42
    uv run python -m scripts.train.run_encdec_sweep mode=index index=42

    # Run configs 0-9
    uv run python -m scripts.train.run_encdec_sweep mode=range range=[0,10]

    # Run all (sequential)
    uv run python -m scripts.train.run_encdec_sweep mode=all

    # Custom GPU count
    uv run python -m scripts.train.run_encdec_sweep mode=index index=0 gpus=8
"""

from __future__ import annotations

import itertools
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import hydra
import torch as t
from omegaconf import DictConfig, ListConfig

from nanochat.config_validation import validate_sweep_mode

# ---------------------------------------------------------------------------
# Sweep dimensions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSelectionConfig:
    distribution: str  # "geometric" or "normal"
    mean: float
    std: float

    @property
    def short_name(self) -> str:
        if self.distribution == "geometric":
            return f"geo_m{self.mean:g}"
        return f"norm_m{self.mean:g}_s{self.std:g}"


@dataclass(frozen=True)
class NegativeSamplingConfig:
    mean: float

    @property
    def short_name(self) -> str:
        return f"geo_m{self.mean:g}"


class FeatureDeltaMode(Enum):
    disabled = "disabled"
    prev_segment = "prev_segment"
    cum_max = "cum_max"
    splice_counterfactual = "splice_counterfactual"


@dataclass(frozen=True)
class FeatureDeltaConfig:
    mode: FeatureDeltaMode
    neg_delta_ratio: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FeatureDeltaMode):
            raise TypeError(
                f"FeatureDeltaConfig.mode must be FeatureDeltaMode, got "
                f"{type(self.mode).__name__}."
            )
        if self.mode is FeatureDeltaMode.disabled:
            if self.neg_delta_ratio is not None:
                raise ValueError(
                    "FeatureDeltaConfig.neg_delta_ratio must be None when "
                    "mode is disabled."
                )
        elif self.neg_delta_ratio is None:
            raise ValueError(
                "FeatureDeltaConfig.neg_delta_ratio is required when feature "
                "delta is enabled."
            )

    @property
    def short_name(self) -> str:
        if self.mode is FeatureDeltaMode.disabled:
            return "disabled"
        assert self.neg_delta_ratio is not None
        return f"{self.mode.value}_r{self.neg_delta_ratio:g}"


@dataclass(frozen=True)
class SweepConfig:
    fs: FeatureSelectionConfig
    ns: NegativeSamplingConfig
    fd: FeatureDeltaConfig
    fp_weight: float

    @property
    def model_tag(self) -> str:
        """Hierarchical model_tag encoding all sweep params except feature_selection
        (which is captured by feature_selection_dirname in the checkpoint path)."""
        return (
            f"sweep_d12/"
            f"ns_{self.ns.short_name}/"
            f"fd_{self.fd.short_name}/"
            f"fpw_{self.fp_weight:g}"
        )

    def hydra_overrides(self, config_name: str = "conditional_encdec_sweep") -> list[str]:
        """Generate Hydra CLI overrides for this config."""
        overrides = [
            f"config_name={config_name}",
            f"model_tag={self.model_tag}",
            # Feature selection
            f"feature_selection.distribution={self.fs.distribution}",
            f"feature_selection.mean={self.fs.mean}",
            f"feature_selection.std={self.fs.std}",
            # Negative sampling
            "negative_selection.distribution=geometric",
            f"negative_selection.mean={self.ns.mean}",
        ]

        # Feature prediction: null disables the head entirely (saves FLOPs),
        # nonzero sets the auxiliary loss weight
        if self.fp_weight == 0.0:
            overrides.append("feature_prediction=null")
        else:
            overrides.append(f"feature_prediction.weight={self.fp_weight}")

        # Feature delta: enabled or disabled
        if self.fd.mode is not FeatureDeltaMode.disabled:
            assert self.fd.neg_delta_ratio is not None
            overrides.extend([
                f"feature_delta.mode={self.fd.mode.value}",
                f"feature_delta.neg_delta_ratio={self.fd.neg_delta_ratio}",
                # delta_weights is required when feature_delta is enabled
                "feature_selection.delta_weights=absolute_percentile",
            ])
        else:
            overrides.append("feature_delta=null")
            # Must not have delta_weights when feature_delta is disabled
            overrides.append("feature_selection.delta_weights=null")

        return overrides


# ---------------------------------------------------------------------------
# Generate all sweep configurations
# ---------------------------------------------------------------------------

def build_sweep_configs() -> list[SweepConfig]:
    """Build the cartesian product of sweep hyperparameters.

    Grid rationale (315 configs, reduced from 960):
    - Feature selection: 7 variants covering both distribution families and the
      practically relevant mean range (4-16). Geometric captures high-variance
      tails; normal(mean, std=1) captures tight concentration; normal(mean, std=4)
      captures moderate spread. Means 4/8/16 span the useful range (32 is rarely
      better than 16 in preliminary tests).
    - Negative sampling: 3 values (4/8/16) — mostly independent of feature
      selection, so fewer points suffice. Mean=32 dropped (diminishing returns).
    - Feature delta: 5 variants unchanged — qualitative mode differences
      (disabled/prev_segment/cum_max) are the most important axis to explore.
    - Feature prediction weight: 3 values with 0.0 baseline added (was missing)
      to measure whether the auxiliary loss helps at all. Replaced 0.01 with 0.0
      since 0.01 and 0.1 showed similar behavior in early tests.
    """
    # Feature selection: 3 geometric + 4 normal = 7 variants
    feature_selections = [
        # Geometric: high-variance sampling (std = mean)
        FeatureSelectionConfig("geometric", 16, 16),
        FeatureSelectionConfig("geometric", 8, 8),
        FeatureSelectionConfig("geometric", 4, 4),
        # Normal: tight (std=1) and moderate (std=4) spread at key means
        FeatureSelectionConfig("normal", 16, 1),
        FeatureSelectionConfig("normal", 16, 4),
        FeatureSelectionConfig("normal", 8, 1),
        FeatureSelectionConfig("normal", 8, 4),
    ]

    # Negative sampling: 3 geometric means
    negative_samplings = [
        NegativeSamplingConfig(16),
        NegativeSamplingConfig(8),
        NegativeSamplingConfig(4),
    ]

    # Feature delta configurations (unchanged — most important axis)
    feature_deltas = [
        FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
        FeatureDeltaConfig(FeatureDeltaMode.prev_segment, 0.1),
        FeatureDeltaConfig(FeatureDeltaMode.prev_segment, 1.0),
        FeatureDeltaConfig(FeatureDeltaMode.cum_max, 0.1),
        FeatureDeltaConfig(FeatureDeltaMode.cum_max, 1.0),
    ]

    # Feature prediction weights: added 0.0 baseline, dropped 0.01
    fp_weights = [0.0, 0.1, 1.0]

    configs = []
    for fs, ns, fd, fpw in itertools.product(
        feature_selections, negative_samplings, feature_deltas, fp_weights
    ):
        configs.append(SweepConfig(fs=fs, ns=ns, fd=fd, fp_weight=fpw))

    return configs


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def build_command(config: SweepConfig, gpus: int) -> list[str]:
    """Build the torchrun command for a single sweep config."""
    cmd = [
        "uv", "run",
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m", "scripts.train.conditional_encdec_train",
        "--",
    ]
    cmd.extend(config.hydra_overrides())
    return cmd


def run_config(config: SweepConfig, index: int, total: int, gpus: int, dry_run: bool = False):
    """Launch a single sweep configuration."""
    cmd = build_command(config, gpus)
    print(f"\n{'='*80}")
    print(f"[{index+1}/{total}] {config.model_tag}")
    print(f"  Feature selection: {config.fs.distribution}(mean={config.fs.mean}, std={config.fs.std})")
    print(f"  Negative sampling: geometric(mean={config.ns.mean})")
    print(f"  Feature delta: {config.fd.short_name}")
    print(f"  FP weight: {config.fp_weight}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return

    project_root = Path(__file__).resolve().parent.parent.parent
    # Source env.sh for LD_LIBRARY_PATH and NANOCHAT_ATTN_IMPL, then exec.
    # Quote each arg to prevent shell expansion (e.g. ~ in Hydra delete overrides).
    # Use PYTHONUNBUFFERED=1 for real-time log output.
    shell_cmd = (
        f"source {project_root}/env.sh && "
        f"export PYTHONUNBUFFERED=1 && "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        + " ".join(shlex.quote(c) for c in cmd)
    )
    result = subprocess.run(["bash", "-c", shell_cmd], cwd=project_root, check=False)
    if result.returncode != 0:
        print(f"  FAILED with return code {result.returncode}", file=sys.stderr)
        return result.returncode
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs/training",
    config_name="run_encdec_sweep",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    validate_sweep_mode(cfg)

    configs = build_sweep_configs()
    total = len(configs)
    gpus = cfg.gpus if cfg.gpus is not None else (t.cuda.device_count() if t.cuda.is_available() else 1)
    print(f"Total sweep configs: {total} (gpus={gpus})")

    if cfg.dump_json is not None:
        data = []
        for i, c in enumerate(configs):
            data.append(
                {
                    "index": i,
                    "model_tag": c.model_tag,
                    "command": shlex.join(build_command(c, gpus)),
                    "overrides": c.hydra_overrides(),
                }
            )
        with open(cfg.dump_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Dumped {len(configs)} configs to {cfg.dump_json}")
        return

    mode = cfg.mode

    if mode == "dry_run":
        for i, c in enumerate(configs):
            run_config(c, i, total, gpus, dry_run=True)
        return

    if isinstance(mode, int):
        if mode < 0 or mode >= total:
            print(f"Error: index {mode} out of range [0, {total})", file=sys.stderr)
            sys.exit(1)
        rc = run_config(configs[mode], mode, total, gpus)
        sys.exit(rc or 0)

    if isinstance(mode, (list, ListConfig)):
        start, end = mode
        start = max(0, start)
        end = min(end, total)
        for i in range(start, end):
            rc = run_config(configs[i], i, total, gpus)
            if rc:
                print(f"Stopping sweep due to failure at index {i}", file=sys.stderr)
                sys.exit(rc)
        return

    # mode is None — run all
    for i, c in enumerate(configs):
        rc = run_config(c, i, total, gpus)
        if rc:
            print(f"Stopping sweep due to failure at index {i}", file=sys.stderr)
            sys.exit(rc)


if __name__ == "__main__":
    main()
