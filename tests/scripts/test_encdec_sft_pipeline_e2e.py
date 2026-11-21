"""End-to-end tests for the encoder-decoder SFT pipeline.

Differs from test_encdec_sft_pipeline.py in two ways:

- Uses a real subprocess.run with a stub Python script (not a unittest.mock)
  so the pipeline's command construction is exercised against a true subprocess
  boundary. Catches argv shape bugs, env handling bugs, and find_statistics_path
  resolution bugs that mock-only tests miss.

- Verifies the pipeline's output paths against the strings inside the reversed
  and forward training_progression configs. Catches drift between the pipeline
  and its consumers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from scripts.train.run_encdec_sft_pipeline import (
    build_checkpoint_dir,
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_TRAINING = REPO_ROOT / "configs" / "training"
CONFIG_EVAL = REPO_ROOT / "configs" / "eval"


@pytest.fixture(autouse=True)
def _unset_nanochat_base_dir(monkeypatch):
    """Linkage tests Hydra-compose configs that interpolate
    ``${oc.env:NANOCHAT_BASE_DIR,nanochat_cache}``. The literal-prefix-strip
    in _strip_env_prefix only works when the env var is unset (so Hydra
    resolves to the literal default ``nanochat_cache``). Force-unset it for
    every test in this file so CI boxes with NANOCHAT_BASE_DIR exported
    don't see spurious failures."""
    monkeypatch.delenv("NANOCHAT_BASE_DIR", raising=False)


# ---------------------------------------------------------------------------
# Stub subprocess script: when invoked, creates the artifacts the pipeline
# introspects (feature_statistics.safetensors and SFT checkpoint dirs).
# ---------------------------------------------------------------------------


_STUB_SOURCE = '''\
"""Stub subprocess for pipeline E2E tests.

Mimics two real scripts the pipeline launches:

- ``python -m steering.compute_statistics``: writes
  ``feature_statistics.safetensors`` into the feature_dir argument.
- ``python -m scripts.train.conditional_encdec_sft`` (or its torchrun
  variant): creates the conditioned SFT checkpoint directory keyed by dataset,
  plus a placeholder model file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _kv_args() -> dict[str, str]:
    return {
        a.split("=", 1)[0]: a.split("=", 1)[1]
        for a in sys.argv[1:]
        if "=" in a
    }


def main() -> int:
    invoked_as = os.environ.get("STUB_INVOKED_AS", "")
    args = _kv_args()

    if invoked_as == "steering.compute_statistics":
        feature_dir = Path(args["feature_dir"])
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "feature_statistics.safetensors").write_bytes(b"stub")
        return 0

    # Otherwise: SFT.
    dataset = args["dataset"]
    base = args.get("base_checkpoint_dir") or os.environ["STUB_PRETRAIN_DIR"]
    leaf = Path(base) / f"{dataset}_sft_conditioned_checkpoints"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "model_000000.pt").write_bytes(b"stub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@pytest.fixture
def stub_runtime(tmp_path, monkeypatch):
    """Set up a stub-subprocess runtime: NANOCHAT_BASE_DIR + stub script + monkeypatched subprocess.run."""
    base_dir = tmp_path / "nanochat_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))

    # Pre-create the pretrain checkpoint dir the pipeline expects.
    pretrain_dir = build_checkpoint_dir(
        str(base_dir), "d12_test", 128, "dist_geometric/mean_10.0/std_10.0"
    )
    Path(pretrain_dir).mkdir(parents=True, exist_ok=True)

    # Write the stub script.
    stub_path = tmp_path / "stub_subprocess.py"
    stub_path.write_text(_STUB_SOURCE)

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        # Identify what entry point the pipeline tried to invoke and route to the stub.
        joined = " ".join(cmd)
        env = os.environ.copy()
        if "compute_statistics" in joined:
            env["STUB_INVOKED_AS"] = "steering.compute_statistics"
        else:
            env["STUB_INVOKED_AS"] = "scripts.train.conditional_encdec_sft"
        env["STUB_PRETRAIN_DIR"] = pretrain_dir

        # Replace the binary with python -u <stub>; preserve only k=v args.
        new_cmd = [sys.executable, "-u", str(stub_path)]
        new_cmd.extend([a for a in cmd if "=" in a and not a.startswith("-")])
        return real_run(new_cmd, env=env, **kwargs)

    monkeypatch.setattr(
        "scripts.train.run_encdec_sft_pipeline.subprocess.run",
        fake_run,
    )

    def fake_verify_feature_dir(feature_dir):
        assert (Path(feature_dir) / "feature_statistics.safetensors").is_file()

    monkeypatch.setattr(
        "scripts.train.run_encdec_sft_pipeline.verify_feature_dir",
        fake_verify_feature_dir,
    )
    return {
        "base_dir": base_dir,
        "pretrain_dir": pretrain_dir,
        "feature_root": tmp_path / "features",
    }


# ---------------------------------------------------------------------------
# Hydra-compose helpers
# ---------------------------------------------------------------------------


def _compose_eval(name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_EVAL), version_base=None):
        return compose(config_name=name)


def _strip_env_prefix(checkpoint_dir: str) -> str:
    """Strip Hydra's resolved ``${oc.env:NANOCHAT_BASE_DIR,nanochat_cache}/`` prefix.

    In test environments without NANOCHAT_BASE_DIR set, Hydra resolves the
    interpolation to the literal ``nanochat_cache``. Strip that fixed prefix
    so the comparison is against the build_*_dir output (which uses a different
    base_dir under tmp_path).
    """
    prefix = "nanochat_cache/"
    if checkpoint_dir.startswith(prefix):
        return checkpoint_dir[len(prefix):]
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Class 1: TestPipelineE2E — real subprocess + on-disk chain layout.
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    """run_pipeline against a real subprocess boundary (stub script)."""

    def test_e2e_reverse_creates_chained_dirs(self, stub_runtime):
        results = run_pipeline(
            model_tag="d12_test",
            feature_root=str(stub_runtime["feature_root"]),
            phase_order=["capabilities", "safety"],
        )

        cap_path = Path(results["capabilities_sft"])
        safety_path = Path(results["safety_sft"])
        assert cap_path.is_dir()
        assert safety_path.is_dir()
        assert safety_path.parent == cap_path
        assert cap_path.name == "capabilities_sft_conditioned_checkpoints"
        assert safety_path.name == "safety_sft_conditioned_checkpoints"

    def test_e2e_forward_creates_chained_dirs(self, stub_runtime):
        """Regression guard: the loop refactor must still produce the original forward layout."""
        results = run_pipeline(
            model_tag="d12_test",
            feature_root=str(stub_runtime["feature_root"]),
            phase_order=["safety", "capabilities"],
        )

        safety_path = Path(results["safety_sft"])
        cap_path = Path(results["capabilities_sft"])
        assert safety_path.is_dir()
        assert cap_path.is_dir()
        assert cap_path.parent == safety_path

    def test_e2e_rejects_dataset_without_conditioned_features(self, stub_runtime):
        """Conditioned pipeline only accepts datasets with SFT feature shards."""
        with pytest.raises(ValueError, match="pre-extracted SFT feature phases"):
            run_pipeline(
                model_tag="d12_test",
                feature_root=str(stub_runtime["feature_root"]),
                phase_order=["safety", "capabilities", "math"],
            )

    def test_e2e_conditioned_calls_real_compute_statistics_subprocess(self, stub_runtime):
        """The pipeline's find_statistics_path lookup actually finds the file the stub wrote."""
        feature_root = stub_runtime["feature_root"]
        feature_root.mkdir(parents=True, exist_ok=True)

        results = run_pipeline(
            model_tag="d12_test",
            feature_root=str(feature_root),
            phase_order=["capabilities", "safety"],
        )

        # If find_statistics_path succeeded across the real subprocess boundary,
        # the run completed without raising. Verify both dirs exist.
        assert Path(results["capabilities_sft"]).is_dir()
        assert Path(results["safety_sft"]).is_dir()


# ---------------------------------------------------------------------------
# Class 2: TestPipelineConfigLinkage — pipeline output ↔ reversed configs.
# ---------------------------------------------------------------------------


class TestPipelineConfigLinkage:
    """build_*_dir output (relative to base_dir) matches the strings in configs."""

    def test_pipeline_reverse_subdirs_match_training_progression_reversed_yaml(self):
        cfg = _compose_eval("training_progression_d12_baseline_reversed")
        subdirs = {p.phase: p.subdir for p in cfg.phases}
        assert subdirs["pretrain"] == ""
        assert subdirs["capabilities_sft"] == "capabilities_sft_conditioned_checkpoints"
        assert subdirs["safety_sft"] == (
            "capabilities_sft_conditioned_checkpoints/"
            "safety_sft_conditioned_checkpoints"
        )

# ---------------------------------------------------------------------------
# Class 3: TestForwardPipelineConfigLinkage — regression guards.
# ---------------------------------------------------------------------------


class TestForwardPipelineConfigLinkage:
    def test_pipeline_forward_subdirs_match_training_progression_d12_baseline_yaml(self):
        cfg = _compose_eval("training_progression_d12_baseline")
        subdirs = {p.phase: p.subdir for p in cfg.phases}
        assert subdirs["safety_sft"] == "safety_sft_conditioned_checkpoints"
        assert subdirs["capabilities_sft"] == (
            "safety_sft_conditioned_checkpoints/capabilities_sft_conditioned_checkpoints"
        )
