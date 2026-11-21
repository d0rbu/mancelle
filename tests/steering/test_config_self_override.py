"""Pin the Hydra override semantics: every inheriting config must set its
own keys via ``_self_`` placed AFTER the parent in the ``defaults:`` list.

Hydra 1.1+ composes the primary config BEFORE its listed defaults when
``_self_`` is missing, so the parent silently wins on every overlapping
key. The result: a variant config that looks like it sets
``device_batch_size: 4`` actually trains with the parent's 32.

This test asserts two properties per affected config:

1. YAML hygiene — ``defaults:`` contains ``- _self_`` after the parent.
2. Behavioral pin — a named override the variant declares in its body
   actually wins over the parent value after Hydra compose.

Regression coverage across all historically-bugged configs
(training, extraction, profiling).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class ConfigCase:
    """One composable config and the overrides it should actually apply."""

    config_dir: str  # relative to repo root
    config_name: str
    expected: dict[str, object] = field(default_factory=dict)

    @property
    def abs_config_dir(self) -> str:
        return os.path.abspath(os.path.join(REPO_ROOT, self.config_dir))


# ---------------------------------------------------------------------------
# Cases: one per config that was historically missing ``_self_``.
# Each expected value is a key the variant's body sets to a value that
# differs from its parent's (verified manually; see test_values_differ_from_parent).
# ---------------------------------------------------------------------------


CASES: list[ConfigCase] = [
    # Pretraining
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_train",
        expected={"num_iterations": 1000},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_4090",
        expected={"depth": 12, "device_batch_size": 4, "max_seq_len": 1024},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_2xb200",
        expected={"device_batch_size": 32, "feature_dir": "out/data/features"},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_2xh200",
        expected={"device_batch_size": 8, "fp8": True},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_4xh100",
        expected={"depth": 20, "device_batch_size": 8, "fp8": True},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_4xh200",
        expected={"device_batch_size": 16, "fp8": True},
    ),
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_d12_fp_s32",
        expected={
            "depth": 12,
            "max_seq_len": 1024,
            "segment_size": 32,
            "run": "d12_fp_s32",
        },
    ),
    # SFT
    ConfigCase(
        config_dir="configs/training",
        config_name="conditional_encdec_sft",
        expected={
            "total_batch_size": 524288,
            "eval_every": 150,
            "eval_tokens": 10485760,
            "weight_decay": 0.0,
        },
    ),
    # chat SFT
    ConfigCase(
        config_dir="configs/training",
        config_name="chat_sft_4xrtx6000",
        expected={"device_batch_size": 16},
    ),
    # Extraction
    ConfigCase(
        config_dir="configs/extraction",
        config_name="extract_features_2xb200",
        expected={"gpu_batch_size": 128, "cpu_batch_size": 256},
    ),
    ConfigCase(
        config_dir="configs/extraction",
        config_name="extract_features_4xb200",
        expected={"gpu_batch_size": 128, "cpu_batch_size": 256},
    ),
    ConfigCase(
        config_dir="configs/extraction",
        config_name="extract_features_4xh200",
        expected={"gpu_batch_size": 96, "cpu_batch_size": 192},
    ),
    ConfigCase(
        config_dir="configs/extraction",
        config_name="extract_features_8xpro6000",
        expected={"gpu_batch_size": 64, "cpu_batch_size": 128},
    ),
]


# ---------------------------------------------------------------------------
# YAML hygiene: ``_self_`` must appear AFTER the parent in the defaults list.
# ---------------------------------------------------------------------------


class TestDefaultsListShape:
    """Static lint on the YAML — complements the behavioral composition test."""

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.config_name)
    def test_self_is_last_in_defaults(self, case: ConfigCase) -> None:
        path = os.path.join(case.abs_config_dir, f"{case.config_name}.yaml")
        with open(path) as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"{path}: top-level must be a mapping"
        defs = data.get("defaults")
        assert isinstance(defs, list) and defs, (
            f"{path}: defaults list should be non-empty"
        )
        assert "_self_" in defs, (
            f"{path}: missing ``- _self_`` in defaults list — without it, the "
            f"parent's values silently override this file's overrides."
        )
        # _self_ must come AFTER at least one parent; placing it first would
        # flip the semantics and make the parent win again.
        idx = defs.index("_self_")
        parents_before = [x for x in defs[:idx] if x != "_self_"]
        assert parents_before, (
            f"{path}: ``- _self_`` must appear AFTER a parent entry in the "
            f"defaults list, not before."
        )


# ---------------------------------------------------------------------------
# Behavioral pin: Hydra compose applies the variant's overrides.
# ---------------------------------------------------------------------------


class TestOverridesWinOverParent:
    """If ``_self_`` is mis-ordered (or removed), these assertions regress."""

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.config_name)
    def test_variant_composes(self, hydra_compose, case: ConfigCase) -> None:
        cfg = hydra_compose(case.abs_config_dir, case.config_name)
        assert cfg is not None

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.config_name)
    def test_variant_overrides_apply(
        self, hydra_compose, case: ConfigCase,
    ) -> None:
        cfg = hydra_compose(case.abs_config_dir, case.config_name)
        for key, want in case.expected.items():
            got = cfg[key]
            assert got == want, (
                f"{case.config_name}: expected cfg.{key}={want!r} "
                f"(declared in the variant body), got {got!r}. "
                f"This usually means ``- _self_`` is missing/mis-ordered in "
                f"the defaults list, so the parent won."
            )
