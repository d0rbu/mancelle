"""AxBench config goldens and stable per-example RNG contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch as t

from steering.eval.axbench_upstream import methods
from steering.eval.self_conditioning_policy import (
    self_conditioning_example_seeds,
)
from steering.segmentation import SegmentMode
from tests.scripts.eval.test_axbench_project_methods import (
    _policy,
    _setup,
    _TargetModel,
)
from tests.steering.self_conditioning_policy_helpers import config_snapshot


def _golden_setup():
    setup = _setup()
    setup.percentile_table = t.arange(396, dtype=t.float64).reshape(4, 99) / 8
    setup.percentile_table[0, 0] = -0.0
    return setup


@pytest.mark.parametrize("mode", [SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens])
@pytest.mark.parametrize("policy_name", ["none", "interpolation", "masking"])
def test_pre_refactor_axbench_golden(mode: SegmentMode, policy_name: str) -> None:
    setup = _golden_setup()
    setup.segment_mode = mode
    setup.turn_self_cond_segment_size_rustbpe = 33 if mode is SegmentMode.turn_delimited_fixed_tokens else None
    config = methods.build_self_conditioning_config(
        setup, _TargetModel(), None if policy_name == "none" else _policy(policy_name), selection_seed=123,
    )
    golden = json.loads((Path(__file__).parent / "golden/self_conditioning_configs.json").read_text())
    assert config_snapshot(config) == golden[f"{mode.value}/{policy_name}"]


def _seeds(**overrides: Any) -> tuple[int, int]:
    arguments: dict[str, Any] = {
        "base_seed": 1234, "task_name": "GSM8K", "original_problem_index": 42,
        "prompt_token_key": (0, 3, 9), "sample_index": 0,
    }
    arguments.update(overrides)
    return self_conditioning_example_seeds(**arguments)


def test_seed_domains_identity_and_shard_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {i: _seeds(original_problem_index=i) for i in range(6)}
    monkeypatch.setenv("SHARD_OFFSET", "17")
    monkeypatch.setenv("RANK", "3")
    assert {i: _seeds(original_problem_index=i) for i in [4, 1, 3, 0, 5, 2]} == expected
    seed = _seeds()
    assert seed == (7638622587628405261, 1736863972553320511)
    assert seed[0] != seed[1]
    assert all(0 <= value < 2**63 for value in seed)
    for change in ({"base_seed": 1}, {"task_name": "MMLU"}, {"original_problem_index": 43},
                   {"prompt_token_key": (0, 3, 10)}, {"sample_index": 1}):
        assert all(a != b for a, b in zip(seed, _seeds(**change), strict=True))
