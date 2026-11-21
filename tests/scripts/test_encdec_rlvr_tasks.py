"""Task-selection contracts and frozen pre-selection sampling parity for RLVR."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train import conditional_encdec_rlvr as trainer
from steering.conditional.encoder_decoder import rlvr_runtime as runtime
from steering.conditional.encoder_decoder import sm12_fa4
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from tests.scripts.test_encdec_rlvr_runtime import _resume_identity_loaded, _rows, _settings

_TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K")
_SELECTIONS = tuple(selection for size in range(1, 5) for selection in permutations(_TASKS, size))
_INVALID_TASKS = (
    None, [], (), "MMLU", b"MMLU", {"MMLU": 1}, {"MMLU"},
    ["MMLU", "MMLU"], ["gsm8k"], ["ARC"], ["MMLU", ""],
    [True], [1], [None], [["MMLU"]],
)


def _legacy_select(rows, *, seed, step, ordinal):
    """Frozen sampler from main 4fbd6396, including its seed derivation."""
    payload = json.dumps(
        ["sha256_namespace_global_prompt_v1", seed, "prompt", step, ordinal],
        separators=(",", ":"),
    )
    seed_value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % 2**63
    rng = random.Random(seed_value)
    task = _TASKS[rng.randrange(4)]
    return rows[task][rng.randrange(len(rows[task]))]


@pytest.mark.parametrize("selection", _SELECTIONS)
def test_every_ordered_task_subset_samples_only_selected_tasks_uniformly(selection):
    rows = {task: tuple((task, index) for index in range(1 + 97 * n)) for n, task in enumerate(_TASKS)}
    runtime.validate_rlvr_settings(_settings(tasks=list(selection)), world_size=2)
    assert runtime.resolve_rlvr_tasks(list(selection)) == selection
    counts = Counter(
        runtime.select_training_row(rows, tasks=selection, seed=42, step=3, ordinal=i)[0]
        for i in range(4000)
    )
    assert set(counts) == set(selection)
    assert all(abs(count - 4000 / len(selection)) < 150 for count in counts.values())


@pytest.mark.parametrize("tasks", _INVALID_TASKS)
def test_bad_tasks_are_rejected_by_settings_and_sampler(tasks):
    with pytest.raises(SFTGRPOError, match="tasks"):
        runtime.validate_rlvr_settings(_settings(tasks=tasks), world_size=1)
    with pytest.raises(SFTGRPOError, match="tasks"):
        runtime.select_training_row(_rows(), tasks=tasks, seed=0, step=0, ordinal=0)


@pytest.mark.parametrize("seed", (0, 42, 2**63 - 1))
@pytest.mark.parametrize("step", (0, 1, 10))
def test_default_and_explicit_all_tasks_are_exactly_legacy_sampling(seed, step):
    rows = {task: tuple((task, index) for index in range(1 + 19 * n)) for n, task in enumerate(_TASKS)}
    for ordinal in range(256):
        expected = _legacy_select(rows, seed=seed, step=step, ordinal=ordinal)
        assert runtime.select_training_row(rows, seed=seed, step=step, ordinal=ordinal) == expected
        assert runtime.select_training_row(rows, tasks=_TASKS, seed=seed, step=step, ordinal=ordinal) == expected


def test_subset_sampling_preserves_global_rng_and_rank_partition_independence():
    rows = _rows()
    tasks = _TASKS[:-1]
    state = random.getstate()
    serial = [runtime.select_training_row(rows, tasks=tasks, seed=91, step=7, ordinal=i) for i in range(128)]
    assert random.getstate() == state
    partitions = {
        i: runtime.select_training_row(rows, tasks=tasks, seed=91, step=7, ordinal=i)
        for rank in range(4) for i in range(rank * 32, (rank + 1) * 32)
    }
    assert [partitions[i] for i in range(128)] == serial


def test_task_order_is_part_of_sampling_and_full_bundle_is_still_required():
    rows = _rows()
    forward = [runtime.select_training_row(rows, tasks=_TASKS, seed=4, step=0, ordinal=i) for i in range(64)]
    reverse = [runtime.select_training_row(rows, tasks=_TASKS[::-1], seed=4, step=0, ordinal=i) for i in range(64)]
    assert forward != reverse
    rows["GSM8K"] = ()
    with pytest.raises(SFTGRPOError, match="nonempty training rows"):
        runtime.select_training_row(rows, tasks=_TASKS[:-1], seed=4, step=0, ordinal=0)


def test_hydra_default_and_mc_only_cli_override():
    directory = Path(__file__).resolve().parents[2] / "configs" / "training"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        default = compose(config_name="conditional_encdec_rlvr")
        subset = compose(config_name="conditional_encdec_rlvr", overrides=["tasks=[ARC-Easy,ARC-Challenge,MMLU]"])
    assert list(default.tasks) == list(_TASKS)
    assert list(subset.tasks) == list(_TASKS[:-1])


def test_resume_identity_normalizes_defaults_and_pins_membership_and_order(monkeypatch):
    monkeypatch.setattr(trainer, "self_conditioned_code_identity", lambda: {"git_commit": "fixed"})
    monkeypatch.setattr(trainer, "feature_action_identity", lambda _policy: {"action": "fixed"})
    loaded = _resume_identity_loaded()

    def identity(**overrides):
        return trainer.resume_identity(loaded, _settings(**overrides), world_size=1,
            parameter_manifest=("weight",), attention_impl="sdpa", deterministic=True)

    default = identity()
    assert default == identity(tasks=list(_TASKS)) == identity(tasks=_TASKS)
    assert default["config"]["tasks"] == list(_TASKS)
    assert len({default["sha256"], identity(tasks=_TASKS[:-1])["sha256"], identity(tasks=_TASKS[::-1])["sha256"]}) == 3


@pytest.mark.parametrize("tasks,error", [(["MMLU", "MMLU"], "tasks"), (["MMLU"], "requires native FA4/CuTe")])
def test_bad_tasks_or_missing_fa4_abort_trainer_before_loading_any_artifact(monkeypatch, tasks, error):
    entered = []
    monkeypatch.setattr(trainer, "compute_init_for_checkpoint_resume", lambda *_args, **_kwargs: (False, 0, 0, 1, "cuda"))
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: entered.append("cleanup"))
    monkeypatch.setattr(trainer, "load_rlvr_inputs", lambda *_args, **_kwargs: pytest.fail("loaded artifacts before admission"))
    monkeypatch.setattr(trainer.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.t.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(sm12_fa4, "_native_fa4", None)
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", "fa4")
    with pytest.raises(SFTGRPOError, match=error):
        trainer.train(OmegaConf.create(_settings(tasks=tasks, device_type="cuda")))
    assert entered == ["cleanup"]


@pytest.mark.parametrize("rank", [0, 1])
def test_initial_parse_warnings_are_rank_zero_only(monkeypatch, tmp_path, rank):
    """Stop at the actual initial telemetry boundary, before any rollout."""
    class InitialTelemetryComplete(Exception):
        pass

    class Optimizer:
        pass

    events = []
    warnings = []

    class Telemetry:
        def __init__(self, *_args, **_kwargs):
            pass

        def log(self, step, metrics, samples):
            assert step == 0
            events.append("initial_telemetry")
            raise InitialTelemetryComplete

        def finish(self, **_kwargs):
            events.append("finish")

    cfg = OmegaConf.create({**_settings(tasks=list(_TASKS[:-1])), "output_dir": str(tmp_path / "run")})
    loaded = SimpleNamespace(model=t.nn.Module(), bundle=SimpleNamespace(train=[SimpleNamespace(task=task) for task in _TASKS]))
    metrics = {f"validation/{task}/{key}": value for task in _TASKS for key, value in (("parse_rate", 0), ("count", 2))}
    monkeypatch.setattr(trainer, "compute_init_for_checkpoint_resume", lambda *_args, **_kwargs: (True, rank, rank, 2, "cpu"))
    monkeypatch.setattr(trainer, "compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(trainer, "_assert_attention_impl", lambda **_kwargs: "cpu")
    monkeypatch.setattr(trainer, "_configure_deterministic_algorithms", lambda **_kwargs: True)
    monkeypatch.setattr(trainer, "load_rlvr_inputs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(trainer, "configure_grpo_trainable_parameters", lambda *_args, **_kwargs: ("weight",))
    monkeypatch.setattr(trainer, "_build_optimizer", lambda *_args, **_kwargs: Optimizer())
    monkeypatch.setattr(trainer, "DistMuonAdamW", Optimizer)
    monkeypatch.setattr(trainer, "resume_identity", lambda *_args, **_kwargs: {"sha256": "identity"})
    monkeypatch.setattr(trainer, "RLVRWandb", Telemetry)
    monkeypatch.setattr(trainer, "evaluate_rlvr", lambda *_args, **_kwargs: (metrics, []))
    monkeypatch.setattr(trainer.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))
    with pytest.raises(InitialTelemetryComplete):
        trainer.train(cfg)
    assert len(warnings) == (4 if rank == 0 else 0)
    assert events == ["initial_telemetry", "finish", "cleanup"]
