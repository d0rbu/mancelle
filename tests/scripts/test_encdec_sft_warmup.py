"""Integer warmup contracts through Hydra, real SFT updates and checkpoint IO."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch as t

from nanochat import checkpoint_manager as checkpoints
from nanochat.optim import MuonAdamW
from scripts.train import conditional_encdec_sft as sft
from steering._common.lr_schedules import sft_flat_decay, sft_warmup_flat_decay
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.segmentation import SegmentMode
from tasks.common import Task
from tests.encdec_test_helpers import N_EMBD
from tests.scripts.test_encdec_sft_segment_mode_startup import _patched_main_runtime

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"


class Conversations(Task):
    """Unequal conversation lengths make consumption independent of updates."""

    def __init__(self, lengths: tuple[int, ...] = (12, 90, 20, 32, 17, 70) * 8):
        super().__init__()
        self.lengths = lengths

    def num_examples(self):
        return len(self.lengths)

    def get_example(self, index):
        return {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "a" * self.lengths[index]},
            ]
        }


@dataclass
class Run:
    rates: list[list[tuple[float, float]]] = field(default_factory=list)
    positions: list[tuple[bool, float]] = field(default_factory=list)
    local_progress: list[float] = field(default_factory=list)
    saves: list[dict] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    models: list[Any] = field(default_factory=list)
    optimizers: list[Any] = field(default_factory=list)
    autocast_modes: list[tuple[bool, t.dtype]] = field(default_factory=list)
    checkpoint_dir: str = ""


@contextmanager
def runtime(
    root: Path,
    *,
    overrides: tuple[str, ...] = (),
    head: str = "headless",
    resume_dir: str | None = None,
    task: Task | None = None,
    device: str = "cpu",
    attention_backend: str = "fa4",
    eval_every: int = 0,
) -> Iterator[Run]:
    """Reuse real artifact fixtures; replace their simplified optimizer with MuonAdamW."""
    real_setup = sft._setup_sft_optimizer
    real_sync = sft.sync_stop_and_progress
    real_evaluate = sft.evaluate_encdec_bpb
    real_compile = t.compile
    t.manual_seed(1739)
    result = Run()
    fp_config = None
    if head != "headless":
        block = (
            FeaturePredictionBlock.TRANSFORMER
            if head == "transformer"
            else FeaturePredictionBlock.MLP
        )
        depth = 0 if head == "linear" else 2
        fp_config = FeaturePredictionConfig(block=block, head_layers=depth)
        overrides += (
            f"feature_prediction_block={block.value}",
            f"feature_prediction_head_layers={depth}",
        )
    with (
        pytest.MonkeyPatch.context() as mp,
        _patched_main_runtime(
            root,
            mp,
            real_store=eval_every == 0,
            eval_every=eval_every,
            train_mode=SegmentMode.fixed_tokens,
            task=Conversations() if task is None else task,
            n_segments_per_conv=1 if head == "headless" else 3,
            feature_pred_config=fp_config,
            n_embd=256 if device == "cuda" and head != "headless" else N_EMBD,
        ),
    ):
        source, _, meta = sft.load_checkpoint("base", None, "cpu")

        def load(_directory, step, device, **kwargs):
            if kwargs.get("load_optimizer"):
                assert resume_dir is not None
                return checkpoints.load_checkpoint(resume_dir, step, device, **kwargs)
            return {k: v.clone().to(device) for k, v in source.items()}, None, copy.deepcopy(meta)

        def save(directory, step, model, optimizer, metadata, *, rank):
            result.events.append("save")
            checkpoints.save_checkpoint(directory, step, model, optimizer, metadata, rank=rank)
            result.checkpoint_dir = str(directory)
            result.saves.append(copy.deepcopy(metadata))

        def setup(model, cfg):
            optimizer = real_setup(model, cfg)
            result.models.append(model)
            result.optimizers.append(optimizer)
            grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
            assert sorted(grouped) == sorted(id(p) for p in model.parameters())

            def before_update(current, _args, _kwargs):
                result.events.append("update")
                result.rates.append([(g["lr"], g["initial_lr"]) for g in current.param_groups])
                assert all(
                    t.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                )

            optimizer.register_step_pre_hook(before_update)
            if device == "cuda":

                def before_forward(_model, _args):
                    result.autocast_modes.append(
                        (t.is_autocast_enabled("cuda"), t.get_autocast_dtype("cuda"))
                    )

                model.register_forward_pre_hook(before_forward)
            return optimizer

        def sync(**kwargs):
            result.events.append("boundary")
            result.local_progress.append(kwargs["progress"])
            position = real_sync(**kwargs)
            result.positions.append(position)
            return position

        def evaluate(*args, **kwargs):
            result.events.append("validation")
            value = real_evaluate(*args, **kwargs)
            assert math.isfinite(value)
            return value

        mp.setattr(sft, "load_checkpoint", load)
        mp.setattr(sft, "save_checkpoint", save)
        mp.setattr(sft, "_setup_sft_optimizer", setup)
        mp.setattr(sft, "sync_stop_and_progress", sync)
        mp.setattr(sft, "evaluate_encdec_bpb", evaluate)
        if device == "cuda":
            mp.setattr(t, "compile", real_compile)
            mp.setattr(sft, "compute_init", lambda _: (False, 0, 0, 1, "cuda"))
            mp.setattr(
                sft.flash_attention,
                "_override_impl",
                sft.flash_attention.FlashAttnImpl(attention_backend),
            )
        # Static scales and independent head rates make accidental group omission visible.
        mp.setattr(
            sft.sys,
            "argv",
            [
                *sft.sys.argv,
                "lr_multiplier=0.4",
                "init_lr_frac=0.3",
                "fp_head_lr=0.003",
                "fp_head_matrix_lr=0.007",
                f"device_type={device}",
                *overrides,
            ],
        )
        with t.compiler.set_stance("force_eager") if device == "cpu" else nullcontext():
            yield result


def assert_exact(actual, expected):
    if isinstance(expected, t.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert t.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_exact(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("window", [0, 1, 20, 73])
def test_schedule_independent_boundaries(window):
    for step, progress, expected_decay in [
        (0, 0.0, 1.0),
        (1, 0.01, 1.0),
        (max(0, window - 1), 0.3, 1.0),
        (window, 0.4, 1.0),
        (80, 0.8, 1.0),
        (90, 0.9, 0.5),
        (100, 1.0, 0.0),
    ]:
        ramp = 1.0 if window == 0 or step >= window else (step + 1) / window
        assert sft_warmup_flat_decay(
            progress, completed_steps=step, warmup_steps=window
        ) == pytest.approx(expected_decay * ramp, abs=1e-15)


def test_disabled_schedule_exact_legacy_grid():
    values = [i / 8192 for i in range(-128, 8321)]
    for boundary in (0.0, 0.8, 1.0):
        values.extend(
            [math.nextafter(boundary, -math.inf), boundary, math.nextafter(boundary, math.inf)]
        )
    for value in values:
        assert sft_warmup_flat_decay(value, completed_steps=10, warmup_steps=0) == sft_flat_decay(
            value
        )


@pytest.mark.parametrize(
    "name",
    sorted(p.stem for p in CONFIG_DIR.glob("conditional_encdec_sft*.yaml") if "grpo" not in p.stem),
)
def test_all_presets(hydra_compose, name):
    cfg = hydra_compose(str(CONFIG_DIR), name)
    assert cfg.lr_warmup_steps == (0 if name == "conditional_encdec_sft_profile_4090" else 20)
    sft._validate_sft_lr_schedule(
        warmup_steps=cfg.lr_warmup_steps, num_iterations=cfg.num_iterations
    )
    assert "lr_warmup_frac" not in cfg


@pytest.mark.parametrize(
    "override", ["-1", "true", "false", "20.0", "0.0", '"20"', "null", "[]", "{}", "[20]"]
)
def test_strict_config_before_compute(monkeypatch, override):
    monkeypatch.setattr(sft.sys, "argv", ["sft", f"lr_warmup_steps={override}"])
    monkeypatch.setattr(sft, "compute_init", lambda *_: pytest.fail("reached device init"))
    with pytest.raises(ValueError, match="lr_warmup_steps must be a nonnegative integer"):
        sft.main()


@pytest.mark.parametrize("value", ["null", "0", "0.2", '"0.2"'])
def test_fraction_override_rejected(monkeypatch, value):
    monkeypatch.setattr(sft.sys, "argv", ["sft", f"+lr_warmup_frac={value}"])
    monkeypatch.setattr(sft, "compute_init", lambda *_: pytest.fail("reached device init"))
    with pytest.raises(ValueError, match="lr_warmup_frac is unsupported"):
        sft.main()


@pytest.mark.parametrize("window,horizon", [(20, 25), (4, 5), (1, 1), (81, 100)])
def test_overlap_before_compute(monkeypatch, window, horizon):
    monkeypatch.setattr(
        sft.sys, "argv", ["sft", f"lr_warmup_steps={window}", f"num_iterations={horizon}"]
    )
    monkeypatch.setattr(sft, "compute_init", lambda *_: pytest.fail("reached device init"))
    with pytest.raises(ValueError, match="strictly before 80%.*lr_warmup_steps=0"):
        sft.main()


@pytest.mark.parametrize(
    "window,horizon,expected", [(20, 26, 26), (0, 1, 1), (0, -1, None), (20, -1, None)]
)
def test_valid_horizons(window, horizon, expected):
    assert sft._validate_sft_lr_schedule(warmup_steps=window, num_iterations=horizon) == (
        window,
        expected,
    )


@pytest.mark.parametrize("head", ["headless", "linear", "mlp", "transformer"])
def test_real_loop_all_groups(tmp_path, head):
    with runtime(tmp_path, head=head, overrides=("num_iterations=26", "lr_warmup_steps=20")) as run:
        sft.main()
    assert isinstance(run.optimizers[0], MuonAdamW)
    assert {g["kind"] for g in run.optimizers[0].param_groups} == {"adamw", "muon"}
    assert len(run.rates) == 26
    for step, rates in enumerate(run.rates):
        decay = 1.0 if step <= 20 else (26 - step) / (26 * 0.2)
        multiplier = min(1.0, (step + 1) / 20) * decay
        for (lr, initial), (_, original) in zip(rates, run.rates[0], strict=True):
            assert initial == original
            assert lr == pytest.approx(initial * multiplier, abs=1e-15)
    assert run.positions[-1] == (True, 1.0)
    assert run.saves[-1]["user_config"]["lr_warmup_steps"] == 20
    if head != "headless":
        assert run.saves[-1]["loop_state"]["smooth_fp_loss"] > 0


@pytest.mark.parametrize("head", ["headless", "linear", "mlp", "transformer"])
def test_disabled_real_training_exact_legacy(tmp_path, head):
    options = ("num_iterations=6", "lr_warmup_steps=0")
    with runtime(tmp_path / "new", head=head, overrides=options) as current:
        sft.main()
    with (
        runtime(tmp_path / "legacy", head=head, overrides=options) as legacy,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(sft, "sft_warmup_flat_decay", lambda progress, **_: sft_flat_decay(progress))
        sft.main()
    assert current.rates == legacy.rates
    assert_exact(current.models[0].state_dict(), legacy.models[0].state_dict())
    assert_exact(current.optimizers[0].state_dict(), legacy.optimizers[0].state_dict())


@pytest.mark.parametrize("accum,prefetch", [(1, 0), (2, 0), (2, 2), (3, 1)])
def test_update_count_not_microbatches_or_prefetch(tmp_path, accum, prefetch):
    with runtime(
        tmp_path,
        overrides=(
            "num_iterations=6",
            "lr_warmup_steps=3",
            f"total_batch_size={128 * accum}",
            f"sft_prefetch_batches={prefetch}",
        ),
    ) as run:
        sft.main()
    assert len(run.rates) == 6
    for step, rates in enumerate(run.rates):
        expected = [1 / 3, 2 / 3, 1.0, 1.0, 1.0, 5 / 6][step]
        assert all(lr == pytest.approx(base * expected) for lr, base in rates)
    assert run.positions == [(s == 6, s / 6) for s in range(7)]


@pytest.mark.parametrize(
    "window,horizon,restart,prefetch,head,batch_size",
    [
        *(
            (2, 6, step, prefetch, "transformer", 256)
            for prefetch in (0, 2)
            for step in (1, 2, 3, 5)
        ),
        *((20, 100, step, 0, "headless", 128) for step in (10, 19, 20, 21, 90)),
    ],
)
def test_exact_checkpoint_restart(tmp_path, window, horizon, restart, prefetch, head, batch_size):
    options = (
        f"num_iterations={horizon}",
        f"lr_warmup_steps={window}",
        f"total_batch_size={batch_size}",
        f"sft_prefetch_batches={prefetch}",
        f"save_every={restart}",
    )
    with runtime(tmp_path / "full", head=head, overrides=options) as full:
        sft.main()
    with runtime(
        tmp_path / "resumed",
        head=head,
        resume_dir=full.checkpoint_dir,
        overrides=(
            *options,
            f"resume_from_step={restart}",
            "lr_multiplier=0.9",
            "init_lr_frac=0.8",
        ),
    ) as resumed:
        sft.main()
    assert resumed.rates == full.rates[restart:]
    if window == 20 and restart == 10:
        assert all(lr == base * (11 / 20) for lr, base in resumed.rates[0])
    assert_exact(resumed.models[0].state_dict(), full.models[0].state_dict())
    assert_exact(resumed.optimizers[0].state_dict(), full.optimizers[0].state_dict())
    for key in (
        "sft_consumed",
        "sft_dataloader_state",
        "sft_artifact_coverage",
        "smooth_train_loss",
        "smooth_fp_loss",
    ):
        assert_exact(resumed.saves[-1]["loop_state"][key], full.saves[-1]["loop_state"][key])


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr_warmup_steps", -1),
        ("lr_warmup_steps", True),
        ("lr_warmup_steps", 20.0),
        ("lr_warmup_steps", "20"),
        ("lr_warmup_steps", None),
        ("lr_warmup_steps", []),
        ("lr_warmup_steps", {}),
        ("num_iterations", 0),
        ("num_iterations", -2),
        ("num_iterations", True),
        ("num_iterations", 100.0),
        ("num_iterations", "100"),
        ("num_iterations", None),
        ("num_iterations", []),
    ],
)
def test_malformed_saved_schedule(field, value):
    saved = {"lr_warmup_steps": 20, "num_iterations": 100, field: value}
    with pytest.raises(ValueError, match=f"{field} must"):
        sft._assert_resume_sft_lr_schedule_matches(
            {"user_config": saved}, warmup_steps=20, num_iterations=100
        )


@pytest.mark.parametrize("saved", [None, [], "bad"])
def test_malformed_saved_user_config(saved):
    with pytest.raises(ValueError, match="requires a user_config mapping"):
        sft._assert_resume_sft_lr_schedule_matches(
            {"user_config": saved}, warmup_steps=20, num_iterations=100
        )


def test_legacy_resume_requires_explicit_zero():
    meta = {"user_config": {"num_iterations": -1}}
    with pytest.raises(ValueError, match="legacy checkpoint.*lr_warmup_steps=0"):
        sft._assert_resume_sft_lr_schedule_matches(meta, warmup_steps=20, num_iterations=None)
    sft._assert_resume_sft_lr_schedule_matches(meta, warmup_steps=0, num_iterations=None)


@pytest.mark.parametrize(
    "overrides", [("lr_warmup_steps=1",), ("num_iterations=7",), ("num_iterations=-1",)]
)
def test_resume_rejection_precedes_optimizer(tmp_path, overrides):
    options = ("num_iterations=6", "lr_warmup_steps=2")
    with runtime(tmp_path / "full", overrides=options) as full:
        sft.main()
    with runtime(
        tmp_path / "restart",
        resume_dir=full.checkpoint_dir,
        overrides=(*options, "resume_from_step=1", *overrides),
    ) as resumed:
        with pytest.raises(ValueError, match="SFT resume LR schedule mismatch"):
            sft.main()
        assert not resumed.optimizers and not resumed.saves


@pytest.mark.parametrize(
    "lengths,window,progress,step",
    [
        ((90,) * 10, 4, 0.8, 4),
        ((90,) * 4, 3, 1.0, 2),
        ((12, 90, 20, 32, 17, 70) * 2, 20, None, None),
    ],
)
def test_epoch_overlap_before_publication(tmp_path, lengths, window, progress, step):
    with runtime(
        tmp_path,
        task=Conversations(lengths),
        eval_every=1,
        overrides=("num_iterations=-1", f"lr_warmup_steps={window}", "eval_tokens=128"),
    ) as run:
        with pytest.raises(ValueError, match="epoch LR warmup must finish strictly before 80%"):
            sft.main()
    assert run.events[-1] == "boundary"
    assert run.positions[-1][1] >= 0.8
    if progress is not None:
        assert run.positions[-1][1] == progress
        assert len(run.rates) == step


@pytest.mark.parametrize("window", [0, 1, 3])
def test_variable_epoch_valid_completion(tmp_path, window):
    with runtime(
        tmp_path,
        overrides=(
            "num_iterations=-1",
            f"lr_warmup_steps={window}",
            "max_seq_len=256",
            "total_batch_size=512",
        ),
    ) as run:
        sft.main()
    assert run.positions[-1][0]
    increments = [b[1] - a[1] for a, b in zip(run.positions, run.positions[1:], strict=False)]
    assert len({round(value, 6) for value in increments}) > 1
    assert run.saves[-1]["step"] == len(run.rates)


def test_fresh_phase_from_sft_weights_starts_own_warmup(tmp_path):
    with runtime(tmp_path / "base", overrides=("num_iterations=6", "lr_warmup_steps=0")) as base:
        sft.main()
    with (
        runtime(tmp_path / "fresh", overrides=("num_iterations=6", "lr_warmup_steps=2")) as fresh,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(
            sft,
            "load_checkpoint",
            lambda _directory, _step, device, **_kwargs: checkpoints.load_checkpoint(
                base.checkpoint_dir, 6, device, load_optimizer=False
            ),
        )
        sft.main()
    assert len(fresh.rates) == 6
    assert all(lr == initial / 2 for lr, initial in fresh.rates[0])


@pytest.mark.parametrize(
    "field,value", [("lr_warmup_steps", None), ("num_iterations", 26.0), ("legacy", None)]
)
def test_real_resume_metadata_rejection_before_optimizer(tmp_path, field, value):
    options = ("num_iterations=26", "lr_warmup_steps=20", "save_every=26")
    with runtime(tmp_path / "base", overrides=options) as base:
        sft.main()
    path = Path(base.checkpoint_dir) / "meta_000026.json"
    meta = json.loads(path.read_text())
    if field == "legacy":
        meta["user_config"].pop("lr_warmup_steps")
    else:
        meta["user_config"][field] = value
    path.write_text(json.dumps(meta))
    with runtime(
        tmp_path / "restart",
        resume_dir=base.checkpoint_dir,
        overrides=(*options, "resume_from_step=26"),
    ) as restarted:
        with pytest.raises(ValueError, match="lr_warmup_steps|num_iterations"):
            sft.main()
    assert not restarted.optimizers and not restarted.rates


def test_epoch_prefetch_preserves_committed_schedule_and_state(tmp_path):
    options = ("num_iterations=-1", "lr_warmup_steps=3", "max_seq_len=256", "total_batch_size=1024")
    with runtime(tmp_path / "sync", overrides=(*options, "sft_prefetch_batches=0")) as synchronous:
        sft.main()
    with runtime(
        tmp_path / "prefetch", overrides=(*options, "sft_prefetch_batches=2")
    ) as prefetched:
        sft.main()
    assert synchronous.rates == prefetched.rates
    assert synchronous.positions == prefetched.positions
    assert synchronous.positions[0] == (False, 0.0)
    for step, (rates, (_, progress)) in enumerate(
        zip(synchronous.rates, synchronous.positions[:-1], strict=True)
    ):
        multiplier = sft_flat_decay(progress) * min(1.0, (step + 1) / 3)
        assert all(lr == initial * multiplier for lr, initial in rates)
    assert_exact(synchronous.models[0].state_dict(), prefetched.models[0].state_dict())
    assert_exact(synchronous.optimizers[0].state_dict(), prefetched.optimizers[0].state_dict())
    assert (
        synchronous.saves[-1]["loop_state"]["sft_dataloader_state"]
        == prefetched.saves[-1]["loop_state"]["sft_dataloader_state"]
    )


def test_actual_legacy_checkpoint_resumes_with_zero(tmp_path):
    options = ("num_iterations=6", "lr_warmup_steps=0")
    with runtime(tmp_path / "base", overrides=options) as full:
        sft.main()
    path = Path(full.checkpoint_dir) / "meta_000002.json"
    meta = json.loads(path.read_text())
    meta["user_config"].pop("lr_warmup_steps")
    path.write_text(json.dumps(meta))
    with runtime(
        tmp_path / "restart",
        resume_dir=full.checkpoint_dir,
        overrides=(*options, "resume_from_step=2"),
    ) as resumed:
        sft.main()
    assert resumed.rates == full.rates[2:]
    assert_exact(resumed.models[0].state_dict(), full.models[0].state_dict())
    assert_exact(resumed.optimizers[0].state_dict(), full.optimizers[0].state_dict())


def test_head_matrices_coalesced_with_trunk_receive_same_schedule(tmp_path):
    with runtime(
        tmp_path,
        head="transformer",
        overrides=("num_iterations=6", "lr_warmup_steps=3", "fp_head_matrix_lr=0.02"),
    ) as run:
        sft.main()
    head_ids = {id(p) for p in run.models[0].feature_pred_head.parameters()}
    assert any(
        {id(p) for p in group["params"]} & head_ids and {id(p) for p in group["params"]} - head_ids
        for group in run.optimizers[0].param_groups
        if group["kind"] == "muon"
    )
    for rates, multiplier in zip(run.rates, [1 / 3, 2 / 3, 1.0, 1.0, 1.0, 5 / 6], strict=True):
        assert all(lr == pytest.approx(base * multiplier) for lr, base in rates)


def test_real_validation_passes_do_not_advance_warmup(tmp_path):
    with runtime(
        tmp_path,
        eval_every=1,
        overrides=("num_iterations=6", "lr_warmup_steps=3", "eval_tokens=128"),
    ) as run:
        sft.main()
    assert run.events.count("validation") == 7  # Includes startup and terminal validation.
    assert len(run.rates) == 6
    for rates, multiplier in zip(run.rates, [1 / 3, 2 / 3, 1.0, 1.0, 1.0, 5 / 6], strict=True):
        assert all(lr == pytest.approx(base * multiplier) for lr, base in rates)
