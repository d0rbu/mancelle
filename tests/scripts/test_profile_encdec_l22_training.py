from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.benchmark.profile_encdec_l22_training as training_profile
from scripts.benchmark.profile_encdec_l22_training import (
    _COMPONENT_NAMES,
    _aggregate_component_diagnostics,
    _aggregate_optimizer_diagnostics,
    _build_topology_meta_model,
    _CudaModulePhaseRecorder,
    _distributed_train_step,
    _forward_loss,
    _optimizer_group_identity,
    _profile_compiled_operators,
    _resolve_topology_model_geometry,
    _summarize_device_operators,
    _synchronize_before_step,
    _synthetic_production_batch,
    _validation_forward,
    collective_inventory,
    distributed_grad_accum_steps,
    hierarchical_ring_ms,
    profile_distributed,
)
from steering._common.compile_helpers import resolve_compile_dynamic
from steering.conditional.encoder_decoder.feature_prediction import (
    segment_lengths,
    source_segment_masks,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

PLANNED_3B_MODEL = {
    "decoder_depth": 27,
    "encoder_depth": 27,
    "width": 1664,
    "head_dim": 128,
    "sequence_length": 2048,
    "window_pattern": "SSSL",
}
PROFILING_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiling"


@pytest.fixture
def distributed_profile_cpu(monkeypatch):
    """Run real configuration/model construction and reporting without CUDA or NCCL."""
    with initialize_config_dir(config_dir=str(PROFILING_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="encdec_l22_training",
            overrides=[
                "distributed.model=${topology.model}",
                "topology.model.decoder_depth=1",
                "topology.model.encoder_depth=1",
                "topology.model.width=16",
                "topology.model.head_dim=8",
                "topology.model.sequence_length=8",
                "topology.model.feature_vocab_size=16",
                "topology.model.feature_prediction_block=transformer",
                "topology.model.feature_prediction_head_layers=3",
                "distributed.device_batch_size=2",
                "distributed.total_batch_size=32",
                "distributed.segment_size=4",
                "distributed.padded_segments=3",
                "distributed.padded_features=24",
                "distributed.warmup_steps=1",
                "distributed.steps=2",
                "distributed.compile_static=auto",
            ],
        )
    raw_model = _build_topology_meta_model(cfg.distributed.model)
    harness = SimpleNamespace(cfg=cfg, raw=raw_model, world_size=1, rank=0, order=[])
    harness.optimizer = object()
    harness.compiled = object()
    harness.cleanup = Mock()
    harness.barrier = Mock()
    harness.inventory = Mock(return_value=[])
    harness.build = Mock(return_value=raw_model)
    harness.prewarm = Mock(side_effect=lambda *_args: harness.order.append("prewarm"))
    harness.setup = Mock(
        side_effect=lambda **_kwargs: (harness.order.append("optimizer"), harness.optimizer)[1]
    )
    harness.compile = Mock(
        side_effect=lambda *_args, **_kwargs: (harness.order.append("compile"), harness.compiled)[1]
    )
    harness.batch = Mock(wraps=_synthetic_production_batch)
    harness.steps = []

    def train_step(model, raw, optimizer, batch, **kwargs):
        assert raw is raw_model and optimizer is harness.optimizer
        assert batch[0].device.type == "cpu"
        harness.steps.append((model, kwargs))
        count = float(len(harness.steps))
        return {
            "forward_ms": 10 + count,
            "backward_ms": 20 + count,
            "optimizer_ms": 5 + count,
            "wall_ms": 40 + count,
            "last_microbatch_loss": count,
        }

    def gather(results, local):
        for rank in range(harness.world_size):
            results[rank] = deepcopy(local)
            results[rank]["rank"] = rank
            for sample in results[rank]["samples"]:
                for key in ("forward_ms", "backward_ms", "optimizer_ms", "wall_ms"):
                    sample[key] += rank * 10

    harness.gather = Mock(side_effect=gather)
    monkeypatch.setattr(
        training_profile,
        "compute_init",
        lambda *_args: (
            harness.world_size > 1,
            harness.rank,
            harness.rank,
            harness.world_size,
            t.device("cpu"),
        ),
    )
    for name, value in {
        "compute_cleanup": harness.cleanup,
        "_build_topology_meta_model": harness.build,
        "_synthetic_production_batch": harness.batch,
        "_distributed_train_step": train_step,
        "collective_inventory": harness.inventory,
        "_nccl_version": lambda: None,
        "get_peak_flops": lambda _name: 1e12,
    }.items():
        monkeypatch.setattr(training_profile, name, value)
    monkeypatch.setattr(raw_model, "prewarm_decoder_block_masks", harness.prewarm)
    monkeypatch.setattr(raw_model, "setup_optimizer", harness.setup)
    monkeypatch.setattr(t, "compile", harness.compile)
    monkeypatch.setattr(training_profile.dist, "barrier", harness.barrier)
    monkeypatch.setattr(training_profile.dist, "all_gather_object", harness.gather)
    for name, value in {
        "is_available": lambda: True,
        "synchronize": lambda: None,
        "reset_peak_memory_stats": lambda *_args: None,
        "max_memory_allocated": lambda *_args: 2**30,
        "max_memory_reserved": lambda *_args: 2 * 2**30,
        "get_device_name": lambda *_args: "CPU fixture",
    }.items():
        monkeypatch.setattr(t.cuda, name, value)
    return harness


@pytest.mark.parametrize(
    ("world_size", "rank", "packing", "compiled"),
    [
        (1, 0, False, False),
        (1, 0, True, True),
        (2, 0, False, True),
        (2, 1, True, True),
    ],
)
def test_full_profile_routes_standalone_and_ddp_and_preserves_samples(
    distributed_profile_cpu, world_size, rank, packing, compiled
):
    h = distributed_profile_cpu
    h.world_size, h.rank = world_size, rank
    h.cfg.distributed.total_batch_size *= world_size
    h.cfg.distributed.muon_parameter_packing = packing
    h.cfg.distributed.compile = compiled
    result = profile_distributed(h.cfg)

    h.setup.assert_called_once_with(muon_parameter_packing=packing)
    h.prewarm.assert_called_once_with(2, 8)
    assert h.order == (
        ["prewarm", "compile", "optimizer"] if compiled else ["prewarm", "optimizer"]
    )
    if compiled:
        h.compile.assert_called_once_with(h.raw, dynamic=None)
    assert len(h.steps) == 3
    assert all(model is (h.compiled if compiled else h.raw) for model, _ in h.steps)
    assert all(
        kwargs == {"grad_accum_steps": 2, "static_max_seg_size": 1024} for _, kwargs in h.steps
    )
    assert h.batch.call_args.kwargs["feature_vocab_size"] == 16
    assert h.batch.call_args.kwargs["rank"] == rank
    assert h.barrier.call_count == (3 if world_size > 1 else 0)
    assert h.gather.call_count == (1 if world_size > 1 else 0)
    h.cleanup.assert_called_once()
    if rank:
        assert result is None
        h.inventory.assert_not_called()
        return
    assert result["world_size"] == world_size
    assert result["muon_parameter_packing"] is packing
    assert result["optimizer_class"] == type(h.optimizer).__name__
    assert result["model_geometry"]["feature_prediction_head_layers"] == 3
    assert result["compiled"] is compiled and result["compile_dynamic"] is None
    assert result["critical_samples_ms"]["wall_ms"] == (
        [42.0, 43.0] if world_size == 1 else [52.0, 53.0]
    )
    assert result["rank_summaries"][0]["last_warmup_loss"] == 1.0
    assert result["rank_summaries"][0]["last_microbatch_losses"] == [2.0, 3.0]
    assert all(
        set(sample) == {"forward_ms", "backward_ms", "optimizer_ms", "wall_ms"}
        for sample in result["rank_summaries"][0]["samples_ms"]
    )
    assert result["validation"] is result["component_diagnostic"] is None
    assert h.inventory.call_count == (1 if world_size > 1 else 0)


@pytest.mark.parametrize("fail_component", [False, True])
def test_full_packed_profile_diagnostics_follow_samples_and_close_on_failure(
    distributed_profile_cpu, monkeypatch, tmp_path, fail_component
):
    h = distributed_profile_cpu
    h.cfg.distributed.muon_parameter_packing = True
    h.cfg.distributed.validation_steps = 1
    h.cfg.distributed.component_diagnostic_steps = 1
    h.cfg.distributed.trace_dir = str(tmp_path)
    h.cfg.distributed.fp8 = True
    validation = Mock(
        side_effect=lambda model, _batch: (
            not model.training or pytest.fail("validation must be eager eval"),
            4.0,
        )[1]
    )
    disable = Mock(side_effect=lambda _model: nullcontext())
    operators = Mock(return_value={"retained": True})
    recorder = Mock()
    component = Mock(
        side_effect=RuntimeError("injected component failure") if fail_component else None,
        return_value={
            "forward_ms": 100.0,
            "backward_ms": 200.0,
            "optimizer_ms": 50.0,
            "wall_ms": 350.0,
            "component_forward_ms": dict.fromkeys(_COMPONENT_NAMES),
            "unattributed_forward_ms": 100.0,
        },
    )
    convert = Mock(side_effect=lambda *_args, **_kwargs: (h.order.append("fp8"), (7, 2))[1])
    monkeypatch.setattr(training_profile, "_distributed_validation_step", validation)
    monkeypatch.setattr(training_profile, "disable_fp8", disable)
    monkeypatch.setattr(training_profile, "_profile_compiled_operators", operators)
    monkeypatch.setattr(training_profile, "_CudaModulePhaseRecorder", lambda _model: recorder)
    monkeypatch.setattr(training_profile, "_component_diagnostic_step", component)
    monkeypatch.setattr(training_profile, "apply_fp8_training", convert)

    if fail_component:
        with pytest.raises(RuntimeError, match="injected component failure"):
            profile_distributed(h.cfg)
    else:
        result = profile_distributed(h.cfg)
        assert result["critical_samples_ms"]["wall_ms"] == [42.0, 43.0]
        assert result["validation"]["critical_forward"]["median_ms"] == 4.0
        assert result["component_diagnostic"]["compiled_operator_profile_rank0"] == {
            "retained": True
        }
        assert result["component_diagnostic"]["forward"]["total"]["median_ms"] == 100.0
        assert result["software"]["fp8_converted_modules"] == 7
        assert result["software"]["fp8_skipped_modules"] == 2

    assert len(h.steps) == 5  # Three ordinary steps, then eager warmup/control.
    assert h.order == ["fp8", "prewarm", "compile", "optimizer"]
    assert all(model is h.raw for model, _ in h.steps[3:])
    assert validation.call_count == 2
    disable.assert_called_once_with(h.raw)
    assert h.raw.training
    assert operators.call_args.args[:3] == (h.compiled, h.raw, h.optimizer)
    assert operators.call_args.kwargs == {
        "row_limit": 20,
        "static_max_seg_size": 1024,
        "trace_dir": tmp_path,
        "rank": 0,
    }
    recorder.close.assert_called_once()
    h.cleanup.assert_called_once()


def test_full_ddp_serialized_diagnostic_keeps_accumulation_outside_timing(
    distributed_profile_cpu, monkeypatch
):
    h = distributed_profile_cpu
    h.world_size = 2
    h.cfg.distributed.total_batch_size *= 2
    h.cfg.distributed.optimizer_diagnostic_steps = 1
    h.optimizer = object.__new__(training_profile.DistMuonAdamW)
    parameter = t.nn.Parameter(t.tensor(4.0))
    forward = Mock(side_effect=lambda *_args, **_kwargs: parameter.square())

    def serialized(optimizer, raw_model):
        assert optimizer is h.optimizer and raw_model is h.raw
        assert parameter.grad is not None
        assert parameter.grad.item() == 8.0  # Two gradients, each divided by GA=2.
        return [
            {
                "kind": "muon",
                "reduce_ms": 1.0,
                "update_gather_ms": 2.0,
                "finish_copy_ms": 3.0,
                "total_ms": 6.0,
            }
        ]

    monkeypatch.setattr(training_profile, "_forward_loss", forward)
    monkeypatch.setattr(training_profile, "_serialized_optimizer_diagnostic_step", serialized)
    result = profile_distributed(h.cfg)

    assert forward.call_count == 2
    assert len(h.steps) == 3
    assert result["critical_samples_ms"]["wall_ms"] == [52.0, 53.0]
    assert result["serialized_optimizer_diagnostic"]["critical_total_samples_ms"] == [6.0]
    h.cleanup.assert_called_once()


@pytest.mark.parametrize(("reports", "save_json"), [(False, True), (True, False), (True, True)])
def test_profile_main_writes_same_result_only_on_reporting_rank(
    monkeypatch, tmp_path, capsys, reports, save_json
):
    result = {"world_size": 2, "value": None} if reports else None
    output = tmp_path / "nested" / "profile.json"
    cfg = OmegaConf.create({"mode": "distributed", "output_json": str(output) if save_json else None})
    monkeypatch.setattr(training_profile, "profile_distributed", lambda _cfg: result)

    training_profile.main.__wrapped__(cfg)

    printed = capsys.readouterr().out
    if reports:
        assert json.loads(printed) == result
        if save_json:
            assert output.read_text() == printed
        else:
            assert not output.exists()
    else:
        assert not output.exists() and not printed


def test_packed_serialized_diagnostic_is_rejected_before_distributed_startup(monkeypatch) -> None:
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)

    def forbidden_compute_init(*_args, **_kwargs):
        pytest.fail("unsupported diagnostic must fail before CUDA/distributed initialization")

    monkeypatch.setattr(training_profile, "compute_init", forbidden_compute_init)
    cfg = OmegaConf.create(
        {"distributed": {"muon_parameter_packing": True, "optimizer_diagnostic_steps": 1}}
    )
    with pytest.raises(ValueError, match="does not support serialized optimizer diagnostics"):
        profile_distributed(cfg)


def test_synthetic_batch_accepts_planned_context_and_conditioning_extent() -> None:
    batch = _synthetic_production_batch(
        device=t.device("cpu"),
        batch_size=2,
        rank=0,
        sequence_length=2048,
        padded_segments=26,
        padded_features=96,
        segment_size=128,
    )

    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = batch
    assert inputs.shape == targets.shape == (2, 2048)
    assert enc_fids.shape == enc_mags.shape == enc_mask.shape == (2, 26, 96)
    assert seg_map.shape == (2, 2048)
    assert seg_doc_id.shape == (2, 26)
    assert fp_targets.shape == (2, 26, 2074)
    assert enc_mask[:, :16, :24].all()
    assert not enc_mask[:, 16:].any()
    assert int(seg_map.max()) == 15
    assert t.equal(seg_doc_id[:, :16], t.arange(2).unsqueeze(1).expand(2, 16))
    assert (seg_doc_id[:, 16:] == -1).all()
    masks = source_segment_masks(segment_lengths(seg_map, 26), seg_doc_id)
    assert masks.valid_source_seg[:, :15].all()
    assert not masks.valid_source_seg[:, 15:].any()
    assert not masks.target_seg_valid[:, 16:].any()


def test_synthetic_batch_rejects_too_few_segment_slots() -> None:
    with pytest.raises(ValueError, match="padded_segments"):
        _synthetic_production_batch(
            device=t.device("cpu"),
            batch_size=1,
            rank=0,
            sequence_length=2048,
            padded_segments=15,
            padded_features=96,
            segment_size=128,
        )


@pytest.mark.parametrize("feature_vocab_size", [844, 2074])
def test_synthetic_model_and_batch_share_compact_feature_vocabulary(feature_vocab_size: int) -> None:
    model = _build_topology_meta_model({"feature_vocab_size": feature_vocab_size})
    batch = _synthetic_production_batch(
        device=t.device("cpu"),
        batch_size=2,
        rank=0,
        feature_vocab_size=feature_vocab_size,
    )

    assert model.feature_pred_head is not None
    assert model.feature_vocab is not None
    assert model.feature_pred_head.weight.shape[0] == feature_vocab_size
    assert len(model.feature_vocab.original_ids) == feature_vocab_size
    assert batch[2].min().item() >= 0
    assert batch[2].max().item() < feature_vocab_size
    assert batch[-1].shape == (2, 13, feature_vocab_size)


@pytest.mark.parametrize("feature_vocab_size", [0, -1, True, 65537])
def test_profile_geometry_rejects_invalid_feature_vocabulary(feature_vocab_size) -> None:
    with pytest.raises(ValueError, match="feature_vocab_size"):
        _resolve_topology_model_geometry({"feature_vocab_size": feature_vocab_size})


def test_validation_forward_matches_conditioned_bpb_call_contract() -> None:
    batch = _synthetic_production_batch(
        device=t.device("cpu"),
        batch_size=2,
        rank=0,
    )

    class ValidationModel(t.nn.Module):
        def forward(self, *args, **kwargs):
            assert len(args) == 6
            assert kwargs == {"loss_reduction": "none"}
            return t.ones_like(args[1], dtype=t.float32)

    loss = _validation_forward(ValidationModel(), batch)

    assert loss.shape == (2, 1024)
    assert not loss.requires_grad


@pytest.mark.parametrize("helper", [_forward_loss, _validation_forward])
def test_forward_helpers_leave_cpu_reference_arithmetic_in_float32(helper) -> None:
    batch = _synthetic_production_batch(device=t.device("cpu"), batch_size=1, rank=0)

    class Model(t.nn.Module):
        def forward(self, *_args, **_kwargs):
            assert not t.is_autocast_enabled("cpu")
            value = t.ones(2, 2) @ t.ones(2, 2)
            assert value.dtype == t.float32
            return SimpleNamespace(loss=value) if helper is _forward_loss else value

    with t.autocast("cpu", dtype=t.bfloat16):
        result = helper(Model(), batch)
    assert result.dtype == t.float32


@pytest.mark.parametrize("helper", [_forward_loss, _validation_forward])
def test_forward_helpers_scope_cuda_bfloat16_autocast_to_forward(helper, monkeypatch) -> None:
    # CUDA policy is observable without allocating GPU tensors. The real-device
    # numerical and timing arm uses the same helper with real CUDA inputs.
    batch = (SimpleNamespace(device=t.device("cuda")),) * 8
    autocast_active = False

    class Autocast:
        def __init__(self, *, device_type, dtype, enabled):
            assert device_type == "cuda"
            assert dtype == t.bfloat16
            assert enabled is True

        def __enter__(self):
            nonlocal autocast_active
            autocast_active = True

        def __exit__(self, *_args):
            nonlocal autocast_active
            autocast_active = False

    parameter = t.nn.Parameter(t.tensor(2.0))

    def model(*_args, **_kwargs):
        assert autocast_active
        value = parameter.square()
        return SimpleNamespace(loss=value) if helper is _forward_loss else value

    monkeypatch.setattr(t, "autocast", Autocast)
    result = helper(model, batch)
    assert not autocast_active
    if helper is _forward_loss:
        result.backward()
        assert parameter.grad is not None
        assert parameter.grad.item() == 4.0
    else:
        assert not result.requires_grad


def test_distributed_profile_restores_dynamo_cache_limit_after_failure(monkeypatch) -> None:
    def unavailable():
        assert t._dynamo.config.cache_size_limit == 32
        return False

    monkeypatch.setattr(t.cuda, "is_available", unavailable)
    with t._dynamo.config.patch(cache_size_limit=7):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            profile_distributed(None)
        assert t._dynamo.config.cache_size_limit == 7


@pytest.mark.parametrize("ddp", [False, True])
def test_step_prelude_synchronizes_after_optional_rank_barrier(ddp, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(training_profile.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(t.cuda, "synchronize", lambda: calls.append("synchronize"))

    _synchronize_before_step(ddp)

    assert calls == (["barrier", "synchronize"] if ddp else ["synchronize"])


@pytest.mark.parametrize("loss_value", [4.0, float("nan"), float("inf"), -float("inf")])
def test_training_step_checks_loss_receipt_after_wall_timer_stops(loss_value, monkeypatch) -> None:
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return float(clock_calls)

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 1.0

    class Loss:
        def __truediv__(self, _divisor):
            return self

        def backward(self):
            pass

        def detach(self):
            return self

        def item(self):
            assert clock_calls == 2, "loss receipt must follow the completed wall interval"
            return loss_value

    monkeypatch.setattr(training_profile.time, "perf_counter", clock)
    monkeypatch.setattr(t.cuda, "Event", Event)
    monkeypatch.setattr(t.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(training_profile, "_forward_loss", lambda *_args, **_kwargs: Loss())
    model = SimpleNamespace(zero_grad=lambda **_kwargs: None)
    optimizer = cast(t.optim.Optimizer, SimpleNamespace(step=lambda: None))

    if loss_value == 4.0:
        sample = _distributed_train_step(model, model, optimizer, (), grad_accum_steps=2)
        assert sample["wall_ms"] == 1000.0
        assert sample["last_microbatch_loss"] == 4.0
    else:
        with pytest.raises(RuntimeError, match="non-finite loss"):
            _distributed_train_step(model, model, optimizer, (), grad_accum_steps=2)


def test_optimizer_group_identity_accounts_for_muon_padding() -> None:
    group = {
        "kind": "muon",
        "params": [t.empty(2, 3), t.empty(2, 3)],
    }

    identity = _optimizer_group_identity(group, world_size=4)

    assert identity["parameter_tensors"] == 2
    assert identity["padded_parameter_tensors"] == 4
    assert identity["payload_padding_ratio"] == 2.0


def test_optimizer_diagnostic_aggregation_uses_rank_critical_samples() -> None:
    def sample(reduce_ms: float, update_ms: float, copy_ms: float) -> dict[str, object]:
        return {
            "group_index": 0,
            "kind": "muon",
            "parameter_tensors": 4,
            "parameter_numel": 24,
            "parameter_shapes": [(2, 3)],
            "padded_parameter_tensors": 4,
            "logical_payload_bytes": 96,
            "unpadded_payload_bytes": 96,
            "payload_padding_ratio": 1.0,
            "reduce_ms": reduce_ms,
            "update_gather_ms": update_ms,
            "finish_copy_ms": copy_ms,
            "total_ms": reduce_ms + update_ms + copy_ms,
        }

    rank_results = [
        {"optimizer_diagnostics": [[sample(1.0, 2.0, 3.0)], [sample(2.0, 3.0, 4.0)]]},
        {"optimizer_diagnostics": [[sample(4.0, 1.0, 2.0)], [sample(1.0, 5.0, 3.0)]]},
    ]

    result = _aggregate_optimizer_diagnostics(rank_results, diagnostic_steps=2)

    assert result is not None
    assert result["critical_total_samples_ms"] == [7.0, 9.0]
    group = result["groups"][0]
    assert group["critical_samples_ms"]["reduce_ms"] == [4.0, 2.0]
    assert group["critical_samples_ms"]["update_gather_ms"] == [2.0, 5.0]


def test_component_aggregation_preserves_critical_rank_additivity() -> None:
    def component_sample(
        forward_ms: float,
        backward_ms: float,
        cross_attention_forward_ms: float,
    ) -> dict[str, object]:
        component_forward = dict.fromkeys(_COMPONENT_NAMES, 0.0)
        component_forward["cross_attention"] = cross_attention_forward_ms
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": 2.0,
            "wall_ms": forward_ms + backward_ms + 2.0,
            "component_forward_ms": component_forward,
            "unattributed_forward_ms": forward_ms - cross_attention_forward_ms,
        }

    rank_results = [
        {
            "rank": 0,
            "compiled_operator_profile": {"rank": 0},
            "component_eager_controls": [
                {
                    "forward_ms": 9.0,
                    "backward_ms": 18.0,
                    "optimizer_ms": 2.0,
                    "wall_ms": 29.0,
                }
            ],
            "component_diagnostics": [component_sample(10.0, 20.0, 6.0)],
        },
        {
            "rank": 1,
            "compiled_operator_profile": {"rank": 1},
            "component_eager_controls": [
                {
                    "forward_ms": 8.0,
                    "backward_ms": 19.0,
                    "optimizer_ms": 2.0,
                    "wall_ms": 29.0,
                }
            ],
            # Rank 1 has the largest cross-attention interval, but rank 0 is the
            # forward critical path. Independent component maxima would destroy
            # the phase's additive interpretation.
            "component_diagnostics": [component_sample(9.0, 21.0, 8.0)],
        },
    ]

    result = _aggregate_component_diagnostics(rank_results, diagnostic_steps=1)

    assert result is not None
    assert result["forward"]["total"]["median_ms"] == 10.0
    assert result["forward"]["components"]["cross_attention"]["median_ms"] == 6.0
    assert result["forward"]["unattributed"]["median_ms"] == 4.0
    assert result["instrumented"]["backward"]["median_ms"] == 21.0
    assert result["compiled_operator_profile_rank0"] == {"rank": 0}


def test_component_hooks_report_direct_method_calls_as_unmeasured(monkeypatch) -> None:
    class Event:
        def record(self):
            pass

        def elapsed_time(self, _other):
            return 0.25

    model = SimpleNamespace(
        encoder=t.nn.Linear(1, 1),
        _decoder_blocks=[],
        _cross_attention_layer_values=list,
        lm_head=t.nn.Identity(),
        feature_pred_head=None,
    )
    monkeypatch.setattr(_CudaModulePhaseRecorder, "_event", staticmethod(Event))
    recorder = _CudaModulePhaseRecorder(cast(EncoderDecoderGPT, model))
    total = {"forward_ms": 1.0, "backward_ms": 2.0, "optimizer_ms": 3.0, "wall_ms": 6.0}

    def train_step(*_args, **_kwargs):
        model.encoder.forward(t.ones(1))
        model.lm_head(t.ones(1))
        return total

    step = Mock(side_effect=train_step)
    monkeypatch.setattr(training_profile, "_distributed_train_step", step)
    try:
        model.lm_head(t.ones(1))  # The diagnostic must reset earlier hook observations.
        optimizer = cast(t.optim.Optimizer, object())
        result = training_profile._component_diagnostic_step(
            cast(EncoderDecoderGPT, model), optimizer, (), recorder,
            grad_accum_steps=2, static_max_seg_size=64,
        )
        measured = result["component_forward_ms"]
        assert measured["encoder"] is None
        assert measured["cross_attention"] is None
        assert measured["lm_head"] == 0.25
        assert result["unattributed_forward_ms"] == 0.75
        assert all(result[key] == value for key, value in total.items())
        step.assert_called_once_with(
            model, model, optimizer, (), grad_accum_steps=2, static_max_seg_size=64,
        )
    finally:
        recorder.close()


@pytest.mark.parametrize("partially_observed", [False, True])
def test_component_aggregation_preserves_unmeasured_components(partially_observed: bool) -> None:
    control = {"forward_ms": 10.0, "backward_ms": 20.0, "optimizer_ms": 5.0, "wall_ms": 35.0}
    components: dict[str, float | None] = dict.fromkeys(_COMPONENT_NAMES, 1.0)
    components["encoder"] = None
    components["cross_attention"] = None
    sample = {
        **control,
        "component_forward_ms": components,
        "unattributed_forward_ms": 7.0,
    }
    second_sample = {
        **sample,
        "component_forward_ms": (
            {**components, "encoder": 0.5, "cross_attention": 0.5}
            if partially_observed
            else components
        ),
        "unattributed_forward_ms": 6.0 if partially_observed else 7.0,
    }
    rank_results = [
        {
            "rank": 0,
            "compiled_operator_profile": {},
            "component_eager_controls": [control, control],
            "component_diagnostics": [sample, second_sample],
        }
    ]

    result = _aggregate_component_diagnostics(rank_results, diagnostic_steps=2)
    assert result is not None
    for name in ("encoder", "cross_attention"):
        assert result["forward"]["components"][name] is None
    assert result["forward"]["components"]["lm_head"]["median_ms"] == 1.0
    assert result["forward"]["unattributed"]["median_ms"] == (6.5 if partially_observed else 7.0)
    assert result["rank_samples"][0]["instrumented"] == [sample, second_sample]


def test_operator_summary_deduplicates_pytorch_device_activities() -> None:
    class Event:
        def __init__(self, key: str, count: int, self_device_time_total: float):
            self.key = key
            self.count = count
            self.self_device_time_total = self_device_time_total

    class Profiler:
        def key_averages(self) -> list[Event]:
            return [
                Event("triton_fused", 12, 2500.0),
                Event("triton_fused", 12, 2500.0),
                Event("aten::mm", 24, 4000.0),
            ]

    result = _summarize_device_operators(Profiler(), row_limit=10)

    assert result == [
        {"name": "aten::mm", "calls": 24, "self_device_ms": 4.0},
        {"name": "triton_fused", "calls": 12, "self_device_ms": 2.5},
    ]


@pytest.mark.parametrize("export_traces", [False, True])
def test_operator_profile_retains_optimizer_evidence_and_all_phase_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    export_traces: bool,
) -> None:
    model = t.nn.Linear(1, 1, bias=False)
    with t.no_grad():
        model.weight.fill_(1.0)
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)
    phases = iter(("forward", "backward", "optimizer"))

    class Profiler:
        def __init__(self, **_kwargs):
            self.phase = next(phases)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def key_averages(self):
            return [SimpleNamespace(key=self.phase, count=1, self_device_time_total=1000.0)]

        def export_chrome_trace(self, path):
            Path(path).write_text(self.phase)

    monkeypatch.setattr(t.profiler, "profile", Profiler)
    monkeypatch.setattr(t.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        training_profile, "_forward_loss", lambda *_args, **_kwargs: model.weight.square().sum()
    )

    result = _profile_compiled_operators(
        model,
        cast(EncoderDecoderGPT, model),
        optimizer,
        (),
        row_limit=3,
        static_max_seg_size=32,
        trace_dir=tmp_path if export_traces else None,
        rank=2,
    )

    assert model.weight.item() == pytest.approx(0.8)
    assert model.weight.grad is None
    for phase in ("forward", "backward", "optimizer"):
        assert result[f"{phase}_top_operators"] == [
            {"name": phase, "calls": 1, "self_device_ms": 1.0}
        ]
        if export_traces:
            trace_path = Path(result["trace_paths"][phase])
            assert trace_path.name == f"{phase}_rank2.json"
            assert trace_path.read_text() == phase
    if not export_traces:
        assert result["trace_paths"] == {}
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("world_size", [8, 64])
def test_collective_inventory_matches_dist_muon_group_structure(world_size: int) -> None:
    messages = collective_inventory(world_size)

    assert Counter(message.kind for message in messages) == {
        "adam_allreduce": 12,
        "adam_reduce_scatter_gather": 8,
        "muon_reduce_scatter_gather": 6,
    }
    assert sum(message.physical_collectives for message in messages) == 40


def test_world_64_muon_padding_expands_exact_collective_payload() -> None:
    world_8 = collective_inventory(8)
    world_64 = collective_inventory(64)
    payload_8 = sum(message.payload_bytes for message in world_8)
    payload_64 = sum(message.payload_bytes for message in world_64)
    unpadded_8 = sum(message.unpadded_payload_bytes for message in world_8)
    unpadded_64 = sum(message.unpadded_payload_bytes for message in world_64)

    assert unpadded_8 == unpadded_64
    assert payload_64 / payload_8 == pytest.approx(1.7808423616)
    assert payload_64 / unpadded_64 == pytest.approx(1.8012288091)


def test_planned_3b_inventory_matches_exact_model_and_world_64_payload() -> None:
    with t.device("meta"):
        model = _build_topology_meta_model(PLANNED_3B_MODEL)
    messages = collective_inventory(64, PLANNED_3B_MODEL)

    assert sum(parameter.numel() for parameter in model.parameters()) == 3_020_935_888
    assert Counter(message.kind for message in messages) == {
        "adam_allreduce": 20,
        "adam_reduce_scatter_gather": 16,
        "muon_reduce_scatter_gather": 6,
    }
    assert sum(message.physical_collectives for message in messages) == 64
    assert sum(message.payload_bytes for message in messages) / 2**30 == pytest.approx(
        13.3581669927
    )
    assert sum(message.unpadded_payload_bytes for message in messages) / 2**30 == pytest.approx(
        11.2538631558
    )


def test_planned_3b_batch8_fallback_preserves_global_batch_at_world_64() -> None:
    assert (
        distributed_grad_accum_steps(
            1_048_576,
            device_batch_size=8,
            sequence_length=2048,
            world_size=64,
        )
        == 1
    )


def test_planned_3b_main_batch_contract_uses_one_microbatch_at_world_32() -> None:
    assert (
        distributed_grad_accum_steps(
            1_048_576,
            device_batch_size=16,
            sequence_length=2048,
            world_size=32,
        )
        == 1
    )


def test_planned_3b_batch16_exceeds_fixed_global_batch_at_world_64() -> None:
    with pytest.raises(ValueError, match="world microbatch"):
        distributed_grad_accum_steps(
            1_048_576,
            device_batch_size=16,
            sequence_length=2048,
            world_size=64,
        )


def test_planned_3b_profile_pins_main_run_batch16() -> None:
    with initialize_config_dir(config_dir=str(PROFILING_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="encdec_l22_training_3b_h200")

    assert cfg.distributed.device_batch_size == 16
    assert cfg.topology.device_batch_size == 16
    assert cfg.distributed.total_batch_size == 1_048_576
    assert cfg.topology.total_batch_size == 1_048_576


def test_topology_geometry_rejects_width_not_divisible_by_head_dim() -> None:
    invalid = {**PLANNED_3B_MODEL, "width": 1600}

    with pytest.raises(ValueError, match="must be divisible"):
        _resolve_topology_model_geometry(invalid)


@pytest.mark.parametrize("block", ["mlp", "transformer"])
@pytest.mark.parametrize("depth", [3, 10])
def test_profile_geometry_constructs_expanded_feature_head(block: str, depth: int) -> None:
    geometry = {
        **PLANNED_3B_MODEL,
        "feature_prediction_block": block,
        "feature_prediction_head_layers": depth,
    }
    model = _build_topology_meta_model(geometry)

    assert isinstance(model.feature_pred_head, ResidualFeaturePredictionHead)
    assert model.feature_pred_head.block.value == block
    assert len(model.feature_pred_head.residual_blocks) == depth
    assert model.feature_pred_head.weight.shape == (2074, 1664)
    assert model.feature_pred_config is not None
    assert model.feature_pred_config.head_layers == depth


@pytest.mark.parametrize("depth", [-1, True, 1.5])
def test_profile_geometry_rejects_invalid_feature_head_depth(depth: object) -> None:
    with pytest.raises(ValueError, match="head_layers"):
        _resolve_topology_model_geometry({"feature_prediction_head_layers": depth})


def test_local_profile_config_binds_feature_head_geometry() -> None:
    with initialize_config_dir(config_dir=str(PROFILING_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="encdec_l22_training",
            overrides=[
                "distributed.model=${topology.model}",
                "topology.model.feature_prediction_block=transformer",
                "topology.model.feature_prediction_head_layers=10",
                "distributed.compile_static=auto",
            ],
        )

    assert cfg.distributed.model.feature_prediction_block == "transformer"
    assert cfg.distributed.model.feature_prediction_head_layers == 10
    assert resolve_compile_dynamic(cfg.distributed.compile_static, mode="sft") is None
    assert _resolve_topology_model_geometry(None)["feature_prediction_head_layers"] == 0


def test_world_64_inventory_coalesces_compatible_muon_groups() -> None:
    messages = collective_inventory(64)

    assert Counter(message.kind for message in messages) == {
        "adam_allreduce": 12,
        "adam_reduce_scatter_gather": 8,
        "muon_reduce_scatter_gather": 6,
    }
    assert sum(message.physical_collectives for message in messages) == 40
    muon = [message for message in messages if message.kind.startswith("muon")]
    assert sum(message.payload_bytes for message in muon) == 2_113_978_368
    assert sum(message.unpadded_payload_bytes for message in muon) == 792_732_672


def test_hierarchical_ring_is_sensitive_to_cross_node_latency() -> None:
    common = {
        "payload_bytes": 128 * 2**20,
        "world_size": 64,
        "gpus_per_node": 8,
        "intra_latency_us": 2.0,
        "intra_bandwidth_gb_s": 450.0,
        "inter_bandwidth_gb_s": 50.0,
    }

    low_latency = hierarchical_ring_ms(inter_latency_us=5.0, **common)
    high_latency = hierarchical_ring_ms(inter_latency_us=100.0, **common)

    assert high_latency > low_latency
    assert high_latency - low_latency == pytest.approx(1.33)


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(2, 16), (4, 8)],
)
def test_distributed_profile_uses_training_batch_contract(
    world_size: int,
    expected: int,
) -> None:
    assert (
        distributed_grad_accum_steps(
            524_288,
            device_batch_size=16,
            sequence_length=1024,
            world_size=world_size,
        )
        == expected
    )


def test_distributed_profile_rejects_fractional_accumulation() -> None:
    with pytest.raises(ValueError, match="world microbatch"):
        distributed_grad_accum_steps(
            524_288,
            device_batch_size=16,
            sequence_length=1024,
            world_size=64,
        )
