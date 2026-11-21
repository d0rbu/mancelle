"""Lock the offload path itself in ``RuntimeState.offload_models``.

The companion test file ``test_runtime_state_offload_restore.py`` covers
the round-trip (offload then re-upload on next access). This file pins
the four behaviors of ``offload_models`` in isolation:

* When ``cfg.offload_between_stages`` is True and the decoder-only
  bundle is loaded, its model is moved to CPU.
* When ``cfg.offload_between_stages`` is True and the encdec bundle is
  loaded on CUDA (skipped without CUDA), its model is moved to CPU.
* When ``cfg.offload_between_stages`` is False, the call is a no-op
  even if bundles are loaded.
* When ``cfg.offload_between_stages`` is True but no bundle has been
  loaded yet, the call must not raise.
"""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle

# Helpers duplicated from ``test_runtime_state_offload_restore.py`` rather
# than imported across test files (per project test-isolation convention).


class _StubModelConfig:
    """Mimic ``model.config.sequence_len`` access used by accessors."""

    sequence_len = 8


class _StubModel(torch.nn.Module):
    """Tiny ``nn.Module`` exposing one parameter and a ``.config``
    attribute. Forward is never invoked in this file."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))
        self.config = _StubModelConfig()


class _StubTokenizer:
    def get_vocab_size(self) -> int:
        return 4


def _make_cfg(
    *, device: str, offload_between_stages: bool = True,
) -> AxBenchBpbConfig:
    """Build a minimal config — only ``device`` and
    ``offload_between_stages`` are read by the code paths under test."""
    return AxBenchBpbConfig(
        source_run_dir="/tmp/source_run",
        output_root="/tmp/output_root",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        device=device,
        offload_between_stages=offload_between_stages,
    )


def _make_runtime_state(
    *, device: str, offload_between_stages: bool = True,
) -> RuntimeState:
    cfg = _make_cfg(
        device=device, offload_between_stages=offload_between_stages,
    )
    return RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)


def _make_dummy_bundle(target_name: str, device: str) -> TargetModelBundle:
    """Build a ``TargetModelBundle`` wrapping a stub model on ``device``.
    Only the model is inspected by ``offload_models``."""
    model = _StubModel().to(device)
    return TargetModelBundle(
        model=model,
        tokenizer=_StubTokenizer(),
        max_seq_len=8,
        target_name=target_name,
    )


# ---------- decoder_only / encdec offload paths ----------


def test_offload_moves_decoder_only_to_cpu() -> None:
    """CPU-only contract: ``offload_models`` must execute the
    ``self._decoder_only.model.to('cpu')`` call without raising. On a
    CPU-only device the ``to('cpu')`` is a no-op, so the parameter is
    still on CPU afterwards — but the code path itself ran."""
    rs = _make_runtime_state(device="cpu")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"

    rs.offload_models()

    assert rs._decoder_only is not None
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"


def test_offload_moves_encdec_to_cpu() -> None:
    """CPU-only contract for the encdec branch."""
    rs = _make_runtime_state(device="cpu")
    rs._encdec = _make_dummy_bundle("encdec", "cpu")
    assert next(rs._encdec.model.parameters()).device.type == "cpu"

    rs.offload_models()

    assert rs._encdec is not None
    assert next(rs._encdec.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_moves_decoder_only_to_cpu_from_cuda() -> None:
    """The real bug-locking variant: model starts on CUDA, after offload
    it must be on CPU. Asserts the ``model.to('cpu')`` call actually
    executed (not just the cfg gate)."""
    rs = _make_runtime_state(device="cuda:0")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cuda:0")
    assert next(rs._decoder_only.model.parameters()).device.type == "cuda"

    rs.offload_models()

    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_moves_encdec_to_cpu_from_cuda() -> None:
    """Bug-locking variant for the encdec branch."""
    rs = _make_runtime_state(device="cuda:0")
    rs._encdec = _make_dummy_bundle("encdec", "cuda:0")
    assert next(rs._encdec.model.parameters()).device.type == "cuda"

    rs.offload_models()

    assert next(rs._encdec.model.parameters()).device.type == "cpu"


def test_offload_moves_both_bundles_when_both_loaded() -> None:
    """Both bundles loaded simultaneously: each branch of the
    ``if self._decoder_only is not None`` / ``if self._encdec is not
    None`` pair must run, leaving both models on CPU."""
    rs = _make_runtime_state(device="cpu")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")
    rs._encdec = _make_dummy_bundle("encdec", "cpu")

    rs.offload_models()

    assert rs._decoder_only is not None
    assert rs._encdec is not None
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"
    assert next(rs._encdec.model.parameters()).device.type == "cpu"


# ---------- no-op branches ----------


def test_offload_no_op_when_disabled() -> None:
    """``cfg.offload_between_stages=False`` short-circuits the function
    before either bundle is touched. The bundles stay loaded and on
    their original device — no exception is raised."""
    rs = _make_runtime_state(device="cpu", offload_between_stages=False)
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")
    rs._encdec = _make_dummy_bundle("encdec", "cpu")

    rs.offload_models()

    # Bundles still attached, models still on cpu (where they started).
    assert rs._decoder_only is not None
    assert rs._encdec is not None
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"
    assert next(rs._encdec.model.parameters()).device.type == "cpu"


def test_offload_no_op_when_bundles_not_loaded() -> None:
    """When neither bundle has been lazy-loaded yet, ``offload_models``
    must not raise (no ``NoneType.model`` AttributeError) — the
    ``if ... is not None`` guards are load-bearing."""
    rs = _make_runtime_state(device="cpu", offload_between_stages=True)
    assert rs._decoder_only is None
    assert rs._encdec is None

    rs.offload_models()  # must not raise

    assert rs._decoder_only is None
    assert rs._encdec is None


def test_offload_no_op_when_only_decoder_only_loaded() -> None:
    """Only ``self._decoder_only`` is loaded; the ``self._encdec`` branch
    must skip cleanly without raising on the ``None`` model access."""
    rs = _make_runtime_state(device="cpu", offload_between_stages=True)
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")
    assert rs._encdec is None

    rs.offload_models()

    assert rs._decoder_only is not None
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu"
    assert rs._encdec is None


def test_offload_no_op_when_only_encdec_loaded() -> None:
    """Only ``self._encdec`` is loaded; the ``self._decoder_only`` branch
    must skip cleanly without raising on the ``None`` model access."""
    rs = _make_runtime_state(device="cpu", offload_between_stages=True)
    rs._encdec = _make_dummy_bundle("encdec", "cpu")
    assert rs._decoder_only is None

    rs.offload_models()

    assert rs._encdec is not None
    assert next(rs._encdec.model.parameters()).device.type == "cpu"
    assert rs._decoder_only is None
