"""After ``offload_models()``, the next ``decoder_only()`` / ``encdec()``
access must return a bundle whose model is on ``cfg.device``, not stuck
on CPU.

Without this guarantee, any forward pass after offload silently
CPU/CUDA-mismatches against device-resident input tensors and crashes
deep inside the model.

Locks the contract that the lazy accessors restore the cached bundle's
model to ``cfg.device`` on every access — even after a previous
``offload_models()`` call has pushed the model to CPU.
"""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import AxBenchBpbError


class _StubModelConfig:
    """Mimic the ``model.config.sequence_len`` access used by the accessors."""
    sequence_len = 8


class _StubModel(torch.nn.Module):
    """Tiny ``nn.Module`` whose only purpose is to expose a parameter
    (so ``next(model.parameters()).device`` is observable) and a
    ``.config.sequence_len`` attribute. The test never invokes its
    ``forward``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))
        self.config = _StubModelConfig()


class _StubTokenizer:
    def get_vocab_size(self) -> int:
        return 4


def _make_cfg(device: str) -> AxBenchBpbConfig:
    """Build a minimal config — only ``device`` and
    ``offload_between_stages`` are read by the code paths under test.
    Path validation is deferred to ``validate_bpb_config``, which we do
    not call here."""
    return AxBenchBpbConfig(
        source_run_dir="/tmp/source_run",
        output_root="/tmp/output_root",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        device=device,
        offload_between_stages=True,
    )


def _make_runtime_state(device: str) -> RuntimeState:
    cfg = _make_cfg(device)
    return RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)


def _make_dummy_bundle(target_name: str, device: str) -> TargetModelBundle:
    """Build a ``TargetModelBundle`` wrapping a stub model on the requested
    device. The bundle's model is the only piece inspected by the
    accessor's device-restoration logic."""
    model = _StubModel().to(device)
    return TargetModelBundle(
        model=model,
        tokenizer=_StubTokenizer(),
        max_seq_len=8,
        target_name=target_name,
    )


# ---------- decoder_only() contract ----------


def test_decoder_only_returns_to_device_after_offload_cpu() -> None:
    """CPU-only contract test: post-offload, the next ``decoder_only()``
    access must return a bundle whose model is on ``cfg.device``.

    On a CPU-only ``cfg.device``, this is trivially satisfied — but the
    contract still has to hold, and this test runs everywhere (no CUDA
    requirement)."""
    rs = _make_runtime_state(device="cpu")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")

    b1 = rs.decoder_only()
    assert next(b1.model.parameters()).device.type == "cpu"

    rs.offload_models()

    b2 = rs.decoder_only()
    assert next(b2.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decoder_only_uploads_back_to_cuda_after_offload() -> None:
    """The real bug-locking test: after ``offload_models()`` pushes the
    cached bundle's model to CPU, the next ``decoder_only()`` access must
    re-upload it to ``cfg.device`` (cuda) so the next forward pass does
    not CPU/CUDA-mismatch against device-resident input tensors."""
    rs = _make_runtime_state(device="cuda:0")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cuda:0")
    assert next(rs._decoder_only.model.parameters()).device.type == "cuda"

    rs.offload_models()
    assert next(rs._decoder_only.model.parameters()).device.type == "cpu", (
        "offload_models() did not move decoder_only.model to CPU"
    )

    b = rs.decoder_only()
    assert next(b.model.parameters()).device.type == "cuda", (
        "decoder_only() returned a bundle still on CPU after offload; "
        "the next forward pass would CPU/CUDA-mismatch"
    )


# ---------- encdec() contract ----------


def test_encdec_returns_to_device_after_offload_cpu() -> None:
    """CPU-only contract test for the encdec accessor."""
    rs = _make_runtime_state(device="cpu")
    rs._encdec = _make_dummy_bundle("encdec", "cpu")

    b1 = rs.encdec()
    assert next(b1.model.parameters()).device.type == "cpu"

    rs.offload_models()

    b2 = rs.encdec()
    assert next(b2.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_encdec_uploads_back_to_cuda_after_offload() -> None:
    """Bug-locking test for the encdec accessor."""
    rs = _make_runtime_state(device="cuda:0")
    rs._encdec = _make_dummy_bundle("encdec", "cuda:0")
    assert next(rs._encdec.model.parameters()).device.type == "cuda"

    rs.offload_models()
    assert next(rs._encdec.model.parameters()).device.type == "cpu", (
        "offload_models() did not move encdec.model to CPU"
    )

    b = rs.encdec()
    assert next(b.model.parameters()).device.type == "cuda", (
        "encdec() returned a bundle still on CPU after offload; "
        "the next forward pass would CPU/CUDA-mismatch"
    )


# ---------- release_base_gemma_and_sae() / release_target_models() (PR #739 Track F1b) ----------


def test_release_base_gemma_and_sae_preserves_extraction_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extraction-to-target transition drops only GPU-heavy resources."""

    rs = _make_runtime_state(device="cpu")
    tokenizer = object()
    rs._base_gemma = object()
    rs._base_sae = object()
    rs._base_gemma_tail_offloaded_after = 8
    rs._tokenizer_extraction = tokenizer
    releases: list[str] = []
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.release_cuda_iteration_memory",
        lambda: releases.append("released"),
    )

    rs.release_base_gemma_and_sae()

    assert rs._base_gemma is None
    assert rs._base_sae is None
    assert rs._base_gemma_tail_offloaded_after is None
    assert rs._tokenizer_extraction is tokenizer
    assert releases == ["released"]

    # Idempotence matters when a resumed stage has no resident extractor.
    rs.release_base_gemma_and_sae()
    assert releases == ["released"]


@pytest.mark.parametrize(
    ("loaded_tail_after", "requested_tail_after"),
    [(None, 8), (8, None)],
)
def test_cached_base_gemma_rejects_tail_residency_contract_switch(
    loaded_tail_after: int | None,
    requested_tail_after: int | None,
) -> None:
    """A mixed-device extractor is reusable only under its load contract."""

    rs = _make_runtime_state(device="cpu")
    model = object()
    sae = object()
    rs._base_gemma = model
    rs._base_sae = sae
    rs._base_gemma_tail_offloaded_after = loaded_tail_after

    assert rs.base_gemma_and_sae(
        offload_transformer_tail_after=loaded_tail_after
    ) == (model, sae)
    with pytest.raises(
        AxBenchBpbError,
        match="cached extraction model tail-residency contract mismatch",
    ):
        rs.base_gemma_and_sae(
            offload_transformer_tail_after=requested_tail_after
        )


def test_release_target_models_clears_all_checkpoint_derived_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clustered extraction handoff is a true unload, not CPU offload."""

    rs = _make_runtime_state(device="cpu")
    rs._decoder_only = _make_dummy_bundle("decoder_only", "cpu")
    rs._encdec = _make_dummy_bundle("encdec", "cpu")
    rs._inactive_sampler = lambda **_kwargs: {1: 1}
    rs._inactive_sampler_loaded = True
    releases: list[str] = []
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.release_cuda_iteration_memory",
        lambda: releases.append("released"),
    )

    rs.release_target_models()

    assert rs._decoder_only is None
    assert rs._encdec is None
    assert rs._inactive_sampler is None
    assert rs._inactive_sampler_loaded is False
    assert releases == ["released"]
