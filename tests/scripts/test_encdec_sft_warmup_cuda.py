"""Compiled BF16/attention SFT updates with the real mixed optimizer on CUDA."""

from __future__ import annotations

import math

import pytest
import torch as t

from scripts.train import conditional_encdec_sft as sft
from steering._common.lr_schedules import sft_flat_decay
from tests.scripts.test_encdec_sft_warmup import assert_exact, runtime

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def deterministic_cuda(monkeypatch):
    """Make the complete-state parity oracle use matched deterministic execution."""
    enabled = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        t.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_compiled_bf16_attention_warmup_decay_and_disabled_parity(tmp_path):
    backend = "fa4" if t.cuda.get_device_capability()[0] >= 9 else "sdpa"
    options = (
        "num_iterations=6",
        "save_every=6",
        "max_seq_len=128",
        "total_batch_size=256",
        "compile_static=auto",
    )
    with runtime(
        tmp_path / "warm",
        head="transformer",
        device="cuda",
        attention_backend=backend,
        overrides=(*options, "lr_warmup_steps=2"),
    ) as warm:
        initial, _, _ = sft.load_checkpoint("base", None, "cuda", load_optimizer=False)
        sft.main()
    assert len(warm.rates) == 6
    assert hasattr(warm.models[0], "_orig_mod")
    assert not t.equal(warm.models[0].transformer.wte.weight, initial["transformer.wte.weight"])
    assert all(
        math.isfinite(warm.saves[-1]["loop_state"][key])
        for key in ("smooth_train_loss", "smooth_fp_loss")
    )
    for step, multiplier in [(0, 0.5), (1, 1.0), (2, 1.0), (5, 5 / 6)]:
        assert all(lr == pytest.approx(base * multiplier) for lr, base in warm.rates[step])
    assert warm.autocast_modes == [(True, t.bfloat16)] * 6
    assert warm.saves[-1]["loop_state"]["smooth_fp_loss"] > 0
    assert all(t.isfinite(p).all() for p in warm.models[0].parameters())
    with runtime(
        tmp_path / "zero",
        head="transformer",
        device="cuda",
        attention_backend=backend,
        overrides=(*options, "lr_warmup_steps=0"),
    ) as zero:
        sft.main()
    with (
        runtime(
            tmp_path / "legacy",
            head="transformer",
            device="cuda",
            attention_backend=backend,
            overrides=(*options, "lr_warmup_steps=0"),
        ) as legacy,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(sft, "sft_warmup_flat_decay", lambda progress, **_: sft_flat_decay(progress))
        sft.main()
    assert zero.rates == legacy.rates
    assert_exact(zero.models[0].state_dict(), legacy.models[0].state_dict())
    assert_exact(zero.optimizers[0].state_dict(), legacy.optimizers[0].state_dict())
