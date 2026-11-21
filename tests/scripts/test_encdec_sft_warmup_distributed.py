"""Two real Gloo ranks exercise the trainer's synchronized warmup boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch as t
import torch.distributed as dist

from scripts.train import conditional_encdec_sft as sft
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.encoder import FeatureEncoder
from tests.padded_encoder_reference import (
    cross_attention_forward_with_cpu_reference,
    feature_encoder_forward_packed_with_cpu_reference,
    feature_encoder_forward_with_cpu_reference,
)
from tests.scripts.test_encdec_sft_warmup import Conversations, runtime


def _worker(rank, world_size, warmup):
    t.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as mp:
        mp.setattr(FeatureEncoder, "forward", feature_encoder_forward_with_cpu_reference)
        mp.setattr(
            FeatureEncoder, "forward_packed", feature_encoder_forward_packed_with_cpu_reference
        )
        mp.setattr(CrossAttentionLayer, "forward", cross_attention_forward_with_cpu_reference)
        # Gloo lacks the Work.get_future API used by DistMuonAdamW's ZeRO-2
        # transport. Keep real local mixed-optimizer math and average gradients
        # with Gloo; this test qualifies the trainer's schedule collectives.
        mp.setattr(model_module, "get_dist_info", lambda: (False, 0, 0, 1))
        with runtime(
            Path(directory),
            task=Conversations((12, 160) * 100),
            overrides=(
                "num_iterations=-1",
                f"lr_warmup_steps={warmup}",
                "max_seq_len=256",
                "total_batch_size=1024",
            ),
        ) as run:
            mp.setattr(sft, "compute_init", lambda _: (True, rank, rank, world_size, "cpu"))
            mp.setattr(sft, "compute_cleanup", lambda: None)
            setup = sft._setup_sft_optimizer

            def averaged_optimizer(model, cfg):
                optimizer = setup(model, cfg)

                def average(_optimizer, _args, _kwargs):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                            parameter.grad.div_(world_size)

                optimizer.register_step_pre_hook(average)
                return optimizer

            mp.setattr(sft, "_setup_sft_optimizer", averaged_optimizer)
            error = None
            try:
                sft.main()
            except ValueError as exc:
                assert "epoch LR warmup must finish strictly before 80%" in str(exc)
                error = str(exc)
            return {
                "rates": run.rates,
                "positions": run.positions,
                "local": run.local_progress,
                "error": error,
                "last_event": run.events[-1],
            }


@pytest.mark.parametrize("warmup", [1, 20])
def test_unequal_rank_progress_schedule_and_coherent_rejection(gloo_multiprocess, warmup):
    left, right = gloo_multiprocess(2, _worker, warmup)
    assert left["local"] != right["local"]
    assert left["rates"] == right["rates"]
    assert left["positions"] == right["positions"]
    assert left["error"] == right["error"]
    if warmup == 20:
        assert left["error"] is not None
        assert left["last_event"] == right["last_event"] == "boundary"
    else:
        assert left["error"] is None
        assert left["positions"][-1][0]
