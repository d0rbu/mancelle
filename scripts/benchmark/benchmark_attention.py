"""
Benchmark attention backends (sdpa, fa3, fa4, flex) with a short training loop.

Runs 200 steps of a small decoder model and logs loss + tokens/sec. The JSON
records the route actually taken by non-cached decoder attention and the route
that cached generation would take; this is intentionally distinct from the
configured backend label.
Saves results to benchmark_{impl}.json.

Usage:
    torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=sdpa
    torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=fa3
    torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=fa4
    torchrun --standalone --nproc_per_node=1 -m scripts.benchmark.benchmark_attention -- impl=flex
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from loguru import logger

import nanochat.flash_attention as fa_module
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_peak_flops,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.dataset import DatasetSplit
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from steering._common.build_meta import resolve_gpt_dimensions

NUM_STEPS = 200
BENCH_DEPTH = 4
BENCH_SEQ_LEN = 512
BENCH_BATCH_SIZE = 4
VALID_IMPLS = ("auto", "sdpa", "fa3", "fa4", "flex")


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    """Split the attention implementation override from Hydra overrides."""
    all_overrides = [a for a in argv if "=" in a and not a.startswith("-")]
    impl = "auto"
    hydra_overrides = []
    for override in all_overrides:
        if override.startswith("impl="):
            impl = override.split("=", 1)[1]
        else:
            hydra_overrides.append(override)
    if impl not in VALID_IMPLS:
        raise ValueError(f"impl must be one of {VALID_IMPLS}, got {impl!r}")
    return impl, hydra_overrides


def _select_backend(impl: str) -> None:
    """Apply the requested flash-attention backend override."""
    if impl in {"fa4", "flex"}:
        if not fa_module.HAS_FLEX_ATTENTION:
            raise RuntimeError(f"impl={impl} requested but FlexAttention is not available")
        if impl == "fa4" and not fa_module.HAS_FA4_BACKEND:
            raise RuntimeError("impl=fa4 requested but the FA4 FLASH backend is not available")
        fa_module._override_impl = FlashAttnImpl.FA4
    elif impl == "fa3":
        if not fa_module.HAS_FA3:
            raise RuntimeError("impl=fa3 requested but FA3 is not available")
        fa_module._override_impl = FlashAttnImpl.FA3
    elif impl == "sdpa":
        fa_module._override_impl = FlashAttnImpl.SDPA
    else:
        fa_module._override_impl = None


def _backend_routes() -> dict[str, str]:
    """Return the actual non-cached and KV-cache dispatch routes.

    ``_active_impl`` is a configured/preferred label. The public non-cached
    function independently evaluates ``_use_flex`` and ``_use_fa3``, while the
    KV-cache function has no FlexAttention path at all. Benchmark artifacts must
    record those executed branches rather than infer them from one label.
    """

    if fa_module._use_flex():
        non_cached = "flex_flash" if fa_module.HAS_FA4_BACKEND else "flex_triton"
    elif fa_module._use_fa3():
        non_cached = "fa3"
    else:
        non_cached = "sdpa"
    kv_cache = (
        "fa3"
        if fa_module.HAS_FA3 and fa_module._override_impl != FlashAttnImpl.SDPA
        else "sdpa"
    )
    return {
        "configured": str(fa_module._active_impl()),
        "non_cached": non_cached,
        "kv_cache": kv_cache,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the attention-backend benchmark."""
    impl, hydra_overrides = _parse_args(sys.argv[1:] if argv is None else argv)
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "training")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base_train", overrides=hydra_overrides)

    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    del ddp
    master_process = ddp_rank == 0
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    try:
        if device_type == "cuda":
            gpu_device_name = torch.cuda.get_device_name(0)
            gpu_peak_flops = get_peak_flops(gpu_device_name)
            logger.info(
                f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}"
            )
        else:
            gpu_device_name = device_type

        _select_backend(impl)
        routes = _backend_routes()
        logger.info(
            "Benchmark attention routes: "
            f"requested={impl}, configured={routes['configured']}, "
            f"non_cached={routes['non_cached']}, kv_cache={routes['kv_cache']}"
        )

        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()

        model_dim, num_heads = resolve_gpt_dimensions(
            depth=BENCH_DEPTH,
            aspect_ratio=cfg.aspect_ratio,
            head_dim=cfg.head_dim,
        )

        model_config = GPTConfig(
            sequence_len=BENCH_SEQ_LEN,
            vocab_size=vocab_size,
            n_layer=BENCH_DEPTH,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern="L",
        )

        with torch.device("meta"):
            model = GPT(model_config)
        model.to_empty(device=device)
        model.init_weights()

        if master_process:
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Benchmark model: depth={BENCH_DEPTH}, dim={model_dim}, "
                f"heads={num_heads}, params={num_params:,}"
            )

        compiled_model = torch.compile(model, dynamic=False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=(device_type == "cuda")
        )

        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            BENCH_BATCH_SIZE,
            BENCH_SEQ_LEN,
            split=DatasetSplit.TRAIN,
            device=device,
        )
        x, y, _ = next(train_loader)

        tokens_per_step = BENCH_BATCH_SIZE * BENCH_SEQ_LEN * ddp_world_size
        losses = []
        tok_per_sec_list = []

        logger.info(
            f"Starting benchmark: {NUM_STEPS} steps, {tokens_per_step} tokens/step"
        )

        model.train()
        for step in range(NUM_STEPS):
            synchronize()
            t0 = time.perf_counter()

            with autocast_ctx:
                loss = compiled_model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            x, y, _ = next(train_loader)

            synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            tps = tokens_per_step / dt
            loss_val = loss.item()
            losses.append(loss_val)
            tok_per_sec_list.append(tps)

            if master_process and step % 20 == 0:
                logger.info(
                    f"step {step:4d}/{NUM_STEPS} | loss {loss_val:.4f} | "
                    f"tok/sec {tps:,.0f} | dt {dt * 1000:.1f}ms"
                )

        if master_process:
            avg_tps = sum(tok_per_sec_list[10:]) / len(tok_per_sec_list[10:])
            final_loss = losses[-1]
            logger.info(
                f"Benchmark done: route={routes['non_cached']}, "
                f"avg tok/sec={avg_tps:,.0f}, "
                f"final loss={final_loss:.4f}"
            )

            results = {
                "impl": routes["non_cached"],
                "requested_impl": impl,
                "attention_routes": routes,
                "num_steps": NUM_STEPS,
                "tokens_per_step": tokens_per_step,
                "avg_tok_per_sec": avg_tps,
                "final_loss": final_loss,
                "losses": losses,
                "tok_per_sec": tok_per_sec_list,
                "model": {
                    "depth": BENCH_DEPTH,
                    "dim": model_dim,
                    "heads": num_heads,
                    "seq_len": BENCH_SEQ_LEN,
                    "batch_size": BENCH_BATCH_SIZE,
                },
                "gpu": gpu_device_name,
            }
            out_path = f"benchmark_{impl}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {out_path}")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
