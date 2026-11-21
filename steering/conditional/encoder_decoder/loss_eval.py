"""Conditioned BPB evaluation helpers for encoder-decoder training/eval."""

import math
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from loguru import logger

from steering.conditional.encoder_decoder.dataloader import EncDecTrainBatch


def accumulate_bpb_nats_bytes(
    loss_flat: torch.Tensor,
    targets_flat: torch.Tensor,
    token_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch's (nats, bytes) contribution under the shared BPB contract.

    This is THE definition both `val/bpb` (real-conditioned) and
    `val/bpb_selfcond` accumulate with — extracted so the two metrics cannot
    drift apart and stop being comparable on the same plot (review finding).
    Semantics: `ignore_index` targets (< 0) and zero-byte tokens contribute
    neither nats nor bytes; the negative-free fast path is byte-identical to
    the masked path for valid targets.
    """
    if (targets_flat.int() < 0).any():
        valid = targets_flat >= 0
        safe_targets = torch.where(valid, targets_flat, torch.zeros_like(targets_flat))
        num_bytes = torch.where(
            valid,
            token_bytes[safe_targets],
            torch.zeros_like(targets_flat, dtype=token_bytes.dtype),
        )
    else:
        num_bytes = token_bytes[targets_flat]
    byte_mask = num_bytes > 0
    return (loss_flat * byte_mask).sum(), num_bytes.sum()


def reduce_bpb(nats: torch.Tensor, num_bytes: torch.Tensor) -> float:
    """All-reduce accumulated (nats, bytes) across ranks and convert to BPB.

    Zero total bytes yields ``inf`` (the shared degenerate-input convention).
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_bytes, op=dist.ReduceOp.SUM)
    nats_value = float(nats.item())
    bytes_value = int(num_bytes.item())
    if bytes_value <= 0:
        return float("inf")
    return nats_value / (math.log(2) * bytes_value)


@dataclass
class ConditionedBpbMetrics:
    """Return type for evaluate_encdec_bpb_conditioned."""

    bpb_conditioned: float
    features_per_segment: list[int]
    avg_features_per_segment: float
    avg_features_per_batch: float
    magnitude_buckets: list[int]


@torch.no_grad()
def evaluate_encdec_bpb_conditioned(model, val_loader, eval_steps, token_bytes, device):
    """Evaluate encoder-decoder BPB with real encoder features only."""

    t0 = time.perf_counter()
    cond_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    cond_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    features_per_segment = []
    magnitude_buckets: list[int] = []
    total_features = 0
    total_tokens = 0
    num_batches = 0

    for _ in range(eval_steps):
        b = EncDecTrainBatch.from_tuple(next(val_loader))
        vx = b.inputs
        vy = b.targets
        v_enc_fids = b.enc_fids
        v_enc_mags = b.enc_mags
        v_enc_mask = b.enc_mask
        v_seg_map = b.seg_map
        v_seg_doc_id = b.seg_doc_id

        total_tokens += vx.numel()

        real_seg_mask = v_seg_doc_id >= 0
        per_seg_counts = v_enc_mask.sum(dim=-1)
        real_counts = per_seg_counts[real_seg_mask]
        features_per_segment.extend(real_counts.cpu().tolist())
        total_features += v_enc_mask.sum().item()
        num_batches += 1

        real_feature_mask = v_enc_mask & real_seg_mask.unsqueeze(-1)
        active_bucket_values = v_enc_mags[real_feature_mask].view(-1).cpu().tolist()
        magnitude_buckets.extend(active_bucket_values)

        loss_cond = model(
            vx, vy, v_enc_fids, v_enc_mags, v_enc_mask, v_seg_map, loss_reduction="none"
        )
        loss_cond = loss_cond.view(-1)

        batch_nats, batch_bytes = accumulate_bpb_nats_bytes(loss_cond, vy.view(-1), token_bytes)
        cond_nats += batch_nats
        cond_bytes += batch_bytes

    bpb_cond = reduce_bpb(cond_nats, cond_bytes)

    avg_features_per_segment = (
        sum(features_per_segment) / len(features_per_segment) if features_per_segment else 0.0
    )
    avg_features_per_batch = total_features / num_batches if num_batches > 0 else 0.0

    elapsed = time.perf_counter() - t0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_tokens = total_tokens * world_size
    tokens_per_sec = all_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"evaluate_encdec_bpb_conditioned: {elapsed:.1f}s | "
        f"{all_tokens} tokens | {tokens_per_sec:.0f} tok/sec"
    )

    return ConditionedBpbMetrics(
        bpb_conditioned=bpb_cond,
        features_per_segment=features_per_segment,
        avg_features_per_segment=avg_features_per_segment,
        avg_features_per_batch=avg_features_per_batch,
        magnitude_buckets=[int(b) for b in magnitude_buckets],
    )
