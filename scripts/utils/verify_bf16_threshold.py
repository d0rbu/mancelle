"""Verify that GPU spot-check violations are caused by bf16 noise near JumpReLU threshold.

Loads a few shards, re-runs extraction, identifies features that differ between
stored and fresh, and checks how close their hidden_pre values are to the
per-feature JumpReLU threshold.

Usage:
    uv run python -m scripts.utils.verify_bf16_threshold
"""

from pathlib import Path

import pyarrow.parquet as pq
import torch as t
from loguru import logger

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.segment_mode_guard import require_fixed_segment_mode
from steering.extract_features import (
    PoolingMode,
    _precompute_batch_cpu_work,
    _truncate_to_training_window,
    extract_sparse_gpu,
    load_sae_and_model,
    pool_segments_batched_gpu,
)
from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    resolve_sae,
)
from steering.segmentation import (
    SegmentMode,
    segment_mode_from_metadata,
)
from steering.validate_features import _select_spot_check_docs

MODEL_NAME = "gemma-2-2b"
SAE_LAYER = 8
SAE_WIDTH = SAEWidth.W_16K
SEGMENT_SIZE = 128
FEATURE_DIR = Path(
    f"out/data/features/s{SEGMENT_SIZE}/l{SAE_LAYER}/"
    f"a_res/p_max_pool/f{SAE_WIDTH.value}/m_{MODEL_NAME}/base"
)
PARQUET_DIR = Path("nanochat_cache/base_data")
N_SHARDS = 5  # Check first N shards
N_DOCS = 3  # Docs per shard


def get_hidden_pre(sae, sae_in: t.Tensor) -> t.Tensor:
    """Compute hidden_pre = sae_in @ W_enc + b_enc (before JumpReLU thresholding)."""
    processed = sae.process_sae_in(sae_in)
    return processed @ sae.W_enc + sae.b_enc


def main():
    device = "cuda" if t.cuda.is_available() else "cpu"
    logger.info(f"Loading model + SAE on {device}")
    resolved = resolve_sae(
        model=MODEL_NAME,
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=SAE_LAYER,
        width=SAE_WIDTH,
    )
    model, sae, tokenizer = load_sae_and_model(
        model_name=MODEL_NAME,
        layer=SAE_LAYER,
        device=device,
        resolved=resolved,
    )
    model.to(t.bfloat16)
    sae.to(t.bfloat16)

    threshold = getattr(sae, "threshold", None)
    if not isinstance(threshold, t.Tensor):
        raise TypeError(
            "Loaded JumpReLU SAE must expose its threshold as a torch.Tensor"
        )
    threshold = threshold.detach()  # (n_features,)
    logger.info(
        f"JumpReLU threshold shape: {threshold.shape}, "
        f"mean={threshold.mean():.4f}, min={threshold.min():.4f}, max={threshold.max():.4f}"
    )

    hook_name = resolved.hook_name
    context_window = DEFAULT_CONTEXT_WINDOW
    segment_size = SEGMENT_SIZE

    shard_paths = glob_features(FEATURE_DIR)[:N_SHARDS]

    all_diffs = []  # (hidden_pre - threshold) for differing features
    all_shared_dists = []  # (hidden_pre - threshold) for shared (non-differing) active features
    total_differing = 0
    total_shared = 0

    for shard_path in shard_paths:
        stem = shard_stem_from_path(shard_path)
        pq_path = PARQUET_DIR / f"{stem}.parquet"
        if not pq_path.exists():
            continue

        data = load_features(str(shard_path))
        require_fixed_segment_mode(
            segment_mode_from_metadata(data, source="bf16 threshold feature shard"),
            surface="bf16 threshold verification",
            source="feature shard",
        )
        table = pq.read_table(str(pq_path), columns=["text"])
        texts = table.column("text").to_pylist()

        doc_token_counts = data["doc_token_counts"]
        n_docs = len(doc_token_counts)
        doc_ids_tensor = data["doc_ids"]
        offsets = data["offsets"]
        stored_feat_ids = data["feature_ids"]

        selected = _select_spot_check_docs(doc_token_counts, n_docs, N_DOCS)

        for doc_id in selected:
            text = texts[doc_id]

            # Truncate and tokenize
            kept_ids, n_tokens = _truncate_to_training_window(
                text,
                model.tokenizer,
                context_window,
                model.cfg.n_ctx,
            )
            if not kept_ids:
                continue

            kept_tensor = t.tensor(kept_ids, dtype=t.long, device=device)

            # Get residual stream activations
            with t.no_grad():
                _, cache = model.run_with_cache(
                    kept_tensor.unsqueeze(0),
                    names_filter=[hook_name],
                )
                act = cache[hook_name].squeeze(0)  # (seq, d_model)

            # Compute hidden_pre
            hidden_pre = get_hidden_pre(sae, act)  # (seq, n_features)

            # Get fresh SAE activations
            sae_acts = sae.encode(act)  # (seq, n_features)

            # Segment and pool
            batch = [(kept_ids, n_tokens)]
            precomputed = _precompute_batch_cpu_work(
                batch,
                model.tokenizer,
                SegmentMode.fixed_tokens,
                segment_size,
            )
            segments, _, _, _ = precomputed[0]

            # Pool SAE activations per segment
            pooled = pool_segments_batched_gpu(
                sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
            )
            fresh_ids_list, _ = extract_sparse_gpu(pooled)

            # Pool hidden_pre per segment (max pool to match)
            hidden_pre_pooled = pool_segments_batched_gpu(
                hidden_pre, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
            )

            # Get stored segments for this doc
            mask = doc_ids_tensor == doc_id
            seg_indices = mask.nonzero(as_tuple=True)[0]

            for seg_i in range(min(len(seg_indices), len(fresh_ids_list))):
                global_idx = seg_indices[seg_i].item()
                s = int(offsets[global_idx].item())
                e = int(offsets[global_idx + 1].item())
                stored_ids = stored_feat_ids[s:e]
                fresh_ids = fresh_ids_list[seg_i]

                stored_set = set(stored_ids.tolist())
                fresh_set = set(fresh_ids.tolist())

                # Features that differ
                only_stored = stored_set - fresh_set
                only_fresh = fresh_set - stored_set
                differing = only_stored | only_fresh
                shared = stored_set & fresh_set

                if not differing:
                    continue

                # Get hidden_pre values for differing features
                seg_hidden_pre = hidden_pre_pooled[seg_i]  # (n_features,)
                for fid in differing:
                    dist = (seg_hidden_pre[fid] - threshold[fid]).float().item()
                    all_diffs.append(dist)

                # Sample some shared features for comparison
                shared_sample = list(shared)[: len(differing) * 2]
                for fid in shared_sample:
                    dist = (seg_hidden_pre[fid] - threshold[fid]).float().item()
                    all_shared_dists.append(dist)

                total_differing += len(differing)
                total_shared += len(shared)

        del data
        logger.info(f"Processed {stem}: {total_differing} differing features so far")

    if not all_diffs:
        logger.info("No differing features found!")
        return

    diffs_tensor = t.tensor(all_diffs)
    shared_tensor = t.tensor(all_shared_dists) if all_shared_dists else t.tensor([0.0])

    abs_diffs = diffs_tensor.abs()
    abs_shared = shared_tensor.abs()

    logger.info(f"\n{'=' * 70}")
    logger.info(
        f"RESULTS: {total_differing} differing features, {total_shared} shared features"
    )
    logger.info(f"{'=' * 70}")
    logger.info("\nDiffering features — |hidden_pre - threshold|:")
    logger.info(f"  mean:   {abs_diffs.mean():.6f}")
    logger.info(f"  median: {abs_diffs.median():.6f}")
    logger.info(f"  p95:    {abs_diffs.quantile(0.95):.6f}")
    logger.info(f"  p99:    {abs_diffs.quantile(0.99):.6f}")
    logger.info(f"  max:    {abs_diffs.max():.6f}")

    logger.info("\nShared (non-differing) active features — |hidden_pre - threshold|:")
    logger.info(f"  mean:   {abs_shared.mean():.6f}")
    logger.info(f"  median: {abs_shared.median():.6f}")
    logger.info(f"  p95:    {abs_shared.quantile(0.95):.6f}")
    logger.info(f"  p99:    {abs_shared.quantile(0.99):.6f}")
    logger.info(f"  max:    {abs_shared.max():.6f}")

    # What fraction of differing features are within various distances of threshold
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        frac = (abs_diffs < eps).float().mean().item()
        logger.info(
            f"\n  Differing features within {eps} of threshold: {frac * 100:.1f}%"
        )

    ratio = (
        abs_shared.mean() / abs_diffs.mean() if abs_diffs.mean() > 0 else float("inf")
    )
    logger.info(f"\nRatio (shared/differing mean distance): {ratio:.1f}x")
    logger.info(
        "  (Higher = differing features are much closer to threshold than shared ones)"
    )


if __name__ == "__main__":
    main()
