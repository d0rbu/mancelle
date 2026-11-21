"""Profile the production l22 encoder-decoder training shape and data path.

This deliberately keeps four kinds of evidence separate:

``data``
    Multiple local CPU processes decode one real packed/pre-selected shard and
    reconstruct rank-striped training batches.  This measures local storage,
    decompression, and batch construction; it is not a GPU or network test.

``train``
    One CUDA process loads an exact encoder-decoder checkpoint and runs real
    packed/pre-selected rows through forward, backward, and optimizer step.

``distributed``
    One CUDA process or multiple torchrun processes run the configured model
    shape with synthetic tensors matching the padded training batch contract.
    Multi-process runs additionally measure NCCL collectives.

``topology``
    Inventory the exact collectives that ``DistMuonAdamW`` would issue for the
    configured model geometry and apply an ideal hierarchical-ring
    latency/bandwidth model.  These numbers are sensitivity estimates, not
    NCCL measurements.

The measured data and train modes remain intentionally specific to the audited
production model shape. Distributed mode defaults to that same 12-layer,
context-1,024 contract but accepts an explicit target geometry and synthetic
conditioning extent so the planned 3B shape can be measured without pretending
random weights or synthetic rows are real artifacts. Topology mode accepts the
same geometry for analytical inventory. Artifact paths remain explicit in the
real-data modes so results cannot silently bind to a different local cache.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import resource
import socket
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import hydra
import torch as t
import torch.distributed as dist
from omegaconf import DictConfig
from torch.cuda import nccl

import nanochat.flash_attention as flash_attention
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.common import compute_cleanup, compute_init, get_peak_flops
from nanochat.fp8_utils import apply_fp8_training, disable_fp8
from nanochat.gpt import GPTConfig
from nanochat.optim import DistMuonAdamW
from scripts.benchmark.bench_encoder_vocab_compaction import (
    N_SAE,
    REAL_V,
    SEQ_LEN,
    build_meta_model,
)
from steering._common.compile_helpers import resolve_compile_dynamic
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    validate_feature_prediction_head_layers,
)
from steering.conditional.encoder_decoder.dataloader import build_fp_targets_from_shard
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import resolve_feature_mask
from steering.conditional.feature_vocab import FeatureVocab
from steering.packed_batch_io import (
    _build_batch_index,
    load_packed_row_shard,
    load_selection_shard,
    reconstruct_batch,
)
from steering.refined_types import PositiveInt

PAD_N_SEG = 13
PAD_F = 96
MAX_SEG_SIZE = 1024
EFFECTIVE_BUCKETS = 1
DECODER_VOCAB_SIZE = 32_768
MODEL_NUM_BUCKETS = 2


@dataclass(frozen=True)
class CollectiveMessage:
    kind: str
    payload_bytes: int
    unpadded_payload_bytes: int
    physical_collectives: int


_COMPONENT_NAMES = (
    "encoder",
    "decoder_blocks",
    "cross_attention",
    "lm_head",
    "feature_prediction_head",
)


class _CudaModulePhaseRecorder:
    """Record non-overlapping top-level model forward CUDA intervals.

    This recorder is deliberately confined to the opt-in eager diagnostic.
    Backward is instead attributed by compiled operator profiles: PyTorch module
    hooks do not enclose parameter-gradient work for integer-input modules such
    as the feature encoder.
    """

    def __init__(self, model: EncoderDecoderGPT):
        self._forward_events: dict[str, list[tuple[t.cuda.Event, t.cuda.Event]]] = {
            name: [] for name in _COMPONENT_NAMES
        }
        self._forward_starts: dict[int, t.cuda.Event] = {}
        self._handles: list[Any] = []

        modules: list[tuple[str, t.nn.Module]] = [("encoder", model.encoder)]
        modules.extend(("decoder_blocks", block) for block in model._decoder_blocks)
        modules.extend(
            ("cross_attention", layer) for layer in model._cross_attention_layer_values()
        )
        modules.append(("lm_head", model.lm_head))
        if model.feature_pred_head is not None:
            modules.append(("feature_prediction_head", model.feature_pred_head))
        for name, module in modules:
            self._register(name, module)

    @staticmethod
    def _event() -> t.cuda.Event:
        return t.cuda.Event(enable_timing=True)

    def _register(self, name: str, module: t.nn.Module) -> None:
        module_id = id(module)

        def forward_pre_hook(_module: t.nn.Module, _inputs: tuple[Any, ...]) -> None:
            start = self._event()
            start.record()
            self._forward_starts[module_id] = start

        def forward_hook(
            _module: t.nn.Module,
            _inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            end = self._event()
            end.record()
            start = self._forward_starts.pop(module_id)
            self._forward_events[name].append((start, end))

        self._handles.extend(
            (
                module.register_forward_pre_hook(forward_pre_hook),
                module.register_forward_hook(forward_hook),
            )
        )

    def reset(self) -> None:
        if self._forward_starts:
            raise RuntimeError("cannot reset component recorder with an active module interval")
        for events in self._forward_events.values():
            events.clear()

    @staticmethod
    def _elapsed(events: list[tuple[t.cuda.Event, t.cuda.Event]]) -> float:
        return sum(start.elapsed_time(end) for start, end in events)

    def elapsed_ms(self) -> dict[str, float | None]:
        if self._forward_starts:
            raise RuntimeError("component recorder still has an active module interval")
        # Packed encoder/cross-attention paths call implementation methods
        # directly and can bypass Module.__call__. Missing hooks are unmeasured,
        # never evidence that the component consumed zero device time.
        return {
            name: self._elapsed(events) if events else None
            for name, events in self._forward_events.items()
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _resolve_topology_model_geometry(
    model_cfg: Mapping[str, Any] | None,
) -> dict[str, int | str]:
    """Resolve and validate the geometry shared by topology/distributed modes."""
    defaults: dict[str, int | str] = {
        "decoder_depth": 12,
        "encoder_depth": 12,
        "width": 768,
        "head_dim": 128,
        "sequence_length": SEQ_LEN,
        "feature_vocab_size": REAL_V,
        "window_pattern": "SSSL",
        "feature_prediction_block": "mlp",
        "feature_prediction_head_layers": 0,
    }
    resolved = {
        key: defaults[key] if model_cfg is None else model_cfg.get(key, defaults[key])
        for key in defaults
    }
    for key in (
        "decoder_depth",
        "encoder_depth",
        "width",
        "head_dim",
        "sequence_length",
        "feature_vocab_size",
    ):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"topology.model.{key} must be a positive integer, got {value!r}")
    if int(resolved["feature_vocab_size"]) > N_SAE:
        raise ValueError("topology.model.feature_vocab_size cannot exceed the original SAE width")
    width = int(resolved["width"])
    head_dim = int(resolved["head_dim"])
    if width % head_dim != 0:
        raise ValueError(
            "topology.model.width must be divisible by topology.model.head_dim: "
            f"{width} % {head_dim} != 0"
        )
    window_pattern = resolved["window_pattern"]
    if not isinstance(window_pattern, str) or not window_pattern:
        raise ValueError(
            "topology.model.window_pattern must be a non-empty string, "
            f"got {window_pattern!r}"
        )
    resolved["feature_prediction_block"] = FeaturePredictionBlock(
        resolved["feature_prediction_block"]
    ).value
    resolved["feature_prediction_head_layers"] = validate_feature_prediction_head_layers(
        resolved["feature_prediction_head_layers"]
    )
    return resolved


def _build_topology_meta_model(
    model_cfg: Mapping[str, Any] | None,
) -> EncoderDecoderGPT:
    """Build the configured inventory model without allocating real storage."""
    geometry = _resolve_topology_model_geometry(model_cfg)
    width = int(geometry["width"])
    head_dim = int(geometry["head_dim"])
    n_head = width // head_dim
    feature_vocab = FeatureVocab(
        n_sae_features=N_SAE,
        original_ids=tuple(range(int(geometry["feature_vocab_size"]))),
    )
    decoder = GPTConfig(
        sequence_len=int(geometry["sequence_length"]),
        vocab_size=DECODER_VOCAB_SIZE,
        n_layer=int(geometry["decoder_depth"]),
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=width,
        window_pattern=str(geometry["window_pattern"]),
    )
    encoder = EncoderConfig(
        n_embd=width,
        depth=int(geometry["encoder_depth"]),
        n_head=n_head,
        n_kv_head=n_head,
        head_dim=head_dim,
    )
    cross_attention = CrossAttentionConfig(
        frequency=1,
        n_head=n_head,
        n_kv_head=n_head,
    )
    feature_prediction = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
        block=FeaturePredictionBlock(geometry["feature_prediction_block"]),
        head_layers=int(geometry["feature_prediction_head_layers"]),
    )
    with t.device("meta"):
        return EncoderDecoderGPT(
            decoder,
            encoder,
            cross_attention,
            n_sae_features=N_SAE,
            num_buckets=MODEL_NUM_BUCKETS,
            feature_pred_config=feature_prediction,
            feature_vocab=feature_vocab,
        )


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary_ms(samples: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _nccl_version() -> str | None:
    if not t.cuda.is_available():
        return None
    try:
        version = nccl.version()
    except (AttributeError, RuntimeError):
        return None
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _build_target_layout(
    feature_mask_path: Path,
    feature_clusters_path: Path,
) -> FeaturePredictionTargetLayout:
    cfg = {
        "feature_mask": {"mode": "whitelist", "path": str(feature_mask_path)},
        "feature_clusters": {"path": str(feature_clusters_path)},
    }
    feature_mask = resolve_feature_mask(cfg, N_SAE)
    resolved_clusters = resolve_feature_clusters(cfg, N_SAE)
    assert resolved_clusters is not None
    clusters = FeatureClusterConsolidation.build(resolved_clusters, feature_mask)
    feature_vocab = effective_feature_vocab(clusters, N_SAE)
    if feature_vocab is None or feature_vocab.size != REAL_V:
        size = N_SAE if feature_vocab is None else feature_vocab.size
        raise ValueError(f"expected production feature vocabulary V={REAL_V}, got {size}")
    return FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(N_SAE),
        feature_mask=feature_mask,
        feature_clusters=clusters,
        feature_vocab=feature_vocab,
    )


def _artifact_paths(cfg: DictConfig) -> tuple[Path, Path, Path, Path]:
    names = ("packed_shard", "selection_shard", "feature_mask", "feature_clusters")
    missing = [name for name in names if cfg.artifacts.get(name) is None]
    if missing:
        raise ValueError(f"mode={cfg.mode} requires artifact paths: {missing}")
    packed, selection, mask, clusters = (Path(str(cfg.artifacts[name])) for name in names)
    return packed, selection, mask, clusters


def _reconstruct_training_batch(
    packed: Any,
    selection: Any,
    row_indices: list[int],
    target_layout: FeaturePredictionTargetLayout,
    *,
    device: str,
) -> tuple[t.Tensor, ...]:
    batch_index = _build_batch_index(packed, row_indices, pad_n_seg=PAD_N_SEG)
    reconstructed = reconstruct_batch(
        packed,
        row_indices,
        selection,
        EFFECTIVE_BUCKETS,
        device=device,
        pad_n_seg=PAD_N_SEG,
        pad_F=PAD_F,
    )
    fp_targets = build_fp_targets_from_shard(
        packed,
        batch_index,
        device=device,
        fp_target_layout=target_layout,
    )
    return (
        reconstructed.inputs,
        reconstructed.targets,
        reconstructed.enc_fids,
        reconstructed.enc_mags,
        reconstructed.enc_mask,
        reconstructed.seg_map,
        reconstructed.seg_doc_id,
        fp_targets,
    )


def _data_worker(
    worker_rank: int,
    workers: int,
    torch_threads: int,
    batch_size: int,
    batches: int,
    packed_path: str,
    selection_path: str,
    feature_mask_path: str,
    feature_clusters_path: str,
    barrier: Any,
    queue: Any,
) -> None:
    try:
        t.set_num_threads(torch_threads)
        layout = _build_target_layout(Path(feature_mask_path), Path(feature_clusters_path))
        barrier.wait()
        started = time.perf_counter()
        packed_started = time.perf_counter()
        packed = load_packed_row_shard(Path(packed_path))
        packed_ms = (time.perf_counter() - packed_started) * 1e3
        selection_started = time.perf_counter()
        selection = load_selection_shard(Path(selection_path))
        selection_ms = (time.perf_counter() - selection_started) * 1e3

        reconstruct_samples: list[float] = []
        for batch_idx in range(batches):
            first_local_row = batch_idx * batch_size
            row_indices = [
                worker_rank + workers * (first_local_row + offset) for offset in range(batch_size)
            ]
            batch_started = time.perf_counter()
            _reconstruct_training_batch(
                packed,
                selection,
                row_indices,
                layout,
                device="cpu",
            )
            reconstruct_samples.append((time.perf_counter() - batch_started) * 1e3)

        total_ms = (time.perf_counter() - started) * 1e3
        local_rows = (packed.n_rows - worker_rank + workers - 1) // workers
        queue.put(
            {
                "worker_rank": worker_rank,
                "torch_threads": torch_threads,
                "packed_decode_ms": packed_ms,
                "selection_decode_ms": selection_ms,
                "reconstruct": _summary_ms(reconstruct_samples),
                "total_ms": total_ms,
                "packed_rows": packed.n_rows,
                "rank_local_rows": local_rows,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        )
    except BaseException as exc:
        queue.put({"worker_rank": worker_rank, "error": repr(exc)})


def profile_data(cfg: DictConfig) -> dict[str, Any]:
    packed_path, selection_path, mask_path, clusters_path = _artifact_paths(cfg)
    data_cfg = cfg.data
    # Fail before creating workers: a path/config error before a child reaches
    # the start barrier would otherwise leave the parent waiting indefinitely.
    _build_target_layout(mask_path, clusters_path)
    context = mp.get_context("spawn")
    runs: list[dict[str, Any]] = []
    for workers in data_cfg.workers:
        barrier = context.Barrier(workers + 1)
        queue = context.Queue()
        processes = [
            context.Process(
                target=_data_worker,
                args=(
                    rank,
                    workers,
                    data_cfg.torch_threads,
                    data_cfg.batch_size,
                    data_cfg.batches,
                    str(packed_path),
                    str(selection_path),
                    str(mask_path),
                    str(clusters_path),
                    barrier,
                    queue,
                ),
            )
            for rank in range(workers)
        ]
        for process in processes:
            process.start()
        barrier.wait()
        wall_started = time.perf_counter()
        worker_results: list[dict[str, Any]] = []
        for _ in processes:
            try:
                worker_results.append(queue.get(timeout=data_cfg.timeout_seconds))
            except Empty as exc:
                raise TimeoutError(f"timed out waiting for {workers} data workers") from exc
        wall_ms = (time.perf_counter() - wall_started) * 1e3
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        failures = [result for result in worker_results if "error" in result]
        if failures:
            raise RuntimeError(f"data worker failure(s): {failures}")

        critical = max(result["total_ms"] for result in worker_results)
        local_batches_per_body = min(
            result["rank_local_rows"] // data_cfg.batch_size for result in worker_results
        )
        runs.append(
            {
                "workers": workers,
                "wall_ms": wall_ms,
                "critical_path_ms": critical,
                "aggregate_profiled_rows_per_second": (
                    workers * data_cfg.batch_size * data_cfg.batches / (critical / 1e3)
                ),
                "local_batches_per_body": local_batches_per_body,
                "critical_decode_amortized_ms_per_local_batch": max(
                    result["packed_decode_ms"] + result["selection_decode_ms"]
                    for result in worker_results
                )
                / local_batches_per_body,
                "workers_detail": sorted(worker_results, key=lambda result: result["worker_rank"]),
            }
        )
    return {
        "evidence": "measured_local_cpu_processes",
        "limitations": (
            "Same-host page cache, CPU decompression, and reconstruction only; "
            "not NCCL, GPU, remote-object-store, or multi-node performance."
        ),
        "packed_shard": str(packed_path.resolve()),
        "selection_shard": str(selection_path.resolve()),
        "batch_size_per_process": data_cfg.batch_size,
        "torch_threads_per_process": data_cfg.torch_threads,
        "profiled_batches_per_process": data_cfg.batches,
        "runs": runs,
    }


def _train_step(
    model: Any,
    optimizer: t.optim.Optimizer,
    batch: tuple[t.Tensor, ...],
) -> Any:
    x, y, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = batch
    result = model(
        x,
        y,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id=seg_doc_id,
        static_max_seg_size=MAX_SEG_SIZE,
        fp_targets=fp_targets,
    )
    result.loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return result


def profile_train(cfg: DictConfig) -> dict[str, Any]:
    if not t.cuda.is_available():
        raise RuntimeError("train mode requires CUDA")
    packed_path, selection_path, mask_path, clusters_path = _artifact_paths(cfg)
    train_cfg = cfg.train
    if train_cfg.checkpoint_dir is None or int(train_cfg.step) < 0:
        raise ValueError("train mode requires train.checkpoint_dir and train.step")
    checkpoint_dir = Path(str(train_cfg.checkpoint_dir))
    device = t.device("cuda")
    target_layout = _build_target_layout(mask_path, clusters_path)
    packed = load_packed_row_shard(packed_path)
    selection = load_selection_shard(selection_path)
    row_indices = list(range(train_cfg.batch_size))
    batch = _reconstruct_training_batch(
        packed,
        selection,
        row_indices,
        target_layout,
        device=str(device),
    )
    t.cuda.synchronize()

    raw_model, _tokenizer, meta = build_encdec_model(
        str(checkpoint_dir),
        train_cfg.step,
        device,
        ModelPhase.TRAIN,
        eval_conditioning=EvalConditioning.not_evaluating(),
    )
    if raw_model.feature_vocab != target_layout.feature_vocab:
        raise ValueError("checkpoint vocabulary does not match the real data target layout")
    model = t.compile(raw_model, dynamic=False) if train_cfg.compile else raw_model
    optimizer = raw_model.setup_optimizer()

    for _ in range(train_cfg.warmup_steps):
        _train_step(model, optimizer, batch)
    t.cuda.synchronize()
    t.cuda.reset_peak_memory_stats()

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    optimizer_ms: list[float] = []
    wall_ms: list[float] = []
    loss = math.nan
    for _ in range(train_cfg.steps):
        start = t.cuda.Event(enable_timing=True)
        after_forward = t.cuda.Event(enable_timing=True)
        after_backward = t.cuda.Event(enable_timing=True)
        after_optimizer = t.cuda.Event(enable_timing=True)
        t.cuda.synchronize()
        wall_started = time.perf_counter()
        start.record()
        x, y, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = batch
        result = model(
            x,
            y,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            static_max_seg_size=MAX_SEG_SIZE,
            fp_targets=fp_targets,
        )
        after_forward.record()
        result.loss.backward()
        after_backward.record()
        optimizer.step()
        raw_model.zero_grad(set_to_none=True)
        after_optimizer.record()
        t.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_started) * 1e3)
        forward_ms.append(start.elapsed_time(after_forward))
        backward_ms.append(after_forward.elapsed_time(after_backward))
        optimizer_ms.append(after_backward.elapsed_time(after_optimizer))
        loss = float(result.loss.detach().item())

    kernel_table: str | None = None
    if train_cfg.profile_kernels:
        with t.profiler.profile(
            activities=[
                t.profiler.ProfilerActivity.CPU,
                t.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            _train_step(model, optimizer, batch)
        t.cuda.synchronize()
        kernel_table = profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=train_cfg.kernel_rows,
        )

    return {
        "evidence": "measured_single_gpu_exact_checkpoint_real_rows",
        "limitations": (
            "One RTX 4090 process; no NCCL and batch size is downscaled from the "
            "production H100 run. Data decode/reconstruction is measured separately."
        ),
        "gpu": t.cuda.get_device_name(device),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_step": meta["step"],
        "packed_shard": str(packed_path.resolve()),
        "selection_shard": str(selection_path.resolve()),
        "compiled": train_cfg.compile,
        "batch_size": train_cfg.batch_size,
        "sequence_length": int(batch[0].shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
        "loss_last": loss,
        "forward": _summary_ms(forward_ms),
        "backward": _summary_ms(backward_ms),
        "optimizer": _summary_ms(optimizer_ms),
        "step_wall": _summary_ms(wall_ms),
        "tokens_per_second_median": (
            train_cfg.batch_size * SEQ_LEN / (statistics.median(wall_ms) / 1e3)
        ),
        "peak_allocated_gib": t.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": t.cuda.max_memory_reserved() / 2**30,
        "kernel_table": kernel_table,
    }


def distributed_grad_accum_steps(
    total_batch_size: int,
    *,
    device_batch_size: int,
    sequence_length: int,
    world_size: int,
) -> int:
    """Resolve the trainer's exact token-batch divisibility contract."""
    world_microbatch_tokens = device_batch_size * sequence_length * world_size
    if total_batch_size % world_microbatch_tokens != 0:
        raise ValueError(
            "total_batch_size must be divisible by the world microbatch: "
            f"{total_batch_size} % ({device_batch_size} * {sequence_length} * "
            f"{world_size}) != 0"
        )
    return total_batch_size // world_microbatch_tokens


def _synthetic_production_batch(
    *,
    device: t.device,
    batch_size: int,
    rank: int,
    sequence_length: int = SEQ_LEN,
    padded_segments: int = PAD_N_SEG,
    padded_features: int = PAD_F,
    segment_size: int = 128,
    feature_vocab_size: int = REAL_V,
) -> tuple[t.Tensor, ...]:
    """Build the real static production shapes without reading artifact bodies."""
    real_segments = math.ceil(sequence_length / segment_size)
    if padded_segments < real_segments:
        raise ValueError(
            "padded_segments must cover every real synthetic segment: "
            f"{padded_segments} < {real_segments}"
        )
    if padded_features < 24:
        raise ValueError(
            "padded_features must cover the synthetic 12-positive/12-negative "
            f"selection: {padded_features} < 24"
        )
    generator = t.Generator(device=device)
    generator.manual_seed(20_260_815 + rank)
    inputs = t.randint(
        0,
        32_768,
        (batch_size, sequence_length),
        dtype=t.int64,
        device=device,
        generator=generator,
    )
    targets = t.randint(
        0,
        32_768,
        (batch_size, sequence_length),
        dtype=t.int64,
        device=device,
        generator=generator,
    )
    enc_fids = t.randint(
        0,
        feature_vocab_size,
        (batch_size, padded_segments, padded_features),
        dtype=t.int64,
        device=device,
        generator=generator,
    )
    enc_mags = t.randint(
        0,
        2,
        (batch_size, padded_segments, padded_features),
        dtype=t.int64,
        device=device,
        generator=generator,
    )
    enc_mask = t.zeros(
        batch_size,
        padded_segments,
        padded_features,
        dtype=t.bool,
        device=device,
    )
    # The configured selector averages 12 positive + 12 negative features.
    enc_mask[:, :real_segments, :24] = True
    seg_map = (
        (t.arange(sequence_length, device=device) // segment_size)
        .unsqueeze(0)
        .expand(batch_size, sequence_length)
        .contiguous()
    )
    seg_doc_id = (
        t.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(batch_size, padded_segments)
        .contiguous()
    )
    seg_doc_id[:, real_segments:] = -1
    fp_targets = t.zeros(
        batch_size,
        padded_segments,
        feature_vocab_size,
        dtype=t.float32,
        device=device,
    )
    return (
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id,
        fp_targets,
    )


def _forward_loss(
    model: Any,
    batch: tuple[t.Tensor, ...],
    *,
    static_max_seg_size: int = MAX_SEG_SIZE,
) -> t.Tensor:
    """Use production SFT autocast while retaining FP32 master parameters."""
    x, y, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = batch
    with t.autocast(device_type=x.device.type, dtype=t.bfloat16, enabled=x.device.type == "cuda"):
        result = model(
            x,
            y,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            static_max_seg_size=static_max_seg_size,
            fp_targets=fp_targets,
        )
    return result.loss


@t.no_grad()
def _validation_forward(
    model: Any,
    batch: tuple[t.Tensor, ...],
) -> t.Tensor:
    """Run the trainer's conditioned-BPB model call on one resident batch."""
    x, y, enc_fids, enc_mags, enc_mask, seg_map, _seg_doc_id, _fp_targets = batch
    with t.autocast(device_type=x.device.type, dtype=t.bfloat16, enabled=x.device.type == "cuda"):
        return model(
            x,
            y,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            loss_reduction="none",
        )


def _distributed_validation_step(
    model: Any,
    batch: tuple[t.Tensor, ...],
) -> float:
    """Measure one synchronized eager BF16 no-grad conditioned-BPB forward."""
    started = time.perf_counter()
    loss = _validation_forward(model, batch)
    loss.sum().item()
    t.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3


def _optimizer_group_identity(group: dict[str, Any], world_size: int) -> dict[str, Any]:
    kind_value = getattr(group["kind"], "value", group["kind"])
    kind = str(kind_value)
    parameters = list(group["params"])
    unpadded_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    if kind == "muon":
        parameter = parameters[0]
        padded_count = math.ceil(len(parameters) / world_size) * world_size
        logical_payload_bytes = padded_count * parameter.numel() * parameter.element_size()
    else:
        padded_count = len(parameters)
        logical_payload_bytes = unpadded_bytes
    return {
        "kind": kind,
        "parameter_tensors": len(parameters),
        "parameter_numel": sum(parameter.numel() for parameter in parameters),
        "parameter_shapes": sorted({tuple(parameter.shape) for parameter in parameters}),
        "padded_parameter_tensors": padded_count,
        "logical_payload_bytes": logical_payload_bytes,
        "unpadded_payload_bytes": unpadded_bytes,
        "payload_padding_ratio": logical_payload_bytes / unpadded_bytes,
    }


@t.no_grad()
def _serialized_optimizer_diagnostic_step(
    optimizer: DistMuonAdamW,
    raw_model: Any,
) -> list[dict[str, Any]]:
    """Time optimizer groups after deliberately removing cross-group overlap."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group_results: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        kind_value = getattr(group["kind"], "value", group["kind"])
        kind = str(kind_value)
        dist.barrier()
        t.cuda.synchronize()

        reduce_started = time.perf_counter()
        if kind == "adamw":
            reduce_info = optimizer._reduce_adamw(group, world_size)
        elif kind == "muon":
            reduce_info = optimizer._reduce_muon(group, world_size)
        else:
            raise ValueError(f"unknown optimizer group kind: {kind}")
        t.cuda.synchronize()
        reduce_ms = (time.perf_counter() - reduce_started) * 1e3

        gather_list: list[dict[str, Any]] = []
        update_started = time.perf_counter()
        if kind == "adamw":
            optimizer._compute_adamw(group, reduce_info, gather_list, rank, world_size)
        else:
            optimizer._compute_muon(group, reduce_info, gather_list, rank)
        t.cuda.synchronize()
        update_gather_ms = (time.perf_counter() - update_started) * 1e3

        copy_started = time.perf_counter()
        optimizer._finish_gathers(gather_list)
        t.cuda.synchronize()
        finish_copy_ms = (time.perf_counter() - copy_started) * 1e3
        group_results.append(
            {
                "group_index": group_index,
                **_optimizer_group_identity(group, world_size),
                "reduce_ms": reduce_ms,
                "update_gather_ms": update_gather_ms,
                "finish_copy_ms": finish_copy_ms,
                "total_ms": reduce_ms + update_gather_ms + finish_copy_ms,
            }
        )
    raw_model.zero_grad(set_to_none=True)
    return group_results


def _aggregate_optimizer_diagnostics(
    rank_results: list[dict[str, Any]],
    diagnostic_steps: int,
) -> dict[str, Any] | None:
    if diagnostic_steps == 0:
        return None
    phases = ("reduce_ms", "update_gather_ms", "finish_copy_ms", "total_ms")
    group_count = len(rank_results[0]["optimizer_diagnostics"][0])
    groups: list[dict[str, Any]] = []
    serialized_total_samples: list[float] = [0.0] * diagnostic_steps
    for group_index in range(group_count):
        first_group = rank_results[0]["optimizer_diagnostics"][0][group_index]
        identity = {key: value for key, value in first_group.items() if key not in phases}
        phase_samples = {
            phase: [
                max(
                    float(result["optimizer_diagnostics"][step][group_index][phase])
                    for result in rank_results
                )
                for step in range(diagnostic_steps)
            ]
            for phase in phases
        }
        for step, sample in enumerate(phase_samples["total_ms"]):
            serialized_total_samples[step] += sample
        groups.append(
            {
                **identity,
                "critical_samples_ms": phase_samples,
                "critical_summary": {
                    phase: _summary_ms(samples) for phase, samples in phase_samples.items()
                },
                "rank_total_samples_ms": [
                    [
                        float(result["optimizer_diagnostics"][step][group_index]["total_ms"])
                        for step in range(diagnostic_steps)
                    ]
                    for result in rank_results
                ],
            }
        )
    return {
        "evidence": "serialized_synchronizing_diagnostic_not_production_throughput",
        "steps": diagnostic_steps,
        "limitations": (
            "Synchronizes every optimizer group and phase, intentionally removing "
            "DistMuonAdamW cross-group communication/computation overlap. Use to locate "
            "expensive group classes, not to predict ordinary optimizer wall time."
        ),
        "critical_total_samples_ms": serialized_total_samples,
        "critical_total": _summary_ms(serialized_total_samples),
        "groups": groups,
    }


def _synchronize_before_step(ddp: bool) -> None:
    if ddp:
        dist.barrier()
    t.cuda.synchronize()


def _distributed_train_step(
    model: Any,
    raw_model: Any,
    optimizer: t.optim.Optimizer,
    batch: tuple[t.Tensor, ...],
    *,
    grad_accum_steps: int,
    static_max_seg_size: int = MAX_SEG_SIZE,
) -> dict[str, float]:
    """Time a complete update, then check and report its last microbatch loss."""
    phase_events: list[tuple[t.cuda.Event, t.cuda.Event, t.cuda.Event]] = []
    optimizer_start = t.cuda.Event(enable_timing=True)
    optimizer_end = t.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    for _ in range(grad_accum_steps):
        forward_start = t.cuda.Event(enable_timing=True)
        forward_end = t.cuda.Event(enable_timing=True)
        backward_end = t.cuda.Event(enable_timing=True)
        forward_start.record()
        microbatch_loss = _forward_loss(
            model,
            batch,
            static_max_seg_size=static_max_seg_size,
        )
        loss = microbatch_loss / grad_accum_steps
        forward_end.record()
        loss.backward()
        backward_end.record()
        phase_events.append((forward_start, forward_end, backward_end))
    optimizer_start.record()
    optimizer.step()
    raw_model.zero_grad(set_to_none=True)
    optimizer_end.record()
    t.cuda.synchronize()
    sample = {
        "forward_ms": sum(start.elapsed_time(end) for start, end, _backward in phase_events),
        "backward_ms": sum(
            forward_end.elapsed_time(backward_end)
            for _start, forward_end, backward_end in phase_events
        ),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "wall_ms": (time.perf_counter() - wall_started) * 1e3,
    }
    # Read the receipt only after every timing interval has stopped. No scalar
    # extraction or finiteness synchronization enters the throughput samples.
    last_loss = float(microbatch_loss.detach().item())
    if not math.isfinite(last_loss):
        raise RuntimeError(f"synthetic training produced non-finite loss: {last_loss}")
    sample["last_microbatch_loss"] = last_loss
    return sample


def _component_diagnostic_step(
    raw_model: EncoderDecoderGPT,
    optimizer: t.optim.Optimizer,
    batch: tuple[t.Tensor, ...],
    recorder: _CudaModulePhaseRecorder,
    *,
    grad_accum_steps: int,
    static_max_seg_size: int,
) -> dict[str, Any]:
    """Run one eager step with component hooks and synchronized CUDA timings."""
    recorder.reset()
    total = _distributed_train_step(
        raw_model,
        raw_model,
        optimizer,
        batch,
        grad_accum_steps=grad_accum_steps,
        static_max_seg_size=static_max_seg_size,
    )
    component_forward_ms = recorder.elapsed_ms()
    forward_attributed_ms = sum(value for value in component_forward_ms.values() if value is not None)
    return {
        **total,
        "component_forward_ms": component_forward_ms,
        "unattributed_forward_ms": max(0.0, total["forward_ms"] - forward_attributed_ms),
    }


def _summarize_device_operators(profiler: Any, row_limit: int) -> list[dict[str, Any]]:
    # PyTorch 2.13 can expose the same named CUDA activity twice (once through
    # its owning operator and once as a device activity). Keep one rank entry
    # per exact name and avoid a misleading additive-total claim.
    by_name: dict[str, Any] = {}
    for event in profiler.key_averages():
        if float(event.self_device_time_total) <= 0.0:
            continue
        name = str(event.key)
        previous = by_name.get(name)
        if previous is None or float(event.self_device_time_total) > float(
            previous.self_device_time_total
        ):
            by_name[name] = event
    events = sorted(
        by_name.values(),
        key=lambda event: float(event.self_device_time_total),
        reverse=True,
    )
    return [
        {
            "name": str(event.key),
            "calls": int(event.count),
            "self_device_ms": float(event.self_device_time_total) / 1e3,
        }
        for event in events[:row_limit]
    ]


def _profile_compiled_operators(
    model: Any,
    raw_model: EncoderDecoderGPT,
    optimizer: t.optim.Optimizer,
    batch: tuple[t.Tensor, ...],
    *,
    row_limit: int,
    static_max_seg_size: int,
    trace_dir: Path | None = None,
    rank: int = 0,
) -> dict[str, Any]:
    """Profile one warmed microbatch and update in three separate CUDA phases."""
    activities = [t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA]
    with t.profiler.profile(activities=activities, record_shapes=True) as forward_profiler:
        loss = _forward_loss(
            model,
            batch,
            static_max_seg_size=static_max_seg_size,
        )
        t.cuda.synchronize()
    with t.profiler.profile(activities=activities, record_shapes=True) as backward_profiler:
        loss.backward()
        t.cuda.synchronize()
    with t.profiler.profile(activities=activities, record_shapes=True) as optimizer_profiler:
        optimizer.step()
        raw_model.zero_grad(set_to_none=True)
        t.cuda.synchronize()
    profilers = (
        ("forward", forward_profiler),
        ("backward", backward_profiler),
        ("optimizer", optimizer_profiler),
    )
    trace_paths: dict[str, str] = {}
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        for phase, profiler in profilers:
            path = trace_dir / f"{phase}_rank{rank}.json"
            profiler.export_chrome_trace(str(path))
            trace_paths[phase] = str(path)
    return {
        "evidence": "one_warmed_compiled_microbatch_operator_profile",
        "trace_paths": trace_paths,
        "limitations": (
            "Operator self-device times are one diagnostic microbatch and optimizer "
            "update; this diagnostic does not reproduce gradient accumulation. "
            "Each phase synchronizes CUDA independently. Use repeated ordinary steps "
            "for throughput and module-hook samples for forward component attribution."
        ),
        **{
            f"{phase}_top_operators": _summarize_device_operators(profiler, row_limit)
            for phase, profiler in profilers
        },
    }


def _critical_samples(
    rank_results: list[dict[str, Any]],
    result_key: str,
    steps: int,
) -> dict[str, list[float]]:
    return {
        phase: [
            max(float(result[result_key][step][phase]) for result in rank_results)
            for step in range(steps)
        ]
        for phase in ("forward_ms", "backward_ms", "optimizer_ms", "wall_ms")
    }


def _aggregate_component_diagnostics(
    rank_results: list[dict[str, Any]],
    diagnostic_steps: int,
) -> dict[str, Any] | None:
    if diagnostic_steps == 0:
        return None

    control = _critical_samples(
        rank_results,
        "component_eager_controls",
        diagnostic_steps,
    )
    instrumented = _critical_samples(
        rank_results,
        "component_diagnostics",
        diagnostic_steps,
    )

    component_samples = {name: [] for name in _COMPONENT_NAMES}
    unattributed_samples: list[float] = []
    fractions = {name: [] for name in (*_COMPONENT_NAMES, "unattributed")}
    for step in range(diagnostic_steps):
        # Preserve additivity by taking the breakdown from the rank that
        # defines the forward critical path, rather than independently
        # maximizing every component across different ranks.
        critical_result = max(
            rank_results,
            key=lambda result: float(result["component_diagnostics"][step]["forward_ms"]),
        )
        sample = critical_result["component_diagnostics"][step]
        total_ms = float(sample["forward_ms"])
        for name in _COMPONENT_NAMES:
            observed = sample["component_forward_ms"][name]
            if observed is None:
                continue
            value = float(observed)
            component_samples[name].append(value)
            fractions[name].append(value / total_ms)
        unattributed = float(sample["unattributed_forward_ms"])
        unattributed_samples.append(unattributed)
        fractions["unattributed"].append(unattributed / total_ms)

    return {
        "evidence": "measured_eager_cuda_module_hooks_synthetic_rows",
        "steps": diagnostic_steps,
        "limitations": (
            "The forward component breakdown is eager because module hooks are not safe "
            "to attach to the compiled production-throughput graph. The adjacent eager "
            "control quantifies instrumentation overhead; use the ordinary compiled "
            "samples for throughput. Backward remains a measured total. Unattributed "
            "forward includes embedding/scalar glue, final normalization, logit softcap, "
            "cross-entropy, feature-prediction loss reduction, and any component whose "
            "direct implementation calls bypass Module hooks. Such components are "
            "unmeasured (null), not zero-cost."
        ),
        "eager_control": {
            phase.removesuffix("_ms"): _summary_ms(samples) for phase, samples in control.items()
        },
        "instrumented": {
            phase.removesuffix("_ms"): _summary_ms(samples)
            for phase, samples in instrumented.items()
        },
        "instrumentation_overhead_fraction_of_wall_median": (
            statistics.median(instrumented["wall_ms"]) / statistics.median(control["wall_ms"]) - 1.0
        ),
        "compiled_operator_profile_rank0": rank_results[0]["compiled_operator_profile"],
        "forward": {
            "total": _summary_ms(instrumented["forward_ms"]),
            "components": {
                name: (
                    {
                        **_summary_ms(samples),
                        "median_fraction_of_forward": statistics.median(fractions[name]),
                    }
                    if len(samples) == diagnostic_steps
                    else None
                )
                for name, samples in component_samples.items()
            },
            "unattributed": {
                **_summary_ms(unattributed_samples),
                "median_fraction_of_forward": statistics.median(fractions["unattributed"]),
            },
        },
        "rank_samples": [
            {
                "rank": result["rank"],
                "controls": result["component_eager_controls"],
                "instrumented": result["component_diagnostics"],
            }
            for result in rank_results
        ],
    }


@t._dynamo.config.patch(cache_size_limit=32)
def profile_distributed(cfg: DictConfig) -> dict[str, Any] | None:
    """Measure synthetic model steps on one GPU or with torchrun collectives."""
    if not t.cuda.is_available():
        raise RuntimeError("distributed mode requires CUDA")
    distributed_cfg = cfg.distributed
    if distributed_cfg.get("muon_parameter_packing", False) and int(
        distributed_cfg.optimizer_diagnostic_steps
    ) > 0:
        raise ValueError(
            "muon_parameter_packing does not support serialized optimizer diagnostics; "
            "use component_diagnostic_steps for traces through optimizer.step()"
        )
    ddp, rank, local_rank, world_size, device = compute_init("cuda")
    try:
        model_cfg = distributed_cfg.get("model")
        geometry = _resolve_topology_model_geometry(model_cfg)
        sequence_length = int(geometry["sequence_length"])
        padded_segments = int(distributed_cfg.padded_segments)
        padded_features = int(distributed_cfg.padded_features)
        segment_size = int(distributed_cfg.segment_size)
        grad_accum_steps = distributed_grad_accum_steps(
            int(distributed_cfg.total_batch_size),
            device_batch_size=int(distributed_cfg.device_batch_size),
            sequence_length=sequence_length,
            world_size=world_size,
        )
        static_max_seg_size = int(distributed_cfg.static_max_seg_size)
        if static_max_seg_size < int(distributed_cfg.segment_size):
            raise ValueError(
                "distributed.static_max_seg_size must cover the synthetic "
                f"segment size: {static_max_seg_size} < {distributed_cfg.segment_size}"
            )
        with t.device("meta"):
            if model_cfg is None:
                vocab = FeatureVocab(
                    n_sae_features=N_SAE,
                    original_ids=tuple(range(REAL_V)),
                )
                raw_model = build_meta_model(vocab, SEQ_LEN)
            else:
                raw_model = _build_topology_meta_model(model_cfg)
        raw_model.to_empty(device=device)
        t.manual_seed(42)
        raw_model.init_weights()
        raw_model.train()
        fp8 = bool(distributed_cfg.fp8)
        fp8_converted = 0
        fp8_skipped = 0
        if fp8:
            fp8_converted, fp8_skipped = apply_fp8_training(
                raw_model,
                recipe=str(distributed_cfg.fp8_recipe),
            )
        raw_model.prewarm_decoder_block_masks(
            int(distributed_cfg.device_batch_size), sequence_length
        )
        compile_dynamic = resolve_compile_dynamic(
            distributed_cfg.get("compile_static", True), mode="sft"
        )
        model = (
            t.compile(raw_model, dynamic=compile_dynamic)
            if bool(distributed_cfg.compile)
            else raw_model
        )
        optimizer = raw_model.setup_optimizer(
            muon_parameter_packing=distributed_cfg.get("muon_parameter_packing", False),
        )
        batch = _synthetic_production_batch(
            device=device,
            batch_size=int(distributed_cfg.device_batch_size),
            rank=rank,
            sequence_length=sequence_length,
            padded_segments=padded_segments,
            padded_features=padded_features,
            segment_size=segment_size,
            feature_vocab_size=int(geometry["feature_vocab_size"]),
        )

        last_warmup_sample: dict[str, float] | None = None
        for _ in range(int(distributed_cfg.warmup_steps)):
            _synchronize_before_step(ddp)
            last_warmup_sample = _distributed_train_step(
                model,
                raw_model,
                optimizer,
                batch,
                grad_accum_steps=grad_accum_steps,
                static_max_seg_size=static_max_seg_size,
            )
        t.cuda.reset_peak_memory_stats()

        samples: list[dict[str, float]] = []
        for _ in range(int(distributed_cfg.steps)):
            _synchronize_before_step(ddp)
            samples.append(
                _distributed_train_step(
                    model,
                    raw_model,
                    optimizer,
                    batch,
                    grad_accum_steps=grad_accum_steps,
                    static_max_seg_size=static_max_seg_size,
                )
            )

        peak_allocated_gib = t.cuda.max_memory_allocated(device) / 2**30
        peak_reserved_gib = t.cuda.max_memory_reserved(device) / 2**30
        validation_steps = int(distributed_cfg.validation_steps)
        validation_warmup_steps = int(distributed_cfg.validation_warmup_steps)
        if validation_steps < 0:
            raise ValueError("distributed.validation_steps must be non-negative")
        if validation_warmup_steps < 0:
            raise ValueError("distributed.validation_warmup_steps must be non-negative")
        validation_samples_ms: list[float] = []
        validation_warmup_peak_allocated_gib: float | None = None
        validation_warmup_peak_reserved_gib: float | None = None
        validation_peak_allocated_gib: float | None = None
        validation_peak_reserved_gib: float | None = None
        if validation_steps > 0:
            raw_model.eval()
            with disable_fp8(raw_model):
                t.cuda.reset_peak_memory_stats(device)
                for _ in range(validation_warmup_steps):
                    _synchronize_before_step(ddp)
                    _distributed_validation_step(raw_model, batch)
                validation_warmup_peak_allocated_gib = (
                    t.cuda.max_memory_allocated(device) / 2**30
                )
                validation_warmup_peak_reserved_gib = (
                    t.cuda.max_memory_reserved(device) / 2**30
                )
                t.cuda.reset_peak_memory_stats(device)
                for _ in range(validation_steps):
                    _synchronize_before_step(ddp)
                    validation_samples_ms.append(
                        _distributed_validation_step(raw_model, batch)
                    )
                validation_peak_allocated_gib = (
                    t.cuda.max_memory_allocated(device) / 2**30
                )
                validation_peak_reserved_gib = (
                    t.cuda.max_memory_reserved(device) / 2**30
                )
            raw_model.train()
        diagnostic_steps = int(distributed_cfg.optimizer_diagnostic_steps)
        if diagnostic_steps < 0:
            raise ValueError("distributed.optimizer_diagnostic_steps must be non-negative")
        if diagnostic_steps > 0 and not isinstance(optimizer, DistMuonAdamW):
            raise TypeError("optimizer diagnostics require DistMuonAdamW")
        optimizer_diagnostics: list[list[dict[str, Any]]] = []
        for _ in range(diagnostic_steps):
            _synchronize_before_step(ddp)
            for _ in range(grad_accum_steps):
                loss = (
                    _forward_loss(
                        model,
                        batch,
                        static_max_seg_size=static_max_seg_size,
                    )
                    / grad_accum_steps
                )
                loss.backward()
            optimizer_diagnostics.append(
                _serialized_optimizer_diagnostic_step(optimizer, raw_model)
            )

        component_steps = int(distributed_cfg.component_diagnostic_steps)
        component_warmup_steps = int(distributed_cfg.component_warmup_steps)
        component_kernel_rows = int(distributed_cfg.component_kernel_rows)
        if component_steps < 0:
            raise ValueError("distributed.component_diagnostic_steps must be non-negative")
        if component_warmup_steps < 0:
            raise ValueError("distributed.component_warmup_steps must be non-negative")
        if component_kernel_rows <= 0:
            raise ValueError("distributed.component_kernel_rows must be positive")
        component_eager_controls: list[dict[str, float]] = []
        component_diagnostics: list[dict[str, Any]] = []
        compiled_operator_profile: dict[str, Any] | None = None
        if component_steps > 0:
            _synchronize_before_step(ddp)
            compiled_operator_profile = _profile_compiled_operators(
                model,
                raw_model,
                optimizer,
                batch,
                row_limit=component_kernel_rows,
                static_max_seg_size=static_max_seg_size,
                trace_dir=(
                    Path(distributed_cfg.trace_dir)
                    if distributed_cfg.get("trace_dir") is not None
                    else None
                ),
                rank=rank,
            )
            for _ in range(component_warmup_steps):
                _synchronize_before_step(ddp)
                _distributed_train_step(
                    raw_model,
                    raw_model,
                    optimizer,
                    batch,
                    grad_accum_steps=grad_accum_steps,
                    static_max_seg_size=static_max_seg_size,
                )
            for _ in range(component_steps):
                _synchronize_before_step(ddp)
                component_eager_controls.append(
                    _distributed_train_step(
                        raw_model,
                        raw_model,
                        optimizer,
                        batch,
                        grad_accum_steps=grad_accum_steps,
                        static_max_seg_size=static_max_seg_size,
                    )
                )

            recorder = _CudaModulePhaseRecorder(raw_model)
            try:
                for _ in range(component_warmup_steps):
                    _synchronize_before_step(ddp)
                    _component_diagnostic_step(
                        raw_model,
                        optimizer,
                        batch,
                        recorder,
                        grad_accum_steps=grad_accum_steps,
                        static_max_seg_size=static_max_seg_size,
                    )
                for _ in range(component_steps):
                    _synchronize_before_step(ddp)
                    component_diagnostics.append(
                        _component_diagnostic_step(
                            raw_model,
                            optimizer,
                            batch,
                            recorder,
                            grad_accum_steps=grad_accum_steps,
                            static_max_seg_size=static_max_seg_size,
                        )
                    )
            finally:
                recorder.close()

        rank_result: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "gpu": t.cuda.get_device_name(device),
            "samples": samples,
            "last_warmup_loss": (
                last_warmup_sample["last_microbatch_loss"]
                if last_warmup_sample is not None
                else None
            ),
            "optimizer_diagnostics": optimizer_diagnostics,
            "component_eager_controls": component_eager_controls,
            "component_diagnostics": component_diagnostics,
            "compiled_operator_profile": compiled_operator_profile,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "validation_samples_ms": validation_samples_ms,
            "validation_warmup_peak_allocated_gib": (
                validation_warmup_peak_allocated_gib
            ),
            "validation_warmup_peak_reserved_gib": validation_warmup_peak_reserved_gib,
            "validation_peak_allocated_gib": validation_peak_allocated_gib,
            "validation_peak_reserved_gib": validation_peak_reserved_gib,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        if ddp:
            dist.all_gather_object(gathered, rank_result)
        else:
            gathered[0] = rank_result
        if rank != 0:
            return None
        rank_results = [result for result in gathered if result is not None]
        critical = _critical_samples(rank_results, "samples", int(distributed_cfg.steps))
        gpu_name = str(rank_results[0]["gpu"])
        peak_flops = get_peak_flops(gpu_name)
        flops_per_token = raw_model.estimate_flops(
            float(distributed_cfg.avg_features_per_segment),
            int(distributed_cfg.segment_size),
        )
        median_wall_s = statistics.median(critical["wall_ms"]) / 1e3
        messages = collective_inventory(world_size, model_cfg) if world_size > 1 else []
        trial_id = distributed_cfg.get("trial_id")
        validation_profile: dict[str, Any] | None = None
        if validation_steps > 0:
            validation_critical_ms = [
                max(
                    float(result["validation_samples_ms"][step])
                    for result in rank_results
                )
                for step in range(validation_steps)
            ]
            validation_profile = {
                "precision": "bf16",
                "no_grad": True,
                "compiled": False,
                "warmup_steps": validation_warmup_steps,
                "steps": validation_steps,
                "critical_forward": _summary_ms(validation_critical_ms),
                "warmup_peak_allocated_gib": max(
                    float(result["validation_warmup_peak_allocated_gib"])
                    for result in rank_results
                ),
                "warmup_peak_reserved_gib": max(
                    float(result["validation_warmup_peak_reserved_gib"])
                    for result in rank_results
                ),
                "peak_allocated_gib": max(
                    float(result["validation_peak_allocated_gib"])
                    for result in rank_results
                ),
                "peak_reserved_gib": max(
                    float(result["validation_peak_reserved_gib"])
                    for result in rank_results
                ),
            }
        return {
            "evidence": (
                "measured_cuda_nccl_configured_model_synthetic_rows"
                if world_size > 1
                else "measured_cuda_configured_model_synthetic_rows"
            ),
            "limitations": (
                "Exact configured model and synthetic encoder tensor shapes, but synthetic rows "
                "and random weights; excludes packed-body decode, remote I/O, H2D staging, "
                "checkpointing, and validation. Small measured world sizes do not exercise "
                "the target world-size optimizer padding or fabric. Uses setup_optimizer "
                "defaults and fixed learning rate/momentum, not the SFT learning-rate and "
                "momentum schedules or fp_head_lr override; this is synthetic performance "
                "screening, not a matched SFT learning trajectory."
            ),
            "trial_id": None if trial_id is None else str(trial_id),
            "software": {
                "torch": t.__version__,
                "cuda": t.version.cuda,
                "nccl": _nccl_version(),
                "attention_impl": flash_attention._active_impl().value,
                "fp8": fp8,
                "autocast_dtype": "bfloat16",
                "dynamo_cache_size_limit": t._dynamo.config.cache_size_limit,
                "fp8_recipe": str(distributed_cfg.fp8_recipe) if fp8 else None,
                "fp8_converted_modules": fp8_converted,
                "fp8_skipped_modules": fp8_skipped,
            },
            "world_size": world_size,
            "nodes": sorted({str(result["hostname"]) for result in rank_results}),
            "gpus": [str(result["gpu"]) for result in rank_results],
            "compiled": bool(distributed_cfg.compile),
            "compile_dynamic": compile_dynamic,
            "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
            "device_batch_size": int(distributed_cfg.device_batch_size),
            "sequence_length": sequence_length,
            "model_geometry": geometry,
            "padded_segments": padded_segments,
            "padded_features": padded_features,
            "static_max_seg_size": static_max_seg_size,
            "cuda_cross_attention_layout": "packed_flex",
            "query_rows_per_decoder_token": 1.0,
            "flattened_kv_rows_per_decoder_token": (
                padded_segments * padded_features / sequence_length
            ),
            "total_batch_size": int(distributed_cfg.total_batch_size),
            "grad_accum_steps": grad_accum_steps,
            "muon_parameter_packing": distributed_cfg.get("muon_parameter_packing", False),
            "optimizer_class": type(optimizer).__name__,
            "validation": validation_profile,
            "forward": _summary_ms(critical["forward_ms"]),
            "backward": _summary_ms(critical["backward_ms"]),
            "optimizer": _summary_ms(critical["optimizer_ms"]),
            "step_wall": _summary_ms(critical["wall_ms"]),
            "critical_samples_ms": critical,
            "optimizer_fraction_of_wall_median": (
                statistics.median(critical["optimizer_ms"])
                / statistics.median(critical["wall_ms"])
            ),
            "tokens_per_second_median": int(
                int(distributed_cfg.total_batch_size) / median_wall_s
            ),
            "mfu_percent_median": (
                100
                * flops_per_token
                * int(distributed_cfg.total_batch_size)
                / median_wall_s
                / (peak_flops * world_size)
            ),
            "estimated_flops_per_token": flops_per_token,
            "logical_optimizer_messages": len(messages),
            "physical_optimizer_collectives": sum(
                message.physical_collectives for message in messages
            ),
            "collective_payload_gib": sum(message.payload_bytes for message in messages)
            / 2**30,
            "serialized_optimizer_diagnostic": _aggregate_optimizer_diagnostics(
                rank_results,
                diagnostic_steps,
            ),
            "component_diagnostic": _aggregate_component_diagnostics(
                rank_results,
                component_steps,
            ),
            "rank_summaries": [
                {
                    "rank": result["rank"],
                    "local_rank": result["local_rank"],
                    "hostname": result["hostname"],
                    "gpu": result["gpu"],
                    "peak_allocated_gib": result["peak_allocated_gib"],
                    "peak_reserved_gib": result["peak_reserved_gib"],
                    "last_warmup_loss": result["last_warmup_loss"],
                    "last_microbatch_losses": [
                        sample["last_microbatch_loss"] for sample in result["samples"]
                    ],
                    "step_wall": _summary_ms(
                        [float(sample["wall_ms"]) for sample in result["samples"]]
                    ),
                    "optimizer": _summary_ms(
                        [float(sample["optimizer_ms"]) for sample in result["samples"]]
                    ),
                    "samples_ms": [
                        {key: value for key, value in sample.items() if key.endswith("_ms")}
                        for sample in result["samples"]
                    ],
                }
                for result in rank_results
            ],
        }
    finally:
        compute_cleanup()


def collective_inventory(
    world_size: int,
    model_cfg: Mapping[str, Any] | None = None,
) -> list[CollectiveMessage]:
    """Reproduce DistMuonAdamW's collective branching without real tensors.

    ``model_cfg=None`` preserves the audited 412M inventory literally. An
    explicit mapping inventories a target geometry; distributed mode can bind
    to the same mapping independently for a measured synthetic full-model run.
    """
    with t.device("meta"):
        if model_cfg is None:
            vocab = FeatureVocab(
                n_sae_features=N_SAE,
                original_ids=tuple(range(REAL_V)),
            )
            model = build_meta_model(vocab, SEQ_LEN)
        else:
            model = _build_topology_meta_model(model_cfg)
    optimizer = model.setup_optimizer()
    messages: list[CollectiveMessage] = []
    for group in optimizer.param_groups:
        if group["kind"] == "adamw":
            for parameter in group["params"]:
                payload = parameter.numel() * parameter.element_size()
                is_allreduce = parameter.numel() < 1024 or parameter.shape[0] % world_size != 0
                messages.append(
                    CollectiveMessage(
                        kind="adam_allreduce" if is_allreduce else "adam_reduce_scatter_gather",
                        payload_bytes=payload,
                        unpadded_payload_bytes=payload,
                        physical_collectives=1 if is_allreduce else 2,
                    )
                )
        else:
            parameters = group["params"]
            parameter = parameters[0]
            bytes_per_parameter = parameter.numel() * parameter.element_size()
            padded_count = math.ceil(len(parameters) / world_size) * world_size
            messages.append(
                CollectiveMessage(
                    kind="muon_reduce_scatter_gather",
                    payload_bytes=padded_count * bytes_per_parameter,
                    unpadded_payload_bytes=len(parameters) * bytes_per_parameter,
                    physical_collectives=2,
                )
            )
    return messages


def hierarchical_ring_ms(
    payload_bytes: int,
    *,
    world_size: int,
    gpus_per_node: int,
    intra_latency_us: float,
    intra_bandwidth_gb_s: float,
    inter_latency_us: float,
    inter_bandwidth_gb_s: float,
) -> float:
    """Ideal hierarchical all-reduce service time for one logical message.

    A DistMuon reduce-scatter plus all-gather pair has the same ring stages as
    one all-reduce, so both map to this logical-message model.
    """
    if world_size % gpus_per_node != 0:
        raise ValueError("world_size must be divisible by gpus_per_node")
    nodes = world_size // gpus_per_node
    intra_latency_s = intra_latency_us * 1e-6
    intra_bandwidth = intra_bandwidth_gb_s * 1e9
    intra = (
        2
        * (gpus_per_node - 1)
        * (intra_latency_s + payload_bytes / gpus_per_node / intra_bandwidth)
    )
    if nodes == 1:
        return intra * 1e3
    inter_latency_s = inter_latency_us * 1e-6
    inter_bandwidth = inter_bandwidth_gb_s * 1e9
    node_shard_bytes = payload_bytes / gpus_per_node
    inter = 2 * (nodes - 1) * (inter_latency_s + node_shard_bytes / nodes / inter_bandwidth)
    return (intra + inter) * 1e3


def profile_topology(cfg: DictConfig) -> dict[str, Any]:
    topology_cfg = cfg.topology
    model_cfg = topology_cfg.get("model")
    geometry = _resolve_topology_model_geometry(model_cfg)
    with t.device("meta"):
        model = _build_topology_meta_model(model_cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    runs: list[dict[str, Any]] = []
    for world_size in topology_cfg.world_sizes:
        messages = collective_inventory(world_size, model_cfg)
        gpus_per_node = min(topology_cfg.gpus_per_node, world_size)
        latencies = topology_cfg.inter_latency_us if world_size > gpus_per_node else [0.0]
        for inter_latency_us in latencies:
            actual_ms = sum(
                hierarchical_ring_ms(
                    message.payload_bytes,
                    world_size=world_size,
                    gpus_per_node=gpus_per_node,
                    intra_latency_us=topology_cfg.intra_latency_us,
                    intra_bandwidth_gb_s=topology_cfg.intra_bandwidth_gb_s,
                    inter_latency_us=inter_latency_us,
                    inter_bandwidth_gb_s=topology_cfg.inter_bandwidth_gb_s,
                )
                for message in messages
            )
            unpadded_ms = sum(
                hierarchical_ring_ms(
                    message.unpadded_payload_bytes,
                    world_size=world_size,
                    gpus_per_node=gpus_per_node,
                    intra_latency_us=topology_cfg.intra_latency_us,
                    intra_bandwidth_gb_s=topology_cfg.intra_bandwidth_gb_s,
                    inter_latency_us=inter_latency_us,
                    inter_bandwidth_gb_s=topology_cfg.inter_bandwidth_gb_s,
                )
                for message in messages
            )
            tokens_per_microbatch = (
                world_size
                * topology_cfg.device_batch_size
                * int(geometry["sequence_length"])
            )
            valid_batch = topology_cfg.total_batch_size % tokens_per_microbatch == 0
            runs.append(
                {
                    "world_size": world_size,
                    "nodes": world_size // gpus_per_node,
                    "gpus_per_node": gpus_per_node,
                    "inter_latency_us": inter_latency_us,
                    "logical_messages": len(messages),
                    "physical_collectives": sum(
                        message.physical_collectives for message in messages
                    ),
                    "collective_payload_gib": sum(message.payload_bytes for message in messages)
                    / 2**30,
                    "unpadded_payload_gib": sum(
                        message.unpadded_payload_bytes for message in messages
                    )
                    / 2**30,
                    "payload_padding_ratio": (
                        sum(message.payload_bytes for message in messages)
                        / sum(message.unpadded_payload_bytes for message in messages)
                    ),
                    "modeled_collective_service_ms": actual_ms,
                    "modeled_unpadded_service_ms": unpadded_ms,
                    "modeled_padding_penalty_ms": actual_ms - unpadded_ms,
                    "tokens_per_world_microbatch": tokens_per_microbatch,
                    "configured_total_batch_size": topology_cfg.total_batch_size,
                    "batch_contract_valid": valid_batch,
                    "gradient_accumulation_steps": (
                        topology_cfg.total_batch_size // tokens_per_microbatch
                        if valid_batch
                        else None
                    ),
                }
            )
    return {
        "evidence": "analytical_sensitivity_model_exact_optimizer_inventory",
        "limitations": (
            "Ideal hierarchical rings, serialized collective service, and no compute/comm "
            "overlap or NCCL protocol effects. Use for sensitivity and launch planning, "
            "not as a throughput prediction."
        ),
        "assumptions": {
            "intra_latency_us": topology_cfg.intra_latency_us,
            "intra_bandwidth_gb_s": topology_cfg.intra_bandwidth_gb_s,
            "inter_bandwidth_gb_s": topology_cfg.inter_bandwidth_gb_s,
        },
        "model_geometry": geometry,
        "parameter_count": parameter_count,
        "runs": runs,
    }


@hydra.main(
    config_path="../../configs/profiling",
    config_name="encdec_l22_training",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.mode == "data":
        result = profile_data(cfg)
    elif cfg.mode == "train":
        result = profile_train(cfg)
    elif cfg.mode == "distributed":
        result = profile_distributed(cfg)
    elif cfg.mode == "topology":
        result = profile_topology(cfg)
    else:
        raise ValueError(
            f"mode must be data, train, distributed, or topology; got {cfg.mode!r}"
        )
    if result is not None:
        serialized = json.dumps(result, indent=2)
        if cfg.get("output_json") is not None:
            output_path = Path(cfg.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n")
        print(serialized)


if __name__ == "__main__":
    main()
