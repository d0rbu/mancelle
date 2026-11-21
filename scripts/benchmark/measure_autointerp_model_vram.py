"""Measure the single-model VRAM footprint of the autointerp token-example stack.

Task 11 of the target-sharding-removal plan: the filter used to run
`max_parallel_workers` separate processes, each holding its own resident copy
of the base transformer (layers 0..L) + SAE in VRAM. Single-process
concurrency now loads that stack exactly ONCE and shares it across every
target/trial. This script produces the concrete number backing that decision:
it loads the real production model stack for the
``gemma3_4bpt_l22_f65k_ci_nobos`` config (Gemma-3-4b-pt truncated to layer 22,
GemmaScope-2 65k-width SAE, native bf16 backend, cuda) via the SAME code path
``CachedTokenExampleProvider`` uses at runtime
(``_ensure_model_loaded`` -> ``build_native_model`` ->
``steering.gemma3_loading.load_gemma3_text_decoder`` + ``SAE.from_pretrained``)
and records the resulting peak/resident VRAM.

Usage:
    PYTHONPATH=. uv run python -m scripts.benchmark.measure_autointerp_model_vram
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch as t
from hydra import compose, initialize_config_dir

from steering.eval.autointerp_filter.cli import build_config
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig, validate_config
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    infer_n_sae_features,
)

_CONFIG_NAME = "precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos"
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "extraction"
_GIB = 1024**3


def _compose_config(output_dir: Path) -> AutoInterpFilterConfig:
    """Build the exact production config for the target config name.

    Uses the compose API (not ``@hydra.main``) since this is a one-shot
    measurement, not a torchrun-launched job. ``output_dir`` is the only
    override: everything else (model, layer, SAE release/l0, backend, dtype,
    device) comes straight from
    ``configs/extraction/precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos.yaml``,
    the same file ``scripts/preprocess/precompute_autointerp_filter.py`` reads.
    """
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        composed = compose(config_name=_CONFIG_NAME, overrides=[f"output_dir={output_dir}"])
    cfg = build_config(composed)
    validate_config(cfg)
    return cfg


def _nvidia_smi_process_bytes(device_index: int) -> int | None:
    """Cross-check torch's allocator accounting against the driver's view of
    this process's resident memory on the target GPU.

    ``nvidia-smi`` counts the whole process footprint (CUDA context,
    cuBLAS/cuDNN workspaces, the caching allocator's reserved-not-just-
    allocated blocks) so it is expected to read somewhat higher than
    ``torch.cuda.memory_allocated``. Returns ``None`` (never raises) if
    ``nvidia-smi`` is unavailable or reports nothing for this pid — this is a
    cross-check, not the primary measurement.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid, used_mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            return used_mib * 1024 * 1024
    return None


def measure_model_vram() -> dict[str, Any]:
    """Load the production autointerp model stack once and record its VRAM cost."""
    if not t.cuda.is_available():
        raise RuntimeError(
            "measure_autointerp_model_vram requires a CUDA device; none is visible to torch"
        )

    with tempfile.TemporaryDirectory(prefix="autointerp_vram_measure_") as tmp_dir:
        cfg = _compose_config(Path(tmp_dir))
        device = t.device(cfg.token_examples.device)
        if device.type != "cuda":
            raise RuntimeError(
                f"{_CONFIG_NAME}: token_examples.device={cfg.token_examples.device!r} "
                "is not a CUDA device"
            )
        device_index = device.index if device.index is not None else t.cuda.current_device()
        device = t.device(f"cuda:{device_index}")

        gc.collect()
        t.cuda.synchronize(device)
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats(device)
        baseline_vram_bytes = t.cuda.memory_allocated(device)

        feature_dir = Path(cfg.feature_dir)
        n_sae_features = infer_n_sae_features(feature_dir, cfg.n_sae_features)
        provider = CachedTokenExampleProvider(
            cfg,
            feature_dir=feature_dir,
            n_sae_features=n_sae_features,
        )
        # Exactly the load path CachedTokenExampleProvider.examples_for_target
        # takes in production: resolve_sae -> build_native_model (Gemma-3 VLM
        # text-decoder remap + structural truncation to layers 0..layer) ->
        # SAE.from_pretrained. No feature shards / parquet data required —
        # _ensure_model_loaded touches disk only for an optional stats
        # sidecar (absent here; falls back to a default context_window).
        provider._ensure_model_loaded()

        t.cuda.synchronize(device)
        peak_vram_bytes = t.cuda.max_memory_allocated(device)
        post_load_vram_bytes = t.cuda.memory_allocated(device)
        model_vram_bytes = post_load_vram_bytes - baseline_vram_bytes

        gpu_total_bytes = t.cuda.get_device_properties(device).total_memory
        gpu_name = t.cuda.get_device_name(device)
        activation_backend = provider._activation_backend
        nvidia_smi_process_bytes = _nvidia_smi_process_bytes(device_index)

    headroom_bytes = gpu_total_bytes - model_vram_bytes
    return {
        "model_vram_bytes": model_vram_bytes,
        "model_vram_gib": model_vram_bytes / _GIB,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gib": peak_vram_bytes / _GIB,
        "gpu_total_bytes": gpu_total_bytes,
        "gpu_total_gib": gpu_total_bytes / _GIB,
        "headroom_bytes": headroom_bytes,
        "headroom_gib": headroom_bytes / _GIB,
        "config": "gemma3_4bpt_l22_f65k",
        "hydra_config_name": _CONFIG_NAME,
        "activation_backend": activation_backend,
        "gpu_name": gpu_name,
        "baseline_vram_bytes": baseline_vram_bytes,
        "nvidia_smi_process_bytes": nvidia_smi_process_bytes,
        "nvidia_smi_process_gib": (
            nvidia_smi_process_bytes / _GIB if nvidia_smi_process_bytes is not None else None
        ),
    }


def main() -> int:
    result = measure_model_vram()
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"one resident model = {result['model_vram_gib']:.2f} GiB, so N-worker "
        f"process-sharding would have needed N x {result['model_vram_gib']:.2f} GiB; "
        f"headroom on the {result['gpu_total_gib']:.1f} GiB card = "
        f"{result['headroom_gib']:.2f} GiB, and since judge concurrency holds no VRAM, "
        "max_concurrent is bounded by the judge rate limit, not GPU memory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
