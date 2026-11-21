"""Microbench: TargetAwareEncoderDecoderEvalWrapper._build_conditioning.

Expected to be the hot path on Path C (encdec conditioned). The fixture
serializes {input_ids, target_mask, context_token_ids, config_hash} from a
real context row in the progression_profiling cache.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.profile_progression_microbench import register
from scripts.train.training_progression import (
    CheckpointSpec,
    _context_provider_for_cfg,
    _wrap_conditioned_model,
    load_progression_model,
    resolve_local_checkpoint_dir,
)
from steering.eval.compacted_eval_loading import (
    conditioned_eval_cache_contract_metadata,
    conditioned_eval_feature_vocab,
)
from steering.eval.context_cache_io import (
    T_CONTEXT_TOKEN_IDS,
    T_CONTEXT_TOKEN_OFFSETS,
)
from steering.eval.feature_config import (
    parse_strict_feature_configs,
    resolve_next_segment_conditioning_offset,
)
from steering.eval.target_conditioning import TargetConditioningMode
from steering.overlay_io import load_overlay_shard
from steering.profiling.statistics import bootstrap_ci


@dataclass(frozen=True)
class _FixtureConfig:
    batch_size: int = 1
    seq_len_distribution: tuple[int, ...] = (
        # Calibrated to the enumerated-contexts parquet (median=364, p10=15, p90=997).
        # dist[1] must be near the live median (seed=2026 always picks index 1).
        15, 364, 256, 384, 472, 512, 768, 900, 1024,
    )
    seed: int = 2026
    cache_dir: str = "out/data/eval_context_features/progression_profiling"
    parquet_path: str = (
        "out/data/eval_contexts/progression_profiling/parquet/shard_00000.parquet"
    )
    features_yaml: str = (
        "out/data/common_features/safety_sft/"
        "geometric_m10_min1_max40_d12_seed1234/"
        "pretrain_ref_local_l8_1shard/features.yaml"
    )
    model_max_seq_len: int = 1024


_FIXTURE_CONFIG = _FixtureConfig()


def config_hash() -> str:
    payload = json.dumps(asdict(_FIXTURE_CONFIG), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_fixture(output_path: Path) -> Path:
    """Construct fixture tensors from the live cache and save to output_path.

    Token sequences are sourced from the c{model_max_seq_len} safetensors shard
    (every context in the parquet has a matching cache row built from it).  The
    parquet existence is checked as a precondition so the error message clearly
    points to the enumeration step rather than the caching step.
    """
    parquet_path = Path(_FIXTURE_CONFIG.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"fixture source parquet not found: {parquet_path}. "
            "Run `uv run python -m scripts.eval.build_eval_context_cache stage=enumerate ...` first."
        )

    # The parquet rows and the cache shards share the same token sequences;
    # we read token IDs directly from the safetensors cache shard rather than
    # the parquet (which stores only raw text, not pre-tokenised IDs).
    max_seq_len = _FIXTURE_CONFIG.model_max_seq_len
    shard_path = (
        Path(_FIXTURE_CONFIG.cache_dir)
        / f"c{max_seq_len}"
        / "contexts_00000.safetensors.zst"
    )
    if not shard_path.exists():
        raise FileNotFoundError(
            f"context cache shard not found: {shard_path}. "
            "Run `uv run python -m scripts.eval.build_eval_context_cache stage=cache ...` first."
        )

    tensors = load_overlay_shard(shard_path)
    token_ids_flat: list[int] = tensors[T_CONTEXT_TOKEN_IDS].cpu().tolist()
    offsets: list[int] = tensors[T_CONTEXT_TOKEN_OFFSETS].cpu().tolist()
    token_lists = [
        token_ids_flat[offsets[i] : offsets[i + 1]]
        for i in range(len(offsets) - 1)
    ]

    rng = random.Random(_FIXTURE_CONFIG.seed)
    target_seq_len = rng.choice(_FIXTURE_CONFIG.seq_len_distribution)

    # Pick the context whose NATIVE length is nearest to target_seq_len and
    # use it UNTRUNCATED. The cache is keyed by the full token-sequence hash;
    # any truncation here produces a different hash and the wrapper raises
    # KeyError on _lookup_context_segments. rng is still advanced by
    # .choice() above so config_hash stays stable across fixture rebuilds.
    if not token_lists:
        raise RuntimeError("cache shard has no token sequences")
    selected_full = min(token_lists, key=lambda toks: abs(len(toks) - target_seq_len))
    selected = [int(t) for t in selected_full]

    context_token_ids = [selected]
    input_ids = torch.tensor(context_token_ids, dtype=torch.long)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    target_mask[:, -1] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": input_ids,
            "target_mask": target_mask,
            "context_token_ids": context_token_ids,
            "config_hash": config_hash(),
        },
        output_path,
    )
    return output_path


def _load_wrapper(device: torch.device):
    """Load a d12_baseline checkpoint and wrap it for conditioned eval."""
    progression_cfg = OmegaConf.create({
        "source": "encdec",
        "model_family": "encoder_decoder",
        "checkpoint_root": "conditional_encdec_checkpoints",
        "checkpoint_prefix": "s128/dist_geometric/mean_10.0/std_10.0",
        "model_tag": "d12_baseline",
        "step": 54600,
        "phase": "pretrain",
        "subdir": "",
        "dtype": "bfloat16",
        "checkpoint_source": "local",
        "checkpoints_dir": None,
        "conditioning": {
            "context_cache_dir": _FIXTURE_CONFIG.cache_dir,
            "identity": {
                "model_id": "gemma-2-2b",
                "sae_id": "layer_8/width_16k/canonical",
                "layer": 8,
                "pooling": "max_pool",
                "feature_selection_hash": None,
            },
        },
    })

    spec = CheckpointSpec(step=54600, phase="pretrain", subdir="")
    local_dir = resolve_local_checkpoint_dir(spec, progression_cfg)
    model, tokenizer, meta, checkpoint_contract = load_progression_model(
        local_dir, 54600, device, progression_cfg
    )
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)

    feat_cfgs = parse_strict_feature_configs(
        {"safety_sft": [_FIXTURE_CONFIG.features_yaml]}
    )
    feature_cfg = feat_cfgs[0]
    context_provider = _context_provider_for_cfg(
        progression_cfg,
        expected_feature_vocab=conditioned_eval_feature_vocab(model),
        expected_contract_metadata=conditioned_eval_cache_contract_metadata(
            checkpoint_contract
        ),
        model_max_seq_len=int(model.dec_config.sequence_len),
        expected_n_sae_features=model.encoder.config.n_sae_features,
        expected_num_buckets=model.encoder.config.num_buckets,
    )
    target_mode = _target_mode_for_checkpoint(meta)
    wrapped = _wrap_conditioned_model(
        model,
        tokenizer,
        feature_cfg,
        context_provider,
        target_mode,
    )
    return wrapped, tokenizer


def _target_mode_for_checkpoint(meta: dict) -> TargetConditioningMode:
    """Mirror progression eval's checkpoint-specific target-segment routing."""
    return TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(
            {"next_segment_conditioning_offset": "auto"},
            meta,
        )
    )


@register("encdec_build_conditioning")
def run_microbench(fixture_path: Path, device_str: str, n_iters: int) -> dict:
    device = torch.device(device_str)
    fixture = torch.load(fixture_path, map_location=device, weights_only=True)
    input_ids = fixture["input_ids"].to(device)
    target_mask = fixture["target_mask"].to(device)
    context_token_ids = fixture["context_token_ids"]

    wrapped, _tokenizer = _load_wrapper(device)

    with wrapped.conditioning_context(target_mask=target_mask, context_token_ids=context_token_ids):
        # Warmup (3, discarded)
        for _ in range(3):
            _ = wrapped._build_conditioning(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed
        samples_ns: list[int] = []
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            _ = wrapped._build_conditioning(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples_ns.append(time.perf_counter_ns() - start)

    samples_ms = [s / 1e6 for s in samples_ns]
    lo, hi, median = bootstrap_ci(samples_ms)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        git_sha = "unknown"

    return {
        "name": "encdec_build_conditioning",
        "median_ms": median,
        "ci_lower_ms": lo,
        "ci_upper_ms": hi,
        "n": n_iters,
        "config_hash": fixture["config_hash"],
        "fixture_path": str(fixture_path),
        "git_sha": git_sha,
    }


def _main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "build_fixture":
        print(
            "Usage: python -m scripts.profile_progression_microbench.encdec_build_conditioning "
            "build_fixture [output_path]"
        )
        sys.exit(2)
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3
        else Path("tests/fixtures/profile_progression/encdec_build_conditioning.pt")
    )
    built = build_fixture(out_path)
    print(f"Wrote fixture: {built}")


if __name__ == "__main__":
    _main()
