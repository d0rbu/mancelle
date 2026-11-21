"""Capture and compare real target-aware CORE scoring numerics across revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch as t
from omegaconf import DictConfig, OmegaConf

import nanochat.core_eval as nanochat_core_eval
import scripts.eval.base_eval as base_eval
from scripts.profile_progression.runner import (
    _build_progression_dictconfig,
    _make_spec_from_cfg,
    _resolve_device,
)
from scripts.train.training_progression import evaluate_single_checkpoint

_LOGIT_MAX_ABS_LIMIT = 0.5
_LOGIT_RMS_LIMIT = 0.05
_LOGIT_MEAN_COSINE_MIN = 0.999
_LOGIT_MIN_RECORD_COSINE_MIN = 0.995
_TOP1_AGREEMENT_MIN = 0.999
_TOP50_OVERLAP_MEAN_MIN = 47.5
_LOSS_MAX_ABS_LIMIT = 0.1
_LOSS_RMS_LIMIT = 0.02
_EVAL_MAX_ABS_LIMIT = 0.2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--core-example-batch-size", type=int)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _capturing_forward(records: list[dict[str, Any]]):
    @t.no_grad()
    def forward_model(model, input_ids, target_mask=None, context_token_ids=None):
        batch_size, seq_len = input_ids.size()
        if getattr(model, "is_target_aware_encdec", False) is True:
            ctx = model.conditioning_context(
                target_mask=target_mask,
                context_token_ids=context_token_ids,
            )
        else:
            ctx = nullcontext()

        with ctx:
            outputs = model(input_ids)
        target_ids = t.roll(input_ids, shifts=-1, dims=1)
        losses = t.nn.functional.cross_entropy(
            outputs.view(batch_size * seq_len, -1),
            target_ids.view(batch_size * seq_len),
            reduction="none",
        ).view(batch_size, seq_len)
        losses[:, -1] = float("nan")
        predictions = outputs.argmax(dim=-1)

        if target_mask is not None:
            mask_cpu = target_mask.detach().to(device="cpu", dtype=t.bool)
            input_cpu = input_ids.detach().to(device="cpu")
            target_cpu = target_ids.detach().to(device="cpu")
            losses_cpu = losses.detach().to(device="cpu", dtype=t.float32)
            predictions_cpu = predictions.detach().to(device="cpu")
            contexts = context_token_ids or [None] * batch_size
            for row in range(batch_size):
                positions = t.nonzero(mask_cpu[row], as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                device_positions = positions.to(device=outputs.device)
                key_payload = {
                    "context_token_ids": contexts[row],
                    "positions": positions.tolist(),
                    "target_token_ids": target_cpu[row, positions].tolist(),
                }
                records.append(
                    {
                        "key": _sha256_json(key_payload),
                        "key_payload": key_payload,
                        "input_token_ids": input_cpu[row].tolist(),
                        "logits": outputs[row, device_positions]
                        .detach()
                        .to(device="cpu", dtype=t.bfloat16),
                        "losses": losses_cpu[row, positions].clone(),
                        "predictions": predictions_cpu[row, positions].clone(),
                    }
                )
        return losses, predictions

    return forward_model


def capture(
    config_path: Path,
    output_path: Path,
    *,
    core_example_batch_size: int | None = None,
) -> None:
    raw_cfg = OmegaConf.load(config_path)
    if not isinstance(raw_cfg, DictConfig):
        raise TypeError(f"expected DictConfig in {config_path}")
    if core_example_batch_size is not None:
        if core_example_batch_size < 1:
            raise ValueError("core_example_batch_size must be positive")
        raw_cfg.progression.core_example_batch_size = core_example_batch_size
    cfg = _build_progression_dictconfig(raw_cfg)
    spec = _make_spec_from_cfg(raw_cfg)
    device = _resolve_device()
    ptdtype = t.float32 if cfg.dtype == "float32" else t.bfloat16
    records: list[dict[str, Any]] = []
    capturing_forward = _capturing_forward(records)
    previous_nanochat_forward = nanochat_core_eval.forward_model
    previous_base_forward = getattr(base_eval, "forward_model", None)
    nanochat_core_eval.forward_model = capturing_forward
    if previous_base_forward is not None:
        base_eval.forward_model = capturing_forward
    try:
        evals = evaluate_single_checkpoint(
            spec=spec,
            device=device,
            ptdtype=ptdtype,
            eval_suites=["core"],
            seeds=[int(cfg.base_seed)],
            cfg=cfg,
            existing_entry=None,
        )
    finally:
        nanochat_core_eval.forward_model = previous_nanochat_forward
        if previous_base_forward is not None:
            base_eval.forward_model = previous_base_forward

    records.sort(key=lambda record: (record["key"], record["key_payload"]["positions"]))
    payload = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "evals": evals,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t.save(payload, output_path)


def _flatten_numbers(value: object, *, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numbers(item, prefix=child))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(_flatten_numbers(item, prefix=child))
        return flattened
    return {}


def compare_payloads(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    reference_keys = [record["key"] for record in reference_records]
    candidate_keys = [record["key"] for record in candidate_records]
    keys_exact = reference_keys == candidate_keys
    if not keys_exact:
        raise ValueError("reference and candidate scored-row identities differ")

    logit_abs_sum = 0.0
    logit_square_sum = 0.0
    logit_count = 0
    logit_max_abs = 0.0
    logit_dot = 0.0
    reference_square_sum = 0.0
    candidate_square_sum = 0.0
    min_record_cosine = 1.0
    top1_matches = 0
    top1_count = 0
    top50_overlap_sum = 0
    loss_abs_sum = 0.0
    loss_square_sum = 0.0
    loss_count = 0
    loss_max_abs = 0.0

    for reference_record, candidate_record in zip(
        reference_records,
        candidate_records,
        strict=True,
    ):
        reference_logits = reference_record["logits"].float()
        candidate_logits = candidate_record["logits"].float()
        if reference_logits.shape != candidate_logits.shape:
            raise ValueError(
                "reference and candidate logit shapes differ: "
                f"{tuple(reference_logits.shape)} != {tuple(candidate_logits.shape)}"
            )
        delta = reference_logits - candidate_logits
        absolute = delta.abs()
        logit_max_abs = max(logit_max_abs, float(absolute.max().item()))
        logit_abs_sum += float(absolute.sum().item())
        logit_square_sum += float(delta.square().sum().item())
        logit_count += delta.numel()
        logit_dot += float((reference_logits * candidate_logits).sum().item())
        reference_square_sum += float(reference_logits.square().sum().item())
        candidate_square_sum += float(candidate_logits.square().sum().item())
        record_cosine = t.nn.functional.cosine_similarity(
            reference_logits.flatten(),
            candidate_logits.flatten(),
            dim=0,
        )
        min_record_cosine = min(min_record_cosine, float(record_cosine.item()))

        reference_top1 = reference_logits.argmax(dim=-1)
        candidate_top1 = candidate_logits.argmax(dim=-1)
        top1_matches += int((reference_top1 == candidate_top1).sum().item())
        top1_count += reference_top1.numel()
        reference_top50 = reference_logits.topk(50, dim=-1).indices
        candidate_top50 = candidate_logits.topk(50, dim=-1).indices
        overlap = (
            (reference_top50.unsqueeze(-1) == candidate_top50.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        )
        top50_overlap_sum += int(overlap.sum().item())

        reference_losses = reference_record["losses"].float()
        candidate_losses = candidate_record["losses"].float()
        loss_delta = reference_losses - candidate_losses
        loss_absolute = loss_delta.abs()
        loss_max_abs = max(loss_max_abs, float(loss_absolute.max().item()))
        loss_abs_sum += float(loss_absolute.sum().item())
        loss_square_sum += float(loss_delta.square().sum().item())
        loss_count += loss_delta.numel()

    mean_cosine = logit_dot / math.sqrt(reference_square_sum * candidate_square_sum)
    top1_agreement = top1_matches / top1_count
    top50_overlap_mean = top50_overlap_sum / top1_count

    reference_evals = _flatten_numbers(reference["evals"])
    candidate_evals = _flatten_numbers(candidate["evals"])
    eval_paths_exact = reference_evals.keys() == candidate_evals.keys()
    if not eval_paths_exact:
        raise ValueError("reference and candidate eval metric paths differ")
    eval_deltas = {key: abs(reference_evals[key] - candidate_evals[key]) for key in reference_evals}
    eval_max_abs = max(eval_deltas.values(), default=0.0)

    metrics = {
        "record_count": len(reference_records),
        "scored_position_count": top1_count,
        "logits_max_abs": logit_max_abs,
        "logits_mean_abs": logit_abs_sum / logit_count,
        "logits_rms": math.sqrt(logit_square_sum / logit_count),
        "logits_mean_cosine": mean_cosine,
        "logits_min_record_cosine": min_record_cosine,
        "top1_agreement": top1_agreement,
        "top50_overlap_mean": top50_overlap_mean,
        "losses_max_abs": loss_max_abs,
        "losses_mean_abs": loss_abs_sum / loss_count,
        "losses_rms": math.sqrt(loss_square_sum / loss_count),
        "eval_metrics_max_abs": eval_max_abs,
        "eval_metric_deltas": eval_deltas,
    }
    gates = {
        "keys_exact": keys_exact,
        "logits_max_abs": logit_max_abs <= _LOGIT_MAX_ABS_LIMIT,
        "logits_rms": metrics["logits_rms"] <= _LOGIT_RMS_LIMIT,
        "logits_mean_cosine": mean_cosine >= _LOGIT_MEAN_COSINE_MIN,
        "logits_min_record_cosine": min_record_cosine >= _LOGIT_MIN_RECORD_COSINE_MIN,
        "top1_agreement": top1_agreement >= _TOP1_AGREEMENT_MIN,
        "top50_overlap_mean": top50_overlap_mean >= _TOP50_OVERLAP_MEAN_MIN,
        "losses_max_abs": loss_max_abs <= _LOSS_MAX_ABS_LIMIT,
        "losses_rms": metrics["losses_rms"] <= _LOSS_RMS_LIMIT,
        "eval_metrics_max_abs": eval_max_abs <= _EVAL_MAX_ABS_LIMIT,
    }
    return {
        "schema_version": 1,
        "reference_git_sha": reference["git_sha"],
        "candidate_git_sha": candidate["git_sha"],
        "thresholds": {
            "logits_max_abs": _LOGIT_MAX_ABS_LIMIT,
            "logits_rms": _LOGIT_RMS_LIMIT,
            "logits_mean_cosine": _LOGIT_MEAN_COSINE_MIN,
            "logits_min_record_cosine": _LOGIT_MIN_RECORD_COSINE_MIN,
            "top1_agreement": _TOP1_AGREEMENT_MIN,
            "top50_overlap_mean": _TOP50_OVERLAP_MEAN_MIN,
            "losses_max_abs": _LOSS_MAX_ABS_LIMIT,
            "losses_rms": _LOSS_RMS_LIMIT,
            "eval_metrics_max_abs": _EVAL_MAX_ABS_LIMIT,
        },
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }


def compare(reference_path: Path, candidate_path: Path, output_path: Path) -> None:
    reference = t.load(reference_path, map_location="cpu", weights_only=False)
    candidate = t.load(candidate_path, map_location="cpu", weights_only=False)
    result = compare_payloads(reference, candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "capture":
        capture(
            args.config,
            args.output,
            core_example_batch_size=args.core_example_batch_size,
        )
        return
    compare(args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
