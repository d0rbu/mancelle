"""Generate or judge one Campaign 2 checkpoint/variant cell, with per-prompt resume."""

from __future__ import annotations

import fcntl
import gc
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from nanochat.tokenizer import get_tokenizer
from scripts.utils.campaign2 import install_json_once, load_strict_json_object
from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.eval.axbench_upstream.methods import (
    build_self_conditioning_config,
    load_extraction_stack,
    load_method_setup,
    load_target_model,
    prepare_prompt_context,
)
from steering.eval.campaign2 import (
    EvalMode,
    Variant,
    cell_output_dir,
    preflight,
    safety_record_receipt,
    validate_safety_record,
)
from steering.eval.campaign2_judge import judge_records
from steering.eval.campaign2_safety import prepare_safety_cell
from steering.eval.campaign2_training_pairs import safety_pairs
from steering.eval.chat_benchmark_targets import normalize_input
from steering.eval.chat_eval_result_identity import semantic_digest
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.harmfulness_generation import (
    HarmfulnessResponseIdentityError,
    validate_harmfulness_response_record,
)
from steering.eval.harmfulness_judge import JudgeClient
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.eval.harmfulness_self_conditioning import (
    generate_harmfulness_response_self_conditioned,
    harmfulness_prompt_seeds,
)
from steering.eval.hf_chat_generation import (
    HuggingFaceChatGenerationEngine,
    load_repository_hf_checkpoint,
)
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider
from steering.refined_types import UnitInterval


def publish_response(output: Path, index: int, row: dict[str, Any], response: str, *, identity: dict[str, Any] | None = None) -> None:
    if not isinstance(response, str):
        raise ValueError(f"Campaign 2 response for prompt {index} must be a string, "
                         f"got {type(response).__name__}")
    record = {**row, "response": response}
    if identity is not None:
        record = {**record, "generation": safety_record_receipt(identity, index, record)}
    install_json_once(output / "records" / f"{index:06d}.json", record)

def generate_encdec(cfg: DictConfig, pending: list[tuple[int, dict]], output: Path) -> None:
    setup = load_method_setup(Path(cfg.checkpoint_dir) / f"meta_{cfg.step:06d}.json", Path(cfg.statistics))
    if Variant(cfg.variant) is Variant.ENCDEC_SAFETY:
        cell = prepare_safety_cell(cfg, setup)
        identity = load_strict_json_object(output / "identity.json")
        if preflight(cfg, safety_cell=cell) != identity:
            raise ValueError("safety cell inputs changed before generation")
        for index, row in pending:
            if index < 0 or index >= len(cell.rows) or cell.rows[index] != row:
                raise ValueError("safety pending prompt index disagrees with admitted pool")
        device = t.device(cfg.device)
        cell.assert_stable()
        model = cell.runtime.model.to(device=device, dtype=t.bfloat16 if device.type == "cuda" else t.float32)
        runtime = replace(cell.runtime, model=model, device=device)
        for index, row in pending:
            cell.assert_stable()
            response = generate_harmfulness_response_self_conditioned(setup=runtime, stable_pool_index=index,
                prompt=row["prompt"], pool=row["pool"], category=row["category"], base_seed=cfg.seed,
                max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
            cell.assert_stable()
            publish_response(output, index, row, response, identity=identity)
        return
    tokenizer = get_tokenizer()
    extraction_model, sae = load_extraction_stack(setup, cfg.device)
    prepared = {}
    for index, row in pending:
        rendered = render_prompt_for_chat_completion(tokenizer, row["prompt"])
        prepared[index] = prepare_prompt_context(setup, extraction_model, sae, tokenizer,
            token_ids=rendered.token_ids, assistant_end_id=rendered.assistant_end_id)
    # Keep Gemma+SAE and the target decoder out of GPU memory at the same time.
    del extraction_model, sae
    gc.collect()
    if cfg.device == "cuda":
        t.cuda.empty_cache()
    model, tokenizer = load_target_model(setup, cfg.device)
    if cfg.device == "cuda":
        model = model.to(dtype=t.bfloat16)
    width = setup.n_sae_features if setup.feature_vocab is None else int(setup.feature_vocab.size)
    policy = SelfCondScoreInterpolation(target_scores=t.zeros(width),
        negative_selection=setup.output_negative_selection, weight=UnitInterval.parse(0.0))
    for index, row in pending:
        context = prepared[index]
        token_seed, selection_seed = harmfulness_prompt_seeds(base_seed=cfg.seed,
            stable_pool_index=index, prompt=row["prompt"], pool=row["pool"], category=row["category"])
        config = build_self_conditioning_config(setup, model, policy, selection_seed=selection_seed)
        provider = SingleEntryContextProvider(context.token_ids, context.segments,
            segment_mode=context.segment_mode, token_assignments=context.token_assignments)
        engine = build_engine_self_conditioned(model, tokenizer, next(model.parameters()).device,
            config, context_provider=provider)
        if len(context.token_ids) + cfg.max_new_tokens > int(model.dec_config.sequence_len):
            raise ValueError(f"prompt {index} plus generation budget exceeds the checkpoint context")
        response = tokenizer.decode(list(engine.generate_stream(list(context.token_ids),
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k,
            seed=token_seed, stop_token_ids={int(tokenizer.get_bos_token_id()), context.assistant_end_id})))
        publish_response(output, index, row, response)

def generate_pythia(cfg: DictConfig, pending: list[tuple[int, dict]], output: Path) -> None:
    loaded = load_repository_hf_checkpoint(cfg.checkpoint_dir, device=cfg.device)
    engine = HuggingFaceChatGenerationEngine(loaded)
    for index, row in pending:
        token_seed, _ = harmfulness_prompt_seeds(base_seed=cfg.seed, stable_pool_index=index,
            prompt=row["prompt"], pool=row["pool"], category=row["category"])
        responses = engine.generate_batch([row["prompt"]], max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, top_k=cfg.top_k, seed=token_seed, render_as_chat=True)
        if len(responses) != 1:
            raise ValueError(f"HF generation returned {len(responses)} responses for 1 prompts")
        publish_response(output, index, row, responses[0])

def generate_axbench(cfg: DictConfig, pending: list[tuple[int, dict]], output: Path) -> None:
    variant = Variant(cfg.variant)
    pairs = safety_pairs(Path(cfg.safety.training_pairs)) if variant is Variant.PYTHIA_DIFFMEAN else []
    request = {
        "variant": variant.value, "checkpoint_dir": cfg.checkpoint_dir, "output_dir": str(output),
        "training_pairs": pairs, "instruction": cfg.safety.instruction, "device": cfg.device,
        "layer": cfg.safety.layer, "factor": cfg.safety.factor, "fit_batch_size": cfg.safety.fit_batch_size,
        "seed": cfg.seed, "temperature": cfg.temperature, "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens,
        "rows": [{"index": index, "seed": harmfulness_prompt_seeds(base_seed=cfg.seed,
            stable_pool_index=index, prompt=row["prompt"], pool=row["pool"], category=row["category"])[0],
            "row": row} for index, row in pending],
    }
    # Invocation request is transport, not retained cell identity (pending rows change on resume).
    request_path = output / "axbench_request.json"
    if request_path.exists():
        request_path.unlink()
    install_json_once(request_path, request)
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "--project", str(repo / "vendor/axbench"), "--no-sync", "python",
        str(repo / "scripts/eval/campaign2_axbench_worker.py"), str(request_path)], check=True,
        env={key: value for key, value in os.environ.items() if key != "UV_PROJECT_ENVIRONMENT"})

def collect_records(output: Path, rows: list[dict], identity: dict | None = None) -> list[dict]:
    records = []
    expected = {f"{index:06d}.json" for index in range(len(rows))}
    if {path.name for path in (output / "records").glob("*.json")} != expected:
        raise ValueError("response coverage differs from the complete prompt pool")
    for index, row in enumerate(rows):
        record = load_strict_json_object(output / "records" / f"{index:06d}.json")
        if identity is not None:
            record = validate_safety_record(record, row, index, identity)
        try:
            record = validate_harmfulness_response_record(record, row, index=index,
                source="Campaign 2 response cache", require_exact_identity=True)
        except HarmfulnessResponseIdentityError as exc:
            raise ValueError(f"response prompt identity mismatch at {index}") from exc
        records.append(record)
    return records

def validate_safety_resume(output: Path, rows: list[dict], identity: dict) -> None:
    """Read-only admission of existing partial or complete cells before launch."""
    if semantic_digest(rows) != identity["safety_binding"]["prompts_sha256"]:
        raise ValueError("safety prompt pool changed after preflight")
    identity_path, prompts_path = output / "identity.json", output / "prompts.json"
    if identity_path.exists() and load_strict_json_object(identity_path) != identity:
        raise ValueError("existing safety cell identity differs from this invocation")
    if prompts_path.exists() and json.loads(prompts_path.read_bytes()) != rows:
        raise ValueError("existing safety cell has different ordered prompts")
    records = list((output / "records").glob("*.json"))
    if (records or prompts_path.exists()) and not identity_path.exists():
        raise ValueError("existing safety artifacts require their original cell identity")
    expected = {f"{index:06d}.json": (index, row) for index, row in enumerate(rows)}
    for path in records:
        if path.name not in expected:
            raise ValueError("safety response coverage exceeds the admitted prompt pool")
        index, row = expected[path.name]
        validate_safety_record(load_strict_json_object(path), row, index, identity)
    if (output / "responses.json").exists():
        receipt = {"cell_sha256": semantic_digest(identity),
            "records_sha256": semantic_digest(collect_records(output, rows, identity))}
        if load_strict_json_object(output / "responses.json") != receipt:
            raise ValueError("safety response artifact disagrees with the admitted records")

def judge(cfg: DictConfig, records: list[dict], output: Path) -> None:
    if not cfg.allow_paid_judge or not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("judging requires allow_paid_judge=true and OPENROUTER_API_KEY")
    client = JudgeClient(model=cfg.judge_model, cache_path="",
        requests_per_minute=cfg.requests_per_minute,
        refusal_fallback_model=cfg.judge_refusal_fallback_model)
    payload = judge_records(client, records, output)
    install_json_once(output / "results.json", payload)

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="campaign2")
def main(cfg: DictConfig) -> None:
    identity = preflight(cfg)
    mode, variant = EvalMode(cfg.mode), Variant(cfg.variant)
    output = cell_output_dir(cfg)
    if mode is EvalMode.PREFLIGHT:
        if variant is Variant.ENCDEC_SAFETY and output.exists():
            pool = resolve_harmfulness_prompt_pool(seed=cfg.seed, max_harmful=cfg.num_harmful, num_benign=cfg.num_benign)
            validate_safety_resume(output, [dict(row) for row in pool.rows], identity)
        logger.info("Campaign 2 CPU preflight passed for {}/{}", cfg.stage, variant.value)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cell.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install_json_once(output / "identity.json", identity)
        pool = resolve_harmfulness_prompt_pool(seed=cfg.seed, max_harmful=cfg.num_harmful, num_benign=cfg.num_benign)
        rows = [dict(row) for row in pool.rows]
        install_json_once(output / "prompts.json", rows)
        if variant is Variant.ENCDEC_SAFETY:
            validate_safety_resume(output, rows, identity)
        if variant is Variant.PYTHIA_DIFFMEAN:
            training = {normalize_input(pair["prompt"]) for pair in safety_pairs(Path(cfg.safety.training_pairs))}
            if training & {normalize_input(row["prompt"]) for row in rows}:
                raise ValueError("separate safety training data overlaps the evaluation prompt pool")
        pending = [(index, row) for index, row in enumerate(rows)
                   if not (output / "records" / f"{index:06d}.json").exists()]
        if mode is EvalMode.GENERATE and pending:
            if variant in (Variant.ENCDEC, Variant.ENCDEC_SAFETY):
                generate_encdec(cfg, pending, output)
            elif variant is Variant.PYTHIA:
                generate_pythia(cfg, pending, output)
            else:
                generate_axbench(cfg, pending, output)
        records = collect_records(output, rows, identity if variant is Variant.ENCDEC_SAFETY else None)
        if variant is Variant.ENCDEC_SAFETY:
            install_json_once(output / "responses.json", {"cell_sha256": semantic_digest(identity),
                "records_sha256": semantic_digest(records)})
        if mode is EvalMode.JUDGE:
            judge(cfg, records, output)

if __name__ == "__main__":
    main()
