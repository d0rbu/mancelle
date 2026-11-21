"""Private JSON transport worker; run only in the pre-staged AxBench environment.

The public CLI is the Hydra campaign2 evaluator. This worker reuses the reviewed
vendor DiffMean fit/intervention and PromptSteering generation implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch as t
from axbench.models.mean import DiffMean
from axbench.models.prompt import PromptSteering
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

TRANSFORMERS_SEED_MODULUS = 2**32
DIFFMEAN_VARIANT = "pythia_diffmean"
PROMPT_VARIANT = "pythia_prompt"


def transformers_seed(seed: int) -> int:
    """Reduce a non-negative campaign seed for numpy-backed transformers seeding."""
    if seed < 0:
        raise ValueError(f"campaign seed must be non-negative, got {seed}")
    return seed % TRANSFORMERS_SEED_MODULUS


def render(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool, limit: int) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
    if not isinstance(text, str):
        raise ValueError("HF chat template must return text")
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if not ids or len(ids) > limit:
        raise ValueError(f"AxBench input exceeds explicit context budget {limit}; refusing truncation")
    return text


def fit_direction(request: dict, model: Any, tokenizer: Any, output: Path) -> Any:
    method = DiffMean(model, tokenizer, request["layer"], device=request["device"],
        training_args=SimpleNamespace(batch_size=request["fit_batch_size"], n_epochs=1))
    fit = output / "diffmean"
    if not (fit / "fit.json").is_file():
        if fit.exists():
            raise ValueError("partial DiffMean fit exists; inspect and move it aside before retry")
        fit.mkdir()
        examples = []
        for pair in request["training_pairs"]:
            for label, key in ((1, "safe_response"), (0, "unsafe_response")):
                text = render(tokenizer, [{"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair[key]}], generation=False, limit=1024)
                examples.append({"input": text, "labels": label, "concept_id": 0})
        frame = pd.DataFrame(examples)
        method.make_model(mode="train", low_rank_dimension=1)
        method.train(frame, prefix_length=1)
        method.ax.to(dtype=t.bfloat16)
        if not bool(t.isfinite(method.ax.proj.weight).all()) or not bool(method.ax.proj.weight.norm() > 0):
            raise ValueError("DiffMean fit produced a nonfinite or zero direction")
        # Calibration is on the selected training responses, never held-out evaluation generations.
        # batch_size=1 also avoids any padded-token influence in vendor latent summaries.
        tokenizer.padding_side = "right"
        latent = method.predict_latent(frame, batch_size=1, prefix_length=1, return_max_act_only=True)
        scale = max(latent["max_act"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("DiffMean calibration requires a positive finite max activation")
        method.save(fit)
        (fit / "fit.json").write_text(json.dumps({"max_act": scale, "layer": request["layer"],
            "training_rows": len(examples), "prefix_length": 1}) + "\n")
    evidence = json.loads((fit / "fit.json").read_text())
    method.load(fit, mode="steering", intervention_type="addition", low_rank_dimension=1)
    method.ax.to(dtype=t.bfloat16)
    method.max_activations = {0: evidence["max_act"]}
    return method


def run(request: dict) -> None:
    method_name = request["variant"]
    if method_name not in {DIFFMEAN_VARIANT, PROMPT_VARIANT}:
        raise ValueError(f"unsupported AxBench method: {method_name}")
    output = Path(request["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(request["checkpoint_dir"], local_files_only=True, trust_remote_code=False)
    if tokenizer is None:
        raise ValueError("HF export did not provide a tokenizer")
    model = AutoModelForCausalLM.from_pretrained(request["checkpoint_dir"], local_files_only=True,
        trust_remote_code=False, torch_dtype=t.bfloat16, attn_implementation="sdpa").to(request["device"])
    model.eval()
    metadata = json.loads((Path(request["checkpoint_dir"]) / "sft_metadata.json").read_text())
    if not isinstance(tokenizer.chat_template, str) or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != metadata["chat_template_sha256"]:
        raise ValueError("loaded AxBench tokenizer template differs from the HF export contract")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, key) != getattr(model.config, key):
            raise ValueError(f"AxBench model/tokenizer {key} mismatch")
    model.generation_config = GenerationConfig(top_k=request["top_k"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id)
    set_seed(request["seed"])
    if method_name == DIFFMEAN_VARIANT:
        method = fit_direction(request, model, tokenizer, output)
    else:
        method = PromptSteering(model, tokenizer, request["layer"], device=request["device"])
    records = output / "records"
    records.mkdir(exist_ok=True)
    for item in request["rows"]:
        prompt = item["row"]["prompt"]
        if method_name == PROMPT_VARIANT:
            prompt = request["instruction"] + "\n\n" + prompt
        rendered = render(tokenizer, [{"role": "user", "content": prompt}], generation=True,
            limit=int(model.config.max_position_embeddings) - request["max_new_tokens"])
        frame = pd.DataFrame([{"input": rendered, "steered_input": rendered,
                               "concept_id": 0, "factor": request["factor"]}])
        set_seed(transformers_seed(int(item["seed"])))
        result = method.predict_steer(frame, batch_size=1, prefix_length=1,
            eval_output_length=request["max_new_tokens"], temperature=request["temperature"],
            compute_perplexity=False)
        responses = result["steered_generation"]
        if len(responses) != 1 or not isinstance(responses[0], str):
            raise ValueError("AxBench must return exactly one response")
        path = records / f"{item['index']:06d}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite generation: {path}")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**item["row"], "response": responses[0]}, sort_keys=True, indent=2) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("expected one private JSON request path from the Campaign 2 evaluator")
    run(json.loads(Path(sys.argv[1]).read_text()))
