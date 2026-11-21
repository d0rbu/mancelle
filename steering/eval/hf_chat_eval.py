"""Transformers inference for the two public instruction-tuned ChatCORE baselines.

Task construction and aggregate scoring belong to scripts.eval.chat_eval. This
module owns only the HF contract and per-task inference; it does not emulate
nanochat token identities, caches, or calculator execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO, cast

import torch as t
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from steering._common.ddp import rank_synchronized_stage
from steering.eval.hf_chat_generation import (
    load_repository_hf_checkpoint,
    resolve_repository_hf_checkpoint_dir,
)
from steering.sft.hf_decoder import validate_hf_source_config

SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen2",
    "HuggingFaceTB/SmolLM3-3B": "smollm3",
}
_REPOSITORY_EXPORT_LABEL = "repository_hf_sft"
_ENGLISH_MONTHS = (
    "January February March April May June July August September October November December"
).split()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _repository_export_path(hf_path: Any) -> Path | None:
    """Resolve an existing local repository export; leave Hub IDs untouched."""
    if not isinstance(hf_path, str):
        return None
    candidate = Path(hf_path).expanduser()
    if not candidate.exists():
        return None
    return resolve_repository_hf_checkpoint_dir(candidate)


def validate_chat_eval_selection(cfg: DictConfig, task_names: Sequence[str]) -> bool:
    """Validate disjoint source selection; return whether HF inference is selected."""
    source, hf_path = cfg.get("source"), cfg.get("hf_path")
    if (source is None) == (hf_path is None):
        raise ValueError("exactly one of source or hf_path must be supplied")
    if hf_path is None:
        if any(
            cfg.get(key) is not None for key in ("hf_revision", "hf_template_date", "output_dir")
        ):
            raise ValueError("hf_revision, hf_template_date and output_dir require hf_path")
        return False
    repository_export = _repository_export_path(hf_path)
    if repository_export is None and (
        not isinstance(hf_path, str) or hf_path not in SUPPORTED_MODELS
    ):
        raise ValueError(
            f"hf_path must be one of {list(SUPPORTED_MODELS)} or one concrete local "
            "repository HF step_<six digits> export"
        )
    revision = cfg.get("hf_revision")
    if repository_export is not None and revision is not None:
        raise ValueError("local repository HF exports require hf_revision=null")
    if repository_export is None and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("hf_revision must be a full lowercase 40-character commit SHA")
    for name in ("model_tag", "step"):
        if cfg.get(name) is not None:
            raise ValueError(f"HF ChatCORE requires {name}=null")
    for name in ("feature_configs", "target_real_configs"):
        if cfg.get(name) not in (None, []):
            raise ValueError(f"HF ChatCORE does not support {name}")
    for name in ("problem_start", "segment_mode", "extraction_tokenizer_id", "stats_path",
                 "target_real_output_dir"):
        if cfg.get(name) is not None:
            raise ValueError(f"HF ChatCORE does not support {name}")
    if cfg.get("next_segment_conditioning_offset", "auto") != "auto":
        raise ValueError("HF ChatCORE requires next_segment_conditioning_offset=auto")
    if cfg.dtype not in ("float32", "bfloat16"):
        raise ValueError("HF ChatCORE dtype must be float32 or bfloat16")
    if repository_export is not None and cfg.dtype != "bfloat16":
        raise ValueError("local repository HF exports require dtype=bfloat16")
    if not isinstance(cfg.get("output_dir"), str) or not cfg.output_dir.strip():
        raise ValueError("HF ChatCORE requires a fresh output_dir")
    if Path(cfg.output_dir).expanduser().exists():
        raise ValueError("HF ChatCORE output_dir already exists; choose a fresh directory")
    if cfg.task_name is not None and not isinstance(cfg.task_name, str):
        raise ValueError("task_name must be null or a pipe-separated string")
    selected_tasks = list(task_names) if cfg.task_name is None else cfg.task_name.split("|")
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("task_name must select distinct tasks")
    if any(name not in task_names for name in selected_tasks):
        raise ValueError(f"task_name must use exact names separated by '|': {list(task_names)}")
    for name in ("batch_size", "num_samples", "max_new_tokens"):
        _positive_int(cfg[name], name)
    for name in ("top_k", "max_problems"):
        if cfg[name] is not None:
            _positive_int(cfg[name], name)
    if (
        isinstance(cfg.temperature, bool)
        or not isinstance(cfg.temperature, (int, float))
        or not math.isfinite(cfg.temperature)
        or cfg.temperature < 0
    ):
        raise ValueError("temperature must be a finite non-negative number")
    template_date = cfg.get("hf_template_date")
    if repository_export is None and SUPPORTED_MODELS[hf_path] == "smollm3":
        if (
            not isinstance(template_date, str)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}", template_date) is None
        ):
            raise ValueError("SmolLM3 requires hf_template_date as YYYY-MM-DD")
        date.fromisoformat(template_date)
    elif template_date is not None:
        raise ValueError("hf_template_date is only supported for SmolLM3")
    return True


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _token_id(value: Any, vocab_size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
        raise ValueError(f"{name} must be a token ID within the model vocabulary, got {value!r}")
    return value


@dataclass
class HFChatEvalRuntime:
    """One loaded HF model and its explicit, reproducible evaluation contract."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    template_date: str | None
    identity: dict[str, Any]
    records: TextIO
    task_counts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Transformers 5.14 fills unset GenerationConfig fields from the model's
        # saved defaults. Keep only explicit request settings and global defaults.
        self.model.generation_config = GenerationConfig()
        self.model.eval()

    @property
    def device(self) -> t.device:
        return next(self.model.parameters()).device

    def _template_clock(self, format_string: str) -> str:
        if format_string != "%d %B %Y" or self.template_date is None:
            raise ValueError("unexpected clock format in the pinned SmolLM3 template")
        value = date.fromisoformat(self.template_date)
        return f"{value.day:02d} {_ENGLISH_MONTHS[value.month - 1]} {value.year}"

    def render(self, conversation: Mapping[str, Any]) -> tuple[str, list[int]]:
        messages = conversation.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or not all(isinstance(message, Mapping) for message in messages)
            or messages[-1].get("role") != "assistant"
        ):
            raise ValueError("ChatCORE conversation must end with a reference assistant turn")
        prefix = messages[:-1]
        if (
            any(
                message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in prefix
            )
            or prefix[-1]["role"] != "user"
        ):
            raise ValueError("HF ChatCORE requires text chat messages ending in a user turn")
        kwargs: dict[str, Any] = {}
        if self.model.config.model_type == "smollm3":
            kwargs = {"enable_thinking": False, "strftime_now": self._template_clock}
        rendered = self.tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        # This is also apply_chat_template(tokenize=True)'s tokenization path.
        # Keep the text for validating categorical continuation boundaries.
        if not isinstance(rendered, str):
            raise ValueError("HF template must render text")
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not isinstance(ids, list) or not ids:
            raise ValueError("HF template must produce one nonempty token sequence")
        ids = [_token_id(token_id, int(self.model.config.vocab_size), "prompt") for token_id in ids]
        return rendered, ids

    def _check_context(self, prompt: list[int], new_tokens: int = 0) -> None:
        limit = int(self.model.config.max_position_embeddings)
        if not prompt or len(prompt) + new_tokens > limit:
            raise ValueError(
                f"HF ChatCORE context overflow: prompt={len(prompt)}, "
                f"max_new_tokens={new_tokens}, architecture_limit={limit}"
            )

    def _batch(self, prompts: list[list[int]]) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        for prompt in prompts:
            self._check_context(prompt)
        width = max(map(len, prompts))
        ids = t.full((len(prompts), width), self.pad_token_id, dtype=t.long, device=self.device)
        mask = t.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, -len(prompt) :] = t.tensor(prompt, dtype=t.long, device=self.device)
            mask[row, -len(prompt) :] = 1
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        return ids, mask, positions

    @t.inference_mode()
    def score_choices(
        self,
        prompts: list[list[int]],
        answer_ids: list[list[int]],
    ) -> list[list[float]]:
        if not prompts or len(prompts) != len(answer_ids) or any(not row for row in answer_ids):
            raise ValueError("categorical prompts and nonempty answer rows must align")
        for row in answer_ids:
            for token_id in row:
                _token_id(token_id, int(self.model.config.vocab_size), "answer")
        ids, mask, positions = self._batch(prompts)
        logits = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            logits_to_keep=1,
        ).logits
        if tuple(logits.shape) != (len(prompts), 1, int(self.model.config.vocab_size)):
            raise ValueError("HF categorical forward returned an unexpected logits shape")
        selected = [logits[row, 0, choices].float() for row, choices in enumerate(answer_ids)]
        if any(not bool(t.isfinite(scores).all()) for scores in selected):
            raise FloatingPointError("non-finite HF categorical answer logits")
        return [scores.tolist() for scores in selected]

    @t.inference_mode()
    def generate(
        self,
        prompt: list[int],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        self._check_context(prompt, max_new_tokens)
        ids = t.tensor([prompt], dtype=t.long, device=self.device)
        sampling = temperature > 0
        sample_count = num_samples if sampling else 1
        kwargs: dict[str, Any] = {}
        if sampling:
            kwargs = {
                "temperature": temperature,
                "top_k": 0 if top_k is None else top_k,
                "top_p": 1.0,
            }
        devices = [self.device.index] if self.device.type == "cuda" else []
        with t.random.fork_rng(devices=devices):
            t.random.default_generator.manual_seed(42)
            if self.device.type == "cuda":
                with t.cuda.device(self.device):
                    t.cuda.manual_seed(42)
            generated = cast(Any, self.model).generate(
                input_ids=ids,
                attention_mask=t.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                num_return_sequences=sample_count,
                use_cache=True,
                eos_token_id=list(self.eos_token_ids),
                pad_token_id=self.pad_token_id,
                **kwargs,
            )
        if (
            not isinstance(generated, t.Tensor)
            or generated.ndim != 2
            or generated.shape[0] != sample_count
            or not len(prompt) <= generated.shape[1] <= len(prompt) + max_new_tokens
            or not t.equal(generated[:, : len(prompt)], ids.expand(sample_count, -1))
        ):
            raise ValueError("HF generate did not preserve the expected prompt prefix and shape")
        completions = []
        for row in generated[:, len(prompt) :].tolist():
            stop = next((i for i, token in enumerate(row) if token in self.eos_token_ids), len(row))
            tokens = row[:stop]
            text = self.tokenizer.decode(
                tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            completions.append(
                {
                    "token_ids": tokens,
                    "text": text,
                    "stop_token_id": row[stop] if stop < len(row) else None,
                    "length_limited": stop == len(row),
                }
            )
        return completions if sampling else completions * num_samples

    def evaluate(
        self,
        task: Any,
        task_name: str,
        *,
        batch_size: int,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        max_problems: int | None,
    ) -> float:
        distributed = dist.is_initialized()
        rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        passed = total = 0
        with rank_synchronized_stage(f"HF {task_name} inference", device=self.device):
            total_rows = len(task) if max_problems is None else min(len(task), max_problems)
            task_identity = {
                "rows": total_rows,
                "dataset_fingerprint": getattr(getattr(task, "ds", None), "_fingerprint", None),
                "word_list_sha256": _digest(task.words) if hasattr(task, "words") else None,
            }
            if total_rows <= 0:
                raise ValueError(f"HF task {task_name} has no examples")
            if task.eval_type not in {"categorical", "generative"}:
                raise ValueError(f"unsupported HF task eval_type {task.eval_type!r}")
            size = batch_size if task.eval_type == "categorical" else 1
            for start in range(rank * size, total_rows, world * size):
                indices = list(range(start, min(start + size, total_rows)))
                conversations = [task[index] for index in indices]
                rendered = [self.render(conversation) for conversation in conversations]
                prompts = [item[1] for item in rendered]
                records: list[dict[str, Any]] = [
                    {
                        "task": task_name,
                        "index": index,
                        "conversation_sha256": _digest(conversation),
                        "prompt_token_ids": prompt,
                        "prompt_sha256": _digest(prompt),
                    }
                    for index, conversation, prompt in zip(
                        indices, conversations, prompts, strict=True
                    )
                ]
                if task.eval_type == "categorical":
                    answers = []
                    for conversation, (text, prompt) in zip(conversations, rendered, strict=True):
                        row = []
                        for label in conversation["letters"]:
                            encoded = self.tokenizer.encode(label, add_special_tokens=False)
                            if (
                                len(encoded) != 1
                                or self.tokenizer.encode(text + label, add_special_tokens=False)
                                != prompt + encoded
                            ):
                                raise ValueError(
                                    f"categorical label {label!r} is not one continuation token"
                                )
                            row.append(encoded[0])
                        answers.append(row)
                    scores = self.score_choices(prompts, answers)
                    for record, conversation, row_scores in zip(
                        records, conversations, scores, strict=True
                    ):
                        winner = max(range(len(row_scores)), key=row_scores.__getitem__)
                        prediction = conversation["letters"][winner]
                        record.update(
                            choices=list(conversation["letters"]),
                            choice_logits=row_scores,
                            prediction=prediction,
                        )
                        record["passed"] = bool(task.evaluate(conversation, prediction))
                else:
                    completions = self.generate(
                        prompts[0],
                        num_samples=num_samples,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    outcomes = [
                        bool(task.evaluate(conversations[0], item["text"])) for item in completions
                    ]
                    records[0].update(
                        completions=completions, outcomes=outcomes, passed=any(outcomes)
                    )
                for record in records:
                    self.records.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    total += 1
                    passed += int(record["passed"])
                self.records.flush()
        if distributed:
            identities: list[Any] = [None] * world
            dist.all_gather_object(identities, task_identity)
            if any(identity != task_identity for identity in identities):
                raise ValueError(f"HF task identity differs across ranks: {task_name}")
        counts = t.tensor([passed, total], dtype=t.long, device=self.device)
        if distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        passed, total = map(int, counts.tolist())
        if total != total_rows:
            raise RuntimeError(
                f"HF task coverage mismatch: expected {total_rows}, evaluated {total}"
            )
        self.task_counts[task_name] = {**task_identity, "passed": passed, "total": total}
        logger.info(f"HF {task_name}: {passed}/{total}")
        return passed / total


def load_hf_chat_eval_runtime(
    cfg: DictConfig, device: t.device, records: TextIO
) -> HFChatEvalRuntime:
    """Load a pinned public baseline or strict local repository SFT export."""
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("HF ChatCORE supports CPU and CUDA")
    repository_export = _repository_export_path(cfg.hf_path)
    if repository_export is not None:
        if cfg.get("hf_revision") is not None:
            raise ValueError("local repository HF exports require hf_revision=null")
        if cfg.dtype != "bfloat16":
            raise ValueError("local repository HF exports require dtype=bfloat16")
        checkpoint = load_repository_hf_checkpoint(repository_export, device=device)
        contract = validate_hf_source_config(
            checkpoint.model.config,
            model_id=checkpoint.source_model_id,
            local_source=True,
        )
        config = checkpoint.model.config
        eos_id = _token_id(config.eos_token_id, config.vocab_size, "EOS")
        pad_id = _token_id(config.pad_token_id, config.vocab_size, "PAD")
        identity = {
            **checkpoint.model_identity,
            "source_model_id": checkpoint.source_model_id,
            "source_revision": checkpoint.source_revision,
            "family": contract.family.value,
            "model_config_sha256": _digest(config.to_dict()),
            "max_position_embeddings": config.max_position_embeddings,
            "dtype": cfg.dtype,
            "attention_implementation": "sdpa",
            "calculator": False,
            "seed_per_example": 42,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "torch": str(t.__version__),
            "transformers": version("transformers"),
        }
        if checkpoint.model_identity.get("format") != _REPOSITORY_EXPORT_LABEL:
            raise ValueError("local HF ChatCORE source is not a repository SFT export")
        return HFChatEvalRuntime(
            checkpoint.model,
            checkpoint.tokenizer,
            (eos_id,),
            pad_id,
            None,
            identity,
            records,
        )
    kwargs: dict[str, Any] = {"revision": cfg.hf_revision, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(cfg.hf_path, **kwargs)
    if config.model_type != SUPPORTED_MODELS[cfg.hf_path]:
        raise ValueError("HF config model_type disagrees with the selected public baseline")
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(cfg.hf_path, **kwargs))
    generation = GenerationConfig.from_pretrained(cfg.hf_path, **kwargs)
    eos = generation.eos_token_id
    eos_ids = tuple(eos if isinstance(eos, list) else [eos])
    if not eos_ids:
        raise ValueError("HF ChatCORE requires at least one EOS token")
    for token in eos_ids:
        _token_id(token, config.vocab_size, "EOS")
    pad_id = _token_id(generation.pad_token_id, config.vocab_size, "PAD")
    if not 0 < len(tokenizer) <= config.vocab_size:
        raise ValueError("HF tokenizer vocabulary exceeds the model vocabulary")
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
        raise ValueError("HF ChatCORE requires one saved chat template")
    _positive_int(config.max_position_embeddings, "max_position_embeddings")
    dtype = t.float32 if cfg.dtype == "float32" else t.bfloat16
    loaded = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path,
        config=config,
        dtype=dtype,
        attn_implementation="sdpa",
        use_safetensors=True,
        output_loading_info=True,
        **kwargs,
    )
    model, loading_info = cast(tuple[PreTrainedModel, dict[str, Any]], loaded)
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(name):
            raise ValueError(f"HF checkpoint did not load exactly: {name}={loading_info[name]}")
    if list(config.architectures or []) != [type(model).__name__]:
        raise ValueError("loaded HF architecture disagrees with the checkpoint config")
    if any(
        cast(t.Tensor, embedding.weight).shape[0] != config.vocab_size
        for embedding in (model.get_input_embeddings(), model.get_output_embeddings())
    ):
        raise ValueError("HF embedding vocabulary does not match config")
    if any(
        parameter.dtype != dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    ):
        raise ValueError("HF model weights do not match the requested dtype")
    t.nn.Module.to(model, device=device)
    identity = {
        "model_id": cfg.hf_path,
        "revision": cfg.hf_revision,
        "architecture": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_vocab_size": config.vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "generation_use_cache": True,
        "model_config_sha256": _digest(config.to_dict()),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            cast(Any, tokenizer).backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "template_date": cfg.get("hf_template_date"),
        "enable_thinking": False if config.model_type == "smollm3" else None,
        "eos_token_ids": list(eos_ids),
        "pad_token_id": pad_id,
        "max_position_embeddings": config.max_position_embeddings,
        "dtype": cfg.dtype,
        "attention_implementation": "sdpa",
        "calculator": False,
        "seed_per_example": 42,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "torch": str(t.__version__),
        "transformers": version("transformers"),
    }
    return HFChatEvalRuntime(
        model,
        tokenizer,
        cast(tuple[int, ...], eos_ids),
        pad_id,
        cfg.get("hf_template_date"),
        identity,
        records,
    )
