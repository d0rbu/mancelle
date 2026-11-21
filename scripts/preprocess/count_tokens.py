"""
Count total tokens for the full training pipeline (pretrain + SFT).

Pretraining tokens are computed arithmetically from scaling laws (no GPU needed).
SFT tokens are computed by iterating the full SFT mixture and tokenizing each conversation.

Usage:
    python -m scripts.preprocess.count_tokens
    python -m scripts.preprocess.count_tokens depth=12
"""

import math
from collections import Counter

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess.convert_sft_to_parquet import (
    prepare_messages_for_full_nanochat_render,
)
from steering._common.build_meta import resolve_gpt_dimensions
from tasks.mixtures import (
    MIXTURE_PARAM_KEYS,
    STEERED_MIXTURES,
    SFTMixture,
    build_mixture,
    select_mixture_overrides,
    steered_mixture_overrides,
)


def build_model_meta(cfg, vocab_size: int) -> GPT:
    """Build a model on meta device (shapes/dtypes only, no data)."""
    model_dim, num_heads = resolve_gpt_dimensions(
        depth=cfg.depth,
        aspect_ratio=cfg.aspect_ratio,
        head_dim=cfg.head_dim,
    )
    config = GPTConfig(
        sequence_len=cfg.max_seq_len,
        vocab_size=vocab_size,
        n_layer=cfg.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=cfg.window_pattern,
    )
    with t.device("meta"):
        model = GPT(config)
    return model


def count_pretrain_tokens(cfg, vocab_size: int) -> dict:
    """Replicate base_train.py scaling law logic to compute pretrain token count."""
    model = build_model_meta(cfg, vocab_size)
    param_counts = model.num_scaling_params()

    # Scaling params = transformer_matrices + lm_head (matches base_train.py:228)
    num_scaling_params = param_counts["transformer_matrices"] + param_counts["lm_head"]
    target_tokens = int(cfg.target_param_data_ratio * num_scaling_params)

    # Reference model for batch size scaling (matches base_train.py:234-236)
    # Override depth for reference — rebuild with depth=12
    d12_cfg = DictConfig({**cfg, "depth": 12})
    d12_ref = build_model_meta(d12_cfg, vocab_size)
    d12_scaling = sum(
        v
        for k, v in d12_ref.num_scaling_params().items()
        if k in ("transformer_matrices", "lm_head")
    )
    D_REF = cfg.target_param_data_ratio * d12_scaling
    B_REF = 2**19  # 524,288 tokens

    # Auto-compute optimal batch size (matches base_train.py:242-245)
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio**0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))

    # Num iterations and total tokens (matches base_train.py:305-311)
    num_iterations = target_tokens // total_batch_size
    total_tokens = total_batch_size * num_iterations

    return {
        "param_counts": param_counts,
        "num_scaling_params": num_scaling_params,
        "target_tokens": target_tokens,
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "total_tokens": total_tokens,
    }


def _label_task(task, kind_seen: Counter[str]) -> str:
    """Return the per_task display label for *task*, including epoch suffixes.

    Mirrors the pre-refactor labelling so downstream log scrapers / report
    consumers stay stable:

    - ``CustomJSON`` -> ``Identity_epochN`` (where N counts CustomJSON
      occurrences in the mixture, 1-indexed).
    - ``MMLU``/``GSM8K``/``WildJailbreak``/``WildGuardMix``/``CoCoNot`` ->
      ``<Class>_epochN`` (1-indexed across each Class's occurrences).
    - ``MetaMathQA`` (with ``gsm_only=True``) -> ``MetaMathQA_GSM``.
    - ``ARC`` -> its subset name (``ARC-Easy`` or ``ARC-Challenge``).
    - Single-occurrence tasks (``SmolTalk``, ``OpenCoder``, ``OrcaMath``,
      ``NemotronSafety``, ``SimpleSpelling``, ``SpellingBee``) -> bare class name.
    """
    cls = type(task).__name__
    epoch_classes = {
        "MMLU",
        "GSM8K",
        "WildJailbreak",
        "WildGuardMix",
        "CoCoNot",
    }
    kind_seen[cls] += 1
    n = kind_seen[cls]
    if cls == "ARC":
        return task.subset
    if cls == "CustomJSON":
        return f"Identity_epoch{n}"
    if cls == "MetaMathQA":
        # The capabilities mixture only ever uses MetaMathQA(gsm_only=True);
        # preserve the historical "_GSM" tag.
        return "MetaMathQA_GSM"
    if cls in epoch_classes:
        return f"{cls}_epoch{n}"
    return cls


def _count_tokens_for_task_list(tokenizer, tasks: list) -> dict:
    """Iterate ``tasks`` and produce a ``{per_task, total_tokens}`` dict.

    Labelling matches the pre-refactor convention; see :func:`_label_task`.
    """
    per_task: dict[str, dict[str, int]] = {}
    grand_total = 0
    kind_seen: Counter[str] = Counter()

    for task in tasks:
        label = _label_task(task, kind_seen)
        task_tokens = 0
        task_len = len(task)
        for i in range(task_len):
            conversation = task[i]
            messages, _ = prepare_messages_for_full_nanochat_render(conversation)
            if messages is None:
                continue
            ids, _ = tokenizer.render_conversation({"messages": messages})
            task_tokens += len(ids)
        per_task[label] = {"rows": task_len, "tokens": task_tokens}
        grand_total += task_tokens
        logger.info(f"  {label}: {task_len:,} rows, {task_tokens:,} tokens")

    return {"per_task": per_task, "total_tokens": grand_total}


def count_sft_tokens(
    tokenizer,
    mmlu_epochs: int = 1,
    gsm8k_epochs: int = 1,
    mixture: SFTMixture = SFTMixture.STANDARD,
    steered_jsonl: str | None = None,
) -> dict:
    """Iterate a STANDARD-family SFT mixture and count tokens per task.

    ``mixture`` is STANDARD or one of its steered members, which share the
    standard knob set plus ``steered_jsonl``.

    Delegates the catalogue of tasks to :func:`tasks.mixtures.build_mixture`
    so this script no longer carries a duplicate inline definition (P1 #1).

    Raises :class:`errors.DataLoadError` (via ``CustomJSON.__init__``)
    if ``identity_conversations.jsonl`` is missing. This is intentional
    — silent skip would mask a real provisioning mistake.
    """
    train_tasks, _val = build_mixture(
        mixture,
        mmlu_epochs=mmlu_epochs,
        gsm8k_epochs=gsm8k_epochs,
        **steered_mixture_overrides(mixture, steered_jsonl),
    )
    return _count_tokens_for_task_list(tokenizer, train_tasks)


def count_capabilities_sft_tokens(tokenizer, cfg) -> dict:
    """Iterate the CAPABILITIES SFT mixture and count tokens per task.

    Delegates the catalogue of tasks to :func:`tasks.mixtures.build_mixture`
    so this script no longer carries a duplicate inline definition (P1 #1).
    """
    train_tasks, _val = build_mixture(
        SFTMixture.CAPABILITIES,
        smoltalk_limit=cfg.smoltalk_limit,
        capabilities_mmlu_epochs=cfg.capabilities_mmlu_epochs,
        opencoder_limit=cfg.opencoder_limit,
        orca_math_limit=cfg.orca_math_limit,
        metamath_limit=cfg.metamath_limit,
        simple_spelling_size=cfg.simple_spelling_size,
        spelling_bee_size=cfg.spelling_bee_size,
    )
    return _count_tokens_for_task_list(tokenizer, train_tasks)


def count_safety_sft_tokens(tokenizer, cfg) -> dict:
    """Iterate the SAFETY SFT mixture and count tokens per task.

    Delegates the catalogue of tasks to :func:`tasks.mixtures.build_mixture`
    so this script no longer carries a duplicate inline definition (P1 #1).

    Raises :class:`errors.DataLoadError` if ``identity_conversations.jsonl``
    is missing (see :func:`count_sft_tokens`).
    """
    train_tasks, _val = build_mixture(
        SFTMixture.SAFETY,
        wildjailbreak_epochs=cfg.wildjailbreak_epochs,
        wildguardmix_epochs=cfg.wildguardmix_epochs,
        coconot_epochs=cfg.coconot_epochs,
        smoltalk_safety_limit=cfg.smoltalk_safety_limit,
    )
    return _count_tokens_for_task_list(tokenizer, train_tasks)


@hydra.main(
    config_path="../../configs/training", config_name="count_tokens", version_base=None
)
def main(cfg: DictConfig):
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocab size: {vocab_size:,}")

    # --- Pretrain tokens ---
    logger.info("=" * 60)
    logger.info("PRETRAINING TOKEN COUNT")
    logger.info("=" * 60)
    pretrain = count_pretrain_tokens(cfg, vocab_size)
    logger.info(f"Model depth: {cfg.depth}")
    logger.info("Parameter counts:")
    for key, value in pretrain["param_counts"].items():
        logger.info(f"  {key:24s}: {value:,}")
    logger.info(
        f"Scaling params (transformer + lm_head): {pretrain['num_scaling_params']:,}"
    )
    logger.info(f"Target param:data ratio: {cfg.target_param_data_ratio}")
    logger.info(f"Auto-computed batch size: {pretrain['total_batch_size']:,}")
    logger.info(f"Num iterations: {pretrain['num_iterations']:,}")
    logger.info(f"Pretrain tokens: {pretrain['total_tokens']:,}")

    # --- SFT tokens ---
    logger.info("=" * 60)
    logger.info(f"SFT TOKEN COUNT (mixture={cfg.mixture})")
    logger.info("=" * 60)
    if cfg.mixture == SFTMixture.SAFETY:
        sft = count_safety_sft_tokens(tokenizer, cfg)
    elif cfg.mixture == SFTMixture.CAPABILITIES:
        sft = count_capabilities_sft_tokens(tokenizer, cfg)
    elif cfg.mixture == SFTMixture.CHATCORE:
        train_tasks, _val = build_mixture(
            SFTMixture.CHATCORE,
            **select_mixture_overrides(
                SFTMixture.CHATCORE, {key: cfg[key] for key in MIXTURE_PARAM_KEYS}
            ),
        )
        sft = _count_tokens_for_task_list(tokenizer, train_tasks)
    elif cfg.mixture in {SFTMixture.STANDARD} | STEERED_MIXTURES:
        sft = count_sft_tokens(
            tokenizer,
            mmlu_epochs=cfg.mmlu_epochs,
            gsm8k_epochs=cfg.gsm8k_epochs,
            mixture=SFTMixture(cfg.mixture),
            steered_jsonl=cfg.steered_jsonl,
        )
    else:
        raise ValueError(f"unknown mixture {cfg.mixture!r}")
    logger.info(f"SFT tokens: {sft['total_tokens']:,}")

    # --- Grand total ---
    grand_total = pretrain["total_tokens"] + sft["total_tokens"]
    logger.info("=" * 60)
    logger.info("GRAND TOTAL")
    logger.info("=" * 60)
    logger.info(f"Pretrain:    {pretrain['total_tokens']:>15,}")
    logger.info(f"SFT:         {sft['total_tokens']:>15,}")
    logger.info(f"Grand total: {grand_total:>15,}")
    logger.info(f"             ({grand_total / 1e9:.2f}B tokens)")


if __name__ == "__main__":
    main()
