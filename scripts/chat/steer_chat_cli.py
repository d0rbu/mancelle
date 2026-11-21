"""Interactive chat CLI for encoder-decoder models with SAE feature conditioning.

Supports both EncoderDecoderGPT (with optional feature conditions) and
decoder-only GPT models.  Uses build_engine_for_features() to construct an
InferenceEngine and streams token output via generate_stream().

Usage:
    python -m scripts.chat.steer_chat_cli
    python -m scripts.chat.steer_chat_cli source=encdec feature_conditions='{0: 1, 5: 2}'
    python -m scripts.chat.steer_chat_cli source=sft temperature=0.8
    python -m scripts.chat.steer_chat_cli prompt="What is 2+2?"
"""

import sys
from contextlib import nullcontext
from enum import StrEnum
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nanochat.checkpoint_manager import ModelPhase, ModelSource, load_model
from nanochat.common import autodetect_device_type, compute_init
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.inference_engine import build_engine_for_features
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.target_real_runner import build_context_feature_provider


class CliMode(StrEnum):
    CHAT = "chat"
    COMPLETION = "completion"


def build_chat_prompt(
    conversation_tokens: list[int],
    user_input: str,
    tokenizer,
    user_start: int,
    user_end: int,
    assistant_start: int,
) -> list[int]:
    """Append a user turn with chat template tokens. Mutates and returns *conversation_tokens*."""
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)
    return conversation_tokens


def build_completion_prompt(user_input: str, tokenizer, bos: int) -> list[int]:
    """Build a fresh token sequence for completion mode — BOS + encoded input, no chat tokens."""
    return [bos] + tokenizer.encode(user_input)


def finalize_chat_response(
    conversation_tokens: list[int],
    response_tokens: list[int],
    assistant_end: int,
) -> list[int]:
    """Ensure assistant_end closes the turn, then extend *conversation_tokens*."""
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
    return conversation_tokens


def main() -> None:
    _config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "serving")
    with initialize_config_dir(config_dir=_config_dir, version_base=None):
        _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name="steer_chat_cli", overrides=_overrides)

    # Init compute
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # Load model (supports both "encdec" and decoder-only sources)
    model, tokenizer, meta = load_model(
        ModelSource(cfg.source),
        device,
        phase=ModelPhase.EVAL,
        model_tag=cfg.model_tag,
        step=cfg.step,
        eval_conditioning=EvalConditioning.deferred("steer_chat_cli"),
    )
    reject_feature_clusters_metadata(meta, "steer_chat_cli checkpoint")
    if isinstance(model, EncoderDecoderGPT):
        require_fixed_checkpoint_segment_mode(meta, surface="steer_chat_cli")

    mode = CliMode(cfg.mode)

    # Parse feature conditions from config
    feature_conditions: dict[int, int] | None = None
    if cfg.feature_conditions:
        feature_conditions = {int(k): int(v) for k, v in cfg.feature_conditions.items()}

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    if mode is CliMode.CHAT:
        user_start = tokenizer.encode_special("<user_start>")
        user_end = tokenizer.encode_special("<user_end>")
        assistant_start = tokenizer.encode_special("<assistant_start>")
        assistant_end = tokenizer.encode_special("<assistant_end>")

    # Build inference engine
    context_provider = None
    if isinstance(model, EncoderDecoderGPT):
        cond = cfg.get("conditioning", None)
        cache_dir = None if cond is None else cond.get("context_cache_dir", None)
        if cache_dir is None:
            raise ValueError(
                "Encoder-decoder chat steering requires conditioning.context_cache_dir "
                "with cached features for the exact prompt/conversation tokens."
            )
        OmegaConf.update(
            cfg,
            "model_max_seq_len",
            int(model.dec_config.sequence_len),
            merge=False,
            force_add=True,
        )
        context_provider = build_context_feature_provider(
            cfg,
            cache_dir=cache_dir,
            error_prefix="conditioning.context_cache_dir",
        )
    engine = build_engine_for_features(
        model, tokenizer, device,
        features=feature_conditions,
        context_provider=context_provider,
        target_mode=TargetConditioningMode.from_next_segment_offset(
            resolve_next_segment_conditioning_offset(cfg, meta)
        ),
    )

    source_label = cfg.source.upper()
    mode_label = mode.name
    cond_label = f" | features: {feature_conditions}" if feature_conditions else ""
    print(f"\nSteered Chat [{source_label} | {mode_label}{cond_label}]")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    if mode is CliMode.CHAT:
        print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens: list[int] = [bos]

    while True:
        if cfg.prompt:
            user_input = cfg.prompt
        else:
            prompt_str = "\nUser: " if mode is CliMode.CHAT else "\n> "
            try:
                user_input = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        if mode is CliMode.CHAT:
            build_chat_prompt(
                conversation_tokens,
                user_input,
                tokenizer,
                user_start,
                user_end,
                assistant_start,
            )

            # Stream the response
            response_tokens: list[int] = []
            print("\nAssistant: ", end="", flush=True)
            with autocast_ctx:
                for tok in engine.generate_stream(
                    conversation_tokens,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    seed=42,
                    stop_token_ids={assistant_end, bos},
                ):
                    response_tokens.append(tok)
                    print(tokenizer.decode([tok]), end="", flush=True)
            print()

            finalize_chat_response(conversation_tokens, response_tokens, assistant_end)
        else:
            # Completion mode: raw text, no chat tokens, no history
            input_tokens = build_completion_prompt(user_input, tokenizer, bos)

            print("", end="", flush=True)
            with autocast_ctx:
                for tok in engine.generate_stream(
                    input_tokens,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    seed=42,
                    stop_token_ids={bos},
                ):
                    print(tokenizer.decode([tok]), end="", flush=True)
            print()

        # Single-shot prompt mode
        if cfg.prompt:
            break


if __name__ == "__main__":
    main()
