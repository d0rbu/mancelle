"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat.chat_cli
python -m scripts.chat.chat_cli source=rl temperature=0.8
"""

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from nanochat.checkpoint_manager import ModelPhase, ModelSource, load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata


def finalize_chat_response(
    conversation_tokens: list[int],
    response_tokens: list[int],
    assistant_end: int,
) -> list[int]:
    """Ensure assistant_end closes the turn, then extend *conversation_tokens*.

    Guards against the IndexError that occurred when the first yielded token
    is a stop token (response_tokens is empty). Mirrors the idiom from
    steer_chat_cli.py::finalize_chat_response.
    """
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
    return conversation_tokens


def main() -> None:
    _config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "serving")
    with initialize_config_dir(config_dir=_config_dir, version_base=None):
        _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name="chat_cli", overrides=_overrides)

    # Init the model and tokenizer

    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )
    model, tokenizer, meta = load_model(
        ModelSource(cfg.source),
        device,
        phase=ModelPhase.EVAL,
        model_tag=cfg.model_tag,
        step=cfg.step,
        eval_conditioning=EvalConditioning.deferred("chat_cli"),
    )
    reject_feature_clusters_metadata(meta, "chat_cli checkpoint")

    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = (
        tokenizer.encode_special("<user_start>"),
        tokenizer.encode_special("<user_end>"),
    )
    assistant_start, assistant_end = (
        tokenizer.encode_special("<assistant_start>"),
        tokenizer.encode_special("<assistant_end>"),
    )

    # Create Engine for efficient generation
    engine = Engine(model, tokenizer)

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]

    while True:
        if cfg.prompt:
            # Get the prompt from the launch command
            user_input = cfg.prompt
        else:
            # Get the prompt interactively from the console
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        # Add User message to the conversation
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        # Kick off the assistant
        conversation_tokens.append(assistant_start)
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 256,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
        }
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)
        with autocast_ctx:
            for token_column, _token_masks in engine.generate(
                conversation_tokens, **generate_kwargs
            ):
                token = token_column[0]  # pop the batch dimension (num_samples=1)
                # Keep and display the complete Engine stream. Forced tool
                # output is part of the visible transcript and next-turn state.
                response_tokens.append(token)
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)
        print()
        # we have to ensure that the assistant end token is the last token
        # so even if generation ends due to max tokens, we have to append it to the end
        finalize_chat_response(conversation_tokens, response_tokens, assistant_end)

        # In the prompt mode, we only want a single response and exit
        if cfg.prompt:
            break


if __name__ == "__main__":
    main()
