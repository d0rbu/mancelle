"""
Set up a pretrained HuggingFace tokenizer for use with nanochat training.

Instead of training a custom BPE tokenizer (tok_train.py), this script downloads
an existing pretrained tokenizer and prepares it for nanochat's training pipeline.

Usage:
    python -m scripts.preprocess.setup_hf_tokenizer google/gemma-2-2b
    python -m scripts.preprocess.setup_hf_tokenizer meta-llama/Llama-3.1-8B

Note: Some models (e.g. Gemma) require accepting license terms on HuggingFace
and authenticating via `huggingface-cli login` or setting the HF_TOKEN env var.
"""

import os
import sys

import torch
from loguru import logger

from nanochat.common import get_base_dir
from nanochat.tokenizer import TransformersTokenizer
from steering.sft.token_bytes import compute_hf_token_bytes


def main(argv: list[str] | None = None) -> None:
    """Download an HF tokenizer + persist token_bytes for BPB eval."""
    args = sys.argv if argv is None else argv
    if len(args) < 2:
        print(
            "Usage: python -m scripts.preprocess.setup_hf_tokenizer "
            "<model_name_or_path>"
        )
        print(
            "Example: python -m scripts.preprocess.setup_hf_tokenizer "
            "google/gemma-2-2b"
        )
        sys.exit(1)

    model_name = args[1]
    logger.info(f"Setting up HuggingFace tokenizer: {model_name}")

    tokenizer = TransformersTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.get_vocab_size()
    bos_token_id = tokenizer.get_bos_token_id()
    logger.info(f"Vocab size: {vocab_size:,}")
    logger.info(f"BOS token ID: {bos_token_id}")

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)

    test_text = "Hello world! This is a test. Numbers: 123, 4567."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    logger.info(
        "Encode/decode sanity check: "
        f"{'PASS' if test_text in decoded else 'WARN (lossy decode)'}"
    )

    logger.info("Computing token_bytes mapping...")
    token_bytes_list = compute_hf_token_bytes(tokenizer)
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    logger.info(f"Saved token_bytes to {token_bytes_path}")

    token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
    if token_bytes_nonzero.numel() > 0:
        logger.info(
            f"Token bytes stats: min={int(token_bytes_nonzero.min())}, "
            f"max={int(token_bytes_nonzero.max())}, "
            f"mean={token_bytes_nonzero.mean():.2f}, "
            f"std={token_bytes_nonzero.std():.2f}"
        )
    special_ids = set(tokenizer.tokenizer.all_special_ids)
    logger.info(f"Special tokens: {len(special_ids)}")
    logger.info(f"Tokenizer setup complete: {tokenizer_dir}")


if __name__ == "__main__":
    main()
