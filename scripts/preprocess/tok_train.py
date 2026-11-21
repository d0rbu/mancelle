"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.

Usage:
    python -m scripts.preprocess.tok_train
    python -m scripts.preprocess.tok_train vocab_size=65536
"""

import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.common import get_base_dir
from nanochat.dataset import DatasetSplit, parquets_iter_batched
from nanochat.report import get_report
from nanochat.tokenizer import RustBPETokenizer


def build_text_iterator(
    cfg: DictConfig,
    batches_iter: Iterable[Iterable[str]],
) -> Iterator[str]:
    """Yield documents subject to per-doc and per-corpus length caps.

    Pure function — accepts an external batches iterable so callers can
    supply real ``parquets_iter_batched`` output or a synthetic generator
    in tests.

    Behaviour:
    1. Flatten each batch into individual docs.
    2. Truncate any doc longer than ``cfg.doc_cap`` characters.
    3. Stop yielding once cumulative emitted-char count exceeds
       ``cfg.max_chars``.

    The truncation comparison ``nchars > cfg.max_chars`` mirrors the
    original module body: the doc whose emission pushes nchars over the
    cap is yielded, and only the *next* doc is suppressed.
    """
    nchars = 0
    for batch in batches_iter:
        for doc in batch:
            doc_text = doc
            if len(doc_text) > cfg.doc_cap:
                doc_text = doc_text[: cfg.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > cfg.max_chars:
                return


def _load_cfg() -> DictConfig:
    """Compose the Hydra config in the module-load pattern."""
    config_dir = str(
        Path(__file__).resolve().parent.parent.parent / "configs" / "data"
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        return compose(config_name="tok_train", overrides=overrides)


def main() -> None:
    cfg = _load_cfg()
    logger.info(f"max_chars: {cfg.max_chars:,}")
    logger.info(f"doc_cap: {cfg.doc_cap:,}")
    logger.info(f"vocab_size: {cfg.vocab_size:,}")

    text_iter = build_text_iterator(cfg, parquets_iter_batched(split=DatasetSplit.TRAIN))

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, cfg.vocab_size)
    train_time = time.time() - t0
    logger.info(f"Training time: {train_time:.2f}s")

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)

    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    # Cache token-id -> UTF-8 byte length for vocab-size-invariant BPB eval.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes_list: list[int] = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    logger.info(f"Saved token_bytes to {token_bytes_path}")

    token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
    get_report().log(
        section="Tokenizer training",
        data=[
            OmegaConf.to_container(cfg, resolve=True),
            {"train_time": train_time},
            {"num_special_tokens": len(special_set)},
            {
                "token_bytes_min": int(token_bytes_nonzero.min().item()),
                "token_bytes_max": int(token_bytes_nonzero.max().item()),
                "token_bytes_mean": token_bytes_nonzero.mean().item(),
                "token_bytes_std": token_bytes_nonzero.std().item(),
            },
        ],
    )


if __name__ == "__main__":
    main()
