"""NVIDIA Nemotron SFT safety dataset.

https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v1
"""

import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

_REPO_ID = "nvidia/Nemotron-SFT-Safety-v1"
_DATA_FILE = "data/train.jsonl"


class NemotronSafety(Task):
    """Nemotron-SFT-Safety-v1 train conversations."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split == "train", "NemotronSafety only provides a train split"
        # Resolve through the hub cache so offline compute nodes can read a
        # login-node prefetch; an hf:// URL needs the live API even when cached.
        data_path = hf_hub_download(_REPO_ID, _DATA_FILE, repo_type="dataset")
        raw_ds = load_dataset("json", data_files=data_path, split="train")

        examples: list[dict] = []
        skipped = 0
        for row in raw_ds:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                skipped += 1
                continue
            normalized: list[dict[str, str]] = []
            valid = True
            for message in messages:
                if not isinstance(message, dict):
                    valid = False
                    break
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    valid = False
                    break
                normalized.append({"role": role, "content": content})
            if not valid:
                skipped += 1
                continue
            examples.append({"messages": normalized})

        if skipped and skipped / (skipped + len(examples)) > 0.5:
            raise DataLoadError(
                f"{self.__class__.__name__}: skip rate exceeds 50% ({skipped} skipped)"
            )
        random.Random(42).shuffle(examples)
        self.examples = examples
        self.length = len(examples)
        logger.info(
            f"NemotronSafety: loaded {self.length} examples, skipped {skipped} rows"
        )

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict:
        return self.examples[index]
