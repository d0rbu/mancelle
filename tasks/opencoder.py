"""OpenCoder SFT stage 2 coding-instruction dataset.

https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2

Stage 2 is published as four train-only configurations.  This task combines
all four configurations into one deterministic conversation source.
"""

import random

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

_DATASET_ID = "OpenCoder-LLM/opc-sft-stage2"
_CONFIGS = (
    "educational_instruct",
    "evol_instruct",
    "mceval_instruct",
    "package_instruct",
)


class OpenCoder(Task):
    """OpenCoder SFT stage 2, combining all four train configurations."""

    def __init__(
        self,
        split: str = "train",
        max_rows: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        assert split == "train", "OpenCoder only provides a train split"
        if max_rows is not None and max_rows < 0:
            raise ValueError(f"max_rows must be non-negative, got {max_rows}")

        examples: list[dict] = []
        skipped = 0
        for config in _CONFIGS:
            raw_ds = load_dataset(_DATASET_ID, config, split="train")
            for row in raw_ds:
                instruction = row.get("instruction")
                output = row.get("output")
                if not isinstance(instruction, str) or not instruction.strip():
                    skipped += 1
                    continue
                if not isinstance(output, str) or not output.strip():
                    skipped += 1
                    continue
                examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": output},
                        ]
                    }
                )

        if skipped and skipped / (skipped + len(examples)) > 0.5:
            raise DataLoadError(
                f"{self.__class__.__name__}: skip rate exceeds 50% ({skipped} skipped)"
            )
        random.Random(42).shuffle(examples)
        if max_rows is not None:
            examples = examples[:max_rows]
        self.examples = examples
        self.length = len(examples)
        logger.info(
            f"OpenCoder SFT stage 2: loaded {self.length} examples, skipped {skipped} rows"
        )

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict:
        return self.examples[index]
