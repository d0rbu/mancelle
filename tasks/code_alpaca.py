"""
CodeAlpaca-20k coding instruction dataset.
https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

Same Alpaca format: instruction, input (optional context), output.
When `input` is non-empty it is merged with the instruction to form the
user prompt. The `output` field becomes the assistant response.

The dataset has only a single HuggingFace split ("train"). We carve out
the last 500 rows (after filtering and shuffle) as a held-out "test"
split for val BPB.
"""

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

_VAL_SIZE = 500


class CodeAlpaca(Task):
    """CodeAlpaca-20k coding instruction-following task (~20K examples)."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"CodeAlpaca split must be train|test, got {split!r}"
        )
        raw_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(
            seed=42
        )

        # Filter out rows with empty instruction or output
        valid_indices = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for i in range(len(raw_ds)):
            row = raw_ds[i]
            if not isinstance(row["instruction"], str) or len(row["instruction"].strip()) == 0:
                skip_reasons["empty_instruction"] = skip_reasons.get("empty_instruction", 0) + 1
                skipped += 1
                continue
            if not isinstance(row["output"], str) or len(row["output"].strip()) == 0:
                skip_reasons["empty_output"] = skip_reasons.get("empty_output", 0) + 1
                skipped += 1
                continue
            valid_indices.append(i)

        ds = raw_ds.select(valid_indices)

        if split == "train":
            self.ds = ds.select(range(len(ds) - _VAL_SIZE))
        else:
            self.ds = ds.select(range(len(ds) - _VAL_SIZE, len(ds)))
        self.length = len(self.ds)
        if skipped > 0:
            total = skipped + len(valid_indices)
            logger.warning(
                f"Skipped {skipped}/{total} rows. Reasons: {dict(skip_reasons)}"
            )
            if total > 0 and skipped / total > 0.5:
                raise DataLoadError(
                    f"{self.__class__.__name__}: skip rate {skipped/total:.0%} exceeds 50% threshold. "
                    f"Reasons: {dict(skip_reasons)}"
                )
        logger.info(
            f"CodeAlpaca ({split}): loaded {self.length} examples, skipped {skipped} rows with empty fields"
        )

    def num_examples(self) -> int:
        """Return total number of examples in the dataset."""
        return self.length

    def get_example(self, index: int) -> dict:
        """Return a single example as a conversation dict.

        If the row has a non-empty ``input`` field it is appended to the
        instruction separated by a newline to provide additional context.
        """
        row = self.ds[index]
        instruction = row["instruction"]
        inp = row["input"]
        output = row["output"]

        # Build user prompt: merge input context into instruction when present
        if inp and inp.strip():
            user_content = f"{instruction}\n\n{inp}"
        else:
            user_content = instruction

        conversation: dict = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ],
        }
        return conversation
