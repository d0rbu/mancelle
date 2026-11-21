"""
Orca-Math: 200K grade school math word problems (synthetic).
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

Purpose-built training data from Microsoft — not a benchmark.
Complements GSM8K with a much larger set of math reasoning examples.

The dataset has only a single HuggingFace split ("train"). We carve out
the last 1000 rows (after shuffle) as a held-out "test" split for val BPB.
"""

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

_VAL_SIZE = 1000


class OrcaMath(Task):
    """microsoft/orca-math-word-problems-200k dataset (~200K examples)."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"OrcaMath split must be train|test, got {split!r}"
        )
        raw_ds = load_dataset(
            "microsoft/orca-math-word-problems-200k", split="train"
        ).shuffle(seed=42)

        # Filter out rows with empty question or answer
        valid_indices = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for i in range(len(raw_ds)):
            row = raw_ds[i]
            if not isinstance(row["question"], str) or len(row["question"].strip()) == 0:
                skip_reasons["empty_question"] = skip_reasons.get("empty_question", 0) + 1
                skipped += 1
                continue
            if not isinstance(row["answer"], str) or len(row["answer"].strip()) == 0:
                skip_reasons["empty_answer"] = skip_reasons.get("empty_answer", 0) + 1
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
            f"OrcaMath ({split}): loaded {self.length} examples, skipped {skipped} rows with empty fields"
        )

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict:
        row = self.ds[index]
        conversation: dict = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        }
        return conversation
