"""
MetaMathQA: 395K augmented math questions (synthetic).
https://huggingface.co/datasets/meta-math/MetaMathQA

Augmented from GSM8K and MATH *training* sets — not a benchmark.
Includes rephrased questions, variable substitution, and answer augmentation.

The dataset has only a single HuggingFace split ("train"). We carve out
the last 1000 rows (after shuffle) as a held-out "test" split for val BPB.
"""

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

_VAL_SIZE = 1000


class MetaMathQA(Task):
    """meta-math/MetaMathQA dataset (~395K examples)."""

    def __init__(
        self, split: str = "train", gsm_only: bool = False, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"MetaMathQA split must be train|test, got {split!r}"
        )
        raw_ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42)

        # Filter out rows with empty fields or non-GSM types (if gsm_only)
        valid_indices = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for i in range(len(raw_ds)):
            row = raw_ds[i]
            if (
                not isinstance(row["query"], str)
                or len(row["query"].strip()) == 0
                or not isinstance(row["response"], str)
                or len(row["response"].strip()) == 0
            ):
                skip_reasons["empty_query_or_response"] = skip_reasons.get("empty_query_or_response", 0) + 1
                skipped += 1
                continue
            if gsm_only:
                row_type = row.get("type")
                if row_type is None:
                    skip_reasons["missing_type"] = skip_reasons.get("missing_type", 0) + 1
                    skipped += 1
                    continue
                if not row_type.startswith("GSM"):
                    skip_reasons["non_gsm_type"] = skip_reasons.get("non_gsm_type", 0) + 1
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
            f"MetaMathQA ({split}): loaded {self.length} examples, skipped {skipped} rows with empty fields"
        )

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict:
        row = self.ds[index]
        conversation: dict = {
            "messages": [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]},
            ],
        }
        return conversation
