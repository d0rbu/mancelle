"""
NoRobots: human-written instruction-following dataset.
https://huggingface.co/datasets/HuggingFaceH4/no_robots

9.5K train, 500 test. 100% human-authored across 10 task categories.
Used as a quality anchor in the safety SFT mixture.

The dataset has a native `messages` format: each row contains a list
of role/content dicts (optional system, then user/assistant alternating).
"""

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task


def _is_valid_conversation(messages: list[dict]) -> bool:
    """Check if a conversation has valid alternating roles and non-empty content."""
    if len(messages) < 1:
        return False

    # Optional system message at the beginning
    if messages[0]["role"] == "system":
        rest = messages[1:]
    else:
        rest = messages

    if len(rest) < 2:
        return False

    for i, msg in enumerate(rest):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            return False
        if not isinstance(msg["content"], str) or len(msg["content"]) == 0:
            return False

    return True


class NoRobots(Task):
    """HuggingFaceH4/no_robots dataset. 9.5K train, 500 test."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"NoRobots supports 'train' and 'test' splits, got {split!r}"
        )
        raw_ds = load_dataset("HuggingFaceH4/no_robots", split=split).shuffle(seed=42)

        # Filter out rows with bad format
        skipped = 0
        skip_reasons: dict[str, int] = {}
        valid_indices = []
        for i in range(len(raw_ds)):
            if _is_valid_conversation(raw_ds[i]["messages"]):
                valid_indices.append(i)
            else:
                skip_reasons["invalid_conversation"] = skip_reasons.get("invalid_conversation", 0) + 1
                skipped += 1

        self.ds = raw_ds.select(valid_indices)
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
            f"NoRobots ({split}): loaded {self.length} examples, skipped {skipped} rows with bad format"
        )

    def num_examples(self) -> int:
        """Return the total number of examples in the dataset."""
        return self.length

    def get_example(self, index: int) -> dict:
        """Return a conversation dict for the given index.

        Format: {"messages": [{"role": ..., "content": ...}, ...]}
        """
        row = self.ds[index]
        return {"messages": row["messages"]}
