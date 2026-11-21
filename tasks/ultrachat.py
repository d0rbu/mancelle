"""
UltraChat 200K: high-quality multi-turn conversational dataset.
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

207K train examples already in native messages format. Used as the
helpfulness component of the safety SFT mixture.
"""

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

# Map canonical split names to the actual HF split names
_SPLIT_MAP = {
    "train": "train_sft",
    "test": "test_sft",
}


class UltraChat(Task):
    """HuggingFaceH4/ultrachat_200k multi-turn conversation dataset."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in _SPLIT_MAP, f"UltraChat split must be train|test, got {split}"
        hf_split = _SPLIT_MAP[split]
        raw_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=hf_split).shuffle(
            seed=42
        )

        # Filter out rows with empty message content
        skipped = 0
        skip_reasons: dict[str, int] = {}
        valid_indices = []
        for i in range(len(raw_ds)):
            messages = raw_ds[i]["messages"]
            if len(messages) < 2:
                skip_reasons["too_few_messages"] = skip_reasons.get("too_few_messages", 0) + 1
                skipped += 1
                continue
            if any(
                not isinstance(m["content"], str) or len(m["content"]) == 0
                for m in messages
            ):
                skip_reasons["invalid_message_content"] = skip_reasons.get("invalid_message_content", 0) + 1
                skipped += 1
                continue
            valid_indices.append(i)

        self.ds = raw_ds.select(valid_indices)
        self.length = len(self.ds)
        if skipped > 0:
            total = skipped + len(valid_indices)
            logger.warning(f"Skipped {skipped}/{total} rows. Reasons: {dict(skip_reasons)}")
            if total > 0 and skipped / total > 0.5:
                raise DataLoadError(
                    f"{self.__class__.__name__}: skip rate {skipped/total:.0%} exceeds 50% threshold. "
                    f"Reasons: {dict(skip_reasons)}"
                )
        logger.info(
            f"UltraChat: loaded {self.length} examples from split '{hf_split}', skipped {skipped} rows with empty content"
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
