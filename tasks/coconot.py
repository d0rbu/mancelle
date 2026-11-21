"""
CoCoNot contextual non-compliance task.
https://huggingface.co/datasets/allenai/coconot

AI2's CoCoNot dataset covers nuanced non-compliance scenarios beyond
simple safety refusals:
- Incomplete requests
- Unsupported requests
- Indeterminate/ambiguous requests
- Humanizing requests (treating the model as human)
- Safety concerns

Each row has a prompt and a model response demonstrating appropriate
non-compliance. We format these as single-turn user/assistant
conversations for SFT.
Rows with empty prompt or response content are skipped.
"""

import random

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task


class CoCoNot(Task):
    """CoCoNot contextual non-compliance dataset for SFT training."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"CoCoNot supports 'train' and 'test' splits, got {split!r}"
        )
        raw_ds = load_dataset("allenai/coconot", "original", split=split)

        # Extract prompt/response pairs, skipping rows with empty content
        self.examples: list[dict] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for row in raw_ds:
            prompt = row["prompt"]
            response = row["response"]

            # Skip rows with empty or whitespace-only content
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                skip_reasons["empty_prompt"] = skip_reasons.get("empty_prompt", 0) + 1
                skipped += 1
                continue
            if not isinstance(response, str) or len(response.strip()) == 0:
                skip_reasons["empty_response"] = skip_reasons.get("empty_response", 0) + 1
                skipped += 1
                continue

            self.examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
            )

        if skipped > 0:
            total = skipped + len(self.examples)
            logger.warning(f"Skipped {skipped}/{total} rows. Reasons: {dict(skip_reasons)}")
            if total > 0 and skipped / total > 0.5:
                raise DataLoadError(
                    f"{self.__class__.__name__}: skip rate {skipped/total:.0%} exceeds 50% threshold. "
                    f"Reasons: {dict(skip_reasons)}"
                )
        logger.info(
            f"CoCoNot ({split}): loaded {len(self.examples)} examples, "
            f"skipped {skipped} rows with empty content"
        )

        # Deterministic shuffle
        rng = random.Random(42)
        rng.shuffle(self.examples)

    def num_examples(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def get_example(self, index: int) -> dict:
        """Get a single conversation by index.

        Returns a dict with 'messages' key containing exactly
        one user message and one assistant message.
        """
        assert 0 <= index < len(self.examples), (
            f"Index {index} out of range [0, {len(self.examples)})"
        )
        conversation = self.examples[index]

        # Postcondition: validate conversation format
        messages = conversation["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "user", (
            f"First message must be user, got {messages[0]['role']}"
        )
        assert messages[1]["role"] == "assistant", (
            f"Second message must be assistant, got {messages[1]['role']}"
        )
        assert (
            isinstance(messages[0]["content"], str) and len(messages[0]["content"]) > 0
        )
        assert (
            isinstance(messages[1]["content"], str) and len(messages[1]["content"]) > 0
        )

        return conversation
