"""
WildGuardMix safety moderation SFT task.
https://huggingface.co/datasets/allenai/wildguardmix

87K train examples with prompt-harm and response-refusal labels.
We filter to only include rows that have both a prompt AND a non-empty
response (the dataset also contains prompt-only classification rows
which we skip).

~38K examples with responses after filtering.
"""

import random

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task


class WildGuardMix(Task):
    """WildGuardMix safety moderation dataset for SFT training.

    Loads the allenai/wildguardmix dataset, filtering to rows that have
    a non-empty response field (skipping prompt-only classification rows).
    """

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"split must be 'train' or 'test', got {split!r}"
        )

        # Map split to the correct dataset config name
        config = "wildguardtrain" if split == "train" else "wildguardtest"
        ds_split = split if split == "test" else "train"
        raw_ds = load_dataset("allenai/wildguardmix", config, split=ds_split)

        # Filter to rows with non-empty responses, skipping prompt-only rows
        self.examples: list[dict] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for row in raw_ds:
            prompt = row["prompt"]
            response = row["response"]

            # Skip prompt-only classification rows
            if response is None or (
                isinstance(response, str) and len(response.strip()) == 0
            ):
                skip_reasons["empty_response"] = skip_reasons.get("empty_response", 0) + 1
                skipped += 1
                continue

            # Validate before appending
            assert isinstance(prompt, str) and len(prompt) > 0, (
                f"Prompt must be a non-empty string, got {repr(prompt)[:80]}"
            )
            assert isinstance(response, str) and len(response) > 0, (
                f"Response must be a non-empty string, got {repr(response)[:80]}"
            )

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
            if total > 0 and skipped / total > 0.6:
                raise DataLoadError(
                    f"{self.__class__.__name__}: skip rate {skipped/total:.0%} exceeds 60% threshold. "
                    f"Reasons: {dict(skip_reasons)}"
                )
        logger.info(
            f"WildGuardMix ({split}): loaded {len(self.examples)} examples "
            f"with responses, skipped {skipped} prompt-only rows"
        )

        # Deterministic shuffle
        rng = random.Random(42)
        rng.shuffle(self.examples)

    def num_examples(self) -> int:
        """Return the number of examples with responses."""
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
