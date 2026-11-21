"""
PKU-SafeRLHF-10K safety alignment task.
https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K

Each row has a prompt and two responses with safety labels.
We extract the safe response (preferring response_0) and format
as a single-turn user/assistant conversation for SFT.
Rows where neither response is safe are skipped.
"""

import random

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task


class PKUSafeRLHF(Task):
    """PKU-SafeRLHF-10K safety alignment dataset for SFT training."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split == "train", "PKU-SafeRLHF-10K only has a train split"
        raw_ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split=split)

        # Extract safe responses, skipping rows where neither response is safe
        self.examples: list[dict] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for row in raw_ds:
            prompt = row["prompt"]
            if row["is_response_0_safe"]:
                safe_response = row["response_0"]
            elif row["is_response_1_safe"]:
                safe_response = row["response_1"]
            else:
                skip_reasons["no_safe_response"] = skip_reasons.get("no_safe_response", 0) + 1
                skipped += 1
                continue

            # Validate before appending
            assert isinstance(prompt, str) and len(prompt) > 0, (
                f"Prompt must be a non-empty string, got {repr(prompt)[:80]}"
            )
            assert isinstance(safe_response, str) and len(safe_response) > 0, (
                f"Safe response must be a non-empty string, got {repr(safe_response)[:80]}"
            )

            self.examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": safe_response},
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
            f"PKU-SafeRLHF-10K: loaded {len(self.examples)} safe examples, "
            f"skipped {skipped} rows with no safe response"
        )

        # Deterministic shuffle
        rng = random.Random(42)
        rng.shuffle(self.examples)

    def num_examples(self) -> int:
        """Return the number of safe examples in the dataset."""
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
