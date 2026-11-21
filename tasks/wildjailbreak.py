"""
WildJailbreak adversarial refusal training dataset by AI2.
https://huggingface.co/datasets/allenai/wildjailbreak

~262K examples across 4 data types:
- vanilla_harmful:      direct harmful prompts + refusal responses
- vanilla_benign:       benign prompts + compliant responses
- adversarial_harmful:  jailbreak-style harmful prompts + refusal responses
- adversarial_benign:   benign prompts disguised as harmful + compliant responses

All 4 types are included to train both refusal AND prevent over-refusal.

Columns:
  - vanilla (str): the vanilla prompt
  - adversarial (str): the adversarial prompt (empty for vanilla data points)
  - tactics (list): jailbreak tactics used
  - completion (str): model response
  - data_type (str): one of the 4 types above

For adversarial data types, we prefer the adversarial prompt over the vanilla
one if it is non-empty, since it is the more challenging version.
"""

import random

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError
from tasks.common import Task

EXPECTED_DATA_TYPES = {
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
}


class WildJailbreak(Task):
    """WildJailbreak adversarial refusal dataset for SFT training."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "eval"), "WildJailbreak split must be train|eval"

        # Dataset is tab-delimited; keep_default_na=False prevents pandas
        # from converting certain strings (like "NA") to NaN.
        raw_ds = load_dataset(
            "allenai/wildjailbreak",
            split,
            delimiter="\t",
            keep_default_na=False,
            split="train",
        )

        self.examples: list[dict] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}

        for row in raw_ds:
            data_type = row["data_type"]
            assert data_type in EXPECTED_DATA_TYPES, (
                f"Unexpected data_type: {data_type!r}"
            )

            # Pick the best prompt: prefer adversarial if non-empty
            adversarial = row.get("adversarial")
            vanilla = row.get("vanilla")
            if adversarial is None and vanilla is None:
                skip_reasons["missing_prompt_fields"] = skip_reasons.get("missing_prompt_fields", 0) + 1
                skipped += 1
                continue
            prompt = (adversarial if adversarial and adversarial.strip() else vanilla) or ""
            completion = row.get("completion", "")

            # Skip rows with empty prompt
            if not prompt or not prompt.strip():
                skip_reasons["empty_prompt"] = skip_reasons.get("empty_prompt", 0) + 1
                skipped += 1
                continue
            # Skip rows with empty completion
            if not completion or not completion.strip():
                skip_reasons["empty_completion"] = skip_reasons.get("empty_completion", 0) + 1
                skipped += 1
                continue

            # Validate types
            assert isinstance(prompt, str) and len(prompt) > 0, (
                f"Prompt must be a non-empty string, got {repr(prompt)[:80]}"
            )
            assert isinstance(completion, str) and len(completion) > 0, (
                f"Completion must be a non-empty string, got {repr(completion)[:80]}"
            )

            self.examples.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "data_type": data_type,
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
            f"WildJailbreak ({split}): loaded {len(self.examples)} examples, "
            f"skipped {skipped} rows with empty prompt/completion"
        )

        # Verify all 4 data types present (train split should have all 4)
        if split == "train":
            seen_types = {ex["data_type"] for ex in self.examples}
            assert seen_types == EXPECTED_DATA_TYPES, (
                f"Expected all 4 data types, got {seen_types}"
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
        one user message and one assistant message, plus 'data_type'.
        """
        assert 0 <= index < len(self.examples), (
            f"Index {index} out of range [0, {len(self.examples)})"
        )
        example = self.examples[index]

        # Postcondition: validate conversation format
        messages = example["messages"]
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

        return example
