"""
Anthropic HH-RLHF harmless-base task.
https://huggingface.co/datasets/Anthropic/hh-rlhf

Human-preferred harmless conversations from the harmless-base subset.
We extract the `chosen` response and parse it from the raw format:

    Human: <question>\n\nAssistant: <answer>\n\nHuman: <followup>\n\nAssistant: <answer>

into structured multi-turn messages.
Rows that fail parsing are skipped.
~42K train examples after parsing.
"""

import random
import re

from datasets import load_dataset
from loguru import logger

from errors import DataLoadError, ParseError
from tasks.common import Task


def _parse_hh_conversation(text: str) -> list[dict]:
    """Parse an Anthropic HH-RLHF conversation string into structured messages.

    The raw format uses ``\\n\\nHuman:`` and ``\\n\\nAssistant:`` as turn markers.
    The very first turn may start with just ``Human:`` (no leading newlines).

    Returns a list of message dicts ``[{"role": ..., "content": ...}, ...]``.
    Raises :class:`ParseError` if the conversation fails validation.

    Validation rules:
    - At least 2 messages
    - First message is from the user
    - Last message is from the assistant
    - Roles strictly alternate user/assistant
    - All message contents are non-empty after stripping
    """
    # The HH "chosen" text uses ``\n\nHuman:`` / ``\n\nAssistant:`` as turn
    # markers; the very first turn may start with just ``Human:`` (no
    # leading newlines).  We use ``re.findall`` (below) to extract
    # (marker, content) pairs.  First, normalize: ensure text starts with
    # ``\n\n`` so the regex is uniform.
    if text.startswith(("Human:", "Assistant:")):
        text = "\n\n" + text
    elif text.startswith(("\n\nHuman:", "\n\nAssistant:")):
        pass  # already in expected format
    else:
        raise ParseError(f"unexpected format: text does not start with 'Human:' — got: {text[:80]!r}")

    # Find all turns: marker followed by content until the next marker or end
    pattern = r"\n\n(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        raise ParseError(f"no conversation turns found in text: {text[:80]!r}")

    role_map = {"Human": "user", "Assistant": "assistant"}
    messages = []
    for marker, content in matches:
        stripped_content = content.strip()
        if not stripped_content:
            raise ParseError(f"empty message content in turn: {marker!r}")
        messages.append({"role": role_map[marker], "content": stripped_content})

    # Validate structure
    if len(messages) < 2:
        raise ParseError(f"fewer than 2 messages: got {len(messages)}")
    if messages[0]["role"] != "user":
        raise ParseError(f"conversation does not start with user: first role is {messages[0]['role']!r}")
    if messages[-1]["role"] != "assistant":
        raise ParseError(f"conversation does not end with assistant: last role is {messages[-1]['role']!r}")

    # Verify strict alternation
    expected_roles = ["user", "assistant"] * ((len(messages) + 1) // 2)
    for i, msg in enumerate(messages):
        expected = expected_roles[i]
        if msg["role"] != expected:
            raise ParseError(f"role alternation violated at index {i}: expected {expected!r}, got {msg['role']!r}")

    return messages


class AnthropicHH(Task):
    """Anthropic HH-RLHF harmless-base dataset for SFT training."""

    def __init__(self, split: str = "train", **kwargs: object) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), (
            f"AnthropicHH supports 'train' and 'test' splits, got {split!r}"
        )
        raw_ds = load_dataset(
            "Anthropic/hh-rlhf", data_dir="harmless-base", split=split
        )

        # Parse chosen conversations, skipping rows that fail parsing
        self.examples: list[dict] = []
        skipped = 0
        skip_reasons: dict[str, int] = {}
        for row in raw_ds:
            chosen_text = row["chosen"]
            try:
                messages = _parse_hh_conversation(chosen_text)
            except ParseError as exc:
                reason = str(exc).split(":")[0] if ":" in str(exc) else str(exc)[:40]
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                skipped += 1
                continue

            self.examples.append({"messages": messages})

        if skipped > 0:
            total = skipped + len(self.examples)
            logger.warning(f"Skipped {skipped}/{total} rows. Reasons: {dict(skip_reasons)}")
            if total > 0 and skipped / total > 0.5:
                raise DataLoadError(
                    f"AnthropicHH: skip rate {skipped/total:.0%} exceeds 50% threshold. "
                    f"Reasons: {dict(skip_reasons)}"
                )

        logger.info(
            f"AnthropicHH harmless-base ({split}): loaded {len(self.examples)} examples, "
            f"skipped {skipped} rows that failed parsing"
        )

        # Deterministic shuffle
        rng = random.Random(42)
        rng.shuffle(self.examples)

    def num_examples(self) -> int:
        """Return the number of parsed examples in the dataset."""
        return len(self.examples)

    def get_example(self, index: int) -> dict:
        """Get a single multi-turn conversation by index.

        Returns a dict with 'messages' key containing a list of
        user/assistant message dicts with strictly alternating roles.
        """
        assert 0 <= index < len(self.examples), (
            f"Index {index} out of range [0, {len(self.examples)})"
        )
        conversation = self.examples[index]

        # Postcondition: validate conversation format
        messages = conversation["messages"]
        assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "user", (
            f"First message must be user, got {messages[0]['role']}"
        )
        assert messages[-1]["role"] == "assistant", (
            f"Last message must be assistant, got {messages[-1]['role']}"
        )
        for i, msg in enumerate(messages):
            assert isinstance(msg["content"], str) and len(msg["content"]) > 0, (
                f"Message {i} content must be a non-empty string"
            )

        return conversation
