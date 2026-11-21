"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import json
import os
from typing import TypedDict, cast

from loguru import logger

from errors import DataLoadError
from tasks.common import Task


class _Message(TypedDict):
    role: str
    content: str


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, allow_missing: bool = False, **kwargs):
        """Initialize CustomJSON.

        Args:
            filepath: Path to the JSONL file.
            allow_missing: If ``False`` (the default), a missing or empty
                ``filepath`` raises :class:`errors.DataLoadError` so that
                identity data absence is loud. Set to ``True`` only when
                the caller explicitly wants zero conversations on missing
                input — e.g. a test harness or an opt-in mixture variant
                that documents the absence.
            **kwargs: Forwarded to :class:`tasks.common.Task` (start/stop/step).
        """
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations: list[list[_Message]] = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            if not allow_missing:
                raise DataLoadError(
                    f"customjson: {filepath} not found or empty; "
                    f"configure a valid path or pass allow_missing=True, "
                    f"or remove customjson from the mixture"
                )
            # Helpful warning: allow_missing=True was explicitly requested.
            logger.warning(
                f"customjson: {filepath} not found; allow_missing=True so "
                f"proceeding with zero conversations"
            )
            logger.info("HINT (Oct 21 2025)")
            logger.info(
                "If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations"
            )
            logger.info(
                "See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139"
            )
            logger.info(
                "Quick fix: simply run the following command to download the file and you're done:"
            )
            logger.info(
                f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
            )

        else:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()
                    if not stripped_line:  # skip empty lines
                        continue
                    raw_messages: object = json.loads(stripped_line)
                    # Validate the conversation structure
                    assert isinstance(raw_messages, list), (
                        f"Expected list of messages, got {type(raw_messages)}"
                    )
                    assert len(raw_messages) >= 2, (
                        f"Conversation must have at least 2 messages, got {len(raw_messages)}"
                    )
                    # Validate message structure and alternating roles
                    for i, message in enumerate(raw_messages):
                        assert isinstance(message, dict), f"Message {i} must be an object"
                        typed_message = cast(dict[str, object], message)
                        assert "role" in typed_message, (
                            f"Message {i} missing 'role' field"
                        )
                        assert "content" in typed_message, (
                            f"Message {i} missing 'content' field"
                        )
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert typed_message["role"] == expected_role, (
                            f"Message {i} has role {typed_message['role']} but should be {expected_role}"
                        )
                        assert isinstance(typed_message["content"], str), (
                            f"Message {i} content must be a string"
                        )

                    # Every element has now been checked against _Message's
                    # complete runtime contract; retain the original dicts so
                    # optional extra JSON fields remain byte-for-byte intact.
                    messages = cast(list[_Message], raw_messages)
                    self.conversations.append(messages)

            # Distinguish "file exists but is empty / all blank lines" from the
            # valid case. The first is an identity-absence bug in disguise.
            if not self.conversations and not allow_missing:
                raise DataLoadError(
                    f"customjson: {filepath} not found or empty; "
                    f"configure a valid path or pass allow_missing=True, "
                    f"or remove customjson from the mixture"
                )

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation
