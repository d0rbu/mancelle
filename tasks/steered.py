"""Steered single-turn conversations and their evenly-spaced SFT mixture.

A steered dataset is a JSONL file of single-turn conversations sampled from a
steered model.  :class:`SteeredTaskMixture` interleaves those rows into a base
mixture at exactly even spacing: any two consecutive steered rows are
``floor(L/M)`` or ``ceil(L/M)`` positions apart, never further.
"""

import json
import os
import random
from collections.abc import Sequence
from typing import Any

from beartype import beartype

from errors import DataLoadError
from tasks.common import Task, TaskMixture

# Roles a steered conversation must carry, in order.  ``tasks/`` deliberately
# keeps no ``steering/`` dependency, so this mirrors ``steering.sft.turn_spans``
# ``TurnRole`` rather than importing it through the heavy ``steering`` package.
_STEERED_ROLES = ("user", "assistant")

# Bound on how much of a rejected payload an error message repeats.
_REPR_LIMIT = 200


def _parse_steered_conversation(
    payload: str, filepath: str, line_number: int
) -> list[dict[str, Any]]:
    """Validate one JSONL line as a single-turn steered conversation."""
    where = f"steered: {filepath} line {line_number}"
    try:
        payload_json = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"{where} is not valid JSON: {exc}") from exc
    # Two line shapes are accepted: the bare message list, and the exported
    # dataset shape ``{"messages": [...], ...}`` (extra keys such as ``id`` are
    # provenance and are ignored).  Anything else is a provisioning error.
    messages = (
        payload_json.get("messages") if isinstance(payload_json, dict) else payload_json
    )
    if not isinstance(messages, list) or len(messages) != len(_STEERED_ROLES):
        # One malformed megabyte-sized line must not become a megabyte-sized
        # exception string and log record: show a bounded prefix instead.
        raise DataLoadError(
            f"{where} must be a JSON list of exactly {len(_STEERED_ROLES)} messages "
            f"{list(_STEERED_ROLES)} (or an object whose 'messages' key holds that "
            f"list), got {repr(payload_json)[:_REPR_LIMIT]}"
        )
    for index, (message, role) in enumerate(zip(messages, _STEERED_ROLES, strict=True)):
        if not isinstance(message, dict):
            raise DataLoadError(f"{where} message {index} must be an object, got {message!r}")
        if message.get("role") != role:
            raise DataLoadError(
                f"{where} message {index} must have role={role!r}, got {message.get('role')!r}"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise DataLoadError(
                f"{where} message {index} content must be a string, got {content!r}"
            )
        if role == "assistant" and not content.strip():
            raise DataLoadError(f"{where} assistant content must be non-blank")
    return messages


class SteeredConversations(Task):
    """Single-turn steered conversations loaded from a JSONL file.

    Each non-blank line holds exactly two message objects — a ``user`` prompt
    and a non-blank ``assistant`` completion — either as a bare JSON list or
    under the ``"messages"`` key of a JSON object (the exported dataset shape;
    its other keys, e.g. ``id``, are ignored).  Unlike
    :class:`tasks.customjson.CustomJSON` there is no ``allow_missing`` escape
    hatch — a steered mixture missing its steered rows is a provisioning error,
    never a silently smaller mixture.
    """

    def __init__(self, filepath: str, **task_kwargs):
        super().__init__(**task_kwargs)
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise DataLoadError(f"steered: {filepath} not found")
        with open(filepath, encoding="utf-8") as handle:
            self.conversations = [
                _parse_steered_conversation(stripped, filepath, line_number)
                for line_number, line in enumerate(handle, start=1)
                if (stripped := line.strip())
            ]
        if not self.conversations:
            raise DataLoadError(f"steered: {filepath} holds zero steered conversations")
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}


class SteeredTaskMixture(TaskMixture):
    """A base mixture with steered rows interleaved at exactly even spacing.

    Steered row ``k`` (in the seed-42 shuffled steered order) sits at combined
    position ``((2 * k + 1) * L) // (2 * M)`` for ``L = N + M`` base+steered
    rows; the base rows keep their plain :class:`TaskMixture` seed-42 order and
    fill the gaps.  The steered task is always LAST in ``self.tasks``, so a
    mixture fingerprint sees one flat per-task length entry for it.

    DDP caveat: the spacing is exact at the DATASET level only.  The conditioned
    SFT generator strides dataset positions by ``world_size``, so a rank's
    steered share can be uneven — a spacing of 6 lands only on odd ranks at
    world size 2, 4 and 8.  The averaged gradient still carries the global
    steered fraction; per-rank coverage is logged by the trainer (PR 5).
    """

    def __init__(self, base_tasks: Sequence[Task], steered: Task, **task_kwargs):
        # Deliberately NOT TaskMixture.__init__: its shuffle would scatter the
        # steered rows through the base order and destroy the even spacing.
        Task.__init__(self, **task_kwargs)
        steered_count = len(steered)
        if steered_count < 1:
            raise ValueError("SteeredTaskMixture requires at least one steered conversation")
        base_order = TaskMixture(list(base_tasks)).index_map
        steered_order = list(range(steered_count))
        random.Random(42).shuffle(steered_order)  # same seed as TaskMixture
        total = len(base_order) + steered_count
        positions = [((2 * k + 1) * total) // (2 * steered_count) for k in range(steered_count)]
        assert positions == sorted(set(positions)) and positions[-1] < total, (
            f"steered positions must be strictly increasing and < {total}: {positions}"
        )
        self.tasks = [*base_tasks, steered]
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        self._steered_positions = frozenset(positions)
        base_rows, steered_rows = iter(base_order), iter(steered_order)
        self.index_map = [
            (self.steered_task_index, next(steered_rows))
            if position in self._steered_positions
            else next(base_rows)
            for position in range(total)
        ]

    @property
    def steered_task_index(self) -> int:
        """Index of the steered task inside ``self.tasks`` (always last)."""
        return len(self.tasks) - 1

    @property
    def steered_positions(self) -> frozenset[int]:
        """Combined positions occupied by a steered conversation."""
        return self._steered_positions

    def is_steered(self, position: int) -> bool:
        """Whether the combined ``position`` holds a steered conversation."""
        return position in self._steered_positions


@beartype
def balanced_standard_caps(lengths: Sequence[int], target: int) -> list[int]:
    """Apportion ``target`` rows across ``lengths`` by largest remainder.

    Each cap is proportional to its task's length, never exceeds it, and the
    caps sum to exactly ``target``.  Ties on the fractional remainder go to the
    lower index so the apportionment is deterministic.
    """
    total = sum(lengths)
    if target < 1:
        raise ValueError(f"balanced_standard_caps: target must be >= 1, got {target}")
    if target > total:
        raise ValueError(
            f"balanced_standard_caps: target={target} exceeds the total available rows ({total})"
        )
    scaled = [length * target for length in lengths]
    caps = [value // total for value in scaled]
    order = sorted(range(len(scaled)), key=lambda index: (-(scaled[index] % total), index))
    for index in order[: target - sum(caps)]:
        caps[index] += 1
    return caps
