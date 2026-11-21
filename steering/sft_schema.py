"""Shared schema and split types for full-sequence SFT parquet artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Flag, StrEnum, auto
from pathlib import Path
from typing import Protocol, runtime_checkable

from beartype import beartype


@runtime_checkable
class SFTConversationSource(Protocol):
    """Structural contract for a converter's ``mixture`` argument.

    Production always passes a ``tasks.common.TaskMixture``.  The converter
    itself only ever calls ``len(mixture)`` and ``mixture[index]`` (indexing
    into the mixture's own deterministic shuffle), so this Protocol names
    exactly that surface, letting focused tests supply lightweight,
    duck-typed fixtures instead of a real ``TaskMixture`` wrapping stub
    ``Task`` subclasses.

    This is a weak, presence-only check: ``@runtime_checkable`` Protocols
    verify that ``__len__``/``__getitem__`` exist, not their signatures or
    semantics, so ``list``, ``dict``, ``str``, and ``tuple`` all satisfy it
    too — only objects missing one of those two dunders (e.g. a bare ``int``)
    are rejected at the ``@beartype`` boundary. Real argument correctness —
    that the conversations actually came from the intended mixture — is
    enforced downstream by the mixture fingerprint
    (``compute_mixture_fingerprint`` / ``_assert_sft_mixture_fingerprint_matches_task``)
    and the startup content-alignment probe, not by this annotation.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> dict: ...


class SFTSplit(StrEnum):
    """A concrete SFT task split."""

    TRAIN = "train"
    VAL = "val"

    @classmethod
    def parse(cls, raw: object) -> SFTSplit:
        """Parse an exact serialized split value."""

        if isinstance(raw, cls):
            return raw
        try:
            return cls(raw)
        except (TypeError, ValueError):
            valid = ", ".join(repr(split.value) for split in cls)
            raise ValueError(f"Invalid SFT split {raw!r}; expected one of {valid}.") from None


class SFTSplitSelection(Flag):
    """One or more SFT splits selected for conversion."""

    TRAIN = auto()
    VAL = auto()
    BOTH = TRAIN | VAL

    @classmethod
    def parse(cls, raw: object) -> SFTSplitSelection:
        """Parse an exact serialized split-selection value.

        Concrete ``SFTSplit`` members are accepted explicitly and map to the
        corresponding single-bit selection; serialized strings additionally
        accept ``"both"``.
        """

        if isinstance(raw, cls):
            return raw
        if raw is SFTSplit.TRAIN:
            return cls.TRAIN
        if raw is SFTSplit.VAL:
            return cls.VAL
        by_name = {
            SFTSplit.TRAIN.value: cls.TRAIN,
            SFTSplit.VAL.value: cls.VAL,
            "both": cls.BOTH,
        }
        parsed = by_name.get(raw) if isinstance(raw, str) else None
        if parsed is not None:
            return parsed
        valid = ", ".join(repr(name) for name in by_name)
        raise ValueError(f"Invalid SFT split selection {raw!r}; expected one of {valid}.") from None

    def splits(self) -> tuple[SFTSplit, ...]:
        """Return selected concrete splits in deterministic train/val order."""

        return tuple(
            split
            for member, split in (
                (SFTSplitSelection.TRAIN, SFTSplit.TRAIN),
                (SFTSplitSelection.VAL, SFTSplit.VAL),
            )
            if member in self
        )


@beartype
def sft_split_dir(train_dir: Path | str, split: SFTSplit) -> Path:
    """Return the on-disk directory for ``split`` from its train directory.

    This is the single production home of the ``_val`` directory-suffix
    convention.
    """

    return Path(train_dir) if split is SFTSplit.TRAIN else Path(f"{train_dir}_val")


@dataclass(frozen=True)
class SFTConversionMeta:
    """Validated contents of an SFT ``conversion_meta.json`` artifact."""

    mixture: str
    split: SFTSplit | None
    format: str
    completed_shards: int
    total_shards: int
    total_conversations: int
    raw: dict[str, object] = field(repr=False, compare=False)

    @classmethod
    def parse(cls, mapping: Mapping[str, object]) -> SFTConversionMeta:
        """Validate and parse one conversion metadata mapping."""

        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"SFT conversion metadata must be a mapping, got {type(mapping).__name__}."
            )
        raw = dict(mapping)

        def required_str(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str):
                raise ValueError(
                    f"SFT conversion metadata field {key!r} must be a string, got {value!r}."
                )
            return value

        def required_int(key: str) -> int:
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"SFT conversion metadata field {key!r} must be an integer, got {value!r}."
                )
            return value

        split_raw = raw.get("split")
        split = None if split_raw is None else SFTSplit.parse(split_raw)
        return cls(
            mixture=required_str("mixture"),
            split=split,
            format=required_str("format"),
            completed_shards=required_int("completed_shards"),
            total_shards=required_int("total_shards"),
            total_conversations=required_int("total_conversations"),
            raw=raw,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialize with the source key set and order preserved."""

        result = dict(self.raw)
        result["mixture"] = self.mixture
        if "split" in result or self.split is not None:
            result["split"] = None if self.split is None else self.split.value
        result["format"] = self.format
        result["completed_shards"] = self.completed_shards
        result["total_shards"] = self.total_shards
        result["total_conversations"] = self.total_conversations
        return result


SFT_LEGACY_COLUMNS = frozenset(
    {
        "messages",
        "response",
        "assistant_char_offset",
        "assistant_loss_mask",
    }
)


def legacy_sft_columns(schema_names: Iterable[str]) -> set[str]:
    """Return removed SFT schema columns present in ``schema_names``."""

    return set(schema_names) & SFT_LEGACY_COLUMNS
