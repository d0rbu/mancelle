"""SFT mixture builders for training and data conversion.

Extracted from scripts/chat_sft.py so both the training script and the
SFT-to-parquet conversion script can reuse the same mixture definitions.
"""

import os
import unicodedata
from collections.abc import Mapping, Sequence

from beartype import beartype
from loguru import logger

from nanochat.common import get_base_dir
from tasks.arc import ARC
from tasks.coconot import CoCoNot
from tasks.common import Task, TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.metamath import MetaMathQA
from tasks.mixture_names import (
    STEERED_MIXTURES,
    SFTMixture,
)
from tasks.mixture_names import (
    sft_val_family as sft_val_family,
)
from tasks.mmlu import MMLU
from tasks.nemotron_safety import NemotronSafety
from tasks.opencoder import OpenCoder
from tasks.orca_math import OrcaMath
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee
from tasks.steered import (
    SteeredConversations,
    SteeredTaskMixture,
    balanced_standard_caps,
)
from tasks.wildguardmix import WildGuardMix
from tasks.wildjailbreak import WildJailbreak

# Default hyperparams for mixture construction.
# scripts/chat_sft.py overrides these from Hydra config.
_DEFAULTS = {
    "mmlu_epochs": 1,
    "gsm8k_epochs": 1,
    "capabilities_mmlu_epochs": 2,
    "smoltalk_limit": 120000,
    "opencoder_limit": 50000,
    "orca_math_limit": None,
    "metamath_limit": None,
    "simple_spelling_size": 200000,
    "spelling_bee_size": 80000,
    "wildjailbreak_epochs": 2,
    "wildguardmix_epochs": 3,
    "coconot_epochs": 2,
    "smoltalk_safety_limit": 120000,
    # Path to the steered single-turn JSONL; required by (and only consumed by)
    # the steered mixtures.
    "steered_jsonl": None,
}

# Public export: the full knob surface a config exposes for mixtures.
MIXTURE_PARAM_KEYS: frozenset[str] = frozenset(_DEFAULTS)

# Per-mixture consumed-parameter declarations.  ``build_mixture`` enforces
# that each builder reads exactly its declared set and rejects overrides
# the selected mixture does not consume.  This keeps standard and
# capabilities-only knobs from being silently dropped when configs are
# forwarded through the shared Hydra surface.
_MIXTURE_KEYS: dict[SFTMixture, frozenset[str]] = {
    SFTMixture.CHATCORE: frozenset(),
    SFTMixture.STANDARD: frozenset(
        {
            "mmlu_epochs",
            "gsm8k_epochs",
            "simple_spelling_size",
            "spelling_bee_size",
        }
    ),
    SFTMixture.CAPABILITIES: frozenset(
        {
            "smoltalk_limit",
            "capabilities_mmlu_epochs",
            "opencoder_limit",
            "orca_math_limit",
            "metamath_limit",
            "simple_spelling_size",
            "spelling_bee_size",
        }
    ),
    SFTMixture.SAFETY: frozenset(
        {
            "wildjailbreak_epochs",
            "wildguardmix_epochs",
            "coconot_epochs",
            "smoltalk_safety_limit",
        }
    ),
}

_MIXTURE_KEYS.update(
    {
        member: _MIXTURE_KEYS[SFTMixture.STANDARD] | {"steered_jsonl"}
        for member in STEERED_MIXTURES
    }
)


@beartype
def steered_mixture_overrides(
    mixture: SFTMixture, steered_jsonl: str | None
) -> dict[str, object]:
    """Return the ``steered_jsonl`` override ``mixture`` needs, rejecting drift.

    A steered mixture REQUIRES the path; every other mixture rejects a set one
    rather than silently ignoring it.
    """
    if mixture in STEERED_MIXTURES:
        if steered_jsonl is None:
            raise ValueError(f"mixture {mixture.value!r} requires steered_jsonl=<path>")
        return {"steered_jsonl": steered_jsonl}
    if steered_jsonl is not None:
        raise ValueError(
            f"mixture {mixture.value!r} does not consume steered_jsonl="
            f"{steered_jsonl!r}, so it would be silently ignored"
        )
    return {}


class _TrackedParams(dict):
    """Params dict that records which keys the selected builder read."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.read_keys: set[str] = set()

    def __getitem__(self, key: str) -> object:
        self.read_keys.add(key)
        return super().__getitem__(key)


def _consumers_of(key: str) -> list[str]:
    return sorted(m.value for m, keys in _MIXTURE_KEYS.items() if key in keys)


def select_mixture_overrides(
    mixture: SFTMixture, values: Mapping[str, object]
) -> dict[str, object]:
    """Select the override subset the given mixture consumes.

    ``values`` must supply exactly ``MIXTURE_PARAM_KEYS`` (the full knob
    surface a training config exposes).  A key OUTSIDE the selected
    mixture's declared set whose value differs from its default raises:
    the operator set a knob that would otherwise be silently ignored.
    """
    expected, got = set(_DEFAULTS), set(values)
    if got != expected:
        raise ValueError(
            "select_mixture_overrides requires exactly the mixture "
            f"parameter key set; missing={sorted(expected - got)} "
            f"unexpected={sorted(got - expected)}"
        )
    inert = {
        key: values[key]
        for key in sorted(expected - _MIXTURE_KEYS[mixture])
        if values[key] != _DEFAULTS[key]
    }
    if inert:
        details = ", ".join(
            f"{key}={value!r} (default {_DEFAULTS[key]!r}, consumed by {_consumers_of(key)})"
            for key, value in inert.items()
        )
        raise ValueError(
            f"mixture '{mixture.value}' does not consume these overridden "
            f"parameter(s), so they would be silently ignored: {details}"
        )
    return {key: values[key] for key in sorted(_MIXTURE_KEYS[mixture])}


def build_mixture(
    mixture: SFTMixture,
    **overrides,
) -> tuple[list[Task], list[Task]]:
    """Build train and val task lists for the given SFT mixture.

    Args:
        mixture: Which mixture to build.
        **overrides: Override default hyperparams (e.g. mmlu_epochs=5).

    Returns:
        (train_tasks, val_tasks) tuple.

    Raises:
        ValueError: If an override is unknown, inert for the selected mixture,
            or the declared consumed-parameter set drifts from builder reads.
    """
    unknown = sorted(set(overrides) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown mixture override(s) {unknown}; valid keys: {sorted(_DEFAULTS)}")
    out_of_set = sorted(set(overrides) - _MIXTURE_KEYS[mixture])
    if out_of_set:
        details = ", ".join(f"{key} (consumed by {_consumers_of(key)})" for key in out_of_set)
        raise ValueError(f"mixture '{mixture.value}' does not consume override(s): {details}")
    params = _TrackedParams({**_DEFAULTS, **overrides})
    base_dir = get_base_dir()
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")

    builders = {
        SFTMixture.CHATCORE: _build_chatcore,
        SFTMixture.STANDARD: _build_standard,
        SFTMixture.CAPABILITIES: _build_capabilities,
        SFTMixture.SAFETY: _build_safety,
        SFTMixture.STANDARD_STEERED: _build_standard_steered,
        SFTMixture.STANDARD_STEERED_BALANCED: _build_standard_steered_balanced,
    }
    result = builders[mixture](params, identity_path)
    declared = _MIXTURE_KEYS[mixture]
    if params.read_keys != declared:
        raise ValueError(
            f"_MIXTURE_KEYS[{mixture.value!r}] drifted from the builder's "
            f"actual reads: declared-but-never-read="
            f"{sorted(declared - params.read_keys)}, read-but-undeclared="
            f"{sorted(params.read_keys - declared)}"
        )
    return result


def build_task_mixtures(
    mixture: SFTMixture, **overrides: object
) -> tuple[TaskMixture, TaskMixture | None]:
    """Build (train, val) TaskMixtures for an SFT mixture.

    val is None when the mixture defines no validation tasks. Overrides
    pass through ``build_mixture`` and are therefore subject to the same
    per-mixture consumption guards.  A steered mixture's train order is the
    evenly-spaced ``SteeredTaskMixture`` over its last (steered) train task.
    """
    train_tasks, val_tasks = build_mixture(mixture, **overrides)
    val = TaskMixture(val_tasks) if val_tasks else None
    if mixture in STEERED_MIXTURES:
        return SteeredTaskMixture(train_tasks[:-1], train_tasks[-1]), val
    return TaskMixture(train_tasks), val


def question_key(question: str) -> str:
    """Match question text despite Unicode width, case, and whitespace differences."""
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def exclude_heldout_questions(
    train_tasks: Sequence[ARC | MMLU | GSM8K],
    val_tasks: Sequence[ARC | MMLU | GSM8K],
) -> None:
    """Exclude held-out questions regardless of choices; preserve row order and validation."""
    heldout = {question_key(question) for task in val_tasks for question in task.ds["question"]}
    for task in train_tasks:
        keep = [
            index
            for index, question in enumerate(task.ds["question"])
            if question_key(question) not in heldout
        ]
        removed = len(task.ds) - len(keep)
        task.ds = task.ds.select(keep)
        logger.info(f"ChatCORE {type(task).__name__}: excluded {removed} held-out question matches")
    if not any(len(task) for task in train_tasks):
        raise ValueError("ChatCORE has no training rows after excluding held-out questions")


def _build_chatcore(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """One source-sized pass with original renderers and held-out questions excluded."""
    del params, identity_path
    train_tasks: list[ARC | MMLU | GSM8K] = [
        ARC(subset="ARC-Easy", split="train"),
        ARC(subset="ARC-Challenge", split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
    ]
    val_tasks: list[ARC | MMLU | GSM8K] = [
        ARC(subset="ARC-Easy", split="test"),
        ARC(subset="ARC-Challenge", split="test"),
        MMLU(subset="all", split="test"),
        GSM8K(subset="main", split="test"),
    ]
    exclude_heldout_questions(train_tasks, val_tasks)
    return list(train_tasks), list(val_tasks)


def _build_standard(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """Standard SFT mixture: personality, math, coding, spelling."""
    train_tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_path),
        CustomJSON(filepath=identity_path),
        *[
            MMLU(subset="auxiliary_train", split="train")
            for _ in range(params["mmlu_epochs"])
        ],
        *[GSM8K(subset="main", split="train") for _ in range(params["gsm8k_epochs"])],
        SimpleSpelling(size=params["simple_spelling_size"], split="train"),
        SpellingBee(size=params["spelling_bee_size"], split="train"),
    ]
    val_tasks = [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ]
    return train_tasks, val_tasks


def _require_steered_task(params: dict, mixture: SFTMixture) -> SteeredConversations:
    """Load the steered JSONL a steered mixture cannot be built without."""
    steered_jsonl = params["steered_jsonl"]
    if steered_jsonl is None:
        raise ValueError(f"mixture {mixture.value!r} requires steered_jsonl=<path>")
    return SteeredConversations(filepath=str(steered_jsonl))


def _build_standard_steered(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """Standard mixture with the steered JSONL appended as the LAST train task."""
    steered = _require_steered_task(params, SFTMixture.STANDARD_STEERED)
    train_tasks, val_tasks = _build_standard(params, identity_path)
    return [*train_tasks, steered], val_tasks


def _build_standard_steered_balanced(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """Standard mixture down-capped to the steered row count (a 50/50 mix).

    Each standard task keeps a share of the retained rows proportional to its
    own length (largest-remainder apportionment).  Tasks whose share rounds to
    zero are KEPT at length 0 so the per-task fingerprint breakdown still
    enumerates the full standard catalogue.
    """
    steered = _require_steered_task(params, SFTMixture.STANDARD_STEERED_BALANCED)
    train_tasks, val_tasks = _build_standard(params, identity_path)
    caps = balanced_standard_caps([len(task) for task in train_tasks], len(steered))
    for task, cap in zip(train_tasks, caps, strict=True):
        # The tasks are already constructed; re-instantiating with stop=cap
        # would re-download. Narrow the existing logical slice instead.
        task.stop = task.start + cap * task.step
        assert len(task) == cap, f"cap {cap} left len={len(task)} on {type(task).__name__}"
    return [*train_tasks, steered], val_tasks


def _build_capabilities(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """Capabilities-focused SFT mixture with chat, coding, and identity context."""
    train_tasks = [
        SmolTalk(split="train", stop=params["smoltalk_limit"]),
        CustomJSON(filepath=identity_path),
        CustomJSON(filepath=identity_path),
        *[
            MMLU(subset="auxiliary_train", split="train")
            for _ in range(params["capabilities_mmlu_epochs"])
        ],
        OpenCoder(split="train", max_rows=params["opencoder_limit"]),
        OrcaMath(split="train", stop=params["orca_math_limit"]),
        MetaMathQA(split="train", gsm_only=True, stop=params["metamath_limit"]),
        SimpleSpelling(size=params["simple_spelling_size"], split="train"),
        SpellingBee(size=params["spelling_bee_size"], split="train"),
    ]
    val_tasks = [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        OrcaMath(split="test"),
        GSM8K(subset="main", split="test", stop=420),
    ]
    return train_tasks, val_tasks


def _build_safety(
    params: dict,
    identity_path: str,
) -> tuple[list[Task], list[Task]]:
    """Safety-focused SFT mixture: refusal, truthfulness, helpfulness."""
    train_tasks = [
        # Refusal
        *[WildJailbreak(split="train") for _ in range(params["wildjailbreak_epochs"])],
        *[WildGuardMix(split="train") for _ in range(params["wildguardmix_epochs"])],
        *[CoCoNot(split="train") for _ in range(params["coconot_epochs"])],
        # Helpfulness
        SmolTalk(split="train", stop=params["smoltalk_safety_limit"]),
        NemotronSafety(split="train"),
        # Identity (carry over from standard)
        CustomJSON(filepath=identity_path),
        CustomJSON(filepath=identity_path),
    ]
    val_tasks = [
        SmolTalk(split="test", stop=5000),
    ]
    return train_tasks, val_tasks
