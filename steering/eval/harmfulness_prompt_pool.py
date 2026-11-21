"""Single derivation of the exact ordered harmfulness prompt pool.

The eval context cache is keyed by exact nanochat token sequence, so a second
prompt derivation that drifts produces total cache misses rather than a visible
selection error. The ``sanity_check/`` smoke tests are deliberately excluded:
they use a different ``rng.sample`` selection and never look prompts up in a
context cache.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import steering.eval.harmfulness_judge as harmfulness_judge


def _validate_pool_controls(
    *,
    seed: int,
    max_harmful: int | None,
) -> None:
    """Reject invalid selection controls before prompt-source I/O."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if max_harmful is not None and (
        isinstance(max_harmful, bool) or not isinstance(max_harmful, int)
    ):
        raise TypeError(
            "max_harmful must be an integer or None, got "
            f"{type(max_harmful).__name__}"
        )


@dataclass(frozen=True)
class HarmfulnessPromptPool:
    """The exact ordered harmfulness prompt pool for one selection triple."""

    rows: tuple[Mapping[str, str], ...]
    seed: int
    max_harmful: int | None
    num_benign: int

    @property
    def harmful(self) -> tuple[Mapping[str, str], ...]:
        """Return the harmful rows in their resolved order."""
        return tuple(row for row in self.rows if row["pool"] == "harmful")

    @property
    def benign(self) -> tuple[Mapping[str, str], ...]:
        """Return the benign rows in their resolved order."""
        return tuple(row for row in self.rows if row["pool"] == "benign")


def build_harmfulness_prompt_pool(
    *,
    harmful: Sequence[Mapping[str, str]],
    benign: Sequence[Mapping[str, str]],
    seed: int,
    max_harmful: int | None,
) -> HarmfulnessPromptPool:
    """Build the shared harmful-then-benign prompt sequence without mutation.

    The positive-cap guard around shuffling is intentionally asymmetric. The
    existing entry points preserve source order when the cap is ``None`` or
    non-positive, but shuffle even when a positive cap exceeds the source pool.
    Keeping that guard preserves their established prompt selection exactly.
    """
    _validate_pool_controls(seed=seed, max_harmful=max_harmful)

    selected_harmful = list(harmful)
    if max_harmful is not None and max_harmful > 0:
        random.Random(seed).shuffle(selected_harmful)
        selected_harmful = selected_harmful[:max_harmful]
    rows = tuple(
        {**prompt, "pool": "harmful"} for prompt in selected_harmful
    ) + tuple(
        {**prompt, "pool": "benign"} for prompt in benign
    )
    return HarmfulnessPromptPool(
        rows=rows,
        seed=seed,
        max_harmful=max_harmful,
        num_benign=len(benign),
    )


def resolve_harmfulness_prompt_pool(
    *,
    seed: int,
    max_harmful: int | None,
    num_benign: int,
) -> HarmfulnessPromptPool:
    """Load and resolve the shared harmful-then-benign prompt sequence.

    ``num_benign`` is validated HERE rather than being left to
    :func:`load_benign_prompts`. That loader does reject the same values, but it
    names its own parameter (``n must be non-negative``), which tells an operator
    who passed ``conditioning`` or ``harm_num_benign`` nothing about which knob
    was wrong. All three controls are validated before either pinned dataset is
    loaded, so invalid configuration fails without source I/O.
    """
    _validate_pool_controls(seed=seed, max_harmful=max_harmful)
    if isinstance(num_benign, bool) or not isinstance(num_benign, int):
        raise TypeError(
            f"num_benign must be an integer, got {type(num_benign).__name__}"
        )
    if num_benign < 0:
        raise ValueError(f"num_benign must be non-negative, got {num_benign}")
    return build_harmfulness_prompt_pool(
        benign=harmfulness_judge.load_benign_prompts(n=num_benign, seed=seed),
        harmful=harmfulness_judge.load_harmful_prompts(),
        seed=seed,
        max_harmful=max_harmful,
    )
