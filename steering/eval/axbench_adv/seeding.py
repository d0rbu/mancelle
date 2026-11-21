"""Stable row-local common-random-number seeds for AxBench-Adv."""

from __future__ import annotations

import hashlib
import struct

from beartype import beartype

from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.eval.axbench_adv.types import QWEN_METHODS
from steering.refined_types import FiniteNumberLike, NonNegativeIntLike

ROW_SEED_SCHEMA_VERSION = 1
_DOMAIN = b"axbench-adv/row-seed\x00"
_MAX_SIGNED_63_BIT = (1 << 63) - 1


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def canonical_factor(factor: FiniteNumberLike) -> str:
    """Return one exact, platform-stable spelling of a finite binary64 factor."""

    value = float(factor)
    if value == 0.0:
        value = 0.0
    return value.hex()


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def row_seed(
    *,
    master_seed: NonNegativeIntLike,
    method: str,
    concept_id: NonNegativeIntLike,
    input_id: NonNegativeIntLike,
    factor: FiniteNumberLike,
    unattacked_prompt: str,
) -> int:
    """Hash one row identity to a deterministic non-negative signed-63-bit seed."""

    for name, value in (
        ("master_seed", master_seed),
        ("concept_id", concept_id),
        ("input_id", input_id),
    ):
        if value > _MAX_SIGNED_63_BIT:
            raise AxBenchAdvError(f"{name} must be in [0, 2^63 - 1]")
    if method not in QWEN_METHODS:
        raise AxBenchAdvError(f"unknown AxBench-Adv method {method!r}")
    try:
        prompt_bytes = unattacked_prompt.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise AxBenchAdvError("unattacked_prompt must be valid UTF-8") from exc

    values = (
        str(ROW_SEED_SCHEMA_VERSION).encode("ascii"),
        str(master_seed).encode("ascii"),
        method.encode("utf-8"),
        str(concept_id).encode("ascii"),
        str(input_id).encode("ascii"),
        canonical_factor(factor).encode("ascii"),
        hashlib.sha256(prompt_bytes).hexdigest().encode("ascii"),
    )
    digest = hashlib.sha256(_DOMAIN + b"".join(_frame(value) for value in values)).digest()
    return int.from_bytes(digest[:8], "big") & _MAX_SIGNED_63_BIT


__all__ = ["ROW_SEED_SCHEMA_VERSION", "canonical_factor", "row_seed"]
