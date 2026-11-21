"""Golden framing and sensitivity tests for row-local seeds."""

from __future__ import annotations

import hashlib
import inspect
import struct

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.seeding import (
    ROW_SEED_SCHEMA_VERSION,
    canonical_factor,
    row_seed,
)

GOLDEN_PROMPT = "café 🌧️"
GOLDEN_PAYLOAD_HEX = (
    "617862656e63682d6164762f726f772d7365656400"
    "000000000000000131"
    "00000000000000023432"
    "000000000000000e50726f6d70745374656572696e67"
    "000000000000000137"
    "000000000000000139"
    "00000000000000143078312e38303030303030303030303030702b30"
    "000000000000004033383434353634306339613830383039636139376665373464316137656461313063646135316163303161306463616532643936366665333966623338336165"
).lower()
GOLDEN_DIGEST_HEX = "d78db2b1a10d66835d947ab2846271f42c125f479eba7da91782c075c68d830c"
GOLDEN_SEED = 6_308_895_128_996_898_435


def test_seed_schema_and_golden_vector() -> None:
    # Payload and full digest are recorded alongside the masked seed so a
    # framing, byte-order, truncation, or sign-mask change is review-visible.
    values = (
        b"1",
        b"42",
        b"PromptSteering",
        b"7",
        b"9",
        b"0x1.8000000000000p+0",
        hashlib.sha256(GOLDEN_PROMPT.encode("utf-8")).hexdigest().encode("ascii"),
    )
    payload = b"axbench-adv/row-seed\x00" + b"".join(
        struct.pack(">Q", len(value)) + value for value in values
    )
    digest = hashlib.sha256(payload).digest()
    assert ROW_SEED_SCHEMA_VERSION == 1
    assert payload.hex() == GOLDEN_PAYLOAD_HEX
    assert digest.hex() == GOLDEN_DIGEST_HEX
    assert int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) == GOLDEN_SEED
    assert row_seed(
        master_seed=42,
        method="PromptSteering",
        concept_id=7,
        input_id=9,
        factor=1.5,
        unattacked_prompt=GOLDEN_PROMPT,
    ) == GOLDEN_SEED


def test_every_included_field_changes_the_seed() -> None:
    base: dict[str, object] = {
        "master_seed": 42,
        "method": "DiffMean",
        "concept_id": 1,
        "input_id": 2,
        "factor": 1.0,
        "unattacked_prompt": "exact prompt",
    }
    baseline = row_seed(**base)  # type: ignore[arg-type]
    mutations = {
        "master_seed": 43,
        "method": "LinearProbe",
        "concept_id": 2,
        "input_id": 3,
        "factor": 1.5,
        "unattacked_prompt": "exact prompt ",
    }
    for field, value in mutations.items():
        changed = {**base, field: value}
        assert row_seed(**changed) != baseline  # ty: ignore[invalid-argument-type]


def test_seed_is_order_invariant_and_excludes_suffix_and_effort_by_construction() -> None:
    rows = [
        (2, "prompt two"),
        (0, "prompt zero"),
        (1, "prompt one"),
    ]

    def seeds(ordered_rows: list[tuple[int, str]]) -> dict[int, int]:
        return {
            input_id: row_seed(
                master_seed=11,
                method="LoRA",
                concept_id=0,
                input_id=input_id,
                factor=1,
                unattacked_prompt=prompt,
            )
            for input_id, prompt in ordered_rows
        }

    assert seeds(rows) == seeds(list(reversed(rows)))
    # Neither suffix nor proposal count is accepted by row_seed, so clean and
    # attacked candidates at every effort level necessarily share this value.
    assert tuple(inspect.signature(row_seed).parameters) == (
        "master_seed",
        "method",
        "concept_id",
        "input_id",
        "factor",
        "unattacked_prompt",
    )


def test_canonical_factor_collapses_equivalent_binary64_values() -> None:
    assert canonical_factor(1) == canonical_factor(1.0) == "0x1.0000000000000p+0"
    assert canonical_factor(-0.0) == canonical_factor(0.0) == "0x0.0p+0"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"master_seed": 1 << 63}, "master_seed must be in"),
        ({"input_id": 1 << 63}, "input_id must be in"),
        ({"method": "SelfConditioning"}, "unknown AxBench-Adv method"),
        ({"unattacked_prompt": "\ud800"}, "must be valid UTF-8"),
    ],
)
def test_invalid_seed_inputs_fail_closed(kwargs: dict[str, object], match: str) -> None:
    valid: dict[str, object] = {
        "master_seed": 0,
        "method": "DiffMean",
        "concept_id": 0,
        "input_id": 0,
        "factor": 1.0,
        "unattacked_prompt": "prompt",
    }
    with pytest.raises(AxBenchAdvError, match=match):
        row_seed(**{**valid, **kwargs})  # ty: ignore[invalid-argument-type]
