"""Regenerate the per-prefix GCG equivalence golden parquet (Phase 0.2).

This is a deterministic capture utility, NOT a test. It runs the CURRENT
``build_faster_gcg_candidates`` on the SHARED single-pair-per-group fixture +
synthetic closures + config from ``_universal_gcg_equiv`` and writes the
resulting candidates table to the committed golden parquet that the Phase-3.4
universal-optimizer equivalence test asserts against byte-for-byte.

Run only when the oracle fixture / synthetic closures / candidate serialization
INTENTIONALLY change (which also flips ``candidate_id``s and digests). ``tests``
is not an installed package, so the repo root must be on ``PYTHONPATH`` for the
``tests.axbench_bpb_adv...`` import to resolve when run as a plain script:

    PYTHONPATH=. uv run python tests/axbench_bpb_adv/functional/_capture_universal_gcg_golden.py

It is import-safe (the build happens inside ``capture_golden``) so the module
can be imported without side effects.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from tests.axbench_bpb_adv.faster_gcg_oracle import build_faster_gcg_candidates
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import (
    equiv_allowed_token_ids,
    equiv_attack_rows,
    equiv_config,
    equiv_embedding_weight,
    equiv_gradient_provider,
    equiv_objective,
    equiv_tokenizer,
    golden_parquet_path,
)


def capture_golden() -> Path:
    """Build the per-prefix candidates and write the golden parquet."""

    candidates = build_faster_gcg_candidates(
        attack_rows=equiv_attack_rows(),
        tokenizer=equiv_tokenizer(),
        cfg=equiv_config(),
        embedding_weight=equiv_embedding_weight(),
        allowed_token_ids=equiv_allowed_token_ids(),
        objective=equiv_objective,
        gradient_provider=equiv_gradient_provider,
    )
    path = golden_parquet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidates, path)
    return path


if __name__ == "__main__":
    written = capture_golden()
    print(f"wrote golden parquet: {written} ({written.stat().st_size} bytes)")
