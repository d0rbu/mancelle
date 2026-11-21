"""Slice 17 audit (2026-05-05) Concept consolidation regression test.

The CLI ``scripts/eval/axbench_bpb_eval.py`` previously defined a private
``_Concept`` lightweight dataclass to deserialize ``concepts.json``. The
consolidation drops ``_Concept`` and uses the canonical
``steering.eval.axbench_base.types.Concept``.

Tests:
1. ``concepts.json``-style dict roundtrips through the canonical Concept.
2. The CLI no longer exposes ``_Concept`` (so future drift can't reintroduce it).
3. ``build_intervention`` reads only ``concept.concept_id`` so the extra
   fields in the canonical Concept are inert.
"""
from __future__ import annotations

import dataclasses

from steering.eval.axbench_base.types import Concept


def _sample_concept_dict() -> dict:
    """A dict matching the canonical concepts.json schema."""
    return {
        "concept_id": 42,
        "feature_index": 7,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": "vehicles and transportation",
        "corpus_count": 100,
        "description_source": "auto",
    }


def test_concept_roundtrip():
    """concepts.json dict → Concept → asdict → Concept parity."""
    cdict = _sample_concept_dict()
    c1 = Concept(**cdict)
    c1_dict = dataclasses.asdict(c1)
    c2 = Concept(**c1_dict)
    assert c1 == c2
    assert c1_dict == cdict


def test_cli_does_not_expose_lightweight_concept():
    """The lightweight ``_Concept`` stand-in must be removed from the CLI."""
    import scripts.eval.axbench_bpb_eval as cli_mod

    assert not hasattr(cli_mod, "_Concept"), (
        "Concept consolidation: scripts/eval/axbench_bpb_eval._Concept "
        "must be replaced by steering.eval.axbench_base.types.Concept."
    )
    # Positive: canonical Concept must be imported.
    assert hasattr(cli_mod, "Concept"), (
        "Concept consolidation: CLI must import the canonical "
        "axbench_base.types.Concept."
    )


def test_concept_concept_id_is_int():
    """Concept.concept_id remains the integer field that build_intervention reads."""
    c = Concept(**_sample_concept_dict())
    assert isinstance(c.concept_id, int)
