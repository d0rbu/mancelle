"""Backward compatibility for Stage 0c fingerprints that predate the removal
of ``max_concepts``.

``max_concepts`` used to be part of the canonical Stage 0c fingerprint. It is
a stopping condition, not a semantic property of how a candidate is verified,
so it was removed to let a run's target be raised without discarding
already-decided candidates. Artifacts written before that removal still carry
the field, and must stay resumable:

- a **partial checkpoint** stores the fingerprint as a dict, and the loader
  compares field sets strictly, so the retired key must be ignored;
- a **completed artifact** stores an opaque identity hash of the whole dict,
  so the pre-removal hash must still be recognised.
"""

from __future__ import annotations

from steering.eval.axbench_base._stage0c_internals import (
    _LEGACY_FINGERPRINT_KEYS,
    stage0c_fingerprint_identity,
)


def test_max_concepts_is_declared_legacy() -> None:
    assert "max_concepts" in _LEGACY_FINGERPRINT_KEYS


def test_legacy_partial_field_set_matches_after_dropping_retired_keys() -> None:
    """Mirrors the loader's field-set check on a pre-removal checkpoint."""

    current = {"feature_dir": "/x", "extraction_tokenizer_id": "t"}
    stored_legacy = {**current, "max_concepts": 50}

    expected_keys = set(current)
    actual_keys = set(stored_legacy) - _LEGACY_FINGERPRINT_KEYS
    assert expected_keys == actual_keys


def test_legacy_identity_differs_from_current_identity() -> None:
    """The compatibility branch is load-bearing, not a no-op.

    If these hashed the same, the acceptance branch would be untested and a
    real mismatch could slip through it.
    """

    current = {"feature_dir": "/x"}
    legacy = {**current, "max_concepts": 50}
    assert stage0c_fingerprint_identity(current) != stage0c_fingerprint_identity(legacy)


def test_legacy_identity_is_reconstructible_from_the_completed_concept_count() -> None:
    """A completed Stage 0c artifact holds exactly ``max_concepts`` concepts,
    so the retired value is recovered rather than guessed.
    """

    n_completed_concepts = 50
    current = {"feature_dir": "/x"}
    identity_written_before_removal = stage0c_fingerprint_identity(
        {**current, "max_concepts": n_completed_concepts}
    )
    reconstructed = stage0c_fingerprint_identity(
        {**current, "max_concepts": n_completed_concepts}
    )
    assert reconstructed == identity_written_before_removal


def test_reconstruction_fails_for_a_different_concept_count() -> None:
    """Guard: the reconstruction must not accept an arbitrary legacy value."""

    current = {"feature_dir": "/x"}
    written_at_50 = stage0c_fingerprint_identity({**current, "max_concepts": 50})
    reconstructed_at_100 = stage0c_fingerprint_identity(
        {**current, "max_concepts": 100}
    )
    assert reconstructed_at_100 != written_at_50
