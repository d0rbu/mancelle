"""Tests for PR-2 fingerprint additions and schema version bump."""

from __future__ import annotations

import json

import pytest

from steering.eval.axbench_base._stage0c_internals import (
    _STAGE0C_CHECKPOINT_SCHEMA_VERSION,
    _compute_checkpoint_fingerprint,
)
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
    LLMClientConfig,
    VerificationPrefillMode,
)


def _fake_cfg(**overrides: object) -> AxBenchBaseConfig:
    """Build a real config containing every fingerprint input.

    A permissive ``MagicMock`` silently manufactures newly accessed config
    attributes.  That made prior schema additions appear to work until a test
    serialized the fingerprint, where the manufactured values failed much
    later with an unrelated ``TypeError``.  A real config makes a missing or
    misspelled field fail at construction time and exercises the same value
    types as production.
    """
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={},
        encdec={},
        device="cpu",
        output_root="/tmp/axbench-base-test",
        validation_shards=[0],
        neuronpedia_model_id="gemma-2-2b",
        feature_extraction_model_id=None,
        neuronpedia_source="8-gemmascope-res-16k",
        extraction_tokenizer_id="google/gemma-2-2b",
        max_concepts=2,
        n_verified_prefixes_per_concept_target=1,
        n_verification_seeds=1,
        min_consistency_passes=1,
        verification_percentile_threshold=0.7,
        verification_length=64,
        min_prefix_chars=1,
        max_prefix_chars=100,
        verification_zero_hit_bail_after=None,
        verification_candidate_attempt_limit=None,
        verification_candidate_scan_limit=None,
        enable_concept_examples_qc=False,
        concept_examples_qc_judge=LLMClientConfig(model="x"),
        activation_example_judge_min_score=1,
        concept_examples_qc_target_surviving=None,
        generation=GenerationConfig(
            master_seed=1,
            temperature=0.7,
            top_k_sampling=50,
        ),
        qc_batch_size=16,
        qc_pipeline=True,
        verification_prefill_mode=VerificationPrefillMode.SHARED,
        stage0c_multi_gpu_block_size=16,
        verification_cross_prefix_batch_size=4,
        verification_api_model=None,
    )
    return cfg.replace(**overrides)


def test_fingerprint_includes_qc_batch_size():
    fp = _compute_checkpoint_fingerprint(_fake_cfg(qc_batch_size=8))
    assert fp["qc_batch_size"] == 8


def test_fingerprint_includes_qc_pipeline():
    fp = _compute_checkpoint_fingerprint(_fake_cfg(qc_pipeline=False))
    assert fp["qc_pipeline"] is False


def test_fingerprint_includes_verification_prefill_mode():
    fp = _compute_checkpoint_fingerprint(_fake_cfg(verification_prefill_mode="repeated"))
    assert fp["verification_prefill_mode"] == "repeated"
    assert "batched_generation_mode" not in fp


def test_fingerprint_changes_when_qc_batch_size_changes():
    fp_a = _compute_checkpoint_fingerprint(_fake_cfg(qc_batch_size=4))
    fp_b = _compute_checkpoint_fingerprint(_fake_cfg(qc_batch_size=16))
    assert fp_a != fp_b


def test_fingerprint_changes_when_qc_pipeline_changes():
    fp_a = _compute_checkpoint_fingerprint(_fake_cfg(qc_pipeline=False))
    fp_b = _compute_checkpoint_fingerprint(_fake_cfg(qc_pipeline=True))
    assert fp_a != fp_b


def test_fingerprint_changes_when_verification_prefill_mode_changes():
    fp_a = _compute_checkpoint_fingerprint(_fake_cfg(verification_prefill_mode="shared"))
    fp_b = _compute_checkpoint_fingerprint(_fake_cfg(verification_prefill_mode="repeated"))
    assert fp_a != fp_b


def test_schema_version_was_bumped():
    """Schema version must be > 1 to invalidate any pre-PR-2 checkpoints."""
    # Whatever the prior value was, the new value must be strictly greater.
    # We don't pin the exact value because it may have been bumped multiple
    # times; we just verify it's >= 2 (any pre-PR-2 baseline value).
    assert _STAGE0C_CHECKPOINT_SCHEMA_VERSION >= 2, (
        f"Expected schema version >= 2 after PR-2 bump; got "
        f"{_STAGE0C_CHECKPOINT_SCHEMA_VERSION}"
    )


def test_schema_version_was_bumped_for_pr4_block_size():
    """Schema version must be >= 3 after the PR-4 block-size field bump.

    The PR-4 fingerprint adds ``stage0c_multi_gpu_block_size``; any
    pre-PR-4 partial checkpoint cannot be safely resumed because the
    block boundaries shift when B changes, so the schema version must
    be strictly greater than the PR-2 baseline of 2.
    """
    assert _STAGE0C_CHECKPOINT_SCHEMA_VERSION >= 3, (
        f"Expected schema version >= 3 after PR-4 bump; got "
        f"{_STAGE0C_CHECKPOINT_SCHEMA_VERSION}"
    )


def test_fingerprint_includes_stage0c_multi_gpu_block_size():
    fp = _compute_checkpoint_fingerprint(_fake_cfg(stage0c_multi_gpu_block_size=8))
    assert fp["stage0c_multi_gpu_block_size"] == 8


def test_fingerprint_includes_verification_cross_prefix_batch_size():
    fp = _compute_checkpoint_fingerprint(
        _fake_cfg(verification_cross_prefix_batch_size=8),
    )
    assert fp["verification_cross_prefix_batch_size"] == 8


def test_fingerprint_changes_when_stage0c_multi_gpu_block_size_changes():
    fp_a = _compute_checkpoint_fingerprint(
        _fake_cfg(stage0c_multi_gpu_block_size=8),
    )
    fp_b = _compute_checkpoint_fingerprint(
        _fake_cfg(stage0c_multi_gpu_block_size=32),
    )
    assert fp_a != fp_b


def test_fingerprint_changes_when_verification_cross_prefix_batch_size_changes():
    fp_a = _compute_checkpoint_fingerprint(
        _fake_cfg(verification_cross_prefix_batch_size=1),
    )
    fp_b = _compute_checkpoint_fingerprint(
        _fake_cfg(verification_cross_prefix_batch_size=8),
    )
    assert fp_a != fp_b


def test_schema_version_was_bumped_for_cross_prefix_batch_size():
    assert _STAGE0C_CHECKPOINT_SCHEMA_VERSION >= 4, (
        f"Expected schema version >= 4 after cross-prefix batching bump; got "
        f"{_STAGE0C_CHECKPOINT_SCHEMA_VERSION}"
    )


def test_schema_version_is_eight_for_versioned_prefill_semantics():
    assert _STAGE0C_CHECKPOINT_SCHEMA_VERSION == 8


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("neuronpedia_model_id", "gemma-3-4b"),
        ("feature_extraction_model_id", "gemma-3-4b-pt"),
        ("neuronpedia_source", "22-gemmascope-2-16k"),
    ],
)
def test_fingerprint_includes_extractor_identity_field(
    field_name: str,
    value: str,
) -> None:
    fp = _compute_checkpoint_fingerprint(_fake_cfg(**{field_name: value}))
    assert fp[field_name] == value


@pytest.mark.parametrize(
    ("field_name", "first", "second"),
    [
        ("neuronpedia_model_id", "gemma-2-2b", "gemma-3-4b"),
        (
            "feature_extraction_model_id",
            "gemma-2-2b",
            "gemma-3-4b-pt",
        ),
        (
            "neuronpedia_source",
            "8-gemmascope-res-16k",
            "22-gemmascope-2-16k",
        ),
    ],
)
def test_fingerprint_changes_when_extractor_identity_changes(
    field_name: str,
    first: str,
    second: str,
) -> None:
    fp_a = _compute_checkpoint_fingerprint(_fake_cfg(**{field_name: first}))
    fp_b = _compute_checkpoint_fingerprint(_fake_cfg(**{field_name: second}))
    assert fp_a[field_name] != fp_b[field_name]
    assert fp_a != fp_b


def test_feature_extraction_model_id_uses_effective_fallback_identity() -> None:
    implicit = _compute_checkpoint_fingerprint(
        _fake_cfg(
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id=None,
        )
    )
    explicit = _compute_checkpoint_fingerprint(
        _fake_cfg(
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b",
        )
    )

    assert implicit["feature_extraction_model_id"] == "gemma-3-4b"
    assert implicit == explicit


def test_fingerprint_is_json_round_trip_safe() -> None:
    """Checkpoint fingerprints must remain directly JSON serializable."""
    fingerprint = _compute_checkpoint_fingerprint(
        _fake_cfg(feature_extraction_model_id="gemma-3-4b-pt")
    )
    assert json.loads(json.dumps(fingerprint)) == fingerprint


def test_fingerprint_is_deterministic():
    """Same cfg -> same fingerprint (no nondeterminism)."""
    fp1 = _compute_checkpoint_fingerprint(_fake_cfg())
    fp2 = _compute_checkpoint_fingerprint(_fake_cfg())
    assert fp1 == fp2


def test_fingerprint_dict_has_exact_key_count():
    """Sentinel: adding a fingerprint field requires bumping
    ``_STAGE0C_CHECKPOINT_SCHEMA_VERSION``.

    A future PR that silently extends ``_compute_checkpoint_fingerprint``
    without bumping the schema version would let a stale-on-disk
    partial checkpoint resume under the new key set, producing
    silent corruption. This test fails when the key count drifts so
    the schema bump is not silently skipped.

    If you intentionally added a fingerprint key:
    1. Bump ``_STAGE0C_CHECKPOINT_SCHEMA_VERSION`` in
       ``_stage0c_internals.py``.
    2. Update the expected count below to match.
    3. Update ``test_fingerprint_dict_has_exact_key_set`` with the
       new key.
    """
    fp = _compute_checkpoint_fingerprint(_fake_cfg())
    assert len(fp) == 28, (
        f"Fingerprint key count changed: {len(fp)}. Either revert the "
        "change OR bump _STAGE0C_CHECKPOINT_SCHEMA_VERSION and update "
        f"this assertion. Current keys: {sorted(fp.keys())!r}"
    )


def test_fingerprint_dict_has_exact_key_set():
    """Sentinel companion: a string-equality check on the full key set.

    Stronger than the count check — catches a swap (e.g., one key
    removed and a different one added with the same total count).
    Bump the schema version AND update this expected set when adding a
    fingerprint field.
    """
    fp = _compute_checkpoint_fingerprint(_fake_cfg())
    expected = {
        "stage0c_algorithm_version",
        "feature_dir",
        "feature_statistics_path",
        "validation_shards",
        "neuronpedia_model_id",
        "feature_extraction_model_id",
        "neuronpedia_source",
        "extraction_tokenizer_id",
        "n_verified_prefixes_per_concept_target",
        "n_verification_seeds",
        "min_consistency_passes",
        "verification_percentile_threshold",
        "verification_length",
        "min_prefix_chars",
        "max_prefix_chars",
        "verification_zero_hit_bail_after",
        "verification_candidate_attempt_limit",
        "verification_candidate_scan_limit",
        "enable_concept_examples_qc",
        "concept_examples_qc_config",
        "generation_master_seed",
        "generation_temperature",
        "generation_top_k_sampling",
        "qc_batch_size",
        "qc_pipeline",
        "verification_prefill_mode",
        "stage0c_multi_gpu_block_size",
        "verification_cross_prefix_batch_size",
    }
    assert set(fp.keys()) == expected, (
        f"Fingerprint key set changed; either revert or bump "
        f"_STAGE0C_CHECKPOINT_SCHEMA_VERSION and update expected. "
        f"Got: {sorted(fp.keys())!r}, expected: {sorted(expected)!r}"
    )


# ---------------------------------------------------------------------------
# `max_concepts` is a stopping condition, not part of Stage 0c identity.
#
# Two runs differing only in `max_concepts` verify each candidate identically
# and walk the same ordered candidate pool; the larger one simply walks
# further. Binding it into the fingerprint forced a full re-verification of
# already-decided candidates whenever the target was raised.
# ---------------------------------------------------------------------------


def test_fingerprint_excludes_max_concepts():
    fp = _compute_checkpoint_fingerprint(_fake_cfg(max_concepts=7))
    assert "max_concepts" not in fp


def test_fingerprint_is_identical_across_differing_max_concepts():
    """The whole point: raising the target must not invalidate prior work."""

    fp_small = _compute_checkpoint_fingerprint(_fake_cfg(max_concepts=50))
    fp_large = _compute_checkpoint_fingerprint(_fake_cfg(max_concepts=100))
    assert fp_small == fp_large


def test_fingerprint_still_changes_for_a_genuine_semantic_field():
    """Guard against the removal having over-reached.

    `n_verified_prefixes_per_concept_target` DOES change how each candidate is
    verified, so it must remain part of identity.
    """

    fp_a = _compute_checkpoint_fingerprint(
        _fake_cfg(n_verified_prefixes_per_concept_target=1)
    )
    fp_b = _compute_checkpoint_fingerprint(
        _fake_cfg(n_verified_prefixes_per_concept_target=2)
    )
    assert fp_a != fp_b
