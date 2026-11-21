"""Unit tests for identity-based reuse of committed clustered bags (#1050).

Each rejection case is pinned to its OWN ``match=`` regex. A shared or bare
``pytest.raises`` would let a mutation that collapses two distinct guards into one
survive the suite, which is the specific hole these tests exist to close.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    VOCAB_FINGERPRINT_META_KEY,
    write_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_identity import (
    BAG_DERIVE_SCHEMA_VERSION,
    BAG_DERIVE_SCHEMA_VERSION_KEY,
    DIAGNOSTIC_METADATA_KEYS,
    OPTIONAL_IDENTITY_DEFAULTS,
    SELF_CONSISTENT_METADATA_KEYS,
    bag_identity_view,
    committed_bag_for_reuse,
)

_FINGERPRINT = "a" * 64
_VOCAB = "uncompacted"
_BAG = FeatureConditioning(entries=((7, 1), (11, 2), (23, 0)))


def _identity(**overrides: str) -> dict[str, str]:
    """The identity a fair-GCG derive would stamp, minus diagnostics/self-consistent."""
    base = {
        "concept_id": "1",
        "mode": "negative",
        "mode_tag": "fair",
        "reg_lambda": "0.0",
        "n_optimize_prefixes": "2",
        "n_held_out_prefixes": "2",
        "n_optimize_members": "4",
        "n_held_out_members": "4",
        "n_rich_completions": "2",
        "n_avoid_completions": "2",
        "k_pos": "3",
        "k_neg": "3",
        "positive_bucket": "1",
        "negative_bucket": "0",
        "extraction_batch_size": "1",
        "master_seed": "0",
        "split_ratio": "0.5",
        "split_seed": "0",
        "max_steps": "1",
        "minibatch_size": "1",
        FEATURE_CONTRACT_FINGERPRINT_META_KEY: _FINGERPRINT,
        VOCAB_FINGERPRINT_META_KEY: _VOCAB,
        BAG_DERIVE_SCHEMA_VERSION_KEY: BAG_DERIVE_SCHEMA_VERSION,
    }
    base.update(overrides)
    return base


def _commit(
    path: Path,
    *,
    identity: dict[str, str] | None = None,
    diagnostics: dict[str, str] | None = None,
    bag: FeatureConditioning = _BAG,
) -> Path:
    """Write a committed bag through the REAL writer, so the pairing is under test."""
    ident = dict(_identity() if identity is None else identity)
    vocab = ident.pop(VOCAB_FINGERPRINT_META_KEY)  # writer owns this key exclusively
    metadata = {
        **ident,
        "n_features": str(len(bag)),
        **(
            diagnostics
            if diagnostics is not None
            else {
                "warm_start_g_steer": "1.0",
                "optimize_g_steer": "1.0883698463439941",
                "held_out_g_steer": "1.1152758598327637",
            }
        ),
    }
    write_ours_gcg_bag(path, bag, metadata=metadata, vocab_fingerprint=vocab)
    return path


def _reuse(path: Path, *, expected: dict[str, str] | None = None):
    return committed_bag_for_reuse(
        path,
        expected_identity=_identity() if expected is None else expected,
        expected_feature_contract_fingerprint=_FINGERPRINT,
    )


# --- 1. the partition is total, disjoint, and fail-closed -------------------------


def test_diagnostic_and_self_consistent_sets_are_disjoint() -> None:
    assert not (DIAGNOSTIC_METADATA_KEYS & SELF_CONSISTENT_METADATA_KEYS)


def test_unknown_metadata_key_is_treated_as_identity() -> None:
    """Fail-closed: a key nobody classified must gate reuse, not be ignored.

    If this inverts, a future metadata key silently stops mattering — the exact
    failure mode the explicit-sets design exists to prevent.
    """
    view = bag_identity_view({"a_brand_new_key": "v", "optimize_g_steer": "1.0"})
    assert view == {"a_brand_new_key": "v"}


def test_identity_view_excludes_every_declared_non_identity_key() -> None:
    metadata = dict.fromkeys(DIAGNOSTIC_METADATA_KEYS | SELF_CONSISTENT_METADATA_KEYS, "x")
    metadata["concept_id"] = "1"
    assert bag_identity_view(metadata) == {"concept_id": "1"}


# --- 2. reuse decision table -----------------------------------------------------


def test_absent_artifact_returns_none_to_derive(tmp_path: Path) -> None:
    assert _reuse(tmp_path / "missing.safetensors") is None


def test_matching_identity_is_reused(tmp_path: Path) -> None:
    path = _commit(tmp_path / "bag.safetensors")
    committed = _reuse(path)
    assert committed is not None
    assert committed.bag == _BAG


def test_reuse_ignores_differing_diagnostics(tmp_path: Path) -> None:
    """THE #1050 case: only the measured objectives moved, so reuse must succeed."""
    path = _commit(
        tmp_path / "bag.safetensors",
        diagnostics={
            "warm_start_g_steer": "1.0",
            "optimize_g_steer": "1.3040275573730469",
            "held_out_g_steer": "1.2465534210205078",
        },
    )
    assert _reuse(path) is not None


@pytest.mark.parametrize(
    "key,changed",
    [
        (FEATURE_CONTRACT_FINGERPRINT_META_KEY, "b" * 64),
        ("master_seed", "1"),
        ("max_steps", "128"),
        ("split_ratio", "0.75"),
        ("n_held_out_members", "5"),
        ("positive_bucket", "2"),
        ("concept_id", "2"),
        (BAG_DERIVE_SCHEMA_VERSION_KEY, "2"),
    ],
)
def test_identity_mismatch_raises(tmp_path: Path, key: str, changed: str) -> None:
    """A genuine input change must stop the run, never be reused over."""
    path = _commit(tmp_path / "bag.safetensors")
    expected = _identity(**{key: changed})
    if key == FEATURE_CONTRACT_FINGERPRINT_META_KEY:
        # The reader's own two-way fingerprint rule fires first, which is correct:
        # one reader, one set of rules (see committed_bag_for_reuse docstring).
        with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
            committed_bag_for_reuse(
                path,
                expected_identity=expected,
                expected_feature_contract_fingerprint=changed,
            )
        return
    with pytest.raises(AxBenchBpbAdvError, match="does not match this run's identity"):
        _reuse(path, expected=expected)


def test_vocab_fingerprint_participates_in_identity(tmp_path: Path) -> None:
    """A different checkpoint must not be reused across.

    The writer owns this key exclusively, so the reuse caller has to add it to the
    expected identity by hand; if that assembly is dropped the checkpoint silently
    leaves the identity.
    """
    path = _commit(tmp_path / "bag.safetensors")
    with pytest.raises(AxBenchBpbAdvError, match="does not match this run's identity"):
        _reuse(path, expected=_identity(**{VOCAB_FINGERPRINT_META_KEY: "other-v"}))


def test_missing_mandatory_identity_key_raises(tmp_path: Path) -> None:
    identity = _identity()
    del identity["master_seed"]
    path = _commit(tmp_path / "bag.safetensors", identity=identity)
    with pytest.raises(AxBenchBpbAdvError, match="missing identity key"):
        _reuse(path)


def test_absent_schema_version_defaults_and_is_reused(tmp_path: Path) -> None:
    """Legacy artifacts predate the version key and must keep resuming (§5.8)."""
    identity = _identity()
    del identity[BAG_DERIVE_SCHEMA_VERSION_KEY]
    path = _commit(tmp_path / "bag.safetensors", identity=identity)
    assert BAG_DERIVE_SCHEMA_VERSION_KEY in OPTIONAL_IDENTITY_DEFAULTS
    assert _reuse(path) is not None


def test_internally_inconsistent_n_features_raises(tmp_path: Path) -> None:
    """n_features cannot be compared against a run that has not derived, so it is
    validated against the artifact's own tensors instead."""
    path = tmp_path / "bag.safetensors"
    identity = _identity()
    vocab = identity.pop(VOCAB_FINGERPRINT_META_KEY)
    write_ours_gcg_bag(
        path,
        _BAG,
        metadata={**identity, "n_features": "999", "optimize_g_steer": "1.0"},
        vocab_fingerprint=vocab,
    )
    with pytest.raises(AxBenchBpbAdvError, match="internally inconsistent"):
        _reuse(path)


def test_unreadable_artifact_raises_rather_than_rederiving(tmp_path: Path) -> None:
    path = tmp_path / "bag.safetensors"
    path.write_bytes(b"not safetensors at all")
    with pytest.raises(AxBenchBpbAdvError, match="could not be read for reuse"):
        _reuse(path)


# --- 3. the mismatch message must be actionable ----------------------------------


def test_mismatch_message_names_every_differing_key(tmp_path: Path) -> None:
    """The opaque byte-compare message cost a full debug cycle to attribute.

    Catches an early-return implementation that reports only the first difference.
    """
    path = _commit(tmp_path / "bag.safetensors")
    expected = _identity(master_seed="9", max_steps="128", concept_id="7")
    with pytest.raises(AxBenchBpbAdvError) as excinfo:
        _reuse(path, expected=expected)
    message = str(excinfo.value)
    for key, committed, this_run in (
        ("master_seed", "'0'", "'9'"),
        ("max_steps", "'1'", "'128'"),
        ("concept_id", "'1'", "'7'"),
    ):
        assert key in message
        assert f"committed={committed}" in message
        assert f"this_run={this_run}" in message


# --- 4. every writer's real metadata shape is classified correctly ----------------


@pytest.mark.parametrize(
    "measured_key",
    [
        "warm_start_g_steer",
        "optimize_g_steer",
        "held_out_g_steer",
        "in_sample_g_steer",
        "warm_penalized_lm_objective",
        "optimized_penalized_lm_objective",
    ],
)
def test_every_writers_measured_key_is_diagnostic(measured_key: str) -> None:
    """All five clustered writers, not just the fair-GCG one.

    Roofline stamps ``in_sample_g_steer`` and general-val stamps the two
    ``*_penalized_lm_objective`` values. Under the fail-closed rule an undeclared
    measured key counts as identity, which would make reuse ALWAYS fail for that
    writer — silently reintroducing #1050 for it.
    """
    assert measured_key in DIAGNOSTIC_METADATA_KEYS
    assert bag_identity_view({measured_key: "1.5", "concept_id": "1"}) == {
        "concept_id": "1"
    }


@pytest.mark.parametrize("config_key", ["split_ratio", "extraction_threshold"])
def test_config_derived_float_keys_stay_identity(config_key: str) -> None:
    """These are ``repr(float(...))`` too, but they come from config.

    Misreading "is a float repr" as "is a diagnostic" would let a real config change
    reuse a bag derived under different settings.
    """
    assert config_key not in DIAGNOSTIC_METADATA_KEYS
    assert bag_identity_view({config_key: "0.5"}) == {config_key: "0.5"}


def test_str_path_is_accepted(tmp_path: Path) -> None:
    """``paths.ours_gcg_bag`` returns a str at some call sites.

    ``write_ours_gcg_bag`` accepts str-or-Path, so the reuse check that guards it must
    not be stricter. Regression: an earlier revision annotated ``Path`` and called
    ``.exists()`` directly, which the clustered warm-start suite caught with
    ``AttributeError: 'str' object has no attribute 'exists'``.
    """
    committed = _commit(tmp_path / "bag.safetensors")
    assert (
        committed_bag_for_reuse(
            str(committed),
            expected_identity=_identity(),
            expected_feature_contract_fingerprint=_FINGERPRINT,
        )
        is not None
    )
    assert (
        committed_bag_for_reuse(
            str(tmp_path / "absent.safetensors"),
            expected_identity=_identity(),
            expected_feature_contract_fingerprint=_FINGERPRINT,
        )
        is None
    )


def test_absent_n_features_is_tolerated(tmp_path: Path) -> None:
    """Deliberate, and pinned so it is not "fixed" into a hard failure later.

    ``n_features`` is redundant with the tensors: reuse consumes the tensors and never
    this count, so a missing stamp cannot make a reused bag wrong. Requiring it would
    turn a benign omission into a resume-blocking error. Disagreement is still fatal --
    ``test_internally_inconsistent_n_features_raises`` covers that direction.
    """
    path = tmp_path / "bag.safetensors"
    identity = _identity()
    vocab = identity.pop(VOCAB_FINGERPRINT_META_KEY)
    write_ours_gcg_bag(
        path,
        _BAG,
        metadata={**identity, "optimize_g_steer": "1.0"},  # no n_features
        vocab_fingerprint=vocab,
    )
    committed = _reuse(path)
    assert committed is not None
    assert committed.bag == _BAG


def test_every_ours_gcg_config_field_is_identity_or_explicitly_excused() -> None:
    """Standing guard against the defect class review found in #1054.

    ``max_forward_batch`` was not a *misclassified* metadata key -- the fail-closed
    partition would have caught that. It was a derive input never stamped into the
    metadata at all, which no amount of metadata classification can detect. A new
    ``OursGcgConfig`` field could repeat that silently.

    So every field must be either stamped into the GCG identity or listed below with a
    reason. Adding a field without doing either fails here rather than at the next
    adversarial review.
    """
    from dataclasses import fields as dc_fields

    from steering.eval.axbench_bpb_adv.config import OursGcgConfig
    from steering.eval.axbench_bpb_adv.gcg_bags import _fair_bag_identity

    #: field -> why it does not need to appear verbatim in the identity.
    EXCUSED = {
        # The identity carries the PER-BAG reg_lambda; the grid is only the
        # iteration set the derive walks.
        "reg_lambda_grid": "per-bag reg_lambda is stamped instead",
        # Same shape as reg_lambda_grid: the identity carries the PER-BAG
        # objective_alpha (stamped below by passing a non-default value), while
        # the grid is only the iteration set. Note the per-bag stamp is
        # strip-default -- absent at alpha == 1 so pre-alpha bags stay
        # loadable -- which is pinned by the alpha strip-default tests.
        "objective_alpha_grid": "per-bag objective_alpha is stamped instead",
        # resolve_k turns these into k_pos/k_neg, which ARE stamped. Pinning the
        # resolved values is strictly stronger: two configs resolving to the same k
        # produce the same bag, so reuse is correct.
        "top_k_pos": "resolved into the stamped k_pos",
        "top_k_neg": "resolved into the stamped k_neg",
        # Bound only into _group_g_steer, which produces the measured *_g_steer
        # DIAGNOSTICS. It cannot change the bag, so it must not gate reuse.
        "mean_g_steer_batch": "affects diagnostics only, never the bag",
        # Consumed by _objective_factory AND the measurements; stamped explicitly.
        "split_ratio": "stamped",
        "split_seed": "stamped",
    }

    identity = _fair_bag_identity(
        cfg=SimpleNamespace(feature_diffmean_extraction_batch_size=1),
        ours=SimpleNamespace(
            master_seed=0,
            split_ratio=0.5,
            split_seed=0,
            max_steps=1,
            minibatch_size=1,
            max_forward_batch=1,
            max_grad_batch=1,
        ),
        concept_id=1,
        mode=SteeringMode.NEGATIVE,
        reg_lambda=0.0,
        # Deliberately NON-default so the per-bag alpha key is present: at the
        # default it is stripped, and this guard is about the key existing at all.
        objective_alpha=0.5,
        optimize_prefixes=(),
        held_out_prefixes=(),
        optimize_members=(),
        held_out_members=(),
        rich_items=(),
        avoid_items=(),
        k_pos=1,
        k_neg=1,
        positive_bucket=1,
        negative_bucket=0,
        cluster_contract=None,
    )

    unaccounted = sorted(
        f.name
        for f in dc_fields(OursGcgConfig)
        if f.name not in identity and f.name not in EXCUSED
    )
    assert not unaccounted, (
        f"OursGcgConfig field(s) {unaccounted} are neither stamped into the GCG bag "
        "identity nor excused. If such a field can change the derived bag, a resumed "
        "run would silently reuse a bag built under a different value (the "
        "max_forward_batch defect from the #1054 review). Stamp it, or add it to "
        "EXCUSED with the reason it cannot affect the bag."
    )
