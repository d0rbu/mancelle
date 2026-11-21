"""Identity-based reuse of committed clustered precomputed bags (#1050).

Why this exists
---------------
Clustered bags are installed through :func:`steering.immutable_io.install_bytes_once`,
which byte-compares an existing artifact against the payload a resumed run produces.
The feature-GCG derive cannot satisfy that: its per-step gradient is a CUDA backward
pass whose atomics do not promise a fixed accumulation order, and ``rank_candidates``
then *amplifies* a sub-ULP difference — a reordered top-k pool makes temperature
sampling draw a different candidate set even from an identical RNG stream. A resumed
run therefore commits a materially different bag and aborts.

Measured on the real l22 clustered checkpoint: with byte-identical inputs (proven by
a bitwise-identical ``warm_start_g_steer`` and every count/config/fingerprint key
matching), two consecutive derives of the same bag differed in 2 of 24 feature ids and
moved ``optimize_g_steer`` by ~20%.

Byte equality was the wrong contract to begin with. This repository already concedes
that real-model GPU numerics are only tolerance-comparable — the feature-GCG batching
equivalence is proven CPU/float32 near-exact but real-model bf16 at a *derived atol*
(``tests/steering/feature_gcg/_reference.py``). Even with
``torch.use_deterministic_algorithms(True)`` forced, resuming on a different GPU,
driver, or torch version would still diverge, so the requirement is environment-coupled
by construction.

What replaces it
----------------
On resume a committed bag is *reused* when its **identity** matches what this run would
stamp, and the derive is skipped. Measured objective values are diagnostics, not
identity. An identity mismatch is a genuine input change and raises.

Metadata is partitioned three ways, and the partition is deliberately **fail-closed**:
membership is defined by the two small explicit sets below, so any *new* metadata key
counts as identity by default. A key that ought to have gated reuse can never be
silently ignored by omission.

* :data:`DIAGNOSTIC_METADATA_KEYS` — measured objective values. Ignored for reuse.
* :data:`SELF_CONSISTENT_METADATA_KEYS` — result-derived, so there is nothing to
  compare against a run that has not derived; validated against the artifact's own
  tensors instead.
* everything else — identity, compared exactly.

Known limit (accepted, not overlooked)
--------------------------------------
Identity pins input *counts*, not input *content*: two different prefix sets of equal
size would pass. That is safe only because every clustered stage revalidates the
current source contract against the committed one before a derive is reached, so a
content change fails earlier. If that layer is ever relaxed, bag reuse stops being
safe and this paragraph is the reason why.

A *legacy* bag written before this module existed carries no
``bag_derive_schema_version``, so it validates as version ``"1"`` and stays reusable.
The residual is that such a bag reused under changed derive semantics is undetectable;
artifacts written from here on carry the version and are protected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag

#: MEASURED objective values, across every clustered bag writer. Not identity: they
#: are recomputed from a nondeterministic derive and are the proximate reason #1050
#: fires.
#:
#: The discriminator is *measured vs config-derived*, NOT "is it a ``repr(float)``".
#: ``split_ratio`` and ``extraction_threshold`` are also stamped as
#: ``repr(float(...))`` but come straight from config, are perfectly reproducible, and
#: MUST stay identity — reading them as diagnostics would stop a genuine config change
#: from invalidating a committed bag.
DIAGNOSTIC_METADATA_KEYS: frozenset[str] = frozenset(
    {
        # gcg_bags.py (fair)
        "warm_start_g_steer",
        "optimize_g_steer",
        "held_out_g_steer",
        # gcg_bags.py / feature_diffmean_bags.py (roofline)
        "in_sample_g_steer",
        # general_val.py
        "warm_penalized_lm_objective",
        "optimized_penalized_lm_objective",
    }
)

#: Derived from the resulting bag, so uncomparable before deriving. Checked against
#: the artifact's own tensors by :func:`committed_bag_for_reuse`.
SELF_CONSISTENT_METADATA_KEYS: frozenset[str] = frozenset({"n_features"})

#: Current derive-semantics version, stamped into every new bag's identity.
BAG_DERIVE_SCHEMA_VERSION_KEY = "bag_derive_schema_version"
BAG_DERIVE_SCHEMA_VERSION = "1"

#: Identity keys that may be ABSENT from an existing artifact, with the value to
#: assume when they are. Every other identity key is mandatory: a missing one means
#: the artifact predates a contract this run depends on, which is an error rather
#: than something to paper over.
OPTIONAL_IDENTITY_DEFAULTS: Mapping[str, str] = {
    BAG_DERIVE_SCHEMA_VERSION_KEY: BAG_DERIVE_SCHEMA_VERSION,
}


@dataclass(frozen=True)
class CommittedBag:
    """A committed bag validated as reusable, plus its metadata."""

    bag: FeatureConditioning
    metadata: Mapping[str, str]


def bag_identity_view(metadata: Mapping[str, str]) -> dict[str, str]:
    """The identity projection of bag metadata.

    Everything except the diagnostic and self-consistent keys. Fail-closed: an
    unrecognised key is identity.
    """

    excluded = DIAGNOSTIC_METADATA_KEYS | SELF_CONSISTENT_METADATA_KEYS
    return {key: value for key, value in metadata.items() if key not in excluded}


def _describe_identity_mismatch(
    *,
    path: Path,
    expected: Mapping[str, str],
    actual: Mapping[str, str],
) -> str:
    """Name EVERY differing key with both values.

    Reporting only the first difference (or none, as the bare byte-compare did) is
    what made #1050 cost a full debug cycle to attribute.
    """

    lines: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        want, got = expected.get(key), actual.get(key)
        if want != got:
            lines.append(f"    {key}: committed={got!r} this_run={want!r}")
    return (
        f"committed clustered precomputed bag at {path} does not match this run's "
        "identity, so it cannot be reused and must not be overwritten. Use a fresh "
        "run_tag if the change is intended. Differing keys:\n" + "\n".join(lines)
    )


def committed_bag_for_reuse(
    path: str | Path,
    *,
    expected_identity: Mapping[str, str],
    expected_feature_contract_fingerprint: str | None,
) -> CommittedBag | None:
    """Return the committed bag when it is reusable, or ``None`` to derive.

    ``None`` selects a real alternate behaviour (derive and write), so it is a genuine
    optional rather than a deferred error.

    Loading goes through :func:`read_ours_gcg_bag` rather than a bespoke reader so the
    reader's own two-way feature-contract-fingerprint rule and structural checks apply
    unchanged — one reader, one set of rules.

    Raises:
        AxBenchBpbAdvError: the artifact is unreadable, internally inconsistent, or
            carries an identity that differs from this run's.
    """

    # ``paths.ours_gcg_bag`` returns a str on some call sites, and
    # ``write_ours_gcg_bag`` deliberately accepts either, so accept both here too
    # rather than making the reuse check stricter than the writer it guards.
    path = Path(path)
    if not path.exists():
        return None

    try:
        raw_bag, metadata = read_ours_gcg_bag(
            path,
            expected_feature_contract_fingerprint=expected_feature_contract_fingerprint,
        )
    except AxBenchBpbAdvError:
        raise
    except Exception as exc:  # unreadable / not safetensors / missing tensors
        raise AxBenchBpbAdvError(
            f"committed clustered precomputed bag at {path} could not be read for "
            f"reuse: {type(exc).__name__}: {exc}. Refusing to silently re-derive over "
            "an existing immutable artifact."
        ) from exc

    # Absence is tolerated deliberately, not by oversight. ``n_features`` is redundant
    # with the tensors -- reuse consumes the tensors, never this count -- so a missing
    # stamp cannot make a reused bag wrong. Requiring it would add a hard resume
    # failure for zero safety gain. When present it must agree, because disagreement
    # means the artifact is internally inconsistent and should not be trusted.
    recorded_n_features = metadata.get("n_features")
    if recorded_n_features is not None and str(len(raw_bag)) != recorded_n_features:
        raise AxBenchBpbAdvError(
            f"committed clustered precomputed bag at {path} is internally "
            f"inconsistent: metadata n_features={recorded_n_features!r} but the "
            f"artifact holds {len(raw_bag)} entries."
        )

    actual_identity = bag_identity_view(metadata)
    for key, default in OPTIONAL_IDENTITY_DEFAULTS.items():
        actual_identity.setdefault(key, default)

    missing = sorted(set(expected_identity) - set(actual_identity))
    if missing:
        raise AxBenchBpbAdvError(
            f"committed clustered precomputed bag at {path} is missing identity "
            f"key(s) {missing}, so it predates a contract this run depends on and "
            "cannot be shown to match. Use a fresh run_tag."
        )

    if actual_identity != dict(expected_identity):
        raise AxBenchBpbAdvError(
            _describe_identity_mismatch(
                path=path, expected=expected_identity, actual=actual_identity
            )
        )

    return CommittedBag(
        bag=FeatureConditioning(entries=tuple(raw_bag.items())),
        metadata=metadata,
    )


__all__ = [
    "BAG_DERIVE_SCHEMA_VERSION",
    "BAG_DERIVE_SCHEMA_VERSION_KEY",
    "DIAGNOSTIC_METADATA_KEYS",
    "OPTIONAL_IDENTITY_DEFAULTS",
    "SELF_CONSISTENT_METADATA_KEYS",
    "CommittedBag",
    "bag_identity_view",
    "committed_bag_for_reuse",
]
