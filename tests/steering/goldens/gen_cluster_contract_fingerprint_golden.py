"""Regenerate the committed B3 fingerprint-differential golden (Codex fix
batch finding 1, HIGH -- the A5 golden pattern applied to the oracle-vs-stack
``ClusterFeatureContract`` differential).

**Why this replaces the previous live-oracle-import test.** The prior
``tests/steering/test_cluster_feature_contract_fingerprint_differential.py``
loaded the frozen oracle's standalone ``cluster_feature_contract.py`` from an
untracked, external sibling worktree (``../pr739-oracle``) at TEST TIME via
``importlib.util.spec_from_file_location`` -- non-hermetic (a normal clone
without that worktree fails every test in the module), CI-broken (no such
worktree exists in CI), and unpinned (mutating that worktree silently changes
what "the oracle" means, with no digest check). This module is a deterministic
CAPTURE UTILITY, **not a test** -- mirroring
``tests/steering/goldens/gen_cluster_selection_golden.py``'s own A5 precedent:
it is the ONLY place that still touches the oracle worktree, and only when
someone deliberately reruns it to regenerate the committed golden. The
consuming test (``tests/steering/test_cluster_feature_contract_fingerprint_
differential.py``) imports only the ``run_tier{1,2}_cell_stack`` functions
below -- it never loads the oracle and has zero runtime dependency on the
``pr739-oracle`` worktree existing.

**Regeneration procedure.** ``tests`` is not an installed package, so the repo
root must be on ``PYTHONPATH``, and the frozen oracle worktree
(``.claude/worktrees/pr739-oracle``, pinned at commit ``_ORACLE_COMMIT``
below) must exist on disk at ``_ORACLE_PATH``:

    PYTHONPATH=. uv run python \
        tests/steering/goldens/gen_cluster_contract_fingerprint_golden.py

This writes ``tests/steering/goldens/cluster_contract_fingerprint_golden.json``
in place. Commit the result. Re-running is idempotent for every TIER 1 cell
and (after golden-side normalization -- see below) every TIER 2 cell.

**Golden schema.** ``{"schema_version", "oracle_worktree", "oracle_commit",
"oracle_source_sha256", "cells": {cell_name: {"tier": "tier1"|"tier2",
"oracle_outcome": {"raises": bool, "error_message": str|None,
"semantic_dict": dict|None, "fingerprint": str|None}}}, "stack_pinned_cells":
{cell_name: {"label": str, "normalized_semantic_dict": dict,
"normalized_semantic_sha256": str}}}``.
TIER 1 cells never touch a real file (their ``StableFileDigest``/statistics/
etc. are hand-built constants), so ``oracle_outcome["semantic_dict"]``/
``["fingerprint"]`` are the RAW, byte-literal ``semantic_dict()``/
``feature_contract_fingerprint`` -- fully reproducible. TIER 2 cells run the
real filesystem preflight, so their ``semantic_dict``/``fingerprint`` fields
hold the NORMALIZED form (see next section) and a SHA-256 of that normalized
form, not the raw, non-reproducible fingerprint. ``stack_pinned_cells``
(Codex fix batch finding 2, HIGH) holds cells whose value the oracle
structurally cannot produce at all (see ``STACK_PINNED_CELLS``) -- these are
captured from the STACK, not the oracle, and honestly labeled as such; the
consuming test additionally recomputes their SHA-256 independently at test
time and asserts it matches both this pinned golden value and the live
``contract.feature_contract_fingerprint``.

**Provenance pinning (Codex pass-2 residual finding 1, HIGH).** File
EXISTENCE proved nothing about the golden's PROVENANCE: ``load_oracle_
module`` used to check only ``is_file()`` before executing the oracle, and
``capture_golden`` then wrote the hardcoded ``_ORACLE_COMMIT`` string
regardless of what it had actually just run -- a mutated oracle source was
silently loaded, executed, and its output captured into a golden still
labeled with the frozen commit (adversarially demonstrated: a changed
oracle produced a golden honestly reproducing the changed semantics while
still claiming ``f357458b...``). ``load_oracle_module`` now hashes the
oracle SOURCE FILE'S raw bytes and asserts it equals the pinned
``EXPECTED_ORACLE_SOURCE_SHA256`` **before** a single line of the oracle
executes, failing loud on drift. ``capture_golden`` additionally records
that same digest as ``oracle_source_sha256`` in the golden JSON, and the
consuming test's ``test_golden_committed`` independently re-checks it
against the identical imported constant -- so a golden that was
regenerated against a drifted oracle (impossible past this file's own
check), or one that was hand-edited after the fact (still very possible --
this is a plain JSON file), is now rejected on either path.

**Provenance pinning, cache-injection closure (Codex pass-3 residual
finding, HIGH).** The pass-2 fix above still had a TOCTOU gap: ``load_
oracle_module`` returned any PRE-EXISTING ``sys.modules[_ORACLE_MODULE_
NAME]`` entry immediately, before the digest check ever ran, and ``capture_
golden`` separately re-hashed the genuine ``_ORACLE_PATH`` off disk for the
golden's own ``oracle_source_sha256`` field -- two independent reads of
"the oracle" that a cache-injected module could pry apart. Adversarially
demonstrated: a modified module object injected directly into that
``sys.modules`` slot executed untouched, while the golden still recorded
the real file's correct commit and source SHA-256, and ``test_golden_
committed`` passed -- the recorded provenance described different bytes
than what actually ran. ``load_oracle_module`` no longer consults ``sys.
modules`` on the way in at all: it reads the oracle source file's bytes
ONCE, hashes that buffer, asserts the digest (as before), then executes
THAT SAME buffer directly via ``compile``/``exec`` against a bare
``spec_from_loader(..., loader=None)`` module -- never a second,
independent reopen of the path -- and returns ``(module, source_sha256)``
as one pair. ``capture_golden`` records that returned digest verbatim
instead of calling ``oracle_source_sha256()`` a second time, so the
recorded provenance is now, by construction, describing the exact bytes
that ran. The module is written into ``sys.modules`` immediately BEFORE
exec runs (empirically required for the oracle's own frozen-dataclass
definitions to self-resolve during class-body execution -- see ``load_
oracle_module``'s own docstring), unconditionally overwriting any
pre-existing slot -- the injection vector is closed by removing the
check-and-return-early path on entry entirely, not by the exact timing of
this one assignment; a failed exec pops the entry back out rather than
leaving a broken module cached under the name.

**Golden reproducibility for TIER 2 (why raw file hashes cannot be pinned).**
Empirically confirmed (scratch probe, not committed; not re-derived here):
this repo's ``steering.stats_io.save_statistics`` (safetensors backend) and
plain ``torch.save`` BOTH produce DIFFERENT raw file bytes for IDENTICAL
logical tensor/metadata content across SEPARATE Python process invocations
(same class of hazard already documented for ``steering.feature_io.
save_features`` in this project's ``reference_save_features_nondeterminism``
memory -- safetensors metadata HashMap key-order randomization -- plus a
second, independent hazard in ``torch.save``'s own zip-based serialization).
This is pre-existing, out-of-scope library/format behavior, not a B3 defect.
Consequently ``checkpoint.semantic_dict()["model_sha256"]`` and
``statistics.canonical_dict()["file_sha256"]`` can NEVER be pinned as a
cross-run-reproducible literal for a real-filesystem TIER 2 fixture --
``_normalize_semantic_for_golden`` replaces both with a fixed sentinel before
the golden is written or compared against. Everything else in the semantic
dict (JSON-encoded metadata written with ``sort_keys=True`` from a fixed,
tmp-path-independent ``user_config.feature_dir``/``feature_clusters.path`` --
see ``_PINNED_FEATURE_DIR``/``_PINNED_CLUSTER_PATH`` below -- plus every
content-derived checksum like ``percentile_table_checksum``, which decodes
identically regardless of header byte order) IS reproducible and is compared
verbatim. TIER 1 cells need none of this: they never call ``write_
steering_stats``/``t.save`` at all.

**Fixture parity.** Every TIER 1/TIER 2 cell below builds its fixture from a
single ``module: ModuleType`` parameter -- either the loaded oracle module or
this repo's own ``steering.eval.cluster_feature_contract`` (imported
normally, as ``contract_module``) -- and calls ``module.<name>`` throughout.
This works because the oracle's single monolithic file defines every name
(``build_cluster_feature_contract``, ``FeatureStatisticsIdentity``,
``CheckpointPin``, etc.) at module level, identically shaped to how B1/B2/B3
compose and re-export them from the facade (verified: both namespaces expose
the same flat attribute surface). The ONE exception is ``StableFileDigest.
stat_signature``'s arity (oracle: 5-tuple; stack: 6-tuple, B1's fix) -- never
serialized into any semantic/canonical payload, so it cannot affect any
fingerprint; each side still needs its own correctly-shaped dummy value to
satisfy its own ``__post_init__``, selected via the ``is_oracle`` flag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch as t

import steering.eval.cluster_feature_contract as contract_module
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    build_clustered_selection_validation,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    build_clustered_selection_shard_integrity,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.splice_counterfactual import (
    packed_row_fingerprint,
    resolve_packed_row_fingerprint_strict,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    compute_selection_hash,
    load_packed_row_shard,
    load_selection_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.sae_resolver import ActivationType, SAERelease, SAEWidth, resolve_sae
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base.fixtures.steering_stats import write_steering_stats

GOLDEN_PATH = Path(__file__).parent / "cluster_contract_fingerprint_golden.json"

_ORACLE_PATH = Path(
    "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/.claude/worktrees/"
    "pr739-oracle/steering/eval/cluster_feature_contract.py"
)
_ORACLE_MODULE_NAME = "_pr739_b3_fingerprint_differential_oracle"
_ORACLE_COMMIT = "f357458b3871534a00372673fc4c40f2e62f2ce7"

# Codex pass-2 residual finding 1 (HIGH): SHA-256 of the frozen oracle's
# SOURCE FILE bytes at commit _ORACLE_COMMIT above -- computed via
# `sha256sum .claude/worktrees/pr739-oracle/steering/eval/cluster_feature_
# contract.py` (independently reproducible; not derived from any semantic
# content this module parses). Public (no leading underscore) so the
# consuming test can import and re-check it directly rather than trusting
# whatever value a golden claims for itself.
EXPECTED_ORACLE_SOURCE_SHA256 = "eb265239a1ee2f68fbce286098d29067b4ebf4f1d8da7decc39bbff5f3fbf8b9"

_A_SHA = "a" * 64
_B_SHA = "b" * 64
_C_SHA = "c" * 64
_D_SHA = "d" * 64

_NONDETERMINISTIC_SENTINEL = "<elided: nondeterministic raw-file-hash, see module docstring>"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def oracle_source_sha256(oracle_path: Path | None = None) -> str:
    """SHA-256 of the frozen oracle's SOURCE FILE bytes, read cold off disk.

    Deliberately computed from the file's raw bytes, never from the
    executed module's parsed semantics or any of its own hashing helpers --
    this is a standalone, independently-callable convenience for manual or
    test-time re-verification (e.g. a regression test wanting a second
    opinion on what the real on-disk oracle currently hashes to). **Not**
    called internally by ``load_oracle_module`` (see Codex pass-3 residual
    finding, HIGH, in that function's own docstring): that function reads
    and hashes its own buffer once, in-line, specifically so the digest it
    checks and the code it executes can never be two separate reads of the
    path with a TOCTOU window between them.

    ``oracle_path`` defaults to the CURRENT value of the module-level
    ``_ORACLE_PATH`` -- resolved here, inside the function body, rather
    than as ``oracle_path: Path = _ORACLE_PATH`` in the signature. A
    signature-level default is bound ONCE, at function-definition time
    (module import time); a caller that repoints the module global (e.g. a
    test simulating a mutated oracle by redirecting ``_ORACLE_PATH`` to a
    tampered copy) would silently keep hashing the ORIGINAL path forever.
    Resolving the default at call time closes that gap.
    """
    path = oracle_path if oracle_path is not None else _ORACLE_PATH
    return _sha256_bytes(path.read_bytes())


def load_oracle_module() -> tuple[ModuleType, str]:
    """Load the frozen oracle's standalone module under a distinct name,
    executing the EXACT byte buffer whose digest was just checked.

    Called ONLY from ``capture_golden`` (hence only when regenerating the
    golden) -- the consuming test never calls this and has no import-time
    dependency on the oracle worktree existing. Returns ``(module,
    source_sha256)`` -- the digest is the SAME one this function computed
    and verified before exec, so a caller (``capture_golden``) that records
    it is recording exactly what ran, not a second, independently-reopened
    reading of the path (see Codex pass-3 residual finding below).

    Codex pass-3 residual finding (HIGH): this function used to return any
    PRE-EXISTING ``sys.modules[_ORACLE_MODULE_NAME]`` entry immediately, on
    a bare membership check, BEFORE the digest check below ever ran --
    trusting the cache slot's contents unconditionally regardless of how
    they got there. ``capture_golden`` then executed whatever that
    returned module was, while separately re-hashing the genuine
    ``_ORACLE_PATH`` fresh off disk for the golden's own ``oracle_source_
    sha256`` field. Codex adversarially reproduced the gap end-to-end: a
    modified module was injected directly into ``sys.modules[_ORACLE_
    MODULE_NAME]``; the injected code executed; the golden nonetheless
    recorded the correct (real, unmodified file's) commit and source
    SHA-256, and ``test_golden_committed`` passed -- the recorded
    provenance described different bytes than what actually ran.

    Fix: this function no longer consults ``sys.modules`` on the way in at
    all, so a pre-populated (injected, poisoned, or merely stale-from-a-
    prior-call-in-the-same-process) cache slot can never be handed back
    instead of a freshly verified load. It reads the oracle source file's
    raw bytes exactly ONCE into a buffer, hashes THAT buffer, asserts the
    digest (still failing loud on drift, as before), and then executes
    THAT SAME buffer directly via ``compile``/``exec`` against a bare
    module built with ``spec_from_loader(..., loader=None)`` -- never a
    ``spec_from_file_location`` + ``exec_module`` reopen of the path, which
    would itself reintroduce a second read (and therefore a second TOCTOU
    window) between the hash and the execution.

    The freshly built module is written into ``sys.modules`` immediately
    BEFORE ``exec`` runs, not after -- empirically required (verified by
    direct reproduction: deferring this to after ``exec`` raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'`` deep
    in ``dataclasses._is_type``) because the oracle's own ``from __future__
    import annotations`` + ``@dataclass(frozen=True)`` combination needs
    ``sys.modules[module.__name__]`` to already resolve to this exact
    module WHILE its class bodies are executing, mirroring exactly what
    Python's own import machinery does for every ordinary module (``_load``
    inserts into ``sys.modules`` before calling ``exec_module``, precisely
    to support this kind of self-reference). This is NOT a reopening of the
    pass-3 gap: the vulnerability was ``load_oracle_module`` *checking for
    and returning* a pre-existing slot on entry, before any verification --
    removed above, with no ``sys.modules`` read anywhere before this point.
    The assignment below is reached only after a fresh read+hash+digest-
    assert has already succeeded in the current call, and it unconditionally
    OVERWRITES whatever (if anything) previously occupied the slot -- an
    injected/poisoned/stale prior entry is discarded, never consulted, never
    returned. If ``exec`` itself then raises, the just-inserted entry is
    popped back out rather than left behind as a broken half-initialized
    module parked under this name.
    """
    if not _ORACLE_PATH.is_file():
        raise FileNotFoundError(
            f"frozen oracle module not found at {_ORACLE_PATH}; golden regeneration "
            "cannot run without it. This is expected/harmless for anyone who is NOT "
            "regenerating the golden: the committed JSON plus this module's stack-"
            "only run_tier1_cell_stack/run_tier2_cell_stack functions are everything "
            "the consuming test needs."
        )
    # Codex pass-2 residual finding 1 (HIGH) + pass-3 residual finding
    # (HIGH, see docstring above): read the oracle SOURCE FILE's raw bytes
    # exactly ONCE here, hash THAT buffer, and exec THAT SAME buffer below
    # -- never a second, independent read of the path, and never a
    # pre-existing sys.modules[...] entry.
    source_bytes = _ORACLE_PATH.read_bytes()
    actual_source_sha256 = _sha256_bytes(source_bytes)
    if actual_source_sha256 != EXPECTED_ORACLE_SOURCE_SHA256:
        raise RuntimeError(
            f"frozen oracle source at {_ORACLE_PATH} has drifted: expected SHA-256 "
            f"{EXPECTED_ORACLE_SOURCE_SHA256}, got {actual_source_sha256}. Golden "
            "regeneration refuses to execute an oracle whose bytes no longer match "
            "what EXPECTED_ORACLE_SOURCE_SHA256/_ORACLE_COMMIT claim. If this drift "
            "is a deliberate oracle update: review the semantic diff by hand first, "
            "then update EXPECTED_ORACLE_SOURCE_SHA256 (and _ORACLE_COMMIT, if the "
            "oracle worktree was re-pinned to a new commit) deliberately -- never "
            "silently -- and regenerate + commit a fresh golden."
        )
    # Codex pass-3 residual finding (HIGH): build a spec with NO loader at
    # all (`loader=None`), so there is no `exec_module`/file-reopen code
    # path left that could substitute different bytes than the ones just
    # hashed above. `__file__` is set by hand so the oracle's own
    # path-relative machinery (if any; none is used today, but nothing here
    # should assume that) sees a sensible value.
    spec = importlib.util.spec_from_loader(_ORACLE_MODULE_NAME, loader=None)
    if spec is None:
        raise ImportError(f"could not build an import spec for oracle module {_ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(_ORACLE_PATH)
    # Written into sys.modules immediately BEFORE exec -- required (see
    # docstring) for the oracle's own frozen-dataclass definitions to
    # resolve during class-body execution, exactly mirroring ordinary
    # importlib semantics. This unconditionally overwrites any pre-existing
    # slot: unlike the pre-fix version, nothing above ever branched on
    # what (if anything) already occupied `_ORACLE_MODULE_NAME` -- this
    # line is reached only via a fresh read+hash+digest-assert that just
    # succeeded, so whatever gets clobbered here was never trusted or
    # consulted in the first place.
    sys.modules[_ORACLE_MODULE_NAME] = module
    try:
        exec(compile(source_bytes, str(_ORACLE_PATH), "exec"), module.__dict__)
    except BaseException:
        # Don't leave a broken, half-initialized module parked under this
        # name for some later lookup to stumble on -- mirrors importlib's
        # own cleanup-on-failed-exec behavior.
        sys.modules.pop(_ORACLE_MODULE_NAME, None)
        raise
    return module, actual_source_sha256


@dataclass(frozen=True)
class CellOutcome:
    """One implementation's outcome for one fixture cell.

    ``contract`` (the live ``ClusterFeatureContract``, when the build
    succeeded) is NEVER part of ``to_json()`` -- it exists purely so the
    CONSUMING TEST can assert on live-object fields the JSON semantic dict
    does not carry (e.g. ``consolidation.is_identity_after_mask``, which is
    deliberately excluded from ``semantic_dict()`` -- see ``ClusterFeature
    Contract``'s own ``field(compare=False, repr=False)``). The golden JSON
    itself only ever stores the JSON-safe fields.
    """

    raises: bool
    error_message: str | None
    semantic_dict: dict[str, Any] | None
    fingerprint: str | None
    contract: Any = None

    def to_json(self) -> dict[str, Any]:
        return {
            "raises": self.raises,
            "error_message": self.error_message,
            "semantic_dict": self.semantic_dict,
            "fingerprint": self.fingerprint,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> CellOutcome:
        return CellOutcome(
            raises=raw["raises"],
            error_message=raw["error_message"],
            semantic_dict=raw["semantic_dict"],
            fingerprint=raw["fingerprint"],
        )


def _outcome_from_call(build: Any) -> CellOutcome:
    try:
        contract = build()
    except Exception as exc:  # noqa: BLE001 -- deliberately catch-all: this
        # records "did this implementation raise" symmetrically for both
        # sides of the differential; it is re-raised nowhere -- the caught
        # exception's type+message is the payload itself.
        return CellOutcome(True, f"{type(exc).__name__}: {exc}", None, None)
    return CellOutcome(
        False, None, contract.semantic_dict(), contract.feature_contract_fingerprint, contract
    )


# ============================================================================
# TIER 1: pure build_cluster_feature_contract cells (no filesystem; never
# calls _assert_checkpoint_agreement, the #759 fingerprint-input site).
# ============================================================================

_TIER1_N_SAE_FEATURES = 8
_TIER1_SAMPLING_SELECTION = (
    json.dumps({"distribution": "geometric", "mean": 2.0}, sort_keys=True, separators=(",", ":")),
    json.dumps({"distribution": "geometric", "mean": 1.0}, sort_keys=True, separators=(",", ":")),
)
_TIER1_THRESHOLD_SELECTION = (
    json.dumps(
        {"threshold_type": "feature_percentile", "threshold": 0.9},
        sort_keys=True,
        separators=(",", ":"),
    ),
    json.dumps(
        {"threshold_type": "feature_percentile", "threshold": 0.5},
        sort_keys=True,
        separators=(",", ":"),
    ),
)


@dataclass(frozen=True)
class Tier1CellSpec:
    clusters: tuple[tuple[int, ...], ...]
    mask_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    selection_json: tuple[str, str] = _TIER1_SAMPLING_SELECTION
    concept_source_mode: str = "statistics_domain"
    concept_source_domain: tuple[int, ...] = (2, 3, 4, 6)
    stats_domain: tuple[int, ...] = (2, 3, 4, 6)
    n_sae_features: int = _TIER1_N_SAE_FEATURES


TIER1_CELLS: dict[str, Tier1CellSpec] = {
    "no_cluster_legacy": Tier1CellSpec(clusters=((1, 0), (2, 7))),
    "multi_member_mask_shifted_representative": Tier1CellSpec(
        clusters=((1, 2, 3),),
        mask_ids=(2, 3, 4, 5, 6),
    ),
    "singleton_after_mask": Tier1CellSpec(
        clusters=((2, 3),),
        mask_ids=(1, 2, 4, 5, 6),
        selection_json=_TIER1_THRESHOLD_SELECTION,
        stats_domain=(1, 2, 4, 6),
        concept_source_domain=(1, 2, 4, 6),
    ),
    "inert_cluster": Tier1CellSpec(
        clusters=((1, 2), (0, 7)),
        concept_source_mode="explicit_whitelist",
        concept_source_domain=(2, 4, 6),
        stats_domain=(2, 3, 4, 6),
    ),
    "threshold_selection_payload": Tier1CellSpec(
        clusters=((1, 2), (4, 5)),
        selection_json=_TIER1_THRESHOLD_SELECTION,
    ),
    "sampling_selection_payload": Tier1CellSpec(
        clusters=((1, 2), (4, 5)),
        selection_json=_TIER1_SAMPLING_SELECTION,
    ),
    # Codex fix batch finding 4 (MEDIUM, also independently found by
    # lens-3), NAMED BUG-FIX DEVIATION from the oracle (opposite direction
    # from every other cell in this matrix): mask filtering can remove
    # EVERY statistics-domain concept source, leaving source_to_
    # representative == () and an empty projecting_source_ids for every
    # representative. The oracle happily returns a "successful" contract in
    # that case; the stack now raises (E4 non-emptiness, mirroring
    # effective_feature_vocab's own E4 invariant at the concept-source-
    # projection layer -- see steering/eval/cluster_feature_contract.py).
    # Reproduces lens-3's exact config: statistics/concept domain
    # (2, 3, 4, 6) against a feature mask whitelisting only ids 500..505,
    # entirely disjoint from that domain. n_sae_features=512 is large
    # enough to hold both the domain and the disjoint mask range (TIER 1
    # never touches the filesystem, so unlike TIER 2 there is no SAEWidth
    # label to keep consistent).
    "empty_concept_projection_after_masking": Tier1CellSpec(
        clusters=((500, 501),),
        mask_ids=tuple(range(500, 506)),
        concept_source_domain=(2, 3, 4, 6),
        stats_domain=(2, 3, 4, 6),
        n_sae_features=512,
    ),
}


def _tier1_model_stat_signature(*, is_oracle: bool) -> tuple[int, ...]:
    return (1, 2, 123, 4, 5) if is_oracle else (0o100644, 1, 2, 123, 4, 5)


def _tier1_meta_stat_signature(*, is_oracle: bool) -> tuple[int, ...]:
    return (1, 3, 45, 4, 5) if is_oracle else (0o100644, 1, 3, 45, 4, 5)


def _run_tier1(module: ModuleType, cell_name: str, root: Path, *, is_oracle: bool) -> CellOutcome:
    spec = TIER1_CELLS[cell_name]

    def _build() -> Any:
        mask = FeatureMask(FeatureMaskMode.whitelist, spec.mask_ids)
        definition = FeatureClusterDefinition(clusters=spec.clusters)
        descriptor = FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=spec.n_sae_features
        )
        resolved_clusters = ResolvedFeatureClusters(
            source_path=root / "clusters.yaml",
            definition=definition,
            descriptor=descriptor,
        )
        stats = module.FeatureStatisticsIdentity(
            file_sha256=_A_SHA,
            percentile_table_checksum=_B_SHA,
            n_sae_features=spec.n_sae_features,
            feature_id_domain=spec.stats_domain,
        )
        directory = module.FeatureDirectoryIdentity(
            segment_size=128,
            layer=8,
            activation_type="residual",
            pooling="max_pool",
            width="16k",
            model="gemma-2-2b",
            dataset_phase="base",
            context_independent_features=False,
            ignore_bos_features=True,
        )
        sae = module.SaeIdentity(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_8/width_16k/canonical",
            hook_name="blocks.8.hook_resid_post",
            model="gemma-2-2b",
            layer=8,
            activation_type="residual",
            width="16k",
            actual_loaded_width=spec.n_sae_features,
        )
        checkpoint_dir = root / "checkpoint"
        # Codex fix batch finding 2 (HIGH): feed the PRODUCTION-correct,
        # post-#759 EFFECTIVE vocabulary -- not the obsolete mask-only
        # shape -- so a byte-equal TIER 1 cell proves agreement on a
        # genuinely-clustered, production-representative input, not an
        # input shape a real checkpoint could never legitimately carry once
        # real consolidation has occurred. build_cluster_feature_contract
        # is PURE (never calls _assert_checkpoint_agreement) and treats
        # whatever vocab its checkpoint pin carries as opaque, so this is a
        # strict generalization, not a special case: for the 2 of 6 cells
        # where consolidation is identity-after-mask (no real
        # consolidation), effective_feature_vocab delegates to the exact
        # same FeatureVocab.from_mask result per its own E3 invariant ("The
        # delegated path adds no behavior... to an unclustered model" --
        # steering/conditional/effective_feature_vocab.py), so using it
        # uniformly for all 6 cells changes nothing there.
        consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)
        vocab = effective_feature_vocab(consolidation, spec.n_sae_features)
        pin = module.CheckpointPin(
            checkpoint_dir=checkpoint_dir,
            resolved_tag="clustered-fixture",
            step=7,
            model_file=module.StableFileDigest(
                path=checkpoint_dir / "model_000007.pt",
                sha256=_C_SHA,
                byte_length=123,
                stat_signature=_tier1_model_stat_signature(is_oracle=is_oracle),
            ),
            metadata_file=module.StableFileDigest(
                path=checkpoint_dir / "meta_000007.json",
                sha256=_D_SHA,
                byte_length=45,
                stat_signature=_tier1_meta_stat_signature(is_oracle=is_oracle),
            ),
            n_sae_features=spec.n_sae_features,
            feature_vocab=vocab,
            load_bearing_metadata_json="{}",
        )
        selection = module.CheckpointSelectionContract(
            positive_selection_json=spec.selection_json[0],
            negative_selection_json=spec.selection_json[1],
            num_buckets=2,
            effective_positive_buckets=1,
            preselection_seed=42,
            selection_hash=_A_SHA,
            selection_manifest_sha256=_B_SHA,
            selection_shards_sha256=_C_SHA,
            packed_row_fingerprint=_D_SHA,
        )
        return module.build_cluster_feature_contract(
            feature_mask=mask,
            feature_clusters=resolved_clusters,
            statistics=stats,
            feature_directory=directory,
            sae=sae,
            checkpoint=pin,
            selection=selection,
            concept_source_mode=spec.concept_source_mode,
            concept_source_domain=spec.concept_source_domain,
        )

    return _outcome_from_call(_build)


def run_tier1_cell_oracle(oracle_module: ModuleType, cell_name: str, root: Path) -> CellOutcome:
    return _run_tier1(oracle_module, cell_name, root, is_oracle=True)


def run_tier1_cell_stack(cell_name: str, root: Path) -> CellOutcome:
    """Stack-only entry point -- the ONLY thing the consuming test imports."""
    return _run_tier1(contract_module, cell_name, root, is_oracle=False)


# ============================================================================
# TIER 2: full resolve_cluster_feature_contract filesystem preflight cells --
# the only entry point that calls _assert_checkpoint_agreement, the one
# fingerprint-input site B2a/B2b's named #759 deviation touches.
# ============================================================================

_TIER2_N_SAE_FEATURES = 16_384
_TIER2_MASK_IDS = tuple(range(6))

# Fixed, portable strings embedded into the checkpoint metadata JSON in place
# of the real (tmp-path-derived, cross-run-unstable) feature_dir/cluster
# path. parse_feature_dir only looks at TRAILING path components (see its own
# docstring: "Only trailing components are used; any leading prefix is
# ignored"), so a fixed, nonexistent root reparses to the identical
# FeatureDirectoryIdentity as the real tmp-path feature_dir built with the
# same segment_size/layer/activation_type/pooling/width/model/dataset_phase
# -- computed via build_feature_dir itself, not hand-typed, so the directory-
# label convention (e.g. activation_type=residual -> path segment "a_res",
# NOT "a_residual") can never drift out of sync by hand-transcription error.
_PINNED_FEATURE_DIR = str(
    build_feature_dir(
        Path("/pinned/golden/features"),
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
)
_PINNED_CLUSTER_PATH = "/pinned/golden/feature_clusters.json"


@dataclass(frozen=True)
class Tier2CellSpec:
    clusters: tuple[tuple[int, ...], ...]
    vocab_kind: str = "wide"  # "wide" (mask-only) | "narrow" (effective_feature_vocab)
    metadata_layout: str = "top_level"  # "top_level" (SFT-style) | "nested_pretrain"
    mask_ids: tuple[int, ...] = _TIER2_MASK_IDS
    n_sae_features: int = _TIER2_N_SAE_FEATURES


TIER2_CELLS: dict[str, Tier2CellSpec] = {
    "trivial_cluster_num_buckets_dedup": Tier2CellSpec(
        clusters=((0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15)),
        vocab_kind="wide",
    ),
    # Codex fix batch finding 3 (MEDIUM): the full fixture matrix only ever
    # exercised top-level/SFT-style encoder metadata layout. B2 deliberately
    # normalizes NESTED pretrain-layout metadata (user_config.encoder/
    # user_config.cross_attention -- cluster_checkpoint_pin.py's
    # _checkpoint_load_bearing_metadata -> nanochat.checkpoint_manager.
    # _extract_encdec_fields), while the oracle only ever reads the
    # TOP-LEVEL encoder_config/xattn_config keys and fingerprints the
    # (present but nested) real values as None. Trivial (identity-after-
    # mask) cluster, same as trivial_cluster_num_buckets_dedup, so BOTH
    # implementations still succeed -- only the semantic fingerprint
    # diverges, and (per Tier2CellSpec.metadata_layout's own "nested_
    # pretrain" branch) num_buckets is deliberately kept top-level so this
    # cell exercises ONLY the encoder/xattn divergence, isolated from the
    # already-named num_buckets-dedup deviation above.
    "pretrain_layout_encoder_xattn_normalization": Tier2CellSpec(
        clusters=((0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15)),
        vocab_kind="wide",
        metadata_layout="nested_pretrain",
    ),
    "genuine_cluster_wide_vocab_rejected": Tier2CellSpec(
        clusters=((0, 1), (2, 3)),
        vocab_kind="wide",
    ),
    "genuine_cluster_effective_vocab_accepted": Tier2CellSpec(
        clusters=((0, 1), (2, 3)),
        vocab_kind="narrow",
    ),
}

# Codex fix batch finding 2 (HIGH), second half: cells whose value the ORACLE
# structurally cannot produce (it raises on this input -- see "genuine_
# cluster_wide_vocab_rejected"'s mirror-image cell) get a STACK-generated
# golden instead of an oracle-generated one, honestly labeled as such. This
# is what makes "resolving the identical on-disk artifact twice yields the
# identical fingerprint" (within-run determinism, checked unconditionally
# regardless of this golden) into something CROSS-RUN and independently
# pinned: a future accidental change to the fingerprint computation for this
# exact scenario is caught even though no oracle value exists to compare
# against.
STACK_PINNED_CELLS: dict[str, str] = {
    "genuine_cluster_effective_vocab_accepted": (
        "post-#759 production-correct value; the pre-#759 oracle cannot produce it"
    ),
}


def _normalize_semantic_for_golden(semantic: dict[str, Any]) -> dict[str, Any]:
    """Replace raw-file-content hashes that are NOT reproducible across
    separate process invocations with a fixed sentinel -- see the module
    docstring's "Golden reproducibility for TIER 2" section. Everything else
    (JSON-encoded metadata, content-derived checksums) is left untouched."""
    normalized = json.loads(json.dumps(semantic))
    checkpoint = normalized.get("checkpoint")
    if isinstance(checkpoint, dict) and "model_sha256" in checkpoint:
        checkpoint["model_sha256"] = _NONDETERMINISTIC_SENTINEL
    statistics = normalized.get("statistics")
    if isinstance(statistics, dict) and "file_sha256" in statistics:
        statistics["file_sha256"] = _NONDETERMINISTIC_SENTINEL
    return normalized


def normalized_sha256(semantic: dict[str, Any]) -> str:
    """Independent SHA-256 of a (already golden-normalized) semantic dict,
    using the same canonicalization convention (sorted keys, compact
    separators) production's own ``_canonical_json``/``_sha256_json`` use --
    replicated here via plain ``json``/``hashlib``, not by calling those
    private helpers, so this is a genuinely separate computation."""
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _selection_max_features_per_segment(selection: SelectionShard) -> int:
    """Return the producer certificate's positive-plus-negative CSR width."""
    positive_widths = (
        selection.sel_offsets[1:].to(t.int64)
        - selection.sel_offsets[:-1].to(t.int64)
    )
    negative_widths = (
        selection.neg_offsets[1:].to(t.int64)
        - selection.neg_offsets[:-1].to(t.int64)
    )
    if positive_widths.shape != negative_widths.shape:
        raise RuntimeError(
            "fixture selection positive/negative CSR domains disagree: "
            f"{positive_widths.shape} != {negative_widths.shape}"
        )
    widths = positive_widths + negative_widths
    return int(widths.max().item()) if widths.numel() else 0


def _write_tier2_fixture(root: Path, spec: Tier2CellSpec) -> tuple[Path, Path, Path, Path, Path]:
    """Build one certified on-disk clustered-preflight fixture tree.

    The fixture follows the producer's completed-publication contract: its
    selection manifest contains both the compact inventory and the schema-v1
    ``clustered_selection_validation`` certificate, built from the exact
    serialized shard bytes and packed-row fingerprint. Returns
    ``(feature_dir, statistics_path, checkpoint_dir, mask_path, cluster_path)``.
    """

    mask = FeatureMask(FeatureMaskMode.whitelist, spec.mask_ids)
    mask_path = root / "accepted_feature_ids.txt"
    mask_path.write_text("".join(f"{feature_id}\n" for feature_id in spec.mask_ids), encoding="utf-8")

    cluster_path = root / "feature_clusters.json"
    definition = FeatureClusterDefinition(clusters=spec.clusters)
    cluster_path.write_text(json.dumps(definition.canonical_dict(), sort_keys=True), encoding="utf-8")
    resolved_clusters = ResolvedFeatureClusters(
        source_path=cluster_path.resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=spec.n_sae_features
        ),
    )

    feature_dir = build_feature_dir(
        root / "features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
    feature_dir.mkdir(parents=True)
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=spec.n_sae_features,
        max_mags={feature_id: float(feature_id + 1) for feature_id in spec.mask_ids},
        absent=range(len(spec.mask_ids), spec.n_sae_features),
    )
    stats = load_statistics(statistics_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, spec.n_sae_features)
    )
    save_percentile_checksum(statistics_path, percentile_checksum)

    packed_dir = feature_dir / "packed_fixture"
    packed_dir.mkdir()
    packed_shard = PackedRowShard(
        tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
        row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
        seg_token_start=t.tensor([0], dtype=t.uint16),
        seg_token_end=t.tensor([4], dtype=t.uint16),
        seg_doc_id=t.tensor([0], dtype=t.uint8),
        seg_feat_offsets=t.tensor([0, 4], dtype=t.int32),
        feature_ids=t.tensor([0, 1, 2, 4], dtype=t.uint16),
        feature_pcts=t.tensor([220, 180, 200, 90], dtype=t.uint8),
    )
    packed_path = save_packed_row_shard(
        packed_shard,
        packed_dir / "shard_00000",
    )
    packed_fixture = load_packed_row_shard(packed_path)
    write_manifest(
        packed_dir,
        "packed-fixture",
        OverlayKind.PACKED_ROW,
        metadata={"pct_checksum": percentile_checksum},
        base_shard_paths=[packed_path],
    )

    positive_selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    negative_selection = {"distribution": "geometric", "mean": 1.0, "std": 1.0}
    selection_hash = compute_selection_hash(
        positive_selection=positive_selection,
        negative_selection=negative_selection,
        delta_config=None,
        num_buckets=2,
        seed=42,
        ddp_rank=0,
        feature_mask=mask.canonical_dict(),
        feature_clusters=resolved_clusters.descriptor_dict(),
    )
    selection_dir = packed_dir / "sel" / selection_hash
    selection_dir.mkdir(parents=True)
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)
    selection_shard = SelectionShard(
        sel_offsets=t.tensor([0, 2], dtype=t.int32),
        sel_fids=t.tensor([0, 2], dtype=t.uint16),
        sel_buckets=t.tensor([0, 0], dtype=t.uint8),
        sel_pcts=t.tensor([220, 200], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1], dtype=t.int32),
        neg_fids=t.tensor([4], dtype=t.uint16),
    )
    selection_path = save_selection_shard(
        selection_shard,
        selection_dir / "shard_00000",
    )
    selection_fixture = load_selection_shard(selection_path)
    cluster_metadata = consolidation.metadata_dict()
    cluster_metadata["status"] = "complete"
    selection_integrity = build_clustered_selection_shard_integrity([selection_path])
    packed_proof = resolve_packed_row_fingerprint_strict(
        packed_dir,
        allow_cache_write=False,
    )
    packed_fingerprint = packed_row_fingerprint(packed_dir)
    if packed_proof.aggregate_sha256 != packed_fingerprint:
        raise RuntimeError(
            "fixture packed-row fingerprint disagrees with the strict proof: "
            f"{packed_fingerprint!r} != {packed_proof.aggregate_sha256!r}"
        )
    if packed_fixture.total_segments != selection_fixture.total_segments:
        raise RuntimeError(
            "fixture packed/selection segment counts disagree: "
            f"{packed_fixture.total_segments} != {selection_fixture.total_segments}"
        )
    clustered_selection_validation = build_clustered_selection_validation(
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=packed_proof.aggregate_sha256,
        packed_content_by_name={
            entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
            for entry in packed_proof.files
        },
        feature_clusters_descriptor=resolved_clusters.descriptor_dict(),
        feature_delta=None,
        effective_buckets=1,
        segment_counts_by_stem={
            "shard_00000": selection_fixture.total_segments,
        },
        max_features_by_stem={
            "shard_00000": _selection_max_features_per_segment(selection_fixture),
        },
    )
    write_manifest(
        selection_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata={
            "feature_mask": mask.canonical_dict(),
            "feature_clusters": cluster_metadata,
            "packed_rows": {"packed_row_fingerprint": packed_fingerprint},
            "selection_shards": selection_integrity,
            CLUSTERED_SELECTION_VALIDATION_KEY: clustered_selection_validation,
        },
        base_shard_paths=[packed_path],
        cache_dir=packed_dir,
    )

    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir()
    if spec.vocab_kind == "wide":
        vocab = FeatureVocab.from_mask(mask, spec.n_sae_features)
    elif spec.vocab_kind == "narrow":
        vocab = effective_feature_vocab(consolidation, spec.n_sae_features)
    else:
        raise ValueError(f"unknown vocab_kind {spec.vocab_kind!r}")
    assert vocab is not None
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((vocab.size, 2)),
            VOCAB_IDS_KEY: t.tensor(vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(spec.n_sae_features, dtype=t.int64),
        },
        checkpoint_dir / "model_000007.pt",
    )

    user_config: dict[str, Any] = {
        "feature_selection": positive_selection,
        "negative_selection": negative_selection,
        "seed": 42,
        "feature_delta": None,
        "feature_prediction": None,
        "feature_dir": _PINNED_FEATURE_DIR,
        "feature_clusters": {"path": _PINNED_CLUSTER_PATH},
        "sae": {
            "release": SAERelease.gemmascope.value,
            "model": "gemma-2-2b",
            "activation_type": ActivationType.residual.value,
            "layer": 8,
            "width": SAEWidth.W_16K.value,
            "l0": None,
        },
    }
    checkpoint_metadata: dict[str, Any] = {
        "model_config": {"n_layer": 2, "n_embd": 8},
        "feature_mask_canonical": mask.canonical_dict(),
        "feature_vocab": vocab.metadata_dict(),
        "feature_clusters_descriptor": resolved_clusters.descriptor_dict(),
        "feature_clusters_canonical": resolved_clusters.canonical_dict(),
    }
    if spec.metadata_layout == "top_level":
        checkpoint_metadata["encoder_config"] = {"n_embd": 8, "depth": 1}
        checkpoint_metadata["xattn_config"] = {"frequency": 1}
        checkpoint_metadata["num_buckets"] = 2
        # Named deviation (B2b Codex pass-2 residual finding 1, MEDIUM --
        # see cluster_checkpoint_pin's own docstring): the oracle's
        # _CHECKPOINT_USER_CONFIG_KEYS still lists "num_buckets", so a RAW
        # user_config.num_buckets duplicates the normalized top-level value
        # a second time on the oracle side only -- the stack's own set no
        # longer includes it. Set on BOTH sides identically (matching a
        # real checkpoint's writer, which logs the same value both ways);
        # only the two implementations' READING of it differs.
        user_config["num_buckets"] = 2
    elif spec.metadata_layout == "nested_pretrain":
        # Codex fix batch finding 3: encoder/cross_attention nested-only
        # under user_config, matching a real pretrain checkpoint's layout.
        # num_buckets stays TOP-LEVEL (agrees on both sides) so ONLY
        # encoder_config/xattn_config diverge -- isolated from the
        # separately-named num_buckets-dedup deviation.
        checkpoint_metadata["num_buckets"] = 2
        user_config["encoder"] = {"n_embd": 8, "depth": 1}
        user_config["cross_attention"] = {"frequency": 1}
    else:
        raise ValueError(f"unknown metadata_layout {spec.metadata_layout!r}")
    checkpoint_metadata["user_config"] = user_config
    (checkpoint_dir / "meta_000007.json").write_text(
        json.dumps(checkpoint_metadata, sort_keys=True), encoding="utf-8"
    )
    return feature_dir, statistics_path, checkpoint_dir, mask_path, cluster_path


def _tier2_request(
    module: ModuleType,
    paths: tuple[Path, Path, Path, Path, Path],
) -> Any:
    feature_dir, statistics_path, checkpoint_dir, mask_path, cluster_path = paths
    return module.ClusterFeatureContractRequest(
        feature_dir=feature_dir,
        feature_statistics_path=statistics_path,
        feature_mask={"mode": "whitelist", "path": str(mask_path)},
        feature_clusters={"path": str(cluster_path)},
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=7,
        checkpoint_tag="fixture-step-7",
    )


def _fake_sae_loader(n_sae_features: int) -> Any:
    resolved_sae = resolve_sae(
        model="gemma-2-2b",
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=8,
        width=SAEWidth.W_16K,
        l0=None,
    )
    fake_sae = type(
        "FakeSae",
        (),
        {
            "W_dec": t.zeros((n_sae_features, 2)),
            "cfg": type(
                "FakeCfg",
                (),
                {"d_in": 2, "metadata": type("FakeMeta", (), {"hook_name": resolved_sae.hook_name})()},
            )(),
        },
    )()
    return lambda _resolved: fake_sae


def _build_tier2_request(module: ModuleType, cell_name: str, root: Path) -> tuple[Any, Any]:
    """Write the on-disk fixture tree ONCE and return ``(request,
    sae_loader)``. Split out from ``_run_tier2`` so a caller that needs to
    resolve the IDENTICAL on-disk artifact more than once (e.g. a within-run
    determinism replay) can do so without rebuilding the fixture a second
    time at the same path -- ``_write_tier2_fixture`` is not idempotent
    (``feature_dir.mkdir(parents=True)`` has no ``exist_ok=True``, by design:
    a second silent write to the same producer path would defeat the whole
    point of a frozen-artifact replay)."""
    spec = TIER2_CELLS[cell_name]
    paths = _write_tier2_fixture(root, spec)
    request = _tier2_request(module, paths)
    sae_loader = _fake_sae_loader(spec.n_sae_features)
    return request, sae_loader


def build_tier2_request_stack(cell_name: str, root: Path) -> tuple[Any, Any]:
    """Stack-only ``(request, sae_loader)`` pair for a fresh on-disk fixture.

    For consumers (e.g. a within-run determinism replay test) that need to
    call ``resolve_cluster_feature_contract`` more than once against the
    SAME already-written files -- ``run_tier2_cell_stack`` below builds and
    resolves in one call and is not suitable for that.
    """
    return _build_tier2_request(contract_module, cell_name, root)


def _resolve_tier2_outcome(module: ModuleType, request: Any, sae_loader: Any) -> CellOutcome:
    try:
        contract = module.resolve_cluster_feature_contract(request, sae_loader=sae_loader)
    except Exception as exc:  # noqa: BLE001 -- see _outcome_from_call
        return CellOutcome(True, f"{type(exc).__name__}: {exc}", None, None)
    normalized = _normalize_semantic_for_golden(contract.semantic_dict())
    return CellOutcome(False, None, normalized, normalized_sha256(normalized), contract)


def _run_tier2(module: ModuleType, cell_name: str, root: Path) -> CellOutcome:
    request, sae_loader = _build_tier2_request(module, cell_name, root)
    return _resolve_tier2_outcome(module, request, sae_loader)


def run_tier2_cell_oracle(oracle_module: ModuleType, cell_name: str, root: Path) -> CellOutcome:
    return _run_tier2(oracle_module, cell_name, root)


def run_tier2_cell_stack(cell_name: str, root: Path) -> CellOutcome:
    """Stack-only entry point -- the ONLY thing the consuming test imports
    for cells that only need to be resolved once."""
    return _run_tier2(contract_module, cell_name, root)


def resolve_tier2_outcome_stack(request: Any, sae_loader: Any) -> CellOutcome:
    """Resolve an already-built stack request -- the replay half of a
    within-run determinism check (pair with ``build_tier2_request_stack``)."""
    return _resolve_tier2_outcome(contract_module, request, sae_loader)


# ============================================================================
# Golden capture.
# ============================================================================


def capture_golden(golden_path: Path = GOLDEN_PATH) -> Path:
    """Build every registered cell, run the ORACLE side of each (plus the
    STACK side for ``STACK_PINNED_CELLS``), and write the committed golden
    JSON. Requires the frozen oracle worktree."""
    import tempfile

    oracle_module, oracle_source_digest = load_oracle_module()
    cells: dict[str, dict[str, Any]] = {}
    for cell_name in sorted(TIER1_CELLS):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_tier1_cell_oracle(oracle_module, cell_name, Path(tmp))
        cells[cell_name] = {"tier": "tier1", "oracle_outcome": outcome.to_json()}
    for cell_name in sorted(TIER2_CELLS):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_tier2_cell_oracle(oracle_module, cell_name, Path(tmp))
        cells[cell_name] = {"tier": "tier2", "oracle_outcome": outcome.to_json()}

    stack_pinned_cells: dict[str, dict[str, Any]] = {}
    for cell_name, label in sorted(STACK_PINNED_CELLS.items()):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_tier2_cell_stack(cell_name, Path(tmp))
        if outcome.raises or outcome.semantic_dict is None or outcome.fingerprint is None:
            raise RuntimeError(
                f"stack-pinned cell {cell_name!r} unexpectedly raised while "
                f"regenerating the golden: {outcome.error_message}"
            )
        stack_pinned_cells[cell_name] = {
            "label": label,
            "normalized_semantic_dict": outcome.semantic_dict,
            "normalized_semantic_sha256": outcome.fingerprint,
        }

    golden = {
        "schema_version": 1,
        "oracle_worktree": ".claude/worktrees/pr739-oracle",
        "oracle_commit": _ORACLE_COMMIT,
        # Codex pass-2 residual finding 1 (HIGH) + pass-3 residual finding
        # (HIGH): record the SAME digest load_oracle_module() computed FROM
        # THE EXACT BUFFER IT EXECUTED, returned above as oracle_source_
        # digest -- never a second, independent re-read of _ORACLE_PATH.
        # Pass-3 proved why that distinction matters: with a separate
        # reopen here, a cache-injected/poisoned oracle module could
        # execute while this line still innocently re-hashed the genuine,
        # unmodified file, so the golden's recorded provenance described
        # different bytes than what actually ran. Single source of truth:
        # whatever load_oracle_module() hashed is exactly what it exec'd,
        # and that is exactly what gets recorded here.
        "oracle_source_sha256": oracle_source_digest,
        "cells": cells,
        "stack_pinned_cells": stack_pinned_cells,
    }
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return golden_path


if __name__ == "__main__":
    written = capture_golden()
    print(f"wrote golden JSON: {written} ({written.stat().st_size} bytes)")
