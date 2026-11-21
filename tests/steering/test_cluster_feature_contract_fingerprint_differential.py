"""BINDING B3 gate (PR #739 Track B Part 3 plan, constraint 2): oracle-vs-stack
``ClusterFeatureContract`` fingerprint differential -- golden-backed.

Codex fix batch finding 1 (HIGH), the A5 golden pattern applied here. This
module used to load the frozen oracle's standalone ``cluster_feature_
contract.py`` from an untracked, external sibling worktree
(``../pr739-oracle``) at TEST TIME -- non-hermetic (a normal clone without
that worktree failed every test here), CI-broken (no such worktree exists in
CI), and unpinned (mutating that worktree silently changed what "the oracle"
meant, with no digest check). It now reads a COMMITTED golden JSON
(``tests/steering/goldens/cluster_contract_fingerprint_golden.json``,
provenance-pinned to the frozen oracle's exact commit) captured by
``tests/steering/goldens/gen_cluster_contract_fingerprint_golden.py`` -- the
only module that still touches the oracle worktree, and only when someone
deliberately reruns it to regenerate the golden. This module imports NOTHING
from that generator except its stack-only ``run_tier{1,2}_cell_stack``
functions and cell registries -- it has zero import-time or runtime
dependency on the oracle worktree existing.

# CONTROLLER: regenerate/commit the golden via
#     PYTHONPATH=. uv run python \
#         tests/steering/goldens/gen_cluster_contract_fingerprint_golden.py
# (requires the frozen oracle worktree at .claude/worktrees/pr739-oracle,
# pinned at commit f357458b3871534a00372673fc4c40f2e62f2ce7). Every test in
# this module below test_golden_committed fails until that JSON is committed
# -- expected, not a bug (mirrors the A5 golden-handoff precedent).

Two tiers, same shape as before the golden conversion:

* TIER 1 (module-level ``test_*_is_byte_equal`` functions) drives the pure,
  filesystem-free ``build_cluster_feature_contract`` — required fixture
  cells: multi-member clusters with mask-shifted representatives, singleton-
  after-mask, inert-cluster, no-cluster/legacy (``clusters=()``-equivalent),
  and both threshold- and sampling-shaped selection payloads. ``build_
  cluster_feature_contract`` never calls ``_assert_checkpoint_agreement``
  (the #759 fingerprint-input site), so every TIER-1 cell is byte-equal by
  construction — this tier proves the REST of the assembly algorithm
  (representative projection, consolidated-away bookkeeping, concept-source
  domain handling, canonical-JSON serialization) is unchanged from the
  oracle.
* TIER 2 (``test_*_diverges_named_759``/``test_trivial_cluster_regime_*``/
  ``test_pretrain_layout_*``) drives the filesystem-backed ``resolve_
  cluster_feature_contract`` — the ONLY entry point that calls ``_assert_
  checkpoint_agreement`` — across FOUR fixture cells (prose corrected here;
  an earlier revision of this docstring under-enumerated it as three — see
  ``test_pretrain_layout_diverges_on_nested_encoder_xattn_normalization``'s
  own docstring for the full ledger correction):
  1. A trivial (no real consolidation) cluster with top-level/SFT-style
     checkpoint metadata, where the oracle's mask-only vocab and
     ``effective_feature_vocab`` agree (byte-equal modulo the named
     num_buckets-dedup deviation).
  2. The SAME trivial cluster again, but with NESTED pretrain-layout
     checkpoint metadata instead (``user_config.encoder``/``user_config.
     cross_attention``) — diverging only on B2's separately-named
     encoder/xattn normalization deviation: the oracle reads solely the
     TOP-LEVEL ``encoder_config``/``xattn_config`` keys and fingerprints
     the (present but nested) real values as ``None``.
  3.-4. A genuine (real consolidation) cluster carrying each of the two
     vocab shapes in turn (both DIVERGE, in opposite directions, both
     justified by #759).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import tests.steering.goldens.gen_cluster_contract_fingerprint_golden as gen_golden_module
from tests.steering.goldens.gen_cluster_contract_fingerprint_golden import (
    _ORACLE_MODULE_NAME,
    EXPECTED_ORACLE_SOURCE_SHA256,
    build_tier2_request_stack,
    load_oracle_module,
    resolve_tier2_outcome_stack,
    run_tier1_cell_stack,
    run_tier2_cell_stack,
)

GOLDEN_PATH = Path(__file__).parent / "goldens" / "cluster_contract_fingerprint_golden.json"

# Frozen, verbatim -- the ONE commit the oracle worktree is pinned to for this
# entire differential (see the generator's own `_ORACLE_COMMIT`). Duplicated
# here as a literal, not imported, matching this module's existing precedent
# of hardcoding the oracle commit directly (see the docstring above and the
# pre-existing controller-regeneration message below) rather than reaching
# into the generator's private `_ORACLE_COMMIT`.
_EXPECTED_ORACLE_COMMIT = "f357458b3871534a00372673fc4c40f2e62f2ce7"


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_committed(golden_path: Path = GOLDEN_PATH) -> None:
    """Golden JSON must exist AND its provenance must independently check out.

    Codex pass-2 residual finding 1 (HIGH): file EXISTENCE alone proved
    nothing about PROVENANCE. Before this fix, a golden regenerated from a
    mutated oracle source (adversarially demonstrated: a changed oracle was
    silently loaded, executed, and its output captured while still labeled
    with the frozen commit string), or one whose ``oracle_commit`` was
    hand-edited/zeroed outright, both passed every test in this module
    unconditionally. Each pinned provenance field is now independently
    re-checked here against a value this test does not merely trust the
    golden to report honestly:

    * ``schema_version`` -- the golden schema this consumer was written
      against.
    * ``oracle_commit`` -- the frozen oracle worktree's pinned commit,
      hardcoded above as ``_EXPECTED_ORACLE_COMMIT`` (independent of
      whatever the golden itself claims).
    * ``oracle_source_sha256`` -- SHA-256 of the oracle SOURCE FILE'S raw
      bytes at that commit, imported from the generator's own
      ``EXPECTED_ORACLE_SOURCE_SHA256`` (a single shared constant, not
      re-derived here, so this check and the generator's own pre-execution
      digest check can never silently drift apart from each other) --
      independently re-hashed and asserted equal by the generator's
      ``load_oracle_module`` BEFORE the oracle ever executes, so a golden
      that reflects a drifted oracle can no longer be regenerated at all;
      this second check additionally catches a golden that was hand-edited
      after the fact (still possible -- it is a plain JSON file).

    ``golden_path`` defaults to the real committed golden; a test can pass a
    different path directly (bypassing pytest's fixture injection, since a
    parameter with a default is never treated as a required fixture) to
    exercise this function's own checks against a deliberately corrupted
    golden -- see ``test_golden_committed_rejects_corrupted_provenance``
    below.
    """
    assert golden_path.is_file(), (
        f"golden JSON not committed at {golden_path} -- the controller must run "
        "`PYTHONPATH=. uv run python "
        "tests/steering/goldens/gen_cluster_contract_fingerprint_golden.py` "
        "(requires the frozen oracle worktree at .claude/worktrees/pr739-oracle, "
        f"pinned at commit {_EXPECTED_ORACLE_COMMIT}) and commit the "
        "resulting JSON. See task-b3-report.md's '## Codex fix batch' section for "
        "the full handoff."
    )
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden.get("schema_version") == 1, (
        f"golden JSON at {golden_path} has schema_version={golden.get('schema_version')!r}, "
        "expected 1 -- regenerate against the current generator/schema"
    )
    assert golden.get("oracle_commit") == _EXPECTED_ORACLE_COMMIT, (
        f"golden JSON at {golden_path} has oracle_commit={golden.get('oracle_commit')!r}, "
        f"expected the frozen oracle worktree's pinned commit {_EXPECTED_ORACLE_COMMIT!r} -- "
        "either the golden was regenerated against a re-pinned oracle worktree without "
        "updating this test's _EXPECTED_ORACLE_COMMIT, or the field was hand-edited"
    )
    assert golden.get("oracle_source_sha256") == EXPECTED_ORACLE_SOURCE_SHA256, (
        f"golden JSON at {golden_path} has oracle_source_sha256="
        f"{golden.get('oracle_source_sha256')!r}, expected {EXPECTED_ORACLE_SOURCE_SHA256!r} -- "
        "either the golden was regenerated against a drifted oracle (load_oracle_module's "
        "own pre-execution digest check should have refused this) or the golden's "
        "provenance field was hand-edited after the fact; regenerate honestly"
    )


@pytest.mark.parametrize(
    ("field", "corrupted_value", "match"),
    [
        (
            "schema_version",
            999,
            "schema_version",
        ),
        (
            "oracle_commit",
            "0" * 40,
            "oracle_commit",
        ),
        (
            "oracle_source_sha256",
            "0" * 64,
            "oracle_source_sha256",
        ),
    ],
)
def test_golden_committed_rejects_corrupted_provenance(
    tmp_path: Path,
    field: str,
    corrupted_value: Any,
    match: str,
) -> None:
    """Codex pass-2 residual finding 1 (HIGH) regression, one case per pinned
    provenance field. Adversarially demonstrated pre-fix: a golden with a
    zeroed ``oracle_commit`` still passed every consuming differential test,
    and a golden regenerated from a semantically mutated oracle was still
    captured and labeled with the frozen commit string. Each field
    ``test_golden_committed`` now independently checks is corrupted here, one
    at a time, on an otherwise-valid golden (the real committed one, copied
    and mutated) -- and the check is proven to actually fire, not merely be
    present as dead code."""
    golden = _load_golden()
    golden[field] = corrupted_value
    corrupted_path = tmp_path / "corrupted_golden.json"
    corrupted_path.write_text(json.dumps(golden), encoding="utf-8")
    with pytest.raises(AssertionError, match=match):
        test_golden_committed(corrupted_path)


def test_load_oracle_module_rejects_cache_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex pass-3 residual finding (HIGH) regression, made HERMETIC in the
    Codex pass-4 follow-up below -- this test now carries ZERO dependency on
    the real, untracked oracle worktree.

    Pre-fix (pass-3), ``load_oracle_module`` returned any PRE-EXISTING
    ``sys.modules[_ORACLE_MODULE_NAME]`` entry on a bare membership check,
    before the digest check ever ran. Codex adversarially demonstrated the
    consequence end-to-end: inject a modified module into that cache slot,
    call the (then-current) ``capture_golden`` -- the injected code
    executes, yet the golden it writes still records the GENUINE file's
    correct commit and ``oracle_source_sha256`` (computed by a *separate*,
    honest re-read of ``_ORACLE_PATH``), and ``test_golden_committed``
    passes. The recorded provenance described different bytes than what
    actually ran.

    Codex pass-4 residual finding: the FIRST version of this regression
    (committed alongside the pass-3 fix) reproduced that attack against the
    REAL oracle -- calling ``load_oracle_module()`` unmodified, which reads
    the hardcoded, machine-specific absolute ``_ORACLE_PATH`` pointing at
    the untracked ``.claude/worktrees/pr739-oracle`` sibling worktree. As an
    UNMARKED DEFAULT test, that made this one function fail with a bare
    ``FileNotFoundError`` in any normal clone or in CI (Codex simulated
    exactly that state and reproduced the failure), reintroducing -- for
    this ONE test, inside an otherwise fully golden-backed, hermetic module
    -- precisely the non-hermetic, CI-broken live-oracle dependency the A5
    golden-JSON conversion (see this module's own docstring and the
    generator's) exists to eliminate.

    Fix: this test no longer touches the real oracle worktree. It writes
    its OWN minimal, self-contained "oracle-shaped" module straight into
    ``tmp_path`` (a two-line file: a ``MARKER`` string constant and a
    trivial function returning it), computes THAT file's real SHA-256, and
    monkeypatches the generator module's ``_ORACLE_PATH``/``EXPECTED_
    ORACLE_SOURCE_SHA256`` globals to point at it. ``load_oracle_module``
    itself is completely indifferent to WHICH file it reads/hashes/execs --
    it has no dependency anywhere on the real oracle's actual semantic
    content (only on the bytes matching whatever digest it is told to
    expect) -- so redirecting both globals exercises the IDENTICAL
    read-hash-exec-and-discard-the-cache code path the pass-3 fix built,
    isolated from the real oracle's existence entirely. The cache-injection
    attack itself is reproduced exactly as before: a poisoned sentinel
    module is pre-populated into ``sys.modules[_ORACLE_MODULE_NAME]``, its
    function records (and would raise) if ever called, and this test
    asserts (a) the returned module's ``MARKER`` is the TEMP oracle's,
    never the sentinel's; (b) the returned digest is the temp file's own
    SHA-256; (c) the sentinel's function was never invoked -- by IDENTITY
    (the sentinel object is never returned, and the cache slot itself is
    overwritten with whatever WAS returned) and by RECORDED BEHAVIOR (a
    call counter proves the poisoned function's body never ran).

    ``_ORACLE_PATH``/``EXPECTED_ORACLE_SOURCE_SHA256`` are restored
    automatically at teardown by ``monkeypatch``. ``sys.modules[
    _ORACLE_MODULE_NAME]`` is restored manually in the ``finally`` block
    below, unchanged from the pass-3 version of this test: it must not
    assume the slot was empty beforehand (a later test importing the
    oracle in the same process would otherwise leak this test's mutation
    forward)."""
    real_marker = "real-temp-oracle"
    temp_oracle_source = (
        f'MARKER = "{real_marker}"\n\n\ndef marker_fn() -> str:\n    return MARKER\n'
    )
    temp_oracle_path = tmp_path / "temp_oracle_shaped_module.py"
    temp_oracle_path.write_text(temp_oracle_source, encoding="utf-8")
    temp_oracle_digest = hashlib.sha256(temp_oracle_path.read_bytes()).hexdigest()

    monkeypatch.setattr(gen_golden_module, "_ORACLE_PATH", temp_oracle_path)
    monkeypatch.setattr(gen_golden_module, "EXPECTED_ORACLE_SOURCE_SHA256", temp_oracle_digest)

    sentinel = ModuleType(_ORACLE_MODULE_NAME)
    poison_calls: list[str] = []

    def _poisoned_marker_fn() -> str:
        poison_calls.append("called")
        raise RuntimeError("cache-injected oracle executed")

    sentinel.MARKER = "poisoned-sentinel"  # ty: ignore[unresolved-attribute]
    sentinel.marker_fn = _poisoned_marker_fn  # ty: ignore[unresolved-attribute]
    previous = sys.modules.get(_ORACLE_MODULE_NAME)
    sys.modules[_ORACLE_MODULE_NAME] = sentinel
    try:
        module, digest = load_oracle_module()

        # Identity: the injected sentinel is never handed back, and the
        # cache slot itself now holds whatever WAS returned -- not the
        # sentinel that was sitting there a moment ago.
        assert module is not sentinel
        assert sys.modules[_ORACLE_MODULE_NAME] is module

        # (a) the real temp oracle's MARKER, never the sentinel's.
        assert module.MARKER == real_marker

        # (b) the returned digest is the temp file's own hash -- the SAME
        # buffer load_oracle_module hashed and executed, not a second,
        # independently-reopened read.
        assert digest == temp_oracle_digest

        # (c) the poison's function never executed: calling the RETURNED
        # module's marker_fn resolves to the real temp oracle's version and
        # succeeds, and the poisoned call counter proves the sentinel's own
        # function body never ran.
        assert module.marker_fn() == real_marker
        assert poison_calls == []
    finally:
        if previous is None:
            sys.modules.pop(_ORACLE_MODULE_NAME, None)
        else:
            sys.modules[_ORACLE_MODULE_NAME] = previous


def _golden_cell(cell_name: str) -> dict[str, Any]:
    golden = _load_golden()
    assert cell_name in golden["cells"], (
        f"golden JSON at {GOLDEN_PATH} is missing cell {cell_name!r} -- the generator "
        "and this consumer have drifted out of sync, or the golden needs "
        "regenerating (see test_golden_committed's message for the command)"
    )
    return golden["cells"][cell_name]


def _normalize_oracle_selection_manifest_hash(
    oracle_semantic: dict[str, Any], stack_semantic: dict[str, Any]
) -> dict[str, Any]:
    """Normalize the frozen oracle's historical certificate-inclusive hash.

    The current stack deliberately excludes the derivable schema-v1
    ``clustered_selection_validation`` block from the selection-manifest
    semantic hash. The frozen oracle predates that rule, so its committed
    Tier-2 semantic payload has a different ``selection_manifest_sha256``
    even after every named behavioral difference is removed. The fixture now
    contains the producer certificate required by the current guard; copy only
    this one derived identity from the live stack and compare every other
    semantic field exactly.
    """
    normalized = json.loads(json.dumps(oracle_semantic))
    oracle_selection = normalized["selection"]
    stack_selection = stack_semantic["selection"]
    assert oracle_selection["selection_manifest_sha256"] != stack_selection[
        "selection_manifest_sha256"
    ], "expected the frozen oracle and current stack to differ on certificate-era hash"
    oracle_selection["selection_manifest_sha256"] = stack_selection[
        "selection_manifest_sha256"
    ]
    return normalized


# ============================================================================
# TIER 1: pure build_cluster_feature_contract differential — byte-equal
# cells over the required structural fixture matrix.
# ============================================================================


def _assert_tier1_byte_equal(cell_name: str, tmp_path: Path) -> Any:
    cell = _golden_cell(cell_name)
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is False, (
        f"golden records the oracle RAISING for {cell_name!r}; this cell is "
        "documented as byte-equal -- golden/test have drifted out of sync"
    )
    stack = run_tier1_cell_stack(cell_name, tmp_path)
    assert not stack.raises, f"stack raised for {cell_name!r}: {stack.error_message}"
    assert stack.semantic_dict == oracle["semantic_dict"]
    assert stack.fingerprint == oracle["fingerprint"]
    return stack


def test_no_cluster_legacy_is_byte_equal(tmp_path: Path) -> None:
    """Fixture cell: no-cluster/legacy. ``FeatureClusterDefinition`` requires
    >= 1 cluster of >= 2 raw members (schema floor — see ``feature_
    clusters.py``'s own ``__post_init__``), so "no clustering" is
    represented the only way the schema allows: every defined cluster's
    second member (0, 7) sits entirely outside the mask=(1..6), so each
    cluster degenerates to a post-mask singleton and the representative
    universe is byte-identical to an unclustered/legacy run's mask-allowed
    set."""
    stack = _assert_tier1_byte_equal("no_cluster_legacy", tmp_path)
    assert stack.contract.consolidation.is_identity_after_mask is True


def test_multi_member_cluster_with_mask_shifted_representative_is_byte_equal(
    tmp_path: Path,
) -> None:
    """Fixture cell: multi-member cluster whose smallest member (1) is
    masked away, shifting the effective representative to 2."""
    stack = _assert_tier1_byte_equal(
        "multi_member_mask_shifted_representative", tmp_path
    )
    assert stack.contract.representative_universe[0] == 2


def test_singleton_after_mask_is_byte_equal(tmp_path: Path) -> None:
    """Fixture cell: cluster (2, 3) with member 3 masked away, leaving a
    post-mask singleton — exercised with threshold-shaped selection."""
    _assert_tier1_byte_equal("singleton_after_mask", tmp_path)


def test_inert_cluster_is_byte_equal(tmp_path: Path) -> None:
    """Fixture cell: cluster (0, 7) with BOTH members entirely outside the
    mask (fully inert — contributes nothing), mixed with a genuine
    multi-member cluster, exercised with explicit-whitelist concept
    source."""
    stack = _assert_tier1_byte_equal("inert_cluster", tmp_path)
    # 0 and 7 are outside the whitelist mask entirely by construction —
    # confirm the fully-inert cluster never enters any output domain.
    all_members = {
        member
        for _, members in stack.contract.representative_to_allowed_members
        for member in members
    }
    assert not ({0, 7} & all_members)


def test_threshold_selection_payload_is_byte_equal(tmp_path: Path) -> None:
    """Fixture cell: threshold-shaped selection JSON, multi-member cluster."""
    _assert_tier1_byte_equal("threshold_selection_payload", tmp_path)


def test_sampling_selection_payload_is_byte_equal(tmp_path: Path) -> None:
    """Fixture cell: sampling-shaped selection JSON, multi-member cluster."""
    _assert_tier1_byte_equal("sampling_selection_payload", tmp_path)


def test_empty_concept_projection_after_masking_diverges_named_bugfix(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 4 (MEDIUM, also independently found by
    lens-3): NAMED BUG-FIX DEVIATION, direct-builder path -- opposite
    direction from every other cell in this matrix (here the ORACLE
    succeeds and the STACK raises; every #759 cell above has the stack
    correctly REJECTING an oracle-accepted input, here the stack correctly
    REJECTS an input the oracle wrongly accepts as "successful").
    Reproduces lens-3's exact config: statistics/concept domain
    (2, 3, 4, 6) against a feature mask whitelisting only ids 500..505,
    entirely disjoint from that domain -- mask filtering removes every
    statistics-domain source, so a "successful" oracle contract would carry
    an empty source_to_representative and empty projecting_source_ids for
    every representative. See ``test_cluster_feature_contract.py``'s
    resolver- and expected-document-path regressions for the other two ways
    a caller can reach this same empty-projection condition."""
    cell = _golden_cell("empty_concept_projection_after_masking")
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is False  # oracle: SUCCEEDS (wrong)
    assert oracle["semantic_dict"] is not None
    assert oracle["semantic_dict"]["concept_source"]["source_to_representative"] == []

    stack = run_tier1_cell_stack("empty_concept_projection_after_masking", tmp_path)
    assert stack.raises
    assert stack.error_message is not None
    assert "concept source projection is empty" in stack.error_message


# ============================================================================
# TIER 2: full resolve_cluster_feature_contract filesystem preflight — the
# only entry point calling _assert_checkpoint_agreement, the one
# fingerprint-input site B2a/B2b's named #759 deviation touches.
# ============================================================================


def test_trivial_cluster_regime_diverges_only_on_the_named_num_buckets_dedup(
    tmp_path: Path,
) -> None:
    """No real consolidation (every cluster degenerates to a post-mask
    singleton; ``is_identity_after_mask=True``): the oracle's ``FeatureVocab.
    from_mask`` and ``effective_feature_vocab`` derive the IDENTICAL vocab,
    so the #759 rewiring is INERT here by construction — but this cell still
    DIVERGES, for a second, unrelated, already-named reason: B2b's Codex
    pass-2 residual finding 1 (MEDIUM, see ``cluster_checkpoint_pin``'s own
    docstring) removed ``num_buckets`` from ``_CHECKPOINT_USER_CONFIG_KEYS``
    because ``_extract_encdec_fields`` already normalizes it into ``load_
    bearing_metadata``'s TOP-LEVEL ``num_buckets`` field — the oracle still
    duplicates the same value a second time inside ``user_config``. Both
    sides agree on the VALUE (2); only the oracle's fingerprint hashes an
    extra, redundant key. Pin the stack's deduplicated shape as authoritative
    and prove it is the ONLY divergence: after removing that one named key
    from the oracle's (golden) semantic dict, the two are byte-identical."""
    cell = _golden_cell("trivial_cluster_num_buckets_dedup")
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is False

    stack = run_tier2_cell_stack("trivial_cluster_num_buckets_dedup", tmp_path)
    assert not stack.raises, stack.error_message
    assert stack.semantic_dict is not None
    assert stack.contract.consolidation.is_identity_after_mask is True

    oracle_user_config = oracle["semantic_dict"]["checkpoint"]["load_bearing_metadata"][
        "user_config"
    ]
    stack_user_config = stack.semantic_dict["checkpoint"]["load_bearing_metadata"][
        "user_config"
    ]
    assert oracle_user_config.get("num_buckets") == 2
    assert "num_buckets" not in stack_user_config
    # Both sides agree on the single, normalized, top-level value.
    assert (
        oracle["semantic_dict"]["checkpoint"]["load_bearing_metadata"]["num_buckets"]
        == stack.semantic_dict["checkpoint"]["load_bearing_metadata"]["num_buckets"]
        == 2
    )
    # The (golden-normalized) fingerprints DIVERGE (expected, named — the point).
    assert oracle["fingerprint"] != stack.fingerprint
    # Prove it is the ONLY divergence: strip the one named duplicate key from
    # the golden oracle's semantic dict and assert full equality follows.
    normalized_oracle_semantic = json.loads(json.dumps(oracle["semantic_dict"]))
    del normalized_oracle_semantic["checkpoint"]["load_bearing_metadata"]["user_config"][
        "num_buckets"
    ]
    normalized_oracle_semantic = _normalize_oracle_selection_manifest_hash(
        normalized_oracle_semantic,
        stack.semantic_dict,
    )
    assert normalized_oracle_semantic == stack.semantic_dict


def test_pretrain_layout_diverges_on_nested_encoder_xattn_normalization(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 3 (MEDIUM): a 4TH, previously un-enumerated
    divergence -- the full fixture matrix only ever exercised top-level/
    SFT-style encoder metadata (see ``trivial_cluster_num_buckets_dedup``
    above). B2 deliberately normalizes NESTED pretrain-layout metadata
    (``user_config.encoder``/``user_config.cross_attention`` --
    ``cluster_checkpoint_pin.py``'s ``_checkpoint_load_bearing_metadata`` ->
    ``nanochat.checkpoint_manager._extract_encdec_fields``, "Codex fix batch
    finding 4" in that module's own docstring), while the oracle only ever
    reads the TOP-LEVEL ``encoder_config``/``xattn_config`` keys and
    fingerprints the (present but nested) real values as ``None``. Same
    trivial (identity-after-mask) cluster as the num_buckets-dedup cell, so
    BOTH implementations still succeed -- only the semantic fingerprint
    diverges. ``num_buckets`` is deliberately kept TOP-LEVEL in this
    fixture (see ``Tier2CellSpec.metadata_layout``'s "nested_pretrain"
    branch in the generator), so this cell isolates ONLY the encoder/xattn
    divergence, cleanly separated from the num_buckets-dedup deviation
    above -- proven below by stripping ONLY those two keys and asserting
    full equality follows.

    Correcting the ledger: this is the divergence class summary's ledger's
    4TH distinct divergent cell, driven by a 3RD distinct named deviation
    (alongside #759 effective-vocab and B2b's num_buckets dedup) -- any
    prior "only three divergences"/"exactly 2 distinct named deviations"
    claim in task-b3-report.md is updated accordingly in this batch's
    report appendix."""
    cell = _golden_cell("pretrain_layout_encoder_xattn_normalization")
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is False

    stack = run_tier2_cell_stack("pretrain_layout_encoder_xattn_normalization", tmp_path)
    assert not stack.raises, stack.error_message
    assert stack.semantic_dict is not None
    assert stack.contract is not None
    assert stack.contract.consolidation.is_identity_after_mask is True

    oracle_lb = oracle["semantic_dict"]["checkpoint"]["load_bearing_metadata"]
    stack_lb = stack.semantic_dict["checkpoint"]["load_bearing_metadata"]
    assert oracle_lb["encoder_config"] is None
    assert oracle_lb["xattn_config"] is None
    assert stack_lb["encoder_config"] == {"n_embd": 8, "depth": 1}
    assert stack_lb["xattn_config"] == {"frequency": 1}
    # num_buckets was deliberately kept top-level in this fixture, so BOTH
    # sides agree on it (and on the whole user_config sub-dict) -- the
    # num_buckets-dedup deviation from the cell above does NOT recur here.
    assert oracle_lb["num_buckets"] == stack_lb["num_buckets"] == 2
    assert oracle_lb["user_config"] == stack_lb["user_config"]

    # The (golden-normalized) fingerprints DIVERGE (expected, named).
    assert oracle["fingerprint"] != stack.fingerprint
    # Prove it is the ONLY divergence: strip ONLY encoder_config/xattn_config
    # from the golden oracle's semantic dict and assert full equality follows.
    normalized_oracle_semantic = json.loads(json.dumps(oracle["semantic_dict"]))
    normalized_lb = normalized_oracle_semantic["checkpoint"]["load_bearing_metadata"]
    normalized_lb["encoder_config"] = stack_lb["encoder_config"]
    normalized_lb["xattn_config"] = stack_lb["xattn_config"]
    normalized_oracle_semantic = _normalize_oracle_selection_manifest_hash(
        normalized_oracle_semantic,
        stack.semantic_dict,
    )
    assert normalized_oracle_semantic == stack.semantic_dict


def test_genuine_cluster_wide_vocab_diverges_named_759(tmp_path: Path) -> None:
    """Genuine consolidation (cluster (0,1) really consolidates 1 away) + a
    checkpoint carrying the WIDE, mask-only vocab (the pre-#759 shape every
    checkpoint used to carry): the oracle ACCEPTS it (blind to cluster
    consolidation); the B1/B2/B3 stack REJECTS it (Codex fix batch finding
    1, CRITICAL — see ``cluster_checkpoint_pin``'s docstring) because a
    genuinely clustered encoder's persistent vocab must be the NARROWER
    effective-vocab shape. DIVERGES; the stack's raise is the correct, NEW,
    authoritative behavior — no fingerprint to pin since neither side
    succeeds with a usable value that could be silently wrong."""
    cell = _golden_cell("genuine_cluster_wide_vocab_rejected")
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is False  # oracle: SUCCEEDS (wrong — #759-blind)
    assert oracle["fingerprint"]

    stack = run_tier2_cell_stack("genuine_cluster_wide_vocab_rejected", tmp_path)
    assert stack.raises
    assert stack.error_message is not None
    assert "effective-vocabulary training" in stack.error_message or "#759" in stack.error_message


def test_genuine_cluster_effective_vocab_diverges_named_759(tmp_path: Path) -> None:
    """Same genuine consolidation, but the checkpoint carries the NARROW,
    ``effective_feature_vocab``-derived shape (the post-#759, CORRECT shape
    a real clustered training run now produces): the oracle REJECTS it (its
    ``FeatureVocab.from_mask``-only comparison cannot recognize the
    narrower, consolidation-aware vocab as valid); the stack ACCEPTS it and
    produces a fingerprint. DIVERGES; the stack's fingerprint is pinned as
    the new, authoritative value THREE independent ways (Codex fix batch
    finding 2, HIGH -- superseding a prior version of this test that only
    checked within-run determinism, which a deterministic-but-WRONG
    fingerprint would also have passed):

    1. Within-run determinism: resolving the identical on-disk artifact
       twice yields the identical fingerprint both times.
    2. Internal consistency: independently recomputing the SHA-256 of the
       contract's own ``semantic_dict()`` (via a plain ``hashlib``/``json``
       call, NOT by reusing production's private ``_sha256_json``/
       ``_canonical_json`` helpers) reproduces ``contract.feature_contract_
       fingerprint`` exactly -- catches a hypothetical bug in the
       fingerprint computation itself that a purely self-referential check
       could not.
    3. Cross-run regression pin: the golden-normalized semantic dict and its
       SHA-256 match a COMMITTED, STACK-GENERATED golden value (honestly
       labeled -- the pre-#759 oracle cannot produce this value at all, so
       there is no oracle value to pin against; see ``STACK_PINNED_CELLS``
       in the generator). This is what makes the value independently
       pinned ACROSS commits, not merely self-determined at test time.
    """
    cell = _golden_cell("genuine_cluster_effective_vocab_accepted")
    oracle = cell["oracle_outcome"]
    assert oracle["raises"] is True  # oracle: RAISES (wrong)
    assert oracle["error_message"] is not None
    assert (
        "state-dict FeatureVocab" in oracle["error_message"]
        or "do not agree" in oracle["error_message"]
    )

    # 1. Build the on-disk fixture ONCE, resolve it TWICE -- "resolving the
    # identical on-disk artifact twice yields the identical fingerprint" is
    # a claim about repeated resolution of ONE frozen artifact, not about
    # two independently rebuilt (if logically identical) fixture trees.
    request, sae_loader = build_tier2_request_stack(
        "genuine_cluster_effective_vocab_accepted", tmp_path
    )
    stack = resolve_tier2_outcome_stack(request, sae_loader)
    assert not stack.raises, stack.error_message
    assert stack.contract is not None
    assert stack.semantic_dict is not None
    replay = resolve_tier2_outcome_stack(request, sae_loader)
    assert replay.fingerprint == stack.fingerprint

    # 2. Internal consistency: independently recomputed SHA-256 of the
    # LIVE contract's raw (non-normalized) semantic_dict() must equal its
    # own feature_contract_fingerprint.
    independent_sha = hashlib.sha256(
        json.dumps(
            stack.contract.semantic_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert independent_sha == stack.contract.feature_contract_fingerprint

    # 3. Cross-run regression pin against the committed, stack-generated
    # golden (golden-normalized form -- see the generator's own "Golden
    # reproducibility" docstring for why the raw form can never be a
    # cross-run-reproducible literal for a real-filesystem fixture).
    golden = _load_golden()
    assert "stack_pinned_cells" in golden, (
        "golden JSON predates the finding-2 stack_pinned_cells section -- "
        "regenerate it (see test_golden_committed's message for the command)"
    )
    pinned = golden["stack_pinned_cells"]["genuine_cluster_effective_vocab_accepted"]
    assert pinned["label"] == (
        "post-#759 production-correct value; the pre-#759 oracle cannot produce it"
    )
    assert stack.semantic_dict == pinned["normalized_semantic_dict"]
    assert stack.fingerprint == pinned["normalized_semantic_sha256"]
