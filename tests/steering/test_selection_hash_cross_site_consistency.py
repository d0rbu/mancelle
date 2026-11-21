"""7-site anti-desync pin: compute_selection_hash kwarg parity + value consistency.

This test is the EXECUTABLE form of the "Change ALL 4 or none" comment annotated
at every production call site.  It has two parts:

Part A — Value-level invariants via the REAL site-4 callable
============================================================
Imports and drives ``compute_canonical_feature_selection_hash`` from
``steering.eval.feature_selection_hash`` (production site 4).  Asserts:
  - no-mask/no-cluster result == omitting either kwarg == frozen golden (I1).
  - blacklist mask changes the hash; blacklist != whitelist != no-mask.
  - a canonical feature-cluster descriptor changes the hash, while ``None``
    remains byte-identical to the pre-feature-cluster hash.
  - canonical id-order invariance: (3,1,2) same as (1,2,3).
  - frozen golden pins the blacklist hash (regen ONLY on intentional semantics change).

Part B — Structural cross-site drift detector via AST
=====================================================
Parses the SEVEN production source files with Python's ``ast`` module and locates
the ``compute_selection_hash(`` call at each documented site.  Extracts the set of
keyword-argument NAMES passed at each site.  Asserts:
  (b1) All 7 sites pass the IDENTICAL set of kwarg names — adding, dropping, or
       renaming a kwarg at any one site fails this test immediately.
  (b2) Every site passes a ``feature_mask=`` keyword arg.  Sites 1, 2, 5, and 7 must
       wrap the value in a canonical_dict(...)-style call (ast.Call whose func
       resolves to "canonical_dict" or site 5's import alias
       "canonical_feature_mask_dict"); sites 3, 4, and 6 must pass a bare Name
       (already canonical). This pins that the mask is canonicalized consistently
       at every site.
  (b3) Every site passes ``feature_clusters=``. Sites 1, 2, and 7 must derive it
       through ``canonical_feature_clusters_descriptor(...)``; sites 3, 4, and 6
       receive an already-canonical descriptor as a bare Name.
  (b4) Every site passes the splice batching value through the shared
       conditional canonicalizer. The producer uses its validated local value;
       training reads the matching config key; wrapper sites pass their
       explicit argument.

The 7 production sites (file:line, enclosing function):
  Site 1: scripts/preprocess/precompute_selected_features.py        (main)
  Site 2: scripts/train/conditional_encdec_train.py                 (discover_packed_selection)
  Site 3: scripts/eval/base_eval.py                                 (_resolve_bpb_selection_dir)
  Site 4: steering/eval/feature_selection_hash.py                   (compute_canonical_feature_selection_hash)
  Site 5: steering/eval/cluster_checkpoint_pin.py                   (_resolve_selection_identity)
  Site 6: steering/eval/cache_identity.py                           (_canonical_hash_from_conditioning, clustered branch)
  Site 7: scripts/train/conditional_encdec_train.py                 (_build_cached_context_provider, clustered cross-check)

A "drift" means someone edited one site to rename/drop/reorder a kwarg, or stopped
calling ``canonical_dict`` before passing the mask.  This silently orphans masked
``sel/<hash>/`` shards on disk.  Part B catches that immediately.

Codex combined-review fix batch, finding 6 (MEDIUM): site 5 (PR #739 Track
B2b, ``_resolve_selection_identity``) was added after this module's
original 4-site design and reconstructed the correct 8 canonical inputs on
day one, but was never added to ``_PRODUCTION_SITES`` — so a future edit
that silently forked site 5's kwargs from the other four would go
completely undetected by this drift guard. Site 5 imports the mask
canonicalizer under a local alias (``canonical_feature_mask_dict``, not the
bare ``canonical_dict`` name sites 1/2 use) — ``_CANONICAL_DICT_ALIASES``
below recognizes both without requiring cluster_checkpoint_pin.py's own
import to change (out of scope; the 4 existing production sites are not
refactored either). ``TestSite5BehavioralParity`` adds a genuine VALUE-level
check beyond Part B's structural kwarg-name/wrapper-call comparison: it
drives the real ``_resolve_selection_identity`` and asserts its hash
matches a directly-constructed ``compute_selection_hash`` call built from
the identical conceptual inputs, so a future site-5 edit that silently
stops forwarding one of its inputs into the hash (while keeping the same
KEYWORD name — invisible to Part B's AST-only check) still fails loud.

PR #739 Track D ("D-seed"): site 6 (``steering/eval/cache_identity.py``'s
``_canonical_hash_from_conditioning``, clustered branch) fixes the verified
gap this module's own drift guard was built to catch -- the clustered
eval-context-cache IDENTITY hash was hardcoded at ``seed=0``
(``feature_selection_hash.py``'s ``compute_canonical_feature_selection_hash``
wrapper), so no clustered cache ever reproduced its checkpoint's real
training-preselection seed. Site 6 branches the clustered case onto a
direct ``compute_selection_hash(seed=cond.seed, ...)`` call, mirroring site
5's checkpoint-preflight identity. ``TestSite6BehavioralParity`` mirrors
``TestSite5BehavioralParity``: a value-level parity check against a
directly-constructed ``compute_selection_hash`` call, PLUS a seed-dependency
sentinel that perturbs ONLY ``seed`` and asserts the hash changes -- the
unification-proof that site 6 actually threads the real seed through
(invisible to Part B's AST-only structural check).

Codex cross-review (D-seed Finding 3, MEDIUM): site 6's fix required a
matching fix at a SEVENTH call site --
``scripts/train/conditional_encdec_train.py``'s
``_build_cached_context_provider``, which independently re-derives the same
clustered prompt-cache identity from the TOP-LEVEL training config as a
cross-check against site 6's resolved identity -- but this new direct
``compute_selection_hash`` call was never added to ``_PRODUCTION_SITES``,
so restoring its ``seed=0`` or dropping a kwarg there would have left this
entire drift suite green. Site 7's ``feature_mask=``/``feature_clusters=``
arguments are live ``canonical_dict(...)``/
``canonical_feature_clusters_descriptor(...)`` calls (matching sites 1/2/5's
pattern, since this site receives raw resolved ``FeatureMask``/
``FeatureClusterConsolidation`` objects, not already-canonical bare
variables like sites 3/4/6); its ``splice_batch_size=`` is a literal
``None`` constant (matching sites 5/6's pattern -- ``delta_config=None`` is
unconditional here too, for the identical reason: this is the delta-free
clustered PROMPT selection cross-check, so splice batching is provably
inert regardless of the value passed).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from omegaconf import OmegaConf

from steering.conditional.feature_clusters import (
    FEATURE_CLUSTERS_SEMANTICS,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    canonical_feature_clusters_descriptor,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode, canonical_dict
from steering.eval.cache_identity import resolve_feature_selection_hash
from steering.eval.cluster_checkpoint_pin import _resolve_selection_identity
from steering.eval.feature_selection_hash import compute_canonical_feature_selection_hash
from steering.packed_batch_io import compute_selection_hash

# ---------------------------------------------------------------------------
# Repo root (two levels up from this test file: tests/steering → tests → repo)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[2]

# ---------------------------------------------------------------------------
# The five production site locations.
# (file_path_relative_to_repo, enclosing_function, approx_line, label)
# Update this table ONLY if a site is deliberately moved.
# ---------------------------------------------------------------------------
_PRODUCTION_SITES: list[tuple[str, str, int, str]] = [
    # (rel-path,                                            enclosing-fn,                         approx-line, label)
    ("scripts/preprocess/precompute_selected_features.py", "main", 1852, "site1-precompute"),
    ("scripts/train/conditional_encdec_train.py", "discover_packed_selection", 1515, "site2-train"),
    ("scripts/eval/base_eval.py", "_resolve_bpb_selection_dir", 383, "site3-eval"),
    (
        "steering/eval/feature_selection_hash.py",
        "compute_canonical_feature_selection_hash",
        84,
        "site4-cache-identity",
    ),
    (
        "steering/eval/cluster_checkpoint_pin.py",
        "_resolve_selection_identity",
        1464,
        "site5-checkpoint-pin",
    ),
    (
        "steering/eval/cache_identity.py",
        "_canonical_hash_from_conditioning",
        213,
        "site6-eval-context-cache-clustered",
    ),
    (
        "scripts/train/conditional_encdec_train.py",
        "_build_cached_context_provider",
        1354,
        "site7-training-crosscheck",
    ),
]

# canonical_dict's local import alias at site 5 (cluster_checkpoint_pin.py
# imports it as `canonical_feature_mask_dict` — see Codex combined-review
# finding 6's module-docstring note above). The 4 original sites are not
# refactored to match; this test recognizes both spellings instead.
_CANONICAL_DICT_ALIASES: frozenset[str] = frozenset(
    {"canonical_dict", "canonical_feature_mask_dict"}
)

# Expected kwarg names — all 7 sites must agree on exactly this set.
_EXPECTED_KWARGS: frozenset[str] = frozenset(
    {
        "positive_selection",
        "negative_selection",
        "delta_config",
        "num_buckets",
        "seed",
        "ddp_rank",
        "feature_mask",
        "feature_clusters",
        "splice_batch_size",
    }
)

# ---------------------------------------------------------------------------
# Shared payload for Part A value-level tests.
# seed=0 / ddp_rank=0 match sites 3 and 4 (sites 1+2 use cfg.seed in
# production; the test normalises to 0 so all sites agree for this payload).
# ---------------------------------------------------------------------------
_POS = {"distribution": "geometric", "mean": 4.0, "std": 4.0}
_NEG = {"distribution": "geometric", "mean": 3.0, "std": 3.0}
_NUM_BUCKETS = 8
_SEED = 0
_DDPRANK = 0
_DELTA = None

_BL_MASK = FeatureMask(FeatureMaskMode.blacklist, (3, 7, 15))
_WL_MASK = FeatureMask(FeatureMaskMode.whitelist, (1, 5, 20))
_CLUSTER_DESCRIPTOR = {
    "schema_version": 1,
    "n_sae_features": 65_536,
    "content_sha256": "1" * 64,
    "semantics": FEATURE_CLUSTERS_SEMANTICS,
}

# ---------------------------------------------------------------------------
# FROZEN GOLDENS
# Computed once (2026-05-18) with the payload above via compute_selection_hash.
# Regen ONLY if compute_selection_hash semantics intentionally change — update
# the value AND this comment together.
# ---------------------------------------------------------------------------
# no-mask golden:
#   compute_selection_hash(_POS, _NEG, None, 8, 0, 0)  (no feature_mask kwarg)
_NO_MASK_GOLDEN = "a1b315291927e2dee50125ac273f5916c71449870e4db68729786b31bb590470"

# blacklist golden (same args as the old _BL_GOLDEN_HASH; value preserved):
#   mask = FeatureMask(FeatureMaskMode.blacklist, (3, 7, 15))
#   compute_selection_hash(..., feature_mask=canonical_dict(mask))
_BL_GOLDEN_HASH = "4da0e7a94317817f8d932ba8ad8e6ac9dea3520fd9d8c2447f8394d13ac2bde5"


# ---------------------------------------------------------------------------
# AST helpers for Part B
# ---------------------------------------------------------------------------


def _get_func_name(node: ast.expr) -> str | None:
    """Return the simple name of a call's func node, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _find_compute_selection_hash_call(
    tree: ast.Module,
    enclosing_fn_name: str,
    approx_line: int,
    site_label: str,
) -> ast.Call:
    """Locate the compute_selection_hash(...) call inside *enclosing_fn_name*.

    Strategy:
      1. Walk the AST to find the FunctionDef whose name == enclosing_fn_name.
      2. Inside that function, collect all ast.Call nodes whose func resolves to
         the name "compute_selection_hash".
      3. If exactly one such call exists, return it.
      4. If multiple calls exist (shouldn't happen at current sites), pick the one
         whose lineno is closest to approx_line.
      5. If no call is found, raise AssertionError with a clear message — a missing
         site IS a desync signal; the test must NOT silently skip.
    """
    # Step 1: find the enclosing function definition
    enclosing: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == enclosing_fn_name:
                enclosing = node
                break

    if enclosing is None:
        raise AssertionError(
            f"[{site_label}] Could not find function '{enclosing_fn_name}' in AST. "
            "Has the function been renamed or removed? Update _PRODUCTION_SITES."
        )

    # Step 2: collect all compute_selection_hash Call nodes inside that function
    calls: list[ast.Call] = []
    for node in ast.walk(enclosing):
        if isinstance(node, ast.Call):
            name = _get_func_name(node.func)
            if name == "compute_selection_hash":
                calls.append(node)

    if not calls:
        raise AssertionError(
            f"[{site_label}] No compute_selection_hash(...) call found inside "
            f"'{enclosing_fn_name}'. Has the call been renamed, moved, or removed? "
            "Update _PRODUCTION_SITES."
        )

    # Step 3/4: disambiguate if multiple calls
    if len(calls) == 1:
        return calls[0]

    # Pick closest to approx_line
    best = min(calls, key=lambda c: abs(c.lineno - approx_line))
    return best


def _extract_site_kwarg_info(
    rel_path: str,
    enclosing_fn: str,
    approx_line: int,
    site_label: str,
) -> tuple[frozenset[str], dict[str, ast.expr]]:
    """Parse the file and return (kwarg_name_set, kwarg_name_to_value_node).

    Raises AssertionError (not skips) if the site cannot be located.
    """
    abs_path = _REPO_ROOT / rel_path
    if not abs_path.is_file():
        raise AssertionError(
            f"[{site_label}] Source file not found: {abs_path}. "
            "Has the file been moved or deleted? Update _PRODUCTION_SITES."
        )
    source = abs_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(abs_path))

    call_node = _find_compute_selection_hash_call(tree, enclosing_fn, approx_line, site_label)

    kwarg_names = frozenset(kw.arg for kw in call_node.keywords if kw.arg is not None)
    kwarg_values: dict[str, ast.expr] = {
        kw.arg: kw.value for kw in call_node.keywords if kw.arg is not None
    }
    return kwarg_names, kwarg_values


# ---------------------------------------------------------------------------
# Part A: Value-level invariants via the REAL site-4 callable
# ---------------------------------------------------------------------------


class TestSite4RealCallable:
    """Drive the REAL production site-4 callable (compute_canonical_feature_selection_hash)
    to verify value-level hash correctness and frozen goldens.

    This genuinely exercises production site 4; it is NOT a stub.
    """

    def test_no_mask_equals_legacy_no_arg(self) -> None:
        """I1: feature_mask=None must produce same hash as omitting kwarg entirely.

        Frozen golden pinned (2026-05-18).  Legacy seldir/ dirs computed without
        the feature_mask kwarg resolve to the same hash as feature_mask=None runs.
        """
        stats_sha = "a" * 64  # arbitrary deterministic sha
        h_none = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=None,
        )
        h_omit = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            # feature_mask omitted
        )
        assert h_none == h_omit, (
            f"I1 violation: feature_mask=None changed hash vs omitting kwarg: {h_none} != {h_omit}"
        )

    def test_no_mask_frozen_golden_direct(self) -> None:
        """I1 back-compat: direct compute_selection_hash (no feature_mask) == frozen golden.

        Frozen golden: _NO_MASK_GOLDEN (2026-05-18).
        Regen ONLY if compute_selection_hash semantics intentionally change.
        """
        h = compute_selection_hash(_POS, _NEG, _DELTA, _NUM_BUCKETS, _SEED, _DDPRANK)
        assert h == _NO_MASK_GOLDEN, (
            f"No-mask golden mismatch: got {h!r}, expected {_NO_MASK_GOLDEN!r}. "
            "If compute_selection_hash semantics changed intentionally, "
            "update _NO_MASK_GOLDEN in this file."
        )

    def test_no_clusters_none_equals_omitted_and_frozen_golden(self) -> None:
        """I1: a disabled cluster block must not perturb the legacy payload."""
        h_omitted = compute_selection_hash(_POS, _NEG, _DELTA, _NUM_BUCKETS, _SEED, _DDPRANK)
        h_none = compute_selection_hash(
            _POS,
            _NEG,
            _DELTA,
            _NUM_BUCKETS,
            _SEED,
            _DDPRANK,
            feature_clusters=None,
        )
        assert h_none == h_omitted == _NO_MASK_GOLDEN

    def test_site4_no_clusters_none_equals_omitted(self) -> None:
        """The cache-identity wrapper preserves the conditional-key invariant."""
        h_omitted = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256="9" * 64,
        )
        h_none = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256="9" * 64,
            feature_clusters=None,
        )
        assert h_none == h_omitted

    def test_site4_legacy_splice_batching_values_equal_omitted(self) -> None:
        """Null/1 must not add a field to the established splice payload."""
        sha = "7" * 64
        delta = {"mode": "splice_counterfactual", "stats_source_sha256": sha}
        omitted = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=sha,
            feature_delta={"mode": "splice_counterfactual"},
        )
        for legacy_value in (None, 1):
            assert compute_canonical_feature_selection_hash(
                positive_selection=_POS,
                negative_selection=_NEG,
                num_buckets=_NUM_BUCKETS,
                stats_source_sha256=sha,
                feature_delta={"mode": "splice_counterfactual"},
                splice_batch_size=legacy_value,
            ) == omitted
            assert compute_selection_hash(
                _POS,
                _NEG,
                delta,
                _NUM_BUCKETS,
                0,
                0,
                splice_batch_size=legacy_value,
            ) == compute_selection_hash(
                _POS,
                _NEG,
                delta,
                _NUM_BUCKETS,
                0,
                0,
            )

    def test_site4_active_batching_matches_direct_splice_hash(self) -> None:
        sha = "6" * 64
        feature_delta = {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
        direct_delta = {**feature_delta, "stats_source_sha256": sha}
        wrapped = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=sha,
            feature_delta=feature_delta,
            splice_batch_size=64,
        )
        direct = compute_selection_hash(
            _POS,
            _NEG,
            direct_delta,
            _NUM_BUCKETS,
            0,
            0,
            splice_batch_size=64,
        )
        legacy = compute_selection_hash(
            _POS,
            _NEG,
            direct_delta,
            _NUM_BUCKETS,
            0,
            0,
        )
        assert wrapped == direct
        assert wrapped != legacy

    def test_cluster_descriptor_changes_hash(self) -> None:
        """Active canonical cluster semantics receive a distinct selection dir."""
        h_none = compute_selection_hash(_POS, _NEG, _DELTA, _NUM_BUCKETS, _SEED, _DDPRANK)
        h_clustered = compute_selection_hash(
            _POS,
            _NEG,
            _DELTA,
            _NUM_BUCKETS,
            _SEED,
            _DDPRANK,
            feature_clusters=_CLUSTER_DESCRIPTOR,
        )
        assert h_clustered != h_none

    def test_cluster_delta_semantics_is_strictly_conditional_with_goldens(self) -> None:
        """Only static-cluster + splice-CF receives the new identity field."""
        descriptor = {
            "schema_version": 1,
            "n_sae_features": 32,
            "content_sha256": "a" * 64,
            "semantics": FEATURE_CLUSTERS_SEMANTICS,
        }
        positive = {
            "distribution": "geometric",
            "mean": 12,
            "std": 12,
            "min_k": 0,
            "max_k": 49,
            "mode": "top_k_percentile",
            "delta_weights": "percentile_delta",
        }
        negative = {
            "distribution": "geometric",
            "mean": 12,
            "min_k": 0,
            "max_k": 49,
        }
        splice = {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}

        unclustered_splice = compute_selection_hash(positive, negative, splice, 8, 42, 0)
        clustered_splice = compute_selection_hash(
            positive,
            negative,
            splice,
            8,
            42,
            0,
            feature_clusters=descriptor,
        )
        clustered_delta_free = compute_selection_hash(
            positive,
            negative,
            None,
            8,
            42,
            0,
            feature_clusters=descriptor,
        )

        # These golden values pin the current all-49 sampling payload.
        assert unclustered_splice == (
            "18f7ce0b0de84156ecccd140b206840d317ed041b0b4624477cfcf8444a82fb5"
        )
        assert clustered_delta_free == (
            "6f3ec36e9c4209965bb118cafd5e6207da2b05be54a69ffb6ad62099c25fffc3"
        )
        # The old clustered-splice payload (without cluster_delta_semantics)
        # hashed to f5a7...; the corrected semantic currency must not collide.
        assert clustered_splice == (
            "4eee52f016976542c1042004e4b4ec2dd719f6213b996c862d58452ea3f1f1e4"
        )
        assert clustered_splice != (
            "f5a7d82d202f57e90d2f22ba232840d39ce165c3aa4c7cb3751ab91745729529"
        )

    def test_site4_cluster_descriptor_matches_direct(self) -> None:
        """Site 4 forwards the exact canonical descriptor into the shared hash."""
        sha = "8" * 64
        delta_config = {"stats_source_sha256": sha}
        wrapped = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=sha,
            feature_clusters=_CLUSTER_DESCRIPTOR,
        )
        direct = compute_selection_hash(
            _POS,
            _NEG,
            delta_config,
            _NUM_BUCKETS,
            0,
            0,
            feature_clusters=_CLUSTER_DESCRIPTOR,
        )
        assert wrapped == direct

    def test_blacklist_frozen_golden(self) -> None:
        """FROZEN GOLDEN: pin the blacklist hash for this payload via site-4 wrapper.

        Golden computed (2026-05-18) with:
            POS={'distribution':'geometric','mean':4.0,'std':4.0}
            NEG={'distribution':'geometric','mean':3.0,'std':3.0}
            delta=None, num_buckets=8, seed=0, ddp_rank=0
            mask={'mode':'blacklist','ids':[3,7,15]}
        Regen ONLY if compute_selection_hash semantics intentionally change.
        """
        # Use direct compute_selection_hash (same as old test) so the golden is stable
        # regardless of stats_source_sha256 injected by the wrapper.
        h = compute_selection_hash(
            _POS,
            _NEG,
            _DELTA,
            _NUM_BUCKETS,
            _SEED,
            _DDPRANK,
            feature_mask=canonical_dict(_BL_MASK),
        )
        assert h == _BL_GOLDEN_HASH, (
            f"Blacklist golden mismatch: got {h!r}, expected {_BL_GOLDEN_HASH!r}. "
            "If compute_selection_hash semantics changed intentionally, "
            "update _BL_GOLDEN_HASH in this file."
        )

    def test_blacklist_differs_from_whitelist(self) -> None:
        """Blacklist and whitelist of same ids produce different hashes (mode is in payload)."""
        wl_same_ids = FeatureMask(FeatureMaskMode.whitelist, _BL_MASK.ids)
        stats_sha = "b" * 64
        h_bl = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(_BL_MASK),
        )
        h_wl = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(wl_same_ids),
        )
        assert h_bl != h_wl, (
            f"Blacklist and whitelist (same ids) must produce different hashes; both = {h_bl}"
        )

    def test_blacklist_whitelist_none_all_differ(self) -> None:
        """blacklist != whitelist != no-mask (all three are distinct)."""
        stats_sha = "c" * 64
        h_bl = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(_BL_MASK),
        )
        h_wl = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(_WL_MASK),
        )
        h_none = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=None,
        )
        assert h_bl != h_none, "Blacklist must differ from no-mask"
        assert h_wl != h_none, "Whitelist must differ from no-mask"
        assert h_bl != h_wl, "Blacklist must differ from whitelist"

    def test_canonical_id_order_invariance(self) -> None:
        """canonical_dict sorts ids, so (3,1,2) and (1,2,3) produce identical hashes."""
        mask_reordered = FeatureMask(FeatureMaskMode.blacklist, (3, 1, 2))
        mask_ordered = FeatureMask(FeatureMaskMode.blacklist, (1, 2, 3))
        stats_sha = "d" * 64
        h_reord = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(mask_reordered),
        )
        h_ord = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=stats_sha,
            feature_delta=None,
            feature_mask=canonical_dict(mask_ordered),
        )
        assert h_reord == h_ord, (
            f"canonical_dict must sort ids: (3,1,2) hash {h_reord} != (1,2,3) hash {h_ord}"
        )

    def test_site4_wrapper_mask_delta_matches_direct(self) -> None:
        """Site 4 wrapper computes the same hash as direct compute_selection_hash for identical inputs.

        Verifies that the wrapper correctly passes through the feature_mask argument
        without transforming it — the mask contribution is identical to the direct path.
        """
        sha = "a" * 64
        canon = canonical_dict(_BL_MASK)
        delta_config = {
            "stats_source_sha256": sha
        }  # what the wrapper injects when feature_delta=None

        h_masked_wrapped = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=sha,
            feature_delta=None,
            feature_mask=canon,
        )
        h_none_wrapped = compute_canonical_feature_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            num_buckets=_NUM_BUCKETS,
            stats_source_sha256=sha,
            feature_delta=None,
            feature_mask=None,
        )
        h_masked_direct = compute_selection_hash(
            _POS, _NEG, delta_config, _NUM_BUCKETS, 0, 0, feature_mask=canon
        )
        h_none_direct = compute_selection_hash(
            _POS, _NEG, delta_config, _NUM_BUCKETS, 0, 0, feature_mask=None
        )

        assert h_masked_wrapped == h_masked_direct, (
            "Site 4 wrapper must equal direct compute_selection_hash for same inputs (masked)"
        )
        assert h_none_wrapped == h_none_direct, (
            "Site 4 wrapper must equal direct compute_selection_hash for same inputs (no mask)"
        )
        assert h_masked_wrapped != h_none_wrapped, (
            "Masked and no-mask hashes must differ through site 4 wrapper"
        )


# ---------------------------------------------------------------------------
# Part B: Structural cross-site drift detector via AST
# ---------------------------------------------------------------------------


class TestCrossSiteASTDriftDetector:
    """Executable form of the "Change ALL FOUR/FIVE/SIX/SEVEN or none"
    anti-desync comments (sites 1-4's own comments still literally say
    "ALL FOUR"/"ALL 4" — they predate site 5 and are not edited here, out
    of this fix batch's scope; site 5's own comment says "ALL FIVE" — see
    steering/eval/cluster_checkpoint_pin.py; sites 6 and 7, added by PR
    #739 Track D "D-seed" and its Codex cross-review fix respectively, are
    not themselves commented with a literal count either, same
    precedent). This test is the actual 7-way enforcement mechanism, not
    the prose.

    Parses the seven production source files with Python's ast module and
    extracts the keyword-argument names passed to compute_selection_hash at
    each documented site.  The test fails (not skips) if any site cannot be
    found, or if any site's kwarg set drifts from the shared expected set.

    A drift = someone renames/drops/adds a kwarg at one site without updating
    all others, or stops wrapping the mask in canonical_dict.  Either change
    would silently orphan masked sel/<hash>/ shards.
    """

    def _load_all_sites(self) -> list[tuple[str, frozenset[str], dict[str, ast.expr]]]:
        """Return [(label, kwarg_names, kwarg_values)] for all 7 sites.

        Raises AssertionError if any site cannot be located.
        """
        results = []
        for rel_path, enclosing_fn, approx_line, label in _PRODUCTION_SITES:
            kwarg_names, kwarg_values = _extract_site_kwarg_info(
                rel_path, enclosing_fn, approx_line, label
            )
            results.append((label, kwarg_names, kwarg_values))
        return results

    def test_b1_all_sites_identical_kwarg_names(self) -> None:
        """(b1) All 7 sites pass the IDENTICAL set of kwarg names to compute_selection_hash.

        If any site adds, drops, or renames a kwarg this test fails immediately.
        Expected kwarg set: positive_selection, negative_selection, delta_config,
        num_buckets, seed, ddp_rank, feature_mask.
        """
        all_sites = self._load_all_sites()

        mismatches: list[str] = []
        for label, kwarg_names, _ in all_sites:
            if kwarg_names != _EXPECTED_KWARGS:
                extra = kwarg_names - _EXPECTED_KWARGS
                missing = _EXPECTED_KWARGS - kwarg_names
                mismatches.append(f"  {label}: extra={sorted(extra)}, missing={sorted(missing)}")

        assert not mismatches, (
            "compute_selection_hash kwarg drift detected — at least one site "
            "does not match the expected kwarg set.\n"
            "Expected: " + str(sorted(_EXPECTED_KWARGS)) + "\n"
            "Drifting sites:\n" + "\n".join(mismatches) + "\n"
            "Update ALL 7 sites together or none (see ANTI-DESYNC comments)."
        )

        # Also assert pairwise agreement so a different expected-set can't hide drift
        labels_and_names = [(lbl, kn) for lbl, kn, _ in all_sites]
        for i, (lbl_a, kn_a) in enumerate(labels_and_names):
            for lbl_b, kn_b in labels_and_names[i + 1 :]:
                assert kn_a == kn_b, (
                    f"Kwarg drift between {lbl_a} and {lbl_b}: "
                    f"{lbl_a}={sorted(kn_a)}, {lbl_b}={sorted(kn_b)}"
                )

    def test_b2_feature_mask_present_and_correctly_canonicalized(self) -> None:
        """(b2) Every site passes feature_mask= AND canonicalizes it consistently.

        Sites 1, 2, 5, and 7 must wrap the value in a canonical_dict(...)
        call (ast.Call whose func resolves to "canonical_dict" or its
        site-5 import alias "canonical_feature_mask_dict" — see
        _CANONICAL_DICT_ALIASES).  Sites 3, 4, and 6 must pass a bare Name
        (already canonical variable: feature_mask_canonical / pass-through
        feature_mask).

        This prevents any site from silently passing a raw/un-canonicalized mask.
        """
        all_sites = self._load_all_sites()

        # Sites that must wrap in canonical_dict(...) (or its site-5 alias)
        sites_requiring_canonical_dict_call = {
            "site1-precompute",
            "site2-train",
            "site5-checkpoint-pin",
            "site7-training-crosscheck",
        }
        # Sites that must pass a bare Name (already canonical)
        sites_requiring_bare_name = {
            "site3-eval",
            "site4-cache-identity",
            "site6-eval-context-cache-clustered",
        }

        for label, kwarg_names, kwarg_values in all_sites:
            # (b2a) feature_mask must be present as a kwarg at every site
            assert "feature_mask" in kwarg_names, (
                f"[{label}] feature_mask= kwarg is MISSING from compute_selection_hash call. "
                "This breaks mask canonicalization. Update ALL 7 sites together."
            )

            mask_value_node = kwarg_values["feature_mask"]

            if label in sites_requiring_canonical_dict_call:
                # Must be a Call whose func resolves to canonical_dict (or
                # its site-5 import alias)
                assert isinstance(mask_value_node, ast.Call), (
                    f"[{label}] feature_mask= value must be a canonical_dict(...)-style call, "
                    f"but got AST node type {type(mask_value_node).__name__}. "
                    "Someone may have passed a raw mask without canonicalization."
                )
                func_name = _get_func_name(mask_value_node.func)
                assert func_name in _CANONICAL_DICT_ALIASES, (
                    f"[{label}] feature_mask= value is a call but func name is "
                    f"{func_name!r}, expected one of {sorted(_CANONICAL_DICT_ALIASES)}. "
                    "Mask canonicalization may have been bypassed."
                )

            elif label in sites_requiring_bare_name:
                # Must be a bare Name (already canonical variable)
                assert isinstance(mask_value_node, ast.Name), (
                    f"[{label}] feature_mask= value must be a bare Name (already-canonical "
                    f"variable), but got AST node type {type(mask_value_node).__name__}. "
                    "If this site now wraps in canonical_dict, update the test expectation."
                )

    def test_b3_feature_clusters_present_and_correctly_canonicalized(self) -> None:
        """Every site hashes the same path-independent cluster descriptor."""
        all_sites = self._load_all_sites()
        sites_requiring_descriptor_call = {
            "site1-precompute",
            "site2-train",
            "site5-checkpoint-pin",
            "site7-training-crosscheck",
        }
        sites_requiring_bare_name = {
            "site3-eval",
            "site4-cache-identity",
            "site6-eval-context-cache-clustered",
        }

        for label, kwarg_names, kwarg_values in all_sites:
            assert "feature_clusters" in kwarg_names, (
                f"[{label}] feature_clusters= is missing from compute_selection_hash; "
                "selection artifacts could collide across cluster semantics"
            )
            value_node = kwarg_values["feature_clusters"]
            if label in sites_requiring_descriptor_call:
                assert isinstance(value_node, ast.Call), (
                    f"[{label}] feature_clusters= must derive the canonical descriptor, "
                    f"got {type(value_node).__name__}"
                )
                assert _get_func_name(value_node.func) == "canonical_feature_clusters_descriptor", (
                    f"[{label}] feature_clusters= bypassed "
                    "canonical_feature_clusters_descriptor(...)"
                )
            elif label in sites_requiring_bare_name:
                assert isinstance(value_node, ast.Name), (
                    f"[{label}] feature_clusters= must be an already-canonical bare Name, "
                    f"got {type(value_node).__name__}"
                )

    def test_b4_splice_batch_size_present_and_wired_to_the_right_value(self) -> None:
        all_sites = self._load_all_sites()
        for label, kwarg_names, kwarg_values in all_sites:
            assert "splice_batch_size" in kwarg_names, (
                f"[{label}] splice_batch_size= is missing from compute_selection_hash; "
                "batched producer and consumer artifacts could desynchronize"
            )
            value_node = kwarg_values["splice_batch_size"]
            if label == "site1-precompute":
                assert isinstance(value_node, ast.Name)
                assert value_node.id == "selection_splice_batch_size"
            elif label == "site2-train":
                assert isinstance(value_node, ast.Call)
                assert isinstance(value_node.func, ast.Attribute)
                assert value_node.func.attr == "get"
                assert value_node.args
                assert isinstance(value_node.args[0], ast.Constant)
                assert value_node.args[0].value == "selection_splice_batch_size"
            elif label == "site5-checkpoint-pin":
                # Checkpoint preflight NO LONGER forbids splice: a
                # splice_counterfactual checkpoint with neg_delta_ratio=1.0 is
                # supported, so delta_config can be a real block here and
                # splice_batch_size is no longer definitionally inert -- it
                # genuinely changes the hash. It is therefore threaded as a
                # Name from the caller-supplied override (the producer's batch
                # size is not recorded in checkpoint metadata), matching
                # site1-precompute's shape. The drift sentinel proving it
                # actually reaches the hash lives in
                # tests/steering/test_cluster_checkpoint_pin.py::
                # test_splice_selection_hash_honours_the_producer_batch_size.
                assert isinstance(value_node, ast.Name)
                assert value_node.id == "selection_splice_batch_size"
            elif label == "site6-eval-context-cache-clustered":
                # D-seed (cache_identity.py's clustered branch): clustered
                # eval-context-cache identity forbids delta/splice
                # selection -- feature_delta=null is already enforced
                # earlier in _canonical_hash_from_conditioning whenever
                # feature_clusters is configured, so delta_config=None here
                # is provably the only reachable value and
                # splice_batch_size is definitionally inert. Passed as a
                # literal ast.Constant (not a Name), mirroring site 5's
                # identical precedent, purely to keep site 6 inside the
                # ANTI-DESYNC kwarg-name set (test_b1) -- it does not
                # change the hash (see TestSite6BehavioralParity's
                # seed-dependency sentinel, which proves the value that
                # DOES change the hash -- seed -- is genuinely threaded
                # through).
                assert isinstance(value_node, ast.Constant)
                assert value_node.value is None
            elif label == "site7-training-crosscheck":
                # Codex cross-review (D-seed Finding 3): this cross-check
                # re-derives the same delta-free clustered PROMPT selection
                # site 6 computes, from the top-level training config --
                # delta_config=None is unconditional here too (this
                # function's own prompt_selection construction always
                # nulls delta_weights and never threads a real
                # feature_delta into this specific call), so
                # splice_batch_size is provably inert for the identical
                # reason as sites 5/6. Passed as a literal ast.Constant
                # (not a Name or a cfg.get(...) call), mirroring sites
                # 5/6's precedent, purely to keep site 7 inside the
                # ANTI-DESYNC kwarg-name set (test_b1) -- it does not
                # change the hash.
                assert isinstance(value_node, ast.Constant)
                assert value_node.value is None
            else:
                assert isinstance(value_node, ast.Name)
                assert value_node.id == "splice_batch_size"


# ---------------------------------------------------------------------------
# Behavioral parity for site 5 (Codex combined-review finding 6, MEDIUM)
# ---------------------------------------------------------------------------


class TestSite5BehavioralParity:
    """Drive the REAL site-5 production function (``_resolve_selection_
    identity``) and prove its hash matches a directly-constructed
    ``compute_selection_hash`` call built from the identical conceptual
    inputs — a genuine VALUE-level check beyond Part B's structural
    kwarg-name/wrapper-call AST comparison, which cannot see whether site 5
    actually forwards every input into the hash it computes.
    """

    def _build_fixture(
        self, *, mask_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    ) -> tuple[dict, FeatureMask, ResolvedFeatureClusters]:
        mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
        definition = FeatureClusterDefinition(clusters=((1, 2), (4, 5)))
        clusters = ResolvedFeatureClusters(
            source_path=Path("clusters.yaml"),
            definition=definition,
            descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=8),
        )
        metadata = {
            "user_config": {
                "feature_selection": _POS,
                "negative_selection": _NEG,
                "num_buckets": 2,
                "seed": 17,
                "feature_delta": None,
                "feature_prediction": None,
            }
        }
        return metadata, mask, clusters

    def test_resolve_selection_identity_hash_matches_a_producer_computed_hash(self) -> None:
        metadata, mask, clusters = self._build_fixture()

        identity = _resolve_selection_identity(
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )

        expected = compute_selection_hash(
            positive_selection=_POS,
            negative_selection=_NEG,
            delta_config=None,
            num_buckets=2,
            seed=17,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
            feature_clusters=canonical_feature_clusters_descriptor(clusters),
            # Byte-identical to omitting this kwarg: delta_config=None above
            # makes canonical_splice_selection_batching short-circuit
            # regardless of the value passed here. Included explicitly so
            # this direct call keeps parity with the real site-5 call's full
            # kwarg set (see cluster_checkpoint_pin.py's
            # _resolve_selection_identity).
            splice_batch_size=None,
        )
        assert identity.selection_hash == expected, (
            "site 5 (_resolve_selection_identity) diverges from a directly-"
            "constructed compute_selection_hash call for the identical "
            "conceptual inputs -- a value-level desync Part B's structural "
            "kwarg check alone cannot catch."
        )

    def test_resolve_selection_identity_hash_actually_depends_on_the_mask(self) -> None:
        """Drift sentinel: perturbing one of the 8 canonical inputs (here,
        the mask) must change the hash -- proves site 5 is not silently
        ignoring an input it claims (via the kwarg-name check above) to
        hash.
        """
        metadata, mask, clusters = self._build_fixture()
        other_metadata, other_mask, other_clusters = self._build_fixture(
            mask_ids=(1, 2, 3, 4, 5, 6, 7)
        )

        identity = _resolve_selection_identity(
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )
        other_identity = _resolve_selection_identity(
            metadata=other_metadata,
            feature_mask=other_mask,
            feature_clusters=other_clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )
        assert other_identity.selection_hash != identity.selection_hash


# ---------------------------------------------------------------------------
# Behavioral parity for site 6 (PR #739 Track D, "D-seed")
# ---------------------------------------------------------------------------


class TestSite6BehavioralParity:
    """Drive the REAL site-6 production path (``resolve_feature_selection_
    hash`` -> ``_canonical_hash_from_conditioning``'s clustered branch) and
    prove its hash matches a directly-constructed ``compute_selection_hash``
    call built from the identical conceptual inputs -- mirrors
    ``TestSite5BehavioralParity``, a genuine VALUE-level check beyond Part
    B's structural kwarg-name/wrapper-call AST comparison. PLUS a
    seed-dependency sentinel (perturbing ONLY ``seed``): the §4
    unification-proof that site 6 actually threads ``conditioning.seed``
    into the hash rather than silently keeping the legacy hardcoded
    ``seed=0`` -- the exact verified D-seed bug this PR fixes.
    """

    _DEFINITION = FeatureClusterDefinition(clusters=((1, 2), (4, 5)))

    def _write_cluster_file(self, tmp_path: Path, *, name: str = "clusters.json") -> Path:
        cluster_path = tmp_path / name
        cluster_path.write_text(
            json.dumps({"schema_version": 1, "clusters": [[1, 2], [4, 5]]}),
            encoding="utf-8",
        )
        return cluster_path

    def _build_cond(self, tmp_path: Path, *, seed: int):
        cluster_path = self._write_cluster_file(tmp_path)
        return OmegaConf.create(
            {
                "n_sae_features": 8,
                "feature_selection": _POS,
                "negative_selection": _NEG,
                "num_buckets": 2,
                "stats_source_sha256": "b" * 64,
                "feature_mask": {"mode": "whitelist", "ids": [1, 2, 3, 4, 5, 6]},
                "feature_clusters": {"path": str(cluster_path)},
                "feature_delta": None,
                "seed": seed,
            }
        )

    def test_site6_hash_matches_the_real_packed_row_producer_convention(
        self, tmp_path: Path
    ) -> None:
        """Codex cross-review fix (Finding 1, HIGH): the ORIGINAL version of
        this test built its "expected" comparison payload with
        ``positive_selection=_POS`` -- bare, no ``delta_weights`` key --
        which is the EXACT SAME (buggy) shape the pre-fix clustered branch
        itself produced. That made the test self-referential: it compared
        the branch against a hand-built payload that independently
        reproduced the same bug, so it could never catch a real desync.

        ``compute_selection_hash`` hashes raw key PRESENCE, not the parsed
        value (``steering.packed_batch_io.compute_selection_hash``) --
        ``SamplingSelection`` parsing treats an absent ``delta_weights`` key
        and ``delta_weights=None`` identically
        (``steering/conditional/sampling.py``), but the hash payload does
        not. The REAL packed-row producer's delta-free selection payload
        (``scripts/train/conditional_encdec_train.py``'s
        ``discover_packed_selection``, the splice_cf original/base-
        validation ``base_val_feature_selection_raw`` branch:
        ``dict(feature_selection_raw); [...]["delta_weights"] = None``)
        RETAINS the key with a null value. This test's "expected" payload
        below hard-codes that SAME convention independently (not by
        importing/calling into ``cache_identity.py``'s implementation), so
        a regression that drops the key again produces a genuine hash
        divergence, not a coincidental match.

        Mutation-checked (see task report): reverting
        ``cache_identity.py``'s clustered branch to pass bare ``fs_d``
        (dropping ``delta_weights``) makes this test FAIL; the fixed branch
        (retaining ``delta_weights: None``) makes it PASS. Red before the
        fix, green after -- not self-referential.
        """
        cond = self._build_cond(tmp_path, seed=17)
        identity_cfg = OmegaConf.create({"feature_selection_hash": None})

        actual = resolve_feature_selection_hash(identity_cfg, cond)

        mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
        clusters = ResolvedFeatureClusters(
            source_path=Path("clusters.yaml"),
            definition=self._DEFINITION,
            descriptor=FeatureClusterDescriptor.from_definition(self._DEFINITION, n_sae_features=8),
        )
        # The real packed-row producer's delta-free convention, hard-coded
        # independently of cache_identity.py: retain delta_weights=None,
        # don't drop the key.
        packed_row_producer_positive_selection = {**_POS, "delta_weights": None}
        expected = compute_selection_hash(
            positive_selection=packed_row_producer_positive_selection,
            negative_selection=_NEG,
            delta_config=None,
            num_buckets=2,
            seed=17,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
            feature_clusters=canonical_feature_clusters_descriptor(clusters),
            # Byte-identical to omitting this kwarg: delta_config=None above
            # makes canonical_splice_selection_batching short-circuit
            # regardless of the value passed here. Included explicitly so
            # this direct call keeps parity with the real site-6 call's full
            # kwarg set (see cache_identity.py's
            # _canonical_hash_from_conditioning clustered branch).
            splice_batch_size=None,
        )
        assert actual == expected, (
            "site 6 (_canonical_hash_from_conditioning's clustered branch) "
            "diverges from the packed-row producer's delta-free payload "
            "convention (delta_weights retained as null) for the identical "
            "conceptual inputs -- a value-level desync Part B's structural "
            "kwarg check alone cannot catch. If this fails, the clustered "
            "branch's positive_selection payload no longer matches what "
            "the real packed-row producer (discover_packed_selection's "
            "base_val_feature_selection_raw) computes, breaking the "
            "eval-cache/packed-row identity unification."
        )

    def test_site6_positive_selection_retains_delta_weights_null_not_dropped(
        self, tmp_path: Path
    ) -> None:
        """Narrower, single-purpose regression pin for Finding 1's exact
        root cause: dropping the ``delta_weights`` key entirely (instead of
        retaining it as an explicit ``None``) silently changes the hash,
        because ``compute_selection_hash`` hashes raw dict key presence.
        Drives the real production function twice with payloads that are
        semantically identical after ``parse_feature_selection`` (an absent
        key vs. an explicit ``None`` both parse to "no delta ranking"), and
        asserts they are NOT treated as the same hash input -- i.e., that
        site 6 does not silently normalize away the difference itself
        (it must consistently retain the key, matching the packed-row
        producer, or this whole test module's cross-site parity would be
        moot).
        """
        cond = self._build_cond(tmp_path, seed=23)
        identity_cfg = OmegaConf.create({"feature_selection_hash": None})
        real_hash = resolve_feature_selection_hash(identity_cfg, cond)

        mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
        clusters = ResolvedFeatureClusters(
            source_path=Path("clusters.yaml"),
            definition=self._DEFINITION,
            descriptor=FeatureClusterDescriptor.from_definition(self._DEFINITION, n_sae_features=8),
        )
        dropped_key_hash = compute_selection_hash(
            positive_selection=dict(_POS),
            negative_selection=_NEG,
            delta_config=None,
            num_buckets=2,
            seed=23,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
            feature_clusters=canonical_feature_clusters_descriptor(clusters),
            splice_batch_size=None,
        )
        retained_null_hash = compute_selection_hash(
            positive_selection={**_POS, "delta_weights": None},
            negative_selection=_NEG,
            delta_config=None,
            num_buckets=2,
            seed=23,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
            feature_clusters=canonical_feature_clusters_descriptor(clusters),
            splice_batch_size=None,
        )
        assert dropped_key_hash != retained_null_hash, (
            "sanity check: compute_selection_hash must be sensitive to "
            "delta_weights key presence, or this whole regression pin is "
            "vacuous"
        )
        assert real_hash == retained_null_hash, (
            "site 6 must retain delta_weights=None (matching the packed-row "
            "producer), not drop the key -- got the dropped-key hash "
            "instead, meaning the Finding 1 regression has resurfaced"
        )
        assert real_hash != dropped_key_hash

    def test_site6_hash_actually_depends_on_the_seed(self, tmp_path: Path) -> None:
        """D-seed unification-proof (§4): perturbing ONLY ``seed`` must
        change the hash -- proves site 6 genuinely threads
        ``conditioning.seed`` through rather than silently still hashing at
        the legacy hardcoded ``seed=0`` (the exact bug this PR fixes: before
        this fix every clustered config collapsed onto one universally-wrong
        cache regardless of the configured seed).
        """
        identity_cfg = OmegaConf.create({"feature_selection_hash": None})
        cond_a = self._build_cond(tmp_path, seed=17)
        cond_b = self._build_cond(tmp_path, seed=18)

        hash_a = resolve_feature_selection_hash(identity_cfg, cond_a)
        hash_b = resolve_feature_selection_hash(identity_cfg, cond_b)

        assert hash_a != hash_b, (
            "site 6 must produce a DIFFERENT feature_selection_hash for two "
            "conds that differ ONLY in conditioning.seed -- if this fails, "
            "the clustered branch is silently ignoring the real seed (the "
            "verified D-seed bug: hardcoded seed=0)."
        )
