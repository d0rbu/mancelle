"""Functional tests for ``cluster_artifacts.py`` against REAL current-main
``ClusterFeatureContract`` objects (PR #739 Track C1).

Nothing in the oracle exercises this: it is the primary NEW work for this
port. Rather than hand-typed ``SimpleNamespace``/dict stand-ins, these
tests build genuine contracts via Track B's own already-merged,
filesystem-free (TIER 1) or filesystem-backed (TIER 2) fixture-matrix
generator (``tests/steering/goldens/gen_cluster_contract_fingerprint_
golden.py``) and drive them through ``cluster_artifacts.py``'s own
functions end to end: ``install_or_validate_cluster_root_marker`` (write,
then re-validate on resume), ``validate_cluster_candidate_provenance_
semantics`` (must accept a validly-derived sidecar), and ``load_cluster_
candidate_identity`` / ``clustered_stage0c_fingerprint_fields``.

This proves the thing that actually matters for this PR: does the REAL,
current-main semantic-dict shape (not a hand-typed stand-in) still satisfy
``cluster_artifacts.py``'s strict structural validator, for every fixture
shape the campaign considers load-bearing.

Required cells (verbatim keys from ``TIER1_CELLS``/``TIER2_CELLS``):
  - TIER 1 (pure, filesystem-free ``build_cluster_feature_contract``):
    ``multi_member_mask_shifted_representative``, ``singleton_after_mask``,
    ``inert_cluster``, ``no_cluster_legacy``.
  - TIER 2 ("both effective-vocab regimes", filesystem-backed
    ``resolve_cluster_feature_contract``): ``genuine_cluster_wide_vocab_
    rejected`` (oracle-shape, pre-#759 -- REJECTED by Track B's own
    ``_assert_checkpoint_agreement`` before a fingerprint ever exists to
    hand to this file, so it is proven rejected upstream rather than
    driven through cluster_artifacts.py's own functions -- there is no
    contract to build a sidecar from) and ``genuine_cluster_effective_
    vocab_accepted`` (post-#759, stack-pinned -- the pre-#759 oracle
    cannot produce this value at all -- driven through the full pipeline
    exactly like the TIER 1 cells).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_base.cluster_artifacts import (
    clustered_stage0c_fingerprint_fields,
    install_or_validate_cluster_root_marker,
    load_cluster_candidate_identity,
    validate_cluster_candidate_provenance_semantics,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config
from tests.steering.goldens.gen_cluster_contract_fingerprint_golden import (
    build_tier2_request_stack,
    resolve_tier2_outcome_stack,
    run_tier1_cell_stack,
)

_TIER1_REQUIRED_CELLS = (
    "multi_member_mask_shifted_representative",
    "singleton_after_mask",
    "inert_cluster",
    "no_cluster_legacy",
)


def _cfg_for(run_tag: str) -> Any:
    return make_valid_config(
        run_tag=run_tag,
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def _derive_valid_provenance_and_candidates(
    semantic_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Hand-build a matching, VALID Stage 0a provenance payload + candidate
    rows purely from a REAL contract's own semantic dict -- mirrors what a
    genuinely correct Stage 0a run would have produced for this exact
    contract (every projected representative simply "succeeds").
    """

    clusters = semantic_dict["feature_clusters"]
    concept_source = semantic_dict["concept_source"]
    members_by_rep = dict(clusters["representative_to_allowed_members"])
    source_projection = concept_source["source_to_representative"]

    projecting_by_rep: dict[int, list[int]] = {}
    for source, rep in source_projection:
        projecting_by_rep.setdefault(rep, []).append(source)
    expected_representatives = list(dict.fromkeys(rep for _source, rep in source_projection))
    assert expected_representatives, (
        "fixture cell has no projecting representatives -- required cells must "
        "have a nonempty source_to_representative projection"
    )

    representatives_rows = []
    candidate_rows = []
    for rep in expected_representatives:
        representatives_rows.append(
            {
                "representative_id": rep,
                "allowed_member_ids": list(members_by_rep[rep]),
                "projecting_source_ids": sorted(projecting_by_rep[rep]),
                "statistics_source_id": rep,
                "outcome": "candidate_emitted",
            }
        )
        candidate_rows.append({"feature_index": rep, "concept_id": rep})

    n = len(expected_representatives)
    provenance = {
        "schema_version": 1,
        "feature_contract_fingerprint": None,
        "candidate_concepts_sha256": None,
        "concept_projection_version": semantic_dict["algorithms"]["concept_projection"],
        "counts": {
            "raw_source_ids": len(concept_source["domain"]),
            "mask_allowed_source_ids": len(source_projection),
            "unique_representatives": n,
            "representatives_with_statistics": n,
            "count_survivors": n,
            "description_fetch_survivors": n,
            "described_survivors": n,
        },
        "representatives": representatives_rows,
    }
    return provenance, candidate_rows


def _install_stage0a_artifacts(
    run_tag: str, contract: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write a fully valid Stage 0a provenance + candidate-concepts pair for
    `contract` under `run_tag` (mirrors an honest Stage 0a run). Does NOT
    touch the root feature-contract marker -- callers install that
    separately via ``install_or_validate_cluster_root_marker`` so tests can
    exercise write-then-resume explicitly. Returns the provenance (with
    fingerprint/sha256 filled) and candidate rows as written.
    """

    provenance, candidate_rows = _derive_valid_provenance_and_candidates(contract.semantic_dict())
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json()
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate_rows), encoding="utf-8")
    provenance["feature_contract_fingerprint"] = contract.feature_contract_fingerprint
    provenance["candidate_concepts_sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return provenance, candidate_rows


# ============================================================================
# TIER 1 required cells: multi-member mask-shifted reps, singleton-after-
# mask, inert cluster, no-cluster/legacy.
# ============================================================================


@pytest.mark.parametrize("cell_name", _TIER1_REQUIRED_CELLS)
def test_tier1_cell_contract_satisfies_full_cluster_artifacts_pipeline(
    cell_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REAL, current-main ClusterFeatureContract for this cell -- not a
    hand-typed stand-in -- must satisfy install_or_validate_cluster_root_
    marker (write, then re-validate-on-resume), validate_cluster_candidate_
    provenance_semantics (must accept), and load_cluster_candidate_identity
    / clustered_stage0c_fingerprint_fields end to end.
    """

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    outcome = run_tier1_cell_stack(cell_name, tmp_path / "fixture")
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None
    run_tag = f"tier1-{cell_name}"
    cfg = _cfg_for(run_tag)

    install_or_validate_cluster_root_marker(cfg, contract)
    install_or_validate_cluster_root_marker(cfg, contract)  # resume: must accept, not re-raise

    _install_stage0a_artifacts(run_tag, contract)

    identity = load_cluster_candidate_identity(
        cfg, expected_fingerprint=contract.feature_contract_fingerprint
    )
    assert identity is not None
    assert identity.feature_contract_fingerprint == contract.feature_contract_fingerprint
    assert (
        identity.candidate_concepts_sha256
        == hashlib.sha256(
            test_paths.default_paths(run_tag).candidate_concepts_json().read_bytes()
        ).hexdigest()
    )

    fields = clustered_stage0c_fingerprint_fields(cfg)
    assert fields == {
        "feature_contract_fingerprint": identity.feature_contract_fingerprint,
        "candidate_cluster_provenance_sha256": identity.candidate_provenance_sha256,
        "cluster_concept_projection_version": identity.concept_projection_version,
    }


@pytest.mark.parametrize("cell_name", _TIER1_REQUIRED_CELLS)
def test_tier1_cell_tampered_member_row_is_rejected(
    cell_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative functional case per cell family: a genuinely-built
    contract's sidecar is seeded validly, then ONE representative_to_
    allowed_members row is mutated after the fact -- the validator must
    still reject the tampered sidecar even though it was seeded from a
    genuinely-built contract.
    """

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    outcome = run_tier1_cell_stack(cell_name, tmp_path / "fixture")
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None
    run_tag = f"tier1-tamper-{cell_name}"
    cfg = _cfg_for(run_tag)
    install_or_validate_cluster_root_marker(cfg, contract)
    provenance, candidate_rows = _install_stage0a_artifacts(run_tag, contract)

    # Sanity: the untampered payload is valid before we mutate it.
    validate_cluster_candidate_provenance_semantics(
        feature_contract=contract.document(),
        candidate_rows=candidate_rows,
        provenance=provenance,
    )

    tampered_provenance = copy.deepcopy(provenance)
    first_row = tampered_provenance["representatives"][0]
    first_row["allowed_member_ids"] = [*first_row["allowed_member_ids"], 999999]

    with pytest.raises(AxBenchBaseError):
        validate_cluster_candidate_provenance_semantics(
            feature_contract=contract.document(),
            candidate_rows=candidate_rows,
            provenance=tampered_provenance,
        )


# ============================================================================
# TIER 2 required cells: both effective-vocab regimes.
# ============================================================================


def test_tier2_genuine_cluster_wide_vocab_is_rejected_upstream_before_reaching_c1(
    tmp_path: Path,
) -> None:
    """Both effective-vocab regimes, half 1 of 2: the pre-#759 wide (mask-
    only) vocab under GENUINE clustering is rejected by Track B's own
    resolve_cluster_feature_contract (_assert_checkpoint_agreement) --
    confirming this scenario never produces a fingerprint for
    cluster_artifacts.py to consume in the first place. Not driven through
    cluster_artifacts.py's own functions: there is no contract to build a
    sidecar from -- this is the negative-space half of the pair.
    """

    request, sae_loader = build_tier2_request_stack("genuine_cluster_wide_vocab_rejected", tmp_path)
    outcome = resolve_tier2_outcome_stack(request, sae_loader)
    assert outcome.raises
    assert outcome.contract is None
    assert outcome.error_message is not None
    assert (
        "effective-vocabulary training" in outcome.error_message or "#759" in outcome.error_message
    )


def test_tier2_genuine_cluster_effective_vocab_satisfies_full_cluster_artifacts_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both effective-vocab regimes, half 2 of 2: the post-#759 narrow
    (effective_feature_vocab) shape -- the value the pre-#759 oracle cannot
    even produce -- succeeds, and its REAL semantic-dict shape must satisfy
    cluster_artifacts.py's strict structural validator end to end, same as
    the TIER 1 cells above.
    """

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True)
    request, sae_loader = build_tier2_request_stack(
        "genuine_cluster_effective_vocab_accepted", fixture_root
    )
    outcome = resolve_tier2_outcome_stack(request, sae_loader)
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None
    run_tag = "tier2-effective-vocab-accepted"
    cfg = _cfg_for(run_tag)

    install_or_validate_cluster_root_marker(cfg, contract)
    install_or_validate_cluster_root_marker(cfg, contract)  # resume: must accept, not re-raise

    _install_stage0a_artifacts(run_tag, contract)

    identity = load_cluster_candidate_identity(
        cfg, expected_fingerprint=contract.feature_contract_fingerprint
    )
    assert identity is not None
    assert identity.feature_contract_fingerprint == contract.feature_contract_fingerprint


def test_tier2_genuine_cluster_effective_vocab_tampered_projection_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative functional case for the tier-2 accepted regime: mutate one
    projecting_source_ids row after seeding a valid sidecar.
    """

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True)
    request, sae_loader = build_tier2_request_stack(
        "genuine_cluster_effective_vocab_accepted", fixture_root
    )
    outcome = resolve_tier2_outcome_stack(request, sae_loader)
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None
    run_tag = "tier2-effective-vocab-tamper"
    cfg = _cfg_for(run_tag)
    install_or_validate_cluster_root_marker(cfg, contract)
    provenance, candidate_rows = _install_stage0a_artifacts(run_tag, contract)

    validate_cluster_candidate_provenance_semantics(
        feature_contract=contract.document(),
        candidate_rows=candidate_rows,
        provenance=provenance,
    )

    tampered_provenance = copy.deepcopy(provenance)
    first_row = tampered_provenance["representatives"][0]
    first_row["projecting_source_ids"] = []

    with pytest.raises(AxBenchBaseError):
        validate_cluster_candidate_provenance_semantics(
            feature_contract=contract.document(),
            candidate_rows=candidate_rows,
            provenance=tampered_provenance,
        )
