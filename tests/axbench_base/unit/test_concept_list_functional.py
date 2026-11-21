"""Functional tests for ``concept_list.py``'s clustered branch against REAL
current-main ``ClusterFeatureContract`` objects (PR #739 Track C2).

Mirrors ``tests/axbench_base/unit/test_cluster_artifacts_functional.py``'s
own rationale (Track C1): the unit tier's ``_cluster_contract_for_concept_
list`` ``SimpleNamespace`` fixture proves the WIRING (dispatch, projection
arithmetic, underfill/raise sites) fast and in isolation, but says nothing
about whether the REAL, current-main ``ClusterFeatureContract`` semantic-dict
shape (Track B's own fixture-matrix generator, not a hand-typed stand-in)
still satisfies ``_prepare_clustered_concept_list``'s own assumptions end to
end -- including the provenance sidecar's round-trip through C1's real
``load_cluster_candidate_identity``. That is what this file proves.

Required cells (verbatim keys from ``TIER1_CELLS``/``TIER2_CELLS``, per the
Track C Part 2 plan's Test Design section):
  - TIER 1 (pure, filesystem-free ``build_cluster_feature_contract``):
    ``multi_member_mask_shifted_representative``, ``singleton_after_mask``,
    ``inert_cluster``, ``no_cluster_legacy``.
  - TIER 2: ``genuine_cluster_effective_vocab_accepted`` (reachable, driven
    through the full pipeline exactly like the TIER 1 cells) and
    ``genuine_cluster_wide_vocab_rejected`` (precondition-pin only: proven
    to raise at contract-build, before a fingerprint even exists to hand to
    ``prepare_concept_list`` -- see the dedicated test below for why
    concept_list.py itself needs no vocab-regime branch of its own).

Named deviation (TIER1 provenance augmentation, evidenced): empirically
confirmed (see task-c2-report.md's Main-Drift Check) that
``build_cluster_feature_contract`` (TIER1's pure constructor) never
populates ``provenance_dict()["feature_statistics_path"]`` -- only
``resolve_cluster_feature_contract`` (the filesystem-backed TIER2 path)
adds it, via exactly the pattern
``replace(contract, provenance_json=_canonical_json({**contract.
provenance_dict(), "feature_statistics_path": ...}))`` (cluster_feature_
contract.py:752-771). ``_prepare_clustered_concept_list`` requires this key
(indexed directly, no ``.get``) because a TIER1 contract was never resolved
against any real statistics file -- structurally, production code NEVER
hands ``prepare_concept_list`` a bare TIER1-shaped contract either
(``cluster_contract`` is only ever produced by ``resolve_cluster_feature_
contract(_ddp)``, E2's territory). ``_with_statistics_path`` below performs
the IDENTICAL augmentation production code performs when promoting a bare
contract to a filesystem-backed one, so every field this file's assertions
actually exercise (representative_universe, source_to_representative,
allowed_members, projecting_sources, fingerprint, semantic_dict) remains
100% genuine and unmodified -- only the one filesystem-linking provenance
field TIER1 structurally cannot produce is supplied.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.cluster_artifacts import (
    install_or_validate_cluster_root_marker,
    load_cluster_candidate_identity,
)
from steering.eval.axbench_base.concept_list import prepare_concept_list
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base.unit.test_concept_list import (
    _make_per_feature_stat,
    _make_stats,
    stub_env,  # noqa: F401 -- imported as a pytest fixture, not called directly
)
from tests.axbench_base.unit.test_config import make_valid_config
from tests.steering.goldens.gen_cluster_contract_fingerprint_golden import (
    build_tier2_request_stack,
    resolve_tier2_outcome_stack,
    run_tier1_cell_stack,
)

_TIER1_REACHABLE_CELLS = (
    "multi_member_mask_shifted_representative",
    "singleton_after_mask",
    "inert_cluster",
    "no_cluster_legacy",
)


def _with_statistics_path(contract: Any, statistics_path: Path) -> Any:
    """Augment a TIER1 contract's provenance with ``feature_statistics_
    path`` -- see the module docstring's "Named deviation" section for why
    this is required and why it does not touch any semantic/fingerprinted
    field. Mirrors ``resolve_cluster_feature_contract``'s own ``provenance.
    update({...}); replace(contract, provenance_json=...)`` pattern
    (cluster_feature_contract.py:752-771)."""
    provenance = dict(contract.provenance_dict())
    provenance["feature_statistics_path"] = str(statistics_path)
    return dataclasses.replace(
        contract,
        provenance_json=json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )


def _unique_representatives(contract: Any) -> tuple[int, ...]:
    """Read ``_prepare_clustered_concept_list``'s own mask->rep_of->dedup
    projection off the LIVE contract -- not hand-typed -- exactly mirroring
    the production computation (concept_list.py's ``source_projection``/
    ``mask_allowed_sources``/``unique_representatives`` sequence)."""
    source_projection = dict(contract.source_to_representative)
    mask_allowed_sources = tuple(
        source_id for source_id in contract.concept_source_domain if source_id in source_projection
    )
    return tuple(dict.fromkeys(source_projection[s] for s in mask_allowed_sources))


def _cfg_for(stub_env, run_tag: str, feature_statistics_path: Path) -> Any:  # noqa: F811
    return make_valid_config(
        run_tag=run_tag,
        output_root=str(stub_env["tmp_path"] / "eval_out"),
        feature_statistics_path=str(feature_statistics_path),
        max_concepts=1,
        min_corpus_count=1,
        min_description_length=1,
        concept_list_seed=42,
        neuronpedia_model_id="gemma-2-2b",
        neuronpedia_source="8-gemmascope-res-16k",
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def _seed_statistics(stub_env, contract: Any, unique_representatives: tuple[int, ...]) -> None:  # noqa: F811
    """Point stub_env's stats/hash doubles at the cell's REAL representative
    universe and the contract's REAL statistics identity (width + sha256),
    so ``_prepare_clustered_concept_list``'s own preflight-agreement checks
    pass against genuine contract fields, not hand-typed constants. Sets
    BOTH the pre-load path-digest holder (``statistics_hash_digests``, the
    ``stream_hash_file_stable`` "before" check) and the fused-read content
    holder (``statistics_content_sha256_holder``, review-fix: the mock
    counterpart of ``_load_feature_statistics``'s own fused hash+read
    check) to the SAME real value -- both must agree for the happy path,
    exactly mirroring what a genuine, unmutated statistics file satisfies
    in production.
    """
    stub_env["stats_holder"]["stats"] = _make_stats(
        {str(rep): _make_per_feature_stat(count=50) for rep in unique_representatives},
        n_sae_features=contract.n_sae_features,
    )
    stub_env["statistics_hash_digests"][:] = [contract.statistics.file_sha256]
    stub_env["statistics_content_sha256_holder"]["sha256"] = contract.statistics.file_sha256


def _assert_self_consistent_and_round_trips(
    stub_env,  # noqa: F811
    cfg: Any,
    contract: Any,
    unique_representatives: tuple[int, ...],
) -> Path:
    """Shared assertion body for every reachable cell: candidate rows match
    the real projection, the provenance sidecar's per-representative rows
    match the real contract fields, exact resume is byte-identical, the
    pair round-trips through C1's real ``load_cluster_candidate_identity``,
    and a mutated sidecar row is rejected on reload. Returns the candidate
    path."""
    out = prepare_concept_list(cfg, cluster_contract=contract)
    rows = json.loads(out.read_text())
    assert {row["feature_index"] for row in rows} == set(unique_representatives)
    for row in rows:
        assert row["corpus_count"] == 50
        assert row["layer"] == contract.feature_directory.layer

    provenance_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).candidate_concepts_cluster_provenance_json()
    provenance = json.loads(provenance_path.read_text())
    by_rep = {r["representative_id"]: r for r in provenance["representatives"]}
    members_by_rep = dict(contract.representative_to_allowed_members)
    projecting_by_rep = dict(contract.representative_to_projecting_sources)
    for rep in unique_representatives:
        assert by_rep[rep]["allowed_member_ids"] == list(members_by_rep[rep])
        assert by_rep[rep]["projecting_source_ids"] == list(projecting_by_rep[rep])
        assert by_rep[rep]["statistics_source_id"] == rep
        assert by_rep[rep]["outcome"] == "candidate_emitted"
    assert provenance["feature_contract_fingerprint"] == contract.feature_contract_fingerprint
    assert provenance["candidate_concepts_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()

    # Exact resume: byte-identical, must not re-fetch or re-raise.
    candidate_before = out.read_bytes()
    provenance_before = provenance_path.read_bytes()
    out_resumed = prepare_concept_list(cfg, cluster_contract=contract)
    assert out_resumed == out
    assert out_resumed.read_bytes() == candidate_before
    assert provenance_path.read_bytes() == provenance_before

    # Round-trips cleanly through C1's real load_cluster_candidate_identity.
    identity = load_cluster_candidate_identity(
        cfg, expected_fingerprint=contract.feature_contract_fingerprint
    )
    assert identity is not None
    assert identity.feature_contract_fingerprint == contract.feature_contract_fingerprint
    assert identity.candidate_concepts_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()

    # A mutated sidecar row IS rejected on reload.
    tampered = copy.deepcopy(provenance)
    first_rep = unique_representatives[0]
    tampered_row = next(
        r for r in tampered["representatives"] if r["representative_id"] == first_rep
    )
    tampered_row["allowed_member_ids"] = [*tampered_row["allowed_member_ids"], 999_999]
    provenance_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AxBenchBaseError):
        load_cluster_candidate_identity(
            cfg, expected_fingerprint=contract.feature_contract_fingerprint
        )

    return out


@pytest.mark.parametrize("cell_name", _TIER1_REACHABLE_CELLS)
def test_tier1_cell_full_pipeline_self_consistent(
    cell_name: str,
    stub_env,  # noqa: F811 -- imported fixture, not a redefinition
) -> None:
    fixture_root = stub_env["tmp_path"] / "fixture" / cell_name
    outcome = run_tier1_cell_stack(cell_name, fixture_root)
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None

    statistics_path = stub_env["tmp_path"] / f"{cell_name}_statistics.safetensors"
    statistics_path.write_bytes(b"not-a-real-safetensors-file-stats-io-is-mocked")
    # Pre-resolved once so it is byte-identical on both sides of
    # _prepare_clustered_concept_list's asymmetric path check (only the
    # cfg.feature_statistics_path side gets .expanduser().resolve() there;
    # the provenance side is compared as-is -- see concept_list.py:101-103).
    statistics_path = statistics_path.expanduser().resolve()
    contract = _with_statistics_path(contract, statistics_path)

    unique_representatives = _unique_representatives(contract)
    assert unique_representatives, f"{cell_name} unexpectedly has no projecting representatives"

    cfg = _cfg_for(stub_env, f"tier1-functional-{cell_name}", statistics_path)
    _seed_statistics(stub_env, contract, unique_representatives)
    install_or_validate_cluster_root_marker(cfg, contract)

    _assert_self_consistent_and_round_trips(stub_env, cfg, contract, unique_representatives)


def test_tier2_genuine_cluster_effective_vocab_accepted_full_pipeline_self_consistent(
    stub_env,  # noqa: F811 -- imported fixture, not a redefinition
) -> None:
    fixture_root = stub_env["tmp_path"] / "fixture" / "tier2_accepted"
    fixture_root.mkdir(parents=True)
    request, sae_loader = build_tier2_request_stack(
        "genuine_cluster_effective_vocab_accepted", fixture_root
    )
    outcome = resolve_tier2_outcome_stack(request, sae_loader)
    assert not outcome.raises, outcome.error_message
    contract = outcome.contract
    assert contract is not None
    # TIER2 contracts already carry a real feature_statistics_path from the
    # filesystem-backed preflight -- no augmentation needed (unlike TIER1).
    # Read it back off the contract's own provenance (already fully
    # resolved by resolve_cluster_feature_contract internally, strict=True)
    # rather than re-deriving from request.feature_statistics_path, so this
    # test cannot silently pass on a coincidental pre-resolve equality.
    statistics_path = Path(contract.provenance_dict()["feature_statistics_path"])
    assert statistics_path.is_file()

    unique_representatives = _unique_representatives(contract)
    assert unique_representatives

    cfg = _cfg_for(stub_env, "tier2-functional-accepted", statistics_path)
    _seed_statistics(stub_env, contract, unique_representatives)
    install_or_validate_cluster_root_marker(cfg, contract)

    _assert_self_consistent_and_round_trips(stub_env, cfg, contract, unique_representatives)


def test_tier2_genuine_cluster_wide_vocab_rejected_precondition_pin(
    stub_env,  # noqa: F811
) -> None:
    """Precondition-pin only (per the plan's Test Design section): the
    pre-#759 wide (mask-only) vocabulary under GENUINE clustering is
    rejected by Track B's own ``resolve_cluster_feature_contract``
    (``_assert_checkpoint_agreement``) before a fingerprint ever exists to
    hand to ``prepare_concept_list``. Documents why concept_list.py itself
    needs no vocab-regime branch: the N/A audit in task-c2-report.md's
    Main-Drift Check table (concept_list.py only reads ``contract.n_sae_
    features: int`` + resolved projection/statistics fields, never
    recomputes a vocabulary) holds because a wide-vocab genuine-cluster
    contract can never even be constructed -- the vocab regime is closed
    one layer up, by construction, confirmed here rather than merely
    asserted."""
    fixture_root = stub_env["tmp_path"] / "fixture" / "tier2_rejected"
    fixture_root.mkdir(parents=True)
    request, sae_loader = build_tier2_request_stack(
        "genuine_cluster_wide_vocab_rejected", fixture_root
    )
    outcome = resolve_tier2_outcome_stack(request, sae_loader)
    assert outcome.raises
    assert outcome.contract is None
    assert outcome.error_message is not None
    assert (
        "effective-vocabulary training" in outcome.error_message or "#759" in outcome.error_message
    )
