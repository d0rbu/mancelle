"""#1249: loading committed fair feature-DiffMean bags as encdec fit conditioning.

The fit conditions on the SAME committed artifact the eval pipeline steers with
-- not a reconstruction. These tests round-trip through the real writer
(`write_ours_gcg_bag`) so what is pinned is the actual on-disk contract:
fingerprint validation, identity agreement between the store layout and the
artifact's own metadata, and hard failure on every path that would silently
reintroduce an unconditioned fit (missing bag, ambiguous grid, empty bag).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb_adv import fit_conditioning as fc
from steering.eval.axbench_bpb_adv.fit_conditioning import (
    load_encdec_fit_conditioning,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    write_ours_gcg_bag,
)

_FP = "f" * 64


def _write_bag(
    store: Path,
    *,
    mode: str,
    concept_id: int,
    entries: tuple[tuple[int, int], ...],
    fingerprint: str | None = _FP,
    lambda_name: str = "lambda_0.0.safetensors",
    metadata_overrides: dict[str, str] | None = None,
) -> Path:
    path = store / "feature_diffmean" / mode / f"c{concept_id}" / lambda_name
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "concept_id": str(concept_id),
        "mode": mode,
        "mode_tag": "feature_diffmean",
        **(metadata_overrides or {}),
    }
    if fingerprint is not None:
        metadata[FEATURE_CONTRACT_FINGERPRINT_META_KEY] = fingerprint
    write_ours_gcg_bag(
        path,
        FeatureConditioning(entries=entries),
        metadata=metadata,
        vocab_fingerprint="uncompacted",
    )
    return path


def test_round_trip_loads_the_exact_committed_entries(tmp_path: Path) -> None:
    _write_bag(tmp_path, mode="positive", concept_id=0, entries=((4, 1), (9, 0)))
    _write_bag(tmp_path, mode="positive", concept_id=3, entries=((2, 1),))
    out = load_encdec_fit_conditioning(
        tmp_path,
        mode="positive",
        concept_ids=[0, 3],
        expected_feature_contract_fingerprint=_FP,
    )
    assert set(out) == {0, 3}
    assert out[0].entries == ((4, 1), (9, 0))
    assert out[3].entries == ((2, 1),)


def test_missing_store_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="no fair feature-DiffMean bag"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[0],
            expected_feature_contract_fingerprint=_FP,
        )


def test_missing_concept_bag_raises_not_skips(tmp_path: Path) -> None:
    """A missing bag must fail the FIT, not shrink the concept set."""
    _write_bag(tmp_path, mode="positive", concept_id=0, entries=((1, 1),))
    with pytest.raises(AxBenchBaseError, match="0 bag file"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[0, 7],
            expected_feature_contract_fingerprint=_FP,
        )


def test_multi_lambda_grid_is_ambiguous_and_raises(tmp_path: Path) -> None:
    _write_bag(tmp_path, mode="positive", concept_id=0, entries=((1, 1),))
    _write_bag(
        tmp_path, mode="positive", concept_id=0, entries=((1, 1),),
        lambda_name="lambda_0.5.safetensors",
    )
    with pytest.raises(AxBenchBaseError, match="2 bag file"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[0],
            expected_feature_contract_fingerprint=_FP,
        )


def test_identity_mismatch_between_layout_and_artifact_raises(tmp_path: Path) -> None:
    """A bag whose own metadata disagrees with its directory is corruption."""
    path = _write_bag(
        tmp_path, mode="positive", concept_id=5, entries=((1, 1),),
        metadata_overrides={"concept_id": "6"},
    )
    assert path.is_file()
    with pytest.raises(AxBenchBaseError, match="concept_id='6'"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[5],
            expected_feature_contract_fingerprint=_FP,
        )


def test_fingerprint_mismatch_propagates(tmp_path: Path) -> None:
    """The contract-fingerprint rule of the underlying reader must apply."""
    _write_bag(tmp_path, mode="positive", concept_id=0, entries=((1, 1),))
    with pytest.raises(Exception, match="fingerprint"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[0],
            expected_feature_contract_fingerprint="a" * 64,
        )


def test_empty_bag_raises_as_the_removed_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty committed bag must not become an unconditioned fit.

    The writer refuses empty bags, so this state means a corrupt or
    hand-assembled artifact -- simulated by stubbing the reader.
    """
    _write_bag(tmp_path, mode="positive", concept_id=0, entries=((1, 1),))
    monkeypatch.setattr(
        fc, "read_ours_gcg_bag", lambda *a, **k: ({}, {
            "concept_id": "0", "mode": "positive", "mode_tag": "feature_diffmean",
        })
    )
    with pytest.raises(AxBenchBaseError, match="no\\s+entries"):
        load_encdec_fit_conditioning(
            tmp_path,
            mode="positive",
            concept_ids=[0],
            expected_feature_contract_fingerprint=_FP,
        )


def test_negative_mode_resolves_its_own_directory(tmp_path: Path) -> None:
    _write_bag(tmp_path, mode="negative", concept_id=1, entries=((6, 0),))
    out = load_encdec_fit_conditioning(
        tmp_path,
        mode="negative",
        concept_ids=[1],
        expected_feature_contract_fingerprint=_FP,
    )
    assert out[1].entries == ((6, 0),)
