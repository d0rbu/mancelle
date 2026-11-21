"""Vocab-restriction teeth for the PLAIN BPB feature-DiffMean derive (issue #695 PR-1).

A compacted encoder-decoder checkpoint holds encoder embedding rows only for a
subset ``V`` of the full ``n_sae`` SAE features; conditioning on an id outside
``V`` trips ``FeatureEncoder``'s OOV assert at score time. The new
``conditions_within_vocab`` stance PERMITS the plain eval to load a compacted
checkpoint, so the plain derive's ``feature_diffmean_bag`` selection MUST restrict
its top-/bottom-``k`` ranking to ``V`` (mirroring the adv fair/roofline derives)
rather than ranking over the full ``n_sae`` and writing out-of-``V`` bags.

These tests drive the REAL ``feature_diffmean_bag`` selection (only the
model-dependent ``_max_pooled_sae_activations`` seam is stubbed, exactly as the
adv unit suite does) end-to-end through ``build_feature_diffmean_bags_plain`` and
read the WRITTEN bag back, so they pin the actual regression site:

* Compacted (``feature_vocab`` non-``None``): every written bag id is in ``V`` and
  the out-of-``V`` global winner is EXCLUDED in favour of the in-``V`` runner-up.
* Uncompacted (``feature_vocab=None``): ``allowed_ids`` threads through as ``None``,
  so the written bag is byte-for-byte the pre-restriction (full-``n_sae``)
  selection -- the legacy no-op contract.

Construction (shared with the adv unit suite's threading trap): with a linear
percentile table (percentile ~= magnitude), rich/avoid magnitudes give
``diff ~= [+9.5, +39.5, -50, +89.5]``. Feature 3 is the global top-1 by ``+diff``
but sits OUTSIDE ``V={0,1,2}``; feature 1 is the top-1 WITHIN ``V``; feature 2 is
the bottom-1 either way. So the unrestricted bag is ``{3:+, 2:-}`` and the
restricted bag is ``{1:+, 2:-}`` -- the two genuinely diverge, which is what makes
these non-vacuous mutation traps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.eval.axbench_bpb.feature_diffmean_bags as module
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.config import FeatureDiffMeanConfig
from steering.eval.axbench_bpb.feature_diffmean_bags import (
    build_feature_diffmean_bags_plain,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv import feature_diffmean
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag

# n_sae = 4; feature 3 is deliberately the out-of-V global top-1 by +diff.
_N_SAE = 4
_ALLOWED_IDS = [0, 1, 2]  # V excludes feature 3
_CONCEPT_ID = 7
_RICH_COMPLETION = "rich completion"
_AVOID_COMPLETION = "avoid completion"
# Reused from the adv unit threading trap (tests/axbench_bpb_adv/unit/
# test_feature_diffmean.py::test_feature_diffmean_bag_threads_allowed_ids_into_selection):
# with the linear percentile table these magnitudes make feature 3 the global
# top-1 (+diff ~= 89.5) and feature 1 the top-1 within V (+diff ~= 39.5).
_ACTS_BY_COMPLETION: dict[str, list[float]] = {
    _RICH_COMPLETION: [10.0, 40.0, 0.0, 90.0],
    _AVOID_COMPLETION: [0.0, 0.0, 50.0, 0.0],
}

# The two selections these inputs produce, by construction.
_RESTRICTED_BAG = {1: 0, 2: 1}  # in-V top-1 (feature 1) + bottom-1 (feature 2)
_UNRESTRICTED_BAG = {3: 0, 2: 1}  # out-of-V global top-1 (feature 3) + feature 2


def _linear_percentile_table(n_features: int) -> t.Tensor:
    """``[n_features, 99]`` table whose every row is ``[1, 2, ..., 99]`` (percentile ~= magnitude)."""
    row = t.arange(1, 100, dtype=t.float64)
    return row.unsqueeze(0).repeat(n_features, 1)


def _write_source(tmp_path: Any) -> SourcePaths:
    """Write a one-concept positive synth parquet with both content classes.

    The completion (``output_text``) is the key the stubbed activation pool reads,
    so each side's per-feature magnitudes are known by construction.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [
        {
            "concept_id": _CONCEPT_ID,
            "content_class": "concept_rich",
            "prefix_text": "rich prefix",
            "output_text": _RICH_COMPLETION,
            "qc_accepted": True,
        },
        {
            "concept_id": _CONCEPT_ID,
            "content_class": "concept_avoiding",
            "prefix_text": "avoid prefix",
            "output_text": _AVOID_COMPLETION,
            "qc_accepted": True,
        },
    ]
    schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            pa.Table.from_pylist(rows).schema,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        source_dir / "synth_training_positive.parquet",
    )
    return SourcePaths(source_dir)


def _stub_pooled_activations(
    items: Any,
    *,
    model: Any,
    sae: Any,
    sae_layer: Any,
    device: Any,
    extraction_batch_size: Any = None,
) -> t.Tensor:
    """``_max_pooled_sae_activations`` seam: inject known per-completion magnitudes.

    This is the exact seam the adv unit suite stubs -- the REAL percentile mapping
    and REAL top-/bottom-``k`` (V-restricted) selection run under test; only the
    base-model + SAE forward is bypassed.
    """
    del model, sae, sae_layer, device, extraction_batch_size
    return t.tensor(
        [_ACTS_BY_COMPLETION[item.completion_text] for item in items],
        dtype=t.float32,
    )


def _apply_common_monkeypatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "assert_sae_matches_encdec", lambda cfg, model: None)
    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda model, cfg: (1, 1))
    monkeypatch.setattr(
        module,
        "parse_neuronpedia_source",
        lambda source: (None, None, 8, None, None),
    )
    monkeypatch.setattr(
        module,
        "build_percentile_table_from_stats",
        lambda path, n_sae: _linear_percentile_table(n_sae),
    )
    # NOTE: feature_diffmean_bag is intentionally NOT patched -- the real selection
    # runs; only its model-forward seam below is stubbed.
    monkeypatch.setattr(
        feature_diffmean, "_max_pooled_sae_activations", _stub_pooled_activations
    )


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        feature_statistics_path="/unused/stats.safetensors",
        filter_unaccepted_samples=True,
        feature_diffmean=FeatureDiffMeanConfig(top_k_pos=1, top_k_neg=1),
        neuronpedia_source="8-gemmascope-res-16k",
    )


def _compacted_model() -> SimpleNamespace:
    """A duck-typed COMPACTED encdec model (pattern from the adv compacted tests).

    ``feature_vocab`` non-``None`` -> compacted; ``encoder.feature_vocab_ids`` is the
    OWN allowed-id buffer ``feature_vocab_allowed_ids`` reads; ``fingerprint()`` is
    stamped into the written bag by ``checkpoint_vocab_fingerprint``.
    """
    return SimpleNamespace(
        encoder=SimpleNamespace(
            config=SimpleNamespace(n_sae_features=_N_SAE),
            feature_vocab_ids=t.tensor(_ALLOWED_IDS, dtype=t.int64),
        ),
        feature_vocab=SimpleNamespace(fingerprint=lambda: "fp-plain-compacted"),
    )


def _uncompacted_model() -> SimpleNamespace:
    """A duck-typed UNCOMPACTED (legacy) model: ``feature_vocab=None`` -> allowed_ids None."""
    return SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=_N_SAE)),
        feature_vocab=None,
    )


def _run_plain_derive(
    *, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, model: SimpleNamespace
) -> dict[int, int]:
    _apply_common_monkeypatches(monkeypatch)
    source_paths = _write_source(tmp_path)
    output_paths = OutputPaths(output_root=tmp_path / "out", run_tag="run")
    build_feature_diffmean_bags_plain(
        cfg=_cfg(),
        output_paths=output_paths,
        source_paths=source_paths,
        methods=[SimpleNamespace(name="feature_diffmean_encdec")],
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        device="cpu",
    )
    bag, _metadata = read_ours_gcg_bag(
        output_paths.ours_gcg_bag(
            _CONCEPT_ID,
            SteeringMode.POSITIVE,
            0.0,
            "feature_diffmean",
        )
    )
    return bag


def test_plain_feature_diffmean_compacted_bag_ids_within_vocab(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """COMPACTED model: every written bag id is in V; the out-of-V global winner is excluded.

    Failing-first / mutation trap: without ``allowed_ids=`` threaded into the
    ``feature_diffmean_bag`` call the derive ranks over the full n_sae and writes
    ``{3:0, 2:1}`` -- feature 3 is OUTSIDE V, so ``3 not in bag`` REDs (and score
    time would raise on the out-of-V id). With the restriction the derive picks
    the in-V runner-up (feature 1) instead: ``{1:0, 2:1}``.
    """
    bag = _run_plain_derive(
        tmp_path=tmp_path, monkeypatch=monkeypatch, model=_compacted_model()
    )
    # (a) EVERY written bag id is within V.
    assert set(bag) <= set(_ALLOWED_IDS), (
        f"written bag {bag} contains ids outside V={_ALLOWED_IDS}"
    )
    # (b) the out-of-V global top-1 (feature 3) is EXCLUDED (non-vacuous: it would
    #     have won the unrestricted slot -- see the uncompacted test below).
    assert 3 not in bag
    # (c) the exact V-restricted selection: in-V top-1 (feature 1) + bottom-1 (feature 2).
    assert bag == _RESTRICTED_BAG
    assert bag != _UNRESTRICTED_BAG


def test_plain_feature_diffmean_uncompacted_selection_is_unchanged(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UNCOMPACTED model (``feature_vocab=None``): the derive is byte-identical to before.

    ``feature_vocab_allowed_ids`` returns ``None`` on a legacy checkpoint, so the
    selection is the unrestricted full-n_sae ranking -- feature 3 (the global
    top-1) IS selected here, which is exactly the pre-fix behaviour the compacted
    path used to (wrongly) inherit. This pins the legacy no-op contract AND is the
    sanity proof that the construction genuinely diverges.
    """
    bag = _run_plain_derive(
        tmp_path=tmp_path, monkeypatch=monkeypatch, model=_uncompacted_model()
    )
    assert bag == _UNRESTRICTED_BAG  # {3: 0, 2: 1} -- feature 3 present, unrestricted
    assert 3 in bag
