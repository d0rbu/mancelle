"""Unit tests for Task 7: GCG bag loading via build_intervention.

Covers all required assertions from the task-7 brief:
  (a) A written fair bag ``{42: 0}`` is loaded and returned as
      ``Intervention.ours(feature_conditions={42: 0})``.
  (b) The negative-mode method loads the negative-mode bag (not the positive one).
  (c) A roofline method loads the per-prefix-specific bag; two distinct prefix_ids
      each load their own bag independently.
  (d) A missing bag RAISES ``AxBenchBpbError`` (Task-10 policy). The score stage's
      prefix coverage guard scores ONLY the prefixes the derive wrote a bag for,
      so once a row reaches ``build_intervention`` its bag must exist; an absent
      bag is a genuine pipeline bug (a stale / never-run derive), not graceful
      degradation.
  (e) A corrupt / unreadable artifact raises ``AxBenchBpbError``.
  Extra: calling ``build_intervention`` for a GCG method without ``adv_paths``
  raises ``AxBenchBpbError``.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept, SteeringMode
from steering.eval.axbench_bpb.intervention import (
    _build_ours_intervention,
    build_intervention,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(concept_id: int = 1) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=0,
        layer=8,
        sae_source="dummy",
        description="test concept",
        corpus_count=10,
        description_source="manual",
    )


def _write_bag(path: Path, bag: dict[int, int]) -> None:
    """Write a minimal safetensors bag in the format expected by read_ours_gcg_bag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_fids = sorted(bag.keys())
    tensors = {
        "feature_ids": t.tensor(sorted_fids, dtype=t.int64),
        "buckets": t.tensor([bag[k] for k in sorted_fids], dtype=t.int64),
    }
    save_file(tensors, str(path))


def _build(
    method_name: str,
    concept: Concept,
    coefficient: float,
    adv_paths: AxBenchBpbAdvPaths | None,
    *,
    source_run_dir: Path,
    core_prefix_id: str | None = None,
) -> Intervention:
    """Thin wrapper around build_intervention for GCG method tests.

    ``model`` stands in for a LEGACY (uncompacted) loaded checkpoint --
    ``feature_vocab=None`` -- matching every bag ``_write_bag`` writes in this
    file (no ``vocab_fingerprint`` metadata at all, the pre-#695 shape). Per the
    checkpoint-V back-compat rule, an absent stamp read against a legacy
    checkpoint loads unchanged, so this keeps every test here byte-identical to
    pre-checkpoint-V-stamping behavior while still exercising the REQUIRED
    ``model`` argument (a bag-backed candidate raises without it -- see
    ``_build_ours_intervention``).
    """
    return build_intervention(
        method=get_method_by_name(method_name),
        concept=concept,
        coefficient=coefficient,
        source_paths=SourcePaths(source_run_dir=source_run_dir),
        diffmean_layer=8,
        target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        adv_paths=adv_paths,
        core_prefix_id=core_prefix_id,
        model=SimpleNamespace(feature_vocab=None),
        apply_max_act_scale=False,
    )


# ---------------------------------------------------------------------------
# (a) Fair positive GCG — written bag is loaded as feature_conditions
# ---------------------------------------------------------------------------


def test_fair_gcg_encdec_loads_bag(tmp_path: Path) -> None:
    """A written fair bag {42: 0} is returned as Intervention.ours({42: 0})."""
    concept = _make_concept(concept_id=1)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    bag_path = paths.ours_gcg_bag(1, "positive", 0.0, "fair")
    _write_bag(bag_path, {42: 0})

    result = _build("ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path)

    assert result == Intervention.ours(feature_conditions={42: 0})


def test_fair_gcg_bag_with_multiple_features(tmp_path: Path) -> None:
    """Multi-feature bag is loaded faithfully — no features dropped."""
    concept = _make_concept(concept_id=1)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    bag_path = paths.ours_gcg_bag(1, "positive", 0.5, "fair")
    _write_bag(bag_path, {10: 0, 20: 0, 30: 0})

    result = _build("ours_gcg_encdec", concept, 0.5, paths, source_run_dir=tmp_path)

    assert result == Intervention.ours(feature_conditions={10: 0, 20: 0, 30: 0})


def test_feature_diffmean_encdec_loads_fixed_universal_bag(tmp_path: Path) -> None:
    """Feature-DiffMean ignores the public coefficient and loads lambda_0.0."""
    concept = _make_concept(concept_id=1)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    bag_path = paths.ours_gcg_bag(1, "positive", 0.0, "feature_diffmean")
    _write_bag(bag_path, {12: 0, 34: 1})

    result = _build(
        "feature_diffmean_encdec",
        concept,
        99.0,
        paths,
        source_run_dir=tmp_path,
    )

    assert result == Intervention.ours(feature_conditions={12: 0, 34: 1})


# ---------------------------------------------------------------------------
# (b) Negative-mode method loads the negative-mode bag, not the positive one
# ---------------------------------------------------------------------------


def test_neg_gcg_encdec_loads_negative_mode_bag(tmp_path: Path) -> None:
    """neg_ours_gcg_encdec loads the ``negative`` mode bag, ignoring positive."""
    concept = _make_concept(concept_id=2)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    # Both mode bags exist; only the negative-mode one should be loaded.
    pos_path = paths.ours_gcg_bag(2, "positive", 0.5, "fair")
    _write_bag(pos_path, {10: 0})

    neg_path = paths.ours_gcg_bag(2, "negative", 0.5, "fair")
    _write_bag(neg_path, {20: 1})

    result = _build("neg_ours_gcg_encdec", concept, 0.5, paths, source_run_dir=tmp_path)

    assert result == Intervention.ours(feature_conditions={20: 1})
    # Extra: confirm the positive bag was NOT loaded.
    assert result != Intervention.ours(feature_conditions={10: 0})


# ---------------------------------------------------------------------------
# (c) Roofline loads the per-prefix-specific bag
# ---------------------------------------------------------------------------


def test_roofline_gcg_loads_per_prefix_bag(tmp_path: Path) -> None:
    """Each prefix_id maps to its own roofline bag — the two don't cross-load."""
    concept = _make_concept(concept_id=3)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    path_a = paths.ours_gcg_bag(3, "positive", 0.0, "roofline", prefix_id="pfx_a")
    path_b = paths.ours_gcg_bag(3, "positive", 0.0, "roofline", prefix_id="pfx_b")
    _write_bag(path_a, {100: 0})
    _write_bag(path_b, {200: 0})

    result_a = _build(
        "ours_gcg_roofline_encdec",
        concept,
        0.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_a",
    )
    result_b = _build(
        "ours_gcg_roofline_encdec",
        concept,
        0.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_b",
    )

    assert result_a == Intervention.ours(feature_conditions={100: 0})
    assert result_b == Intervention.ours(feature_conditions={200: 0})


def test_roofline_feature_diffmean_loads_per_prefix_bag(tmp_path: Path) -> None:
    """Roofline Feature-DiffMean is per-prefix and fixed at lambda_0.0."""
    concept = _make_concept(concept_id=3)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    path_a = paths.ours_gcg_bag(
        3,
        "positive",
        0.0,
        "feature_diffmean_roofline",
        prefix_id="pfx_a",
    )
    path_b = paths.ours_gcg_bag(
        3,
        "positive",
        0.0,
        "feature_diffmean_roofline",
        prefix_id="pfx_b",
    )
    _write_bag(path_a, {101: 0})
    _write_bag(path_b, {202: 0})

    result_a = _build(
        "feature_diffmean_roofline_encdec",
        concept,
        99.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_a",
    )
    result_b = _build(
        "feature_diffmean_roofline_encdec",
        concept,
        99.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_b",
    )

    assert result_a == Intervention.ours(feature_conditions={101: 0})
    assert result_b == Intervention.ours(feature_conditions={202: 0})


def test_neg_roofline_gcg_loads_negative_per_prefix_bag(tmp_path: Path) -> None:
    """neg_ours_gcg_roofline_encdec loads the negative-mode roofline bag."""
    concept = _make_concept(concept_id=4)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    path = paths.ours_gcg_bag(4, "negative", 1.0, "roofline", prefix_id="pfx_neg")
    _write_bag(path, {55: 1})

    result = _build(
        "neg_ours_gcg_roofline_encdec",
        concept,
        1.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_neg",
    )

    assert result == Intervention.ours(feature_conditions={55: 1})


def test_neg_roofline_feature_diffmean_loads_negative_per_prefix_bag(
    tmp_path: Path,
) -> None:
    """Negative roofline Feature-DiffMean loads the negative per-prefix bag."""
    concept = _make_concept(concept_id=4)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    path = paths.ours_gcg_bag(
        4,
        "negative",
        0.0,
        "feature_diffmean_roofline",
        prefix_id="pfx_neg",
    )
    _write_bag(path, {56: 1})

    result = _build(
        "neg_feature_diffmean_roofline_encdec",
        concept,
        1.0,
        paths,
        source_run_dir=tmp_path,
        core_prefix_id="pfx_neg",
    )

    assert result == Intervention.ours(feature_conditions={56: 1})


# ---------------------------------------------------------------------------
# (d) Missing bag → raises AxBenchBpbError (Task-10: the guard guarantees only
#     prefixes-with-bags reach here, so a missing bag is a genuine pipeline bug)
# ---------------------------------------------------------------------------


def test_missing_fair_bag_raises(tmp_path: Path) -> None:
    """A missing fair bag RAISES AxBenchBpbError (no warn+empty degradation)."""
    concept = _make_concept(concept_id=5)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    # Deliberately write NO bag for concept_id=5.

    with pytest.raises(AxBenchBpbError, match="precomputed bag not found"):
        _build("ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path)


def test_missing_roofline_bag_raises(tmp_path: Path) -> None:
    """A missing roofline bag RAISES AxBenchBpbError (no warn+empty degradation)."""
    concept = _make_concept(concept_id=6)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")

    with pytest.raises(AxBenchBpbError, match="precomputed bag not found"):
        _build(
            "ours_gcg_roofline_encdec",
            concept,
            0.0,
            paths,
            source_run_dir=tmp_path,
            core_prefix_id="absent_prefix",
        )


# ---------------------------------------------------------------------------
# (e) Corrupt artifact → raises AxBenchBpbError
# ---------------------------------------------------------------------------


def test_corrupt_bag_raises_axbench_bpb_error(tmp_path: Path) -> None:
    """A corrupt bag file raises AxBenchBpbError (not a raw safetensors exception)."""
    concept = _make_concept(concept_id=7)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    bag_path = paths.ours_gcg_bag(7, "positive", 0.0, "fair")
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    bag_path.write_bytes(b"this is not a valid safetensors file")

    with pytest.raises(AxBenchBpbError, match="corrupt"):
        _build("ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path)


def test_corrupt_roofline_bag_raises_axbench_bpb_error(tmp_path: Path) -> None:
    """Corrupt roofline bag also raises AxBenchBpbError."""
    concept = _make_concept(concept_id=8)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    bag_path = paths.ours_gcg_bag(8, "positive", 0.0, "roofline", prefix_id="corrupt_pfx")
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    bag_path.write_bytes(b"\x00\x01\x02\x03 garbage")

    with pytest.raises(AxBenchBpbError, match="corrupt"):
        _build(
            "ours_gcg_roofline_encdec",
            concept,
            0.0,
            paths,
            source_run_dir=tmp_path,
            core_prefix_id="corrupt_pfx",
        )


# ---------------------------------------------------------------------------
# Extra: adv_paths=None for a GCG method raises AxBenchBpbError
# ---------------------------------------------------------------------------


def test_gcg_without_adv_paths_raises(tmp_path: Path) -> None:
    """build_intervention for a GCG method without adv_paths raises AxBenchBpbError."""
    concept = _make_concept(concept_id=9)
    with pytest.raises(AxBenchBpbError, match="adv_paths"):
        _build("ours_gcg_encdec", concept, 0.0, None, source_run_dir=tmp_path)


# ---------------------------------------------------------------------------
# Regression: plain-string partition_modifier_name must still dispatch
# (max-review finding #2 — the enum rewrite's `.value` crashed on the string
# form that some Method-like objects / fixtures carry).
# ---------------------------------------------------------------------------


def test_gcg_dispatch_tolerates_plain_string_modifier_name(tmp_path: Path) -> None:
    """A gcg method whose partition_modifier_name is the plain STRING dispatches
    to the bag fast-path (no AttributeError on `.value`).

    The frozenset membership check this replaced accepted both the
    PartitionModifier member and its string value (StrEnum hash equality); the
    classification must preserve that tolerance. A gcg-named string reaching
    ``_build_ours_intervention`` with `.value` would crash instead of loading
    the bag.
    """
    concept = _make_concept(concept_id=3)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="strmod")
    _write_bag(paths.ours_gcg_bag(3, "positive", 0.0, "fair"), {7: 0})

    method = SimpleNamespace(
        name="ours_gcg_encdec",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name="ours_gcg_encdec",  # the PLAIN STRING form
    )
    result = _build_ours_intervention(
        method=method,
        concept=concept,
        coefficient=0.0,
        source_paths=SourcePaths(source_run_dir=tmp_path),
        target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
        inactive_sampler=None,
        rng=random.Random(0),
        adv_paths=paths,
        model=SimpleNamespace(feature_vocab=None),
    )
    assert result == Intervention.ours(feature_conditions={7: 0})


def test_gcg_dispatch_plain_string_missing_adv_paths_raises(tmp_path: Path) -> None:
    """The plain-string gcg form still reaches the adv_paths guard (dispatched as
    gcg, not crashed): without adv_paths it raises AxBenchBpbError, proving the
    fast-path was entered rather than skipped-by-exception."""
    method = SimpleNamespace(
        name="ours_gcg_roofline_encdec",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name="ours_gcg_roofline_encdec",
    )
    with pytest.raises(AxBenchBpbError, match="adv_paths"):
        _build_ours_intervention(
            method=method,
            concept=_make_concept(concept_id=4),
            coefficient=0.0,
            source_paths=SourcePaths(source_run_dir=tmp_path),
            target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
            inactive_sampler=None,
            rng=random.Random(0),
            adv_paths=None,
            core_prefix_id="p0",
            model=SimpleNamespace(feature_vocab=None),
        )
