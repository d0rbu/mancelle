"""Tests for ``steering.eval.clustered_cache_seed_contract`` (PR #739 Track D,
D-contract, Option B: seed-only checkpoint resolution).

``assert_clustered_seed_matches_checkpoint(cfg)`` resolves the target
checkpoint's real training-time ``seed`` DIRECTLY via ``resolve_checkpoint_
pin`` (the same TOCTOU-safe hash-and-load Track B uses) and compares it
against ``cfg.seed`` -- WITHOUT running Track B's full ``resolve_cluster_
feature_contract`` mask/cluster/SAE/statistics/selection agreement. That
narrowing is the whole point: the full agreement's ``_resolve_selection_
identity`` hard-rejects any checkpoint carrying ``delta_weights``/
``feature_delta``/``feature_prediction``, which describes the one real
clustered production checkpoint (splice-CF/feature-prediction) this module
exists to guard. See the module docstring for the full rationale.

Two tiers, both fast-lane (``tmp_path`` only, no network, no GPU, no
``@slow``/``@requires_feature_data`` marker):

* Tier 1 -- fully synthetic unit tests with ``resolve_checkpoint_pin`` and
  ``resolve_eval_checkpoint_directory`` (the two primitives this module
  actually calls) monkeypatched to call-counting stubs, proving the opt-in
  skip and the seed-format guard never reach a resolve, and that the stub's
  own seed/pin fields flow through to the returned ``CheckpointSeedProvenance``
  unchanged.
* Tier 2 -- real-fixture tests through the genuine, unmocked
  ``resolve_checkpoint_pin``, using a MINIMAL checkpoint-dir writer (a
  loadable ``model_{step:06d}.pt`` vocab state-dict + a
  ``meta_{step:06d}.json`` carrying ``user_config.seed``) -- no statistics,
  packed rows, selection shards, SAE, or manifests, because this narrow
  seed-only check never reads any of that (unlike Track B's full contract,
  which the old, now-superseded, full-preflight fixture this replaces had to
  satisfy). The CENTERPIECE here proves a delta-aware (splice-CF/
  feature-prediction) checkpoint -- the exact shape the OLD design rejected
  -- is now ACCEPTED, with a companion test proving that divergence is real
  by driving the identical fixture through the OLD design's own selection-
  identity resolver and confirming THAT one still raises.

The binding, build-entrypoint-level centerpiece (proving the check fires
strictly before ``glob_features``) lives in ``test_build_eval_context_cache.py``
since it needs the full ``build_cache_from_config`` cfg shape that module
already builds. The ``clustered_context_cache_metadata`` byte-identity/
collision tests live in ``test_context_cache_io.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import steering.eval.clustered_cache_seed_contract as seed_contract_module
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
)
from steering.eval.cluster_checkpoint_pin import _resolve_selection_identity
from steering.eval.cluster_contract_identity import ClusterFeatureContractError
from steering.eval.clustered_cache_seed_contract import (
    CheckpointSeedProvenance,
    assert_clustered_seed_matches_checkpoint,
)

# ============================================================================
# Tier 1: fully synthetic, resolve_checkpoint_pin + resolve_eval_checkpoint_
# directory monkeypatched to call-counting stubs. No real filesystem fixture
# -- proves the opt-in-skip / malformed-seed-raise branches never reach a
# resolve, and that a stub's returned pin/metadata flow through unchanged.
# ============================================================================


class _CountingPin:
    """Stand-in for ``resolve_checkpoint_pin``: counts calls and always
    returns the canned ``(fake_pin, fake_metadata, {})`` triple built from
    exactly the fields ``assert_clustered_seed_matches_checkpoint`` reads off
    a real pin (``resolved_tag``/``step``/``model_file.sha256``/
    ``metadata_file.sha256``) plus the checkpoint's real training-time
    ``user_config.seed``.
    """

    def __init__(
        self,
        *,
        checkpoint_seed: Any = 99,
        metadata: dict[str, Any] | None = None,
        resolved_tag: str = "stub-tag",
        step: int = 7,
        model_sha256: str = "a" * 64,
        metadata_sha256: str = "b" * 64,
    ) -> None:
        self.calls = 0
        self.fake_pin = SimpleNamespace(
            resolved_tag=resolved_tag,
            step=step,
            model_file=SimpleNamespace(sha256=model_sha256),
            metadata_file=SimpleNamespace(sha256=metadata_sha256),
        )
        self.fake_metadata: dict[str, Any] = (
            {"user_config": {"seed": checkpoint_seed}} if metadata is None else metadata
        )

    def __call__(self, checkpoint_dir: Any, *, step: Any, resolved_tag: Any) -> Any:
        self.calls += 1
        return self.fake_pin, self.fake_metadata, {}


class _CountingDirectoryResolver:
    """Stand-in for ``resolve_eval_checkpoint_directory``: counts calls and
    always returns a fixed fake ``(directory, resolved_tag)`` pair, so Tier 1
    tests never touch the real filesystem."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs: Any) -> tuple[Path, str]:
        self.calls += 1
        return Path("/fake/ckpt"), "faketag"


def _patch_resolvers(
    monkeypatch,
    *,
    checkpoint_seed: Any = 99,
    metadata: dict[str, Any] | None = None,
    resolved_tag: str = "stub-tag",
    step: int = 7,
    model_sha256: str = "a" * 64,
    metadata_sha256: str = "b" * 64,
) -> tuple[_CountingPin, _CountingDirectoryResolver]:
    pin_stub = _CountingPin(
        checkpoint_seed=checkpoint_seed,
        metadata=metadata,
        resolved_tag=resolved_tag,
        step=step,
        model_sha256=model_sha256,
        metadata_sha256=metadata_sha256,
    )
    dir_stub = _CountingDirectoryResolver()
    monkeypatch.setattr(seed_contract_module, "resolve_checkpoint_pin", pin_stub)
    monkeypatch.setattr(seed_contract_module, "resolve_eval_checkpoint_directory", dir_stub)
    return pin_stub, dir_stub


def test_no_checkpoint_ref_skips_resolve_and_returns_none(monkeypatch):
    pin_stub, dir_stub = _patch_resolvers(monkeypatch)

    cfg = OmegaConf.create({"seed": 42})
    result = assert_clustered_seed_matches_checkpoint(cfg)

    assert result is None
    assert pin_stub.calls == 0
    assert dir_stub.calls == 0


@pytest.mark.parametrize("bad_seed", [None, True, -1, 3.0, "42"])
def test_malformed_seed_raises_before_any_resolve(monkeypatch, bad_seed):
    pin_stub, dir_stub = _patch_resolvers(monkeypatch)

    cfg = OmegaConf.create({"checkpoint_dir": "/fake/checkpoint", "seed": bad_seed})
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)

    assert pin_stub.calls == 0
    # The seed-format guard is the cheapest check and fires BEFORE even
    # resolve_eval_checkpoint_directory (see the production module's own
    # "Cheapest check first" comment) -- kept stubbed defensively (per the
    # spec) but its call count must also be 0 here, proving the ordering.
    assert dir_stub.calls == 0


def test_checkpoint_source_alone_also_opts_in(monkeypatch):
    """``checkpoint_source`` (not just ``checkpoint_dir``) also triggers the
    cross-check -- both are independently sufficient per
    ``resolve_eval_checkpoint_directory``'s XOR contract."""

    pin_stub, dir_stub = _patch_resolvers(monkeypatch)

    cfg = OmegaConf.create({"checkpoint_source": "clustered", "seed": None})
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)
    assert pin_stub.calls == 0
    assert dir_stub.calls == 0


@pytest.mark.parametrize(
    "lone_field",
    ["checkpoint_model_tag", "checkpoint_step", "checkpoint_tag"],
)
def test_partial_checkpoint_ref_opts_in(monkeypatch, lone_field):
    """A PARTIAL checkpoint reference -- a lone ``checkpoint_model_tag``,
    ``checkpoint_step``, or ``checkpoint_tag`` with no ``checkpoint_dir``/
    ``checkpoint_source`` -- must NOT be silently treated as "no reference"
    (which would skip the cross-check entirely, bypassing the exact guarantee
    this module provides). It opts in exactly like ``checkpoint_source``
    alone: proven here via the malformed-``seed`` guard firing
    (``pin_stub.calls == 0``), i.e. it got PAST the opt-in gate. The
    downstream real-resolver fail-loud is proven separately below.

    ``checkpoint_step`` uses ``0`` -- an explicit selector for which
    ``is None`` is False -- to pin that the gate keys on presence, not
    falsiness."""

    pin_stub, dir_stub = _patch_resolvers(monkeypatch)

    lone_value: Any = 0 if lone_field == "checkpoint_step" else "some-runtime-ref"
    cfg = OmegaConf.create({lone_field: lone_value, "seed": None})
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)
    assert pin_stub.calls == 0
    assert dir_stub.calls == 0


def test_partial_checkpoint_ref_fails_loud_through_real_resolver():
    """End-to-end proof the partial-reference opt-in is load-bearing, not
    cosmetic: with a VALID ``seed`` (so the cheap seed-format guard passes and
    cannot mask the outcome), a lone ``checkpoint_model_tag`` drives the
    genuine, unmocked ``resolve_eval_checkpoint_directory`` to fail loud on
    the half-configured runtime reference (``checkpoint_source`` missing)
    BEFORE any checkpoint I/O. This is the exact hole the gate closes: a
    partial reference can never silently skip the cross-check.

    No monkeypatch -- the real resolver runs, and there are no mask/cluster/
    stats cfg fields to fake either (the new API takes no such kwargs): the
    XOR gate raises before touching the filesystem at all."""

    cfg = OmegaConf.create({"checkpoint_model_tag": "some-runtime-ref", "seed": 42})
    with pytest.raises(ClusterFeatureContractError) as exc_info:
        assert_clustered_seed_matches_checkpoint(cfg)
    assert "checkpoint_source" in str(exc_info.value)


def test_stub_seed_mismatch_raises_naming_both_seeds(monkeypatch):
    pin_stub, dir_stub = _patch_resolvers(monkeypatch, checkpoint_seed=99)

    cfg = OmegaConf.create({"checkpoint_dir": "/fake/checkpoint", "seed": 42})
    with pytest.raises(ClusterFeatureContractError) as exc_info:
        assert_clustered_seed_matches_checkpoint(cfg)

    message = str(exc_info.value)
    assert "42" in message
    assert "99" in message
    assert pin_stub.calls == 1
    assert dir_stub.calls == 1


def test_stub_seed_match_returns_provenance(monkeypatch):
    pin_stub, dir_stub = _patch_resolvers(
        monkeypatch,
        checkpoint_seed=99,
        resolved_tag="stub-tag",
        step=7,
        model_sha256="a" * 64,
        metadata_sha256="b" * 64,
    )

    cfg = OmegaConf.create({"checkpoint_dir": "/fake/checkpoint", "seed": 99})
    result = assert_clustered_seed_matches_checkpoint(cfg)

    assert result is not None
    # Tied to the STUB's own pin fields (not re-hardcoded) so a production
    # bug that swaps/drops a field cannot pass this test by accident.
    pin = pin_stub.fake_pin
    assert result.resolved_tag == pin.resolved_tag
    assert result.step == pin.step
    assert result.model_sha256 == pin.model_file.sha256
    assert result.metadata_sha256 == pin.metadata_file.sha256
    assert result.preselection_seed == 99
    assert pin_stub.calls == 1


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {"user_config": {}},
        {"user_config": {"seed": "42"}},
    ],
    ids=["seed_key_absent", "seed_wrong_type"],
)
def test_checkpoint_missing_user_config_seed_raises(monkeypatch, bad_metadata):
    pin_stub, dir_stub = _patch_resolvers(monkeypatch, metadata=bad_metadata)

    cfg = OmegaConf.create({"checkpoint_dir": "/fake/checkpoint", "seed": 42})
    with pytest.raises(ClusterFeatureContractError, match="preselection seed"):
        assert_clustered_seed_matches_checkpoint(cfg)
    assert pin_stub.calls == 1


# ============================================================================
# Tier 2 fixture: a MINIMAL checkpoint-dir writer -- only what
# resolve_checkpoint_pin actually needs (a loadable model_{step:06d}.pt vocab
# state-dict + a meta_{step:06d}.json with user_config.seed). This narrow
# seed-only check reads NEITHER statistics, packed rows, selection shards,
# NOR an SAE -- unlike Track B's full resolve_cluster_feature_contract, which
# the old (pre-Option-B) version of this fixture had to fully satisfy.
# ============================================================================


def _write_checkpoint(
    tmp_path: Path, *, seed: int, step: int = 7, delta_aware: bool = False
) -> Path:
    """Write the minimal checkpoint dir ``resolve_checkpoint_pin`` needs.

    ``delta_aware=True`` writes the splice-counterfactual/feature-prediction
    ``user_config`` shape (``delta_weights``, ``feature_delta``,
    ``feature_prediction`` all set) -- the one real clustered production
    checkpoint's shape (``configs/training/cluster_fp/
    conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml``), which the OLD
    full contract's ``_resolve_selection_identity`` unconditionally rejects
    (see ``test_delta_aware_checkpoint_would_fail_full_contract`` below) but
    this module's narrow seed-only check does not even look at.
    """

    checkpoint_dir = tmp_path / f"ckpt_seed{seed}_{int(delta_aware)}"
    checkpoint_dir.mkdir()
    n_sae_features = 16_384
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(6)))
    definition = FeatureClusterDefinition(clusters=((0, 1), (2, 3)))
    cluster_path = checkpoint_dir / "feature_clusters.json"
    cluster_path.write_text(json.dumps(definition.canonical_dict(), sort_keys=True), encoding="utf-8")
    resolved_clusters = ResolvedFeatureClusters(
        source_path=cluster_path.resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=n_sae_features),
    )
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert feature_vocab is not None
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((feature_vocab.size, 2)),
            VOCAB_IDS_KEY: t.tensor(feature_vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
        },
        checkpoint_dir / f"model_{step:06d}.pt",
    )
    feature_selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    user_config = {
        "feature_selection": feature_selection,
        "negative_selection": {"distribution": "geometric", "mean": 1.0, "std": 1.0},
        "num_buckets": 2,
        "seed": seed,
        "feature_delta": None,
        "feature_prediction": None,
    }
    if delta_aware:  # splice-CF / feature-prediction shape (the whole point of Option B)
        user_config["feature_selection"] = {**feature_selection, "delta_weights": "percentile_delta"}
        user_config["feature_delta"] = {
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        }
        user_config["feature_prediction"] = {
            "block": "mlp",
            "head_layers": 0,
            "weight": 0.1,
            "layer": -1,
            "loss_mode": "max_pool",
            "loss_fn": "bce",
        }
    metadata = {
        "model_config": {"n_layer": 2, "n_embd": 8},
        "encoder_config": {"n_embd": 8, "depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": mask.canonical_dict(),
        "feature_vocab": feature_vocab.metadata_dict(),
        "feature_clusters_descriptor": resolved_clusters.descriptor_dict(),
        "feature_clusters_canonical": resolved_clusters.canonical_dict(),
        "user_config": user_config,
    }
    (checkpoint_dir / f"meta_{step:06d}.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return checkpoint_dir


def _assert_sha256_hex(value: str) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    int(value, 16)  # raises ValueError if not valid hex


def _cfg_for_checkpoint(checkpoint_dir: Path, *, seed: Any, step: int = 7) -> DictConfig:
    return OmegaConf.create(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": step,
            "seed": seed,
        }
    )


# ============================================================================
# Tier 2: real fixtures, real (unmocked) resolve_checkpoint_pin.
# ============================================================================


def test_real_checkpoint_seed_match_returns_provenance(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path, seed=42)
    cfg = _cfg_for_checkpoint(checkpoint_dir, seed=42)

    result = assert_clustered_seed_matches_checkpoint(cfg)

    assert result is not None
    assert result.preselection_seed == 42
    assert result.step == 7
    _assert_sha256_hex(result.model_sha256)
    _assert_sha256_hex(result.metadata_sha256)


def test_real_checkpoint_seed_mismatch_raises_naming_both_seeds(tmp_path):
    checkpoint_dir = _write_checkpoint(tmp_path, seed=99)
    cfg = _cfg_for_checkpoint(checkpoint_dir, seed=42)

    with pytest.raises(ClusterFeatureContractError) as exc_info:
        assert_clustered_seed_matches_checkpoint(cfg)

    message = str(exc_info.value)
    assert "42" in message
    assert "99" in message


def test_real_delta_aware_checkpoint_seed_match_is_accepted(tmp_path):
    """CENTERPIECE: a delta-aware (splice-counterfactual + feature-
    prediction) checkpoint whose configured seed matches is ACCEPTED --
    returns a provenance, does NOT raise.

    This is the exact case the OLD design (Track B's full
    ``resolve_cluster_feature_contract``, via ``_resolve_selection_
    identity``) REJECTED outright: that function raises unconditionally
    whenever ``positive.delta_weights is not None`` or
    ``user_config["feature_delta"]``/``["feature_prediction"]`` is set --
    see ``test_delta_aware_checkpoint_would_fail_full_contract`` below, which
    proves that rejection against this EXACT fixture. Option B fixes this
    because the one real clustered production checkpoint
    (``configs/training/cluster_fp/
    conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml``) is delta-aware --
    a seed cross-check that could never accept it would be useless for the
    one checkpoint it exists to guard.
    """

    checkpoint_dir = _write_checkpoint(tmp_path, seed=42, delta_aware=True)
    cfg = _cfg_for_checkpoint(checkpoint_dir, seed=42)

    result = assert_clustered_seed_matches_checkpoint(cfg)

    assert result is not None
    assert result.preselection_seed == 42


def test_delta_aware_checkpoint_would_fail_full_contract(tmp_path):
    """Companion to the centerpiece above: proves the redesign's divergence
    is REAL, not vacuous. Feeds the exact SAME delta-aware checkpoint
    metadata the centerpiece proves ``assert_clustered_seed_matches_
    checkpoint`` now ACCEPTS through the OLD full-contract's own selection-
    identity resolver (``_resolve_selection_identity``, still the innards of
    Track B's ``resolve_cluster_feature_contract`` on current main) and
    confirms THAT one still raises.

    Without this test, the centerpiece could pass even if
    ``assert_clustered_seed_matches_checkpoint`` were accidentally wired to
    still run the full agreement (a regression that would silently
    reintroduce the exact bug Option B fixes) -- there would be no evidence
    the two designs actually disagree on this fixture. This test is that
    evidence: same metadata, same mask, same clusters, OLD resolver, still
    rejected.
    """

    checkpoint_dir = _write_checkpoint(tmp_path, seed=42, delta_aware=True)
    metadata = json.loads((checkpoint_dir / "meta_000007.json").read_text())
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(6)))
    definition = FeatureClusterDefinition(clusters=((0, 1), (2, 3)))
    resolved_clusters = ResolvedFeatureClusters(
        source_path=(checkpoint_dir / "feature_clusters.json").resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=16_384),
    )

    with pytest.raises(ClusterFeatureContractError, match="feature_prediction"):
        _resolve_selection_identity(
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=resolved_clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_real_checkpoint_mask_shape_is_irrelevant(tmp_path):
    """The module never parses ``feature_mask`` at all (see the module
    docstring's "imposes NO constraint on the build's ``feature_mask`` shape"
    claim) -- proven here by driving the SAME real checkpoint through three
    wildly different ``feature_mask`` cfg shapes (null, inline whitelist,
    and a path form pointing at a file that does not even exist on disk) and
    asserting the returned provenance is identical every time. If the
    production code ever started reading ``cfg.feature_mask`` (e.g. to
    resolve/validate it), the nonexistent-path variant would raise instead
    of returning -- this test would catch that regression immediately.
    """

    checkpoint_dir = _write_checkpoint(tmp_path, seed=42)
    base = {"checkpoint_dir": str(checkpoint_dir), "checkpoint_step": 7, "seed": 42}

    result_null = assert_clustered_seed_matches_checkpoint(
        OmegaConf.create({**base, "feature_mask": None})
    )
    result_inline = assert_clustered_seed_matches_checkpoint(
        OmegaConf.create({**base, "feature_mask": {"mode": "whitelist", "ids": [0, 1]}})
    )
    result_path = assert_clustered_seed_matches_checkpoint(
        OmegaConf.create(
            {**base, "feature_mask": {"mode": "whitelist", "path": "/does/not/exist.txt"}}
        )
    )

    assert result_null is not None
    assert result_null == result_inline == result_path
    assert isinstance(result_null, CheckpointSeedProvenance)


# ============================================================================
# Fail-loud TYPE contract: the module documents raising ClusterFeatureContract
# Error when the checkpoint "cannot be resolved/loaded". The strict path
# resolution inside resolve_eval_checkpoint_directory / resolve_checkpoint_pin
# (missing dir, empty dir, nonexistent explicit step) raises a bare OSError/
# FileNotFoundError; the module normalizes those so callers see the one
# promised type. (The first two scenarios genuinely leak FileNotFoundError
# absent the normalization, so these are non-vacuous.)
# ============================================================================


def test_missing_checkpoint_dir_raises_cluster_feature_contract_error(tmp_path):
    """A missing ``checkpoint_dir`` surfaces as ``ClusterFeatureContractError``
    (this module's documented failure type), not the bare ``FileNotFoundError``
    ``resolve_eval_checkpoint_directory``'s ``Path.resolve(strict=True)`` raises
    unwrapped."""

    cfg = _cfg_for_checkpoint(tmp_path / "does_not_exist", seed=42)
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)


def test_empty_checkpoint_dir_no_step_raises_cluster_feature_contract_error(tmp_path):
    """An existing but EMPTY checkpoint dir with no explicit step drives
    ``find_last_step`` to raise ``FileNotFoundError('No checkpoints found')``;
    it must surface as ``ClusterFeatureContractError``, not leak."""

    empty_dir = tmp_path / "empty_ckpt"
    empty_dir.mkdir()
    cfg = OmegaConf.create({"checkpoint_dir": str(empty_dir), "seed": 42})
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)


def test_nonexistent_explicit_step_raises_cluster_feature_contract_error(tmp_path):
    """A real checkpoint dir + an explicit ``checkpoint_step`` whose
    ``model_``/``meta_`` files do not exist must surface as
    ``ClusterFeatureContractError``, not a bare ``FileNotFoundError`` from the
    checkpoint's strict file resolution."""

    checkpoint_dir = _write_checkpoint(tmp_path, seed=42, step=7)
    cfg = _cfg_for_checkpoint(checkpoint_dir, seed=42, step=999)
    with pytest.raises(ClusterFeatureContractError):
        assert_clustered_seed_matches_checkpoint(cfg)
